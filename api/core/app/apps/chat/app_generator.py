import logging
import threading
import uuid
from collections.abc import Generator, Mapping
from typing import Any, Literal, Union, overload

from flask import Flask, current_app
from pydantic import ValidationError

from configs import dify_config
from constants import UUID_NIL
from core.app.app_config.easy_ui_based_app.model_config.converter import ModelConfigConverter
from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.app.apps.base_app_queue_manager import AppQueueManager, GenerateTaskStoppedError, PublishFrom
from core.app.apps.chat.app_config_manager import ChatAppConfigManager
from core.app.apps.chat.app_runner import ChatAppRunner
from core.app.apps.chat.generate_response_converter import ChatAppGenerateResponseConverter
from core.app.apps.message_based_app_generator import MessageBasedAppGenerator
from core.app.apps.message_based_app_queue_manager import MessageBasedAppQueueManager
from core.app.entities.app_invoke_entities import ChatAppGenerateEntity, InvokeFrom
from core.model_runtime.errors.invoke import InvokeAuthorizationError
from core.ops.ops_trace_manager import TraceQueueManager
from extensions.ext_database import db
from factories import file_factory
from models.account import Account
from models.model import App, EndUser
from services.conversation_service import ConversationService
from services.errors.message import MessageNotExistsError

logger = logging.getLogger(__name__)


class ChatAppGenerator(MessageBasedAppGenerator):
    @overload
    def generate(
        self,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[True],
    ) -> Generator[Mapping | str, None, None]: ...

    @overload
    def generate(
        self,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: Literal[False],
    ) -> Mapping[str, Any]: ...

    @overload
    def generate(
        self,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool,
    ) -> Union[Mapping[str, Any], Generator[Mapping[str, Any] | str, None, None]]: ...

    def generate(
        self,
        app_model: App,
        user: Union[Account, EndUser],
        args: Mapping[str, Any],
        invoke_from: InvokeFrom,
        streaming: bool = True,
    ) -> Union[Mapping[str, Any], Generator[Mapping[str, Any] | str, None, None]]:
        """
        Generate App response.

        :param app_model: App
        :param user: account or end user
        :param args: request args
        :param invoke_from: invoke from source
        :param streaming: is stream
        """
        if not args.get("query"):
            raise ValueError("query is required")

        query = args["query"]
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        query = query.replace("\x00", "")
        inputs = args["inputs"]

        extras = {"auto_generate_conversation_name": args.get("auto_generate_name", True)}

        # get conversation
        conversation = None
        conversation_id = args.get("conversation_id")
        if conversation_id:
            conversation = ConversationService.get_conversation(
                app_model=app_model, conversation_id=conversation_id, user=user
            )
        # get app model config
        app_model_config = self._get_app_model_config(app_model=app_model, conversation=conversation)

        # validate override model config
        override_model_config_dict = None
        if args.get("model_config"):
            if invoke_from != InvokeFrom.DEBUGGER:
                raise ValueError("Only in App debug mode can override model config")

            # validate config
            override_model_config_dict = ChatAppConfigManager.config_validate(
                tenant_id=app_model.tenant_id, config=args.get("model_config", {})
            )

            # always enable retriever resource in debugger mode
            override_model_config_dict["retriever_resource"] = {"enabled": True}

        # parse files
        files = args["files"] if args.get("files") else []
        file_extra_config = FileUploadConfigManager.convert(override_model_config_dict or app_model_config.to_dict())
        if file_extra_config:
            file_objs = file_factory.build_from_mappings(
                mappings=files,
                tenant_id=app_model.tenant_id,
                config=file_extra_config,
            )
        else:
            file_objs = []

        # convert to app config
        app_config = ChatAppConfigManager.get_app_config(
            app_model=app_model,
            app_model_config=app_model_config,
            conversation=conversation,
            override_config_dict=override_model_config_dict,
        )

        # get tracing instance
        trace_manager = TraceQueueManager(app_id=app_model.id)

        # init application generate entity
        application_generate_entity = ChatAppGenerateEntity(
            task_id=str(uuid.uuid4()),
            app_config=app_config,
            model_conf=ModelConfigConverter.convert(app_config),
            file_upload_config=file_extra_config,
            conversation_id=conversation.id if conversation else None,
            inputs=self._prepare_user_inputs(
                user_inputs=inputs, variables=app_config.variables, tenant_id=app_model.tenant_id
            ),
            query=query,
            files=list(file_objs),
            parent_message_id=args.get("parent_message_id") if invoke_from != InvokeFrom.SERVICE_API else UUID_NIL,
            user_id=user.id,
            invoke_from=invoke_from,
            extras=extras,
            trace_manager=trace_manager,
            stream=streaming,
        )

        # init generate records
        (conversation, message) = self._init_generate_records(application_generate_entity, conversation)

        # init queue manager
        queue_manager = MessageBasedAppQueueManager(
            task_id=application_generate_entity.task_id,
            user_id=application_generate_entity.user_id,
            invoke_from=application_generate_entity.invoke_from,
            conversation_id=conversation.id,
            app_mode=conversation.mode,
            message_id=message.id,
        )

        # new thread
        worker_thread = threading.Thread(
            target=self._generate_worker,
            kwargs={
                "flask_app": current_app._get_current_object(),  # type: ignore
                "application_generate_entity": application_generate_entity,
                "queue_manager": queue_manager,
                "conversation_id": conversation.id,
                "message_id": message.id,
            },
        )

        worker_thread.start()

        # return response or stream generator
        response = self._handle_response(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            conversation=conversation,
            message=message,
            user=user,
            stream=streaming,
        )

        return ChatAppGenerateResponseConverter.convert(response=response, invoke_from=invoke_from)

    def _generate_worker(
            self,
            flask_app: Flask,
            application_generate_entity: ChatAppGenerateEntity,
            queue_manager: AppQueueManager,
            conversation_id: str,
            message_id: str,
    ) -> None:
        """
        在新线程中生成工作器（worker）

        :param flask_app: Flask应用实例
        :param application_generate_entity: 聊天应用生成实体，包含生成所需的各种参数
        :param queue_manager: 队列管理器，用于管理消息队列
        :param conversation_id: 会话ID，标识当前对话
        :param message_id: 消息ID，标识当前消息
        :return: None
        """
        with flask_app.app_context():  # 确保在Flask应用上下文中执行
            try:
                # 获取会话和消息
                conversation = self._get_conversation(conversation_id)
                message = self._get_message(message_id)
                if message is None:
                    raise MessageNotExistsError("消息不存在")

                # 创建聊天应用运行器并执行
                runner = ChatAppRunner()
                runner.run(
                    application_generate_entity=application_generate_entity,
                    queue_manager=queue_manager,
                    conversation=conversation,
                    message=message,
                )
            except GenerateTaskStoppedError:  # 生成任务被停止
                pass
            except InvokeAuthorizationError:  # 调用授权错误（如API key错误）
                queue_manager.publish_error(
                    InvokeAuthorizationError("提供的API key不正确"),
                    PublishFrom.APPLICATION_MANAGER
                )
            except ValidationError as e:  # 数据验证错误
                logger.exception("生成时出现验证错误")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except ValueError as e:  # 数值错误
                if dify_config.DEBUG:  # 调试模式下记录详细错误
                    logger.exception("生成时出现错误")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            except Exception as e:  # 其他未知错误
                logger.exception("生成时出现未知错误")
                queue_manager.publish_error(e, PublishFrom.APPLICATION_MANAGER)
            finally:
                db.session.close()  # 确保数据库会话被关闭
