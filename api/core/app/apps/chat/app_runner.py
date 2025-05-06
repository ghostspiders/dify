import logging
from typing import cast

from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.base_app_runner import AppRunner
from core.app.apps.chat.app_config_manager import ChatAppConfig
from core.app.entities.app_invoke_entities import (
    ChatAppGenerateEntity,
)
from core.app.entities.queue_entities import QueueAnnotationReplyEvent
from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities.message_entities import ImagePromptMessageContent
from core.moderation.base import ModerationError
from core.rag.retrieval.dataset_retrieval import DatasetRetrieval
from extensions.ext_database import db
from models.model import App, Conversation, Message

logger = logging.getLogger(__name__)


class ChatAppRunner(AppRunner):
    """
    聊天应用运行器，继承自AppRunner基类
    负责处理聊天应用的完整执行流程，包括：
    - 输入处理
    - 记忆管理
    - 提示词组织
    - 内容审查
    - 模型调用
    - 结果处理
    """

    def run(
            self,
            application_generate_entity: ChatAppGenerateEntity,  # 应用生成实体，包含所有生成参数
            queue_manager: AppQueueManager,  # 队列管理器，用于事件发布
            conversation: Conversation,  # 当前会话对象
            message: Message,  # 当前消息对象
    ) -> None:
        """
        运行聊天应用的主方法

        参数:
            application_generate_entity: 包含应用配置、模型配置、用户输入等
            queue_manager: 用于发布各种事件到消息队列
            conversation: 当前会话的数据库对象
            message: 当前消息的数据库对象
        """
        # 获取应用配置并转换为聊天应用专用配置
        app_config = application_generate_entity.app_config
        app_config = cast(ChatAppConfig, app_config)

        # 查询应用
        app_record = db.session.query(App).filter(App.id == app_config.app_id).first()
        if not app_record:
            raise ValueError("应用不存在")

        # 获取输入参数
        inputs = application_generate_entity.inputs  # 模板变量输入
        query = application_generate_entity.query  # 用户查询文本
        files = application_generate_entity.files  # 上传的文件

        # 处理图片细节配置（默认为LOW）
        image_detail_config = (
            application_generate_entity.file_upload_config.image_config.detail
            if (
                    application_generate_entity.file_upload_config
                    and application_generate_entity.file_upload_config.image_config
            )
            else None
        )
        image_detail_config = image_detail_config or ImagePromptMessageContent.DETAIL.LOW

        # 初始化记忆系统（如果有会话ID）
        memory = None
        if application_generate_entity.conversation_id:
            model_instance = ModelInstance(
                provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
                model=application_generate_entity.model_conf.model,
            )
            memory = TokenBufferMemory(conversation=conversation, model_instance=model_instance)

        # 第一次组织提示消息（包含模板、输入、查询、文件、记忆等）
        prompt_messages, stop = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query,
            memory=memory,
            image_detail_config=image_detail_config,
        )

        # 输入内容审查（敏感词过滤）
        try:
            _, inputs, query = self.moderation_for_inputs(
                app_id=app_record.id,
                tenant_id=app_config.tenant_id,
                app_generate_entity=application_generate_entity,
                inputs=inputs,
                query=query,
                message_id=message.id,
            )
        except ModerationError as e:
            # 如果审查不通过，直接返回错误信息
            self.direct_output(
                queue_manager=queue_manager,
                app_generate_entity=application_generate_entity,
                prompt_messages=prompt_messages,
                text=str(e),
                stream=application_generate_entity.stream,
            )
            return

        # 处理标注回复（如果查询匹配已有标注）
        if query:
            annotation_reply = self.query_app_annotations_to_reply(
                app_record=app_record,
                message=message,
                query=query,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from,
            )

            if annotation_reply:
                # 发布标注回复事件
                queue_manager.publish(
                    QueueAnnotationReplyEvent(message_annotation_id=annotation_reply.id),
                    PublishFrom.APPLICATION_MANAGER,
                )
                # 直接返回标注内容
                self.direct_output(
                    queue_manager=queue_manager,
                    app_generate_entity=application_generate_entity,
                    prompt_messages=prompt_messages,
                    text=annotation_reply.content,
                    stream=application_generate_entity.stream,
                )
                return

        # 从外部数据工具填充变量（如果有配置）
        external_data_tools = app_config.external_data_variables
        if external_data_tools:
            inputs = self.fill_in_inputs_from_external_data_tools(
                tenant_id=app_record.tenant_id,
                app_id=app_record.id,
                external_data_tools=external_data_tools,
                inputs=inputs,
                query=query,
            )

        # 从数据集获取上下文（如果配置了数据集）
        context = None
        if app_config.dataset and app_config.dataset.dataset_ids:
            hit_callback = DatasetIndexToolCallbackHandler(
                queue_manager,
                app_record.id,
                message.id,
                application_generate_entity.user_id,
                application_generate_entity.invoke_from,
            )

            dataset_retrieval = DatasetRetrieval(application_generate_entity)
            context = dataset_retrieval.retrieve(
                app_id=app_record.id,
                user_id=application_generate_entity.user_id,
                tenant_id=app_record.tenant_id,
                model_config=application_generate_entity.model_conf,
                config=app_config.dataset,
                query=query,
                invoke_from=application_generate_entity.invoke_from,
                show_retrieve_source=app_config.additional_features.show_retrieve_source,
                hit_callback=hit_callback,
                memory=memory,
                message_id=message.id,
                inputs=inputs,
            )

        # 重新组织提示消息（加入外部数据和数据集上下文）
        prompt_messages, stop = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=inputs,
            files=files,
            query=query,
            context=context,
            memory=memory,
            image_detail_config=image_detail_config,
        )

        # 托管内容审查检查
        hosting_moderation_result = self.check_hosting_moderation(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            prompt_messages=prompt_messages,
        )
        if hosting_moderation_result:
            return

        # 重新计算最大token数（防止超过模型限制）
        self.recalc_llm_max_tokens(
            model_config=application_generate_entity.model_conf,
            prompt_messages=prompt_messages
        )

        # 创建模型实例并调用LLM
        model_instance = ModelInstance(
            provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
            model=application_generate_entity.model_conf.model,
        )

        db.session.close()  # 关闭数据库会话

        # 调用大语言模型
        invoke_result = model_instance.invoke_llm(
            prompt_messages=prompt_messages,
            model_parameters=application_generate_entity.model_conf.parameters,
            stop=stop,
            stream=application_generate_entity.stream,  # 是否流式输出
            user=application_generate_entity.user_id,
        )

        # 处理调用结果
        self._handle_invoke_result(
            invoke_result=invoke_result,
            queue_manager=queue_manager,
            stream=application_generate_entity.stream
        )