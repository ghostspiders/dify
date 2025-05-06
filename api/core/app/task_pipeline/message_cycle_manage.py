import logging
from threading import Thread
from typing import Optional, Union

from flask import Flask, current_app

from configs import dify_config
from core.app.entities.app_invoke_entities import (
    AdvancedChatAppGenerateEntity,
    AgentChatAppGenerateEntity,
    ChatAppGenerateEntity,
    CompletionAppGenerateEntity,
)
from core.app.entities.queue_entities import (
    QueueAnnotationReplyEvent,
    QueueMessageFileEvent,
    QueueRetrieverResourcesEvent,
)
from core.app.entities.task_entities import (
    EasyUITaskState,
    MessageFileStreamResponse,
    MessageReplaceStreamResponse,
    MessageStreamResponse,
    WorkflowTaskState,
)
from core.llm_generator.llm_generator import LLMGenerator
from core.tools.tool_file_manager import ToolFileManager
from extensions.ext_database import db
from models.model import AppMode, Conversation, MessageAnnotation, MessageFile
from services.annotation_service import AppAnnotationService


class MessageCycleManage:
    """
    消息生命周期管理类，负责处理消息相关的各种操作：
    - 会话名称生成
    - 标注回复处理
    - 检索资源处理
    - 消息文件处理
    - 消息流响应转换
    """

    def __init__(
            self,
            *,
            application_generate_entity: Union[
                ChatAppGenerateEntity,
                CompletionAppGenerateEntity,
                AgentChatAppGenerateEntity,
                AdvancedChatAppGenerateEntity,
            ],
            task_state: Union[EasyUITaskState, WorkflowTaskState],
    ) -> None:
        """
        初始化消息周期管理器

        :param application_generate_entity: 应用生成实体(多种类型)
        :param task_state: 任务状态对象
        """
        self._application_generate_entity = application_generate_entity
        self._task_state = task_state

    def _generate_conversation_name(self, *, conversation_id: str, query: str) -> Optional[Thread]:
        """
        生成会话名称（异步线程方式）

        :param conversation_id: 会话ID
        :param query: 用户查询内容
        :return: 生成线程对象或None（如果不需生成）
        """
        # 补全应用不需要生成会话名称
        if isinstance(self._application_generate_entity, CompletionAppGenerateEntity):
            return None

        # 检查是否为第一条消息且配置了自动生成名称
        is_first_message = self._application_generate_entity.conversation_id is None
        extras = self._application_generate_entity.extras
        auto_generate_conversation_name = extras.get("auto_generate_conversation_name", True)

        if auto_generate_conversation_name and is_first_message:
            # 创建并启动生成线程
            thread = Thread(
                target=self._generate_conversation_name_worker,
                kwargs={
                    "flask_app": current_app._get_current_object(),  # 获取当前Flask应用
                    "conversation_id": conversation_id,
                    "query": query,
                },
            )
            thread.start()
            return thread

        return None

    def _generate_conversation_name_worker(self, flask_app: Flask, conversation_id: str, query: str):
        """
        会话名称生成工作线程

        :param flask_app: Flask应用实例
        :param conversation_id: 会话ID
        :param query: 用户查询内容
        """
        with flask_app.app_context():  # 确保在应用上下文中执行
            # 查询会话记录
            conversation = db.session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if not conversation:
                return

            # 非补全模式才生成名称
            if conversation.mode != AppMode.COMPLETION.value:
                app_model = conversation.app
                if not app_model:
                    return

                # 调用LLM生成会话名称
                try:
                    name = LLMGenerator.generate_conversation_name(app_model.tenant_id, query)
                    conversation.name = name
                except Exception as e:
                    if dify_config.DEBUG:
                        logging.exception(f"生成会话名称失败, conversation_id: {conversation_id}")
                    pass

                # 保存会话名称
                db.session.merge(conversation)
                db.session.commit()
                db.session.close()

    def _handle_annotation_reply(self, event: QueueAnnotationReplyEvent) -> Optional[MessageAnnotation]:
        """
        处理标注回复事件

        :param event: 标注回复事件
        :return: 消息标注对象或None
        """
        annotation = AppAnnotationService.get_annotation_by_id(event.message_annotation_id)
        if annotation:
            account = annotation.account
            # 在任务状态中记录标注回复信息
            self._task_state.metadata["annotation_reply"] = {
                "id": annotation.id,
                "account": {"id": annotation.account_id, "name": account.name if account else "Dify user"},
            }
            return annotation
        return None

    def _handle_retriever_resources(self, event: QueueRetrieverResourcesEvent) -> None:
        """
        处理检索资源事件

        :param event: 检索资源事件
        """
        # 如果应用配置显示检索来源，则记录检索资源
        if self._application_generate_entity.app_config.additional_features.show_retrieve_source:
            self._task_state.metadata["retriever_resources"] = event.retriever_resources

    def _message_file_to_stream_response(self, event: QueueMessageFileEvent) -> Optional[MessageFileStreamResponse]:
        """
        将消息文件事件转换为流式响应

        :param event: 消息文件事件
        :return: 消息文件流式响应或None
        """
        message_file = db.session.query(MessageFile).filter(MessageFile.id == event.message_file_id).first()

        if message_file and message_file.url is not None:
            # 从URL中提取文件ID
            tool_file_id = message_file.url.split("/")[-1]
            tool_file_id = tool_file_id.split(".")[0]  # 去除扩展名

            # 获取文件扩展名
            if "." in message_file.url:
                extension = f".{message_file.url.split('.')[-1]}"
                if len(extension) > 10:  # 扩展名过长处理
                    extension = ".bin"
            else:
                extension = ".bin"

            # 处理URL（远程URL直接使用，本地文件生成签名URL）
            url = message_file.url if message_file.url.startswith("http") \
                else ToolFileManager.sign_file(tool_file_id=tool_file_id, extension=extension)

            return MessageFileStreamResponse(
                task_id=self._application_generate_entity.task_id,
                id=message_file.id,
                type=message_file.type,
                belongs_to=message_file.belongs_to or "user",
                url=url,
            )
        return None

    def _message_to_stream_response(
            self, answer: str, message_id: str, from_variable_selector: Optional[list[str]] = None
    ) -> MessageStreamResponse:
        """
        将消息转换为流式响应

        :param answer: 回答内容
        :param message_id: 消息ID
        :param from_variable_selector: 变量选择器来源(可选)
        :return: 消息流式响应
        """
        return MessageStreamResponse(
            task_id=self._application_generate_entity.task_id,
            id=message_id,
            answer=answer,
            from_variable_selector=from_variable_selector,
        )

    def _message_replace_to_stream_response(self, answer: str) -> MessageReplaceStreamResponse:
        """
        将消息替换内容转换为流式响应

        :param answer: 替换内容
        :return: 消息替换流式响应
        """
        return MessageReplaceStreamResponse(task_id=self._application_generate_entity.task_id, answer=answer)