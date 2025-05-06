import logging
import time
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.app.apps.base_app_queue_manager import AppQueueManager
from core.app.entities.app_invoke_entities import (
    AppGenerateEntity,
)
from core.app.entities.queue_entities import (
    QueueErrorEvent,
)
from core.app.entities.task_entities import (
    ErrorStreamResponse,
    PingStreamResponse,
)
from core.errors.error import QuotaExceededError
from core.model_runtime.errors.invoke import InvokeAuthorizationError, InvokeError
from core.moderation.output_moderation import ModerationRule, OutputModeration
from models.model import Message

logger = logging.getLogger(__name__)


class BasedGenerateTaskPipeline:
    """
    生成任务处理管道基类，负责为应用程序生成流式输出和状态管理
    提供错误处理、输出审查等基础功能
    """

    def __init__(
        self,
        application_generate_entity: AppGenerateEntity,  # 应用生成实体，包含任务所需参数
        queue_manager: AppQueueManager,  # 队列管理器，用于消息队列操作
        stream: bool,  # 是否使用流式输出模式
    ) -> None:
        self._application_generate_entity = application_generate_entity  # 存储生成任务实体
        self._queue_manager = queue_manager  # 存储队列管理器实例
        self._start_at = time.perf_counter()  # 记录任务开始时间(性能计时)
        self._output_moderation_handler = self._init_output_moderation()  # 初始化输出内容审查处理器
        self._stream = stream  # 流式模式标志位

    def _handle_error(
        self,
        *,
        event: QueueErrorEvent,  # 队列错误事件对象
        session: Session | None = None,  # 数据库会话(可选)
        message_id: str = ""  # 关联消息ID(可选)
    ):
        """处理任务执行过程中出现的错误"""
        logger.debug("error: %s", event.error)
        e = event.error
        err: Exception  # 最终返回的异常对象

        # 根据错误类型进行转换处理
        if isinstance(e, InvokeAuthorizationError):
            err = InvokeAuthorizationError("提供的API密钥不正确")
        elif isinstance(e, InvokeError | ValueError):
            err = e  # 直接使用原始错误
        else:
            # 从错误对象中提取描述信息，若无则使用字符串表示
            err = Exception(e.description if getattr(e, "description", None) is not None else str(e))

        # 如果缺少必要参数则直接返回错误
        if not message_id or not session:
            return err

        # 查询关联的消息记录
        stmt = select(Message).where(Message.id == message_id)
        message = session.scalar(stmt)
        if not message:
            return err

        # 更新消息状态为错误
        err_desc = self._error_to_desc(err)
        message.status = "error"
        message.error = err_desc
        return err

    def _error_to_desc(self, e: Exception) -> str:
        """
        将异常转换为用户友好的描述信息
        :param e: 异常对象
        :return: 错误描述字符串
        """
        # 处理配额耗尽特殊场景
        if isinstance(e, QuotaExceededError):
            return (
                "您的DIFY托管模型提供商配额已用完。"
                "请前往设置->模型提供商完成自己的提供商凭证配置。"
            )

        # 优先使用异常的description属性，其次使用字符串表示
        message = getattr(e, "description", str(e))
        if not message:
            message = "内部服务器错误，请联系技术支持。"

        return message

    def _error_to_stream_response(self, e: Exception):
        """
        将错误转换为流式响应对象
        :param e: 异常对象
        :return: 错误流式响应对象
        """
        return ErrorStreamResponse(
            task_id=self._application_generate_entity.task_id,  # 携带任务ID
            err=e  # 错误对象
        )

    def _ping_stream_response(self) -> PingStreamResponse:
        """
        生成心跳流式响应(用于保持连接)
        :return: 心跳响应对象
        """
        return PingStreamResponse(task_id=self._application_generate_entity.task_id)

    def _init_output_moderation(self) -> Optional[OutputModeration]:
        """
        初始化输出内容审查处理器
        :return: 输出审查处理器实例或None
        """
        app_config = self._application_generate_entity.app_config
        sensitive_word_avoidance = app_config.sensitive_word_avoidance  # 获取敏感词避免配置

        if sensitive_word_avoidance:
            return OutputModeration(
                tenant_id=app_config.tenant_id,  # 租户ID
                app_id=app_config.app_id,  # 应用ID
                rule=ModerationRule(  # 审查规则
                    type=sensitive_word_avoidance.type,
                    config=sensitive_word_avoidance.config
                ),
                queue_manager=self._queue_manager,  # 传入队列管理器
            )
        return None  # 无敏感词配置时返回None

    def _handle_output_moderation_when_task_finished(self, completion: str) -> Optional[str]:
        """
        任务完成时处理输出内容审查
        :param completion: 生成的完整内容
        :return: 审查后的内容(如有修改)或None
        """
        if self._output_moderation_handler:
            self._output_moderation_handler.stop_thread()  # 停止审查线程

            # 执行最终内容审查
            completion = self._output_moderation_handler.moderation_completion(
                completion=completion,
                public_event=False  # 不发布公开事件
            )

            self._output_moderation_handler = None  # 清理处理器

            return completion  # 返回可能被修改的内容

        return None  # 无审查处理器时返回None