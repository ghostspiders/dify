from collections.abc import Sequence
from typing import Optional

from core.app.app_config.features.file_upload.manager import FileUploadConfigManager
from core.file import file_manager
from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageRole,
    TextPromptMessageContent,
    UserPromptMessage,
)
from core.model_runtime.entities.message_entities import PromptMessageContentUnionTypes
from core.prompt.utils.extract_thread_messages import extract_thread_messages
from extensions.ext_database import db
from factories import file_factory
from models.model import AppMode, Conversation, Message, MessageFile
from models.workflow import WorkflowRun


class TokenBufferMemory:
    """
    基于Token缓冲的记忆管理系统
    用于管理对话历史，根据token限制智能截断历史消息
    """

    def __init__(self, conversation: Conversation, model_instance: ModelInstance) -> None:
        """
        初始化记忆系统

        :param conversation: 当前会话对象
        :param model_instance: 模型实例，用于token计算
        """
        self.conversation = conversation
        self.model_instance = model_instance

    def get_history_prompt_messages(
            self, max_token_limit: int = 2000, message_limit: Optional[int] = None
    ) -> Sequence[PromptMessage]:
        """
        获取历史对话的提示消息列表（带token限制）

        :param max_token_limit: 最大token限制，默认2000
        :param message_limit: 消息数量限制（可选）
        :return: 提示消息序列
        """
        app_record = self.conversation.app  # 获取关联的应用记录

        # 查询最近的对话消息（按时间倒序）
        query = (
            db.session.query(
                Message.id,
                Message.query,  # 用户查询内容
                Message.answer,  # 助手回复内容
                Message.created_at,  # 创建时间
                Message.workflow_run_id,  # 工作流运行ID
                Message.parent_message_id,  # 父消息ID
                Message.answer_tokens,  # 回答token数
            )
            .filter(
                Message.conversation_id == self.conversation.id,  # 当前会话的消息
            )
            .order_by(Message.created_at.desc())  # 按创建时间降序
        )

        # 设置消息数量限制（默认最多500条）
        if message_limit and message_limit > 0:
            message_limit = min(message_limit, 500)
        else:
            message_limit = 500

        messages = query.limit(message_limit).all()  # 执行查询

        # 提取当前消息线程中的消息（过滤无关分支）
        thread_messages = extract_thread_messages(messages)

        # 如果最新消息是刚创建的空消息（尚未回答），则从记忆中移除
        if thread_messages and not thread_messages[0].answer and thread_messages[0].answer_tokens == 0:
            thread_messages.pop(0)

        messages = list(reversed(thread_messages))  # 反转列表变为时间正序

        prompt_messages: list[PromptMessage] = []  # 准备提示消息列表

        for message in messages:
            # 查询消息关联的文件
            files = db.session.query(MessageFile).filter(MessageFile.message_id == message.id).all()

            if files:  # 处理带文件的消息
                file_extra_config = None
                # 根据会话模式获取文件上传配置
                if self.conversation.mode not in {AppMode.ADVANCED_CHAT, AppMode.WORKFLOW}:
                    file_extra_config = FileUploadConfigManager.convert(self.conversation.model_config)
                else:
                    if message.workflow_run_id:
                        workflow_run = (
                            db.session.query(WorkflowRun).filter(WorkflowRun.id == message.workflow_run_id).first()
                        )
                        if workflow_run and workflow_run.workflow:
                            file_extra_config = FileUploadConfigManager.convert(
                                workflow_run.workflow.features_dict, is_vision=False
                            )

                detail = ImagePromptMessageContent.DETAIL.LOW  # 默认图片细节级别
                if file_extra_config and app_record:
                    # 构建文件对象
                    file_objs = file_factory.build_from_message_files(
                        message_files=files, tenant_id=app_record.tenant_id, config=file_extra_config
                    )
                    if file_extra_config.image_config and file_extra_config.image_config.detail:
                        detail = file_extra_config.image_config.detail
                else:
                    file_objs = []

                if not file_objs:  # 无有效文件对象
                    prompt_messages.append(UserPromptMessage(content=message.query))
                else:  # 构建复合内容消息（文本+文件）
                    prompt_message_contents: list[PromptMessageContentUnionTypes] = []
                    prompt_message_contents.append(TextPromptMessageContent(data=message.query))  # 文本部分
                    for file in file_objs:
                        # 将文件转换为提示消息内容
                        prompt_message = file_manager.to_prompt_message_content(
                            file,
                            image_detail_config=detail,
                        )
                        prompt_message_contents.append(prompt_message)

                    prompt_messages.append(UserPromptMessage(content=prompt_message_contents))
            else:  # 纯文本消息
                prompt_messages.append(UserPromptMessage(content=message.query))

            # 添加助手回复
            prompt_messages.append(AssistantPromptMessage(content=message.answer))

        if not prompt_messages:
            return []  # 无历史消息时返回空列表

        # Token数量检查与截断
        curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)
        if curr_message_tokens > max_token_limit:
            pruned_memory = []
            # 从最早的消息开始移除，直到满足token限制
            while curr_message_tokens > max_token_limit and len(prompt_messages) > 1:
                pruned_memory.append(prompt_messages.pop(0))
                curr_message_tokens = self.model_instance.get_llm_num_tokens(prompt_messages)

        return prompt_messages

    def get_history_prompt_text(
            self,
            human_prefix: str = "Human",  # 用户前缀标识
            ai_prefix: str = "Assistant",  # AI前缀标识
            max_token_limit: int = 2000,  # 最大token限制
            message_limit: Optional[int] = None,  # 消息数量限制
    ) -> str:
        """
        获取格式化后的历史对话文本（用于纯文本场景）

        :param human_prefix: 用户消息前缀
        :param ai_prefix: AI消息前缀
        :param max_token_limit: token限制
        :param message_limit: 消息数量限制
        :return: 格式化后的对话文本
        """
        # 获取提示消息列表
        prompt_messages = self.get_history_prompt_messages(
            max_token_limit=max_token_limit,
            message_limit=message_limit
        )

        string_messages = []  # 存储格式化后的消息
        for m in prompt_messages:
            # 确定角色前缀
            if m.role == PromptMessageRole.USER:
                role = human_prefix
            elif m.role == PromptMessageRole.ASSISTANT:
                role = ai_prefix
            else:
                continue  # 跳过其他角色

            # 处理复合内容消息
            if isinstance(m.content, list):
                inner_msg = ""
                for content in m.content:
                    if isinstance(content, TextPromptMessageContent):  # 文本内容
                        inner_msg += f"{content.data}\n"
                    elif isinstance(content, ImagePromptMessageContent):  # 图片内容
                        inner_msg += "[image]\n"

                string_messages.append(f"{role}: {inner_msg.strip()}")
            else:  # 简单文本消息
                message = f"{role}: {m.content}"
                string_messages.append(message)

        return "\n".join(string_messages)  # 合并为单一字符串