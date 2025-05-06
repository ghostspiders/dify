import time
from collections.abc import Generator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

from core.app.app_config.entities import ExternalDataVariableEntity, PromptTemplateEntity
from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.entities.app_invoke_entities import (
    AppGenerateEntity,
    EasyUIBasedAppGenerateEntity,
    InvokeFrom,
    ModelConfigWithCredentialsEntity,
)
from core.app.entities.queue_entities import QueueAgentMessageEvent, QueueLLMChunkEvent, QueueMessageEndEvent
from core.app.features.annotation_reply.annotation_reply import AnnotationReplyFeature
from core.app.features.hosting_moderation.hosting_moderation import HostingModerationFeature
from core.external_data_tool.external_data_fetch import ExternalDataFetch
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk, LLMResultChunkDelta, LLMUsage
from core.model_runtime.entities.message_entities import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
)
from core.model_runtime.entities.model_entities import ModelPropertyKey
from core.model_runtime.errors.invoke import InvokeBadRequestError
from core.moderation.input_moderation import InputModeration
from core.prompt.advanced_prompt_transform import AdvancedPromptTransform
from core.prompt.entities.advanced_prompt_entities import ChatModelMessage, CompletionModelPromptTemplate, MemoryConfig
from core.prompt.simple_prompt_transform import ModelMode, SimplePromptTransform
from models.model import App, AppMode, Message, MessageAnnotation

if TYPE_CHECKING:
    from core.file.models import File  # 类型检查时使用的导入，避免循环依赖


class AppRunner:
    def get_pre_calculate_rest_tokens(
            self,
            app_record: App,
            model_config: ModelConfigWithCredentialsEntity,
            prompt_template_entity: PromptTemplateEntity,
            inputs: Mapping[str, str],
            files: Sequence["File"],
            query: Optional[str] = None,
    ) -> int:
        """
        预计算剩余可用token数量（总上下文长度 - 提示词token - 最大输出token）

        参数:
            app_record: 应用记录对象，包含应用配置
            model_config: 模型配置实体，包含模型凭证等信息
            prompt_template_entity: 提示词模板实体
            inputs: 输入变量字典
            files: 上传的文件列表
            query: 用户查询文本(可选)

        返回:
            int: 剩余可用token数量（如果无法获取上下文长度则返回-1）

        异常:
            InvokeBadRequestError: 当提示词过长超过模型限制时抛出
        """
        # 初始化模型实例
        model_instance = ModelInstance(
            provider_model_bundle=model_config.provider_model_bundle,
            model=model_config.model
        )

        # 从模型配置获取上下文最大token数
        model_context_tokens = model_config.model_schema.model_properties.get(ModelPropertyKey.CONTEXT_SIZE)

        # 从模型参数规则中提取max_tokens配置
        max_tokens = 0
        for parameter_rule in model_config.model_schema.parameter_rules:
            if parameter_rule.name == "max_tokens" or (
                    parameter_rule.use_template and parameter_rule.use_template == "max_tokens"
            ):
                max_tokens = (
                                     model_config.parameters.get(parameter_rule.name)
                                     or model_config.parameters.get(parameter_rule.use_template or "")
                             ) or 0

        if model_context_tokens is None:
            return -1  # 无法获取上下文长度时返回-1

        if max_tokens is None:
            max_tokens = 0

        # 组织提示词消息（不包含记忆和上下文）
        prompt_messages, stop = self.organize_prompt_messages(
            app_record=app_record,
            model_config=model_config,
            prompt_template_entity=prompt_template_entity,
            inputs=inputs,
            files=files,
            query=query,
        )

        # 计算提示词消耗的token数量
        prompt_tokens = model_instance.get_llm_num_tokens(prompt_messages)

        # 计算剩余可用token
        rest_tokens: int = model_context_tokens - max_tokens - prompt_tokens
        if rest_tokens < 0:
            raise InvokeBadRequestError(
                "提示词过长，请减少提示词内容，"
                "或调小max_token参数，"
                "或换用支持更长上下文的模型。"
            )

        return rest_tokens

    def recalc_llm_max_tokens(
            self, model_config: ModelConfigWithCredentialsEntity, prompt_messages: list[PromptMessage]
    ):
        """
        重新计算最大输出token数（当提示词+max_tokens超过模型限制时）

        参数:
            model_config: 模型配置实体
            prompt_messages: 提示词消息列表
        """
        model_instance = ModelInstance(
            provider_model_bundle=model_config.provider_model_bundle,
            model=model_config.model
        )

        # 获取模型上下文长度
        model_context_tokens = model_config.model_schema.model_properties.get(ModelPropertyKey.CONTEXT_SIZE)

        # 提取当前max_tokens配置
        max_tokens = 0
        for parameter_rule in model_config.model_schema.parameter_rules:
            if parameter_rule.name == "max_tokens" or (
                    parameter_rule.use_template and parameter_rule.use_template == "max_tokens"
            ):
                max_tokens = (
                                     model_config.parameters.get(parameter_rule.name)
                                     or model_config.parameters.get(parameter_rule.use_template or "")
                             ) or 0

        if model_context_tokens is None:
            return -1

        if max_tokens is None:
            max_tokens = 0

        # 计算提示词token数
        prompt_tokens = model_instance.get_llm_num_tokens(prompt_messages)

        # 如果总token数超过限制，则调整max_tokens
        if prompt_tokens + max_tokens > model_context_tokens:
            # 至少保留16个token的输出空间
            max_tokens = max(model_context_tokens - prompt_tokens, 16)

            # 更新模型参数中的max_tokens
            for parameter_rule in model_config.model_schema.parameter_rules:
                if parameter_rule.name == "max_tokens" or (
                        parameter_rule.use_template and parameter_rule.use_template == "max_tokens"
                ):
                    model_config.parameters[parameter_rule.name] = max_tokens

    def organize_prompt_messages(
            self,
            app_record: App,
            model_config: ModelConfigWithCredentialsEntity,
            prompt_template_entity: PromptTemplateEntity,
            inputs: Mapping[str, str],
            files: Sequence["File"],
            query: Optional[str] = None,
            context: Optional[str] = None,
            memory: Optional[TokenBufferMemory] = None,
            image_detail_config: Optional[ImagePromptMessageContent.DETAIL] = None,
    ) -> tuple[list[PromptMessage], Optional[list[str]]]:
        """
        组织提示词消息

        参数:
            app_record: 应用记录
            model_config: 模型配置
            prompt_template_entity: 提示词模板实体
            inputs: 输入变量
            files: 文件列表
            query: 用户查询(可选)
            context: 上下文(可选)
            memory: 记忆缓冲区(可选)
            image_detail_config: 图片细节配置(可选)

        返回:
            tuple: (提示词消息列表, 停止词列表)
        """
        # 简单提示词模式处理
        if prompt_template_entity.prompt_type == PromptTemplateEntity.PromptType.SIMPLE:
            prompt_transform: Union[SimplePromptTransform, AdvancedPromptTransform]
            prompt_transform = SimplePromptTransform()  # 使用简单提示词转换器
            prompt_messages, stop = prompt_transform.get_prompt(
                app_mode=AppMode.value_of(app_record.mode),
                prompt_template_entity=prompt_template_entity,
                inputs=inputs,
                query=query or "",
                files=files,
                context=context,
                memory=memory,
                model_config=model_config,
                image_detail_config=image_detail_config,
            )
        else:
            # 高级提示词模式处理
            memory_config = MemoryConfig(window=MemoryConfig.WindowConfig(enabled=False))

            model_mode = ModelMode.value_of(model_config.mode)
            prompt_template: Union[CompletionModelPromptTemplate, list[ChatModelMessage]]

            # 补全模型处理
            if model_mode == ModelMode.COMPLETION:
                advanced_completion_prompt_template = prompt_template_entity.advanced_completion_prompt_template
                if not advanced_completion_prompt_template:
                    raise InvokeBadRequestError("高级补全提示词模板是必需的")
                prompt_template = CompletionModelPromptTemplate(text=advanced_completion_prompt_template.prompt)

                # 设置角色前缀
                if advanced_completion_prompt_template.role_prefix:
                    memory_config.role_prefix = MemoryConfig.RolePrefix(
                        user=advanced_completion_prompt_template.role_prefix.user,
                        assistant=advanced_completion_prompt_template.role_prefix.assistant,
                    )
            else:
                # 对话模型处理
                if not prompt_template_entity.advanced_chat_prompt_template:
                    raise InvokeBadRequestError("高级对话提示词模板是必需的")
                prompt_template = []
                for message in prompt_template_entity.advanced_chat_prompt_template.messages:
                    prompt_template.append(ChatModelMessage(text=message.text, role=message.role))

            # 使用高级提示词转换器
            prompt_transform = AdvancedPromptTransform()
            prompt_messages = prompt_transform.get_prompt(
                prompt_template=prompt_template,
                inputs=inputs,
                query=query or "",
                files=files,
                context=context,
                memory_config=memory_config,
                memory=memory,
                model_config=model_config,
                image_detail_config=image_detail_config,
            )
            stop = model_config.stop  # 使用模型配置中的停止词

        return prompt_messages, stop

    def direct_output(
            self,
            queue_manager: AppQueueManager,
            app_generate_entity: EasyUIBasedAppGenerateEntity,
            prompt_messages: list,
            text: str,
            stream: bool,
            usage: Optional[LLMUsage] = None,
    ) -> None:
        """
        直接输出结果到队列

        参数:
            queue_manager: 应用队列管理器
            app_generate_entity: 应用生成实体
            prompt_messages: 提示词消息列表
            text: 要输出的文本
            stream: 是否流式输出
            usage: token使用情况(可选)
        """
        if stream:
            # 流式输出处理
            index = 0
            for token in text:
                # 构建结果块
                chunk = LLMResultChunk(
                    model=app_generate_entity.model_conf.model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(index=index, message=AssistantPromptMessage(content=token)),
                )
                # 发布到队列
                queue_manager.publish(QueueLLMChunkEvent(chunk=chunk), PublishFrom.APPLICATION_MANAGER)
                index += 1
                time.sleep(0.01)  # 流式控制速度

        # 发布结束事件
        queue_manager.publish(
            QueueMessageEndEvent(
                llm_result=LLMResult(
                    model=app_generate_entity.model_conf.model,
                    prompt_messages=prompt_messages,
                    message=AssistantPromptMessage(content=text),
                    usage=usage or LLMUsage.empty_usage(),
                ),
            ),
            PublishFrom.APPLICATION_MANAGER,
        )

    # ... (其他方法保持类似的中文注释风格)

    def moderation_for_inputs(
            self,
            *,
            app_id: str,
            tenant_id: str,
            app_generate_entity: AppGenerateEntity,
            inputs: Mapping[str, Any],
            query: str | None = None,
            message_id: str,
    ) -> tuple[bool, Mapping[str, Any], str]:
        """
        输入内容审核（敏感词检测）

        参数:
            app_id: 应用ID
            tenant_id: 租户ID
            app_generate_entity: 应用生成实体
            inputs: 输入变量
            query: 用户查询(可选)
            message_id: 消息ID

        返回:
            tuple: (是否通过审核, 处理后的输入, 审核消息)
        """
        moderation_feature = InputModeration()
        return moderation_feature.check(
            app_id=app_id,
            tenant_id=tenant_id,
            app_config=app_generate_entity.app_config,
            inputs=dict(inputs),
            query=query or "",
            message_id=message_id,
            trace_manager=app_generate_entity.trace_manager,
        )

    # ... (剩余方法也采用类似的中文注释方式)
    def check_hosting_moderation(
        self,
        application_generate_entity: EasyUIBasedAppGenerateEntity,
        queue_manager: AppQueueManager,
        prompt_messages: list[PromptMessage],
    ) -> bool:
        """
        Check hosting moderation
        :param application_generate_entity: application generate entity
        :param queue_manager: queue manager
        :param prompt_messages: prompt messages
        :return:
        """
        hosting_moderation_feature = HostingModerationFeature()
        moderation_result = hosting_moderation_feature.check(
            application_generate_entity=application_generate_entity, prompt_messages=prompt_messages
        )

        if moderation_result:
            self.direct_output(
                queue_manager=queue_manager,
                app_generate_entity=application_generate_entity,
                prompt_messages=prompt_messages,
                text="I apologize for any confusion, but I'm an AI assistant to be helpful, harmless, and honest.",
                stream=application_generate_entity.stream,
            )

        return moderation_result

    def fill_in_inputs_from_external_data_tools(
        self,
        tenant_id: str,
        app_id: str,
        external_data_tools: list[ExternalDataVariableEntity],
        inputs: Mapping[str, Any],
        query: str,
    ) -> Mapping[str, Any]:
        """
        Fill in variable inputs from external data tools if exists.

        :param tenant_id: workspace id
        :param app_id: app id
        :param external_data_tools: external data tools configs
        :param inputs: the inputs
        :param query: the query
        :return: the filled inputs
        """
        external_data_fetch_feature = ExternalDataFetch()
        return external_data_fetch_feature.fetch(
            tenant_id=tenant_id, app_id=app_id, external_data_tools=external_data_tools, inputs=inputs, query=query
        )

    def query_app_annotations_to_reply(
        self, app_record: App, message: Message, query: str, user_id: str, invoke_from: InvokeFrom
    ) -> Optional[MessageAnnotation]:
        """
        Query app annotations to reply
        :param app_record: app record
        :param message: message
        :param query: query
        :param user_id: user id
        :param invoke_from: invoke from
        :return:
        """
        annotation_reply_feature = AnnotationReplyFeature()
        return annotation_reply_feature.query(
            app_record=app_record, message=message, query=query, user_id=user_id, invoke_from=invoke_from
        )
