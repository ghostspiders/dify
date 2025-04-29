import logging
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import IO, Any, Literal, Optional, Union, cast, overload

from configs import dify_config
from core.entities.embedding_type import EmbeddingInputType
from core.entities.provider_configuration import ProviderConfiguration, ProviderModelBundle
from core.entities.provider_entities import ModelLoadBalancingConfiguration
from core.errors.error import ProviderTokenNotInitError
from core.model_runtime.callbacks.base_callback import Callback
from core.model_runtime.entities.llm_entities import LLMResult
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.rerank_entities import RerankResult
from core.model_runtime.entities.text_embedding_entities import TextEmbeddingResult
from core.model_runtime.errors.invoke import InvokeAuthorizationError, InvokeConnectionError, InvokeRateLimitError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.moderation_model import ModerationModel
from core.model_runtime.model_providers.__base.rerank_model import RerankModel
from core.model_runtime.model_providers.__base.speech2text_model import Speech2TextModel
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.model_runtime.model_providers.__base.tts_model import TTSModel
from core.provider_manager import ProviderManager
from extensions.ext_redis import redis_client
from models.provider import ProviderType

logger = logging.getLogger(__name__)


class ModelInstance:
    """模型实例类，用于管理特定模型的配置、凭证及负载均衡"""

    def __init__(self, provider_model_bundle: ProviderModelBundle, model: str) -> None:
        """初始化模型实例

        Args:
            provider_model_bundle: 提供模型配置的包，包含提供商配置和模型类型实例
            model: 目标模型名称
        """
        self.provider_model_bundle = provider_model_bundle
        self.model = model
        self.provider = provider_model_bundle.configuration.provider.provider  # 提供商名称
        self.credentials = self._fetch_credentials_from_bundle(provider_model_bundle, model)  # 模型凭证
        self.model_type_instance = self.provider_model_bundle.model_type_instance  # 模型类型实例
        # 负载均衡管理器（如果启用）
        self.load_balancing_manager = self._get_load_balancing_manager(
            configuration=provider_model_bundle.configuration,
            model_type=provider_model_bundle.model_type_instance.model_type,
            model=model,
            credentials=self.credentials,
        )

    @staticmethod
    def _fetch_credentials_from_bundle(provider_model_bundle: ProviderModelBundle, model: str) -> dict:
        """从配置包中获取指定模型的访问凭证

        Args:
            provider_model_bundle: 提供模型配置的包
            model: 目标模型名称

        Returns:
            dict: 模型凭证字典

        Raises:
            ProviderTokenNotInitError: 当凭证未初始化时抛出
        """
        configuration = provider_model_bundle.configuration
        model_type = provider_model_bundle.model_type_instance.model_type
        # 获取当前模型的凭证配置
        credentials = configuration.get_current_credentials(model_type=model_type, model=model)

        if credentials is None:
            raise ProviderTokenNotInitError(f"Model {model} credentials is not initialized.")

        return credentials

    @staticmethod
    def _get_load_balancing_manager(
            configuration: ProviderConfiguration, model_type: ModelType, model: str, credentials: dict
    ) -> Optional["LBModelManager"]:
        """创建负载均衡管理器（如果配置中启用）

        Args:
            configuration: 提供商配置
            model_type: 模型类型
            model: 模型名称
            credentials: 模型凭证字典

        Returns:
            LBModelManager: 负载均衡管理器实例（如果启用），否则返回None
        """
        # 仅当使用自定义提供商且配置了负载均衡时启用
        if configuration.model_settings and configuration.using_provider_type == ProviderType.CUSTOM:
            current_model_setting = None
            # 遍历模型设置查找匹配项
            for model_setting in configuration.model_settings:
                if model_setting.model_type == model_type and model_setting.model == model:
                    current_model_setting = model_setting
                    break

            # 检查负载均衡配置
            if current_model_setting and current_model_setting.load_balancing_configs:
                # 初始化负载均衡管理器
                lb_model_manager = LBModelManager(
                    tenant_id=configuration.tenant_id,
                    provider=configuration.provider.provider,
                    model_type=model_type,
                    model=model,
                    load_balancing_configs=current_model_setting.load_balancing_configs,
                    managed_credentials=credentials if configuration.custom_configuration.provider else None,
                )
                return lb_model_manager

        return None

    @overload
    def invoke_llm(
            self,
            prompt_messages: list[PromptMessage],
            model_parameters: Optional[dict] = None,
            tools: Sequence[PromptMessageTool] | None = None,
            stop: Optional[list[str]] = None,
            stream: Literal[True] = True,
            user: Optional[str] = None,
            callbacks: Optional[list[Callback]] = None,
    ) -> Generator:
        """调用大语言模型（流式模式）

        Args:
            prompt_messages: 输入的提示消息列表
            model_parameters: 模型参数覆盖
            tools: 可用工具列表
            stop: 停止生成的关键词序列
            stream: 是否启用流式输出（强制为True）
            user: 用户标识符
            callbacks: 回调函数列表

        Returns:
            Generator: 流式响应生成器
        """
        ...

    def invoke_llm(
            self,
            prompt_messages: Sequence[PromptMessage],
            model_parameters: Optional[dict] = None,
            tools: Sequence[PromptMessageTool] | None = None,
            stop: Optional[Sequence[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
            callbacks: Optional[list[Callback]] = None,
    ) -> Union[LLMResult, Generator]:
        """
        调用大语言模型（支持流式/非流式模式）

        Args:
            prompt_messages: 提示消息序列，构成LLM输入的对话上下文
            model_parameters: 模型参数覆盖字典，用于自定义生成参数（如temperature, top_p）
            tools: 工具调用列表，定义模型可使用的外部工具
            stop: 停止词序列，触发这些词时停止文本生成
            stream: 是否启用流式响应模式，默认True
            user: 终端用户ID，用于审计和配额管理
            callbacks: 回调函数列表，用于处理中间结果或日志记录

        Returns:
            Union[LLMResult, Generator]: 流式模式返回生成器对象，非流式返回完整结果对象

        Raises:
            Exception: 当模型类型实例不是LargeLanguageModel时抛出
        """
        # 类型安全检查：确保当前模型实例是大语言模型类型
        if not isinstance(self.model_type_instance, LargeLanguageModel):
            raise Exception("Model type instance is not LargeLanguageModel")

        # 显式类型转换确保类型安全
        self.model_type_instance = cast(LargeLanguageModel, self.model_type_instance)

        # 通过轮询机制调用模型实例的invoke方法
        return cast(
            Union[LLMResult, Generator],
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,  # 底层模型调用方法
                model=self.model,  # 模型名称标识
                credentials=self.credentials,  # 认证凭据
                prompt_messages=prompt_messages,  # 透传所有参数
                model_parameters=model_parameters,
                tools=tools,
                stop=stop,
                stream=stream,
                user=user,
                callbacks=callbacks,
            ),
        )

    def get_llm_num_tokens(
            self, prompt_messages: Sequence[PromptMessage], tools: Optional[Sequence[PromptMessageTool]] = None
    ) -> int:
        """
        计算提示消息的token数量

        Args:
            prompt_messages: 需要计算token的提示消息序列
            tools: 工具调用列表（影响token计算）

        Returns:
            int: 总token数量

        Raises:
            Exception: 模型类型不匹配时抛出
        """
        if not isinstance(self.model_type_instance, LargeLanguageModel):
            raise Exception("Model type instance is not LargeLanguageModel")

        self.model_type_instance = cast(LargeLanguageModel, self.model_type_instance)
        return cast(
            int,
            self._round_robin_invoke(
                function=self.model_type_instance.get_num_tokens,  # 调用底层计数方法
                model=self.model,
                credentials=self.credentials,
                prompt_messages=prompt_messages,
                tools=tools,
            ),
        )

    def invoke_text_embedding(
            self, texts: list[str], user: Optional[str] = None,
            input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT
    ) -> TextEmbeddingResult:
        """
        执行文本嵌入向量化

        Args:
            texts: 需要嵌入的文本列表
            user: 终端用户标识（可选）
            input_type: 输入类型枚举，指定文档/查询模式（影响嵌入策略）

        Returns:
            TextEmbeddingResult: 包含嵌入向量和元数据的结果对象

        Raises:
            Exception: 当模型类型不是文本嵌入模型时抛出
        """
        if not isinstance(self.model_type_instance, TextEmbeddingModel):
            raise Exception("Model type instance is not TextEmbeddingModel")

        self.model_type_instance = cast(TextEmbeddingModel, self.model_type_instance)
        return cast(
            TextEmbeddingResult,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials,
                texts=texts,
                user=user,
                input_type=input_type,
            ),
        )

    def get_text_embedding_num_tokens(self, texts: list[str]) -> list[int]:
        """
        获取每个文本的token数量

        Args:
            texts: 需要计算的文本列表

        Returns:
            list[int]: 每个文本对应的token数量列表

        Raises:
            Exception: 模型类型不匹配时抛出
        """
        if not isinstance(self.model_type_instance, TextEmbeddingModel):
            raise Exception("Model type instance is not TextEmbeddingModel")

        self.model_type_instance = cast(TextEmbeddingModel, self.model_type_instance)
        return cast(
            list[int],
            self._round_robin_invoke(
                function=self.model_type_instance.get_num_tokens,
                model=self.model,
                credentials=self.credentials,
                texts=texts,
            ),
        )

    def invoke_rerank(
            self,
            query: str,
            docs: list[str],
            score_threshold: Optional[float] = None,
            top_n: Optional[int] = None,
            user: Optional[str] = None,
    ) -> RerankResult:
        """
        执行文档重排序

        Args:
            query: 查询文本
            docs: 需要排序的文档列表
            score_threshold: 分数阈值，仅返回高于此值的文档（可选）
            top_n: 返回结果的数量限制（可选）
            user: 终端用户标识（可选）

        Returns:
            RerankResult: 包含排序结果和得分的对象

        Raises:
            Exception: 当模型类型不是RerankModel时抛出
        """
        if not isinstance(self.model_type_instance, RerankModel):
            raise Exception("Model type instance is not RerankModel")

        self.model_type_instance = cast(RerankModel, self.model_type_instance)
        return cast(
            RerankResult,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials,
                query=query,
                docs=docs,
                score_threshold=score_threshold,
                top_n=top_n,
                user=user,
            ),
        )
class ModelManager:
    """模型管理器，用于获取和管理模型实例"""

    def __init__(self) -> None:
        """初始化方法，创建提供者管理器实例"""
        self._provider_manager = ProviderManager()  # 提供商管理器实例，用于处理提供商相关操作

    def get_model_instance(self, tenant_id: str, provider: str, model_type: ModelType, model: str) -> ModelInstance:
        """
        根据参数获取指定模型实例

        :param tenant_id: 租户ID，用于隔离不同租户的模型配置
        :param provider: 提供商名称（如AWS、Azure等），为空时使用默认提供商
        :param model_type: 模型类型，用于区分不同用途的模型（如预测、分类等）
        :param model: 具体模型名称
        :return: 初始化配置好的模型实例
        """
        # 如果未指定提供商，则获取该类型模型的默认实例
        if not provider:
            return self.get_default_model_instance(tenant_id, model_type)

        # 从提供商管理器获取对应模型的配置包
        provider_model_bundle = self._provider_manager.get_provider_model_bundle(
            tenant_id=tenant_id,
            provider=provider,
            model_type=model_type
        )

        # 使用配置包和模型名称创建模型实例
        return ModelInstance(provider_model_bundle, model)

    def get_default_provider_model_name(self, tenant_id: str, model_type: ModelType) -> tuple[str | None, str | None]:
        """
        获取指定模型类型的默认提供商及首个模型名称

        :param tenant_id: 租户ID
        :param model_type: 模型类型
        :return: (提供商名称, 模型名称)元组，如果不存在则返回(None, None)
        """
        # 通过提供商管理器获取第一个可用提供商及其第一个模型
        return self._provider_manager.get_first_provider_first_model(tenant_id, model_type)

    def get_default_model_instance(self, tenant_id: str, model_type: ModelType) -> ModelInstance:
        """
        获取指定模型类型的默认模型实例

        :param tenant_id: 租户ID
        :param model_type: 模型类型
        :return: 初始化配置好的默认模型实例
        :raises ProviderTokenNotInitError: 当默认模型不存在时抛出异常
        """
        # 从提供商管理器获取默认模型实体
        default_model_entity = self._provider_manager.get_default_model(
            tenant_id=tenant_id,
            model_type=model_type
        )

        # 检查是否成功获取默认模型配置
        if not default_model_entity:
            raise ProviderTokenNotInitError(f"未找到 {model_type} 类型的默认模型配置")

        # 递归调用自身方法创建模型实例
        return self.get_model_instance(
            tenant_id=tenant_id,
            provider=default_model_entity.provider.provider,  # 从实体中提取提供商名称
            model_type=model_type,
            model=default_model_entity.model,  # 从实体中提取模型名称
        )


class LBModelManager:
    """负载均衡模型管理器，用于管理多个模型配置并实现轮询调度"""

    def __init__(
            self,
            tenant_id: str,
            provider: str,
            model_type: ModelType,
            model: str,
            load_balancing_configs: list[ModelLoadBalancingConfiguration],
            managed_credentials: Optional[dict] = None,
    ) -> None:
        """
        初始化负载均衡管理器

        :param tenant_id: 租户ID，用于隔离不同租户的配置
        :param provider: 服务提供商名称（如AWS、Azure等）
        :param model_type: 模型类型，用于区分不同用途的模型
        :param model: 具体模型名称
        :param load_balancing_configs: 负载均衡配置列表（包含多个模型部署配置）
        :param managed_credentials: 托管凭证，当配置名为__inherit__时使用的凭证
        """
        self._tenant_id = tenant_id
        self._provider = provider
        self._model_type = model_type
        self._model = model
        self._load_balancing_configs = load_balancing_configs

        # 处理特殊配置项 __inherit__
        for load_balancing_config in self._load_balancing_configs[:]:  # 使用列表浅拷贝进行遍历
            if load_balancing_config.name == "__inherit__":
                if not managed_credentials:
                    # 当没有托管凭证时移除该配置项
                    self._load_balancing_configs.remove(load_balancing_config)
                else:
                    # 注入托管凭证到配置中
                    load_balancing_config.credentials = managed_credentials

    def fetch_next(self) -> Optional[ModelLoadBalancingConfiguration]:
        """
        轮询获取下一个可用模型配置（轮询策略）

        实现特点：
        - 使用Redis维护全局索引计数器
        - 自动跳过处于冷却期的配置
        - 当所有配置都处于冷却时返回None

        :return: 可用的模型配置项，若无可用配置返回None
        """
        # 构造Redis缓存键（包含租户、提供商、模型类型和模型名称）
        cache_key = "model_lb_index:{}:{}:{}:{}".format(
            self._tenant_id, self._provider, self._model_type.value, self._model
        )

        cooldown_load_balancing_configs = []  # 记录处于冷却期的配置
        max_index = len(self._load_balancing_configs)

        while True:
            # 原子操作递增索引值（线程安全）
            current_index = redis_client.incr(cache_key)
            current_index = cast(int, current_index)

            # 防止索引值过大溢出
            if current_index >= 10000000:
                current_index = 1
                redis_client.set(cache_key, current_index)

            # 设置索引键的过期时间（1小时）
            redis_client.expire(cache_key, 3600)

            # 计算实际索引位置
            if current_index > max_index:
                current_index = current_index % max_index

            real_index = current_index - 1
            if real_index > max_index:
                real_index = 0

            # 获取当前轮询到的配置
            config: ModelLoadBalancingConfiguration = self._load_balancing_configs[real_index]

            if self.in_cooldown(config):
                # 记录冷却中的配置
                cooldown_load_balancing_configs.append(config)
                if len(cooldown_load_balancing_configs) >= len(self._load_balancing_configs):
                    # 全部配置都处于冷却期时返回空
                    return None
                continue  # 跳过冷却中的配置

            # 调试模式输出日志
            if dify_config.DEBUG:
                logger.info(
                    f"Model LB\nid: {config.id}\nname:{config.name}\n"
                    f"tenant_id: {self._tenant_id}\nprovider: {self._provider}\n"
                    f"model_type: {self._model_type.value}\nmodel: {self._model}"
                )

            return config

        return None

    def cooldown(self, config: ModelLoadBalancingConfiguration, expire: int = 60) -> None:
        """
        设置配置项的冷却时间（用于故障转移后暂时屏蔽问题节点）

        :param config: 需要冷却的配置项
        :param expire: 冷却持续时间（秒），默认60秒
        """
        # 构造冷却状态缓存键（包含配置ID）
        cooldown_cache_key = "model_lb_index:cooldown:{}:{}:{}:{}:{}".format(
            self._tenant_id, self._provider, self._model_type.value, self._model, config.id
        )

        # 设置带过期时间的冷却状态标记
        redis_client.setex(cooldown_cache_key, expire, "true")

    def in_cooldown(self, config: ModelLoadBalancingConfiguration) -> bool:
        """
        检查配置项是否处于冷却期

        :param config: 需要检查的配置项
        :return: True表示处于冷却期，False表示可用
        """
        # 构造冷却状态缓存键
        cooldown_cache_key = "model_lb_index:cooldown:{}:{}:{}:{}:{}".format(
            self._tenant_id, self._provider, self._model_type.value, self._model, config.id
        )

        # 检查Redis中是否存在冷却标记
        res: bool = redis_client.exists(cooldown_cache_key)
        return res

    @staticmethod
    def get_config_in_cooldown_and_ttl(
            tenant_id: str, provider: str, model_type: ModelType, model: str, config_id: str
    ) -> tuple[bool, int]:
        """
        获取配置项的冷却状态及剩余时间

        :param tenant_id: 租户ID
        :param provider: 服务提供商
        :param model_type: 模型类型
        :param model: 模型名称
        :param config_id: 配置项ID
        :return: (是否在冷却中, 剩余秒数)
        """
        # 构造冷却状态缓存键
        cooldown_cache_key = "model_lb_index:cooldown:{}:{}:{}:{}:{}".format(
            tenant_id, provider, model_type.value, model, config_id
        )

        # 获取剩余时间（TTL）
        ttl = redis_client.ttl(cooldown_cache_key)
        if ttl == -2:  # -2表示键不存在
            return False, 0

        ttl = cast(int, ttl)
        return True, ttl
