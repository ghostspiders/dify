import hashlib
import logging
import os
from collections.abc import Sequence
from threading import Lock
from typing import Optional

from pydantic import BaseModel

import contexts
from core.helper.position_helper import get_provider_position_map, sort_to_dict_by_position_map
from core.model_runtime.entities.model_entities import AIModelEntity, ModelType
from core.model_runtime.entities.provider_entities import ProviderConfig, ProviderEntity, SimpleProviderEntity
from core.model_runtime.model_providers.__base.ai_model import AIModel
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.moderation_model import ModerationModel
from core.model_runtime.model_providers.__base.rerank_model import RerankModel
from core.model_runtime.model_providers.__base.speech2text_model import Speech2TextModel
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.model_runtime.model_providers.__base.tts_model import TTSModel
from core.model_runtime.schema_validators.model_credential_schema_validator import ModelCredentialSchemaValidator
from core.model_runtime.schema_validators.provider_credential_schema_validator import ProviderCredentialSchemaValidator
from core.plugin.entities.plugin import ModelProviderID
from core.plugin.entities.plugin_daemon import PluginModelProviderEntity
from core.plugin.manager.asset import PluginAssetManager
from core.plugin.manager.model import PluginModelManager

logger = logging.getLogger(__name__)


class ModelProviderExtension(BaseModel):
    """
    模型提供商的扩展信息封装类（基于 Pydantic BaseModel）
    用于存储插件模型提供商实体及其排序位置
    """
    plugin_model_provider_entity: PluginModelProviderEntity  # 插件模型提供商实体对象
    position: Optional[int] = None  # 可选的位置序号（用于排序）


class ModelProviderFactory:
    """
    模型提供商工厂类
    核心功能：管理插件模型提供商，并按预设位置排序返回
    """
    provider_position_map: dict[str, int]  # 提供商名称到排序位置的映射字典

    def __init__(self, tenant_id: str) -> None:
        """
        初始化方法
        :param tenant_id: 租户ID（用于隔离不同租户的模型提供商）
        """
        self.provider_position_map = {}  # 初始化空位置映射
        self.tenant_id = tenant_id
        self.plugin_model_manager = PluginModelManager()  # 插件模型管理器实例

        # 如果位置映射未加载，则从配置文件读取
        if not self.provider_position_map:
            current_path = os.path.abspath(__file__)  # 获取当前文件绝对路径
            model_providers_path = os.path.dirname(current_path)  # 获取所在目录路径
            # 从_position.yaml加载提供商排序配置
            self.provider_position_map = get_provider_position_map(model_providers_path)

    def get_providers(self) -> Sequence[ProviderEntity]:
        """
        获取所有模型提供商（已按配置排序）
        :return: 排序后的提供商声明对象列表
        """
        # 1. 获取插件模型提供商原始数据
        plugin_providers = self.get_plugin_model_providers()

        # 2. 转换为ModelProviderExtension对象（附加排序能力）
        model_provider_extensions = [
            ModelProviderExtension(plugin_model_provider_entity=provider)
            for provider in plugin_providers
        ]

        # 3. 按预设位置排序
        sorted_extensions = sort_to_dict_by_position_map(
            position_map=self.provider_position_map,
            data=model_provider_extensions,
            name_func=lambda x: x.plugin_model_provider_entity.declaration.provider,
        )

        # 4. 提取并返回排序后的声明对象
        return [ext.plugin_model_provider_entity.declaration for ext in sorted_extensions.values()]

    def get_plugin_model_providers(self) -> Sequence[PluginModelProviderEntity]:
        """
        获取所有插件模型提供商原始数据（带线程安全缓存）
        :return: 插件模型提供商实体列表
        """
        # 检查线程上下文是否初始化
        try:
            contexts.plugin_model_providers.get()  # 尝试获取缓存
        except LookupError:
            # 初始化上下文缓存和锁
            contexts.plugin_model_providers.set(None)
            contexts.plugin_model_providers_lock.set(Lock())

        # 线程安全操作（防止多线程重复加载）
        with contexts.plugin_model_providers_lock.get():
            plugin_model_providers = contexts.plugin_model_providers.get()

            # 如果已有缓存直接返回
            if plugin_model_providers is not None:
                return plugin_model_providers

            # 初始化缓存列表
            plugin_model_providers = []
            contexts.plugin_model_providers.set(plugin_model_providers)

            # 从插件管理器获取原始数据
            plugin_providers = self.plugin_model_manager.fetch_model_providers(self.tenant_id)

            # 处理提供商名称（添加插件ID前缀）
            for provider in plugin_providers:
                provider.declaration.provider = f"{provider.plugin_id}/{provider.declaration.provider}"
                plugin_model_providers.append(provider)

            return plugin_model_providers

    def get_provider_schema(self, provider: str) -> ProviderEntity:
        """
        获取指定提供商的架构声明信息
        :param provider: 提供商名称（格式：插件ID/提供商名 或 内置提供商名）
        :return: 提供商的声明对象（包含能力定义、凭证规则等）
        """
        # 通过名称获取插件模型提供商实体，并返回其声明对象
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)
        return plugin_model_provider_entity.declaration

    def get_plugin_model_provider(self, provider: str) -> PluginModelProviderEntity:
        """
        根据名称获取具体的插件模型提供商实体
        :param provider: 提供商名称（自动处理内置提供商格式转换）
        :raises ValueError: 当提供商不存在时抛出
        :return: 插件模型提供商实体对象
        """
        # 如果名称中不包含"/"，则认为是内置提供商，转换为标准格式（如 "openai" -> "system/openai"）
        if "/" not in provider:
            provider = str(ModelProviderID(provider))  # 转换为系统预定义ID格式

        # 获取所有插件模型提供商列表（带缓存）
        plugin_model_provider_entities = self.get_plugin_model_providers()

        # 通过名称匹配目标提供商
        plugin_model_provider_entity = next(
            (p for p in plugin_model_provider_entities if p.declaration.provider == provider),
            None,
        )

        # 验证提供商是否存在
        if not plugin_model_provider_entity:
            raise ValueError(f"无效的提供商: {provider}")

        return plugin_model_provider_entity

    def provider_credentials_validate(self, *, provider: str, credentials: dict) -> dict:
        """
        验证并过滤提供商的凭证信息
        :param provider: 提供商名称
        :param credentials: 待验证的凭证字典（需符合provider_credential_schema定义）
        :raises ValueError: 当凭证验证失败时抛出
        :return: 过滤后的合法凭证字典
        """
        # 1. 获取提供商实体
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)

        # 2. 检查该提供商是否定义了凭证规则
        provider_credential_schema = plugin_model_provider_entity.declaration.provider_credential_schema
        if not provider_credential_schema:
            raise ValueError(f"提供商 {provider} 未定义凭证规则(provider_credential_schema)")

        # 3. 使用规则验证器处理凭证
        validator = ProviderCredentialSchemaValidator(provider_credential_schema)
        filtered_credentials = validator.validate_and_filter(credentials)  # 验证+过滤非法字段

        # 4. 通过插件管理器进行业务层验证（如检查API Key有效性）
        self.plugin_model_manager.validate_provider_credentials(
            tenant_id=self.tenant_id,
            user_id="unknown",  # 未指定用户时使用默认值
            plugin_id=plugin_model_provider_entity.plugin_id,
            provider=plugin_model_provider_entity.provider,
            credentials=filtered_credentials,
        )

        return filtered_credentials  # 返回处理后的安全凭证

    def model_credentials_validate(
            self, *, provider: str, model_type: ModelType, model: str, credentials: dict
    ) -> dict:
        """
        验证模型级别的凭证信息（如API Key、区域等）
        :param provider: 提供商名称（格式：插件ID/提供商名）
        :param model_type: 模型类型（如LLM、TEXT_EMBEDDING等）
        :param model: 具体模型名称（如"gpt-4"）
        :param credentials: 待验证的凭证字典
        :raises ValueError: 当凭证无效或规则未定义时抛出
        :return: 过滤后的安全凭证字典
        """
        # 1. 获取提供商实体
        plugin_model_provider_entity = self.get_plugin_model_provider(provider=provider)

        # 2. 检查模型凭证规则是否存在
        model_credential_schema = plugin_model_provider_entity.declaration.model_credential_schema
        if not model_credential_schema:
            raise ValueError(f"提供商 {provider} 未定义模型凭证规则(model_credential_schema)")

        # 3. 根据模型类型验证凭证格式
        validator = ModelCredentialSchemaValidator(model_type, model_credential_schema)
        filtered_credentials = validator.validate_and_filter(credentials)  # 过滤非法字段

        # 4. 调用插件管理器进行业务验证（如检查API Key有效性）
        self.plugin_model_manager.validate_model_credentials(
            tenant_id=self.tenant_id,
            user_id="unknown",  # 默认用户标识
            plugin_id=plugin_model_provider_entity.plugin_id,
            provider=plugin_model_provider_entity.provider,
            model_type=model_type.value,  # 枚举值转字符串
            model=model,
            credentials=filtered_credentials,
        )

        return filtered_credentials

    def get_model_schema(
            self, *, provider: str, model_type: ModelType, model: str, credentials: dict
    ) -> AIModelEntity | None:
        """
        获取模型架构定义（带缓存机制）
        :param provider: 提供商名称
        :param model_type: 模型类型
        :param model: 模型名称
        :param credentials: 凭证信息（用于缓存键生成）
        :return: 模型实体对象（包含输入输出定义等），不存在时返回None
        """
        # 1. 生成唯一缓存键（包含租户、插件、模型及凭证的MD5哈希）
        plugin_id, provider_name = self.get_plugin_id_and_provider_name_from_provider(provider)
        cache_key = f"{self.tenant_id}:{plugin_id}:{provider_name}:{model_type.value}:{model}"
        sorted_credentials = sorted(credentials.items()) if credentials else []
        cache_key += ":".join([hashlib.md5(f"{k}:{v}".encode()).hexdigest() for k, v in sorted_credentials])

        # 2. 初始化线程安全的缓存
        try:
            contexts.plugin_model_schemas.get()
        except LookupError:
            contexts.plugin_model_schemas.set({})
            contexts.plugin_model_schema_lock.set(Lock())

        # 3. 加锁访问缓存
        with contexts.plugin_model_schema_lock.get():
            cached_schemas = contexts.plugin_model_schemas.get()
            if cache_key in cached_schemas:
                return cached_schemas[cache_key]  # 缓存命中

            # 4. 从插件管理器获取模型架构
            schema = self.plugin_model_manager.get_model_schema(
                tenant_id=self.tenant_id,
                user_id="unknown",
                plugin_id=plugin_id,
                provider=provider_name,
                model_type=model_type.value,
                model=model,
                credentials=credentials or {},
            )

            # 5. 更新缓存
            if schema:
                cached_schemas[cache_key] = schema

            return schema

    def get_models(
            self,
            *,
            provider: Optional[str] = None,
            model_type: Optional[ModelType] = None,
            provider_configs: Optional[list[ProviderConfig]] = None,
    ) -> list[SimpleProviderEntity]:
        """
        获取符合条件的模型列表（支持按提供商和模型类型过滤）
        :param provider: 可选，指定提供商名称
        :param model_type: 可选，指定模型类型
        :param provider_configs: 可选，提供商配置列表（用于凭证预加载）
        :return: 简化版提供商实体列表（包含模型信息）
        """
        provider_configs = provider_configs or []
        provider_credentials_dict = {c.provider: c.credentials for c in provider_configs}

        # 遍历所有插件提供商
        providers = []
        for plugin_model_provider_entity in self.get_plugin_model_providers():
            # 按提供商名称过滤
            if provider and plugin_model_provider_entity.declaration.provider != provider:
                continue

            provider_schema = plugin_model_provider_entity.declaration

            # 按模型类型过滤
            model_types = provider_schema.supported_model_types
            if model_type and model_type not in model_types:
                continue
            model_types = [model_type] if model_type else model_types

            # 收集匹配的模型
            all_model_type_models = [
                model_schema
                for model_schema in provider_schema.models
                if not model_type or model_schema.model_type == model_type
            ]

            # 转换为简化版实体
            simple_provider_schema = provider_schema.to_simple_provider()
            simple_provider_schema.models.extend(all_model_type_models)
            providers.append(simple_provider_schema)

        return providers

    def get_model_type_instance(self, provider: str, model_type: ModelType) -> AIModel:
        """
        根据提供商和模型类型创建模型实例
        :param provider: 提供商名称
        :param model_type: 模型类型枚举
        :return: 具体模型类的实例（如LargeLanguageModel）
        """
        plugin_id, provider_name = self.get_plugin_id_and_provider_name_from_provider(provider)
        init_params = {
            "tenant_id": self.tenant_id,
            "plugin_id": plugin_id,
            "provider_name": provider_name,
            "plugin_model_provider": self.get_plugin_model_provider(provider),
        }

        # 根据模型类型返回具体实现类
        if model_type == ModelType.LLM:
            return LargeLanguageModel(**init_params)
        elif model_type == ModelType.TEXT_EMBEDDING:
            return TextEmbeddingModel(**init_params)
        # ...其他模型类型处理（略）

    def get_provider_icon(self, provider: str, icon_type: str, lang: str) -> tuple[bytes, str]:
        """
        获取提供商的图标（支持多语言）
        :param provider: 提供商名称
        :param icon_type: 图标类型（icon_small/icon_large）
        :param lang: 语言（zh_Hans/en_US）
        :return: (图标二进制数据, MIME类型)
        :raises ValueError: 当图标不存在时抛出
        """
        provider_schema = self.get_provider_schema(provider)
        icon_attr = getattr(provider_schema, f"icon_{icon_type.lower()}")

        # 验证图标是否存在
        if not icon_attr:
            raise ValueError(f"提供商 {provider} 没有{icon_type}图标")

        # 获取对应语言的图标文件名
        file_name = getattr(icon_attr, lang.lower(), None)
        if not file_name:
            raise ValueError(f"提供商 {provider} 的{icon_type}图标未支持语言 {lang}")

        # 根据文件扩展名确定MIME类型
        extension = file_name.split(".")[-1].lower()
        mime_type = {
            "jpg": "image/jpeg",
            "png": "image/png",
            # ...其他类型映射（略）
        }.get(extension, "image/png")

        # 从插件资源管理器获取图标数据
        plugin_asset_manager = PluginAssetManager()
        return plugin_asset_manager.fetch_asset(tenant_id=self.tenant_id, id=file_name), mime_type

    def get_plugin_id_and_provider_name_from_provider(self, provider: str) -> tuple[str, str]:
        """
        从提供商名称中提取插件ID和提供商名
        :param provider: 完整提供商名称（如"plugin1/gpt-4"）
        :return: (plugin_id, provider_name)
        """
        provider_id = ModelProviderID(provider)
        return provider_id.plugin_id, provider_id.provider_name