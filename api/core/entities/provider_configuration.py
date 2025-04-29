import datetime
import json
import logging
from collections import defaultdict
from collections.abc import Iterator, Sequence
from json import JSONDecodeError
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from constants import HIDDEN_VALUE
from core.entities.model_entities import ModelStatus, ModelWithProviderEntity, SimpleModelProviderEntity
from core.entities.provider_entities import (
    CustomConfiguration,
    ModelSettings,
    SystemConfiguration,
    SystemConfigurationStatus,
)
from core.helper import encrypter
from core.helper.model_provider_cache import ProviderCredentialsCache, ProviderCredentialsCacheType
from core.model_runtime.entities.model_entities import AIModelEntity, FetchFrom, ModelType
from core.model_runtime.entities.provider_entities import (
    ConfigurateMethod,
    CredentialFormSchema,
    FormType,
    ProviderEntity,
)
from core.model_runtime.model_providers.__base.ai_model import AIModel
from core.model_runtime.model_providers.model_provider_factory import ModelProviderFactory
from core.plugin.entities.plugin import ModelProviderID
from extensions.ext_database import db
from models.provider import (
    LoadBalancingModelConfig,
    Provider,
    ProviderModel,
    ProviderModelSetting,
    ProviderType,
    TenantPreferredModelProvider,
)

logger = logging.getLogger(__name__)

original_provider_configurate_methods: dict[str, list[ConfigurateMethod]] = {}


class ProviderConfiguration(BaseModel):
    """提供商配置数据模型，用于管理不同租户的模型服务配置"""

    # 基础配置字段
    tenant_id: str  # 租户唯一标识
    provider: ProviderEntity  # 提供商实体对象（包含提供商详细信息）
    preferred_provider_type: ProviderType  # 优先选择的提供商类型（用户预设）
    using_provider_type: ProviderType  # 当前实际使用的提供商类型
    system_configuration: SystemConfiguration  # 系统级配置（含配额限制等）
    custom_configuration: CustomConfiguration  # 自定义配置（用户个性化设置）
    model_settings: list[ModelSettings]  # 模型级别开关设置

    # Pydantic模型配置（禁用命名空间保护）
    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **data):
        """初始化方法，处理提供商配置方法逻辑"""
        super().__init__(**data)

        # 维护原始配置方法集合
        if self.provider.provider not in original_provider_configurate_methods:
            original_provider_configurate_methods[self.provider.provider] = []

        # 备份提供商的初始配置方法
        for configurate_method in self.provider.configurate_methods:
            original_provider_configurate_methods[self.provider.provider].append(configurate_method)

        # 自动添加预定义模型配置方法（当存在模型限制且未配置时）
        if original_provider_configurate_methods[self.provider.provider] == [ConfigurateMethod.CUSTOMIZABLE_MODEL]:
            if (
                    any(
                        len(quota_configuration.restrict_models) > 0
                        for quota_configuration in self.system_configuration.quota_configurations
                    )
                    and ConfigurateMethod.PREDEFINED_MODEL not in self.provider.configurate_methods
            ):
                self.provider.configurate_methods.append(ConfigurateMethod.PREDEFINED_MODEL)

    def get_current_credentials(self, model_type: ModelType, model: str) -> Optional[dict]:
        """
        获取当前可用的认证凭证（自动合并系统配置与自定义配置）

        :param model_type: 模型类型（如文本生成/图像识别）
        :param model: 具体模型名称
        :return: 合并后的凭证字典
        :raises ValueError: 当模型被管理员禁用时抛出
        """
        # 检查模型启用状态
        if self.model_settings:
            for model_setting in self.model_settings:
                if model_setting.model_type == model_type and model_setting.model == model:
                    if not model_setting.enabled:
                        raise ValueError(f"模型 {model} 已被管理员禁用")

        # 系统配置模式处理逻辑
        if self.using_provider_type == ProviderType.SYSTEM:
            restrict_models = []
            # 匹配当前配额类型的配置
            for quota_configuration in self.system_configuration.quota_configurations:
                if self.system_configuration.current_quota_type != quota_configuration.quota_type:
                    continue

                restrict_models = quota_configuration.restrict_models

            # 深拷贝凭证防止污染原始数据
            copy_credentials = (
                self.system_configuration.credentials.copy()
                if self.system_configuration.credentials
                else {}
            )

            # 注入基础模型名称（当存在限制时）
            if restrict_models:
                for restrict_model in restrict_models:
                    if (
                            restrict_model.model_type == model_type
                            and restrict_model.model == model
                            and restrict_model.base_model_name
                    ):
                        copy_credentials["base_model_name"] = restrict_model.base_model_name
            return copy_credentials
        # 自定义配置模式处理逻辑
        else:
            credentials = None
            # 优先查找模型级配置
            if self.custom_configuration.models:
                for model_configuration in self.custom_configuration.models:
                    if model_configuration.model_type == model_type and model_configuration.model == model:
                        credentials = model_configuration.credentials
                        break

            # 回退到提供商级配置
            if not credentials and self.custom_configuration.provider:
                credentials = self.custom_configuration.provider.credentials

            return credentials

    def get_system_configuration_status(self) -> Optional[SystemConfigurationStatus]:
        """
        获取系统配置状态（用于监控配额使用情况）

        :return: 系统配置状态枚举值
        """
        if self.system_configuration.enabled is False:
            return SystemConfigurationStatus.UNSUPPORTED  # 系统配置未启用

        # 获取当前配额配置
        current_quota_type = self.system_configuration.current_quota_type
        current_quota_configuration = next(
            (q for q in self.system_configuration.quota_configurations if q.quota_type == current_quota_type),
            None
        )

        if current_quota_configuration is None:
            return None  # 无对应配额配置

        return (
            SystemConfigurationStatus.ACTIVE  # 配额有效
            if current_quota_configuration.is_valid
            else SystemConfigurationStatus.QUOTA_EXCEEDED  # 配额超限
        )

    def is_custom_configuration_available(self) -> bool:
        """
        检查是否存在有效的自定义配置

        :return: True表示存在可用自定义配置
        """
        # 判断标准：存在提供商级配置或至少一个模型级配置
        return self.custom_configuration.provider is not None or len(self.custom_configuration.models) > 0

    def get_custom_credentials(self, obfuscated: bool = False) -> dict | None:
        """
        获取自定义凭证信息

        :param obfuscated: 是否对敏感数据进行脱敏处理
        :return: 凭证字典（可能包含脱敏数据）或 None（当无自定义配置时）
        """
        # 检查是否存在自定义配置的模型提供商
        if self.custom_configuration.provider is None:
            return None

        # 获取原始凭证信息
        credentials = self.custom_configuration.provider.credentials
        if not obfuscated:
            return credentials  # 直接返回未脱敏的凭证

        # 当需要脱敏时，调用脱敏处理方法
        return self.obfuscated_credentials(
            credentials=credentials,
            # 获取凭证表单结构的模式定义，用于确定需要脱敏的字段
            credential_form_schemas=self.provider.provider_credential_schema.credential_form_schemas
            if self.provider.provider_credential_schema
            else [],
        )

    def _get_custom_provider_credentials(self) -> Provider | None:
        """
        从数据库获取自定义模型提供商的凭证记录

        :return: Provider 对象或 None（未找到记录时）
        """
        # 确定提供商名称列表（处理 langgenius 的特殊情况）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 查询数据库获取提供商记录
        provider_record = (
            db.session.query(Provider)
            .filter(
                Provider.tenant_id == self.tenant_id,  # 租户隔离
                Provider.provider_type == ProviderType.CUSTOM.value,  # 仅限自定义类型
                Provider.provider_name.in_(provider_names),  # 名称匹配
            )
            .first()
        )

        return provider_record

    def custom_credentials_validate(self, credentials: dict) -> tuple[Provider | None, dict]:
        """
        验证并处理自定义凭证

        :param credentials: 待验证的凭证字典
        :return: 包含 Provider 对象和处理后凭证的元组
        """
        # 获取关联的提供商记录
        provider_record = self._get_custom_provider_credentials()

        # 提取需要加密处理的敏感字段名列表
        provider_credential_secret_variables = self.extract_secret_variables(
            self.provider.provider_credential_schema.credential_form_schemas
            if self.provider.provider_credential_schema
            else []
        )

        # 处理已有凭证的加密配置
        if provider_record:
            try:
                # 处理旧版非 JSON 格式的配置（兼容性处理）
                if provider_record.encrypted_config:
                    if not provider_record.encrypted_config.startswith("{"):
                        original_credentials = {"openai_api_key": provider_record.encrypted_config}
                    else:
                        original_credentials = json.loads(provider_record.encrypted_config)
                else:
                    original_credentials = {}
            except JSONDecodeError:
                # JSON 解析失败时回退空字典
                original_credentials = {}

        # 加密处理敏感字段
        for key, value in credentials.items():
            if key in provider_credential_secret_variables:
                # 当收到隐藏占位符时，保留原始加密值
                if value == HIDDEN_VALUE and key in original_credentials:
                    credentials[key] = encrypter.decrypt_token(self.tenant_id, original_credentials[key])

        # 使用工厂模式进行凭证验证
        model_provider_factory = ModelProviderFactory(self.tenant_id)
        credentials = model_provider_factory.provider_credentials_validate(
            provider=self.provider.provider,
            credentials=credentials
        )

        # 对敏感字段进行加密处理
        for key, value in credentials.items():
            if key in provider_credential_secret_variables:
                credentials[key] = encrypter.encrypt_token(self.tenant_id, value)

        return provider_record, credentials

    def add_or_update_custom_credentials(self, credentials: dict) -> None:
        """
        添加或更新自定义供应商的凭证配置

        参数:
            credentials: 包含供应商凭证配置的字典

        流程:
            1. 验证凭证配置有效性
            2. 如果存在现有记录则更新，否则创建新记录
            3. 保存到数据库并刷新更新时间
            4. 清理凭证缓存
            5. 切换首选供应商类型为自定义
        """
        # 验证自定义供应商配置格式
        provider_record, credentials = self.custom_credentials_validate(credentials)

        # 保存供应商信息（优先使用配额供应商，不自动切换首选供应商）
        if provider_record:
            # 更新现有记录
            provider_record.encrypted_config = json.dumps(credentials)  # 序列化配置信息
            provider_record.is_valid = True  # 标记配置有效
            provider_record.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)  # 更新时间戳
            db.session.commit()  # 提交数据库变更
        else:
            # 创建新供应商记录
            provider_record = Provider()
            provider_record.tenant_id = self.tenant_id  # 租户ID
            provider_record.provider_name = self.provider.provider  # 供应商名称
            provider_record.provider_type = ProviderType.CUSTOM.value  # 设置为自定义类型
            provider_record.encrypted_config = json.dumps(credentials)  # 加密存储配置
            provider_record.is_valid = True  # 标记为有效配置

            db.session.add(provider_record)  # 添加新记录
            db.session.commit()  # 提交数据库

        # 清理凭证缓存
        ProviderCredentialsCache(
            tenant_id=self.tenant_id,
            identity_id=provider_record.id,
            cache_type=ProviderCredentialsCacheType.PROVIDER
        ).delete()

        # 切换首选供应商为自定义类型
        self.switch_preferred_provider_type(ProviderType.CUSTOM)

    def delete_custom_credentials(self) -> None:
        """
        删除自定义供应商凭证配置

        流程:
            1. 获取现有自定义供应商配置
            2. 切换回系统默认供应商
            3. 删除数据库记录
            4. 清理相关缓存
        """
        # 查询现有自定义供应商配置
        provider_record = self._get_custom_provider_credentials()

        if provider_record:
            # 切换回系统供应商
            self.switch_preferred_provider_type(ProviderType.SYSTEM)

            # 删除数据库记录
            db.session.delete(provider_record)
            db.session.commit()  # 提交删除操作

            # 清理关联缓存
            ProviderCredentialsCache(
                tenant_id=self.tenant_id,
                identity_id=provider_record.id,
                cache_type=ProviderCredentialsCacheType.PROVIDER,
            ).delete()

    def get_custom_model_credentials(
            self, model_type: ModelType, model: str, obfuscated: bool = False
    ) -> Optional[dict]:
        """
        获取指定AI模型的凭证配置（支持脱敏处理）

        参数:
            model_type: 模型类型 (枚举)
            model: 模型名称
            obfuscated: 是否对敏感信息脱敏

        返回:
            dict: 包含凭证配置的字典（找不到返回None）
        """
        # 检查是否存在自定义模型配置
        if not self.custom_configuration.models:
            return None

        # 遍历模型配置列表
        for model_configuration in self.custom_configuration.models:
            # 匹配模型类型和名称
            if model_configuration.model_type == model_type and model_configuration.model == model:
                credentials = model_configuration.credentials

                # 直接返回原始凭证
                if not obfuscated:
                    return credentials

                # 对敏感字段进行脱敏处理
                return self.obfuscated_credentials(
                    credentials=credentials,
                    # 使用凭证模式定义中的脱敏规则
                    credential_form_schemas=self.provider.model_credential_schema.credential_form_schemas
                    if self.provider.model_credential_schema  # 检查是否存在模式定义
                    else [],  # 无模式则返回空列表
                )

        return None

    def _get_custom_model_credentials(
            self,
            model_type: ModelType,
            model: str,
    ) -> ProviderModel | None:
        """
        获取指定模型的自定义凭证配置记录

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModel: 数据库中的模型凭证记录，找不到返回None

        流程:
            1. 构建提供商标识（兼容LangGenius的特殊处理）
            2. 查询数据库获取匹配租户+提供商+模型类型+模型名称的记录
        """
        # 构建提供商标识（兼容LangGenius的特殊情况）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]

        # 如果是LangGenius提供商需要同时查询原始名称
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 执行数据库查询
        provider_model_record = (
            db.session.query(ProviderModel)
            .filter(
                ProviderModel.tenant_id == self.tenant_id,
                ProviderModel.provider_name.in_(provider_names),  # 匹配多个提供商名称
                ProviderModel.model_name == model,
                ProviderModel.model_type == model_type.to_origin_model_type(),  # 转换为底层模型类型
            )
            .first()
        )

        return provider_model_record

    def custom_model_credentials_validate(
            self, model_type: ModelType, model: str, credentials: dict
    ) -> tuple[ProviderModel | None, dict]:
        """
        验证并处理自定义模型凭证

        参数:
            model_type: 模型类型枚举
            model: 模型名称
            credentials: 待验证的凭证字典

        返回:
            tuple: (数据库记录对象, 处理后的凭证字典)

        流程:
            1. 获取现有数据库记录
            2. 提取需要加密的敏感字段定义
            3. 处理隐藏值保持原值逻辑（当输入__HIDDEN__时）
            4. 执行模型凭证验证
            5. 对敏感字段进行加密处理
        """
        # 获取现有模型配置记录
        provider_model_record = self._get_custom_model_credentials(model_type, model)

        # 提取凭证中的敏感字段定义（用于后续加解密处理）
        provider_credential_secret_variables = self.extract_secret_variables(
            self.provider.model_credential_schema.credential_form_schemas
            if self.provider.model_credential_schema  # 检查是否存在凭证模式定义
            else []
        )

        if provider_model_record:
            try:
                # 解析现有的加密配置（处理可能的空值情况）
                original_credentials = (
                    json.loads(provider_model_record.encrypted_config)
                    if provider_model_record.encrypted_config
                    else {}
                )
            except JSONDecodeError:
                original_credentials = {}

            # 处理加密字段的隐藏值逻辑
            for key, value in credentials.items():
                if key in provider_credential_secret_variables:
                    # 当收到隐藏标记时保持原有加密值
                    if value == HIDDEN_VALUE and key in original_credentials:
                        credentials[key] = encrypter.decrypt_token(
                            self.tenant_id,
                            original_credentials[key]  # 使用原始加密值解密
                        )

        # 调用工厂方法进行凭证验证（验证业务逻辑）
        model_provider_factory = ModelProviderFactory(self.tenant_id)
        credentials = model_provider_factory.model_credentials_validate(
            provider=self.provider.provider,
            model_type=model_type,
            model=model,
            credentials=credentials
        )

        # 对敏感字段进行加密处理
        for key, value in credentials.items():
            if key in provider_credential_secret_variables:
                credentials[key] = encrypter.encrypt_token(self.tenant_id, value)

        return provider_model_record, credentials

    def add_or_update_custom_model_credentials(self, model_type: ModelType, model: str, credentials: dict) -> None:
        """
        添加/更新自定义模型凭证配置

        参数:
            model_type: 模型类型枚举
            model: 模型名称
            credentials: 凭证配置字典

        流程:
            1. 验证凭证有效性并处理敏感数据
            2. 更新现有记录或创建新记录
            3. 更新数据库时间戳
            4. 清理相关缓存
            5. 注意保持原有配额优先策略（不自动切换首选提供商）
        """
        # 验证凭证并获取处理后的数据
        provider_model_record, credentials = self.custom_model_credentials_validate(model_type, model, credentials)

        # 保存到数据库
        if provider_model_record:
            # 更新现有记录
            provider_model_record.encrypted_config = json.dumps(credentials)  # 序列化配置
            provider_model_record.is_valid = True  # 标记为有效配置
            provider_model_record.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)  # 更新时间戳
            db.session.commit()  # 提交更新
        else:
            # 创建新记录
            provider_model_record = ProviderModel()
            provider_model_record.tenant_id = self.tenant_id  # 租户ID
            provider_model_record.provider_name = self.provider.provider  # 提供商名称
            provider_model_record.model_name = model  # 模型名称
            provider_model_record.model_type = model_type.to_origin_model_type()  # 转换存储类型
            provider_model_record.encrypted_config = json.dumps(credentials)  # 加密存储配置
            provider_model_record.is_valid = True  # 标记有效

            db.session.add(provider_model_record)  # 添加新记录
            db.session.commit()  # 提交创建

        # 清理模型凭证缓存（保证下次获取最新配置）
        provider_model_credentials_cache = ProviderCredentialsCache(
            tenant_id=self.tenant_id,
            identity_id=provider_model_record.id,
            cache_type=ProviderCredentialsCacheType.MODEL,  # 模型类型缓存
        )
        provider_model_credentials_cache.delete()

    def delete_custom_model_credentials(self, model_type: ModelType, model: str) -> None:
        """
        删除指定模型的自定义凭证配置

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        流程:
            1. 查询现有模型配置记录
            2. 删除数据库记录
            3. 清理关联的凭证缓存
        """
        # 获取现有配置记录
        provider_model_record = self._get_custom_model_credentials(model_type, model)

        if provider_model_record:
            # 执行数据库删除
            db.session.delete(provider_model_record)
            db.session.commit()  # 立即提交事务

            # 清理模型凭证缓存
            ProviderCredentialsCache(
                tenant_id=self.tenant_id,
                identity_id=provider_model_record.id,
                cache_type=ProviderCredentialsCacheType.MODEL,  # 指定模型类型缓存
            ).delete()

    def _get_provider_model_setting(self, model_type: ModelType, model: str) -> ProviderModelSetting | None:
        """
        获取模型的启用/禁用状态配置（内部方法）

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 模型状态配置对象，找不到返回None

        特殊处理:
            - 兼容LangGenius提供商的名称查询
            - 转换模型类型到原始存储格式
        """
        # 构建提供商标识列表（兼容LangGenius）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 查询数据库配置
        return (
            db.session.query(ProviderModelSetting)
            .filter(
                ProviderModelSetting.tenant_id == self.tenant_id,  # 租户匹配
                ProviderModelSetting.provider_name.in_(provider_names),  # 多提供商查询
                ProviderModelSetting.model_type == model_type.to_origin_model_type(),  # 类型转换
                ProviderModelSetting.model_name == model,  # 精确匹配模型名称
            )
            .first()
        )

    def enable_model(self, model_type: ModelType, model: str) -> ProviderModelSetting:
        """
        启用指定模型

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 更新后的配置对象

        流程:
            1. 查找现有配置，存在则更新启用状态
            2. 不存在则创建新配置记录
            3. 更新/创建时间戳
        """
        model_setting = self._get_provider_model_setting(model_type, model)

        if model_setting:
            # 更新现有记录
            model_setting.enabled = True  # 设置启用标志
            model_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)  # UTC时间戳
            db.session.commit()  # 提交更新
        else:
            # 创建新配置记录
            model_setting = ProviderModelSetting()
            model_setting.tenant_id = self.tenant_id
            model_setting.provider_name = self.provider.provider
            model_setting.model_type = model_type.to_origin_model_type()  # 转换存储格式
            model_setting.model_name = model
            model_setting.enabled = True  # 默认启用
            db.session.add(model_setting)
            db.session.commit()  # 提交创建

        return model_setting

    def disable_model(self, model_type: ModelType, model: str) -> ProviderModelSetting:
        """
        禁用指定模型

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 更新后的配置对象

        注意:
            - 与enable_model逻辑对称，但设置enabled=False
            - 保持更新时间戳逻辑一致
        """
        model_setting = self._get_provider_model_setting(model_type, model)

        if model_setting:
            # 更新现有记录
            model_setting.enabled = False  # 设置禁用标志
            model_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            db.session.commit()
        else:
            # 创建禁用状态的记录
            model_setting = ProviderModelSetting()
            model_setting.tenant_id = self.tenant_id
            model_setting.provider_name = self.provider.provider
            model_setting.model_type = model_type.to_origin_model_type()
            model_setting.model_name = model
            model_setting.enabled = False  # 初始状态为禁用
            db.session.add(model_setting)
            db.session.commit()

        return model_setting

    def get_provider_model_setting(self, model_type: ModelType, model: str) -> Optional[ProviderModelSetting]:
        """
        获取模型的启用状态配置

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 配置对象或None
        """
        return self._get_provider_model_setting(model_type, model)

    def _get_load_balancing_config(self, model_type: ModelType, model: str) -> Optional[LoadBalancingModelConfig]:
        """
        获取模型的负载均衡配置（内部方法）

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            LoadBalancingModelConfig: 负载均衡配置对象或None

        特殊处理:
            - 兼容LangGenius提供商的名称查询
            - 转换模型类型到存储格式
        """
        # 构建提供商标识列表
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 查询负载均衡配置
        return (
            db.session.query(LoadBalancingModelConfig)
            .filter(
                LoadBalancingModelConfig.tenant_id == self.tenant_id,
                LoadBalancingModelConfig.provider_name.in_(provider_names),  # 多提供商匹配
                LoadBalancingModelConfig.model_type == model_type.to_origin_model_type(),  # 类型转换
                LoadBalancingModelConfig.model_name == model,  # 精确匹配模型
            )
            .first()
        )

    def enable_model_load_balancing(self, model_type: ModelType, model: str) -> ProviderModelSetting:
        """
        启用指定模型的负载均衡功能

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 更新后的模型配置对象

        流程:
            1. 验证负载均衡配置有效性（至少需要2个配置）
            2. 更新或创建模型配置记录
            3. 设置负载均衡启用标志
            4. 刷新更新时间戳

        异常:
            ValueError: 当负载均衡配置不足时抛出
        """
        # 构建提供商标识列表（兼容LangGenius特殊处理）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 检查负载均衡配置数量（必须>1个配置才能启用）
        load_balancing_config_count = (
            db.session.query(LoadBalancingModelConfig)
            .filter(
                LoadBalancingModelConfig.tenant_id == self.tenant_id,
                LoadBalancingModelConfig.provider_name.in_(provider_names),
                LoadBalancingModelConfig.model_type == model_type.to_origin_model_type(),  # 转换存储格式
                LoadBalancingModelConfig.model_name == model
            )
            .count()
        )

        if load_balancing_config_count <= 1:
            raise ValueError("负载均衡配置必须包含至少2个可用配置")

        # 获取或创建模型配置记录
        model_setting = self._get_provider_model_setting(model_type, model)

        if model_setting:
            # 更新现有配置
            model_setting.load_balancing_enabled = True  # 启用负载均衡标志
            model_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)  # UTC时间戳
            db.session.commit()
        else:
            # 创建新配置记录
            model_setting = ProviderModelSetting()
            model_setting.tenant_id = self.tenant_id
            model_setting.provider_name = self.provider.provider
            model_setting.model_type = model_type.to_origin_model_type()
            model_setting.model_name = model
            model_setting.load_balancing_enabled = True  # 初始启用状态
            db.session.add(model_setting)
            db.session.commit()

        return model_setting

    def disable_model_load_balancing(self, model_type: ModelType, model: str) -> ProviderModelSetting:
        """
        禁用指定模型的负载均衡功能

        参数:
            model_type: 模型类型枚举
            model: 模型名称

        返回:
            ProviderModelSetting: 更新后的模型配置对象

        流程:
            1. 查询现有模型配置
            2. 设置负载均衡禁用标志
            3. 更新数据库记录时间戳
        """
        # 构建提供商标识列表（兼容LangGenius）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 获取模型配置记录
        model_setting = (
            db.session.query(ProviderModelSetting)
            .filter(
                ProviderModelSetting.tenant_id == self.tenant_id,
                ProviderModelSetting.provider_name.in_(provider_names),
                ProviderModelSetting.model_type == model_type.to_origin_model_type(),
                ProviderModelSetting.model_name == model
            )
            .first()
        )

        if model_setting:
            # 更新现有记录
            model_setting.load_balancing_enabled = False  # 关闭负载均衡
            model_setting.updated_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
            db.session.commit()
        else:
            # 创建禁用状态的配置记录（保持数据完整性）
            model_setting = ProviderModelSetting()
            model_setting.tenant_id = self.tenant_id
            model_setting.provider_name = self.provider.provider
            model_setting.model_type = model_type.to_origin_model_type()
            model_setting.model_name = model
            model_setting.load_balancing_enabled = False  # 初始禁用状态
            db.session.add(model_setting)
            db.session.commit()

        return model_setting

    def get_model_type_instance(self, model_type: ModelType) -> AIModel:
        """
        获取指定模型类型的实例对象

        参数:
            model_type: 模型类型枚举

        返回:
            AIModel: 模型实例对象

        说明:
            - 通过工厂模式获取标准化的模型实例
            - 用于执行模型相关操作（推理、评估等）
        """
        # 使用工厂模式获取模型实例
        model_provider_factory = ModelProviderFactory(self.tenant_id)
        return model_provider_factory.get_model_type_instance(
            provider=self.provider.provider,
            model_type=model_type
        )

    def get_model_schema(self, model_type: ModelType, model: str, credentials: dict) -> AIModelEntity | None:
        """
        获取指定模型的结构定义

        参数:
            model_type: 模型类型枚举
            model: 模型名称
            credentials: 模型凭证配置

        返回:
            AIModelEntity: 包含模型结构信息的实体对象
            None: 当模型不存在时返回

        作用:
            - 获取模型输入/输出格式
            - 验证模型参数合法性
            - 获取模型能力描述
        """
        # 通过工厂模式获取模型结构定义
        model_provider_factory = ModelProviderFactory(self.tenant_id)
        return model_provider_factory.get_model_schema(
            provider=self.provider.provider,
            model_type=model_type,
            model=model,
            credentials=credentials  # 需要有效的凭证配置
        )

    def switch_preferred_provider_type(self, provider_type: ProviderType) -> None:
        """
        切换首选模型提供商类型

        参数:
            provider_type: 提供商类型枚举值

        流程:
            1. 验证当前类型无需切换
            2. 系统类型提供商需要验证可用性
            3. 处理LangGenius特殊提供商标识
            4. 更新或创建数据库记录
            5. 提交事务保证数据持久化

        注意:
            - 当切换为SYSTEM类型时需确保系统配置已启用
            - 兼容LangGenius提供商的双名称查询逻辑
        """
        # 无需切换的情况直接返回
        if provider_type == self.preferred_provider_type:
            return

        # 系统类型提供商需要验证可用性
        if provider_type == ProviderType.SYSTEM and not self.system_configuration.enabled:
            return

        # 构建提供商标识列表（兼容LangGenius）
        model_provider_id = ModelProviderID(self.provider.provider)
        provider_names = [self.provider.provider]
        if model_provider_id.is_langgenius():
            provider_names.append(model_provider_id.provider_name)

        # 查询现有首选项记录
        preferred_model_provider = (
            db.session.query(TenantPreferredModelProvider)
            .filter(
                TenantPreferredModelProvider.tenant_id == self.tenant_id,
                TenantPreferredModelProvider.provider_name.in_(provider_names),
            )
            .first()
        )

        if preferred_model_provider:
            # 更新现有记录
            preferred_model_provider.preferred_provider_type = provider_type.value
        else:
            # 创建新记录
            preferred_model_provider = TenantPreferredModelProvider()
            preferred_model_provider.tenant_id = self.tenant_id
            preferred_model_provider.provider_name = self.provider.provider
            preferred_model_provider.preferred_provider_type = provider_type.value
            db.session.add(preferred_model_provider)

        db.session.commit()  # 立即提交事务

    def extract_secret_variables(self, credential_form_schemas: list[CredentialFormSchema]) -> list[str]:
        """
        从凭证表单结构中提取敏感字段标识

        参数:
            credential_form_schemas: 凭证表单结构定义列表

        返回:
            list[str]: 需要加密处理的字段名列表

        说明:
            - 识别所有类型为SECRET_INPUT的表单字段
            - 用于后续的加密/脱敏处理
        """
        secret_input_form_variables = []
        for credential_form_schema in credential_form_schemas:
            # 筛选密文输入类型的字段
            if credential_form_schema.type == FormType.SECRET_INPUT:
                secret_input_form_variables.append(credential_form_schema.variable)

        return secret_input_form_variables

    def obfuscated_credentials(self, credentials: dict, credential_form_schemas: list[CredentialFormSchema]) -> dict:
        """
        对凭证中的敏感字段进行脱敏处理

        参数:
            credentials: 原始凭证字典
            credential_form_schemas: 凭证表单结构定义

        返回:
            dict: 脱敏后的凭证字典

        安全措施:
            - 创建数据副本避免修改原始数据
            - 使用加密模块的标准脱敏方法
            - 仅处理预定义的敏感字段
        """
        # 获取需要脱敏的字段列表
        credential_secret_variables = self.extract_secret_variables(credential_form_schemas)

        # 创建副本进行脱敏处理
        copy_credentials = credentials.copy()
        for key, value in copy_credentials.items():
            if key in credential_secret_variables:
                # 调用加密模块进行标准脱敏
                copy_credentials[key] = encrypter.obfuscated_token(value)

        return copy_credentials

    def get_provider_model(
            self, model_type: ModelType, model: str, only_active: bool = False
    ) -> Optional[ModelWithProviderEntity]:
        """
        获取指定模型的提供商实体信息

        参数:
            model_type: 模型类型枚举
            model: 模型名称
            only_active: 是否仅返回激活状态的模型

        返回:
            ModelWithProviderEntity: 包含提供商信息的模型实体
            None: 未找到匹配模型时返回

        流程:
            1. 获取当前提供商的所有模型列表
            2. 根据过滤条件筛选匹配项
            3. 返回第一个匹配的模型实体
        """
        # 获取模型列表（可能过滤非活跃模型）
        provider_models = self.get_provider_models(model_type, only_active)

        # 线性搜索匹配模型名称
        for provider_model in provider_models:
            if provider_model.model == model:
                return provider_model

        return None

    def get_provider_models(
            self, model_type: Optional[ModelType] = None, only_active: bool = False
    ) -> list[ModelWithProviderEntity]:
        """
        获取当前提供商的所有可用模型列表

        参数:
            model_type: 指定过滤的模型类型（可选）
            only_active: 是否只返回已启用的活跃模型

        返回:
            list[ModelWithProviderEntity]: 包含提供商信息的模型实体列表

        流程:
            1. 初始化模型提供商工厂获取提供商元数据
            2. 确定需要处理的模型类型范围
            3. 构建模型配置的快速查找映射表
            4. 根据提供商类型获取系统/自定义模型
            5. 应用活跃模型过滤条件
            6. 按模型类型排序返回结果
        """
        # 初始化模型提供商工厂
        model_provider_factory = ModelProviderFactory(self.tenant_id)
        # 获取当前提供商的结构定义
        provider_schema = model_provider_factory.get_provider_schema(self.provider.provider)

        # 确定目标模型类型集合
        model_types: list[ModelType] = []
        if model_type:
            model_types.append(model_type)  # 指定单个模型类型
        else:
            # 获取提供商支持的所有模型类型
            model_types = list(provider_schema.supported_model_types)

        # 构建模型配置的快速查找映射表 {模型类型: {模型名称: 配置}}
        model_setting_map: defaultdict[ModelType, dict[str, ModelSettings]] = defaultdict(dict)
        for model_setting in self.model_settings:
            model_setting_map[model_setting.model_type][model_setting.model] = model_setting

        # 根据提供商类型选择获取方式
        if self.using_provider_type == ProviderType.SYSTEM:
            provider_models = self._get_system_provider_models(
                model_types=model_types,
                provider_schema=provider_schema,
                model_setting_map=model_setting_map
            )
        else:
            provider_models = self._get_custom_provider_models(
                model_types=model_types,
                provider_schema=provider_schema,
                model_setting_map=model_setting_map
            )

        # 应用活跃模型过滤
        if only_active:
            provider_models = [m for m in provider_models if m.status == ModelStatus.ACTIVE]

        # 按模型类型值排序后返回
        return sorted(provider_models, key=lambda x: x.model_type.value)

    def _get_system_provider_models(
            self,
            model_types: Sequence[ModelType],
            provider_schema: ProviderEntity,
            model_setting_map: dict[ModelType, dict[str, ModelSettings]],
    ) -> list[ModelWithProviderEntity]:
        """
        获取系统提供商的模型列表（内部方法）

        参数:
            model_types: 需要处理的模型类型集合
            provider_schema: 提供商结构定义
            model_setting_map: 模型配置映射表

        返回:
            list[ModelWithProviderEntity]: 系统模型实体列表

        处理逻辑:
            1. 遍历所有目标模型类型
            2. 匹配提供商支持的模型定义
            3. 根据配置确定模型启用状态
            4. 处理系统配额限制的特殊情况
            5. 尝试获取自定义模型结构
        """
        provider_models = []

        # 第一阶段：基础模型处理
        for model_type in model_types:
            # 遍历提供商定义的所有模型
            for m in provider_schema.models:
                if m.model_type != model_type:
                    continue  # 类型不匹配则跳过

                # 默认状态为活跃
                status = ModelStatus.ACTIVE
                # 检查是否存在禁用配置
                if m.model in model_setting_map.get(m.model_type, {}):
                    model_setting = model_setting_map[m.model_type][m.model]
                    if model_setting.enabled is False:
                        status = ModelStatus.DISABLED  # 标记为已禁用

                # 构建模型实体对象
                provider_models.append(
                    ModelWithProviderEntity(
                        model=m.model,
                        label=m.label,
                        model_type=m.model_type,
                        features=m.features,
                        fetch_from=m.fetch_from,
                        model_properties=m.model_properties,
                        deprecated=m.deprecated,
                        provider=SimpleModelProviderEntity(self.provider),
                        status=status,
                    )
                )

        # 第二阶段：处理配置方法
        # 维护全局配置方法缓存
        if self.provider.provider not in original_provider_configurate_methods:
            original_provider_configurate_methods[self.provider.provider] = []

        # 缓存当前提供商的配置方法
        for configurate_method in provider_schema.configurate_methods:
            if configurate_method not in original_provider_configurate_methods[self.provider.provider]:
                original_provider_configurate_methods[self.provider.provider].append(configurate_method)

        # 判断是否使用自定义模型
        should_use_custom_model = False
        if original_provider_configurate_methods[self.provider.provider] == [ConfigurateMethod.CUSTOMIZABLE_MODEL]:
            should_use_custom_model = True

        # 第三阶段：处理系统配额限制
        for quota_configuration in self.system_configuration.quota_configurations:
            # 只处理当前生效的配额类型
            if self.system_configuration.current_quota_type != quota_configuration.quota_type:
                continue

            # 获取限制模型列表
            restrict_models = quota_configuration.restrict_models
            if len(restrict_models) == 0:
                break  # 无限制则跳过

            # 处理需要自定义模型的情况
            if should_use_custom_model:
                # 检查是否为纯自定义模型配置方式
                if original_provider_configurate_methods[self.provider.provider] == [
                    ConfigurateMethod.CUSTOMIZABLE_MODEL
                ]:
                    # 遍历每个限制模型
                    for restrict_model in restrict_models:
                        # 复制系统凭证作为基础
                        copy_credentials = (
                            self.system_configuration.credentials.copy()
                            if self.system_configuration.credentials
                            else {}
                        )
                        # 添加基础模型名称参数
                        if restrict_model.base_model_name:
                            copy_credentials["base_model_name"] = restrict_model.base_model_name

                        try:
                            # 尝试获取自定义模型结构
                            custom_model_schema = self.get_model_schema(
                                model_type=restrict_model.model_type,
                                model=restrict_model.model,
                                credentials=copy_credentials
                            )
                            # 将自定义模型加入结果列表
                            provider_models.append(custom_model_schema)
                        except Exception as e:
                            # 记录获取失败日志
                            logger.error(f"获取自定义模型结构失败: {str(e)}")

        return provider_models


class ProviderConfigurations(BaseModel):
    """
    提供商配置管理核心类

    职责:
        - 集中管理租户下所有AI模型提供商的配置信息
        - 提供跨提供商的模型查询能力
        - 实现配置数据的字典式访问接口

    特性:
        - 继承自Pydantic BaseModel，支持数据验证
        - 自动处理提供商标识标准化（ModelProviderID转换）
        - 支持复杂模式降级逻辑（系统模式->自定义模式->未配置）
    """

    tenant_id: str = Field(..., description="租户唯一标识")
    configurations: dict[str, ProviderConfiguration] = Field(
        default_factory=dict,
        description="提供商配置字典，key为标准化提供商标识（含斜杠格式）"
    )

    def __init__(self, tenant_id: str):
        """初始化租户配置容器"""
        super().__init__(tenant_id=tenant_id)

    def get_models(
            self,
            provider: Optional[str] = None,
            model_type: Optional[ModelType] = None,
            only_active: bool = False
    ) -> list[ModelWithProviderEntity]:
        """
        获取当前租户可用的所有模型列表（跨提供商）

        参数:
            provider: 指定过滤的提供商标识（可选）
            model_type: 指定过滤的模型类型（可选）
            only_active: 是否只返回已激活的可用模型

        返回:
            list[ModelWithProviderEntity]: 包含完整提供商信息的模型实体列表

        处理逻辑:
            1. 遍历所有已配置的提供商
            2. 根据provider参数过滤指定提供商
            3. 调用各提供商的get_provider_models获取模型列表
            4. 汇总并返回跨提供商的模型集合

        典型场景:
            - 全局模型检索时使用（如模型选择器）
            - 租户配额分析时获取全量模型数据
        """
        all_models = []
        # 遍历所有提供商配置
        for provider_configuration in self.values():
            # 按提供商标识过滤
            if provider and provider_configuration.provider.provider != provider:
                continue

            # 聚合各提供商的模型列表
            all_models.extend(provider_configuration.get_provider_models(model_type, only_active))

        return all_models

    def to_list(self) -> list[ProviderConfiguration]:
        """
        将字典式配置转换为列表形式

        返回:
            list[ProviderConfiguration]: 按插入顺序排列的配置对象列表

        使用场景:
            - 需要顺序遍历配置时
            - 与其他列表式API交互时
        """
        return list(self.values())

    def __getitem__(self, key: str) -> ProviderConfiguration:
        """
        字典式访问方法（支持自动ID转换）

        示例:
            >>> configs['openai']  # 自动转换为 'openai/api'
        """
        # 标准化提供商标识
        if "/" not in key:
            key = str(ModelProviderID(key))
        return self.configurations[key]

    def __setitem__(self, key: str, value: ProviderConfiguration) -> None:
        """设置提供商配置（自动校验类型）"""
        self.configurations[key] = value

    def __iter__(self) -> Iterator[str]:
        """迭代所有标准化提供商标识"""
        return iter(self.configurations)

    def values(self) -> Iterator[ProviderConfiguration]:
        """获取所有配置对象的迭代器"""
        return iter(self.configurations.values())

    def get(self, key: str, default: Any = None) -> ProviderConfiguration | None:
        """
        安全获取配置方法（带默认值）

        参数:
            key: 原始提供商标识或标准化ID
            default: 未找到时的默认值
        """
        # 自动转换普通提供商标识
        if "/" not in key:
            key = str(ModelProviderID(key))
        return self.configurations.get(key, default)


class ProviderModelBundle(BaseModel):
    """
    AI模型提供商功能包

    作用:
        封装提供商配置与模型实例的绑定关系
        实现配置验证与模型实例的类型安全关联

    特性:
        - 继承自Pydantic BaseModel，支持自动数据验证:ml-citation{ref="1,3" data="citationList"}
        - 允许包含非Pydantic原生类型的自定义类实例:ml-citation{ref="6" data="citationList"}
        - 禁用Pydantic的命名空间保护机制
    """

    configuration: ProviderConfiguration
    """提供商配置对象，包含认证信息、模型设置等业务参数"""

    model_type_instance: AIModel
    """AI模型实例，承载具体的模型推理与业务逻辑实现"""

    # Pydantic配置参数
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许任意类型字段(如自定义类):ml-citation{ref="6,8" data="citationList"}
        protected_namespaces=()  # 禁用Pydantic的保留命名空间保护:ml-citation{ref="6" data="citationList"}
    )
