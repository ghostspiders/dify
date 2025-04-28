from typing import Optional

from flask import Flask
from pydantic import BaseModel

from configs import dify_config
from core.entities import DEFAULT_PLUGIN_ID
from core.entities.provider_entities import ProviderQuotaType, QuotaUnit, RestrictModel
from core.model_runtime.entities.model_entities import ModelType


class HostingQuota(BaseModel):
    quota_type: ProviderQuotaType
    restrict_models: list[RestrictModel] = []


class TrialHostingQuota(HostingQuota):
    quota_type: ProviderQuotaType = ProviderQuotaType.TRIAL
    quota_limit: int = 0
    """Quota limit for the hosting provider models. -1 means unlimited."""


class PaidHostingQuota(HostingQuota):
    quota_type: ProviderQuotaType = ProviderQuotaType.PAID


class FreeHostingQuota(HostingQuota):
    quota_type: ProviderQuotaType = ProviderQuotaType.FREE


class HostingProvider(BaseModel):
    enabled: bool = False
    credentials: Optional[dict] = None
    quota_unit: Optional[QuotaUnit] = None
    quotas: list[HostingQuota] = []


class HostedModerationConfig(BaseModel):
    enabled: bool = False
    providers: list[str] = []


class HostingConfiguration:
    # 提供者映射字典，键为字符串，值为HostingProvider对象
    provider_map: dict[str, HostingProvider]
    # 可选的内容审核配置
    moderation_config: Optional[HostedModerationConfig] = None

    def __init__(self) -> None:
        # 初始化提供者映射为空字典
        self.provider_map = {}
        # 初始化审核配置为None
        self.moderation_config = None

    def init_app(self, app: Flask) -> None:
        # 如果不是CLOUD版本，则不初始化
        if dify_config.EDITION != "CLOUD":
            return

        # 初始化各个AI服务提供者并添加到provider_map中
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/azure_openai/azure_openai"] = self.init_azure_openai()
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/openai/openai"] = self.init_openai()
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/anthropic/anthropic"] = self.init_anthropic()
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/minimax/minimax"] = self.init_minimax()
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/spark/spark"] = self.init_spark()
        self.provider_map[f"{DEFAULT_PLUGIN_ID}/zhipuai/zhipuai"] = self.init_zhipuai()

        # 初始化内容审核配置
        self.moderation_config = self.init_moderation_config()

    @staticmethod
    def init_azure_openai() -> HostingProvider:
        # 设置配额单位为次数
        quota_unit = QuotaUnit.TIMES
        # 如果Azure OpenAI服务启用
        if dify_config.HOSTED_AZURE_OPENAI_ENABLED:
            # 设置凭证信息
            credentials = {
                "openai_api_key": dify_config.HOSTED_AZURE_OPENAI_API_KEY,
                "openai_api_base": dify_config.HOSTED_AZURE_OPENAI_API_BASE,
                "base_model_name": "gpt-35-turbo",
            }

            quotas: list[HostingQuota] = []
            hosted_quota_limit = dify_config.HOSTED_AZURE_OPENAI_QUOTA_LIMIT
            # 创建试用配额对象，包含限制的模型列表
            trial_quota = TrialHostingQuota(
                quota_limit=hosted_quota_limit,
                restrict_models=[
                    # 各种GPT-4模型
                    RestrictModel(model="gpt-4", base_model_name="gpt-4", model_type=ModelType.LLM),
                    RestrictModel(model="gpt-4o", base_model_name="gpt-4o", model_type=ModelType.LLM),
                    RestrictModel(model="gpt-4o-mini", base_model_name="gpt-4o-mini", model_type=ModelType.LLM),
                    RestrictModel(model="gpt-4-32k", base_model_name="gpt-4-32k", model_type=ModelType.LLM),
                    RestrictModel(
                        model="gpt-4-1106-preview", base_model_name="gpt-4-1106-preview", model_type=ModelType.LLM
                    ),
                    RestrictModel(
                        model="gpt-4-vision-preview", base_model_name="gpt-4-vision-preview", model_type=ModelType.LLM
                    ),
                    # 各种GPT-3.5模型
                    RestrictModel(model="gpt-35-turbo", base_model_name="gpt-35-turbo", model_type=ModelType.LLM),
                    RestrictModel(
                        model="gpt-35-turbo-1106", base_model_name="gpt-35-turbo-1106", model_type=ModelType.LLM
                    ),
                    RestrictModel(
                        model="gpt-35-turbo-instruct", base_model_name="gpt-35-turbo-instruct", model_type=ModelType.LLM
                    ),
                    RestrictModel(
                        model="gpt-35-turbo-16k", base_model_name="gpt-35-turbo-16k", model_type=ModelType.LLM
                    ),
                    # 其他模型
                    RestrictModel(
                        model="text-davinci-003", base_model_name="text-davinci-003", model_type=ModelType.LLM
                    ),
                    # 文本嵌入模型
                    RestrictModel(
                        model="text-embedding-ada-002",
                        base_model_name="text-embedding-ada-002",
                        model_type=ModelType.TEXT_EMBEDDING,
                    ),
                    RestrictModel(
                        model="text-embedding-3-small",
                        base_model_name="text-embedding-3-small",
                        model_type=ModelType.TEXT_EMBEDDING,
                    ),
                    RestrictModel(
                        model="text-embedding-3-large",
                        base_model_name="text-embedding-3-large",
                        model_type=ModelType.TEXT_EMBEDDING,
                    ),
                ],
            )
            quotas.append(trial_quota)

            # 返回启用的HostingProvider对象
            return HostingProvider(enabled=True, credentials=credentials, quota_unit=quota_unit, quotas=quotas)

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    def init_openai(self) -> HostingProvider:
        # 设置配额单位为积分
        quota_unit = QuotaUnit.CREDITS
        quotas: list[HostingQuota] = []

        # 如果OpenAI试用服务启用
        if dify_config.HOSTED_OPENAI_TRIAL_ENABLED:
            hosted_quota_limit = dify_config.HOSTED_OPENAI_QUOTA_LIMIT
            # 从环境变量解析试用模型
            trial_models = self.parse_restrict_models_from_env("HOSTED_OPENAI_TRIAL_MODELS")
            trial_quota = TrialHostingQuota(quota_limit=hosted_quota_limit, restrict_models=trial_models)
            quotas.append(trial_quota)

        # 如果OpenAI付费服务启用
        if dify_config.HOSTED_OPENAI_PAID_ENABLED:
            # 从环境变量解析付费模型
            paid_models = self.parse_restrict_models_from_env("HOSTED_OPENAI_PAID_MODELS")
            paid_quota = PaidHostingQuota(restrict_models=paid_models)
            quotas.append(paid_quota)

        # 如果有配额配置
        if len(quotas) > 0:
            credentials = {
                "openai_api_key": dify_config.HOSTED_OPENAI_API_KEY,
            }

            # 可选的基础URL
            if dify_config.HOSTED_OPENAI_API_BASE:
                credentials["openai_api_base"] = dify_config.HOSTED_OPENAI_API_BASE

            # 可选的组织ID
            if dify_config.HOSTED_OPENAI_API_ORGANIZATION:
                credentials["openai_organization"] = dify_config.HOSTED_OPENAI_API_ORGANIZATION

            return HostingProvider(enabled=True, credentials=credentials, quota_unit=quota_unit, quotas=quotas)

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    @staticmethod
    def init_anthropic() -> HostingProvider:
        # 设置配额单位为tokens
        quota_unit = QuotaUnit.TOKENS
        quotas: list[HostingQuota] = []

        # 如果Anthropic试用服务启用
        if dify_config.HOSTED_ANTHROPIC_TRIAL_ENABLED:
            hosted_quota_limit = dify_config.HOSTED_ANTHROPIC_QUOTA_LIMIT
            trial_quota = TrialHostingQuota(quota_limit=hosted_quota_limit)
            quotas.append(trial_quota)

        # 如果Anthropic付费服务启用
        if dify_config.HOSTED_ANTHROPIC_PAID_ENABLED:
            paid_quota = PaidHostingQuota()
            quotas.append(paid_quota)

        # 如果有配额配置
        if len(quotas) > 0:
            credentials = {
                "anthropic_api_key": dify_config.HOSTED_ANTHROPIC_API_KEY,
            }

            # 可选的基础URL
            if dify_config.HOSTED_ANTHROPIC_API_BASE:
                credentials["anthropic_api_url"] = dify_config.HOSTED_ANTHROPIC_API_BASE

            return HostingProvider(enabled=True, credentials=credentials, quota_unit=quota_unit, quotas=quotas)

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    @staticmethod
    def init_minimax() -> HostingProvider:
        # 设置配额单位为tokens
        quota_unit = QuotaUnit.TOKENS
        # 如果MiniMax服务启用
        if dify_config.HOSTED_MINIMAX_ENABLED:
            quotas: list[HostingQuota] = [FreeHostingQuota()]  # 使用免费配额

            return HostingProvider(
                enabled=True,
                credentials=None,  # 使用提供者的凭证
                quota_unit=quota_unit,
                quotas=quotas,
            )

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    @staticmethod
    def init_spark() -> HostingProvider:
        # 设置配额单位为tokens
        quota_unit = QuotaUnit.TOKENS
        # 如果Spark服务启用
        if dify_config.HOSTED_SPARK_ENABLED:
            quotas: list[HostingQuota] = [FreeHostingQuota()]  # 使用免费配额

            return HostingProvider(
                enabled=True,
                credentials=None,  # 使用提供者的凭证
                quota_unit=quota_unit,
                quotas=quotas,
            )

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    @staticmethod
    def init_zhipuai() -> HostingProvider:
        # 设置配额单位为tokens
        quota_unit = QuotaUnit.TOKENS
        # 如果ZhipuAI服务启用
        if dify_config.HOSTED_ZHIPUAI_ENABLED:
            quotas: list[HostingQuota] = [FreeHostingQuota()]  # 使用免费配额

            return HostingProvider(
                enabled=True,
                credentials=None,  # 使用提供者的凭证
                quota_unit=quota_unit,
                quotas=quotas,
            )

        # 返回禁用的HostingProvider对象
        return HostingProvider(
            enabled=False,
            quota_unit=quota_unit,
        )

    @staticmethod
    def init_moderation_config() -> HostedModerationConfig:
        # 如果内容审核启用且有提供者配置
        if dify_config.HOSTED_MODERATION_ENABLED and dify_config.HOSTED_MODERATION_PROVIDERS:
            providers = dify_config.HOSTED_MODERATION_PROVIDERS.split(",")
            hosted_providers = []
            for provider in providers:
                # 如果提供者格式不完整，添加默认插件ID
                if "/" not in provider:
                    provider = f"{DEFAULT_PLUGIN_ID}/{provider}/{provider}"
                hosted_providers.append(provider)

            return HostedModerationConfig(enabled=True, providers=hosted_providers)

        # 返回禁用的内容审核配置
        return HostedModerationConfig(enabled=False)

    @staticmethod
    def init_moderation_config() -> HostedModerationConfig:
        # 如果内容审核启用且有提供者配置
        if dify_config.HOSTED_MODERATION_ENABLED and dify_config.HOSTED_MODERATION_PROVIDERS:
            providers = dify_config.HOSTED_MODERATION_PROVIDERS.split(",")
            hosted_providers = []
            for provider in providers:
                # 如果提供者格式不完整，添加默认插件ID
                if "/" not in provider:
                    provider = f"{DEFAULT_PLUGIN_ID}/{provider}/{provider}"
                hosted_providers.append(provider)

            return HostedModerationConfig(enabled=True, providers=hosted_providers)
        # 返回禁用的内容审核配置
        return HostedModerationConfig(enabled=False)

    @staticmethod
    def parse_restrict_models_from_env(env_var: str) -> list[RestrictModel]:
        # 从环境变量获取模型字符串
        models_str = dify_config.model_dump().get(env_var)
        # 分割字符串为列表
        models_list = models_str.split(",") if models_str else []
        # 返回限制模型列表
        return [
            RestrictModel(model=model_name.strip(), model_type=ModelType.LLM)
            for model_name in models_list
            if model_name.strip()
        ]
