import logging
from typing import Any

from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# 导入各配置模块
from .deploy import DeploymentConfig  # 部署相关配置
from .enterprise import EnterpriseFeatureConfig  # 企业版功能配置
from .extra import ExtraServiceConfig  # 额外服务配置
from .feature import FeatureConfig  # 功能开关配置
from .middleware import MiddlewareConfig  # 中间件配置
from .observability import ObservabilityConfig  # 可观测性配置
from .packaging import PackagingInfo  # 打包信息
from .remote_settings_sources import RemoteSettingsSource, RemoteSettingsSourceConfig, RemoteSettingsSourceName
from .remote_settings_sources.apollo import ApolloSettingsSource  # Apollo配置中心
from .remote_settings_sources.nacos import NacosSettingsSource  # Nacos配置中心

logger = logging.getLogger(__name__)


class RemoteSettingsSourceFactory(PydanticBaseSettingsSource):
    """远程配置源工厂类，用于从不同配置中心加载配置"""

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """获取字段值（需子类实现）"""
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        """执行远程配置加载"""
        current_state = self.current_state
        # 获取配置的远程源名称
        remote_source_name = current_state.get("REMOTE_SETTINGS_SOURCE_NAME")
        if not remote_source_name:
            return {}  # 未配置远程源时返回空

        remote_source: RemoteSettingsSource | None = None
        # 根据配置选择不同的配置中心实现
        match remote_source_name:
            case RemoteSettingsSourceName.APOLLO:
                remote_source = ApolloSettingsSource(current_state)  # Apollo配置中心
            case RemoteSettingsSourceName.NACOS:
                remote_source = NacosSettingsSource(current_state)  # Nacos配置中心
            case _:
                logger.warning(f"不支持的远程配置源: {remote_source_name}")
                return {}

        d: dict[str, Any] = {}

        # 遍历所有配置字段，从远程源获取值
        for field_name, field in self.settings_cls.model_fields.items():
            # 从远程源获取字段值
            field_value, field_key, value_is_complex = remote_source.get_field_value(field, field_name)
            # 对字段值进行预处理
            field_value = remote_source.prepare_field_value(field_name, field, field_value, value_is_complex)
            if field_value is not None:
                d[field_key] = field_value  # 只添加非None值

        return d


class DifyConfig(
    # 配置继承顺序（也是文档中的显示顺序）
    PackagingInfo,  # 打包信息
    DeploymentConfig,  # 部署配置
    FeatureConfig,  # 功能配置
    MiddlewareConfig,  # 中间件配置
    ExtraServiceConfig,  # 额外服务配置
    ObservabilityConfig,  # 可观测性配置
    RemoteSettingsSourceConfig,  # 远程配置源配置
    EnterpriseFeatureConfig,  # 企业版功能配置（需商业授权）
):
    """Dify核心配置类，聚合所有配置组"""

    model_config = SettingsConfigDict(
        env_file=".env",  # 从.env文件读取配置
        env_file_encoding="utf-8",  # 文件编码
        extra="ignore",  # 忽略额外字段
    )

    # 重要提示（代码注释）：
    # 添加新配置前，请考虑将其放在合适的现有配置组中
    # 或创建新的配置组，以保持代码可读性和可维护性

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,  # 初始化设置
            env_settings: PydanticBaseSettingsSource,  # 环境变量设置
            dotenv_settings: PydanticBaseSettingsSource,  # .env文件设置
            file_secret_settings: PydanticBaseSettingsSource,  # 密钥文件设置
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """自定义配置源加载顺序"""
        return (
            init_settings,  # 1. 初始化参数
            env_settings,  # 2. 环境变量
            RemoteSettingsSourceFactory(settings_cls),  # 3. 远程配置中心
            dotenv_settings,  # 4. .env文件
            file_secret_settings,  # 5. 密钥文件
        )