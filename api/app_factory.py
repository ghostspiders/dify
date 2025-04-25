import logging
import time

from configs import dify_config
from contexts.wrapper import RecyclableContextVar
from dify_app import DifyApp


# ----------------------------
# 应用工厂函数
# ----------------------------
def create_flask_app_with_configs() -> DifyApp:
    """
    创建一个基础的Flask应用实例
    并从.env文件加载配置

    Returns:
        DifyApp: 初始化了基本配置的Flask应用实例
    """
    # 创建DifyApp实例（继承自Flask）
    dify_app = DifyApp(__name__)
    # 从配置对象加载配置（使用pydantic的model_dump()转换为字典）
    dify_app.config.from_mapping(dify_config.model_dump())

    # 添加请求前钩子函数
    @dify_app.before_request
    def before_request():
        # 为每个请求增加唯一的标识符
        # 通过上下文变量管理请求生命周期
        RecyclableContextVar.increment_thread_recycles()

    return dify_app


def create_app() -> DifyApp:
    """
    创建完整的应用实例
    包含所有扩展的初始化

    Returns:
        DifyApp: 完全初始化的Flask应用实例
    """
    start_time = time.perf_counter()
    # 创建基础应用实例
    app = create_flask_app_with_configs()
    # 初始化所有扩展
    initialize_extensions(app)
    end_time = time.perf_counter()

    # 调试模式下记录初始化耗时
    if dify_config.DEBUG:
        logging.info(f"应用初始化完成 (耗时 {round((end_time - start_time) * 1000, 2)} 毫秒)")
    return app


def initialize_extensions(app: DifyApp):
    """
    初始化所有Flask扩展

    Args:
        app (DifyApp): Flask应用实例
    """
    # 导入所有扩展模块
    from extensions import (
        ext_app_metrics,  # 应用指标监控
        ext_blueprints,  # 蓝图路由
        ext_celery,  # Celery异步任务
        ext_code_based_extension,  # 代码基础扩展
        ext_commands,  # CLI命令
        ext_compress,  # 响应压缩
        ext_database,  # 数据库
        ext_hosting_provider,  # 托管服务提供商
        ext_import_modules,  # 模块导入
        ext_logging,  # 日志系统
        ext_login,  # 登录认证
        ext_mail,  # 邮件服务
        ext_migrate,  # 数据库迁移
        ext_otel,  # OpenTelemetry
        ext_otel_patch,  # OpenTelemetry补丁
        ext_proxy_fix,  # 代理修复
        ext_redis,  # Redis
        ext_repositories,  # 数据仓库
        ext_sentry,  # Sentry错误监控
        ext_set_secretkey,  # 密钥设置
        ext_storage,  # 存储系统
        ext_timezone,  # 时区设置
        ext_warnings,  # 警告处理
    )

    # 扩展加载顺序很重要，有依赖关系
    extensions = [
        ext_timezone,  # 时区设置（应最早加载）
        ext_logging,  # 日志系统
        ext_warnings,  # 警告处理
        ext_import_modules,  # 动态加载模块
        ext_set_secretkey,  # 设置应用密钥
        ext_compress,  # 响应压缩
        ext_code_based_extension,  # 代码基础扩展
        ext_database,  # 数据库ORM
        ext_app_metrics,  # 应用指标
        ext_migrate,  # 数据库迁移工具
        ext_redis,  # Redis缓存
        ext_storage,  # 文件存储
        ext_repositories,  # 数据仓库模式
        ext_celery,  # 异步任务队列
        ext_login,  # 用户认证
        ext_mail,  # 邮件服务
        ext_hosting_provider,  # 云服务提供商集成
        ext_sentry,  # 错误监控
        ext_proxy_fix,  # 代理头处理
        ext_blueprints,  # 路由蓝图
        ext_commands,  # CLI命令
        ext_otel_patch,  # OpenTelemetry补丁（需在otel前加载）
        ext_otel,  # 分布式追踪
    ]

    # 按顺序初始化每个扩展
    for ext in extensions:
        # 获取扩展的短名称（去掉模块路径）
        short_name = ext.__name__.split(".")[-1]

        # 检查扩展是否启用（默认启用）
        is_enabled = ext.is_enabled() if hasattr(ext, "is_enabled") else True

        if not is_enabled:
            if dify_config.DEBUG:
                logging.info(f"跳过扩展: {short_name}")
            continue

        # 记录扩展加载时间
        start_time = time.perf_counter()
        ext.init_app(app)  # 初始化扩展
        end_time = time.perf_counter()

        if dify_config.DEBUG:
            logging.info(f"已加载扩展 {short_name} (耗时 {round((end_time - start_time) * 1000, 2)} 毫秒)")


def create_migrations_app():
    """
    创建专门用于数据库迁移的轻量级应用
    只初始化必要的数据库相关扩展

    Returns:
        Flask: 仅包含数据库和迁移扩展的应用实例
    """
    app = create_flask_app_with_configs()
    from extensions import ext_database, ext_migrate

    # 只初始化数据库和迁移相关的扩展
    ext_database.init_app(app)  # 数据库
    ext_migrate.init_app(app)  # 迁移工具

    return app