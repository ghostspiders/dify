from datetime import timedelta
import pytz
from celery import Celery, Task  # type: ignore
from celery.schedules import crontab  # type: ignore

from configs import dify_config  # 项目配置
from dify_app import DifyApp  # Flask应用类


def init_app(app: DifyApp) -> Celery:
    """初始化Celery并集成到Flask应用

    Args:
        app: Flask应用实例

    Returns:
        Celery: 配置完成的Celery应用实例
    """

    # 自定义Task类，确保任务在Flask应用上下文中执行
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():  # 自动管理应用上下文
                return self.run(*args, **kwargs)

    # Redis哨兵模式配置
    broker_transport_options = {}
    if dify_config.CELERY_USE_SENTINEL:
        broker_transport_options = {
            "master_name": dify_config.CELERY_SENTINEL_MASTER_NAME,  # 主节点名称
            "sentinel_kwargs": {
                "socket_timeout": dify_config.CELERY_SENTINEL_SOCKET_TIMEOUT,  # 超时设置
            },
        }

    # 创建Celery实例
    celery_app = Celery(
        app.name,  # 使用Flask应用名称
        task_cls=FlaskTask,  # 使用自定义Task类
        broker=dify_config.CELERY_BROKER_URL,  # 消息代理地址
        backend=dify_config.CELERY_BACKEND,  # 结果存储后端
        task_ignore_result=True,  # 忽略任务结果
    )

    # SSL连接配置
    ssl_options = {
        "ssl_cert_reqs": None,
        "ssl_ca_certs": None,
        "ssl_certfile": None,
        "ssl_keyfile": None,
    }

    # 基础配置
    celery_app.conf.update(
        result_backend=dify_config.CELERY_RESULT_BACKEND,  # 结果后端
        broker_transport_options=broker_transport_options,  # 哨兵配置
        broker_connection_retry_on_startup=True,  # 启动时重连
        worker_log_format=dify_config.LOG_FORMAT,  # 日志格式
        worker_task_log_format=dify_config.LOG_FORMAT,  # 任务日志格式
        worker_hijack_root_logger=False,  # 不劫持根日志
        timezone=pytz.timezone(dify_config.LOG_TZ or "UTC"),  # 时区设置
    )

    # 启用SSL连接
    if dify_config.BROKER_USE_SSL:
        celery_app.conf.update(
            broker_use_ssl=ssl_options,  # SSL配置
        )

    # 日志文件配置
    if dify_config.LOG_FILE:
        celery_app.conf.update(
            worker_logfile=dify_config.LOG_FILE,  # 日志文件路径
        )

    # 设为默认Celery实例并挂载到Flask扩展
    celery_app.set_default()
    app.extensions["celery"] = celery_app

    # 自动导入的任务模块列表
    imports = [
        "schedule.clean_embedding_cache_task",  # 清理嵌入缓存
        "schedule.clean_unused_datasets_task",  # 清理未使用数据集
        "schedule.create_tidb_serverless_task",  # 创建TiDB Serverless实例
        "schedule.update_tidb_serverless_status_task",  # 更新TiDB状态
        "schedule.clean_messages",  # 清理消息
        "schedule.mail_clean_document_notify_task",  # 文档清理邮件通知
    ]

    # 定时任务配置
    day = dify_config.CELERY_BEAT_SCHEDULER_TIME  # 基础间隔天数
    beat_schedule = {
        # 每天执行的清理任务
        "clean_embedding_cache_task": {
            "task": "schedule.clean_embedding_cache_task.clean_embedding_cache_task",
            "schedule": timedelta(days=day),  # 间隔天数
        },
        "clean_unused_datasets_task": {
            "task": "schedule.clean_unused_datasets_task.clean_unused_datasets_task",
            "schedule": timedelta(days=day),
        },
        # 每小时执行的任务
        "create_tidb_serverless_task": {
            "task": "schedule.create_tidb_serverless_task.create_tidb_serverless_task",
            "schedule": crontab(minute="0", hour="*"),  # 每小时0分
        },
        # 每10分钟执行的任务
        "update_tidb_serverless_status_task": {
            "task": "schedule.update_tidb_serverless_status_task.update_tidb_serverless_status_task",
            "schedule": timedelta(minutes=10),
        },
        # 每天执行的清理任务
        "clean_messages": {
            "task": "schedule.clean_messages.clean_messages",
            "schedule": timedelta(days=day),
        },
        # 每周一上午10点执行的任务
        "mail_clean_document_notify_task": {
            "task": "schedule.mail_clean_document_notify_task.mail_clean_document_notify_task",
            "schedule": crontab(minute="0", hour="10", day_of_week="1"),  # 周一10:00
        },
    }

    # 更新Celery配置
    celery_app.conf.update(beat_schedule=beat_schedule, imports=imports)

    return celery_app