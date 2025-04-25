import os
import sys


def is_db_command():
    """检查当前是否是执行数据库迁移命令"""
    # 判断条件：参数数量大于1 且 主程序是flask 且 第一个参数是'db'
    if len(sys.argv) > 1 and sys.argv[0].endswith("flask") and sys.argv[1] == "db":
        return True
    return False


# 创建应用
if is_db_command():
    # 如果是数据库迁移命令，创建专门用于迁移的轻量级应用
    from app_factory import create_migrations_app

    app = create_migrations_app()
else:
    # 以下是正常应用启动流程

    # JetBrains的Python调试器与gevent兼容性不好，
    # 因此在调试模式下禁用gevent。
    # 如果使用debugpy并设置GEVENT_SUPPORT=True，可以启用gevent调试
    if (flask_debug := os.environ.get("FLASK_DEBUG", "0")) and flask_debug.lower() in {"false", "0", "no"}:
        # 导入gevent并打猴子补丁
        from gevent import monkey  # type: ignore

        # gevent猴子补丁
        monkey.patch_all()

        # gRPC的gevent支持
        from grpc.experimental import gevent as grpc_gevent  # type: ignore

        grpc_gevent.init_gevent()

        # PostgreSQL的gevent支持
        import psycogreen.gevent  # type: ignore

        psycogreen.gevent.patch_psycopg()

    # 创建正式应用
    from app_factory import create_app

    app = create_app()
    # 获取Celery实例
    celery = app.extensions["celery"]

if __name__ == "__main__":
    # 启动应用，监听所有网络接口的5001端口
    app.run(host="0.0.0.0", port=5001)