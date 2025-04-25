from configs import dify_config  # 导入Dify配置
from dify_app import DifyApp  # 导入自定义Flask应用类


def init_app(app: DifyApp):
    """
    初始化Flask应用的路由和CORS配置

    Args:
        app (DifyApp): Flask应用实例
    """
    # 注册蓝图路由
    from flask_cors import CORS  # type: ignore  # CORS跨域支持

    # 导入各业务模块的蓝图
    from controllers.console import bp as console_app_bp  # 控制台接口
    from controllers.files import bp as files_bp  # 文件操作接口
    from controllers.inner_api import bp as inner_api_bp  # 内部API接口
    from controllers.service_api import bp as service_api_bp  # 服务API接口
    from controllers.web import bp as web_bp  # Web页面接口

    # 配置服务API接口的CORS规则
    CORS(
        service_api_bp,
        allow_headers=["Content-Type", "Authorization", "X-App-Code"],  # 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],  # 允许的HTTP方法
    )
    app.register_blueprint(service_api_bp)  # 注册服务API蓝图

    # 配置Web接口的CORS规则（更严格的限制）
    CORS(
        web_bp,
        resources={r"/*": {"origins": dify_config.WEB_API_CORS_ALLOW_ORIGINS}},  # 允许的源(从配置读取)
        supports_credentials=True,  # 允许携带凭据(cookie等)
        allow_headers=["Content-Type", "Authorization", "X-App-Code"],  # 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],  # 允许的HTTP方法
        expose_headers=["X-Version", "X-Env"],  # 暴露给客户端的响应头
    )
    app.register_blueprint(web_bp)  # 注册Web蓝图

    # 配置控制台接口的CORS规则
    CORS(
        console_app_bp,
        resources={r"/*": {"origins": dify_config.CONSOLE_CORS_ALLOW_ORIGINS}},  # 允许的源(从配置读取)
        supports_credentials=True,  # 允许携带凭据
        allow_headers=["Content-Type", "Authorization"],  # 允许的请求头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],  # 允许的HTTP方法
        expose_headers=["X-Version", "X-Env"],  # 暴露给客户端的响应头
    )
    app.register_blueprint(console_app_bp)  # 注册控制台蓝图

    # 配置文件接口的CORS规则（基础配置）
    CORS(
        files_bp,
        allow_headers=["Content-Type"],  # 只允许Content-Type头
        methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"]  # 允许的标准HTTP方法
    )
    app.register_blueprint(files_bp)  # 注册文件操作蓝图

    # 注册内部API蓝图（不需要CORS，默认仅限内部调用）
    app.register_blueprint(inner_api_bp)