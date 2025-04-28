import json

import flask_login  # type: ignore
from flask import Response, request
from flask_login import user_loaded_from_request, user_logged_in
from werkzeug.exceptions import Unauthorized

import contexts
from dify_app import DifyApp
from libs.passport import PassportService
from services.account_service import AccountService

# 初始化Flask-Login的LoginManager实例
login_manager = flask_login.LoginManager()


# Flask-Login配置
@login_manager.request_loader
def load_user_from_request(request_from_flask_login):
    """从请求中加载用户"""
    # 只处理特定蓝图（console和inner_api）的请求
    if request.blueprint not in {"console", "inner_api"}:
        return None

    # 检查Authorization头
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        # 如果没有Authorization头，尝试从查询参数获取token
        auth_token = request.args.get("_token")
        if not auth_token:
            raise Unauthorized("无效的授权token")
    else:
        # 验证Authorization头格式（Bearer Token格式）
        if " " not in auth_header:
            raise Unauthorized("无效的Authorization头格式，应为'Bearer <api-key>'格式")
        auth_scheme, auth_token = auth_header.split(None, 1)
        auth_scheme = auth_scheme.lower()
        if auth_scheme != "bearer":
            raise Unauthorized("无效的Authorization头格式，应为'Bearer <api-key>'格式")

    # 验证token并解码
    decoded = PassportService().verify(auth_token)
    user_id = decoded.get("user_id")

    # 加载已登录的账户
    logged_in_account = AccountService.load_logged_in_account(account_id=user_id)
    return logged_in_account


# 用户登录信号处理
@user_logged_in.connect
@user_loaded_from_request.connect
def on_user_logged_in(_sender, user):
    """当用户登录时调用"""
    if user:
        # 设置当前租户ID到上下文
        contexts.tenant_id.set(user.current_tenant_id)


# 未授权请求处理
@login_manager.unauthorized_handler
def unauthorized_handler():
    """处理未授权的请求"""
    return Response(
        json.dumps({"code": "unauthorized", "message": "未授权"}),
        status=401,
        content_type="application/json",
    )


def init_app(app: DifyApp):
    """初始化应用"""
    # 将LoginManager与Flask应用关联
    login_manager.init_app(app)
