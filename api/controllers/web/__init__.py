from flask import Blueprint  # Flask蓝图，用于模块化路由

from libs.external_api import ExternalApi  # 自定义的外部API处理类

# 导入各API资源类
from .files import FileApi  # 本地文件操作API
from .remote_files import RemoteFileInfoApi, RemoteFileUploadApi  # 远程文件操作API

# 创建名为"web"的蓝图，URL前缀为/api
# __name__表示当前模块名，用于Flask定位资源
bp = Blueprint("web", __name__, url_prefix="/api")

# 使用自定义ExternalApi类初始化API路由
api = ExternalApi(bp)

# ----------------------------
# 文件相关路由
# ----------------------------
# 注册本地文件上传接口
# POST /api/files/upload
api.add_resource(FileApi, "/files/upload")

# ----------------------------
# 远程文件相关路由
# ----------------------------
# 注册远程文件信息获取接口
# GET /api/remote-files/<url>
api.add_resource(RemoteFileInfoApi, "/remote-files/<path:url>")

# 注册远程文件上传接口
# POST /api/remote-files/upload
api.add_resource(RemoteFileUploadApi, "/remote-files/upload")

# ----------------------------
# 导入其他模块的路由（延迟导入避免循环依赖）
# ----------------------------
from . import (
    app,        # 应用管理API
    audio,      # 音频处理API
    completion, # 补全功能API
    conversation, # 会话管理API
    feature,    # 功能开关API
    message,    # 消息处理API
    passport,   # 认证相关API
    saved_message, # 消息保存API
    site,       # 站点管理API
    workflow    # 工作流API
)