import re
import sys
from typing import Any

from flask import current_app, got_request_exception
from flask_restful import Api, http_status_message  # type: ignore
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException

from core.errors.error import AppInvokeQuotaExceededError


class ExternalApi(Api):
    """自定义API异常处理器，继承自flask_restful.Api"""

    def handle_error(self, e):
        """统一异常处理方法，将各种异常转换为标准API响应格式

        Args:
            e: 捕获的异常对象

        Returns:
            Response: 包含错误信息的标准化响应
        """
        # 发送异常信号，便于其他监听器处理
        got_request_exception.send(current_app, exception=e)

        headers = Headers()

        # HTTP异常处理（404/403等）
        if isinstance(e, HTTPException):
            if e.response is not None:
                return e.get_response()

            status_code = e.code
            # 生成默认错误数据结构：蛇形命名错误码 + 描述信息
            default_data = {
                "code": re.sub(r"(?<!^)(?=[A-Z])", "_", type(e).__name__).lower(),
                "message": getattr(e, "description", http_status_message(status_code)),
                "status": status_code,
            }

            # 特殊处理JSON解析错误
            if default_data["message"] == "Failed to decode JSON object: Expecting value: line 1 column 1 (char 0)":
                default_data["message"] = "无效的JSON数据或JSON数据为空"

            headers = e.get_response().headers

        # 参数值错误（400）
        elif isinstance(e, ValueError):
            status_code = 400
            default_data = {
                "code": "invalid_param",
                "message": str(e),
                "status": status_code,
            }

        # 自定义限流异常（429）
        elif isinstance(e, AppInvokeQuotaExceededError):
            status_code = 429
            default_data = {
                "code": "too_many_requests",
                "message": str(e),
                "status": status_code,
            }

        # 其他未捕获异常（500）
        else:
            status_code = 500
            default_data = {
                "message": http_status_message(status_code),
            }

        # 移除重复的Content-Length头（Werkzeug兼容处理）
        remove_headers = ("Content-Length",)
        for header in remove_headers:
            headers.pop(header, None)

        # 合并自定义错误数据
        data = getattr(e, "data", default_data)
        error_cls_name = type(e).__name__

        # 检查是否有预定义的错误处理模板
        if error_cls_name in self.errors:
            custom_data = self.errors.get(error_cls_name, {}).copy()
            status_code = custom_data.get("status", 500)

            # 格式化错误消息模板
            if "message" in custom_data:
                custom_data["message"] = custom_data["message"].format(
                    message=str(e.description if hasattr(e, "description") else e)
                )
                data.update(custom_data)

                # 记录500错误日志
                if status_code and status_code >= 500:
                    exc_info = sys.exc_info()
                if exc_info[1] is None:
                    exc_info = None
                current_app.log_exception(exc_info)

                # 特殊处理406不接受错误（媒体类型协商）
                if status_code == 406 and self.default_mediatype is None:
                    supported_mediatypes = list(self.representations.keys())
                    fallback_mediatype = supported_mediatypes[0] if supported_mediatypes else "text/plain"
                    data = {"code": "not_acceptable", "message": data.get("message")}
                    resp = self.make_response(data, status_code, headers, fallback_mediatype=fallback_mediatype)

                # 特殊处理400参数错误
                elif status_code == 400:
                    if isinstance(data.get("message"), dict):
                        param_key, param_value = list(data.get("message", {}).items())[0]
                        data = {"code": "invalid_param", "message": param_value, "params": param_key}
                    else:
                        data.setdefault("code", "unknown")
                        resp = self.make_response(data, status_code, headers)

                # 常规错误响应
                else:
                    data.setdefault("code", "unknown")
                    resp = self.make_response(data, status_code, headers)

                # 401未授权特殊处理
                if status_code == 401:
                    resp = self.unauthorized(resp)

        return resp