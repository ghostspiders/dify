import json
import logging
import random
import re
import string
import subprocess
import time
import uuid
from collections.abc import Generator, Mapping
from datetime import datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Optional, Union, cast
from zoneinfo import available_timezones

from flask import Response, stream_with_context
from flask_restful import fields  # type: ignore

from configs import dify_config
from core.app.features.rate_limiting.rate_limit import RateLimitGenerator
from core.file import helpers as file_helpers
from extensions.ext_redis import redis_client

if TYPE_CHECKING:
    from models.account import Account


def run(script):
    """执行shell脚本命令"""
    return subprocess.getstatusoutput("source /root/.bashrc && " + script)


class AppIconUrlField(fields.Raw):
    """应用图标URL字段格式化类"""

    def output(self, key, obj):
        """生成应用图标的签名URL"""
        if obj is None:
            return None

        from models.model import App, IconType, Site

        # 处理字典或对象输入
        if isinstance(obj, dict) and "app" in obj:
            obj = obj["app"]

        # 只处理图片类型的图标
        if isinstance(obj, App | Site) and obj.icon_type == IconType.IMAGE.value:
            return file_helpers.get_signed_file_url(obj.icon)
        return None


class AvatarUrlField(fields.Raw):
    """用户头像URL字段格式化类"""

    def output(self, key, obj):
        """生成用户头像的签名URL"""
        if obj is None:
            return None

        from models.account import Account

        if isinstance(obj, Account) and obj.avatar is not None:
            return file_helpers.get_signed_file_url(obj.avatar)
        return None


class TimestampField(fields.Raw):
    """时间戳字段格式化类"""

    def format(self, value) -> int:
        """将datetime对象转换为时间戳"""
        return int(value.timestamp())


def email(email):
    """验证邮箱格式"""
    pattern = r"^[\w\.!#$%&'*+\-/=?^_`{|}~]+@([\w-]+\.)+[\w-]{2,}$"
    if re.match(pattern, email) is not None:
        return email

    raise ValueError(f"{email} is not a valid email")


def uuid_value(value):
    """验证UUID格式"""
    if value == "":
        return str(value)

    try:
        uuid_obj = uuid.UUID(value)
        return str(uuid_obj)
    except ValueError:
        raise ValueError(f"{value} is not a valid uuid")


def alphanumeric(value: str):
    """验证只包含字母数字和下划线"""
    if re.match(r"^[a-zA-Z0-9_]+$", value):
        return value
    raise ValueError(f"{value} is not a valid alphanumeric value")


def timestamp_value(timestamp):
    """验证时间戳格式"""
    try:
        int_timestamp = int(timestamp)
        if int_timestamp < 0:
            raise ValueError
        return int_timestamp
    except ValueError:
        raise ValueError(f"{timestamp} is not a valid timestamp")


class StrLen:
    """字符串长度验证器"""

    def __init__(self, max_length, argument="argument"):
        self.max_length = max_length
        self.argument = argument

    def __call__(self, value):
        """验证字符串长度"""
        if len(value) > self.max_length:
            raise ValueError(
                f"Invalid {self.argument}: {value}. {self.argument} cannot exceed length {self.max_length}"
            )
        return value


class FloatRange:
    """浮点数范围验证器"""

    def __init__(self, low, high, argument="argument"):
        self.low = low
        self.high = high
        self.argument = argument

    def __call__(self, value):
        """验证浮点数是否在指定范围内"""
        value = _get_float(value)
        if value < self.low or value > self.high:
            raise ValueError(
                f"Invalid {self.argument}: {value}. {self.argument} must be within the range {self.low} - {self.high}"
            )
        return value


class DatetimeString:
    """日期时间字符串验证器"""

    def __init__(self, format, argument="argument"):
        self.format = format
        self.argument = argument

    def __call__(self, value):
        """验证日期时间格式"""
        try:
            datetime.strptime(value, self.format)
        except ValueError:
            raise ValueError(
                f"Invalid {self.argument}: {value}. {self.argument} must be conform to the format {self.format}"
            )
        return value


def _get_float(value):
    """转换为浮点数"""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{value} is not a valid float")


def timezone(timezone_string):
    """验证时区有效性"""
    if timezone_string and timezone_string in available_timezones():
        return timezone_string
    raise ValueError(f"{timezone_string} is not a valid timezone")


def generate_string(n):
    """生成随机字母数字字符串"""
    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(n))


def extract_remote_ip(request) -> str:
    """从请求头提取客户端真实IP"""
    if request.headers.get("CF-Connecting-IP"):
        return cast(str, request.headers.get("Cf-Connecting-Ip"))
    elif request.headers.getlist("X-Forwarded-For"):
        return cast(str, request.headers.getlist("X-Forwarded-For")[0])
    return cast(str, request.remote_addr)


def generate_text_hash(text: str) -> str:
    """生成文本的SHA256哈希"""
    return sha256(f"{text}None".encode()).hexdigest()


def compact_generate_response(response: Union[Mapping, Generator, RateLimitGenerator]) -> Response:
    """生成紧凑的API响应"""
    if isinstance(response, dict):
        return Response(response=json.dumps(response), status=200, mimetype="application/json")

    def generate() -> Generator:
        yield from response

    return Response(stream_with_context(generate()), status=200, mimetype="text/event-stream")


class TokenManager:
    """令牌管理工具类"""

    @classmethod
    def generate_token(
            cls,
            token_type: str,
            account: Optional["Account"] = None,
            email: Optional[str] = None,
            additional_data: Optional[dict] = None,
    ) -> str:
        """生成并存储新令牌"""
        if account is None and email is None:
            raise ValueError("必须提供账户或邮箱")

        account_id = account.id if account else None
        account_email = account.email if account else email

        # 撤销旧令牌(如果存在)
        if account_id:
            if old_token := cls._get_current_token_for_account(account_id, token_type):
                cls.revoke_token(old_token.decode("utf-8") if isinstance(old_token, bytes) else old_token, token_type)

        # 生成新令牌
        token = str(uuid.uuid4())
        token_data = {"account_id": account_id, "email": account_email, "token_type": token_type}
        if additional_data:
            token_data.update(additional_data)

        # 设置Redis过期时间
        expiry_minutes = dify_config.model_dump().get(f"{token_type.upper()}_TOKEN_EXPIRY_MINUTES")
        if expiry_minutes is None:
            raise ValueError(f"{token_type}令牌的过期时间未设置")

        redis_client.setex(
            cls._get_token_key(token, token_type),
            int(expiry_minutes * 60),
            json.dumps(token_data)
        )

        # 关联账户和令牌
        if account_id:
            cls._set_current_token_for_account(account_id, token, token_type, expiry_minutes)

        return token

    @classmethod
    def _get_token_key(cls, token: str, token_type: str) -> str:
        """获取令牌的Redis键"""
        return f"{token_type}:token:{token}"

    @classmethod
    def revoke_token(cls, token: str, token_type: str):
        """撤销指定令牌"""
        redis_client.delete(cls._get_token_key(token, token_type))

    @classmethod
    def get_token_data(cls, token: str, token_type: str) -> Optional[dict[str, Any]]:
        """获取令牌数据"""
        token_data_json = redis_client.get(cls._get_token_key(token, token_type))
        if token_data_json is None:
            logging.warning(f"{token_type}令牌{token}未找到")
            return None
        return json.loads(token_data_json)

    @classmethod
    def _get_current_token_for_account(cls, account_id: str, token_type: str) -> Optional[str]:
        """获取账户当前令牌"""
        return redis_client.get(cls._get_account_token_key(account_id, token_type))

    @classmethod
    def _set_current_token_for_account(
            cls, account_id: str, token: str, token_type: str, expiry_hours: Union[int, float]
    ):
        """设置账户当前令牌"""
        redis_client.setex(
            cls._get_account_token_key(account_id, token_type),
            int(expiry_hours * 60 * 60),
            token
        )

    @classmethod
    def _get_account_token_key(cls, account_id: str, token_type: str) -> str:
        """获取账户令牌的Redis键"""
        return f"{token_type}:account:{account_id}"


class RateLimiter:
    """速率限制器"""

    def __init__(self, prefix: str, max_attempts: int, time_window: int):
        """
        :param prefix: Redis键前缀
        :param max_attempts: 时间窗口内最大尝试次数
        :param time_window: 时间窗口(秒)
        """
        self.prefix = prefix
        self.max_attempts = max_attempts
        self.time_window = time_window

    def _get_key(self, email: str) -> str:
        """获取速率限制的Redis键"""
        return f"{self.prefix}:{email}"

    def is_rate_limited(self, email: str) -> bool:
        """检查是否达到速率限制"""
        key = self._get_key(email)
        current_time = int(time.time())

        # 清理过期记录
        redis_client.zremrangebyscore(key, "-inf", current_time - self.time_window)

        # 获取当前窗口内的尝试次数
        attempts = redis_client.zcard(key)
        return attempts and int(attempts) >= self.max_attempts

    def increment_rate_limit(self, email: str):
        """增加速率限制计数"""
        key = self._get_key(email)
        current_time = int(time.time())

        redis_client.zadd(key, {current_time: current_time})
        redis_client.expire(key, self.time_window * 2)