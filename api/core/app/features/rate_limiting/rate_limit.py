import logging
import time
import uuid
from collections.abc import Generator, Mapping
from datetime import timedelta
from typing import Any, Optional, Union

from core.errors.error import AppInvokeQuotaExceededError
from extensions.ext_redis import redis_client  # Redis客户端实例

logger = logging.getLogger(__name__)


class RateLimit:
    """速率限制器，基于Redis实现客户端并发请求控制"""

    # Redis键模板定义
    _MAX_ACTIVE_REQUESTS_KEY = "dify:rate_limit:{}:max_active_requests"  # 最大并发数存储键
    _ACTIVE_REQUESTS_KEY = "dify:rate_limit:{}:active_requests"  # 活跃请求记录键
    _UNLIMITED_REQUEST_ID = "unlimited_request_id"  # 无限制请求标识
    _REQUEST_MAX_ALIVE_TIME = 10 * 60  # 单个请求最大存活时间(10分钟)
    _ACTIVE_REQUESTS_COUNT_FLUSH_INTERVAL = 5 * 60  # 活跃请求计数刷新间隔(5分钟)

    _instance_dict: dict[str, "RateLimit"] = {}  # 单例模式实例池

    def __new__(cls: type["RateLimit"], client_id: str, max_active_requests: int):
        """单例模式实现，保证每个client_id对应唯一实例"""
        if client_id not in cls._instance_dict:
            instance = super().__new__(cls)
            cls._instance_dict[client_id] = instance
        return cls._instance_dict[client_id]

    def __init__(self, client_id: str, max_active_requests: int):
        """初始化速率限制器
        :param client_id: 客户端唯一标识
        :param max_active_requests: 最大允许并发请求数
        """
        self.max_active_requests = max_active_requests
        # 如果已禁用或已初始化则跳过
        if self.disabled():
            return
        if hasattr(self, "initialized"):
            return

        self.initialized = True
        self.client_id = client_id
        # Redis键生成
        self.active_requests_key = self._ACTIVE_REQUESTS_KEY.format(client_id)  # 活跃请求哈希表键
        self.max_active_requests_key = self._MAX_ACTIVE_REQUESTS_KEY.format(client_id)  # 最大请求数键
        self.last_recalculate_time = float("-inf")  # 上次刷新时间初始化为最小值
        self.flush_cache(use_local_value=True)  # 初始化时强制同步本地配置到Redis

    def flush_cache(self, use_local_value=False):
        """刷新Redis缓存数据
        :param use_local_value: 是否使用本地配置覆盖Redis存储值
        """
        if self.disabled():
            return

        self.last_recalculate_time = time.time()

        # 处理最大并发数键
        if use_local_value or not redis_client.exists(self.max_active_requests_key):
            # 本地配置优先或键不存在时设置值
            redis_client.setex(
                self.max_active_requests_key,
                timedelta(days=1),
                self.max_active_requests
            )
        else:
            # 从Redis读取最新配置
            self.max_active_requests = int(redis_client.get(self.max_active_requests_key).decode("utf-8"))
        # 刷新过期时间
        redis_client.expire(self.max_active_requests_key, timedelta(days=1))

        # 处理活跃请求哈希表
        if not redis_client.exists(self.active_requests_key):
            return
        # 获取所有活跃请求记录
        request_details = redis_client.hgetall(self.active_requests_key)
        # 刷新哈希表过期时间
        redis_client.expire(self.active_requests_key, timedelta(days=1))
        # 筛选超时请求(存活超过10分钟)
        timeout_requests = [
            k for k, v in request_details.items()
            if time.time() - float(v.decode("utf-8")) > RateLimit._REQUEST_MAX_ALIVE_TIME
        ]
        # 批量删除超时记录
        if timeout_requests:
            redis_client.hdel(self.active_requests_key, *timeout_requests)

    def enter(self, request_id: Optional[str] = None) -> str:
        """进入请求处理流程
        :param request_id: 可选请求标识，未提供时自动生成
        :return: 本次请求的唯一标识
        :raises AppInvokeQuotaExceededError: 超过并发限制时抛出
        """
        if self.disabled():
            return RateLimit._UNLIMITED_REQUEST_ID

        # 达到刷新间隔时刷新缓存
        if time.time() - self.last_recalculate_time > RateLimit._ACTIVE_REQUESTS_COUNT_FLUSH_INTERVAL:
            self.flush_cache()

        # 生成请求ID（UUID）
        if not request_id:
            request_id = RateLimit.gen_request_key()

        # 获取当前活跃请求数
        active_requests_count = redis_client.hlen(self.active_requests_key)
        # 并发数检查
        if active_requests_count >= self.max_active_requests:
            raise AppInvokeQuotaExceededError(
                f"Too many requests. Please try again later. The current maximum concurrent requests allowed "
                f"for {self.client_id} is {self.max_active_requests}."
            )
        # 记录新请求（哈希表字段：request_id -> 时间戳）
        redis_client.hset(self.active_requests_key, request_id, str(time.time()))
        return request_id

    def exit(self, request_id: str):
        """退出请求处理流程
        :param request_id: 要移除的请求标识
        """
        if request_id == RateLimit._UNLIMITED_REQUEST_ID:
            return
        # 从哈希表删除对应请求
        redis_client.hdel(self.active_requests_key, request_id)

    def disabled(self):
        """检查是否禁用速率限制（max_active_requests <= 0时禁用）"""
        return self.max_active_requests <= 0

    @staticmethod
    def gen_request_key() -> str:
        """生成UUID格式的请求标识"""
        return str(uuid.uuid4())

    def generate(self, generator: Union[Generator[str, None, None], Mapping[str, Any]], request_id: str):
        """生成带速率限制控制的生成器
        :param generator: 原始生成器或字典数据
        :param request_id: 关联的请求标识
        :return: 包装后的生成器或原始数据
        """
        if isinstance(generator, Mapping):
            return generator
        else:
            return RateLimitGenerator(rate_limit=self, generator=generator, request_id=request_id)


class RateLimitGenerator:
    """速率限制生成器包装类，确保生成器迭代完成后释放请求计数"""

    def __init__(self, rate_limit: RateLimit, generator: Generator[str, None, None], request_id: str):
        self.rate_limit = rate_limit  # 关联的速率限制器
        self.generator = generator  # 被包装的生成器
        self.request_id = request_id  # 请求标识
        self.closed = False  # 关闭状态标记

    def __iter__(self):
        """返回迭代器自身"""
        return self

    def __next__(self):
        """迭代获取下一个值"""
        if self.closed:
            raise StopIteration
        try:
            return next(self.generator)
        except Exception:
            self.close()  # 异常时关闭
            raise

    def close(self):
        """关闭生成器并释放请求计数"""
        if not self.closed:
            self.closed = True
            self.rate_limit.exit