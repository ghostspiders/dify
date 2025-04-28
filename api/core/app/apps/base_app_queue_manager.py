import queue
import time
from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

from sqlalchemy.orm import DeclarativeMeta

from configs import dify_config
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.entities.queue_entities import (
    AppQueueEvent,
    MessageQueueMessage,
    QueueErrorEvent,
    QueuePingEvent,
    QueueStopEvent,
    WorkflowQueueMessage,
)
from extensions.ext_redis import redis_client


class PublishFrom(Enum):
    APPLICATION_MANAGER = 1
    TASK_PIPELINE = 2


class AppQueueManager:
    """应用队列管理类，负责处理任务队列的监听、发布和停止操作"""

    def __init__(self, task_id: str, user_id: str, invoke_from: InvokeFrom) -> None:
        """初始化队列管理器

        Args:
            task_id: 任务唯一标识
            user_id: 用户唯一标识
            invoke_from: 调用来源枚举

        Raises:
            ValueError: 当缺少user_id时抛出
        """
        if not user_id:
            raise ValueError("user is required")

        self._task_id = task_id
        self._user_id = user_id
        self._invoke_from = invoke_from

        # 根据调用来源确定用户前缀（用于Redis键隔离）
        user_prefix = "account" if self._invoke_from in {InvokeFrom.EXPLORE, InvokeFrom.DEBUGGER} else "end-user"
        # 将任务归属关系存入Redis，有效期30分钟（1800秒）
        redis_client.setex(
            AppQueueManager._generate_task_belong_cache_key(self._task_id), 1800, f"{user_prefix}-{self._user_id}"
        )

        # 创建线程安全的队列对象
        q: queue.Queue[WorkflowQueueMessage | MessageQueueMessage | None] = queue.Queue()
        self._q = q

    def listen(self):
        """监听队列消息的生成器方法
        持续从队列中获取消息，支持超时自动停止和手动停止两种方式

        Yields:
            队列中的消息对象

        Note:
            当达到最大执行时间或检测到停止标志时，会发送停止信号
            每隔10秒会发送心跳信号保持连接
        """
        # 获取应用最大执行时间配置
        listen_timeout = dify_config.APP_MAX_EXECUTION_TIME
        start_time = time.time()
        last_ping_time: int | float = 0

        while True:
            try:
                # 带超时的队列获取（1秒）
                message = self._q.get(timeout=1)
                if message is None:  # 收到停止信号
                    break

                yield message  # 生成消息
            except queue.Empty:
                continue
            finally:
                elapsed_time = time.time() - start_time
                # 超时或手动停止时处理
                if elapsed_time >= listen_timeout or self._is_stopped():
                    # 发布停止信号（发送两次确保客户端接收）
                    self.publish(
                        QueueStopEvent(stopped_by=QueueStopEvent.StopBy.USER_MANUAL), PublishFrom.TASK_PIPELINE
                    )

                # 心跳机制：每10秒发送一次
                if elapsed_time // 10 > last_ping_time:
                    self.publish(QueuePingEvent(), PublishFrom.TASK_PIPELINE)
                    last_ping_time = elapsed_time // 10

    def stop_listen(self) -> None:
        """停止监听队列（线程安全操作）
        通过向队列发送None作为停止信号
        """
        self._q.put(None)

    def publish_error(self, e, pub_from: PublishFrom) -> None:
        """发布错误事件到队列
        Args:
            e: 异常对象
            pub_from: 发布来源枚举
        """
        self.publish(QueueErrorEvent(error=e), pub_from)

    def publish(self, event: AppQueueEvent, pub_from: PublishFrom) -> None:
        """通用发布方法（入口点）
        执行安全检查后调用具体发布实现

        Args:
            event: 队列事件对象
            pub_from: 发布来源枚举
        """
        self._check_for_sqlalchemy_models(event.model_dump())  # 安全检查
        self._publish(event, pub_from)  # 调用抽象方法

    @abstractmethod
    def _publish(self, event: AppQueueEvent, pub_from: PublishFrom) -> None:
        """抽象方法：具体发布逻辑由子类实现
        Args:
            event: 队列事件对象
            pub_from: 发布来源枚举
        """
        raise NotImplementedError

    @classmethod
    def set_stop_flag(cls, task_id: str, invoke_from: InvokeFrom, user_id: str) -> None:
        """类方法：设置任务停止标志
        验证用户权限后设置Redis停止标志

        Args:
            task_id: 任务ID
            invoke_from: 调用来源
            user_id: 用户ID
        """
        # 获取任务归属信息
        result: Optional[Any] = redis_client.get(cls._generate_task_belong_cache_key(task_id))
        if result is None:
            return

        # 验证用户权限
        user_prefix = "account" if invoke_from in {InvokeFrom.EXPLORE, InvokeFrom.DEBUGGER} else "end-user"
        if result.decode("utf-8") != f"{user_prefix}-{user_id}":
            return

        # 设置停止标志（有效期10分钟）
        stopped_cache_key = cls._generate_stopped_cache_key(task_id)
        redis_client.setex(stopped_cache_key, 600, 1)

    def _is_stopped(self) -> bool:
        """内部方法：检查任务是否被停止
        通过查询Redis停止标志判断

        Returns:
            bool: 是否已停止
        """
        stopped_cache_key = AppQueueManager._generate_stopped_cache_key(self._task_id)
        result = redis_client.get(stopped_cache_key)
        return result is not None  # 存在标志即表示已停止

    @classmethod
    def _generate_task_belong_cache_key(cls, task_id: str) -> str:
        """生成任务归属缓存键
        Args:
            task_id: 任务ID
        Returns:
            Redis键字符串
        """
        return f"generate_task_belong:{task_id}"

    @classmethod
    def _generate_stopped_cache_key(cls, task_id: str) -> str:
        """生成停止标志缓存键
        Args:
            task_id: 任务ID
        Returns:
            Redis键字符串
        """
        return f"generate_task_stopped:{task_id}"

    def _check_for_sqlalchemy_models(self, data: Any):
        """递归检查数据结构中是否包含SQLAlchemy模型实例
        防止因传递ORM模型导致线程安全问题

        Args:
            data: 待检查的数据

        Raises:
            TypeError: 当检测到模型实例时抛出
        """
        # 字典类型递归检查
        if isinstance(data, dict):
            for key, value in data.items():
                self._check_for_sqlalchemy_models(value)
        # 列表类型递归检查
        elif isinstance(data, list):
            for item in data:
                self._check_for_sqlalchemy_models(item)
        # 模型实例检查
        else:
            if isinstance(data, DeclarativeMeta) or hasattr(data, "_sa_instance_state"):
                raise TypeError(
                    "Critical Error: Passing SQLAlchemy Model instances that cause thread safety issues is not allowed."
                )


class GenerateTaskStoppedError(Exception):
    pass
