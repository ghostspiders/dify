from typing import Any, Union

import redis
from redis.cluster import ClusterNode, RedisCluster
from redis.connection import Connection, SSLConnection
from redis.sentinel import Sentinel

from configs import dify_config
from dify_app import DifyApp


class RedisClientWrapper:
    """
    Redis客户端包装类，解决当Sentinel返回新的Redis实例时，
    全局`redis_client`变量无法更新的问题。

    该类支持延迟初始化Redis客户端，允许在必要时重新初始化新的Redis实例。
    这在Redis实例可能动态变化的场景中特别有用，例如在Sentinel管理的Redis设置中发生故障转移时。

    属性:
        _client (redis.Redis): 实际的Redis客户端实例。在调用`initialize`方法前保持为None。

    方法:
        initialize(client): 如果Redis客户端尚未初始化，则进行初始化
        __getattr__(item): 将属性访问委托给Redis客户端，如果客户端未初始化则抛出错误
    """

    def __init__(self):
        """初始化Redis客户端包装器"""
        self._client = None  # 实际的Redis客户端实例

    def initialize(self, client):
        """初始化Redis客户端"""
        if self._client is None:
            self._client = client

    def __getattr__(self, item):
        """委托所有属性/方法访问到实际的Redis客户端"""
        if self._client is None:
            raise RuntimeError("Redis客户端未初始化，请先调用init_app")
        return getattr(self._client, item)


# 全局Redis客户端实例
redis_client = RedisClientWrapper()


def init_app(app: DifyApp):
    """初始化Redis连接"""
    global redis_client

    # 1. 确定连接类(普通连接或SSL连接)
    connection_class: type[Union[Connection, SSLConnection]] = Connection
    if dify_config.REDIS_USE_SSL:
        connection_class = SSLConnection

    # 2. 准备基本Redis连接参数
    redis_params: dict[str, Any] = {
        "username": dify_config.REDIS_USERNAME,
        "password": dify_config.REDIS_PASSWORD or None,  # 空密码的临时解决方案
        "db": dify_config.REDIS_DB,
        "encoding": "utf-8",
        "encoding_errors": "strict",
        "decode_responses": False,  # 保持原始字节数据
    }

    # 3. 根据配置选择Redis连接模式
    if dify_config.REDIS_USE_SENTINEL:
        # 哨兵模式配置
        assert dify_config.REDIS_SENTINELS is not None, "启用哨兵模式必须设置REDIS_SENTINELS"

        # 解析哨兵节点配置
        sentinel_hosts = [
            (node.split(":")[0], int(node.split(":")[1]))
            for node in dify_config.REDIS_SENTINELS.split(",")
        ]

        # 创建哨兵客户端
        sentinel = Sentinel(
            sentinel_hosts,
            sentinel_kwargs={
                "socket_timeout": dify_config.REDIS_SENTINEL_SOCKET_TIMEOUT,
                "username": dify_config.REDIS_SENTINEL_USERNAME,
                "password": dify_config.REDIS_SENTINEL_PASSWORD,
            },
        )

        # 获取主节点客户端
        master = sentinel.master_for(dify_config.REDIS_SENTINEL_SERVICE_NAME, **redis_params)
        redis_client.initialize(master)

    elif dify_config.REDIS_USE_CLUSTERS:
        # 集群模式配置
        assert dify_config.REDIS_CLUSTERS is not None, "启用集群模式必须设置REDIS_CLUSTERS"

        # 解析集群节点配置
        nodes = [
            ClusterNode(host=node.split(":")[0], port=int(node.split(":")[1]))
            for node in dify_config.REDIS_CLUSTERS.split(",")
        ]

        # 创建集群客户端(忽略mypy类型检查错误)
        redis_client.initialize(RedisCluster(
            startup_nodes=nodes,
            password=dify_config.REDIS_CLUSTERS_PASSWORD
        ))  # type: ignore

    else:
        # 单节点模式配置
        redis_params.update({
            "host": dify_config.REDIS_HOST,
            "port": dify_config.REDIS_PORT,
            "connection_class": connection_class,
        })

        # 创建连接池和Redis客户端
        pool = redis.ConnectionPool(**redis_params)
        redis_client.initialize(redis.Redis(connection_pool=pool))

    # 将Redis客户端注册到Flask扩展中
    app.extensions["redis"] = redis_client
