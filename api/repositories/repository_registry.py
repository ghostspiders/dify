"""
仓库实现的注册中心模块

该模块负责向仓库工厂注册工厂函数，用于创建不同类型的仓库实现。
"""

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import sessionmaker

from configs import dify_config
from core.repository.repository_factory import RepositoryFactory
from extensions.ext_database import db
from repositories.workflow_node_execution import SQLAlchemyWorkflowNodeExecutionRepository

logger = logging.getLogger(__name__)

# 存储类型常量
STORAGE_TYPE_RDBMS = "rdbms"  # 关系型数据库存储
STORAGE_TYPE_HYBRID = "hybrid"  # 混合存储(暂未实现)


def register_repositories() -> None:
    """
    向仓库工厂注册仓库工厂函数

    根据配置设置决定注册哪种仓库实现
    """
    # 获取工作流节点执行记录的存储配置
    workflow_node_execution_storage = dify_config.WORKFLOW_NODE_EXECUTION_STORAGE

    # 根据存储类型注册对应的实现
    if workflow_node_execution_storage == STORAGE_TYPE_RDBMS:
        # 注册关系型数据库实现的工厂函数
        logger.info("正在为WorkflowNodeExecution仓库注册RDBMS存储实现")
        RepositoryFactory.register_workflow_node_execution_factory(create_workflow_node_execution_repository)
    elif workflow_node_execution_storage == STORAGE_TYPE_HYBRID:
        # 混合存储暂未实现
        raise NotImplementedError("WorkflowNodeExecution仓库的混合存储实现尚未完成")
    else:
        # 不支持的存储类型
        raise ValueError(
            f"不支持的WorkflowNodeExecution仓库存储类型'{workflow_node_execution_storage}'。"
            f"支持的存储类型: {STORAGE_TYPE_RDBMS}"
        )


def create_workflow_node_execution_repository(params: Mapping[str, Any]) -> SQLAlchemyWorkflowNodeExecutionRepository:
    """
    创建SQLAlchemy实现的WorkflowNodeExecution仓库实例

    该工厂函数用于创建RDBMS存储类型的仓库

    参数:
        params: 创建仓库所需的参数字典，包含:
            - tenant_id: 必填。多租户的租户ID
            - app_id: 可选。用于过滤的应用程序ID
            - session_factory: 可选。SQLAlchemy的sessionmaker实例。
              如果未提供，将使用全局数据库引擎创建新的sessionmaker

    返回:
        WorkflowNodeExecutionRepository实例

    异常:
        ValueError: 如果缺少必要参数
    """
    # 获取必需的租户ID参数
    tenant_id = params.get("tenant_id")
    if tenant_id is None:
        raise ValueError("使用RDBMS存储时，必须提供tenant_id参数")

    # 获取可选的app_id参数
    app_id = params.get("app_id")

    # 如果参数中提供了session_factory则使用，否则基于全局db引擎创建
    session_factory = params.get("session_factory")
    if session_factory is None:
        # 使用全局db引擎创建sessionmaker
        session_factory = sessionmaker(bind=db.engine)

    # 创建并返回仓库实例
    return SQLAlchemyWorkflowNodeExecutionRepository(
        session_factory=session_factory,  # SQLAlchemy会话工厂
        tenant_id=tenant_id,  # 租户ID
        app_id=app_id  # 应用ID(可选)
    )