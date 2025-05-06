import logging
from typing import Optional

from core.app.entities.app_invoke_entities import InvokeFrom
from core.rag.datasource.vdb.vector_factory import Vector
from extensions.ext_database import db
from models.dataset import Dataset
from models.model import App, AppAnnotationSetting, Message, MessageAnnotation
from services.annotation_service import AppAnnotationService
from services.dataset_service import DatasetCollectionBindingService

logger = logging.getLogger(__name__)


class AnnotationReplyFeature:
    def query(
            self, app_record: App, message: Message, query: str, user_id: str, invoke_from: InvokeFrom
    ) -> Optional[MessageAnnotation]:
        """
        查询应用标注数据生成回复

        参数:
            app_record: 应用记录对象，包含应用基本信息
            message: 消息对象，当前会话消息
            query: 用户查询文本
            user_id: 用户ID
            invoke_from: 调用来源枚举

        返回:
            Optional[MessageAnnotation]: 符合条件的标注回复，未找到则返回None
        """
        # 查询应用的标注设置
        annotation_setting = (
            db.session.query(AppAnnotationSetting)
            .filter(AppAnnotationSetting.app_id == app_record.id)
            .first()
        )

        # 如果没有配置标注设置则直接返回
        if not annotation_setting:
            return None

        # 获取集合绑定详情
        collection_binding_detail = annotation_setting.collection_binding_detail

        try:
            # 设置相似度分数阈值，默认为1
            score_threshold = annotation_setting.score_threshold or 1
            # 获取嵌入模型提供商和模型名称
            embedding_provider_name = collection_binding_detail.provider_name
            embedding_model_name = collection_binding_detail.model_name

            # 获取数据集集合绑定信息
            dataset_collection_binding = DatasetCollectionBindingService.get_dataset_collection_binding(
                embedding_provider_name,
                embedding_model_name,
                "annotation"  # 绑定类型为标注
            )

            # 创建数据集对象
            dataset = Dataset(
                id=app_record.id,
                tenant_id=app_record.tenant_id,
                indexing_technique="high_quality",  # 使用高质量索引技术
                embedding_model_provider=embedding_provider_name,
                embedding_model=embedding_model_name,
                collection_binding_id=dataset_collection_binding.id,
            )

            # 创建向量搜索对象，指定需要返回的属性
            vector = Vector(dataset, attributes=["doc_id", "annotation_id", "app_id"])

            # 执行向量搜索，返回最相似的1条结果
            documents = vector.search_by_vector(
                query=query,
                top_k=1,  # 返回最相似的1条
                score_threshold=score_threshold,  # 分数阈值
                filter={"group_id": [dataset.id]}  # 按数据集ID过滤
            )

            # 如果找到匹配结果且包含元数据
            if documents and documents[0].metadata:
                annotation_id = documents[0].metadata["annotation_id"]  # 获取标注ID
                score = documents[0].metadata["score"]  # 获取相似度分数

                # 根据ID获取标注详情
                annotation = AppAnnotationService.get_annotation_by_id(annotation_id)

                if annotation:
                    # 判断调用来源
                    if invoke_from in {InvokeFrom.SERVICE_API, InvokeFrom.WEB_APP}:
                        from_source = "api"  # API调用
                    else:
                        from_source = "console"  # 控制台调用

                    # 添加标注使用历史记录
                    AppAnnotationService.add_annotation_history(
                        annotation.id,
                        app_record.id,
                        annotation.question,  # 标注问题
                        annotation.content,  # 标注内容
                        query,  # 用户实际查询
                        user_id,
                        message.id,
                        from_source,
                        score,  # 相似度分数
                    )

                    return annotation  # 返回找到的标注

        except Exception as e:
            # 记录查询失败的异常日志
            logger.warning(f"查询标注失败, 异常: {str(e)}.")
            return None

        return None  # 默认返回None