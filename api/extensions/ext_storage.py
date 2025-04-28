import logging
from collections.abc import Callable, Generator
from typing import Literal, Union, overload

from flask import Flask

from configs import dify_config
from dify_app import DifyApp
from extensions.storage.base_storage import BaseStorage
from extensions.storage.storage_type import StorageType

logger = logging.getLogger(__name__)


class Storage:
    """存储服务管理器，支持多种云存储后端"""

    def init_app(self, app: Flask):
        """
        初始化存储服务

        参数:
            app: Flask应用实例
        """
        # 根据配置获取对应的存储工厂函数
        storage_factory = self.get_storage_factory(dify_config.STORAGE_TYPE)
        with app.app_context():
            # 初始化存储运行实例
            self.storage_runner = storage_factory()

    @staticmethod
    def get_storage_factory(storage_type: str) -> Callable[[], BaseStorage]:
        """
        根据存储类型获取对应的存储工厂函数

        参数:
            storage_type: 存储类型字符串

        返回:
            返回对应存储类型的工厂函数

        异常:
            ValueError: 当存储类型不支持时抛出
        """
        match storage_type:
            case StorageType.S3:
                from extensions.storage.aws_s3_storage import AwsS3Storage
                return AwsS3Storage

            case StorageType.OPENDAL:
                from extensions.storage.opendal_storage import OpenDALStorage
                return lambda: OpenDALStorage(dify_config.OPENDAL_SCHEME)

            case StorageType.LOCAL:
                from extensions.storage.opendal_storage import OpenDALStorage
                return lambda: OpenDALStorage(scheme="fs", root=dify_config.STORAGE_LOCAL_PATH)

            case StorageType.AZURE_BLOB:
                from extensions.storage.azure_blob_storage import AzureBlobStorage
                return AzureBlobStorage

            case StorageType.ALIYUN_OSS:
                from extensions.storage.aliyun_oss_storage import AliyunOssStorage
                return AliyunOssStorage

            case StorageType.GOOGLE_STORAGE:
                from extensions.storage.google_cloud_storage import GoogleCloudStorage
                return GoogleCloudStorage

            case StorageType.TENCENT_COS:
                from extensions.storage.tencent_cos_storage import TencentCosStorage
                return TencentCosStorage

            case StorageType.OCI_STORAGE:
                from extensions.storage.oracle_oci_storage import OracleOCIStorage
                return OracleOCIStorage

            case StorageType.HUAWEI_OBS:
                from extensions.storage.huawei_obs_storage import HuaweiObsStorage
                return HuaweiObsStorage

            case StorageType.BAIDU_OBS:
                from extensions.storage.baidu_obs_storage import BaiduObsStorage
                return BaiduObsStorage

            case StorageType.VOLCENGINE_TOS:
                from extensions.storage.volcengine_tos_storage import VolcengineTosStorage
                return VolcengineTosStorage

            case StorageType.SUPBASE:
                from extensions.storage.supabase_storage import SupabaseStorage
                return SupabaseStorage

            case _:
                raise ValueError(f"不支持的存储类型 {storage_type}")

    def save(self, filename: str, data: bytes):
        """保存文件到存储服务"""
        self.storage_runner.save(filename, data)

    @overload
    def load(self, filename: str, /, *, stream: Literal[False] = False) -> bytes:
        ...

    @overload
    def load(self, filename: str, /, *, stream: Literal[True]) -> Generator:
        ...

    def load(self, filename: str, /, *, stream: bool = False) -> Union[bytes, Generator]:
        """
        加载文件内容

        参数:
            filename: 文件名
            stream: 是否使用流式加载

        返回:
            文件内容(bytes)或生成器(流式)
        """
        if stream:
            return self.load_stream(filename)
        else:
            return self.load_once(filename)

    def load_once(self, filename: str) -> bytes:
        """一次性加载整个文件内容"""
        return self.storage_runner.load_once(filename)

    def load_stream(self, filename: str) -> Generator:
        """流式加载文件内容"""
        return self.storage_runner.load_stream(filename)

    def download(self, filename: str, target_filepath: str):
        """下载文件到本地路径"""
        self.storage_runner.download(filename, target_filepath)

    def exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        return self.storage_runner.exists(filename)

    def delete(self, filename: str) -> bool:
        """删除文件"""
        return self.storage_runner.delete(filename)


# 全局存储服务实例
storage = Storage()


def init_app(app: DifyApp):
    """初始化应用的存储服务"""
    storage.init_app(app)