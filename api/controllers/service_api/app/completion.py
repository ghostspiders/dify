import logging

from flask_restful import Resource, reqparse  # type: ignore
from werkzeug.exceptions import InternalServerError, NotFound

import services
from controllers.service_api import api
from controllers.service_api.app.error import (
    AppUnavailableError,
    CompletionRequestError,
    ConversationCompletedError,
    NotChatAppError,
    ProviderModelCurrentlyNotSupportError,
    ProviderNotInitializeError,
    ProviderQuotaExceededError,
)
from controllers.service_api.wraps import FetchUserArg, WhereisUserArg, validate_app_token
from controllers.web.error import InvokeRateLimitError as InvokeRateLimitHttpError
from core.app.apps.base_app_queue_manager import AppQueueManager
from core.app.entities.app_invoke_entities import InvokeFrom
from core.errors.error import (
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from core.model_runtime.errors.invoke import InvokeError
from libs import helper
from libs.helper import uuid_value
from models.model import App, AppMode, EndUser
from services.app_generate_service import AppGenerateService
from services.errors.llm import InvokeRateLimitError


class CompletionApi(Resource):
    """文本补全API资源类"""

    @validate_app_token(fetch_user_arg=FetchUserArg(fetch_from=WhereisUserArg.JSON, required=True))
    def post(self, app_model: App, end_user: EndUser):
        """
        处理文本补全请求
        :param app_model: 应用模型实例
        :param end_user: 终端用户实例
        :return: 生成结果或错误响应
        """
        # 验证应用模式
        if app_model.mode != "completion":
            raise AppUnavailableError()

        # 解析请求参数
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", type=dict, required=True, location="json")  # 输入参数
        parser.add_argument("query", type=str, location="json", default="")  # 查询文本
        parser.add_argument("files", type=list, required=False, location="json")  # 文件列表
        parser.add_argument("response_mode", type=str, choices=["blocking", "streaming"], location="json")  # 响应模式
        parser.add_argument("retriever_from", type=str, required=False, default="dev", location="json")  # 检索来源

        args = parser.parse_args()
        streaming = args["response_mode"] == "streaming"  # 是否流式响应
        args["auto_generate_name"] = False  # 禁用自动生成名称

        try:
            # 调用生成服务
            response = AppGenerateService.generate(
                app_model=app_model,
                user=end_user,
                args=args,
                invoke_from=InvokeFrom.SERVICE_API,  # 调用来源标记
                streaming=streaming,
            )
            return helper.compact_generate_response(response)  # 返回格式化响应
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logging.exception("App model config broken.")
            raise AppUnavailableError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise e
        except Exception:
            logging.exception("internal server error.")
            raise InternalServerError()


class CompletionStopApi(Resource):
    """停止文本补全任务API资源类"""

    @validate_app_token(fetch_user_arg=FetchUserArg(fetch_from=WhereisUserArg.JSON, required=True))
    def post(self, app_model: App, end_user: EndUser, task_id):
        """
        停止正在进行的补全任务
        :param app_model: 应用模型实例
        :param end_user: 终端用户实例
        :param task_id: 任务ID
        :return: 操作结果
        """
        if app_model.mode != "completion":
            raise AppUnavailableError()

        # 设置任务停止标志
        AppQueueManager.set_stop_flag(task_id, InvokeFrom.SERVICE_API, end_user.id)
        return {"result": "success"}, 200


class ChatApi(Resource):
    """聊天API资源类"""

    @validate_app_token(fetch_user_arg=FetchUserArg(fetch_from=WhereisUserArg.JSON, required=True))
    def post(self, app_model: App, end_user: EndUser):
        """
        处理聊天请求
        :param app_model: 应用模型实例
        :param end_user: 终端用户实例
        :return: 生成结果或错误响应
        """
        # 验证应用模式是否为聊天类型
        app_mode = AppMode.value_of(app_model.mode)
        if app_mode not in {AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.ADVANCED_CHAT}:
            raise NotChatAppError()

        # 解析请求参数
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", type=dict, required=True, location="json")  # 输入参数
        parser.add_argument("query", type=str, required=True, location="json")  # 查询文本
        parser.add_argument("files", type=list, required=False, location="json")  # 文件列表
        parser.add_argument("response_mode", type=str, choices=["blocking", "streaming"], location="json")  # 响应模式
        parser.add_argument("conversation_id", type=uuid_value, location="json")  # 会话ID
        parser.add_argument("retriever_from", type=str, required=False, default="dev", location="json")  # 检索来源
        parser.add_argument("auto_generate_name", type=bool, required=False, default=True, location="json")  # 自动命名

        args = parser.parse_args()
        streaming = args["response_mode"] == "streaming"  # 是否流式响应

        try:
            # 调用生成服务
            response = AppGenerateService.generate(
                app_model=app_model,
                user=end_user,
                args=args,
                invoke_from=InvokeFrom.SERVICE_API,
                streaming=streaming
            )
            return helper.compact_generate_response(response)  # 返回格式化响应
        except services.errors.conversation.ConversationNotExistsError:
            raise NotFound("Conversation Not Exists.")
        except services.errors.conversation.ConversationCompletedError:
            raise ConversationCompletedError()
        except services.errors.app_model_config.AppModelConfigBrokenError:
            logging.exception("App model config broken.")
            raise AppUnavailableError()
        except ProviderTokenNotInitError as ex:
            raise ProviderNotInitializeError(ex.description)
        except QuotaExceededError:
            raise ProviderQuotaExceededError()
        except ModelCurrentlyNotSupportError:
            raise ProviderModelCurrentlyNotSupportError()
        except InvokeRateLimitError as ex:
            raise InvokeRateLimitHttpError(ex.description)
        except InvokeError as e:
            raise CompletionRequestError(e.description)
        except ValueError as e:
            raise e
        except Exception:
            logging.exception("internal server error.")
            raise InternalServerError()


class ChatStopApi(Resource):
    """停止聊天任务API资源类"""

    @validate_app_token(fetch_user_arg=FetchUserArg(fetch_from=WhereisUserArg.JSON, required=True))
    def post(self, app_model: App, end_user: EndUser, task_id):
        """
        停止正在进行的聊天任务
        :param app_model: 应用模型实例
        :param end_user: 终端用户实例
        :param task_id: 任务ID
        :return: 操作结果
        """
        app_mode = AppMode.value_of(app_model.mode)
        if app_mode not in {AppMode.CHAT, AppMode.AGENT_CHAT, AppMode.ADVANCED_CHAT}:
            raise NotChatAppError()

        # 设置任务停止标志
        AppQueueManager.set_stop_flag(task_id, InvokeFrom.SERVICE_API, end_user.id)
        return {"result": "success"}, 200


# 注册API路由
api.add_resource(CompletionApi, "/completion-messages")  # 文本补全接口
api.add_resource(CompletionStopApi, "/completion-messages/<string:task_id>/stop")  # 停止补全任务接口
api.add_resource(ChatApi, "/chat-messages")  # 聊天接口
api.add_resource(ChatStopApi, "/chat-messages/<string:task_id>/stop")  # 停止聊天任务接口