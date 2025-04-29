from collections.abc import Generator, Mapping
from typing import Any, Union

from openai._exceptions import RateLimitError

from configs import dify_config
from core.app.apps.advanced_chat.app_generator import AdvancedChatAppGenerator
from core.app.apps.agent_chat.app_generator import AgentChatAppGenerator
from core.app.apps.chat.app_generator import ChatAppGenerator
from core.app.apps.completion.app_generator import CompletionAppGenerator
from core.app.apps.workflow.app_generator import WorkflowAppGenerator
from core.app.entities.app_invoke_entities import InvokeFrom
from core.app.features.rate_limiting import RateLimit
from libs.helper import RateLimiter
from models.model import Account, App, AppMode, EndUser
from models.workflow import Workflow
from services.billing_service import BillingService
from services.errors.llm import InvokeRateLimitError
from services.workflow_service import WorkflowService


class AppGenerateService:
    # 系统级别的速率限制器，每天每个租户最多允许 APP_DAILY_RATE_LIMIT 次请求
    system_rate_limiter = RateLimiter("app_daily_rate_limiter", dify_config.APP_DAILY_RATE_LIMIT, 86400)

    @classmethod
    def generate(
            cls,
            app_model: App,
            user: Union[Account, EndUser],
            args: Mapping[str, Any],
            invoke_from: InvokeFrom,
            streaming: bool = True,
    ):
        """
        应用内容生成主入口
        :param app_model: 应用模型对象
        :param user: 用户对象，可以是账户或终端用户
        :param args: 调用参数字典
        :param invoke_from: 调用来源枚举
        :param streaming: 是否使用流式传输，默认为True
        :return: 生成的内容事件流
        """

        # 系统级速率限制检查（仅在计费启用时生效）
        if dify_config.BILLING_ENABLED:
            # 获取租户的订阅信息
            limit_info = BillingService.get_info(app_model.tenant_id)
            # 沙盒计划用户需要检查日请求限制
            if limit_info["subscription"]["plan"] == "sandbox":
                if cls.system_rate_limiter.is_rate_limited(app_model.tenant_id):
                    raise InvokeRateLimitError(
                        "请求频率超限，请升级订阅计划 "
                        f"或保持每日请求数不超过 {dify_config.APP_DAILY_RATE_LIMIT}"
                    )
                # 通过检查后增加计数器
                cls.system_rate_limiter.increment_rate_limit(app_model.tenant_id)

        # 应用级别的并发请求限制
        max_active_request = AppGenerateService._get_max_active_requests(app_model)
        rate_limit = RateLimit(app_model.id, max_active_request)  # 创建应用级限流器
        request_id = RateLimit.gen_request_key()  # 生成唯一请求ID

        try:
            request_id = rate_limit.enter(request_id)  # 进入速率限制区

            # 根据应用模式选择对应的生成器
            if app_model.mode == AppMode.COMPLETION.value:
                # 补全模式：使用Completion生成器，转换结果为事件流
                return rate_limit.generate(
                    CompletionAppGenerator.convert_to_event_stream(
                        CompletionAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id=request_id,
                )

            elif app_model.mode == AppMode.AGENT_CHAT.value or app_model.is_agent:
                # 智能体聊天模式：使用AgentChat生成器
                return rate_limit.generate(
                    AgentChatAppGenerator.convert_to_event_stream(
                        AgentChatAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id,
                )

            elif app_model.mode == AppMode.CHAT.value:
                # 基础聊天模式：使用Chat生成器
                return rate_limit.generate(
                    ChatAppGenerator.convert_to_event_stream(
                        ChatAppGenerator().generate(
                            app_model=app_model, user=user, args=args, invoke_from=invoke_from, streaming=streaming
                        ),
                    ),
                    request_id=request_id,
                )

            elif app_model.mode == AppMode.ADVANCED_CHAT.value:
                # 高级聊天模式：获取工作流后使用AdvancedChat生成器
                workflow = cls._get_workflow(app_model, invoke_from)
                return rate_limit.generate(
                    AdvancedChatAppGenerator.convert_to_event_stream(
                        AdvancedChatAppGenerator().generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                        ),
                    ),
                    request_id=request_id,
                )

            elif app_model.mode == AppMode.WORKFLOW.value:
                # 工作流模式：获取工作流后使用Workflow生成器
                workflow = cls._get_workflow(app_model, invoke_from)
                return rate_limit.generate(
                    WorkflowAppGenerator.convert_to_event_stream(
                        WorkflowAppGenerator().generate(
                            app_model=app_model,
                            workflow=workflow,
                            user=user,
                            args=args,
                            invoke_from=invoke_from,
                            streaming=streaming,
                            call_depth=0,
                            workflow_thread_pool_id=None,
                        ),
                    ),
                    request_id,
                )

            else:
                # 无效的应用模式报错
                raise ValueError(f"无效的应用模式 {app_model.mode}")

        except RateLimitError as e:
            # 捕获限流错误并转换异常类型
            raise InvokeRateLimitError(str(e))
        except Exception:
            # 其他异常发生时退出限流计数器
            rate_limit.exit(request_id)
            raise
        finally:
            # 非流式请求处理完成后立即释放计数
            if not streaming:
                rate_limit.exit(request_id)

    @staticmethod
    def _get_max_active_requests(app_model: App) -> int:
        """获取应用允许的最大并发请求数"""
        max_active_requests = app_model.max_active_requests
        # 未配置时使用默认值10
        return max_active_requests if max_active_requests is not None else 10
