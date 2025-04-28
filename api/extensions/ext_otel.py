import atexit
import logging
import os
import platform
import socket
import sys
from typing import Union

from celery.signals import worker_init  # type: ignore
from flask_login import user_loaded_from_request, user_logged_in  # type: ignore
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.metrics import get_meter, get_meter_provider, set_meter_provider
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3Format
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Span, get_current_span, get_tracer_provider, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.status import StatusCode

from configs import dify_config
from dify_app import DifyApp


@user_logged_in.connect
@user_loaded_from_request.connect
def on_user_loaded(_sender, user):
    """用户登录/加载时的处理函数"""
    if user:
        # 获取当前跟踪span
        current_span = get_current_span()
        if current_span:
            # 在span中设置租户ID和用户ID属性
            current_span.set_attribute("service.tenant.id", user.current_tenant_id)
            current_span.set_attribute("service.user.id", user.id)


def init_app(app: DifyApp):
    """初始化OpenTelemetry应用监控"""
    if dify_config.ENABLE_OTEL:
        # 1. 设置上下文传播
        setup_context_propagation()

        # 2. 创建资源描述(遵循OpenTelemetry语义约定1.32.0)
        resource = Resource(
            attributes={
                ResourceAttributes.SERVICE_NAME: dify_config.APPLICATION_NAME,  # 服务名称
                ResourceAttributes.SERVICE_VERSION: f"dify-{dify_config.CURRENT_VERSION}-{dify_config.COMMIT_SHA}",
                # 服务版本
                ResourceAttributes.PROCESS_PID: os.getpid(),  # 进程ID
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: f"{dify_config.DEPLOY_ENV}-{dify_config.EDITION}",  # 部署环境
                ResourceAttributes.HOST_NAME: socket.gethostname(),  # 主机名
                ResourceAttributes.HOST_ARCH: platform.machine(),  # 主机架构
                "custom.deployment.git_commit": dify_config.COMMIT_SHA,  # Git提交哈希
                ResourceAttributes.HOST_ID: platform.node(),  # 主机ID
                ResourceAttributes.OS_TYPE: platform.system().lower(),  # 操作系统类型
                ResourceAttributes.OS_DESCRIPTION: platform.platform(),  # 操作系统描述
                ResourceAttributes.OS_VERSION: platform.version(),  # 操作系统版本
            }
        )

        # 3. 配置采样率
        sampler = ParentBasedTraceIdRatio(dify_config.OTEL_SAMPLING_RATE)
        provider = TracerProvider(resource=resource, sampler=sampler)
        set_tracer_provider(provider)

        # 4. 配置导出器(OTLP或控制台)
        exporter: Union[OTLPSpanExporter, ConsoleSpanExporter]
        metric_exporter: Union[OTLPMetricExporter, ConsoleMetricExporter]
        if dify_config.OTEL_EXPORTER_TYPE == "otlp":
            # OTLP导出器配置
            exporter = OTLPSpanExporter(
                endpoint=dify_config.OTLP_BASE_ENDPOINT + "/v1/traces",
                headers={"Authorization": f"Bearer {dify_config.OTLP_API_KEY}"},
            )
            metric_exporter = OTLPMetricExporter(
                endpoint=dify_config.OTLP_BASE_ENDPOINT + "/v1/metrics",
                headers={"Authorization": f"Bearer {dify_config.OTLP_API_KEY}"},
            )
        else:
            # 控制台导出器(开发环境使用)
            exporter = ConsoleSpanExporter()
            metric_exporter = ConsoleMetricExporter()

        # 5. 配置批量span处理器
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=dify_config.OTEL_MAX_QUEUE_SIZE,
                schedule_delay_millis=dify_config.OTEL_BATCH_EXPORT_SCHEDULE_DELAY,
                max_export_batch_size=dify_config.OTEL_MAX_EXPORT_BATCH_SIZE,
                export_timeout_millis=dify_config.OTEL_BATCH_EXPORT_TIMEOUT,
            )
        )

        # 6. 配置指标导出
        reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=dify_config.OTEL_METRIC_EXPORT_INTERVAL,
            export_timeout_millis=dify_config.OTEL_METRIC_EXPORT_TIMEOUT,
        )
        set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))

        # 7. 初始化应用监控
        if not is_celery_worker():
            # 非Celery worker初始化Flask监控
            init_flask_instrumentor(app)
            # 初始化Celery监控
            CeleryInstrumentor(tracer_provider=get_tracer_provider(), meter_provider=get_meter_provider()).instrument()

        # 8. 初始化SQLAlchemy监控
        init_sqlalchemy_instrumentor(app)

        # 9. 注册退出时的清理函数
        atexit.register(shutdown_tracer)


def is_celery_worker():
    """检查当前进程是否是Celery worker"""
    return "celery" in sys.argv[0].lower()


def init_flask_instrumentor(app: DifyApp):
    """初始化Flask应用监控"""
    # 创建HTTP指标计量器
    meter = get_meter("http_metrics", version=dify_config.CURRENT_VERSION)
    _http_response_counter = meter.create_counter(
        "http.server.response.count",
        description="HTTP响应总数(按状态码统计)",
        unit="{response}"
    )

    def response_hook(span: Span, status: str, response_headers: list):
        """响应钩子函数，用于处理HTTP响应"""
        if span and span.is_recording():
            # 设置span状态
            if status.startswith("2"):
                span.set_status(StatusCode.OK)
            else:
                span.set_status(StatusCode.ERROR, status)

            # 记录HTTP响应指标
            status = status.split(" ")[0]
            status_code = int(status)
            status_class = f"{status_code // 100}xx"
            _http_response_counter.add(1, {"status_code": status_code, "status_class": status_class})

    # 初始化Flask监控器
    instrumentor = FlaskInstrumentor()
    if dify_config.DEBUG:
        logging.info("正在初始化Flask监控器")
    instrumentor.instrument_app(app, response_hook=response_hook)


def init_sqlalchemy_instrumentor(app: DifyApp):
    """初始化SQLAlchemy监控"""
    with app.app_context():
        # 获取所有SQLAlchemy引擎实例
        engines = list(app.extensions["sqlalchemy"].engines.values())
        SQLAlchemyInstrumentor().instrument(enable_commenter=True, engines=engines)


def setup_context_propagation():
    """设置上下文传播方式"""
    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),  # W3C标准跟踪上下文
                B3Format(),  # B3传播格式(被许多系统使用)
            ]
        )
    )


@worker_init.connect(weak=False)
def init_celery_worker(*args, **kwargs):
    """初始化Celery worker监控"""
    tracer_provider = get_tracer_provider()
    metric_provider = get_meter_provider()
    if dify_config.DEBUG:
        logging.info("正在为Celery worker初始化OpenTelemetry")
    CeleryInstrumentor(tracer_provider=tracer_provider, meter_provider=metric_provider).instrument()


def shutdown_tracer():
    """关闭跟踪器时的清理函数"""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()  # 强制刷新所有未导出的数据