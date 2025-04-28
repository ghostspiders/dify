import logging
from typing import Optional

from flask import Flask

from configs import dify_config
from dify_app import DifyApp


class Mail:
    """邮件发送服务类"""

    def __init__(self):
        # 初始化邮件客户端和默认发件人
        self._client = None
        self._default_send_from = None

    def is_inited(self) -> bool:
        """检查邮件客户端是否已初始化"""
        return self._client is not None

    def init_app(self, app: Flask):
        """初始化邮件服务"""
        # 获取配置中的邮件类型
        mail_type = dify_config.MAIL_TYPE
        if not mail_type:
            logging.warning("未设置MAIL_TYPE")
            return

        # 设置默认发件人
        if dify_config.MAIL_DEFAULT_SEND_FROM:
            self._default_send_from = dify_config.MAIL_DEFAULT_SEND_FROM

        # 根据邮件类型初始化不同的邮件客户端
        match mail_type:
            case "resend":
                import resend  # type: ignore

                # 检查Resend API密钥
                api_key = dify_config.RESEND_API_KEY
                if not api_key:
                    raise ValueError("未设置RESEND_API_KEY")

                # 设置Resend API URL（如果有配置）
                api_url = dify_config.RESEND_API_URL
                if api_url:
                    resend.api_url = api_url

                # 初始化Resend客户端
                resend.api_key = api_key
                self._client = resend.Emails

            case "smtp":
                from libs.smtp import SMTPClient

                # 检查SMTP必要配置
                if not dify_config.SMTP_SERVER or not dify_config.SMTP_PORT:
                    raise ValueError("SMTP类型需要配置SMTP_SERVER和SMTP_PORT")
                if not dify_config.SMTP_USE_TLS and dify_config.SMTP_OPPORTUNISTIC_TLS:
                    raise ValueError("未启用SMTP_USE_TLS时不能使用SMTP_OPPORTUNISTIC_TLS")

                # 初始化SMTP客户端
                self._client = SMTPClient(
                    server=dify_config.SMTP_SERVER,
                    port=dify_config.SMTP_PORT,
                    username=dify_config.SMTP_USERNAME or "",
                    password=dify_config.SMTP_PASSWORD or "",
                    _from=dify_config.MAIL_DEFAULT_SEND_FROM or "",
                    use_tls=dify_config.SMTP_USE_TLS,
                    opportunistic_tls=dify_config.SMTP_OPPORTUNISTIC_TLS,
                )
            case _:
                raise ValueError(f"不支持的邮件类型: {mail_type}")

    def send(self, to: str, subject: str, html: str, from_: Optional[str] = None):
        """发送邮件"""
        if not self._client:
            raise ValueError("邮件客户端未初始化")

        # 设置发件人（优先使用参数中的发件人，其次使用默认发件人）
        if not from_ and self._default_send_from:
            from_ = self._default_send_from

        # 参数校验
        if not from_:
            raise ValueError("未设置发件人")
        if not to:
            raise ValueError("未设置收件人")
        if not subject:
            raise ValueError("未设置邮件主题")
        if not html:
            raise ValueError("未设置邮件内容")

        # 发送邮件
        self._client.send(
            {
                "from": from_,
                "to": to,
                "subject": subject,
                "html": html,
            }
        )


def is_enabled() -> bool:
    """检查邮件服务是否启用"""
    return dify_config.MAIL_TYPE is not None and dify_config.MAIL_TYPE != ""


def init_app(app: DifyApp):
    """初始化应用的邮件服务"""
    mail.init_app(app)


# 创建全局邮件服务实例
mail = Mail()