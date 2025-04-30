from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from core.model_runtime.entities.llm_entities import LLMResult, LLMResultChunk
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.model_providers.__base.ai_model import AIModel

# 定义 ANSI 颜色代码映射（用于终端文本着色）
_TEXT_COLOR_MAPPING = {
    "blue": "36;1",     # 蓝色
    "yellow": "33;1",   # 黄色
    "pink": "38;5;200", # 粉色
    "green": "32;1",    # 绿色
    "red": "31;1",      # 红色
}


class Callback(ABC):
    """
    LLM 回调函数的基类（仅用于大语言模型）。
    子类必须实现所有抽象方法。
    """
    raise_error: bool = False  # 是否在出错时抛出异常

    @abstractmethod
    def on_before_invoke(
        self,
        llm_instance: AIModel,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> None:
        """
        【回调】LLM 调用前的钩子函数
        参数说明：
        - llm_instance: LLM 实例对象
        - model: 模型名称（如 "gpt-4"）
        - credentials: 模型认证信息（如 API Key）
        - prompt_messages: 输入的提示消息列表
        - model_parameters: 模型参数（如 temperature）
        - tools: 可用的工具列表（用于工具调用）
        - stop: 停止词列表（触发停止生成）
        - stream: 是否流式返回响应
        - user: 用户唯一标识
        """
        raise NotImplementedError()

    @abstractmethod
    def on_new_chunk(
        self,
        llm_instance: AIModel,
        chunk: LLMResultChunk,
        model: str,
        credentials: dict,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ):
        """
        【回调】收到流式响应分块时的钩子函数
        参数说明：
        - chunk: 流式返回的数据分块
        （其他参数同 on_before_invoke）
        """
        raise NotImplementedError()

    @abstractmethod
    def on_after_invoke(
        self,
        llm_instance: AIModel,
        result: LLMResult,
        model: str,
        credentials: dict,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> None:
        """
        【回调】LLM 调用完成后的钩子函数
        参数说明：
        - result: 完整的 LLM 返回结果
        （其他参数同 on_before_invoke）
        """
        raise NotImplementedError()

    @abstractmethod
    def on_invoke_error(
        self,
        llm_instance: AIModel,
        ex: Exception,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> None:
        """
        【回调】LLM 调用出错时的钩子函数
        参数说明：
        - ex: 异常对象
        （其他参数同 on_before_invoke）
        """
        raise NotImplementedError()

    def print_text(self, text: str, color: Optional[str] = None, end: str = "") -> None:
        """
        辅助方法：打印带颜色的文本（默认不换行）
        - text: 要打印的文本
        - color: 颜色名称（参考 _TEXT_COLOR_MAPPING）
        - end: 行尾字符（默认为空）
        """
        text_to_print = self._get_colored_text(text, color) if color else text
        print(text_to_print, end=end)

    def _get_colored_text(self, text: str, color: str) -> str:
        """
        内部方法：生成带 ANSI 颜色代码的文本
        - color_str: 从 _TEXT_COLOR_MAPPING 获取的颜色代码
        - 格式：\u001b[颜色代码m文本\u001b[0m（重置颜色）
        """
        color_str = _TEXT_COLOR_MAPPING[color]
        return f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m"
