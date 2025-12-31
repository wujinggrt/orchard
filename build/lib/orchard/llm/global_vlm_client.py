#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Global LLM manager singleton wrapper
Provides global access to LLMManager instances
"""

from typing import Union, Any
from orchard.llm.llm_client import LLMClient
from orchard.utils.logging_utils import get_logger
from orchard.utils.config import omega_conf_to_dataclass
from orchard.configs.global_config import get_config
import threading
from orchard.llm.schema import Message
from openai.types.chat import ChatCompletion


logger = get_logger(__name__)


class GlobalVLMClient:
    """
    Global LLM manager (singleton pattern)

    如果需要拓展到多个 LLM 客户端，使用多个模型，则可以把 _instance 实例换成字典，
    在 YAML 配置中设置关键字为 model_name，再配置 api 等内容。例如：
    ```yaml
    llm_model:
      - model: "qwen2.5-vl-3b-instruct"
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key: "*****"
        temperature: 0.7
      - model: "qwen2.5-vl-72b-instruct"
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key: "*****"
        temperature: 0.7
    ```
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize VLM client，and metadata."""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._vlm_client: LLMClient | None = None
                    self._auto_initialized = False
                    GlobalVLMClient._initialized = True

    @classmethod
    def get_instance(cls) -> "GlobalVLMClient":
        """
        Get global LLM manager instance
        """
        instance = cls()
        if not instance._auto_initialized and instance._vlm_client is None:
            instance._auto_initialize()
        return instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def _auto_initialize(self):
        """Auto-initialize VLM client"""
        if self._auto_initialized:
            return
        # from opencontext.tools.tools_executor import ToolsExecutor
        # self._tools_executor = ToolsExecutor()
        try:
            vlm_config = get_config(dot_path="vlm_model")
            if not vlm_config:
                logger.warning("No vlm config found in vlm_model")
                self._auto_initialized = True
                return

            self._vlm_client = omega_conf_to_dataclass(vlm_config)
            # self._vlm_client = LLMClient(config=vlm_config)
            logger.info("GlobalVLMClient auto-initialized successfully")
            self._auto_initialized = True
        except Exception as e:
            logger.error(f"GlobalVLMClient auto-initialization failed: {e}")
            self._auto_initialized = True

    def is_initialized(self) -> bool:
        return self._vlm_client is not None

    def reinitialize(self):
        """
        Thread-safe reinitialization of VLM client
        """
        with self._lock:
            try:
                vlm_config = get_config("vlm_model")
                if not vlm_config:
                    logger.error("No vlm config found during reinitialize")
                    raise ValueError("No vlm config found")
                new_client = LLMClient(config=vlm_config)
                self._vlm_client = new_client
                logger.info("GlobalVLMClient reinitialized successfully")

            except Exception as e:
                logger.error(f"Failed to reinitialize VLM client: {e}")
                return False
            return True

    def get_chat_completion(
        self,
        *,
        messages: list[Union[dict, Message]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion:
        if self._vlm_client is None:
            raise RuntimeError("VLM client is not initialized.")
        response = self._vlm_client.get_chat_completion(
            messages=messages, temperature=temperature, tools=tools
        )
        return response

    async def async_get_chat_completion(
        self,
        *,
        messages: list[Union[dict, Message]],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatCompletion:
        if self._vlm_client is None:
            raise RuntimeError("VLM client is not initialized.")
        response = await self._vlm_client.async_get_chat_completion(
            messages=messages, temperature=temperature, tools=tools
        )
        return response


def is_initialized() -> bool:
    return GlobalVLMClient.get_instance()._auto_initialized


def get_chat_completion_content(
    *,
    messages: list[Union[dict, Message]],
    temperature: float = 0.7,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    response = GlobalVLMClient.get_instance().get_chat_completion(
        messages=messages, temperature=temperature, tools=tools
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty or invalid response from LLM")
    content = response.choices[0].message.content
    return content


async def async_get_chat_completion_content(
    *,
    messages: list[dict | Message],
    temperature: float = 0.7,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    response = await GlobalVLMClient.get_instance().async_get_chat_completion(
        messages=messages, temperature=temperature, tools=tools
    )
    if not response.choices or not response.choices[0].message.content:
        raise ValueError("Empty or invalid response from LLM")
    content = response.choices[0].message.content
    return content
