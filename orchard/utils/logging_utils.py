# -*- coding: utf-8 -*-

"""
Logging utilities - Provides logging related functionality

如果是模块级别的日志，那么在 py 文件使用 logger = get_logger(__name__) 即可。
如果是类级别的日志，那么在类中绑定属性 self.logger = get_logger(self.__class__.__name__)。
"""

from typing import Any
from orchard.utils.logger import log_manager, log


def setup_logging(*, config: dict[str, Any]):
    """
    Setup logging

    Args:
        config (dict[str, Any]): Logging configuration
    """
    log_manager.configure(config=config)
    log.info("Logging setup completed")


def get_logger(name: str):
    """
    Get logger instance
    """
    return log.bind(name=name)
