#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entity normalization tool base class
"""

from typing import Any
from abc import ABC, abstractmethod
from orchard.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTool(ABC):
    """Base class for entity tools"""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get tool name"""
        pass

    @classmethod
    @abstractmethod
    def get_description(cls) -> str:
        """Get tool description"""
        pass

    @classmethod
    @abstractmethod
    def get_parameters(cls) -> dict[str, Any]:
        """Get tool parameters definition"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any]:
        """Execute tool operation"""
        pass

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        """Get tool definition for LLM calls"""
        return dict(
            name=cls.get_name(),
            description=cls.get_description(),
            parameters=cls.get_parameters(),
        )
