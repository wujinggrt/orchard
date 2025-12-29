#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global configuration manager, providing a unified interface for accessing configurations and prompts
"""

from datetime import datetime
import threading
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


def now_resolver(pattern: str):
    """Handle ${now:} time formatting"""
    return datetime.now().strftime(pattern)


OmegaConf.register_new_resolver("now", now_resolver, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


class GlobalConfig:
    """
    Global Configuration Manager (Singleton Pattern)

    Provides a unified interface for accessing configurations and prompts, avoiding the need to pass configuration objects between components.
    All components can access the configuration via GlobalConfig.get_instance().
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
        """Initialize the global configuration manager"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config: DictConfig | None = None
                    self._prompts: DictConfig | None = None

                    self._auto_initialized = False
                    GlobalConfig._initialized = True

    @classmethod
    def get_instance(cls) -> "GlobalConfig":
        """
        Get the global configuration manager instance
        """
        instance = cls()
        # If not yet initialized, try to auto-initialize
        if not instance._auto_initialized and instance._config is None:
            instance._auto_initialize()
        return instance

    def _auto_initialize(self):
        """Automatically initialize the configuration"""
        if self._auto_initialized:
            return

        try:
            # Try to load the configuration automatically

            config_path = "."

            with hydra.initialize(config_path=config_path, version_base=None):
                self._config = hydra.compose(config_name="config")
                OmegaConf.resolve(self._config)

            with hydra.initialize(config_path=config_path, version_base=None):
                self._prompts = hydra.compose(config_name="prompts")
            assert self._prompts is not None, "Prompts not initialized"
        except Exception as e:
            # logger.error(f"ERROR: GlobalConfig auto-initialization failed: {e}")
            print(f"ERROR: GlobalConfig auto-initialization failed: {e}")
            raise
        finally:
            self._auto_initialized = True

    @property
    def config(self) -> DictConfig:
        """
        Get configuration
        """
        assert self._config is not None, "Config not initialized"
        return self._config

    @property
    def prompts(self) -> DictConfig:
        """
        Get prompts
        """
        assert self._prompts, "Prompts not initialized"
        return self._prompts

    def save_config(self, *, config_path: str | None = None) -> bool:
        """
        Save configuration to a file
        """
        # TODO
        return False


def get_config(*, dot_path: str | None = None) -> DictConfig | Any:
    """
    Convenience function to get a configuration value.

    之所以参数是 path，因为配置是层级的 "module.submodule.key"
    """
    global_config = GlobalConfig.get_instance()
    config = global_config.config
    if dot_path is None:
        return config

    keys = dot_path.split(".")
    value = config
    for key in keys:
        value = value[key]
    return value


def get_prompt(*, dot_name: str) -> str:
    """Convenience function to get a prompt"""
    prompts = GlobalConfig.get_instance().prompts
    keys = dot_name.split(".")
    value = prompts
    for key in keys:
        value = value[key]
    return value


def get_prompt_group(*, dot_name: str) -> DictConfig:
    """Convenience function to get a prompt group"""
    prompts = GlobalConfig.get_instance().prompts
    keys = dot_name.split(".")
    value = prompts
    for key in keys:
        value = value[key]
    return value
