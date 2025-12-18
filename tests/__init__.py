"""
Пакет тестов для лабораторных работ.
"""

from .test_cat_image import TestCatImage, TestAsyncLoggingIntegration
from .test_processor_v2 import TestAsyncProcessorV2, TestAsyncProcessorV2Sync
from .test_async_logging import TestAsyncLogging

__all__ = [
    'TestCatImage',
    'TestAsyncLoggingIntegration',
    'TestAsyncProcessor', 
    'TestAsyncProcessorSync',
    'TestAsyncProcessorV2',
    'TestAsyncProcessorV2Sync',
    'TestAsyncLogging'
]