# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager

import torch

try:
    import torch_tensorrt

    _TRT_VERSION = tuple(int(x) for x in torch_tensorrt.__version__.split(".")[:2])
except (ImportError, AttributeError, ValueError):
    _TRT_VERSION = (999, 0)  # Assume latest version if not installed


@contextmanager
def nullcontext_decorator(*args, **kwargs):
    """Works with torch_tensorrt >= 2.9.0"""
    yield


class NullContextDecoratorCompat:
    """Class-based implementation for torch_tensorrt < 2.9.0 compatibility.

    Unlike @contextmanager decorated functions which create _GeneratorContextManager,
    this class-based implementation doesn't cause graph breaks in torch.dynamo.

    Supports both context manager usage and decorator usage:
        - with NullContextDecoratorCompat("name"): ...
        - @NullContextDecoratorCompat("name")
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, func):
        """Support usage as a decorator - just return the function unchanged."""
        return func


# Use class-based implementation for torch_tensorrt < 2.9.0 to avoid graph break
_USE_COMPAT_NULLCONTEXT = _TRT_VERSION < (2, 9)

ENABLE_NVTX_DECORATOR = os.environ.get("ENABLE_NVTX_DECORATOR", "0") == "1"

if ENABLE_NVTX_DECORATOR:
    NVTXRangeDecorator = torch.cuda.nvtx.range
elif _USE_COMPAT_NULLCONTEXT:
    NVTXRangeDecorator = NullContextDecoratorCompat
else:
    NVTXRangeDecorator = nullcontext_decorator
