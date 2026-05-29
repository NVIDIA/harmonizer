# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python interface for utils"""

from .profiling import NVTXRangeDecorator

__all__ = [
    "NVTXRangeDecorator",
]
