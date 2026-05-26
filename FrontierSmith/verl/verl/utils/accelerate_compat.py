# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations in the License.

"""
Compatibility patch for transformers 5.x + accelerate.

Transformers 5.x attaches _is_hf_initialized to parameters. When accelerate's
register_empty_parameter passes **kwargs to Parameter.__new__(), PyTorch rejects
this unknown argument. This patch pops _is_hf_initialized before the Parameter
is created.

See: https://github.com/huggingface/transformers/issues/44291
     https://github.com/huggingface/accelerate/pull/3943
"""

import importlib.metadata
from contextlib import contextmanager
from typing import Optional

from packaging import version


def _patch_accelerate_init_on_device():
    """Patch accelerate.big_modeling.init_on_device to fix _is_hf_initialized compatibility."""
    import torch
    import torch.nn as nn

    import accelerate.big_modeling as bm
    from accelerate.utils import parse_flag_from_env

    old_init_on_device = bm.init_on_device

    @contextmanager
    def patched_init_on_device(device: torch.device, include_buffers: Optional[bool] = None):
        if include_buffers is None:
            include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

        if include_buffers:
            with device:
                yield
            return

        old_register_parameter = nn.Module.register_parameter
        if include_buffers:
            old_register_buffer = nn.Module.register_buffer

        def register_empty_parameter(module, name, param):
            old_register_parameter(module, name, param)
            if param is not None:
                param_cls = type(module._parameters[name])
                kwargs = module._parameters[name].__dict__
                kwargs.pop("_is_hf_initialized", None)  # transformers 5.x compatibility
                kwargs["requires_grad"] = param.requires_grad
                module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

        def register_empty_buffer(module, name, buffer, persistent=True):
            old_register_buffer(module, name, buffer, persistent=persistent)
            if buffer is not None:
                module._buffers[name] = module._buffers[name].to(device)

        if include_buffers:
            tensor_constructors_to_patch = {
                torch_function_name: getattr(torch, torch_function_name)
                for torch_function_name in ["empty", "zeros", "ones", "full"]
            }
        else:
            tensor_constructors_to_patch = {}

        def patch_tensor_constructor(fn):
            def wrapper(*args, **kwargs):
                kwargs["device"] = device
                return fn(*args, **kwargs)

            return wrapper

        try:
            nn.Module.register_parameter = register_empty_parameter
            if include_buffers:
                nn.Module.register_buffer = register_empty_buffer
            for torch_function_name in tensor_constructors_to_patch.keys():
                setattr(
                    torch,
                    torch_function_name,
                    patch_tensor_constructor(getattr(torch, torch_function_name)),
                )
            yield
        finally:
            nn.Module.register_parameter = old_register_parameter
            if include_buffers:
                nn.Module.register_buffer = old_register_buffer
            for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
                setattr(torch, torch_function_name, old_torch_function)

    bm.init_on_device = patched_init_on_device


def apply_accelerate_transformers5_compat():
    """Apply compatibility patch if using transformers >= 5.0 with accelerate."""
    try:
        tf_version = version.parse(importlib.metadata.version("transformers"))
    except importlib.metadata.PackageNotFoundError:
        return

    if tf_version >= version.parse("5.0.0"):
        _patch_accelerate_init_on_device()
