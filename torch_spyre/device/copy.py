# Copyright 2025 The Torch-Spyre Authors.
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
# limitations under the License.

"""
Device copy operations for Spyre tensors.

This module provides copy functionality between CPU and Spyre device, as well as
device-to-device copies. The device-to-device copy is implemented using the
identity operation from the inductor backend, which creates the proper SDSC
(Super DSC) for efficient on-device copying.

Architecture:
    - Host → Device: Uses DMA operations (DMAI) via C++ copy_host_to_device
    - Device → Host: Uses DMA operations (DMAO) via C++ copy_device_to_host
    - Device → Device: Uses identity operation via inductor for on-device compute
"""

import torch
import torch_spyre._C as _C

def copy_device_to_device(src: torch.Tensor, dst: torch.Tensor) -> None:
    """
    Copy tensor from one Spyre device to another using identity operation.
    
    This function uses the inductor's identity operation to perform an efficient
    device-to-device copy. The identity operation creates the proper SDSC that
    executes the copy on the device compute units.
    
    Args:
        src: Source tensor on Spyre device
        dst: Destination tensor on Spyre device
        
    Raises:
        RuntimeError: If tensors are not on Spyre device or have incompatible shapes/dtypes
        
    Note:
        Both tensors must be on Spyre device (is_privateuseone() == True).
        The tensors must have the same shape and dtype.
    """
    # Lazy import to avoid build-time import cycle
    from torch_spyre._inductor.constants import IDENTITY_OP
    
    # Validate inputs
    if not src.is_privateuseone():
        raise RuntimeError(f"Source tensor must be on Spyre device, got {src.device}")
    if not dst.is_privateuseone():
        raise RuntimeError(f"Destination tensor must be on Spyre device, got {dst.device}")
    
    if src.shape != dst.shape:
        raise RuntimeError(
            f"Shape mismatch: source {src.shape} vs destination {dst.shape}"
        )
    
    if src.dtype != dst.dtype:
        raise RuntimeError(
            f"Dtype mismatch: source {src.dtype} vs destination {dst.dtype}"
        )
    
    @torch.compile(backend="spyre")
    def _copy_kernel(x):
        return x.clone()
    
    dst = _copy_kernel(src)


def spyre_copy_from_py(
    self: torch.Tensor, dst: torch.Tensor, non_blocking: bool
) -> torch.Tensor:
    """
    Main copy orchestration function for Spyre tensors.
    
    This function is called from C++ spyre_copy_from and routes the copy
    operation to the appropriate implementation based on source and destination
    device locations.
    
    Args:
        self: Source tensor
        dst: Destination tensor
        non_blocking: Whether to perform asynchronous copy (currently unused)
        
    Returns:
        The destination tensor
        
    Note:
        Type conversion is not supported. Source and destination must have
        the same dtype (checked in C++ before calling this function).
        For unsupported device combinations, falls back to upstream implementation.
    """
    # Handle scalar tensors (dim == 0) for host-to-device case
    if self.is_cpu() and dst.is_privateuseone():
        if self.dim() == 0:
            # Reshape scalar to 1-element tensor for DMA
            tmp_tensor = self.reshape([1])
            _C.copy_host_to_device(tmp_tensor, dst)
        else:
            _C.copy_host_to_device(self, dst)
        return dst
    
    elif self.is_privateuseone() and dst.is_cpu():
        _C.copy_device_to_host(self, dst)
        return dst
    
    elif self.is_privateuseone() and dst.is_privateuseone():
        # Device-to-device copy using identity operation
        copy_device_to_device(self, dst)
        return dst
    
    else:
        # For all other cases, fallback to the upstream implementation
        return torch.ops.aten._copy_from(self, dst, non_blocking)
