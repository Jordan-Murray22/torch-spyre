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

import torch
import torch_spyre._C as _C


def copy_device_to_device(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Copy tensor from one Spyre device to another using identity op.

    This function uses the inductor's identity op to perform device-to-device copy.
    The identity operation creates the proper SDSC that executes the copy on the
    device compute units.

    Args:
        src: Source tensor on Spyre device
        dst: Destination tensor on Spyre device

    Raises:
        RuntimeError: If tensors are not on Spyre device or have incompatible shapes/dtypes
    """

    # Validate inputs - check if tensors are on Spyre device and the same shape & dtype
    if src.device.type != "spyre":
        raise RuntimeError(f"Source tensor must be on Spyre device, got {src.device}")
    if dst.device.type != "spyre":
        raise RuntimeError(
            f"Destination tensor must be on Spyre device, got {dst.device}"
        )

    if src.shape != dst.shape:
        raise RuntimeError(
            f"Shape mismatch: source {src.shape} vs destination {dst.shape}"
        )

    if src.dtype != dst.dtype:
        raise RuntimeError(
            f"Dtype mismatch: source {src.dtype} vs destination {dst.dtype}"
        )

    @torch.compile
    def _copy_kernel(x):
        return x.detach().clone()

    dst.data = _copy_kernel(src).data
    return dst


def spyre_copy_from_py(
    self: torch.Tensor, dst: torch.Tensor, non_blocking: bool
) -> torch.Tensor:
    """
    Main copy orchestration function for Spyre tensors.

    This function is called from C++ spyre_copy_from and routes the copy
    operation to the appropriate implementation based on source and destination
    device locations.

    Architecture:
    - Host → Device: Uses DMA operations (DMAI) via C++ copy_host_to_device
    - Device → Host: Uses DMA operations (DMAO) via C++ copy_device_to_host
    - Device → Device: Uses identity operation via inductor for on-device compute

    Args:
        self: Source tensor
        dst: Destination tensor
        non_blocking: Whether to perform asynchronous copy (currently unused)

    Returns:
        The destination tensor
    """
    src_is_cpu = self.device.type == "cpu"
    dst_is_cpu = dst.device.type == "cpu"
    src_is_spyre = self.device.type == "spyre"
    dst_is_spyre = dst.device.type == "spyre"

    if src_is_cpu and dst_is_spyre:
        if self.dim() == 0:
            # Reshape scalar to 1-element tensor for DMA
            tmp_tensor = self.reshape([1])
            _C.copy_host_to_device(tmp_tensor, dst)
        else:
            _C.copy_host_to_device(self, dst)
        return dst

    elif src_is_spyre and dst_is_cpu:
        _C.copy_device_to_host(self, dst)
        return dst

    elif src_is_spyre and dst_is_spyre:
        # Device-to-device copy using identity operation
        dst = copy_device_to_device(self, dst)
        return dst

    else:
        # For all other cases, fallback to the upstream implementation
        return torch.ops.aten._copy_from(self, dst, non_blocking)
        # Unsupported copy operation
        raise RuntimeError(
            f"Unsupported copy operation from {self.device} to {dst.device}"
        )
