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

@torch.library.impl("aten::_copy_from", "PrivateUse1")
def spyre_copy_from(self, dst, non_blocking=False):
    # Implement PyTorch _copy_from semantics:
    # 1. Check if views of same data
    if self.data_ptr() == dst.data_ptr() and self.storage_offset() == dst.storage_offset():
        return self
    
    # 2. Check if numel is 0
    if self.numel() == 0:
        return self
    
    # 3. Do the real copy
    if self.device.type == "cpu" and dst.device.type == "spyre":
        return _C.copy_host_to_device(self, dst)
    elif self.device.type == "spyre" and dst.device.type == "cpu":
        return _C.copy_device_to_host(self, dst)
    elif self.device.type == "spyre" and self.device == dst.device:
        # Device-to-device copy: use torch.ops.spyre.copy which preserves layout
        @torch.compile(dynamic=False)
        def _copy_kernel(x):
            # Use spyre.copy which preserves input's SpyreTensorLayout
            return torch.ops.spyre.copy(x)
        
        # Execute and assign to dst's storage
        dst.data = _copy_kernel(self).data
        return dst
    else:
        return torch.ops.aten._copy_from.default(self, dst, non_blocking)