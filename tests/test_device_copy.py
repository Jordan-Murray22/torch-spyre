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
Tests for device-to-device copy functionality using identity operation.
"""

import pytest
import torch


class TestDeviceToDeviceCopy:
    """Test suite for device-to-device copy operations."""
    
    def test_device_to_device(self):
        """Test device-to-device copy using tensor.copy_() method."""
        # Create source tensor

        src = torch.randn(3, dtype=torch.float16, device="spyre")
        dst = torch.empty(3, dtype=torch.float16, device="spyre")
        print(f"src: {src}")
        print(f"dst before: {dst}")
        dst.copy_(src)
        print(f"dst after: {dst}")
        
        # Verify the copy worked
        assert torch.allclose(src,dst)

    def test_host_to_device(self):
        """Test that host-to-device copy still works after changes."""
        src = torch.randn(10, 20)
        dst = torch.empty(10, 20, device='spyre')
        
        dst.copy_(src)
        
        assert torch.allclose(src, dst.cpu())

    def test_device_to_host(self):
        """Test that device-to-host copy still works after changes."""
        src = torch.randn(10, 20, device='spyre')
        dst = torch.empty(10, 20)
        
        dst.copy_(src)
        
        assert torch.allclose(src.cpu(), dst)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
