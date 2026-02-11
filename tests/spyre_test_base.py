# torch_spyre/test/spyre_test_base.py

# DO NOT import common_device_type — runpy.run_path provides it via globals()

import torch
import torch_spyre  # Register the backend
import re
from torch.testing._internal.common_device_type import (DeviceTypeTestBase)

DEFAULT_FLOATING_PRECISION = 1e-3

DISABLED_TESTS = {
    "TestViewOps": {
        "test_reshape_noncontiguous_spyre",      # Known limitation
        "test_view_dtype_new",
        "test_view_dtype_upsize_errors",
        "test_view_as_complex",           # Complex not supported
        "test_view_as_real",
        "test_view_tensor_split",
        "test_view_tensor_hsplit",
        "test_view_tensor_vsplit",
        "test_view_tensor_dsplit",
        "test_imag_noncomplex",
        "test_real_imag_view",
        "test_conj_imag_view",
        "test_conj_view_with_shared_memory",
        "test_set_real_imag",
        "test_diagonal_view",
        "test_select_view",
        "test_unbind_view",
        "test_expand_view",
        "test_expand_as_view",
        "test_narrow_view",
        "test_permute_view",
        "test_transpose_view",
        "test_transpose_inplace_view",
        "test_t_view",
        "test_t_inplace_view",
        "test_T_view",
        "test_unfold_view",
        "test_squeeze_view",
        "test_squeeze_inplace_view",
        "test_unsqueeze_view",
        "test_unsqueeze_inplace_view",
        "test_as_strided_view",
        "test_as_strided_inplace_view",
        "test_view_view",
        "test_view_as_view",
        "test_contiguous_nonview",
        "test_reshape_view",
        "test_reshape_as_view",
        "test_reshape_nonview",
        "test_flatten_view",
        "test_flatten_nonview",
        "test_basic_indexing_slice_view",
        "test_basic_indexing_ellipses_view",
        "test_basic_indexing_newaxis_view",
        "test_advanced_indexing_nonview",
        "test_advanced_indexing_assignment",
        "test_chunk_view",
        "test_split_view",
        "test_movedim_view",
        "test_view_copy",
        "test_view_copy_output_contiguous",
        "test_view_copy_out",
    },
    "TestOldViewOps": {
        "test_ravel",
        "test_empty_reshape",
        "test_expand",
        "test_view_empty",
        "test_reshape",
        "test_flatten",
        "test_big_transpose",
        "test_T",
        "test_transposes",
        "test_transposes_errors",
        "test_python_types",
        "test_memory_format_resize_as",
        "test_memory_format_resize_",
        "test_transpose_invalid",
        "test_transpose_vs_numpy",
        "test_atleast",
        "test_broadcast_to",
        "test_view",
        "test_reshape_view_semantics",
        "test_contiguous",
        "test_tensor_split_sections",
        "test_tensor_split_indices",
        "test_tensor_split_errors",
        "test_resize_all_dtypes_and_devices",
        "test_resize_as_all_dtypes_and_devices",
        "test_as_strided_overflow_storage_offset",
        "test_view_all_dtypes_and_devices",
    },
    "TestCommon": {
        "test_compare_cpu_cholesky_.*",    # Decomposition not implemented
    },
}

PRECISION_OVERRIDES = {
    "test_sum": 1e-2,
    "test_softmax": 1e-3,
    "test_batch_norm": 1e-1,
}

class SpyreTestBase(DeviceTypeTestBase):  # DeviceTypeTestBase available via globals()
    device_type = 'spyre'
    precision = DEFAULT_FLOATING_PRECISION
    
    unsupported_dtypes = {
        torch.complex32, torch.complex64, torch.complex128,
    }
    
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls):
        test_name = name + '_' + cls.device_type
        class_name = generic_cls.__name__
        
        # Check disabled tests
        if class_name in DISABLED_TESTS:
            disabled = DISABLED_TESTS[class_name]
            for pattern in disabled:
                if re.match(pattern, test_name) or re.match(pattern, name):
                    @wraps(test)
                    def skipped_test(self, test=test):
                        raise unittest.SkipTest('skipped on Spyre')
                        setattr(cls, test_name, skipped_test)
                        return
        
        # Apply precision overrides
        if name in PRECISION_OVERRIDES:
            cls.precision = PRECISION_OVERRIDES[name]
        else:
            cls.precision = DEFAULT_FLOATING_PRECISION
        
        # Delegate to parent for actual instantiation
        super().instantiate_test(name, test, generic_cls=generic_cls)


TEST_CLASS = SpyreTestBase