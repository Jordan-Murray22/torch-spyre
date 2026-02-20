import torch
import re
import unittest
from functools import wraps

DEFAULT_FLOATING_PRECISION = 1e-3

DISABLED_TESTS = {
    "TestViewOps": {  # Verifies that view-related tensor operations (ex. view, reshape, squeeze/unsqueeze, etc.) produce correct shapes, preserve data, and follow PyTorch’s current aliasing and memory-sharing semantics.
        "test_view_dtype_new",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/648
        "test_view_dtype_upsize_errors",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/650
        "test_view_as_complex",  # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651 & Complex Not supported
        # "test_view_as_real", # Caught with Unsupported dtype filter https://github.com/torch-spyre/torch-spyre/issues/652
        "test_view_tensor_split",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/653
        "test_view_tensor_hsplit",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/674
        "test_view_tensor_vsplit",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/675
        "test_view_tensor_dsplit",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/676
        "test_imag_noncomplex",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/677
        # "test_real_imag_view", # Caught with Unsupported dtype filter https://github.com/torch-spyre/torch-spyre/issues/694
        # "test_conj_imag_view", # Caught with Unsupported dtype filter https://github.com/torch-spyre/torch-spyre/issues/697
        "test_conj_view_with_shared_memory",  # Signal Received: 8 (Floating point exception) https://github.com/torch-spyre/torch-spyre/issues/678
        # "test_set_real_imag", # Caught with Unsupported dtype filter https://github.com/torch-spyre/torch-spyre/issues/700
        "test_diagonal_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_select_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_unbind_view",  # NotImplementedError aten::zero_ https://github.com/torch-spyre/torch-spyre/issues/630
        "test_expand_view",  # RuntimeError: aten::expand() takes 2 positional argument(s) but 3 was/were given https://github.com/torch-spyre/torch-spyre/issues/606
        "test_expand_as_view",  # RuntimeError: aten::expand() takes 2 positional argument(s) but 3 was/were given https://github.com/torch-spyre/torch-spyre/issues/606
        "test_narrow_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_permute_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_transpose_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_transpose_inplace_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_t_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_t_inplace_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_T_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_unfold_view",  # NotImplementedError aten::unfold https://github.com/torch-spyre/torch-spyre/issues/688
        "test_squeeze_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_squeeze_inplace_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_unsqueeze_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_unsqueeze_inplace_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_as_strided_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_as_strided_inplace_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_view_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_view_as_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_contiguous_nonview",  # Failing with corrupted double-linked list https://github.com/torch-spyre/torch-spyre/issues/689
        "test_reshape_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_reshape_as_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_reshape_nonview",  # Failing with smallbin double linked list corrupted https://github.com/torch-spyre/torch-spyre/issues/690
        "test_flatten_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_flatten_nonview",  # Signal Received: 6 corrupted size vs. prev_size https://github.com/torch-spyre/torch-spyre/issues/691
        "test_basic_indexing_slice_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_basic_indexing_ellipses_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_basic_indexing_newaxis_view",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_advanced_indexing_nonview",  # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_advanced_indexing_assignment",  # NotImplementedError aten::_index_put_impl_ https://github.com/torch-spyre/torch-spyre/issues/692
        "test_movedim_view",  # NotImplementedError aten::zero_ https://github.com/torch-spyre/torch-spyre/issues/630
        "test_view_copy",  # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_view_copy_output_contiguous",  # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_view_copy_out",  # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
    },
    "TestOldViewOps": { # Ensures backward compatibility by testing legacy view behaviors and edge cases from older PyTorch implementations
        "test_ravel", # NotImplementedError as_strided https://github.com/torch-spyre/torch-spyre/issues/687
        "test_empty_reshape", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_expand", # NotImplementedError aten::uniform_ 
        "test_view_empty", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_reshape", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_flatten", #AssertionError: The length of the sequences mismatch: 1 != 0 
        "test_big_transpose", # NotImplementedError aten::uniform_ 
        "test_T", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_transposes", # Signal Received: 8 (Floating point exception)
        "test_transposes_errors", # Signal Received: 8 (Floating point exception)
        "test_python_types", # Signal Received: 8 (Floating point exception)
        "test_memory_format_resize_as", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_memory_format_resize_", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_transpose_invalid", # NotImplementedError aten::random_.from
        "test_transpose_vs_numpy", # Signal Received: 11 (Segmentation fault)
        "test_atleast", # Signal Received: 11 (Segmentation fault)
        "test_broadcast_to", # NotImplementedError aten::random_.from and aten::uniform_ 
        "test_view", # NotImplementedError aten::uniform_ 
        "test_reshape_view_semantics", # Signal Received: 8 (Floating point exception)
        "test_contiguous", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_tensor_split_sections", # Signal Received: 8 (Floating point exception)
        "test_tensor_split_indices", # Signal Received: 8 (Floating point exception)
        "test_tensor_split_errors", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_resize_all_dtypes_and_devices", # NotImplementedError aten::resize_ 
        "test_resize_as_all_dtypes_and_devices", # NotImplementedError aten::resize_ 
        "test_as_strided_overflow_storage_offset", # NotImplementedError aten::normal_ https://github.com/torch-spyre/torch-spyre/issues/651
        "test_view_all_dtypes_and_devices", # Signal Received: 8 (Floating point exception) 
    },
}

# TODO: Add overrides for specific cases when we need them
PRECISION_OVERRIDES = {}  # type: ignore[var-annotated]


# Match infrastructure — adapted from pytorch/xla pytorch_test_base.py
class MatchSet(object):
    def __init__(self):
        self.exact = set()
        self.regex = set()


# Converts a dict of lists into a dict of MatchSet objects, categorizing each match string as either an exact match (alphanumeric) or a regex pattern.
def prepare_match_set(s):
    ps = dict()
    for k, v in s.items():
        mset = MatchSet()
        for m in v:
            if re.match(r"\w+$", m):
                mset.exact.add(m)
            else:
                mset.regex.add(m)
        ps[k] = mset
    return ps


# Returns True if the given name matches either an exact string in the MatchSet or any of its regex patterns, otherwise returns False.
def match_name(name, mset):
    if name in mset.exact:
        return True
    for m in mset.regex:
        if re.match(m, name):
            return True
    return False


# Pre-process once at import time
DISABLED_TESTS_MATCH = prepare_match_set(DISABLED_TESTS)


def _get_disabled_mset(cls_name, generic_name):
    return DISABLED_TESTS_MATCH.get(cls_name) or DISABLED_TESTS_MATCH.get(generic_name)


# Remove built-in PrivateUse1TestBase so only SpyreTestBase handles
# the privateuse1 device type.  This prevents the nondeterministic
# overwrite when list(set(...)) randomizes order.
# TODO: figure out why this filter is needed - expected to use default PrivateUse1TestBase
device_type_test_bases[:] = [  # type: ignore[name-defined] # noqa: F821
    b
    for b in device_type_test_bases  # type: ignore[name-defined] # noqa: F821
    if b is not PrivateUse1TestBase  # type: ignore[name-defined] # noqa: F821
]


# PrivateUse1TestBase injected via globals()
class SpyreTestBase(PrivateUse1TestBase):  # type: ignore[name-defined] # noqa: F821
    device_type = "spyre"
    precision = DEFAULT_FLOATING_PRECISION

    # For testing the cases that are turned off automatically by the data type filter, setting to true will enable the tests that are in the unsupported_dtypes dict
    test_unsup_dtypes = False  # Default is False

    unsupported_dtypes = {
        torch.complex32,
        torch.complex64,
        torch.complex128,
        # torch.float64, # Not supported on Spyre but leaving commented so we see test_conj_self tests pass
        # torch.bfloat16, # Not supported on Spyre but leaving commented so we see test_conj_self tests pass
        # torch.int16, # Not supported on Spyre but leaving commented so we see test_conj_self tests pass
        # torch.int32, # Not supported on Spyre but leaving commented so we see test_conj_self tests pass
        # torch.uint8, # Not supported on Spyre but leaving commented so we see test_conj_self tests pass
    }

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls):
        # Resolve the actual device name (privateuse1 -> spyre)
        cls_device_type = (
            cls.device_type
            if cls.device_type != "privateuse1"
            else torch._C._get_privateuse1_backend_name()
        )
        test_name_with_device = name + "_" + cls_device_type

        # Look up disabled set by both the generated class name
        # (e.g. TestViewOpsPRIVATEUSE1) and the generic class name
        # (e.g. TestViewOps)
        disabled_mset = _get_disabled_mset(cls.__name__, generic_cls.__name__)

        base_is_disabled = disabled_mset is not None and (
            match_name(name, disabled_mset)
            or match_name(test_name_with_device, disabled_mset)
        )

        # Per-test precision override (no overrides currently)
        cls.precision = PRECISION_OVERRIDES.get(name, DEFAULT_FLOATING_PRECISION)

        # ── Snapshot existing methods, let parent do all the work ──────────
        existing_methods = set(cls.__dict__.keys())
        super().instantiate_test(name, test, generic_cls=generic_cls)
        new_methods = set(cls.__dict__.keys()) - existing_methods

        @wraps(test)
        def skip_test(self, test=test):
            raise unittest.SkipTest("Skipped for Spyre")

        @wraps(test)
        def skip_test_dtypes(self, test=test):
            raise unittest.SkipTest("dtype unsupported on Spyre")

        for method_name in new_methods:
            should_skip = base_is_disabled
            should_skip_dtype = False

            # Check if this specific variant matches a disabled pattern
            if not should_skip and disabled_mset is not None:
                should_skip = match_name(method_name, disabled_mset)

            # Skip unsupported dtypes for Spyre
            if not cls.test_unsup_dtypes:
                for dtype in cls.unsupported_dtypes:
                    dtype_str = str(dtype).split(".")[-1]
                    if dtype_str in method_name:
                        should_skip_dtype = True
                        break

            if should_skip_dtype:
                setattr(cls, method_name, skip_test_dtypes)
            elif should_skip:
                setattr(cls, method_name, skip_test)


TEST_CLASS = SpyreTestBase
