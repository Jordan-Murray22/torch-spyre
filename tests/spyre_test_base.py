import torch
import re
import unittest
from functools import wraps

DEFAULT_FLOATING_PRECISION = 1e-3

DISABLED_TESTS = {
    "TestViewOps": {
        "test_reshape_noncontiguous_spyre",  # Known limitation
        "test_view_dtype_new",
        "test_view_dtype_upsize_errors",
        "test_view_as_complex",  # Complex not supported
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
        "test_compare_cpu_cholesky_.*",  # Decomposition not implemented
    },
}

PRECISION_OVERRIDES = {
    "test_sum": 1e-2,
    "test_softmax": 1e-3,
    "test_batch_norm": 1e-1,
}


# ---------------------------------------------------------------------------
# Match infrastructure — adapted from pytorch/xla pytorch_test_base.py
# ---------------------------------------------------------------------------
class MatchSet(object):
    def __init__(self):
        self.exact = set()
        self.regex = set()


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
    device_type = "privateuse1"
    precision = DEFAULT_FLOATING_PRECISION

    unsupported_dtypes = {
        torch.complex32,
        torch.complex64,
        torch.complex128,
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

        # Per-test precision override
        cls.precision = PRECISION_OVERRIDES.get(name, DEFAULT_FLOATING_PRECISION)

        # ── Snapshot existing methods, let parent do all the work ──────────
        existing_methods = set(cls.__dict__.keys())
        super().instantiate_test(name, test, generic_cls=generic_cls)
        new_methods = set(cls.__dict__.keys()) - existing_methods

        @wraps(test)
        def skip_test(self, test=test):
            raise unittest.SkipTest("Skipped for Spyre")

        for method_name in new_methods:
            should_skip = base_is_disabled

            # Check if this specific variant matches a disabled pattern
            if not should_skip and disabled_mset is not None:
                should_skip = match_name(method_name, disabled_mset)

            if should_skip:
                setattr(cls, method_name, skip_test)


TEST_CLASS = SpyreTestBase
