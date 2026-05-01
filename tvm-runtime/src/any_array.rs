//! Heterogeneous-tolerant Array element extraction.
//!
//! Upstream `tvm_ffi::Array<T>` requires all elements to satisfy `T` at
//! conversion time. VM packed-func returns (e.g. `batch_decode` returning
//! `[logits, kv_cache, rnn_state]`) are heterogeneous arrays that cannot be
//! cast to `Array<Tensor>` as a whole.
//!
//! This module provides a low-level helper that goes through the tvm-ffi C
//! ABI, retrieves the `i`-th element of any array-type `Any`, and downcasts
//! it to the target type one at a time.

use tvm_ffi::collections::array::ArrayObj;
use tvm_ffi::{Any, AnyView};
use tvm_ffi_sys::{TVMFFIAny, TVMFFITypeIndex};

/// Retrieve element `index` of an array-typed `Any`, downcast to `T`.
///
/// # Safety
///
/// Caller must ensure `value` is a valid Any whose underlying container is
/// an `ArrayObj`. The function will return an error if the type index
/// disagrees or if the index is out of bounds.
pub unsafe fn get_from_any_array<T>(value: Any, index: usize) -> anyhow::Result<T>
where
    for<'a> T: TryFrom<AnyView<'a>>,
    for<'a> <T as TryFrom<AnyView<'a>>>::Error: std::fmt::Debug,
{
    // Any and AnyView<'_> share TVMFFIAny layout; peek the tag + payload.
    let raw: &TVMFFIAny = std::mem::transmute(&value);
    if raw.type_index != TVMFFITypeIndex::kTVMFFIArray as i32 {
        anyhow::bail!(
            "expected Any of type kTVMFFIArray ({}), found type_index {}",
            TVMFFITypeIndex::kTVMFFIArray as i32,
            raw.type_index
        );
    }

    // The v_obj pointer is a TVMFFIObject* which happens to be the header of
    // the upstream ArrayObj (single-inheritance on Object).
    let container: &ArrayObj = &*(raw.data_union.v_obj as *const ArrayObj);
    if (index as i64) >= container.size {
        anyhow::bail!(
            "index {} out of bounds for array of length {}",
            index,
            container.size
        );
    }

    // `container.data` is kept as a *void* to the element buffer; it is
    // updated in lockstep with the internal extra-items region.
    let base_ptr = container.data as *const TVMFFIAny;
    let elem_any: &TVMFFIAny = &*base_ptr.add(index);
    let view: AnyView<'_> = std::mem::transmute_copy(elem_any);
    T::try_from(view).map_err(|e| anyhow::anyhow!("downcast failed: {:?}", e))
}
