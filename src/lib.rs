#![allow(incomplete_features, internal_features, stable_features)]
#![feature(
    ptr_sub_ptr,
    maybe_uninit_slice,
    auto_traits,
    negative_impls,
    specialization,
    const_trait_impl,
    inline_const,
    core_intrinsics,
    sized_type_properties,
    generic_const_exprs,
    maybe_uninit_uninit_array_transpose
)]

use core::cmp::{self, Ordering};
use core::intrinsics;
use core::mem::{self, MaybeUninit, SizedTypeProperties};

mod drift;
mod merge;
mod pivot;
mod quicksort;
mod smallsort;

#[inline(always)]
pub fn sort<T: Ord>(v: &mut [T]) {
    stable_sort(v, |a, b| a.lt(b))
}

#[inline(always)]
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], mut compare: F) {
    stable_sort(v, |a, b| compare(a, b) == Ordering::Less);
}

#[inline(always)]
fn stable_sort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], mut is_less: F) {
    driftsort::<T, F, Vec<T>>(v, &mut is_less);
}

#[inline(always)]
fn driftsort<T, F: FnMut(&T, &T) -> bool, BufT: BufGuard<T>>(v: &mut [T], is_less: &mut F) {
    // Arrays of zero-sized types are always all-equal, and thus sorted.
    if T::IS_ZST {
        return;
    }

    // Instrumenting the standard library showed that 90+% of the calls to sort
    // by rustc are either of size 0 or 1.
    let len = v.len();
    if intrinsics::likely(len < 2) {
        return;
    }

    // More advanced sorting methods than insertion sort are faster if called in
    // a hot loop for small inputs, but for general-purpose code the small
    // binary size of insertion sort is more important. The instruction cache in
    // modern processors is very valuable, and for a single sort call in general
    // purpose code any gains from an advanced method are cancelled by icache
    // misses during the sort, and thrashing the i-cache for surrounding code.
    const MAX_LEN_ALWAYS_INSERTION_SORT: usize = 20;
    if intrinsics::likely(len <= MAX_LEN_ALWAYS_INSERTION_SORT) {
        smallsort::insertion_sort_shift_left(v, 1, is_less);
        return;
    }

    driftsort_main::<T, F, BufT>(v, is_less);
}

// Deliberately don't inline the main sorting routine entrypoint to ensure the
// inlined insertion sort i-cache footprint remains minimal.
#[inline(never)]
fn driftsort_main<T, F: FnMut(&T, &T) -> bool, BufT: BufGuard<T>>(v: &mut [T], is_less: &mut F) {
    use crate::smallsort::SmallSortTypeImpl;

    // By allocating n elements of memory we can ensure the entire input can
    // be sorted using stable quicksort, which allows better performance on
    // random and low-cardinality distributions. However, we still want to
    // reduce our memory usage to n / 2 for large inputs. We do this by scaling
    // our allocation as max(n / 2, min(n, 8MB)), ensuring we scale like n for
    // small inputs and n / 2 for large inputs, without a sudden dropoff. We
    // also need to ensure our alloc >= MIN_SMALL_SORT_SCRATCH_LEN, as the
    // small-sort always needs this much memory.
    const MAX_FULL_ALLOC_BYTES: usize = 8_000_000; // 8MB
    let max_full_alloc = MAX_FULL_ALLOC_BYTES / mem::size_of::<T>();
    let len = v.len();
    let alloc_len = cmp::max(len / 2, cmp::min(len, max_full_alloc));
    let alloc_len = cmp::max(alloc_len, crate::smallsort::MIN_SMALL_SORT_SCRATCH_LEN);

    // For small inputs 4KiB of stack storage suffices, which allows us to avoid
    // calling the (de-)allocator. Benchmarks showed this was quite beneficial.
    let mut stack_buf = AlignedStorage::<T, 4096>::new();
    let stack_scratch = stack_buf.as_uninit_slice_mut();
    let mut heap_buf;
    let scratch = if stack_scratch.len() >= alloc_len {
        stack_scratch
    } else {
        heap_buf = BufT::with_capacity(alloc_len);
        heap_buf.as_uninit_slice_mut()
    };

    // For small inputs using quicksort is not yet beneficial, and a single
    // small-sort or two small-sorts plus a single merge outperforms it, so use
    // eager mode.
    let eager_sort = len <= T::SMALL_SORT_THRESHOLD * 2;
    drift::sort(v, scratch, eager_sort, is_less);
}

trait BufGuard<T> {
    fn with_capacity(capacity: usize) -> Self;
    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>];
}

impl<T> BufGuard<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.spare_capacity_mut()
    }
}

#[repr(C)]
struct AlignedStorage<T, const N: usize> {
    _align: [T; 0],
    storage: [MaybeUninit<u8>; N],
}

impl<T, const N: usize> AlignedStorage<T, N> {
    fn new() -> Self {
        let storage: [MaybeUninit<u8>; N] = MaybeUninit::uninit().transpose();

        Self {
            _align: [],
            storage: storage,
        }
    }

    fn as_uninit_slice_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let len = N / mem::size_of::<T>();

        // SAFETY: `_align` ensures we are correctly aligned.
        unsafe { core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), len) }
    }
}

// --- Type info ---

/// # Safety
///
/// This is an internal trait that must match Rust interior mutability rules.
///
/// Direct copy of stdlib internal implementation of Freeze.
pub(crate) unsafe auto trait Freeze {}

impl<T: ?Sized> !Freeze for core::cell::UnsafeCell<T> {}
unsafe impl<T: ?Sized> Freeze for core::marker::PhantomData<T> {}
unsafe impl<T: ?Sized> Freeze for *const T {}
unsafe impl<T: ?Sized> Freeze for *mut T {}
unsafe impl<T: ?Sized> Freeze for &T {}
unsafe impl<T: ?Sized> Freeze for &mut T {}

#[const_trait]
trait IsFreeze {
    const IS_FREEZE: bool;
}

impl<T> const IsFreeze for T {
    default const IS_FREEZE: bool = false;
}

impl<T: Freeze> const IsFreeze for T {
    const IS_FREEZE: bool = true;
}

#[must_use]
const fn has_direct_interior_mutability<T>() -> bool {
    // If a type has interior mutability it may alter itself during comparison
    // in a way that must be preserved after the sort operation concludes.
    // Otherwise a type like Mutex<Option<Box<str>>> could lead to double free.
    !T::IS_FREEZE
}
