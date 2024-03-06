#![allow(incomplete_features, internal_features)]
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
    generic_const_exprs
)]

use core::cmp::{self, Ordering};
use core::intrinsics;
use core::mem::{self, MaybeUninit, SizedTypeProperties};
use core::slice;

mod drift;
mod merge;
mod pivot;
mod quicksort;
mod smallsort;

use crate::smallsort::MAX_LEN_ALWAYS_INSERTION_SORT;

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
    // misses during the sort, and thrashing the icache for surrounding code.

    if intrinsics::likely(len <= MAX_LEN_ALWAYS_INSERTION_SORT) {
        smallsort::insertion_sort_shift_left(v, 1, is_less);
        return;
    }

    driftsort_main::<T, F, BufT>(v, is_less);
}

// Deliberately don't inline the core logic to ensure the inlined insertion sort i-cache footprint
// is minimal.
#[inline(never)]
fn driftsort_main<T, F: FnMut(&T, &T) -> bool, BufT: BufGuard<T>>(v: &mut [T], is_less: &mut F) {
    // Pick whichever is greater:
    //  - alloc len elements up to MAX_FULL_ALLOC_BYTES
    //  - alloc len / 2 elements
    // This allows us to use the most performant algorithms for small-medium
    // sized inputs while scaling down to len / 2 for larger inputs. We need at
    // least len / 2 for our stable merging routine.
    const MAX_FULL_ALLOC_BYTES: usize = 8_000_000;

    let len = v.len();

    // Using the hybrid quick + merge-sort has performance issues with the transition from insertion
    // sort to main loop. A more gradual and smoother transition can be had by using an always eager
    // merge approach as long as it can be served by at most 3 merges. Another factor that affects
    // transition performance, is the allocation size. The small-sort has a lower limit of
    // `MIN_SMALL_SORT_SCRATCH_LEN` e.g. 48 which is much larger than for example 21. Where we only
    // need a scratch buffer of length 10. Especially fully ascending and descending inputs in range
    // `20..100` that are a bit larger than a `u64`, e.g. `String` spend most of their time doing
    // the allocation. An alternativ would be to do lazy scratch allocation, but that comes with
    // it's own set of complex tradeoffs and makes the code substantially more complex.
    let eager_sort = len <= MAX_LEN_ALWAYS_INSERTION_SORT * 4;
    let len_div_2 = len / 2;

    let alloc_size = if eager_sort {
        len_div_2
    } else {
        // There are local checks that ensure that it would abort if the thresholds are tweaked in a
        // way that make this value ever smaller than `MIN_SMALL_SORT_SCRATCH_LEN`
        let full_alloc_size = cmp::min(len, MAX_FULL_ALLOC_BYTES / mem::size_of::<T>());

        cmp::max(len_div_2, full_alloc_size)
    };

    let mut buf = BufT::with_capacity(alloc_size);
    let scratch_slice =
        unsafe { slice::from_raw_parts_mut(buf.mut_ptr() as *mut MaybeUninit<T>, buf.capacity()) };

    drift::sort(v, scratch_slice, eager_sort, is_less);
}

fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) {
    // Limit the number of imbalanced partitions to `2 * floor(log2(len))`.
    // The binary OR by one is used to eliminate the zero-check in the logarithm.
    let limit = 2 * (v.len() | 1).ilog2();
    crate::quicksort::stable_quicksort(v, scratch, limit, None, is_less);
}

trait BufGuard<T> {
    fn with_capacity(capacity: usize) -> Self;
    fn mut_ptr(&mut self) -> *mut T;
    fn capacity(&self) -> usize;
}

impl<T> BufGuard<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    fn mut_ptr(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
    fn capacity(&self) -> usize {
        self.capacity()
    }
}

// --- Type info ---

/// # Safety
///
/// This is an internal trait that must match Rust
/// interior mutability rules.
///
/// Can the type have interior mutability, this is checked by testing if T is Freeze. If the type can
/// have interior mutability it may alter itself during comparison in a way that must be observed
/// after the sort operation concludes. Otherwise a type like Mutex<Option<Box<str>>> could lead to
/// double free.
///
/// Direct copy of stdlib internal implementation of Freeze.
pub(crate) unsafe auto trait Freeze {}

impl<T: ?Sized> !Freeze for core::cell::UnsafeCell<T> {}
unsafe impl<T: ?Sized> Freeze for core::marker::PhantomData<T> {}
unsafe impl<T: ?Sized> Freeze for *const T {}
unsafe impl<T: ?Sized> Freeze for *mut T {}
unsafe impl<T: ?Sized> Freeze for &T {}
unsafe impl<T: ?Sized> Freeze for &mut T {}

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
    // Can the type have interior mutability, this is checked by testing if T is Freeze. If the type
    // can have interior mutability it may alter itself during comparison in a way that must be
    // observed after the sort operation concludes. Otherwise a type like Mutex<Option<Box<str>>>
    // could lead to double free.
    !T::IS_FREEZE
}
