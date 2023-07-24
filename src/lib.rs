#![allow(incomplete_features)]
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
mod quicksort;
mod smallsort;

/// Compactly stores the length of a run, and whether or not it is sorted. This
/// can always fit in a usize because the maximum slice length is isize::MAX.
#[derive(Copy, Clone)]
struct DriftsortRun(usize);

impl DriftsortRun {
    #[inline(always)]
    pub fn new_sorted(length: usize) -> Self {
        Self((length << 1) | 1)
    }

    #[inline(always)]
    pub fn new_unsorted(length: usize) -> Self {
        Self((length << 1) | 0)
    }

    #[inline(always)]
    pub fn sorted(self) -> bool {
        self.0 & 1 == 1
    }

    #[inline(always)]
    pub fn len(self) -> usize {
        self.0 >> 1
    }
}

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
fn driftsort<T, F, BufT>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
    BufT: BufGuard<T>,
{
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
    const MAX_LEN_ALWAYS_INSERTION_SORT: usize = 20;
    if intrinsics::likely(len <= MAX_LEN_ALWAYS_INSERTION_SORT) {
        smallsort::insertion_sort_shift_left(v, 1, is_less);
        return;
    }

    driftsort_main::<T, F, BufT>(v, is_less);
}

#[inline(never)]
fn driftsort_main<T, F, BufT>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
    BufT: BufGuard<T>,
{
    // Pick whichever is greater:
    //  - alloc len elements up to MAX_FULL_ALLOC_BYTES
    //  - alloc len / 2 elements
    // This allows us to use the most performant algorithms for small-medium
    // sized inputs while scaling down to len / 2 for larger inputs. We need at
    // least len / 2 for our stable merging routine.
    const MAX_FULL_ALLOC_BYTES: usize = 8_000_000;
    let len = v.len();
    let full_alloc_size = cmp::min(len, MAX_FULL_ALLOC_BYTES / mem::size_of::<T>());
    let alloc_size = cmp::max(len / 2, full_alloc_size);

    let mut buf = BufT::with_capacity(alloc_size);
    let scratch_slice =
        unsafe { slice::from_raw_parts_mut(buf.mut_ptr() as *mut MaybeUninit<T>, buf.capacity()) };

    drift::sort(v, scratch_slice, false, is_less);
}

#[inline(never)]
fn physical_merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mid: usize,
    is_less: &mut F,
) {
    merge::merge(v, scratch, mid, is_less)
}

#[inline(never)]
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

/// Creates a new logical run.
///
/// A logical run can either be sorted or unsorted. If there is a pre-existing
/// run of length min_good_run_len (or longer) starting at v[0] we find and
/// return it, otherwise we return a run of length min_good_run_len that is
/// eagerly sorted when eager_sort is true, and left unsorted otherwise.
fn create_run<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    min_good_run_len: usize,
    eager_sort: bool,
    is_less: &mut F,
) -> DriftsortRun {
    let (run_len, was_reversed) = find_existing_run(v, is_less);
    if run_len >= min_good_run_len {
        if was_reversed {
            v[..run_len].reverse();
        }
        DriftsortRun::new_sorted(run_len)
    } else {
        let new_run_len = cmp::min(min_good_run_len, v.len());
        if eager_sort {
            smallsort::sort_small(&mut v[..new_run_len], is_less);
            DriftsortRun::new_sorted(new_run_len)
        } else {
            DriftsortRun::new_unsorted(new_run_len)
        }
    }
}

/// Finds a run of sorted elements starting at the beginning of the slice.
///
/// Returns the length of the run, and a bool that is false when the run
/// is ascending, and true if the run strictly descending.
fn find_existing_run<T, F>(v: &[T], is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    if len < 2 {
        return (len, false);
    }

    unsafe {
        // SAFETY: We checked that len >= 2, so 0 and 1 are valid indices.
        // This also means that run_len < len implies run_len and
        // run_len - 1 are valid indices as well.
        let mut run_len = 2;
        let strictly_descending = is_less(v.get_unchecked(1), v.get_unchecked(0));
        if strictly_descending {
            while run_len < len && is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1)) {
                run_len += 1;
            }
        } else {
            while run_len < len && !is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1))
            {
                run_len += 1;
            }
        }
        (run_len, strictly_descending)
    }
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

// Can the type have interior mutability, this is checked by testing if T is Freeze. If the type can
// have interior mutability it may alter itself during comparison in a way that must be observed
// after the sort operation concludes. Otherwise a type like Mutex<Option<Box<str>>> could lead to
// double free.
//
// Direct copy of stdlib internal implementation of Freeze.
pub(crate) unsafe auto trait Freeze {}

impl<T: ?Sized> !Freeze for core::cell::UnsafeCell<T> {}
unsafe impl<T: ?Sized> Freeze for core::marker::PhantomData<T> {}
unsafe impl<T: ?Sized> Freeze for *const T {}
unsafe impl<T: ?Sized> Freeze for *mut T {}
unsafe impl<T: ?Sized> Freeze for &T {}
unsafe impl<T: ?Sized> Freeze for &mut T {}

#[const_trait]
trait IsFreeze {
    fn value() -> bool;
}

impl<T> const IsFreeze for T {
    default fn value() -> bool {
        false
    }
}

impl<T: Freeze> const IsFreeze for T {
    fn value() -> bool {
        true
    }
}

#[must_use]
const fn has_direct_interior_mutability<T>() -> bool {
    // Can the type have interior mutability, this is checked by testing if T is Freeze. If the type
    // can have interior mutability it may alter itself during comparison in a way that must be
    // observed after the sort operation concludes. Otherwise a type like Mutex<Option<Box<str>>>
    // could lead to double free.
    !<T as IsFreeze>::value()
}

trait IsTrue<const B: bool> {}
impl IsTrue<true> for () {}
