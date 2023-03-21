#![allow(incomplete_features)]
#![feature(
    ptr_sub_ptr,
    maybe_uninit_slice,
    auto_traits,
    negative_impls,
    specialization,
    const_trait_impl,
    inline_const
)]

use core::cmp::{self, Ordering};
use core::mem::MaybeUninit;
use core::ptr;

mod drift;
mod merge;
mod quicksort;
mod smallsort;

const FALLBACK_RUN_LEN: usize = 10;

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
    driftsort(v, |a, b| a.lt(b))
}

#[inline(always)]
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], mut compare: F) {
    driftsort(v, |a, b| compare(a, b) == Ordering::Less);
}

#[inline(always)]
fn driftsort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], mut is_less: F) {
    if v.len() < 2 || std::mem::size_of::<T>() == 0 {
        return;
    }

    // TODO insertion sort for small slices.

    slow_path_sort(v, &mut is_less);
}

#[inline(never)]
#[cold]
fn slow_path_sort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], is_less: &mut F) {
    let alloc_size = cmp::max(v.len() / 2, 64); // TODO use const from quicksort. Or just N.

    let mut scratch: Vec<T> = Vec::with_capacity(alloc_size);
    let scratch_slice = unsafe {
        std::slice::from_raw_parts_mut(
            scratch.as_mut_ptr().cast::<MaybeUninit<T>>(),
            scratch.capacity(),
        )
    };

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
    crate::quicksort::stable_quicksort(v, scratch, limit, ptr::null(), is_less);
}

/// Create a new logical run, that is either sorted or unsorted.
fn create_run<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    mut min_good_run_len: usize,
    eager_sort: bool,
    is_less: &mut F,
) -> DriftsortRun {
    // FIXME: run detection.

    let len = v.len();

    let (streak_end, was_reversed) = find_streak(v, is_less);

    if eager_sort {
        min_good_run_len = FALLBACK_RUN_LEN;
    }

    // It's important to have a relatively high entry barrier for pre-sorted runs, as the presence
    // of a single such run will force on average several merge operations and shrink the max
    // quicksort size a lot. Which impact low-cardinality filtering performance.
    if streak_end >= min_good_run_len {
        if was_reversed {
            v[..streak_end].reverse();
        }

        DriftsortRun::new_sorted(streak_end)
    } else {
        if !eager_sort {
            // min_good_run_len serves dual duty here, if no streak was found, create a relatively
            // large unsorted run to avoid calling find_streak all the time. This also puts a limit
            // on how many logical merges have to be done, but this plays a minor role performance
            // wise.
            DriftsortRun::new_unsorted(cmp::min(min_good_run_len, len))
        } else {
            // We are not allowed to generate unsorted sequences in this mode. This mode is used as
            // fallback algorithm for quicksort. Essentially falling back to merge sort.
            let run_end = cmp::min(FALLBACK_RUN_LEN, len);
            smallsort::sort_small(&mut v[..run_end], is_less);

            DriftsortRun::new_sorted(run_end)
        }
    }
}

/// Finds a streak of presorted elements starting at the beginning of the slice. Returns the first
/// value that is not part of said streak, and a bool denoting wether the streak was reversed.
/// Streaks can be increasing or decreasing.
fn find_streak<T, F>(v: &[T], is_less: &mut F) -> (usize, bool)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    if len < 2 {
        return (len, false);
    }

    let mut end = 2;

    // SAFETY: See below specific.
    unsafe {
        // SAFETY: We checked that len >= 2, so 0 and 1 are valid indices.
        let assume_reverse = is_less(v.get_unchecked(1), v.get_unchecked(0));

        // SAFETY: We know end >= 2 and check end < len.
        // From that follows that accessing v at end and end - 1 is safe.
        if assume_reverse {
            while end < len && is_less(v.get_unchecked(end), v.get_unchecked(end - 1)) {
                end += 1;
            }

            (end, true)
        } else {
            while end < len && !is_less(v.get_unchecked(end), v.get_unchecked(end - 1)) {
                end += 1;
            }
            (end, false)
        }
    }
}

// --- Type info ---

// Can the type have interior mutability, this is checked by testing if T is Copy. If the type can
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
    // - Can the type have interior mutability, this is checked by testing if T is Freeze.
    //   If the type can have interior mutability it may alter itself during comparison in a way
    //   that must be observed after the sort operation concludes.
    //   Otherwise a type like Mutex<Option<Box<str>>> could lead to double free.
    !<T as IsFreeze>::value()
}
