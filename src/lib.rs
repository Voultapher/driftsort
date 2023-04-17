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
    sized_type_properties
)]

use core::cmp::{self, Ordering};
use core::intrinsics;
use core::mem::{self, MaybeUninit, SizedTypeProperties};
use core::slice;

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
    // Sorting has no meaningful behavior on zero-sized types.
    if T::IS_ZST {
        return;
    }

    let len = v.len();

    // This path is critical for very small inputs. Always pick insertion sort for these inputs,
    // without any other analysis. This is perf critical for small inputs, in cold code.
    const MAX_LEN_ALWAYS_INSERTION_SORT: usize = 20;

    // Instrumenting the standard library showed that 90+% of the calls to sort by rustc are either
    // of size 0 or 1. Make this path extra fast by assuming the branch is likely.
    if intrinsics::likely(len < 2) {
        return;
    }

    // It's important to differentiate between small-sort performance for small slices and
    // small-sort performance sorting small sub-slices as part of the main quicksort loop. For the
    // former, testing showed that the representative benchmarks for real-world performance are cold
    // CPU state and not single-size hot benchmarks. For the latter the CPU will call them many
    // times, so hot benchmarks are fine and more realistic. And it's worth it to optimize sorting
    // small sub-slices with more sophisticated solutions than insertion sort.

    if intrinsics::likely(len <= MAX_LEN_ALWAYS_INSERTION_SORT) {
        // More specialized and faster options, extending the range of allocation free sorting
        // are possible but come at a great cost of additional code, which is problematic for
        // compile-times.
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
    // Allocating len instead of len / 2 allows the quicksort to work on the full size, which can
    // give speedups especially for low cardinality inputs where common values are filtered out only
    // once, instead of twice. And it allows bi-directional merging the full input. However to
    // reduce peak memory usage for large inputs, fall back to allocating len / 2 if a certain
    // threshold is passed.
    const MAX_FULL_ALLOC_BYTES: usize = 8_000_000; // 8MB

    let len = v.len();

    // Pick whichever is greater:
    //
    //  - alloc n up to MAX_FULL_ALLOC_BYTES
    //  - alloc n / 2
    //
    // This serves to make the impact and performance cliff when going above the threshold less
    // severe than immediately switching to len / 2.
    let full_alloc_size = cmp::min(len, MAX_FULL_ALLOC_BYTES / mem::size_of::<T>());
    let alloc_size = cmp::max(len / 2, full_alloc_size);

    let mut buf = <BufT as BufGuard<T>>::with_capacity(alloc_size);

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
            let run_end = cmp::min(crate::quicksort::SMALL_SORT_THRESHOLD, len);
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

#[must_use]
const fn is_cheap_to_move<T>() -> bool {
    mem::size_of::<T>() <= mem::size_of::<[usize; 4]>()
}
