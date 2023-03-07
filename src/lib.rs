#![feature(ptr_sub_ptr, maybe_uninit_slice)]

const SMALL_SORT_THRESH: usize = 32;

use std::cmp::Ordering;
use std::mem::MaybeUninit;

mod drift;
mod quicksort;
mod smallsort;

/// Compactly stores the length of a run, and whether or not it is sorted. This
/// can always fit in a usize because the maximum slice length is isize::MAX.
#[derive(Copy, Clone)]
struct LengthAndSorted(usize);

impl LengthAndSorted {
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

    slow_path_sort(v, &mut is_less);
}

#[inline(never)]
#[cold]
fn slow_path_sort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], is_less: &mut F) {
    let alloc_size = SMALL_SORT_THRESH.max(v.len() / 2);
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
    _scratch: &mut [MaybeUninit<T>],
    _mid: usize,
    is_less: &mut F,
) {
    // FIXME
    v.sort_by(|a, b| {
        if is_less(a, b) {
            std::cmp::Ordering::Less
        } else if is_less(b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
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
    crate::quicksort::stable_quicksort(v, scratch, limit, is_less);
}

#[inline(never)]
fn create_run<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    eager_sort: bool,
    is_less: &mut F,
) -> LengthAndSorted {
    // FIXME: run detection.

    // TODO: unlikely?
    if eager_sort {
        let len = v.len().min(32);
        smallsort::sort_small(&mut v[..len], is_less);
        LengthAndSorted::new_sorted(len)
    } else {
        LengthAndSorted::new_unsorted(v.len().min(32))
    }
}
