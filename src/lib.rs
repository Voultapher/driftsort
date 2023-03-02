#![allow(dead_code, unused_variables)]

const SMALL_SORT_THRESH: usize = 32;

use std::cmp::Ordering;
use std::mem::MaybeUninit;

mod glide;
mod logical_run;

#[inline(always)]
pub fn sort<T: Ord>(v: &mut [T]) {
    driftsort(v, |a, b| a.lt(b))
}

#[inline(always)]
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(v: &mut [T], mut compare: F) {
    driftsort(v, |a, b| compare(a, b) == Ordering::Less);
}

#[inline(always)]
pub fn driftsort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], mut is_less: F) {
    if v.len() < 2 || std::mem::size_of::<T>() == 0 {
        return;
    }

    slow_path_sort(v, &mut is_less);
}

#[inline(never)]
pub fn physical_merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mid: usize,
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
pub fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
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
    })
}

#[inline(never)]
#[cold]
pub fn slow_path_sort<T, F: FnMut(&T, &T) -> bool>(v: &mut [T], is_less: &mut F) {
    let alloc_size = SMALL_SORT_THRESH.max(v.len() / 2);
    let mut scratch: Vec<T> = Vec::with_capacity(alloc_size);
    let scratch_slice = unsafe {
        std::slice::from_raw_parts_mut(
            scratch.as_mut_ptr().cast::<MaybeUninit<T>>(),
            scratch.capacity(),
        )
    };
    glide::sort(v, scratch_slice, false, is_less);
}
