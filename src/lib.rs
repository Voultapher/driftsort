#![allow(dead_code, unused_variables)]

const SMALL_SORT_THRESH: usize = 32;

use std::cmp::Ordering;
use std::mem::MaybeUninit;

mod glide;
mod glide_dyn_dispatch;
mod glide_dyn_dispatch2;
mod logical_run;

#[inline(always)]
pub fn sort<T: Ord>(el: &mut [T]) {
    driftsort(el, |a, b| a.lt(b))
}

#[inline(always)]
pub fn sort_by<T, F: FnMut(&T, &T) -> Ordering>(el: &mut [T], mut compare: F) {
    driftsort(el, |a, b| compare(a, b) == Ordering::Less);
}

#[inline(always)]
pub fn driftsort<T, F: FnMut(&T, &T) -> bool>(el: &mut [T], mut is_less: F) {
    if el.len() < 2 || std::mem::size_of::<T>() == 0 {
        return;
    }

    slow_path_sort_dyn(el, &mut is_less);
}

pub trait SortOps {
    fn create_run(&mut self, start: usize, eager_sort: bool) -> glide_dyn_dispatch::LogicalRun;
    fn physical_sort(&mut self, start: usize, end: usize);
    fn physical_merge(&mut self, start: usize, mid: usize, end: usize);
}

pub trait SortOps2 {
    fn create_run(
        &mut self,
        start: usize,
        eager_sort: bool,
    ) -> glide_dyn_dispatch2::LengthAndSorted;
    fn physical_sort(&mut self, start: usize, end: usize);
    fn physical_merge(&mut self, start: usize, mid: usize, end: usize);
}

struct SortParams<'a, T, F> {
    el: &'a mut [T],
    scratch: &'a mut [MaybeUninit<T>],
    is_less: F,
}

// Physical operations.
impl<'a, T, F: FnMut(&T, &T) -> bool> SortOps for SortParams<'a, T, F> {
    fn create_run(&mut self, start: usize, eager_sort: bool) -> glide_dyn_dispatch::LogicalRun {
        // FIXME: actually detect runs.
        glide_dyn_dispatch::LogicalRun::new_unsorted(
            start,
            self.el.len().saturating_sub(start).min(32),
        )
    }

    fn physical_sort(&mut self, start: usize, end: usize) {
        // FIXME
        self.el[start..end].sort_by(|a, b| {
            if (self.is_less)(a, b) {
                std::cmp::Ordering::Less
            } else if (self.is_less)(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }

    fn physical_merge(&mut self, start: usize, mid: usize, end: usize) {
        // FIXME
        self.el[start..end].sort_by(|a, b| {
            if (self.is_less)(a, b) {
                std::cmp::Ordering::Less
            } else if (self.is_less)(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
}

// Physical operations.
impl<'a, T, F: FnMut(&T, &T) -> bool> SortOps2 for SortParams<'a, T, F> {
    fn create_run(
        &mut self,
        start: usize,
        eager_sort: bool,
    ) -> glide_dyn_dispatch2::LengthAndSorted {
        // FIXME: actually detect runs.
        glide_dyn_dispatch2::LengthAndSorted::new(
            self.el.len().saturating_sub(start).min(32),
            false,
        )
    }

    fn physical_sort(&mut self, start: usize, end: usize) {
        // FIXME
        self.el[start..end].sort_by(|a, b| {
            if (self.is_less)(a, b) {
                std::cmp::Ordering::Less
            } else if (self.is_less)(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }

    fn physical_merge(&mut self, start: usize, mid: usize, end: usize) {
        // FIXME
        self.el[start..end].sort_by(|a, b| {
            if (self.is_less)(a, b) {
                std::cmp::Ordering::Less
            } else if (self.is_less)(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
}

#[inline(never)]
pub fn physical_merge<T, F: FnMut(&T, &T) -> bool>(
    el: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mid: usize,
    is_less: &mut F,
) {
    // FIXME
    el.sort_by(|a, b| {
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
    el: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    is_less: &mut F,
) {
    // FIXME
    el.sort_by(|a, b| {
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
pub fn slow_path_sort<T, F: FnMut(&T, &T) -> bool>(el: &mut [T], is_less: &mut F) {
    let alloc_size = SMALL_SORT_THRESH.max(el.len() / 2);
    let mut scratch: Vec<T> = Vec::with_capacity(alloc_size);
    let scratch_slice = unsafe {
        std::slice::from_raw_parts_mut(
            scratch.as_mut_ptr().cast::<MaybeUninit<T>>(),
            scratch.capacity(),
        )
    };
    glide::sort(el, scratch_slice, false, is_less);
}

#[inline(never)]
#[cold]
pub fn slow_path_sort_dyn<T, F: FnMut(&T, &T) -> bool>(el: &mut [T], is_less: &mut F) {
    let alloc_size = SMALL_SORT_THRESH.max(el.len() / 2);
    let mut scratch: Vec<T> = Vec::with_capacity(alloc_size);
    let scratch_slice = unsafe {
        std::slice::from_raw_parts_mut(
            scratch.as_mut_ptr().cast::<MaybeUninit<T>>(),
            scratch.capacity(),
        )
    };
    let el_len = el.len();
    let scratch_len = scratch_slice.len();
    let mut params = SortParams {
        el,
        scratch: scratch_slice,
        is_less,
    };
    glide_dyn_dispatch2::sort(&mut params, false, el_len, scratch_len);
}
