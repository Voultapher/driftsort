#![allow(dead_code, unused_variables)]

const SMALL_SORT_THRESH: usize = 32;

use std::mem::MaybeUninit;

mod glide;
mod logical_run;


#[inline(never)]
pub fn physical_merge<T, F: FnMut(&T, &T) -> bool>(el: &mut [T], scratch: &mut [MaybeUninit<T>], mid: usize, is_less: &mut F) {
    // FIXME
    el.sort_by(|a, b| if is_less(a, b) { std::cmp::Ordering::Less } else if is_less(b, a) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal });
}

#[inline(never)]
pub fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(el: &mut [T], scratch: &mut [MaybeUninit<T>], is_less: &mut F) {
    // FIXME
    el.sort_by(|a, b| if is_less(a, b) { std::cmp::Ordering::Less } else if is_less(b, a) { std::cmp::Ordering::Greater } else { std::cmp::Ordering::Equal } )
}









#[inline(always)]
pub fn sort<T: Ord>(el: &mut [T]) {
    if el.len() < 2 || std::mem::size_of::<T>() == 0 {
        return;
    }
    
    slow_path_sort(el)
}

#[inline(never)]
#[cold]
pub fn slow_path_sort<T: Ord>(el: &mut [T]) {
    let alloc_size = SMALL_SORT_THRESH.max(el.len() / 2);
    let mut scratch: Vec<T> = Vec::with_capacity(alloc_size);
    let scratch_slice = unsafe {
        std::slice::from_raw_parts_mut(scratch.as_mut_ptr().cast::<MaybeUninit<T>>(), scratch.capacity())
    };
    glide::sort(el, scratch_slice, false, &mut T::lt);
}




