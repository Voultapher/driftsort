use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

/// Sorts `v` using strategies optimized for small sizes.
pub fn sort_small<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    <T as SmallSortTypeImpl>::small_sort(v, is_less);
}

// When dropped, copies from `src` into `dest`.
struct InsertionHole<T> {
    src: *const T,
    dest: *mut T,
}

impl<T> Drop for InsertionHole<T> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::copy_nonoverlapping(self.src, self.dest, 1);
        }
    }
}

/// Inserts `v[v.len() - 1]` into pre-sorted sequence `v[..v.len() - 1]` so that whole `v[..]`
/// becomes sorted. Returns the insert position.
unsafe fn insert_tail<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    debug_assert!(len >= 2);

    let arr_ptr = v.as_mut_ptr();
    let i = len - 1;

    // SAFETY: caller must ensure v is at least len 2.
    unsafe {
        // See insert_head which talks about why this approach is beneficial.
        let i_ptr = arr_ptr.add(i);

        // It's important that we use i_ptr here. If this check is positive and we continue,
        // We want to make sure that no other copy of the value was seen by is_less.
        // Otherwise we would have to copy it back.
        if is_less(&*i_ptr, &*i_ptr.sub(1)) {
            // It's important, that we use tmp for comparison from now on. As it is the value that
            // will be copied back. And notionally we could have created a divergence if we copy
            // back the wrong value.
            let tmp = ManuallyDrop::new(ptr::read(i_ptr));
            // Intermediate state of the insertion process is always tracked by `hole`, which
            // serves two purposes:
            // 1. Protects integrity of `v` from panics in `is_less`.
            // 2. Fills the remaining hole in `v` in the end.
            //
            // Panic safety:
            //
            // If `is_less` panics at any point during the process, `hole` will get dropped and
            // fill the hole in `v` with `tmp`, thus ensuring that `v` still holds every object it
            // initially held exactly once.
            let mut hole = InsertionHole {
                src: &*tmp,
                dest: i_ptr.sub(1),
            };
            ptr::copy_nonoverlapping(hole.dest, i_ptr, 1);

            // SAFETY: We know i is at least 1.
            for j in (0..(i - 1)).rev() {
                let j_ptr = arr_ptr.add(j);
                if !is_less(&*tmp, &*j_ptr) {
                    break;
                }

                ptr::copy_nonoverlapping(j_ptr, hole.dest, 1);
                hole.dest = j_ptr;
            }
            // `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
        }
    }
}

/// Sort `v` assuming `v[..offset]` is already sorted.
pub fn insertion_sort_shift_left<T, F>(v: &mut [T], offset: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    // Using assert here improves performance.
    assert!(offset != 0 && offset <= len);

    // Shift each element of the unsorted region v[i..] as far left as is needed to make v sorted.
    for i in offset..len {
        // SAFETY: we tested that `offset` must be at least 1, so this loop is only entered if len
        // >= 2.
        unsafe {
            insert_tail(&mut v[..=i], is_less);
        }
    }
}

// Use a trait to focus code-gen on only the parts actually relevant for the type. Avoid generating
// LLVM-IR for the sorting-network and median-networks for types that don't qualify.
trait SmallSortTypeImpl: Sized {
    fn small_sort<F>(v: &mut [Self], is_less: &mut F)
    where
        F: FnMut(&Self, &Self) -> bool;
}

impl<T> SmallSortTypeImpl for T {
    default fn small_sort<F>(v: &mut [Self], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        if v.len() >= 2 {
            insertion_sort_shift_left(v, 1, is_less);
        }
    }
}

impl<T: Copy + crate::Freeze> SmallSortTypeImpl for T {
    fn small_sort<F>(v: &mut [Self], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        const MAX_SIZE: usize = crate::quicksort::SMALL_SORT_THRESHOLD;

        let len = v.len();

        if const { !crate::is_cheap_to_move::<T>() } {
            if len >= 2 {
                insertion_sort_shift_left(v, 1, is_less);
            }
            return;
        }

        let mut scratch = MaybeUninit::<[T; MAX_SIZE]>::uninit();
        let scratch_ptr = scratch.as_mut_ptr() as *mut T;

        if len >= 16 {
            let even_len = len - (len % 2 != 0) as usize;
            let len_div_2 = even_len / 2;

            // SAFETY: scratch_ptr is valid and has enough space.
            unsafe {
                sort8_stable(&mut v[..8], scratch_ptr, is_less);
                sort8_stable(&mut v[len_div_2..(len_div_2 + 8)], scratch_ptr, is_less);
            }

            insertion_sort_shift_left(&mut v[0..len_div_2], 8, is_less);
            insertion_sort_shift_left(&mut v[len_div_2..], 8, is_less);

            // SAFETY: We checked that T is Copy and thus observation safe. Should is_less panic v
            // was not modified in parity_merge and retains it's original input. swap and v must not
            // alias and swap has v.len() space.
            unsafe {
                bi_directional_merge_even(&mut v[..even_len], scratch_ptr, is_less);
                ptr::copy_nonoverlapping(scratch_ptr, v.as_mut_ptr(), even_len);
            }

            if len != even_len {
                // SAFETY: We know len >= 2.
                unsafe {
                    insert_tail(v, is_less);
                }
            }
        } else if len >= 2 {
            let offset = if len >= 8 {
                // SAFETY: scratch_ptr is valid and has enough space.
                unsafe {
                    sort8_stable(&mut v[..8], scratch_ptr, is_less);
                }

                8
            } else {
                1
            };

            insertion_sort_shift_left(v, offset, is_less);
        }
    }
}

/// Swap two values in array pointed to by a_ptr and b_ptr if b is less than a.
#[inline(always)]
unsafe fn branchless_swap<T>(a_ptr: *mut T, b_ptr: *mut T, should_swap: bool) {
    // SAFETY: the caller must guarantee that `a_ptr` and `b_ptr` are valid for writes
    // and properly aligned, and part of the same allocation, and do not alias.

    // This is a branchless version of swap if.
    // The equivalent code with a branch would be:
    //
    // if should_swap {
    //     ptr::swap_nonoverlapping(a_ptr, b_ptr, 1);
    // }

    // Give ourselves some scratch space to work with.
    // We do not have to worry about drops: `MaybeUninit` does nothing when dropped.
    let mut tmp = MaybeUninit::<T>::uninit();

    // The goal is to generate cmov instructions here.
    let a_swap_ptr = if should_swap { b_ptr } else { a_ptr };
    let b_swap_ptr = if should_swap { a_ptr } else { b_ptr };

    ptr::copy_nonoverlapping(b_swap_ptr, tmp.as_mut_ptr(), 1);
    ptr::copy(a_swap_ptr, a_ptr, 1);
    ptr::copy_nonoverlapping(tmp.as_ptr(), b_ptr, 1);
}

/// Swap two values in array pointed to by a_ptr and b_ptr if b is less than a.
#[inline(always)]
pub unsafe fn swap_next_if_less<T, F>(arr_ptr: *mut T, idx: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let a_ptr = arr_ptr.add(idx);
    let b_ptr = arr_ptr.add(idx + 1);

    // PANIC SAFETY: if is_less panics, no scratch memory was created and the slice should still be
    // in a well defined state, without duplicates.

    // Important to only swap if it is more and not if it is equal. is_less should return false for
    // equal, so we don't swap.
    let should_swap = is_less(&*b_ptr, &*a_ptr);
    branchless_swap(a_ptr, b_ptr, should_swap);
}

// Never inline this function to avoid code bloat. It still optimizes nicely and has practically no
// performance impact.
#[inline(never)]
fn sort4_stable<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: caller must ensure v.len() >= 4.
    assert!(v.len() == 4);

    let arr_ptr = v.as_mut_ptr();

    // Optimal sorting network see:
    // https://bertdobbelaere.github.io/sorting_networks.html.

    // SAFETY: We checked the len.
    unsafe {
        swap_next_if_less(arr_ptr, 0, is_less);
        swap_next_if_less(arr_ptr, 2, is_less);
        swap_next_if_less(arr_ptr, 1, is_less);

        swap_next_if_less(arr_ptr, 0, is_less);
        swap_next_if_less(arr_ptr, 2, is_less);
        swap_next_if_less(arr_ptr, 1, is_less);
    }
}

#[inline(never)]
unsafe fn sort8_stable<T, F>(v: &mut [T], scratch_ptr: *mut T, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: caller must ensure v.len() >= 4.
    assert!(v.len() == 8);

    sort4_stable(&mut v[..4], is_less);
    sort4_stable(&mut v[4..8], is_less);

    // SAFETY: We checked that T is Copy and thus observation safe.
    // Should is_less panic v was not modified in parity_merge and retains it's original input.
    // swap and v must not alias and swap has v.len() space.
    unsafe {
        bi_directional_merge_even(&mut v[..8], scratch_ptr, is_less);
        ptr::copy_nonoverlapping(scratch_ptr, v.as_mut_ptr(), 8);
    }
}

#[inline(always)]
unsafe fn merge_up<T, F>(
    mut src_left: *const T,
    mut src_right: *const T,
    mut dest_ptr: *mut T,
    is_less: &mut F,
) -> (*const T, *const T, *mut T)
where
    F: FnMut(&T, &T) -> bool,
{
    // This is a branchless merge utility function.
    // The equivalent code with a branch would be:
    //
    // if !is_less(&*src_right, &*src_left) {
    //     ptr::copy_nonoverlapping(src_left, dest_ptr, 1);
    //     src_left = src_left.wrapping_add(1);
    // } else {
    //     ptr::copy_nonoverlapping(src_right, dest_ptr, 1);
    //     src_right = src_right.wrapping_add(1);
    // }
    // dest_ptr = dest_ptr.add(1);

    // SAFETY: The caller must guarantee that `src_left`, `src_right` are valid to read and
    // `dest_ptr` is valid to write, while not aliasing.
    unsafe {
        let is_l = !is_less(&*src_right, &*src_left);
        let copy_ptr = if is_l { src_left } else { src_right };
        ptr::copy_nonoverlapping(copy_ptr, dest_ptr, 1);
        src_right = src_right.wrapping_add(!is_l as usize);
        src_left = src_left.wrapping_add(is_l as usize);
        dest_ptr = dest_ptr.add(1);
    }

    (src_left, src_right, dest_ptr)
}

#[inline(always)]
unsafe fn merge_down<T, F>(
    mut src_left: *const T,
    mut src_right: *const T,
    mut dest_ptr: *mut T,
    is_less: &mut F,
) -> (*const T, *const T, *mut T)
where
    F: FnMut(&T, &T) -> bool,
{
    // This is a branchless merge utility function.
    // The equivalent code with a branch would be:
    //
    // if !is_less(&*src_right, &*src_left) {
    //     ptr::copy_nonoverlapping(src_right, dest_ptr, 1);
    //     src_right = src_right.wrapping_sub(1);
    // } else {
    //     ptr::copy_nonoverlapping(src_left, dest_ptr, 1);
    //     src_left = src_left.wrapping_sub(1);
    // }
    // dest_ptr = dest_ptr.sub(1);

    // SAFETY: The caller must guarantee that `src_left`, `src_right` are valid to read and
    // `dest_ptr` is valid to write, while not aliasing.
    unsafe {
        let is_l = !is_less(&*src_right, &*src_left);
        let copy_ptr = if is_l { src_right } else { src_left };
        ptr::copy_nonoverlapping(copy_ptr, dest_ptr, 1);
        src_right = src_right.wrapping_sub(is_l as usize);
        src_left = src_left.wrapping_sub(!is_l as usize);
        dest_ptr = dest_ptr.sub(1);
    }

    (src_left, src_right, dest_ptr)
}

/// Merge v assuming the len is even and v[..len / 2] and v[len / 2..] are sorted.
///
/// Original idea for bi-directional merging by Igor van den Hoven (quadsort), adapted to only use
/// merge up and down. In contrast to the original parity_merge function, it performs 2 writes
/// instead of 4 per iteration. Ord violation detection was added.
pub unsafe fn bi_directional_merge_even<T, F>(v: &[T], dest_ptr: *mut T, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: the caller must guarantee that `dest_ptr` is valid for v.len() writes.
    // Also `v.as_ptr` and `dest_ptr` must not alias.
    //
    // The caller must guarantee that T cannot modify itself inside is_less.
    // merge_up and merge_down read left and right pointers and potentially modify the stack value
    // they point to, if T has interior mutability. This may leave one or two potential writes to
    // the stack value un-observed when dest is copied onto of src.

    // It helps to visualize the merge:
    //
    // Initial:
    //
    //  |ptr_data (in dest)
    //  |ptr_left           |ptr_right
    //  v                   v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //                     ^                   ^
    //                     |t_ptr_left         |t_ptr_right
    //                                         |t_ptr_data (in dest)
    //
    // After:
    //
    //                      |ptr_data (in dest)
    //        |ptr_left     |           |ptr_right
    //        v             v           v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //       ^             ^           ^
    //       |t_ptr_left   |           |t_ptr_right
    //                     |t_ptr_data (in dest)
    //
    //
    // Note, the pointers that have been written, are now one past where they were read and
    // copied. written == incremented or decremented + copy to dest.

    assert!(const { !crate::has_direct_interior_mutability::<T>() });

    let len = v.len();
    let src_ptr = v.as_ptr();

    let len_div_2 = len / 2;

    // SAFETY: No matter what the result of the user-provided comparison function is, all 4 read
    // pointers will always be in-bounds. Writing `ptr_data` and `t_ptr_data` will always be in
    // bounds if the caller guarantees that `dest_ptr` is valid for `v.len()` writes.
    unsafe {
        let mut ptr_left = src_ptr;
        let mut ptr_right = src_ptr.wrapping_add(len_div_2);
        let mut ptr_data = dest_ptr;

        let mut t_ptr_left = src_ptr.wrapping_add(len_div_2 - 1);
        let mut t_ptr_right = src_ptr.wrapping_add(len - 1);
        let mut t_ptr_data = dest_ptr.wrapping_add(len - 1);

        for _ in 0..len_div_2 {
            (ptr_left, ptr_right, ptr_data) = merge_up(ptr_left, ptr_right, ptr_data, is_less);
            (t_ptr_left, t_ptr_right, t_ptr_data) =
                merge_down(t_ptr_left, t_ptr_right, t_ptr_data, is_less);
        }

        let left_diff = (ptr_left as usize).wrapping_sub(t_ptr_left as usize);
        let right_diff = (ptr_right as usize).wrapping_sub(t_ptr_right as usize);

        if !(left_diff == mem::size_of::<T>() && right_diff == mem::size_of::<T>()) {
            panic_on_ord_violation();
        }
    }
}

#[inline(never)]
fn panic_on_ord_violation() -> ! {
    panic!("Ord violation");
}
