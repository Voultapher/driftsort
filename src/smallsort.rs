use core::intrinsics;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

// It's important to differentiate between small-sort performance for small slices and
// small-sort performance sorting small sub-slices as part of the main quicksort loop. For the
// former, testing showed that the representative benchmarks for real-world performance are cold
// CPU state and not single-size hot benchmarks. For the latter the CPU will call them many
// times, so hot benchmarks are fine and more realistic. And it's worth it to optimize sorting
// small sub-slices with more sophisticated solutions than insertion sort.

// Use a trait to focus code-gen on only the parts actually relevant for the type. Avoid generating
// LLVM-IR for the sorting-network and median-networks for types that don't qualify.
pub(crate) trait SmallSortTypeImpl: Sized {
    const MAX_LEN_SMALL_SORT: usize;

    /// Sorts `v` using strategies optimized for small sizes.
    fn sort_small<F>(v: &mut [Self], scratch: &mut [MaybeUninit<Self>], is_less: &mut F)
    where
        F: FnMut(&Self, &Self) -> bool;
}

impl<T> SmallSortTypeImpl for T {
    default const MAX_LEN_SMALL_SORT: usize = 16;

    default fn sort_small<F>(v: &mut [Self], _scratch: &mut [MaybeUninit<Self>], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        if v.len() >= 2 {
            insertion_sort_shift_left(v, 1, is_less);
        }
    }
}

pub(crate) const MIN_SMALL_SORT_SCRATCH_LEN: usize = i32::MAX_LEN_SMALL_SORT + 16;

impl<T> SmallSortTypeImpl for T
where
    T: crate::Freeze,
    (): crate::IsTrue<{ mem::size_of::<T>() <= 96 }>,
{
    const MAX_LEN_SMALL_SORT: usize = 20;

    fn sort_small<F>(v: &mut [Self], scratch: &mut [MaybeUninit<Self>], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        let len = v.len();

        if len >= 2 {
            if scratch.len() < MIN_SMALL_SORT_SCRATCH_LEN {
                intrinsics::abort();
            }

            let v_base = v.as_mut_ptr();

            let offset = if len >= 8 {
                let len_div_2 = len / 2;

                // SAFETY: TODO
                unsafe {
                    // Help the compiler, avoids unnecessary panic code because of slicing.
                    intrinsics::assume(v.len() > len_div_2);
                    intrinsics::assume(scratch.len() > len_div_2);

                    let scratch_base_m = scratch.as_mut_ptr();

                    let presorted_len = if len >= 16 {
                        // SAFETY: scratch_base is valid and has enough space.
                        sort8_stable(
                            v,
                            &mut *ptr::slice_from_raw_parts_mut(
                                scratch_base_m.add(T::MAX_LEN_SMALL_SORT),
                                8,
                            ),
                            &mut *ptr::slice_from_raw_parts_mut(scratch_base_m, 8),
                            is_less,
                        );

                        sort8_stable(
                            &v[len_div_2..],
                            &mut *ptr::slice_from_raw_parts_mut(
                                scratch_base_m.add(T::MAX_LEN_SMALL_SORT + 8),
                                8,
                            ),
                            &mut *ptr::slice_from_raw_parts_mut(scratch_base_m.add(len_div_2), 8),
                            is_less,
                        );

                        8
                    } else {
                        // SAFETY: scratch_base is valid and has enough space.
                        sort4_stable(v, scratch, is_less);
                        sort4_stable(
                            &*ptr::slice_from_raw_parts(v_base.add(len_div_2), 4),
                            &mut scratch[len_div_2..],
                            is_less,
                        );

                        4
                    };

                    let scratch_base = scratch.as_mut_ptr() as *mut T;

                    for offset in [0, len_div_2] {
                        let src = scratch_base.add(offset);
                        let dst = v_base.add(offset);

                        for i in presorted_len..len_div_2 {
                            ptr::copy_nonoverlapping(dst.add(i), src.add(i), 1);
                            let scratch_slice = &mut *ptr::slice_from_raw_parts_mut(src, i + 1);
                            insert_tail(scratch_slice, is_less);
                        }
                    }

                    let even_len = len - (len % 2);

                    // See comment in `DropGuard::drop`.
                    let drop_guard = DropGuard {
                        src: scratch_base,
                        dst: v_base,
                        len: even_len,
                    };

                    // It's faster to merge directly into `v` and copy over the 'safe' elements of
                    // `scratch` into v only if there was a panic. This technique is similar to
                    // ping-pong merging.
                    bi_directional_merge_even(
                        &*ptr::slice_from_raw_parts(drop_guard.src, drop_guard.len),
                        &mut *ptr::slice_from_raw_parts_mut(
                            drop_guard.dst as *mut MaybeUninit<T>,
                            drop_guard.len,
                        ),
                        is_less,
                    );
                    mem::forget(drop_guard);

                    even_len
                }
            } else {
                1
            };

            insertion_sort_shift_left(v, offset, is_less);
        }

        struct DropGuard<T> {
            src: *mut T,
            dst: *mut T,
            len: usize,
        }

        impl<T> Drop for DropGuard<T> {
            fn drop(&mut self) {
                // SAFETY: `src` must hold the original `len` elements of `v` in any order. And dst
                // must be valid to write `len` elements.
                unsafe {
                    ptr::copy_nonoverlapping(self.src, self.dst, self.len);
                }
            }
        }
    }
}

struct GapGuard<T> {
    pos: *mut T,
    value: ManuallyDrop<T>,
}

impl<T> Drop for GapGuard<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::copy_nonoverlapping(&*self.value, self.pos, 1);
        }
    }
}

/// Inserts `v[v.len() - 1]` into pre-sorted sequence `v[..v.len() - 1]` so that whole `v[..]`
/// becomes sorted. Returns the insert position.
/// Inserts `v[v.len() - 1]` into pre-sorted sequence `v[..v.len() - 1]` so that whole `v[..]`
/// becomes sorted.
unsafe fn insert_tail<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    if v.len() < 2 {
        intrinsics::abort();
    }

    let v_base = v.as_mut_ptr();
    let i = v.len() - 1;

    // SAFETY: caller must ensure v is at least len 2.
    unsafe {
        // See insert_head which talks about why this approach is beneficial.
        let v_i = v_base.add(i);

        // It's important that we use v_i here. If this check is positive and we continue,
        // We want to make sure that no other copy of the value was seen by is_less.
        // Otherwise we would have to copy it back.
        if is_less(&*v_i, &*v_i.sub(1)) {
            // It's important, that we use tmp for comparison from now on. As it is the value that
            // will be copied back. And notionally we could have created a divergence if we copy
            // back the wrong value.
            // Intermediate state of the insertion process is always tracked by `gap`, which
            // serves two purposes:
            // 1. Protects integrity of `v` from panics in `is_less`.
            // 2. Fills the remaining gap in `v` in the end.
            //
            // Panic safety:
            //
            // If `is_less` panics at any point during the process, `gap` will get dropped and
            // fill the gap in `v` with `tmp`, thus ensuring that `v` still holds every object it
            // initially held exactly once.
            let mut gap = GapGuard {
                pos: v_i.sub(1),
                value: mem::ManuallyDrop::new(ptr::read(v_i)),
            };
            ptr::copy_nonoverlapping(gap.pos, v_i, 1);

            // SAFETY: We know i is at least 1.
            for j in (0..(i - 1)).rev() {
                let v_j = v_base.add(j);
                if !is_less(&*gap.value, &*v_j) {
                    break;
                }

                ptr::copy_nonoverlapping(v_j, gap.pos, 1);
                gap.pos = v_j;
            }
            // `gap` gets dropped and thus copies `tmp` into the remaining gap in `v`.
        }
    }
}

/// Sort `v` assuming `v[..offset]` is already sorted.
pub fn insertion_sort_shift_left<T, F>(v: &mut [T], offset: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    if offset == 0 || offset > len {
        intrinsics::abort();
    }

    // Shift each element of the unsorted region v[i..] as far left as is needed to make v sorted.
    for i in offset..len {
        // SAFETY: we tested that `offset` must be at least 1, so this loop is only entered if len
        // >= 2.
        unsafe {
            insert_tail(&mut v[..=i], is_less);
        }
    }
}

/// SAFETY: The caller MUST guarantee that `v_base` is valid for 4 reads and `dest_ptr` is valid
/// for 4 writes. The result will be stored in `dst[0..4]`.
pub unsafe fn sort4_stable<T, F>(v: &[T], dst: &mut [MaybeUninit<T>], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // By limiting select to picking pointers, we are guaranteed good cmov code-gen regardless of
    // type T layout. Further this only does 5 instead of 6 comparisons compared to a stable
    // transposition 4 element sorting-network. Also by only operating on pointers, we get optimal
    // element copy usage. Doing exactly 1 copy per element.

    let v_base = v.as_ptr();
    let dst_base = dst.as_mut_ptr() as *mut T;

    unsafe {
        // Stably create two pairs a <= b and c <= d.
        let c1 = is_less(&*v_base.add(1), &*v_base);
        let c2 = is_less(&*v_base.add(3), &*v_base.add(2));
        let a = v_base.add(c1 as usize);
        let b = v_base.add(!c1 as usize);
        let c = v_base.add(2 + c2 as usize);
        let d = v_base.add(2 + (!c2 as usize));

        // Compare (a, c) and (b, d) to identify max/min. We're left with two
        // unknown elements, but because we are a stable sort we must know which
        // one is leftmost and which one is rightmost.
        // c3, c4 | min max unknown_left unknown_right
        //  0,  0 |  a   d    b         c
        //  0,  1 |  a   b    c         d
        //  1,  0 |  c   d    a         b
        //  1,  1 |  c   b    a         d
        let c3 = is_less(&*c, &*a);
        let c4 = is_less(&*d, &*b);
        let min = select(c3, c, a);
        let max = select(c4, b, d);
        let unknown_left = select(c3, a, select(c4, c, b));
        let unknown_right = select(c4, d, select(c3, b, c));

        // Sort the last two unknown elements.
        let c5 = is_less(&*unknown_right, &*unknown_left);
        let lo = select(c5, unknown_right, unknown_left);
        let hi = select(c5, unknown_left, unknown_right);

        ptr::copy_nonoverlapping(min, dst_base, 1);
        ptr::copy_nonoverlapping(lo, dst_base.add(1), 1);
        ptr::copy_nonoverlapping(hi, dst_base.add(2), 1);
        ptr::copy_nonoverlapping(max, dst_base.add(3), 1);
    }

    #[inline(always)]
    pub fn select<T>(cond: bool, if_true: *const T, if_false: *const T) -> *const T {
        if cond {
            if_true
        } else {
            if_false
        }
    }
}

/// SAFETY: The caller MUST guarantee that `v_base` is valid for 8 reads and writes, `scratch_base`
/// and `dst` MUST be valid for 8 writes. The result will be stored in `dst[0..8]`.
unsafe fn sort8_stable<T, F>(
    v: &[T],
    scratch: &mut [MaybeUninit<T>],
    dst: &mut [MaybeUninit<T>],
    is_less: &mut F,
) where
    T: crate::Freeze,
    F: FnMut(&T, &T) -> bool,
{
    intrinsics::assume(v.len() >= 8 && scratch.len() >= 8 && dst.len() >= 8);

    // SAFETY: The caller must guarantee that scratch_base is valid for 8 writes, and that v_base is
    // valid for 8 reads.
    unsafe {
        sort4_stable(v, scratch, is_less);
        sort4_stable(&v[4..], &mut scratch[4..], is_less);
    }

    // SAFETY: TODO
    unsafe {
        bi_directional_merge_even(
            &*ptr::slice_from_raw_parts(scratch.as_ptr() as *const T, 8),
            dst,
            is_less,
        );
    }
}

#[inline(always)]
unsafe fn merge_up<T, F>(
    mut left_src: *const T,
    mut right_src: *const T,
    mut dst: *mut T,
    is_less: &mut F,
) -> (*const T, *const T, *mut T)
where
    F: FnMut(&T, &T) -> bool,
{
    // This is a branchless merge utility function.
    // The equivalent code with a branch would be:
    //
    // if !is_less(&*right_src, &*left_src) {
    //     ptr::copy_nonoverlapping(left_src, dst, 1);
    //     left_src = left_src.wrapping_add(1);
    // } else {
    //     ptr::copy_nonoverlapping(right_src, dst, 1);
    //     right_src = right_src.wrapping_add(1);
    // }
    // dst = dst.add(1);

    // SAFETY: The caller must guarantee that `left_src`, `right_src` are valid to read and
    // `dst` is valid to write, while not aliasing.
    unsafe {
        let is_l = !is_less(&*right_src, &*left_src);
        let src = if is_l { left_src } else { right_src };
        ptr::copy_nonoverlapping(src, dst, 1);
        right_src = right_src.wrapping_add(!is_l as usize);
        left_src = left_src.wrapping_add(is_l as usize);
        dst = dst.add(1);
    }

    (left_src, right_src, dst)
}

#[inline(always)]
unsafe fn merge_down<T, F>(
    mut left_src: *const T,
    mut right_src: *const T,
    mut dst: *mut T,
    is_less: &mut F,
) -> (*const T, *const T, *mut T)
where
    F: FnMut(&T, &T) -> bool,
{
    // This is a branchless merge utility function.
    // The equivalent code with a branch would be:
    //
    // if !is_less(&*right_src, &*left_src) {
    //     ptr::copy_nonoverlapping(right_src, dst, 1);
    //     right_src = right_src.wrapping_sub(1);
    // } else {
    //     ptr::copy_nonoverlapping(left_src, dst, 1);
    //     left_src = left_src.wrapping_sub(1);
    // }
    // dst = dst.sub(1);

    // SAFETY: The caller must guarantee that `left_src`, `right_src` are valid to read and
    // `dst` is valid to write, while not aliasing.
    unsafe {
        let is_l = !is_less(&*right_src, &*left_src);
        let src = if is_l { right_src } else { left_src };
        ptr::copy_nonoverlapping(src, dst, 1);
        right_src = right_src.wrapping_sub(is_l as usize);
        left_src = left_src.wrapping_sub(!is_l as usize);
        dst = dst.sub(1);
    }

    (left_src, right_src, dst)
}

/// Merge v assuming the len is even and v[..len / 2] and v[len / 2..] are sorted.
///
/// Original idea for bi-directional merging by Igor van den Hoven (quadsort), adapted to only use
/// merge up and down. In contrast to the original parity_merge function, it performs 2 writes
/// instead of 4 per iteration. Ord violation detection was added.
///
// SAFETY: the caller must guarantee that `dst` is valid for v.len() writes.
// Also `v.as_ptr` and `dst` must not alias.
unsafe fn bi_directional_merge_even<T, F>(v: &[T], dst: &mut [MaybeUninit<T>], is_less: &mut F)
where
    T: crate::Freeze,
    F: FnMut(&T, &T) -> bool,
{
    // The caller must guarantee that T cannot modify itself inside is_less.
    // merge_up and merge_down read left and right pointers and potentially modify the stack value
    // they point to, if T has interior mutability. This may leave one or two potential writes to
    // the stack value un-observed when dst is copied onto of src.

    // It helps to visualize the merge:
    //
    // Initial:
    //
    //  |dst_fwd (in dst)
    //  |left_fwd           |right_fwd
    //  v                   v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //                     ^                   ^
    //                     |left_rev           |right_rev
    //                                         |dst_rev (in dst)
    //
    // After:
    //
    //                      |dst_fwd (in dst)
    //        |left_fwd     |           |right_fwd
    //        v             v           v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //       ^             ^           ^
    //       |left_rev     |           |right_rev
    //                     |dst_rev (in dst)
    //
    //
    // Note, the pointers that have been written, are now one past where they were read and
    // copied. written == incremented or decremented + copy to dst.

    // This can avoid useless code-gen.
    intrinsics::assume(v.len() >= 2 && v.len() % 2 == 0 && v.len() == dst.len());

    let len = v.len();
    let v_base = v.as_ptr();
    let dst_base = dst.as_mut_ptr() as *mut T;

    let len_div_2 = len / 2;

    // SAFETY: No matter what the result of the user-provided comparison function is, all 4 read
    // pointers will always be in-bounds. Writing `dst` and `dst_rev` will always be in
    // bounds if the caller guarantees that `dst` is valid for `v.len()` writes.
    unsafe {
        let mut left_fwd = v_base;
        let mut right_fwd = v_base.wrapping_add(len_div_2);
        let mut dst_fwd = dst_base;

        let mut left_rev = v_base.wrapping_add(len_div_2 - 1);
        let mut right_rev = v_base.wrapping_add(len - 1);
        let mut dst_rev = dst_base.wrapping_add(len - 1);

        for _ in 0..len_div_2 {
            (left_fwd, right_fwd, dst_fwd) = merge_up(left_fwd, right_fwd, dst_fwd, is_less);
            (left_rev, right_rev, dst_rev) = merge_down(left_rev, right_rev, dst_rev, is_less);
        }

        let left_diff = (left_fwd as usize).wrapping_sub(left_rev as usize);
        let right_diff = (right_fwd as usize).wrapping_sub(right_rev as usize);

        if !(left_diff == mem::size_of::<T>() && right_diff == mem::size_of::<T>()) {
            panic_on_ord_violation();
        }
    }
}

#[inline(never)]
fn panic_on_ord_violation() -> ! {
    panic!("Ord violation");
}
