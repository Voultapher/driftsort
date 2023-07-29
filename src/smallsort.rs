use core::intrinsics;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

// It's important to differentiate between small-sort performance for small slices and
// small-sort performance sorting small sub-slices as part of the main quicksort loop. For the
// former, testing showed that the representative benchmarks for real-world performance are cold
// CPU state and not single-size hot benchmarks. For the latter the CPU will call them many
// times, so hot benchmarks are fine and more realistic. And it's worth it to optimize sorting
// small sub-slices with more sophisticated solutions than insertion sort.

/// Sorts `v` using strategies optimized for small sizes.
pub fn sort_small<T, F>(v: &mut [T], is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    T::small_sort(v, is_less);
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

impl<T: crate::Freeze> SmallSortTypeImpl for T {
    fn small_sort<F>(v: &mut [Self], is_less: &mut F)
    where
        F: FnMut(&T, &T) -> bool,
    {
        const MAX_SIZE: usize = crate::quicksort::SMALL_SORT_THRESHOLD;

        let len = v.len();

        let mut scratch = MaybeUninit::<[T; MAX_SIZE]>::uninit();
        let scratch_base = scratch.as_mut_ptr() as *mut T;

        if len >= 16 && len <= MAX_SIZE {
            let even_len = len - (len % 2);
            let len_div_2 = even_len / 2;

            // SAFETY: scratch_base is valid and has enough space. And we checked that both
            // v[..len_div_2] and v[len_div_2..] are at least 8 large.
            unsafe {
                let v_base = v.as_mut_ptr();
                sort8_stable(v_base, scratch_base, is_less);
                sort8_stable(v_base.add(len_div_2), scratch_base, is_less);
            }

            insertion_sort_shift_left(&mut v[0..len_div_2], 8, is_less);
            insertion_sort_shift_left(&mut v[len_div_2..], 8, is_less);

            // SAFETY: We checked that T is Freeze and thus observation safe. Should is_less panic v
            // was not modified in parity_merge and retains it's original input. swap and v must not
            // alias and swap has v.len() space.
            unsafe {
                bi_directional_merge_even(&mut v[..even_len], scratch_base, is_less);
                ptr::copy_nonoverlapping(scratch_base, v.as_mut_ptr(), even_len);
            }

            if len != even_len {
                // SAFETY: We know len >= 2.
                unsafe {
                    insert_tail(v, is_less);
                }
            }
        } else if len >= 2 {
            let offset = if len >= 8 {
                // SAFETY: scratch_base is valid and has enough space.
                unsafe {
                    sort8_stable(v.as_mut_ptr(), scratch_base, is_less);
                }

                8
            } else {
                1
            };

            insertion_sort_shift_left(v, offset, is_less);
        }
    }
}

/// SAFETY: The caller MUST guarantee that `v_base` is valid for 4 reads and `dest_ptr` is valid
/// for 4 writes.
pub unsafe fn sort4_stable<T, F>(v_base: *const T, dst: *mut T, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    // By limiting select to picking pointers, we are guaranteed good cmov code-gen regardless of
    // type T layout. Further this only does 5 instead of 6 comparisons compared to a stable
    // transposition 4 element sorting-network. Also by only operating on pointers, we get optimal
    // element copy usage. Doing exactly 1 copy per element.

    // let v_base = v.as_ptr();

    unsafe {
        // Stably create two pairs a <= b and c <= d.
        let c1 = is_less(&*v_base.add(1), &*v_base) as usize;
        let c2 = is_less(&*v_base.add(3), &*v_base.add(2)) as usize;
        let a = v_base.add(c1);
        let b = v_base.add(c1 ^ 1);
        let c = v_base.add(2 + c2);
        let d = v_base.add(2 + (c2 ^ 1));

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

        ptr::copy_nonoverlapping(min, dst, 1);
        ptr::copy_nonoverlapping(lo, dst.add(1), 1);
        ptr::copy_nonoverlapping(hi, dst.add(2), 1);
        ptr::copy_nonoverlapping(max, dst.add(3), 1);
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

/// SAFETY: The caller MUST guarantee that `v_base` is valid for 8 reads and writes, and
/// `scratch_base` is valid for 8 writes.
#[inline(never)]
unsafe fn sort8_stable<T, F>(v_base: *mut T, scratch_base: *mut T, is_less: &mut F)
where
    T: crate::Freeze,
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: The caller must guarantee that scratch_base is valid for 8 writes, and that v_base is
    // valid for 8 reads.
    unsafe {
        sort4_stable(v_base, scratch_base, is_less);
        sort4_stable(v_base.add(4), scratch_base.add(4), is_less);
    }

    // SAFETY: We checked that T is Freeze and thus observation safe.
    // Should is_less panic v was not modified in parity_merge and retains its original input.
    // swap and v must not alias and swap has v.len() space.
    unsafe {
        // It's slightly faster to merge directly into v and copy over the 'safe' elements of swap
        // into v only if there was a panic. This technique is also known as ping-pong merge.
        let drop_guard = DropGuard {
            src: scratch_base,
            dst: v_base,
        };
        bi_directional_merge_even(
            &*ptr::slice_from_raw_parts(scratch_base, 8),
            v_base,
            is_less,
        );
        mem::forget(drop_guard);
    }

    struct DropGuard<T> {
        src: *const T,
        dst: *mut T,
    }

    impl<T> Drop for DropGuard<T> {
        fn drop(&mut self) {
            // SAFETY: `T` is not a zero-sized type, src must hold the original 8 elements of v in
            // any order. And dst must be valid to write 8 elements.
            unsafe {
                ptr::copy_nonoverlapping(self.src, self.dst, 8);
            }
        }
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
unsafe fn bi_directional_merge_even<T, F>(v: &[T], dst: *mut T, is_less: &mut F)
where
    T: crate::Freeze,
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: the caller must guarantee that `dst` is valid for v.len() writes.
    // Also `v.as_ptr` and `dst` must not alias.
    //
    // The caller must guarantee that T cannot modify itself inside is_less.
    // merge_up and merge_down read left and right pointers and potentially modify the stack value
    // they point to, if T has interior mutability. This may leave one or two potential writes to
    // the stack value un-observed when dst is copied onto of src.

    // It helps to visualize the merge:
    //
    // Initial:
    //
    //  |dst (in dst)
    //  |left               |right
    //  v                   v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //                     ^                   ^
    //                     |left_rev           |right_rev
    //                                         |dst_rev (in dst)
    //
    // After:
    //
    //                      |dst (in dst)
    //        |left         |           |right
    //        v             v           v
    // [xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx]
    //       ^             ^           ^
    //       |left_rev     |           |right_rev
    //                     |dst_rev (in dst)
    //
    //
    // Note, the pointers that have been written, are now one past where they were read and
    // copied. written == incremented or decremented + copy to dst.

    let len = v.len();
    let src = v.as_ptr();

    let len_div_2 = len / 2;

    // SAFETY: No matter what the result of the user-provided comparison function is, all 4 read
    // pointers will always be in-bounds. Writing `dst` and `dst_rev` will always be in
    // bounds if the caller guarantees that `dst` is valid for `v.len()` writes.
    unsafe {
        let mut left = src;
        let mut right = src.wrapping_add(len_div_2);
        let mut dst = dst;

        let mut left_rev = src.wrapping_add(len_div_2 - 1);
        let mut right_rev = src.wrapping_add(len - 1);
        let mut dst_rev = dst.wrapping_add(len - 1);

        for _ in 0..len_div_2 {
            (left, right, dst) = merge_up(left, right, dst, is_less);
            (left_rev, right_rev, dst_rev) = merge_down(left_rev, right_rev, dst_rev, is_less);
        }

        let left_diff = (left as usize).wrapping_sub(left_rev as usize);
        let right_diff = (right as usize).wrapping_sub(right_rev as usize);

        if !(left_diff == mem::size_of::<T>() && right_diff == mem::size_of::<T>()) {
            panic_on_ord_violation();
        }
    }
}

#[inline(never)]
fn panic_on_ord_violation() -> ! {
    panic!("Ord violation");
}
