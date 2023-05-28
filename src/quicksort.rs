use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

use crate::has_direct_interior_mutability;

// Switch to a dedicated small array sorting algorithm below this threshold.
pub const SMALL_SORT_THRESHOLD: usize = 20;

// Recursively select a pseudomedian if above this threshold.
const PSEUDO_MEDIAN_REC_THRESHOLD: usize = 64;

/// Sorts `v` recursively using quicksort.
///
/// `limit` ensures we do not stack overflow and do not go quadratic. If reached
/// we switch to purely mergesort by eager sorting.
pub fn stable_quicksort<T, F>(
    mut v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mut limit: u32,
    mut ancestor_pivot: Option<&T>,
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
    // To improve filtering out of common values with equal partition, we remember the
    // ancestor_pivot and use that to compare it to the next pivot selection. Because we can't move
    // the relative position of the pivot in a stable sort and subsequent partitioning may change
    // the position, its easier to simply make a copy of the pivot value and use that for further
    // comparisons.

    loop {
        if v.len() <= SMALL_SORT_THRESHOLD {
            crate::smallsort::sort_small(v, is_less);
            return;
        }

        if limit == 0 {
            crate::drift::sort(v, scratch, true, is_less);
            return;
        }
        limit -= 1;

        let pivot = choose_pivot(v, is_less);

        let mut should_do_equal_partition = false;

        // If the chosen pivot is equal to the ancestor_pivot, then it's the smallest element in the
        // slice. Partition the slice into elements equal to and elements greater than the pivot.
        // This case is usually hit when the slice contains many duplicate elements.
        if let Some(a_pivot) = ancestor_pivot {
            should_do_equal_partition = !is_less(a_pivot, &v[pivot]);
        }

        // SAFETY: See we only use this value for Feeze types, otherwise self-modifications via
        // `is_less` would not be observed and this would be unsound.
        //
        // It's important we do this after we picked the pivot and checked it against the
        // ancestor_pivot, but before we change v again by partitioning.
        let pivot_copy = unsafe { ManuallyDrop::new(ptr::read(&v[pivot])) };

        let mut mid = 0;

        if !should_do_equal_partition {
            mid = stable_partition(v, scratch, pivot, is_less, false);

            // Fallback for non Freeze types.
            should_do_equal_partition = mid == 0;
        }

        if should_do_equal_partition {
            let mid_eq = stable_partition(v, scratch, pivot, &mut |a, b| !is_less(b, a), true);
            v = &mut v[mid_eq..];
            ancestor_pivot = None;
            continue;
        }

        let (left, right) = v.split_at_mut(mid);

        let new_ancestor_pivot = if const { !has_direct_interior_mutability::<T>() } {
            // SAFETY: pivot_copy is valid and initialized, lives on the stack and as a consequence
            // outlives the immediate call to stable_quicksort.
            unsafe { Some(&*(&pivot_copy as *const ManuallyDrop<T> as *const T)) }
        } else {
            None
        };

        // Processing right side with recursion.
        stable_quicksort(right, scratch, limit, new_ancestor_pivot, is_less);

        // Processing left side with next loop iteration.
        v = left;
    }
}

/// Selects a pivot from v. Algorithm taken from glidesort by Orson Peters.
///
/// This chooses a pivot by sampling an adaptive amount of points, approximating
/// the quality of a median of sqrt(n) elements.
fn choose_pivot<T, F>(v: &[T], is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    // SAFETY: TODO
    unsafe {
        // We use unsafe code and raw pointers here because we're dealing with
        // heavy recursion. Passing safe slices around would involve a lot of
        // branches and function call overhead.
        let arr_ptr = v.as_ptr();

        let len_div_8 = len / 8;
        let a = arr_ptr;
        let b = arr_ptr.add(len_div_8 * 4);
        let c = arr_ptr.add(len_div_8 * 7);

        if len < PSEUDO_MEDIAN_REC_THRESHOLD {
            median3(a, b, c, is_less).sub_ptr(arr_ptr)
        } else {
            median3_rec(a, b, c, len_div_8, is_less).sub_ptr(arr_ptr)
        }
    }
}

/// Calculates an approximate median of 3 elements from sections a, b, c, or recursively from an
/// approximation of each, if they're large enough. By dividing the size of each section by 8 when
/// recursing we have logarithmic recursion depth and overall sample from
/// f(n) = 3*f(n/8) -> f(n) = O(n^(log(3)/log(8))) ~= O(n^0.528) elements.
///
/// SAFETY: a, b, c must point to the start of initialized regions of memory of
/// at least n elements.
#[inline(never)]
unsafe fn median3_rec<T, F>(
    mut a: *const T,
    mut b: *const T,
    mut c: *const T,
    n: usize,
    is_less: &mut F,
) -> *const T
where
    F: FnMut(&T, &T) -> bool,
{
    // SAFETY: TODO
    unsafe {
        if n * 8 >= PSEUDO_MEDIAN_REC_THRESHOLD {
            let n8 = n / 8;
            a = median3_rec(a, a.add(n8 * 4), a.add(n8 * 7), n8, is_less);
            b = median3_rec(b, b.add(n8 * 4), b.add(n8 * 7), n8, is_less);
            c = median3_rec(c, c.add(n8 * 4), c.add(n8 * 7), n8, is_less);
        }
        median3(a, b, c, is_less)
    }
}

/// Calculates the median of 3 elements.
///
/// SAFETY: a, b, c must be valid initialized elements.
#[inline(always)]
unsafe fn median3<T, F>(a: *const T, b: *const T, c: *const T, is_less: &mut F) -> *const T
where
    F: FnMut(&T, &T) -> bool,
{
    // Compiler tends to make this branchless when sensible, and avoids the
    // third comparison when not.
    unsafe {
        let x = is_less(&*a, &*b);
        let y = is_less(&*a, &*c);
        if x == y {
            // If x=y=0 then b, c <= a. In this case we want to return max(b, c).
            // If x=y=1 then a < b, c. In this case we want to return min(b, c).
            // By toggling the outcome of b < c using XOR x we get this behavior.
            let z = is_less(&*b, &*c);

            if z ^ x {
                c
            } else {
                b
            }
        } else {
            // Either c <= a < b or b <= a < c, thus a is our median.
            a
        }
    }
}

// FIXME remove
fn stable_partition<T, F>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot_pos: usize,
    is_less: &mut F,
    pivot_goes_left: bool, // FIXME remove
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    if v.len() >= 128 && pivot_pos != v.len() - 1 {
        // // FIXME testing.
        // unsafe {
        //     type DebugT = i32;
        //     let mut v_copy = mem::transmute::<&[T], &[DebugT]>(v).to_vec();
        //     let v_check = mem::transmute::<&mut [DebugT], &mut [T]>(&mut v_copy);
        //     let out_check =
        //         stable_partition_simple(v_check, scratch, pivot_pos, is_less, pivot_goes_left);

        //     let out = stable_partition_bi(v, scratch, pivot_pos, is_less, pivot_goes_left);

        //     assert_eq!(mem::transmute::<&[T], &[DebugT]>(v), &v_copy,);
        //     assert_eq!(out, out_check);

        //     out
        // }

        stable_partition_bi(v, scratch, pivot_pos, is_less, pivot_goes_left)
    } else {
        stable_partition_simple(v, scratch, pivot_pos, is_less, pivot_goes_left)
    }
}

/// This is a branchless version of:
///
/// ```rust
/// if $is_less(&*$elem_ptr, $pivot) {
///     ptr::copy_nonoverlapping($elem_ptr, $lt_out_ptr, 1);
///     $lt_out_ptr = $lt_out_ptr.add(1);
/// } else {
///     ptr::copy_nonoverlapping($elem_ptr, $ge_out_ptr, 1);
///     $ge_out_ptr = $ge_out_ptr.sub(1);
/// }
///
/// $elem_ptr = $elem_ptr.add(1);
/// ```
///
/// This is a macro to avoid function call overhead for debug builds. And it simplifies value
/// mutation.
macro_rules! partition_select {
    ($elem_ptr:expr, $lt_out_ptr:expr, $ge_out_ptr:expr, $pivot:expr, $is_less:expr) => {{
        let is_l = $is_less(&*$elem_ptr, $pivot);
        let out_ptr = if is_l { $lt_out_ptr } else { $ge_out_ptr };
        ptr::copy_nonoverlapping($elem_ptr, out_ptr, 1);
        $elem_ptr = $elem_ptr.add(1);
        $lt_out_ptr = $lt_out_ptr.add(is_l as usize);
        // TODO is that sub thing really better?
        $ge_out_ptr = $ge_out_ptr.sub(1).add(is_l as usize);
    }};
}

// TODO check perf impact of using partition_select_out in partition_select.
macro_rules! partition_select_out {
    ($lt_out_ptr:expr, $ge_out_ptr:expr, $cond:expr) => {{
        let out_ptr = if $cond { $lt_out_ptr } else { $ge_out_ptr };
        $lt_out_ptr = $lt_out_ptr.add($cond as usize);
        // TODO is that sub thing really better?
        $ge_out_ptr = $ge_out_ptr.sub(1).add($cond as usize);
        out_ptr
    }};
}
/// Partitions `v` into elements smaller than `pivot`, followed by elements
/// greater than or equal to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
fn stable_partition_bi<T, F>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot_pos: usize,
    is_less: &mut F,
    pivot_goes_left: bool,
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    // The idea is to interleave data-independent iterations inside the same loop iteration. This
    // greatly improves instruction-level parallelism (ILP). Calling memcpy is expensive especially
    // if len(v) is small. We want to avoid the branch to check if the element we are comparing to
    // is the pivot element. The naive tiling would have 4 memcpy and 4 reverse copies. By being
    // careful how we split up the slice we can reduce this down to 2 memcpy and 2 reverse copies.

    // First we split the slice into 4 regions. So that len(A1) == len(A2) and len(B1) == len(B2),
    // and len(A1) + len(B1) == len(v) / 2.
    //
    // e == regular element
    // p == pivot element
    // x == extra element that doesn't fit into even len parts.
    // c == chop off element because of uneven len
    //
    // E.g. len(v) == 36, pivot_pos == 7:
    // [eeeeeeep|eeeeeeeeee|eeeeeeex|eeeeeeeeee]
    //    A1         B1        A2         B2     len(A1) == 7, len(B1) == 10
    //
    // E.g. len(v) == 36, pivot_pos == 19:
    // [ex|eeeeeeeeeeeeeeee|ep|eeeeeeeeeeeeeeee]
    //  A1        B1        A2        B2         len(A1) == 1, len(B1) == 16
    //
    // E.g. len(v) == 37, pivot_pos == 0:
    // [p|eeeeeeeeeeeeeeeee|x|eeeeeeeeeeeeeeeeec]
    //  A1        B1        A2        B2         len(A1) == 0, len(B1) == 17
    //
    // E.g. len(v) == 37, pivot_pos == 20:
    // [eex|eeeeeeeeeeeeeee|eep|eeeeeeeeeeeeeeec]
    //  A1        B1        A2        B2         len(A1) == 2, len(B1) == 15
    //
    // E.g. len(v) == 37, pivot_pos == 36: TODO
    // [eeeeeeeeeeeeeeeee||eeeeeeeeeeeeeeeeep|]
    //        A1         B1        A2        B2  len(A1) == 17, len(B1) == 0
    //
    // First pivot step in scratch, A1 and A2 in parallel growing from ends and middle.
    // scratch: [->   A1    <-|->    A2   <-]
    //          [A1 |    | A1 | A2 |    | A2]
    //          [<  |    | >= | <  |    | >=]
    //
    // Second pivot step in scratch, B1 and B2 in parallel growing from previous call bounds.
    // scratch: [A1 | B1 | A1 | A2 | B2 | A2]
    //          [<  |-><-| >= | <  |-><-| >=]
    //          [<  |<|>=| >= | <  |<|>=| >=]

    let len = v.len();
    let arr_ptr = v.as_mut_ptr();

    assert!(scratch.len() >= len && pivot_pos < len);
    let scratch_ptr = MaybeUninit::slice_as_mut_ptr(scratch);

    // SAFETY: TODO
    let pivot_guard = unsafe {
        PivotGuard {
            value: ManuallyDrop::new(ptr::read(&v[pivot_pos])),
            hole: arr_ptr.add(pivot_pos),
        }
    };
    let pivot: &T = &pivot_guard.value;

    // Maybe saturating sub?
    // dbg!(len, pivot_pos, pivot_goes_left);

    let len_div_2 = len / 2;
    let even_len = len - (len % 2 != 0) as usize;
    let pivot_on_left_side = pivot_pos < len_div_2;

    let a_len = pivot_pos - if pivot_on_left_side { 0 } else { len_div_2 };
    let b_len = len_div_2 - (a_len + 1);
    // println!("\nlen: {len}, pivot_pos: {pivot_pos} a_len: {a_len} b_len: {b_len}");

    // debug_assert!((a_len + b_len + is_even_len as usize) == len_div_2);

    // SAFETY:
    // - x_ptr[..len] is valid to read
    // - x_ptr[..len] does not overlap with pivot (relevant if T has interior mutability)
    // - x_lt_out_ptr[..len] is valid to write
    // - x_ge_out_ptr.sub(len)[..len] is valid to write
    unsafe {
        // // FIXME
        // type DebugT = i32;
        // let v_as_x = std::mem::transmute::<&[T], &[DebugT]>(v);
        // let scratch_as_x = std::mem::transmute::<&[MaybeUninit<T>], &[DebugT]>(&scratch[..len]);
        // for i in 0..len {
        //     (scratch_ptr.add(i) as *mut DebugT).write(0);
        // }
        // println!("v: {v_as_x:?}");
        // println!("s: {scratch_as_x:?}");

        // lt == less than, ge == greater or equal
        let mut a1_ptr = arr_ptr;
        let mut a1_lt_out_ptr = scratch_ptr;
        let mut a1_ge_out_ptr = scratch_ptr.add(len_div_2 - 1);

        let mut a2_ptr = arr_ptr.add(len_div_2);
        let mut a2_lt_out_ptr = scratch_ptr.add(len_div_2);
        let mut a2_ge_out_ptr = scratch_ptr.add(even_len - 1);

        // println!(
        //     "a1_lt_out_ptr pos: {}, a1_ge_out_ptr pos: {}",
        //     a1_lt_out_ptr.sub_ptr(scratch_ptr),
        //     a1_ge_out_ptr.sub_ptr(scratch_ptr)
        // );

        // TODO perf of for _ in 0..a_len loop vs while a1_ptr -> end
        // let end_ptr = arr_ptr.add(a_len);
        // while a1_ptr < end_ptr {
        for _ in 0..a_len {
            partition_select!(a1_ptr, a1_lt_out_ptr, a1_ge_out_ptr, pivot, is_less);
            partition_select!(a2_ptr, a2_lt_out_ptr, a2_ge_out_ptr, pivot, is_less);
        }

        // println!(
        //     "a1_ptr pos: {} a2_ptr pos: {}",
        //     a1_ptr.sub_ptr(arr_ptr),
        //     a2_ptr.sub_ptr(arr_ptr)
        // );

        // println!(
        //     "a1_lt_out_ptr pos: {}, a1_ge_out_ptr pos: {}",
        //     a1_lt_out_ptr.sub_ptr(scratch_ptr),
        //     a1_ge_out_ptr.sub_ptr(scratch_ptr)
        // );

        // println!("s1: {scratch_as_x:?}");
        // Remember where to copy pivot into, and take care of x elem.
        let pivot_hole_ptr;
        if pivot_on_left_side {
            pivot_hole_ptr = partition_select_out!(a1_lt_out_ptr, a1_ge_out_ptr, pivot_goes_left);
            a1_ptr = a1_ptr.add(1);

            // x elem is on the right side.
            partition_select!(a2_ptr, a2_lt_out_ptr, a2_ge_out_ptr, pivot, is_less);
        } else {
            pivot_hole_ptr = partition_select_out!(a2_lt_out_ptr, a2_ge_out_ptr, pivot_goes_left);
            a2_ptr = a2_ptr.add(1);

            // x elem is on the left side.
            partition_select!(a1_ptr, a1_lt_out_ptr, a1_ge_out_ptr, pivot, is_less);
        }
        // println!("s2: {scratch_as_x:?}");

        // println!(
        //     "a1_lt_out_ptr pos: {}, a1_ge_out_ptr pos: {}",
        //     a1_lt_out_ptr.sub_ptr(scratch_ptr),
        //     a1_ge_out_ptr.sub_ptr(scratch_ptr)
        // );

        let mut b1_ptr = a1_ptr;
        let mut b1_lt_out_ptr = a1_lt_out_ptr;
        let mut b1_ge_out_ptr = a1_ge_out_ptr;

        let mut b2_ptr = a2_ptr;
        let mut b2_lt_out_ptr = a2_lt_out_ptr;
        let mut b2_ge_out_ptr = a2_ge_out_ptr;

        // let end_ptr = arr_ptr.add(len);
        for _ in 0..b_len {
            partition_select!(b1_ptr, b1_lt_out_ptr, b1_ge_out_ptr, pivot, is_less);
            partition_select!(b2_ptr, b2_lt_out_ptr, b2_ge_out_ptr, pivot, is_less);
        }

        // println!("s3: {scratch_as_x:?}");

        // println!(
        //     "pivot val: {}",
        //     &*(&pivot_guard.value as &T as *const T as *const DebugT)
        // );

        let a1_b1_lt_len = b1_lt_out_ptr.sub_ptr(scratch_ptr);
        let a2_b2_lt_len = b2_lt_out_ptr.sub_ptr(scratch_ptr.add(len_div_2));

        let c_elem_ptr = arr_ptr.add(len - 1);
        let mut c_elem_lt = false;
        if len != even_len {
            // Take care of c elem. It sits at the end of b2. It will have to either go to the end
            // of the lt elements or go to the end of v, where it already is.
            // TODO branchless.
            c_elem_lt = is_less(&*c_elem_ptr, pivot);
        }

        // Now that any possible observation of pivot has happened we copy it.
        // println!(
        //     "pivot_hole_ptr pos: {} val: {}",
        //     pivot_hole_ptr.sub_ptr(scratch_ptr),
        //     *(pivot_hole_ptr as *const DebugT)
        // );
        ptr::copy_nonoverlapping(pivot, pivot_hole_ptr, 1);
        mem::forget(pivot_guard);
        // println!(
        //     "sP: {scratch_as_x:?} pos: {}",
        //     pivot_hole_ptr.sub_ptr(scratch_ptr)
        // );

        // Copy the elements that were less than from scratch into v.
        let mut out_ptr = arr_ptr;
        ptr::copy_nonoverlapping(scratch_ptr, out_ptr, a1_b1_lt_len);
        out_ptr = out_ptr.add(a1_b1_lt_len);
        // println!("v1: {v_as_x:?}");

        ptr::copy_nonoverlapping(scratch_ptr.add(len_div_2), out_ptr, a2_b2_lt_len);
        out_ptr = out_ptr.add(a2_b2_lt_len);
        // println!("v2: {v_as_x:?}");

        if c_elem_lt {
            ptr::copy_nonoverlapping(c_elem_ptr, out_ptr, 1);
            out_ptr = out_ptr.add(1);
        }

        let lt_len = out_ptr.sub_ptr(v.as_ptr());

        // Copy the elements that were equal or more from scratch into v and reverse them.
        // TODO maybe loop fusion here for the common part of 1 and 2.
        let mut a1_b1_ge_ptr = scratch_ptr.add(len_div_2 - 1);
        while a1_b1_ge_ptr >= b1_lt_out_ptr {
            // TODO bench !=
            // TODO is a for i loop better?

            ptr::copy_nonoverlapping(a1_b1_ge_ptr, out_ptr, 1);
            out_ptr = out_ptr.add(1);
            a1_b1_ge_ptr = a1_b1_ge_ptr.sub(1);
        }
        // println!("v3: {v_as_x:?}");

        let mut a2_b2_ge_ptr = scratch_ptr.add(even_len - 1);
        while a2_b2_ge_ptr >= b2_lt_out_ptr {
            // TODO is a for i loop better?
            // assert!(out_ptr.sub_ptr(arr_ptr) < len);
            // println!(
            //     "copying {}, pos {} -> {}",
            //     *(a2_b2_ge_ptr as *const DebugT),
            //     a2_b2_ge_ptr.sub_ptr(scratch_ptr),
            //     out_ptr.sub_ptr(arr_ptr)
            // );

            ptr::copy_nonoverlapping(a2_b2_ge_ptr, out_ptr, 1);
            out_ptr = out_ptr.add(1);
            a2_b2_ge_ptr = a2_b2_ge_ptr.sub(1);
        }
        // println!("v4: {v_as_x:?}");

        lt_len
    }
}

// It's crucial that pivot_hole will be copied back to the input if any comparison in the
// loop panics. Because it could have changed due to interior mutability.
struct PivotGuard<T> {
    value: ManuallyDrop<T>,
    hole: *mut T,
}

impl<T> Drop for PivotGuard<T> {
    fn drop(&mut self) {
        unsafe {
            ptr::copy_nonoverlapping(&*self.value, self.hole, 1);
        }
    }
}

/// Partitions `v` into elements smaller than `pivot`, followed by elements
/// greater than or equal to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
fn stable_partition_simple<T, F>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot_pos: usize,
    is_less: &mut F,
    pivot_goes_left: bool,
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    let arr = v.as_mut_ptr();

    // Inside the main partitioning loop we MUST NOT compare out stack copy of the pivot value with
    // the original value in the slice `v`. If we just write the value as pointed to by `src` into
    // `buf` as it was in the input slice `v` we would risk that the call to the user-provided
    // `is_less` modifies the value pointed to by `src`. This could be UB for types such as
    // `Mutex<Option<Box<String>>>` where during the comparison it replaces the box with None,
    // leading to double free. As the value written back into `v` from `buf` did not observe that
    // modification.

    // SAFETY: TODO
    unsafe {
        assert!(scratch.len() >= len);
        let buf = MaybeUninit::slice_as_mut_ptr(scratch);

        let pivot = PivotGuard {
            value: ManuallyDrop::new(ptr::read(&v[pivot_pos])),
            hole: arr.add(pivot_pos),
        };

        let mut pivot_partioned_ptr = ptr::null_mut();
        let mut l_count = 0;
        let mut reverse_out = buf.add(len);
        for i in 0..len {
            reverse_out = reverse_out.sub(1);

            // This should only happen once and should be predicted very well. This is required to
            // handle types with interior mutability. See comment above for more info.
            if i == pivot_pos {
                // We move the pivot in its correct place later.
                if pivot_goes_left {
                    pivot_partioned_ptr = buf.add(l_count);
                    l_count += 1;
                } else {
                    pivot_partioned_ptr = reverse_out.add(l_count);
                }
                continue;
            }

            let src = arr.add(i);
            let less_than_pivot = is_less(&*src, &pivot.value);
            let dst = if less_than_pivot {
                buf.add(l_count)
            } else {
                reverse_out.add(l_count)
            };
            ptr::copy_nonoverlapping(src, dst, 1);

            l_count += less_than_pivot as usize;
        }

        // Move pivot into its correct position.
        ptr::copy_nonoverlapping(&*pivot.value, pivot_partioned_ptr, 1);
        mem::forget(pivot);

        // Copy all the elements that were not equal directly from swap to v.
        ptr::copy_nonoverlapping(buf, arr, l_count);

        // Copy the elements that were equal or more from the buf into v and reverse them.
        let rev_buf_ptr = buf.add(len - 1);
        for i in 0..len - l_count {
            ptr::copy_nonoverlapping(rev_buf_ptr.sub(i), arr.add(l_count + i), 1);
        }

        l_count
    }
}
