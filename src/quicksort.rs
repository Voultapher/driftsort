use core::intrinsics;
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
            mid = stable_partition(v, scratch, pivot, is_less);

            // Fallback for non Freeze types.
            should_do_equal_partition = mid == 0;
        }

        if should_do_equal_partition {
            let mid_eq = stable_partition(v, scratch, pivot, &mut |a, b| !is_less(b, a));
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

/// Selects a pivot from `v`. Algorithm taken from glidesort by Orson Peters.
///
/// This chooses a pivot by sampling an adaptive amount of points, approximating
/// the quality of a median of sqrt(n) elements.
fn choose_pivot<T, F>(v: &[T], is_less: &mut F) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    // SAFETY: The pointer operations are guaranteed to be in-bounds no matter the len of `v`. From
    // which follows the calls to median3 and median3_rec are provided with pointers to valid
    // elements and thus safe.
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
    // SAFETY: See function safety description.
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

/// Takes the input slice `v` and re-arranges elements such that when the call returns normally
/// all elements that compare true for `is_less(elem, pivot)` where `pivot == v[pivot_pos]` are
/// on the left side of `v` followed by the other elements, notionally considered greater or
/// equal to `pivot`.
///
/// Returns the number of elements that are compared true for `is_less(elem, pivot)`.
///
/// If `is_less` does not implement a total order the resulting order and return value are
/// unspecified. All original elements will remain in `v` and any possible modifications via
/// interior mutability will be observable. Same is true if `is_less` panics or `v.len()`
/// exceeds `scratch.len()`.
fn stable_partition<T, F>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot_pos: usize,
    is_less: &mut F,
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    let arr_ptr = v.as_mut_ptr();

    if intrinsics::unlikely(scratch.len() < len || pivot_pos >= len) {
        debug_assert!(false); // That's a logic bug in the implementation.
        return 0;
    }

    let scratch_ptr = MaybeUninit::slice_as_mut_ptr(scratch);

    // SAFETY: We checked that `pivot_pos` is in-bounds and that `scratch` is valid for `len`
    // writes, fulfilling the safety contract of partition_fill_scratch. Assuming
    // partition_fill_scratch works as documented `scratch` should hold valid elements that observed
    // all possible changes to them, and can then be copied back into `v`.
    unsafe {
        // We can just use the value inside the slice and avoid a drop guard around a stack copy
        // of the value, because we only write into scratch during the scan loop. This
        // simplifies the code and shows no perf difference.
        let pivot_ptr = arr_ptr.add(pivot_pos);

        let lt_count = T::partition_fill_scratch(arr_ptr, len, scratch_ptr, pivot_ptr, is_less);

        // Copy all the elements that were not equal directly from swap to v.
        ptr::copy_nonoverlapping(scratch_ptr, arr_ptr, lt_count);

        // Copy the elements that were equal or more from the buf into v and reverse them.
        let rev_buf_ptr = scratch_ptr.add(len - 1);
        for i in 0..len - lt_count {
            ptr::copy_nonoverlapping(rev_buf_ptr.sub(i), arr_ptr.add(lt_count + i), 1);
        }

        lt_count
    }
}

trait StablePartitionTypeImpl: Sized {
    /// Takes a slice of `len` pointed to by `arr_ptr` and fills `scratch_ptr` with a partitioned
    /// copy of the values according to `is_less`.
    ///
    /// Example [05162738] -> [01238765]
    ///
    /// SAFETY: The caller MUST ensure that `arr_ptr` points to a valid slice of `len` elements and
    /// that `scratch_ptr` is valid for `len` writes.
    unsafe fn partition_fill_scratch<F>(
        arr_ptr: *mut Self,
        len: usize,
        scratch_ptr: *mut Self,
        pivot_ptr: *const Self,
        is_less: &mut F,
    ) -> usize
    where
        F: FnMut(&Self, &Self) -> bool;
}

impl<T> StablePartitionTypeImpl for T {
    /// See [`StablePartitionTypeImpl::partition_fill_scratch`].
    default unsafe fn partition_fill_scratch<F>(
        arr_ptr: *mut Self,
        len: usize,
        scratch_ptr: *mut Self,
        pivot_ptr: *const Self,
        is_less: &mut F,
    ) -> usize
    where
        F: FnMut(&Self, &Self) -> bool,
    {
        // We need to take special care of types with interior mutability. `is_less` can modify the
        // values it is provided. For example if `pivot_ptr` points to an element in the middle. A
        // copy of the backing element would be written into the scratch space, and later
        // modifications to the element behind `pivot_ptr` would be missed by subsequent calls to
        // `is_less(&*elem_ptr, &*pivot_ptr)`. This can quickly lead to UB, e.g.
        // `Mutex<Option<Box<String>>>` could miss an update where the `Option` is set to `None`
        // which would cause a double free.

        // SAFETY: The element access is arr_ptr + i, where i < len, which makes it proven
        // in-bounds, assuming the caller upholds the function safety contract. The two output
        // pointers `scratch_ptr` and `ge_out_ptr` each point to a unique location within the range
        // of `scratch_ptr`, and the combination of always doing decrementing `ge_out_ptr` and
        // conditionally incrementing `lt_count` ensures that every location of `scratch_ptr` will
        // be written. If `is_less` panics, only un-observed copies were written into the scratch
        // space.
        unsafe {
            let mut pivot_out_ptr = ptr::null_mut();

            // lt == less than, ge == greater or equal
            let mut lt_count = 0;
            let mut ge_out_ptr = scratch_ptr.add(len);

            for i in 0..len {
                let elem_ptr = arr_ptr.add(i);
                ge_out_ptr = ge_out_ptr.sub(1);

                let is_less_than_pivot = is_less(&*elem_ptr, &*pivot_ptr);

                let dst_ptr_base = if is_less_than_pivot {
                    scratch_ptr
                } else {
                    ge_out_ptr
                };
                let dst_ptr = dst_ptr_base.add(lt_count);

                ptr::copy_nonoverlapping(elem_ptr, dst_ptr, 1);

                if const { crate::has_direct_interior_mutability::<T>() }
                    && intrinsics::unlikely(elem_ptr as *const T == pivot_ptr)
                {
                    pivot_out_ptr = dst_ptr;
                }

                lt_count += is_less_than_pivot as usize;
            }

            if const { crate::has_direct_interior_mutability::<T>() } {
                ptr::copy_nonoverlapping(pivot_ptr, pivot_out_ptr, 1);
            }

            lt_count
        }
    }
}

/// Specialization for int like types.
impl<T> StablePartitionTypeImpl for T
where
    T: crate::Freeze + Copy,
    (): crate::IsTrue<{ mem::size_of::<T>() <= (mem::size_of::<u64>() * 2) }>,
{
    /// See [`StablePartitionTypeImpl::partition_fill_scratch`].
    unsafe fn partition_fill_scratch<F>(
        arr_ptr: *mut Self,
        len: usize,
        scratch_ptr: *mut Self,
        pivot_ptr: *const Self,
        is_less: &mut F,
    ) -> usize
    where
        F: FnMut(&Self, &Self) -> bool,
    {
        // Partitioning loop manually unrolled to ensure good performance. Example T == u64, on x86
        // LLVM unrolls this loop but not on Arm. A compile time fixed size loop as based on
        // `unroll_len` is reliably unrolled by all backends. And if `unroll_len` is `1` the inner
        // loop can trivially be removed.
        //
        // The scheme used to unroll is somewhat weird, and focused on avoiding multi-instantiation
        // of the inner loop part, which can have large effects on compile-time for non integer like
        // types.
        //
        // Benchmarks show that for any Type of at most 16 bytes, double storing is more efficient
        // than conditional store, especially on Firestorm (apple-m1). It is also less at risk of
        // having the compiler generating a branch instead of conditional store.

        // SAFETY: The element access is arr_ptr + i, where i < len, which makes it proven
        // in-bounds, assuming the caller upholds the function safety contract. The two output
        // pointers `scratch_ptr` and `ge_out_ptr` each point to a unique location within the range
        // of `scratch_ptr`, and the combination of always doing decrementing `ge_out_ptr` and
        // conditionally incrementing `lt_count` ensures that every location of `scratch_ptr` will
        // be written. If `is_less` panics, only un-observed copies were written into the scratch
        // space.
        unsafe {
            const UNROLL_LEN: usize = 4;

            // lt == less than, ge == greater or equal
            let mut lt_count = 0;
            let mut ge_out_ptr = scratch_ptr.add(len);

            macro_rules! loop_body {
                ($elem_ptr:expr) => {
                    let elem_ptr = $elem_ptr;
                    ge_out_ptr = ge_out_ptr.sub(1);

                    let is_less_than_pivot = is_less(&*elem_ptr, &*pivot_ptr);

                    ptr::copy_nonoverlapping(elem_ptr, scratch_ptr.add(lt_count), 1);
                    ptr::copy_nonoverlapping(elem_ptr, ge_out_ptr.add(lt_count), 1);

                    lt_count += is_less_than_pivot as usize;
                };
            }

            let mut offset = 0;
            for _ in 0..(len / UNROLL_LEN) {
                for unroll_i in 0..UNROLL_LEN {
                    loop_body!(arr_ptr.add(offset + unroll_i));
                }
                offset += UNROLL_LEN;
            }

            for i in 0..(len % UNROLL_LEN) {
                loop_body!(arr_ptr.add(offset + i));
            }

            lt_count
        }
    }
}
