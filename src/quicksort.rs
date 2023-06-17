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

/// Partitions `v` into elements smaller than `pivot`, followed by elements
/// greater than or equal to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
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

    // Inside the main partitioning loop we MUST NOT compare our stack copy of the pivot value with
    // the original value in the slice `v`. If we just write the value as pointed to by `src_ptr`
    // into `sctratch_ptr` as it was in the input slice `v` we would risk that the call to the
    // user-provided `is_less` modifies the value pointed to by `src_ptr`. This could be UB for
    // types such as `Mutex<Option<Box<String>>>` where during the comparison it replaces the box
    // with None, leading to double free. As the value written back into `v` from `sctratch_ptr` did
    // not observe that modification.

    // SAFETY: TODO
    unsafe {
        assert!(scratch.len() >= len);
        let scratch_ptr = MaybeUninit::slice_as_mut_ptr(scratch);

        let pivot_guard = PivotGuard {
            value: ManuallyDrop::new(ptr::read(&v[pivot_pos])),
            hole: arr_ptr.add(pivot_pos),
        };
        let pivot: &T = &pivot_guard.value;

        let mut pivot_out_ptr = ptr::null_mut();

        // lt == less than, ge == greater or equal
        let mut lt_count = 0;
        let mut ge_out_ptr = scratch_ptr.add(len);
        for i in 0..len {
            ge_out_ptr = ge_out_ptr.sub(1);

            if const { crate::has_direct_interior_mutability::<T>() } {
                //  This is required to
                // handle types with interior mutability. See comment above for more info.
                if intrinsics::unlikely(i == pivot_pos) {
                    // We move the pivot in its correct place later.
                    if is_less(pivot, pivot) {
                        pivot_out_ptr = scratch_ptr.add(lt_count);
                        lt_count += 1;
                    } else {
                        pivot_out_ptr = ge_out_ptr.add(lt_count);
                    }
                    continue;
                }
            }

            let elem_ptr = arr_ptr.add(i);
            let is_less_than_pivot = is_less(&*elem_ptr, pivot);

            // Benchmarks show that especially on apple-m1 for anything at most the size of a u64
            // double storing is more efficient than conditional store. It is also less at risk of
            // having the compiler generating a branch instead of conditional store.
            if const { mem::size_of::<T>() <= mem::size_of::<usize>() } {
                ptr::copy_nonoverlapping(elem_ptr, scratch_ptr.add(lt_count), 1);
                ptr::copy_nonoverlapping(elem_ptr, ge_out_ptr.add(lt_count), 1);
            } else {
                let dst = if is_less_than_pivot {
                    scratch_ptr
                } else {
                    ge_out_ptr
                };
                ptr::copy_nonoverlapping(elem_ptr, dst.add(lt_count), 1);
            }

            lt_count += is_less_than_pivot as usize;
        }

        // Now that any possible observation of pivot has happened we copy it.
        if const { has_direct_interior_mutability::<T>() } {
            ptr::copy_nonoverlapping(pivot, pivot_out_ptr, 1);
        }
        core::mem::forget(pivot_guard);

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
