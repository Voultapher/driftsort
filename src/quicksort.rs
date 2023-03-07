use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr;

// Switch to a dedicated small array sorting algorithm below this threshold.
const SMALL_SORT_THRESHOLD: usize = 20;

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
    is_less: &mut F,
) where
    F: FnMut(&T, &T) -> bool,
{
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
        let mid = stable_partition(v, scratch, pivot, is_less, false);

        // Empty left partition almost surely means second time we use this pivot.
        // Swap to partition that filters equal elements on the left.
        if mid == 0 {
            let mid = stable_partition(v, scratch, pivot, &mut |a, b| !is_less(b, a), true);
            v = &mut v[mid..];
            continue;
        }

        let (left, right) = v.split_at_mut(mid);
        stable_quicksort(left, scratch, limit, is_less);
        v = right;
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
    pivot_goes_left: bool,
) -> usize
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    let arr = v.as_mut_ptr();

    unsafe {
        assert!(scratch.len() >= len);
        let buf = MaybeUninit::slice_as_mut_ptr(scratch);

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

        let pivot = PivotGuard {
            value: ManuallyDrop::new(ptr::read(&v[pivot_pos])),
            hole: arr.add(pivot_pos),
        };

        let mut pivot_partioned_ptr = ptr::null_mut();
        let mut l_count = 0;
        let mut reverse_out = buf.add(len);
        for i in 0..len {
            reverse_out = reverse_out.sub(1);
            
            // This should only happen once and should be predicted very well.
            if i == pivot_pos {
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

        // Copy pivot_val into it's correct position.
        ptr::copy_nonoverlapping(&*pivot.value, pivot_partioned_ptr, 1);
        core::mem::forget(pivot);

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
