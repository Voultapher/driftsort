use core::intrinsics;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

use crate::has_direct_interior_mutability;
use crate::pivot::choose_pivot;
use crate::smallsort::SmallSortTypeImpl;

/// Sorts `v` recursively using quicksort.
///
/// `limit` when initialized with `c*log(v.len())` for some c ensures we do not
/// overflow the stack or go quadratic.
pub fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(
    mut v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mut limit: u32,
    mut left_ancestor_pivot: Option<&T>,
    is_less: &mut F,
) {
    loop {
        let len = v.len();

        if len <= T::MAX_LEN_SMALL_SORT {
            T::sort_small(v, scratch, is_less);
            return;
        }

        if limit == 0 {
            crate::drift::sort(v, scratch, true, is_less);
            return;
        }
        limit -= 1;

        // SAFETY: We only access the temporary copy for Freeze types, otherwise
        // self-modifications via `is_less` would not be observed and this would
        // be unsound. Our temporary copy does not escape this scope.
        let pivot_idx = choose_pivot(v, is_less);
        // SAFETY: choose_pivot promises to return a valid pivot index.
        unsafe {
            intrinsics::assume(pivot_idx < v.len());
        }
        let pivot_copy = unsafe { ManuallyDrop::new(ptr::read(&v[pivot_idx])) };
        let pivot_ref = (!has_direct_interior_mutability::<T>()).then_some(&*pivot_copy);

        // We choose a pivot, and check if this pivot is equal to our left
        // ancestor. If true, we do a partition putting equal elements on the
        // left and do not recurse on it. This gives O(n log k) sorting for k
        // distinct values, a strategy borrowed from pdqsort. For types with
        // interior mutability we can't soundly create a temporary copy of the
        // ancestor pivot, and use left_partition_len == 0 as our method for
        // detecting when we re-use a pivot, which means we do at most three
        // partition operations with pivot p instead of the optimal two.
        let mut perform_equal_partition = false;
        if let Some(la_pivot) = left_ancestor_pivot {
            perform_equal_partition = !is_less(la_pivot, &v[pivot_idx]);
        }

        let mut left_partition_len = 0;
        if !perform_equal_partition {
            left_partition_len = stable_partition(v, scratch, pivot_idx, is_less);
            perform_equal_partition = left_partition_len == 0;
        }

        if perform_equal_partition {
            let mid_eq = stable_partition(v, scratch, pivot_idx, &mut |a, b| !is_less(b, a));
            v = &mut v[mid_eq..];
            left_ancestor_pivot = None;
            continue;
        }

        // Process left side with the next loop iter, right side with recursion.
        let (left, right) = v.split_at_mut(left_partition_len);
        stable_quicksort(right, scratch, limit, pivot_ref, is_less);
        v = left;
    }
}

/// Partitions `v` using pivot `p = v[pivot_pos]` and returns the number of
/// elements less than `p`. The relative order of elements that compare < p and
/// those that compare >= p is preserved - it is a stable partition.
///
/// If `is_less` is not a strict total order or panics, `scratch.len() < v.len()`,
/// or `pivot_pos >= v.len()`, the result and `v`'s state is sound but unspecified.
fn stable_partition<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    pivot_pos: usize,
    is_less: &mut F,
) -> usize {
    let num_lt = T::partition_fill_scratch(v, scratch, pivot_pos, is_less);

    // SAFETY: partition_fill_scratch guarantees that scratch is initialized
    // with a permuted copy of `v`, and that num_lt <= v.len(). Copying
    // scratch[0..num_lt] and scratch[num_lt..v.len()] back is thus
    // sound, as the values in scratch will never be read again, meaning our
    // copies semantically act as moves, permuting `v`.
    unsafe {
        let len = v.len();
        let v_base = v.as_mut_ptr();
        let scratch_base = MaybeUninit::slice_as_mut_ptr(scratch);

        // Copy all the elements < p directly from swap to v.
        ptr::copy_nonoverlapping(scratch_base, v_base, num_lt);

        // Copy the elements >= p in reverse order.
        for i in 0..len - num_lt {
            ptr::copy_nonoverlapping(scratch_base.add(len - 1 - i), v_base.add(num_lt + i), 1);
        }

        num_lt
    }
}

trait StablePartitionTypeImpl: Sized {
    /// Performs the same operation as [`stable_partition`], except it stores the
    /// permuted elements as copies in `scratch`, with the >= partition in
    /// reverse order.
    fn partition_fill_scratch<F: FnMut(&Self, &Self) -> bool>(
        v: &[Self],
        scratch: &mut [MaybeUninit<Self>],
        pivot_pos: usize,
        is_less: &mut F,
    ) -> usize;
}

impl<T> StablePartitionTypeImpl for T {
    /// See [`StablePartitionTypeImpl::partition_fill_scratch`].
    default fn partition_fill_scratch<F: FnMut(&T, &T) -> bool>(
        v: &[T],
        scratch: &mut [MaybeUninit<T>],
        pivot_pos: usize,
        is_less: &mut F,
    ) -> usize {
        let len = v.len();
        let v_base = v.as_ptr();
        let scratch_base = MaybeUninit::slice_as_mut_ptr(scratch);

        if intrinsics::unlikely(scratch.len() < len || pivot_pos >= len) {
            core::intrinsics::abort()
        }

        unsafe {
            // Abbreviations: lt == less than, ge == greater or equal.
            //
            // SAFETY: we checked that pivot_pos is in-bounds above, and that
            // scratch has length at least len. As we do binary classification
            // into lt or ge, the invariant num_lt + num_ge = i always holds at
            // the start of each iteration. For micro-optimization reasons we
            // write i - num_lt instead of num_gt. Since num_lt increases by at
            // most 1 each iteration and since i < len, this proves our
            // destination indices num_lt and len - 1 - num_ge stay
            // in-bounds, and are never equal except the final iteration when
            // num_lt = len - 1 - (len - 1 - num_lt) = len - 1 - num_ge.
            // We write one different element to scratch each iteration thus
            // scratch[0..len] will be initialized with a permutation of v.
            //
            // Should a panic occur, the copies in the scratch space are simply
            // forgotten - even with interior mutability all data is still in v.
            let pivot = v_base.add(pivot_pos);
            let mut pivot_in_scratch = ptr::null_mut();
            let mut num_lt = 0;
            let mut scratch_rev = scratch_base.add(len);
            for i in 0..len {
                let scan = v_base.add(i);
                scratch_rev = scratch_rev.sub(1);

                let is_less_than_pivot = is_less(&*scan, &*pivot);
                let dst = if is_less_than_pivot {
                    scratch_base.add(num_lt) // i + num_lt
                } else {
                    scratch_rev.add(num_lt) // len - (i + 1) + num_lt = len - 1 - num_ge
                };

                // Save pivot location in scratch for later.
                if const { crate::has_direct_interior_mutability::<T>() }
                    && intrinsics::unlikely(scan == pivot)
                {
                    pivot_in_scratch = dst;
                }

                ptr::copy_nonoverlapping(scan, dst, 1);
                num_lt += is_less_than_pivot as usize;
            }

            // SAFETY: if T has interior mutability our copy in scratch can be
            // outdated, update it.
            if const { crate::has_direct_interior_mutability::<T>() } {
                ptr::copy_nonoverlapping(pivot, pivot_in_scratch, 1);
            }

            num_lt
        }
    }
}

/// Specialization for small types, through traits to not invoke compile time
/// penalties for loop unrolling when not used.
///
/// Benchmarks show that for small types simply storing to *both* potential
/// destinations is more efficient than a conditional store. It is also less at
/// risk of having the compiler generating a branch instead of conditional
/// store. And explicit loop unrolling is also often very beneficial.
impl<T> StablePartitionTypeImpl for T
where
    T: Copy,
    (): crate::IsTrue<{ mem::size_of::<T>() <= (mem::size_of::<u64>() * 2) }>,
{
    /// See [`StablePartitionTypeImpl::partition_fill_scratch`].
    fn partition_fill_scratch<F: FnMut(&T, &T) -> bool>(
        v: &[T],
        scratch: &mut [MaybeUninit<T>],
        pivot_pos: usize,
        is_less: &mut F,
    ) -> usize {
        let len = v.len();
        let v_base = v.as_ptr();
        let scratch_base = MaybeUninit::slice_as_mut_ptr(scratch);

        if intrinsics::unlikely(scratch.len() < len || pivot_pos >= len) {
            core::intrinsics::abort()
        }

        unsafe {
            // SAFETY: exactly the same invariants and logic as the
            // non-specialized impl. The conditional store being replaced by a
            // double copy changes nothing, on all but the final iteration the
            // bad write will simply be overwritten by a later iteration, and on
            // the final iteration we write to the same index twice. And we do
            // naive loop unrolling where the exact same loop body is just
            // repeated.
            let pivot = v_base.add(pivot_pos);
            let mut pivot_in_scratch = ptr::null_mut();
            let mut num_lt = 0;
            let mut scratch_rev = scratch_base.add(len);
            macro_rules! loop_body {
                ($i:expr) => {
                    let scan = v_base.add($i);
                    scratch_rev = scratch_rev.sub(1);

                    let is_less_than_pivot = is_less(&*scan, &*pivot);
                    ptr::copy_nonoverlapping(scan, scratch_base.add(num_lt), 1);
                    ptr::copy_nonoverlapping(scan, scratch_rev.add(num_lt), 1);

                    // Save pivot location in scratch for later.
                    if const { crate::has_direct_interior_mutability::<T>() }
                        && intrinsics::unlikely(scan as *const T == pivot)
                    {
                        pivot_in_scratch = if is_less_than_pivot {
                            scratch_base.add(num_lt) // i + num_lt
                        } else {
                            scratch_rev.add(num_lt) // len - (i + 1) + num_lt = len - 1 - num_ge
                        };
                    }

                    num_lt += is_less_than_pivot as usize;
                };
            }

            /// To ensure good performance across platforms we explicitly unroll using a
            /// fixed-size inner loop. We do not simply call the loop body multiple times as
            /// this increases compile times significantly more, and the compiler unrolls
            /// a fixed loop just as well, if it is sensible.
            const UNROLL_LEN: usize = 4;
            let mut offset = 0;
            for _ in 0..(len / UNROLL_LEN) {
                for unroll_i in 0..UNROLL_LEN {
                    loop_body!(offset + unroll_i);
                }
                offset += UNROLL_LEN;
            }

            for i in 0..(len % UNROLL_LEN) {
                loop_body!(offset + i);
            }

            // SAFETY: if T has interior mutability our copy in scratch can be
            // outdated, update it.
            if const { crate::has_direct_interior_mutability::<T>() } {
                ptr::copy_nonoverlapping(pivot, pivot_in_scratch, 1);
            }

            num_lt
        }
    }
}
