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
#[inline(never)]
pub fn stable_quicksort<T, F: FnMut(&T, &T) -> bool>(
    mut v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    mut limit: u32,
    mut left_ancestor_pivot: Option<&T>,
    is_less: &mut F,
) {
    loop {
        let len = v.len();

        if len <= T::SMALL_SORT_THRESHOLD {
            T::sort_small(v, scratch, is_less);
            return;
        }

        if limit == 0 {
            crate::drift::sort(v, scratch, true, is_less);
            return;
        }
        limit -= 1;

        let pivot_pos = choose_pivot(v, is_less);
        // SAFETY: choose_pivot promises to return a valid pivot index.
        unsafe {
            intrinsics::assume(pivot_pos < v.len());
        }

        // SAFETY: We only access the temporary copy for Freeze types, otherwise
        // self-modifications via `is_less` would not be observed and this would
        // be unsound. Our temporary copy does not escape this scope.
        let pivot_copy = unsafe { ManuallyDrop::new(ptr::read(&v[pivot_pos])) };
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
            perform_equal_partition = !is_less(la_pivot, &v[pivot_pos]);
        }

        let mut left_partition_len = 0;
        if !perform_equal_partition {
            left_partition_len = stable_partition(v, scratch, pivot_pos, false, is_less);
            perform_equal_partition = left_partition_len == 0;
        }

        if perform_equal_partition {
            let mid_eq = stable_partition(v, scratch, pivot_pos, true, &mut |a, b| !is_less(b, a));
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

struct PartitionState<T> {
    // The current element that is being looked at, scans left to right through slice.
    scan: *const T,
    // Counts the number of elements that compared less-than, also works around:
    // https://github.com/rust-lang/rust/issues/117128
    num_lt: usize,
    // Reverse scratch output pointer.
    scratch_rev: *mut T,
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
    pivot_goes_left: bool,
    is_less: &mut F,
) -> usize {
    let len = v.len();

    if intrinsics::unlikely(scratch.len() < len || pivot_pos >= len) {
        core::intrinsics::abort()
    }

    let v_base = v.as_ptr();
    let scratch_base = MaybeUninit::slice_as_mut_ptr(scratch);

    // TODO explain logic on high level.

    // Regarding auto unrolling and manual unrolling. Auto unrolling as tested with rustc 1.75 is
    // somewhat run-time and binary-size inefficient, because it performs additional math to
    // calculate the loop end, which we can avoid by precomputing the loop end. Also auto unrolling
    // only happens on x86 but not on Arm where doing so for the Firestorm micro-architecture yields
    // a 15+% performance improvement. Manual unrolling via repeated code has a large negative
    // impact on debug compile-times, and unrolling via `for _ in 0..UNROLL_LEN` has a 10-20% perf
    // penalty when compiling with `opt-level=s` which is deemed unacceptable for such a crucial
    // component of the sort implementation.

    // SAFETY: we checked that pivot_pos is in-bounds above, and that scratch has length at least
    // len. As we do binary classification into lt or ge, the invariant num_lt + num_ge = i always
    // holds at the start of each iteration. For micro-optimization reasons we write i - num_lt
    // instead of num_gt. Since num_lt increases by at most 1 each iteration and since i < len, this
    // proves our destination indices num_lt and len - 1 - num_ge stay in-bounds, and are never
    // equal except the final iteration when num_lt = len - 1 - (len - 1 - num_lt) = len - 1 -
    // num_ge. We write one different element to scratch each iteration thus scratch[0..len] will be
    // initialized with a permutation of v. TODO extend with nested loop logic.
    unsafe {
        // SAFETY: exactly the same invariants and logic as the non-specialized impl. And we do
        // naive loop unrolling where the exact same loop body is just repeated.
        let pivot = v_base.add(pivot_pos);

        let mut loop_body = |state: &mut PartitionState<T>| {
            // println!("state.scan: {}", state.scan.sub_ptr(v_base));
            state.scratch_rev = state.scratch_rev.sub(1);

            // println!("loop_body state.scan: {:?}", *(state.scan as *const DebugT));
            let is_less_than_pivot = is_less(&*state.scan, &*pivot);
            let dst_base = if is_less_than_pivot {
                scratch_base // i + num_lt
            } else {
                state.scratch_rev // len - (i + 1) + num_lt = len - 1 - num_ge
            };
            ptr::copy_nonoverlapping(state.scan, dst_base.add(state.num_lt), 1);

            state.num_lt += is_less_than_pivot as usize;
            state.scan = state.scan.add(1);
        };

        let mut state = PartitionState {
            scan: v_base,
            num_lt: 0,
            scratch_rev: scratch_base.add(len),
        };

        let mut pivot_in_scratch = ptr::null_mut();
        let mut loop_end_pos = pivot_pos;

        // Ideally this outer loop won't be unrolled, to save binary size.
        loop {
            if const { mem::size_of::<T>() <= 16 } {
                const UNROLL_LEN: usize = 4;
                let unroll_end = v_base.add(loop_end_pos.saturating_sub(UNROLL_LEN - 1));
                while state.scan < unroll_end {
                    loop_body(&mut state);
                    loop_body(&mut state);
                    loop_body(&mut state);
                    loop_body(&mut state);
                }
            }

            let loop_end = v_base.add(loop_end_pos);
            while state.scan < loop_end {
                loop_body(&mut state);
            }

            if loop_end_pos == len {
                break;
            }

            // Handle pivot, doing it this way neatly handles type with interior mutability and
            // avoids self comparison as well as a branch in the inner partition loop.
            state.scratch_rev = state.scratch_rev.sub(1);
            let pivot_dst_base = if pivot_goes_left {
                scratch_base
            } else {
                state.scratch_rev
            };
            pivot_in_scratch = pivot_dst_base.add(state.num_lt);
            state.num_lt += pivot_goes_left as usize;
            state.scan = state.scan.add(1);

            loop_end_pos = len;
        }

        // `pivot` must only be copied after all possible modifications to it have been observed.
        ptr::copy_nonoverlapping(pivot, pivot_in_scratch, 1);

        // SAFETY: partition_fill_scratch guarantees that scratch is initialized with a permuted
        // copy of `v`, and that num_lt <= v.len(). Copying scratch[0..num_lt] and
        // scratch[num_lt..v.len()] back is thus sound, as the values in scratch will never be read
        // again, meaning our copies semantically act as moves, permuting `v`. Copy all the elements
        // < p directly from swap to v.
        let v_base = v.as_mut_ptr();
        ptr::copy_nonoverlapping(scratch_base, v_base, state.num_lt);

        // Copy the elements >= p in reverse order.
        for i in 0..len - state.num_lt {
            ptr::copy_nonoverlapping(
                scratch_base.add(len - 1 - i),
                v_base.add(state.num_lt + i),
                1,
            );
        }

        state.num_lt
    }
}
