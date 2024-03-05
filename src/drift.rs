use core::cmp;
use core::intrinsics;
use core::mem::MaybeUninit;

use crate::merge::merge;
use crate::stable_quicksort;
use crate::DriftsortRun;

// Lazy logical runs as in Glidesort.
#[inline(always)]
fn logical_merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    left: DriftsortRun,
    right: DriftsortRun,
    is_less: &mut F,
) -> DriftsortRun {
    // If one or both of the runs are sorted do a physical merge, using
    // quicksort to sort the unsorted run if present. We also *need* to
    // physically merge if the combined runs would not fit in the scratch space
    // anymore (as this would mean we are no longer able to to quicksort them).
    let len = v.len();
    let can_fit_in_scratch = len <= scratch.len();
    if !can_fit_in_scratch || left.sorted() || right.sorted() {
        if !left.sorted() {
            stable_quicksort(&mut v[..left.len()], scratch, is_less);
        }
        if !right.sorted() {
            stable_quicksort(&mut v[left.len()..], scratch, is_less);
        }
        merge(v, scratch, left.len(), is_less);

        DriftsortRun::new_sorted(len)
    } else {
        DriftsortRun::new_unsorted(len)
    }
}

// Nearly-Optimal Mergesorts: Fast, Practical Sorting Methods That Optimally
// Adapt to Existing Runs by J. Ian Munro and Sebastian Wild.
//
// This method forms a binary merge tree, where each internal node corresponds
// to a splitting point between the adjacent runs that have to be merged. If we
// visualize our array as the number line from 0 to 1, we want to find the
// dyadic fraction with smallest denominator that lies between the midpoints of
// our to-be-merged slices. The exponent in the dyadic fraction indicates the
// desired depth in the binary merge tree this internal node wishes to have.
// This does not always correspond to the actual depth due to the inherent
// imbalance in runs, but we follow it as closely as possible.
//
// As an optimization we rescale the number line from [0, 1) to [0, 2^62). Then
// finding the simplest dyadic fraction between midpoints corresponds to finding
// the most significant bit difference of the midpoints. We save scale_factor =
// ceil(2^62 / n) to perform this rescaling using a multiplication, avoiding
// having to repeatedly do integer divides. This rescaling isn't exact when n is
// not a power of two since we use integers and not reals, but the result is
// very close, and in fact when n < 2^30 the resulting tree is equivalent as the
// approximation errors stay entirely in the lower order bits.
//
// Thus for the splitting point between two adjacent slices [a, b) and [b, c)
// the desired depth of the corresponding merge node is CLZ((a+b)*f ^ (b+c)*f),
// where CLZ counts the number of leading zeros in an integer and f is our scale
// factor. Note that we omitted the division by two in the midpoint
// calculations, as this simply shifts the bits by one position (and thus always
// adds one to the result), and we only care about the relative depths.
//
// Finally, if we try to upper bound x = (a+b)*f giving x = (n-1 + n) * ceil(2^62 / n) then
//    x < (2^62 / n + 1) * 2n
//    x < 2^63 + 2n
// So as long as n < 2^62 we find that x < 2^64, meaning our operations do not
// overflow.
#[inline(always)]
fn merge_tree_scale_factor(n: usize) -> u64 {
    if usize::BITS > u64::BITS {
        panic!("Platform not supported");
    }

    ((1 << 62) + n as u64 - 1) / n as u64
}

// Note: merge_tree_depth output is < 64 when left < right as f*x and f*y must
// differ in some bit, and is <= 64 always.
#[inline(always)]
fn merge_tree_depth(left: usize, mid: usize, right: usize, scale_factor: u64) -> u8 {
    let x = left as u64 + mid as u64;
    let y = mid as u64 + right as u64;
    ((scale_factor * x) ^ (scale_factor * y)).leading_zeros() as u8
}

fn sqrt_approx(n: usize) -> usize {
    // Note that sqrt(n) = n^(1/2), and that 2^log2(n) = n. We combine these
    // two facts to approximate sqrt(n) as 2^(log2(n) / 2). Because our integer
    // log floors we want to add 0.5 to compensate for this on average, so our
    // initial approximation is 2^((1 + floor(log2(n))) / 2).
    //
    // We then apply an iteration of Newton's method to improve our
    // approximation, which for sqrt(n) is a1 = (a0 + n / a0) / 2.
    //
    // Finally we note that the exponentiation / division can be done directly
    // with shifts. We OR with 1 to avoid zero-checks in the integer log.
    let ilog = (n | 1).ilog2();
    let shift = (1 + ilog) / 2;
    ((1 << shift) + (n >> shift)) / 2
}

pub fn sort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    eager_sort: bool,
    is_less: &mut F,
) {
    let len = v.len();
    if len < 2 {
        return; // Removing this length check *increases* code size.
    }
    let scale_factor = merge_tree_scale_factor(len);

    // It's important to have a relatively high entry barrier for pre-sorted runs, as the presence
    // of a single such run will force on average several merge operations and shrink the maximum
    // quicksort size a lot. For that reason we use sqrt(len) as our pre-sorted run threshold, with
    // SMALL_SORT_THRESHOLD as the lower limit. When eagerly sorting we also use
    // SMALL_SORT_THRESHOLD as our threshold, as we will call small_sort on any runs smaller than
    // this.
    let min_good_run_len = if eager_sort {
        crate::MAX_LEN_ALWAYS_INSERTION_SORT / 2
    } else if len <= (crate::MAX_LEN_ALWAYS_INSERTION_SORT * crate::MAX_LEN_ALWAYS_INSERTION_SORT) {
        crate::MAX_LEN_ALWAYS_INSERTION_SORT
    } else {
        sqrt_approx(len)
    };

    // (stack_len, runs, desired_depths) together form a stack maintaining run
    // information for the powersort heuristic. desired_depths[i] is the desired
    // depth of the merge node that merges runs[i] with the run that comes after
    // it.
    let mut stack_len = 0;
    let mut run_storage = MaybeUninit::<[DriftsortRun; 66]>::uninit();
    let runs: *mut DriftsortRun = run_storage.as_mut_ptr().cast();
    let mut desired_depth_storage = MaybeUninit::<[u8; 66]>::uninit();
    let desired_depths: *mut u8 = desired_depth_storage.as_mut_ptr().cast();

    let mut scan_idx = 0;
    let mut prev_run = DriftsortRun::new_sorted(0); // Initial dummy run.
    loop {
        // Compute the next run and the desired depth of the merge node between
        // prev_run and next_run. On the last iteration we create a dummy run
        // with root-level desired depth to fully collapse the merge tree.
        let (next_run, desired_depth);
        if scan_idx < len {
            next_run = create_run(&mut v[scan_idx..], min_good_run_len, eager_sort, is_less);
            desired_depth = merge_tree_depth(
                scan_idx - prev_run.len(),
                scan_idx,
                scan_idx + next_run.len(),
                scale_factor,
            );
        } else {
            next_run = DriftsortRun::new_sorted(0);
            desired_depth = 0;
        };

        // Process the merge nodes between earlier runs[i] that have a desire to
        // be deeper in the merge tree than the merge node for the splitpoint
        // between prev_run and next_run.
        unsafe {
            // SAFETY: first note that this is the only place we modify stack_len,
            // runs or desired depths. We maintain the following invariants:
            //  1. The first stack_len elements of runs/desired_depths are initialized.
            //  2. For all valid i > 0, desired_depths[i] < desired_depths[i+1].
            //  3. The sum of all valid runs[i].len() plus prev_run.len() equals
            //     scan_idx.
            while stack_len > 1 && *desired_depths.add(stack_len - 1) >= desired_depth {
                // Desired depth greater than the upcoming desired depth, pop
                // left neighbor run from stack and merge into prev_run.
                let left = *runs.add(stack_len - 1);
                let merged_len = left.len() + prev_run.len();
                let merge_start_idx = scan_idx - merged_len;
                let merge_slice = v.get_unchecked_mut(merge_start_idx..scan_idx);
                prev_run = logical_merge(merge_slice, scratch, left, prev_run, is_less);
                stack_len -= 1;
            }

            // We now know that desired_depths[stack_len - 1] < desired_depth,
            // maintaining our invariant. This also guarantees we don't overflow
            // the stack as merge_tree_depth(..) <= 64 and thus we can only have
            // 64 distinct values on the stack before pushing, plus our initial
            // dummy run, while our capacity is 66.
            *runs.add(stack_len) = prev_run;
            *desired_depths.add(stack_len) = desired_depth;
            stack_len += 1;
        }

        // Break before overriding the last run with our dummy run.
        if scan_idx >= len {
            break;
        }

        scan_idx += next_run.len();
        prev_run = next_run;
    }

    if !prev_run.sorted() {
        stable_quicksort(v, scratch, is_less);
    }
}

/// Creates a new logical run.
///
/// A logical run can either be sorted or unsorted. If there is a pre-existing
/// run of length min_good_run_len (or longer) starting at v[0] we find and
/// return it, otherwise we return a run of length min_good_run_len that is
/// eagerly sorted when eager_sort is true, and left unsorted otherwise.
fn create_run<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    min_good_run_len: usize,
    eager_sort: bool,
    is_less: &mut F,
) -> DriftsortRun {
    let len = v.len();
    let (run_len, was_reversed) = find_existing_run(v, is_less);

    // SAFETY: find_existing_run promises to return a valid run_len.
    unsafe {
        intrinsics::assume(run_len <= len);
    }

    if run_len >= min_good_run_len {
        if was_reversed {
            v[..run_len].reverse();
        }
        DriftsortRun::new_sorted(run_len)
    } else {
        if eager_sort {
            // While eager sorting we want to create eager blocks, and we want
            // to limit ourselves to MAX_LEN_ALWAYS_INSERTION_SORT so we can
            // re-use the insertion sort that is inlined in our main entrypoint.
            // But we want to prevent leaving a small imbalanced leftover merge.
            let new_run_len = if len <= crate::MAX_LEN_ALWAYS_INSERTION_SORT {
                len
            } else if len <= 2 * crate::MAX_LEN_ALWAYS_INSERTION_SORT {
                len / 2
            } else {
                crate::MAX_LEN_ALWAYS_INSERTION_SORT
            };

            // SAFETY: new_run_len <= len in all cases.
            let new_run_slice = unsafe { v.get_unchecked_mut(..new_run_len) };

            crate::driftsort::<T, F, Vec<T>>(new_run_slice, is_less);
            DriftsortRun::new_sorted(new_run_len)
        } else {
            let skip = cmp::min(min_good_run_len, len);
            DriftsortRun::new_unsorted(skip)
        }
    }
}

/// Finds a run of sorted elements starting at the beginning of the slice.
///
/// Returns the length of the run, and a bool that is false when the run
/// is ascending, and true if the run strictly descending.
fn find_existing_run<T, F: FnMut(&T, &T) -> bool>(v: &[T], is_less: &mut F) -> (usize, bool) {
    let len = v.len();
    if len < 2 {
        return (len, false);
    }

    unsafe {
        // SAFETY: We checked that len >= 2, so 0 and 1 are valid indices.
        // This also means that run_len < len implies run_len and
        // run_len - 1 are valid indices as well.
        let mut run_len = 2;
        let strictly_descending = is_less(v.get_unchecked(1), v.get_unchecked(0));
        if strictly_descending {
            while run_len < len && is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1)) {
                run_len += 1;
            }
        } else {
            while run_len < len && !is_less(v.get_unchecked(run_len), v.get_unchecked(run_len - 1))
            {
                run_len += 1;
            }
        }
        (run_len, strictly_descending)
    }
}
