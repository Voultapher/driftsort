use core::mem::MaybeUninit;

use crate::LengthAndSorted;

// Unstable shim.
#[must_use]
#[inline(always)]
pub const fn uninit_array<T, const N: usize>() -> [MaybeUninit<T>; N] {
    // SAFETY: An uninitialized `[MaybeUninit<_>; LEN]` is valid.
    unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() }
}

// Lazy logical runs as in Glidesort.
#[inline(always)]
fn logical_merge<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    left: LengthAndSorted,
    right: LengthAndSorted,
    orig_array_len: usize,
    is_less: &mut F,
) -> LengthAndSorted {
    // We *need* to physically merge if the combined runs do not fit in the
    // scratch space anymore (as this would mean we are no longer able to
    // to quicksort them).
    //
    // If both our inputs are sorted, it makes sense to merge them.
    //
    // Finally, if only one of our inputs is sorted we quicksort the other one
    // and merge iff the combined length is significant enough to be worth
    // it to switch to merges. We consider it significant if the combined
    // length is at least sqrt of the original array length. Otherwise we simply
    // forget the run is sorted and treat it as unsorted data.
    let n = v.len();
    let doesnt_fit_in_scratch = n > scratch.len();
    let num_sorted_plus_significance = (n.saturating_mul(n) >= orig_array_len) as usize
        + (left.sorted() as usize)
        + (right.sorted() as usize);

    if doesnt_fit_in_scratch || num_sorted_plus_significance >= 2 {
        if !left.sorted() {
            crate::stable_quicksort(&mut v[..left.len()], scratch, is_less);
        }
        if !right.sorted() {
            crate::stable_quicksort(&mut v[left.len()..], scratch, is_less);
        }
        crate::physical_merge(v, scratch, left.len(), is_less);
        LengthAndSorted::new(n, true)
    } else {
        LengthAndSorted::new(n, false)
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

#[inline(never)]
pub fn sort<T, F: FnMut(&T, &T) -> bool>(
    v: &mut [T],
    scratch: &mut [MaybeUninit<T>],
    eager_sort: bool,
    is_less: &mut F,
) {
    if v.len() < 2 {
        return; // Removing this length check *increases* code size.
    }

    let scale_factor = merge_tree_scale_factor(v.len());

    // (stack_len, runs, desired_depths) together form a stack maintaining run
    // information for the powersort heuristic. desired_depths[i] is the desired
    // depth of the merge node that merges runs[i] with the run that comes after
    // it.
    let mut stack_len = 0;
    let mut runs: [MaybeUninit<LengthAndSorted>; 64] = uninit_array();
    let mut desired_depths: [MaybeUninit<u8>; 64] = uninit_array();

    let mut scan_idx = 0;
    let mut prev_run = LengthAndSorted::new(0, true);
    loop {
        // Compute the next run and the desired depth of the merge node between
        // prev_run and next_run. On the last iteration we create a dummy run
        // with root-level desired depth to fully collapse the merge tree.
        let (next_run, desired_depth);
        if scan_idx < v.len() {
            next_run = crate::create_run(&mut v[scan_idx..], eager_sort, is_less);
            desired_depth = merge_tree_depth(
                scan_idx - prev_run.len(),
                scan_idx,
                scan_idx + next_run.len(),
                scale_factor,
            );
        } else {
            next_run = LengthAndSorted::new(0, true);
            desired_depth = 0;
        };

        // Process the merge nodes between earlier runs[i] that have a desire to
        // be deeper in the merge tree than the merge node for the splitpoint
        // between prev_run and next_run.
        unsafe {
            // SAFETY: first note that this is the only place we modify stack_len,
            // runs or desired depths. We maintain the following invariants:
            //  1. The first stack_len elements of runs/desired_depths are initialized.
            //  2. For all valid i, desired_depths[i] < desired_depths[i+1].
            //  3. The sum of all valid runs[i].len() plus prev_run.len() equals
            //     scan_idx.
            while stack_len > 0
                && desired_depths.get_unchecked(stack_len - 1).assume_init() >= desired_depth
            {
                // Desired depth greater than the upcoming desired depth, pop
                // from stack and merge into prev_run.
                let left = runs.get_unchecked(stack_len - 1).assume_init();
                let merged_len = left.len() + prev_run.len();
                let merge_start_idx = scan_idx - merged_len;
                let v_len = v.len();
                let sl = v.get_unchecked_mut(merge_start_idx..scan_idx);
                prev_run = logical_merge(sl, scratch, left, prev_run, v_len, is_less);
                stack_len -= 1;
            }

            // We now know that desired_depths[stack_len - 1] < desired_depth,
            // maintaining our invariant. This also guarantees we don't overflow
            // the stack as merge_tree_depth(..) <= 64 and thus we can only have
            // 63 distinct values on the stack before pushing.
            *runs.get_unchecked_mut(stack_len) = MaybeUninit::new(prev_run);
            *desired_depths.get_unchecked_mut(stack_len) = MaybeUninit::new(desired_depth);
            stack_len += 1;
        }

        // Break before overriding the last run with our dummy run.
        if scan_idx >= v.len() {
            break;
        }

        scan_idx += next_run.len();
        prev_run = next_run;
    }

    if !prev_run.sorted() {
        crate::stable_quicksort(v, scratch, is_less);
    }
}
