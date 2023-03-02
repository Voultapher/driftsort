use core::mem::MaybeUninit;

// Lazy logical runs as in Glidesort.
#[inline(always)]
fn should_physically_merge(
    left: LengthAndSorted,
    right: LengthAndSorted,
    el_len: usize,
    scratch_len: usize,
) -> bool {
    // We *need* to physically merge if the combined runs do not fit in the
    // scratch space anymore (as this would mean we are no longer able to
    // to quicksort them).
    //
    // If both our inputs are sorted, it makes sense to merge them.
    //
    // Finally, if only one of our inputs is sorted we quicksort the other one
    // and merge iff the combined length is significant enough to be worth
    // it to switch to merges. We consider it significant if the combined
    // length is at least sqrt(n). Otherwise we simply forget the run is sorted
    // and treat it as unsorted data.
    let left_sorted = (left.0 & 1) == 1;
    let right_sorted = (right.0 & 1) == 1;
    let total_length = left.len() + right.len();
    (total_length > scratch_len)
        | (left_sorted & right_sorted)
        | (left_sorted | right_sorted) & (total_length.saturating_mul(total_length) >= el_len)
}

#[inline(always)]
fn logical_merge(
    start_idx: usize,
    left: LengthAndSorted,
    right: LengthAndSorted,
    sort_ops: &mut dyn crate::SortOps2,
    el_len: usize,
    scratch_len: usize,
) -> LengthAndSorted {
    if should_physically_merge(left, right, el_len, scratch_len) {
        if !left.sorted() {
            sort_ops.physical_sort(start_idx, start_idx + left.len());
        }
        if !right.sorted() {
            sort_ops.physical_sort(start_idx + left.len(), start_idx + left.len() + right.len());
        }
        sort_ops.physical_merge(
            start_idx,
            start_idx + left.len(),
            start_idx + left.len() + right.len(),
        );
        LengthAndSorted::new(left.len() + right.len(), true)
    } else {
        LengthAndSorted::new(left.len() + right.len(), false)
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
// differ in some bit.
#[inline(always)]
fn merge_tree_depth(left: usize, mid: usize, right: usize, scale_factor: u64) -> u8 {
    let x = left as u64 + mid as u64;
    let y = mid as u64 + right as u64;
    ((scale_factor * x) ^ (scale_factor * y)).leading_zeros() as u8
}

#[derive(Copy, Clone)]
pub struct LengthAndSorted(usize);

impl LengthAndSorted {
    #[inline(always)]
    pub fn new(length: usize, sorted: bool) -> Self {
        Self((length << 1) | sorted as usize)
    }

    #[inline(always)]
    pub fn sorted(&self) -> bool {
        self.0 & 1 == 1
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0 >> 1
    }
}

#[inline(always)]
pub fn sort(
    sort_ops: &mut dyn crate::SortOps2,
    eager_sort: bool,
    el_len: usize,
    scratch_len: usize,
) {
    let scale_factor = merge_tree_scale_factor(el_len);

    // desired_depths[i] is the desired depth in the merge tree of the merge
    // node connecting runs[i] with the run that comes after it.
    let mut runs: [MaybeUninit<LengthAndSorted>; 64] =
        unsafe { MaybeUninit::uninit().assume_init() };
    let mut desired_depths: [MaybeUninit<u8>; 64] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut stack_len = 0;

    let mut last_index = 0;
    let mut prev_run = LengthAndSorted(1);
    loop {
        // Compute the next run and the desired depth of the merge node between
        // prev_run and next_run. On the last iteration we create a dummy run
        // with root desired depth to fully collapse the merge tree.
        let (next_run, desired_depth);
        if last_index < el_len {
            next_run = sort_ops.create_run(last_index, eager_sort);
            desired_depth = merge_tree_depth(
                last_index - prev_run.len(),
                last_index,
                last_index + next_run.len(),
                scale_factor,
            );
        } else {
            next_run = LengthAndSorted(0);
            desired_depth = 0;
        };

        // Process the merge nodes between earlier runs that have a desire to be
        // deeper in the merge tree than the merge between prev_run and
        // next_run.
        unsafe {
            while stack_len > 0
                && desired_depths.get_unchecked(stack_len - 1).assume_init() >= desired_depth
            {
                let left = runs.get_unchecked(stack_len - 1).assume_init();
                let start_idx = last_index - left.len() - prev_run.len();
                prev_run = logical_merge(start_idx, left, prev_run, sort_ops, el_len, scratch_len);
                stack_len -= 1;
            }
        }

        if last_index >= el_len {
            break;
        }

        unsafe {
            *runs.get_unchecked_mut(stack_len) = MaybeUninit::new(prev_run);
            *desired_depths.get_unchecked_mut(stack_len) = MaybeUninit::new(desired_depth);
            stack_len += 1;
        }

        last_index += next_run.len();
        prev_run = next_run;
    }

    if !prev_run.sorted() {
        sort_ops.physical_sort(last_index - prev_run.len(), prev_run.len())
    }
}
