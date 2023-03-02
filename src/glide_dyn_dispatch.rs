use core::mem::MaybeUninit;

// Lazy logical runs as in Glidesort.
pub struct LogicalRun {
    // (start_idx << 1) | sorted, doesn't overflow because idx <= isize::MAX.
    // We don't use a bool because it would increase our stack size from 16 to
    // 24 bytes due to alignment.
    start_and_sorted: usize,
    length: usize,
}

impl LogicalRun {
    #[inline(always)]
    pub fn new_sorted(start: usize, length: usize) -> Self {
        Self {
            start_and_sorted: (start << 1) | 1,
            length,
        }
    }

    #[inline(always)]
    pub fn new_unsorted(start: usize, length: usize) -> Self {
        Self {
            start_and_sorted: start << 1,
            length,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline(always)]
    pub fn start_idx(&self) -> usize {
        self.start_and_sorted >> 1
    }

    #[inline(always)]
    fn should_physically_merge(&self, other: &Self, el_len: usize, scratch_len: usize) -> bool {
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
        let left_sorted = (self.start_and_sorted & 1) == 1;
        let right_sorted = (other.start_and_sorted & 1) == 1;
        let total_length = self.length + other.length;
        (total_length > scratch_len)
            | (left_sorted & right_sorted)
            | (left_sorted | right_sorted) & (total_length.saturating_mul(total_length) >= el_len)
    }
}

fn logical_merge(
    left: LogicalRun,
    right: LogicalRun,
    sort_ops: &mut dyn crate::SortOps,
    el_len: usize,
    scratch_len: usize,
) -> LogicalRun {
    if left.should_physically_merge(&right, el_len, scratch_len) {
        if left.start_and_sorted & 1 == 0 {
            sort_ops.physical_sort(left.start_idx(), left.start_idx() + left.len());
        }
        if right.start_and_sorted & 1 == 0 {
            sort_ops.physical_sort(right.start_idx(), right.start_idx() + right.len());
        }
        sort_ops.physical_merge(
            left.start_idx(),
            right.start_idx(),
            right.start_idx() + right.len(),
        );
        LogicalRun {
            start_and_sorted: left.start_and_sorted | 1,
            length: left.length + right.length,
        }
    } else {
        LogicalRun {
            start_and_sorted: left.start_and_sorted & !1,
            length: left.length + right.length,
        }
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

/// A stack of merge nodes. For each node we track its desired depth in the
/// merge tree, as well as its left child.
struct MergeStack {
    left_children: [MaybeUninit<LogicalRun>; 64],
    desired_depths: [MaybeUninit<u8>; 64],
    len: usize,
}

impl MergeStack {
    /// Creates an empty merge stack.
    #[inline(always)]
    fn new() -> Self {
        unsafe {
            // SAFETY: an array of MaybeUninit's is trivially init.
            Self {
                left_children: MaybeUninit::uninit().assume_init(),
                desired_depths: MaybeUninit::uninit().assume_init(),
                len: 0,
            }
        }
    }

    /// Push a merge node on the stack given its left child and desired depth.
    ///
    /// SAFETY: the stack may not be full (64 elements).
    #[inline(always)]
    unsafe fn push_node_unchecked(&mut self, left_child: LogicalRun, desired_depth: u8) {
        unsafe {
            *self.left_children.get_unchecked_mut(self.len) = MaybeUninit::new(left_child);
            *self.desired_depths.get_unchecked_mut(self.len) = MaybeUninit::new(desired_depth);
            self.len += 1;
        }
    }

    /// Pop a merge node off the stack, returning its left child.
    #[inline(always)]
    fn pop_node(&mut self) -> Option<LogicalRun> {
        if self.len == 0 {
            return None;
        }

        // SAFETY: len > 0 guarantees this is initialized by a previous push.
        self.len -= 1;
        Some(unsafe {
            self.left_children
                .get_unchecked(self.len)
                .assume_init_read()
        })
    }

    /// Pops from the top of the stack if the merge node at the top of the stack
    /// has a desired depth deeper than or equal to the given depth, returning
    /// the left child of the merge node.
    #[inline(always)]
    fn pop_if_deeper_or_eq_to(&mut self, depth: u8) -> Option<LogicalRun> {
        if self.len == 0 {
            return None;
        }

        // SAFETY: len > 0 guarantees this is initialized by a previous push.
        unsafe {
            let top_depth = self
                .desired_depths
                .get_unchecked(self.len - 1)
                .assume_init();
            if top_depth < depth {
                return None;
            }

            self.len -= 1;
            Some(
                self.left_children
                    .get_unchecked(self.len)
                    .assume_init_read(),
            )
        }
    }
}

pub fn sort(
    sort_ops: &mut dyn crate::SortOps,
    eager_sort: bool,
    el_len: usize,
    scratch_len: usize,
) {
    let scale_factor = merge_tree_scale_factor(el_len);
    let mut merge_stack = MergeStack::new();

    let mut prev_run_start_idx = 0;
    let mut prev_run;
    prev_run = sort_ops.create_run(0, eager_sort);
    while prev_run_start_idx + prev_run.len() < el_len {
        let next_run_start_idx = prev_run_start_idx + prev_run.len();
        let next_run;
        next_run = sort_ops.create_run(next_run_start_idx, eager_sort);

        let desired_depth = merge_tree_depth(
            prev_run_start_idx,
            next_run_start_idx,
            next_run_start_idx + next_run.len(),
            scale_factor,
        );

        // Create the left child of our next node and eagerly merge all nodes
        // with a deeper desired merge depth into it.
        let mut left_child = prev_run;
        while let Some(left_descendant) = merge_stack.pop_if_deeper_or_eq_to(desired_depth) {
            left_child = logical_merge(left_descendant, left_child, sort_ops, el_len, scratch_len);
        }

        unsafe {
            // SAFETY: we just maintained the invariant that desired_depth > top_depth.
            // This means the stack must consist of strictly increasing depths. Since
            // desired depths are all < 64 this ensures our stack can contain at
            // most 64 values and we do not overflow, as this is the only place we
            // ever push to the stack.
            merge_stack.push_node_unchecked(left_child, desired_depth);
        }
        prev_run_start_idx = next_run_start_idx;
        prev_run = next_run;
    }

    // Collapse the stack down to a single logical run and physically sort it.
    let mut result = prev_run;
    while let Some(left_child) = merge_stack.pop_node() {
        result = logical_merge(left_child, result, sort_ops, el_len, scratch_len);
    }
    if result.start_and_sorted & 1 == 0 {
        sort_ops.physical_sort(result.start_idx(), result.len())
    }
}
