use core::mem::MaybeUninit;
use core::ptr;

/// Merges non-decreasing runs `v[..mid]` and `v[mid..]` using `buf` as temporary storage, and
/// stores the result into `v[..]`. Does O(v.len()) comparisons and
/// O(v.len() * (1 + v.len() / scratch.len())) moves.
#[inline(always)]
pub fn merge<T, F>(v: &mut [T], scratch: &mut [MaybeUninit<T>], mid: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();
    let v_base = v.as_mut_ptr();
    let scratch_len = scratch.len();
    let scratch_base = MaybeUninit::slice_as_mut_ptr(scratch);

    unsafe {
        // SAFETY
        // The scratch and element array respectively have the following layouts:
        //
        //     |     merged elements    |    free space    |
        //     ^ scratch_base           ^ scratch_out      ^ scratch_end
        //
        //     | merged elements |    gap      | unmerged left |    gap    | unmerged right |
        //     ^ v_base          ^ merged_out  ^ left          ^ left_end  ^ right          ^ v_end
        //
        // Note that the 'gaps' here are purely logical, not physical. We
        // strictly copy from the element array to the scratch, and leave the
        // input array completely untouched, should a panic occur. Only when we
        // are done or the scratch buffer is full do we copy back the merged
        // elements into the source array, closing the gaps. This is a panicless
        // procedure, and thus safe. We never call the comparison operator again
        // on any element that was copied, so interior mutability is not a problem.
        let scratch_end = scratch_base.add(scratch_len);
        let v_end = v_base.add(len);

        let mut left = v_base;
        let mut left_end = left.add(mid);
        let mut right = left_end;
        let mut scratch_out = scratch_base;
        let mut merged_out = v_base;
        let mut merge_done = false;

        while !merge_done {
            // Fill the scratch space with merged elements.
            let free_scratch_space = scratch_end.sub_ptr(scratch_out);
            let left_len = left_end.sub_ptr(left);
            let right_len = v_end.sub_ptr(right);
            let safe_iters = free_scratch_space.min(left_len).min(right_len);
            for _ in 0..safe_iters {
                let right_less = is_less(&*right, &*left);
                let src = if right_less { right } else { left };
                ptr::copy_nonoverlapping(src, scratch_out, 1);

                scratch_out = scratch_out.add(1);
                left = left.add((!right_less) as usize);
                right = right.add(right_less as usize);
            }

            merge_done = left == left_end || right == v_end;
            if scratch_out == scratch_end || merge_done {
                // Move the remaining left elements next to the right elements.
                let new_left_len = left_end.sub_ptr(left);
                let new_left = right.sub(new_left_len);
                ptr::copy(left, new_left, new_left_len);
                left = new_left;
                left_end = left.add(new_left_len);

                // Move merged elements in scratch back to v and reset scratch.
                let merged_n = scratch_out.sub_ptr(scratch_base);
                ptr::copy_nonoverlapping(scratch_base, merged_out, merged_n);
                merged_out = merged_out.add(merged_n);
                scratch_out = scratch_base;
            }
        }
    }
}
