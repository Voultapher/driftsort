use core::cmp;
use core::intrinsics;
use core::mem::MaybeUninit;
use core::ptr;

/// Merges non-decreasing runs `v[..mid]` and `v[mid..]` using `buf` as temporary storage, and
/// stores the result into `v[..]`.
#[inline(never)]
pub fn merge<T, F>(v: &mut [T], scratch: &mut [MaybeUninit<T>], mid: usize, is_less: &mut F)
where
    F: FnMut(&T, &T) -> bool,
{
    let len = v.len();

    if mid == 0 || mid >= len || scratch.len() < cmp::min(mid, len - mid) {
        intrinsics::abort();
    }

    // SAFETY: We checked that the two slices must be non-empty and `mid` must be in bounds. The
    // caller has to guarantee that Buffer `buf` must be long enough to hold a copy of the shorter
    // slice. Also, `T` must not be a zero-sized type. We checked that T is observation safe. Should
    // is_less panic v was not modified in bi_directional_merge and retains it's original input.
    // buffer and v must not alias and swap has v.len() space.
    unsafe {
        // The merge process first copies the shorter run into `buf`. Then it traces the newly
        // copied run and the longer run forwards (or backwards), comparing their next unconsumed
        // elements and copying the lesser (or greater) one into `v`.
        //
        // As soon as the shorter run is fully consumed, the process is done. If the longer run gets
        // consumed first, then we must copy whatever is left of the shorter run into the remaining
        // gap in `v`.
        //
        // Intermediate state of the process is always tracked by `gap`, which serves two purposes:
        // 1. Protects integrity of `v` from panics in `is_less`.
        // 2. Fills the remaining gap in `v` if the longer run gets consumed first.
        //
        // Panic safety:
        //
        // If `is_less` panics at any point during the process, `gap` will get dropped and fill the
        // gap in `v` with the unconsumed range in `buf`, thus ensuring that `v` still holds every
        // object it initially held exactly once.

        let buf = MaybeUninit::slice_as_mut_ptr(scratch);

        let v_base = v.as_mut_ptr();
        let v_mid = v_base.add(mid);
        let v_end = v_base.add(len);

        let left_len = mid;
        let right_len = len - mid;

        let left_is_shorter = left_len <= right_len;

        let save_base = if left_is_shorter { v_base } else { v_mid };
        let save_len = if left_is_shorter { left_len } else { right_len };

        ptr::copy_nonoverlapping(save_base, buf, save_len);

        let mut merge_state;

        if left_is_shorter {
            merge_state = MergeState {
                start: buf,
                end: buf.add(mid),
                dst: v_base,
            };

            merge_state.merge_up(v_mid, v_end, is_less);
        } else {
            merge_state = MergeState {
                start: buf,
                end: buf.add(len - mid),
                dst: v_mid,
            };

            merge_state.merge_down(v_base, buf, v_end, is_less);
        }
        // Finally, `merge_state` gets dropped. If the shorter run was not fully consumed, whatever
        // remains of it will now be copied into the hole in `v`.
    }

    // When dropped, copies the range `start..end` into `dst..`.
    struct MergeState<T> {
        start: *mut T,
        end: *mut T,
        dst: *mut T,
    }

    impl<T> MergeState<T> {
        unsafe fn merge_up<F: FnMut(&T, &T) -> bool>(
            &mut self,
            mut right: *mut T,
            right_end: *const T,
            is_less: &mut F,
        ) {
            // left == self.start
            // out == self.dst

            while self.start != self.end && right as *const T != right_end {
                let consume_left = !is_less(&*right, &*self.start);

                let src = if consume_left { self.start } else { right };
                ptr::copy_nonoverlapping(src, self.dst, 1);

                self.start = self.start.add(consume_left as usize);
                right = right.add(!consume_left as usize);

                self.dst = self.dst.add(1);
            }
        }

        unsafe fn merge_down<F: FnMut(&T, &T) -> bool>(
            &mut self,
            left_end: *const T,
            right_end: *const T,
            mut out: *mut T,
            is_less: &mut F,
        ) {
            // left == self.dst;
            // right == self.end;

            loop {
                let left = self.dst.sub(1);
                let right = self.end.sub(1);
                out = out.sub(1);

                let consume_left = is_less(&*right, &*left);

                let src = if consume_left { left } else { right };
                ptr::copy_nonoverlapping(src, out, 1);

                self.dst = left.add(!consume_left as usize);
                self.end = right.add(consume_left as usize);

                if self.dst as *const T == left_end || self.end as *const T == right_end {
                    break;
                }
            }
        }
    }

    impl<T> Drop for MergeState<T> {
        fn drop(&mut self) {
            // SAFETY: `T` is not a zero-sized type, and these are pointers into a slice's elements.
            unsafe {
                let len = self.end.sub_ptr(self.start);
                ptr::copy_nonoverlapping(self.start, self.dst, len);
            }
        }
    }
}
