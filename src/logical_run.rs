use std::mem::MaybeUninit;

use crate::stable_quicksort;

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
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline(always)]
    fn start_idx(&self) -> usize {
        self.start_and_sorted >> 1
    }

    pub fn create<T, F: FnMut(&T, &T) -> bool>(
        el: &mut [T],
        start: usize,
        eager_sort: bool,
        is_less: &mut F,
    ) -> Self {
        // FIXME: actually detect runs.
        Self {
            start_and_sorted: start << 1,
            length: el.len().saturating_sub(start).min(32),
        }
    }

    // TODO: Should this be inline or not? Less monomorphization on T (?). Smaller
    // total code size inlined.
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

    pub fn logical_merge<T, F: FnMut(&T, &T) -> bool>(
        mut self,
        el: &mut [T],
        scratch: &mut [MaybeUninit<T>],
        mut other: LogicalRun,
        is_less: &mut F,
    ) -> Self {
        if self.should_physically_merge(&other, el.len(), scratch.len()) {
            self.physical_sort(el, scratch, is_less);
            other.physical_sort(el, scratch, is_less);
            crate::physical_merge(
                &mut el[self.start_idx()..other.start_idx() + other.length],
                // unsafe { el.get_unchecked_mut(self.start_idx()..other.start_idx() + other.length) },
                scratch,
                self.length,
                is_less,
            );
            Self {
                start_and_sorted: self.start_and_sorted | 1,
                length: self.length + other.length,
            }
        } else {
            Self {
                start_and_sorted: self.start_and_sorted & !1,
                length: self.length + other.length,
            }
        }
    }

    pub fn physical_sort<T, F: FnMut(&T, &T) -> bool>(
        &mut self,
        el: &mut [T],
        scratch: &mut [MaybeUninit<T>],
        is_less: &mut F,
    ) {
        if self.start_and_sorted & 1 == 0 {
            stable_quicksort(
                &mut el[self.start_idx()..self.start_idx() + self.length],
                // unsafe { el.get_unchecked_mut(self.start_idx()..self.start_idx() + self.length) },
                scratch,
                is_less,
            );
            self.start_and_sorted |= 1;
        }
    }
}
