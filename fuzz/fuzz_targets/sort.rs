#![no_main]

use libfuzzer_sys::fuzz_target;
use driftsort::sort;

fuzz_target!(|data: &[u8]| {

    let mut vec: Vec<u8> = data.to_vec();
    sort(&mut vec);

    for window in vec.windows(2) {
        assert!(window[0] <= window[1]);
    }

});
