fn main() {
    let mut v = vec![3, 1, 2, 0];
    let mut v2 = vec!["3", "1", "2", "0"];
    
    // v.sort();
    newsort::sort(&mut v);
    // newsort::sort(&mut v2);
    dbg!(v);

}