#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

use bridge as bento;

pub mod arithmetic;

#[repr(transparent)]
#[derive(Clone, Copy, bridge::Pod)]
#[pod(crate = bridge)]
pub struct Record(pub [u32; 2]);

pub fn scale(value: bridge::Value) -> bridge::Value {
    bridge::addition_chain!(value, 181)
}

bridge::embed_struct! {
    pub static RECORD: Record = "record.bin";
}

#[test]
fn derives_without_a_direct_support_dependency() {
    arithmetic::check();
    assert_eq!(bridge::shift!(&[4, 0, 0, 0], 1), [2, 0, 0, 0]);
    assert_eq!(RECORD.0, [0x0403_0201, 0x0807_0605]);
    assert_eq!(bridge::bytes_of(RECORD), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(scale(bridge::Value(7)).0, 1267);
}
