#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

#[repr(transparent)]
#[derive(Clone, Copy, bridge::Pod)]
#[pod(crate = bridge)]
pub struct Record(pub [u32; 2]);

bridge::embed_struct! {
    pub static RECORD: Record = "record.bin";
}

#[test]
fn derives_without_a_direct_support_dependency() {
    assert_eq!(RECORD.0, [0x0403_0201, 0x0807_0605]);
    assert_eq!(bridge::bytes_of(RECORD), &[1, 2, 3, 4, 5, 6, 7, 8]);
}
