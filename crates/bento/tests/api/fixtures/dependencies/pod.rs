use super::bento;

#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
#[pod(crate = bento)]
pub struct Record {
    pub low: u16,
    pub high: u16,
    pub value: u32,
}

bento::embed_struct! {
    /// A record stored in a source-relative file.
    pub static RECORD: Record = "record.bin";
}

// The expansion must preserve the caller's `BYTES` constant in the length.
const BYTES: usize = 2;
bento::embed_array! {
    pub static WORDS: [u32; BYTES] = concat!(env!("CARGO_MANIFEST_DIR"), "/src/record.bin");
}

bento::embed_array! {
    #[cfg(any())]
    pub static DISABLED: [bool; 100] = "missing.bin";
}

#[test]
fn embedded_records_and_arrays() {
    assert_eq!(RECORD.low, 0x0201);
    assert_eq!(RECORD.high, 0x0403);
    assert_eq!(RECORD.value, 0x0807_0605);
    assert_eq!(*WORDS, [0x0403_0201, 0x0807_0605]);
    assert_eq!(bento::bytes_of(RECORD), bento::bytes_of_slice(WORDS));
}
