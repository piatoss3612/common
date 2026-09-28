//! Verifies that embedded files recover the generator's record values.
#![forbid(unsafe_code)]

mod record;

bento::embed_struct! {
    static RECORD: record::Record = concat!(env!("OUT_DIR"), "/record.bin");
}

bento::embed_array! {
    static RECORDS: [record::Record; 2] = concat!(env!("OUT_DIR"), "/records.bin");
}

fn main() {
    assert_eq!(*RECORD, record::RECORD);
    assert_eq!(*RECORDS, record::RECORDS);
    assert_eq!(bento::bytes_of(RECORD), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        bento::bytes_of_slice(RECORDS),
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 254, 255, 252, 253, 248, 249, 250, 251
        ],
    );
}
