//! Embeds records written in their validated little-endian layout.

use zakura_bento as bento;

#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
struct Record {
    low: u16,
    high: u16,
    value: u32,
}

bento::embed_struct! {
    static RECORD: Record = "data/record.bin";
}

bento::embed_array! {
    static WORDS: [u32; 2] = "data/record.bin";
}

fn main() {
    let record = Record {
        low: 0x0201,
        high: 0x0403,
        value: 0x0807_0605,
    };

    // A generator can write this byte view directly to `data/record.bin`.
    assert_eq!(bento::bytes_of(&record), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(bento::bytes_of(RECORD), bento::bytes_of(&record));
    assert_eq!(WORDS, &[0x0403_0201, 0x0807_0605]);
    assert_eq!(bento::bytes_of_slice(WORDS), bento::bytes_of(RECORD));
}
