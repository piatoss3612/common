//! The stored format shared by the artifact generator and consumer.

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bento::Pod)]
pub struct Record {
    pub low: u16,
    pub high: u16,
    pub value: u32,
}

pub const RECORD: Record = Record {
    low: 0x0201,
    high: 0x0403,
    value: 0x0807_0605,
};

pub const RECORDS: [Record; 2] = [
    RECORD,
    Record {
        low: 0xfffe,
        high: 0xfdfc,
        value: 0xfbfa_f9f8,
    },
];
