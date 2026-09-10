#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(any(feature = "record", feature = "zero-array"), derive(bento::Pod))]
pub struct Value(pub u64);

impl bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0.wrapping_mul(2))
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

// Compiles on either endianness when no storage operation is requested.
pub fn scale(value: Value) -> Value {
    bento::addition_chain!(value, 181)
}

#[cfg(feature = "record")]
bento::embed_struct! {
    pub static RECORD: Value = "record.bin";
}

#[cfg(feature = "empty-record")]
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Empty;

#[cfg(feature = "empty-record")]
pub static EMPTY: &Empty = bento::AlignedBytes([]).as_value();

#[cfg(feature = "primitive")]
pub static PRIMITIVE: &u64 = bento::AlignedBytes([0; 8]).as_value();

#[cfg(feature = "zero-array")]
pub static ZERO_ARRAY: &[Value; 0] = bento::AlignedBytes([]).as_array();
