#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

use bento::const_arithmetic::{U256, U320, m255, u256};

// Numeric word order is independent of the target's byte order. These
// assertions run during compilation, including on targets we cannot execute.
pub const MODULUS: U256 =
    u256::from_hex("0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");
pub const ENCODED: U256 = m255::from_u256(&MODULUS, &[u64::MAX; 4]);
pub const ROOT: U256 = m255::two_adic_root_of_unity(&[97, 0, 0, 0], 5, 5);
pub const RATIO: U320 = u256::round_shifted_ratio(&[u64::MAX; 4], u128::MAX, 384);

const _: () = {
    let decoded = m255::to_u256(&MODULUS, &ENCODED);
    assert!(decoded[0] == 37 && decoded[1] == 0 && decoded[2] == 0 && decoded[3] == 0);
    let root = m255::to_u256(&[97, 0, 0, 0], &ROOT);
    assert!(root[0] == 28 && root[1] == 0 && root[2] == 0 && root[3] == 0);
    assert!(RATIO[0] == 1 && RATIO[1] == 0 && RATIO[2] == u64::MAX);
    assert!(RATIO[3] == u64::MAX && RATIO[4] == 0);
};

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
