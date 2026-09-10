#![no_std]
#![deny(warnings)]

pub fn scale(value: u64) -> u64 {
    macros::addition_chain!(value, 2)
}
