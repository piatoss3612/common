#![no_std]
#![deny(warnings)]

pub fn scale(value: u64) -> u64 {
    macros::addition_chain!(value, 2)
}

#[repr(C)]
#[derive(Clone, Copy, macros::Pod)]
pub struct Record(pub u32);
