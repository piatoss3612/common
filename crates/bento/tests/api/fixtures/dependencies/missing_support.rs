#![no_std]
#![deny(warnings)]

#[repr(C)]
#[derive(Clone, Copy, macros::Pod)]
pub struct Record(pub u32);
