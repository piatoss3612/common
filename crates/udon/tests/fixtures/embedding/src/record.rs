//! The field record shared by the build script and its consumer.

use udon::field::{Fp, Fq};

#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct FieldValues {
    pub fp: [Fp; 4],
    pub fq: [Fq; 4],
}
