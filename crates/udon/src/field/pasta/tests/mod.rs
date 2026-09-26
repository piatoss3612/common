//! Independent checks of the Pasta field implementation.

use super::test_support::*;

#[cfg(feature = "traits")]
mod adapter;
mod arithmetic;
mod batch;
mod batch_inversion;
mod constants;
mod encoding;
mod kernels;
mod parameters;
mod products;
mod properties;
mod uint;

mod constant_prefix;
mod sqrt_ratios;

mod specialization;
