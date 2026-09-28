//! Optional consumer contracts and adapters over native Pasta arithmetic.

#[allow(
    unsafe_code,
    reason = "Transparent consumer wrappers borrow native buffers without copying."
)]
mod adapter;
mod traits;

pub use adapter::FieldAdapter;
pub use traits::Field;
