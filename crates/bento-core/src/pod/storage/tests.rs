extern crate std;

use super::*;

#[test]
fn casts_round_trip_through_stored_bytes() {
    static BYTES: AlignedBytes<64> = AlignedBytes([7; 64]);
    let values: &'static [u16; 32] = BYTES.as_array();
    assert!(values.iter().all(|&value| value == 0x0707));
    assert_eq!(bytes_of_slice(&values[..]), &BYTES.0);
    assert_eq!(bytes_of_slice(values).as_ptr(), BYTES.0.as_ptr());

    let single: &'static [u64; 8] = BYTES.as_value();
    assert_eq!(bytes_of(single), &BYTES.0);
    assert_eq!(core::ptr::from_ref(single).cast::<u8>(), BYTES.0.as_ptr());
    assert_eq!(single[0], 0x0707_0707_0707_0707);
}

#[test]
fn primitives_preserve_little_endian_bytes_at_boundaries() {
    macro_rules! check {
        ($($ty:ty),*) => {$(
            for value in [0, 1, <$ty>::MAX / 2, <$ty>::MAX] {
                assert_eq!(bytes_of(&value), value.to_le_bytes());
            }
        )*};
    }
    check!(u8, u16, u32, u64);
}
