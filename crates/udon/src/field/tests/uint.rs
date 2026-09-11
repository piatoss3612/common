use super::*;

#[test]
fn integer_windows_and_shifts_match_big_integers() {
    let mut state = 0x3c6e_f372_fe94_f82b;
    for _ in 0..16 {
        let bytes = deterministic_bytes::<32>(&mut state);
        let value = CanonicalUint::from_le_bytes(bytes);
        let x = BigUint::from_bytes_le(&bytes);
        assert_eq!(value.to_le_bytes(), bytes);
        assert_eq!(integer(&value.limbs()), x);
        assert_eq!(value.highest_set_bit(), Some(x.bits() as usize - 1));
        for offset in 0..=256 {
            assert_eq!(
                value.bit(offset),
                (offset < 256).then(|| x.bit(offset as u64))
            );
            assert_eq!(integer(&value.shr(offset).limbs()), &x >> offset);
            assert_eq!(value.fits_in_bits(offset), x.bits() <= offset as u64);
            for width in [0, 1, 2, 31, 63, 64, 65, 127, 128, 129, 255, 256, 257] {
                let valid = width != 0 && offset + width <= 256;
                let slice = value.bit_slice(offset, width);
                assert_eq!(slice.is_some(), valid);
                let expected = (&x >> offset) & ((BigUint::from(1u8) << width) - 1u8);
                if let Some(slice) = slice {
                    assert_eq!(integer(&slice.limbs()), expected);
                }
                let window = value.window(offset, width);
                assert_eq!(window.is_some(), valid && width <= 64);
                if let Some(window) = window {
                    assert_eq!(BigUint::from(window), expected);
                }
            }
        }
        for (offset, width) in [(usize::MAX, 1), (1, usize::MAX), (0, usize::MAX)] {
            assert_eq!(value.window(offset, width), None);
            assert_eq!(value.bit_slice(offset, width), None);
        }
        assert_eq!(value.bit(usize::MAX), None);
        assert_eq!(value.shr(usize::MAX).limbs(), [0; 4]);
        assert!(value.fits_in_bits(usize::MAX));
    }
}

#[test]
fn integer_powers_and_addition_cover_carries_and_overflow() {
    let zero = CanonicalUint::from_limbs([0; 4]);
    assert_eq!(zero.highest_set_bit(), None);
    assert!(zero.fits_in_bits(0));
    let mut values = vec![zero, CanonicalUint::from_limbs([u64::MAX; 4])];
    for bit in 0..256 {
        let power = CanonicalUint::power_of_two(bit).unwrap();
        assert_eq!(integer(&power.limbs()), BigUint::from(1u8) << bit);
        assert_eq!(power.highest_set_bit(), Some(bit));
        assert!(!power.fits_in_bits(bit));
        assert!(power.fits_in_bits(bit + 1));
        values.push(power);
        values.push(CanonicalUint::from_limbs(limbs(
            &((BigUint::from(1u8) << bit) - 1u8),
        )));
    }
    for bit in [256, 257, usize::MAX] {
        assert_eq!(CanonicalUint::power_of_two(bit), None);
    }
    for value in values {
        for addend in [0, 1, u128::from(u64::MAX), 1 << 64, u128::MAX] {
            let expected = integer(&value.limbs()) + addend;
            let actual = value.checked_add_u128(addend);
            assert_eq!(actual.is_some(), expected.bits() <= 256);
            if let Some(actual) = actual {
                assert_eq!(integer(&actual.limbs()), expected);
            }
        }
    }
}
