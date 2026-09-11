use super::*;

fn check_reduction<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut state = 0xbb67_ae85_84ca_a73b;
    for bytes in [deterministic_bytes::<1024>(&mut state), [0xff; 1024]] {
        for length in (0..=257).chain([511, 512, 513, 1024]) {
            let expected = BigUint::from_bytes_le(&bytes[..length]) % &p;
            assert_value(
                PastaField::<M>::from_bytes_reduced(&bytes[..length]),
                &expected,
            );
        }
        let wide = bytes[..64].try_into().unwrap();
        assert_value(
            PastaField::<M>::from_wide_bytes_reduced(wide),
            &BigUint::from_bytes_le(wide),
        );
    }
    let halves = [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        (BigUint::from(1u8) << 256usize) - 1u8,
    ];
    for low in &halves {
        for high in &halves {
            let mut bytes = [0; 64];
            bytes[..32].copy_from_slice(&CanonicalUint::from_limbs(limbs(low)).to_le_bytes());
            bytes[32..].copy_from_slice(&CanonicalUint::from_limbs(limbs(high)).to_le_bytes());
            assert_value(
                PastaField::<M>::from_wide_bytes_reduced(&bytes),
                &(low + (high << 256usize)),
            );
        }
    }
    for _ in 0..256 {
        let bytes = deterministic_bytes::<32>(&mut state);
        let x = BigUint::from_bytes_le(&bytes);
        let uint = CanonicalUint::from_le_bytes(bytes);
        assert_eq!(PastaField::<M>::from_bytes(bytes).is_some(), x < p);
        assert_eq!(
            PastaField::<M>::from_bytes(bytes),
            PastaField::from_canonical_uint(uint)
        );
        assert_value(PastaField::<M>::from_uint_reduced(uint), &x);
    }
}

#[test]
fn little_endian_reduction_covers_each_chunk_boundary() {
    check_reduction::<PallasBase>();
    check_reduction::<PallasScalar>();
}

fn check_encodings<M: PrimeModulus>() {
    let p = modulus::<M>();
    for x in [
        BigUint::from(0u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        (BigUint::from(1u8) << 256usize) - 1u8,
    ] {
        let uint = CanonicalUint::from_limbs(limbs(&x));
        assert_eq!(PastaField::<M>::from_canonical_uint(uint).is_some(), x < p);
        assert_eq!(
            PastaField::<M>::from_bytes(uint.to_le_bytes()).is_some(),
            x < p
        );
        assert_value(PastaField::<M>::from_uint_reduced(uint), &x);
    }
    for (value, x) in samples::<M>(64) {
        assert_eq!(PastaField::<M>::from_bytes(value.to_bytes()), Some(value));
        assert_eq!(integer(&value.to_canonical_uint().limbs()), x);
        assert_eq!(
            PastaField::<M>::from_montgomery_limbs(value.montgomery_limbs()),
            value
        );
        assert_eq!(
            PastaField::<M>::from_hex(&std::format!("0x{x:064x}")),
            value
        );
        assert_eq!(value.is_odd(), x.bit(0));
    }
    for signed in [i64::MIN, -(1 << 62), -1, 0, 1, 1 << 62, i64::MAX] {
        assert_value(
            PastaField::<M>::from_i64(signed),
            &signed_mod(BigInt::from(signed), &p),
        );
    }
    for malformed in [
        "",
        "0x01",
        "0000000000000000000000000000000000000000000000000000000000000001",
        "0X0000000000000000000000000000000000000000000000000000000000000001",
        "0x000000000000000000000000000000000000000000000000000000000000000g",
        "0x00000000000000000000000000000000000000000000000000000000000000001",
        "0x00000000000000000000000000000000000000000000000000000000000000é",
    ] {
        assert!(std::panic::catch_unwind(|| PastaField::<M>::from_hex(malformed)).is_err());
    }
}

#[test]
fn canonical_encodings_and_signed_inputs_cover_extremes() {
    check_encodings::<PallasBase>();
    check_encodings::<PallasScalar>();
}
