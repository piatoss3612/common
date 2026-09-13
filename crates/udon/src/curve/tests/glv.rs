use num_bigint::{BigInt, BigUint};

use super::{reference::Reference, *};
use crate::curve::{
    eisenstein::{REPRESENTATIVES, recode},
    parameters::GlvParameters,
};
use crate::test_support::modulus;

fn big_limbs(limbs: &[u64]) -> BigUint {
    BigUint::from_bytes_le(
        &limbs
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn from_big<M: PrimeModulus>(integer: &BigUint) -> PastaField<M> {
    let mut bytes = [0; 32];
    let value = integer.to_bytes_le();
    bytes[..value.len()].copy_from_slice(&value);
    PastaField::from_bytes(bytes).unwrap()
}

fn digit_pair(code: u8) -> (i8, i8) {
    if code == 0 {
        return (0, 0);
    }
    assert!(code <= 48);
    let value = usize::from(code - 1);
    let (mut a, mut b) = REPRESENTATIVES[value / 6];
    for _ in 0..(value % 6) / 2 {
        (a, b) = (-b, a - b);
    }
    if value % 2 == 1 { (-a, -b) } else { (a, b) }
}

fn assert_digits(a: i128, b: i128) {
    let (digits, len) = recode(a, b);
    let mut reconstructed = (BigInt::from(0), BigInt::from(0));
    for &code in digits[..len].iter().rev() {
        let (a, b) = digit_pair(code);
        reconstructed.0 = (reconstructed.0 << 1) + a;
        reconstructed.1 = (reconstructed.1 << 1) + b;
    }
    assert_eq!(reconstructed, (BigInt::from(a), BigInt::from(b)));
    assert!(digits[len..].iter().all(|digit| *digit == 0));
}

fn decomposition<C: PastaCurve>() {
    let lattice = GlvParameters::<C>::BASIS;
    let n = modulus::<C::Scalar>();
    let a = BigUint::from(lattice.a);
    let b = BigUint::from(lattice.b);
    let d = BigUint::from(lattice.d);
    let lambda = BigUint::from_bytes_le(&PastaField::<C::Scalar>::zeta().to_bytes());
    assert_eq!(&a * &d + &b * &b, n);
    assert_eq!((&lambda * &b) % &n, a);
    assert_eq!((&b + &lambda * &d) % &n, BigUint::from(0_u32));
    let scale = BigUint::from(1_u32) << 384;
    let g1 = big_limbs(&lattice.g1);
    let g2 = big_limbs(&lattice.g2);
    assert_eq!(g1, (&scale * &d + (&n >> 1)) / &n);
    assert_eq!(g2, (&scale * &b + (&n >> 1)) / &n);
    assert!((&a + &b) / 2_u32 + 1_u32 < BigUint::from(1_u32) << 127);
    assert!((&b + &d) / 2_u32 + 1_u32 < BigUint::from(1_u32) << 127);

    let mut scalars = scalar_corpus::<C>();
    for bit in 0..255 {
        let power =
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
        scalars.extend([
            power.sub(&PastaField::ONE),
            power,
            power.add(&PastaField::ONE),
        ]);
    }
    for scalar in field_samples::<C::Scalar>().take(2048) {
        scalars.extend([scalar, scalar.neg()]);
    }
    // Cross rounding boundaries near both ends and throughout each coefficient
    // range. Check the exact fixed-point boundary as well as the ideal ratio.
    for (g, coordinate) in [(&g1, &d), (&g2, &b)] {
        for q in [
            BigUint::from(0_u32),
            BigUint::from(1_u32),
            coordinate / 3_u32,
            coordinate / 2_u32,
            coordinate - 1_u32,
        ] {
            for boundary in [
                ((&q * 2_u32 + 1_u32) * &scale) / (g * 2_u32),
                ((&q * 2_u32 + 1_u32) * &n) / (coordinate * 2_u32),
            ] {
                for k in [&boundary - 1_u32, boundary.clone(), &boundary + 1_u32] {
                    if k < n {
                        scalars.push(from_big(&k));
                    }
                }
            }
        }
    }
    let mut signs = [false; 4];
    for scalar in scalars {
        let k = BigUint::from_bytes_le(&scalar.to_bytes());
        let q1 = (&g1 * &k + (&scale >> 1)) / &scale;
        let q2 = (&g2 * &k + (&scale >> 1)) / &scale;
        let expected_a = BigInt::from(k.clone()) - BigInt::from(&q1 * &a + &q2 * &b);
        let expected_b = BigInt::from(&q1 * &b) - BigInt::from(&q2 * &d);
        let (a, b) = glv_decompose::<C>(&scalar);
        assert_ne!(a, i128::MIN);
        assert_ne!(b, i128::MIN);
        assert!(a.unsigned_abs() <= (lattice.a + lattice.b) / 2 + 1);
        assert!(b.unsigned_abs() <= (lattice.b + lattice.d) / 2 + 1);
        assert_eq!((BigInt::from(a), BigInt::from(b)), (expected_a, expected_b));
        let n_signed = BigInt::from(n.clone());
        let reconstructed =
            (BigInt::from(a) + BigInt::from(b) * BigInt::from(lambda.clone())) % &n_signed;
        assert_eq!((reconstructed + &n_signed) % &n_signed, BigInt::from(k));
        signs[usize::from(a < 0) * 2 + usize::from(b < 0)] = true;
        assert_digits(a, b);
    }
    assert!(signs.into_iter().all(|seen| seen));
}

#[test]
fn lattice_rounding_and_signed_reconstruction_match_integers() {
    decomposition::<Pallas>();
    decomposition::<Vesta>();
}

#[test]
fn joint_digits_cover_rotations_and_extreme_halves() {
    for code in 1..=48 {
        let (a, b) = digit_pair(code);
        let (digits, len) = recode(i128::from(a), i128::from(b));
        assert_eq!(len, 1);
        assert_eq!(digits[0], code);
    }
    let extremes = [0, 1, -1, 5, -5, i128::MAX, -i128::MAX];
    for a in extremes {
        for b in extremes {
            assert_digits(a, b);
        }
    }
    // Exhaust the terminal region used in the digit-buffer bound.
    for a in -5..=5 {
        for b in -5..=5 {
            assert!(recode(a, b).1 <= 5);
            assert_digits(a, b);
        }
    }
}

#[test]
fn signed_windows_reconstruct_partial_windows_and_carries() {
    use crate::curve::fixed_base::signed_window_digits;
    for w in 2..=8 {
        let n = 128_usize.div_ceil(w);
        let mut values = vec![0, 1, u128::MAX, i128::MAX as u128];
        for bit in (w - 1..128).step_by(w) {
            let power = 1_u128 << bit;
            values.extend([power - 1, power, power + 1]);
        }
        for value in values {
            let (digits, carry) = signed_window_digits(value, w);
            let mut integer = BigInt::from(u8::from(carry));
            for &digit in digits[..n].iter().rev() {
                assert!((-(1 << (w - 1))..1 << (w - 1)).contains(&digit));
                integer = (integer << w) + digit;
            }
            assert_eq!(integer, BigInt::from(value));
        }
    }
}

#[test]
fn pasta_lattice_bounds_allow_only_second_half_width_two_carries() {
    use crate::curve::fixed_base::signed_window_digits;
    for bounds in [
        GlvParameters::<Pallas>::BOUNDS,
        GlvParameters::<Vesta>::BOUNDS,
    ] {
        // Final carry is monotone in the magnitude. These conservative integer
        // upper bounds exclude every other width and the first width-2 half.
        for w in 2..=8 {
            for (half, bound) in bounds.into_iter().enumerate() {
                assert_eq!(signed_window_digits(bound, w).1, w == 2 && half == 1);
            }
        }
    }
}

fn endomorphisms<C: PastaCurve>() {
    let p = modulus::<C::Base>();
    let lambda = BigUint::from_bytes_le(&PastaField::<C::Scalar>::zeta().to_bytes());
    let generator = AffinePoint::<C>::GENERATOR;
    let cached = PreparedAffinePoint::from_affine(&generator);
    let mut rotated = generator;
    for rotation in 0..3 {
        assert_eq!(generator.rotated(rotation), rotated);
        assert_eq!(cached.rotated(rotation), rotated);
        rotated = rotated.endomorphism();
    }
    assert!(generator.valid_cache());
    assert!(cached.valid_cache());
    let reference = Reference::generator(&p);
    reference
        .mul(&lambda, &p)
        .assert_point(&generator.endomorphism().to_point());
    for point in [
        Point::IDENTITY,
        generator.to_point(),
        generator.neg().to_point(),
        generator.to_point().double(),
    ] {
        assert_eq!(point.endomorphism().endomorphism().endomorphism(), point);
        let scaled = scaled(&point, 19);
        assert_eq!(scaled.endomorphism(), point.endomorphism().to_projective());
        assert_eq!(
            scaled.endomorphism().coordinates().2,
            scaled.coordinates().2
        );
        for scalar in scalar_corpus::<C>() {
            // The retained binary ladder is independent of GLV and recoding.
            let expected = crate::curve::scalar::multiply(&scalar, |sum| sum.add(&scaled));
            assert_eq!(point.mul_projective(&scalar), expected);
            assert_eq!(scaled.mul(&scalar), expected);
        }
    }
}

#[test]
fn endomorphism_pairing_and_scaled_multiplication() {
    endomorphisms::<Pallas>();
    endomorphisms::<Vesta>();
}
