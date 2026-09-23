use super::*;
use crate::field::{ConstantPrefix, ConstantPrefixError};

fn assert_raw<M: PrimeModulus>(values: &[PastaField<M>], expected: PastaField<M>) {
    assert!(
        values
            .iter()
            .all(|value| value.montgomery_limbs() == expected.montgomery_limbs())
    );
}

fn contracts_field<M: PrimeModulus>() {
    let sentinel = PastaField::<M>::from_montgomery_limbs(limbs(&(modulus::<M>() * 2u8 - 1u8)));
    let tail = [PastaField::<M>::ONE; 2];
    assert_eq!(
        ConstantPrefix::new(1, &sentinel, &tail).unwrap_err(),
        ConstantPrefixError::TailTooLong { length: 1, tail: 2 }
    );
    let input = ConstantPrefix::new(4, &sentinel, &tail).unwrap();
    assert_eq!(input.len(), 4);
    assert!(!input.is_empty());
    assert_eq!(input.prefix_len(), 2);
    assert_eq!(input.constant().reduce(), sentinel.reduce());
    assert_eq!(input.tail().as_ptr(), tail.as_ptr());
    let mut out = [sentinel; 6];
    assert!(input.write_values(&mut out[..3]).is_err());
    assert_raw(&out, sentinel);
    input.write_values(&mut out).unwrap();
    assert_raw(&out[..2], sentinel);
    assert_raw(&out[2..4], PastaField::ONE);
    assert_raw(&out[4..], sentinel);
    let empty = ConstantPrefix::new(0, &sentinel, &tail[..0]).unwrap();
    assert!(empty.is_empty());
    empty.write_values(&mut out).unwrap();
    let huge = ConstantPrefix::new(usize::MAX, &sentinel, &tail).unwrap();
    assert_eq!(huge.prefix_len(), usize::MAX - 2);
    assert_eq!(
        huge.write_values(&mut out),
        Err(ConstantPrefixError::OutputTooShort {
            required: usize::MAX,
            actual: 6
        })
    );
    let full = ConstantPrefix::new(tail.len(), &sentinel, &tail).unwrap();
    assert_eq!(full.prefix_len(), 0);
    full.write_values(&mut out).unwrap();
    assert_raw(&out[..tail.len()], PastaField::ONE);
    let reduced_tail = tail.map(PastaField::reduce);
    let reduced = ConstantPrefix::new(4, &sentinel.reduce(), &reduced_tail).unwrap();
    reduced.write_values(&mut out).unwrap();
    assert_raw(&out[..2], sentinel.reduce().into_loose());
    assert_raw(&out[2..4], PastaField::<M>::ONE.reduce().into_loose());
}

#[test]
fn constant_prefix_storage_and_validation() {
    contracts_field::<PallasBase>();
    contracts_field::<PallasScalar>();
}
