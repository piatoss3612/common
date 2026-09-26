//! Consumer adapter operators and borrowed views of native Pasta field storage.

use crate::field::pasta::test_support::samples;
use crate::field::{Field, FieldAdapter, PallasBase, PallasScalar, PastaField, PrimeModulus};

#[test]
#[allow(
    clippy::op_ref,
    reason = "Every owned and borrowed operator form is part of the consumer API."
)]
fn operators_preserve_native_arithmetic() {
    fn check<M: PrimeModulus>() {
        for (a, _) in samples::<M>(8) {
            for (b, _) in samples::<M>(8) {
                let (x, y) = (FieldAdapter::new(a), FieldAdapter::new(b));
                macro_rules! binary {
                    ($op:tt, $assign:tt, $method:ident) => {
                        let expected = FieldAdapter::new(a.$method(&b));
                        assert_eq!(x $op y, expected);
                        assert_eq!(x $op &y, expected);
                        assert_eq!(&x $op y, expected);
                        assert_eq!(&x $op &y, expected);
                        let mut actual = x;
                        actual $assign y;
                        assert_eq!(actual, expected);
                        actual = x;
                        actual $assign &y;
                        assert_eq!(actual, expected);
                    }
                }
                binary!(+, +=, add);
                binary!(-, -=, sub);
                binary!(*, *=, mul);
                assert_eq!((-x).into_inner().reduce(), a.neg().reduce());
                assert_eq!((-&x).into_inner().reduce(), a.neg().reduce());
                assert_eq!(
                    [x, y].iter().sum::<FieldAdapter<M>>().into_inner().reduce(),
                    a.add(&b).reduce()
                );
                assert_eq!(
                    [x, y]
                        .into_iter()
                        .sum::<FieldAdapter<M>>()
                        .into_inner()
                        .reduce(),
                    a.add(&b).reduce()
                );
            }
        }
        assert_eq!(
            core::iter::empty::<FieldAdapter<M>>().sum::<FieldAdapter<M>>(),
            FieldAdapter::ZERO
        );
        assert_eq!(
            core::iter::empty::<&FieldAdapter<M>>().product::<FieldAdapter<M>>(),
            FieldAdapter::ONE
        );
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn equality_compares_field_values_across_loose_representatives() {
    fn check<M: PrimeModulus>() {
        for (value, _) in samples::<M>(16) {
            let reduced = value.reduce();
            let (limbs, carry) =
                crate::field::word::add_limbs(&reduced.montgomery_limbs(), &M::MODULUS);
            assert_eq!(carry, 0);
            let redundant = PastaField::<M>::from_montgomery_limbs(limbs);
            assert_ne!(reduced.montgomery_limbs(), redundant.montgomery_limbs());
            assert_eq!(reduced, redundant.reduce());
            assert_eq!(
                FieldAdapter::new(reduced.into_loose()),
                FieldAdapter::new(redundant)
            );
            assert_ne!(
                FieldAdapter::new(redundant),
                FieldAdapter::new(reduced.add(&PastaField::<M>::ONE))
            );
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn transparent_views_preserve_field_storage() {
    fn check<M: PrimeModulus>() {
        let mut native = [
            PastaField::ZERO,
            PastaField::from_u64(7),
            PastaField::from_montgomery_limbs(M::MODULUS),
        ];
        for range in [0..0, 1..1, 0..3, 1..3] {
            let slice = &native[range];
            let wrapped = FieldAdapter::from_slice(slice);
            assert_eq!(wrapped.as_ptr().cast::<PastaField<M>>(), slice.as_ptr());
            assert!(core::ptr::eq(FieldAdapter::as_slice(wrapped), slice));
            assert_eq!(bento::bytes_of_slice(wrapped), bento::bytes_of_slice(slice));
        }
        assert!(core::ptr::eq(
            FieldAdapter::from_ref(&native[2]).as_inner(),
            &native[2]
        ));
        let original = native;
        let wrapped = FieldAdapter::from_slice_mut(&mut native);
        wrapped[0] += FieldAdapter::from(3);
        *wrapped[1].as_inner_mut() = PastaField::from_u64(11);
        FieldAdapter::as_slice_mut(wrapped)[2] = PastaField::ONE;
        assert_eq!(
            native.map(PastaField::reduce),
            [3, 11, 1].map(PastaField::from_u64)
        );
        let mut owned = original.map(FieldAdapter::new);
        let slice = FieldAdapter::as_slice_mut(&mut owned);
        slice[0] = PastaField::ONE;
        FieldAdapter::from_slice_mut(slice)[1] += FieldAdapter::ONE;
        assert_eq!(
            owned.map(|value| value.into_inner().reduce()),
            [1, 8, 0].map(PastaField::from_u64)
        );
        let rows = [original, native];
        let borrowed = FieldAdapter::from_rows(&rows);
        assert_eq!(
            borrowed.as_ptr().cast::<[PastaField<M>; 3]>(),
            rows.as_ptr()
        );
        assert_eq!(
            bento::bytes_of_slice(FieldAdapter::as_slice(&borrowed[1])),
            bento::bytes_of_slice(&native)
        );
        assert!(FieldAdapter::<M>::from_rows::<0>(&[[]])[0].is_empty());
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}
