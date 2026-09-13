//! Shared-inversion compact tables and same-scalar ladders.

use core::marker::PhantomData;

use super::{
    AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, EisensteinScalar,
    EisensteinTable, PastaCurve, ProjectivePoint, batch::invert_nonzero, check_length,
    check_scratch, checked_count, eisenstein,
};
use crate::{
    exec::{Executor, TaskBudget},
    field::PastaField,
};

const TABLE_AFFINE_MIN: usize = 8;
const LADDER_AFFINE_MIN: usize = 64;

/// Borrowed compact tables, stored as consecutive groups of eight entries.
///
/// Each group has [`EisensteinTable`]'s order, with its base in entry zero.
/// Batch preparation and same-scalar multiplication can share inversions across
/// bases. Callers supply storage and scoped execution; all operations are
/// variable-time and require no allocation. Both [`CurveTableEntry`]
/// representations are supported, and empty batches are accepted.
///
/// ```
/// use zakura_udon::{
///     curve::{
///         CurveTableRequirements, EisensteinScalar, EisensteinTableBatch,
///         Pallas, PallasAffine, PallasProjective,
///     },
///     exec::{SerialExecutor, TaskBudget},
///     field::{Fp, Fq},
/// };
///
/// const N: usize = 64;
/// const R: CurveTableRequirements = match EisensteinTableBatch::<Pallas>::requirements(N) {
///     Ok(r) => r,
///     Err(_) => panic!("batch is too large"),
/// };
/// const MUL: usize = match EisensteinTableBatch::<Pallas>::multiplication_scratch(N) {
///     Ok(n) => n,
///     Err(_) => panic!("batch is too large"),
/// };
/// const FIELD: usize = if MUL > R.field_scratch {
///     MUL
/// } else {
///     R.field_scratch
/// };
/// let bases = [PallasAffine::GENERATOR; N];
/// let mut entries = [PallasAffine::GENERATOR; R.table_entries];
/// let mut projective = [PallasProjective::IDENTITY; R.projective_scratch];
/// let mut field = [Fp::ZERO; FIELD];
/// let tables = EisensteinTableBatch::prepare(
///     &bases,
///     &mut entries,
///     &mut projective,
///     &mut field,
///     TaskBudget::SERIAL,
///     &SerialExecutor,
/// )?;
/// let scalar = EisensteinScalar::new(&Fq::from_u64(42));
/// let mut output = [PallasProjective::IDENTITY; N];
/// tables.mul_prepared(
///     &scalar,
///     &mut output,
///     &mut field,
///     TaskBudget::SERIAL,
///     &SerialExecutor,
/// )?;
/// for (i, product) in output.iter().enumerate() {
///     assert_eq!(*product, tables.get(i).unwrap().mul_prepared(&scalar));
/// }
/// # Ok::<(), zakura_udon::curve::CurveError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct EisensteinTableBatch<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    entries: &'a [E],
    marker: PhantomData<C>,
}

impl<'a, C: PastaCurve, E: CurveTableEntry<C>> EisensteinTableBatch<'a, C, E> {
    /// Returns exact entry and minimum scratch counts for batch preparation.
    ///
    /// Counts depend on the number of `bases`, independently of the task budget.
    /// Returns [`CurveError::SizeOverflow`] if a buffer exceeds slice limits.
    pub const fn requirements(bases: usize) -> Result<CurveTableRequirements, CurveError> {
        let table_entries = match checked_count::<E>(bases, 8) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        let projective_scratch = if bases < TABLE_AFFINE_MIN {
            match checked_count::<ProjectivePoint<C>>(bases, 8) {
                Ok(n) => n,
                Err(e) => return Err(e),
            }
        } else {
            0
        };
        let field_scratch = match checked_count::<PastaField<C::Base>>(
            bases,
            if bases < TABLE_AFFINE_MIN { 8 } else { 4 },
        ) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };
        Ok(CurveTableRequirements {
            table_entries,
            projective_scratch,
            field_scratch,
        })
    }

    /// Prepares one table per base in caller-selected entry storage.
    ///
    /// Size buffers with [`Self::requirements`] for `bases.len()`. `entries`
    /// must have the exact reported length; scratch may be larger, with unused
    /// tails left untouched. Initial contents do not matter. The returned view
    /// borrows only `entries`, leaving scratch available for other work.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::InvalidBase`] for invalid coordinates or caches,
    /// [`CurveError::LengthMismatch`] for an incorrect entry count,
    /// [`CurveError::ScratchTooSmall`] for insufficient scratch, or
    /// [`CurveError::SizeOverflow`] if sizing exceeds slice limits. All checks
    /// precede writes, so returned errors leave every buffer unchanged.
    ///
    /// # Panics
    ///
    /// An executor panic may leave buffers partially written. All scoped jobs
    /// finish or unwind before it propagates, as required by [`Executor`].
    pub fn prepare<B: CurveTableEntry<C>, X: Executor>(
        bases: &[B],
        entries: &'a mut [E],
        projective: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        let r = Self::requirements(bases.len())?;
        check_length("entries", r.table_entries, entries.len())?;
        check_scratch("projective", r.projective_scratch, projective.len())?;
        check_scratch("field", r.field_scratch, field.len())?;
        for base in bases {
            let p = base.affine();
            if AffinePoint::<C>::from_xy(p.x, p.y).is_none() || !base.valid_cache() {
                return Err(CurveError::InvalidBase);
            }
        }
        prepare_inner(
            bases,
            entries,
            &mut projective[..r.projective_scratch],
            &mut field[..r.field_scratch],
            budget.get(),
            executor,
        );
        Ok(Self {
            entries,
            marker: PhantomData,
        })
    }

    /// Binds stored tables after validating every multiple and cached coordinate.
    ///
    /// Returns [`CurveError::InvalidTableLayout`] for an incomplete group,
    /// [`CurveError::InvalidBase`] for invalid affine coordinates in entry zero,
    /// or [`CurveError::InvalidTable`] for an incorrect multiple or cache,
    /// including entry zero's cache. Uses no scratch.
    pub fn bind(entries: &'a [E]) -> Result<Self, CurveError> {
        let batch = Self::bind_trusted(entries)?;
        batch.validate()?;
        Ok(batch)
    }

    /// Binds tables whose multiples and caches the owner has established.
    ///
    /// Checks the flat layout and entry zero's affine coordinates in each group,
    /// returning [`CurveError::InvalidTableLayout`] or [`CurveError::InvalidBase`]
    /// as described by [`Self::bind`]. All multiples and cached coordinates,
    /// including entry zero's cache, must already satisfy [`EisensteinTable`]'s
    /// mathematical contract. Violations remain memory-safe but may make
    /// arithmetic panic or return incorrect results.
    pub fn bind_trusted(entries: &'a [E]) -> Result<Self, CurveError> {
        if !entries.len().is_multiple_of(8) {
            return Err(CurveError::InvalidTableLayout);
        }
        for group in entries.chunks_exact(8) {
            let p = group[0].affine();
            if AffinePoint::<C>::from_xy(p.x, p.y).is_none() {
                return Err(CurveError::InvalidBase);
            }
        }
        Ok(Self {
            entries,
            marker: PhantomData,
        })
    }

    /// Checks every table's specified multiples and cached coordinates.
    ///
    /// Returns [`CurveError::InvalidTable`] for an invalid entry. Uses no scratch.
    pub fn validate(&self) -> Result<(), CurveError> {
        for i in 0..self.len() {
            self.get(i).unwrap().validate()?;
        }
        Ok(())
    }

    /// Returns the number of bases.
    pub const fn len(&self) -> usize {
        self.entries.len() / 8
    }

    /// Returns whether there are no tables.
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrows the flat table storage.
    pub const fn as_slice(&self) -> &'a [E] {
        self.entries
    }

    /// Borrows an individual table, or returns `None` for an out-of-range index.
    ///
    /// The table inherits this batch's entry contract without revalidation.
    pub fn get(&self, index: usize) -> Option<EisensteinTable<'a, C, E>> {
        if index >= self.len() {
            return None;
        }
        let entries = &self.entries[index * 8..(index + 1) * 8];
        Some(EisensteinTable {
            base: entries[0].affine(),
            entries,
        })
    }

    /// Returns the minimum number of field elements for batch multiplication.
    ///
    /// Pass the number of tables as `bases`. Counts are independent of the scalar
    /// and task budget and apply to both [`Self::mul`] and [`Self::mul_prepared`].
    /// Returns [`CurveError::SizeOverflow`] if the buffer exceeds slice limits.
    pub const fn multiplication_scratch(bases: usize) -> Result<usize, CurveError> {
        checked_count::<PastaField<C::Base>>(bases, if bases < LADDER_AFFINE_MIN { 0 } else { 5 })
    }

    /// Multiplies every base by the same reduced scalar, in table order.
    ///
    /// Equivalent to preparing an [`EisensteinScalar`] and calling
    /// [`Self::mul_prepared`], including its buffer, error, and panic contracts.
    /// The scalar must satisfy [`PastaField`]'s reduced-residue invariant.
    pub fn mul<X: Executor>(
        &self,
        scalar: &PastaField<C::Scalar>,
        output: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<(), CurveError> {
        self.mul_prepared(
            &EisensteinScalar::new(scalar),
            output,
            field,
            budget,
            executor,
        )
    }

    /// Multiplies bases in table order using reusable scalar digits.
    ///
    /// Size field scratch with [`Self::multiplication_scratch`] for `self.len()`.
    /// Initial buffer contents do not matter; scratch beyond the reported count
    /// is untouched. A zero scalar writes identities. Entries must satisfy the
    /// mathematical contract of [`EisensteinTable::mul`].
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::LengthMismatch`] unless `output.len() == self.len()`,
    /// [`CurveError::ScratchTooSmall`] for insufficient field scratch, or
    /// [`CurveError::SizeOverflow`] if sizing exceeds slice limits. All checks
    /// precede writes, so returned errors leave output and scratch unchanged.
    ///
    /// # Panics
    ///
    /// An executor panic may leave output and scratch partially written. All
    /// scoped jobs finish or unwind before it propagates, as required by
    /// [`Executor`]. The buffers can then be reused without clearing them.
    pub fn mul_prepared<X: Executor>(
        &self,
        scalar: &EisensteinScalar<C>,
        output: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<(), CurveError> {
        let n = self.len();
        let required = Self::multiplication_scratch(n)?;
        check_length("output", n, output.len())?;
        check_scratch("field", required, field.len())?;
        let digits = scalar.digits();
        // This modular check is shared by the whole batch, and deliberately
        // absent from scalar preparation used by ordinary compact-table muls.
        let affine = n >= LADDER_AFFINE_MIN && !digits.is_empty() && ladder_safe::<C>(digits);
        multiply_inner(
            self.entries,
            digits,
            output,
            &mut field[..required],
            affine,
            budget.get(),
            executor,
        );
        Ok(())
    }
}

pub(super) fn prepare_inner<
    C: PastaCurve,
    B: CurveTableEntry<C>,
    E: CurveTableEntry<C>,
    X: Executor,
>(
    bases: &[B],
    entries: &mut [E],
    projective: &mut [ProjectivePoint<C>],
    field: &mut [PastaField<C::Base>],
    tasks: usize,
    executor: &X,
) {
    let n = bases.len();
    let tasks = tasks.min((n / TABLE_AFFINE_MIN).max(1));
    if tasks > 1 {
        let left_tasks = tasks / 2;
        let mid = n / tasks * left_tasks;
        let (a, b) = entries.split_at_mut(mid * 8);
        let (fa, fb) = field.split_at_mut(mid * 4);
        executor.join(
            || prepare_inner(&bases[..mid], a, &mut [], fa, left_tasks, executor),
            || prepare_inner(&bases[mid..], b, &mut [], fb, tasks - left_tasks, executor),
        );
    } else if n < TABLE_AFFINE_MIN {
        for (base, points) in bases.iter().zip(projective.chunks_exact_mut(8)) {
            points.copy_from_slice(&eisenstein::representatives(&base.affine().to_projective()));
        }
        eisenstein::normalize(projective, field, entries);
    } else {
        prepare_affine(bases, entries, field);
    }
}

// Every chord below has distinct x coordinates: equality would make one of
// the small nonzero Eisenstein coefficient differences or sums vanish. Their
// norms are far below either prime group order. No exceptional-point branches
// or projective intermediates are needed for these nonidentity inputs.
fn chord<C: PastaCurve>(
    p: AffinePoint<C>,
    q: AffinePoint<C>,
    inverse: PastaField<C::Base>,
) -> AffinePoint<C> {
    let slope = q.y.sub(&p.y).mul(&inverse);
    let x = slope.square().sub(&p.x).sub(&q.x);
    let y = slope.mul(&p.x.sub(&x)).sub(&p.y);
    AffinePoint {
        x,
        y,
        marker: PhantomData,
    }
}

fn prepare_affine<C: PastaCurve, B: CurveTableEntry<C>, E: CurveTableEntry<C>>(
    bases: &[B],
    entries: &mut [E],
    field: &mut [PastaField<C::Base>],
) {
    let n = bases.len();
    let (denom, prefix) = field.split_at_mut(2 * n);
    for (i, base) in bases.iter().enumerate() {
        let p = base.affine();
        entries[8 * i] = E::from_affine(&p);
        denom[i] = p.endomorphism().x.sub(&p.x);
    }
    invert_nonzero(&mut denom[..n], prefix);
    for (i, group) in entries.chunks_exact_mut(8).enumerate() {
        let p = group[0].affine();
        let d = chord(p, p.endomorphism().neg(), denom[i]);
        group[1] = E::from_affine(&d);
        denom[i] = d.endomorphism().x.sub(&d.x);
    }
    invert_nonzero(&mut denom[..n], prefix);
    for (i, group) in entries.chunks_exact_mut(8).enumerate() {
        let d = group[1].affine();
        let b = chord(d, d.endomorphism().neg(), denom[i]);
        let minus_three = b.endomorphism().endomorphism();
        group[4] = E::from_affine(&minus_three.neg());
    }
    for (i, group) in entries.chunks_exact(8).enumerate() {
        let minus_three = group[4].affine().neg();
        let phi = group[0].rotated(1);
        denom[2 * i] = minus_three.x.sub(&phi.x);
        denom[2 * i + 1] = minus_three.endomorphism().endomorphism().x.sub(&phi.x);
    }
    // Each +/- pair shares a chord denominator, so four additions per base
    // need just two inverse entries and one inversion phase.
    invert_nonzero(denom, prefix);
    for (i, group) in entries.chunks_exact_mut(8).enumerate() {
        let phi = group[0].rotated(1);
        let minus_three = group[4].affine().neg();
        let b_phi = minus_three.endomorphism().endomorphism();
        group[5] = E::from_affine(&chord(phi, minus_three, denom[2 * i]).neg());
        group[3] = E::from_affine(
            &chord(phi, minus_three.neg(), denom[2 * i])
                .endomorphism()
                .neg(),
        );
        group[2] = E::from_affine(&chord(phi, b_phi.neg(), denom[2 * i + 1]).endomorphism());
        let four_b = chord(phi, b_phi, denom[2 * i + 1]);
        group[6] = E::from_affine(&four_b.endomorphism().endomorphism());
    }
    for (i, group) in entries.chunks_exact(8).enumerate() {
        denom[i] = group[6].rotated(1).x.sub(&group[0].rotated(1).x);
    }
    invert_nonzero(&mut denom[..n], prefix);
    for (i, group) in entries.chunks_exact_mut(8).enumerate() {
        let p = chord(group[0].rotated(1), group[6].rotated(1), denom[i]);
        group[7] = E::from_affine(&p.endomorphism().endomorphism());
    }
}

fn digit_scalar<C: PastaCurve>(code: u8) -> PastaField<C::Scalar> {
    let value = usize::from(code - 1);
    let (mut a, mut b) = eisenstein::REPRESENTATIVES[value / 6];
    for _ in 0..(value % 6) / 2 {
        (a, b) = (-b, a - b);
    }
    let signed = |x: i8| {
        let f = PastaField::from_u64(u64::from(x.unsigned_abs()));
        if x < 0 { f.neg() } else { f }
    };
    let d = signed(a).add(&signed(b).mul(&PastaField::zeta()));
    if value & 1 == 1 { d.neg() } else { d }
}

pub(super) fn ladder_safe<C: PastaCurve>(digits: &[u8]) -> bool {
    let Some((&top, rest)) = digits.split_last() else {
        return false;
    };
    let mut s = digit_scalar::<C>(top);
    // All nonidentity bases have the same prime order. Checking the schedule
    // in the scalar field is therefore exact for every base in the batch,
    // including schedules whose intermediate integer coefficients wrap.
    for &code in rest.iter().rev() {
        let twice = s.double();
        if code == 0 {
            s = twice;
        } else {
            let d = digit_scalar::<C>(code);
            if d == s || d == twice.neg() {
                return false;
            }
            s = twice.add(&d);
        }
    }
    true
}

fn multiply_inner<C: PastaCurve, E: CurveTableEntry<C>, X: Executor>(
    entries: &[E],
    digits: &[u8],
    output: &mut [ProjectivePoint<C>],
    field: &mut [PastaField<C::Base>],
    affine: bool,
    tasks: usize,
    executor: &X,
) {
    let n = output.len();
    let tasks = tasks.min((n / LADDER_AFFINE_MIN).max(1));
    if tasks > 1 {
        let left_tasks = tasks / 2;
        let mid = n / tasks * left_tasks;
        let (a, b) = output.split_at_mut(mid);
        let (fa, fb) = field.split_at_mut(mid * 5);
        executor.join(
            || {
                multiply_inner(
                    &entries[..mid * 8],
                    digits,
                    a,
                    fa,
                    affine,
                    left_tasks,
                    executor,
                )
            },
            || {
                multiply_inner(
                    &entries[mid * 8..],
                    digits,
                    b,
                    fb,
                    affine,
                    tasks - left_tasks,
                    executor,
                )
            },
        );
    } else if affine {
        affine_ladder(entries, digits, output, field);
    } else {
        for (group, result) in entries.chunks_exact(8).zip(output) {
            *result = eisenstein::multiply(group, digits);
        }
    }
}

fn affine_ladder<C: PastaCurve, E: CurveTableEntry<C>>(
    entries: &[E],
    digits: &[u8],
    output: &mut [ProjectivePoint<C>],
    field: &mut [PastaField<C::Base>],
) {
    let n = output.len();
    let (denom, rest) = field.split_at_mut(n);
    let (prefix, rest) = rest.split_at_mut(n);
    let (hs, rest) = rest.split_at_mut(n);
    let (rs, h2s) = rest.split_at_mut(n);
    let (&top, digits) = digits.split_last().unwrap();
    for (group, result) in entries.chunks_exact(8).zip(output.iter_mut()) {
        *result = eisenstein::digit_point(group, top).to_projective();
    }
    for &code in digits.iter().rev() {
        if code == 0 {
            for (d, p) in denom.iter_mut().zip(output.iter()) {
                *d = p.y.double();
            }
            invert_nonzero(denom, prefix);
            for (p, d) in output.iter_mut().zip(denom.iter()) {
                let xx = p.x.square();
                let slope = xx.double().add(&xx).mul(d);
                let x = slope.square().sub(&p.x.double());
                p.y = slope.mul(&p.x.sub(&x)).sub(&p.y);
                p.x = x;
            }
        } else {
            // Fuse 2P + D for P=(x,y), D=(u,v). Put h=u-x, r=v-y and
            // den=h²(2x+u)-r². With a=y/den, b=a*h and c=b*h²,
            // lambda=c-r gives x'=u+4b*lambda and
            // y'=-y-(1+4a*lambda)(lambda+c). These are the two chord
            // additions with their intermediate denominator eliminated.
            // Only den needs inversion; h=0 is allowed when D=-P.
            // den vanishes for D=P or D=-2P, ruled out by ladder_safe.
            for (i, (group, p)) in entries.chunks_exact(8).zip(output.iter_mut()).enumerate() {
                let d = eisenstein::digit_point(group, code);
                hs[i] = d.x.sub(&p.x);
                rs[i] = d.y.sub(&p.y);
                h2s[i] = hs[i].square();
                denom[i] = h2s[i].mul(&p.x.double().add(&d.x)).sub(&rs[i].square());
                // The finish needs the digit x and the original y only.
                p.x = d.x;
            }
            invert_nonzero(denom, prefix);
            for (i, p) in output.iter_mut().enumerate() {
                let a = p.y.mul(&denom[i]);
                let b = a.mul(&hs[i]);
                let c = b.mul(&h2s[i]);
                let lambda = c.sub(&rs[i]);
                p.x = p.x.add(&b.mul(&lambda).double().double());
                p.y = p.y.neg().sub(
                    &PastaField::ONE
                        .add(&a.mul(&lambda).double().double())
                        .mul(&lambda.add(&c)),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::{Pallas, Vesta, parameters::GlvParameters},
        exec::SerialExecutor,
    };
    use std::{vec, vec::Vec};

    fn exceptions<C: PastaCurve>() {
        let lattice = GlvParameters::<C>::BASIS;
        let (wrapped, len) = eisenstein::recode(lattice.a as i128, -(lattice.b as i128));
        let cases: [(&[u8], bool); 4] = [
            (&[1, 1], false),
            (&[2, 1], true),
            (&[2, 0, 1], true),
            (&wrapped[..len], false),
        ];
        let bases: Vec<_> = (1..=35)
            .map(|i| {
                *AffinePoint::<C>::GENERATOR
                    .mul_projective(&PastaField::from_u64(i))
                    .to_point()
                    .as_affine()
                    .unwrap()
            })
            .collect();
        let mut entries = vec![AffinePoint::GENERATOR; 8 * bases.len()];
        let mut field = vec![PastaField::ZERO; 5 * bases.len()];
        EisensteinTableBatch::prepare(
            &bases,
            &mut entries,
            &mut [],
            &mut field,
            TaskBudget::SERIAL,
            &SerialExecutor,
        )
        .unwrap();
        for (digits, safe) in cases {
            assert_eq!(ladder_safe::<C>(digits), safe);
            let mut scalar = PastaField::ZERO;
            for &code in digits.iter().rev() {
                scalar = scalar.double();
                if code != 0 {
                    scalar = scalar.add(&digit_scalar::<C>(code));
                }
            }
            if digits.len() == len {
                assert_eq!(scalar, PastaField::ZERO);
            }
            let mut output = vec![ProjectivePoint::IDENTITY; bases.len()];
            multiply_inner(
                &entries,
                digits,
                &mut output,
                &mut field,
                safe,
                1,
                &SerialExecutor,
            );
            for (base, actual) in bases.iter().zip(output) {
                let expected = crate::curve::scalar::multiply(&scalar, |sum| sum.add_mixed(base));
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn exact_modular_exception_gate_and_projective_fallback() {
        exceptions::<Pallas>();
        exceptions::<Vesta>();
    }
}
