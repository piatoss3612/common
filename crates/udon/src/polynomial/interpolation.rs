use crate::field::{Loose, PastaField, PrimeModulus, ProductSum, ReductionState, batch_invert};

#[cfg(test)]
#[path = "tests/interpolation.rs"]
mod tests;

/// Invalid interpolation data or insufficient caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterpolationError {
    /// Two points represent the same field value.
    DuplicatePoints {
        /// Earlier point index.
        first: usize,
        /// Later point index.
        second: usize,
    },
    /// Weight or denominator storage cannot hold one entry per point.
    WeightsTooShort {
        /// Required number of entries.
        required: usize,
        /// Supplied number of entries.
        actual: usize,
    },
    /// Values must correspond exactly to the ordered points.
    ValueCount {
        /// Number of points.
        expected: usize,
        /// Number of supplied values.
        actual: usize,
    },
    /// Coefficient output cannot hold one entry per point.
    OutputTooShort {
        /// Required number of coefficients.
        required: usize,
        /// Supplied number of coefficients.
        actual: usize,
    },
    /// Operation scratch cannot hold one entry per point.
    ScratchTooShort {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        actual: usize,
    },
}

impl core::fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicatePoints { first, second } => {
                write!(f, "interpolation points {first} and {second} are equal")
            }
            Self::WeightsTooShort { required, actual } => write!(
                f,
                "interpolation requires {required} weights, received {actual}"
            ),
            Self::ValueCount { expected, actual } => write!(
                f,
                "interpolation requires exactly {expected} values, received {actual}"
            ),
            Self::OutputTooShort { required, actual } => write!(
                f,
                "interpolation requires {required} coefficients, received {actual}"
            ),
            Self::ScratchTooShort { required, actual } => write!(
                f,
                "interpolation requires {required} scratch fields, received {actual}"
            ),
        }
    }
}

impl core::error::Error for InterpolationError {}

/// A distinct ordered point set awaiting caller-managed weight inversion.
///
/// Created by [`InterpolationPlan::prepare_denominators`]. This descriptor
/// borrows only the points, so denominator buffers can share a call to
/// [`batch_invert_groups`](crate::field::batch_invert_groups) with other
/// operations. Keep it paired with its prepared entries. [`Self::complete`]
/// checks length, not mathematical contents.
#[derive(Clone, Copy)]
pub struct InterpolationPreparation<'a, M: PrimeModulus, S: ReductionState = Loose> {
    points: &'a [PastaField<M, S>],
}

impl<M: PrimeModulus, S: ReductionState> core::fmt::Debug for InterpolationPreparation<'_, M, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InterpolationPreparation")
            .field("points", &self.points)
            .finish()
    }
}

impl<'a, M: PrimeModulus, S: ReductionState> InterpolationPreparation<'a, M, S> {
    /// Borrows the unscaled inverses of this preparation's denominators.
    ///
    /// The first `points.len()` weights must be the inverses of the prepared
    /// entries in their original order. This content requirement is the caller's
    /// responsibility; incorrect entries give incorrect results.
    /// Short storage is rejected; extra entries are ignored. No data is mutated,
    /// no arithmetic is performed, and no allocation or scratch is needed.
    pub fn complete(
        self,
        weights: &'a [PastaField<M>],
    ) -> Result<InterpolationPlan<'a, M, S>, InterpolationError> {
        InterpolationPlan::bind(self.points, weights)
    }
}

/// Borrows distinct points and barycentric weights for small-set interpolation.
///
/// For ordered points `x_i`, the weights are
/// `w_i = 1 / product(x_i - x_j, j != i)`. A matching slice of `n` values defines
/// the unique polynomial of degree below `n` taking those values at the points.
/// Empty inputs represent zero; one point defines a constant. Points and values
/// may be reduced or loose. Preparation rejects duplicate field values even
/// when their raw representations differ.
///
/// [`Self::prepare`] fills caller-owned weight storage; [`Self::bind`] borrows
/// existing weights, trusting their contents and the distinctness of the points.
/// The same plan serves different value slices and query points. A plan can also
/// be used once and dropped, allowing its weight storage to be reused as scratch.
///
/// Preparation and coefficient recovery take quadratic work in the point count;
/// evaluation takes linear work without inversion. All operations are serial,
/// variable-time and allocation-free, with constant auxiliary space beyond the
/// supplied buffers. Weight storage and operation scratch each use `n` fields;
/// the handle stores two slice descriptors. No sizing sum can overflow. Large
/// power-of-two domains can instead use the transforms in [`crate::fft`].
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::{evaluate, InterpolationPlan}};
/// let points = [0, 2, 5].map(<Fp>::from_u64);
/// let values = [3, 11, 38].map(<Fp>::from_u64); // 3 + 2*X + X^2
/// let mut weights = [Fp::ZERO; 3];
/// let mut scratch = [Fp::ZERO; 3];
/// let plan = InterpolationPlan::prepare(&points, &mut weights, &mut scratch)?;
/// let mut coefficients = [Fp::ZERO; 3];
/// plan.interpolate(&values, &mut coefficients, &mut scratch)?;
/// assert_eq!(coefficients.map(Fp::reduce), [3, 2, 1].map(|n| <Fp>::from_u64(n).reduce()));
/// let query = <Fp>::from_u64(7);
/// assert_eq!(plan.evaluate(&values, &query, &mut scratch)?.reduce(),
///            evaluate(&coefficients, &query).reduce());
/// # Ok::<(), zakura_udon::polynomial::InterpolationError>(())
/// ```
#[derive(Clone, Copy)]
pub struct InterpolationPlan<'a, M: PrimeModulus, S: ReductionState = Loose> {
    points: &'a [PastaField<M, S>],
    weights: &'a [PastaField<M>],
}

impl<M: PrimeModulus, S: ReductionState> core::fmt::Debug for InterpolationPlan<'_, M, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InterpolationPlan")
            .field("points", &self.points)
            .field("weights", &self.weights)
            .finish()
    }
}

impl<'a, M: PrimeModulus, S: ReductionState> InterpolationPlan<'a, M, S> {
    /// Prepares weights for distinct `points`, borrowing their ordered prefix.
    ///
    /// Weight storage must hold at least `points.len()` entries. Duplicate
    /// points and short storage are rejected before any weights or scratch are
    /// written. Unused weight and scratch tails beyond that count are untouched.
    /// Initial buffer contents are ignored; exclusive borrows keep writable
    /// storage disjoint from points and from each other.
    ///
    /// Any inversion scratch length is accepted. [`batch_invert`] uses one
    /// inversion with at least `n` scratch fields for `n >= 2`, bounded batches
    /// with less scratch, or individual inverses with empty scratch. Empty and
    /// singleton preparations need no inversion or scratch writes.
    pub fn prepare(
        points: &'a [PastaField<M, S>],
        weights: &'a mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<Self, InterpolationError> {
        let preparation = Self::prepare_denominators(points, weights)?;
        if points.len() > 1 {
            batch_invert(&mut weights[..points.len()], scratch);
        }
        preparation.complete(weights)
    }

    /// Writes `product(x_i - x_j, j != i)` for a shared inversion batch.
    ///
    /// Storage, duplicate rejection, and untouched-tail rules match
    /// [`Self::prepare`]. All checks precede writes. Empty points write nothing;
    /// a singleton writes one. Preparation takes quadratic work without scratch
    /// or inversion. Invert the written prefix without scaling, then give its
    /// entries to [`InterpolationPreparation::complete`].
    ///
    /// ```
    /// use zakura_udon::{field::{Fp, batch_invert_groups}, polynomial::InterpolationPlan};
    /// let a = [0, 1].map(<Fp>::from_u64);
    /// let b = [2, 4, 7].map(<Fp>::from_u64);
    /// let mut wa = [Fp::ZERO; 2];
    /// let mut wb = [Fp::ZERO; 3];
    /// let pa = InterpolationPlan::prepare_denominators(&a, &mut wa)?;
    /// let pb = InterpolationPlan::prepare_denominators(&b, &mut wb)?;
    /// batch_invert_groups(&mut [&mut wa[..], &mut wb[..]], &mut [Fp::ZERO; 5]);
    /// let a_plan = pa.complete(&wa)?;
    /// let b_plan = pb.complete(&wb)?;
    /// let query = <Fp>::from_u64(9);
    /// assert_eq!(a_plan.evaluate(&a, &query, &mut [Fp::ZERO; 2])?.reduce(), query.reduce());
    /// assert_eq!(b_plan.evaluate(&b, &query, &mut [Fp::ZERO; 3])?.reduce(), query.reduce());
    /// # Ok::<(), zakura_udon::polynomial::InterpolationError>(())
    /// ```
    pub fn prepare_denominators(
        points: &'a [PastaField<M, S>],
        denominators: &mut [PastaField<M>],
    ) -> Result<InterpolationPreparation<'a, M, S>, InterpolationError> {
        check_weights(points.len(), denominators.len())?;
        check_points(points)?;
        let denominators = &mut denominators[..points.len()];
        denominators.fill(PastaField::ONE);
        for (i, point) in points.iter().enumerate() {
            for j in i + 1..points.len() {
                let difference = point.sub(&points[j]);
                denominators[i] = denominators[i].mul(&difference);
                denominators[j] = denominators[j].mul(&difference.neg());
            }
        }
        Ok(InterpolationPreparation { points })
    }

    /// Borrows distinct points and their barycentric weights with constant work.
    ///
    /// The caller must supply distinct field values and, for each point `x_i`,
    /// the weight `1 / product(x_i - x_j, j != i)`. These mathematical contents
    /// are not checked; incorrect inputs give incorrect results. Loose
    /// representations are accepted. Short weight storage is rejected, and
    /// extra entries are ignored. Binding performs no arithmetic and needs
    /// no scratch.
    pub fn bind(
        points: &'a [PastaField<M, S>],
        weights: &'a [PastaField<M>],
    ) -> Result<Self, InterpolationError> {
        check_weights(points.len(), weights.len())?;
        Ok(Self {
            points,
            weights: &weights[..points.len()],
        })
    }

    /// Distinct interpolation points in their supplied order.
    pub const fn points(&self) -> &'a [PastaField<M, S>] {
        self.points
    }

    /// Barycentric weights in the same order as [`Self::points`].
    pub const fn weights(&self) -> &'a [PastaField<M>] {
        self.weights
    }

    /// Writes ascending coefficients of the interpolating polynomial.
    ///
    /// Values must correspond exactly to the plan's ordered points. Output and
    /// scratch each need at least `n = points.len()` fields, including for a
    /// singleton. Only those prefixes are written; unused tails are unchanged.
    /// All dimensions are checked before writing either buffer. Empty input
    /// writes nothing and returns zero. Otherwise returns `n`, without trimming
    /// leading zero coefficients. A singleton copies its value.
    ///
    /// Scratch contents are ignored. Rust's exclusive borrows keep writable
    /// buffers disjoint from each other, values and retained data. Execution
    /// takes quadratic work without inversion or additional storage.
    pub fn interpolate(
        &self,
        values: &[PastaField<M, impl ReductionState>],
        output: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<usize, InterpolationError> {
        self.check_values(values.len())?;
        let n = self.points.len();
        if output.len() < n {
            return Err(InterpolationError::OutputTooShort {
                required: n,
                actual: output.len(),
            });
        }
        self.check_scratch(scratch.len())?;
        let Some((last, rest)) = self.points.split_last() else {
            return Ok(0);
        };
        let output = &mut output[..n];
        if n == 1 {
            output[0] = values[0].into_loose();
            return Ok(n);
        }
        let vanishing = &mut scratch[..n];
        // Store the full vanishing polynomial with its leading one implicit.
        // Preparing all but the last factor fits in n fields, then multiplication
        // by that final factor discards only the known leading coefficient.
        super::vanishing_polynomial(rest, vanishing).unwrap();
        let negative_last = last.neg();
        for j in (1..n).rev() {
            vanishing[j] = vanishing[j].mul_add(&negative_last, &vanishing[j - 1]);
        }
        vanishing[0] = vanishing[0].mul(&negative_last);
        output.fill(PastaField::ZERO);
        for ((point, weight), value) in self.points.iter().zip(self.weights).zip(values) {
            let scale = weight.mul(value);
            if scale.is_zero() {
                continue;
            }
            let mut quotient = PastaField::ONE;
            for j in (0..n).rev() {
                output[j] = scale.mul_add(&quotient, &output[j]);
                if j != 0 {
                    quotient = quotient.mul_add(point, &vanishing[j]);
                }
            }
        }
        Ok(n)
    }

    /// Evaluates the interpolant without constructing its coefficients.
    ///
    /// Values must correspond exactly to the plan's ordered points. Scratch
    /// needs at least `points.len()` fields, even for exact nodes and singletons.
    /// Length errors leave scratch unchanged. Only the required prefix may be
    /// overwritten, and its initial contents are ignored. Values can share
    /// immutable point or weight storage; scratch must be disjoint.
    ///
    /// Empty input returns zero. Exact node hits return the matching value;
    /// singletons return their value at every query. These cases leave scratch
    /// unchanged. Other queries sum `w_i * values[i] * product(point - x_j,
    /// j != i)` using prefix and suffix products, taking linear work with no
    /// inversion, allocation or additional scratch.
    pub fn evaluate(
        &self,
        values: &[PastaField<M, impl ReductionState>],
        point: &PastaField<M, impl ReductionState>,
        scratch: &mut [PastaField<M>],
    ) -> Result<PastaField<M>, InterpolationError> {
        self.check_values(values.len())?;
        self.check_scratch(scratch.len())?;
        if values.len() == 1 {
            return Ok(values[0].into_loose());
        }
        for (node, value) in self.points.iter().zip(values) {
            if point.sub(node).is_zero() {
                return Ok(value.into_loose());
            }
        }
        let prefixes = &mut scratch[..self.points.len()];
        let mut prefix = PastaField::ONE;
        for (i, (node, slot)) in self.points.iter().zip(prefixes.iter_mut()).enumerate() {
            *slot = prefix;
            if i + 1 < self.points.len() {
                prefix = prefix.mul(&point.sub(node));
            }
        }
        let mut suffix = PastaField::ONE;
        let mut sum = ProductSum::new();
        for i in (0..self.points.len()).rev() {
            let basis = prefixes[i].mul(&suffix).mul(&self.weights[i]);
            sum.add_product(&basis, &values[i]);
            if i != 0 {
                suffix = suffix.mul(&point.sub(&self.points[i]));
            }
        }
        Ok(sum.finish())
    }

    fn check_values(&self, actual: usize) -> Result<(), InterpolationError> {
        if actual != self.points.len() {
            return Err(InterpolationError::ValueCount {
                expected: self.points.len(),
                actual,
            });
        }
        Ok(())
    }

    fn check_scratch(&self, actual: usize) -> Result<(), InterpolationError> {
        if actual < self.points.len() {
            return Err(InterpolationError::ScratchTooShort {
                required: self.points.len(),
                actual,
            });
        }
        Ok(())
    }
}

fn check_weights(required: usize, actual: usize) -> Result<(), InterpolationError> {
    if actual < required {
        return Err(InterpolationError::WeightsTooShort { required, actual });
    }
    Ok(())
}

fn check_points<M: PrimeModulus, S: ReductionState>(
    points: &[PastaField<M, S>],
) -> Result<(), InterpolationError> {
    for (second, point) in points.iter().enumerate() {
        for (first, earlier) in points[..second].iter().enumerate() {
            if point.sub(earlier).is_zero() {
                return Err(InterpolationError::DuplicatePoints { first, second });
            }
        }
    }
    Ok(())
}
