use zakura_udon::{
    fft::{CosetDomain, TableRequirements, Tables, TablesMut, Transform},
    field::{PastaField, PrimeModulus},
};

/// Owns immutable tables independently of reusable polynomial storage.
///
/// Plans borrow the tables after preparation, so neither owner needs to store
/// references into itself.
pub struct OwnedTables<M: PrimeModulus> {
    domain: CosetDomain<M>,
    forward: Vec<PastaField<M>>,
    inverse: Vec<PastaField<M>>,
    finish: Vec<PastaField<M>>,
    scales: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> OwnedTables<M> {
    pub fn new(domain: CosetDomain<M>) -> Self {
        let r = TableRequirements::for_domain(domain);
        let mut result = Self {
            domain,
            forward: vec![PastaField::ZERO; r.twiddles],
            inverse: vec![PastaField::ZERO; r.twiddles],
            finish: vec![PastaField::ZERO; r.twiddles],
            scales: vec![PastaField::ZERO; r.inverse_scales],
        };
        TablesMut {
            forward: Some(&mut result.forward),
            inverse: Some(&mut result.inverse),
            inverse_finish: Some(&mut result.finish),
            inverse_scales: Some(&mut result.scales),
        }
        .prepare(domain);
        result
    }

    pub fn plan(&self) -> Transform<'_, M> {
        Tables {
            forward: Some(&self.forward),
            inverse: Some(&self.inverse),
            inverse_finish: Some(&self.finish),
            inverse_scales: Some(&self.scales),
        }
        // The owner prepared these entries and exposes no mutation.
        .bind_trusted(self.domain)
    }

    pub fn capacity_bytes(&self) -> usize {
        (self.forward.capacity()
            + self.inverse.capacity()
            + self.finish.capacity()
            + self.scales.capacity())
            * size_of::<PastaField<M>>()
    }
}

pub struct FftWorkspace<M: PrimeModulus> {
    pub coefficients: Vec<PastaField<M>>,
    pub output: Vec<PastaField<M>>,
    pub product: Vec<PastaField<M>>,
    pub classes: Vec<PastaField<M>>,
    pub scratch: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> FftWorkspace<M> {
    pub fn new(coefficients: usize, output: usize, classes: usize, scratch: usize) -> Self {
        Self {
            coefficients: vec![PastaField::ZERO; coefficients],
            output: vec![PastaField::ZERO; output],
            product: vec![PastaField::ZERO; output],
            classes: vec![PastaField::ZERO; classes],
            scratch: vec![PastaField::ZERO; scratch],
        }
    }

    pub fn capacities(&self) -> [usize; 5] {
        [
            self.coefficients.capacity(),
            self.output.capacity(),
            self.product.capacity(),
            self.classes.capacity(),
            self.scratch.capacity(),
        ]
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacities().iter().sum::<usize>() * size_of::<PastaField<M>>()
    }
}

/// Collects disjoint residues before allowing class interpolation.
///
/// Tracks up to 64 residue identities to reject duplicate submissions and
/// consumption of a partially filled class. Producer completion belongs to the
/// caller.
pub struct ClassBuilder<'a, M: PrimeModulus> {
    values: &'a mut [PastaField<M>],
    residues: usize,
    completed: u64,
}

impl<'a, M: PrimeModulus> ClassBuilder<'a, M> {
    pub fn new(plan: Transform<'a, M>, buffer: &'a mut [PastaField<M>], residues: usize) -> Self {
        assert!(residues.is_power_of_two() && residues <= 64 && residues <= buffer.len());
        assert_eq!(plan.domain().size(), buffer.len());
        Self {
            values: buffer,
            residues,
            completed: 0,
        }
    }

    pub fn submit(&mut self, residue: usize, values: &[PastaField<M>]) -> Result<(), &'static str> {
        if residue >= self.residues || values.len() != self.values.len() / self.residues {
            return Err("invalid producer range");
        }
        let bit = 1u64 << residue;
        if self.completed & bit != 0 {
            return Err("duplicate producer range");
        }
        let bits = self.values.len().ilog2();
        for (i, value) in values.iter().enumerate() {
            let natural = residue + i * self.residues;
            let row = natural.reverse_bits().wrapping_shr(usize::BITS - bits);
            self.values[row] = *value;
        }
        self.completed |= bit;
        Ok(())
    }

    pub fn finish(self) -> Result<&'a mut [PastaField<M>], &'static str> {
        if self.completed.count_ones() as usize != self.residues {
            return Err("unfinished producers");
        }
        Ok(self.values)
    }
}
