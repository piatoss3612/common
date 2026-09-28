//! Private kernel controls used by selection and arithmetic tests.

use crate::exec::TaskBudget;
use core::num::NonZeroUsize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Accumulation {
    Auto,
    Affine,
    Projective,
    Hybrid,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Algorithm {
    Auto,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Differential tests force this kernel.")
    )]
    Joint,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "Differential tests force this kernel.")
    )]
    Booth {
        width: Option<u32>,
        accumulation: Accumulation,
    },
    StreamingBooth {
        width: Option<u32>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ArithmeticOptions {
    pub(super) max_terms_per_pass: Option<NonZeroUsize>,
    pub(super) chunk_size: Option<NonZeroUsize>,
    pub(super) algorithm: Algorithm,
}
impl ArithmeticOptions {
    pub(super) const DEFAULT: Self = Self {
        max_terms_per_pass: None,
        chunk_size: None,
        algorithm: Algorithm::Auto,
    };
    #[cfg(test)]
    pub(super) const fn with_max_terms_per_pass(mut self, cap: Option<NonZeroUsize>) -> Self {
        self.max_terms_per_pass = cap;
        self
    }
    #[cfg(test)]
    pub(super) const fn with_chunk_size(mut self, terms: NonZeroUsize) -> Self {
        self.chunk_size = Some(terms);
        self
    }
    #[cfg(test)]
    pub(super) const fn with_algorithm(
        mut self,
        algorithm: Algorithm,
    ) -> Result<Self, InvalidWindow> {
        if let Algorithm::Booth {
            width: Some(bits), ..
        }
        | Algorithm::StreamingBooth { width: Some(bits) } = algorithm
            && (bits < 4 || bits > 12)
        {
            return Err(InvalidWindow { bits });
        }
        self.algorithm = algorithm;
        Ok(self)
    }
    pub(super) const fn streaming(self) -> bool {
        matches!(self.algorithm, Algorithm::StreamingBooth { .. })
    }
    pub(super) const fn accumulation(self) -> Accumulation {
        match self.algorithm {
            Algorithm::Booth { accumulation, .. } => accumulation,
            Algorithm::StreamingBooth { .. } => Accumulation::Projective,
            _ => Accumulation::Auto,
        }
    }
    pub(super) const fn chunk_cap(self) -> usize {
        match self.chunk_size {
            Some(cap) => cap.get(),
            None if self.streaming() => 256,
            None => usize::MAX,
        }
    }
}
impl Default for ArithmeticOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchOptions {
    pub(super) arithmetic: ArithmeticOptions,
    pub(super) task_budget: TaskBudget,
    pub(super) memory_limit: Option<usize>,
}
impl BatchOptions {
    pub(super) const fn new(arithmetic: ArithmeticOptions) -> Self {
        Self {
            arithmetic,
            task_budget: TaskBudget::SERIAL,
            memory_limit: None,
        }
    }
    pub(super) const fn with_task_budget(mut self, budget: TaskBudget) -> Self {
        self.task_budget = budget;
        self
    }
    #[cfg(test)]
    pub(super) const fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }
    #[cfg(test)]
    pub(super) const fn arithmetic(&self) -> ArithmeticOptions {
        self.arithmetic
    }
    #[cfg(test)]
    pub(super) const fn task_budget(&self) -> TaskBudget {
        self.task_budget
    }
    #[cfg(test)]
    pub(super) const fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }
}
impl Default for BatchOptions {
    fn default() -> Self {
        Self::new(ArithmeticOptions::DEFAULT)
    }
}
impl From<crate::exec::ExecutionOptions> for BatchOptions {
    fn from(options: crate::exec::ExecutionOptions) -> Self {
        Self {
            arithmetic: ArithmeticOptions::DEFAULT,
            task_budget: options.task_budget(),
            memory_limit: options.memory_limit(),
        }
    }
}

/// A test-selected Booth width outside `4..=12`.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct InvalidWindow {
    pub(super) bits: u32,
}
