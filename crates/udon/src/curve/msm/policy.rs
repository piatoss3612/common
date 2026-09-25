//! Private kernel controls used by selection and arithmetic tests.

#[cfg(test)]
use super::CurveError;
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
pub(super) enum Kernel {
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
    pub(super) kernel: Kernel,
}
impl ArithmeticOptions {
    pub(super) const DEFAULT: Self = Self {
        max_terms_per_pass: None,
        chunk_size: None,
        kernel: Kernel::Auto,
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
    pub(super) const fn with_kernel(mut self, kernel: Kernel) -> Result<Self, CurveError> {
        if let Kernel::Booth {
            width: Some(bits), ..
        }
        | Kernel::StreamingBooth { width: Some(bits) } = kernel
            && (bits < 4 || bits > 12)
        {
            return Err(CurveError::InvalidMsmWindow { bits });
        }
        self.kernel = kernel;
        Ok(self)
    }
    pub(super) const fn streaming(self) -> bool {
        matches!(self.kernel, Kernel::StreamingBooth { .. })
    }
    pub(super) const fn accumulation(self) -> Accumulation {
        match self.kernel {
            Kernel::Booth { accumulation, .. } => accumulation,
            Kernel::StreamingBooth { .. } => Accumulation::Projective,
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
