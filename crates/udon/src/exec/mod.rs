//! Scoped execution and task budgets for caller-owned work.
//!
//! [`Executor`] lets a caller supply a runtime for two jobs that complete before
//! the call returns. [`TaskBudget`] divides a concurrency allowance between
//! outer jobs and their nested work. [`for_each_mut`] schedules separate buffer
//! owners or borrowed tiles; [`for_each_chunk_mut`] schedules contiguous chunks.
//! These helpers and [`SerialExecutor`] do not allocate. A caller's executor
//! and jobs may use their own resources.
//!
//! Jobs can return different types, including values borrowed from their inputs:
//!
//! ```
//! use zakura_udon::exec::{Executor, SerialExecutor};
//!
//! let mut values = [1, 2, 3, 4];
//! let (left, right) = values.split_at_mut(2);
//! let (first, sum) = SerialExecutor.join(
//!     || { left[0] += 10; &mut left[0] },
//!     || right.iter().sum::<i32>(),
//! );
//! assert_eq!((*first, sum), (11, 7));
//! ```
//!
//! Independent tiles can share a total task budget with nested FFTs. Each
//! callback receives its own allowance, passed here to
//! [`ExecutionOptions::with_task_budget`].
//! A slice of `Vec`s or other owners works the same way:
//!
//! ```
//! use zakura_udon::{
//!     exec::{ExecutionOptions, SerialExecutor, TaskBudget, for_each_mut},
//!     fft::{Domain, Transform},
//!     field::Fp,
//! };
//!
//! let plan = Transform::new(Domain::new(2).unwrap().subgroup());
//! let mut first = [Fp::ONE; 4];
//! let mut second = [Fp::from_u64(2); 4];
//! let mut tiles = [&mut first[..], &mut second[..]];
//! let budget = TaskBudget::new(4).unwrap();
//! for_each_mut(&mut tiles, budget, &SerialExecutor, |_, tile, inner| {
//!     let options = ExecutionOptions::default().with_task_budget(inner);
//!     let mut scratch = [Fp::ZERO; 4];
//!     plan.forward(tile, options, &SerialExecutor, &mut scratch).unwrap();
//!     plan.inverse(tile, options, &SerialExecutor, &mut scratch).unwrap();
//! });
//! assert_eq!(first.map(|value| value.reduce()), [Fp::ONE; 4]);
//! assert_eq!(second.map(|value| value.reduce()), [Fp::from_u64(2); 4]);
//! ```

mod budget;
pub mod execution;
mod executor;
mod scoped;

pub use budget::{ExecutionOptions, TaskBudget};
pub use executor::{Executor, SerialExecutor};
pub(crate) use scoped::for_each_task_mut;
pub use scoped::{for_each_chunk_mut, for_each_mut};

#[cfg(test)]
mod tests;
