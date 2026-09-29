//! AArch64 Montgomery multiplication, addition, and subtraction for loose
//! Pasta residues.
//!
//! This is the crate's only unsafe code, enabled by the `aarch64-asm` feature
//! on AArch64 targets and unavailable under Miri. The addition and
//! subtraction blocks keep both carry chains and the conditional correction
//! in one flag-carried sequence. The multiplication sequence is
//! a register-renamed transcription of the multiplication routine in
//! Supranational's Semolina 0.1.4 (`mul_mont_pasta`, Apache-2.0), through the
//! inline form in zakura-pasta-curves, without the closing conditional
//! subtraction: the kernel keeps the loose `[0, 2p)` contract of the Rust
//! kernel in `pasta::montgomery` and computes the same integer, so the Rust
//! kernel remains its oracle.
//!
//! The block relies on the shared Pasta modulus shape, `p[2] = 0` and
//! `p[3] = 2^62`, and keeps a five-limb accumulator. With both inputs below
//! `2p`, `lhs[3] <= 2^63`, with equality only for values in `[2^255, 2p)`,
//! so `high(lhs[3] * b)` is at most `2^63 - 1` and the fifth limb, which
//! also absorbs one carry, cannot wrap; each round's sum
//! `acc + lhs * b + q * p` stays below `2^320`. The final candidate `(lhs * rhs + m * p) / R` is
//! below `2p < R` by the closure proof in `pasta::montgomery::square_run`, so
//! the fifth limb is zero and is not returned. The block has no branches or
//! memory accesses.

#![allow(unsafe_code)]

use core::arch::asm;

/// Computes `lhs * rhs * R^-1 mod p` in `[0, 2p)` for inputs below `2p`.
///
/// `p0` and `p1` are the two low modulus limbs and `inv` is
/// `-p^-1 mod 2^64`; the upper limbs are the fixed Pasta shape.
#[inline(always)]
pub(crate) fn montgomery_multiply_loose(
    lhs: &[u64; 4],
    rhs: &[u64; 4],
    p0: u64,
    p1: u64,
    inv: u64,
) -> [u64; 4] {
    let (o0, o1, o2, o3): (u64, u64, u64, u64);
    // SAFETY: straight-line register-only arithmetic with no memory access
    // and no stack use; the outputs depend only on the declared inputs.
    unsafe {
        asm!(
            "mul {r0}, {a0}, {b0}",             // r0 = low(a[0] * b[0]).
            "mul {r1}, {a1}, {b0}",             // r1 = low(a[1] * b[0]).
            "mul {r2}, {a2}, {b0}",             // r2 = low(a[2] * b[0]).
            "mul {r3}, {a3}, {b0}",             // r3 = low(a[3] * b[0]).

            "umulh {t0}, {a0}, {b0}",           // t0 = high(a[0] * b[0]).
            "umulh {t1}, {a1}, {b0}",           // t1 = high(a[1] * b[0]).
            "mul {q}, {inv}, {r0}",             // q = r0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b0}",           // t2 = high(a[2] * b[0]).
            "umulh {t3}, {a3}, {b0}",           // t3 = high(a[3] * b[0]).
            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[0]) into limb 1.
            // low(q * p[0]) cancels r0 and is discarded by the limb shift.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[0]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[0]) and carry.
            // q * p[2] is zero because p[2] = 0.
            "adc {r4}, xzr, {t3}",              // Finish a * b[0] with its fifth limb.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            // Carry from r0 + low(q*p[0]) is one exactly when r0 is nonzero.
            "subs xzr, {r0}, #1",               // Set that carry without computing the zero sum.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry; p[2]'s product is zero.
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry into the fifth limb.

            // Drop the cancelled low limb: (a*b[0] + q*p) / 2^64.
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b1}",             // t0 = low(a[0] * b[1]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b1}",             // t1 = low(a[1] * b[1]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b1}",             // t2 = low(a[2] * b[1]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b1}",             // t3 = low(a[3] * b[1]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 1: add a * b[1] to the reduced accumulator.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[1]) to limb 0.
            "umulh {t0}, {a0}, {b1}",           // t0 = high(a[0] * b[1]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[1]) and carry.
            "umulh {t1}, {a1}, {b1}",           // t1 = high(a[1] * b[1]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[1]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b1}",           // t2 = high(a[2] * b[1]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[1]) and carry.
            "umulh {t3}, {a3}, {b1}",           // t3 = high(a[3] * b[1]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[1]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[1]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[1]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[1]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift after round 1 while starting a * b[2].
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b2}",             // t0 = low(a[0] * b[2]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b2}",             // t1 = low(a[1] * b[2]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b2}",             // t2 = low(a[2] * b[2]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b2}",             // t3 = low(a[3] * b[2]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 2: add a * b[2] and cancel the resulting low limb.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[2]) to limb 0.
            "umulh {t0}, {a0}, {b2}",           // t0 = high(a[0] * b[2]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[2]) and carry.
            "umulh {t1}, {a1}, {b2}",           // t1 = high(a[1] * b[2]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[2]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b2}",           // t2 = high(a[2] * b[2]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[2]) and carry.
            "umulh {t3}, {a3}, {b2}",           // t3 = high(a[3] * b[2]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[2]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[2]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[2]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[2]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift after round 2 while starting a * b[3].
            "adds {r0}, {r1}, {t0}",            // New limb 0 includes high(q * p[0]).
            "mul {t0}, {a0}, {b3}",             // t0 = low(a[0] * b[3]).
            "adcs {r1}, {r2}, {t1}",            // New limb 1 includes high(q * p[1]).
            "mul {t1}, {a1}, {b3}",             // t1 = low(a[1] * b[3]).
            "adcs {r2}, {r3}, xzr",             // New limb 2; p[2] contributes zero.
            "mul {t2}, {a2}, {b3}",             // t2 = low(a[2] * b[3]).
            "adcs {r3}, {r4}, {t3}",            // New limb 3 includes high(q * p[3]).
            "mul {t3}, {a3}, {b3}",             // t3 = low(a[3] * b[3]).
            "adc {r4}, xzr, xzr",               // Capture the reduction carry as limb 4.

            // Round 3: add a * b[3] and perform the last Montgomery cancellation.
            "adds {r0}, {r0}, {t0}",            // Add low(a[0] * b[3]) to limb 0.
            "umulh {t0}, {a0}, {b3}",           // t0 = high(a[0] * b[3]).
            "adcs {r1}, {r1}, {t1}",            // Add low(a[1] * b[3]) and carry.
            "umulh {t1}, {a1}, {b3}",           // t1 = high(a[1] * b[3]).
            "adcs {r2}, {r2}, {t2}",            // Add low(a[2] * b[3]) and carry.
            "mul {q}, {inv}, {r0}",             // q = current limb 0 * inv mod 2^64.
            "umulh {t2}, {a2}, {b3}",           // t2 = high(a[2] * b[3]).
            "adcs {r3}, {r3}, {t3}",            // Add low(a[3] * b[3]) and carry.
            "umulh {t3}, {a3}, {b3}",           // t3 = high(a[3] * b[3]).
            "adc {r4}, {r4}, xzr",              // Propagate multiplication carry to limb 4.

            "adds {r1}, {r1}, {t0}",            // Add high(a[0] * b[3]) to limb 1.
            // low(q * p[0]) cancels r0.
            "adcs {r2}, {r2}, {t1}",            // Add high(a[1] * b[3]) and carry.
            "mul {t1}, {p1}, {q}",              // t1 = low(q * p[1]).
            "adcs {r3}, {r3}, {t2}",            // Add high(a[2] * b[3]) and carry.
            // low(q * p[2]) is zero.
            "adc {r4}, {r4}, {t3}",             // Add high(a[3] * b[3]) and final carry.
            "lsl {t3}, {q}, #62",               // t3 = low(q * p[3]).
            "subs xzr, {r0}, #1",               // Set the low-limb cancellation carry.
            "umulh {t0}, {p0}, {q}",            // t0 = high(q * p[0]).
            "adcs {r1}, {r1}, {t1}",            // Add low(q * p[1]) and cancellation carry.
            "umulh {t1}, {p1}, {q}",            // t1 = high(q * p[1]).
            "adcs {r2}, {r2}, xzr",             // Propagate carry across zero p[2].
            // high(q * p[2]) is zero.
            "adcs {r3}, {r3}, {t3}",            // Add low(q * p[3]) and carry.
            "lsr {t3}, {q}, #2",                // t3 = high(q * p[3]).
            "adc {r4}, {r4}, xzr",              // Propagate carry to limb 4.

            // Shift out the fourth cancelled limb. For loose inputs below 2p the
            // candidate (lhs*rhs + m*p)/R is below 2p < R (see the module docs),
            // so no fifth limb exists.
            "adds {r0}, {r1}, {t0}",            // Final candidate limb 0.
            "adcs {r1}, {r2}, {t1}",            // Final candidate limb 1.
            "adcs {r2}, {r3}, xzr",             // Final candidate limb 2.
            "adcs {r3}, {r4}, {t3}",            // Final candidate limb 3.
            a0 = in(reg) lhs[0],
            a1 = in(reg) lhs[1],
            a2 = in(reg) lhs[2],
            a3 = in(reg) lhs[3],
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            p0 = in(reg) p0,
            p1 = in(reg) p1,
            inv = in(reg) inv,
            q = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            r0 = out(reg) o0,
            r1 = out(reg) o1,
            r2 = out(reg) o2,
            r3 = out(reg) o3,
            r4 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [o0, o1, o2, o3]
}

/// Computes `lhs + rhs mod 2p` in `[0, 2p)` for inputs below `2p`.
///
/// `twice_modulus` is `2p`. The sum is below `4p`, which can exceed the
/// radix, so the block keeps the top carry and folds it into the single
/// conditional subtraction of `2p`: the reduced candidate is selected when
/// the addition carried or the subtraction did not borrow. This computes the
/// same function as the portable kernel in `pasta::montgomery` on all inputs.
#[inline(always)]
pub(crate) fn add_loose(lhs: &[u64; 4], rhs: &[u64; 4], twice_modulus: &[u64; 4]) -> [u64; 4] {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line register-only arithmetic with no memory access
    // and no stack use; the outputs depend only on the declared inputs.
    unsafe {
        asm!(
            "adds {r0}, {r0}, {b0}",
            "adcs {r1}, {r1}, {b1}",
            "adcs {r2}, {r2}, {b2}",
            "adcs {r3}, {r3}, {b3}",
            "cset {c}, cs",
            "subs {t0}, {r0}, {m0}",
            "sbcs {t1}, {r1}, {m1}",
            "sbcs {t2}, {r2}, {m2}",
            "sbcs {t3}, {r3}, {m3}",
            "adcs {c}, {c}, xzr",
            "csel {r0}, {t0}, {r0}, ne",
            "csel {r1}, {t1}, {r1}, ne",
            "csel {r2}, {t2}, {r2}, ne",
            "csel {r3}, {t3}, {r3}, ne",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            m0 = in(reg) twice_modulus[0],
            m1 = in(reg) twice_modulus[1],
            m2 = in(reg) twice_modulus[2],
            m3 = in(reg) twice_modulus[3],
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            c = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}

/// Computes `lhs - rhs mod 2p` in `[0, 2p)` for inputs below `2p`.
///
/// `twice_modulus` is `2p`, added back exactly when the subtraction borrows;
/// the final carry is discarded after wrapping modulo `2^256`. This computes
/// the same function as the portable kernel in `pasta::montgomery` on all
/// inputs.
#[inline(always)]
pub(crate) fn sub_loose(lhs: &[u64; 4], rhs: &[u64; 4], twice_modulus: &[u64; 4]) -> [u64; 4] {
    let [mut r0, mut r1, mut r2, mut r3] = *lhs;
    // SAFETY: straight-line register-only arithmetic with no memory access
    // and no stack use; the outputs depend only on the declared inputs.
    unsafe {
        asm!(
            "subs {r0}, {r0}, {b0}",
            "sbcs {r1}, {r1}, {b1}",
            "sbcs {r2}, {r2}, {b2}",
            "sbcs {r3}, {r3}, {b3}",
            "csel {t0}, {m0}, xzr, cc",
            "csel {t1}, {m1}, xzr, cc",
            "csel {t2}, {m2}, xzr, cc",
            "csel {t3}, {m3}, xzr, cc",
            "adds {r0}, {r0}, {t0}",
            "adcs {r1}, {r1}, {t1}",
            "adcs {r2}, {r2}, {t2}",
            "adc {r3}, {r3}, {t3}",
            r0 = inout(reg) r0,
            r1 = inout(reg) r1,
            r2 = inout(reg) r2,
            r3 = inout(reg) r3,
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            m0 = in(reg) twice_modulus[0],
            m1 = in(reg) twice_modulus[1],
            m2 = in(reg) twice_modulus[2],
            m3 = in(reg) twice_modulus[3],
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    [r0, r1, r2, r3]
}
