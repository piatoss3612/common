//! Unreduced 256-by-256 multiplication accumulated into eight limbs.
//!
//! Write B = 2^64. Each row adds low product words, then high words.
//! Row i consumes the previous row's pending carry at word i + 4 and
//! leaves its own final carry pending at word i + 5. Thus no row needs
//! to propagate through the untouched upper accumulator words.
//! The highest high-product word plus the previous pending carry and the
//! low-chain carry can require 65 bits. Preserve its overflow separately,
//! then add the high-chain carry-out at the same next-word position.
//! After row i, the initial low i + 5 words plus the processed partial
//! product are below 2 * B^(i + 5), so their combined carry is 0 or 1.
//! The returned overflow is at most one because acc + lhs * rhs < 2^513.
//! All input limbs may be arbitrary u64 values; no field bound is assumed.

use core::arch::asm;

#[inline(always)]
pub(super) fn mul_accumulate(
    accumulator: [u64; 8],
    lhs: &[u64; 4],
    rhs: &[u64; 4],
) -> ([u64; 8], u64) {
    let [
        mut d0,
        mut d1,
        mut d2,
        mut d3,
        mut d4,
        mut d5,
        mut d6,
        mut d7,
    ] = accumulator;
    let overflow;
    // SAFETY: only register arithmetic, all inputs and clobbers declared.
    // There are no memory accesses or data-dependent control flow.
    unsafe {
        asm!(
            // Schoolbook row 0, starting at accumulator limb 0.
            "mul {t0}, {a0}, {b0}",
            "mul {t1}, {a0}, {b1}",
            "mul {t2}, {a0}, {b2}",
            "mul {t3}, {a0}, {b3}",
            "adds {d0}, {d0}, {t0}",
            "adcs {d1}, {d1}, {t1}",
            "adcs {d2}, {d2}, {t2}",
            "adcs {d3}, {d3}, {t3}",
            // high(a0 * b3) <= B - 2, so this carry addition fits.
            "umulh {t4}, {a0}, {b3}",
            "adc {t4}, {t4}, xzr",
            "umulh {t0}, {a0}, {b0}",
            "adds {d1}, {d1}, {t0}",
            "umulh {t0}, {a0}, {b1}",
            "adcs {d2}, {d2}, {t0}",
            "umulh {t0}, {a0}, {b2}",
            "adcs {d3}, {d3}, {t0}",
            "adcs {d4}, {d4}, {t4}",
            "adc {overflow}, xzr, xzr",
            // Schoolbook row 1, starting at accumulator limb 1.
            "mul {t0}, {a1}, {b0}",
            "mul {t1}, {a1}, {b1}",
            "mul {t2}, {a1}, {b2}",
            "mul {t3}, {a1}, {b3}",
            "adds {d1}, {d1}, {t0}",
            "adcs {d2}, {d2}, {t1}",
            "adcs {d3}, {d3}, {t2}",
            "adcs {d4}, {d4}, {t3}",
            // Both incoming carries belong at word 5. Keep the 65th
            // bit of high(a1 * b3) + pending + low_chain_carry.
            "umulh {t4}, {a1}, {b3}",
            "adcs {t4}, {t4}, {overflow}",
            "adc {overflow}, xzr, xzr",
            "umulh {t0}, {a1}, {b0}",
            "adds {d2}, {d2}, {t0}",
            "umulh {t0}, {a1}, {b1}",
            "adcs {d3}, {d3}, {t0}",
            "umulh {t0}, {a1}, {b2}",
            "adcs {d4}, {d4}, {t0}",
            "adcs {d5}, {d5}, {t4}",
            "adc {overflow}, {overflow}, xzr",
            // Schoolbook row 2, starting at accumulator limb 2.
            "mul {t0}, {a2}, {b0}",
            "mul {t1}, {a2}, {b1}",
            "mul {t2}, {a2}, {b2}",
            "mul {t3}, {a2}, {b3}",
            "adds {d2}, {d2}, {t0}",
            "adcs {d3}, {d3}, {t1}",
            "adcs {d4}, {d4}, {t2}",
            "adcs {d5}, {d5}, {t3}",
            // Both incoming carries belong at word 6. Keep the 65th
            // bit of high(a2 * b3) + pending + low_chain_carry.
            "umulh {t4}, {a2}, {b3}",
            "adcs {t4}, {t4}, {overflow}",
            "adc {overflow}, xzr, xzr",
            "umulh {t0}, {a2}, {b0}",
            "adds {d3}, {d3}, {t0}",
            "umulh {t0}, {a2}, {b1}",
            "adcs {d4}, {d4}, {t0}",
            "umulh {t0}, {a2}, {b2}",
            "adcs {d5}, {d5}, {t0}",
            "adcs {d6}, {d6}, {t4}",
            "adc {overflow}, {overflow}, xzr",
            // Schoolbook row 3, starting at accumulator limb 3.
            "mul {t0}, {a3}, {b0}",
            "mul {t1}, {a3}, {b1}",
            "mul {t2}, {a3}, {b2}",
            "mul {t3}, {a3}, {b3}",
            "adds {d3}, {d3}, {t0}",
            "adcs {d4}, {d4}, {t1}",
            "adcs {d5}, {d5}, {t2}",
            "adcs {d6}, {d6}, {t3}",
            // Both incoming carries belong at word 7. Keep the 65th
            // bit of high(a3 * b3) + pending + low_chain_carry.
            "umulh {t4}, {a3}, {b3}",
            "adcs {t4}, {t4}, {overflow}",
            "adc {overflow}, xzr, xzr",
            "umulh {t0}, {a3}, {b0}",
            "adds {d4}, {d4}, {t0}",
            "umulh {t0}, {a3}, {b1}",
            "adcs {d5}, {d5}, {t0}",
            "umulh {t0}, {a3}, {b2}",
            "adcs {d6}, {d6}, {t0}",
            "adcs {d7}, {d7}, {t4}",
            "adc {overflow}, {overflow}, xzr",
            d0 = inout(reg) d0,
            d1 = inout(reg) d1,
            d2 = inout(reg) d2,
            d3 = inout(reg) d3,
            d4 = inout(reg) d4,
            d5 = inout(reg) d5,
            d6 = inout(reg) d6,
            d7 = inout(reg) d7,
            a0 = in(reg) lhs[0],
            a1 = in(reg) lhs[1],
            a2 = in(reg) lhs[2],
            a3 = in(reg) lhs[3],
            b0 = in(reg) rhs[0],
            b1 = in(reg) rhs[1],
            b2 = in(reg) rhs[2],
            b3 = in(reg) rhs[3],
            overflow = out(reg) overflow,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            t4 = out(reg) _,
            options(pure, nomem, nostack),
        );
    }
    ([d0, d1, d2, d3, d4, d5, d6, d7], overflow)
}
