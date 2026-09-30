//! Variable-time modular inversion for the Pasta base fields, using 62-bit
//! divsteps (safegcd).
//!
//! Portions adapted from libsecp256k1's `src/modinv64_impl.h`:
//! Copyright (c) 2020 Peter Dettman. Distributed under the MIT software
//! license; see the workspace `LICENSE-MIT` file.
//!
//! Ported from `pasta_curves::fields::modinv62`.
//!
//! This is a near-verbatim port of libsecp256k1's signed-62 variable-time
//! modular inversion, specialized to the Pasta base fields:
//!
//! - The coefficient `e` is seeded with $R^2 \bmod m$ (rather than $1$), so a
//!   Montgomery-form input $X = aR$ yields the Montgomery-form inverse
//!   $a^{-1}R$ directly: no conversion into or out of Montgomery form, and no
//!   Montgomery reduction anywhere.
//! - Both moduli are $[m_0, m_1, 2, 0, 64]$ in signed radix $2^{62}$, so the
//!   `k*m` corrections in the coefficient updates need only two general
//!   multiplications (limbs 0 and 1); limb 2 is `k << 1`, limb 3 vanishes,
//!   and limb 4 is `k << 6`. The first `f,g` update exploits the same shape
//!   (the initial `f` *is* the modulus).
//! - `f,g` are updated before `d,e` each batch, so the terminal batch computes
//!   only the one coefficient row that becomes the inverse, and the first
//!   batch (where `d = 0`) skips every product against `d`.
//!
//! The scaled invariants maintained across batches are
//! $X d \equiv R^2 f \pmod m$ and $X e \equiv R^2 g \pmod m$; at termination
//! $g = 0$ and $f = \sigma \in \{-1, 1\}$, so $\sigma d \equiv R^2 X^{-1}
//! \equiv a^{-1} R \pmod m$.
//!
//! Every inner-loop invariant inherited from upstream (`VERIFY_CHECK`) is kept
//! live in unit tests — including `--release` test runs — via the `verify!`
//! macro below, and compiled out of production builds.
//!
//! # Warning
//!
//! This inversion is **variable-time in the value being inverted**: its
//! trailing-zero counts, divstep branches, batch count, and active limb length
//! all depend on the input. It must only reach values whose timing is
//! acceptable to leak.
//!
//! # References
//!
//! - Daniel J. Bernstein, Bo-Yin Yang: "Fast constant-time gcd computation and
//!   modular inversion" <https://eprint.iacr.org/2019/266>
//! - libsecp256k1, `src/modinv64_impl.h` (the ported implementation) and
//!   `doc/safegcd_implementation.md`
//!   <https://github.com/bitcoin-core/secp256k1>
//! - Thomas Pornin: "Optimized Binary GCD for Modular Inversion"
//!   <https://eprint.iacr.org/2020/972> (the constant-time alternative)

use super::{PallasBase, PallasScalar, PrimeModulus};

/// Mask of the low 62 bits of a word.
const MASK62: u64 = (1u64 << 62) - 1;

/// Per-field constants for the divstep inversion, in signed radix $2^{62}$.
///
/// The kernels hardcode the shared sparse shape of both Pasta moduli
/// (`MODULUS_62[2] == 2`, `MODULUS_62[3] == 0`, `MODULUS_62[4] == 64`); the const
/// assertions below pin that assumption to the constants.
pub(super) trait InvParams {
    /// The field modulus $m$, radix $2^{62}$, least-significant limb first.
    const MODULUS_62: [i64; 5];
    /// $m^{-1} \bmod 2^{62}$ (note: *not* the Montgomery `INV`, which is
    /// $-m^{-1} \bmod 2^{64}$).
    const MU: u64;
    /// $R^2 = 2^{512} \bmod m$, radix $2^{62}$: the initial value of `e`.
    const R2_62: [i64; 5];
}

impl<M: PrimeModulus> InvParams for M {
    const MODULUS_62: [i64; 5] = M::MODULUS_SIGNED62;
    const MU: u64 = M::MONTGOMERY_INV.wrapping_neg() & MASK62;
    const R2_62: [i64; 5] = to_signed62(&M::R2);
}

/// The kernels bake in the sparse modulus shape; pin it (and the low-limb
/// inverse relation) at compile time for both parameter sets.
const _: () = {
    const fn check<P: InvParams>() {
        assert!(P::MODULUS_62[2] == 2 && P::MODULUS_62[3] == 0 && P::MODULUS_62[4] == 64);
        assert!(P::MU.wrapping_mul(P::MODULUS_62[0] as u64) & MASK62 == 1);
    }
    check::<PallasBase>();
    check::<PallasScalar>();
};

/// A signed multi-word integer in radix $2^{62}$, least-significant limb
/// first. In canonical form all limbs below the active length are in
/// $[0, 2^{62})$ and the top active limb carries the sign, so the sign of the
/// value is the sign of that limb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Signed62([i64; 5]);

/// The transition matrix for one batch of 62 divsteps. Row sums are bounded
/// by $|u| + |v| \le 2^{62}$ and $|q| + |r| \le 2^{62}$, and the determinant
/// is exactly $2^{62}$.
#[derive(Clone, Copy, Debug)]
struct Trans2x2 {
    u: i64,
    v: i64,
    q: i64,
    r: i64,
}

/// Computes the wider cancellation multiplier used after a negative-eta
/// divstep swap on AArch64.
///
/// Keeping this out of line prevents the AArch64 backend from speculatively
/// evaluating both cancellation formulas before selecting one.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn negative_eta_multiplier(f: u64, g: u64, mask: u64) -> u64 {
    f.wrapping_mul(g)
        .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
        & mask
}

/// An upstream `VERIFY_CHECK`: live in unit tests (including `--release`
/// runs, where `debug_assert!` would be inert) and in debug builds; compiled
/// out of production builds. Keep every check ported from upstream on this
/// macro so the user's `cargo test --release` exercises them.
macro_rules! verify {
    ($cond:expr_2021 $(, $($arg:tt)+)?) => {
        if cfg!(any(test, debug_assertions)) {
            assert!($cond $(, $($arg)+)?);
        }
    };
}

/// Value-level bound checks, too expensive for unconditional inlining: a port
/// of upstream's `secp256k1_modinv64_mul_cmp_62` machinery, active only where
/// `verify!` is.
#[cfg(any(test, debug_assertions))]
mod checks {
    use super::{MASK62, Signed62};
    use core::cmp::Ordering;

    /// Computes `a * factor` (using `alen` active limbs of `a`) as a 5-limb
    /// signed-62 value with all but the top limb normalized.
    #[allow(clippy::needless_range_loop)]
    pub(super) fn mul_62(a: &Signed62, alen: usize, factor: i64) -> Signed62 {
        let mut c: i128 = 0;
        let mut r = [0i64; 5];
        for i in 0..4 {
            if i < alen {
                c += i128::from(a.0[i]) * i128::from(factor);
            }
            r[i] = (c as u64 & MASK62) as i64;
            c >>= 62;
        }
        if 4 < alen {
            c += i128::from(a.0[4]) * i128::from(factor);
        }
        assert!(c == i128::from(c as i64), "mul_62 top limb must fit i64");
        r[4] = c as i64;
        Signed62(r)
    }

    /// Compares `a` (with `alen` active limbs) against `b * factor`.
    pub(super) fn mul_cmp_62(a: &Signed62, alen: usize, b: &Signed62, factor: i64) -> Ordering {
        let am = mul_62(a, alen, 1); // normalizes all but the top limb of a
        let bm = mul_62(b, 5, factor);
        for i in 0..4 {
            assert!(am.0[i] >> 62 == 0, "mul_cmp operand not normalized");
            assert!(bm.0[i] >> 62 == 0, "mul_cmp operand not normalized");
        }
        for i in (0..5).rev() {
            if am.0[i] < bm.0[i] {
                return Ordering::Less;
            }
            if am.0[i] > bm.0[i] {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

    /// Asserts $-2m < x < m$ (the coefficient range invariant).
    pub(super) fn assert_coeff_range<P: super::InvParams>(x: &Signed62, what: &str) {
        let m = Signed62(P::MODULUS_62);
        assert!(
            mul_cmp_62(x, 5, &m, -2) == Ordering::Greater,
            "{} must exceed -2m",
            what
        );
        assert!(
            mul_cmp_62(x, 5, &m, 1) == Ordering::Less,
            "{} must be below m",
            what
        );
    }

    /// Asserts $-m < x \le m$ / $-m < x < m$ (the `f`/`g` range invariants).
    pub(super) fn assert_fg_range<P: super::InvParams>(f: &Signed62, g: &Signed62, len: usize) {
        let m = Signed62(P::MODULUS_62);
        assert!(
            mul_cmp_62(f, len, &m, -1) == Ordering::Greater,
            "f must exceed -m"
        );
        assert!(
            mul_cmp_62(f, len, &m, 1) != Ordering::Greater,
            "f must not exceed m"
        );
        assert!(
            mul_cmp_62(g, len, &m, -1) == Ordering::Greater,
            "g must exceed -m"
        );
        assert!(
            mul_cmp_62(g, len, &m, 1) == Ordering::Less,
            "g must be below m"
        );
    }
}

/// Repacks an unsigned 256-bit integer into five radix-2^62 limbs.
#[inline]
pub(super) const fn to_signed62(x: &[u64; 4]) -> [i64; 5] {
    [
        (x[0] & MASK62) as i64,
        (((x[0] >> 62) | (x[1] << 2)) & MASK62) as i64,
        (((x[1] >> 60) | (x[2] << 4)) & MASK62) as i64,
        (((x[2] >> 58) | (x[3] << 6)) & MASK62) as i64,
        (x[3] >> 56) as i64,
    ]
}

/// Repacks a canonical 4x64-bit little-endian field representation (a value
/// in $[0, m)$) into signed radix $2^{62}$.
fn pack62(x: &[u64; 4]) -> Signed62 {
    let out = Signed62(to_signed62(x));
    verify!(
        out.0[4] >= 0 && out.0[4] <= 0x7f,
        "packed top limb out of range"
    );
    out
}

/// Repacks a canonical signed-62 value in $[0, m)$ (a [`normalize_62`]
/// output) back into the 4x64-bit little-endian field representation.
fn unpack62(v: &Signed62) -> [u64; 4] {
    let v0 = v.0[0] as u64;
    let v1 = v.0[1] as u64;
    let v2 = v.0[2] as u64;
    let v3 = v.0[3] as u64;
    let v4 = v.0[4] as u64;
    verify!(
        v0 <= MASK62 && v1 <= MASK62 && v2 <= MASK62 && v3 <= MASK62 && v4 <= 0x7f,
        "unpack input must be canonical and reduced"
    );
    [
        v0 | (v1 << 62),
        (v1 >> 2) | (v2 << 60),
        (v2 >> 4) | (v3 << 58),
        (v3 >> 6) | (v4 << 56),
    ]
}

/// Computes the transition matrix for 62 variable-time divsteps, reading only
/// the bottom limbs of `f` and `g` (which must be odd / the current `eta`).
/// Returns the matrix and the updated `eta`.
///
/// This is upstream's `secp256k1_modinv64_divsteps_62_var`: trailing-zero
/// counts batch the halvings of `g`, and small modular-inverse formulas cancel
/// up to 4 bits (`eta >= 0`) or 6 bits (`eta < 0`, after the swap) of `g` per
/// inner iteration.
#[allow(clippy::many_single_char_names)]
fn divsteps_62_var(mut eta: i64, f0: u64, g0: u64) -> (Trans2x2, i64) {
    let mut u: u64 = 1;
    let mut v: u64 = 0;
    let mut q: u64 = 0;
    let mut r: u64 = 1;
    let mut f = f0;
    let mut g = g0;
    let mut i: u32 = 62;

    loop {
        // Use a sentinel bit to count zeros only up to i.
        let zeros = (g | (u64::MAX << i)).trailing_zeros();
        // Perform zeros divsteps at once; they all just divide g by two.
        g >>= zeros;
        u <<= zeros;
        v <<= zeros;
        eta -= i64::from(zeros);
        i -= zeros;
        // We're done once we've performed 62 divsteps.
        if i == 0 {
            break;
        }
        verify!(f & 1 == 1, "f must be odd");
        verify!(g & 1 == 1, "g must be odd after shifting out its zeros");
        verify!(
            u.wrapping_mul(f0).wrapping_add(v.wrapping_mul(g0)) == f << (62 - i),
            "top row of T must reproduce f"
        );
        verify!(
            q.wrapping_mul(f0).wrapping_add(r.wrapping_mul(g0)) == g << (62 - i),
            "bottom row of T must reproduce g"
        );
        // eta starts at -1 and moves by at most one per divstep, and a full
        // inversion performs at most 744 divsteps.
        verify!((-745..=745).contains(&eta), "eta out of range");
        let limit;
        let m;
        let w;
        if eta < 0 {
            // If eta is negative, negate it and replace f,g with g,-f.
            eta = -eta;
            let tmp = f;
            f = g;
            g = tmp.wrapping_neg();
            let tmp = u;
            u = q;
            q = tmp.wrapping_neg();
            let tmp = v;
            v = r;
            r = tmp.wrapping_neg();
            // Use a formula to cancel out up to 6 bits of g, as f is odd:
            // f*(f*f - 2) is the inverse of -f modulo 64.
            limit = if eta as u32 + 1 > i {
                i
            } else {
                eta as u32 + 1
            };
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 63;
            #[cfg(target_arch = "aarch64")]
            {
                w = negative_eta_multiplier(f, g, m);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                w = f
                    .wrapping_mul(g)
                    .wrapping_mul(f.wrapping_mul(f).wrapping_sub(2))
                    & m;
            }
        } else {
            // A simpler formula that cancels up to 4 bits of g:
            // f + (((f + 1) & 4) << 1) is the inverse of f modulo 16.
            limit = if eta as u32 + 1 > i {
                i
            } else {
                eta as u32 + 1
            };
            verify!((1..=62).contains(&limit), "limit out of range");
            m = (u64::MAX >> (64 - limit)) & 15;
            let w0 = f.wrapping_add((f.wrapping_add(1) & 4) << 1);
            w = w0.wrapping_neg().wrapping_mul(g) & m;
        }
        g = g.wrapping_add(f.wrapping_mul(w));
        q = q.wrapping_add(u.wrapping_mul(w));
        r = r.wrapping_add(v.wrapping_mul(w));
        verify!(g & m == 0, "the masked low bits of g must cancel");
    }
    let t = Trans2x2 {
        u: u as i64,
        v: v as i64,
        q: q as i64,
        r: r as i64,
    };
    // The determinant of T must be a power of two: this guarantees the
    // matrix-vector products below preserve the relative sizes of f and g.
    verify!(
        i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q) == 1 << 62,
        "determinant of T must be 2^62"
    );
    #[cfg(any(test, debug_assertions))]
    {
        assert!(
            t.u.unsigned_abs() + t.v.unsigned_abs() <= 1 << 62,
            "|u| + |v| must not exceed 2^62"
        );
        assert!(
            t.q.unsigned_abs() + t.r.unsigned_abs() <= 1 << 62,
            "|q| + |r| must not exceed 2^62"
        );
    }
    (t, eta)
}

/// Applies the transition matrix to the full-width `f` and `g` (over `len`
/// active limbs), dividing exactly by $2^{62}$. Upstream's
/// `secp256k1_modinv64_update_fg_62_var`.
#[allow(clippy::many_single_char_names)]
fn update_fg_62_var(len: usize, f: &mut Signed62, g: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    verify!((1..=5).contains(&len), "active length out of range");
    let mut fi = f.0[0];
    let mut gi = g.0[0];
    let mut cf = i128::from(u) * i128::from(fi) + i128::from(v) * i128::from(gi);
    let mut cg = i128::from(q) * i128::from(fi) + i128::from(r) * i128::from(gi);
    // The bottom 62 bits of t*[f,g] are zero by construction of t.
    verify!(cf as u64 & MASK62 == 0, "low bits of new f must vanish");
    verify!(cg as u64 & MASK62 == 0, "low bits of new g must vanish");
    cf >>= 62;
    cg >>= 62;
    for j in 1..len {
        fi = f.0[j];
        gi = g.0[j];
        cf += i128::from(u) * i128::from(fi) + i128::from(v) * i128::from(gi);
        cg += i128::from(q) * i128::from(fi) + i128::from(r) * i128::from(gi);
        f.0[j - 1] = (cf as u64 & MASK62) as i64;
        cf >>= 62;
        g.0[j - 1] = (cg as u64 & MASK62) as i64;
        cg >>= 62;
    }
    verify!(cf == i128::from(cf as i64), "f tail must fit one limb");
    verify!(cg == i128::from(cg as i64), "g tail must fit one limb");
    f.0[len - 1] = cf as i64;
    g.0[len - 1] = cg as i64;
}

/// [`update_fg_62_var`] specialized to the first batch, where `f` is still
/// the modulus `[m0, m1, 2, 0, 64]`: the three sparse `f` limbs become shifts
/// of the matrix entries. Emits exactly the limbs the generic update would.
#[allow(clippy::many_single_char_names)]
fn update_fg_62_first<P: InvParams>(f: &mut Signed62, g: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    verify!(f.0 == P::MODULUS_62, "first f,g update requires f = m");
    let m0 = P::MODULUS_62[0];
    let m1 = P::MODULUS_62[1];
    // Limb 0.
    let mut cf = i128::from(u) * i128::from(m0) + i128::from(v) * i128::from(g.0[0]);
    let mut cg = i128::from(q) * i128::from(m0) + i128::from(r) * i128::from(g.0[0]);
    verify!(cf as u64 & MASK62 == 0, "low bits of new f must vanish");
    verify!(cg as u64 & MASK62 == 0, "low bits of new g must vanish");
    cf >>= 62;
    cg >>= 62;
    // Limb 1.
    cf += i128::from(u) * i128::from(m1) + i128::from(v) * i128::from(g.0[1]);
    cg += i128::from(q) * i128::from(m1) + i128::from(r) * i128::from(g.0[1]);
    f.0[0] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[0] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 2: m2 = 2.
    cf += (i128::from(u) << 1) + i128::from(v) * i128::from(g.0[2]);
    cg += (i128::from(q) << 1) + i128::from(r) * i128::from(g.0[2]);
    f.0[1] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[1] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 3: m3 = 0.
    cf += i128::from(v) * i128::from(g.0[3]);
    cg += i128::from(r) * i128::from(g.0[3]);
    f.0[2] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[2] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    // Limb 4: m4 = 64.
    cf += (i128::from(u) << 6) + i128::from(v) * i128::from(g.0[4]);
    cg += (i128::from(q) << 6) + i128::from(r) * i128::from(g.0[4]);
    f.0[3] = (cf as u64 & MASK62) as i64;
    cf >>= 62;
    g.0[3] = (cg as u64 & MASK62) as i64;
    cg >>= 62;
    verify!(cf == i128::from(cf as i64), "f tail must fit one limb");
    verify!(cg == i128::from(cg as i64), "g tail must fit one limb");
    f.0[4] = cf as i64;
    g.0[4] = cg as i64;
}

/// Applies the transition matrix to the coefficients `d`, `e` modulo $m$,
/// dividing exactly by $2^{62}$. Upstream's
/// `secp256k1_modinv64_update_de_62`, with the sparse modulus limbs 2 and 4
/// applied as shifts and limb 3 skipped.
///
/// Maintains $-2m < d, e < m$: with the sign corrections `(u & sd) + (v & se)`
/// and the $\mu$-derived cancellation term, the numerator of each row stays
/// within $(-2^{63} m, 2^{62} m)$, so the exact shift by 62 lands back in
/// $(-2m, m)$. The bound depends only on the ranges of `d` and `e`, never on
/// their values.
#[allow(clippy::many_single_char_names)]
fn update_de_62<P: InvParams>(d: &mut Signed62, e: &mut Signed62, t: &Trans2x2) {
    let u = t.u;
    let v = t.v;
    let q = t.q;
    let r = t.r;
    let d0 = d.0[0];
    let d1 = d.0[1];
    let d2 = d.0[2];
    let d3 = d.0[3];
    let d4 = d.0[4];
    let e0 = e.0[0];
    let e1 = e.0[1];
    let e2 = e.0[2];
    let e3 = e.0[3];
    let e4 = e.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "d");
        checks::assert_coeff_range::<P>(e, "e");
    }
    // [md, me] start as zero; plus [u, q] if d is negative; plus [v, r] if e
    // is negative.
    let sd = d4 >> 63;
    let se = e4 >> 63;
    let mut md = (u & sd) + (v & se);
    let mut me = (q & sd) + (r & se);
    // Begin computing t*[d, e].
    let mut cd = i128::from(u) * i128::from(d0) + i128::from(v) * i128::from(e0);
    let mut ce = i128::from(q) * i128::from(d0) + i128::from(r) * i128::from(e0);
    // Correct md, me so that t*[d, e] + m*[md, me] has 62 zero bottom bits.
    md = md.wrapping_sub((P::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    me = me.wrapping_sub((P::MU.wrapping_mul(ce as u64).wrapping_add(me as u64) & MASK62) as i64);
    // Limb 0 of the modulus contribution (general multiplication).
    cd += i128::from(P::MODULUS_62[0]) * i128::from(md);
    ce += i128::from(P::MODULUS_62[0]) * i128::from(me);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    verify!(ce as u64 & MASK62 == 0, "low bits of new e must vanish");
    cd >>= 62;
    ce >>= 62;
    // Limb 1 (general multiplication by m1).
    cd += i128::from(u) * i128::from(d1) + i128::from(v) * i128::from(e1);
    ce += i128::from(q) * i128::from(d1) + i128::from(r) * i128::from(e1);
    cd += i128::from(P::MODULUS_62[1]) * i128::from(md);
    ce += i128::from(P::MODULUS_62[1]) * i128::from(me);
    d.0[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[0] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 2: m2 = 2, so the modulus contribution is a shift.
    cd += i128::from(u) * i128::from(d2) + i128::from(v) * i128::from(e2);
    ce += i128::from(q) * i128::from(d2) + i128::from(r) * i128::from(e2);
    cd += i128::from(md) << 1;
    ce += i128::from(me) << 1;
    d.0[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[1] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 3: m3 = 0, no modulus contribution.
    cd += i128::from(u) * i128::from(d3) + i128::from(v) * i128::from(e3);
    ce += i128::from(q) * i128::from(d3) + i128::from(r) * i128::from(e3);
    d.0[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[2] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 4: m4 = 64, so the modulus contribution is a shift.
    cd += i128::from(u) * i128::from(d4) + i128::from(v) * i128::from(e4);
    ce += i128::from(q) * i128::from(d4) + i128::from(r) * i128::from(e4);
    cd += i128::from(md) << 6;
    ce += i128::from(me) << 6;
    d.0[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e.0[3] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    verify!(ce == i128::from(ce as i64), "e tail must fit one limb");
    d.0[4] = cd as i64;
    e.0[4] = ce as i64;
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "new d");
        checks::assert_coeff_range::<P>(e, "new e");
    }
}

/// [`update_de_62`] specialized to the first batch, where `d = 0` (so every
/// product against `d` vanishes and both sign masks are zero, since the
/// initial `e = R^2` satisfies $0 \le e < m$). Returns the new `(d, e)`.
#[allow(clippy::many_single_char_names)]
fn update_de_62_first<P: InvParams>(e: &Signed62, t: &Trans2x2) -> (Signed62, Signed62) {
    // d = 0, so the u and q columns of the matrix contribute nothing.
    let v = t.v;
    let r = t.r;
    let e0 = e.0[0];
    let e1 = e.0[1];
    let e2 = e.0[2];
    let e3 = e.0[3];
    let e4 = e.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        use core::cmp::Ordering;
        assert!(
            checks::mul_cmp_62(e, 5, &Signed62([0; 5]), 0) != Ordering::Less,
            "first-batch e must be non-negative"
        );
        checks::assert_coeff_range::<P>(e, "e");
    }
    // d = 0 and e >= 0, so both sign corrections vanish.
    let mut cd = i128::from(v) * i128::from(e0);
    let mut ce = i128::from(r) * i128::from(e0);
    let mut md = (P::MU.wrapping_mul(cd as u64) & MASK62) as i64;
    md = md.wrapping_neg();
    let mut me = (P::MU.wrapping_mul(ce as u64) & MASK62) as i64;
    me = me.wrapping_neg();
    cd += i128::from(P::MODULUS_62[0]) * i128::from(md);
    ce += i128::from(P::MODULUS_62[0]) * i128::from(me);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    verify!(ce as u64 & MASK62 == 0, "low bits of new e must vanish");
    cd >>= 62;
    ce >>= 62;
    let mut d_out = [0i64; 5];
    let mut e_out = [0i64; 5];
    // Limb 1.
    cd += i128::from(v) * i128::from(e1) + i128::from(P::MODULUS_62[1]) * i128::from(md);
    ce += i128::from(r) * i128::from(e1) + i128::from(P::MODULUS_62[1]) * i128::from(me);
    d_out[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[0] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 2: m2 = 2.
    cd += i128::from(v) * i128::from(e2) + (i128::from(md) << 1);
    ce += i128::from(r) * i128::from(e2) + (i128::from(me) << 1);
    d_out[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[1] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 3: m3 = 0.
    cd += i128::from(v) * i128::from(e3);
    ce += i128::from(r) * i128::from(e3);
    d_out[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[2] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    // Limb 4: m4 = 64.
    cd += i128::from(v) * i128::from(e4) + (i128::from(md) << 6);
    ce += i128::from(r) * i128::from(e4) + (i128::from(me) << 6);
    d_out[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    e_out[3] = (ce as u64 & MASK62) as i64;
    ce >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    verify!(ce == i128::from(ce as i64), "e tail must fit one limb");
    d_out[4] = cd as i64;
    e_out[4] = ce as i64;
    let d_out = Signed62(d_out);
    let e_out = Signed62(e_out);
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(&d_out, "new d");
        checks::assert_coeff_range::<P>(&e_out, "new e");
    }
    (d_out, e_out)
}

/// The terminal coefficient update: only the top row of the matrix, since
/// after the final batch only `d` feeds the result and `e` is dead.
#[allow(clippy::many_single_char_names)]
fn update_d_only_62<P: InvParams>(d: &mut Signed62, e: &Signed62, u: i64, v: i64) {
    let d0 = d.0[0];
    let d1 = d.0[1];
    let d2 = d.0[2];
    let d3 = d.0[3];
    let d4 = d.0[4];
    #[cfg(any(test, debug_assertions))]
    {
        checks::assert_coeff_range::<P>(d, "d");
        checks::assert_coeff_range::<P>(e, "e");
    }
    let sd = d4 >> 63;
    let se = e.0[4] >> 63;
    let mut md = (u & sd) + (v & se);
    let mut cd = i128::from(u) * i128::from(d0) + i128::from(v) * i128::from(e.0[0]);
    md = md.wrapping_sub((P::MU.wrapping_mul(cd as u64).wrapping_add(md as u64) & MASK62) as i64);
    cd += i128::from(P::MODULUS_62[0]) * i128::from(md);
    verify!(cd as u64 & MASK62 == 0, "low bits of new d must vanish");
    cd >>= 62;
    cd += i128::from(u) * i128::from(d1) + i128::from(v) * i128::from(e.0[1]);
    cd += i128::from(P::MODULUS_62[1]) * i128::from(md);
    d.0[0] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d2) + i128::from(v) * i128::from(e.0[2]);
    cd += i128::from(md) << 1;
    d.0[1] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d3) + i128::from(v) * i128::from(e.0[3]);
    d.0[2] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    cd += i128::from(u) * i128::from(d4) + i128::from(v) * i128::from(e.0[4]);
    cd += i128::from(md) << 6;
    d.0[3] = (cd as u64 & MASK62) as i64;
    cd >>= 62;
    verify!(cd == i128::from(cd as i64), "d tail must fit one limb");
    d.0[4] = cd as i64;
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<P>(d, "new d");
}

/// Normalizes `r` from $(-2m, m)$ to $[0, m)$, negating first if `sign` is
/// negative. Upstream's `secp256k1_modinv64_normalize_62` (the mask-based
/// sequence, kept for 1:1 auditability even though this path is
/// variable-time).
fn normalize_62<P: InvParams>(r: &mut Signed62, sign: i64) {
    #[cfg(any(test, debug_assertions))]
    checks::assert_coeff_range::<P>(r, "normalize input");
    let mut r0 = r.0[0];
    let mut r1 = r.0[1];
    let mut r2 = r.0[2];
    let mut r3 = r.0[3];
    let mut r4 = r.0[4];

    // Add the modulus if the input is negative, bringing r to (-m, m); then
    // negate if requested (still (-m, m)). All limbs remain within i64.
    let mut cond_add = r4 >> 63;
    r0 += P::MODULUS_62[0] & cond_add;
    r1 += P::MODULUS_62[1] & cond_add;
    r2 += P::MODULUS_62[2] & cond_add;
    r3 += P::MODULUS_62[3] & cond_add;
    r4 += P::MODULUS_62[4] & cond_add;
    let cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;
    // Propagate the carries.
    r1 += r0 >> 62;
    r0 &= MASK62 as i64;
    r2 += r1 >> 62;
    r1 &= MASK62 as i64;
    r3 += r2 >> 62;
    r2 &= MASK62 as i64;
    r4 += r3 >> 62;
    r3 &= MASK62 as i64;
    // Add the modulus again if the result is still negative, bringing r to
    // [0, m), and propagate once more.
    cond_add = r4 >> 63;
    r0 += P::MODULUS_62[0] & cond_add;
    r1 += P::MODULUS_62[1] & cond_add;
    r2 += P::MODULUS_62[2] & cond_add;
    r3 += P::MODULUS_62[3] & cond_add;
    r4 += P::MODULUS_62[4] & cond_add;
    r1 += r0 >> 62;
    r0 &= MASK62 as i64;
    r2 += r1 >> 62;
    r1 &= MASK62 as i64;
    r3 += r2 >> 62;
    r2 &= MASK62 as i64;
    r4 += r3 >> 62;
    r3 &= MASK62 as i64;

    r.0 = [r0, r1, r2, r3, r4];
    #[cfg(any(test, debug_assertions))]
    {
        use core::cmp::Ordering;
        assert!(
            r0 >> 62 == 0 && r1 >> 62 == 0 && r2 >> 62 == 0 && r3 >> 62 == 0 && r4 >> 62 == 0,
            "normalize output must be canonical"
        );
        assert!(
            checks::mul_cmp_62(r, 5, &Signed62([0; 5]), 0) != Ordering::Less,
            "normalize output must be non-negative"
        );
        assert!(
            checks::mul_cmp_62(r, 5, &Signed62(P::MODULUS_62), 1) == Ordering::Less,
            "normalize output must be reduced"
        );
    }
}

/// Whether the `len` active limbs of `g` are all zero (bottom limb checked
/// first as a fast path, as upstream does).
fn is_zero(g: &Signed62, len: usize) -> bool {
    if g.0[0] != 0 {
        return false;
    }
    let mut cond = 0i64;
    for limb in &g.0[1..len] {
        cond |= limb;
    }
    cond == 0
}

/// Shrinks the active length of `f` and `g` by one when both top limbs are
/// pure sign extensions of the limb below, folding their signs down.
fn shrink_len(f: &mut Signed62, g: &mut Signed62, len: &mut usize) {
    let l = *len;
    let fn_ = f.0[l - 1];
    let gn = g.0[l - 1];
    let mut cond = ((l as i64) - 2) >> 63;
    cond |= fn_ ^ (fn_ >> 63);
    cond |= gn ^ (gn >> 63);
    if cond == 0 {
        f.0[l - 2] = (f.0[l - 2] as u64 | (fn_ as u64) << 62) as i64;
        g.0[l - 2] = (g.0[l - 2] as u64 | (gn as u64) << 62) as i64;
        *len = l - 1;
    }
}

/// Inverts a nonzero field element given by its canonical 4x64-bit
/// little-endian internal (Montgomery) representation, returning the
/// representation of the inverse — see the module docs for why the result is
/// already in Montgomery form. Returns `None` for zero.
pub(super) fn invert<P: InvParams>(x: &[u64; 4]) -> Option<[u64; 4]> {
    invert_counted::<P>(x).map(|(limbs, _)| limbs)
}

/// [`invert`], additionally reporting how many 62-divstep batches ran (used
/// by tests to pin the batch-count distribution).
pub(super) fn invert_counted<P: InvParams>(x: &[u64; 4]) -> Option<([u64; 4], u32)> {
    if x == &[0u64; 4] {
        return None;
    }
    let mut f = Signed62(P::MODULUS_62);
    let mut g = pack62(x);
    let mut d;
    let mut e = Signed62(P::R2_62);
    let mut eta: i64 = -1;
    let mut len: usize = 5;

    // Batch 1, specialized: f = m is sparse, d = 0, and 0 <= e = R^2 < m.
    //
    // This batch can never terminate the algorithm: g' = (q*m + r*g)/2^62 = 0
    // with 0 < g < m would need m | r*g, hence (gcd(g, m) = 1) m | r, hence
    // (|r| <= 2^62 < m) r = 0, and then q*m = 2^62*g' = 0 forces q = 0 —
    // making det T = u*r - v*q = 0, contradicting det T = 2^62. Even if this
    // reasoning were somehow violated, the next batch's divsteps on g0 = 0
    // produce the exact identity matrix scaled by 2^62, and batch 2
    // terminates correctly.
    let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
    eta = new_eta;
    update_fg_62_first::<P>(&mut f, &mut g, &t);
    verify!(
        !is_zero(&g, len),
        "the first divstep batch cannot terminate"
    );
    let (nd, ne) = update_de_62_first::<P>(&e, &t);
    d = nd;
    e = ne;
    #[cfg(any(test, debug_assertions))]
    checks::assert_fg_range::<P>(&f, &g, len);
    shrink_len(&mut f, &mut g, &mut len);

    // 12 * 62 = 744 divsteps upper-bound the requirement of
    // floor((49 * 255 + 57) / 17) = 738 for 255-bit inputs (Bernstein-Yang;
    // upstream documents the analogous 741 for 256-bit inputs).
    for batch in 2..=12u32 {
        let (t, new_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
        eta = new_eta;
        // Update f,g first: if this batch terminates, only the top row of the
        // coefficient update is needed.
        update_fg_62_var(len, &mut f, &mut g, &t);
        if is_zero(&g, len) {
            update_d_only_62::<P>(&mut d, &e, t.u, t.v);
            #[cfg(any(test, debug_assertions))]
            {
                use core::cmp::Ordering;
                let one = Signed62([1, 0, 0, 0, 0]);
                assert!(
                    checks::mul_cmp_62(&f, len, &one, 1) == Ordering::Equal
                        || checks::mul_cmp_62(&f, len, &one, -1) == Ordering::Equal,
                    "f must be +/-1 at termination"
                );
            }
            // The sign of f lives in its top *active* limb.
            normalize_62::<P>(&mut d, f.0[len - 1]);
            return Some((unpack62(&d), batch));
        }
        update_de_62::<P>(&mut d, &mut e, &t);
        #[cfg(any(test, debug_assertions))]
        checks::assert_fg_range::<P>(&f, &g, len);
        shrink_len(&mut f, &mut g, &mut len);
    }
    panic!("safegcd inversion exceeded its 744-divstep bound");
}

#[cfg(test)]
mod tests;
