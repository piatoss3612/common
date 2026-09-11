//! Const-evaluated entry points re-exported explicitly by the facade.
//!
//! Defining these in core keeps `$crate` anchored to the implementation without
//! exposing its functions or contexts through the facade. Match fixed argument
//! lists so callers cannot select another implementation item.

/// Parses a `0x`-prefixed, 64-digit hexadecimal integer into four limbs.
///
/// Evaluates
/// [`u256::from_hex`](crate::const_arithmetic::u256::from_hex)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_from_hex {
    ($value:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::from_hex($value) }
    };
}

/// Returns whether `a >= b` as unsigned 256-bit integers.
///
/// Evaluates
/// [`u256::ge`](crate::const_arithmetic::u256::ge)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_ge {
    ($a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::ge($a, $b) }
    };
}

/// Returns the wrapped 256-bit sum and its carry bit.
///
/// Evaluates
/// [`u256::add_with_carry`](crate::const_arithmetic::u256::add_with_carry)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_add_with_carry {
    ($a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::add_with_carry($a, $b) }
    };
}

/// Returns the wrapped 256-bit difference and its borrow bit.
///
/// Evaluates
/// [`u256::sub_with_borrow`](crate::const_arithmetic::u256::sub_with_borrow)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_sub_with_borrow {
    ($a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::sub_with_borrow($a, $b) }
    };
}

/// Subtracts a single-word integer, rejecting underflow.
///
/// Evaluates
/// [`u256::sub_u64`](crate::const_arithmetic::u256::sub_u64)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_sub_u64 {
    ($value:expr, $small:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::sub_u64($value, $small) }
    };
}

/// Shifts an unsigned integer right by fewer than 256 bits.
///
/// Evaluates
/// [`u256::shr`](crate::const_arithmetic::u256::shr)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_shr {
    ($value:expr, $shift:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::shr($value, $shift) }
    };
}

/// Divides an unsigned integer by a nonzero single-word divisor exactly.
///
/// Evaluates
/// [`u256::div_exact_u64`](crate::const_arithmetic::u256::div_exact_u64)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_div_exact_u64 {
    ($value:expr, $divisor:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::div_exact_u64($value, $divisor) }
    };
}

/// Returns the exact eight-limb product of two unsigned 256-bit integers.
///
/// Evaluates
/// [`u256::mul_wide`](crate::const_arithmetic::u256::mul_wide)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_mul_wide {
    ($a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::mul_wide($a, $b) }
    };
}

/// Returns the odd cofactor of `modulus - 1` as an ordinary integer.
///
/// Evaluates
/// [`u256::odd_cofactor`](crate::const_arithmetic::u256::odd_cofactor)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_odd_cofactor {
    ($modulus:expr, $two_adicity:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::odd_cofactor($modulus, $two_adicity) }
    };
}

/// Returns `(t - 1) / 2` for the odd cofactor `t` of `modulus - 1`.
///
/// Evaluates
/// [`u256::tonelli_shanks_exponent`](crate::const_arithmetic::u256::tonelli_shanks_exponent)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_tonelli_shanks_exponent {
    ($modulus:expr, $two_adicity:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::tonelli_shanks_exponent($modulus, $two_adicity) }
    };
}

/// Rounds `value * 2^shift / denominator` to five limbs, with ties upward.
///
/// Evaluates
/// [`u256::round_shifted_ratio`](crate::const_arithmetic::u256::round_shifted_ratio)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! u256_round_shifted_ratio {
    ($denominator:expr, $value:expr, $shift:expr $(,)?) => {
        const { $crate::const_arithmetic::u256::round_shifted_ratio($denominator, $value, $shift) }
    };
}

/// Checks that `modulus` is odd and satisfies `2 < modulus < 2^255`.
///
/// Evaluates
/// [`m255::assert_modulus`](crate::const_arithmetic::m255::assert_modulus)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_assert_modulus {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::assert_modulus($modulus) }
    };
}

/// Adds two reduced residues modulo `modulus`.
///
/// Evaluates
/// [`m255::add`](crate::const_arithmetic::m255::add)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_add {
    ($modulus:expr, $a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::add($modulus, $a, $b) }
    };
}

/// Returns `2^exponent mod modulus` as an ordinary integer.
///
/// Evaluates
/// [`m255::pow2_mod`](crate::const_arithmetic::m255::pow2_mod)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_pow2_mod {
    ($modulus:expr, $exponent:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::pow2_mod($modulus, $exponent) }
    };
}

/// Returns Montgomery one: `2^256 mod modulus`.
///
/// Evaluates
/// [`m255::one`](crate::const_arithmetic::m255::one)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_one {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::one($modulus) }
    };
}

/// Returns the ordinary conversion factor `2^512 mod modulus`.
///
/// Evaluates
/// [`m255::r2`](crate::const_arithmetic::m255::r2)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_r2 {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::r2($modulus) }
    };
}

/// Returns `-low^-1 mod 2^64` for an odd low word.
///
/// Evaluates
/// [`m255::reduction_coefficient`](crate::const_arithmetic::m255::reduction_coefficient)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_reduction_coefficient {
    ($low:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::reduction_coefficient($low) }
    };
}

/// Removes one Montgomery factor from a 512-bit integer below `modulus * 2^256`.
///
/// Evaluates
/// [`m255::reduce_wide`](crate::const_arithmetic::m255::reduce_wide)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_reduce_wide {
    ($modulus:expr, $value:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::reduce_wide($modulus, $value) }
    };
}

/// Returns the reduced Montgomery product `a * b * 2^-256 mod modulus`.
///
/// Evaluates
/// [`m255::mul`](crate::const_arithmetic::m255::mul)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_mul {
    ($modulus:expr, $a:expr, $b:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::mul($modulus, $a, $b) }
    };
}

/// Converts any ordinary 256-bit integer to a reduced Montgomery residue.
///
/// Evaluates
/// [`m255::from_u256`](crate::const_arithmetic::m255::from_u256)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_from_u256 {
    ($modulus:expr, $value:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::from_u256($modulus, $value) }
    };
}

/// Converts any ordinary 64-bit integer to a reduced Montgomery residue.
///
/// Evaluates
/// [`m255::from_u64`](crate::const_arithmetic::m255::from_u64)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_from_u64 {
    ($modulus:expr, $value:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::from_u64($modulus, $value) }
    };
}

/// Decodes a reduced Montgomery residue into an ordinary integer.
///
/// Evaluates
/// [`m255::to_u256`](crate::const_arithmetic::m255::to_u256)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_to_u256 {
    ($modulus:expr, $value:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::to_u256($modulus, $value) }
    };
}

/// Raises a reduced Montgomery residue to an ordinary 256-bit exponent.
///
/// Evaluates
/// [`m255::pow`](crate::const_arithmetic::m255::pow)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_pow {
    ($modulus:expr, $base:expr, $exponent:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::pow($modulus, $base, $exponent) }
    };
}

/// Inverts a nonzero reduced Montgomery residue, assuming a prime modulus.
///
/// Evaluates
/// [`m255::invert_prime`](crate::const_arithmetic::m255::invert_prime)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_invert_prime {
    ($modulus:expr, $value:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::invert_prime($modulus, $value) }
    };
}

/// Derives the Montgomery root `generator^((modulus - 1) / 2^two_adicity)`.
///
/// Evaluates
/// [`m255::two_adic_root_of_unity`](crate::const_arithmetic::m255::two_adic_root_of_unity)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_two_adic_root_of_unity {
    ($modulus:expr, $generator:expr, $two_adicity:expr $(,)?) => {
        const {
            $crate::const_arithmetic::m255::two_adic_root_of_unity(
                $modulus,
                $generator,
                $two_adicity,
            )
        }
    };
}

/// Derives the Montgomery generator `generator^(2^two_adicity)`.
///
/// Evaluates
/// [`m255::odd_order_generator`](crate::const_arithmetic::m255::odd_order_generator)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_odd_order_generator {
    ($modulus:expr, $generator:expr, $two_adicity:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::odd_order_generator($modulus, $generator, $two_adicity) }
    };
}

/// Derives the Montgomery cube root `generator^((modulus - 1) / 3)`.
///
/// Evaluates
/// [`m255::cube_root_of_unity`](crate::const_arithmetic::m255::cube_root_of_unity)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_cube_root_of_unity {
    ($modulus:expr, $generator:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::cube_root_of_unity($modulus, $generator) }
    };
}

/// Returns the reduced Montgomery representation of the inverse of two.
///
/// Evaluates
/// [`m255::two_inverse`](crate::const_arithmetic::m255::two_inverse)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
#[macro_export]
macro_rules! m255_two_inverse {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::two_inverse($modulus) }
    };
}

/// Builds a table of reduced Montgomery inverse powers of two.
///
/// Evaluates
/// [`m255::inverse_powers_of_two`](crate::const_arithmetic::m255::inverse_powers_of_two)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
///
/// The table length is inferred from the result type, or supplied explicitly
/// as `inverse_powers_of_two!(modulus; N)`.
#[macro_export]
macro_rules! m255_inverse_powers_of_two {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::inverse_powers_of_two($modulus) }
    };
    ($modulus:expr; $len:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::inverse_powers_of_two::<{ $len }>($modulus) }
    };
}

/// Builds Montgomery correction factors for batched safegcd inversion.
///
/// Evaluates
/// [`m255::safegcd_corrections_62_64`](crate::const_arithmetic::m255::safegcd_corrections_62_64)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
///
/// The table length is inferred from the result type, or supplied explicitly
/// as `safegcd_corrections_62_64!(modulus; N)`.
#[macro_export]
macro_rules! m255_safegcd_corrections_62_64 {
    ($modulus:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::safegcd_corrections_62_64($modulus) }
    };
    ($modulus:expr; $len:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::safegcd_corrections_62_64::<{ $len }>($modulus) }
    };
}

/// Builds forward and inverse Montgomery root tables indexed by logarithmic order.
///
/// Evaluates
/// [`m255::two_adic_root_tables`](crate::const_arithmetic::m255::two_adic_root_tables)
/// inside `const { ... }`, with the same arguments and result. Panics become
/// compilation errors.
///
/// The table length is inferred from the result type, or supplied explicitly
/// as `two_adic_root_tables!(modulus, generator, two_adicity; N)`.
#[macro_export]
macro_rules! m255_two_adic_root_tables {
    ($modulus:expr, $generator:expr, $two_adicity:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::two_adic_root_tables($modulus, $generator, $two_adicity) }
    };
    ($modulus:expr, $generator:expr, $two_adicity:expr; $len:expr $(,)?) => {
        const { $crate::const_arithmetic::m255::two_adic_root_tables::<{ $len }>($modulus, $generator, $two_adicity) }
    };
}
