//! Fixed Pasta vectors, complemented by independent integer derivations.

use bento::const_arithmetic::U256;

use super::{PallasBase, PallasScalar, PastaField, PrimeModulus};

// Fixed literals pin root orientation as well as algebraic identities.
struct Vectors {
    modulus: U256,
    montgomery_inv: u64,
    r: U256,
    r2: U256,
    b448: U256,
    sqrt_exponent: U256,
    two_inverse: U256,
    root_of_unity: U256,
    root_of_unity_inverse: U256,
    delta: U256,
    zeta: U256,
    zeta_inverse: U256,
}

/// The Pallas base field (the Vesta scalar field).
const PALLAS_BASE: Vectors = Vectors {
    modulus: [
        0x992d_30ed_0000_0001,
        0x2246_98fc_094c_f91b,
        0,
        0x4000_0000_0000_0000,
    ],
    montgomery_inv: 0x992d_30ec_ffff_ffff,
    r: [
        0x3478_6d38_ffff_fffd,
        0x992c_350b_e419_14ad,
        0xffff_ffff_ffff_ffff,
        0x3fff_ffff_ffff_ffff,
    ],
    r2: [
        0x8c78_ecb3_0000_000f,
        0xd7d3_0dbd_8b0d_e0e7,
        0x7797_a99b_c3c9_5d18,
        0x096d_41af_7b9c_b714,
    ],
    b448: [
        0x9b98_58f2_94cf_91ba,
        0x8635_bd2c_4252_b065,
        0x496d_41af_7b9c_b714,
        0x1b4b_3c4b_ffff_fffc,
    ],
    sqrt_exponent: [
        0x04a6_7c8d_cc96_9876,
        0x0000_0000_1123_4c7e,
        0,
        0x0000_0000_2000_0000,
    ],
    two_inverse: [
        0x66d2_cf12_ffff_ffff,
        0xddb9_6703_f6b3_06e4,
        0xffff_ffff_ffff_ffff,
        0x3fff_ffff_ffff_ffff,
    ],
    root_of_unity: [
        0xa28d_b849_bad6_dbf0,
        0x9083_cd03_d3b5_39df,
        0xfba6_b9ca_9dc8_448e,
        0x3ec9_2874_7b89_c6da,
    ],
    root_of_unity_inverse: [
        0x5cfe_5f67_cb15_5442,
        0x1764_60c2_734c_4621,
        0xdf81_0016_4521_4110,
        0x1804_7fb9_f910_68bc,
    ],
    delta: [
        0x5965_e9af_9d65_1171,
        0xc7c5_9c1b_b222_e936,
        0x272a_caec_59b6_a78c,
        0x08eb_004e_7903_b751,
    ],
    zeta: [
        0x0202_1cf6_619a_153d,
        0x9e8c_2697_4980_b78e,
        0x2a67_6d5c_c87a_4666,
        0x15d8_049d_a7a1_7876,
    ],
    zeta_inverse: [
        0xfbdf_d7aa_9e65_eac8,
        0x0cd4_d654_e500_25fb,
        0xd598_92a3_3785_b99a,
        0x2a27_fb62_585e_8789,
    ],
};

/// The Pallas scalar field (the Vesta base field).
const PALLAS_SCALAR: Vectors = Vectors {
    modulus: [
        0x8c46_eb21_0000_0001,
        0x2246_98fc_0994_a8dd,
        0,
        0x4000_0000_0000_0000,
    ],
    montgomery_inv: 0x8c46_eb20_ffff_ffff,
    r: [
        0x5b2b_3e9c_ffff_fffd,
        0x992c_350b_e342_0567,
        0xffff_ffff_ffff_ffff,
        0x3fff_ffff_ffff_ffff,
    ],
    r2: [
        0xfc96_78ff_0000_000f,
        0x67bb_433d_891a_16e3,
        0x7fae_2310_04cc_f590,
        0x096d_41af_7ccf_daa9,
    ],
    b448: [
        0xcc92_0bb9_994a_8dd9,
        0x87a7_dcbe_1ff6_e0d7,
        0x496d_41af_7ccf_daa9,
        0x0ee4_537b_ffff_fffc,
    ],
    sqrt_exponent: [
        0x04ca_546e_c623_7590,
        0x0000_0000_1123_4c7e,
        0,
        0x0000_0000_2000_0000,
    ],
    two_inverse: [
        0x73b9_14de_ffff_ffff,
        0xddb9_6703_f66b_5722,
        0xffff_ffff_ffff_ffff,
        0x3fff_ffff_ffff_ffff,
    ],
    root_of_unity: [
        0x2180_7742_8c99_42de,
        0xcc49_5789_21b6_0494,
        0xac2e_5d27_b2ef_bee2,
        0x0b79_fa89_7f2d_b056,
    ],
    root_of_unity_inverse: [
        0xb990_773d_23d2_2e85,
        0x4ece_919a_03f3_c012,
        0x2c9a_c8b9_aa3b_a50b,
        0x364d_5dfa_434d_9efa,
    ],
    delta: [
        0xfea3_5ff4_7ef3_ee87,
        0xc982_cbdd_717f_d9c6,
        0x0d25_c7ce_50b8_ab59,
        0x17bf_fd4c_c7ac_17c1,
    ],
    zeta: [
        0x7c54_1a84_8011_1122,
        0x4063_0b9c_56ed_29da,
        0x02c2_75fb_135b_2b29,
        0x121d_29f8_8824_5b10,
    ],
    zeta_inverse: [
        0x410e_7d20_7fee_eee3,
        0x6afd_f14f_d8fa_2279,
        0xfd3d_8a04_eca4_d4d7,
        0x2de2_d607_77db_a4ef,
    ],
};

fn check_field<M: PrimeModulus>(vectors: &Vectors, sqrt_exponent: [u64; 4]) {
    assert_eq!(M::MODULUS, vectors.modulus);
    assert_eq!(M::MONTGOMERY_INV, vectors.montgomery_inv);
    assert_eq!(M::R, vectors.r);
    assert_eq!(M::R2, vectors.r2);
    assert_eq!(M::B448, vectors.b448);
    assert_eq!(sqrt_exponent, vectors.sqrt_exponent);
    assert_eq!(M::TWO_INVERSE, vectors.two_inverse);
    assert_eq!(
        PastaField::<M>::root_of_unity(32)
            .unwrap()
            .montgomery_limbs(),
        vectors.root_of_unity,
    );
    assert_eq!(
        PastaField::<M>::root_of_unity_inverse(32)
            .unwrap()
            .montgomery_limbs(),
        vectors.root_of_unity_inverse,
    );
    assert_eq!(M::DELTA, vectors.delta);
    assert_eq!(M::ZETA, vectors.zeta);
    assert_eq!(M::ZETA_INVERSE, vectors.zeta_inverse);
}

#[test]
fn pallas_base_constants_match_fixed_vectors() {
    check_field::<PallasBase>(&PALLAS_BASE, super::parameters::PALLAS_BASE_SQRT_EXPONENT);
}

#[test]
fn pallas_scalar_constants_match_fixed_vectors() {
    check_field::<PallasScalar>(
        &PALLAS_SCALAR,
        super::parameters::PALLAS_SCALAR_SQRT_EXPONENT,
    );
}
