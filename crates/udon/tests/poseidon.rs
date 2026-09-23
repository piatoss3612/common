//! Sanity checks and value pins for the Pasta Poseidon parameters.

use zakura_udon::{
    field::Field,
    poseidon::{PALLAS_BASE, PALLAS_SCALAR, PoseidonParameters},
};

/// FNV-1a over 128 bits, folding every byte into the running hash.
fn fnv1a_128(bytes: impl Iterator<Item = u8>) -> u128 {
    const OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    bytes.fold(OFFSET_BASIS, |hash, byte| {
        (hash ^ u128::from(byte)).wrapping_mul(PRIME)
    })
}

/// Digest of every table entry as canonical little-endian bytes: the round
/// constants row by row, then the MDS rows. Independent of the Montgomery
/// representation the tables are stored in.
fn digest<F: Field, const T: usize>(parameters: &PoseidonParameters<F, T>) -> u128 {
    fnv1a_128(
        parameters
            .round_constants
            .iter()
            .flatten()
            .chain(parameters.mds.iter().flatten())
            .flat_map(|value| value.to_bytes()),
    )
}

fn assert_well_formed<F: Field, const T: usize>(parameters: &PoseidonParameters<F, T>) {
    assert_eq!(parameters.width(), 5);
    assert_eq!(parameters.rate(), 4);
    assert_eq!(parameters.full_rounds, 8);
    assert_eq!(parameters.partial_rounds, 56);
    assert_eq!(parameters.alpha, 5);
    assert_eq!(parameters.rounds(), 64);
    assert_eq!(parameters.round_constants.len(), 64);

    let all = || {
        parameters
            .round_constants
            .iter()
            .flatten()
            .chain(parameters.mds.iter().flatten())
    };
    assert!(all().all(|value| !value.is_zero()));
    // A transcription slip would most likely repeat or zero an entry.
    let mut seen: Vec<[u8; 32]> = all().map(|value| value.to_bytes()).collect();
    let count = seen.len();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), count);

    assert_eq!(F::NUM_BITS, 255);
}

#[test]
fn pallas_base_parameters_are_well_formed() {
    assert_well_formed(&PALLAS_BASE);
}

#[test]
fn pallas_scalar_parameters_are_well_formed() {
    assert_well_formed(&PALLAS_SCALAR);
}

// The digests pin every one of the 345 entries per field. They were computed
// from tables that matched, entry for entry, the Sage reference captures in
// ragu's `qa/params/reference` at revision 4df34e723c4ee3a1541921aa24821ff86daf4c76
// (`pallas-t5.txt` SHA-256 703f71fd3138e969a74090594fb264c31982623949ec7236c64e92d588942ee1,
// `vesta-t5.txt` SHA-256 b9db55c59b712efa0891cbdb385f6a61566bb1710e66a752e28e761e430e1bbb),
// produced by `daira/pasta-hadeshash` revision 5959f2684a25b372fba347e62467efb00e7e2c3f
// with `generate_parameters_grain.sage 1 0 255 5 8 56 <modulus>`. A table change
// requires re-verifying against that generator before updating a digest; never
// update a digest to make a failing comparison pass.

#[test]
fn pallas_base_tables_match_the_pinned_digest() {
    assert_eq!(digest(&PALLAS_BASE), PALLAS_BASE_DIGEST);
}

#[test]
fn pallas_scalar_tables_match_the_pinned_digest() {
    assert_eq!(digest(&PALLAS_SCALAR), PALLAS_SCALAR_DIGEST);
}

const PALLAS_BASE_DIGEST: u128 = 0x38e9_acb9_6cdd_7395_996b_f0e8_a8f2_cacf;
const PALLAS_SCALAR_DIGEST: u128 = 0x2cba_8835_0552_681a_0f83_ea33_ccb4_049a;

/// Recomputes the digests after a verified table change:
/// `cargo test -p zakura-udon --test poseidon -- --ignored print_digests --nocapture`.
#[test]
#[ignore = "prints the digests for pinning; run explicitly"]
fn print_digests() {
    println!("PALLAS_BASE_DIGEST = {:#034x}", digest(&PALLAS_BASE));
    println!("PALLAS_SCALAR_DIGEST = {:#034x}", digest(&PALLAS_SCALAR));
}
