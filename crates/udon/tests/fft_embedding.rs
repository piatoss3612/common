//! A downstream owner prepares FFT tables and embeds them in a `no_std` library.

#[path = "support/consumer.rs"]
mod consumer;

use consumer::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn generated_fft_tables_embed_in_a_downstream_consumer() {
    let consumer = Consumer::new("fft-embedding-consumer", "fft_embedding", "udon", &[]);
    for features in ["", "sqrt-table-large"] {
        for (damage, diagnostic) in [
            ("", None),
            (
                "field",
                Some("embedded FFT tables must match the domain: InvalidTables"),
            ),
            (
                "scales",
                Some("embedded residue scales must match the domain: InvalidTables"),
            ),
            (
                "metadata",
                Some("embedded metadata must match the domain: InvalidTables"),
            ),
            (
                "schema",
                Some("embedded metadata must match the domain: InvalidTables"),
            ),
            (
                "kind",
                Some("embedded metadata must match the domain: InvalidTables"),
            ),
            (
                "packed",
                Some("embedded packed twiddles must match the domain: InvalidTables"),
            ),
            (
                "truncate",
                Some("embedded byte length must equal the requested type's size"),
            ),
        ] {
            consumer.check(
                "run",
                features,
                &[("FFT_ARTIFACT_DAMAGE", damage)],
                diagnostic,
                &["src/lib.rs"],
            );
        }
    }
}
