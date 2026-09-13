//! An owner embeds both table kinds and entry layouts for both curves.

#[path = "support/consumer.rs"]
mod consumer;

use consumer::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn generated_curve_tables_embed_in_a_downstream_consumer() {
    let consumer = Consumer::new("curve-embedding-consumer", "curve_embedding", "udon", &[]);
    for features in ["", "alloc,sqrt-table-large"] {
        for (damage, diagnostic) in [
            ("", None),
            (
                "srs-root",
                Some("embedded SRS metadata must match the curve and domain"),
            ),
            (
                "srs-order",
                Some("embedded SRS must match natural Lagrange order"),
            ),
            (
                "coordinate",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "point",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "order",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "carry",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "base",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "schema",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "curve",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "window",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "cache",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "compact-cache",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "compact-order",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "compact-cached-order",
                Some("embedded table must match its base: InvalidTable"),
            ),
            (
                "table-kind",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "entry-layout",
                Some("embedded metadata must match the curve and layout"),
            ),
            (
                "truncate",
                Some("embedded byte length must equal the requested type's size"),
            ),
        ] {
            consumer.check(
                "run",
                features,
                &[("CURVE_ARTIFACT_DAMAGE", damage)],
                diagnostic,
                &["src/lib.rs", "src/record.rs"],
            );
        }
    }
}
