//! Checks storage contracts through the compiler, including deferred assertions.
//!
//! Expansion unit tests can inspect the emitted assertions, but only full builds
//! evaluate them for concrete types. Cases here use Rust tokens for readability;
//! the harness writes them as source at the compiler boundary.
#![forbid(unsafe_code)]

use std::{fs, path::PathBuf};

use proc_macro2::TokenStream;
use quote::quote;

mod support;
use support::{cargo, diagnostics};

struct Consumer {
    directory: PathBuf,
}

impl Consumer {
    fn new() -> Self {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let directory = workspace.join("target/pod-consumers");
        fs::create_dir_all(directory.join("src/bin")).unwrap();
        let facade = workspace.join("crates/bento");
        fs::write(
            directory.join("Cargo.toml"),
            format!(
                "[package]\nname = \"pod-compile-tests\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\
                 [workspace]\n[dependencies]\nbento = {{ package = \"zakura-bento\", path = {facade:?} }}\n"
            ),
        )
        .unwrap();

        // Keep consumer dependency versions consistent with the workspace so
        // their sources are available for offline builds.
        fs::copy(workspace.join("Cargo.lock"), directory.join("Cargo.lock")).unwrap();
        fs::write(directory.join("record.pod"), [1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        fs::write(directory.join("empty.pod"), []).unwrap();
        Self { directory }
    }

    fn build(&self, name: &str, source: TokenStream, error: Option<&str>) {
        fs::write(
            self.directory.join(format!("src/bin/{name}.rs")),
            quote!(#![forbid(unsafe_code)] #source).to_string(),
        )
        .unwrap();
        let output = cargo(
            &self.directory,
            &["build", "--release", "--quiet", "--bin", name],
        );
        let diagnostic = diagnostics(&output);
        match error {
            Some(expected) => {
                assert!(!output.status.success(), "{name} unexpectedly compiled");
                assert!(
                    diagnostic.contains(expected),
                    "{name} failed for the wrong reason; expected {expected:?}:\n{diagnostic}"
                );
                assert!(
                    diagnostic.contains(&format!("src/bin/{name}.rs:")),
                    "{diagnostic}"
                );
            }
            None => assert!(output.status.success(), "{name} failed:\n{diagnostic}"),
        }
    }
}

#[test]
fn pod_contract_is_enforced_during_codegen() {
    let consumer = Consumer::new();
    padded_layouts_are_checked_at_storage_operations(&consumer);
    invalid_definitions_are_reported_by_derive(&consumer);
    excessive_alignment_is_rejected(&consumer);
    static_views_require_exact_byte_lengths(&consumer);
}

#[rustfmt::skip]
fn padded_layouts_are_checked_at_storage_operations(consumer: &Consumer) {
    let pair = quote! {
        #[repr(C)]
        #[derive(Clone, Copy, bento::Pod)]
        struct Pair<A, B>(A, B);
        type Bad = Pair<u8, u32>;
    };
    consumer.build(
        "ordinary_padded_value",
        quote! {
            #pair
            fn main() {
                std::hint::black_box(Pair(1_u8, 2_u32));
            }
        },
        None,
    );
    for (name, body) in [
        ("shadowed_assert_macro", quote! {
            macro_rules! assert { ($($tokens:tt)*) => {}; }
            fn main() {
                std::hint::black_box(bento::bytes_of(&Pair(1_u8, 2_u32)));
            }
        }),
        ("explicit_assertion", quote! {
            const _: () = <Bad as bento::Pod>::ASSERT_LAYOUT;
            fn main() {}
        }),
        ("bytes_of", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of(&Pair(1_u8, 2_u32)));
            }
        }),
        ("bytes_of_slice", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of_slice(&[Pair(1_u8, 2_u32)]));
            }
        }),
        ("empty_slice", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of_slice::<Bad>(&[]));
            }
        }),
        ("zero_array", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of::<[Bad; 0]>(&[]));
            }
        }),
        ("nested_array", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of(&Pair([Pair(1_u8, 2_u32)], 3_u32)));
            }
        }),
        ("nested_zero_array", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of(&Pair([] as [Bad; 0], 3_u32)));
            }
        }),
        ("as_value", quote! {
            static B: bento::AlignedBytes<8> = bento::AlignedBytes([0; 8]);
            fn main() {
                std::hint::black_box(B.as_value::<Bad>());
            }
        }),
        ("as_array", quote! {
            static B: bento::AlignedBytes<8> = bento::AlignedBytes([0; 8]);
            fn main() {
                std::hint::black_box(B.as_array::<Bad, 1>());
            }
        }),
        ("as_empty_array", quote! {
            static B: bento::AlignedBytes<0> = bento::AlignedBytes([]);
            fn main() {
                std::hint::black_box(B.as_array::<Bad, 0>());
            }
        }),
        ("const_view", quote! {
            static B: bento::AlignedBytes<8> = bento::AlignedBytes([0; 8]);
            static V: &Bad = B.as_value();
            fn main() {
                std::hint::black_box(V);
            }
        }),
        ("trailing_padding", quote! {
            fn main() {
                std::hint::black_box(bento::bytes_of(&Pair(1_u32, 2_u8)));
            }
        }),
        ("embed_struct", quote! {
            bento::embed_struct! {
                static RECORD: Bad = concat!(env!("CARGO_MANIFEST_DIR"), "/record.pod");
            }
            fn main() {}
        }),
        ("embed_array", quote! {
            bento::embed_array! {
                static RECORDS: [Bad; 1] = concat!(env!("CARGO_MANIFEST_DIR"), "/record.pod");
            }
            fn main() {}
        }),
        ("embed_empty_array", quote! {
            bento::embed_array! {
                static RECORDS: [Bad; 0] = concat!(env!("CARGO_MANIFEST_DIR"), "/empty.pod");
            }
            fn main() {}
        }),
    ] {
        consumer.build(name, quote!(#pair #body), Some("Pod struct must have no padding"));
    }
}

#[rustfmt::skip]
fn invalid_definitions_are_reported_by_derive(consumer: &Consumer) {
    for (name, definition, expected) in [
        (
            "unspecified",
            quote!(struct Record(u32);),
            "Pod requires repr(C) or repr(transparent)",
        ),
        (
            "packed",
            quote!(#[repr(C, packed)] struct Record(u32);),
            "Pod supports only repr(C)",
        ),
        (
            "enumeration",
            quote!(#[repr(C)] enum Record { A, B }),
            "Pod requires a struct",
        ),
        (
            "union",
            quote!(#[repr(C)] union Record { x: u32, y: u32 }),
            "Pod requires a struct",
        ),
        ("bool_field", quote!(#[repr(C)] struct Record(bool);), "bool: Pod"),
        ("char_field", quote!(#[repr(C)] struct Record(char);), "char: Pod"),
        ("usize_field", quote!(#[repr(C)] struct Record(usize);), "usize: Pod"),
        ("u128_field", quote!(#[repr(C)] struct Record(u128);), "u128: Pod"),
        (
            "nonzero_field",
            quote!(#[repr(C)] struct Record(core::num::NonZeroU32);),
            "NonZero<u32>: Pod",
        ),
        (
            "reference_field",
            quote!(#[repr(C)] struct Record(&'static u32);),
            "&'static u32: Pod",
        ),
        (
            "pointer_field",
            quote!(#[repr(C)] struct Record(*const u32);),
            "*const u32: Pod",
        ),
        (
            "unknown_option",
            quote!(#[pod(unknown)] #[repr(C)] struct Record(u32);),
            "expected `crate = path`",
        ),
        (
            "duplicate_path",
            quote!(#[pod(crate = bento, crate = bento)] #[repr(C)] struct Record(u32);),
            "duplicate Pod crate path",
        ),
    ] {
        consumer.build(
            name,
            quote! {
                #[derive(Clone, Copy, bento::Pod)]
                #definition
                fn main() {}
            },
            Some(expected),
        );
    }
}

fn excessive_alignment_is_rejected(consumer: &Consumer) {
    consumer.build(
        "over_aligned",
        quote! {
            #[repr(C, align(128))]
            #[derive(Clone, Copy, bento::Pod)]
            struct Record([u8; 128]);
            fn main() {
                std::hint::black_box(bento::bytes_of(&Record([0; 128])));
            }
        },
        Some("over-aligned Pod type"),
    );
}

#[rustfmt::skip]
fn static_views_require_exact_byte_lengths(consumer: &Consumer) {
    for (name, source) in [
        ("wrong_embed_array_length", quote! {
            bento::embed_array! {
                static WORDS: [u32; 3] = concat!(env!("CARGO_MANIFEST_DIR"), "/record.pod");
            }
            fn main() {}
        }),
        ("wrong_embed_value_length", quote! {
            bento::embed_struct! {
                static WORD: u32 = concat!(env!("CARGO_MANIFEST_DIR"), "/record.pod");
            }
            fn main() {}
        }),
        ("wrong_value_length", quote! {
            static B: bento::AlignedBytes<8> = bento::AlignedBytes([0; 8]);
            static V: &u32 = B.as_value();
            fn main() {}
        }),
        ("wrong_array_length", quote! {
            static B: bento::AlignedBytes<8> = bento::AlignedBytes([0; 8]);
            static V: &[u32; 3] = B.as_array();
            fn main() {}
        }),
    ] {
        consumer.build(
            name,
            source,
            Some("embedded byte length must equal the requested type's size"),
        );
    }
}
