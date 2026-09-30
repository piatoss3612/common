fn main() {
    println!("cargo:rustc-check-cfg=cfg(udon_aarch64_asm)");
    println!("cargo:rerun-if-changed=src/asm/pasta_mul-armv8.S");
    #[cfg(feature = "aarch64-asm")]
    build_aarch64_asm();
}

#[cfg(feature = "aarch64-asm")]
fn build_aarch64_asm() {
    use std::env;

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let endian = env::var("CARGO_CFG_TARGET_ENDIAN").unwrap();
    let family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let width = env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap();
    if arch == "aarch64"
        && endian == "little"
        && width == "64"
        && (family.split(',').any(|family| family == "unix") || os == "none")
    {
        cc::Build::new()
            .file("src/asm/pasta_mul-armv8.S")
            .compile("zakura_udon_aarch64");
        println!("cargo:rustc-cfg=udon_aarch64_asm");
    }
}
