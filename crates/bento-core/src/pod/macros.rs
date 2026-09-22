//! File embedding macros exported at the crate root.
//!
//! Generated support paths use `$crate` or `::core`, so expansion does not depend
//! on imports or on the name of the caller's dependency. Typed views evaluate
//! layout assertions during const initialization.
//!
//! The aligned backing bytes are borrowed through an anonymous constant so no
//! generated storage name can shadow the caller's length or path expression.

/// Embeds a file as a typed static array without runtime initialization.
///
/// `embed_array! { vis static NAME: [T; LEN] = path; }` declares a
/// `static NAME: &'static [T; LEN]`. The file must contain exactly
/// `size_of::<[T; LEN]>()` bytes, and `T` must pass
/// [`Pod::ASSERT_LAYOUT`](crate::Pod::ASSERT_LAYOUT). Invalid lengths and layouts
/// fail during compilation. Files can be produced with
/// [`bytes_of_slice`](crate::bytes_of_slice).
///
/// The path accepts the same expressions as [`core::include_bytes!`]. A literal
/// is relative to the invoking source file; build scripts can supply files using
/// `concat!(env!("OUT_DIR"), "/records.pod")`.
///
/// Attributes and visibility apply to the typed static. The file must follow
/// the consumer's stored type definition; see [`Pod`](crate::Pod) for the layout
/// contract and the artifact owner's obligations.
#[macro_export]
macro_rules! embed_array {
    ($(#[$attribute:meta])* $vis:vis static $name:ident: [$element:ty; $len:expr] = $path:expr;) => {
        $crate::embed_struct! {
            $(#[$attribute])*
            $vis static $name: [$element; $len] = $path;
        }
    };
}

/// Embeds a file as a typed static value without runtime initialization.
///
/// `embed_struct! { vis static NAME: T = path; }` declares a
/// `static NAME: &'static T`. The file must contain exactly `size_of::<T>()`
/// bytes, and `T` must pass [`Pod::ASSERT_LAYOUT`](crate::Pod::ASSERT_LAYOUT).
/// Invalid lengths and layouts fail during compilation. Files can be produced
/// with [`bytes_of`](crate::bytes_of).
///
/// Paths, attributes, visibility, and the storage contract follow
/// [`embed_array!`](crate::embed_array).
#[macro_export]
macro_rules! embed_struct {
    ($(#[$attribute:meta])* $vis:vis static $name:ident: $record:ty = $path:expr;) => {
        $(#[$attribute])*
        $vis static $name: &'static $record = {
            const { &$crate::AlignedBytes(*::core::include_bytes!($path)) }
                .as_value::<$record>()
        };
    };
}
