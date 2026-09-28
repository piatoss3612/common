//! Layout and byte-view round trips through the public crate paths.
#![forbid(unsafe_code)]

use zakura_bento as bento;

use bento::{AlignedBytes, MAX_ALIGN, bytes_of, bytes_of_slice};

/// A test record that mixes integer widths without padding.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
struct Record {
    low: u16,
    high: u16,
    wide: u32,
}

#[test]
fn pod_records_round_trip_through_stored_bytes() {
    assert_eq!(size_of::<Record>(), 8);
    assert_eq!(align_of::<Record>(), 4);
    let record = Record {
        low: 0x0201,
        high: 0x0403,
        wide: 0x0807_0605,
    };
    assert_eq!(bytes_of(&record), [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(
        bytes_of(&record).as_ptr(),
        core::ptr::from_ref(&record).cast()
    );
}

/// A record aligned exactly to [`MAX_ALIGN`].
#[derive(Clone, Copy, bento::Pod)]
#[repr(C, align(64))]
struct CacheLine([u64; 8]);

#[test]
fn storage_is_aligned_to_the_advertised_limit() {
    assert_eq!(MAX_ALIGN, 64);
    assert_eq!(size_of::<CacheLine>(), MAX_ALIGN);
    assert_eq!(align_of::<CacheLine>(), MAX_ALIGN);

    static BYTES: AlignedBytes<192> = AlignedBytes([9; 192]);
    assert_eq!(BYTES.0.as_ptr().addr() % MAX_ALIGN, 0);
    let lines: &'static [CacheLine; 3] = BYTES.as_array();
    assert_eq!(core::ptr::from_ref(lines).addr() % MAX_ALIGN, 0);
    assert!(
        lines
            .iter()
            .all(|line| line.0 == [0x0909_0909_0909_0909; 8])
    );

    static ONE: AlignedBytes<64> = AlignedBytes([1; 64]);
    let line: &'static CacheLine = ONE.as_value();
    assert_eq!(bytes_of(line), &ONE.0);
}

#[test]
fn generic_records_and_phantom_markers() {
    use core::marker::PhantomData;

    // The marker implements neither `Pod` nor `Copy`; it is never stored.
    enum Marker {}
    #[repr(transparent)]
    #[derive(bento::Pod)]
    struct Tagged<T: ?Sized> {
        value: [u32; 2],
        marker: PhantomData<T>,
    }
    impl<T: ?Sized> Copy for Tagged<T> {}
    impl<T: ?Sized> Clone for Tagged<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    #[repr(C)]
    #[derive(Clone, Copy, bento::Pod)]
    struct Pair<A, B>(A, B);
    #[repr(C)]
    #[derive(Clone, Copy, bento::Pod)]
    struct Block<T, const N: usize>
    where
        T: Copy,
    {
        values: [T; N],
    }
    #[repr(C)]
    #[derive(Clone, Copy, bento::Pod)]
    struct Unit;

    let tagged = Tagged::<Marker> {
        value: [0x0403_0201, 0x0807_0605],
        marker: PhantomData,
    };
    assert_eq!(bytes_of(&tagged), &[1, 2, 3, 4, 5, 6, 7, 8]);
    let block = Block {
        values: [Pair(0x0201_u16, 0x0403_u16); 2],
    };
    assert_eq!(bytes_of(&block), &[1, 2, 3, 4, 1, 2, 3, 4]);
    assert!(bytes_of(&PhantomData::<[Marker]>).is_empty());
    assert!(bytes_of(&Unit).is_empty());
    assert!(bytes_of_slice(&[Unit; 3]).is_empty());
    assert!(bytes_of_slice::<u64>(&[]).is_empty());
    static EMPTY: AlignedBytes<0> = AlignedBytes([]);
    assert!(bytes_of(EMPTY.as_value::<Unit>()).is_empty());
    assert_eq!(EMPTY.as_array::<Unit, 3>().len(), 3);
    assert_eq!(EMPTY.as_array::<u64, 0>().len(), 0);

    let tagged = Tagged::<str> {
        value: [0; 2],
        marker: PhantomData,
    };
    assert_eq!(bytes_of(&tagged), &[0; 8]);

    // Constructing a padded instantiation does not use the `Pod` contract.
    let padded = Pair(1_u8, 2_u32);
    assert_eq!(padded.1, 2);
}

#[test]
fn file_embedding_preserves_records_arrays_and_empty_layouts() {
    bento::embed_struct! {
        static RECORD: Record = "fixtures/record.bin";
    }
    bento::embed_array! {
        static WORDS: [u32; 2] = "fixtures/record.bin";
    }
    bento::embed_array! {
        static EMPTY: [u64; 0] = "fixtures/empty.bin";
    }
    #[repr(C, align(64))]
    #[derive(Clone, Copy, bento::Pod)]
    struct Empty;
    bento::embed_struct! {
        static ZERO: Empty = "fixtures/empty.bin";
    }
    bento::embed_array! {
        static ZEROS: [Empty; 3] = "fixtures/empty.bin";
    }
    assert_eq!(RECORD.low, 0x0201);
    assert_eq!(RECORD.high, 0x0403);
    assert_eq!(RECORD.wide, 0x0807_0605);
    assert_eq!(*WORDS, [0x0403_0201, 0x0807_0605]);
    assert_eq!(bytes_of(RECORD), bytes_of_slice(WORDS));
    assert!(EMPTY.is_empty());
    assert!(bytes_of(ZERO).is_empty());
    assert_eq!(core::ptr::from_ref(ZERO).addr() % MAX_ALIGN, 0);
    assert_eq!(ZEROS.len(), 3);
    assert!(bytes_of_slice(ZEROS).is_empty());
}
