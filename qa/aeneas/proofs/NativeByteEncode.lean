import NativeByteDecode

open Aeneas Std Result
open NativeField.zakura_udon.field.pasta.uint.CanonicalUint

namespace UdonVerify.Native
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

def limbBytes (limbs : A4) (index : Nat) : List U8 :=
  (core.num.U64.to_le_bytes limbs[index]!).val

theorem limbBytes_length (limbs : A4) (index : Nat) :
    (limbBytes limbs index).length = 8 := by simp [limbBytes]

theorem limbBytes_value (limbs : A4) (index : Nat) :
    byteValue (limbBytes limbs index) = limbs[index]!.val := by
  have h := bitvec_from_le_value (limbBytes limbs index)
  rw [← h]
  have hmap : (limbBytes limbs index).map U8.bv = limbs[index]!.bv.toLEBytes := by
    simp [limbBytes, core.num.U64.to_le_bytes, List.map_map]
  rw [hmap, BitVec.fromLEBytes_toLEBytes (by decide)]
  simp only [BitVec.toNat_cast]
  rfl

theorem encode_loop_unfold (value : CanonicalUint) (bytes : Bytes32) (index : Usize) :
    to_le_bytes_loop value bytes index = (do
      let result ← to_le_bytes_loop.body value bytes index
      match result with
      | .done result => ok result
      | .cont (next, index) => to_le_bytes_loop value next index) := by
  unfold to_le_bytes_loop
  rw [loop]
  congr 1
  funext result
  cases result with
  | done result => rfl
  | cont result => rcases result with ⟨next, index⟩; rfl

@[step]
theorem encode_body_spec (value : CanonicalUint) (bytes : Bytes32) (index : Usize)
    (hi : index.val < 4) :
    to_le_bytes_loop.body value bytes index ⦃ result =>
      ∃ next : Bytes32, ∃ index' : Usize,
        result = .cont (next, index') ∧ index'.val = index.val + 1 ∧
        next.val = bytes.val.setSlice! (index.val * 8) (limbBytes value.limbs index.val) ⦄ := by
  unfold to_le_bytes_loop.body
  step -grind as ⟨slice, hslice⟩
  have hlen : Slice.len slice = 4#usize := by
    rw [hslice]
    simp [Slice.len]
    scalar_tac
  have hib : index < 4#usize := by scalar_tac
  simp only [hlen, if_pos hib]
  step -grind as ⟨start, hstart⟩
  step -grind as ⟨index', hindex⟩
  step -grind as ⟨finish, hfinish⟩
  step -grind as ⟨target, back, htarget, htargetlen, hback⟩
  step -grind as ⟨word, hword⟩
  step -grind as ⟨encoded, hencoded⟩
  step -grind as ⟨source, hsource⟩
  have hcopylen : source.length = target.length := by
    rw [htargetlen]
    simp only [hsource, Array.length_to_slice]
    scalar_tac
  step -grind as ⟨copied, hcopied⟩
  refine ⟨back copied, index', rfl, hindex, ?_⟩
  rw [hback, hcopied, hsource]
  have hwordbang : word = value.limbs[index.val]! := by
    simp_lists [hword]
  simp [Array.to_slice, hencoded, hwordbang, limbBytes,
    core.num.U64.to_le_bytes, hstart]

theorem encode_body_done (value : CanonicalUint) (bytes : Bytes32) :
    to_le_bytes_loop.body value bytes 4#usize = ok (.done bytes) := by
  unfold to_le_bytes_loop.body
  simp only [lift, bind_ok]
  rw [if_neg (by scalar_tac)]

@[step]
theorem encode_loop_spec (value : CanonicalUint) (bytes : Bytes32) :
    to_le_bytes_loop value bytes 0#usize ⦃ out =>
      out.val = limbBytes value.limbs 0 ++ limbBytes value.limbs 1 ++
        limbBytes value.limbs 2 ++ limbBytes value.limbs 3 ⦄ := by
  rw [encode_loop_unfold]
  step -grind with (encode_body_spec value bytes 0#usize (by decide))
    as ⟨result0, a0, i0, hr0, hi0, h0⟩
  subst result0
  have hi0eq : i0 = 1#usize := by scalar_tac
  subst i0
  dsimp only
  rw [encode_loop_unfold]
  step -grind with (encode_body_spec value a0 1#usize (by decide))
    as ⟨result1, a1, i1, hr1, hi1, h1⟩
  subst result1
  have hi1eq : i1 = 2#usize := by scalar_tac
  subst i1
  dsimp only
  rw [encode_loop_unfold]
  step -grind with (encode_body_spec value a1 2#usize (by decide))
    as ⟨result2, a2, i2, hr2, hi2, h2⟩
  subst result2
  have hi2eq : i2 = 3#usize := by scalar_tac
  subst i2
  dsimp only
  rw [encode_loop_unfold]
  step -grind with (encode_body_spec value a2 3#usize (by decide))
    as ⟨result3, a3, i3, hr3, hi3, h3⟩
  subst result3
  have hi3eq : i3 = 4#usize := by scalar_tac
  subst i3
  dsimp only
  rw [encode_loop_unfold, encode_body_done]
  simp only [bind_ok, WP.spec_ok]
  norm_num only at h0 h1 h2 h3
  rw [h3, h2, h1, h0]
  have hbyteslen : bytes.val.length = 32 := by scalar_tac
  have hfirst : bytes.val.setSlice! 0 (limbBytes value.limbs 0) =
      limbBytes value.limbs 0 ++ bytes.val.drop 8 := by
    simp [List.setSlice!, limbBytes_length, hbyteslen]
  rw [hfirst]
  rw [setSlice_append_suffix (limbBytes value.limbs 0) (limbBytes value.limbs 1)
    (bytes.val.drop 8) 8 (limbBytes_length _ _) (by simp [limbBytes_length, hbyteslen])]
  simp only [limbBytes_length, List.drop_drop]
  norm_num only
  rw [setSlice_append_suffix (limbBytes value.limbs 0 ++ limbBytes value.limbs 1)
    (limbBytes value.limbs 2) (bytes.val.drop 16) 16
    (by simp [limbBytes_length]) (by simp [limbBytes_length, hbyteslen])]
  simp only [limbBytes_length, List.drop_drop]
  norm_num only
  rw [setSlice_append_suffix
    (limbBytes value.limbs 0 ++ limbBytes value.limbs 1 ++ limbBytes value.limbs 2)
    (limbBytes value.limbs 3) (bytes.val.drop 24) 24
    (by simp [limbBytes_length]) (by simp [limbBytes_length, hbyteslen])]
  simp [limbBytes_length, List.drop_drop, hbyteslen]

@[step]
theorem uint_to_bytes_spec (value : CanonicalUint) :
    to_le_bytes value ⦃ out => byteValue out.val = val4 value.limbs ⦄ := by
  unfold to_le_bytes
  step -grind with (encode_loop_spec value (Array.repeat 32#usize 0#u8))
    as ⟨out, hout⟩
  rw [hout]
  simp only [byteValue_append, List.length_append, limbBytes_length, limbBytes_value]
  norm_num [B, val4]

#print axioms uint_to_bytes_spec

end UdonVerify.Native
