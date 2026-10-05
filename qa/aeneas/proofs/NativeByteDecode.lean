import NativeByteMath
import NativeBytePartition

open Aeneas Std Result
open NativeField.zakura_udon.field.pasta.uint.CanonicalUint

namespace UdonVerify.Native
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

abbrev Bytes32 := Array U8 32#usize

def chunkValue (bytes : Bytes32) (index : Nat) : Nat :=
  byteValue (bytes.val.slice (index * 8) ((index + 1) * 8))

theorem decode_loop_unfold (bytes : Bytes32) (limbs : A4) (index : Usize) :
    from_le_bytes_loop bytes limbs index = (do
      let result ← from_le_bytes_loop.body bytes limbs index
      match result with
      | .done result => ok result
      | .cont (next, index) => from_le_bytes_loop bytes next index) := by
  unfold from_le_bytes_loop
  rw [loop]
  congr 1
  funext result
  cases result with
  | done result => rfl
  | cont result => rcases result with ⟨next, index⟩; rfl

@[step]
theorem decode_body_spec (bytes : Bytes32) (limbs : A4) (index : Usize)
    (hi : index.val < 4) :
    from_le_bytes_loop.body bytes limbs index ⦃ result =>
      ∃ next : A4, ∃ index' : Usize,
        result = .cont (next, index') ∧ index'.val = index.val + 1 ∧
        ∀ j : Nat, j < 4 → next[j]!.val =
          if j = index.val then chunkValue bytes j else limbs[j]!.val ⦄ := by
  unfold from_le_bytes_loop.body
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
  step -grind as ⟨chunk, hchunk, hchunklen⟩
  step -grind as ⟨r, hr⟩
  have hchunk8 : chunk.length = 8 := by omega
  cases r with
  | Err err => cases err; simp only at hr; omega
  | Ok a =>
    simp only at hr
    simp only [core.result.Result.unwrap, bind_ok]
    step -grind as ⟨word, hword⟩
    have hwordval : word.val = chunkValue bytes index.val := by
      have hw : word.val = byteValue a.val := by
        simpa only [UScalar.val, BitVec.toNat_cast, bitvec_from_le_value] using
          congrArg BitVec.toNat hword
      rw [hw, hr.1, hchunk]
      simp [Array.to_slice, chunkValue, hstart, hfinish, hindex]
    step -grind as ⟨next, hnext⟩
    refine ⟨next, index', rfl, hindex, ?_⟩
    intro j hj
    rw [hnext]
    by_cases hji : j = index.val
    · subst j
      have hset : (limbs.set index word)[index.val]! = word := by
        apply Array.getElem!_Nat_set_eq
        exact ⟨rfl, by simpa using hi⟩
      simp only [hset, if_true, hwordval]
    · simp only [if_neg hji, Array.getElem!_Nat_set_ne _ _ _ _ (Ne.symm hji)]

theorem decode_body_done (bytes : Bytes32) (limbs : A4) :
    from_le_bytes_loop.body bytes limbs 4#usize = ok (.done limbs) := by
  unfold from_le_bytes_loop.body
  simp only [lift, bind_ok]
  rw [if_neg (by scalar_tac)]

@[step]
theorem decode_loop_spec (bytes : Bytes32) (limbs : A4) :
    from_le_bytes_loop bytes limbs 0#usize ⦃ out =>
      ∀ j : Nat, j < 4 → out[j]!.val = chunkValue bytes j ⦄ := by
  rw [decode_loop_unfold]
  step -grind with (decode_body_spec bytes limbs 0#usize (by decide))
    as ⟨result0, a0, i0, hr0, hi0, h0⟩
  subst result0
  simp only [bind_ok]
  have hi0eq : i0 = 1#usize := by scalar_tac
  subst i0
  rw [decode_loop_unfold]
  step -grind with (decode_body_spec bytes a0 1#usize (by decide))
    as ⟨result1, a1, i1, hr1, hi1, h1⟩
  subst result1
  simp only [bind_ok]
  have hi1eq : i1 = 2#usize := by scalar_tac
  subst i1
  rw [decode_loop_unfold]
  step -grind with (decode_body_spec bytes a1 2#usize (by decide))
    as ⟨result2, a2, i2, hr2, hi2, h2⟩
  subst result2
  simp only [bind_ok]
  have hi2eq : i2 = 3#usize := by scalar_tac
  subst i2
  rw [decode_loop_unfold]
  step -grind with (decode_body_spec bytes a2 3#usize (by decide))
    as ⟨result3, a3, i3, hr3, hi3, h3⟩
  subst result3
  simp only [bind_ok]
  have hi3eq : i3 = 4#usize := by scalar_tac
  subst i3
  rw [decode_loop_unfold, decode_body_done]
  simp only [bind_ok, WP.spec_ok]
  intro j hj
  rw [h3 j hj, h2 j hj, h1 j hj, h0 j hj]
  have hjcases : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 := by omega
  rcases hjcases with rfl | rfl | rfl | rfl <;> norm_num

@[step]
theorem uint_from_bytes_spec (bytes : Bytes32) :
    from_le_bytes bytes ⦃ out => val4 out.limbs = byteValue bytes.val ⦄ := by
  unfold from_le_bytes
  step -grind with (decode_loop_spec bytes (Array.repeat 4#usize 0#u64))
    as ⟨limbs, hlimbs⟩
  rw [byteValue_four_chunks]
  simp only [val4]
  rw [hlimbs 0 (by decide), hlimbs 1 (by decide), hlimbs 2 (by decide),
    hlimbs 3 (by decide)]
  simp [chunkValue]

#print axioms uint_from_bytes_spec

end UdonVerify.Native
