import NativeRepresentation
import NativeSigned

open Aeneas Std Result

namespace UdonVerify.Native
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

@[step]
theorem from_u128_reduced_spec {M : Type} (inst : Modulus M)
    (params : PastaParameters (kernelInst inst)) (constants : ConversionParameters inst params)
    (inverse : Nat) (hinverse : Nat.ModEq (val4 params.modulus) (R*inverse) 1) (value : U128) :
    NativeField.zakura_udon.field.pasta.PastaField.from_u128 inst reducedInst value
      ⦃ out => val4 out.limbs < val4 params.modulus ∧
        decode (val4 params.modulus) inverse out.limbs=value.val ⦄ := by
  unfold NativeField.zakura_udon.field.pasta.PastaField.from_u128
  simp only [lift,bind_ok]
  step -grind with U128.ShiftRight_IScalar_spec as ⟨high,hhigh,hhighb⟩
  have hvalue : val4 (Array.make 4#usize
      [UScalar.cast .U64 value,UScalar.cast .U64 high,0#u64,0#u64])=value.val := by
    simp [val4,UScalar.cast_val_eq,Nat.shiftRight_eq_div_pow,hhigh,U64.numBits]
    change value.val%B+B*(value.val/B%B)=value.val
    have hhi : value.val / B < B := by norm_num only [B]; scalar_tac
    rw [Nat.mod_eq_of_lt hhi]
    exact Nat.mod_add_div value.val B
  have hbound : val4 (Array.make 4#usize
      [UScalar.cast .U64 value,UScalar.cast .U64 high,0#u64,0#u64]) < val4 params.modulus := by
    rw [hvalue]
    have hp := params.offset_positive
    norm_num only [R,B] at hp
    scalar_tac
  step -grind with (from_canonical_reduced_spec inst params constants inverse hinverse
      (Array.make 4#usize [UScalar.cast .U64 value,UScalar.cast .U64 high,0#u64,0#u64]) hbound)
      as ⟨out,hout,hdecode⟩
  exact ⟨hout,by simpa only [hvalue] using hdecode⟩


@[step]
theorem from_i64_reduced_spec {M : Type} (inst : Modulus M)
    (params : PastaParameters (kernelInst inst)) (constants : ConversionParameters inst params)
    (inverse : Nat) (hinverse : Nat.ModEq (val4 params.modulus) (R*inverse) 1) (value : I64) :
    NativeField.zakura_udon.field.pasta.PastaField.from_i64 inst reducedInst value ⦃ out =>
      val4 out.limbs<val4 params.modulus ∧ decode (val4 params.modulus) inverse out.limbs=
        if value.val<0 then (val4 params.modulus-value.val.natAbs)%val4 params.modulus
        else value.val.natAbs ⦄ := by
  unfold NativeField.zakura_udon.field.pasta.PastaField.from_i64
  step -grind with (unsigned_abs_spec value) as ⟨magnitude,hmagnitude⟩
  step -grind with (from_u64_reduced_spec inst params constants inverse hinverse magnitude)
    as ⟨unsigned,hunsigned,hdecode⟩
  split
  · have hnegative : value.val<0 := by scalar_tac
    step -grind with (negate_spec inst params unsigned (by omega)) as ⟨negative,hneg,hmod⟩
    step -grind with (from_loose_reduced_spec inst params negative.limbs hneg)
      as ⟨out,hcanonical,hout⟩
    have hn := decoded_difference (val4 params.modulus) inverse 0
      (val4 unsigned.limbs) (val4 negative.limbs) params.positive hmod
    simp only [zero_mul,Nat.zero_mod,zero_add] at hn
    change decode (val4 params.modulus) inverse negative.limbs=
      (val4 params.modulus-decode (val4 params.modulus) inverse unsigned.limbs)%val4 params.modulus at hn
    rw [hdecode,hmagnitude] at hn
    refine ⟨hcanonical,?_⟩
    unfold decode
    rw [hout,decoded_reduce]
    simpa only [hnegative,↓reduceIte,decode] using hn
  · have hnonnegative : ¬value.val<0 := by scalar_tac
    simp only [WP.spec_ok]
    exact ⟨hunsigned,by simp only [hnonnegative,↓reduceIte,hdecode,hmagnitude]⟩

@[step]
theorem fp_from_u128_reduced_spec (value : U128) :
    NativeField.zakura_udon.field.pasta.PastaField.from_u128 fpNativeInst reducedInst value
    ⦃ out => val4 out.limbs<fpPrime ∧ decode fpPrime fpRadixInverse out.limbs=value.val ⦄ := by
  step -grind with (from_u128_reduced_spec fpNativeInst fpNativeParameters
    fpConversionParameters fpRadixInverse fp_radix_inverse value) as ⟨out,hbound,hvalue⟩
  exact ⟨hbound,hvalue⟩

#print axioms fp_from_u128_reduced_spec


@[step]
theorem fp_from_i64_reduced_spec (value : I64) :
    NativeField.zakura_udon.field.pasta.PastaField.from_i64 fpNativeInst reducedInst value
    ⦃ out => val4 out.limbs<fpPrime ∧ decode fpPrime fpRadixInverse out.limbs=if value.val<0 then (fpPrime-value.val.natAbs)%fpPrime else value.val.natAbs ⦄ := by
  step -grind with (from_i64_reduced_spec fpNativeInst fpNativeParameters
    fpConversionParameters fpRadixInverse fp_radix_inverse value) as ⟨out,hbound,hvalue⟩
  exact ⟨hbound,hvalue⟩

#print axioms fp_from_i64_reduced_spec


@[step]
theorem fq_from_u128_reduced_spec (value : U128) :
    NativeField.zakura_udon.field.pasta.PastaField.from_u128 fqNativeInst reducedInst value
    ⦃ out => val4 out.limbs<fqPrime ∧ decode fqPrime fqRadixInverse out.limbs=value.val ⦄ := by
  step -grind with (from_u128_reduced_spec fqNativeInst fqNativeParameters
    fqConversionParameters fqRadixInverse fq_radix_inverse value) as ⟨out,hbound,hvalue⟩
  exact ⟨hbound,hvalue⟩

#print axioms fq_from_u128_reduced_spec


@[step]
theorem fq_from_i64_reduced_spec (value : I64) :
    NativeField.zakura_udon.field.pasta.PastaField.from_i64 fqNativeInst reducedInst value
    ⦃ out => val4 out.limbs<fqPrime ∧ decode fqPrime fqRadixInverse out.limbs=if value.val<0 then (fqPrime-value.val.natAbs)%fqPrime else value.val.natAbs ⦄ := by
  step -grind with (from_i64_reduced_spec fqNativeInst fqNativeParameters
    fqConversionParameters fqRadixInverse fq_radix_inverse value) as ⟨out,hbound,hvalue⟩
  exact ⟨hbound,hvalue⟩

#print axioms fq_from_i64_reduced_spec

end UdonVerify.Native
