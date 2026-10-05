import NativeBits

open Aeneas Std Result

namespace UdonVerify.Native
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

theorem decoded_difference (p inverse a b u : Nat) (hp : 0<p)
    (h : Nat.ModEq p (u+b) a) :
    (u*inverse)%p = ((a*inverse)%p+p-(b*inverse)%p)%p := by
  have hs := h.mul_right inverse
  rw [add_mul] at hs
  have hu := Nat.mod_lt (u*inverse) hp
  have hb := Nat.mod_lt (b*inverse) hp
  have hcanon : Nat.ModEq p ((u*inverse)%p+(b*inverse)%p) ((a*inverse)%p) := by
    unfold Nat.ModEq
    rw [Nat.mod_mod,←Nat.add_mod]
    exact hs
  have hadd : (a*inverse)%p+p-(b*inverse)%p+(b*inverse)%p = (a*inverse)%p+p := by omega
  have htarget : Nat.ModEq p ((a*inverse)%p+p-(b*inverse)%p+(b*inverse)%p) ((a*inverse)%p) := by
    rw [hadd]
    simp [Nat.ModEq,Nat.add_mod]
  have hc := Nat.ModEq.add_right_cancel' ((b*inverse)%p) (hcanon.trans htarget.symm)
  rwa [Nat.ModEq,Nat.mod_eq_of_lt hu] at hc

@[step]
theorem subtract_spec {M S T : Type} (inst : Modulus M)
    (params : PastaParameters (kernelInst inst)) (lhs : Element M S) (rhs : Element M T)
    (ha : val4 lhs.limbs < 2*val4 params.modulus)
    (hb : val4 rhs.limbs < 2*val4 params.modulus) :
    NativeField.zakura_udon.field.pasta.PastaField.sub inst lhs rhs ⦃ out =>
      val4 out.limbs < 2*val4 params.modulus ∧
      Nat.ModEq (val4 params.modulus) (val4 out.limbs+val4 rhs.limbs) (val4 lhs.limbs) ⦄ := by
  unfold NativeField.zakura_udon.field.pasta.PastaField.sub
  rw [subtract_limbs_eq]
  step -grind with (subtract_limbs_spec lhs.limbs rhs.limbs) as ⟨limbs,borrow,hborrow,hsub⟩
  step -grind with (wrapping_neg_bit_spec borrow hborrow) as ⟨mask,hmask⟩
  have htwice : inst.sealedParametersInst.TWICE_MODULUS = ok params.twice := params.twice_ok
  rw [htwice]
  simp only [bind_ok]
  step -grind with Array.index_usize_spec as ⟨m0,hm0⟩
  step -grind with UScalar.and_spec as ⟨w0,hw0,hw0b⟩
  step -grind with Array.index_usize_spec as ⟨m1,hm1⟩
  step -grind with UScalar.and_spec as ⟨w1,hw1,hw1b⟩
  step -grind with Array.index_usize_spec as ⟨m2,hm2⟩
  step -grind with UScalar.and_spec as ⟨w2,hw2,hw2b⟩
  step -grind with Array.index_usize_spec as ⟨m3,hm3⟩
  step -grind with UScalar.and_spec as ⟨w3,hw3,hw3b⟩
  have hmasked : val4 (Array.make 4#usize [w0,w1,w2,w3]) = 2*val4 params.modulus*borrow.val := by
    by_cases hz : borrow.val=0
    · simp only [hz,↓reduceIte] at hmask
      simp_all [val4,and_zero]
    · have hone : borrow.val=1 := by omega
      simp only [hz,↓reduceIte] at hmask
      subst mask
      simp only [and_full] at hw0 hw1 hw2 hw3
      simpa [val4,hw0,hw1,hw2,hw3,hm0,hm1,hm2,hm3,hone] using params.twice_val
  rw [add_limbs_eq]
  step -grind with (add_limbs_spec limbs (Array.make 4#usize [w0,w1,w2,w3]))
    as ⟨sum,carry,hcarry,hsum⟩
  rw [hmasked] at hsum
  have hlimbs := val4_lt limbs
  have hsumlt := val4_lt sum
  have hcert : val4 sum+val4 rhs.limbs = val4 lhs.limbs+2*val4 params.modulus*borrow.val := by
    have hceq : carry.val=borrow.val := by
      norm_num only [R,B] at hlimbs hsumlt hsub hsum
      by_cases hz : borrow.val=0
      · simp only [hz,mul_zero,add_zero] at hsub hsum
        omega
      · have hone : borrow.val=1 := by omega
        simp only [hone,mul_one] at hsub hsum
        omega
    rw [hceq] at hsum
    omega
  have hbound : val4 sum < 2*val4 params.modulus := by
    norm_num only [R,B] at hsub hlimbs
    by_cases hz : borrow.val=0
    · rw [hz] at hcert
      omega
    · have hone : borrow.val=1 := by omega
      rw [hone] at hcert hsub
      omega
  step -grind with (from_montgomery_loose_spec inst params sum hbound) as ⟨out,hout⟩
  refine ⟨by simpa only [hout] using hbound,?_⟩
  unfold Nat.ModEq
  rw [hout,hcert]
  simp [Nat.add_mod,Nat.mul_mod]

@[step]
theorem fp_subtract_spec (lhs rhs : Element Base Loose)
    (ha : val4 lhs.limbs < 2*fpPrime) (hb : val4 rhs.limbs < 2*fpPrime) :
    NativeField.fp_sub lhs rhs ⦃ out => val4 out.limbs < 2*fpPrime ∧
      decode fpPrime fpRadixInverse out.limbs =
        (decode fpPrime fpRadixInverse lhs.limbs+fpPrime-decode fpPrime fpRadixInverse rhs.limbs)%fpPrime ⦄ := by
  unfold NativeField.fp_sub
  step -grind with (subtract_spec fpNativeInst fpNativeParameters lhs rhs ha hb)
    as ⟨out,hbound,hmod⟩
  refine ⟨hbound,?_⟩
  exact decoded_difference fpPrime fpRadixInverse _ _ _ fpParameters.positive hmod

@[step]
theorem fq_subtract_spec (lhs rhs : Element Scalar Loose)
    (ha : val4 lhs.limbs < 2*fqPrime) (hb : val4 rhs.limbs < 2*fqPrime) :
    NativeField.fq_sub lhs rhs ⦃ out => val4 out.limbs < 2*fqPrime ∧
      decode fqPrime fqRadixInverse out.limbs =
        (decode fqPrime fqRadixInverse lhs.limbs+fqPrime-decode fqPrime fqRadixInverse rhs.limbs)%fqPrime ⦄ := by
  unfold NativeField.fq_sub
  step -grind with (subtract_spec fqNativeInst fqNativeParameters lhs rhs ha hb)
    as ⟨out,hbound,hmod⟩
  refine ⟨hbound,?_⟩
  exact decoded_difference fqPrime fqRadixInverse _ _ _ fqParameters.positive hmod

#print axioms fp_subtract_spec
#print axioms fq_subtract_spec

end UdonVerify.Native
