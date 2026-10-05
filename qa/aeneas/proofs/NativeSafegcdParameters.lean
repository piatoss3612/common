import NativeCanonical

open Aeneas Std Result

namespace UdonVerify.Native
set_option maxHeartbeats 8000000
set_option maxRecDepth 16384
set_option exponentiation.threshold 1024

abbrev Signed62 := Array I64 5#usize
abbrev OffsetLimbs := Array U64 5#usize
abbrev CorrectionTable := Array A4 12#usize

def signedRadix : Int := 2 ^ 62

def signed62Value (limbs : Signed62) : Int :=
  limbs[0]!.val + signedRadix * limbs[1]!.val + signedRadix ^ 2 * limbs[2]!.val +
    signedRadix ^ 3 * limbs[3]!.val + signedRadix ^ 4 * limbs[4]!.val

def signed62Normalized (limbs : Signed62) : Prop :=
  ∀ k : Fin 4, 0 ≤ limbs[k.val]!.val ∧ limbs[k.val]!.val < signedRadix

def offsetValue (limbs : OffsetLimbs) : Nat :=
  limbs[0]!.val + B * limbs[1]!.val + B ^ 2 * limbs[2]!.val +
    B ^ 3 * limbs[3]!.val + B ^ 4 * limbs[4]!.val

abbrev fpSignedModulus : Signed62 :=
  NativeField.zakura_udon.field.pasta.parameters.PallasBase.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasBase.MODULUS_SIGNED62
abbrev fqSignedModulus : Signed62 :=
  NativeField.zakura_udon.field.pasta.parameters.PallasScalar.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasScalar.MODULUS_SIGNED62
abbrev fpSafegcdOffset : OffsetLimbs :=
  NativeField.zakura_udon.field.pasta.parameters.PallasBase.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasBase.SAFEGCD_OFFSET
abbrev fqSafegcdOffset : OffsetLimbs :=
  NativeField.zakura_udon.field.pasta.parameters.PallasScalar.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasScalar.SAFEGCD_OFFSET
abbrev fpSafegcdCorrections : CorrectionTable :=
  NativeField.zakura_udon.field.pasta.parameters.PallasBase.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasBase.SAFEGCD_CORRECTIONS
abbrev fqSafegcdCorrections : CorrectionTable :=
  NativeField.zakura_udon.field.pasta.parameters.PallasScalar.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasScalar.SAFEGCD_CORRECTIONS

def safegcdParameterChecks (p inverse : Nat) (signed : Signed62)
    (offset : OffsetLimbs) (corrections : CorrectionTable) : Prop :=
  signed62Normalized signed ∧ signed62Value signed = (p : Int) ∧
    0 ≤ signed[4]!.val ∧ signed[4]!.val < 128 ∧ offsetValue offset = p * 2 ^ 63 ∧
    ∀ k : Fin 12, val4 corrections[k.val]! < p ∧
      val4 corrections[k.val]! = (R * 4 ^ (k.val + 1)) % p ∧
      decode p inverse corrections[k.val]! = 4 ^ (k.val + 1) % p

theorem fp_safegcd_parameter_checks :
    safegcdParameterChecks fpPrime fpRadixInverse fpSignedModulus fpSafegcdOffset fpSafegcdCorrections := by
  unfold safegcdParameterChecks signed62Normalized
  with_unfolding_all decide

theorem fq_safegcd_parameter_checks :
    safegcdParameterChecks fqPrime fqRadixInverse fqSignedModulus fqSafegcdOffset fqSafegcdCorrections := by
  unfold safegcdParameterChecks signed62Normalized
  with_unfolding_all decide

#print axioms fp_safegcd_parameter_checks
#print axioms fq_safegcd_parameter_checks

end UdonVerify.Native
