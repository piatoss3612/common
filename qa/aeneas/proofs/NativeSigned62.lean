import NativeSigned62Input

open Aeneas Std Result
open NativeField.zakura_udon.field.pasta.safegcd

namespace UdonVerify.Native
set_option maxHeartbeats 8000000
set_option maxRecDepth 16384
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

theorem packing_updates_correct (limbs : A4) (out : Signed62)
    (d0 d1 d2 d3 d4 : I64)
    (h0 : d0.val = (packed62Digit limbs 0 : Int))
    (h1 : d1.val = (packed62Digit limbs 1 : Int))
    (h2 : d2.val = (packed62Digit limbs 2 : Int))
    (h3 : d3.val = (packed62Digit limbs 3 : Int))
    (h4 : d4.val = (packed62Digit limbs 4 : Int)) :
    ∀ k : Fin 5,
      (((((out.set 0#usize d0).set 1#usize d1).set 2#usize d2).set
        3#usize d3).set 4#usize d4)[k.val]!.val = (packed62Digit limbs k.val : Int) := by
  intro k
  fin_cases k <;> simp_all

theorem packing_update_chain_correct (limbs : A4) (out out0 out1 out2 out3 out4 : Signed62)
    (d0 d1 d2 d3 d4 : I64)
    (hs0 : out0 = out.set 0#usize d0)
    (hs1 : out1 = out0.set 1#usize d1)
    (hs2 : out2 = out1.set 2#usize d2)
    (hs3 : out3 = out2.set 3#usize d3)
    (hs4 : out4 = out3.set 4#usize d4)
    (h0 : d0.val = (packed62Digit limbs 0 : Int))
    (h1 : d1.val = (packed62Digit limbs 1 : Int))
    (h2 : d2.val = (packed62Digit limbs 2 : Int))
    (h3 : d3.val = (packed62Digit limbs 3 : Int))
    (h4 : d4.val = (packed62Digit limbs 4 : Int)) :
    ∀ k : Fin 5, out4[k.val]!.val = (packed62Digit limbs k.val : Int) := by
  rw [hs4, hs3, hs2, hs1, hs0]
  exact packing_updates_correct limbs out d0 d1 d2 d3 d4 h0 h1 h2 h3 h4

@[step]
theorem to_signed62_spec (limbs : A4) (hbound : val4 limbs < 2 ^ 255) :
    to_signed62 limbs ⦃ out => signed62Normalized out ∧
      signed62Value out = (val4 limbs : Int) ∧ 0 ≤ out[4]!.val ∧ out[4]!.val < 128 ⦄ := by
  unfold to_signed62
  rw [packing_loop_unfold]
  step -grind with (packing_body0_spec limbs _) as ⟨flow0, out0, digit0, hflow0, hset0, hdigit0⟩
  rw [hflow0]
  dsimp only
  rw [packing_loop_unfold]
  step -grind with (packing_body1_spec limbs _) as ⟨flow1, out1, digit1, hflow1, hset1, hdigit1⟩
  rw [hflow1]
  dsimp only
  rw [packing_loop_unfold]
  step -grind with (packing_body2_spec limbs _) as ⟨flow2, out2, digit2, hflow2, hset2, hdigit2⟩
  rw [hflow2]
  dsimp only
  rw [packing_loop_unfold]
  step -grind with (packing_body3_spec limbs _) as ⟨flow3, out3, digit3, hflow3, hset3, hdigit3⟩
  rw [hflow3]
  dsimp only
  rw [packing_loop_unfold]
  step -grind with (packing_body4_spec limbs _) as ⟨flow4, out4, digit4, hflow4, hset4, hdigit4⟩
  rw [hflow4]
  dsimp only
  rw [packing_loop_unfold]
  unfold to_signed62_loop.body
  simp only [show ¬((5#usize) < 5#usize) from by decide, ↓reduceIte, bind_ok, WP.spec_ok]
  apply packed62_representation limbs out4 hbound
  exact packing_update_chain_correct limbs (Array.repeat 5#usize 0#i64)
    out0 out1 out2 out3 out4 digit0 digit1 digit2 digit3 digit4
    hset0 hset1 hset2 hset3 hset4 hdigit0 hdigit1 hdigit2 hdigit3 hdigit4

#print axioms to_signed62_spec

end UdonVerify.Native
