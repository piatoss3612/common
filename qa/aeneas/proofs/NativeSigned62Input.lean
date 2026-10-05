import NativeSigned62Math

open Aeneas Std Result
open NativeField.zakura_udon.field.pasta.safegcd

namespace UdonVerify.Native
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

theorem signed62_cast_value (value : U64) (hbound : value.val < 2 ^ 62) :
    (UScalar.hcast .I64 value).val = (value.val : Int) := by
  rw [UScalar.hcast_val_eq]
  change Int.bmod (value.val : Int) (2 ^ 64) = (value.val : Int)
  apply Arith.Int.bmod_pow2_eq_of_inBounds 63 <;> omega

theorem signed62_mask_value :
    (IScalar.hcast .U64 4611686018427387903#i64).val = 2 ^ 62 - 1 := by decide

theorem packing_loop_unfold (limbs : A4) (out : Signed62) (index : Usize) :
    to_signed62_loop limbs out index = (do
      let flow ← to_signed62_loop.body limbs out index
      match flow with
      | .done value => ok value
      | .cont next => to_signed62_loop limbs next.1 next.2) := by
  unfold to_signed62_loop
  rw [loop]
  congr 1
  funext flow
  cases flow with
  | done value => rfl
  | cont next => cases next; rfl

@[step]
theorem packing_body0_spec (limbs : A4) (out : Signed62) :
    to_signed62_loop.body limbs out 0#usize ⦃ flow => ∃ nextOut : Signed62, ∃ digit : I64,
      flow = .cont (nextOut, 1#usize) ∧ nextOut = out.set 0#usize digit ∧
        digit.val = (packed62Digit limbs 0 : Int) ⦄ := by
  unfold to_signed62_loop.body
  simp only [show (0#usize) < 5#usize from by decide, if_pos]
  step -grind as ⟨bit, hbit⟩
  have hbitEq : bit = 0#usize := by scalar_tac
  subst bit
  step -grind as ⟨word, hword⟩
  have hwordEq : word = 0#usize := by scalar_tac
  subst word
  step -grind as ⟨shift, hshift⟩
  have hshiftEq : shift = 0#usize := by scalar_tac
  subst shift
  step -grind as ⟨limb, hlimb⟩
  step -grind with U64.ShiftRight_spec as ⟨value, hvalue, hvalueBV⟩
  simp only [show ((0#usize) != 0#usize) = false from by decide,
    Bool.false_eq_true, ↓reduceIte, bind_ok]
  simp only [SIGNED62_MASK, lift, bind_ok]
  let masked : U64 := value &&& IScalar.hcast .U64 4611686018427387903#i64
  have hmaskedValue : masked.val = packed62Digit limbs 0 := by
    simp only [Nat.shiftRight_eq_div_pow] at hvalue
    norm_num only at hvalue
    simp only [masked, UScalar.val_and, signed62_mask_value, packed62Digit]
    rw [hvalue, hlimb]
    simp only [Nat.div_one]
    simpa [Array.getElem!_Nat_eq, limbs.property] using
      Nat.and_two_pow_sub_one_eq_mod limbs.val[0] 62
  have hmaskedBound : masked.val < 2 ^ 62 := by
    rw [hmaskedValue]
    exact packed62_digit_bound limbs ⟨0, by decide⟩
  step -grind with Array.update_spec as ⟨updated, hupdated⟩
  step -grind as ⟨next, hnext⟩
  have hnextEq : next = 1#usize := by scalar_tac
  subst next
  refine ⟨updated, UScalar.hcast .I64 masked, rfl, hupdated, ?_⟩
  rw [signed62_cast_value masked hmaskedBound, hmaskedValue]

#print axioms signed62_cast_value
#print axioms packing_body0_spec

@[step]
theorem packing_body1_spec (limbs : A4) (out : Signed62) :
    to_signed62_loop.body limbs out 1#usize ⦃ flow => ∃ nextOut : Signed62, ∃ digit : I64,
      flow = .cont (nextOut, 2#usize) ∧ nextOut = out.set 1#usize digit ∧
        digit.val = (packed62Digit limbs 1 : Int) ⦄ := by
  unfold to_signed62_loop.body
  simp only [show (1#usize) < 5#usize from by decide, if_pos]
  step -grind as ⟨bit, hbit⟩
  have hbitEq : bit = 62#usize := by scalar_tac
  subst bit
  step -grind as ⟨word, hword⟩
  have hwordEq : word = 0#usize := by scalar_tac
  subst word
  step -grind as ⟨shift, hshift⟩
  have hshiftEq : shift = 62#usize := by scalar_tac
  subst shift
  step -grind as ⟨limb, hlimb⟩
  step -grind with U64.ShiftRight_spec as ⟨value, hvalue, hvalueBV⟩
  simp only [show ((62#usize) != 0#usize) = true from by decide,
    ↓reduceIte]
  step -grind as ⟨upperIndex, hupperIndex⟩
  have hupperIndexEq : upperIndex = 1#usize := by scalar_tac
  subst upperIndex
  simp only [show (1#usize) < 4#usize from by decide, if_pos]
  step -grind as ⟨upper, hupper⟩
  step -grind as ⟨leftShift, hleftShift⟩
  have hleftShiftEq : leftShift = 2#usize := by scalar_tac
  subst leftShift
  step -grind with U64.ShiftLeft_spec as ⟨high, hhigh, hhighBV⟩
  simp only [SIGNED62_MASK, lift, bind_ok]
  let joined : U64 := value ||| high
  let masked : U64 := joined &&& IScalar.hcast .U64 4611686018427387903#i64
  have hmaskedValue : masked.val = packed62Digit limbs 1 := by
    have hjoin := signed62_join limb.val upper.val ⟨0, by decide⟩ (by scalar_tac)
    simp only [Nat.shiftRight_eq_div_pow] at hvalue
    simp only [Nat.shiftLeft_eq] at hhigh
    norm_num only [U64.size, U64.numBits, UScalarTy.numBits] at hvalue hhigh hjoin
    simp only [masked, joined, UScalar.val_and, UScalar.val_or, signed62_mask_value]
    rw [Nat.and_two_pow_sub_one_eq_mod _ 62]
    simpa [hvalue, hhigh, hlimb, hupper, packed62Digit,
      Array.getElem!_Nat_eq, limbs.property] using hjoin
  have hmaskedBound : masked.val < 2 ^ 62 := by
    rw [hmaskedValue]
    exact packed62_digit_bound limbs ⟨1, by decide⟩
  step -grind with Array.update_spec as ⟨updated, hupdated⟩
  step -grind as ⟨next, hnext⟩
  have hnextEq : next = 2#usize := by scalar_tac
  subst next
  refine ⟨updated, UScalar.hcast .I64 masked, rfl, hupdated, ?_⟩
  rw [signed62_cast_value masked hmaskedBound, hmaskedValue]

#print axioms packing_body1_spec

@[step]
theorem packing_body2_spec (limbs : A4) (out : Signed62) :
    to_signed62_loop.body limbs out 2#usize ⦃ flow => ∃ nextOut : Signed62, ∃ digit : I64,
      flow = .cont (nextOut, 3#usize) ∧ nextOut = out.set 2#usize digit ∧
        digit.val = (packed62Digit limbs 2 : Int) ⦄ := by
  unfold to_signed62_loop.body
  simp only [show (2#usize) < 5#usize from by decide, if_pos]
  step -grind as ⟨bit, hbit⟩
  have hbitEq : bit = 124#usize := by scalar_tac
  subst bit
  step -grind as ⟨word, hword⟩
  have hwordEq : word = 1#usize := by scalar_tac
  subst word
  step -grind as ⟨shift, hshift⟩
  have hshiftEq : shift = 60#usize := by scalar_tac
  subst shift
  step -grind as ⟨limb, hlimb⟩
  step -grind with U64.ShiftRight_spec as ⟨value, hvalue, hvalueBV⟩
  simp only [show ((60#usize) != 0#usize) = true from by decide,
    ↓reduceIte]
  step -grind as ⟨upperIndex, hupperIndex⟩
  have hupperIndexEq : upperIndex = 2#usize := by scalar_tac
  subst upperIndex
  simp only [show (2#usize) < 4#usize from by decide, if_pos]
  step -grind as ⟨upper, hupper⟩
  step -grind as ⟨leftShift, hleftShift⟩
  have hleftShiftEq : leftShift = 4#usize := by scalar_tac
  subst leftShift
  step -grind with U64.ShiftLeft_spec as ⟨high, hhigh, hhighBV⟩
  simp only [SIGNED62_MASK, lift, bind_ok]
  let joined : U64 := value ||| high
  let masked : U64 := joined &&& IScalar.hcast .U64 4611686018427387903#i64
  have hmaskedValue : masked.val = packed62Digit limbs 2 := by
    have hjoin := signed62_join limb.val upper.val ⟨1, by decide⟩ (by scalar_tac)
    simp only [Nat.shiftRight_eq_div_pow] at hvalue
    simp only [Nat.shiftLeft_eq] at hhigh
    norm_num only [U64.size, U64.numBits, UScalarTy.numBits] at hvalue hhigh hjoin
    simp only [masked, joined, UScalar.val_and, UScalar.val_or, signed62_mask_value]
    rw [Nat.and_two_pow_sub_one_eq_mod _ 62]
    simpa [hvalue, hhigh, hlimb, hupper, packed62Digit,
      Array.getElem!_Nat_eq, limbs.property] using hjoin
  have hmaskedBound : masked.val < 2 ^ 62 := by
    rw [hmaskedValue]
    exact packed62_digit_bound limbs ⟨2, by decide⟩
  step -grind with Array.update_spec as ⟨updated, hupdated⟩
  step -grind as ⟨next, hnext⟩
  have hnextEq : next = 3#usize := by scalar_tac
  subst next
  refine ⟨updated, UScalar.hcast .I64 masked, rfl, hupdated, ?_⟩
  rw [signed62_cast_value masked hmaskedBound, hmaskedValue]

#print axioms packing_body2_spec

@[step]
theorem packing_body3_spec (limbs : A4) (out : Signed62) :
    to_signed62_loop.body limbs out 3#usize ⦃ flow => ∃ nextOut : Signed62, ∃ digit : I64,
      flow = .cont (nextOut, 4#usize) ∧ nextOut = out.set 3#usize digit ∧
        digit.val = (packed62Digit limbs 3 : Int) ⦄ := by
  unfold to_signed62_loop.body
  simp only [show (3#usize) < 5#usize from by decide, if_pos]
  step -grind as ⟨bit, hbit⟩
  have hbitEq : bit = 186#usize := by scalar_tac
  subst bit
  step -grind as ⟨word, hword⟩
  have hwordEq : word = 2#usize := by scalar_tac
  subst word
  step -grind as ⟨shift, hshift⟩
  have hshiftEq : shift = 58#usize := by scalar_tac
  subst shift
  step -grind as ⟨limb, hlimb⟩
  step -grind with U64.ShiftRight_spec as ⟨value, hvalue, hvalueBV⟩
  simp only [show ((58#usize) != 0#usize) = true from by decide,
    ↓reduceIte]
  step -grind as ⟨upperIndex, hupperIndex⟩
  have hupperIndexEq : upperIndex = 3#usize := by scalar_tac
  subst upperIndex
  simp only [show (3#usize) < 4#usize from by decide, if_pos]
  step -grind as ⟨upper, hupper⟩
  step -grind as ⟨leftShift, hleftShift⟩
  have hleftShiftEq : leftShift = 6#usize := by scalar_tac
  subst leftShift
  step -grind with U64.ShiftLeft_spec as ⟨high, hhigh, hhighBV⟩
  simp only [SIGNED62_MASK, lift, bind_ok]
  let joined : U64 := value ||| high
  let masked : U64 := joined &&& IScalar.hcast .U64 4611686018427387903#i64
  have hmaskedValue : masked.val = packed62Digit limbs 3 := by
    have hjoin := signed62_join limb.val upper.val ⟨2, by decide⟩ (by scalar_tac)
    simp only [Nat.shiftRight_eq_div_pow] at hvalue
    simp only [Nat.shiftLeft_eq] at hhigh
    norm_num only [U64.size, U64.numBits, UScalarTy.numBits] at hvalue hhigh hjoin
    simp only [masked, joined, UScalar.val_and, UScalar.val_or, signed62_mask_value]
    rw [Nat.and_two_pow_sub_one_eq_mod _ 62]
    simpa [hvalue, hhigh, hlimb, hupper, packed62Digit,
      Array.getElem!_Nat_eq, limbs.property] using hjoin
  have hmaskedBound : masked.val < 2 ^ 62 := by
    rw [hmaskedValue]
    exact packed62_digit_bound limbs ⟨3, by decide⟩
  step -grind with Array.update_spec as ⟨updated, hupdated⟩
  step -grind as ⟨next, hnext⟩
  have hnextEq : next = 4#usize := by scalar_tac
  subst next
  refine ⟨updated, UScalar.hcast .I64 masked, rfl, hupdated, ?_⟩
  rw [signed62_cast_value masked hmaskedBound, hmaskedValue]

#print axioms packing_body3_spec

@[step]
theorem packing_body4_spec (limbs : A4) (out : Signed62) :
    to_signed62_loop.body limbs out 4#usize ⦃ flow => ∃ nextOut : Signed62, ∃ digit : I64,
      flow = .cont (nextOut, 5#usize) ∧ nextOut = out.set 4#usize digit ∧
        digit.val = (packed62Digit limbs 4 : Int) ⦄ := by
  unfold to_signed62_loop.body
  simp only [show (4#usize) < 5#usize from by decide, if_pos]
  step -grind as ⟨bit, hbit⟩
  have hbitEq : bit = 248#usize := by scalar_tac
  subst bit
  step -grind as ⟨word, hword⟩
  have hwordEq : word = 3#usize := by scalar_tac
  subst word
  step -grind as ⟨shift, hshift⟩
  have hshiftEq : shift = 56#usize := by scalar_tac
  subst shift
  step -grind as ⟨limb, hlimb⟩
  step -grind with U64.ShiftRight_spec as ⟨value, hvalue, hvalueBV⟩
  simp only [show ((56#usize) != 0#usize) = true from by decide,
    ↓reduceIte]
  step -grind as ⟨upperIndex, hupperIndex⟩
  have hupperIndexEq : upperIndex = 4#usize := by scalar_tac
  subst upperIndex
  simp only [show ¬((4#usize) < 4#usize) from by decide, ↓reduceIte, bind_ok]
  simp only [SIGNED62_MASK, lift, bind_ok]
  let masked : U64 := value &&& IScalar.hcast .U64 4611686018427387903#i64
  have hmaskedValue : masked.val = packed62Digit limbs 4 := by
    simp only [Nat.shiftRight_eq_div_pow] at hvalue
    norm_num only at hvalue
    have hquotient : limb.val / 2 ^ 56 < 2 ^ 62 := by scalar_tac
    simp only [masked, UScalar.val_and, signed62_mask_value]
    rw [Nat.and_two_pow_sub_one_eq_mod _ 62, hvalue]
    simpa [hlimb, packed62Digit, Array.getElem!_Nat_eq, limbs.property] using
      Nat.mod_eq_of_lt hquotient
  have hmaskedBound : masked.val < 2 ^ 62 := by
    rw [hmaskedValue]
    exact packed62_digit_bound limbs ⟨4, by decide⟩
  step -grind with Array.update_spec as ⟨updated, hupdated⟩
  step -grind as ⟨next, hnext⟩
  have hnextEq : next = 5#usize := by scalar_tac
  subst next
  refine ⟨updated, UScalar.hcast .I64 masked, rfl, hupdated, ?_⟩
  rw [signed62_cast_value masked hmaskedBound, hmaskedValue]

#print axioms packing_body4_spec


end UdonVerify.Native
