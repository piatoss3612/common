import NativeDouble

open Aeneas Std Result
open NativeField.zakura_udon.field.pasta.small

namespace UdonVerify.Native
set_option maxHeartbeats 6000000
set_option maxRecDepth 8192
set_option exponentiation.threshold 1024
set_option allowUnsafeReducibility true
attribute [local reducible] WP.Post

theorem small_loop_unfold (K : U64) (limbs : A4) (carry : U64) (index : Usize) :
    multiply_loop K limbs carry index = (do
      let r ← multiply_loop.body K limbs carry index
      match r with
      | .done r => ok r
      | .cont (a,c,i) => multiply_loop K a c i) := by
  unfold multiply_loop
  rw [loop]
  congr 1
  funext r
  cases r with
  | done r => rfl
  | cont r => rcases r with ⟨a,c,i⟩; rfl

@[step]
theorem small_loop_spec (K : U64) (limbs : A4) :
    multiply_loop K limbs 0#u64 0#usize ⦃ out carry =>
      val4 out + R*carry.val = K.val*val4 limbs ⦄ := by
  rw [small_loop_unfold]
  unfold multiply_loop.body
  simp only [show (0#usize)<4#usize from by decide,if_pos]
  try simp only [Std.bind_assoc]
  step -grind as ⟨l0,hl0⟩
  rw [mac_eq]
  step -grind with kernel_mac_identity as ⟨pair0,h0⟩
  rcases pair0 with ⟨r0,c0⟩
  try dsimp only
  try simp only [Std.bind_assoc]
  step -grind as ⟨a0,ha0⟩
  step -grind as ⟨i0,hi0⟩
  have hi0eq : i0=1#usize := by scalar_tac
  subst i0 a0
  try simp only [bind_ok]
  rw [small_loop_unfold]
  unfold multiply_loop.body
  simp only [show (1#usize)<4#usize from by decide,if_pos]
  try simp only [Std.bind_assoc]
  step -grind as ⟨l1,hl1⟩
  rw [mac_eq]
  step -grind with kernel_mac_identity as ⟨pair1,h1⟩
  rcases pair1 with ⟨r1,c1⟩
  try dsimp only
  try simp only [Std.bind_assoc]
  step -grind as ⟨a1,ha1⟩
  step -grind as ⟨i1,hi1⟩
  have hi1eq : i1=2#usize := by scalar_tac
  subst i1 a1
  try simp only [bind_ok]
  rw [small_loop_unfold]
  unfold multiply_loop.body
  simp only [show (2#usize)<4#usize from by decide,if_pos]
  try simp only [Std.bind_assoc]
  step -grind as ⟨l2,hl2⟩
  rw [mac_eq]
  step -grind with kernel_mac_identity as ⟨pair2,h2⟩
  rcases pair2 with ⟨r2,c2⟩
  try dsimp only
  try simp only [Std.bind_assoc]
  step -grind as ⟨a2,ha2⟩
  step -grind as ⟨i2,hi2⟩
  have hi2eq : i2=3#usize := by scalar_tac
  subst i2 a2
  try simp only [bind_ok]
  rw [small_loop_unfold]
  unfold multiply_loop.body
  simp only [show (3#usize)<4#usize from by decide,if_pos]
  try simp only [Std.bind_assoc]
  step -grind as ⟨l3,hl3⟩
  rw [mac_eq]
  step -grind with kernel_mac_identity as ⟨pair3,h3⟩
  rcases pair3 with ⟨r3,c3⟩
  try dsimp only
  try simp only [Std.bind_assoc]
  step -grind as ⟨a3,ha3⟩
  step -grind as ⟨i3,hi3⟩
  have hi3eq : i3=4#usize := by scalar_tac
  subst i3 a3
  try simp only [bind_ok]
  rw [small_loop_unfold]
  unfold multiply_loop.body
  simp only [show ¬((4#usize)<4#usize) from by decide,if_neg,bind_ok,WP.spec_ok]
  simp_all [val4]
  unfold R
  linear_combination h0+B*h1+B^2*h2+B^3*h3

#print axioms small_loop_spec
end UdonVerify.Native
