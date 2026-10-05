module
public import Aeneas
public import NativeField.Types
@[expose] public section

open Aeneas Std Result

/-- Cloning the zero-sized marker preserves its sole value. -/
@[rust_fun
  "core::marker::{core::clone::Clone<core::marker::PhantomData<@T>>}::clone"]
def core.marker.PhantomData.Insts.CoreCloneClone.clone {T : Type}
    (marker : NativeField.core.marker.PhantomData T) :
    Result (NativeField.core.marker.PhantomData T) :=
  ok marker
