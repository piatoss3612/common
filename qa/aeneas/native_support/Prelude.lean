module
public import Aeneas
@[expose] public section

open Aeneas Std

/-- The Rust `Ordering` discriminants, checked against the Charon enum export. -/
instance : Discriminant Ordering I8 where
  read_discriminant
    | .lt => (-1)#i8
    | .eq => 0#i8
    | .gt => 1#i8
