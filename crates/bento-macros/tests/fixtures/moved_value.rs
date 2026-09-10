use facade_default::Value;

fn main() {
    let value = Value(7);
    let _ = zakura_bento::addition_chain!(value, 2);
    drop(value);
}
