#[derive(Clone)]
struct Value;

fn main() {
    // Even the scalar-one case must require `AdditionChain`.
    let _ = zakura_bento::addition_chain!(Value, 1);
}
