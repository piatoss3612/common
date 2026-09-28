use addition_chain_consumer::Value;

fn main() {
    const SCALAR: u64 = 7;
    let _ = zakura_bento::addition_chain!(Value(7), SCALAR);
}
