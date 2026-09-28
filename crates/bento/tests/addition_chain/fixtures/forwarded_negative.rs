use addition_chain_consumer::Value;

macro_rules! scale {
    ($scalar:literal) => {
        zakura_bento::addition_chain!(Value(7), $scalar)
    };
}

fn main() {
    let _ = scale!(-1);
}
