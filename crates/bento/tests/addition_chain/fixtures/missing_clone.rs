struct Value;

impl zakura_bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self
    }
    fn add(&self, _: &Self) -> Self {
        Self
    }
}

fn main() {
    let _ = zakura_bento::addition_chain!(Value, 2);
}
