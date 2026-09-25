#![forbid(unsafe_code)]
#![deny(warnings)]

mod pass;
mod reject_curve;
mod reject_fft;
mod reject_field;

fn main() {
    pass::run();
    reject_field::check::<arithmetic::field::PallasBase>();
    reject_field::check::<arithmetic::field::PallasScalar>();
    reject_curve::check::<arithmetic::curve::Pallas>();
    reject_curve::check::<arithmetic::curve::Vesta>();
    reject_fft::check();
}
