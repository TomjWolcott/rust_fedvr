use crate::complex_wrapper::Complex;

struct ApproxOdeOptions {
    num_intervals: usize,
    interval_size: f64,
    num_quad_points: usize
}

// pub fn approx_ode<IN, OUT>(options: ApproxOdeOptions, in_0: IN, out_0: OUT) -> impl Fn(IN) -> OUT {
//
// }