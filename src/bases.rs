use std::f64::consts::PI;
use lapack::c64;
use lapack::fortran::dgesv;
use crate::gauss_quadrature::gauss_lobatto_quadrature;
use crate::{factorial, lagrange, lagrange_deriv, sample};
use crate::lapack_wrapper::display_matrix;

pub trait Basis: Clone {
    fn eval_at(&self, i: usize, x: f64) -> c64;

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64;

    fn len(&self) -> usize;

    fn weight(&self, i: usize) -> f64;
}

#[derive(Clone)]
pub struct HarmonicEigenfunctions(pub usize);

impl HarmonicEigenfunctions {
    pub fn eigenfunction(n: usize, x: f64) -> f64 {
        PI.powf(-0.25) * (-x*x/2.0).exp() * (0..n/2)
            .map(|m| {
                // The factorials have to be unwrapped and calculated like this to avoid: inf / inf = NaN
                let a=(-1f64).powi(m as i32) * (1..=n).map(|k| {
                    let b = (k as f64 / 2.0).sqrt() * // sqrt(n!) / sqrt(2^n)
                        if k <= m { 1.0 / (k as f64) } else { 1.0 } * // m!
                        if k <= n - 2*m { 2.0 * x / (k as f64) } else { 1.0 }; // 2^(n-2m) / (n-2m)!


                    println!("    k={k}: {b}");

                    b
                }).product::<f64>();

                println!("  m={m}: {a}");

                a
            }).sum::<f64>()
    }

    pub fn eigenvalues(&self) -> Vec<c64> {
        (0..self.len()).map(|i| c64::from(i as f64 + 0.5)).collect()
    }
}

impl Basis for HarmonicEigenfunctions {
    fn eval_at(&self, i: usize, x: f64) -> c64 {
        Self::eigenfunction(i, x).into()
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        coefficients.iter().enumerate()
            .map(|(i, &c)| c * Self::eigenfunction(i, x))
            .fold(0.0.into(), |acc, z| acc + z)
    }

    fn len(&self) -> usize {
        self.0
    }

    fn weight(&self, _i: usize) -> f64 {
        1.0
    }
}

#[derive(Clone)]
pub struct Grid {
    pub(crate) points: Vec<f64>
}

impl Grid {
    pub fn new(n_x: usize, x_i: f64, x_f: f64) -> Self {
        let dx = (x_f - x_i) / (n_x as f64);
        let points = (0..n_x).map(|i| x_i + (i as f64) * dx).collect();
        Grid { points }
    }

    pub fn point(&self, index: usize) -> f64 {
        self.points[index]
    }

    pub fn delta(&self) -> f64 {
        self.points[1] - self.points[0]
    }
}

impl Basis for Grid {
    fn eval_at(&self, i: usize, x: f64) -> c64 {
        (if (x - self.points[i]).abs() < self.delta() / 2.0 { 1.0 } else { 0.0 }).into()
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        let n1 = ((x - self.points[0]) / self.delta()) as usize;
        let n2 = (n1 + 1).min(self.len() - 1);
        let x_mid = (x - self.points[n1]) / self.delta();

        (1.0 - x_mid) * coefficients[n1] + x_mid * coefficients[n2]
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn weight(&self, _i: usize) -> f64 {
        self.delta()
    }
}

#[derive(Clone)]
pub struct BackwardGrid {
    pub(crate) points: Vec<f64>,
    bands: usize
}

impl BackwardGrid {
    pub fn new(n_x: usize, x_i: f64, x_f: f64, bands: usize) -> Self {
        let dx = (x_f - x_i) / (n_x as f64 - 1.0);
        let points = (0..n_x).map(|i| x_i + (i as f64) * dx).collect();
        Self { points, bands }
    }

    pub fn point(&self, index: usize) -> f64 {
        self.points[index]
    }

    pub fn delta(&self) -> f64 {
        self.points[1] - self.points[0]
    }

    /// 3-8 bands are supported, coefs from wikipedia: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    pub fn second_deriv_coefficients(&self) -> Vec<f64> {
        match self.bands {
            3 => vec![1.0, -2.0, 1.0],
            4 => vec![2.0, -5.0, 4.0, -1.0],
            5 => vec![35.0/12.0, -26.0/3.0, 19.0/2.0, -14.0/3.0, 11.0/12.0],
            6 => vec![15.0/4.0, -77.0/6.0, 107.0/6.0, -13.0, 61.0/12.0, -5.0/6.0],
            7 => vec![203.0/45.0, -87.0/5.0, 117.0/4.0, -254.0/9.0, 33.0/2.0, -27.0/5.0, 137.0/180.0],
            8 => vec![469.0/90.0, -223.0/10.0, 879.0/20.0, -949.0/18.0, 41.0, -201.0/10.0, 1019.0/180.0, -7.0/10.0],
            _ => panic!("Unsupported number of bands: {}", self.bands)
        }
    }
}

impl Basis for BackwardGrid {
    fn eval_at(&self, i: usize, x: f64) -> c64 {
        (if (x - self.points[i]).abs() < self.delta() / 2.0 { 1.0 } else { 0.0 }).into()
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        let n1 = ((x - self.points[0]) / self.delta()) as usize;
        let n2 = (n1 + 1).min(self.len() - 1);
        let x_mid = (x - self.points[n1]) / self.delta();

        (1.0 - x_mid) * coefficients[n1] + x_mid * coefficients[n2]
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn weight(&self, _i: usize) -> f64 {
        self.delta()
    }
}

#[derive(Default, Clone)]
pub struct LagrangePolynomials {
    points: Vec<Vec<f64>>,
    weights: Vec<f64>,
    pub(crate) dims: usize
}

impl LagrangePolynomials {
    pub fn new(num_points: usize, num_intervals: usize, start: f64, end: f64) -> Self {
        let delta = (end - start) / (num_intervals as f64);

        let quad_points = gauss_lobatto_quadrature(num_points, start, start + delta);

        let points = (0..num_intervals)
            .map(|n| quad_points.iter().map(|&(t, _)| t + n as f64 * delta).collect())
            .collect();

        let weights = quad_points.iter().map(|&(_, w)| w).collect();
        let dims = num_points * num_intervals - num_intervals + 1;

        Self { points, weights, dims }
    }

    pub fn is_bridge(&self, index: usize) -> bool {
        let (num_intervals, num_points) = self.get_dims();
        let (q, i) = self.get_indices(index);

        (q != 0 && i == 0) || (q != num_intervals-1 && i == num_points-1)
    }

    pub fn both_bridges(&self, index1: usize, index2: usize) -> bool {
        let (q1, i1) = self.get_indices(index1);
        let (q2, i2) = self.get_indices(index2);

        q1 == q2 && i1 == i2 && self.is_bridge(index1)
    }

    pub fn point(&self, index: usize) -> f64 {
        let (q, i) = self.get_indices(index);

        self.points[q][i]
    }

    pub fn get_weight(&self, index: usize) -> f64 {
        let (_, i) = self.get_indices(index);
        self.weights[i]
    }

    pub fn get_dims(&self) -> (usize, usize) {
        (self.points.len(), self.weights.len())
    }

    pub fn get_indices(&self, index: usize) -> (usize, usize) {
        let (num_intervals, num_points) = self.get_dims();

        if index == 0 {
            (0, 0)
        } else if index == self.dims - 1 {
            (num_intervals - 1, num_points - 1)
        } else {
            ((index - 1) / (num_points - 1), (index - 1) % (num_points - 1) + 1)
        }
    }

    pub fn into_basis_fns(self) -> Box<dyn Fn(usize, f64) -> f64> {
        Box::new(move |index, t| {
            self.l(index, t)
        })
    }

    pub fn l(&self, index: usize, t: f64) -> f64 {
        let (_, num_points) = self.get_dims();
        let (q, i) = self.get_indices(index);

        if self.is_bridge(index) && t < self.point(index) {
            lagrange(&self.points[q], i, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if self.is_bridge(index) && self.point(index) < t {
            lagrange(&self.points[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }

    pub fn l_deriv(&self, index: usize, t: f64) -> f64 {
        let (_, num_points) = self.get_dims();
        let (q, i) = self.get_indices(index);

        if self.is_bridge(index) && t < self.point(index) {
            lagrange_deriv(&self.points[q], i, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if self.is_bridge(index) && self.point(index) < t {
            lagrange_deriv(&self.points[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange_deriv(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }
}

#[test]
fn simple_lagrange_poly_int() {
    let n = 100;
    let nq = 10;
    let polys = LagrangePolynomials::new(n, nq, 0.0, 5.0);
    let g = |x: f64| -2.0*x*(-x*x).exp();
    let f_expected = |x: f64| (-x*x).exp();
    let init = 1.0;

    let mut matrix = vec![0.0; polys.dims*polys.dims];
    let mut rhs = vec![0.0; polys.dims];

    matrix[0] = 1.0;
    rhs[0] = init;

    for j in 1..polys.dims {
        let mult = if polys.is_bridge(j) { 2.0 } else { 1.0 };
        rhs[j] = mult * g(polys.point(j));

        for i in 0..polys.dims {
            let both_mult = if polys.both_bridges(j, i) { 0.0 } else { 1.0 };

            matrix[i * polys.dims + j] = both_mult * polys.l_deriv(i, polys.point(j));
        }
    }

    // display_matrix(&matrix.iter().map(|x| x.into()).collect::<Vec<c64>>()[..], polys.dims);

    let mut info = 0;

    dgesv(
        polys.dims as i32,
        1,
        &mut matrix[..],
        polys.dims as i32,
        &mut vec![0; polys.dims],
        &mut rhs[..],
        polys.dims as i32,
        &mut info
    );

    match info {
        0 => (),
        n @ ..=-1 => panic!("Illegal value in argument {}", -n),
        n => panic!("U({},{}) is exactly zero", n, n),
    };

    let sol = rhs.iter().map(|&x| x.into()).collect::<Vec<c64>>();

    let mut max_err: f64 = 0.0;

    println!("f(________) =       Expected       vs       Computed      ");

    sample(1000, 0.0, 5.0, |x| {
        let expected = f_expected(x);
        let computed = polys.eval(x, &sol[..]);
        let err = (computed - expected).norm();
        max_err = max_err.max(err);

        println!("f({:6.4}) = {: ^20} vs {: ^20} -- err: {err:.5e}", x, format!("{:.6}", expected), format!("{:.6}", computed));
        // assert!(err < 1e-10);
    });

    println!("Max error: {:.5e}", max_err);
}

#[test]
fn simple_lagrange_poly_approx() {
    let n = 100;
    let nq = 10;

    let (start, end) = (0.0, 1.0);
    let polys = LagrangePolynomials::new(n, nq, start, end);
    let expected: fn(f64) -> c64 = |x| (-x).exp().into();

    let coefs: Vec<c64> = (0..polys.dims).map(|i| {
        expected(polys.point(i))
    }).collect();

    let mut max_err: f64 = 0.0;

    println!("f(________) =       Expected       vs       Computed      ");
    sample(1000, start, end, |x| {
        let expected = expected(x);
        let computed = polys.eval(x, &coefs);
        let err = (computed - expected).norm();
        max_err = max_err.max(err);

        println!("f({:6.4}) = {: ^20} vs {: ^20} -- err: {err:.5e}", x, format!("{:.6}", expected), format!("{:.6}", computed));
        assert!(err < 1e-10);
    });

    println!("Max error: {:.5e}", max_err);


}

impl Basis for LagrangePolynomials {
    fn eval_at(&self, i: usize, x: f64) -> c64 {
        self.l(i, x).into()
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        (0..self.dims)
            .map(|i| coefficients[i] * self.l(i, x))
            .fold(0.0.into(), |acc, z| acc + z)
    }

    fn len(&self) -> usize {
        self.dims
    }

    fn weight(&self, i: usize) -> f64 {
        self.weights[i]
    }
}