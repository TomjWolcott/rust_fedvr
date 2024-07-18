use lapack::c64;
use crate::gauss_quadrature::gauss_lobatto_quadrature;
use crate::{lagrange, lagrange_deriv};

pub trait Basis {
    fn eval_at(&self, i: usize, x: f64) -> f64;

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64;

    fn dims(&self) -> usize;
}

// Don't impl clone, it shouldn't be!
pub struct Grid {
    pub(crate) points: Vec<f64>
}

impl Grid {
    pub fn new(n_x: usize, x_i: f64, x_f: f64) -> Self {
        let dx = (x_f - x_i) / (n_x as f64);
        let points = (0..n_x).map(|i| x_i + (i as f64) * dx).collect();
        Grid { points }
    }

    pub fn delta(&self) -> f64 {
        self.points[1] - self.points[0]
    }
}

impl Basis for Grid {
    fn eval_at(&self, i: usize, x: f64) -> f64 {
        if (x - self.points[i]).abs() < self.delta() / 2.0 { 1.0 } else { 0.0 }
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        let n1 = ((x - self.points[0]) / self.delta()) as usize;
        let n2 = (n1 + 1).min(self.dims() - 1);
        let x_mid = (x - self.points[n1]) / self.delta();

        (1.0 - x_mid) * coefficients[n1] + x_mid * coefficients[n2]
    }

    fn dims(&self) -> usize {
        self.points.len()
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

    pub fn point(&self, index: usize) -> f64 {
        let (q, i) = self.get_indices(index);

        self.points[q][i]
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
            ((index - 1) / num_points, (index - 1) % num_points + 1)
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

        if index != 0 && i == 0 && t < self.points[q][i] {
            lagrange(&self.points[q - 1], num_points - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != self.dims - 1 && i == num_points - 1 && self.points[q][i] < t {
            lagrange(&self.points[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }

    pub fn l_deriv(&self, index: usize, t: f64) -> f64 {
        let (_, num_points) = self.get_dims();
        let (q, i) = self.get_indices(index);

        if (index != 0 && i == 0) || (index != self.dims && i == num_points - 1 && self.points[q][i] < t) {
            0.0
        } else {
            lagrange_deriv(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }
}

impl Basis for LagrangePolynomials {
    fn eval_at(&self, i: usize, x: f64) -> f64 {
        self.l(i, x)
    }

    fn eval(&self, x: f64, coefficients: &[c64]) -> c64 {
        (0..self.dims)
            .map(|i| coefficients[i] * self.l(i, x))
            .fold(0.0.into(), |acc, z| acc + z)
    }

    fn dims(&self) -> usize {
        self.dims
    }
}