use std::ops::{Add, Mul};
use lapack::c64;
use lapack::fortran::zgesv;

pub trait Matrix: Clone {
    fn solve_system(self, rhs: Vec<c64>) -> Vec<c64>;

    fn add_num(self, z: c64) -> Self;

    fn mul_num(self, z: c64) -> Self;

    fn mul_vec(self, v: &[c64]) -> Vec<c64>;
}

#[derive(Clone)]
pub struct SymTridiagonalMatrix {
    pub diagonal: Vec<c64>,
    pub off_diagonal: Vec<c64>,
}

impl Matrix for SymTridiagonalMatrix {
    fn solve_system(mut self, mut rhs: Vec<c64>) -> Vec<c64> {
        self.diagonal[0] = 1.0 / self.diagonal[0];
        rhs[0] = self.diagonal[0] * rhs[0];

        for k in 1..self.diagonal.len() {
            self.diagonal[k] = 1.0 / (self.diagonal[k] - self.off_diagonal[k-1] * self.diagonal[k-1] * self.off_diagonal[k-1]);
            rhs[k] = self.diagonal[k] * (rhs[k] - self.off_diagonal[k-1] * rhs[k-1]);
        }

        for k in (0..self.diagonal.len()-1).rev() {
            let tmp = rhs[k+1];
            rhs[k] -= self.diagonal[k] * self.off_diagonal[k] * tmp;
        }

        rhs
    }

    fn add_num(mut self, z: c64) -> Self {
        self.diagonal.iter_mut().for_each(|x| *x += z);
        self
    }

    fn mul_num(mut self, z: c64) -> Self {
        for i in 0..self.diagonal.len() {
            self.diagonal[i] *= z;

            if i > 0 {
                self.off_diagonal[i-1] *= z;
            }
        }

        self
    }

    fn mul_vec(mut self, v: &[c64]) -> Vec<c64> {
        for i in 0..self.diagonal.len() {
            self.diagonal[i] *= v[i];

            if i > 0 {
                self.diagonal[i] += v[i-1] * self.off_diagonal[i-1]
            }

            if i < self.diagonal.len()-1 {
                self.diagonal[i] += v[i+1] * self.off_diagonal[i]
            }
        }

        self.diagonal
    }
}

#[derive(Clone)]
pub struct LowerTriangularConstBandedMatrix {
    pub diagonal: Vec<c64>,
    pub(crate) band_consts: Vec<c64>,
}

impl Matrix for LowerTriangularConstBandedMatrix {
    fn solve_system(mut self, mut rhs: Vec<c64>) -> Vec<c64> {
        let n = self.diagonal.len();

        rhs[0] /= self.diagonal[0];

        for i in 1..n {
            for j in 0..self.band_consts.len().min(i) {
                let temp = rhs[i - j - 1];
                rhs[i] += temp * self.band_consts[j];
            }

            rhs[i] /= self.diagonal[i];
        }

        rhs
    }

    fn add_num(mut self, z: c64) -> Self {
        self.diagonal.iter_mut().for_each(|x| *x += z);
        self
    }

    fn mul_num(mut self, z: c64) -> Self {
        self.diagonal.iter_mut().for_each(|x| *x *= z);
        self.band_consts.iter_mut().for_each(|x| *x *= z);

        self
    }

    fn mul_vec(mut self, v: &[c64]) -> Vec<c64> {
        let n = self.diagonal.len();
        let mut result = vec![0.0.into(); n];

        for i in 0..n {
            for j in 0..self.band_consts.len().min(i) {
                result[i] += v[i - j - 1] * self.band_consts[j];
            }

            result[i] += self.diagonal[i] * v[i];
        }

        result
    }
}

/*
#[derive(Clone)]
pub struct SymConstBandedMatrix {
    pub diagonal: Vec<c64>,
    band_consts: Vec<c64>,
}

impl Matrix for SymConstBandedMatrix {
    fn solve_system(mut self, mut rhs: Vec<c64>) -> Vec<c64> {
        let n = self.diagonal.len();

        for i in 0..n {
            rhs[i] /= self.diagonal[i];
        }

        for i in 1..n {
            rhs[i] -= self.band_consts[0] * rhs[i-1];
        }

        for i in 1..n {
            rhs[n-i-1] -= self.band_consts[1] * rhs[n-i];
        }

        rhs
    }

    fn add_num(mut self, z: c64) -> Self {
        self.diagonal.iter_mut().for_each(|x| *x += z);
        self
    }

    fn mul_num(mut self, z: c64) -> Self {
        self.diagonal.iter_mut().for_each(|x| *x *= z);
        self.band_consts.iter_mut().for_each(|x| *x *= z);

        self
    }

    fn mul_vec(mut self, v: &[c64]) -> Vec<c64> {
        let n = self.diagonal.len();
        let mut result = vec![0.0.into(); n];

        for i in 0..n {
            result[i] += self.diagonal[i] * v[i];

            if i > 0 {
                result[i] += self.band_consts[0] * v[i-1];
            }

            if i < n-1 {
                result[i] += self.band_consts[1] * v[i+1];
            }
        }

        result
    }
}
 */

#[derive(Clone)]
pub struct FullMatrix {
    data: Vec<c64>,
    dims: usize
}

impl Matrix for FullMatrix {
    fn solve_system(mut self, mut rhs: Vec<c64>) -> Vec<c64> {
        let mut ipiv = vec![0; self.dims];

        println!("CALLING zgesv");
        zgesv(
            self.dims as i32,
            1,
            &mut self.data[..],
            self.dims as i32,
            &mut ipiv,
            &mut rhs,
            self.dims as i32,
            &mut 0
        );
        println!("FINISHED zgesv");

        rhs
    }

    fn add_num(mut self, z: c64) -> Self {
        for i in 0..self.dims {
            self.data[i * (self.dims + 1)] += z;
        }

        self
    }

    fn mul_num(mut self, z: c64) -> Self {
        self.data.iter_mut().for_each(|x| *x *= z);

        self
    }

    fn mul_vec(mut self, v: &[c64]) -> Vec<c64> {
        let mut result = vec![0.0.into(); self.dims];

        for i in 0..self.dims {
            for j in 0..self.dims {
                result[i] += self.data[i * self.dims + j] * v[j];
            }
        }

        result
    }
}