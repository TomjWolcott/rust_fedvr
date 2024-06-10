use rgsl::types::ComplexF64;
use std::{iter::Sum, ops::*};
use std::fmt::Debug;
use lapack::c64;
use lapack::fortran::zgesv;

/// This is a wrapper around rgsl::types::ComplexF64, so I can use operator overloading on complex numbers
#[derive(Clone, Copy)]
pub struct Complex(ComplexF64);

impl Debug for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4} + {:.4}i", self.0.real(), self.0.imaginary())
    }
}

pub const I: Complex = Complex(ComplexF64 { dat: [0.0, 1.0] });
#[allow(dead_code)]
pub const E: Complex = Complex(ComplexF64 {
    dat: [std::f64::consts::E, 0.0],
});

#[allow(dead_code)]
pub const ONE: Complex = Complex(ComplexF64 { dat: [1.0, 0.0] });
#[allow(dead_code)]
pub const ZERO: Complex = Complex(ComplexF64 { dat: [0.0, 0.0] });

impl Into<c64> for Complex {
    fn into(self) -> c64 {
        c64::new(self.0.dat[0], self.0.dat[1])
    }
}

impl From<f64> for Complex {
    fn from(f: f64) -> Self {
        Self(ComplexF64 { dat: [f, 0.0] })
    }
}

impl From<c64> for Complex {
    fn from(c: c64) -> Self {
        Self(ComplexF64 { dat: [c.re, c.im] })
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        -1.0 * self
    }
}

impl Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self(self.0.mul_real(rhs))
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Complex {
        rhs * self
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self(self.0.div_real(rhs))
    }
}

impl Div<Complex> for f64 {
    type Output = Complex;

    fn div(self, rhs: Complex) -> Complex {
        Complex(ComplexF64 { dat: [self, 0.0] }) / rhs
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self(self.0.div(&rhs.0))
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(&rhs.0))
    }
}

impl Add<f64> for Complex {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Self(self.0.add_real(rhs))
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl Sum for Complex {
    fn sum<I: Iterator<Item = Complex>>(iter: I) -> Self {
        iter.fold(Complex(ComplexF64 { dat: [0.0, 0.0] }), |a, b| a + b)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(&rhs.0))
    }
}

impl Sub<f64> for Complex {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Self(self.0.sub_real(rhs))
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(&rhs.0))
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.4} + {:.4}i", self.0.real(), self.0.imaginary())
    }
}

impl Complex {
    pub fn pow(self, n: Self) -> Self {
        Self(self.0.pow(&n.0))
    }

    pub fn magnitude(self) -> f64 {
        (self.0.dat[0] * self.0.dat[0] + self.0.dat[1] * self.0.dat[1]).sqrt()
    }

    pub fn exp(self) -> Self {
        Self(self.0.exp())
    }

    pub fn sin(self) -> Self {
        Self(self.0.sin())
    }

    pub fn cos(self) -> Self {
        Self(self.0.cos())
    }

    pub fn tan(self) -> Self {
        Self(self.0.tan())
    }

    pub fn sinh(self) -> Self {
        Self(self.0.sinh())
    }

    pub fn cosh(self) -> Self {
        Self(self.0.cosh())
    }

    pub fn tanh(self) -> Self {
        Self(self.0.tanh())
    }

    pub fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    pub fn log10(self) -> Self {
        Self(self.0.log10())
    }

    pub fn arg(self) -> f64 {
        self.0.arg()
    }

    pub fn real(self) -> f64 {
        self.0.real()
    }

    pub fn imag(self) -> f64 {
        self.0.imaginary()
    }

    pub fn conj(self) -> Self {
        Self(self.0.conjugate())
    }
}

impl Deref for Complex {
    type Target = ComplexF64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Complex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
pub struct ComplexVector {
    pub data: Vec<c64>,
}

impl Debug for ComplexVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data.iter().map(|c| Complex(ComplexF64 { dat: [c.re, c.im] })).collect::<Vec<Complex>>())
    }
}

#[derive(Clone)]
pub struct ComplexMatrix {
    pub data: Vec<c64>,
    pub rows: usize,
    pub cols: usize,
}

impl Debug for ComplexMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                s.push_str(&format!("{: ^24}", format!("{:.4} + {:.4}i ", self.data[j * self.cols + i].re, self.data[j * self.cols + i].im)));
            }
            s.push_str("\n");
        }

        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub struct LapackError {
    pub info: i32,
    pub message: String,
}

impl ComplexMatrix {
    pub fn solve_systems(&self, b: ComplexVector) -> Result<ComplexVector, LapackError> {
        let mut b = b.data;

        let mut ipiv = vec![0; self.cols];
        let mut info = 0;

        zgesv(
            self.cols as i32,
            1,
            &mut self.data.clone(),
            self.rows as i32,
            &mut ipiv,
            &mut b,
            self.cols as i32,
            &mut info
        );

        match info {
            0 => Ok(ComplexVector { data: b }),
            n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
            n => Err(LapackError { info, message: format!("U({},{}) is exactly zero", n, n) }),
        }
    }
}