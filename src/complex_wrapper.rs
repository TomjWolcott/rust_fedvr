use rgsl::types::ComplexF64;
use std::{iter::Sum, ops::*};
use std::f64::consts::PI;
use std::fmt::{Debug};
use colored::Colorize;
use colors_transform::{Color as ColorTransform, Hsl};
use lapack::c64;
use lapack::fortran::{zgeev, zgesv, zgetrf, zgetri, zstein};

/// This is a wrapper around rgsl::types::ComplexF64, so I can use operator overloading on complex numbers
#[derive(Clone, Copy)]
pub struct Complex(ComplexF64);

#[test]
fn complex_mem_layout() {
    let z1 = Complex::new(1.37, -8.33);
    let z2 = c64::new(1.37, -8.33);
    let v1: Vec<Complex> = (0..10).map(|i| Complex::new(i as f64, -(i as f64))).collect();

    for &z in v1.iter() {
        println!("z1s: {z}");
    }

    let v2: Vec<c64> = unsafe { std::mem::transmute(v1) };

    for z in v2 {
        println!("z2s: {z}");
    }

    unsafe {
        let r1: *const Complex = &z1;
        let r2: *const c64 = &z2;
        println!(
            "{:x?}\n{:x?}",
            std::slice::from_raw_parts(r1 as *const u8, std::mem::size_of::<Complex>()),
            std::slice::from_raw_parts(r2 as *const u8, std::mem::size_of::<c64>())
        );
    }
}

impl Complex {
    pub fn block(&self) -> String {
        let (r, g, b) = self.rgb();
        format!("{}", "██".truecolor(r, g, b))
    }
}

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

impl From<Complex> for c64 {
    fn from(c: Complex) -> Self {
        c64::new(c.dat[0], c.dat[1])
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

impl Add<Complex> for f64 {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Complex {
        Complex(ComplexF64 { dat: [self, 0.0] }) + rhs
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl AddAssign<Complex> for c64 {
    fn add_assign(&mut self, rhs: Complex) {
        *self = *self + <Complex as Into<c64>>::into(rhs)
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

impl Sub<Complex> for f64 {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Complex {
        Complex(ComplexF64 { dat: [self, 0.0] }) - rhs
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
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
    pub fn new(re: f64, im: f64) -> Self {
        Self(ComplexF64 { dat: [re, im] })
    }

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

    pub fn rgb(&self) -> (u8, u8, u8) {
        let hue = (360.0 * (self.arg() / (2.0 * PI) + 0.5) + 180.0) % 360.0;
        let (r, g, b) = if self.magnitude() == 0.0 {
            (40.0, 40.0, 40.0)
        } else {
            Hsl::from(hue as f32, 80.0, 5.0 * (self.magnitude().log10() + 3.0) as f32).to_rgb().as_tuple()
        };

        ((2.55 * r) as u8, (2.55 * g) as u8, (2.55 * b) as u8)
    }

    pub fn rgb_just_hue(&self) -> (u8, u8, u8) {
        let hue = (360.0 * (self.arg() / (2.0 * PI) + 0.5) + 180.0) % 360.0;
        let (r, g, b) = Hsl::from(hue as f32, 80.0, 20.0).to_rgb().as_tuple();

        ((2.55 * r) as u8, (2.55 * g) as u8, (2.55 * b) as u8)
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

#[derive(Clone, Default)]
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

impl Mul<ComplexVector> for ComplexMatrix {
    type Output = ComplexVector;

    fn mul(self, rhs: ComplexVector) -> Self::Output {
        let mut data = vec![ZERO.into(); self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i] += self.data[j * self.cols + i] * rhs.data[j];
            }
        }

        ComplexVector { data }
    }
}

impl ComplexMatrix {
    pub fn solve_systems(&self, b: ComplexVector) -> Result<ComplexVector, LapackError> {
        let mut b = b.data;

        let mut ipiv = vec![0; self.cols];
        let mut info = 0;
        println!("CALLING zgesv");
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
        println!("FINISHED zgesv");

        match info {
            0 => Ok(ComplexVector { data: b }),
            n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
            n => Err(LapackError { info, message: format!("U({},{}) is exactly zero", n, n) }),
        }
    }

    pub fn inverse(mut self) -> Result<Self, LapackError> {
        let mut info = 0;
        let mut ipiv = vec![0; self.cols];
        let mut work = vec![ZERO.into(); 1];

        zgetrf(
            self.rows as i32,
            self.cols as i32,
            &mut self.data[..],
            self.rows as i32,
            &mut ipiv[..],
            &mut info
        );

        if info < 0 {
            return Err(LapackError { info, message: format!("Illegal value in argument {}", -info) });
        } else if info > 0 {
            return Err(LapackError { info, message: format!("U({},{}) is exactly zero", info, info) });
        }

        zgetri(
            self.rows as i32,
            &mut [][..],
            self.rows as i32,
            &mut [][..],
            &mut work[..],
            -1,
            &mut info
        );

        if info != 0 {
            return Err(LapackError { info, message: "Failed to retrieve lwork".to_string() });
        }

        let lwork = work[0].re as i32;
        let mut work = vec![ZERO.into(); lwork as usize];

        zgetri(
            self.rows as i32,
            &mut self.data[..],
            self.rows as i32,
            &mut ipiv[..],
            &mut work[..],
            lwork,
            &mut info
        );

        match info {
            0 => Ok(self),
            n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
            n => Err(LapackError { info, message: format!("U({},{}) is exactly zero", n, n) }),
        }
    }

    pub fn multiply_vector(&self, v: &ComplexVector) -> ComplexVector {
        let mut data = vec![ZERO.into(); self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                data[i] += self.data[j * self.cols + i] * v.data[j];
            }
        }

        ComplexVector { data }
    }

    pub fn eigenvalues(&self) -> Result<Vec<Complex>, LapackError> {
        let mut info = 0;
        let mut eigenvalues = vec![c64::new(0.0, 0.0); self.rows];

        zgeev(
            'N' as u8,
            'N' as u8,
            self.rows as i32,
            &mut self.data.clone()[..],
            self.rows as i32,
            &mut eigenvalues[..],
            &mut [],
            1,
            &mut [],
            1,
            &mut vec![0.0.into(); self.rows * 2],
            (self.rows * 2) as i32,
            &mut vec![0.0.into(); self.rows * 2],
            &mut info
        );

        match info {
            0 => Ok(eigenvalues.iter().map(|&z| Complex::from(z)).collect()),
            n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
            n => Err(LapackError { info, message: format!("The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed; elements and {}+1:{} of W contain eigenvalues which have converged.", n, self.rows) }),
        }
    }

    pub fn eigen_real_sym_tridiagonal(&self) -> Result<Vec<(Complex, ComplexVector)>, LapackError> {
        let mut info = 0;
        let eigenvalues = self.eigenvalues()?;
        let mut eigenvectors = vec![c64::new(0.0, 0.0); self.rows * self.cols];

        zstein(
            self.rows as i32,
            &(0..self.rows).map(|i| self.data[i * self.rows + i].re).collect::<Vec<f64>>()[..],
            &(0..self.rows-1).map(|i| self.data[i * self.rows + i + 1].re).collect::<Vec<f64>>()[..],
            self.cols as i32,
            &mut eigenvalues.iter().map(|z| z.real()).collect::<Vec<f64>>()[..],
            &mut vec![1; self.rows],
            &mut vec![1; self.rows],
            &mut eigenvectors[..],
            self.rows as i32,
            &mut vec![0.0; 5 * self.rows][..],
            &mut vec![0; self.rows][..],
            &mut 0,
            &mut info
        );

        match info {
            0 => Ok(eigenvalues.iter().enumerate().map(
                |(i, &z)| (Complex::from(z), ComplexVector { data: eigenvectors[i * self.rows..(i+1) * self.rows].to_vec() })
            ).collect()),
            n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
            n => Err(LapackError { info, message: format!("The QR algorithm failed to compute all the eigenvalues, and no eigenvectors have been computed; elements and {}+1:{} of W contain eigenvalues which have converged.", n, self.rows) }),
        }
    }

    pub fn print_fancy(&self) {
        for i in 0..self.cols {
            for j in 0..self.rows {
                let z = Complex::from(self.data[i + self.rows * j]);
                let str = match z.magnitude() {
                    ..=0.0 => "┼─",
                    _ => "██"
                };

                let hue = (360.0 * (z.arg() / (2.0 * PI) + 0.5) + 180.0) % 360.0;
                let (r, g, b) = if z.magnitude() == 0.0 {
                    (40.0, 40.0, 40.0)
                } else {
                    Hsl::from(hue as f32, 80.0, 5.0 * (z.magnitude().log10() + 3.0) as f32).to_rgb().as_tuple()
                };

                print!("{}", str.truecolor((2.55 * r) as u8, (2.55 * g) as u8, (2.55 * b) as u8));
            }

            println!();
        }
    }

    pub fn print_fancy_with(&self, v1: &ComplexVector, v2: &ComplexVector) {
        for j in 0..self.rows {
            for i in 0..self.cols + 2 {
                if j == self.rows / 2 && i == self.cols + 1 {
                    print!(" = ");
                } else if j == self.rows / 2 && i == self.cols {
                    print!(" * ");
                } else if i >= self.cols {
                    print!("   ");
                }

                let z = Complex::from(if i == self.cols + 1 {
                    v2.data[j]
                } else if i == self.cols {
                    v1.data[j]
                } else {
                    self.data[i * self.rows + j]
                });

                let str = match z.magnitude() {
                    ..=0.0 => "┼─",
                    _ => "██"
                };

                let (r, g, b) = z.rgb();

                print!("{}", str.truecolor(r, g, b));
            }

            println!();
        }
    }
}