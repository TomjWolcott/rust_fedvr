use std::f64::consts::PI;
use colored::Colorize;
use colors_transform::{Color, Hsl};
use lapack::c64;
use lapack::fortran::{zgesv, zgetrf, zgetrf2, zgetri};
use crate::complex_wrapper::{LapackError};

pub fn solve_systems(mut matrix: Vec<c64>, dims: usize, mut vector: Vec<c64>) -> Result<Vec<c64>, LapackError> {
    let mut ipiv = vec![0; dims];
    let mut info = 0;

    println!("CALLING zgesv");
    zgesv(
        dims as i32,
        1,
        &mut matrix,
        dims as i32,
        &mut ipiv,
        &mut vector,
        dims as i32,
        &mut info
    );
    println!("FINISHED zgesv");

    match info {
        0 => Ok(vector),
        n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
        n => Err(LapackError { info, message: format!("U({},{}) is exactly zero", n, n) }),
    }
}

pub fn complex_to_rgb(z: c64) -> (u8, u8, u8) {
    let hue = (360.0 * (z.arg() / (2.0 * PI) + 0.5) + 180.0) % 360.0;
    let (r, g, b) = if z.norm() == 0.0 {
        (40.0, 40.0, 40.0)
    } else {
        Hsl::from(hue as f32, 80.0, 5.0 * (z.norm().log10() + 3.0) as f32).to_rgb().as_tuple()
    };

    ((2.55 * r) as u8, (2.55 * g) as u8, (2.55 * b) as u8)
}

pub fn complex_to_rgb_just_hue(z: c64) -> (u8, u8, u8) {
    let hue = (360.0 * (z.arg() / (2.0 * PI) + 0.5) + 180.0) % 360.0;
    let (r, g, b) = Hsl::from(hue as f32, 80.0, 20.0).to_rgb().as_tuple();

    ((2.55 * r) as u8, (2.55 * g) as u8, (2.55 * b) as u8)
}

pub fn display_matrix(matrix: &[c64], dims: usize) {
    for i in 0..dims {
        for j in 0..dims {
            let z = matrix[i + j * dims];
            let (r, g, b) = complex_to_rgb(z);
            let str = match z.norm_sqr() {
                ..=0.0 => "┼─",
                _ => "██"
            };

            print!("{}", str.truecolor(r, g, b));
        }
        println!();
    }
}

pub fn display_special_matrix(beta: c64, matrices: &[c64], vector: &[c64], dims: usize, num_blocks: usize) {
    for i in 0..dims*num_blocks {
        for j in 0..=dims*num_blocks {
            let z = if j == dims*num_blocks {
                if i == dims*num_blocks/2 { print!(" * ?? = "); } else { print!("   ??   "); }
                vector[i]
            } else if i / dims == j / dims {
                matrices[(j%dims)*dims + (i%dims) + (i/dims)*dims*dims]
            } else if i == j + dims || i + dims == j {
                beta
            } else {
                0.0.into()
            };

            let (r, g, b) = complex_to_rgb(z);
            let str = match z.norm_sqr() {
                ..=0.0 => "┼─",
                _ => "██"
            };

            print!("{}", str.truecolor(r, g, b));
        }

        println!();
    }
}

pub fn display_other_special_matrix(off_diagonal_diagonals: &[c64], diagonal_blocks: &[c64], vector: &[c64], dims: usize, num_blocks: usize) {
    for i in 0..dims*num_blocks {
        for j in 0..=dims*num_blocks {
            let z = if j == dims*num_blocks {
                if i == dims*num_blocks/2 { print!(" * ?? = "); } else { print!("   ??   "); }
                vector[i]
            } else if i / dims == j / dims {
                diagonal_blocks[(j%dims)*dims + (i%dims) + (i/dims)*dims*dims]
            } else if i == j + dims || i + dims == j {
                off_diagonal_diagonals[i.min(j)]
            } else {
                0.0.into()
            };

            let (r, g, b) = complex_to_rgb(z);
            let str = match z.norm_sqr() {
                ..=0.0 => "┼─",
                _ => "██"
            };

            print!("{}", str.truecolor(r, g, b));
        }

        println!();
    }
}

pub fn display_system(matrix: &[c64], vector: &[c64], dims: usize) {
    for i in 0..dims {
        for j in 0..(dims+1) {
            let z = if j == dims { vector[i] } else { matrix[i + j * dims] };
            let (r, g, b) = complex_to_rgb(z);
            let str = match z.norm_sqr() {
                ..=0.0 => "┼─",
                _ => "██"
            };

            if j == dims && i == dims / 2 {
                print!(" * ?? = {}", str.truecolor(r, g, b));
            } else if j == dims {
                print!("   ??   {}", str.truecolor(r, g, b));
            } else {
                print!("{}", str.truecolor(r, g, b));
            }
        }

        println!();
    }
}

pub fn print_matrix(matrix: &[c64], rows: usize, cols: usize) {
    // let mut s = String::new();
    // for i in 0..self.rows {
    //     for j in 0..self.cols {
    //         s.push_str(&format!("{: ^24}", format!("{:.4} + {:.4}i ", self.data[j * self.cols + i].re, self.data[j * self.cols + i].im)));
    //     }
    //     s.push_str("\n");
    // }
    //
    // write!(f, "{}", s)

    for i in 0..rows {
        for j in 0..cols {
            print!("{: ^24}", format!("{:.4} + {:.4}i ", matrix[j * rows + i].re, matrix[j * rows + i].im));
        }
        println!();
    }
}

/// Solves the block-tridiagonal matrix where the off-diagonal blocks are all βI and the
/// diagonal blocks are full matrices indexed by i: A_i
///
/// Matrix:
/// ```text
/// | A_1  β*I   0  |   | x1 |   | v1 |
/// | β*Ι  A_2  β*Ι | * | x2 | = | v2 |
/// |  0   β*Ι  Α_3 |   | x3 |   | v3 |
/// ```
pub fn special_block_tridiagonal_solve(
    beta: c64,
    mut diagonal_blocks: Vec<c64>,
    mut rhs_vectors: Vec<c64>,
    num_blocks: usize,
    dims: usize
) -> Result<Vec<c64>, LapackError> {
    let lwork = optimal_zgetri_lwork(dims);
    let mut work: Vec<c64> = vec![0.0.into(); lwork];
    let mut ipiv: Vec<i32> = vec![0; dims];
    let mut temp: Vec<c64> = vec![0.0.into(); dims];
    let beta_sqr = beta*beta;

    let m = get_slice(&mut diagonal_blocks[..], dims*dims, 0);
    let rhs = get_slice(&mut rhs_vectors[..], dims, 0);

    invert(dims, m, &mut ipiv, &mut work)?;
    matrix_multiply(dims, 1.0, &*rhs, &*m, &mut temp);
    rhs.copy_from_slice(&*temp);

    for k in 1..num_blocks {
        let (m, m_prev) = get_neighboring_slices(&mut diagonal_blocks[..], dims*dims, k, k-1);
        let (rhs, rhs_prev) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k-1);

        // println!("Forward (i={k})");

        for i in 0..dims {
            for j in 0..dims {
                m[i * dims + j] -= beta_sqr * m_prev[i * dims + j];
            }
            
            temp[i] = 0.0.into();
            rhs[i] -= beta * rhs_prev[i];
        }

        invert(dims, m, &mut ipiv, &mut work)?;

        matrix_multiply(dims, 1.0, rhs, m, &mut temp);
        rhs.copy_from_slice(&*temp);
    }

    for k in (0..num_blocks-1).rev() {
        let (rhs, rhs_next) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k + 1);
        let m = get_slice(&mut diagonal_blocks[..], dims * dims, k);

        // println!("Backward (i={k})");

        matrix_multiply(dims, -beta, &*rhs_next, &*m, rhs);
    }

    Ok(rhs_vectors)
}

#[test]
fn test_special_tridiagonal_solve() {
    let dims_t = 2;
    let n_x = 3;
    let mut matrices: Vec<c64> = vec![
        1, 2,
        3, 4,
               2, 3,
               4, 5,
                     -1, 0,
                      1, 2
    ].into_iter().map(|x| (x as f64).into()).collect();
    let beta = 2.0;
    let rhs_vectors: Vec<c64> = vec![
        c64::new(1.0, 2.0),
        c64::new(9.0, 8.0),

        c64::new(9.0, -2.0),
        c64::new(15.0, 2.0),

        c64::new(1.0, -2.0),
        c64::new(5.0, 2.0)
    ];
    let expected_sol = vec![
        c64::new(1.0, 0.0),
        c64::new(0.0, 2.0),

        c64::new(0.0, -1.0),
        c64::new(3.0, 0.0),

        c64::new(-1.0, 0.0),
        c64::new(0.0, 1.0)
    ];

    for k in 0..n_x {
        let m = get_slice(&mut matrices[..], dims_t*dims_t, k);
        println!("Before transpose:");
        print_matrix(m, dims_t, dims_t);
        transpose(dims_t, m);
        println!("After transpose:");
        print_matrix(m, dims_t, dims_t);
    }

    // let m = matrices.clone();

    let solution = special_block_tridiagonal_solve(
        beta.into(), matrices.clone(), rhs_vectors.clone(), n_x, dims_t
    ).unwrap();

    let mut rhs_computed = vec![0.0.into(); dims_t*n_x];

    matrix_multiply_special_tridiagonal(dims_t, n_x, &solution, beta, &matrices, &mut rhs_computed);

    println!("Solution:");
    for (
        (&sol, &expected_sol),
        (&rhs, &expected_rhs)
    ) in solution.iter()
        .zip(expected_sol.iter())
        .zip(rhs_computed.iter().zip(rhs_vectors.iter()))
    {
        println!(
            "  | {: ^20} | vs. | {: ^20} | ---- | {: ^20} | vs. | {: ^20} |",
            format!("{:.4}", sol), format!("{:.4}", expected_sol),
            format!("{:.4}", rhs), format!("{:.4}", expected_rhs)
        );
    }
}

/// Solves the block-tridiagonal matrix where the off-diagonal blocks are diagonal matrices B_i and
/// the diagonal blocks are full matrices A_i
///
/// Matrix:
/// ```text
/// | A_1  B_1   0  |   | x1 |   | v1 |
/// | B_1  A_2  B_3 | * | x2 | = | v2 |
/// |  0   B_2  Α_3 |   | x3 |   | v3 |
/// ```
pub fn other_special_block_tridiagonal_solve(
    mut off_diagonal_diagonals: Vec<c64>,
    mut diagonal_blocks: Vec<c64>,
    mut rhs_vectors: Vec<c64>,
    num_blocks: usize,
    dims: usize
) -> Result<Vec<c64>, LapackError> {
    let lwork = optimal_zgetri_lwork(dims);
    let mut work: Vec<c64> = vec![0.0.into(); lwork];
    let mut ipiv: Vec<i32> = vec![0; dims];
    let mut temp: Vec<c64> = vec![0.0.into(); dims];

    let m = get_slice(&mut diagonal_blocks[..], dims*dims, 0);
    let rhs = get_slice(&mut rhs_vectors[..], dims, 0);

    invert(dims, m, &mut ipiv, &mut work)?;
    matrix_multiply(dims, 1.0, &*rhs, &*m, &mut temp);
    rhs.copy_from_slice(&*temp);

    for k in 1..num_blocks {
        let (m, m_prev) = get_neighboring_slices(&mut diagonal_blocks[..], dims*dims, k, k-1);
        let (rhs, rhs_prev) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k-1);
        let b = get_slice(&mut off_diagonal_diagonals[..], dims, k-1);

        // println!("Forward (i={k})");

        for i in 0..dims {
            for j in 0..dims {
                m[i * dims + j] -= b[i] * b[j] * m_prev[i * dims + j];
            }

            temp[i] = 0.0.into();
            rhs[i] -= b[i] * rhs_prev[i];
        }

        invert(dims, m, &mut ipiv, &mut work)?;

        matrix_multiply(dims, 1.0, rhs, m, &mut temp);
        rhs.copy_from_slice(&*temp);
    }

    for k in (0..num_blocks-1).rev() {
        let (rhs, rhs_next) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k + 1);
        let m = &*get_slice(&mut diagonal_blocks[..], dims * dims, k);
        let b = &*get_slice(&mut off_diagonal_diagonals[..], dims, k);

        // println!("Backward (i={k})");

        for i in 0..dims {
            for j in 0..dims {
                rhs[j] -= b[i] * rhs_next[i] * m[i * dims + j];
            }
        }
    }

    Ok(rhs_vectors)
}

pub fn other_special_block_tridiagonal_solve_backwards(
    mut off_diagonal_diagonals: Vec<c64>,
    mut diagonal_blocks: Vec<c64>,
    mut rhs_vectors: Vec<c64>,
    num_blocks: usize,
    dims: usize
) -> Result<Vec<c64>, LapackError> {
    let lwork = optimal_zgetri_lwork(dims);
    let mut work: Vec<c64> = vec![0.0.into(); lwork];
    let mut ipiv: Vec<i32> = vec![0; dims];
    let mut temp: Vec<c64> = vec![0.0.into(); dims];

    let m = get_slice(&mut diagonal_blocks[..], dims*dims, num_blocks-1);
    let rhs = get_slice(&mut rhs_vectors[..], dims, num_blocks-1);

    invert(dims, m, &mut ipiv, &mut work)?;
    matrix_multiply(dims, 1.0, &*rhs, &*m, &mut temp);
    rhs.copy_from_slice(&*temp);

    for k in (0..num_blocks-1).rev() {
        let (m, m_prev) = get_neighboring_slices(&mut diagonal_blocks[..], dims*dims, k, k+1);
        let (rhs, rhs_prev) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k+1);
        let b = get_slice(&mut off_diagonal_diagonals[..], dims, k);

        // println!("Forward (i={k})");

        for i in 0..dims {
            for j in 0..dims {
                m[i * dims + j] -= b[i] * b[j] * m_prev[i * dims + j];
            }

            temp[i] = 0.0.into();
            rhs[i] -= b[i] * rhs_prev[i];
        }

        invert(dims, m, &mut ipiv, &mut work)?;

        matrix_multiply(dims, 1.0, rhs, m, &mut temp);
        rhs.copy_from_slice(&*temp);
    }

    for k in 1..num_blocks {
        let (rhs, rhs_next) = get_neighboring_slices(&mut rhs_vectors[..], dims, k, k - 1);
        let m = &*get_slice(&mut diagonal_blocks[..], dims * dims, k);
        let b = &*get_slice(&mut off_diagonal_diagonals[..], dims, k-1);

        // println!("Backward (i={k})");

        for i in 0..dims {
            for j in 0..dims {
                rhs[j] -= b[i] * rhs_next[i] * m[i * dims + j];
            }
        }
    }

    Ok(rhs_vectors)
}

#[test]
fn test_other_special_tridiagonal_solve() {
    let dims_t = 2;
    let n_x = 3;
    let mut matrices: Vec<c64> = vec![
        1, 2,
        3, 4,
               2, 3,
               4, 5,
                     -1, 0,
                      1, 2
    ].into_iter().map(|x| (x as f64).into()).collect();
    let off_diagonal_diagonals: Vec<c64> = vec![
        1,
        2,
        3,
        4,
    ].into_iter().map(|x| (x as f64).into()).collect();
    let rhs_vectors: Vec<c64> = vec![
        c64::new(1.0, 3.0),
        c64::new(9.0, 8.0),

        c64::new(7.0, -2.0),
        c64::new(15.0, 4.0),

        c64::new(1.0, -3.0),
        c64::new(11.0, 2.0)
    ];
    let expected_sol = vec![
        c64::new(1.0, 0.0),
        c64::new(0.0, 2.0),

        c64::new(0.0, -1.0),
        c64::new(3.0, 0.0),

        c64::new(-1.0, 0.0),
        c64::new(0.0, 1.0)
    ];

    for k in 0..n_x {
        let m = get_slice(&mut matrices[..], dims_t*dims_t, k);
        println!("Before transpose:");
        print_matrix(m, dims_t, dims_t);
        transpose(dims_t, m);
        println!("After transpose:");
        print_matrix(m, dims_t, dims_t);
    }

    // let m = matrices.clone();

    let solution = other_special_block_tridiagonal_solve_backwards(
        off_diagonal_diagonals, matrices.clone(), rhs_vectors.clone(), n_x, dims_t
    ).unwrap();

    let rhs_computed: Vec<c64> = vec![0.0.into(); dims_t*n_x];

    // matrix_multiply_special_tridiagonal(dims_t, n_x, &solution, beta, &matrices, &mut rhs_computed);

    println!("Solution:");

    for (
        (&sol, &expected_sol),
        (&rhs, &expected_rhs)
    ) in solution.iter()
        .zip(expected_sol.iter())
        .zip(rhs_computed.iter().zip(rhs_vectors.iter()))
    {
        println!(
            "  | {: ^20} | vs. | {: ^20} | ---- | {: ^20} | vs. | {: ^20} |",
            format!("{:.4}", sol), format!("{:.4}", expected_sol),
            format!("{:.4}", rhs), format!("{:.4}", expected_rhs)
        );
    }
}

fn get_slice<T>(slice: &mut [T], dims: usize, i: usize) -> &mut [T] {
    &mut slice[i*dims..(i+1)*dims]
}

fn get_neighboring_slices<T>(slice: &mut [T], dims: usize, i: usize, j: usize) -> (&mut [T], &mut [T]) {
    if j > i {
        let (left, right) = slice.split_at_mut(j*dims);

        (&mut left[i*dims..(i+1)*dims], &mut right[..dims])
    } else if i > j {
        let (left, right) = slice.split_at_mut(i*dims);

        (&mut right[..dims], &mut left[j*dims..(j+1)*dims])
    } else {
        panic!("Cannot get neighboring slices for the same slice");
    }
}

pub fn transpose(dims: usize, matrix: &mut [c64]) {
    for i in 0..dims {
        for j in 0..i {
            let temp = matrix[i * dims + j];
            matrix[i * dims + j] = matrix[j * dims + i];
            matrix[j * dims + i] = temp;
        }
    }
}

pub fn matrix_multiply_special_tridiagonal(
    dims: usize,
    num_blocks: usize,
    vectors: &[c64],
    beta: f64,
    diagonal_blocks: &[c64],
    offset: &mut [c64]
) {
    for k in 0..num_blocks {
        for i in 0..dims {
            if k > 0 {
                offset[k*dims + i] += beta * vectors[(k-1)*dims+i];
            }

            if k < num_blocks-1 {
                offset[k*dims + i] += beta * vectors[(k+1)*dims+i];
            }

            for j in 0..dims {
                offset[k*dims + i] += diagonal_blocks[k*dims*dims + j*dims + i] * vectors[k*dims + j];
            }
        }
    }
}

#[test]
fn test_special_matrix_mul() {
    let mut matrices: Vec<c64> = vec![
        1, 2,
        3, 4,
               2, 3,
               4, 5,
                     -1, 0,
                      1, 2
    ].into_iter().map(|x| (x as f64).into()).collect();
    for k in 0..3 {
        let m = get_slice(&mut matrices[..], 4, k);
        println!("Before transpose:");
        print_matrix(m, 2, 2);
        transpose(2, m);
        println!("After transpose:");
        print_matrix(m, 2, 2);
    }
    let beta = 2.0;
    let expected_sol = vec![
        c64::new(1.0, 0.0),
        c64::new(0.0, 2.0),

        c64::new(0.0, -1.0),
        c64::new(3.0, 0.0),

        c64::new(-1.0, 0.0),
        c64::new(0.0, 1.0)
    ];

    let mut rhs_computed = vec![0.0.into(); 6];
    matrix_multiply_special_tridiagonal(2, 3, &expected_sol, beta, &matrices, &mut rhs_computed);

    println!("Solution: {:?}", rhs_computed);
}

/// Computes alpha * matrix * vector + offset and stores the result in offset
pub fn matrix_multiply<T: std::ops::Mul<c64, Output = c64> + Copy>(dims: usize, alpha: T, vector: &[c64], matrix: &[c64], offset: &mut [c64]) {
    for i in 0..dims {
        for j in 0..dims {
            offset[j] += alpha * vector[i] * matrix[i * dims + j];
        }
    }
}

pub fn invert(dims: usize, matrix: &mut [c64], ipiv: &mut [i32], work: &mut [c64]) -> Result<(), LapackError> {
    let mut info = 0;
    let lwork = work.len() as i32;

    zgetrf(
        dims as i32,
        dims as i32,
        &mut matrix[..],
        dims as i32,
        &mut ipiv[..],
        &mut info
    );

    if info < 0 {
        return Err(LapackError { info, message: format!("Illegal value in argument {}", -info) });
    } else if info > 0 {
        return Err(LapackError { info, message: format!("U({},{}) is exactly zero", info, info) });
    }

    zgetri(
        dims as i32,
        &mut matrix[..],
        dims as i32,
        &mut ipiv[..],
        &mut work[..],
        lwork,
        &mut info
    );

    match info {
        0 => Ok(()),
        n @ ..=-1 => Err(LapackError { info, message: format!("Illegal value in argument {}", -n) }),
        n => Err(LapackError { info, message: format!("U({},{}) is exactly zero", n, n) }),
    }
}

pub fn optimal_zgetri_lwork(n: usize) -> usize {
    let mut work = [0.0.into()];

    zgetri(
        n as i32,
        &mut [][..],
        n as i32,
        &mut [][..],
        &mut work[..],
        -1,
        &mut 0
    );

    work[0].re as usize
}