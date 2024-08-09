use std::time::{Duration, Instant};
use colored::{Color, Colorize};
use itertools::Itertools;
use lapack::c64;
use lapack::fortran::zgesv;
use plotly::{Configuration, Layout, Plot, Scatter};
use plotly::color::NamedColor;
use plotly::common::{DashType, Line, Marker, Mode, Title};
use plotly::layout::{Axis, AxisType};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use crate::bases::{BackwardGrid, Basis, Grid, HarmonicEigenfunctions, LagrangePolynomials};
use crate::lapack_wrapper::{display_matrix, display_other_special_matrix, display_system, other_special_block_tridiagonal_solve, other_special_block_tridiagonal_solve_backwards};
use crate::matrices::{LowerTriangularConstBandedMatrix, Matrix, SymTridiagonalMatrix};
use crate::sample;
use crate::tdse_1d::{harmonic_ground_state_probability, Tdse1dProblem, Tdse1dSpatialBasis};
use rayon::iter::ParallelIterator;
use crate::schrodinger::plot_result;

pub trait Problem {
    type In;
    type Out;
}

pub trait Solution<PROBLEM: Problem> {
    fn eval(&self, i: PROBLEM::In) -> PROBLEM::Out;
}

pub trait Solver<PROBLEM: Problem> {
    type Solution<'a>: Solution<PROBLEM> + Send + Sync where Self: 'a;

    fn solve(&self, problem: &PROBLEM, init: Vec<c64>) -> Self::Solution<'_>;

    fn default_init(&self, problem: &PROBLEM) -> Vec<c64>;
}

pub struct CrankNicolsonSolver<B: Basis + Send + Sync> {
    t_i: f64,
    t_f: f64,
    delta_t: f64,
    space_basis: B
}

impl<B: Tdse1dSpatialBasis + Send + Sync> Solver<Tdse1dProblem> for CrankNicolsonSolver<B> {
    type Solution<'a> = CrankNicolsonSolution<'a, B> where Self: 'a;
    
    fn solve(&self, problem: &Tdse1dProblem, init: Vec<c64>) -> Self::Solution<'_> {
        let num_iters = ((self.t_f - self.t_i) / self.delta_t) as usize;
        let n_x = self.space_basis.len();
        let mut psi = init;
        let mut solutions = vec![0.0.into(); (num_iters+1) * n_x];

        print_mem("solutions", &*solutions);

        for i in 0..num_iters {
            solutions[i * n_x..(i+1)*n_x].copy_from_slice(&psi[..]);
            let t = self.t_i + (i as f64) * self.delta_t;
            // if i % 10000 < 5 { println!("Crank: {i} -- t = {t:.5} -- [{}]", psi[0..10].iter().map(|z| format!("{z:.4}")).join(", ")); }

            let hamiltonian = self.space_basis.get_hamiltonian(&problem, t);
            // let hamiltonian = SymTridiagonalMatrix {
            //     diagonal: (0..n_x).map(|i| self.space_basis.get_hamiltonian_element(&problem, t, i, i)).collect(),
            //     off_diagonal: (0..n_x-1).map(|i| self.space_basis.get_hamiltonian_element(&problem, t, i, i+1)).collect(),
            // };

            let matrix = hamiltonian.clone()
                .mul_num(c64::i() * self.delta_t / 2.0)
                .add_num(1.0.into());

            let rhs = hamiltonian
                .mul_num(-c64::i() * self.delta_t / 2.0)
                .add_num(1.0.into())
                .mul_vec(&psi);

            psi = matrix.solve_system(rhs);
        }

        solutions[num_iters * n_x..].copy_from_slice(&psi[..]);

        Self::Solution {
            solutions,
            solver: &self
        }
    }

    fn default_init(&self, problem: &Tdse1dProblem) -> Vec<c64> {
        self.space_basis.eigenstate(problem, 0)
    }
}

pub struct CrankNicolsonSolution<'a, B: Basis + Send + Sync> {
    solutions: Vec<c64>,
    solver: &'a CrankNicolsonSolver<B>
}

impl<'a, B: Basis + Send + Sync> CrankNicolsonSolution<'a, B> {
    fn state_t(&self, t: f64) -> Vec<c64> {
        let n_x = self.solver.space_basis.len();
        let max_it = self.solutions.len() / n_x;
        let i1 = ((t - self.solver.t_i) / self.solver.delta_t).floor() as usize;
        let i2 = (((t - self.solver.t_i) / self.solver.delta_t).ceil() as usize).max(max_it-1);
        let mid_t = (t - ((i1 as f64) * self.solver.delta_t + self.solver.t_i)) / self.solver.delta_t;

        (0..n_x).map(|j| {
            (1.0 - mid_t) * self.solutions[i1 * n_x + j] + mid_t * self.solutions[i2 * n_x + j]
        }).collect()
    }
}

impl<'a, B: Basis + Send + Sync> Solution<Tdse1dProblem> for CrankNicolsonSolution<'a, B> {
    fn eval(&self, (x, t): (f64, f64)) -> c64 {
        self.solver.space_basis.eval(x,  &self.state_t(t))
    }
}

#[test]
fn test_crank_nicolson_solvera() {
    let solver = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.001,
        space_basis: Grid::new(10000, -5e1, 5e1),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        0.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    plot_probabilities(
        "crank_grid",
        (0.0, 100.0),
        solution,
        vec![(0, ground_state)],
        harmonic_ground_state_probability(0.0, 1.0, 100.0)
    );
}

#[test]
fn test_crank_nicolson_solver_eigen()  {
    let solver = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.0001,
        space_basis: HarmonicEigenfunctions(400),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        0.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    plot_probabilities(
        "cranky_eigen",
        (0.0, 100.0),
        solution,
        (0..5).map(|i| {
            if i == 4 {
                (100, solver.space_basis.eigenstate(&problem, 100))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(0.0, 1.0, 100.0)
    );
}

#[test]
fn test_crank_nicolson_solver_back_grid() {
    let solver = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.001,
        space_basis: BackwardGrid::new(10000, -5e1, 5e1, 7),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        0.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    plot_probabilities(
        "crank_back_grid",
        (0.0, 100.0),
        solution,
        vec![(0, ground_state)],
        harmonic_ground_state_probability(0.0, 1.0, 100.0)
    );
}

pub struct DvrSolver<BX: Basis + Send + Sync> {
    polys: LagrangePolynomials,
    space_basis: BX
}

pub struct DvrSolution<'a, BX: Basis + Send + Sync> {
    solution: Vec<c64>,
    solver: &'a DvrSolver<BX>
}

impl<'a, B: Basis + Send + Sync> Solution<Tdse1dProblem> for DvrSolution<'a, B> {
    fn eval(&self, (x, t): (f64, f64)) -> c64 {
        let (n_t, n_x) = (self.solver.polys.dims, self.solver.space_basis.len());
        let mut result: c64 = 0.0.into();

        for i in 0..n_x {
            result += self.solver.space_basis.eval_at(i, x) * self.solver.polys.eval(t, &self.solution[i * n_t..(i+1) * n_t]);
        }

        result
    }
}

fn print_mem<T: ?Sized>(name: &str, t: &T) {
    let size = std::mem::size_of_val(t);

    println!("{name}: {}", match size.ilog(1024) {
        0 => format!("{size} B"),
        1 => format!("{:.3} KB", size as f64 / 2f64.powi(10)),
        2 => format!("{:.3} MB", size as f64 / 2f64.powi(20)),
        3 => format!("{:.3} GB", size as f64 / 2f64.powi(30)),
        4 => format!("{:.3} TB", size as f64 / 2f64.powi(40)),
        _ => format!("{size:.3e} B")
    })
}

impl<BX: Tdse1dSpatialBasis<Matrix=SymTridiagonalMatrix> + Send + Sync> Solver<Tdse1dProblem> for DvrSolver<BX> {
    type Solution<'a> = DvrSolution<'a, BX> where Self: 'a;

    fn solve(&self, problem: &Tdse1dProblem, init: Vec<c64>) -> Self::Solution<'_> {
        let (n_t, n_x) = (self.polys.dims, self.space_basis.len());
        let l_derivs: Vec<c64> = (0..n_t*n_t).map(|i| {
            let (j, k) = (i % n_t, i / n_t);

            if self.polys.both_bridges(j, k) {
                0.0.into()
            } else { c64::new(0.0, self.polys.l_deriv(k, self.polys.point(j))) }
        }).collect();
        let mut diagonal_blocks: Vec<c64> = l_derivs.repeat(n_x);
        let mut off_diagonal_diagonals: Vec<c64> = vec![0.0.into(); n_t * (n_x-1)];
        let mut rhs: Vec<c64> = vec![0.0.into(); n_t*n_x];
        println!("Memory:");
        print_mem("    diagonal_blocks", &*diagonal_blocks);
        print_mem("    off_diagonal_diagonals", &*off_diagonal_diagonals);
        print_mem("    rhs", &*rhs);

        for i in 0..n_x {
            // if i > 0 && (i-1) % 100 == 0 { println!("Finished diagonal blocks {}-{}", i-100, i); }

            for j in 0..n_t {
                let t = self.polys.point(j);

                let i_bridge_mult = if self.polys.is_bridge(i) { 2.0 } else { 1.0 };
                diagonal_blocks[j * (n_t+1) + i * n_t*n_t] -= i_bridge_mult * self.space_basis.get_hamiltonian_element(problem, t, i, i);

                if i < n_x - 1 {
                    off_diagonal_diagonals[j + i * n_t] -= i_bridge_mult * self.space_basis.get_hamiltonian_element(problem, t, i, i+1);
                }
            }
        }

        for i in 0..n_x {
            rhs[i * n_t] = init[i];

            if i > 0 {
                off_diagonal_diagonals[(i-1)*n_t] = 0.0.into();
            }

            for j in 0..n_t {
                diagonal_blocks[j*n_t + i * n_t*n_t] = if j == 0 { 1.0.into() } else { 0.0.into() }
            }
        }

        // display_other_special_matrix(&off_diagonal_diagonals[..], &diagonal_blocks[..], &rhs[..], n_t, n_x);

        // diagonal_blocks.iter().enumerate().for_each(|(i, z)| {
        //     if z.is_nan() {
        //         panic!("NaN in diagonal_blocks @ i={}/k={}/j={}: {:?}", i % n_t, (i / n_t) % n_t, i / (n_t*n_t), &diagonal_blocks[i-5..i+5])
        //     }
        //
        //     if z.is_infinite() {
        //         panic!("Inf in diagonal_blocks @ i={}/k={}/j={}: {:?}", i % n_t, (i / n_t) % n_t, i / (n_t*n_t), &diagonal_blocks[i-5..i+5])
        //     }
        // });
        //
        // off_diagonal_diagonals.iter().enumerate().for_each(|(i, z)| {
        //     if z.is_nan() {
        //         panic!("NaN in off_diagonal_diagonals @ i={}/j={}: {:?}", i%n_t, i/n_t, &off_diagonal_diagonals[i-5..i+5])
        //     }
        //
        //     if z.is_infinite() {
        //         panic!("Inf in off_diagonal_diagonals @ i={}/j={}: {:?}", i%n_t, i/n_t, &off_diagonal_diagonals[i-5..i+5])
        //     }
        // });

        let solution = other_special_block_tridiagonal_solve_backwards(
            off_diagonal_diagonals, diagonal_blocks, rhs, n_x, n_t
        ).unwrap();

        DvrSolution {
            solution,
            solver: &self,
        }
    }

    fn default_init(&self, problem: &Tdse1dProblem) -> Vec<c64> {
        self.space_basis.eigenstate(problem, 0)
    }
}

impl Solver<Tdse1dProblem> for DvrSolver<BackwardGrid> {
    type Solution<'a> = DvrSolution<'a, BackwardGrid> where Self: 'a;

    fn solve(&self, problem: &Tdse1dProblem, init: Vec<c64>) -> Self::Solution<'_> {
        let (n_t, n_x) = (self.polys.dims, self.space_basis.len());

        let mut matrix = vec![0.0.into(); n_t * n_t];
        let mut rhs: Vec<c64>  = vec![0.0.into(); n_x * n_t];
        let mut ipiv = vec![0; n_t];

        let mut coefs: Vec<c64> = self.space_basis.second_deriv_coefficients()
            .into_iter()
            .map(|coef| (coef / (2.0 * self.space_basis.delta() * self.space_basis.delta())).into()).collect_vec();
        let center_coef = coefs.remove(0);

        println!("center_coef: {}, coefs: {:?}", center_coef, coefs);

        for i in 0..n_x {
            let x = self.space_basis.points[i];

            rhs[i * n_t] = init[i];

            for l in 0..coefs.len().min(i) {
                rhs[i * n_t] += coefs[l] * init[i - l - 1];
            }

            for k in 0..n_t {
                matrix[k * n_t] = (if k == 0 { 1.0 } else { 0.0 }).into();
            }

            // row index
            for j in 1..n_t {
                let t = self.polys.point(j);
                let j_bridge_mult = if self.polys.is_bridge(j) { 2.0 } else { 1.0 };

                for l in 0..coefs.len().min(i) {
                    let temp = rhs[(i - l - 1) * n_t + j];
                    rhs[i * n_t + j] -= coefs[l] * temp;
                }

                // col index
                for k in 0..n_t {
                    matrix[j + k * n_t] = if self.polys.both_bridges(j, k) {
                        0.0.into()
                    } else { c64::i() * self.polys.l_deriv(k, t) };
                }

                matrix[j * (n_t+1)] -= j_bridge_mult * (
                    x * (problem.electric)(t) +
                        (problem.potential)(x) +
                        center_coef
                );
            }
            // println!();
            // display_system(&matrix[..], &rhs[i * n_t..(i+1) * n_t], n_t);

            let mut info = 0;

            zgesv(
                n_t as i32,
                1,
                &mut matrix[..],
                n_t as i32,
                &mut ipiv,
                &mut rhs[i * n_t..(i+1) * n_t],
                n_t as i32,
                &mut info
            );

            if info != 0 {
                panic!("zgesv failed with err {info}");
            }
        }

        DvrSolution {
            solution: rhs,
            solver: &self,
        }
    }

    fn default_init(&self, problem: &Tdse1dProblem) -> Vec<c64> {
        self.space_basis.eigenstate(problem, 0)
    }
}

#[test]
fn test_dvr_solvera() {
    let t_f = 10.0;
    let w0 = 0.0;

    let solver = DvrSolver {
        polys: LagrangePolynomials::new(100, 1, 0.0, t_f),
        space_basis: Grid::new(7000, -3e1, 3e1),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        w0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    // plot_result(
    //     0.0, 100.0,
    //     -5e1, 5e1,
    //     &|x, t| solution.eval((x, t)),
    //     |x, t| 0.0.into(),
    //     "dvr_wavefunction"
    // );

    plot_probabilities(
        "dvr_grid",
        (0.0, t_f),
        solution,
        (0..6).map(|i| {
            if i == 5 {
                (100, solver.space_basis.eigenstate(&problem, 100))
            } else if i == 4 {
                (20, solver.space_basis.eigenstate(&problem, 20))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(w0, 1.0, 100.0)
    );
}

#[test]
fn test_dvr_solver_eigen() {
    let t_f = 100.0;
    let w0 = 0.0;

    let solver = DvrSolver {
        polys: LagrangePolynomials::new(600, 1, 0.0, t_f),
        space_basis: HarmonicEigenfunctions(400),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        w0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    print_mem("ground_state", &*ground_state);
    print_mem("solver", &solver);
    print_mem("solver.polys", &solver);
    let solution = solver.solve(&problem, ground_state.clone());

    /*
    let mut prob_sums = vec![0.0; 150];
    sample(solver.space_basis.0, 0.0, solver.space_basis.0 as f64, |n| {
        let state = solver.space_basis.eigenstate(&problem, n as usize);
        print!("\n[{: ^5}] ({: ^12}): ", (n as usize).to_string(), format!("{:.4e}", solution.state_prob(0.0, &state[..])));
        let mut i = 0;
        sample(prob_sums.len(), 0.0, t_f, |t| {
            let z = solution.state_prob(t, &state[..]);
            let p = (z*255.0) as u8;
            let p2 = ((z.log10() + 10.0) * 25.5) as u8;
            prob_sums[i] += z;
            i += 1;

            print!("{}", "██".truecolor(p, p, p));
        });
    });

    let mut i = 0;
    print!("\n[ 70  ] ( 1.5754e-53 ): ");
    print!("\n             Norms Err: ");
    sample(prob_sums.len(), 0.0, t_f, |t| {
        let z = (prob_sums[i] - 1.0).abs();
        let p = (z*255.0) as u8;
        let p2 = ((z.log10() + 10.0) * 25.5) as u8;
        i += 1;

        print!("{}", "██".truecolor(p, p2, p2));
    });

    println!("\n{:?}", prob_sums);

 */

    // plot_result(
    //     0.0, 100.0,
    //     -1e5, 1e5,
    //     &|x, t| solution.eval((x, t)),
    //     |x, t| 0.0.into(),
    //     "dvr_eigen_wavefunction"
    // );

    plot_probabilities(
        "dvr_eigen",
        (0.0, t_f),
        solution,
        (0..5).map(|i| {
            if i == 4 {
                (100, solver.space_basis.eigenstate(&problem, 100))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(w0, 1.0, 100.0)
    );
}

#[test]
fn test_dvr_backward_grid() {
    let solver = DvrSolver {
        polys: LagrangePolynomials::new(200, 10, 0.0, 1.0),
        space_basis: BackwardGrid::new(1001, -5e1, 5e1, 7),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        0.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    solution.state_prob(0.0, &ground_state[..]);

    // plot_result(
    //     0.0, 100.0,
    //     -5e1, 5e1,
    //     &|x, t| solution.eval((x, t)),
    //     |x, t| 0.0.into(),
    //     "dvr_wavefunction"
    // );

    plot_probabilities(
        "dvr_back_grid",
        (0.0, 100.0),
        solution,
        (0..5).map(|i| {
            if i == 4 {
                (10, solver.space_basis.eigenstate(&problem, 10))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(0.0, 1.0, 100.0)
    );
}

/*
pub struct FiniteElementSolver<PROBLEM: Problem, SOLVER: Solver<PROBLEM>> {
    solver: SOLVER,
    num_iterations: usize
}

pub struct FiniteElementSolution<'a, PROBLEM: Problem, SOLUTION: GetNextInitial<PROBLEM> + 'a> {
    solutions: Vec<SOLUTION>
}

pub trait GetNextInitial<PROBLEM>: Solution<PROBLEM> {
    fn get_next_initial(&self) -> Vec<c64>;
}

impl<BX: Basis + Send + Sync> GetNextInitial<Tdse1dProblem> for DvrSolution<BX> {
    fn get_next_initial(&self) -> Vec<c64> {
        (0..self.solver.space_basis.len())
            .map(|i| self.solution[self.solver.polys.len() * (i+1) - 1])
            .collect()
    }
}

impl<'a, SOLUTION> Solution<Tdse1dProblem> for FiniteElementSolution<'a, Tdse1dProblem, SOLUTION> {
    fn eval(&self, (x, t): (f64, f64)) -> c64 {

    }
}*/

pub struct ItvoltSolver<B: Basis + Send + Sync> {
    t_i: f64,
    t_f: f64,
    delta_t: f64,
    space_basis: B
}

impl<B: Tdse1dSpatialBasis + Send + Sync> Solver<Tdse1dProblem> for ItvoltSolver<B> {
    type Solution<'a> = ItvoltSolution<'a, B> where Self: 'a;

    fn solve(&self, problem: &Tdse1dProblem, init: Vec<c64>) -> Self::Solution<'_> {
        let num_iters = ((self.t_f - self.t_i) / self.delta_t) as usize;
        let n_x = self.space_basis.len();
        let mut psi = init;
        let mut solutions = vec![0.0.into(); (num_iters+1) * n_x];

        for i in 0..num_iters {
            solutions[i * n_x..(i+1)*n_x].copy_from_slice(&psi[..]);
            let t = self.t_i + (i as f64) * self.delta_t;
            // if i % 10000 < 5 { println!("Crank: {i} -- t = {t:.5} -- [{}]", psi[0..10].iter().map(|z| format!("{z:.4}")).join(", ")); }

            let hamiltonian = self.space_basis.get_hamiltonian(&problem, t);

            let matrix = hamiltonian.clone()
                .mul_num(c64::i() * self.delta_t)
                .add_num(1.0.into());

            let rhs = hamiltonian
                .mul_num(-c64::i() * self.delta_t)
                .add_num(1.0.into())
                .mul_vec(&psi);

            psi = matrix.solve_system(rhs);
        }

        solutions[num_iters * n_x..].copy_from_slice(&psi[..]);

        Self::Solution {
            solution: solutions,
            solver: &self
        }
    }

    fn default_init(&self, problem: &Tdse1dProblem) -> Vec<c64> {
        self.space_basis.eigenstate(problem, 0)
    }
}

pub struct ItvoltSolution<'a, B: Basis + Send + Sync> {
    solution: Vec<c64>,
    solver: &'a ItvoltSolver<B>
}

impl<'a, B: Basis + Send + Sync> Solution<Tdse1dProblem> for ItvoltSolution<'a, B> {
    fn eval(&self, (x, t): (f64, f64)) -> c64 {
        let (n_t, n_x) = (self.solver.space_basis.len(), self.solver.space_basis.len());
        let mut result: c64 = 0.0.into();

        for i in 0..n_x {
            result += self.solver.space_basis.eval_at(i, x) * self.solution[i * n_t];
        }

        result
    }
}

trait PlottableSolution: Solution<Tdse1dProblem> {
    fn state_prob(&self, t: f64, state: &[c64]) -> f64;

    fn norm(&self, t: f64) -> f64;
}

impl<'a, B: Basis + Send + Sync> PlottableSolution for CrankNicolsonSolution<'a, B> {
    fn state_prob(&self, t: f64, state: &[c64]) -> f64 {
        self.state_t(t).iter().enumerate()
            .map(|(j, z)| self.solver.space_basis.weight(j) * z.conj() * state[j])
            .fold(c64::from(0.0), |acc, z| acc+z)
            .norm_sqr()
    }

    fn norm(&self, t: f64) -> f64 {
        self.state_t(t).iter().enumerate()
            .map(|(j, z)| self.solver.space_basis.weight(j) * z.norm_sqr())
            .sum()
    }
}

impl<'a, B: Basis + Send + Sync> PlottableSolution for DvrSolution<'a, B> {
    fn state_prob(&self, t: f64, state: &[c64]) -> f64 {
        let (n_t, n_x) = (self.solver.polys.dims, self.solver.space_basis.len());
        let mut norm: c64 = 0.0.into();
        let mut time_basis = (0..n_t).map(|i| self.solver.polys.eval_at(i, t)).collect_vec();

        for i in 0..n_x {
            let psi = (0..n_t).map(|j| {
                self.solution[j + i * n_t] * time_basis[j]
            }).fold(0.0.into(), |acc: c64, z| acc+z);

            norm += self.solver.space_basis.weight(i) * psi.conj() * state[i];
            // println!("sum: {norm:.6e} after {:?}", &self.solution[i * n_t]);
        }

        norm.norm_sqr()
    }

    //  fn state_prob(&self, t: f64, state: &[c64]) -> f64 {
    //     let (n_t, n_x) = (self.solver.polys.dims, self.solver.space_basis.len());
    //     let mut norm: c64 = 0.0.into();
    //
    //     for i in 0..n_x {
    //         norm += self.solver.space_basis.weight(i) * self.solver.polys.eval(t, &self.solution[i * n_t..(i+1) * n_t]).conj() * state[i];
    //         // println!("sum: {norm:.6e} after {:?}", &self.solution[i * n_t]);
    //     }
    //
    //     norm.norm_sqr()
    // }


    fn norm(&self, t: f64) -> f64 {
        let (n_t, n_x) = (self.solver.polys.dims, self.solver.space_basis.len());
        let mut norm = 0.0;
        let mut time_basis = (0..n_t).map(|i| self.solver.polys.eval_at(i, t)).collect_vec();

        for i in 0..n_x {
            let psi = (0..n_t).map(|j| {
                self.solution[j + i * n_t] * time_basis[j]
            }).fold(c64::new(0.0, 0.0), |acc, z| acc+z);

            norm += self.solver.space_basis.weight(i) * psi.norm_sqr();
        }

        norm
    }
}

pub fn plot_probabilities(
    plot_name: &str,
    (t_i, t_f): (f64, f64),
    solution: impl PlottableSolution,
    states: Vec<(usize, Vec<c64>)>,
    expected_prob: impl Fn(f64, usize) -> f64
) {
    println!("P_0(______) =       Expected       vs       Computed      ");

    let get_color = |n| match n {
        0 => NamedColor::Red,
        1 => NamedColor::Green,
        2 => NamedColor::Blue,
        3 => NamedColor::Purple,
        4 => NamedColor::Orange,
        5 => NamedColor::Brown,
        _ => NamedColor::Gray
    };

    let mut max_err: f64 = 0.0;
    let mut xs = Vec::new();
    let mut ys_e = vec![Vec::new(); states.len()];
    let mut ys_c = vec![Vec::new(); states.len()];
    let mut ys_n = Vec::new();
    let mut ys_err = vec![Vec::new(); states.len()];

    sample(200, t_i, t_f, |t| {
        xs.push(t);
        ys_n.push(solution.norm(t));
        for (i, (n, state)) in states.iter().enumerate() {
            let expected = expected_prob(t, *n);
            let computed = solution.state_prob(t, &state[..]);
            let err = (expected - computed).abs();
            println!("P_0({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", format!("{:.6}", expected), format!("{:.6}", computed));
            max_err = max_err.max(err);

            ys_e[i].push(expected);
            ys_c[i].push(computed);
            ys_err[i].push(err);
        }
    });

    let mut prob_plot = Plot::new();

    let mut add_ys = |name: &str, ys: Vec<f64>, color: NamedColor, dash_type| {
        let scatter = Scatter::new(xs.clone(), ys)
            .name(name)
            .line(Line::new().width(2.0).color(color).dash(dash_type))
            .mode(Mode::Lines);

        prob_plot.add_trace(scatter);
    };

    let first_state_n = states[0].0;

    for (n, _) in states.iter() {
        let n = *n - first_state_n;

        add_ys(format!("Expected ψ_{n}").as_str(), ys_e.remove(0), get_color(n), DashType::Solid);
        add_ys(format!("Computed ψ_{n}").as_str(), ys_c.remove(0), get_color(n), DashType::Dot);
        add_ys(format!("Err ψ_{n}").as_str(), ys_err.remove(0), get_color(n), DashType::Dash);
    }

    add_ys("Norm", ys_n, NamedColor::Black, DashType::Solid);

    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::new("time")))
            .y_axis(Axis::new().title(Title::new("P_0")))
            .title( Title::new("Probability of eigenstates over time"))
    );
    prob_plot.write_html(format!("output/{}.html", plot_name));

    println!("MAX ERROR: {:e}", max_err);
}

macro_rules! solve_in_thread {
    ($s:ident, $solver:ident, $problem:ident) => {
        $s .spawn(|| {
            let time_start = Instant::now();
            let state = $solver .default_init(& $problem);
            let sol = $solver .solve(& $problem, state.clone());
            let elapsed = time_start.elapsed();
            println!("Finished: {:?}", elapsed);
            (elapsed, sol, state)
        })
    };
}

macro_rules! compare_from_solvers {
    ($plot_name:expr, $problem:expr, [$(($name:expr, $solver:expr)),*], $expected_prob:expr) => {{
        compare_errs($plot_name, std::thread::scope(|s| {
            let threads = vec![
                $(s.spawn(|| {
                    println!("{} started", $name);
                    let t_0 = Instant::now();
                    let states: Vec<Vec<c64>> = (0..1).map(|i| $solver .space_basis.eigenstate(& $problem, i)).collect();
                    let sol = $solver .solve($problem, $solver .default_init(&$problem));
                    let dt = t_0.elapsed();
                    println!("{} finished, taking {:?}", $name, dt);

                    ($name, Box::new(sol) as Box<dyn PlottableSolution + Send + Sync>, states, t_0.elapsed())
                }).join().unwrap()),*
            ];

            threads.into_iter()
                .map(|thread| thread)
                .collect::<Vec<_>>()
        }), $expected_prob)
    }};
}

#[test]
fn compare_everything() {
    let solver1 = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.00005,
        space_basis: HarmonicEigenfunctions(500),
    };

    let solver2 = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.001,
        space_basis: Grid::new(10000, -5e1, 5e1),
    };

    let solver3 = DvrSolver {
        polys: LagrangePolynomials::new(600, 1, 0.0, 100.0),
        space_basis: HarmonicEigenfunctions(500),
    };

    let solver4 = DvrSolver {
        polys: LagrangePolynomials::new(60, 5, 0.0, 100.0),
        space_basis: Grid::new(5000, -5e1, 5e1),
    };

    let w0 = 0.0;

    let problem = Tdse1dProblem::harmonic_oscillator(
        w0,
        1.0,
        100.0
    );

    let name = format!("compare_solvers_w0={}", w0);

    println!("plot name: {name:?}");

    compare_from_solvers!(
        name.as_str(),
        &problem,
        [
            ("Crank Eigen".to_string(), solver1),
            ("Crank Grid".to_string(), solver2),
            ("DVR Eigen".to_string(), solver3),
            ("DVR Grid".to_string(), solver4)
        ],
        harmonic_ground_state_probability(w0, 1.0, 100.0)
    );
}

fn compare_errs<'a>(
    plot_name: &str,
    solutions: Vec<(String, Box<dyn PlottableSolution + 'a + Send + Sync>, Vec<Vec<c64>>, Duration)>,
    expected_prob: impl Fn(f64, usize) -> f64
) {
    println!("Starting plotting");

    let mut xs = Vec::new();
    let mut y_errs = vec![Vec::new(); solutions.len()];

    sample(800, 0.0, 100.0, |t| {
        xs.push(t);

        for ((_, sol, states, _), y_err) in solutions.iter().zip(y_errs.iter_mut()) {
            y_err.push(states.iter().enumerate().map(|(i, state)| {
                (sol.state_prob(t, &state[..]) - expected_prob(t, i)).abs()
            }).sum::<f64>() / 1.0);
        }
    });

    let mut prob_plot = Plot::new();

    for (
        (name, sol, _, dt),
        y_err
    ) in solutions.iter().zip(y_errs.into_iter()) {
        println!("{name} took {dt:?}");

        let scatter = Scatter::new(xs.clone(), y_err)
            .name(name)
            .line(Line::new().width(2.0))
            .marker(Marker::new().size(4))
            .mode(Mode::LinesMarkers);

        prob_plot.add_trace(scatter);
    }

    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::new("time")))
            .y_axis(Axis::new().title(Title::new("Error")).type_(AxisType::Log).tick_format(".2e"))
            .title(Title::new("Average Error of the first 10 eigenstates over time"))
    );
    prob_plot.write_html(format!("output/{}.html", plot_name));
}

#[test]
pub fn plot_hydrogen() {
    let t_f = 200.0;

    let solver1 = DvrSolver {
        polys: LagrangePolynomials::new(500, 1, 0.0, t_f),
        space_basis: Grid::new(10001, -2e2, 2e2),
    };

    let solver2 = CrankNicolsonSolver {
        t_i: 0.0,
        t_f,
        delta_t: 0.004,
        space_basis: Grid::new(10001, -2e2, 2e2),
    };

    let problem = Tdse1dProblem::hydrogen_laser_pulse(
        0.148,
        0.1,
        0.0,
        true,
        0.0,
        100.0
    );

    let ground_state = solver1.space_basis.eigenstate(&problem, 0);
    let solution = solver1.solve(&problem, ground_state.clone());

    let solution_crank = solver2.solve(&problem, ground_state.clone());

    plot_probabilities(
        "HYDROGEN_dvr_grid",
        (0.0, t_f),
        solution,
        (0..1).map(|i| {
            if i == 5 {
                (100, solver1.space_basis.eigenstate(&problem, 100))
            } else if i == 4 {
                (20, solver1.space_basis.eigenstate(&problem, 20))
            } else {
                (i, solver1.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        |t, _| solution_crank.state_prob(t, &ground_state[..])
    );
}