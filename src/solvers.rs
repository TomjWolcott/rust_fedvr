use std::sync::Arc;
use std::thread::ScopedJoinHandle;
use std::time::{Duration, Instant};
use colored::Color;
use itertools::Itertools;
use lapack::c64;
use plotly::{Configuration, Layout, Plot, Scatter};
use plotly::color::NamedColor;
use plotly::common::{DashType, Line, Marker, Mode, Title};
use plotly::layout::{Axis, AxisType};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use crate::bases::{Basis, Grid, HarmonicEigenfunctions, LagrangePolynomials};
use crate::lapack_wrapper::{display_other_special_matrix, other_special_block_tridiagonal_solve};
use crate::matrices::{Matrix, SymTridiagonalMatrix};
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

        for i in 0..num_iters {
            solutions[i * n_x..(i+1)*n_x].copy_from_slice(&psi[..]);
            let t = self.t_i + (i as f64) * self.delta_t;
            // if i % 10000 < 5 { println!("Crank: {i} -- t = {t:.5} -- [{}]", psi[0..10].iter().map(|z| format!("{z:.4}")).join(", ")); }

            let hamiltonian = self.space_basis.get_hamiltonian(&problem, t);

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
fn test_crank_nicolson_solvera()  {
    let solver = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.001,
        space_basis: Grid::new(10000, -5e1, 5e1),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        1.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    plot_probabilities(
        "crank_grid",
        solution,
        vec![(0, ground_state)],
        harmonic_ground_state_probability(1.0, 1.0, 100.0)
    );
}

#[test]
fn test_crank_nicolson_solver_eigen()  {
    let solver = CrankNicolsonSolver {
        t_i: 0.0,
        t_f: 100.0,
        delta_t: 0.00001,
        space_basis: HarmonicEigenfunctions(1000),
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

impl<BX: Tdse1dSpatialBasis<Matrix=SymTridiagonalMatrix> + Send + Sync> Solver<Tdse1dProblem> for DvrSolver<BX> {
    type Solution<'a> = DvrSolution<'a, BX> where Self: 'a;

    fn solve(&self, problem: &Tdse1dProblem, init: Vec<c64>) -> Self::Solution<'_> {
        let (n_t, n_x) = (self.polys.dims, self.space_basis.len());
        let mut diagonal_blocks: Vec<c64> = vec![0.0.into(); n_t*n_t * n_x];
        let mut off_diagonal_diagonals: Vec<c64> = vec![0.0.into(); n_t * (n_x-1)];
        let mut rhs: Vec<c64> = vec![0.0.into(); n_t*n_x];

        for i in 0..n_t {
            let t = self.polys.point(i);
            let hamiltonian = self.space_basis.get_hamiltonian(problem, t);

            for j in 0..n_x {
                let i_bridge_mult = if self.polys.is_bridge(i) { 2.0 } else { 1.0 };
                diagonal_blocks[i * (n_t+1) + j * n_t*n_t] -= i_bridge_mult * hamiltonian.diagonal[j];

                if j > 0 {
                    off_diagonal_diagonals[i + (j-1) * n_t] -= i_bridge_mult * hamiltonian.off_diagonal[j-1];
                }

                for k in 0..n_t {
                    diagonal_blocks[i + k * n_t + j * n_t*n_t] += if self.polys.is_bridge(k) && self.polys.is_bridge(i) {
                        0.0.into()
                    } else { c64::i() * self.polys.l_deriv(k, t) };
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

        // display_other_special_matrix(&off_diagonal_diagonals, &diagonal_blocks, &rhs, n_t, n_x);

        let solution = other_special_block_tridiagonal_solve(
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

#[test]
fn test_dvr_solvera()  {
    let solver = DvrSolver {
        polys: LagrangePolynomials::new(20, 20, 0.0, 100.0),
        space_basis: Grid::new(4001, -5e1, 5e1),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        1.0,
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
        solution,
        (0..5).map(|i| {
            if i == 4 {
                (100, solver.space_basis.eigenstate(&problem, 100))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(1.0, 1.0, 100.0)
    );
}

#[test]
fn test_dvr_solver_eigen()  {
    let solver = DvrSolver {
        polys: LagrangePolynomials::new(200, 20, 0.0, 100.0),
        space_basis: HarmonicEigenfunctions(600),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        1.0,
        1.0,
        100.0
    );

    let ground_state = solver.space_basis.eigenstate(&problem, 0);
    let solution = solver.solve(&problem, ground_state.clone());

    // plot_result(
    //     0.0, 100.0,
    //     -1e5, 1e5,
    //     &|x, t| solution.eval((x, t)),
    //     |x, t| 0.0.into(),
    //     "dvr_eigen_wavefunction"
    // );

    plot_probabilities(
        "dvr_eigen",
        solution,
        (0..5).map(|i| {
            if i == 4 {
                (100, solver.space_basis.eigenstate(&problem, 100))
            } else {
                (i, solver.space_basis.eigenstate(&problem, i))
            }
        }).collect(),
        harmonic_ground_state_probability(1.0, 1.0, 100.0)
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
}
 */



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

        for i in 0..n_x {
            norm += self.solver.space_basis.weight(i) * self.solver.polys.eval(t, &self.solution[i * n_t..(i+1) * n_t]).conj() * state[i];
        }

        norm.norm_sqr()
    }

    fn norm(&self, t: f64) -> f64 {
        let (n_t, n_x) = (self.solver.polys.dims, self.solver.space_basis.len());
        let mut norm = 0.0;

        for i in 0..n_x {
            norm += self.solver.space_basis.weight(i) * self.solver.polys.eval(t, &self.solution[i * n_t..(i+1) * n_t]).norm_sqr();
        }

        norm
    }
}

pub fn plot_probabilities(
    plot_name: &str,
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
        _ => NamedColor::Gray
    };

    let mut max_err: f64 = 0.0;
    let mut xs = Vec::new();
    let mut ys_e = vec![Vec::new(); states.len()];
    let mut ys_c = vec![Vec::new(); states.len()];
    let mut ys_n = Vec::new();
    let mut ys_err = vec![Vec::new(); states.len()];

    sample(2000, 0.0, 100.0, |t| {
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
        // add_ys(format!("Err ψ_{n}").as_str(), ys_err.remove(0), get_color(n), DashType::Dot);
    }

    add_ys("Norm", ys_n, NamedColor::Black, DashType::Solid);

    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::new("time")))
            .y_axis(Axis::new().title(Title::new("P_0")))
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
                    let states: Vec<Vec<c64>> = (0..10).map(|i| $solver .space_basis.eigenstate(& $problem, i)).collect();
                    let sol = $solver .solve($problem, $solver .default_init(&$problem));
                    let dt = t_0.elapsed();
                    println!("{} finished, taking {:?}", $name, dt);

                    ($name, Box::new(sol) as Box<dyn PlottableSolution + Send + Sync>, states, t_0.elapsed())
                })),*
            ];

            threads.into_iter()
                .map(|thread| thread.join().unwrap())
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
        polys: LagrangePolynomials::new(280, 1, 0.0, 100.0),
        space_basis: HarmonicEigenfunctions(500),
    };

    let solver4 = DvrSolver {
        polys: LagrangePolynomials::new(100, 1, 0.0, 100.0),
        space_basis: Grid::new(10000, -5e1, 5e1),
    };

    let problem = Tdse1dProblem::harmonic_oscillator(
        1.0,
        1.0,
        100.0
    );

    compare_from_solvers!(
        "compare_solvers_w0=1",
        &problem,
        [
            ("Crank Eigen".to_string(), solver1),
            ("Crank Grid".to_string(), solver2),
            ("DVR Eigen".to_string(), solver3),
            ("DVR Grid".to_string(), solver4)
        ],
        harmonic_ground_state_probability(1.0, 1.0, 100.0)
    );
}

pub fn compare_errs<'a>(
    plot_name: &str,
    solutions: Vec<(String, Box<dyn PlottableSolution + 'a + Send + Sync>, Vec<Vec<c64>>, Duration)>,
    expected_prob: impl Fn(f64, usize) -> f64
) {
    println!("Starting plotting");

    let mut xs = Vec::new();
    let mut y_errs = vec![Vec::new(); solutions.len()];

    sample(200, 0.0, 100.0, |t| {
        let expected = expected_prob(t, 0);
        xs.push(t);

        for ((_, sol, states, _), y_err) in solutions.iter().zip(y_errs.iter_mut()) {
            y_err.push(states.iter().enumerate().map(|(i, state)| {
                (sol.state_prob(t, &state[..]) - expected_prob(t, i)).abs()
            }).sum::<f64>() / 10.0);
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
    );
    prob_plot.write_html(format!("output/{}.html", plot_name));
}