use std::f64::consts::PI;
use itertools::Itertools;
use lapack::c64;
use lapack::fortran::{zgesv, zgtsv, zsteqr};
use plotly::{Configuration, Layout, Mesh3D, Plot, Scatter};
use plotly::common::{Line, Marker, Mode, Title};
use plotly::layout::Axis;
use crate::complex_wrapper::LapackError;
use crate::{sample};
use crate::lapack_wrapper::{complex_to_rgb_just_hue, display_special_matrix, display_system, solve_systems, special_block_tridiagonal_solve};
use crate::schrodinger::plot_result;
use plotly::color::Rgb as PlotlyRgb;
use crate::bases::{BackwardGrid, Basis, Grid, HarmonicEigenfunctions, LagrangePolynomials};
use crate::matrices::{SymTridiagonalMatrix, Matrix, LowerTriangularConstBandedMatrix};
use crate::solvers::Problem;

pub trait Tdse1dSpatialBasis: Basis {
    type Matrix: Matrix;

    fn get_hamiltonian(&self, problem: &Tdse1dProblem, t: f64) -> Self::Matrix;

    fn get_hamiltonian_element(&self, problem: &Tdse1dProblem, t: f64, i: usize, j: usize) -> c64;

    fn eigenstate(&self, problem: &Tdse1dProblem, n: usize) -> Vec<c64>;
}

impl Tdse1dSpatialBasis for Grid {
    type Matrix = SymTridiagonalMatrix;

    fn get_hamiltonian(&self, problem: &Tdse1dProblem, t: f64) -> Self::Matrix {
        SymTridiagonalMatrix {
            diagonal: self.points.iter().map(|&x_i| c64::new(0.0, 0.0) +
                x_i * (problem.electric)(t) +
                (problem.potential)(x_i) +
                1.0 / (self.delta() * self.delta())
            ).collect(),
            off_diagonal: vec![(-1.0 / (2.0 * self.delta() * self.delta())).into(); self.points.len()-1]
        }
    }

    fn get_hamiltonian_element(&self, problem: &Tdse1dProblem, t: f64, i: usize, j: usize) -> c64 {
        (if i == j {
            self.points[i] * (problem.electric)(t) +
                (problem.potential)(self.points[i]) +
                1.0 / (self.delta() * self.delta())
        } else if i == j + 1 || i + 1 == j {
            -1.0 / (2.0 * self.delta() * self.delta())
        } else {
            0.0
        }).into()
    }

    fn eigenstate(&self, problem: &Tdse1dProblem, n: usize) -> Vec<c64> {
        get_eigenstate(n, &self.points, &problem.potential).unwrap().1
    }
}

impl Tdse1dSpatialBasis for HarmonicEigenfunctions {

    type Matrix = SymTridiagonalMatrix;
    fn get_hamiltonian(&self, problem: &Tdse1dProblem, t: f64) -> Self::Matrix {
        SymTridiagonalMatrix {
            diagonal: self.eigenvalues(),
            off_diagonal: (1..self.len())
                .map(|j| ((j as f64 / 2.0).sqrt() * (problem.electric)(t)).into())
                .collect(),
        }
    }

    fn get_hamiltonian_element(&self, problem: &Tdse1dProblem, t: f64, i: usize, j: usize) -> c64 {
        if i == j {
            self.eigenvalues()[i]
        } else if i == j + 1 || i + 1 == j {
            ((i.max(j) as f64 / 2.0).sqrt() * (problem.electric)(t)).into()
        } else {
            0.0.into()
        }
    }

    fn eigenstate(&self, _: &Tdse1dProblem, n: usize) -> Vec<c64> {
        let mut initial = vec![0.0.into(); self.len()];
        initial[n] = 1.0.into();

        initial
    }
}

impl Tdse1dSpatialBasis for BackwardGrid {
    type Matrix = LowerTriangularConstBandedMatrix;

    fn get_hamiltonian(&self, problem: &Tdse1dProblem, t: f64) -> Self::Matrix {
        let mut coefs = self.second_deriv_coefficients()
            .into_iter()
            .map(|coef| (-coef / (2.0 * self.delta() * self.delta())).into()).collect_vec();
        let center_coef = coefs.remove(0);

        LowerTriangularConstBandedMatrix {
            diagonal: self.points.iter().map(|&x_i| c64::new(0.0, 0.0) +
                x_i * (problem.electric)(t) +
                (problem.potential)(x_i) +
                center_coef
            ).collect(),
            band_consts: coefs
        }
    }

    fn get_hamiltonian_element(&self, problem: &Tdse1dProblem, t: f64, i: usize, j: usize) -> c64 {
        let coefs = self.second_deriv_coefficients();

        if i == j {
            c64::new(0.0, 0.0) +
                self.points[i] * (problem.electric)(t) +
                (problem.potential)(self.points[i]) +
                (-coefs[0] / (2.0 * self.delta() * self.delta()))
        } else if i < j && i + coefs.len() >= j {
            (-coefs[j - i] / (2.0 * self.delta() * self.delta())).into()
        } else {
            0.0.into()
        }
    }

    fn eigenstate(&self, problem: &Tdse1dProblem, n: usize) -> Vec<c64> {
        get_eigenstate(n, &self.points, &problem.potential).unwrap().1
    }
}

impl Problem for Tdse1dProblem {
    type In = (f64, f64);
    type Out = c64;
}

// pub struct Tdse1dSolution2 {
//     vector: Vec<c64>,
//     space_basis: Box<dyn Basis>,
//     time_basis: Box<dyn Basis>
// }
//
// impl Tdse1dSolution2 {
//     fn eval(&self, x: f64, t: f64) -> c64 {
//         let n_t = self.time_basis.len();
//         let n_x = self.space_basis.len();
//         let mut result = 0.0.into();
//         for i in 0..n_x {
//             result += self.time_basis.eval(t, &self.vector[i*n_t..(i+1)*n_t])
//         }
//
//         result
//     }
// }

pub(crate) fn plot_result_with_expected_prob<T: Into<c64>>(
    t_initial: f64,
    t_final: f64,
    x_initial: f64,
    x_final: f64,
    psi_computed: &impl Fn(f64, f64) -> T,
    psi_0: impl Fn(f64, f64) -> T,
    prob_expected: impl Fn(f64) -> f64,
    plot_name: &str
) {
    let num_points_t = 200;
    let num_points_x = 200;

    let mut xs = Vec::with_capacity(num_points_x*num_points_t);
    let mut ys = Vec::with_capacity(num_points_x*num_points_t);
    let mut zs = Vec::with_capacity(num_points_x*num_points_t);
    let mut is = Vec::with_capacity(2*num_points_x*num_points_t);
    let mut js = Vec::with_capacity(2*num_points_x*num_points_t);
    let mut ks = Vec::with_capacity(2*num_points_x*num_points_t);
    let mut colors = Vec::with_capacity(num_points_x*num_points_t);

    // let mut zs0 = Vec::with_capacity(num_points_x*num_points_t);
    // let mut colors0 = Vec::with_capacity(num_points_x*num_points_t);

    let mut ts = Vec::with_capacity(num_points_t);
    let mut probs = Vec::with_capacity(num_points_t);
    let mut probs_expected = Vec::with_capacity(num_points_t);
    let mut magnitudes = Vec::with_capacity(num_points_t);

    sample(num_points_t, t_initial, t_final, |t| {
        let mut prob = c64::new(0.0, 0.0);
        let mut magnitude = 0.0;

        sample(num_points_x, x_initial, x_final, |x| {
            let computed: c64 = psi_computed(x, t).into();
            let t_mid = (t - t_initial) / (t_final - t_initial);
            let show_black = t_mid < 0.01 || t_mid > 0.99 || (false && (x > x_initial + 2e-1 && psi_computed(x - 1e-1, t).into().norm() < computed.norm()) &&
                (x < x_final - 2e-1 && psi_computed(x + 1e-1, t).into().norm() < computed.norm()));
            // println!("    Ψ({t:.4}, {x:.4}) = {: ^20}", computed.to_string());
            xs.push(x);
            ys.push(t);
            zs.push(computed.norm().powi(2));

            if xs.len() % num_points_x > 0 && xs.len() / num_points_x < num_points_t - 1 {
                // triangle 1
                is.push(xs.len() - 1);
                js.push(xs.len() + num_points_x);
                ks.push(xs.len());
                // triangle 2
                is.push(xs.len() - 1);
                js.push(xs.len() + num_points_x);
                ks.push(xs.len() - 1 + num_points_x);

                let (r, g, b) = if show_black {
                    (0, 0, 0)
                } else { complex_to_rgb_just_hue(computed) };
                colors.push(PlotlyRgb::new(r, g, b));
                colors.push(PlotlyRgb::new(r, g, b));
            }

            // let psi0_computed = psi_0(x, t);
            // zs0.push(psi0_computed.magnitude().powi(2));
            // let (r0, g0, b0) = psi0_computed.rgb_just_hue();
            // colors0.push(PlotlyRgb::new(r0, g0, b0));

            prob += computed.conj() * psi_0(x, 0.0).into();
            magnitude += (computed.conj() * computed).re;
        });

        ts.push(t);
        probs.push(((x_final - x_initial) * prob / num_points_x as f64).norm().powi(2));
        probs_expected.push(prob_expected(t));
        magnitudes.push((x_final - x_initial) * magnitude / num_points_x as f64);
    });

    // xs.append(&mut vec![0.0, 0.0]);
    // ys.append(&mut vec![0.0, 0.0]);
    // zs.append(&mut vec![0.4, 1.3]);

    let mesh = Mesh3D::new(xs, ys, zs, is, js, ks)
        .name("Ψ(x,t)")
        .opacity(1.0)
        .flat_shading(true)
        .face_color(colors);

    // let scatter = Scatter3D::new(xs.clone(), ys.clone(), zs)
    //     .name("Ψ(x,t)")
    //     .line(Line::new().width(0.0))
    //     .mode(Mode::Markers)
    //     .marker(Marker::new().size(1).color_array(colors));

    // let scatter_psi_0 = Scatter3D::new(xs, ys, zs0)
    //     .name("Ψ_0(x,t)")
    //     .line(Line::new().width(0.0))
    //     .mode(Mode::Markers)
    //     .marker(Marker::new().size(1).color_array(colors0));

    let mut wave_func_plot = Plot::new();
    wave_func_plot.add_trace(mesh);
    wave_func_plot.set_configuration(Configuration::new().fill_frame(true).frame_margins(0.0));
    wave_func_plot.set_layout(
        Layout::new()
    );
    let mut html = wave_func_plot.to_html();

    let mag_scatter = Scatter::new(ts.clone(), magnitudes)
        .name("<Ψ*|Ψ>")
        .line(Line::new().width(1.0))
        .mode(Mode::Markers);

    let scatter = Scatter::new(ts.clone(), probs)
        .name("<Ψ*|Ψ_0>")
        .line(Line::new().width(1.0))
        .mode(Mode::Markers);

    let scatter_exp = Scatter::new(ts, probs_expected)
        .name("<Ψ*|Ψ_0> expected")
        .line(Line::new().width(1.0))
        .mode(Mode::Markers);

    let mut prob_plot = Plot::new();
    prob_plot.add_trace(scatter);
    prob_plot.add_trace(mag_scatter);
    prob_plot.add_trace(scatter_exp);
    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::new("position")))
    );
    let prob_html = prob_plot.to_html();

    html = html
        .replace("<head>", "<head><style>body { overflow: scroll!important; }</style>")
        .replace("<body>", "<body style=\"display: flex; flex-direction: column;\">")
        .replace("plotly-html-element", "plotly-html-element-wave")
        .replace("\"layout\": {", "\"layout\": {
          \"scene\": {
            \"xaxis\": {
              \"title\": {
                \"text\": \"position\"
              }
            },
            \"yaxis\": {
              \"title\": {
                \"text\": \"time\"
              }
            },
            \"zaxis\": {
              \"title\": {
                \"text\": \"Ψ(x,t)\"
              }
            }
          },
        ");

    html = format!("{}{}", &html[..(html.len() - 17)], &prob_html[264..]);
    println!("{}", format!("output/{}.html", plot_name));
    std::fs::write(format!("output/{}.html", plot_name), html).expect("Unable to write file");
}

// pub struct DiffeqOptions {
//     debug: bool,
//     plot_name: Option<String>,
//     show_matrix: bool
// }
//
// // If I wanted to generalize further and create a complicated API, this is what it'd be
// pub trait DiffeqBasis {
//     type Problem;
//     type Solution;
//
//     fn solve(&self, options: DiffeqOptions, problem: &Self::Problem, initial: Vec<c64>) -> Self::Solution;
// }

#[derive(Clone)]
pub struct Tdse1dOptions {
    initial_time_settings: Option<(f64, Vec<c64>)>,
    polys: LagrangePolynomials,
    t_final: f64,
    n_t: usize,
    nq_t: usize,
    xs: Vec<f64>,
    x_initial: f64,
    x_final: f64,
    n_x: usize,
    delta_x: f64,
    driven_state: Option<(f64, Vec<c64>)>,
    debug: bool,
    plot_name: Option<String>,
    show_matrix: bool
}

impl Tdse1dOptions {
    fn new(
        t_final: f64, n_t: usize, nq_t: usize,
        x_initial: f64, x_final: f64, n_x: usize
    ) -> Self {
        let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);

        Self {
            polys: LagrangePolynomials::new(n_t, nq_t, 0.0, t_final),
            t_final, n_t, nq_t,
            xs: (0..n_x).map(|i| i as f64 * delta_x + x_initial).collect(),
            x_initial, x_final, n_x, delta_x,
            ..Self::default()
        }
    }

    fn with_debug(mut self) -> Self {
        self.debug = true;
        self
    }

    fn with_plot(mut self, plot_name: String) -> Self {
        self.plot_name = Some(plot_name);
        self
    }

    fn with_matrix(mut self) -> Self {
        self.show_matrix = true;
        self
    }

    fn with_driven_state(mut self, problem: &Tdse1dProblem, n: usize) -> Self {
        let (energy, state) = get_eigenstate(n, &self.xs, &problem.potential)
            .expect("Couldn't get eigenstate");

        self.driven_state = Some((energy, state));

        self
    }

    fn set_new_interval(&mut self, t_initial: f64, t_final: f64, psi_initial: Vec<c64>) {
        self.initial_time_settings = Some((t_initial, psi_initial));
        self.t_final = t_final;
        self.polys = LagrangePolynomials::new(self.n_t, self.nq_t, t_initial, t_final);
    }

    fn t_initial(&self) -> f64 {
        self.initial_time_settings.as_ref().map(|(t, _)| *t).unwrap_or(0.0)
    }

    fn psi_0_vec(&self, t: f64) -> Vec<c64> {
        if let Some((energy, state)) = self.driven_state.clone() {
            state.iter().map(|&psi| c64::exp(&(-c64::i() * energy * t)) * psi).collect()
        } else {
            vec![0.0.into(); self.n_x]
        }
    }

    fn psi_0<'a>(&self) -> Box<dyn Fn(usize, f64) -> c64 + 'static> {
        if let Some((energy, state)) = self.driven_state.clone() {
            Box::new(move |n, t| {
                c64::exp(&(-c64::i() * energy * t)) * state[n]
            })
        } else {
            Box::new(|_n, _t| 0.0.into())
        }
    }
}

impl Default for Tdse1dOptions {
    fn default() -> Self {
        Tdse1dOptions {
            initial_time_settings: None,
            polys: LagrangePolynomials::default(),
            t_final: 100.0,
            n_t: 100,
            nq_t: 1,
            xs: Vec::new(),
            x_initial: -2e2,
            x_final: 2e2,
            n_x: 4001,
            delta_x: 0.1,
            driven_state: None,
            debug: false,
            plot_name: None,
            show_matrix: false
        }
    }
}

pub struct Tdse1dProblem {
    pub potential: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>,
    pub electric: Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>
}

impl Tdse1dProblem {
    pub fn two_level_atom(
        electric_0: f64,
        pulse_final: f64
    ) -> Self {
        Self {
            potential: Box::new(move |x| -1.0 / x.abs()),
            // potential: Box::new(move |x| -1.0 / (1.0 + x.powi(2)).sqrt()),
            // potential: Box::new(move |_x| 0.0),
            electric: Box::new(move |t| {
                0.5 * electric_0 * (PI * t / pulse_final).sin().powi(2)
            }),
        }
    }

    pub fn harmonic_oscillator(
        angular_frequency: f64,
        electric_0: f64,
        pulse_final: f64
    ) -> Self {
        Self {
            potential: Box::new(move |x| 0.5 * x.powi(2)),
            electric: Box::new(move |t| {
                electric_0 * (PI * t / pulse_final).sin().powi(2) * (angular_frequency * t).cos()
            }),
        }
    }

    pub fn hydrogen_laser_pulse(
        angular_frequency: f64,
        electric_0: f64,
        phase_shift: f64,
        smooth_pulse: bool,
        pulse_initial: f64,
        pulse_final: f64
    ) -> Self {
        Self {
            potential: Box::new(move |x| -1.0 / (1.0 + x.powi(2)).sqrt()),
            electric: Box::new(move |t| {
                let pulse = if t < pulse_initial || pulse_final < t {
                    0.0
                } else if smooth_pulse {
                    (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
                } else { 1.0 };

                electric_0 * (angular_frequency * t + phase_shift).sin() * pulse
            }),
        }
    }
}

/*
pub struct Tdse1dBasisGrid {
    ts: LagrangePolynomials,
    xs: Grid
}

impl Tdse1dBasisGrid {
    pub fn new(
        n_t: usize, t_i: f64, t_f: f64,
        n_x: usize, x_i: f64, x_f: f64
    ) -> Self {
        Self {
            ts: LagrangePolynomials::new(n_t, 1, t_i, t_f),
            xs: Grid::new(n_x, x_i, x_f)
        }
    }
}

impl DiffeqBasis for Tdse1dBasisGrid {
    type Problem = Tdse1dProblem;
    type Solution = Tdse1dSolution;

    fn solve(&self, options: DiffeqOptions, problem: &Self::Problem, initial: Vec<c64>) -> Self::Solution {
        let (n_x, n_t) = (self.xs.len(), self.ts.len());
        let mut matrix: Vec<c64> = vec![0.0.into(); n_t * n_t * n_x];
        let mut rhs: Vec<c64> = vec![0.0.into(); n_t * n_x];

        let DiffeqOptions {
            debug, plot_name, show_matrix
        } = options;

        let Tdse1dProblem {
            potential, electric, ..
        } = problem;

        let beta = 1.0 / (2.0 * self.xs.delta().powi(2));

        let mut matrix: Vec<c64> = vec![0.0.into(); n_t * n_t * n_x];
        let mut rhs: Vec<c64> = vec![0.0.into(); n_t * n_x];

        for n in 0..n_x {
            let x_n = self.xs.point(n);

            for index_it in 0..n_t {
                let index_i = index_it + n * n_t * n_t;
                let t_i = self.ts.point(index_i);
                let alpha = potential(x_n) + x_n * electric(t_i) + 1.0 / self.xs.delta().powi(2);

                for index_jt in 0..n_t {
                    matrix[index_it + n_t * index_jt + n * n_t * n_t] = (
                        c64::i() * self.ts.l_deriv(index_jt, t_i) -
                            if index_it == index_jt { alpha } else { 0.0 }
                    ).into();
                }
            }
        }

        for n in 0..n_x {
            rhs[n * n_t] = initial[n];

            if n > 0 {
                rhs[n * n_t] += beta * initial[n-1];
            }

            if n < n_x - 1 {
                rhs[n * n_t] += beta * initial[n+1];
            }

            for index in 0..n_t {
                matrix[index * n_t + n * n_t * n_t] = (if index == 0 { 1.0 } else { 0.0 }).into()
            }
        }

        if show_matrix {
            println!("Matrix:");
            display_special_matrix(beta.into(), &matrix, &rhs, n_t, n_x);
        }

        let result = match special_block_tridiagonal_solve(beta.into(), matrix, rhs, n_x, n_t) {
            Ok(sol) => sol,
            Err(err) => {
                panic!("Couldn't lapack: {:?}", err);
            }
        };

        let solution = Tdse1dSolution {
            vector: result,
            t_basis_fns: self.ts.clone().into_basis_fns(),
            dims_t: n_t,
            xs: self.xs.points.clone(),
            psi_0: Box::new(|_, _| 0.0.into()),
        };

        if debug && plot_name.is_some() { println!("PLOTTING"); }

        if let Some(plot_name) = plot_name {
            let (ground_energy, ground_state) = get_eigenstate(0, &solution.xs, &problem.potential)
                .expect("Couldn't get eigenstate");
            let psi_0 = |x: f64, t: f64| {
                (-c64::i() * ground_energy * t).exp() * solution.interp_x(x, |n| ground_state[n])
            };

            plot_result(
                self.ts.first(), self.ts.last(),
                self.xs.first(), self.xs.last(),
                &solution.get_wave_fn(), &psi_0, &plot_name
            );
        }

        solution
    }
}

pub struct Tdse1dBasisEigenfunctions {
    ts: LagrangePolynomials,
    num_eigenfunctions: usize,
}

impl Tdse1dBasisEigenfunctions {
    pub fn new(
        n_t: usize, t_i: f64, t_f: f64,
        num_eigenfunctions: usize
    ) -> Self {
        Self {
            ts: LagrangePolynomials::new(n_t, 1, t_i, t_f),
            num_eigenfunctions
        }
    }

    pub fn psi_n(&self, n: usize, x: f64) -> c64 {
        let n_fact = (1..=n).product::<usize>() as f64;
        let c_n = PI.powf(-0.25) / (n_fact * 2f64.powi(n as i32)).sqrt();

        (c_n * hermite(n, x) * (-x*x / 2.0).exp()).into()
    }
}*/

pub struct Tdse1dSolution {
    vector: Vec<c64>,
    t_basis_fns: Box<dyn Fn(usize, f64) -> f64 + 'static>,
    dims_t: usize,
    xs: Vec<f64>,
    psi_0: Box<dyn Fn(usize, f64) -> c64 + 'static>
}

impl Tdse1dSolution {
    fn assert_in_bounds_x(&self, x: f64) {
        if !(self.xs[0] <= x && x <= self.xs[self.xs.len() - 1]) {
            panic!("x: {x} is out of bounds, {:.4} <= {x:.4} <= {:.4}", self.xs[0], self.xs[self.xs.len() - 1]);
        }
    }

    fn interp_x<T>(&self, x: f64, func: impl Fn(usize) -> T) -> T where
        f64: std::ops::Mul<T, Output = T>,
        T: std::ops::Add<T, Output = T>
    {
        let (x_initial, n_x) = (self.xs[0], self.xs.len());
        let delta_x = self.xs[1] - self.xs[0];
        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = (((x - x_initial) / delta_x).ceil() as usize).min(n_x - 1);
        let x_mid = (x - self.xs[n1]) / delta_x;

        (1.0 - x_mid) * func(n1) + x_mid * func(n2)
    }

    fn get_wave_fn<'a>(&'a self) -> impl Fn(f64, f64) -> c64 + 'a {
        move |x: f64, t: f64| {
            self.assert_in_bounds_x(x);

            self.interp_x(x, |n| (self.psi_0)(n, t)) + (0..self.dims_t).map(|index| {
                self.interp_x(x, |n| self.vector[index + n * self.dims_t]) * (self.t_basis_fns)(index, t)
            }).fold(c64::new(0.0, 0.0), |acc, z| acc + z)
        }
    }

    fn psi(&self, n: usize, t: f64) -> c64 {
        (self.psi_0)(n, t) + (0..self.dims_t).map(|index| {
            self.vector[index + n * self.dims_t] * (self.t_basis_fns)(index, t)
        }).fold(c64::new(0.0, 0.0), |acc, z| acc + z)
    }

    fn get_population_probability_fn<'a>(&'a self, n: usize, potential: &impl Fn(f64) -> f64) -> impl Fn(f64) -> f64 + 'a {
        let (_, state) = get_eigenstate(n, &self.xs, &potential).expect("Couldn't get eigenstate");
        let n_x = self.xs.len();
        let delta_x = self.xs[1] - self.xs[0];

        move |t| {
            (0..n_x).map(|n| {
                self.psi(n, t).conj() * state[n] * delta_x
            }).fold(c64::from(0.0), |acc, z| acc + z).norm_sqr()
        }
    }

    fn get_norm_fn<'a>(&'a self) -> impl Fn(f64) -> f64 + 'a {
        let n_x = self.xs.len();
        let delta_x = self.xs[1] - self.xs[0];

        move |t| {
            (0..n_x).map(|n| {
                self.psi(n, t).norm_sqr() * delta_x
            }).sum::<f64>().sqrt()
        }
    }
}

/*
#[test]
fn test_two_level_atom_solve() {
    let t_final = 9000.0;
    let num_intervals = 90;

    let options = Tdse1dOptions {
        t_final: t_final / num_intervals as f64,
        n_t: 40,
        nq_t: 1,
        x_initial: -5e0,
        x_final: 5e0,
        n_x: 101,
        ..Default::default()
    }.with_debug();

    let e0 = 2.0 * PI / 9000.0;
    let pulse_final = 9000.0;
    let problem = Tdse1dProblem::two_level_atom(&options, e0, pulse_final);

    let (_, ground_state) = ground_state(
        options.n_x, options.x_initial, options.x_final, &problem.potential
    ).unwrap();

    let wave_function = repeated_solve_tdse_1d(options.clone(), &problem, num_intervals);

    let prob_ground_expected = |t| {
        f64::cos(0.25 * e0 * (t - pulse_final / (2.0 * PI) * f64::sin(2.0 * PI * t / pulse_final))).powi(2)
    };

    let prob_ground_computed = |t| {
        qk61(|x| (wave_function(x, t).conj() * c64::from(ground_state(x))).norm(), options.x_initial, options.x_final).0
    };

    plot_result_with_expected_prob(
        options.t_initial(),
        t_final,
        options.x_initial,
        options.x_final,
        &wave_function,
        |x, t| c64::from(ground_state(x)),
        prob_ground_expected,
        "two_level_atom"
    );

    let mut err_max: f64 = 0.0;

    println!("c_g(______) =       Expected       vs       Computed      ");

    sample(1000, 0.0, t_final, |t| {
        let expected = prob_ground_expected(t);
        let computed = prob_ground_computed(t);
        let err = (expected - computed).abs();
        println!("c_g({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", expected.to_string(), computed.to_string());
        err_max = err_max.max(err);
    });

    println!("MAX ERROR: {:.4e}", err_max);
}
*/

#[test]
fn show_off_init_conditions() {
    let problem = Tdse1dProblem::harmonic_oscillator(
        0.0, 1.0, 100.0
    );

    let mut options = Tdse1dOptions::new(
        10.0, 4, 1,
        -2e1, 2e1, 5
    ).with_debug().with_matrix();

    let (_, ground_state) = get_eigenstate(0, &options.xs, &problem.potential)
        .expect("Couldn't get eigenstate");

    options.set_new_interval(0.0, 10.0, ground_state);

    let _ = solve_tdse_1d(options.clone(), &problem);
}

#[test]
fn test_harmonic_solve() {
    let problem = Tdse1dProblem::harmonic_oscillator(
        1.0, 1.0, 100.0
    );

    let options = Tdse1dOptions::new(
        10.0, 20, 1,
        -5e1, 5e1, 100001
    ).with_debug().with_driven_state(&problem, 0);//.with_plot("harmonic_oscillator".to_string());

    let solution = repeated_solve_tdse_1d(options.clone(), &problem, 10, solve_tdse_1d_optimized);

    let prob_ground_expected = harmonic_ground_state_probability(1.0, 1.0, 100.0);
    let prob_ground_computed = solution.get_population_probability_fn(0, &problem.potential);
    let norm_computed = solution.get_norm_fn();

    println!("P_0(______) =       Expected       vs       Computed      ");

    let mut max_err: f64 = 0.0;
    let mut xs = Vec::new();
    let mut ys_e = Vec::new();
    let mut ys_c = Vec::new();
    let mut ys_n = Vec::new();
    let mut ys_err = Vec::new();

    sample(200, 0.0, 100.0, |t| {
        let expected = prob_ground_expected(t, 0);
        let computed = prob_ground_computed(t);
        let err = (expected - computed).abs();
        println!("P_0({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", format!("{:.6}", expected), format!("{:.6}", computed));
        max_err = max_err.max(err);

        xs.push(t);
        ys_e.push(expected);
        ys_c.push(computed);
        ys_n.push(norm_computed(t));
        ys_err.push(err);
    });

    let mut prob_plot = Plot::new();

    let mut add_ys = |name: &str, ys: Vec<f64>| {
        let scatter = Scatter::new(xs.clone(), ys)
            .name(name)
            .line(Line::new().width(2.0))
            .marker(Marker::new().size(4))
            .mode(Mode::LinesMarkers);

        prob_plot.add_trace(scatter);
    };

    add_ys("Expected", ys_e);
    add_ys("Computed", ys_c);
    add_ys("Norm", ys_n);
    add_ys("Error", ys_err);

    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.set_layout(
        Layout::new()
            .x_axis(Axis::new().title(Title::new("time")))
            .y_axis(Axis::new().title(Title::new("P_0")))
    );
    prob_plot.write_html("output/harmonic_error.html");

    println!("MAX ERROR: {:e}", max_err);
}

#[test]
fn test_hydrogen_solve() {
    let problem = Tdse1dProblem::hydrogen_laser_pulse(
        0.148, 0.1, 0.0,
        false, 0.0, 1200.0
    );

    let options = Tdse1dOptions::new(
        10.0, 100, 1,
        -2e2, 2e2, 1001
    ).with_debug()
        .with_plot("hydrogen_atom".to_string())
        .with_driven_state(&problem, 0);

    let _wave_function = repeated_solve_tdse_1d(options, &problem, 10, solve_tdse_1d_optimized);
}

pub fn solve_tdse_1d(options: Tdse1dOptions, problem: &Tdse1dProblem) -> Tdse1dSolution {
    let psi_0 = options.psi_0();
    let Tdse1dOptions {
        initial_time_settings, polys, t_final,
        xs, x_initial, x_final, n_x, delta_x,
        debug, plot_name, show_matrix, ..
    } = options;

    let Tdse1dProblem { potential, electric, .. } = problem;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    // let psi_initial = initial_time_settings.as_ref()
    //     .map(|(_, psi)| psi.clone())
    //     .unwrap_or(vec![0.0.into(); n_x]);

    let beta = 1.0 / (2.0 * delta_x.powi(2));

    let dims_t = polys.dims;
    let dims = dims_t * n_x;
    let mut matrix: Vec<c64> = vec![0.0.into(); dims.pow(2)];

    let mut vector: Vec<c64> = vec![0.0.into(); dims];

    for n in 0..n_x {
        for index_it in 0..dims_t {
            let index_i = index_it + n * dims_t;
            let t_i = polys.point(index_it);
            let alpha = potential(xs[n]) + xs[n] * electric(t_i) + 1.0 / delta_x.powi(2);

            vector[index_i] = (xs[n] * electric(t_i) * psi_0(n, t_i)).into();

            for index_jt in 0..dims_t {
                let index_j = index_jt + n * dims_t;
                
                matrix[index_i + dims * index_j] = (c64::i() * polys.l_deriv(index_jt, t_i) - if index_it == index_jt { alpha } else { 0.0 }).into();
                
                if n > 0 && index_i == index_j {
                    matrix[index_i + dims * (index_jt + (n-1) * dims_t)] = beta.into();
                }
                
                if n < n_x - 1 && index_i == index_j {
                    matrix[index_i + dims * (index_jt + (n+1) * dims_t)] = beta.into();
                }
            }
        }
    }

    // for n in 0..n_x {
    //     vector[n * dims_t] = psi_initial[n].into();
    //
    //     for index in 0..dims {
    //         matrix[index * dims + n * dims_t] = if n == index / dims_t {
    //             polys.l(index % dims_t, t_initial).into()
    //         } else { 0.0.into() };
    //     }
    // }

    if show_matrix {
        println!("Matrix:");
        display_system(&matrix, &vector, dims);
    }

    let result = solve_systems(matrix, dims, vector)
        .expect("Couldn't solve systems of equations");

    let solution = Tdse1dSolution {
        vector: result,
        t_basis_fns: polys.into_basis_fns(),
        dims_t,
        xs,
        psi_0,
    };

    if debug && plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = plot_name {
        let (ground_energy, ground_state) = get_eigenstate(0, &solution.xs, &problem.potential)
            .expect("Couldn't get eigenstate");
        let psi_0 = |x: f64, t: f64| {
            (-c64::i() * ground_energy * t).exp() * solution.interp_x(x, |n| ground_state[n])
        };

        plot_result(t_initial, t_final, x_initial, x_final, &solution.get_wave_fn(), &psi_0, &plot_name);
    }

    solution
}

pub fn solve_tdse_1d_optimized(options: Tdse1dOptions, problem: &Tdse1dProblem) -> Tdse1dSolution {
    let psi_0 = options.psi_0();
    let Tdse1dOptions {
        initial_time_settings, polys, t_final,
        xs, x_initial, x_final, n_x, delta_x,
        debug, plot_name, show_matrix, ..
    } = options;

    let Tdse1dProblem { potential, electric, .. } = problem;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    let psi_initial = initial_time_settings.as_ref()
        .map(|(_, psi)| psi.clone())
        .unwrap_or(vec![0.0.into(); n_x]);

    let beta = 1.0 / (2.0 * delta_x.powi(2));

    let dims_t = polys.len();
    let dims = dims_t * n_x;
    let mut matrix: Vec<c64> = vec![0.0.into(); dims_t*dims_t * n_x];

    let mut rhs: Vec<c64> = vec![0.0.into(); dims];

    for n in 0..n_x {
        for index_it in 0..dims_t {
            // let index_i = index_it + n * dims_t*dims_t;
            let t_i = polys.point(index_it);
            let alpha = potential(xs[n]) + xs[n]*electric(t_i) + 1.0 / delta_x.powi(2);

            rhs[index_it + n*dims_t] = (xs[n]*electric(t_i) * psi_0(n, t_i)).into();

            for index_jt in 0..dims_t {
                matrix[index_it + dims_t * index_jt + n * dims_t*dims_t] = (
                    c64::i() * polys.l_deriv(index_jt, t_i) -
                        if index_it == index_jt { alpha } else { 0.0 }
                ).into();
            }
        }
    }

    for n in 0..n_x {
        rhs[n * dims_t] = psi_initial[n];

        if n > 0 {
            rhs[n * dims_t] += beta * psi_initial[n-1];
        }

        if n < n_x - 1 {
            rhs[n * dims_t] += beta * psi_initial[n+1];
        }

        for index in 0..dims_t {
            matrix[index * dims_t + n * dims_t*dims_t] = (if index == 0 { 1.0 } else { 0.0 }).into()
        }
    }

    if show_matrix {
        println!("Matrix:");
        display_special_matrix(beta.into(), &matrix, &rhs, dims_t, n_x);
    }

    let result = match special_block_tridiagonal_solve(beta.into(), matrix, rhs, n_x, dims_t) {
        Ok(sol) => sol,
        Err(err) => {
            panic!("Couldn't lapack: {:?}", err);
        }
    };

    let solution = Tdse1dSolution {
        vector: result,
        t_basis_fns: polys.into_basis_fns(),
        dims_t,
        xs,
        psi_0,
    };

    if debug && plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = plot_name {
        let (ground_energy, ground_state) = get_eigenstate(0, &solution.xs, &problem.potential)
            .expect("Couldn't get eigenstate");
        let psi_0 = |x: f64, t: f64| {
            (-c64::i() * ground_energy * t).exp() * solution.interp_x(x, |n| ground_state[n])
        };

        plot_result(t_initial, t_final, x_initial, x_final, &solution.get_wave_fn(), &psi_0, &plot_name);
    }

    solution
}

pub fn repeated_solve_tdse_1d(
    mut options: Tdse1dOptions,
    problem: &Tdse1dProblem,
    num_intervals: usize,
    solver: impl Fn(Tdse1dOptions, &Tdse1dProblem) -> Tdse1dSolution
) -> Tdse1dSolution {
    let plot_name = options.plot_name.take();

    let t_initial = options.t_initial();
    let time_final = num_intervals as f64 * (options.t_final - t_initial) + t_initial;
    let delta_time = options.t_final - t_initial;
    let num_intervals = ((time_final / delta_time) as f64).ceil() as usize;
    let dims_t = options.polys.dims;
    let mut wave_func_vector = vec![0.0.into(); options.n_x];
    let mut t_basis_fns: Vec<Box<dyn Fn(usize, f64) -> f64>> = Vec::new();
    let mut solution = vec![0.0.into(); options.n_x * num_intervals * dims_t];

    if options.debug { println!("Getting the ground state"); }

    for k in 0..num_intervals {
        if options.debug {
            let time_start = k as f64 * delta_time;
            let time_end = (k + 1) as f64 * delta_time;

            println!("Interval: {}/{num_intervals}", k + 1);
            println!("Time: {time_start} -> {time_end}");
        }

        let wave_function = solver(options.clone(), &problem);

        for n in 0..options.n_x {
            wave_func_vector[n] = wave_function.vector[n * dims_t + dims_t - 1];

            for index_t in 0..dims_t {
                solution[n * dims_t * num_intervals + k * dims_t + index_t] =
                    wave_function.vector[n * dims_t + index_t];
            }
        }

        if options.debug { println!(); }

        options.set_new_interval(
            (k + 1) as f64 * delta_time,
            (k + 2) as f64 * delta_time,
            wave_func_vector.clone()
        );

        t_basis_fns.push(Box::new(wave_function.t_basis_fns));
    }

    let psi_0 = options.psi_0();
    let solution = Tdse1dSolution {
        vector: solution,
        t_basis_fns: Box::new(move |index, t| {
            let k = index / dims_t;

            if k as f64 * delta_time + t_initial <= t && t < (k + 1) as f64 * delta_time + t_initial {
                t_basis_fns[k](index % dims_t, t)
            } else { 0.0 }
        }),
        dims_t: dims_t * num_intervals,
        xs: options.xs,
        psi_0,
    };

    if options.debug && plot_name.is_some() { println!("Plotting the result"); }

    if let Some(plot_name) = plot_name {
        let (ground_energy, ground_state) = get_eigenstate(0, &solution.xs, &problem.potential)
            .expect("Couldn't get eigenstate");
        let psi_0 = |x: f64, t: f64| {
            (-c64::i() * ground_energy * t).exp() * solution.interp_x(x, |n| ground_state[n])
        };

        plot_result(
            t_initial, time_final,
            options.x_initial, options.x_final,
            &solution.get_wave_fn(),
            &psi_0,
            &plot_name
        );
    }

    solution
}

pub fn harmonic_ground_state_probability(w0: f64, e0: f64, pulse_final: f64) -> impl Fn(f64, usize) -> f64 {
    let alpha = 2.0 * PI / pulse_final;

    move |t, n| {
        let (i1, i2) = if w0 == 0.0 {(
            -e0/4.0 * (2.0 * t.sin() - ((alpha-1.0)*t).sin()/(alpha-1.0) - ((alpha+1.0)*t).sin()/(alpha+1.0)),
            -e0/4.0 * (-2.0 * t.cos() + 2.0 + ((alpha+1.0)*t).cos()/(alpha+1.0) - 1.0 / (alpha+1.0) - ((alpha-1.0)*t).cos()/(alpha-1.0) + 1.0 / (alpha-1.0))
        )} else if w0 == 1.0 {(
            // copilot wrote this...
            -e0/8.0 * (2.0 * t - 2.0 * (alpha * t).sin() / alpha + (2.0 * t).sin() - ((alpha-2.0)*t).sin()/(alpha-2.0) - ((alpha+2.0)*t).sin()/(alpha+2.0)),
            -e0/8.0 * (-(2.0 * t).cos() + 1.0 + ((alpha+2.0)*t).cos()/(alpha+2.0) - 1.0 / (alpha+2.0) - ((alpha-2.0)*t).cos()/(alpha-2.0) + 1.0 / (alpha-2.0))
        )} else { panic!("Solution for ω={} not currently implemented", w0) };

        let x_0 = t.sin() * i1 - t.cos() * i2;
        let p_0 = t.cos() * i1 + t.sin() * i2;

        let g_t = (x_0 + c64::i() * p_0).powf(2.0)/4.0 - x_0*x_0/2.0;
        let h_t = (x_0 + c64::i() * p_0) / 2.0;

        let mut prob = g_t.exp().norm_sqr();

        for i in 1..=n {
            prob = 2.0 / (i as f64) * h_t.norm_sqr() * prob;
        }

        prob
    }
}

pub fn get_eigenstate(
    n: usize, xs: &[f64],
    potential: &impl Fn(f64) -> f64
) -> Result<(f64, Vec<c64>), LapackError> {
    let delta_x = xs[1] - xs[0];
    let beta = -1.0 / (2.0 * delta_x.powi(2));
    let mut info = 0;

    let mut diagonal = xs.iter()
        .map(|&x| -2.0 * beta + potential(x))
        .collect::<Vec<f64>>();
    let mut off_diagonal = vec![beta; xs.len() - 1];
    let mut workspace=  vec![0.0; 4 * xs.len()];

    zsteqr(
        'N' as u8,
        xs.len() as i32,
        &mut diagonal[..],
        &mut off_diagonal[..],
        &mut [][..],
        xs.len() as i32,
        &mut workspace[..],
        &mut info
    );

    let mut eigenvector = vec![1.0.into(); xs.len()];
    let state_energy = diagonal[n];
    let mut diagonal = xs.iter()
        .map(|&x| (-2.0 * beta + potential(x) - state_energy).into())
        .collect::<Vec<c64>>();

    for _ in 0..10 {
        let mut info = 0;

        diagonal.iter_mut()
            .zip(xs.iter())
            .for_each(|(d, &x)| *d = (-2.0 * beta + potential(x) - state_energy).into());

        zgtsv(
            xs.len() as i32,
            1,
            &mut vec![beta.into(); xs.len() - 1][..],
            &mut diagonal[..],
            &mut vec![beta.into(); xs.len() - 1][..],
            &mut eigenvector,
            xs.len() as i32,
            &mut info
        );

        let norm = eigenvector.iter()
            .map(|x| x.norm_sqr())
            .sum::<f64>()
            .sqrt();

        eigenvector.iter_mut().for_each(|x| *x = *x / norm);

        if info != 0 {
            panic!("lapack error on zgtsv(...): {info}")
        }
    }

    eigenvector.iter_mut().for_each(|x| *x = *x / delta_x.sqrt());

    match info {
        0 => Ok((state_energy, eigenvector)),
        i @ ..=-1 => Err(LapackError {
            info: i,
            message: format!("The value at {i} had an illegal value")
        }),
        i => Err(LapackError {
            info: i,
            message: format!("The algorithm has failed to find all the eigenvalues in a total of 30*N iterations, {i} elements of E have not converged to zero")
        })
    }
}

#[test]
fn test_get_eigenstate() {
    let n_x: usize = 3001;
    let x_initial: f64 = -100.0;
    let x_final: f64 = 100.0;
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let xs = (0..n_x).map(|i| i as f64 * delta_x + x_initial).collect::<Vec<f64>>();
    let potential = |x: f64|  -1.0 / (1.0 + x.powi(2)).sqrt();
    let mut plot = Plot::new();

    for n in 100..101 {
        let (energy, state) = get_eigenstate(
            n,
            &xs,
            &potential
        ).unwrap();

        let mut ts: Vec<f64> = Vec::new();
        let mut ys = Vec::new();
        let mut magnitude_sqr = 0.0;
        for i in 0..n_x {
            ts.push(xs[i]);
            ys.push(state[i].norm());

            magnitude_sqr += state[i].norm_sqr() * delta_x;
        }

        let scatter = Scatter::new(ts, ys)
            .name(format!("Ψ_{}(x)", n))
            .line(Line::new().width(0.0))
            .mode(Mode::Markers);

        plot.add_trace(scatter);
        println!("Ψ_{}(x) -- norm: {:.4}, energy: {:.4}", n, magnitude_sqr.sqrt(), energy);
    }
    plot.set_configuration(Configuration::new().fill_frame(true));
    plot.write_html("output/eigenstates.html");
}