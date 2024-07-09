use std::f64::consts::PI;
use lapack::c64;
use lapack::fortran::{zgtsv, zsteqr};
use plotly::{Configuration, Plot, Scatter};
use plotly::common::{Line, Mode};
use crate::complex_wrapper::{Complex, LapackError, ONE};
use crate::gauss_quadrature::gauss_lobatto_quadrature;
use crate::{lagrange, lagrange_deriv};

#[derive(Clone)]
pub struct Tdse1dOptions {
    initial_time_settings: Option<(f64, Vec<Complex>)>,
    t_final: f64,
    n_t: usize,
    nq_t: usize,
    x_initial: f64,
    x_final: f64,
    n_x: usize,
    potential: fn(f64) -> f64,
    time_dependence: fn(f64, f64) -> f64,
    debug: bool,
    plot_name: Option<String>,
    show_matrix: bool
}

impl Tdse1dOptions {
    fn t_initial(&self) -> f64 {
        self.initial_time_settings.as_ref().map(|(t, _)| *t).unwrap_or(0.0)
    }

    fn set_harmonic_oscillator(
        &mut self,
        angular_frequency: f64,
        electric_0: f64
    ) {
        let (t_initial, t_final) = (self.t_initial(), self.t_final);

        self.potential = |x| 0.5 * x.powi(2);
        self.time_dependence = |x, t| {
            x * electric_0 * (PI * (t - t_initial) / (t_final - t_initial)).sin().powi(2) * (angular_frequency * t).cos()
        };
    }

    fn set_hydrogen_laser_pulse(
        &mut self,
        angular_frequency: f64,
        electric_0: f64,
        phase_shift: f64,
        smooth_pulse: bool,
        pulse_initial: f64,
        pulse_final: f64
    ) {
        let (t_initial, t_final) = (self.t_initial(), self.t_final);

        self.potential = |x| -1.0 / (1.0 * x.powi(2)).sqrt();
        self.time_dependence = |x, t| {
            let pulse = if t < pulse_initial || pulse_final < t {
                0.0
            } else if  smooth_pulse {
                (PI * (t - t_initial) / (t_final - t_initial)).sin().powi(2)
            } else { 1.0 };

            x * electric_0 * (angular_frequency * t + phase_shift).sin() * pulse
        };
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
}

impl Default for Tdse1dOptions {
    fn default() -> Self {
        Tdse1dOptions {
            initial_time_settings: None,
            t_final: 100.0,
            n_t: 100,
            nq_t: 1,
            x_initial: -2e2,
            x_final: 2e2,
            n_x: 4001,
            potential: |_x| 0.0,
            time_dependence: |_x, _t| 0.0,
            debug: false,
            plot_name: None,
            show_matrix: false
        }
    }
}

pub fn ground_state(
    n_x: usize, x_initial: f64, x_final: f64,
    potential: &impl Fn(f64) -> f64
) -> Result<(f64, impl Fn(f64) -> Complex), LapackError> {
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = -1.0 / (2.0 * delta_x.powi(2));
    let mut info = 0;

    let mut diagonal = (0..n_x)
        .map(|i| -2.0 * beta + potential(i as f64 * delta_x + x_initial))
        .collect::<Vec<f64>>();
    let mut off_diagonal = vec![beta; n_x - 1];
    let mut workspace=  vec![0.0; 4 * n_x];

    zsteqr(
        'N' as u8,
        n_x as i32,
        &mut diagonal[..],
        &mut off_diagonal[..],
        &mut [][..],
        n_x as i32,
        &mut workspace[..],
        &mut info
    );

    let mut eigenvector = vec![ONE.into(); n_x];
    let ground_state_energy = diagonal[0];

    for _ in 0..10 {
        let mut diagonal = (0..n_x)
            .map(|i| (-2.0 * beta + potential(i as f64 * delta_x + x_initial) - ground_state_energy).into())
            .collect::<Vec<c64>>();
        let mut info = 0;

        zgtsv(
            n_x as i32,
            1,
            &mut vec![beta.into(); n_x - 1][..],
            &mut diagonal[..],
            &mut vec![beta.into(); n_x - 1][..],
            &mut eigenvector,
            n_x as i32,
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

    let magnitude = (0..n_x).map(|i| eigenvector[i].norm_sqr() * delta_x).sum::<f64>().sqrt();

    match info {
        0 => Ok((ground_state_energy, move |x: f64| {
            let n1 = ((x - x_initial) / delta_x).floor() as usize;
            let n2 = ((x - x_initial) / delta_x).ceil() as usize;
            let x_mid = (x - n1 as f64 * delta_x - x_initial) / delta_x;
            let psi_n = eigenvector[n1];
            let psi_n_plus_1 = if n2 == n_x { psi_n } else { eigenvector[n2] };

            (((1.0 - x_mid) * psi_n + x_mid * psi_n_plus_1) / magnitude).into()
        })),
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
fn test_ground_state() {
    let n_x: usize = 501;
    let x_initial: f64 = -100.0;
    let x_final: f64 = 100.0;
    let potential = |x: f64|  -1.0 / (1.0 + x.powi(2)).sqrt();

    let (_, ground_state) = ground_state(
        n_x,
        x_initial,
        x_final,
        &potential
    ).unwrap();

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for i in 0..5*n_x {
        let x_i = i as f64 * (x_final - x_initial) / ((5 * n_x) as f64 - 1.0) + x_initial;
        let computed = ground_state(x_i);

        xs.push(x_i);
        ys.push(computed.magnitude());

        println!("Ψ_0({:.4}) = {:.8}", x_i, computed);
    }

    let scatter = Scatter::new(xs, ys)
        .name("Ψ_0")
        .line(Line::new().width(0.0))
        .mode(Mode::Markers);

    let mut plot = Plot::new();
    plot.add_trace(scatter);
    plot.set_configuration(Configuration::new().fill_frame(true));
    plot.write_html("output/ground_state.html");
}

pub struct LagrangePolynomials {
    points: Vec<Vec<f64>>,
    weights: Vec<f64>
}

impl LagrangePolynomials {
    pub fn new(num_points: usize, num_intervals: usize, start: f64, end: f64) -> Self {
        let delta = (end - start) / (num_intervals as f64);

        let quad_points = gauss_lobatto_quadrature(num_points, start, start + delta);

        let points = (0..num_intervals)
            .map(|n| quad_points.iter().map(|&(t, _)| t + n as f64 * delta).collect())
            .collect();

        let weights = quad_points.iter().map(|&(_, w)| w).collect();

        Self { points, weights }
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.points.len(), self.weights.len())
    }

    pub fn get_indices(&self, index: usize) -> (usize, usize) {
        let (num_points, num_intervals) = self.dims();

        if index == 0 {
            (0, 0)
        } else if index == num_points * num_intervals - 1 {
            (num_intervals - 1, num_points - 1)
        } else {
            ((index - 1) / num_points, (index - 1) % num_points + 1)
        }
    }

    pub fn l(&self, index: usize, t: f64) -> f64 {
        let (num_points, num_intervals) = self.dims();
        let (q, i) = self.get_indices(index);

        if index != 0 && i == 0 && t < self.points[q][i] {
            lagrange(&self.points[q - 1], num_points - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != num_points * num_intervals - 1 && i == num_points - 1 && self.points[q][i] < t {
            lagrange(&self.points[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }

    pub fn l_deriv(&self, index: usize, t: f64) -> f64 {
        let (num_points, num_intervals) = self.dims();
        let (q, i) = self.get_indices(index);

        if index != 0 && i == 0 {
            lagrange_deriv(&self.points[q - 1], num_points - 1, t) + lagrange_deriv(&self.points[q], i, t)
        } else if index != num_points * num_intervals - 1 && i == num_points - 1 && self.points[q][i] < t {
            lagrange_deriv(&self.points[q + 1], 0, t) + lagrange_deriv(&self.points[q], i, t)
        } else {
            lagrange_deriv(&self.points[q], i, t) // Use the lagrange in the current interval
        }
    }
}
