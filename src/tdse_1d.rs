use std::f64::consts::PI;
use lapack::c64;
use lapack::fortran::{zgtsv, zsteqr};
use crate::complex_wrapper::{Complex, LapackError, ONE};
use crate::schrodinger::ground_state;

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