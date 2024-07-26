use std::f64::consts::PI;
use std::io::Write;
use lapack::c64;
use lapack::fortran::{zgtsv};
use plotly::{Configuration, Layout, Plot, Scatter};
use plotly::common::{Line, Marker, Mode, Title};
use plotly::layout::Axis;
use crate::tdse_1d::{get_eigenstate, harmonic_ground_state_probability};

#[test]
fn crank_nicolson() {
    let potential = |x: f64| 0.5 * x*x;
    let e0 = 1.0;
    let pulse_final = 100.0;
    let w0 = 0.0;
    let time_dependence = |x, t: f64| {
        x * e0 * (PI * t / pulse_final).sin().powi(2) * (w0 * t).cos()
    };

    let (t_i, t_f) = (0.0, 100.0);
    let delta_t = 0.0001;
    let num_iters = ((t_f - t_i) / delta_t) as usize;

    let (x_i, x_f) = (-2e2, 2e2);
    let n_x = 4001;
    let delta_x = (x_f - x_i) / (n_x as f64 - 1.0);
    let xs: Vec<f64> = (0..n_x).map(|i| x_i + (i as f64) * delta_x).collect();
    let (_, ground_state) = get_eigenstate(0, &xs, &potential).unwrap();
    let mut psi = ground_state.clone();

    let beta = 1.0 / (2.0 * delta_x * delta_x);

    println!("P_0(______) =       Expected       vs       Computed      ");

    let mut max_err: f64 = 0.0;
    let mut ts = Vec::new();
    let mut ys_e = Vec::new();
    let mut ys_c = Vec::new();
    let mut ys_n = Vec::new();
    let mut ys_err = Vec::new();
    let prob_ground_expected = harmonic_ground_state_probability(w0, e0, pulse_final);

    let mut diagonals: Vec<c64> = vec![0.0.into(); n_x];
    let mut rhs: Vec<c64> = vec![0.0.into(); n_x];

    for i in 0..=num_iters {
        let t = t_i + (i as f64) * delta_t;

        for n in 0..n_x {
            let alpha = time_dependence(xs[n], t) + potential(xs[n]) + 1.0 / (delta_x * delta_x);

            rhs[n] = (1.0 - c64::i() * delta_t / 2.0 * alpha) * psi[n];
            diagonals[n] = 1.0 + c64::i() * delta_t / 2.0* alpha;

            if n > 0 {
                rhs[n] += c64::i() * delta_t / 2.0 * beta * psi[n-1];
            }

            if n < n_x - 1 {
                rhs[n] += c64::i() * delta_t / 2.0 * beta * psi[n+1];
            }
        }

        if i % (num_iters / 5000) == 0 {
            print!("█");
            std::io::stdout().flush().unwrap();
        }

        if i % (num_iters / 200) == 0 {
            let expected = prob_ground_expected(t, 0);
            let mut computed: c64 = 0.0.into();
            let mut norm = 0.0;

            for n in 0..n_x {
                computed += psi[n].conj() * ground_state[n] * delta_x;
                norm += psi[n].norm_sqr() * delta_x;
            }

            let err = (expected - computed.norm_sqr()).abs();
            max_err = max_err.max(err);
            print!("\nP_0({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}  ", format!("{:.6}", expected), format!("{:.6}", computed.norm_sqr()));

            ts.push(t);
            ys_e.push(expected);
            ys_c.push(computed.norm_sqr());
            ys_n.push(norm);
            ys_err.push(err);
        }

        let mut off_diagonals = vec![-c64::i() * beta * delta_t / 2.0; n_x - 1];

        zgtsv(
            n_x as i32,
            1,
            &mut off_diagonals.clone(),
            &mut diagonals,
            &mut off_diagonals,
            &mut rhs,
            n_x as i32,
            &mut 0
        );

        psi.clone_from_slice(&rhs);
    }

    let mut prob_plot = Plot::new();

    let mut add_ys = |name: &str, ys: Vec<f64>| {
        let scatter = Scatter::new(ts.clone(), ys)
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
    prob_plot.write_html("output/harmonic_error_crank.html");

    println!("MAX ERROR: {:e}", max_err);
}