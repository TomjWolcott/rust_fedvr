use std::f64::consts::PI;
use std::io::Write;
use colored::Colorize;
use lapack::c64;
use lapack::fortran::{zgtsv, zsteqr};
use plotly::color::Rgb as PlotlyRgb;
use plotly::{Configuration, Layout, Mesh3D, Plot, Scatter};
use plotly::common::{Line, Mode, Title};
use plotly::layout::Axis;
use rayon::prelude::*;
use crate::complex_wrapper::{Complex, ComplexMatrix, ComplexVector, I, LapackError, ONE, ZERO};
use crate::gauss_quadrature::gauss_lobatto_quadrature;
use crate::{lagrange, lagrange_deriv, sample, tdse_1d};
use crate::lapack_wrapper::{complex_to_rgb, complex_to_rgb_just_hue};

/*fn plot_result2(
    t_initial: f64,
    t_final: f64,
    x_initial: f64,
    x_final: f64,
    psi_computed: &impl Fn(f64, f64) -> Complex,
    psi_0: impl Fn(f64, f64) -> Complex,
    plot_name: &str
) {
    let num_points_t = 800;
    let num_points_x = 800;

    let mut xs = Vec::with_capacity(num_points_x*num_points_t);
    let mut ys = Vec::with_capacity(num_points_x*num_points_t);
    let mut zs = Vec::with_capacity(num_points_x*num_points_t);
    let mut colors = Vec::with_capacity(num_points_x*num_points_t);

    // let mut zs0 = Vec::with_capacity(num_points_x*num_points_t);
    // let mut colors0 = Vec::with_capacity(num_points_x*num_points_t);

    let mut ts = Vec::with_capacity(num_points_t);
    let mut probs = Vec::with_capacity(num_points_t);

    sample(num_points_t, t_initial, t_final, |t| {
        let mut prob = ZERO;

        sample(num_points_x, x_initial, x_final, |x| {
            let computed = psi_computed(x, t);
            let t_mid = (t - t_initial) / (t_final - t_initial);
            let show_black = t_mid < 0.01 || t_mid > 0.99 || (false && (x > x_initial + 2e-1 && psi_computed(x - 1e-1, t).magnitude() < computed.magnitude()) &&
                (x < x_final - 2e-1 && psi_computed(x + 1e-1, t).magnitude() < computed.magnitude()));
            // println!("    Ψ({t:.4}, {x:.4}) = {: ^20}", computed.to_string());
            xs.push(x);
            ys.push(t);
            zs.push(computed.magnitude().powi(2));
            let (r, g, b) = if show_black {
                (0, 0, 0)
            } else { computed.rgb_just_hue() };
            colors.push(PlotlyRgb::new(r, g, b));

            // let psi0_computed = psi_0(x, t);
            // zs0.push(psi0_computed.magnitude().powi(2));
            // let (r0, g0, b0) = psi0_computed.rgb_just_hue();
            // colors0.push(PlotlyRgb::new(r0, g0, b0));

            prob += computed.conj() * psi_0(x, 0.0);
        });

        ts.push(t);
        probs.push(((x_final - x_initial) * prob / num_points_x as f64).magnitude().powi(2));
    });

    // xs.append(&mut vec![0.0, 0.0]);
    // ys.append(&mut vec![0.0, 0.0]);
    // zs.append(&mut vec![0.4, 1.3]);

    let scatter = Scatter3D::new(xs.clone(), ys.clone(), zs)
        .name("Ψ(x,t)")
        .line(Line::new().width(0.0))
        .mode(Mode::Markers)
        .marker(Marker::new().size(1).color_array(colors));

    // let scatter_psi_0 = Scatter3D::new(xs, ys, zs0)
    //     .name("Ψ_0(x,t)")
    //     .line(Line::new().width(0.0))
    //     .mode(Mode::Markers)
    //     .marker(Marker::new().size(1).color_array(colors0));

    let mut wave_func_plot = Plot::new();
    wave_func_plot.add_trace(scatter);
    // wave_func_plot.add_trace(scatter_psi_0);
    wave_func_plot.set_configuration(Configuration::new().fill_frame(true).frame_margins(0.0));
    wave_func_plot.write_html(format!("output/{}.html", plot_name));

    let scatter = Scatter::new(ts, probs)
        .name("Probability")
        .line(Line::new().width(0.0))
        .mode(Mode::Markers);

    let mut prob_plot = Plot::new();
    prob_plot.add_trace(scatter);
    prob_plot.set_configuration(Configuration::new().fill_frame(true));
    prob_plot.write_html(format!("output/{}_prob.html", plot_name));

}*/

#[test]
fn test_plotter() {
    plot_result(
        0.0, 2e-1, -1e-3, 1e-3,
        &|_x: f64, t: f64| Complex::from(if t > 0.1 { 1.0 } else { -1.0 }),
        |_x: f64, _t: f64| Complex::from(1.0),
        "test_plotter"
    );
}

pub(crate) fn plot_result<T: Into<c64>>(
    t_initial: f64,
    t_final: f64,
    x_initial: f64,
    x_final: f64,
    psi_computed: &impl Fn(f64, f64) -> T,
    psi_0: impl Fn(f64, f64) -> T,
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

    let scatter = Scatter::new(ts, probs)
        .name("<Ψ*|Ψ_0>")
        .line(Line::new().width(1.0))
        .mode(Mode::Markers);

    let mut prob_plot = Plot::new();
    prob_plot.add_trace(scatter);
    prob_plot.add_trace(mag_scatter);
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

/*
#[test]
fn schrodinger_eq_by_discretization_v1() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 6.0);
    let (x_initial, x_final) = (-2e1, 2e1);
    let (n_t, n_x) = (30, 51);
    let nq_t = 3;

    let (t_0, psi_0) = (
        0.0,
        // |_x: f64| Complex::from(1.0)
        gaussian_pulse(0.0, 3.0)
    );

    // Define the parameters for the differential equation
    let ang_frequency = 100.0;
    let phase_shift = 0.0;
    let electric_0 = 1.0;
    let potential =
        |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
        // |x: f64| if 8.0 < x && x < 7.0 { 1000.0 } else { 0.0 };
        // |x: f64| 0.0;
    let (pulse_initial, pulse_final) = (0.0, 3.0);
    let electric_field = |t: f64| if pulse_initial <= t && t <= pulse_final {
        electric_0 * (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
    } else {
        0.0
    };

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points.iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + 1;
    let dims = dims_t * n_x;
    let l = |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t - 1 {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t - 1 && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    // (1) Set up the problem for each x
    for n in 0..n_x {
        for q in 0..nq_t {
            for (i, (&t_i, &w_i)) in ts[q].iter().zip(ws.iter()).enumerate() {
                let i_index = q * (n_t - 1) + i + n * dims_t;
                let alpha = -2.0 * beta + potential(xs[n]) -
                    xs[n] * electric_field(t_i) * (ang_frequency * t_i + phase_shift).sin();

                // The lapack functions assume column major order
                for j in 0..n_t {
                    let jt_index = q * (n_t - 1) + j;
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let m_n_i_j = if is_bridge {
                        2.0 * alpha * ONE
                    } else {
                        1.0 * if i == j { alpha } else { 0.0 } -
                            I * lagrange_deriv(&ts[q], j, t_i)
                    };

                    matrix.data[i_index + (jt_index + n * dims_t) * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }
                }
            }
        }
    }

    // (2) Set up the initial conditions
    for n in 0..n_x {
        vector.data[n * dims_t] = psi_0(xs[n]).into();

        for index in 0..dims {
            matrix.data[index * matrix.rows + n * dims_t] = if n == index / dims_t {
                l(index % dims_t, t_0).into()
            } else { 0.0.into() };
        }
    }

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            println!("Matrix fancy:");
            matrix.print_fancy();
            println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
            println!("Vector (b_i): {:?}", vector);
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    // println!("\nSystems of eqs:");
    // matrix.print_fancy_with(&result, &vector);
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index % dims_t, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index % dims_t, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index % dims_t, t)).into()
            }
        }).sum::<Complex>()
    };

    plot_result(
        t_initial, t_final, x_initial, x_final, &psi_computed,
        |x: f64, t: f64| (-I * electric_0 * t).exp() * psi_0(x),
        "schrodinger_eq_by_discretization"
    );
}

#[test]
fn schrodinger_eq_by_discretization_v2() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 2.0);
    let (x_initial, x_final) = (-2e2, 2e2);
    let (n_t, n_x) = (10, 4001);
    let nq_t = 1;

    // Define the parameters for the differential equation
    let ang_frequency = 0.148;
    let phase_shift = 0.0;
    let electric_0 = 0.1;
    let potential = |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
    let ground_state = ground_state(5*n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * electric_0 * t).exp() * ground_state(x);

    let (pulse_initial, pulse_final) = (0.0, 100.0);
    let electric_field = |t: f64| if pulse_initial <= t && t <= pulse_final {
        electric_0 * (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
    } else {
        0.0
    };
    let time_dep = |x: f64, t|
        x * electric_field(t) * (ang_frequency * t + phase_shift).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t;
    let dims = dims_t * n_x;
    let l = |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    println!("SETTING UP MATRIX");

    // (1) Set up the problem for each x
    for n in 0..n_x {
        println!("n: {n}");

        for q in 0..nq_t {
            for (i, &t_i) in ts[q].iter().enumerate() {
                if i == 0 && q == 0 { continue };

                let i_index = q * (n_t - 1) + i + n * dims_t - 1;
                let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                let is_bridge_i = (q != 0 && i == 0) ||
                    (q != nq_t - 1 && i == n_t - 1);

                let b_i = if is_bridge_i {
                    2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                } else {
                    time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                };

                vector.data[i_index] = b_i.into();

                // The lapack functions assume column major order
                for j in 0..n_t {
                    if j == 0 && q == 0 { continue };

                    let jt_index = q * (n_t - 1) + j - 1;
                    let j_index = jt_index + n * dims_t;
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let m_n_i_j = if is_bridge {
                        2.0 * -alpha * ONE
                    } else {
                        I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }

                    };

                    matrix.data[i_index + j_index * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * -beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * -beta * ONE;
                    }
                }
            }
        }
    }

    println!("SOLVING IT NOW");

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            println!("Matrix fancy:");
            matrix.print_fancy();
            println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
            println!("Vector (b_i): {:?}", vector);
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    println!("DONE SOLVING");

    // println!("\nSystems of eqs:");
    // matrix.print_fancy_with(&result, &vector);
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index+1, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, psi_0, "schrodinger_eq_by_discretization_v2");
}

#[test]
fn schrodinger_basic_iterative() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 2.0);
    let (x_initial, x_final) = (-2e2, 2e2);
    let (n_t, n_x) = (10, 4001);
    let nq_t = 1;

    // Define the parameters for the differential equation
    let ang_frequency = 0.148;
    let phase_shift = 0.0;
    let electric_0 = 0.1;
    let potential = |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
    let ground_state = ground_state(5*n_x, x_initial, x_final, potential).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * electric_0 * t).exp() * ground_state(x);

    let (pulse_initial, pulse_final) = (0.0, 100.0);
    let electric_field = |t: f64| if pulse_initial <= t && t <= pulse_final {
        electric_0 * (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
    } else {
        0.0
    };
    let time_dep = |x: f64, t|
    x * electric_field(t) * (ang_frequency * t + phase_shift).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + 1;
    let dims = dims_t * n_x;
    let l = |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    println!("SETTING UP MATRIX");

    // (1) Set up the problem for each x
    for n in 0..n_x {
        println!("n: {n}");

        for q in 0..nq_t {
            for (i, &t_i) in ts[q].iter().enumerate() {
                let i_index = q * (n_t - 1) + i + n * dims_t;
                let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                let is_bridge_i = (q != 0 && i == 0) ||
                    (q != nq_t - 1 && i == n_t - 1);

                let b_i = if is_bridge_i {
                    2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                } else {
                    time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                };

                vector.data[i_index] = b_i.into();

                // The lapack functions assume column major order
                for j in 0..n_t {
                    let jt_index = q * (n_t - 1) + j;
                    let j_index = jt_index + n * dims_t;
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let m_n_i_j = if is_bridge {
                        2.0 * -alpha * ONE
                    } else {
                        I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }

                    };

                    matrix.data[i_index + j_index * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * -beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * -beta * ONE;
                    }
                }
            }
        }
    }

    println!("SOLVING IT NOW");

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            println!("Matrix fancy:");
            matrix.print_fancy();
            println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
            println!("Vector (b_i): {:?}", vector);
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    println!("DONE SOLVING");

    // println!("\nSystems of eqs:");
    // matrix.print_fancy_with(&result, &vector);
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index+1, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    plot_result(t_initial, t_final, x_initial, x_final, psi_computed, psi_0, "schrodinger_basic_iterative");
}*/

#[derive(Clone)]
pub struct GetWaveFuncOptions {
    initial_time_settings: Option<(f64, Vec<Complex>)>,
    t_final: f64,
    n_t: usize,
    nq_t: usize,
    x_initial: f64,
    x_final: f64,
    n_x: usize,
    ang_frequency: f64,
    phase_shift: f64,
    electric_0: f64,
    potential: fn(f64) -> f64,
    pulse_initial: f64,
    pulse_final: f64,
    smooth_wave: bool,
    debug: bool,
    plot_name: Option<String>,
    show_matrix: bool
}

impl GetWaveFuncOptions {
    fn new(
        t_final: f64,
        n_t: usize,
        nq_t: usize,
        (x_initial, x_final): (f64, f64),
        n_x: usize,
    ) -> Self {
        GetWaveFuncOptions {
            t_final, n_t, nq_t, x_initial, x_final, n_x,
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
}

impl Default for GetWaveFuncOptions {
    fn default() -> Self {
        GetWaveFuncOptions {
            initial_time_settings: None,
            t_final: 1200.0,
            n_t: 1200,
            nq_t: 1,
            x_initial: -2e2,
            x_final: 2e2,
            n_x: 4001,
            ang_frequency: 0.148,
            phase_shift: 0.0,
            electric_0: 0.1,
            potential: |x| -1.0 / (1.0 + x.powi(2)).sqrt(),
            pulse_initial: 0.0,
            pulse_final: 1200.0,
            smooth_wave: true,
            debug: false,
            plot_name: None,
            show_matrix: false
        }
    }
}


pub fn compute_wave_function_simpler(options: GetWaveFuncOptions) -> impl Fn(f64, f64) -> Complex {
    let GetWaveFuncOptions {
        initial_time_settings, t_final, n_t, nq_t, x_initial, x_final, n_x,
        ang_frequency, phase_shift, electric_0, potential,
        pulse_initial, pulse_final, smooth_wave, debug, show_matrix, ..
    } = options;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    let psi_initial = initial_time_settings.as_ref()
        .map(|(_, psi)| psi.clone())
        .unwrap_or(vec![ZERO; n_x]);

    let (ground_energy, ground_state) = ground_state(n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = move |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);
    println!("Ground energy: {ground_energy} -- electric_0: {electric_0}");

    println!("in compute -- psi_0({:.8}, {:.8}) = {}", 0.0, t_final, psi_0(0.0, t_final));

    let electric_field = |t| if pulse_initial <= t && t <= pulse_final {
        electric_0 * if smooth_wave {
            ((PI * (t - pulse_initial) / (pulse_final - pulse_initial)) as f64).sin().powi(2)
        } else { 1.0 }
    } else { 0.0 };
    let time_dep = |x: f64, t| x * electric_field(t) * ((ang_frequency * t + phase_shift) as f64).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + 1;
    let dims = dims_t * n_x;
    let get_indices = move |index: usize| {
        if index == 0 {
            (0, 0)
        } else if index == dims_t - 1 {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        }
    };
    let l = |index: usize, t| {
        // Gets the interval and the index within the interval
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    if debug { println!("INITIALIZING THE MATRIX"); }

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    if debug { println!("SETTING UP MATRIX"); }

    for n in 0..n_x {
        for q in 0..nq_t {
            for (i, &t_i) in ts[q].iter().enumerate() {
                let it_index = q * (n_t - 1) + i;
                let i_index = it_index + n * dims_t;
                let alpha = 1.0 / delta_x.powi(2) + potential(xs[n]) + time_dep(xs[n], t_i);
                let is_bridge_i = (q != 0 && i == 0) ||
                    (q != nq_t - 1 && i == n_t - 1);

                let b_i = if is_bridge_i {
                    2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                } else {
                    time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                };

                vector.data[i_index] = b_i.into();

                // The lapack functions assume column major order
                for j in 0..n_t {
                    let jt_index = q * (n_t - 1) + j;
                    let j_index = jt_index + n * dims_t;
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let m_n_i_j = if is_bridge {
                        2.0 * -alpha * ONE
                    } else {
                        I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                    };

                    matrix.data[i_index + j_index * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1)*dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1)*dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }
                }
            }
        }
    }

    for n in 0..n_x {
        // if debug { println!("n={n} -- psi_initial[n]: {}", psi_initial[n]); }
        vector.data[n * dims_t] = (psi_initial[n]).into();

        for index in 0..dims {
            matrix.data[index * matrix.rows + n * dims_t] = if n == index / dims_t {
                (l(index % dims_t, t_initial)).into()
            } else { 0.0.into() };
        }
    }

    if debug { println!("SOLVING IT NOW"); }

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            if show_matrix {
                println!("\nSystems of eqs:");
                matrix.print_fancy();
                println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
                println!("Vector (b_i): {:?}", vector);
            }
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    if debug { println!("DONE SOLVING"); }

    if show_matrix {
        println!("\nSystems of eqs:");
        matrix.print_fancy_with(&result, &vector);
    }

    let l = move |index: usize, t| {
        // Gets the interval and the index within the interval
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index, t)).into()
            }
        }).sum::<Complex>()
    };

    if debug && options.plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = options.plot_name {
        plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, &psi_0, &plot_name);
    }

    move |x: f64, t: f64| {
        if !(x_initial <= x && x <= x_final) { panic!("x: {x} is out of bounds"); }
        if !(t_initial <= t && t <= t_final) { panic!("t: {t} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index, t)).into()
            }
        }).sum::<Complex>()
    }
}

#[test]
fn test_compute_wave() {
    let mut options = GetWaveFuncOptions::new(
        1200.0,
        80,
        1,
        (-2e2, 2e2),
        101
    ).with_debug().with_plot("test_compute_wave".to_string());

    options.pulse_final = 1200.0;

    let _ = compute_wave_function_simpler(options);
}

#[test]
fn show_matrix_repeated__compute_wave() {
    let mut options = GetWaveFuncOptions::new(
        10.0,
        10,
        1,
        (-5e1, 5e1),
        9
    ).with_debug().with_matrix();

    let time_final = 200.0;
    let delta_time = 10.0;
    let delta_x = (options.x_final - options.x_initial) / (options.n_x as f64 - 1.0);
    let num_intervals = ((time_final / delta_time) as f64).ceil() as usize;
    let mut wave_func_vector = vec![ZERO.into(); options.n_x];
    let mut wave_functions: Vec<Box<dyn Fn(f64, f64) -> Complex>> = Vec::new();

    println!("Getting the ground state");
    let (ground_energy, ground_state) = ground_state(
        5*options.n_x, options.x_initial, options.x_final, &options.potential
    ).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);

    options.pulse_final = 1200.0;

    for i in 0..num_intervals {
        let time_start = i as f64 * delta_time;
        let time_end = (i + 1) as f64 * delta_time;

        if options.debug {
            println!("Interval: {}/{num_intervals}", i);
            println!("Time: {time_start} -> {time_end}");
        }

        options.t_final = time_end;

        let wave_function = compute_wave_function_simpler(options.clone());

        for n in 0..options.n_x {
            let x_n = n as f64 * delta_x + options.x_initial;
            wave_func_vector[n] = wave_function(x_n, time_end) - psi_0(x_n, time_end);
        }

        println!();

        options.initial_time_settings = Some((time_end, wave_func_vector.clone()));

        wave_functions.push(Box::new(wave_function));
    }

    let wave_function = |x, t| {
        let interval = ((t / delta_time) as f64).floor() as usize;
        wave_functions[interval](x, t)
    };

    println!("Plotting the result");

    plot_result(
        0.0,
        time_final,
        options.x_initial,
        options.x_final,
        &wave_function,
        &psi_0,
        "repeated_compute_wave"
    );
}

#[test]
fn repeated_compute_wave() {
    let mut options = GetWaveFuncOptions::new(
        10.0,
        20,
        1,
        (-5e1, 5e1),
        501
    ).with_debug();//.with_matrix();

    options.smooth_wave = false;

    let time_final = 100.0;
    let delta_time = 10.0;
    let num_intervals = ((time_final / delta_time) as f64).ceil() as usize;
    let mut wave_func_vector = vec![ZERO.into(); options.n_x];
    let mut wave_functions: Vec<Box<dyn Fn(f64, f64) -> Complex>> = Vec::new();

    // options.electric_0 = 0.0;

    println!("Getting the ground state");
    let (ground_energy, ground_state) = ground_state(
        options.n_x, options.x_initial, options.x_final, &options.potential
    ).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);

    options.pulse_final = 1200.0;

    let x = 0.0;

    for i in 0..num_intervals {
        let time_start = i as f64 * delta_time;
        let time_end = (i + 1) as f64 * delta_time;

        if options.debug {
            println!("Interval: {}/{num_intervals}", i+1);
            println!("Time: {time_start} -> {time_end}");
            println!("psi_0({x:.8}, {:.8}) = {}", time_end, psi_0(x, time_end))
        }

        options.t_final = time_end;

        let wave_function = compute_wave_function_simpler(options.clone());

        for n in 0..options.n_x {
            let x_n = n as f64 * (options.x_final - options.x_initial) / (options.n_x as f64 - 1.0) + options.x_initial;
            wave_func_vector[n] = wave_function(x_n, time_end) - psi_0(x_n, time_end);
        }

        println!();

        options.initial_time_settings = Some((time_end, wave_func_vector.clone()));

        wave_functions.push(Box::new(wave_function));
    }

    let wave_function = |x, t| {
        let interval = ((t / delta_time) as f64).floor() as usize;
        wave_functions[interval](x, t)
    };

    println!("Plotting the result");

    sample(10, -1e-5, 1e-5, &mut |t| {
        println!("Ψ({x:.8}, {:.8}) = {}", delta_time + t, wave_function(x, delta_time + t));
    });

    plot_result(
        0.0,
        time_final,
        options.x_initial,
        options.x_final,
        &wave_function,
        &psi_0,
        "repeated_compute_wave"
    );
}

#[derive(Clone)]
pub struct IterativeSolveOptions {
    max_iteration: usize,
    tolerance: f64,
    num_groups: usize,
    center_block_size: Option<usize>
}

#[test]
fn test_iterative_schrodinger_solve_with_center() {
    let mut options = GetWaveFuncOptions::new(
        10.0,
        400,
        1,
        (-5e1, 5e1),
        1001
    ).with_debug().with_plot("test_iterative_schrodinger_solve".to_string());

    options.smooth_wave = false;
    options.pulse_final = 1200.0;

    let iter_options = IterativeSolveOptions {
        max_iteration: 1000,
        tolerance: 1e-8,
        num_groups: 9,
        center_block_size: None
    };

    let _ = compute_iterative_solve(options, iter_options);
}

// Not currently working
#[test]
fn iterative_repeat() {
    let mut options = GetWaveFuncOptions::new(
        10.0,
        100,
        1,
        (-2e1, 2e1),
        4001
    ).with_debug();//.with_matrix();

    options.smooth_wave = false;

    let iterative_options = IterativeSolveOptions {
        max_iteration: 100,
        tolerance: 1e-8,
        num_groups: 9,
        center_block_size: None,
    };

    let time_final = 100.0;
    let delta_time = 10.0;
    let num_intervals = ((time_final / delta_time) as f64).ceil() as usize;
    let mut wave_func_vector = vec![ZERO.into(); options.n_x];
    let mut wave_functions: Vec<Box<dyn Fn(f64, f64) -> Complex>> = Vec::new();

    options.pulse_final = 1200.0;

    for i in 0..num_intervals {
        let time_end = (i + 1) as f64 * delta_time;

        options.t_final = time_end;

        let (_, wave_function) = compute_iterative_solve(options.clone(), iterative_options.clone());

        for n in 0..options.n_x {
            let x_n = n as f64 * (options.x_final - options.x_initial) / (options.n_x as f64 - 1.0) + options.x_initial;
            wave_func_vector[n] = wave_function(x_n, time_end);
        }

        println!();

        options.initial_time_settings = Some((time_end, wave_func_vector.clone()));

        wave_functions.push(Box::new(wave_function));
    }

    let wave_function = |x, t| {
        let interval = ((t / delta_time) as f64).floor() as usize;
        wave_functions[interval](x, t)
    };

    println!("Getting the ground state");
    let (ground_energy, ground_state) = ground_state(
        options.n_x, options.x_initial, options.x_final, &options.potential
    ).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);

    println!("Plotting the result");

    plot_result(
        0.0,
        time_final,
        options.x_initial,
        options.x_final,
        &wave_function,
        &psi_0,
        "iterative_repeated_compute_wave"
    );
}

#[test]
fn get_convergence_curve() {
    let n_xs = [
        51, 101, 201, 401, 801, 1601, 3201, 6401
    ];

    let n_ts = [
        5, 10, 20, 40, 80, 160, 320, 640
    ];

    let max_iteration = 100;

    for n_x in n_xs.iter() {
        for n_t in n_ts.iter() {
            let options = GetWaveFuncOptions::new(
                100.0,
                *n_t,
                1,
                (-2e2, 2e2),
                *n_x
            );

            let iter_options = IterativeSolveOptions {
                max_iteration,
                tolerance: 1e-8,
                num_groups: 9,
                center_block_size: None
            };

            let (k, _) = compute_iterative_solve(options, iter_options);

            if k < max_iteration - 1 {
                println!("Convergence: n_x: {n_x}, n_t: {n_t}, k: {k}");
            } else {
                println!("No convergence for n_x: {n_x}, n_t: {n_t}")
            }
        }
    }
}

fn compute_iterative_solve(
    options: GetWaveFuncOptions,
    iter_options: IterativeSolveOptions
) -> (usize, impl Fn(f64, f64) -> Complex) {
    let GetWaveFuncOptions {
        initial_time_settings, t_final, n_t, nq_t, x_initial, x_final, n_x,
        ang_frequency, phase_shift, electric_0, potential,
        pulse_initial, pulse_final, smooth_wave, debug, ..
    } = options;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    let is_simple_initial = initial_time_settings.is_none();
    let init_offset = if initial_time_settings.is_none() { 0 } else { 1 };

    if debug { println!("Computing the ground state"); }
    let (ground_energy, ground_state) = ground_state(n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = move |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);

    let electric_field = |t| if pulse_initial <= t && t <= pulse_final {
        electric_0 * if smooth_wave {
            ((PI * (t - pulse_initial) / (pulse_final - pulse_initial)) as f64).sin().powi(2)
        } else { 1.0 }
    } else { 0.0 };
    let time_dep = |x: f64, t| x * electric_field(t) * ((ang_frequency * t + phase_shift) as f64).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + init_offset;
    let dims = dims_t * n_x;
    let get_indices = |index: usize| {
        if index == 0 {
            (0, 0)
        } else if index == dims_t - init_offset {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        }
    };
    let l = |index: usize, t| {
        // Gets the interval and the index within the interval
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    if debug { println!("Setting up the matrix"); }

    let IterativeSolveOptions {
        max_iteration, tolerance, num_groups, center_block_size
    } = iter_options;

    let chunk_size = n_x.div_ceil(num_groups);
    let mut solution = (0..dims).map(|index| {
        (psi_0(xs[index / dims_t], 0.0) * 1.0).into()
    }).collect::<Vec<Complex>>();
    let mut prev_solution = vec![ZERO; dims];
    let mut matrix_inverses: Vec<ComplexMatrix> = vec![ComplexMatrix::default(); n_x];
    let mut prev_max_err = 0.0;
    let mut max_k = max_iteration;
    // It is assumed that num_groups is odd
    let center_block_range = center_block_size.map(|size| (
        n_x / 2 - size.min(chunk_size) / 2..n_x / 2 + size.min(chunk_size) / 2
    ));

    matrix_inverses.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, slice)| {
            let thread_id = format!("{}", thread_id::get() % 1000).magenta();

            if debug {
                println!(
                    "[{thread_id}]: Computing inverses for chunk #{chunk_index}: {}-{}",
                    chunk_index * chunk_size,
                    chunk_index * chunk_size + slice.len()
                );
                let _ = std::io::stdout().flush();
            }

            for chunk_n in 0..slice.len() {
                let n = chunk_index * chunk_size + chunk_n;
                let block_range = if let Some(range) = center_block_range.clone() {
                    if range.start == n { range } else if range.contains(&n) { 0..0 } else { n..n+1 }
                } else { n..n+1 };
                let matrix_size = block_range.len() * dims_t;
                let mut matrix = ComplexMatrix {
                    data: vec![ZERO.into(); matrix_size.pow(2)],
                    rows: matrix_size,
                    cols: matrix_size
                };

                if block_range.len() == 0 { continue; }

                for n in block_range.clone() {
                    for q in 0..nq_t {
                        for (i, &t_i) in ts[q].iter().enumerate() {
                            if i == 0 && q == 0 && is_simple_initial { continue };

                            let it_index = q * (n_t - 1) + i + init_offset - 1;
                            let i_index = it_index + (n - block_range.start) * dims_t;
                            let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);

                            // The lapack functions assume column major order
                            for j in 0..n_t {
                                if j == 0 && q == 0 && is_simple_initial { continue };

                                let jt_index = q * (n_t - 1) + j + init_offset - 1;
                                let j_index = jt_index + (n - block_range.start) * dims_t;
                                let is_bridge = (q != 0 && i == 0 && j == 0) ||
                                    (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                                let m_n_i_j = if is_bridge {
                                    2.0 * -alpha * ONE
                                } else {
                                    I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                                };

                                matrix.data[i_index + j_index * matrix.rows] = m_n_i_j.into();

                                if n > block_range.start && i == j {
                                    matrix.data[i_index + (j_index - dims_t) * matrix.rows] +=
                                        if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                                }

                                if n < block_range.end-1 && i == j {
                                    matrix.data[i_index + (j_index + dims_t) * matrix.rows] +=
                                        if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                                }
                            }
                        }
                    }
                }

                if let Some((t_0, _)) = initial_time_settings {
                    for n in block_range.clone() {
                        for index in 0..matrix_size {
                            matrix.data[index * matrix.rows + (n - block_range.start) * dims_t] = if n == index / dims_t {
                                (l(index % dims_t, t_0)).into()
                            } else { 0.0.into() };
                        }
                    }
                }

                // if block_range.len() > 1 && debug {
                //     std::thread::sleep(Duration::from_secs(1));
                //     println!("[{thread_id}]: Matrix @ {}:", n);
                //     matrix.print_fancy();
                // }

                let inverse =  matrix.inverse().unwrap();

                // if block_range.len() > 1 && debug {
                //     println!("And inverse: ");
                //     inverse.print_fancy();
                // }

                slice[chunk_n] = inverse;
            }

            if debug {
                println!("[{thread_id}]: Finished with inverses for chunk #{chunk_index}");
                let _ = std::io::stdout().flush();
            }
        });

    for k in 0..max_iteration {
        (solution, prev_solution) = (prev_solution, solution);

        let max_err: f64 = solution.par_chunks_mut(dims_t * chunk_size)
            .enumerate()
            .map(|(chunk_index, slice)| {
                let thread_id = format!("{}", thread_id::get() % 1000).yellow();

                let mut max_err: f64 = 0.0;
                if debug {
                    println!(
                        "[{thread_id}]: Starting chunk #{chunk_index}: {}-{}",
                        chunk_index * chunk_size,
                        chunk_index * chunk_size + slice.len() / dims_t
                    );
                    let _ = std::io::stdout().flush();
                }

                for chunk_n in 0..(slice.len() / dims_t) {
                    let n = chunk_index * chunk_size + chunk_n;
                    let block_range = if let Some(range) = center_block_range.clone() {
                        if range.start == n { range } else if range.contains(&n) { 0..0 } else { n..n+1 }
                    } else { n..n+1 };
                    let vector_size = block_range.len() * dims_t;

                    if block_range.len() == 0 { continue; }

                    let mut vector = ComplexVector {
                        data: vec![ZERO.into(); vector_size]
                    };

                    for n in block_range.clone() {
                        for q in 0..nq_t {
                            for (i, &t_i) in ts[q].iter().enumerate() {
                                if i == 0 && q == 0 && is_simple_initial { continue };

                                let it_index = q * (n_t - 1) + i + init_offset - 1;
                                let i_index = it_index + (n - block_range.start) * dims_t;
                                let is_bridge_i = (q != 0 && i == 0) ||
                                    (q != nq_t - 1 && i == n_t - 1);

                                let mut b_i = if is_bridge_i {
                                    2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                                } else {
                                    time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                                };

                                if n > 0 && n == block_range.start {
                                    b_i -= prev_solution[it_index + (n-1) * dims_t] *
                                        if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                                }

                                if n < n_x - 1 && n == block_range.end - 1 {
                                    b_i -= prev_solution[it_index + (n+1) * dims_t] *
                                        if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                                }

                                vector.data[i_index] = b_i.into();
                            }
                        }
                    }

                    if let Some((_, wave_func)) = &initial_time_settings {
                        vector.data[0] = wave_func[n].into();
                    }

                    let result = matrix_inverses[n].multiply_vector(&vector);

                    // if block_range.len() > 1 && debug {
                    //     std::thread::sleep(Duration::from_secs(1));
                    //     println!("\nMatrix fancy at {}:", n);
                    //     matrix_inverses[n].print_fancy_with(&vector, &result);
                    // }

                    for (i, c_i) in result.data.into_iter().map(Complex::from).enumerate() {
                        let index = n * dims_t + i;
                        max_err = max_err.max((c_i - prev_solution[index]).magnitude());
                        slice[i + chunk_n * dims_t] = c_i;
                    }
                }

                if debug {
                    let percent_change = if prev_max_err == 0.0 {
                        0.0
                    } else {
                        100.0 * (max_err - prev_max_err) / prev_max_err
                    };
                    let mut diff = format!("{}{:.4}%", if percent_change > 0.0 { "+" } else { "" }, percent_change).normal();

                    diff = if percent_change < 0.0 {
                        diff.green()
                    } else if percent_change > 0.0 {
                        diff.red()
                    } else {
                        diff.blue()
                    };
                    println!("[{thread_id}]: Finished chunk #{chunk_index} with k = {k: >3}: {:.4e}, {diff}", max_err);
                }

                max_err
            }).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        if debug {
            let percent_change = if prev_max_err == 0.0 {
                0.0
            } else {
                100.0 * (max_err - prev_max_err) / prev_max_err
            };
            let mut diff = format!("{}{:.4}%", if percent_change > 0.0 { "+" } else { "" }, percent_change).bold();

            diff = if percent_change < 0.0 {
                diff.green()
            } else if percent_change > 0.0 {
                diff.red()
            } else {
                diff.blue()
            };

            println!(
                "MAX ERROR ({k: >3}): {:.4e}, {diff}\n\n",
                max_err
            );

            prev_max_err = max_err;
        }

        if max_err.is_infinite() {
            if debug { println!("INFINITY!"); }
            max_k = usize::MAX;
            break;
        }

        if max_err < tolerance {
            if debug { println!("BROKEN!"); }
            max_k = k;
            break;
        }
    }

    let l = move |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(solution[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(solution[index + n2 * dims_t] * l(index+1, t))
            } else {
                (solution[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    if debug && options.plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = options.plot_name {
        plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, &psi_0, &plot_name);
    }

    (max_k, move |x: f64, t: f64| {
        if !(t_initial <= t && t <= t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x <= x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(solution[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(solution[index + n2 * dims_t] * l(index+1, t))
            } else {
                (solution[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    })
}



pub fn ground_state(
    n_x: usize, x_initial: f64, x_final: f64,
    potential: &impl Fn(f64) -> f64
) -> Result<(f64, impl Fn(f64) -> Complex + Clone + 'static), LapackError> {
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


// OLD:
/*



pub fn compute_wave_function(options: GetWaveFuncOptions) -> impl Fn(f64, f64) -> Complex {
    let GetWaveFuncOptions {
        initial_time_settings, t_final, n_t, nq_t, x_initial, x_final, n_x,
        ang_frequency, phase_shift, electric_0, potential,
        pulse_initial, pulse_final, smooth_wave, debug, show_matrix, ..
    } = options;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    let is_simple_initial = initial_time_settings.is_none();
    let init_offset = if initial_time_settings.is_none() { 0 } else { 1 };

    let (ground_energy, ground_state) = ground_state(n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = move |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);
    println!("Ground energy: {ground_energy} -- electric_0: {electric_0}");

    let electric_field = |t| if pulse_initial <= t && t <= pulse_final {
        electric_0 * if smooth_wave {
            ((PI * (t - pulse_initial) / (pulse_final - pulse_initial)) as f64).sin().powi(2)
        } else { 1.0 }
    } else { 0.0 };
    let time_dep = |x: f64, t| x * electric_field(t) * ((ang_frequency * t + phase_shift) as f64).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + init_offset;
    let dims = dims_t * n_x;
    let get_indices = |index: usize| {
        if index == 0 {
            (0, 0)
        } else if index == dims_t - init_offset {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        }
    };
    let l = |index: usize, t| {
        // Gets the interval and the index within the interval
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    if debug { println!("INITIALIZING THE MATRIX"); }

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    if debug { println!("SETTING UP MATRIX"); }

    for n in 0..n_x {
        for q in 0..nq_t {
            for (i, &t_i) in ts[q].iter().enumerate() {
                if i == 0 && q == 0 && is_simple_initial { continue };

                let it_index = q * (n_t - 1) + i + init_offset - 1;
                let i_index = it_index + n * dims_t;
                let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                let is_bridge_i = (q != 0 && i == 0) ||
                    (q != nq_t - 1 && i == n_t - 1);

                let b_i = if is_bridge_i {
                    2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                } else {
                    time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                };

                vector.data[i_index] = b_i.into();

                // The lapack functions assume column major order
                for j in 0..n_t {
                    if j == 0 && q == 0 && is_simple_initial { continue };

                    let jt_index = q * (n_t - 1) + j + init_offset - 1;
                    let j_index = jt_index + n * dims_t;
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let m_n_i_j = if is_bridge {
                        2.0 * -alpha * ONE
                    } else {
                        I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                    };

                    matrix.data[i_index + j_index * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1)*dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1)*dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * beta * ONE;
                    }
                }
            }
        }
    }

    if let Some((t_0, wave_func)) = initial_time_settings {
        for n in 0..n_x {
            vector.data[n * dims_t] = (wave_func[n] - psi_0(xs[n], t_0)).into();

            for index in 0..dims {
                let (_, i) = get_indices(index % dims_t);
                matrix.data[index * matrix.rows + n * dims_t] = if n == index / dims_t {
                    (l(index % dims_t, t_0)).into()
                } else { 0.0.into() };
            }
        }
    }

    if debug { println!("SOLVING IT NOW"); }

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            if show_matrix {
                println!("\nSystems of eqs:");
                matrix.print_fancy();
                println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
                println!("Vector (b_i): {:?}", vector);
            }
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    if debug { println!("DONE SOLVING"); }

    if show_matrix {
        println!("\nSystems of eqs:");
        matrix.print_fancy_with(&result, &vector);
    }

    let l = move |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index+1, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    if debug && options.plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = options.plot_name {
        plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, &psi_0, &plot_name);
    }

    move |x: f64, t: f64| {
        if !(x_initial <= x && x <= x_final) { panic!("x: {x} is out of bounds"); }
        if !(t_initial <= t && t <= t_final) { panic!("t: {t} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result.data[index + n1 * dims_t] * l(index+1-init_offset, t)) +
                    x_mid * Complex::from(result.data[index + n2 * dims_t] * l(index+1-init_offset, t))
            } else {
                (result.data[index + n1 * dims_t] * l(index+1-init_offset, t)).into()
            }
        }).sum::<Complex>()
    }
}

#[test]
fn schrodinger_eq_solving_by_sep_of_vars() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 2e-1);
    let (x_initial, x_final) = (-1e-3, 1e-3);
    let (n_t, n_x) = (10, 10);
    let (nq_t, nq_x) = (1, 1);
    let (t_0, x_0, psi_0) = (0.0, 0.0, Complex::from(1.0));

    // Define the parameters for the differential equation
    let omega = 10.0;
    let electric_0 = 1.0;
    let potential = |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
    let electric_field = |t: f64| if t_initial <= t && t <= t_final {
        electric_0 * (PI * t / (t_final - t_initial)).sin().powi(2)
    } else {
        0.0
    };

    let delta_t = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (nq_x as f64);
    let quad_points_t = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_t);
    let quad_points_x = gauss_lobatto_quadrature(n_x, x_initial, x_initial + delta_x);
    // let ws_t: Vec<f64> = quad_points_t.iter().map(|&(_, w)| w).collect();
    // let ws_x: Vec<f64> = quad_points_x.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points_t.iter().map(|&(t, _)| t + q as f64 * delta_t).collect())
        .collect();
    let xs: Vec<Vec<f64>> = (0..nq_x)
        .map(|q| quad_points_x.iter().map(|&(x, _)| x + q as f64 * delta_x).collect())
        .collect();

    let dims_t = n_t * nq_t - nq_t + 1;
    let dims_x = n_x * nq_x - nq_x + 1;
    let dims = dims_t * dims_x;
    let get_indices = |index_u, dims_u, nq_u, n_u| {
        // Gets the interval and the index within the interval
        if index_u == 0 {
            (0, 0)
        } else if index_u == dims_u-1 {
            (nq_u -1, n_u -1)
        } else {
            ((index_u - 1) / (n_u -1), (index_u - 1) % (n_u -1) + 1)
        }
    };
    let l_u = |index_u, dims_u, nq_u, n_u, us: &Vec<Vec<f64>>, u| {
        let (q, i) = get_indices(index_u, dims_u, nq_u, n_u);

        if index_u != 0 && i == 0 && u < us[q][i] {
            crate::lagrange(&us[q-1], n_u -1, u) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index_u != dims_u-1 && i == n_u -1 && us[q][i] < u {
            crate::lagrange(&us[q+1], 0, u) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            crate::lagrange(&us[q], i, u) // Use the lagrange in the current interval
        }
    };
    let l = |index, t, x| {
        let index_t = index % dims_t;
        let index_x = index / dims_t;

        l_u(index_t, dims_t, nq_t, n_t, &ts, t) * l_u(index_x, dims_x, nq_x, n_x, &xs, x)
    };

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); dims.pow(2)],
        rows: dims,
        cols: dims
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); dims]
    };

    // (1) Set up the problem
    for index_ti in 0..dims_t {
        let (q_t, i_t) = get_indices(index_ti, dims_t, nq_t, n_t);

        for index_xi in 0..dims_x {
            let (q_x, i_x) = get_indices(index_xi, dims_x, nq_x, n_x);
            let index_i = index_ti + index_xi * dims_t;

            // The lapack functions assume column major order
            for j_t in 0..n_t {
                let index_tj = q_t * (n_t - 1) + j_t;
                let t_j = ts[q_t][j_t];

                for j_x in 0..n_x {
                    let index_xj = q_x * (n_x - 1) + j_x;
                    let x_j = xs[q_x][j_x];

                    let index_j = index_tj + index_xj * dims_t;
                    let mut m_i_j: Complex = 0.0.into();

                    // Square blocks along the diagonal
                    if i_x == j_x {
                        m_i_j += I * crate::lagrange_deriv(&ts[q_t], i_t, t_j);
                    }

                    // diagonal stripes all across
                    if i_t == j_t {
                        m_i_j += 0.5 * crate::lagrange_deriv_deriv(&xs[q_x], i_x, x_j) * ONE;
                    }

                    // Main diagonal
                    if i_x == j_x && i_t == j_t {
                        m_i_j += electric_field(t_j) * (omega * t_j).sin() - potential(x_j) * ONE;
                    }

                    matrix.data[index_i + index_j * dims] = m_i_j.into();
                }
            }
        }
    }

    // (2) Set up the initial conditions
    vector.data[0] = psi_0.into();

    for index_t in 0..dims_t {
        for index_x in 0..dims_x {
            let index = index_t + index_x * dims_t;

            matrix.data[index * matrix.rows] = l(index, t_0, x_0).into();
        }
    }

    // (3) Solve the systems of equations
    let result = match matrix.solve_systems(vector.clone()) {
        Ok(result) => result,
        Err(err) => {
            println!("Matrix fancy:");
            matrix.print_fancy();
            println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
            println!("Vector (b_i): {:?}", vector);
            panic!("Error while trying to solve the systems of equations: {:?}\n", err);
        }
    };

    // println!("\nSystems of eqs:");
    // matrix.print_fancy_with(&result, &vector);
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }

        (0..dims_t).map(|index| {
            (result.data[index] * l(index, t, x)).into()
        }).sum::<Complex>()
    };

    plot_result(
        t_initial, t_final, x_initial, x_final, &psi_computed,
        |_x: f64, _t: f64| psi_0,
        "schrodinger_eq_solving_by_sep_of_vars"
    );
}

fn gaussian_pulse(k0: f64, sigma: f64) -> impl Fn(f64) -> Complex {
    move |x: f64| (-(x / sigma).powi(2) / 4.0 + I * k0 * x).exp() / (2.0 * PI * sigma.powi(2)).powf(0.25)
}


#[test]
fn iterative_schrodinger_solve() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 100.0);
    let (x_initial, x_final) = (-2e2, 2e2);
    let (n_t, n_x) = (80, 1001);
    let nq_t = 5;

    // Define the parameters for the differential equation
    let ang_frequency = 0.148;
    let phase_shift = 0.0;
    let electric_0 = 0.1;
    let potential = |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
    let (ground_energy, ground_state) = ground_state(n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);


    let (pulse_initial, pulse_final) = (0.0, 30.0);
    let electric_field = |t: f64| if pulse_initial <= t && t <= pulse_final {
        electric_0 * (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
    } else {
        0.0
    };
    let time_dep = |x: f64, t|
        x * electric_field(t) * (ang_frequency * t + phase_shift).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t;
    let dims = dims_t * n_x;
    let l = |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };


    let mut solution = vec![ZERO; dims];
    let mut prev_solution = vec![ZERO; dims];

    let k_max = 100;
    let tolerance = 1e-8;

    for k in 0..k_max {
        (solution, prev_solution) = (prev_solution, solution);

        let max_err: f64 = solution.par_chunks_mut(dims_t).enumerate().map(|(n, slice)| {
            // println!("n: {n}");
            let mut matrix = ComplexMatrix {
                data: vec![ZERO.into(); dims_t.pow(2)],
                rows: dims_t,
                cols: dims_t
            };

            let mut vector = ComplexVector {
                data: vec![ZERO.into(); dims_t]
            };

            let mut max_err: f64 = 0.0;

            for q in 0..nq_t {
                for (i, &t_i) in ts[q].iter().enumerate() {
                    if i == 0 && q == 0 { continue };

                    let i_index = q * (n_t - 1) + i - 1;
                    let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                    let is_bridge_i = (q != 0 && i == 0) ||
                        (q != nq_t - 1 && i == n_t - 1);

                    let mut b_i = if is_bridge_i {
                        2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                    } else {
                        time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                    };

                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    if 0 < n {
                        b_i -= prev_solution[i_index + (n-1) * dims_t] *
                            if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                    }
                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    if n < n_x - 1 {
                        b_i -= prev_solution[i_index + (n+1) * dims_t] *
                            if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                    }
                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    vector.data[i_index] = b_i.into();

                    // The lapack functions assume column major order
                    for j in 0..n_t {
                        if j == 0 && q == 0 { continue };

                        let j_index = q * (n_t - 1) + j - 1;
                        let is_bridge = (q != 0 && i == 0 && j == 0) ||
                            (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                        let m_n_i_j = if is_bridge {
                            2.0 * -alpha * ONE
                        } else {
                            I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                        };
                        // if j == 1 && q == 0 && i == 4 && n == 5 { println!("m_n=5_i=4_j=1: {}", m_n_i_j) };

                        matrix.data[i_index + j_index * matrix.rows] = m_n_i_j.into();
                    }
                }
            }

            let result = match matrix.solve_systems(vector.clone()) {
                Ok(result) => result,
                Err(err) => {
                    println!("Matrix fancy at {}:", n);
                    matrix.print_fancy();
                    println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
                    println!("Vector (b_i): {:?}", vector);
                    panic!("Error while trying to solve the systems of equations: {:?}\n", err);
                }
            };

            if n == 5 {
                // println!("\nMatrix fancy at {}:", n);
                // matrix.print_fancy_with(&result, &vector);
            }

            for (i, c_i) in result.data.into_iter().map(Complex::from).enumerate() {
                let index = n * dims_t + i;
                max_err = max_err.max((c_i - prev_solution[index]).magnitude());
                slice[i] = c_i;

                // println!("sol[{} / {}] = {}", index, dims, c_i.to_string());
            }

            max_err
        }).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        /*for n in 0..n_x {
            for q in 0..nq_t {
                for (i, &t_i) in ts[q].iter().enumerate() {
                    if i == 0 && q == 0 { continue };

                    let i_index = q * (n_t - 1) + i - 1;
                    let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                    let is_bridge_i = (q != 0 && i == 0) ||
                        (q != nq_t - 1 && i == n_t - 1);

                    let mut b_i = if is_bridge_i {
                        2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                    } else {
                        time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                    };

                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    if 0 < n {
                        b_i -= prev_solution[i_index + (n-1) * dims_t] *
                            if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                    }
                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    if n < n_x - 1 {
                        b_i -= prev_solution[i_index + (n+1) * dims_t] *
                            if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                    }
                    // if i == 4 && n==5 && q==0 {println!("b_i: {}", b_i.to_string());}

                    vector.data[i_index] = b_i.into();

                    // The lapack functions assume column major order
                    for j in 0..n_t {
                        if j == 0 && q == 0 { continue };

                        let j_index = q * (n_t - 1) + j - 1;
                        let is_bridge = (q != 0 && i == 0 && j == 0) ||
                            (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                        let m_n_i_j = if is_bridge {
                            2.0 * -alpha * ONE
                        } else {
                            I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                        };
                        // if j == 1 && q == 0 && i == 4 && n == 5 { println!("m_n=5_i=4_j=1: {}", m_n_i_j) };

                        matrix.data[i_index + j_index * matrix.rows] = m_n_i_j.into();
                    }
                }
            }

            let result = match matrix.solve_systems(vector.clone()) {
                Ok(result) => result,
                Err(err) => {
                    println!("Matrix fancy at {}:", n);
                    matrix.print_fancy();
                    println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
                    println!("Vector (b_i): {:?}", vector);
                    panic!("Error while trying to solve the systems of equations: {:?}\n", err);
                }
            };

            if n == 5 {
                // println!("\nMatrix fancy at {}:", n);
                // matrix.print_fancy_with(&result, &vector);
            }

            for (i, c_i) in result.data.into_iter().map(Complex::from).enumerate() {
                let index = n * dims_t + i;
                max_err = max_err.max((c_i - prev_solution[index]).magnitude());
                solution[index] = c_i;

                // println!("sol[{} / {}] = {}", index, dims, c_i.to_string());
            }
        }*/

        println!("MAX ERROR ({k: >3}): {:.4e}", max_err);
        // println!("Result: [{}]\n{:?}", solution.iter().map(|c| c.block()).collect::<Vec<String>>().join(","), solution);

        if max_err < tolerance {
            println!("BROKEN!");
            break;
        }
    }

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(solution[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(solution[index + n2 * dims_t] * l(index+1, t))
            } else {
                (solution[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    println!("PLOTTING STARTED!!");

    plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, psi_0, "iterative_schrodinger_solve");
}

#[test]
fn test_iterative_schrodinger_solve_old() {
    let mut options = GetWaveFuncOptions::new(
        100.0,
        100,
        1,
        (-2e2, 2e2),
        501
    ).with_debug().with_plot("test_iterative_schrodinger_solve".to_string());

    options.smooth_wave = false;

    let iter_options = IterativeSolveOptions {
        max_iteration: 100,
        tolerance: 1e-8,
        num_groups: 9,
        center_block_size: None
    };

    let _ = compute_iterative_solve_old(options, iter_options);
}

fn compute_iterative_solve_old(
    options: GetWaveFuncOptions,
    iter_options: IterativeSolveOptions
) -> (usize, impl Fn(f64, f64) -> Complex) {
    let GetWaveFuncOptions {
        initial_time_settings, t_final, n_t, nq_t, x_initial, x_final, n_x,
        ang_frequency, phase_shift, electric_0, potential,
        pulse_initial, pulse_final, smooth_wave, debug, show_matrix, ..
    } = options;

    let t_initial = *initial_time_settings.as_ref().map(|(t, _)| t).unwrap_or(&0.0);
    let is_simple_initial = initial_time_settings.is_none();
    let init_offset = if initial_time_settings.is_none() { 0 } else { 1 };

    if debug { println!("Computing the ground state"); }
    let (ground_energy, ground_state) = ground_state(n_x, x_initial, x_final, &potential).unwrap();
    let psi_0 = move |x: f64, t: f64| (-I * ground_energy * t).exp() * ground_state(x);

    let electric_field = |t| if pulse_initial <= t && t <= pulse_final {
        electric_0 * if smooth_wave {
            ((PI * (t - pulse_initial) / (pulse_final - pulse_initial)) as f64).sin().powi(2)
        } else { 1.0 }
    } else { 0.0 };
    let time_dep = |x: f64, t| x * electric_field(t) * ((ang_frequency * t + phase_shift) as f64).sin();

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points[0..].iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + init_offset;
    let dims = dims_t * n_x;
    let get_indices = |index: usize| {
        if index == 0 {
            (0, 0)
        } else if index == dims_t - init_offset {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        }
    };
    let l = |index: usize, t| {
        // Gets the interval and the index within the interval
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    if debug { println!("Setting up the matrix"); }

    let IterativeSolveOptions {
        max_iteration, tolerance, num_groups, ..
    } = iter_options;

    let chunk_size = n_x.div_ceil(num_groups);
    let mut solution = (0..dims).map(|index| {
        (potential(xs[index / dims_t]) * 0.0).into()
    }).collect::<Vec<Complex>>();
    let mut prev_solution = vec![ZERO; dims];
    let mut matrix_inverses: Vec<ComplexMatrix> = vec![ComplexMatrix::default(); n_x];
    let mut prev_max_err = 0.0;
    let mut max_k = max_iteration;

    matrix_inverses.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_index, slice)| {
            let thread_id = format!("{}", thread_id::get() % 1000).magenta();

            if debug {
                println!(
                    "[{thread_id}]: Computing inverses for chunk #{chunk_index}: {}-{}",
                    chunk_index * chunk_size,
                    chunk_index * chunk_size + slice.len()
                );
                let _ = std::io::stdout().flush();
            }

            for chunk_n in 0..slice.len() {
                let n = chunk_index * chunk_size + chunk_n;
                let mut matrix = ComplexMatrix {
                    data: vec![ZERO.into(); dims_t.pow(2)],
                    rows: dims_t,
                    cols: dims_t
                };

                for q in 0..nq_t {
                    for (i, &t_i) in ts[q].iter().enumerate() {
                        if i == 0 && q == 0 && is_simple_initial { continue };

                        let i_index = q * (n_t - 1) + i + init_offset - 1;
                        let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);

                        // The lapack functions assume column major order
                        for j in 0..n_t {
                            if j == 0 && q == 0 && is_simple_initial { continue };

                            let j_index = q * (n_t - 1) + j + init_offset - 1;
                            let is_bridge = (q != 0 && i == 0 && j == 0) ||
                                (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                            let m_n_i_j = if is_bridge {
                                2.0 * -alpha * ONE
                            } else {
                                I * lagrange_deriv(&ts[q], j, t_i) - if i == j { alpha } else { 0.0 }
                            };

                            matrix.data[i_index + j_index * matrix.rows] = m_n_i_j.into();
                        }
                    }
                }

                if let Some((t_0, _)) = &initial_time_settings {
                    for index in 0..dims {
                        // let (_, i) = get_indices(index % dims_t);
                        matrix.data[index * matrix.rows] = if n == index / dims_t {
                            (l(index % dims_t, *t_0)).into()
                        } else { 0.0.into() };
                    }
                }

                // if n == n_x / 2 && debug {
                //     std::thread::sleep(Duration::from_secs(1));
                //     println!("[{thread_id}]: Matrix @ {}:", n);
                //     matrix.print_fancy();
                // }

                let inverse =  matrix.inverse().unwrap();

                // if n == n_x / 2 && debug {
                //     println!("And inverse: ");
                //     inverse.print_fancy();
                // }

                slice[chunk_n] = inverse;
            }

            if debug {
                println!("[{thread_id}]: Finished with inverses for chunk #{chunk_index}");
                let _ = std::io::stdout().flush();
            }
        });

    for k in 0..max_iteration {
        (solution, prev_solution) = (prev_solution, solution);

        let max_err: f64 = solution.par_chunks_mut(dims_t * chunk_size)
            .enumerate()
            .map(|(chunk_index, slice)| {
                let thread_id = format!("{}", thread_id::get() % 1000).yellow();

                let mut max_err: f64 = 0.0;
                if debug {
                    println!(
                        "[{thread_id}]: Starting chunk #{chunk_index}: {}-{}",
                        chunk_index * chunk_size,
                        chunk_index * chunk_size + slice.len() / dims_t
                    );
                    let _ = std::io::stdout().flush();
                }

                let mut vector = ComplexVector {
                    data: vec![ZERO.into(); dims_t]
                };

                for chunk_n in 0..(slice.len() / dims_t) {
                    let n = chunk_index * chunk_size + chunk_n;

                    for q in 0..nq_t {
                        for (i, &t_i) in ts[q].iter().enumerate() {
                            if i == 0 && q == 0 && is_simple_initial { continue };

                            let i_index = q * (n_t - 1) + i + init_offset - 1;
                            let alpha = -2.0 * beta + potential(xs[n]) - time_dep(xs[n], t_i);
                            let is_bridge_i = (q != 0 && i == 0) ||
                                (q != nq_t - 1 && i == n_t - 1);

                            let mut b_i = if is_bridge_i {
                                2.0 * time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                            } else {
                                time_dep(xs[n], t_i) * psi_0(xs[n], t_i)
                            };

                            if 0 < n {
                                b_i -= prev_solution[i_index + (n-1) * dims_t] *
                                    if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                            }

                            if n < n_x - 1 {
                                b_i -= prev_solution[i_index + (n+1) * dims_t] *
                                    if is_bridge_i { 2.0 } else { 1.0 } * beta * ONE;
                            }

                            vector.data[i_index] = b_i.into();
                        }
                    }

                    if let Some((_, wave_func)) = &initial_time_settings {
                        vector.data[0] = wave_func[n].into();
                    }

                    let result = matrix_inverses[n].multiply_vector(&vector);

                    if n == n_x / 2 && debug {
                        // println!("\nMatrix fancy at {}:", n);
                        // matrix.print_fancy_with(&result, &vector);
                    }

                    for (i, c_i) in result.data.into_iter().map(Complex::from).enumerate() {
                        let index = n * dims_t + i;
                        max_err = max_err.max((c_i - prev_solution[index]).magnitude());
                        slice[i + chunk_n * dims_t] = c_i;
                    }
                }

                if debug {
                    let percent_change = if prev_max_err == 0.0 {
                        0.0
                    } else {
                        100.0 * (max_err - prev_max_err) / prev_max_err
                    };
                    let mut diff = format!("{}{:.4}%", if percent_change > 0.0 { "+" } else { "" }, percent_change).normal();

                    diff = if percent_change < 0.0 {
                        diff.green()
                    } else if percent_change > 0.0 {
                        diff.red()
                    } else {
                        diff.blue()
                    };
                    println!("[{thread_id}]: Finished chunk #{chunk_index} with k = {k: >3}: {:.4e}, {diff}", max_err);
                }

                max_err
            }).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        if debug {
            let percent_change = if prev_max_err == 0.0 {
                0.0
            } else {
                100.0 * (max_err - prev_max_err) / prev_max_err
            };
            let mut diff = format!("{}{:.4}%", if percent_change > 0.0 { "+" } else { "" }, percent_change).bold();

            diff = if percent_change < 0.0 {
                diff.green()
            } else if percent_change > 0.0 {
                diff.red()
            } else {
                diff.blue()
            };

            println!(
                "MAX ERROR ({k: >3}): {:.4e}, {diff}\n\n",
                max_err
            );

            prev_max_err = max_err;
        }

        if max_err.is_infinite() {
            if debug { println!("INFINITY!"); }
            max_k = usize::MAX;
            break;
        }

        if max_err < tolerance {
            if debug { println!("BROKEN!"); }
            max_k = k;
            break;
        }
    }

    let l = move |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims_t {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(solution[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(solution[index + n2 * dims_t] * l(index+1, t))
            } else {
                (solution[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    };

    if debug && options.plot_name.is_some() { println!("PLOTTING"); }

    if let Some(plot_name) = options.plot_name {
        plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, &psi_0, &plot_name);
    }

    (max_k, move |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        psi_0(x, t) + (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(solution[index + n1 * dims_t] * l(index+1, t)) +
                    x_mid * Complex::from(solution[index + n2 * dims_t] * l(index+1, t))
            } else {
                (solution[index + n1 * dims_t] * l(index+1, t)).into()
            }
        }).sum::<Complex>()
    })
}
#[test]
fn schrodinger_eq_discrete_jacobi() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 15.0);
    let (x_initial, x_final) = (-4e1, 4e1);
    let (n_t, n_x) = (30, 201);
    let nq_t = 1;

    let sigma = 3.0;
    let k0 = 0.0;
    let (t_0, psi_0) = (
        0.0,
        // |_x: f64| Complex::from(1.0)
        |x: f64| (-(x / sigma).powi(2) / 4.0 + I * k0 * x).exp() / (2.0 * PI * sigma.powi(2)).powf(0.25)
    );

    // Define the parameters for the differential equation
    let ang_frequency = 1000.0;
    let phase_shift = 0.0;
    let electric_0 = 1.0;
    let potential =
        |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
    // |x: f64| if 4.0 < x && x < 7.0 { 1000.0 } else { 0.0 };
    // |x: f64| 0.0;
    let (pulse_initial, pulse_final) = (0.0, 6.0);
    let electric_field = |t: f64| if pulse_initial <= t && t <= pulse_final {
        electric_0 * (PI * (t - pulse_initial) / (pulse_final - pulse_initial)).sin().powi(2)
    } else {
        0.0
    };

    let delta_time = (t_final - t_initial) / (nq_t as f64);
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2)) * ONE;
    let quad_points = gauss_lobatto_quadrature(n_t, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..nq_t)
        .map(|q| quad_points.iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();
    let xs: Vec<f64> = (0..n_x).map(|n| n as f64 * delta_x + x_initial).collect();

    let dims_t = n_t * nq_t - nq_t + 1;
    let dims = dims_t * n_x;
    let get_indices = |index| {
        // Gets the interval and the index within the interval
        if index == 0 {
            (0, 0)
        } else if index == dims_t - 1 {
            (nq_t - 1, n_t - 1)
        } else {
            ((index - 1) / (n_t - 1), (index - 1) % (n_t - 1) + 1)
        }
    };
    let l = |index, t| {
        let (q, i) = get_indices(index);

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q - 1], n_t - 1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims_t - 1 && i == n_t - 1 && ts[q][i] < t {
            lagrange(&ts[q + 1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
        } else {
            lagrange(&ts[q], i, t) // Use the lagrange in the current interval
        }
    };

    let result = jacobi_iteration(
        dims, 1e-12, 100,
        |index| {
            let (q, j) = get_indices(index % dims_t);
            let t_j = ts[q][j];
            let alpha = -2.0 * beta + potential(xs[index / dims_t]) -
                xs[index / dims_t] * electric_field(t_j) * (ang_frequency * t_j + phase_shift).sin();

            (alpha - lagrange_deriv(&ts[q], j, t_j) * I) * ws[j]
        },
        |index| {
            let n = index / dims_t;
            let (q, i) = get_indices(index % dims_t);
            let t_i = ts[q][i];
            let alpha = -2.0 * beta + potential(xs[n]) -
                xs[index / dims_t] * electric_field(t_i) * (ang_frequency * t_i + phase_shift).sin();

            if index % dims_t == 0 {
                (0..dims_t).map(|j| {
                    (j + n * dims_t, l(index % dims_t, t_0) * ONE)
                }).collect()
            } else {
                (0..dims_t).map(|j| {
                    let t_j = ts[q][j];
                    let is_bridge = (q != 0 && i == 0 && j == 0) ||
                        (q != nq_t - 1 && i == n_t - 1 && j == n_t - 1);

                    let a_ij = if is_bridge {
                        2.0 * ws[i] * alpha * ONE
                    } else {
                        ws[i] * (if i == j { alpha } else { 0.0.into() } - I * lagrange_deriv(&ts[q], i, t_j))
                    };

                    (j + n * dims_t, a_ij)
                }).chain([
                    if n > 0 { Some((i + (n-1) * dims_t, beta)) } else { None },
                    if n < dims_t - 1 { Some((i + (n+1) * dims_t, beta)) } else { None },
                ].into_iter().filter_map(|x| x)).collect()
            }
        },
        |index| {
            if index % dims_t == 0 {
                psi_0(xs[index / dims_t]).into()
            } else {
                0.0.into()
            }
        }
    );


    // (4) Test Accuracy
    let psi_computed = |x: f64, t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        if !(x_initial <= x && x < x_final) { panic!("x: {x} is out of bounds"); }

        let n1 = ((x - x_initial) / delta_x).floor() as usize;
        let n2 = ((x - x_initial) / delta_x).ceil() as usize;
        let x_mid = (x - xs[n1]) / delta_x;

        (0..dims_t).map(|index| {
            if n2 < n_x {
                (1.0 - x_mid) * Complex::from(result[index + n1 * dims_t] * l(index % dims_t, t)) +
                    x_mid * Complex::from(result[index + n2 * dims_t] * l(index % dims_t, t))
            } else {
                (result[index + n1 * dims_t] * l(index % dims_t, t)).into()
            }
        }).sum::<Complex>()
    };

    plot_result(t_initial, t_final, x_initial, x_final, &psi_computed, |x, _t| psi_0(x), "schrodinger_eq_discrete_jacobi");
}

pub fn jacobi_iteration(
    dims: usize,
    tolerance: f64, max_iter: usize,
    a_ii: impl Fn(usize) -> Complex,
    row_elements: impl Fn(usize) -> Vec<(usize, Complex)>,
    b_i: impl Fn(usize) -> Complex
) -> Vec<Complex> {
    let mut prev_solution = vec![ZERO; dims];
    let mut solution: Vec<Complex> = (0..dims).map(|i| b_i(i)).collect();

    for _ in 0..max_iter {
        let mut max_err: f64 = 0.0;

        for i in 0..dims {
            (prev_solution, solution) = (solution, prev_solution);

            let sum: Complex = row_elements(i).into_iter().map(|(j, a_ij)| {
                if i == j { ZERO } else { a_ij * prev_solution[j] }
            }).sum();

            solution[i] = (b_i(i) - sum) / a_ii(i);

            max_err = max_err.max((solution[i] - prev_solution[i]).magnitude());
        }

        if max_err < tolerance {
            break;
        }
    }

    solution
}

#[test]
fn schrodinger_eigen() {
    let n_x: usize = 25000;
    let delta_x: f64 = 0.01;
    let beta = -1.0 / (2.0 * delta_x.powi(2));
    let potential = |x: f64|  -1.0 / (1.0 + x.powi(2)).sqrt();

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); n_x.pow(2)],
        rows: n_x,
        cols: n_x
    };

    for i in 0..n_x {
        let x_i = (i as f64 - (n_x as f64) / 2.0) * delta_x;
        let alpha = -2.0 * beta + potential(x_i);
        // println!("x_{} = {:.4}", i, x_i);

        for j in 0..n_x {
            let m_i_j = if i == j {
                alpha.into()
            } else if i == j + 1 || i + 1 == j {
                beta.into()
            } else {
                0.0.into()
            };

            matrix.data[i + j * matrix.rows] = m_i_j;
        }
    }

    // matrix.print_fancy();
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);

    let mut eigenvalues = matrix.eigenvalues().unwrap();
    let mut max_err: f64 = 0.0;

    eigenvalues.sort_by(|z1, z2| z1.real().partial_cmp(&z2.real()).unwrap());

    println!("λ____ =    Expected vs   Computed  ");
    for (i, eigenvalue) in eigenvalues.iter().enumerate() {
        let Some(&expected_eigenvalue) = [
            -0.6698,
            -0.2749,
            -0.1515,
            -0.09270,
            -0.06354,
            -0.04550,
            -0.03461,
            -0.02689,
            -0.02171,
            -0.01773
        ].get(i) else { break };
        let err = (*eigenvalue - expected_eigenvalue).magnitude();
        println!("λ_{: <3} = {:.8} vs {:.8} --- {:.4e}", i, expected_eigenvalue, eigenvalue.real(), err);
        max_err = max_err.max(err);
    }

    println!("MAX ERROR: {:.4e}", max_err);
}

 */