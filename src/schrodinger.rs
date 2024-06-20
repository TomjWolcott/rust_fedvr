use std::f64::consts::PI;
use plotly::{Configuration, Plot, Scatter3D};
use plotly::color::Rgb as PlotlyRgb;
use plotly::common::{Line, Marker, Mode};
use crate::complex_wrapper::{Complex, ComplexMatrix, ComplexVector, I, ONE, ZERO};
use crate::gauss_quadrature::gauss_lobatto_quadrature;
use crate::{lagrange, lagrange_deriv, sample};

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
    let psi_computed = |t: f64, x: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }

        (0..dims_t).map(|index| {
            (result.data[index] * l(index, t, x)).into()
        }).sum::<Complex>()
    };

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut colors = Vec::new();

    sample(500, t_initial, t_final, |t| {
        // println!("Ψ({t:.4}, ______)     = ____________________");
        sample(500, x_initial+1e-4, x_final, |x| {
            let computed = psi_computed(t, x);
            // println!("    Ψ({t:.4}, {x:.4}) = {: ^20}", computed.to_string());
            xs.push(x);
            ys.push(t);
            zs.push(computed.magnitude());
            let (r, g, b) = computed.rgb_just_hue();
            colors.push(PlotlyRgb::new(r, g, b));
        });
    });

    let scatter = Scatter3D::new(xs, ys, zs)
        .name("Ψ(x,t)")
        .line(Line::new().width(0.0))
        .mode(Mode::Markers)
        .marker(Marker::new().size(1).color_array(colors));

    let mut plot = Plot::new();
    plot.add_trace(scatter);
    plot.set_configuration(Configuration::new().fill_frame(true));
    plot.write_html("output/schrodinger_plot.html");
}

#[test]
fn schrodinger_eq_by_discretization() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 5.0);
    let (x_initial, x_final) = (-2e1, 2e1);
    let (n_t, n_x) = (20, 101);
    let nq_t = 3;

    let sigma = 3.0;
    let k0 = 0.0;
    let (t_0, psi_0) = (
        0.0,
        // |_x: f64| Complex::from(1.0)
        |x: f64| (-(x / sigma).powi(2) / 4.0 + I * k0 * x).exp() / (2.0 * PI * sigma.powi(2)).powf(0.25)
    );

    // Define the parameters for the differential equation
    let ang_frequency = 0.1;
    let phase_shift = 0.0;
    let electric_0 = 1.0;
    let potential =
        |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();
        // |x: f64| if 4.0 < x && x < 7.0 { 1000.0 } else { 0.0 };
        // |x: f64| 0.0;
    let electric_field = |t: f64| if t_initial <= t && t <= t_final {
        electric_0 * (PI * (t - t_initial) / (t_final - t_initial)).sin().powi(2)
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
                        2.0 * w_i * alpha * ONE
                    } else {
                        w_i * if i == j { alpha } else { 0.0 } -
                            I * w_i * lagrange_deriv(&ts[q], j, t_i)
                    };

                    matrix.data[i_index + (jt_index + n * dims_t) * matrix.rows] += m_n_i_j;

                    if 0 < n && i == j {
                        matrix.data[i_index + (jt_index + (n - 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * w_i * beta * ONE;
                    }

                    if n < n_x - 1 && i == j {
                        matrix.data[i_index + (jt_index + (n + 1) * dims_t) * matrix.rows] +=
                            if is_bridge { 2.0 } else { 1.0 } * w_i * beta * ONE;
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
    let psi_computed = |t: f64, x: f64| {
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

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    let mut colors = Vec::new();

    sample(200, t_initial, t_final, |t| {
        // println!("Ψ({t:.4}, ______)     = ____________________");
        sample(3000, x_initial, x_final, |x| {
            let computed = psi_computed(t, x);
            // println!("    Ψ({t:.4}, {x:.4}) = {: ^20}", computed.to_string());
            xs.push(x);
            ys.push(t);
            zs.push(computed.magnitude().powi(2));
            let (r, g, b) = computed.rgb_just_hue();
            colors.push(PlotlyRgb::new(r, g, b));
        });
    });

    // xs.append(&mut vec![0.0, 0.0]);
    // ys.append(&mut vec![0.0, 0.0]);
    // zs.append(&mut vec![0.4, 1.3]);

    let scatter = Scatter3D::new(xs, ys, zs)
        .name("Ψ(x,t)")
        .line(Line::new().width(0.0))
        .mode(Mode::Markers)
        .marker(Marker::new().size(1).color_array(colors));

    let mut plot = Plot::new();
    plot.add_trace(scatter);
    plot.set_configuration(Configuration::new().fill_frame(true));
    plot.write_html("output/schrodinger_plot_discretization.html");
}

#[test]
fn schrodinger_eigen() {
    let (x_initial, x_final) = (-1000.0, 1000.0);
    let n_x: usize = 10;
    let delta_x = (x_final - x_initial) / (n_x as f64 - 1.0);
    let beta = 1.0 / (-2.0 * delta_x.powi(2));
    let potential = |x: f64| -1.0 / (1.0 + x.powi(2)).sqrt();

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); n_x.pow(2)],
        rows: n_x,
        cols: n_x
    };

    for i in 0..n_x {
        let x_i = (i as f64) / (n_x as f64 - 1.0) * (x_final - x_initial) + x_initial;
        let alpha = -2.0 * beta + potential(x_i);
        println!("x_{} = {:.4}", i, x_i);

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

    matrix.print_fancy();

    let eigenvalues = matrix.eigenvalues().unwrap();
    let mut max_err: f64 = 0.0;

    println!("λ____ =    Expected vs   Computed  ");
    for (i, eigenvalue) in eigenvalues.iter().enumerate() {
        let expected_eigenvalue = 1.0 / delta_x.powi(2) * (1.0 - ((i + 1) as f64 * PI / (n_x + 2) as f64).cos());
        let err = (*eigenvalue - expected_eigenvalue).abs();
        println!("λ_{: <3} = {:.8} vs {:.8} --- {:.4e}", i, eigenvalue.real(), expected_eigenvalue, err);
        max_err = max_err.max(err);
    }

    println!("MAX ERROR: {:.4e}", max_err);
}
