use lapack::c64;
use crate::complex_wrapper::{Complex, ComplexMatrix, ComplexVector, E, I, ONE, ZERO};
use crate::gauss_quadrature::gauss_lobatto_quadrature;

mod gauss_quadrature;
mod complex_wrapper;

fn main() {

}

/// Assumes that xs is sorted in increasing order
fn lagrange(xs: &Vec<f64>, i: usize, x: f64) -> f64 {
    if xs[0] - 1e-8 <= x && x <= *xs.last().unwrap() + 1e-8 {
        xs.iter().enumerate().fold(1.0, |acc, (j, &x_j)| {
            acc * if i == j { 1.0 } else { (x - x_j) / (xs[i] - x_j) }
        })
    } else {
        0.0
    }
}

fn lagrange_deriv(xs: &Vec<f64>, i: usize, mut x: f64) -> f64 {
    if xs[0] - 1e-8 <= x && x <= *xs.last().unwrap() + 1e-8 {
        x += 1e-12;

        lagrange(xs, i, x) * xs.iter().enumerate().map(|(j, &x_j)| {
            if i == j { 0.0 } else { 1.0 / (x - x_j) }
        }).sum::<f64>()
    } else {
        0.0
    }
}

#[test]
fn test_solving_ode_at_once() {
    // (0) Define the parameters
    let num_quad_points = 80;
    let (t_i, t_f) = (0.0, 5.0);
    let quad_points = gauss_lobatto_quadrature(num_quad_points, t_i, t_f);
    let (t_0, psi_0) = (1.0, 1.0);

    let ts: Vec<f64> = quad_points.iter().map(|(t_i, _)| *t_i).collect();
    let ws: Vec<f64> = quad_points.iter().map(|(_, w_i)| *w_i).collect::<Vec<f64>>();
    let l = |i, x| lagrange(&ts, i, x) / ws[i].sqrt();

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); num_quad_points.pow(2)],
        rows: num_quad_points,
        cols: num_quad_points
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); num_quad_points]
    };

    // (1) Set up the problem
    for (i, &(t_i, w_i)) in quad_points.iter().enumerate() {
        vector.data[i] = (w_i.sqrt() * t_i).into();
    }

    // The lapack functions assume column major order
    for (i, &(t_i, w_i)) in quad_points.iter().enumerate() {
        for (j, &w_j) in ws.iter().enumerate() {
            let m_i_j = I * w_i.sqrt() / w_j.sqrt() * lagrange_deriv(&ts, j, t_i) - if i == j { t_i } else { 0.0 };
            let index = j * num_quad_points + i;

            matrix.data[index] = m_i_j.into();
        }
    }

    // (2) Insert the initial Conditions
    vector.data[0] = psi_0.into();

    for j in 0..num_quad_points {
        let index = j * num_quad_points;

        matrix.data[index] = l(j, t_0).into();
    }

    // (3) Solve the systems of equations
    let result = matrix.solve_systems(vector.clone()).unwrap();

    println!("\nSystems of eqs:");
    matrix.print_fancy_with(&result, &vector);
    // println!("Matrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let exp_at_init = E.pow(-I * t_0.powi(2) / 2.0);
    let psi_expected = |t: f64| ((psi_0 + 1.0) / exp_at_init) * E.pow(-I * t.powi(2) / 2.0) - 1.0;
    let psi_computed = |t| result.data.iter().enumerate()
        .map(|(i, &c)| Complex::from(c) * l(i, t))
        .sum::<Complex>();

    let mut err_max: f64 = 0.0;

    println!("Ψ(______) =       Expected       vs       Computed      ");

    sample(100, t_i, t_f, |t| {
        let expected = psi_expected(t);
        let computed = psi_computed(t);
        let err = (expected - computed).magnitude();
        println!("Ψ({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", expected.to_string(), computed.to_string());
        err_max = err_max.max(err);
    });

    println!("MAX ERROR: {:.4e}", err_max);
}

#[test]
fn many_interval_ode_solving() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 5.0);
    let delta_time = 1.25;
    let num_quad_points = 20;
    let (t_0, psi_0) = (1.0, Complex::from(1.0));

    let (mut t_q, mut psi_q) = (t_0, psi_0);
    let num_intervals = ((t_final - t_initial) / delta_time) as usize;
    let mut quad_points = gauss_lobatto_quadrature(num_quad_points, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let mut ts: Vec<f64> = quad_points.iter().map(|&(t, _)| t).collect();

    let mut intervals_basis_fns: Vec<Box<dyn Fn(usize, f64) -> f64>> = Vec::new();
    let mut intervals_coefficients: Vec<Vec<c64>> = Vec::new();

    let mut matrix = ComplexMatrix {
        data: vec![ZERO.into(); num_quad_points.pow(2)],
        rows: num_quad_points,
        cols: num_quad_points
    };

    let mut vector = ComplexVector {
        data: vec![ZERO.into(); num_quad_points]
    };

    for q in 0..num_intervals {
        let time_end = (q + 1) as f64 * delta_time;
        let (ts_clone, ws_clone) = (ts.clone(), ws.clone());
        let l = Box::new(move |i: usize, x| {
            // let w: f64 = if is_on_boundary(i) { (2.0 * ws_clone[i]).sqrt() } else { ws_clone[i].sqrt() };

            lagrange(&ts_clone, i, x) / ws_clone[i].sqrt()
        });

        // (1) Set up the initial conditions
        vector.data[0] = psi_q.into();

        for j in 0..num_quad_points {
            let index = j * num_quad_points;

            matrix.data[index] = l(j, t_q).into();
        }

        // (2) Set up the problem
        for (i, &(t_i, w_i)) in quad_points.iter().enumerate().skip(1) {
            vector.data[i] = (w_i.sqrt() * t_i).into();

            // The lapack functions assume column major order
            for (j, &w_j) in ws.iter().enumerate() {
                let m_i_j = I * w_i.sqrt() / w_j.sqrt() * lagrange_deriv(&ts, j, t_i) - if i == j { t_i } else { 0.0 };
                let index = j * num_quad_points + i;

                matrix.data[index] = m_i_j.into();
            }
        }

        let result = matrix.solve_systems(vector.clone()).unwrap();

        println!("\nSystems of eqs:");
        matrix.print_fancy_with(&result, &vector);
        // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
        // println!("Vector (b_i): {:?}", vector);
        // println!("Result (c_j): {:?}\n", result);

        // (3) Restate the initial conditions
        t_q = time_end;
        psi_q = result.data.iter().enumerate()
            .map(|(i, &c)| Complex::from(c) * l(i, t_q))
            .sum::<Complex>();

        intervals_coefficients.push(result.data);
        intervals_basis_fns.push(l);

        quad_points.iter_mut().for_each(|(t, _)| *t += delta_time);
        ts.iter_mut().for_each(|t| *t += delta_time);
    }

    // (4) Test Accuracy
    let psi_computed = |t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        let index = ((t - t_initial) / delta_time) as usize;

        intervals_coefficients[index].iter()
            .enumerate()
            .map(|(i, &c_i)| Complex::from(c_i) * intervals_basis_fns[index](i, t))
            .sum::<Complex>()
    };

    let exp_at_init = E.pow(-I * t_0.powi(2) / 2.0);
    let psi_expected = |t: f64| {
        ((psi_0 + 1.0) / exp_at_init) * E.pow(-I * t.powi(2) / 2.0) - 1.0
    };

    let mut err_max: f64 = 0.0;

    println!("Ψ(______) =       Expected       vs       Computed      ");

    sample(100, t_initial, t_final, |t| {
        let expected = psi_expected(t);
        let computed = psi_computed(t);
        let err = (expected - computed).magnitude();
        println!("Ψ({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", expected.to_string(), computed.to_string());
        err_max = err_max.max(err);
    });

    println!("MAX ERROR: {:.4e}", err_max);
}

#[test]
fn ode_solving_via_bridges() {
    // (0) Define the parameters
    let (t_initial, t_final) = (0.0, 5.0);
    let n = 20;
    let num_intervals = 4;
    let (t_0, psi_0) = (1.0, Complex::from(1.0));

    let delta_time = (t_final - t_initial) / (num_intervals as f64);
    let quad_points = gauss_lobatto_quadrature(n, t_initial, t_initial + delta_time);
    let ws: Vec<f64> = quad_points.iter().map(|&(_, w)| w).collect();
    let ts: Vec<Vec<f64>> = (0..num_intervals)
        .map(|q| quad_points.iter().map(|&(t, _)| t + q as f64 * delta_time).collect())
        .collect();


    let dims = n * num_intervals - num_intervals + 1;
    let l = |index, t| {
        // Gets the interval and the index within the interval
        let (q, i) = if index == 0 {
            (0, 0)
        } else if index == dims-1 {
            (num_intervals-1, n-1)
        } else {
            ((index - 1) / (n-1), (index - 1) % (n-1) + 1)
        };

        if index != 0 && i == 0 && t < ts[q][i] {
            lagrange(&ts[q-1], n-1, t) // l^q_0 bridge, use the lagrange in the previous interval
        } else if index != dims-1 && i == n-1 && ts[q][i] < t {
            lagrange(&ts[q+1], 0, t) // l^q_{n-1} bridge, use the lagrange in the next interval
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

    // (1) Set up the problem
    for q in 0..num_intervals {
        for (i, (&t_i, &w_i)) in ts[q].iter().zip(ws.iter()).enumerate() {
            let i_index = q * (n-1) + i;

            let b_i = if q != 0 && i == 0 {
                2.0 * w_i * t_i
            } else if q != num_intervals-1 && i == n-1 {
                2.0 * w_i * t_i
            } else {
                w_i * t_i
            };

            vector.data[i_index] = b_i.into();

            // The lapack functions assume column major order
            for j in 0..n {
                let j_index = q * (n-1) + j;

                let m_i_j = if q != 0 && i == 0 && j == 0 {
                    -2.0 * w_i * t_i * ONE
                } else if q != num_intervals-1 && i == n-1 && j == n-1 {
                    -2.0 * w_i * t_i * ONE
                } else {
                    I * w_i * lagrange_deriv(&ts[q], j, t_i)
                        - w_i * if i == j { t_i } else { 0.0 }
                };

                matrix.data[i_index + j_index * matrix.rows] = m_i_j.into();
            }
        }
    }

    // (2) Set up the initial conditions
    vector.data[0] = psi_0.into();

    for index in 0..dims {
        matrix.data[index * matrix.rows] = l(index, t_0).into();
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

    println!("\nSystems of eqs:");
    matrix.print_fancy_with(&result, &vector);
    // println!("\nMatrix (M_{{i,j}}):\n{:?}", matrix);
    // println!("Vector (b_i): {:?}", vector);
    // println!("Result (c_j): {:?}\n", result);

    // (4) Test Accuracy
    let psi_computed = |t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }

        (0..dims).map(|index| {
            (result.data[index] * l(index, t)).into()
        }).sum::<Complex>()
    };

    let exp_at_init = E.pow(-I * t_0.powi(2) / 2.0);
    let psi_expected = |t: f64| {
        ((psi_0 + 1.0) / exp_at_init) * E.pow(-I * t.powi(2) / 2.0) - 1.0
    };

    let mut err_max: f64 = 0.0;

    println!("Ψ(______) =       Expected       vs       Computed      ");

    sample(100, t_initial, t_final, |t| {
        let expected = psi_expected(t);
        let computed = psi_computed(t);
        let err = (expected - computed).magnitude();
        println!("Ψ({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", expected.to_string(), computed.to_string());
        err_max = err_max.max(err);
    });

    println!("MAX ERROR: {:.4e}", err_max);
}

fn sample(num_samples: usize, from: f64, to: f64, mut func: impl FnMut(f64)) {
    let step = (to - from) / num_samples as f64;

    for i in 0..num_samples {
        let x = from + i as f64 * step;
        func(x);
    }

}