use lapack::c64;
use rgsl::ComplexF64;
use rgsl::numerical_differentiation::deriv_central;
use crate::complex_wrapper::{Complex, ComplexMatrix, ComplexVector, E, I, ZERO};
use crate::gauss_quadrature::gauss_lobatto_quadrature;

mod gauss_quadrature;
mod complex_wrapper;
mod ode_solver;

fn main() {
    println!("Hello, world!");
}

fn lagrange(xs: &Vec<f64>, i: usize, x: f64) -> f64 {
    xs.iter().enumerate().fold(1.0, |acc, (j, x_j)| {
        acc * if i == j { 1.0 } else { (x - x_j) / (xs[i] - x_j) }
    })
}

#[test]
fn test_lagrange() {
    println!("{}", lagrange(&vec![0.0, 1.1, 2.02, 3.8], 2, 3.80000000001));
}

fn lagrange_deriv(xs: &Vec<f64>, i: usize, mut x: f64) -> f64 {
    x += 1e-12;

    lagrange(xs, i, x) * xs.iter().enumerate().map(|(j, &x_j)| {
        if i == j { 0.0 } else { 1.0 / (x - x_j) }
    }).sum::<f64>()
}

#[test]
fn test_lagrange_derivative() {
    let xs = vec![0.0, 1.0, 2.1, 2.6, 3.99];
    let test_xs = [1.2, 0.2, 5.2, 1.1, 3.4, 1.0];
    let i = 2;

    for x in test_xs.iter() {
        let computed = lagrange_deriv(&xs, i, *x);
        let expected = deriv_central(|x| lagrange(&xs, i, x), *x, 1e-10).unwrap().0;

        println!("l_i'({x}) = {} vs {}", computed, expected);
        assert!((computed - expected).abs() < 1e-4);
    }
}

#[test]
fn test_solving_ode() {
    let num_quad_points = 51;
    let (t_i, t_f) = (0.0, 1.0);
    let quad_points = gauss_lobatto_quadrature(num_quad_points, t_i, t_f);
    let (t_0, psi_0) = (0.0, 1.0);

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

    // (3) Print out the solution
    let result = matrix.solve_systems(vector.clone()).unwrap();

    println!("Matrix (M_{{i,j}}):\n{:?}", matrix);
    println!("Vector (b_i): {:?}", vector);
    println!("Result (c_j): {:?}\n", result);

    // (4) Compute error
    let exp_at_init = E.pow(-I * t_0.powi(2) / 2.0);
    let psi_expected = |t: f64| ((psi_0 + 1.0) / exp_at_init) * E.pow(-I * t.powi(2) / 2.0) - 1.0;
    let psi_computed = |t| result.data.iter().enumerate()
        .map(|(i, &c)| Complex::from(c) * l(i, t))
        .sum::<Complex>();

    let mut err_max: f64 = 0.0;
    let num_tests = 201;

    println!("Ψ(______) =       Expected       vs       Computed      ");

    for i in 0..num_tests {
        let t = i as f64 / (num_tests - 1) as f64 * (t_f - t_i) + t_i;

        let err = (psi_expected(t) - psi_computed(t)).magnitude();
        err_max = err_max.max(err);

        println!("Ψ({t:.4}) = {: ^20} vs {: ^20} -- error: {:.4e}", psi_expected(t).to_string(), psi_computed(t).to_string(), err);
    }

    println!("MAX ERROR: {:.4e}", err_max);
}

#[test]
fn many_interval_ode_solving() {
    let (t_initial, t_final) = (0.0, 200.0);
    let delta_time = 1.0;
    let num_quad_points = 500;
    // let (t_0, psi_0) = (1.0, Complex::from(c64::new(-0.7950, -1.9895)));
    let (t_0, psi_0) = (0.0, Complex::from(1.0));

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
        let (time_start, time_end) = (q as f64 * delta_time, (q + 1) as f64 * delta_time);
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
        }

        // The lapack functions assume column major order
        for (i, &(t_i, w_i)) in quad_points.iter().enumerate().skip(1) {
            for (j, &w_j) in ws.iter().enumerate() {
                let m_i_j = I * w_i.sqrt() / w_j.sqrt() * lagrange_deriv(&ts, j, t_i) - if i == j { t_i } else { 0.0 };
                let index = j * num_quad_points + i;

                matrix.data[index] = m_i_j.into();
            }
        }

        let result = matrix.solve_systems(vector.clone()).unwrap();

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

    let psi_computed = |t: f64| {
        if !(t_initial <= t && t < t_final) { panic!("t: {t} is out of bounds"); }
        let index = ((t - t_0) / delta_time) as usize;

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

    sample(1000, t_initial, t_final, |t| {
        let err = (psi_expected(t) - psi_computed(t)).magnitude();
        println!("Ψ({t:.4}) = {: ^20} vs {: ^20} -- error: {err:.4e}", psi_expected(t).to_string(), psi_computed(t).to_string());
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

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use lapack::fortran::{dgesv, zgesv};

    #[test]
    fn lapack() {
        let mut a = vec![1.0, 3.0, 2.0, 5.0];
        let mut b = vec![1.0, 4.0];
        let mut ipiv = vec![0; 2];
        let mut info = 0;

        dgesv(
            2,
            1,
            &mut a,
            2,
            &mut ipiv,
            &mut b,
            2,
            &mut info
        );

        println!("ipiv: {:?}, {:?}", ipiv, b);
    }
}
