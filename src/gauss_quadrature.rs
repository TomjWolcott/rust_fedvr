use std::f64::consts::PI;
use itertools::Itertools;
use lapack::c64;
use plotly::Plot;
use rgsl::legendre::polynomials::{legendre_Pl, legendre_Pl_deriv_array};
use rgsl::numerical_differentiation::deriv_central;

fn signed_powf(x: f64, p: f64) -> f64 {
    x.signum() * x.abs().powf(p)
}

fn interval_point_spread(x: f64) -> f64 {
    (PI * (x - 1.0) / 2.0).cos()
}

pub fn gauss_lobatto_quadrature(num_points: usize, from: f64, to: f64) -> Vec<(f64, f64)> {
    let intervals = 2 * num_points;
    let n = num_points as f64;

    let legendre_poly_deriv = |mut x: f64| {
        x = x.clamp(-1.0 + 1e-8, 1.0 - 1e-8);
        (n-1.0) / (1.0 - x*x) * (legendre_Pl(num_points as i32 - 2, x) - x * legendre_Pl(num_points as i32 - 1, x))
    };

    let mut quad_points: Vec<(f64, f64)> = (0..=intervals)
        .map(|i| interval_point_spread(2.0 * (i as f64) / (intervals-1) as f64 - 1.0))
        .tuple_windows()
        .filter(|&(x1, x2)| legendre_poly_deriv(x1) * legendre_poly_deriv(x2) < 0.0)
        .map(|(x1, x2)| {
            let mut x = (x1 + x2) / 2.0;

            for k in 0..2000 {
                let prev_x = x;
                x = x - legendre_poly_deriv(x) / deriv_central(legendre_poly_deriv, x, 1e-10).unwrap().0;
                if (x - prev_x).abs() < 1e-10 { break; }
            }

            x
        }).map(|x| (x, 2.0 / (n * (n - 1.0) * legendre_Pl((n - 1.0) as i32, x).powi(2))))
        .map(|(x, w)| ((0.5 * x + 0.5) * (to - from) + from, w * (to - from) / 2.0))
        .collect();

    let end_weight = 2.0 / (n * (n - 1.0)) * (to - from) / 2.0;
    quad_points.insert(0, (from, end_weight));
    quad_points.push((to, end_weight));

    println!("Final # of points: {}", quad_points.len());

    quad_points
}

pub fn gauss_lobatto_quadrature2(num_points: usize, from: f64, to: f64) -> Vec<(f64, f64)> {
    let legendre_poly_deriv = |x: f64| {
        (0..=(num_points / 2 - 1)).map(|i| {
            let n = (num_points % 2 + 2 * i) as i32;

            (2.0 * n as f64 + 1.0) * legendre_Pl(n, x.clamp(-1.0, 1.0))
        }).sum::<f64>()
    };
    let n = num_points as f64;

    let mut points: Vec<f64> = (0..num_points-2).map(|i| {
        1.0 * (2.0 * ((i+1) as f64 / (num_points-1) as f64) - 1.0)
    }).collect();

    let mut prev_points = points.clone();

    for _ in 0..2000 {
        let mut max_err: f64 = 0.0;

        for i in (0..points.len()) {
            points[i] = points[i] - legendre_poly_deriv(points[i]) / points.iter()
                .filter(|x_cur| **x_cur != points[i])
                .map(|x_cur| points[i] - x_cur)
                .product::<f64>();

            max_err = max_err.max((points[i] - prev_points[i]).abs());
        }

        if max_err < 1e-10 { break; }
    }

    points.sort_by(|x1, x2| x1.partial_cmp(x2).unwrap());

    let mut quad_points: Vec<(f64, f64)> = points.iter()
        .map(|x| (x, 2.0 / (n * (n - 1.0) * legendre_Pl((n - 1.0) as i32, *x).powi(2))))
        .map(|(x, w)| ((0.5 * x + 0.5) * (to - from) + from, w * (to - from) / 2.0))
        .collect();

    let end_weight = 2.0 / (n * (n - 1.0)) * (to - from) / 2.0;
    quad_points.insert(0, (from, end_weight));
    quad_points.push((to, end_weight));

    quad_points
}

#[test]
fn test_gauss_lobatto_quadrature() {
    let num_points = 20000;
    println!("start");
    let quad_points = gauss_lobatto_quadrature(num_points, -1.0, 1.0);

    println!("points: {}", quad_points.len());

    for (i, (x, w)) in quad_points.iter().enumerate() {
        // println!("[{i: >5}]: x = {}, w = {}", x, w);
    }

    let xs = (0..quad_points.len()).map(|i| 2.0 * i as f64 / (quad_points.len() - 1) as f64 - 1.0).collect_vec();
    let ys_e = quad_points.iter().map(|(x, _)| *x).collect_vec();
    let ys_c = xs.iter().map(|x| interval_point_spread(*x)).collect_vec();
    let ys_err = ys_e.iter().zip(ys_c.iter()).map(|(e, c)| (e - c).abs()).collect_vec();

    let mut max_err: f64 = 0.0;

    for (e, c) in ys_e.iter().zip(ys_c.iter()) {
        max_err = max_err.max((e - c).abs());

        // println!("e = {}, c = {}, err = {}", e, c, (e - c).abs());
    }

    println!("max_err = {:.4e}", max_err);

    // plot points
    let mut plot = Plot::new();
    plot.add_trace(plotly::Scatter::new(
        xs.clone(),
        ys_e
    ).name("points"));
    plot.add_trace(plotly::Scatter::new(
        xs.clone(),
        ys_c,
    ).name("point spread"));
    plot.add_trace(plotly::Scatter::new(
        xs,
        ys_err,
    ).name("error"));
    plot.write_html("output/gauss_lobatto_quadrature.html");
}