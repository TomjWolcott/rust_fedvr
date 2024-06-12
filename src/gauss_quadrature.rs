use itertools::Itertools;
use rgsl::legendre::polynomials::legendre_Pl;
use rgsl::numerical_differentiation::deriv_central;
use rgsl::Pow;

pub fn gauss_lobatto_quadrature(num_points: usize, from: f64, to: f64) -> Vec<(f64, f64)> {
    let legendre_poly_deriv = |x: f64| {
        (0..=(num_points / 2 - 1)).map(|i| {
            let n = (num_points % 2 + 2 * i) as i32;

            (2.0 * n as f64 + 1.0) * legendre_Pl(n, x.clamp(-1.0, 1.0))
        }).sum()
    };

    let intervals = 100 * num_points;
    let n = num_points as f64;

    let mut quad_points: Vec<(f64, f64)> = (0..=intervals)
        .map(|i| 2.0 * (i as f64) / intervals as f64 - 1.00000000001)
        .tuple_windows()
        .filter(|&(x1, x2)| legendre_poly_deriv(x1) * legendre_poly_deriv(x2) < 0.0)
        .map(|(x1, x2)| {
            let mut x = (x1 + x2) / 2.0;

            for _ in 0..200 {
                let prev_x = x;
                x = x - legendre_poly_deriv(x) / deriv_central(legendre_poly_deriv, x, 1e-9).unwrap().0;
                if (x - prev_x).abs() > 1e-13 { break; }
            }

            x
        }).map(|x| (x, 2.0 / (n * (n - 1.0) * legendre_Pl((n - 1.0) as i32, x).pow_2())))
        .map(|(x, w)| ((0.5 * x + 0.5) * (to - from) + from, w * (to - from) / 2.0))
        .collect();

    let end_weight = 2.0 / (n * (n - 1.0)) * (to - from) / 2.0;
    quad_points.insert(0, (from, end_weight));
    quad_points.push((to, end_weight));

    quad_points
}

#[test]
fn test_gauss_lobatto_quadrature() {
    let num_points = 4;
    let quad_points = gauss_lobatto_quadrature(num_points, -1.0, 1.0);

    for (x, w) in quad_points.iter() {
        println!("x = {}, w = {}", x, w);
    }
}