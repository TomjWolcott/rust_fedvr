// use lapack::c64;
// use crate::bases::LagrangePolynomials;
// use crate::solvers::{Problem, Solver};
//
// pub struct SimpleOdeProblem;
//
// impl Problem for SimpleOdeProblem {
//     type In = f64;
//     type Out = c64;
// }
//
// pub struct SimpleDvrSolver {
//     polys: LagrangePolynomials
// }
//
// pub struct SimpleDvrSolution<'a> {
//     solution: Vec<c64>,
//     solver: &'a SimpleDvrSolver
// }
//
// impl Solver<SimpleOdeProblem> for SimpleDvrSolver {
//     type Solution = SimpleDvrSolution;
//
//
// }