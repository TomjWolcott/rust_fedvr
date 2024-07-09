# Finite Element Discrete Variable Representation for a simple 1D ODE in Rust
## Running the code
To run the code, first install [rust](https://www.rust-lang.org/learn/get-started) and then follow [these instructions](https://docs.rs/GSL/latest/rgsl/index.html) to download GSL.  Next clone this repository and navigate to the `cargo.toml` file, change "accelerate" in lapack's features list to one of the following:
|  OS | feature |
|-----|-------|
| MacOS | `"accelerate"` |
| Ubuntu | `"netlib"` |
| Windows | ??? |

Even though I'm not using this exact crate, look [here](https://docs.rs/nalgebra-lapack/latest/nalgebra_lapack/) for more information.

Once this is done you can run any function with the label `#[test]` using `cargo test [function name here] --release -- --nocapture`.  The simple ode problem tests output text in stdout comparing the found vs. expected solution, while the TDSE problem tests will save plots in html format in the output folder which will contain the wavefunction over time in a 3D plot (with the z axis representing magnitude and hue representing phase) and the population probability of the ground state over time underneath (you might have to move the mouse into the corner in order to scroll down).

Some things to keep in mind: `GetWaveFuncOptions` and `compute_wave_function` were my first attempt to abstract out some of the logic, I'll be rewriting that to use Tdse1dOptions in the future to take advantage of the LagrangePolynomials struct which isn't currently being used.

## Systems of Equations Visualizations
Largely to help with debugging, the systems of equations solved are printed out in the terminal as grids of colored blocks where each 
block: `██` is an element in the matrix/vector, the hue represents the complex phase (+1 = red, +i = green, -1 = cyan, -i = purple), and the brightness represents the magnitude on a log10 scale.
Additionally, zeros are displayed as grid markings: `┼─`.  Look at the example below to see how it is printed out

<img width="266" alt="Screenshot 2024-06-12 at 13 50 34" src="https://github.com/TomjWolcott/rust_fedvr_for_simple_ode/assets/134332655/625f8fab-092c-4945-9a65-b6399cbfc247">

This shows $`M_{i,j} * c_j = b_i`$ where $`M_{i,j}`$ and $`b_i`$ are computed and $`c_j`$ is solved for.  $`M_{i,j}`$ is the N x N block on the left, 
$`c_j`$ is in the middle and $`b_i`$ is on the right.  Also note that the rows of the matrix are indexed by $`i`$ and the columns are indexed by $`j`$.

**Note:** Not all terminals will print this nicely, some will ignore the color and others might interpret the color incorrectly.
