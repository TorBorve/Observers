use std::ops::{Deref, DerefMut};

use rand::distributions::Distribution;

extern crate nalgebra as na;

/// Linear Time-Invariant System
/// x(k+1) = A*x(k) + B*u(k)
/// y(k) = C*x(k) + D*u(k)
/// No noise is considered
#[allow(non_snake_case)]
struct LinearSystem<const NX: usize, const NY: usize, const NU: usize> {
    A: na::SMatrix<f64, NX, NX>,
    B: na::SMatrix<f64, NX, NU>,
    C: na::SMatrix<f64, NY, NX>,
    D: na::SMatrix<f64, NY, NU>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> LinearSystem<NX, NY, NU> {
    pub fn new(
        A: na::SMatrix<f64, NX, NX>,
        B: na::SMatrix<f64, NX, NU>,
        C: na::SMatrix<f64, NY, NX>,
        D: na::SMatrix<f64, NY, NU>,
    ) -> Self {
        Self { A, B, C, D }
    }
}

/// Linear Time-Invariant System with Gaussian noise
/// x(k+1) = A*x(k) + B*u(k) + w(k)
/// y(k) = C*x(k) + D*u(k) + v(k)
struct GaussianLinearSystem<const NX: usize, const NY: usize, const NU: usize> {
    system: LinearSystem<NX, NY, NU>,
    w_cov: na::SMatrix<f64, NX, NX>,
    v_cov: na::SMatrix<f64, NY, NY>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> GaussianLinearSystem<NX, NY, NU> {
    pub fn new(
        A: na::SMatrix<f64, NX, NX>,
        B: na::SMatrix<f64, NX, NU>,
        C: na::SMatrix<f64, NY, NX>,
        D: na::SMatrix<f64, NY, NU>,
        w_cov: na::SMatrix<f64, NX, NX>,
        v_cov: na::SMatrix<f64, NY, NY>,
    ) -> Self {
        Self {
            system: LinearSystem::new(A, B, C, D),
            w_cov,
            v_cov,
        }
    }
}

trait Model<const NX: usize, const NY: usize, const NU: usize> {
    /// Simulate the system for a given initial state and input sequence
    /// Returns the state and output series (x(k), y(k))
    fn simulate(
        &self,
        x0: &na::SVector<f64, NX>,
        u: &Vec<na::SVector<f64, NU>>,
    ) -> (Vec<na::SVector<f64, NX>>, Vec<na::SVector<f64, NY>>);
    /// Simulate one discrete time step
    /// Returns the next state and the output (x(k+1), y(k))
    fn simulate_step(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
    ) -> (na::SVector<f64, NX>, na::SVector<f64, NY>);
}

impl<const NX: usize, const NY: usize, const NU: usize> Model<NX, NY, NU>
    for LinearSystem<NX, NY, NU>
{
    fn simulate(
        &self,
        x0: &na::SVector<f64, NX>,
        u: &Vec<na::SVector<f64, NU>>,
    ) -> (Vec<na::SVector<f64, NX>>, Vec<na::SVector<f64, NY>>) {
        let mut x_k = *x0;
        let mut y_series = Vec::with_capacity(u.len());
        let mut x_series = Vec::with_capacity(u.len());

        for u_k in u {
            let y_k = self.C * x_k + self.D * u_k;
            let x_k_next = self.A * x_k + self.B * u_k;

            y_series.push(y_k);
            x_series.push(x_k); // align with y_k
            x_k = x_k_next;
        }
        assert!(y_series.len() == u.len());
        assert!(x_series.len() == u.len());
        (x_series, y_series)
    }

    fn simulate_step(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
    ) -> (na::SVector<f64, NX>, na::SVector<f64, NY>) {
        let x_next = self.A * x + self.B * u;
        let y = self.C * x + self.D * u;
        (x_next, y)
    }
}

/// Generate a multivariate normal sampler for a given mean and covariance
/// The sampler is a closure that returns a sample of the multivariate normal distribution
fn multivariate_normal_sampler<const N: usize>(
    mean: &na::SVector<f64, N>,
    cov: &na::SMatrix<f64, N, N>,
) -> Box<dyn FnMut() -> na::SVector<f64, N>> {
    let mean_dyn = na::DVector::<f64>::from_iterator(N, mean.iter().copied());
    if *cov == na::SMatrix::<f64, N, N>::zeros() {
        // no noise
        let mean_closure = mean.clone();
        let gen_func = move || -> na::SVector<f64, N> { mean_closure };
        return Box::new(gen_func);
    }
    let cov_dyn = na::DMatrix::<f64>::from_iterator(N, N, cov.iter().copied());
    let sampler =
        statrs::distribution::MultivariateNormal::new_from_nalgebra(mean_dyn, cov_dyn).unwrap();
    use statrs::statistics::VarianceN;
    assert!(*cov == sampler.variance().unwrap());
    let mut r = rand::rngs::OsRng;
    let gen_func = move || -> na::SVector<f64, N> {
        na::SVector::<f64, N>::from_iterator(sampler.sample(&mut r).iter().copied())
    };
    Box::new(gen_func)
}

impl<const NX: usize, const NY: usize, const NU: usize> Model<NX, NY, NU>
    for GaussianLinearSystem<NX, NY, NU>
{
    fn simulate(
        &self,
        x0: &na::SVector<f64, NX>,
        u: &Vec<na::SVector<f64, NU>>,
    ) -> (Vec<na::SVector<f64, NX>>, Vec<na::SVector<f64, NY>>) {
        let mut x_k = *x0;
        let mut y_series = Vec::with_capacity(u.len());
        let mut x_series = Vec::with_capacity(u.len());

        let mut v_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NY>::zeros(), &self.v_cov);
        let mut w_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NX>::zeros(), &self.w_cov);

        for u_k in u {
            let v_k = v_sampler();
            let w_k = w_sampler();

            let y_k = self.system.C * x_k + self.system.D * u_k + v_k;
            let x_k_next = self.system.A * x_k + self.system.B * u_k + w_k;

            y_series.push(y_k);
            x_series.push(x_k); // align with y_k
            x_k = x_k_next;
        }
        assert!(y_series.len() == u.len());
        assert!(x_series.len() == u.len());
        (x_series, y_series)
    }
    fn simulate_step(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
    ) -> (na::SVector<f64, NX>, na::SVector<f64, NY>) {
        let mut v_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NY>::zeros(), &self.v_cov);
        let mut w_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NX>::zeros(), &self.w_cov);

        let v = v_sampler();
        let w = w_sampler();

        let x_next = self.system.A * x + self.system.B * u + w;
        let y = self.system.C * x + self.system.D * u + v;
        (x_next, y)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn simulate_linear() {
        const NX: usize = 2;
        const NY: usize = 2;
        const NU: usize = 1;

        let a_matrix = na::SMatrix::<f64, NX, NX>::new_random();
        let b_matrix = na::SMatrix::<f64, NX, NU>::new_random();
        let c_matrix = na::SMatrix::<f64, NY, NX>::new_random();
        let d_matrix = na::SMatrix::<f64, NY, NU>::new_random();

        let system = super::LinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix);

        let x_init = na::SVector::<f64, NX>::new_random();
        let u = vec![na::SVector::<f64, NU>::new_random(); 100];

        use super::Model;
        let (x_series, y_series) = system.simulate(&x_init, &u);

        assert_eq!(x_series.len(), u.len());
        assert_eq!(y_series.len(), u.len());
        assert_eq!(
            x_series[0], x_init,
            "Initial state is the first element of x_series"
        );

        let (x_next, y) = system.simulate_step(&x_init, &u[0]);
        assert_eq!(x_series[1], x_next, "x(1) is located at x_series[1]");
        assert_eq!(y_series[0], y, "y(0) is located at y_series[0]");
    }

    #[test]
    fn simulate_gaussian_linear() {
        const NX: usize = 2;
        const NY: usize = 2;
        const NU: usize = 1;

        let a_matrix = na::SMatrix::<f64, NX, NX>::new_random();
        let b_matrix = na::SMatrix::<f64, NX, NU>::new_random();
        let c_matrix = na::SMatrix::<f64, NY, NX>::new_random();
        let d_matrix = na::SMatrix::<f64, NY, NU>::new_random();

        let w_cov_sqrt = na::SMatrix::<f64, NX, NX>::new_random().lower_triangle();
        let w_cov = w_cov_sqrt.transpose() * w_cov_sqrt; // ensure positive definite
        let v_cov_sqrt = na::SMatrix::<f64, NY, NY>::new_random().lower_triangle();
        let v_cov = v_cov_sqrt.transpose() * v_cov_sqrt;

        let system =
            super::GaussianLinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix, w_cov, v_cov);

        let x_init = na::SVector::<f64, NX>::new_random();
        let u = vec![na::SVector::<f64, NU>::new_random(); 100];

        use super::Model;
        let (x_series, y_series) = system.simulate(&x_init, &u);

        assert_eq!(x_series.len(), u.len());
        assert_eq!(y_series.len(), u.len());
        assert_eq!(
            x_series[0], x_init,
            "Initial state is the first element of x_series"
        );

        let (x_next, y) = system.simulate_step(&x_init, &u[0]);
    }

    #[test]
    fn simulate_gaussian_linear_no_noise() {
        const NX: usize = 5;
        const NY: usize = 5;
        const NU: usize = 2;

        let a_matrix = na::SMatrix::<f64, NX, NX>::new_random();
        let b_matrix = na::SMatrix::<f64, NX, NU>::new_random();
        let c_matrix = na::SMatrix::<f64, NY, NX>::new_random();
        let d_matrix = na::SMatrix::<f64, NY, NU>::new_random();

        let w_cov = na::SMatrix::<f64, NX, NX>::zeros();
        let v_cov = na::SMatrix::<f64, NY, NY>::zeros();

        let system_guassian =
            super::GaussianLinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix, w_cov, v_cov);
        let system_linear = super::LinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix);

        let x_init = na::SVector::<f64, NX>::new_random();
        let u = vec![na::SVector::<f64, NU>::new_random(); 100];

        use super::Model;
        let (x_series_gaussian, y_series_gaussian) = system_guassian.simulate(&x_init, &u);
        let (x_series_lin, y_series_lin) = system_linear.simulate(&x_init, &u);

        assert_eq!(
            x_series_gaussian, x_series_lin,
            "No noise, the state series should be the same"
        );
        assert_eq!(
            y_series_gaussian, y_series_lin,
            "No noise, the output series should be the same"
        );
    }
}
