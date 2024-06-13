extern crate nalgebra as na;
use rand::distributions::Distribution;

pub trait ObserverModel<
    const NX: usize,
    const NU: usize,
    const NW: usize,
    const NY: usize,
    const ND: usize,
    const NV: usize,
>
{
    fn state_model(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NW>,
    ) -> na::SVector<f64, NX>;

    fn meas_model(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, ND>,
        v: &na::SVector<f64, NV>,
    ) -> na::SVector<f64, NY>;

    fn gen_state_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, NW>>;

    fn gen_meas_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, NV>>;

    fn simulate(
        &self,
        x0: &na::SVector<f64, NX>,
        u_series: &Vec<na::SVector<f64, NU>>,
        d_series: &Vec<na::SVector<f64, ND>>,
    ) -> (Vec<na::SVector<f64, NX>>, Vec<na::SVector<f64, NY>>) {
        let num_steps = u_series.len();
        assert_eq!(u_series.len(), d_series.len());
        let v_series = self.gen_meas_noise(num_steps);
        let w_series = self.gen_state_noise(num_steps);

        let mut y_series = Vec::with_capacity(num_steps);
        let mut x_series = Vec::with_capacity(num_steps);

        let mut x_k = *x0;

        for i in 0..num_steps {
            let d_k = &d_series[i];
            let u_k = &u_series[i];
            let v_k = &v_series[i];
            let w_k = &w_series[i];

            let y_k = self.meas_model(&x_k, d_k, v_k);
            let x_k_next = self.state_model(&x_k, u_k, w_k);

            y_series.push(y_k);
            x_series.push(x_k);

            x_k = x_k_next;
        }

        assert_eq!(y_series.len(), num_steps);
        assert_eq!(x_series.len(), num_steps);

        (x_series, y_series)
    }
}

pub trait Differentiable<const NX: usize, const NU: usize, const NW: usize, const NY: usize, const ND: usize, const NV: usize> {
    fn state_model_dx(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NW>,
    ) -> na::SMatrix<f64, NX, NX>;

    fn state_model_dw(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NW>,
    ) -> na::SMatrix<f64, NX, NW>;

    fn meas_model_dx(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, ND>,
        v: &na::SVector<f64, NV>,
    ) -> na::SMatrix<f64, NY, NX>;

    fn meas_model_dv(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, ND>,
        v: &na::SVector<f64, NV>,
    ) -> na::SMatrix<f64, NY, NV>;
}

pub trait Linear<const NX: usize, const NU: usize, const NW: usize, const NY: usize, const ND: usize, const NV: usize> {
    fn linear_state_model_dx(&self) -> na::SMatrix<f64, NX, NX>;

    fn linear_state_model_du(&self) -> na::SMatrix<f64, NX, NU>;

    fn linear_state_model_dw(&self) -> na::SMatrix<f64, NX, NW>;

    fn linear_meas_model_dx(&self) -> na::SMatrix<f64, NY, NX>;

    fn linear_meas_model_dd(&self) -> na::SMatrix<f64, NY, ND>;

    fn linear_meas_model_dv(&self) -> na::SMatrix<f64, NY, NV>;
}

impl<T, const NX: usize, const NU: usize, const NW: usize, const NY: usize, const ND: usize, const NV: usize>
    Differentiable<NX, NU, NW, NY, ND, NV> for T
where
    T: Linear<NX, NU, NW, NY, ND, NV>,
{
    fn state_model_dx(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NW>,
    ) -> na::SMatrix<f64, NX, NX> {
        _ = (x, u, w);
        self.linear_state_model_dx()
    }

    fn state_model_dw(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NW>,
    ) -> na::SMatrix<f64, NX, NW> {
        _ = (x, u, w);
        self.linear_state_model_dw()
    }

    fn meas_model_dx(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, ND>,
        v: &na::SVector<f64, NV>,
    ) -> na::SMatrix<f64, NY, NX> {
        _ = (x, d, v);
        self.linear_meas_model_dx()
    }

    fn meas_model_dv(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, ND>,
        v: &na::SVector<f64, NV>,
    ) -> na::SMatrix<f64, NY, NV> {
        _ = (x, d, v);
        self.linear_meas_model_dv()
    }
}

pub trait NoiseNormalDistrubuted<const NV: usize, const NW: usize> {
    fn mean_state_noise(&self) -> na::SVector<f64, NW>;
    fn cov_state_noise(&self) -> na::SMatrix<f64, NW, NW>;

    fn mean_meas_noise(&self) -> na::SVector<f64, NV>;
    fn cov_meas_noise(&self) -> na::SMatrix<f64, NV, NV>;
}

/// Linear Time-Invariant System
/// x(k+1) = A*x(k) + B*u(k)
/// y(k) = C*x(k) + D*u(k)
/// No noise is considered
#[derive(Copy, Clone)]
pub struct LinearSystem<const NX: usize, const NY: usize, const NU: usize> {
    pub a_matrix: na::SMatrix<f64, NX, NX>,
    pub b_matrix: na::SMatrix<f64, NX, NU>,
    pub c_matrix: na::SMatrix<f64, NY, NX>,
    pub d_matrix: na::SMatrix<f64, NY, NU>,
}

impl<const NX: usize, const NY: usize, const NU: usize> LinearSystem<NX, NY, NU> {
    pub fn new(
        a_matrix: na::SMatrix<f64, NX, NX>,
        b_matrix: na::SMatrix<f64, NX, NU>,
        c_matrix: na::SMatrix<f64, NY, NX>,
        d_matrix: na::SMatrix<f64, NY, NU>,
    ) -> Self {
        Self {
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
        }
    }
}

impl<const NX: usize, const NY: usize, const NU: usize> ObserverModel<NX, NU, 0, NY, NU, 0>
    for LinearSystem<NX, NY, NU>
{
    fn state_model(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, 0>,
    ) -> na::SVector<f64, NX> {
        _ = w;
        self.a_matrix * x + self.b_matrix * u
    }

    fn meas_model(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, NU>,
        v: &na::SVector<f64, 0>,
    ) -> na::SVector<f64, NY> {
        _ = v;
        self.c_matrix * x + self.d_matrix * d
    }

    fn gen_state_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, 0>> {
        vec![na::SVector::<f64, 0>::zeros(); num_noise_samples]
    }

    fn gen_meas_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, 0>> {
        vec![na::SVector::<f64, 0>::zeros(); num_noise_samples]
    }
}

impl<const NX: usize, const NY: usize, const NU: usize> Linear<NX, NU, 0, NY, NU, 0>
    for LinearSystem<NX, NY, NU>
{
    fn linear_state_model_dx(&self) -> na::SMatrix<f64, NX, NX> {
        self.a_matrix
    }

    fn linear_state_model_du(&self) -> na::SMatrix<f64, NX, NU> {
        self.b_matrix
    }

    fn linear_state_model_dw(&self) -> na::SMatrix<f64, NX, 0> {
        na::SMatrix::<f64, NX, 0>::zeros()
    }

    fn linear_meas_model_dx(&self) -> na::SMatrix<f64, NY, NX> {
        self.c_matrix
    }

    fn linear_meas_model_dd(&self) -> na::SMatrix<f64, NY, NU> {
        self.d_matrix
    }

    fn linear_meas_model_dv(&self) -> na::SMatrix<f64, NY, 0> {
        na::SMatrix::<f64, NY, 0>::zeros()
    }
}

/// Linear Time-Invariant System with Gaussian noise
/// x(k+1) = A*x(k) + B*u(k) + w(k)
/// y(k) = C*x(k) + D*u(k) + v(k)
#[derive(Copy, Clone)]
pub struct GaussianLinearSystem<const NX: usize, const NY: usize, const NU: usize> {
    pub system: LinearSystem<NX, NY, NU>,
    pub w_cov: na::SMatrix<f64, NX, NX>,
    pub v_cov: na::SMatrix<f64, NY, NY>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> GaussianLinearSystem<NX, NY, NU> {
    pub fn new(
        a_matrix: na::SMatrix<f64, NX, NX>,
        b_matrix: na::SMatrix<f64, NX, NU>,
        c_matrix: na::SMatrix<f64, NY, NX>,
        d_matrix: na::SMatrix<f64, NY, NU>,
        w_cov: na::SMatrix<f64, NX, NX>,
        v_cov: na::SMatrix<f64, NY, NY>,
    ) -> Self {
        Self {
            system: LinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix),
            w_cov,
            v_cov,
        }
    }
}

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

impl<const NX: usize, const NY: usize, const NU: usize> ObserverModel<NX, NU, NX, NY, NU, NY>
    for GaussianLinearSystem<NX, NY, NU>
{
    fn state_model(
        &self,
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        w: &na::SVector<f64, NX>,
    ) -> na::SVector<f64, NX> {
        self.system.a_matrix * x + self.system.b_matrix * u + w
    }

    fn meas_model(
        &self,
        x: &na::SVector<f64, NX>,
        d: &na::SVector<f64, NU>,
        v: &na::SVector<f64, NY>,
    ) -> na::SVector<f64, NY> {
        self.system.c_matrix * x + self.system.d_matrix * d + v
    }

    fn gen_state_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, NX>> {
        let mut noise_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NX>::zeros(), &self.w_cov);
        let noise_samples: Vec<_> = (0..num_noise_samples).map(|_| noise_sampler()).collect();
        assert_eq!(noise_samples.len(), num_noise_samples);
        noise_samples
    }

    fn gen_meas_noise(&self, num_noise_samples: usize) -> Vec<na::SVector<f64, NY>> {
        let mut noise_sampler =
            multivariate_normal_sampler(&na::SVector::<f64, NY>::zeros(), &self.v_cov);
        let noise_samples: Vec<_> = (0..num_noise_samples).map(|_| noise_sampler()).collect();
        assert_eq!(noise_samples.len(), num_noise_samples);
        noise_samples
    }
}

impl<const NX: usize, const NY: usize, const NU: usize> NoiseNormalDistrubuted<NY, NX>
    for GaussianLinearSystem<NX, NY, NU>
{
    fn mean_state_noise(&self) -> na::SVector<f64, NX> {
        na::SVector::<f64, NX>::zeros()
    }
    fn cov_state_noise(&self) -> na::SMatrix<f64, NX, NX> {
        self.w_cov
    }

    fn mean_meas_noise(&self) -> na::SVector<f64, NY> {
        na::SVector::<f64, NY>::zeros()

    }
    fn cov_meas_noise(&self) -> na::SMatrix<f64, NY, NY> {
        self.v_cov
    }
}

impl<const NX: usize, const NY: usize, const NU: usize> Linear<NX, NU, NX, NY, NU, NY>
    for GaussianLinearSystem<NX, NY, NU>
{
    fn linear_state_model_dx(&self) -> na::SMatrix<f64, NX, NX> {
        self.system.a_matrix
    }

    fn linear_state_model_du(&self) -> na::SMatrix<f64, NX, NU> {
        self.system.b_matrix
    }

    fn linear_state_model_dw(&self) -> na::SMatrix<f64, NX, NX> {
        na::SMatrix::<f64, NX, NX>::identity()
    }

    fn linear_meas_model_dx(&self) -> na::SMatrix<f64, NY, NX> {
        self.system.c_matrix
    }

    fn linear_meas_model_dd(&self) -> na::SMatrix<f64, NY, NU> {
        self.system.d_matrix
    }

    fn linear_meas_model_dv(&self) -> na::SMatrix<f64, NY, NY> {
        na::SMatrix::<f64, NY, NY>::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulate_linear() {
        const NX: usize = 2;
        const NY: usize = 2;
        const NU: usize = 1;

        let a_matrix = na::SMatrix::<f64, NX, NX>::new_random();
        let b_matrix = na::SMatrix::<f64, NX, NU>::new_random();
        let c_matrix = na::SMatrix::<f64, NY, NX>::new_random();
        let d_matrix = na::SMatrix::<f64, NY, NU>::new_random();

        let system = LinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix);

        let x_init = na::SVector::<f64, NX>::new_random();
        let u = vec![na::SVector::<f64, NU>::new_random(); 100];

        let (x_series, y_series) = system.simulate(&x_init, &u, &u);

        assert_eq!(x_series.len(), u.len());
        assert_eq!(y_series.len(), u.len());
        assert_eq!(
            x_series[0], x_init,
            "Initial state is the first element of x_series"
        );

        let x_next = system.state_model(&x_init, &u[0], &na::SVector::<f64, 0>::zeros());
        let y = system.meas_model(&x_init, &u[0], &na::SVector::<f64, 0>::zeros());
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

        let (x_series, y_series) = system.simulate(&x_init, &u, &u);

        assert_eq!(x_series.len(), u.len());
        assert_eq!(y_series.len(), u.len());
        assert_eq!(
            x_series[0], x_init,
            "Initial state is the first element of x_series"
        );
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

        let (x_series_gaussian, y_series_gaussian) = system_guassian.simulate(&x_init, &u, &u);
        let (x_series_lin, y_series_lin) = system_linear.simulate(&x_init, &u, &u);

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
