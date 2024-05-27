extern crate nalgebra as na;

use std::usize;

use crate::linsystheory::dare;
use crate::models::GaussianLinearSystem;
use crate::observers::observer::Observer;

use super::luenberger::LuenbergerObserver;

#[allow(non_snake_case)]
#[derive(Copy, Clone)]
pub struct KalmanFilter<const NX: usize, const NY: usize, const NU: usize> {
    // Model
    model: GaussianLinearSystem<NX, NY, NU>,

    // Kalman Filter state
    P_m: na::SMatrix<f64, NX, NX>,
    x_hat_m: na::SVector<f64, NX>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> KalmanFilter<NX, NY, NU> {
    pub fn new(
        model: GaussianLinearSystem<NX, NY, NU>,
        P_m_init: na::SMatrix<f64, NX, NX>,
        x_hat_init: na::SVector<f64, NX>,
    ) -> Self {
        Self {
            model,
            P_m: P_m_init,
            x_hat_m: x_hat_init,
        }
    }
    /// Update the Kalman Filter state
    /// Returns new (x_hat_m, P_m)
    fn update_pure(
        A: &na::SMatrix<f64, NX, NX>,
        B: &na::SMatrix<f64, NX, NU>,
        C: &na::SMatrix<f64, NY, NX>,
        D: &na::SMatrix<f64, NY, NU>,
        W: &na::SMatrix<f64, NX, NX>,
        V: &na::SMatrix<f64, NY, NY>,
        x_hat_m: &na::SVector<f64, NX>,
        P_m: &na::SMatrix<f64, NX, NX>,
        u: &na::SVector<f64, NU>,
        y: &na::SVector<f64, NY>,
    ) -> (na::SVector<f64, NX>, na::SMatrix<f64, NX, NX>) {
        // Prediction
        let x_hat_p = A * x_hat_m + B * u;
        let P_p = A * P_m * A.transpose() + W;

        // Update
        let K = P_p * C.transpose() * (C * P_p * C.transpose() + V).try_inverse().unwrap();
        let y_pred = C * x_hat_p + D * u;
        let x_hat_m = x_hat_p + K * (y - y_pred);
        let P_m = (na::SMatrix::<f64, NX, NX>::identity() - K * C) * P_p;

        (x_hat_m, P_m)
    }

    pub fn get_covariance(&self) -> &na::SMatrix<f64, NX, NX> {
        &self.P_m
    }
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> Observer<NX, NY, NU>
    for KalmanFilter<NX, NY, NU>
{
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {
        // Call static method to update the state
        let (x_hat_m, P_m) = Self::update_pure(
            self.model.A(),
            self.model.B(),
            self.model.C(),
            self.model.D(),
            self.model.w_cov(),
            self.model.v_cov(),
            &self.x_hat_m,
            &self.P_m,
            u,
            y,
        );
        self.x_hat_m = x_hat_m;
        self.P_m = P_m;
    }

    fn get_estimate(&self) -> na::SVector<f64, NX> {
        self.x_hat_m.clone()
    }
}

pub fn steady_state_kalman_gain<const NX: usize, const NY: usize, const NU: usize>(
    model: &GaussianLinearSystem<NX, NY, NU>,
) -> Option<na::SMatrix<f64, NX, NY>> {
    // let a_iter = model.A()
    let mut a_matrix = na::DMatrix::zeros(NX, NX);
    a_matrix.copy_from(model.A());
    let mut c_matrix = na::DMatrix::zeros(NY, NX);
    c_matrix.copy_from(model.C());
    let mut q_matrix = na::DMatrix::zeros(NX, NX);
    q_matrix.copy_from(model.w_cov());
    let mut r_matrix = na::DMatrix::zeros(NY, NY);
    r_matrix.copy_from(model.v_cov());

    let p_matrix = dare(
        &a_matrix.transpose(),
        &c_matrix.transpose(),
        &q_matrix,
        &r_matrix,
    )?;
    let kalman_gain = &p_matrix
        * &c_matrix.transpose()
        * (&c_matrix * p_matrix * c_matrix.transpose() + r_matrix).try_inverse()?;

    assert_eq!(kalman_gain.ncols(), NY);
    assert_eq!(kalman_gain.nrows(), NX);
    let skalman_gain = na::SMatrix::<f64, NX, NY>::from_iterator(kalman_gain.iter().cloned());
    Some(skalman_gain)
}

pub fn steady_state_kalman_filter<const NX: usize, const NY: usize, const NU: usize>(
    model: &GaussianLinearSystem<NX, NY, NU>,
    x_hat_init: &na::SVector<f64, NX>,
) -> Option<LuenbergerObserver<NX, NY, NU>> {
    let gain_matrix = steady_state_kalman_gain(model)?;
    Some(LuenbergerObserver::new(
        model.system,
        gain_matrix,
        *x_hat_init,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GaussianLinearSystem;
    use crate::models::Model;

    #[test]
    fn kalman_filter() {
        let a_matrix = na::SMatrix::<f64, 1, 1>::new(0.9);
        let b_matrix = na::SMatrix::<f64, 1, 1>::new(1.0);
        let c_matrix = na::SMatrix::<f64, 1, 1>::new(1.0);
        let d_matrix = na::SMatrix::<f64, 1, 1>::new(0.0);

        let w_cov = na::SMatrix::<f64, 1, 1>::new(0.1);
        let v_cov = na::SMatrix::<f64, 1, 1>::new(0.1);
        let model = GaussianLinearSystem::new(a_matrix, b_matrix, c_matrix, d_matrix, w_cov, v_cov);

        let x_init = na::SVector::<f64, 1>::new(1.0);
        let x_hat_init = na::SVector::<f64, 1>::new(0.0);
        let p_m_init = na::SMatrix::<f64, 1, 1>::identity();

        let mut kalman_filter = KalmanFilter::new(model, p_m_init, x_hat_init);

        let mut x = x_init.clone();
        for _ in 0..1000 {
            let u = na::SVector::<f64, 1>::new(0.0);
            let (x_new, y) = model.simulate_step(&x, &u);
            x = x_new;
            kalman_filter.update(&u, &y);
        }

        let x_est = kalman_filter.get_estimate();
        let p_m = kalman_filter.get_covariance();
        let std_deav = p_m.trace().sqrt();

        assert!(std_deav > 0.0, "Covariance matrix is not positive definite");
        assert!(
            std_deav < v_cov.trace().sqrt(),
            "Covariance matrix is too large"
        );
        assert!(
            (x_est - x).norm() < 5.0 * std_deav,
            "Error too large for given standard deviation"
        );
    }

    #[test]
    fn steady_state_kalman() {
        const NX: usize = 2;
        const NU: usize = 1;
        const NY: usize = 2;

        let a_mat = na::SMatrix::<f64, NX, NX>::new(0.9, 0.3, 0.0, 1.2);
        let b_mat = na::SMatrix::<f64, NX, NU>::new(0.0, 1.);
        let c_mat = na::SMatrix::<f64, NY, NX>::identity();
        let d_mat = na::SMatrix::<f64, NY, NU>::zeros();

        let w_cov = na::SMatrix::<f64, NX, NX>::identity();
        let v_cov = na::SMatrix::<f64, NY, NY>::identity();

        let model = GaussianLinearSystem::new(a_mat, b_mat, c_mat, d_mat, w_cov, v_cov);

        let mut kalman = KalmanFilter::new(
            model,
            na::SMatrix::<f64, NX, NX>::identity(),
            na::SVector::<f64, NX>::zeros(),
        );

        let kalman_ss_opt = steady_state_kalman_filter(&model, &na::SVector::<f64, NX>::zeros());
        assert!(kalman_ss_opt.is_some());
        let mut kalman_ss = kalman_ss_opt.unwrap();

        let n_sim = 10000;
        let x0 = na::SVector::<f64, NX>::new(1., 1.);
        let u_seq = vec![na::SVector::<f64, NU>::zeros(); n_sim];
        let (_x_seq, y_seq) = model.simulate(&x0, &u_seq);

        for (y, u) in y_seq.iter().zip(u_seq.iter()) {
            kalman.update(u, y);
            kalman_ss.update(u, y);
        }

        let p_m = kalman.get_covariance().clone();
        let p_p = a_mat * p_m * a_mat.transpose() + w_cov;

        let kalman_gain = &p_p
            * &c_mat.transpose()
            * (&c_mat * p_p * c_mat.transpose() + v_cov)
                .try_inverse()
                .unwrap();

        let l_gain = kalman_ss.gain_matrix().clone();

        println!("Steady State Gain: {l_gain}");
        println!("Kalman gain after {n_sim} iterations: {kalman_gain}");
        approx::assert_relative_eq!(l_gain, kalman_gain, epsilon = 1e-4);
    }
}
