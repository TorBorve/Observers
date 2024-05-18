extern crate nalgebra as na;

use crate::models::GaussianLinearSystem;
use crate::observers::observer::Observer;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::GaussianLinearSystem;
    use crate::models::Model;

    #[test]
    fn it_works() {
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
        assert!(std_deav < v_cov.trace().sqrt(), "Covariance matrix is too large");
        assert!((x_est - x).norm() < 5.0*std_deav, "Error too large for given standard deviation");
    }
}
