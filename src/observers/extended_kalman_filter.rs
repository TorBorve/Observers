use crate::models::{Differentiable, NoiseNormalDistrubuted, ObserverModel};

use super::observer::Observer;

extern crate nalgebra as na;

pub struct ExtendedKalmanFilter<T, const NX: usize, const NY: usize, const NU: usize>
where
    T: ObserverModel<NX, NU, NX, NY, NU, NY>
        + Differentiable<NX, NU, NX, NY, NU, NY>
        + NoiseNormalDistrubuted<NY, NX>,
{
    model: T,
    cov_est: na::SMatrix<f64, NX, NX>,
    x_est: na::SVector<f64, NX>,
}

impl<
        T: ObserverModel<NX, NU, NX, NY, NU, NY>
            + Differentiable<NX, NU, NX, NY, NU, NY>
            + NoiseNormalDistrubuted<NY, NX>,
        const NX: usize,
        const NY: usize,
        const NU: usize,
    > ExtendedKalmanFilter<T, NX, NY, NU>
{
    pub fn new(
        model: T,
        cov_est_init: na::SMatrix<f64, NX, NX>,
        x_est_init: na::SVector<f64, NX>,
    ) -> Self {
        Self {
            model,
            cov_est: cov_est_init,
            x_est: x_est_init,
        }
    }

    pub fn get_covariance(&self) -> &na::SMatrix<f64, NX, NX> {
        &self.cov_est
    }
}

impl<
        T: ObserverModel<NX, NU, NX, NY, NU, NY>
            + Differentiable<NX, NU, NX, NY, NU, NY>
            + NoiseNormalDistrubuted<NY, NX>,
        const NX: usize,
        const NY: usize,
        const NU: usize,
    > Observer<NX, NY, NU> for ExtendedKalmanFilter<T, NX, NY, NU>
{

    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {
        let a_mat = self.model.state_model_dx(&self.x_est, u, &na::SVector::<f64, NX>::zeros());
        let w_cov_scale = self.model.state_model_dw(&self.x_est, u, &na::SVector::<f64, NX>::zeros());
        let c_mat = self.model.meas_model_dx(&self.x_est, u, &na::SVector::<f64, NY>::zeros());
        let v_cov_scale = self.model.meas_model_dv(&self.x_est, u, &na::SVector::<f64, NY>::zeros());

        let w_cov_scaled = w_cov_scale * self.model.cov_state_noise() * w_cov_scale.transpose();
        let v_cov_scaled = v_cov_scale * self.model.cov_meas_noise() * v_cov_scale.transpose();
        // Avoid using by mistake later
        std::mem::drop(v_cov_scale);
        std::mem::drop(w_cov_scale);

        // Priori update
        let x_est_p = self.model.state_model(&self.x_est, u, &na::SVector::<f64, NX>::zeros());
        let cov_est_p = a_mat * self.cov_est * a_mat.transpose() + w_cov_scaled;

        // Measurment update
        let k_gain = cov_est_p * c_mat.transpose() * (c_mat * cov_est_p * c_mat.transpose() + v_cov_scaled).try_inverse().unwrap();
        let y_pred = self.model.meas_model(&x_est_p, u, &na::SVector::<f64, NY>::zeros());
        let x_est_m = x_est_p + k_gain * (y - y_pred);
        let cov_est_m = (na::SMatrix::<f64, NX, NX>::identity() - k_gain * c_mat) * cov_est_p;

        self.x_est = x_est_m;
        self.cov_est = cov_est_m;
    }

    fn get_estimate(&self) -> na::SVector<f64, NX> {
        self.x_est.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{models::GaussianLinearSystem, observers::kalman_filter::{self, KalmanFilter}};

    #[test]
    fn extended_kalman_filter_linear_model() {
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

        let mut extended_kalman_filter = ExtendedKalmanFilter::new(model, p_m_init, x_hat_init);
        let mut kalman_filter = KalmanFilter::new(model, p_m_init, x_hat_init);

        let mut x = x_init.clone();
        for _ in 0..1000 {
            let u = na::SVector::<f64, 1>::new(0.0);
            let x_new = model.state_model(&x, &u, &na::SVector::<f64, 1>::zeros());
            let y = model.meas_model(&x, &u, &na::SVector::<f64, 1>::zeros());

            x = x_new;
            kalman_filter.update(&u, &y);
            extended_kalman_filter.update(&u, &y);
        }

        approx::assert_relative_eq!(extended_kalman_filter.get_estimate(), kalman_filter.get_estimate());
        approx::assert_relative_eq!(extended_kalman_filter.get_covariance(), kalman_filter.get_covariance());

        let x_est = extended_kalman_filter.get_estimate();
        let p_m = extended_kalman_filter.get_covariance();
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


}