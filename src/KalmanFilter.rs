extern crate nalgebra as na;

#[allow(non_snake_case)]
pub struct KalmanFilter<const NX: usize, const NY: usize, const NU: usize> {
    // Model
    A: na::SMatrix<f64, NX, NX>,
    B: na::SMatrix<f64, NX, NU>,
    C: na::SMatrix<f64, NY, NX>,
    D: na::SMatrix<f64, NY, NU>,

    // Uncertainty
    Q: na::SMatrix<f64, NX, NX>,
    R: na::SMatrix<f64, NY, NY>,

    // Kalman Filter state
    P_m: na::SMatrix<f64, NX, NX>,
    x_hat_m: na::SVector<f64, NX>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> KalmanFilter<NX, NY, NU> {
    pub fn new(
        A: na::SMatrix<f64, NX, NX>,
        B: na::SMatrix<f64, NX, NU>,
        C: na::SMatrix<f64, NY, NX>,
        D: na::SMatrix<f64, NY, NU>,
        Q: na::SMatrix<f64, NX, NX>,
        R: na::SMatrix<f64, NY, NY>,
        P_m_init: na::SMatrix<f64, NX, NX>,
        x_hat_init: na::SVector<f64, NX>,
    ) -> Self {
        Self {
            A,
            B,
            C,
            D,
            Q,
            R,
            P_m: P_m_init,
            x_hat_m: x_hat_init,
        }
    }
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> crate::Observer::Observer<NX, NY, NU>
    for KalmanFilter<NX, NY, NU>
{
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {
        // Prediction
        let x_hat_p = self.A * self.x_hat_m + self.B * u;
        let P_p = self.A * self.P_m * self.A.transpose() + self.Q;

        // Update
        let K = P_p * self.C.transpose() * (self.C * P_p * self.C.transpose() + self.R).try_inverse().unwrap();
        let y_pred = self.C * x_hat_p + self.D * u;
        self.x_hat_m = x_hat_p + K * (y - y_pred);
        self.P_m = (na::SMatrix::<f64, NX, NX>::identity() - K * self.C) * P_p;
    }

    fn get_estimate(&self) -> na::SVector<f64, NX> {
        self.x_hat_m.clone()
    }
}

