extern crate nalgebra as na;

#[allow(non_snake_case)]
pub struct LuenbergerObserver<const NX: usize, const NY: usize, const NU: usize> {
    // x_hat = A*x_hat + B*u + L*(y - C*x_hat - D*u)
    A: na::SMatrix<f64, NX, NX>,
    B: na::SMatrix<f64, NX, NU>,
    C: na::SMatrix<f64, NY, NX>,
    D: na::SMatrix<f64, NY, NU>,
    L: na::SMatrix<f64, NX, NY>,
    x_hat: na::SVector<f64, NX>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> LuenbergerObserver<NX, NY, NU> {
    pub fn new(
        A: na::SMatrix<f64, NX, NX>,
        B: na::SMatrix<f64, NX, NU>,
        C: na::SMatrix<f64, NY, NX>,
        D: na::SMatrix<f64, NY, NU>,
        L: na::SMatrix<f64, NX, NY>,
        x_hat: na::SVector<f64, NX>,
    ) -> Self {
        Self {
            A,
            B,
            C,
            D,
            L,
            x_hat,
        }
    }
}

impl<const NX: usize, const NY: usize, const NU: usize> crate::Observer::Observer<NX, NY, NU>
    for LuenbergerObserver<NX, NY, NU>
{
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {
        // let temp = self.A * self.x_hat;
        let x_pred = self.A * self.x_hat + self.B * u;
        let y_est = self.C * self.x_hat + self.D * u;

        self.x_hat = x_pred + self.L * (y - y_est);
    }

    fn get_estimate(&self) -> na::SVector<f64, NX> {
        self.x_hat.clone()
    }
}
