extern crate nalgebra as na;

use crate::models::LinearSystem;

#[allow(non_snake_case)]
#[derive(Clone, Copy)]
pub struct LuenbergerObserver<const NX: usize, const NY: usize, const NU: usize> {
    // x_hat = A*x_hat + B*u + L*(y - C*x_hat - D*u)
    model: LinearSystem<NX, NY, NU>,
    L: na::SMatrix<f64, NX, NY>,
    x_hat: na::SVector<f64, NX>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> LuenbergerObserver<NX, NY, NU> {
    pub fn new(
        model: LinearSystem<NX, NY, NU>,
        L: na::SMatrix<f64, NX, NY>,
        x_hat: na::SVector<f64, NX>,
    ) -> Self {
        Self { model, L, x_hat }
    }
}

impl<const NX: usize, const NY: usize, const NU: usize>
    crate::observers::observer::Observer<NX, NY, NU> for LuenbergerObserver<NX, NY, NU>
{
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {
        // let temp = self.A * self.x_hat;
        let x_pred = self.model.A * self.x_hat + self.model.B * u;
        let y_est = self.model.C * self.x_hat + self.model.D * u;

        self.x_hat = x_pred + self.L * (y - y_est);
    }

    fn get_estimate(&self) -> na::SVector<f64, NX> {
        self.x_hat.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observers::observer::Observer;

    #[test]
    #[allow(non_snake_case)]
    fn it_runs() {
        let A = na::SMatrix::<f64, 2, 2>::new(1.0, 0.1, 0.0, 1.0);
        let B = na::SMatrix::<f64, 2, 1>::new(0.0, 0.1);
        let C = na::SMatrix::<f64, 1, 2>::new(1.0, 0.0);
        let D = na::SMatrix::<f64, 1, 1>::new(0.0);
        let model = LinearSystem::new(A, B, C, D);
        let L = na::SMatrix::<f64, 2, 1>::new(0.1, 0.1);
        let x_hat = na::SVector::<f64, 2>::new(0.0, 0.0);
        let mut observer = LuenbergerObserver::new(model, L, x_hat);

        let u = na::SVector::<f64, 1>::new(1.0);
        let y = na::SVector::<f64, 1>::new(1.0);
        observer.update(&u, &y);

        let _ = observer.get_estimate();
    }
}
