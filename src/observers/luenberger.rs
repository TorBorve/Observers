extern crate nalgebra as na;

use crate::models::{Linear, LinearSystem, ObserverModel};


#[allow(non_snake_case)]
#[derive(Clone, Copy)]
pub struct LuenbergerObserver<T, const NX: usize, const NY: usize, const NU: usize>
where
T: ObserverModel<NX, NU, 0, NY, NU, 0> + Linear<NX, NU, 0, NY, NU, 0>
{
    // x_hat = A*x_hat + B*u + L*(y - C*x_hat - D*u)
    model: T,
    L: na::SMatrix<f64, NX, NY>,
    x_hat: na::SVector<f64, NX>,
}

#[allow(non_snake_case)]
impl<T: ObserverModel<NX, NU, 0, NY, NU, 0> + Linear<NX, NU, 0, NY, NU, 0> , const NX: usize, const NY: usize, const NU: usize> LuenbergerObserver<T, NX, NY, NU>
{
    pub fn new(
        model: T,
        L: na::SMatrix<f64, NX, NY>,
        x_hat: na::SVector<f64, NX>,
    ) -> Self {
        Self { model, L, x_hat }
    }

    pub fn gain_matrix(&self) -> &na::SMatrix<f64, NX, NY> {
        &self.L
    }
}

impl<T: ObserverModel<NX, NU, 0, NY, NU, 0> + Linear<NX, NU, 0, NY, NU, 0>, const NX: usize, const NY: usize, const NU: usize>
    crate::observers::observer::Observer<NX, NY, NU> for LuenbergerObserver<T, NX, NY, NU>
{
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>) {

        let x_pred = self.model.state_model(&self.x_hat, u, &na::SVector::<f64, 0>::zeros());
        let y_est = self.model.meas_model(&self.x_hat, u, &na::SVector::<f64, 0>::zeros());

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
