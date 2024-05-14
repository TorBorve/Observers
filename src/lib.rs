extern crate nalgebra as na;
extern crate rand;

mod KalmanFilter;
mod LuenbergerObserver;
mod Observer;

use crate::Observer::Observer as ObserverTrait;
// use crate::LuenbergerObserver::LuenbergerObserver as LuenbergerObserverStruct;
// use crate::KalmanFilter::KalmanFilter as KalmanFilterStruct;

use rand::prelude::*;

#[cfg(test)]
mod tests {
    use na::SMatrix;

    use self::KalmanFilter::KalmanFilter;

    use super::*;

    #[test]
    fn it_works() {
        let a_matrix = na::SMatrix::<f64, 1, 1>::new(0.9);
        let b_matrix = na::SMatrix::<f64, 1, 1>::new(1.0);
        let c_matrix = na::SMatrix::<f64, 1, 1>::new(1.0);
        let d_matrix = na::SMatrix::<f64, 1, 1>::new(0.0);

        let l_matrix = na::SMatrix::<f64, 1, 1>::new(0.1);

        let x_init = na::SVector::<f64, 1>::new(1.0);
        let x_hat_init = na::SVector::<f64, 1>::new(0.0);

        let mut observer = LuenbergerObserver::LuenbergerObserver::new(
            a_matrix, b_matrix, c_matrix, d_matrix, l_matrix, x_hat_init,
        );

        let q_matrix = na::SMatrix::<f64, 1, 1>::new(0.1);
        let r_matrix = na::SMatrix::<f64, 1, 1>::new(0.1);
        let mut kalman_filter = KalmanFilter::new(
            a_matrix,
            b_matrix,
            c_matrix,
            d_matrix,
            q_matrix,
            r_matrix,
            SMatrix::<f64, 1, 1>::identity(),
            x_hat_init,
        );

        // assert!(
        //     (a_matrix - l_matrix * b_matrix)[(0, 0)] < 1f64
        //         && (a_matrix - l_matrix * b_matrix)[(0, 0)] > -1f64
        // );

        let mut x = x_init.clone();
        let u = na::SVector::<f64, 1>::new(1.0);

        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let noise = SMatrix::<f64, 1, 1>::new(rng.gen_range(-0.1..0.1));

            let y = c_matrix * x + d_matrix * u + noise;
            x = a_matrix * x + b_matrix * u;
            observer.update(&u, &y);
            kalman_filter.update(&u, &y);
            let x_hat = observer.get_estimate();
            let x_hat_kalman = kalman_filter.get_estimate();
            println!(
                "x: {:?}, x_hat_L: {:?}, error_L: {:?}, x_hat_K: {:?}, error_K: {:?}",
                x,
                x_hat,
                (x - x_hat).abs(),
                x_hat_kalman,
                (x - x_hat_kalman).abs()
            );
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
