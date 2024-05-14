extern crate nalgebra as na;

pub trait Observer<const NX: usize, const NY: usize, const NU: usize> {
    fn update(&mut self, u: &na::SVector<f64, NU>, y: &na::SVector<f64, NY>);
    fn get_estimate(&self) -> na::SVector<f64, NX>;
}
