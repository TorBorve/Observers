extern crate nalgebra as na;

pub trait Model<const NX: usize, const NU: usize, const NV: usize, const NY: usize> {
    fn step(
        x: &na::SVector<f64, NX>,
        u: &na::SVector<f64, NU>,
        v: &na::SVector<f64, NV>,
    ) -> na::SVector<f64, NY>;
}

pub trait StateModel<const NX: usize, const NU: usize, const NV: usize>:
    Model<NX, NU, NV, NX>
{
}

pub trait Differentiable<const NY: usize, const NX: usize, const NV: usize> {
    fn dfdx(x: &na::SVector<f64, NX>, v: &na::SVector<f64, NV>) -> na::SMatrix<f64, NY, NX>;

    fn dfdv(x: &na::SVector<f64, NX>, v: &na::SVector<f64, NV>) -> na::SMatrix<f64, NY, NV>;
}

pub trait Linear<const NY: usize, const NX: usize, const NV: usize>:
    Differentiable<NY, NX, NV>
{
    fn dfdx() -> na::SMatrix<f64, NY, NX>;

    fn dfdv() -> na::SMatrix<f64, NY, NV>;
}

pub trait ObsModel<
    const NX: usize,
    const NU: usize,
    const NV: usize,
    const ND: usize,
    const NW: usize,
    const NY: usize,
>:
StateModel<NX, ND, NW> + Model<NX, NU, NV, NY>
{}



/// Linear Time-Invariant System
/// x(k+1) = A*x(k) + B*u(k)
/// y(k) = C*x(k) + D*u(k)
/// No noise is considered
#[allow(non_snake_case)]
#[derive(Copy, Clone)]
pub struct LinearSystem<const NX: usize, const NY: usize, const NU: usize> {
    pub A: na::SMatrix<f64, NX, NX>,
    pub B: na::SMatrix<f64, NX, NU>,
    pub C: na::SMatrix<f64, NY, NX>,
    pub D: na::SMatrix<f64, NY, NU>,
}

#[allow(non_snake_case)]
impl<const NX: usize, const NY: usize, const NU: usize> LinearSystem<NX, NY, NU> {
    pub fn new(
        A: na::SMatrix<f64, NX, NX>,
        B: na::SMatrix<f64, NX, NU>,
        C: na::SMatrix<f64, NY, NX>,
        D: na::SMatrix<f64, NY, NU>,
    ) -> Self {
        Self { A, B, C, D }
    }

    pub fn A(&self) -> &na::SMatrix<f64, NX, NX> {
        &self.A
    }
    pub fn B(&self) -> &na::SMatrix<f64, NX, NU> {
        &self.B
    }
    pub fn C(&self) -> &na::SMatrix<f64, NY, NX> {
        &self.C
    }
    pub fn D(&self) -> &na::SMatrix<f64, NY, NU> {
        &self.D
    }
}


// pub struct ObsModel<
//     T,
//     U,
//     const NX: usize,
//     const NU: usize,
//     const NV: usize,
//     const ND: usize,
//     const NW: usize,
//     const NY: usize,
// > where
//     T: StateModel<NX, ND, NW>,
//     U: Model<NX, NU, NV, NY>,
// {
//     pub state_model: T,
//     pub meas_model: U,
// }

// impl<
//         T,
//         U,
//         const NX: usize,
//         const NU: usize,
//         const NV: usize,
//         const ND: usize,
//         const NW: usize,
//         const NY: usize,
//     > ObsModel<T, U, NX, NU, NV, ND, NW, NY>
// where
//     T: StateModel<NX, ND, NW>,
//     U: Model<NX, NU, NV, NY>,
// {
//     pub fn new(state_model: T, meas_model: U) -> Self {
//         Self {
//             state_model,
//             meas_model,
//         }
//     }
// }

// pub struct LinearModel<const NX: usize, const NU: usize, const NY: usize> {
//     pub a_matrix: na::SMatrix<f64, NY, NX>,
//     pub b_matrix: na::SMatrix<f64, NY, NU>,
// }

// impl<const NX: usize, const NU: usize, const NY: usize> LinearModel<NX, NU, NY> {
//     pub fn new(a_matrix: na::SMatrix<f64, NY, NX>, b_matrix: na::SMatrix<f64, NY, NU>) -> Self {
//         Self { a_matrix, b_matrix }
//     }
// }

#[cfg(test)]
mod tests {

    #[test]
    fn test_structs() {}
}
