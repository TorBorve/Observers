use debug_print::debug_println;

extern crate nalgebra as na;

#[allow(non_snake_case)]
/// Determine if closed loop system is stable.
/// A_clp: closed loop system. Assumed to be discrete
pub fn is_stable(A_clp: &na::DMatrix<f64>) -> bool {
    // Convert to complex matrix to solve eigenvalues
    let A_clp_complex = A_clp.map(|x| na::Complex::new(x, 0.0));
    match A_clp_complex.eigenvalues() {
        Some(eigenvalues) => {
            for eigenvalue in eigenvalues.iter() {
                use na::ComplexField;
                if eigenvalue.abs() >= 1.0 {
                    return false;
                }
            }
        }
        None => {
            todo!("Handle not solving eigenvalue");
        }
    }
    true
}

#[allow(non_snake_case)]
/// Determine if A and C is detectable
/// See PHB-test: https://en.wikipedia.org/wiki/Hautus_lemma
pub fn is_detectable(A: &na::DMatrix<f64>, C: &na::DMatrix<f64>) -> bool {
    phb_test_fulfilled(A, C, true).unwrap()
}

#[allow(non_snake_case)]
pub fn is_stabilizable(A: &na::DMatrix<f64>, B: &na::DMatrix<f64>) -> bool {
    phb_test_fulfilled(A, B, false).unwrap()
}

#[allow(non_snake_case)]
fn phb_test_fulfilled(
    A: &na::DMatrix<f64>,
    L: &na::DMatrix<f64>,
    vertical_stack: bool,
) -> Option<bool> {
    assert!(A.is_square());
    if vertical_stack {
        assert!(A.ncols() == L.ncols());
    } else {
        assert!(A.nrows() == L.nrows());
    }

    let A_complex = A.map(|x| na::Complex::new(x, 0.0));
    match A_complex.eigenvalues() {
        Some(eigenvalues) => {
            for eigenvalue in eigenvalues.iter() {
                use na::ComplexField;
                if eigenvalue.abs() >= 1.0 {
                    // check that rank [A - eigenvalue*I; C]  = n
                    let n = A.nrows();
                    let A_lambda_block = na::DMatrix::from_fn(n, n, |i, j| {
                        if i == j {
                            A[(i, j)] - eigenvalue.re
                        } else {
                            A[(i, j)]
                        }
                    });
                    let phb_matrix = match vertical_stack {
                        true => na::DMatrix::from_fn(n + L.nrows(), n, |i, j| {
                            if i < n {
                                A_lambda_block[(i, j)]
                            } else {
                                L[(i - n, j)]
                            }
                        }),
                        false => na::DMatrix::from_fn(n, n + L.ncols(), |i, j| {
                            if j < n {
                                A_lambda_block[(i, j)]
                            } else {
                                L[(i, j - n)]
                            }
                        }),
                    };

                    let eps = 1.0e-5; // TODO: Choose good epsilon
                    if phb_matrix.rank(eps) != n {
                        return Some(false);
                    }
                }
            }
        }
        None => {
            todo!("Could not get eigen values")
        }
    }
    Some(true)
}

#[allow(non_snake_case)]
/// Solve the discrete algebraic Riccati equation
/// P = A^TPA - A^TPB(R+B^TPB)^-1B^TPA + Q
pub fn dare(
    A: &na::DMatrix<f64>,
    B: &na::DMatrix<f64>,
    Q: &na::DMatrix<f64>,
    R: &na::DMatrix<f64>,
) -> Option<na::DMatrix<f64>> {
    // Check conditions before solving
    assert!(A.is_square());
    assert!(Q.is_square());
    assert!(R.is_square());
    assert_eq!(Q.ncols(), A.ncols());
    assert_eq!(B.nrows(), A.nrows());
    assert_eq!(B.ncols(), R.ncols());

    debug_println!("Q^1/2");
    let G = Q.clone().cholesky()?.l(); // Q must be Positive Semi-Definite
    debug_println!("Check deteca..");
    if !is_detectable(A, &G) || !is_stabilizable(A, B) {
        println!("Not detectable or stabilizable");
        return None;
    }

    debug_println!("R cholesky try");
    R.clone().cholesky()?;
    let R_inv = R.clone().try_inverse()?;
    let A_inv = A.clone().try_inverse()?;
    let S = B * &R_inv * B.transpose();

    let z_11 = A + &S * A_inv.transpose() * Q;
    let z_12 = -S * A_inv.transpose();
    let z_21 = -A_inv.transpose() * Q;
    let z_22 = A_inv.transpose();

    let mut Z = na::DMatrix::zeros(z_11.nrows() + z_22.nrows(), z_11.ncols() + z_12.ncols());
    Z.view_mut((0, 0), z_11.shape()).copy_from(&z_11);
    Z.view_mut((0, z_11.ncols()), z_12.shape()).copy_from(&z_12);
    Z.view_mut((z_11.nrows(), 0), z_21.shape()).copy_from(&z_21);
    Z.view_mut(z_11.shape(), z_22.shape()).copy_from(&z_22);

    // Use Schur decomposition to solve the Riccati equation
    // https://math.stackexchange.com/questions/3119575/how-can-i-solve-the-discrete-algebraic-riccati-equations
    // At the time of writing this the nalgebra::Schur method does not sort the eigenvalues as is nescessary. Therefore, LAPACK is used.
    // See: https://netlib.org/lapack/explore-html/d5/d38/group__gees_gab48df0b5c60d7961190d868087f485bc.html#gab48df0b5c60d7961190d868087f485bc

    extern "C" fn select_stable_eigen_fn(real: *const f64, _imag: *const f64) -> i32 {
        unsafe { (*real < 1.0) as i32 }
    }

    let select_stable_eigen: lapack::Select2F64 = Some(select_stable_eigen_fn);
    let n = Z.ncols();
    let mut a: Vec<f64> = (*Z.clone().data.as_vec()).clone();
    let mut sdim = 0; // out variable
    let mut wr = vec![0.0; n];
    let mut wi = vec![0.0; n];
    let mut vs = vec![0.0; n * n];
    let mut bwork = vec![0; n];
    let mut info = 0;

    // First query to find optimal work size.
    let mut schur_lapack_fn = |work_ref: &mut [f64], lwork: i32| unsafe {
        lapack::dgees(
            b'V',
            b'S',
            select_stable_eigen,
            n as i32,
            &mut a,
            n as i32,
            &mut sdim,
            &mut wr,
            &mut wi,
            &mut vs,
            n as i32,
            work_ref,
            lwork,
            &mut bwork,
            &mut info,
        );
        assert_eq!(info, 0);
    };
    let mut work = vec![0.0; 1];
    let mut lwork = -1; // set to -1 to first get the optimal work size.
    schur_lapack_fn(&mut work, lwork);

    lwork = work[0] as i32; // optimal work size stored in work[0]
    work = vec![0.0; lwork as usize];

    schur_lapack_fn(&mut work, lwork);
    std::mem::drop(schur_lapack_fn); // drop borrowed variables

    // A = Z*T*(Z^T)
    // a = T
    // vs = Z
    // let t_mat = na::DMatrix::from_vec(n, n, a);
    let z_mat = na::DMatrix::from_vec(n, n, vs);
    let U = z_mat.clone();

    let u_11 = U.view((0, 0), z_11.shape());
    let u_21 = U.view((z_11.nrows(), 0), z_21.shape());

    // P = U_21 * U_11^-1;
    let P = u_21 * u_11.try_inverse()?;

    assert!(P.clone().cholesky().is_some(), "P should be PSD");
    approx::assert_relative_eq!(
        P,
        A.transpose() * &P * A
            - A.transpose()
                * &P
                * B
                * ((R + B.transpose() * &P * B).try_inverse()?)
                * B.transpose()
                * &P
                * A
            + Q,
        epsilon = 1.0e-4
    );
    Some(P)
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;
    use crate::linsystheory::is_stable;

    #[allow(non_snake_case)]
    fn dare_bruteforce(
        A: &na::DMatrix<f64>,
        B: &na::DMatrix<f64>,
        Q: &na::DMatrix<f64>,
        R: &na::DMatrix<f64>,
        niter: u32,
    ) -> Option<na::DMatrix<f64>> {
        // P_(k+1) = A^TPA - A^TPB(R+B^TPB)^-1B^TPA + Q
        let mut P = Q.clone();
        for _ in 0..niter {
            P = A.transpose() * P.clone() * A
                - A.transpose()
                    * P.clone()
                    * B
                    * ((R + B.transpose() * P.clone() * B).try_inverse()?)
                    * B.transpose()
                    * P.clone()
                    * A
                + Q;
        }
        assert!(P.clone().cholesky().is_some(), "P should be PSD");
        approx::assert_relative_eq!(
            P,
            A.transpose() * &P * A
                - A.transpose()
                    * &P
                    * B
                    * ((R + B.transpose() * &P * B).try_inverse()?)
                    * B.transpose()
                    * &P
                    * A
                + Q,
            epsilon = 1.0e-4
        );
        Some(P)
    }

    #[test]
    fn test_is_stable() {
        println!("Test is stable");
        // let mut a_matrix = na::SMatrix::<f64, 2, 2>::new(0.0, 1,0, -1.0, 0.0);
        let mut m = na::DMatrix::from_vec(2, 2, vec![0.0, 1.0, -1.0, 0.0]);
        assert_eq!(is_stable(&m), false, "Undamped system is not stable");
        m = na::DMatrix::from_vec(2, 2, vec![0.0, 0.9, -0.9, 0.0]);
        assert_eq!(is_stable(&m), true, "Damped system is stable");

        m = na::DMatrix::from_vec(2, 2, vec![1.2, 0.0, 0.0, 1.0]);
        assert_eq!(is_stable(&m), false, "Unstable system is not stable");

        m = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        assert_eq!(is_stable(&m), true, "Stable system is stable");
    }

    #[test]
    fn test_detectability() {
        println!("Test detectability");

        let a_matrix = na::DMatrix::new_random(5, 5);
        let c_matrix = na::DMatrix::identity(5, 5);
        assert_eq!(
            is_detectable(&a_matrix, &c_matrix),
            true,
            "Detectable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![1.2, 0.0, 0.0, 1.2]);
        let c_matrix = na::DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        assert_eq!(
            is_detectable(&a_matrix, &c_matrix),
            false,
            "Undetectable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let c_matrix = na::DMatrix::from_vec(1, 2, vec![0.0, 1.0]);
        assert_eq!(
            is_detectable(&a_matrix, &c_matrix),
            true,
            "Detectable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 2.0]);
        let c_matrix = na::DMatrix::from_vec(1, 2, vec![1.0, 0.0]);
        assert_eq!(
            is_detectable(&a_matrix, &c_matrix),
            false,
            "Undetectable system"
        );
    }

    #[test]
    fn test_stabilizability() {
        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let b_matrix = na::DMatrix::from_vec(2, 1, vec![0.0, 0.0]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            true,
            "Stabilizable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![1.2, 0.0, 0.0, 1.2]);
        let b_matrix = na::DMatrix::from_vec(2, 1, vec![0.0, 0.0]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            false,
            "Unstabilizable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 2.0]);
        let b_matrix = na::DMatrix::from_vec(2, 1, vec![0.0, 1.0]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            true,
            "Stabilizable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![3.0, 0.0, 0.0, 2.0]);
        let b_matrix = na::DMatrix::from_vec(2, 1, vec![0.0, 1.0]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            false,
            "Unstabilizable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let b_matrix = na::DMatrix::from_vec(2, 1, vec![2.0, 1.0]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            false,
            "Unstabilizable system"
        );

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.7]);
        let b_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 5.3]);
        assert_eq!(
            is_stabilizable(&a_matrix, &b_matrix),
            true,
            "Stabilizable system"
        );
    }

    #[test]
    fn test_dare() {
        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let b_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let q_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let r_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let p_matrix_opt = dare(&a_matrix, &b_matrix, &q_matrix, &r_matrix);
        assert!(p_matrix_opt.is_some());
        let p_matrix = p_matrix_opt.unwrap();

        let p_matrix_iter_opt = dare_bruteforce(&a_matrix, &b_matrix, &q_matrix, &r_matrix, 1000);
        assert!(p_matrix_iter_opt.is_some());
        let p_matrix_iter = p_matrix_iter_opt.unwrap();

        approx::assert_relative_eq!(p_matrix, p_matrix_iter, epsilon = 1e-4);

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.0, 0.0, 0.9]);
        let b_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let q_matrix = na::DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
        let r_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let p_matrix_opt = dare(&a_matrix, &b_matrix, &q_matrix, &r_matrix);
        assert!(p_matrix_opt.is_none());

        let a_matrix = na::DMatrix::from_vec(2, 2, vec![10.23, 0.0, 0.0, 0.9]);
        let b_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 0.0]);
        let q_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let r_matrix = na::DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
        let p_matrix_opt = dare(&a_matrix, &b_matrix, &q_matrix, &r_matrix);
        assert!(p_matrix_opt.is_some());
        let p_matrix = p_matrix_opt.unwrap();

        let p_matrix_iter_opt = dare_bruteforce(&a_matrix, &b_matrix, &q_matrix, &r_matrix, 10000);
        assert!(p_matrix_iter_opt.is_some());
        let p_matrix_iter = p_matrix_iter_opt.unwrap();

        approx::assert_relative_eq!(p_matrix, p_matrix_iter, epsilon = 1e-4);
    }
}
