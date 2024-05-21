

extern crate nalgebra as na;

#[allow(non_snake_case)]
/// Determine if closed loop system is stable.
/// A_clp: closed loop system. Assumed to be discrete
fn is_stable(A_clp: &na::DMatrix<f64>) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linsystheory::is_stable;

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
}
