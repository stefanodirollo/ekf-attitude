use nalgebra::UnitQuaternion;
use nalgebra::{SMatrix, SVector};

// Define custom types for fixed-size matrices and vectors
type Vector4_1 = nalgebra::SVector<f64, 4>;
type Vector7_1 = nalgebra::SVector<f64, 7>;
type Matrix7_7 = nalgebra::SMatrix<f64, 7, 7>;
type Matrix3_3 = nalgebra::SMatrix<f64, 3, 3>;
type Matrix3_7 = nalgebra::SMatrix<f64, 3, 7>;

/// EKF data struct
pub struct EKF {
    /// state State vector with quaternion and gyro biases: [q0, q1, q2, q3, bx, by, bz]
    pub state: Vector7_1,
    /// covariance Covariance matrix P
    pub covariance: Matrix7_7,
    /// process_noise Process noise Q
    pub process_noise: Matrix7_7,
    /// measurement_noise Measurement noise R
    pub measurement_noise: Matrix3_3,
}

impl EKF {
    /// Create a new EKF instance, passing accelerometer data to calculate the initial quaternion
    /// (avoids using 0's for initial orientation)
    ///
    /// # Arguments
    /// * `acc_data` - Accelerometer data vector (optional)
    ///
    /// # Returns
    /// An instance of EKF with initialized quaternion
    ///
    pub fn new(acc_data: Option<[f64; 3]>) -> Self {
        let (q0, q1, q2, q3) = if let Some(acc_data) = acc_data {
            let norm = (acc_data[0].powi(2) + acc_data[1].powi(2) + acc_data[2].powi(2)).sqrt();
            let ax = acc_data[0] / norm;
            let ay = acc_data[1] / norm;
            let az = acc_data[2] / norm;
            // Calculate quaternion from accelerometer data
            let q0 = (1.0 + az).sqrt() / 2.0;
            let q1 = -ay / (2.0 * q0);
            let q2 = ax / (2.0 * q0);
            let q3 = 0.0;
            let norm = (q0.powi(2) + q1.powi(2) + q2.powi(2) + q3.powi(2)).sqrt();
            let q0 = q0 / norm;
            let q1 = q1 / norm;
            let q2 = q2 / norm;
            let q3 = q3 / norm;
            (q0, q1, q2, q3)
        } else {
            (1.0, 0.0, 0.0, 0.0) // Default to identity quaternion
        };

        // Initialize process and measurement noise matrices
        let mut process_noise = Matrix7_7::zeros();
        process_noise[(0, 0)] = 0.05; // q0
        process_noise[(1, 1)] = 0.05; // q1
        process_noise[(2, 2)] = 0.05; // q2
        process_noise[(3, 3)] = 0.05; // q3
        process_noise[(4, 4)] = 0.01; // bx
        process_noise[(5, 5)] = 0.01; // by
        process_noise[(6, 6)] = 0.01; // bz

        let mut measurement_noise = Matrix3_3::zeros();
        measurement_noise[(0, 0)] = 0.02; // ax
        measurement_noise[(1, 1)] = 0.02; // ay
        measurement_noise[(2, 2)] = 0.02; // az

        EKF {
            state: Vector7_1::new(q0, q1, q2, q3, 0.0, 0.0, 0.0),
            covariance: Matrix7_7::identity(),
            process_noise,
            measurement_noise,
        }
    }
}
