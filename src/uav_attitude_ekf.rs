use crate::quaternion_utils::{dq_from_gyro, q_left, q_mul, q_norm, q_right, skew};
use nalgebra::{SMatrix, SVector, Vector3, Vector4, vector};
//use nqlgebra::UnitQuaternion;

// Define custom types for fixed-size matrices and vectors
type Vector7_1 = SVector<f64, 7>;
type Matrix7_7 = SMatrix<f64, 7, 7>;
type Matrix3_3 = SMatrix<f64, 3, 3>;
type Matrix3_7 = SMatrix<f64, 3, 7>;

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
            let norm_a = (acc_data[0].powi(2) + acc_data[1].powi(2) + acc_data[2].powi(2)).sqrt();
            let (ax, ay, az) = (
                acc_data[0] / norm_a,
                acc_data[1] / norm_a,
                acc_data[2] / norm_a,
            );

            // Calculate quaternion from accelerometer data
            let mut q0: f64 = (0.5 * (1.0 + az)).sqrt();
            let mut q1: f64 = -ay / (2.0 * q0);
            let mut q2: f64 = ax / (2.0 * q0);
            let mut q3: f64 = 0.0;
            let norm: f64 = (q0.powi(2) + q1.powi(2) + q2.powi(2) + q3.powi(2)).sqrt();
            q0 /= norm;
            q1 /= norm;
            q2 /= norm;
            q3 /= norm;
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
            state: vector!(q0, q1, q2, q3, 0.0, 0.0, 0.0),
            covariance: Matrix7_7::identity(),
            process_noise,
            measurement_noise,
        }
    }

    /// Runs the EKF **prediction step** for attitude (quaternion) and gyro bias using the quaternion exponential.
    ///
    /// # Arguments
    /// * `gyro` - Measured angular rate in the **body** frame `[rad/s]`. The method subtracts the current bias
    /// *          estimate from `self.state[4..=6]` internally.
    /// * `dt`   - Integration time step in **seconds**.
    ///
    /// # Returns
    /// Updates `self.state` and `self.covariance` **in place**. No value is returned
    ///
    pub fn predict(&mut self, gyro: Vector3<f64>, dt: f64) {
        // Unpack q and b
        let q_prev = Vector4::new(self.state[0], self.state[1], self.state[2], self.state[3]);
        let b_prev = Vector3::new(self.state[4], self.state[5], self.state[6]);
        // Bias-corrected angular rate
        let omega_eff = gyro - b_prev;
        // Delta quaternion (exact for constant omega on dt)
        let dq = dq_from_gyro(omega_eff, dt);
        // Propagate quaternion: q_k|k-1 = δq ⊗ q_{k-1}
        let q_pred = q_norm(q_mul(dq, q_prev));

        // Write back state (bias constant in prediction)
        self.state[0] = q_pred[0];
        self.state[1] = q_pred[1];
        self.state[2] = q_pred[2];
        self.state[3] = q_pred[3];
        // self.state[4..=6] are the same (b_prev)

        // ---- Covariance prediction: P = Φ P Φᵀ + Q
        // Build Φ = [[ Q_L(δq) ,  Q_R(q_prev) * ∂δq/∂b ],
        //            [ 0       ,  I3                      ]]
        let phi_qq = q_left(dq); // 4x4
        let d_dq_db = d_deltaq_db(omega_eff, dt); // 4x3
        let phi_qb = qr(q_prev) * d_dq_db; // 4x3
    }
}

/// Short description.
///
/// # Arguments
/// * `param1` - description
/// * `param2` - description
///
/// # Returns
/// return description.
///
/// # Example
/// ```
/// let res = crate::module::fn_name(param1, param2);
/// assert_eq!(res, expected);
/// ```
pub fn fn_name(param1: i32, param2: i32) -> i32 {
    param1 + param2
}
