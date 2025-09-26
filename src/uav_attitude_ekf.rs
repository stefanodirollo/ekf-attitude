use crate::quaternion_utils::{
    ddq_db, dh_dq, dq_from_gyro, inertial2body, q_left, q_mul, q_norm, q_right,
};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3, Vector4, vector};
//use nqlgebra::UnitQuaternion;

// Define custom types for fixed-size matrices and vectors
type Vector7_1 = SVector<f64, 7>;
type Matrix7_7 = SMatrix<f64, 7, 7>;
type Matrix3_7 = SMatrix<f64, 3, 7>;

/// EKF data struct
pub struct AttitudeEKF {
    /// state State vector with quaternion and gyro biases: [q0, q1, q2, q3, bx, by, bz]
    pub state: Vector7_1,
    /// covariance Covariance matrix P
    pub covariance: Matrix7_7,
    /// process_noise Process noise Q
    pub process_noise: Matrix7_7,
    /// measurement_noise Measurement noise R
    pub measurement_noise: Matrix3<f64>,
}

impl AttitudeEKF {
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
            let mut q0 = (0.5 * (1.0 + az)).sqrt();
            let mut q1 = -ay / (2.0 * q0);
            let mut q2 = ax / (2.0 * q0);
            let mut q3 = 0.0_f64;
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

        let mut measurement_noise = Matrix3::zeros();
        measurement_noise[(0, 0)] = 0.02; // ax
        measurement_noise[(1, 1)] = 0.02; // ay
        measurement_noise[(2, 2)] = 0.02; // az

        AttitudeEKF {
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
    ///   estimate from `self.state[4..=6]` internally.
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
        let ddq_db = ddq_db(omega_eff, dt); // 4x3
        let phi_qb = q_right(q_prev) * ddq_db; // 4x3
        let mut phi = Matrix7_7::identity();
        phi.fixed_view_mut::<4, 4>(0, 0).copy_from(&phi_qq); // top-left 4x4
        phi.fixed_view_mut::<4, 3>(0, 4).copy_from(&phi_qb); // top-right 4x3

        self.covariance = phi * self.covariance * phi.transpose() + self.process_noise;
    }

    pub fn update(&mut self, z_acc: Vector3<f64>, g_i: Vector3<f64>) {
        // 1) Predicted measurement h(x) = R(q) * g_i
        let q0 = self.state[0];
        let qv = Vector3::new(self.state[1], self.state[2], self.state[3]);
        let h = inertial2body(q0, &qv, &g_i);

        // 2) Jacobian H = [ ∂h/∂q (3x4) | 0 (3x3) ]
        let h_q = dh_dq(q0, &qv, &g_i);
        let mut H: Matrix3_7 = Matrix3_7::zeros();
        H.fixed_view_mut::<3, 4>(0, 0).copy_from(&h_q);

        // 3) Innovation
        let y = z_acc - h;

        // 4) Kalman gain K = P Hᵀ (H P Hᵀ + R)⁻¹
        let ht = H.transpose();
        let s = H * self.covariance * ht + self.measurement_noise; // 3x3
        // Prefer a robust inversion (Cholesky) for SPD S
        let s_inv = s.cholesky().map(|c| c.inverse()).unwrap_or_else(|| {
            s.try_inverse()
                .expect("Innovation covariance not invertible")
        });
        let k = self.covariance * ht * s_inv; // 7x3

        // 5) State update (additive on [q,b]), then re-normalize quaternion
        self.state += k * y;
        self.q_state_norm();

        // 6) Covariance update (Joseph form)
        let i = Matrix7_7::identity();
        let i_kh = i - k * H;
        self.covariance =
            i_kh * self.covariance * i_kh.transpose() + k * self.measurement_noise * k.transpose();
    }

    fn q_state_norm(&mut self) {
        let q = self.state.fixed_rows::<4>(0).into_owned();
        let q_normalized = q_norm(q);
        self.state.fixed_rows_mut::<4>(0).copy_from(&q_normalized);
    }
}
