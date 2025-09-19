use nalgebra::{Matrix3, Matrix4, RowVector3, SMatrix, Vector3, Vector4};

type Matrix4_3 = SMatrix<f64, 4, 3>;

/// Quaternion product: q ⊗ p, scalar-first [w,x,y,z]
///
/// # Arguments
/// * `q` - First quaternion
/// * `p` - Second quaternion
///
/// # Returns
/// Multiplied quaternions
///
pub fn q_mul(q: Vector4<f64>, p: Vector4<f64>) -> Vector4<f64> {
    let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    let (pw, px, py, pz) = (p[0], p[1], p[2], p[3]);
    Vector4::new(
        qw * pw - qx * px - qy * py - qz * pz,
        qw * px + qx * pw + qy * pz - qz * py,
        qw * py - qx * pz + qy * pw + qz * px,
        qw * pz + qx * py - qy * px + qz * pw,
    )
}

/// Quaternion normalization
///
/// # Arguments
/// * `q` - Input quaternion
///
/// # Returns
/// Normalized quaternion
///
pub fn q_norm(q: Vector4<f64>) -> Vector4<f64> {
    let q_square_norm = q.dot(&q);
    if q_square_norm <= f64::EPSILON {
        return Vector4::new(1.0, 0.0, 0.0, 0.0);
    }
    q / q_square_norm.sqrt()
}

/// Left-multiplication matrix Q_L(q) such that q ⊗ r = Q_L(q) r
///
/// # Arguments
/// * `q` - Input quaternion
///
/// # Returns
/// Left-multiplication matrix Q_L(q)
///`
pub fn q_left(q: Vector4<f64>) -> Matrix4<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    Matrix4::from_row_slice(&[w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w])
}

/// Right-multiplication matrix Q_R(q) such that r ⊗ q = Q_R(q) r
///
/// # Arguments
/// * `q` - Input quaternion
///
/// # Returns
/// Right-multiplication matrix Q_R(q)
///`
pub fn q_right(q: Vector4<f64>) -> Matrix4<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    Matrix4::from_row_slice(&[w, -x, -y, -z, x, w, z, -y, y, -z, w, x, z, y, -x, w])
}

/// Skew-symmetric matrix [v×] so that [v×] x = v × x
///
/// # Arguments
/// * `v` - Input vector
///
/// # Returns
/// Skew-symmetric matrix [v×]
///`
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

/// Builds the **delta quaternion** for one prediction step from a body-rate
/// measurement, assuming the angular rate is (approximately) constant over `dt`.
///
/// Mathematically:
/// - Let **θ** = ω_eff · dt  (integrated angle, in rad)
/// - Let **φ** = ||θ||
/// - The delta quaternion is:  
///   δq = exp(½ · [0, θ]) = [ cos(φ/2),  (sin(φ/2)/φ) · θ ]
///
/// This function returns that unit quaternion in **scalar-first** order `[w, x, y, z]`.
/// A small-angle branch avoids division by zero and improves numerical stability,
/// ensuring the first-order consistency with Euler integration:
/// δq ≈ [ 1, ½ θ ]  +  O(dt²).
///
/// # Arguments
/// * `omega_eff` - Bias-corrected body angular rate `ω_m - b` in **rad/s**.
/// * `dt`        - Time step in **seconds**.
///
/// # Returns
/// * `Vector4<f64>` — The delta quaternion **δq = [w, x, y, z]** (approximately unit-norm).
///
/// # Panics
/// * This function does **not** panic.
///
/// # Complexity
/// * O(1), allocation-free.
///
/// # Notes
/// * Apply it to propagate attitude: `q_next = δq ⊗ q_prev`.
/// * For very small `φ`, it uses the series  
///   `cos(φ/2) ≈ 1 − φ²/8`, `sin(φ/2)/φ ≈ 1/2 − φ²/48`.
/// * If you change the small-angle threshold, keep it consistent with double precision.
///
/// # Examples
/// ```
/// use nalgebra::{Vector3, Vector4};
///
/// // zero rate -> identity delta quaternion
/// let dq = dq_from_gyro(Vector3::new(0.0, 0.0, 0.0), 0.01);
/// assert!((dq - Vector4::new(1.0, 0.0, 0.0, 0.0)).norm() < 1e-12);
///
/// // rotation about X: ω = 2 rad/s for dt = 0.5 s  ->  φ = 1 rad
/// let dq = dq_from_gyro(Vector3::new(2.0, 0.0, 0.0), 0.5);
/// let w = (0.5_f64).cos();               // cos(φ/2)
/// let x = (0.5_f64).sin();               // sin(φ/2) along X
/// assert!((dq - Vector4::new(w, x, 0.0, 0.0)).norm() < 1e-12);
/// ```
pub fn dq_from_gyro(omega_eff: Vector3<f64>, dt: f64) -> Vector4<f64> {
    let theta = omega_eff * dt;
    let phi = theta.norm();
    let half_phi = 0.5 * phi;

    if phi < 1e-9 {
        let phi2 = phi * phi;
        let c = 1.0 - phi2 / 8.0; // approximately cos(phi/2)
        let a = 0.5 - phi2 / 48.0; // approximately sin(phi/2) / phi
        let v = theta * a;
        Vector4::new(c, v[0], v[1], v[2])
    } else {
        let c = half_phi.cos();
        let s = half_phi.sin();
        let a = s / phi; // sin(phi/2) / phi
        let v = theta * a;
        Vector4::new(c, v[0], v[1], v[2])
    }
}

/// Computes the Jacobian **∂(δq)/∂b** of the delta–quaternion w.r.t. the gyro bias `b`,
/// evaluated at the bias–corrected angular rate `omega_eff = ω_m - b`.
///
/// Let  
///   θ = (ω_m − b)·Δt,  φ = ||θ||,  δq = exp(½·[0, θ]) = [ w; v ] .  
/// Then
///   ∂(δq)/∂b = [ ∂w/∂θ ; ∂v/∂θ ] · ∂θ/∂b  with  ∂θ/∂b = −Δt·I₃.
///
/// Closed-form derivatives used here (small-angle safe):
///   ∂w/∂θ = −(sin(φ/2)/(2φ)) · θᵀ       (row 1×3)  
///   ∂v/∂θ = a·I₃ + k·θθᵀ                (3×3)  
/// where a = sin(φ/2)/φ and k = ((½)cos(φ/2)φ − sin(φ/2))/φ³,  
/// with series for φ→0:  cos(φ/2)≈1−φ²/8,  a≈½−φ²/48,  k≈−1/24.
///
/// # Arguments
/// * `omega_eff` — Bias-corrected body rate (rad/s), i.e. `ω_m − b`.
/// * `dt`        — Time step Δt (s).
///
/// # Returns
/// * `4×3` matrix: Jacobian **∂(δq)/∂b**, rows ordered `[dw/db; dv/db]` and
///   quaternion in scalar-first order `[w, x, y, z]`.
///
/// # Notes
/// * At zero rate (θ=0) one gets ∂w/∂b = 0 and ∂v/∂b = −½·Δt·I₃.
/// * This Jacobian is the **top-right block** of the discrete transition Φ for
///   a full-state EKF with `x=[q,b]`:  Φ = [[Q_L(δq), Q_R(q_prev)·∂δq/∂b],[0,I]].
///
/// # Example
/// ```
/// use nalgebra::{SMatrix, Vector3};
/// // If you don't have the alias in your crate, define it for the test:
/// type Matrix3_3 = SMatrix<f64, 3, 3>;
///
/// // Zero rate: θ=0 ⇒ ∂δq/∂b = [0; -½·dt·I]
/// let dt = 0.01;
/// let j = d_deltaq_db(Vector3::new(0.0, 0.0, 0.0), dt);
/// let top = j.fixed_rows::<1>(0);
/// let bot = j.fixed_rows::<3>(1);
/// assert!(top.iter().all(|e| e.abs() < 1e-12));
/// let expect = -0.5 * dt * Matrix3_3::identity();
/// assert!((bot - expect).norm() < 1e-12);
/// ```
pub fn ddq_db(omega_eff: Vector3<f64>, dt: f64) -> Matrix4_3 {
    let theta = omega_eff * dt;
    let phi = theta.norm();

    // Scalars for small-angle-stable derivatives
    let (s, a, k) = if phi < 1e-9 {
        // series: c≈1−φ²/8, a≈½−φ²/48, k≈−1/24
        let phi2 = phi * phi;
        let a = 0.5 - phi2 / 48.0;
        let k = -1.0 / 24.0;
        (0.0, a, k)
    } else {
        let half_phi = 0.5 * phi;
        let c = half_phi.cos();
        let s = half_phi.sin();
        let a = s / phi; // sin(φ/2)/φ
        // k = ((1/2) c φ − s) / φ^3
        let k = ((0.5 * c * phi) - s) / (phi * phi * phi);
        (s, a, k)
    };

    // ∂w/∂θ = − (s / (2φ)) θᵀ  (row 1×3). For φ→0, s≈φ/2 ⇒ factor≈1/4.
    let mut dw_dtheta = RowVector3::zeros();
    if phi < 1e-9 {
        dw_dtheta.copy_from(&RowVector3::from_row_slice(&[
            -0.25 * theta[0],
            -0.25 * theta[1],
            -0.25 * theta[2],
        ]));
    } else {
        let factor = -(s / (2.0 * phi));
        dw_dtheta.copy_from(&RowVector3::from_row_slice(&[
            factor * theta[0],
            factor * theta[1],
            factor * theta[2],
        ]));
    }

    // ∂v/∂θ = a·I + k·θθᵀ  (3×3)
    let i3 = Matrix3::identity();
    let vv = theta * theta.transpose(); // θθᵀ
    let dv_dtheta = a * i3 + k * vv;

    // Chain rule: ∂δq/∂b = [ ∂w/∂θ ; ∂v/∂θ ] · (−Δt·I)
    let mut d = Matrix4_3::zeros();
    d.fixed_view_mut::<1, 3>(0, 0).copy_from(&(-dt * dw_dtheta));
    d.fixed_view_mut::<3, 3>(1, 0).copy_from(&(-dt * dv_dtheta));
    d
}
