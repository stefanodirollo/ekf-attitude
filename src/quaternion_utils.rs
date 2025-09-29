use nalgebra::{Matrix3, Matrix4, RowVector3, SMatrix, Vector3, Vector4};

type Matrix4_3 = SMatrix<f64, 4, 3>;

/// Quaternion product: q ⊗ p, in **scalar-first** order `[w, x, y, z]`
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

/// Left-multiplication matrix $Q_L(\mathbf{q})$ such that
/// $\mathbf{r} \otimes \mathbf{q} = Q_L(\mathbf{q})\\mathbf{r}$
///
/// # Arguments
/// * `q` — Input quaternion
///
/// # Returns
/// * Right-multiplication matrix $Q_L(\mathbf{q})$
///
pub fn q_left(q: Vector4<f64>) -> Matrix4<f64> {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    Matrix4::from_row_slice(&[w, -x, -y, -z, x, w, -z, y, y, z, w, -x, z, -y, x, w])
}

/// Right-multiplication matrix $Q_R(\mathbf{q})$ such that
/// $\mathbf{r} \otimes \mathbf{q} = Q_R(\mathbf{q})\\mathbf{r}$
///
/// # Arguments
/// * `q` — Input quaternion
///
/// # Returns
/// * Right-multiplication matrix $Q_R(\mathbf{q})$
///
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
/// - Let $ \boldsymbol{\theta} = \boldsymbol{\omega}_{\text{eff}}\cdot dt $  (integrated angle vector, in rad)
/// - Let $ \varphi = \lVert \boldsymbol{\theta} \rVert $
/// - The delta quaternion is:  
///   $$ \delta q = \exp\Bigl(\tfrac{1}{2}[0, \boldsymbol{\theta}]\Bigr)
///      = \Bigl[ \cos\frac{\varphi}{2},
///                \frac{\sin(\varphi/2)}{\varphi}\boldsymbol{\theta} \Bigr] $$
///
/// This function returns that unit quaternion in **scalar-first** order $[w, x, y, z]$.
/// A small-angle branch avoids division by zero and improves numerical stability,
/// ensuring the first-order consistency with Euler integration:
///
/// $$ \delta q \approx \bigl[1,\tfrac{1}{2}\boldsymbol{\theta}\bigr] + O(dt^2). $$
///
/// # Arguments
/// * `omega_eff` — Bias-corrected body angular rate vector $ \boldsymbol{\omega}_m - \boldsymbol{b} $ in **rad/s**.
/// * `dt`        — Time step in **seconds**.
///
/// # Returns
/// * `Vector4<f64>` — The delta quaternion  
///   $ \delta q = [w, x, y, z] $ (approximately unit-norm).
///
/// # Notes
/// * Apply it to propagate attitude: $ q_{\text{next}} = \delta q \otimes q_{\text{prev}} $.
/// * For very small $ \varphi $, it uses the series expansions:  
///   $ \cos\bigl(\tfrac{\varphi}{2}\bigr) \approx 1 - \tfrac{\varphi^2}{8} $,  
///   $ \frac{\sin(\varphi/2)}{\varphi} \approx \tfrac{1}{2} - \tfrac{\varphi^2}{48} $.
/// * If you change the small-angle threshold, keep it consistent with double precision.
///
/// # Examples
/// ```
/// use nalgebra::{Vector3, Vector4};
/// use ekf_attitude::quaternion_utils::dq_from_gyro;
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
/// evaluated at the bias–corrected angular rate $\boldsymbol{\omega}_{\text{eff}} = \boldsymbol{\omega}_m - \boldsymbol{b}$.
///
/// Let
/// - $ \boldsymbol{\theta} = (\boldsymbol{\omega}_m - \boldsymbol{b}) \cdot \Delta t $
/// - $ \varphi = \lVert \boldsymbol{\theta} \rVert $
/// - $ \delta q = \exp\Bigl(\tfrac{1}{2}[0,\boldsymbol{\theta}]\Bigr) = [w, \boldsymbol{v}] $
///
/// Then:
///
/// $$
/// \frac{\partial \boldsymbol{\delta q}}{\partial \boldsymbol{b}} =
/// \begin{bmatrix}
///   \frac{\partial w}{\partial \boldsymbol{\theta}} \\\\
///   \frac{\partial \boldsymbol{v}}{\partial \boldsymbol{\theta}}
/// \end{bmatrix}
/// \cdot
/// \frac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{b}}
/// $$
///
/// with $ \tfrac{\partial \boldsymbol{\theta}}{\partial \boldsymbol{b}} = -\Delta t \cdot I_3 $
///
/// Closed-form derivatives used here (small-angle safe):
/// - $ \tfrac{\partial w}{\partial \boldsymbol{\theta}}
///     = -\dfrac{\sin(\varphi/2)}{2\varphi}\cdot \boldsymbol{\theta}^\top $  (row $1\times 3$)  
/// - $ \tfrac{\partial \boldsymbol{v}}{\partial \boldsymbol{\theta}}
///     = a\cdot I_3 + k\boldsymbol{\theta}\boldsymbol{\theta}^\top $  ($3\times 3$)
///
/// where $ a = \dfrac{\sin(\varphi/2)}{\varphi} $ and $ k = \dfrac{ \tfrac{1}{2}\cos(\varphi/2)\varphi - \sin(\varphi/2)}{\varphi^3} $
///
/// With series expansions for small $\varphi \to 0$:  
/// - $ \cos(\tfrac{\varphi}{2}) \approx 1 - \tfrac{\varphi^2}{8} $  
/// - $ a \approx \tfrac{1}{2} - \tfrac{\varphi^2}{48} $  
/// - $ k \approx -\tfrac{1}{24} $
///
/// # Arguments
/// * `omega_eff` — Bias-corrected body rate vector (rad/s), i.e. $ \boldsymbol{\omega}_m - \boldsymbol{b} $.
/// * `dt`        — Time step $\Delta t$ (s).
///
/// # Returns
/// * `4×3` matrix: Jacobian $ \tfrac{\partial \delta \boldsymbol{q}}{\partial \boldsymbol{b}} $, rows ordered $[\partial w/\partial b \partial \boldsymbol{v}/\partial b]$
///   with quaternion in scalar-first order $[w, x, y, z]$.
///
/// # Notes
/// * At zero rate ($\boldsymbol{\theta}=0$):  
///   - $ \tfrac{\partial w}{\partial \boldsymbol{b}} = 0 $  
///   - $ \tfrac{\partial \boldsymbol{v}}{\partial \boldsymbol{b}} = -\tfrac{1}{2}\Delta tI_3 $
/// * This Jacobian is the **top-right block** of the discrete transition $\Phi$ for
///   a full-state EKF with $ x = [\boldsymbol{q},\boldsymbol{b}] $:  
///   $$ \Phi =
///     \begin{bmatrix}
///         Q_L(\delta q) & Q_R(q_{\text{prev}}) \tfrac{\partial \delta \boldsymbol{q}}{\partial \boldsymbol{b}} \\\\
///         0 & I
///     \end{bmatrix}
///   $$
///
/// # Example
/// ```
/// use nalgebra::{Matrix3, Vector3};
/// use ekf_attitude::quaternion_utils::ddq_db;
///
/// // Zero rate: θ=0 ⇒ ∂δq/∂b = [0; -½·dt·I]
/// let dt = 0.01;
/// let j = ddq_db(Vector3::new(0.0, 0.0, 0.0), dt);
/// let top = j.fixed_rows::<1>(0);
/// let bot = j.fixed_rows::<3>(1);
/// assert!(top.iter().all(|e| e.abs() < 1e-12));
/// let expect = -0.5 * dt * Matrix3::identity();
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

/// Rotates a 3D vector from the inertial/world frame **into the body frame** using a
/// **unit quaternion** in scalar-first form, i.e. `q = [q0, qv.x, qv.y, qv.z]`.
///
/// # Math
/// Implements the closed-form “sandwich product” expansion
/// \[ R(q)\,u = (q0^2 - qv·qv)\,u + 2\,qv\,(qv·u) + 2\,q0\,(qv × u) \]
/// which is equivalent to ` [0, R(q)u] = q ⊗ [0, u] ⊗ q* ` for a unit quaternion.
/// This avoids building the 3×3 rotation matrix and uses only dot/cross products.
///
/// # Arguments
/// * `q0` — scalar part `w` of the quaternion.
/// * `qv` — vector part `(x, y, z)` of the quaternion.
/// * `vi` — vector expressed in the **inertial** frame to be rotated into the **body** frame.
///
/// # Returns
/// Vector expressed in the **body** frame: `vb = R(q) * vi`.
///
/// # Conventions
/// - This function assumes the quaternion performs the **active rotation** `R(q): inertial → body`,
///   i.e. `[0, vb] = q ⊗ [0, vi] ⊗ q*`.
/// - If in your code `q` encodes the opposite mapping (e.g. **body → inertial**), pass the
///   conjugate instead: call `inertial2body(q0, &(-qv), vi)`.
/// - `q` should be **unit-norm**. If not, the result is scaled by `‖q‖^2` (since `q*` is not `q⁻¹`
///   unless `‖q‖=1`). Normalize beforehand if needed.
///
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use std::f64::consts::FRAC_1_SQRT_2; // cos(pi/4) = sin(pi/4)
/// use ekf_attitude::quaternion_utils::inertial2body;
///
/// // 90° about +Z: q = [cos(π/4), 0, 0, sin(π/4)]
/// let q0 = FRAC_1_SQRT_2;
/// let qv = Vector3::new(0.0, 0.0, FRAC_1_SQRT_2);
/// let vi = Vector3::new(1.0, 0.0, 0.0); // x-axis in inertial
///
/// let vb = inertial2body(q0, &qv, &vi);
/// assert!((vb - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
/// ```
///
/// # Notes
/// - `O(1)` time, no heap allocations.
/// - For robustness during debugging you may assert near-unit norm of `q`.
pub fn inertial2body(q0: f64, qv: &Vector3<f64>, vi: &Vector3<f64>) -> Vector3<f64> {
    // Closed-form quaternion rotation (scalar-first):
    // R(q) * vi = (q0^2 - ||qv||^2) * vi + 2 qv (qv·vi) + 2 q0 (qv × vi)

    // Computation of (q0^2 - ||qv||^2):
    let w2_minus_v2 = q0 * q0 - qv.dot(qv);

    // qv·vi (scalar)
    let dot = qv.dot(vi);

    // qv × vi (vector)
    let cross = qv.cross(vi);

    // Assemble the expression:
    w2_minus_v2 * vi + 2.0 * (qv * dot) + 2.0 * (q0 * cross)
}

/// Computes the Jacobian of the acceleration with respect to the quaternion.
///
/// Given a quaternion represented as $\mathbf{q} = [w, \mathbf{q_v}]$ and a gravity vector $\mathbf{g} \in \mathbb{R}^3$
/// this function returns the Jacobian matrix:
///
/// $$
/// J \in \mathbb{R}^{3 \times 4}, \quad
/// J =
/// \begin{bmatrix}
///     \frac{\partial h}{\partial w}
///     \frac{\partial h}{\partial x}
///     \frac{\partial h}{\partial y}
///     \frac{\partial h}{\partial z}
/// \end{bmatrix}
/// $$
///
/// with the following intermediate terms:
///
/// $$
/// \frac{\partial h}{\partial \mathbf{q_v}} = 2 (\mathbf{q}_v \cdot \mathbf{g}) I_3
///     + 2 \mathbf{q_v} \mathbf{g}^T
///     - 2 \mathbf{g} \mathbf{q_v}^T
///     - 2 w \mathrm{skew}(\mathbf{g})
/// $$
///
/// $$
/// \frac{\partial h}{\partial w} = 2 q_0 \mathbf{g} + 2 (\mathbf{q_v} \times \mathbf{g})
/// $$
///
/// # Arguments
/// * `q0` - Scalar part of the quaternion.
/// * `qv` - Vector part of the quaternion ($\mathbf{q_v}$).
/// * `g` - Gravity vector ($\mathbf{g}$).
///
/// # Returns
/// * `SMatrix<f64, 3, 4>` - The Jacobian matrix
///
/// # Example
/// ```
/// use nalgebra::Vector3;
/// use ekf_attitude::quaternion_utils::{dh_dq};
/// use physical_constants::STANDARD_ACCELERATION_OF_GRAVITY;
///
/// let q0 = 1.0;
/// let qv = Vector3::new(0.0, 0.0, 0.0);
/// let g = Vector3::new(0.0, 0.0, -9.81);
/// let j = dh_dq(q0, &qv, &g);
/// assert_eq!(j.shape(), (3,4));
/// ```
pub fn dh_dq(q0: f64, qv: &Vector3<f64>, g: &Vector3<f64>) -> SMatrix<f64, 3, 4> {
    let dot = qv.dot(g);
    let i3 = Matrix3::identity();
    let j_qv = 2.0 * dot * i3 + 2.0 * (qv * g.transpose())
        - 2.0 * (g * qv.transpose())
        - 2.0 * q0 * skew(g);
    let j_w = 2.0 * q0 * g + 2.0 * (qv.cross(g));
    SMatrix::<f64, 3, 4>::from_columns(&[
        j_w,
        j_qv.column(0).into(),
        j_qv.column(1).into(),
        j_qv.column(2).into(),
    ])
}
