use ekf_attitude::uav_attitude_ekf::AttitudeEKF;
use nalgebra::Vector3;

#[test]
fn it_ekf() {
    let mut ekf = AttitudeEKF::new(None);

    // Example gyroscope data (roll rate, pitch rate, yaw rate in rad/s)
    let gyro_data = Vector3::new(0.01, -0.02, 0.03);
    let dt = 0.005;

    // Prediction phase
    ekf.predict(gyro_data, dt);

    // Example accelerometer data (x, y, z acceleration in m/s^2)
    let acc_data = Vector3::new(0.0, 9.81, 0.0);

    // Update phase
    ekf.update(acc_data);
}
