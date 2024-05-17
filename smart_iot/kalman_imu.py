import numpy as np
from scipy.linalg import inv
from typing import Tuple

"""Class for ESP32 IMU sensor fusion using Kalman filter"""


class KalmanIMU:
    def __init__(self, Q: np.ndarray, R: np.ndarray, P: np.ndarray, dt: float):
        """
        Initialize the KalmanIMU object.

        Args:
            Q: Process noise covariance matrix.
            R: Measurement noise covariance matrix.
            P: Initial state covariance matrix.
            dt: Time step.
        """
        self.Q = Q
        self.R = R
        self.P = P
        self.dt = dt
        self.A = np.array(
            [
                [1, -dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, -dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, -dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
        self.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.I_M = np.eye(6)

    def predict(self, gyro: np.ndarray, accel: np.ndarray) -> None:
        """
        Perform the prediction step of the Kalman filter.

        Args:
            gyro: Gyroscope measurements.
            accel: Accelerometer measurements.
        """
        # State prediction
        x = self.A @ self.x
        # Covariance prediction
        P = self.A @ self.P @ self.A.T + self.Q
        # Update state and covariance
        self.x = x
        self.P = P

    def update(self, accel: np.ndarray) -> None:
        """
        Perform the update step of the Kalman filter.

        Args:
            accel: Accelerometer measurements.
        """
        # Kalman gain
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
        # State update
        x = self.x + K @ (accel - self.H @ self.x)
        # Covariance update
        P = (self.I_M - K @ self.H) @ self.P
        # Update state and covariance
        self.x = x
        self.P = P

    def get_orientation(self) -> np.ndarray:
        """
        Get the estimated orientation.

        Returns:
            Estimated orientation as a numpy array.
        """
        return self.x[0:3]

    def get_angular_velocity(self) -> np.ndarray:
        """
        Get the estimated angular velocity.

        Returns:
            Estimated angular velocity as a numpy array.
        """
        return self.x[3:6]

    def get_covariance(self) -> np.ndarray:
        """
        Get the current state covariance matrix.

        Returns:
            Current state covariance matrix as a numpy array.
        """
        return self.P

    def set_orientation(self, orientation: np.ndarray) -> None:
        """
        Set the orientation.

        Args:
            orientation: Orientation to set.
        """
        self.x[0:3] = orientation

    def set_angular_velocity(self, angular_velocity: np.ndarray) -> None:
        """
        Set the angular velocity.

        Args:
            angular_velocity: Angular velocity to set.
        """
        self.x[3:6] = angular_velocity

    def set_covariance(self, covariance: np.ndarray) -> None:
        """
        Set the state covariance matrix.

        Args:
            covariance: Covariance matrix to set.
        """
        self.P = covariance

    def set_dt(self, dt: float) -> None:
        """
        Set the time step.

        Args:
            dt: Time step to set.
        """
        self.dt = dt
        self.A = np.array(
            [
                [1, -dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, -dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, -dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )

    def set_Q(self, Q: np.ndarray) -> None:
        """
        Set the process noise covariance matrix.

        Args:
            Q: Process noise covariance matrix to set.
        """
        self.Q = Q

    def set_R(self, R: np.ndarray) -> None:
        """
        Set the measurement noise covariance matrix.

        Args:
            R: Measurement noise covariance matrix to set.
        """
        self.R = R

    def set_P(self, P: np.ndarray) -> None:
        """
        Set the state covariance matrix.

        Args:
            P: State covariance matrix to set.
        """
        self.P = P

    def get_Q(self) -> np.ndarray:
        """
        Get the process noise covariance matrix.

        Returns:
            Process noise covariance matrix as a numpy array.
        """
        return self.Q

    def get_R(self) -> np.ndarray:
        """
        Get the measurement noise covariance matrix.

        Returns:
            Measurement noise covariance matrix as a numpy array.
        """
        return self.R

    def get_P(self) -> np.ndarray:
        """
        Get the current state covariance matrix.

        Returns:
            Current state covariance matrix as a numpy array.
        """
        return self.P

    def get_dt(self) -> float:
        """
        Get the time step.

        Returns:
            Time step as a float.
        """
        return self.dt

    def get_A(self) -> np.ndarray:
        """
        Get the state transition matrix.

        Returns:
            State transition matrix as a numpy array.
        """
        return self.A

    def get_H(self) -> np.ndarray:
        """
        Get the measurement matrix.

        Returns:
            Measurement matrix as a numpy array.
        """
        return self.H

    def get_x(self) -> np.ndarray:
        """
        Get the current state vector.

        Returns:
            Current state vector as a numpy array.
        """
        return self.x

    def get_I(self) -> np.ndarray:
        """
        Get the identity matrix.

        Returns:
            Identity matrix as a numpy array.
        """
        return self.I_M

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current state.

        Returns:
            Tuple containing the current state vector and state covariance matrix.
        """
        return self.x, self.P

    def set_state(self, x: np.ndarray, P: np.ndarray) -> None:
        """
        Set the current state.

        Args:
            x: State vector to set.
            P: State covariance matrix to set.
        """
        self.x = x
        self.P = P
