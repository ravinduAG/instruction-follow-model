import numpy as np

class MovementAnalyzer:

    def __init__(self):
        self.angle_history = []

    def update(self, left_angle, right_angle):
        self.angle_history.append((left_angle, right_angle))

    def compute_features(self):

        if len(self.angle_history) < 1:
            return None

        angles = np.array(self.angle_history)

        avg_joint_angle_error = np.mean(np.abs(angles - 90))  # target angle example

        movement_smoothness = np.std(angles)

        reaction_time = len(self.angle_history) / 30  # assuming 30 FPS

        error_repetition_count = np.sum(
            np.abs(angles - 90) > 30
        )

        sequence_accuracy = 1 - (error_repetition_count / len(angles))

        instruction_delay = reaction_time * (1 - sequence_accuracy)

        return {
            "reaction_time": reaction_time,
            "sequence_accuracy": sequence_accuracy,
            "avg_joint_angle_error": avg_joint_angle_error,
            "movement_smoothness": movement_smoothness,
            "instruction_delay": instruction_delay,
            "error_repetition_count": int(error_repetition_count)
        }