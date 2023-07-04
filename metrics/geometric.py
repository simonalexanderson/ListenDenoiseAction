# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Ruilong Li

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import utils as feat_utils

FPS = 60

print("TODO: verify that JOINT_NAMES is in the correct order <geometric.py>")
JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
]


def extract_manual_features(positions, fps=FPS):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = []
    f = ManualFeatures(positions, fps=fps)
    for _ in range(1, positions.shape[0]):
        pose_features = [
            f.f_nmove("Neck", "RightUpLeg", "LeftUpLeg", "RightHand", 1.8 * f.hl),
            f.f_nmove("Neck", "LeftUpLeg", "RightUpLeg", "LeftHand", 1.8 * f.hl),
            f.f_nplane("Spine1", "Neck", "Neck", "RightHand", 0.2 * f.hl),
            f.f_nplane("Spine1", "Neck", "Neck", "LeftHand", 0.2 * f.hl),
            f.f_move("Spine", "Spine1", "Spine1", "RightHand", 1.8 * f.hl),
            f.f_move("Spine", "Spine1", "Spine1", "LeftHand", 1.8 * f.hl),
            f.f_angle("RightArm", "RightShoulder", "RightArm", "RightHand", [0, 110]),
            f.f_angle("LeftArm", "LeftShoulder", "LeftArm", "LeftHand", [0, 110]),
            f.f_nplane(
                "LeftShoulder", "RightShoulder", "LeftHand", "RightHand", 2.5 * f.sw
            ),
            f.f_move("LeftHand", "RightHand", "RightHand", "LeftHand", 1.4 * f.hl),
            f.f_move("RightHand", "Hips", "LeftHand", "Hips", 1.4 * f.hl),
            f.f_move("LeftHand", "Hips", "RightHand", "Hips", 1.4 * f.hl),
            f.f_fast("RightHand", 2.5 * f.hl),
            f.f_fast("LeftHand", 2.5 * f.hl),
            f.f_plane("Hips", "LeftUpLeg", "LeftToeBase", "RightFoot", 0.38 * f.hl),
            f.f_plane("Hips", "RightUpLeg", "RightToeBase", "LeftFoot", 0.38 * f.hl),
            f.f_nplane("zero", "y_unit", "y_min", "RightFoot", 1.2 * f.hl),
            f.f_nplane("zero", "y_unit", "y_min", "LeftFoot", 1.2 * f.hl),
            f.f_nplane("LeftUpLeg", "RightUpLeg", "LeftFoot", "RightFoot", 2.1 * f.hw),
            f.f_angle("RightLeg", "RightUpLeg", "RightLeg", "RightFoot", [0, 110]),
            f.f_angle("LeftLeg", "LeftUpLeg", "LeftLeg", "LeftFoot", [0, 110]),
            f.f_fast("RightFoot", 2.5 * f.hl),
            f.f_fast("LeftFoot", 2.5 * f.hl),
            f.f_angle("Neck", "Hips", "RightShoulder", "RightArm", [25, 180]),
            f.f_angle("Neck", "Hips", "LeftShoulder", "LeftArm", [25, 180]),
            f.f_angle("Neck", "Hips", "RightUpLeg", "RightLeg", [50, 180]),
            f.f_angle("Neck", "Hips", "LeftUpLeg", "LeftLeg", [50, 180]),
            f.f_plane("RightFoot", "Neck", "LeftFoot", "Hips", 0.5 * f.hl),
            f.f_angle("Neck", "Hips", "zero", "y_unit", [70, 110]),
            f.f_nplane("zero", "minus_y_unit", "y_min", "RightHand", -1.2 * f.hl),
            f.f_nplane("zero", "minus_y_unit", "y_min", "LeftHand", -1.2 * f.hl),
            f.f_fast("Hips", 2.3 * f.hl),
        ]
        features.append(pose_features)
        f.next_frame()
    features = np.array(features, dtype=np.float32).mean(axis=0)
    return features


class ManualFeatures:
    def __init__(
        self,
        positions,
        fps,
        joint_names=JOINT_NAMES,
    ):
        self.positions = positions
        self.joint_names = joint_names
        self.frame_num = 1
        self.time_per_frame = 1.0 / fps
        # humerus length, should width, hip width
        self.hl, self.sw, self.hw = self._compute_reference_bone_lengths()

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        if j == "y_unit":
            return [0, 1, 0]
        elif j == "minus_y_unit":
            return [0, -1, 0]
        elif j == "zero":
            return [0, 0, 0]
        elif j == "y_min":
            return [
                0,
                min([y for (_, y, _) in self.positions[self.frame_num]]),
                0,
            ]
        elif j == "RightToeBase":
            # NOTE: The `positions` array does not contain a RightToeBase joint because we remove it during
            # data preprocessing due to its noisiness. Instead, the position of this joint is defined by a
            # fixed offset from the RightFoot joint.

            foot_pos = self.positions[self.frame_num][self.joint_names.index("RightFoot")]
            return foot_pos + self.joint_offsets["RightToeBase"]
        elif j == "LeftToeBase":
            # NOTE: The `positions` array does not contain a LeftToeBase joint because we remove it during
            # data preprocessing due to its noisiness. Instead, the position of this joint is defined by a
            # fixed offset from the LeftFoot joint.

            foot_pos = self.positions[self.frame_num][self.joint_names.index("LeftFoot")]
            return foot_pos + self.joint_offsets["LeftToeBase"]

        return self.positions[self.frame_num][self.joint_names.index(j)]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1][self.joint_names.index(j)]

    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold(
            j1, j1_prev, j2, j2_prev, j3, j3_prev, range, self.time_per_frame
        )

    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [
            self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]
        ]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold_normal(
            j1, j1_prev, j2, j3, j4, j4_prev, range, time_per_frame=self.time_per_frame
        )

    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane(j1, j2, j3, j4, threshold)

    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane_normal(j1, j2, j3, j4, threshold)

    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.angle_within_range(j1, j2, j3, j4, range)

    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return feat_utils.velocity_above_threshold(
            j1, j1_prev, threshold, time_per_frame=self.time_per_frame
        )

    def _compute_reference_bone_lengths(self):
        """
        Return the lengths of the three segments that are used as reference
        threshold values in the computation of geometric features. The three
        segments are, respectively, `humerus_length, shoulder_width, hip_width`.
        """

        # Joint offsets relative to parent joint, copied from the BVH files
        # NOTE: these are already in XYZ axis order
        self.joint_offsets = {
            "Hips": np.array([0.00000, 0.00000, 0.00000]),
            "Spine": np.array([21.01180, 87.28050, -105.43400]),
            "Spine1": np.array([0.00000, 7.77975, 0.00000]),
            "Neck": np.array([-0.00001, 22.65660, 0.00008]),
            "Head": np.array([-0.00002, 24.94720, 0.00000]),
            "LeftShoulder": np.array([0.00000, 14.70570, 1.89751]),
            "LeftArm": np.array([3.79250, 20.81930, -0.05065]),
            "LeftForeArm": np.array([12.48180, -0.00002, 0.00000]),
            "LeftHand": np.array([28.71410, 0.00005, 0.00063]),
            "RightShoulder": np.array([23.41480, 0.11658, 0.32115]),
            "RightArm": np.array([-3.79391, 20.81930, -0.05065]),
            "RightForeArm": np.array([-12.48180, 0.00000, 0.00000]),
            "RightHand": np.array([-28.71400, 0.00000, 0.00019]),
            "LeftUpLeg": np.array([-23.76070, 0.08180, 0.14467]),
            "LeftLeg": np.array([9.48750, 0.00000, 0.00000]),
            "LeftFoot": np.array([0.00003, -35.71610, -0.00021]),
            "LeftToeBase": np.array([0.27801, -40.38490, 0.04978]),
            "RightUpLeg": np.array([0.05770, -40.85830, 0.04628]),
            "RightLeg": np.array([-9.48750, 0.00000, 0.00000]),
            "RightFoot ": np.array([0.00001, -35.71600, 0.00087]),
            "RightToeBase": np.array([0.05770, -40.85830, 0.04628]),
        }

        r_shoulder_position = (
            self.joint_offsets["Spine"]
            + self.joint_offsets["Spine1"]
            + self.joint_offsets["RightShoulder"]
        )
        l_shoulder_position = (
            self.joint_offsets["Spine"]
            + self.joint_offsets["Spine1"]
            + self.joint_offsets["LeftShoulder"]
        )
        l_elbow_position = l_shoulder_position + self.joint_offsets["LeftArm"]

        humerus_length = feat_utils.distance_between_points(
            l_shoulder_position, l_elbow_position
        )

        shoulder_width = feat_utils.distance_between_points(
            l_shoulder_position, r_shoulder_position
        )

        hip_width = feat_utils.distance_between_points(
            # These joints' only parent is the root, which is at the origin,
            # so we can just take their offset as the position
            self.joint_offsets["LeftUpLeg"],
            self.joint_offsets["RightUpLeg"],
        )

        return humerus_length, shoulder_width, hip_width
