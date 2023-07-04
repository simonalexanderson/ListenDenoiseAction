import torch
import os
from os.path import basename, join, splitext

import joblib as jl
import kinetic as kinetic_features
import geometric_kth
import numpy as np
from kinetic import KineticFeatures
from pymo.data import Joint, MocapData
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.writers import *
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import utils as feat_utils

HAS_PRINTED_WINDOW_GENERATOR_WARNING = False
# -------------------------------------------------------------------


def extract_geometric_features(positions, fps):
    """
    INPUT: (n_frames, n_joints, 3) array of joint positions relative
           to first frame root position

    OUTPUT: (n_joints * 3) array of kinetic features for the whole sequence
    """

    return geometric_kth.extract_manual_features(positions, fps=fps)


# -------------------------------------------------------------------


def extract_kinetic_features(positions, fps):
    """
    INPUT: (n_frames, n_joints, 3) array of joint positions relative
           to first frame root position

    OUTPUT: (n_joints * 3) array of kinetic features for the whole sequence
    """
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions, frame_time=1.0 / fps, up_vec="y")
    kinetic_feature_vector = []
    for i in tqdm(range(positions.shape[1]), leave=False):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


# -------------------------------------------------------------------


def window_generator(positions, window_length, fps):
    """
    positions: (n_recordings, n_frames, n_joints, 3)
    """
    # TODO: we should figure out how to window the reference data for the FID computations.
    #       until then, we print a warning explaining the current settings!
    global HAS_PRINTED_WINDOW_GENERATOR_WARNING
    if not HAS_PRINTED_WINDOW_GENERATOR_WARNING:
        print("*" * 60)
        print(
            "WARNING: windowing uses stride=window_length (non-overlapping windows) which might be undesired!"
        )
        print("*" * 60)
        HAS_PRINTED_WINDOW_GENERATOR_WARNING = True

    assert fps == 60, "window_generator only supports 60 fps"

    for sequence in positions:
        window_idxs = torch.arange(0, len(sequence)).unfold(
            0, window_length, window_length
        )

        for idx in window_idxs:
            yield sequence[idx]


def extract_and_save_joint_positions(bvh_dir, wav_dir, fps, save_filename, filter=""):
    get_filename = lambda file_path: splitext(basename(file_path))[0]

    # Load all BVH files with a corresponding wav file
    bvh_files = [
        fname
        for fname in sorted(os.listdir(bvh_dir))
        if fname.endswith(".bvh") and filter in get_filename(fname)
    ]
    wav_files = [
        fname
        for fname in sorted(os.listdir(wav_dir))
        if fname.endswith(".wav") and filter in get_filename(fname)
    ]

    # Check if the BVH files can be paired with the wav files perfectly
    for (bvh, wav) in zip(bvh_files, wav_files):
        assert get_filename(bvh) == get_filename(wav)

    # Parse BVH files and transform them to joint positions
    parsed_files = []
    for bvh_file in tqdm(bvh_files, desc="Parsing BVH files"):
        parser = BVHParser()
        parsed_files.append(parser.parse(join(bvh_dir, bvh_file)))

    # fmt: off
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=fps, keep_all=False)),
        ('jtsel', JointSelector(["Spine", "Spine1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"], include_root=True)),
        ('pos', MocapParameterizer('position')), 
        ('npf', Numpyfier())
    ])
    # fmt: on

    # Save joint positions
    print("Extracting joint positions from the BVH files")
    positions = data_pipe.fit_transform(parsed_files)
    np.save(save_filename, positions)


def main(bvh_dir, wav_dir, fps, window_length, filter=""):
    print("_" * 60)
    print(f"{bvh_dir=}", f"{wav_dir=}", f"{window_length=}", f"{fps=}", sep="\n")
    print(f"FILTER ON: `{filter}`" if filter != "" else "FILTER OFF")
    print(chr(8254) * 60) # upperscore character

    save_filename = f"positions_{fps}fps.npy"
    if filter != "":
        save_filename = f"{filter}_{save_filename}"

    # Load joint positions
    if not os.path.isfile(save_filename):
        extract_and_save_joint_positions(bvh_dir, wav_dir, fps, save_filename, filter)
    else:
        print(f"Loading joint positions from {save_filename}")
    positions = np.load(save_filename, allow_pickle=True)

    # INPUT: `positions`: (n_seq, n_frames, n_joints, 3) array of GLOBAL joint positions
    positions = [seq.reshape(-1, 19, 3) for seq in positions]

    # convert to relative offset w.r.t. root by subtracting the first joint of the first frame
    positions = [seq - seq[:1, :1, :] for seq in positions]
    
    n_windows = len(list(window_generator(positions, window_length, fps)))
    kinetic_features = np.stack(
        [
            extract_kinetic_features(window, fps)
            for window in tqdm(
                window_generator(positions, window_length, fps),
                desc="Extracting kinetic features...",
                total=n_windows,
            )
        ]
    )

    geometric_features = np.stack(
        [
            extract_geometric_features(window, fps)
            for window in tqdm(
                window_generator(positions, window_length, fps),
                desc="Extracting geometric features...",
                total=n_windows,
            )
        ]
    )

    fname_prefix = f"{filter}_" if filter != "" else ""
    fname_suffix = f"_{window_length=}_{fps=}"

    print("Saving kinetic features in shape", kinetic_features.shape)
    np.save(f"{fname_prefix}kinetic_features{fname_suffix}.npy", kinetic_features)

    print("Saving geometric features in shape", geometric_features.shape)
    np.save(f"{fname_prefix}geometric_features{fname_suffix}.npy", geometric_features)


if __name__ == "__main__":
    for filter in ["kthstreet_gLH"]:
        main(
            fps=60,
            bvh_dir="/home/simonal/data/dance_source/bvh/",
            wav_dir="/home/simonal/data/dance_source/wav/",
            window_length=600,  # 10 second windows
            filter=filter,
        )
