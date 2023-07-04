import os
from os.path import basename, join, splitext

import geometric
import joblib as jl
import kinetic
import numpy as np
import torch
from kinetic import KineticFeatures
from pymo.data import Joint, MocapData
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.writers import *
from scipy import linalg
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import utils as feat_utils

# -------------------------------------------------------------------


def extract_geometric_features(positions, fps):
    """
    INPUT: (n_frames, n_joints, 3) array of joint positions relative
           to first frame root position

    OUTPUT: 32 dimensional array of geometric features for the whole sequence
    """

    return geometric.extract_manual_features(positions, fps=fps)


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


# SOURCE: https://github.com/mseitzer/pytorch-fid
def frechet_distance(feat1, feat2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(feat1.mean(axis=0))
    mu2 = np.atleast_1d(feat2.mean(axis=0))

    sigma1 = np.atleast_2d(np.cov(feat1, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(feat2, rowvar=False))

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------


def normalize(ground_truth, generated):
    mean = ground_truth.mean(axis=0)
    std = ground_truth.std(axis=0)

    return (ground_truth - mean) / (std + 1e-10), (generated - mean) / (std + 1e-10)


# -------------------------------------------------------------------


def compute_FID_scores(sequences_gen, sequences_gt, fps=60):
    """
    Compute and return the FID_k (kinetic FID) and the FID_g (geometric FID) metrics.

    Inputs:
        sequences_gen: a list of arrays with shape (n_frames, n_joints, 3) containing global joint positions
                       in the XYZ axis order. The number of frames may vary within the list.

        sequences_gt:  a list of arrays with shape (n_frames, n_joints, 3) containing global joint positions
                       in the XYZ axis order. The number of frames may vary within the list.

    Returns:
        FID_k, FID_g:  The two FID scores as a tuple.
    """
    # convert to relative offset w.r.t. root by subtracting the first joint of the first frame
    sequences_gen = [seq - seq[:1, :1, :] for seq in sequences_gen]
    sequences_gt = [seq - seq[:1, :1, :] for seq in sequences_gt]

    kinetic_features_gen = np.stack(
        [extract_kinetic_features(seq, fps=fps) for seq in tqdm(sequences_gen)]
    )
    kinetic_features_gt = np.stack(
        [extract_kinetic_features(seq, fps=fps) for seq in tqdm(sequences_gt)]
    )

    geometric_features_gen = np.stack(
        [extract_geometric_features(seq, fps=fps) for seq in tqdm(sequences_gen)]
    )

    geometric_features_gt = np.stack(
        [extract_geometric_features(seq, fps=fps) for seq in tqdm(sequences_gt)]
    )

    FID_k = frechet_distance(*normalize(kinetic_features_gt, kinetic_features_gen))
    FID_g = frechet_distance(*normalize(geometric_features_gt, geometric_features_gen))

    return FID_k, FID_g
