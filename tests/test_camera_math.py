import math

import pytest
import torch

from pytti.Transforms import (
    perspective_matrix,
    quaternion_matrix,
    translation_matrix,
)


def test_identity_quaternion():
    assert torch.allclose(quaternion_matrix(1, 0, 0, 0), torch.eye(4))


def test_quaternion_is_normalized():
    assert torch.allclose(quaternion_matrix(2, 0, 0, 0), torch.eye(4))


def test_zero_quaternion_raises():
    with pytest.raises(ValueError):
        quaternion_matrix(0, 0, 0, 0)


def test_quaternion_90_degrees_about_z():
    half = math.radians(90) / 2
    R = quaternion_matrix(math.cos(half), 0, 0, math.sin(half))
    x_axis = torch.tensor([1.0, 0.0, 0.0, 1.0])
    rotated = R @ x_axis
    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-6)


def test_translation_matrix():
    T = translation_matrix(1.0, 2.0, 3.0)
    p = torch.tensor([10.0, 20.0, 30.0, 1.0])
    assert torch.allclose(T @ p, torch.tensor([11.0, 22.0, 33.0, 1.0]))


def test_rotation_composes_after_translation():
    # zoom_3d builds R @ T: translate first, then rotate (column-vector
    # convention), matching the original glm implementation
    half = math.radians(90) / 2
    R = quaternion_matrix(math.cos(half), 0, 0, math.sin(half))
    T = translation_matrix(1.0, 0.0, 0.0)
    p = torch.tensor([0.0, 0.0, 0.0, 1.0])
    moved = (R @ T) @ p
    # origin -> translated to (1,0,0) -> rotated 90° about z -> (0,1,0)
    assert torch.allclose(moved, torch.tensor([0.0, 1.0, 0.0, 1.0]), atol=1e-6)


def test_perspective_projection_xy():
    # fov 90°, square aspect: g = 1, so x_ndc = x / -z
    P = perspective_matrix(math.radians(90), 1.0)
    point = torch.tensor([1.0, 0.5, -2.0, 1.0])
    clip = P @ point
    ndc = clip / clip[3]
    assert ndc[0].item() == pytest.approx(0.5)
    assert ndc[1].item() == pytest.approx(0.25)


def test_perspective_aspect_scales_x_only():
    P = perspective_matrix(math.radians(90), 2.0)
    point = torch.tensor([1.0, 1.0, -1.0, 1.0])
    ndc = (P @ point) / (P @ point)[3]
    assert ndc[0].item() == pytest.approx(0.5)
    assert ndc[1].item() == pytest.approx(1.0)
