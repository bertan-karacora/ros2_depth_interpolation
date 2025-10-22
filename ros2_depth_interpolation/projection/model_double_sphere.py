import math

import torch

from ros2_depth_interpolation.projection.model_camera import ModelCamera


class ModelDoubleSphere(ModelCamera):
    """Implemented according to:
    V. Usenko, N. Demmel, and D. Cremers: The Double Sphere Camera Model.
    Proceedings of the International Conference on 3D Vision (3DV) (2018).
    URL: https://arxiv.org/pdf/1807.08957.pdf."""

    keys_params_distortion = ["xi", "alpha"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = self.params_distortion["alpha"]
        self.xi = self.params_distortion["xi"]

    @torch.inference_mode()
    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.half()

        coords_xyz = coords_xyz.to(self.device)

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        # Eq. (41)
        d1 = torch.sqrt(x**2 + y**2 + z**2)
        # Eq. (45)
        w1 = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        # Eq. (44)
        w2 = (w1 + self.xi) / math.sqrt(2.0 * w1 * self.xi + self.xi**2 + 1.0)
        # Eq. (43)
        mask_valid = z > -w2 * d1

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            x = x[mask_valid][None, ...]
            y = y[mask_valid][None, ...]
            z = z[mask_valid][None, ...]
            d1 = d1[mask_valid][None, ...]
            mask_valid = torch.ones_like(z, dtype=torch.bool)

        # Eq. (42)
        z_shifted = self.xi * d1 + z
        d2 = torch.sqrt(x**2 + y**2 + z_shifted**2)
        # Eq. (40)
        denominator = self.alpha * d2 + (1 - self.alpha) * z_shifted
        u = self.fx * x / denominator + self.cx
        v = self.fy * y / denominator + self.cy
        coords_uv = torch.stack((u, v), dim=1)

        if use_mask_fov:
            mask_left = coords_uv[:, 0, :] >= 0
            mask_top = coords_uv[:, 1, :] >= 0
            mask_right = coords_uv[:, 0, :] < self.shape_image[2]
            mask_bottom = coords_uv[:, 1, :] < self.shape_image[1]
            mask_valid *= mask_left * mask_top * mask_right * mask_bottom

        return coords_uv, mask_valid

    @torch.inference_mode()
    def project_image_onto_points(self, coords_uv, use_invalid_coords=True, use_half_precision=True):
        """Project 2D image onto 3D unit sphere.
        Shape of coords_uv: (B, 2, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_uv = coords_uv.half()

        coords_uv = coords_uv.to(self.device)

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        # Eq. (47)
        mx = (u - self.cx) / self.fx
        # Eq. (48)
        my = (v - self.cy) / self.fy
        # Eq. (49)
        square_r = mx**2 + my**2
        # Eq. (51) can be written to use this
        term = 1.0 - (2.0 * self.alpha - 1.0) * square_r
        # Eq. (51)
        mask_valid = term >= 0.0 if self.alpha > 0.5 else torch.ones_like(term, dtype=torch.bool)

        # Note: Only working for batchsize 1
        if not use_invalid_coords and mask_valid.shape[0] == 1:
            mx = mx[mask_valid][None, ...]
            my = my[mask_valid][None, ...]
            square_r = square_r[mask_valid][None, ...]
            term = term[mask_valid][None, ...]
            mask_valid = torch.ones_like(term, dtype=torch.bool)

        # Eq. (50)
        mz = (1.0 - self.alpha**2 * square_r) / (self.alpha * torch.sqrt(term) + 1.0 - self.alpha)
        # Eq. (46)
        factor = (mz * self.xi + torch.sqrt(mz**2 + (1.0 - self.xi**2) * square_r)) / (mz**2 + square_r)
        coords_xyz = factor[:, None, :] * torch.stack((mx, my, mz), dim=1)
        coords_xyz[:, 2, :] -= self.xi

        return coords_xyz, mask_valid
