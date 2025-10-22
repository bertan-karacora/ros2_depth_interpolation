import torch

from ros2_depth_interpolation.projection.model_camera import ModelCamera


class ModelRationalPolynomial(ModelCamera):
    # Note: Distortion is ignored for now since they do not have a noticeable impact for the gemini.

    keys_params_distortion = ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k1 = self.params_distortion["k1"]
        self.k2 = self.params_distortion["k2"]
        self.k3 = self.params_distortion["k3"]
        self.k4 = self.params_distortion["k4"]
        self.k5 = self.params_distortion["k5"]
        self.k6 = self.params_distortion["k6"]
        self.p1 = self.params_distortion["p1"]
        self.p2 = self.params_distortion["p2"]

    @torch.inference_mode()
    def project_points_onto_image(self, coords_xyz, use_invalid_coords=True, use_mask_fov=True, use_half_precision=True):
        """Project 3D points onto 2D image.
        Shape of coords_xyz: (B, 3, N)
        Coordinate frame of points: [right, down, front]
        Coordinate frame of image: [right, down]"""
        if use_half_precision:
            coords_xyz = coords_xyz.half()

        x, y, z = coords_xyz[:, 0, :], coords_xyz[:, 1, :], coords_xyz[:, 2, :]

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        coords_uv = torch.stack((u, v), axis=1)

        mask_valid = torch.ones_like(u, dtype=bool)

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

        u, v = coords_uv[:, 0, :], coords_uv[:, 1, :]

        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        mz = torch.ones_like(mx)

        factor = 1.0 / torch.sqrt(mx**2 + my**2 + 1.0)
        coords_xyz = factor[:, None, :] * torch.stack((mx, my, mz), axis=1)

        mask_valid = torch.ones_like(mx, dtype=bool)

        return coords_xyz, mask_valid
