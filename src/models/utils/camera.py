import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F
import numpy as np
from .graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCenterShift
import kornia
from torchvision.utils import save_image
# from ..utils.render_utils import (
#     build_gt_normal_map,
#     quat_to_rotmat,
#     render_primitives,
#     rotmat_to_quat,
# )



# def loadCam(args, id, cam_info: CameraInfo, resolution_scale):
#     orig_h, orig_w = args.hw

#     if args.resolution == -1:
#         global_down = 1
#     else:
#         global_down = orig_w / args.resolution

#     scale = float(global_down) * float(resolution_scale)
#     resolution = (int(orig_w / scale), int(orig_h / scale))

#     vfov = args.vfov
#     hfov = args.hfov
#     if cam_info.pointcloud_camera is not None:
#         intensity = cam_info.intensity
#         if intensity is None:
#             intensity = np.ones_like(cam_info.pointcloud_camera)[:, 0]

#         w = resolution[0]
#         h = resolution[1]

#         pts_depth = np.zeros([1, h, w])
#         pts_intensity = np.zeros([1, h, w])
#         point_camera = cam_info.pointcloud_camera
#         x = point_camera[:, 0]
#         y = point_camera[:, 1]
#         z = point_camera[:, 2]
#         phi = np.arctan2(x, z)
#         theta = np.arctan2(np.sqrt(x ** 2 + z ** 2), -y)
#         r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

#         VFOV_max = np.pi / 2 - vfov[0] * np.pi / 180
#         VFOV_min = np.pi / 2 - vfov[1] * np.pi / 180
#         HFOV_max = hfov[1] * np.pi / 180
#         HFOV_min = hfov[0] * np.pi / 180

#         theta = (theta - VFOV_min) * h / (VFOV_max - VFOV_min)
#         phi = (phi - HFOV_min) * w / (HFOV_max - HFOV_min)
#         uvz = np.stack((theta, phi, r, intensity), 1)

#         uvz = uvz[uvz[:, 0] >= -0.5]
#         uvz = uvz[uvz[:, 0] < h - 0.5]
#         uvz = uvz[uvz[:, 1] >= -0.5]
#         uvz = uvz[uvz[:, 1] < w - 0.5]
#         uv = uvz[:, :2]
#         uv = np.around(uv).astype(int)

#         for i in range(uv.shape[0]):
#             x, y = uv[i]
#             if pts_depth[0, x, y] == 0:
#                 pts_depth[0, x, y] = uvz[i, 2]
#                 pts_intensity[0, x, y] = uvz[i, 3]
#             elif uvz[i, 2] < pts_depth[0, x, y]:
#                 pts_depth[0, x, y] = uvz[i, 2]
#                 pts_intensity[0, x, y] = uvz[i, 3]

#         pts_depth = torch.from_numpy(pts_depth).float().cuda()
#         pts_intensity = torch.from_numpy(pts_intensity).float().cuda()
#     else:
#         pts_depth = None
#         pts_intensity = None

#     return Camera(
#         colmap_id=cam_info.uid,
#         uid=id,
#         R=cam_info.R,
#         T=cam_info.T,
#         vfov=vfov,
#         hfov=hfov,
#         data_device=args.data_device,
#         timestamp=cam_info.timestamp,
#         resolution=resolution,
#         pts_depth=pts_depth,
#         pts_intensity=pts_intensity,
#         towards=cam_info.towards
#     )


# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     for id, c in enumerate(tqdm(cam_infos)):
#         camera_list.append(loadCam(args, id, c, resolution_scale))

#     return camera_list





class Camera(nn.Module):
    def __init__(self, R, T, vfov=None, hfov=None, 
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cpu", timestamp=0.0,
                 timestamp_sec=None,
                 resolution=None, 
                 pts_depth=None, pts_intensity=None, points=None,
                 viewmatrix=None,
                 ring_to_elevation_deg=None,
                 ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.vfov = vfov
        self.hfov = hfov
        self.resolution = resolution
        

        # try:
        #     self.data_device = torch.device(data_device)
        # except Exception as e:
        #     print(e)
        #     print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
        #     self.data_device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.ring_to_elevation_deg = (
            list(ring_to_elevation_deg) if ring_to_elevation_deg is not None else None
        )
        self.row_to_theta = self._build_row_to_theta(
            self.image_height,
            vfov,
            self.ring_to_elevation_deg,
        )

        self.pts_depth = pts_depth if pts_depth is not None else pts_depth
        self.pts_intensity = pts_intensity if pts_intensity is not None else pts_intensity
        self.points = points

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if viewmatrix is None:
            view = getWorld2View2(R, T, trans, scale)
        else:
            view = np.asarray(viewmatrix, dtype=np.float32)
        self.world_view_transform = torch.tensor(view).transpose(0, 1)
        self.projection_matrix = torch.eye(4) # no use
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.timestamp = timestamp
        self.timestamp_sec = (
            None if timestamp_sec is None else float(timestamp_sec)
        )
        self.grid = kornia.utils.create_meshgrid(
            self.image_height,
            self.image_width,
            normalized_coordinates=False,
            device="cpu",
        )[0]

    # Ray origin of every row in this camera's frame; None means the camera centre.
    row_origin = None

    @staticmethod
    def _build_row_to_theta(image_height, vfov, ring_to_elevation_deg=None):
        if ring_to_elevation_deg is not None:
            row_to_el = np.sort(np.asarray(list(ring_to_elevation_deg), dtype=np.float32))[::-1]
            if row_to_el.shape[0] != int(image_height):
                raise ValueError(
                    f"ring_to_elevation_deg length {row_to_el.shape[0]} "
                    f"does not match image_height {image_height}"
                )
        else:
            row = np.arange(int(image_height), dtype=np.float32)
            row_to_el = float(vfov[1]) - row / float(image_height) * (float(vfov[1]) - float(vfov[0]))
        row_to_theta = np.pi / 2.0 - np.radians(row_to_el)
        return torch.from_numpy(row_to_theta.astype(np.float32))


class StackedPanoramaCamera(Camera):
    """A LiDAR panorama whose row blocks cast rays from different origins.

    A spinning LiDAR's emitter blocks sit at different heights, so one ray
    origin cannot serve every row. Each block is an ordinary Camera whose
    viewmatrix places the origin at that block's emitters. The renderer draws
    the blocks separately and stacks rows ``kept_rows`` of each, top to
    bottom, into this camera's image.

    Render each block with the full row table and keep only its rows. The
    rasterizer clamps a Gaussian centre outside a camera's row table to the
    edge row, and both its tile rectangle and its screen-space low-pass filter
    use that clamped centre. A block camera cut at the block boundary would
    therefore paint spurious coverage along that boundary.

    This camera's own frame is centred on the sensor. ``row_origin`` gives
    each row's ray origin in that frame, for back-projecting range images.
    """

    def __init__(self, sub_cameras, row_origin, kept_rows=None, **camera_kwargs):
        super().__init__(**camera_kwargs)
        self.sub_cameras = list(sub_cameras)
        if kept_rows is None:
            kept_rows = [(0, int(camera.image_height)) for camera in self.sub_cameras]
        self.kept_rows = [(int(first), int(last)) for first, last in kept_rows]
        if len(self.kept_rows) != len(self.sub_cameras) or any(
            not 0 <= first < last <= int(camera.image_height)
            for (first, last), camera in zip(self.kept_rows, self.sub_cameras)
        ):
            raise ValueError("kept_rows needs one valid row range per sub-camera")
        if sum(last - first for first, last in self.kept_rows) != int(self.image_height):
            raise ValueError("Kept sub-camera rows must add up to the panorama height")
        if any(int(camera.image_width) != int(self.image_width) for camera in self.sub_cameras):
            raise ValueError("Sub-cameras must share the panorama width")
        stacked = torch.cat([camera.row_to_theta[first:last]
                             for camera, (first, last) in zip(self.sub_cameras, self.kept_rows)])
        if not torch.allclose(stacked, self.row_to_theta):
            raise ValueError("Sub-camera row angles must stack into the panorama's rows")
        self.row_origin = torch.as_tensor(row_origin, dtype=torch.float32).reshape(
            int(self.image_height), 3
        )
