import numpy as np

class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy
    
    def project(self, kps3d):
        homo_kps3d = np.concatenate((kps3d, np.ones(kps3d.shape[:-1] + (1,))), axis=-1)
        homo_kps3d = homo_kps3d.reshape(*homo_kps3d.shape, 1)
        sz = len(homo_kps3d.shape)
        P = self.projection.reshape((1,)*(sz-2) + (3, 4))
        homo_kps2d = (P @ homo_kps3d).reshape(*homo_kps3d.shape[:-2], -1)
        kps2d = homo_kps2d[..., :2] / homo_kps2d[..., 2:3]
        return kps2d

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


# class Camera:
#     """
#     The class to describe cameras
#     """
#     def __init__(self, **kwargs):
#         """
#             R: 3x3 Rotation matrix
#             T: 3x1 Translation vector
#             f: 2x1 [fx, fy] focal length
#             c: 2x1 [cx, cy] camera center
#             k: 3x1 radial distortion param
#             p: 2x1 tangential distortion param
#         """
#         self.R = kwargs["R"]
#         self.T = kwargs["T"]
#         self.fx = kwargs["fx"].item()
#         self.fy = kwargs["fy"].item()
#         self.cx = kwargs["cx"].item()
#         self.cy = kwargs["cy"].item()
#         self.k = kwargs["k"]
#         self.p = kwargs["p"]
#         self.K = np.array([
#                 [self.fx, 0, self.cx],
#                 [0, self.fy, self.cy],
#                 [0, 0, 1]
#             ])
#         self.calculate_projection()

#     def calculate_projection(self):
#         self.P = self.K @ self.R @ np.concatenate((np.eye(3), -self.T.reshape(3, 1)), axis=1)
    
#     def update_after_crop_and_resize(self, bbox, image_shape):
#         hb, wb = bbox[3] - bbox[1], bbox[2] - bbox[0]
#         lamx = image_shape[0] / wb
#         lamy = image_shape[1] / hb

#         self.cx = lamx * (self.cx - bbox[0])
#         self.cy = lamy * (self.cy - bbox[1])
#         self.fx *= lamx
#         self.fy *= lamy
#         self.K = np.array([
#                 [self.fx, 0, self.cx],
#                 [0, self.fy, self.cy],
#                 [0, 0, 1]
#             ])
#         self.calculate_projection()
