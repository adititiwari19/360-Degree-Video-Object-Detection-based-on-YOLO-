#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import numpy as np

from scipy.interpolate import RegularGridInterpolator as interp2d

class ImageRecorder(object):

    """
    Record normal view video given 360 degree video.
    """

    def __init__(self, sphereW, sphereH, view_angle=65.5, imgW=640):
        self.sphereW = sphereW
        self.sphereH = sphereH

        self._imgW = imgW
        #w, h = 4, 3
        w, h = 4, 4 # Jiaxi

        self._imgH = h * self._imgW / w
        self._Y = np.arange(self._imgH) + (self._imgW - self._imgH)/2

        TX, TY = self._meshgrid()
        R, ANGy = self._compute_radius(view_angle, TY)

        self._R = R
        self._ANGy = ANGy
        self._Z = TX

    def _meshgrid(self):
        """Construct mesh point
        :returns: TX, TY

        """
        TX, TY = np.meshgrid(range(self._imgW), range(self._imgW))
        TX = TX[self._Y,:]
        TY = TY[self._Y,:]

        TX = TX.astype(np.float64) - 0.5
        TX -= self._imgW/2

        TY = TY.astype(np.float64) - 0.5
        TY -= self._imgW/2
        return TX, TY

    def _compute_radius(self, view_angle, TY):
        _view_angle = np.pi * view_angle / 180.
        r = self._imgW/2 / np.tan(_view_angle/2)
        R = np.sqrt(np.power(TY, 2) + r**2)
        ANGy = np.arctan(-TY/r)
        return R, ANGy

    def catch(self, x, y, image):
        Px, Py = self._sample_points(x, y)
        warped_image = self._warp_image(Px, Py, image)
        return warped_image

    def _sample_points(self, x, y):
        angle_x, angle_y = self._direct_camera(x, y)
        Px = (angle_x + np.pi) / (2*np.pi) * self.sphereW + 0.5
        Py = (np.pi/2 - angle_y) / np.pi * self.sphereH + 0.5
        INDx = Px < 1
        Px[INDx] += self.sphereW
        return Px, Py

    def _direct_camera(self, rotate_x, rotate_y):
        angle_y = self._ANGy + rotate_y
        X = np.sin(angle_y) * self._R
        Y = - np.cos(angle_y) * self._R
        Z = self._Z

        INDn = np.abs(angle_y) > np.pi/2

        angle_x = np.arctan(Z / -Y)
        RZY = np.linalg.norm(np.stack((Y, Z), axis=0), axis=0)
        angle_y = np.arctan(X / RZY)

        angle_x[INDn] += np.pi
        angle_x += rotate_x

        INDy = angle_y < -np.pi/2
        angle_y[INDy] = -np.pi -angle_y[INDy]
        angle_x[INDy] = angle_x[INDy] + np.pi

        INDx = angle_x <= -np.pi
        angle_x[INDx] += 2*np.pi
        INDx = angle_x > np.pi
        angle_x[INDx] -= 2*np.pi
        return angle_x, angle_y

    def _warp_image(self, Px, Py, frame):
        # Boundary Points of the Window in the 360 Image, Cartesian Coordinate
        minX = max(0, int(np.floor(Px.min())))
        minY = max(0, int(np.floor(Py.min())))

        maxX = min(int(self.sphereW), int(np.ceil(Px.max())))
        maxY = min(int(self.sphereH), int(np.ceil(Py.max())))

        im = frame[minY:maxY, minX:maxX, :]
        Px -= minX
        Py -= minY
        warped_images = []

        y_grid = np.arange(im.shape[0])
        x_grid = np.arange(im.shape[1])
        samples = np.vstack([Py.ravel(), Px.ravel()]).transpose()
        for c in xrange(3):
            full_image = interp2d((y_grid, x_grid), im[:,:,c],
                                   bounds_error=False,
                                   method='linear',
                                   fill_value=None)
            warped_image = full_image(samples).reshape(Px.shape)
            warped_images.append(warped_image)
        warped_image = np.stack(warped_images, axis=2)
        return warped_image