# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve
from skimage.transform import resize
from scipy.ndimage import affine_transform

from niftynet.layer.base_layer import Layer

class PSFLayer(Layer):
    """
        This layer is applied after the inherent data augmentation layers of NiftyNet, 
        which pertain to geometric transformations applied to both the input (soon to be 
        degraded) images and the ground truth (target) images. In this case, the input 
        images constitute the ground truth images, as well. The PSFLayer creates on-the-fly 
        degradations to the input images.
    """

    def __init__(self, image_name, binary_masking_func=None):

        self.image_name = image_name
        super(PSFLayer, self).__init__(name='psf_layer')

    def layer_op(self, image, mask=None):
        if isinstance(image, dict):
            image_data = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_data = np.asarray(image, dtype=np.float32)
        image_mask = np.ones_like(image_data, dtype=np.bool)

        # Gaussian PSF standard deviation per axis.
        sigmax = np.random.uniform(1e-5,2.5)
        sigmay = np.random.uniform(1e-5,2.5)
        sigmaz = np.random.uniform(1e-5,2.5)

        # Axis for truncation (slice-select axis)
        axis = 2

        # Downscaling factor per axis.
        scalex = np.random.choice(4)+1
        scaley = np.random.choice(4)+1
        scalez = np.random.choice(4)+1

        # Standard deviation of additive white Gaussian noise.
        wgnsigma = np.random.uniform(0,40)

        # Covariance matrix of Gaussian PSF.
        PSF_Cov = np.array([[sigmax**2,0,0],
                            [0,sigmay**2,0],
                            [0,0,sigmaz**2]])

        # PSF kernel axis-wise length.
        length = 4*np.ceil(np.sqrt(np.amax(PSF_Cov))) + 1

        # Since nominal spatial resolution in ground truth images is approximately 1mm isotropic, 
        # the axis-wise distance between the centers of two consecutive voxels with respect to x, 
        # y, and z axes is either 0 or 1mm. Hence, sampling the Gaussian pdf at the respective 
        # discrete grid, as outlined in [1], and applying normalization yields the Gaussian kernel for PSF.
        # [1] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html
        f = np.floor(length/2)
        x, y, z = np.mgrid[-f:(f+1), -f:(f+1), -f:(f+1)]
        pos = np.empty(x.shape + (3,))
        pos[:, :, :, 0] = x
        pos[:, :, :, 1] = y
        pos[:, :, :, 2] = z
        mean = [0, 0, 0]
        rv = multivariate_normal(mean, PSF_Cov)

        # The resulting kernel is normalized to defuce the PSF kernel.
        PSF = rv.pdf(pos)/np.sum(rv.pdf(pos))

        # Truncation of one axis (always the slice-select axis)
        truncation = np.floor(length/3).astype(np.int)

        if axis == 0:
            PSF[:truncation,:,:] = 0
            PSF[-truncation:,:,:] = 0
        elif axis == 1:
            PSF[:,:truncation,:] = 0
            PSF[:,-truncation:,:] = 0
        else:
            PSF[:,:,:truncation] = 0
            PSF[:,:,-truncation:] = 0

        # Re-normalization of the PSF kernel.
        PSF = PSF/np.sum(PSF)

        if image_data.ndim == 3:
            image_data = apply_PSF(image_data, PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez)
        if image_data.ndim == 5:
            for m in range(image_data.shape[4]):
                if m == 0: # T1 modality
                    for t in range(image_data.shape[3]):
                        image_data[..., t, m] = apply_PSF_T1(
                            image_data[..., t, m], PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez)
                elif m == 1: # T2 modality
                    for t in range(image_data.shape[3]):
                        image_data[..., t, m] = apply_PSF_T2(
                            image_data[..., t, m], PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez)
                else: # PD modality
                    for t in range(image_data.shape[3]):
                        image_data[..., t, m] = apply_PSF_PD(
                            image_data[..., t, m], PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez)

        if isinstance(image, dict):
            image[self.image_name] = image_data
            mask = {self.image_name: image_mask}
            return image, mask
        else:
            return image_data, image_mask

def apply_PSF_T1(image, PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez):

    # No rotation is applied to the T1 modality as in the case of static dataset generation.
    # This is due to the fact that the PSF layer is applied after the inherent data augmentation
    # layers of NiftyNet, which carry out geometric transformations.

    # The ground truth T1 image is convolved with the PSF kernel. 
    # It is to be noted that the application of PSF is axis-aligned with 
    # respect to the acquisition system and not to the random rotation 
    # applied to the ground truth image based on the data augmentation 
    # layers of NiftyNet.
    data_blurred = fftconvolve(image, PSF, mode='same')
            
    # The blurred image is downsampled.
    data_blurred_downsampled = resize(data_blurred, (int(image.shape[0]/scalex), int(image.shape[1]/scaley), int(image.shape[2]/scalez)), order = 3, mode = 'reflect')
            
    # Noise is added to the blurred and downsampled image.
    whitenoise = np.random.normal(0, wgnsigma, (data_blurred_downsampled.shape[0], data_blurred_downsampled.shape[1], data_blurred_downsampled.shape[2]))
    data_blurred_downsampled_noisy = data_blurred_downsampled + whitenoise
            
    # The blurred, downsampled, and noisy image is interpolated and, 
    # thus, it now matches the size of its respective ground truth image.
    data_blurred_downsampled_noisy_interp = resize(data_blurred_downsampled_noisy, (image.shape[0], image.shape[1], image.shape[2]), order = 3, mode = 'reflect') 

    image = data_blurred_downsampled_noisy_interp
    
    return image

def apply_PSF_T2(image, PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez):

    # The input T2 image constitutes a ground truth T2 which is already registered to its respective T1 image.
    # Assuming an unknown registration transformation, we simply rotate the input T2 image based on a pre-defined 
    # combination of angles. Convolution with PSF, application of downsampling, and addition of noise are then 
    # carried out. The resulting image is rotated back to its initial position. The reason for applying a rotation 
    # to the T2 image is in order to have axis-aligned PSF and downsampling operation.
    theta_x = 45
    theta_y = 10
    theta_z = -30
    theta_x = theta_x/180.0*math.pi
    Rx = np.array([[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = theta_y/180.0*math.pi
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
    theta_z = theta_z/180.0*math.pi
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    center_ = 0.5 * np.asarray(image.shape, dtype=np.int64)
    c_offset = center_ - center_.dot(R)
    image[...] = affine_transform(image[...], R.T, c_offset, order = 3)

    # The ground truth T2 image is convolved with the PSF kernel. 
    # It is to be noted that the application of PSF is axis-aligned with 
    # respect to the acquisition system and not to the random rotation 
    # applied to the ground truth image based on the data augmentation 
    # layers of NiftyNet.
    data_blurred = fftconvolve(image, PSF, mode='same')

    # The blurred image is downsampled.
    data_blurred_downsampled = resize(data_blurred, (int(image.shape[0]/scalex), int(image.shape[1]/scaley), int(image.shape[2]/scalez)), order = 3, mode = 'reflect')

    # Noise is added to the blurred and downsampled image.
    whitenoise = np.random.normal(0, wgnsigma, (data_blurred_downsampled.shape[0], data_blurred_downsampled.shape[1], data_blurred_downsampled.shape[2]))
    data_blurred_downsampled_noisy = data_blurred_downsampled + whitenoise

    # The blurred, downsampled, and noisy image is interpolated and, 
    # thus, it now matches the size of its respective ground truth image.
    data_blurred_downsampled_noisy_interp = resize(data_blurred_downsampled_noisy, (image.shape[0], image.shape[1], image.shape[2]), order = 3, mode = 'reflect') 

    # The interpolated image is rotated back to match the orientation of the input T2 image, which is registered to T1.
    theta_x = -theta_x
    Rx = np.array([[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = -theta_y
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
    theta_z = -theta_z
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    center_ = 0.5 * np.asarray(image.shape, dtype=np.int64)
    c_offset = center_ - center_.dot(R)
    image[...] = affine_transform(data_blurred_downsampled_noisy_interp, R.T, c_offset, order = 3)

    return image

def apply_PSF_PD(image, PSF, sigmax, sigmay, sigmaz, wgnsigma, scalex, scaley, scalez):

    # The input PD image constitutes a ground truth PD which is already registered to its respective T1 image.
    # Assuming an unknown registration transformation, we simply rotate the input PD image based on a pre-defined 
    # combination of angles. Convolution with PSF, application of downsampling, and addition of noise are then 
    # carried out. The resulting image is rotated back to its initial position. The reason for applying a rotation 
    # to the PD image is in order to have axis-aligned PSF and downsampling operation.
    theta_x = -45
    theta_y = -10
    theta_z = 30
    theta_x = theta_x/180.0*math.pi
    Rx = np.array([[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = theta_y/180.0*math.pi
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
    theta_z = theta_z/180.0*math.pi
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    center_ = 0.5 * np.asarray(image.shape, dtype=np.int64)
    c_offset = center_ - center_.dot(R)
    image[...] = affine_transform(image[...], R.T, c_offset, order = 3)

    # The ground truth PD image is convolved with the PSF kernel. 
    # It is to be noted that the application of PSF is axis-aligned with 
    # respect to the acquisition system and not to the random rotation 
    # applied to the ground truth image based on the data augmentation 
    # layers of NiftyNet.
    data_blurred = fftconvolve(image, PSF, mode='same')

    # The blurred image is downsampled.
    data_blurred_downsampled = resize(data_blurred, (int(image.shape[0]/scalex), int(image.shape[1]/scaley), int(image.shape[2]/scalez)), order = 3, mode = 'reflect')

    # Noise is added to the blurred and downsampled image.
    whitenoise = np.random.normal(0, wgnsigma, (data_blurred_downsampled.shape[0], data_blurred_downsampled.shape[1], data_blurred_downsampled.shape[2]))
    data_blurred_downsampled_noisy = data_blurred_downsampled + whitenoise

    # The blurred, downsampled, and noisy image is interpolated and, 
    # thus, it now matches the size of its respective ground truth image.
    data_blurred_downsampled_noisy_interp = resize(data_blurred_downsampled_noisy, (image.shape[0], image.shape[1], image.shape[2]), order = 3, mode = 'reflect') 

    # The interpolated image is rotated back to match the orientation of the input PD image, which is registered to T1.
    theta_x = -theta_x
    Rx = np.array([[1, 0, 0],[0, np.cos(theta_x), -np.sin(theta_x)],[0, np.sin(theta_x), np.cos(theta_x)]])
    theta_y = -theta_y
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
    theta_z = -theta_z
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],[np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = np.matmul(np.matmul(Rz, Ry), Rx)

    center_ = 0.5 * np.asarray(image.shape, dtype=np.int64)
    c_offset = center_ - center_.dot(R)
    image[...] = affine_transform(data_blurred_downsampled_noisy_interp, R.T, c_offset, order = 3)

    return image