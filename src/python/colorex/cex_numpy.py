# Copyright 2021 Alex Harvill
# SPDX-License-Identifier: Apache-2.0
'colorex numpy'
import numpy as np

from colorex.cex_constants import (
    REC_709_LUMA_WEIGHTS,
    MAX_COMPONENT_VALUE,
    SMALL_COMPONENT_VALUE,
    M_RGB_TO_XYZ_T,
    M_XYZ_TO_RGB_T,
    D50_TO_D65_T,
)


def gamma_correct(values, gamma):
  'apply a gamma power curve'
  #pylint: disable=assignment-from-no-return
  return np.power(values, gamma)


def srgb_to_rgb_aprox_gamma_22(srgb_img):
  '''
  approximate srgb conversion using a gamma correct with power 2.2
  this approach is very common in vfx where srgb is used rarely
  '''
  return gamma_correct(srgb_img, gamma=2.2)


def rgb_to_srgb_aprox_gamma_22(rgb_img):
  '''
  approximate inverse srgb conversion using a gamma correct with power 1.0/2.2
  this approach is very common in vfx where srgb is used rarely
  '''
  return gamma_correct(rgb_img, gamma=1.0 / 2.2)


def srgb_to_rgb2(srgb_img):
  '''
  2.4 gamma and linear below .04045
  very close to the approach taken internally to skimagewhen when converting:
    srgb > rgb > xyz

  skimage does not expose the srgb > rgb transform
  '''
  arr = srgb_img.copy()
  mask = arr > 0.04045
  arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
  arr[~mask] /= 12.92
  return arr


def srgb_to_rgb(srgb):
  '''
  convert from a gamma 2.4 color space to linear rgb
  this code can be directly adapted to keras or another autodiff framework
  '''

  srgb = np.clip(srgb, SMALL_COMPONENT_VALUE, MAX_COMPONENT_VALUE)

  linear_mask = (srgb <= 0.04045).astype(np.float32)

  exponential_mask = (srgb > 0.04045).astype(np.float32)

  linear_pixels = srgb / 12.92

  exponential_pixels = np.power((srgb + 0.055) / 1.055, 2.4)

  return linear_pixels * linear_mask + exponential_pixels * exponential_mask


def rgb_to_srgb(rgb):
  '''
  convert from linear rgb to a gamma 2.4 color space
  this code can be directly adapted to keras or another autodiff framework
  '''

  rgb = np.clip(rgb, SMALL_COMPONENT_VALUE, MAX_COMPONENT_VALUE)

  linear_mask = (rgb <= 0.0031308).astype(np.float32)

  exponential_mask = (rgb > 0.0031308).astype(np.float32)

  linear_pixels = rgb * 12.92

  exponential_pixels = 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055

  return linear_pixels * linear_mask + exponential_pixels * exponential_mask


def rgb_to_luminance(rgb, luma_weights=REC_709_LUMA_WEIGHTS):
  'luminance of a color array, or higher dim color images'

  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

  return r * luma_weights[0] + g * luma_weights[1] + b * luma_weights[2]


def xyz_to_xyy(XYZ):
  '''
  convert from XYZ color space to xyY
    XYZ: consistent units for each component
    xyY: normalized chromaticity with xy in 0-1, Y in 0-inf
  https://en.wikipedia.org/wiki/CIE_1931_color_space
  http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
  '''

  X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]

  XYZ_sum = X + Y + Z

  invalid_mask = (XYZ_sum < SMALL_COMPONENT_VALUE).astype(np.float32)
  valid_mask = 1.0 - invalid_mask

  # if xyz_sum == 0, set to 1.0
  XYZ_sum = invalid_mask + valid_mask * XYZ_sum

  x = X / XYZ_sum
  y = Y / XYZ_sum

  x *= valid_mask
  y *= valid_mask
  Y *= valid_mask

  return np.stack([x, y, Y], axis=-1)


def xyy_to_xyz(xyY):
  '''
  convert from xyY color space to XYZ
    xyY: normalized chromaticity with xy in 0-1, Y in 0-inf
    XYZ: consistent units for each component
  https://en.wikipedia.org/wiki/CIE_1931_color_space
  http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
  '''

  x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]

  invalid_mask = (y < SMALL_COMPONENT_VALUE).astype(np.float32)
  valid_mask = 1.0 - invalid_mask

  y = invalid_mask + valid_mask * y
  norm = Y / y

  X = x * norm
  Z = (1 - x - y) * norm

  X *= valid_mask
  Y *= valid_mask
  Z *= valid_mask

  return np.stack([X, Y, Z], axis=-1)


def point_or_points_or_image_wrapper(func):
  '''
  a decorated function will be called with reshaped input and output to support
    a single 3d point: pt.shape = (3,)
    an array of 3d points: pts.shape = (N,3)
    an image of 3d points: pts.shape = (H,W,3)

  wrapped functions should internally support the array of 3d points case (N,3)
  '''

  def inner(point_or_points_or_image, *args, **kwargs):

    result_shape = list(point_or_points_or_image.shape)

    if len(result_shape) == 1:
      img_shape = [1, 1] + result_shape

    elif len(result_shape) == 2:
      img_shape = [1] + result_shape

    elif len(result_shape) == 3:
      img_shape = list(result_shape)

    assert img_shape[-1] == 3
    points_shape = (img_shape[0] * img_shape[1], img_shape[2])
    points = point_or_points_or_image.reshape(points_shape)

    result = func(points, *args, **kwargs)

    return result.reshape(result_shape)

  return inner


@point_or_points_or_image_wrapper
def D50_to_D65(points):
  'matrix transformation from D50 whitepoint to D65'
  return np.matmul(points, D50_TO_D65_T)


@point_or_points_or_image_wrapper
def rgb_to_xyz(points):
  'matrix transformation from linear RGB to XYZ color space'
  return np.matmul(points, M_RGB_TO_XYZ_T)


@point_or_points_or_image_wrapper
def xyz_to_rgb(points):
  'matrix transformation from XYZ to linear RGB color space'
  return np.matmul(points, M_XYZ_TO_RGB_T)
