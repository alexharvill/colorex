# Copyright 2021 Alex Harvill
# SPDX-License-Identifier: Apache-2.0
'colorex keras layers'
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from colorex.cex_constants import (
    REC_709_LUMA_WEIGHTS,
    MAX_COMPONENT_VALUE,
    SMALL_COMPONENT_VALUE,
    XYZ_D65_2A_WHITEPOINT,
    M_RGB_TO_XYZ_T,
    M_XYZ_TO_RGB_T,
)

from colorex.cex_constants import Space as S


def srgb_to_rgb(srgb):
  'convert from a gamma 2.4 color space to linear rgb'

  srgb = K.clip(srgb, SMALL_COMPONENT_VALUE, MAX_COMPONENT_VALUE)

  linear_mask = K.cast(srgb <= 0.04045, dtype='float32')

  exponential_mask = K.cast(srgb > 0.04045, dtype='float32')

  linear_pixels = srgb / 12.92

  exponential_pixels = K.pow((srgb + 0.055) / 1.055, 2.4)

  return linear_pixels * linear_mask + exponential_pixels * exponential_mask


def rgb_to_srgb(rgb):
  'convert from linear rgb to a gamma 2.4 color space'

  rgb = K.clip(rgb, SMALL_COMPONENT_VALUE, MAX_COMPONENT_VALUE)

  linear_mask = K.cast(rgb <= 0.0031308, dtype='float32')

  exponential_mask = K.cast(rgb > 0.0031308, dtype='float32')

  linear_pixels = rgb * 12.92

  exponential_pixels = 1.055 * K.pow(rgb, 1.0 / 2.4) - 0.055

  return linear_pixels * linear_mask + exponential_pixels * exponential_mask


def rgb_to_xyz(rgb):
  'convert from gamma 1.0 RGB color space to XYZ'
  return K.dot(rgb, K.constant(M_RGB_TO_XYZ_T))


def xyz_to_rgb(xyz):
  'convert from XYZ to a gamma 1.0 RGB color space'
  return K.dot(xyz, K.constant(M_XYZ_TO_RGB_T))


def xyz_to_lab(xyz):
  'convert from a CIEXYZ space to CIELa*b*'

  xyz = xyz / K.constant(XYZ_D65_2A_WHITEPOINT)

  xyz = K.clip(xyz, SMALL_COMPONENT_VALUE, MAX_COMPONENT_VALUE)

  epsilon = 0.008856  #(6.0 / 29.0)**3 # use hardcoded value to match skimage for validation

  linear_mask = K.cast(xyz <= epsilon, dtype='float32')

  cuberoot_mask = K.cast(xyz > epsilon, dtype='float32')

  linear_pixels = 7.787 * xyz + 16.0 / 116.0

  cuberoot_pixels = K.pow(xyz, 1.0 / 3.0)

  xyz = linear_pixels * linear_mask + cuberoot_pixels * cuberoot_mask

  x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

  # Vector scaling
  L = (116.0 * y) - 16.0
  a = 500.0 * (x - y)
  b = 200.0 * (y - z)

  return K.stack([L, a, b], axis=-1)


def lab_to_xyz(lab):
  'convert from lab to xyz color space assuming a D65 whitepoint + 2deg angle'

  l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
  y = (l + 16.0) / 116.0
  x = (a / 500.0) + y
  z = y - (b / 200.0)

  z = K.clip(z, 0.0, 1e20)

  xyz = K.stack([x, y, z], axis=-1)

  epsilon = 6.0 / 29.0

  linear_mask = K.cast(xyz < epsilon, dtype='float32')

  cube_mask = K.cast(xyz >= epsilon, dtype='float32')

  linear_pixels = (xyz - 16.0 / 116.) / 7.787

  cube_pixels = K.pow(xyz, 3.0)

  xyz = linear_pixels * linear_mask + cube_pixels * cube_mask

  xyz = xyz * K.constant(XYZ_D65_2A_WHITEPOINT)

  return xyz


def rgb_to_luminance(rgb, luma_weights=REC_709_LUMA_WEIGHTS):
  'luminance of a color array, or higher dim color images'

  r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

  return r * luma_weights[0] + g * luma_weights[1] + b * luma_weights[2]


####  following transforms are macros using the above primitive transforms


def xyz_to_srgb(xyz):
  'xyz > rgb > srgb'
  tmp = xyz_to_rgb(xyz)
  return rgb_to_srgb(tmp)


def srgb_to_xyz(srgb):
  'srgb > rgb > xyz'
  tmp = srgb_to_rgb(srgb)
  return rgb_to_xyz(tmp)


def srgb_to_lab(srgb):
  'srgb -> xyz -> lab'
  tmp = srgb_to_xyz(srgb)
  return xyz_to_lab(tmp)


def lab_to_srgb(lab):
  'lab > xyz > srgb'
  tmp = lab_to_xyz(lab)
  return xyz_to_srgb(tmp)


def rgb_to_lab(rgb):
  'rgb -> xyz -> lab'
  tmp = rgb_to_xyz(rgb)
  return xyz_to_lab(tmp)


def lab_to_rgb(lab):
  'lab > xyz > rgb'
  tmp = lab_to_xyz(lab)
  return xyz_to_rgb(tmp)


class Bias(keras.layers.Layer):
  'simple layer for testing'

  def __init__(self, **kwargs):
    self.bias = None
    super(Bias, self).__init__(**kwargs)

  def build(self, input_shape):
    'creates trainable weight variable for this bias layer'
    self.bias = self.add_weight(
        name='weights',
        shape=(1,),
        initializer='zeros',
        trainable=True,
    )

    super(Bias, self).build(input_shape)  # will set self.built = True

  def call(self, inputs, **kwargs):
    'builds an output tensor for this op'
    return inputs + self.bias


class ColorSpace(keras.layers.Layer):
  'convert from_space to_space'

  transforms = {
      (S.SRGB, S.RGB): srgb_to_rgb,
      (S.SRGB, S.XYZ): srgb_to_xyz,
      (S.SRGB, S.LAB): srgb_to_lab,
      (S.SRGB, S.LUM): None,
      (S.RGB, S.SRGB): rgb_to_srgb,
      (S.RGB, S.XYZ): rgb_to_xyz,
      (S.RGB, S.LAB): rgb_to_lab,
      (S.RGB, S.LUM): rgb_to_luminance,
      (S.XYZ, S.RGB): xyz_to_rgb,
      (S.XYZ, S.LAB): xyz_to_lab,
      (S.XYZ, S.SRGB): xyz_to_srgb,
      (S.XYZ, S.LUM): None,
      (S.LAB, S.XYZ): lab_to_xyz,
      (S.LAB, S.SRGB): lab_to_srgb,
      (S.LAB, S.RGB): lab_to_rgb,
      (S.LAB, S.LUM): None,
  }

  def __init__(self, from_space, to_space, **kwargs):
    self.from_space = S[from_space]
    self.to_space = S[to_space]
    super(ColorSpace, self).__init__(**kwargs)

  def call(self, inputs, **kwargs):
    'builds an output tensor for this op'
    t = self.transforms.get((self.from_space, self.to_space))
    if t is None:
      ValueError(f'bad transform[{self.from_space.name},{self.to_space.name}]')
    return t(inputs)

  def get_config(self):
    'save from and to attributes'
    return dict(
        super(ColorSpace, self).get_config(),
        from_space=self.from_space.name,
        to_space=self.to_space.name,
    )

  def compute_output_shape(self, input_shape):
    'some transforms remove the color dimension'
    result = input_shape
    if self.to_space in (S.LUM,):
      result = input_shape[:-1]
    return result
