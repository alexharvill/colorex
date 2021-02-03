# Copyright 2021 Alex Harvill
# SPDX-License-Identifier: Apache-2.0
'colorex constants'
import enum
import numpy as np

REC_709_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

MAX_COMPONENT_VALUE = np.finfo(np.float32).max
SMALL_COMPONENT_VALUE = 1.0e-7
XYZ_D65_2A_WHITEPOINT = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

#https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
M_RGB_TO_XYZ = np.array([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
])
M_XYZ_TO_RGB = np.linalg.inv(M_RGB_TO_XYZ)
M_RGB_TO_XYZ_T = np.ascontiguousarray(M_RGB_TO_XYZ.T)
M_XYZ_TO_RGB_T = np.ascontiguousarray(M_XYZ_TO_RGB.T)

#https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
YCBCR_MIN, YCBCR_YMAX, YCBCR_CMAX, YCBCR_OFFSET = 16.0, 235.0, 240.0, 128.0
M_RGB_TO_YCBCR = np.array([
    [65.481, 128.553, 24.966],
    [-37.797, -74.203, 112.0],
    [112.0, -93.786, -18.214],
])
M_YCBCR_TO_RGB = np.linalg.inv(M_RGB_TO_YCBCR)
M_RGB_TO_YCBCR_T = np.ascontiguousarray(M_RGB_TO_YCBCR.T)
M_YCBCR_TO_RGB_T = np.ascontiguousarray(M_YCBCR_TO_RGB.T)

D50_TO_D65 = np.eye(3, dtype=np.float32)
D50_TO_D65[0, 0] = 0.85027254
D50_TO_D65[1, 1] = 1.02490389
D50_TO_D65[2, 2] = 1.38509774
D50_TO_D65_T = np.ascontiguousarray(D50_TO_D65.T)


class Space(enum.Enum):
  'named color spaces'
  RGB = 0
  SRGB = 1
  XYZ = 2
  xyY = 3
  LAB = 4
  YCbCr = 5
  LUM = 6


S = Space
