# Copyright 2021 Alex Harvill
# SPDX-License-Identifier: Apache-2.0
'test colorex keras layers'

import logging
import unittest
import skimage
import numpy as np

from colorex import cex_numpy
from colorex.cex_constants import Space as S

cex_keras = None


class TestColorexKeras(unittest.TestCase):
  'to be run via unittest discovery'

  def setUp(self):
    '''
    when using unittest discovery to find all tests in a git repo
    importing keras at the top of a test can slow everything down
    delay importing keras to cut down on startup time for other tests
    '''
    #pylint: disable=global-statement,import-outside-toplevel
    import colorex.cex_keras
    global cex_keras
    cex_keras = colorex.cex_keras

  def tearDown(self):
    cex_keras.K.clear_session()

  def test_cex_keras_color_space_layer(self):
    'test convert between color spaces against skimage'

    # build a test image by from the RGB cube
    cube_size = 16
    H = W = 64
    N = cube_size**3
    assert N == H * W
    int_cube = np.array(np.mgrid[:cube_size, :cube_size, :cube_size]).T
    srgb = int_cube.astype(np.float32)
    srgb /= float(cube_size - 1)
    srgb = srgb.reshape((H, W, 3))

    rgb = cex_numpy.srgb_to_rgb2(srgb)

    lab = skimage.color.rgb2lab(srgb)
    xyz = skimage.color.rgb2xyz(srgb)
    lab2 = skimage.color.xyz2lab(xyz)
    xyz2 = skimage.color.lab2xyz(lab)
    srgb2 = skimage.color.xyz2rgb(xyz)
    srgb3 = skimage.color.lab2rgb(lab)

    xyy = cex_numpy.xyz_to_xyy(xyz)
    xyz3 = cex_numpy.xyy_to_xyz(xyy)

    for s_from, s_to, source, target in [
        (S.RGB, S.SRGB, rgb, srgb),
        (S.SRGB, S.RGB, srgb, rgb),
        (S.SRGB, S.XYZ, srgb, xyz),
        (S.SRGB, S.LAB, srgb, lab),
        (S.XYZ, S.LAB, xyz, lab2),
        (S.XYZ, S.SRGB, xyz, srgb2),
        (S.LAB, S.XYZ, lab, xyz2),
        (S.LAB, S.SRGB, lab, srgb3),
        (S.XYZ, S.xyY, xyz, xyy),
        (S.xyY, S.XYZ, xyy, xyz3),
    ]:
      x = cex_keras.keras.layers.Input(
          batch_shape=(None, None, None, 3),
          name='img_input',
          dtype='float32',
      )

      # adds zero initially
      m = cex_keras.Bias(name='bias')(x)
      name = f'{s_from.name}_to_{s_to.name}'
      y_op = cex_keras.ColorSpace(s_from.name, s_to.name, name=name)

      y = y_op(m)

      mtmp = cex_keras.keras.models.Model(x, y, name=name)

      mtmp.compile(
          optimizer=cex_keras.keras.optimizers.Adam(lr=0.1,),
          loss='mse',
          metrics=['mse'],
      )

      tmp_path = '/tmp/%s.h5' % (name)

      mtmp.save(tmp_path)

      m = cex_keras.keras.models.load_model(
          tmp_path,
          compile=True,
          custom_objects=dict(
              Bias=cex_keras.Bias,
              ColorSpace=cex_keras.ColorSpace,
          ),
      )

      logging.debug('')
      m.summary(print_fn=logging.debug)

      actual = m.predict(np.expand_dims(source, 0))[0]

      if target is not None:
        # try to get atol back to 8e-5
        assert actual.shape == target.shape
        assert np.allclose(actual, target,
                           atol=0.0007), 'failure in %s' % (name)

      if target is None:
        target = source.copy()

      # solve for a bias of 0.1
      r = m.fit(
          np.expand_dims(source + .1, 0),
          np.expand_dims(target, 0),
          epochs=10,
          verbose=int(logging.getLogger().level == logging.DEBUG),
      )

      mse = np.array(r.history['mean_squared_error'])
      assert np.isfinite(mse.mean())

  def test_cex_numpy_xyz_to_xyy(self):
    'check numpy xyz<>xyy roundtrip'
    cube_size = 16
    H = W = 64
    N = cube_size**3
    assert N == H * W
    int_cube = np.array(np.mgrid[:cube_size, :cube_size, :cube_size]).T
    rgb = int_cube.astype(np.float32)
    rgb /= float(cube_size - 1)
    rgb = rgb.reshape((H, W, 3))

    xyz = cex_numpy.rgb_to_xyz(rgb)
    xyy = cex_numpy.xyz_to_xyy(xyz)
    xyz2 = cex_numpy.xyy_to_xyz(xyy)

    assert np.allclose(xyz, xyz2)
