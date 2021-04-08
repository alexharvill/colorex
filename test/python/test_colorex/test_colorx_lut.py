# Copyright 2021 Alex Harvill
# SPDX-License-Identifier: Apache-2.0
'test colorex lookup table op'

import unittest
import numpy as np

cex_lut = None
cex_keras = None
tfp = None


class TestColorexLUT(unittest.TestCase):
  'to be run via unittest discovery'

  def setUp(self):
    '''
    when using unittest discovery to find all tests in a git repo
    importing keras at the top of a test can slow everything down
    delay importing keras to cut down on startup time for other tests
    '''
    #pylint: disable=global-statement,import-outside-toplevel
    import colorex.cex_keras
    import colorex.cex_lut
    import tensorflow_probability

    global cex_lut
    global cex_keras
    global tfp
    cex_keras = colorex.cex_keras
    cex_lut = colorex.cex_lut
    tfp = tensorflow_probability

  def tearDown(self):
    cex_keras.K.clear_session()

  def test_cex_lut_vs_tfp(self):
    'check cex lut vs a brute force loop based tfp version'

    N, HW, C = 32, 64, 3
    D = 16

    np.random.seed(0)
    batched_luts = np.random.random(size=(N, D, D, D, C)).astype(np.float32)
    batched_xs = np.random.random(size=(N, HW, C)).astype(np.float32)

    batched_values = cex_lut.color_lookup_table(
        x=batched_xs,
        x_ref_min=np.zeros((N, C), dtype=np.float32),
        x_ref_max=np.ones((N, C), dtype=np.float32),
        y_ref=batched_luts,
        fill_value='constant_extension',
        name='batched',
    ).numpy()

    loop_values = np.zeros((N, HW, C), dtype=np.float32)
    for b_idx, (lut, xs) in enumerate(zip(batched_luts, batched_xs)):
      for hw_idx, x in enumerate(xs):
        loop_values[b_idx, hw_idx] = tfp.math.batch_interp_regular_nd_grid(
            x=x.reshape((1, C)),
            x_ref_min=np.zeros((C), dtype=np.float32),
            x_ref_max=np.ones((C), dtype=np.float32),
            y_ref=lut,
            axis=-4,
            fill_value='constant_extension',
            name='loopN',
        ).numpy()

    assert np.allclose(loop_values, batched_values)
