# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''
Lookup table ops forked and modified from:
https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/interpolation.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util

__all__ = ['nd_lookup_table']


def nd_lookup_table(
    x,
    x_ref_min,
    x_ref_max,
    y_ref,
    axis,
    fill_value='constant_extension',
    name=None,
):
  '''Multi-linear interpolation on a regular (constant spacing) grid.

  Given [a batch of] reference values, this function computes a multi-linear
  interpolant and evaluates it on [a batch of] of new `x` values.

  The interpolant is built from reference values indexed by `nd` dimensions
  of `y_ref`, starting at `axis`.

  For example, take the case of a `2-D` scalar valued function and no leading
  batch dimensions.  In this case, `y_ref.shape = [C1, C2]` and `y_ref[i, j]`
  is the reference value corresponding to grid point

  ```
  [x_ref_min[0] + i * (x_ref_max[0] - x_ref_min[0]) / (C1 - 1),
   x_ref_min[1] + j * (x_ref_max[1] - x_ref_min[1]) / (C2 - 1)]
  ```

  In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
  with leading dimensions in `x`, `x_ref_min`, `x_ref_max`.

  Args:
    x: Numeric `Tensor` The x-coordinates of the interpolated output values for
      each batch.  Shape `[..., D, nd]`, designating [a batch of] `D`
      coordinates in `nd` space.  `D` must be `>= 1` and is not a batch dim.
    x_ref_min:  `Tensor` of same `dtype` as `x`.  The minimum values of the
      (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
    x_ref_max:  `Tensor` of same `dtype` as `x`.  The maximum values of the
      (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
    y_ref:  `Tensor` of same `dtype` as `x`.  The reference output values. Shape
      `[..., C1, ..., Cnd, B1,...,BM]`, designating [a batch of] reference
      values indexed by `nd` dimensions, of a shape `[B1,...,BM]` valued
      function (for `M >= 0`).
    axis:  Scalar integer `Tensor`.  Dimensions `[axis, axis + nd)` of `y_ref`
      index the interpolation table.  E.g. `3-D` interpolation of a scalar
      valued function requires `axis=-3` and a `3-D` matrix valued function
      requires `axis=-5`.
    fill_value:  Determines what values output should take for `x` values that
      are below `x_ref_min` or above `x_ref_max`. Scalar `Tensor` or
      'constant_extension' ==> Extend as constant function.
      Default value: `'constant_extension'`
    name:  A name to prepend to created ops.
      Default value: `'batch_interp_regular_nd_grid'`.

  Returns:
    y_interp:  Interpolation between members of `y_ref`, at points `x`.
      `Tensor` of same `dtype` as `x`, and shape `[..., D, B1, ..., BM].`

  Raises:
    ValueError:  If `rank(x) < 2` is determined statically.
    ValueError:  If `axis` is not a scalar is determined statically.
    ValueError:  If `axis + nd > rank(y_ref)` is determined statically.

  #### Examples

  Interpolate a function of one variable.

  ```python
  y_ref = tf.exp(tf.linspace(start=0., stop=10., num=20))

  tfp.math.batch_interp_regular_nd_grid(
      # x.shape = [3, 1], x_ref_min/max.shape = [1].  Trailing `1` for `1-D`.
      x=[[6.0], [0.5], [3.3]], x_ref_min=[0.], x_ref_max=[10.], y_ref=y_ref,
      axis=0)
  ==> approx [exp(6.0), exp(0.5), exp(3.3)]
  ```

  Interpolate a scalar function of two variables.

  ```python
  x_ref_min = [0., 0.]
  x_ref_max = [2 * np.pi, 2 * np.pi]

  # Build y_ref.
  x0s, x1s = tf.meshgrid(
      tf.linspace(x_ref_min[0], x_ref_max[0], num=100),
      tf.linspace(x_ref_min[1], x_ref_max[1], num=100),
      indexing='ij')

  def func(x0, x1):
    return tf.sin(x0) * tf.cos(x1)

  y_ref = func(x0s, x1s)

  x = np.pi * tf.random.uniform(shape=(10, 2))

  tfp.math.batch_interp_regular_nd_grid(x, x_ref_min, x_ref_max, y_ref, axis=-2)
  ==> tf.sin(x[:, 0]) * tf.cos(x[:, 1])
  ```

  '''
  with tf.name_scope(name or 'interp_regular_nd_grid'):
    dtype = dtype_util.common_dtype([x, x_ref_min, x_ref_max, y_ref],
                                    dtype_hint=tf.float32)

    # Arg checking.
    if isinstance(fill_value, str):
      if fill_value != 'constant_extension':
        raise ValueError(
            'A fill value ({}) was not an allowed string ({})'.format(
                fill_value, 'constant_extension'))
    else:
      fill_value = tf.convert_to_tensor(fill_value,
                                        name='fill_value',
                                        dtype=dtype)
      _assert_ndims_statically(fill_value, expect_ndims=0)

    # x.shape = [..., nd].
    x = tf.convert_to_tensor(x, name='x', dtype=dtype)
    _assert_ndims_statically(x, expect_ndims_at_least=2)

    # y_ref.shape = [..., C1,...,Cnd, B1,...,BM]
    y_ref = tf.convert_to_tensor(y_ref, name='y_ref', dtype=dtype)

    # x_ref_min.shape = [nd]
    x_ref_min = tf.convert_to_tensor(x_ref_min, name='x_ref_min', dtype=dtype)
    x_ref_max = tf.convert_to_tensor(x_ref_max, name='x_ref_max', dtype=dtype)
    _assert_ndims_statically(x_ref_min,
                             expect_ndims_at_least=1,
                             expect_static=True)
    _assert_ndims_statically(x_ref_max,
                             expect_ndims_at_least=1,
                             expect_static=True)

    # nd is the number of dimensions indexing the interpolation table, it's the
    # 'nd' in the function name.
    nd = tf.compat.dimension_value(x_ref_min.shape[-1])
    if nd is None:
      raise ValueError('`x_ref_min.shape[-1]` must be known statically.')
    tensorshape_util.assert_is_compatible_with(x_ref_max.shape[-1:],
                                               x_ref_min.shape[-1:])

    # Convert axis and check it statically.
    axis = tf.convert_to_tensor(axis, dtype=tf.int32, name='axis')
    axis = prefer_static.non_negative_axis(axis, tf.rank(y_ref))
    tensorshape_util.assert_has_rank(axis.shape, 0)
    axis_ = tf.get_static_value(axis)
    y_ref_rank_ = tf.get_static_value(tf.rank(y_ref))
    if axis_ is not None and y_ref_rank_ is not None:
      if axis_ + nd > y_ref_rank_:
        raise ValueError(
            'Since dims `[axis, axis + nd)` index the interpolation table, we '
            'must have `axis + nd <= rank(y_ref)`.  Found: '
            '`axis`: {},  rank(y_ref): {}, and inferred `nd` from trailing '
            'dimensions of `x_ref_min` to be {}.'.format(
                axis_, y_ref_rank_, nd))

    x_batch_shape = tf.shape(x)[:-2]
    x_ref_min_batch_shape = tf.shape(x_ref_min)[:-1]
    x_ref_max_batch_shape = tf.shape(x_ref_max)[:-1]
    y_ref_batch_shape = tf.shape(y_ref)[:axis]

    # Do a brute-force broadcast of batch dims (add zeros).
    batch_shape = y_ref_batch_shape
    for tensor in [x_batch_shape, x_ref_min_batch_shape, x_ref_max_batch_shape]:
      batch_shape = tf.broadcast_dynamic_shape(batch_shape, tensor)

    def _batch_of_zeros_with_rightmost_singletons(n_singletons):
      '''Return Tensor of zeros with some singletons on the rightmost dims.'''
      ones = tf.ones(shape=[n_singletons], dtype=tf.int32)
      return tf.zeros(shape=tf.concat([batch_shape, ones], axis=0), dtype=dtype)

    x += _batch_of_zeros_with_rightmost_singletons(n_singletons=2)
    x_ref_min += _batch_of_zeros_with_rightmost_singletons(n_singletons=1)
    x_ref_max += _batch_of_zeros_with_rightmost_singletons(n_singletons=1)
    y_ref += _batch_of_zeros_with_rightmost_singletons(
        n_singletons=tf.rank(y_ref) - axis)

    return _batch_interp_with_gather_nd(
        x=x,
        x_ref_min=x_ref_min,
        x_ref_max=x_ref_max,
        y_ref=y_ref,
        nd=nd,
        fill_value=fill_value,
        batch_dims=tf.get_static_value(tf.rank(x)) - 2)


def _batch_interp_with_gather_nd(
    x,
    x_ref_min,
    x_ref_max,
    y_ref,
    nd,
    fill_value,
    batch_dims,
):
  '''N-D interpolation that works with leading batch dims.'''
  dtype = x.dtype

  # In this function,
  # x.shape = [A1, ..., An, D, nd], where n = batch_dims
  # and
  # y_ref.shape = [A1, ..., An, C1, C2,..., Cnd, B1,...,BM]
  # y_ref[A1, ..., An, i1,...,ind] is a shape [B1,...,BM] Tensor with the value
  # at index [i1,...,ind] in the interpolation table.
  #  and x_ref_max have shapes [A1, ..., An, nd].

  # ny[k] is number of y reference points in interp dim k.
  ny = tf.cast(tf.shape(y_ref)[batch_dims:batch_dims + nd], dtype)

  # Map [x_ref_min, x_ref_max] to [0, ny - 1].
  # This is the (fractional) index of x.
  # x_idx_unclipped[A1, ..., An, d, k] is the fractional index into dim k of
  # interpolation table for the dth x value.
  x_ref_min_expanded = tf.expand_dims(x_ref_min, axis=-2)
  x_ref_max_expanded = tf.expand_dims(x_ref_max, axis=-2)
  x_idx_unclipped = (ny - 1) * (x - x_ref_min_expanded) / (x_ref_max_expanded -
                                                           x_ref_min_expanded)

  # Wherever x is NaN, x_idx_unclipped will be NaN as well.
  # Keep track of the nan indices here (so we can impute NaN later).
  # Also eliminate any NaN indices, since there is not NaN in 32bit.
  nan_idx = tf.math.is_nan(x_idx_unclipped)
  x_idx_unclipped = tf.where(nan_idx, tf.cast(0., dtype=dtype), x_idx_unclipped)

  # x_idx.shape = [A1, ..., An, D, nd]
  x_idx = tf.clip_by_value(x_idx_unclipped, tf.zeros((), dtype=dtype), ny - 1)

  # Get the index above and below x_idx.
  # Naively we could set idx_below = floor(x_idx), idx_above = ceil(x_idx),
  # however, this results in idx_below == idx_above whenever x is on a grid.
  # This in turn results in y_ref_below == y_ref_above, and then the gradient
  # at this point is zero.  So here we 'jitter' one of idx_below, idx_above,
  # so that they are at different values.  This jittering does not affect the
  # interpolated value, but does make the gradient nonzero (unless of course
  # the y_ref values are the same).
  idx_below = tf.floor(x_idx)
  idx_above = tf.minimum(idx_below + 1, ny - 1)
  idx_below = tf.maximum(idx_above - 1, 0)

  # These are the values of y_ref corresponding to above/below indices.
  # idx_below_int32.shape = x.shape[:-1] + [nd]
  idx_below_int32 = tf.cast(idx_below, dtype=tf.int32)
  idx_above_int32 = tf.cast(idx_above, dtype=tf.int32)

  # idx_below_list is a length nd list of shape x.shape[:-1] int32 tensors.
  idx_below_list = tf.unstack(idx_below_int32, axis=-1)
  idx_above_list = tf.unstack(idx_above_int32, axis=-1)

  # Use t to get a convex combination of the below/above values.
  # t.shape = [A1, ..., An, D, nd]
  t = x_idx - idx_below

  # x, and tensors shaped like x, need to be added to, and selected with
  # (using tf.where) the output y.  This requires appending singletons.
  def _expand_x_fn(tensor):
    # Reshape tensor to tensor.shape + [1] * M.
    extended_shape = tf.concat(
        [tf.shape(tensor),
         tf.ones_like(tf.shape(y_ref)[batch_dims + nd:])],
        axis=0)
    return tf.reshape(tensor, extended_shape)

  # Now, t.shape = [A1, ..., An, D, nd] + [1] * (rank(y_ref) - nd - batch_dims)
  t = _expand_x_fn(t)
  s = 1 - t

  # Re-insert NaN wherever x was NaN.
  nan_idx = _expand_x_fn(nan_idx)
  t = tf.where(nan_idx, tf.constant(np.nan, dtype), t)

  terms = []
  # Our work above has located x's fractional index inside a cube of above/below
  # indices. The distance to the below indices is t, and to the above indices
  # is s.
  # Drawing lines from x to the cube walls, we get 2**nd smaller cubes. Each
  # term in the result is a product of a reference point, gathered from y_ref,
  # multiplied by a volume.  The volume is that of the cube opposite to the
  # reference point.  E.g. if the reference point is below x in every axis, the
  # volume is that of the cube with corner above x in every axis, s[0]*...*s[nd]
  # We could probably do this with one massive gather, but that would be very
  # unreadable and un-debuggable.  It also would create a large Tensor.
  for zero_ones_list in _binary_count(nd):
    gather_from_y_ref_idx = []
    opposite_volume_t_idx = []
    opposite_volume_s_idx = []
    for k, zero_or_one in enumerate(zero_ones_list):
      if zero_or_one == 0:
        # If the kth iterate has zero_or_one = 0,
        # Will gather from the 'below' reference point along axis k.
        gather_from_y_ref_idx.append(idx_below_list[k])
        # Now append the index to gather for computing opposite_volume.
        # This could be done by initializing opposite_volume to 1, then here:
        #  opposite_volume *= tf.gather(s, indices=k, axis=tf.rank(x) - 1)
        # but that puts a gather in the 'inner loop.'  Better to append the
        # index and do one larger gather down below.
        opposite_volume_s_idx.append(k)
      else:
        gather_from_y_ref_idx.append(idx_above_list[k])
        # Append an index to gather, having the same effect as
        #   opposite_volume *= tf.gather(t, indices=k, axis=tf.rank(x) - 1)
        opposite_volume_t_idx.append(k)

    # Compute opposite_volume (volume of cube opposite the ref point):
    # Recall t.shape = s.shape = [D, nd] + [1, ..., 1]
    # Gather from t and s along the 'nd' axis, which is rank(x) - 1.
    ov_axis = tf.rank(x) - 1
    opposite_volume = (tf.reduce_prod(
        tf.gather(
            t,
            indices=tf.cast(opposite_volume_t_idx, dtype=tf.int32),
            axis=ov_axis,
        ),
        axis=ov_axis,
    ) * tf.reduce_prod(
        tf.gather(
            s,
            indices=tf.cast(opposite_volume_s_idx, dtype=tf.int32),
            axis=ov_axis,
        ),
        axis=ov_axis,
    ))

    y_ref_pt = tf.gather_nd(
        y_ref,
        tf.stack(gather_from_y_ref_idx, axis=-1),
        batch_dims=batch_dims,
    )

    terms.append(y_ref_pt * opposite_volume)

  y = tf.math.add_n(terms)

  if tf.debugging.is_numeric_tensor(fill_value):
    # Recall x_idx_unclipped.shape = [D, nd],
    # so here we check if it was out of bounds in any of the nd dims.
    # Thus, oob_idx.shape = [D].
    oob_idx = tf.reduce_any(
        (x_idx_unclipped < 0) | (x_idx_unclipped > ny - 1),
        axis=-1,
    )

    # Now, y.shape = [D, B1,...,BM], so we'll have to broadcast oob_idx.

    oob_idx = _expand_x_fn(oob_idx)  # Shape [D, 1,...,1]
    oob_idx |= tf.fill(tf.shape(y), False)
    y = tf.where(oob_idx, fill_value, y)
  return y


def _assert_ndims_statically(
    x,
    expect_ndims=None,
    expect_ndims_at_least=None,
    expect_static=False,
):
  '''Assert that Tensor x has expected number of dimensions.'''
  ndims = tensorshape_util.rank(x.shape)
  if ndims is None:
    if expect_static:
      raise ValueError('Expected static ndims. Found: {}'.format(x))
    return
  if expect_ndims is not None and ndims != expect_ndims:
    raise ValueError('ndims must be {}.  Found: {}'.format(expect_ndims, ndims))
  if expect_ndims_at_least is not None and ndims < expect_ndims_at_least:
    raise ValueError('ndims must be at least {}. Found {}'.format(
        expect_ndims_at_least, ndims))


def _make_expand_x_fn_for_non_batch_interpolation(y_ref, axis):
  '''Make func to expand left/right (of axis) dims of tensors shaped like x.'''
  # This expansion is to help x broadcast with `y`, the output.
  # In the non-batch case, the output shape is going to be
  #   y_ref.shape[:axis] + x.shape + y_ref.shape[axis+1:]

  # Recall we made axis non-negative
  y_ref_shape = tf.shape(y_ref)
  y_ref_shape_left = y_ref_shape[:axis]
  y_ref_shape_right = y_ref_shape[axis + 1:]

  def expand_ends(x, broadcast=False):
    '''Expand x so it can bcast w/ tensors of output shape.'''
    # Assume out_shape = A + x.shape + B, and rank(A) = axis.
    # Expand with singletons with same rank as A, B.
    expanded_shape = tf.pad(
        tensor=tf.shape(x),
        paddings=[[axis, tf.size(y_ref_shape_right)]],
        constant_values=1,
    )
    x_expanded = tf.reshape(x, expanded_shape)

    if broadcast:
      out_shape = tf.concat(
          (
              y_ref_shape_left,
              tf.shape(x),
              y_ref_shape_right,
          ),
          axis=0,
      )
      if dtype_util.is_bool(x.dtype):
        x_expanded = x_expanded | tf.cast(tf.zeros(out_shape), tf.bool)
      else:
        x_expanded += tf.zeros(out_shape, dtype=x.dtype)
    return x_expanded

  return expand_ends


def _binary_count(n):
  '''Count `n` binary digits from [0...0] to [1...1].'''
  return list(itertools.product([0, 1], repeat=n))
