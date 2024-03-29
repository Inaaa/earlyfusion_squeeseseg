# Author: Bichen Wu (bichen@berkeley.edu) 02/20/2017

"""SqueezeSeg model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeSeg(ModelSkeleton):
  def __init__(self, mc, gpu_id=0):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc)

      self._add_forward_graph()
      self._add_output_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

      self._add_summary_ops()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv0_0 = self._conv_layer(
        'conv0_0', self.lidar_input, filters=32, size=3, stride=2,
        padding='SAME', freeze=False, xavier=True
    )  ##32*384*1248*7  -> 32*192*624*32
    conv0_0_skip = self._conv_layer(
        'conv1_skip', self.lidar_input, filters=32, size=1, stride=1,
        padding='SAME', freeze=False, xavier=True) ##32*384*1248*32

    conv0_1 = self._conv_layer(
        'conv0_1', conv0_0, filters=64, size=3, stride=1,
        padding='SAME', freeze=False, xavier=True
    )##-> 32*192*624*64
    pool0 = self._pooling_layer(
        'pool1', conv0_1, size=3, stride1=2,stride2=2, padding='SAME')##->32*96*312*64


    conv1 = self._conv_layer(
        'conv1', pool0, filters=64, size=3, stride=1,
        padding='SAME', freeze=False, xavier=True)   ## ->32*96*312*64

    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride1=2, stride2=2, padding='SAME')  ## ->32*48*156*64

    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)  ##->32*48*156*128
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)  ##->32*48*156*128
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride1=1, stride2=2, padding='SAME')  ##->32*48*78*128

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False) ##32*48*78*256
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False) ##32*48*78*256
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride1=2,stride2=2, padding='SAME')  ##32*24*39*256

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)    ##32*24*39*384
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)   ##32*24*39*384
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)   ##32*24*39*512
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)   ##32*24*39*512

    # Deconvolation
    fire10 = self._fire_deconv(
        'fire_deconv10', fire9, s1x1=64, e1x1=128, e3x3=128, factors=[2, 2],
        stddev=0.1)  ##32*46*76*256
    fire10_fuse = tf.add(fire10, fire5, name='fure10_fuse')  #32*48*78*256


    fire11 = self._fire_deconv(
        'fire_deconv11', fire10_fuse, s1x1=32, e1x1=64, e3x3=64, factors=[1, 2],
        stddev=0.1)  #32*48*156*128
    fire11_fuse = tf.add(fire11, fire3, name='fire11_fuse')  #32*48*156*128

    fire12 = self._fire_deconv(
        'fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[2, 2],
        stddev=0.1) #32*96*312*64
    fire12_fuse = tf.add(fire12, conv1, name='fire12_fuse')  ##32*96*312*64

    fire13 = self._fire_deconv(
        'fire_deconv13', fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[2, 2],
        stddev=0.1)  ##32*192*624*64
    fire13_fuse = tf.add(fire13, conv0_1, name='fire13_fuse') ##32*192*624*64

    fire14 = self._fire_deconv(
        'fire_deconv14', fire13_fuse, s1x1=16, e1x1=16, e3x3=16, factors=[2, 2],
        stddev=0.1)  ##32*384*1248*32
    fire14_fuse = tf.add(fire14, conv0_0_skip, name='fire14_fuse') ##32*384*1248*32

    drop14 = tf.nn.dropout(fire14_fuse, self.keep_prob, name='drop14')  ##32*384*1248*32


    self.output_prob = self._conv_layer(
        'out_prob', drop14, filters=mc.NUM_CLASS, size=3, stride=1,
        padding='SAME', relu=False, stddev=0.1)  ####32*384*1248*2


  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.001,
      freeze=False):
    """Fire layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

  def _fire_deconv(self, layer_name, inputs, s1x1, e1x1, e3x3, 
                   factors=[1, 2], freeze=False, stddev=0.001):
    """Fire deconvolution layer constructor.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    assert len(factors) == 2,'factors should be an array of size 2'

    ksize_h = factors[0] * 2 - factors[0] % 2
    ksize_w = factors[1] * 2 - factors[1] % 2

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)
    deconv = self._deconv_layer(
        layer_name+'/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
        stride=factors, padding='SAME', init='bilinear')
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', deconv, filters=e1x1, size=1, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', deconv, filters=e3x3, size=3, stride=1,
        padding='SAME', freeze=freeze, stddev=stddev)

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
