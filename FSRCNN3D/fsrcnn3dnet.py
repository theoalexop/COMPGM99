# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.cubic_resize import CubicResizeLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer

class FSRCNN3D(BaseNet):
    """
    Implementation of FSRCNN [1] with 3D Kernel Spatial Support, based on NiftyNet [2]. 
    This implementation utilizes highres3dnet.py [3] as template. 
    [1] Dong et al., "C. Dong et al. Accelerating the Super-Resolution Convolutional Neural Network". 
    In Proceedings of European Conference on Computer Vision (ECCV), 2016. 
    [2] https://github.com/NifTK/NiftyNet
    [3] https://github.com/NifTK/NiftyNet/blob/dev/niftynet/network/highres3dnet.py
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='FSRCNN3D'):

        super(FSRCNN3D, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 56, 'kernel_size': 5},
            {'name': 'conv_1', 'n_features': 12, 'kernel_size': 1},
            {'name': 'conv_2', 'n_features': 12, 'kernel_size': 3, 'repeat':4},
            {'name': 'conv_3', 'n_features': 56, 'kernel_size': 1},
            {'name': 'deconv', 'n_features': num_classes, 'kernel_size': 9}]

    def layer_op(self,
                 images,
                 is_training=True,
                 layer_id=-1,
                 **unused_kwargs):
        assert (layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0))
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []

        images2 = CubicResizeLayer((16,16,16))(images)

        params = self.layers[0]
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = first_conv_layer(images2, is_training)
        layer_instances.append((first_conv_layer, flow))

        params = self.layers[1]
        conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = conv_layer(flow, is_training)
        layer_instances.append((conv_layer, flow))

        params = self.layers[2]
        for j in range(params['repeat']):
            conv_layer = ConvolutionalLayer(
                n_output_chns=params['n_features'],
                kernel_size=params['kernel_size'],
                with_bias=True,
                with_bn=False,
                acti_func=self.acti_func,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='%s_%d' % (params['name'], j))
            flow = conv_layer(flow, is_training)
            layer_instances.append((conv_layer, flow))

        params = self.layers[3]
        conv_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = conv_layer(flow, is_training)
        layer_instances.append((conv_layer, flow))

        params = self.layers[4]
        deconv_layer = DeconvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            stride=2,
            padding='SAME',
            with_bias=True,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = deconv_layer(flow, is_training)
        layer_instances.append((deconv_layer, flow))

        if is_training:
            self._print(layer_instances)
            return layer_instances[-1][1]
        return layer_instances[layer_id][1]

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
