# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

"""
A variant of HighRes3DNet with an extra convolutional layer encompassing dropout is 
originally found in [1]. 
The modification incorporated to the aforementioned variant pertain to: 1) the addition 
of a main identity map connection from end-to-end and 2) batch normalization layers are 
removed, while bias is enabled to all convolutional layers.
[1] https://github.com/NifTK/NiftyNet/blob/dev/niftynet/network/highres3dnet_large.py
"""

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.network.base_net import BaseNet
from niftynet.network.highres3dnet import HighResBlock
from niftynet.layer.elementwise import ElementwiseLayer

class HighRes3DNetLarge(BaseNet):

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='HighRes3DNet'):

        super(HighRes3DNetLarge, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 16, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 16, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_2', 'n_features': 32, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'res_3', 'n_features': 64, 'kernels': (3, 3), 'repeat': 3},
            {'name': 'conv_1', 'n_features': 64, 'kernel_size': 3},
            {'name': 'conv_2', 'n_features': 64, 'kernel_size': 1},
            {'name': 'conv_3', 'n_features': num_classes, 'kernel_size': 1}]

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
        input_tensor_res = images

        ### first convolution layer
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
        flow = first_conv_layer(images, is_training)
        layer_instances.append((first_conv_layer, flow))

        ### resblocks, all kernels dilated by 1 (normal convolution)
        params = self.layers[1]
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 2
        params = self.layers[2]
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### resblocks, all kernels dilated by 4
        params = self.layers[3]
        with DilatedTensor(flow, dilation_factor=4) as dilated:
            for j in range(params['repeat']):
                res_block = HighResBlock(
                    params['n_features'],
                    params['kernels'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % (params['name'], j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        ### 3x3x3 convolution layer
        params = self.layers[4]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        ### 1x1x1 convolution layer
        params = self.layers[5]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training, keep_prob = 0.9)
        layer_instances.append((fc_layer, flow))

        ### 1x1x1 convolution layer
        params = self.layers[6]
        fc_layer = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            with_bias=True,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        flow = fc_layer(flow, is_training)
        layer_instances.append((fc_layer, flow))

        output_tensor_res = ElementwiseLayer('SUM')(input_tensor_res, flow)

        # set training properties
        if is_training:
            self._print(layer_instances)
            # return layer_instances[-1][1]
            return output_tensor_res
        # return layer_instances[layer_id][1]
        return output_tensor_res

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)
