# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.elementwise import ElementwiseLayer

class VDSR3D(BaseNet):
    """
    Implementation of VDSR [1] with 3D Kernel Spatial Support.

    J. Kim et al., "Accurate Image Super-Resolution Using Very Deep Convolutional Networks". 
    In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1646-1654, 
    June 2016.
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='VDSR3D'):

        super(VDSR3D, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'conv_0', 'n_features': 64, 'kernel_size': 3},
            {'name': 'conv_1', 'n_features': 64, 'kernel_size': 3, 'repeat':18},
            {'name': 'conv_2', 'n_features': num_classes, 'kernel_size': 3}]

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

        ### 
        params = self.layers[1]
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

        ### 
        params = self.layers[2]
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
