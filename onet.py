# -*- coding: utf-8 -*-
"""The model is based on Chemnet which can be trained to predict various properties of
small organic molecules, including regression and classification.


# Reference
- [A Deep Neural Network with Minimal Chemistry Knowledge Matches the Performance of 
Expert-developed QSAR/QSPR Models](https://arxiv.org/abs/1706.06689)

"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,Concatenate,Dropout
from keras.layers import GlobalAveragePooling2D, Input, Flatten, MaxPooling2D,Lambda,GlobalMaxPooling2D
#from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
#import os
#import pickle
#from split_training_and_test_set import grab_dataset # split_training_and_test_set.py



batch_size = 32
epochs     = 10
filters = 16 # the number of filters in each layer of reduction block or inception-resnet block
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.7, epsilon=1e-8, decay=0.0)
data_augmentation = True

def grab_dataset(filename):
    '''
    Reload dataset from a previously dumped file
    '''	
    #import pickle
    with open(filename,'rb') as fo:
        z = pickle.load(fo,encoding='bytes')
    return z


def conv2d(x,
           filters,
           kernel_size,
           strides=1,
           padding='same',
           activation='relu',
           use_bias=False,
           name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == "channels_first" else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis,scale=False,name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def inception_resnet_block(x,
                           scale,
                           block_type,
                           block_idx,
                           N=filters,
                           activation='relu'):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='irA'`
        - Inception-ResNet-B: `block_type='irB'`
        - Inception-ResNet-C: `block_type='irC'`

    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch. Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'irA'`, `'irB'` or `'irC'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='irA', block_idx=0`, ane the layer names will have
            a common prefix `'irA_0'`.
        activation: activation function to use at the end of the block
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
        N: scaling factor to scale the number of filters for each  Conv2D layer inside
            an Inception-Resnet Block.
    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'irA'`,
            `'irB'` or `'irC'`.
    """
    if  N is not int:
    	N = int(N)

    if block_type == 'irA':
        branch_0 = conv2d(x,N, 1)
        branch_1 = conv2d(x,N, 1)
        branch_1 = conv2d(branch_1,N, 3)
        branch_2 = conv2d(x,N, 1)
        branch_2 = conv2d(branch_2,int(1.5*N), 3)
        branch_2 = conv2d(branch_2,int(2.0*N), 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'irB':
        branch_0 = conv2d(x,N, 1)
        branch_1 = conv2d(x,N, 1)
        branch_1 = conv2d(branch_1,int(1.25*N), (1,7))
        branch_1 = conv2d(branch_1,int(1.25*N), (7,1))
        branches = [branch_0, branch_1]
    elif block_type == 'irC':
        branch_0 = conv2d(x,N, 1)
        branch_1 = conv2d(x,N, 1)
        branch_1 = conv2d(branch_1,int(1.167*N), (1,3))
        branch_1 = conv2d(branch_1,int(1.334*N), (3,1))
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "irA", "irB" or "irC", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d(x=mixed,
                filters=K.int_shape(x)[channel_axis],
                kernel_size=1,
                activation=None,
                use_bias=True,
                name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    # keras.layers.Lambda(lambda...) can warp arbitrary expressions into a layer.
    # x = Lambda... sum up two parts: the previous x and 'scanned' x.
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x             


def reduction_block(x,block_type,block_idx,activation='relu',N=filters):
    """Adds a Inception-ResNet block.

    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='rA'`
        - Inception-ResNet-B: `block_type='rB'`

    # Arguments
        x: input tensor.
        block_type: `'rA'`, `'rB'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
            are repeated many times in this network. We use `block_idx` to identify
            each of the repetitions. For example, the first Inception-ResNet-A block
            will have `block_type='irA', block_idx=0`, ane the layer names will have
            a common prefix `'irA_0'`.
        activation: activation function to use at the end of the block
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
        N: scaling factor to scale the number of filters for each  conv2d layer inside
            an Inception-Resnet Block.
    # Returns
        Output tensor for the block.

    # Raises
        ValueError: if `block_type` is not one of `'irA'`,
            `'irB'` or `'irC'`.
    """
    if block_type == 'rA':
        branch_0 = MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(x)
        branch_1 = conv2d(x,filters=int(1.5*N),kernel_size=(3,3),strides=2,padding='valid')
        branch_2 = conv2d(x,N, 1)
        branch_2 = conv2d(branch_2,N, 3)
        branch_2 = conv2d(branch_2,int(1.5*N),3,2,padding='valid')
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'rB':
        branch_0 = MaxPooling2D(pool_size=(3,3),strides=2,padding='valid')(x)
        branch_1 = conv2d(x,N,1)
        branch_1 = conv2d(branch_1,int(1.5*N), (3,3),strides=2,padding='valid')
        branch_2 = conv2d(x,N, 1)
        branch_2 = conv2d(branch_2,int(1.126*N), (3,3),strides=2,padding='valid')
        branch_3 = conv2d(x,N, 1)
        branch_3 = conv2d(branch_3,int(1.126*N), (3,1))
        branch_3 = conv2d(branch_3,int(1.251*N), (3,1),strides=2,padding='valid')        
        branches = [branch_0, branch_1, branch_2]
    else:
        raise ValueError('Unknown Inception-ResNet block type.'
                         'Expects "rA", "rB" or "rC", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)

    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(mixed)
    return x


# img_input to be defined
img_input = Input(shape=(80,80,1))

x = conv2d(img_input, 32, 4, strides=2,padding='valid')
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irA',
                           block_idx=1,
                           N=filters)
'''
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irA',
                           block_idx=2,
                           N=filters)

x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irA',
                           block_idx=1,
                           N=filters) '''
x = reduction_block(x,
                    block_type='rA',
                    block_idx=1,
                    N=filters)
'''
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irB',
                           block_idx=1,
                           N=filters)
                           '''
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irB',
                           block_idx=2,
                           N=filters)
x = reduction_block(x,
                    block_type='rB',
                    block_idx=1,
                    N=filters)
'''
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irC',
                           block_idx=1,
                           N=filters)
'''
x = inception_resnet_block(x,
                           scale=1.0,
                           block_type='irC',
                           block_idx=2,
                           N=filters)      
x = GlobalMaxPooling2D()(x)
#x = Flatten()(x)
#x = Dense(16,activation=None,use_bias=True)(x)
#x = Dropout(0.25)(x)
y = Dense(1,activation=None,use_bias=True)(x)


model = keras.models.Model(inputs=img_input,outputs=y)
model.compile(optimizer=optimizer,loss='mean_squared_error')



x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")



if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=180,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        workers=4)


scores = model.evaluate(x_test,y_test,verbose=1)
print("scores:",scores)