# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf


def x_entr( p_y_given_x_train, y_gt, weightPerClass, eps=1e-6 ):
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
    # weightPerClass is a vector with 1 element per class.
    
    #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
    log_p_y_given_x_train = tf.math.log( p_y_given_x_train + eps)
    
    weightPerClass5D = tf.reshape(weightPerClass, shape=[1, tf.shape(p_y_given_x_train)[1], 1, 1, 1])
    weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClass5D
    
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    
    num_samples = tf.cast( tf.reduce_prod( tf.shape(y_gt) ), "float32")
    
    return - (1./ num_samples) * tf.reduce_sum( weighted_log_p_y_given_x_train * y_one_hot )


#default custom loss
def iou(p_y_given_x_train, y_gt, eps=1e-5):
    # Intersection-Over-Union / Jaccard: https://en.wikipedia.org/wiki/Jaccard_index
    # Analysed in: Nowozin S, Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case, CVPR 2014
    # First computes IOU per class. Finally averages over the class-ious.
    # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
    # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]

    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    """
    ones_at_real_negs = tf.cast( tf.less(y_one_hot, 0.0001), dtype="float32") # tf.equal(y_one_hot,0), but less may be more stable with floats.
    numer = tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train * ones_at_real_negs, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    iou = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_iou = tf.reduce_mean(iou) # Along the class-axis. Mean DSC of classes.
    cost = 1. - av_class_iou
    """

    Channel1_weighted_log_p_y_given_x_train = p_y_given_x_train[:, 1, :, :, :]
    Channel1_y_one_hot = y_one_hot[:, 1, :, :, :]

    product = Channel1_weighted_log_p_y_given_x_train * Channel1_y_one_hot

    maxprod = tf.math.reduce_max(product, axis=[1, 2, 3])

    maxonehot = tf.math.reduce_max(Channel1_y_one_hot, axis=[1, 2, 3])

    resultTP = tf.divide(tf.math.reduce_sum(maxprod) + 1, tf.math.reduce_sum(maxonehot) + 1)

    # True negatives

    Channel0_y_one_hot = y_one_hot[:, 0, :, :, :]
    PatchesNegOneHot = tf.math.reduce_min(Channel0_y_one_hot, axis=[1, 2, 3])
    MaxPrbNeg = tf.multiply(PatchesNegOneHot, tf.math.reduce_max(p_y_given_x_train[:, 1, :, :, :], axis=[1, 2, 3]))

    resultTN = 1. - tf.divide(tf.math.reduce_sum(MaxPrbNeg) + 1, tf.math.reduce_sum(PatchesNegOneHot) + 1)

    # cost = 1. - ((0.004 * resultTN + 0.996 * resultTP))
    cost = 1. - ((0.05 * resultTN + 0.95 * resultTP))

    return cost


# def labels_to_one_hot(ground_truth, num_classes=1):
#     """
#     Converts ground truth labels to one-hot, sparse tensors.
#     Used extensively in segmentation losses.
#     :param ground_truth: ground truth categorical labels (rank `N`)
#     :param num_classes: A scalar defining the depth of the one hot dimension
#         (see `depth` of `tf.one_hot`)
#     :return: one-hot sparse tf tensor
#         (rank `N+1`; new axis appended at the end)
#     """
#     # read input/output shapes
#     if isinstance(num_classes, tf.Tensor):
#         num_classes_tf = tf.cast(num_classes, tf.int32)
#     else:
#         num_classes_tf = tf.constant(num_classes, tf.int32)
#     input_shape = tf.shape(ground_truth)
#     output_shape = tf.concat(
#         [input_shape, tf.reshape(num_classes_tf, (1,))], 0)
#
#     if num_classes == 1:
#         # need a sparse representation?
#         return tf.reshape(ground_truth, output_shape)
#
#     # squeeze the spatial shape
#     ground_truth = tf.reshape(ground_truth, (-1,))
#     # shape of squeezed output
#     dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)
#
#     # create a rank-2 sparse tensor
#     ground_truth = tf.cast(ground_truth, tf.int64)
#     ids = tf.range(tf.cast(dense_shape[0], tf.int64), dtype=tf.int64)
#     ids = tf.stack([ids, ground_truth], axis=1)
#     one_hot = tf.SparseTensor(
#         indices=ids,
#         values=tf.ones_like(ground_truth, dtype=tf.float32),
#         dense_shape=tf.cast(dense_shape, tf.int64))
#
#     # resume the spatial dims
#     one_hot = tf.sparse_reshape(one_hot, output_shape)
#     return one_hot

# # sensitivity specificity loss
# def iou(p_y_given_x_train, y_gt, eps=1e-5):
#     # Intersection-Over-Union / Jaccard: https://en.wikipedia.org/wiki/Jaccard_index
#     # Analysed in: Nowozin S, Optimal Decisions from Probabilistic Models: the Intersection-over-Union Case, CVPR 2014
#     # First computes IOU per class. Finally averages over the class-ious.
#     # p_y_given_x_train : tensor5 [batchSize, classes, r, c, z]
#     # y: T.itensor4('y'). Dimensions [batchSize, r, c, z]
#
#     y_one_hot = tf.one_hot(indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
#     """
#     ones_at_real_negs = tf.cast( tf.less(y_one_hot, 0.0001), dtype="float32") # tf.equal(y_one_hot,0), but less may be more stable with floats.
#     numer = tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
#     denom = tf.reduce_sum(p_y_given_x_train * ones_at_real_negs, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
#     iou = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
#     av_class_iou = tf.reduce_mean(iou) # Along the class-axis. Mean DSC of classes.
#     cost = 1. - av_class_iou
#     """
#
#     # Sensitivity part
#     channel1_weighted_log_p_y_given_x_train = p_y_given_x_train[:, 1, :, :, :]
#     channel1_y_one_hot = y_one_hot[:, 1, :, :, :]
#     diff_squre = tf.math.squared_difference(channel1_y_one_hot, channel1_weighted_log_p_y_given_x_train)
#     product1 = diff_squre * channel1_y_one_hot
#     sum1 = tf.math.reduce_sum(product1)
#     total_positive = tf.math.reduce_sum(channel1_y_one_hot)
#     sensitivity_part = tf.math.divide(sum1, total_positive + eps)
#
#     # Specificity part
#     one_cold = 1 - channel1_y_one_hot
#     product2 = diff_squre * one_cold
#     sum2 = tf.math.reduce_sum(product2)
#     total_negative = tf.math.reduce_sum(one_cold)
#     specificity_part = tf.math.divide(sum2, total_negative + eps)
#     r = 0.005
#     cost = ((1 - r) * specificity_part + r * sensitivity_part)
#
#     return cost

#
# def iou(p_y_given_x_train, y_gt, eps=1e-5):
#     prediction = tf.cast(p_y_given_x_train, tf.float32)
#     one_hot = labels_to_one_hot(y_gt, tf.shape(prediction)[-1])
#
#     one_hot = tf.sparse_tensor_to_dense(one_hot)
#     # value of unity everywhere except for the previous 'hot' locations
#     one_cold = 1 - one_hot
#
#     # chosen region may contain no voxels of a given label. Prevents nans.
#
#     squared_error = tf.square(one_hot - prediction)
#     specificity_part = tf.reduce_sum(
#         squared_error * one_hot, 0) / \
#                        (tf.reduce_sum(one_hot, 0) + eps)
#     sensitivity_part = \
#         (tf.reduce_sum(tf.multiply(squared_error, one_cold), 0) /
#          (tf.reduce_sum(one_cold, 0) + eps))
#     r = 0.5
#     cost = tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)
#     return cost


def dsc(p_y_given_x_train, y_gt, eps=1e-5):
    # Similar to Intersection-Over-Union / Jaccard above.
    # Dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    y_one_hot = tf.one_hot( indices=y_gt, depth=tf.shape(p_y_given_x_train)[1], axis=1, dtype="float32" )
    numer = 2. * tf.reduce_sum(p_y_given_x_train * y_one_hot, axis=(0,2,3,4)) # 2 * TP
    denom = tf.reduce_sum(p_y_given_x_train, axis=(0,2,3,4)) + tf.reduce_sum(y_one_hot, axis=(0,2,3,4)) # Pred + RP
    dsc = (numer + eps) / (denom + eps) # eps in both num/den => dsc=1 when class missing.
    av_class_dsc = tf.reduce_mean(dsc) # Along the class-axis. Mean DSC of classes. 
    cost = 1. - av_class_dsc
    return cost

def cost_L1(prms):
    # prms: list of tensors
    cost = 0
    for prm in prms:
        cost += tf.reduce_sum(tf.abs(prm))
    return cost

def cost_L2(prms) : #Called for L2 weigths regularisation
    # prms: list of tensors
    cost = 0
    for prm in prms:
        cost += tf.reduce_sum(prm ** 2)
    return cost
    

