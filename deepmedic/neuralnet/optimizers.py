# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import tensorflow as tf

# Abstract
class Optimizer(object):
    def __init__(self, params_to_opt):
        self._params_to_opt = params_to_opt
        self._initialize_vars()
    
    # Abstract
    def _initialize_vars(self):
        raise NotImplementedError("Not implemented virtual function.")
    
    # Abstract
    def get_update_ops_given_grads(self):
        raise NotImplementedError("Not implemented virtual function.")
    
    # No need to use. Compute outside, and pass to _get_update_ops
    def get_grads_for_params_responsible(self, cost):
        # create a list of gradients for all parameters that this is optimizing
        return tf.gradients(cost, self._params_to_opt)
    
    def get_update_ops_given_cost(self, cost) :
        grads = self.get_grads_for_params_responsible(cost)
        return self.get_update_ops_given_grads(grads)

    def get_update_ops_task(self):
        raise NotImplementedError("Not implemented virtual function.")
    
class SgdOptimizer(Optimizer):
    def __init__(self,
                 params_to_opt,
                 learning_rate,
                 momentum,
                 momentumTypeNONNormalized0orNormalized1,
                 classicMomentum0OrNesterov1):
        
        self.name = "SgdOptimizer"
        
        self._learning_rate = learning_rate # tf.var
        self._momentum = momentum # tf.var
        self._momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self._classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        
        self._velocities_for_mom = None # list  tf.var
        
        Optimizer.__init__(self, params_to_opt)
        
    def _initialize_vars(self):
        self._velocities_for_mom = []
        for param in self._params_to_opt :
            self._velocities_for_mom.append( tf.Variable(param * 0., dtype="float32", name="velocities_for_mom") )
            
    # Mostly call get_update_ops_given_cost. This will be used in case I want to monitor grads, computed externally and passed here.
    def get_update_ops_given_grads(self, grads) :
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self._momentum * self._momentumTypeNONNormalized0orNormalized1
        
        for param, grad, v in zip(self._params_to_opt, grads, self._velocities_for_mom) :
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * self._learning_rate * grad
            newVelocity = self._momentum * v - stepToGradientDirection
            
            if self._classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self._momentum * newVelocity - stepToGradientDirection
                
            updates.append( tf.compat.v1.assign(ref=v, value=newVelocity, validate_shape=True) )  # I can do (1-mom)*learnRate*grad.
            updates.append( tf.compat.v1.assign(ref=param, value=param+updateToParam, validate_shape=True) )
            
        return updates
    
    
class AdamOptimizer(Optimizer):
    def __init__(self,
                 params_to_opt,
                 learning_rate,
                 b1_adam,
                 b2_adam,
                 eps):
        
        self.name = "AdamOptimizer"
        
        self._learning_rate = learning_rate
        self._b1_adam = b1_adam
        self._b2_adam = b2_adam
        self._eps = eps
        
        self._i_adam = None
        self._means_of_grads = None
        self._vars_of_grads = None

        
        Optimizer.__init__(self, params_to_opt)
        
    def _initialize_vars(self) :
        self._i_adam = tf.Variable(0.0, dtype="float32", name="i_adam")  # Current iteration of Adam
        self._means_of_grads = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self._vars_of_grads = []  # list of variances of grads for all parameters, for ADAM optimizer.
        for param in self._params_to_opt :
            self._means_of_grads.append( tf.Variable(param * 0., dtype="float32", name="means_of_grads") )
            self._vars_of_grads.append( tf.Variable(param * 0., dtype="float32", name="vars_of_grads") )
            
    def get_update_ops_given_grads(self, grads) :
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        updates = []
        
        i = self._i_adam
        i_t = i + 1.
        fix1 = 1. - (self._b1_adam)**i_t
        fix2 = 1. - (self._b2_adam)**i_t
        lr_t = self._learning_rate * (tf.sqrt(fix2) / fix1)
        for param, grad, m, v in zip(self._params_to_opt, grads, self._means_of_grads, self._vars_of_grads):
            m_t = (self._b1_adam * m) + ((1. - self._b1_adam) * grad)
            v_t = (self._b2_adam * v) + ((1. - self._b2_adam) * tf.square(grad))  # Double check this with the paper.
            grad_t = m_t / (tf.sqrt(v_t) + self._eps)
            param_t = param - (lr_t * grad_t)
            
            updates.append( tf.compat.v1.assign(ref=m, value=m_t, validate_shape=True) )
            updates.append( tf.compat.v1.assign(ref=v, value=v_t, validate_shape=True) )
            updates.append( tf.compat.v1.assign(ref=param, value=param_t, validate_shape=True) )
        updates.append( tf.compat.v1.assign(ref=i, value=i_t, validate_shape=True) )
        
        return updates
    
    
class RmsPropOptimizer(Optimizer):
    def __init__(self,
                 params_to_opt,
                 learning_rate,
                 momentum,
                 momentumTypeNONNormalized0orNormalized1,
                 classicMomentum0OrNesterov1,
                 rho,
                 eps):
        
        self.name = "RmsPropOptimizer"
        
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self._classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        self._rho = rho
        self._eps = eps
        
        self._accu_grad_squared = None
        self._velocities_for_mom = None
        
        Optimizer.__init__(self, params_to_opt)
        
    def _initialize_vars(self) :
        self._accu_grad_squared = []
        self._velocities_for_mom = []
        for param in self._params_to_opt :
            self._accu_grad_squared.append( tf.Variable(param * 0., dtype="float32", name="accu_grad_squared") ) # accumulates the mean of the grad's square.
            self._velocities_for_mom.append( tf.Variable(param * 0., dtype="float32", name="velocities_for_mom") )
            
    def get_update_ops_given_grads(self, grads) :
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self._momentum * self._momentumTypeNONNormalized0orNormalized1
        
        for param, grad, accu, v in zip( self._params_to_opt, grads, self._accu_grad_squared, self._velocities_for_mom ):
            accu_new = self._rho * accu + (1 - self._rho) * tf.square(grad)
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * (self._learning_rate * grad / tf.sqrt(accu_new + self._eps))
            newVelocity = self._momentum * v - stepToGradientDirection
            
            if self._classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self._momentum * newVelocity - stepToGradientDirection
                
            updates.append( tf.compat.v1.assign(ref=accu, value=accu_new, validate_shape=True) )
            updates.append( tf.compat.v1.assign(ref=v, value=newVelocity, validate_shape=True) )  # I can do (1-mom)*learnRate*grad.
            updates.append( tf.compat.v1.assign(ref=param, value=param+updateToParam, validate_shape=True) )
            
        return updates


class RmsPropOptimizerSI(Optimizer):
    def __init__(self,
                 params_to_opt,
                 learning_rate,
                 momentum,
                 momentumTypeNONNormalized0orNormalized1,
                 classicMomentum0OrNesterov1,
                 rho,
                 eps,
                 c):

        self.name = "RmsPropOptimizerSI"

        self._learning_rate = learning_rate
        self._momentum = momentum
        self._momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self._classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        self._rho = rho
        self._eps = eps
        self._c = c

        self._accu_grad_squared = None
        self._velocities_for_mom = None

        Optimizer.__init__(self, params_to_opt)

    def _initialize_vars(self):
        self._accu_grad_squared = []
        self._velocities_for_mom = []
        self._ws = []  # list of ws, for synaptic intelligence (SI)  HYX
        self._omegas = []  # list of omegas, for synaptic intelligence (SI)  HYX
        self._init_vars = []  # list of variable values from last task/trained model, for synaptic intelligence (SI), HYX
        for param in self._params_to_opt:
            self._accu_grad_squared.append(tf.Variable(param * 0., dtype="float32",
                                                       name="accu_grad_squared"))  # accumulates the mean of the grad's square.
            self._velocities_for_mom.append(tf.Variable(param * 0., dtype="float32", name="velocities_for_mom"))

            self._ws.append(tf.Variable(param * 0., dtype="float32", name="ws_SI"))
            self._omegas.append(tf.Variable(param * 0., dtype="float32", name="omega_SI"))
            self._init_vars.append(tf.Variable(tf.identity(param), dtype="float32", name="init_vars_SI"))



    def get_update_ops_given_grads(self, grads):
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self._momentum * self._momentumTypeNONNormalized0orNormalized1

        for param, grad, accu, v, w, omega, init_var in zip(self._params_to_opt, grads, self._accu_grad_squared, self._velocities_for_mom, self._ws, self._omegas, self._init_vars):
            # accu_new = self._rho * accu + (1 - self._rho) * tf.square(grad)
            # stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * (
            #             self._learning_rate * grad / tf.sqrt(accu_new + self._eps))
            # newVelocity = self._momentum * v - stepToGradientDirection
            #
            # if self._classicMomentum0OrNesterov1 == 0:
            #     updateToParam = newVelocity
            # else:  # Nesterov
            #     updateToParam = self._momentum * newVelocity - stepToGradientDirection

            weight_diff = param - init_var
            regularizer = tf.multiply(weight_diff, 2 * self._c * omega)
            # print("grads", grad.shape())
            # print("regularizer", regularizer.shape())
            grad2 = grad + regularizer

            accu_new = self._rho * accu + (1 - self._rho) * tf.square(grad2)
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * (
                    self._learning_rate * grad2 / tf.sqrt(accu_new + self._eps))
            newVelocity = self._momentum * v - stepToGradientDirection

            if self._classicMomentum0OrNesterov1 == 0:
                updateToParam = newVelocity
            else:  # Nesterov
                updateToParam = self._momentum * newVelocity - stepToGradientDirection

            param_t = param + updateToParam
            w_t = w - updateToParam * grad

            updates.append(tf.compat.v1.assign(ref=accu, value=accu_new, validate_shape=True))
            updates.append(
                tf.compat.v1.assign(ref=v, value=newVelocity, validate_shape=True))  # I can do (1-mom)*learnRate*grad.
            updates.append(tf.compat.v1.assign(ref=param, value=param_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=w, value=w_t, validate_shape=True))
            # updates.append(omega)
            # updates.append(init_var)

        return updates

    def get_update_ops_task(self):
        updates = []
        for param,  w, omega, init_var in zip(self._params_to_opt, self._ws, self._omegas, self._init_vars):
            delta = param - init_var
            omega_t = omega + w/(delta ** 2 + self._eps)
            # print("w", tf.math.reduce_sum(w).eval())
            # print("omega", tf.math.reduce_sum(omega_t).eval())
            init_var_t = tf.identity(param)
            w_t = w * 0

            updates.append(tf.compat.v1.assign(ref=w, value=w_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=omega, value=omega_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=init_var, value=init_var_t, validate_shape=True))

        return updates

    
class AdamOptimizerSI(Optimizer):
    def __init__(self,
                 params_to_opt,
                 learning_rate,
                 b1_adam,
                 b2_adam,
                 eps,
                 c
                 ):

        self.name = "AdamOptimizerSI"

        self._learning_rate = learning_rate
        self._b1_adam = b1_adam
        self._b2_adam = b2_adam
        self.c = c  # relaxation parameter for SI regularization
        self._eps = eps

        self._i_adam = None
        self._means_of_grads = None
        self._vars_of_grads = None
        self._ws = None
        self._omegas = None
        self._init_vars = None

        Optimizer.__init__(self, params_to_opt)

    def _initialize_vars(self):
        self._i_adam = tf.Variable(0.0, dtype="float32", name="i_adam")  # Current iteration of Adam
        self._means_of_grads = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self._vars_of_grads = []  # list of variances of grads for all parameters, for ADAM optimizer.
        self._ws = []  # list of ws, for synaptic intelligence (SI)  HYX
        self._omegas = []  # list of omegas, for synaptic intelligence (SI)  HYX
        self._init_vars = [] # list of variable values from last task/trained model, for synaptic intelligence (SI), HYX
        for param in self._params_to_opt:
            self._means_of_grads.append(tf.Variable(param * 0., dtype="float32", name="means_of_grads"))
            self._vars_of_grads.append(tf.Variable(param * 0., dtype="float32", name="vars_of_grads"))
            self._ws.append(tf.Variable(param * 0., dtype="float32", name="ws_SI"))
            self._omegas.append(tf.Variable(param * 0., dtype="float32", name="omega_SI"))
            self._init_vars.append(tf.Variable(tf.identity(param), dtype="float32", name="init_vars_SI"))


    def get_update_ops_given_grads(self, grads):
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        # Step updates
        updates = []

        i = self._i_adam
        i_t = i + 1.
        fix1 = 1. - (self._b1_adam) ** i_t
        fix2 = 1. - (self._b2_adam) ** i_t
        lr_t = self._learning_rate * (tf.sqrt(fix2) / fix1)
        for param, grad, m, v, w, omega, init_var in zip(self._params_to_opt, grads, self._means_of_grads,
                                                         self._vars_of_grads, self._ws, self._omegas, self._init_vars):
            m_t = (self._b1_adam * m) + ((1. - self._b1_adam) * grad)
            v_t = (self._b2_adam * v) + ((1. - self._b2_adam) * tf.square(grad))  # Double check this with the paper.
            grad_t = m_t / (tf.sqrt(v_t) + self._eps)
            weight_dif = param - init_var
            grad_reg_t = grad_t + tf.multiply(weight_dif, 2 * self.c * omega)
            param_t = param - (lr_t * grad_reg_t)

            diff = param_t - param

            # if tf.math.equal(diff, 0):
            #     print('Path diff wrong warning!')
            w_t = w - diff * grad_t

            updates.append(tf.compat.v1.assign(ref=m, value=m_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=v, value=v_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=param, value=param_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=w, value=w_t, validate_shape=True))
            updates.append(omega)
            updates.append(init_var)
        updates.append(tf.compat.v1.assign(ref=i, value=i_t, validate_shape=True))

        return updates

    def get_update_ops_task(self):
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        updates = []
        i = self._i_adam
        for param, m, v, w, omega, init_var in zip(self._params_to_opt, self._means_of_grads,
                                                         self._vars_of_grads, self._ws, self._omegas, self._init_vars):
            delta = param - init_var
            omega_t = omega + w/(delta ** 2 + self._eps)
            init_var_t = tf.identity(param)
            w_t = w * 0

            updates.append(m)
            updates.append(v)
            updates.append(param)
            updates.append(tf.compat.v1.assign(ref=w, value=w_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=omega, value=omega_t, validate_shape=True))
            updates.append(tf.compat.v1.assign(ref=init_var, value=init_var_t, validate_shape=True))
        updates.append(i)

        return updates
"""
From https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617 :
ClassicMomentum:
(1) v_t = mu * v_t-1 - lr * gradient_f(params_t)
(2) params_t = params_t-1 + v_t
(3) params_t = params_t-1 + mu * v_t-1 - lr * gradient_f(params_t-1)

Nesterov momentum:
(4) v_t = mu * v_t-1 - lr * gradient_f(params_t-1 + mu * v_t-1)
(5) params_t = params_t-1 + v_t

alternative formulation for Nesterov momentum:
(6) v_t = mu * v_t-1 - lr * gradient_f(params_t-1)
(7) params_t = params_t-1 + mu * v_t - lr * gradient_f(params_t-1)
(8) params_t = params_t-1 + mu**2 * v_t-1 - (1+mu) * lr * gradient_f(params_t-1)

Can also find help for optimizers in Lasagne: https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
"""
    

