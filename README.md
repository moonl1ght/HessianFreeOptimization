# Hessian Free Optimization
Hessian Free Optimizer based on Tensorflow

Implemented Hessian-free optimization features from Martens (2010) and 
Martens and Sutskever (2011):
* Gauss-Newton approximation
* Early termination
* Tikhonov damping
* Preconditioner for conjugate gradient method
* Levenberg-Marquardt heuristic for adapting damping coefficient

Main computation features
-------------

Computing Hessian and vector product:

```python
# self.W: weights
# grads: Gradients of the network
# self.damp_pl: Tikhonov damping coefficient
# vec: Vector that is multiplied by the Jacobian.
grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
Hv = tf.gradients(grad_v, self.W, stop_gradients=vec)
Hv = [hv + self.damp_pl * v for hv, v in zip(Hv, vec)]
```

Computing Jacobian and vector product or R-operator: 

```python
# f: Objective function
# x: Parameters with respect to which computes Jacobian matrix.
# vec: Vector that is multiplied by the Jacobian.
r = [tf.reduce_sum([tf.reduce_sum(v * tf.gradients(f, x)[i])
  for i, v in enumerate(vec)])
  for f in tf.unstack(f)]
```

Computing Gauss-Newton matrix (J'HJ) and vector product:

```python
# self.output: activation of output layer
# self.W: weights
# self.damp_pl: Tikhonov damping coefficient
Jv = self.__Rop(self.output, self.W, vec)
Jv = tf.reshape(tf.stack(Jv), [-1, 1])
HJv = tf.gradients(tf.matmul(tf.transpose(tf.gradients(self.loss,
  self.output)[0]), Jv), self.output, stop_gradients=Jv)[0]
JHJv = tf.gradients(tf.matmul(tf.transpose(HJv), self.output), self.W,
  stop_gradients=HJv)
JHJv = [gv + self.damp_pl * v for gv, v in zip(JHJv, vec)]
```

Tested on:
------------
* python 3.6.4
* Tensorflow v1.5.0
