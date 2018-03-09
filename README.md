# Hessian Free Optimization
Hessian Free Optimizer based on Tensorflow and Python

Features
-------------

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
