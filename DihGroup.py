from keras import  backend as K
from keras.engine.topology import Layer
import random

class Dih4URandom(Layer):
    """
        Dih4 Group:
        -- e - identity operation
        -- r - Pi/4 rotate operation
        -- r**2 - Pi/2 rotate operation
        -- r**3 - 3*Pi/4 rotate operation
        -- s - reflect
        -- r*s
        -- r**2*s
        -- r**3*s

    """
    def __init__(self,**kwargs):
        super(Dih4URandom, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, training=None):
        def dih_proc():
            # uniform gen random int(0-7)
            n = random.randint(0, 7)
            # uniform choice Dih4 group operation
            # which implement for x
            if n==0:
                return self._e(x)
            elif n==1:
                return self._r90(x)
            elif n==2:
                return self._r180(x)
            elif n==3:
                return self._r270(x)
            elif n==4:
                return self._s(x)
            elif n==5:
                return self._r90(self._s(x))
            elif n==6:
                return self._r180(self._s(x))
            elif n==7:
                return self._r270(self._s(x))

        return K.in_train_phase(dih_proc, x,
                training=training)

    def compute_output_shape(self, input_shape):
        return input_shape
    # implementation Dih4 group operation
    # e
    def _e(self, x):
        return x
    # r
    def _r90(self, x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[:,::-1,:]
        return x
    # r**2
    def _r180(self, x):
        return x[::-1,::-1,:]
    # r**3
    def _r270(self, x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[::-1,:,:]
        return x
    # s
    def _s(self,x):
        return x[::-1,:,:]

class Dih4(Layer):
    """
        Dih4 Group:
        -- e - identity operation
        -- r - Pi/4 rotate operation
        -- r**2 - Pi/2 rotate operation
        -- r**3 - 3*Pi/4 rotate operation
        -- s - reflect
        -- r*s
        -- r**2*s
        -- r**3*s

    """
    def __init__(self,**kwargs):
        super(Dih4, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, training=None):
        x1 = self._e(x)
        x2 = self._r90(x)
        x3 = self._r180(x)
        x4 = self._r270(x)
        x5 = self._s(x)
        x6 = self._r90(self._s(x))
        x7 = self._r180(self._s(x))
        x8 = self._r270(self._s(x))
        return [x1,x2,x3,x4,x5,x6,x7,x8]

    def compute_output_shape(self, input_shape):
        return [8, input_shape[0], input_shape[1], input_shape[2]]
    # implementation Dih4 group operation
    # e
    def _e(self, x):
        return x
    # r
    def _r90(self, x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[:,::-1,:]
        return x
    # r**2
    def _r180(self, x):
        return x[::-1,::-1,:]
    # r**3
    def _r270(self, x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[::-1,:,:]
        return x
    # s
    def _s(self,x):
        return x[::-1,:,:]
