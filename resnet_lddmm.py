import tensorflow as tf

class DenseEulerFBlock(tf.keras.Model):
    def __init__(self):
        super(DenseEulerFBlock, self).__init__()
        self.initialiser = tf.keras.initializers.HeNormal()
        
        self.d1 = tf.keras.layers.Dense(500, activation='relu')
        self.d2 = tf.keras.layers.Dense(500, activation=None)
        self.d3 = tf.keras.layers.Dense(2, activation=None, use_bias=False)
        
    def call(self, input_tensor, training=False):
        return self.d3(self.d2(self.d1(input_tensor)))

class DenseEulerMergeBlock(tf.keras.Model):
    def __init__(self):
        super(DenseEulerMergeBlock, self).__init__()
        
    def call(self, input_tensor, training=False):
        return tf.nn.relu(input_tensor)

def DenseCombinedLoss(d1, d2, d3, d4, d5, d6, m6, truth, sigma=0.1):
    regularisation_loss = 0.5*(tf.norm(d1) + tf.norm(d2) + tf.norm(d3) + tf.norm(d4) + tf.norm(d5) + tf.norm(d6))/6
    data_term           = 0.5*tf.norm(m6-truth)/sigma
    return regularisation_loss + data_term

before_batch = (before_batch - np.min(before_batch))/np.max(before_batch) # some normalisation
after_batch = (after_batch - np.min(after_batch))/np.max(after_batch)

tf.keras.backend.clear_session() # reseting counters
initializer = tf.keras.initializers.HeNormal() # He weight initialisations

input0 = tf.keras.Input(shape=(1, 2)) # expected shape for 2D input; multiple points controlled by batch size.

d1 = DenseEulerFBlock()(input0)
m1 = DenseEulerMergeBlock()(input0 + d1)

d2 = DenseEulerFBlock()(m1)
m2 = DenseEulerMergeBlock()(m1 + d2)

d3 = DenseEulerFBlock()(m2)
m3 = DenseEulerMergeBlock()(m2 + d3)

d4 = DenseEulerFBlock()(m3)
m4 = DenseEulerMergeBlock()(m3 + d4)

d5 = DenseEulerFBlock()(m4)
m5 = DenseEulerMergeBlock()(m4 + d5)

d6 = DenseEulerFBlock()(m5)
m6 = DenseEulerMergeBlock()(m5 + d6)


true0 = tf.keras.Input(shape=(1, 2))
model = tf.keras.Model([input0, true0], [input0, m1, m2, m3, m4, m5, m6, true0])
model.add_loss(DenseCombinedLoss(d1, d2, d3, d4, d5, d6, m6, true0, sigma=0.1))

opt = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=opt, loss=None)