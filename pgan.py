from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU, Dense, Flatten, Reshape
from tqdm import tqdm
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
"""
TODO:
    - Dataloader, class that loads data to the correct input format.
        Must handle different batch sizes.
        Try with CIFAR10?
    - Data augmenter class for standardizing and preprocessing the data.
    - Generator, discriminator models, custom layers.
    - Trainer class that handles both networks training and growing.
    - Loss handler class perhaps? Might not be needed.
    - Evaluator class.

    Read paper for details regarding loss, weight normalization, input normalization etc.
"""
class DataLoader:
    def __init__(self):
        pass

    def get_data(self):
        # Skip labels and test data
        (x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
        ds = tf.data.Dataset.from_tensor_slices(x_train)
        ds.shuffle(buffer_size=1000)
        return ds

    def get_dataloaders(self):
        return {"train": self.get_data()}

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = 0.0

    def call(self, inputs):
        assert len(inputs) == 2, "Expects list of exactly two inputs"
        return self.alpha * inputs[0] + (1.0 - self.alpha) * inputs[1]

class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
        self.eps = 1e-8

    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = tf.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = tf.reduce_mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = tf.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = tf.reduce_mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = tf.concat([inputs, output], axis=-1)
        return combined

class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
        self.eps = 1e-8

    def call(self, x):
        squared = tf.square(x)
        mean_squared = tf.reduce_mean(squared, axis=-1, keepdims=True)
        mean_squared += self.eps
        l2 = tf.sqrt(mean_squared)
        norm = x / l2
        return norm # Again, probably have to check shapes

class PGAN:
    def __init__(self):
        self.latent_shape = (1, 1, 128) # Channels last
        self.starting_filters = 128
        self.leaky_relu = 0.2

    def build_initial_d(self):
        initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
        constraint = tf.keras.constraints.MaxNorm(1.0)
        image_input = Input(shape=(4, 4, 3))
        # conv 1x1
        x = Conv2D(filters=self.starting_filters, kernel_size=1, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(image_input)
        x = LeakyReLU(self.leaky_relu)(x)
        # conv 3x3 (output block)
        x = MinibatchStdev()(x)
        x = Conv2D(filters=self.starting_filters, kernel_size=3, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        x = LeakyReLU(self.leaky_relu)(x)
        # conv 4x4
        x = Conv2D(filters=self.starting_filters, kernel_size=3, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        x = LeakyReLU(self.leaky_relu)(x)
        # dense output layer
        x = Flatten()(x)
        x = Dense(1)(x)
        d = Model(
            inputs=image_input,
            outputs=x,
            name="discriminator",
        )
        d.summary()
        return d

    def build_initial_g(self):
        initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
        constraint = tf.keras.constraints.MaxNorm(1.0)
        latent_input = Input(shape=self.latent_shape, name="latent_input")
        x = Dense(self.starting_filters * 4 * 4, kernel_constraint=constraint, kernel_initializer=initializer)(latent_input)
        x = Reshape((4, 4, self.starting_filters))(x)
        x = Conv2D(filters=self.starting_filters, kernel_size=4, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(self.leaky_relu)(x)
        x = Conv2D(filters=self.starting_filters, kernel_size=3, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(self.leaky_relu)(x)
        image_output = Conv2D(filters=3, kernel_size=1, padding="same", name="g_image_output", kernel_constraint=constraint, kernel_initializer=initializer)(x) # toRGB
        g = Model(
            inputs=latent_input,
            outputs=image_output,
            name="generator",
        )
        g.summary()
        return g

    def add_d_block(self, old_model):
        # weight initialization
        initializer = tf.initializers.RandomNormal(stddev=0.02)
        # weight constraint
        constraint = tf.keras.constraints.max_norm(1.0)
        # get shape of existing model
        in_shape = list(old_model.input.shape)
        # define new input shape as double the size
        input_shape = (in_shape[-2]*2, in_shape[-2]*2, in_shape[-1])
        image_input = Input(shape=input_shape)
        # define new input processing layer
        x = Conv2D(128, (1,1), padding='same', kernel_initializer=initializer, kernel_constraint=constraint)(image_input)
        x = LeakyReLU(alpha=0.2)(x)
        # define new block
        x = Conv2D(128, (3,3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3,3), padding='same', kernel_initializer=initializer, kernel_constraint=constraint)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D()(x)
        block_new = x
        # skip the input, 1x1 and activation for the old model
        for i in range(3, len(old_model.layers)):
            x = old_model.layers[i](x)
        # define straight-through model
        fixed_d = Model(image_input, x, name="fixed_d")
        fixed_d.summary()
        # compile model
        # downsample the new larger image
        downsample = AveragePooling2D()(image_input)
        # connect old input processing to downsampled new input
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        # fade in output of old model input layer with new input
        x = WeightedSum()([block_old, block_new])
        # skip the input, 1x1 and activation for the old model
        for i in range(3, len(old_model.layers)):
            x = old_model.layers[i](x)
        # define straight-through model
        ws_d = Model(image_input, x, name="ws_d")
        ws_d.summary()
        # compile model
        return (fixed_d, ws_d)


    def add_g_block(self, old_model):
        initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.02)
        constraint = tf.keras.constraints.MaxNorm(1.0)
        block_end = old_model.layers[-2].output
        upsampling = UpSampling2D()(block_end)
        x = Conv2D(self.starting_filters, 3, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(upsampling)
        x = PixelNormalization()(x)
        x = LeakyReLU(self.leaky_relu)(x)
        x = Conv2D(self.starting_filters, 3, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        x = PixelNormalization()(x)
        x = LeakyReLU(self.leaky_relu)(x)
        image_output = Conv2D(3, 1, padding="same", kernel_constraint=constraint, kernel_initializer=initializer)(x)
        fixed_g = Model(inputs=old_model.input, outputs=image_output, name="fixed_g")
        fixed_g.summary()
        out_old = old_model.layers[-1]
        image_output2 = out_old(upsampling)
        merged = WeightedSum()([image_output2, image_output])
        ws_g = Model(inputs=old_model.input, outputs=merged, name="ws_g")
        ws_g.summary()
        return (fixed_g, ws_g)

    def update_alpha(self, g, d, alpha):
        for model in [g, d]:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    layer.alpha = alpha

    def wasserstein_loss(self, y_true, y_pred):
        eps = 1e-3
        expected_val = tf.reduce_mean(tf.square(y_pred))
        return tf.reduce_mean(y_true * y_pred) + eps * expected_val

    def generate_latent(self, batch_size):
        noise = tf.random.normal(shape=(batch_size, *self.latent_shape))
        return noise

class Augmentor:
    def __init__(self):
        pass

    def __call__(self, x):
        return tf.image.per_image_standardization(x)
    
    def resize(self, x, size):
        x = tf.image.resize(x, [size, size])
        return x

class Trainer:
    def __init__(self):
        self.dataloaders = self.get_dataloaders()
        self.augmentor = Augmentor()
        self.pgan = PGAN()
        self.fixed_g = self.pgan.build_initial_g()
        self.fixed_d = self.pgan.build_initial_d()
        self.ws_g = None
        self.ws_d = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

    def get_dataloaders(self):
        loader = DataLoader()
        return loader.get_dataloaders()

    def train_step(self, real_imgs, g, d):
        batch_size = real_imgs.shape[0]
        real_labels = tf.ones(shape=(batch_size, ))
        # Train d on real images
        with tf.GradientTape() as grad:
            real_pred = d(real_imgs)
            d_loss = self.pgan.wasserstein_loss(real_labels, real_pred)
        gradients = grad.gradient(d_loss, d.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, d.trainable_variables))
        d_batch_loss = tf.reduce_mean(d_loss)
        # Train d on generated images
        latent_noise = self.pgan.generate_latent(real_imgs.shape[0])
        with tf.GradientTape() as grad:
            gen_imgs = g(latent_noise)
            gen_labels = -tf.ones(shape=(batch_size, ))
            gen_pred = d(gen_imgs)
            d_loss = self.pgan.wasserstein_loss(gen_labels, gen_pred)
        gradients = grad.gradient(d_loss, d.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, d.trainable_variables))
        d_batch_loss = tf.reduce_mean(d_loss)
        d_batch_loss /= 2
        # Train g
        latent_noise = self.pgan.generate_latent(2 * batch_size)
        with tf.GradientTape() as grad:
            gen_imgs = g(latent_noise)
            gen_labels = tf.ones(shape=(2 * batch_size, ))
            gen_pred = d(gen_imgs)
            d_loss = self.pgan.wasserstein_loss(gen_labels, gen_pred)
            g_loss = d_loss
        gradients = grad.gradient(g_loss, g.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, g.trainable_variables))
        return tf.reduce_mean(d_loss), d_batch_loss


    def training_loop(self, n_epochs):
        loader = self.dataloaders["train"]
        batch_size = 8
        loader = loader.batch(batch_size)
        n_steps = len(loader) // batch_size
        fade_in = False
        for epoch in range(n_epochs):
            loader.shuffle(buffer_size=1000)
            if fade_in:
                (self.fixed_g, self.ws_g) = self.pgan.add_g_block(self.fixed_g)
                (self.fixed_d, self.ws_d) = self.pgan.add_d_block(self.fixed_d)
                g = self.ws_g
                d = self.ws_d
                d_alpha = 1 / (n_steps - 1)
                alpha = 0.0
            else:
                g = self.fixed_g
                d = self.fixed_d

            pbar = tqdm(loader)
            for inputs in pbar:
                inputs = self.augmentor(inputs)
                inputs = self.augmentor.resize(inputs, d.input.shape[-2])
                if fade_in:
                    alpha += d_alpha
                    self.pgan.update_alpha(self.ws_g, self.ws_d, alpha)
                g_loss, d_loss = self.train_step(inputs, g, d)
                pbar.set_description(f"Gen loss: {g_loss.numpy():.3f}, Disc loss: {d_loss.numpy():.3f}")
            self.plot_gen_sample(g)
            fade_in = not fade_in

    def plot_gen_sample(self, g):
        noise = self.pgan.generate_latent(1)
        gen_img = g(noise)
        plt.imshow(gen_img[0])
        plt.show()

                

trainer = Trainer()
trainer.training_loop(7)
# 4x4
# 8x8
# 8x8
# 16x16
# 16x16
# 32x32
# 32x32