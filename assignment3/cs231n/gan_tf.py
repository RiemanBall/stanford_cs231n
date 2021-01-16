import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Inputs:
    - x: TensorFlow Tensor with arbitrary shape
    - alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    # TODO: implement leaky ReLU
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.where(x > 0, x, x * alpha)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def sample_noise(batch_size, dim, seed=None):
    """Generate random uniform noise from -1 to 1.
    
    Inputs:
    - batch_size: integer giving the batch size of noise to generate
    - dim: integer giving the dimension of the noise to generate
    
    Returns:
    TensorFlow Tensor containing uniform noise in [-1, 1] with shape [batch_size, dim]
    """
    if seed is not None:
        tf.random.set_seed(seed)
    # TODO: sample and return noise
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return tf.random.uniform((batch_size, dim), minval=-1, maxval=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def discriminator(seed=None):
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    if seed is not None:
        tf.random.set_seed(seed)

    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(256, input_shape=(784,), activation=leaky_relu, name='discriminator/denseLayer1'),
            # tf.keras.layers.LeakyReLU(alpha = 0.01, name='discriminator/leakyReLU1'),
            tf.keras.layers.Dense(256, activation=leaky_relu, name='discriminator/denseLayer2'),
            # tf.keras.layers.LeakyReLU(alpha = 0.01, name='discriminator/leakyReLU2'),
            tf.keras.layers.Dense(1, name='discriminator/denseLayer3')
        ]
    )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def generator(noise_dim=NOISE_DIM, seed=None):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """

    if seed is not None:
        tf.random.set_seed(seed)
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(1024, input_shape=(noise_dim,), activation='relu', name='generator/denseLayer1'),
            tf.keras.layers.Dense(1024, activation='relu', name='generator/denseLayer2'),
            tf.keras.layers.Dense(784, activation='tanh', name='generator/denseLayer3')
        ]
    )

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N_real = tf.shape(logits_real)[0]
    N_fake = tf.shape(logits_fake)[0]

    bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)

    loss = bce(tf.ones((N_real, 1)), logits_real) + bce(tf.zeros((N_fake, 1)), logits_fake)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: Tensorflow Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: Tensorflow Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = tf.shape(logits_fake)[0]

    bce = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    loss = bce(tf.ones((N, 1)), logits_fake)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss


def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    
    Inputs:
    - learning_rate: learning rate to use for both solvers
    - beta1: beta1 parameter for both solvers (first moment decay)
    
    Returns:
    - D_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    - G_solver: instance of tf.optimizers.Adam with correct learning_rate and beta1
    """
    # TODO: create an AdamOptimizer for D_solver and G_solver
    D_solver = None
    G_solver = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    D_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)
    G_solver = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return D_solver, G_solver

def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: Tensor of shape (N, 1) giving scores for the real data.
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = 0.5 * (tf.reduce_mean(tf.square(scores_real - 1.0)) + tf.reduce_mean(tf.square(scores_fake)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: Tensor of shape (N, 1) giving scores for the fake data.
    
    Outputs:
    - loss: A Tensor containing the loss.
    """
    loss = None
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.5 * (tf.reduce_mean(tf.square(scores_fake - 1.0)))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss

def dc_discriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    model = None
    ##############################################################################
    # TODO: Implement architecture                                               #
    #                                                                            #
    # HINT: tf.keras.models.Sequential might be helpful.                         #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)))
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), activation = leaky_relu, padding = 'valid'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2))
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (5, 5), activation = leaky_relu, padding = 'valid'))
    model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units = 4 * 4 * 64, activation = leaky_relu))
    model.add(tf.keras.layers.Dense(units = 1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return model


def dc_generator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    model = tf.keras.models.Sequential()
    # TODO: implement architecture
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.add(tf.keras.layers.Dense(units = 1024, input_shape=(noise_dim,), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units = 7 * 7 * 128, activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    model.add(tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = 2, padding = 'same', activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(filters = 1 , kernel_size = (4, 4), strides = 2, padding = 'same', activation = 'tanh'))
    model.add(tf.keras.layers.Flatten())

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)****
    return model


# Design your own GAN

## ResNet helping class
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, n_channels, strides = 1, activation = None):
        super(ResBlock, self).__init__()
        self.strides = strides
        self.n_channels = n_channels

        kernel_initializer = tf.keras.initializers.VarianceScaling(2.0)
        kernel_size = (3,3)
        self.conv1 = tf.keras.layers.Conv2D(n_channels, kernel_size = kernel_size, strides=strides, padding="SAME", kernel_initializer=kernel_initializer)

        self.bn1 = tf.keras.layers.BatchNormalization()

        if activation is None:
            self.activation1 = tfa.layers.Maxout(n_channels // 2)
        else:
            self.activation1 = activation

        self.conv2 = tf.keras.layers.Conv2D(n_channels * 2, kernel_size = kernel_size, padding = "SAME", kernel_initializer = kernel_initializer)

        self.bn2 = tf.keras.layers.BatchNormalization()

        if activation is None:
            self.activation2 = tfa.layers.Maxout(n_channels)
        else:
            self.activation1 = activation

        if strides > 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(n_channels, kernel_size = (1,1), strides=strides))
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs , training = False):
        x_shortcut = self.downsample(inputs)

        x_res = self.conv1(inputs)
        x_res = self.bn1(x_res, training = training)
        x_res = self.activation1(x_res)
        x_res = self.conv2(x_res)
        x_res = self.bn2(x_res, training = training)
        x_res = self.activation2(x_res)

        return tf.keras.layers.concatenate([x_res, x_shortcut])


def myGenerator(noise_dim=NOISE_DIM):
    """Generate images from a random noise vector.
    
    Inputs:
    - z: TensorFlow Tensor of random noise with shape [batch_size, noise_dim]
    
    Returns:
    TensorFlow Tensor of generated images, with shape [batch_size, 784].
    """
    inputs = tf.keras.Input(shape = (noise_dim, ))

    x = tf.keras.layers.Dense(units = 7 * 7 * 128)(inputs)
    x = tf.keras.layers.BatchNormalization()(x) #, training = True
    x = leaky_relu(x)
    
    x_shortcut = tf.keras.layers.Reshape((7, 7, 128))(x)
    
    # ResBlock1
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = 1, padding = 'same')(x_shortcut)
    x = tf.keras.layers.BatchNormalization()(x) #, training = True
    x = leaky_relu(x)
    x = tf.keras.layers.concatenate([x, x_shortcut])

    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = 2, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x) #, training = True
    x_shortcut = leaky_relu(x)

    # ResBlock2
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = 1, padding = 'same')(x_shortcut)
    x = tf.keras.layers.BatchNormalization()(x) #, training = True
    x = leaky_relu(x)
    x = tf.keras.layers.concatenate([x, x_shortcut])
    
    x = tf.keras.layers.Conv2DTranspose(filters = 1 , kernel_size = (4, 4), strides = 2, padding = 'same', activation = 'tanh')(x)
    outputs = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model


def myDiscriminator():
    """Compute discriminator score for a batch of input images.
    
    Inputs:
    - x: TensorFlow Tensor of flattened input images, shape [batch_size, 784]
    
    Returns:
    TensorFlow Tensor with shape [batch_size, 1], containing the score 
    for an image being real for each input image.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            ResBlock(32, 2),
            ResBlock(32, 1),
            ResBlock(64, 2),
            ResBlock(64, 1),
            ResBlock(128, 2),
            ResBlock(128, 1),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units = 1)
        ]
    )

    return model


def computeGradPenalty(D, real, fake):
    """
    Computes the gradient penalty in WGAN-GP.
    
    Inputs:
    - D: discriminator model
    - real: Tensor of shape (N, 784) flatten real images.
    - fake: Tensor of shape (N, 784) flatten fake images.
    
    Returns:
    - gp: Tensor containing (scalar) the gradient penalty.
    """
    # uniform sample of interpolation
    N = tf.minimum(tf.shape(real)[0], tf.shape(fake)[0])
    alpha = tf.random.uniform(shape = (N, 1), minval = 0.0, maxval = 1.0)

    interpolation = alpha * real[:N, :] + (1 - alpha) * fake[:N, :]
    
    with tf.GradientTape() as tape:
        tape.watch(interpolation)
        disc_score = D(interpolation, training = True)

    grad = tape.gradient(disc_score, interpolation)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1,]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)

    return gp


def myDiscriminatorLoss(D, real_imgs, fake_imgs, logits_real, logits_fake, lambda_gp = 10.0):
    """
    Computes the discriminator loss in WGAN-GP.
    
    Inputs:
    - D: discriminator model
    - real_imgs: Tensor of shape (N, 784) flatten real images.
    - fake_imgs: Tensor of shape (N, 784) flatten fake images.
    - logits_real: Tensor of shape (N, 1) giving scores for the real data.
    - logits_fake: Tensor of shape (N, 1) giving scores for the fake data.
    - lambda_gp: float. Weight for gradient penalty
    
    Returns:
    - loss: Tensor containing (scalar) the loss for the discriminator.
    """
    grad_penalty = computeGradPenalty(D, real_imgs, fake_imgs)
    loss = -tf.reduce_mean(logits_real) + tf.reduce_mean(logits_fake) + lambda_gp * grad_penalty

    return loss


def myGeneratorLoss(logits_fake):
    """
    Computes the generator loss in WGAN-GP.

    Inputs:
    - logits_fake: Tensorflow Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: Tensorflow Tensor containing the (scalar) loss for the generator.
    """
    loss = -tf.reduce_mean(logits_fake)
    
    return loss


# a giant helper function
def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss,\
              train_gen_every = 1, flip_disc_labels_rate = None, WGAN_GP = False,\
              show_every=250, print_every=20, batch_size=128, num_epochs=10, noise_size=96):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - D: Discriminator model
    - G: Generator model
    - D_solver: an Optimizer for Discriminator
    - G_solver: an Optimizer for Generator
    - generator_loss: Generator loss
    - discriminator_loss: Discriminator loss
    Returns:
        Nothing
    """
    assert show_every % train_gen_every == 0

    mnist = MNIST(batch_size=batch_size, shuffle=True)
    
    iter_count = 0
    images = []
    flip = False

    for epoch in range(1, num_epochs + 1):
        for (x, _) in mnist:

            if flip_disc_labels_rate is not None:
                flip = tf.random.uniform(shape = []) < (flip_disc_labels_rate / epoch)

            with tf.GradientTape() as tape:
                # Real data and logit
                real_data = preprocess_img(x)
                logits_real = D(real_data, training = True)

                # Fake data and logit
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = tf.reshape(G(g_fake_seed, training = True), [batch_size, 784])
                logits_fake = D(fake_images, training = True)
                
                if flip:
                    real_data, fake_images = fake_images, real_data
                    logits_real, logits_fake = logits_fake, logits_real

                if WGAN_GP:
                    d_total_error = discriminator_loss(D, real_data, fake_images, logits_real, logits_fake)
                else:
                    d_total_error = discriminator_loss(logits_real, logits_fake)

                # Update
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)      
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))
            
            if (iter_count % train_gen_every == 0): 
                with tf.GradientTape() as tape:
                    # Fake images
                    g_fake_seed = sample_noise(batch_size, noise_size)
                    fake_images = G(g_fake_seed, training = True)

                    # Logit
                    gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]), training = True)

                    # Compute generator loss
                    g_error = generator_loss(gen_logits_fake)
                    
                    # Update
                    g_gradients = tape.gradient(g_error, G.trainable_variables)      
                    G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % show_every == 0):
                print(f"Epoch: {epoch}, Iter: {iter_count}, D: {d_total_error}, G:{g_error}")
                imgs_numpy = fake_images.cpu().numpy()
                images.append(imgs_numpy[0:16])
                show_images(images[-1])
                plt.show()
                
            iter_count += 1
    
    # random noise fed into our generator
    z = sample_noise(batch_size, noise_size)
    # generated images
    G_sample = G(z)
    
    return images, G_sample[:16]

class MNIST(object):
    def __init__(self, batch_size, shuffle=False):
        """
        Construct an iterator object over the MNIST data
        
        Inputs:
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        train = None
        with tf.device('cpu:0'):
            train, _ = tf.keras.datasets.mnist.load_data()

        X, y = train
        X = X.astype(np.float32)/255
        X = X.reshape((X.shape[0], -1))
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B)) 

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.shape) for p in model.weights])
    return param_count

def show_images(images):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return
