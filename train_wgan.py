import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape, concatenate
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
import keras.backend as K

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Here is where we will load the dataset stored in dataset_path. In this script
# we will use the Caltech-UCSD Birds-200-2011 dataset which includes 11788
# images from 200 different birds. We will feed the images without applying
# the provided bounding boxes from the dataset. The data will only be resized
# and normalized. Keras ImageDataGenerator will be used for loading the dataset
def load_dataset(dataset_path, batch_size, image_shape):
    dataset_generator = ImageDataGenerator()
    dataset_generator = dataset_generator.flow_from_directory(
        dataset_path, target_size=(image_shape[0], image_shape[1]),
        batch_size=batch_size,
        class_mode=None)

    return dataset_generator


# Let's define our Wasserstein Loss function. We apply the mean in order to be
# able to compare outputs with different batch sizes
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# Creates the critic model. This model tries to classify images as real
# or fake. In WGAN contrarilly to DCGAN, the output does not need to be a probability
# that's why it is called a crtitc, because it does not explicitly classify as fake or
# real.
# Important note: in the original pytorch implementation of the artice, the biases
# are set to false, here I left them as default.
def construct_critic(image_shape):

    # weights need to be initialized with close values near zero to avoid
    # clipping
    weights_initializer = RandomNormal(mean=0., stddev=0.01)

    critic = Sequential()
    critic.add(Conv2D(filters=64, kernel_size=(5, 5),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer,
                      input_shape=(image_shape)))
    critic.add(LeakyReLU(0.2))

    critic.add(Conv2D(filters=128, kernel_size=(5, 5),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer))
    critic.add(BatchNormalization(momentum=0.5))
    critic.add(LeakyReLU(0.2))

    critic.add(Conv2D(filters=256, kernel_size=(5, 5),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer))
    critic.add(BatchNormalization(momentum=0.5))
    critic.add(LeakyReLU(0.2))

    critic.add(Conv2D(filters=512, kernel_size=(5, 5),
                      strides=(2, 2), padding='same',
                      data_format='channels_last',
                      kernel_initializer=weights_initializer))
    critic.add(BatchNormalization(momentum=0.5))
    critic.add(LeakyReLU(0.2))

    critic.add(Flatten())

    # We output two layers, one witch predicts the class and other that
    # tries to figure if image is fake or not
    critic.add(Dense(units=1, activation=None))
    optimizer = RMSprop(lr=0.00005)
    critic.compile(loss=wasserstein_loss,
                   optimizer=optimizer,
                   metrics=None)

    return critic


# Creates the generator model. This model has an input of random noise and
# generates an image that will try mislead the critic.
# Important note: in the original pytorch implementation of the artice, the biases
# are set to false, here I left them as default.
def construct_generator():

    weights_initializer = RandomNormal(mean=0., stddev=0.01)

    generator = Sequential()

    generator.add(Dense(units=4 * 4 * 512,
                        kernel_initializer=weights_initializer,
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(4, 4, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=3, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer=weights_initializer))
    generator.add(Activation('tanh'))

    optimizer = RMSprop(lr=0.00005)
    generator.compile(loss=wasserstein_loss,
                      optimizer=optimizer,
                      metrics=None)

    return generator


# Function that allows writing the loss to tensorboard to visualize the
# plots of the losses. In tensorboard you will have 5 different loss plots
# shown correspondent to critic fake loss, critic real loss, generator loss,
# Critic Real los - Critic Fake Loss and Critic Real Loss + Critic Fake loss
def write_to_tensorboard(generator_step, summary_writer,
                         losses):

    summary = tf.Summary()

    value = summary.value.add()
    value.simple_value = losses[1]
    value.tag = 'Critic Real Loss'

    value = summary.value.add()
    value.simple_value = losses[2]
    value.tag = 'Critic Fake Loss'

    value = summary.value.add()
    value.simple_value = losses[3]
    value.tag = 'Generator Loss'

    value = summary.value.add()
    value.simple_value = losses[1] - losses[2]
    value.tag = 'Critic Loss (D_real - D_fake)'

    value = summary.value.add()
    value.simple_value = losses[1] + losses[2]
    value.tag = 'Critic Loss (D_fake + D_real)'

    summary_writer.add_summary(summary, generator_step)
    summary_writer.flush()


# Displays a figure of the generated images and saves them in as .png image
def save_generated_images(generated_images, generator_iteration):

    # Create the plot
    plt.figure(figsize=(8, 8), num=1)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    save_name = 'generated images/generatedSamples_genIter' + \
        str(generator_iteration + 1) + '.png'

    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.pause(0.0000000001)
    plt.show()


# Main train function
def train_wgan(batch_size, epochs, image_shape, dataset_path):

    # Build the adversarial model that consists in the generator output
    # connected to the critic
    generator = construct_generator()
    critic = construct_critic(image_shape)

    gan = Sequential()
    # Only false for the adversarial model
    critic.trainable = False
    gan.add(generator)
    gan.add(critic)

    optimizer = RMSprop(lr=0.00005)
    gan.compile(loss=wasserstein_loss,
                optimizer=optimizer,
                metrics=None)

    # Create a dataset Generator with help of keras
    dataset_generator = load_dataset(dataset_path, batch_size, image_shape)

    # 11788 is the total number of images on the bird dataset
    number_of_batches = int(11788 / batch_size)

    # Tensorboard log variable
    summary_writer = tf.summary.FileWriter('./logs/WGAN')

    # Create the plot that will show the losses
    plt.ion()

    # Variables used for loss saving
    generator_iterations = 0
    d_loss = 0
    d_real = 0
    d_fake = 0
    g_loss = 0

    # Let's train the WGAN for n epochs
    for epoch in range(epochs):

        current_batch = 0

        while current_batch < number_of_batches:

            start_time = time.time()

            # Just like the v2 version of paper, in the first 25 epochs, the critic
            # is updated 100 times for each generator update. Occasionally (each 500
            # epochs this is repeated). In the other epochs the default value is 5
            if generator_iterations < 25 or (generator_iterations + 1) % 500 == 0:
                critic_iterations = 100
            else:
                critic_iterations = 5

            # Update the critic a number of critic iterations
            for critic_iteration in range(critic_iterations):

                if current_batch > number_of_batches:
                    break

                real_images = dataset_generator.next()
                real_images /= 127.5
                real_images -= 1
                current_batch += 1

                # The last batch is smaller than the other ones, so we need to
                # take that into account
                current_batch_size = real_images.shape[0]

                # Generate noise
                noise = np.random.normal(0, 1,
                                         size=(current_batch_size,) + (1, 1, 100))

                # Generate images
                generated_images = generator.predict(noise)

                # Add some noise to the labels that will be fed to the critic
                real_y = np.ones(current_batch_size)
                fake_y = np.ones(current_batch_size) * -1

                # Let's train the critic
                critic.trainable = True

                # Clip the weights to small numbers near zero
                for layer in critic.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -0.01, 0.01) for w in weights]
                    layer.set_weights(weights)

                d_real = critic.train_on_batch(real_images, real_y)
                d_fake = critic.train_on_batch(generated_images, fake_y)

                d_loss = d_real - d_fake

            # numpy array that will store the losses to be passed to tensorboard
            losses = np.empty(shape=1)
            losses = np.append(losses, d_real)
            losses = np.append(losses, d_fake)

            # Update the generator
            critic.trainable = False

            noise = np.random.normal(0, 1,
                                     size=(current_batch_size,) + (1, 1, 100))

            # We try to mislead the critic by giving the opposite labels
            fake_y = np.ones(current_batch_size)
            g_loss = gan.train_on_batch(noise, fake_y)

            losses = np.append(losses, g_loss)

            # Each 100 generator iterations show and save images
            if ((generator_iterations + 1) % 100 == 0):
                noise = np.random.normal(0, 1, size=(64,) + (1, 1, 100))
                generated_images = generator.predict(noise)
                save_generated_images(generated_images, generator_iterations)

            # Update tensorboard plots
            write_to_tensorboard(generator_iterations, summary_writer, losses)

            time_elapsed = time.time() - start_time
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f - %f s'
                  % (epoch, epochs, current_batch, number_of_batches, generator_iterations,
                     d_loss, g_loss, d_real, d_fake, time_elapsed))

            generator_iterations += 1

        if (epoch + 1) % 5 == 0:
            critic.trainable = True
            generator.save('models/generator_epoch' + str(epoch) + '.hdf5')
            critic.save('models/critic_epoch' + str(epoch) + '.hdf5')


def main():
    dataset_path = '/media/tfreitas/LENOVO/Datasets/CUB_200_2011/CUB_200_2011/images/'
    batch_size = 64
    image_shape = (64, 64, 3)
    epochs = 5000
    train_wgan(batch_size, epochs,
               image_shape, dataset_path)
    K.clear_session()


if __name__ == "__main__":
    main()
