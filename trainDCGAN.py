from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Here is where we will load the dataset stored in datasetPath. In this script
# we will use the Caltech-UCSD Birds-200-2011 dataset which includes 11788
# images from 200 different birds. We will feed the images without applying
# the provided bounding boxes from the dataset. The data will only be resized
# and normalized. Keras ImageDataGenerator will be used for loading the dataset
def loadDataset(datasetPath, batchSize, image_shape):

    datasetGenerator = ImageDataGenerator()
    datasetGenerator = datasetGenerator.flow_from_directory(datasetPath,
                                                            target_size=(image_shape[0], image_shape[1]),
                                                            batch_size = batchSize,
                                                            class_mode=None)

    return datasetGenerator

# Creates the discriminator model. This model tries to classify images as real
# or fake.
def constructDiscriminator(image_shape):

    discriminator = Sequential()
    discriminator.add(Conv2D(filters = 64, kernel_size = (5, 5),
                             strides = (2, 2), padding='same',
                             data_format = 'channels_last',
                             kernel_initializer = 'glorot_uniform',
                             input_shape = (image_shape)))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters = 128, kernel_size = (5, 5),
                             strides = (2, 2), padding = 'same',
                             data_format = 'channels_last',
                             kernel_initializer = 'glorot_uniform'))
    discriminator.add(BatchNormalization(momentum = 0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters = 256, kernel_size = (5, 5),
                             strides = (2, 2), padding = 'same',
                             data_format = 'channels_last',
                             kernel_initializer = 'glorot_uniform'))
    discriminator.add(BatchNormalization(momentum = 0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Conv2D(filters = 512, kernel_size = (5, 5),
                             strides = (2, 2), padding = 'same',
                             data_format = 'channels_last',
                             kernel_initializer = 'glorot_uniform'))
    discriminator.add(BatchNormalization(momentum = 0.5))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Flatten())
    discriminator.add(Dense(1))
    discriminator.add(Activation('sigmoid'))


    optimizer = Adam(lr = 0.0002, beta_1 = 0.5)
    discriminator.compile(loss = 'binary_crossentropy',
                          optimizer = optimizer,
                          metrics=None)

    return discriminator

# Creates the generator model. This model has an input of random noise and
# generates an image that will try mislead the discriminator.
def constructGenerator(image_shape):

    generator = Sequential()

    generator.add(Dense(units = 4 * 4 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape = (1, 1, 100)))
    generator.add(Reshape(target_shape = (4, 4, 512)))
    generator.add(BatchNormalization(momentum = 0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters = 256, kernel_size = (5, 5),
                                  strides = (2, 2), padding='same',
                                  data_format = 'channels_last',
                                  kernel_initializer = 'glorot_uniform'))
    generator.add(BatchNormalization(momentum = 0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters = 128, kernel_size = (5, 5),
                                  strides = (2, 2), padding='same',
                                  data_format = 'channels_last',
                                  kernel_initializer = 'glorot_uniform'))
    generator.add(BatchNormalization(momentum = 0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters = 64, kernel_size = (5, 5),
                                  strides = (2, 2), padding='same',
                                  data_format = 'channels_last',
                                  kernel_initializer = 'glorot_uniform'))
    generator.add(BatchNormalization(momentum = 0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters = 3, kernel_size = (5, 5),
                                  strides = (2, 2), padding='same',
                                  data_format = 'channels_last',
                                  kernel_initializer = 'glorot_uniform'))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr = 0.00015, beta_1 = 0.5)
    generator.compile(loss = 'binary_crossentropy',
                      optimizer = optimizer,
                      metrics=None)

    return generator

# Displays a figure of the generated images and saves them in as .png image
def saveGeneratedImages(generatedImages, epoch, batchNumber):

    plt.figure(figsize = (8, 8), num = 2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace = 0, hspace = 0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generatedImages[i, :,:,:]
        image += 1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    saveName = 'generated images/generatedSamples_epoch' + str(epoch + 1) + '_batch' + str(batchNumber + 1) + '.png'
    plt.savefig(saveName, bbox_inches = 'tight', pad_inches = 0)
    plt.pause(0.0000000001)
    plt.show()

def trainDCGAN(batchSize, epochs,
               imageShape, datasetPath):

    # Build the adversarial model that consists in the generator output
    # connected to the discriminator
    generator = constructGenerator(imageShape)
    discriminator = constructDiscriminator(imageShape)

    GAN = Sequential()
    # Only false for the adversarial model
    discriminator.trainable = False
    GAN.add(generator)
    GAN.add(discriminator)

    optimizer = Adam(lr = 0.00015, beta_1 = 0.5)
    GAN.compile(loss = 'binary_crossentropy', optimizer = optimizer,
                metrics=None)

    # Create a dataset Generator with help of keras
    datasetGenerator = loadDataset(datasetPath, batchSize, imageShape)

    # 11788 is the total number of images on the bird dataset
    numberOfBatches = int(11788 / batchSize)

    # Variables that will be used to plot the losses from the discriminator and
    # the adversarial models
    adversarialLoss = np.empty(shape=1)
    discriminatorLoss = np.empty(shape=1)
    batches = np.empty(shape=1)
    # Create the plot that will show the losses
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()

    currentBatch = 0;

    # Let's train the DCGAN for n epochs
    for epoch in range(epochs):

        print("Epoch " + str(epoch+1) + "/" + str(epochs) + " :")

        for batchNumber in range(numberOfBatches):

            startTime = time.time()

            # Get the current batch and normalize the images between -1 and 1
            realImages = datasetGenerator.next()
            realImages /= 127.5
            realImages -= 1

            # The last batch is smaller than the other ones, so we need to
            # take that into account
            currentBatchSize = realImages.shape[0]

            # Generate noise
            noise = np.random.normal(0, 1,
                                     size=(currentBatchSize,) + (1,1,100))

            # Generate images
            generatedImages = generator.predict(noise)

            # Each 20 batches show and save images
            if ((batchNumber + 1) % 50 == 0) and currentBatchSize == batchSize:
                saveGeneratedImages(generatedImages, epoch, batchNumber)

            # Add some noise to the labels that will be fed to the discriminator
            real_y = np.ones(currentBatchSize) - np.random.random_sample(currentBatchSize) * 0.2
            fake_y = np.random.random_sample(currentBatchSize)*0.2

            # Let's train the discriminator
            discriminator.trainable = True

            d_loss = discriminator.train_on_batch(realImages, real_y)
            d_loss += discriminator.train_on_batch(generatedImages, fake_y)

            discriminatorLoss = np.append(discriminatorLoss, d_loss)

            # Now it's time to train the generator
            discriminator.trainable = False

            noise = np.random.normal(0, 1,
                                     size=(currentBatchSize*2,) + (1,1,100))

            # We try to mislead the discriminator by giving the opposite labels
            fake_y = np.ones(currentBatchSize*2) - np.random.random_sample(currentBatchSize*2) * 0.2

            g_loss = GAN.train_on_batch(noise, fake_y)
            adversarialLoss = np.append(adversarialLoss, g_loss)
            batches = np.append(batches, currentBatch)

            time_elapsed = time.time() - startTime

            # Display and plot the results
            print("     Batch " +  str(batchNumber+1) + "/" +
                  str(numberOfBatches) +
                  " generator loss | discriminator loss : " +
                  str(g_loss) + " | " + str(d_loss) + ' - batch took ' +
                  str(time_elapsed) + ' s.')

            currentBatch += 1

        if (epoch + 1) % 5 == 0:
            discriminator.trainable = True
            generator.save('models/generator_epoch' + str(epoch) + '.hdf5')
            discriminator.save('models/discriminator_epoch'+ str(epoch) + '.hdf5')

        plt.figure(1)
        plt.plot(batches, adversarialLoss, color='green', label = 'Generator Loss')
        plt.plot(batches, discriminatorLoss, color='blue', label = 'Discriminator Loss')
        plt.title("DCGAN Train")
        plt.xlabel("Batch Iteration")
        plt.ylabel("Loss")
        if epoch == 0:
            plt.legend()
        plt.pause(0.0000000001)
        plt.show()

        plt.savefig('trainingLossPlot.png')


def main():
    datasetPath = '/media/tfreitas/LENOVO/Datasets/CUB_200_2011/CUB_200_2011/images/'
    batchSize = 64
    imageShape = (64, 64, 3)
    epochs = 190
    trainDCGAN(batchSize, epochs,
               imageShape, datasetPath)

if __name__ == "__main__":
    main()
