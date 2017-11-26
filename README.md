# DCGAN for Bird Generation (Under construction)

This repository was created for me to familiarize with DCGANs and its peculiarities. The code uses Keras library and uses the Caltech-UCSD Birds-200-2011 dataset.

## Caltech-UCSD Birds-200-2011 dataset

This dataset has 11788 images from 200 different birds. It includes also some additional information such as segmentations, attributes and bounding boxes that will not be used in this project. Here is a look of the different images in this dataset:

![Alt text](http://www.vision.caltech.edu/visipedia/collage.jpg)

I was looking for a different dataset for my personal experiments, and when I found it, it seemed like a good idea to try generate birds from this dataset.

## DCGANS

I tried to implement a DCGAN (Deep Convolutional GAN) since most of GANs are at least loosely based on the DCGAN architecture. The original DCGAN architecture proposed by Radford et al. (2015), can be shown in the next image:

(ADD IMAGE)

Like traditional GANs, DCGANs consist in a discriminator which tries to classify images as real or fake, and a generator that tries to produce samples that will by fed to the discriminator trying to mislead it. DCGANs usually refer to this specific style of architecture shown in the previous figure, since previously proposed GANs were also deep and convolutional prior to this work. 

From the original paper we can find some tips to make DCGAN work:

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided
convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- The last layer of the generator and first layer of the discriminator are not batch normalized, so that the model can learn the correct mean and scale of the data distribution
- Remove fully connected hidden layers for deeper architectures
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.
- The use of the Adam optimizer rather than SGD with momentum.

### DCGAN train

Although I tried different architectures based on other papers and repositories, the best result was achieved using a traditional DCGAN architecture with LeakyReLu's in both the discriminator and the generator. 

In each training iteration:
1. A batch of images of the original dataset is obtained and is normalized between -1 and 1.
2. A number of noise vectors equal to the batch size, each one with size (1, 1, 100), is generated. These vectors will be the ones that will generate the new images.
3. The noise vectors are used by the generator to create a batch of generated fake images.
4. Add some noise to the labels of both real and fake images (instead of just giving ones and zeros to the discriminator)
5. Train the discriminator with the real and fake images separately.
6. Generate a number of batch size x 2 of noise vectors, that will be used to train the generator.
7. Assign opposite labels to the generated images with noise (we are trying to mislead the discriminator)
8. Train the discriminator with the generated images. 

#### Notes

Each 50 iterations the generated images are saved in the disk as .png files and each 5 epochs the models are saved in the disk. It is important to have the folders generated images and models in the same path of the trainDCGAN.py file. Also the loss plot is saved in the end of each epoch.

## Results



## References
- Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001
- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
- Goodfellow, Ian. "NIPS 2016 tutorial: Generative adversarial networks." arXiv preprint arXiv:1701.00160 (2016).

## Credits

Of course I have based my code on other projects and repositories. I would like to give credit to some of them: 

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
