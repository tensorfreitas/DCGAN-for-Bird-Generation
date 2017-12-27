# DCGAN for Bird Generation

This repository was created for me to familiarize with DCGANs and its peculiarities. The code uses Keras library and the Caltech-UCSD Birds-200-2011 dataset.

## Caltech-UCSD Birds-200-2011 dataset

This dataset has 11788 images from 200 different birds. It includes also some additional information such as segmentations, attributes and bounding boxes that will not be used in this project. Here are some of the different images in this dataset:

<p align="center">
  <img src="http://www.vision.caltech.edu/visipedia/collage.jpg" alt="CaltechBirdExample"/>
</p>

I was looking for a different dataset for my personal experiments, and when I found it, it seemed like a good idea to try generate birds from this dataset.

## DCGANS

I tried to implement a DCGAN (Deep Convolutional GAN) since most of GANs are at least loosely based on the DCGAN architecture. The original DCGAN architecture proposed by Radford et al. (2015), can be shown in the next image:

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/33244947-c882506e-d2f8-11e7-89e4-611fdc7dacd9.png" alt="dcgan_arch"/>
</p>

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

Although I tried different architectures based on other papers and repositories, the best result was achieved using a traditional DCGAN architecture (see the code for more datails)

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

Each epoch, 3 bathes of generated images are saved in the disk as .png files and each 5 epochs the models are saved in the disk. Also the loss plot is saved in the end of each epoch.

It is important to have the folders generated images and models in the same path of the trainDCGAN.py file. 

## WGANs (Under Construction)

After training the DCGAN, I wanted to compare the results with the WGAN proposed by Arjovsky et al. (2017), therefore I also implemented code for WGAN training in Keras. 

In this paper, it is proposed a meaningful loss function, which helps debugging GANs and tuning its parameters and architectures. This loss function, the Wasserstein loss (or Earth-Moving loss), is shown to have a relationship with image quality. The use of this new loss function has proven to avoid mode collapses (generating similar images) even when fully connected layer GANS are used or when batch normalization is not used.

The word discriminator in this paper is replaced by critic since its purpose is not to classify directly if the image is true or fake – we have no longer a binary classification problem. The critic no longer needs to output a probability. When training the critic, the labels provided for the real images are 1’s and for the fake images are negative 1’s. 

Fake and real images are interpreted as coming from a probability distribution, the WGAN tries to measure the difference between these distributions and minimize it (by minimizing the loss). Since the computation of Wasserstein distance is intractable, the paper uses an approximation:

```
K.mean(y_true * y_predicted)
```

This only works if the network used is K-Lipschitz. In order this to be true, weights are clamped in the discriminator to very small values near zero. The authors admit this is not the ideal solution, but in practice works (notice that the improved version of the WGAN resolves this problem in a more interesting way). 

In the traditional DCGANs to get good images we need to balance the train between the generator and the discriminator. With the new Wasserstein loss, this problem is solved: since the Wasserstein distance is almost every time differentiable, the critic can be train to convergence before training the generator, since the gradients become more accurate in this process. 

Regarding the training some topics need to be referred:

- The critic is trained for 100 times for each generator train iteration in the first 25 generator train updates. After this, it is trained for 5 iterations for each generator iteration (every 500 generator iterations the critic is trained again for 100 times). 
- Each critic iteration the weights are clipped to smaller values near zero (-0.01 to 0.01)
- The optimizer should not use momentum and a low learning rate should be used
- Training is very slow due to the learning rate and the number of critic updates in each generator update. 

The loss function is expected is similar to the shown by the paper:

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/34366989-3fe579b4-ea9c-11e7-9b49-00f9c7712f48.png" alt="WGAN_loss"/>
</p>

Additionally there are some interesting facts that I discovered that I think might help someone:

- Since we have no longer a classification problem the loss should no longer tend to 0.5. For a perfect generator the loss should be 0. 
- When the critic isn't trained optimally, the loss curves aren't as smooth as shown in the paper, leading to the presence of peaks. This can occur due to high learning rates, momentum of the optimizer and low number of discriminator iterations. In his github repo, the original author refers that: "when the curve is making a big jump like this and keep iterating the critic till the curve goes up again to roughly were it was before the jump."

## Results

Evolution of generated images along 190 epochs:

<p align="center">
  <img src="https://github.com/Goldesel23/DCGAN-for-Bird-Generation/blob/master/generatedImages2.gif" alt="ResultsGIF"/>
</p>

Here are some of the Images Generated in the End of the Training:

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/33324399-6cbeb88a-d447-11e7-8717-a8a5455036fe.png" alt="generatedsamples_epoch190_batch50"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/33324400-6ce27144-d447-11e7-9a6f-a6faa15d04f9.png" alt="generatedsamples_epoch190_batch100"/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/33324401-6d0c8a60-d447-11e7-8450-9f171e7bec17.png" alt="generatedsamples_epoch190_batch150"/>
</p>

And here is the generator and discriminator loss evolution during training:

<p align="center">
  <img src="https://user-images.githubusercontent.com/10371630/33324663-279c9262-d448-11e7-9b0a-7b5c375f7c7c.png" alt="traininglossplot"/>
</p>

The training was done in a laptop with Ubuntu 16.04 with a NVIDIA GTX-960M and it took a dozen of hours. I am quite satisfied with the result. The images are a little blurred but it seems like the generator learnt how to make artistic bird images. 

The results could be better but I'm quite satisfied with the mutant birds it generated. You can say what you want but these are my mutant birds... Despite that I'll try to make improvements in the future to the DCGAN when I have some time. 

### WGAN Results (Under Construction)

## Possible Improvements

If I have some time later I will probably try some of these:

- Use the cropped images with the dataset bounding boxes instead of the full images. 
- Use Wassestein GANs to get images with better image quality and keep track of the convergence during training. 
- Use the label information to train a better Generator, for example using Auxilliary GANs or Conditional GANs
- Explore different types of architectures

## References
- Wah C., Branson S., Welinder P., Perona P., Belongie S. “The Caltech-UCSD Birds-200-2011 Dataset.” Computation & Neural Systems Technical Report, CNS-TR-2011-001
- Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
- Goodfellow, Ian. "NIPS 2016 tutorial: Generative adversarial networks." arXiv preprint arXiv:1701.00160 (2016).
- Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein gan." arXiv preprint arXiv:1701.07875 (2017).

## Credits

Of course I have based my code on other projects and repositories. I would like to give credit to some of them: 

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/Newmu/dcgan_code)
- [DCGAN on Tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [Anime-Face-GAN-Keras](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)
- [Keras-DCGAN](https://github.com/jacobgil/keras-dcgan)
- [GAN-Sandbox](https://github.com/wayaai/GAN-Sandbox)
- [Original Pytorch Implementation of WGAN](https://github.com/martinarjovsky/WassersteinGAN)
- [Wasserstein GAN in Keras - Blog post](https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html)
- [Read-through: Wasserstein GAN - Blog post](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)
- [Tensorflow implementation of WGAN](https://github.com/shekkizh/WassersteinGAN.tensorflow)
