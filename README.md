# dcgan (with wgan loss)
TensorFlow 0.12.1  implementation

It currently works well on LSUN for vanilla gan loss.

To run WGAN, it is better to set ```d_learningrate=1e-5, g_learningrate=1e-5, z~norm``` and ```f_h=4, f_w=4, Cc=64``` in both generator and discriminator (same setting as the original [torch implementation](https://github.com/martinarjovsky/WassersteinGAN)).

## Results of GAN Loss

After 1.5 Epoch with Batch Normalization:

![DCGAN with BN](https://github.com/lovecambi/dcgan/blob/master/imgs/dcgan_BN1.5ep.jpg)

After 1 Epoch without Batch Normalization (but conv and deconv with bias):

![DCGAN no BN](https://github.com/lovecambi/dcgan/blob/master/imgs/dcgan_noBN1ep.jpg)
