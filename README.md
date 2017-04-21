# DCGAN (WGAN)
TensorFlow 1.0 compatible implementation

It currently works well on LSUN for vanilla gan loss, and converges slow for wgan loss (too small learning rate) :sweat_smile: .

To run WGAN, set 
```
d_learningrate=5e-5
g_learningrate=5e-5
Wloss=True
Adam=False
noise='normal'
``` 
and
```
f_h=4, f_w=4, Cc=64
``` 
in both generator and discriminator (same setting as the original [torch implementation](https://github.com/martinarjovsky/WassersteinGAN)).

## Results of GAN Loss

After 1.5 Epoch with Batch Normalization:

![DCGAN with BN](https://github.com/lovecambi/dcgan/blob/master/imgs/dcgan_BN1.5ep.jpg)

After 1 Epoch without Batch Normalization (but with bias; if no bias, the generated image is still sharp but less light):

![DCGAN no BN](https://github.com/lovecambi/DCGAN/blob/master/imgs/dcgan_noBN1ep.jpg)

## Results of WGAN Loss

After 4 Epochs with Batch Normalization:

![WGAN with BN](https://github.com/lovecambi/DCGAN/blob/master/imgs/WGAN_BN4ep.jpg)
