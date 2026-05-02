# EMBEDDING FOURIER FOR ULTRA-HIGH-DEFINITION LOW-LIGHT IMAGE ENHANCEMENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Ultra-High-Definition (UHD) photo has gradually become the standard configuration in advanced imaging devices. The new standard unveils many issues in existing approaches for low-light image enhancement (LLIE), especially in dealing with the intricate issue of joint luminance enhancement and noise removal while remaining efficient. Unlike existing methods that address the problem in the spatial domain, we propose a new solution, UHDFour, that embeds Fourier transform into a cascaded network. Our approach is motivated by a few unique characteristics in the Fourier domain: 1) most luminance information concentrates on amplitudes while noise is closely related to phases, and 2) a high-resolution image and its low-resolution version share similar amplitude patterns. Through embedding Fourier into our network, the amplitude and phase of a low-light image are separately processed to avoid amplifying noise when enhancing luminance. Besides, UHDFour is scalable to UHD images by implementing amplitude and phase enhancement under the low-resolution regime and then adjusting the high-resolution scale with few computations. We also contribute the first real UHD LLIE dataset, UHD-LL, that contains 2,150 low-noise/normal-clear 4K image pairs with diverse darkness and noise levels captured in different scenarios. With this dataset, we systematically analyze the performance of existing LLIE methods for processing UHD images and demonstrate the advantage of our solution. We believe our new framework, coupled with the dataset, would push the frontier of LLIE towards UHD. Code and the dataset will be released.

# 1 INTRODUCTION

With the advent of advanced imaging sensors and displays, Ultra-High-Definition (UHD) imaging has witnessed rapid development in recent years. While UHD imaging offers broad applications and makes a significant difference in picture quality, the extra pixels also challenge the efficiency of existing image processing algorithms.

In this study, we focus on one of the most challenging tasks in image restoration, namely low-light image enhancement (LLIE), where one needs to jointly enhance the luminance and remove inherent noises caused by sensors and dim environments. Further to these challenges, we lift the difficulty by demanding efficient processing in the UHD regime.

Despite the remarkable progress in low-light image enhancement (LLIE) (Li et al., 2021a), existing methods, as shown in Figure 1, show apparent drawbacks when they are used to process real-world UHD low-light images. This is because (1) most methods (Guo et al., 2020; Liu et al., 2021b; Ma et al., 2022) only focus on luminance enhancement and fail in removing noise; (2) some approaches (Wu et al., 2022; Xu et al., 2022) simultaneously enhance luminance and remove noise in the spatial domain, resulting in the suboptimal enhancement; and (3) existing methods are mainly trained on low-resolution (LR) data, leading to the incompatibility with high-resolution (HR) inputs; and (4) some studies adopt heavy structures, thus being inefficient for processing UHD images. More discussion on related work is provided in the Appendix.

To overcome the challenges aforementioned, we present a new idea for performing LLIE in the Fourier Domain. Our approach differs significantly from existing solutions that process images in the spatial domain. In particular, our method, named as UHDFour, is motivated by our observation of two interesting phenomena in the Fourier domain of low-light noisy images: i) luminance and noise can be decomposed to a certain extent in the Fourier domain. Specifically, luminance would manifest as amplitude while noise is closely related to phase, and ii) the amplitude patterns of images

![](images/51e451ffd9ced8d0594b2e562b6d0eb3c4608c9427c3c6c3d4e10e0e55799391.jpg)  
Figure 1: Visual results of state of the arts (Zhao et al. (Zhao et al., 2021), URetinex-Net (Wu et al., 2022), and SNR-Aware (Xu et al., 2022)) pre-trained on an existing low-light image dataset for processing the real-world UHD low-light images. We amplify the brightness of the input UHD low-light images 10 times (top right corner of the first column) to show details and noise. These officially released models were trained using existing paired LR images with mild noise (i.e., the LOL dataset (Wei et al., 2018)). Existing models cannot cope with challenging UHD low-light images well.

![](images/883c0e3dc3f3552e1552e877ac1b15a8458f606f0f0c3021e216d100aa1384bf.jpg)

![](images/43a0fe6234e1406d2ad6037de35432311eb13e7c25af84c3025ffe6496095f22.jpg)

![](images/758b11869a5c59a8ab54475e459c5bd4be26eb44d22fc24ab5c177c1f2fc79ed.jpg)

![](images/164472d7ec5f08117b102d1a458914d69d9efbf9a521ae53cd70d3fe260a0925.jpg)

of different resolutions are similar. These observations inspire the design of our network, which handles luminance and noise separately in the Fourier domain. This design is advantageous as it avoids amplifying noise when enhancing luminance, a common issue encountered in existing spatial domain-based methods. In addition, the fact that amplitude patterns of images of different resolutions are similar motivates us to save computation by first processing in the low-resolution regime and performing essential adjustments only in the high-resolution scale.

We also contribute the first benchmark for UHD LLIE. The dataset, named UHD-LL, contains 2,150 low-noise/normal-clear 4K UHD image pairs with diverse darkness and noise levels captured in different scenarios. Unlike existing datasets that either synthesize or retouch low-light images to obtain the paired input and target sets, we capture real image pairs. During data acquisition, special care is implemented to minimize geometric and photometric misalignment due to camera shake and dynamic environment. With the new UHD-LL dataset, we design a series of quantitative and quantitative benchmarks to analyze the performance of existing LLIE methods and demonstrate the effectiveness of our method.

Our contributions are summarized as follows: (1) We propose a new solution for UHD LLIE that is inspired by unique characteristics observed in the Fourier domain. In comparison to existing LLIE methods, the proposed framework shows exceptional effectiveness and efficiency in addressing the joint task of luminance enhancement and noise removal in the UHD regime. (2) We contribute the first UHD LLIE dataset, which contains 2,150 pairs of 4K UHD low-noise/normal-clear data, covering diverse noise and darkness levels and scenes. (3) We conduct a systematical analysis of existing LLIE methods on UHD data.

# 2 OUR APPROACH

In this section, we first discuss our observations in analyzing low-light images in the Fourier domain, and then present the proposed solution.

# 2.1 OBSERVATIONS IN FOURIER DOMAIN

Here we provide more details to supplement the observations we highlighted in Sec. 1. We analyze real UHD low-light images in the Fourier domain and provide a concise illustration in Figure 2. Specifically, (a) Swapping the amplitude of a low-light and noisy (low-noise) image with that of its corresponding normal-light and clear (normal-clear) image produces a normal-light and noisy (normal-noise) image and a low-light and clear (low-clear) image. We show more examples in the Appendix. The result suggests that the luminance and noise can be decomposed to a certain extent in the Fourier domain. In particular, most luminance information is expressed as amplitudes, and noises are revealed in phases. This inspires us to process luminance and noise separately in the Fourier domain. (b) The amplitude patterns of an HR normal-clear image and its LR versions are similar and are different from the corresponding HR low-noise counterpart. Such a characteristic offers us the possibility to first enhance the amplitude of an LR scale with more computations and then only make minor adjustments in the HR scale. In this way, most computations can be conducted in the LR space, reducing the computational complexity.

![](images/f3591c6254405454fa58af15f1c89dad68863862a6df3a7d955a6615d1eab62b.jpg)  
Figure 2: Motivations. We observed that (a) luminance and noise can be 'decomposed' to a certain extent in the Fourier domain and (b) HR image and its LR versions share similar amplitude patterns. The amplitude and phase are produced by Fast Fourier Transform (FFT) and the compositional images are obtained by Inverse FFT (IFFT). For visualization, we show the amplitude and phase in imagery format with common transformations. Lines of the same color indicate a set of FFT/IFFT transforms. The red triangles mark the similar pattern (obviously different from the gray one). Zoom in for the details and noise. We show more examples and analysis in the Appendix.

![](images/b4e6fe4870601e92fec9776228ff3e333ff687c80257a6c1a99b3d9f742a28bc.jpg)

# 2.2 THE UHDFOUR NETWORK

![](images/7f6a62c84a56cbf0ab33d6ae38d691d9ad69755317c36d56e2e5f2c3c7b0cda1.jpg)  
Figure 3: Overview of UHDFour. Our approach consists of an LRNet and an HRDNet. The LRNet is an encoder-decoder network that produces  $8 \times$  downsampled result  $\hat{y}_8$  and the refined amplitude  $A_r$  and phase  $P_r$  features. We omit the skip connections for brevity. The HRNet contains an Adjustment Block and the upsampling operation, producing the final result  $\hat{y}$ . Most computation is conducted in the LRNet.

Overview. UHDFour aims to map an UHD low-noise input image  $x \in \mathbb{R}^{H \times W \times C}$  to its corresponding normal-clear version  $y \in \mathbb{R}^{H \times W \times C}$ , where  $H, W,$  and  $C$  represent height, width, and channel, respectively. Figure 3 shows the overview of UHDFour. It consists of an LRNet and an HRNet.

Motivated by the observation in Sec. 2.1, LRNet takes the most computation of the whole network. Its input is first embedded into the feature domain by a Conv layer. To reduce computational complexity, we downsample the features to  $1/8$  of the original resolution by bilinear interpolation. Then, the LR features go through an encoder-decoder network, which contains four FouSpa Blocks with two  $2 \times$  downsample and two  $2 \times$  upsample operations, obtaining outputs features. The outputs features are respectively fed to FFT to obtain the refined amplitude  $A_r$  and phase  $P_r$  features and a Conv layer to estimate the LR normal-clear image  $\hat{y}_8 \in \mathbb{R}^{H/8 \times W/8 \times C}$ .

The outputs of LRNet coupled with the input are fed to the HRNet. Specifically, the input  $x$  is first reshaped to  $x_{pu} \in \mathbb{R}^{H \times W \times C \times 64}$  via PixelUnshuffle ( $8 \times \downarrow$ ) to preserve original information, and then fed to an Adjustment Block. With the refined amplitude  $A_r$  and phase  $P_r$  features, the Adjustment Block produces adjusted features that are reshaped to the original height and width of input  $x$  via Pixelshuffle ( $8 \times \uparrow$ ). Finally, we resize the estimated LR normal-clear image  $y_8$  to the original size of input  $x$  via bilinear interpolation and combine it with the upsampled features to estimate the final HR normal-clear image  $\hat{y}$ . We detail the key components as follows.

FouSpa Block. In Sec. 2.1, we observe that luminance and noise can be decomposed in the Fourier domain. Hence, we design the FouSpa Block to parallelly implement amplitude and phase enhancement in the Fourier domain and feature enhancement in the spatial domain. As shown in Figure 4(a), the input features are forked into the Fourier and Spatial branches. In the Fourier branch, FFT is first used to obtain the amplitude component  $(A)$  and phase component  $(P)$ . The two components

![](images/7df934fe74b4a5de6861f6e15ffa672e5579a80329234ea4e20eea037f9ab6e5.jpg)  
Figure 4: Structures of the FouSpa Block (a) and Adjustment Block (b).

![](images/ffbb0ad7746e0a6241d067c08d4dae0d2cbcadfda113982040dcd58ca9dede21.jpg)

are separately fed to two Conv layers with  $1 \times 1$  kernel. Note that when processing amplitude and phase, we only use  $1 \times 1$  kernel to avoid damaging the structure information. Then, we transform them back to the spatial domain via IFFT and concatenate them with the spatial features enhanced by a Half Instance Normalization (HIN) unit (Chen et al., 2021). We adopt the HIN unit based on its efficiency. The concatenated features are further fed to a Conv layer and then combined with the input features in a residual manner.

Adjustment Block. The Adjustment Block is the main structure of the HRNet, and it is lightweight. As shown in Figure 4(b), the Adjustment Block shares a similar structure with the FouSpa Block. Differently, in the Fourier branch, with the refined amplitude  $A_{r}$  features obtained from the LRNet, we use Spatial Feature Transform (SFT) (Wang et al., 2018) to modulate the amplitude features of the input  $x_{pu}$  via simple affine transformation. Such a transformation or adjustment is possible because the luminance, as global information, manifests as amplitude components, and the amplitude patterns of an HR scale and its LR scales are similar (as discussed in Sec. 2.1). Note that we cannot modulate the phase because of its periodicity. Besides, we do not find an explicit relationship between the HR scale's phase and its LR scales. However, we empirically find that concatenating the refined phase  $P_{r}$  features achieved from the LRNet with the phase features of the input  $x_{pu}$  improves the final performance. We thus apply such concatenation in our solution.

Losses. We use  $l_{1}$  to supervise  $\hat{y}_{8}$  and  $\hat{y}$ . We also add perceptual loss to supervise  $\hat{y}_{8}$  while the use of perceptual loss on  $\hat{y}$  is impracticable because of its high resolution. Instead, we add SSIM loss  $\mathcal{L}_{ssim}$  on  $\hat{y}$ . The final loss  $\mathcal{L}$  is the combination of these losses:

$$
\mathcal {L} = \left\| \hat {y} - y \right\| _ {1} + 0. 0 0 0 4 \times \mathcal {L} _ {\text {s s i m}} (\hat {y}, y) + 0. 1 \times \left\| \hat {y} _ {8} - y _ {8} \right\| _ {1} + 0. 0 0 0 2 \times \left\| \mathrm {V G G} \left(\hat {y} _ {8}\right) - \mathrm {V G G} \left(y _ {8}\right) \right\| _ {2}, \tag {1}
$$

where  $y$  is the ground truth,  $y_{8}$  is the  $8\times$  downsampled version of  $y$ , VGG is the pre-trained VGG19 network, in which we use four scales to supervise training.

# 3 UHD-LL DATASET

We collect a real low-noise/normal-clear paired image dataset that contains 2,150 pairs of 4K UHD data. Several samples are shown in Figure 5.

Images are collected from a camera mounted on a tripod to ensure stability. Two cameras, i.e., a Sony  $\alpha 7$  III camera and a Sony Alpha a6300 camera, are used to offer diversity. The ground truth (or normal-clear) image is captured with a small ISO  $\in [100,800]$  in a bright scene (indoor or outdoor). The corresponding low-noise image is acquired by increasing the ISO  $\in [1000,20000]$  and reducing the exposure time. Due to the constraints of exposure gears in the cameras, shooting in the large ISO range may produce bright images, which opposes the purpose of capturing low-light and noisy images. Thus, in some cases, we put a neutral-density (ND) filter with different ratios on the camera lens to capture low-noise images. In this way, we can increase the ISO to generate heavier noises and simultaneously obtain extremely dark images, enriching the diversity of darkness and noise levels.

The main challenge of collecting real paired data is to reduce misalignment caused by camera shakes and dynamic objects. We take several measures to ameliorate the issue. In particular, apart from

Table 1: Comparison between classic LLIE datasets and our UHD-LL dataset. 'Number': the number of paired images. 'Resolution': the average resolution of the dataset. 'Noise': low-light images contain noise. 'Real': both low-light images and GT are acquired in real scenes.  

<table><tr><td>Dataset</td><td>Number</td><td>Resolution</td><td>Noise</td><td>Real</td></tr><tr><td>MIT-Adobe FiveK</td><td>5,000</td><td>4000×2500</td><td></td><td></td></tr><tr><td>Exposure-Errors</td><td>24,000</td><td>1000×900</td><td></td><td></td></tr><tr><td>LOL</td><td>500/789</td><td>600×400</td><td>✓</td><td>✓</td></tr><tr><td>UHD-LL (Ours)</td><td>2,150</td><td>3840×2160</td><td>✓</td><td>✓</td></tr></table>

![](images/4d712720f42cae24a0b0a4c8761fe3017bed0fe895569ec6af3e4b889dcde6e9.jpg)

![](images/588a8d26416a9540d6150709be2c1a33cfa7b7012ba64b56a212be8ae6eaf8e2.jpg)  
Figure 5: Samples from the proposed UHD-LL dataset.

![](images/c420b1eff755d5df857227dd79e72dbcb291c5dc528207194c7ede2716781476.jpg)

![](images/77446d22532d2866f9c7a2e24d5261116a980ea360a374451d3c87319a5563de.jpg)

using a tripod, we also use remote control software (Imaging Edge) to adjust the exposure time and ISO value to avoid any physical contact with the camera. To further reduce subtle misalignments, we adopt an image alignment algorithm (Evangelidis & Psarakis, 2008) to estimate the affine matrix and align the low-light image and its ground truth. We improve the alignment method by applying AdaIN (Huang & Belongie, 2017) before the affine matrix estimation to reduce the intensity gap between the pair. Finally, we hire annotators to check all paired images carefully and discard those that still exhibit misalignments.

We split the UHD-LL dataset into two parts: 2,000 pairs for training and 115 pairs for testing. The training and test partitions are exclusive in their scene and data. We also ensure consistency in pixel intensity distribution between the training and test splits. More analysis of this data, e.g., the pixel intensity and Signal-to-Noise Ratio (SNR) distributions, can be found in the Appendix.

A comparison between our UHD-LL dataset and existing paired low-light image datasets with RGB format is presented in Table 1. The LOL dataset (two versions: LOL-v1: 500 images; LOL-v2: 789 images) is most related to our UHD-LL dataset as both focus on real low-light images with noise. The LOL-v2 contains all images of the LOL-v1. In contrast to the LOL dataset, our dataset features a more extensive collection, where diverse darkness and noise levels from rich types of scenes are considered. Moreover, the images of our dataset have higher resolutions than those from the LOL dataset. As shown in Figure 1, the models pre-trained on the LOL dataset cannot handle the cases in our UHD-LL dataset due to its insufficient training data, which are low-resolution and contains mostly mild noises.

# 4 EXPERIMENTS

Implementation. We implement our method with PyTorch and train it on six NVIDIA Tesla V100 GPUs. We use an ADAM optimizer for network optimization. The learning rate is set to 0.0001. A batch size of 6 is applied. We fix the channels of each Conv layer to 16, except for the Conv layers associated with outputs. We use the Conv layer with stride  $= 2$  and  $4 \times 4$  kernels to implement the  $2 \times$  downsample operation in the encoder and interpolation to implement the  $2 \times$  upsample operation in the decoder in the LRNet. Unless otherwise stated, the Conv layer uses stride  $= 1$  and  $3 \times 3$  kernels. We use the training data in the UHD-LL dataset to train our model. Images are randomly cropped into patches of size  $512 \times 512$  for training.

Compared Methods. We include 12 state-of-the-art methods (19 models in total) for our benchmarking study and performance comparison. These methods include 10 light enhancement methods: DRBN (CVPR'20) (Yang et al., 2020a), Zero-DCE (CVPR'20) (Guo et al., 2020), Zero-DCE++ (TPAMI'21) (Li et al., 2021b), RUAS (CVPR'21) (Liu et al., 2021b), Zhao et al. (ICCV'21) (Zhao et al., 2021), EnlightenGAN (TIP'21) (Jiang et al., 2021), Afifi et al. (CVPR'21) (Afifi et al., 2021), SCI (CVPR'22) (Ma et al., 2022), SNR-Aware (CVPR'22) (Xu et al., 2022), URetinex-Net (CVPR'22) (Wu et al., 2022) and 2 Transformers: Uformer (CVPR'22) (Wang et al., 2022) and Restormer (CVPR'22) (Zamir et al., 2022). We use their released models and also retrain them using the same training data as our method. Note that some methods provide different models trained using different datasets. Due to the heavy models used in Restormer (Zamir et al., 2022) and SNR-Aware (Xu et al., 2022), we cannot infer the full-resolution results of both methods on UHD images, despite using a GPU with 48G memory. Following previous UHD study (Zheng et al., 2021), we resort to two strategies for this situation: (1) We downsample the input to the largest size that the model can handle and then resize the result to the original resolution, denoted by the subscript 're

Table 2: Benchmarking study on the testing set of our UHD-LL. All models are released from the original papers and trained on the corresponding datasets. The best and second results are in red and blue, respectively.  

<table><tr><td>Methods</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>MUSIQ↑</td><td>NIQE↓</td><td>NIMA↑</td><td>Training Sets</td></tr><tr><td>input</td><td>9.926</td><td>0.482</td><td>0.551</td><td>26.779</td><td>5.379</td><td>2.269</td><td>-</td></tr><tr><td>DRBN (CVPR&#x27;20)</td><td>15.455</td><td>0.689</td><td>0.450</td><td>34.925</td><td>4.408</td><td>2.154</td><td>LOL-v2</td></tr><tr><td>Zero-DCE (CVPR&#x27;20)</td><td>17.081</td><td>0.664</td><td>0.509</td><td>35.488</td><td>5.006</td><td>2.139</td><td>SICE</td></tr><tr><td>Zero-DCE++ (TPAMI&#x27;21)</td><td>17.648</td><td>0.672</td><td>0.506</td><td>32.520</td><td>4.887</td><td>2.211</td><td>SICE</td></tr><tr><td>RUAS-LOL (CVPR&#x27;21)</td><td>11.761</td><td>0.701</td><td>0.514</td><td>28.396</td><td>5.909</td><td>2.565</td><td>LOL-v2</td></tr><tr><td>RUAS-MIT5K (CVPR&#x27;21)</td><td>14.250</td><td>0.586</td><td>0.553</td><td>29.900</td><td>5.407</td><td>2.270</td><td>MIT-Adobe FiveK</td></tr><tr><td>RUAS-DarkFace (CVPR&#x27;21)</td><td>11.325</td><td>0.583</td><td>0.596</td><td>28.256</td><td>6.160</td><td>2.561</td><td>DarkFace</td></tr><tr><td>Zhao et al.-MIT5K (ICCV&#x27;21)</td><td>15.177</td><td>0.547</td><td>0.530</td><td>32.127</td><td>4.495</td><td>2.208</td><td>MIT-Adobe FiveK</td></tr><tr><td>Zhao et al.-LOL (ICCV&#x27;21)</td><td>18.604</td><td>0.694</td><td>0.479</td><td>32.392</td><td>4.248</td><td>2.183</td><td>LOL-v1</td></tr><tr><td>EnlightenGAN (TIP&#x27;21)</td><td>17.637</td><td>0.767</td><td>0.459</td><td>27.441</td><td>5.497</td><td>1.977</td><td>Assembled</td></tr><tr><td>Afifi et al. (CVPR&#x27;21)</td><td>18.212</td><td>0.610</td><td>0.479</td><td>33.970</td><td>4.793</td><td>2.217</td><td>Exposure-Errors</td></tr><tr><td>SCI-easy (CVPR&#x27;22)</td><td>15.536</td><td>0.610</td><td>0.501</td><td>31.848</td><td>4.897</td><td>2.166</td><td>MIT-Adobe FiveK</td></tr><tr><td>SCI-medium (CVPR&#x27;22)</td><td>15.481</td><td>0.622</td><td>0.528</td><td>31.474</td><td>4.941</td><td>2.211</td><td>LOL+LSRW</td></tr><tr><td>SCI-difficult (CVPR&#x27;22)</td><td>17.872</td><td>0.578</td><td>0.544</td><td>36.219</td><td>5.218</td><td>2.106</td><td>DarkFace</td></tr><tr><td>SNR-Aware-LOLV1resize (CVPR&#x27;22)</td><td>15.737</td><td>0.802</td><td>0.448</td><td>20.385</td><td>9.591</td><td>2.275</td><td>LOL-v1</td></tr><tr><td>SNR-Aware-LOLV1stitch (CVPR&#x27;22)</td><td>15.536</td><td>0.695</td><td>0.468</td><td>33.098</td><td>3.961</td><td>2.387</td><td>LOL-v1</td></tr><tr><td>SNR-Aware-LOLV2realresize (CVPR&#x27;22)</td><td>15.954</td><td>0.742</td><td>0.471</td><td>23.494</td><td>9.257</td><td>2.001</td><td>LOL-v2</td></tr><tr><td>SNR-Aware-LOLV2realstitch (CVPR&#x27;22)</td><td>14.616</td><td>0.634</td><td>0.488</td><td>33.477</td><td>4.143</td><td>2.577</td><td>LOL-v2</td></tr><tr><td>SNR-Aware-LOLV2syntheticresize (CVPR&#x27;22)</td><td>16.031</td><td>0.748</td><td>0.494</td><td>20.065</td><td>9.963</td><td>2.248</td><td>LOL-syn</td></tr><tr><td>SNR-Aware-LOLV2syntheticstitch (CVPR&#x27;22)</td><td>15.887</td><td>0.675</td><td>0.497</td><td>31.473</td><td>4.460</td><td>2.484</td><td>LOL-syn</td></tr><tr><td>URetinex-Net (CVPR&#x27;22)</td><td>20.689</td><td>0.706</td><td>0.457</td><td>35.434</td><td>4.974</td><td>2.181</td><td>LOL-v1</td></tr></table>

size'. (2) We split the input into four patches without overlapping and then stitch the result, denoted by the subscript 'stitch'.

Evaluation Metrics. We employ full-reference image quality assessment metrics PSNR, SSIM (Wang et al., 2004), and LPIPS (Alex version) (Zhang et al., 2018) to quantify the performance of different methods. We also adopt the non-reference image quality evaluator (NIQE) (Mittal et al., 2013) and the multi-scale image quality Transformer (MUSIQ) (trained on KonIQ-10k dataset) (Ke et al., 2021) for assessing the restoration quality. We notice that the quantitative results reported by different papers diverge. For a fair comparison, we adopt the commonly-used IQA PyTorch Toolbox<sup>1</sup> to compute the quantitative results of all compared methods. We also test the trainable parameters and running time for processing UHD 4K data.

# 4.1 BENCHMARKING EXISTING MODELS

To validate the performance of existing LLIE methods that were trained using their original training data, we directly use the released models for evaluation on the UHD low-light images. These original training datasets include LOL (Wei et al., 2018), MIT-Adobe-FiveK (Bychkovsky et al., 2011), Exposure-Errors (Afifi et al., 2021), SICE (Cai et al., 2018), LSRW (Hai et al., 2024), and DarkFace (Yang et al., 2020b). EnlightenGAN uses the assemble training data from existing datasets (Wei et al., 2018; Dang-Nguyen et al., 2015; Kalantari & Ramamoorthi, 2017; Cai et al., 2018). In addition, the LOL-v1 and LOL-v2 contain real low-light images while LOL-syn is a synthetic dataset. Due to the limited space, we only show relatively good results. As shown in Figure 6, all methods can improve the luminance of the input image. However, they fail in producing visually pleasing results. DRBN and EnlightenGAN introduce artifacts. RUAS-LOL and RUAS-DarkFace yield over-exposed results. Color deviation is observed in the results of EnlightenGAN and Afifi et al. All methods cannot handle the noise well and even amplify noise.

We also summarize the quantitative performance of different methods and verify the effectiveness of commonly used non-reference metrics for UHD low-light images in Table 2. URetinex-Net achieves the highest PSNR score while SNR-Aware-LOLv1 is the best performer in terms of SSIM and LPIPS. For non-reference metrics, SCI-difficult, Zhao et al.-LOL, and RUAS-LOL are the winners under MUSIQ, NIQE, and NIMA, respectively. From Figure 6 and Table 2, we found the non-reference metrics designed for generic image quality assessment cannot accurately assess the subjective quality of the enhanced UHD low-light images. For example, RUAS-LOL suffers from obvious over-exposure in the result while it is the best performer under the NIMA metric.

In summary, the performance of existing released models is unsatisfactory when they are used to enhance the UHD low-light images. The darkness, noise, and artifacts still exist in the results.

![](images/6e4e220bab422b038c13a7014a0a43c6c37a840d4f4b7a1cb23ab123472f047e.jpg)

![](images/c32cd23cf9ef3c244144677757bb1fb20ab054e97819eb2f3fb4e96cec242334.jpg)  
input

![](images/b6dff466e4e61f3fbeec99592f57d81b667bcb02fa25017eba484dfd0f515a81.jpg)

![](images/44312d47dcd88bb4e29d99af67e86c36948f9951a33183eb4c27f23a573952bb.jpg)

![](images/67fae74b3c48d9aa22f8729ff506d982d1ae57e86f95a2f050d23ee7e566138c.jpg)  
DRBN

![](images/d03a826de21241e9488532362134926b07a9c5d4547f46b19793809b4922fdc2.jpg)

![](images/c574faa51ee97a6aee0e4da2e86bbfff7794f45451a9bfc7404bef59d1667dfc.jpg)  
Zero-DCE

![](images/1a82b8c5575883a0c077f0b7368703c6d7fd991dc62d0bce76d9f26adada584b.jpg)

![](images/96af818e364b37ff0954f96bf111c6ebc29f8c4be02e69a58d740cebef3fe8f9.jpg)  
RUAS-LOL

![](images/7f63155540b8702111892966c289ee3a7e1aea060d169513c7c8bc4dbb6c772e.jpg)

![](images/2e71e2d76e0a364cf4a3abb4aa1bf0ac1fd6c26f3b4f9ad1908d925539d46b56.jpg)  
RUAS-DarkFace

![](images/b269e4ce17d715f867ca5bf309da49fe6f2285086335851154a2f4c44ed0fd25.jpg)

![](images/f2dd6156be911feae39ba2640994cc2d9aa4cde0e888dc55632a73a8b6aaaf96.jpg)  
Zhao et al.-LOL

![](images/4fa2e7a7ea36a4cb31692d30ad3b2f6097f19bd551e49b75f5acdcaacd0c4611.jpg)

![](images/f2ef7d100cff9ddc2bdb9d60dbeaff76235e1c493c17fd328d2ea08a68d551db.jpg)  
EnlightenGAN

![](images/214c3637710c0a6d533eb13650da11a5db0eb8a86a371266ddb25b16d067b7e2.jpg)

![](images/46a7022f50d1a40591ecda1dfa907611916a699aa78cea880ce99fb535479e7b.jpg)  
Afifi et al.

![](images/315c5dee30d59910a473280b9da11243a31cbf22ca0fe4e477709dee49407eb3.jpg)

![](images/dec71ffc99a9be7d2916864485d541512773b3cc1bb8bdf6d8a356fc02af717c.jpg)  
SCI-difficult

![](images/69bd567c916118ffdf387fe78d1e3b9f9418dc7730464a8a1a6afd6fe962c03c.jpg)  
Figure 6: Visual comparison between state of the arts for restoring a UHD low-light image. We use the released model directly in this evaluation. All released models cannot handle the UHD low-light image well. More results can be found in the Appendix.

![](images/73e0619270611b171e200d5dde2c28fa412442da28ef89dc3c9cbfc0817735a9.jpg)

![](images/ed05e131dd7ceea47b6e45447d15a46a04c0657788c0ff36efd4376c4c95a6f7.jpg)  
SRN-Aware-LOLv1

![](images/508bf89e727deffc14bbd294e82cca0341dbc5acef4a7a3d0571c5253e4c1dc9.jpg)

![](images/165cdffd791159667976a94857245cb73f22c39d0b5263e5b54048c8d0841eff.jpg)  
URetinex-Net

![](images/4d49796ace2c01d66b0be8cfad8095c3e5104be101fbe82537fd8493f3f38693.jpg)

![](images/1d022e9451926e1c5e3c0ea0e136f8bbb90b1c2631ef7c262a5d89d6bb5d9030.jpg)  
GT

Compared with luminance enhancement, noise is the more significant challenge for these methods. No method can handle the noise issue well. The joint task of luminance enhancement and noise removal raises a new challenge for LLIE, especially under limited computational resources. We also observe a gap between visual results and the scores of non-reference metrics for UHD LLIE. The gap calls for more specialized non-reference metrics for UHD LLIE.

# 4.2 COMPARING RETRAINED MODELS

Besides the released models, we also retrain existing methods on our UHD-LL training data and compare their performance with our method. Due to the limited space, we only compare our method with several good performers. More results can be found in the Appendix. As shown in Figure 7, our UHDFour produces a clear and normal-light result close to the ground truth. In comparison, Zero-DCE++, RUAS, Afifi et al., SCI, and Restormer experience color deviations. Zero-DCE, Zero-DCE++, RUAS, Zhao et al., Afifi et al., and SCI cannot remove the noise due to the limitations of their network designs. These methods mainly focus on luminance enhancement. SNR-Aware, Uformer, and Restormer have strong modeling capability because of the use of Transformer structures. However, the three methods still leave noise on the results and introduce artifacts.

The quantitative comparison is presented in Table 3. Our UHDFour achieves state-of-the-art performance in terms of PSNR, SSIM, and LPIPS scores and outperforms the compared methods with a large margin. The Transformer-based methods such as SNR-Aware and Restormer rank the second best. Our method has the fastest processing speed for UHD images as most computation is conducted in the LR space.

![](images/c61b622846383fd94772d97d264e584c653480281531ae6350406f5b59685131.jpg)

![](images/a0f9404d3ce873e5645b846fb1e171853e156d81b7df87a351cc16a29326d8c8.jpg)  
input

![](images/5e5986a37204eeb9688ed7cd06a22bc68d4cc804e769117e86f525ebced13cfe.jpg)

![](images/7f182ed9f0fc7c1f318df5885dbc03128499026c66a74c690cc12f69a8951398.jpg)

![](images/2e7f42bf3ef52f9d6a3c964fcd7d4da6d5b6201682f5113cc37b4391e4574775.jpg)  
Zero-DCE

![](images/3a268e6cfab775139195800c08038e21c2e44f6cf54e19b4ec674c9dcf9c6d70.jpg)

![](images/9c5e6800e3a4a037768e831ae6503df45beb6b10b6ce1abe53e2f8f581936ce3.jpg)

![](images/7d0fa5680fe47401e3d505363d6c8cdc869b4b0d287a98b6d7774e756c81b057.jpg)  
Zero-DCE++

![](images/15ecc10c480971c23fbe184b9a767fa16488bfe8dac8c912f82f2ceb11eae465.jpg)

![](images/12f52f46724c13e0c2468505c09df1c607037a1a6593950d77a26828534fc5c6.jpg)

![](images/d0311d16d6dbf28d9d7861901aa32aa42a0623c19790114da500b7a2103d2918.jpg)  
RUAS

![](images/2bb18fb5e507f5a63397aabdf5fe799288500b316de079eeced042572bdaceaa.jpg)

![](images/4002238b1cf478d24ae307de520d8632a57344e91956268444d477aa3a2165c9.jpg)  
Zhao et al.

![](images/a4959912e95fe2c28db9e002ae3b7cd05ae3ea98d4ccb63d559071847619f59f.jpg)

![](images/7574849f7758b722e60631f8682ae7f07e54265b01502e02f66f85a6319764ad.jpg)

![](images/b1bd6a198579bdbc0654c5fea296422609ea1659aa2b2f6b0964baee15353ed6.jpg)  
Afifi et al.

![](images/59e857c5f7e4b03f93136ef267847b7e400826f96782b1ad23fa6299bee85f77.jpg)

![](images/0edeea19db7e4a99ab9973ed8e1ef266ef152f5788e96a5169762ef039546062.jpg)

![](images/4af9a6e100bcd96fa5b00595ebf5d78463cbcadda2584ecd035d806591fc465a.jpg)  
SCI

![](images/fe7557d55fd67660135ff098607592591c38c8ca14d899eb5e354d69eb68fe0c.jpg)

![](images/8999dfff926c3ba841b92478e1b3fea837253eef3661074030403d1ffea9a451.jpg)

![](images/bd75e0dd5b66160aebeb6c82db50fb066ca88a84b625abb9a36f1c8d446fbbf5.jpg)  
SRN-Aware

![](images/626980714372c55ba56b73baf6407376ca287344da7ca8bd096292bd7b5f360f.jpg)

![](images/9644fd1f190d858a5f38c1ad59be130fe90d7a8ca570ecc0dc79acc7f56e09da.jpg)

![](images/1e058579cbb99b8b0e77dc525c1ec057932848742f17ae11a437164dc7acac12.jpg)  
Uformer

![](images/852a8b13791cb31fe1578e5c1d725b4109e010fa9ac8bf5acfd4d6d225d79700.jpg)  
Figure 7: Visual comparison between the retrained state of the arts on the UHD-LL dataset. All compared models leave noise, artifacts, or color deviations in the results. Our method achieves a visually pleasing result.

![](images/d6f0bf4b5909a00f7be577ab9efa6e9b6d77ba4d63b8353560862dc4be9d25ff.jpg)

![](images/36ab637c21ff8ac8c854fa73adf1dae69aa7c32c0bf2bf30354c0c7df67e259f.jpg)  
Restormer

![](images/3ca131593369d380962f9940edb5a176ad9d53954e74997506e4cac5c635a13b.jpg)

![](images/8cdaa766178a12507530a481d11813a292f5d69392560771572ba9454cc9434f.jpg)

![](images/4a4c0494819bc834fe02b66795f9b84de3160671891338a2c996420c8324ebb9.jpg)  
UHDFour (Ours)

![](images/3740808364334b12fc309b2caeac4d6f1cc7a88d28813b2f67097e0b0a88990a.jpg)

![](images/4f5bab41b5d5785a8b328fadc6098d8c46b6026b629438d255c0f44a6021140d.jpg)

![](images/efebb8d1f300b40ffa2f1431ab2ffbdc9a9cdc0040c28ed4117fda13a4163755.jpg)  
GT

![](images/3dc7aedc53d0e33e7b586c1f2cb936ec8760c2dc366f238b98422ee46ff84fb2.jpg)

Table 3: Quantitative comparison of the retrained state of the arts on the UHD-LL dataset. The best result is in red whereas the second one is in blue. RT: Running time. The training code of URetinex-Net is not released.  

<table><tr><td>Methods</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>Parameter↓</td><td>RT↓</td></tr><tr><td>Zero-DCE (CVPR&#x27;20)</td><td>17.075</td><td>0.663</td><td>0.513</td><td>79.416K</td><td>0.353s</td></tr><tr><td>Zero-DCE++ (TPAMI&#x27;21)</td><td>16.410</td><td>0.630</td><td>0.530</td><td>10.561K</td><td>0.327s</td></tr><tr><td>RUAS (CVPR&#x27;21)</td><td>13.562</td><td>0.749</td><td>0.460</td><td>3.438K</td><td>0.379s</td></tr><tr><td>Zhao et al. (ICCV&#x27;21)</td><td>21.964</td><td>0.870</td><td>0.324</td><td>11.560M</td><td>6.900s</td></tr><tr><td>Afifi et al. (CVPR&#x27;21)</td><td>20.805</td><td>0.740</td><td>0.440</td><td>70.154M</td><td>1.631s</td></tr><tr><td>SCI (CVPR&#x27;22)</td><td>16.057</td><td>0.625</td><td>0.533</td><td>0.258K</td><td>0.308s</td></tr><tr><td>SNR-AwareResize (CVPR&#x27;22)</td><td>22.717</td><td>0.877</td><td>0.304</td><td>40.084M</td><td>0.026s</td></tr><tr><td>SNR-Aware_stitch (CVPR&#x27;22)</td><td>22.170</td><td>0.866</td><td>0.307</td><td>40.084M</td><td>0.035s</td></tr><tr><td>Uformer (CVPR&#x27;22)</td><td>19.283</td><td>0.849</td><td>0.356</td><td>20.628M</td><td>0.235s</td></tr><tr><td>RestormerResize (CVPR&#x27;22)</td><td>22.597</td><td>0.878</td><td>0.280</td><td>26.112M</td><td>0.368s</td></tr><tr><td>Restormer_stitch (CVPR&#x27;22)</td><td>22.252</td><td>0.871</td><td>0.289</td><td>26.112M</td><td>0.368s</td></tr><tr><td>UHDFour (Ours)</td><td>26.226</td><td>0.900</td><td>0.239</td><td>17.537M</td><td>0.024s</td></tr></table>

To further verify the effectiveness of our network, we compare our approach with several methods, including Retinex-Net Wei et al. (2018), Zero-DCE (Guo et al., 2020), AGLLNet (Lv et al., 2021), Zhao et al. (Zhao et al., 2021), RUAS (Liu et al., 2021b), SCI (Ma et al., 2022), and URetinex-Net (Wu et al., 2022), that were pre-trained or fine-tuned on the LOL dataset (Wei et al., 2018). Due to the mild noise and low-resolution images in the LOL dataset, we change the  $8 \times$  downsample and upsample operations to  $2 \times$  and retrain our network following the settings of recent work (Wu et al., 2022). And such characteristics of LOL dataset prohibit us from showing the full potential of our method in removing noise and processing high-resolution images. Even though our goal is not to pursue state-of-the-art performance on the LOL dataset, our method achieves satisfactory performance as presented in Table 5. The visual results are provided in Figure 8.

Table 4: Quantitative comparison on the LOL dataset. The best result is in red whereas the second one is in blue.  

<table><tr><td>Methods</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td>input</td><td>7.77</td><td>0.19</td></tr><tr><td>Retinex-Net (BMVC&#x27;18)</td><td>16.77</td><td>0.54</td></tr><tr><td>Zero-DCE (CVPR&#x27;20)</td><td>16.79</td><td>0.67</td></tr><tr><td>AGLLNet (IJCV&#x27;21)</td><td>17.52</td><td>0.77</td></tr><tr><td>Zhao et al. (ICCV&#x27;21)</td><td>21.67</td><td>0.87</td></tr><tr><td>RUAS (CVPR&#x27;21)</td><td>16.44</td><td>0.70</td></tr><tr><td>SCI (CVPR&#x27;22)</td><td>14.78</td><td>0.62</td></tr><tr><td>URetinex-Net (CVPR&#x27;22)</td><td>19.84</td><td>0.87</td></tr><tr><td>UHDFour (Ours)</td><td>23.09</td><td>0.87</td></tr></table>

Table 5: Quantitative comparison of ablated models. FB: Fourier Branch; SB: Spatial Branch; AM: Amplitude Modulation; PG: Phase Guidance; and Concat: Concatenation.  

<table><tr><td rowspan="2">#</td><td colspan="2">FouSpa Block</td><td colspan="3">Adjustment Block</td><td>Output</td><td>Performance</td></tr><tr><td>FB</td><td>SB</td><td>AM</td><td>PG</td><td>SB</td><td>Concat</td><td>PSNR/SSIM</td></tr><tr><td>1</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>24.123/0.877</td></tr><tr><td>2</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>24.722/0.874</td></tr><tr><td>3</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>24.005/0.853</td></tr><tr><td>4</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>25.529/0.883</td></tr><tr><td>5</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>24.828/0.874</td></tr><tr><td>6</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td><td>24.513/0.872</td></tr><tr><td>7</td><td>✓</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>24.106/0.863</td></tr><tr><td>8</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>25.616/0.887</td></tr><tr><td>9</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>26.226/0.900</td></tr></table>

![](images/10495a24bc07f704fc60c7d039be10065b6dd5de3848d2351238bafba7763b79.jpg)  
input

![](images/105558f0fdf861cbde514d5c56ecb3d300cb76ff8e8d8945ad4dca29431bb8e7.jpg)  
Retinex-Net

![](images/8d428af863f17a91a82427d06e35a0a20cf483b267ae6252ff199b8cff970ab8.jpg)  
Zero-DCE

![](images/5ae54d1c72c7c523f45d7639b100c99e30d34a3e8f26682b87bbb0568c8dbd07.jpg)  
AGLLNet

![](images/6357913e16b25c5ef35cee594fba4eaab4f21c747f30826f8f99583585d257c5.jpg)  
Zhao et al.

![](images/953a02e418fb0566c528e400c3d28de91b36a4834e37ec80333be0f166d708ed.jpg)  
RUAS

![](images/447ba5eeea81540183f1ded784a45facf80325ae0f7b0ea9a97f5f52819df34b.jpg)  
SCI

![](images/88da8e0ce3c5ed2775acfa90023285563f78f519a7f2d5216ae8bc809886b596.jpg)  
Figure 8: Visual comparison on the LOL dataset. More results can be found in the Appendix.  
URetinex-Net

![](images/df0795afbad04a7d32d0d1415a5bb828d6cd9b27038ddc499b666bd4b292b6f7.jpg)  
UHDFour (Ours)

![](images/eb7f318163729dbfa2ceaf8ba25024b1079caba3dc8eeadd171da64a896b4abe.jpg)  
GT

# 4.3 ABLATION STUDY

We present ablation studies to demonstrate the effectiveness of the main components in our design. For the FouSpa Block, we remove the Fourier branch (#1), remove the Spatial branch (#2), and replace the FouSpa Block (i.e., without Fourier and Spatial branches) with the Residual Block of comparable parameters (#3). For the Adjustment Block, we remove the Amplitude Modulation (#4), remove the Phase Guidance (#5), remove the Spatial branch (#6), and replace the Adjustment Block (i.e., without Fourier and Spatial branches) with the Residual Block of comparable parameters (#7). For the final output, we remove the concatenation of the LR normal-clear result  $(\hat{y}_8)$ , indicated as #8. Unless otherwise stated, all training settings remain unchanged as the implementation of full model, denoted as #9.

The quantitative comparison of the ablated models on the UHD-LL testing set is presented in Table 5. We also show the visual comparison in the Appendix. As shown, all the key designs contribute to the best performance of the full model. Without the Fourier branch (#1), the quantitative scores significantly drop. The result suggests that processing amplitude and phase separately improves the performance of luminance enhancement and noise removal. From the results of #2, the Spatial branch also boosts the performance. However, replacing the FouSpa Block with the Residual Block (#3) cannot achieve comparable performance with the full model (#9), indicating the effectiveness of the FouSpa Block. For the Adjustment Block, the Amplitude Modulation (#4), Phase Guidance (#5), and Spatial branch (#6) jointly verify its effectiveness. Such a block cannot be replaced by a Residual Block (#7). From the results of #8, we can see that it is necessary to estimate the LR result.

# 5 CONCLUSION

The success of our method is inspired by the characteristics of real low-light and noisy images in the Fourier domain. Thanks to the unique design of our network that handles luminance and noises in the Fourier domain, it outperforms state-of-the-art methods in UHD LLIE with appealing efficiency. With the contribution of the first real UHD LLIE dataset, it becomes possible to compare existing methods with real UHD low-light images. Our experiments are limited to image enhancement; we have not provided data and benchmarks in the video domain. Our exploration has not considered adversarial losses due to memory constraints. Nevertheless, we believe our method and the dataset can bring new opportunities and challenges to the community. The usefulness of Fourier operations may go beyond our work and see potential in areas like image decomposition and disentanglement. With improved efficiency, it may be adopted for applications that demand real-time response, e.g., enhancing the perception of autonomous vehicles in the dark.

# REFERENCES

Mahmoud Affi, Konstantinos G. Derpanis, Bjorn Ommer, and Michael S. Brown. Learning multiscale photo exposure correction. In CVPR, 2021.  
Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Fredo Durand. Learning photographic global tonal adjustment with a dataset of input/output image pairs. In CVPR, 2011.  
Jianrui Cai, Shuhang Gu, and Lei Zhang. Learning a deep single image contrast enhancer from multi-exposure image. IEEE Transactions on Image Processing, 27(4):2049-2062, 2018.  
Chen Chen, Qifeng Chen, Jia Xu, and Koltun Vladlen. Learning to see in the dark. In CVPR, 2018.  
Liangyu Chen, Xin Lu, Jie Zhang, Xiaojie Chu, and Chengpeng Chen. Hinet: Half instance normalization network for image restoration. In CVPR, 2021.  
Duc-Tien Dang-Nguyen, Cecilia Pasquini, Valentina Conotter, and Giulia Boato. Raise: A raw images dataset for digital image forensics. In ACMMSC, 2015.  
Georgios Evangelidis and Emmanouil Psarakis. Parametric image alignment using enhanced correlation coefficient maximization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(10):1858-1865, 2008.  
Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, and Runmin Cong. Zero-reference deep curve estimation for low-light image enhancement. In CVPR, 2020.  
Xin Guo, Xueyang Fu, Man Zhou, Zhen Huang, Jialun Peng, and Zhengjun Zha. Exploring fourier prior for single image rain removal. In *IJCAI*, 2022.  
Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. R2rnet: Low-light image enhancement via real-low to real-normal network. arXiv preprint arXiv:2106.14501, 2024.  
Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV, 2017.  
Yifan Jiang, Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, and Zhangyang Wang. EnlightenGAN: Deep light enhancement without paired supervision. IEEE Transactions on Image Processing, 30:2340-2349, 2021.  
Nima Khademi Kalantari and Ravi Ramamoorthi. Deep high dynamic range imaging of dynamic scenes. ACM Transactions on Graph, 36:144, 2017.  
Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In ICCV, 2021.  
Hieu Le and Dimitris Samaras. Shadow removal via shadow image decomposition. In ICCV, 2019.  
Chongyi Li, Chunle Guo, Linhao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, and Chen Change Loy. Low-light image and video enhancement using deep learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021a.  
Chongyi Li, Chunle Guo, and Chen Change Loy. Learning to enhance low-light image via zero-reference deep curve estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021b.  
Jiaying Liu, Dejia Xu, Wenhan Yang, Minhao Fan, and Haofeng Haung. Benchmarking low-light image enhancement and beyond. International Journal of Computer Vision, 2021a.  
Risheng Liu, Long Ma, Jiaao Zhang, Xin Fan, and Zhongxuan Luo. Retinex-inspired unrolling with cooperative prior architecture search for low-light image enhancement. In CVPR, 2021b.  
Feifan Lv, Yu Li, and Feng Lu. Attention guided low-light image enhancement with a large scale low-light simulation dataset. International Journal of Computer Vision, 2021.

Long Ma, Tengyu Ma, Risheng Liu, Xin Fan, and Zhongxuan Luo. Toward fast, flexible, and robust low-light image enhancement. In CVPR, 2022.  
Anish Mittal, Rajiv Soundararajan, and Alan C. Bovik. Making a "completely blind" image quality analyzer. IEEE Signal Processing Letters, 20(3):209-212, 2013.  
Ruixing Wang, Qing Zhang, Chi-Wing Fu, Xiaoyong Shen, Wei-Shi Zheng, and Jiaya Jia. Underexposed photo enhancement using deep illumination estimation. In CVPR, 2019.  
Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Recovering realistic texture in image super-resolution by deep spatial feature transform. In CVPR, 2018.  
Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, and Houqiang Li. Uformer: A general u-shaped transformer for image restoration. In CVPR, 2022.  
Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600-612, 2004.  
Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. Deep retina decomposition for low-light enhancement. In BMVC, 2018.  
Wenhui Wu, Jian Weng, Pingping Zhang, Xu Wang, Wenhan Yang, and Jianmin Jiang. Uretinex-net: Retinex-based deep unfolding network for low-light image enhancement. In CVPR, 2022.  
Ke Xu, Xin Yang, Baocai Yin, and Rynson W. H. Lau. Learning to restore low-light images via decomposition-and-enhancement. In CVPR, 2020.  
Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, and jiaya Jia. SNR-aware low-light image enhancement. In CVPR, 2022.  
Wenhan Yang, Shiqi Wang, Yuming Fang, Yue Wang, and Jiaying Liu. From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement. In CVPR, 2020a.  
Wenhan Yang, Ye Yuan, Wenqi Ren, and et al. Advancing image understanding in poor visibility environments: A collective benchmark study. IEEE Transactions on Image Processing, 29:5737-5752, 2020b.  
Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022.  
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.  
Lin Zhao, Shaoping Lu, Tao Chen, Zhenglu Yang, and Ariel Shamir. Deep symmetric network for underexposed image enhancement with recurrent attentional learning. In ICCV, 2021.  
Zhuoran Zheng, Wenqi Ren, Xiaochun Cao, Tao Wang, and Xiuyi Jia. Ultra-high-definition image hdr reconstruction via collaborative bilateral learning. In ICCV, 2021.
