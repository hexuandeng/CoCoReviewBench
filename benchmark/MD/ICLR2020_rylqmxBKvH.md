# UNSUPERVISED SPATIOTEMPORAL DATA INPAINTING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We tackle the problem of inpainting occluded area in spatiotemporal sequences, such as cloud occluded satellite observations, in an unsupervised manner. We place ourselves in the setting where there is neither access to paired nor unpaired training data. We consider several cases in which the underlying information of the observed sequence in certain areas is lost through an observation operator. In this case, the only available information is provided by the observation of the sequence, the nature of the measurement process and its associated statistics. We propose an unsupervised-learning framework to retrieve the most probable sequence using a generative adversarial network. We demonstrate the capacity of our model to exhibit strong reconstruction capacity on several video datasets such as satellite sequences or natural videos.

# 1 INTRODUCTION

We consider the problem of reconstructing missing information from image sequences. The problem occurs in many different settings and for different types of sequences. For example, in remote sensing applications, satellite imagery are frequently occluded by meteorological perturbations such as clouds and rains (Singh & Komodakis, 2018). Recovering missing satellite data is an active research topic. Approaches range from simple interpolation to sophisticated data assimilation methods. The latter is often a model-based approach that relies on analytical models of the underlying observed phenomenon (Ubelmann et al., 2015; Sirjacobs et al., 2011; Lguensat et al., 2017). Model-free data based methods have also been developed such as DINEOF (Alvera-Azcarate, 2011). Note that for physical observation modeling problems of this type, there is never any direct supervision available. Another example concerns natural videos. Here, information can be occluded by moving objects such as fences (Yamashita et al., 2010), raindrops (Qian et al., 2018), persons (Kim et al., 2019), stains on photographic films (Tang et al., 2011). Video and image imputation have given rise to a large body of literature. Recent Deep Learning (DL) advances have motivated the development of general imputation methods relying on generative models such as GANs (Wang et al., 2018a; Xu et al., 2019; Kim et al., 2019). They all make use of supervision and require the availability of a ground truth, which is absent in many real-world problems. Data driven supervised methods have thus attained impressive results and are able to accurately complete a large missing region. However, reconstructing the missing information in videos when supervision is unavailable is still an open problem and there have been only a few works exploring this direction. For example, Newson et al. (2014) propose a simple but effective method for occlusions in natural videos that replaces occluded parts with information from their neighborhood.

We consider here unsupervised video reconstruction. We propose a model which can be used on different types of image sequences, physical or natural videos, and for a large variety of occlusion processes. Our method does not make any assumption on the nature of the image sequence, it does not require any prior knowledge like most methods used for physical images do. It is especially well suited when the occlusion is complex thus forbidding the use of ad hoc techniques, e.g., the patch method of Newson et al. (2014). The method extends to sequences from ideas recently developed for still images based on generative networks (Bora et al., 2018; Pajot et al., 2019; Li et al., 2019). This is up to our knowledge the first attempt to solve the problem of unsupervised video completion using general ML methods. This method is fully data driven and does not use any hand-defined analytical prior on the signal. Priors on the unobserved signal are directly learned from the data for solving an underlying inverse problem. The method is then applicable to a large variety of video signals.

Our main contributions are the following:

- We propose a new framework and model for large-scale image sequence inpainting learning, in a fully unsupervised context.  
- This model can be used for a variety of image sequences and for different occlusion processes.  
- Extensive evaluations are performed on realistic simulated satellite data and on natural videos with different occlusion processes.

# 2 METHOD

# 2.1 PROBLEM SETTING

We suppose that there exists an unknown spatiotemporal sequence  $\pmb{x} \sim \mathfrak{p}_{\mathbf{X}}, \pmb{x} \in \mathbb{R}^{C \times T \times H \times W}$ , where  $\pmb{x}$  is a tensor denoting a  $C$ -channel sequence composed of  $T$  frames of  $H \times W$  pixels. We denote  $\pmb{x}_t$  the  $t$ -th frame of the sequence and  $\pmb{x}_{t_1}^{t_2}$  the subsequence from the  $t_1$ -th to the  $t_2$ -th frame inclusive. With this notation,  $\pmb{x} = \pmb{x}_1^T$ . We do not have access to the original signal  $\pmb{x}$  but only to corrupted observation sequences of this signal  $\pmb{y} \sim \mathrm{p}_{\mathbf{Y}}, \pmb{y} \in \mathbb{R}^{C \times T \times H \times W}$ . Our objective is to reconstruct  $\pmb{x}$  from the corresponding observation  $\pmb{y}$ . For example,  $\pmb{x}$  can be sea surface temperature (SST) at successive times while image sequence  $\pmb{y}$  is SST measurements via IR satellites occluded by moving clouds. We will suppose that  $\pmb{y}$  is obtained from  $\pmb{x}$  via a measurement process modeled through a stochastic operator  $F$  as follows:

$$
\boldsymbol {y} = F (\boldsymbol {x}, \boldsymbol {m}) = \boldsymbol {x} \odot \boldsymbol {m} + c \cdot \bar {\boldsymbol {m}} \tag {1}
$$

where  $m \sim p_{\mathrm{M}}$  is an occlusion mask, generated from a known distribution with the same size as  $x$  and with components in  $\{0,1\}$ , where 0 holds for a masked pixel.  $\bar{m}$  denotes the complement of  $m$ ,  $\odot$  is the element-wise multiplication, all the masked pixels are supposed to be reset to a constant  $c$  which could be 0 or 1 depending on the observation process (see Section 3). Random variables  $\mathbf{X}$  and  $\mathbf{M}$  are assumed to be independent and  $F$  is assumed differentiable w.r.t.  $x$ . In the following, we will suppose that one can retrieve the mask  $m$  directly from the observation  $y$ . This is not very restrictive since in most situations this is easy to do. We denote  $T$  the mask extractor  $T(y) = m$ .

Our objective is then to recover the sequence  $\mathbf{x}$  from the observations  $\mathbf{y}$  and the corresponding binary masks  $\mathbf{m}$ . Adopting a probabilistic viewpoint, we want to select a reconstruction  $\mathbf{x}^*$  which is the most plausible under the posterior distribution  $p_{\mathbf{X}|\mathbf{Y}}(\cdot |\mathbf{y})$ .

# 2.2 MODEL

We formulate the problem as finding the most probable sequence conditioned on observations:

$$
\boldsymbol {x} ^ {*} = \underset {\boldsymbol {x}} {\arg \max } \log \mathrm {p} _ {\mathbf {X} | \mathbf {Y}} (\boldsymbol {x} | \boldsymbol {y}) = \underset {\boldsymbol {x}} {\arg \max } \log \mathrm {p} _ {\mathbf {X}} (\boldsymbol {x}) + \log \mathrm {p} _ {\mathbf {Y} | \mathbf {X}} (\boldsymbol {y} | \boldsymbol {x}) \tag {2}
$$

The prior term  $\log \mathrm{p}_{\mathbf{X}}(\boldsymbol{x})$  is unknown since we are in an unsupervised setting, while the likelihood  $\log \mathrm{p}_{\mathbf{Y}|\mathbf{X}}(\boldsymbol{y}|\boldsymbol{x})$  does not lead to analytical or simple computational solution.

To tackle these issues, let us introduce a mapping  $G: \mathbf{Y} \mapsto \mathbf{X}$ , parameterized by a neural network  $\varphi$  and associating measurement  $\pmb{y}$  to its estimate  $\pmb{x}$ .  $G$  will allow us to approximate the underlying distribution of training sequences. By plugging  $G(\pmb{y})$  into Equation 2, the objective becomes:

$$
G ^ {*} = \arg \max  _ {G} \underbrace {\mathbb {E} _ {\boldsymbol {y} \sim \mathrm {p} _ {\boldsymbol {Y}}} [ \log \mathrm {p} _ {\boldsymbol {X}} (G (\boldsymbol {y})) ]} _ {\text {p r i o r}} + \underbrace {\mathbb {E} _ {\boldsymbol {y} \sim \mathrm {p} _ {\boldsymbol {Y}}} [ \log \mathrm {p} _ {\boldsymbol {Y} \mid \boldsymbol {X}} (\boldsymbol {y} \mid G (\boldsymbol {y})) ]} _ {\text {l i k e l i h o o d}} \tag {3}
$$

# 2.3 PRIOR HANDLING

Let us first handle the prior term in Equation 3. We want the distribution induced from  $G(\pmb{y})$  to be close to  $\mathfrak{p}_{\mathbf{X}}$ . In order to do so, we will use an adversarial approach. We will build on the ideas introduced in Bora et al. (2018); Pajot et al. (2019) for still images. The process is illustrated in Figure 1. For a given observation  $\pmb{y}$ , we want to generate an approximation of the unknown true sequence  $\hat{\pmb{x}} \equiv G(\pmb{y})$ . The prior  $\mathfrak{p}_{\mathbf{X}}$  being unknown, the only available information source is the observation  $\pmb{y}$  and the noise prior  $\mathfrak{p}_{\mathbf{M}}$ . For a given generated signal  $\hat{\pmb{x}}$ , we compute a corrupted version of  $\hat{\pmb{x}}$  through

![](images/e666a2582295abe58e02dcffcdacd78249dfedc3d0237f54e0b887ad5f4232d5.jpg)  
Figure 1: Schema of our model. Generator  $G$  takes a sequence  $\pmb{y}$  and outputs an inpainted sequence  $\hat{\pmb{x}}$ ; measurement process  $F$  takes the inpainted sequence then outputs fake observations  $\hat{\pmb{y}}$ .

the known mask  $\hat{m}, \hat{y} \equiv F(\hat{x}, \hat{m})$  with  $\hat{m} \sim \mathrm{p}_{\mathbf{M}}$ . We will train  $G$  to make the distributions of  $\mathbf{y}$  and  $\hat{\mathbf{y}}$  indistinguishable. In order to succeed, the generator  $G$  will have to remove the corruption from  $\mathbf{y}$  and recover a sample  $\hat{x}$  from distribution  $\mathrm{p}_{\mathbf{X}}$ . Generator  $G$  will then act as an inpainter conditioned on  $\mathbf{y}$ . This will enforce the distribution of the reconstructed sequences  $\hat{x}$  to be close to the distribution of true ones  $\mathbf{x}$  and maximize the prior term.

A direct application of the adversarial training idea suggests using a discriminator operating directly on the sequences. We found out that using an additional discriminator on frames worked better than using a unique one operating on sequences. We then use two discriminators  $D_{s}$  and  $D_{f}$  respectively associated with whole sequences and with individual frames to optimize  $G$ .  $D_{s}$  separates sequences  $\mathbf{y}$  and  $\hat{\mathbf{y}}$ .  $D_{f}$  distinguishes real frames  $\mathbf{y}_{t}$  from fake ones  $\hat{\mathbf{y}}_{t}$ . The loss function used for training  $G$ ,  $D_{s}$ , and  $D_{f}$  is:

$$
\min  _ {G} \mathcal {L} (G) = \max  _ {D _ {s}, D _ {f}} \mathbb {E} _ {\boldsymbol {y} \sim \mathrm {p} _ {\boldsymbol {Y}}, \hat {\boldsymbol {y}} \sim \mathrm {p} _ {\boldsymbol {Y}} ^ {G}} [ \log D _ {s} (\boldsymbol {y}) + \log (1 - D _ {s} (\hat {\boldsymbol {y}})) + \frac {1}{T} \sum_ {t = 1} ^ {T} \log D _ {f} (\boldsymbol {y} _ {t}) + \log (1 - D _ {f} (\hat {\boldsymbol {y}} _ {t})) ] \tag {4}
$$

with  $\mathfrak{p}_{\mathbf{Y}}^{G}(\boldsymbol{y}) \equiv \mathbb{E}_{\boldsymbol{m} \sim \mathfrak{p}_{\mathbf{M}}, \boldsymbol{x} \sim \mathfrak{p}_{\mathbf{X}}^{G}}[\mathfrak{p}_{\mathbf{Y}|\mathbf{X}, \mathbf{M}}(\boldsymbol{y}|\boldsymbol{x}, \boldsymbol{m})]$ , corresponding to the distribution of the corrupted sequences  $\hat{\boldsymbol{y}}$  generated via the measurement operator  $F$ .  $\mathfrak{p}_{\mathbf{X}}^{G}(\boldsymbol{x})$  is the distribution of  $\hat{\boldsymbol{x}}$  induced by  $G$  from  $\boldsymbol{y}$ , i.e.  $\hat{\boldsymbol{x}} = G(\boldsymbol{y})$ .

# 2.4 LIKELIHOOD HANDLING

Let us now handle the likelihood term in Equation 3:

$$
\mathbb {E} _ {\boldsymbol {y} \sim \mathrm {p} _ {\boldsymbol {Y}}} [ \log \mathrm {p} _ {\boldsymbol {Y} | \boldsymbol {X}} (\boldsymbol {y} | G (\boldsymbol {y})) ]. \tag {5}
$$

This likelihood is maximised when we are able to perfectly reconstruct  $\pmb{y}$  from  $G(\pmb{y})$ . One way to ensure this property is to constrain  $G$  to directly use  $\pmb{y}$  for the non occluded area of the reconstructed image  $G(\pmb{y})$ . This can be easily achieved through the following mapping:

$$
G (\boldsymbol {y}) \equiv \varphi (\boldsymbol {y}) \odot \bar {\boldsymbol {m}} + \boldsymbol {y} \odot \boldsymbol {m} \tag {6}
$$

where  $\varphi$  is a NN responsible for reconstructing the missing part of  $\mathbf{y}$ ,  $m = T(\mathbf{y})$  is the mask retrieved from  $\mathbf{y}$ .  $G$  maps  $\mathbf{Y}$  to  $\mathbf{X}$  with the help of mask  $\mathbf{m}$  to ensure that the network will only generate values for occluded pixel, while keeping all the information from  $\mathbf{y}$ . To summarize, optimizing the prior term amounts at training  $\varphi$  for inputting the missing pixels while optimizing the likelihood term is simply achieved by copying the non occluded portion of  $\mathbf{y}$ .

# 2.5 TRAINING

$G$  is optimized by descending the prior loss and  $D_{s}, D_{f}$  by ascending it. Sequence discriminator  $D_{s}$  focuses on temporal dependence and coherence of pixel changes. Frame discriminator  $D_{f}$  keeps an eye on spatial feature coherence of observation frames.

# 3 EXPERIMENTS

We evaluate our model on four datasets, characteristic of different types of image sequences. The first one, SST, is a realistic simulation of satellite observations. The other three are natural video datasets: FaceForensics++, KTH, and BAIR, initially respectively used as benchmarks for forgery detection, motion detection, and video prediction.

# 3.1 DATASETS

SST The Sea Surface Temperature dataset used for the experiments includes 2 subsets of GLOBAL Sea Physical Analysis and Forecasting Product<sup>1</sup> from E.U. Copernicus Marine Service Information. This is a monitor system providing simulated but realistic global ocean SST data, which integrates satellite-derived and in situ data by assimilation. Our dataset is a part of the hourly mean SST, the finest timescale we have access to. The data we use is a part of the archive of analysis integrating real-world data. We retrieved our training-and-validation set and test set respectively from two different marine regions. Detailed data description and information for accessing the dataset are provided in Appendix A.

FaceForensics++ (Rössler et al., 2019) This dataset contains 1000 videos of non-occluded face movements on a static background. It was initially created for forgery detection. In our case, we extracted the faces from the original unforged videos with face_recognition $^2$ , thus keeping only the changing component of the videos. The faces have been cropped and resized to  $64 \times 64$ .

KTH (Schuldt et al., 2004) A human action dataset containing 2391 video clips of 6 human actions. The videos have been recorded with 25 subjects in different environments. All frames have been resized to  $64 \times 64$ .

BAIR Robot Pushing Dataset (Ebert et al., 2017) This dataset contains 44374 videos recorded by an one-armed robot. It pushes objects and changes movement direction in a stochastic manner. All videos share similar tabletop with static background. All frames have been resized to  $64 \times 64$ .

# 3.2 MEASUREMENT PROCESSES

The above datasets provide ground truth videos without corruption. In order to generate corrupted observation sequences, we simulate different types of occlusion depending on the nature of the videos. Each corruption process is defined as a stochastic operator  $F$  as in Equation 1 with mask distribution  $\mathbf{p}_{\mathrm{M}}$ . For a given video one then generates a sequence of random masks, one mask being then associated to each frame of the sequence. Note that except for the Remove-Pixel corruption process where two successive corruptions are independent, for all processes, the generated corruption sequences are time-dependent: the corruption pattern at time  $t$  will depend on the one at time  $t - 1$ .

Cloud This process is specific for the SST dataset. It simulates realistically video cloud masks on satellite images. Cloud masks are simulated using Liquid Water Path (LWP) data (measured in  $\mathrm{g} / \mathrm{m}^2$ ), which characterizes the total amount of liquid water present in the atmosphere between two points. The LWP data are generated by PyCLES (Pressel et al., 2015) $^3$ , a large eddy simulation system. It simulates the evolution of clouds in time based on a variant of anelastic equations of atmospheric motion. Collected LWP data record mask videos of clouds. The binary masks are then obtained by setting the image pixels to 0 when their LWP value is above a threshold. This produces realistic cloud coverage of the captured regions, see Figure 2a. Pixels occluded by the mask are set to  $c = 1$ . Thresholds are selected in the interval 55 to  $80~\mathrm{g} / \mathrm{m}^2$  to simulate clouds at different occlusion rates. Statistics about the occluded area at different thresholds are presented in Table 2a. For simulating occlusion, for each SST image sequence, we sample randomly a sequence of masks from the LWP dataset to be applied to the SST sequence.

Raindrops This process is a simplified model of random raindrops between subject and camera, taking into account a blurring effect when raindrops leave traces during exposure. It generates a set of white bars, each with a random length  $\theta_{l}$  and a constant width  $w$ . Bars move down at a random speed  $\theta_{v}$ , starting from a random initial position  $\theta_{p}$ . All these values are normalized w.r.t the frame edge length in  $[0, 1]$ . The number of raindrops is pre-defined. Bars return to the top once completely out of frame, see Figure 2. Pixels occluded by the mask are reset to  $c = 1$ . Note that as for Cloud, this is a time-dependent measurement process, meaning that two successive masks are correlated.

Remove-Pixel This measurement roughly mimics severe damages on vintage films. It masks randomly a fixed proportion  $p \in ]0,1[$  of pixels at each time step and reset them to  $c = 0$ , see Figure 2. Mask for each frame is generated independently regardless the evolution of time. This is the only time-independent measurement considered here.

Vertical-Moving-Bar This simple measurement operator generates a vertical white bar crossing the sequence, very roughly mimicking a fence or any similar obstacle. The bar is generated with the following distribution parameters: width  $\theta_w$ , initial position  $\theta_p$ , horizontal constant velocity  $\theta_v$ . These values are in  $]0, 1[$  as for Raindrops. The moving direction is chosen randomly. The bar reappears on the opposite side once it reaches the border. Masked pixels in observation are reset to  $c = 1$ . This is a time-dependent measurement.

# 3.3 BASELINES

Unsupervised Approaches We use two unsupervised baselines, one adapted for SST and the other one specific of natural videos. The former is DINEOF (Alvera-Azcarate, 2011). This is a state-of-the-art data-driven completion method in geophysics, and it has been used for SST observations, chlorophyll, salinity etc. It is a parameter-free interpolation technique based on empirical orthogonal function (EOF). It adopts an iterative algorithm that calculates at each iteration a truncated decomposition of EOF from known pixels, then replaces the values marked as missing by a reconstruction from calculated EOF. DINEOF does not make any assumption on the form of missing area and as such could be used for other domains as well and for different types of complex occlusion processes. However, DINEOF has been developed for remote sensing and does not ensure the coherence between different input channels (e.g. for RGB images).

The other one is Newson et al. (2014), one of the very few methods for unsupervised natural video inpainting. It is representative of patch-based approaches and it is still today state-of-the-art for many natural video occlusion processes. It searches for the nearest neighbours of occluded area using an Approximate Nearest Neighbour (ANN) search. The occluded area is reconstructed by assembling information from these neighbours at multiple scales. The form of the researched patches is supposed to be rectangular cuboids, e.g. a  $5 \times 5 \times 5$  spatiotemporal tensor, which limits its capability to adapt to more complex cases like Cloud, Raindrops, Remove-Pixel.

Supervised Approaches As already mentioned, there exists several supervised approaches to sequence inpainting (Huang et al., 2016; Xu et al., 2019; Kim et al., 2019). In order to evaluate the performance of our unsupervised method w.r.t. supervised ones, we compared with two supervised baselines. As our goal is not to beat state-of-the-art supervised techniques, we used two supervised adaptation of our model, respectively trained using unpaired and paired supervision. They are described below.

UNPAIRED VARIANT This is a supervised variant of our model in which we have access to unpaired samples from  $\mathfrak{p}_{\mathbf{X}}$  and  $\mathfrak{p}_{\mathbf{Y}}$ . The model is illustrated in Appendix C. Because we have access to clean  $\mathbf{x}$  data, it is then possible to supervise the approximation  $\hat{\boldsymbol{x}} = G(\boldsymbol{y})$  by discriminating directly between samples  $\mathbf{x}$  from the signal distribution and the output of the reconstruction network  $\hat{\boldsymbol{x}}$ .

PAIRED VARIANT Here we have access to corrupted-uncorrupted pairs  $(\pmb{y},\pmb{x})$  from the joint distribution  $\mathfrak{p}_{\mathbf{Y},\mathbf{X}}$ . Given the masked image  $\pmb{y}$ , the reconstruction is obtained by regressing  $\pmb{y}$  to the associated complete image  $\pmb{x}$  using a  $L^1$  loss. In order to avoid blurry samples, we add an adversarial term in the objective, which helps  $G$  to produce realistic samples. This model is similar to the Vid2Vid (Wang et al., 2018b) model, except that they rely on optical flow which is not available in our case because of the masked regions. The model is illustrated in appendix C.

![](images/587bed5da31d53980ca8e57998a81d338540edb82ab9c80a4b2d71b53e2caed8.jpg)  
(a) SST

![](images/c6f0b366d006c536db11ebed948083dea697b0b14dd3dbcce5b222c8d3963a1c.jpg)  
(b) FaceForensics++

![](images/8efcb43e65aeb601b0323706e5bd6864cd0f1021aa5e272d21a04c3d03afb469.jpg)  
(c) KTH

![](images/af526df2e6b861a6de6eb2330acf6f56d0e15527704876f9670cae9a48d13019.jpg)  
(d) BAIR  
Figure 2: Samples from test sets. SST data (a) are masked with Cloud, natural video datasets (b,c,d) are masked with Remove-Pixel and Raindrops. Sequences are accelerated 3 times to make movements more visible. Each sample from top row to bottom: observed  $\pmb{y}$ , and recovered  $\hat{\pmb{x}}$ .

# 3.4 NETWORK ARCHITECTURE AND TRAINING DETAILS

We use the same networks for all the experiments. For generator  $G$ , we use a ResNet-type self-attention network (Zhang et al., 2019), which is composed of 3D-ResNet blocks and spatial self-attention layers. Frame discriminator  $D_{f}$  is a 2D convolutional NN trained for binary classification. Sequence discriminator  $D_{s}$  uses the same structure as  $D_{f}$  but with 3D convolutions. These networks can process sequences of any time length. See Appendix B for more details about the networks.

Let us now detail the training procedure for each dataset: (a) For SST data, the model is trained on 300 sequences, validated on 66 sequences, and tested on 60 sequences. Each sequence is composed of 24 frames. We use SST data degraded by cloud masks at LWP threshold  $70\mathrm{g / m}^2$  for training since they include sufficient information for both SST and cloud dynamics. (b) For FaceForensics++, KTH, and BAIR, we pick randomly  $5\%$  data for validation, another  $5\%$  data for test, and keep the remaining for training. Sequences are truncated or padded to 35 frames to be able to fit into GPU memory.

For time-independent process Remove-Pixel, we use plain pixel value as feature and directly let the sequence discriminator capture the dynamics. For time-dependent Raindrops, Moving-Vertical-Bar and Cloud, we further reinforce the sequence discriminator  $D_{s}$  to focus on temporal component by extracting inter-frame difference features as the underlying dynamics reflected by this feature is more expressive than plain pixels.  $D_{s}$  will therefore distinguish between  $\phi \equiv [\pmb{y}_2 - \pmb{y}_1,\dots ,\pmb{y}_N - \pmb{y}_{N - 1}]$  and  $\hat{\phi}\equiv [\hat{\pmb{y}}_2 - \hat{\pmb{y}}_1,\dots ,\hat{\pmb{y}}_N - \hat{\pmb{y}}_{N - 1}]$ .

We use hinge loss for Equation 4 as in Zhang et al. (2019). Following standard practice, all three networks are trained using Adam optimizer with a learning rate of  $1 \times 10^{-4}$  and  $(\beta_{1}, \beta_{2}) = (0, 0.999)$ . All networks are initialized with normal distribution with a gain of 0.02. We apply spectral normalization for all parametric layers. The experiments were made on one NVIDIA GeForce GTX Titan X GPU. $^{4}$

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">Raindrops</td><td colspan="3">Remove-Pixel</td><td colspan="3">Vertical-Moving-Bar</td></tr><tr><td>FID</td><td>FVD</td><td>MAE</td><td>FID</td><td>FVD</td><td>MAE</td><td>FID</td><td>FVD</td><td>MAE</td></tr><tr><td rowspan="3">FF++</td><td>Ours</td><td>43.72</td><td>1574.89</td><td>.0834±.0187</td><td>93.28</td><td>1460.02</td><td>.0894±.0137</td><td>19.12</td><td>493.57</td><td>.1304±.0972</td></tr><tr><td>(1)</td><td>75.93</td><td>3424.11</td><td>.1208±.0272</td><td>110.15</td><td>3091.67</td><td>.0752±.0161</td><td>56.58</td><td>5775.25</td><td>.3286±.0815</td></tr><tr><td>(2)</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>9.04</td><td>316.55</td><td>.0494±.0501</td></tr><tr><td rowspan="3">KTH</td><td>Ours</td><td>56.56</td><td>2522.81</td><td>.0380±.0062</td><td>56.16</td><td>2639.24</td><td>.0429±.0037</td><td>39.05</td><td>588.94</td><td>.0711±.0505</td></tr><tr><td>(1)</td><td>71.69</td><td>6400.44</td><td>.0522±.0073</td><td>82.45</td><td>6660.02</td><td>.0403±.0040</td><td>34.90</td><td>3408.19</td><td>.0959±.0402</td></tr><tr><td>(2)</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>11.88</td><td>354.01</td><td>.0268±.0403</td></tr><tr><td rowspan="3">BAIR</td><td>Ours</td><td>27.33</td><td>1194.19</td><td>.0821±.0153</td><td>53.80</td><td>2073.90</td><td>.0997±.0087</td><td>11.55</td><td>496.38</td><td>.1619±.0590</td></tr><tr><td>(1)</td><td>89.87</td><td>4456.08</td><td>.2345±.0274</td><td>140.20</td><td>4014.17</td><td>.1424±.0103</td><td>67.06</td><td>7361.77</td><td>.5579±.0766</td></tr><tr><td>(2)</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>-*</td><td>10.31</td><td>340.97</td><td>.1082±.0873</td></tr></table>

# 3.5 EVALUATION METRICS

Our objective is to find the most plausible sequence. We use as main performance measures of the generated frames, Fréchet Inception Distance (FID, Heusel et al., 2017) and Fréchet Video Distance (FVD, Unterthiner et al., 2018). Both compare the activation distribution of the generated samples from  $\mathsf{p}_{\mathbf{X}}^{G}$  to the real one sampled from  $\mathsf{p}_{\mathbf{X}}$ . These distributions are extracted from activation layers of NNs, which are pre-trained respectively on natural image classification tasks for FID and video classification tasks for FVD. The two distances are calculated for the whole sequence including occluded and non-occluded region. Besides FID and FVD, we also evaluate the reconstruction error as a complimentary metric. We use for that Mean Average Error (MAE), which indicates the absolute deviation from the real data. MAE is calculated solely within the occluded area.

# 4 RESULTS

# 4.1 COMPARISON WITH BASELINES

Results for SST Data Table 2a shows the results for SST data with simulated clouds at different occlusion rates. For most occlusion rates, the generated sequences have an MAE under  $0.1^{\circ}\mathrm{C}$  which is well below the reference baseline (see Table 2a). They also have good FID and FVD values, which means that they are spatially and temporally realistic (See Figure 2a for examples). For heavily occluded area, our model can realistically reconstruct the data around the border, the reconstruction near the center of the cloud is of lower quality. We compare our results in Table 2a at  $70\%$  occlusion, with DINEOF, the sola agnostic method for image reconstruction in IR images. The error reduction w.r.t. DINEOF is about  $40\%$  for MAE. We have not been able to obtain results for Newson et al. (2014) in reasonable time for such complex masks. Note that Newson et al. (2014) specifically designed for imputation in natural videos is not adapted for this type of occlusion.

Table 1: Results for FaceForensics, KTH, and BAIR. Compared with (1) Alvera-Azcarate (2011) and (2) Newson et al. (2014). *Unable to finish.  

<table><tr><td>LWP (g/m2)</td><td>Occluded Area (%)</td><td>FID</td><td>FVD</td><td>MAE (°C)</td></tr><tr><td>55</td><td>79.9± 9.6</td><td>32.49</td><td>134.40</td><td>.1273± .0443</td></tr><tr><td>60</td><td>69.6±12.8</td><td>22.95</td><td>79.13</td><td>.1047± .0396</td></tr><tr><td>65</td><td>55.9±15.1</td><td>17.75</td><td>75.07</td><td>.0988± .0378</td></tr><tr><td>70</td><td>39.5±14.6</td><td>8.01</td><td>40.76</td><td>.0739± .0324</td></tr><tr><td>75</td><td>24.5±11.5</td><td>5.58</td><td>30.07</td><td>.0698± .0305</td></tr><tr><td>80</td><td>13.4± 7.8</td><td>1.77</td><td>9.89</td><td>.0497± .0237</td></tr><tr><td>All</td><td>47.1±11.9</td><td>14.76</td><td>61.55</td><td>.0874± .0347</td></tr></table>

(a) Results with clouds generated at different LWP thresholds.  

<table><tr><td>Method</td><td>FID</td><td>FVD</td><td>MAE (°C)</td></tr><tr><td>Ours</td><td>8.01</td><td>40.76</td><td>.0739±.0324</td></tr><tr><td>Alvera-Azcaúrate (2011)</td><td>27.99</td><td>323.61</td><td>.1214±.0248</td></tr><tr><td>Newson et al. (2014)</td><td>-*</td><td>-*</td><td>-*</td></tr></table>

(b) Comparison of results with clouds at LWP threshold  $70\mathrm{g / m}^2$  . \*Unable to finish.

Table 2: Results for SST dataset.

Results for Videos Table 1 gathers the results obtained for the three natural video datasets with artificial measurements (Raindrops, Remove-Pixel, and Vertical-Moving-Bar). For all measurements, the FID and FVD performance obtained by our model are  $20\% -50\%$  better than DINEOF. This means that our model better controls the both spatial and the temporal generation quality than DINEOF. Globally, we achieve better MAE scores notably for color videos with few exceptions (performance are close for Remove-Pixel). As for Newson et al. (2014), the calculation could not be terminated in a reasonable time for highly complex measurements such as Raindrops, and Remove-Pixel, which make searching for cuboid patches in non-occluded area extremely hard. Newson et al. (2014) performs

better when the form of masks is simple such as the Vertical-Moving-Bars, for which completing patches could be easily found in neighbour frames. However, the computation time of Newson et al. (2014) is much longer than our model. Note that reduced computation time was an argument put forward in their publication. For a 30-frame  $64 \times 64$  video, Newson et al. (2014) costs on average 1 minute, versus around 1 second by our model.

Comparison with Supervised Baselines Table 3 compares our model with the two supervised (unpaired and paired) variants described in Section 3.3. Unsurprisingly, the performance of supervised models is far better than the ones of our unsupervised model. We can find out that the access to the ground truth reduce dramatically all three metrics. By using supervision, FID is halved and FVD is between two

and three times smaller. The error reduction is smaller with MAE. We also notice that the unpaired version performs better than the paired one in terms of sequence completion quality (FVD) as the  $L^1$  loss introduces a strong constraint for the reconstruction. This shows to what extent the absence of ground truth will affect generation quality and the extra difficulty while dealing with partial observations.

<table><tr><td>Method</td><td>FID</td><td>FVD</td><td>MAE</td></tr><tr><td>Ours, Unsupervised</td><td>43.72</td><td>1574.89</td><td>.0834±.0187</td></tr><tr><td>Unpaired, Supervised</td><td>20.86</td><td>575.08</td><td>.0547±.0105</td></tr><tr><td>Paired, Supervised</td><td>22.17</td><td>720.75</td><td>.0555±.0108</td></tr></table>

# 4.2 ABLATION STUDY

We also conduct additional experiments in order to quantify the importance of the temporal component.

In a first series of experiments, we remove the sequence component from our model, i.e. removing the sequence discriminator  $D_{s}$  and replacing 3D generator by 2D one generating individually frames. Table 4 shows that our model clearly improves temporal quality by reducing FVD by a ratio of 7 compared to the

Table 3: Comparison with supervised baselines for FaceForensics++ with Raindrops.  

<table><tr><td>Method</td><td>FID</td><td>FVD</td><td>MAE (°C)</td></tr><tr><td>Ours</td><td>8.01</td><td>40.76</td><td>.0739±.0324</td></tr><tr><td>Recurrent variant</td><td>13.29</td><td>67.37</td><td>.0960±.0431</td></tr><tr><td>Static variant</td><td>35.91</td><td>279.78</td><td>.1036±.0047</td></tr></table>

Table 4: Comparison of results for SST data for ablation study

model without the temporal component (denoted Static variant in the table). Note that FID is also clearly improved by a factor of 4. This gives more evidence that the model is able to exploit temporal dependency for its image completion task. We provide samples for this part in Appendix D in Figure 12.

Our model generates a frame at time  $t$ ,  $\hat{\pmb{x}}_t$  from a whole sequence of observations  $\pmb{y}$ . In a second series of experiments, we conditioned the generation of frames  $\hat{\pmb{x}}_t$  only on past observations. We feed past observations into a convolutional RNN (we used GRU in our experiments) and generate the reconstructed frame, still denoted  $G(\pmb{y})$  by abuse of notation, from the last hidden state of the RNN, which encodes all past observations. The spatial discriminator operates as before, while the sequence discriminator operates on past observations only, instead of the full sequence of observations in our model. See Appendix C for an illustration and for further description. Results in Table 4 - Recurrent variant, show that using only past observations makes the completion less realistic and less accurate, but it still clearly outperforms the model without time dependency.

# 5 RELATED WORK

There is currently, up to our knowledge, no other learning-based approach trying to solve the problem of spatiotemporal data completion in a purely unsupervised manner. We will review below related contributions for image and video reconstruction, data assimilation, and domain translation.

Image Reconstruction Video or more generally spatiotemporal sequence completion can be considered as an extension of image completion problems. The first attempts for image completion and inpainting were all supervised. Xie et al. (2012) uses convolutional NNs for regressing observations to ground truth images. This typically produces blurry outputs. To overcome this issue, some authors introduce textures (Yang et al., 2016), while many others make use of GANs (Pathak et al., 2016; Yu et al., 2018). More recently, unsupervised approaches have been developed by considering only corrupted images. Ulyanov et al. (2017); Lehtinen et al. (2018) show that it is possible to learn the

underlying data distribution and to reconstruct images from observations when a model of observation process is given or when the noise is zero-mean. Such restrictive hypothesis have been removed in the seminal work of Bora et al. (2018). They introduce AmbientGAN to unconditionally generate data distribution without supervision from corrupted observations under the assumption that the stochastic measurement process is known. MisGAN (Li et al., 2019) extend this idea and to learn jointly the mask and the original data distributions. Both contributions objective is data generation and not completion like we do here. Pajot et al. (2019) propose to conditionally recover images from corrupted observations only by solving a maximum a posteriori (MAP) estimation problem, implemented with an adversarial framework. This is limited to still images.

Video Inpainting Video inpainting has been mainly considered in a supervised framework. Object-based (Cheung et al., 2006) and patch-based (Newson et al., 2014) approaches introduced before the deep learning era generally rely on prior segmentation of moving objects and background or strong assumptions on video content. Flow-based methods have been used to model spatial appearance and local pixel movement between consecutive frames. Huang et al. (2016) propose to guide nonparametric patch-based optimization with forward and backward optical flow. Xu et al. (2019); Kim et al. (2019) try to resolve the problem through neural optical flow estimation, which requires extra pre-trained network. More recently end-to-end learning approaches have been proposed. For example, Wang et al. (2018a) propose frame-level generation decomposition by combining a video inpainter with a frame-wise refinement inpainter. Extensions of image inpainting methods are also proposed in Chang et al. (2019). All these learning-based methods are trained with supervision and have been developed for natural videos.

Data Assimilation for Remote Sensing For remote sensing applications, optimal Interpolation (OI) is widely used in operational products (Donlon et al., 2012). It produces a linear estimate for the occluded area. Model-based assimilation methods (Ubelmann et al., 2015) rely on explicit physical dynamic priors and demand significant computational power. Purely data-driven methods based on empirical orthogonal functions (EOF, Beckers & Rixen, 2003) use basically matrix factorization to achieve temporal interpolation. Recent advances in Analog Data Assimilation (AnDA, Lguensat et al., 2017; Fablet et al., 2018) combine analog forecasting methods with data-driven assimilation using implicit knowledge of dynamical prior. These methods rely either on interpolation or exploit some priors on the nature of the underlying process. Recently, learning methods have started to be exploited in this field. Shibata et al. (2017) propose to apply learning-based frame-level inpainting enhanced with optical flow using simple assumptions on pixel movement. In a later paper, Shibata et al. (2018), they recover the missing data using an adversarial approach to supervise on some extra occluded area w.r.t the original partial observations. This approach still reconstruct data frame by frame.

Domain Translation Reconstruction can also be considered as a translation problem between two domains, incomplete observations and full unobserved data. For images, Pix2Pix (Isola et al., 2016) utilizes GANs to project data from domain A to domain B with paired data. CycleGAN (Zhu et al., 2017) propose to use two generator-discriminator pairs to model the transformation between two domains. For videos, Wang et al. (2018b) propose Vid2Vid by adding a multi-scale temporal discriminator in Pix2Pix to supervise the optical flow. RecycleGAN (Bansal et al., 2018) is based on the idea of CycleGAN by adding a temporal transformation in both domains. However, these methods require full data from the two domains, and sometimes the supervision on motion, when no supervision is available in our setting.

# 6 CONCLUSION

We have proposed a GAN-based framework to complete partially observed spatiotemporal data. Our model utilizes a generator to complete missing pixels in observation sequences with the help of two discriminators classifying real and generated observation sequences. We show that our model is able to complete spatiotemporal data without ground truth supervision when we have a stochastic model of the occlusion process. Our results for SST data and natural videos show that the recovered sequences are realistic, especially when the occluded area is highly complex.

# REFERENCES

Barth A. Sirj Jacobs D. Lenartz F. Beckers J. Alvera-Azcarate, A. Data interpolating empirical orthogonal functions (dineof): a tool for geophysical data analyses. Mediterranean Marine Science, 12(3), 2011. doi: 10.12681/mms.64.  
Aayush Bansal, Shugao Ma, Deva Ramanan, and Yaser Sheikh. Recycle-GAN: Unsupervised video retargeting. CoRR, abs/1808.05174, 2018. URL http://arxiv.org/abs/1808.05174.  
J. M. Beckers and M. Rixen. EOF calculations and data filling from incomplete oceanographic datasets. Journal of Atmospheric and Oceanic Technology, 20(12):1839-1856, 2003. doi: 10. 1175/1520-0426(2003)020<1839:ECADFF>2.0.CO;2. URL https://doi.org/10.1175/ 1520-0426 (2003) 020<1839:ECADFF>2.0.CO;2.  
Ashish Bora, Eric Price, and Alexandros G. Dimakis. AmbientGAN: Generative models from lossy measurements. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=Hy7fDog0b.  
Ya-Liang Chang, Zhe Yu Liu, Kuan-Ying Lee, and Winston Hsu. Free-form video inpainting with 3D gated convolution and temporal PatchGAN. CoRR, abs/1904.10247, 2019. URL http://arxiv.org/abs/1904.10247.  
S. S. Cheung, J. Zhao, and M. V. Venkatesh. Efficient object-based video inpainting. In 2006 International Conference on Image Processing, pp. 705-708, Oct 2006. doi: 10.1109/ICIP.2006.312432.  
Craig J. Donlon, Matthew Martin, John Stark, Jonah Roberts-Jones, Emma Fiedler, and Werenfrid Wimmer. The operational sea surface temperature and sea ice analysis (OSTIA) system. Remote Sensing of Environment, 116:140 - 158, 2012. ISSN 0034-4257. doi: https://doi.org/10.1016/j.rse.2010.10.017. URL http://www.sciencedirect.com/science/article/pii/S0034425711002197. Advanced Along Track Scanning Radiometer (AATSR) Special Issue.  
Frederik Ebert, Chelsea Finn, Alex X. Lee, and Sergey Levine. Self-Supervised Visual Planning with Temporal Skip Connections. arXiv e-prints, art. arXiv:1710.05268, Oct 2017.  
Ronan Fablet, Phi Huynh Viet, Redouane Lguensat, Pierre-Henri Horrein, and Bertrand Chapron. Spatio-temporal interpolation of cloudy SST fields using conditional analog data assimilation. Remote Sensing, 10(2), 2018. ISSN 2072-4292. doi: 10.3390/rs10020310. URL https://www.mdpi.com/2072-4292/10/2/310.  
Madec Gurvan, Romain Bourdalle-Badie, Pierre-Antoine Bouttier, Clément Bricaud, Diego Bruciaferri, Daley Calvert, Jérôme Chanut, Emanuela Clementi, Andrew Coward, Damiano Delrosso, Christian Ethé, Simona Flavoni, Tim Graham, James Harle, Doroteaciro Iovino, Dan Lea, Claire Lévy, Tomas Lovato, Nicolas Martin, Sébastien Masson, Silvia Mocavero, Julien Paul, Clément Rouset, Dave Storkey, Andrea Storto, and Martin Vancoppenolle. NEMO ocean engine, October 2017. URL https://doi.org/10.5281/zenodo.3248739. Fix broken cross-references, still revision 8625 from SVN repository.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (eds.), Computer Vision - ECCV 2016, pp. 630-645, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46493-0.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a nash equilibrium. CoRR, abs/1706.08500, 2017. URL http://arxiv.org/abs/1706.08500.  
Jia-Bin Huang, Sing Bing Kang, Narendra Ahuja, and Johannes Kopf. Temporally coherent completion of dynamic video. ACM Trans. Graph., 35(6):196:1-196:11, November 2016. ISSN 0730-0301. doi: 10.1145/2980179.2982398. URL http://doi.acm.org/10.1145/2980179.2982398.

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. CoRR, abs/1611.07004, 2016. URL http://arxiv.org/abs/1611.07004.  
Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. Deep video inpainting. CoRR, abs/1905.01639, 2019. URL http://arxiv.org/abs/1905.01639.  
Jaakko Lehtinen, Jacob Munkberg, Jon Hasselgren, Samuli Laine, Tero Karras, Miika Aittala, and Timo Aila. Noise2noise: Learning image restoration without clean data. CoRR, abs/1803.04189, 2018. URL http://arxiv.org/abs/1803.04189.  
Redouane Lguensat, Pierre Tandeo, Pierre Ailliot, Manuel PULIDO, and Ronan Fablet. The Analog Data Assimilation. Monthly Weather Review, 145(10):4093 - 4107, October 2017. doi: 10.1175/MWR-D-16-0441.1. URL https://hal.archives-ouvertes.fr/hal-01609141.  
Steven Cheng-Xian Li, Bo Jiang, and Benjamin M. Marlin. MisGAN: Learning from incomplete data with generative adversarial networks. CoRR, abs/1902.09599, 2019. URL http://arxiv.org/abs/1902.09599.  
Alasdair Newson, Andres Almansa, Matthieu Fradet, Yann Gousseau, and Patrick Pérez. Video inpainting of complex scenes. SIAM J. Imaging Sciences, 7:1993-2019, 2014.  
Arthur Pajot, Emmanuel de Bezenac, and Patrick Gallinari. Unsupervised adversarial image reconstruction. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=BJg4Z3RqF7.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. CoRR, abs/1604.07379, 2016. URL http://arxiv.org/abs/1604.07379.  
Kyle G. Pressel, Colleen M. Kaul, Tapio Schneider, Zhihong Tan, and Siddhartha Mishra. Large-eddy simulation in an anelastic framework with closed water and entropy balances. Journal of Advances in Modeling Earth Systems, 7(3):1425-1456, 2015. doi: 10.1002/2015MS000496. URL https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2015MS000496.  
Rui Qian, Robby T. Tan, Wenhan Yang, Jiajun Su, and Jiaying Liu. Attentive generative adversarial network for raindrop removal from a single image. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.  
Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. CoRR, abs/1901.08971, 2019.  
Christian Schuldt, Ivan Laptev, and Barbara Caputo. Recognizing human actions: A localsvm approach. In Proceedings of the Pattern Recognition, 17th International Conference on (ICPR'04) Volume 3 - Volume 03, ICPR '04, pp. 32-36, Washington, DC, USA, 2004. IEEE Computer Society. ISBN 0-7695-2128-2. doi: 10.1109/ICPR.2004.747. URL http://dx.doi.org/10.1109/ICPR.2004.747.  
S. Shibata, M. Iiyama, A. Hashimoto, and M. Minoh. Restoration of sea surface temperature images by learning-based and optical-flow-based inpainting. In 2017 IEEE International Conference on Multimedia and Expo (ICME), pp. 193-198, July 2017. doi: 10.1109/ICME.2017.8019401.  
S. Shibata, M. Iiyama, A. Hashimoto, and M. Minoh. Restoration of sea surface temperature satellite images using a partially occluded training set. In 2018 24th International Conference on Pattern Recognition (ICPR), pp. 2771-2776, Aug 2018.  
Mennatullah Siam, Sepehr Valipour, Martin Jagersand, and Nilanan Ray. Convolutional Gated Recurrent Networks for Video Segmentation. arXiv e-prints, art. arXiv:1611.05435, Nov 2016.  
P. Singh and N. Komodakis. Cloud-GAN: Cloud removal for Sentinel-2 imagery using a cyclic consistent generative adversarial networks. In IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, pp. 1772-1775, July 2018. doi: 10.1109/IGARSS.2018.8519033.

Damien Sirjacobs, Aida Alvera-Azcarate, Alexander Barth, Geneviève Lacroix, YoungJe Park, Bouchra Nechad, Kevin Ruddick, and Jean-Marie Beckers. Cloud filling of ocean colour and sea surface temperature remote sensing products over the southern north sea by the data interpolating empirical orthogonal functions methodology. Journal of Sea Research, 65(1):114 - 130, 2011. ISSN 1385-1101. doi: https://doi.org/10.1016/j.seares.2010.08.002. URL http://www.sciencedirect.com/science/article/pii/S1385110110001036.  
N. C. Tang, C. Hsu, C. Su, T. K. Shih, and H. M. Liao. Video inpainting on digitized vintage films via maintaining spatiotemporal continuity. IEEE Transactions on Multimedia, 13(4):602-614, Aug 2011. ISSN 1520-9210. doi: 10.1109/TMM.2011.2112642.  
Clement Ubelmann, Patrice Klein, and Lee-Lueng Fu. Dynamic interpolation of sea surface height and potential applications for future high-resolution altimetry mapping. Journal of Atmospheric and Oceanic Technology, 32(1):177-184, 2015. doi: 10.1175/JTECH-D-14-00152.1. URL https://doi.org/10.1175/JTECH-D-14-00152.1.  
Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky. Deep image prior. CoRR, abs/1711.10925, 2017. URL http://arxiv.org/abs/1711.10925.  
Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. CoRR, abs/1812.01717, 2018. URL http://arxiv.org/abs/1812.01717.  
Chuan Wang, Haibin Huang, Xiaoguang Han, and Jue Wang. Video inpainting by jointly learning temporal structure and spatial details. CoRR, abs/1806.08482, 2018a. URL http://arxiv.org/abs/1806.08482.  
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-to-video synthesis. CoRR, abs/1808.06601, 2018b. URL http://arxiv.org/abs/1808.06601.  
X. Wang, R. Girshick, A. Gupta, and K. H. Non-local neural networks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7794-7803, June 2018c. doi: 10.1109/CVPR.2018.00813.  
Junyuan Xie, Linli Xu, and Enhong Chen. Image denoising and inpainting with deep neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 25, pp. 341-349. Curran Associates, Inc., 2012. URL http://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf.  
Rui Xu, Xiaoxiao Li, Bolei Zhou, and Chen Change Loy. Deep flow-guided video inpainting. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.  
A. Yamashita, A. Matsui, and T. Kaneko. Fence removal from multi-focus images. In 2010 20th International Conference on Pattern Recognition, pp. 4532-4535, Aug 2010. doi: 10.1109/ICPR.2010.1101.  
Chao Yang, Xin Lu, Zhe Lin, Eli Shechtman, Oliver Wang, and Hao Li. High-resolution image inpainting using multi-scale neural patch synthesis. CoRR, abs/1611.09969, 2016. URL http://arxiv.org/abs/1611.09969.  
Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Generative image inpainting with contextual attention. CoRR, abs/1801.07892, 2018. URL http://arxiv.org/abs/1801.07892.  
Han Zhang, Ian Goodfellow, Dimitris Metaxas, and Augustus Odena. Self-attention generative adversarial networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 7354–7363, Long Beach, California, USA, 09–15 Jun 2019. PMLR. URL http://proceedings.mlr.press/v97/zhang19d.html.

<table><tr><td></td><td>Module</td><td>Nb. Input Channel</td><td>Nb. Output Channel</td><td>Activation</td></tr><tr><td colspan="5">Encoder</td></tr><tr><td>1</td><td>3D ResNet block</td><td>\( C_{img} \)</td><td>\( C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>2</td><td>3D ResNet block</td><td>\( C_{base} \)</td><td>\( 16C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>3</td><td>3D ResNet block</td><td>\( 16C_{base} \)</td><td>\( 16C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td colspan="5">Decoder</td></tr><tr><td>4</td><td>3D ResNet block</td><td>\( 16C_{base} \)</td><td>\( 8C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>5</td><td>3D ResNet block</td><td>\( 8C_{base} \)</td><td>\( 4C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>6</td><td>3D ResNet block</td><td>\( 4C_{base} \)</td><td>\( 2C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>7</td><td>Spatial Self-Attention</td><td>\( 2C_{base} \)</td><td>\( 2C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>8</td><td>3D ResNet block</td><td>\( 2C_{base} \)</td><td>\( C_{base} \)</td><td>\( ReLU^* \)</td></tr><tr><td>9</td><td>3D Batch Norm.</td><td>\( 2C_{base} \)</td><td>\( 2C_{base} \)</td><td>ReLU</td></tr><tr><td>10</td><td>3D Conv.</td><td>\( C_{base} \)</td><td>\( C_{img} \)</td><td>tanh</td></tr></table>

(a) Generator structure. Kernel size 3, stride 1.  
*Activation inside the module

<table><tr><td></td><td>Module</td><td>Nb. Input Channel</td><td>Nb. Output Channel</td><td>Spatial Stride</td><td>Activation</td></tr><tr><td>1</td><td>2D/3D Conv.</td><td>Cimg</td><td>Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>2</td><td>2D/3D Conv.</td><td>Cbase</td><td>2Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>3</td><td>2D/3D Conv.</td><td>2Cbase</td><td>4Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>4</td><td>2D/3D Conv.</td><td>4Cbase</td><td>8Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>5</td><td>2D/3D Conv.</td><td>8Cbase</td><td>8Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>6</td><td>2D/3D Conv.</td><td>8Cbase</td><td>8Cbase</td><td>2</td><td>LeakyReLU†</td></tr><tr><td>7</td><td>2D/3D Conv.</td><td>8Cbase</td><td>8Cbase</td><td>1</td><td>LeakyReLU†</td></tr><tr><td>8</td><td>2D/3D Conv.</td><td>8Cbase</td><td>8Cbase</td><td>1</td><td>—</td></tr></table>

(b) PatchGAN Discriminator. Kernel size 3. Stride of temporal dimension is always 1. 2D convolution for frame discriminator  $D_{f}$ , 3D for sequence one  $D_{s}$ .  $^{\dagger}$  Negative slope 0.2.

Table 5: Architecture of networks.

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. CoRR, abs/1703.10593, 2017. URL http:// arxiv.org/abs/1703.10593.
