# D3AD: DYNAMIC DENOISING DIFFUSION PROBABILISTIC MODEL FOR ANOMALY DETECTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Diffusion models have found valuable applications in anomaly detection by capturing the nominal data distribution and identifying anomalies via reconstruction. Despite their merits, they struggle to localize anomalies of varying scales, especially larger anomalies like entire missing components. Addressing this, we present a novel framework that enhances the capability of diffusion models, by extending the previous introduced implicit conditioning approach Meng et al. (2022) in three significant ways. First, we incorporate a dynamic step size computation that allows for variable noising steps in the forward process guided by an initial anomaly prediction. Second, we demonstrate that denoising an only scaled input, without any added noise, outperforms conventional denoising process. Third, we project images in a latent space to abstract away from fine details that interfere with reconstruction of large missing components. Additionally, we propose a fine-tuning mechanism that facilitates the model to effectively grasp the nuances of the target domain. Our method undergoes rigorous evaluation on two prominent anomaly detection datasets VISA and BTAD, yielding state-of-the-art performance. Importantly, our framework effectively localizes anomalies regardless of their scale, marking a pivotal advancement in diffusion-based anomaly detection. All code will be made public upon acceptance.

# 1 INTRODUCTION

Anomaly detection (AD) and related tasks such as identifying out-of-distribution data and detecting novel patterns, holds significant importance within the industrial sector. Applications range from detecting component defects Roth et al. (2022); Zou et al. (2022) and fraudulent activities Ahmed et al. (2016) to assistance in medical diagnoses Baur et al. (2019); Wyatt et al. (2022) through identification of diseases. Overlooked anomalies in these applications could result in adverse financial and safety repercussions. In the manufacturing sector, flawed components which remain undetected lead to high scrap costs or customer complaints. Moreover, manual inspection of defects is a laborious task which often results in visual strain, especially when assessing reflective parts repeatedly. Motivated by these challenges, we explore the intricacies of visual anomaly detection within industrial contexts. In computer vision, anomaly detection entails both classifying images as anomalous or normal and segmenting/localizing anomalous regions.

Typically, due to the scarcity of abnormal samples, an unsupervised approach is often employed for AD whereby a one-class classifier is trained on only nominal data. Such approaches can be grouped into representation-based and reconstruction-based methods. The latter reconstructs an anomalous input image, which is anomaly-free since the model is only trained on nominal data; thereby anomalies can be detected by simple comparison of the input with it's reconstruction. However, previous generative models Bergmann et al. (2019c); Gong et al. (2019) are easily biased towards the flawed input image leading to a reconstruction with the anomaly or artifacts. Diffusion models Sohl-Dickstein et al. (2015); Ho et al. (2020) have shown success in image and video synthesis Nichol et al. (2022); Rombach et al. (2022); Blattmann et al. (2023), 3D reconstruction Poole et al. (2023), music generation Kong et al. (2021) etc. They have also been used for the AD task acquiring promising results Wyatt et al. (2022); Mousakhan et al. (2023) but their full potential in anomaly detection remain untapped.

![](images/323d2ccd7bc038bbe0d9df3dea8e77d288b0b4bfdd2a4220437a3fa8c5a0c597.jpg)  
Figure 1: D3AD segmentation results of anomalies across scales from VisA and BTAD.

![](images/6971ca5455d0c397f7415b06529858f3fa9e9ef1133e0b230b7faf1583926004.jpg)  
Figure 2: Dynamic conditioning whereby the amount of added noise is a function of the input image and training dataset dependent on an initial guess of the severity of the anomaly.

Anomalies occur in diverse forms from small scratches to complete missing components, see Figure 1. In previous AD diffusion models, we observe that simple application of fixed noise to an anomalous input image, known as static implicit conditioning Meng et al. (2022), is insufficient to address the entire range of anomaly types and sizes. Therefore, we propose to compute the number of noising steps (noise amount) as a function of the input image and nominal training set, see Figure 2. This dynamic adjustment aids in precise segmentation of anomalies, which is often the weakest attribute of diffusion models in comparison with representation-based methods. To further abstract away from pixel-level details, we adopt a latent diffusion model and show that a latent representation along with the corresponding reconstruction provides state-of-the-art anomaly heatmaps while requiring less computing resources. Finally, our framework does not require noise to be added at inference time whereby a test image is directly denoised into a predicted reconstruction.

# Our main contributions are as follows:

- We propose a dynamic conditioning mechanism where the maximum noise is computed using prior information about the anomaly provided by a KNN model of domain adapted features.  
- We propose a domain adaptation mechanism that aims to learn the target domain as well as reconstruction errors.  
- We propose to train a latent diffusion model for the task of anomaly detection to achieve precise anomaly heatmaps.  
- We perform extensive evaluation and ablation studies on our approach and demonstrate state-of-the-art performance in segmentation of anomalies at all scales.

# 2 RELATED WORK

Reconstruction Methods These methods hinge on the premise that trained models are unable to generate anomalies, resulting in large disparity between an anomalous input and its reconstruction. Autoencoders have been vastly explored Bergmann et al. (2019c); Gong et al. (2019), however, the reconstructions often include the anomalous region resulting in erroneous anomaly heatmaps. An improvement has been to combine (variational) Autoencoder Kingma & Welling (2014) with adversarial training, leveraging a discriminator, to spot anomalies Baur et al. (2019); Sabokrou et al.

(2018). However, these methods still suffer from significant reconstruction error. GANs have also been explored for anomaly detection. For instance, Schlegl et al. (2017) introduced a feature-wise and visual loss. In their approach, nearest latent representation of input images is iteratively sought. In contrast, Akcay et al. (2019) employed an encoder-decoder-encoder architecture, optimizing both image and latent representation reconstructions. A discriminator then compared features from the original and reconstructed images. Alternative techniques, as cited in Haselmann et al. (2018); Zavrtanik et al. (2021b); Ristea et al. (2022), approach the problem as an in-painting task whereby random patches from images are obscured, and neural networks learn to infer the missing data. DRAEM Zavrtanik et al. (2021a) used an end-to-end approach relying on synthetic data. Though reconstruction-based methods have had some success, they suffer from generated anomalies or artifacts within the reconstructions. Recent innovation have explored the potential of diffusion models in AD making use of an implicit conditioning proposed by SDEdit Meng et al. (2022). Works by Wyatt et al. (2022); Zhang et al. (2023); Mousakhan et al. (2023) have showcased success in achieving high quality anomaly heatmaps however, these approaches fail in the face of large sized defects. Our D3AD method is agnostic to anomaly size and is capable of detecting a wide range of anomalies with varying severity.

Representation Methods These methods gauge the discrepancy between the feature representation of test data and the learned representations of nominal data. This learned representation might either be a prototypical representation or the feature space mapping itself. PaDim Defard et al. (2021) employs a patch-wise extraction and concatenation of features from multiple CNN layers. An empirical sample mean and covariance matrix for each patch's feature vector is then computed. Anomalies are pinpointed based on the Mahalanobis distance between patches. Spade Cohen & Hoshen (2020) emphasizes this distance principle, computing the average distance of an image to its k-nearest neighbours pixel-wise and thresholding to discover anomalies. Patchcore Roth et al. (2022) is a synthesis of both PaDim and Spade, employing a patch strategy, with each patch being compared to a coreset of all other patches. The distance comparison mirrors Spade, focusing on the average distance to k-nearest neighbours within the coreset. Similarly CFA Lee et al. (2022) combines the patch based approach with metric learning. Another line of work utilises normalising flows Rudolph et al. (2020); Yu et al. (2021); Gudovskiy et al. (2022) to directly estimate the likelihood function whereby sample in the low-density regions can instantly be identified as anomalies. Nonetheless, none of these approaches generate an anomaly-free rendition of the input image. This capability is highly sought after in an industrial context, as it fosters trust and provides valuable insights into the model's decision-making process.

Domain Adaptation Most prior approaches employ pretrained feature extractors to map raw images into a latent space. However, these feature extractors often lack adaptation to the target domain, resulting in artifacts for reconstruction-based methods and inaccuracies in representation-based comparisons. To address this, domain adaptation techniques have been explored. For instance, SimpleNet Liu et al. (2023) enhances a pretrained feature extractor with a domain adaptation layer and uses Gaussian noise to perturb features and training a discriminator to distinguish native from perturbed features. In contrast, RD4AD Deng & Li (2022) adopts an encoder-decoder structure, with the student network receiving the teacher's latent representation instead of the original image.  $\mathrm{RD} + +$  Tien et al. (2023) extends this approach by incorporating additional projection layers to filter out anomalous information. Inspired by these successes, we implement a fine-tuning strategy for the pretrained feature extractors in order to leverage the demonstrated benefit.

# 3 BACKGROUND

We use a class of generative models called diffusion probabilistic models Sohl-Dickstein et al. (2015); Ho et al. (2020). In these, parameterized Markov chains with  $T$  steps are used to gradually add noise to input data  $\pmb{x}_0\sim q(\pmb{x}_0)$  until all information is lost. The inspiration stems from principles of nonequilibrium thermodynamics Sohl-Dickstein et al. (2015). Neural networks are then parameterised to learn the unknown reverse process, in effect learning a denoising model. The forward process  $q$  is defined as:

$$
q \left(\boldsymbol {x} _ {t} \mid \boldsymbol {x} _ {t - 1}\right) = \mathcal {N} \left(\boldsymbol {x} _ {t}; \sqrt {1 - \beta_ {t}} \boldsymbol {x} _ {t - 1}, \beta_ {t} \mathbf {I}\right) \tag {1}
$$

$$
q \left(\boldsymbol {x} _ {t} \mid \boldsymbol {x} _ {0}\right) = \mathcal {N} \left(\boldsymbol {x} _ {t}; \sqrt {\bar {\alpha} _ {t}} \boldsymbol {x} _ {0}, (1 - \bar {\alpha} _ {t}) \mathbf {I}\right) \tag {2}
$$

$$
\boldsymbol {x} _ {t} = \sqrt {\bar {\alpha} _ {t}} \boldsymbol {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon}, \quad \text {w h e r e} \quad \boldsymbol {\epsilon} \sim \mathcal {N} (0, \mathbf {I}) \tag {3}
$$

Usually the  $\beta_{t}$  are chosen as hyperparameters of the form  $\beta_{t} \in (0,1)$  with a variance schedule  $\beta_{0} < \beta_{1} < \ldots < \beta_{T}$  such that the signal of the input gets sequentially disturbed. For direct sampling the  $\beta_{t}$  parameters are simplified to a compactor notation:  $\alpha_{t} = 1 - \beta_{t}$  and  $\bar{\alpha}_{t} = \prod_{s=1}^{t} \alpha_{s}$ . Furthermore with large  $T$  and small  $\beta_{t}$ , the distribution of  $x_{T}$  approaches a standard normal which enables sampling from a normal distribution in the reverse process  $p$  parameterized by  $\theta$ . This is defined as:

$$
p _ {\theta} \left(\boldsymbol {x} _ {t - 1} \mid \boldsymbol {x} _ {t}\right) = \mathcal {N} \left(\boldsymbol {x} _ {t - 1}; \boldsymbol {\mu} _ {\theta} \left(\boldsymbol {x} _ {t}, t\right), \beta_ {t} \mathbf {I}\right) \tag {4}
$$

This corresponds to the DDPM Ho et al. (2020) formulation, where the variance is equivalent to the forward process while other works found better performance with learning the covariance matrix Nichol & Dhariwal (2021). DDPM is trained by predicting the initially added noise  $\epsilon$  which corresponds to predicting  $\mu_{\theta}$  and leads to the training objective:

$$
L _ {\text {s i m p l e}} (\theta) = \mathbb {E} _ {t, \boldsymbol {x} _ {0}, \epsilon} [ | | \boldsymbol {\epsilon} - \boldsymbol {\epsilon} _ {\theta} \left(\sqrt {\bar {\alpha} _ {t}} \boldsymbol {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon}, t\right) | | _ {2} ^ {2} ] \tag {5}
$$

The noising and denoising is performed in pixel space which is computationally expensive therefore Rombach et al. (2022) proposed to utilise latent spaces. An encoder  $\mathcal{E}$  of a continuous or quantized VAE is used to project an image  $x_0$  into a lower dimension  $z_0 = \mathcal{E}(x_0)$  while a decoder  $\mathcal{D}$  aims to reconstruct this such that  $x_0 \simeq \hat{x}_0 = \mathcal{D}(z_0)$ . The following objective function is used:

$$
L _ {s i m p l e - l a t e n t} (\theta) = \mathbb {E} _ {t, \mathcal {E} \left(\boldsymbol {x} _ {0}\right), \epsilon} [ \| \epsilon - \epsilon_ {\theta} \left(\sqrt {\bar {\alpha} _ {t}} \boldsymbol {z} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon , t\right) \| _ {2} ^ {2} ] \tag {6}
$$

A faster sampling approach is proposed by DDIM Song et al. (2022) where a non-Markovian formulation of the DDPM objective is employed allowing sampling steps to be omitted. This implies that a diffusion model trained according to objective Eq. 5 or Eq. 6 can be used to accelerate the sampling without the need for retraining. Their proposed sampling procedure is:

$$
\boldsymbol {x} _ {\tau_ {i - 1}} = \sqrt {\bar {\alpha} _ {\tau_ {i - 1}}} \boldsymbol {f} _ {\theta} ^ {(\tau)} \left(\boldsymbol {x} _ {\tau}\right) + \sqrt {1 - \bar {\alpha} _ {\tau_ {i - 1}} - \sigma_ {\tau_ {i}} ^ {2}} \boldsymbol {\epsilon} _ {\theta} \left(\boldsymbol {x} _ {\tau_ {i}}, \tau_ {i}\right) + \sigma_ {\tau_ {i}} \boldsymbol {\epsilon} _ {\tau_ {i}} \tag {7}
$$

Here  $\tau_{i}, i \in [1, \dots, S]$  acts as an index for subset  $\{\pmb{x}_{\tau_1}, \dots, \pmb{x}_{\tau_S}\}$  of length  $S$  with  $\tau$  as increasing sub-sequence of  $[1, \dots, T]$ . Moreover, an estimation of  $\pmb{x}_0$  is obtained at every time step, denoted by  $\pmb{f}_{\theta}^{(t)}(\pmb{x}_t) = \frac{\pmb{x}_t - \sqrt{1 - \overline{\alpha}_t} \epsilon_{\theta}(\pmb{x}_t, t)}{\sqrt{\overline{\alpha}_t}}$  which utilizes the error prediction  $\epsilon$  according to equation 3. DDIM further demonstrates varying levels of stochasticity within the model with also a fully deterministic version which corresponds to  $\sigma_{\tau_i} = 0$  for all  $\tau_i$ .

Guidance and conditioning the sampling process of diffusion models has been recently explored and often requires training on the conditioning with either an extra classifier Dhariwal & Nichol (2021) or classifier-free guidance Ho & Salimans (2021). Recent work on AD with diffusion models Mousakhan et al. (2023) showed a guiding mechanism which does not require explicit conditional training. Guidance is achieved directly during inference by updating the predicted noise term using  $\boldsymbol{x}_0$  or respectively  $\boldsymbol{z}_0$  as:

$$
\hat {\boldsymbol {\epsilon}} _ {t} = \boldsymbol {\epsilon} _ {\theta} (\boldsymbol {x} _ {t}, t) - \eta \sqrt {1 - \bar {\alpha} _ {t}} (\tilde {\boldsymbol {x}} _ {t} - \boldsymbol {x} _ {t}) \quad \text {w i t h} \quad \tilde {\boldsymbol {x}} _ {t} = \sqrt {\bar {\alpha} _ {t}} \boldsymbol {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon} _ {\theta} (\boldsymbol {x} _ {t}, t) \tag {8}
$$

where  $\eta$  controls the temperature of guidance. This updated noise term can then be used in the DDIM sampling formulation 7 to result in the intended reconstruction  $\hat{z}_0$  and corresponding  $\hat{x}_0$ .

# 4 METHOD

Diffusion models for AD learn the distribution of only nominal data such that they are unable to reconstruct anomalous regions leading to a large distance between input image  $\boldsymbol{x}_0$  and its reconstruction  $\hat{\boldsymbol{x}}_0$ . Previous approaches rely on implicit conditioning Meng et al. (2022), whereby the input is noised until a fixed time step  $\hat{T} < T$  such that some input signal remains allowing for targeted reconstruction. We improve on this in two ways, first we discover that an noiseless and only scaled input  $\boldsymbol{x}_{\hat{T}} = \boldsymbol{x}_0\sqrt{\bar{\alpha}_{\hat{T}}}$  is optimal for anomaly segmentation since it sufficiently reinforces the implicit conditioning applied on the model. Second we propose to choose forward time step  $\hat{T}$  dynamically based on an initial estimate of the anomaly. Furthermore, we adopt the architecture of

unconditional latent diffusion model to abstract away from pixel-level representation which allows for improved reconstruction of large anomalies such as missing components in a resource efficient latent space. Our reconstruction and dynamic implicit conditioning frameworks are illustrated in Figure 3. Algorithm 1 describes the reconstruction process where we utilise the error correction (lines 6 and 7), proposed by DDAD Mousakhan et al. (2023), for guidance and the DDIM (Eq. 7) sampling procedure. Algorithm 2 details our dynamic conditioning mechanism for selecting optimum  $\hat{T}$  for the forward process. Training the diffusion model is according to the objective function in Eq. 6 without modifications.

# 4.1 DYNAMIC IMPLICIT CONDITIONING

We introduce dynamic implicit conditioning (DIC) into the model's architecture. Specifically, we set a maximum implicit conditioning level denoted by  $T_{max} \in \{1, \dots, T\}$ . This is selected such that the signal-to-noise ratio remains high. We then establish a quantization of the maximum steps into increments ranging up to  $T_{max}$  with which we compute the dynamic implicit conditioning level  $\hat{T}$  for each image according to an initial estimate of the anomaly.

Bin construction Our quantization is founded upon equidistant bins denoted as  $b \in B$ . These bins are determined from the average KNN distances of the training set's feature representations. Given that  $\phi$  is a pretrained domain adapted feature extractor, and  $\phi_j$  outputs the feature map of the  $j^{\text{th}}$  layer block, for data point  $x_0 \in \mathcal{X}_{\text{Train}}$ , the features are extracted as  $y_0 = \phi_j(x_0)$  with  $y_0 \in Y_{\text{Train}}$ . Utilizing  $y_0$ , a KNN-search is executed on the entire feature training set  $Y_{\text{Train}}$  using the L1-Norm. The K-nearest neighbors of  $y_0$  are represented by the set  $\{y_{s_1}, \dots, y_{s_K}\}$ . Subsequently, we compute the mean distance to these KNNs and denote it as  $\bar{y}_0$ . While this method is susceptible to outliers due to its reliance on the arithmetic mean, it is anticipated that anomalous data will be substantially more distinct than regular data. Thus, any outlier within the regular data would be beneficial as it would lead to a wider range for the bins. We compute the average distance for each sample in the training set. Furthermore using the computed average distances, we delineate  $|B|$  evenly spaced bins.

Dynamic Implicit Conditioning (DIC) We denote DIC by function  $g(\pmb{x}_0, \mathcal{X}_{Train}, T_{max})$  described in Algorithm 2. A visual representation of this mechanism is illustrated in Figure 3. During inference, for a new image  $\pmb{x}_0$ , we first utilise  $\phi_j$  to extract features of  $\pmb{x}_0$  and perform a KNN search on  $Y_{train}$ . The distances are averaged to compute  $\bar{y}_0$  which is then placed into bin  $b$  via a binary search function  $\psi$  on all  $b \in B$ . The selected bin  $b$  serves as an initial estimate of the severity of the anomaly in the input image compared to the nominal training data. The dynamic time step  $\hat{T}$  is then simply computed as a fraction of  $T_{max}$  based on the selected bin.

![](images/21cc4b2840074054de914ff0ffb03f10506ebe724c46ad456a80812706ff5e01.jpg)  
Figure 3: Reconstruction Architecture: An input  $\boldsymbol{x}_0$  is fed to the DIC to determine the level it must be perturbed  $\hat{T}$ .  $\boldsymbol{x}_0$  is also projected to a latent representation  $\boldsymbol{z}_0$ . Denoising is performed in the latent space leading to a predicted latent  $\hat{\boldsymbol{z}}_0$  which is decoded into a reconstruction  $\hat{\boldsymbol{x}}_0$ . DIC: The average distance of extracted features of a test image to the K nearest neighbours from the training set is quantized, using equally sized predefined bins, to then determine the dynamic noisng step  $\hat{T}$ .

![](images/441365672436e5254d09772568bc467f3ae3136748ca49ead8c26afa8f272a4b.jpg)

Algorithm 1 Dynamic Reconstruction  
1: input  $\mathbf{x}_0$   
2:  $\hat{T} = g(\mathbf{x}_0, \mathcal{X}_{Train}, T_{max})$   
3:  $\mathbf{z}_0 = \mathcal{E}(\mathbf{x}_0)$   
4:  $\mathbf{z}_{\hat{T}} = \mathbf{z}_0 \sqrt{\bar{\alpha}_{\hat{T}}}$  no noise  
5: for  $t = \hat{T}, \dots, 1$  do  
6:  $\tilde{\mathbf{z}}_t = \sqrt{\bar{\alpha}_t} \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta (\mathbf{z}_t, t)$   
7:  $\hat{\epsilon}_t = \epsilon_\theta (\mathbf{z}_t, t) - \eta \sqrt{1 - \bar{\alpha}_t} (\tilde{\mathbf{z}}_t - \mathbf{z}_t)$   
8:  $\hat{\mathbf{z}}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \mathbf{z}_{\theta,0} + \sqrt{1 - \bar{\alpha}_{t-1}} \hat{\epsilon}_t$   
9: end for  
10:  $\hat{\mathbf{x}}_0 = \mathcal{D}(\hat{\mathbf{z}}_0)$   
11: return  $\hat{\mathbf{x}}_0, \hat{\mathbf{z}}_0$

Algorithm 2 Dynamic Implicit Conditioning  $g$  
1: input  $\mathbf{x}_0$   
2: input  $T_{max}$   
3:  $Y_{Train} = \phi_j(\mathcal{X}_{Train})$   
4:  $\mathbf{y}_0 = \phi_j(\mathbf{x}_0)$   
5:  $\{\mathbf{y}_{s_1},\dots,\mathbf{y}_{s_K}\} = \mathrm{KNN}(\mathbf{y}_0,Y_{train},K)$   
6:  $\bar{\mathbf{y}}_0 = \frac{1}{K}\sum_{j = 1}^{K}||\mathbf{y}_0 - \mathbf{y}_{s_j}||$   
7:  $b = \psi (\bar{\mathbf{y}}_0)\#$  binary search  
8:  $\hat{T} = \left\lfloor \frac{b}{|B|} T_{max}\right\rfloor$   
9: return  $\hat{T}$

# 4.2 ANOMALY SCORING AND MAP CONSTRUCTION

We adopt the convention of comparing the input image with its reconstruction to generate the final anomaly map as illustrated in Figure 4. We compare the latent representation  $\mathbf{z}_0$  with its reconstruction  $\hat{\mathbf{z}}_0$  to construct a latent anomaly map  $l_{map}$ . Similarly, we compare the features of the input image  $\mathbf{x}_0$  against its reconstruction  $\hat{\mathbf{x}}_0$  to construct a feature anomaly map  $f_{map}$ . A weighted combination generates the final anomaly map  $A_{map}$ .

The feature anomaly map  $f_{map}$  is determined by first computing the features of an input image  $\mathbf{x}_0$  and its reconstruction  $\hat{\mathbf{x}}_0$  using a pretrained and domain adapted feature extractor  $\phi$  (section 4.3). A cosine distance between the extracted feature blocks at  $\mathbb{J} \subseteq \{1, \dots, J\}$  layers of a ResNet-34 yields the feature anomaly map. Given that feature blocks at different layers may present divergent dimensionalities, these are upsampled to achieve uniformity. The feature anomaly map  $f_{map}$  is articulated as  $f_{map}(\mathbf{x}_0, \hat{\mathbf{x}}_0) = \sum_{j \in \mathbb{J}} (\cos_d(\phi_j(\mathbf{x}_0), \phi_j(\hat{\mathbf{x}}_0))$ .

Since our approach relies on learning a denoising diffusion model on the latent representation, we further compute distances between the input image latent representation  $\mathbf{z}_0$  and its reconstructed counterpart  $\hat{\mathbf{z}}_0$ . Utilizing the L1-Norm for each pixel, a latent anomaly map is deduced as  $l_{map}(\mathbf{z}_0, \hat{\mathbf{z}}_0) = ||\mathbf{z}_0 - \hat{\mathbf{z}}_0||_1$ .

The final anomaly map  $A_{map}$  is simply a linear combination of the normalized feature-based distance and the latent pixel-wise distance as follows:

$$
A _ {\text {m a p}} = \lambda * l _ {\text {m a p}} \left(\boldsymbol {z} _ {0}, \hat {\boldsymbol {z}} _ {0}\right) + (1 - \lambda) * f _ {\text {m a p}} \left(\boldsymbol {x} _ {0}, \hat {\boldsymbol {x}} _ {0}\right) \tag {9}
$$

Subsequently, an established threshold facilitates the categorization of every pixel and image, marking them as either anomalous or typical. The global image anomaly score is selected as the maximum pixel-level anomaly score within the entire image.

![](images/b4cd160ea5a2c5675cf3aa70260644e10c6f8eac69f88e14b368f273cc744fba.jpg)  
Figure 4: Overview of the Anomaly Map construction. Feature heatmap  $(f_{map})$  are computed as cosine distances of the features of the input  $x_0$  and its reconstruction  $\hat{x}_0$  whereas latent heatmap  $(l_{map})$  is calculated using an  $\mathcal{L}1$  distance between the corresponding latent representations of  $x_0$  and  $\hat{x}_0$ . These combine linearly to form the final anomaly heatmap  $(A_{map})$ .

# 4.3 DOMAIN ADAPTATION

We leverage domain-adapted features for both the dynamic implicit conditioning and the construction of the feature anomaly map  $f_{map}$ . Our objective is to grasp the intricacies associated with the target domain. With the use of variational autoencoders (VAEs) having pretrained encoders and decoder introduces artifacts and reconstruction inaccuracies. These are incorrectly flagged as anomalous regions during comparison. To address this, we introduce a loss function to fine-tune the feature extractor  $\phi$  by further training for  $\gamma$  epochs. This function is designed to minimize the feature distance between the input image  $x_0$  and its reconstruction  $\hat{x}_0$  as follows where GAP refers to global average pooling:

$$
L _ {D A} \left(\boldsymbol {x} _ {0}, \hat {\boldsymbol {x}} _ {0}\right) = \sum_ {j = 1} ^ {J} \operatorname {G A P} \left(1 - \frac {\phi_ {j} \left(\boldsymbol {x} _ {0}\right) ^ {T} \phi_ {j} \left(\hat {\boldsymbol {x}} _ {0}\right)}{\left| \left| \phi_ {j} \left(\boldsymbol {x} _ {0}\right) \right| \right| \left| \left| \phi_ {j} \left(\hat {\boldsymbol {x}} _ {0}\right) \right| \right|}\right). \tag {10}
$$

# 5 EXPERIMENTS

Datasets We employ two widely used benchmarking datasets to evaluate the veracity of our approach, namely VisA Zou et al. (2022) and BTAD Mishra et al. (2021) dataset. VisA dataset presents a collection of 10,821 high-resolution RGB images, segregated into 9,621 regular and 1,200 anomalous instances. Comprehensive annotations are available in the form of both image and pixel-level labels. The dataset comprises of 12 different classes with a large variety of scale and type of anomalies. BTAD dataset comprises of RGB images showcasing three unique industrial products. There are 2540 images in total where each anomalous image is paired with a pixel-level ground truth mask.

Evaluation Metrics We evaluate our approach using standard metrics for anomaly detection, namely pixel-wise AUROC (P-AUROC), image-wise AUROC (I-AUROC) and the PRO metric. P-AUROC is ascertained by setting a threshold on the anomaly score of individual pixels. A critical caveat of P-AUROC is its potential for overestimation, primarily because a majority of pixels are typically normal. Such skewed distribution occasionally renders a misleadingly optimistic performance portrayal. Addressing this limitation, the PRO metric Bergmann et al. (2019a) levels the playing field by ensuring equal weighting for both minuscule and pronounced anomalies. This balance is achieved by averaging the true positive rate over regions defined by the ground truth, thereby offering a more discerning evaluative metric making it our primary choice for evaluation. The image-wise AUROC (I-AUROC) is employed to present an evaluation of image-based anomaly detection, where precise segmentation of the anomaly is unimportant.

Implementation Details We employ an unconditional Unet from Rombach et al. (2022) with an 8x downsampling within our diffusion model. For KNN, we set  $K = 20$  with  $\mathcal{L}1$  distance. Both dynamic conditioning and anomaly map construction utilize a ResNet-34 pretrained on ImageNet and fine-tuned. Domain adaptation is performed for up to 3 epochs using identical Unet settings.  $T_{max}$  is set at 80 for VisA and remains unchanged for BTAD. We chose  $|B| = 10$  which leads to a percentage-quantization mapping of increments of  $10\%$  steps of  $T_{max}$ . However, we set the minimum bin to 2, ensuring that we don't rely solely on prior information. Lastly, the DDIM formulation with 10 steps is adopted for sampling, with the DIC step rounded to the nearest multiple of 10. All experiments were carried out on one Nvidia RTX 8000. Further implementational details are present in Appendix A.1.

Anomaly Detection Results We conduct comprehensive experiments on the VisA dataset to evaluate the capability of our proposed method in detecting and segmenting anomalies. Table 1 details the performance of our method. Notably, D3AD excels in 8 of the 12 classes in segmentation accuracy as evident from PRO values, and in 3 of 12 classes for I-AUROC whilst achieving comparable performance in remaining classes. The aggregate performance across all classes yields an I-AUROC of  $96.0\%$ , paralleling the performance of the state-of-the-art method, RD4AD. Whereas there is a clear superiority of our method in segmentation achieving an average of  $94.1\%$ , outperforming the contemporary state-of-the-art by  $2.7\%$  points.

In an evaluation alongside other diffusion-based models, as documented in Table 2, D3AD achieves superior anomaly localisation performance on the VisA benchmark. When assessed using PRO and

P-AUROC, 3DAD demonstrates an enhancement, achieving results higher by at least  $0.9\%$  points for both metrics compared to previous diffusion state-of-the-art approaches. Figure 1 offers a teaser of D3AD's qualitative performance, with a comprehensive evaluation provided in appendix A.2. Significantly, the method excels in precise segmentation and effectively handling large anomalies.

Further results from the BTAD benchmark are consolidated in Table 3. Here, D3AD exhibits competitive performance in terms of I-AUROC. More prominently, and following previous trend, segmentation evaluated using PRO highlight our method achieving unparalleled results, surpassing the closest competitors by a margin of 5.9 percentage points.

![](images/c9b238dbe9d2e02da1781bd3eab7b1a27860ee2a0acf31aa013bc409aa92bfe4.jpg)  
Figure 5: Histogram of the binning values for the training set in blue and test set in orange, showing a distribution shift to larger values for the test set. Displayed are categories from VisA and BTAD.

![](images/fc65806f5c30fbc25ead3ffe37e49deb4b70639cfcfc3c1ca59eb029038df0fc.jpg)

![](images/3c9ebe855fe9dc4c5e26fcddf915212a6afb3c123d7805106596bb3eb185d340.jpg)

![](images/cb8d3b9e86d379b627abdf4c71a111c83893dddeed752aeab756d7a3c66a6dd1.jpg)

![](images/74735625102a7be58666170e911e1cfe594f7b5e28f5297003b8044803c60fce.jpg)

Table 1: Anomaly classification and localization performance (I-AUROC, PRO) of various methods on VisA benchmark. The best results are highlighted in bold.  

<table><tr><td rowspan="2">Method</td><td colspan="4">Representation-based</td><td colspan="2">Reconstruction-based</td></tr><tr><td>SPADE</td><td>PaDiM</td><td>RD4AD</td><td>PatchCore</td><td>DRAEM</td><td>D3AD (Ours)</td></tr><tr><td>Candle</td><td>(91.0,93.2)</td><td>(91.6,95.7)</td><td>(92.2,92.2)</td><td>(98.6,94.0)</td><td>(91.8,93.7)</td><td>(95.6,92.7)</td></tr><tr><td>Capsules</td><td>(61.4,36.1)</td><td>(70.7,76.9)</td><td>(90.1,56.9)</td><td>(81.6,85.5)</td><td>(74.7,84.5)</td><td>(88.5,95.7)</td></tr><tr><td>Cashew</td><td>(97.8,57.4)</td><td>(93.0,87.9)</td><td>(99.6,79.0)</td><td>(97.3,94.5)</td><td>(95.1,51.8)</td><td>(94.2,89.4)</td></tr><tr><td>Chewing gum</td><td>(85.8,93.9)</td><td>(98.8,83.5)</td><td>(99.7,92.5)</td><td>(99.1,84.6)</td><td>(94.8,60.4)</td><td>(99.7,94.1)</td></tr><tr><td>Fryum</td><td>(88.6,91.3)</td><td>(88.6,80.2)</td><td>(96.6,81.0)</td><td>(96.2,85.3)</td><td>(97.4,93.1)</td><td>(96.5,91.7)</td></tr><tr><td>Macaroni1</td><td>(95.2,61.3 )</td><td>(87.0,92.1)</td><td>(98.4,71.3)</td><td>(97.5,95.4)</td><td>(97.2,96.7)</td><td>(94.3,99.3)</td></tr><tr><td>Macaroni2</td><td>(87.9,63.4)</td><td>(70.5,75.4)</td><td>(97.6,68.0)</td><td>(78.1,94.4)</td><td>(85.0,92.6)</td><td>(92.5,98.3)</td></tr><tr><td>PCB1</td><td>(72.1,38.4)</td><td>(94.7,91.3)</td><td>(97.6,43.2)</td><td>(98.5,94.3)</td><td>(47.6,24.8)</td><td>(97.7,96.4)</td></tr><tr><td>PCB2</td><td>(50.7,42.2)</td><td>(88.5,88.7)</td><td>(91.1,46.4)</td><td>(97.3,89.2)</td><td>(89.8,49.4)</td><td>(98.3,94.0)</td></tr><tr><td>PCB3</td><td>(90.5,80.3)</td><td>(91.0,84.9)</td><td>(95.5,80.3)</td><td>(97.9,90.9)</td><td>(92.0,89.7)</td><td>(97.4,94.2)</td></tr><tr><td>PCB4</td><td>(83.1,71.6)</td><td>(97.5,81.6)</td><td>(96.5,72.2)</td><td>(99.6,90.1)</td><td>(98.6,64.3)</td><td>(99.8,86.4)</td></tr><tr><td>Pipe fryum</td><td>(81.1,61.7)</td><td>(97.0,92.5)</td><td>(97.0,68.3)</td><td>(99.8,95.7)</td><td>(100,75.9)</td><td>(96.9,97.2)</td></tr><tr><td>Average</td><td>(82.1,65.9)</td><td>(89.1,85.9)</td><td>(96.0,70.9)</td><td>(95.1,91.2)</td><td>(88.7,73.1)</td><td>(96.0,94.1)</td></tr></table>

Ablation Studies To understand the significance of each component in our D3AD model, we executed an ablation study using the VisA dataset to evaluate our proposed dynamic implicit conditioning mechanism, domain adapted feature extractor and input scaling without noising method.

Table 4 delves into the efficacy of our dynamic implicit conditioning (DIC). The DIC was compared against each quartile of the selected  $T_{max}$ , ranging from  $25\%$  to  $100\%$  of 80. The DIC consistently registered superior I-AUROC and P-AUROC scores, surpassing the second-best 80-step static model by margins of 0.6 and 1.2 percentage points, respectively. While PRO scores remained fairly consistent across different maximum step choices, the 20-step model slightly outperformed others with a score of 94.3, a slender 0.2 percentage points above the DIC. Given that PRO evaluates anomalies

Table 2: Detection and segmentation performance of diffusion based methods (AnoDDPM Wyatt et al. (2022), DiffusionAD Zhang et al. (2023), DDAD Mousakhan et al. (2023)) on VisA.  

<table><tr><td>Method</td><td>AnoDDPM</td><td>DiffusionAD</td><td>DDAD</td><td>D3AD (Ours)</td></tr><tr><td>I-AUROC</td><td>78.2</td><td>97.8</td><td>99.3</td><td>96.0</td></tr><tr><td>P-AUROC</td><td>-</td><td>-</td><td>97.0</td><td>97.9</td></tr><tr><td>PRO</td><td>60.5</td><td>93.2</td><td>92.0</td><td>94.1</td></tr></table>

Table 3: Anomaly classification and localization performance (I-AUROC, PRO) of various methods on BTAD benchmark. The best results are highlighted in bold.  

<table><tr><td rowspan="2">Method</td><td colspan="5">Representation-based</td><td>Reconstruction-based</td></tr><tr><td>FastFlow</td><td>CFA</td><td>PatchCore</td><td>RD4AD</td><td>RD++</td><td>D3AD (Ours)</td></tr><tr><td>Class 01</td><td>(99.4,71.7)</td><td>(98.1,72.0)</td><td>(96.7,64.9)</td><td>(96.3,75.3)</td><td>(96.8,73.2)</td><td>(98.9,80.0)</td></tr><tr><td>Class 02</td><td>(82.4,63.1)</td><td>(85.5,53.2)</td><td>(81.4,47.3)</td><td>(86.6,68.2)</td><td>(90.1,71.3)</td><td>(87.0,71.7)</td></tr><tr><td>Class 03</td><td>(91.1,79.5)</td><td>(99.0,94.1)</td><td>(100.0,67.7)</td><td>(100.0,87.8)</td><td>(100.0,87.4)</td><td>(99.7,97.8)</td></tr><tr><td>Average</td><td>(91.0,71.4)</td><td>(94.2,73.1)</td><td>(92.7,60.0)</td><td>(94.3,77.1)</td><td>(95.6,77.3)</td><td>(95.2,83.2)</td></tr></table>

uniformly across all scales, and P-AUROC is more sensitive to large-scale anomalies, our observations suggest that the DIC adeptly identifies large anomalies, without compromising its efficiency across varying scales. The distribution of the initial signal is depicted in Figure 5 while Figure 6 shows the qualitative effect of DIC. It is apparent that a dynamically computed time step (DIC Mask) provided the most similar anomaly mask prediction to the ground truth (GT) mask, in comparison to fixed time step masks shown from  $100\% -25\%$  of  $T$ .

Table 5 illustrates the effects of the domain adaptation in the feature extractor and introducing a scaled, yet noiseless, input. Using a model without domain-adapted feature extraction and conventional noised input as the baseline, we observe notable improvements with the integration of each component. Particularly, the modified implicit conditioning, indicated as "downscaling (DS)" in the table, emerges as the most impactful modification. A detailed qualitative visualisation is shown in appendix Figures 10 to 13 whereas a quantitative study of this effect is present in Figure 14.

![](images/d8c77f930d0f142c0d66275a7ac7377a1ba787fff0fb3f139ee5102c3af12193.jpg)  
Figure 6: Overview of prediction masks for different levels of maximum static noise levels and the DIC. DIC tends to segment large anomalies more faithfully

Table 4: Impact of Dynamic Implicit Conditioning (DIC)  

<table><tr><td rowspan="2">Max. Step</td><td colspan="3">Performance</td></tr><tr><td>I-AUROC ↑</td><td>PRO ↑</td><td>P-AUROC ↑</td></tr><tr><td>25%(20)</td><td>95.2</td><td>94.3</td><td>96.7</td></tr><tr><td>50%(40)</td><td>94.7</td><td>94.1</td><td>96.6</td></tr><tr><td>75%(60)</td><td>95.0</td><td>94.2</td><td>96.7</td></tr><tr><td>100%(80)</td><td>95.4</td><td>94.0</td><td>96.7</td></tr><tr><td>DIC(g(.))</td><td>96.0</td><td>94.1</td><td>97.9</td></tr></table>

Table 5: Impact of Downscaling (DS) and Domain Adaptation (DA)  

<table><tr><td colspan="2">Ablation</td><td colspan="3">Performance</td></tr><tr><td>DS</td><td>DA</td><td>I-AUROC ↑</td><td>PRO ↑</td><td>P-AUROC ↑</td></tr><tr><td>-</td><td>-</td><td>89.2</td><td>82.0</td><td>92.3</td></tr><tr><td>✓</td><td>-</td><td>95.4</td><td>92.0</td><td>96.9</td></tr><tr><td>-</td><td>✓</td><td>90.8</td><td>83.8</td><td>93.2</td></tr><tr><td>✓</td><td>✓</td><td>96.0</td><td>94.1</td><td>97.9</td></tr></table>

# 6 CONCLUSION

We propose to rethink the convention, of diffusion models for the unsupervised anomaly detection task, of noising all samples to the same time step and instead use prior information to dynamically adjust such implicit conditioning. Moreover we show that initial noising is counter productive and that a domain adapted feature extractor provides additional information for detection and localization. We introduced D3AD that combined all the proposed steps into an architecture which achieves state-of-the-art performance on the VisA benchmark with  $96\%$  I-AUROC and  $94.1\%$  PRO. Furthermore we showed that the segmentation performance measured by P-AUROC and PRO exceeds all previous suggested diffusion based models for unsupervised anomaly detection on VisA. A limitation of the framework is slower inference speed, which can potentially be addressed through innovations like precomputed features and more efficient approximations for anomaly severity, these are reserved for future work.

# REFERENCES

Mohiuddin Ahmed, Abdun Naser Mahmood, and Md. Rafiqul Islam. A survey of anomaly detection techniques in financial domain. Future Generation Computer Systems, 55:278-288, 2016. ISSN 0167-739X. doi: https://doi.org/10.1016/j_future.2015.01.001. URL https://www.sciencedirect.com/science/article/pii/S0167739X15000023.  
Samet Akcay, Amir Atapour-Abarghouei, and Toby P. Breckon. Ganomaly: Semi-supervised anomaly detection via adversarial training. In C. V. Jawahar, Hongdong Li, Greg Mori, and Konrad Schindler (eds.), Computer Vision - ACCV 2018, pp. 622-637, Cham, 2019. Springer International Publishing. ISBN 978-3-030-20893-6.  
Samet Akcay, Dick Ameln, Ashwin Vaidya, Barath Lakshmanan, Nilesh Ahuja, and Utku Genc. Anomalib: A deep learning library for anomaly detection, 2022.  
Christoph Baur, Benedikt Wiestler, Shadi Albarqouni, and Nassir Navab. Deep autoencoding models for unsupervised anomaly segmentation in brain mr images. In Alessandro Crimi, Spyridon Bakas, Hugo Kuijf, Farahani Keyvan, Mauricio Reyes, and Theo van Walsum (eds.), *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries*, pp. 161-169, Cham, 2019. Springer International Publishing. ISBN 978-3-030-11723-8.  
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad — a comprehensive real-world dataset for unsupervised anomaly detection. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9584–9592, 2019a. doi: 10.1109/CVPR.2019.00982.  
Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad-a comprehensive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9592-9600, 2019b.  
Paul Bergmann, Sindy Löwe, Michael Fauser, David Sattlegger, and Carsten Steger. Improving unsupervised defect segmentation by applying structural similarity to autoencoders. In Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications. SCITEPRESS - Science and Technology Publications, 2019c. doi: 10.5220/0007364503720380. URL https://doi.org/10.5220%2F0007364503720380.  
Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563-22575, 2023.  
Niv Cohen and Yedid Hoshen. Sub-image anomaly detection with deep pyramid correspondences. CoRR, abs/2005.02357, 2020. URL https://arxiv.org/abs/2005.02357.  
Thomas Defard, Aleksandr Setkov, Angelique Loesch, and Romaric Audigier. Padim: a patch distribution modeling framework for anomaly detection and localization. In International Conference on Pattern Recognition, pp. 475-489. Springer, 2021.  
H. Deng and X. Li. Anomaly detection via reverse distillation from one-class embedding. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9727-9736, Los Alamitos, CA, USA, jun 2022. IEEE Computer Society. doi: 10.1109/CVPR52688.2022.00951. URL https://doi.ieeeccomputersociety.org/10.1109/CVPR52688.2022.00951.  
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 8780-8794. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf.  
Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, and Anton van den Hengel. Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1705-1714, 2019.

Denis Gudovskiy, Shun Ishizaka, and Kazuki Kozuka. Cflow-ad: Real-time unsupervised anomaly detection with localization via conditional normalizing flows. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 98-107, 2022.  
Matthias Haselmann, Dieter P Gruber, and ul Tabatabai. Anomaly detection using deep learning based image completion. In 2018 17th IEEE international conference on machine learning and applications (ICMLA), pp. 1237-1242. IEEE, 2018.  
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL https://openreview.net/forum?id=qw8AKxfYbI.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840-6851, 2020.  
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and Yann LeCun (eds.), ICLR, 2014. URL http://dblp.uni-trier.de/db/conf/iclr/iclr2014.html#KingmaW13.  
Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=a-xFK8Ymz5J.  
Sungwook Lee, Seunghyun Lee, and Byung Cheol Song. Cfa: Coupled-hypersphere-based feature adaptation for target-oriented anomaly localization. IEEE Access, 10:78446-78454, 2022. doi: 10.1109/ACCESS.2022.3193699.  
Zhikang Liu, Yiming Zhou, Yuansheng Xu, and Zilei Wang. Simplenet: A simple network for image anomaly detection and localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20402-20411, June 2023.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.  
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=aBsCjcPu_tE.  
Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti. Vt-adj: A vision transformer network for image anomaly detection and localization. In 2021 IEEE 30th International Symposium on Industrial Electronics (ISIE), pp. 01-06. IEEE, 2021.  
Arian Mousakhan, Thomas Brox, and Jawad Tayyub. Anomaly detection with conditioned denoising diffusion models, 2023.  
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021.  
Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 16784-16804. PMLR, 2022. URL https://proceedings.mlr.press/v162/nichol22a.html.  
Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=FjNys5c7VvY.

Nicolae-Cătălin Ristea, Neelu Madan, Radu Tudor Ionescu, Kamal Nasrollahi, Fahad Shahbaz Khan, Thomas B Moeslund, and Mubarak Shah. Self-supervised predictive convolutional attentive block for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13576-13586, 2022.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684-10695, 2022.  
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, and Peter Gehler. Towards total recall in industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14318-14328, 2022.  
Marco Rudolph, Bastian Wandt, and Bodo Rosenhahn. Same same but differnet: Semi-supervised defect detection with normalizing flows. CoRR, abs/2008.12577, 2020. URL https:// arxiv.org/abs/2008.12577.  
Mohammad Sabokrou, Mohammad Khalooei, Mahmood Fathy, and Ehsan Adeli. Adversarily learned one-class classifier for novelty detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3379-3388, 2018.  
Thomas Schlegl, Philipp Seebock, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, and Georg Langs. Unsupervised anomaly detection with generative adversarial networks to guide marker discovery. CoRR, abs/1703.05921, 2017. URL http://arxiv.org/abs/1703.05921.  
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256-2265. PMLR, 2015.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022.  
Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran, Ta Duc Huy, Soan Duong, Chanh D Tr Nguyen, and Steven QH Truong. Revisiting reverse distillation for anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24511-24520, 2023.  
Julian Wyatt, Adam Leach, Sebastian M. Schmon, and Chris G. Willcocks. Anoddpm: Anomaly detection with denoising diffusion probabilistic models using simplex noise. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 649-655, 2022. doi: 10.1109/CVPRW56347.2022.00080.  
Jiawei Yu, Ye Zheng, Xiang Wang, Wei Li, Yushuang Wu, Rui Zhao, and Liwei Wu. Fastflow: Unsupervised anomaly detection and localization via 2d normalizing flows, 2021.  
Vitjan Zavrtanik, Matej Kristan, and Danijel Skocaj. *Dram - A discriminatively trained reconstruction embedding for surface anomaly detection*. *CoRR*, abs/2108.07610, 2021a. URL https://arxiv.org/abs/2108.07610.  
Vitjan Zavrtanik, Matej Kristan, and Danijel Skočaj. Reconstruction by inpainting for visual anomaly detection. Pattern Recognition, 112:107706, 2021b.  
Hui Zhang, Zheng Wang, Zuxuan Wu, and Yu-Gang Jiang. Diffusionad: Denoising diffusion for anomaly detection. arXiv preprint arXiv:2303.08730, 2023.  
Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. Spot-the-difference self-supervised pre-training for anomaly detection and segmentation. In ECCV 2022, 2022.
