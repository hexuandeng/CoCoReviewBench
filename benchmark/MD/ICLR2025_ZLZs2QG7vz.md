# NECOMIMI: NEURAL-COGNITIVE MULTIMODAL EEG-INFORMED IMAGE GENERATION WITH DIFFUSION MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

NECOMIMI (NEural-COgnitive MultImodal EEG-Informed Image Generation with Diffusion Models) introduces a novel framework for generating images directly from EEG signals using advanced diffusion models. Unlike previous works that focused solely on EEG-image classification through contrastive learning, NECOMIMI extends this task to image generation. The proposed NERV EEG encoder demonstrates state-of-the-art (SoTA) performance across multiple zero-shot classification tasks, including 2-way, 4-way, and 200-way, and achieves top results in our newly proposed CAT Score, which evaluates the quality of EEG-generated images based on semantic concepts. A key discovery of this work is that the model tends to generate abstract or generalized images, such as landscapes, rather than specific objects, highlighting the inherent challenges of translating noisy and low-resolution EEG data into detailed visual outputs. Additionally, we introduce the CAT Score as a new metric tailored for EEG-to-image evaluation and establish a benchmark on the ThingsEEG dataset. This study underscores the potential of EEG-to-image generation while revealing the complexities and challenges that remain in bridging neural activity with visual representation.

![](images/d25abb8d347e7137b8f8ea2e75e6beac6d483ee6fc69b8cb4b129be25a0fe824.jpg)  
Figure 1: This image demonstrates the capability of the NECOMIMI model to reconstruct images purely from EEG data without using the "Seen" images (ground truth) as embeddings during the generation process. The two-stage NECOMIMI architecture effectively extracts semantic information from noisy EEG signals, showing that it can capture and represent the underlying concepts from brainwave activity. The bottom row of images, generated solely from EEG input, highlights the potential of NECOMIMI to approximate the content of the "Seen" images in the top row, even in the absence of any direct visual reference or embedding.

# 1 INTRODUCTION

Electroencephalography (EEG) is one of the most ancient techniques used to measure neuronal activity in the human brain. Mary (1959); Millett (2001). Its application has significant value in clinical practice, particularly in diagnosing epilepsy. Reif et al. (2016), depression Li et al. (2023) and sleep disorders Hussain et al. (2022), as well as in assessing dysfunctions in sensory transmission pathways Thoma et al. (2003) and more Perrottelli et al. (2021). Historically, the analysis of EEG signals was limited to visual inspection of amplitude and frequency changes over time. However, with advancements in digital technology, the methodology has evolved significantly, shifting towards a more comprehensive analysis of the temporal and spatial characteristics of these signals.

(2016). As a result of this evolution, EEG has gained recognition as a potent tool for capturing brain functions in real-time, particularly in the sub-second range. Despite its advantages, EEG has traditionally suffered from poor spatial resolution, making it challenging to pinpoint the precise brain areas responsible for the measured neuronal activity at the scalp Li et al. (2022). In recent years, there has been a surge of interest in utilizing EEG for more sophisticated applications, such as image recognition and reconstruction Mai et al. (2023). These advancements have led to significant improvements in the accuracy of image recognition tasks, underscoring the potential of EEG as a bridge between neural activity and visual representation Spampinato et al. (2016); Kavasidis et al. (2017). The growing interest in using EEG for image recognition is rooted in its ability to capture the temporal dynamics of neuronal activity, though its spatial resolution remains a challenge. Innovative methodologies, including deep learning techniques and generative models like Generative Adversarial Networks (GANs) Goodfellow et al. (2014) and diffusion models Ho et al. (2020), have enhanced the accuracy and effectiveness of EEG-based systems, allowing for the generation of photorealistic images based on neural signals Kavasidis et al. (2017); Kumar et al. (2017); Singh et al. (2023). Notably, studies have demonstrated the feasibility of decoding natural images from EEG signals, employing innovative frameworks that align EEG responses with paired image stimuli Bai et al. (2023). However, most of the current works claiming to be EEG-to-image are essentially still imaged-to-image in nature, with EEG information primarily used to slightly guide the transformation of the input image by adding noise Kavasidis et al. (2017); Palazzo et al. (2017); Khare et al. (2022); Bai et al. (2023). In order to achieve a truly meaningful EEG-to-image generation, this work, named NECOMIMI (NEural-COgnitive MultImodal eeg-inforMed Image generation with diffusion models), introduces an innovative framework focused on EEG-based image generation, combining advanced diffusion model techniques.

This paper presents several key innovations as follows:

- We propose a novel EEG encoder, NERV, which achieves state-of-the-art performance in multimodal contrastive learning tasks.  
- Unlike previous work that primarily focused on image-to-image generation with EEG features as guidance, we introduce a comprehensive two-stage EEG-to-image multimodal generative framework. This not only extends prior contrastive learning between EEG and images but also applies it to image generation.  
- To address the conceptual differences between EEG-to-image and traditional text-to-image tasks, we propose a new quantification method, the Category-based Assessment Table (CAT) Score, which evaluates image generation performance based on semantic concepts rather than image distribution.  
- We establish a CAT score benchmark standard using Vision Language Model (VLM) on the ThingsEEG dataset.  
- Additionally, we uncover some notable findings and phenomena regarding the EEG-to-image generation process.

# 2 RELATED WORKS

# 2.1 THE POTENTIAL OF EEG DATA

In a typical experiment studying brain responses related to visual processes, a person looks at a series of images while a brain scanner or recording device captures their brain signals for analysis. There are various non-invasive methods to capture these brain responses, like fMRI, EEG, and MEG, each with different sensitivity levels. However, we still don't fully understand what this data really means, and even more importantly, how to interpret it. In a pioneering study Nishimoto et al. (2011), the researchers tried to generate impressions of what the subjects saw using fMRI images, based on a large image dataset taken from YouTube. However, this method has challenges, like the complexity and high cost of using an fMRI scanner. To overcome these drawbacks, a lot of research has shifted to using electrophysiological responses, particularly EEG, which has lower spatial resolution than most other methods but much higher temporal resolution. EEG recordings are also cheaper and easier to conduct, but the data is often noisy and affected by external factors, making it harder to reconstruct the original stimulus. Most image recognition and/or generation from brain signals nowadays is done using fMRI data Zhang et al. (2023), while EEG, being noisier, is used much less often.

# 2.2 USING EEG INFORMATION ON IMAGE GENERATION AND RECONSTRUCTION

Building on this shift towards EEG, prior to efforts in generating images directly from brain data, the concept of using EEG signals for image classification was introduced by the study Spampinato et al. (2017). This work first demonstrated the feasibility of decoding visual categories from EEG recordings using deep learning models, setting a foundation for leveraging neural signals in image-related tasks. However, the dataset they used was relatively small, which limited the generalization of their findings. Further advancements in generative models, specifically with the introduction of Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN), opened new possibilities for image generation. The VAE model proposed by Kingma & Welling (2013; 2019) achieved data generation and reconstruction by learning the latent distribution of data. The GAN model introduced by Goodfellow et al. (2014) utilized adversarial training between a generator and a discriminator to produce highly realistic images. Building on these methods, Brain2Image Kavasidis et al. (2017) was the first to use VAE to guide image generation from EEG features. Following that, EEG-GAN Palazzo et al. (2017) presented the first EEG-based image generation model, using LSTM Hochreiter & Schmidhuber (1997) to extract EEG information and guide the GAN for image generation. After this, there were still many EEG-to-image works based on GAN that emerged, with most of them focusing on improving the GAN architecture and the way it interacts with the EEG encoder, like in ThoughtViz Tirupattur et al. (2018), VG-GAN-VC Jiao et al. (2019), BrainMedia Fares et al. (2020), and EEG2IMAGE Singh et al. (2023), etc. However, in all these works, a common and challenging problem is figuring out how to effectively use EEG data to guide image generation and reconstruction. This challenge of training neural networks to align multimodal information wasn't effectively addressed until the emergence of CLIP Radford et al. (2021a), which provided a much better solution. Since then, some works have also applied this approach to EEG-based image generation.

# 2.3 CONTRASTIVE LEARNING-BASED WORKS ON EEG-IMAGE TASKS

To the best of our knowledge, EEGCLIP Singh et al. (2024) was the first to use contrastive learning to align EEG and image data. However, in this work, this aspect was only an exploratory attempt and did not further utilize the framework for downstream tasks like zero-shot image recognition. The next challenge lies in designing a better EEG encoder for contrastive learning, based on the rich image embeddings extracted from a CLIP-based image pre-trained encoder. Some recent works have explored this direction, such as NICE Song et al. (2024), MUSE Chen & Wei (2024), ATM Li et al. (2024), and Chen et al. (2024c). Some researchers have even attempted quantum-classical hybrid computing and quantum EEG encoder Chen et al. (2024a) to perform quantum contrastive learning Chen et al. (2024b). Most current works focus on tackling zero-shot classification, where the model is tested on unseen both EEG data and images that it hasn't encountered during training. The goal is to compute similarity scores for image recognition, aiming to enhance the model's generalization performance on out-of-sample data. As contrastive learning architectures for EEG-based image recognition mature, and inspired by test-to-image frameworks in other generative fields, the invention of diffusion models has addressed the instability issues associated with previous GAN-based generation methods to some extent. While there are already EEG-based image reconstruction efforts using diffusion models, such as NeuroVision Khare et al. (2022), DreamDiffusion Bai et al. (2023), DM-RE2I Zeng et al. (2023), BrainViz Fu et al. (2023), NeuroImagen Lan et al. (2023), and EEGVision Guo (2024), most of these works still largely rely on image-based features, with EEG data serving as supplementary information for the diffusion process. While these methods have made significant strides in computer vision, they primarily rely on images as input and are not designed to process non-visual signals like EEG directly. Currently, models designed specifically for direct generation tasks using pure EEG features or embeddings, where EEG functions similarly to a prompt command, are still quite rare. This work seeks to introduce a flexible, plug-and-play architecture: NECOMIMI, which not only expands upon previous recognition-focused approaches but also extends them into EEG-to-image generation tasks based on modern diffusion models.

![](images/af0e2724a23a29b0edb2c9018e2a0ec574c0789a6143bdd83f7816efb3d26ead.jpg)  
Based model training phase

![](images/1a01b2d54223d44d88f965484953bd5b2f9a32a5ca57e819210b68eec55130d4.jpg)  
Based model zero-shot testing phase

![](images/645d09ebce57dadba833bea53d26d5a0292cebe418408a7e09588656dd8d2f82.jpg)  
Based model one-stage image generation

![](images/5c8b16608d9b9cb6212249cac50610670596b553719dcd4d2a9a0225010f14cc.jpg)  
MLP Projector

![](images/8e83635636f6929038894302e3024633eb185489b338b6b0afdcff4abfa5d5dc.jpg)  
Based model two-stage image generation  
Figure 2: The figure illustrates the entire workflow of the EEG-based image generation model.

# 3 METHODOLOGY

# 3.1 OVERVIEW

This chapter provides a detailed overview of an advanced EEG-to-image generation model utilizing deep learning techniques and diffusion models. While the framework includes a one-stage image generation phase, we found that its performance was suboptimal. Consequently, the model is primarily designed as a two-stage process, which will be discussed in detail in later sections. The overall structure consists of four phases: the training phase, zero-shot testing, one-stage image generation, and two-stage image generation, each contributing to the transformation of raw EEG data into meaningful visual outputs.

# 3.2 TRAINING PHASE

In the initial training phase, both visual image  $\in \mathbb{R}^{h\times w\times ch}$  and EEG data  $\in \mathbb{R}^{e\times d}$  are processed in parallel to establish a shared embedding space, where  $h$  is the height of the image,  $w$  is the width of the image,  $ch$  is the number of channels (e.g., RGB channels),  $e$  is the number of electrodes (channels), and  $d$  is the number of data points (time samples). Training set images are first passed through a pre-trained image encoder, which transforms the images into latent representations called image embeddings I. In this work, we use a pretrained Vision Transformer (ViT) Dosovitskiy et al. (2020) from CLIP model Radford et al. (2021a) as the image encoder, which outputs embeddings of size  $\mathbb{R}^{1\times 10^{24}}$  for each image. Simultaneously, the EEG signals from the corresponding sessions are processed by a custom EEG encoder to produce EEG embeddings E. As for the EEG encoder, in this work, we extended several existing works like NICE Song et al. (2024), MUSE Chen & Wei (2024), Nervformer Chen & Wei (2024) and ATM Li et al. (2024) to enable EEG-to-image capabilities. Additionally, we proposed a new EEG encoder, NERV, which is specifically designed for noisy, multi-channel time series data like EEG, based on a multi-attention mechanism.

These embeddings are projected into a unified space via an MLP Projector, where they are trained using the InfoNCE loss. This contrastive learning loss function ensures that corresponding image and EEG embeddings are aligned in the latent space, enhancing the model's ability to understand and link neural patterns to visual stimuli. Standard contrastive learning employs the InfoNCE loss as defined

by Oord et al. (2018); He et al. (2020); Radford et al. (2021b):

$$
\mathcal {L} _ {\text {I n f o N C E}} = - \mathbb {E} \left[ \log \frac {\exp \left(S _ {\mathbf {E} , \mathbf {I}} / \tau\right)}{\sum_ {k = 1} ^ {N} \exp \left(S _ {\mathbf {E} , \mathbf {I} _ {k}} / \tau\right)} \right] \tag {1}
$$

where the  $S_{\mathbf{E},\mathbf{I}}$  represents the similarity score between the EEG embeddings  $\mathbf{E}$ , and the paired image embeddings  $\mathbf{I}$ , and the  $\tau$  is learned temperature parameter.

# 3.3 ZERO-SHOT TESTING PHASE

Once trained, the model enters the zero-shot testing phase. This phase focuses on evaluating the model's ability to generalize to unseen data. Here, the EEG signals and images from the test set are encoded using the pre-trained encoders, and their respective embeddings are projected through the MLP Projector. The testing groups are separated into multiple divisions—2-way, 4-way, 10-way, 50-way, 100-way and beyond—allowing for a structured comparison between the EEG and image embeddings. The final similarity scores between embeddings determine the model's classification accuracy, enabling the assessment of how well the model understands new EEG data without additional training.

# 3.4 ONE-STAGE IMAGE GENERATION

In the one-stage image generation process, the EEG embeddings from the testing set are directly used as inputs to reconstruct images. By incorporating the IP-Adapter Ye et al. (2023), which was originally designed to use images as prompts, due to its compact design, enhances image prompt flexibility within pre-trained text-to-image models. We adapt it in this work as a means to transform EEG embeddings into "feature prompts" for the image generation process. The conditioned embeddings are then processed by the Stable Diffusion XL-Turbo model Podell et al. (2023); Luo et al. (2024), a faster version of Stable Diffusion XL designed for rapid image synthesis, which reconstructs the final images based on the input EEG data. This method offers a streamlined approach to EEG-based image generation, relying on a single transformation stage to produce meaningful visual outputs from neural signals. The start of the EEG-conditioned diffusion phase is critical for generating images based on EEG data. This phase uses a classifier-free guidance method, which pairs CLIP embeddings and EEG embeddings  $(\mathbf{I},\mathbf{E})$ . By applying advanced generative techniques, the diffusion process is adapted to use the EEG embedding  $\mathbf{E}$  to model the distribution of the CLIP embeddings  $p(\mathbf{I}|\mathbf{E})$ . The CLIP embedding  $\mathbf{I}$ , generated during this stage, lays the foundation for the next phase of image generation. The architecture integrates a simplified U-Net model, represented as  $\epsilon_{\mathrm{prior}}(\mathbf{I}^t,t,\mathbf{E})$ , where  $\mathbf{I}^t$  is the noisy CLIP embedding at a specific diffusion step  $t$ .

The classifier-free guidance method helps refine the diffusion model (DM) using a specific EEG condition  $\mathbf{E}$ . This approach synchronizes the outputs of both a conditional and an unconditional model. The final model equation is expressed as:

$$
\epsilon_ {\text {p r i o r}} ^ {w} (\mathbf {I} ^ {t}, t, \mathbf {E}) = (1 + w) \epsilon_ {\text {p r i o r}} (\mathbf {I} ^ {t}, t, \mathbf {E}) - w \epsilon_ {\text {p r i o r}} (\mathbf {I} ^ {t}, t), \tag {2}
$$

where  $w \geq 0$  controls the guidance scale. This technique allows for training both the conditional and unconditional models within the same network, periodically replacing the EEG embedding  $\mathbf{E}$  with a null value to enhance training variation (about  $10\%$  of the data points). The main goal is to improve the quality of generated images while maintaining diversity.

However, we were surprised to find that when using EEG embeddings directly as prompts for the diffusion model, the generated images mostly turned out to be landscapes, regardless of the category. We will discuss the detailed results in later sections. As a result, we attempted a 2-stage approach for image generation.

# 3.5 TWO-STAGE IMAGE GENERATION

The prior diffusion stage plays a crucial role in generating an intermediate representation Zhu & Mumford (1997), such as a CLIP image embedding, from a text caption Ramesh et al. (2022). This representation is then used by the diffusion decoder to produce the final image. This two-stage

process enhances image diversity, maintains photorealism, and allows for efficient and controlled image generation Scotti et al. (2023). The two-stage image generation process introduces a more complex and refined method of synthesizing images from EEG data. In this approach, the EEG embeddings are first processed by a Diffusion U-Net, which applies additional transformations to enhance the representation of the neural data. After passing through the U-Net, the modified EEG embeddings are fed into the Stable Diffusion XL-Turbo model, with the assistance of the IP-Adaptor. This two-step transformation ensures a more nuanced generation process, potentially leading to higher-quality images by incorporating deeper layers of refinement. The first step of stage-1 is training the prior diffusion model. The main purpose of training is to let the model learn how to recover the original embedding from noisy embeddings. The specific steps are as follows: (a) Randomly replace conditional EEG embeddings  $c_{\mathrm{emb}}$  with None with a  $10\%$  probability:

$$
c _ {\text {e m b}} = \text {N o n e}, \quad \text {i f r a n d o m ()} <   0. 1 \tag {3}
$$

(b) Add random noise to the target embedding  $h_{\mathrm{emb}}$ , perturb it using the scheduler at a timestep  $t$ , use the symbol  $S_{add\_noise}$  to represent the scheduler add noise function:

$$
\hat {h} _ {\mathrm {e m b}} (t) = \mathcal {S} _ {\text {a d d n o i s e}} \left(h _ {\mathrm {e m b}}, \epsilon , t\right) \tag {4}
$$

where  $\epsilon \sim \mathcal{N}(0, I)$  is the random noise, and  $t$  is a randomly sampled timestep. (c) The model receives the perturbed embedding  $\hat{h}_{\mathrm{emb}}(t)$  and conditional embedding  $c_{\mathrm{emb}}$ , and predicts the noise. Use the symbol  $\mathcal{D}_{\mathrm{prior}}$  to represent the diffusion prior function:

$$
\epsilon_ {\text {p r e d}} = \mathcal {D} _ {\text {p r i o r}} \left(\hat {h} _ {\text {e m b}} (t), t, c _ {\text {e m b}}\right) \tag {5}
$$

(d) Compute the loss using Mean Squared Error (MSE) between the predicted noise and the actual noise:

$$
L = \frac {1}{N} \sum_ {i = 1} ^ {N} \left(\epsilon_ {\text {p r e d}} ^ {(i)} - \epsilon^ {(i)}\right) ^ {2} \tag {6}
$$

(e) Perform backpropagation on the loss  $L$ , and update the model parameters using the optimizer:

$$
\theta \leftarrow \theta - \eta \nabla_ {\theta} L \tag {7}
$$

where  $\eta$  is the learning rate and  $\theta$  represents the model's parameters.

The last step of stage-1 is generation process. The main purpose of the generation process is to gradually denoise and generate the final embedding based on the conditional EEG embedding  $c_{\mathrm{emb}}$ , starting from random noise. The specific steps are as follows: (a) Generate a sequence of timesteps  $t$ , which will be used for the denoising process, define  $\mathcal{T} = \{t_1, t_2, \dots, t_T\}$  to represent the set of time steps sampled from the total steps  $T$ :

$$
\left\{t _ {1}, t _ {2}, \dots , t _ {T} \right\} \sim \mathcal {T} (T) \tag {8}
$$

where  $T$  is the total number of denoising steps. (b) Initialize random noise embedding  $h_T$ , which serves as the starting point for the generation process:

$$
h _ {T} \sim \mathcal {N} (0, I) \tag {9}
$$

(c) Starting from timestep  $T$ , iteratively apply the model to predict noise and denoise the embedding until  $t = 0$ . Each step depends on the conditional embedding  $c_{\mathrm{emb}}$ :

If using conditional embedding, perform both unconditional and conditional noise prediction at each step:

$$
\epsilon_ {\text {p r e d} _ {\text {c o n d}}} = \mathcal {D} _ {\text {p r i o r}} \left(h _ {t}, t, c _ {\text {e m b}}\right) \tag {10}
$$

$$
\epsilon_ {\text {p r e d} \text {u n c o n d}} = \mathcal {D} _ {\text {p r i o r}} \left(h _ {t}, t\right) \tag {11}
$$

Then combine the results using classifier-free guidance, define  $\alpha_{\mathrm{guide}}$  as the guidance scale:

$$
\epsilon_ {\text {p r e d}} = \epsilon_ {\text {p r e d} _ {\text {u n c o n d}}} + \alpha_ {\text {g u i d e}} \times \left(\epsilon_ {\text {p r e d} _ {\text {c o n d}}} - \epsilon_ {\text {p r e d} _ {\text {u n c o n d}}}\right) \tag {12}
$$

Finally, update the noisy embedding based on the predicted noise, use the symbol  $S_{step}$  to represent the scheduler step function:

$$
h _ {t - 1} = \mathcal {S} _ {\text {s t e p}} \left(\epsilon_ {\text {p r e d}}, t, h _ {t}\right) \tag {13}
$$

(d) After the denoising process is complete,  $h_{output}$  represents the final generated embedding of a EEG, which is the model's output:

$$
h _ {\text {o u t p u t}} = h _ {\text {g e n e r a t e d}} \in \mathbb {R} ^ {1 \times 1 0 2 4} \tag {14}
$$

The stage-2 is input the  $h_{output}$  into the IP-adaptor as a prompt to generate the image by Stable Diffusion XL-Turbo model.

![](images/498373ab7368d61e7ad80cab564d08d8c970c8731706d235301c5f5a431f1fa8.jpg)  
Figure 3: This diagram shows the overall structure and workflow of the NERV EEG encoder model.

# 3.6 NERV EEG ENCODER

This diagram 3 illustrates the structure of NERV, a neural network encoder designed for EEG signal processing. The workflow starts with a linear projection of the flattened EEG nodes, followed by position encoding to retain temporal information. EEG signals pass through a Transformer layer and undergo instance normalization. The model then applies both spatial-temporal convolution (blue) to extract spatial features followed by temporal features and temporal-spatial convolution (yellow) to extract temporal features first, then spatial features. Multi-head self-attention mechanisms are applied to both feature sets, followed by layer normalization and residual connections. The cross-attention block (red) fuses the temporal and spatial features, which are further processed by a feed-forward layer before final output. The class token, position embeddings, and patch tokens are all part of the input sequence processed through these steps, ultimately yielding the output features for EEG-based tasks.

# 3.7 CATEGORY-BASED ASSESSMENT TABLE (CAT) SCORE

Unlike traditional image-to-image or text-to-image models driven by image representations, EEG-to-image models face unique challenges. In the current NECOMIMI architecture, the model can only capture broad semantic information from EEG signals rather than fine-grained details. For example, suppose the ground truth EEG data was recorded while a subject was observing an aircraft carrier. When using Model A as the EEG encoder in NECOMIMI, the generated image is a jet, while using Model B results in an image of a sheep. To objectively assess performance, we need a standard that scores Model A higher than Model B in such cases.

Why not use existing evaluation metrics? Traditional metrics like Structural Similarity Index (SSIM) Wang et al. (2004) measure structural similarity between the ground truth and generated image, while the Inception Score (IS) Salimans et al. (2016) and Fréchet Inception Distance (FID) Heusel et al. (2017) focus on the accuracy of image categories and its distribution. However, EEG captures more abstract semantic information, and we cannot guarantee that the subject's thoughts during EEG recording perfectly align with the ground truth image. This makes traditional evaluation methods unfair for EEG-to-image tasks.

To address this, we propose the Category-Based Assessment Table (CAT) Score, a new metric specifically designed for EEG-to-image evaluation. In the ThingsEEG test dataset (which contains 200 categories with one image per category), each image is manually labeled with two tags for broad

categories, one for a specific category, and one for background content, resulting in a total of five tags per image. We extracted the tags by ChatGPT-4o OpenAI et al. (2023). The entire test dataset thus comprises 200 images  $\times 5$  tags  $= 1,000$  points. Using manual annotation, we can determine whether the categories of generated images match these labels, providing a fair assessment for EEG-to-image models. For more details on the ThingsEEG categories, please refer to the appendix.

# 4 EXPERIMENTS

# 4.1 DATASETS AND PREPROCESSING

The ThingsEEG dataset Gifford et al. (2022) consists of a large set of EEG recordings obtained through a rapid serial visual presentation (RSVP) paradigm. The responses were collected from 10 participants who viewed a total of 16,740 natural images from the THINGS database Hebart et al. (2019). The dataset contains 1654 training categories, each with 10 images, and 200 test categories, each with a single image. The EEG data were recorded using 64-channel EASYCAP equipment, and preprocessing involved segmenting the data into trials from 0 to  $1000\mathrm{ms}$  after the stimulus was shown, with baseline correction based on the pre-stimulus period. EEG responses for each image were averaged over multiple repetitions.

# 4.2 EXPERIMENT DETAILS

Due to the significant impact that different versions of the CLIP package can have on the results of contrastive learning, this work ensures a fair comparison of various EEG encoders by rerunning all experiments using a unified CLIP-ViT environment, where available open-source code (e.g., Song et al.  $(2024)^{1}$ , Chen & Wei  $(2024)^{2}$ , Li et al.  $(2024)^{3}$ ) was utilized. Another factor that can influence contrastive learning is batch size. Therefore, all experiments in this work were conducted with a batch size of 1024. The final results are averaged from the best outcomes of 5 random seed training sessions, each running for 200 epochs. We employ the AdamW optimizer, setting the learning rate to 0.0002 and parameters  $\beta_{1} = 0.5$  and  $\beta_{2} = 0.999$ . The  $\tau$  in contrastive learning initialized with  $log(1 / 0.07)$ . The NERV model achieves the best results with 5 multi-heads, while the Transformer layer has 1 multi-head and the cross-attention layer has 8 multi-heads. The time step is 50 in diffusion model. All experiments, including both EEG encoder training and prior diffusion model processing, were performed on a machine equipped with an A100 GPU.

# 4.3 CLASSIFICATION RESULTS

In Table 1, the classification accuracy for both 2-way and 4-way zero-shot tasks is evaluated across ten subjects. Our new model NERV consistently achieves the best performance, particularly excelling in the 2-way classification task, where it maintains top accuracy across most subjects. It achieves an average accuracy of  $94.8\%$  in the 2-way classification and  $86.8\%$  in the 4-way classification, outperforming other methods like NICE Song et al. (2024), MUSE Chen & Wei (2024), and ATM-S Li et al. (2024). While NICE and MUSE perform strongly in some subjects, they often fall short of NERV's performance. NICE has an average of  $91.3\%$  in the 2-way task and  $81.3\%$  in the 4-way task, with MUSE trailing behind with averages of  $92.2\%$  (2-way) and  $82.8\%$  (4-way). ATM-S performs comparably to NICE and MUSE in some subjects but falls short on average with  $86.5\%$  in the 4-way classification. In Table 2, the results for the more challenging 200-way zero-shot classification task show that NERV also performs the best, especially in the top-1 accuracy. ATM-S and NERV perform similarly, but NERV shows stronger performance in most subjects. NERV achieves an average top-1 accuracy of  $27.9\%$  and top-5 accuracy of  $54.7\%$ , leading over all other methods. In contrast, Nervformer Chen & Wei (2024) and BraVL Du et al. (2023) show weaker performance, especially in the top-1 accuracy, where they average  $19.8\%$  and  $5.8\%$ , respectively. For the results of other 10-way, 50-way, and 100-way zero-shot classifications, please refer to the appendix. In summary, NERV consistently outperforms its competitors in both tasks, demonstrating the strongest zero-shot

classification capability, particularly when distinguishing between a large number of categories, making it the most effective model in these experiments.

Table 1: Overall accuracy (%) of 2-way and 4-way zero-shot classification using CLIP-ViT as image encoder: top-1 and top-5. The parts in bold represent the best results, while the underlined parts are the second best.  

<table><tr><td rowspan="2">Method</td><td colspan="2">Subject 1</td><td colspan="2">Subject 2</td><td colspan="2">Subject 3</td><td colspan="2">Subject 4</td><td colspan="2">Subject 5</td><td colspan="2">Subject 6</td><td colspan="2">Subject 7</td><td colspan="2">Subject 8</td><td colspan="2">Subject 9</td><td colspan="2">Subject 10</td><td colspan="2">Ave</td></tr><tr><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td><td>2-way</td><td>4-way</td></tr><tr><td colspan="23">Subject dependent - train and test on one subject</td></tr><tr><td>Nervformer</td><td>89.9</td><td>76.9</td><td>91.3</td><td>80.7</td><td>91.6</td><td>80.8</td><td>94.3</td><td>85.9</td><td>86.3</td><td>70.4</td><td>91.1</td><td>82.5</td><td>92.5</td><td>81.6</td><td>96.2</td><td>88.3</td><td>92.0</td><td>83.7</td><td>92.4</td><td>83.1</td><td>91.8</td><td>81.4</td></tr><tr><td>NICE</td><td>91.7</td><td>80.4</td><td>89.8</td><td>77.4</td><td>93.5</td><td>83.7</td><td>94.0</td><td>84.9</td><td>85.9</td><td>70.3</td><td>89.1</td><td>81.7</td><td>91.2</td><td>81.7</td><td>95.8</td><td>89.2</td><td>87.9</td><td>76.5</td><td>93.8</td><td>87.1</td><td>91.3</td><td>81.3</td></tr><tr><td>MUSE</td><td>90.1</td><td>78.4</td><td>90.3</td><td>76.8</td><td>93.4</td><td>85.6</td><td>93.6</td><td>87.5</td><td>88.3</td><td>74.2</td><td>93.1</td><td>85.3</td><td>93.1</td><td>82.8</td><td>95.4</td><td>87.7</td><td>90.5</td><td>81.8</td><td>94.4</td><td>88.1</td><td>92.2</td><td>82.8</td></tr><tr><td>ATM-S</td><td>94.8</td><td>84.9</td><td>93.5</td><td>86.3</td><td>95.3</td><td>89.0</td><td>95.9</td><td>87.3</td><td>90.8</td><td>78.5</td><td>94.1</td><td>85.2</td><td>94.2</td><td>87.1</td><td>96.6</td><td>92.9</td><td>94.1</td><td>86.8</td><td>94.7</td><td>87.0</td><td>94.4</td><td>86.5</td></tr><tr><td>NERV (ours)</td><td>95.3</td><td>85.7</td><td>96.0</td><td>88.8</td><td>95.9</td><td>91.2</td><td>95.8</td><td>87.4</td><td>90.8</td><td>80.4</td><td>93.6</td><td>84.0</td><td>94.7</td><td>86.2</td><td>96.8</td><td>92.3</td><td>94.4</td><td>84.2</td><td>94.8</td><td>87.6</td><td>94.8</td><td>86.8</td></tr></table>

Table 2: Overall accuracy (%) of 200-way zero-shot classification using CLIP-ViT as image encoder: top-1 and top-5. The parts in bold represent the best results, while the underlined parts are the second best.  

<table><tr><td rowspan="2">Method</td><td colspan="2">Subject 1</td><td colspan="2">Subject 2</td><td colspan="2">Subject 3</td><td colspan="2">Subject 4</td><td colspan="2">Subject 5</td><td colspan="2">Subject 6</td><td colspan="2">Subject 7</td><td colspan="2">Subject 8</td><td colspan="2">Subject 9</td><td colspan="2">Subject 10</td><td colspan="2">Ave</td></tr><tr><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td><td>top-1</td><td>top-5</td></tr><tr><td colspan="23">Subject dependent - train and test on one subject</td></tr><tr><td>BraVL</td><td>6.1</td><td>17.9</td><td>4.9</td><td>14.9</td><td>5.6</td><td>17.4</td><td>5.0</td><td>15.1</td><td>4.0</td><td>13.4</td><td>6.0</td><td>18.2</td><td>6.5</td><td>20.4</td><td>8.8</td><td>23.7</td><td>4.3</td><td>14.0</td><td>7.0</td><td>19.7</td><td>5.8</td><td>17.5</td></tr><tr><td>Nerformer</td><td>15.0</td><td>36.7</td><td>15.6</td><td>40.0</td><td>19.7</td><td>44.9</td><td>23.3</td><td>54.4</td><td>13.0</td><td>29.1</td><td>18.9</td><td>42.2</td><td>19.5</td><td>42.0</td><td>30.3</td><td>60.0</td><td>20.1</td><td>46.3</td><td>22.9</td><td>47.1</td><td>19.8</td><td>44.3</td></tr><tr><td>NICE</td><td>19.3</td><td>44.8</td><td>15.2</td><td>38.2</td><td>23.9</td><td>51.4</td><td>24.1</td><td>51.6</td><td>11.0</td><td>30.7</td><td>18.5</td><td>43.8</td><td>21.0</td><td>47.9</td><td>32.5</td><td>63.5</td><td>18.2</td><td>42.4</td><td>27.4</td><td>57.1</td><td>21.1</td><td>47.1</td></tr><tr><td>MUSE</td><td>19.8</td><td>41.1</td><td>15.3</td><td>34.2</td><td>24.7</td><td>52.6</td><td>24.7</td><td>52.6</td><td>12.1</td><td>33.7</td><td>22.1</td><td>51.9</td><td>21.0</td><td>48.6</td><td>33.2</td><td>59.9</td><td>19.1</td><td>43.0</td><td>25.0</td><td>55.2</td><td>21.7</td><td>47.3</td></tr><tr><td>ATM-S</td><td>25.8</td><td>54.1</td><td>24.6</td><td>52.6</td><td>28.4</td><td>62.9</td><td>25.9</td><td>57.8</td><td>16.2</td><td>41.9</td><td>21.2</td><td>53.0</td><td>25.9</td><td>57.2</td><td>37.9</td><td>71.1</td><td>26.0</td><td>53.9</td><td>30.0</td><td>60.9</td><td>26.2</td><td>56.5</td></tr><tr><td>NERV (ours)</td><td>25.4</td><td>51.2</td><td>24.1</td><td>51.1</td><td>28.6</td><td>53.9</td><td>30.0</td><td>58.4</td><td>19.3</td><td>43.9</td><td>24.9</td><td>52.3</td><td>26.1</td><td>51.6</td><td>40.8</td><td>67.4</td><td>27.0</td><td>55.2</td><td>32.3</td><td>61.6</td><td>27.9</td><td>54.7</td></tr></table>

# 4.4 PERFORMANCE COMPARISON OF DIFFERENT GENERATIVE MODELS

Here, we introduce our newly proposed CAT Score method, which quantifies and evaluates the quality of EEG-generated images based on semantic concepts rather than pixel structure. Detailed CAT Score labels can be found in the appendix. To our surprise, while our proposed NERV method achieved SoTA on the CAT Score, no EEG encoder has surpassed a score of 500 in this evaluation out of a possible 1000 points. This highlights both the rigor of the CAT Score and the challenging nature of the pure EEG-to-Image task.

Table 3: Overall CAT score  $\times  {1000}$  of NECOMIMI EEG-to-Image generation with several EEG encoders.  

<table><tr><td></td><td>Subject 1</td><td>Subject 2</td><td>Subject 3</td><td>Subject 4</td><td>Subject 5</td><td>Subject 6</td><td>Subject 7</td><td>Subject 8</td><td>Subject 9</td><td>Subject 10</td><td>Ave</td></tr><tr><td>EEG Encoder</td><td></td><td></td><td></td><td></td><td></td><td>CAT Score</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Nervformer</td><td>432</td><td>457</td><td>429</td><td>454</td><td>475</td><td>463</td><td>404</td><td>438</td><td>427</td><td>410</td><td>438.9</td></tr><tr><td>NICE</td><td>426</td><td>456</td><td>445</td><td>447</td><td>411</td><td>454</td><td>438</td><td>443</td><td>426</td><td>429</td><td>437.5</td></tr><tr><td>MUSE</td><td>438</td><td>456</td><td>434</td><td>416</td><td>426</td><td>463</td><td>443</td><td>437</td><td>410</td><td>468</td><td>439.1</td></tr><tr><td>ATM-S</td><td>413</td><td>419</td><td>411</td><td>464</td><td>427</td><td>469</td><td>442</td><td>472</td><td>431</td><td>445</td><td>439.3</td></tr><tr><td>NERV (ours)</td><td>445</td><td>436</td><td>432</td><td>456</td><td>438</td><td>466</td><td>410</td><td>437</td><td>433</td><td>444</td><td>439.7</td></tr></table>

# 4.5 FINDINGS IN EEG-TO-IMAGE

We have observed some interesting findings from the pure EEG-to-Image process. As shown in the third row of Figure 4, the images generated by the diffusion model from embeddings compressed from EEG signals mainly consist of landscapes, which differ significantly from the original images (ground truth). Several factors may contribute to this phenomenon. For example, EEG signals are a high-noise, low-resolution form of data, capturing only certain aspects of brain activity. Moreover, we are currently unable to assess whether the brainwave data recorded from the subjects accurately captures the complete information of the original images, as the subjects might have been distracted and thinking about other things during the recording. This makes it difficult for the embeddings extracted from EEG signals to capture sufficient details, particularly when it comes to high-resolution object recognition (such as cats or specific items). As a result, the model tends to generate relatively vague or abstract images, like landscapes. Alternatively, the EEG signals may reflect higher-level abstract concepts or emotions associated with viewing the images rather than concrete objects or

![](images/8bc0a05cc3405905a1321b73c13e54b764191cca378441a64c7f3c85be1a64e3.jpg)  
Figure 4: The image illustrates the progression of visual representations generated using different embedding techniques in a diffusion model: (a) Top row: The original images shown to subjects (ground truth). (b) Second row: Images generated by the CLIP-ViT embeddings of the original images. (c) Third row: Images generated by one-stage method using pure EEG embeddings with NERV EEG encoder. (d) Fourth row: Images generated by two-stage NECOMIMI method using pure EEG embeddings with NERV EEG encoder.

details. Since these abstract concepts are often related to the scene, background, or the brain's broad perception of the environment, the model is more likely to generate abstract or general images, such as landscapes, instead of specific objects.

Additionally, the training of the model on EEG signals may still be insufficient. The diffusion model may not yet fully understand and generate images from EEG signals, especially when it lacks enough data or optimization to map EEG signals to specific visual information. As a result, the model might more easily generate the types of images it is "accustomed" to producing, such as landscapes, which may constitute a significant portion of the training data. The gap between the vision modality and the neural modality (EEG) is also substantial. EEG signals may not directly correspond to detailed objects in images, so the model tends to generate "safe options," like landscapes, which may have been more prevalent in the image generation samples during training. This leads to what can be described as "hallucinations." These factors collectively contribute to the significant differences between the images generated from EEG signals and the ground truth, particularly the failure in specific object recognition. This work can be considered a forward-looking exploration, as this field is just beginning to develop.

# 5 DISCUSSION AND CONCLUSION

The NECOMIMI framework expands previous works on EEG-Image contrastive learning classification by enabling image generation, filling a gap in prior research and opening new possibilities for EEG applications. We introduced the SoTA EEG encoder, NERV, which achieved top performance in 2-way, 4-way, and 200-way zero-shot classification tasks, as well as in the CAT Score evaluation, demonstrating its effectiveness in EEG-based generative tasks. A key finding is that the model often generates abstract images, like landscapes, rather than specific objects. This suggests that EEG data, being noisy and low-resolution, captures broad semantic concepts rather than detailed visuals. The gap between neural signals and visual stimuli remains a challenge for precise image generation. We also proposed the CAT Score, a new metric tailored for EEG-to-image generation, and established its benchmark on the ThingsEEG dataset. Surprisingly, we found that EEG encoder performance may not strongly correlate with the quality of generated images, providing new insights into the limitations and challenges of this task. In conclusion, NECOMIMI demonstrates the potential of EEG-to-image generation while highlighting the complexities of translating neural signals into accurate visual representations. Future research should focus on refining models to better capture detailed information from EEG signals.

# REFERENCES

Yunpeng Bai, Xintao Wang, Yan-pei Cao, Yixiao Ge, Chun Yuan, and Ying Shan. Dreamdiffusion: Generating high-quality images from brain eeg signals, 2023. URL https://arxiv.org/abs/2306.16934.  
Chi-Sheng Chen and Chun-Shu Wei. Mind's eye: Image recognition by eeg via multimodal similarity-keeping contrastive learning, 2024. URL https://arxiv.org/abs/2406.16910.  
Chi-Sheng Chen, Samuel Yen-Chi Chen, Aidan Hung-Wen Tsai, and Chun-Shu Wei. Qeegnet: Quantum machine learning for enhanced electroencephalography encoding, 2024a. URL https://arxiv.org/abs/2407.19214.  
Chi-Sheng Chen, Aidan Hung-Wen Tsai, and Sheng-Chieh Huang. Quantum multimodal contrastive learning framework, 2024b. URL https://arxiv.org/abs/2408.13919.  
Hongzhou Chen, Lianghua He, Yihang Liu, and Longzhen Yang. Visual neural decoding via improved visual-eeg semantic consistency, 2024c. URL https://arxiv.org/abs/2408.06788.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2020. URL https://arxiv.org/abs/2010.11929.  
Changde Du, Kaicheng Fu, Jinpeng Li, and Huiguang He. Decoding visual neural representations by multimodal learning of brain-visual-linguistic features. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.  
Louis EK;Frey. Electroencephalography (eeg): An introductory text and atlas of normal and abnormal findings in adults, children, and infants [internet], 2016. URL https://pubmed.ncbi.nlm.nih.gov/27748095/.  
Ahmed Fares, Sheng-hua Zhong, and Jianmin Jiang. Brain-media: A dual conditioned and lateralization supported gan (dcls-gan) towards visualization of image-evoked brain activities. In Proceedings of the 28th ACM International Conference on Multimedia, MM '20, pp. 1764-1772, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450379885. doi: 10.1145/3394171.3413858. URL https://doi.org/10.1145/3394171.3413858.  
Honghao Fu, Zhiqi Shen, Jing Jih Chin, and Hao Wang. Brainvis: Exploring the bridge between brain and visual signals via image reconstruction, 2023. URL https://arxiv.org/abs/2312.14871.  
Alessandro T Gifford, Kshitij Dwivedi, Gemma Roig, and Radoslaw M Cichy. A large and rich eeg dataset for modeling human visual object recognition. NeuroImage, 264:119754, 2022.  
Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014. URL https://arxiv.org/abs/1406.2661.  
Huangtao Guo. Eegvision: Reconstructing vision from human brain signals. Applied Mathematics and Nonlinear Sciences, 9(1), Jan 2024. doi: https://doi.org/10.2478/amns-2024-1856. URL https://sciendo.com/article/10.2478/amns-2024-1856.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729-9738, 2020.  
Martin N Hebart, Adam H Dickter, Alexis Kidder, Wan Y Kwok, Anna Corriveau, Caitlin Van Wicklin, and Chris I Baker. Things: A database of 1,854 object concepts and more than 26,000 naturalistic object images. *PloS one*, 14(10):e0223792, 2019.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. URL https://arxiv.org/abs/2006.11239.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural Comput.*, 9(8): 1735-1780, nov 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL https://doi.org/10.1162/neco.1997.9.8.1735.  
I. Hussain, Md. Azam Hossain, Rafsan Jany, Md. Azam Hossain, M. Uddin, A. Kamal, Y. Ku, and Jik-Soo Kim. Quantitative evaluation of eeg-biomarkers for prediction of sleep stages. Sensors (Basel, Switzerland), 22, 2022. doi: 10.3390/s22083079.  
Zhicheng Jiao, Haoxuan You, Fan Yang, Xin Li, Han Zhang, and Dinggang Shen. Decoding eeg by visual-guided deep neural networks. *Ijcai*.org, pp. 1387-1393, 2019. URL https://www.ijcai.org/proceedings/2019/192.  
Isaak Kavasidis, Simone Palazzo, Concetto Spampinato, Daniela Giordano, and Mubarak Shah. Brain2image: Converting brain signals into images. In Proceedings of the 25th ACM International Conference on Multimedia, MM '17, pp. 1809-1817, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450349062. doi: 10.1145/3123266.3127907. URL https://doi.org/10.1145/3123266.3127907.  
Sanchita Khare, Rajiv Nayan Choubey, Loveleen Amar, and Venkanna Udutalapalli. Neurovision: perceived image regeneration using cprogan. Neural Computing and Applications, 34 (8):5979-5991, Jan 2022. doi: https://doi.org/10.1007/s00521-021-06774-1. URL https://link.springer.com/article/10.1007/s00521-021-06774-1.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2013. URL https://arxiv.org/abs/1312.6114.  
Diederik P Kingma and Max Welling. An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4):307-392, Jan 2019. doi: https://doi.org/10.1561/2200000056. URL https://arxiv.org/abs/1906.02691.  
Pradeep Kumar, Rajkumar Saini, Partha Pratim Roy, Pawan Kumar Sahu, and Debi Prosad Dogra. Envisioned speech recognition using eeg sensors. Personal and Ubiquitous Computing, 22(1): 185-199, Sep 2017. doi: https://doi.org/10.1007/s00779-017-1083-4. URL https://link.springer.com/article/10.1007/s00779-017-1083-4.  
Yu-Ting Lan, Kan Ren, Yansen Wang, Wei-Long Zheng, Dongsheng Li, Bao-Liang Lu, and Lili Qiu. Seeing through the brain: Image reconstruction of visual perception from human brain signals, 2023. URL https://arxiv.org/abs/2308.02510.  
Cheng-Ta Li, Chi-Sheng Chen, Chih-Ming Cheng, Chung-Ping Chen, Jen-Ping Chen, Mu-Hong Chen, Ya-Mei Bai, and Shih-Jen Tsai. Prediction of antidepressant responses to non-invasive brain stimulation using frontal electroencephalogram signals: Cross-dataset comparisons and validation. Journal of Affective Disorders, 343:86-95, Dec 2023. doi: https://doi.org/10.1016/j.jad.2023.08.059. URL https://www.sciencedirect.com/science/article/abs/pii/S0165032723010388.  
Dongyang Li, Chen Wei, Shiying Li, Jiachen Zou, and Quanying Liu. Visual decoding and reconstruction via eeg embeddings with guided diffusion, 2024. URL https://arxiv.org/abs/2403.07721.  
Rihui Li, Dalin Yang, Feng Fang, K. Hong, A. Reiss, and Yingchun Zhang. Concurrent fnirs and eeg for brain function investigation: A systematic, methodology-focused review. Sensors (Basel, Switzerland), 22, 2022. doi: 10.3390/s22155865.

Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module, 2024. URL https://arxiv.org/abs/2311.05556.  
Weijian Mai, Jian Zhang, Pengfei Fang, and Zhijun Zhang. Brain-conditional multimodal synthesis: A survey and taxonomy, 2023. URL https://arxiv.org/abs/2401.00430.  
Mary. The eeg in epilepsy a historical note. *Epilepsia*, 1(1-5):328–336, Jan 1959. doi: https://doi.org/10.1111/j.1528-1157.1959.tb04270.x. URL https://onlinelibrary.wiley.com/doi/10.1111/j.1528-1157.1959.tb04270.x.  
David Millett. Hans berger: From psychic energy to the eeg. Perspectives in Biology and Medicine, 44(4):522-542, Sep 2001. doi: https://doi.org/10.1353/pbm.2001.0070. URL https://muse.jhu.edu/article/26086.  
Shinji Nishimoto, An T. Vu, Thomas Naselaris, Yuval Benjamini, Bin Yu, and Jack L. Gallant. Reconstructing visual experiences from brain activity evoked by natural movies. Current Biology, 21(19): 1641-1646, 2011. ISSN 0960-9822. doi: https://doi.org/10.1016/j.cub.2011.08.031. URL https://www.sciencedirect.com/science/article/pii/S0960982211009377.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgium, and Irwan Bello. Gpt-4 technical report, 2023. URL https://arxiv.org/abs/2303.08774.  
S. Palazzo, C. Spampinato, I. Kavasidis, D. Giordano, and M. Shah. Generative adversarial networks conditioned by brain signals. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 3430-3438, 2017. doi: 10.1109/ICCV.2017.369.  
A. Perrottelli, G. Giordano, F. Brando, L. Giuliani, and A. Mucci. Eeg-based measures in at-risk mental state and early stages of schizophrenia: A systematic review. Frontiers in Psychiatry, 12, 2021. doi: 10.3389/fpsyt.2021.653642.  
Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. URL https://arxiv.org/abs/2307.01952.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021a. URL https://arxiv.org/abs/2103.00020.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748-8763. PMLR, 2021b.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. URL https://arxiv.org/abs/2204.06125.  
Philipp S Reif, Adam Strzelczyk, and Felix Rosenow. The history of invasive eeg evaluation in epilepsy patients. *Seizure*, 41:191–195, Apr 2016. doi: https://doi.org/10.1016/j.seizure.2016.04.006. URL https://www.seizure-journal.com/article/S1059-1311(16)30022-X/fulltext.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans, 2016. URL https://arxiv.org/abs/1606.03498.

Paul S Scotti, Atmadeep Banerjee, Jimmie Goode, Stepan Shabalin, Alex Nguyen, Ethan Cohen, Aidan J Dempster, Nathalie Verlinde, Elad Yundler, David Weisberg, Kenneth A Norman, and Tanishq Mathew Abraham. Reconstructing the mind's eye: fmri-to-image with contrastive learning and diffusion priors, 2023. URL https://arxiv.org/abs/2305.18274.  
P. Singh, D. Dalal, G. Vashishtha, K. Miyapuram, and S. Raman. Learning robust deep visual representations from eeg brain recordings. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 7538-7547, Los Alamitos, CA, USA, jan 2024. IEEE Computer Society. doi: 10.1109/WACV57701.2024.00738. URL https://doi.ieeeccomputersociety.org/10.1109/WACV57701.2024.00738.  
Prajwal Singh, Pankaj Pandey, Krishna Miyapuram, and Shanmuganathan Raman. Eeg2image: Image reconstruction from eeg brain signals, 2023. URL https://arxiv.org/abs/2302.10121.  
Yonghao Song, Bingchuan Liu, Xiang Li, Nanlin Shi, Yijun Wang, and Xiaorong Gao. Decoding natural images from eeg for object recognition, 2024. URL https://arxiv.org/abs/2308.13234.  
C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, and M. Shah. Deep learning human mind for automated visual classification. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4503-4511, 2017. doi: 10.1109/CVPR.2017.479.  
Concetto Spampinato, Simone Palazzo, Isaak Kavasidis, Daniela Giordano, Mubarak Shah, and Nasim Souly. Deep learning human mind for automated visual classification, 2016. URL https://arxiv.org/abs/1609.00344.  
R. Thoma, F. Hanlon, S. Moses, J. Christopher Edgar, Mingxiong Huang, M. Weisend, J. Irwin, A. Sherwood, K. Paulson, J. Bustillo, L. Adler, Gregory A. Miller, and J. Canive. Lateralization of auditory sensory gating and neuropsychological dysfunction in schizophrenia. The American journal of psychiatry, 160 9:1595-605, 2003. doi: 10.1176/APPI.AJP.160.9.1595.  
Praveen Tirupattur, Yogesh Singh Rawat, Concetto Spampinato, and Mubarak Shah. Thoughtviz: Visualizing human thoughts using generative adversarial network. In Proceedings of the 26th ACM International Conference on Multimedia, MM '18, pp. 950-958, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356657. doi: 10.1145/3240508.3240641. URL https://doi.org/10.1145/3240508.3240641.  
Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600-612, 2004. doi: 10.1109/TIP.2003.819861.  
Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models, 2023. URL https://arxiv.org/abs/2308.06721.  
Hong Zeng, Nianzhang Xia, Dongguan Qian, Motonobu Hattori, Chu Wang, and Wanzeng Kong. Dmre2i: A framework based on diffusion model for the reconstruction from eeg to image. Biomedical Signal Processing and Control, 86:105125-105125, Sep 2023. doi: https://doi.org/10.1016/j.bspc.2023.105125. URL https://www.sciencedirect.com/science/article/abs/pii/S174680942300558X?via%3Dihub.  
Chenshuang Zhang, Chaoning Zhang, Mengchun Zhang, and In So Kweon. Text-to-image diffusion models in generative ai: A survey, 2023. URL https://arxiv.org/abs/2303.07909.  
Song Chun Zhu and D. Mumford. Prior learning and gibbs reaction-diffusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(11):1236-1250, 1997. doi: 10.1109/34.632983.
