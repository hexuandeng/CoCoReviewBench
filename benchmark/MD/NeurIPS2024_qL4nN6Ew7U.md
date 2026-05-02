# Fantasy: Transformer Meets Transformer in Text-to-Image Generation

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We present Fantasy, an efficient text-to-image generation model marrying the decoder-only Large Language Models (LLMs) and transformer-based masked image modeling (MIM). While diffusion models are currently in a leading position in this task, we demonstrate that with appropriate training strategies and high-quality data, MIM can also achieve comparable performance. By incorporating pre-trained decoder-only LLMs as the text encoder, we observe a significant improvement in text fidelity compared to the widely used CLIP text encoder, enhancing the text-image alignment. Our training approach involves two stages: 1) large-scale concept alignment pre-training, and 2) fine-tuning with high-quality instruction-image data. Evaluations on FID, HPSv2 benchmarks, and human feedback demonstrate the competitive performance of Fantasy against state-of-the-art diffusion and autoregressive models.

# 1 Introduction

Recent advances in text-to-image (T2I) models [3, 5, 12] have become focal points within the computer vision field. Most advances in T2I models, focused on generating high-quality images based on relatively short descriptions, struggle with intricate long-text semantic alignment due to inherent structure constraints and data limitations. Text encoders used for T2I fall into three categories: CLIP [30], encoder-decoder LLMs, and decoder-only LLMs. Models using encoder-decoder LLMs like T5-XXL [31] have shown improved text-image alignment over CLIP by exploiting enhanced text understanding, increasing token capacity, yet without delving into the semantic alignment for longer texts. ParaDiffusion [43] indicates that directly aligning text embeddings with visual features without prior image-text knowledge is not the most effective approach. Previous works [38, 45] have highlighted shortcomings in existing text-image datasets [37], including image-text mismatches, a lack of informative content, and a pronounced long-tail effect. These deficiencies notably impair training efficiency for T2I models and restrict their ability to learn complex semantic alignment.

Existing diffusion-based T2I models [33, 5, 9, 26] have achieved unprecedented quality. However, as detailed in Fig. 1, these advanced models come with significant computational demands. The

![](images/07a85ab800dbee7c7b135c88371bb5ae5765624f05781fb8f9bee91fe3f79d93.jpg)  
Figure 1: Comparison of data usage, training time and image quality. Colors from dark to light represent parameters increasing in size, and circles from small to large indicate improvements in image quality.

![](images/123d006913e837928f7ac89f93be5887940155ad867d5f48f9ac38bf87d1e602.jpg)  
A furry cat

![](images/e2e73ecfd7e5f05a7029207eebcd29da61ccffa469c21e8f7af17a9e4bf731cd.jpg)  
Ted bundy in a pixar movie.

![](images/599a21dd26a7fb28ece8f484d295c31742e594fe38dac0a2dee49d83297db532.jpg)  
Studio photo portrait of Lain Iwakura wearing floral garlands over her traditional dress.

![](images/91641f46dce9f13d30418ff73c22a150a82fc051ec69d31dc290bc6882953ccb.jpg)  
A snowy Sweden lake in a vibrant, cinematic style with intense detail and raytracing.

![](images/dc21845fc500b4b6871d2961818138c969f3eb4d72f9d1848cfc87496e22be9c.jpg)  
A tiny planet image of Rio de Janeiro.

![](images/1e1d49a7b1a1e7ee8a405fb4bb69520f7ee2559901b49a1a282f24cc8e59452c.jpg)  
Beautiful warm tavern seen from the outside, middle age, river crossed by a bridge next to the tavern, crepuscular light.

![](images/b6ccaf56e0bbd0628e2f7cfe74ec88d2f55e38cca3c265110b3565d9fe36b10b.jpg)  
The solitary great tree centered in the image. cloudless sunny sky, little islands in the flooded plain.

![](images/be81aa8e663caa79dfa2909ba7f9a2e3726e3e48040dab308e1933ccdfb894f1.jpg)  
A 3d render of a cute, blue, anthropomorphic dragon with ice crystals growing off her, sharp focus.

![](images/345ae8292b70eb69ab6a2b3b3d1fe6a25e05e2370d7e595d336cdf58d9f2a4e3.jpg)  
Majestic ornate great hall, grand library, baroque, torches, stained glass windows, moonlight rays, dreamy mood.

![](images/40cc791cfaa8c7a23313f6a5982987faae83ffe07b3e7cd7265062ff377bde40.jpg)  
Figure 2: Samples produced by Fantasy  $(512 \times 512)$ . Each image, generated in 1.26 seconds (without super-resolution models), is accompanied by a descriptive caption showcasing diverse styles and comprehension.  
Breath taking beautiful, aesthetically pleasing, gouache ocean waves ripples, sea foam, sunset, digital concept art.

considerable expenses of these models create significant barriers for researchers and entrepreneurs. Meanwhile, economical text-to-image models [25, 15, 48] compromise on image quality, yielding lower resolution and diminished aesthetic appeal.

Given these challenges, a pivotal question arises: Can we develop a resource-efficient, high-quality image generator for long instructions? In this paper, we present Fantasy, significantly reducing training demands while maintaining the capability of instruction understanding and competitive image generation quality, as shown in Fig. 2. To achieve this, we propose three core designs:

Efficient T2I network. To leverage the powerful understanding ability of a decoder-only LLM, we choose the lightweight Phi-2 [24] as our text encoder. We derive discrete image tokens from a pre-trained VQGAN [27], and employ Transformer-based masked image modeling (MIM) as our T2I architecture. We also utilize the pre-trained VQGAN decoder [27] for pixel space restoration.

Hierarchical Training strategy. We propose a thoughtfully two-stage training strategy to address the high computational demands of current leading models while maintaining competitive performance: (1) large-scale concept alignment pre-training, (2) high-quality instruction-image fine-tuning. To facilitate a coarse image-text alignment, we initially train the T2I model from scratch using relatively lower-quality data. We then fine-tune the pre-trained T2I model and LLM on text-image pair data rich in information density with superior aesthetic quality.

High-quality data. To achieve rough alignment while pre-training, we select the large-scale dataset LAION-2B [37] and employ the filtering strategy proposed by DataComp [14]. We collect long-text prompts and corresponding high-quality synthesized images for instruction tuning, including DiffusionDB [42] and JourneyDB [39]. We further filter and discard texts with special characters and data containing violence or pornography, retaining only instructions exceeding 30 words.

Our main contributions are summarized as follows:

1. We present Fantasy, a novel framework that is the first to integrate a lightweight decoder-only LLM and a Transformer-based MIM for text-to-image synthesis, allowing for long-form text alignment.  
2. We show that our two-stage training strategy with high-quality data enables MIM to achieve comparable performance at a significantly reduced training cost.  
3. We provide comprehensive validation of the model's efficacy based on automated metrics and human feedback for visual appeal and text faithfulness.

![](images/912ac91e70d2eba8b5270f76b4fc9032bcf5efaeb5c99de88f0c2223e001b556.jpg)  
Stage 1: Large-scale concept alignment pre-training

![](images/65ff1f8a4af6ea9bb4c3697a985fecd992861f907996de7573a03d3b254caa7d.jpg)  
Figure 3: (Up) Overview of Fantasy featuring text encoder, VQGAN (encoder  $\mathcal{E}$  and decoder  $\mathcal{D}$ ), masked image generator  $\mathcal{G}$ , and super-resolution model. (Down) Our training pipeline involves two stages. The red parts are trainable and the blue parts are frozen; the yellow part is optionally utilized during inference.

![](images/bcb2955d4eee4f168b218536da7b86de4593257eee83f74f8d1d742a150a44e2.jpg)  
Stage 2: Instruction fine-tuning

# 2 Method

# 2.1 Problem Formulation

As depicted in Fig. 3, Fantasy consists of a pre-trained text encoder  $\mathcal{T}$ , a transformer-based masked image generator  $\mathcal{G}$ , a sampler  $S$ , a frozen VQGAN, and a pre-trained super-resolution model.  $\mathcal{T}$  maps a text prompt  $t$  to a continuous embedding space.  $\mathcal{G}$  processes a text embedding  $e$  to generate logits  $l$  for the visual token sequence.  $\mathcal{S}$  draws a sequence of visual tokens  $v$  from logits via iterative decoding [4], which runs  $N$  steps of inference conditioned on the text embeddings  $e$  and visual tokens decoded from previous steps. Finally,  $\mathcal{D}$  maps the sequence of discrete tokens to pixel space  $Z$ . To summarize, given a text prompt  $t$ , an image  $\hat{x}$  is synthesized as follows:

$$
\hat {x} = \mathcal {D} (\mathcal {S} (\mathcal {G}, \mathcal {T} (t))), \quad l _ {n} = \mathcal {G} (v _ {n}, \mathcal {T} (t)), \quad v _ {n} = \mathcal {M} (\mathcal {E} (x)) \tag {1}
$$

where  $n$  is the synthesis step, and  $l_{n}$  are logits, from which the next set of visual tokens  $v_{n + 1}$  are sampled.  $\mathcal{M}$  denotes the masking operator that applies masks to the token in  $v_{n}$ . We refer to [4, 3] for details on the iterative decoding process. The Phi-2 [24] for  $\mathcal{T}$  and VQGAN [8] for encoder  $\mathcal{E}$  and decoder  $\mathcal{D}$  are used.  $\mathcal{G}$  is trained on a large text-image pairs  $D$  using masked visual token modeling loss:

$$
\mathcal {L} = \mathbb {E} _ {(x, t) \sim D} \left[ C E \left(l _ {N}, \mathcal {E} (x)\right) \right], \tag {2}
$$

where  $CE$  is a weighted cross-entropy calculated by summing only over the unmasked tokens.

# 2.2 Model Architecture

# 2.2.1 VQGAN as Image Processor

VQGAN [8] is capable of transforming each image into discrete tokens with higher-level semantic information from a learned codebook, while ignoring low level noise. The autoregressive tokens prediction of VQGAN shares the same form as text tokens generated by LLMs. Prior research [46] has shown that unifying vision and language by the same token space could enhance the coherency for vision-text alignment. Furthermore, compared with RGB pixels, the visual token representation has proven to reduce disk storage and improve the capability of robustness and generalization.

To reduce the computational burden, we initially compress an RGB image  $v \in \mathbb{R}^{H \times W \times 3}$  into a diminished representation with a resolution of  $h \times w \times 3$ , where  $h = H / f$  and  $w = W / f$ , with  $f$  denoting the downsampling factor. We then employ a pre-trained f16 VQGAN [27] encoder  $\mathcal{E}$  to quantize images  $x \in \mathbb{R}^{3 \times 256 \times 256}$  into discrete tokens of spatial dimensions  $16 \times 16$  from a pre-trained codebook  $\mathcal{Z} = \{z_k\}_{k=1}^K$  consisting of  $K = 8192$  vectors, resulting in the quantized representation  $z = \mathcal{E}(x, \mathcal{Z})$ .

# 2.2.2 LLM as Text Encoder

Recent studies [10, 5, 3] tend to use encoder-decoder LLMs [31] for text encoding over CLIP [30], which is adept at handling tasks that involve complex mappings between input and output sequences. Due to the tremendous success of ChatGPT, attention has been drawn to models that consist solely of a decoder. Also, [43] presents an insight that efficiently fine-tuning a more powerful decoder-only LLM can yield stronger performance in long-text alignment. Consequently, to capitalize on the enhanced semantic comprehension and generalization potential of LLMs while simultaneously reducing the training burden, we employ Phi-2 [24], a state-of-the-art, lightweight LLM, as the text encoder.

Given the text prompt  $t$ , Fantasy first passes it through Phi-2, extracting the text embedding from the last hidden layer  $L$ . However, typically, decoder-only architectures are not adept at feature extraction and mapping tasks. [23] proposes that the conceptual representations learned by LLM's are roughly linearly mappable to those learned by models trained on vision tasks. Therefore, the embedding vectors are linearly projected to the hidden size of the image generator  $\mathcal{G}$ :

$$
c = \mathcal {P} \left(\mathcal {T} _ {L} (t)\right) \tag {3}
$$

where  $\mathcal{T}(\cdot)$  denotes the decoder-only Phi-2 and  $L$  is the index of the last hidden layer.  $\mathcal{P}$  represents the projection from text space to visual space, and  $c$  is the text feature suitable for the image generator.

# 2.2.3 MIM as Image Generator

MIM narrows the gap between its modeling and the extensively studied area of language modeling, making it straightforward to leverage the findings of the LLMs research community. Therefore, we adopt a masked transformer as the image generator backbone of Fantasy [46].

During training, we leave the projected text embeddings  $c$  unmasked and the image tokens  $z$  are masked at a variable masking rate based on a Cosine scheduling  $\mathcal{M}$  as [4, 3]. Specifically, for each training example, we sample a masking rate  $r$  from [0, 1] from a truncated  $\text{arccos}$  distribution with density function  $p(r) = \frac{2}{\pi} (1 - r^2)^{-\frac{1}{2}}$ . While autoregressive methods learn fixed-order token distributions  $P(z_i | z_{<i})$ , random masking with variable ratios enables learning  $P(z_i | z_{\neq i})$  for any token subset, crucial for our parallel sampling scheme. The sampling of a new state  $s_{n+1}$  at each successive step is conditioned on the previous state and the specified text condition  $c$ :

$$
P (s \mid c) = \int P \left(s _ {N} \mid s _ {N - 1}, c\right) \prod_ {n = 1} ^ {N - 1} P \left(s _ {n} \mid s _ {n - 1}, c\right) d s _ {1} \dots d s _ {N - 1} \tag {4}
$$

For each training example, the most confidently predicted tokens are revealed at each step  $n$ , maintaining  $\cos \left( \frac{n}{N} \cdot \frac{\pi}{2} \right)$  masked until reaching  $N$  total steps.

For the base model, we use a variant of MaskGiT [4], a masked image generative Transformer to predict randomly masked tokens by attending to tokens in all directions. Leveraging the multi-layered structure of the Transformer, we have developed scalable image generators with varying layer counts, ranging in size from 257M parameters to 611M parameters (for the image generator; the Phi-2 model has an additional 2.7B parameters). We first employ a series of Cross Attention blocks to optimize text-driven feature extraction, before passing through  $O$  layers of the masked image generator. Each layer  $o$  of the Transformer is again formed by Multi-Head Self-Attentuib(MSA), LayerNorm (LN), Cross Attention (CA) and Multi-Layer Perceptron (MLP) blocks:

$$
Y _ {o} = \operatorname {M S A} \left(\ln \left(Z _ {o}\right)\right), \quad Z _ {o + 1} = \operatorname {M L P} \left(\operatorname {C A} \left(\left(\ln \left(Y _ {o}\right), c\right)\right)\right). \tag {5}
$$

At the output layer, to reduce the training burden, ConvMLP [18] is utilized to transform masked image embeddings into logits sets, aligning with the VQGAN codebook dimensions. Eventually, the reconstructed lower-resolution tokens are restored with the pre-trained  $256 \times 256$  resolution VQGAN decoder to the pixel space, resulting in the generated image  $\hat{x}$ :

$$
\hat {x} = \mathcal {D} (\operatorname {C o n v M L P} \left(Z _ {O}\right), \mathcal {Z}) \tag {6}
$$

# 2.3 Training Strategy

Fig. 3 illustrates Fantasy's two-stage training approach. Following prior works[43, 35, 9], we employ large-scale pre-training to achieve general text-image concept alignment, and simultaneous fine-tuning of Phi-2 [24] and the masked image generator using high-quality instruction-image pairs.

Pre-training Stage. To perform general text-image concept alignment, the VQGAN and LLM weights are frozen, and only the image generator is pre-trained on deduplicated LAION-2B [37] with images above a 4.5 aesthetic score. We exclusively preserve prompts in English, filter out images above a  $50\%$  watermark probability or above a  $45\%$  NSFW probability, yielding a final set of 9 million images. Since the computational cost of upsampling is much lower than training a super-resolution model, Fantasy is started with training at a resolution of  $256 \times 256$ . Note that the pre-training only needs approximate image-text alignment, substantially lowering the training costs.

Fine-tuning Stage. [43] has proven that LLMs trained solely on text data lack prior image-text knowledge, and that merely aligning their text embeddings with visual features might not be optimal. Therefore, in the second stage, we gather an internal dataset of 7 million high-quality instruction-image pairs to fine-tune both the Phi-2 model and the image generator of Fantasy, which ensures enhanced compatibility of text embeddings within the text-image pair space, facilitating the use of decoder-only LLMs in text-to-image generation tasks and harnessing their inherent advantages. To prevent catastrophic forgetting in LLMs and preserve their understanding abilities during training, we select questions from BIG-bench [2] and monitor the common sense question-answering ability of Phi-2 in real-time throughout the training process. We construct our training dataset for the fine-tuning stage by incorporating JourneyDB [39] and an internal synthetic dataset to enhance the aesthetic quality of generated images beyond realistic photographs. To facilitate instruction-image alignment learning, we retain only data with descriptions exceeding 30 words, as these provide enough detailed insights into the image objects, including attributes and spatial relations.

With this approach, Fantasy trains a 0.6B parameter T2I model in about 69 A100 GPU days, significantly reducing computation compared to existing diffusion-based methods, while maintaining comparable visual and numerical fidelity. Throughout this paper, we present a comprehensive evaluation of Fantasy's efficacy, showcasing the potential in training high-quality transformer-based image synthesis models compared to diffusion-based models in future.

# 2.4 High-quality Data Collection

To ensure rough alignment in the pre-training phase, we utilize the large-scale dataset LAION-2B [37] and apply the filtering strategy developed by DataComp [14]. Furthermore, we gather long-text prompts and corresponding high-quality images to achieve finer-grained text-image alignment through instruction tuning. CapsFusion [47] employs a fine-tuned LLaMA [40] for recaptioning LAION-2B [37] and LAION-COCO [1]. However, this approach still results in suboptimal image quality and occasional mismatches between images and text. SAM-LLAVA [5] utilizes LLaVA [20] to recaption the SAM dataset [17], which leads to images with blurred faces, a consequence of the dataset's inherent face-blurring. Therefore, we shift focus to synthesize images, mainly including DiffusionDB [42] and JourneyDB [39], produced by Stable Diffusion and MidJourney, respectively. To augment the diversity of the images, we minimize the use of datasets from specific domains, such as gaming and anime. Furthermore, we implement filtering to discard texts with special characters and data containing violence or pornography, retaining only instructions exceeding 30 words.

# 3 Experiments

In this section, we outline detailed training, inference, and evaluation protocols, followed by comprehensive comparisons across three key metrics.

# 3.1 Implementation Details

**Training Details.** Different from the prior works [9, 43, 32, 34], we used a lightweight but powerful decoder-only large language model Phi-2 [24] as the text encoder. Diverging from prior approaches that extract a standard and fixed short text tokens, we extend the extraction to 256 tokens to master long-term instruction-image alignment, ensuring precise alignment for more fine-grained prompts. For the entire training process, we train Fantasy on  $4 \times \mathrm{A}100$  80G GPUs and set the accumulation step to 2. At different stages, we employ varying learning rate strategies with single-cycle cosine annealing decay. Furthermore, the AdamW optimizer [22] is utilized with a weight decay of 0.01. Fantasy trains a 0.6B parameter T2I model in about 84.5 A100 GPU days, significantly reducing computation compared to existing diffusion-based methods as shown in Fig. 1.

Table 1: Evaluation of diffusion (upper) and transformer (down) models on HPSv2. We underline the highest value and color the first above Fantasy in blue.  

<table><tr><td>Model</td><td>Type</td><td>Params</td><td>Animation</td><td>Concept-art</td><td>Painting</td><td>Photo</td><td>DrawBench [36]</td></tr><tr><td>GLIDE [25]</td><td>Diff</td><td>5.0B</td><td>23.34 ± 0.198</td><td>23.08 ± 0.174</td><td>23.27 ± 0.178</td><td>24.50 ± 0.290</td><td>25.05 ± 0.84</td></tr><tr><td>VQ-Diffusion [15]</td><td>Diff</td><td>0.37B</td><td>24.97 ± 0.186</td><td>24.70 ± 0.149</td><td>25.01 ± 0.145</td><td>25.71 ± 0.222</td><td>25.44 ± 0.83</td></tr><tr><td>Latent Diffusion [34]</td><td>Diff</td><td>1.45B</td><td>25.73 ± 0.125</td><td>25.15 ± 0.140</td><td>25.25 ± 0.178</td><td>26.97 ± 0.183</td><td>26.17 ± 0.85</td></tr><tr><td>DALL-E 2 [26]</td><td>Diff</td><td>6.5B</td><td>27.34 ± 0.175</td><td>26.54 ± 0.127</td><td>26.68 ± 0.156</td><td>27.24 ± 0.198</td><td>27.16 ± 0.64</td></tr><tr><td>Stable Diffusion v1.4 [33]</td><td>Diff</td><td>0.8B</td><td>27.26 ± 0.156</td><td>26.61 ± 0.082</td><td>26.66 ± 0.143</td><td>27.27 ± 0.226</td><td>27.23 ± 0.57</td></tr><tr><td>Stable Diffusion v2.0 [33]</td><td>Diff</td><td>0.8B</td><td>27.48 ± 0.174</td><td>26.89 ± 0.076</td><td>26.86 ± 0.120</td><td>27.46 ± 0.198</td><td>27.31 ± 0.68</td></tr><tr><td>DeepFloyd-XL [11]</td><td>Diff</td><td>4.3B</td><td>27.64 ± 0.108</td><td>26.83 ± 0.137</td><td>26.86 ± 0.131</td><td>27.75 ± 0.171</td><td>27.64 ± 0.72</td></tr><tr><td>LAFITE [48]</td><td>Trans</td><td>0.075B</td><td>24.63 ± 0.101</td><td>24.38 ± 0.087</td><td>24.43 ± 0.155</td><td>25.81 ± 0.213</td><td>25.23 ± 0.72</td></tr><tr><td>FuseDream [21]</td><td>Trans</td><td>-</td><td>25.26 ± 0.125</td><td>25.15 ± 0.107</td><td>25.13 ± 0.183</td><td>25.57 ± 0.248</td><td>25.72 ± 0.71</td></tr><tr><td>DALL-E mini [7]</td><td>Trans</td><td>0.4B</td><td>26.10 ± 0.132</td><td>25.56 ± 0.137</td><td>25.56 ± 0.112</td><td>26.12 ± 0.233</td><td>26.34 ± 0.76</td></tr><tr><td>VQGAN + CLIP [8]</td><td>Trans</td><td>0.2B</td><td>26.44 ± 0.152</td><td>26.53 ± 0.075</td><td>26.47 ± 0.111</td><td>26.12 ± 0.210</td><td>26.38 ± 0.43</td></tr><tr><td>CogView2 [12]</td><td>Trans</td><td>6B</td><td>26.50 ± 0.129</td><td>26.59 ± 0.119</td><td>26.33 ± 0.100</td><td>26.44 ± 0.271</td><td>26.17 ± 0.74</td></tr><tr><td>Fantasy (ours)</td><td>Trans</td><td>0.6B</td><td>27.03±0.131</td><td>26.66±0.117</td><td>26.72±0.176</td><td>26.80±0.174</td><td>26.78±0.523</td></tr></table>

Table 2: Comparison with recent T2I models. 'Trained' indicates the model develops a text encoder from scratch, foregoing a pre-trained one.  

<table><tr><td>Method</td><td>Type</td><td>Text Encoder</td><td>#Params</td><td>#Images</td><td>FID-30K (↓)</td></tr><tr><td>LDM [34]</td><td>Diff</td><td>Trained</td><td>1.4B</td><td>400M</td><td>12.64</td></tr><tr><td>GLIDE [25]</td><td>Diff</td><td>Trained</td><td>5.0B</td><td>-</td><td>12.24</td></tr><tr><td>DALL-E 2 [26]</td><td>Diff</td><td>CLIP</td><td>6.5B</td><td>650M</td><td>10.39</td></tr><tr><td>Stable Diffusion v1.5 [33]</td><td>Diff</td><td>CLIP</td><td>0.9B</td><td>2000M</td><td>9.62</td></tr><tr><td>SD XL [29]</td><td>Diff</td><td>CLIP</td><td>2.6B</td><td>-</td><td>&gt;18</td></tr><tr><td>Würstchen [28]</td><td>Diff</td><td>CLIP</td><td>0.99B</td><td>1420M</td><td>23.6</td></tr><tr><td>ParaDiffusion [43]</td><td>Diff</td><td>LLaMA V2</td><td>1.3B</td><td>&gt;300M</td><td>9.64</td></tr><tr><td>Pixart-α [5]</td><td>Diff</td><td>T5</td><td>0.6B</td><td>-</td><td>5.51</td></tr><tr><td>Cogview2 [12]</td><td>Trans</td><td>CogLM</td><td>6B</td><td>35M</td><td>24.0</td></tr><tr><td>Muse [3]</td><td>Trans</td><td>T5-XXL</td><td>3B</td><td>460M</td><td>7.88</td></tr><tr><td>Fantasy</td><td>Trans</td><td>Phi-2</td><td>0.6B</td><td>16M</td><td>23.4</td></tr></table>

Inference Details. We use  $N = 32$  sampling steps in all of our evaluation experiments. Since Fantasy is trained at a resolution of  $256 \times 256$ , we employ the pre-trained diffusion-based super-resolution model StableSR [41] to upscale images to  $512 \times 512$ .

Evaluation Metrics. We comprehensively evaluate Fantasy via four primary metrics, i.e., alignment on HPSv2 [44], FID [16] on MSCOCO dataset [19] and human evaluation on a collected dataset.

# 3.2 Performance Comparisons and Analysis

Results on HPSv2. We utilize HPSv2 [44] as our primary automated metric, a preference prediction model which can be used to compare images generated with the same prompt across five categories: anime, concept art, paintings, photography, and DrawBench [36]. We present the results of HPSv2 between Fantasy and other state-of-the-art generative models in Tab. 1. Fantasy exhibited outstanding performance across all key aspects among previous Transformer-based methods like CogView2 [12], which is expected. The results also reveal its competitive performance compared to prior diffusion-based methods, especially in concept-art and painting, demonstrating similar performance to DALL-E 2 [26]. This remarkable performance is primarily attributed to the text-image alignment learning in fine-tuning stage, where high-quality text-image pairs were leveraged to achieve superior alignment capabilities. In comparison, DeepFloyd-XL and other diffusion-based models achieve better scores, while utilizing larger models with significantly higher compute budget.

Results on FID. We employ FID [16] to evaluate our models on COCO-30K [19]. To allow for a fair comparison, all images are downsampled to  $256 \times 256$  pixels. The comparison between our method and other methods in FID, and their training time is summarized in Tab. 2. We observe that the FID of Fantasy is substantially higher compared to other state-of-the-art models. Visual inspections reveal that images generated by Fantasy are smoother than those from other leading T2I models. This discrepancy is most noticeable in real-world images like COCO, on which we compute the FID-metric. Although the state-of-the-art models [43, 11, 29] exhibit lower FID, it relies on unaffordable resources. Furthermore, prior studies [29, 5, 11] have demonstrated that FID may not

![](images/990db17e14aec3e41ea9082ca792b50f41147aec8d3705c4740cf5dad610f730.jpg)  
(a) User study on long prompts.

![](images/01d2a486043d52713f7c86099df4844a932ebe65adbb86700e0830fef4398ace.jpg)  
Figure 4: User study on prompts with different length. VC., CV2., FT., SD., and PA. refer to VQGAN+CLIP [8], CogView2 [12], our Fantasy, Stable Diffusion v2.0 [33], and Pixart-  $\alpha$  [5].  
(b) User study on short prompts.

be an appropriate metric for image quality evaluation, as a lower score does not necessarily reflect superior image generation, and it is more authoritative to use the evaluation of human users.

# 3.3 Results on Human Evaluation

Following prior works [5, 43, 28], we also conduct a study with human participants to supplement our evaluation and provide a more intuitive assessment of Fantasy's performance. Participants are asked to select a preference of the images based on the visual appeal of the generated images and the precision of alignments between the text prompts and the corresponding images.

As involving human evaluators can be time-consuming, we choose the top-performing open-source diffusion-based models (e.g., SD XL [33], and Pixart-  $\alpha$  [5]) and transformer-based models (e.g., VQGAN+CLIP [8] and CogView2 [12]) as our baseline, which are accessible through APIs and capable of generating images. We randomly select a total of 600 prompts from existing prompt sets (e.g., ParaPrompt [43], ViLG-300 [13], COCO Captions [6]). To comprehensively contrast the capabilities of Fantasy and other models in interpreting text prompts of varying lengths, we allocate one subset to consist of 300 prompts ranging from 10 to 30 characters and another subset comprising 300 prompts exceeding 30 characters. For each model, we use a consistent set to generate images, which are then evaluated by 50 individuals.

Fig. 4a clearly demonstrates that images generated on relatively long text prompts (longer than 30 words) by Fantasy are distinctly favored among the four models in both two perspective, especially for text-image alignment, aligning closely with the intended use case of Fantasy. As illustrated in Fig. 4b, for text prompts shorter than 30 words, our model outperforms existing open-source Transformer-based models in fidelity and alignment for shorter prompts. Our model slightly lags behind diffusion-based models in visual appeal, limited by the 8,192 size of VQGAN's codebook and not targeting visual appeal. Simultaneously, Fantasy lacks a distinct advantage in text-image alignment in the short subset. We hypothesize that this is due to two main reasons: diffusion-based models' ability to handle shorter prompts, and vague prompts generating diverse images that make preferences more subjective, thus biasing outcomes towards aesthetically superior images. In summary, the human preference experiments confirm the observation made in the HPSv2 benchmarks.

# 3.4 Case Study

Fig. 5 vividly illustrates Fantasy's superior visual appeal and text-image alignment over leading open-source transformer-based T2I models [12, 8] and diffusion-based T2I models [29, 26]. Fantasy significantly surpasses existing transformer-based T2I models, matches the performance of SDXL [29], and qualitatively outperforms Dall-E 2 [26]. Despite being trained on images with a resolution of  $256 \times 256$ , Fantasy ensures generated low-resolution images contain sufficient details, indirectly supporting long prompts. Limited by computing resources, we haven't

A close-up photo of a person. The subject is a male. He was wearing a wide-brimmed hat, a gray-white beard on his face, a brown coat. His facial expression looked pensive and serious, with the clear blue sky in the background.

![](images/3c683467d969c974859270d16e79aab8b8dd32614ba382c31604a7e601b05241.jpg)  
ParaDiffusion

![](images/e725592a2e1f223ecb80bb681b5b25fdc7c22582809b0a8ee4c16c69abf64f3e.jpg)  
Fantasy

A young man wearing a black leather jacket and tie stood behind an old door, his gaze firmly fixed on the camera. The door had patterns of leaves and flowers on it, revealing a yellow background. His hair was casually curled and he appeared to be deep in thought or contemplating something.

![](images/7cd582088bdc8b38224579feb0fe8284bfc26b422e67dd1bfbad0654616d63c9.jpg)  
Figure 6: Visual Comparison with ParaDiffusion [43]: Red markings and boxes highlight text misalignments in images generated by ParaDiffusion.  
ParaDiffusion

![](images/997f8c3666b3c70e09f6e715b31de92e1462ce213a595354f9e77832abe7a230.jpg)  
Fantasy

![](images/af47e6d76d0a9569c5ca09bde69d2eba1d1f9581004e213b90ee35c441498722.jpg)  
Figure 5: Visual comparison with existing T2I models. (a) A hamster resembling a horse. (b) A frontal portrait of a anime girl with chin length pink hair wearing sunglasses and a white T-shirt smiling. (c) A colorful illustration of a suburban neighborhood on an ancient post-apocalyptic planet featuring creatures made by Jim Henson's workshop. (d) A blue-haired girl with soft features stares directly at the camera in an extreme close-up Instagram picture. (e) A building in a landscape by Ivan Aivazovsky. (f) Aoshima's masterpiece depicts a forest illuminated by morning light. (g) The image is a highly detailed portrait of an oak in GTA V, created using Unreal Engine and featuring fantasy artwork by various artists.

Table 3: Ablation study on two stages with the best bolded. 'Base' indicates the model after the pre-training stage.  

<table><tr><td>Model</td><td>Training Part</td><td>Animation</td><td>Concept-art</td><td>Painting</td><td>Photo</td><td>DrawBench [36]</td></tr><tr><td>Base</td><td>MIM</td><td>25.27 ± 0.190</td><td>24.20 ± 0.166</td><td>24.60 ± 0.146</td><td>25.32 ± 0.208</td><td>25.49 ± 0.230</td></tr><tr><td>Fantasy</td><td>MIM+Phi-2</td><td>27.03±0.131</td><td>26.66±0.117</td><td>26.72±0.176</td><td>26.80±0.174</td><td>26.78±0.521</td></tr></table>

trained on higher resolutions like  $512 \times 512$  but aim to enhance Fantasy by training at higher resolutions in the future.

ParaDiffusion [43] pioneers the use of decoder-only large language models as text encoders in text-to-image generation. As illustrated in Fig. 6, our observations suggest that Fantasy more closely aligns details with prompts than ParaDiffusion [43].

# 262 4 Ablation Study

This section analyzes the effects of LLMs fine-tuning, and model scale on Fantasy's performance through ablation studies. More ablation study refers to appendix.

# 265 4.1 Effect of Language Model Fine-tuning

To assess the effect of training strategies on the comprehension of complex instructions, we perform a human preference evaluation, as detailed in Sec. 3.3, using a subset of 300 prompts longer than 30 characters. 'Base' denotes general text-image alignment with filtered LAION-2B [1] in the pre-training stage. Compared to the base model, our synergy fine-tuning with Phi-2 demonstrates a notable improvement in all aspects in Tab. 3.

271

Table 4: Ablation study on models at different scales with the best bolded. DB. represents DrawBench [36].  

<table><tr><td>Layers</td><td>Param</td><td>Animation</td><td>Concept-art</td><td>Painting</td><td>Photo</td><td>DB.</td></tr><tr><td>6</td><td>257M</td><td>25.79±0.15</td><td>25.84±0.11</td><td>25.92±0.19</td><td>25.63±0.18</td><td>25.18±0.22</td></tr><tr><td>12</td><td>421M</td><td>26.34±0.17</td><td>26.29±0.06</td><td>26.45±0.17</td><td>26.19±0.17</td><td>25.68±0.14</td></tr><tr><td>22</td><td>611M</td><td>27.03±0.13</td><td>26.66±0.11</td><td>26.72±0.17</td><td>26.80±0.17</td><td>26.78±0.52</td></tr></table>

Table 5: Training cost for Fantasy at 3 different scales. BS. denotes batch size and LR. denotes learning rate.  

<table><tr><td rowspan="2">Layers</td><td colspan="3">Pre-training</td><td colspan="3">Fine-tuning</td></tr><tr><td>Steps (K)</td><td>BS.</td><td>LR.</td><td>Steps (K)</td><td>BS.</td><td>LR.</td></tr><tr><td>6</td><td>180</td><td>768</td><td>1e-4</td><td>180</td><td>192</td><td>1e-4</td></tr><tr><td>12</td><td>220</td><td>768</td><td>1e-4</td><td>250</td><td>192</td><td>1e-4</td></tr><tr><td>22</td><td>370</td><td>256</td><td>5e-4</td><td>280</td><td>128</td><td>3e-4</td></tr></table>

# 4.2 Scale of Image Generator

The hierarchical structure of the Transformer allows us to train image generators with varying numbers of Transformer layers. As shown in Tab. 4, we evaluate models of different sizes on the HPSv2 benchmark. The insight indicates that as trainable parameters increase from 257 million to 611 million, performance consistently improves. Therefore, we set the number of Transformer layers to 22 with 611 million trainable parameters as the optimal setting. Tab. 5 showcases the required resources for models of three different scales. Fig. 7 offers visual comparisons across models of varying scales, illustrating a clear trend: models with fewer parameters underperform on the HPSv2 benchmark, frequently resulting in distorted images and omitted details, yet they may still generate acceptable outcomes. Significantly, the visual quality diverges as model size increases, highlighting the potential for scaling up masked image modeling to enhance instruction-image alignment and elevate generation quality.

![](images/bf11237f90999daf45b2eea36285b4d30bf0ed331c398a1f19d053846fa709fb.jpg)  
Figure 7: Examples generated by models at different scales:  $1^{st}$  column for 6 layers,  $2^{nd}$  column for 12 layers and  $3^{rd}$  column for 22 layers.

# 5 Limitations and Social Impact

Limitations. Despite Fantasy achieving competitive performance in text-image alignment and visual appeal, it requires improvements in handling complex scenes. We propose two possible strategies to overcome the challenge in future research: Firstly, augmenting the dataset with high-quality images can enhance diversity and refine the model. Secondly, since the scale of the masked image generator affects instruction-image alignment, training an upscale image generator based on higher resolution left further explored.

Social Impact. Generative models for media bring both benefits and challenges. They foster creativity and make technology more accessible, yet pose risks by facilitating the creation of manipulated content, spreading misinformation, and exacerbating biases, particularly affecting women with deep fakes. Concerns also include the potential exposure of sensitive training data collected without consent. Despite generative models potentially offering better data representation, the impact of combining adversarial training with likelihood-based objectives on data distortion remains a crucial research area. Ethical considerations of these models are significant and require thorough exploration.

# 6 Conclusion

In this paper, we introduce Fantasy, a lightweight and efficient text-to-image model that combines Large Language Models (LLMs) with a transformer-based masked image modeling (MIM), effectively transferring semantic understanding capabilities from LLMs to the text-to-image generation. With our proposed two-stage training strategy and high-quality dataset, Fantasy significantly reduces computational requirements while producing high-fidelity images. Extensive experiments demonstrate that Fantasy achieves comparable performance to models trained with significantly more computational resources, illustrating the viability of our approach and suggesting potential efficient scalability to even larger masked image modeling for text-to-image generation.

# References

[1] Köpf Andreas, Vencu Richard, Coombes Theo, and Beaumont Romain. Laion coco: 600m synthetic captions from laion2b-en.[eb/ol], 2022.  
[2] BIG bench authors. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023.  
[3] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.  
[4] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11315-11325, 2022.  
[5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426, 2023.  
[6] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arxiv 2015. arXiv preprint arXiv:1504.00325, 2015.  
[7] Craiyon. Dall-e mini: Generate images from any text prompt. https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-images-from-any-text-prompt--VmlldzoyMDE4NDAy, 2023. Accessed: 2024-02-27.  
[8] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance. In European Conference on Computer Vision, pages 88–105. Springer, 2022.  
[9] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.  
[10] Deepfloyd. Deepfloyd. https://www deepfloyd.ai/, 2023.  
[11] DeepFloyd. IF-I-XL-v1.0: A model by deepfloyd on hugging face models. https://huggingface.co/DeepFloyd/IF-I-XL-v1.0, 2023. Accessed: 2024-02-28.  
[12] Ming Ding, Wendi Zheng, Wenyi Hong, and Jie Tang. Cogview2: Faster and better text-to-image generation via hierarchical transformers. Advances in Neural Information Processing Systems, 35:16890-16902, 2022.  
[13] Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie-vilg 2.0: Improving text-to-image diffusion model with knowledge-enhanced mixture-of-denoising-experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10135-10145, 2023.  
[14] Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36, 2024.  
[15] Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10696-10706, 2022.  
[16] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.  
[17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.  
[18] Jiachen Li, Ali Hassani, Steven Walton, and Humphrey Shi. Convmlp: Hierarchical convolutional mlp's for vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6306-6315, 2023.  
[19] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740-755. Springer, 2014.  
[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.  
[21] Xingchao Liu, Chengyue Gong, Lemeng Wu, Shujian Zhang, Hao Su, and Qiang Liu. Fusedream: Training-free text-to-image generation with improved clip+ gan space optimization. arXiv preprint arXiv:2112.01573, 2021.  
[22] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[23] Jack Merullo, Louis Castricato, Carsten Eickhoff, and Ellie Pavlick. Linearly mapping from image to text space. arXiv preprint arXiv:2209.15162, 2022.  
[24] Microsoft. Phi-2. https://huggingface.co/microsoft/phi-2, 2023.  
[25] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.  
[26] OpenAI. Dall-e 2. https://openai.com/dall-e-2, 2022.  
[27] Suraj Patil, William Berman, Robin Rombach, and Patrick von Platen. amused: An open muse reproduction. arXiv preprint arXiv:2401.01808, 2024.  
[28] Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. Würstchen: An efficient architecture for large-scale text-to-image diffusion models. In The Twelfth International Conference on Learning Representations, 2023.  
[29] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.  
[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR, 2021.  
[31] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020.  
[32] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.  
[33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684-10695, June 2022.  
[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684-10695, 2022.  
[35] Chitwan Sahara, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding, 2022. URL https://arxiv.org/abs/2205.11487, 4.  
[36] Chitwan Sahara, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479-36494, 2022.  
[37] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in Neural Information Processing Systems, 35:25278-25294, 2022.  
[38] Eyal Segalis, Dani Valevski, Danny Lumen, Yossi Matias, and Yaniv Leviathan. A picture is worth a thousand words: Principled recaptioning improves image generation. arXiv preprint arXiv:2310.16656, 2023.  
[39] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. Journeydb: A benchmark for generative image understanding. Advances in Neural Information Processing Systems, 36, 2024.  
[40] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  
[41] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. arXiv preprint arXiv:2305.07015, 2023.  
[42] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. arXiv:2210.14896 [cs], 2022.  
[43] Weijia Wu, Zhuang Li, Yefei He, Mike Zheng Shou, Chunhua Shen, Lele Cheng, Yan Li, Tingting Gao, Di Zhang, and Zhongyuan Wang. Paragraph-to-image generation with information-enriched diffusion model. arXiv preprint arXiv:2311.14284, 2023.  
[44] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[45] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. arXiv preprint arXiv:2401.11708, 2024.  
[46] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion-tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.  
[47] Qiying Yu, Quan Sun, Xiaosong Zhang, Yufeng Cui, Fan Zhang, Xinlong Wang, and Jingjing Liu. Capsfusion: Rethinking image-text data at scale. arXiv preprint arXiv:2310.20550, 2023.  
[48] Y Zhou, R Zhang, C Chen, C Li, C Tensmeyer, T Yu, J Gu, J Xu, and T Sun. Lafite: Towards language-free training for text-to-image generation. arxiv 2021. arXiv preprint arXiv:2111.13792.
