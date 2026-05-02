# BEiT: BERT PRE-TRAINING OF IMAGE TRANSFORMERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT (Devlin et al., 2019) developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e., image patches (such as  $16 \times 16$  pixels), and visual tokens (i.e., discrete tokens). We first "tokenize" the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods.

# 1 INTRODUCTION

Transformer (Vaswani et al., 2017) has achieved promising performance in computer vision (Dosovitskiy et al., 2020; Touvron et al., 2020). However, empirical studies show that vision Transformers require more training data than convolutional neural networks. In order to solve the data-hungry issue, self-supervised pre-training is a promising solution to leverage large-scale image data. Several strands of methods have been explored for vision Transformers, such as contrastive learning (Chen et al., 2021; Xie et al., 2021), and self-distillation (Caron et al., 2021).

Concurrently, BERT (Devlin et al., 2019) has achieved great success in natural language processing. Its masked language modeling task first randomly masks some proportion of tokens within a text, and then recovers the masked tokens based on the Transformer encoding results of the corrupted text. Motivated by BERT, we turn to the denoising auto-encoding idea to pretrain vision Transformers, which has not been well studied by the vision community. It is challenging to directly apply BERT-style pre-training for image data. First of all, there is no pre-exist vocabulary for vision Transformer's input unit, i.e., image patches. So we cannot simply employ a softmax classifier to predict over all possible candidates for masked patches. In contrast, the language vocabulary, such as words and BPE (Sennrich et al., 2016), is well-defined and eases auto-encoding prediction. A straightforward alternative is regarding the task as a regression problem, which predicts the raw pixels of masked patches. However, such pixel-level recovery task tends to waste modeling capability on pre-training short-range dependencies and high-frequency details (Ramesh et al., 2021). Our goal is to overcome the above issues for pre-training of vision Transformers.

In this work, we introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Inspired by BERT, we propose a pre-training task, namely, masked image modeling (MIM). As shown in Figure 1, MIM uses two views for each images, i.e., image patches, and visual tokens. We split the image into a grid of patches that are the input representation of backbone Transformer. Moreover, we "tokenize" the image to discrete visual tokens, which is obtained by the latent codes of discrete VAE (Ramesh et al., 2021). During pre-training, we randomly mask some proportion of image patches, and feed the corrupted input to Transformer. The model learns to recover the visual tokens of the original image, instead of the raw pixels of masked patches.

![](images/b96ff900bcc1f1ed2ff2bbd576f7e846d363a50808dc607be91fd8b389b46df7.jpg)  
Figure 1: Overview of BEiT pre-training. Before pre-training, we learn an "image tokenizer" via autoencoding-style reconstruction, where an image is tokenized into discrete visual tokens according to the learned vocabulary. During pre-training, each image has two views, i.e., image patches, and visual tokens. We randomly mask some proportion of image patches (gray patches in the figure) and replace them with a special mask embedding [M]. Then the patches are fed to a backbone vision Transformer. The pre-training task aims at predicting the visual tokens of the original image based on the encoding vectors of the corrupted image.

We perform self-supervised learning and then fine-tune the pretrained BEiT on two downstream tasks, i.e., image classification, and semantic segmentation. Experimental results indicate that BEiT outperforms both from-scratch training and previous strong self-supervised models. Moreover, BEiT is complementary to supervised pre-training. Performance of BEiT can be further improved by intermediate fine-tuning with ImageNet labels. Ablation studies show that our proposed techniques are critical to the effectiveness of BERT-style pre-training for image data. Apart from performance, the improvements of convergence speed and stability of fine-tuning reduce training costs on end tasks. In addition, we demonstrate that self-supervised BEiT can learn reasonable semantic regions via pre-training, unleashing the rich supervision signals contained in images.

Our contributions are summarized as follows:

- We propose a masked image modeling task to pretrain vision Transformers in a self-supervised manner. We also provide a theoretical explanation from the perspective of variational autoencoder.  
- We pretrain BEiT and conduct extensive fine-tuning experiments on downstream tasks, such as image classification, and semantic segmentation.  
- We present that the self-attention mechanism of self-supervised BEiT learns to distinguish semantic regions and object boundaries, although without using any human annotation.

# 2 METHODS

Given an input image  $x$ , BEiT encodes it to contextualized vector representations. As shown in Figure 1, BEiT is pretrained by the masked image modeling (MIM) task in a self-supervised learning manner. MIM aims at recovering the masked image patches based on encoding vectors. For downstream tasks (such as image classification, and semantic segmentation), we append task layers upon pretrained BEiT and fine-tune the parameters on the specific datasets.

# 2.1 IMAGE REPRESENTATIONS

The images have two views of representations in our method, namely, image patch, and visual tokens. The two types serve as input and output representations during pre-training, respectively.

# 2.1.1 IMAGE PATCH

The 2D image is split into a sequence of patches (Dosovitskiy et al., 2020), so that a standard Transformer can directly accept image data. Formally, we reshape the image  $\pmb{x} \in \mathbb{R}^{H \times W \times C}$  into  $N = HW / P^2$  patches  $\pmb{x}^p \in \mathbb{R}^{N \times (P^2C)}$ , where  $C$  is the number of channels,  $(H, W)$  is the input image resolution, and  $(P, P)$  is the resolution of each patch. The image patches  $\{\pmb{x}_i^p\}_{i=1}^N$  are flattened into vectors and are linearly projected, which is similar to word embeddings in BERT (Devlin et al., 2019). Image patches preserve raw pixels and are used as input features in BEiT.

In our experiments, we split each  $224 \times 224$  image into a  $14 \times 14$  grid of image patches, where each patch is  $16 \times 16$ .

# 2.1.2 VISUALTOKEN

Similar to natural language, we represent the image as a sequence of discrete tokens obtained by an "image tokenizer", instead of raw pixels. Specifically, we tokenize the image  $\boldsymbol{x} \in \mathbb{R}^{H \times W \times C}$  into  $z = [z_1, \ldots, z_N] \in \mathcal{V}^{h \times w}$ , where the vocabulary  $\mathcal{V} = \{1, \ldots, |\mathcal{V}|\}$  contains discrete token indices.

Following (Ramesh et al., 2021), we use the image tokenizer learned by discrete variational autoencoder (dVAE). There are two modules during visual token learning, namely, tokenizer and decoder. The tokenizer  $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$  maps image pixels  $\boldsymbol{x}$  into discrete tokens  $\boldsymbol{z}$  according to a visual codebook (i.e., vocabulary). The decoder  $p_{\psi}(\boldsymbol{x}|\boldsymbol{z})$  learns to reconstruct the input image  $\boldsymbol{x}$  based on the visual tokens  $\boldsymbol{z}$ . The reconstruction objective can be written as  $\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\psi}(\boldsymbol{x}|\boldsymbol{z})]$ . Because the latent visual tokens are discrete, the model training is non-differentiable. Gumbel-softmax relaxation (Jang et al., 2017; Maddison et al., 2017) is employed to train the model parameters. Moreover, a uniform prior is put on  $q_{\phi}$  during dVAE training. Refer to (Ramesh et al., 2021) for more training details of the image tokenizer.

We tokenize each image to a  $14 \times 14$  grid of visual tokens. Notice the number of visual tokens and the number of image patches for one image are the same. The vocabulary size is set to  $|\mathcal{V}| = 8192$ . In our work, we directly use the publicly available<sup>1</sup> image tokenizer described in (Ramesh et al., 2021). We also compare it with a re-implemented tokenizer in Appendix C.

# 2.2 BACKBONE NETWORK: IMAGE TRANSFORMER

Following ViT (Dosovitskiy et al., 2020), we use the standard Transformer (Vaswani et al., 2017) as the backbone network. So the results can be directly compared with previous work in terms of the network architecture.

The input of Transformer is a sequence of image patches  $\{\pmb{x}_i^p\}_{i=1}^N$ . The patches are then linearly projected to obtain patch embeddings  $\pmb{E}\pmb{x}_i^p$ , where  $\pmb{E} \in \mathbb{R}^{(P^2C) \times D}$ . Moreover, we prepend a special token [S] to the input sequence. We also add standard learnable 1D position embeddings  $\pmb{E}_{pos} \in \mathbb{R}^{N \times D}$  to patch embeddings. The input vectors  $\pmb{H}_0 = [e_{[S]}, \pmb{E}\pmb{x}_i^p, \dots, \pmb{E}\pmb{x}_N^p] + \pmb{E}_{pos}$  is fed into Transformer. The encoder contains  $L$  layers of Transformer blocks  $\pmb{H}^l = \mathrm{Transformer}(\pmb{H}^{l-1})$ , where  $l = 1, \dots, L$ . The output vectors of the last layer  $\pmb{H}^L = [\pmb{h}_{[S]}^L, \pmb{h}_1^L, \dots, \pmb{h}_N^L]$  are used as the encoded representations for the image patches, where  $\pmb{h}_i^L$  is the vector of the  $i$ -th image patch.

# 2.3 PRE-TRAINING BEIT: MASKED IMAGE MODELING

We propose a masked image modeling (MIM) task. We randomly mask some percentage of image patches, and then predict the visual tokens that are corresponding to the masked patches.

Figure 1 shows the overview of our method. As presented in Section 2.1, given an input image  $\pmb{x}$ , we split it into  $N$  image patches  $(\{\pmb{x}_i^P\}_{i=1}^N)$ , and tokenize it to  $N$  visual tokens  $(\{z_i\}_{i=1}^N)$ . We

randomly mask approximately  $40\%$  image patches, where the masked positions are denoted as  $\mathcal{M} \in \{1, \dots, N\}^{0.4N}$ . Next we replace the masked patches with a learnable embedding  $e_{[\mathbb{M}]} \in \mathbb{R}^D$ . The corrupted image patches  $x^{\mathcal{M}} = \{x_i^p : i \notin \mathcal{M}\}_{i=1}^N \cup \{e_{[\mathbb{M}]} : i \in \mathcal{M}\}_{i=1}^N$  are then fed into the  $L$ -layer Transformer as described in Section 2.2. The final hidden vectors  $\{h_i^L\}_{i=1}^N$  are regarded as encoded representations of the input patches. For each masked position  $\{h_i^L : i \in \mathcal{M}\}_{i=1}^N$ , we use a softmax classifier to predict the corresponding visual tokens  $p_{\mathrm{MIM}}(z'|x^{\mathcal{M}}) = \mathrm{softmax}_{z'}(W_c h_i^L + b_c)$  where  $x^{\mathcal{M}}$  is the corrupted image,  $W_c \in \mathbb{R}^{|\mathcal{V}| \times D}$ , and  $b_c \in \mathbb{R}^{|\mathcal{V}|}$ . The pre-training objective is to maximize the log-likelihood of the correct visual tokens  $z_i$  given the corrupted image:

$$
\max  \sum_ {x \in \mathcal {D}} \mathbb {E} _ {\mathcal {M}} \left[ \sum_ {i \in \mathcal {M}} \log p _ {\mathrm {M I M}} \left(z _ {i} \mid x ^ {\mathcal {M}}\right) \right] \tag {1}
$$

where  $\mathcal{D}$  is the training corpus,  $\mathcal{M}$  represents randomly masked positions, and  $x^{\mathcal{M}}$  is the corrupted image that is masked according to  $\mathcal{M}$ .

Rather than randomly choosing patches for the masked positions  $\mathcal{M}$ , we employ blockwise masking in our work. As summarized in Algorithm 1, a block of image patches is masked each time. For each block, we set the minimum number of patches to 16. Then we randomly choose an aspect ratio for the masking block. We repeat the above two steps until obtaining enough masked patches, i.e.,  $0.4N$ , where  $N$  is the total number of image patches, and 0.4 is masking ratio.

Algorithm 1 Blockwise Masking  
Input:  $N(= h\times w)$  image patches   
Output: Masked positions  $\mathcal{M}$ $\mathcal{M}\gets \{\}$    
repeat   
 $s\gets \mathrm{Rand}(16,0.4N - |\mathcal{M}|)$  ▷ Block size   
 $r\gets \mathrm{Rand}(0.3,\frac{1}{0.3})$  ▷ Aspect ratio of block   
 $a\gets \sqrt{s\cdot r};b\gets \sqrt{s / r}$ $t\gets \mathrm{Rand}(0,h - a);l\gets \mathrm{Rand}(0,w - b)$ $\mathcal{M}\gets \mathcal{M}\bigcup \{(i,j):i\in [t,t + a),j\in [l,l + b)\}$    
until  $|\mathcal{M}| > 0.4N$  ▷ Masking ratio is  $40\%$    
return  $\mathcal{M}$

The MIM task is greatly inspired by masked language modeling (Devlin et al., 2019), which is one of the most successful pre-training objective in natural language processing. Moreover, blockwise (or n-gram) masking is also widely applied in BERT-like models (Joshi et al., 2020; Bao et al., 2020; Raffel et al., 2020). However, directly using pixel-level auto-encoding for vision pre-training pushes the model to focus on short-range dependencies and high-frequency details (Ramesh et al., 2021). BEiT overcomes the above issue by predicting discrete visual tokens, which summarizes the details to high-level abstractions. Ablation studies in Section 3.3 show that our proposed method significantly outperforms pixel-level auto-encoding.

# 2.4 FROM THE PERSPECTIVE OF VARIATIONAL AUTOENCODER

The BEiT pre-training can be viewed as variational autoencoder (Kingma & Welling, 2014) training. Let  $x$  denote the original image,  $\tilde{x}$  the masked image, and  $z$  the visual tokens. Considering the evidence lower bound (ELBO) of the log-likelihood  $p(x|\tilde{x})$ , i.e., recovering the original image from its corrupted version:

$$
\sum_ {\left(x _ {i}, \tilde {x} _ {i}\right) \in \mathcal {D}} \log p \left(x _ {i} \mid \tilde {x} _ {i}\right) \geq \sum_ {\left(x _ {i}, \tilde {x} _ {i}\right) \in \mathcal {D}} \left(\underbrace {\mathbb {E} _ {z _ {i} \sim q _ {\phi} (\mathbf {z} \mid x _ {i})} \left[ \log p _ {\psi} \left(x _ {i} \mid z _ {i}\right) \right]} _ {\text {V i s u a l T o k e n R e c o n s t r u c t i o n}} - D _ {\mathrm {K L}} \left[ q _ {\phi} (\mathbf {z} \mid x _ {i}), p _ {\theta} (\mathbf {z} \mid \tilde {x} _ {i}) \right]\right) \tag {2}
$$

where (1)  $q_{\phi}(z|x)$  denotes the image tokenizer that obtains visual tokens; (2)  $p_{\psi}(x|z)$  decodes the original image given input visual tokens; (3)  $p_{\theta}(z|\tilde{x})$  recovers the visual tokens based on the masked image, which is our MIM pre-training task.

We learn the model following a two-stage procedure similar to (van den Oord et al., 2017; Razavi et al., 2019). In the first stage, we obtain the image tokenizer as a discrete variational autoencoder (Ramesh et al., 2021). Specifically, the first stage minimizes the reconstruction loss  $-\mathbb{E}_{z_i\sim q_\phi (\mathbf{z}|x_i)}[\log p_\psi (x_i|z_i)]$  with an uniform prior as described in Equation (2). In the second stage, we learn the prior  $p_{\theta}$  while keeping  $q_{\phi}$  and  $p_{\psi}$  fixed. We simplify  $q_{\phi}(\mathbf{z}|x_i)$  to a one-point distribution with the most likely visual tokens  $\hat{z}_i = \arg \max_z q_\phi (z|x_i)$ . Then Equation (2) can be rewritten as:

$$
\sum_ {\left(x _ {i}, \tilde {x} _ {i}\right) \in \mathcal {D}} \left(\underbrace {\mathbb {E} _ {z _ {i} \sim q _ {\phi} (z \mid x _ {i})} \left[ \log p _ {\psi} \left(x _ {i} \mid z _ {i}\right) \right]} _ {\text {S t a g e 1 : V i s u a l T o k e n R e c o n s t r u c t i o n}} + \underbrace {\log p _ {\theta} \left(\hat {z} _ {i} \mid \tilde {x} _ {i}\right)} _ {\text {S t a g e 2 : M a s k e d I m a g e M o d e l i n g}}\right) \tag {3}
$$

where the second term is our BEIT pre-training objective.

# 2.5 PRE-TRAINING SETUP

The network architecture of BEiT follows that of ViT-Base (Dosovitskiy et al., 2020) for a fair comparison. We use a 12-layer Transformer with 768 hidden size, and 12 attention heads. The intermediate size of feed-forward networks is 3072. We employ the default  $16 \times 16$  input patch size. We directly borrow the image tokenizer trained by Ramesh et al. (2021). The vocabulary size of visual tokens is 8192.

We pretrain BEIT on the training set of ImageNet-1K (Russakovsky et al., 2015), which contains about  $1.2\mathrm{M}$  images. Our augmentation policy includes random resized cropping, horizontal flipping, color jittering (Wu et al., 2018). Notice that we do not use the labels for self-supervised learning. We use the  $224 \times 224$  resolution in our experiments. So the input is split to  $14 \times 14$  image patches, and the same amount of visual tokens. We randomly mask at most 75 patches (i.e., roughly  $40\%$  of total image patches).

The pre-training runs for about 500k steps (i.e., 800 epochs) with 2k batch size. Adam (Loshchilov & Hutter, 2019) with  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$  is employed for optimization. The learning rate is set to  $1.5\mathrm{e - }3$ , with a warmup of 10 epochs, and cosine learning rate decay. The weight decay is 0.05. We employ stochastic depth (Huang et al., 2016) with a 0.1 rate, and disable dropout. The 500k training steps take about five days using 16 Nvidia Telsa V100 32GB GPU cards.

# 2.6 FINE-TUNING BEIT ON DOWNSSTREAM VISION TASKS

After pre-training BEiT, we append a task layer upon the Transformer, and fine-tune the parameters on downstream tasks, like BERT. We take image classification and semantic segmentation as examples in our work. It is straightforward to leverage the pre-training-then-fine-tuning paradigm on other vision tasks with BEiT.

Image classification. For image classification tasks, we directly employ a simple linear classifier as the task layer. Specifically, we use average pooling to aggregate the representations, and feed the global to a softmax classifier. The category probabilities are computed as  $\mathrm{softmax}(\mathrm{avg}(\{h_i^L\}_{i=1}^N W_c))$ , where  $h_i^L$  is the final encoding vector of the  $i$ -th image patch,  $W_c \in \mathbb{R}^{D \times C}$  is a parameter matrix, and  $C$  is the number of labels. We maximize the likelihood of labeled data by updating the parameters of BEiT and the softmax classifier.

Semantic segmentation. For semantic segmentation, we follow the task layer used in SETR-PUP (Zheng et al., 2020). To be specific, we use pretrained BEiT as a backbone encoder, and incorporate several deconvolution layers as decoder to produce segmentation. The model is also end-to-end fine-tuned similar to image classification.

Intermediate fine-tuning. After self-supervised pre-training, we can further train BEiT on a data-rich intermediate dataset (i.e., ImageNet-1K in our work), and then finetune the model on the target downstream tasks. Such intermediate fine-tuning is the common practice of BERT fine-tuning in NLP (Pruksachatkun et al., 2020). We directly follow the method for BEiT.

# 3 EXPERIMENTS

We conduct full fine-tuning experiments on image classification and semantic segmentation. Moreover, we present various ablation studies for pre-training and analyze the representations learned by BEiT. We also report linear probes on ImageNet in Appendix D.

# 3.1 IMAGE CLASSIFICATION

The image classification task classifies input images to various categories. We evaluate BEiT on the ILSVRC-2012 ImageNet dataset (Russakovsky et al., 2015) with 1k classes and 1.3M images. We directly follow the most of hyperparameters of DeiT (Touvron et al., 2020) in our fine-tuning experiments for a fair comparison. We reduce fine-tuning epochs compared with training from scratch, as BEiT has been pre-trained. Accordingly, we use a larger learning rate with layer-wise decay. The detailed hyperparameters are summarized in Appendix H.

Table 1: Top-1 accuracy on ImageNet-1K. We evaluate base- ("-B") and large-size ("-L") models at resolutions  ${224} \times  {224}$  and  ${384} \times  {384}$  . †: iGPT-1.36B contains 1.36 billion parameters, while others are base-size models. ‡: ViT  ${}_{384}$  -B-JFT300M is pretrained with the "masked patch prediction" task on Google's in-house 300M images, while others use ImageNet.  

<table><tr><td>Models</td><td>Model Size</td><td>Resolution</td><td>ImageNet</td></tr><tr><td colspan="4">Training from scratch (i.e., random initialization)</td></tr><tr><td>ViT384-B (Dosovitskiy et al., 2020)</td><td>86M</td><td>3842</td><td>77.9</td></tr><tr><td>ViT384-L (Dosovitskiy et al., 2020)</td><td>307M</td><td>3842</td><td>76.5</td></tr><tr><td>DeiT-B (Touvron et al., 2020)</td><td>86M</td><td>2242</td><td>81.8</td></tr><tr><td>DeiT384-B (Touvron et al., 2020)</td><td>86M</td><td>3842</td><td>83.1</td></tr><tr><td colspan="4">Supervised Pre-Training on ImageNet-22K (using labeled data)</td></tr><tr><td>ViT384-B (Dosovitskiy et al., 2020)</td><td>86M</td><td>3842</td><td>84.0</td></tr><tr><td>ViT384-L (Dosovitskiy et al., 2020)</td><td>307M</td><td>3842</td><td>85.2</td></tr><tr><td colspan="4">Self-Supervised Pre-Training on ImageNet-1K (without labeled data)</td></tr><tr><td>iGPT-1.36B† (Chen et al., 2020a)</td><td>1.36B</td><td>2242</td><td>66.5</td></tr><tr><td>ViT384-B-JFT300M‡ (Dosovitskiy et al., 2020)</td><td>86M</td><td>3842</td><td>79.9</td></tr><tr><td>MoCo v3-B (Chen et al., 2021)</td><td>86M</td><td>2242</td><td>83.2</td></tr><tr><td>MoCo v3-L (Chen et al., 2021)</td><td>307M</td><td>2242</td><td>84.1</td></tr><tr><td>DINO-B (Caron et al., 2021)</td><td>86M</td><td>2242</td><td>82.8</td></tr><tr><td>BEiT-B (ours)</td><td>86M</td><td>2242</td><td>83.2</td></tr><tr><td>BEiT384-B (ours)</td><td>86M</td><td>3842</td><td>84.6</td></tr><tr><td>BEiT-L (ours)</td><td>307M</td><td>2242</td><td>85.2</td></tr><tr><td>BEiT384-L (ours)</td><td>307M</td><td>3842</td><td>86.3</td></tr></table>

Table 1 reports top-1 accuracy on image classification. We compare BEiT with vision Transformers trained by random initialization, supervised pre-training, and previous self-supervised learning methods. All the compared models are base-size, except iGPT has 1.36B parameters. Pre-training is conducted on ImageNet for the comparison purpose, except ViT-JFT300M is pretrained on Google's in-house 300M images.

Compared with the models trained by random initialization, we find that pre-trained BEiT significantly improves performance on both datasets. BEiT improves the performance on ImageNet, which shows the effectiveness under the rich-resource setting.

Moreover, we compare BEiT with previous state-of-the-art self-supervised methods for Transformer, such as DINO (Caron et al., 2021), and MoCo v3 (Chen et al., 2021). Our proposed method outperforms previous models on ImageNet fine-tuning. Among them, iGPT-1.36B (Chen et al., 2020a) uses much more parameters (i.e., 1.36B vs 86M), and ViT-JFT300M (Dosovitskiy et al., 2020) is pretrained on larger corpus (i.e., 300M vs 1.3M), while others pretrain ViT-Base on ImageNet-1K. iGPT-1.36B and ViT-JFT300M are the most comparable methods, which also follows auto-encoding pre-training for vision Transformer. Specifically, iGPT uses clustered image tokens as both input and output for image GPT or image BERT. In contrast, we use image patches as input to preserve raw pixels, and employ discrete visual tokens as a prediction bottleneck. ViT-JFT300 predicts the mean, 3-bit color of each masked patch, rather than visual tokens learned by discrete VAE. We also pretrain the self-supervised tasks of BEiT and DINO in a multi-task learning manner, which is presented in Appendix E.

In addition, we evaluate our proposed method with intermediate fine-tuning. In other words, we first pretrain BEiT in a self-supervised manner, and then fine-tune the pretrained model on ImageNet with labeled data. The results show that BEiT is complementary to supervised pre-training, achieving additional gain after intermediate fine-tuning on ImageNet.

Fine-tuning to  $384 \times 384$  resolution. After fine-tuning with resolution  $224 \times 224$ , we additionally fine-tune the model on  $384 \times 384$  images by 10 more epochs. We follow the standard higher-resolution setting of DeiT (Touvron et al., 2020), except using fewer epochs. Notice that we keep patch size the same for both  $224 \times 224$  and  $384 \times 384$  images. So the input sequence length of Transformers

![](images/4f206ef22b96ca135c55781a89d29c4f53d46aa9685f3af294067ce6c43f7cbd.jpg)  
Table 2: Convergence curves of training DeiT from scratch and fine-tuning BEiT on ImageNet-1K.

Table 3: Results of semantic segmentation on ADE20K. We use SETR-PUP (Zheng et al., 2020) as the task layer and report results of single-scale inference.  

<table><tr><td>Models</td><td>ADE20K</td></tr><tr><td>Supervised Pre-Training on ImageNet</td><td>45.3</td></tr><tr><td>DINO (Caron et al., 2021)</td><td>44.1</td></tr><tr><td>BEiT (ours)</td><td>45.6</td></tr><tr><td>BEiT + Intermediate Fine-Tuning (ours)</td><td>47.7</td></tr></table>

becomes longer for higher resolutions. Table 1 shows that higher resolution improves the BEIT results by  $1+$  points on ImageNet. More importantly,  $\mathrm{BEIT}_{384}$  pretrained on ImageNet-1K even outperforms supervised pre-training  $\mathrm{ViT}_{384}$  that uses ImageNet-22K, when they use the same input resolution.

Scaling up to larger size. We further scale up BEiT to the large size (same as ViT-L). As shown in Table 1,  $\mathrm{ViT}_{384}$ -L is worse than  $\mathrm{ViT}_{384}$  on ImageNet, when training from scratch. The results verifies the data-hungry issue of vision Transformers. Supervised pre-training on ImageNet-22K partially relieves the issue, where  $\mathrm{ViT}_{384}$ -L finally outperforms  $\mathrm{ViT}_{384}$  by 1.2. In comparison, BEiT-L is better than BEiT by 2.0, and  $\mathrm{BEiT}_{384}$ -L outperforms  $\mathrm{BEiT}_{384}$  by 1.7. In other words, the benefits of scaling up BEiT from base to large are greater than supervised pre-training with ImageNet-22K. More importantly, comparing between  $\mathrm{BEiT}_{384}$  with  $\mathrm{ViT}_{384}$  that conducts supervised pre-training on ImageNet-22K, the improvements of BEiT become greater along with scaling the size from base (i.e., 0.6) to large (i.e., 1.1). The results suggest that BEiT tends to help more for extremely larger models (such as 1B, or 10B), especially when labeled data are insufficient $^2$  to conduct supervised pre-training $^3$  for such large models.

Convergence curves. Figure 2 compares the convergence curves of the training-from-scratch and pre-training-then-fine-tuning paradigms. We find that fine-tuning BEiT not only achieves better performance, but also converging much faster than training DeiT from scratch. Moreover, fine-tuning BEiT can reach reasonable numbers within very few epochs.

# 3.2 SEMANTIC SEGMENTATION

Semantic segmentation aims to predict a corresponding class for each pixel of the input image. We evaluate BEiT on the ADE20K benchmark (Zhou et al., 2019) with 25K images and 150 semantic categories. We report the metric of mean Intersection of Union (mIoU) averaged over all semantic categories. As presented in Section 2.6, we directly follow the task layer and the most of hyperparameters described in SETR-PUP (Zheng et al., 2020). On ADE20K, we use Adam (Loshchilov & Hutter, 2019) as the optimizer. The learning rate is set to 1e-3 with layer-wise decay similar to image classification. We conduct fine-tuning for 160K steps. The batch size is 16. The detailed hyperparameters are described in Appendix I.

As shown in Table 3, we compare BEiT with supervised pre-training that relies on labeled data of ImageNet. We find that our proposed method achieves better performance than supervised pretraining, although BEiT does not require manual annotations for pre-training. Moreover, we employ intermediate fine-tuning for BEiT on ImageNet, i.e., we first fine-tune pretrained BEiT on ImageNet, and then fine-tune the model on ADE20K. The results indicate that intermediate fine-tuning further improves BEiT on semantic segmentation.

Table 4: Ablation studies for BEiT pre-training on image classification and semantic segmentation.  

<table><tr><td>Models</td><td>ImageNet</td><td>ADE20K</td></tr><tr><td>BEiT (300 Epochs)</td><td>82.86</td><td>44.65</td></tr><tr><td>- Blockwise masking</td><td>82.77</td><td>42.93</td></tr><tr><td>- Visual tokens (i.e., recover masked pixels)</td><td>81.04</td><td>41.38</td></tr><tr><td>- Visual tokens - Blockwise masking</td><td>80.50</td><td>37.09</td></tr><tr><td>+ Recover 100% visual tokens</td><td>82.59</td><td>40.93</td></tr><tr><td>- Masking + Recover 100% visual tokens</td><td>81.67</td><td>36.73</td></tr><tr><td>Pretrain longer (800 epochs)</td><td>83.19</td><td>45.58</td></tr></table>

# 3.3 ABLATION STUDIES

We conduct ablation studies to analyze the contributions of each component in BEIT. The models are evaluated on image classification (i.e., ImageNet) and semantic segmentation (i.e., ADE20K). We set the default pre-training steps to 300 epochs for the ablation studies, which is  $37.5\%$  of the total steps used in the previous experiments.

Table 4 reports the results of various model variants. First, we ablate blockwise masking by randomly sample masked positions. We find that blockwise masking is beneficial on both tasks, especially on semantic segmentation. Second, we ablate the usage of visual tokens by predicting the raw pixels of masked patches, i.e., the pre-training task becomes a pixel regression problem to recover masked patches. Our proposed masked image modeling task significantly outperforms naive pixel-level auto-encoding. Compared with the results in Table 1, the ablation result is worse than training vision Transformer from scratch on two tasks. The results indicate that the prediction of visual tokens is the key ingredient of BEiT. Third, we ablate the usage of visual tokens and blockwise masking together. We find that blockwise masking is even more helpful for pixel-level auto-encoding, which relieves the suffering of short-distance dependency. Forth, recovering all the visual tokens harms performance on downstream tasks. Fifth, we compare BEiT with different training steps. Pre-training the model longer can further improve performance on downstream tasks.

# 3.4 ANALYSIS OF SELF-ATTENTION MAP

We show that the self-attention mechanism in BEiT can separate objects, even though our pre-training does not rely on any manual annotation at all. Similar properties are also observed by Caron et al. (2021). The probing images are taken from the MS COCO (Lin et al., 2014) corpus to avoid appearing in the pre-training data.

As shown in Figure 2, we plot the self-attention map for different reference points within an image. The visualizations are produced by attention scores computed via query-key product in the last layer. For each reference point, we use the corresponding patch as query, and show which patch it attends to. After pre-training, BEiT learns to distinguish semantic regions using self-attention heads, without any task-specific supervision. The property partially indicates the reason why BEiT is able to help downstream tasks. Such knowledge acquired by BEiT potentially improves the generalization ability of fine-tuned models, especially on small-scale datasets.

# 4 RELATED WORK

Self-supervised visual representation learning. Various methods have been introduced over the years to pretrain vision models in a self-supervised manner. Pioneering works design clever pretext tasks, such as predicting the patch orderings (Noroozi & Favaro, 2016), colorization (Zhang et al., 2016), and predicting rotation angles (Komodakis & Gidaris, 2018). In addition, Trinh et al. (2019) propose to mask some patches within an image, and classify whether the masked patches are real or fake for each masked position. The method is similar to the masked version of Jigsaw pretraining (Noroozi & Favaro, 2016). The recent strand of research follows contrastive paradigm (Wu et al., 2018; Oord et al., 2018; Hjelm et al., 2019; Bachman et al., 2019; He et al., 2020; Chen et al.,

![](images/1a3acdb0afb42afb940bf3cc963b79499b7a744724031b05577dfcc4d60de1a1.jpg)  
Figure 2: Self-attention map for different reference points. The self-attention mechanism in BEiT is able to separate objects, although self-supervised pre-training does not use manual annotations.

2020b;c). The models typically regard various data augmentations as different views of an image, and then make the representations of positive pairs similar while pushing negative pairs away. In order to obtain enough informative negative samples in contrastive learning, the methods usually rely on large memory banks (Wu et al., 2018; He et al., 2020) or large batch size (Chen et al., 2020b). BYOL (Grill et al., 2020) and SimSiam (Chen & He, 2020) further eliminate the requirement of negative samples, using various techniques to avoid representation collapse. Another strand of methods use clustering to organize image examples (Caron et al., 2018; Asano et al., 2020; Caron et al., 2020; Li et al., 2021).

Self-supervised vision Transformers. Pre-training vision Transformers has received significant attention recently due to the data-hungry issue. iGPT (Chen et al., 2020a) first creates a 9-bit color palette by k-means clustering RGB pixels, and then uses the clustered tokens to represent images. Next iGPT uses the tasks of BERT and GPT to pretrain Transformers. In comparison, our proposed method uses image patches as input without losing pixel-level information. Moreover, our visual tokens are obtained by discrete VAE instead of clustering. ViT (Dosovitskiy et al., 2020) conducts a preliminary exploration with the masked patch prediction task, which predicts the 3-bit mean color of the masked patches. Dosovitskiy et al. (2020) also report that pixel-level auto-encoding performs worse, although it is the most straightforward translation of BERT from NLP to CV. Rather than using heuristically designed pre-training tasks, our proposed model leverages visual tokens learned by discrete VAE, which not only achieves better performance but also is better theoretically motivated. Apart from masked auto-encoding, other mainstream research works use contrastive learning (Chen et al., 2021; Xie et al., 2021), and self-distillation (Caron et al., 2021). In comparison, BEiT can achieve several times of improvement in terms of pre-training throughput, and memory consumption. The advantages make BEiT appealing to scale up vision Transformers.

# 5 CONCLUSION

We introduce a self-supervised pre-training framework for vision Transformers, achieving strong fine-tuning results on downstream tasks, such as image classification, and semantic segmentation. We show that the proposed method is critical to make BERT-like pre-training (i.e., auto-encoding with masked input) work well for image Transformers. We also present the intriguing property of automatically acquired knowledge about semantic regions, without using any human-annotated data. In the future, we would like to scale up BEiT pre-training in terms of data size and model size. Moreover, we will conduct multi-modal pre-training in a more unified way, using the similar objectives and the shared architecture for texts and images.

# REFERENCES

Yuki M. Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In International Conference on Learning Representations (ICLR), 2020.  
Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.  
Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Jianfeng Gao, Songhao Piao, Ming Zhou, and Hsiao-Wuen Hon. UniLMv2: Pseudo-masked language models for unified language model pre-training. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, volume 119 of Proceedings of Machine Learning Research, pp. 642-652. PMLR, 2020. URL http://proceedings.mlr.press/v119/bao20a.html.  
Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 132-149, 2018.  
Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Advances in Neural Information Processing Systems, volume 33, pp. 9912-9924. Curran Associates, Inc., 2020.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. arXiv preprint arXiv:2104.14294, 2021.  
Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 1691-1703. PMLR, 13-18 Jul 2020a. URL http://proceedings.mlr.press/v119/chen20s.html.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. preprint arXiv:2002.05709, 2020b.  
Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. preprint arXiv:2011.10566, 2020.  
Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. preprint arXiv:2003.04297, 2020c.  
Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. ArXiv, abs/2104.02057, 2021.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4171–4186. Association for Computational Linguistics, 2019.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. preprint arXiv:2010.11929, 2020.  
Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bk1r3j0cKX.  
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. Deep networks with stochastic depth. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling (eds.), Computer Vision – ECCV 2016, pp. 646–661, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46493-0.  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In 5th International Conference on Learning Representations, ICLR 2017, Toulouse, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https://openreview.net/forum?id=rkE3y85ee.  
Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and Omer Levy. SpanBERT: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics, 8:64-77, 2020. doi: 10.1162/tacl_a_00300. URL https://www.aclweb.org/anthology/2020.tacl-1.5.  
Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR 2014, 2014.  
Nikos Komodakis and Spyros Gidaris. Unsupervised representation learning by predicting image rotations. In International Conference on Learning Representations (ICLR), 2018.  
A. Krizhevsky and G. Hinton. Learning multiple layers of features from tiny images. Master's thesis, Department of Computer Science, University of Toronto, 2009.  
Junnan Li, Pan Zhou, Caiming Xiong, and Steven Hoi. Prototypical contrastive learning of unsupervised representations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=KmykpuSrjcq.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740-755. Springer, 2014.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030, 2021.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.  
Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. In International Conference on Learning Representations, 2017.  
Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European conference on computer vision, pp. 69-84. Springer, 2016.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. preprint arXiv:1807.03748, 2018.  
Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, and Samuel R. Bowman. Intermediate-task transfer learning with pretrained language models: When and why does it work? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, July 2020.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1-140:67, 2020. URL http://jmlr.org/papers/v21/20-074.html.  
A. Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. ArXiv, abs/2102.12092, 2021.  
Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. IJCV, 2015.  
Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1715-1725, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1162. URL https://www.aclweb.org/anthology/P16-1162.  
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. preprint arXiv:2012.12877, 2020.  
Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. arXiv preprint arXiv:2103.17239, 2021.  
Trieu H Trinh, Minh-Thang Luong, and Quoc V Le. Selfie: Self-supervised pretraining for image embedding. arXiv preprint arXiv:1906.02940, 2019.  
Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, pp. 6309-6318, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 5998-6008, 2017.  
Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR, 2018.  
Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In ECCV, 2018.  
Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu. Self-supervised learning with swin transformers. arXiv preprint arXiv:2105.04553, 2021.  
Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. arXiv preprint arXiv:2106.04560, 2021.  
Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, 2016.  
Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip H. S. Torr, and Li Zhang. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. CoRR, abs/2012.15840, 2020. URL https://arxiv.org/abs/2012.15840.  
Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ADE20K dataset. Int. J. Comput. Vis., 127(3): 302-321, 2019. doi: 10.1007/s11263-018-1140-0. URL https://doi.org/10.1007/s11263-018-1140-0.
