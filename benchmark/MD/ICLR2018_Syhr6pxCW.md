# PIXELNN: EXAMPLE-BASED IMAGE SYNTHESIS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a simple nearest-neighbor (NN) approach that synthesizes high-frequency photorealistic images from an "incomplete" signal such as a low-resolution image, a surface normal map, or edges. Current state-of-the-art deep generative models designed for such conditional image synthesis lack two important things: (1) they are unable to generate a large set of diverse outputs, due to the mode collapse problem. (2) they are not interpretable, making it difficult to control the synthesized output. We demonstrate that NN approaches potentially address such limitations, but suffer in accuracy on small datasets. We design a simple pipeline that combines the best of both worlds: the first stage uses a convolutional neural network (CNN) to map the input to a (overly-smoothed) image, and the second stage uses a pixel-wise nearest neighbor method to map the smoothed output to multiple high-quality, high-frequency outputs in a controllable manner. Importantly, pixel-wise matching allows our method to compose novel high-frequency content by cutting-and-pasting pixels from different training exemplars. We demonstrate our approach for various input modalities, and for various domains ranging from human faces, pets, shoes, and handbags.

![](images/7b639aae5e431d1b783b9424cdfbee9bda69aeef2579ec0ceca4e3ae08bc134e.jpg)  
12x12 Input (x8)

![](images/f2d954a59aa21bfabc7a36437bc4dddb06390cce7d368ef9634733095826c64d.jpg)  
Our Approach

![](images/95c3c1e3851761fed85f566647008d5b3db6b51c7d7db321abfb0a73f0df1db6.jpg)  
Surface Normal Map

![](images/38210fdfe9739d4d12f6bd6ccedb4581b38d8f7acac150c748cef85f5a0f2946.jpg)  
Our Approach

![](images/e57da4ea4e95feb152ee58ceff2b319f5ce533d96b39f0c4f8a8b9c8246f0d9d.jpg)  
Edges

![](images/6cb2f32c4f43c2097dfdc1dad5879530cafc1700ff8cbc1561ade08d72748059.jpg)  
Our Approach

![](images/b29799a9b393c008012a01a8308a6216fc8e5e5b54a735104a2142f93d977fee.jpg)  
(a) Low-Resolution to High-Resolution

![](images/9f8f3a3d9e9a92381e1076ea64c2c474a6bf31f673a05372f1291251d1918275.jpg)  
(d) Edges-to-Shoes (Multiple Outputs)

![](images/73821d569ad6ee278ac3ea31e300c739dcf23129224fce1357646eb791442787.jpg)  
Figure 1: Our approach generates photorealistic output for various "incomplete" signals such as a low resolution image, a surface normal map, and edges/boundaries for human faces, cats, dogs, shoes, and handbags. Importantly, our approach can easily generate multiple outputs for a given input which was not possible in previous approaches (Isola et al., 2016) due to mode-collapse problem. Best viewed in electronic format.

![](images/bdd09761d427f1635069c5c8b82ed294334a106b05a048647f77c7c9e5fb6f46.jpg)  
(c) Edges-to-RGB  
(e) Edges-to-Handbags (Multiple Outputs)

# 1 INTRODUCTION

We consider the task of generating high-resolution photo-realistic images from incomplete input such as a low-resolution image, sketches, surface normal map, or label mask. Such a task has a number of practical applications such as upsampling/colorizing legacy footage, texture synthesis for graphics applications, and semantic image understanding for vision through analysis-by-synthesis. These problems share a common underlying structure: a human/machine is given a signal that is missing considerable details, and the task is to reconstruct plausible details.

![](images/c3405c19a8310168131dd13444d1525e9273dafab0221ca78fc46a7f1195fa60.jpg)

![](images/7f6471d1bd099f3739c12a54c80b96b610b3a1d4e6310b50d8c302b01602a181.jpg)  
(a) Edges-to-Shoes

![](images/d361efb6e164a9ea417b96922f7b392c5c616a729566d7f2c9dadb14965a60ec.jpg)  
Figure 2: Mode collapse problem for GANs: We ran pix-to-pix pipeline of Isola et al. (2016) 72 times. Despite the random noise set using dropout at test time, we observe similar output generated each time. Here we try to show 6 possible diverse examples of generation for a hand-picked best-looking output from Isola et al. (2016).

![](images/609be11412d5c809ab4dfc7b333c752a178213140a6dbd01782499650b5f6f91.jpg)

![](images/a5fac735d5d678d850ab8c4ad94dbb688c4725a155d6ad58cba409a31fe6313b.jpg)  
(b) Edges-to-Cats-and-Dogs

![](images/12d2b2edc33717c09d877ee4c72b1c92b6321d7a0bca326a5a25f093e936d674.jpg)

![](images/e191435602531b280c0b393bc15efb85dba584ba7d02b5fba4c7edd8681fe2e5.jpg)

Consider the edge map of cat in Figure 1-c. When we humans look at this edge map, we can easily imagine multiple variations of whiskers, eyes, and stripes that could be viable and pleasing to the eye. Indeed, the task of image synthesis has been well explored, not just for its practical applications but also for its aesthetic appeal.

GANs: Current state-of-the-art approaches rely on generative adversarial networks (GANs) (Goodfellow et al., 2014), and most relevant to us, conditional GANS that generate image conditioned on an input signal (Denton et al., 2015; Radford et al., 2015; Isola et al., 2016). We argue that there are two prominent limitations to such popular formalisms: (1) First and foremost, humans can imagine multiple plausible output images given a incomplete input. We see this rich space of potential outputs as a vital part of the human capacity to imagine and generate. Conditional GANs are in principle able to generate multiple outputs through the injection of noise, but in practice suffer from limited diversity (i.e., mode collapse) (Fig. 2). Recent approaches even remove the noise altogether, treating conditional image synthesis as regression problem (Chen & Koltun, 2017). (2) Deep networks are still difficult to explain or interpret, making the synthesized output difficult to modify. One implication is that users are not able to control the synthesized output. Moreover, the right mechanism for even specifying user constraints (e.g., "generate a cat image that looks like my cat") is unclear. This restricts applicability, particularly for graphics tasks.

Nearest-neighbors: To address these limitations, we appeal to a classic learning architecture that can naturally allow for multiple outputs and user-control: non-parametric models, or nearest-neighbors (NN). Though quite a classic approach (Efros & Leung, 1999; Freeman et al., 2002; Hertzmann et al., 2001; Johnson et al., 2011), it has largely been abandoned in recent history with the advent of deep architectures. Intuitively, NN matches an incomplete input query to a large corpus of training pairs of (incomplete inputs, high-quality outputs), and simply returns the corresponding output. This trivially generalizes to multiple outputs through  $K$ -NN and allows for intuitive user control through on-the-fly modification of the training corpus - e.g., by restricting the training exemplars to those that "look like my cat".

In practice, there are several limitations in applying NN for conditional image synthesis. The first is a practical lack of training data. The second is a lack of an obvious distance metric. And the last is a computational challenge of scaling search to large training sets.

Approach: To reduce the dependency on training data, we take a compositional approach by matching local pixels instead of global images. This allows us to synthesize a face by "copy-pasting" the eye of one training image, the nose of another, etc. Compositions dramatically increase the representational power of our approach: given that we want to synthesize an image of  $K$  pixels using  $N$  training images (with  $K$  pixels each), we can synthesize an exponential number  $(NK)^{K}$  of compositions, versus a linear number of global matches  $(N)$ . A significant challenge, however, is defining an appropriate feature descriptor for matching pixels in the incomplete input signal. We would like to capture context (such that whisker pixels are matched only to other whiskers) while allowing for compositionality (left-facing whiskers may match to right-facing whiskers). To do so, we make use of deep features, as described below.

Pipeline: Our precise pipeline (Figure 3) works in two stages. (1) We first train an initial regressor (CNN) that maps the incomplete input into a single output image. This output image suffers from the aforementioned limitations - it is a single output that will tend to look like a "smoothed" average of all the potential images that could be generated. (2) We then perform nearest-neighbor queries on

![](images/d99e28a5a9e1d7c4cef769304874d2fcb800cf8024ad172aae41a9f435850400.jpg)  
Input

![](images/a465bf84cb7047245bf718cc92d03af226491cb5476c58fd71665330d6602a06.jpg)  
Stage-1: Regression

![](images/fcdb36ca5a25e5aee832fe24014ae760ba56378f58b84060921989624d5b30ec.jpg)  
Pixel Representation  
Figure 3: Overview of pipeline: Our approach is a two-stage pipeline. The first stage directly regresses an image from an incomplete input (using a CNN trained with  $l_{2}$  loss). This image will tend to look like a "smoothed" average of all the potential images that could be generated. In the second stage, we look for matching pixels in similarly-smoothed training images. Importantly, we match pixels using multiscale descriptors that capture the appropriate levels of context (such that eye pixels tend to match only to eyes). To do so, we make use of off-the-shelf hypercolumn features extracted from a CNN trained for semantic pixel segmentation. By varying the size of the matched set of pixels, we can generate multiple outputs (on the right).

![](images/b9fc0d5f86efed3c6d18b347ab9c35f0e8e5521f494035530f0cd9f30239529b.jpg)  
Stage-2: Contextual Copy-Pasting

![](images/651a3aa74487f9dbab135e667f27cae87df64f4d851069cd82e434331c0a47bd.jpg)

![](images/a999991df6fa669a787ee1845683ad66b30283ca83e891f2f1f929b922f0f3bc.jpg)

pixels from this regressed output. Importantly, pixels are matched (to regressed outputs from training data) using a multiscale deep descriptor that captures the appropriate level of context. This enjoys the aforementioned benefits - we can efficiently match to an exponential number of training examples in an interpretable and controllable manner. Finally, an interesting byproduct of our approach is the generation of dense, pixel-level correspondences from the training set to the final synthesized outputs.

# 2 RELATED WORK

Our work is inspired by a large body of work on discriminative and generative models, nearest neighbors architectures, pixel-level tasks, and dense pixel-level correspondences. We provide a broad overview, focusing on those most relevant to our approach.

Synthesis with CNNs: Convolutional Neural Networks (CNNs) have enjoyed great success for various discriminative pixel-level tasks such as segmentation (Bansal et al., 2017; Long et al., 2015), depth and surface normal estimation (Bansal et al., 2016; Eigen et al., 2013; Eigen & Fergus, 2015), semantic boundary detection (Bansal et al., 2017; Xie & Tu, 2015) etc. Such networks are usually trained using standard losses (such as softmax or  $l_{2}$  regression) on image-label data pairs. However, such networks do not typically perform well for the inverse problem of image synthesis from a (incomplete) label, though exceptions do exist (Chen & Koltun, 2017). A major innovation was the introduction of adversarially-trained generative networks (GANs) (Goodfellow et al., 2014). This formulation was hugely influential in computer visions, having been applied to various image generation tasks that condition on a low-resolution image (Denton et al., 2015; Ledig et al., 2016), segmentation mask (Isola et al., 2016), surface normal map (Wang & Gupta, 2016) and other inputs (Chen et al., 2016; Huang et al., 2016; Radford et al., 2015; Wu et al., 2016; Zhang et al., 2016a; Zhu et al., 2017). Most related to us is Isola et al. (2016) who proposed a general loss function for adversarial learning, applying it to a diverse set of image synthesis tasks. Importantly, they report the problem of mode collapse, and so cannot generate diverse outputs nor control the synthesis with user-defined constraints (unlike our work).

Interpretability and user-control: Interpreting and explaining the outputs of generative deep networks is an open problem. As a community, we do not have a clear understanding of what, where, and how outputs are generated. Our work is fundamentally based on copy-pasting information via nearest neighbors, which explicitly reveals how each pixel-level output is generated (by in turn revealing where it was copied from). This makes our synthesized outputs quite interpretable. One important consequence is the ability to intuitively edit and control the process of synthesis. Zhu et al. (2016) provide a user with controls for editing image such as color, and outline. But instead of using a predefined set of editing operations, we allow a user to have an arbitrarily-fine level of control through on-the-fly editing of the exemplar set (e.g., "resynthesize the output using the eye from this training image and the nose from that one").

Correspondence: An important byproduct of pixelwise NN is the generation of pixelwise correspondences between the synthesized output and training examples. Establishing such pixel-level correspondence has been one of the core challenges in computer vision (Choy et al., 2016; Kanazawa

![](images/1e16452510fd88f58c35211f3feaa5ed5582cda2610148b5c7fb0cd95f8f3d25.jpg)  
Low Resolution Image (12x12)

![](images/5b2fd42a4570c95d7dfd1ae4b5850cf8a95f9aae41346ceed4dfabd41b283e38.jpg)  
Low-frequency to Mid-frequency

![](images/b840b74e9373a85ee3d4c4971e5135de7680c62e0072052edc623eab476a44ce.jpg)  
Mid-frequency to High-frequency

![](images/c42f49fd90daa407bec99c2ba4e6a39b24b948828b4b05d91ea86f40b74f9e0e.jpg)  
Original High Resolution Image

![](images/91b6500c73ad591525feea7a4a8ea62657d4b9e3051974686e787e84fbd5b53e.jpg)  
Figure 4: Frequency Analysis: We show the image and its corresponding Fourier spectrum. Note how the frequency spectrum improve as we move from left to right. The Fourier spectrum of our final output closely matches that of original high resolution image.

![](images/87ae3beb2bc0325289fae26deacf389282b84e00aafa89a594802753170402f3.jpg)  
Input

![](images/f9ccb7e3a17850738ef8aa1ae3bc4e5fd56b12f4f1d8a3cc1d1dd889f9449d3a.jpg)

![](images/3c6bf5616a8eaa18bd75ed08b1e711b7455042f83a2f5fc438ffbb6db9c547a8.jpg)  
Global

![](images/471dec9838309136accb60fb983d02aad7b74adbbe144a3e523b251d080dddff.jpg)

![](images/55ad45291d496b6fac248eea3a3f7980216a0ac0c552f2dd6e84fbfdfb53d652.jpg)  
Compositional

![](images/82d118bd0cdc040cafe90f45214f1b9e427521f611eee4d0a4fb8063adfc5faf.jpg)

![](images/5c428e92734a107fbab764a938f94fc18f2931455181f4afc1f80c193bf7d851.jpg)  
Figure 5: Global vs. Compositional: Given the low-resolution input images on the left, we show high-frequency output obtained with a global nearest neighbor versus a compositional reconstruction. We visualize the correspondences associated with the compositional reconstruction on the right. We surround the reconstruction with 8 neighboring training examples, and color code pixels to denote correspondences. For example, when reconstructing the female face, forehead pixels are copied from the top-left neighbor (orange), while right-eye pixels are copied from the bottom-left neighbor (green).

et al., 2016; Liu et al., 2011; Long et al., 2014; Wei et al., 2015; Zhou et al., 2016a,b). Tappen & Liu (2012) used SIFT flow (Liu et al., 2011) to hallucinate details for image super-resolution. Zhou et al. (2016b) proposed a CNN to predict appearance flow that can be used to transfer information from input views to synthesize a new view. Kanazawa et al. (2016) generate 3D reconstructions by training a CNN to learn correspondence between object instances. Our work follows from the crucial observation of Long et al. (2014), who suggested that features from pre-trained convnets can also be used for pixel-level correspondences. In this work, we make an additional empirical observation: hypercolumn features trained for semantic segmentation learn nuances and details better than one trained for image classification. This finding helped us to establish semantic correspondences between the pixels in query and training images, and enabled us to extract high-frequency information from the training examples to synthesize a new image from a given input.

Nonparametrics: Our work closely follows data-driven approaches that make use of nearest neighbors (Efros & Leung, 1999; Freeman et al., 2002; Ren et al., 2005; Hays & Efros, 2007; Johnson et al., 2011; Shrivastava et al., 2011). Hays & Efros (2007) match a query image to 2 million training images for various tasks such as image completion. We make use of dramatically smaller training sets by allowing for compositional matches. Liu et al. (2007) propose a two-step pipeline for face hallucination where global constraints capture overall structure, and local constraints produce photorealistic local features. While they focus on the task of facial super-resolution, we address variety of synthesis applications.

Final, our compositional approach is inspired by Boiman & Irani (2006; 2007), who reconstruct a query image via compositions of training examples.

# 3 PixelNN: ONE-TO-MANY MAPPINGS

We define the problem of conditional image synthesis as follows: given an input  $x$  to be conditioned on (such as an edge map, normal depth map, or low-resolution image), synthesize a high-quality output image(s). To describe our approach, we focus on illustrative the task of image super-resolution, where the input is a low-resolution image. We assume we are given training pairs of input/outputs,

![](images/2913f5a30897abcd61aa0dd4a42e52ce549a2193f45510b2b92d434b4cce82cf.jpg)

![](images/19c376cb1d1f53221c374794c13236c0fc2aecee4a19e19ad9426fddf9362244.jpg)

![](images/0f208b757843195bce3a36dcf2de0fd10efa145f207e587de5c5355610f3ca9c.jpg)  
Input

![](images/d92622293dc5ae5e87bd56370867ee665bb32f0ed4241707481b3c30f33dd4fc.jpg)

![](images/710933c6db3d40d4793baea86fc4aa4d95f6b439c92d1e06063cdfd06af839fc.jpg)

![](images/86817076147816f0cbe73b2845b86181ac65b3e932017fa8e6c3e57d8cae8c6d.jpg)  
Intermediate

![](images/b4120924a71da33c4f7f1f6d91f429554b87d33f86765618e34f4bf1a8b59c65.jpg)

![](images/a451097bff1eef4eedfcce15b3f3c9b95fe7291e0986a621753f7239cfc67e73.jpg)

![](images/9a1c6f24081e9b336e84e908b5a6f6b341dba62b8a7d9886d6c5e0bfc36a6a04.jpg)  
Output

![](images/de61dcd106de089e89b10113ca82bf727002116506078908b5ed6047ead2632e.jpg)

![](images/61f8a223086836f5dbd39bea70088f141659ae54cf910b1ccbdb3ca1a5c6ec99.jpg)

![](images/4e18e5b754e0afffbee64de218a313c69e68b7a9cc2934da211c3b5b29ae793f.jpg)  
Original Image  
Figure 6: Edges/Normals to RGB: Our approach used for faces, cats, and dogs to generate RGB maps for a given edge/normal map as input. One output was picked from the multiple generations.

![](images/7c8b8d9e3d73d3f70770e0133b7d4f158e175d1e342a1d941e21c5aa79df367c.jpg)

![](images/c99cca63bdd7f46207be124081556994a5f12a17e9c26296fea9b48484eb6767.jpg)

![](images/e504758fb058cfa41ed389713b9ca4ed3b7c9e9f3a4a03bf1ac9e9ce00dc0040.jpg)  
Input

![](images/0370fc9643ddee643cd61cb7f962bb983dffeea99683a4a1e8c1e20a955721d6.jpg)

![](images/aff8ebb50b2c0698919fff15ed49e12c8a492b4ea7fb4bae44b9ad9c07268b06.jpg)

![](images/e3d8ccc17fc2fc1c239c3fae7556e6fac59a3af67cf01c271c581ccf33074681.jpg)  
Intermediate

![](images/c2803cff5544148a4987d2ad39511b9cc598e76175580c9ca93486c6b8d342bb.jpg)

![](images/f3cf8b0c0f05c7a526ba968662ded9afbbce9f173295ef3c3c20cb8d19fa84cb.jpg)

![](images/3f0a7e08775ddbe31531dc661d2a5c5ea1a54502578e6d151f663a7adfc1150b.jpg)  
Output

![](images/c8db8ecc828c001c9b3184355e8a1ec761b9d3d603369c97fae9ce50440f112b.jpg)

![](images/55bafffd22e769574d39091b8dab8d5203fac8f3a7d16a4e662f85a2563cdc00.jpg)

![](images/70ae4a54b673e26f510c184ba5701727b4e750dbbf561b212d5a719dcfede2d3.jpg)  
Original Image

written as  $(x_{n},y_{n})$ . The simplest approach would be formulating this task as a (nonlinear) regression problem:

$$
\min  _ {w} \left\| w \right\| ^ {2} + \sum_ {n} \left\| y _ {n} - f \left(x _ {n}; w\right) \right\| _ {l _ {2}} \tag {1}
$$

where  $f(x_{n};w)$  refers to the output of an arbitrary (possibly nonlinear) regressor parameterized with  $w$ . In our formulation, we use a fully-convolutional neural net – specifically, PixelNet (Bansal et al., 2017) – as our nonlinear regressor. For our purposes, this regressor could be any trainable black-box mapping function. But crucially, such functions generate one-to-one mappings, while our underlying thesis is that conditional image synthesis should generate many mappings from an input. By treating synthesis as a regression problem, it is well-known that outputs tend to be oversmoothed (Johnson et al., 2016). In the context of the image colorization task (where the input is a grayscale image), such outputs tend to desaturated (Larsson et al., 2016; Zhang et al., 2016b).

Frequency analysis: Let us analyze this smoothing a bit further. Predicted outputs  $f(x)$  (we drop the dependence on  $w$  to simplify notation) are particularly straightforward to analyze in the context of super-resolution (where the conditional input  $x$  is a low-resolution image). Given a low-resolution image of a face, there may exist multiple textures (e.g., wrinkles) or subtle shape cues (e.g., of local features such as noses) that could be reasonably generated as output. In practice, this set of outputs tends to be "blurred" into a single output returned by a regressor. This can be readably seen in a frequency analysis of the input, output, and original target image (Fig. 4). In general, we see that the regressor generates mid-frequencies fairly well, but fails to return much high-frequency content. We make the operational assumption that a single output suffices for mid-frequency output, but multiple outputs are required to capture the space of possible high-frequency textures.

Global/Exemplar Matching: To capture multiple possible outputs, we appeal to a classic non-parametric approaches in computer vision. We note that a simple K-nearest-neighbor (KNN) algorithm has the trivial ability to report back  $K$  outputs. However, rather than using a KNN model to return an entire image, we can use it to predict the (multiple possible) high-frequencies missing from  $f(x)$ :

$$
G l o b a l (x) = f (x) + \left(y _ {k} - f \left(x _ {k}\right)\right) \quad \text {w h e r e} \quad k = \underset {n} {\operatorname {a r g m i n}} \operatorname {D i s t} \left(f (x), f \left(x _ {n}\right)\right) \tag {2}
$$

where  $Dist$  is some distance function measuring similarity between two (mid-frequency) reconstructions. To generate multiple outputs, one can report back the  $K$  best matches from the training set instead of the overall best match.

Compositional Matching: However, the above is limited to report back high frequency images in the training set. As we previously argued, we can synthesize a much larger set of outputs by copying and pasting (high-frequency) patches from the training set. To allow for such compositional matchings, we simply match individual pixels rather than global images. Writing  $f_{i}(x)$  for the  $i^{th}$

![](images/16815a2db244e2805bc78398bb6064792a70adc9027d21a44d373b9217dddbe9.jpg)

![](images/9b03cb8fa325c1d729d6ed52d988c03aadb59e5ec4bb5ee5271c0b9e51b2442e.jpg)

![](images/56728a4561cbc71d381ecddea13476a81bb383152f5e386395a2b337e8da7424.jpg)

![](images/3ffb2d5a8b9f7faef81d3fa11863b70ebd115b77ea794ab561a2c019931c8ad5.jpg)

![](images/47afdebcad4e53fddf4d3cc300cafa112e51f84c9e42d0ab7b15d05ca9640f76.jpg)  
Input

![](images/41ac495096ef2406a5ed07b985a2492faf09443cbe721879c82a725050b5c104.jpg)  
Intermediate

![](images/4679bcb64e83e01e954b22406b085c871997434f474482e2aadefa5e6c32c8c2.jpg)  
Output

![](images/481c8d9932a7390af5d32207de5f04939fc3be68e4b2248a3e10740aa4b40e10.jpg)  
Original Image

![](images/57507a993c8a39534fef99698ae996765a23dfc79e88f032e50420981ff103d3.jpg)

![](images/79ee74dab989021469f44175981b78e1115734cd959b5b87aa6118ea1656b8f7.jpg)

![](images/29327ccb5004866cfd42b5f4f9f765e7d088fd580f76cd3df7e7bc103f246a19.jpg)

![](images/95d0b34eb66f8e7fba1fcb2c2b50aa4299092234a28f97a6c4a1696a525937eb.jpg)

![](images/1f7637a8627345081f57ceaf821dd8e4b295690ad5a85ccd36c9f04113fd2710.jpg)  
Input

![](images/44d4f2ccc80efbce17b8973de9d4f0fa147b11a37765cc1753f41c2065df63d8.jpg)  
Intermediate

![](images/c33294c3bebfeb6d942d31b234dc9612073dba28c8ddf1bbac65cfe83fa099fa.jpg)  
Output

![](images/6354a99238639db7ca5769389ec9e6e79bb7027af55f98e4b92c5b548ba1c883.jpg)  
Original Image

![](images/578711399b777ae5bc105727662ef25f4ba799189a7b43c6c413552f0f61a13f.jpg)  
Figure 7: Low-Resolution to High-Resolution: We used our approach for hallucinating  $96 \times 96$  images from an input  $12 \times 12$  low-resolution image. One output was picked from multiple generations.

![](images/2afc5a5df2375ab5b6578d802aee5109edd42999653b507ac5d7f69c0af6b8c1.jpg)

![](images/d61321c641fed8b5add86e591845b79b0e36cf5c3544e21ce4b186eda2298305.jpg)

![](images/9af89956bbc309020e2dc54508f7dbeb89c171a2ab9b5d2fad66fff4df8fac33.jpg)

![](images/399cb69608b761609368d3195eb4647d46e7e928a3043031388b8090e3977e6d.jpg)

![](images/a018d19b14001b496dec3f7e2e0225365b5fc7902ccd5f8d264708491667fc61.jpg)

![](images/9e6cf20b50e63bc39d39dc2151b0c812916b7a404b8761ae4f823e6110385bf9.jpg)

![](images/b6021b2974fec840890d20d1982f85c829ee1b281e48224f7530b417940b0f69.jpg)

![](images/901e697fb3b70d24d073cc0fdcef075c98fcaa6fe5c7c1089492c3e06b90ac78.jpg)

![](images/cfbe5e2d2ad95edff624f2ab36125ea5b6fc6b3a84b6a5f09182399fbe1b2b69.jpg)  
Input  
Figure 8: Edges-to-Shoes: Our approach used to generate multiple outputs of shoes from the edges. We picked seven distinct examples from multiple generations.

![](images/14fb606bd6546925ddfcc8b3b682082f1d6113367efb6a7d1bbfce1c6bf1aecc.jpg)  
Output-1

![](images/8f11d6c4d1f6c17acdc5659a924550adef0982484992da914e2be770599b4866.jpg)  
Output-2

![](images/58e6a61301eeb32bc724ef119b5507247bf1aea98d06c6b957a7374c92ee0c3d.jpg)  
Output-3

![](images/f5c217e8f7cd8a0e34cd05874985777af5d3aebb4fafa0ceb2ea328a3e235706.jpg)  
Output-4

![](images/786b717d575d0ffc676de66e1e425c4d94166194cc9976d1d3c93834573a04fb.jpg)  
Output-5

![](images/a8a349f83240114efc8bf069c6375528447b53bfdb251750758a6cfa169c524a.jpg)  
Output-6

![](images/124d4b8be009bb554864916a696fc77d8813f831fa64508ae526b456604c230b.jpg)  
Output-7

![](images/f4e2163c7396d21074a8e7a4207c21e8b6d2598504c5af51b8f8d7bb90c2d7d3.jpg)  
Original Image

pixel in the reconstructed image, the final composed output can be written as:

$$
C o m p _ {i} (x) = f _ {i} (x) + \left(y _ {j k} - f _ {j} \left(x _ {k}\right)\right) \quad \text {w h e r e} \quad (j, k) = \underset {m, n} {\operatorname {a r g m i n}} \operatorname {D i s t} \left(f _ {i} (x), f _ {m} \left(x _ {n}\right)\right) \tag {3}
$$

where  $y_{jk}$  refers to the output pixel  $j$  in training example  $k$ .

Distance functions: A crucial question in non-parametric matching is the choice of distance function. To compare global images, contemporary approaches tend to learn a deep embedding where similarity is preserved (Bell & Bala, 2015; Chopra et al., 2005; Long et al., 2015). Distance functions for pixels are much more subtle (3). In theory, one could also learn a metric for pixel matching, but this requires large-scale training data with dense pixel-level correspondances.

Pixel representations: Suppose we are trying to generate the left corner of an eye. If our distance function takes into account only local information around the corner, we might mistakenly match to the other eye or mouth. If our distance function takes into account only global information, then compositional matching reduces to global (exemplar) matching. Instead, we exploit the insight from previous works that different layers of a deep network tend to capture different amounts of spatial context (due to varying receptive fields) (Bansal et al., 2017; Hariharan et al., 2015; Raiko et al., 2012; Sermanet et al., 2013). Hypercolumn descriptors (Hariharan et al., 2015) aggregate such information across multiple layers into a highly accurate, multi-scale pixel representation (visualized in Fig. 3). We construct a pixel descriptor using features from conv- $\{1_2, 2_2, 3_3, 4_3, 5_3\}$  for a PixelNet model trained for semantic segmentation (on PASCAL Context (Mottaghi et al., 2014)).

To measure pixel similarity, we compute cosine distances between two descriptors. We visualize the compositional matches (and associated correspondences) in Figure. 5. Finally, Figure 6, and Figure 7 shows the output of our approach for various input modalities.

Efficient search: We have so far avoided the question of run-time for our pixel-wise NN search. A naive approach would be to exhaustively search for every pixel in the dataset but that would make the computation vary linearly with the size of dataset. On the other hand, deep generative models outpace naive NN search, which is one of the reasons for their popularity over NN search. To speed up search, we made some non-linear approximations: Given a reconstructed image  $f(x)$ , we first (1) find the global K-NN using conv-5 features and then (2) search for pixel-wise matches only in a  $T \times T$  pixel window around pixel  $i$  in this set of  $K$  images. In practice, we vary  $K$  from  $\{1, 2, .., 10\}$  and  $T$  from  $\{1, 3, 5, 10, 96\}$  and generate 72 candidate outputs for a given input. Because the size of synthesized image is  $96 \times 96$ , our search parameters include both a fully-compositional output

![](images/8a3aac13207819d839a3a6b351091ff3575fe905b0f87c343fe1e442ee890752.jpg)

![](images/598ad0af6088597c46f8a3f520ad67c8811dfb996deaaaeeedb909ad43a8e318.jpg)

![](images/efe88055b4ef9db3f074b0e8ed3e35c632c894429044b39d7ad07c3045cb2517.jpg)

![](images/b00db3f777ef07b882bed7c5249c84fdf5010c77dc1a6ff0fc85a16a3d59d377.jpg)

![](images/d45743ab3fe408e6adfad2f6858aab32f0cb66b8bef12bfded96929333bed700.jpg)

![](images/02a25f78041641ba98576cbeffef1110668d60ceb687aa82e5ad578cb735d239.jpg)

![](images/79ad86f4fbe9a0771f8a4bb6838a7c3a2576a7e49f9d8802b0398ce213bf238c.jpg)

![](images/a0225e29ac3bdef2435236d27c5450fcf7808da3b1081ad6a6a52d1edc7c1c32.jpg)

![](images/c855178604adfb6af48abda5ade6e5a7426bb6aec5d2cbd3870ff296d68ac5e8.jpg)

![](images/3969ebca970182420d2b5dcb8f5c9a0a4ee7b93c37be3aa111c7017720543b47.jpg)  
Input

![](images/8ff971781ae3d2062631bcf5178cc39def1a513eb4dadde7341165c9453c3df9.jpg)  
Output-1

![](images/8904c7853c96908bdb5e886e9c51f7857c6adaf18975438d27e21e0fea1b2031.jpg)  
Output-2

![](images/1a54a083d0fbb22531b2e3644f692322d1dcc5495030c203ad700416b19faea2.jpg)  
Output-3

![](images/413d1c2915f45fe356ec249f24ca469c1136d32af906afdbc4e099da05d227d8.jpg)  
Output-4

![](images/f73278ae67e1b5d2a751ca606f8190924489c0a3e37b857b6d07e4e897486787.jpg)  
Output-5

![](images/651c13f07a1d2904303dece91771e3a1c9b3f2981e4158b5d4848869eb8a6834.jpg)  
Output-6

![](images/7f88f2ab14f5c2b34b3b430f053dd4227d53198e3bc1f980175f8a5d6127ae6b.jpg)  
Output-7

![](images/ec79328fe39ba8bf1f24f670c43142c8061852046a21c4104966647eddb9f3f0.jpg)  
Original Image

![](images/348d206c556a80059ae70101ed3dbf9084873a3b7eac06bbfe68c529aecc1169.jpg)  
Figure 9: Edges-to-Bags: Our approach used to generate multiple outputs of bags from the edges. We picked seven distinct examples from multiple generations.

![](images/4ec520ca0c5fdd2aefbd1cac6502f5fb2d2704c3580204b2c6d4e456d1d97b9d.jpg)

![](images/1e4aa02071a85b3819fc8e5dd11fd1c28c555027b9d2e66af7126689cd8a04d2.jpg)  
Input  
$(K = 10, T = 96)$  and a fully global exemplar match  $(K = 1, T = 1)$  as candidate outputs. Figure 8, Figure 9, and Figure 10 show examples of multiple outputs generated using our approach by simply varying these parameters.

![](images/f89745f05f7632729b4a43b52cd1c527d753d61a113221614b035abc96d81df5.jpg)

![](images/1369cfce21940b19a44df66844e219d419a1ba183998c5c7fa8181851906b6dd.jpg)

![](images/fb19193a6ea139c8d00aa1494ec93700f000117d6ca45ad12ba5dcd1cfe519f5.jpg)  
Output-1

![](images/60c58f0180c13ca9bf993a18d208b070b3c383abbdd7b044cf382d96ba02c7ed.jpg)

![](images/43ae2cf534fc16cd5f20964f4306a6c9d8e4e3b26253f19e18fe3788bd24ba72.jpg)

![](images/513900294a13e4d8507cebe115de3dbcf9e3dbb9fb8447f6ad0d51afcd727450.jpg)  
Output-2

![](images/95c8d9ae1811cdfe4bdde4e68c2a48fa770097ec398dac34cc31208426f5ba26.jpg)

![](images/b7113c765e2f821a1887bac4a7b63d371e0ce53acd27a78765dab55b12bbeba1.jpg)

![](images/bdef31525b517f00d39fb7055e9f2b0e53a0e05a2d628a3dd6f019987203bb58.jpg)  
Output-3

![](images/94fa6608be8bd6139bfd0f6973f5c4607b23a10fcf4d284cb8c09fd7e9fd042d.jpg)

![](images/949cb6aa7db5db4f6ed2205c0ceddc0ed6195678390fd41939f01d4384975ce3.jpg)

![](images/6b01a7b8e69c0e68bec112a4aea88b2751ce7b8e3bc0bb5cfde0ede04cb20ea6.jpg)  
Output-4  
Figure 10: Multiple Outputs for Edges/Normals to RGB: Our approach used to generate multiple outputs of faces, cats, and dogs from the edges normals. As an example, note how the subtle details such as eyes, stripes, and whiskers of cat (left) that could not be inferred from the edge map are different in multiple generations.

![](images/dfe9b1ab3d4e957542dae29c89c28ec79c8ede25ce8e13e0a5c6e67c88de7eaa.jpg)

![](images/a8d0e278292402d075c55963efaa723cac5a8d24cba3c6f807c81975891f07bd.jpg)

![](images/ba2eccf2cb71f75206b5f1c5fcf3dd83043205f66def2d6f14396b272f442537.jpg)  
Input

![](images/4b552cfdd3f44db773fa04f789e16f40036e2f245a572042313b965fd7608b7d.jpg)

![](images/dd6ef0f70ff30ecfc0604a28ff90bb04f7f0974254c54d0e1b5debcf2c12c177.jpg)

![](images/e7bf8d3a3b9b12e96f8bc1188820d5de0f8c7ffc6a897b24512b48aa4643ec5c.jpg)  
Output-1

![](images/d628ddf8477ec89cd3cc3364ab31bb40dac49874a52ab164599208347c590315.jpg)

![](images/abb77bfb1821ee9416860b24b3784a36aedb08ca0eb1722ca308465cf4f8bab0.jpg)

![](images/df2bb1141d9687e96be892c74ab243ae22fee2542386c183ddd442c457dd7cbf.jpg)  
Output-2

![](images/5a9a593ae9f3ec22e8e36922977f2eaf2e5046366d7f2f2f535f449ed04d3373.jpg)

![](images/e7ea1cb7dcff445df872a5befd11d759157dc277be4075f5fe1c5df73d3bcec1.jpg)

![](images/031ecad3d206540b4c36436ddd2aa949ace992a12ecdc52f133823fa60b92014.jpg)  
Output-3

![](images/a2c6cfd2fe89781721393635d392e93b667a25231113a784f5e7a70a8ef3af38.jpg)

![](images/deef7e1ea866c3e9fe434a5094b7b9b7615d8e02b56f334c62df14575e151e60.jpg)

![](images/32bb13aef0c4a71cab7fa2d2e1fc8472705cdf6a1b0fb61d2b2406676ae1d81a.jpg)  
Output-4

# 4 EXPERIMENTS

We now present our findings for multiple modalities such as a low-resolution image  $(12\times 12$  image), a surface normal map, and edges/boundaries for domains such as human faces, cats, dogs, handbags, and shoes. We compare our approach both quantitatively and qualitatively with the recent work of Isola et al. (2016) that use generative adversarial networks for pixel-to-pixel translation.

Dataset: We conduct experiments for human faces, cats and dogs, shoes, and handbags using various modalities.

Human Faces We use 100,000 images from the training set of CUHK CelebA dataset (Liu et al., 2015) to train a regression model and do NN. We used the subset of test images to evaluate our approach. The images were resized to  $96 \times 96$  following Gucluturk et al. (2016).

Cats and Dogs: We use 3,686 images of cats and dogs from the Oxford-IIIT Pet dataset (Parkhi et al., 2012). Of these 3,000 images were used for training, and remaining 686 for evaluation. We used the bounding box annotation made available by Parkhi et al. (2012) to extract head of the cats and dogs.

For human faces, and cats and dogs, we used the pre-trained PixelNet (Bansal et al., 2017) to extract surface normal and edge maps. We did not do any post-processing (NMS) to the outputs of edge detection.

Shoes & Handbags: We followed Isola et al. (2016) for this setting. 50,000 training images of shoes were used from (Yu & Grauman, 2014), and 137,000 images of Amazon handbags from (Zhu et al., 2016). The edge maps for this data was computed using HED (Xie & Tu, 2015) by Isola et al. (2016).

<table><tr><td>Normals-to-RGB</td><td>Mean</td><td>Median</td><td>RMSE</td><td>11.25°</td><td>22.5°</td><td>30°</td><td>AP</td></tr><tr><td colspan="8">Human Faces</td></tr><tr><td>Pix-to-Pix</td><td>17.2</td><td>14.3</td><td>21.0</td><td>37.2</td><td>74.7</td><td>86.8</td><td>0.34</td></tr><tr><td>Pix-to-Pix (Oracle)</td><td>15.8</td><td>13.1</td><td>19.4</td><td>41.9</td><td>78.5</td><td>89.3</td><td>0.34</td></tr><tr><td>PixelNN (Rand-1)</td><td>12.8</td><td>10.4</td><td>16.0</td><td>54.2</td><td>86.6</td><td>94.1</td><td>0.38</td></tr><tr><td>PixelNN (Oracle)</td><td>10.8</td><td>8.7</td><td>13.5</td><td>63.7</td><td>91.6</td><td>96.7</td><td>0.42</td></tr><tr><td colspan="8">Pets and Dogs</td></tr><tr><td>Pix-to-Pix</td><td>14.7</td><td>12.8</td><td>17.5</td><td>42.6</td><td>82.5</td><td>92.9</td><td>0.82</td></tr><tr><td>Pix-to-Pix (Oracle)</td><td>13.2</td><td>11.4</td><td>15.7</td><td>49.2</td><td>87.1</td><td>95.3</td><td>0.85</td></tr><tr><td>PixelNN (Rand-1)</td><td>16.6</td><td>14.3</td><td>19.8</td><td>36.8</td><td>76.2</td><td>88.8</td><td>0.80</td></tr><tr><td>PixelNN (Oracle)</td><td>13.8</td><td>11.9</td><td>16.6</td><td>46.9</td><td>84.9</td><td>94.1</td><td>0.92</td></tr></table>

<table><tr><td>Edges-to-RGB</td><td>AP</td><td>Mean</td><td>Median</td><td>RMSE</td><td>11.25°</td><td>22.5°</td><td>30°</td></tr><tr><td colspan="8">Human Faces</td></tr><tr><td>Pix-to-Pix</td><td>0.35</td><td>12.1</td><td>9.6</td><td>15.5</td><td>58.1</td><td>88.1</td><td>94.7</td></tr><tr><td>Pix-to-Pix(Oracle)</td><td>0.35</td><td>11.5</td><td>9.1</td><td>14.6</td><td>61.1</td><td>89.7</td><td>95.6</td></tr><tr><td>PixelNN (Rand-1)</td><td>0.38</td><td>13.3</td><td>10.6</td><td>16.8</td><td>52.9</td><td>85.0</td><td>92.9</td></tr><tr><td>PixelNN (Oracle)</td><td>0.41</td><td>11.3</td><td>9.0</td><td>14.4</td><td>61.6</td><td>90.0</td><td>95.7</td></tr><tr><td colspan="8">Cats and Dogs</td></tr><tr><td>Pix-to-Pix</td><td>0.78</td><td>18.2</td><td>16.0</td><td>21.8</td><td>32.4</td><td>71.0</td><td>85.1</td></tr><tr><td>Pix-to-Pix (Oracle)</td><td>0.81</td><td>16.5</td><td>14.2</td><td>19.8</td><td>37.2</td><td>76.4</td><td>89.0</td></tr><tr><td>PixelNN (Rand-1)</td><td>0.77</td><td>18.9</td><td>16.4</td><td>22.5</td><td>30.3</td><td>68.9</td><td>83.5</td></tr><tr><td>PixelNN (Oracle)</td><td>0.89</td><td>16.3</td><td>14.1</td><td>19.6</td><td>37.6</td><td>77.0</td><td>89.4</td></tr></table>

Table 1: We compared our approach, PixelNN, with the GAN-based formulation of Isola et al. (2016) for human faces, and cats and dogs. We used an off-the-shelf PixelNet model trained for surface normal estimation and edge detection. We use the output from real images as ground truth surface normal and edge map respectively.

![](images/c3d96902a0c1363e47499de72e0ffe5454e09459a5499efed1624711ede067f2.jpg)  
Figure 11: Comparison of our approach with Pix-to-Pix (Isola et al., 2016).

Qualitative Evaluation: Figure 11 shows the comparison of our NN based approach (PixelNN) with Isola et al. (2016) (Pix-to-Pix).

Quantitative Evaluation: We quantitatively evaluate our approach to measure if our generated outputs for human faces, cats and dogs can be used to determine surface normal and edges from an off-the-shelf trained PixelNet (Bansal et al., 2017) model for surface normal estimation and edge detection. The outputs from the real images are considered as ground truth for evaluation as it gives an indication of how far are we from them. Somewhat similar approach is used by Isola et al. (2016) to measure their synthesized cityscape outputs and compare against the output using real world images, and Wang & Gupta (2016) for object detection evaluation.

We compute six statistics, previously used by (Bansal et al., 2016; Eigen & Fergus, 2015; Fouhey et al., 2013; Wang et al., 2015), over the angular error between the normals from a synthesized image and normals from real image to evaluate the performance - Mean, Median, RMSE,  $11.25^{\circ}$ ,  $22.5^{\circ}$ , and  $30^{\circ}$  - The first three criteria capture the mean, median, and RMSE of angular error, where lower is better. The last three criteria capture the percentage of pixels within a given angular error, where higher is better. We evaluate the edge detection performance using average precision (AP).

Table 1 quantitatively shows the performance of our approach with (Isola et al., 2016). Our approach generates multiple outputs and we do not have any direct way of ranking the outputs, therefore we show the performance using a random selection from one of 72 outputs, and an oracle selecting the best output. To do a fair comparison, we ran trained models for Pix-to-Pix (Isola et al., 2016) 72 times and used an oracle for selecting the best output as well. We observe that our approach generates better multiple outputs as performance improves significantly from a random selection to oracle as compared with Isola et al. (2016). Our approach, though based on simple NN, achieves result quantitatively and qualitatively competitive (and many times better than) with state-of-the-art models based on GANs and produce outputs close to natural images.

Controllable synthesis: Finally, NN provides a user with intuitive control over the synthesis process. We explore a simple approach based on on-the-fly pruning of the training set. Instead of matching to the entire training library, a user can specify a subset of relevant training examples. Figure 12 shows an example of controllable synthesis. A user "instructs" the system to generate an image that looks like a particular dog-breed by either denoting the subset of training exemplars

![](images/7548b6936e2e0e0ad397f3e9f3d12f9919f2e867f37fae8fea72bed587b5665b.jpg)  
Input

![](images/24258bbcc2c123b304c1a30883158d1609a2e44fce51a70b1f9758f1f6c3b26b.jpg)  
User Input: similar to Egyptian Mau

![](images/58261624801452d78c420188b84f55a5f898c98fbba5fb96dcb15427dd7c00b7.jpg)  
User Input: similar to Maine Coon

![](images/4ab75a05852975a9ba810f8457e22a2aab8cfba93f80b07b9ac7ddd468172ef2.jpg)  
User Input: similar to British Shorthair

![](images/198fae3fbbdd3eeaeb4020f213038490e71b940e698e1987f2cecf455e586de6.jpg)  
User Input: similar to Abyssinian/Egyptian Mau

![](images/1cc03b70b45e7bd96f1dbd1108d2749c522f7541e7cc6936e15cba3c4346cc25.jpg)  
User Input: similar to Yorkshire Terrier

![](images/02e1e116bda143f1d3c6f9ebb485fce9ce7afb9045125daf69d7a2b635e315bc.jpg)  
Figure 12: Controllable synthesis: We generate the output of cats given a user input from a edge map. From the edge map, we do not know what type of cat it is. A user can suggest what kind of the output they would like, and our approach can copy-paste the information.  
Input

![](images/8daca0f2089d2000b9742c64a591602f38b0be4c646e1f946e696df3db955d4a.jpg)  
PixelNN

![](images/61e8d22b6f884ae663595ea907b4f77d73e48cd7710162cffd7f1b31f54c10d7.jpg)  
Original

![](images/c70ac8fa69c569cdaa7c626ba807a2df721502a90b561e7cc01e2cac3dae85ce.jpg)  
Input

![](images/c38b9b42e5e8a053cd8d402dc4d1bed1d313d2bbd32d1564e0b6ab89298f132a.jpg)  
PixelNN

![](images/a68e5789b6f94dbd10391030e37164976cc0faf426c808c742bac29ae8e71191.jpg)  
Original

![](images/5b675424c87c16d5a59f9054fc190ababe5075bd35e573640a6720be8c893c40.jpg)  
Input

![](images/b2135623fd9cb9673240dbda61a61c486d2a49564f484fd2d48eab937af3e8e8.jpg)  
PixelNN

![](images/7772150f44c2cb7a66ae28ca13a32a4bde5c4dac3159f8a58e29f10549c640a8.jpg)  
Original

![](images/482bbdaa193177a42f55605cc1bd0e7884f1c463d4215b714a96eaa03c6fedf6.jpg)  
Input  
Figure 13: Failure Cases: We show some failure cases for different input types. Our approach mostly fails when it is not able to find suitable nearest neighbors.

![](images/6045f2a4e8c28562428eb8981f794f51560fd0ea704866f630fe3efbd08e1443.jpg)  
PixelNN

![](images/2838928c92272a6412a11e13fd1d90bb2decf9b3385c5369c20eef004b75c2be.jpg)  
Original

![](images/2d3758b3f4c966a57f88b62b3bf44a9c9f266604f13790e7a4602e13b2a8a467.jpg)  
Input

![](images/582b5c6789b37da4b6573403353cffa0ac2eab2252312249473b935ef3e0de23.jpg)  
PixelNN

![](images/f590934b2ab718a8d5c02019e5a8209cb495eb065d973bc3faa7fd4b1e7de8f6.jpg)  
Original

![](images/8760e855c7ab318711c82aeffd96c6634f8f0698b8571e6e8b2700b14f3478f1.jpg)  
Input

![](images/5b786d8ec1d56a5c0c5aea039ca3da3d365c47c589528faa7dd9d4ae11ff5463.jpg)  
PixelNN

![](images/d65941b5d8e71fbe8474dde4ee733df2a3816d7b7b6d6a64a931038301bd705f.jpg)  
Original

(e.g., through a subcategory label), or providing an image that can be used to construct an on-the-fly neighbor set.

Failure cases: Our approach mostly fails when there are no suitable NNs to extract the information from. Figure 13 shows some example failure cases of our approach. One way to deal with this problem is to do exhaustive pixel-wise NN search but that would increase the run-time to generate the output. We believe that system-level optimization such as  $\mathrm{Scanner}^1$ , may potentially be useful in improving the run-time performance for pixel-wise NNs.

# 5 DISCUSSION

We present a simple approach to image synthesis based on compositional nearest-neighbors. Our approach somewhat suggests that GANs themselves may operate in a compositional "copy-and-paste" fashion. Indeed, examining the impressive outputs of recent synthesis methods suggests that some amount of local memorization is happening. However, by making this process explicit, our system is able to naturally generate multiple outputs, while being interpretable and amenable to user constraints. An interesting byproduct of our approach is dense pixel-level correspondences. If training images are augmented with semantic label masks, these labels can be transferred using our correspondences, implying that our approach may also be useful for image analysis through label transfer (Liu et al., 2011).

# REFERENCES

Aayush Bansal, Bryan Russell, and Abhinav Gupta. Marr Revisited: 2D-3D model alignment via surface normal prediction. In CVPR, 2016.  
Aayush Bansal, Xinlei Chen, Bryan Russell, Abhinav Gupta, and Deva Ramanan. Pixelnet: Representation of the pixels, by the pixels, and for the pixels. arXiv:1702.06506, 2017.  
Sean Bell and Kavita Bala. Learning visual similarity for product design with convolutional neural networks. ACM Transactions on Graphics, 2015.  
Oren Boiman and Michal Irani. Similarity by composition. In NIPS, 2006.

Oren Boiman and Michal Irani. Detecting irregularities in images and in video. *IJCV*, 2007.  
Qifeng Chen and Vladlen Koltun. Photographic image synthesis with cascaded refinement networks. arXiv preprint arXiv:1707.09405, 2017.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. CoRR, abs/1606.03657, 2016.  
Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In CVPR, 2005.  
Christopher B Choy, JunYoung Gwak, Silvio Savarese, and Manmohan Chandraker. Universal correspondence network. In NIPS, 2016.  
Emily L. Denton, Soumith Chintala, Arthur Szlam, and Robert Fergus. Deep generative image models using a laplacian pyramid of adversarial networks. CoRR, abs/1506.05751, 2015.  
Alexei A. Efros and Thomas K. Leung. Texture synthesis by non-parametric sampling. In ICCV, 1999.  
David Eigen and Rob Fergus. Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture. In ICCV, 2015.  
David Eigen, Dilip Krishnan, and Rob Fergus. Restoring an image taken through a window covered with dirt or rain. In ICCV, 2013.  
David F. Fouhey, Abhinav Gupta, and Martial Hebert. Data-driven 3D primitives for single image understanding. In ICCV, 2013.  
William T. Freeman, Thouis R. Jones, and Egon C Pasztor. Example-based super-resolution. IEEE Comput. Graph. Appl., 22(2):56-65, March 2002. ISSN 0272-1716. doi: 10.1109/38.988747. URL http://dx.doi.org/10.1109/38.988747.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, abs/1406.2661, 2014.  
Yagmur Gucluturk, Umut GuCLU, Rob van Lier, and Marcel A. J. van Gerven. Convolutional sketch inversion. In ECCV, 2016.  
Bharath Hariharan, Pablo Arbeláez, Ross Girshick, and Jitendra Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.  
James Hays and Alexei A Efros. Scene completion using millions of photographs. ACM Transactions on Graphics, 2007.  
Aaron Hertzmann, Charles E. Jacobs, Nuria Oliver, Brian Curless, and David H. Salesin. Image analogies. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques. ACM, 2001.  
Xun Huang, Yixuan Li, Omid Poursaeed, John E. Hopcroft, and Serge J. Belongie. Stacked generative adversarial networks. CoRR, abs/1612.04357, 2016.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. arxiv, 2016.  
Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and superresolution. In ECCV, 2016.  
Micah K. Johnson, Kevin Dale, Shai Avidan, Hanspeter Pfister, William T. Freeman, and Wojciech Matusik. Cg2real: Improving the realism of computer generated images using a large collection of photographs. IEEE Transactions on Visualization and Computer Graphics, 2011.  
Angjoo Kanazawa, David W. Jacobs, and Manmohan Chandraker. Warpnet: Weakly supervised matching for single-view reconstruction. CoRR, abs/1604.05592, 2016.  
Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Learning representations for automatic colorization. In ECCV, 2016.  
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew P. Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network. CoRR, abs/1609.04802, 2016.

Ce Liu, Heung-Yeung Shum, and William T. Freeman. Face hallucination: Theory and practice. IJCV, 2007.  
Ce Liu, Jenny Yuen, and Antonio Torralba. Sift flow: Dense correspondence across scenes and its applications. IEEE Trans. Pattern Anal. Mach. Intell., 2011.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In ICCV, 2015.  
Jonathan Long, Ning Zhang, and Trevor Darrell. Do convnets learn correspondence? In NIPS, 2014.  
Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional models for semantic segmentation. In CVPR, 2015.  
Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In  $CVPR$ , 2014.  
O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In CVPR, 2012.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. CoRR, abs/1511.06434, 2015.  
Tapani Raiko, Harri Valpola, and Yann LeCun. Deep learning made easier by linear transformations in perceptrons. In AISTATS, volume 22, pp. 924-932, 2012.  
Liu Ren, Alton Patrick, Alexei A. Efros, Jessica K. Hodgins, and James M. Rehg. A data-driven approach to quantifying natural human motion. ACM Trans. Graph., 2005.  
Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and Yann LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013.  
Abhinav Shrivastava, Tomasz Malisiewicz, Abhinav Gupta, and Alexei A. Efros. Data-driven visual similarity for cross-domain image matching. ACM Transaction of Graphics (TOG), 2011.  
Marshall F. Tappen and Ce Liu. A bayesian approach to alignment-based image hallucination. In ECCV, 2012.  
Xiaolong Wang and Abhinav Gupta. Generative image modeling using style and structure adversarial networks. In ECCV, 2016.  
Xiaolong Wang, David Fouhey, and Abhinav Gupta. Designing deep networks for surface normal estimation. In CVPR, 2015.  
Lingyu Wei, Qixing Huang, Duygu Ceylan, Etienne Vouga, and Hao Li. Dense human body correspondences using convolutional networks. CoRR, abs/1511.05904, 2015.  
Jiajun Wu, Chengkai Zhang, Tianfan Xue, William T Freeman, and Joshua B Tenenbaum. Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In NIPS, 2016.  
Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In ICCV, 2015.  
A. Yu and K. Grauman. Fine-Grained Visual Comparisons with Local Learning. In CVPR, 2014.  
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaolei Huang, Xiaogang Wang, and Dimitris N. Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. CoRR, abs/1612.03242, 2016a.  
Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. ECCV, 2016b.  
Tinghui Zhou, Philipp Krahenbuhl, Mathieu Aubry, Qixing Huang, and Alexei A. Efros. Learning dense correspondence via 3d-guided cycle consistency. In CVPR, 2016a.  
Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In ECCV, 2016b.  
Jun-Yan Zhu, Philipp Krahenbihl, Eli Shechtman, and Alexei A. Efros. Generative visual manipulation on the natural image manifold. In ECCV, 2016.  
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. CoRR, abs/1703.10593, 2017.