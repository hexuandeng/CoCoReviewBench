# BASISVAE: ORTHOGONAL LATENT SPACE FOR DEEP DISENTANGLED REPRESENTATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

The variational autoencoder, one of the generative models, defines the latent space for the data representation, and uses variational inference to infer the posterior probability. Several methods have been devised to disentangle the latent space for controlling the generative model easily. However, due to the excessive constraints, the more disentangled the latent space is, the lower quality the generative model has. A disentangled generative model would allocate a single feature of the generated data to the only single latent variable. In this paper, we propose a method to decompose the latent space into basis, and reconstruct it by linear combination of the latent bases. The proposed model called BasisVAE consists of the encoder that extracts the features of data and estimates the coefficients for linear combination of the latent bases, and the decoder that reconstructs the data with the combined latent bases. In this method, a single latent basis is subject to change in a single generative factor, and relatively invariant to the changes in other factors. It maintains the performance while relaxing the constraint for disentanglement on a basis, as we no longer need to decompose latent space on a standard basis. Experiments on the well-known benchmark datasets of MNIST, 3DFaces and CelebA demonstrate the efficacy of the proposed method, compared to other state-of-the-art methods. The proposed model not only defines the latent space to be separated by the generative factors, but also shows the better quality of the generated and reconstructed images. The disentangled representation is verified with the generated images and the simple classifier trained on the output of the encoder.

# 1 INTRODUCTION

The proper choice of data representation is highly correlated with the difficulty of task learning for a given machine learning approach (Higgins et al., 2017; Kim et al., 2018; Kim & Cho, 2018; 2019). Using a representation appropriate to specific task and data domain can significantly improve the robustness and successful learning of the model (Kim et al., 2018; Kim & Cho, 2018; 2019; Bengio et al., 2013). In particular, disentangled representation is useful when dealing with data with various features, and can be effective for a large variety of domains and tasks (Bengio et al., 2013; Ridgeway, 2016). A latent space is disentangled if single latent units are subject to changes in single generative factors, and relatively invariant to changes in other factors (Bengio et al., 2013). For example, a generative model trained on a dataset of facial images learns independent latent units subject to single independent generative factors such as hair color, gender, and emotion. We define the disentangled representation using equation (1). A change of single generative factor is consistent to the change of single coefficient  $c_{i}$ , but not to  $c_{j}$  for  $i \neq j$ .

$$
\mathbf {z} = \sum_ {i} c _ {i} \mathbf {e} _ {\mathbf {i}} \tag {1}
$$

where  $\mathbf{z} \in \mathbb{R}^Z$  is a latent variable,  $\mathbf{e_i} \in \mathbb{R}^Z$  is a standard unit vector, and  $c_i \in \mathbb{R}$  is a coefficient. Disentangled representation can be useful in several machine learning tasks including transfer learning and zero-shot learning (Lake et al., 2017). Moreover, unlike most representation learning algorithms, disentangled representation can be interpreted because they are consistent with the variability of the data (Dupont, 2018). The variational autoencoder (VAE) is used to define the latent space by approximating the posterior distribution with approximation as follows (Kingma & Welling, 2013).

$$
\log p _ {\theta} (x) \geq - \mathcal {D} _ {K L} \left[ q _ {\phi} (z | x) \| p _ {\theta} (z) \right] + \mathbb {E} _ {q _ {\phi} (z | x)} [ \log p _ {\theta} (x | z) ] \tag {2}
$$

![](images/1850d7202e2197b4f09f523eb5b5950e7b207210c8e9edab5a1c7f7baf4220ef.jpg)  
Figure 1: In the vanilla VAE model, an interpolation experiment results in only changing the single generative factor only for (a) gender and (b) skin color.

where  $\mathbb{D}_K L$  is Kullback-Leibler divergence,  $q_{\phi}(z|x)$  is a posterior distribution inferred by encoder,  $p_{\theta}(z)$  is a prior distribution, and  $p(x|z)$  is a likelihood or decoder. Since the VAEs are powerful to define the latent space, it is often used for disentangled representation learning (Higgins et al., 2017; Dupont, 2018). However, in most cases, the quality of the generated data is relatively low because of the added constraints to the loss function (Kim & Cho, 2019). This is because the scale of the latent variable to represent the generative factor drops from  $\mathbb{R}^Z$  to  $\mathbb{R}^1$  when the generative model is  $f: \mathcal{Z} \subset \mathbb{R}^Z \to \mathcal{X} \subset \mathbb{R}^X$ , where  $Z$  and  $X$  are the dimensions of the latent space and data, respectively. Several researchers have studied for disentanglement, but the trade-off with performance has not been considered (Higgins et al., 2017; Dupont, 2018; Chen et al., 2016). In a vanilla VAE, one generative factor changes in the direction of element in a non-standard basis as shown in Fig. 1. With this result, if the latent space can be decomposed with basis  $\mathcal{B} = \{\mathbf{b}_1, \dots, \mathbf{b}_n\}$  to denote latent variable  $z$  as  $\Sigma_i c_i \mathbf{b}_i$ , the single generative factor is associated with the single coefficient. Therefore, disentangled representation learning is achieved by the following two constrains:

1. Each coefficient is subject to change in single generative factor, and relatively invariant to the changes in other factors.  
2. The generative model is trained to make the basis of the latent space as a standard basis  $\mathcal{B}_0 = \{\mathbf{e_1},\dots ,\mathbf{e_n}\}$

In this paper, we focus on the first constraint to formulate disentangled representation without the second constraint. The rest of this paper is organized as follows. In Section 2, we introduce the research for learning disentangled representation. The work we have done in this paper and the proposed model are presented in Section 3 and the evaluation is discussed in Section 4. Section 5 presents a summary and some future works.

# 2 RELATED WORKS

Many studies have been conducted to learn a data representation. It is used on various applications from feature extraction to dimension reduction. Approaches are divided into two categories: conventional methods and deep learning models. Principal component analysis (PCA) or independent component analysis (ICA) are well-known methods to extract features and reduce the size of data (Smith, 2002; Hyvarinen & Oja, 2000). Dictionary learning develops a set (dictionary) of representative elements from the data such that each datum can be expressed as a weighted sum of the atoms. The elements and weights can be found by minimizing the error with L1 regularization on the weights to enable sparsity (Mairal et al., 2009; Lee et al., 2007; Aharon et al., 2006). They adopted the methods such as basis on linear algebra that defines the materials and mixes them appropriately to represent the data. In another approach, Kingma and Welling proposed auto-encoding variational Bayes to approximate the posterior distribution (Kingma & Welling, 2013). Radford et al. showed that the walking in the latent space resulted in semantic changes (Radford et al., 2015). Oord et al. proposed a vector-quantized VAE to learn a discrete latent representation (van den Oord & Vinyals, 2017). It is not disentangled, but somewhat with general representation to prevent posterior collapse (i.e., violation of the first constraint). Chen et al. presented InfoGAN that learned interpretable rep

Figure 2: The architecture of the proposed model, BasisVAE.  
![](images/cc9a5413fa04a0ad14dd90f06ebd3949685a50a32b45a8ceaf414d28337e98d4.jpg)  
E: encoder  
D: decoder  
$f(x)\in \mathbb{R}^{n_x}$  
coefficients for basis  
$\Sigma_{f(x)} \in \mathbb{R}^{n_x}$ : vari  
$\mathcal{M}_{\mathcal{B}} \in \mathbb{R}^{Z \times n_x}$ : basis matrix

resentation by using mutual information (Chen et al., 2016). Higgins et al. introduced an adjustable hyperparameter  $\beta$  that balanced latent channel capacity and independence constraints with reconstruction accuracy (betaVAE) (Higgins et al., 2017). Dupont improved betaVAE by using a joint distribution of continuous and discrete latent variables (Dupont, 2018). Deep learning frameworks showed promise in disentangling factors of variation, but there was a degrade in the quality of the generated data due to the trade-off. In this paper, we propose a method to learn disentangled representation while maintaining the quality of the generated data by learning materials and weights for data representation like dictionary learning and disentangling factors like deep learning approach.

# 3 THE PROPOSED METHOD

The architecture of a proposed model that constructs disentangled representation (i.e., the association of a single basis element with a single generative factor) with a coefficient of basis element rather than a latent unit is shown in Fig. 2. Unlike the conventional VAE that outputs the mean and variance of the latent space expressed as a normal distribution, the encoder of BasisVAE outputs the coefficient  $f(\mathbf{x}) = \mathbf{c}$  associated with elements of the basis B. The latent variable  $\mathbf{z}$  is sampled from the Gaussian distribution  $\mathcal{N}(\mathcal{M}_{\mathcal{B}} \cdot f(\mathbf{x}), \Sigma_{f(\mathbf{x})})$ , where operator  $\cdot$  means matrix multiplication,  $\Sigma_{f(\mathbf{x})}$  is a variance computed by encoder, and  $\mathcal{M}_{\mathcal{B}} = [\mathbf{b_1} | \dots | \mathbf{b_n}]$  is a matrix form of bases. The theoretical background, loss function, and algorithms of the proposed model are discussed in detail in the following sections.

# 3.1 LATENT SPACE DECOMPOSITION

For the first constraint mentioned in the introduction, it is proved in Theorem 1 that the latent space can be decomposed as a set of single basis elements that are subject to a single generative factor. It is enough to show that the latent variable  $\mathbf{z}$  in the equation (2) can be decomposed into latent variables  $\mathbf{z}_1, \dots, \mathbf{z}_n$ , called latent basis, associated with a single generative factor, not into latent units, and the evidence lower bound (ELBO) is maintained. Let  $n_x$  be the number of features that data  $x$  has and  $\mathbf{z}_1, \dots, \mathbf{z}_{\mathbf{n}_x}$  be the corresponding independent latent variables. Theorem 1. Let the latent variable  $\mathbf{z}$  in ELBO be decomposed into independent latent variables  $\mathbf{z}_1, \dots, \mathbf{z}_{\mathbf{n}_x}$  associated with a single generative factor such that  $p(\mathbf{z}) = \Pi_i p(\mathbf{z}_i)$ , then the ELBO with respect to  $\mathbf{z}$  is equal to the average values of the ELBO with respect to  $\mathbf{z}_i$ . The  $q_{\phi}(z|x)$  which the expectation value in equation (1) with respect to should be modified as the form of  $q_{\phi}(z_i|x)_i$ . We prove Lemma 1 in order to prove Theorem 1. Lemma 1. If  $z_1, \dots, z_n$  are independent and  $L$  is a linear operator,  $\mathbb{E}_{z_1, \dots, z_n}[L(z_1, \dots, z_n)] = \Sigma_i L(E_{z_i}[z_i])$  where  $a_i$  is a coefficient of  $z_i$  in  $L$ . Proof. We just show it in the case of  $n = 2$ .

$$
\begin{array}{l} \mathbb {E} _ {z _ {1}, z _ {2}} \left[ L \left(z _ {1}, z _ {2}\right) \right] \\ = \int_ {z _ {1}} \int_ {z _ {2}} p (z _ {1}, z _ {2}) \left(a _ {1} z _ {1} + a _ {2} z _ {2}\right) d z _ {2} d z _ {1} \\ = \int_ {z _ {1}} p \left(z _ {1}\right) a _ {1} z _ {1} d z _ {1} + \int_ {z _ {2}} p \left(z _ {2}\right) a _ {2} z _ {2} d z _ {2} \tag {3} \\ = a _ {1} \mathbb {E} _ {z _ {1}} \left(z _ {1}\right) + a _ {2} \mathbb {E} _ {z _ {2}} \left(z _ {2}\right) \\ = L \left(\mathbb {E} _ {z _ {1}} \left[ z _ {1} \right], \mathbb {E} _ {z _ {2}} \left[ z _ {2} \right]\right) \\ \end{array}
$$

![](images/de452597de1f0dbfd968c5739719179ab58e6c50a83b96f5815aec893c6a2c3d.jpg)  
Figure 3: The visualization of (a) the general latent space, (b) the disentangled latent space, and (c) latent space of the proposed model with two coordinates.

![](images/fc587f71c5dccd0942d2afc1bdf4014f15aaea7e068afb03a9f139401b2788aa.jpg)

![](images/48aaeadca4f3ba596955c19c8bf37a6c03418554c8f73da04a3228346ebda41d.jpg)

Proof of Theorem 1. Since the latent variable  $z$  can be decomposed as independent latent variable  $\mathbf{z}_1, \dots, \mathbf{z}_{n_x}$ , equation (4) is derived from equation (2).

$$
\log p _ {\theta} (x) \geq - \mathcal {D} _ {K L} \left[ q _ {\phi} \left(z _ {1}, \dots , z _ {n _ {x}} | x\right) \| p _ {\theta} \left(z _ {1}, \dots , z _ {n _ {x}}\right) \right] \tag {4}
$$

As  $z_{1},\dots ,z_{n_{x}}$  are independent,

$$
\begin{array}{l} \log p _ {\theta} (x) \geq - \mathcal {D} _ {K L} \left[ q _ {\phi} \left(z _ {1} \mid z _ {2}, \dots , z _ {n _ {x}}, x\right) \dots q _ {\phi} \left(z _ {n _ {x}} \mid x\right) \| p _ {\theta} \left(z _ {1}\right) \dots p _ {\theta} \left(z _ {n _ {x}}\right) \right] \tag {5} \\ + \mathbb {E} _ {q _ {\phi} (z _ {1} | z _ {2}, \dots , z _ {n _ {x}}, x) \dots q _ {\phi} (z _ {n _ {x}} | x)} [ \log p _ {\theta} (x | z _ {1}, \dots , z _ {n _ {x}}) ] \\ \end{array}
$$

$$
\begin{array}{l} \log p _ {\theta} (x) \geq - \mathcal {D} _ {K L} \left[ \Pi_ {i} ^ {n _ {x}} q _ {\phi} \left(z _ {i} \mid x\right) \| \Pi_ {i} ^ {n _ {x}} p _ {\theta} \left(z _ {i}\right) \right] \tag {6} \\ + \mathbb {E} _ {\Pi_ {i} ^ {n _ {x}} q _ {\phi} (z _ {i} | x)} [ \log [ \Pi_ {i} ^ {n _ {x}} p _ {\theta} (x | z _ {i}) / p ^ {n _ {x} - 1} (x) ] ] \\ \end{array}
$$

$$
\begin{array}{l} \log p _ {\theta} (x) \geq - \mathcal {D} _ {K L} \left[ \Pi_ {i} ^ {n _ {x}} q _ {\phi} \left(z _ {i} \mid x\right) \| \Pi_ {i} ^ {n _ {x}} p _ {\theta} \left(z _ {i}\right) \right] \tag {7} \\ + \mathbb {E} _ {\Pi_ {i} ^ {n _ {x}} q _ {\phi} (z _ {i} | x)} [ \log \Pi_ {i} ^ {n _ {x}} p _ {\theta} (x | z _ {i}) ] - (n _ {x} - 1) p (x) \\ \end{array}
$$

By Lemma 1, we can separate the expectation as follows:

$$
\log p _ {\theta} (x) \geq \frac {1}{n _ {x}} \Sigma_ {i} ^ {n _ {x}} \left[ \mathbb {E} _ {q _ {\phi} \left(z _ {i} \mid x\right)} \left[ \log p _ {\theta} \left(x \mid z _ {i}\right) \right] - \mathcal {D} _ {K L} \left[ q _ {\phi} \left(z _ {i} \mid x\right) \| p _ {\theta} \left(z _ {i}\right) \right] \right] \square \tag {8}
$$

As a result of equation (8), we can say that the first term of RHS is the reconstruction error, and the second term associates the latent space with the data which has  $i$ -feature represented as  $\mathbf{z_i}$ . In the next section, BasisVAE is proposed to maximize the lower bound shown in equation (8), with  $\mathbf{z_1}, \dots, \mathbf{z_{n_x}}$  becoming independent.

# 3.2 BASISVAE

We set  $n_x$  as the number of features existing in the set  $\mathcal{X}$  of data and latent variable  $\mathbf{z}$  as linear combination of  $\mathbf{z}_1, \dots, \mathbf{z}_{\mathbf{n}_x}$ . By the assumption,  $\mathbf{z}_1, \dots, \mathbf{z}_{\mathbf{n}_x}$  are independent, and for any  $\mathbf{z}$ ,  $\mathbf{z} = \sum_i c_i \mathbf{z}_i$  so that the set  $\mathcal{B} = \{\mathbf{z}_1, \dots, \mathbf{z}_{\mathbf{n}_x}\}$  is the basis of the latent space. For the sake of convenience, let the elements of  $\mathcal{B}$  be denoted as  $\mathbf{b}_1, \dots, \mathbf{b}_{\mathbf{n}_x}$ . The output of the encoder is coefficients  $c_1, \dots, c_{n_x}$  because, otherwise, the model is not different with vanilla VAE and cannot achieve the disentangled representation. The goal of the previous research is to change the latent space from (a) to (b) in Fig. 3, but the proposed method changes from (a) to (c). It maintains the area responsible for a single generative factor but achieves disentangled representation using coefficient  $c_i$ . In this method, the model can learn a disentangled representation with coefficients (constraint 1). Besides, it does not have to define the basis of latent space as standard basis (without constraint 2). The direction of the latent basis  $\mathbf{b}_i$  is not limited to two (the latent unit becomes larger or smaller), but is set in all directions in  $\mathbb{R}^Z$ , thus representing the information in various ways. We train the encoder so that  $c_i = 1$  and  $c_j = 0$  if the input data has  $i$ -feature and no  $j$ -feature. The latent variable  $\mathbf{z}$  is sampled from the normal distribution  $\mathcal{N}(\mathcal{M}_{\mathcal{B}} \cdot f(\mathbf{x}), \Sigma_{f(\mathbf{x})})$  having the linear combination  $\Sigma_i c_i \mathbf{z}_i$  as mean, and  $\Sigma_{f(\mathbf{x})}$  as variance, where  $\mathcal{M}_{\mathcal{B}} = [\mathbf{b}_1 | \dots | \mathbf{b}_n]$  and  $f(\mathbf{x}) = (c_1, \dots, c_n)$ . The decoder is trained to reconstruct the data  $\mathbf{x}$  with  $\mathbf{z}$ . Algorithm 1 describes the process of defining the latent space through the encoder and reconstructing the data through the decoder. Three losses are defined to train the latent space in the proposed process: 1) reconstruction loss  $\mathcal{L}_{recon}$ , 2) inference loss  $\mathcal{L}_{KL}$ , and 3)

Algorithm 1 The process to define the latent space and reconstruct the data  
1: Input: Data  $\{x_i\}_{i = 1}^N$  encoder  $q_{\phi}$  decoder  $p_{\theta}$  , and basis matrix  $\mathcal{M}_{\mathcal{B}}$    
2: Output: trained encoder  $q_{\phi}$  , trained decoder  $p_{\theta}$  , and trained basis matrix  $\mathcal{M}_{\mathcal{B}}$    
3: Initialize  $q_{\phi},p_{\theta}$    
4: for epochs do   
5: for batches do   
6: Sample x from the dataset   
7:  $c\gets q_{\phi}(x)$    
8: Sample  $z$  from  $\mathcal{N}(\mathcal{M}_{\mathcal{B}}\cdot c^{T},\Sigma_{f(x)})$    
9:  $\hat{x}\gets p_{\theta}(z)$    
10: Update  $q_{\phi},p_{\theta},\mathcal{M}_{\mathcal{B}}$  with equation (12)   
11: end for   
12: end for   
13: return  $q_{\phi},p_{\theta},\mathcal{M}_{\mathcal{B}}$

basis loss  $\mathcal{L}_{\mathcal{B}}$  as follows.

$$
\mathcal {L} _ {\text {r e c o n}} = l (x, G (\mathcal {C} (z | x))), \mathcal {C} (z | x) \sim \mathcal {N} \left(\mathcal {M} _ {\mathcal {B}} \cdot f (x), \Sigma_ {f (x)}\right) \tag {9}
$$

$$
\mathcal {L} _ {K L} = \mathcal {D} _ {K L} [ \mathcal {N} (f (x), \Sigma_ {f (x)}) \| p _ {\theta} (z) ] \tag {10}
$$

$$
\mathcal {L} _ {\mathcal {B}} = \left\| \mathcal {M} _ {\mathcal {B}} ^ {T} \mathcal {M} _ {\mathcal {B}} - I \right\| _ {2} ^ {2} \tag {11}
$$

where  $l$  is the binary function for measuring the reconstruction error,  $f(x)$  is the output of the encoder, and  $\mathcal{M}_{\mathcal{B}} = [b_1|\dots |b_{n_x}]$  is the basis matrix. Since the elements in  $\mathcal{M}_{\mathcal{B}}$  have to be independent, i.e.,  $b_{i}\cdot b_{j} = 0$  if  $i\neq j$ , and  $b_{i}\cdot b_{i} = 1$ ,  $\mathcal{M}_{\mathcal{B}}^{T}\mathcal{M}_{\mathcal{B}}$  should be identity matrix  $I$  during training. The total loss of the proposed model is as follows.

$$
\mathcal {L} = \alpha \mathcal {L} _ {\text {r e c o n}} + \beta \mathcal {L} _ {K L} + \gamma \mathcal {L} _ {\mathcal {B}} \tag {12}
$$

where  $\alpha, \beta$ , and  $\gamma$  are the hyperparameters for balancing between the losses.

# 4 EXPERIMENTS

# 4.1 DATASET AND EXPERIMENTAL SETTINGS

To verify the performance of the proposed model, we use the MNIST, 3DFaces and CelebA datasets (LeCun et al., 1998; Liu et al., 2015; Paysan et al., 2009). The CelebA is a dataset with large-scale face attributes. We crop the initial  $178 \times 218$  size to  $138 \times 138$  and resize them as  $128 \times 128$ . There are total 202,599 face images and we use 162,769 images as training data and the rest as test data. The pixel values are normalized between 0 and 1. The weights of the model are initialized with the method proposed by Glorot and Bengio (Glorot & Bengio, 2010). The encoder consists of eight convolutional layers whose filter size is  $5 \times 5$  with LeakyReLU activation function followed by dropout and batch normalization layer (Maas et al., 2013; Srivastava et al., 2014; Ioffe & Szegedy, 2015). The decoder is composed of four convolutional layers and 4 deconvolutional layers with ReLU activation function followed by several layers like encoder (Nair & Hinton, 2010).  $\alpha$ ,  $\beta$ , and  $\gamma$  are set as 0.0004, 1, and 0.1, respectively. The binary function for measuring the reconstruction error is set as Bojanowski et al. did (Bojanowski et al., 2017). BasisVAE is trained for 100 epochs with 100 batch size. The optimizer used to train the model is Adam proposed by Kingma and Ba (Kingma & Ba, 2014).

# 4.2 GENERATED IMAGES

To verify the performance of the proposed model, the performance of BasisVAE is compared with the performance of vanilla VAE and betaVAE, which have the same structure, but different output of encoder to the proposed model, in three aspects: Reconstruction, random generation, and disentanglement.

![](images/854e1205000d64df8484590156b5867546d4ae15f429b7ddffb1d4b6f78c2adb.jpg)  
Figure 4: (Top) The original images and (bottom) the reconstructed images on (a) MNIST and (b) CelebA datasets. Appendix A shows more generated images.

![](images/5f043f498bc672589530712e2a913e47990897377e9e507e2b502e99a4537797.jpg)  
Figure 5: The generated images of (a) MNIST and (b) CelebA. In the MNIST dataset, the generated data is organized in each row by class.

# 4.2.1 RECONSTRUCTION

We evaluate the reconstruction performance of BasisVAE with MNIST and CelebA datasets. Fig. 4 shows the reconstructed images for the original images. In Table 1, we show the structural similarity (SSIM) and peak signal-to-noise ratio (PSNR) values together with the comparison model for the quantitative evaluation of the performance. The experiment is repeated 10 times to compute the SSIM and PSNR values between the actual images and the generated images by the model trained on CelebA dataset. The results of the t-test show that the performance of the BasisVAE is superior to that of the other models statistically.

# 4.2.2 RANDOM GENERATION

The generated data by BasisVAE learned with MNIST and CelebA are illustrated in Fig. 5. Frechet inception distance is used to evaluate the quality of the generated images (Heusel et al., 2017) as shown in Table 2. The p-value obtained from the t-test was less than 0.05, indicating a statistically significant difference in performance.

# 4.2.3 GENERATION FROM BASIS

We conduct an experiment to verify that the basis learned through BasisVAE has actually influenced the construction of the disentangled representation. BasisVAE generates the images with basis  $\mathbf{b_i}$  by setting the coefficients as  $c_{i} = 1$  and  $c_{j}$  for  $i\neq j$ . The feature corresponding to each basis  $\mathbf{b_i}$  is shown on the Figs. 6 and 7. We also use a 3DFaces dataset as well as CelebA dataset to identify the

Table 1: The results of evaluating the reconstruction performance with SSIM and PSNR.  

<table><tr><td colspan="2"></td><td>VAE</td><td>β VAE</td><td>BasisVAE</td></tr><tr><td rowspan="3">SSIM</td><td>Average</td><td>0.7071</td><td>0.6142</td><td>0.7965</td></tr><tr><td>Std. dev.</td><td>6.0×10-6</td><td>6.9×10-6</td><td>4.6×10-6</td></tr><tr><td>p-value</td><td>2.4×10-25</td><td>1.3×10-30</td><td>-</td></tr><tr><td rowspan="3">PSNR</td><td>Average</td><td>64.989</td><td>61.512</td><td>67.882</td></tr><tr><td>Std. dev.</td><td>0.004</td><td>0.004</td><td>0.004</td></tr><tr><td>p-value</td><td>2.1×1026</td><td>1.1×10-32</td><td>-</td></tr></table>

Table 2: Comparison of image generation quality by FID score.  

<table><tr><td></td><td>VAE</td><td>βVAE</td><td>BasisVAE</td></tr><tr><td>Average</td><td>112.883</td><td>168.239</td><td>78.449</td></tr><tr><td>Std. dev.</td><td>1.309</td><td>1.964</td><td>2.877</td></tr><tr><td>p-value</td><td>1.49×10-21</td><td>1.85×10-28</td><td>-</td></tr></table>

![](images/021dcec3f999856d7ffc67d1cfef18c560ce64299248adb989cb1290e394c2cd.jpg)  
Figure 6: The images generated from a single basis. The corresponding feature is shown above the image.

![](images/0f94aa9ce2b0bd37454d95f9c7434f76aebe581f4b8e990bcf52157a74393ebc.jpg)  
Figure 7: The images generated from a single basis. The value of the coefficient is linearly changed along the row. The corresponding characteristics are shown in the left of the images.

![](images/e00c0c6d8a8931b8c16d6f3ebfc5406ae78035b4f049e1f251a7ad90c925b4a8.jpg)  
Figure 8: The value of the coefficient is linearly changed along the row. The corresponding characteristics are shown in the side of the images. Appendix A shows more generated images.

![](images/b81fc1553a0d1760c6c780781f1b67b4dbd369981af8fe4af4b02833ef2d5092.jpg)  
Figure 9: Randomly generated images with the coefficient  $c_{i}$  for  $\mathbf{b_i}$  fixed as 1. According to the basis element, the images reflecting the corresponding feature are generated. Appendix A shows more generated images.

characteristic change with coefficient size. As shown in Figs. 8 and 9, we can see the basis element corresponding to azimuth and lighting in 3DFaces dataset and to hair color, bags under eyes, bald, etc. in CelebA dataset. To show that single basis element is subject to a single generative factor, we randomly generate an image with the coefficient  $c_{i}$  for  $\mathbf{b}_{\mathrm{i}}$  fixed as 1, as shown in Fig. 9. To quantitatively evaluate the disentangled representation, logistic regression is trained to classify the features by inputting the output of the encoder into itself. The more disentangled the latent space is, the higher accuracy the model achieves. We train about 40 binary classifiers for 40 classes of CelebA dataset, and the average accuracy is shown in Table 3.

# 5 CONCLUSION

In this paper, we have formulated the disentangled representation learning with two constraints. By proving the Theorem, it is shown that the latent space can be decomposed as independent latent variables associated with single generative factor. We have shown that the proposed BasisVAE constructs disentangled representation without the second constraint by constructing the basis of the latent space. Furthermore, BasisVAE outperforms the vanilla VAE and  $\beta$ VAE in both performance and disentanglement. Since our method can be applied to other VAEs by changing the output of the encoder as coefficients for basis element and adding loss  $\mathcal{L}_{\mathcal{B}}$ , we will verify the versatility and validity by applying it to other models. The performance of the proposed model will be evaluated with other well-known benchmark datasets such as CIFAR10, 3DFaces, and ImageNet. In addition, we will achieve the higher quality of the generated data and interpretability of the latent space by constructing disentangled latent space in generative adversarial network.

Table 3: Results of classification using the output of the encoder. The logistic regression model for each class is trained to classify the one class. Appendix B shows more details in the numerical results for each attributes.  

<table><tr><td></td><td>VAE</td><td>βVAE</td><td>BasisVAE</td></tr><tr><td>Average</td><td>81.90</td><td>84.44</td><td>89.82</td></tr><tr><td>Std. dev.</td><td>0.015</td><td>0.009</td><td>0.005</td></tr><tr><td>p-value</td><td>0.001</td><td>0.004</td><td>-</td></tr></table>

# REFERENCES

M. Aharon, M. Elad, and A. Bruckstein. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Trans. on Signal Processing, 54(11):4311-4322, 2006.  
Y. Bengio, A. Courville, and P. Vincent. Representation learning: A review and new perspectives. IEEE Trans. on Pattern Analysis and Machine Intelligence, 35(8):1798-1828, 2013.  
P. Bojanowski, A. Joulin, D. Lopez-Paz, and A. Szlam. Optimizing the latent space of generative networks. arXiv preprint arXiv:1707.05776, 2017.  
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2172-2180, 2016.  
E. Dupont. Learning disentangled joint continuous and discrete representations. In Advances in Neural Information Processing Systems, pp. 710-720, 2018.  
X. Glorot and Y. Bengio. Understanding the difficulty of training deep feedforward neural networks. In Int. Conf. on Artificial Intelligence and Statistics, pp. 249-256, 2010.  
M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, pp. 6626-6637, 2017.  
I. Higgins, L. Matthew, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner. Beta-vae: Learning basic visual concepts with a constrained variational framework. Int. Conf. on Learning Representation, 2(5):6, 2017.  
A Hyvarinen and E Oja. Independent component analysis: algorithms and applications. Neural networks, 13(4-5):411-430, 2000.  
S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
J.Y. Kim and S.B. Cho. Detecting intrusive malware with a hybrid generative deep learning model. In Int. Conf. on Intelligent Data Engineering and Automated Learning, pp. 499-507. Springer, 2018.  
J.Y. Kim and S.B. Cho. Electric energy consumption prediction by deep learning with state explainable autoencoder. *Energies*, 12(4):739, 2019.  
J.Y. Kim, S.J. Bu, and S.B. Cho. Zero-day malware detection using transferred generative adversarial networks based on deep autoencoders. Information Sciences, 460:83-102, 2018.  
D.P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
D.P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
B.M. Lake, T.D. Ullman, J.B. Tenenbaum, and S.J. Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40, 2017.  
Y LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proc. of the IEEE, 86(11):2278-2324, 1998.  
H Lee, A Battle, R Raina, and A.Y. Ng. Efficient sparse coding algorithms. In Advances in Neural Information Processing Systems, pp. 801-808, 2007.  
Z. Liu, P. Luo, X. Wang, and X. Tang. Deep learning face attributes in the wild. In IEEE Int. Conf. on Computer Vision, pp. 3730-3738, 2015.  
A.L. Maas, A.Y. Hannun, and A.Y. Ng. Rectifier nonlinearities improve neural network acoustic models. Int. Conf. on Machine Learning, 30(1):3, 2013.

J Mairal, J Ponce, G Sapiro, A Zisserman, and F.R. Bach. Supervised dictionary learning. In Advances in Neural Information Processing Systems, pp. 1033-1040, 2009.  
V. Nair and G. Hinton. Rectified linear units improve restricted boltzmann machines. In Int. Conf. on Machine Learning, pp. 807-814, 2010.  
P. Paysan, R. Knothe, B. Amberg, S. Romdhani, and T. Vetter. A 3d face model for pose and illumination invariant face recognition. In Int. Conf. on Advanced Video and Signal Based Surveillance, pp. 296-301. IEEE, 2009.  
A Radford, L Metz, and S Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
K. Ridgeway. A survey of inductive biases for factorial representation-learning. arXiv preprint arXiv:1612.05299, 2016.  
L.I. Smith. A tutorial on principal components analysis. Univ. of Otago, Technical Report, 2002.  
N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15 (1):1929-1958, 2014.  
A van den Oord and O Vinyals. Neural discrete representation learning. In Advances in Neural Information Processing Systems, pp. 6306-6315, 2017.

![](images/635ec68ebd05fd5c1fda058f774dba5b28d455a3a3459cbc373b0695be6d5568.jpg)  
A APPENDIX A: MORE IMAGES GENERATED  
Figure 10: The generated CelebA images. Image blur is less than conventional VAE.

![](images/1899a1d6531e27bda84363e543010af32ef58b7a2b7b92fa5bfff125c9437eee.jpg)  
Figure 11: The value of the coefficient is linearly changed along the row. The corresponding characteristics are shown in the side of the images.

![](images/e6dba3e80c793c3530619599df3d7b8f615866b5b483ac86217dcf755e33a2b1.jpg)  
Black hair

![](images/fba42cdb8e6ec4edabaed5a7826110408059434d5f65c25f4acf6e3b51894c86.jpg)  
Bags under eyes

![](images/35818285f95d31bc4d2dbbc804ee5b1265ab29845cfb74c502fdbf96ff19a8e8.jpg)  
Bald

![](images/f65b94eacd65f72e3b69d7ef90cf502e805c266c7beda91f5d00ac5f095f72a6.jpg)  
Bangs

![](images/b40a41a5acb6031607245810058483ad8145c3474d65df8f61a34d99ee937d93.jpg)  
Pale skin  
Figure 12: Randomly generated images with the coefficient  $c_{i}$  for  $\mathbf{b_i}$  fixed as 1. According to the basis element, images reflecting the corresponding feature are generated.

![](images/b00292a717a34cec14d94a4446f8c17cafdaac48ac6ce5ef756414b8254a41a9.jpg)  
Mustache
