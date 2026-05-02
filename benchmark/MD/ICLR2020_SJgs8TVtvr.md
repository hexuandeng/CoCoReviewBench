# MIXTURE-OF-EXPERTS VARIATIONAL AUTOENCODER FOR CLUSTERING AND GENERATING FROM SIMILARITY-BASED REPRESENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Clustering high-dimensional data, such as images or biological measurements, is a long-standing problem and has been studied extensively. Recently, Deep Clustering gained popularity due to the non-linearity of neural networks, which allows for flexibility in fitting the specific peculiarities of complex data. Here we introduce the Mixture-of-Experts Similarity Variational Autoencoder (MoE-Sim-VAE), a novel generative clustering model. The model can learn multi-modal distributions of high-dimensional data and use these to generate realistic data with high efficacy and efficiency. MoE-Sim-VAE is based on a Variational Autoencoder (VAE), where the decoder consists of a Mixture-of-Experts (MoE) architecture. This specific architecture allows for various modes of the data to be automatically learned by means of the experts. Additionally, we encourage the latent representation of our model to follow a Gaussian mixture distribution and to accurately represent the similarities between the data points. We assess the performance of our model on synthetic data, the MNIST benchmark data set, and a challenging real-world task of defining cell subpopulations from mass cytometry (CyTOF) measurements on hundreds of different datasets. MoE-Sim-VAE exhibits superior clustering performance on all these tasks in comparison to the baselines and we show that the MoE architecture in the decoder reduces the computational cost of sampling specific data modes with high fidelity.

# 1 INTRODUCTION

Clustering has been studied extensively (Aljalbout et al., 2018; Min et al., 2018) in machine learning. Recently, many Deep Clustering approaches were proposed, which modified (Variational) Autoencoder ((V)AE) architectures (Min et al., 2018; Zhang et al., 2017) or with varying regularization of the latent representation (Dizaji et al., 2017; Jiang et al., 2017; Yang et al., 2017; Fortuin et al., 2019).

Reconstruction error usually drives the definition of the latent representation learned from an AE or VAE. The representation for AE models is unconstrained and typically places data objects close to each other according to an implicit similarity measure that also yields favorable reconstruction error. In contrast, VAE models regularize the latent representation such that the represented inputs follow a certain variational distribution. This construction enables sampling from the latent representation and data generation via the decoder of a VAE. Typically, the variational distribution is assumed standard Gaussian, but for example, Jiang et al. (2017) introduced a mixture of Gaussian variational distribution for clustering purposes.

A key component of clustering approaches is the choice of similarity metric for the considered data objects which we try to group (Irani et al., 2016). Such similarity metrics are either defined a priori or learned from the data to specifically solve classification tasks via a Siamese network architecture (Chopra et al., 2005). Dimensionality reduction approaches, such as UMAP (McInnes et al., 2018) or t-SNE (van der Maaten & Hinton, 2008), allow to specify a similarity metric for projection and thereby define the data separation in the inferred latent representation.

In this work, we introduce the Mixture-of-Experts Similarity Variational Autoencoder (MoE-SimVAE), a new deep architecture that performs similarity-based representation learning, clustering of the data and generation of data from each specific data mode. Due to a combined loss function,

![](images/eec8d1a510ae07264293f573ad8fc69604c69f9f0a844d4892d73fd25313d502.jpg)  
Figure 1: Overview of the proposed model MoE-Sim-VAE. Data (in panel A) gets encoded via a encoder network (B) into a latent representation (C) which is trained to be a mixture of standard Gaussians. Via a clustering network (G), which is trained to reconstruct a user-defined similarity matrix (F), the encoded samples get assigned to the data mode-specific decoder subnetworks (which we call experts) in the MoE Decoder (D). The experts reconstruct the original input data and can be used for data generation when sampling from the variational distribution (E).

it can be jointly optimized. We assess the scope of the model on synthetic data and we present superior clustering performance on MNIST. Moreover, in an ablation study, we show the efficiency and precision of MoE-Sim-VAE for data generation purposes in comparison to the most related state-of-the-art method (Jiang et al., 2017). Finally, we show an application of MoE-Sim-VAE on a real-world clustering problem in biology on multiple datasets.

Our main contributions are to

- Develop a novel autoencoder architecture for  
- similarity-based representation learning  
- unsupervised clustering  
- accurate and efficient data generation

- Embed the Mixture-of-Expert architecture into a Variational Autoencoder setup to train a separate generator for each data mode  
- Show superior clustering performance of the model on benchmark dataset and real-world biological data

# 2 MIXTURE-OF-EXPERTS SIMILARITY VARIATIONAL AUTOENCODER

Here we introduce the Mixture-of-Experts Similarity Variational Autoencoder (MoE-Sim-VAE, Figure 1). The model is based on the Variational Autoencoder (Kingma & Welling, 2014). While the encoder network is shared across all data points, the decoder of the MoE-Sim-VAE consists of a number of  $K$  different subnetworks, forming a Mixture-of-Experts architecture (Shazeer et al., 2017). Each subnetworks constitutes a generator for a specific data mode and is learned from the data.

The variational distribution over the latent representation is defined to be a mixture of multivariate Gaussians, first introduced by Jiang et al. (2017). In our model, we aim to learn the mixture components in the latent representation to be standard Gaussians

$$
z \sim \sum_ {k = 0} ^ {K} \omega_ {k} \mathcal {N} \left(\boldsymbol {\mu} _ {k}, \boldsymbol {I}\right) \tag {1}
$$

where  $\omega_{k}$  are mixture coefficients,  $\pmb{\mu}_{k}$  are the means for each mixture component,  $\pmb{I}$  is the identity matrix and  $K$  is the number of mixture components. Similar to optimizing an Evidence Lower Bound (ELBO), we penalize the latent representation via the reconstruction loss of the data  $\mathcal{L}_{\text{reconst}}$  and by using the Kullback-Leibler (KL) divergence for multivariate Gaussians (Jiang et al., 2017) on the latent representation

$$
\mathcal {L} _ {K L} = D _ {K L} \left(\mathcal {N} _ {0}, \mathcal {N} _ {1}\right) = \frac {1}{2} \left\{t r \left(\boldsymbol {\Sigma} _ {1} ^ {- 1} \boldsymbol {\Sigma} _ {0}\right) + \left(\boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {0}\right) ^ {T} \boldsymbol {\Sigma} _ {1} ^ {- 1} \left(\boldsymbol {\mu} _ {1} - \boldsymbol {\mu} _ {0}\right) - k + l n \frac {\left| \boldsymbol {\Sigma} _ {1} \right|}{\left| \boldsymbol {\Sigma} _ {0} \right|} \right\} \tag {2}
$$

where  $k$  is a constant,  $\mathcal{N}_0 \sim \mathcal{N}(\pmb{\mu}_0, \pmb{\Sigma}_0 = \pmb{I})$ , and  $\pmb{I}$  is the identity matrix. Further,  $\mathcal{N}_{1} \sim \mathcal{N}(\pmb{\mu}_1, \pmb{\Sigma}_1 = \text{diag}(\sigma_j))$ , where  $\sigma_j$  for  $j = 1, \dots, D$ , for a number of dimensions  $D$ , is estimated from the samples of the latent representation. Finally, we assume  $\pmb{\mu}_0 = \pmb{\mu}_1$  resulting in the following simplified objective

$$
\mathcal {L} _ {K L} = D _ {K L} \left(\mathcal {N} _ {0}, \mathcal {N} _ {1}\right) = \frac {1}{2} \left\{t r \left(\boldsymbol {\Sigma} _ {1} ^ {- 1} \boldsymbol {\Sigma} _ {0}\right) - k + l n \frac {\left| \boldsymbol {\Sigma} _ {1} \right|}{\left| \boldsymbol {\Sigma} _ {0} \right|} \right\}, \tag {3}
$$

to penalize exclusively the covariance of each cluster. It remains to define the reconstruction loss  $\mathcal{L}_{\text{reconst}}$ , where we choose a Binary Cross-Entropy

$$
\mathcal {L} _ {\text {r e c o n s t}} = \sum_ {i} ^ {N} \sum_ {d} ^ {D} x _ {i, d} \log \left(x _ {i, d} ^ {\text {r e c o n s t}}\right) \tag {4}
$$

between the original data  $x$  (scaled between 0 and 1) and the reconstructed data  $x^{reconst}$ , where  $i$  iterates the batch size  $N$  and  $d$  the dimensions of the data  $D$ . Finally the loss for the VAE part is defined by

$$
\mathcal {L} _ {V A E} = \mathcal {L} _ {\text {r e c o n s t}} + \pi_ {1} \mathcal {L} _ {K L} \tag {5}
$$

with a weighting coefficient  $\pi_1$  which can be optimized as a hyperparameter.

# SIMILARITY CLUSTERING AND GATING OF LATENT REPRESENTATION

Training of a data mode-specific generator expert requires samples from the same data mode. This necessitates to solve a clustering problem, that is, mapping the data via the latent representation into  $K$  clusters, each corresponding to one of the  $K$  generator experts. We solve this clustering problem via a clustering network, also referred to as gating network for MoE models. It takes as input the latent representation  $z_{i}$  of sample  $i$  and outputs probabilities  $p_{ik} \in [0,1]$  for clustering sample  $i$  into cluster  $k$ . According to this cluster assignment, sample  $i$  is then gated to expert  $k = \operatorname{argmax}_k p_{ik}$  for each sample  $i$ . We further define the cluster centers  $\pmb{\mu}_{k}$  for  $k \in \{1,\dots,K\}$  similar as in the Expectation Maximization (EM) algorithm for Gaussian Mixture models (Bishop, 2006) as

$$
\boldsymbol {\mu} _ {k} = \frac {1}{N _ {k}} \sum_ {i = 1} ^ {N} p _ {i k} \boldsymbol {z} _ {i}, \tag {6}
$$

where  $N_{k}$  is the absolute number of data points assigned to cluster  $k$  based on highest probability  $p_{ik}$  for each sample  $i = 1,\dots ,N$ . The Gaussian mixture distributed latent representation (via KL loss in Equation 3) is motivation for the empirical computation of the cluster means and further, similar as in the EM algorithm, it allows iterative optimization of the means of the Gaussians. We train the clustering network to reconstruct a data-driven similarity matrix  $\mathbf{S}$ , using the Binary Cross-Entropy

$$
\mathcal {L} _ {\text {S i m i l a r i t y}} = \sum_ {i} ^ {N} \sum_ {j} ^ {N} S _ {i, j} \log \left(\left(\boldsymbol {P} \boldsymbol {P} ^ {T}\right) _ {i, j}\right) \tag {7}
$$

to minimize the error in  $PP^T \approx S$ , with  $P := \{p_{ik}\}_{i \in \{1, \dots, N\}, k \in \{1, \dots, K\}}$  where  $N$  is the number of samples (e.g., batch size). Intuitively,  $PP^T$  approximates the similarity matrix  $S$  since values in  $PP^T$  are only close to 1 when similar data objects are assigned to the same cluster, similar to the entries in the adjacency similarity matrix  $S$ . This similarity matrix is derived in an unsupervised way in our experiments (e.g. UMAP projection of the data and k-nearest-neighbors or distance thresholding to define the adjacency matrix for the batch), but can also be used to include weakly-supervised information (e.g., knowledge about diseased vs. non-diseased patients). If labels are

available, the model could even be used to derive a latent representation with supervision. The similarity feature in MoE-Sim-VAE thus allows to include prior knowledge about the best similarity measure on the data.

Moreover, we apply the DEPICT loss from Dizaji et al. (2017), to improve the robustness of the clustering. For the DEPICT loss, we additionally propagate a noisy probability  $\hat{p}_{ik}$  through the clustering network using dropout after each layer. The goal is to predict the same cluster for both, the noisy  $\hat{p}_{ik}$  and the clean probability  $p_{ik}$  (without applying dropout). Dizaji et al. (2017) derived as objective function a standard cross-entropy loss

$$
\mathcal {L} _ {D E P I C T} = - \frac {1}{N} \sum_ {i = 0} ^ {N} \sum_ {k = 0} ^ {K} q _ {i k} \log \left(\hat {p} _ {i k}\right) \tag {8}
$$

whereby  $q_{ik}$  is computed via the auxiliary function

$$
q _ {i k} = \frac {p _ {i k} / \left(\sum_ {i ^ {\prime}} p _ {i ^ {\prime} k}\right) ^ {\frac {1}{2}}}{\sum_ {k ^ {\prime}} p _ {i k ^ {\prime}} / \left(\sum_ {i ^ {\prime}} p _ {i ^ {\prime} k ^ {\prime}}\right) ^ {\frac {1}{2}}} \tag {9}
$$

where we refer to Dizaji et al. (2017) for exact derivation. The DEPICT loss encourages the model to learn invariant features from the latent representation for clustering with respect to noise (Dizaji et al., 2017). Looking at it from a different perspective, the loss helps to define a latent representation which has those invariant features to be able to reconstruct the similarity and therefore the clustering correctly. The complete clustering loss function  $\mathcal{L}_{\text{Clustering}}$  is then defined by

$$
\mathcal {L} _ {\text {C l u s t e r i n g}} = \mathcal {L} _ {\text {S i m i l a r i t y}} + \pi_ {2} \mathcal {L} _ {\text {D E P I C T}} \tag {10}
$$

with a mixture coefficient  $\pi_{2}$  which can be optimized as a hyperparameter.

# MOE-SIM-VAE LOSS FUNCTION

Finally, the MoE-Sim-VAE model loss is defined by

$$
\mathcal {L} _ {\text {M o E - S i m - V A E}} = \underbrace {\mathcal {L} _ {\text {V A E}}} _ {\mathcal {L} _ {\text {r e c o n s t}} + \pi_ {1} \mathcal {L} _ {K L}} + \underbrace {\mathcal {L} _ {\text {C l u s t e r i n g}}} _ {\mathcal {L} _ {\text {S i m i l a r i t y}} + \pi_ {2} \mathcal {L} _ {\text {D E P I C T}}} \tag {11}
$$

which consists of the two main loss functions  $\mathcal{L}_{VAE}$ , acting as a regularization for the latent representation, and  $\mathcal{L}_{Clustering}$ , which helps to learn the mixture components based on an a priori defined data similarity. The model objective function  $\mathcal{L}_{MoE-Sim-VAE}$  can then be optimized end-to-end to train all parts of the model.

# 3 RELATED WORK

(V)AEs have been extensively used for clustering (Xie et al., 2016; Dizaji et al., 2017; Li et al., 2017; Yang et al., 2017; Saito & Tan, 2017; Chen et al., 2017; Aljalbout et al., 2018; Fortuin et al., 2019). The most related approaches to MoE-Sim-VAE are Jiang et al. (2017) and Zhang et al. (2017).

Jiang et al. (2017) introduced the VaDE model, comprising a mixture of Gaussians as underlying distribution in the latent representation of a Variational Autoencoder. Optimizing the Evidence Lower Bound (ELBO) of the log-likelihood of the data can be rewritten to optimize the reconstruction loss of the data and KL divergence between the variational posterior and the mixture of Gaussians prior. Jiang et al. (2017) motivate the use of two separate networks for reconstruction and the generation process of the model. Further, to effectively generate images from a specific data mode and to increase image quality, the sampled points have to surpass a certain posterior threshold and are otherwise rejected. This leads to an increased computational effort. The MoE Decoder of our model, which is used for both reconstruction and generation, does not need such a threshold, as we discuss in more detail in Section 4.2.1.

Zhang et al. (2017) have introduced a mixture of autoencoders (MIXAE) model. The latent representation of the MIXAE is defined as the concatenation of the latent representation vectors of each single autoencoder in the model. Based on this concatenated latent representation, a Mixture Assignment Network predicts probabilities which are used in the Mixture Aggregation to form the output of the generator network. Each AE model learns the manifold of a specific cluster, similarly to our MoE Decoder. However, MIXAE does not optimize a variational distribution, such that generation of data from a distribution over the latent representation is not possible, in contrast to the MoE-Sim-VAE (Figure 2).

Table 1: Performance comparison of our method MoE-Sim-VAE with several published methods. The Table is extracted from Aljalbout et al. (2018). (" - ": metric not reported)  

<table><tr><td>METHOD</td><td>NMI</td><td>ACC</td></tr><tr><td>JULE, Yang et al. (2016b)</td><td>0.915</td><td>-</td></tr><tr><td>CCNN, Hsu &amp; Lin (2017)</td><td>0.876</td><td>-</td></tr><tr><td>DEC, Xie et al. (2016)</td><td>0.8</td><td>0.843</td></tr><tr><td>DBC, Li et al. (2017)</td><td>0.917</td><td>0.964</td></tr><tr><td>DEPICT, Dizaji et al. (2017)</td><td>0.916</td><td>0.965</td></tr><tr><td>DCN, Yang et al. (2017)</td><td>0.81</td><td>0.83</td></tr><tr><td>Neural Clustering, Saito &amp; Tan (2017)</td><td>-</td><td>0.966</td></tr><tr><td>UMMC, Chen et al. (2017)</td><td>0.864</td><td>-</td></tr><tr><td>VaDE, Jiang et al. (2017)</td><td>-</td><td>0.945</td></tr><tr><td>TAGnet, Wang et al. (2016)</td><td>0.651</td><td>0.692</td></tr><tr><td>IMSAT, Hu et al. (2017)</td><td>-</td><td>0.984</td></tr><tr><td>Aljalbout et al. (2018)</td><td>0.923</td><td>0.961</td></tr><tr><td>MoE-Sim-VAE (proposed)</td><td>0.935</td><td>0.975</td></tr></table>

# 4 EXPERIMENTS

We evaluate the MoE-Sim-VAE using synthetic data and the MNIST data set of handwritten digits (LeCun et al., 1998) for clustering and data generation. Furthermore, we performed an ablation study to demonstrate the importance of the MoE Decoder. Finally, we present experiments on a real-world application of defining cellular subpopulations from mass cytometry measurements (Bandura et al., 2009) of multiple publicly available datasets (Weber & Robinson, 2016; Bodenmiller et al., 2012). Model implementation details are reported in the appendix in section A.1

We found that our model achieves superior clustering performance compared to other models on synthetic, MNIST and real-world datasets. Moreover, we show that MoE-Sim-VAE can more effectively and efficiently generate data from specific modes in comparison to other methods.

# 4.1 EVALUATION OF MOE-SIM-VAE ON SYNTHETIC DATA

We evaluated our model using data sampled from a 100-dimensional multivariate Gaussian with equal mixture weights for each component. We tested two aspects of our model: Firstly, we evaluated up to how many clusters our model can fit well. Therefore, we sampled data from distributions with up to a hundred mixture components. For this experiment, we assume knowledge of the true number of clusters in the data for both methods, MoE-Sim-VAE and GMMs. Secondly, we tested if our model is able to identify the true number of clusters in the data. The similarity matrix  $S$  was defined as an adjacency matrix over the data items. Adjacency indicators were based on projecting the data via dimensionality reduction with UMAP (McInnes et al., 2018) and selecting neighbors according to a distance threshold. Details on model parameters can be found in Section A.1.1.

MoE-Sim-VAE performs better or comparable to the baseline for the number of clusters of up to 40 (Figure A1a). The model predicts with a close to perfect F-measure until reaching a true number of clusters of 30. Within the range of true number of clusters from 30 to 40, the model performs comparable to GMMs. Further, MoE-Sim-VAE learns the true number of clusters on its own (Figure A1b). For up to 23 components in the data, MoE-Sim-VAE learns the true number of clusters even when defining a model with  $K = 40$  experts in the MoE Decoder. This suggests that the model is robust to misspecification regarding the number of experts.

# 4.2 UNSUPERVISED CLUSTERING, EMBEDDING AND DATA GENERATION OF MNIST

We trained a MoE-Sim-VAE model on images from MNIST. We compared our model against multiple models which were recently reviewed in Aljalbout et al. (2018), and specifically against VaDE (Jiang et al., 2017) which shares similar properties with MoE-Sim-VAE (see Sec 3).

We compare the models with the Normalized Mutual Information (NMI) criterion but also classification accuracy (ACC) (Table 1). The MoE-Sim-VAE outperforms the other methods w.r.t. cluster

![](images/e2777acf3550ee6d365cb5859a123701c9e0b379f10c4295dcbf844530cf99c7.jpg)  
Figure 2: Generation of MNIST digit images. Data points from the latent representation were sampled from the variational distribution (A) which is learned to be a mixture of standard Gaussians and then clustered and gated (B) to the data-mode-specific experts of the MoE Decoder (C). (D) All samples from the variational distribution were correctly classified and therefore also correctly gated.

ing performance when comparing NMI and achieves the second-best result when comparing ACC. Note that we used the number of experts  $k = 10$  in our model to fit the existing number of digits in MNIST. Regarding the similarity measure, we decided to use as similarity a UMAP projection (McInnes et al., 2018) of MNIST and then apply k-nearest-neighbors of each sample in a batch. More details on the model are reported in Section A.1.2.

In addition to the clustering network, we can make use of the latent representation for image generation purposes. The latent representation is trained as a mixture of standard Gaussians. The means of these Gaussians are the centers of the clusters trained via the clustering network. Therefore, the variational distribution can be sampled from and gated to the cluster-specific expert in the MoE-decoder. The expert then generates new data points for the specific data mode. Results and the schematic are displayed in Figure 2 and in more detail and with greater sample size in the Appendix in Figure A2.

# 4.2.1 WHY DOES A MOE DECODER ACTUALLY MATTER?

In an ablation study, we compare the two models MoE-Sim-VAE and VaDE (Jiang et al., 2017) on generating MNIST images with the request for a specific digit. The goal is to show that a MoE decoder, as proposed in our model, is beneficial. We focus our comparison to VaDE since this model, as the MoE-Sim-VAE, resorts to a mixture of Gaussian latent representation but differs in generating images by means of a single decoder network instead of a Mixture-of-Expert decoder network. The rationale for our design choice is to ensure that smaller sub-networks learn to reproduce and generate specific modes of the data, in this case of specific MNIST digits.

To show that both models' latent representations are separating the different clusters well, we computed the Maximum Mean Discrepancy (MMD), defined in Section A.1.2. The MMD can be interpreted as a distance between distributions computed based on samples drawn from these distributions. The heatmaps of the MMDs for VaDE and MoE-Sim-VAE as well as an UMAP projection of the latent representation colored with the mixture component confirm visually the separation of the clusters in the latent representations of both models (Fig. A3). As a result, we can conclude that both latent representations can separate the clusters of respective digits well, such that the decoder gets well-defined samples to generate the requested digit. Therefore, the main difference of generating specific digits arises in the decoder/generator networks.

We evaluated the importance of the MoE-Decoder to (1) accurately generate requested digits and (2) be efficient in generating requested digits. Specifically, we sampled 10,000 points from each mixture component in the latent representation, generated images, and used the model's internal clustering to assign a probability to which digits were generated. To generate correct and high-quality images with VaDE, the posterior of the latent representation needs to be evaluated for each sample. This was done for the different thresholds  $\phi \in [0.0,0.1,0.2,\dots ,0.9,0.999]$ . The default threshold Jiang et al. (2017) used was  $\phi = 0.999$ . Instead of thresholding the latent representation, we ran the generation process for MoE-Sim-VAE for each threshold with the same settings. To gen

Table 2: Comparison of MoE-Sim-VAE performance to competitor methods in defining cell type composition in CyTOF measurements. The results in the table are extracted from the review paper of Weber & Robinson (2016), where 18 methods are compared on four different datasets. Our model outperforms the baselines on four out of five data sets.  

<table><tr><td>Method</td><td>Levine_32dim</td><td>Levine_13dim</td><td>Samusik_01</td><td>Samusik_all</td></tr><tr><td>ACCENSE</td><td>0.494</td><td>0.358</td><td>0.517</td><td>0.502</td></tr><tr><td>ClusterX</td><td>0.682</td><td>0.474</td><td>0.571</td><td>0.603</td></tr><tr><td>DensVM</td><td>0.66</td><td>0.448</td><td>0.239</td><td>0.496</td></tr><tr><td>FLOCK</td><td>0.727</td><td>0.379</td><td>0.608</td><td>0.631</td></tr><tr><td>flowClust</td><td>NA</td><td>0.416</td><td>0.612</td><td>0.61</td></tr><tr><td>flowMeans</td><td>0.769</td><td>0.518</td><td>0.625</td><td>0.653</td></tr><tr><td>flowMerge</td><td>NA</td><td>0.247</td><td>0.452</td><td>0.341</td></tr><tr><td>flowPeaks</td><td>0.237</td><td>0.215</td><td>0.058</td><td>0.323</td></tr><tr><td>FlowSOM</td><td>0.78</td><td>0.495</td><td>0.707</td><td>0.702</td></tr><tr><td>FlowSOM_pre</td><td>0.502</td><td>0.422</td><td>0.583</td><td>0.528</td></tr><tr><td>immunoClust</td><td>0.413</td><td>0.308</td><td>0.552</td><td>0.523</td></tr><tr><td>kmeans</td><td>0.42</td><td>0.435</td><td>0.65</td><td>0.59</td></tr><tr><td>PhenoGraph</td><td>0.563</td><td>0.468</td><td>0.671</td><td>0.653</td></tr><tr><td>Rclusterpp</td><td>0.605</td><td>0.465</td><td>0.637</td><td>0.613</td></tr><tr><td>SamSPECTRAL</td><td>0.512</td><td>0.253</td><td>0.263</td><td>0.138</td></tr><tr><td>SPADE</td><td>NA</td><td>0.127</td><td>0.169</td><td>0.13</td></tr><tr><td>SWIFT</td><td>0.177</td><td>0.179</td><td>0.202</td><td>0.208</td></tr><tr><td>Xshift</td><td>0.691</td><td>0.47</td><td>0.679</td><td>0.657</td></tr><tr><td>MoE-Sim-VAE (proposed)</td><td>0.70</td><td>0.68</td><td>0.76</td><td>0.74</td></tr></table>

erate images from VaDE we used the Python implementation<sup>1</sup> and model weights publicly available from Jiang et al. (2017).

As a result of this analysis we report a confusion matrix for MoE-Sim-VAE in Figure A5, the confusion matrices for each threshold for VaDE in Figure A6, the accuracy of generating a requested digit and the number of runs required in Figure A4. In summary, one can see that the MoE-Sim-VAE generates digits more accurately with fewer resources required. This can especially be seen when comparing the number of iterations required to fulfill the default posterior threshold of 0.999. VaDE needs nearly 2 million iterations to find samples that fulfill the aforementioned threshold criterion whereas the MoE-Sim-VAE only requires 10,000 for a comparable sample accuracy. In comparison the mean accuracy over all thresholds for MoE-Sim-VAE is 0.970, whereas VaDE reaches on average 0.944. VaDE reaches a maximum accuracy of 0.995, which costs the aforementioned 2 million iterations for generating 100,000 images, whereas MoE-Sim-VAE reaches a maximum accuracy of 0.971 with 100,000 runs, without accounting for a systematic generating/clustering error (confusing 5 and 8) of MoE-Sim-VAE which can be seen in the confusion matrix in Figure A5.

# 4.3 LEARNING CELL TYPE COMPOSITION IN PERIPHERAL BLOOD MONONUCLEAR CELLS USING CYTOF MEASUREMENTS

In the following, we want to show representation learning performance on a real-world problem in biology. Specifically, we focus on cell type definition from single-cell measurements. Cytometry by time-of-flight mass spectrometry (CyTOF) (Bandura et al., 2009) is a state-of-the-art technique allowing measurement of up to 1,000 cells per second and in parallel over 40 protein markers of the cells (Kay et al., 2013). Defining biologically relevant cell subpopulations by clustering this data is a common learning task (Aghaeepour et al., 2013; Weber & Robinson, 2016).

Many methods have been developed to tackle the problem introduced above and were compared on four publicly available datasets in Weber & Robinson (2016). The best out of 18 methods were FlowSOM (Gassen et al., 2015), PhenoGraph (Levine et al., 2015) and X-shift (Samusik et al., 2016). These are based on k-nearest-neighbors heuristics, either defined from a spanning graph or from estimating the data density. In contrast to these methods, MoE-Sim-VAE can map new

![](images/dae99e35bb3bf46be5eb8378bcccd472a540bd0e9e4eec9a120d08bfa88d05be.jpg)  
Figure 3: Comparison of MoE-Sim-VAE to the most popular competitor methods on defining cell types in peripheral blood mononuclear cell data via CyTOF measurements. On the x-axis different inhibitor treatments are listed whereas the y-axis reports the respective F-measure, defined in Equation 12, as performance measure of the methods. Each violin plot represents a run on a different inhibitor with multiple wells, whereas the line connects the means of the performance on the specific inhibitor.

cells into the latent representation, assign probabilities for cell types and infer an interpretable latent representation allowing intuitive downstream analysis by domain experts.

We applied MoE-Sim-VAE to the same datasets as in Weber & Robinson (2016) and achieve superior results in classification using the F-measure (Equation 12) in three out of four datasets. Similarly as in Weber & Robinson (2016) we trained MoE-Sim-VAE 30 times and report in Table 2 (adopted from Weber & Robinson (2016)) the means across all runs. The reproducibility of our model for each dataset can be seen in Figure A7.

Further, we trained a MoE-Sim-VAE model on 268 datasets from Bodenmiller et al. (2012) (more details on the data in A.1.3), and achieve superior classification results of cell subpopulations in the data when comparing to state-of-the-art methods in this field (PhenoGraph, X-Shift, FlowSOM). Exact results can be seen in Table A1 or visualized in Figure 3. More details on the MoE-Sim-VAE setting used for all results on CyTOF data are reported in the appendix (Section A.1.3).

# 5 CONCLUSION

Our MoE-Sim-VAE model can infer similarity-based representations, perform clustering tasks, and efficiently as well as accurately generate high-dimensional data. The training of the model is performed by optimizing a joint objective function consisting of data reconstruction, clustering, and KL loss, where the latter regularizes the latent representation. On synthetic data, we have shown the strengths and limitations of the model. On the benchmark dataset of MNIST, we presented superior clustering performance and the efficiency and accuracy of MoE-Sim-VAE in generating high-dimensional data. On the biological real-world task of defining cell subpopulations in complex single-cell data, we show superior clustering performances compared to state-of-the-art methods on over 270 datasets and therefore demonstrate MoE-Sim-VAE's real-world usefulness.

Future work might include to add adversarial training to the MoE decoder, which could improve image generation to create even more realistic images. Also, specific applications might benefit from replacing the Gaussian with a different mixture model. So far the MoE-Sim-VAE's similarity measure has to be defined by the user. Relaxing this requirement and allowing for learning a useful similarity measure automatically for inferring latent representations will be an interesting extension to explore. This could be useful in a weakly-supervised setting, which often occurs for example in clinical data consisting of healthy and diseased patients. Minor details between a healthy and diseased patient might make a huge difference and could be learned from the data using neural networks.

# REFERENCES

N. AghaEEPour, G. Finak, FlowCAP Consortium, DREAM Consortium, H. Hoos, TR. Mosmann, R. Brinkman, R. Gottardo, and Rh. Scheuermann. Critical assessment of automated flow cytometry data analysis techniques. Nature Methods, 2013.  
Elie Aljalbout, Vladimir Golkov, Yawar Siddiqui, Maximilian Strobel, and Daniel Cremers. Clustering with deep learning: Taxonomy and new methods. arXiv, 2018.  
DR. Bandura, VI. Baranov, OI. Ornatsky, A. Antonov, R. Kinach, X. Lou, S. Pavlov, S. Vorobiev, JE. Dick, and SD. Tanner. Mass cytometry: Technique for real time single cell multitarget immunoassay based on inductively coupled plasma time-of-flight mass spectrometry. Analytical Chemistry, 2009.  
Christopher M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.  
Bernd Bodenmiller, Eli R. Zunder, Rachel Finck, Tiffany J. Chen, Erica S. Savig, Robert V. Brugnner, Erin F. Simonds, Sean C. Bendall, Peter O. Krutzik Karen Sachs, and Garry P. Nolan. Multiplexed mass cytometry profiling of cellular states perturbed by small-molecule regulators. Nature Biotechnology, 2012.  
D. Chen, J. Lv, and Z. Yi. Unsupervised multi-manifold clustering by learning deep representation. *Workshops at the AAAI Conference on Artificial Intelligence*, 2017.  
S. Chopra, R. Hadsell, and Y. LeCun. Learning a similarity metric discriminatively, with application to face verification. IEEE, 2005.  
Kamran Ghasdi Dizaji, Amirhossein Herandi, Cheng Deng, Weidong Cai, and Heng Huang. Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization. arXiv, 2017.  
Vincent Fortuin, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. Somvae: Interpretable discrete representation learning on time series. Conference paper at ICLR 2019, 2019.  
Sofie Van Gassen, Britt Callebaut, Mary J. Van Helden, Bart N. Lambrecht, Piet Demeester, Tom Dhaene, and Yvan Saeys. Flowsom: Using self-organizing maps for visualization and interpretation of cytometry data. Cytometry Part A, 2015.  
Arthur Gretton, Karsten Borgwardt, Malte J. Rasch, Bernhard Scholkopf, and Alexander J. Smola. A kernel method for the two-sample problem. arXiv, 2008.  
C.-C. Hsu and C.-W. Lin. Cnn-based joint clustering and representation learning with feature drift compensation for large-scale image data. arXiv, 2017.  
W. Hu, T. Miyato, S. Tokui, E. Matsumoto, and M. Sugiyama. Learning discrete representations via information maximizing self augmented training. arXiv, 2017.  
Jasmine Irani, Nitin Pise, and Madhura Phatak. Clustering techniques and the similarity measures used in clustering: A survey. International Journal of Computer Applications, 2016.  
Zhuxi Jiang, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. Variational deep embedding: An unsupervised and generative approach to clustering. arXiv, 2017.  
Alexander W. Kay, Dara M. Strauss-Albee, and Catherine A. Blish. Application of mass cytometry (cytof) for functional and phenotypic analysis of natural killer cells. Methods in Molecular Biology, 2013.  
D. P. Kingma and M. Welling. Auto-encoding variational bayes. International Conference on Learning Representations (ICLR), 2014.  
Yann LeCun, Leon Botto, Yoshua Bengi, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 1998.

Jacob H. Levine, Erin F. Simonds, Sean C. Bendall, Kara L. Davis, El ad D. Amir, Michelle D. Tadmor, Oren Litvin, Harris G. Fienberg, Astraea Jager, Eli R. Zunder, Rachel Finck, Amanda L. Gedman, Ina Radtke, James R. Downing, Dana Peer, and Garry P. Nolan. Data-driven phenotypic dissection of aml reveals progenitor-like cells that correlate with prognosis. Cell, 2015.  
F. Li, H. Qiao, B. Zhang, and X. Xi. Discriminatively boosted image clustering with fully convolutional autoencoders. arXiv, 2017.  
Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv, 2018.  
Erxue Min, Xifeng Guo, Qiang Liu, Gen Zhang, Jianjing Cui, and Jun Long. A survey of clustering with deep learning: From the perspective of network architecture. IEEE, 2018.  
Peng Qiu, Erin F. Simonds, Sean C. Bendall, Kenneth D. Gibbs Jr., Robert V. Bruggner, Michael D. Linderman, Karen Sachs, Garry P. Nolan, and Sylvia K. Plevritis. Extracting a cellular hierarchy from high-dimensional cytometry data with spade. Nature Biotechnology, 2011.  
S. Saito and R. T. Tan. Neural clustering: Concatenating layers for better projections. Workshop track at ICLR 2017, 2017.  
Nikolay Samusik, Zinaida Good, Matthew H. Spitzer, Kara L. Davis, and Garry P. Nolan. Automated mapping of phenotype space with single-cell data. Nature Methods, 2016.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le1, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layers. arXiv, 2017.  
Dougal J. Sutherland, Hsiao-Yu Tung, Heiko Strathmann, Soumyajit De, Aaditya Ramdas, Alex Smola, and Arthur Gretton. Generative models and model criticism via optimized maximum mean discrepancy. arXiv, 2019.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 2008.  
Z. Wang, S. Chang, J. Zhou, M. Wang, and T. S. Huang. Learning a task-specific deep architecture for clustering. Proceedings of the SIAM International Conference on Data Mining (ICDM), 2016.  
Lukas M. Weber and Mark D. Robinson. Comparison of clustering methods for highdimensional singlecell flow and mass cytometry data. Cytometry Part A, 2016.  
J. Xie, R. Girshick, and A. Farhadi. Unsupervised deep embedding for clustering analysis. International Conference on Machine Learning (ICML), 2016.  
Bo Yang, Xiao Fu, Nicholas D. Sidiropoulos, and Mingyi Hong. Towards k-means-friendly spaces: Simultaneous deep learning and clustering. arXiv, 2017.  
J. Yang, D. Parikh, and D.Batra. Joint unsupervised learning of deep representations and image clusters. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016b.  
Dejiao Zhang, Yifan Sun, Brian Eriksson, and Laura Balzano. Deep unsupervised clustering using mixture of autoencoders. arXiv, 2017.
