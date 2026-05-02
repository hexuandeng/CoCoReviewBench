# DISENTANGLED CONDITIONAL VARIATIONAL AUTOENCODER FOR UNSUPERVISED ANOMALY DETECTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, generative models have shown promising performance in anomaly detection tasks. Specifically, autoencoders learn representations of high-dimensional data, and their reconstruction ability can be used to assess whether a new instance is likely to be anomalous. However, the primary challenge of unsupervised anomaly detection (UAD) is in learning appropriate disentangled features and avoiding information loss, while incorporating known sources of variation to improve the reconstruction. In this paper, we propose a novel architecture of generative autoencoder by combining the frameworks of  $\beta$ -VAE, conditional variational autoencoder (CVAE), and the principle of total correlation (TC). We show that our architecture improves the disentanglement of latent features, optimizes TC loss more efficiently, and improves the ability to detect anomalies in an unsupervised manner with respect to high-dimensional instances, such as in imaging datasets. Through both qualitative and quantitative experiments on several benchmark datasets, we demonstrate that our proposed method excels in terms of both anomaly detection and capturing disentangled features. Our analysis underlines the importance of learning disentangled features for UAD tasks.

# 1 INTRODUCTION

Unsupervised anomaly detection (UAD) has been a fertile ground for methodological research for several decades. Recently, generative models, such as Variational Autoencoders (VAEs) (Kingma & Welling, 2014) and Generative Adversarial Networks (GANs) (Goodfellow et al., 2020; Arjovsky et al., 2017), have shown exceptional performance at UAD tasks. By learning the distribution of normal data, generative models can naturally score new data as anomalous based on how well they can be reconstructed. For a recent review of deep learning for anomaly detection, see Pang et al. (2021).

In a complex task like UAD, disentanglement as a meta-prior encourages latent factors to be captured by different independent variables in the low-dimensional representation. This phenomenon has been on display in recent work that has used representation learning as a backbone for developing new VAE architectures. Some of the methods proposed new objective functions (Higgins et al., 2017; Mathieu et al., 2019), efficient decomposition of the evidence lower bound (ELBO) (Chen et al., 2018), partitioning of the latent space by adding a regularization term to the mutual information function (Zhao et al., 2017), introducing disentanglement metrics (Kim & Mnih, 2018), and penalizing total correlation (TC) loss (Gao et al., 2019). Penalized TC efficiently learns disentangled features and minimizes the dependence across the dimension of the latent space. However, it often leads to a loss of information, which leads to lower reconstruction quality. For example, methods such as  $\beta$ -VAE, Disentangling by Factorising (FactorVAE) (Kim & Mnih, 2018), and Relevance FactorVAE (RFVAE) (Kim et al., 2019) encourage more factorized representations with the cost of either losing reconstruction quality or losing a considerable among of information about the data and drop in disentanglement performance. To draw clear boundaries between an anomalous sample and a normal sample, we must minimize information loss.

To address these limitations, we present Disentangled Conditional Variational Autoencoder (dCVAE). Our approach is based on multivariate mutual information theory. Our main contribution is

a generative modeling architecture which learns disentangled representations of the data while minimizing the loss of information and thus maintaining good reconstruction capabilities. We achieve this by modeling known sources of variation, in a similar fashion as Conditional VAE (Pol et al., 2019).

Our paper is structured as follows. We first briefly discuss related methods (Section 2), draw connection between them, and present our proposed method dCVAE (Section 3). In Section 4, we discuss our experimental design including competing methods, datasets, and model configuration. Finally, experimental results are presented in Section 5, and Section 6 concludes this paper.

# 2 RELATED WORK

In this section, we discuss related work on autoencoders. We focus on two types of architecture: extensions of VAE enforcing disentanglement, and architectures based on mutual information theory.

# 2.1  $\beta$ -VAE

$\beta$ -VAE and its extensions proposed by (Higgins et al., 2017; Mathieu et al., 2019; Chen et al., 2018) is an augmentation of the original VAE with learning constraints of  $\beta$  applied to the objective function of the VAE. The idea of including such a hyper-parameter is to balance the latent channel capacity and improve the reconstruction accuracy. As a result,  $\beta$ -VAE is capable of discovering the disentangled latent factors and generating more realistic samples while retaining the small distance between the actual and estimated distributions.

Recall the objective function of VAE proposed by Kingma & Welling (2014):

$$
L _ {\mathrm {V A E}} (\theta , \phi) = - \mathbb {E} _ {\mathbf {z} \sim q _ {\phi} (\mathbf {z} | \mathbf {x})} \log p _ {\theta} (\mathbf {x} \mid \mathbf {z}) + D _ {\mathrm {K L}} \left(q _ {\phi} (\mathbf {z} \mid \mathbf {x}) \| p _ {\theta} (\mathbf {z})\right). \tag {1}
$$

Here,  $p_{\theta}(\mathbf{x} \mid \mathbf{z})$  is the probabilistic decoder,  $q_{\phi}(\mathbf{z} \mid \mathbf{x})$  is the recognition model, KLD is denoted by  $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))$  parameterized by the weights  $(\theta)$  and bias  $(\phi)$  of inference and generative models. As the incentive of  $\beta$ -VAE is to introduce the disentangling property, maximizing the probability of generating original data, and minimizing the distance between them, a constant  $\delta$  is introduced in the objective VAE to formulate the approximate posterior distributions as below:

$$
\max  _ {\phi , \theta} \mathbb {E} _ {\boldsymbol {x} \sim \mathrm {X}} \left[ \mathbb {E} _ {q _ {\phi} (z | \boldsymbol {x})} \left[ \log p _ {\theta} (\boldsymbol {x} \mid z) \right] \right] \quad \text {s u c h t h a t} D _ {\mathrm {K L}} \left(q _ {\phi} (z \mid \boldsymbol {x}) \| p (z)\right) <   \delta . \tag {2}
$$

Rewriting the Equation in Lagrangian form and using the KKT conditions, Higgins et al. (2017) derive the following objective function:

$$
\mathcal {L} _ {\beta V A E} (\theta , \phi) = \mathbb {E} _ {q _ {\phi} (z | x)} [ \log p _ {\theta} (x \mid z) ] - \beta D _ {\mathrm {K L}} \left(q _ {\phi} (z \mid x) \| p (z)\right), \tag {3}
$$

Here,  $\beta$  is the regularization coefficient that enforces the constraints to limit the capacity of the latent information  $z$ . When  $\beta = 1$ , we recover the original VAE. Increasing the value of  $\beta > 1$  enforces the constraints to capture disentanglement. However, Hoffman et al. (2017) argue that with an implicit prior, optimizing the regularized ELBO is equivalent to performing variational expectation maximization (EM).

# 2.2 FACTORVAE

Disentangling by Factorising or FactorVAE is another modification of  $\beta$ -VAE proposed by Kim & Mnih (2018). FactorVAE emphasizes the trade-off between disentanglement and reconstruction quality. The authors primarily focused on the objective function of the VAE and  $\beta$ -VAE. The authors propose a new loss function to mitigate the loss of information that arise while penalizing both the mutual information and the KLD to enforce disentangled latent factors.

According to Hoffman & Johnson (2016) and Makhzani & Frey (2017), the objective function of  $\beta$ -VAE can be further extended into:

$$
\mathbb {E} _ {p _ {\text {d a t a}} (x)} [ K L (q (z \mid x) \| p (z)) ] = I (x; z) + K L (q (z) \| p (z)), \tag {4}
$$

Here,  $I(x;z)$  is the mutual information between  $x$  and  $z$  under the joint distribution  $p_{\mathrm{data}}(x)q(z\mid x)$ . FactorVAE learns the second term of  $KL(q(z)\| p(z))$  and resolved the aforementioned issues by introducing total correlation penalty and density-ratio trick to approximate the distribution  $\bar{q} (z)$  generated by  $d$  samples from  $q(z)$ . The loss function of the FactorVAE is as follows:

$$
\begin{array}{l} \mathbb {E} _ {q (z \mid x ^ {(i)})} [ \log p (x ^ {(i)} \mid z) ] - K L (q (z \mid x ^ {(i)}) \| p (z)) \tag {5} \\ - \gamma K L (q (z) \| q (z)) \\ \end{array}
$$

# 2.3 THE PRINCIPLE OF TOTAL CORRELATION EXPLANATION (COREX)

Gao et al. (2019) introduced CorEx to mitigate the problem of learning disentangled and interpretable representations in a purely information-theoretic way. In general, for VAE, we assume a generative model where  $\mathbf{x}$  is a function of a latent variable  $\mathbf{z}$ , and afterward maximize the log likelihood of  $\mathbf{x}$ . On the other hand, CorEx follows the reverse process where  $\mathbf{z}$  is a stochastic function of  $\mathbf{x}$  parameterized by  $\theta$ , i.e.,  $p_{\theta}(\mathbf{z} \mid \mathbf{x})$ , and seek to estimate the joint distribution  $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{z} \mid \mathbf{x}) p(\mathbf{x})$ . The underlying true data distribution maximizes the following objective:

$$
\begin{array}{l} \mathcal {L} (\theta ; \mathbf {x}) = \underbrace {T C _ {\theta} (\mathbf {x} ; \mathbf {z})} _ {\text {i n f o r m a t i v e n e s s}} - \underbrace {T C _ {\theta} (\mathbf {z})} _ {\text {(d i s) e n t a n g l e m e n t}} \tag {6} \\ = T C (\mathbf {x}) - T C _ {\theta} (\mathbf {x} \mid \mathbf {z}) - T C _ {\theta} (\mathbf {z}). \\ \end{array}
$$

Recall the definition of the total correlation (TC) in terms of entropy  $H(\mathbf{x})$  (Studený & Vejnarová, 1998):

$$
T C (\mathbf {x}) = \sum_ {i = 1} ^ {d} H \left(\mathbf {x} _ {i}\right) - H (\mathbf {x}) = D _ {K L} \left(p (\mathbf {x}) \| \prod_ {i = 1} ^ {d} p \left(\mathbf {x} _ {i}\right)\right). \tag {7}
$$

By non-negativity of TC, Equation 6 naturally forms variational lower bound  $TC(\mathbf{x})$  to the CorEx objective, i.e.,  $TC(\mathbf{x}) \geq \mathcal{L}(\theta; \mathbf{x})$  for any  $\theta$ . Equation 6 can be rewritten in terms of mutual information  $I(\mathbf{x} : \mathbf{z}) = H(\mathbf{x}) - H(\mathbf{x} \mid \mathbf{z}) = H(\mathbf{z}) - H(\mathbf{z} \mid \mathbf{x})$ . Further constraining the search space  $p_{\theta}(\mathbf{z} \mid \mathbf{x})$  to have the factorized form  $p_{\theta}(\mathbf{z} \mid \mathbf{x}) = \prod_{i=1}^{m} p_{\theta}(\mathbf{z}_i \mid \mathbf{x})$  and the mutual information terms can be bounded by approximating the conditional distributions  $p_{\theta}(\mathbf{x}_j \mid \mathbf{z})$  and  $p_{\theta}(\mathbf{z}_j \mid \mathbf{x})$ . Finally, we can further rewrite and derive the lower bound for the objective function:

$$
\begin{array}{l} \mathcal {L} (\theta ; \mathrm {x}) = \sum_ {i = 1} ^ {d} I _ {\theta} \left(\mathrm {x} _ {i}: \mathrm {z}\right) - \sum_ {i = 1} ^ {m} I _ {\theta} \left(\mathrm {z} _ {i}: \mathrm {x}\right) \\ \geq \left(\sum_ {i = 1} ^ {d} H \left(x _ {i}\right)\right) + E _ {p _ {\theta} (x, z)} \left(\log \underbrace {q _ {\phi} (x \mid z)} _ {\text {d e c o d e r}}\right) \tag {8} \\ - D _ {K L} (\underbrace {p _ {\theta} (z \mid x)} _ {\text {e n c o d e r}} \| r _ {\alpha} (z)). \\ \end{array}
$$

# 2.4 TOTAL CORRELATION VARIATIONAL AUTOENCODER  $(\beta$  -TCVAE)

Chen et al. (2018) proposed disentanglement in their learned representations by adjusting the functional structure of the ELBO objective. The authors argued that each dimension of a disentangled representation should be able to represent a different factor of variation in the data and be changed independently of the other dimensions.  $\beta$ -TCVAE modifies the originally proposed ELBO objective by Higgins et al. (2017) forcing the algorithm to learn representations without explicitly making restrictions or reduction to the latent space. Recall the ELBO objective function (Equation 3) of  $\beta$ -VAE:

$$
\mathcal {L} _ {\beta V A E} (\theta , \phi) = \mathbb {E} _ {q _ {\phi} (z | x)} [ \log p _ {\theta} (x \mid z) ] - \beta D _ {\mathrm {K L}} \left(q _ {\phi} (z \mid x) \| p (z)\right) \tag {9}
$$

To introduce TC and disentanglement into the original  $\beta$ -VAE, Chen et al. decomposed the original KLD into Index-Code MI, Total Correlation and Dimension-wise KL terms. Furthermore, in the ELBO TC-Decomposition, each training samples are identified with a unique index  $\mathbf{n}$  and a uniform random variable that refers to the aggregated posterior as  $q(z) = \sum_{n=1}^{N} q(z \mid n)p(n)$  and can be denoted as:

$$
\begin{array}{l} \mathbb {E} _ {p (n)} [ \mathrm {K L} (q (z \mid n) \| p (z)) ] = \mathrm {K L} (q (z, n) \| q (z) p (n)) + \mathrm {K L} \left(q (z) \| \prod_ {j} q \left(z _ {j}\right)\right) \tag {10} \\ + \sum_ {j} \operatorname {K L} \left(q \left(z _ {j}\right) \| p \left(z _ {j}\right)\right) \\ \end{array}
$$

Finally, with a set of latent variables  $z_{j}$ , with known factors  $v_{k}$ , the authors introduced a disentanglement measuring metric called mutual information gap (MIG) and defined in terms of empirical mutual information  $I_{n}(z_{j};v_{k})$ :

$$
\frac {1}{K} \sum_ {k = 1} ^ {K} \frac {1}{H \left(v _ {k}\right)} \left(I _ {n} \left(z _ {j (k)}; v _ {k}\right) - \max  _ {j \neq j (k)} I _ {n} \left(z _ {j}; v _ {k}\right)\right) \tag {11}
$$

Here,  $j^{(k)} = \operatorname{argmax}_j I_n(z_j; v_k)$  and  $K$  is the number of known factors under  $v_k$ .

# 3 DISENTANGLED CONDITIONAL VARIATIONAL AUTOENCODER (DCVAE)

Our approach builds on CorEx and models known sources of variation in the data, in a manner similar to Conditional Variational Autoencoder (CVAE) Pol et al. (2019). In what follows, we will represent this known source of variation using the variable  $C$ . In the experiment below,  $C$  is discrete and represents the class of each image. Modifying Equation 6 to incorporate  $C$ , we get

$$
\mathcal {L} (\theta ; x, c) = T C _ {\theta} (x \mid c) - T C _ {\theta} (x \mid z, c) - T C _ {\theta} (z \mid c). \tag {12}
$$

Recall that the first two terms measure the amount of correlation explained by  $z$ , and by maximizing it, we maximize the informativeness of the latent representation. The third term measures the correlation between the components of  $z$ , and by minimizing it, we maximize the disentanglement between the latent dimensions.

Using Mutual Information Theory (Studený & Vejnarová, 1998), we can define the conditional differential entropy of  $H(x)$  given  $c$  and interpret mutual information as a reduction in uncertainty after conditioning:

$$
\begin{array}{l} I (x: z \mid c) = H (x \mid c) + H (z \mid c) - H (x, z \mid c) \\ I (x: z \mid c) = H (x \mid c) - H (x \mid z, c) = H (z \mid c) - H (z \mid x, c). \tag {13} \\ \end{array}
$$

We can now rewrite Equation 12 using derived mutual information theory from Equation 13:

$$
\mathcal {L} (\theta ; x, c) = \sum_ {j = 1} ^ {p} I (x _ {j}: z \mid c) - \sum_ {j = 1} ^ {d} I (z _ {j}: x \mid c). \tag {14}
$$

Now, consider the KLD between  $p_{\theta}(\mathbf{x} \mid \mathbf{z}, c)$  and an approximating distribution  $q_{\phi}(\mathbf{x} \mid \mathbf{z}, c)$ . In terms of expectations with respect to the joint distribution  $p_{\theta}(\mathbf{x}, \mathbf{z} \mid c)$ , we can write:

$$
- H (x \mid z, c) = E \left(\log p _ {\theta} (x \mid z, c)\right) \geq E \left(\log q _ {\phi} (x \mid z, c)\right). \tag {15}
$$

Combining Equation 14 and 15 and assuming an approximating distribution  $r_{\alpha}(z_j \mid c)$  for  $p_{\theta}(z_j \mid c)$ , we obtain two inequalities:

$$
I \left(x _ {j}: z \mid c\right) = H \left(x _ {j} \mid c\right) - H \left(x _ {j} \mid z, c\right) \geq H \left(x _ {j} \mid c\right) + E \left(\log q _ {\phi} (x \mid z, c)\right), \tag {16}
$$

$$
I \left(z _ {j}: x \mid c\right) = D _ {K L} \left(p _ {\theta} \left(z _ {j} \mid x, c\right) \| r _ {\alpha} \left(z _ {j} \mid c\right)\right). \tag {17}
$$

Combining these bounds, we finally derive a lower bound for the objective function for dCVAE:

$$
\mathcal {L} (\theta ; x, c) \geq \sum_ {j = 1} ^ {p} H (x _ {j} \mid c) + E (\log q _ {\phi} (x \mid z, c)) - \sum_ {j = 1} ^ {d} D _ {K L} \left(p _ {(z _ {j} | x, c) \| r (z _ {j} | c)}\right). \tag {18}
$$

Equation 18 illustrates the lower bound objective function of dCVAE where  $q_{\phi}(\mathbf{x} \mid \mathbf{z}, c)$  is the generative model or decoder and  $p_{\theta}(\mathbf{z}_j \mid \mathbf{x}, c)$  is the recognition model or encoder.

# 4 EXPERIMENTS

In the experiments below, we compare our dCVAE method to five baseline methods: VAE, CVAE,  $\beta$ -VAE, Factor-VAE, and RFVAE. The first two methods were selected as well-known baselines that do not explicitly enforce disentanglement; on the other hand, the latter three methods seek to achieve a disentangled representation of the data.

# 4.1 DATASETS

We evaluate dCVAE and other baseline models on the following four datasets. Three datasets (MNIST (Deng, 2012), Fashion-MNIST (Xiao et al., 2017), KMNIST (Clanuwat et al., 2018)) are trained for UAD. The fourth dataset (EMNIST (Cohen et al., 2017)) is used for testing on a real-world dataset to assess overall performance. A more detailed description of these datasets follows:

- MNIST and Fashion-MNIST (FMNIST) Firstly, we apply all models to two benchmark datasets, MNIST and Fashion-MNIST, for a fair comparison with other baseline methods. We used 10 classes with 60000 and 10000 training and testing samples for both datasets with  $28 \times 28 \times 1$  pixels channel.  
- KMNIST Secondly, we applied the same training process to another complex dataset, Kuzushiji-MNIST or KMNIST. KMNIST is a drop-in replacement for the MNIST dataset, a Japanese cursive writing style. KMNIST contains similar 10 classes with 60000 and 10000 training and testing samples with  $28 \times 28 \times 1$  pixels channel.  
- EMNIST Finally, all models are tested on the Extending MNIST or EMNIST Dataset. Using all 62 classes (digit 0-9, letters uppercase A-Z and lowercase a-z) with 700000 and 80000 training and testing samples with  $28 \times 28 \times 1$  pixels channels. The dataset was processed from NIST Special Database 19 Brother (1995) and contained handwritten digits and characters collected from over 500 writers.

# 4.2 RECONSTRUCTION ERROR AND ANOMALY SCORE

Leveraging methods for the discriminator as the anomaly score and drawing separation between normal and anomalous data is challenging for the divergent architectures of autoencoders. Depending on the task the architecture is trained for, the discriminator varies greatly. In general, the UAD methods utilize reconstruction error (Baur et al., 2018), distribution-based error (Goldstein & Uchida, 2016), and density-based error (Kiran et al., 2018) scores to distinguish normal and anomalous data. Formally, for each input  $x$ , a test input  $\widehat{x}_l$  is considered to be anomalous if reconstruction error or Anomaly score(  $\mathcal{A}$  ) is greater than the minimum threshold value and denoted as follows:

$$
\mathcal {A} (\hat {x}) = \| x - \mathrm {D} (\mathrm {G} (\hat {x})) \| _ {2}. \tag {19}
$$

# 4.3 PERFORMANCE METRICS

One of the challenges of measuring the performance of disentanglement is to apply appropriate metrics based on the nature of the dataset, not of latent factors or dimensions in the latent space. Therefore, considering the different model architectures and datasets, we first measure the performance using Numerical AUC Score, reconstruction error  $(\mathcal{A})$ , and negative ELBO score  $(\mathcal{E})$ . These metrics provide a quantifiable method of accuracy, while also measuring the disentanglement among the latent factors.

We also measure performance qualitatively by visualizing the latent space and the 2D-manifold. Both allow us to visualize the orthogonality between latent features and demonstrate the accuracy of the models to handle reduced latent variables and the ability to reconstruct samples.

# 4.4 MODEL CONFIGURATION

A fixed set of hyper-parameters are chosen to formulate a similar platform for all models and identify the computational cost and reproducibility of the models. Although baseline models that we chose,  $\beta$ -VAE, FactorVAE, RFVAE are highly sensitive to hyper-parameters tuning, the hyper-parameters throughout the experiment are kept consistent to observe how the models perform under similar values. A minimal 50 epochs are used to train the datasets. For MNIST, FMNIST, and KMNIST the batch size is kept to 64, with primary and secondary learning rates as  $\alpha = 10^{-5}$  and  $\alpha = 10^{-3}$  respectively. However, for the EMNIST dataset, the batch size increased to 128, and learning rates as  $\alpha = 10^{-6}$  and  $\alpha = 10^{-5}$ .

# 5 RESULTS AND DISCUSSION

In this section, we evaluate the results of dCVAE and other baseline methods on the downstream task of anomaly detection. A considerable volume of results was produced from our exhaustive evaluation. However, accounting for limitations of space here, we elected to focus on the results from EMNIST and KMNIST datasets in the main text. The remaining results (MNIST and FMNIST) are presented as Supplementary Material.

We show the results of our evaluation in three stages: firstly, using sample reconstruction and the negative ELBO score  $(\mathcal{E})$  with reconstruction error  $\mathcal{A}$ , we evaluate and compare the disentanglement ability of dCVAE with baseline architectures. Secondly, we use the UMAP algorithm (Sainburg et al., 2021) to reduce dimensions and visualize both latent representation, as well as interpolation of the 2D-manifold to distinguish the TC by comparing information loss and effects of modeling known sources of variation. Finally, we present AUC scores and training time to summarize the overall accuracy of the experimented methods.

We evaluate the quality of disentanglement by considering explicit separation of  $\mathcal{A}$  between normal and anomalous data and minimization of  $\mathcal{E}$ . A better disentanglement is achieved when:

(a) A higher reconstruction error  $\mathcal{A}$  for anomalous sample and lower reconstruction error  $\mathcal{A}$  for normal sample is obtained and  
(b)  $\mathcal{E}$  is minimized by enforcing regularization that either minimizes the negative ELBO decomposition  $D_{KL}\left(p_{(z_j|x,c)\| r(z_j|c)}\right)$  or regularizes the approximate posterior  $q_{\phi}(\mathbf{z}\mid \mathbf{x})$

A clear boundary in terms of learning efficient disentanglement between dCVAE and baseline methods can be observed from both EMNIST (Figure 1) and KMNIST (Figure 2) reconstruction. The first row corresponds to anomalous reconstruction and the second row shows normal sample reconstruction. Both  $\mathcal{E}$  and  $\mathcal{A}$  score suggests that dCVAE captures more independent factors and identifies anomalous and normal samples efficiently. This observation strongly justifies one of our primary claims, namely that dCVAE incorporates the disentanglement learning through enforcing TC and restrict independent latent variables to prioritize the minimization of the divergence. The other disentanglement methods presented here either only emphasize TC (indicated by the dependence between random variables) or introduce  $\beta$  (weighing the prior enforcement term), which limits the ability to learn randomness in a case when the hyperparameters are not tuned for certain dimensions.

![](images/6e92eb53e54a64e8ef3592f25c23b98f1b0da8180be33459d524ee4f8f63b83b.jpg)  
(a)  $\mathcal{E} = -235$ $\mathcal{A} = 0.97$

![](images/ed184f53a17e30e076536f5500f8ae1e95a6f6ca7a8dc23f80b684f043940645.jpg)  
(b)  $\mathcal{E} = -248$ $\mathcal{A} = 0.90$

![](images/48866a6465a5691d06f2735cc8149e5165e1dcb2c9ec0160597b95f25cab628b.jpg)  
(c)  $\mathcal{E} = -245$ $\mathcal{A} = 0.91$

![](images/bdcd193afdb2b0f53398b896980c05c2c202c95ff3a5fdc396b17ec5ac84dce9.jpg)  
(d)  $\mathcal{E} = -246$ $\mathcal{A} = 0.91$

![](images/20f689bd4a16a5bf890a1d2ab3301ccd9c37f5e867142520194f0fb50e020c64.jpg)  
(e)  $\mathcal{E} = -244$ $\mathcal{A} = 0.92$

![](images/bf0865315e0b32ea55d67399dac3e6ab67e44d5ce86befcf2b299b7e8c75f099.jpg)  
(f)  $\mathcal{E} = -246$ $\mathcal{A} = 0.92$

![](images/e7aae09bf4792928c194c84527f36533e1a3285312bf79bf5705cc32bee19a4c.jpg)  
(g)  $\mathcal{E} = -171$ $\mathcal{A} = 0.33$

![](images/d41d16786ac030e139b5c9d2e58a314251465f511b823228b8a0693c0aa4a2a7.jpg)  
(h)  $\mathcal{E} = -199$ $\mathcal{A} = 0.48$

![](images/5bfa255e835eb4118cab03e077cee1719ff36040f553cadb74c0510f585f2ac5.jpg)  
Figure 1: Reconstruction for digit zero (0) and the capital letter O. Here,  $\mathcal{E}$  refers to Negative ELBO score and  $\mathcal{A}$  is the reconstruction error or anomaly score. Only dCVAE and FactorVAE show steady improvement for both types of reconstruction. All the other methods misclassify the samples. Moreover, we can observe higher reconstruction error and ELBO scores compared to MNIST (Figure A1) and FMNIST (Figure A2).  
(i)  $\mathcal{E} = -180$ $\mathcal{A} = 0.53$  
(j)  $\mathcal{E} = -185$ $\mathcal{A} = 0.52$

![](images/918aeb3872d0543c4b415b9a3a51999f92b0ec301e3fd0f8d7ac21c11ce1a93c.jpg)

![](images/3b18aa02bad51f3574179be1b62ca9e8ea36feebbf722d1b836386e047513568.jpg)  
(k)  $\mathcal{E} = -195$ $\mathcal{A} = 0.57$

![](images/80759cbaa5fa126f4695d28d6fdda96cb2fae8b6786fcdfcae2ac083ab0937cb.jpg)  
(1)  $\mathcal{E} = -190$ $\mathcal{A} = 0.55$

![](images/43feb1c03f9f66e36db9165a7e28e1c7752f9b6b6925ce6712d43f18b6bd358c.jpg)  
(a)  $\mathcal{E} = -251$ $\mathcal{A} = 0.98$

![](images/305c9b0e371474005e584563fc2d85f91efd4428c577432e6b682cac6a58f3e9.jpg)  
(b)  $\mathcal{E} = -281$ $\mathcal{A} = 0.90$

![](images/7937df53528ebea4b101adba18166bee09935c4fe69198eadc575bc854bc338d.jpg)  
(c)  $\mathcal{E} = -279$ $\mathcal{A} = 0.98$

![](images/97ab0109c605ed268e3b8d6fa92fb69e46306e5d0e47816a60bd4e5006a6f4a1.jpg)  
(d)  $\mathcal{E} = -266$ $\mathcal{A} = 0.97$

![](images/ff5836a7328eee0781dac2af078b597cd5b7e65a8043207175c8b24eedba99b0.jpg)  
(e)  $\mathcal{E} = -277$ $\mathcal{A} = 0.97$

![](images/dfc195eda362a77d7484686984aff66ef8f15a7440adf2ee9ef0b9d2702a8eba.jpg)  
(f)  $\mathcal{E} = -270$ $\mathcal{A} = 0.96$

![](images/410926dc3ce74c980b82a757c1f37d7522785d39dea314d5df5c3069357f35bf.jpg)  
(g)  $\mathcal{E} = -188$ $\mathcal{A} = 0.28$  
Figure 2: In KMNIST dataset, without dCVAE, all other methods fail to classify both anomalous and normal samples. Reconstruction scores suggest FactorVAE, VAE almost fail to distinguish normal and anomalous observations. Since the stroke of the samples are similar in this dataset, methods that only emphasize disentanglement or empirical approximation lose more information in latent variable resulting in false anomaly detection.

![](images/d2ad5deff085aef6189ac796e1816f844fb053fc30f178d29a29052efed5e20c.jpg)  
(h)  $\mathcal{E} = -211$ $\mathcal{A} = 0.47$

![](images/ddf9450a9c88aa2ca1bd337377dcecdfb273dc4715ba5e2335c09e281018fa5c.jpg)  
(i)  $\mathcal{E} = -185$ $\mathcal{A} = 0.41$

![](images/5bfbef7c118489425f49ca334fdbd72e6c7895c1b8fe2b209b47d1dd2a3be5f9.jpg)  
(j)  $\mathcal{E} = -201$ $\mathcal{A} = 0.57$

![](images/c7ac5aa15fa65f4828be126e36860b54f416c17669698430b01db33ecf600a4f.jpg)  
(k)  $\mathcal{E} = -183$ $\mathcal{A} = 0.41$

![](images/34a21eac8c79393254bd410a8659aa6741ca1c778432b008926c2e4e926c56ee.jpg)  
(1)  $\mathcal{E} = -170$ $\mathcal{A} = 0.38$

The second observation is drawn using latent representation (Figure 3) and 2D-manifold embeddings (Figure 4 and 5). Through this experiment, we observe the effect of modeling using a known source of variation (i.e. introducing conditional variable  $C$  into the objective function) and minimizing information loss through multivariate mutual information theory (i.e. decomposition of TC). We can observe clear similarities between KLD loss and modeling with known score of variance in a reduced latent space. Due to enforced divergence loss, the plot of VAE and  $\beta$ -VAE are noticeably different from other architectures. Feature space is more compact for VAE,  $\beta$ -VAE, and we can see the cluster of the different classes are not well separated. However, conditioning the generative function (encoder) of CVAE and dCVAE provides the leverage to construct higher feature space and retain more accurate information in 2D-manifold (EMNIST, Figure 4; and KMNIST, Figure 5). Furthermore, TC reduces the correlation among disentanglement degrees when a specific feature is

learned (shape, strokes, color, boundaries). Such classes can be observed to cluster together and the other gets scattered with higher feature space (Figure 3). Compared to other methods, it is evident that dCVAE maintain consistent latent space and create separate clusters more accurately. This indicates that more disentangled variables are captured, and they retain more information through conditioning the generative model by minimizing the ELBO  $D_{KL}\left(p_{(z_j|x,c)\parallel r(z_j|c)}\right)$ .

![](images/ad736e15892ef630db9d52318dd6ef9342bfde39be40221623f367a345ede882.jpg)

![](images/4e96bd93e08242ffd22cb279fd3e89308790de478583035ee47169dbfbccd9f8.jpg)

![](images/7c6ff2d8064ed7445198df0e7baa159e3404fc33b5d8487e22defe9e11550362.jpg)

![](images/2cd95b563398814b1a93a6ec44d4c76ea99913db8fbca64af20f9973480af806.jpg)  
(a) EMNIST

![](images/9d41878d1acbc263780cf19697d52ab64327e1b903b60cc6e926e2055c7a2a0f.jpg)

![](images/602221acb30d40448d679daf0731c3b7cde147b2b0dd2a01ab01e74f4f9afe7e.jpg)

![](images/7a746f98b9d0043264ba890ced4518939e811d127f86aad7a37ddfe89a5a38c3.jpg)

![](images/d2c36f498c6e94dc3a0b6c43766ce980e4584a8a9a541e05a01be86dc0956dc6.jpg)

![](images/588d8df3b7437d5d465c22cb241fdcb33af7a093eb44a0052d41d79af3d38216.jpg)

![](images/c3d56407056cee98f096844b4b23ac40bccab2a8d90f5fed5989e8a3f5fbeb11.jpg)  
Figure 3: Latent Representation of EMNIST and KMNIST

![](images/d6ed2298bda47c862d72ac34ac307a07ef3f7c81a98cff888dd3ea2da3c78ee8.jpg)  
(b) KMNIST

![](images/fff291d8a3b055b9253a3671e8347a3fa481ee0656a78cbaae49d730313ffdcc.jpg)

![](images/399257e19f7700f878dbe26963ac7f739b86e5fdfe83f334e29aa2f68952b2d3.jpg)  
(a) dCVAE

![](images/eff79d28ba295fa0427db79e97344ff821f36591d6e9053933fb9de396e61321.jpg)  
(b) VAE

![](images/b1284cd8ad08eca9ca9fdacb04758455e19d9e1ba9b9cebaa3b100f3bad454a0.jpg)  
(c) CVAE

![](images/f5f46870b8a40f20d2b0391127dad30034622bdd4c7699c8d02654c965531151.jpg)  
(d) FactorVAE

![](images/10853ab7da21f68f11a57d465fe9ce7bfed421acb0db0b982947db68585229a8.jpg)  
(e)  $\beta$  -VAE

![](images/d5b018a3405681a7374a58d6650679b7cbf4e67166e1b3d2ae73fe9ab867cdab.jpg)  
(f) RFVAE

![](images/8fc566a04edd1f0cd3bd20cfde0eef3965915c91e38e01c649ec2f28467c93c5.jpg)  
(a) dCVAE

![](images/28c3803728058223b06feb3440af69f2478e978d1c483221ee7d3e460d253b64.jpg)  
Figure 4: Manifold Embeddings (EMNIST)  
(b) VAE  
Figure 5: Manifold Embeddings (KMNIST)

![](images/685746f439feb455fbdb7de9933674d4222aaff749ee757c96562615a33b7943.jpg)  
(c) CVAE

![](images/3005e216cf51ed823c8cfe60618e830c2bb2101ca9cf2307f959f5045b1e9584.jpg)  
(d) FactorVAE

![](images/26a3075506ea330623d15621c6e1a714a8d7a2683035f7a8afea0e4abe34e7ef.jpg)  
(e)  $\beta$  -VAE

![](images/adf6f3afe3e7ded0bd0959fda6f6ee0a150cde86951ba2330081600a5adba4d2.jpg)  
(f) RFVAE

Finally, Table 1 illustrates the results of model evaluation through AUC score and training time. dCVAE outperforms other methods in terms of AUC score. However, for larger divergent datasets like KMNIST and EMNIST, VAE shows lower training time compared to dCVAE. Since VAE only optimizes the negative log-likelihood, reconstruction loss and prior enforcement term, the training takes fewer latent variables to regularize, resulting in less training time. Nevertheless, compared to methods that incorporate TC (e.g. FactorVAE and RFVAE) or a constraint on the posterior  $(\beta$ -VAE), our proposed dCVAE scales to all larger datasets with higher classification accuracy.

Table 1: Evaluation metrics score  

<table><tr><td rowspan="2">Model</td><td colspan="2">MNIST</td><td colspan="2">FMNIST</td><td colspan="2">EMNIST</td><td colspan="2">KMNIST</td></tr><tr><td>AUC</td><td>Training Time (min)</td><td>AUC</td><td>Training Time (min)</td><td>AUC</td><td>Training Time (min)</td><td>AUC</td><td>Training Time (min)</td></tr><tr><td>dCVAE</td><td>88.31</td><td>37</td><td>88.63</td><td>44</td><td>78.98</td><td>102</td><td>61.02</td><td>95</td></tr><tr><td>VAE</td><td>88.21</td><td>37</td><td>84.12</td><td>39</td><td>67.23</td><td>92</td><td>51.13</td><td>78</td></tr><tr><td>CVAE</td><td>87.57</td><td>43</td><td>83.31</td><td>48</td><td>66.01</td><td>117</td><td>42.35</td><td>104</td></tr><tr><td>FactorVAE</td><td>87.11</td><td>53</td><td>82.78</td><td>50</td><td>62.91</td><td>138</td><td>49.23</td><td>117</td></tr><tr><td>β-VAE</td><td>85.31</td><td>51</td><td>82.31</td><td>53</td><td>65.12</td><td>123</td><td>50.01</td><td>119</td></tr><tr><td>RFVAE</td><td>85.31</td><td>55</td><td>81.11</td><td>57</td><td>55.03</td><td>130</td><td>49.51</td><td>132</td></tr></table>

The only trade-offs in our proposed method seem to occur when minimizing the negative ELBO loss. In certain conditions, dCVAE reaches a lower reconstruction loss (anomalous sample) yet minimizes the negative ELBO score (Figure 3, 4). In general, negative ELBO loss should illustrate symmetrical change with reconstruction error. Such inconsistency could lead to a significant drop in the classification accuracy, thus leading to a false anomaly detection result.

# 6 CONCLUSION

In this research, we present a novel generative variational model dCVAE, to improve the unsupervised anomaly detection task through disentanglement learning, TC loss, and minimizing trade-offs between reconstruction loss and reconstruction quality. Introducing a conditional variable to mitigate the loss of information effectively captures more disentangled features and produces more accurate reconstructions. Such architecture could be used in a wider range of applications, including generating controlled image synthesis, efficient molecular design and generation, source separation for bio-signals and images, and conditional text generation. Future research direction includes investigating in the gap between the posterior and the prior distribution, resolving the trade-offs between loss function and reconstruction, and inspect dCVAE using different disentanglement metrics.

# REPRODUCIBILITY STATEMENT

In this research, we carefully considered reproducibility in designing and conducting all experiments. In our supplemental texts, we have attached our source code. The experiments are designed independently to make the results reproducible. Image reconstruction and generation, 2D-Manifold embeddings, training time, and ELBO score calculation are performed separately from other downstream tasks like classification accuracy, reconstruction error, and latent representation. Furthermore, we used both TensorFlow and PyTorch frameworks to remove package dependencies. To remove the library dependencies and installation issues, virtual environment and package requirement files are also added. Finally, to make the results more accessible, we also provided randomly generated images with supplementary texts.

# REFERENCES

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International conference on machine learning, pp. 214-223. PMLR, 2017.  
Christoph Baur, Benedikt Wiestler, Shadi Albarqouni, and Nassir Navab. Deep autoencoding models for unsupervised anomaly segmentation in brain mr images. In International MICCAI brainlesion workshop, pp. 161-169. Springer, 2018.  
Ricky TQ Chen, Xuechen Li, Roger Grosse, and David Duvenaud. Isolating sources of disentanglement in VAEs. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 2615-2625, 2018.  
Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. Emmist: an extension of mnist to handwritten letters (2017). arXiv preprint arXiv:1702.05373, 2017.  
Li Deng. The MNIST database of handwritten digit images for machine learning research [best of the web]. IEEE Signal Processing Magazine, 29(6):141-142, 2012.  
Shuyang Gao, Rob Brekelmans, Greg Ver Steeg, and Aram Galstyan. Auto-encoding total correlation explanation. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1157-1166. PMLR, 2019.  
Markus Goldstein and Seiichi Uchida. A comparative evaluation of unsupervised anomaly detection algorithms for multivariate data. *PloS one*, 11(4):e0152173, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139-144, 2020.  
Patrick J Grother. Nist special database 19. Handprinted forms and characters database, National Institute of Standards and Technology, 10, 1995.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017.  
Matthew D Hoffman and Matthew J Johnson. Elbo surgery: yet another way to carve up the variational evidence lower bound. In Workshop in Advances in Approximate Bayesian Inference, NIPS, volume 1, 2016.  
Matthew D Hoffman, Carlos Riquelme, and Matthew J Johnson. The  $\beta$ -vae's implicit prior. In Workshop on Bayesian Deep Learning, NIPS, pp. 1-5, 2017.  
Hyunjik Kim and Andriy Mnih. Disentangling by factorising. In International Conference on Machine Learning, pp. 2649-2658. PMLR, 2018.  
Minyoung Kim, Yuting Wang, Pritish Sahu, and Vladimir Pavlovic. Relevance factor vae: Learning and identifying disentangled factors. arXiv preprint arXiv:1902.01568, 2019.  
Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In 2nd International Conference on Learning Representations, ICLR, 2014.  
B Ravi Kiran, Dilip Mathew Thomas, and Ranjith Parakkal. An overview of deep learning based methods for unsupervised and semi-supervised anomaly detection in videos. Journal of Imaging, 4(2):36, 2018.  
Alireza Makhzani and Brendan J Frey. Pixelgan autoencoders. Advances in Neural Information Processing Systems, 30, 2017.

Emile Mathieu, Tom Rainforth, Nana Siddharth, and Yee Whye Teh. Disentangling disentanglement in variational autoencoders. In International Conference on Machine Learning, pp. 4402-4412. PMLR, 2019.  
Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review. ACM Computing Surveys (CSUR), 54(2):1-38, 2021.  
Adrian Alan Pol, Victor Berger, Cecile Germain, Gianluca Cerminara, and Maurizio Pierini. Anomaly detection with conditional variational autoencoders. In 2019 18th IEEE international conference on machine learning and applications (ICMLA), pp. 1651-1657. IEEE, 2019.  
Tim Sainburg, Leland McInnes, and Timothy Q. Gentner. Parametric UMAP Embeddings for Representation and Semisupervised Learning. Neural Computation, 33(11):2881-2907, 10 2021. ISSN 0899-7667. doi: 10.1162/neco_a_01434. URL https://doi.org/10.1162/neco_a_01434.  
Milan Studeny and Jirina Vejnarova. The multiinformation function as a tool for measuring stochastic dependence. In Learning in graphical models, pp. 261-297. Springer, 1998.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
Shengjia Zhao, Jiaming Song, and Stefano Ermon. Infovae: Information maximizing variational autoencoders. arXiv preprint arXiv:1706.02262, 2017.
