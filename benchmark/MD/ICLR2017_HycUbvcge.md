# DEEP GENERALIZED CANONICAL CORRELATION ANALYSIS

Adrian Benton, Huda Khayrallah, Biman Gujral, Drew Reisinger, Sheng Zhang, Raman Arora

Center for Language and Speech Processing

Johns Hopkins University

Baltimore, MD 21218, USA

adrian†, huda*, bgujral1*, reisinger, zsheng2*, arora†

$^{\star}$ @jhu.edu,  ${}^{\circ}$  @cogsci.jhu.edu,  $\dagger$  @cs.jhu.edu

# ABSTRACT

We present Deep Generalized Canonical Correlation Analysis (DGCCA) – a method for learning nonlinear transformations of arbitrarily many views of data, such that the resulting transformations are maximally informative of each other. While methods for nonlinear two-view representation learning (Deep CCA, (Andrew et al., 2013)) and linear many-view representation learning (Generalized CCA (Horst, 1961)) exist, DGCCA is the first multiview representation learning technique that combines the flexibility of nonlinear (deep) representation learning with the statistical power of incorporating information from many independent sources, or views. We present the DGCCA formulation as well as an efficient stochastic optimization algorithm for solving it. We learn DGCCA representations on two distinct datasets for three downstream tasks: phonetic transcription from acoustic and articulatory measurements, and recommending hashtags or friends to Twitter users. We find that DGCCA representations soundly beat existing methods at phonetic transcription and hashtag recommendation, and in general perform no worse than standard linear many-view techniques.

# 1 INTRODUCTION

Multiview representation learning refers to settings where one has access to many "views" of data, at train time. Views often correspond to different modalities or independent information about examples: a scene represented as a series of audio and image frames, a social media user characterized by the messages they post and who they friend, or a speech utterance and the configuration of the speaker's tongue. Multiview techniques learn a representation of data that captures the sources of variation common to all views.

Multiview representation techniques are attractive for intuitive reasons. A representation that is able to explain many views of the data is more likely to capture meaningful variation than a representation that is a good fit for only one of the views. They are also attractive for the theoretical reasons. For example, (Anandkumar et al., 2014) show that certain classes of latent variable models, such as Hidden Markov Models, Gaussian Mixture Models, and Latent Dirichlet Allocation models, can be optimally learned with multiview spectral techniques. Representations learned from many views will generalize better than one, since the learned representations are forced to accurately capture variation in all views at the same time (Sridharan & Kakade, 2008) – each view acts as a regularizer, constraining the possible representations that can be learned. These methods are often based on canonical correlation analysis (CCA), a classical statistical technique proposed by Hotelling (1936).

In spite of encouraging theoretical guarantees, multiview learning techniques cannot freely model nonlinear relationships between arbitrarily many views. Either they are able to model variation across many views, but can only learn linear mappings to the shared space Horst (1961), or they simply cannot be applied to data with more than two views using existing techniques based on kernel CCA (Hardoon et al., 2004) and deep CCA Andrew et al. (2013).

Here we present Deep Generalized Canonical Correlation Analysis (DGCCA). Unlike previous correlation-based multiview techniques, DGCCA learns a shared representation from data with arbitrarily many views and simultaneously learns nonlinear mappings from each view to this shared space. The only (mild) constraint is that these nonlinear mappings from views to shared space must be differentiable. Our main methodological contribution is the derivation of the gradient update for the Generalized Canonical Correlation Analysis (GCCA) objective Horst (1961). Our practical contribution is much greater: the first nonlinear multiview representation learning technique that learns from more than two views.

We also evaluate DGCCA-learned representations on two distinct datasets and three downstream tasks: phonetic transcription from aligned speech and articulatory data, and Twitter hashtag and friend recommendation from six text and network feature views. We find that downstream performance of DGCCA representations is ultimately task-dependent. However, we find clear gains in performance from many-view DGCCA for tasks previously shown to benefit from representation learning on more than two views, with up to  $4\%$  improvement in heldout accuracy for phonetic transcription.

The paper is organized as follows. We review prior work in Section 2. In Section 3 we describe DGCCA. Empirical results on a synthetic dataset, and three downstream tasks are presented in Section 4. We conclude with future directions in Section 5.

# 2 PRIOR WORK

# 2.1 MULTIVIEW LEARNING TECHNIQUES

Some of most successful techniques for multiview representation learning are based on canonical correlation analysis (Wang et al., 2015a;b) and its extension to the nonlinear and many view settings, which we describe in this section.

# 2.1.1 CANONICAL CORRELATION ANALYSIS (CCA)

Canonical correlation analysis (CCA) (Hotelling, 1936) is a statistical method that finds maximally correlated linear projections of two random vectors and is a fundamental multiview learning technique. Given two input views,  $X_{1} \in \mathbb{R}^{d_{1}}$  and  $X_{2} \in \mathbb{R}^{d_{2}}$ , with covariance matrices,  $\Sigma_{11}$  and  $\Sigma_{22}$ , respectively, and cross-covariance matrix,  $\Sigma_{12}$ , CCA finds directions that maximize the correlation between them:

$$
\begin{array}{l} (u_{1}^{*},u_{2}^{*}) = \operatorname *{argmax}_{u_{1}\in \mathbb{R}^{d_{1}},u_{2}\in \mathbb{R}^{d_{2}}}corr(u_{1}^{\top}X_{1},u_{2}^{\top}X_{2}) \\ = \operatorname * {a r g m a x} _ {u _ {1} \in \mathbb {R} ^ {d _ {1}}, u _ {2} \in \mathbb {R} ^ {d _ {2}}} \frac {u _ {1} ^ {\top} \Sigma_ {1 2} u _ {2}}{\sqrt {u _ {1} ^ {\top} \Sigma_ {1 1} u _ {1} u _ {2} ^ {\top} \Sigma_ {2 2} u _ {2}}} \\ \end{array}
$$

Since this formulation is invariant to affine transformations of  $u_{1}$  and  $u_{2}$ , we can write it as the following constrained optimization formulation:

$$
\left(u _ {1} ^ {*}, u _ {2} ^ {*}\right) = \underset {u _ {1} ^ {\top} \Sigma_ {1 1} u _ {1} = u _ {2} ^ {\top} \Sigma_ {2 2} u _ {2} = 1} {\operatorname {a r g m a x}} u _ {1} ^ {\top} \Sigma_ {1 2} u _ {2} \tag {1}
$$

This technique has two limitations that have led to significant extensions: First, it is limited to learning representations that are linear transformations of the data in each view, and second, it can only leverage two input views.

# 2.1.2 DEEP CANONICAL CORRELATION ANALYSIS (DCCA)

Deep CCA (DCCA) (Andrew et al., 2013) is an extension of CCA that addresses the first limitation by finding maximally linearly correlated non-linear transformations of two vectors. It does this by

passing each of the input views through stacked non-linear representations and performing CCA on the outputs.

Let us use  $f_{1}(X_{1})$  and  $f_{2}(X_{2})$  to represent the network outputs. The weights,  $W_{1}$  and  $W_{2}$ , of these networks are trained through standard backpropagation to maximize the CCA objective:

$$
(u _ {1} ^ {*}, u _ {2} ^ {*}, W _ {1} ^ {*}, W _ {2} ^ {*}) = \operatorname * {a r g m a x} _ {u _ {1}, u _ {2}} c o r r (u _ {1} ^ {\top} f _ {1} (X _ {1}), u _ {2} ^ {\top} f _ {2} (X _ {2}))
$$

DCCA is still limited to only 2 input views.

# 2.1.3 GENERALIZED CANONICAL CORRELATION ANALYSIS (GCCA)

Another extension of CCA, which addresses the limitation on the number of views, is Generalized CCA (GCCA) (Horst, 1961). It corresponds to solving the optimization problem in Equation (2), of finding a shared representation  $G$  of  $J$  different views, where  $N$  is the number of data points,  $d_{j}$  is the dimensionality of the  $j$ th view,  $r$  is the dimensionality of the learned representation, and  $X_{j} \in \mathbb{R}^{d_{j} \times N}$  is the data matrix for the  $j$ th view.

$$
\underset {U _ {j} \in \mathbb {R} ^ {d _ {j} \times r}, G \in \mathbb {R} ^ {r \times N}} {\text {m i n i m i z e}} \sum_ {j = 1} ^ {J} \| G - U _ {j} ^ {\top} X _ {j} \| _ {F} ^ {2} \tag {2}
$$

subject to  $GG^{\top} = I_r$

Solving GCCA requires finding an eigendecomposition of an  $N \times N$  matrix, which scales quadratically with sample size and leads to memory constraints. Unlike CCA and DCCA, which only learn projections or transformations on each of the views, GCCA also learns a view-independent representation  $G$  that best reconstructs all of the view-specific representations simultaneously. The key limitation of GCCA is that it can only learn linear transformations of each view.

# 3 DEEP GENERALIZED CANONICAL CORRELATION ANALYSIS (DGCCA)

In this section, we present deep GCCA (DGCCA): a multiview representation learning technique that benefits from the expressive power of deep neural networks and can also leverage statistical strength from more than two views in data, unlike Deep CCA which is limited to only two views. More fundamentally deep CCA and deep GCCA have very different objectives and optimization problems, and it is not immediately clear how to extend deep CCA to more than two views.

DGCCA learns a nonlinear map for each view in order to maximize the correlation between the learnt representations across different views. In training, DGCCA passes the input vectors in each view through multiple layers of nonlinear transformations and backpropagates the gradient of the GCCA objective with respect to network parameters to tune each view's network, as illustrated in Figure 1. The objective is to train networks so as to reduce the GCCA reconstruction error among their outputs. At test time, new data can be projected by feeding them through the learned network for each view.

We now formally define the DGCCA problem. We consider  $J$  views in our data, and let  $X_{j} \in \mathbb{R}^{d_{j} \times N}$  denote the  $j^{th}$  input matrix. The network for the  $j^{th}$  view consists of  $K_{j}$  layers. Assume, for simplicity, that each layer in the  $j^{th}$  view network has  $c_{j}$  units with a final (output) layer of size  $o_{j}$ . The output of the  $k^{th}$  layer for the  $j^{th}$  view is  $h_{k}^{j} = s(W_{k}^{j} h_{k-1}^{j})$ , where  $s: \mathbb{R} \to \mathbb{R}$  is a nonlinear activation function and  $W_{k}^{j} \in \mathbb{R}^{c_{k} \times c_{k-1}}$  is the weight matrix for the  $k^{th}$  layer of the  $j^{th}$  view network. We denote the output of the final layer as  $f_{j}(X_{j})$ .

DGCCA can be expressed as the following optimization problem: find weight matrices  $W^{j} = \{W_{1}^{j},\dots ,W_{K_{j}}^{j}\}$  defining the functions  $f_{j}$ , and linear transformations  $U_{j}$  (of the output of the  $j^{th}$

![](images/4b1d0c1c5f9b586bc5c81d5e58db444d1b6aac80635b0bc63d60b350610a0158.jpg)  
Figure 1: A schematic of DGCCA with deep networks for  $J$  views.

network), for  $j = 1, \dots, J$ , that

$$
\underset {U _ {j} \in \mathbb {R} ^ {o _ {j} \times r}, G \in \mathbb {R} ^ {r \times N}} {\text {m i n i m i z e}} \sum_ {j = 1} ^ {J} \| G - U _ {j} ^ {\top} f _ {j} \left(X _ {j}\right) \| _ {F} ^ {2}, \tag {3}
$$

subject to  $GG^{\top} = I_r$

where  $G\in \mathbb{R}^{r\times N}$  is the shared representation we are interested in learning.

**Optimization:** We solve the DGCCA optimization problem using stochastic gradient descent (SGD) with mini-batches. In particular, we estimate the gradient of the DGCCA objective in Problem 3 on a mini-batch of samples that is mapped through the network and use back-propagation to update the weight matrices,  $W^{j}$ 's. However, note that the DGCCA optimization problem is a constrained optimization problem. It is not immediately clear how to perform projected gradient descent with back-propagation. Instead, we characterize the objective function of the GCCA problem at an optimum, and compute its gradient with respect to the inputs to GCCA, i.e. with respect to the network outputs. These gradients are then back-propagated through the network to update  $W^{j}$ 's.

Although the relationship between DGCCA and GCCA is analogous to the relationship between DCCA and CCA, derivation of the GCCA objective gradient with respect to the network output layers is non-trivial. The main difficulty stems from the fact that there is no natural extension of the correlation objective to more than two random variables. Therefore, we consider, instead, correlations between every pair of views, stack them in a  $J \times J$  matrix and maximize a certain matrix norm for that matrix. For GCCA, this suggests an optimization problem that maximizes the sum of correlations between a shared representation and each view. Since the objective as well as the constraints of the generalized CCA problem are very different from that of the CCA problem, it is not immediately obvious how to extend Deep CCA to Deep GCCA.

Next, we show a sketch of the gradient derivation, the full derivation is given in appendix A. It is easy to show that the solution to the GCCA problem is given by solving an eigenvalue problem. In particular, define  $C_{jj} = f(X_j)f(X_j)^\top \in \mathbb{R}^{o_j \times o_j}$ , to be the scaled empirical covariance matrix of the  $j^{th}$  network output, and  $P_j = f(X_j)^\top C_{jj}^{-1}f(X_j) \in \mathbb{R}^{N \times N}$  be the corresponding projection matrix that whitens the data; note that  $P_j$  is symmetric and idempotent. We define  $M = \sum_{j=1}^{J} P_j$ . Since each  $P_j$  is positive semi-definite, so is  $M$ . Then, it is easy to check that the rows of  $G$  are the top  $r$  (orthonormal) eigenvectors of  $M$ , and  $U_j = C_{jj}^{-1}f(X_j)G^\top$ . Thus, at the minimum of the objective, we can rewrite the reconstruction error as follows:

$$
\begin{array}{l} \sum_ {j = 1} ^ {J} \| G - U _ {j} ^ {\top} f _ {j} (X _ {j}) \| _ {F} ^ {2} = \sum_ {j = 1} ^ {J} \| G - G f _ {j} (X _ {j}) ^ {\top} C _ {j j} ^ {- 1} f _ {j} (X _ {j}) \| _ {F} ^ {2} \\ = r J - \operatorname {T r} \left(G M G ^ {\top}\right) \\ \end{array}
$$

Minimizing the GCCA objective (w.r.t. the weights of the neural networks) means maximizing  $\mathrm{Tr}(GMG^{\top})$ , which is the sum of eigenvalues  $L = \sum_{i=1}^{r} \lambda_i(M)$ . Taking the derivative of  $L$  with

respect to each output layer  $f_{j}(X_{j})$  we have:

$$
\frac {\partial L}{\partial f _ {j} (X _ {j})} = 2 U _ {j} G - 2 U _ {j} U _ {j} ^ {\top} f _ {j} (X _ {j})
$$

Thus, the gradient is the difference between the  $r$ -dimensional auxiliary representation  $G$  embedded into the subspace spanned by the columns of  $U_{j}$  (the first term) and the projection of the actual data in  $f_{j}(X_{j})$  onto the said subspace (the second term). Intuitively, if the auxiliary representation  $G$  is far away from the view-specific representation  $U_{j}^{\top}f_{j}(X_{j})$ , then the network weights should receive a large update.

# 4 EXPERIMENTS

# 4.1 SYNTHETIC MULTIVIEW MIXTURE MODEL

In this section, we apply DGCCA to a small synthetic data set to show how it preserves the generative structure of data sampled from a multiview mixture model. The data we use for this experiment are plotted in Figure 2. Points that share the same color across different views are sampled from the same mixture component.

![](images/43684cd8b287163264d1462a82aa4f6df03fdc69cf39d696b685624d07140663.jpg)  
Figure 2: Synthetic data used in in Section 4.1 experiments.

![](images/192b7fd676e5a71f4feb4f0f4de260353161228a6da0c5c4a16494b896ae53de.jpg)

![](images/8bd3305b34d5f17de6d1bf99e4fcc94208ba55f4db9ed1c09bdd095f76992c7b.jpg)

Importantly, in each view, there is no linear transformation of the data that separates the two mixture components, in the sense that the generative structure of the data could not be exploited by a linear model. This point is reinforced by Figure 3(a), which shows the two-dimensional representation  $G$  learned by applying (linear) GCCA to the data in Figure 2. The learned representation completely loses the structure of the data.

![](images/94c830ac1aa19301f10200b3ea4261c953a31b14f06af463f404fa06aac404bc.jpg)  
(a) GCCA

![](images/f918128e7c719607506cdee012f68c193dbfc5f19b612ccab4f501c28d21d7f1.jpg)  
(b) DGCCA  
Figure 3: The matrix  $G$  learned from applying (linear) GCCA or DGCCA to the data in Figure 2.

We can contrast the failure of GCCA to preserve structure with the result of applying DGCCA; in this case, the input neural networks had three hidden layers with ten units each with weights randomly initialized. We plotted the representation  $G$  learned by DGCCA in Figure 3 (b). In this representation, the mixture components are easily separated by a linear classifier; in fact, the structure is largely preserved even after projection onto the first coordinate of  $G$ .

It is also illustrative to consider the view-specific representations learned by DGCCA, that is, to consider the outputs of the neural networks that were trained to maximize the GCCA objective. We plotted the representations in Figure 4. For each view, we have learned a nonlinear mapping that

does remarkably well at making the mixture components linearly separable. Recall that absolutely no direct supervision was given about which mixture component each point was generated from. The only training signals available to the networks were the reconstruction errors between the network outputs and the learned representation  $G$ .

![](images/4e646e3acd4dca7b8ebb5ce2e4e1f42fbfc69876f48fbccf2951800c4c2a1b98.jpg)  
Figure 4: Outputs of the trained input neural networks in Section 4.1 applied to the data in Figure 2.

# 4.2 PHONEME CLASSIFICATION

In this section, we discuss experiments on the University of Wisconsin X-ray Microbeam Database (XRMB) (Westbury, 1994). XRMB contains acoustic and articulatory recordings as well as phonemic labels. We present phoneme classification results on the acoustic vectors projected using DCCA, GCCA, and DGCCA. We set acoustic and articulatory data as the two views and phoneme labels as the third view for GCCA and DGCCA. For classification, we run K-nearest neighbor classification (Cover & Hart, 1967) on the projected result.

# 4.2.1 DATA

We use the same train/tune/test split of the data as Arora & Livescu (2014). To limit experiment runtime, we use a subset of speakers for our experiments. We run a set of cross-speaker experiments using the male speaker JW11 for training and two splits of JW24 for tuning and testing. We also perform parameter tuning for the third view with 5-fold cross validation using a single speaker, JW11. For both experiments, we use acoustic and articulatory measurements as the two views in DCCA. Following the pre-processing in Andrew et al. (2013), we get 273 and 112 dimensional feature vectors for the first and second view respectively. Each speaker has  $\sim 50,000$  frames. For the third view in GCCA and DGCCA, we use 39-dimensional one-hot vectors corresponding to the labels for each frame, following Arora & Livescu (2014).

# 4.2.2 PARAMETERS

We use a fixed network size and regularization for the first two views, each containing three hidden layers with sigmoid activation functions. Hidden layers for the acoustic view were all width 1024, and layers in the articulatory view all had width 512 units. L2 penalty constants of 0.0001 and 0.01 were used to train the acoustic and articulatory view networks, respectively. The output layer dimension of each network is set to 30 for DCCA and DGCCA. For the 5-fold speaker-dependent experiments, we performed a grid search for the network sizes in  $\{128, 256, 512, 1024\}$  and covariance matrix regularization in  $\{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}\}$  for the third view in each fold. We fix the hyperparameters for these experiments optimizing the networks with minibatch stochastic gradient descent with a step size of 0.005, batch size of 2000, and no learning decay or momentum. The third view neural network had an L2 penalty of 0.0005.

# 4.2.3 RESULTS

As we show in Table 1, DGCCA improves upon both the linear multiview GCCA and the non-linear 2-view DCCA for both the cross-speaker and speaker-dependent cross-validated tasks.

In addition to accuracy, we examine the reconstruction error, i.e. the objective in Equation 3, obtained from the objective in GCCA and DGCCA. This sharp improvement in reconstruction error shows that a non-linear algorithm can better model the data.

![](images/6e3006c7009f68a12b0c92723c05ecd4eb9af8b79ca28bdb3a63544911f104b1.jpg)  
(a) GCCA  
Figure 5: The confusion matrix for speaker-dependent GCCA and DGCCA

![](images/e6e0767518c8c88ee2aa709b4cee5151a86b12e6361884a4e2057c26a4b0a10e.jpg)  
(b) DGCCA

In this experimental setup, DCCA under-performs the baseline of simply running KNN on the original acoustic view. Prior work considered the output of DCCA stacked on to the central frame of the original acoustic view (39 dimensions). This poor performance, in the absence of original features, indicates that it was not able to find a more informative projection than original acoustic features based on correlation with the articulatory view within the first 30 dimensions.

Table 1: KNN phoneme classification performance  

<table><tr><td rowspan="2">METHOD</td><td colspan="3">CROSS-SPEAKER</td><td colspan="3">SPEAKER-DEPENDENT</td></tr><tr><td>DEV ACC</td><td>TEST ACC</td><td>REC ERROR</td><td>DEV ACC</td><td>TEST ACC</td><td>REC ERROR</td></tr><tr><td>MFCC</td><td>48.89</td><td>49.28</td><td></td><td>66.27</td><td>66.22</td><td></td></tr><tr><td>DCCA</td><td>45.40</td><td>46.06</td><td></td><td>65.88</td><td>65.81</td><td></td></tr><tr><td>GCCA</td><td>49.59</td><td>50.18</td><td>40.67</td><td>69.52</td><td>69.78</td><td>40.39</td></tr><tr><td>DGCCA</td><td>53.78</td><td>54.22</td><td>35.89</td><td>72.62</td><td>72.33</td><td>20.52</td></tr></table>

To highlight the improvements of DGCCA over GCCA, Figure 5 presents a subset of the the confusion matrices on speaker-dependent test data. In particular, we observe large improvements in the classification of  $D$ ,  $F$ ,  $K$ ,  $SH$ ,  $V$  and  $Y$ . GCCA outperforms DGCCA for  $UH$  and  $DH$ . These matrices also highlight the common misclassifications that DGCCA improves upon. For instance, DGCCA rectifies the frequent misclassification of  $V$  as  $P$ ,  $R$  and  $B$  by GCCA. In addition, commonly incorrect classification of phonemes such as  $S$  and  $T$  is corrected by DGCCA, which enables better performance on other voiceless consonants such as like  $F$ ,  $K$  and  $SH$ . Vowels are classified with almost equal accuracy by both the methods.

# 4.3 TWITTER USER HASHTAG & FRIEND RECOMMENDATION

Linear multiview techniques are effective at recommending hashtag and friends for Twitter users (Benton et al., 2016). In this experiment, six views of a Twitter user were constructed by applying principal component analysis (PCA) to the bag-of-words representations of (1) tweets posted by the ego user, (2) other mentioned users, (3) their friends, and (4) their followers, as well as one-hot encodings of the local (5) friend and (6) follower networks. We learn and evaluate DGCCA models on identical training, development, and test sets as Benton et al. (2016), and evaluate the DGCCA representations on macro precision at 1000 (P@1000) and recall at 1000 (R@1000) for the hashtag and friend recommendation tasks described there.

We trained 40 different DGCCA model architectures, each with identical architectures for each view, where the width of the hidden and output layers,  $c_{1}$  and  $c_{2}$ , for each view are drawn uniformly from

[10, 1000], and the auxiliary representation width  $r$  is drawn uniformly from  $[10, c_2]^5$ . All networks used RLUs as activation functions, and were optimized with Adam (Kingma & Ba, 2014) for 200 epochs<sup>6</sup>. Networks were trained on  $90\%$  of 102,328 Twitter users, with  $10\%$  of users used as a tuning set to estimate heldout reconstruction error for model selection. We report development and test results for the best performing model on the downstream task development set. Learning rate was set to  $10^{-4}$  with an L1 and L2 regularization constants of 0.01 and 0.001 for all weights<sup>7</sup>.

Table 2: Dev/test performance at Twitter friend and hashtag recommendation tasks.  

<table><tr><td rowspan="2">ALGORITHM</td><td colspan="2">FRIEND</td><td colspan="2">HASHTAG</td></tr><tr><td>P@1000</td><td>R@1000</td><td>P@1000</td><td>R@1000</td></tr><tr><td>PCA[TEXT+NET]</td><td>0.445/0.439</td><td>0.149/0.147</td><td>0.011/0.008</td><td>0.312/0.290</td></tr><tr><td>GCCA[TEXT]</td><td>0.244/0.249</td><td>0.080/0.081</td><td>0.012/0.009</td><td>0.351/0.326</td></tr><tr><td>GCCA[TEXT+NET]</td><td>0.271/0.276</td><td>0.088/0.089</td><td>0.012/0.010</td><td>0.359/0.334</td></tr><tr><td>DGCCA[TEXT+NET]</td><td>0.297/0.268</td><td>0.099/0.090</td><td>0.013/0.010</td><td>0.385/0.373</td></tr><tr><td>WGCCA[TEXT]</td><td>0.269/0.279</td><td>0.089/0.091</td><td>0.012/0.009</td><td>0.357/0.325</td></tr><tr><td>WGCCA[TEXT+NET]</td><td>0.376/0.364</td><td>0.123/0.120</td><td>0.013/0.009</td><td>0.360/0.346</td></tr></table>

Table 2 displays the performance of DGCCA compared to PCA{text+net] (PCA applied to concatenation of view feature vectors), linear GCCA applied to the four text views, [text], and all views, [text+net], along with a weighted GCCA variant (WGCCA). We learned PCA, GCCA, and WGCCA representations of width  $r \in \{10,20,50,100,200,300,400,500,750,1000\}$ , and report the best performing representations on the development set.

There are several points to note: First is that DGCCA outperforms linear methods at hashtag recommendation by a wide margin in terms of recall. This is exciting because this task was shown to benefit from incorporating more than just two views from Twitter users. These results suggest that a nonlinear transformation of the input views can yield additional gains in performance. In addition, WGCCA models sweep over every possible weighting of views with weights in  $\{0, 0.25, 1.0\}$ . WGCCA has a distinct advantage in that the model is allowed to discriminatively weight views to maximize downstream performance. The fact that DGCCA is able to outperform WGCCA at hashtag recommendation is encouraging, since WGCCA has much more freedom to discard uninformative views, whereas the DGCCA objective forces networks to minimize reconstruction error equally across all views. As noted in Benton et al. (2016), only the friend network view was useful for learning representations for friend recommendation (corroborated by performance of PCA applied to friend network view), so it is unsurprising that DGCCA when applied to all views cannot compete with WGCCA representations learned on the single useful friend network view<sup>8</sup>.

# 5 CONCLUSION

We present DGCCA, a method for non-linear multiview representation learning from an arbitrary number of views. We show that DGCCA clearly outperforms prior work when using labels as a third view (Andrew et al., 2013; Arora & Livescu, 2014; Wang et al., 2015c), and can successfully exploit multiple views to learn user representations useful for downstream tasks such as hashtag recommendation for Twitter users. Most exciting is that the representation learning community now has a generic tool for learning non-linear representations from arbitrarily many views. To date, multiview learning techniques were either restricted to learning representations from no more than two views, or strictly linear transformations of the input views.

# REFERENCES

Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. The Journal of Machine Learning Research, 15(1):2773-2832, 2014.  
Galen Andrew, Raman Arora, Jeff Bilmes, and Karen Livescu. Deep canonical correlation analysis. In Proceedings of the 30th International Conference on Machine Learning, pp. 1247-1255, 2013.  
Raman Arora and Karen Livescu. Multi-view learning with supervision for transformed bottleneck features. In Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on, pp. 2499-2503. IEEE, 2014.  
Adrian Benton, Raman Arora, and Mark Dredze. Learning multiview embeddings of twitter users. In The 54th Annual Meeting of the Association for Computational Linguistics, pp. 14, 2016.  
Thomas M Cover and Peter E Hart. Nearest neighbor pattern classification. Information Theory, IEEE Transactions on, 13(1):21-27, 1967.  
David R Hardoon, Sandor Szedmak, and John Shawe-Taylor. Canonical correlation analysis: An overview with application to learning methods. Neural computation, 16(12):2639-2664, 2004.  
Paul Horst. Generalized canonical correlations and their applications to experimental data. Journal of Clinical Psychology, 17(4), 1961.  
Harold Hotelling. Relations between two sets of variates. Biometrika, pp. 321-377, 1936.  
Jon R. Kettenring. Canonical analysis of several sets of variables. Biometrika, 58(3):433-451, 1971.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Kaare Petersen and Michael Pedersen. The matrix cookbook, Nov 2012. URL http://www2.imm.dtu.dk/pubdb/p.php?3274. Version 20121115.  
Karthik Sridharan and Sham M Kakade. An information theoretic framework for multi-view learning. In Proceedings of COLT, 2008.  
Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. Unsupervised learning of acoustic features via deep canonical correlation analysis. In Proc. of the IEEE Int. Conf. Acoustics, Speech and Sig. Proc. (ICASSP'15), 2015a.  
Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. On deep multi-view representation learning. In Proc. of the 32nd Int. Conf. Machine Learning (ICML 2015), 2015b.  
Weiran Wang, Raman Arora, Karen Livescu, and Nathan Srebro. Stochastic optimization for deep cca via nonlinear orthogonal iterations. In Proceedings of the 53rd Annual Allerton Conference on Communication, Control and Computing (ALLERTON), 2015c.  
John R. Westbury. X-ray microbeam speech production database users handbook. In Waisman Center on Mental Retardation & Human Development University of Wisconsin Madison, WI 53705-2280, 1994.
