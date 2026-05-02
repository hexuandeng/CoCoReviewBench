# SVMAX: A FEATURE EMBEDDING REGULARIZER

Anonymous authors

Paper under double-blind review

# ABSTRACT

A neural network regularizer (e.g., weight decay) boosts performance by explicitly penalizing the complexity of a network. In this paper, we penalize inferior network activations – feature embeddings – which in turn regularize the network's weights implicitly. We propose singular value maximization (SVMax) to learn a uniform feature embedding. The SVMax regularizer integrates seamlessly with both supervised and unsupervised learning. During training, our formulation mitigates model collapse and enables larger learning rates. Thus, our formulation converges in fewer epochs, which reduces the training computational cost. We evaluate the SVMax regularizer using both retrieval and generative adversarial networks. We leverage a synthetic mixture of Gaussians dataset to evaluate SVMax in an unsupervised setting. For retrieval networks, SVMax achieves significant improvement margins across various ranking losses.

# 1 INTRODUCTION

A neural network's knowledge is embodied in both its weights and activations. This difference manifests in how network pruning and knowledge distillation tackle the model compression problem. While pruning literature Li et al. (2016); Luo et al. (2017); Yu et al. (2018) compresses models by removing less significant weights, knowledge distillation Hinton et al. (2015) reduces computational complexity by matching a cumbersome network's last layer activations (logits). This perspective, of weight-knowledge versus activation-knowledge, emphasizes how neural network literature is dominated by explicit weight regularizers. In contrast, this paper leverages singular value decomposition (SVD) to regularize a network through its last layer activations – its feature embedding.

Our formulation is inspired by principal component analysis (PCA). Given a set of points and their covariance, PCA yields the set of orthogonal eigenvectors sorted by their eigenvalues. The principal component (first eigenvector) is the axis with the highest variation (largest eigenvalue) as shown in Figure 1c. The eigenvalues from PCA, and similarly the singular values from SVD, provide insights about the embedding space structure. As such, by regularizing the singular values, we reshape the feature embedding.

The main contribution of this paper is to leverage the singular value decomposition of a network's activations to regularize the embedding space. We achieve this objective through singular value maximization (SVMax). The SVMax regularizer is oblivious to both the input-class (labels) and the sampling strategy. Thus it promotes a uniform embedding space in both supervised and unsupervised learning. Furthermore, we present a mathematical analysis of the mean singular value's lower and upper bounds. This analysis makes tuning the SVMax's balancing-hyperparameter easier, when the feature embedding is normalized to the unit circle.

The SVMax regularizer promotes a uniform embedding space. During training, SVMax speeds up convergence by enabling large learning rates. The SVMax regularizer integrates seamlessly with various ranking losses. We apply the SVMax regularizer to the last feature embedding layer, but the same formulation can be applied to intermediate layers. The SVMax regularizer mitigates model collapse in both retrieval networks and generative adversarial networks (GANs) Goodfellow et al. (2014); Srivastava et al. (2017); Metz et al. (2017). Furthermore, the SVMax regularizer is useful when training unsupervised feature embedding networks with a contrastive loss (e.g., CPC) Noroozi et al. (2017); Oord et al. (2018); He et al. (2019); Tian et al. (2019).

In summary, we propose singular value maximization to regularize the feature embedding. In addition, we present a mathematical analysis of the mean singular value's lower and upper bounds

![](images/06a7670bcddee02de8286c9a5932a82ab81be31b66433d546dd5f373eb764058.jpg)  
(a) Singular values: (11.25, 1.14)

![](images/11f1ebb9b02109bc71a75cfa7df97540a0a1f08a3bbf12878c45801722aa422d.jpg)  
Figure 1: Feature embeddings scattered over the  $2D$  unit circle. In (a), the features are polarized across a single axis; the singular value of the principal (horizontal) axis is large while singular value of the secondary (vertical) axis is small, respectively. In (b), the features are spread uniformly across both dimensions; both singular values are comparably large. (c) depicts the PCA analysis of a toy 2D Gaussian dataset to demonstrate our intuition. The principal component (green) has the highest eigenvalue, i.e., the axis with the highest variation, while the second component (red) has a smaller eigenvalue. Maximizing all eigenvalues promotes data dispersion across all dimensions. In this paper, we maximize the mean singular value to regularize the feature embedding and avoid a model collapse.  
(b) Singular values: (8.99, 6.86)

![](images/f0aa4c21b5875749cc476ceac49ca481ba38cd4a2b009204257739b696f353b4.jpg)  
(c) PCA of a toy dataset

to reduce hyperparameter tuning (Sec. 3). We quantitatively evaluate how the SVMax regularizer significantly boosts the performance of ranking losses (Sec. 4.1). And we provide a qualitative evaluation of using SVMax in the unsupervised learning setting via GAN training (Sec. 4.2).

# 2 RELATED WORK

Network weight regularizers dominate the deep learning regularizer literature, because they support a large spectrum of tasks and architectures. Singular value decomposition (SVD) has been applied as a weight regularizer in several recent works Zhang et al. (2018); Sedghi et al. (2018); Guo & Ye (2019). Zhang et al. (2018) employ SVD to avoid vanishing and exploding gradients in recurrent neural networks. Similarly, Guo & Ye (2019) bound the singular values of the convolutional layer around 1 to preserve the layer's input and output norms. A bounded output norm mitigates the exploding/vanishing gradient problem. Weight regularizers share the common limitation that they do not enforce an explicit feature embedding objective and are thus ineffective against model collapse.

Feature embedding regularizers have also been extensively studied, especially for classification networks Rippel et al. (2015); Wen et al. (2016); He et al. (2018); Hoffman et al. (2019); Taha et al. (2020). These regularizers aim to maximize class margins, class compactness, or both simultaneously. For instance, Wen et al. (2016) propose center loss to explicitly learn class representatives and thus promote class compactness. In classification tasks, test samples are assumed to lie within the same classes of the training set, i.e., closed-set identification. However, retrieval tasks, such as product re-identification, assume an open-set setting. Because of this, a retrieval network regularizer should aim to spread features across many dimensions to fully utilize the expressive power of the embedding space.

Recent literature Sablayrolles et al. (2018); Zhang et al. (2017) has recognized the importance of a spread-out feature embedding. However, this literature is tailored to triplet loss and therefore assumes a particular sampling procedure. In this paper, we leverage SVD as a regularizer because it is simple, differentiable Ionescu et al. (2015), and class oblivious. SVD has been used to promote low rank models to learn compact intermediate layer representations Kliegl et al. (2017); Sanyal et al. (2019). This helps compress the network and speed up matrix multiplications on embedded devices (iPhone and Raspberry Pi). In contrast, we regularize the embedding space through a high rank objective. By maximizing the mean singular value, we promote a higher rank representation - a spread-out embedding.

# 3 SINGULAR VALUE MAXIMIZATION (SVMAX)

We first introduce our mathematical notation. Let  $\mathcal{I}$  denote the image space and  $E_{\mathcal{I}} \in R^{d}$  denote the feature embeddings space, where  $d$  is the dimension of the features. A feature embedding network is a function  $F_{\theta}: \mathcal{I} \to E_{\mathcal{I}}$ , parameterized by the network's weights  $\theta$ . We quantify similarity between an image pair  $(\mathcal{I}_1, \mathcal{I}_2)$  via the Euclidean distance in feature space, i.e.,  $\|E_{\mathcal{I}_1} - E_{\mathcal{I}_2}\|_2$ .

During training, a  $2D$  matrix  $E \in R^{b \times d}$  stores  $b$  samples' embeddings, where  $b$  is the mini-batch size. Assuming  $b \geq d$ , the singular value decomposition (SVD) of  $E$  provides the singular values  $S = [s_1,.., s_i,.., s_d]$ , where  $s_1$  and  $s_d$  are the largest and smallest singular values, respectively. We maximize the mean singular value,  $s_\mu = \frac{1}{d} \sum_{i=1}^d s_i$ , to regularize the network's last layer activations - the feature embedding. By maximizing the mean singular value, the deep network spreads out its embeddings. This has the added benefit of implicitly regularizing the network's weights  $\theta$ . The proposed SVMax regularizer integrates with both supervised and unsupervised feature embedding networks as follows

$$
L _ {\mathrm {N N}} = L _ {r} - \lambda \frac {1}{d} \sum_ {i = 1} ^ {d} s _ {i} = L _ {r} - \lambda s _ {\mu}, \tag {1}
$$

where  $L_{r}$  is the original network loss and  $\lambda$  is a balancing hyperparameter.

Lower and Upper Bounds of the Mean Singular Value: One caveat to equation 1 is the hyperparameter  $\lambda$ . It is difficult to tune because the mean singular value  $s_{\mu}$  depends on the range of values inside  $E$  and its dimensions  $(b, d)$ . Thus, changing the batch size or embedding dimension requires a different  $\lambda$ . To address this, we utilize a common assumption in metric learning – the unit circle (L2-normalized) embedding assumption. This assumption provides both lower and upper bounds on ranking losses. This will allow us to impose lower and upper bounds on  $s_{\mu}$ .

For an L2-normalized embedding  $E$ , the largest singular value  $s_1$  is maximum when the matrix-rank of  $E$  equals one, i.e.,  $\text{rank}(E) = 1$ , and  $s_i = 0$  for  $i \in [2, d]$ . Horn & Johnson (1991) provide an upper bound on this largest singular value  $s_1$  as  $s^*(E) \leq \sqrt{||E||_1||E||_\infty}$ . This holds in equality for all L2-normalized  $E \in R^{b \times d}$  with  $\text{rank}(E) = 1$ . For an L2-normalized matrix  $E$  with  $||E||_1 = b$  and  $||E||_\infty = 1$ , this gives:

$$
s ^ {*} (E) = \sqrt {\left\| E \right\| _ {1} \left\| E \right\| _ {\infty}} = \sqrt {b}. \tag {2}
$$

Thus, the lower bound  $L$  on  $s_\mu$  is  $L = \frac{s^*(E)}{d} = \frac{\sqrt{b}}{d}$ .

Similarly, an upper bound is defined on the sum of the singular values Turkmen & Civiciv (2007); Kong et al. (2018); Friedland & Lim (2016). This summation is formally known as the nuclear norm of a matrix  $||E||_{*}$ . Hu (2015) established an upper bound on this summation using the Frobenius Norm  $||E||_{F}$  as follows

$$
\left| \left| E \right| \right| _ {*} \leq \sqrt {\frac {b \times d}{\operatorname* {m a x} (b , d)}} \left| \left| E \right| \right| _ {F}, \tag {3}
$$

where  $||E||_F = \left(\sum_{i=1}^{rows} \sum_{j=1}^{cols} |E_{ij}|^2\right)^{\frac{1}{2}} = \sqrt{b}$  because of the L2-normalization assumption.

Accordingly, the lower and upper bounds of  $s_{\mu}$  are  $[L,U] = \left[\frac{s^{*}(E)}{d},\frac{\|E\|_{*}}{d}\right]$ . With these bounds, we rewrite our final loss function as follows

$$
L _ {\mathrm {N N}} = L _ {r} + \lambda \exp \left(\frac {U - s _ {\mu}}{U - L}\right). \tag {4}
$$

The SVMax regularizer grows exponentially  $\in [1, e]$ . We employ this loss function in all our retrieval experiments. It is important to note that the L2-normalized assumption makes  $\lambda$  tuning easier, but it is not required. Equation 4 makes the hyperparameter  $\lambda$  only dependent on the range of  $L_{r}$  which is also bounded for ranking losses.

Lower and Upper Bounds of Ranking Losses: We briefly show that ranking losses are bounded when assuming an L2-normalized embedding. Equations 5 and 6 show triplet and contrastive losses,

Table 1: Quantitative evaluation on CUB-200-2011 with batch size  $b = 144$ , embedding dimension  $d = 128$  and multiple learning rates  $lr = \{0.01, 0.001, 0.0001\}$ .  $\triangle_{R@1}$  column indicates the R@1 improvement margin relative to the vanilla ranking loss. A large learning rate  $lr$  increases the chance of model collapse, while a small  $lr$  slows convergence.  $\lambda$  is dependent on the ranking loss.  

<table><tr><td rowspan="2">Method</td><td colspan="4">lr = 0.01</td><td colspan="4">lr = 0.001</td><td colspan="4">lr = 0.0001</td></tr><tr><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td></tr><tr><td colspan="13">Contrastive</td></tr><tr><td>Vanilla</td><td>0.435</td><td>25.73</td><td>58.88</td><td>-</td><td>0.443</td><td>28.68</td><td>64.70</td><td>-</td><td>0.413</td><td>24.49</td><td>59.54</td><td>-</td></tr><tr><td>Spread-out</td><td>0.440</td><td>24.54</td><td>57.16</td><td>-1.18</td><td>0.479</td><td>32.12</td><td>66.83</td><td>3.44</td><td>0.458</td><td>31.85</td><td>67.45</td><td>7.36</td></tr><tr><td>SVMax (Ours)</td><td>0.527</td><td>41.26</td><td>75.24</td><td>15.53</td><td>0.547</td><td>43.11</td><td>77.26</td><td>14.43</td><td>0.449</td><td>29.56</td><td>65.50</td><td>5.06</td></tr><tr><td colspan="13">Triplet Loss</td></tr><tr><td>Vanilla</td><td>0.496</td><td>29.34</td><td>67.96</td><td>-</td><td>0.477</td><td>28.88</td><td>64.60</td><td>-</td><td>0.449</td><td>24.86</td><td>61.14</td><td>-</td></tr><tr><td>Spread-out</td><td>0.545</td><td>43.60</td><td>76.98</td><td>14.26</td><td>0.557</td><td>44.02</td><td>78.54</td><td>15.14</td><td>0.435</td><td>28.33</td><td>64.33</td><td>3.46</td></tr><tr><td>SVMax λ = 1 (Ours)</td><td>0.556</td><td>43.21</td><td>77.43</td><td>13.88</td><td>0.527</td><td>39.13</td><td>74.17</td><td>10.25</td><td>0.401</td><td>25.07</td><td>60.01</td><td>0.20</td></tr><tr><td>SVMax λ = 0.1 (Ours)</td><td>0.547</td><td>43.80</td><td>77.97</td><td>14.47</td><td>0.557</td><td>43.89</td><td>78.44</td><td>15.01</td><td>0.436</td><td>28.22</td><td>64.40</td><td>3.36</td></tr><tr><td colspan="13">N-pair</td></tr><tr><td>Vanilla</td><td>0.402</td><td>18.96</td><td>50.32</td><td>-</td><td>0.452</td><td>27.65</td><td>63.10</td><td>-</td><td>0.455</td><td>31.41</td><td>66.95</td><td>-</td></tr><tr><td>Spread-out</td><td>0.416</td><td>20.64</td><td>52.80</td><td>1.69</td><td>0.483</td><td>32.46</td><td>66.41</td><td>4.81</td><td>0.474</td><td>33.39</td><td>68.80</td><td>1.98</td></tr><tr><td>SVMax (Ours)</td><td>0.483</td><td>34.62</td><td>68.11</td><td>15.67</td><td>0.547</td><td>43.79</td><td>77.31</td><td>16.14</td><td>0.488</td><td>34.13</td><td>69.92</td><td>2.72</td></tr><tr><td colspan="13">Angular</td></tr><tr><td>Vanilla</td><td>0.470</td><td>28.54</td><td>60.03</td><td>-</td><td>0.508</td><td>38.94</td><td>72.82</td><td>-</td><td>0.538</td><td>41.80</td><td>76.18</td><td>-</td></tr><tr><td>Spread-out</td><td>.471</td><td>28.29</td><td>60.26</td><td>-0.25</td><td>0.508</td><td>38.96</td><td>72.86</td><td>0.02</td><td>0.538</td><td>41.81</td><td>76.23</td><td>0.02</td></tr><tr><td>SVMax (Ours)</td><td>0.487</td><td>32.88</td><td>66.27</td><td>4.34</td><td>0.523</td><td>41.29</td><td>74.71</td><td>2.35</td><td>0.531</td><td>42.00</td><td>76.30</td><td>0.20</td></tr></table>

respectively, and their corresponding bounds  $[L,U]$

$$
\mathrm {T L} _ {(a, p, n) \in T} = \left[ \left(D (\lfloor a \rfloor , \lfloor p \rfloor) - D (\lfloor a \rfloor , \lfloor n \rfloor) + m\right) \right] _ {+} \quad \xrightarrow {[ L , U ]} [ 0, 2 + m ], \tag {5}
$$

$$
\mathrm {C L} _ {(x, y) \in P} = (1 - \delta_ {x, y}) D (\lfloor x \rfloor , \lfloor y \rfloor)) + \delta_ {x, y} [ m - D (\lfloor x \rfloor , \lfloor y \rfloor)) ] _ {+} \quad \stackrel {[ L, U ]} {\longrightarrow} \quad [ 0, 2 ], \tag {6}
$$

where  $[\bullet]_{+} = \max(0, \bullet)$ ,  $m < 2$  is the margin between classes, since 2 is the maximum distance on the unit circle.  $\lfloor \cdot \rfloor$  and  $D(\cdot)$  are the embedding and Euclidean distance functions, respectively. In equation 5,  $a$ ,  $p$ , and  $n$  are the anchor, positive, and negative images in a single triplet  $(a, p, n)$  from the triplets set  $T$ . In equation 6,  $x$  and  $y$  form a single pair of images from the pairs set  $P$ .  $\delta_{x,y} = 1$  when  $x$  and  $y$  belong to different classes; zero otherwise. In the supplementary material, we (1) show similar analysis for N-pair and angular losses, (2) provide an SVMax evaluation on small training batches, i.e.,  $b < d$ , and (3) evaluate the computational complexity of SVMax.

# 4 EXPERIMENTS

In this section, we evaluate SVMax using both supervised and unsupervised learning. We leverage retrieval and generative adversarial networks for quantitative and qualitative evaluations.

# 4.1 RETRIEVAL NETWORKS

Technical Details: We evaluate the SVMax regularizer quantitatively using three datasets: CUB-200-2011 Wah et al. (2011), Stanford CARS196 Krause et al. (2013), and Stanford Online Products Oh Song et al. (2016). We use GoogLeNet Szegedy et al. (2015) and ResNet50 He et al. (2016); both pretrained on ImageNet Deng et al. (2009) and fine-tuned for  $K$  iterations. These are standard retrieval datasets and architectures. By default, the embedding  $\in R^{d=128}$  is normalized to the unit circle. In all experiments, a batch size  $b = 144$  is employed, the learning rate  $lr$  is fixed for  $K/2$  iterations then decayed polynomially to  $1e - 7$  at iteration  $K$ . We use the SGD optimizer with 0.9 momentum. Each batch contains  $p$  different classes and  $l$  different samples per class. For example, triplet loss employs  $p = 24$  different classes and  $l = 6$  instances per class. The mini-batch of N-pair loss contains 72 classes and a single positive pair per class, i.e.  $p = 72$  and  $l = 2$ . This same mini-batch setting is used for angular loss. For contrastive loss,  $p = 36$  and  $l = 4$  are divided into 72 positive and 72 negative pairs. For CUB-200 and CARS196,  $K = 5,000$  iterations; for Stanford Online Products,  $K = 20,000$ .

Table 2: Quantitative evaluation on Stanford Online Products.  

<table><tr><td rowspan="2">Method</td><td colspan="4">lr = 0.01</td><td colspan="4">lr = 0.001</td><td colspan="4">lr = 0.0001</td></tr><tr><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td><td>NMI</td><td>R@1</td><td>R@8</td><td>ΔR@1</td></tr><tr><td colspan="13">Contrastive</td></tr><tr><td>Vanilla</td><td>0.816</td><td>18.23</td><td>34.07</td><td>-</td><td>0.820</td><td>28.70</td><td>43.27</td><td>-</td><td>0.813</td><td>34.30</td><td>48.49</td><td>-</td></tr><tr><td>Spread-out</td><td>0.811</td><td>18.87</td><td>35.74</td><td>0.64</td><td>0.822</td><td>29.97</td><td>46.69</td><td>1.27</td><td>0.824</td><td>36.15</td><td>51.22</td><td>1.85</td></tr><tr><td>SVMax (Ours)</td><td>0.875</td><td>61.82</td><td>78.90</td><td>43.59</td><td>0.854</td><td>53.94</td><td>70.92</td><td>25.25</td><td>0.832</td><td>41.96</td><td>57.44</td><td>7.66</td></tr><tr><td colspan="13">Triplet Loss</td></tr><tr><td>Vanilla</td><td>0.891</td><td>71.96</td><td>86.24</td><td>-</td><td>0.873</td><td>64.09</td><td>80.07</td><td>-</td><td>0.840</td><td>46.29</td><td>62.57</td><td>-</td></tr><tr><td>Spread-out</td><td>0.890</td><td>71.60</td><td>85.73</td><td>-0.36</td><td>0.872</td><td>64.23</td><td>80.10</td><td>0.14</td><td>0.840</td><td>46.68</td><td>63.04</td><td>0.39</td></tr><tr><td>SVMax λ = 1 (Ours)</td><td>0.868</td><td>63.82</td><td>80.95</td><td>-8.15</td><td>0.857</td><td>58.04</td><td>75.14</td><td>-6.04</td><td>0.836</td><td>44.62</td><td>60.76</td><td>-1.67</td></tr><tr><td>SVMax λ = 0.1 (Ours)</td><td>0.889</td><td>71.48</td><td>85.97</td><td>-0.49</td><td>0.872</td><td>64.23</td><td>80.14</td><td>0.14</td><td>0.840</td><td>46.64</td><td>62.95</td><td>0.35</td></tr><tr><td colspan="13">N-pair</td></tr><tr><td>Vanilla</td><td>0.798</td><td>12.86</td><td>24.53</td><td>-</td><td>0.815</td><td>23.83</td><td>38.97</td><td>-</td><td>0.818</td><td>33.98</td><td>48.56</td><td>-</td></tr><tr><td>Spread-out</td><td>0.803</td><td>16.58</td><td>31.91</td><td>3.72</td><td>0.824</td><td>32.88</td><td>50.34</td><td>9.05</td><td>0.825</td><td>37.39</td><td>52.55</td><td>3.40</td></tr><tr><td>SVMax (Ours)</td><td>0.871</td><td>57.76</td><td>76.05</td><td>44.90</td><td>0.858</td><td>54.70</td><td>71.57</td><td>30.87</td><td>0.835</td><td>43.04</td><td>58.78</td><td>9.06</td></tr><tr><td colspan="13">Angular</td></tr><tr><td>Vanilla</td><td>0.883</td><td>62.83</td><td>80.13</td><td>-</td><td>0.885</td><td>66.93</td><td>82.12</td><td>-</td><td>0.856</td><td>54.29</td><td>71.14</td><td>-</td></tr><tr><td>Spread-out</td><td>0.883</td><td>62.73</td><td>79.96</td><td>-0.10</td><td>0.885</td><td>66.91</td><td>82.09</td><td>-0.02</td><td>0.856</td><td>54.30</td><td>71.10</td><td>0.02</td></tr><tr><td>SVMax (Ours)</td><td>0.885</td><td>65.44</td><td>81.73</td><td>2.61</td><td>0.884</td><td>67.28</td><td>82.47</td><td>0.35</td><td>0.855</td><td>54.88</td><td>71.47</td><td>0.59</td></tr></table>

Baselines: We evaluate the SVMax regularizer using contrastive Hadsell et al. (2006), hard triplet Hoffer & Ailon (2015); Hermans et al. (2017), N-pair Sohn (2016) and angular Wang et al. (2017) losses. We use the margin  $m = 1$  for contrastive loss,  $m = 0.2$  for triplet loss, and the angle bound  $\alpha = 45^{\circ}$  for angular loss. Similar to SVMax, multiple regularizers Kumar et al. (2016); Zhang et al. (2017); Sanyal et al. (2019); Chen & Deng (2019) promote a uniform embedding space. Unlike SVMax, these regularizers require a supervised setting to push anchor-negative pairs apart. We employ the spread-out regularizer Zhang et al. (2017) as a baseline for its simplicity, with default hyperparameter  $\alpha = 1$ . To enable the spread-out regularizer on non-triplet ranking losses, we pair every anchor with a random negative sample from the training mini-batch.

Evaluation Metrics: For quantitative evaluation, we use the Recall@K metric and Normalized Mutual Info (NMI) on the test split.

The hyperparameter:  $\lambda = 1$  for both contrastive and N-pair losses,  $\lambda = 0.1$  for triplet loss, and  $\lambda = 2$  for angular loss. We fix  $\lambda$  across datasets, architectures, and other hyperparameters  $(b,d)$ .

Results: Tables 1 and 2 present quantitative retrieval evaluation on CUB-200 and Stanford Online Products datasets – both using GoogLeNet. These tables provide in depth analysis and emphasize our improvement margins on a small and large dataset. Figure 2 provides quantitative evaluation on Stanford CARS196. We report the qualitative retrieval evaluation and quantitative evaluation on ResNet50 in the supplementary material. Our training hyperparameters – learning rate  $lr$  and number of iterations  $K$  – do not favor a particular ranking loss.

We evaluate SVM on various learning rates. A large learning rate, e.g.,  $lr = 0.01$ , speeds up convergence, but increases the chance of model collapse. In contrast, a small rate, e.g.,  $lr = 0.0001$ , is likely to avoid model collapse but is slow to converge. This undesirable effect is tolerable for small datasets – where increasing the number of training iterations  $K$  does not drastically increase the overall training time – but it is infeasible for large datasets. For contrastive and N-pair losses, SVM is significantly superior to both the vanilla and spread-out baselines, especially with a large learning rate. A small  $lr$  slows convergence and all approaches become equivalent. The spread-out regularizer Zhang et al. (2017) and its hyperparameters are tuned for triplet loss. Thus, for this particular ranking loss, the SVM and spread-out regularizers are on par.

In our experiments, we employ a large learning rate because it is the simplest factor to introduce a model collapse. However, the learning rate is not the only factor. Another factor is the training dataset size and its intra-class variations. A small dataset with large intra-class variations increases the chances of a model collapse. For example, a pair of dissimilar birds from the same class justifies a model collapse when coupled with a large learning rate. The hard triplet loss experiments emphasize this point because every anchor is paired with the hardest positive and negative samples.

![](images/530d3639bf6c30efa1c294bb7ac6efff2ce95bd3559c277d313fc59ceafd0ecc.jpg)

![](images/0069a4b1007789d5b72430cd4c441e4e2df2a7524646368ba155d1732129b655.jpg)

![](images/ffc32572b3ce656820c94d7c21a7bac2def6ffbfb203b9a09392bf06d26f55f5.jpg)

![](images/bf9acefd88f556cb2452ab35c248a9cda57db03556bbdb72e0ec0d78ad3cc31c.jpg)

![](images/d923f67fc63406904e2c9b4a58df426a2227d7bac4e976fb54d276c886eb1072.jpg)  
Figure 2: Quantitative evaluation on Stanford CARS196. X and Y-axis denote the learning rate  $lr$  and recall@1 performance, respectively.

On small fine-grained datasets like CUB-200 or CARS196, the vanilla hard triplet loss suffers significantly. Yet, the same implementation is superior on a big dataset like Stanford Online Products. By carefully tuning the training hyperparameter on CUB-200, it is possible to avoid a degenerate solution. However, this tedious tuning process is unnecessary when using either the spread-out or the SVMax regularizer.

The vanilla N-pair loss underperforms because it does not support feature embedding on the unit circle. Both spread-out and SVMax mitigate this limitation. For angular loss, a bigger  $\lambda = 2$  is employed to cope with the angular loss range. SVMax is a class oblivious regularizer. Thus,  $\lambda$  should be significant enough to contribute to the loss function without dominating the ranking loss.

Wu et al. (2017) show that the distance between any anchor-negative pair, which is randomly sampled from an  $n$ -dimensional unit sphere, follows the normal distribution  $N(\sqrt{2}, \frac{1}{2n})$ . This mean distance  $\sqrt{2}$  is large relative to the triplet loss margin  $m = 0.2$ , but comparable to the contrastive loss margin  $m = 1$ . Accordingly, triplet loss converges to zero after a few iterations, because most triplets satisfy the margin  $m = 0.2$  constraint. When triplet loss equals zero, the SVMax regularizer with  $\lambda = 1$  becomes the dominant term. However, the SVMax regularizer should not dominate because it is oblivious to data annotations; it equally pushes anchor-positive and anchor-negative pairs apart. Reducing  $\lambda$  to 0.1 solves this problem.

A less aggressive triplet loss Schroff et al. (2015); Xuan et al. (2020) is another way to avoid model collapse. For instance, Schroff et al. (2015) have proposed a triplet loss variant that employs semi-hard negatives. The semi-hard triplet loss is more stable than the aggressive hard triplet and lifted structured losses Oh Song et al. (2016). Unfortunately, the semi-hard triplet loss assumes a very large mini-batch ( $b = 1,800$  in Schroff et al. (2015)), which is impractical. Furthermore, when model collapse is avoided, aggressive triplet loss variants achieve superior performance Hermans et al. (2017). In contrast, the SVMax regularizer only requires a larger mini-batch than the embedding dimension, i.e.,  $b \geq d$ , a natural constraint for retrieval networks which favor compact embedding dimensions. Additionally, SVMax does not make any assumption about the sampling procedure. Thus, unlike Sablayrolles et al. (2018); Zhang et al. (2017), SVMax supports various supervised ranking losses.

# 4.2 GENERATIVE ADVERSARIAL NETWORKS

Model collapse is one of the main challenges of training generative adversarial networks (GANs) Metz et al. (2017); Srivastava et al. (2017); Mao et al. (2019); Salimans et al. (2016). To tackle this challenge, Metz et al. (2017) propose an unrolled-GAN to prevent the generator from overfitting to the discriminator. In an unrolled-GAN, the generator observes the discriminator for  $l$  steps before updating the generator's parameters using the gradient from the final step. Alternatively, we leverage the simpler SVMax regularizer to avoid model collapse. We evaluate our regularizer using a simple GAN on a 2D mixture of 8 Gaussians arranged in a circle. This 2D baseline Metz et al. (2017); Srivastava et al. (2017); Bang & Shim (2018) provides a simple qualitative evaluation and demonstrates SVMax's potential in unsupervised learning. We leverage this simple baseline because we assume  $b \geq d$ , which does not hold for images.

<table><tr><td>Method</td><td>Step 1</td><td>Step 5k</td><td>Step 10k</td><td>Step 15k</td><td>Step 20k</td><td>Step 25k</td><td>Target</td></tr><tr><td>Vanilla GAN</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla GAN + SVMax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unrolled GAN (5 steps)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unrolled GAN (5 steps) + SVMax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 3: The SVMax regularizer mitigates model collapse in a GAN trained on a toy 2D mixture of Gaussians dataset. Columns show heatmaps of the generator distributions at different training steps (iterations). The final column shows the groundtruth distribution. The first row shows the distributions generated by training a vanilla GAN suffering a model collapse. The second row shows the generated distribution when penalizing the generator's fake embedding with the SVMax regularizer. The third and fourth rows show two distributions generated using an unrolled-GAN with and without the SVMax regularizer, respectively. This high resolution figure is best viewed on a screen with zoom capabilities.

Figure 3 shows the dynamics of the GAN generator through time. We use a public PyTorch implementation<sup>1</sup> of Metz et al. (2017). We made a single modification to the code to use a relatively large learning rate, i.e.,  $lr = 0.025$  for both the generator and discriminator. This single modification is a simple and fast way to induce model collapse. The mixture of Gaussians circle has a radius  $r = 2$ , i.e., the generated fake embedding is neither L2-normalized nor strictly bounded by a network layer. We kept the radius parameter unchanged to emphasize that neither L2-normalization nor strict-bounds are required. To mitigate the impact of lurking variables (e.g., random network initialization and mini-batch sampling), we fix the random generator's seed for all experiments. We apply SVMax to a vanilla and an unrolled GAN for five steps. We apply the vanilla SVMax regularizer (Eq. 1), i.e.,  $L_{\mathrm{NN}} = L_{\mathrm{GAN}} - \lambda s_{\mu}$ , where  $\lambda = 0.01$  and  $s_{\mu}$  is mean singular value of the generator fake embedding.

GANs are typically used to generate high resolution images. This high-resolution output is the main limitation of the SVMax regularizer. The current formulation assumes the batch size is bigger than the embedding dimension, i.e.,  $b \geq d$ . This constraint is trivial for the Gaussians mixture 2D dataset and retrieval networks with a compact embedding dimensionality (e.g.,  $d = \{128, 256\}$ ). However, this constraint hinders high resolution image generators because the mini-batch size constraint becomes  $b \geq W \times H \times C$ , where  $W$ ,  $H$ , and  $C$  are the generated image's width, height, and number of channels, respectively. Nevertheless, this GAN experiment emphasizes the potential of the SVMax regularizer in unsupervised learning.

# 4.3 ABLATION STUDY

In this section, we evaluate two hypotheses: (1) the SVMax regularizer boosts retrieval performance because it learns a uniform feature embedding, (2) the same SVMax hyperparameter  $\lambda$  supports different embedding dimensions and batch sizes – the main objective of the mean singular value's bounds analysis.

To evaluate the SVMax regularizer's impact on feature embeddings, we embed the MNIST dataset onto the 2D unit circle. In this experiment, we used a tiny CNN (one convolutional layer and one hidden layer). Figure 4 shows the embedding space after training for  $t$  epochs. When using the SVMax regularizer, the feature embeddings spread out more uniformly and rapidly than the vanilla contrastive loss.

<table><tr><td>Method</td><td>Epoch 1</td><td>Epoch 2</td><td>Epoch 4</td><td>Epoch 8</td><td>Epoch 16</td><td>Epoch 32</td><td>Epoch 64</td></tr><tr><td>Contrastive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Contrastive + SVMax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/c922d6d8ef0a6be9d837e93c35c87ee660e4b9dc119d4a837eb8f993e29abd5c.jpg)  
Figure 4: Qualitative feature embedding evaluation using the MNIST dataset projected onto the 2D unit circle. The first row shows the feature embedding learned using a vanilla contrastive loss and the second row applies the SVMax-regularized. A random subset of the test split is projected for visualization purpose. Different colors denote different classes. The regularized feature embedding spreads out uniformly and rapidly. The supplementary material shows the feature embedding evolves vividly up to 200 epochs. This high resolution figure is best seen on a screen.  
Figure 5: Quantitative evaluation on CUB-200-2011 with various batch sizes  $b = \{288,72\}$  and embedding dimensions  $d = \{256,64\}$  to demonstrate the stability of our hyperparameter.  $\lambda = 1$  for contrastive loss and  $\lambda = 0.1$  for triplet loss.

The mean singular value bound analysis makes tuning the hyperparameter  $\lambda$  easier. This hyperparameter becomes only dependent on the ranking loss and independent of both the batch size and the embedding dimension. Figure 5 presents a quantitative evaluation using the CUB-200 dataset. We explore various batch sizes  $b = \{288,72\}$  and embedding dimensions  $d = \{256,64\}$ . We employ a MobileNetV2 Sandler et al. (2018) to fit the big batch  $b = 288$  on a 24GB GPU. The supplementary material contains a similar evaluation on the Stanford Online Products and CARS196 datasets.

# 5 CONCLUSION

We have proposed singular value maximization (SVMax) as a feature embedding regularizer. SVMax promotes a uniform embedding, mitigates model collapse, and enables large learning rates. Unlike other embedding regularizers, the SVMax regularizer supports a large spectrum of ranking losses. Moreover, it is oblivious to data annotation and, as such, supports both supervised and unsupervised learning. Qualitative evaluation using a generative adversarial network demonstrates SVMax's potential in unsupervised learning. Quantitative retrieval evaluation highlights significant performance improvements due to the SVMax regularizer.

# REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.  
Duhyeon Bang and Hyunjung Shim. Mggan: Solving mode collapse using manifold guided training. arXiv preprint arXiv:1804.04391, 2018.  
Binghui Chen and Weihong Deng. Energy confused adversarial metric learning for zero-shot image retrieval and clustering. In AAAI, 2019.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.  
Shmuel Friedland and Lek-Heng Lim. The computational complexity of duality. SIAM Journal on Optimization, 26(4):2378-2393, 2016.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2014.  
Peichang Guo and Qiang Ye. On regularization for a convolutional kernel in neural networks. arXiv preprint arXiv:1906.04866, 2019.  
Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. arXiv preprint arXiv:1911.05722, 2019.  
Xinwei He, Yang Zhou, Zhichao Zhou, Song Bai, and Xiang Bai. Triplet-center loss for multi-view 3d object retrieval. arXiv preprint arXiv:1803.06189, 2018.  
Alexander Hermans, Lucas Beyer, and Bastian Leibe. In defense of the triplet loss for person re-identification. arXiv preprint arXiv:1703.07737, 2017.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Elad Hoffer and Nir Ailon. Deep metric learning using triplet network. In International Workshop on Similarity-Based Pattern Recognition, 2015.  
Judy Hoffman, Daniel A Roberts, and Sho Yaida. Robust learning with jacobian regularization. arXiv preprint arXiv:1908.02729, 2019.  
Roger A Horn and Charles R Johnson. Topics in matrix analysis: Cambridge university press. Cambridge, UK, 1991.  
Shenglong Hu. Relations of the nuclear norm of a tensor and its matrix flattenings. Linear Algebra and its Applications, 478:188-199, 2015.  
Catalin Ionescu, Orestis Vantzos, and Cristian Sminchisescu. Training deep networks with structured layers by matrix backpropagation. arXiv preprint arXiv:1509.07838, 2015.  
Markus Kliegl, Siddharth Goyal, Kexin Zhao, Kavya Srinet, and Mohammad Shoeybi. Trace norm regularization and faster inference for embedded speech recognition rnns. arXiv preprint arXiv:1710.09026, 2017.  
Xu Kong, Jicheng Li, and Xiaolong Wang. New estimations on the upper bounds for the nuclear norm of a tensor. Journal of inequalities and applications, 2018(1):282, 2018.  
Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, 2013.

BG Kumar, Gustavo Carneiro, Ian Reid, et al. Learning local image descriptors with deep siamese and triplet convolutional networks by minimising global loss functions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5385-5394, 2016.  
Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016.  
Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In ICCV, 2017.  
Qi Mao, Hsin-Ying Lee, Hung-Yu Tseng, Siwei Ma, and Ming-Hsuan Yang. Mode seeking generative adversarial networks for diverse image synthesis. In CVPR, 2019.  
Luke Metz, Ben Poole, David Pfau, and Jascha Sohl-Dickstein. Unrolled generative adversarial networks. 2017.  
Mehdi Noroozi, Hamed Piri siavash, and Paolo Favaro. Representation learning by learning to count. In ICCV, 2017.  
Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. Deep metric learning via lifted structured feature embedding. In CVPR, 2016.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Oren Rippel, Manohar Paluri, Piotr Dollar, and Lubomir Bourdev. Metric learning with adaptive density discrimination. arXiv preprint arXiv:1511.05939, 2015.  
Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Hervé Jégou. Spreading vectors for similarity search. arXiv preprint arXiv:1806.03198, 2018.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In NIPS, 2016.  
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. *Mobilenetv2: Inverted residuals and linear bottlenecks*. In *CVPR*, 2018.  
Amartya Sanyal, Varun Kanade, and Philip H. S. Torr. Learning low-rank representations. arXiv preprint arXiv:1804.07090, 2019.  
Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In CVPR, 2015.  
Hanie Sedghi, Vineet Gupta, and Philip M Long. The singular values of convolutional layers. arXiv preprint arXiv:1805.10408, 2018.  
Kihyuk Sohn. Improved deep metric learning with multi-class n-pair loss objective. In NIPS, 2016.  
Akash Srivastava, Lazar Valkov, Chris Russell, Michael U Gutmann, and Charles Sutton. Veegan: Reducing mode collapse in gans using implicit variational learning. In *NSIP*, 2017.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In CVPR, 2015.  
Ahmed Taha, Yi-Ting Chen, Teruhisa Mitsu, Abhinav Shrivastava, and Larry Davis. Boosting standard classification architectures through a ranking regularizer. In WACV, 2020.  
Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849, 2019.  
Ramazan Turkmen and Haci Civiciv. Some bounds for the singular values of matrices. Applied Mathematical Sciences, 1(49):2443-2449, 2007.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.  
Jian Wang, Feng Zhou, Shilei Wen, Xiao Liu, and Yuanqing Lin. Deep metric learning with angular loss. In ICCV, 2017.  
Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao. A discriminative feature learning approach for deep face recognition. In ECCV, 2016.  
Chao-Yuan Wu, R Manmatha, Alexander J Smola, and Philipp Krahenbuhl. Sampling matters in deep embedding learning. In ICCV, 2017.  
Hong Xuan, Abby Stylianou, and Robert Pless. Improved embeddings with easy positive triplet mining. In WACV, 2020.  
Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In CVPR, 2018.  
Jiong Zhang, Qi Lei, and Inderjit S Dhillon. Stabilizing gradients for deep neural networks via efficient svd parameterization. arXiv preprint arXiv:1803.09327, 2018.  
Xu Zhang, Felix X Yu, Sanjiv Kumar, and Shih-Fu Chang. Learning spread-out local feature descriptors. In ICCV, 2017.  
Yizhe Zhu, Mohamed Elhoseiny, Bingchen Liu, Xi Peng, and Ahmed Elgammal. A generative adversarial approach for zero-shot learning from noisy texts. In CVPR, 2018.