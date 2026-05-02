# LEARNING JOINT WASSERSTEIN AUTO-ENCODERS FOR JOINT DISTRIBUTION MATCHING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the joint distribution matching problem which aims at learning bidirectional mappings to match the joint distribution of two domains. This problem occurs in unsupervised image-to-image translation and video-to-video synthesis tasks, which, however, has two critical challenges: (i) it is difficult to exploit sufficient information from the joint distribution; (ii) how to theoretically and experimentally evaluate the generalization performance remains an open question. To address the above challenges, we propose a new optimization problem and design a novel Joint Wasserstein Auto-Encoders (JWAE) to minimize the Wasserstein distance of the joint distributions in two domains. We theoretically prove that the generalization ability of the proposed method can be guaranteed by minimizing the Wasserstein distance of joint distributions. To verify the generalization ability, we apply our method to unsupervised video-to-video synthesis by performing video frame interpolation and producing visually smooth videos in two domains, simultaneously. Both qualitative and quantitative comparisons demonstrate the superiority of our method over several state-of-the-arts.

# 1 INTRODUCTION

The joint distribution matching problem has attracted extensive attention in computer vision, such as unsupervised unsupervised image-to-image translation (I2IT) (Zhu et al., 2017; Liu et al., 2017) and video-to-video synthesis (V2VS) (Bashkirova et al., 2018). The goal of this problem is to learn the bidirectional mappings between unpaired data in two different domains. Unlike the marginal distribution in each domain, learning a joint distribution is often ignored and has the following two critical challenges.

The first key challenge, from a probabilistic modeling perspective, is how to exploit the joint distribution of unpaired data by learning the bidirectional mappings between two different domains. In the unsupervised learning setting, there are two sets of samples drawn separately from two marginal distributions in two domains. Based on the coupling theory (Lindvall, 2002), there exist an infinite set of joint distributions given two marginal distributions, and hence infinite bidirectional mappings between two different domains. Therefore, directly learning the joint distribution without additional information between the marginal distributions is a highly ill-posed problem. Recently, many studies (Zhu et al., 2017; Yi et al., 2017; Kim et al., 2017) have been proposed to learn the mappings in two domains separately, which may incur the joint distribution mismatching issue. Therefore, how to exploit sufficient information from the joint distribution still remains an open question.

Another important challenge is that the generalization ability w.r.t. the learned joint distribution of two different domains is still unknown. Existing theoretical results (Pan et al., 2018; Galanti et al., 2018) ignore the joint distribution of different data and cannot guarantee the generalization ability of such joint distribution. Moreover, it is also very hard to evaluate the generalization ability practically. Regarding this issue, according to (Bojanowski et al., 2018), the generalization ability can be evaluated by the interpolation performance in the target domain. In this sense, we can extend image-to-image translation to video space by performing video interpolation in one domain and investigating the performance of the translated video in another domain. To achieve this, one may directly apply existing unsupervised image-to-image translation methods (Zhu et al., 2017; Kim et al., 2017; Yi et al., 2017). However, these methods may result in significantly incoherent videos

with low visual quality. Therefore, it is important to design an effective joint distribution learning method and provide necessary theoretical analysis.

Regarding the above two challenges, in this paper, we propose a Joint Wasserstein Auto-Encoders (JWAE) to learn the bidirectional mappings between two domains by minimizing Wasserstein distance of joint distributions. Relying on the optimal transport theory, we are able to exploit sufficient information by matching latent distributions of images in two domains.

The contributions of this paper are summarized as follows:

- We propose a novel JWAE to solve the joint distribution matching problem. Based on Theorem 1, an intractable primal problem of optimal transport can be reduced to a simple optimization problem. Moreover, our method is a generalization of CycleGAN (Liu et al., 2017) and UNIT (Liu et al., 2017).  
- We provide a generalization bound of JWAE (see Theorem 4). In particular, we theoretically prove that the generalization ability of our method w.r.t. the learned joint distribution can be guaranteed by minimizing Wasserstein distance of joint distributions.  
- To practically verify the generalization ability, we apply our method to unsupervised video-to-video synthesis and obtain two visually smooth videos in two different domains. Experiments on real-world datasets show the superiority of the proposed method over several state-of-the-arts.

# 2 RELATED WORK

In this paper, we consider the joint distribution matching problem. Recently, this problem has attracted extensive attention in image-to-image translation and video-to-video synthesis.

Image-to-image translation. Recently, Generative adversarial networks (GAN) (Goodfellow et al., 2014; Cao et al., 2018; Salimans et al., 2018), Variational Auto-Encoders (VAE) (Kingma & Welling, 2014) and Wasserstein Auto-Encoders (WAE) (Tolstikhin et al., 2017) have emerged as popular techniques for the image-to-image translation (I2IT) problem. For the unsupervised I2IT problem, CycleGAN (Zhu et al., 2017), DiscoGAN (Kim et al., 2017) and DualGAN (Yi et al., 2017) aim at minimizing the adversarial loss and the cycle-consistent loss in different domains, which may induce a joint distribution mismatching issue. To address this, CoGAN (Liu & Tuzel, 2016) learns a joint distribution by enforcing a weight-sharing constraint. Moreover, UNIT (Liu et al., 2017) builds upon CoGAN by using a shared-latent space assumption and the same weight-sharing constraint. However, these methods are not well-supported by any theoretical justifications.

Video-to-video synthesis. In this paper, we consider unsupervised video-to-video synthesis (V2VS) problem. Existing image-to-image methods (Zhu et al., 2017; Kim et al., 2017; Yi et al., 2017) cannot be directly used in the video-to-video synthesis problem, we further combine some video frame interpolation methods (Zhou et al., 2016; Ji et al., 2017; Niklaus et al., 2017; Liu et al., 2018) to synthesize video. Although UNIT (Liu et al., 2017) can be applied to video synthesis by interpolating in the latent space, it often results in temporally incoherent videos of low visual quality. Recently, a video-to-video translation method (Bashkirova et al., 2018) is proposed to translate a video in one domain to a video in another domain, but this method can not conduct video frame interpolation. Moreover, Wang et al. (2018) propose a video-to-video synthesis method and synthesize video results, but it cannot work for the unsupervised learning setting.

# 3 NOTATIONS

We use calligraphic letters (e.g.,  $\mathcal{X}$ ) for sets, capital letters (e.g.,  $X$ ) for random variables, and bold lower case letter (e.g.,  $\mathbf{x}$ ) for their corresponding values. We denote probability distributions with capital letters (i.e.,  $P(X)$ ) and corresponding densities with bold lower case letters (i.e.,  $p(\mathbf{x})$ ). Let  $\mathcal{P}(\mathcal{X})$  be the set of all the probability measures over  $\mathcal{X}$ , and  $P_X$  be the marginal distribution over  $\mathcal{X}$ .  $\mathcal{S}_x = \{\mathbf{x}_i\}_{i=1}^N$  and  $\mathcal{S}_y = \{\mathbf{y}_i\}_{i=1}^M$  are two sets of unpaired training data. For convenience, we assume  $N = M$ . We denote by  $X \in \mathcal{X}$  and  $Y \in \mathcal{Y}$  two true images from two different domains,  $X'$  and  $Y'$  generated images sampled from the models, and  $Z \in \mathcal{Z}$  a latent code. For a set  $\mathcal{S}$  and two functions  $F: \mathcal{S} \to \mathbb{R}$  and  $G: \mathcal{S} \to \mathbb{R}$ , we denote  $F(s) \lesssim G(s), \forall s \in \mathcal{S}$  if and only if  $\exists C_1, C_2 > 0$  (independent of  $s$ ) such that  $F(s) \leq C_1 \cdot G(s) + C_2$ .

![](images/2daf033270b17e7a7b1a986b27346a50a4fde47b72e2bd3a1bccca0b21c0c24c.jpg)  
(a) Scheme of JWAE  
(b) Interpolation based video-to-video synthesis  
Figure 1: Demonstrations of (a) the JWAE scheme and (b) the interpolation based V2VS method.

# 4 PROPOSED METHOD

In this section, we propose a novel Joint Wasserstein Auto-Encoders (JWAE) method to solve the joint distribution matching problem. The overall scheme of JWAE is shown in Figure 1. Lastly, we propose a interpolation based V2VS method in Algorithm 1 and Algorithm 2.

# 4.1 JOINT DISTRIBUTION MATCHING PROBLEM

To learn a joint distribution, one can learn a shared latent space for two different domains (Liu et al., 2017). In this sense, any pair of images in different domains can be mapped to the same latent representation. We define latent variable models  $P_{G_1}$  and  $P_{G_2}$  by a two-step procedure: first a code  $Z \in \mathcal{Z}$  is sampled from some prior distribution  $P_Z$  and then  $Z$  can be mapped to  $X$  and  $Y$ , respectively. Then, we have the following densities:

$$
p _ {G _ {1}} (\mathbf {x}) := \int_ {\mathcal {Z}} p _ {G _ {1}} (\mathbf {x} | \mathbf {z}) p _ {\mathbf {z}} (\mathbf {z}) d \mathbf {z}, \forall \mathbf {x} \in \mathcal {X}, \quad p _ {G _ {2}} (\mathbf {y}) := \int_ {\mathcal {Z}} p _ {G _ {2}} (\mathbf {y} | \mathbf {z}) p _ {\mathbf {z}} (\mathbf {z}) d \mathbf {z}, \forall \mathbf {y} \in \mathcal {Y}, \tag {1}
$$

where all involved densities are properly defined. In this paper, we focus on non-random decoders  $G_{1} \colon \mathcal{Z} \to \mathcal{X}$  and  $G_{2} \colon \mathcal{Z} \to \mathcal{Y}$ , i.e., generative models  $P_{G_1}(X'|Z)$  and  $P_{G_2}(Y'|Z)$  deterministically mapping  $Z$  to  $X' = G_{1}(Z)$  and  $Y' = G_{2}(Z)$ , respectively. Based on these two models, we then construct joint distributions in two domains and define the following Wasserstein distance.

Wasserstein distance between joint distributions. Let  $P_A(X, Y')$  and  $P_B(X', Y)$  be joint distributions between real and generated images. Based on the optimal transport theory (Villani, 2008), we minimize Wasserstein distance  $\mathcal{W}(P_A, P_B)$  between joint distributions  $P_A$  and  $P_B$ , i.e.,

$$
\mathcal {W} _ {c} \left(P _ {\mathcal {A}}, P _ {\mathcal {B}}\right) = \min  _ {P \in \mathcal {P} \left(P _ {\mathcal {A}}, P _ {\mathcal {B}}\right)} \mathbb {E} _ {\left(X, Y ^ {\prime}; X ^ {\prime}, Y\right) \sim P} \left[ c \left(X, Y ^ {\prime}; X ^ {\prime}, Y\right) \right], \tag {2}
$$

where  $\mathcal{P}(P_A, P_B)$  is the set of couplings which is composed of joint probability distributions with the probability distributions  $(P_A, P_B)$ , and  $c(X, Y'; X', Y)$  is any measurable cost function between two joint probability distributions  $P_A$  and  $P_B$ .

In practice, there are two important challenges on Wasserstein distance and the cost function. First, directly optimizing Problem (2) raises intractable computational and statistical difficulties (Genevay et al., 2018). Second, how to choose a cost function is very challenging. In this paper, we set  $c(X,Y';X',Y) = c_1(X,X') + c_2(Y',Y)$  (Bhushan Damodaran et al., 2018), where  $c_1$  and  $c_2$  can be any metric to measure the distance in two different feature spaces  $(\mathcal{X} \times \mathcal{X})$  and  $(\mathcal{Y} \times \mathcal{Y})$ , respectively. This cost function helps to derive the following theorem so that the intractable Problem (2) can be reduced to a simple optimization problem.

Theorem 1 Given two deterministic models  $P_{G_1}(X'|Z)$  and  $P_{G_2}(Y'|Z)$  as Dirac measures, i.e.,  $P_{G_1}(X'|Z = \mathbf{z}) = \delta_{G_1(\mathbf{z})}$  and  $P_{G_2}(Y'|Z = \mathbf{z}) = \delta_{G_2(\mathbf{z})}$  for all  $\mathbf{z} \in \mathcal{Z}$ , we have

$$
\mathcal {W} _ {c} \left(P _ {\mathcal {A}}, P _ {\mathcal {B}}\right) = \inf  _ {Q \in \mathcal {Q} _ {1}} \mathbb {E} _ {P _ {X}} \mathbb {E} _ {Q \left(Z _ {1} \mid X\right)} \left[ c _ {1} \left(X, G _ {1} \left(Z _ {1}\right)\right) \right] + \inf  _ {Q \in \mathcal {Q} _ {2}} \mathbb {E} _ {P _ {Y}} \mathbb {E} _ {Q \left(Z _ {2} \mid Y\right)} \left[ c _ {2} \left(G _ {2} \left(Z _ {2}\right), Y\right) \right], \tag {3}
$$

where  $\mathcal{Q}_1 = \{Q(Z_1|X)|Q_{Z_1} = P_Z = Q_{Z_2},P_Y = P_{G_2}\}$  and  $\mathcal{Q}_2 = \{Q(Z_2|Y)|Q_{Z_1} = P_Z = Q_{Z_2},P_X = P_{G_1}\}$  are the set of all probabilistic encoders, where  $Q_{Z_1}$  and  $Q_{Z_2}$  are the marginal distributions of  $Z_{1}\sim Q(Z_{1}|X)$  and  $Z_{2}\sim Q(Z_{2}|Y)$ , where  $X\sim P_X$  and  $Y\sim P_Y$ , respectively.

As previously mentioned, finding an optimal couplings between joint distributions  $P_{\mathcal{A}}$  and  $P_{\mathcal{B}}$  is very challenging. Fortunately, according to Theorem 1, we can instead optimize problem (3) for joint distribution matching. The details of objective functions and optimizations are given below.

# 4.2 JOINT WASSERSTEIN AUTO-ENCODERS

As shown in Figure 1, given real data  $X$  and  $Y$ , we learn the cross-domain mappings (i.e.,  $E_1 \circ G_2$  and  $E_2 \circ G_1$ ) to generate samples  $Y'$  and  $X'$  such that the generated distributions are close to the real distribution, i.e.,  $P_X = P_{G_1}$ ,  $P_Y = P_{G_2}$ . Moreover, the latent distributions generated by two AutoEncoders (i.e.,  $E_1 \circ G_1$  and  $E_2 \circ G_2$ ) should be close to each other, i.e.,  $Q_{Z_1} = Q_{Z_2}$ . To optimize Problem (3), we relax these constraints  $P_X = P_{G_1}$ ,  $P_Y = P_{G_2}$  and  $Q_{Z_1} = P_Z = Q_{Z_2}$  by introducing penalties into (3). Then we minimize the regularized optimization problem:

$$
\begin{array}{l} \widehat {\mathcal {W}} _ {c} \left(P _ {\mathcal {A}}, P _ {\mathcal {B}}\right) = \inf  _ {Q \in \mathcal {Q} _ {1}} \mathbb {E} _ {P _ {X}} \mathbb {E} _ {Q \left(Z _ {1} \mid X\right)} \left[ c _ {1} \left(X, G _ {1} \left(Z _ {1}\right)\right) \right] + \inf  _ {Q \in \mathcal {Q} _ {2}} \mathbb {E} _ {P _ {Y}} \mathbb {E} _ {Q \left(Z _ {2} \mid Y\right)} \left[ c _ {2} \left(Y, G _ {2} \left(Z _ {2}\right)\right) \right] \tag {4} \\ + \alpha \mathcal {D} _ {X} \left(P _ {X}, P _ {G _ {1}}\right) + \beta \mathcal {D} _ {Y} \left(P _ {Y}, P _ {G _ {2}}\right) + \rho \mathcal {D} _ {Z} \left(Q _ {Z _ {1}}, Q _ {Z _ {2}}\right), \\ \end{array}
$$

where  $\alpha, \beta, \rho$  are positive hyper-parameters,  $\mathcal{D}_X(P_X, P_{G_1}), \mathcal{D}_Y(P_Y, P_{G_2})$  and  $\mathcal{D}_Z(Q_{Z_1}, Q_{Z_2})$  can be arbitrary distribution divergence between two distributions. The above problems involve two kinds of functions, namely the reconstruction loss and distribution divergence.

(i) Reconstruction loss. In practice, we minimize the empirical reconstruction losses of  $\mathbb{E}_{P_X}\mathbb{E}_{Q(Z_1|X)}[c_1(X,G_1(Z_1))]$  and  $\mathbb{E}_{P_Y}\mathbb{E}_{Q(Z_2|Y)}[c_2(Y,G_2(Z_2))]$ , denoted by  $\mathcal{R}_x(E_1,E_2,G_1,G_2)$  and  $\mathcal{R}_y(E_1,E_2,G_1,G_2)$ , respectively. Taking the case for domain  $\mathcal{Q}_1$  as an example, the empirical reconstruction loss  $\mathcal{R}_x(E_1,E_2,G_1,G_2)$  can be rewritten as follows:

$$
\mathcal {R} _ {x} \left(E _ {1}, E _ {2}, G _ {1}, G _ {2}\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \underbrace {c _ {1} \left(\mathbf {x} _ {i} , G _ {1} \left(E _ {1} \left(\mathbf {x} _ {i}\right)\right)\right)} _ {\text {A u t o - E n c o d e r l o s s}} + \underbrace {c _ {1} \left(\mathbf {x} _ {i} , G _ {1} \left(E _ {2} \left(G _ {2} \left(E _ {1} \left(\mathbf {x} _ {i}\right)\right)\right)\right)\right)} _ {\text {C y c l e c o n s i s t e n c y l o s s}}, \tag {5}
$$

Here, the first term represents the loss on the Auto-Encoders reconstruction and the second term inherently enforce the cycle consistency that widely studied in image-to-image translation tasks (Zhu et al., 2017; Liu et al., 2017; Kim et al., 2017; Yi et al., 2017). However, unlike existing methods, in our paper, the cycle consistency loss is directly derived from the joint distribution matching problem. The loss  $\mathcal{R}_y(E_1,E_2,G_1,G_2)$  can be constructed similarly. Last, let  $\mathcal{R}(E_1,E_2,G_1,G_2) = \mathcal{R}_x(E_1,E_2,G_1,G_2) + \mathcal{R}_y(E_1,E_2,G_1,G_2)$  be the final reconstruction loss.

(ii) Distribution divergence. The distribution divergences  $\mathcal{D}_X(P_X, P_{G_1}), \mathcal{D}_Y(P_Y, P_{G_2})$  and  $\mathcal{D}_Z(Q_{Z_1}, Q_{Z_2})$  in (4) can be measured by GAN divergence (Goodfellow et al., 2014; LI et al., 2017). Maximum Mean Discrepancy (MMD) and Kullback-Leibler (KL) divergence, etc. Here, we use triple GAN (LI et al., 2017) to measure the GAN divergence, denoted by  $\mathrm{GAN}(P_X, P_{G_1})$ ,  $\mathrm{GAN}(P_Y, P_{G_2})$  and  $\mathrm{GAN}(Q_{Z_1}, Q_{Z_2})$ , respectively. Taking  $\mathrm{GAN}(P_X, P_{G_1})$  as example, the loss function  $\mathcal{L}_x(E_1, E_2, G_1, G_2, D_x)$  can be formulated as

$$
\begin{array}{l} \mathcal {L} _ {x} \left(E _ {1}, E _ {2}, G _ {1}, G _ {2}, D _ {x}\right) = \frac {1}{M} \sum_ {i = 1} ^ {M} 2 \log \left(D _ {x} \left(\mathbf {x} _ {i}\right)\right) + \log \left(1 - D _ {x} \left(G _ {1} \left(E _ {2} \left(G _ {2} \left(E _ {1} (\mathbf {x} _ {i})\right)\right)\right)\right) \right. \tag {6} \\ + \frac {1}{N} \sum_ {i = 1} ^ {N} \log (1 - D _ {x} (G _ {1} (E _ {2} (\mathbf {y} _ {i}))). \\ \end{array}
$$

The loss functions  $\mathcal{L}_y(E_1,E_2,G_1,G_2,D_x)$  and  $\mathcal{L}_z(E_1,E_2,G_1,G_2,D_z)$  w.r.t.  $\mathrm{GAN}(P_Y,P_{G_2})$  and  $\mathrm{GAN}(Q_{Z_1},Q_{Z_2})$  can similarly constructed. Please find the details in supplementary materials.

# 4.3 A CONCRETE EXAMPLE: INTERPOLATION BASED VIDEO-TO-VIDEO SYNTHESIS

Since the generalization ability can be evaluated by the performance of interpolation (Bojanowski et al., 2018), we apply our method on the interpolation based video-to-video synthesis (V2VS) problem. The training and inference methods are shown in Algorithm 1 and Algorithm 2, respectively.

In the training, given the images  $\{\mathbf{x}_i\}_{i=1}^M$  and  $\{\mathbf{y}_j\}_{j=1}^N$  in two different domains, we seek to learn a joint distribution mapping between these two distributions (see Algorithm 1). In the inference, given two input frames  $\mathbf{x}_{\mathrm{begin}}$  and  $\mathbf{x}_{\mathrm{end}}$ , we apply our method to perform interpolation based video-to-video synthesis to produce two videos in two different domains. Specifically, we perform linear

<table><tr><td>Algorithm 1 Training details for JWAE.</td><td>Algorithm 2 Inference for unsupervised V2VS.</td></tr><tr><td>Input: Training data in two different domains: 
{xi}M i=1 and {yj}N j=1.</td><td>Input: Testing data pair in the first domain: 
{xbegin, xend}.</td></tr><tr><td>Initialization: Models: E1, E2, G1, G2; 
Discriminators: Dw, w∈{x, y, z}.</td><td>Step 1: Video frame interpolation 
zbegin = E1(xbegin), zend = E1(xend) 
ˆz = αzbegin+(1-α)zend, α ∈ (0,1) 
ˆx = G1(ˆz)</td></tr><tr><td>repeat</td><td></td></tr><tr><td>Update Dx, Dy, Dz by ascending:</td><td>Synthesized video: {xbegin,ˆx, xend}</td></tr><tr><td>∑wLw(E1, E2, G1, G2, Dw), w∈{x, y, z}</td><td>Step 2: Video translation 
ybegin = G2(zbegin),ˆy = G2(ˆz), 
yend = G2(zend)</td></tr><tr><td>Update E1, E2, G1, G2 by descending:</td><td></td></tr><tr><td>∑wLw(E1, E2, G1, G2, Dw)+R(E1, E2, G1, G2)</td><td>Synthesized video: {ybegin,ˆy, yend}</td></tr><tr><td>until models converged</td><td></td></tr></table>

interpolation based on the latent space  $\mathcal{Z}$  extracted from the first domain and then decode it to produce a corresponding video in the second domain (see Figure 1 (b) and Algorithm 2). In this sense, we can directly measure the quality of the synthesized video in the second domain to evaluate the generalization ability of the learned joint distribution mapping.

# 5 GENERALIZATION ANALYSIS

In this section, we analyze generalization performance of JWAE. To begin with, we provide the definitions of the generalization error and probabilistic cross-domain Lipschitzness.

Definition 1 (Generalization Error) Define cross-domain functions as  $f = G_{2} \circ E_{1}$  and  $f = G_{1} \circ E_{2}$  when  $Q_{Z_1} = Q_{Z_2}$ , and cost functions  $c_{1}$  and  $c_{2}$  which are bounded, symmetric,  $L_{c}$ -Lipschitz and satisfies the triangle inequality, and given two joint distributions  $P_{\mathcal{A}}(X,Y)$  and  $P_{\mathcal{B}}(X,Y)$ , where  $Y$  in  $P_{\mathcal{A}}(X,Y)$  and  $X$  in  $P_{\mathcal{B}}(X,Y)$  are unknown, then the generalization error  $E(f,g)$  becomes:

$$
E (f, g) = \mathbb {E} _ {(X, Y) \sim P _ {\mathcal {A}} (X, Y)} [ c _ {2} (Y, f (X)) ] + \mathbb {E} _ {(X, Y) \sim P _ {\mathcal {B}} (X, Y)} [ c _ {1} (X, g (Y)) ]. \tag {7}
$$

Note that in I2IT problem, it is common to assume that two close samples will have close outputs with high probability, i.e., it satisfies a probabilistic Lipschitzness assumption (Courty et al., 2017; Urner et al., 2011). For convenience, we extend the definition as follows.

Definition 2 ( $\phi$ -Probabilistic Cross-domain Lipschitzness) Given real and the generated marginal distribution  $P_X$  and  $P_{X'}$ , and let  $\phi: \mathbb{R}^+ \to [0,1]$ , we say that a function  $f: \mathcal{X} \to \mathcal{Y}$  w.r.t. a joint distribution set  $\mathcal{P}(P_X, P_{X'})$  over  $P_X$  and  $P_{X'}$  is  $\phi$ -Lipschitz if for all  $\alpha > 0$ ,

$$
P _ {\left(X, X ^ {\prime}\right) \sim \mathcal {P} \left(P _ {X}, P _ {X ^ {\prime}}\right)} \left[ \| f (X) - f \left(X ^ {\prime}\right) \| <   \alpha c _ {1} \left(X, X ^ {\prime}\right) \right] \geq 1 - \phi (\alpha). \tag {8}
$$

Intuitively, given a joint distribution set  $\mathcal{P}(P_X, P_{X'})$ , a function  $f$  satisfying the  $\alpha$ -Lipschitz property holds with some probability. Then, we have the following results on the generalization error.

Theorem 2 Let  $P^{*} = \arg \min_{P\in \mathcal{P}(P_A^f,P_B^g)}\mathbb{E}_{(X,Y^f;X^g,Y)\sim P}[c(X,Y^f;X^g,Y)]$  with Lipschitz cost functions  $c_{1}$  and  $c_{2}$ . Let functions  $f^{*}\in \mathcal{F}$  and  $g^{*}\in \mathcal{G}$  be probabilistic cross-domain Lipschitzness w.r.t.  $P^{*}$  that minimizes the joint error  $E(f^{*},g^{*})$ . Given  $M$  and  $N$  instances drawn form  $P_{X}$  and  $P_{Y}$ , respectively, with  $L_{c_1}\alpha = L_{c_2}\beta = 1$ , the following relation holds with probability at least  $1 - \delta$ :

$$
E (f, g) \lesssim \mathcal {W} (\widehat {P} _ {\mathcal {A}} ^ {f}, \widehat {P} _ {\mathcal {B}} ^ {g}) + \sqrt {\log \left(\frac {1}{\delta}\right)} \left(\frac {1}{\sqrt {M}} + \frac {1}{\sqrt {N}}\right) + E \left(f ^ {*}, g ^ {*}\right) + \widetilde {\phi} (\alpha , \beta), \tag {9}
$$

where  $E(f^{*},g^{*}) = E_{\mathcal{A}}(f^{*}) + E_{\mathcal{B}}^{g}(f^{*}) + E_{\mathcal{B}}(g^{*}) + E_{\mathcal{A}}^{f}(g^{*})$  and  $\widetilde{\phi} (\alpha ,\beta) = L_{c_1}M_1\phi (\alpha) + L_{c_2}M_2\phi (\beta)$  where  $\| f^{*}(X_{1}) - f^{*}(X_{2})\| \leq M_{1},\forall X_{1},X_{2}\in \mathcal{X}$  and  $\| g^{*}(Y_{1}) - g^{*}(Y_{2})\| \leq M_{2},\forall Y_{1},Y_{2}\in \mathcal{V}$ .

Remark 1 Theorem 4 provides an upper bound on the generalization error. The first term in right hand of (9) corresponds to the empirical version of (2); While the second term means that we should minimize it with sufficient unpaired data from two different domains. The third term  $E(f^{*},g^{*})$  correspond to the joint error. When the last term  $\widetilde{\phi} (\alpha ,\beta)$  is sufficiently small (i.e.,  $f$  and  $g$  satisfy Lipschitz property with high probability), the cross-domain mappings would be well learned.

# 6 EXPERIMENTS

We apply our method to interpolation based video-to-video synthesis (V2VS) in the unsupervised setting. To be specific, we firstly conduct video interpolation between two input frames in one domain and then translate it to produce a corresponding video in another domain.

Datasets. We conduct experiments on two widely used benchmark datasets, namely Cityscapes (Cordts et al., 2016) and SYNTHIA (Ros et al., 2016). (i) Cityscapes contains  $2048 \times 1024$  street scene video of several German cities and a portion of ground truth semantic segmentation in the video. To obtain more semantic segmentation masks, following (Wang et al., 2018), we use a pre-trained DeepLab V3 network (Chen et al., 2017) to extract extra segmentation videos. (ii) We also study the generality of our algorithm on the unpaired dataset SYNTHIA (Ros et al., 2016), which contains a large collection of synthetic videos in different scenes and seasons. We perform unsupervised V2VS on four splits of SYNTHIA with different seasons, i.e., spring, summer, fall and winter. In this paper, we adopt the winter split as the common domain and train models to translate videos from winter to the other three seasons.

Evaluation metrics. For quantitative comparisons, we adopt Fréchet Inception Distance (FID) (Heusel et al., 2017) to evaluate the quality of the frames in the synthesized videos. FID captures the similarity of the generated samples to real ones and correlates well with human judgement. Moreover, we also use a variant of FID (Wang et al., 2018) (FID4Video) to evaluate the quality of video. FID4Video measures the distribution similarity based on the extracted feature of videos. In general, for both FID and FID4Video, a lower score means the better performance.

# 6.1 IMPLEMENTATION DETAILS

We implement our method based on PyTorch $^{34}$ . We follow the experimental settings in CycleGAN (Zhu et al., 2017). For the optimization, we use Adam solver (Kingma & Ba, 2015) with a mini-batch size of 1 to train the models, and use a learning rate of 0.0002 for the first 100 epochs and gradually decrease it to zero for the next 100 epochs. Following (Zhu et al., 2017), we set  $\alpha = \beta = 0.1$  in Eqn. (4). By default, we set  $\rho = 0.1$  in our experiments.

# 6.2 BASELINE METHODS

We adopt several state-of-the-art baselines, including UNIT (Liu et al., 2017) with the latent space interpolation and several constructed variants of CycleGAN (Zhu et al., 2017) using different view synthesis algorithms. For those constructed baselines, we first conduct video interpolation and then perform image-to-image translation. Moreover, we also construct a variant of our method to conduct ablation study on the Triple GAN loss. All considered baselines are summarized as follows.

- UNIT (Liu et al., 2017). UNIT is a state-of-the-art unsupervised I2IT method which can perform supervised video-to-video synthesis by interpolating in the latent space.  
- DVF-Cycle. This method combines the view synthesis method DVF (Liu et al., 2018) with CycleGAN (Zhu et al., 2017). To be specific, DVF produces videos by video interpolation in one domain. Then, we use CycleGAN to translate the generated video to another domain.  
- DVM-Cycle. We use a geometrical view synthesis DVM (Ji et al., 2017) for video synthesis, and we replace DVF in DVF-Cycle with DVM and construct a new baseline called DVM-Cycle.  
- AdaConv-Cycle. We also compare against a state-of-the-art video interpolation method AdaConv (Niklaus et al., 2017). For cross-domain video synthesis, we combine this method with a pre-trained CycleGAN model and term it AdaConv-CycleGAN in the following experiments.  
- W/O-Triple. To investigate the effect of the Triple GAN loss, we construct a baseline method by removing it from our method. We refer to the method without Triple GAN loss as W/O-Triple.

Table 1: Performance comparisons with state-of-the-art baselines on Cityscapes and SYNTHIA.  

<table><tr><td rowspan="3">Method</td><td colspan="4">Cityscapes</td><td colspan="6">SYNTHIA</td></tr><tr><td colspan="2">photo2segmentation</td><td colspan="2">segmentation2photo</td><td colspan="2">winter2spring</td><td colspan="2">winter2summer</td><td colspan="2">winter2fall</td></tr><tr><td>FID</td><td>FID4Video</td><td>FID</td><td>FID4Video</td><td>FID</td><td>FID4Video</td><td>FID</td><td>FID4Video</td><td>FID</td><td>FID4Video</td></tr><tr><td>DVF-Cycle</td><td>110.59</td><td>23.95</td><td>151.27</td><td>40.61</td><td>152.44</td><td>42.22</td><td>160.69</td><td>42.43</td><td>163.13</td><td>41.04</td></tr><tr><td>DVM-Cycle</td><td>50.51</td><td>17.33</td><td>116.62</td><td>40.83</td><td>129.80</td><td>38.19</td><td>140.86</td><td>36.66</td><td>129.02</td><td>36.64</td></tr><tr><td>AdaConv-Cycle</td><td>33.50</td><td>14.96</td><td>99.67</td><td>30.24</td><td>117.40</td><td>23.83</td><td>126.01</td><td>20.62</td><td>110.52</td><td>16.77</td></tr><tr><td>UNIT</td><td>31.27</td><td>10.12</td><td>76.72</td><td>29.21</td><td>96.40</td><td>23.12</td><td>108.01</td><td>24.70</td><td>97.73</td><td>20.39</td></tr><tr><td>W/O-Triple</td><td>24.32</td><td>8.34</td><td>47.41</td><td>27.37</td><td>92.77</td><td>21.83</td><td>84.83</td><td>19.54</td><td>91.37</td><td>15.87</td></tr><tr><td>Ours</td><td>22.74</td><td>6.80</td><td>43.48</td><td>25.87</td><td>88.24</td><td>21.37</td><td>77.12</td><td>17.99</td><td>87.50</td><td>14.14</td></tr></table>

![](images/d602566f5b9764fac56e606dbd2cd3e63ce0685960120eae5a6a8fe9b808fe22.jpg)  
Figure 2: Comparisons of different methods for photo  $\leftrightarrow$  segmentation translation on Cityscapes dataset. We first synthesize a video of street scene and then translate it to the segmentation domain (Top), and vice versa for the mapping from segmentation to street scene (Bottom).

# 6.3 QUANTITATIVE COMPARISONS

We compare the performance on Cityscapes and SYNTHIA and show the results in Table 1. We can draw the following observations. First, our method consistently outperforms the baselines in terms of both FID and FID4Video scores. It means that our method produces frames and videos of promising quality and exhibits strong generalization ability. Second, with the help of Triple GAN loss, our method achieves better results than W/O-Triple for both FID and FID4Video. This indicates that the reconstructed images after the cycle translation helps to learn a better joint distribution. The above observations demonstrate the superiority of our method over the competitive methods.

# 6.4 VISUAL COMPARISONS

Visual results on Cityscapes. We first interpolate videos in the cityscape domain and then translate them to the segmentation domain. We compare the visual quality of both the interpolated and the translated images in Figure 2. From Figure 2 (top), our method produces sharper cityscape images and yields more accurate results in the semantic segmentation domain, which significantly outperforms the baseline methods, and vice versa in Figure 2 (bottom).

Visual results on SYNTHIA. We further evaluate the performance of our method on SYNTHIA. We synthesize videos among the domains of four seasons shown in Figure 3. First, our method is able to produce sharper images when interpolating the missing in-between frames (see top row of Figure 3). Second, the translated frames produced by our method in the other three seasons look

![](images/c133614978be46dfd2e7fa882076f0c858e2eca1f356c49c47973699552f2b54.jpg)  
Figure 3: Comparison of different methods for season translation on SYNTHIA dataset. Top row: The synthesized video in the winter domain. Rows 2-4: The corresponding translated video in the domains of the other three seasons, i.e., spring, summer and fall.

Table 2: Influence of  $\rho$  for the adversarial loss on  $\mathcal{Z}$  with different values. We compare the results of winter  $\leftrightarrow$  summer on SYNTHIA dataset in terms of FID and FID4Video scores.  

<table><tr><td rowspan="2">ρ</td><td colspan="2">winter2summer</td><td colspan="2">summer2winter</td></tr><tr><td>FID</td><td>FID4Video</td><td>FID</td><td>FID4Video</td></tr><tr><td>0.01</td><td>94.91</td><td>20.29</td><td>107.65</td><td>18.90</td></tr><tr><td>0.1</td><td>77.12</td><td>17.99</td><td>89.03</td><td>17.36</td></tr><tr><td>1</td><td>89.07</td><td>21.04</td><td>102.18</td><td>18.63</td></tr><tr><td>10</td><td>101.07</td><td>23.66</td><td>108.47</td><td>20.50</td></tr></table>

more photo-realistic than those produced by the other baseline methods (see the shape of cars in rows 2-4 of Figure 3). These results demonstrate that our method is able to produce more promising videos and consistently outperforms other methods in different domains.

# 6.5 INFLUENCE OF  $\rho$  FOR THE ADVERSARIAL LOSS ON  $\mathcal{Z}$

We study the effect of the trade-off parameter  $\rho$  over the adversarial loss on  $\mathcal{Z}$  in Eqn. (4). The results are shown in Table 2. Given a very small weight  $\rho = 0.01$ , the model obtains larger FID and FID4Video scores compared to that with  $\rho = 0.1$ . When we increase it to  $\rho = 1$  and  $\rho = 10$ , we also observe large performance degrades. Therefore, we suggest setting  $\rho = 0.1$  in our method.

# 7 CONCLUSION

In this paper, we have proposed a novel joint Wasserstein Auto-Encoders method for the joint distribution matching problem. Instead of directly optimizing the primal problem of Wasserstein distance, we turn to propose a simple but effective optimization problem. In this way, we are able to conduct analysis on the generalization ability of JWAE and theoretically prove that minimizing the Wasserstein distance can guarantee the generalization ability. Extensive experiments on unsupervised V2VS task over several benchmark datasets demonstrate the superiority of the proposed method over the state-of-the-art methods.

# REFERENCES

Dina Bashkirova, Ben Usman, and Kate Saenko. Unsupervised video-to-video translation. arXiv preprint arXiv:1806.03698, 2018.  
Bharath Bhushan Damodaran, Benjamin Kellenberger, Remi Flamary, Devis Tuia, and Nicolas Courty. Deepjdot: Deep joint distribution optimal transport for unsupervised domain adaptation. In European Conference on Computer Vision, 2018.  
Piotr Bojanowski, Armand Joulin, David Lopez-Paz, and Arthur Szlam. Optimizing the latent space of generative networks. In International Conference on Machine Learning, 2018.  
François Bolley, Arnaud Guillin, and Cédric Villani. Quantitative concentration inequalities for empirical measures on non-compact spaces. *Probability Theory and Related Fields*, 2007.  
Jiezhang Cao, Yong Guo, Qingyao Wu, Chunhua Shen, Junzhou Huang, and Mingkui Tan. Adversarial learning with local coordinate coding. In International Conference on Machine Learning, 2018.  
Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Computer Vision and Pattern Recognition, 2016.  
Nicolas Courty, Rémi Flamary, Amaury Habrard, and Alain Rakotomamonjy. Joint distribution optimal transportation for domain adaptation. In Advances in Neural Information Processing Systems, 2017.  
Tomer Galanti, Sagie Benaim, and Lior Wolf. Generalization bounds for unsupervised cross-domain mapping with wgans. arXiv preprint arXiv:1807.08501, 2018.  
Aude Geneva, Gabriel Peyre, and Marco Cuturi. Learning generative models with sinkhorn divergences. In Artificial Intelligence and Statistics, 2018.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, 2017.  
Dinghuang Ji, Junghyun Kwon, Max McFarland, and Silvio Savarese. Deep view morphing. In Computer Vision and Pattern Recognition, 2017.  
Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, and Jiwon Kim. Learning to discover cross-domain relations with generative adversarial networks. In International Conference on Machine Learning, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference for Learning Representations, 2015.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In International Conference on Learning Representations, 2014.  
Chongxuan LI, Taufik Xu, Jun Zhu, and Bo Zhang. Triple generative adversarial nets. In Advances in Neural Information Processing Systems, 2017.  
Torgny Lindvall. Lectures on the Coupling Method. Courier Corporation, 2002.  
Ming-Yu Liu and Oncel Tuzel. Coupled generative adversarial networks. In Advances in Neural Information Processing Systems, 2016.

Ming-Yu Liu, Thomas Breuel, and Jan Kautz. Unsupervised image-to-image translation networks. In Advances in Neural Information Processing Systems, 2017.  
Ziwei Liu, Raymond A Yeh, Xiaou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In International Conference on Computer Vision, 2018.  
Simon Niklaus, Long Mai, and Feng Liu. Video frame interpolation via adaptive separable convolution. In International Conference on Computer Vision, 2017.  
Xudong Pan, Mi Zhang, and Daizong Ding. Theoretical analysis of image-to-image translation with adversarial learning. In International Conference on Machine Learning, 2018.  
Svetlozar Todorov Rachev et al. Duality theorems for kantorovich-rubinstein and wasserstein functionals. Instytut Matematyczny Polskiej Akademi Nauk (Warszawa), 1990.  
German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M. Lopez. The synthia dataset: A large collection of synthetic images for semantic segmentation of urban scenes. In Computer Vision and Pattern Recognition, 2016.  
Tim Salimans, Han Zhang, Alec Radford, and Dimitris Metaxas. Improving GANs using optimal transport. In International Conference on Learning Representations, 2018.  
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schoelkopf. Wasserstein autoencoders. In International Conference on Learning Representations, 2017.  
Ruth Urner, Shai Shalev-Shwartz, and Shai Ben-David. Access to unlabeled data can speed up prediction time. In International Conference on Machine Learning, 2011.  
Cédric Villani. Optimal Transport: Old and New. Springer Science & Business Media, 2008.  
Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu, Guilin Liu, Andrew Tao, Jan Kautz, and Bryan Catanzaro. Video-to-video synthesis. In Advances in Neural Information Processing Systems, 2018.  
Zili Yi, Hao (Richard) Zhang, Ping Tan, and Minglun Gong. Dualgan: Unsupervised dual learning for image-to-image translation. In International Conference on Computer Vision, 2017.  
Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A Efros. View synthesis by appearance flow. In European conference on computer vision, 2016.  
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In International Conference on Computer Vision, 2017.
