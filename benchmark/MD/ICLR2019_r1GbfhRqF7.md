# KERNEL CHANGE-POINT DETECTION WITH AUXILIARY DEEP GENERATIVE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Detecting the emergence of abrupt property changes in time series is a challenging problem. Kernel two-sample test has been studied for this task which makes fewer assumptions on the distributions than traditional parametric approaches. However, selecting kernels is non-trivial in practice. Although kernel selection for the two-sample test has been studied, the insufficient samples in change point detection problem hinder the success of those developed kernel selection algorithms. In this paper, we propose KL-CPD, a novel kernel learning framework for time series CPD that optimizes a lower bound of test power via an auxiliary generative model. With deep kernel parameterization, KL-CPD endows kernel two-sample test with the data-driven kernel to detect different types of change-points in real-world applications. The proposed approach significantly outperformed other state-of-the-art methods in our comparative evaluation of benchmark datasets and simulation studies.

# 1 INTRODUCTION

Detecting changes in the temporal evolution of a system (biological, physical, mechanical, etc.) in time series analysis has attracted considerable attention in machine learning and data mining for decades (Basseville et al., 1993; Brodsky & Darkhovsky, 2013). This task, commonly referred to as change-point detection (CPD) or anomaly detection in the literature, aims to predict significant changing points in a temporal sequence of observations. CPD has a broad range of real-world applications such as medical diagnostics (Gardner et al., 2006), industrial quality control (Basu & Meckesheimer, 2007), financial market analysis (Pepelyshev & Polunchenko, 2015) and more.

As shown in Fig. 1, we focus on the retrospective CPD (Takeuchi & Yamanishi, 2006; Li et al., 2015a), which allows a flexible time window to react on the change-points. Retrospective CPD not only enjoys more robust detection (Chandola et al., 2009) but embraces many applications such as climate change detection (Reeves et al., 2007), genetic sequence analysis (Wang et al., 2011), networks intrusion detection (Yamanishi et al., 2004), to name just a few. Various methods have been developed (Gustafsson & Gustafsson, 2000), and many of them are parametric with strong assumptions on the distributions (Basseville et al., 1993; Gustafsson, 1996), including auto-regressive models (Yamanishi & Takeuchi, 2002) and state-space models (Kawahara et al., 2007) for tracking changes in the mean, the variance, and the spectrum.

![](images/4da6340e16d0e165eb4e5717d02a2e1a5c380781a440d6cc73a3639328441d83.jpg)  
Figure 1: A sliding window over the time series input with two intervals: the past and the current, where  $w_{l}, w_{r}$  are the size of the past and current interval, respectively.  $X^{(l)}, X^{(r)}$  consists of the data in the past and current interval, respectively.

Ideally, the detection algorithm should be free of distributional assumptions to have robust performance as neither true data distributions nor anomaly types are known a priori. Thus the parametric assumptions in many works are unavoidably a limiting factor in practice. As an alternative, nonparametric and kernel approaches are free of distributional assumptions and hence enjoy the advantage to produce more robust performance over a broader class of data distributions.

Kernel two-sample test has been applied to time series CPD with some success. For example, Harchaoui et al. (2009) presented a test statistic based upon the maximum kernel fisher discriminant ratio for hypothesis testing and Li et al. (2015a) proposed a computational efficient test statistic based on maximum mean discrepancy with block sampling techniques. The performance of kernel methods, nevertheless, relies heavily on the choice of kernels. Gretton et al. (2007; 2012a) conducted kernel selection for RBF kernel bandwidths via median heuristic. While this is certainly straightforward, it has no guarantees of optimality regarding the statistical test power of hypothesis testing. Gretton et al. (2012b) show explicitly optimizing the test power leads to better kernel choice for hypothesis testing under mild conditions. Kernel selection by optimizing the test power, however, is not directly applicable for time series CPD, as we point out in Section 3.

In this paper, we propose KL-CPD, a kernel learning framework for time series CPD. Our main contributions are three folds.

- In Section 3, we first observe the inaptness of existing kernel learning approaches in a simulated example. We then propose to optimize a lower bound of the test power via an auxiliary generative model, which aims at serving as a surrogate of the abnormal events.  
- In Section 4, we present a deep kernel parametrization of our framework, which endows a data-driven kernel for the kernel two-sample test. KL-CPD induces composition kernels by combining RNNs and RBF kernels that are suitable for the time series applications.  
- In Section 5, we conduct extensive benchmark evaluation showing the outstanding performance of KL-CPD in real-world CPD applications. With simulation-based analysis in Section 6, in addition, we can see the proposed method not only boosts the kernel power but also evades the performance degradation as data dimensionality of time series increases.

# 2 PRELIMINARY

Given a sequence of  $d$ -dimensional observations  $\{x_{1},\ldots ,x_{t},\ldots \} ,x_{i}\in \mathbb{R}^{d}$ , our goal is to detect the existence of a change-point, such that before the change-point, samples are i.i.d from a distribution  $\mathbb{P}$ , while after the change-point, samples are i.i.d from a different distribution  $\mathbb{Q}$ . Suppose at current time  $t$  and the window size  $w$ , denote the past window segment  $X^{(l)} = \{x_{t - w},\dots,x_{t - 1}\}$  and the current window segment  $X^{(r)} = \{x_{t},\dots,x_{t + w - 1}\}$ . We compute the maximum mean discrepancy (MMD) between  $X^{(l)}$  and  $X^{(r)}$ , and use it as the plausibility of change-points: The higher the distribution discrepancy, the more likely the point is a change-point.

# 2.1 MMD AND TEST POWER

We review maximum mean discrepancy (MMD) and its use to two-sample test, which are two cornerstones in this work. Let  $k$  be the kernel of a reproducing kernel Hilbert space (RKHS)  $\mathcal{H}_k$  of functions on a set  $\mathcal{X}$ . We assume that  $k$  is measurable and bounded,  $\sup_{x\in \mathcal{X}}k(x,x) < \infty$ . MMD is a nonparametric probabilistic distance commonly used in two-sample-test (Gretton et al., 2007; 2012a). Given a kernel  $k$ , the MMD distance between two distributions  $\mathbb{P}$  and  $\mathbb{Q}$  is defined as

$$
M _ {k} (\mathbb {P}, \mathbb {Q}) := \| \mu_ {\mathbb {P}} - \mu_ {\mathbb {Q}} \| _ {\mathcal {H} _ {k}} ^ {2} = \mathbb {E} _ {\mathbb {P}} [ k (x, x ^ {\prime}) ] - 2 \mathbb {E} _ {\mathbb {P}, \mathbb {Q}} [ k (x, y) ] + \mathbb {E} _ {\mathbb {Q}} [ k (y, y ^ {\prime}) ],
$$

where  $\mu_{\mathbb{P}} = \mathbb{E}_{x\sim \mathbb{P}}[k(x,\cdot))],\mu_{\mathbb{Q}} = \mathbb{E}_{y\sim \mathbb{Q}}[k(y,\cdot))]$  are the kernel mean embedding for  $\mathbb{P}$  and  $\mathbb{Q}$ , respectively. In practice we use finite samples from distributions to estimate MMD distance. Given  $X = \{x_{1},\ldots ,x_{m}\} \sim \mathbb{P}$  and  $Y = \{y_{1},\dots ,y_{m}\} \sim \mathbb{Q}$ , one unbiased estimator of  $M_{k}(\mathbb{P},\mathbb{Q})$  is

$$
\hat {M} _ {k} (X, Y) := \frac {1}{\binom {m} {2}} \sum_ {i \neq i ^ {\prime}} k (x _ {i}, x _ {i ^ {\prime}}) - \frac {2}{m ^ {2}} \sum_ {i, j} k (x _ {i}, y _ {j}) + \frac {1}{\binom {m} {2}} \sum_ {j \neq j ^ {\prime}} k (y _ {j}, y _ {j ^ {\prime}}).
$$

which has nearly minimal variance among unbiased estimators (Gretton et al., 2012a, Lemma 6).

For any characteristic kernel  $k$ ,  $M_{k}(\mathbb{P}, \mathbb{Q})$  is non-negative and in particular  $M_{k}(\mathbb{P}, \mathbb{Q}) = 0$  iff  $\mathbb{P} = \mathbb{Q}$ . However, the estimator  $\hat{M}_{k}(X, X^{\prime})$  may not be 0 even though  $X, X^{\prime} \sim \mathbb{P}$  due to finite sample size. Hypothesis test instead offers thorough statistical guarantees of whether two finite sample sets are the same distribution. Following Gretton et al. (2012a), the hypothesis test is defined by the null hypothesis  $H_{0}: \mathbb{P} = \mathbb{Q}$  and alternative  $H_{1}: \mathbb{P} \neq \mathbb{Q}$ , using test statistic  $m\hat{M}_{k}(X, Y)$ . For a given

allowable false rejection probability  $\alpha$  (i.e., false positive rate or Type I error), we choose a test threshold  $c_{\alpha}$  and reject  $H_0$  if  $m\hat{M}_k(X,Y) > c_\alpha$ .

We now describe the objective to choose the kernel  $k$  for maximizing the test power (Gretton et al., 2012b; Sutherland et al., 2017). First, note that, under the alternative  $H_{1}:\mathbb{P}\neq \mathbb{Q},\hat{M}_{k}$  is asymptotically normal,

$$
\frac {\hat {M} _ {k} (X , Y) - M _ {k} (\mathbb {P} , \mathbb {Q})}{\sqrt {V _ {m} (\mathbb {P} , \mathbb {Q})}} \xrightarrow {\mathcal {D}} \mathcal {N} (0, 1), \tag {1}
$$

where  $V_{m}(\mathbb{P},\mathbb{Q})$  denotes the asymptotic variance of the  $\hat{M}_k$  estimator. The test power is then

$$
\Pr \left(m \hat {M} _ {k} (X, Y) > c _ {\alpha}\right) \longrightarrow \Phi \left(\frac {M _ {k} (\mathbb {P} , \mathbb {Q})}{\sqrt {V _ {m} (\mathbb {P} , \mathbb {Q})}} - \frac {c _ {\alpha}}{m \sqrt {V _ {m} (\mathbb {P} , \mathbb {Q})}}\right) \tag {2}
$$

where  $\Phi$  is the CDF of the standard normal distribution. Given a set of kernels  $\mathcal{K}$ , We aim to choose a kernel  $k\in \mathcal{K}$  to maximize the test power, which is equivalent to maximizing the argument of  $\Phi$ .

# 3 OPTIMIZING TEST POWER FOR CHANGE-POINT DETECTION

In time series CPD, we denote  $\mathbb{P}$  as the distribution of usual events and  $\mathbb{Q}$  as the distribution for the event when change-points happen. The difficulty of choosing kernels via optimizing test power in Eq. (2) is that we have very limited samples from the abnormal distribution  $\mathbb{Q}$ . Kernel learning in this case may easily overfit, leading to sub-optimal performance in time series CPD.

# 3.1 DIFFICULTIES OF OPTIMIZING KERNELS FOR CPD

To demonstrate how limited samples of  $\mathbb{Q}$  would affect optimizing test power, we consider kernel selection for Gaussian RBF kernels on the blobs dataset (Gretton et al., 2012b; Sutherland et al., 2017), which is considered hard for kernel two-sample test.  $\mathbb{P}$  is a  $5\times 5$  grid of two-dimensional standard normals, with spacing 15 between the centers.  $\mathbb{Q}$  is laid out identically, but with covariance  $\frac{\epsilon_q - 1}{\epsilon_q + 1}$  between the coordinates (so the ratio of eigenvalues in the variance is  $\epsilon_{q}$ ). Left panel of Fig. 2 shows  $X\sim \mathbb{P}$  (red samples),  $Y\sim \mathbb{Q}$  (blue dense samples),  $\tilde{Y}\sim \mathbb{Q}$  (blue sparse samples) with  $\epsilon_q = 6$ . Note that when  $\epsilon_q = 1$ ,  $\mathbb{P} = \mathbb{Q}$ .

For  $\epsilon_q\in \{4,6,8,10,12,14\}$ , we take 10000 samples for  $X,Y$  and 200 samples for  $\tilde{Y}$ . We consider two objectives for choosing kernels: 1) median heuristic; 2) max-ratio  $\eta_{k^*}(X,Y) = \arg \max_k\hat{M}_k(X,Y) / \sqrt{V_m(X,Y)}$ ; among 20 kernel bandwidths. We repeat this process 1000 times and report the test power under false rejection rate  $\alpha = 0.05$ . As shown in the right panel of Fig. 2, optimizing kernels using limited samples  $\tilde{Y}$  significantly decreases the test power compared to  $Y$  (blue curve down to the cyan curve). This result not only verifies our claim on the inaptness of existing kernel learning objectives for CPD task, but also stimulates us with the following question, How to optimize kernels with very limited samples from  $\mathbb{Q}$ , even none in an extreme?

# 3.2 A PRACTICAL LOWER BOUND ON OPTIMIZING TEST POWER

We first assume there exist a surrogate distribution  $\mathbb{G}$  that we can easily draw samples from  $(Z\sim \mathbb{G},$ $|Z|\gg |\tilde{Y} |)$ , and also satisfies the following property:

$$
M _ {k} (\mathbb {P}, \mathbb {P}) <   M _ {k} (\mathbb {P}, \mathbb {G}) <   M _ {k} (\mathbb {P}, \mathbb {Q}), \forall k \in \mathcal {K}, \tag {3}
$$

Besides, we assume dealing with non-trivial case of  $\mathbb{P}$  and  $\mathbb{Q}$  where a lower bound  $\frac{1}{m} v_{l} \leq V_{m,k}(\mathbb{P},\mathbb{Q}), \forall k$  exists. Since  $M_{k}(\mathbb{P},\mathbb{Q})$  is bounded, there exists an upper bound  $v_{u}$ . With bounded variance  $\frac{v_l}{m} \leq V_{m,k}(\mathbb{P},\mathbb{Q}) \leq \frac{v_u}{m}$  condition, we derive an lower bound  $\gamma_{k*}(\mathbb{P},\mathbb{G})$  of the test power

$$
\max  _ {k \in \mathcal {K}} \frac {M _ {k} (\mathbb {P} , \mathbb {Q})}{\sqrt {V _ {m} (\mathbb {P} , \mathbb {Q})}} - \frac {c _ {\alpha} / m}{\sqrt {V _ {m} (\mathbb {P} , \mathbb {Q})}} \geq \max  _ {k \in \mathcal {K}} \frac {M _ {k} (\mathbb {P} , \mathbb {Q})}{\sqrt {v _ {u} / m}} - \frac {c _ {\alpha}}{\sqrt {m v _ {l}}} \geq \max  _ {k \in \mathcal {K}} \frac {M _ {k} (\mathbb {P} , \mathbb {G})}{\sqrt {v _ {u} / m}} - \frac {c _ {\alpha}}{\sqrt {m v _ {l}}} = \gamma_ {k *} (\mathbb {P}, \mathbb {G}). \tag {4}
$$

Just for now in the blob toy experiment, we artifact this distribution  $\mathbb{G}$  by mimicking  $\mathbb{Q}$  with the covariance  $\epsilon_g = \epsilon_q - 2$ . We defer the discussion on how to find  $\mathbb{G}$  in the later subsection 3.3.

![](images/8482651b864d5e65d86de8319ed97ede47c22554a7f40f58aa2df6f8de9919a2.jpg)

![](images/4597f9f4463b3ed598e764a55a6dc7d9c50568011091fdc17ee382812f921ef0.jpg)

![](images/3eac0c6b302a8b8050729e08dca05cc6ab86ecdd709ee03da604090b7490e9be.jpg)  
Figure 2: Left:  $5 \times 5$  Gaussian grid, samples from  $\mathbb{P}$ ,  $\mathbb{Q}$  and  $\mathbb{G}$ . We discuss two cases of  $\mathbb{Q}$ , one of sufficient samples, the other of insufficient samples. Right: Test power of kernel selection versus  $\epsilon_{q}$ . Choosing kernels by  $\gamma_{k^{*}}(X,Z)$  using a surrogate distribution  $\mathbb{G}$  is advantageous when we do not have sufficient samples from  $\mathbb{Q}$ , which is typically the case in time series CPD task.

![](images/ed306946fbe134639803fc4c2534846c867db23daad1712665e6924e4802637e.jpg)

![](images/a264d1c4263b1792af4f4b2f0fbfb28d0c646cf352199968bdb8a6ad222e9b56.jpg)

Choosing kernels via  $\gamma_{k^*}(X,Z)$  using surrogate samples  $Z\sim \mathbb{G}$ , as represented by the green curve in Fig. 2, substantially boosts the test power compared to  $\eta_{k*}(X,\tilde{Y})$  with sparse samples  $\tilde{Y}\sim \mathbb{Q}$ .

Test Threshold Approximation Under  $H_0: \mathbb{P} = \mathbb{Q}$ ,  $m\hat{M}_k(X,Y)$  converges asymptotically to a distribution that depends on the unknown data distribution  $\mathbb{P}$  (Gretton et al., 2012a, Theorem 12); we thus cannot evaluate the test threshold  $c_{\alpha}$  in closed form. Common ways of estimating threshold include the permutation test and a estimated null distribution based on approximating the eigenspectrum of the kernel. Nonetheless, both are still computational demanding in practice. Even with the estimated threshold, it is difficult to optimize  $c_{\alpha}$  because it is a function of  $k$  and  $\mathbb{P}$ .

For  $X, X' \sim \mathbb{P}$ , we know that  $c_{\alpha}$  is a function of the empirical estimator  $\hat{M}_k(X, X')$  that controls the Type I error. Bounding  $\hat{M}_k(X, X')$  could be an approximation of bounding  $c_{\alpha}$ . Therefore, we propose the following objective that maximizing a lower bound of test power

$$
\underset {k \in \mathcal {K}} {\operatorname {a r g m a x}} M _ {k} (\mathbb {P}, \mathbb {G}) - \lambda \hat {M} _ {k} (X, X ^ {\prime}), \tag {5}
$$

where  $\lambda$  is a hyper-parameter to control the trade-off between Type-I and Type-II errors, as well as absorbing the constants  $m, v_{l}, v_{u}$  in variance approximation. Note that in experiment, the optimization of Eq. (5) is solved using the unbiased estimator of  $M_{k}(\mathbb{P},\mathbb{G})$  with empirical samples.

# 3.3 SURROGATE DISTRIBUTIONS USING GENERATIVE MODELS

The remaining question is how to construct the surrogate distribution  $\mathbb{G}$  without any sample from  $\mathbb{Q}$ . Injecting random noise to  $\mathbb{P}$  is a simple way to construct  $\mathbb{G}$ . While straightforward, it may result in a sub-optimal  $\mathbb{G}$  because of sensitivity to the level of injected random noise. As no prior knowledge of  $\mathbb{Q}$ , to ensure (3) hold for any possible  $\mathbb{Q}$  (e.g.  $\mathbb{Q} \neq \mathbb{P}$  but  $\mathbb{Q} \approx \mathbb{P}$ ), intuitively, we have to make  $\mathbb{G}$  as closed to  $\mathbb{P}$  as possible. We propose to learn an auxiliary generative model  $\mathbb{G}_{\theta}$  parameterized by  $\theta$  such that

$$
\hat {M} _ {k} (X, X ^ {\prime}) <   \min _ {\theta} M _ {k} (\mathbb {P}, \mathbb {G} _ {\theta}) <   M _ {k} (\mathbb {P}, \mathbb {Q}), \forall k \in \mathcal {K}.
$$

To ensure the first inequality hold, we set early stopping criterion when solving  $\mathbb{G}_{\theta}$  in practice. Also, if  $\mathbb{P}$  is sophisticated, which is common in time series cases, limited capacity of parametrization of  $\mathbb{G}_{\theta}$  with finite size model (e.g. neural networks) (Arora et al., 2017) and finite samples of  $\mathbb{P}$  also hinder us to fully recover  $\mathbb{P}$ . Therefore, we result in a min-max formulation to consider all possible  $k \in \mathcal{K}$  when we learn  $\mathbb{G}$ ,

$$
\min  _ {\theta} \max  _ {k \in \mathcal {K}} M _ {k} (\mathbb {P}, \mathbb {G} _ {\theta}) - \lambda \hat {M} _ {k} (X, X ^ {\prime}), \tag {6}
$$

and solve the kernel for the hypothesis test in the mean time. In experiment, we use simple alternative (stochastic) gradient descent to solve each other.

Lastly, we remark that although the resulted objective (6) is similar to Li et al. (2017), the motivation and explanation are different. One major difference is we aim to find  $k$  with highest test power while their goal is finding  $\mathbb{G}_{\theta}$  to approximate  $\mathbb{P}$ . A more detailed discussion can be found in Appendix A.

# 4 KLCPD: REALIZATION FOR TIME SERIES APPLICATIONS

In this section, we present a realization of the kernel learning framework for time series CPD.

**Compositional Kernels** To have a more expressive kernel for complex time series, we consider compositional kernels  $\tilde{k} = k \circ f$  that combines RBF kernels  $k$  with injective functions  $f_{\phi}$ :

$$
K = \left\{\tilde {k} \mid \tilde {k} (x, x ^ {\prime}) = \exp (- \| f _ {\phi} (x) - f _ {\phi} (x) ^ {\prime} \| ^ {2}) \right\}. \tag {7}
$$

The resulted kernel  $\tilde{k}$  is still characteristic if  $f$  is an injective function and  $k$  is characteristic (Gretton et al., 2012a). This ensures the MMD endowed by  $\tilde{k}$  is still a valid probabilistic distance. One example function class is  $\{f_{\phi}|f_{\phi}(x) = \phi x,\phi >0\}$ , equivalent to the kernel bandwidth tuning. Inspired by the recent success of combining deep neural networks into kernels (Wilson et al., 2016; Al-Shedivat et al., 2017; Li et al., 2017), we parameterize the injective functions  $f_{\phi}$  by recurrent neural networks (RNNs) to capture the temporal dynamics of time series.

For an injective function  $f$ , there exists a function  $F$  such that  $F(f(x)) = x, \forall x \in \mathcal{X}$ , which can be approximated by an auto-encoder via sequence-to-sequence architecture for time series. One practical realization of  $f$  would be a RNN encoder parametrized by  $\phi$  while the function  $F$  is a RNN decoder parametrized by  $\psi$  trained to minimize the reconstruction loss. Thus, our final objective is

$$
\min  _ {\theta} \max  _ {\phi} M _ {f _ {\phi}} (\mathbb {P}, \mathbb {G} _ {\theta}) - \lambda \cdot \hat {M} _ {f _ {\phi}} (X, X ^ {\prime}) - \beta \cdot \mathbb {E} _ {\nu \in \mathbb {P} \cup \mathbb {G} _ {\theta}} \| \nu - F _ {\psi} (f _ {\phi} (\nu)) \| _ {2} ^ {2}. \tag {8}
$$

Practical Implementation In practice, we consider two consecutive windows in mini-batch to estimate  $\hat{M}_{f_{\phi}}(X,X^{\prime})$  in an online fashion for the sake of efficiency. Specifically, the sample  $X\sim \mathbb{P}$  is divided into the left window segment  $X^{(l)} = \{x_{t - w},\ldots ,x_{t - 1}\}$  and the right window segment  $X^{(r)} = \{x_{t},\dots,x_{t + w - 1}\}$  such that  $X = \{X^{(l)},X^{(r)}\}$ . We now reveal implementation details of the auxiliary generative model and the deep kernel.

Generator  $g_{\theta}$  Instead of modeling the explicit density  $\mathbb{G}_{\theta}$ , we model a generator  $g_{\theta}$  where we can draw samples from. The goal of  $g_{\theta}$  is to generate plausibly counterfeit but natural samples based on historical  $X\sim \mathbb{P}$ , which is similar to the conditional GANs (Mirza & Osindero, 2014; Isola et al., 2017). We use sequence-to-sequence (Seq2Seq) architectures (Sutskever et al., 2014) where  $g_{\theta_e}$  encodes time series into hidden states, and  $g_{\theta_d}$  decodes it with the distributional autoregressive process to approximate the surrogate sample  $Z$ :

$$
H = g _ {\theta_ {e}} \big (X ^ {(l)}, \mathbf {0} \big), \quad \tilde {h} = h _ {t - 1} + \omega , \quad Z = g _ {\theta_ {d}} \big (X _ {\gg 1} ^ {(r)}, \tilde {h} \big).
$$

where  $\omega \sim \mathbb{P}(W)$  is a  $d_h$ -dimensional random noise sampled from a base distribution  $\mathbb{P}(W)$  (e.g., uniform, Gaussian).  $H = [h_{t - w},\dots ,h_{t - 1}]\in \mathbb{R}^{d_h\times w}$  is a sequence of hidden states of the generator's encoder.  $X_{\gg 1}^{(r)} = \{\mathbf{0},x_t,x_{t + 1},\ldots ,x_{t + w - 2}\}$  denotes right shift one unit operator over  $X^{(r)}$

Deep Kernel Parametrization We aim to maximize a lower bound of test power via backpropagation on  $\phi$  using the deep kernel form  $\tilde{k} = k \circ f_{\phi}$ . On the other hand, we can also view the deep kernel parametrization as an embedding learning on the injective function  $f_{\phi}(x)$  that can be distinguished by MMD. Similar to the design of generator, the deep kernel is a Seq2Seq framework with one GRU layer of the follow form:

$$
H _ {\nu} = f _ {\phi} (\nu), \hat {\nu} = F _ {\psi} (H _ {\nu}).
$$

where  $\nu \sim \mathbb{P} \cup \mathbb{G}_{\theta}$  are from either the time series data  $X$  or the generated sample  $Z \sim g_{\theta}(\omega | X)$ .

We present an realization of KL-CPD in Algorithm 1 with the weight-clipping technique. The stopping condition is based on a maximum number of epochs or the detecting power of kernel MMD  $M_{f_{\phi}}(\mathbb{P},\mathbb{G}_{\theta})\leq \epsilon$ . This ensures that the surrogate  $\mathbb{G}_{\theta}$  is not too close to  $\mathbb{P}$ , as motivated in Sec. 3.2.

Algorithm 1: KL-CPD, our proposed algorithm.  
input:  $\alpha$  the learning rate,  $c$  the clipping parameter,  $w$  the window size,  $n_c$  the number of iterations of deep kernels training per generator update.  
while  $M_{k\circ f_{\phi}}(\mathbb{P},\mathbb{G}_{\theta}) > \epsilon$  do  
for  $t = 1,\dots ,n_c$  do  
Sample a minibatch  $X_{t}\sim \mathbb{P}$  denote  $X_{t} = \{X_{t}^{(l)},X_{t}^{(r)}\}$  and  $\omega \sim \mathbb{P}(\Omega)$   
gradient  $(\phi)\gets \nabla_{\phi}M_{k\circ f_{\phi}}(\mathbb{P},\mathbb{G}_{\theta}) - \lambda \hat{M}_{k\circ f_{\phi}}(X_{t}^{(l)},X_{t}^{(r)}) - \beta \mathbb{E}_{\nu \sim \mathbb{P}\cup \mathbb{G}_{\theta}}\| \nu -F_{\psi}(f_{\phi}(\nu))\| _2^2$ $\phi \gets \phi +\alpha \cdot \mathrm{RMSProp}(\phi ,\mathrm{gradient}(\phi))$ $\phi \gets \mathrm{clip}(\phi , - c,c)$   
Sample a minibatch  $X_{t^{\prime}}\sim \mathbb{P}$  denote  $X_{t^{\prime}} = \{X_{t^{\prime}}^{(l)},X_{t^{\prime}}^{(r)}\}$  and  $\omega \sim \mathbb{P}(\Omega)$   
gradient  $(\theta)\gets \nabla_{\theta}M_{k\circ f_{\phi}}(\mathbb{P},\mathbb{G}_{\theta})$ $\theta \gets \theta -\alpha \cdot \mathrm{Adam}(\theta ,\mathrm{gradient}(\theta))$

# 5 EVALUATION ON REAL-WORLD DATA

The section presents a comparative evaluation of the proposed KL-CPD and seven representative baselines on benchmark datasets from real-world applications of CPD, including the domains of biology, environmental science, human activity sensing, and network traffic loads. The data statistics are summarized in Table 1. We pre-process all dataset by normalizing each dimension in the range of [0, 1]. Detailed descriptions are available in Appendix B.1.

Following Lai et al. (2018); Saatci et al. (2010); Liu et al. (2013), the datasets are split into the training set  $(60\%)$ , validation set  $(20\%)$  and test set  $(20\%)$  in chronological order. Note that training is fully unsupervised for all methods while labels in the validation set are used for hyperparameters tuning.

For quantitative evaluation, we consider receiver operating characteristic (ROC) curves of

<table><tr><td>Dataset</td><td>T</td><td>#sequences</td><td>domain</td><td>#labels</td></tr><tr><td>Bee-Dance</td><td>826.66</td><td>6</td><td>R3</td><td>19.5</td></tr><tr><td>Fishkiller</td><td>45175</td><td>1</td><td>R+</td><td>899</td></tr><tr><td>HASC</td><td>39397</td><td>1</td><td>R3</td><td>65</td></tr><tr><td>Yahoo</td><td>1432.13</td><td>15</td><td>R+</td><td>36.06</td></tr></table>

anomaly detection results, and measure the area-under-the-curve (AUC) as the evaluation metric. AUC is commonly used in CPD literature (Li et al., 2015a; Liu et al., 2013; Xu et al., 2017).

We compare KL-CPD with real-time CPD methods (ARMA, ARGP, RNN,LSTNet) and retrospective CPD methods (ARGP-BOCPD, RDR-KCPD, Mstats-KCPD). Details are in Appendix B.3. Note that OPT-MMD is a deep kernel learning baseline which optimizes MMD by treating past samples as  $\mathbb{P}$  and the current window as  $\mathbb{Q}$  (insufficient samples).

Table 1: Dataset.  $T$  is length of time series, #labels is average number of labeled change points.  

<table><tr><td>Method</td><td>Bee-Dance</td><td>Fishkiller</td><td>HASC</td><td>Yahoo</td></tr><tr><td>ARMA (Box, 2013)</td><td>0.5368</td><td>0.8794</td><td>0.5863</td><td>0.8615</td></tr><tr><td>ARGP (Candela et al., 2003)</td><td>0.5833</td><td>0.8813</td><td>0.6448</td><td>0.9318</td></tr><tr><td>RNN (Cho et al., 2014)</td><td>0.5827</td><td>0.8872</td><td>0.6128</td><td>0.8508</td></tr><tr><td>LSTNet (Lai et al., 2018)</td><td>0.6168</td><td>0.9127</td><td>0.5077</td><td>0.8863</td></tr><tr><td>ARGP-BOCPD (Saatçi et al., 2010)</td><td>0.5089</td><td>0.8333</td><td>0.6421</td><td>0.9130</td></tr><tr><td>RDR-KCPD (Liu et al., 2013)</td><td>0.5197</td><td>0.4942</td><td>0.4217</td><td>0.6029</td></tr><tr><td>Mstats-KCPD (Li et al., 2015a)</td><td>0.5616</td><td>0.6392</td><td>0.5199</td><td>0.6961</td></tr><tr><td>OPT-MMD</td><td>0.5262</td><td>0.7517</td><td>0.6176</td><td>0.8193</td></tr><tr><td>KL-CPD (Proposed method)</td><td>0.6767</td><td>0.9596</td><td>0.6490</td><td>0.9146</td></tr></table>

Table 2: AUC on four real-world datasets. KL-CPD has the best AUC on three out of four datasets.

# 5.1 MAIN RESULTS

In Table 2, the first four rows present the real-time CPD methods, followed by three retrospective-CPD models, and the last is our proposed method. KL-CPD shows significant improvement over the other methods on all the datasets, except being in a second place on the Yahoo dataset, with

$2\%$  lower AUC compared to the leading ARGP. This confirms the importance of data-driven kernel selection and effectiveness of our kernel learning framework. Notice that OPT-MMD performs not so good compared to KL-CPD, which again verifies our simulated example in Sec. 3 that directly applying existing kernel learning approaches with insufficient samples may not be suitable for real-world CPD task.

Distribution matching approaches like RDR-KCPD and Mstats-KCPD are not as competitive as KL-CPD, and often inferior to real-time CPD methods. One explanation is both RDR-KCPD and Mstats-KCPD measure the distribution distance in the original data space with simple kernel selection using the median heuristic. The change-points may be hard to detect without the latent embedding learned by neural networks.

KL-CPD, instead, leverages RNN to extract useful contexts and encodes time series in a discriminative embedding (latent space) on which kernel two-sample test is used to detection changing points. This also explains the inferior performance of Mstats-KCPD which uses kernel MMD with a fix RBF kernel. That is, using a fixed kernel to detect versatile types of change points is likely to fail.

![](images/bd3b15eda23f9cb0939e431618a092714351b02ddfc105a9d811efbc7096a89d.jpg)  
Figure 3: Ablation test of KL-CPD.

![](images/dbd5bf1345a2af34de08d32495e51ee0c43107ddfa2d53920c888c54abadbd1b.jpg)  
Figure 4: AUC vs. different window size  $w_{r}$  on Bee-Dance.

# 5.2 ABLATION TEST ON LEARNING KERNELS WITH DIFFERENT ENCODERS

We further examine how different encoders  $f_{\phi}$  affects KL-CPD. For MMD-dataspace,  $f_{\phi}$  is an identity map, equivalent to kernel selection with median heuristic in data space. For MMD-codespace,  $\{f_{\phi}, F_{\psi}\}$  is a Seq2Seq autoencoder minimizing reconstruction loss without optimizing test power. For MMD-negsample, the same objective as KL-CPD except for replacing the auxiliary generator with injecting Gaussian noise to  $\mathbb{P}$ .

The results are shown in Figure 3. We first notice the mild improvement of MMD-codespace over MMD-dataspace, showing that using MMD on the induced latent space is effective for discovering beneficial kernels for time series CPD. Next, we see MMD-negsample outperforms MMD-codespace, showing the advantages of injecting a random perturbation to the current interval to approximate  $g_{\theta}(z|X^{(l)})$ . This also justify the validity of the proposed lower bound approach by optimizing  $M_{k}(\mathbb{P},\mathbb{G})$ , which is effective even if we adopt simple perturbed  $\mathbb{P}$  as  $\mathbb{G}$ . Finally, KL-CPD models the  $\mathbb{G}$  with an auxiliary generator  $g_{\theta}$  to obtain conditional samples that are more complex and subtle than the perturbed samples in MMD-negsample, resulting in even better performance.

In Figure 4, we also demonstrate how the tolerance of delay  $w_{r}$  influences the performance. Due to space limit, results other than Bee-Dance dataset are omitted, given they share similar trends. KL-CPD shows competitive AUC mostly, only slightly decreases when  $w_{r} = 5$ . MMD-dataspace and MMD-codespace, in contrast, AUC degradation is much severe under low tolerance of delay  $(w_{r} = \{5,10\})$ . The conditional generated samples from KL-CPD can be found in Appendix B.5.

# 6 IN-DEPTH ANALYSIS ON SIMULATED DATA

To further explore the performance of KL-CPD with controlled experiments, we follow other time series CPD papers (Takeuchi & Yamanishi, 2006; Liu et al., 2013; Matteson & James, 2014) to create three simulated datasets each with a representative change-point characteristic: jumping mean, scaling variance, and alternating between two mixtures of Gaussian (Gaussian-Mixtures). More description of the generated process see Appendix B.2.

<table><tr><td>Method</td><td>Jumping-Mean</td><td>Scaling-Variance</td><td>Gaussian-Mixtures</td></tr><tr><td>ARMA</td><td>0.7731 (0.06)</td><td>0.4801 (0.07)</td><td>0.5035 (0.08)</td></tr><tr><td>ARGP</td><td>0.4770 (0.03)</td><td>0.4910 (0.07)</td><td>0.5027 (0.08)</td></tr><tr><td>RNN</td><td>0.5053 (0.03)</td><td>0.5177 (0.08)</td><td>0.5053 (0.08)</td></tr><tr><td>LSTNet</td><td>0.7694 (0.09)</td><td>0.4906 (0.07)</td><td>0.4985 (0.07)</td></tr><tr><td>ARGP-BOCPD</td><td>0.7983 (0.06)</td><td>0.4767 (0.08)</td><td>0.5027 (0.08)</td></tr><tr><td>RDR-KCPD</td><td>0.6484 (0.11)</td><td>0.7574 (0.06)</td><td>0.6022 (0.11)</td></tr><tr><td>Mstats-KCPD</td><td>0.7309 (0.05)</td><td>0.7534 (0.04)</td><td>0.6026 (0.08)</td></tr><tr><td>KL-CPD</td><td>0.9454 (0.02)</td><td>0.8823 (0.03)</td><td>0.6782 (0.05)</td></tr></table>

Table 3: AUC on three artificial datasets. Mean and standard deviation under 10 random seeds.

# 6.1 MAIN RESULTS ON SIMULATED DATA

The results are summarized in Table 3. KL-CPD achieves the best in all cases. Interestingly, retrospective-CPD (ARGP-BOCPD, RDR-KCPD, Mstats-KCPD) have better results compared to real-time CPD (ARMA, ARGP, RNN,LSTNet), which is not the case in real-world datasets. This suggests low reconstruction error does not necessarily lead to good CPD accuracies.

As for why Mstats-KCPD does not have comparable performance as KL-CPD, given that both of them use MMD as distribution distance? Notice that Mstats-KCPD assumes the reference time series (training data) follows the same distribution as the current interval. However, if the reference time series is highly non-stationary, it is more accurate to compute the distribution distance between the latest past window and the current window, which is the essence of KL-CPD.

# 6.2 MMD VERSUS DIMENSIONALITY OF DATA

We study how different encoders  $f_{\phi}$  would affect the power of MMD versus the dimensionality of data. We generate an simulated time series dataset by sampling between two multivariate Gaussian  $\mathcal{N}(0, \sigma_1^2 I_d)$  and  $\mathcal{N}(0, \sigma_2^2 I_d)$  where the dimension  $d = \{2, 4, 6, \ldots, 20\}$  and  $\sigma_1 = 0.75, \sigma_2 = 1.25$ .

Figure 5 plots the one-dimension data and AUC results. We see that all methods remain equally strong in low dimensions ( $d \leq 10$ ), while MMD-dataspace decreases significantly as data dimensionality increases ( $d \geq 12$ ). An explanation is non-parametric statistical models require the sample size to grow exponentially with the dimensionality of data, which limits the performance of MMD-dataspace because of the fixed sample size. On the other hand, MMD-codespace and KL-CPD are conducting kernel two-sample test on a learned low dimension codespace, which moderately alleviates this issue. Also, KL-CPD finds a better kernel (embedding) than MMD-codespace by optimizing the lower bound of the test power.

![](images/43af0c1e90a76cccb06f0351e4e95131272951c8bb87dc3b2232954e794edf00.jpg)  
Figure 5: MMD with different encoder  $f_{\phi_e}$  versus data dimension, under 10 random seeds.

![](images/c2f03d77a69a47b690e79a6216f7379f7ee36ebb2082e7100eea252a66b4dfd7.jpg)

# 7 CONCLUSION

In this paper, we propose KL-CPD, a novel kernel learning framework that optimizes a lower bound on test power of kernel two-sample test. The deep kernel parametrization of KL-CPD combines the latent space of RNNs with RBF kernels that effectively detect a variety of change-points from different real-world applications. Extensive evaluation of our new approach along with strong baseline methods on benchmark datasets shows the outstanding performance of the proposed method in retrospective CPD. With simulation analysis in addition we can see that the new method not only boosts the kernel power but also evades the performance degradation as data dimensionality increases.

# REFERENCES

Maruan Al-Shedivat, Andrew Gordon Wilson, Yunus Saatchi, Zhiting Hu, and Eric P Xing. Learning scalable deep kernels with recurrent structure. JMLR, 2017.  
Michael Arbel, Dougal J Sutherland, Mikołaj Bińkowski, and Arthur Gretton. On gradient regularizers for mmd gans. In NIPS, 2018.  
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (gans). In ICML, 2017.  
Michèle Basseville, Igor V Nikiforov, et al. Detection of abrupt changes: theory and application. Prentice Hall Englewood Cliffs, 1993.  
Sabyasachi Basu and Martin Meckesheimer. Automatic outlier detection for time series: an application to sensor data. Knowledge and Information Systems, 2007.  
Mikołaj Binkowski, Dougal J Sutherland, Michael Arbel, and Arthur Gretton. Demystifying mmd gans. In ICLR, 2018.  
George Box. Box and jenkins: time series analysis, forecasting and control. A Very British Affair, ser. Palgrave Advanced Texts in Econometrics. Palgrave Macmillan UK, 2013.  
E Brodsky and Boris S Darkhovsky. Nonparametric methods in change point problems. Springer Science & Business Media, 2013.  
Joaquin Quinonero Candela, Agathe Girard, Jan Larsen, and Carl Edward Rasmussen. Propagation of uncertainty in bayesian kernel models-application to multiple-step ahead forecasting. In ICASSP. IEEE, 2003.  
Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM computing surveys (CSUR), 2009.  
Kyunghyun Cho, Bart van Merrienboer, Caglar Gülçehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In EMNLP, 2014. URL http://aclweb.org/anthology/D/D14/D14-1179.pdf.  
Gintare Karolina Dziugaite, Daniel M Roy, and Zoubin Ghahramani. Training generative neural networks via maximum mean discrepancy optimization. In UAI, 2015.  
Andrew B Gardner, Abba M Krieger, George Vachtsevanos, and Brian Litt. One-class novelty detection for seizure analysis from intracranial eeg. JMLR, 2006.  
Arthur Gretton, Karsten M Borgwardt, Malte Rasch, Bernhard Scholkopf, and Alex J Smola. A kernel method for the two-sample-problem. In NIPS, 2007.  
Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. JMLR, 2012a.  
Arthur Gretton, Dino Sejdinovic, Heiko Strathmann, Sivaraman Balakrishnan, Massimiliano Pontil, Kenji Fukumizu, and Bharath K Sriperumbudur. Optimal kernel choice for large-scale two-sample tests. In NIPS, 2012b.  
Fredrik Gustafsson. The marginalized likelihood ratio test for detecting abrupt changes. IEEE Transactions on automatic control, 1996.  
Fredrik Gustafsson and Fredrik Gustafsson. Adaptive filtering and change detection. CiteSeer, 2000.  
Zaid Harchaoui, Eric Moulines, and Francis R Bach. Kernel change-point analysis. In NIPS, 2009.  
Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In CVPR, 2017.  
Yoshinobu Kawahara, Takehisa Yairi, and Kazuo Machida. Change-point detection in time-series data based on subspace identification. In ICDM. IEEE, 2007.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In SIGIR, 2018.  
Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás Póczos. Mmd gan: Towards deeper understanding of moment matching network. In NIPS, 2017.  
Shuang Li, Yao Xie, Hanjun Dai, and Le Song. M-statistic for kernel change-point detection. In NIPS, 2015a.  
Yujia Li, Kevin Swersky, and Rich Zemel. Generative moment matching networks. In ICML, pp. 1718-1727, 2015b.  
Song Liu, Makoto Yamada, Nigel Collier, and Masashi Sugiyama. Change-point detection in time-series data by relative density-ratio estimation. Neural Networks, 2013.  
David S Matteson and Nicholas A James. A nonparametric approach for multiple change point analysis of multivariate data. Journal of the American Statistical Association, 2014.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Youssef Mroueh, Chun-Liang Li, Tom Sercu, Anant Raj, and Yu Cheng. Sobolev gan. In ICLR, 2018.  
Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017.  
Andrey Pepelyshev and Aleksey S Polunchenko. Real-time financial surveillance via quickest change-point detection methods. arXiv preprint arXiv:1509.01570, 2015.  
Jaxk Reeves, Jien Chen, Xiaolan L Wang, Robert Lund, and Qi Qi Lu. A review and comparison of changepoint detection techniques for climate data. Journal of Applied Meteorology and Climatology, 2007.  
Yunus Saatci, Ryan Turner, and Carl Edward Rasmussen. Gaussian process change point models. In ICML, June 2010.  
Dougal J Sutherland, Hsiao-Yu Tung, Heiko Strathmann, Soumyajit De, Aaditya Ramdas, Alex Smola, and Arthur Gretton. Generative models and model criticism via optimized maximum mean discrepancy. In ICLR, 2017.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In NIPS, 2014.  
Jun-ichi Takeuchi and Kenji Yamanishi. A unifying framework for detecting outliers and change points from time series. IEEE transactions on Knowledge and Data Engineering, 2006.  
Yao Wang, Chunguo Wu, Zhaohua Ji, Binghong Wang, and Yanchun Liang. Non-parametric change-point method for differential gene expression detection. PloS one, 2011.  
Andrew Gordon Wilson, Zhiting Hu, Ruslan Salakhutdinov, and Eric P Xing. Deep kernel learning. In AISTATS, 2016.  
Zhao Xu, Kristian Kersting, and Lorenzo von Ritter. Stochastic online anomaly analysis for streaming time series. In *IJCAI*, 2017.  
Kenji Yamanishi and Jun-ichi Takeuchi. A unifying framework for detecting outliers and change points from non-stationary time series data. In SIGKDD. ACM, 2002.  
Kenji Yamanishi, Jun-Ichi Takeuchi, Graham Williams, and Peter Milne. On-line unsupervised outlier detection using finite mixtures with discounting learning algorithms. Data Mining and Knowledge Discovery, 2004.
