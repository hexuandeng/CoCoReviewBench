# A UNIFIED FRAMEWORK FOR RANDOMIZED SMOOTHING BASED CERTIFIED DEFENSES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Randomized smoothing, which was recently proved to be a certified defensive technique, has received considerable attention due to its scalability to large datasets and neural networks. However, several important questions still remain unanswered in the existing frameworks, such as (i) whether Gaussian mechanism is an optimal choice for certifying  $\ell_2$ -normed robustness, and (ii) whether randomized smoothing can certify  $\ell_{\infty}$ -normed robustness (on high-dimensional datasets like ImageNet). To answer these questions, we introduce a unified and self-contained framework to study randomized smoothing-based certified defenses, where we mainly focus on the two most popular norms in adversarial machine learning, i.e.,  $\ell_2$  and  $\ell_{\infty}$  norm. We answer the above two questions by first demonstrating that Gaussian mechanism and Exponential mechanism are the (near) optimal options to certify the  $\ell_2$  and  $\ell_{\infty}$ -normed robustness. We further show that the largest  $\ell_{\infty}$  radius certified by randomized smoothing is upper bounded by  $O(1 / \sqrt{d})$ , where  $d$  is the dimensionality of the data. This theoretical finding suggests that certifying  $\ell_{\infty}$ -normed robustness by randomized smoothing may not be scalable to high-dimensional data. The veracity of our framework and analysis is verified by extensive evaluations on CIFAR10 and ImageNet.

# 1 INTRODUCTION

The past decade has witnessed tremendous success of deep learning in handling various learning tasks like image classification (Krizhevsky et al., 2012), natural language processing (Cho et al., 2014), and game playing (Silver et al., 2016). Nevertheless, a major unresolved issue of deep learning is its vulnerability to adversarial samples that are almost indistinguishable from natural samples to humans but can mislead deep neural networks (DNNs) to make wrong predictions with high confidence (Szegedy et al., 2013; Goodfellow et al., 2014). This phenomenon, referred to as adversarial attack, is considered to be one of the biggest threats to the deployment of many deep learning systems. Thus, a great deal of effort has been devoted to developing defensive techniques for it. However, the majority of the existing defenses are of heuristic nature (i.e., without any theoretical guarantees), implying that they may be ineffective against stronger attacks. Recent works (He et al., 2017; Athalye et al., 2018; Uesato et al., 2018) have confirmed this concern, and showed that most of those heuristic defenses actually fail to defend stronger adaptive attacks. This forces us to shift our attentions to certifiable defenses as they can classify all the samples in a predefined neighborhood of the natural samples with a theoretically-guaranteed error bound. Among all existing certifiable defensive techniques, randomized smoothing emerges as the most popular one due to its scalability to large datasets and arbitrary networks. Remarkably, using the Gaussian mechanism for randomized smoothing, Cohen et al. (2019) successfully certify  $49\%$  accuracy on the original ImageNet dataset under adversarial perturbations with  $\ell_2$  norm less than 0.5. Despite these successes, there are still several unanswered questions regarding randomized smoothing based certified defenses. One of such questions is, why should Gaussian noise be used for randomized smoothing to certify  $\ell_2$ -normed robustness, and is Gaussian mechanism the best option? Another important question is regarding the generalizability of this method to other norms, especially the  $\ell_{\infty}$  norm. If randomized smoothing can be used to certify  $\ell_{\infty}$ -normed robustness, what mechanism is the optimal choice?

To shed light on the above questions, we propose in this paper a unified and self-contained framework for randomized smoothing-based certified defenses. We look at the problem from a differential privacy's point of view and present two types of robustness in this framework. One is motivated by

<table><tr><td rowspan="2">Mechanism</td><td colspan="2">l2-normed</td><td colspan="2">l∞-normed</td></tr><tr><td>D∞ Robustness</td><td>DMR Robustness</td><td>D∞ Robustness</td><td>DMR Robustness</td></tr><tr><td rowspan="2">Gaussian</td><td rowspan="2">unable to certify</td><td>near optimal</td><td rowspan="2">unable to certify</td><td>near optimal</td></tr><tr><td>r scales in O(1)</td><td>r scales in O(1/√d log d)</td></tr><tr><td rowspan="2">Exponential</td><td rowspan="2">not optimal</td><td rowspan="2">not optimal</td><td>optimal</td><td rowspan="2">not optimal</td></tr><tr><td>r scales in O(1/d)</td></tr></table>

Table 1: Summary of our framework

$\epsilon$ -differential privacy ( $\epsilon$ -DP), which uses  $\infty$ -divergence to measure the distance between the probabilities of predictions on randomized natural samples and randomized adversarial samples and is therefore called  $D_{\infty}$  robustness. The other is inspired by  $\epsilon$ -zero concentrated differential privacy ( $\epsilon$ -zCDP) that uses the Maximal Relative Rényi (MR) divergence as the probability distance measurement and is called  $D_{MR}$  robustness. For both of them, we focus on certifying robustness in either  $\ell_2$  or  $\ell_{\infty}$  norm by randomized smoothing. Specifically, our contributions are five-fold:

1. We propose a unified and self-contained framework for certifying  $D_{\infty}$  and/or  $D_{MR}$  robustness in  $\ell_2$  and  $\ell_{\infty}$  norms by randomized smoothing.  
2. In our framework, we demonstrate that the Gaussian mechanism is a near optimal choice for certifying  $D_{MR}$  robustness in  $\ell_2$  norm, and the robust radius is  $O(1)$ .  
3. We also prove that an exponential mechanism is the optimal choice for certifying  $D_{\infty}$  robustness in  $\ell_{\infty}$  norm, but the robust radius is only  $O(1 / d)$ , making it unscalable to high-dimensional data.  
4. We show that the Gaussian mechanism is also a near optimal choice for certifying  $D_{MR}$  robustness in  $\ell_{\infty}$  norm, but the robust radius is  $O(1 / \sqrt{d\log d})$ , making it also hardly scalable to high-dimensional data.  
5. The largest robust  $\ell_{\infty}$  radius that can be certified by randomized smoothing to achieve  $D_{MR}$  robustness is upper bounded by  $O(1 / \sqrt{d})$

Table 1 summarizes the (near) optimal mechanisms of our framework for certifying the  $\ell_2$  and  $\ell_{\infty}$ -normed robustness.

# 2 RELATED WORK

There are three main approaches for certified defenses. The first approach formulates the task of adversarial verification as an optimization problem and solves it by relaxations (Dvijotham et al., 2018; Raghunathan et al., 2018; Wong & Kolter, 2018). The second approach certifies the bounds of the outputs, given the perturbation size, using interval analysis (Mirman et al., 2018; Wang et al., 2018; Gowal et al., 2018). The main issue of the above two approaches is that they can hardly be scaled to large datasets and networks due to the associated high computational complexity. The third approach uses randomized smoothing to certify robustness, and is gaining popularity recently due to its strong scalability (Lecuyer et al., 2018; Li et al., 2018; Cohen et al., 2019). For this approach, Lecuyer et al. (2018) showed that randomized smoothing can certify the  $\ell_2$  and  $\ell_1$ -normed robustness by using inequalities from differential privacy. Li et al. (2018) achieved a stronger guarantee on the  $\ell_2$ -normed robustness using tools from information theory. Cohen et al. (2019) further obtained a tight guarantee on the  $\ell_2$ -normed robustness using Gaussian noise. A remaining issue in all of these works is that they did not give answers to questions like why Gaussian noise is used to certify the  $\ell_2$ -normed robustness and what is the best mechanism to certify the  $\ell_{\infty}$ -normed robustness. To answer these questions, we present in this paper a new general framework to study randomized smoothing based certified defenses.

# 3 ROBUSTNESS MOTIVATED BY DIFFERENTIAL PRIVACY

In this section, we introduce our general framework. Let  $\mathbf{x}$  be a data sample and  $y\in \mathcal{V}$  be its label, where  $\mathcal{V}$  is the label set. We denote by  $f(\cdot)$  a deterministic classifier with prediction  $f(x)$  for any data sample  $x$ . If there exists an  $\mathbf{x}'$  in a small  $l_{p}$  ball centered at  $\mathbf{x}$  and with  $f(\mathbf{x}')\neq f(\mathbf{x})$ ,  $\mathbf{x}'$  is viewed as an adversarial sample.

Definition 1 (Randomized Classifier (Cohen et al., 2019)). Given an input  $x$ , the prediction of a randomized classifier  $g(\cdot)$  is defined as

$$
\operatorname *{argmax}_{c\in \mathcal{Y}}P(g(x) = c).
$$

Specifically, for a randomized smoothing classifier  $g(x) = f(x + Z)$ , where  $Z$  is a random vector and  $f(\cdot)$  is a deterministic classifier, the prediction of  $x$  is the class of  $c$  whose region  $S \triangleq \{ \tilde{x} \in R^d, f(\tilde{x}) = c \}$  has the largest probability measure in the distribution of  $x + Z$  ( $\tilde{x} \sim p(x + Z)$ ).

Before introducing our framework, we first recall the definition of robustness for a deterministic classifier in (Diochnos et al., 2018).

Definition 2 (Robustness (Diochnos et al., 2018)). For a given classifier  $f$ , a sample  $x$  and some norm  $\| \cdot \|$ .  $f$  is  $(r, \| \cdot \|)$ -(error-region) robust on the sample  $x$  if

$$
\forall x ^ {\prime} \in \mathbb {B} (x, r), f (x) = f \left(x ^ {\prime}\right), \tag {1}
$$

where  $\mathbb{B}(x,r)$  is the ball centered at  $x$  and with norm  $\| \cdot \|$  and radius  $r$ .

Note that in Definition 2, the classifier is assumed to be deterministic. To generalize the concept of robustness to randomized classifiers (see Definition 1), we define a relaxed version of the (error-region) robustness. Since  $g(x)$  is a random value, instead of using equality, we measure the difference between  $g(x)$  and  $g(x')$  by a certain divergence. This leads us to the following definition, which is a basic concept in our framework that will be used throughout the paper.

Definition 3 (Relaxed Robustness). For a given (randomized) classifier  $g(\cdot)$ , a sample  $x$  and some norm  $\| \cdot \|$ , the classifier  $g$  is  $(r, D, \| \cdot \|, \epsilon)$ -(error-region) robust on  $x$  if

$$
\forall x ^ {\prime} \in \mathbb {B} (x, r), \max  \left\{D \left(g (x), g \left(x ^ {\prime}\right)\right), D \left(g \left(x ^ {\prime}\right), g (x)\right) \right\} \leq \epsilon . \tag {2}
$$

where  $D$  is some divergence metric between two probability distributions. The max function is used to ensure that the measurement is symmetric.

Compared with Definition 2, there are two additional terms in Definition 3:  $\epsilon$  represents the "distance" or difference between the distributions of  $g(x)$  and  $g(x')$ . When  $\epsilon$  is small, we expect that the distributions of predictions on  $x$  and  $x'$ , i.e.,  $g(x)$  and  $g(x')$ , are almost the same, which is just a generalization of the equality in Definition 2.  $D$  is some divergence measurement between two probability distributions. In this paper, we use two types of divergence,  $\infty$ -Divergence and Maximal Relative Rényi Divergence, to measure the distance between two probability distributions. Correspondingly, we have two types of robustness called  $D_{\infty}$  and  $D_{MR}$  robustness.

Definition 4 ( $\infty$ -Divergence). The  $\infty$ -Divergence  $D_{\infty}$  of distributions  $P$  and  $Q$  is defined as

$$
D _ {\infty} (P \| Q) = \sup  _ {x \in s u p p (Q)} \log \frac {P (x)}{Q (x)},
$$

where  $\operatorname{supp}(Q)$  is the support of the distribution  $Q$ .

Definition 5 (Maximal Relative Rényi Divergence). The Maximal Relative Rényi Divergence  $D_{MR}(P\| Q)$  of distributions  $P$  and  $Q$  is defined as

$$
D _ {M R} (P \| Q) = \max  _ {\alpha \in (1, \infty)} \frac {D _ {\alpha} (P \| Q)}{\alpha},
$$

where  $D_{\alpha}(P\| Q)$  is the Renyi divergence between  $P$  and  $Q$ , which is defined as

$$
D _ {\alpha} (P \| Q) = \frac {1}{\alpha - 1} \log \mathbb {E} _ {x \sim Q} (\frac {P (x)}{Q (x)}) ^ {\alpha}.
$$

Definition 6 ( $D_{\infty}$  Robustness). A randomized smoothing mechanism  $\mathcal{A}(\cdot)$  (including classifiers) is a  $(r, D_{\infty}, \| \cdot \|, \epsilon)$ -robust mechanism if

$$
\forall x ^ {\prime} \in \mathbb {B} (x, r), \max  \left\{D _ {\infty} \left(\mathcal {A} (x), \mathcal {A} \left(x ^ {\prime}\right)\right), D \left(\mathcal {A} \left(x ^ {\prime}\right), \mathcal {A} (x)\right) \right\} \leq \epsilon , \tag {3}
$$

where  $\| \cdot \|$  is the norm of the ball  $\mathbb{B}(x,r)$ . If a randomized smoothing classifier  $g(\cdot)$  satisfies Eq. (3), it is a  $(r, D_{\infty}, \| \cdot \|, \epsilon)$ -robust classifier or it certifies  $D_{\infty}$  Robustness.

$D_{\infty}$  Robustness is motivated by the notion of  $\epsilon$ -differential privacy ( $\epsilon$ -DP) (Dwork et al., 2006). To achieve  $\epsilon$ -DP for a randomized algorithm, we can use several mechanisms such as Laplacian mechanism or Exponential mechanism (see (Dwork et al., 2014) for details). However, it is known that adding Gaussian noise often does not lead to  $\epsilon$ -DP, but rather  $(\epsilon, \delta)$ -DP (Dwork et al., 2014) which has an additional parameter  $\delta$  and thus is harder to be incorporated in our framework. To alleviate this issue, we employ Maximal Relative Rényi Divergence as the probability distance measurement to define another type of robustness, namely  $D_{MR}$  robustness.

Definition 7 ( $D_{MR}$  Robustness). A randomized smoothing mechanism  $\mathcal{A}(\cdot)$  is a  $(r, D_{MR}, \| \cdot \|, \epsilon)$ -robust mechanism if

$$
\forall x ^ {\prime} \in \mathbb {B} (x, r), \max  \left\{D _ {M R} (\mathcal {A} (x), \mathcal {A} \left(x ^ {\prime}\right)), D _ {M R} (\mathcal {A} \left(x ^ {\prime}\right), \mathcal {A} (x)) \right\} \leq \epsilon . \tag {4}
$$

If a randomized smoothing classifier  $g(\cdot)$  satisfies Eq. (4), it is a  $(r, D_{MR}, \| \cdot \|, \epsilon)$ -robust classifier or it certifies  $D_{MR}$  Robustness.

$D_{MR}$  Robustness is inspired by the notion of zero-Concentrated Differential Privacy (zCDP) (Bun & Steinke, 2016), whose connection to DP is shown in the following theorem.

Theorem 8 ((Bun & Steinke, 2016)). Let  $P$  and  $Q$  be two probability distributions satisfying the conditions of  $D_{\infty}(P\|Q) \leq \epsilon$  and  $D_{\infty}(Q\|P) \leq \epsilon$ . Then,  $D_{MR}(P\|Q) \leq \frac{1}{2}\epsilon^2$ .

Theorem 8 indicates that  $D_{MR}$ -robustness is a relaxed version of  $D_{\infty}$ -robustness.

Theorem 9 (Postprocessing Property). Let  $g(x) = f(\mathcal{A}(x))$  be a randomized classifier, where  $f(\cdot)$  is any deterministic function (classifier).  $g(\cdot)$  is  $(r, D, \| \cdot \|, \epsilon)$ -robust if  $\mathcal{A}(\cdot)$  is  $(r, D, \| \cdot \|, \epsilon)$ -robust (where  $D$  includes  $D_{\infty}$  and  $D_{MR}$ ).

The above theorem is derived from the post-processing properties of DP and zCDP. A detailed proof (explanation) is given in Appendix B. This property allows us to concentrate only on the randomized smoothing mechanism  $\mathcal{A}$  without needing to consider the specific form of the deterministic function (classifier)  $f(\cdot)$ . Next, we consider the cases of certifying  $D_{\infty}$  or  $D_{MR}$  robustness using  $\ell_2$  and  $\ell_{\infty}$ -norm.

# 3.1 CERTIFYING  $\ell_2$ -NORMED ROBUSTNESS

The following theorem shows that randomized smoothing by the Gaussian mechanism is  $(r, D_{MR}, \| \cdot \|, \epsilon)$ -robust.

Theorem 10. Let  $f$  be any classifier and  $g(x) = f(x + z)$  be its corresponding randomized classifier for samples  $x \in \mathbb{R}^d$ , where  $z \sim \mathcal{N}(0, \sigma^2 I_d)$ . Then,  $g(\cdot)$  is  $(r, D_{MR}, \| \cdot \|_2, \frac{r^2}{2\sigma^2})$ -robust on any  $x$ . Moreover, let  $\epsilon$  denote  $\frac{r^2}{2\sigma^2}$ . Then, for any  $\lambda > 0$  and any measurable set  $S \neq \emptyset$ , the following holds with probability at least  $1 - \exp(-\frac{\lambda^2}{4\epsilon})$ ,

$$
\log \frac {P (g (x) \in S)}{P (g \left(x ^ {\prime}\right) \in S)} \leq \lambda + \sqrt {\epsilon}. \tag {5}
$$

That is, when  $\lambda = c\sqrt{\epsilon}$ ,  $\log \frac{P(g(x) \in S)}{P(g(x')) \in S} \leq (c + 1)\sqrt{\epsilon}$  with probability  $1 - \exp \left(-\frac{c^2}{4}\right)$ . In practice,  $c = 3$  is enough to achieve a high probability.

Corollary 11. Adding Gaussian noise  $z \in \mathcal{N}(0, \sigma^2 I_d)$  can defend any  $x' \in \mathbb{B}(x, r = \sqrt{2\epsilon}\sigma)$  that satisfies the condition of  $D_{MR}(g(x)\| g(x')) \leq \epsilon$  with probability at least  $1 - \exp(-\frac{c^2}{4})$ . Furthermore,  $\sqrt{\epsilon}$  can be calculated (bounded) by  $(\log p_a - \log p_b) / 2(1 + c)$  or  $(\log p_a / (1 - p_a)) / 2(1 + c)$  (binary case), where  $p_a$  and  $p_b$  are respectively the probabilities of the randomized classifier  $g(\cdot)$  returning the most probable class  $c_a$  and the runner-up class  $c_b$  on input  $x$ .

Detailed proofs for Theorem 10, Corollary 11, and all the following theorems are provided in Appendix B. From Theorem 9, we can see that for classifiers like  $g(x) = f(x + z)$ , we only need to prove that the randomized mechanism  $\mathcal{A}(x) = x + z (z \sim \mathcal{N}(0, \sigma^2 I_d))$  is  $(r, D_{MR}, \| \cdot \|_2, \frac{r^2}{2\sigma^2})$ -robust. Also, the connection between  $\epsilon$  and  $p_a, p_b$  can be derived for all  $\epsilon$  or  $\sqrt{\epsilon}$  (in the certified radii) as in Corollary 11. Note that a similar theorem has also been proved by Cohen et al. (2019). But there are some major differences between our framework and theirs (Cohen et al., 2019). Specifically, our

framework certifies the robustness with a probability of failure, and the certified radius  $r$  depends on  $c$  that controls the probability of failure. A smaller  $c$  yields a larger  $r$  compared to those in Cohen et al. (2019), and vice versa. Moreover, in our framework, we show that the Gaussian mechanism is a near optimal option, by providing a lower bound below for all possible noises that can certify the  $\ell_2$ -normed  $D_{MR}$  robustness.

Next, we consider the following unanswered question (i.e., the first question). Since there are infinite ways of sampling  $z$ , a natural problem is to determine whether Gaussian mechanism is the optimal option to certify the  $\ell_2$ -normed  $D_{MR}$  robustness. To answer this question, we first give a lower bound on the magnitude of the noise  $z$  added in the randomized smoothing mechanism  $\mathcal{A}(x) = x + z$  to ensure that  $\mathcal{A}(x)$ , as well as  $f(\mathcal{A}(x))$ , is  $(r, D_{MR}, \| \cdot \|_2, \epsilon)$ -robust. If the magnitude of Gaussian noise is close to the lower bound, then Gaussian mechanism is considered as "near optimal".

Theorem 12 (Lower Bound of the Noise). For any  $\epsilon \leq O(1)$ , if there is a  $(2r, D_{MR}, \| \cdot \|_2, \frac{\epsilon}{2})$ -robust randomized smoothing mechanism  $\mathcal{A}(x) = x + z: [0, \frac{r}{\sqrt{d}}]^d \mapsto [0, \frac{r}{\sqrt{d}}]^d$  such that for all  $x \in [0, \frac{r}{\sqrt{d}}]^d$ ,

$$
\mathbb {E} [ \| z \| _ {\infty} ] = \mathbb {E} _ {\mathcal {A}} \| \mathcal {A} (x) - x \| _ {\infty} \leq \alpha ,
$$

for some  $\alpha \leq O(1)$ , then it must be true that  $\alpha \geq \Omega\left(\frac{r}{\sqrt{\epsilon}}\right)$ . In another word,  $\Omega\left(\frac{r}{\sqrt{\epsilon}}\right)$  is the lower bound of the expected  $\ell_{\infty}$  norm of the random noise.

Theorem 12 indicates that the expected  $\ell_{\infty}$  norm of the added random noise should be at least  $\Omega \left(\frac{r}{\sqrt{\epsilon}}\right)$  to guarantee  $(r, D_{MR}, \| \cdot \|_2, \epsilon)$ -robustness. For Gaussian mechanism, the expected  $\ell_{\infty}$  norm is  $O(\sigma \sqrt{\log d})$  ((Orabona & Pál, 2015)), which is  $O\left(\frac{r}{\sqrt{\epsilon}} \sqrt{\log d}\right)$  according to Corollary 11. This means that Gaussian mechanism is near optimal (up to an  $O(\sqrt{\log d})$  factor). Equivalently, if we fix the magnitude of the expected  $\ell_{\infty}$ -norm of the added noise as  $\alpha$ , the largest radius  $r$  that can be certified by any  $(r, D_{MR}, \| \cdot \|_2, \epsilon)$ -robust randomized smoothing mechanisms is upper bounded by  $O(\alpha \sqrt{\epsilon})$ , which is also close to the robust radius guaranteed by Gaussian mechanism (up to an  $O(\sqrt{\log d})$  factor).

# 3.2 CERTIFYING  $\ell_{\infty}$ -NORMED ROBUSTNESS

Previous work on the randomized smoothing-based certified defenses (Cohen et al., 2019; Li et al., 2018) mainly uses Gaussian noise to certify the  $\ell_2$ -normed robustness. Thus, another natural question (i.e., the second question) is to determine whether randomized smoothing can use some mechanism to certify the  $\ell_{\infty}$ -normed robustness. In this section, we consider this question using our general framework.

Before extending our result to the  $\ell_{\infty}$ -normed case, we first recall the  $\ell_{2}$ -normed case and investigate the form of the density function of Gaussian noise:  $p(z) \propto \exp \left(-\frac{\|\pmb{z}\|_2^2}{\sigma^2}\right)$ . Based on this, we conjecture that, to certify  $\ell_{\infty}$ -normed robustness, we can sample the noise using an exponential mechanism:

$$
p (\boldsymbol {z}) \propto \exp \left(- \frac {\| \boldsymbol {z} \| _ {\infty}}{\sigma}\right). \tag {6}
$$

We show in the following theorem that randomized smoothing by (6) certifies  $(r, D_{MR}, \| \cdot \|_{\infty}, \cdot)$ -robustness, which could be considered as an extension of the  $\ell_2$ -normed case. Moreover, we can prove that it is  $(r, D_{\infty}, \| \cdot \|_{\infty}, \cdot)$ -robust. However, the certified radius  $r$  is  $O(1 / d)$ , which implies that it is unscalable to high-dimensional data.

Theorem 13. Let  $f$  be any classifier and  $g(x) = f(x + z)$  be its corresponding randomized classifier for sample  $x \in \mathbb{R}^d$ , where the noise  $z \sim p(z)$  in (6). Then,  $g(\cdot)$  is  $(r, D_{MR}, \| \cdot \|_{\infty}, \frac{r^2}{2\sigma^2})$ -robust. Moreover, it is  $(r, D_{\infty}, \| \cdot \|_{\infty}, \frac{r}{\sigma})$ -robust.

Remark 14. Due to the high dimensionality (i.e., large  $d$ ) of samples in real world applications, directly sampling  $z \sim p(z)$  by the Markov Chain Monte Carlo (MCMC) algorithm requires a large number of random-walks that can incur high computational cost. To alleviate this issue, we adopt an efficient sampling method from (Steinke & Ullman, 2015) that first samples  $R$  from  $\text{Gamma}(d + 1, \sigma)$  and then samples  $\mathbf{z}$  from  $[-R, R]^d$  uniformly. The complexity of this sampling algorithm is only  $O(d)$ .

Comparing Theorems 10 and 13, we can see that randomized smoothing via (6) can certify a region that has (almost) the same radius as that of Gaussian distribution in the  $\ell_2$ -normed case, due to similarity in their density functions and the robustness guarantees. In the following theorem we show that the magnitude of the noise added by (6) is much larger than that of Gaussian distribution in the  $\ell_2$ -normed case.

Theorem 15. For the distribution that can guarantee Theorem 13, the following theorem holds

$$
\mathbb {E} _ {z} [ \| z \| _ {\infty} ] = d \sigma . \tag {7}
$$

Note that compared with the Gaussian noise added in Theorem 10 which satisfies the condition of  $\mathbb{E}_z[\| z\|_\infty ] = O(\sigma \sqrt{\log d})$ , the expected  $\ell_{\infty}$ -norm of the distribution in (6) is proportional to the dimensionality  $d$  of the data, which is quite large. This means that for any image data, at least one pixel will be perturbed by the magnitude of  $d\sigma$ , which will completely ruin the accuracy of the classification network. However, if we want the noise to have a magnitude of  $O(1)$ ,  $\sigma$  needs to be  $O(1 / d)$ , and so does the robust radius.

Theorem 15 is a somewhat negative result for randomized smoothing using distribution (6) to certify the  $\ell_{\infty}$ -normed robustness. Thus, an immediate question is whether exponential mechanism is the right choice to certify the  $\ell_{\infty}$ -normed robustness. The following theorem shows that for any  $(r, D_{\infty}, \| \cdot \|_{\infty}, \frac{r}{\sigma})$ -robust randomized smoothing mechanism, the expected  $\ell_{\infty}$ -norm of the added noise is lower bounded by  $\Omega(d\sigma)$ . Thus, combining the following theorem with Theorem 15, we can conclude that the exponential mechanism is actually an optimal choice to certify  $D_{\infty}$  robustness.

Theorem 16. For any  $(2r, D_{\infty}, \| \cdot \|_{\infty}, \frac{\epsilon}{2})$ -robust mechanism  $\mathcal{A}(x) = x + z: [0, r]^d \mapsto [0, r]^d$  such that

$$
\mathbb {E} [ \| z \| _ {\infty} ] = \mathbb {E} _ {\mathcal {A}} \| \mathcal {A} (x) - x \| _ {\infty} \leq \alpha , \forall x \in [ 0, r ] ^ {d},
$$

it must be true that  $\alpha \geq \Omega \left(\frac{rd}{\epsilon}\right)$ .

From Theorem 16 we can see that, for any  $(\cdot ,D_{\infty},\| \cdot \|_{\infty},\frac{\epsilon}{2})$  -robust randomized smoothing mechanism, if we fix the expectation of the  $\ell_{\infty}$  -norm of the added noise in the exponential mechanism as  $\alpha$ , the largest  $\ell_{\infty}$  radius that can be certified is upper bounded by  $O(\alpha \epsilon /d)$ . Compared with the  $\ell_{2}$ -normed case in Theorem 11, we can see that there is an additional factor of  $O(1 / d)$ , which makes it unscalable to high-dimensional data. Equivalently, if we want the same radius to be certified as in the Theorem 10, the expected  $\ell_{\infty}$  -norm of the added noise needs to be at least  $\Omega (\frac{rd}{\epsilon})$ , which will be too large for any image data.

The less than ideal lower bound in Theorem 16 is for  $D_{\infty}$ -robustness. Since  $D_{MR}$ -robustness is more relaxed than  $D_{\infty}$ -robustness, a natural question is thus to determine whether the lower bound can be improved by switching to  $D_{MR}$ -robustness. Unfortunately, the following theorem shows that a similar phenomenon still holds for  $D_{MR}$ -robustness.

Theorem 17. For any  $(2r, D_{MR}, \| \cdot \|_{\infty}, \frac{\epsilon}{2})$ -robust mechanism  $\mathcal{A}(x) = x + z : [0, r]^d \mapsto [0, r]^d$  such that

$$
\mathbb {E} [ \| z \| _ {\infty} ] = \mathbb {E} _ {\mathcal {A}} \| \mathcal {A} (x) - x \| _ {\infty} \leq \alpha , \forall x \in [ 0, r ] ^ {d},
$$

it must be true that  $\alpha \geq \Omega \left(\frac{r\sqrt{d}}{\sqrt{\epsilon}}\right)$ .

From Theorems 17 and 15 we can see that in the definition of  $(2r, D_{MR}, \| \cdot \|_{\infty})$ -robustness, adding noise according to (6) is not near optimal. The following theorem shows that in this case, Gaussian mechanism is actually a near optimal choice.

Theorem 18. Let  $r, \epsilon > 0$  be some fixed number and  $\mathcal{A}(x) = x + z$  with  $z \sim \mathcal{N}(0, \frac{dr^2}{2\epsilon})$ . Then,  $\mathcal{A}(\cdot)$  is  $(r, D_{MR}, \| \cdot \|_{\infty}, \epsilon)$ -robust. Moreover,  $\mathbb{E}[\| z \|_{\infty}] = \mathbb{E}_{\mathcal{A}} \| \mathcal{A}(x) - x \|_{\infty}$  is upper bounded by  $O\left(\frac{r\sqrt{d\log d}}{\sqrt{\epsilon}}\right)$ .

From Theorem 17 and 18, we can conclude that for all randomized smoothing mechanisms that are  $(\cdot ,0,D_{MR},\| \cdot \|_{\infty},\frac{\epsilon}{2})$  -robust, if the expected  $\ell_{\infty}$  -norm of the added noise is fixed to be  $\alpha$ , the largest radius that can be certified is upper bounded by  $O(\frac{\sqrt{\epsilon}\alpha}{\sqrt{d}})$ , and the largest radius that can be certified by Gaussian mechanism is  $O(1 / \sqrt{d\log d})$  (and  $\sigma$  is  $\Omega (\frac{\alpha}{\sqrt{\log d}})$ ). If  $\alpha$  and  $\epsilon$  are both set

![](images/7d2bbfa6e3981ed731f25c246c6b6e1a813467067b845ab2587febdfb1e85e1d.jpg)  
Figure 1: Certifying  $D_{MR}$  robustness in  $\ell_2$  norm on CIFAR-10: vary the Gaussian noise used in the training process and fix the  $\sigma$  of the Gaussian mechanism as  $\sigma = 0.5$ .  $c = 1$  (left) and  $c = 3$  (right)

![](images/cd7f18255f01f58245520b069c64190eb133f1b667fa27a1346d6331ad1032c1.jpg)

to be  $O(1)$ , the largest radius that can be certified using Gaussian mechanism to achieve  $D_{MR}$ -robustness is greater than the largest radius that can be certified to achieve  $D_{\infty}$ -robustness by at least a factor of  $O(\sqrt{d / \log d})$ . This is reasonable since the definition of  $D_{MR}$ -robustness is more relaxed. Obviously, there is some trade-off between the rigorousness of the notion of robustness and the largest certified robust radius, i.e., when the robustness is relaxed, the largest certified radius increases. We will investigate this trade-off more in the future research.

# 4 EXPERIMENTS

# 4.1 DATASETS AND MODELS

The performance of our framework is verified on two widely-used datasets, i.e., CIFAR10 and ImageNet*. Following Cohen et al. (2019), we use a 110-layer residual network and the classical ResNet-50 as the base models for CIFAR10 and ImageNet respectively. Note that it may be difficult for the models to classify noisy images without seeing any noisy samples in the training stage. Thus, we train all the models by adding appropriate Gaussian noise on the training images. The certified accuracy for radius  $R$  is defined as the fraction of the test set whose certified radii are larger than  $R^{\dagger}$ . The value of  $\epsilon$  in all our derived certified radii can be calculated by  $p_a$  (or  $p_a$  and  $p_b$ ) as shown in the proof of Corollary 11. It is also worth noting that we do not compare our results with (Cohen et al., 2019) in the experiments because our framework and (Cohen et al., 2019) endow robustness with different definitions. Moreover, our work does not aim at improving the tightness of the guarantee on the  $\ell_2$ -normed robustness but aims at presenting a general and self-contained framework to study some remaining issues, such as the optimality of the Gaussian mechanism, and the specific mechanisms to certify the  $\ell_{\infty}$ -normed robustness.

# 4.2 EMPIRICAL RESULTS

Certifying the  $\ell_2$ -normed Robustness To certify the  $\ell_2$ -normed Robustness, as we explained in previous section, Gaussian mechanism is a near optimal option. Thus, we mainly evaluate the performance of Gaussian mechanism in our framework. We first fix the value of  $\sigma$  in Gaussian mechanism and show the certified accuracy of the classifiers trained by varied Gaussian noises in Figure 1. As shown in Figure 1, using  $\sigma = 0.50$  Gaussian noise to train the classifier is a good setting here. So in Figure 2, we evaluate the Gaussian mechanism with different  $\sigma$  values on the classifier trained by  $\sigma = 0.50$  Gaussian noise. Overall, on CIFAR-10, our framework can certify approximately  $20\%$  accuracy under  $\ell_2 = 1.0$  perturbation $^{\ddagger}$ . We also show the results on ImageNet by Figures 4 and 5 in Appendix C.

Certifying the  $\ell_{\infty}$ -normed Robustness To certify the  $\ell_{\infty}$ -normed robustness, we evaluate the performance of the Exponential mechanism in the definition of  $D_{\infty}$ -robustness and the Gaussian

![](images/9224949c3051c71d0299ac647643ada2fa52d4cffeed1f60f709e55b32f02cb8.jpg)  
Figure 2: Certifying  $D_{MR}$  robustness in  $\ell_2$  norm on CIFAR-10: vary the  $\sigma$  in the Gaussian mechanism and fix  $\sigma$  of the training noise as  $\sigma = 0.50$ .  $c = 1$  (left) and  $c = 3$  (right)

![](images/0aa64fa93c9234b6addd7e55b2e2fe5b76afae1039c6e7840d552162f904d703.jpg)

![](images/23db535788853a32dbaa9445ba9edabff987260cfa273f1140bc25b85015d66e.jpg)  
Figure 3: Certifying  $D_{\infty}$  robustness and  $D_{MR}$  robustness in  $\ell_{\infty}$  norm on CIFAR-10: vary the  $\sigma$  in the Exponential mechanism (left) vary the  $\sigma$  in the Gaussian mechanism (right). The classifier is trained with  $\sigma = 0.50$  Gaussian noise.

![](images/3b9764b256ae78516ab848be06d04a951d40a28e968bb45983e4ac50a2a01f6f.jpg)

mechanism in the definition of  $D_{MR}$ -robustness. As shown in Figure 3, the  $\ell_{\infty}$  radii that can be certified by Gaussian mechanism are about  $10 \sim 20$  times (i.e.,  $O(\sqrt{d / \log d})$  with  $d = 3072$  as shown in our theories) larger than the  $\ell_{\infty}$  radii certified by the exponential mechanism. On ImageNet, as shown in Figure 6 in Appendix C, the robust radii are less than  $1 / 255$  (due to scaling in  $O(1 / d)$  or  $O(1 / \sqrt{d \log d})$ ), indicating that certifying the  $\ell_{\infty}$ -normed robustness by randomized smoothing may not be applicable to high-dimensional data.

# 5 CONCLUSION

In this paper, we present a general framework for certifying two types of robustness ( $D_{\infty}$  and  $D_{MR}$ -robustness) in the  $\ell_2$  and  $\ell_{\infty}$  norms by randomized smoothing. Under our framework, we first give the answers to the remaining questions in the previous studies on randomized smoothing-based certifiable defenses, i.e., the optimality of Gaussian mechanism and the possibility to certify the  $\ell_{\infty}$ -normed robustness. Specifically, we demonstrate that (i) Gaussian mechanism is a near optimal option to certify  $D_{MR}$ -robustness in  $\ell_2$  norm by giving a lower bound on all  $D_{MR}$ -robust mechanisms, with certified radii scaling in  $O(1)$ ; (ii) an exponential mechanism is the optimal choice for certifying  $D_{\infty}$ -robustness in  $\ell_{\infty}$  norm, with certified radii scaling in  $O(1 / d)$ ; (iii) Gaussian mechanism is a near optimal option to certify  $D_{MR}$ -robustness in  $\ell_{\infty}$  norm, with certified radii scaling in  $O(1 / \sqrt{d\log d})$ ; (iv) the largest  $\ell_{\infty}$  radius that can be certified by randomized smoothing in our framework is upper bounded by  $O(1 / \sqrt{d})$ , indicating that randomized smoothing may not be scalable to high-dimensional data in terms of certifying the  $\ell_{\infty}$ -normed robustness.

# REFERENCES

Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.  
Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Theory of Cryptography Conference, pp. 635-658. Springer, 2016.  
Mark Bun, Jonathan Ullman, and Salil Vadhan. Fingerprinting codes and the price of approximate differential privacy. SIAM Journal on Computing, 47(5):1888-1938, 2018.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
Jeremy M Cohen, Elan Rosenfeld, and J Zico Kolter. Certified adversarial robustness via randomized smoothing. arXiv preprint arXiv:1902.02918, 2019.  
Dimitrios Diochnos, Saeed Mahloujifar, and Mohammad Mahmoody. Adversarial risk and robustness: General definitions and implications for the uniform distribution. In Advances in Neural Information Processing Systems, pp. 10359-10368, 2018.  
Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy A Mann, and Pushmeet Kohli. A dual approach to scalable verification of deep networks. In UAI, pp. 550-559, 2018.  
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pp. 265-284. Springer, 2006.  
Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3-4):211-407, 2014.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Timothy Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. arXiv preprint arXiv:1810.12715, 2018.  
Moritz Hardt and Kunal Talwar. On the geometry of differential privacy. In Proceedings of the forty-second ACM symposium on Theory of computing, pp. 705-714. ACM, 2010.  
Warren He, James Wei, Xinyun Chen, Nicholas Carlini, and Dawn Song. Adversarial example defense: Ensembles of weak defenses are not strong. In 11th USENIX Workshop on Offensive Technologies (WOOT 17), 2017.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Mathias Lecuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and Suman Jana. Certified robustness to adversarial examples with differential privacy. arXiv preprint arXiv:1802.03471, 2018.  
Bai Li, Changyou Chen, Wenlin Wang, and Lawrence Carin. Second-order adversarial attack and certifiable robustness. arXiv preprint arXiv:1809.03113, 2018.  
Matthew Mirman, Timon Gehr, and Martin Vechev. Differentiable abstract interpretation for provably robust neural networks. In International Conference on Machine Learning, pp. 3575-3583, 2018.  
Francesco Orabona and David Pál. Optimal non-asymptotic lower bound on the minimax regret of learning with expert advice. arXiv preprint arXiv:1511.02176, 2015.

Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. arXiv preprint arXiv:1801.09344, 2018.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484-489, 2016.  
Thomas Steinke and Jonathan Ullman. Between pure and approximate differential privacy. arXiv preprint arXiv:1501.06095, 2015.  
Thomas Steinke and Jonathan Ullman. Between pure and approximate differential privacy. Journal of Privacy and Confidentiality, 7(2), 2016.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199, 2013.  
Jonathan Uesato, Brendan ODonoghue, Pushmeet Kohli, and Aaron Oord. Adversarial risk and the dangers of evaluating against weak attacks. In International Conference on Machine Learning, pp. 5032-5041, 2018.  
Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge University Press, 2018.  
Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, and Suman Jana. Efficient formal safety analysis of neural networks. In Advances in Neural Information Processing Systems, pp. 6367-6377, 2018.  
Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In International Conference on Machine Learning, pp. 5283-5292, 2018.
