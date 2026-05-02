# GRADIENT PERTURBATION IS UNDERRATED FOR DIFFERENTIALLY PRIVATE CONVEX OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Gradient perturbation, widely used for differentially private optimization, injects noise at every iterative update to guarantee differential privacy. Previous work first determines the noise level that can satisfy the privacy requirement and then analyzes the utility of noisy gradient updates as in non-private case. In this paper, we explore how the privacy noise affects the optimization property. We show that for differentially private convex optimization, the utility guarantee of both DP-GD and DP-SGD is determined by an expected curvature rather than the minimum curvature. The expected curvature represents the average curvature over the optimization path, which is usually much larger than the minimum curvature and hence can help us achieve a significantly improved utility guarantee. By using the expected curvature, our theory justifies the advantage of gradient perturbation over other perturbation methods and closes the gap between theory and practice. Extensive experiments on real world datasets corroborate our theoretical findings.

# 1 INTRODUCTION

Machine learning has become a powerful tool for many practical applications. The training process often needs access to some private dataset, e.g., applications in financial and medical fields. Recent work has shown that the model learned from training data may leak unintended information of individual records (Fredrikson et al., 2015; Wu et al., 2016; Shokri et al., 2017; Hitaj et al., 2017). It is known that Differential privacy  $(DP)$  (Dwork et al., 2006a,b) is a golden standard for privacy preserving data analysis. It provides provable privacy guarantee by ensuring the influence of any individual record is negligible. It has been deployed into real world applications by large-scale corporations and U.S. Census Bureau (Erlingsson et al., 2014; McMillan, 2016; Abowd, 2016; Ding et al., 2017).

We study the fundamental problem when differential privacy meets machine learning: the differentially private empirical risk minimization (DP-ERM) problem (Chaudhuri & Monteoni, 2009; Chaudhuri et al., 2011; Kifer et al., 2012; Bassily et al., 2014; Talwar et al., 2015; Wu et al., 2017; Zhang et al., 2017; Wang et al., 2017; Smith et al., 2017; Jayaraman et al., 2018; Feldman et al., 2018; Iyengar et al., 2019). DP-ERM minimizes the empirical risk while guaranteeing that the output of learning algorithm is differentially private with respect to the training data. Such privacy guarantee provides strong protection against potential adversaries (Hitaj et al., 2017; Rahman et al., 2018). In order to guarantee privacy, it is necessary to introduce randomness to the algorithm. There are usually three ways to introduce randomness according to the time of adding noise: output perturbation, objective perturbation and gradient perturbation.

Output perturbation (Wu et al., 2017; Zhang et al., 2017) first runs the learning algorithm the same as in the non-private case then adds noise to the output parameter. Objective perturbation (Chaudhuri et al., 2011; Kifer et al., 2012; Iyengar et al., 2019) perturbs the objective (i.e., the empirical loss) then release the minimizer of the perturbed objective. Gradient perturbation (Song et al., 2013; Bassily et al., 2014; Abadi et al., 2016; Wang et al., 2017; Lee & Kifer, 2018; Jayaraman et al., 2018) perturbs each intermediate update. If each

update is differentially private, the composition theorem of differential privacy ensures the whole learning procedure is differentially private.

Gradient perturbation comes with several advantages over output/objective perturbations. Firstly, gradient perturbation does not require strong assumption on the objective because it only needs to bound the sensitivity of gradient update rather than the whole learning process. Secondly, gradient perturbation can release the noisy gradient at each iteration without damaging the privacy guarantee as differential privacy is immune to post processing (Dwork et al., 2014). Thus, it is a more favorable choice for certain applications such as distributed optimization (Rajkumar & Agarwal, 2012; Agarwal et al., 2018; Jayaraman et al., 2018). At last, gradient perturbation often achieves better empirical utility than output/objective perturbations for DP-ERM.

However, the existing theoretical utility guarantee for gradient perturbation is the same as or strictly inferior to that of other perturbation methods as shown in Table 1. This motivates us to ask

"What is wrong with the theory for gradient perturbation? Can we justify the empirical advantage of gradient perturbation theoretically?"

We revisit the analysis for gradient perturbation approach. Previous work (Bassily et al., 2014; Wang et al., 2017; Jayaraman et al., 2018) derive the utility guarantee of gradient perturbation via two steps. They first determine the noise variance at each step that meets the privacy requirement and then derive the utility guarantee by using the convergence analysis the same as in non-private case. However, the noise to guarantee privacy naturally affects the optimization procedure, but previous approach does not exploit the interaction between privacy noise and optimization of gradient perturbation.

In this paper, we utilize the fact that the privacy noise affects the optimization procedure and establish new and much tighter utility guarantees for gradient perturbation approaches. Our contribution can be summarized as follows.

- We introduce an expected curvature that can characterize the optimization property accurately when there is perturbation noise at each gradient update.  
- We establish the utility guarantees for DP-GD for both convex and strongly convex objectives based on the expected curvature rather than the usual minimum curvature.  
- We also establish the utility guarantees for DP-SGD for both convex and strongly convex objectives based on the expected curvature. To the best of our knowledge, this is the first work to remove the dependency on minimum curvature for DP-ERM algorithms.

In DP-ERM literature, there is a gap between the utility guarantee of non-strongly convex objectives and that of strongly convex objectives. However, by using the expected curvature, we show that some of the non-strongly convex objectives can achieve the same order of utility guarantee as the strongly convex objectives, matching the empirical observation. This is because the expected curvature could be relatively large even for non-strongly convex objectives.

As we mentioned earlier, prior to our work, there is a mismatch between theoretical guarantee and empirical observation of gradient perturbation approach compared with other two perturbation approaches. Our result theoretically justifies the advantage of gradient perturbation and close the mismatch.

# 1.1 PAPER ORGANIZATION

The rest of this paper is organized as follows. Section 2 introduces notations and the DP-ERM task. In Sections 3, we first introduce the expected curvature and establish the utility guarantee of both DP-GD and DP-SGD based on such expected curvature. Then we give some discussion on three perturbation approaches. We conduct extensive experiments in Section 4. Finally, we conclude in Section 5.

Table 1: Expected excess empirical risk bounds under  $(\epsilon, \delta)$ -DP, where  $n$  and  $p$  are the number of samples and the number of parameters, respectively, and  $\beta, \mu$  and  $\nu$  are the smooth coefficient, the strongly convex coefficient and the expected curvature, respectively, and  $\nu \geq \mu$  (see Section 3.1). We note that  $\mu = 0$  denotes the convex but not strongly convex objective. The Lipschitz constant  $L$  is assumed to be 1. We omit  $\log(1/\delta)$  for simplicity.  

<table><tr><td>Authors</td><td>Perturbation</td><td>Algorithm</td><td>Utility (μ = 0)</td><td>Utility (μ &gt; 0)</td></tr><tr><td>Chaudhuri et al. (2011)</td><td>Objective</td><td>N/A</td><td>O(√p/nε)</td><td>O(p/μn2ε2)</td></tr><tr><td>Zhang et al. (2017)</td><td>Output</td><td>GD</td><td>O((√βp/nε)2/3)</td><td>O(βp/μ2n2ε2)</td></tr><tr><td>Bassily et al. (2014)</td><td>Gradient</td><td>SGD</td><td>O(√p log3/2(n)/nε)</td><td>O(p log2(n)/μn2ε2)</td></tr><tr><td>Jayaraman et al. (2018)</td><td>Gradient</td><td>GD</td><td>N/A</td><td>O(βp log2(n)/μ2n2ε2)</td></tr><tr><td>Ours</td><td>Gradient</td><td>GD</td><td>O(√p/nε ∧ βp log(n)/ν2n2ε2)</td><td>O(βp log(n)/ν2n2ε2)</td></tr><tr><td>Ours</td><td>Gradient</td><td>SGD</td><td>O(√p log(n)/nε ∧ p log(n)/νn2ε2)</td><td>O(p log(n)/νn2ε2)</td></tr></table>

# 2 PRELIMINARY

We introduce notations and definitions in this section. Given dataset  $D = \{d_{1},\ldots ,d_{n}\}$ , the objective function  $F(\pmb {x};D)$  is defined as  $F(\pmb {x};D)\triangleq \frac{1}{n}\sum_{i = 1}^{n}f(\pmb {x};d_i)$ , where  $f(\pmb {x};d_i):\mathbb{R}^p\to \mathbb{R}$  is the loss of model  $\pmb {x}\in \mathbb{R}^p$  for the record  $d_{i}$ .

For simplicity, we use  $F(\pmb{x})$  to denote  $F(\pmb{x};D)$ . We use  $\| \pmb{v} \|$  to denote the  $l_{2}$  norm of a vector  $\pmb{v}$ . We use  $\mathcal{X}_f^* = \arg \min_{\pmb{x} \in \mathbb{R}^p} f(\pmb{x})$  to denote the set of optimal solutions of  $f(\pmb{x})$ . Throughout this paper, we assume  $\mathcal{X}_f^*$  non-empty.

Definition 1 (Objective properties). For any  $\pmb{x}, \pmb{y} \in \mathbb{R}^p$ , a function  $f: \mathbb{R}^p \to \mathbb{R}$

- is  $L$ -lipschitz if  $|f(\pmb{x}) - f(\pmb{y})| \leq L\|\pmb{x} - \pmb{y}\|$ .  
is  $\beta$ -smooth if  $f(\pmb{y}) \leq f(\pmb{x}) + \langle \nabla f(\pmb{x}), \pmb{y} - \pmb{x} \rangle + \frac{\beta}{2} \| \pmb{y} - \pmb{x} \|^2$ .  
- is convex if  $\langle \nabla f(\pmb{x}) - \nabla f(\pmb{y}), \pmb{x} - \pmb{y} \rangle \geq 0$ .  
- is  $\mu$ -strongly convex (or  $\mu$ -SC) if  $\langle \nabla f(\pmb{x}) - \nabla f(\pmb{y}), \pmb{x} - \pmb{y} \rangle \geq \mu \| \pmb{x} - \pmb{y} \|^2$ .

The strong convexity coefficient  $\mu$  is the lower bound of the minimum curvature of function  $f$  over the domain.

We say that two datasets  $D, D'$  are neighboring datasets (denoted as  $D \sim D'$ ) if  $D$  can be obtained by arbitrarily modifying one record in  $D'$  (or vice versa). In this paper we consider  $(\epsilon, \delta)$ -differential privacy as follows.

Definition 2  $(\epsilon, \delta)$ -DP (Dwork et al., 2006a,b)). A randomized mechanism  $\mathcal{M}: D \to \mathcal{R}$  guarantees  $(\epsilon, \delta)$ -differential privacy if for any two neighboring input datasets  $D, D'$  and for any subset of outputs  $S \subseteq \mathcal{R}$  it holds that  $Pr[\mathcal{M}(D) \in S] \leq e^{\epsilon} Pr[\mathcal{M}(D') \in S] + \delta$ .

We note that  $\delta$  can be viewed as the probability that original  $\epsilon$ -DP fails and a meaningful setting requires  $\delta \ll \frac{1}{n}$ . By its definition, differential privacy controls the maximum influence that any individual record can produce. Smaller  $\epsilon, \delta$  implies less information leak but usually leads to worse utility. One can adjust  $\epsilon, \delta$  to trade off between privacy and utility.

DP-ERM requires the output  $\pmb{x}_{out} \in \mathbb{R}^p$  is differentially private with respect to the input dataset  $D$ . Let  $\pmb{x}_* \in \mathcal{X}_F^*$  be one of the optimal solutions of  $F(\pmb{x})$ , the utility of DP-ERM algorithm is measured by expected excess empirical risk:  $\mathbb{E}[F(\pmb{x}_{out}) - F(\pmb{x}_*)]$ , where the expectation is taken over the algorithm randomness.

# 3 MAIN RESULTS

In this section, we first define the expected curvature  $\nu$  and explain why it depends only on the average curvature. We then use such expected curvature to improve the analysis of both DP-SGD and DP-GD.

# 3.1 EXPECTED CURVATURE

In non-private setting, the analysis of convex optimization relies on the strongly convex coefficient  $\mu$ , which is the minimum curvature over the domain and can be extremely small for some common objectives. Previous work on DP-ERM uses the same analysis as in non-private case and therefore the resulting utility bounds rely on the minimum curvature. In our analysis, however, we avoid the dependency on the minimum curvature by exploiting how the privacy noise affects the optimization. With the perturbation noise, the expected curvature that the optimization path encounters is related to the average curvature instead of the minimum curvature. Definition 3 uses  $\nu$  to capture such average curvature with Gaussian noise. We use  $\boldsymbol{x}_{*} = \arg \min_{\boldsymbol{x}\in \mathcal{X}_{*}}\| \boldsymbol{x} - \boldsymbol{x}_{1}\|$  to denote the closest solution to the initial point.

Definition 3 (Expected curvature). A convex function  $F: \mathbb{R}^p \to \mathbb{R}$ , has expected curvature  $\nu$  with respect to noise  $\mathcal{N}(0, \sigma^2 I_p)$  if for any  $\boldsymbol{x} \in \mathbb{R}^p$  and  $\tilde{\boldsymbol{x}} = \boldsymbol{x} - \boldsymbol{z}$  where  $\boldsymbol{z} \sim \mathcal{N}(0, \sigma^2 I_p)$ , it holds that

$$
\mathbb {E} \left[ \left\langle \nabla F (\tilde {\boldsymbol {x}}), \tilde {\boldsymbol {x}} - \boldsymbol {x} _ {*} \right\rangle \right] \geq \nu \mathbb {E} \left[ \| \tilde {\boldsymbol {x}} - \boldsymbol {x} _ {*} \| ^ {2} \right], \tag {1}
$$

where the expectation is taken with respect to  $\mathbf{z}$ .

Claim 1. If  $F$  is  $\mu$ -strongly convex, we have  $\nu \geq \mu$ .

Proof. It can be verified that  $\nu = \mu$  always holds because of the strongly convex definition.

![](images/e96fe81178d1934687ed89730513803c1d8a6b4303e5769117b7043167285146.jpg)

In fact,  $\nu$  represents the average curvature and is much larger than  $\mu$ . We use  $\pmb{x}'$  to denote the transpose of  $\pmb{x}$ . Let  $\pmb{H}_{\pmb{x}} = \nabla^{2}F(\pmb{x})$  be the Hessian matrix evaluated at  $\pmb{x}$ . Using Taylor expansion to approximate the l.h.s. of Eq (1), we have

$$
\mathbb {E} [ \left\langle \nabla F (\boldsymbol {x}) - \boldsymbol {H} _ {\boldsymbol {x}} \boldsymbol {z}, \boldsymbol {x} - \boldsymbol {z} - \boldsymbol {x} _ {*} \right\rangle ] \geq \nu \mathbb {E} [ \| \boldsymbol {x} - \boldsymbol {z} - \boldsymbol {x} _ {*} \| ^ {2} ],
$$

$$
\langle \nabla F (\boldsymbol {x}), \boldsymbol {x} - \boldsymbol {x} _ {*} \rangle + \mathbb {E} [ \boldsymbol {z} ^ {\prime} H _ {\boldsymbol {x}} \boldsymbol {z} ] \geq \nu \mathbb {E} [ \| \boldsymbol {x} - \boldsymbol {z} - \boldsymbol {x} _ {*} \| ^ {2} ], \tag {2}
$$

$$
\left\langle \nabla F (\boldsymbol {x}), \boldsymbol {x} - \boldsymbol {x} _ {*} \right\rangle + \sigma^ {2} \operatorname {t r} \left(\boldsymbol {H} _ {\boldsymbol {x}}\right) \geq \nu \left\| \boldsymbol {x} - \boldsymbol {x} _ {*} \right\| ^ {2} + p \nu \sigma^ {2}.
$$

For positive semi-definite matrix  $H_{\pmb{x}}$ , we have  $\mathrm{tr}(H_{\pmb{x}})$  is the sum of the eigenvalues of  $H_{\pmb{x}}$ . Rearrange the resulting inequality shows Definition 3 holds for any  $\nu \lesssim \frac{\mathrm{tr}(H_{\pmb{x}})\sigma^2 + \mu\|\pmb{x} - \pmb{x}_{*}\|^2}{p\sigma^2 + \|\pmb{x} - \pmb{x}_{*}\|^2}$ .

For relatively large  $\sigma^2$ , this implies  $\nu \approx \frac{\mathrm{tr}(H_{\pmb{x}})}{p}$  that is the average curvature at  $\pmb{x}$ . Large variance is a reasonable setting because meaningful differential privacy guarantee requires non-trivial amount of noise.

The above analysis suggests that  $\nu$  can be independent of and much larger than  $\mu$ . This is indeed true for many convex objectives. Let us take the  $l_{2}$  regularized logistic regression as an example. The objective is strongly convex only due to the  $l_{2}$  regularizer. Thus, the minimum curvature (strongly convex coefficient) is the regularization coefficient  $\lambda$ . Sharmir et al. [1] shows the optimal choice of  $\lambda$  is  $\Theta(n^{-1/2})$  (Section 4.3 in [1]). In practice, typical choice of  $\lambda$  is even smaller and could be on the order of  $n^{-1}$ . Figure 1 compares the minimum and average curvatures of regularized logistic regression during the training process. The average curvature is basically unaffected by the regularization term  $\lambda$ . In contrast, the minimum curvature reaches  $\lambda$  in first few steps. Therefore removing the dependence on minimum curvature is a significant improvement.

Perturbation noise is necessary to attain  $\nu >\mu$ . We note that  $\nu = \mu$  when the training process does not involve perturbation noise (corresponding to  $\sigma = 0$  in Definition 3). For example, objective/output perturbation cannot utilize this expected curvature condition as no noise is injected in their training process. Therefore, among three existing perturbation methods, gradient perturbation is the only method can leverage such effect of noise.

![](images/57f42747edf1d79fa73c144a59a64329630cd8c35dda37b1719e6f488ee0fe6d.jpg)  
Figure 1: Curvatures of regularized logistic regression on Adult dataset. Cross symbol represents minimum curvature.

![](images/92e6dd359627a41badaa4a41847e16493e82b338ccf2974a7d93e88f46e369ea.jpg)  
Figure 2: Illustration of a generic loss function in the high dimensional setting  $(p > n$ , Figure 3 in Negahban et al. (2012)).

We note that  $\mu = 0$  does not necessarily lead to  $\nu = 0$ . A concrete example is given in Figure 2 (from Negahban et al. (2012)). It provides an illustration of the loss function in the high-dimensional  $(p > n)$  setting, i.e., the restricted strongly convex scenario: the loss is curved in certain directions but completely flat in others. The average curvature of such objective is always positive but the worst curvature is 0. Moreover, as shown in Figure 1, the average curvature of logistic regression is positive during the training procedure even the regularization term is 0. As we will show later,  $\nu$  being positive over the optimization path is sufficient for our optimization analysis.

# 3.2 UTILITY GUARANTEE OF DP-GD BASED ON EXPECTED CURVATURE

In this section we show that the expected curvature can be used to improve the utility bound of DP-GD (Algorithm 1).

Algorithm 1: Differentially Private Gradient Descent (DP-GD)  
Input: Privacy parameters  $\epsilon, \delta$ ; running steps  $T$ ; learning rate  $\eta$ . Loss function  $F(\pmb{x})$  with Lipschitz constant  $L$ .  
for  $t = 1$  to  $T$  do  
    Compute  $\pmb{g}_t = \nabla F(\pmb{x}_t)$ .  
    Update parameter  $\pmb{x}_{t + 1} = \pmb{x}_t - \eta_t(\pmb{g}_t + \pmb{z}_t)$ , where  $\pmb{z}_t \sim \mathcal{N}(0, \sigma_t^2 I_p)$ .  
end for

Algorithm 1 is  $(\epsilon, \delta)$ -DP if we set  $\sigma_t = \Theta \left( \frac{L \sqrt{T \log(1 / \delta)}}{n \epsilon} \right)$  (Jayaraman et al., 2018). Let  $x_1, \ldots, x_T$  be the training path and  $\nu = \min \{\nu_1, \ldots, \nu_T\}$  be the minimum expected curvature over the path. Now we present the utility guarantee of DP-GD for the case of  $\nu > 0$ .

Theorem 1 (Utility guarantee,  $\nu >0$ ). Suppose  $F$  is  $\beta$ -smooth. Set  $\eta \leq \frac{1}{\beta}$ ,  $T = \frac{2\log(n)}{\eta\nu}$  and  $\sigma_t = \Theta\left(L\sqrt{T\log(1 / \delta)} /n\epsilon\right)$ , we have

$$
\mathbb {E} \left[ F \left(\boldsymbol {x} _ {T + 1}\right) - F \left(\boldsymbol {x} _ {*}\right) \right] = \mathcal {O} \left(\frac {\beta p \log (n) L ^ {2} \log (1 / \delta)}{\nu^ {2} n ^ {2} \epsilon^ {2}}\right).
$$

Proof. All proofs in this paper are relegated to Appendix A.

![](images/d313d4afe6bf0848a481fdab96d4cd3bafb4e2026b6363844761e53828aedea5.jpg)

Remark 1. Theorem 3 only depends on the expected curvature over the training path  $\nu$ .

The expectation is taken over the algorithm randomness if without specification. Theorem 1 significantly improves the original analysis of DP-GD because of our arguments in Section 3.1. If  $\nu = 0$ , then the curvatures are flatten in all directions. One example is the linear function, which is used by Bassily et al. (2014) to derive their utility lower bound. Such simple function may not be commonly used as loss function in practice. Nonetheless, we give the utility guarantee for the case of  $\nu = 0$  in Theorem 2.

Theorem 2 (Utility guarantee,  $\nu = 0$ ). Suppose  $F$  is  $\beta$ -smooth. Set  $\eta = \frac{1}{\beta}$ ,  $T = \frac{n\beta\epsilon}{\sqrt{p}}$  and  $\sigma_t = \Theta\left(L\sqrt{T\log(1/\delta)}/n\epsilon\right)$ . Let  $\bar{\mathbf{x}} = \frac{1}{T}\sum_{i=1}^{T}\mathbf{x}_{i+1}$ , we have

$$
\mathbb {E} \left[ F \left(\bar {\boldsymbol {x}}\right) - F \left(\boldsymbol {x} _ {*}\right) \right] = \mathcal {O} \left(\frac {\sqrt {p} L ^ {2} \log (1 / \delta)}{n \epsilon}\right).
$$

We use parameter averaging to reduce the influence of perturbation noise because gradient update does not have strong contraction effect when  $\nu = 0$ .

# 3.3 UTI LTI Y G U A R A N T E E OF DP-SGD BASED ON EXPECTED CUR VATURE

Stochastic gradient descent has become one of the most popular optimization methods because of the cheap one-iteration cost. In this section we show that expected curvature can also improve the utility analysis for DP-SGD (Algorithm 2). We note that  $\nabla f(\pmb{x})$  represents an element from the subgradient set evaluated at  $\pmb{x}$  when the objective is not smooth. Before stating our theorem, we introduce the moments accountant technique (Lemma 1) that is essential to establish privacy guarantee.

Lemma 1 (Abadi et al. (2016)). There exist constants  $c_{1}$  and  $c_{2}$  so that given running steps  $T$ , for any  $\epsilon < c_{1}T / n^{2}$ , Algorithm 2 is  $(\epsilon, \delta)$ -differentially private for any  $\delta > 0$  if we choose  $\sigma \geq c_{2}\frac{\sqrt{T\log(1 / \delta)}}{n\epsilon}$ .

Algorithm 2: Differentially Private Stochastic Gradient Descent (DP-SGD)  
Input :Dataset  $D = \{d_1,\dots ,d_n\}$  . Individual loss function:  $f_{i}(\pmb {x}) = f(\pmb{x})$  Lipschitz constant  $L$  Number of iterations:  $T$  . Learning rate:  $\eta$  1 for  $t = 1$  to  $T$  do Sample  $i_t$  from [n] uniformly. Compute  $\pmb {g}_t = \nabla f_{i_t}(\pmb {x}_t)$  Update parameter  $\pmb{x}_{t + 1} = \pmb {x}_t - \eta_t(\pmb {g}_t + \pmb {z}_t)$  where  $\pmb {z}_t\sim \mathcal{N}\left(0,L^2\sigma^2 I_p\right)$  end

For the case of  $\nu > 0$ , Theorem 3 presents the utility guarantee of DP-SGD.

Theorem 3 (Utility guarantee,  $\nu >0$ ). Choose  $\sigma$  based on Lemma 1 to guarantee  $(\epsilon ,\delta)$ -DP. Set  $\eta_t = \frac{1}{\nu t}$  and  $T = n^{2}\epsilon^{2}$ , we have

$$
\mathbb {E} \left[ F \left(\boldsymbol {x} _ {T}\right) - F \left(\boldsymbol {x} _ {*}\right) \right] = \mathcal {O} \left(\frac {p L ^ {2} \log (n) \log (1 / \delta)}{n ^ {2} \epsilon^ {2} \nu}\right).
$$

Remark 2. Theorem 3 does not require smooth assumption.

Theorem 3 shows the utility guarantee of DP-SGD also depends on  $\nu$  rather than  $\mu$ . We set  $T = \Theta(n^2)$  following Bassily et al. (2014). We note that  $T = \Theta(n^2)$  is necessary even for non-private SGD to reach  $1/n^2$  precision. We next show for a relatively coarse precision, the running time can be reduced significantly.

Theorem 4. Choose  $\sigma$  based on Lemma 1 to guarantee  $(\epsilon, \delta)$ - $DP$ . Set  $\eta_t = \frac{1}{\nu t}$  and  $T = \frac{n\epsilon}{\sqrt{p}}$ . Suppose  $p < n^2$ , we have

$$
\mathbb {E} [ F (\pmb {x} _ {T}) - F (\pmb {x} _ {*}) ] = \mathcal {O} \left(\frac {\sqrt {p} L ^ {2} \log (n)}{n \epsilon \nu}\right).
$$

We note that the analysis of Bassily et al. (2014) yields  $\mathbb{E}[F(\pmb{x}_T) - F(\pmb{x}_*)] = \mathcal{O}\left(\frac{\sqrt{p}L^2\log^2(n)}{n\epsilon\mu}\right)$  if setting  $T = \frac{n\epsilon}{\sqrt{p}}$ , which still depends on the minimum curvature. Theorem 5 shows the utility for the case of  $\nu = 0$ .

Theorem 5 (Utility guarantee,  $\nu = 0$ ). Assume  $\| \pmb{x}_t\| \leq D$  for  $t\in [T]$ . Choose  $\sigma$  based on Lemma 1 to guarantee  $(\epsilon ,\delta)$ -DP. Let  $G = L\sqrt{1 + p\sigma^2}$ , set  $\eta_t = \frac{D}{G\sqrt{t}}$  and  $T = n^{2}\epsilon^{2}$ , we have

$$
\mathbb {E} [ F (\boldsymbol {x} _ {T}) - F (\boldsymbol {x} _ {*}) ] = \mathcal {O} \left(\frac {\sqrt {p \log (1 / \delta)} L \log (n)}{n \epsilon}\right).
$$

This utility guarantee can be derived from Theorem 2 in (Shamir & Zhang, 2013).

# 3.4 DISCUSSION ON THREE PERTURBATION APPROACHES.

In this section, we briefly discuss two other perturbation approaches and compare them to the gradient perturbation approach.

Output perturbation (Wu et al., 2017; Zhang et al., 2017) perturbs the learning algorithm after training. It adds noise to the resulting model of non-private learning process. The magnitude of perturbation noise is propositional to the maximum influence one record can cause on the learned model. Take the gradient descent algorithm as an example. At each step, the gradient of different records would diverge the two sets of parameters generated by neighboring datasets, the maximum distance expansion is related to the Lipschitz coefficient. At the same time, the gradient of the same records in two datasets would shrink the parameter distance because of the contraction effect of the gradient update. The contraction effect depends on the smooth and strongly convex coefficient. Smaller strongly convex coefficient leads to weaker contraction. The sensitivity of output perturbation algorithm is the upper bound on the largest possible final distance between two sets of parameters.

Objective perturbation (Chaudhuri et al., 2011; Kifer et al., 2012; Iyengar et al., 2019) perturbs the objective function before training. It requires the objective function to be strongly convex to guarantee the uniqueness of the solution. It first adds  $L_{2}$  regularization to obtain strong convexity if the original objective is not strongly convex. Then it perturbs the objective with a random linear term. The sensitivity of objective perturbation is the maximum change of the minimizer that one record can produce. Chaudhuri et al. (2011) and Kifer et al. (2012) use the largest and the smallest eigenvalue (i.e. the smooth and strongly convex coefficient) of the objective's Hessian matrix to upper bound such change.

In comparison, gradient perturbation is more flexible than output/objective perturbation. For example, to bound the sensitivity, gradient perturbation only requires Lipschitz coefficient which can be easily obtained by using the gradient clipping technique. However, both output and objective perturbation further need to compute the smooth coefficient, which is hard for some common objectives such as softmax regression.

More critically, output/objective perturbation cannot utilize the expected curvature condition because their training process does not contain perturbation noise. Moreover, they have to consider the worst performance of learning algorithm. That is because DP makes the worst case assumption on query function and output/objective perturbation treat the whole learning algorithm as a single query to private dataset. This explains why their utility guarantee depends on the worst curvature of the objective.

# 4 EXPERIMENT

In this section, we evaluate the performance of DP-GD and DP-SGD on multiple real world datasets. We use the benchmark datasets provided by Iyengar et al. (2019). Objective functions are logistic regression and softmax regression for binary and multi-class datasets, respectively.

Datasets. The benchmark datasets include two multi-class datasets (MNIST, Covertype) and five binary datasets, and three of them are high dimensional (Gisette, Real-sim, RCV1). Following Iyengar et al. (2019), we use  $80\%$  data for training and the rest for testing. Detailed description of datasets can be found in Appendix B

Implementation details. We track Rényi differentially privacy (RDP) (Mironov, 2017) and convert it to  $(\epsilon, \delta)$ -DP. Running step  $T$  is chosen from  $\{50, 200, 800\}$  for both DP-GD

Table 2: Algorithm validation accuracy (in %) on various kinds of real world datasets. Privacy parameter  $\epsilon$  is 0.1 for binary dataset and 1 for multi-classes datasets.  

<table><tr><td></td><td>KDDCup99</td><td>Adult</td><td>MNIST</td><td>Covertype</td><td>Gisette</td><td>Real-sim</td><td>RCV1</td></tr><tr><td>Non private</td><td>99.1</td><td>84.8</td><td>91.9</td><td>71.2</td><td>96.6</td><td>93.3</td><td>93.5</td></tr><tr><td>AMP1</td><td>97.5</td><td>79.3</td><td>71.9</td><td>64.3</td><td>62.8</td><td>73.1</td><td>64.5</td></tr><tr><td>Out-SGD</td><td>98.1</td><td>77.4</td><td>69.4</td><td>62.4</td><td>62.3</td><td>73.2</td><td>66.7</td></tr><tr><td>DP-SGD</td><td>98.7</td><td>80.4</td><td>87.5</td><td>67.7</td><td>63.0</td><td>73.8</td><td>70.4</td></tr><tr><td>DP-GD</td><td>98.7</td><td>80.9</td><td>88.6</td><td>66.2</td><td>67.3</td><td>76.1</td><td>74.9</td></tr></table>

![](images/1be74b4be672345a31ff4bc833801c9a9d88b5d8bf71bd490e614e4e0e2b1ab1.jpg)  
Figure 3: Algorithm validation accuracy (in %) with varying  $\epsilon$ . NP represents non-private baseline. Detailed description about evaluated datasets can be found in Table 3.

![](images/09bbcda8ff37176999f8cbc5e205ceaa12458c89e8f786997e7f9dccd22d95b6.jpg)

![](images/24230b9e8de583aa6f84fd7fa1f87e36dc7e74d4984f7efac66fa72ad0f327ea.jpg)

![](images/4469fa34420f9b3640a8bf668459d3626302bb7b99a3ea8ea9d88e78f7a41d80.jpg)

and DP-SGD. For DP-SGD, we use moments accountant to track the privacy loss and the sampling ratio is set as 0.1. The standard deviation of the added noise  $\sigma$  is set to be the smallest value such that the privacy budget is allowable to run desired steps. We ensure each loss function is Lipschitz by clipping individual gradient. The method in Goodfellow (2015) allows us to clip individual gradient efficiently. Clipping threshold is set as 1 (0.5 for high dimensional datasets because of the sparse gradient). For DP-GD, learning rate is chosen from  $\{0.1, 1.0, 5.0\}$  ( $\{0.2, 2.0, 10.0\}$  for high dimensional datasets). The learning rate of DP-SGD is twice as large as DP-GD and it is divided by 2 at the middle of training. Privacy parameter  $\delta$  is set as  $\frac{1}{n^2}$ . The  $l_2$  regularization coefficient is set as  $1 \times 10^{-4}$ . All reported numbers are averaged over 20 runs.

Baseline algorithms. The baseline algorithms include state-of-the-art objective and output perturbation algorithms. For objective perturbation, we use Approximate Minima Perturbation (AMP) (Iyengar et al., 2019). For output perturbation, we use the algorithm in Wu et al. (2017) (Output perturbation SGD). We adopt the implementation and hyperparameters in Iyengar et al. (2019) for both algorithms. For multi-class classification tasks, Wu et al. (2017) and Iyengar et al. (2019) divide the privacy budget evenly and train multiple binary classifiers because their algorithms need to compute smooth coefficient before training and therefore are not directly applicable to softmax regression.

Experiment results. The validation accuracy results for all evaluated algorithms with  $\epsilon = 0.1$  (1.0 for multi-class datasets) are presented in Table 2. We also plot the accuracy results with varying  $\epsilon$  in Figure 3. These results confirm our theory in Section 3: gradient perturbation achieves better performance than other perturbation methods as it leverages the average curvature.

# 5 CONCLUSION

In this paper, we show the privacy noise actually helps optimization analysis, which can be used to improve the utility guarantee of both DP-GD and DP-SGD. Our result theoretically justifies the empirical superiority of gradient perturbation over other methods and advance the state of the art utility guarantee of DP-ERM algorithms. Experiments on real world datasets corroborate our theoretical findings nicely. In the future, it is interesting to consider how to utilize the expected curvature condition to improve the utility guarantee of other gradient perturbation based algorithms.

# REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In ACM SIGSAC Conference on Computer and Communications Security, 2016.  
John M Abowd. The challenge of scientific reproducibility and privacy protection for statistical agencies. *Census Scientific Advisory Committee*, 2016.  
Naman Agarwal, Ananda Theertha Suresh, Felix Xinnan X Yu, Sanjiv Kumar, and Brendan McMahan. cpgd: Communication-efficient and differentially-private distributed sgd. In Advances in Neural Information Processing Systems, 2018.  
Raef Bassily, Adam Smith, and Abhradeep Thakurta. Differentially private empirical risk minimization: Efficient algorithms and tight error bounds. Annual Symposium on Foundations of Computer Science, 2014.  
Kamalika Chaudhuri and Claire Monteleoni. Privacy-preserving logistic regression. In Advances in Neural Information Processing Systems, 2009.  
Kamalika Chaudhuri, Claire Monteleoni, and Anand D Sarwate. Differentially private empirical risk minimization. Journal of Machine Learning Research, 2011.  
Bolin Ding, Janardhan Kulkarni, and Sergey Yekhanin. Collecting telemetry data privately. In Advances in Neural Information Processing Systems, 2017.  
Cynthia Dwork, Krishnamaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Annual International Conference on the Theory and Applications of Cryptographic Techniques, 2006a.  
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, 2006b.  
Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 2014.  
Ulfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: Randomized aggregatable privacy-preserving ordinal response. In Proceedings of the 2014 ACM SIGSAC conference on computer and communications security, 2014.  
Vitaly Feldman, Ilya Mironov, Kunal Talwar, and Abhradeep Thakurta. Privacy amplification by iteration. In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS), 2018.  
Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In ACM SIGSAC Conference on Computer and Communications Security, 2015.  
Ian Goodfellow. Efficient per-example gradient computations. arXiv preprint arXiv:1510.01799, 2015.  
Briland Hitaj, Giuseppe Ateniese, and Fernando Pérez-Cruz. Deep models under the gan: information leakage from collaborative deep learning. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, 2017.  
Roger Iyengar, Joseph P Near, Dawn Song, Om Thakkar, Abhradeep Thakurta, and Lun Wang. Towards practical differentially private convex optimization. In IEEE Symposium on Security and Privacy, 2019.  
Bargav Jayaraman, Lingxiao Wang, David Evans, and Quanquan Gu. Distributed learning without distress: Privacy-preserving empirical risk minimization. In Advances in Neural Information Processing Systems, 2018.  
Daniel Kifer, Adam Smith, and Abhradeep Thakurta. Private convex empirical risk minimization and high-dimensional regression. In Conference on Learning Theory, 2012.  
Jaewoo Lee and Daniel Kifer. Concentrated differentially private gradient descent with adaptive per-iteration privacy budget. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018.

Robert McMillan. Apple tries to peek at user habits without violating privacy. The Wall Street Journal, 2016.  
Ilya Mironov. Rényi differential privacy. In IEEE 30th Computer Security Foundations Symposium (CSF), 2017.  
Sahand N Negahban, Pradeep Ravikumar, Martin J Wainwright, Bin Yu, et al. A unified framework for high-dimensional analysis of  $m$ -estimators with decomposable regularizers. Statistical Science, 27(4):538-557, 2012.  
Md Atiqur Rahman, Tanzila Rahman, Robert Laganiere, Noman Mohammed, and Yang Wang. Membership inference attack against differentially private deep learning model. Transactions on Data Privacy, 2018.  
Arun Rajkumar and Shivani Agarwal. A differentially private stochastic gradient descent algorithm for multiparty classification. In Artificial Intelligence and Statistics, 2012.  
Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In International Conference on Machine Learning, 2013.  
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy (SP), 2017.  
Adam Smith, Abhradeep Thakurta, and Jalaj Upadhyay. Is interaction necessary for distributed private learning? In IEEE Symposium on Security and Privacy (SP). IEEE, 2017.  
Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE, 2013.  
Kunal Talwar, Abhradeep Guha Thakurta, and Li Zhang. Nearly optimal private lasso. In Advances in Neural Information Processing Systems, 2015.  
Di Wang, Minwei Ye, and Jinhui Xu. Differentially private empirical risk minimization revisited: Faster and more general. In Advances in Neural Information Processing Systems, 2017.  
Xi Wu, Matthew Fredrikson, Somesh Jha, and Jeffrey F Naughton. A methodology for formalizing model-inversion attacks. In 2016 IEEE 29th Computer Security Foundations Symposium (CSF), 2016.  
Xi Wu, Fengan Li, Arun Kumar, Kamalika Chaudhuri, Somesh Jha, and Jeffrey Naughton. Bolt-on differential privacy for scalable stochastic gradient descent-based analytics. In ACM International Conference on Management of Data, 2017.  
Jiaqi Zhang, Kai Zheng, Wenlong Mou, and Liwei Wang. Efficient private erm for smooth objectives. In International Joint Conference on Artificial Intelligence, 2017.
