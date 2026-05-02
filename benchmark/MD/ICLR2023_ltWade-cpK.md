# OPTIMAL ACTIVATION FUNCTIONS FOR THE RANDOM FEATURES REGRESSION MODEL

Anonymous authors

Paper under double-blind review

# ABSTRACT

The asymptotic mean squared test error and sensitivity of the Random Features Regression model (RFR) have been recently studied. We build on this work and identify in closed-form the family of Activation Functions (AFs) that minimize a combination of the test error and sensitivity of the RFR under different notions of functional parsimony. We find scenarios under which the optimal AFs are linear, saturated linear functions, or expressible in terms of Hermite polynomials. Finally, we show how using optimal AFs impacts well established properties of the RFR model, such as its double descent curve, and the dependency of its optimal regularization parameter on the observation noise level.

# 1 INTRODUCTION

For many neural network (NN) architectures, the test error does not monotonically increase as a model's complexity increases but can go down with the training error both at low and high complexity levels. This phenomenon, the double descent curve, defies intuition and has motivated new frameworks to explain it. Explanations have been advanced involving linear regression with random covariates (Belkin et al., 2020; Hastie et al., 2022), kernel regression (Belkin et al., 2019; Liang & Rakhlin, 2020), the neural tangent kernel model (Jacot et al., 2018), and the Random Features Regression (RFR) model (Mei & Montanari, 2022). These frameworks allow queries beyond the generalization power of NNs. For example, they have been used to study networks' robustness properties (Hassani & Javanmard, 2022; Tripuraneni et al., 2021).

One aspect within reach and unstudied to this day is finding optimal Activation Functions (AFs) for these models. It is known that AFs affect a network's approximation accuracy and efforts optimize AFs have been undertaken. Previous work has justified the choice of AFs empirically, e.g., Ramachandran et al. (2017), or provided numerical procedures to learn AF parameters, sometimes jointly with models' parameters, e.g. Unser (2019). See Rasamoelina et al. (2020) for commonly used AFs and Section 2 for how AFs have been previously derived.

We derive for the first time closed-form optimal AFs such that an explicit objective function involving the asymptotic test error and sensitivity of a model is minimized. Setting aside empirical and principled but numerical methods, all past principled and analytical approaches to design AFs focus on non accuracy related considerations, e.g. Milletaré et al. (2019). We focus on AFs for the RFR model and expand its understanding. We preview a few surprising conclusions extracted from our main results:

1. The optimal AF can be linear, in which case the RFR model is a linear model. For example, if no regularization is used for training, and for low complexity models, a linear AF is often preferred if we want to minimize test error. For high complexity models a non-linear AF is often better;  
2. A linear optimal AF can destroy the double descent curve behaviour and achieve small test error with much fewer samples than e.g. a ReLU;  
3. When, apart from the test error, the sensitivity of a model becomes important, optimal AFs that without sensitivity considerations were linear can become non-linear, and vice-versa;  
4. Using an optimal AF with an arbitrary regularization during training can lead to the same, or better, test error as using a non-optimal AF, e.g. ReLU, and optimal regularization.

# 1.1 PROBLEM SET UP

We consider the effect of AFs on finding an approximation  $f$  to a square-integrable function  $f_{d}$  on the  $d$ -dimensional sphere  $\mathbb{S}^{d - 1}(\sqrt{d})$ , the function  $f_{d}$  having been randomly generated. The approximation  $f$  is to be learnt from training data  $\mathcal{D} = \{\pmb{x}_i, y_i\}_{i=1}^n$  where  $\pmb{x}_i \in \mathbb{S}^{d - 1}(\sqrt{d})$ , the variables  $\{\pmb{x}_i\}_{i=1}^n$  are i.i.d. uniformly sampled from  $\mathbb{S}^{d - 1}(\sqrt{d})$ , and  $y_i = f_d(\pmb{x}_i) + \epsilon_i$ , where the noise variables  $\{\epsilon_i\}_{i=1}^n$  are i.i.d. with  $\mathbb{E}(\epsilon_i) = 0$ ,  $\mathbb{E}(\epsilon_i^2) = \tau^2$ , and  $\mathbb{E}(\epsilon_i^4) < \infty$ .

The approximation  $f$  is defined according to the RFR model. The RFR model can be viewed as a two-layer NN with random first-layer weights encoded by a matrix  $\Theta \in \mathbb{R}^{N\times d}$  with  $i$ th row  $\theta_{i}\in \mathbb{R}^{d}$  satisfying  $\| \pmb {\theta}_i\| = \sqrt{d}$ , with  $\{\pmb {\theta}_i\}$  i.i.d. uniform on  $\mathbb{S}^{d - 1}(\sqrt{d})$ , and with to-be-learnt second-layer weights encoded by a vector  $\pmb {a} = [a_i]_{i = 1}^N = \mathbb{R}^N$ . Unless specified otherwise, the norm  $\| \cdot \|$  denotes the Euclidean norm. The RFR model defines  $f_{\pmb{a},\Theta}:\mathbb{S}^{d - 1}(\sqrt{d})\mapsto \mathbb{R}$  such that

$$
f _ {\boldsymbol {a}, \Theta} (\boldsymbol {x}) = \sum_ {i = 1} ^ {N} a _ {i} \sigma \left(\left\langle \boldsymbol {\theta} _ {i}, \boldsymbol {x} \right\rangle / \sqrt {d}\right). \tag {1}
$$

where  $\sigma (\cdot)$  is the AF that is the target of our study and  $\langle x,y\rangle$  denotes the inner product between vectors  $x$  and  $y$ . When clear from the context, we write  $f_{a,\Theta}$  as  $f$ , omitting the model's parameters. The optimal weights  $a^{\star}$  are learnt using ridge regression with regularization parameter  $\lambda \geq 0$ , namely,

$$
\boldsymbol {a} ^ {\star} = \boldsymbol {a} ^ {\star} (\lambda , \mathcal {D}) = \arg \min  _ {\boldsymbol {a} \in \mathbb {R} ^ {N}} \left\{\frac {1}{n} \sum_ {j = 1} ^ {n} \left(y _ {j} - \sum_ {i = 1} ^ {N} a _ {i} \sigma \left(\left\langle \boldsymbol {\theta} _ {i}, \boldsymbol {x} _ {j} \right\rangle / \sqrt {d}\right)\right) ^ {2} + \frac {N \lambda}{d} \| \boldsymbol {a} \| ^ {2} \right\}. \tag {2}
$$

We will tackle this question: What is the simplest  $\sigma$  that leads to the best approximation of  $f_{d}$ ?

We quantify the simplicity of an AF  $\sigma$  with its norm in different functional spaces. Namely, either

$$
\left\| \sigma \right\| _ {1} \triangleq \mathbb {E} \left(\left| \sigma^ {\prime} (Z) \right|\right), \quad \text {o r} \quad (3) \quad \| \sigma \| _ {2} \triangleq \sqrt {\mathbb {E} \left(\left(\sigma^ {\prime} (Z)\right) ^ {2}\right)}, \tag {4}
$$

where  $\sigma'$  is the derivative of  $\sigma$  and the expectations are with respect to a normal random variable  $Z$  with zero mean and unit variance, i.e.  $Z \sim \mathcal{N}(0,1)$ . For a comment on these choices please read Appendix A. We quantify the quality with which  $f = f_{\pmb{a}^{\star},\Theta}$  approximates  $f_{d}$  via  $O$ , a linear combination of the mean squared error and the sensitivity of  $f$  to perturbations in its input (cf. Novak et al. (2018)). For  $\alpha \in [0,1]$ ,  $\pmb{x}$  uniform on  $\mathbb{S}^{d-1}(\sqrt{d})$ , we define

$$
O \triangleq (1 - \alpha) \mathcal {E} + \alpha S, \quad (5) \quad \text {w h e r e} \mathcal {E} \triangleq \mathbb {E} ((f (\boldsymbol {x}) - f _ {d} (\boldsymbol {x})) ^ {2}), \quad (6) \quad \text {a n d} S \triangleq \mathbb {E} (\| \nabla_ {x} f (\boldsymbol {x}) \| ^ {2}). \tag {7}
$$

Like in Mei & Montanari (2022); D'Amour et al. (2020), we operate in an asymptotic setting where  $n, d, N \to \infty$ , and have constant ratios between them, namely,  $N / d \to \psi_1$  and  $n / d \to \psi_2$ . In this asymptotic setting, it does not matter if in defining (6) and (7), in addition to taking the expectation with respect to the test data  $x$ , independently of  $\mathcal{D}$ , we also take expectations over  $\mathcal{D}$  and the random features in RFR. This is because when  $n, d, N \to \infty$  with the ratios defined above,  $\mathcal{E}$  and  $S$  will concentrate around their means (Mei & Montanari, 2022; D'Amour et al., 2020).

Mathematically, denoting by  $\| \sigma \|$  either (4) or (3), our goal is to study the solutions of the problem

$$
\min  _ {\sigma^ {\star}} \| \sigma^ {\star} \| \text {s u b j e c t} \sigma^ {\star} \in \arg \min  _ {\sigma} O (\sigma). \tag {8}
$$

Notice that the outer optimization only affects the selection of optimal AF in so far as the inner optimization does not uniquely define  $\sigma^{\star}$ , which, as we will later see, it does not.

# 2 RELATED WORK

First attempts to optimize AFs include Poli (1996), Weingaertner et al. (2002), and Khan et al. (2013), where genetic and evolutionary algorithms were used to learn how to numerically combine different AFs from a library into the same network. More recently, Ramachandran et al. (2017) used reinforcement learning to empirically discover AFs that minimize test accuracy. Their search was done over AFs that were a combination of basic units. This work produced the Swish AF. Similarly, Goyal et al. (2020) defined AFs as the weighted sum of a pre-defined basis and searched for optimal weights via training. Unser (2019) provided a theoretical foundation to simultaneously learn a NN's weights and continuous piecewise-linear AFs. They showed that learning in their framework is

compatible with learning in current existing deep-ReLU, parametric ReLU, APL (adaptive piecewise-linear) and MaxOut architectures. Tavakoli et al. (2021) parameterized continuous piece-wise linear AFs and numerically learnt their parameters to improve both accuracy and robustness to adversarial perturbations. They numerically compared the performance of their SPLASH framework with that of using ReLUs, leaky-ReLUs (Maas et al., 2013), PReLUs (He et al., 2015), tanh units, sigmoid units, ELUs (Clevert et al., 2015), maxout units (Goodfellow et al., 2013), Swish units, and APL units (Agostinelli et al., 2014). Similarly, Zhou et al. (2021) parameterized AFs as piece-wise linear units and learnt the AFs parameters to optimize different tasks. Banerjee et al. (2019) proposed an empirical method to learn variations of ReLUs. Bubeck et al. (2020) studied 2-layer NNs and gave a condition on the Lipschitz constant of a polynomial AF for the network to perfectly fit data. They related this condition to the model's parameter-size and robustness and numerically related the number of ReLUs in the model to its robustness.

Several papers proposed new AFs and empirically studied their performance without systematically tuning them. Milletaré et al. (2019) identified ReLU and Swish as naturally arising components of a statistical mechanics model. Rozsa & Boult (2019) introduced a "tent"-shaped AF that improves robustness without adversarial training, while not hurting the accuracy on non-adversarial examples. Zhou et al. (2020) proposed an AF called SRS that can overcome the non-zero mean, negative missing, and unbounded output in ReLUs. Their work was purely empirical. Wuraola & Patel (2018) developed the SQuared Natural Law AF. Nicolae (2018) proposed the Piece-wise Linear Unit AF.

The RFR model was introduced by Rahimi & Recht (2007a) as a way to project input data into a low dimensional random features space and it has since then been studied considerably. A great part of the literature has drawn connections between the expressive power of NNs and that of the RFR model, often via the study of Gaussian processes. For example, Williams (1996) did this in the context of shallow but infinitely wide NN and the works Garriga-Alonso et al. (2019); Novak et al. (2019); de G. Matthews et al. (2018); Hazan & Jaakkola (2015) did this for deep networks. Daniely et al. (2016); Daniely (2017) connected the RFR model to training a NN with gradient descent.

In addition to Mei & Montanari (2022); D'Amour et al. (2020), already discussed, other papers studied the approximation properties of the RFR model. Ghorbani et al. (2021) studied both the RFR model and the neural tangent kernel model and provided conditions under which these models can fit polynomials in the raw features up to a maximum degree. These conditions were provided under two regimes, when  $n \to \infty$  and  $N, d$  large but finite, or when  $N \to \infty$  and  $n, d$  large but finite. Their results hold under weak assumptions on the AFs. Tripuraneni et al. (2021) used the RFR model to compute how robust the test error is to distribution shifts between training and test data. This was done in a high-dimensional asymptotic limit when random features and training data are normal distributed. The derivations hold for a generic AF that satisfies some mild assumptions similar to the assumptions in this paper. Hassani & Javanmard (2022) characterized the role of overparametrization on the adversarial robustness for the RFR model under an asymptotic regime when learning a linear function with normal-distributed random weights and normal samples. Their AF was a shifted ReLU.

Finally, a few papers have studied the behavior of models similar to the RFR but within a different context. For example, Taheri et al. (2021) and Bean et al. (2013) seek to compute the optimal loss function under similar asymptotic regimes of large data sets.

# 3 BACKGROUND ON THE ASYMPTOTIC PROPERTIES OF THE RFR MODEL

Here we will review recently derived closed-form expressions for the asymptotic mean squared error and sensitivity of the RFR model, which are the starting point of our work. First, however, we explain the use-inspired reasons for our setup. Our assumptions are the same as, or very similar to, those of published theoretical papers, e.g. Jacot et al. (2018); Yang et al. (2021); Ghorbani et al. (2021); Mel & Pennington (2022), the last one presented at ICLR 22. Although theoretical progress cannot be made otherwise yet, this not diminish our contributions, which are novel and impactful.

1. Data on a sphere: Normalization of input data is a best practice when learning with NNs (Huang et al., 2020). Assuming that input data lives on a sphere is one type of normalization.  
2. Random features: The seminal work of Rahimi & Recht (2007a) showed the success of using random features on real datasets. For a recent review on their use see Cao et al. (2018).  
3. Asymptotic setting: Mei & Montanari (2022) empirically showed that the convergence to the asymptotic regime is relatively fast, even with just a few hundreds of dimensions. Most real world applications involve larger dimensions  $d$ , lots of data  $n$ , and lots of neurons  $N$ .

4. Shallow architecture: For a finite input dimension  $d$ , the RFR model can learn arbitrary functions as the number of features  $N$  grows large (Bach, 2017; Rahimi & Recht, 2007b; Ghorbani et al., 2021). Existing proof techniques make it very hard yet to extend our type of analysis to more than two layers or complex architectures. A few papers consider models with depth  $> 2$  but do not tackle our problem and have other heavy restrictions on the model, e.g. Pennington et al. (2018).  
5. Regularization: Using regularization during training to control the weights' magnitude is common. It can help convergence speed and generalization error (Goodfellow et al., 2016). For a review on different types of regularization for learning with NNs see Kukačka et al. (2017).

We make the following assumptions, which we assume hold in the theorems in this section.

Assumption 1. We assume that the  $AF\sigma$  is weakly differentiable with weak derivative  $\sigma'$ , it satisfies  $|\sigma(u)|, |\sigma'(u)| \leq c_0 e^{c_1 |u|} \forall u \in \mathbb{R}$  for some constants  $0 < c_0, c_1 < \infty$ , and that it also satisfies

$$
\mu_ {0} = \mathbb {E} \{\sigma (Z) \}, \quad \mu_ {1} = \mathbb {E} \{Z \sigma (Z) \}, \quad \mu_ {2} = \mathbb {E} \{\sigma (Z) ^ {2} \}, \quad \mu_ {\star} ^ {2} = \mu_ {2} - \mu_ {0} ^ {2} - \mu_ {1} ^ {2}, \quad \zeta = \mu_ {1} / \mu_ {\star}, \tag {9}
$$

for some  $\mu_0, \mu_1, \mu_2 \in \mathbb{R}$ , where the expectations are with respect to  $Z \sim \mathcal{N}(0, 1)$ .

Assumption 2. We assume that  $N = N(d)$  and  $n = n(d)$  such that the following limits exist in  $(0, \infty)$ :  $\lim_{d \to \infty} N(d) / d = \psi_1$  and  $\lim_{d \to \infty} n(d) / d = \psi_2$ .

Assumption 3. We assume that  $y_{i} = f_{d}(\pmb{x}_{i}) + \epsilon_{i}$ , where  $\{\epsilon_i\}_{i\leq n}\sim i.i.d.\mathbb{P}_{\epsilon}$  are independent of  $\{\pmb {x}_i\}_{i\leq n}$  with  $\mathbb{E}(\epsilon_1) = 0$ ,  $\mathbb{E}(\epsilon_1^2) = \tau^2$ ,  $\mathbb{E}(\epsilon_1^4) < \infty$ , expectations with respect to  $\{\epsilon_i\}$ . Furthermore,

$$
f _ {d} (\boldsymbol {x}) = \beta_ {d, 0} + \left\langle \beta_ {d, 1}, \boldsymbol {x} \right\rangle + f _ {d} ^ {\mathrm {N L}} (\boldsymbol {x}), \tag {10}
$$

where  $\beta_{d,0} \in \mathbb{R}$ ,  $\beta_{d,1} \in \mathbb{R}^d$  are deterministic with  $\lim_{d \to \infty} \beta_{d,0}^2 = F_0^2$ ,  $\lim_{d \to \infty} \| \beta_{d,1} \|_2^2 = F_1^2 > 0$ . The non-linear  $f_d^{\mathrm{NL}}$  is a centered Gaussian process indexed by  $\pmb{x} \in \mathbb{S}^{d-1}(\sqrt{d})$ , with covariance

$$
\mathbb {E} _ {f _ {d} ^ {\mathrm {N L}}} \left\{f _ {d} ^ {\mathrm {N L}} \left(\boldsymbol {x} _ {1}\right) f _ {d} ^ {\mathrm {N L}} \left(\boldsymbol {x} _ {2}\right) \right\} = \Sigma_ {d} \left(\left\langle \boldsymbol {x} _ {1}, \boldsymbol {x} _ {2} \right\rangle / d\right), \tag {11}
$$

where  $\Sigma_d(\cdot)$  satisfies  $\mathbb{E}_{\boldsymbol{x} \sim \mathrm{Unif}(\mathbb{S}^{d-1}(\sqrt{d}))} \{\Sigma_d(x_1 / \sqrt{d})\} = 0$ ,  $\mathbb{E}_{\boldsymbol{x} \sim \mathrm{Unif}(\mathbb{S}^{d-1}(\sqrt{d}))} \{\Sigma_d(x_1 / \sqrt{d})x_1\} = 0$ , where  $x_1$  is the 1st component of  $\boldsymbol{x}$ . We define the Signal to Noise Ratio (SNR)  $\rho$  by

$$
\rho = F _ {1} ^ {2} / \left(F _ {\star} ^ {2} + \tau^ {2}\right), \text {w h e r e} F _ {\star} ^ {2} \triangleq \lim  _ {d \rightarrow \infty} \Sigma_ {d} (1). \tag {12}
$$

Informally,  $\mu_{\star}$  quantifies how non-linear the AF is (cf. Lemma 4.1),  $\psi_{1}$  quantifies the complexity of the RFR model relative to the dimension  $d$ ,  $\psi_{2}$  quantifies the amount of data used for training relative to  $d$ ,  $\tau^{2}$  is the variance of the observation noise,  $F_{1}$  is the magnitude of the linear component of our target function  $f_{d}$ , which is controlled by  $\beta_{d,1}$ ,  $F_{\star}$  is the magnitude of the non-linear component  $f_{d}^{\mathrm{NL}}$  in the target function, and  $\rho$  is the ratio between the magnitude of the linear component and the magnitude of all of the sources of randomness in the noisy function  $f_{d} + \epsilon$ . Recall that all of our results will be derived in the asymptotic regime when  $d \to \infty$ .

Our contributions are divided into two parts, Section 4.1 and Section 4.2. The theorems' statements in Section 4.2 quickly get prohibitively complex as they are stated more generally, with lots of special cases having to be discussed. Hence, in Section 4.2 we display our analysis on the following three different important regimes:  $R_{1}$ : Ridgeless limit regime, when  $\lambda \to 0^{+}$ ;  $R_{2}$ : Highly overparameterized limit, when  $\psi_{1} \to \infty$ ;  $R_{3}$ : Large sample limit, when  $\psi_{2} \to \infty$ . Section 4.1's results are general and not restricted to these regimes. In the context of the RFR model, these regimes were introduced and discussed in Mei & Montanari (2022). For what follows we define  $\overline{\lambda} \triangleq \lambda / \mu_{\star}^2$ . Any " $\lim_{d \to \infty} X = Y$ " should be interpreted as  $X$  converging to  $Y$  in probability with respect to the training data  $\mathcal{D}$ , the random features  $\Theta$ , and the random target  $f_{d}$  as  $d \to \infty$ .

3.1 ASYMPTOTIC MEAN SQUARED TEST ERROR OF THE RFR MODEL

The following theorems are a specialization of a more general theorem, Theorem 12 Mei & Montanari (2022), which we include in the Appendix C for completeness.

Theorem 1 (Theorem 3 Mei & Montanari (2022)). The asymptotic test error (6) for regime  $R_{1}$  equals

$$
\mathcal {E} _ {R _ {1}} ^ {\infty} \equiv \lim  _ {\lambda \rightarrow 0 ^ {+}} \lim  _ {d \rightarrow \infty} \mathcal {E} = F _ {1} ^ {2} \mathcal {B} _ {\text {r l e s s}} (\zeta , \psi_ {1}, \psi_ {2}) + \left(\tau^ {2} + F _ {\star} ^ {2}\right) \mathcal {V} _ {\text {r l e s s}} (\zeta , \psi_ {1}, \psi_ {2}) + F _ {\star} ^ {2}, \tag {13}
$$

where  $\mathcal{B}_{\mathrm{rless}}(\zeta, \psi_1, \psi_2) \equiv \mathcal{E}_{1,\mathrm{rless}} / \mathcal{E}_{0,\mathrm{rless}}$ ,  $\mathcal{V}_{\mathrm{rless}}(\zeta, \psi_1, \psi_2) \equiv \mathcal{E}_{2,\mathrm{rless}} / \mathcal{E}_{0,\mathrm{rless}}$ , and the functions  $\mathcal{E}_{0,\mathrm{rless}}$ ,  $\mathcal{E}_{1,\mathrm{rless}}$  and  $\mathcal{E}_{2,\mathrm{rless}}$  are polynomials that are functions of  $\zeta^2$ ,  $\psi_1$ ,  $\psi_2$  and  $\chi$ , where  $\chi$  is a function of  $\psi \equiv \min\{\psi_1, \psi_2\}$  and  $\zeta^2$ . See Appendix B for details.

Remark 1. As a function of  $\psi_{1}$ ,  $\mathcal{E}_{R_1}^\infty$  has a discontinuity at  $\psi_{1} = \psi_{2}$  called the interpolation threshold. For  $\psi_{2}$  high enough, and for  $\psi_{1} < \psi_{2}$ ,  $\mathcal{E}_{R_1}^\infty$  decreases, reaches a minimum and then explodes approaching  $\psi_{2}$ . However, past  $\psi_{2}$ ,  $\mathcal{E}_{R_1}^\infty$  decreases again with  $\psi_{1}$ . This double descent behavior has been observed/studied in many settings, including Mei & Montanari (2022) and references therein.

Theorem 2 (Theorem 4 Mei & Montanari (2022)). The asymptotic test error (6) for regime  $R_{2}$  equals

$$
\mathcal {E} _ {R _ {2}} ^ {\infty} \equiv \lim  _ {\psi_ {1} \rightarrow \infty} \lim  _ {d \rightarrow \infty} \mathcal {E} = F _ {1} ^ {2} \mathcal {B} _ {\text {w i d e}} (\zeta , \psi_ {2}, \bar {\lambda}) + \left(\tau^ {2} + F _ {\star} ^ {2}\right) \mathcal {V} _ {\text {w i d e}} (\zeta , \psi_ {2}, \bar {\lambda}) + F _ {\star} ^ {2}, \tag {14}
$$

where  $\mathcal{B}_{\mathrm{wide}}(\zeta, \psi_2, \overline{\lambda}) \equiv (\psi_2\omega_2 - \psi_2) / ((\psi_2 - 1)\omega_2^3 + (1 - 3\psi_2)\omega_2^2 + 3\psi_2\omega_2 - \psi_2), \mathcal{V}_{\mathrm{wide}}(\zeta, \psi_2, \overline{\lambda}) \equiv (\omega_2^3 - \omega_2^2) / ((\psi_2 - 1)\omega_2^3 + (1 - 3\psi_2)\omega_2^2 + 3\psi_2\omega_2 - \psi_2)$  and

$$
\omega_ {2} \equiv - \left(\sqrt {\left(\psi_ {2} \zeta^ {2} - \zeta^ {2} - \bar {\lambda} \psi_ {2} - 1\right) ^ {2} + 4 \psi_ {2} \zeta^ {2} (\bar {\lambda} \psi_ {2} + 1)} + \psi_ {2} \zeta^ {2} - \zeta^ {2} - \bar {\lambda} \psi_ {2} - 1\right) / (2 (\bar {\lambda} \psi_ {2} + 1)). \tag {15}
$$

Theorem 3 (Theorem 5 Mei & Montanari (2022)). The asymptotic test error (6) for regime  $R_{3}$  equals

$$
\mathcal {E} _ {R _ {3}} ^ {\infty} \equiv \lim  _ {\psi_ {2} \rightarrow \infty} \lim  _ {d \rightarrow \infty} \mathcal {E} = F _ {1} ^ {2} \mathcal {B} _ {\text {l s a m p}} (\zeta , \psi_ {1}, \lambda / \mu_ {\star} ^ {2}) + F _ {\star} ^ {2}, \text {w h e r e} \tag {16}
$$

$$
\begin{array}{l} \mathcal {B} _ {\mathrm {l s a m p}} (\zeta , \psi_ {1}, \overline {{\lambda}}) \equiv (((\omega_ {1} ^ {3} - \omega_ {1} ^ {2}) / \zeta^ {2}) + \psi_ {1} \omega_ {1} - \psi_ {1}) / ((\psi_ {1} - 1) \omega_ {1} ^ {3} + (1 - 3 \psi_ {1}) \omega_ {1} ^ {2} + 3 \psi_ {1} \omega_ {1} - \psi_ {1}), a n d \\ \omega_ {1} \equiv - \left(\sqrt {\left(\psi_ {1} \zeta^ {2} - \zeta^ {2} - \bar {\lambda} \psi_ {1} - 1\right) ^ {2} + 4 \psi_ {1} \zeta^ {2} (\bar {\lambda} \psi_ {1} + 1)} + \psi_ {1} \zeta^ {2} - \zeta^ {2} - \bar {\lambda} \psi_ {1} - 1\right) / \left(2 (\bar {\lambda} \psi_ {1} + 1)\right). \tag {17} \\ \end{array}
$$

# 3.2 ASYMPTOTIC SENSITIVITY OF THE RFR MODEL

We derive a sensitivity formula for regimes  $R_{1}, R_{2}, R_{3}$ . Our theorems are a specialization (proofs in Appendix I) of the more general Theorem 13 that we include in the Appendix C for completeness.

Theorem 4. The sensitivity (7) for regime  $R_{1}$  equals

$$
S _ {R _ {1}} ^ {\infty} \equiv \lim  _ {\lambda \rightarrow 0 ^ {+}} \lim  _ {d \rightarrow \infty} S = \zeta^ {2} \left(\frac {F _ {1} ^ {2} \mathscr {D} _ {1 , \text {r l e s s}} (\zeta , \psi_ {1} , \psi_ {2})}{\left(\chi \zeta^ {2} - 1\right) \mathscr {D} _ {0 , \text {r l e s s}} (\zeta , \psi_ {1} , \psi_ {2})} + \frac {\left(F _ {\star} ^ {2} + \tau^ {2}\right) \mathscr {D} _ {2 , \text {r l e s s}} (\zeta , \psi_ {1} , \psi_ {2})}{\mathscr {D} _ {0 , \text {r l e s s}} (\zeta , \psi_ {1} , \psi_ {2})}\right), \tag {18}
$$

where  $\mathcal{D}_{0,\mathrm{rless}}(\zeta, \psi_1, \psi_2)$ ,  $\mathcal{D}_{1,\mathrm{rless}}(\zeta, \psi_1, \psi_2)$ , and  $\mathcal{D}_{2,\mathrm{rless}}(\zeta, \psi_1, \psi_2)$  are polynomials defined in Appendix D.

Theorem 5. Let  $\omega_{2}$  equal (15). The sensitivity (7) for regime  $R_{2}$  equals

$$
S _ {R _ {2}} ^ {\infty} \equiv \lim  _ {\psi_ {1} \rightarrow \infty} \lim  _ {d \rightarrow \infty} S = \frac {\omega_ {2} {} ^ {2} \left(\left(F _ {\star} ^ {2} + \tau^ {2}\right) (- 1 + \omega_ {2}) + F _ {1} ^ {2} (- 1 - \psi_ {2} + \omega_ {2} (- 1 + \psi_ {2}))\right)}{(- 1 + \omega_ {2}) (\psi_ {2} - 2 \omega_ {2} \psi_ {2} + \omega_ {2} {} ^ {2} (- 1 + \psi_ {2}))}. \tag {19}
$$

Theorem 6. Let  $\omega_{1}$  equal (17). The sensitivity (7) for regime  $R_{3}$  equals

$$
S _ {R _ {3}} ^ {\infty} \equiv \lim  _ {\psi_ {2} \rightarrow \infty} \lim  _ {d \rightarrow \infty} S = F _ {1} ^ {2} \left(1 + \left(2 / (- 1 + \omega_ {1})\right) + \left(\psi_ {1} / \left(\psi_ {1} - 2 \omega_ {1} \psi_ {1} + \omega_ {1} ^ {2} (- 1 + \psi_ {1})\right)\right)\right). \tag {20}
$$

# 4 MAIN RESULTS

We will find the simplest AFs that lead to the best trade-off between approximation accuracy and sensitivity for the RFR model. Mathematically, we will solve (8). From the theorems in Section 3 we know that  $\mathcal{E}$  and  $S$ , and hence  $O = (1 - \alpha)\mathcal{E} + \alpha S$ , only depend on the AF via  $\mu_0, \mu_1, \mu_2$ . Therefore, we will proceed in two steps. In Section 4.1, we will fix  $\mu_0, \mu_1, \mu_2$ , and find  $\sigma$  with associated values  $\mu_0, \mu_1, \mu_2$  that has minimal norm, either (4) or (3). In Section 4.2, we will find values of  $\mu_0, \mu_1, \mu_2$  that minimize  $O = (1 - \alpha)\mathcal{E} + \alpha S$ . Together, these specify optimal AFs for the RFR model.

It is the case that properties of the RFR model other than the test error and sensitivity also only depend on the AF via  $\mu_0, \mu_1, \mu_2$ . One example is the robustness of the RFR model to disparities between the training and test data distribution (Tripuraneni et al., 2021). Although we do not focus on these other properties, the results in Section (4.1) can be used to generate optimal AFs for them as well, as long as, similar to in Section 4.2, we can obtain  $\mu_0, \mu_1, \mu_2$  that optimize these other properties.

We made the decision to, as often as possible, simplify expressions by manipulating them to expose the signal to noise ratio  $\rho = F_1^2 / (\tau^2 + F_\star^2)$ ,  $F_1 > 0$ , rather than using the variables  $F_1, \tau$ , and  $F_\star$ . The only downside is that conclusions in the regime  $\tau = F_\star = 0$  require a bit more of effort to be extracted, often been readable in the limit  $\rho \to \infty$ .

The complete proofs of our main results can be found in Appendix I and their main ideas below. The proofs of Section 4.2 are algebraically heavy and we provide a Mathematical file to symbolically check expressions of both theorem statements and proofs in the supplementary material.

# 4.1 OPTIMAL ACTIVATION FUNCTIONS GIVEN FIXED  $\mu_0,\mu_1$  , AND  $\mu_{2}$

Since one of our goals is knowing when an optimal AF is linear we start with the following lemma.

Lemma 4.1. The  $AF$ $\sigma$  is linear (almost surely) if and only if  $\mu_{\star}^{2}\triangleq \mu_{2} - \mu_{1}^{2} - \mu_{0}^{2} = 0$

We now state results for the norms (4) and (3). The problem we will solve under both norms is similar. Let  $Z \sim \mathcal{N}(0,1)$ . We consider solving the following functional problem, where  $i = 1$  or 2,

$$
\min  _ {\sigma} \| \sigma \| _ {i} \text {s u b j e c t t o} \mathbb {E} (\sigma (Z)) = \mu_ {0}, \mathbb {E} (Z \sigma (Z)) = \mu_ {1}, \mathbb {E} (\sigma (Z) ^ {2}) = \mu_ {2}, \text {w i t h} Z \sim \mathcal {N} (0, 1). \tag {21}
$$

If  $i = 2$ , we seek solutions over the Gaussian-weighted Lebesgue space of twice weak-differentiable functions that have  $\mathbb{E}((\sigma(Z))^2)$  and  $\mathbb{E}((\sigma'(Z))^2)$  defined and finite. If  $i = 1$ , we seek solutions over the Gaussian-weighted Lebesgue space of weak-differentiable functions that have  $\mathbb{E}((\sigma(Z))^2)$  and  $\mathbb{E}(|\sigma'(Z)|)$  defined and finite. The derivative  $\sigma'$  is to be understood in a weak sense.

Since  $\sigma$  is a one-dimensional function, the requirement of existence of weak derivative implies that there exists a function  $\nu$  that is absolute continuous and that agrees with  $\sigma$  almost everywhere (Rudin et al., 1976). Therefore, any specific solution we propose should be understood as an equivalent class of functions that agree with  $\nu$  up to a set of measure zero with respect to the Gaussian measure.

Theorem 7. The minimizers of (21) for  $i = 2$ , i.e.  $\| \sigma \| ^2 = \mathbb{E}((\sigma '(Z))^2)$ , are

$$
\sigma (x) = a x ^ {2} + b x + c, \text {w h e r e} a = \pm \mu_ {\star} / \sqrt {2}, b = \mu_ {1}, \text {a n d} c = \mu_ {0} - a. \tag {22}
$$

In Theorem 7, if  $\mu_{\star} = 0$  there is only one minimizer, a linear function. If  $\mu_{\star} > 0$ , there are exactly two minimizers, both quadratic functions. Note that both minimizers satisfy the growth constraints of Assumption 1, and hence can be used within the analysis of the RFR model. We note that quadratic AFs have been empirically studied in the past, e.g. Wuraola & Patel (2018).

Theorem 8. One minimizer of (21) for  $i = 1$ , i.e.  $\| \sigma \| = \mathbb{E}(|\sigma '(Z)|)$ , is

$$
\sigma (x) = \mu_ {0} + b \max  \left\{\min  \{x, - s \}, s \right\}, \tag {23}
$$

where  $b = \frac{\mu_1}{\operatorname{erf}(s / \sqrt{2})}$ ,  $\operatorname{erf}(s / \sqrt{2})$  is the Guass error function, and  $s \in \mathbb{R}$  is the unique solution to the equation  $\zeta^2 \triangleq \mu_1^2 / \mu_\star^2 = g(s)$  if  $\mu_\star \neq 0$ , and  $s = +\infty$  if  $\mu_\star = 0$ , where  $g$  is specified in Appendix E.

When  $\|\sigma\| = \mathbb{E}(|\sigma'(Z)|)$ , we can characterize the complete solution family to (21). These are AFs of the form  $\sigma(x) = a + b \max \{s, \min \{t, x\}\}$ , where  $a, b, s,$  and  $t$  are chosen such that the constraints in (21) hold. It is possible to explicitly write  $a$  and  $b$  as a function of  $\mu_0, \mu_1, s, t$ , and express  $s, t$  as the solution of  $E(s, t) = \mu_1^2 / \mu_\star^2$ , where  $E(\cdot, \cdot)$  has explicit form. In this case, for each  $\mu_0, \mu_1, \mu_2$  there are an infinite number of optimal AFs since  $E(s, t) = \mu_1^2 / \mu_\star^2$  has an infinite number of solutions. ReLU's are included in this family as  $t \to \infty$ . The involved lengthy expressions do not bring any new insights, so we state and prove only Thr. 8, which is a specialization of the general theorem to  $s = -t$ .

Proofs' main ideas: We give the main ideas behind the proof of Theorem 7. The proof of Theorem 8 follows similar techniques. The first-order optimality conditions imply that  $-2x\sigma'(x) + 2\sigma''(x) + \lambda_1 + \lambda_2x + \lambda_3\sigma(x) = 0$ , where the Lagrange multipliers  $\lambda_1, \lambda_2$ , and  $\lambda_3$  must be later chosen such that  $\mathbb{E}\{\sigma(Z)\} = \mu_0$ ,  $\mathbb{E}\{Z\sigma(Z)\} = \mu_1$ , and  $\mathbb{E}\{\sigma^2(Z)\} = \mu_2$ . Using the change of variable  $\sigma(x) = \tilde{\sigma}(x/\sqrt{2}) - \lambda_1/\lambda_3 - x\lambda_2/\left(\lambda_3 - 1\right)$  we obtain  $-2x\tilde{\sigma}'(x) + \tilde{\sigma}''(x) + \lambda_3\tilde{\sigma}(x) = 0$  which is the Hermite ODE, which is well studied in physics, e.g. it appears in the study of the quantum harmonic oscillator. The cases  $\lambda_3 \in \{0,3\}$  require special treatment. Using a finite energy/norm condition we can prove that  $\lambda_3$  is quantized. In particular  $\lambda_3 = 4k$ ,  $k = 1,2,\ldots$ , which implies that  $\sigma(x) = -\lambda_1/\lambda_3 - \lambda_2x/\left(\lambda_3 - 2\right) + cH_{2k}(x/\sqrt{2})$ , where  $H_i$  is the  $i$ th Hermite polynomial and  $c$  a constant. The energy/norm is minimal when  $k = 1$ , which implies a quadratic AF.

# 4.2 ACTIVATION FUNCTION PARAMETERS FOR THE OPTIMAL TRADE-OFF BETWEEN SENSITIVITY AND MSE

We will find AF parameters that minimize a linear combination of sensitivity and test error. We are interested in an asymptotic analytical treatment in the three regimes mentioned in Section 3. To be specific, we will compute

$$
\mathcal {U} _ {R _ {i}} \left(\psi_ {1}, \psi_ {2}, \tau , \alpha , F _ {1}, F _ {\star}, \lambda\right) \equiv \underset {\mu_ {0}, \mu_ {1}, \mu_ {2}} {\arg \min } (1 - \alpha) \mathcal {E} _ {R _ {i}} ^ {\infty} + \alpha \mathcal {S} _ {R _ {i}} ^ {\infty}, \text {w h e r e} i = 1, 2, \text {o r} 3. \tag {24}
$$

We are not aware of previous work explicitly studying the trade-off between  $\mathcal{E}$  and  $S$  for the RFR model. For the RFR model, the work of Mei & Montanari (2022) studies only the test error and D'Amour et al. (2020) studies a definition of sensitivity related but different from ours. Other papers have studied trade-offs between robustness and error measures related but different than ours and for other models, e.g. Tsipras et al. (2018); Zhang et al. (2019).

When  $\alpha = 1$ , problem (24) reduces to minimizing the sensitivity. This has a trivial solution: the AF, and hence  $f$ , must be a constant, and  $S = 0$ . Therefore, below we focus on the case when  $\alpha \in [0,1)$ .

Special notation: In Theorem 9 we use the following special notation. Given two potential choices for AF parameters, say  $x$  and  $y$ , we define  $x \sqcup y$  to mean that  $x$  exists and that  $y$  might exist or not, and that  $x \sqcup y = y$  if  $y$  exists and it leads to a smaller value of  $(1 - \alpha)\mathcal{E} + \alpha S$  than using  $x$ , and otherwise  $x \sqcup y = x$ . Note that  $x \sqcup y$  and  $y \sqcup x$  make different statements about the existence of  $x$  and  $y$ . This notation is important to interpret the results of Table 1 in Theorem 9.

Theorem 9. Let  $\alpha \in [0,1)$ ,  $\psi \equiv \min\{\psi_1, \psi_2\}$  and  $\overline{\psi} \equiv \max\{\psi_1, \psi_2\}$ . We have that

$$
\mathcal {U} _ {R _ {1}} = \left\{\left(\mu_ {0}, \mu_ {1}, \mu_ {2}\right): x \mu_ {1} ^ {2} (- 1 + x + \psi) = \mu_ {\star} ^ {2} (\psi + x) \right\}, \text {w h e r e} x \text {i s a s i n T a b l e 1 .} \tag {25}
$$

<table><tr><td></td><td>β1≤ψ</td><td>β2&lt;ψ&lt;β1</td><td>β3&lt;ψ≤β2</td><td>ψ≤β3</td></tr><tr><td>(α&lt;αL)∧E1</td><td>xR</td><td>xR ⊥ x1</td><td>xR</td><td>xR</td></tr><tr><td>(α&lt;αL)∧E2∧(α&gt;αC)</td><td>x1</td><td>x1 ⊥ x3</td><td>x1</td><td>--</td></tr><tr><td>(α&lt;αL)∧E2∧(α&lt;αC)</td><td>x1</td><td>x1 ⊥ x3</td><td>x1</td><td>--</td></tr><tr><td>(α&gt;αL)∧E1∧(α&gt;αC)</td><td>--</td><td>xL</td><td>xL</td><td>xL</td></tr><tr><td>(α&gt;αL)∧E1∧(α&lt;αC)</td><td>--</td><td>xR</td><td>xR</td><td>xR</td></tr><tr><td>(α&gt;αL)∧E2</td><td>xL</td><td>xL ⊥ x2</td><td>xL ⊥ x2</td><td>xL</td></tr></table>

Table 1: The optimal AFs (25) depends on  $x$  according to this table. Cells with " - " never happen. The values of  ${x}_{1},{x}_{2},{x}_{3},{\beta }_{1},{\beta }_{2},{\beta }_{3},{\alpha }_{L},{\alpha }_{C},{\alpha }_{R},{x}_{L},{x}_{R}$  ,and the events  ${E}_{1}$  and  ${E}_{2}$  are specified below.

In Table 1,  $x_{1}, x_{2}$ , and  $x_{3}$  are the smallest, second smallest and third smallest roots of a 4th degree polynomial  $p(x)$ , specified in Appendix F, in the range  $(x_{L}, x_{R}) \triangleq (-\psi, \min \{0, 1 - \psi\})$ , if these exist. The variables  $\beta_{1}, \beta_{2}, \beta_{3}, \alpha_{L}, \alpha_{C}, \alpha_{R}$ , and the conditions  $E_{1}$  and  $E_{2}$  are as follows.

If  $\psi_{1} < \psi_{2}$  then  $E_{1} = (\alpha < \alpha_{R})$ ,  $E_{2} = (\alpha > \alpha_{R})$

$$
\beta_ {1} = \min  \left\{\psi_ {1} - 4, - 3 \psi_ {1} \right\} - 8 \sqrt {\left| 1 - \psi_ {1} \right|} \max  \left\{1 / \psi_ {1}, \psi_ {1} ^ {3 / 2} \right\} + 8 \max  \left\{1 / \psi_ {1}, \psi_ {1} ^ {2} \right\}, \tag {26}
$$

$$
\beta_ {2} = \psi_ {1} (\psi_ {1} + 2) / (\psi_ {1} + 1), \tag {27} \quad \beta_ {3} = \psi_ {1} + | 1 - \psi_ {1} | \min \left\{\psi_ {1}, 1 / \psi_ {1} \right\}, \tag{28}
$$

$$
\alpha_ {L} = \psi_ {2} / (\psi_ {2} + 1 + \rho^ {- 1}), \quad (2 9) \quad \alpha_ {C} = \psi_ {2} / (2 \psi_ {2} + 1 - \psi_ {1} + \max  \{0, 1 - \psi_ {1} \} + \rho^ {- 1}), \tag {30}
$$

$$
\alpha_ {R} = \psi_ {2} / \left(3 \psi_ {2} + 1 - 2 \min  \left\{1 + \psi_ {1}, 2 \psi_ {1} \right\} + \rho^ {- 1}\right). \tag {31}
$$

The case when  $\psi_{1} > \psi_{2}$  is described in Appendix F.4.

Remark 2. Excluding trivial scenarios where the optimal AF is a constant, and hence  $f$  is also constant, it follows directly from (25) that the optimal AF is linear if and only if  $x = x_{R}$ . With this information and Table 1, we have all the information needed to find exactly when the optimal AF is, or is not, linear. For regime  $R_{1}$ , changing  $\alpha$  alone can change the optimal AF from linear to non-linear and vice-versa (see e.g. 3rd column of Table 1), which justifies the observation 3 in Section 1.

Remark 3. For the cases considered in Table 1,  $x$  is unique. When  $\alpha \in \{\alpha_R, \alpha_L, \alpha_C\}$ , or when  $(\psi_1 > \psi_2) \wedge (\psi_1 \in \{A, B\})$ , we can lose the uniqueness of  $x$ . Yet, we can still explicitly characterize the sets of optimal  $x$  and of optimal AFs parameters. For simplicity we omit these cases from Thr. 9.

Remark 4. Theorem 9's proof gives relationships among Table 1's constants that imply that (1) no two rows/columns simultaneously hold and (2) in some cases some cells might not hold. See App. F.

We do not consider  $\psi_{1} = \psi_{2}$  in Theorem 9 because it implies  $(1 - \alpha)\mathcal{E}_{R_1}^\infty +\alpha \mathcal{S}_{R_1}^\infty$  is not defined. Note that  $\psi_{1} = \psi_{2}$  has been called the interpolation threshold in the context of studying the generalization properties of the RFR model under vanishing regularization (Mei & Montanari, 2022). See Remark 1.

The set (25) of optimal AFs parameters is invariant under scaling and shifting of  $\sigma$ . In particular, let  $\mu(\sigma)$  be  $\mu_0, \mu_1, \mu_2$  associated with  $\sigma$ . If  $\mu(\sigma) \in \mathcal{U}_{R_1}$  then  $\mu(a\sigma + b) \in \mathcal{U}_{R_1}$ . Also, when  $x \in \{x_L, x_R\}$  we can compute the optimal value of the objective explicitly. For example, if  $\psi_1 < \psi_2$  and  $x = x_L$  the optimal value of the objective is  $\frac{(1 - \alpha)(\psi_2(F_1^2 + F_\star^2) + \psi_1\tau^2)}{\psi_2 - \psi_1}$ . If  $\psi_1 > \psi_2$  and  $x =$

$x_{R}$  the optimal value of the objective is  $\frac{\alpha F_1^2(\psi_1 - \psi_2) + F_\star^2((\alpha - 1)\psi_2 - \alpha) + \alpha(\psi_1 - 1)\tau^2 - \psi_1\tau^2}{\psi_1 - \psi_2}$  if  $\psi_{1} < 1$  and  $\frac{F_1^2(\alpha(2\psi_1 - 1)(\psi_1 - \psi_2) + (\psi_1 - 1)\psi_2) + F_\star^2((\alpha - 1)\psi_2 - \alpha\psi_1) - \psi_1\tau^2}{\psi_1 - \psi_2}$  if  $\psi_{1} \geq 1$ . This follows by substitution.

Theorem 10. Let  $\alpha \in [0,1)$ . We have that,

$$
\mathcal {U} _ {R _ {2}} = \left\{\left(\mu_ {0}, \mu_ {1}, \mu_ {2}\right): \mu_ {1} ^ {2} (- 1 + 2 \psi_ {2} - x) (- 1 + x) = - 2 \left(\mu_ {\star} ^ {2} + \lambda \psi_ {2}\right) (1 + x) \right\}, \tag {32}
$$

where  $x$  is the unique solution to  $p(x) = 0$  in the range  $x \in (-1, \min\{1, -1 + 2\psi_2\})$ , where  $p(x) \triangleq p_0 + p_1x + p_2x^2 + p_3x^3 + p_4x^4$  with coefficients described in Appendix G.

Remark 5. The only way to get  $\mu_{\star} = 0$ , and hence a linear optimal AF is if  $x$  simultaneously satisfies  $\mu_1^2 (-1 + 2\psi_2 - x)(-1 + x) = -2\lambda \psi_2(1 + x)$  and  $p(x) = 0$ . Since the first equation does not depend on  $\alpha$ , but the zeros of  $p(x) = 0$  change continuously with  $\alpha$ , only very special choices of parameters lead to linear AFs. In general, regime  $R_{2}$  does not have optimal linear AFs.

Theorem 11. Let  $\alpha \in [0,1)$ . We have that

$$
\mathcal {U} _ {R _ {3}} = \left\{ \begin{array}{l l} \left\{\left(\mu_ {0}, \mu_ {1}, \mu_ {2}\right): \mu_ {\star} = 0 \wedge \mu_ {1} = \infty \right\}, & i f \alpha = 0 \vee \left(\psi_ {1} = 1 \wedge 0 <   \alpha \leq \frac {1}{4}\right) \\ \left\{\left(\mu_ {0}, \mu_ {1}, \mu_ {2}\right): \mu_ {\star} = 0 \wedge \mu_ {1} ^ {2} = \frac {- 4 \alpha^ {2} \lambda + 3 \alpha \lambda + \sqrt {\alpha} \lambda}{1 6 \alpha^ {2} - 8 \alpha + 1} \right\}, & i f \psi_ {1} = 1 \wedge \alpha > \frac {1}{4} \\ \left\{\left(\mu_ {0}, \mu_ {1}, \mu_ {2}\right): \mu_ {\star} = 0 \wedge \mu_ {1} ^ {2} (- 1 + 2 \psi_ {1} - x) (- 1 + x) + 2 \lambda \psi_ {1} (1 + x) = 0 \right\}, & i f \psi_ {1} \neq 1 \end{array} \right. \tag {33}
$$

where  $x$  is the unique solution to  $p(x) = 0$  in the range  $x \in (-1, \min\{1, -1 + 2\psi_2\})$ , where  $p(x)$  is defined like in Theorem 10 but with  $\rho \to \infty$  and with  $\psi_2$  replaced by  $\psi_1$ .

Remark 6. The optimal AF is always linear and independent of the noise variables  $F_{\star}$  and  $\tau$

Remark 7. When  $\psi_{1} = 1, \alpha \leq \frac{1}{4}$  there is no optimal AF inside our AF search space since no AF can satisfy  $\mu_{1} = \infty$ . Rather, there exists a sequence of valid AFs with decreasing  $O$  whose  $\mu_{1} \to \infty$ .

Remark 8. We can compute the optimal objective in closed-form in some scenarios. When  $\alpha = 0$  the optimal objective is  $F_{\star}^{2}$ . When  $\psi_{1} = 1 \wedge 0 < \alpha \leq \frac{1}{4}$ , the optimal objective approaches  $\alpha F_{1}^{2} + (1 - \alpha) F_{\star}^{2}$  as  $\mu_{1} \to \infty$ . When  $\psi_{1} = 1 \wedge \alpha > \frac{1}{4}$ , the optimal objective is  $F_{1}^{2}(4\sqrt{\alpha} - 1 - 3\alpha) + F_{\star}^{2}(1 - \alpha)$ .

Proofs' main ideas: We give the main ideas behind the proof of Theorem 10. The proof of Theorems 9 and 11 follows similar techniques but require more care. The objective  $O$  only depends on AF parameters via  $\omega_{2} = \omega_{2}(\psi_{2},\mu_{1}^{2},\mu_{\star}^{2})$ . We use the Möbius transformation  $x = (1 + \omega_{2}) / (\omega_{2} - 1)$  such that the infinite range  $\omega_{2} \in [-\infty,0]$  gets mapped to the finite range  $x \in [-1,1]$ . We then focus the rest of the proof on minimizing  $O = O(x)$  over the range of  $x$ . First we show that given that  $\mu_{1}^{2},\mu_{\star}^{2} \geq 0,\psi_{1} > 0$ , the range of  $x$  can be reduced to  $x \in [x_L,x_R] \triangleq [-1,\min \{1, - 1 + 2\psi_2\}]$ . Then we compute  $\mathrm{d}O / \mathrm{d}x$  and  $\mathrm{d}^2 O / \mathrm{d}x^2$ , which turns out to be rational functions of  $x$ . We then show that if  $x \in [-1,\min \{1, - 1 + 2\psi_2\}]$  then  $\mathrm{d}^2 O / \mathrm{d}x^2 >0$ , so  $O$  is strictly convex. We also show that  $\mathrm{d}O / \mathrm{d}x < 0$  at  $x_{L}$  and  $\mathrm{d}O / \mathrm{d}x > 0$  at  $x_{R}$ , thus  $x_{R}$  and  $x_{L}$  cannot be minimizers. Finally, we show that the zeros of the numerator  $p(x)$  of the rational function  $\mathrm{d}O / \mathrm{d}x$  differ from the denominator's zeros. So the optimal  $x$  is the unique solution to  $p(x)$  in  $[x_L,x_R]$ .

# 4.3 IMPORTANT OBSERVATIONS

Together, Sections 4.1 and 4.2 explicitly and fully characterize the solutions of (8) in the ridgeless, overparametrized, and large sample regimes. A few important observations follow from our theory. In Appendix H we discuss more on this topic and include details on the observations below.

Observation 1: In regime  $R_{1}$ , and if  $\alpha = 0, \psi_{1} < \psi_{2}$ , the optimal AF is linear. This follows from Theorem 9 and Remark 2. Indeed, expressions simplify and we get that  $p(x) = \rho \psi_{2} (\psi_{1} - (x + \psi_{1})^{2})^{2}$  if  $\psi_{1} \neq 1$  or  $p(x) = -(2 + x)^{2} \rho \psi_{2}$  if  $\psi_{1} = 1$ , which implies that  $x_{1}$  does not exist (since it would be outside of  $(-x_{L}, x_{R})$ ). Hence, the first row of Table 1 always gives  $x = x_{R}$  and the optimal AF is linear. Also, when  $\alpha = 0, \psi_{1} < \psi_{2}$ , we can explicitly compute the optimal objective (see paragraph before Theorem 10). If furthermore  $\tau = F_{\star} = 0$ , we can show that also when  $\psi_{1} > \psi_{2}$ ,  $x_{1}$  does not exist and  $x = x_{R}$ , therefore the optimal objective and AF when  $\psi_{1} > \psi_{2}$  have the same formula as when  $\psi_{1} < \psi_{2}$ . Hence, if  $\alpha = \tau = F_{\star} = 0, \psi_{1} < \psi_{2}$ , from the formula one can conclude that choosing an optimal linear AF destroys the double descent curve if  $\psi_{2} > 1^{1}$ , the test error becoming exactly zero for  $\psi_{1} \geq 1$ . This contrasts with choosing a non-linear, sub-optimal, AF which will exhibit a double descent curve. This justifies observation 1 (low complexity  $\psi_{1} < \psi_{2}$ ) and observation 2 in Sec. 1. Fig. 1-(A,B) illustrates this and details the high-complexity  $(\psi_{1} > \psi_{2})$  observation.

![](images/bf2e0eab41f6bce267179cf9afb4c510cabc318a61965a196b6652622314bb45.jpg)  
Figure 1: (A) Consider the regime  $R_{1}$ . In a noiseless setting, if  $\psi_{2} > 1$ , the evolution of  $O$  versus  $\psi_{1}$ , when an optimal linear AF  $\sigma^{\star}$  is used, can achieve 0 test error for  $\psi_{1} \geq 1$ . However, if a non-linear  $\sigma_{\mathrm{ReLU}}$  is used, we observe the typical double descent curve. (B) Consider the regime  $R_{1}$ . If there is observation noise  $\tau > 0$ , the evolution of  $O$  versus  $\psi_{1}$  with a linear AF  $\sigma_{\mathrm{linear}}$  is only optimal for  $\psi_{1} < \psi_{2}$ . For  $\psi_{1} > \psi_{2}$ ,  $O$  is optimal for a linear AF until  $\psi_{1} < C$  ( $C = 5$  for the parameters here). For  $\psi_{1} > C$  a non-linear AF  $\sigma_{\star}$ , here close to but different from a ReLU, achieves minimal  $O$ . (C) Consider the regime  $R_{2}$ . When a ReLU is used (green curves), the evolution of  $O$  versus  $\lambda$  for both low and high Signal to Noise Ratio (SNR)  $\rho$  is only optimal for a special choice of  $\lambda$ , achieving the minimum  $O_{\sigma_{\mathrm{ReLU}}}(\lambda^{\star})$ . However, also for the same low and high SNR settings, when an optimal (non-linear) AF is used (orange curves), we obtain the same, or slightly better,  $O_{\sigma^{\star}}$  regardless of any careful choice for  $\lambda$ . For low SNR ( $\tau^{2} = 10$ ) we have  $O_{\sigma^{\star}} = O_{\sigma_{\mathrm{ReLU}}}(\lambda^{\star}) = 0.512$  and for high SNR ( $\tau^{2} = 5$ ) we get  $O_{\sigma_{\mathrm{ReLU}}}(\lambda^{\star}) = 0.0220 > O_{\sigma^{\star}} = 0.0217$ . (D) In a situation just like in (C) but with even higher SNR, the difference between the minimum  $O$  that can be achieved with a particular choice of  $\lambda$  (blue line ordinate value  $O_{\sigma_{\mathrm{ReLU}}}(\lambda^{\star})$ ) and the value of  $O$  with any choice of  $\lambda$  but with an optimal (non-linear) AF (orange line ordinate value  $O_{\sigma^{\star}}$ ) becomes clearly visible. We include inside of each plot the parameters used. See Appendix H.1 for how to reproduce this figure.

Observation 2: In regime  $R_{2}$ , looking at Theorem 2 and Theorem 5, one sees that both  $\mathcal{E}$  and  $S$ , and hence the objective  $O$  (cf. (5)), only depend on the optimal AF parameters via  $\omega_{2}$ . In particular, we can solve (24) by searching for the  $\omega_{2}$  that achieves the smallest objective. Given the definition of  $\omega_{2} = \omega_{2}(\psi_{2},\zeta^{2},\mu_{\star},\lambda)$  in (15), fixing  $\lambda$  and changing  $\zeta$  or  $\mu_{\star}$  always allows one to span a larger range of values for  $\omega_{2}$  than fixing the AF's parameters  $\zeta, \mu_{\star}$  and changing  $\lambda$ . In particular, a tedious calculation shows that in the first case the achievable range for  $\omega_{2}$  is  $[\frac{\psi_2}{\min\{0^-, -1 + \psi_2\}}, 0]$  which contains the range in the second case which is  $\left[\frac{1}{2}\left(\zeta^2(-\psi_2) - \sqrt{\zeta^2\left(\psi_2\left(\zeta^2(\psi_2 - 2) + 2\right) + \zeta^2 + 2\right) + 1} + \zeta^2 + 1\right), 0\right]$ . This implies that while for a fixed AF one needs to tune  $\lambda$  during learning for best performance, if an optimal AF is used, regardless of  $\lambda$ , we always achieve either equal or better performance. This justifies the observation 4 made in Section 1. This is illustrated in Figure 1-(C,D).

In Appendix J we have experiments involving real data that show consistency with these observations. The supplementary material has code to generate Fig. 1 and the figures in Appendix J for real data.

# 5 CONCLUSION AND FUTURE WORK

We solved our task of finding optimal Activation Functions (AFs) for the Random Features Regression (RFR) model and characterized when these are linear, or non-linear. Our theory yielded interesting connections between the best AF to use and the regime in which the RFR model operates: e.g. in some regimes using a linear AF is optimal; in other regimes we can avoid hyperparameter tuning by choosing an optimal non-linear AF.

We reduced the gap between the practice and theory of AFs' design, but parts remain to be closed. For example, we could only obtain explicit equations for optimal AFs under two functional norms in the optimization problem from which we extract them. We will explore other norms in the future. We will also explore adding higher order moment restrictions to the AF since some of these higher order constraints appear in the theoretical analysis of neural models (Ghorbani et al., 2021).

In future work we plan to extend our theory to extract AFs for the RFR model that optimize a combination of test error and robustness to test/train distribution shifts and adversarial attacks. The starting point will be Tripuraneni et al. (2021) and Hassani & Javanmard (2022) discussed in Section 2, the results of Hassani & Javanmard (2022) needing to be generalized from a ReLU to general AFs before we can optimize the AFs' parameters.

# REFERENCES

Forest Agostinelli, Matthew Hoffman, Peter Sadowski, and Pierre Baldi. Learning activation functions to improve deep neural networks, 2014.  
Francis Bach. On the equivalence between kernel quadrature rules and random feature expansions. The Journal of Machine Learning Research, 18(1):714-751, 2017.  
Chaity Banerjee, Tathagata Mukherjee, and Eduardo Pasiliao Jr. An empirical study on generalizations of the relu activation function. In Proceedings of the 2019 ACM Southeast Conference, pp. 164-167, 2019.  
Derek Bean, Peter J Bickel, Noureddine El Karoui, and Bin Yu. Optimal m-estimation in high-dimensional regression. Proceedings of the National Academy of Sciences, 110(36):14563-14568, 2013.  
Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine learning practice and the bias-variance trade-off. arXiv preprint arXiv:1812.11118, 2018.  
Mikhail Belkin, Alexander Rakhlin, and Alexandre B Tsybakov. Does data interpolation contradict statistical optimality? In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 1611-1619. PMLR, 2019.  
Mikhail Belkin, Daniel Hsu, and Ji Xu. Two models of double descent for weak features. SIAM Journal on Mathematics of Data Science, 2(4):1167-1180, 2020.  
Sebastien Bubeck, Yuanzhi Li, and Dheeraj Nagaraj. A law of robustness for two-layers neural networks, 2020.  
Weipeng Cao, Xizhao Wang, Zhong Ming, and Jinzhu Gao. A review on neural networks with random weights. Neurocomputing, 275:278-287, 2018.  
Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus), 2015.  
Alexander D'Amour, Katherine Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew D Hoffman, et al. Underspecification presents challenges for credibility in modern machine learning. arXiv preprint arXiv:2011.03395, 2020.  
Amit Daniely. Sgd learns the conjugate kernel class of the network. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/file/489d0396e6826eb0c1e611d82ca8b215-Paper.pdf.  
Amit Daniely, Roy Frostig, and Yoram Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper/2016/file/abea47ba24142ed16b7d8fbf2c740e0d-Paper.pdf.  
Alexander G. de G. Matthews, Jiri Hron, Mark Rowland, Richard E. Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H1-nGgWC-.  
Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE signal processing magazine, 29(6):141-142, 2012.  
Adrià Garriga-Alonso, Carl Edward Rasmussen, and Laurence Aitchison. Deep convolutional networks as shallow gaussian processes. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=BkIfsiOcKm.

Izrail Moiseevitch Gelfand, Richard A Silverman, et al. Calculus of variations. Courier Corporation, 2000.  
Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. The Annals of Statistics, 49(2):1029-1054, 2021.  
Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.  
Ian J Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron Courville, and Yoshua Bengio. Maxout networks. arXiv preprint arXiv:1302.4389, 2013.  
Mohit Goyal, Rajan Goyal, and Brejesh Lall. Improved polynomial neural networks with normalised activations. In 2020 International Joint Conference on Neural Networks (IJCNN), pp. 1-8, 2020. doi: 10.1109/IJCNN48605.2020.9207535.  
Hamed Hassani and Adel Javanmard. The curse of overparametrization in adversarial training: Precise analysis of robust generalization for random features regression. arXiv preprint arXiv:2201.05149, 2022.  
Trevor Hastie, Andrea Montanari, Saharon Rosset, and Ryan J Tibshirani. Surprises in high-dimensional ridgeless least squares interpolation. The Annals of Statistics, 50(2):949-986, 2022.  
Tamir Hazan and T. Jaakkola. Steps toward deep kernel methods from infinite neural networks. *ArXiv*, abs/1508.05133, 2015.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. 2015 IEEE International Conference on Computer Vision (ICCV), Dec 2015. doi: 10.1109/iccv.2015.123. URL http://dx.doi.org/10.1109/ICCV.2015.123.  
Lei Huang, Jie Qin, Yi Zhou, Fan Zhu, Li Liu, and Ling Shao. Normalization techniques in training dnns: Methodology, analysis and application. arXiv preprint arXiv:2009.12836, 2020.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.  
Maryam Mahsal Khan, Arbab Masood Ahmad, Gul Muhammad Khan, and Julian F Miller. Fast learning neural networks using cartesian genetic programming. Neurocomputing, 121:274-289, 2013.  
Jan Kukačka, Vladimir Golkov, and Daniel Cremers. Regularization for deep learning: A taxonomy. arXiv preprint arXiv:1710.10686, 2017.  
Tengyuan Liang and Alexander Rakhlin. Just interpolate: Kernel "ridgeless" regression can generalize. The Annals of Statistics, 48(3):1329-1347, 2020.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, pp. 3, 2013.  
Song Mei and Andrea Montanari. The generalization error of random features regression: Precise asymptotics and the double descent curve. Communications on Pure and Applied Mathematics, 75 (4):667-766, 2022.  
Gabriel Mel and Jeffrey Pennington. Anisotropic random feature regression in high dimensions. In International Conference on Learning Representations, 2022.  
Mirco Miletarí, Thiparat Chotibut, and Paolo E. Trevisanutto. Mean field theory of activation functions in deep neural networks, 2019.  
Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124003, 2021.  
Andrei Nicolae. Plu: The piecewise linear unit activation function. arXiv preprint arXiv:1809.09534, 2018.

Roman Novak, Yasaman Bahri, Daniel A Abolafia, Jeffrey Pennington, and Jascha Sohl-Dickstein. Sensitivity and generalization in neural networks: an empirical study. arXiv preprint arXiv:1802.08760, 2018.  
Roman Novak, Lechao Xiao, Yasaman Bahri, Jaehoon Lee, Greg Yang, Daniel A. Abolafia, Jeffrey Pennington, and Jascha Sohl-dickstein. Bayesian deep convolutional networks with many channels are gaussian processes. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=B1g30j0qF7.  
Jeffrey Pennington, Samuel Schoenholz, and Surya Ganguli. The emergence of spectral universality in deep networks. In International Conference on Artificial Intelligence and Statistics, pp. 1924-1932. PMLR, 2018.  
Riccardo Poli. Parallel distributed genetic programming. University of Birmingham, Cognitive Science Research Centre, 1996.  
Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis (eds.), Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007a. URL https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf.  
Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. Advances in neural information processing systems, 20, 2007b.  
Prajit Ramachandran, Barret Zoph, and Quoc V. Le. Searching for activation functions, 2017.  
Andrinandrasana David Rasamoelina, Fouzia Adjailia, and Peter Sincák. A review of activation function for artificial neural network. In 2020 IEEE 18th World Symposium on Applied Machine Intelligence and Informatics (SAMI), pp. 281-286. IEEE, 2020.  
Andras Rozsa and Terrance E. Boult. Improved adversarial robustness by reducing open space risk via tent activations. 2019.  
Walter Rudin et al. Principles of mathematical analysis, volume 3. McGraw-hill New York, 1976.  
Hossein Taheri, Ramtin Pedarsani, and Christos Thrampoulidis. Fundamental limits of ridge-regularized empirical risk minimization in high dimensions. In International Conference on Artificial Intelligence and Statistics, pp. 2773-2781. PMLR, 2021.  
Mohammadamin Tavakoli, Forest Agostinelli, and Pierre Baldi. Splash: Learnable activation functions for improving accuracy and adversarial robustness. Neural Networks, 140:1-12, 2021. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2021.02.023. URL https://www.sciencedirect.com/science/article/pii/S0893608021000733.  
Nilesh Tripuraneni, Ben Adlam, and Jeffrey Pennington. Overparameterization improves robustness to covariate shift in high dimensions. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=PxMfDdPnTfV.  
Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Alexander Turner, and Aleksander Madry. Robustness may be at odds with accuracy. arXiv preprint arXiv:1805.12152, 2018.  
Michael Unser. A representative theorem for deep neural networks, 2019.  
Daniel Weingaertner, Victor K Tatai, Ricardo R Gudwin, and Fernando J Von Zuben. Hierarchical evolution of heterogeneous neural networks. In Proceedings of the 2002 Congress on Evolutionary Computation. CEC'02 (Cat. No. 02TH8600), volume 2, pp. 1775-1780. IEEE, 2002.  
Christopher Williams. Computing with infinite networks. In M. C. Mozer, M. Jordan, and T. Petsche (eds.), Advances in Neural Information Processing Systems, volume 9. MIT Press, 1996. URL https://proceedings.neurips.cc/paper/1996/file/ae5e3ce40e0404a45ecacaaf05e5f735-Paper.pdf.

Adedamola Wuraola and Nitish Patel. Sqnl: A new computationally efficient activation function. In 2018 International Joint Conference on Neural Networks (IJCNN), pp. 1-7. IEEE, 2018.  
Zitong Yang, Yu Bai, and Song Mei. Exact gap between generalization error and uniform convergence in random feature models. In International Conference on Machine Learning, pp. 11704-11715. PMLR, 2021.  
Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In International conference on machine learning, pp. 7472-7482. PMLR, 2019.  
Yuan Zhou, Dandan Li, Shuwei Huo, and Sun-Yuan Kung. Soft-root-sign activation function. arXiv preprint arXiv:2003.00547, 2020.  
Yuong Zhou, Zezhou Zhu, and Zhao Zhong. Learning specialized activation functions with the piecewise linear unit, 2021.
