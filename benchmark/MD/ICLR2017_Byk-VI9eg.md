# GENERATIVE MULTI-ADVERSARIAL NETWORKS

Ishan Durugkar*, Ian Gemp*, Sridhar Mahadevan

College of Information and Computer Sciences

University of Massachusetts, Amherst

Amherst, MA 01060, USA

{idurugkar, imgemp, mahadeva}@cs.umass.edu

# ABSTRACT

Generative adversarial networks (GANs) are a framework for producing a generative model by way of a two-player minimax game. In this paper, we propose the Generative Multi-Adversarial Network (GMAN), a framework that extends GANs to multiple discriminators. In previous work, the successful training of GANs requires modifying the minimax objective to accelerate training early on. In contrast, GMAN can be reliably trained with the original, untampered objective. We explore a number of design perspectives with the discriminator role ranging from formidable adversary to forgiving teacher. Image generation tasks comparing the proposed framework to standard GANs demonstrate GMAN produces higher quality samples in a fraction of the iterations when measured by a pairwise GAM-type metric.

# 1 INTRODUCTION

Generative adversarial networks (Goodfellow et al. (2014)) (GANs) are a framework for producing a generative model by way of a two-player minimax game. One player, the generator, attempts to generate realistic data samples by transforming noisy samples,  $z$ , drawn from a simple distribution (e.g.,  $z \sim \mathcal{N}(0,1)$ ) using a transformation function  $G_{\theta}(z)$  with learned weights,  $\theta$ . The generator receives feedback as to how realistic its synthetic sample is from another player, the discriminator, which attempts to discern between synthetic data samples produced by the generator and samples drawn from an actual dataset using a function  $D_{\omega}(x)$  with learned weights,  $\omega$ .

The GAN framework is one of the more recent successes in a line of research on adversarial training in machine learning (Schmidhuber (1992); Bagnell (2005); Ajakan et al. (2014)) where games between learners are carefully crafted so that Nash equilibria coincide with some set of desired optimality criteria. Preliminary work on GANs focused on generating images (e.g., MNIST (LeCun et al. (1998)), CIFAR (Krizhevsky (2009))), however, GANs have proven useful in a variety of application domains including learning censored representations (Edwards & Storkey (2015)), imitating expert policies (Ho & Ermon (2016)), and domain transfer (Yoo et al. (2016)). Work extending GANs to semi-supervised learning (Chen et al. (2016); Mirza & Osindero (2014); Gauthier (2014); Springenberg (2015)), inference (Makhzani et al. (2015); Dumoulin et al. (2016)), and improved image generation (Im et al. (2016); Denton et al. (2015); Radford et al. (2015)) have shown promise as well.

Despite these successes, GANs are reputedly difficult to train. While research is still underway to improve training techniques and heuristics (Salimans et al. (2016)), most approaches have focused on understanding and generalizing GANs theoretically with the aim of exploring more tractable formulations (Zhao et al. (2016); Li et al. (2015); Uehara et al. (2016); Nowozin et al. (2016)).

In this paper, we theoretically and empirically justify generalizing the GAN framework to multiple discriminators. We review GANs and related work in Section 2. In Section 3, we present our  $N$ -discriminator extension to the GAN framework (Generative Multi-Adversarial Networks) with several variants which range the role of the discriminator from formidable adversary to forgiving teacher. Section 3.3 explains how this extension makes training with the untampered minimax objective tractable. In Section 4, we define an intuitive metric (GMAM) to quantify GMAN perfor

mance and evaluate our framework on a variety of image generation tasks. Section 5 concludes with a summary of our contributions and directions for future research.

Contributions—To summarize, our main contributions are: i) a multi-discriminator GAN framework, GMAN, that allows training with the original, untampered minimax objective; ii) a generative multi-adversarial metric (GMAM) to perform pairwise evaluation of separately trained frameworks; iii) a particular instance of GMAN,  $\mathrm{GMAN}^*$ , that allows the generator to automatically regulate training and reach higher performance (as measured by GMAM) in a fraction of the training time required for the standard GAN model.

# 2 GENERATIVE ADVERSARIAL NETWORKS

The original formulation of a GAN is a minimax game between a generator,  $G_{\theta}(z):z\to x$ , and a discriminator,  $D_{\omega}(x):x\rightarrow [0,1]$

$$
\min  _ {G} \max  _ {D \in \mathcal {D}} V (D, G) = \mathbb {E} _ {x \sim p _ {\text {d a t a}} (x)} \left[ \log (D (x)) \right] + \mathbb {E} _ {z \sim p _ {z} (z)} \left[ \log (1 - D (G (z))) \right], \tag {1}
$$

where  $p_{data}(x)$  is the true data distribution and  $p_z(z)$  is a simple (usually fixed) distribution that is easy to draw samples from (e.g.,  $\mathcal{N}(0,1)$ ). We differentiate between the function space of discriminators,  $\mathcal{D}$ , and elements of this space,  $D$ . Let  $p_G(x)$  be the distribution induced by the generator,  $G_{\theta}(z)$ . We assume  $D, G$  to be deep neural networks as is typically the case.

In their original work, Goodfellow et al. (2014) proved that given sufficient network capacities and an oracle providing the optimal discriminator,  $D^{*} = \arg \max_{\mathcal{D}} V(D, G)$ , gradient descent will recover the desired globally optimal solution,  $p_{G}(x) = p_{data}(x)$ , so that the generator distribution exactly matches the data distribution. In practice, they replaced the second term,  $\log(1 - D(G(z)))$ , with  $-\log(D(G(z)))$  to enhance gradient signals at the start of the game; note this is no longer a zero-sum game. Part of their convergence and optimality proof involves using the oracle,  $D^{*}$ , to reduce the minimax game to a minimization over  $G$  only:

$$
\min  _ {G} V \left(D ^ {*}, G\right) = \min  _ {G} \left\{C (G) = - \log (4) + 2 \cdot J S D \left(p _ {\text {d a t a}} \| p _ {G}\right) \right\} \tag {2}
$$

where  $JSD$  denotes the Jensen-Shannon divergence. Note that minimizing  $C(G)$  necessarily minimizes  $JSD$ , however, we are rarely able to obtain  $D^{*}$  and so we instead minimize  $V(D,G)$ , which is only a lower bound.

This perspective of minimizing the distance between the distributions,  $p_{data}$  and  $p_G$ , motivated Li et al. (2015) to develop a generative model that matches all moments of  $p_G(x)$  with  $p_{data}(x)$  (at optimality) by minimizing maximum mean discrepancy (MMD). Another approach, EBGAN, (Zhao et al. (2016)) explores a larger class of games (non-zero-sum games) which generalize the generator and discriminator objectives to take real-valued "energies" as input instead of probabilities. Nowozin et al. (2016) and then Uehara et al. (2016) extended the  $JSD$  perspective on GANs to more general divergences, specifically  $f$ -divergences and then Bregman-divergences respectively.

In general, these approaches focus on exploring fundamental reformulations of  $V(D,G)$ . Similarly, our work focuses on a fundamental reformulation, however, our aim is to provide a framework that accelerates training of the generator to a more robust state irrespective of the choice of  $V$ .

# 3 MULTIPLE DISCRIMINATORS

The introduction of multiple discriminators brings with it a number of design possibilities. Here, we explore approaches ranging between two extremes: 1) a more discriminating  $D$  (better approximating  $\max_{\mathcal{D}} V(D, G)$ ) and 2) a  $D$  better matched to the generator's capabilities. Mathematically, we reformulate  $G$ 's objective as  $\min_G \max_{\mathcal{D}_{Team}} F(V(D_1, G), \ldots, V(D_N, G))$  for different choices of  $F$  (see Figure 1) where  $\mathcal{D}_{Team}$  denotes the combinatorial space of discriminator teams. Each  $D_i$  on the other hand, is still expected to independently maximize its own  $V(D_i, G)$ . We sometimes abbreviate  $V(D_i, G)$  with  $V_i$  and  $F(V_1, \ldots, V_N)$  with  $F_G(V_i)$ .

![](images/4f38d8060e71ff0ae1c60b85e454860ecdc2b91415f0fcb03fc881efe18936e9.jpg)  
Figure 1: (GMAN) The generator trains against the best available discriminator ( $F \coloneqq \max$ ). We explore alternatives to  $F$  in Sections 3.3 & 3.4.

# 3.1 MAXIMIZING V(D,G)

For a fixed  $G$ , maximizing  $F_{G}(V_{i})$  with  $F \coloneqq \max$  and  $N$  randomly instantiated copies of our discriminator is functionally equivalent to optimizing  $V$  (e.g., stochastic gradient ascent) with random restarts in parallel and then presenting  $\max_{i \in \{1, \dots, N\}} V(D_{i}, G)$  as the loss to the generator—a very pragmatic approach to the difficulties presented by the non-convexity of  $V$  caused by the deep net. Requiring the generator to minimize the max forces  $G$  to generate high fidelity samples that must hold up under the scrutiny of all  $N$  discriminators, each potentially representing a distinct local maximum.

In practice,  $\max_{D_i\in \mathcal{D}}V(D_i,G)$  is not performed to convergence (or global optimality), so the above problem is oversimplified. Furthermore, introducing  $N$  discriminators affects the dynamics of the game which affects the trajectories of the discriminators. This prevents us from claiming  $\max \{V_1(t),\ldots ,V_N(t)\} >\max \{V_1'(t)\} \forall t$  even if we initialize  $D_{1}(0) = D_{1}'(0)$  as it is unlikely that  $D_{1}(t) = D_{1}'(t)$  at some time  $t$  after the start of the game.

# 3.2 BOOSTING

We can also consider taking the max over  $N$  discriminators as a form of boosting for the discriminator's online classification problem (online because  $G$  can produce an infinite data stream). The boosted discriminator is given a sample  $x_{t}$  and must predict whether it came from the generator or the dataset. The booster then makes its prediction using the predictions of the  $N$  weaker  $D_{i}$ .

There are a few differences between taking the max (case 1) and online boosting (case 2). In case 1, our booster is limited to selecting a single weak discriminator (i.e. a pure strategy), while in case 2, many boosting algorithms more generally use linear combinations of the discriminators. Moreover, in case 2, a booster must make a prediction before receiving a loss function. In case 1, we assume access to the loss function at prediction time, which allows us to compute the max.

It is possible to train the weak discriminators using boosting and then ignore the booster's prediction by instead presenting  $\max \{V_i\}$ . We explore both variants in our experiments, using the adaptive algorithm proposed in Beygelzimer et al. (2015). Unfortunately, boosting failed to produce promising results on the image generation tasks. It is possible that boosting produces too strong an adversary for learning which motivates the next section. Boosting results appear in Appendix A.5.

# 3.3 REGULATING THE DISCRIMINATOR

The previous perspectives focus on improving the discriminator with the goal of presenting a better approximation of  $\max_{\mathcal{D}} V(D, G)$  to the generator. Our third perspective asks the question, "Is  $\max_{\mathcal{D}} V(D, G)$  too harsh a critic?"

# 3.3.1 Soft-DISCRIMINATOR

In practice, training against a far superior discriminator can impede the generator's learning. This is because the generator is unlikely to generate any samples considered "realistic" by the discriminator's standards, and so the generator will receive uniformly negative feedback. This is problematic because the information contained in the gradient derived from negative feedback only dictates

where to drive down  $p_G(x)$ , not specifically where to increase  $p_G(x)$ . Furthermore, driving down  $p_G(x)$  necessarily increases  $p_G(x)$  in other regions of  $\mathcal{X}$  (to maintain  $\int_{\mathcal{X}} p_G(x) = 1$ ) which may or may not contain samples from the true dataset (whack-a-mole dilemma). In contrast, a generator is more likely to see positive feedback against a more lenient discriminator, which may better guide a generator towards amassing  $p_G(x)$  in approximately correct regions of  $\mathcal{X}$ .

For this reason, we explore a variety of functions that allow us to soften the max operator. We choose to focus on soft versions of the three classical Pythagorean means parameterized by  $\lambda$  where  $\lambda = 0$  corresponds to the mean and the max is recovered as  $\lambda \to \infty$ :

$$
\mathrm {A M} _ {\text {s o f t}} (V, \lambda) = \sum_ {i} ^ {N} w _ {i} V _ {i} \tag {3}
$$

$$
\mathrm {G M} _ {\text {s o f t}} (V, \lambda) = - \exp \left(\sum_ {i} ^ {N} w _ {i} \log (- V _ {i})\right) \tag {4}
$$

$$
\mathrm {H M} _ {\text {s o f t}} (V, \lambda) = \left(\sum_ {i} ^ {N} w _ {i} V _ {i} ^ {- 1}\right) ^ {- 1} \tag {5}
$$

where  $w_{i} = e^{\lambda V_{i}} / \Sigma_{j}e^{\lambda V_{j}}$  with  $\lambda \geq 0, V_{i} < 0$ . Using a softmax also has the well known advantage of being differentiable (as opposed to subdifferentiable for max). Note that we only require continuity to guarantee that computing the softmax is actually equivalent to computing  $V(\tilde{D},G)$  where  $\tilde{D}$  is some convex combination of  $D_{i}$  (see Appendix A.3).

# 3.3.2 USING THE ORIGINAL MINIMAX OBJECTIVE

To illustrate the effect the softmax has on training, observe that the component of  $AM_{soft}(V,0)$  relevant to generator training can be rewritten as

$$
\frac {1}{N} \sum_ {i} ^ {N} \mathbb {E} _ {x \sim p _ {G} (x)} [ \log (1 - D _ {i} (x)) ] = \frac {1}{N} \mathbb {E} _ {x \sim p _ {G} (x)} [ \log (z) ]. \tag {6}
$$

where  $z = \prod_{i}^{N}(1 - D_{i}(x))$ . Note that the generator gradient,  $|\frac{\partial\log(z)}{\partial z}|$ , is minimized at  $z = 1$  over  $z\in (0,1]^{1}$ . From this form, it is clear that  $z = 1$  if and only if  $D_{i} = 0\forall i$ , so  $G$  only receives a vanishing gradient if all  $D_{i}$  agree that the sample is fake; this is especially unlikely for large  $N$ . In other words,  $G$  only needs to fool a single  $D_{i}$  to receive constructive feedback. This result allows the generator to successfully minimize the original generator objective,  $\log (1 - D)$ . This is in contrast to the more popular objective,  $-\log (D)$ , introduced to artificially enhance gradients at the start of training.

At the beginning of training, when  $\max_{D_i} V(D_i, G)$  is likely too harsh a critic for the generator, we can set  $\lambda$  closer to zero to use the mean, increasing the odds of providing constructive feedback to the generator. In addition, the discriminators have the added benefit of functioning as an ensemble, reducing the variance of the feedback presented to the generator, which is especially important when the discriminators are far from optimal and are still learning a reasonable decision boundary. As training progresses and the discriminators improve, we can increase  $\lambda$  to become more critical of the generator for more refined training.

# 3.3.3 MAINTAINING MULTIPLE HYPOTHESES

We argue for this ensemble approach on a more fundamental level as well. Here, we draw on the density ratio estimation perspective of GANs (Uehara et al. (2016)). The original GAN proof assumes we have access to  $p_{data}(x)$ , if only implicitly. In most cases of interest, the discriminator only has access to a finite dataset sampled from  $p_{data}(x)$ ; therefore, when computing expectations of  $V(D,G)$ , we only draw samples from our finite dataset. This is equivalent to training a GAN with  $p_{data}(x) = \tilde{p}_{data}$  which is a distribution consisting of point masses on all the data points in the dataset. For the sake of argument, let's assume we are training a discriminator and generator, each

with infinite capacity. In this case, the global optimum  $(p_G(x) = \tilde{p}_{data(x)})$  fails to capture any of the interesting structure from  $p_{data}(x)$ , the true distribution we are trying to learn. Therefore, it is actually critical that we avoid this global optimum.

![](images/26a2dddb8756c5935f994188ff37d5ae49b856552d2582a268cb032f6fae785f.jpg)  
Figure 2: Consider a dataset consisting of the nine 1-dimensional samples in black. Their corresponding probability mass function is given in light gray. After training GMAN, three discriminators converge to distinct local optima which implicitly define distributions over the data (red, blue, yellow). Each discriminator may specialize in discriminating a region of the data space (placing more diffuse mass in other regions). Averaging over the three discriminators results in the distribution in black, which we expect has higher likelihood under reasonable assumptions on the structure of the true distribution.

In practice, this degenerate result is avoided by employing learners with limited capacity, but we might better accomplish this by simultaneously training a variety of limited capacity discriminators. With this approach, we might obtain a diverse set of seemingly tenable hypotheses for the true  $p_{data}(x)$ . Averaging over these multiple locally optimal discriminators increases the entropy of  $\tilde{p}_{data}(x)$  by diffusing the probability mass over the data space (see Figure 2 for an example).

# 3.4 AUTOMATING REGULATION

The problem of keeping the discriminator and generator in balance has been widely recognized in previous work with GANs. Issues with unstable dynamics, oscillatory behavior, and generator collapse are not uncommon. In addition, the discriminator is often times able to achieve a high degree of classification accuracy (producing a single scalar) before the generator has made sufficient progress on the arguably more difficult generative task (producing a high dimensional sample). Salimans et al. (2016) suggested label smoothing to reduce the vulnerability of the generator to a relatively superior discriminator. Here, we explore an approach that enables the generator to automatically temper the performance of the discriminator when necessary, but still encourages the generator to challenge itself against more accurate adversaries. Specifically, we augment the generator objective:

$$
\min  _ {G, \lambda > 0} F _ {G} (V _ {i}) - f (\lambda) \tag {7}
$$

where  $f(\lambda)$  is monotonically increasing in  $\lambda$  which appears in the softmax equations, (3)-(5). In experiments, we simply set  $f(\lambda) = c\lambda$  with  $c$  a constant (e.g., 0.001). The generator is incentivized to increase  $\lambda$  to reduce its objective at the expense of competing against the best available adversary  $D^{*}$  (see Appendix A.4).

# 4 EVALUATION

Evaluating GANs is still an open problem. In their original work, Goodfellow et al. (2014) report log likelihood estimates from Gaussian Parzen windows, which they admit, has high variance and does not perform well in high dimensional settings. Salimans et al. (2016) recommend an Inception score, however, it assumes labels exist for the dataset. Recently, Im et al. (2016) introduced the Generative Adversarial Metric (GAM) for making pairwise comparisons between independently trained GAN models. The core idea behind their approach is given two generator, discriminator pairs  $(G_{1},D_{1})$  and  $(G_{2},D_{2})$ , we should be able to learn their relative performance by judging each generator under the opponent's discriminator.

# 4.1 METRIC

In GMAN, the opponent may have multiple discriminators, which makes it unclear how to perform the swaps needed for GAM. We introduce a variant of GAM, the generative multi-adversarial metric (GMAM), that is amenable to training with multiple discriminators,

$$
\mathrm {G M A M} = \log \left(\frac {F _ {G _ {b}} ^ {a} \left(V _ {i} ^ {a}\right)}{F _ {G _ {a}} ^ {a} \left(V _ {i} ^ {a}\right)} / \frac {F _ {G _ {a}} ^ {a} \left(V _ {i} ^ {b}\right)}{F _ {G _ {b}} ^ {b} \left(V _ {i} ^ {b}\right)}\right). \tag {8}
$$

where  $a$  and  $b$  refer to the two GMAN variants (see 3 for notation  $F_{G}(V_{i})$ ). The idea here is similar. If  $G_{2}$  performs better than  $G_{1}$  with respect to both  $D_{1}$  and  $D_{2}$ , then  $\mathrm{GMAM} > 0$  (remember  $V \leq 0$  always). If  $G_{1}$  performs better in both cases,  $\mathrm{GMAM} < 0$ , otherwise, the result is indeterminate.

# 4.2 EXPERIMENTS

We evaluate the aforementioned variations of GMAN on a variety of image generation tasks: MNIST (LeCun et al. (1998)), CIFAR-10 (Krizhevsky (2009)) and CelebA (Liu et al. (2015)). We focus on rates of convergence to steady state along with quality of the steady state generator according to the GMAM metric. To summarize, loosely in order of increasing discriminator leniency, we compare

- F-boost: A single AdaBoost.OL-boosted discriminator (see Appendix A.5).  
- P-boost:  $D_{i}$  is trained according to AdaBoost.OL. A max over the weak learner losses is presented to the generator instead of the boosted prediction (see Appendix A.5).  
- GMAN-max:  $\max \{V_i\}$  is presented to the generator.  
- GAN: Standard GAN with a single discriminator (see Appendix A.1.3).  
- mod-GAN: GAN with modified objective (generator minimizes -  $\log (D(G(z)))$  
- GMAN- $\lambda$ : GMAN with  $F := \text{arithmetic softmax}$  with parameter  $\lambda$ .  
-  $\mathrm{GMAN}^*$ : The arithmetic softmax is controlled by the generator through  $\lambda$ .

All generator and discriminator models are deep (de)convolutional networks (Radford et al. (2015)), and aside from the boosted variants, all are trained with Adam (Kingma & Ba (2014)) and batch normalization (Ioffe & Szegedy (2015)). Discriminators convert the real-valued outputs of their networks to probabilities with squashed-sigmoids to prevent saturating logarithms in the minimax objective  $(\epsilon + \frac{1 - 2\epsilon}{1 + e^{-z}})$ . See Appendix A.6 for further details. We test GMAN systems with  $N = \{2, 5\}$  discriminators. We maintain discriminator diversity by varying dropout and network depth.

# 4.2.1 MNIST

Figures 3 and 4 reveal that increasing the number of discriminators reduces the number of iterations to steady-state by  $2\mathrm{x}$  on MNIST; increasing  $N$  (the size of the discriminator ensemble) also has the added benefit of reducing the variance of the game dynamics.

Figure 5 corroborates this conclusion with recognizable digits appearing approximately an epoch before the single discriminator run; digits at steady-state appear slightly sharper as well.

Our GMAM metric (see Table 1) agrees with the relative quality of images in Figure 5 with  $\mathrm{GMAN}^*$  achieving the best overall performance.

<table><tr><td></td><td>Score</td><td>Variant</td><td>GMAN*</td><td>GMAN-0</td><td>GMAN-max</td><td>mod-GAN</td></tr><tr><td rowspan="4">↑</td><td>0.127</td><td>GMAN*</td><td>-</td><td>-0.018 ± 0.011</td><td>-0.031 ± 0.016</td><td>-0.095 ± 0.038</td></tr><tr><td>0.007</td><td>GMAN-0</td><td>0.021 ± 0.007</td><td>-</td><td>-0.017 ± 0.017</td><td>-0.019 ± 0.029</td></tr><tr><td>-0.034</td><td>GMAN-max</td><td>0.024 ± 0.021</td><td>0.009 ± 0.012</td><td>-</td><td>-0.008 ± 0.033</td></tr><tr><td>-0.122</td><td>mod-GAN</td><td>0.082 ± 0.034</td><td>0.016 ± 0.025</td><td>0.014 ± 0.015</td><td>-</td></tr></table>

Table 1: Pairwise GMAM metric means with stdev for select models on MNIST. For each column, a positive GMAM indicates better performance relative to the row opponent; negative implies worse. Scores are obtained by summing each variant's column.

![](images/67edde231163acce1d798800ff478787924e5b65d9098d8d5dd5fa6645044de3.jpg)  
Figure 3: Generator objective averaged over 5 training runs on MNIST dataset. Increasing the number of discriminators accelerates convergence to steady state (solid line) and reduces variance (filled shadow).

![](images/a7cd67fbbd9a08ea3e85cc30db2912a6b1481439f8de38cc335c2d22df20a046.jpg)  
Figure 4: Cumulative  $stdev$  of the generator objective over a sliding window of 500 iterations. Lower values indicate a more steady-state.  $\mathrm{GMAN^{*}}$  with  $N = 5$  achieves steady-state at  $\approx 2\mathrm{x}$  speed of GAN  $(N = 1)$ .

![](images/8951918d28fdc4c1a6ab3c296afebf64ead46db071bbc1fd7bee82bdeaab5c6a.jpg)  
Figure 5: Comparison of image quality across epochs for  $N = \{1,2,5\}$  using GMAN-0 on MNIST.

![](images/f545733749329327e129823be463daf913f53f638d92711640c33833a33c8110.jpg)  
Figure 6 reveals  $\mathrm{GMAN^{*}}$  's attempt to regulate the difficulty of the game to accelerate learning. Figure 7 displays the GMAM scores comparing fixed  $\lambda$  's to the variable  $\lambda$  controlled by  $\mathrm{GMAN^{*}}$ .  
Figure 6: GMAN* regulates difficulty of the game by adjusting  $\lambda$ . Initially,  $G$  reduces  $\lambda$  to ease learning and then gradually increases  $\lambda$  for a more challenging learning environment.  
Figure 7: Pairwise  $\frac{\mathrm{GMAM}}{\mathrm{stdev}\, \mathrm{of}\, \mathrm{GMAM}}$  for GMAN- $\lambda$  and GMAN- $\lambda^{*}$  over 5 runs on MNIST.

<table><tr><td></td><td>Score</td><td>λ
(N=5)</td><td>λ*</td><td>λ=1</td><td>λ=0</td></tr><tr><td rowspan="3">Better↑</td><td>0.028</td><td>λ*</td><td>-</td><td>-0.007 ±0.009</td><td>-0.018 ±0.012</td></tr><tr><td>0.001</td><td>λ=1</td><td>0.008 ±0.008</td><td>-</td><td>-0.007 ±0.010</td></tr><tr><td>-0.025</td><td>λ=0</td><td>0.020 ±0.007</td><td>0.008 ±0.009</td><td>-</td></tr></table>

# 4.2.2 CELEBA & CIFAR-10

We see similar accelerated convergence behavior for the CelebA dataset in Figure 8.

![](images/805f41405f3b5dd88f63f93b4d2ae55f1a7d12bf744af63849cbc1e3b82bdb24.jpg)  
Figure 8: Image quality improvement across number of generators at same number of iterations for GMAN-0 on CelebA.

Figure 9 displays images generated by GMAN-0 on CIFAR-10. See Appendix A.1 for more results.

![](images/8e9f8f6b869c1c86cb02fc5a9113326ea314edd4b913a920d5d9874d4dfe9f5f.jpg)  
Figure 9: Images generated by GMAN-0 on the CIFAR-10 dataset.

We also found that GMAN is robust to the unimodal behavior seen in GANs where the generator always emits the same point. We believe this is due to the fact that, in GMAN, the generator must appease a diverse set of discriminators in each minibatch. Emitting a single point will score well for a single discriminator at the large expense of the rest of the discriminators. Currently, the most popular approach for patching this failure mode is minibatch discrimination which is quadratic in batchsize. GMAN, on the other hand, is linear in batch size. We leave validation for future work.

# 5 CONCLUSION

We introduced multiple discriminators into the GAN framework and explored discriminator roles ranging from a formidable adversary to a forgiving teacher. We found that allowing the generator to automatically tune its learning schedule (GMAN*) outperformed GANs with a single discriminator on a variety of image generation tasks. In general, GMAN variants achieved faster convergence to a higher quality steady state as measured by a GAM-type metric (GMAM). In addition, GMAN makes using the original GAN objective possible by increasing the odds of the generator receiving constructive feedback.

In future work, we will look at more sophisticated mechanisms for letting the generator control the game as well as other ways to ensure diversity among the discriminators. Introducing multiple generators is conceptually an obvious next step, however, we expect difficulties to arise from more complex game dynamics. For this reason, game theory and game design will likely be important.

# ACKNOWLEDGMENTS

We acknowledge helpful conversations with Stefan Dernbach, Archan Ray, Luke Vilnis, Ben Turtel, Stephen Giguere, Rajarshi Das, and Subhransu Maji.

# BIBLIOGRAPHY

Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467, 2016.  
Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, and Mario Marchand. Domain-adversarial neural networks. arXiv preprint arXiv:1412.4446, 2014.  
J Andrew Bagnell. Robust supervised learning. In Proceedings Of The National Conference On Artificial Intelligence, volume 20, pp. 714. Menlo Park, CA; Cambridge, MA; London; AAAI Press; MIT Press; 1999, 2005.  
Alina Beygelzimer, Satyen Kale, and Haipeng Luo. Optimal and adaptive algorithms for online boosting. arXiv preprint arXiv:1502.02651, 2015.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Info-gan: Interpretable representation learning by information maximizing generative adversarial nets. arXiv preprint arXiv:1606.03657, 2016.  
Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. In Advances in neural information processing systems, pp. 1486-1494, 2015.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. arXiv preprint arXiv:1606.00704, 2016.  
Harrison Edwards and Amos Storkey. Censoring representations with an adversary. arXiv preprint arXiv:1511.05897, 2015.  
Jon Gauthier. Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014, 2014.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. arXiv preprint arXiv:1606.03476, 2016.  
Daniel Jiwoong Im, Chris Dongjoo Kim, Hui Jiang, and Roland Memisevic. Generating images with recurrent adversarial networks. arXiv preprint arXiv:1602.05110, 2016.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. Master's Thesis, 2009.  
Yann LeCun, Corinna Cortes, and Christopher JC Burges. The mnist database of handwritten digits, 1998.  
Yujia Li, Kevin Swersky, and Richard Zemel. Generative moment matching networks. In International Conference on Machine Learning, pp. 1718-1727, 2015.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.  
Alireza Makhzani, Jonathon Shlens, Navdeep Jaitly, and Ian Goodfellow. Adversarial autoencoders. arXiv preprint arXiv:1511.05644, 2015.

Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. arXiv preprint arXiv:1606.00709, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Siamak Ravanbakhsh, Francois Lanusse, Rachel Mandelbaum, Jeff Schneider, and Barnabas Poczos. Enabling dark energy science with deep generative models of galaxy images. arXiv preprint arXiv:1609.05796, 2016.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. arXiv preprint arXiv:1606.03498, 2016.  
Jürgen Schmidhuber. Learning factorial codes by predictability minimization. *Neural Computation*, 4(6):863-879, 1992.  
Jost Tobias Springenberg. Unsupervised and semi-supervised learning with categorical generative adversarial networks. arXiv preprint arXiv:1511.06390, 2015.  
Masatoshi Uehara, Issei Sato, Masahiro Suzuki, Kotaro Nakayama, and Yutaka Matsuo. Generative adversarial nets from a density ratio estimation perspective. arXiv preprint arXiv:1610.02920, 2016.  
Donggeun Yoo, Namil Kim, Sunggyun Park, Anthony S Paek, and In So Kweon. Pixel-level domain transfer. arXiv preprint arXiv:1603.07442, 2016.  
Matthew D Zeiler, Dilip Krishnan, Graham W Taylor, and Rob Fergus. Deconvolutional networks. In Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on, pp. 2528-2535. IEEE, 2010.  
Junbo Zhao, Michael Mathieu, and Yann LeCun. Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126, 2016.
