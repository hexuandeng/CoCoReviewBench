# ACCELERATING HEP SIMULATIONS WITH NEURAL IMPORTANCE SAMPLING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Virtually all high-energy-physics (HEP) simulations for the LHC rely on Monte Carlo using importance sampling by means of the VEGAS algorithm. However, complex high-precision calculations have become a challenge for the standard toolbox. As a result, there has been keen interest in HEP for modern machine learning to power adaptive sampling. Despite previous work proving that normalizing-flow-powered neural importance sampling (NIS) sometimes outperforms VEGAS, existing research has still left major questions open, which we intend to solve by introducing ZüNIS, a fully automated NIS library. We first show how to extend the original formulation of NIS to reuse samples over multiple gradient steps, yielding a significant improvement for slow functions. We then benchmark ZüNIS over a range of problems and show high performance with limited fine-tuning. The library can be used by non-experts with minimal effort, which is crucial to become a mature tool for the wider HEP public.

# 1 INTRODUCTION

High-Energy-Physics (HEP) simulations are at the heart of the Large Hadron Collider (LHC) program for studying the fundamental laws of nature. Most HEP predictions are expressed as expectation values, evaluated numerically as Monte Carlo (MC) integrals. This permits both the integration of the very complex functions and the reproduction of the data selection process by experiments.

Most HEP simulations tools (Alwall et al., 2014) perform MC integrals using importance sampling, which allows to adaptively sample points to speed up convergence while keeping independent and identically distributed samples, crucial to reproduce experimental analyses which can only ingest uniformly-weighted data, typically produced by rejection sampling (see appendix A).

The most popular tool to optimize importance sampling is by far the VEGAS algorithm (Lepage, 1980; 2021), which fights the curse of dimensionality by assuming no correlations between the variables. While this is rarely the case in general, a good understanding of the integrand function can help significantly. Indeed optimized parametrizations using multichannelling (Kleiss et al., 1986; Ohl, 1999; Kleiss & Pittau, 1994) have become bread-and-butter tools for HEP event generation simulators, with good success for leading-order (LO) calculations. However, as simulations get more complex, either by having more complex final states or by including higher orders in perturbation theory, performance degrades fast.

There is much room for investing computational time into improving sampling (ATLAS Collaboration, 2020): modern HEP theoretical calculations are taking epic proportions and can require hours for a single function evaluation (Jones, 2018). Furthermore, unweighting samples can be extremely inefficient, with upwards of  $90\%$  sampled points discarded (Foundation et al., 2020). More powerful importance sampling algorithms would therefore be a welcome improvement (Buckley, 2020; WG et al., 2021).

First attempts to use machine learning (ML) to address this challenge explored using classical neural networks to sample (Bendavid, 2017; Klimek & Perelstein, 2020; Chen et al., 2021) but typically suffer from excessive computational costs. Another avenue of research has been to leverage generative models successful in other fields such as generative adversarial networks (Butter et al., 2019; Di Sipio et al., 2019; Butter et al., 2020; Ahdida et al., 2019; Hashemi et al., 2019; Carrazza & Dreyer, 2019). While such approaches do improve sampling speed by a large factor, they have ma

jor limitations. In particular, they have no theoretical guarantees of providing a correct answer on average (Matchev et al., 2021) and poor control of uncertainties.

To avoid these disadvantages, our work exploits Neural Importance Sampling (NIS) (Müller et al., 2019; Zheng & Zwicker, 2019), which relies on normalizing flows and has strong theoretical guarantees.

A number of exploratory papers have been published on using NIS for LHC simulations (Gao et al., 2020b; Bothmann et al., 2020; Gao et al., 2020a), as well as closely related variations (Bellagente et al., 2021; Stienen & Verheyen, 2021), but most studies have focused on preliminary investigation of performance without much concern for the practical usability of the method. Indeed, training requires function evaluations, which we are trying to minimize and data-efficiency training is therefore an important but under-appreciated concern. Furthermore, few authors have provided usable open source code, making the adoption of the technique in the HEP community difficult.

As a result, we introduce ZUNIS, a PyTorch-based library providing robust and usable NIS, usable by non-experts. We show that NIS is not only performant but can be formulated in a way that maximizes the use of a given function-evaluation budget. Not only does this optimize training, but this allows for cheap hyperparameter tuning by reusing sampled data. ZUNIS is flexible and extendable and provides multiple training strategies so that experts can fine-tune it to their needs, thanks to a detailed documentation.

# 2 BACKGROUND

# 2.1 IMPORTANCE SAMPLING AS AN OPTIMIZATION PROBLEM

Importance sampling relies on the interpretation of integrals as expectation values. Indeed, let us consider an integral over a finite volume:

$$
I = \int_ {\Omega} d x f (x), \quad \text {w h e r e} V (\Omega) = \int_ {\Omega} d x \text {i s f i n i t e .} \tag {1}
$$

Let  $p$  be a strictly positive probability distribution over  $\Omega$ , we can re-express our integral as an expectation value

$$
I = \int_ {\Omega} p (x) d x \frac {f (x)}{p (x)} = \underset {x \sim p} {\mathbb {E}} \left(\frac {f (x)}{p (x)}\right), \tag {2}
$$

which motivates the introduction of a new estimator:

$$
\hat {I} _ {N} ^ {p} = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {f \left(X _ {i}\right)}{p \left(X _ {i}\right)}, \quad X _ {i} \sim p, \tag {3}
$$

whose mean is indeed  $I$  and whose standard deviation is  $\frac{\sigma(f, p)}{\sqrt{N}}$ , where  $\sigma(f, p)$  is the standard deviation of  $f(X) / p(X)$  for  $X \sim p$ :

$$
\sigma^ {2} (f, p) = \underset {x \sim p} {\mathbb {E}} \left(\left(\frac {f (x)}{p (x)}\right) ^ {2}\right) - I ^ {2}. \tag {4}
$$

The problem statement of importance sampling is to find the probability distribution function  $p$  that minimizes the variance of our estimator for a given  $N$ .

One can show that the minimum is obtained for  $p(x) \propto |f(x)|$  (see appendix E). This exact solution is however of no use in practice because there is in general no efficient way to sample from  $|f|$ .

Instead of relying on the exact solution, we therefore need to find an approximate solution from a family of functions from which we can efficiently sample. All importance sampling algorithms provide such a family and an optimization procedure to select the best candidate. In Neural Importance Sampling, we rely on Normalizing Flows, which we can optimize using stochastic gradient descent, as we shall discuss in the following section.

# 2.2 NORMALIZING FLOWS AND COUPLING CELLS

Normalizing flows (Tabak & Vanden-Eijnden, 2010; Tabak & Turner, 2013; Rippel & Adams, 2013; Rezende & Mohamed, 2015) provide a way to generate complex probability distribution functions from simpler ones using parametric changes of variables that can be learned to approximate a target distribution. As such, normalizing flows are diffeomorphisms: invertible, (nearly-everywhere) differentiable mappings with a differentiable inverse.

Let us start by restating the basics of the approach, and consider open sets  $A, \Omega \subset R^n$ . A diffeomorphism  $T: A \to \Omega$  not only maps points through the relation  $x = T(u)$  for  $u \in A$ ,  $x \in \Omega$ , but also probability distribution functions: for every PDF  $p$  over  $A$ , there is a PDF  $q$  on  $\Omega$  such that, if  $u \sim p(u)$ ,  $T(u) = x \sim q(x)$ . The induced PDF  $q$  is expressed as

$$
q (x = T (u)) = p (u) \left| J _ {T} (u) \right| ^ {- 1}, \tag {5}
$$

where  $J_{T}$  is the Jacobian determinant of  $T$ :

$$
J _ {T} (u) = \det  \frac {\partial T _ {i}}{\partial u _ {j}} (u). \tag {6}
$$

In the world of machine learning,  $T$  is called a normalizing flow and is typically part of a parametric family of diffeomorphisms  $(T(\cdot ,\theta))$  such that gradients  $\nabla_{\theta}J_{T}$  are tractable.

A necessary condition for this approach is the existence of families of diffeomorphisms with tractable Jacobians that can be optimized over. Coupling cell mappings perfectly satisfy this requirement (Dinh et al., 2015; 2017; Müller et al., 2018): they are neural-network-parametrized bijections whose Jacobian factor can be obtained analytically without backpropagation or expensive determinant calculation. As such, they provide a good candidate for importance sampling as long as they can be trained to learn an unnormalized target function, which is exactly what neural importance sampling proposes.

# 2.3 NEURAL IMPORTANCE SAMPLING

Neural importance sampling was introduced in the context of computer graphics (Müller et al., 2018) and proposes to use normalizing flows as a family of probability distributions over which to solve the minimization problem of importance sampling.

$$
\mathcal {L} (\theta) = \int_ {\Omega} \frac {f ^ {2} (x)}{p (x , \theta)}. \tag {7}
$$

Of course, to actually do so, one needs to find a way to explicitly evaluate  $\mathcal{L}(\theta)$  and the original neural importance sampling approach proposes to approximate it using importance sampling. One needs to be careful that the gradient of the estimator of the loss need not be the estimator of the gradient of the loss. The gradient of the loss can be expressed as

$$
\nabla_ {\theta} \mathcal {L} (\theta) = - \int_ {\Omega} \frac {f ^ {2} (x)}{p (x , \theta)} \nabla_ {\theta} \log p (x, \theta), \tag {8}
$$

for which an estimator is proposed as

$$
\widehat {\nabla} _ {\theta} \mathcal {L} (\theta) = - \sum_ {i = 0} ^ {N} \left(\frac {f \left(X _ {i}\right)}{p \left(X _ {i} , \theta\right)}\right) ^ {2} \nabla_ {\theta} \log p \left(X _ {i}, \theta\right), \quad X _ {i} \sim p. \tag {9}
$$

The authors also observed that other loss functions are possible which share the same global minimum as the variance based loss: for example, the Kullback-Leibler divergence  $D_{KL}$  between two functions is also minimized when they are equal. Such alternative loss functions are not guaranteed to work for importance sampling, but they prove quite successful in practice. After training to minimize the loss estimator of eq. (9), the normalizing flows provide a tractable probability distribution function from which to sample points and estimate the integral.

# 3 THE ZUNIS FRAMEWORK: CONCEPTS AND ALGORITHMS

In this section we give a high-level overview of the organization of the ZUNIS library and the algorithms it uses for importance sampling. The major conceptual innovation we provide in ZUNIS is a more flexible and data-efficient way of training normalizing flows in the context of importance sampling. This relies on a more rigorous formulation of the connection between the theoretical expression of ideal loss functions in terms of integrals and their practical realizations as random estimators than in previous literature. We describe this improvement in section 3.2.

# 3.1 THE ZUNIS LIBRARY

On the practical side, ZUNIS is a PyTorch-based library which implements many ideas formulated in previous work but organizes them in the form of a modular software library with an easy-to-use interface and well-defined building blocks. We believe this structure will help non-specialist use it without understanding all the nuts and bolts, while experts can easily add new features to responds to their needs. The ZUNIS library relies on three layers of abstractions which steer the different aspects of using normalizing flows to learn probability distributions from un-normalized functions and compute their integrals:

- Flows, which implement a bijective mapping which transforms points and computes the corresponding Jacobian are described in appendix F.1  
- Trainers, which provide the infrastructure to perform training steps and sample from flow models are described in appendix F.2  
- Integrators, which use trainers to steer the training of a model and compute integrals are described in appendix F.3

# 3.2 EFFICIENT TRAINING FOR IMPORTANCE SAMPLING

In this section, we describe how we train probability distributions within ZUNIS using gradient-based optimizers. We want to find which member  $p(x,\theta)$  of a parametric family of probability distribution functions minimizes eq. (7), which is also an integral we need to estimate. We will therefore define an auxiliary probability distribution function  $q(x)$ , independent from  $\theta$ , from which we sample to estimate or loss function:

$$
\int d x \frac {f (x) ^ {2}}{p (x , \theta)} = \underset {x \sim q} {\mathbb {E}} \frac {f (x) ^ {2}}{p (x , \theta) q (x)}. \tag {10}
$$

This is the basis for the general method we use for training probability distributions within ZUNIS, described in algorithm 1. Because the sampling distribution is separated from the model to train, the same point sample can be reused for multiple training steps, which is not possible when using eq. (9). This is particularly important for high-precision particle physics predictions that involve high-order perturbative calculations or complex detector simulations because function evaluations can be extremely costly.

From which distribution  $q$  should we sample for our loss estimation? Without any information on our function, a simple choice is to sample uniformly. Another possible approach is to use the information gathered during training, while respecting the constraint that  $\nabla_{\theta}q(x) = 0$ . To this end, we can take inspiration from deep  $Q$ -learning in which two copies of the models are used: one is trained using gradient-based optimization and is used to evaluate PDF values while the other is used for sampling and has its parameters frozen except for infrequent updates.

# 4 PERFORMANCE ASSESSMENT

In this section, we evaluate ZUNIS on a variety of test functions to assess its performance and compare it to the commonly used VEGAS algorithm (Peter Lepage, 1978; Ohl, 1999). We first produce a few low dimensional examples for illustrative purposes, then move on to integrating parametric functions in various dimensions and finally evaluate performance on particle scattering matrix elements.

# 4.1 LOW-DIMENSIONAL EXAMPLES

Let us start by illustrating the effectiveness of ZUNIS in a low dimensional setting where we can readily visualize results. We define three functions over the two dimensional hypercube:

$$
f _ {\text {c a m e l}} (x) = \exp \left(- \left(\frac {x - \mu_ {1}}{\sigma}\right) ^ {2}\right) + \exp \left(- \left(\frac {x - \mu_ {2}}{\sigma}\right) ^ {2}\right), \tag {11}
$$

$$
f _ {\varnothing} (x) = \min  \left[ 1, \exp \left(- \left(\frac {| x | - r}{\sigma_ {\varnothing}}\right) ^ {2}\right) + \exp \left(- \left(\frac {a \cdot x}{\sigma_ {\varnothing}}\right) ^ {2}\right) \right] \tag {12}
$$

$$
f _ {\sin} (x) = \cos (k \cdot x) ^ {2}, \tag {13}
$$

to which we refer respectively as the camel, slashed circle and sinusoidal target functions. We set their parameters as follows

$$
\mu_ {1} = \binom {0. 2 5} {0. 2 5}, \mu_ {2} = \binom {0. 7 5} {0. 7 5}, a = \binom {1} {- 1}, k = \binom {6} {6}, \sigma = 0. 1, \sigma_ {\emptyset} = 0. 1, r = 0. 3 \tag {14}
$$

We chose these functions because they illustrate different failure modes of the VEGAS algorithm (see appendix C).

We ran the ZUNIS Integrator with default arguments over ten repetitions for each function and report the performance of the trained integral estimator compared to a flat estimator and to VEGAS in table 1. Overall, ZUNIS Integrators learn to sample from their target function extremely well: we outperform VEGAS by a factor 100 for the camel and the slashed circle functions and a factor 30 for the sinusoidal function and VEGAS itself provides no advantage over uniform sampling for the latter two.

Table 1: Variance reduction (high is good) for the camel, slashed circle and sinusoidal functions compared to uniform sampling and to VEGAS over 10 repetitions.  

<table><tr><td>Variance Reduction</td><td>Camel</td><td>Slashed Circle</td><td>Sinusoidal</td></tr><tr><td>vs. uniform</td><td>1.8 ± 0.4 × 103</td><td>8.9 ± 0.9 × 101</td><td>2.0 ± 0.5 × 102</td></tr><tr><td>vs. VEGAS</td><td>7.0 ± 1.4 × 102</td><td>8.8 ± 0.9 × 101</td><td>1.6 ± 0.5 × 102</td></tr></table>

We further illustrate the performance of our trained models by comparing the target functions and density histogram for points sampled from the normalizing flows in fig. 1, which shows great qualitative agreement.

# 4.2 SYSTEMATIC BENCHMARKS

Let us now take a more systematic approach to benchmarking ZUNIS. We compare ZUNIS Integrators against uniform integration and VEGAS using the following three metrics:

- integrand variance: this is a measure of the quality of the learned random variable sampler (section 2.1)  
- unweighting efficiency: this is a measure of how efficient the learned sampler is for generating i.i.d data with rejection sampling as defined in appendix A  
- training time: this is a measure of wall-clock time need to complete training $^2$ .  
- sampling time: a measure of wall-clock time per sampled point².

For this experiment, we focus on the camel function defined in eq. (11) and scan a variety of settings and dimensions. We always set the two peaks on the hyperdiagonal and span from 2 to 32 dimensions over function variances between  $10^{-2}$  and  $10^{2}$  as shown in table 3.

![](images/8de28222e3be0bc194c247e52e90e3ced068b06f4fb8f324bca91a6f28f3d7f8.jpg)

![](images/ff7e1f3a49722987c3561ea1bd121ebd582673d6752a3aa1def5a6a14153c344.jpg)

![](images/02d317154dbe33eaa698ec716a6c0de10a1bddea36b4c829f4220c42f0cf0823.jpg)

![](images/d591ad734379e72dca469d2d7813f4ab57c7f8861c8e34a837765c6ff4ab468d.jpg)

![](images/ca962fe4aa53b698f95ccdd19317cb82e918130ba8ecf06408317f10b75778eb.jpg)

![](images/f6fdbdcda809483fa6363d856682b954354ec4177b4f3d3f76566b6e2212f288.jpg)

![](images/5bd5bc72c6c8b03a62991d2d103926b143dfe73c996fec6f4104267fb8cd198f.jpg)  
(a) Camel function

![](images/1435cbd549a2ddcffa72a084b6809ddd803d89a4ee17a9eb9ba703ce86a961ed.jpg)  
(b) Sinusoidal function

![](images/b3e7f92f746e662f23990957fff7251b4119d9b3494c301b31b22119a4c97952.jpg)  
Figure 1: Comparison between target functions and point sampling densities for 1a the camel function, 1b the sinusoidal function, 1c the slashed circle function. Supplementary fig. 7 shows how points are mapped from latent to target space.  
(c) Slashed circle function

We first test ZUNIS by employing the default configuration over the 35 camel functions defined in table 3 to show the solid improvement Integrators provide over alternative approaches without any parameter tuning. Except in the low variance limit, ZUNIS can reduce the required number of points sampled to attain a given precision on integral estimates, attaining speed-ups of up to  $\times 1000$  both compared to uniform sampling and VEGAS-based importance sampling, as shown in fig. 2a-2b and table 4. Unweighting efficiencies are also boosted significantly, although more mildly than variances, as shown in fig. 2c-2d, which we could attribute to PDF underestimation in regions with low point density; the nature of the veto algorithm makes it very sensitive to a few bad behaving points in the whole dataset.

ZUNIS does not, however, outclass VEGAS on all metrics by far: as shown in fig. 2, training is a few hundred times slower than VEGAS and sampling is 10-50 times slower, all while ZUNIS runs on GPUs. This is to be expected given the much increased computational complexity of normalizing flows compared to the VEGAS algorithm. As such, ZUNIS is not a general replacement for VEGAS, but provides a clear advantage for integrating time-intensive functions, where sampling is a negligible overhead, such as precision high-energy-physics simulations.

We have shown that ZUNIS is a very performant importance sampling and event generation tool and provides significant improvements over existing tools, while requiring little fine tuning from users. Another key result is that the new approach to training we introduced in section 3.2 has a large positive impact on performance. Indeed, as we show in fig. 2, re-using samples for training

![](images/e298cf690bb6d8cab138924c648432fb4036ad4f819e63ff4ea2717d426ef908.jpg)  
(a)

![](images/fce23b01d4a770c2aea1ae89a01b340a7bf2c0a6f382f4361b02528bb318e1af.jpg)

![](images/b60e4a323ec0e94774d3740b49d98dca77fb5f3caf428667a43a457dd7a5c549.jpg)  
(c)

![](images/f75c60322dc982df7999a78ba70ec9dc8d1ab5b84b8767fff3d1eb583db262d0.jpg)  
(b)  
(d)

![](images/08e6189c990cc82250cdc20df6a1ea554de88f92413fc2776b7551f41b2c805c.jpg)  
Figure 2: Benchmarking ZUNIS against uniform sampling and VEGAS with default settings. In (2a-2b), we show the sampling speed-up (ratio of integrand variance) as a function of the relative standard deviation of the integrand, while we show the unweighting speed-up (ratio of unweighting efficiencies) in (2c-2d).  
Figure 3: Comparison of the training and sampling speed of ZUNIS and VEGAS. As can be expected, ZUNIS is much slower than VEGAS, both for training and sampling, although larger batch sizes can better leverage the power of hardware accelerators.  
(a)

![](images/309a9564d334fc1c07d433d55c75de951cd4f68d20f88c01218365d0ce525ef7.jpg)  
(b)

over multiple epochs provides a 2- to 10-fold increase in convergence speed, making training much more data-efficient.

# 4.3 MADGRAPH CROSS SECTION INTEGRALS

Cross-sections are integrals of quantum transition matrix elements for a scattering process such as a LHC collision and express the probability that specific particles are produced. Matrix elements themselves are un-normalized probability distributions for the configuration of the outgoing particles: it is therefore both valuable to integrate them to know the overall frequency of a given

![](images/fcc3832da5f696f6b3629815a9e87712a0e303bc72bed5fa7a93ef2968e4131c.jpg)  
(a)

![](images/c9c1949fee2919111170e375fa74b47b1ea2a39772bc2d721ef6728137e66844.jpg)  
Figure 4: Effect of repeatedly training on the same sample of points over multiple epochs. For all settings, there is a large improvement when going from one to moderate epoch counts, with a peak around 5-10. Larger number of epochs lead to overfitting, which impacts performance negatively.  
(b)

scattering process, and to sample from them to understand how particles will be spatially distributed as they fly off the collision point.

We study the performance of ZUNIS in comparison to VEGAS by studying three simple processes at leading order in perturbation theory,  $e^{-} \mu \rightarrow e^{-} \mu$  via  $Z$ ,  $dd \rightarrow dd$  via  $Z$  and  $uc \rightarrow ucg$  (with 3-jet cuts based on  $\Delta R$ ), see table 2 and fig. 5. We use the first process as a very easy reference while the two other, quark-initiated processes are used to illustrate specific points. Indeed, both feature narrow regions of their integration space with large enhancements, due respectively to  $Z$ -boson resonances and infra-red divergences.

![](images/cca384a1fe8dcd9617eb6e661d58999aea9762c941b2199872720be31421b025.jpg)  
(a)  
Figure 5: Sample Feynman Diagrams for  $e^{-}\mu \rightarrow e^{-}\mu$  via  $Z$ ,  $d\bar{d} \rightarrow d\bar{d}$  via  $Z$  and  $uc \rightarrow ucg$ .

![](images/441b34d6eb7f497a67f385fa018ad7e36509f2412332b1fdb72ec3a5125d4a72.jpg)  
(b)

![](images/f9219617d95acca9a04d9f3d3103fc59df60ab3f621732fa962cdc4ae43bce9c.jpg)  
(c)

Table 2: Comparison of the three test processes.  

<table><tr><td></td><td>e-μ → e-μ via Z</td><td>dd → dd via Z</td><td>uc → ucg</td></tr><tr><td>dimensions</td><td>2</td><td>4</td><td>7</td></tr><tr><td>normalized standard deviation</td><td>1.45 × 10-2</td><td>6.57 × 10-2</td><td>0.96</td></tr></table>

We evaluate the matrix elements for these three processes by using the FORTRAN standalone interface of MADGRAPH5_AMC@NLO (Alwall et al., 2014). The two hadronic processes are convolved with parton-distribution functions from LHPDF6 (Buckley et al., 2015). We parametrize phase space (the particle configuration space) using the RAMBO on diet algorithm (Plätzer, 2013) implemented for PyTorch in TORCHPS (Götz, 2021).

We report benchmark results in 6, in which we trained over 500,000 points for each process using near-default configuration, scanning only over variance and Kullback-Leibler losses.

We confirm our previous finding that processes with very small variance like  $e^{-}\mu \rightarrow e^{-}\mu$ , no integral convergence acceleration is achieved, but find however that ZUNIS improves unweighting even in this situation. The two hadronic processes illustrate features we have observed when integrating

![](images/e0bd74d358e12363d5577c6c92afb182bd216ab7ecfb6592846027a2dd8df492.jpg)  
(a)

![](images/73fb5225aa75123815e08c6f483635e9bd28684077271e355b5877dcc62b17f4.jpg)  
Figure 6: Average performance of ZUNIS over 20 runs relative to VEGAS, measured by the relative speed-up in 6a and by the relative unweighting efficiency in 6b.  
(b)

cross sections: there is a large stochasticity of training and different processes are optimized by different loss function choices<sup>3</sup>.

The performance of  $dd \to dd$  shows nice improvement with ZUNIS while that of  $uc \to ucg$  is more limited. This is actually quite natural to understand when one compares performance against uniform sampling (see appendix D.3): it is in fact VEGAS, which performs significantly better on  $uc \to ucg$  compared to  $dd \to dd$  because the parametrization of RAMBO is based on splitting invariant masses, making them aligned with the enhancements in the  $ucg$  phase space and allowing great VEGAS performance. This drives a key conclusion for the potential role of ZUNIS in the HEP simulation landscape: not to replace VEGAS, but to fill in the gaps where it fails due to inadequate parametrizations, as we illustrate here by using non-multichanneled  $dd \to dd$  as a proxy for more complex processes.

# 5 CONCLUSION

We have showed that ZUNIS can outperform VEGAS both in terms of integral convergence rate and unweighting efficiency on specific cases, at the cost of a significant increase in training and sampling time, which is an acceptable tradeoff for high-precision HEP computations with high costs. In this context, the introduction of efficient training is a key element to making the most of the power of neural importance sampling where function evaluation costs are a major concern. While further testing is required to ascertain how far NIS can fill the gaps left by VEGAS for integrating complex functions, there is already good evidence that ZUNIS can provide needed improvements in specific cases. We hope that the publication of a usable toolbox for NIS such as ZUNIS will stir a wider audience within the HEP community to apply the method so that the exact boundaries its applicability can be more clearly ascertained.

# 6 REPRODUCIBILITY STATEMENT

An anonymized version of the library is available on Github. The code itself can be installed through PyPI using pip install zunis. The data to reproduce the experiments can be generated using scripts provided in the repository at experiments/benchmarks, in which Jupyter notebooks are also available to reproduce the figures of the paper. The following scripts are available:

- benchmarks_03/camel/run_benchmark_default.sh to generate camel integration data  
- benchmarks_04/camel/run_benchmark_default.sh to generate camel sampling speed data

- benchmark_madgraph/ex_benchmarkEMU.sh to generate  $e^{-}\mu \rightarrow e^{-}\mu$  via Z integration data  
- benchmark_madgraph/ex_benchmark_dd.sh to generate  $dd \rightarrow dd$  via Zintegration data  
- benchmark_madgraph/ex_benchmark.ucg.sh to generate  $uc \to ucg$  integration data

# REFERENCES

C. Ahdida, R. Albanese, A. Alexandrov, A. Anokhina, S. Aoki, G. Arduino, E. Atkin, N. Azorskiy, J.J. Back, A. Bagulya, and et al. Fast simulation of muons produced at the ship experiment using generative adversarial networks. Journal of Instrumentation, 14(11):P11028-P11028, Nov 2019. ISSN 1748-0221. doi: 10.1088/1748-0221/14/11/p11028. URL http://dx.doi.org/10.1088/1748-0221/14/11/P11028.  
J. Alwall, R. Frederix, S. Frixione, V. Hirschi, F. Maltoni, O. Mattelaer, H.-S. Shao, T. Stelzer, P. Torrielli, and M. Zaro. The automated computation of tree-level and next-to-leading order differential cross sections, and their matching to parton shower simulations. Journal of High Energy Physics, 2014(7), Jul 2014. ISSN 1029-8479. doi: 10.1007/jhep07(2014)079. URL http://dx.doi.org/10.1007/JHEP07 (2014) 079.

The ATLAS Collaboration. ATLAS HL-LHC Computing Conceptual Design Report. 2020.  
Marco Bellagente, Manuel Haußmann, Michel Luchmann, and Tilman Plehn. Understanding Event-Generation Networks via Uncertainties. arXiv:2104.04543 [hep-ph], April 2021.  
Joshua Bendavid. Efficient Monte Carlo Integration Using Boosted Decision Trees and Generative Deep Neural Networks. 2017.  
Enrico Bothmann, Timo Janßen, Max Knobbe, Tobias Schmale, and Steffen Schumann. Exploring phase space with Neural Importance Sampling. SciPost Physics, 8(4):069, April 2020. ISSN 2542-4653. doi: 10.21468/SciPostPhys.8.4.069.  
Johann Brehmer and Kyle Cranmer. Flows for simultaneous manifold learning and density estimation. 2020.  
Andy Buckley. Computational challenges for MC event generation. Journal of Physics: Conference Series, 1525:012023, April 2020. ISSN 1742-6596. doi: 10.1088/1742-6596/1525/1/012023.  
Andy Buckley, James Ferrando, Stephen Lloyd, Karl Nordström, Ben Page, Martin Rüfenacht, Marek Schönherr, and Graeme Watt. LHAPDF6: parton density access in the LHC precision era. The European Physical Journal C, 75(3), Mar 2015. ISSN 1434-6052. doi: 10.1140/epjc/s10052-015-3318-8. URL http://dx.doi.org/10.1140/epjc/s10052-015-3318-8.  
Anja Butter, Tilman Plehn, and Ramon Winterhalder. How to gan lhc events. SciPost Physics, 7 (6), Dec 2019. ISSN 2542-4653. doi: 10.21468/scipostphys.7.6.075. URL http://dx.doi.org/10.21468/SciPostPhys.7.6.075.  
Anja Butter, Tilman Plehn, and Ramon Winterhalder. How to gan event subtraction. SciPost Physics Core, 3(2), Nov 2020. ISSN 2666-9366. doi: 10.21468/scipostphyscore.3.2.009. URL http://dx.doi.org/10.21468/SciPostPhysCore.3.2.009.  
Stefano Carrazza and Frédéric A. Dreyer. Lund jet images from generative and cycle-consistent adversarial networks. The European Physical Journal C, 79(11), Nov 2019. ISSN 1434-6052. doi: 10.1140/epjc/s10052-019-7501-1. URL http://dx.doi.org/10.1140/epjc/s10052-019-7501-1.  
I.-Kai Chen, Matthew D. Klimek, and Maxim Perelstein. Improved Neural Network Monte Carlo Simulation. SciPost Physics, 10(1):023, January 2021. ISSN 2542-4653. doi: 10.21468/SciPostPhys.10.1.023.

Riccardo Di Sipio, Michele Faucci Giannelli, Sana Ketabchi Haghighat, and Serena Palazzo. Dijetgan: a generative-adversarial network approach for the simulation of qcd dijet events at the lhc. Journal of High Energy Physics, 2019(8), Aug 2019. ISSN 1029-8479. doi: 10.1007/jhep08(2019)110. URL http://dx.doi.org/10.1007/JHEP08(2019)110.  
Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear Independent Components Estimation, 2015.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP, 2017.  
HEP Software Foundation, Thea Aarrestad, Simone Amoroso, Markus Julian Atkinson, Joshua Bendavid, Tommaso Boccali, Andrea Bocci, Andy Buckley, Matteo Cacciari, Paolo Calafiura, Philippe Canal, Federico Carminati, Taylor Childers, Vitaliano Ciulli, Gloria Corti, Davide Costanzo, Justin Gage Dezoort, Caterina Doglioni, Javier Mauricio Duarte, Agnieszka Dziurda, Peter Elmer, Markus Elsing, V. Daniel Elvira, Giulio Eulisse, Javier Fernandez Menendez, Conor Fitzpatrick, Rikkert Frederix, Stefano Frixione, Krzysztof L. Genser, Andrei Gheata, Francesco Giuli, Vladimir V. Gligorov, Hadrien Benjamin Grasland, Heather Gray, Lindsey Gray, Alexander Grohsjean, Christian Gutschow, Stephan Hageboeck, Philip Coleman Harris, Benedikt Hegner, Lukas Heinrich, Burt Holzman, Walter Hopkins, Shih-Chieh Hsu, Stefan Hoche, Philip James Ilten, Vladimir Ivantchenko, Chris Jones, Michel Jouvin, Teng Jian Khoo, Ivan Kisel, Kyle Knoepfel, Dmitri Konstantinov, Attila Krasznahorkay, Frank Krauss, Benjamin Edward Krikler, David Lange, Paul Laycock, Qiang Li, Kilian Lieret, Miaoyuan Liu, Vladimir Loncar, Leif Lonnblad, Fabio Maltoni, Michelangelo Mangano, Zachary Louis Marshall, Pere Mato, Olivier Mattelaer, Joshua Angus McFayden, Samuel Meehan, Alaettin Serhan Mete, Ben Morgan, Stephen Mrenna, Servesh Muralidharan, Ben Nachman, Mark S. Neubauer, Tobias Neumann, Jennifer Ngadiuba, Isobel Ojalvo, Kevin Pedro, Maurizio Perini, Danilo Piparo, Jim Pivarski, Simon Platzer, Witold Pokorski, Adrian Alan Pol, Stefan Prestel, Alberto Ribon, Martin Ritter, Andrea Rizzi, Eduardo Rodrigues, Stefan Roiser, Holger Schulz, Markus Schulz, Marek Schonherr, Elizabeth Sexton-Kennedy, Frank Siegert, Andrzej Siódmok, Graeme A. Stewart, Malik Sudhir, Sioni Paris Summers, Savannah Jennifer Thais, Nhan Viet Tran, Andrea Valassi, Marc Verderi, Dorothea Vom Bruch, Gordon T. Watts, Torre Wenaus, and Efe Yazgan. HL-LHC Computing Review: Common Tools and Community Software. arXiv:2008.13636 [hep-ex, physics:physics], August 2020. doi: 10.5281/zenodo.4009114.  
Christina Gao, Stefan Höche, Joshua Isaacson, Claudius Krause, and Holger Schulz. Event generation with normalizing flows. Physical Review D, 101(7):076002, April 2020a. ISSN 2470-0010, 2470-0029. doi: 10.1103/PhysRevD.101.076002.  
Christina Gao, Joshua Isaacson, and Claudius Krause. I-flow: High-dimensional Integration and Sampling with Normalizing Flows. Machine Learning: Science and Technology, 1(4):045023, November 2020b. ISSN 2632-2153. doi: 10.1088/2632-2153/abab62.  
Niklas Götz. NGoetz/TorchPS: -v1.0.1, March 2021. URL https://doi.org/10.5281/ zenodo.4639109. Available at https://github.com/NGoetz/TorchPS/tree/v1.0.1, version 1.0.1.  
Bobak Hashemi, Nick Amin, Kaustuv Datta, Dominick Olivito, and Maurizio Pierini. Lhc analysis-specific datasets with generative adversarial networks, 2019.  
F. James. Monte Carlo Theory and Practice. Rept. Prog. Phys., 43:1145, 1980. doi: 10.1088/0034-4885/43/9/002.  
S.P. Jones. Higgs Boson Pair Production: Monte Carlo Generator Interface and Parton Shower. Acta Physica Polonica B Proceedings Supplement, 11(2):295, 2018. ISSN 1899-2358, 2082-7865. doi: 10.5506/PhysPolBSupp.11.295.  
R. Kleiss and R. Pittau. Weight optimization in multichannel Monte Carlo. Computer Physics Communications, 83(2-3):141-146, December 1994. ISSN 00104655. doi: 10.1016/0010-4655(94)90043-4.

R Kleiss, W. J Stirling, and S. D Ellis. A new Monte Carlo treatment of multiparticle phase space at high energies. Computer Physics Communications, 40(2):359-373, June 1986. ISSN 0010-4655. doi: 10.1016/0010-4655(86)90119-0.  
Matthew Klimek and Maxim Perelstein. Neural network-based approach to phase space integration. SciPost Physics, 9(4):053, October 2020. ISSN 2542-4653. doi: 10.21468/SciPostPhys.9.4.053.  
G. Peter Lepage. VEGAS: AN ADAPTIVE MULTIDIMENSIONAL INTEGRATION PROGRAM. March 1980.  
G. Peter Lepage. Adaptive Multidimensional Integration: VEGAS Enhanced. Journal of Computational Physics, 439:110386, August 2021. ISSN 00219991. doi: 10.1016/j.jcp.2021.110386.  
Konstantin T. Matchev, Alexander Roman, and Prasanth Shyamsundar. Uncertainties associated with GAN-generated datasets in high energy physics. arXiv:2002.06307 [hep-ex, physics:heph, physics:physics], June 2021.  
Thomas Müller, Brian McWilliams, Fabrice Rousselle, Markus Gross, and Jan Novák. Neural Importance Sampling, 2018.  
Thomas Müller, Brian Mcwilliams, Fabrice Rousselle, Markus Gross, and Jan Novák. Neural Importance Sampling. ACM Transactions on Graphics, 38(5):1-19, November 2019. ISSN 0730-0301, 1557-7368. doi: 10.1145/3341156.  
Thorsten Ohl. Vegas revisited: Adaptive Monte Carlo integration beyond factorization. Comput. Phys. Commun., 120:13-19, 1999. doi: 10.1016/S0010-4655(99)00209-X.  
G Peter Lepage. A new algorithm for adaptive multidimensional integration. Journal of Computational Physics, 27(2):192-203, 1978. ISSN 0021-9991. doi: https://doi.org/10.1016/0021-9991(78)90004-9. URL https://www.sciencedirect.com/science/article/pii/0021999178900049.  
Simon Platzer. RAMBO on diet, 2013.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. 32nd International Conference on Machine Learning, ICML 2015, 2:1530-1538, 2015. ISSN 9781510810587.  
Oren Rippel and Ryan Prescott Adams. High-dimensional probability estimation with deep density models. arXiv preprint arXiv:1302.5125, 2013.  
Bob Stienen and Rob Verheyen. Phase Space Sampling and Inference from Weighted Events with Autoregressive Flows. SciPost Physics, 10(2):038, February 2021. ISSN 2542-4653. doi: 10.21468/SciPostPhys.10.2.038.  
Esteban G Tabak and Cristina V Turner. A family of nonparametric density estimation algorithms. Communications on Pure and Applied Mathematics, 66(2):145-164, 2013.  
Esteban G Tabak and Eric Vanden-Eijnden. Density estimation by dual ascent of the log-likelihood. Communications in Mathematical Sciences, 8(1):217-233, 2010.  
The HSF Physics Event Generator WG, Andrea Valassi, Efe Yazgan, Josh McFayden, Simone Amoroso, Joshua Bendavid, Andy Buckley, Matteo Cacciari, Taylor Childers, Vitaliano Ciulli, Rikkert Frederix, Stefano Frixione, Francesco Giuli, Alexander Grohsjean, Christian Gutschow, Stefan Hoche, Walter Hopkins, Philip Ilten, Dmitri Konstantinov, Frank Krauss, Qiang Li, Leif Lonnblad, Fabio Maltoni, Michelangelo Mangano, Zach Marshall, Olivier Mattelaer, Javier Fernandez Menendez, Stephen Mrenna, Servesh Muralidharan, Tobias Neumann, Simon Platzer, Stefan Prestel, Stefan Roiser, Marek Schonherr, Holger Schulz, Markus Schulz, Elizabeth Sexton-Kennedy, Frank Siegert, Andrzej Siódmok, and Graeme A. Stewart. Challenges in Monte Carlo event generator software for High-Luminosity LHC. Computing and Software for Big Science, 5 (1):12, December 2021. ISSN 2510-2036, 2510-2044. doi: 10.1007/s41781-021-00055-1.  
Quan Zheng and Matthias Zwicker. Learning to Importance Sample in Primary Sample Space. Computer Graphics Forum, 38(2):169-179, 2019. ISSN 1467-8659. doi: 10.1111/ogf.13628.
