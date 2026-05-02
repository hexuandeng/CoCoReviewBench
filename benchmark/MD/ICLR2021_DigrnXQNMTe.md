# A GENERALIZED PROBABILITY KERNEL ON DISCRETE DISTRIBUTIONS AND ITS APPLICATION IN TWO-SAMPLE TEST

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a generalized probability kernel on discrete distributions with finite support. This probability kernel, defined as kernel between distributions instead of samples, generalizes a variety of existing discrepancy statistics including maximum mean discrepancy(MMD) as well as kernelized Stein discrepancy(KSD), and extends to more general cases. For both existing and newly proposed statistics, we estimate them through empirical frequency and illustrate the strategy to search for the unbiased ones. We further analyze several unbiased plugin(empirical) estimators for the task of two-sample test. Our work connects the fields of discrete distribution-property estimation and kernel-based hypothesis test, which might shed light on more new possibilities.

# 1 INTRODUCTION

We focus on the two-sample problem, which is given two i.i.d samples  $\{x_{1}, x_{2}, \ldots, x_{n}\}, \{y_{1}, y_{2}, \ldots, y_{n}\}$ , could we infer the descrepancy between underline distributions they are drawn from. For such a problem, the option of hypothesis test i.e. two-sample test is most popular, and a variety of statistics in estimating the descrepancy is proposed. In recent years, RKHS based method such as maximum mean discrepancy(MMD) has gained a lot of attention. Gretton et al. (2012) has shown that in a universal-RKHS  $F$ ,  $MMD(F, p, q) = 0$  if and only if  $p = q$ , thus could serve as a valid statistic for the two-sample hypothesis test. Gretton et al. (2012) further provides unbiased estimator of MMD with fast asymptotic convergence rate, illustrating its advantages. The work of MMD is then extended to the problem of goodness-of-fit, where only one sample  $\{x_{1}, x_{2}, \ldots, x_{n}\}$  drawn from distribution  $p$  is given and the other distribution  $q$  is given by a probability model. A modification of MMD through the usage of Stein operator results in kernelized Stein discrepancy(KSD) Qiang et al. (2016), and is then further extended to discrete setting Yang et al. (2018). All of the above methods have achieved great success which illustrates the advantages of RKHS-based methods over distribution free analysis.

On the other hand, estimating distribution properties with plugin(empirical) estimators on discrete setting is an active research area in recent years, where people focus on problem settings with large support size but not so large sample size. The Bernstein polynomial technique is introduced in analyzing the bias of the plugin estimators Yi & Alon (2020), which provides remarkable progress on bias-reduction methods of the plugin estimators. It is thus interesting to ask if the plugin estimators could motivate new results for the RKHS-based two-sample test.

Another interesting topic is about the probability kernel, defined as kernel function over probabilities, instead of over samples. As is easily seen, any discrepancy measure of distribution  $p$  and  $q$  could potentially be valid probability kernels, not so much work focuses on this. While Jebara et al. (2004) introduced the so called probability product kernels which generalize a variety of discrepancy measures, its properties remain further study.

Motivated by above observations, our work focuses on a specialized probability kernel function which is a direct generalization of sample-based RKHS methods such as MMD, KSD. We focus on using plugin-estimator as the default estimator of the kernel function we defined, and illustrate that with the help of Bernstein polynomial techniques, we could analyze the bias of these plugin-

estimators. Our work thus connects the fields of discrete distribution-property estimation and kernel-based hypothesis test, which brings interesting possibilities.

# 2 OUR CONTRIBUTIONS

To summarize, our contribution is introducing a framework of the generalized probability kernel(GPK) which in general treats the function space of probability mass function as a subspace of universal-RKHS  $F$ . The resulting family of  $GPK[F]$  is valuable especially when they are equipped with a plugin-estimator. We introduce a series of techniques in analyzing bias of these plugin-estimators and illustrate a procedure of searching for members of  $GPK[F]$  family suitable for applications such as the two-sample test.

# 3 NOTATION

We use bold symbol  $\pmb{v}$  in representing a probability function over a discrete support, and  $[k]$  in representing all the discrete values (labels) the samples can take  $\{x_1, x_2, \dots, x_k\}$ , and symbol  $v_i$  in representing a probability measure of each label  $p_i = p(x_i)$ .

# 4 GENERALIZED PROBABILITY KERNEL

Probability kernel function, defined as kernel function between distributions instead of samples, is a natural extension of the idea of kernel function in sample space.

Definition 1. Given distribution  $\mathbf{p}$  and  $\mathbf{q}$  belongs to a family of discrete distribution with the same finite support in the function space  $X \in \mathbb{R}^k$ , where  $k$  is the support size, we define the probability kernel function as  $K(p,q)$ , which is a kernel function maps from  $X \times X \in \mathbb{R}^k \times \mathbb{R}^k$  to real value  $d \in \mathbb{R}$ , where  $d$  indicates the discrepancy between distributions.

Many discrepancy measures, such as MMD, can serve as probability kernel functions, but people usually don't use the term of probability kernel function when describing them. The reason is that for most of the time, we only consider a limited number of distributions, and do not need or have the resources to navigate through all the distributions within the family. For example, when looking into the two-sample problem, we usually assume two samples  $\{x_1,x_2,\dots,x_n\}$  and  $\{y_{1},y_{2},\ldots ,y_{n}\}$  are i.i.d drawn from two distributions  $p$  and  $q$ , and use the discrepancy measure MMD[F,p,q] to determine if  $p$  and  $q$  are indistinguishable in the RKHS  $F$ . We do not consider all other distributions in  $F$  that is irrelevant to our samples! So far the idea of kernel function between distributions is in practice not so much useful, however, here in this paper, we propose, when considering the plugin-estimator of many of the existing discrepancy measures, it is beneficial to view them as probability kernel functions.

We firstly illustrate this idea through ananlyzing MMD from the viewpoint of probability kernel, and then introduce our definition of generalized probability kernel, which is a specialized probability kernel function with some interesting properties.

# 4.1 MMD AS PROBABILITY KERNEL

We directly start from the definition of squared MMD

By definition, MMD is an instance of integral probability metric for functions in certain RKHS  $\mathcal{F}$ .

$$
\begin{array}{l} \mathbb {M M D} ^ {2} [ \mathcal {F}, p, q ] = \left[ \sup  _ {\| f \| _ {\mathcal {H}} \leq 1} \left(\mathbf {E} _ {x} [ f (x) ] - \mathbf {E} _ {y} [ f (y) ]\right) \right] ^ {2} \\ = \left[ \sup  _ {\| f \| _ {\mathcal {H}} \leq 1} \left\langle \mu_ {p} - \mu_ {q}, f \right\rangle_ {\mathcal {H}} \right] ^ {2} \\ = \left\| \mu_ {p} - \mu_ {q} \right\| _ {\mathcal {H}} ^ {2} \\ \end{array}
$$

The process of Gretton et al. (2012) takes advantage of reproducing property of RKHS, and realize a more widely used form via decomposing the product-sum form into sum-product.

$$
\mathbb {M M D} ^ {2} [ \mathcal {F}, p, q ] = \mathbf {E} _ {x, x ^ {\prime}} [ k (x, x ^ {\prime}) ] - 2 \mathbf {E} _ {x, y} [ k (x, y) ] + \mathbf {E} _ {y y ^ {\prime}} [ k (y, y ^ {\prime}) ]
$$

Where  $x$  and  $x'$  are independent random variables with distribution  $p$ , and  $y$  and  $y'$  are independent random variables with distribution  $q$

The benefit of doing so is that we do not need to evaluate the embedding functions anymore. This is important since for some widely used universal kernels such as RBF kernel, their embedding functions are expensive or even intractable to be directly evaluated.

We next derive our probability kernel form of MMD. The intuition is that we can regroup sample summations according to each label, and result in a functional form of probability for each label under distribution  $p$  and  $q$ .

Recall

$$
\begin{array}{l} \mathbb {M M D} ^ {2} [ \mathcal {F}, p, q ] = \| \mu_ {p} - \mu_ {q} \| _ {\mathcal {H}} ^ {2} \\ = \left\| \mathbf {E} _ {x \sim p} \phi (x) - \mathbf {E} _ {x ^ {\prime} \sim q} \phi (x ^ {\prime}) \right\| _ {\mathcal {H}} ^ {2} \\ \end{array}
$$

Given discrete distribution with finite support, we can write the expectation exactly with distribution probability function  $p(x), q(x)$

$$
\begin{array}{l} \mathbb {M M D} ^ {2} [ \mathcal {F}, p, q ] = \| \mathbf {E} _ {x \sim p} \phi (x) - \mathbf {E} _ {x ^ {\prime} \sim q} \phi (x ^ {\prime}) \| _ {\mathcal {H}} ^ {2} \\ = \left\| \sum_ {i \in [ k ]} \phi (x _ {i}) p _ {i} - \sum_ {i \in [ k ]} \phi (x _ {i}) q _ {i} \right\| _ {\mathcal {H}} ^ {2} \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(p _ {i} - q _ {i}\right) \phi \left(x _ {i}\right) \phi \left(x _ {j}\right) \left(p _ {j} - q _ {j}\right) \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(p _ {i} - q _ {i}\right) K _ {i j} \left(p _ {j} - q _ {j}\right) \\ \end{array}
$$

We are ready to define our generalized probability kernel function, which directly extend the above formula of MMD.

Definition 2 (Generalized probability kernel). Given the family  $S$  of discrete distribution on labels  $\{y_1, y_2, \dots, y_n\}$ , and a universal-RKHS  $F$  defined by kernel  $K$ , where  $k(y_i, y_j)$  maps from  $\mathbb{R}^n \times \mathbb{R}^n$  to  $\mathbb{R}$ , and a element-wise mapping functional on functions in the RKHS  $\phi(\pmb{p}, \pmb{q})$ , which maps from  $\mathbb{R}^n \times \mathbb{R}^n$  to  $\mathbb{R}^n$ , the generalized probability kernel function on distribution  $\pmb{p}$ ,  $\pmb{q} \in S$  is  $k_{prob}(\pmb{p}, \pmb{q}) = \phi(\pmb{p}, \pmb{q}) K\phi(\pmb{p}, \pmb{q})^T = \sum_{i \in [k]} \sum_{j \in [k]} \phi(p_i, q_i) K_{ij} \phi(p_j, q_j)$

More specifically, we are interested in GPK with a narrowed family of mapping functions.

Definition 3 (type I generalized probability kernel). A type  $I$  generalized probability kernel is a GPK that equipped with the mapping function  $\phi$  that satisfies  $\| \phi (\pmb {p},\pmb {q})\| = 0$  if and only if  $\pmb {p} = \pmb{q}$

Obviously, MMD is the special case of this type I generalized probability kernel given  $\phi(p, q) = p - q$

We use the term generalized probability kernel(GPK) as this family of kernel functions directly extends the case of MMD, which utilizes a RKHS in measuring the discrepancy between two distributions. In the following sections, we will show that other kernel based discrepancy measures such as KSD also belong to this family of probability kernel.

Below we firstly exam some of the very important properties of MMD through the GPK point of view, and then illustrate that these properties could be analyzed in the same way for more memebers in the GPK family.

# 4.2 MMD AS METRIC MEASURE

One of the reasons why MMD is suitable as metric for two-sample test is due to the theorem according to Gretton et al. (2012)

Theorem 1. Let  $F$  be a unit ball in a universal RKHS  $H$ , defined on the compact metric space  $X$ , with associated continuous kernel  $k(\cdot, \cdot)$ . Then MMD $[F, \pmb{p}, \pmb{q}] = 0$  if and only if  $\pmb{p} = \pmb{q}$ .

Gretton et al. (2012) have proved this theorem using the universal properties of RKHS, below we provide a more brief proof.

Proof. recall

$$
\begin{array}{l} \mathbb {M M D} ^ {2} [ \mathcal {F}, \boldsymbol {p}, \boldsymbol {q} ] = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} (p _ {i} - q _ {i}) K _ {i j} (p _ {j} - q _ {j}) \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} v _ {i} K _ {i j} v _ {j} \\ = \boldsymbol {v} K \boldsymbol {v} ^ {T} \\ \end{array}
$$

where  $\pmb{v} = \pmb{p} - \pmb{q}$  is also a function in RKHS. Since K in an universal kernel, it is a positive definite matrix in discrete setting. Thus by definition of positive definite matrix,  $\pmb{v}K\pmb{v}^T \geq 0$ , where equality holds if and only if  $\pmb{v} = \mathbf{0}$ , and since  $\pmb{v} = \pmb{p} - \pmb{q}$ , this condition further means  $\pmb{p} = \pmb{q}$

This proof procedure could be easily applied to type I GPK cases and we have the theorem: Theorem

Theorem 2. Let  $F$  be a unit ball in a universal RKHS  $H$ , defined on the compact metric space  $X$ , with associated continuous kernel  $k(\cdot, \cdot)$ . Then type I GPK $[F, \pmb{p}, \pmb{q}] = 0$  if and only if  $\pmb{p} = \pmb{q}$ .

Proof. Following the proof of theorem 1., we just need to add  $\pmb{v} = \phi(\pmb{p}, \pmb{q}) = 0$  implies  $\pmb{p} = \pmb{q}$ , given the definition of type I GPK, we will have the same result as in theorem 1.

The theorem guarantees that GPK is a family of functions with potential usage for two-sample test.

The above results may look trivial, since given the universal properties of universal-RKHS, all functions on metric space  $X$  rely in the same RKHS, so do  $p, q$  and  $\phi(p, q)$ . That is to say, no matter how we change the mapping function  $\phi$  as defined in GPK, as long as the kernel function  $K$  is fixed, we are always dealing with the same RKHS. However, different mapping functions will make a difference when we consider estimating these GPK directly using plugin(empirical) estimators.

# 4.3 PLGIN-ESTIMATOR FOR GPK

Here we propose to estimate the term above empirically, i.e. replace  $p(x), q(x)$  with its empirical frequencies

Definition 4. We define the plugin-estimator of a given member of GPK family as

$$
\mathbb {G P K} _ {E} [ \mathcal {F}, p, q ] = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \phi \left(\frac {N _ {i}}{n _ {1}} - \frac {M _ {i}}{m _ {1}}\right) K _ {i j} \phi \left(\frac {N _ {j}}{n _ {2}} - \frac {M _ {j}}{m _ {2}}\right)
$$

Where  $n_1, n_2, m_1, m_2$  denotes different i.i.d samples drawn from distribution  $p$  and  $q$  with sample size  $n_1, n_2, m_1, m_2$ . Here our setting is different from that of Gretton et al. (2012), where in their setting  $n_1, n_2$  represent the same sample from  $p$  and so does for  $m_1, m_2$  and  $q$ . Another way of viewing this is that for our setting, given two samples  $\{x_1, x_2, \dots, x_n\}$ ,  $\{y_1, y_2, \dots, y_n\}$  from  $p$  and  $q$ , we depart each sample of  $x$  and  $y$  into two parts, yielding 4 different samples with size  $n_1, n_2, m_1, m_2$ , and then calculate the empirical frequencies for plugin-estimator defined above.

The reason for doing this is that it is easier to analyze the unbiasedness of the resulting plugin-estimator. However, the procedure of Gretton et al. (2012) will definitely result in higher sample efficiency and analyzing the effect of such procedure for our plugin-estimator remains future work.

The introducing of plugin-estimator is one of our reasons for proposing the GPK family. As we have already shown, all memethers in  $GPK[F]$  share a common RKHS and it is not clear how

different mapping function  $\phi$  will lead to the difference in measure discrepancy between distributions. However, if we consider that all members in  $GPK[F]$  are estimated using plugin-estimators, the difference is clear. Using different mapping function  $\phi$ , some members of  $GPK[F]$  have unbiased plugin-estimators, some members have not. Also, the variance of plugin-estimator for different members are different, yielding the arts of choosing best  $GPK[F]$  member for task such as two-sample test.

We firstly illustrate our techniques of analyzing bias of plugin estimator through the example of MMD case.

# 4.3.1 PLGIN ESTIMATOR OF MMD IS UNBIASED

As we already know from Gretton et al. (2012), MMD has unbiased estimator building with U-statistics and since our setting of MMD as a member of GPK is actually equivalent to original MMD setting, we can decompose the plugin-estimator back into sample-based estimator as in Gretton et al. (2012):

Firstly we expand the right hand side of our original formula

$$
\begin{array}{l} \mathbb {M M D} _ {E} ^ {2} [ \mathcal {F}, p, q ] = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(\frac {N _ {i}}{n _ {1}} - \frac {M _ {i}}{m _ {1}}\right) K _ {i j} \left(\frac {N _ {j}}{n _ {2}} - \frac {M _ {j}}{m _ {2}}\right) \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(\frac {N _ {i}}{n _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}} + \frac {M _ {i}}{m _ {1}} k _ {i j} \frac {M _ {j}}{m _ {2}} - \frac {N _ {i}}{n _ {1}} k _ {i j} \frac {M _ {j}}{m _ {2}} - \frac {M _ {i}}{m _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}}\right) \\ \end{array}
$$

The right hand side is a summation of 4 parts, we label them as (1), (2), (3), (4)

We take part (1) for example, and the same procedure works for all the 4 parts.

$$
\begin{array}{l} \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \frac {N _ {i}}{n _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}} = \frac {1}{n _ {1} n _ {2}} \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} N _ {i} K _ {i j} N _ {j} \\ = \frac {1}{n _ {1} n _ {2}} \sum_ {x _ {s}} ^ {n _ {1}} \sum_ {x _ {t}} ^ {n _ {2}} K (x _ {s}, x _ {t}) \\ \end{array}
$$

Repeat the procedure above for (1), (2), (3), (4), we get:

$$
\begin{array}{l} \mathrm {M M D} _ {E} ^ {2} [ \mathcal {F}, X, Y ] = \frac {1}{m _ {1} m _ {2}} \sum_ {i = 1} ^ {m _ {1}} \sum_ {j = 1} ^ {m _ {2}} k (x _ {i}, x _ {j}) + \frac {1}{n _ {1} n _ {2}} \sum_ {i = 1} ^ {n _ {1}} \sum_ {j = 1} ^ {n _ {2}} k (y _ {i}, y _ {j}) \\ - \frac {1}{m _ {1} n _ {2}} \sum_ {i = 1} ^ {m _ {1}} \sum_ {j = 1} ^ {n _ {2}} k (x _ {i}, y _ {j}) - \frac {1}{m _ {2} n _ {1}} \sum_ {i = 1} ^ {m _ {2}} \sum_ {j = 1} ^ {n _ {1}} k (x _ {i}, y _ {j}) \\ \end{array}
$$

Note for this procedure,  $x_{i}, x_{j}$  are from different samples, so do  $y_{i}, y_{j}$ . This estimator is the same as the unbiased estimator in Gretton et al. (2012).

Although U-statistics is a very convenient technique in building unbiased estimators, we can not always take advantage of this techniques, since for our setting of GPK family, mapping function  $\phi$  is not always a linear function, thus GPK can not always be decomposed back into sample-based term. Here we propose the other proof procedure with the help of Bernstein polynomial that will work for all members in GPK family(see Appendix for a brief introduction of Bernstein polynomial).

Theorem 3.  $\mathbb{M}\mathbb{M}\mathbb{D}_E^2 [\mathcal{F},p,q]$  is unbiased estimator of  $\mathbb{M}\mathbb{M}\mathbb{D}^2 [\mathcal{F},p,q]$

Proof. Recall

$$
\mathbb {M M D} _ {E} ^ {2} [ \mathcal {F}, p, q ] = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(\frac {N _ {i}}{n _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}} + \frac {M _ {i}}{m _ {1}} k _ {i j} \frac {M _ {j}}{m _ {2}} - \frac {N _ {i}}{n _ {1}} k _ {i j} \frac {M _ {j}}{m _ {2}} - \frac {M _ {i}}{m _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}}\right)
$$

The right hand side is a summation of 4 parts, we label them as (1), (2), (3), (4)

We take part (1) for example, and the same procedure works for all the 4 parts.

notate  $f_{1}(x) = x$ , thus the Bernstein polynomial of  $f_{1}$  is  $B_{n}(f_{1}, p) = p$

$$
\begin{array}{l} \mathbb {E} \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \frac {N _ {i}}{n _ {1}} k _ {i j} \frac {N _ {j}}{n _ {2}} = \sum_ {i \in [ k ]} \mathbb {E} _ {N _ {i} \sim b i n (n _ {1}, p _ {i})} \left(\frac {N _ {i}}{n _ {1}} \sum_ {j \in [ k ]} \mathbb {E} _ {N _ {j} \sim b i n (n _ {2}, p _ {j})} \frac {N _ {j}}{n _ {2}} K _ {i j}\right) \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} p _ {i} K _ {i j} p _ {j} \\ \end{array}
$$

We put our detailed derivation of above equation in Appendix.

Repeat the procedure above for (1), (2), (3), (4), we get:

$$
\mathbb {E} \left(\mathbb {M M D} _ {E} ^ {2} [ \mathcal {F}, p, q ]\right) = \mathbb {M M D} ^ {2} [ \mathcal {F}, p, q ]
$$

Thus the empirical estimation we proposed above is an unbiased estimator of MMD.

![](images/b18d09a35a376a67aa254ed98dc1205f5d8922a4cd3d82ce701c33b7b53152da.jpg)

# 5 MEMBERS IN THE GPK FAMILY AND KERNELS BUILT FROM GPK FAMILY

In section 2, we have introduced GPK family through analyzing MMD. We have already known MMD is a member of GPK family with unbiased plugin-estimator thus suitable for the two-sample test, but it is still unclear whether there exists other members equipped with the same property.

Another interesting question is that, since members in GPK family are kernel functions, the composition rule of kernel function also apply to them. That is to say, we can build new kernel functions based on existing GPK family, the properties of the resulting kernel functions is also of interests.

# 5.1 KERNELIZED STEIN DISCREPANCY AND ITS PLGIN IN ESTIMATOR

Our first example of GPK is the kernelized Stein discrepancy(KSD). KSD is originally proposed for goodness-of-fit, where only one sample  $\{x_{1}, x_{2}, \ldots, x_{n}\} \in p$  is given, and one probability model is used to provide the evaluation of the sample on  $q$  defined by the model. The beauty of KSD is that through the introducing of Stein operator, one does not need to sample from  $q$  to evaluate the discrepancy between  $q$  and  $p$ . Although the KSD is built for goodness-of-fitting setting and has realized great success, it is still interesting to analyze it under the two-sample test, and its relationship to MMD. Our GPK framework provides a special viewpoint of this.

We firstly introduce the existing results for KSD, more specifically, the KSD for discrete distribution as proposed by Yang et al. (2018).

Definition 5 (discrete Stein discrepancy). Let  $\mathcal{X}$  be a finite set. For a family  $\mathcal{F}$  of functions  $f: \mathcal{X}^d \to \mathbb{R}^d$ , define the discrete Stein discrepancy between two positive pmfs  $p, q$  as

$$
\mathbb {D} (q \| p) := \sup  _ {\mathbf {f} \in \mathcal {F}} \| \mathbb {E} _ {\mathbf {x} \sim q} \left[ \mathcal {T} _ {p} \mathbf {f} (\mathbf {x}) \right] - \mathbb {E} _ {\mathbf {x} \sim p} \left[ \mathcal {T} _ {p} \mathbf {f} (\mathbf {x}) \right] \|
$$

where  $\mathcal{A}_p\mathbf{f}(\mathbf{x}) = \mathbf{s}_p(\mathbf{x})\mathbf{f}(\mathbf{x})^\top - \Delta^*\mathbf{f}(\mathbf{x})$  is the different Stein operator w.r.t.  $\pmb{p}$ . Qiang et al. (2016)

For the setting of discrete distribution, follow Yang et al. (2018):

Let  $\mathcal{L}$  be any operator defined on the space of functions  $\mathcal{F} = \{f:\mathcal{X}^d\to \mathbb{R}\}$  that can be written in the form 5

$$
\mathcal {L} f (\mathbf {x}) = \sum_ {\mathbf {x} ^ {\prime} \in \mathcal {X} ^ {d}} g \left(\mathbf {x}, \mathbf {x} ^ {\prime}\right) f \left(\mathbf {x} ^ {\prime}\right), \quad \forall f \in \mathcal {F}
$$

for some bivariate (possibly vector-valued) functions  $g$  on  $\mathcal{X}^d\times \mathcal{X}^d$ . Define a dual operator  $\mathcal{L}^*$  via

$$
\mathcal {L} ^ {*} f (\mathbf {x}) = \sum_ {\mathbf {x} ^ {\prime} \in \mathcal {X} ^ {d}} g \left(\mathbf {x} ^ {\prime}, \mathbf {x}\right) f \left(\mathbf {x} ^ {\prime}\right), \quad \forall f \in \mathcal {F}
$$

Theorem 4. Note  $\mathcal{F} = \{f:\mathcal{X}^d\to \mathbb{R}\}$ . For any positive probability mass function  $\pmb{p}$  on  $\mathcal{X}^d$ , a linear operator  $\mathcal{T}_p$  satisfies Stein's identity

$$
\mathbb {E} _ {\mathbf {x} \sim p} \left[ \mathcal {T} _ {p} f (\mathbf {x}) \right] = 0
$$

for all functions  $f \in \mathcal{F}$  if and only if there exist linear operators  $\mathcal{L}$  and  $\mathcal{L}^*$  of the forms (5) and (6), such that

$$
\mathcal {T} _ {p} f (\mathbf {x}) = \frac {\mathcal {L} p (\mathbf {x})}{p (\mathbf {x})} f (\mathbf {x}) - \mathcal {L} ^ {*} f (\mathbf {x})
$$

holds for all  $\mathbf{x} \in \mathcal{X}^d$  and functions  $f \in \mathcal{F}$ .

Theorem 5. Denote  $s_p(x) = \frac{Lp(x)}{p(x)}$  The KDSD could be written as

$$
\mathbb {D} (q \| p) ^ {2} = \mathbb {E} _ {\mathbf {x}, \mathbf {x} ^ {\prime} \sim q} \left[ \left(s _ {p} (x) - s _ {q} (x)\right) k \left(\mathbf {x}, \mathbf {x} ^ {\prime}\right) \left(s _ {p} \left(x ^ {\prime}\right) - s _ {q} \left(x ^ {\prime}\right)\right) \right]
$$

So far we have introduced all preliminary we need for analyzing KSD under view point of GPK. Next we illustrate that KDSD we defined above belongs to GPK family.

Theorem 6.  $KDSD[F,\pmb{p},\pmb{q}]$  we defined above is a member of  $GPK[F]$  family

Proof. We firstly rewrite expectation into form of weighted sum

$$
\begin{array}{l} \mathbb {D} (q \| p) ^ {2} = \mathbb {E} _ {\mathbf {x}, \mathbf {x} ^ {\prime} \sim q} \left[ \left(s _ {p} (x) - s _ {q} (x)\right) k \left(\mathbf {x}, \mathbf {x} ^ {\prime}\right) \left(s _ {p} (x ^ {\prime}) - s _ {q} (x ^ {\prime})\right) \right] \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} \left(\frac {L p _ {i}}{p _ {i}} - \frac {L q _ {i}}{q _ {i}}\right) q _ {i} K _ {i j} \left(\frac {L p _ {j}}{p _ {j}} - \frac {L q _ {j}}{q _ {j}}\right) q _ {j} \\ \end{array}
$$

We define  $\phi (\pmb {p},\pmb {q}) = \pmb {q}(L\pmb {p} / / \pmb {p} - L\pmb {q} / / \pmb {q})$  where  $^*$  means element-wise division of two vectors in each dimension. We can easily see KDSD is a member of GPK family

Note that original objective of KDSD is goodness-of-fit test, thus the ground truth for distribution  $p(x)$  is already known (the distribution given by the model), thus it is safe to justing sampling according to  $q(x)$  and use  $p(x)$  for evaluation. However for two-sample test,  $p(x)$  should be treated equally with  $q(x)$ . Bearing this intuition, we propose a symmetric form of KDSD for two-sample test scenario:

Definition 6. We define symmetric-KDSD(SKDSD) to be

$$
\mathbb {S K D S D} (q \| p) ^ {2} = \left\| \mathbb {E} _ {\mathbf {x} \sim q} \left[ \mathcal {T} _ {p} \mathbf {f} (\mathbf {x}) \right] - \mathbb {E} _ {\mathbf {x} \sim p} \left[ \mathcal {T} _ {q} \mathbf {f} (\mathbf {x}) \right] \right\| _ {\mathcal {H}} ^ {2}
$$

The SKDSD is a new probability kernel function by summation of two KDSD probability kernel, thus it is a kernel function generated by composition of members in  $GPK[F]$  family, and the plugin-estimator of this probability kernel function is unbiased.

Theorem 7.  $\mathbf{SKDSD}_E^2 [\mathcal{F},p,q]$  is unbiased estimator of  $\mathbf{SKDSD}^2 [\mathcal{F},p,q]$

Proof.

$$
\begin{array}{l} \mathbb {S K D S D} (q \| p) ^ {2} = \left\| \mathbb {E} _ {\mathbf {x} \sim q} \left[ \mathcal {T} _ {p} \mathbf {f} (\mathbf {x}) \right] - \mathbb {E} _ {\mathbf {x} \sim p} \left[ \mathcal {T} _ {q} \mathbf {f} (\mathbf {x}) \right] \right\| _ {\mathcal {H}} ^ {2} \\ = \left\| \sum_ {i \in [ k ]} \mathcal {T} _ {p _ {i}} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right) q _ {i} - \sum_ {i \in [ k ]} \mathcal {T} _ {q _ {i}} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right) p _ {i} \right\| _ {\mathcal {H}} ^ {2} \\ = \left\| \sum_ {i \in [ k ]} \left(\frac {L p _ {i}}{p _ {i}} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right) - L ^ {*} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right)\right) q _ {i} - \sum_ {i \in [ k ]} \left(\frac {L q _ {i}}{q _ {i}} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right) - L ^ {*} \mathbf {f} \left(\mathbf {x} _ {\mathbf {i}}\right)\right) p _ {i} \right\| _ {\mathcal {H}} ^ {2} \\ = \sum_ {i \in [ k ]} \sum_ {j \in [ k ]} L p _ {i} L q _ {j} K _ {i j} - L p _ {i} q _ {j} K _ {i j} ^ {*} - L q _ {i} p _ {j} K _ {i j} ^ {*} + p _ {i} q _ {j} K _ {i j} ^ {* *} \\ \end{array}
$$

since the operator  $L,L^{*}$  is a linear operator, the same proof procedure of MMD also works for SKDSD

# 5.2 POLYNOMIAL PROBABILITY KERNEL

Next we analyze a simple group of members belongs to GPK family

Definition 7 (polynomial probability kernel). A polynomial probability kernel  $\mathbb{P}\mathbb{O}_{lk}[F,\pmb {p},\pmb {q}]$  is a special case of type I GPK which mapping function  $\phi$  is a polynomial function  $\phi (\pmb {p},\pmb {q}) = \pmb {p}^l -\pmb {q}^k$

As we can see, MMD is a member of polynomial probability kernel as  $\mathbb{P}\mathbb{O}_{11}[F,\pmb {p},\pmb {q}]$

As a brief example, we illustrate the polynomial probability kernel with mapping function  $\phi(\pmb{p}, \pmb{q}) = \pmb{p}^2 - \pmb{q}^2$ . Interestingly, we show that although its plugin-estimator is biased, it could be modified to be unbiased. This result is obtained through analyzing the unbiasedness of the plugin-estimator of this GPK family which illustrates the power of our framework.

Theorem 8. the default plugin-estimator of  $\mathbb{P}\mathbb{O}_{22}[\mathcal{F},\pmb {p},\pmb {q}]$  is a biased estimator

Proof. We put our formal proof in Appendix A.3

![](images/4ac64e9d1008c77ee0e49c88e9b48eb23739f3b11873cac6078216d89a03b2d8.jpg)

The interesting result is that by including  $\mathbb{P}\mathbb{O}_{12}[\mathcal{F},\pmb {p},\pmb {q}],\mathbb{P}\mathbb{O}_{21}[\mathcal{F},\pmb {p},\pmb {q}]$  and  $\mathbb{P}\mathbb{O}_{11}[\mathcal{F},\pmb {p},\pmb {q}]$  into consideration, we can see the linear combination the default of empirical-estimators of these probability kernels may yield unbiased estimators. The basic idea is to check the bias of the each  $\mathbb{P}\mathbb{O}_{lkE}^{2}[\mathcal{F},p,q]$  using the techniques of Bernstein polynomial, and then correct the bias with another  $\mathbb{P}\mathbb{O}_{lkE}^{2}[\mathcal{F},p,q]$ .

Theorem 9.  $\frac{m}{m - 1}\left(\mathbb{P}\mathbb{O}_{12E}^2 [\mathcal{F},p,q] - \frac{1}{m}\mathbb{P}\mathbb{O}_{1E}^2 [\mathcal{F},p,q]\right)$  is an unbiased estimator of  $\mathbb{P}\mathbb{O}_{12}^{2}[\mathcal{F},p,q]$

Proof. We put our detailed proof in appendix A.4.

![](images/7da4d7e2920f1906fa4ae10c56ffaf3e14c3ffe7038b84a3a0021b7d18d351e2.jpg)

Theorem 10.  $\left(\frac{m}{m - 1}\right)^2\left(\mathbb{P}\mathbb{O}_{22E}^2 [\mathcal{F},p,q] - \frac{1}{m}\mathbb{P}\mathbb{O}_{12E}^2 [\mathcal{F},p,q] - \frac{1}{m}\mathbb{P}\mathbb{O}_{21E}^2 [\mathcal{F},p,q] + \frac{1}{m^2}\mathbb{P}\mathbb{O}_{12E}^2 [\mathcal{F},p,q]\right)$  is an unbiased estimator of  $\mathbb{P}\mathbb{O}_{22}^{2}[\mathcal{F},p,q]$

Proof. We put our detailed proof in appendix A.5

![](images/eac579b774bfe03402e6a6a322befb4a9d8d44c064e09c123ea180f9f1d0f38a.jpg)

# 6 TWO-SAMPLE TEST USING GENERALIZED PROBABILITY KERNEL

In this section, we briefly discuss the potential usage of the GPK family in two-sample test. We firstly summarize the requirements for a statistic to be suitable for the two-sample test, then come to each of the special GPK family members we have discussed so far.

Following the work of Gretton et al. (2012), we summarize that, for a statistic to be suitable for the two-sample test, it should:

- provide the property that  $D(p\| q) = 0$  if and only if  $p = q$ .  
- should have unbiased estimators, or at least estimators with bounded bias  
- the estimator in requirement 2. should have the variance is analyzable so as to provide valid test threshold and test power.

The first requirement is satisfied by all the members of type I GPK we have defined, and this includes MMD,  $\mathbb{P}\mathbb{O}_{22}[F,p,q]$  as examples. The second requirement is satisfied by the members of GPK family with unbiased plugin-estimators, which includes MMD, SKDSD  $\mathbb{P}\mathbb{O}_{22}[F,p,q]$  as examples.

The third requirement has not been discussed in this paper so far, but using the technique of Bernstein polynomial analysis, it is possible and remain further study.

# REFERENCES

Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 12:723-773, 2012.

Tony Jebara, Risi Kondor, and Andrew Howard. Probability product kernels. J. Mach. Learn. Res., 5:819-844, December 2004. ISSN 1532-4435.  
Liu Qiang, Lee Jason, and Jordan Michael. A kernelizedstein discrepancy for goodness-of-fit tests. International Conference on Machine Learning, 48:276-284, 2016.  
Jiasen Yang, Qiang Liu, Vinayak Rao, and Jennifer Neville. Goodness-of-fit testing for discrete distributions via stein discrepancy. International Conference on Machine Learning, 80:276-284, 2018.  
Hao Yi and Orlitsky Alon. Data amplification: Instance-optimal property estimation. International Conference on Machine Learning, 2020.
