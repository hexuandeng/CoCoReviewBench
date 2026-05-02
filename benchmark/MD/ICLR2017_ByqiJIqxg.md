# ONLINE BAYESIAN TRANSFER LEARNING FOR SEQUENTIAL DATA MODELING

Priyank Jain<sup>1</sup>, Zhitang Chen<sup>4</sup>, Pablo Carbajal<sup>1,5</sup>, Edith Law<sup>1</sup>, Laura Middleton<sup>2</sup>, Kayla Regan<sup>2</sup>, Mike Schaekermann<sup>1</sup>, James Tung<sup>3</sup>, Pascal Poupart<sup>1,5</sup>

pjaini@uwaterloo.ca, chenzhitang2@huawei.com, pablo@veedata.io, {edith.law,lmiddlet,kregan}@uwaterloo.ca, {mschaekermann,james.tung,ppoupart}@uwaterloo.ca

<sup>1</sup> David R. Cheriton School of Computer Science, University of Waterloo, Ontario, Canada  
$^{2}$  Department of Kinesiology, University of Waterloo, Ontario, Canada  
<sup>3</sup> Dept. of Mechanical and Mechatronics Engineering, University of Waterloo, Ontario, Canada  
<sup>4</sup> Noah's Ark Laboratory, Huawei Technologies, Hong Kong, China  
<sup>5</sup> Veedata Inc., Kitchener, Ontario, Canada

# ABSTRACT

We consider the problem of inferring a sequence of hidden states associated with a sequence of observations produced by an individual within a population. Instead of learning a single sequence model for the population (which does not account for variations within the population), we learn a set of basis sequence models based on different individuals. The sequence of hidden states for a new individual is inferred in an online fashion by estimating a distribution over the basis models that best explain the sequence of observations of this new individual. We explain how to do this in the context of hidden Markov models with Gaussian mixture models that are learned based on streaming data by online Bayesian moment matching. The resulting transfer learning technique is demonstrated with three real-word applications: activity recognition based on smartphone sensors, sleep classification based on electroencephalography data and the prediction of the direction of future packet flows between a pair of servers in telecommunication networks.

# 1 INTRODUCTION

In several application domains, data instances are produced by a population of individuals that exhibit a variety of different characteristics. For instance, in activity recognition, different individuals might walk or run with different gait patterns. Similarly, in sleep studies, different individuals might exhibit different patterns for the same sleep stages. In telecommunication networks, software applications might generate packet flows between two servers according to different patterns. In such scenarios, it is tempting to treat the population as a homogeneous source of data and to learn a single average model for the population. However, this average model will perform poorly in recognition tasks for individuals that differ significantly from the average. Hence, there is a need for transfer learning techniques that take into account the variations between individuals within a population.

We consider the problem of inferring a sequence of hidden states based on a sequence of observations produced by an individual within a population. Our first contribution is an online Bayesian moment matching technique to estimate the parameters of a hidden Markov model (HMM) with observation distributions represented by Gaussian mixture models (GMMs). This approach allows us to learn separate basis models for different individuals based on streaming data. The second contribution is an unsupervised online technique that infers a probability distribution over the basis models that best explain the sequence of observations of a new individual. The classification of hidden states can then be refined in an online fashion based on the individuals that most resemble the new individual. Furthermore, since the basis models are fixed at classification time and we only learn the weight of each model, good classification accuracy can be obtained more quickly as the stream of observations of the new individual are processed. The third contribution of this work is the demonstration of this approach across different real-world applications, which include activity

recognition, sleep classification and the prediction of packet flow direction in telecommunication networks.

The paper is organized as follows. Section 2 reviews some related work on transfer learning. Section 3 provides some background regarding hidden Markov models Bayesian Moment Matching algorithm Gaussian mixture models. Section 4 describes the proposed online transfer learning technique. Section 5 illustrates the transfer learning technique in three real-world tasks: activity recognition, sleep stage classification and flow direction prediction. Finally, Section 6 concludes the paper and discusses directions for future work.

# 2 RELATED WORK

There is a large literature on transfer learning (Pan & Yang, 2010; Taylor & Stone, 2009; Shao et al., 2015; Cook et al., 2013). Depending on the problem, the input features, the output labels or the distribution over the features and the labels may be different for the source and target domains. In this work, we assume that the same input features are measured and the same output labels are inferred in the source and target domains. The main problem that we consider is subject variability within a population of individuals, which means that different individuals exhibit different distributions over the features and the labels. The problem of subject variability has been studied in several papers. Chieu et al. (2006) describe how to augment conditional random fields with a subject hidden variable to obtain a mixture of conditional random fields that can naturally infer a distribution over the closest subjects in a training population when inferring the activities of a new individual based on physiological data. Rashidi & Cook (2009) proposed a data mining technique with a similarity measure to facilitate the transfer of activity recognition across different people. Chattopadhyay et al. (2011) describe a similarity measure with an intrinsic manifold that preserve the topology of surface electromyography (SEMG) while mitigating distributional differences among individuals. Zhao et al. (2011) proposed a transfer learning technique that starts by training a decision tree to recognize the activities of a user based on smartphone accelerometry. The decision tree is gradually adjusted to a new user by a clustering technique that successively re-weights the training data based on the unlabeled data of the new individual. These approaches mitigate subject variability by various offline transfer learning techniques. In contrast, we propose an online transfer learning technique since the applications that we consider exhibit sequences of observations that arrive in a streaming fashion and therefore require an online technique that can infer the hidden state of each observation as it arrives.

In the next section, we describe an online transfer learning technique for hidden Markov models with Gaussian mixture models. The approach learns different transition and emission models for each individual in the training population. Those models are then treated as basis models to speed up the online learning process for new individuals. More specifically, a weighted combination of the basis models is learned for each new individual. This idea is related to boosting techniques for transfer learning (Dai et al., 2007; Yao & Doretto, 2010; Al-Stouhi & Reddy, 2011) that estimate a weighted combination of base classifiers. However, note that we focus on sequence modeling problems where the classes of consecutive data points are correlated while transfer learning by boosting assumes that the data points are identically and independently distributed.

# 3 BACKGROUND

In this section, we give a brief overview of hidden Markov models (HMMs) and review the Bayesian moment matching (BMM) algorithm in detail with an example. We will use both HMMs and BMM subsequently in our transfer learning algorithm described in Section 4.

# 3.1 HIDDEN MARKOV MODELS

In a hidden Markov model (HMM), each observation  $X_{t}$  is associated with a hidden state  $Y_{t}$ . The Markov property states that the current state depends only on the previous state. HMMs have been widely used in domains involving sequential data like speech recognition, activity recognition, natural language processing etc. An HMM is represented by two distributions

- Transition distribution: The transition distribution models the change in the value of the hidden state over time. The distribution over the current state  $Y_{t}$  given that the previous state is  $Y_{t - 1} = j$  is denoted by  $\theta_{j} = \operatorname{Pr}(Y_{t}|Y_{t - 1} = j)$  where  $\theta_{j} = \{\theta_{1j},\dots,\theta_{Nj}\}$ ,  $N$  is the total number of states and  $\theta_{ij} = \operatorname{Pr}(Y_t = i|Y_{t - 1} = j)$ .  
- Emission distribution: The emission distribution models the effect of the hidden state on the observation  $X_{t}$  at any given time  $t$  and is given by  $\operatorname*{Pr}(X_t|Y_t)$ . In this work, we model the emission distribution as a mixture of Gaussians with  $M$  components, i.e.,  $\operatorname*{Pr}(X_t|Y_t = j) = \sum_{i=1}^{M} w_i \mathcal{N}(\mathbf{X}_t; \boldsymbol{\mu}_i^j, \Sigma_i^j)$

In this paper, we will first estimate the parameters of the transition and emission distributions by Bayesian learning from a set of source domains (individuals). Subsequently, we will use these distributions as basis functions when estimating the transition and emission distributions of a target domain in which we wish to predict the hidden state for each observation. Parameter learning of an HMM using Bayesian learning is done by calculating the posterior over the parameters given a prior distribution.

$$
\Pr \left(\Theta , \Phi , Y _ {t} = j | X _ {t}, Y _ {t - 1} = i\right) \propto \overbrace {\Pr (X _ {t} | Y _ {t} = j)} ^ {\text {E m i s s i o n d i s t r i b u t i o n}} \overbrace {\Pr (Y _ {t} = j | Y _ {t - 1} = i)} ^ {\text {T r a n s i t i o n P r o b a b i l i t y}} \overbrace {\Pr (\Theta , \Phi , Y _ {t - 1} = i | X _ {1 : t - 1})} ^ {\text {P r i o r f o r} t - 1}
$$

$\forall j \in \{1, 2, \dots, N\}$  where  $\Theta$  and  $\Phi$  parametrize the transition and emission distributions respectively.

# 3.2 BAYESIAN MOMENT MATCHING ALGORITHM

The Bayesian moment matching (BMM) algorithm for Gaussian Mixture Models was proposed by Jaini & Poupart (2016); Jaini et al. (2016). Exact Bayesian learning of mixture models based on streaming data is intractable because the number of terms in the posterior after observing each observation increases exponentially. BMM circumvents this issue by projecting the distribution of the exact posterior  $P$  on a tractable family of distributions  $\tilde{P}$  by matching a set of sufficient moments. In this section, we give a brief overview of the BMM algorithm with an example.

Let  $\mathbf{X}^{1:n}$  be a set of  $d$ -dimensional i.i.d observations following  $\operatorname*{Pr}(\mathbf{X}|\Theta) = \sum_{i=1}^{M} w_i \mathcal{N}(x; \boldsymbol{\mu}_i, \Lambda^{-1})$  where  $\Theta = \{(w_1, \boldsymbol{\mu}_1, \Lambda_1^{-1}), (w_2, \boldsymbol{\mu}_2, \Lambda_2^{-1}), \dots, (w_M, \boldsymbol{\mu}_M, \Lambda_M^{-1})\}$  and  $M$  is known.

We choose the prior as a product of a Dirichlet distribution over the weights  $\mathbf{w}$  and  $M$  Normal-Wishart distributions corresponding to the parameters  $(\pmb{\mu}, \Lambda^{-1})$  of each Gaussian component. Such a prior forms a conjugate probability pair of the likelihood and is hence desirable. Concretely,  $P_0(\Theta) = \text{Dir}(\mathbf{w}|\alpha)\prod_{i=1}^{M}\mathcal{NW}(\pmb{\mu}_i, \Lambda_i|\delta_i, \kappa_i, \mathbf{W}_i, \nu_i)$  where  $\mathbf{w} = (w_1, w_2, \dots, w_M)$ ,  $\alpha = (\alpha_1, \alpha_2, \dots, \alpha_M)$ ,  $\mathbf{W}$  is a symmetric positive definite matrix,  $\kappa > 0$  is real,  $\delta \in \mathbb{R}^d$  and  $\nu > d - 1$  is real. The posterior  $P_1(\Theta|\mathbf{X}_1)$  after observing the first data point  $\mathbf{X}_1$  is given by

$$
\begin{array}{l} P _ {1} (\Theta | \mathbf {X} _ {1}) \propto P _ {0} (\Theta) \Pr (\mathbf {X} _ {1} | \Theta) \\ \propto D i r (\mathbf {w} | \boldsymbol {\alpha}) \prod_ {i = 1} ^ {M} \mathcal {N W} (\boldsymbol {\mu} _ {i}, \Lambda_ {i} | \boldsymbol {\delta} _ {i}, \kappa_ {i}, \mathbf {W} _ {i}, \nu_ {i}) \sum_ {j = 1} ^ {M} w _ {j} \mathcal {N} \left(\mathbf {X} _ {1}; \boldsymbol {\mu} _ {j}, \Lambda_ {j} ^ {- 1}\right) \\ \end{array}
$$

Since a Normal-Wishart distribution is a conjugate prior for a Normal distribution with unknown mean and precision matrix,  $\mathcal{NW}(\pmb{\mu}_i,\Lambda_i|\pmb{\delta}_i,\kappa_i,\mathbf{W}_i,\nu_i)\mathcal{N}\big(\mathbf{X}_1;\pmb{\mu}_i,\Lambda_i^{-1}\big) = c\mathcal{NW}(\pmb{\mu}_i,\Lambda_i|\hat{\pmb{\delta}}_i,\hat{\kappa}_i,\hat{\mathbf{W}}_i,\hat{\nu}_i)$  where  $c$  is some constant. Similarly,  $w_{j}Dir(\mathbf{w}|\alpha_{1},\alpha_{2},\dots ,\alpha_{j},\dots ,\alpha_{M}) = kDir(w_{1},w_{2},\dots,w_{M}|\alpha_{1},\alpha_{2},\dots \hat{\alpha}_{j}\dots ,\alpha_{M})$  where  $k$  is some constant. Therefore,  $P_{1}(\Theta |\mathbf{X}_{1})$  is

$$
P _ {1} (\boldsymbol {\Theta} | \mathbf {X} _ {1}) = \frac {1}{Z} \sum_ {j = 1} ^ {M} \left(c _ {j} D i r (\mathbf {w} | \hat {\boldsymbol {\alpha}} _ {j}) \mathcal {N W} (\boldsymbol {\mu} _ {j}, \Lambda_ {j} | \hat {\boldsymbol {\delta}} _ {j}, \hat {\kappa} _ {j}, \hat {\mathbf {W}} _ {j}, \hat {\nu} _ {j}) \prod_ {i \neq j} ^ {M} \mathcal {N W} (\boldsymbol {\mu} _ {i}, \Lambda_ {i} | \boldsymbol {\delta} _ {i}, \kappa_ {i}, \mathbf {W} _ {i}, \nu_ {i})\right)
$$

where  $\hat{\alpha}_j = (\alpha_1, \alpha_2, \dots, \hat{\alpha}_j, \dots, \alpha_M)$  and  $Z$  is the normalization constant. The equation above suggests that the posterior is a mixture of product of distributions where each product component in the summation has the same form as that of the family of distributions of the prior  $P_0(\Theta)$ . It is evident that the terms in the posterior grow by a factor of  $M$  for each iteration, which is problematic. The Bayesian moment matching algorithm approximates this mixture  $P_1(\Theta)$  with a single product

of Dirichlet and Normal-Wishart distributions  $\tilde{P}_1(\Theta)$  by matching all the sufficient moments of  $P_1$  with  $\tilde{P}_1$  which belongs to the same family of distributions as the prior:

$$
\tilde {P} _ {1} (\Theta) = D i r (\mathbf {w} | \boldsymbol {\alpha} ^ {1}) \prod_ {i = 1} ^ {M} \mathcal {N W} (\boldsymbol {\mu} _ {i}, \Lambda_ {i} | \boldsymbol {\delta} _ {i} ^ {1}, \kappa_ {i} ^ {1}, \mathbf {W} _ {i} ^ {1}, \nu_ {i} ^ {1})
$$

We evaluate the parameters  $\alpha^1, \delta_i^1, \kappa_i^1, \mathbf{W}_i^1, \nu_i^1 \forall i \in \{1, 2,.., M\}$  by matching a set of sufficient moments of  $\tilde{P}_1(\Theta)$  with  $P_1(\Theta)$ . The set of sufficient moments in this case is  $S = \{\pmb{\mu}_j, \pmb{\mu}_j \pmb{\mu}_j^T, \pmb{\Lambda}_j, \pmb{\Lambda}_{j_{km}}^2, w_j, w_j^2\} \forall j \in 1, 2,.., M$  where  $\Lambda_{j_{km}}^2$  is the  $(k,m)^{th}$  element of the matrix  $\pmb{\Lambda}_j$ . The expressions for sufficient moments are given by  $\mathbb{E}[g] = \int_{\Theta} g P_1(\Theta) d(\Theta)$ . The parameters of  $\tilde{P}_1$  can be computed from the following set of equations

$$
\mathbb {E} [ w _ {i} ] = \frac {\alpha_ {i}}{\sum_ {j} \alpha_ {j}}; \qquad \mathbb {E} [ w _ {i} ^ {2} ] = \frac {(\alpha_ {i}) (\alpha_ {i} + 1)}{\left(\sum_ {j} \alpha_ {j}\right) \left(1 + \sum_ {j} \alpha_ {j}\right)}
$$

$$
\mathbb {E} [ \boldsymbol {\Lambda} ] = \nu \mathbf {W}; \quad \operatorname {V a r} (\boldsymbol {\Lambda} _ {i j}) = \nu (\mathbf {W} _ {i j} ^ {2} + \mathbf {W} _ {i i} \mathbf {W} _ {j j})
$$

$$
\mathbb {E} [ \boldsymbol {\mu} ] = \boldsymbol {\delta}; \qquad \mathbb {E} [ (\boldsymbol {\mu} - \boldsymbol {\delta}) (\boldsymbol {\mu} - \boldsymbol {\delta}) ^ {T} ] = \frac {\kappa + 1}{\kappa (\nu - d - 1)} \mathbf {W} ^ {- 1}
$$

Using this set of equations, the exact posterior  $P_{1}(\Theta)$  can be approximated with  $\tilde{P}_{1}(\Theta)$ . This posterior will then be the prior for the next iteration and we keep following the steps above iteratively to finally have a distribution  $\tilde{P}_n(\Theta)$  after observing a stream of data  $\mathbf{X}^{1:n}$ . The estimate is  $\hat{\Theta} = \mathbb{E}[\tilde{P}_n(\Theta)]$ . The exact calculations for the Bayesian Moment Matching algorithm are given in appendix A.

# 4 TRANSFER LEARNING USING BMM

In this section, we first motivate the need for an online transfer learning algorithm for sequential data modeling and then explain in detail the different steps of the algorithm. The complete algorithm is given in Alg. (1).

# 4.1 MOTIVATION

Several applications produce data instances from a population of individuals that exhibit a variety of different traits. For example, for the task of activity recognition, different individuals will have different gait patterns despite the fact that they are performing the same activity (e.g., walking, running, standing, etc.). Therefore, it is problematic to make predictions in such domains by considering the population to be homogeneous; however, every population will have individuals resembling each other in some characteristics. This suggests that we can use individuals in a population to make predictions about similar individuals by identifying those individuals who closely resemble each other. However, identifying individuals with similar traits is not straightforward. Alternatively, weights can be assigned to each individual in a population based on a target individual (individual on whom predictions are to be made). All those individuals who resemble closely the target individual will receive higher weights than those with dissimilar traits. Subsequently, predictions about the behavior of the target individual will be based mostly on the observed behavior of the similar individuals.

Our transfer learning algorithm addresses precisely these issues. It has three main steps - first, it learns a model (transition and emission distributions) for each source domain (or individual in a population) that best explains the observations of that source domain. Next, given a target domain (or target individual), it identifies those individuals that closely resemble the target individual by estimating a basis weight associated to each source domain. A higher weight for a source domain implies that the corresponding individual resembles more closely the target individual. Finally, it predicts the hidden states for each observation in the target domain by using the models learned in the source domain and the basis weights that are given to each transition and emission distribution of the source domains. We now explain each step of the algorithm in detail below.

# 4.2 SOURCE DOMAIN - TRAINING

The first step is to learn a model for each source domain in the training data. Suppose that we have labeled sequence data for  $K$  different source domains. Let

$$
Y _ {t} ^ {k} = \text {h i d n e n s t a t e l e l a b} t \text {f o r s o u r c e d o m a i n} k
$$

$$
X _ {t} ^ {k} = \text {f e a t u r e v e c t o r a t t i m e s p t} t \text {f o r s o u r c e d o m a i n} k
$$

Let the sequence of observations be given by  $X_{1:T}^{k} = \{X_{1}^{k}, X_{2}^{k}, \dots, X_{T}^{k}\}$  and the hidden states be  $\{Y_{1}^{k}, Y_{2}^{k}, \dots, Y_{T}^{k}\}$  where  $Y_{t}^{k} \in \{1, 2, \dots, N\} \forall t$ . Furthermore, let us define

$$
\Theta_ {i j} ^ {k} = \Pr \left(Y _ {t} ^ {k} = i \mid Y _ {t - 1} ^ {k} = j\right) \text {i . e . t h e t r a n s i t i o n p r o b a b i l i t y f r o m s t a t e i t o s t a t e} j
$$

We denote the transition matrix for the  $k^{th}$  source domain with  $\Theta^k$ . Let the emission distribution be modeled by a mixture of Gaussian with  $M$  components. This implies

$$
\Pr \left(X _ {t} ^ {k} \mid Y _ {t} ^ {k} = j\right) = \sum_ {m = 1} ^ {M} w _ {j _ {m}} ^ {k} \mathcal {N} \left(X _ {t} ^ {k} \mid \boldsymbol {\mu} _ {j _ {m}} ^ {k}, \Sigma_ {j _ {m}} ^ {k}\right) \quad \forall j \in \{1, 2,.., N \}
$$

Our aim is to learn the parameters characterizing the transition and the emission distribution for each source domain. More precisely, if

$$
\Phi^ {k} = \left\{\phi_ {1} ^ {k}, \phi_ {2} ^ {k},..., \phi_ {N} ^ {k} \right\} \text {w h e r e} \phi_ {i} ^ {k} = \left\{\left(w _ {i _ {1}} ^ {k}, \boldsymbol {\mu} _ {i _ {1}} ^ {k}, \Sigma_ {i _ {1}} ^ {k}\right),..., \left(w _ {i _ {M}} ^ {k}, \boldsymbol {\mu} _ {i _ {M}} ^ {k}, \Sigma_ {i _ {M}} ^ {k}\right) \right\}
$$

then we want to learn the parameters  $\Theta^k$  for the transition distribution and  $\Phi^k$  for the emission distribution for each source domain  $k\in \{1,2,\dots,K\}$ . Since, we use a hidden Markov model, the update equation at each time step for a source domain  $k$  is

$$
P r \left(\Theta , \Phi , Y _ {t} ^ {k} = j \mid X _ {t} ^ {k}, Y _ {t - 1} ^ {k} = i\right) \propto \overbrace {\Pr \left(X _ {t} ^ {k} \mid Y _ {t} ^ {k} = j\right)} ^ {\text {E m i s s i o n d i s t r i b u t i o n}} \overbrace {\Pr \left(Y _ {t} ^ {k} = j \mid Y _ {t - 1} ^ {k} = i\right)} ^ {\text {T r a n s i t i o n P r o b a b i l i t y}} \overbrace {\Pr \left(\Theta^ {k} , \Phi^ {k} , Y _ {t - 1} ^ {k} = i \mid X _ {1 : t - 1} ^ {k}\right)} ^ {\text {P r i o r f o r} t - 1}
$$

The prior over  $(\Theta^k,\Phi^k)$  is given by

$$
\Pr \left(\Theta^ {k}, \Phi^ {k}\right) = \left[ \prod_ {i = 1} ^ {N} D i r \left(\theta_ {i} ^ {k} \mid \boldsymbol {\alpha} _ {i} ^ {k}\right) \right] \left[ \prod_ {j = 1} ^ {N} D i r \left(\mathbf {w} _ {j} ^ {k}; \beta_ {j} ^ {k}\right) \prod_ {u = 1} ^ {M} \mathcal {N W} \left(\boldsymbol {\mu} _ {j _ {u}} ^ {k}, \boldsymbol {\Lambda} _ {j _ {u}} ^ {k}; \boldsymbol {\delta} _ {j _ {u}} ^ {k}, \kappa_ {j _ {u}} ^ {k}, \mathbf {W} _ {j _ {u}} ^ {k}, v _ {j _ {u}} ^ {k}\right) \right] \tag {2}
$$

After substituting the relevant terms in Eq (1), we get

$$
\Pr \left(\Theta , \Phi , Y _ {t} ^ {k} = j | X _ {t} ^ {k}, Y _ {t - 1} ^ {k} = i\right) \propto \sum_ {m = 1} ^ {M} w _ {j _ {m}} ^ {k} \mathcal {N} \left(X _ {t} ^ {k} \mid \boldsymbol {\mu} _ {j _ {m}} ^ {k}, \Sigma_ {j _ {m}} ^ {k}\right) \theta_ {j i} ^ {k} \left[ \prod_ {i = 1} ^ {N} D i r \left(\theta_ {i} ^ {k} \mid \boldsymbol {\alpha} _ {i} ^ {k}\right) \right]
$$

$$
\left[ \prod_ {j = 1} ^ {N} D i r \left(\mathbf {w} _ {j} ^ {k}; \beta_ {j} ^ {k}\right) \prod_ {u = 1} ^ {M} \mathcal {N W} \left(\boldsymbol {\mu} _ {j u} ^ {k}, \boldsymbol {\Lambda} _ {j u} ^ {k}; \boldsymbol {\delta} _ {j u} ^ {k}, \kappa_ {j u} ^ {k}, \mathbf {W} _ {j u} ^ {k}, v _ {j u} ^ {k}\right) \right] \quad \forall j \in \{1, 2, \dots , N \} \tag {3}
$$

Further,  $\Lambda_{j_u}^k = (\Sigma_{j_u}^k)^{-1}$ . The prior in Eq (2) can be understood as having the following components

- Transition Distribution: Each column of the  $N \times N$  transition matrix specifies the probability of making a transition from that column index to another state given by the row index. We define a Dirichlet distribution as a prior over each column of the transition matrix. Hence,  $\prod_{i=1}^{N} \text{Dir}(\theta_i^k | \alpha_i^k)$  is the prior over  $\Theta^k$ .  
- Emission Distribution:  $Dir(\mathbf{w}_j^k;\beta_j^k)\prod_{u = 1}^{M}\mathcal{NW}(\pmb{\mu}_{j_u}^k,\pmb{\Lambda}_{j_u}^k;\delta_{j_u}^k,\kappa_{j_u}^k,\mathbf{W}_{j_u}^k,v_{j_u}^k)$  defines a prior over a mixture of Gaussians for hidden state  $j$  with  $M$  components where the Dirichlet distribution is the prior over the mixture weights and the Normal-Wishart distribution is the prior over the mean and precision matrix of the mixture components. We take a product over  $j$  to obtain a prior over all emission distributions.

The posterior distribution (Eq (3)) after each observation is a mixture of products of distributions where each component has the same form as the prior distribution since  $\operatorname*{Pr}(X_t^k |Y_t^k = j)$  is a mixture of Gaussians. Therefore, the number of terms in the posterior increases exponentially if we perform exact Bayesian learning. To circumvent this, we use BMM for Gaussian Mixture Models as described in (Jaini et al., 2016; Jaini & Poupart, 2016)3. The complete calculations for learning in the source domain are given in appendix B.

# 4.3 TARGET DOMAIN - PREDICTION

The goal is to predict the hidden states for a target individual (or domain) as we observe a sequence of observations. In the previous step, we learned the transition and emission distributions individually for  $K$  different sources. These sources can be thought of as individuals in a population. The transition and emission distributions learned from the individual sources form a basis for the transition and emission distributions of the target domain. Specifically, let the transition distribution for the  $k^{th}$  source be denoted by  $g(\Theta^k)$  and emission distribution be denoted by  $f(\Phi_j^k)$  for a certain hidden state  $j$ . Then, the transition and emission distributions for the target domain is a weighted combination given by

$$
\Pr \left(Y _ {t} = j \mid Y _ {t - 1} = i\right) = \sum_ {m = 1} ^ {K} \lambda_ {m} \Pr \left(Y _ {t} ^ {m} = j \mid Y _ {t - 1} ^ {m} = i\right) = \sum_ {m = 1} ^ {K} \lambda_ {m} g \left(\Theta_ {j i} ^ {m}\right) \tag {4}
$$

$$
\Pr \left(X _ {t} \mid Y _ {t} = j\right) = \sum_ {k = 1} ^ {K} \pi_ {k} \Pr \left(X _ {t} ^ {k} \mid Y _ {t} ^ {k} = j\right) = \sum_ {k = 1} ^ {K} \pi_ {k} f \left(\Phi_ {j} ^ {k}\right) \tag {5}
$$

We first need to compute the basis weights  $\lambda = (\lambda_1, \lambda_2, \dots, \lambda_K)$  and  $\pi = (\pi_1, \pi_2, \dots, \pi_K)$ . We estimate  $(\lambda, \pi)$  in an unsupervised manner using BMM. We define a Dirichlet prior over  $\lambda$  and  $\pi$ , i.e.  $\operatorname{Pr}(\lambda, \pi) = \text{Dir}(\lambda; \gamma) \text{Dir}(\pi; \nu)$ . The posterior after observing a new data point is

$$
\begin{array}{l} \Pr \left(\boldsymbol {\lambda}, \boldsymbol {\pi}, Y _ {t} = j \mid X _ {t}\right) \propto \Pr \left(X _ {t} \mid Y _ {t} = j\right) \sum_ {i = 1} ^ {N} \Pr \left(Y _ {t} = j \mid Y _ {t - 1} = i\right) \Pr \left(\boldsymbol {\lambda}, \boldsymbol {\pi}, Y _ {t - 1} = i\right) (6) \\ \propto \sum_ {k = 1} ^ {K} \pi_ {k} f \left(\Phi_ {j} ^ {k}\right) \sum_ {i = 1} ^ {N} \sum_ {m = 1} ^ {K} \lambda_ {m} g \left(\Theta_ {j i} ^ {m}\right) D i r (\boldsymbol {\lambda}; \boldsymbol {\gamma}) D i r (\boldsymbol {\pi}; \boldsymbol {\nu}) (7) \\ \propto \sum_ {k, m} ^ {K} \sum_ {i = 1} ^ {N} C (i, j, k, m) D i r (\boldsymbol {\pi}; \hat {\boldsymbol {\nu}}) D i r (\boldsymbol {\lambda}; \hat {\boldsymbol {\gamma}}) (8) \\ \end{array}
$$

where  $f(\Phi_j^k)g(\Theta_{ji}^m)$  are known from the source domains,  $\pi_kDir(\pmb{\pi};\pmb{\nu}) = a_kDir(\pmb{\pi};\hat{\pmb{\nu}})$ ,  $\lambda_mDir(\pmb{\lambda};\pmb{\gamma}) = b_mDir(\pmb{\lambda};\hat{\pmb{\gamma}})$  and  $C(i,j,k,m) = a_kb_mf(\Phi_j^k)g(\Theta_{ji}^m)$ . The exact calculations are given in Appendix C. We approximate the posterior in Eq (8) by projecting it onto a tractable family of distributions with the same set of sufficient moments as the posterior using the Bayesian Moment Matching approach. Finally, the estimate of  $(\pmb{\lambda},\pmb{\pi})$  is the expected value of the final posterior. This completes the learning stage.

The transition and emission distributions for the target domain are the weighted combination of transition and emission distributions learned in the source domain respectively. The advantage of this linear combination is to account for heterogeneity in the data. The learning step in the target domain will ensure that only those source domains that resemble closely the target domain are given higher weights. This helps to bias the predictions according to the closest basis models when the population is not homogeneous.

Predictions can be made in two different manners

- Online - initialize the prior over  $\lambda$  and  $\pi$  to be uniform. As each new data point is observed in a sequence, a prediction is made based on the mean of the current posterior over  $\lambda$  and  $\pi$  and subsequently the posterior is updated based on Eq (8).  
- Offline - compute the posterior of  $\lambda$  and  $\pi$  based on Eq (8) by using the entire sequence of observations of the target individual. Once, the posterior is computed, predict the hidden states for each observation in the sequence based on the mean estimates of the posterior.

Algorithm (1) gives the complete algorithm for transfer learning by Bayesian Moment Matching.

# 5 EXPERIMENTS AND RESULTS

This section describes experiments on three tasks from different domains - activity recognition, sleep cycle prediction among healthy individuals and patients suffering from Parkinson's disease and packet flow prediction in telecommunication networks.

Algorithm 1 Online Transfer Learning by Bayesian Moment Matching  
1: Input (Learning): labeled sequence data from multiple domains (individuals)  
2: Input (Prediction): unlabeled sequence data from individuals  
3: Output: labels for hidden states  
Source Domain - learning transition and emission distribution  
4: Input: labeled sequence data from  $K$  domains  
5: specify # of hidden states : nClass  
6: specify # of components in GMM : nComponents  
7: procedure LEARNSOURCEHMM(data, nClass, nComponents)  
8: for  $k = 1:K$  do  
9: Let  $f(\Theta, \Phi)$  be a family of probability distributions with parameters  $\gamma$   
10: Initialize a prior  $P_0^k(\Theta, \Phi)$  from  $f$  over transition and emission parameters respectively  
11: for  $n = 1:N_k$  do  $\triangleright N_k$ : size of data for  $k^{th}$  source domain  
12: Compute  $P_n(\Theta, \Phi)$  from  $P_{n-1}(\Theta, \Phi)$  using Eq. 3  
13: Using BMM approximate  $P_n$  with  $\tilde{P}_n(\Theta, \Phi) = f(\Theta, \Phi|\gamma)$   
14: Return :  $\hat{\Theta} = \mathbb{E}_{\Theta}[\tilde{P}_n(\Theta, \Phi)]$   
15: Return :  $\hat{\Phi} = \mathbb{E}_{\Phi}[\tilde{P}_n(\Theta, \Phi)]$   
16: Return : emission and transition distributions for each source  
Target Domain - learning basis weights for each source domain & prediction  
17: Input: unlabeled sequence data  
18: procedure PREDICTTARGETDOMAIN(data, sourceDistributions)  
19: Let  $g(\lambda, \pi) = Dir(\lambda; \gamma)Dir(\pi; \nu)$  be a family of probability distributions  
20: Initialize a prior  $P_0(\lambda, \pi)$  from  $g$  with equal weights to each source distribution  
21: for  $n = 1:N$  do  $\triangleright N$ : size of data for target domain  
22: Compute  $P_n(\Theta, \Phi)$  from  $P_{n-1}(\Theta, \Phi)$  using Eq. 8  
23: Using BMM approximate  $P_n$  with  $\tilde{P}_n(\lambda, \pi) = g(\lambda, \pi)$   
24: Predict :  $\hat{Y}_n = \operatorname{argmax}_j Pr\left(\lambda, \pi, Y_n = j|X_n\right)$  using Eq (8)  
25: Return :  $\hat{\lambda} = \mathbb{E}_{\lambda}[\tilde{P}_n(\lambda, \pi)]$   
26: Return :  $\hat{\pi} = \mathbb{E}_{\pi}[\tilde{P}_n(\lambda, \pi)]$   
27: Return : prediction  $\hat{Y}_n$

# EXPERIMENTAL SETUP

For each task, we compare our online transfer learning algorithm to EM and a baseline algorithm that both learn a single HMM with mixtures of Gaussians as emissions by treating the population as homogeneous. The baseline algorithm uses Bayesian Moment Matching to learn the parameters of the HMM. Concretely, we have data collected from several individuals (or sources) in a population for each task. For transfer learning, we train an HMM with mixture of Gaussian emission distributions for each source (or individual) except the target individual. For the target individual, we estimate a posterior over the basis weights in an online and unsupervised fashion and make online predictions about the hidden states. We compare the performance of our transfer learning algorithm against the EM and baseline algorithms that treat the population as homogeneous, i.e., we train an HMM by combining the data from all the sources except the target individual. Then, using this model, we make online predictions about the hidden states of the target individual. We report the results based on leave-one-out cross validation where the data of a different individual is left out in each round. For each task, we treat every individual as a target individual once. For a fair comparison, the HMM model learned for both the baseline algorithm and the EM algorithm has the same number of components as the HMM model learned by the online transfer learning algorithm. For each task, we run experiments 10 times with each individual taken as target and the rest acting as source domains for training. We report the average percentage accuracy and use the Wilcoxon signed rank test (Wilcoxon, 1950) to compute a  $p$ -value and report statistical significance when the  $p$ -value is less than 0.05. In the following sections, we discuss the results for each task in detail.

# ACTIVITY RECOGNITION

As part of an on-going study to promote physical activity, we collected smartphone data with 19 participants and tested our transfer learning algorithm to recognize 5 different kinds of activities: sitting, standing, walking, running and in-a-moving-vehicle. While APIs already exist to automatically recognize walking, running and in-a-moving-vehicle by Android and Apple smartphones, sitting and standing are not available in the standard APIs. Furthermore, our long term goal is to obtain robust recognition algorithms for older adults and individuals with perturbed gait (e.g., due to a stroke). Labeled data was obtained by instructing the 19 participants to walk at varying speeds for  $4\mathrm{min}$ , run for  $2\mathrm{min}$ , stand for  $2\mathrm{min}$ , sit for  $2\mathrm{min}$  and ride a moving vehicle to a destination of their choice. The data collected was segmented in epochs of 1 second where 48 features (means and standard deviations of the 3D accelerometer in each epoch) were computed by the smartphone. The online transfer learning algorithm learned an HMM over 18 individuals which acted as basis models for prediction on the  $19^{th}$  individual. In this manner, we ran experiments for each individual 10 times to get a statistical measure of the results.

Table (1) compares the average percentage accuracy of prediction for activity recognition with 19 different individuals. It demonstrates that the transfer learning algorithm performed better than the baseline on 15 individuals and in other cases its accuracy was close to the baseline. Furthermore, it is also worth noting that in most cases, the confusion in the algorithm's prediction was between the following pairs of classes: In a Moving Vehicle—Standing and In a Moving Vehicle—Sitting. This is expected because in most cases the person was either standing/sitting in a bus or sitting in a car. Table (1) also demonstrates the superior performance of online transfer learning algorithm as compared to the EM algorithm.

Table 1: Average percentage accuracy of prediction for activity recognition on 19 different individuals. The best results among the Baseline, the EM algorithm and Transfer Learning algorithm are highlighted in bold font.  $\uparrow (\mathrm{or}\downarrow)$  indicates that the method has significantly better (or worse) accuracy than the baseline under the Wilcoxon signed rank test with  $p$ -value  $< 0.05$ .

<table><tr><td>TARGET
DOMAIN</td><td>BASELINE</td><td>EM</td><td>TRANSFER LEARNING
ALGORITHM</td></tr><tr><td>PERSON 1</td><td>91.29</td><td>83.57</td><td>88.36↓</td></tr><tr><td>PERSON 2</td><td>81.37</td><td>79.87</td><td>87.65↑</td></tr><tr><td>PERSON 3</td><td>74.68</td><td>75.91</td><td>93.15↑</td></tr><tr><td>PERSON 4</td><td>73.39</td><td>68.29</td><td>84.70↑</td></tr><tr><td>PERSON 5</td><td>95.94</td><td>89.59</td><td>99.75↑</td></tr><tr><td>PERSON 6</td><td>73.98</td><td>69.77</td><td>96.43↑</td></tr><tr><td>PERSON 7</td><td>57.62</td><td>55.15</td><td>70.75↑</td></tr><tr><td>PERSON 8</td><td>91.72</td><td>86.05</td><td>97.80↑</td></tr><tr><td>PERSON 9</td><td>81.19</td><td>78.88</td><td>88.75↑</td></tr><tr><td>PERSON 10</td><td>99.12</td><td>93.60</td><td>97.35↓</td></tr><tr><td>PERSON 11</td><td>76.59</td><td>74.67</td><td>88.75↑</td></tr><tr><td>PERSON 12</td><td>55.36</td><td>59.71</td><td>95.05↑</td></tr><tr><td>PERSON 13</td><td>79.66</td><td>73.46</td><td>97.60↑</td></tr><tr><td>PERSON 14</td><td>92.06</td><td>89.11</td><td>93.12↑</td></tr><tr><td>PERSON 15</td><td>79.25</td><td>72.24</td><td>94.20↑</td></tr><tr><td>PERSON 16</td><td>84.08</td><td>79.23</td><td>83.51↓</td></tr><tr><td>PERSON 17</td><td>93.95</td><td>91.03</td><td>97.60↑</td></tr><tr><td>PERSON 18</td><td>82.84</td><td>74.88</td><td>87.20↑</td></tr><tr><td>PERSON 19</td><td>95.97</td><td>89.06</td><td>95.06↓</td></tr></table>

# SLEEP STAGE CLASSIFICATION

Sleep disruption can lead to various health issues. Understanding and analyzing sleep patterns, therefore, has great potential to significantly improve the quality of life for both patients and healthy individuals. In both clinical and research settings, the standard tool for quantifying sleep architecture and physiology is polysomnography (PSG), which is the measurement of electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiography (ECG), and respiratory function of an individual during sleep. The analysis of sleep architecture is of relevance for

the diagnosis of several neurological disorders, e.g., Parkinson's disease (Peeraully et al., 2012), because neurological anomalies often also reflect in variations of a patient's sleep patterns. Typically, PSG data is divided into 30-second epochs and classified into 5 stages of sleep — wake (W), rapid eye movement sleep (REM) or one of 3 non-REM sleep stages (N1, N2, and N3) — based on the visual identification of specific signal features on the EEG, EOG, and EMG channels. Epochs that cannot be distinctly sorted into one of the 5 stages are labeled as Unknown. While it is a valuable clinical and research tool, visual classification of EEG data remains time consuming, requiring up to 2 hours for a highly trained technologist to classify all the epochs within a typical 7-hour PSG recording. Beyond that, inter-scorer agreement rates remain low around 80 (Rosenberg & Van Hout, 2013). High annotation costs and low inter-scorer agreement rates have motivated efforts to develop fully automated approaches for sleep stage classification (Anderer et al., 2005; Jensen et al., 2010; Mal, 2013; Punjabi et al., 2015). However, many of these methods result in generic cross-patient classifiers that fail to reach levels of accuracy and reliability high enough to be adopted in real-world medical settings.

The polysomnograms (PSGs) we used for our evaluation were obtained at a clinical neurophysiology laboratory in Toronto (name anonymized) according to the American Academy of Sleep Medicine guidelines using a Grael HD PSG amplifier (Compumedics, Victoria, Australia). We selected recordings from 142 patients obtained between 2009 and 2015. Out of these 142 recordings, 91 were from healthy subjects and 51 were from patients with Parkinson's disease. Each recording was manually scored by a single registered PSG technologist. Recordings were first segmented into fixed-sized windows of 30 second epochs. To reduce complexity and processing time, we only retained EEG channel C4-A1, which is deemed especially important for sleep stage classification (Sil, 2007). Channel selection and segmentation resulted in a ground truth data set where each instance was represented by a single-channel time series of 7680 floating point numbers corresponding to 30 seconds of C4-A1, sampled at  $256\mathrm{Hz}$ . A vector of 26 scalar features was extracted from each epoch. Bao et al. (2011) and Motamedi-Fakhr et al. (2014) give a detailed listing and explanation of all 26 features. The online transfer learning algorithm learned an HMM over 50 individuals chosen at random which acted as basis models for prediction on the target individual. We did not use all 140 individuals for the basis models because it resulted in sources getting sparse weights diluting the effect of heterogeneity. We completed the experiments for each individual 10 times in this manner to get a statistical measure of the results.

![](images/bec535a6fc89b2fae103c51b8b3030efc5af8d1b62cb2b24c119b50c17bc46ac.jpg)  
(a) Percentage accuracy

![](images/dc15fe424960b1189258ab63ea13d52bbcfcbd8e7da085d548cfbb89cdf6ac1a.jpg)  
(b) Accuracy difference  
Figure 1: Performance comparison of online transfer learning algorithm and baseline for the task of sleep stage classification.

Fig. 1 and 2 compare the performance of the online transfer learning algorithm with the baseline algorithm and the EM algorithm respectively. Fig. 1a compares the average percentage accuracy for our online transfer learning technique and the baseline algorithm and Fig. 2a compares EM and online transfer learning. The blue + signs represent the accuracy of the baseline algorithm and the red o represent the accuracy of the online transfer learning algorithm. The black line is a reference line that passes through the points plotting the accuracy of the online transfer Learning algorithm. The accuracy is plotted against each individual patient. The blue + signs are always below the black line indicating superior performance of the transfer learning algorithm. Fig. 1b and 2b plot the difference between the accuracy of the baseline algorithm and the transfer learning algorithm.

In the top plot, the difference in accuracy is for each patient corresponding to those shown in Fig. 1a and 2b. In the bottom plot, the difference in accuracy is plotted after sorting. A reference line of 0 is also plotted for the case when there is no difference in performance. The plots suggest that for a majority of patients the transfer learning technique outperforms both the baseline algorithm and EM. The results are statistically significant under the Wilcoxon signed rank test with  $p$ -value  $< 0.05$ .

![](images/5ca9fd35314c0990a4042cb9c6d4ca453cc392cb398f37e45aca1dcb6c0fbde5.jpg)  
(a) Percentage accuracy

![](images/21470cdc1a00baad5805d0f4c585f24756510748428a233f4e2b08826697aa6d.jpg)

![](images/d0553bd7943f7d50c2be1d6bd7a4e8dbedaa4ff3673273662bfffbafc575c4f9.jpg)  
(b) Accuracy difference  
Figure 2: Performance comparison of online transfer learning algorithm and EM algorithm for the task of sleep stage classification.

# FLOW DIRECTION PREDICTION

Accurate prediction of future traffic plays an important role in proactive network control. Proactive network control means that if we know the future traffic (including directions and traffic volume), then we have more time to find a better policy for the network routing, priority scheduling as well as rate control in order to maximize network throughput while minimizing transmission delay, packet loss rate, etc.

Better understanding the behavior of TCP connections in certain applications can provide important input to automatic application type detection, especially in those scenarios where network traffic is encrypted and DPI (Deep Packet Inspection) is nearly impossible. Different applications can be distinguished by the distinct behavior of their TCP connections, which are well described by the corresponding HMMs.

We performed our experiments with a publicly available dataset of real traffic from academic buildings. The dataset consists of packet traces with TCP flows. For our experiments, we only consider three packet sizes and flow size as the features. The hidden labels are the source of generation of the packet, i.e., Server or Client. We divided the dataset into 9 domains with each domain consisting of a number of observation sequences. For the online transfer learning algorithm, we learned an HMM for each of 8 sources that acted as basis models for prediction on the  $9^{th}$  source. We compared the performance of the online transfer learning algorithm with EM and the baseline algorithm which treat the data as homogeneous. Table 2 reports the average (of 10 experimental runs) percentage accuracy for each source. The online transfer learning algorithm performs better than both the baseline and the EM algorithm. The results are statistically significant under the Wilcoxon signed rank test with  $p$ -value  $< 0.05$ .

# 6 CONCLUSION

In many applications, data is produced by a population of individuals that exhibit a certain degree of variability. Traditionally, machine learning techniques ignore this variability and train a single model under the assumption that the population is homogeneous. While several offline transfer learning techniques have already been proposed to account for population heterogeneity, this work describes the first online transfer learning technique (to our knowledge) that incrementally determines which source models best explain a streaming sequence of observations while predicting the corresponding hidden states. We achieved this by adapting the online Bayesian moment matching algorithm originally developed for mixture models to hidden Markov models. Experimental results confirm

Table 2: Average percentage accuracy of prediction for flow direction prediction for 9 different domains. The best results among the Baseline, the EM algorithm and Transfer Learning algorithm are highlighted in bold font.  $\uparrow (\mathrm{or}\downarrow)$  indicates that the method has significantly better (or worse) accuracy than the baseline under Wilcoxon signed rank test with pvalue  $< 0.05$ .  

<table><tr><td>TARGET DOMAIN</td><td>BASELINE</td><td>EM</td><td>TRANSFER LEARNING</td></tr><tr><td>SOURCE 1</td><td>72.00</td><td>54.90</td><td>71.02 ↓</td></tr><tr><td>SOURCE 2</td><td>85.33</td><td>89.10</td><td>86.50↓</td></tr><tr><td>SOURCE 3</td><td>80.33</td><td>81.90</td><td>83.33↑</td></tr><tr><td>SOURCE 4</td><td>86.50</td><td>75.80</td><td>87.17↑</td></tr><tr><td>SOURCE 5</td><td>87.33</td><td>82.80</td><td>86.00↓</td></tr><tr><td>SOURCE 6</td><td>93.33</td><td>78.20</td><td>93.50↑</td></tr><tr><td>SOURCE 7</td><td>95.17</td><td>90.70</td><td>95.33↑</td></tr><tr><td>SOURCE 8</td><td>89.83</td><td>91.14</td><td>91.63↑</td></tr><tr><td>SOURCE 9</td><td>76.67</td><td>75.68</td><td>78.83↑</td></tr></table>

the effectiveness of the approach in three real-world applications: activity recognition, sleep stage recognition and flow direction prediction.

In the future, this work could be extended in several directions. Since it is not always clear how many basis models should be used and that the observation sequences of target individuals can necessarily be explained by a weighted combination of basis models, it would be interesting to explore techniques that can automatically determine a good number of basis models and that can generate new basis models on the fly when existing ones are insufficient. Furthermore, since recurrent neural networks (RNNs) have been shown to outperform HMMs with GMM emission distributions in some applications such as speech recognition (Graves et al., 2013), it would be interesting to generalize our online transfer learning technique to RNNs.

# REFERENCES

The Visual Scoring of Sleep in Adults. Journal of Clinical Sleep Medicine, 3(2):121-131, mar 2007. ISSN 1550-9389.  
Performance of an Automated Polysomnography Scoring System Versus Computer-assisted Manual Scoring. Sleep, 36(4):573-582, apr 2013. ISSN 1550-9109. doi: 10.5665/sleep.2548.  
Samir Al-Stouhi and Chandan K Reddy. Adaptive boosting for transfer learning using dynamic updates. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 60-75. Springer, 2011.  
Peter Anderer, Georg Gruber, Silvia Parapatics, Michael Woertz, Tatiana Miazhynskaia, Gerhard Klosch, Bernd Saletu, Josef Zeitlhofer, Manuel J Barbanoj, Heidi Danker-Hopfe, Sari-Leena Himanen, Bob Kemp, Thomas Penzel, Michael Grozinger, Dieter Kunz, Peter Rappelsberger, Alois Schlogl, and Georg Dorffner. An E-health Solution for Automatic Sleep Classification According to Rechtschaffen and Kales: Validation Study of the Somnolyzer 24 x 7 Utilizing the Siesta Database. Neuropsychobiology, 51(3):115-133, 2005. ISSN 0302-282X. doi: 10.1159/000085205.  
Forrest S Bao, Xin Liu, and Christina Zhang. PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction. Computational Intelligence and Neuroscience, 2011:1-7, 2011. ISSN 1687-5265. doi: 10.1155/2011/406391.  
Rita Chattopadhyay, Narayanan Chatakram Krishnan, and Sethuraman Panchanathan. Topology preserving domain adaptation for addressing subject based variability in semg signal. In AAAI Spring Symposium: Computational Physiology, pp. 4-9, 2011.  
Hai Leong Chieu, Wee Sun Lee, and Leslie P Kaelbling. Activity recognition from physiological data using conditional random fields. 2006.  
Diane Cook, Kyle D Feuz, and Narayanan C Krishnan. Transfer learning for activity recognition: A survey. Knowledge and information systems, 36(3):537-556, 2013.

Wenyuan Dai, Qiang Yang, Gui-Rong Xue, and Yong Yu. Boosting for transfer learning. In Proceedings of the 24th international conference on Machine learning, pp. 193-200. ACM, 2007.  
Morris H. Degroot. Optimal statistical decisions. McGraw-Hill Book Company, New York, St Louis, San Francisco, 1970. ISBN 0-07-016242-5. URL http://opac.inria.fr/record=b1080767.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing, pp. 6645-6649. IEEE, 2013.  
Priyank Jaini and Pascal Poupart. Online and distributed learning of gaussian mixture models by bayesian moment matching. arXiv preprint arXiv:1609.05881, 2016.  
Priyank Jaini, Abdullah Rashwan, Han Zhao, Yue Liu, Ershad Banijamali, Zhitang Chen, and Pascal Poupart. Online algorithms for sum-product networks with continuous variables. In Proceedings of the Eighth International Conference on Probabilistic Graphical Models, pp. 228-239, 2016.  
Peter S Jensen, Helge B D Sorensen, Helle L Leonthin, and Poul Jennum. Automatic Sleep Scoring in Normals and in Individuals with Neurodegenerative Disorders According to New International Sleep Scoring Criteria. Journal of Clinical Neurophysiology: Official Publication of the American Electroencephalographic Society, 27(4):296-302, aug 2010. ISSN 1537-1603. doi: 10.1097/WNP.0b013e3181eaad4b.  
Shayan Motamedi-Fakhr, Mohamed Moshrefi-Torbati, Martyn Hill, Catherine M Hill, and Paul R White. Signal Processing Techniques Applied to Human Sleep EEG Signals - A Review. Biomedical Signal Processing and Control, 10:21-33, mar 2014. ISSN 17468094. doi: 10.1016/j.bspc.2013.12.003.  
Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Transactions on knowledge and data engineering, 22(10):1345-1359, 2010.  
Tasneem Peeraully, Ming-Hui Yong, Sudhansu Chokroverty, and Eng-King Tan. Sleep and Parkinson's disease: A review of case-control polysomnography studies. *Movement Disorders*, 27(14): 1729-1737, dec 2012. ISSN 08853185. doi: 10.1002/mds.25197.  
Naresh M Punjabi, Naima Shifa, Georg Dorffner, Susheel Patil, Grace Pien, and Rashmi N Aurora. Computer-Assisted Automated Scoring of Polysomnograms Using the Somnolyzer System. Sleep, 38(10):1555-1566, 2015. ISSN 1550-9109. doi: 10.5665/sleep.5046.  
Parisa Rashidi and Diane J Cook. Transferring learned activities in smart environments. In Intelligent Environments, pp. 185-192, 2009.  
Richard S. Rosenberg and Steven Van Hout. The American Academy of Sleep Medicine Inter-scorer Reliability Program: Sleep Stage Scoring. Journal of Clinical Sleep Medicine, jan 2013. ISSN 1550-9389. doi: 10.5664/jcsm.2350.  
Ling Shao, Fan Zhu, and Xuelong Li. Transfer learning for visual categorization: A survey. IEEE transactions on neural networks and learning systems, 26(5):1019-1034, 2015.  
Matthew E Taylor and Peter Stone. Transfer learning for reinforcement learning domains: A survey. Journal of Machine Learning Research, 10(Jul):1633-1685, 2009.  
Frank Wilcoxon. Some rapid approximate statistical procedures. Annals of the New York Academy of Sciences, pp. 808-814, 1950.  
Yi Yao and Gianfranco Doretto. Boosting for transfer learning with multiple sources. In Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on, pp. 1855-1862. IEEE, 2010.  
Zhongtang Zhao, Yiqiang Chen, Junfa Liu, Zhiqi Shen, and Mingjie Liu. Cross-people mobile-phone based activity recognition. In Twenty-Second International Joint Conference on Artificial Intelligence, 2011.
