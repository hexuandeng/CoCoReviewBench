# Offline RL Without Off-Policy Evaluation

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Most prior approaches to offline reinforcement learning (RL) have taken an iterative actor-critic approach involving off-policy evaluation. In this paper we show that simply doing one step of constrained/regularized policy improvement using an on-policy Q estimate of the behavior policy performs surprisingly well. This one-step algorithm beats the previously reported results of iterative algorithms on a large portion of the D4RL benchmark. The simple one-step baseline achieves this strong performance without many of the tricks used by previously proposed iterative algorithms and is more robust to hyperparameters. We argue that the relatively poor performance of iterative approaches is a result of the high variance inherent in doing off-policy evaluation and magnified by the repeated optimization of policies against those high-variance estimates. In addition, we hypothesize that the strong performance of the one-step algorithm is due to a combination of favorable structure in the environment and behavior policy.

# 1 Introduction

An important step towards effective real-world RL is to improve sample efficiency. One avenue towards this goal is offline RL (also known as batch RL) where we attempt to learn a new policy from data collected by some other behavior policy without interacting with the environment. Recent work in offline RL is well summarized by Levine et al. [2020].

In this paper, we challenge the dominant paradigm in the deep offline RL literature that primarily relies on actor-critic style algorithms that alternate between policy evaluation and policy improvement [Fujimoto et al., 2018a, 2019, Peng et al., 2019, Kumar et al., 2019, 2020, Wang et al., 2020b, Wu et al., 2019, Kostrikov et al., 2021, Jaques et al., 2019, Siegel et al., 2020, Nachum et al., 2019]. All these algorithms rely heavily on off-policy evaluation to learn the critic. Instead, we find that a simple baseline which only performs one step of policy improvement using the behavior Q function often outperforms the more complicated iterative algorithms. Explicitly, we find that our one-step algorithm beats prior results of iterative algorithms on most of the gym-mujoco [Brockman et al., 2016] and Adroit [Rajeswaran et al., 2017] tasks in the the D4RL benchmark suite [Fu et al., 2020].

We then dive deeper to understand why such a simple baseline is effective. First, we examine what goes wrong for the iterative algorithms. When these algorithms struggle, it is often due to poor off-policy evaluation leading to inaccurate Q values. We attribute this to two causes: (1) distribution shift between the behavior policy and the policy to be evaluated, and (2) iterative error exploitation whereby policy optimization introduces bias and dynamic programming propagates this bias across the state space. We show that empirically both issues exist in the benchmark tasks and that one way to avoid these issues is to simply avoid off-policy evaluation entirely.

Finally, we recognize that while the one-step algorithm is a strong baseline, it is not always the best choice. In the final section we provide some guidance about when iterative algorithms can perform better than the simple one-step baseline. Namely, when the dataset is large and behavior policy has good coverage of the state-action space, then off-policy evaluation can succeed and iterative

algorithms can be effective. In contrast, if the behavior policy is already fairly good, but as a result does not have full coverage, then one-step algorithms are often preferable.

# Our main contributions are:

- A demonstration that a simple baseline of one step of policy improvement outperforms more complicated iterative algorithms on a broad set of offline RL problems.  
- An examination of failure modes of off-policy evaluation in iterative offline RL algorithms.  
- A description of when one-step algorithms are likely to outperform iterative approaches.

# 2 Setting and notation

![](images/cdb813858ba7bd110e18899cbdf68b7b15b4eebd051c47198436ce133ef9ba44.jpg)  
Figure 1: A cartoon illustration of the difference between one-step and multi-step methods. All algorithms constrain themselves to a neighborhood of "safe" policies around  $\beta$ . A one-step approach (left) only uses the on-policy  $\widehat{Q}^{\beta}$ , while a multi-step approach (right) repeatedly uses off-policy  $\widehat{Q}^{\pi_i}$ .

We will consider an offline RL setup as follows. Let  $\mathcal{M} = \{\mathcal{S},\mathcal{A},\rho ,P,R,\gamma \}$  be a discounted infinite-horizon MDP. In this work we focus on applications in continuous control, so we will generally assume that both  $\mathcal{S}$  and  $\mathcal{A}$  are continuous and bounded. We consider the offline setting where rather than interacting with  $\mathcal{M}$ , we only have access to a dataset  $D_{N}$  of  $N$  tuples of  $(s_i,a_i,r_i)$  collected by some behavior policy  $\beta$  with initial state distribution  $\rho$ . Let  $r(s,a) = \mathbb{E}_{r|s,a}[r]$  be the expected reward. Define the state-action value function for any policy  $\pi$  by  $Q^{\pi}(s,a)\coloneqq \mathbb{E}_{P,\pi |s_0 = s,a_0 = a}[\sum_{t = 0}^{\infty}\gamma^t r(s_t,a_t)]$ . The objective is to maximize the expected return  $J$  of the learned policy:

$$
J (\pi) := \underset {\rho , P, \pi} {\mathbb {E}} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r \left(s _ {t}, a _ {t}\right) \right] = \underset { \begin{array}{c} s \sim \rho \\ a \sim \pi | s \end{array} } {\mathbb {E}} [ Q ^ {\pi} (s, a) ]. \tag {1}
$$

Following Fu et al. [2020] and others in this line of work, we allow access to the environment to tune a small  $(< 10)$  set of hyperparameters. See Paine et al. [2020] for a discussion of the active area of research on hyperparameter tuning for offline RL. We also discuss this further in Appendix C.

# 3 Related work

Most prior work on deep offline RL consists of iterative actor-critic algorithms. The primary innovation of each paper is to propose a different mechanism to ensure that he learned policy does not stray too far from the data generated by the behavior policy. Broadly, we group these methods into three camps: policy constraints/regularization, modified of imitation learning, and Q regularization:

1. The majority of prior work acts directly on the policy. Some authors have proposed explicit constraints on the learned policy to only select actions where  $(s, a)$  has sufficient support under the data generating distribution [Fujimoto et al., 2018a, 2019, Laroche et al., 2019]. Another proposal is to regularize the learned policy towards the behavior policy Wu et al. [2019] usually either with a KL divergence [Jaques et al., 2019] or MMD [Kumar et al., 2019]. This is a very straightforward way to stay close to the behavior with a hyperparameter that determines just how close. All of these algorithms are iterative and rely on off-policy evaluation.  
2. Siegel et al. [2020], Wang et al. [2020b], Chen et al. [2020] all use algorithms that filter out datapoints with low Q values and then perform imitation learning. Wang et al. [2018], Peng et al. [2019] use a weighted imitation learning algorithm where the weights are determined by exponentiated Q values. These algorithms are iterative.  
3. Another way to prevent the learned policy from choosing unknown actions is to incorporate some form of regularization to encourage staying near the behavior and being pessimistic about unknown state, action pairs [Wu et al., 2019, Nachum et al., 2019, Kumar et al., 2020, Kostrikov et al., 2021]. However, properly being able to quantify uncertainty about unknown states is notoriously difficult when dealing with neural network value functions [Buckman et al., 2020]. Again all of these algorithms are iterative.

Some recent work has also noted that optimizing policies based on the behavior value function can perform surprisingly well [Gulcehre et al., 2020, Goo and Niekum, 2020]. However, these papers propose complicated variants of the one-step approach involving ensembles, non-standard regularizers and paraterizations or ensembles and distributional Q functions. In contrast, we implement the simplest possible one-step algorithms without any modifications to the network architecture or standard regularizers/constraints. Moreover, we focus on providing an analysis of when and why this simple baseline works.

There are also important connections between the one-step algorithm and the literature on conservative policy improvement [Kakade and Langford, 2002, Schulman et al., 2015, Achiam et al., 2017], which we discuss in more detail in Appendix B.

# 4 Defining the algorithms

In this section we provide a unified algorithmic template for offline RL algorithms as offline approximate modified policy iteration. We show how this template captures our one-step algorithm as well as a multi-step policy iteration algorithm and an iterative actor-critic algorithm. Then any choice of policy evaluation and policy improvement operators defines one-step, multi-step, and iterative algorithms.

# 4.1 Algorithmic template

We consider a generic offline approximate modified policy iteration (OAMPI) scheme, shown in Algorithm 1. Essentially the algorithm alternates between two steps. First, there is a policy evaluation step where we estimate the Q function of the current policy  $\pi_{k-1}$  by  $\widehat{Q}^{\pi_{k-1}}$  using only the dataset  $D_N$ . Implementations also often use the prior Q estimate  $\widehat{Q}^{\pi_{k-2}}$  to warm-start the approximation process. Second, there is a policy improvement step. This step takes in the estimated Q function  $\widehat{Q}^{\pi_{k-1}}$ , the estimated behavior  $\hat{\beta}$ , and the dataset  $D_N$  and produces a new policy  $\pi_k$ . Again an algorithm may use  $\pi_{k-1}$  to warm-start the optimization. Moreover, we expect this improvement step to be regularized or constrained to ensure that  $\pi_k$  remains in the support of  $\beta$  and  $D_N$ . Choices for this regularization/constraint are discussed below. Now we discuss a few ways to instantiate the template.

# Algorithm 1: OAMPI

input:  $K$ , dataset  $D_N$ , estimated behavior  $\hat{\beta}$

Set  $\pi_0 = \hat{\beta}$ . Initialize  $\widehat{Q}^{\pi -1}$  randomly.

for  $k = 1,\dots ,K$  do

Policy evaluation:  $\widehat{Q}^{\pi_{k - 1}} = \mathcal{Q}(\pi_{k - 1},D_N,\widehat{Q}^{\pi_{k - 2}})$

Policy improvement:  $\pi_{k} = \mathcal{I}(\widehat{Q}^{\pi_{k - 1}},\widehat{\beta},D_{N},\pi_{k - 1})$

end

One-step. The simplest algorithm sets the number of iterations  $K = 1$ . We train the policy evaluation to estimate  $Q^{\beta}$ , and then use one of the policy improvement operators discussed below to find the resulting  $\pi_1$ .

Multi-step. The multi-step algorithm now sets  $K > 1$ . The evaluation operator must evaluate off-policy

since  $D_N$  is collected by  $\beta$ , but evaluation steps for  $K \geq 2$  require evaluating policies  $\pi_{k-1} \neq \beta$ .

Each iteration is trained to convergence in both the estimation and improvement steps.

Iterative actor-critic. An actor critic approach looks somewhat like multistep policy iteration, but does not attempt to train to convergence at each iteration. Instead, each iteration consists of one gradient step to update the Q estimate and one gradient step to improve the policy. Since all of the evaluation and improvement operators that we consider are gradient-based, this algorithm can adapt the same evaluation and improvement operators used by the multi-step algorithm. Most algorithms from the literature fall into this category [Fujimoto et al., 2018a, Kumar et al., 2019, 2020, Wu et al., 2019, Wang et al., 2020b, Siegel et al., 2020].

# 4.2 Policy evaluation operators

Following prior work on continuous state and action problems, we always evaluate by simple fitted Q evaluation [Fujimoto et al., 2018a, Kumar et al., 2019, Siegel et al., 2020, Wang et al., 2020b, Paine et al., 2020, Wang et al., 2021]. Explicitly the evaluation step for the one-step or multi-step

algorithms looks like

$$
\mathcal {Q} \left(\pi_ {k - 1}, D _ {N}, \widehat {Q} ^ {\pi_ {k - 2}}\right) = \arg \min  _ {Q} \sum_ {i = 1} ^ {N} \left(r \left(s _ {i}, a _ {i}\right) + \gamma \underset {a ^ {\prime} \sim \pi_ {k - 1} \mid s _ {i} ^ {\prime}} {\mathbb {E}} Q \left(s _ {i} ^ {\prime}, a ^ {\prime}\right) - Q \left(s _ {i}, a _ {i}\right)\right) ^ {2}, \tag {2}
$$

where the right hand side may depend on  $\widehat{Q}^{\pi_{k-2}}$  to warm-start optimization. In practice this is optimized by stochastic gradient descent with the use of a target network [Mnih et al., 2015]. For the iterative algorithm the arg min is replaced by a single stochastic gradient step. We estimate the expectation over next state by a single sample from  $\pi_{k-1}$  (or from the dataset in the case when  $\pi_{k-1} = \hat{\beta}$ ). See Voloshin et al. [2019], Wang et al. [2021] for more comprehensive examinations of this evaluation step.

# 4.3 Policy improvement operators

To instantiate the template, we also need to choose a specific policy improvement operator  $\mathcal{I}$ . We consider the following improvement operators selected from those discussed in the related work section. Each operator has a hyperparameter controlling deviation from the behavior policy.

Behavior cloning. The simplest baseline worth including is to just return  $\hat{\beta}$  as the new policy  $\pi$ . Any policy improvement operator ought to perform at least as well as this baseline.

Constrained policy updates. Algorithms like BCQ [Fujimoto et al., 2018a] and SPIBB [Laroche et al., 2019] constrain the policy updates to be within the support of the data/behavior. In favor of simplicity, we implement a simplified version of the BCQ algorithm that removes the policy correction network which we call Easy BCQ. We define a new policy  $\hat{\pi}_k^M$  by drawing  $M$  samples from  $\hat{\beta}$  and then executing the one with the highest value according to  $\hat{Q}^\beta$ . Explicitly:

$$
\hat {\pi} _ {k} ^ {M} (a | s) = \mathbb {1} [ a = \arg \max  _ {a _ {j}} \left\{\widehat {Q} ^ {\pi_ {k - 1}} (s, a _ {j}): a _ {j} \sim \pi_ {k - 1} (\cdot | s), 1 \leq j \leq M \right\} ]. \tag {3}
$$

Regularized policy updates. Another common idea proposed in the literature is to regularize towards the behavior policy [Wu et al., 2019, Jaques et al., 2019, Kumar et al., 2019, Ma et al., 2019]. For a general divergence  $D$  we can define an algorithm that maximizes a regularized objective:

$$
\hat {\pi} _ {k} ^ {\alpha} = \arg \max  _ {\pi} \sum_ {i} \mathbb {E} _ {a \sim \pi | s} [ \widehat {Q} ^ {\pi_ {k - 1}} (s _ {i}, a) ] - \alpha D (\hat {\beta} (\cdot | s _ {i}), \pi (\cdot | s _ {i})) \tag {4}
$$

A comprehensive review of different variants of this method can be found in Wu et al. [2019] which does not find dramatic differences across regularization techniques. In practice, we will use reverse KL divergence, i.e.  $KL(\pi (\cdot |s_i)\| \hat{\beta} (\cdot |s_i))$ . To compute the reverse KL, we draw samples from  $\pi (\cdot |s_i)$  and use the density estimate  $\hat{\beta}$  to compute the divergence. Intuitively, this regularization forces  $\pi$  to remain within the support of  $\beta$  rather than incentivizing  $\pi$  to cover beta.

Variants of imitation learning. Another idea, proposed by [Wang et al., 2018, Siegel et al., 2020, Wang et al., 2020b, Chen et al., 2020] is to modify an imitation learning algorithm either by filtering or weighting the observed actions so as to get a policy improvement. The weighted version that we implement uses exponentiated advantage estimates to weight the observed actions:

$$
\hat {\pi} _ {k} ^ {\tau} = \arg \max  _ {\pi} \sum_ {i} \exp \left(\tau \left(\widehat {Q} ^ {\pi_ {k - 1}} \left(s _ {i}, a _ {i}\right) - \widehat {V} \left(s _ {i}\right)\right)\right) \log \pi \left(a _ {i} \mid s _ {i}\right). \tag {5}
$$

# 5 Benchmark Results

Our main empirical finding is that one step of policy improvement is sufficient to beat state of the art results on much of the D4RL benchmark suite Fu et al. [2020]. This is striking since prior work focuses on iteratively estimating the Q function of the current policy iterate, but we only use one-step derived from  $\widehat{Q}^{\beta}$ . Results are shown in Table 1. Full experimental details are in Appendix C.

As we can see in the table, all of the one-step algorithms usually outperform the best iterative algorithms tested by Fu et al. [2020]. The one notable exception is the case of random data (especially

Table 1: Results of one-step algorithms on the D4RL benchmark. The first column gives the best results across several iterative algorithms considered in Fu et al. [2020]. We run 3 seeds and each algorithm is tuned over 6 values of their respective hyperparameter. We report the mean and standard deviation over seeds on 100 evaluation episodes per seed. We **bold** the best result on each dataset and **blue** any result where a one-step algorithm beat the best reported iterative result from Fu et al. [2020]. We use m for medium, m-e for medium-expert, m-re for medium-replay, r for random, and c for cloned.

<table><tr><td rowspan="2"></td><td>Iterative</td><td colspan="4">One-step</td></tr><tr><td>Fu et al. [2020]</td><td>BC</td><td>Easy BCQ</td><td>Rev. KL Reg</td><td>Exp. Weight</td></tr><tr><td>halfcheetah-m</td><td>46.3</td><td>41.9 ± 0.1</td><td>52.6 ± 0.2</td><td>55.2 ± 0.4</td><td>48.4 ± 0.1</td></tr><tr><td>walker2d-m</td><td>81.1</td><td>68.6 ± 6.3</td><td>87.2 ± 1.3</td><td>85.9 ± 1.4</td><td>81.8 ± 2.2</td></tr><tr><td>hopper-m</td><td>58.8</td><td>49.9 ± 3.1</td><td>74.5 ± 6.2</td><td>83.7 ± 4.5</td><td>59.6 ± 2.5</td></tr><tr><td>halfcheetah-m-e</td><td>64.7</td><td>61.1 ± 2.7</td><td>78.2 ± 1.6</td><td>93.8 ± 0.5</td><td>93.4 ± 1.6</td></tr><tr><td>walker2d-m-e</td><td>111.0</td><td>78.5 ± 22.4</td><td>112.2 ± 0.3</td><td>111.2 ± 0.2</td><td>113.0 ± 0.4</td></tr><tr><td>hopper-m-e</td><td>111.9</td><td>49.1 ± 4.3</td><td>85.1 ± 2.2</td><td>98.7 ± 7.5</td><td>103.3 ± 9.1</td></tr><tr><td>halfcheetah-m-re</td><td>47.7</td><td>34.6 ± 0.9</td><td>38.3 ± 0.3</td><td>41.9 ± 0.5</td><td>38.1 ± 1.3</td></tr><tr><td>walker2d-m-re</td><td>26.7</td><td>26.6 ± 3.4</td><td>69.1 ± 4.2</td><td>74.9 ± 6.6</td><td>49.5 ± 12.0</td></tr><tr><td>hopper-m-re</td><td>48.6</td><td>23.1 ± 2.7</td><td>78.4 ± 7.2</td><td>92.3 ± 1.1</td><td>97.5 ± 0.7</td></tr><tr><td>halfcheetah-r</td><td>35.4</td><td>2.2 ± 0.0</td><td>5.4 ± 0.3</td><td>8.8 ± 3.8</td><td>3.2 ± 0.1</td></tr><tr><td>walker2d-r</td><td>7.3</td><td>0.9 ± 0.1</td><td>3.7 ± 0.1</td><td>6.2 ± 0.7</td><td>5.6 ± 0.8</td></tr><tr><td>hopper-r</td><td>12.2</td><td>2.0 ± 0.1</td><td>6.6 ± 0.1</td><td>7.9 ± 0.7</td><td>7.5 ± 0.4</td></tr><tr><td>pen-c</td><td>56.9</td><td>46.9 ± 11.0</td><td>65.9 ± 3.6</td><td>57.4 ± 3.5</td><td>60.0 ± 4.1</td></tr><tr><td>hammer-c</td><td>2.1</td><td>0.4 ± 0.1</td><td>2.9 ± 0.5</td><td>0.2 ± 0.1</td><td>2.1 ± 0.7</td></tr><tr><td>relocate-c</td><td>-0.1</td><td>-0.1 ± 0.0</td><td>0.3 ± 0.2</td><td>0.2 ± 0.1</td><td>0.2 ± 0.1</td></tr><tr><td>door-c</td><td>0.4</td><td>0.0 ± 0.1</td><td>0.6 ± 0.6</td><td>0.2 ± 0.7</td><td>0.2 ± 0.3</td></tr></table>

on halfcheetah), where iterative algorithms have a clear advantage. We will discuss potential causes of this further in Section 7.

To give a more direct comparison that controls for any potential implementation details, we use our implementation of reverse KL regularization to create multi-step and iterative algorithms. We are not using algorithmic modifications like Q ensembles, regularized Q values, or early stopping that have been used in prior work. But, our iterative algorithm recovers similar performance to prior regularized actor-critic approaches. These results are shown in Table 2.

Put together, these results immediately suggest some guidance to the practitioner: it is worthwhile to run the one-step algorithm as a baseline before trying something more elaborate. The one-step algorithm is substantially simpler than prior work, but usually achieves better performance.

# 6 What goes wrong for iterative algorithms?

The benchmark experiments show that one step of policy improvement often beats iterative and multi-step algorithms. In this section we dive deeper

to understand why this happens. First, by examining the learning curves of each of the algorithms we note that iterative algorithms require stronger regularization to avoid instability. Then we identify two causes of this instability: distribution shift and iterative error exploitation.

Distribution shift causes evaluation error by reducing the effective sample size in the fixed dataset for evaluating the current policy and has been extensively considered in prior work as discussed below.

Table 2: Results of reverse KL regularization on the D4RL benchmark across one-step, multi-step, and iterative algorithms. Again we run 3 seeds and 6 hyperparameters and report the mean and standard deviation across seeds using 100 evaluation episodes.

<table><tr><td></td><td>One-step</td><td>Multi-step</td><td>Iterative</td></tr><tr><td>halfcheetah-m</td><td>55.2 ± 0.4</td><td>59.3 ± 0.7</td><td>51.2 ± 0.2</td></tr><tr><td>walker2d-m</td><td>85.9 ± 1.4</td><td>74.5 ± 2.8</td><td>74.8 ± 0.7</td></tr><tr><td>hopper-m</td><td>83.7 ± 4.5</td><td>54.8 ± 4.3</td><td>54.7 ± 1.9</td></tr><tr><td>halfcheetah-m-e</td><td>93.8 ± 0.5</td><td>94.2 ± 0.5</td><td>93.7 ± 0.6</td></tr><tr><td>walker2d-m-e</td><td>111.2 ± 0.2</td><td>109.8 ± 0.3</td><td>108.7 ± 0.6</td></tr><tr><td>hopper-m-e</td><td>98.7 ± 7.5</td><td>90.6 ± 18.8</td><td>94.5 ± 11.9</td></tr><tr><td>halfcheetah-r</td><td>8.8 ± 3.8</td><td>18.3 ± 6.5</td><td>21.2 ± 5.2</td></tr><tr><td>walker2d-r</td><td>6.2 ± 0.7</td><td>5.4 ± 0.2</td><td>5.4 ± 0.4</td></tr><tr><td>hopper-r</td><td>7.9 ± 0.7</td><td>21.9 ± 8.9</td><td>9.7 ± 0.4</td></tr></table>

Iterative error exploitation occurs when we repeatedly optimize policies against our Q estimates and exploit their errors. This introduces a bias towards overestimation at each step (much like the training error in supervised learning is biased to be lower than the test error). Moreover, by iteratively re-using the data and using prior Q estimates to warmstart training at each step, the errors from one step are amplified at the next. This type of error is particular to multi-step and iterative algorithms.

# 6.1 Learning curves and hyperparameter sensitivity

To begin to understand why iterative and multi-step algorithms can fail it is instructive to look at the learning curves. As shown in Figure 2, we often observe that the iterative algorithm will begin to learn and then crash. Regularization can help to prevent this crash since strong enough regularization towards the behavior policy ensures that the evaluation is nearly on-policy.

![](images/696be9acacecb76a566d4112f02f4d91b28118ad486ae0fe462d9d1664d76134.jpg)  
Figure 2: Learning curves and final performance on halfcheetah-medium across different algorithms and regularization hyperparameters. Error bars show min and max over 3 seeds. Similar figures for other datasets from D4RL can be found in Appendix D.

![](images/21bde71d2903e849279411ae8a437ed63c6d149fa5eaac0a8bd031df04cff38c.jpg)

![](images/a470a440a64855c092bddbaf4d8b6ffb7d1c3529185a5dd945000bafe11c2d2b.jpg)

![](images/4213432980eaa750ed347eff8412c8f83a26926a7dd6a5fc25d95c2d6ea96d56.jpg)

In contrast, the one-step algorithm is more robust to the regularization hyperparameter. The rightmost panel of the figure shows this clearly. While iterative and multi-step algorithms can have their performance degrade very rapidly with the wrong setting of the hyperparameter, the one-step approach is more stable. Moreover, we usually find that the optimal setting of the regularization hyperparameter is lower for the one-step algorithm than the iterative or multi-step approaches.

# 6.2 Distribution shift

Any algorithm that relies on off-policy evaluation will struggle with distribution shift in the evaluation step. Trying to evaluate a policy that is substantially different from the behavior reduces the effective sample size and increases the variance of the estimates. Explicitly, by distribution shift we mean the shift between the behavior distribution (the distribution over state-action pairs in the dataset) and the evaluation distribution (the distribution that would be induced by the policy  $\pi$  we want to evaluate).

Prior work. There is a substantial body of prior theoretical work that suggests that off-policy evaluation can be difficult and this difficulty scales with some measure of distribution shift. Wang et al. [2020a], Amortila et al. [2020], Zanette [2021] give exponential (in horizon) lower bounds on sample complexity in the linear setting even with good feature representations that can represent the desired Q function and assuming good data coverage. Upper bounds generally require very strong assumptions on both the representation and limits on the distribution shift [Wang et al., 2021, Duan et al., 2020, Chen and Jiang, 2019]. Moreover, the assumed bounds on distribution shift can be exponential in horizon in the worst case. On the empirical side, Wang et al. [2021] demonstrates issues with distribution shift when learning from pre-trained features and provides a nice discussion of why distribution shift causes error amplification. Fujimoto et al. [2018a] raises a similar issue under the name "extrapolation error". Regularization and constraints are meant to reduce issues stemming from distribution shift, but also reduce the potential for improvement over the behavior.

Empirical evidence. Both the multi-step and iterative algorithms in our experiments rely on off-policy evaluation as a key subroutine. We examine how easy it is to evaluate the policies encountered along the learning trajectory. To control for issues of iterative error exploitation (discussed in the next subsection), we train Q estimators from scratch on a heldout evaluation dataset sampled from the behavior policy. We then evaluate these trained Q function on rollouts from 1000 datapoints sampled from the replay buffer. Results are shown in Figure 3.

The results show a correlation between KL and MSE. Moreover, we see that the MSE generally increases over training. One way to mitigate this, as seen in the figure, is to use a large value of  $\alpha$ . We just cannot take a very large step before running into problems with distribution shift. But, when we take such a small step, the information from the on-policy  $\widehat{Q}^{\beta}$  is about as useful as the newly estimated  $\widehat{Q}^{\pi}$ . This is seen, for example, in Figure 2 where we get very similar performance across algorithms at high levels of regularization.

![](images/a75bdcdc807698fb8b34761ab701ac9346b6b9a722900a1f7266aa02fac915b5.jpg)  
Figure 3: Results of running the iterative algorithm on halfcheetah-medium. Each checkpointed policy is evaluated by a Q function trained from scratch on heldout data. MSE refers to  $\mathbb{E}_{s,a\sim \beta}[\hat{Q}^{\pi_i}(s,a) - Q^{\pi_i}(s,a)]$  and KL refers to  $\mathbb{E}_{s\sim \beta}[KL(\pi (\cdot |s)\| \beta (\cdot |s)]$ . Left: 90 policies taken from various points in training with various hyperparameters and random seeds. Center: MSE learning curves. Right: KL learning curves. Error bars show min and max over 3 random seeds.

![](images/d68243a42454fb91152f97ecec7b2f56e5a5b4f4de0432969df2b3feb083a279.jpg)

![](images/23276eda940e5af23a30f3316f743fc2e63c038c5aa9f5765d36e9b663b36d25.jpg)

# 6.3 Iterative error exploitation

The previous subsection identifies how any algorithm that uses off-policy evaluation is fundamentally limited by distribution shift, even if we were given fresh data and trained Q functions from scratch at every iteration. But, in practice, iterative algorithms repeatedly iterate between optimizing policies against estimated Q functions and re-estimating the Q functions using the same data and using the Q function from the previous step to warm-start the re-estimation. This induces dependence between steps that causes a problem that we call iterative error exploitation.

Intuition about the problem. In short, iterative error exploitation happens because  $\pi_{i}$  tends to choose overestimated actions in the policy improvement step, and then this overestimation propagates via dynamic programming in the policy evaluation step. To illustrate this issue more formally, consider the following: at each  $s,a$  we suffer some Bellman error  $\varepsilon_{\beta}^{\pi}(s,a)$  based on our fixed dataset collected by  $\beta$ . Formally,

$$
\widehat {Q} ^ {\pi} (s, a) = r (s, a) + \gamma \underset { \begin{array}{c} s ^ {\prime} | s, a \\ a ^ {\prime} \sim \pi | s ^ {\prime} \end{array} } {\mathbb {E}} [ \widehat {Q} ^ {\pi} \left(s ^ {\prime}, a ^ {\prime}\right) ] + \varepsilon_ {\beta} ^ {\pi} (s, a). \tag {6}
$$

Intuitively,  $\varepsilon_{\beta}^{\pi}$  will be larger at state-action with less coverage in the dataset collected by  $\beta$ . Note that  $\varepsilon_{\beta}^{\pi}$  can absorb all noise due to our finite dataset as well as function approximation error.

All that is needed to cause iterative error exploitation is that the  $\epsilon_{\beta}^{\pi}$  are highly correlated across different  $\pi$ , but for simplicity, we will assume that  $\varepsilon_{\beta}^{\pi}$  is the same for all policies  $\pi$  estimated from our fixed offline dataset and instead write  $\varepsilon_{\beta}$ . Now that the errors do not depend on the policy we can treat the errors as auxiliary rewards that obscure the true rewards and see that

$$
\widehat {Q} ^ {\pi} (s, a) = Q ^ {\pi} (s, a) + \widetilde {Q} _ {\beta} ^ {\pi} (s, a), \quad \widetilde {Q} _ {\beta} ^ {\pi} (s, a) := \underset {\pi | s _ {0}, a _ {0} = s, a} {\mathbb {E}} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} \varepsilon_ {\beta} \left(s _ {t}, a _ {t}\right) \right]. \tag {7}
$$

This assumption is somewhat reasonable since we expect the error to primarily depend on the data. And, when the prior Q function is used to warm-start the current one (as is generally the case in practice), the approximation errors are automatically passed between steps.

Now we can explain the problem. Recall that under our assumption the  $\varepsilon_{\beta}$  are fixed once we have a dataset and likely to have larger magnitude the further we go from the support of the dataset. So, with each step  $\pi_{i}$  is able to better maximize  $\varepsilon_{\beta}$ , thus moving further from  $\beta$  and increasing the magnitude

of  $\widetilde{Q}_{\beta}^{\pi_i}$  relative to  $Q^{\pi_i}$ . Even though  $Q^{\pi_i}$  may provide better signal than  $Q^{\beta}$ , it can easily be drowned out by  $\widetilde{Q}_{\beta}^{\pi_i}$ . In contrast,  $\widetilde{Q}_{\beta}^{\beta}$  has small magnitude, so the one-step algorithm is robust to errors<sup>1</sup>.

An example. Now we consider a simple gridworld example to illustrate iterative error exploitation. This example fits exactly into the setup outlined above since all errors are due to reward estimation so the  $\varepsilon_{\beta}$  is indeed constant over all  $\pi$ . The gridworld we consider has one deterministic good state with reward 1 and many stochastic bad states that have rewards distributed as  $\mathcal{N}(-0.5,1)$ . We collect a dataset of 100 trajectories, each of length 100. One run of the multi-step offline regularized policy iteration algorithm is illustrated in Figure 4.

![](images/2940b4ff6c3e2827fd133e43c716e22be1533f63d9af708d5f955703c4faa90f.jpg)

![](images/1d1c438c297865603a221221d79b76eb77b6847560bfacb730f9ec9d17b94730.jpg)

![](images/d8f45d478bd1b2f5f08361ba61325ebc04d3819299ec97a0feb2ef731b89a98d.jpg)

![](images/a4ecea2370ac2cc168727932b60ca4ea2f9597b29234375b05f2a63fc0719c5a.jpg)

![](images/a3fa1193c72c568698f34a74e6a610239bd36a808acb477da57fe66f9c2c90e6.jpg)

![](images/a8a6c31529c4eda8f7b0b2a2f2abf3adec7d1eadb9e532f3e6e5706ac123ef8e.jpg)  
Figure 4: An illustration of multi-step offline regularized policy iteration. The leftmost panel in each row shows the true reward (top) or error  $\varepsilon_{\beta}$  (bottom). Then each subsequent panel plots  $\pi_{i}$  (with arrow size proportional to  $\pi_{i}(a|s)$ ) over either  $Q^{\pi_i}$  (top) or  $\widetilde{Q}_{\beta}^{\pi}$  (bottom), averaged over actions at each state. The one-step policy  $(\pi_1)$  has the highest value. The behavior policy here is a mixture of optimal  $\pi^{*}$  and uniform  $u$  with coefficient 0.2 so that  $\beta = 0.2\cdot \pi^{*} + 0.8\cdot u$ . We set  $\alpha = 0.1$  as the regularization parameter for reverse KL regularization.

![](images/dc0e7cfc8e36e1063bf1fa6422281d00e7aaf6142a058df99b19f593ce6333f4.jpg)

![](images/54a8050f306eeb4c1d55a748706b905f29a60ed30473339617e733fb730c17b8.jpg)

![](images/b18b0f18b3822883efa23f4194f788052502b3a168b9c9bf88f9b1e57656bdc4.jpg)

![](images/72bf849646807f9a4350304de0f69564b4de60e791628e9f3e979530292339e1.jpg)

In the example, like in the D4RL benchmark, we see that one step outperforms multiple steps of improvement. Intuitively, when there are so many noisy states, it is likely that a few of them will be overestimated. Since the data is re-used for each step, these overestimations persist and propagate across the state space due to iterative error exploitation. This property of having many bad, but poorly estimated states likely also exists in the high-dimensional control problems encountered in the benchmark where there are many ways for the robots to fall down that are not observed in the data for non-random behavior.

Moreover, both settings have larger errors in areas where we have less data. So even though the errors in the gridworld are caused by noise in the rewards, while errors in D4RL are caused by function approximation, we think this is a useful mental model of the problem.

![](images/6d184e3a90276dfea608b15bc79c6678abc98bf8199a7cb1557b21a55f2867a4.jpg)

![](images/0fe2cc96d31b5dece7ca39cc609f9ae592f6b302a8c19d96d35b3806ac40b5db.jpg)

Empirical evidence. In practice we cannot easily visualize the progression of errors. However, the dependence between steps still arises

Figure 5: Histograms of overestimation error  $(\widehat{Q}^{\pi_i}(s,a) - Q^{\pi_i}(s,a))$  on halfcheetah-medium with the iterative algorithm. Left: errors from the training Q function. Right: errors from an independently trained Q function.

as overestimation of the Q values. We can track the overestimation of the Q values over training as a way to measure how much bias is being induced by optimizing against our dependent Q estimators. As a control we can also train Q estimators from scratch on independently sampled evaluation data. These independently trained Q functions do not have the same overestimation bias even though the squared error does tend to increase as the policy moves further from the behavior (as seen in Figure 3). Explicitly, we track 1000 state, action pairs from the replay buffer over training. For each checkpointed policy we perform 3 rollouts at each state to get an estimate of the true Q value and compare this to the estimated Q value. Results are shown in Figure 5.

# 7 When are multiple steps useful?

So far we have focused on why the one-step algorithm often works better than the multi-step and iterative algorithms. However, we do not want to give the impression that one-step is always better. Indeed, our own experiments in Section 5 show a clear advantage for the multi-step and iterative approaches when we have randomly collected data. While we cannot offer a precise delineation of when one-step will outperform multi-step, in this section we offer some intuition as to when we can expect to see benefits from multiple steps of policy improvement.

As seen in Section 6, multi-step and iterative algorithms have problems when they propagate estimation errors. This is especially problematic in noisy and/or high dimensional environments. While the multi-step algorithms propagate this noise more widely than the one-step algorithm, they also propagate the signal. So, when we have sufficient coverage to reduce the magnitude of the noise, this increased propagation of signal can be beneficial. The D4RL experiments suggest that we are usually on the side of the tradeoff where the errors are large enough to make one-step preferable.

In Appendix A we illustrate a simple gridworld example where a slight modification of the behavior policy from Figure 4 makes multi-step dramatically outperform one-step. This modified behavior policy (1) has better coverage of the noisy states (which reduces error, helping multi-step), and (2) does a worse job propagating the reward from the good state (hurting one-step).

We can also test empirically how the behavior policy effects the tradeoff between error and signal propagation. To do this we construct a simple experiment where we mix data from the random behavior policy with data from the medium behavior policy. Explicitly we construct a dataset  $D$  out of the datasets  $D_r$  for random and  $D_m$  for medium such that each trajectory in  $D$  comes from the medium dataset with probability  $p_m$ . So for  $p_m = 0$  we have the random dataset and  $p_m = 1$  we have the

medium dataset, and in between we have various mixtures. Results are shown in Figure 6. It takes surprisingly little data from the medium policy for one-step to outperform the iterative algorithm.

![](images/0cfb618b5e725bdc34370db5af2f7d9a82920d6938f2acd7356a15f65f4d36d8.jpg)  
Figure 6: Performance of all three algorithms with reverse KL regularization across mixtures between halfcheetah-random and halfcheetah-medium. Error bars indicate min and max over 3 seeds.

# 8 Discussion, limitations, and future work

This paper presents the surprising effectiveness of a simple one-step baseline for offline RL. We examine the failure modes of iterative algorithms and the conditions where we might expect them to outperform the simple one-step baseline. This provides guidance to a practitioner that the simple one-step baseline is a good place to start when approaching an offline RL problem.

But, we leave many questions unanswered. One main limitation is that we lack a clear theoretical characterization of which environments and behaviors can guarantee that one-step outperforms multi-step or visa versa. Such results will likely require strong assumptions, but could provide useful insight. We don't expect this to be easy as it requires understanding policy iteration which has been notoriously difficult to analyze, often converging much faster than the theory would suggest [Sutton and Barto, 2018, Agarwal et al., 2019]. Another limitation is that while only using one step is perhaps the simplest way to avoid the problems of off-policy evaluation, there are possibly other more elaborate algorithmic solutions that we did not consider here. However, our strong empirical results suggest that the one-step algorithm is at least a strong baseline.

# References

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International Conference on Machine Learning, pages 22-31. PMLR, 2017.  
Alekh Agarwal, Nan Jiang, and S. Kakade. Reinforcement learning: Theory and algorithms. 2019.  
P. Amortila, Nan Jiang, and Tengyang Xie. A variant of the wang-foster-kakade lower bound for the discounted setting. ArXiv, abs/2011.01075, 2020.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. CoRR, abs/1606.01540, 2016. URL http://arxiv.org/abs/1606.01540.  
Jacob Buckman, Carles Gelada, and Marc G. Bellemare. The importance of pessimism in fixed-dataset policy optimization, 2020.  
Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In Proceedings of the 36th International Conference on Machine Learning. PMLR, 2019.  
Xinyue Chen, Zijian Zhou, Zheng Wang, Che Wang, Yanqiu Wu, and Keith Ross. Bail: Best-action imitation learning for batch deep reinforcement learning. Advances in Neural Information Processing Systems, 33, 2020.  
Yaqi Duan, Zeyu Jia, and Mengdi Wang. Minimax-optimal off-policy evaluation with linear function approximation. In International Conference on Machine Learning, pages 2701-2709. PMLR, 2020.  
Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. arXiv preprint arXiv:1812.02900, 2018a.  
Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018b.  
Scott Fujimoto, Edoardo Conti, Mohammad Ghavamzadeh, and Joelle Pineau. Benchmarking batch deep reinforcement learning algorithms. arXiv preprint arXiv:1910.01708, 2019.  
Wonjoon Goo and Scott Niekum. You only evaluate once - a simple baseline algorithm for offline rl. In Offline Reinforcement Learning Workshop at Neural Information Processing Systems, 2020.  
Caglar Gulcehre, Ziyu Wang, Alexander Novikov, Tom Le Paine, Sergio Gomez Colmenarejo, Konrad Zolna, Rishabh Agarwal, Josh Merel, Daniel Mankowitz, Cosmin Paduraru, et al. Rl unplugged: Benchmarks for offline reinforcement learning. arXiv preprint arXiv:2006.13888, 2020.  
Natasha Jaques, Asma Ghandeharioun, Judy Hanwen Shen, Craig Ferguson, Agata Lapedriza, Noah Jones, Shixiang Gu, and Rosalind Picard. Way off-policy batch deep reinforcement learning of implicit human preferences in dialog, 2019.  
Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In ICML, volume 2, pages 267-274, 2002.  
Ilya Kostrikov, Jonathan Thompson, Rob Fergus, and Ofir Nachum. Offline reinforcement learning with fisher divergence: critical regularization. arXiv preprint arXiv:2103.08050, 2021.  
Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. In Advances in Neural Information Processing Systems, pages 11761-11771, 2019.  
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. arXiv preprint arXiv:2006.04779, 2020.

Romain Laroche, Paul Trichelair, and Remi Tachet Des Combes. Safe policy improvement with baseline bootstrapping. In International Conference on Machine Learning, pages 3652-3661. PMLR, 2019.  
Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.  
Yifei Ma, Yu-Xiang Wang, et al. Imitation-regularized offline learning. arXiv preprint arXiv:1901.04723, 2019.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074, 2019.  
Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas. Hyperparameter selection for offline reinforcement learning, 2020.  
Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.  
Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. arXiv preprint arXiv:1709.10087, 2017.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889-1897, 2015.  
Noah Siegel, Jost Tobias Springenberg, Felix Berkenkamp, Abbas Abdolmaleki, Michael Neunert, Thomas Lampe, Roland Hafner, Nicolas Heess, and Martin Riedmiller. Keep doing what worked: Behavior modelling priors for offline reinforcement learning. In International Conference on Learning Representations, 2020.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In Thirtieth AAAI conference on artificial intelligence, 2016.  
Cameron Voloshin, Hoang M Le, Nan Jiang, and Yisong Yue. Empirical study of off-policy policy evaluation for reinforcement learning. arXiv preprint arXiv:1911.06854, 2019.  
Qing Wang, Jiechao Xiong, Lei Han, Han Liu, Tong Zhang, et al. Exponentially weighted imitation learning for batched historical data. In Advances in Neural Information Processing Systems, pages 6288-6297, 2018.  
Ruosong Wang, Dean P. Foster, and Sham M. Kakade. What are the statistical limits of offline rl with linear function approximation?, 2020a.  
Ruosong Wang, Yifan Wu, Ruslan Salakhutdinov, and Sham M Kakade. Instabilities of offline rl with pre-trained neural representation. arXiv preprint arXiv:2103.04947, 2021.  
Ziyu Wang, Alexander Novikov, Konrad Zolna, Josh S Merel, Jost Tobias Springenberg, Scott E Reed, Bobak Shahriari, Noah Siegel, Caglar Gulcehre, Nicolas Heess, et al. Critic regularized regression. Advances in Neural Information Processing Systems, 33, 2020b.  
Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning, 2019.  
Andrea Zanette. Exponential lower bounds for batch reinforcement learning: Batch rl can be exponentially harder than online rl, 2021.
