# TOWARDS SIMPLICITY IN DEEP REINFORCEMENT LEARNING: STREAMLINED OFF-POLICY LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The field of Deep Reinforcement Learning (DRL) has recently seen a surge in the popularity of maximum entropy reinforcement learning algorithms. Their popularity stems from the intuitive interpretation of the maximum entropy objective and their superior sample efficiency on standard benchmarks. In this paper, we seek to understand the primary contribution of the entropy term to the performance of maximum entropy algorithms. For the Mujoco benchmark, we demonstrate that the entropy term in Soft Actor Critic (SAC) principally addresses the bounded nature of the action spaces. With this insight, we propose a simple normalization scheme which allows a streamlined algorithm without entropy maximization match the performance of SAC. Our experimental results demonstrate a need to revisit the benefits of entropy regularization in DRL. We also propose a simple non-uniform sampling method for selecting transitions from the replay buffer during training. We further show that the streamlined algorithm with the simple non-uniform sampling scheme outperforms SAC and achieves state-of-the-art performance on challenging continuous control tasks.

# 1 INTRODUCTION

Off-policy deep Reinforcement Learning (RL) algorithms aim to improve sample efficiency by reusing past experience. Recently a number of new off-policy Deep Reinforcement Learning algorithms have been proposed for control tasks with continuous state and action spaces, including Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3) (Lillicrap et al., 2015; Fujimoto et al., 2018). TD3, in particular, has been shown to be significantly more sample efficient than popular on-policy methods for a wide range of Mujoco benchmarks.

The field of Deep Reinforcement Learning (DRL) has also recently seen a surge in the popularity of maximum entropy reinforcement learning algorithms. Their popularity stems from the intuitive interpretation of the maximum entropy objective and their superior sample efficiency on standard benchmarks. In particular, Soft Actor Critic (SAC), which combines off-policy learning with maximum-entropy RL, not only has many attractive theoretical properties, but can also give superior performance on a wide-range of Mujoco environments, including on the high-dimensional environment Humanoid for which both DDPG and TD3 perform poorly (Haarnoja et al., 2018a;b; Langlois et al., 2019). The TD3 and SAC algorithms share many common features, including an actor-critic structure, off-policy learning, and the use of double Q-networks (Van Hasselt et al., 2016). The primary difference between the two approaches is that SAC employs maximum entropy reinforcement learning whereas TD3 does not.

In this paper, we first seek to understand the primary contribution of the entropy term to the performance of maximum entropy algorithms. For the Mujoco benchmark, we demonstrate that when using the standard objective without entropy along with standard additive noise exploration, there is often insufficient exploration due to the bounded nature of the action spaces. Specifically, the outputs of the policy network are often way outside the bounds of the action space, so that they need to be squashed to fit within the action space. The squashing results in actions persistently taking on their maximal values, so that there is insufficient exploration. In contrast, the entropy term in the SAC objective forces the outputs to have sensible values, so that even with squashing, exploration is maintained. We conclude that the entropy term in the objective for Soft Actor Critic principally addresses the bounded nature of the action spaces in the Mujoco environments.

With this insight, we propose Streamlined Off Policy (SOP), a streamlined algorithm using the standard objective without the entropy term. SOP employs a simple normalization scheme to address the bounded nature of the action spaces, thereby allowing for satisfactory exploration throughout training. Our experimental results show that SOP matches the sample-efficiency and robustness performance of SAC, including on the more challenging Ant and Humanoid environments. This demonstrates a need to revisit the benefits of entropy maximization in DRL.

Keeping with the theme of simplicity with the goal of meeting Occam's principle, we also propose a simple non-uniform sampling method for selecting transitions from the replay buffer during training. In vanilla SOP (as well as in DDPG, TD3, and SAC), samples from the replay buffer are chosen uniformly at random during training. Our method, called Emphasizing Recent Experience (ERE), samples more aggressively recent experience while not neglecting past experience. Unlike Priority Experience Replay (PER) (Schaul et al., 2015), a popular non-uniform sampling scheme for the Atari environments, ERE is only a few lines of code and does not rely on any sophisticated data structures. We show that SOP combined with ERE out-performs SAC and provides state of the art performance. For example, for Ant and Humanoid, it improves over SAC by  $24\%$  with one million samples. Furthermore, we also investigate combining SOP with PER, and show  $\mathrm{SOP + ERE}$  also out-performs the more complicated  $\mathrm{SOP + PER}$  scheme.

The contributions of this paper are thus threefold. First, we uncover the primary contribution of the entropy term of maximum entropy RL algorithms when the environments have bounded action spaces. Second, we develop a new streamlined algorithm which does not employ entropy maximization but nevertheless matches the sampling efficiency and robustness performance of SAC for the Mujoco benchmarks. And third, we combine our streamlined algorithm with a simple non-uniform sampling scheme to achieve state-of-the-art performance for the Mujoco benchmark. We provide anonymized code for reproducibility<sup>1</sup>.

# 2 PRELIMINARIES

We represent an environment as a Markov Decision Process (MDP) which is defined by the tuple  $(S, \mathcal{A}, r, p, \gamma)$ , where  $S$  and  $\mathcal{A}$  are continuous multi-dimensional state and action spaces,  $r(s, a)$  is a bounded reward function,  $p(s'|s, a)$  is a transition function, and  $\gamma$  is the discount factor. Let  $s(t)$  and  $a(t)$  respectively denote the state of the environment and the action chosen at time  $t$ . Let  $\pi = \pi(a|s)$ ,  $s \in S$ ,  $a \in \mathcal{A}$  denote the policy. We further denote  $K$  for the dimension of the action space, and write  $a_k$  for the  $k$ th component of an action  $a \in \mathcal{A}$ , that is,  $a = (a_1, \dots, a_K)$ .

The expected discounted return for policy  $\pi$  beginning in state  $s$  is given by:

$$
V _ {\pi} (s) = \mathbb {E} _ {\pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r (s (t), a (t)) | s (0) = s \right] \tag {1}
$$

Standard MDP and reinforcement learning problem formulations seek to maximize  $V_{\pi}(s)$  over policies  $\pi$ . For finite state and action spaces, under suitable conditions for continuous state and action spaces, the optimal policy is deterministic (Puterman, 2014; Bertsekas & Tsitsiklis, 1996). In reinforcement learning with unknown environment, exploration is required to learn a suitable policy.

In DRL with continuous action spaces, typically the policy is modeled by a parameterized policy network which takes as input a state  $s$  and outputs a value  $\mu(s; \theta)$ , where  $\theta$  represents the current parameters of the policy network (Schulman et al., 2015; 2017; Vuong et al., 2018; Lillicrap et al., 2015; Fujimoto et al., 2018). During training, the actual action taken when in state  $s$  often takes the form  $a = \mu(s; \theta) + \epsilon$  where  $\epsilon$  is a random  $K$ -dimensional vector which is independently drawn at each time step and may, in some circumstances, also depend on  $\theta$ . During testing,  $\epsilon$  is set to zero.

# 2.1 ENTROPY MAXIMIZATION RL

Maximum entropy reinforcement learning takes a different approach than (1) by optimizing policies to maximize both the expected return and the expected entropy of the policy (Ziebart et al., 2008; Ziebart, 2010; Todorov, 2008; Rawlik et al., 2013; Levine & Koltun, 2013; Levine et al., 2016; Nachum et al., 2017; Haarnoja et al., 2017; 2018a;b).

In particular, with maximization entropy RL, the objective is to maximize

$$
V _ {\pi} (s) = \mathbb {E} _ {\pi} \left[ \sum_ {t = 0} ^ {\infty} \gamma^ {t} r (s (t), a (t)) + \lambda H (\pi (\cdot | s (t))) | s (0) = s \right] \tag {2}
$$

where  $H(\pi (\cdot |\cdot s))$  is the entropy of the policy when in state  $s$ , and the temperature parameter  $\lambda$  determines the relative importance of the entropy term against the reward.

For entropy maximization DRL, when given state  $s$  the policy network will typically output a  $K$ -dimensional vector  $\sigma(s; \theta)$  in addition to the vector  $\mu(s; \theta)$ . The action selected when in state  $s$  is then modeled as  $\mu(s; \theta) + \epsilon$  where  $\epsilon \sim N(0, \sigma(s; \theta))$ .

Maximum entropy RL has been touted to have a number of conceptual and practical advantages for DRL (Haarnoja et al., 2018a,b). For example, it has been argued that the policy is incentivized to explore more widely, while giving up on clearly unpromising avenues. It has also been argued that the policy can capture multiple modes of near-optimal behavior, that is, in problem settings where multiple actions seem equally attractive, the policy will commit equal probability mass to those actions. In this paper, we will highlight another advantage, namely, retaining sufficient exploration when facing bounded action spaces.

# 3 THE SQUASHING EXPLORATION PROBLEM

# 3.1 BOUNDED ACTION SPACES

Continuous environments typically have bounded action spaces, that is, along each action dimension  $k$  there is a minimum possible action value  $a_{k}^{\mathrm{min}}$  and a maximum possible action value  $a_{k}^{\mathrm{max}}$ . When selecting an action, the action needs to be selected within these bounds before the action can be taken. DRL algorithms often handle this by squashing the action so that it fits within the bounds. For example, if along any one dimension the value  $\mu (s;\theta) + \epsilon$  exceeds  $a_{\mathrm{max}}$ , the action is set (clipped) to  $a_{\mathrm{max}}$ . Alternatively, a smooth form of squashing can be employed. For example, suppose  $a_{k}^{\mathrm{min}} = -M$  and  $a_{k}^{\mathrm{max}} = +M$  for some positive number  $M$ , then a smooth form of squashing could use  $a = M\tanh (\mu (s;\theta) + \epsilon)$  in which  $\tanh ()$  is being applied to each component of the  $K$ -dimensional vector. DDPG (Hou et al., 2017) and TD3 (Fujimoto et al., 2018) use clipping, and SAC (Haarnoja et al., 2018a,b) uses smooth squashing with the  $\tanh ()$  function. For concreteness, henceforth we will assume that smooth squashing with the  $\tanh ()$  is employed.

We note that an environment may actually allow the agent to input actions that are outside the bounds. In this case, the environment will typically first clip the actions internally before passing them on to the "actual" environment (Fujita & Maeda, 2018).

We now make a simple but crucial observation: squashing actions so that they fit into a bounded action space can have a disastrous effect on additive-noise exploration strategies. To see this, let the output of the policy network be denoted by  $\mu(s) = (\mu_1(s), \ldots, \mu_K(s))$ . Consider an action taken along one dimension  $k$ , and suppose  $\mu_k(s) >> 1$  and  $|\epsilon_k|$  is relatively small compared to  $\mu_k(s)$ . Then the action  $a_k = M \tanh(\mu_k(s) + \epsilon_k)$  will be very close (essentially equal) to  $M$ . If the condition  $\mu_k(s) >> 1$  persists over many consecutive states, then  $a_k$  will remain close to 1 for all these states, and consequently there will be essentially no exploration along the  $k$ th dimension. We will refer to this problem as the squashing exploration problem. We will argue that algorithms such as DDPG and TD3 based on the standard objective (1) with additive noise exploration can be greatly impaired by squashing exploration.

# 3.2 WHAT DOES ENTROPY MAXIMIZATION BRING TO SAC FOR THE MUJUCO ENVIRONMENTS?

SAC is a maximum-entropy based off-policy DRL algorithm which provides good performance across all of the Mujuco benchmark environments. To the best of our knowledge, it currently provides state of the art performance for the Mujoco benchmark. In this section, we argue that the principle contribution of the entropy term in the SAC objective is to resolve the squashing exploration problem, thereby maintaining sufficient exploration when facing bounded action spaces. To argue this, we consider two DRL algorithms: SAC with adaptive temperature (Haarnoja et al., 2018b), and

SAC with entropy removed altogether (temperature set to zero) but everything else the same. We refer to them as SAC and as SAC without entropy. For SAC without entropy, for exploration we use additive zero-mean Gaussian noise with  $\sigma$  fixed at 0.3. Both algorithms use tanh squashing. We compare these two algorithms on two Mujoco environments: Humanoid-v2 and Walker-v2.

Figure 1 shows the performance of the two algorithms with 10 seeds. We see that for Humanoid, SAC with entropy maximization performs much better than SAC without entropy maximization. However, for Walker, SAC without entropy performs nearly as well as SAC, implying maximum entropy RL is not as critical for this environment.

![](images/3968e0c292984894fcaec8a126008907b0e4d4f1f6913d290b7626648d99c8a0.jpg)  
(a) Humanoid-v2

![](images/67316a354e2a339e484d0b88c5bc9801f1d4ed76b56ab4b52bec06398daf29e9.jpg)  
(b) Walker2d-v2  
Figure 1: SAC performance with and without entropy maximization

To understand why entropy maximization is important for one environment but less so for another, we examine the actions selected when training these two algorithms. Humanoid and Walker have action dimensions  $K = 17$  and  $K = 6$ , respectively. Here we show representative results for one dimension for both environments, and provide the results for all the dimensions in the Appendix. The top and bottom rows of Figure 2 shows results for Humanoid and Walker, respectively. The first column shows the  $\mu_{k}$  values for an interval of 1,000 consecutive time steps, namely, for time steps 599,000 to 600,000. The second column shows the actual action values passed to the environment again for time steps 599,000 to 600,000. The third and fourth columns show a concatenation of 10 such intervals of 1000 time steps, with each interval coming from a larger interval of 100,000 time steps. The first and third columns use a log scale on the y-axis.

The top and bottom rows of Figure 2 are strikingly different. For Humanoid using SAC (which uses entropy maximization), the  $|\mu_k|$  values are small, mostly in the range [-1.5,1.5], and fluctuate significantly. This allows the action values to also fluctuate significantly, providing exploration in the action space. On the other hand, for SAC without entropy the  $|\mu_k|$  values are typically huge, most of which are well outside the interval [-10,10]. This causes the actions  $a_{k}$  to be persistently clustered at either  $M$  or  $-M$ , leading to essentially no exploration along that dimension. As shown in the Appendix, this property (lack of exploration for SAC without entropy maximization) does not hold for just a few dimensions, but instead for all 17 dimensions. For Walker, we see that for both algorithms, the  $\mu_{k}$  values are sensible, mostly in the range [-1,1] and therefore the actions chosen by both algorithms exhibit exploration.

In conclusion, the principle benefit of maximum entropy RL in SAC for the Mujuco environments is that it resolves the squashing exploration problem. For some environments (such as Walker), the outputs of the policy network take on sensible values, so that sufficient exploration is maintained and overall good performance is achieved without the need for entropy maximization. For other environments (such as Humanoid), entropy maximization is needed to reduce the magnitudes of the outputs so that exploration is maintained and overall good performance is achieved.

# 4 STREAMLINED OFF-POLICY (SOP) ALGORITHM

Given the observations in the previous section, a natural question is: is it possible to design a streamlined off policy algorithm that does not employ entropy maximization but offers performance comparable to SAC (which has entropy maximization)?

As we observed in the previous section, without entropy maximization, in some environments the policy network output values  $|\mu_k|$ ,  $k = 1,\dots ,K$  can become persistently huge, which leads to insufficient exploration due to the squashing. A simple solution is to modify the outputs of the policy

![](images/9e1feee8482cce13383bcb6222f471e85c77891a471b5e107bbe253772716baa.jpg)

![](images/dd93986606fb7fc1c00bcfdf3db0ce8e0fee4e0b97dd8e0afefcdb3266165a4b.jpg)  
(a) Humanoid-v2

![](images/957a92a39cb892f4e6e81be81d16ec5c1138ba0d8e847f81ea80f9c2863ea53c.jpg)

![](images/9d63be1f4520fd046f872be7004caabcf7cb404d662749427692c1e83c1ebc40.jpg)

![](images/81f336740e42f8f80a1ce44ac6182daba39a0409bb2f07176a523d2c7bd6dde6.jpg)  
Figure 2:  $\mu_{k}$  and  $a_{k}$  values from SAC and SAC without entropy maximization

![](images/c1d822260d8f3c83bb5a7064348b7755b7ba73b7ad9dff01d2d6687ae1f8f705.jpg)  
(b) Walker2d-v2

![](images/601dd5a5b5b96a9b8519343860decec7a165df31774c45749bd3eceb296e914a.jpg)

![](images/77108e435cb727a71102cf771ac671364dd911572a2b4359a4f83f87eeadbf74.jpg)

network by normalizing the output values when they collectively (across the action dimensions) become too large. To this end, for any  $K$ -dimensional vector  $x = (x_{1}, \ldots, x_{K})$  let  $\|x\|_{p}$  denote the  $L_{p}$  norm of  $x$ . Let  $\beta$  be a constant (hyper parameter) close to 1. The normalization procedure is as follows. Let  $\mu = (\mu_{1}, \ldots, \mu_{K})$  be the output of the original policy network. If  $\|\mu\|_{p} / K > \beta$ , then we reset  $\mu_{k} \gets \mu_{k} K \beta / \|\mu\|_{p}$  for all  $k = 1, \ldots, K$ ; otherwise, we leave  $\mu$  unchanged. With this normalization, we are assured that  $\|\mu\|_{p} / K$  is never greater than  $\beta$ . Henceforth we assume the policy network has been modified with the simple normalization scheme just described.

Our Streamlined Off Policy (SOP) algorithm is described in Algorithm 1. The algorithm is essentially DDPG plus the normalization described above, plus double Q-learning (Van Hasselt et al., 2016) and target policy smoothing (Fujimoto et al., 2018). Another way of looking at it is as TD3 plus the normalization described above, minus the delayed policy updates and the target policy parameters. SOP also uses tanh squashing instead of clipping, since tanh gives somewhat better performance in our experiments. The SOP algorithm is "streamlined" as it has no entropy terms, temperature adaptation, target policy parameters or delayed policy updates.

Algorithm 1 Streamlined Off-Policy  
1: Input: initial policy parameters  $\theta$ , Q-function parameters  $\phi_1, \phi_2$ , empty replay buffer  $\mathcal{D}$   
2: Set target parameters equal to main parameters  $\phi_{\mathrm{targ},\mathrm{i}} \gets \phi_{\mathrm{i}}$  for  $\mathrm{i} = 1, 2$   
3: repeat  
4: Generate an episode using actions  $a = M \tanh(\mu_{\theta}(s) + \epsilon)$  where  $\epsilon \sim \mathcal{N}(0, \sigma_1)$ .  
5: for  $j$  in range(however many updates) do  
6: Randomly sample a batch of transitions,  $B = \{(s, a, r, s)\}$  from  $\mathcal{D}$   
7: Compute targets for Q functions:  
[ y_q(r, s') = r + \gamma \min_{i=1,2} Q_{\phi_{\mathrm{targ},i}}(s', M \tanh(\mu_{\theta}(s') + \delta)) \quad \delta \sim \mathcal{N}(0, \sigma_2) ]  
8: Update Q-functions by one step of gradient descent using  
[ \nabla_{\phi_i} \frac{1}{|B|} \sum_{(s, a, r, s') \in B} (Q_{\phi, i}(s, a) - y_q(r, s'))^2 \text{ for } i = 1, 2 ]  
9: Update policy by one step of gradient ascent using  
[ \nabla_{\theta} \frac{1}{|B|} \sum_{s \in B} Q_{\phi, 1}(s, M \tanh(\mu_{\theta}(s))) ]  
10: Update target networks with  
[ \phi_{\mathrm{targ},\mathrm{i}} \gets \rho \phi_{\mathrm{targ},\mathrm{i}} + (1 - \rho) \phi_i \text{ for } i = 1, 2 ]

# 4.1 EXPERIMENTAL RESULTS FOR SOP

Without performing a careful hyper-parameter search, we found  $\sigma_{1} = \sigma_{2} = 0.3$  and  $\beta = 1.2$  works well for all environments. For the normalization for SOP, we use  $p = 1$ , that is, the  $L_{1}$  norm.

Figure 3 compares SAC (with temperature adaptation (Haarnoja et al., 2018a;b)) with SOP for five of the most challenging Mujuco environments. Using the same baseline code, we train with ten different random seeds for each of the two algorithms. Each algorithm performs five evaluation rollouts every 5000 environment steps. The solid curves correspond to the mean, and the shaded region to the standard deviation of the returns over the ten seeds.

Results show that SOP and SAC have essentially the same sample-efficiency performance and robustness across all environments. This result confirms that when using a simple output normalization in the policy network, the performance of SAC can be achieved without maximum entropy RL.

In the Appendix we provide an ablation study for SOP, which shows a major performance drop when removing either double Q-learning or normalization, whereas removing target policy smoothing (Fujimoto et al., 2018) results in only a small performance drop in some environments.

![](images/b21930cdb2b020c12e61ea14e80da3404bb3afe41af427990ffa1039850b03d3.jpg)

![](images/e9c93bedc0e2adb9249a2763faee778b6b49b0ba13262f4a7b637bb0fd371dc0.jpg)  
(b) Walker2d-v2

![](images/a795554be86476dce9907a42326662c6cec0ca3746488144c97f45ffd09e30cf.jpg)

![](images/b516741716b84bab7081150dcf093b15561d547199c07d701bc62b536c80bc34.jpg)  
(a) Hopper-v2  
(d) Ant-v2  
Figure 3: Streamlined Off-Policy (SOP) versus SAC

![](images/950d200d4de4e4cbd9f54be6c331f05c323a2341689061c5d0b6d200deafe95d.jpg)  
(c) HalfCheetah-v2  
(e) Humanoid-v2

# 5 NON-UNIFORM SAMPLING

We now show how a small change in the sampling scheme for SOP can achieve state of the art performance for the Mujoco benchmark. We call this sampling scheme Emphasizing Recent Experience (ERE). ERE has 3 core features:  $(i)$  It is a general method applicable to any off-policy algorithm;  $(ii)$  It requires no special data structure, is very simple to implement, and has near-zero computational overhead;  $(iii)$  It only introduces one additional important hyperparameter.

The basic idea is: during the parameter update phase, the first mini-batch is sampled from the entire buffer, then for each subsequent mini-batch we gradually reduce our range of sampling to sample more aggressively from more recent data. Specifically, assume that in the current update phase we are to make 1000 mini-batch updates. Let  $N$  be the max size of the buffer. Then for the  $k^{th}$  update, we sample uniformly from the most recent  $c_k$  data points, where  $c_k = N \cdot \eta^k$  and  $\eta \in (0,1]$  is a hyper-parameter that determines how much emphasis we put on recent data.  $\eta = 1$  is uniform sampling. When  $\eta < 1$ ,  $c_k$  decreases as we perform each update.  $\eta$  can be made to adapt to the learning speed of the agent so that we do not have to tune it for each environment.

The effect of such a sampling formulation is twofold. The first is recent data have a higher chance of being sampled. The second is that we do this in an ordered way: we first sample from all the data in the buffer, and gradually shrink the range of sampling to only sample from the most recent data. This scheme reduces the chance of over-writing parameter changes made by new data with parameter changes made by old data (French, 1999; McClelland et al., 1995; McCloskey & Cohen, 1989; Ratcliff, 1990; Robins, 1995). This process allows us to quickly obtain new information

from recent data, and better approximate the value functions near recently-visited states, while still maintaining an acceptable approximation near states visited in the more distant past.

What is the effect of replacing uniform sampling with ERE? First note if we do uniform sampling on a fixed buffer, the expected number of times a data point is sampled is the same for all data points. Now consider a scenario where we have a buffer of size 1000 (FIFO queue), we collect one data point at a time, and we then perform one update with mini-batch size of one. If we start with an empty buffer and sample uniformly, as data fills the buffer, each data point gets less and less chance of being sampled. Specifically, over a period of 1000 updates, the expected number of times the  $t$ th data point is sampled is:  $1 / t + 1 / (t + 1) + \dots + 1 / T$ . Figure 4f shows the expected number of times a data point is sampled as a function of its position in the buffer. We see that older data points have a much higher expected number of times of being sampled compared to newer data points. This is undesirable because when the agent is improving and exploring new areas of the state space; the new data points may contain more interesting information than the old ones, which have already been updated many times.

When we apply the ERE scheme, we effectively skew the curve towards assigning higher expected number of samples for the newer data, allowing the newer data to be frequently sampled soon after being collected, which can accelerate the learning process. Further algorithmic detail and analysis on ERE can be found in the Appendix.

# 5.1 EXPERIMENTAL RESULTS FOR SOP+ERE

Figure 4 compares the performance of SOP,  $\mathrm{SOP + ERE}$  and SAC.  $\mathrm{SOP + ERE}$  learns faster than SAC and vanilla SOP in all Mujoco environments.  $\mathrm{SOP + ERE}$  also greatly improves overall performance for the two most challenging environments, Ant and Humanoid. For SOP we found that fine tuning  $\sigma$  for each environment can give further improvement in sample efficiency, but for fairness of comparison, we use exactly the same hyperparameters for all environments. In table 1, we show the mean test episode return and std across 10 random seeds at 1M timesteps for all environments. The last column displays the percentage improvement of  $\mathrm{SOP + ERE}$  over SAC, showing hat  $\mathrm{SOP + ERE}$  achieves state of the art performance. In both Ant and Humanoid,  $\mathrm{SOP + ERE}$  improves average performance by  $24\%$  over SAC at 1 million timesteps. As for the std,  $\mathrm{SOP + ERE}$  gives lower values, and for Humanoid a higher value.

![](images/8340322eb4b1d36f15dfd4bac8d5fb0c641e56af83e848f748fe09a94a8569cb.jpg)  
(a) Hopper-v2

![](images/7f6d285a051333ddcbc49e5fc8aaa5bfac1ed8687218f0b88b638f6effb071ee.jpg)  
(b) Walker2d-v2

![](images/0d571f2e4202d3957d1e51f1dfed81139ab9974ce6c54726524c10e885bc67ca.jpg)  
(c) HalfCheetah-v2

![](images/dfab034964f0e3fc2e7b0d2c6563acba83c40632a9f6ca0c96e749bd33994669.jpg)  
(d) Ant-v2

![](images/8e4c82cbda965194767dd6397c26c5fe164a560fe1e03f6945e6b58e1969e79f.jpg)  
(e) Humanoid-v2

![](images/486c8883310fc4cbbca435060417a8236de99952676cecf40cdf04395893c50b.jpg)  
(f) Uniform and ERE sampling  
Figure 4: (a) to (e) show Streamlined Off-Policy (SOP) with ERE sampling versus SAC. (f) shows over a period of 1000 updates, the expected number of times the  $t$ th data point is sampled (with  $\eta = 0.996$ ). ERE allows new data to be sampled many times soon after being collected.

Table 1: Performance comparison at one million samples. Last column shows percentage improvement of SOP+ERE over SAC.  

<table><tr><td>Environment</td><td>SAC Adaptive</td><td>SOP</td><td>SOP+ERE</td><td>Improvement</td></tr><tr><td>Hopper</td><td>3161.2 ± 381.0</td><td>3277.3 ± 162.5</td><td>3378.9 ± 180.7</td><td>6.9%</td></tr><tr><td>Walker</td><td>4801.5 ± 514.5</td><td>4546.7 ± 491.4</td><td>5291.2 ± 557.9</td><td>10.2%</td></tr><tr><td>HalfCheetah</td><td>10,963.7 ± 512.4</td><td>9,945.5 ± 599.5</td><td>11,786.1 ± 632.7</td><td>7.5%</td></tr><tr><td>Ant</td><td>4153.7 ± 925.0</td><td>4250.8 ± 602.8</td><td>5145.3 ± 319.2</td><td>23.9%</td></tr><tr><td>Humanoid</td><td>5076.2 ± 148.1</td><td>4998.1 ± 106.6</td><td>6297.8 ± 516.4</td><td>24.1%</td></tr></table>

# 6 RELATED WORK

In recent years, there has been significant progress in improving the sample efficiency of DRL for continuous robotic locomotion tasks with off-policy algorithms (Lillicrap et al., 2015; Fujimoto et al., 2018; Haarnoja et al., 2018a;b). There is also a significant body of research on maximum entropy RL methods (Ziebart et al., 2008; Ziebart, 2010; Todorov, 2008; Rawlik et al., 2013; Levine & Koltun, 2013; Levine et al., 2016; Nachum et al., 2017; Haarnoja et al., 2017; 2018a;b).

By taking clipping in the Mujoco environments explicitly into account, Fujita & Maeda (2018) modified the policy gradient algorithm to reduce variance and provide superior performance among on-policy algorithms. Eisenach et al. (2018) extend the work of Fujita & Maeda (2018) for when an action may be direction. Hausknecht & Stone (2015) and Chou et al. (2017) also explores DRL in the context of bounded action spaces. Dalal et al. (2018) consider safe exploration in the context of constrained action spaces.

Uniform sampling is the most common way to sample from a replay buffer. One of the most well-known alternatives is prioritized experience replay (PER) (Schaul et al., 2015). PER uses the absolute TD-error of a data point as the measure for priority, and data points with higher priority will have a higher chance of being sampled. This method has been tested on DQN (Mnih et al., 2015) and double DQN (DDQN) (Van Hasselt et al., 2016) with significant improvement. PER has been combined with theueling architecture (Wang et al., 2015), with an ensemble of recurrent DQN (Schulze & Schulze, 2018), and PER is one of six crucial components in Rainbow (Hessel et al., 2018), which achieves state-of-the-art on the Atari game environments. PER has also been successfully applied to other algorithms such as DDPG (Hou et al., 2017) and can be implemented in a distributed manner (Horgan et al., 2018). There are other methods proposed to make better use of the replay buffer. In Sample Efficient Actor-Critic with Experience Replay (ACER), the algorithm has an on-policy part and an off-policy part, with a hyper-parameter controlling the ratio of off-policy updates to on-policy updates (Wang et al., 2016). The RACER algorithm (Novati & Koumoutsakos, 2018) selectively removes data points from the buffer, based on the degree of "off-policyness" which is measured by their importance sampling weight, bringing improvement to DDPG (Lillicrap et al., 2015), NAF (Gu et al., 2016) and PPO (Schulman et al., 2017). In De Bruin et al. (2015), replay buffers of different sizes were tested on DDPG, and result shows that a large enough buffer with enough data diversity can lead to better performance. Finally, with Hindsight Experience Replay (HER)(Andrychowicz et al., 2017), priority can be given to trajectories with lower density estimation(Zhao & Tresp, 2019) to tackle multi-goal, sparse reward environments.

# 7 CONCLUSION

In this paper we first showed that the primary role of maximum entropy RL for the Mujoco benchmark is to maintain satisfactory exploration in the presence of bounded action spaces. We then developed a new streamlined algorithm which does not employ entropy maximization but nevertheless matches the sampling efficiency and robustness performance of SAC for the Mujoco benchmarks. Our experimental results demonstrate a need to revisit the benefits of entropy regularization in DRL. Finally, we combined our streamlined algorithm with a simple non-uniform sampling scheme to achieve state-of-the-art performance for the Mujoco benchmark.

# REFERENCES

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems, pp. 5048-5058, 2017.  
Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming, volume 5. Athena Scientific Belmont, MA, 1996.  
Po-Wei Chou, Daniel Maturana, and Sebastian Scherer. Improving stochastic policy gradients in continuous control with deep reinforcement learning using the beta distribution. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 834-843. JMLR.org, 2017.  
Gal Dalal, Krishnamurthy Dvijotham, Matej Vecerik, Todd Hester, Cosmin Paduraru, and Yuval Tassa. Safe exploration in continuous action spaces. arXiv preprint arXiv:1801.08757, 2018.  
Tim De Bruin, Jens Kober, Karl Tuyls, and Robert Babuška. The importance of experience replay database composition in deep reinforcement learning. In Deep reinforcement learning workshop, NIPS, 2015.  
Carson Eisenach, Haichuan Yang, Ji Liu, and Han Liu. Marginal policy gradients: A unified family of estimators for bounded action spaces with applications. arXiv preprint arXiv:1806.05134, 2018.  
Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128-135, 1999.  
Justin Fu, Aviral Kumar, Matthew Soh, and Sergey Levine. Diagnosing bottlenecks in deep q-learning algorithms. arXiv preprint arXiv:1902.10250, 2019.  
Scott Fujimoto, Herke van Hoof, and Dave Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018.  
Yasuhiro Fujita and Shin-ichi Maeda. Clipped action policy gradient. arXiv preprint arXiv:1802.07564, 2018.  
Shixiang Gu, Timothy Lillicrap, Ilya Sutskever, and Sergey Levine. Continuous deep q-learning with model-based acceleration. In International Conference on Machine Learning, pp. 2829-2838, 2016.  
Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1352-1361. JMLR.org, 2017.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018a.  
Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018b.  
Matthew Hausknecht and Peter Stone. Deep reinforcement learning in parameterized action space. arXiv preprint arXiv:1511.04143, 2015.  
Matteo Hessel, Joseph Modayil, Hado Van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, and David Silver. Rainbow: Combining improvements in deep reinforcement learning. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Dan Horgan, John Quan, David Budden, Gabriel Barth-Maron, Matteo Hessel, Hado Van Hasselt, and David Silver. Distributed prioritized experience replay. arXiv preprint arXiv:1803.00933, 2018.

Yuenan Hou, Lifeng Liu, Qing Wei, Xudong Xu, and Chunlin Chen. A novel ddpg method with prioritized experience replay. In 2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 316-321. IEEE, 2017.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Eric Langlois, Shunshi Zhang, Guodong Zhang, Pieter Abbeel, and Jimmy Ba. Benchmarking model-based reinforcement learning. arXiv preprint arXiv:1907.02057, 2019.  
Sergey Levine and Vladlen Koltun. Guided policy search. In International Conference on Machine Learning, pp. 1-9, 2013.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334-1373, 2016.  
Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
James L McClelland, Bruce L McNaughton, and Randall C O'reilly. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. *Psychological review*, 102(3):419, 1995.  
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In *Psychology of learning and motivation*, volume 24, pp. 109-165. Elsevier, 1989.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems, pp. 2775-2785, 2017.  
Guido Novati and Petros Koumoutsakos. Remember and forget for experience replay. arXiv preprint arXiv:1807.05827, 2018.  
Martin L Puterman. Markov Decision Processes.: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 2014.  
Roger Ratcliff. Connectionist models of recognition memory: constraints imposed by learning and forgetting functions. Psychological review, 97(2):285, 1990.  
Konrad Rawlik, Marc Toussaint, and Sethu Vijayakumar. On stochastic optimal control and reinforcement learning by approximate inference. In Twenty-Third International Joint Conference on Artificial Intelligence, 2013.  
Anthony Robins. Catastrophic forgetting, rehearsal and pseudorehearsal. _Connection Science_, 7(2): 123-146, 1995.  
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, pp. 1889-1897, 2015.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Christopher Schulze and Marcus Schulze. Vizdoom: Drqn with prioritized experience replay, double-q learning and snapshot ensembling. In Proceedings of SAI Intelligent Systems Conference, pp. 1-17. Springer, 2018.

Emanuel Todorov. General duality between optimal control and estimation. In 2008 47th IEEE Conference on Decision and Control, pp. 4286-4292. IEEE, 2008.  
Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. In AAAI, volume 2, pp. 5. Phoenix, AZ, 2016.  
Quan Vuong, Yiming Zhang, and Keith W Ross. Supervised policy update for deep reinforcement learning. arXiv preprint arXiv:1805.11706, 2018.  
Ziyu Wang, Tom Schaul, Matteo Hessel, Hado Van Hasselt, Marc Lanctot, and Nando De Freitas. *Dueling network architectures for deep reinforcement learning.* arXiv preprint arXiv:1511.06581, 2015.  
Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with experience replay. arXiv preprint arXiv:1611.01224, 2016.  
Rui Zhao and Volker Tresp. Curiosity-driven experience prioritization via density estimation. arXiv preprint arXiv:1902.08039, 2019.  
Brian D Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. PhD thesis, figshare, 2010.  
Brian D Ziebart, Andrew Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. 2008.
