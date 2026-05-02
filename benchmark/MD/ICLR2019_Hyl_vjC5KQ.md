# HIERARCHICAL REINFORCEMENT LEARNING VIA ADVANTAGE-WEIGHTED INFORMATION MAXIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Real-world tasks are often highly structured. Hierarchical reinforcement learning (HRL) has attracted research interest as an approach for leveraging the hierarchical structure of a given task in reinforcement learning (RL). However, identifying the hierarchical policy structure that enhances the performance of RL in not a trivial task. In this paper, we propose an HRL method that learns a latent variable of a hierarchical policy using mutual information maximization. Our approach can be interpreted as a way to learn a discrete and latent representation of the state-action space in an unsupervised manner. To estimate the density of states and actions induced by the unknown optimal policy, we introduce advantage-weighted importance sampling. In our HRL method, the gating policy learns to select option policies based on an option-value function, and these option policies are optimized based on the deterministic policy gradient method. This framework is derived by leveraging the analogy between a monolithic policy in standard RL and a hierarchical policy in HRL by using a deterministic option policy. Experimental results indicate that our HRL approach can learn a diversity of options and that it can enhance the performance of RL in continuous control tasks.

# 1 INTRODUCTION

Reinforcement learning (RL) has been successfully applied to a variety of tasks, including board games (Silver et al., 2016), robotic manipulation tasks (Levine et al., 2016), and video games (Mnih et al., 2015). Hierarchical reinforcement learning (HRL) is a type of RL that leverages the hierarchical structure of a given task by learning a hierarchical policy (Sutton et al., 1999; Dietterich, 2000). Past studies in this field have shown that HRL can solve challenging tasks in the video game domain (Vezhnevets et al., 2017; Bacon et al., 2017) and robotic manipulation (Daniel et al., 2016; Osa et al., 2016). In HRL, lower-level policies, which are often referred to as option policies, learn different behavior/control patterns, and the upper-level policy, which is often referred to as the gating policy, learns to select option policies. Recent studies have developed HRL methods using deep learning (Goodfellow et al., 2016) and have shown that HRL can yield impressive performance for complex tasks (Bacon et al., 2017; Harb & P. Bacon, 2018; Frans et al., 2018; Vezhnevets et al., 2017; Haarnoja et al., 2018). However, identifying the hierarchical policy structure that yields efficient learning is not a trivial task. The problem involves learning a sufficient variety of types of behavior to solve a given task.

In this study, we present an HRL method via the mutual information maximization with advantage-weighted importance, which we refer to as adInfoHRL. We formulate the problem of learning a latent variable in a hierarchical policy as one of learning discrete and interpretable representations of states and actions. To estimate mutual information with respect to the density induced by an unknown optimal policy, we introduce advantage-weighted importance weights. Our approach can be considered to divide the state-action space based on an information maximization criterion, and it learns option policies corresponding to each region of the state-action space. We derive adInfoHRL as an HRL method based on deterministic option policies that are trained based on an extension of the deterministic policy gradient (Silver et al., 2014; Fujimoto et al., 2018). The contributions of this paper are threefold:

1. We propose the learning of a latent variable of a hierarchical policy as a discrete and hidden representation of the state-action space in an unsupervised manner. To estimate mutual information with respect to the density of state and action induced by an unknown optimal policy, we introduce advantage-weighted importance.  
2. We propose an HRL method, where the option policies are optimized based on the deterministic policy gradient and the gating policy selects the option that maximizes the expected return.  
3. We implemented our HRL approach based on Twin Delayed Deep Deterministic policy gradient algorithm (TD3) proposed by Fujimoto et al. (2018). The results show that our proposed method adInfoHRL can learn a diversity of options on continuous control tasks. Moreover, our approach can improve the performance of TD3 on such tasks as the Walker2d and Ant tasks in OpenAI Gym with MuJoco simulator.

# 2 BACKGROUND

In this section, we formulate the problem of HRL in this paper and describe methods related to our proposal.

# 2.1 HIERARCHICAL REINFORCEMENT LEARNING

We consider tasks that can be modeled as a Markov decision process (MDP), consisting of a state space  $S$ , an action space  $\mathcal{A}$ , a reward function  $r: S \times \mathcal{A} \mapsto \mathbb{R}$ , an initial state distribution  $\rho(\pmb{s}_0)$ , and a transition probability  $p(\pmb{s}_{t+1} | \pmb{s}_t, \pmb{a}_t)$  that defines the probability of transitioning from state  $\pmb{s}_t$  and action  $\pmb{a}_t$  at time  $t$  to next state  $\pmb{s}_{t+1}$ . The return is defined as  $R_t = \sum_{i=t}^{T} \gamma^{i-t} r(\pmb{s}_i, \pmb{a}_i)$ , where  $\gamma$  is a discount factor, and policy  $\pi(\pmb{a} | \pmb{s})$  is defined as the density of action  $\pmb{a}$  given state  $\pmb{s}$ . Let  $d^\pi(\pmb{s}) = \sum_{t=0}^{T} \gamma^t p(\pmb{s}_t = \pmb{s})$  denote the discounted visitation frequency induced by the policy  $\pi$ . The goal of reinforcement learning is to learn a policy that maximizes the expected return  $J(\pi) = \mathbb{E}_{\pmb{s}_0, \pmb{a}_0, \dots} [R_0]$  where  $s_0 \sim \rho(\pmb{s}_0), \pmb{a} \sim \pi$  and  $s_{t+1} \sim p(\pmb{s}_{t+1} | \pmb{s}_t, \pmb{a}_t)$ . By defining the Q-function as  $Q^\pi(\pmb{s}, \pmb{a}) = \mathbb{E}_{\pmb{s}_0, \pmb{a}_0, \dots} [R_t | \pmb{s}_t = \pmb{s}, \pmb{a}_t = \pmb{a}]$ , the objective function of reinforcement learning can be rewritten as follows:

$$
J (\pi) = \iint d ^ {\pi} (\boldsymbol {s}) \pi (\boldsymbol {a} | \boldsymbol {s}) Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) \mathrm {d} \boldsymbol {a} \mathrm {d} \boldsymbol {s}. \tag {1}
$$

Herein, we consider hierarchical policy as given below

$$
\pi (\boldsymbol {a} | \boldsymbol {s}) = \sum_ {\boldsymbol {o} \in \mathcal {O}} \pi (\boldsymbol {o} | \boldsymbol {s}) \pi (\boldsymbol {a} | \boldsymbol {s}, o), \tag {2}
$$

where  $o$  is the latent variable and  $\mathcal{O}$  is the set of possible values of  $o$ . Many existing HRL methods employ a policy structure of this form (Frans et al., 2018; Vezhnevets et al., 2017; Bacon et al., 2017; Florensa et al., 2017; Daniel et al., 2016; Osa & Sugiyama, 2018). In general, latent variable  $o$  can be discrete (Frans et al., 2018; Bacon et al., 2017; Florensa et al., 2017; Daniel et al., 2016; Osa & Sugiyama, 2018) or continuous (Vezhnevets et al., 2017).  $\pi(o|s)$  is often referred to as a gating policy (Daniel et al., 2016; Osa & Sugiyama, 2018), policy over options (Bacon et al., 2017), or manager (Vezhnevets et al., 2017). Likewise,  $\pi(a|s, o)$  is often referred to as an option policy (Osa & Sugiyama, 2018), sub-policy (Daniel et al., 2016), or worker (Vezhnevets et al., 2017). In HRL, the objective function is given by

$$
J (\pi) = \iint d ^ {\pi} (\boldsymbol {s}) \sum_ {o \in \mathcal {O}} \pi (o | \boldsymbol {s}) \pi (\boldsymbol {a} | \boldsymbol {s}, o) Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) \mathrm {d} \boldsymbol {a} \mathrm {d} \boldsymbol {s}. \tag {3}
$$

As discussed in the literature on inverse RL (Ziebart, 2010), multiple policies can yield equivalent expected returns. This indicates that there exist multiple solutions to latent variable  $o$  that maximizes the expected return. To obtain the preferable solution for  $o$ , we need to impose additional constraints in HRL. Although prior work has employed regularizers (Bacon et al., 2017) and constraints (Daniel et al., 2016) to obtain various option policies, the method of learning a good latent variable  $o$  that improves sample-efficiency of the learning process remains unclear. In this study we propose the learning of the latent variable by maximizing mutual information between latent variables and state-action pairs.

# 2.2 REPRESENTATION LEARNING VIA INFORMATION MAXIMIZATION

Recent studies such as those by Chen et al. (2016); Hu et al. (2017); Li et al. (2017) have shown that an interpretable representation can be learned by maximizing mutual information. Given a dataset  $X = (x_{1},\dots,x_{n})$ , regularized information maximization (RIM) proposed by Gomes et al. (2010) involves learning a conditional model  $\hat{p}(y|\boldsymbol{x};\boldsymbol{\eta})$  with parameter vector  $\boldsymbol{\eta}$  that predicts a label  $y$ . The objective of RIM is to minimize

$$
\ell (\boldsymbol {\eta}) - \lambda I _ {\boldsymbol {\eta}} (\boldsymbol {x}; y), \tag {4}
$$

where  $\ell(\eta)$  is the regularization term,  $I_{\eta}(\pmb{x};y)$  is mutual information, and  $\lambda$  is a coefficient. The mutual information can be decomposed as  $I_{\eta}(\pmb{x};y) = H(y) - H(y|\pmb{x})$  where  $H(y)$  is entropy and  $H(y|\pmb{x})$  the conditional entropy. Increasing  $H(y)$  conduces the label to be uniformly distributed, and decreasing  $H(y|\pmb{x})$  conduces to clear cluster assignments. Although RIM was originally developed for unsupervised clustering problems, the concept is applicable to various problems that require learning a hidden discrete representation. In this study, we formulate the problem of learning the latent variable  $o$  of a hierarchical policy as one of learning a latent representation of the state-action space.

# 2.3 DETERMINISTIC POLICY GRADIENT

The deterministic policy gradient (DPG) algorithm was developed for learning a monolithic deterministic policy  $\mu_{\theta}(s): S \mapsto \mathcal{A}$  by Silver et al. (2014). In off-policy RL, the objective is to maximize the expectation of the return, averaged over the state distribution induced by a behavior policy  $\beta(\boldsymbol{a}|\boldsymbol{s})$ :

$$
J (\pi) = \iint d ^ {\beta} (s) \pi (a | s) Q ^ {\pi} (s, a) d a d s. \tag {5}
$$

When a policy is deterministic, the objective becomes  $J(\pi) = \int d^{\beta}(s)Q^{\pi}\bigl (s,\pmb {\mu}_{\theta}(s)\bigr)\mathrm{d}s$ . Silver et al. (2014) have shown that the gradient of a deterministic policy is given by

$$
\nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {s} \sim d ^ {\beta} (\boldsymbol {s})} [ Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) ] = \mathbb {E} _ {\boldsymbol {s} \sim d ^ {\beta} (\boldsymbol {s})} \left[ \nabla_ {\boldsymbol {\theta}} \boldsymbol {\mu} _ {\boldsymbol {\theta}} (\boldsymbol {s}) \nabla_ {\boldsymbol {a}} Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) | _ {\boldsymbol {a} = \boldsymbol {\mu} _ {\boldsymbol {\theta}} (\boldsymbol {s})} \right]. \tag {6}
$$

The DPG algorithm has been extended to the deep deterministic policy gradient (DDPG) for continuous control problems that require neural network policies (Lillicrap et al., 2016). TD3 proposed by Fujimoto et al. (2018) is a variant of DDPG that outperforms the state-of-the-art on-policy methods such as TRPO (Schulman et al., 2017a) and PPO (Schulman et al., 2017b) in certain domains. We extend this deterministic policy gradient to learn a hierarchical policy.

# 3 HRL VIA ADVANTAGE-WEIGHTED INFORMATION MAXIMIZATION

In this section, we propose a novel HRL method based on advantage-weighted information maximization. We first introduce the latent representation learning via advantage-weighted information maximization, and we then describe the HRL framework based on deterministic option policies.

Advantage-Weighted Importance Although prior work has often considered  $H(o|s)$  or  $I(s;o)$ , which results in a division of the state space, we are interested in using  $I\big((s,a);o\big)$  for dividing the state-action space instead. A schematic sketch of our approach is shown in Figure 1. As shown in the left side of Figure 1, the advantage function often has multiple modes. Ideally, each option policies should correspond to separate modes of the advantage function. However, it is non-trivial to find the modes of the advantage function in practice. For this purpose, we introduce advantage-weighted importance.

We assume that an optimal policy is of the form

$$
\pi (\boldsymbol {a} | \boldsymbol {s}) = \frac {\exp (A ^ {\pi} (\boldsymbol {s} , \boldsymbol {a}))}{Z}, \tag {7}
$$

where  $A^{\pi}(\pmb{s},\pmb{a}) = Q^{\pi}(\pmb{s},\pmb{a}) - V^{\pi}(\pmb{s})$  is the advantage function,  $V^{\pi}(s)$  is the state value function, and  $Z$  is the partition function. A policy with this form has been often employed in the RL literature (Akrour et al., 2016; Deisenroth et al., 2013). When following such a policy, an action with the

![](images/2083133a58675b51a07bf3c27071e77b646d77ec61d054d28ef5947d48bc838f.jpg)  
Advantage function with multiple modes

![](images/c4d2afccdd76b4b2768127662b6010698ef2b0cd690a9aaa145c0caa78076656.jpg)  
Figure 1: Schematic sketch of our HRL approach. By using the advantage-weighted importance, finding the modes of the advantage-function can be reduced to the problem of finding the modes of the density induced by the optimal policy.

![](images/6e0c797d326706dae88540adb173e3fbe7240ddf4a888fc275ce5d25c4dea975.jpg)

![](images/c175013a55f1aec12b4f1b3b219c662daddbbc6d626cd393bdb94794cbee3154.jpg)  
Sample density drawn from an arbitrary mass function  
Sample density with advantage-weighted importance weight

larger advantage is drawn with a higher probability. Under this assumption, finding the modes of the advantage function is equivalent to finding those of the density of induced by the optimal policy. Thus, finding the modes of the advantage function can be reduced to the problem of clustering the samples induced by the optimal policy.

However, samples induced by the optimal policy are not available in practice. Although those induced by non-optimal policies are available, the clusters obtained from such samples do not correspond to the modes of the advantage function. To estimate the density induced by an unknown optimal policy, we employ an importance sampling approach. We assume that the change of the state distribution induced by the policy update is sufficiently small, namely,  $d^{\pi_{\mathrm{new}}}(s) \approx d^{\pi_{\mathrm{old}}}(s)$ . Then, the importance weight can be approximated as

$$
W (\boldsymbol {s}, \boldsymbol {a}) = \frac {p ^ {\pi_ {\text {n e w}}} (\boldsymbol {s} , \boldsymbol {a})}{p ^ {\beta} (\boldsymbol {a} , \boldsymbol {s})} = \frac {d ^ {\pi_ {\text {n e w}}} (\boldsymbol {s}) \pi^ {\text {n e w}} (\boldsymbol {a} | \boldsymbol {s})}{d ^ {\pi} (\boldsymbol {s}) \beta (\boldsymbol {a} | \boldsymbol {s})} \approx \frac {\pi^ {\text {n e w}} (\boldsymbol {a} | \boldsymbol {s})}{\beta (\boldsymbol {a} | \boldsymbol {s})} = \frac {\exp (A (\boldsymbol {s} , \boldsymbol {a}))}{Z \beta (\boldsymbol {a} | \boldsymbol {s})}, \tag {8}
$$

and the normalized importance weight is given by

$$
\tilde {W} (\boldsymbol {s}, \boldsymbol {a}) = \frac {W (\boldsymbol {s} , \boldsymbol {a})}{\sum_ {j = 1} ^ {N} W (\boldsymbol {s} _ {j} , \boldsymbol {a} _ {j})} = \frac {\frac {\exp (A (\boldsymbol {s} , \boldsymbol {a}))}{Z \beta (\boldsymbol {a} | \boldsymbol {s})}}{\sum_ {j = 1} ^ {N} \frac {\exp (A (\boldsymbol {s} _ {j} , \boldsymbol {a} _ {j}))}{Z \beta (\boldsymbol {a} _ {j} | \boldsymbol {s} _ {j})}} = \frac {\frac {\exp (A (\boldsymbol {s} , \boldsymbol {a}))}{\beta (\boldsymbol {a} | \boldsymbol {s})}}{\sum_ {j = 1} ^ {N} \frac {\exp (A (\boldsymbol {s} _ {j} , \boldsymbol {a} _ {j}))}{\beta (\boldsymbol {a} _ {j} | \boldsymbol {s} _ {j})}}. \tag {9}
$$

As the partition function  $Z$  is canceled, we do not need to compute  $Z$  when computing the importance weight in practice. We call this importance weight  $W$  the advantage-weighted importance and employ it to compute the objective function used to estimate the latent variable.

Latent Representation Learning via Advantage-Weighted Information Maximization To enable the learning of the latent variable  $o$  of a hierarchical policy, we train a neural network that estimates  $p(o|s,a;\eta)$  parameterized with vector  $\eta$ , which we refer to as the option network. To minimize mutual information with respect to the density induced by an unknown optimal policy, the objective function of the option network is computed using the advantage-weighted importance. The option network is trained to minimize

$$
\mathcal {L} _ {\text {o p t i o n}} (\boldsymbol {\eta}) = D _ {\mathrm {K L}} \left(p (o | \boldsymbol {s} ^ {\text {n o i s e}}, \boldsymbol {a} ^ {\text {n o i s e}}; \boldsymbol {\eta}) | | p (o | \boldsymbol {s}, \boldsymbol {a}; \boldsymbol {\eta})\right) + \lambda \left(\hat {H} (o | \boldsymbol {s}, \boldsymbol {a}; \boldsymbol {\eta}) - \hat {H} (o; \boldsymbol {\eta})\right), \tag {10}
$$

where  $s^{\mathrm{noise}} = s + \epsilon_s$ ,  $a^{\mathrm{noise}} = a + \epsilon_a$ ,  $\epsilon_s$  and  $\epsilon_a$  denotes white noise. The first term penalizes dissimilarity between the original state-action pair and a perturbed one. It penalizes the neural network to learn a meaningful representation (Hu et al., 2017). To compute the entropy terms, we employ advantage-weighted importance.  $\hat{H}(o; \eta) = -\hat{p}(o; \eta) \log \hat{p}(o; \eta)$  is the empirical estimate of the entropy  $H(o)$ , where  $\hat{p}(o; \eta)$  is given by

$$
\hat {p} (o; \boldsymbol {\eta}) = \frac {1}{N} \sum_ {i = 1} ^ {N} W \left(\boldsymbol {s} _ {i}, \boldsymbol {a} _ {i}\right) p (o | \boldsymbol {s} _ {i}, \boldsymbol {a} _ {i}; \boldsymbol {\eta}). \tag {11}
$$

Likewise,  $\hat{H}(o|s, \boldsymbol{a}; \boldsymbol{\eta})$  is the empirical estimate of the conditional entropy  $H(o|s, \boldsymbol{a})$  given by

$$
\hat {H} (o | \boldsymbol {s}, \boldsymbol {a}; \boldsymbol {\eta}) = \frac {1}{N} \sum_ {i} ^ {N} W (\boldsymbol {s} _ {i}, \boldsymbol {a} _ {i}) p (o | \boldsymbol {s} _ {i}, \boldsymbol {a} _ {i}; \boldsymbol {\eta}) \log p (o | \boldsymbol {s} _ {i}, \boldsymbol {a} _ {i}; \boldsymbol {\eta}). \tag {12}
$$

The derivations of Equations (11) and (12) are provided in Appendix A. To train the option network, we store the samples collected by the  $M$  most recent behavior policies, to which we refer as on-policy buffer  $D_{\mathrm{on}}$ . Although the algorithm works with entire samples stored in the replay buffer, we observe that the use of the on-policy buffer for latent representation learning exhibits better performance. For this reason, we decided to use the on-policy buffer in our implementation. Therefore, while the algorithm is off-policy in the sense that the option is learned from samples collected by behavior policies, our implementation is "semi"on-policy in the sense that we use samples collected by the most recent behavior policies.

HRL Objective with Deterministic Option Policies Instead of stochastic option policies, we consider deterministic option policies and model them using separate neural networks. We denote by  $\pi (\pmb {a}|\pmb {s},o) = \pmb{\mu}_{\pmb{\theta}}^{o}(\pmb {s})$  deterministic option policies parameterized by vector  $\pmb{\theta}$ . The objective function of off-policy HRL with deterministic option policies can then be obtained by replacing  $\pi (\pmb {a}|\pmb {s})$  with  $\sum_{o\in \mathcal{O}}\pi (o|\pmb {s})\pi (\pmb {a}|\pmb {s},o)$  in Equation (5):

$$
J (\boldsymbol {w}, \boldsymbol {\theta}) = \int d ^ {\beta} (\boldsymbol {s}) \sum_ {o \in \mathcal {O}} \pi (o | \boldsymbol {s}) Q ^ {\pi} \left(\boldsymbol {s}, \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s}); \boldsymbol {w}\right) \mathrm {d} \boldsymbol {s}, \tag {13}
$$

where  $Q^{\pi}(s, a; \boldsymbol{w})$  is an approximated Q-function parameterized using vector  $\boldsymbol{w}$ . This form of the objective function is analogous to Equation (5). Thus, we can extend standard RL techniques to the learning of the gating policy  $\pi(o|s)$  in HRL with deterministic option policies.

In HRL, the goal of the gating policy is to generate a value of  $o$  that maximizes the conditional expectation of the return:

$$
Q _ {\Omega} ^ {\pi} (\boldsymbol {s}, o) = \mathbb {E} [ R | \boldsymbol {s} _ {t} = \boldsymbol {s}, o _ {t} = o ] = \int \pi (\boldsymbol {a} | \boldsymbol {s}, o) Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) \mathrm {d} \boldsymbol {a}, \tag {14}
$$

which is often referred to as the option-value function (Sutton et al., 1999). When option policies are stochastic, it is often necessary to approximate the option-value function  $Q_{\Omega}^{\pi}(s,o)$  in addition to the action-value function  $Q^{\pi}(s,a)$ . However, in our case, the option-value function for deterministic option policies is given by

$$
Q _ {\Omega} ^ {\pi} (\boldsymbol {s}, o) = Q ^ {\pi} \left(\boldsymbol {s}, \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s})\right), \tag {15}
$$

which we can estimate using the deterministic option policy  $\mu_{\pmb{\theta}}^{o}(s)$  and the approximated action-value function  $Q^{\pi}(\pmb {s},\pmb {a};\pmb {w})$  . In this work we employ the softmax gating policy of the form

$$
\pi (o | s) = \frac {\exp \left(Q ^ {\pi} \left(s , \boldsymbol {\mu} _ {\theta} ^ {o} (s) ; \boldsymbol {w}\right)\right)}{\sum_ {o \in \mathcal {O}} \exp \left(Q ^ {\pi} \left(s , \boldsymbol {\mu} _ {\theta} ^ {o} (s) ; \boldsymbol {w}\right)\right)}, \tag {16}
$$

which encodes the exploration in its form (Daniel et al., 2016). The state value function is given as

$$
V ^ {\pi} (\boldsymbol {s}) = \sum_ {o \in \mathcal {O}} \pi (o | \boldsymbol {s}) Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s}); \boldsymbol {w}), \tag {17}
$$

which can be computed using Equation (16). We use this state-value function when computing the advantage-weighted importance. In this study, the Q-function is trained in a manner proposed by Fujimoto et al. (2018). Two neural networks  $(Q_{\boldsymbol{w}_1}^\pi, Q_{\boldsymbol{w}_2}^\pi)$  are trained to estimate the Q-function, and the target value of the Q-function is computed as  $y_i = r_i + \min_{1,2} Q(\boldsymbol{s}_i, \boldsymbol{a}_i)$  for sample  $(s_i, \boldsymbol{a}_i, \boldsymbol{a}_i', r_i)$  in a batch sampled from a replay buffer, where  $r_i = r(\boldsymbol{s}_i, \boldsymbol{a}_i)$ . In this study, the gating policy determines the option once every  $N$  time steps, i.e.,  $t = 0, N, 2N, \ldots$

Deterministic Policy Gradient for Option Policies Neural networks that model  $\pmb{\mu}_{\theta}^{o}(\pmb{a}|\pmb{s})$  for  $o = 1,\dots,O$ , which we refer to as option-policy networks, are trained separately for each option. In the learning phase,  $p(o|s,a)$  is estimated by the option network. Then, samples are assigned to option  $o^{*} = \arg \max_{o}p(o|s,a)$  and are used to update the option-policy network that corresponds to  $o^{*}$ .

Algorithm 1 HRL via Advantage-Weighted Information Maximization (adInfoHRL)  
Input: Number of options  $O$ , size of on-policy buffer  
Initialize: Replay buffer  $\mathcal{D}_R$ , on-policy buffer  $\mathcal{D}_{\mathrm{on}}$ , network parameters  $\eta, \theta, w, \theta^{\mathrm{target}}, w^{\mathrm{target}}$  repeat  
for  $t = 0$  to  $t = T$  do  
Draw an option for a given  $s$ :  $o \sim \pi(o|s)$   
Draw an action  $a \sim \beta(a|s, o) = \mu_{\theta}^{o}(s) + \epsilon$   
Record a data sample  $(s, a, r, s')$   
Aggregate the data in  $\mathcal{D}_R$  and  $\mathcal{D}_{\mathrm{on}}$   
end for  
if the on-policy buffer is full then  
Update the option network by minimizing  $\mathcal{L}_{\mathrm{option}}(\eta)$  for samples in  $\mathcal{D}_{\mathrm{on}}$   
Clear the on-policy buffer  $\mathcal{D}_{\mathrm{on}}$   
end if  
Sample a batch  $\mathcal{D}_{\mathrm{batch}} \in \mathcal{D}_R$   
Update the Q network parameter  $w$   
if  $t$  mod  $d$  then  
Estimate  $p(o|s_i, a_i)$  for  $(s_i, a_i) \in \mathcal{D}_{\mathrm{batch}}$  using the option network  
Assign samples  $(s_i, a_i) \in \mathcal{D}_{\mathrm{batch}}$  to the option  $o^* = \arg \max p(o|s_i, a_i)$   
Update the option policy networks  $\mu_{\theta}^{o}(s)$  for  $o = 1, \dots, O$   
Update the target networks:  $w_{\mathrm{target}} \gets \tau w + (1 - \tau)w_{\mathrm{target}}, \theta_{\mathrm{target}} \gets \tau \theta + (1 - \tau)\theta_{\mathrm{target}}$   
end if  
until the convergence  
return  $\theta$

When performing a rollout,  $o$  is drawn by following the gating policy in Equation (16), and an action is generated by the selected option-policy network.

We learn option policies  $\pmb{\mu}_{\pmb{\theta}}^{o}(\pmb{s})$  for  $o = 1,\dots ,O$  to maximize the conditional expectation of the return given  $o$ , with respect to the state distribution induced by the behavior policy  $\beta (\pmb {a}|\pmb {s})$ :

$$
\mathbb {E} _ {\boldsymbol {s} \sim d ^ {\beta} (\boldsymbol {s})} [ Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) | o _ {t} = o ] = \int d ^ {\beta} (\boldsymbol {s}) Q ^ {\pi} \left(\boldsymbol {s}, \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s})\right) \mathrm {d} \boldsymbol {s}. \tag {18}
$$

The deterministic policy gradient of our option-policy  $\mu_{\theta}^{o}(s)$  is given by

$$
\nabla_ {\boldsymbol {\theta}} \mathbb {E} _ {\boldsymbol {s} \sim d ^ {\beta} (\boldsymbol {s})} [ Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) | o _ {t} = o ] = \mathbb {E} _ {\boldsymbol {s} \sim d ^ {\beta} (\boldsymbol {s})} \left[ \nabla_ {\boldsymbol {\theta}} \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s}) \nabla_ {\boldsymbol {a}} Q ^ {\pi} (\boldsymbol {s}, \boldsymbol {a}) | _ {\boldsymbol {a} = \boldsymbol {\mu} _ {\boldsymbol {\theta}} ^ {o} (\boldsymbol {s})} \right]. \tag {19}
$$

The procedure of adInfoHRL is summarized by Algorithm 1. As in TD3 (Fujimoto et al., 2018), we employed the soft update using a target value network and a target policy network.

# 4 EXPERIMENTS

We evaluated the proposed algorithm adInfoHRL on the OpenAI Gym platform (Brockman et al., 2016) with the MuJoCo Physics simulator (Todorov et al., 2012). We compared its performance with that of PPO implemented in OpenAI baselines (Dhariwal et al., 2017) and TD3. Henderson et al. (2018) have recently claimed that algorithm performance varies across environment, there is thus no clearly best method for all benchmark environments, and off-policy and on-policy methods have advantages in different problem domains. To analyze the performance of adInfoHRL, we compared it with state-of-the-art algorithms for both on-policy and off-policy methods, although we focused on the comparison with TD3, as our implementation of adInfoHRL is based on it. To determine the effect of learning the latent variable via information maximization, we used the same network architectures for the actor and critic in adInfoHRL and TD3. The gating policy updated variable  $o$  once every three time steps. We tested the performance of adInfoHRL with two and four options.

The activation of options over time and snapshots of the learned option policies on the Walker2d task are shown in Figure 2. One can see that the option policies are activated in different phases of locomotion. While the option indicated by yellow in Figure 2 corresponds to the phase for

![](images/8697ecbafea85dd320dabb90d190250d056fb4afe1f82a933b6b8db756d0ed0e.jpg)  
Figure 2: Activation of the four options over time steps on the Walker2d task.

![](images/ab056d7a3c0551dfa25ba31bc24e243c187d65e53642e3e98d01371e19ab1163.jpg)

![](images/40410f6ede54bca7af985f76ec461d255eeb62a04fa9028cb05d45305b87c317.jpg)  
(b) Results on Ant.

![](images/54d7adb80565f8aadc1d09092947c62f6ffd8803baa5b577c2abc73e2fd66a76.jpg)  
(c) Results on HalfCheetah.

![](images/0c2694274b89001c5dacf3fe0291ebac6d73e2b5bd5c5f72387c29363d5751ba.jpg)  
(a) Results on Walker2d.  
(d) Results on Hopper.  
Figure 3: Performance of adInfoHRL. (a)-(d) show comparison with baseline methods. (e) and (f) show the output of the option network and the activation of options on Walker2d, respectively.

![](images/f9f3485219a76cb5783976c81117ff4b1fd8ceeab5c3572eedc8e978f89a309f.jpg)  
(e) Output of the option network in (f) Activation of options in the state the state-action space on Walker2d. space on Walker2d.

![](images/a13e813c73003d46df7d0d1c7f4ddbe10de13b238fdbe444eed9f84bae47f44c.jpg)

kicking the floor, the option indicated by blue corresponds to the phase when the agent was on the fly. Visualization of the options learned on the HalfCheetah and Ant tasks are shown in Appendix D.

The averaged return of five trials is reported in Figure 3(a)-(d). AdInfoHRL yields the best performance on  $\mathrm{Ant}^{1}$  and Walker2d, whereas the performance of TD3 and adInfoHRL was comparable on HalfCheetah and Hopper, and PPO outperformed the other methods on Hopper. Henderson et al. (2018) claimed that on-policy methods show their superiority on tasks with unstable dynamics, and our experimental results are in line with such previous studies. The results show that the optimal number of options is dependent on the task: two options for HalfCheetah and Ant, and four options for Walker2d. AdInfoHRL exhibited the sample efficiency on Ant and Walker2d in the sense that it required fewer samples than TD3 to achieve comparable performance on those tasks. The concept underlying adInfoHRL is to divide the state-action space to deal with the multi-modal advantage function and learn option policies corresponding to separate modes of the advantage function. Therefore, adInfoHRL shows its superiority on tasks with the multi-modal advantage function and not on tasks with a simple advantage function. Thus, it is natural that the benefit of adInfoHRL is dependent on the characteristics of the task.

The outputs of the option network and the activation of options on Walker2d are shown in Figure 3(e)-(f). For visualization, the dimensionality was reduced using t-SNE (van der Maaten & Hinton, 2008). The state-action space is clearly divided into separate domains in Figure 3(e). As shown in Figure 3(f), the options are activated in different domains of the state space, which indicates that diverse options are learned by adInfoHRL.

# 5 RELATED WORK AND DISCUSSION

Past studies have proposed several ways to deal with the latent variable in HRL. The recent work by Smith et al. (2018) proposed inferred option policy gradients (IOPG), which is derived as an extension of policy gradient to the option framework. Nachum et al. (2018) recently proposed off-policy target correction for HRL on goal-oriented tasks, where a higher-level policy instructs a lower-level policy by generating the goal signal instead of an inferred latent variable. A popular approach for learning the latent variable in HRL is the variational approach. The recent work by Haarnoja et al. (2018) is based on soft Actor Critic, and the latent variable is inferred using the variational approach. The work by Hausman et al. (2018) is also closely related to the variational approach, and they proposed a method for learning a latent variable of a hierarchical policy via a variational bound. On the contrary, our method learns the latent variable by maximizing mutual information with advantage-weighted importance. The work by Florenssa et al. (2017) also considered the mutual information in their formulation. They employed mutual information between a subset of the state and the latent variable as a bonus reward to prevent option policies from collapsing into a single mode. Our approach is different from that previous study in the sense that we employ mutual information between the latent variable and the state-action pairs, which leads to the division of the state-action space instead of considering only the state space. We think that dividing the state-action space is an efficient approach when the advantage function is multi-modal, as depicted in Figure 1.

Our method can be interpreted as an approach that divides the state-action space based on the mutual information criterion. This concept is related to that of Divide and Conquer (DnC) proposed by Ghosh et al. (2018), although DnC clusters the initial states and does not consider switching between option policies during the execution of a single trajectory.

InfoGAIL proposed by Li et al. (2017) learns the interpretable representation of the state-action space the mutual information maximization. InfoGAIL can be interpreted as a method that divides the state-action space based on the density induced by an expert's policy by maximizing the regularized mutual information objective. In this sense, it is closely related to our method, although the problem setting of interest is different.

A recent study by Rajeswaran et al. (2017) indicates that a relatively simple policy architecture can achieve high performance in continuous control tasks. The results of this study do not conflict with that indication. The main conclusion from this work is that learning the latent discrete representation of the state-action space can be beneficial for learning, and we think that it can be achieved with simple policy models.

In this study we developed adInfoHRL based on deterministic option policies. However, the concept of dividing the state-action space via advantage-weighted importance can be applied to stochastic policy gradients as well. Further investigation in this direction is necessary in future work.

# 6 CONCLUSIONS

We proposed a novel HRL method, hierarchical reinforcement learning via advantage-weighted information maximization. In our framework, the latent variable of a hierarchical policy is learned as a discrete latent representation of the state-action space in an unsupervised way. Our HRL framework is derived by considering deterministic option policies and by leveraging the analogy between the gating policy for HRL and a monolithic policy for the standard RL. The results of the experiments indicate that adInfoHRL can learn a diversity of options on continuous control tasks. Our results also suggested that our approach can improve the performance of TD3 in certain problem domains.

# REFERENCES

R. Akrour, A. Abdolmaleki, H. Abdulsamad, and G. Neumann. Model-free trajectory optimization for reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), 2016.  
P. L. Bacon, J. Harb, and D. Precup. The option-critic architecture. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2017.  
G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. Openai gym. In arXiv, 2016.  
X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel. InfoGAN: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in Neural Information Processing Systems (NIPS), pp. 2172-2180, 2016.  
C. Daniel, G. Neumann, O. Kroemer, and J. Peters. Hierarchical relative entropy policy search. Journal of Machine Learning Research, 17:1-50, 2016.  
M. P. Deisenroth, G. Neumann, and J. Peters. A survey on policy search for robotics. Foundations and Trends in Robotics, pp. 388-403, 2013.  
P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu. Openai baselines. https://github.com/openai/baselines, 2017.  
T. G Dietterich. Hierarchical reinforcement learning with the MAXQ value function decomposition. Journal of Artificial Intelligence Research, 13:227-303, 2000.  
Y. Duan, X. Chen, R. Houthooft, J. Schulman, and P. Abbeel. Benchmarking deep reinforcement learning for continuous control. In Proceedings of the International Conference on Machine Learning (ICML), 2016.  
C. Florensa, Y. Duan, and P. Abbeel. Stochastic neural networks for hierarchical reinforcement learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2017.  
K. Frans, J. Ho, X. Chen, P. Abbeel, and J. Schulman. Meta learning shared hierarchies. In Proceedings of the International Conference on Learning Representations (ICLR), 2018.  
S. Fujimoto, H. van Hoof, and D. Meger. Addressing function approximation error in actor-critic methods. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1587-1596, Stockholmssan, Stockholm Sweden, 10-15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/fujimoto18a.html.  
D. Ghosh, A. Singh, A. Rajeswaran, V. Kumar, and S. Levine. Divide-and-conquer reinforcement learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2018.  
R. Gomes, A. Krause, and P. Perona. Discriminative clustering by regularized information maximization. In Advances in Neural Information Processing Systems (NIPS), 2010.  
I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning. MIT Press, 2016. http://www.deeplearningbook.org.  
T. Haarnoja, K. Hartikainen, P. Abbeel, and S. Levine. Latent space policies for hierarchical reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), 2018.  
J. Harb and D. Precup P. Bacon, M. Klissarov. When waiting is not an option: Learning options with a deliberation cost. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018.

K. Hausman, J. T. Springenberg, Z. Wang, N. Heess, and M. Riedmiller. Learning an embedding space for transferable robot skills. In Proceedings of the International Conference on Learning Representations (ICLR), 2018.  
P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018.  
W. Hu, T. Miyato, S. Tokui, E. Matsumoto, and M. Sugiyama. Learning discrete representations via information maximizing self augmented training. In Proceedings of the International Conference on Machine Learning (ICML). 2017.  
S. Levine, C. Finn, T. Darrell, and P. Abbeel. End-to-end training of deep visuomotor policies. Journal of Machine Learning Research, 17(39):1-40, 2016. URL http://jmlr.org/papers/v17/15-522.html.  
Y. Li, J. Song, and S. Ermon. InfoGAIL: Interpretable imitation learning from visual demonstrations. In Advances in Neural Information Processing Systems (NIPS), pp. 3812-3822, 2017.  
T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2016.  
V. Mnih, K. Kavukcuoglu1, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Ried-miller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518:529-533, 2015.  
O. Nachum, S. Gu, H. Lee, and S. Levine. Data-efficient hierarchical reinforcement learning. arXiv, 2018. URL https://arxiv.org/abs/1805.08296.  
T. Osa and M. Sugiyama. Hierarchical policy search via return-weighted density estimation. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2018.  
T. Osa, J. Peters, and G. Neumann. Experiments with hierarchical reinforcement learning of multiple grasping policies. In Proceedings of the International Symposium on Experimental Robotics (ISER), 2016.  
A. Rajeswaran, K. Lowrey, E. Todorov, and S. Kakade. Towards generalization and simplicity in continuous control. In Advances in Neural Information Processing Systems (NIPS), 2017.  
J. Schulman, X. Chen, and P. Abbeel. Equivalence between policy gradients and soft q-learning. In arXiv, 2017a.  
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv, 2017b.  
D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller. Deterministic policy gradient algorithms. In Proceedings of the International Conference on Machine Learning (ICML), 2014.  
D. Silver, A. Huang, C. J Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484-489, 2016.  
M. J. A. Smith, H. Van Hoof, and J. Pineau. An inference-based policy gradient method for learning options. In Proceedings of the International Conference on Machine Learning (ICML), 2018.  
R. Sutton, D. Precup, and S. Singh. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial intelligence, 1999.  
E. Todorov, T. Erez, , and Yuval Tassa. Mujoco: A physics engine for model-based control. In Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012.

L. van der Maaten and G. Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 2008.  
A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu. FeUdal networks for hierarchical reinforcement learning. In Proceedings of the International Conference on Machine Learning (ICML), 2017.  
B. Ziebart. Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy. PhD thesis, Carnegie Mellon University, 2010.
