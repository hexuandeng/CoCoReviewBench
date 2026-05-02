# CORRECTING DATA DISTRIBUTION MISMATCH IN OFFLINE META-REINFORCEMENT LEARNING WITH FEWSHOT ONLINE ADAPTATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Offline meta-reinforcement learning (offline meta-RL) extracts knowledge from a given dataset of multiple tasks and achieves fast adaptation to new tasks. Recent offline meta-RL methods typically use task-dependent behavior policies (e.g., training RL agents on each individual task) to collect a multi-task dataset and learn an offline meta-policy. However, these methods always require extra information for fast adaptation, such as offline context for testing tasks or oracle reward functions. Offline meta-RL with few-shot online adaptation remains an open problem. In this paper, we first formally characterize a unique challenge under this setting: data distribution mismatch between offline training and online adaptation. This distribution mismatch may lead to unreliable offline policy evaluation and the regular adaptation methods of online meta-RL will suffer. To address this challenge, we introduce a novel mechanism of data distribution correction, which ensures the consistency between offline and online evaluation by filtering out out-of-distribution episodes in online adaptation. As few-shot out-of-distribution episodes usually have lower returns, we propose a Greedy Context-based data distribution Correction approach, called GCC, which greedily infers how to solve new tasks. GCC diversely samples "task hypotheses" from the current posterior belief and selects a greedy hypothesis with the highest return to update the task belief. Our method is the first to provide an effective online adaptation without additional information, and can be combined with off-the-shelf context-based offline meta-training algorithms. Empirical experiments show that GCC achieves state-of-the-art performance on the Meta-World ML1 benchmark compared to baselines with/without offline adaptation.

# 1 INTRODUCTION

Human intelligence is capable of learning a wide variety of skills from past history and can adapt to new environments by transferring skills with limited experience. Current reinforcement learning (RL) has surpassed human-level performance (Mnih et al., 2015; Silver et al., 2017; Hafner et al., 2019) but requires a vast amount of experience. However, in many real-world applications, RL encounters two major challenges: multi-task efficiency and costly online interactions. In multi-task settings, such as robotic manipulation or locomotion (Yu et al., 2020b), agents are expected to solve new tasks with few-shot adaptation with previously learned knowledge. Moreover, collecting sufficient exploratory interactions is usually expensive or dangerous in robotics (Rafailov et al., 2021), autonomous driving (Yu et al., 2018), and healthcare (Gottesman et al., 2019). One popular paradigm for breaking this practical barrier is offline meta reinforcement learning (offline meta-RL; Li et al., 2020; Mitchell et al., 2021), which trains a meta-RL agent with pre-collected offline multi-task datasets and enables fast policy adaptation to unseen tasks.

Recent offline meta-RL methods have been proposed to utilize a multi-task dataset collected by task-dependent behavior policies (Li et al., 2020; Mitchell et al., 2021; Dorfman et al., 2021). They show promise by solving new tasks with few-shot adaptation. However, existing offline meta-RL approaches require additional information or assumptions for fast adaptation. For example, FOCAL (Li et al., 2020) and MACAW (Mitchell et al., 2021) conduct offline adaptation and offline contexts are available for unseen tasks. BORel (Dorfman et al., 2021) and SMAC (Pong et al., 2022) employ few-shot online adaptation, but the former assumes known reward functions, and the latter requires free interactions with the environment without reward supervision. Therefore, achieving effective few-shot online adaptation remains an open problem for offline meta-RL.

![](images/66bef4cbed84e57cc7acf8a92261bf264233450fbbd0ad382f7b71bfec041a41.jpg)  
Figure 1: Illustration of data distribution mismatch between offline training and online adaptation.

One particular challenge of meta-RL compared to meta-supervised learning is the need to learn how to explore in the testing environments (Finn & Levine, 2019). In offline meta-RL, the gap of reward and transition distribution between offline training and online adaptation presents a unique conundrum for meta-policy learning, namely data distribution mismatch. As illustrated in Figure 1, when we collect an offline dataset using the expert policy for each task, the robot will be meta-trained on all successful trajectories  $(\rightarrow)$ . The robot will fast explore the environment to reduce task uncertainty in meta-testing. However, when it tries the middle path and hits a stone, this failed adaptation trajectory  $(\rightarrow)$  does not match the training data distribution, which can lead to false task inference or adaptation. To formally characterize this phenomenon, we utilize the perspective of Bayesian RL (BRL; Duff, 2002; Zintgraf et al., 2019; Dorfman et al., 2021) that maintains a task belief given the context history and learns a meta-policy on belief states. Our theoretical analysis shows that task-dependent data collection results in an inconsistency between offline meta-policy evaluation and online adaptation evaluation, which contrasts with policy evaluation consistency in offline single-task RL (Fujimoto et al., 2019). To deal with this inconsistency, we can choose either to trust the offline dataset or to trust new experience and continue online exploration. The latter may not be able to collect sufficient data in few-shot adaptation, and thus not feasible for meta-RL. Therefore, we adopt the former strategy and introduce a mechanism of online data distribution correction that filters out out-of-distribution episodes and provides a theoretical consistency guarantee in policy evaluations.

To realize our theoretical implications in practical settings, we propose a context-based offline meta-RL algorithm with a novel online adaptation mechanism, called Greedy Context-based data distribution Correction (GCC). To align adaptation context with the meta-training distribution, GCC utilizes greedy task inference, which diversely samples "task hypotheses" and selects a hypothesis with the highest return to update the belief. For example in Figure 1, the robot will try other paths and the failed adaptation trajectory (middle) will not be used for task inference because it is out-of-distribution with a lower return. To our best knowledge, our method is the first to design a delicate context mechanism to achieve effective online adaptation for offline meta-RL and has the advantage of directly combining with off-the-shelf context-based offline meta-training algorithms.

Our main contribution is to formalize a specific challenge (i.e., data distribution mismatch) in offline meta-RL with online adaptation and propose a greedy context mechanism with theoretical motivation. We extensively evaluate the performance of GCC in didactic problems proposed by prior work (Rakelly et al., 2019; Zhang et al., 2021) and Meta-World ML1 benchmark with 50 tasks (Yu et al., 2020b). In these didactic problems, GCC demonstrates that our context mechanism can accurately infer task identification, whereas the original online adaptation methods will suffer due to out-of-distribution data. Empirical results on the more challenging Meta-World ML1 benchmark show that GCC significantly outperforms baselines with few-shot online adaptation, and achieves better or comparable performance than offline adaptation baselines with expert context.

# 2 NOTATIONS AND PRELIMINARIES

# 2.1 STANDARD META-RL

The goal of meta-RL (Finn et al., 2017; Rakelly et al., 2019) is to train a meta-policy that can quickly adapt to new tasks using  $N$  adaptation episodes. The standard meta-rl setting deals with a distribution  $p(\kappa)$  over MDPs, in which each task  $\kappa_{i}$  sampled from  $p(\kappa)$  presents a finite-horizon MDP (Zintgraf et al., 2019; Yin & Wang, 2021).  $\kappa_{i}$  is defined by a tuple  $(S, \mathcal{A}, \mathcal{R}, H, P^{\kappa_{i}}, R^{\kappa_{i}})$ , including state space  $S$ , action space  $\mathcal{A}$ , reward space  $\mathcal{R}$ , planning horizon  $H$ , transition function  $P^{\kappa_{i}}(s'|s,a)$ , and reward function  $R^{\kappa_{i}}(r|s,a)$ . Denote  $\mathcal{K}$  is the space of task  $\kappa_{i}$ . In this paper, we assume dynamics function  $P$  and reward function  $R$  may vary across tasks and share a common structure. The meta-RL

algorithms repeatedly sample batches of tasks to train a meta-policy. In the meta-testing, agents aim to rapidly adapt a good policy for new tasks drawn from  $p(\kappa)$ .

From a perspective of Bayesian RL (BRL; Ghavamzadeh et al., 2015), recent meta-RL methods (Zintgraf et al., 2019) utilize a Bayes-adaptive Markov decision process (BAMDP; Duff, 2002) to formalize standard meta-RL. A BAMDP is a belief MDP (Kaelbling et al., 1998) of a special Partially Observable MDP (POMDP; Astrom, 1965) whose unobserved state information presents unknown task identification in  $N$  adaptation episodes. BAMDPs are defined as a tuple  $M^{+} = \left(S^{+},\mathcal{A},\mathcal{R},H^{+},P_{0}^{+},P^{+},R^{+}\right)$  (Zintgraf et al., 2019), in which  $S^{+} = S\times \mathcal{B}$  is the hyper-state space,  $\mathcal{B}$  is the space of task beliefs over meta-RL MDPs,  $\mathcal{A}$  is the action space,  $\mathcal{R}$  is the reward space,  $H^{+} = N\times H$  is the planning horizon across adaptation episodes,  $P_0^+ (s_0^+)$  is the initial hyperstate distribution representing task distribution  $p(\kappa)$ ,  $P^{+}\left(s_{t + 1}^{+}|s_{t}^{+},a_{t},r_{t}\right)$  is the transition function, and  $R^{+}\left(r_{t}|s_{t}^{+},a_{t}\right)$  is the reward function. A meta-policy  $\pi^{+}\left(a_{t}|s_{t}^{+}\right)$  on BAMDPs prescribes a distribution over actions for each hyper-state  $s_t^+ = (s_t,b_t)$ . The objective of meta-RL agents is to find a meta-policy  $\pi^{+}$  that maximizes expected return, i.e., online policy evaluation denoted by  $\mathcal{J}_{M^+}(\pi^+)$ . Formal descriptions are deferred to Appendix A.1.3.

# 2.2 OFFLINE META-RL

In the offline meta-RL setting, a meta-learner only has access to an offline multi-task dataset  $\mathcal{D}^+$  and is not allowed to interact with the environment during meta-training (Li et al., 2020). Recent offline meta-RL methods (Dorfman et al., 2021) always utilize task-dependent behavior policies  $p(\mu|\kappa)$ , which represents the random variable of the behavior policy  $\mu(a|s)$  conditioned on the random variable of the task  $\kappa$ . For brevity, we overload  $[\mu] = p(\mu|\kappa)$ . Similar to related work on offline RL (Yin & Wang, 2021), we assume that  $\mathcal{D}^+$  consists of multiple i.i.d. trajectories that are collected by executing task-dependent policies  $[\mu]$  in  $M^+$ . We Define the reward and transition distribution of the task-dependent data collection by  $\mathbb{P}_{M^+,[\mu]}$  (Jin et al., 2021), i.e., for each step  $t$  in a trajectory,

$$
\mathbb {P} _ {M ^ {+}, [ \mu ]} \left(r _ {t}, s _ {t + 1} \mid s _ {t} ^ {+}, a _ {t}\right) \propto \mathbb {E} _ {\kappa_ {i} \sim p (\kappa), \mu_ {i} \sim p (\mu | \kappa_ {i})} \left[ \mathbb {P} _ {\kappa_ {i}} \left(r _ {t}, s _ {t + 1} \mid s _ {t}, a _ {t}\right) \cdot p _ {M ^ {+}} \left(s _ {t} ^ {+} \mid \kappa_ {i}, \mu_ {i}\right) \right], \tag {1}
$$

where the reward and transition distribution of data collection with  $\mu_{i}$  in a task  $\kappa_{i}$  is defined as

$$
\mathbb {P} _ {\kappa_ {i}} \left(r _ {t}, s _ {t + 1} \mid s _ {t}, a _ {t}\right) = R ^ {\kappa_ {i}} \left(r _ {t} \mid s _ {t}, a _ {t}\right) \cdot P ^ {\kappa_ {i}} \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right), \tag {2}
$$

and  $p_{M^{+}}(s_{t}^{+}|\kappa_{i},\mu_{i})$  denotes the probability of  $s_t^+$  when executing  $\mu_{i}$  in a task  $\kappa_{i}$ . Note that the offline dataset  $\mathcal{D}^{+}$  can be narrow and a large amount of state-action pairs are not covered. These unseen state-action pairs will be erroneously estimated to have unrealistic values, called a phenomenon of extrapolation error (Fujimoto et al., 2019). To overcome extrapolation error in offline RL, related works (Fujimoto et al., 2019) introduce batch-constrained RL, which restricts the action space in order to force policy selection of an agent with respect to a given dataset. A policy  $\pi^{+}$  is defined to be batch-constrained by  $\mathcal{D}^{+}$  if  $\pi^{+}(a|s^{+}) = 0$  for all  $(s^{+},a)$  tuples that are not contained in  $\mathcal{D}^{+}$ . Offline RL (Liu et al., 2020; Chen & Jiang, 2019) approximates policy evaluation for a batch-constrained policy  $\pi^{+}$  by sampling from an offline dataset  $\mathcal{D}^{+}$ , which is denoted by  $\mathcal{I}_{\mathcal{D}^{+}}(\pi^{+})$  and called Approximate Dynamic Programming (ADP; Bertsekas & Tsitsiklis, 1995). During meta-testing, RL agents perform online adaptation using a meta-policy  $\pi^{+}$  in new tasks drawn from meta-RL task distribution. The reward and transition distribution of data collection with  $\pi^{+}$  in  $M^{+}$  during adaptation is defined by

$$
\mathbb {P} _ {M ^ {+}} \left(r _ {t}, s _ {t + 1} \mid s _ {t} ^ {+}, a _ {t}\right) = R ^ {+} \left(r _ {t} \mid s _ {t} ^ {+}, a _ {t}\right) \cdot P ^ {+} \left(s _ {t + 1} \mid s _ {t} ^ {+}, a _ {t}\right), \tag {3}
$$

where  $R^{+}, P^{+}$  are defined in  $M^{+}$ . Detailed formulations are deferred to Appendix A.1.4.

# 3 THEORY: DATA DISTRIBUTION MISMATCH CORRECTION

Consistency of training and testing conditions is an important principle for machine learning (Vinyals et al., 2016; Finn & Levine, 2019). Recently, offline meta-RL with task-dependent behavior policies (Li et al., 2020; Mitchell et al., 2021) faces a special challenge: the reward and transition distribution in offline training and online adaptation may not match. We first build a theory about data distribution mismatch to understand this phenomenon. Our theoretical analysis is based on Bayesian RL (BRL; Zintgraf et al., 2019) and demonstrates that data distribution mismatch may lead to a large gap between online and offline policy evaluation, a problem which does not appear in offline single-task RL (Fujimoto et al., 2019) settings. To address this challenge, we propose a new mechanism to correct data distribution, which transforms the BAMDPs (Duff, 2002) to enrich information about task-dependent behavior policies into overall beliefs. This transformed BAMDPs provide reliable policy evaluation by requiring the agent to filter out out-of-distribution episodes.

# 3.1 DATA DISTRIBUTION MISMATCH IN OFFLINE META-RL

We define data distribution mismatch in offline meta-RL as follows.

Definition 1 (Data Distribution Mismatch). In a BAMDP  $M^{+}$ , for each task-dependent behavior policies  $[\mu]$  and batch-constrained meta-policy  $\pi^{+}$ , the data distribution mismatch between  $\pi^{+}$  and  $[\mu]$  is defined by that  $\exists s_{t}^{+}, a_{t}, s.t.$ ,

$$
p _ {M ^ {+}} ^ {\pi^ {+}} \left(s _ {t} ^ {+}, a _ {t}\right) > 0 \quad a n d \quad \mathbb {P} _ {M ^ {+}} \left(r _ {t}, s _ {t + 1} \mid s _ {t} ^ {+}, a _ {t}\right) \neq \mathbb {P} _ {M ^ {+}, [ \mu ]} \left(r _ {t}, s _ {t + 1} \mid s _ {t} ^ {+}, a _ {t}\right), \tag {4}
$$

where  $p_{M^+}^{\pi^+}(s_t^+, a_t)$  is the probability of reaching a tuple  $(s_t^+, a_t)$  while executing  $\pi^+$  in  $M^+$  (formal definition deferred to Appendix A.2), and  $\mathbb{P}_{M^+}, \mathbb{P}_{M^+,[\mu]}$  are the reward and transition distribution of data collection defined in Eq. (1) and (3), respectively.

The data distribution induced by  $\pi^{+}$  and  $[\mu]$  mismatches when the reward and transition distribution of  $\pi^{+}$  and  $[\mu]$  differs in a tuple  $(s_{t}^{+},a_{t})$ , in which the agent can reach this tuple by executing  $\pi^{+}$  in  $M^{+}$ , i.e.,  $p_{M^{+}}^{\pi^{+}}(s_{t}^{+},a_{t}) > 0$ . Note that if  $\pi^{+}$  can reach a tuple  $(s_{t}^{+},a_{t})$ , this tuple is guaranteed to be contained into the offline dataset, i.e.,  $p_{M^{+}}^{\left[\mu\right]}(s_{t}^{+},a_{t}) > 0$ , because a batch-constrained policy  $\pi^{+}$  will not select an action outside of the dataset collected by  $[\mu]$ , as introduced in Section 2.2.

Theorem 1. There exists task-dependent behavior policies  $[\mu]$  and a batch-constrained meta-policy  $\pi^{+}$  in a BAMDP  $M^{+}$  such that the data distribution induced by  $\pi^{+}$  and  $[\mu]$  does not match.

To serve a concrete example, we construct an offline meta-RL setting shown in Figure 2. In this example, there are  $v$  meta-RL tasks  $\mathcal{K} = \{\kappa_1,\dots ,\kappa_v\}$  and  $v$  behavior policies  $\{\mu_1,\ldots ,\mu_v\}$ , where  $v\geq 2$ . Each task  $\kappa_{i}$  has one state  $s_0$ ,  $2v$  actions, and horizon  $H = 1$ . For each task  $\kappa_{i}$ , RL agents can receive reward 1 performing action  $a_{i}$ . A common structure shared in  $\mathcal{K}$  is that the reward function for actions  $\{a_{v + 1},\dots ,a_{2v}\}$  is 0, and the RL agent will not perform these actions to maximize the expected future return. During adaptation, the RL agent can interact with the environment within  $v$  episodes. The task distribution is uni

form, the behavior policy of task  $\kappa_{i}$  is  $\mu_{i}$ , and each behavior policy  $\mu_{i}$  will perform  $a_{i}$ . When a batch-constrained meta-policy  $\pi^{+}$  selects an action  $\tilde{a}$  in the initial state  $s_0^+$ , we find that

![](images/7e86740a350f8528108926a90b9c63c5bfd92f8c8c458e75f5c6099db2eec41d.jpg)  
Figure 2: A concrete example, which has  $v$  meta-RL tasks, one state,  $2v$  actions,  $v$  behavior polices, horizon  $H = 1$  in a episode, and  $v$  adaptation episodes, where  $v \geq 2$ .

$$
p (\kappa_ {i}) = \frac {1}{v}, p (\mu_ {i} | \kappa_ {i}) = 1, \text {a n d} \mu_ {i} (a _ {i} | s _ {0}) = 1
$$

$$
\mathbb {P} _ {M ^ {+}} \left(r = 1 \mid s _ {0} ^ {+}, \tilde {a}\right) = \frac {1}{v} \neq \mathbb {P} _ {M ^ {+}, [ \mu ]} \left(r = 1 \mid s _ {0} ^ {+}, \tilde {a}\right) = 1, \tag {5}
$$

in which there is the probability of  $\frac{1}{v}$  to sample a corresponding testing task, whose reward function of  $\tilde{a}$  is 1, whereas the reward in the offline dataset collected by  $[\mu]$  is all 1.

Proposition 1. There exists a BAMDP  $M^{+}$  and a batch-constrained meta-policy  $\pi^{+}$  such that, the gap of policy evaluation of  $\pi^{+}$  between offline meta-training and online adaptation is at least  $\frac{H^{+}}{2}$ , where  $H^{+}$  is the planning horizon of  $M^{+}$  defined in Section 2.1.

In the example shown in Figure 2, an offline multi-task dataset  $D^{+}$  is drawn from the task-dependent data collection  $\mathbb{P}_{M^{+},[\mu]}$ . Since the reward of  $D^{+}$  is all 1, for each batch-constrained meta-policy  $\pi^{+}$ , the offline evaluation of  $\pi^{+}$  in  $D^{+}$  is  $\mathcal{J}_{D^{+}}(\pi^{+}) = H^{+} = vH$ , as defined in Section 2.2. The optimal meta-policy  $\pi^{+,*}$  in this example is to enumerate  $a_{1},\ldots ,a_{v}$  until the task identification is inferred from the action with a reward of 1. A meta-policy  $\pi^{+,*}$  needs to explore in the testing environments and its online policy evaluation is  $\mathcal{J}_{M^{+}}(\pi^{+,*}) = \frac{H^{+}}{2}$ . The detailed proof is deferred to Appendix A.2. Thus, the evaluation gap between offline meta-training and online adaptation is

$$
\left| \mathcal {J} _ {M ^ {+}} (\pi^ {+}) - \mathcal {J} _ {D ^ {+}} (\pi^ {+}) \right| \geq \mathcal {J} _ {D ^ {+}} (\pi^ {+}) - \mathcal {J} _ {M ^ {+}} (\pi^ {+, *}) = \frac {H ^ {+}}{2}. \tag {6}
$$

Proposition 1 suggests that offline policy evaluation may fail and it is imperative to build a connection between online and offline evaluation for effective meta-policy training.

# 3.2 DATA DISTRIBUTION CORRECTION MECHANISM

To correct data distribution mismatch, we define the transformed BAMDPs as follows.

Definition 2 (Transformed BAMDPs, informal). A transformed BAMDP is defined by a new BAMDP  $\overline{M}^{+}$ , whose belief  $\bar{b}_{t}$  is about meta-RL MDPs  $\kappa$  and task-dependent behavior policies  $[\mu]$ . The goal of meta-RL agents is to find a meta-policy  $\bar{\pi}^{+}\left(a_{t}\big|\overline{s}_{t}^{+}\right)$  that maximizes online policy evaluation  $\mathcal{I}_{\overline{M}^{+}}(\bar{\pi}^{+})$ , where  $\bar{s}_t^+ = (s_t,\bar{b}_t)$  is a hyper-state of  $\overline{M}^{+}$ .

Transformed BAMDPs maintain an overall belief about the task and behavior policies given the current context history. Compared to the original BAMDPs stated in Section 2.2, the transformed BAMDPs incorporate additional information about offline data collection into beliefs. Meanwhile, this transformation introduces an extra condition to the online adaptation process, as indicated by the following fact.

Fact 1. Transformed BAMDPs implicitly require the agent to filter out out-of-distribution episodes.

During online adaptation, RL agents construct a hyper-state  $\bar{s}_t^+ = (s_t,\bar{b}_t)$  from the context history and perform a meta-policy  $\bar{\pi}^{+}\left(a_{t}\big|\bar{s}_{t}^{+}\right)$ . The new belief  $\bar{b}_t$  accounts for the uncertainty of task MDPs and behavior policies. However, the context history may be conflict with the uncertainty estimation of behavior policies, i.e., RL agents cannot update their beliefs  $\bar{b}_t$  when they has observed an event that they believe to have probability zero. For example in Figure 2, RL agents can take an action  $a_1$  and receive a reward of 0 during online adaptation. After obsevering an action  $a_1$ , the posterior belief will be  $p\left(\kappa_1,\mu_1\big|\bar{b}_t\right) = 1$ , in which it is contradictory because the reward function of  $a_1$  is 1 in the task  $\kappa_{1}$ . To support feasible Bayesian belief updating, transformed BAMDPs require the RL agents to filter out out-of-distribution episodes and we can derive the following theorem.

Theorem 2. In a transformed BAMDP  $\overline{M}^{+}$ , for each task-dependent behavior policies  $[\mu]$  and batch-constrained meta-policy  $\bar{\pi}^{+}$ , the data distribution induced by  $\bar{\pi}^{+}$  and  $[\mu]$  matches after filtering out out-of-distribution episodes in online adaptation. Besides, the policy evaluation of  $\bar{\pi}^{+}$  in offline meta-training and online adaptation will be asymptotically consistent, as the offline dataset grows.

In Theorem 2, the proof of consistent policy evaluation is similar to that in offline single-task RL (Yin & Wang, 2021), whose proof is deferred to Appendix A.2. This theorem indicates that we can design a delicate context mechanism to correct data distribution and guarantee the final performance of online adaptation by maximizing the expected future return during offline meta-training.

# 4 GCC: GreEDy CONTEXT-BASED DATA DISTRIBUTION CORRECTION

In this section, we investigate a scheme to address data distribution mismatch in offline meta-RL with few-shot online adaptation. Inspired by the theoretical results discussed in Section 3, we aim to distinguish whether an adaptation episode is in the distribution of the given meta-training dataset. RL agents can utilize in-distribution context to infer how to solve meta-testing tasks. However, in practice, accurate out-of-distribution quantification is challenging (Yu et al., 2021; Wang et al., 2021). To address this challenge, we introduce a novel Greedy Context-based data distribution Correction (GCC), which elaborates a greedy online adaptation mechanism and can directly combine with off-the-shelf offline meta-RL algorithms. GCC consists of two main components: (i) an off-the-shelf context-based offline meta-training method which extracts meta-knowledge from a given multi-task dataset, and (ii) a greedy context-based online adaptation that realizes a selective context mechanism to infer how to solve meta-testing tasks. The whole algorithm is illustrated in Algorithm 1.

# 4.1 CONTEXT-BASED OFFLINE META-TRAINING

To support effective offline meta-training, we employ a state-of-the-art off-the-shelf context-based algorithm, i.e., FOCAL (Li et al., 2020), which follows the algorithmic framework of a popular meta-RL approach, PEARL (Rakelly et al., 2019). In this meta-training paradigm, task identification  $\kappa$  is modeled by a latent task variable  $z\in \mathcal{Z} = \mathbb{R}^d$ , where  $d$  is the dimension of the latent space. GCC meta-trains a context encoder  $q(z|c)$ , a policy  $\pi (a|s,z)$ , and a value function  $Q(s,a,z)$ , where  $c$  is the context information including states, actions, rewards, and next states. The encoder  $q$  infers a task belief about the latent task variable  $z$  on the space  $\mathcal{Z}$  based on the received context. We use  $q(z)$  to denote the prior distribution for  $c = \emptyset$ . The policy  $\pi$  and value function  $Q$  are conditioned

on the latent task variable  $z$ , in which the representation of  $z$  can be end-to-end trained on the RL losses of  $\pi$  or  $Q$ . In addition to the gradient from  $\pi$  or  $Q$ , recent offline meta-RL (Li et al., 2020) also uses a contrastive loss to help the representation of  $z$  distinguish different tasks. We argue that meta-training the inference network  $q$  within a given dataset implicitly incorporates the information about offline data collection into the belief, as discussed in Definition 3. Formally, in the offline setting, the prior distribution  $q(z)$  can approximately present a distribution of the latent task variable  $z$  over each meta-training task, i.e., a sample  $\tilde{z}$  drawn from  $q(z)$  is equivalent to sampling a task during meta-training and inferring  $\tilde{z}$  using the offline context from the given offline dataset. During meta-testing, few-shot out-of-distribution online adaptation episodes may confuse the context encoder  $q(z|c)$  and RL agents need to use in-distribution context to provide reliable task beliefs.

# 4.2 GreEDy CONTEXT-BASED META-TESTING

GCC is a context-based meta-RL algorithm (Rakelly et al., 2019), whose adaptation protocol follows the framework of posterior sampling (Strens, 2000). The RL agent will iteratively update task belief by interacting with the environment and improve policy based on belief states. This adaptation paradigm is generalized from Bayesian inference (Thompson, 1933) and has a solid background in RL theory (Agrawal & Goyal, 2012). To ensure a feasible Bayesian belief update, we adopt a common heuristic of offline RL (Fujimoto et al., 2019; Kostrikov et al., 2021), i.e., few-shot out-of-distribution data usually has lower return. This heuristic suggests that the true return of episodes in online adaptation can be a good measure to distinguish data in offline meta-training distribution. In this way, GCC contains two steps to perform a greedy posterior belief update at each iteration: (i) a diverse sampling of latent task variables, and (ii) a greedy context mechanism that selects a task embedding with the highest return to update the belief. For each iteration  $t$ , denote the current task belief by  $b_{t}$ , the meta-testing task by  $\kappa_{test}$ , and the number of belief updating iterations by  $n_{it}$ .

A diverse sampling of latent task variables will generate  $n_t^z$  candidates of the task embedding  $z_t$ , denoting by  $\mathcal{Z}_t = \{z_t^i\}_{i=0}^{n_t^z - 1}$ , to provide various policies  $\pi(a|s, z_t^i)$  for the subsequent context selection mechanism. Due to the contrastive loss applied to the representation of  $z$  (Li et al., 2020), a closer task embedding  $z$  may yield a more similar policy. Hence, to encourage the diversity of policies, each candidate  $z_t^i \in \mathcal{Z}_t$  is designed by

$$
z _ {t} ^ {i} = \underset {\tilde {z} \in \tilde {\mathcal {Z}} _ {t} ^ {i}} {\arg \max } \left(\min  \left(\underset {j <   i} {\min } \left\| \tilde {z} - z _ {t} ^ {j} \right\| _ {2}, \underset {t ^ {\prime} <   t, k <   n _ {t ^ {\prime}} ^ {z}} {\min } \left\| \tilde {z} - z _ {t ^ {\prime}} ^ {k} \right\| _ {2}\right)\right) \text {a n d} \tilde {\mathcal {Z}} _ {t} ^ {i} = \left\{\tilde {z} _ {t} ^ {i, u} \sim b _ {t} \right\} _ {u = 0} ^ {n _ {\bar {z}} - 1}, (7)
$$

in which  $z_{t}^{i}$  aims to maximize the minimum distance to the previous embeddings among  $n_{\tilde{z}}$  samples. This greedy method is similar to Farthest-Point Clustering (Gonzalez, 1985), a two-approximation algorithm for an NP-hard problem Minimax Facility Location (Fowler et al., 1981), which seeks a set of locations to minimize the maximum distance from other facilities to the set.

A greedy selective context mechanism will select a latent task variable  $z_{t} \in \mathcal{Z}_{t}$  with the highest return to update the task belief  $b_{t}$ . To evaluate the return of each  $z_{t}^{i}$ , GCC utilizes the policy  $\pi\left(a|s,z_t^i\right)$  to draw  $n_e$  episodes in  $\kappa_{test}$ , denoting by  $\mathcal{E}_t^i = \left\{e_t^{i,j}\right\}_{j = 0}^{n_e - 1}$ . The policy evaluation of  $z_{t}^{i}$  can be approximated by sampling:

$$
\mathcal {J} _ {\kappa_ {t e s t}} \left(z _ {t} ^ {i}\right) = \mathcal {J} _ {\kappa_ {t e s t}} \left(\pi \left(a | s, z _ {t} ^ {i}\right)\right) \approx \widetilde {\mathcal {J}} _ {\kappa_ {t e s t}} \left(\mathcal {E} _ {t} ^ {i}\right) = \frac {1}{n _ {e}} \sum_ {j = 0} ^ {n _ {e} - 1} \left(\sum_ {k = 0} ^ {H - 1} r _ {k} \left(e _ {t} ^ {i, j}\right)\right), \tag {8}
$$

where  $\widetilde{\mathcal{J}}_{\kappa_{test}}(\mathcal{E}_t^i)$  is the average return of episodes  $\mathcal{E}_t^i$  and  $r_k(e_t^{i,j})$  is the reward of  $k$ -th step in an episode  $e_t^{i,j}$ . The task belief update in GCC consists of two phases: (i) an initial stage and (ii) an iterative optimization process. In the initial stage, GCC aims to find a reliable initial task inference  $z_0$  using a large amount of  $n_0^z$  diverse candidates, i.e., the initial context is  $c_0 = \arg \max_{\mathcal{E}_0 \in \{\mathcal{E}_0^i\}} \widetilde{\mathcal{J}}_{\kappa_{test}}(\mathcal{E}_0)$ , maintaining the corresponding task embedding  $z_0$ , and deriving the posterior belief  $b_1 = q(z|c_0)$ . In the following iterations, GCC utilizes an iterative optimization method to maximize final performance during few-shot online adaptation, i.e., when  $t > 1$ , let  $n_t^z = 1$  and if  $\widetilde{\mathcal{J}}_{\kappa_{test}}(\mathcal{E}_t^0) > \widetilde{\mathcal{J}}_{\kappa_{test}}(c_{t-1})$ , we have  $c_t = \mathcal{E}_t^0$  and update the posterior belief  $b_{t+1} = q(z| \cup_{t' \leq t} c_{t'})$ , otherwise  $c_t = c_{t-1}$  and keep the belief  $b_{t+1} = b_t$ . The final policy  $\pi(a|s, z_t)$  will depend on the optimal task embedding  $z_t$  with the highest return.

Algorithm 1 GCC: Greedy Context-based data distribution Correction  
1: Require: An offline multi-task dataset  $\mathcal{D}^+$ , a meta-testing task  $\kappa_{test} \sim p(\kappa)$ , the number of iterations  $n_{it}, n_{0}^{z}$ , and a context-based offline meta-training algorithm  $\mathbb{A}$  (i.e., FOCAL)  
2: Randomly initialize a context encoder  $q(z|c)$ , a policy  $\pi(a|s, z)$ , and a value function  $Q(s, a, z)$   
3: Offline meta-train  $q$ ,  $\pi$ , and  $Q$  given an algorithm  $\mathbb{A}$  and a dataset  $\mathcal{D}^+$ $\triangleright$  Offline meta-training  
4: Generate a prior task distribution  $q(z)$  using the dataset  $\mathcal{D}^+$ $\triangleright$  Start online meta-testing  
5: Collect diverse adaptation episodes  $\{\mathcal{E}_0^i\}_{i=0}^{n_0^z-1}$  using  $q(z)$  and  $\pi$  in  $\kappa_{test}$   
6: Compute the greedy context  $c_0$ , task embedding  $z_0$ , and posterior belief  $b_1$ $\triangleright$  An initial stage  
7: for  $t = 1\dots n_{it} - 1$  do  $\triangleright$  An iterative optimization process  
8: Collect a diverse episode  $\mathcal{E}_t^0$  using  $b_t$  and  $\pi(a|s, z_t)$  in  $\kappa_{test}$   
9: Compute the greedy context  $c_t$  and posterior belief  $b_{t+1}$   
10: Derive the final policy  $\pi_{\mathrm{out}}(a|s, z_t)$  with the optimal task embedding  $z_t$   
11: Return:  $\pi_{\mathrm{out}}$

# 5 EXPERIMENTS

In this section, we first study a didactic example to analyze the out-of-distribution problem, and show how GCC alleviates this problem by its greedy selective mechanism. Then we conduct large-scale experiments on Meta-World ML1(Yu et al., 2020a), a popular meta-RL benchmark that consists of 50 robot arm manipulation task sets. Finally, we perform ablation studies to analyze GCC's sensitivity to hyper-parameter settings and dataset qualities. Following FOCAL (Li et al., 2020), we use expert-level datasets sampled by policies trained with SAC on the corresponding tasks. We compare against FOCAL (Li et al., 2020) and MACAW (Mitchell et al., 2021), as well as their online adaptation variants. FOCAL is built upon PEARL (Rakelly et al., 2019) and uses contrastive losses to learn context embeddings, while MACAW is a MAML-based (Finn et al., 2017) algorithm and incorporates AWR (Peng et al., 2019). Both algorithms are originally proposed for the offline adaptation settings (i.e., with expert context). For online adaptation, we use online experiences instead of expert contexts, and adopt PEARL and MAML's adaptation manner to FOCAL and MACAW, respectively. Results in this section are averaged over six random seeds, and variance is measured by  $95\%$  bootstrapped confidence interval. Detailed hyper-parameter settings are deferred to Appendix B.

# 5.1 DIDACTIC EXAMPLE

We introduce Point-Robot, a simple 2D navigation task set commonly used in meta-RL (Rakelly et al., 2019; Zhang et al., 2021). Figure 3(a) illustrates the distribution mismatch between offline meta-training and online adaptation, as the dataset is collected by task-dependent behavior policies. As a consequence, directly performing adaptation with the online collected trajectories lead to poor adaptation performance, as shown in Figure 3(b). GCC fixes this problem by filtering out out-of-distribution data, and

greedily selecting trajectories. At the end of the initial stage, GCC only updates its belief with the orange trajectory as it has the highest return. After the initial stage, GCC iteratively optimizes the posterior belief to get the final policy. As shown in Figure 3(b), GCC achieves comparable performance to FOCAL with expert context and significantly outperforms FOCAL with online adaptation.

![](images/a9824fddf78e7800ec2ed9eb347996eb69272bd30129284efb834be4a840c288.jpg)  
(a)

![](images/009e3945814f2b1aa201f2833deea4f23ec52803cbb633de47d0300efb0f4a02.jpg)  
Figure 3: (a) Illustration of data distribution mismatch between offline meta-training (blue) and online adaptation (green and red trajectories). (b) Adaptation performance of GCC, FOCAL, and FOCAL with expert context.  
(b)

# 5.2 MAIN RESULTS

We evaluate on Meta-World ML1(Yu et al., 2020a), a popular meta-RL benchmark that consists of 50 robot arm manipulation task sets. Each task set consists of 50 tasks with different goals. For each task set, we use 40 tasks as meta-training tasks, and remain the other 10 tasks as meta-testing tasks. As shown in Table 1, GCC significantly outperforms baselines under the online context setting. With

expert contexts, FOCAL and MACAW both achieves reasonable performance. GCC achieves better or comparable performance to baselines with expert contexts, which implies that expert contexts may not be necessary for offline meta-RL. Under online contexts, FOCAL fails due to the data distribution mismatch between offline training and online adaptation. MACAW has the ability of online fine-tuning as it is based on MAML, but it also suffers from the distribution mismatch problem, and online fine-tuning can hardly improve its performance within a few adaptation episodes.

Table 1: Algorithms' normalized scores averaged over 50 Meta-World ML1 task sets. Scores are normalized by expert-level policy return.  

<table><tr><td>GCC</td><td>FOCAL</td><td>MACAW</td><td>FOCAL with Expert Context</td><td>MACAW with Expert Context</td></tr><tr><td>0.73 ± 0.07</td><td>0.53 ± 0.08</td><td>0.18 ± 0.09</td><td>0.67 ± 0.07</td><td>0.68 ± 0.07</td></tr></table>

Table 2 shows algorithms' performance on 20 representative Meta-World ML1 task sets, as well a sparse-reward version of Point-Robot and Cheetah-Vel, which are popular meta-RL tasks (Li et al., 2020). GCC achieves remarkable performance in most tasks and may fail in some hard tasks as offline meta-training is difficult. We find that GCC achieves better or comparable performance to baselines with expert contexts on 33 out of the 50 task sets. Detailed algorithm performance on all 50 tasks as well as comparison to baselines with expert contexts are deferred to Appendix C.

Table 2: Performance on example tasks, a bunch of Meta-World ML1 tasks with normalized scores.  

<table><tr><td>Example Environments</td><td>GCC</td><td>FOCAL</td><td>MACAW</td></tr><tr><td>Coffee-Push-V2</td><td>1.26 ± 0.13</td><td>0.66 ± 0.07</td><td>0.01 ± 0.01</td></tr><tr><td>Faucet-Close-V2</td><td>1.12 ± 0.01</td><td>1.06 ± 0.02</td><td>0.07 ± 0.01</td></tr><tr><td>Faucet-Open-V2</td><td>1.05 ± 0.02</td><td>1.01 ± 0.02</td><td>0.08 ± 0.04</td></tr><tr><td>Door-Close-V2</td><td>0.99 ± 0.00</td><td>0.97 ± 0.01</td><td>0.00 ± 0.00</td></tr><tr><td>Drawer-Close-V2</td><td>0.99 ± 0.02</td><td>0.96 ± 0.04</td><td>0.53 ± 0.50</td></tr><tr><td>Door-Lock-V2</td><td>0.97 ± 0.01</td><td>0.90 ± 0.02</td><td>0.25 ± 0.11</td></tr><tr><td>Plate-Slide-Back-V2</td><td>0.96 ± 0.02</td><td>0.58 ± 0.06</td><td>0.21 ± 0.17</td></tr><tr><td>Dial-Turn-V2</td><td>0.91 ± 0.05</td><td>0.84 ± 0.09</td><td>0.00 ± 0.00</td></tr><tr><td>Handle-Press-V2</td><td>0.88 ± 0.05</td><td>0.87 ± 0.02</td><td>0.28 ± 0.10</td></tr><tr><td>Hamme-V2</td><td>0.84 ± 0.06</td><td>0.59 ± 0.07</td><td>0.10 ± 0.01</td></tr><tr><td>Button-Press-V2</td><td>0.74 ± 0.08</td><td>0.68 ± 0.14</td><td>0.02 ± 0.01</td></tr><tr><td>Push-Wall-V2</td><td>0.71 ± 0.15</td><td>0.43 ± 0.06</td><td>0.23 ± 0.18</td></tr><tr><td>Hand-Insert-V2</td><td>0.63 ± 0.04</td><td>0.29 ± 0.07</td><td>0.02 ± 0.01</td></tr><tr><td>Peg-Unplug-Side-V2</td><td>0.56 ± 0.07</td><td>0.19 ± 0.09</td><td>0.00 ± 0.00</td></tr><tr><td>Bin-Picking-V2</td><td>0.53 ± 0.16</td><td>0.31 ± 0.21</td><td>0.66 ± 0.11</td></tr><tr><td>Soccer-V2</td><td>0.44 ± 0.04</td><td>0.11 ± 0.03</td><td>0.38 ± 0.31</td></tr><tr><td>Coffee-Pull-V2</td><td>0.40 ± 0.05</td><td>0.23 ± 0.04</td><td>0.19 ± 0.12</td></tr><tr><td>Pick-Place-Wall-V2</td><td>0.28 ± 0.12</td><td>0.09 ± 0.04</td><td>0.39 ± 0.25</td></tr><tr><td>Pick-Out-Of-Hole-V2</td><td>0.26 ± 0.25</td><td>0.16 ± 0.16</td><td>0.59 ± 0.06</td></tr><tr><td>Handle-Pull-Side-V2</td><td>0.14 ± 0.04</td><td>0.13 ± 0.09</td><td>0.00 ± 0.00</td></tr><tr><td>Cheetah-Vel</td><td>-171.52 ± 21.96</td><td>-287.70 ± 30.62</td><td>-233.97 ± 23.46</td></tr><tr><td>Point-Robot</td><td>-5.10 ± 0.26</td><td>-15.38 ± 0.95</td><td>-14.61 ± 0.98</td></tr><tr><td>Point-Robot-Sparse</td><td>7.78 ± 0.64</td><td>0.83 ± 0.37</td><td>0.00 ± 0.00</td></tr></table>

# 5.3 ABLATION STUDY

In this subsection, we conduct various ablation studies to investigate the robustness of GCC in dataset quality and hyper-parameters.

Initial stage length. Table 3 shows GCC's performance with different initial stage lengths. The total number of adaptation episodes is 20. We find that GCC performs well during 10-15 episodes, which is  $50\% -75\%$  of the total number of adaptation episodes. A small initial stage length (5) may lead to a possibly unreliable task belief and cause a degrade in performance. The 19-episode does not perform the iterative optimization process, and the task belief updates will not converge.

Number of latent task variables sampled in the initial phase.  $n_{\tilde{z}}$  controls the number of diverse samples used to produce the task embedding candidates  $z_{t}^{i}$ . As shown in Table 4, GCC is robust to changes of  $n_{\tilde{z}}$ , and works in a large range from 5 to 20.

Dataset Quality. We test GCC and baselines with several "medium" datasets, which are collected by periodically evaluating policies of SAC. As shown in Table 5, GCC still significantly outperforms baseline algorithms on medium datasets, which implies GCC's ability to learn various datasets.

Table 3: GCC's performance with various initial stage lengths.  

<table><tr><td>Environment</td><td>5 Episodes</td><td>10 Episodes</td><td>15 Episodes</td><td>19 Episodes</td></tr><tr><td>Point-Robot</td><td>-6.04 ± 0.31</td><td>-5.11 ± 0.21</td><td>-5.10 ± 0.26</td><td>-5.37 ± 0.11</td></tr><tr><td>Point-Robot-Sparse</td><td>4.04 ± 0.58</td><td>7.78 ± 0.64</td><td>8.07 ± 0.62</td><td>7.29 ± 0.50</td></tr></table>

Table 4: GCC's performance with various  ${n}_{\bar{z}}\mathrm{\;s}$  .  

<table><tr><td>Environment</td><td>n̅z=1</td><td>n̅z=5</td><td>n̅z=10</td><td>n̅z=20</td></tr><tr><td>Point-Robot</td><td>-5.92 ± 0.31</td><td>-5.11 ± 0.21</td><td>-4.94 ± 0.16</td><td>-4.99 ± 0.23</td></tr><tr><td>Point-Robot-Sparse</td><td>5.66 ± 0.63</td><td>7.78 ± 0.64</td><td>7.31 ± 0.74</td><td>7.78 ± 0.57</td></tr></table>

Table 5: Algorithms' performance on datasets of various qualities.  

<table><tr><td>Environment</td><td>GCC</td><td>FOCAL</td><td>MACAW</td></tr><tr><td>Sweep-V2</td><td>0.77 ± 0.04</td><td>0.32 ± 0.08</td><td>0.20 ± 0.20</td></tr><tr><td>Sweep-V2-Medium</td><td>0.59 ± 0.13</td><td>0.38 ± 0.13</td><td>0.04 ± 0.03</td></tr><tr><td>Peg-Insert-Side-V2</td><td>0.30 ± 0.04</td><td>0.08 ± 0.03</td><td>0.00 ± 0.00</td></tr><tr><td>Peg-Insert-Side-V2-Medium</td><td>0.30 ± 0.14</td><td>0.10 ± 0.07</td><td>0.00 ± 0.00</td></tr></table>

# 6 RELATED WORK

In the literature, offline meta-RL methods utilize a context-based (Rakelly et al., 2019) or gradient-based (Finn et al., 2017) meta-RL framework to solve new tasks with few-shot adaptation. They utilize the techniques of contrastive learning (Li et al., 2020; Yuan & Lu, 2022), more expressive power (Mitchell et al., 2021), or reward relabeling (Dorfman et al., 2021; Pong et al., 2022) with various popular offline single-task RL tricks, i.e., using KL divergence (Wu et al., 2019; Peng et al., 2019; Nair et al., 2020) or explicitly constraining the policy to be close to the dataset (Fujimoto et al., 2019; Zhou et al., 2020). However, these methods always require extra information for fast adaptation, such as offline context for testing tasks (Li et al., 2020; Mitchell et al., 2021; Yuan & Lu, 2022), oracle reward functions (Dorfman et al., 2021), or free interactions without reward supervision (Pong et al., 2022). To address the challenge, we propose GCC, a greedy context mechanism with theoretical motivation, to perform effective online adaptation without requiring additional information.

The concepts of distribution shift in  $z$ -space in Pong et al. (2022) and MDP ambiguity in Dorfman et al. (2021) are related to the data distribution mismatch proposed in this paper. We reveal that the task-dependent behavior policies will induce different reward and transition distribution between offline meta-training and online adaptation, which is an essential factor in the phenomenon of distribution shift in  $z$ -space (Pong et al., 2022). After filtering out these out-of-distribution data, GCC can maintain an overall belief about the task with behavior policies to address MDP ambiguity.

# 7 CONCLUSION

This paper formalizes data distribution mismatch in offline meta-RL with online adaptation and introduces GCC, a novel context-based online adaptation approach. Inspired by theoretical implications, GCC adopts a greedy context mechanism to filter out out-of-distribution with lower return for online data correction. We demonstrate that GCC can perform accurate task inference and achieve state-of-the-art performance on Meta-World ML1 benchmark with 50 tasks. Compared to offline adaptation baselines with expert context, GCC also performs better or comparably, suggesting that offline context may not be necessary for the testing environments. One potential future direction is to extend GCC to gradient-based online adaptation methods with data distribution correction.

# REFERENCES

Shipra Agrawal and Navin Goyal. Analysis of thompson sampling for the multi-armed bandit problem. In Conference on learning theory, pp. 39-1. JMLR Workshop and Conference Proceedings, 2012.  
Alex Slivkins Alekh Agarwal. Lecture 10: Reinforcement learning. In COMS E6998.001, Columbia University. 2017. OpenCourseLecture.  
Karl J Astrom. Optimal control of markov decision processes with incomplete state estimation. J. Math. Anal. Applic., 10:174-205, 1965.  
Dimitri P Bertsekas and John N Tsitsiklis. Neuro-dynamic programming: an overview. In Proceedings of 1995 34th IEEE conference on decision and control, volume 1, pp. 560-564. IEEE, 1995.  
Anthony R Cassandra, Leslie Pack Kaelbling, and Michael L Littman. Acting optimally in partially observable stochastic domains. In Aaai, volume 94, pp. 1023-1028, 1994.  
Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International Conference on Machine Learning, pp. 1042-1051. PMLR, 2019.  
Ron Dorfman, Idan Shenfeld, and Aviv Tamar. Offline meta reinforcement learning-identifiability challenges and effective data collection strategies. Advances in Neural Information Processing Systems, 34, 2021.  
Simon S Du, Sham M Kakade, Ruosong Wang, and Lin F Yang. Is a good representation sufficient for sample efficient reinforcement learning? arXiv preprint arXiv:1910.03016, 2019.  
Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. Rl2: Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.  
Michael O'Gordon Duff. Optimal Learning: Computational procedures for Bayes-adaptive Markov decision processes. University of Massachusetts Amherst, 2002.  
Chelsea Finn and Sergey Levine. Meta-learning: from few-shot learning to rapid reinforcement learning. In ICML, 2019.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1126-1135. JMLR.org, 2017.  
Robert J Fowler, Michael S Paterson, and Steven L Tanimoto. Optimal packing and covering in the plane are np-complete. Information processing letters, 12(3):133-137, 1981.  
Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, pp. 2052-2062. PMLR, 2019.  
Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, Aviv Tamar, et al. Bayesian reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 8(5-6):359-483, 2015.  
Teofilo F Gonzalez. Clustering to minimize the maximum intercluster distance. Theoretical computer science, 38:293-306, 1985.  
Omer Gottesman, Fredrik Johansson, Matthieu Komorowski, Aldo Faisal, David Sontag, Finale Doshi-Velez, and Leo Anthony Celi. Guidelines for reinforcement learning in healthcare. Nature medicine, 25(1):16-18, 2019.  
Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2019.  
Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In International Conference on Machine Learning, pp. 5084-5096. PMLR, 2021.  
Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99-134, 1998.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021.  
Aviral Kumar, Justin Fu, Matthew Soh, George Tucker, and Sergey Levine. Stabilizing off-policy q-learning via bootstrapping error reduction. Advances in Neural Information Processing Systems, 32, 2019.  
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179-1191, 2020.  
Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.  
Lanqing Li, Rui Yang, and Dijun Luo. Focal: Efficient fully-offline meta-reinforcement learning via distance metric learning and behavior regularization. arXiv preprint arXiv:2010.01112, 2020.  
Yao Liu, Adith Swaminathan, Alekh Agarwal, and Emma Brunskill. Provably good batch reinforcement learning without great exploration. arXiv preprint arXiv:2007.08202, 2020.  
Eric Mitchell, Rafael Rafailov, Xue Bin Peng, Sergey Levine, and Chelsea Finn. Offline meta-reinforcement learning with advantage weighting. In International Conference on Machine Learning, pp. 7780-7791. PMLR, 2021.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529, 2015.  
Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.  
Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. arXiv preprint arXiv:1910.00177, 2019.  
Vitchyr H Pong, Ashvin V Nair, Laura M Smith, Catherine Huang, and Sergey Levine. Offline meta-reinforcement learning with online self-supervision. In International Conference on Machine Learning, pp. 17811-17829. PMLR, 2022.  
Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, and Chelsea Finn. Offline reinforcement learning from images with latent space models. In Learning for Dynamics and Control, pp. 1154-1168. PMLR, 2021.  
Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In International Conference on Machine Learning, pp. 5331-5340, 2019.  
Tongzheng Ren, Jialian Li, Bo Dai, Simon S Du, and Sujay Sanghavi. Nearly horizon-free offline reinforcement learning. Advances in neural information processing systems, 34, 2021.  
Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Pessimistic q-learning for offline reinforcement learning: Towards optimal sample complexity. arXiv preprint arXiv:2202.13890, 2022.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354-359, 2017.  
Richard D Smallwood and Edward J Sondik. The optimal control of partially observable markov processes over a finite horizon. Operations research, 21(5):1071-1088, 1973.  
Malcolm Strens. A bayesian framework for reinforcement learning. In ICML, volume 2000, pp. 943-950, 2000.  
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.

Csaba Szepesvári. Lecture 17: Batch rl: Introduction discussion. In CMPUT 653: Theoretical Foundations of Reinforcement Learning, University of Alberta. 2022. OpenCourseLecture.  
William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3/4):285-294, 1933.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.  
Jane X Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, and Matt Botvinick. Learning to reinforcement learn. arXiv preprint arXiv:1611.05763, 2016.  
Jianhao Wang, Wenzhe Li, Haozhe Jiang, Guangxiang Zhu, Siyuan Li, and Chongjie Zhang. Offline reinforcement learning with reverse model-based imagination. Advances in Neural Information Processing Systems, 34:29420-29432, 2021.  
Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.  
Ming Yin and Yu-Xiang Wang. Towards instance-optimal offline reinforcement learning with pessimism. Advances in neural information processing systems, 34, 2021.  
Ming Yin, Yu Bai, and Yu-Xiang Wang. Near-optimal provable uniform convergence in offline policy evaluation for reinforcement learning. arXiv preprint arXiv:2007.03760, 2020.  
Ming Yin, Yu Bai, and Yu-Xiang Wang. Near-optimal offline reinforcement learning via double variance reduction. Advances in neural information processing systems, 34, 2021.  
Fisher Yu, Wenqi Xian, Yingying Chen, Fangchen Liu, Mike Liao, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving video database with scalable annotation tooling. arXiv preprint arXiv:1805.04687, 2(5):6, 2018.  
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pp. 1094-1100. PMLR, 2020a.  
Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on robot learning, pp. 1094-1100. PMLR, 2020b.  
Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. Combo: Conservative offline model-based policy optimization. arXiv preprint arXiv:2102.08363, 2021.  
Haoqi Yuan and Zongqing Lu. Robust task representations for offline meta-reinforcement learning via contrastive learning. In International Conference on Machine Learning, pp. 25747-25759. PMLR, 2022.  
Jin Zhang, Jianhao Wang, Hao Hu, Tong Chen, Yingfeng Chen, Changjie Fan, and Chongjie Zhang. Metacure: Meta reinforcement learning with empowerment-driven exploration. In International Conference on Machine Learning, pp. 12600-12610. PMLR, 2021.  
Wenxuan Zhou, Sujay Bajracharya, and David Held. Plas: Latent action space for offline reinforcement learning. arXiv preprint arXiv:2011.07213, 2020.  
Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, and Shimon Whiteson. Varibad: A very good method for bayes-adaptive deep rl via meta-learning. In International Conference on Learning Representations, 2019.
