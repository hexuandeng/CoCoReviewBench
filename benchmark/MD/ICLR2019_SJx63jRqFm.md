# DIVERSITY IS ALL YOU NEED: LEARNING SKILLS WITHOUT A REWARD FUNCTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Intelligent creatures can explore their environments and learn useful skills without supervision. In this paper, we propose "Diversity is All You Need"(DIAYN), a method for learning useful skills without a reward function. Our proposed method learns skills by maximizing an information theoretic objective using a maximum entropy policy. On a variety of simulated robotic tasks, we show that this simple objective results in the unsupervised emergence of diverse skills, such as walking and jumping. In a number of reinforcement learning benchmark environments, our method is able to learn a skill that solves the benchmark task despite never receiving the true task reward. We show how pretrained skills can provide a good parameter initialization for downstream tasks, and can be composed hierarchically to solve complex, sparse reward tasks. Our results suggest that unsupervised discovery of skills can serve as an effective pretraining mechanism for overcoming challenges of exploration and data efficiency in reinforcement learning.

# 1 INTRODUCTION

Deep reinforcement learning (RL) has been demonstrated to effectively learn a wide range of reward-driven skills, including playing games (Mnih et al., 2013; Silver et al., 2016), controlling robots (Gu et al., 2017; Schulman et al., 2015b), and navigating complex environments (Zhu et al., 2017; Mirowski et al., 2016). However, intelligent creatures can explore their environments and learn useful skills even without supervision, so that when they are later faced with specific goals, they can use those skills to satisfy the new goals quickly and efficiently.

Learning skills without reward has several practical applications. Environments with sparse rewards effectively have no reward until the agent randomly reaches a goal state. Learning useful skills without supervision may help address challenges in exploration in these environments. For long horizon tasks, skills discovered without reward can serve as primitives for hierarchical RL, effectively shortening the episode length. In many practical settings, interacting with the environment is essentially free, but evaluating the reward requires human feedback (Christiano et al., 2017). Unsupervised learning of skills may reduce the amount of supervision necessary to learn a task. While we can take the human out of the loop by designing a reward function, it is challenging to design a reward function that elicits the desired behaviors from the agent (Hadfield-Menell et al., 2017). Finally, when given an unfamiliar environment, it is challenging to determine the range of tasks an agent should be able to learn. Unsupervised skill discovery partially answers this question.<sup>1</sup>

Autonomous acquisition of useful skills without any reward signal is an exceedingly challenging problem. A skill is a latent-conditioned policy that alters that state of the environment in a consistent way. We consider the setting where the reward function is unknown, so we want to learn a set of skills by maximizing the utility of this set. Making progress on this problem requires specifying a learning objective that ensures that each skill individually is distinct and that the skills collectively explore large parts of the state space. In this paper, we show how a simple objective based on mutual information can enable RL agents to autonomously discover such skills. These skills are useful for a number of applications, including hierarchical reinforcement learning and imitation learning.

We propose a method for learning diverse skills with deep RL in the absence of any rewards. We hypothesize that in order to acquire skills that are useful, we must train the skills so that they

maximize coverage over the set of possible behaviors. While one skill might perform a useless behavior like random dithering, other skills should perform behaviors that are distinguishable from random dithering, and therefore more useful. A key idea in our work is to use discriminability between skills as an objective. Further, skills that are distinguishable are not necessarily maximally diverse – a slight difference in states makes two skills distinguishable, but not necessarily diverse in a semantically meaningful way. To combat problem, we want to learn skills that not only are distinguishable, but also are as diverse as possible. By learning distinguishable skills that are as random as possible, we can “push” the skills away from each other, making each skill robust to perturbations and effectively exploring the environment. By maximizing this objective, we can learn skills that run forward, do backflips, skip backwards, and perform face flops (see Figure 3).

Our paper makes five contributions. First, we propose a method for learning useful skills without any rewards. We formalize our discriminability goal as maximizing an information theoretic objective with a maximum entropy policy. Second, we show that this simple exploration objective results in the unsupervised emergence of diverse skills, such as running and jumping, on several simulated robotic tasks. In a number of RL benchmark environments, our method is able to solve the benchmark task despite never receiving the true task reward. In these environments, some of the learned skills correspond to solving the task, and each skill that solves the task does so in a distinct manner. Third, we propose a simple method for using learned skills for hierarchical RL and find this methods solves challenging tasks. Four, we demonstrate how skills discovered can be quickly adapted to solve a new task. Finally, we show how skills discovered can be used for imitation learning.

# 2 RELATED WORK

Previous work on hierarchical RL has learned skills to maximize a single, known, reward function by jointly learning a set of skills and a meta-controller (e.g., (Bacon et al., 2017; Heess et al., 2016; Dayan & Hinton, 1993; Frans et al., 2017; Krishnan et al., 2017; Florensa et al., 2017)). One problem with joint training (also noted by Shazeer et al. (2017)) is that the meta-policy does not select "bad" options, so these options do not receive any reward signal to improve. Our work prevents this degeneracy by using a random meta-policy during unsupervised skill-learning, such that neither the skills nor the meta-policy are aiming to solve any single task. A second importance difference is that our approach learns skills with no reward. Eschewing a reward function not only avoids the difficult problem of reward design, but also allows our method to learn task-agnostic.

Related work has also examined connections between RL and information theory (Ziebart et al., 2008; Schulman et al., 2017; Nachum et al., 2017; Haarnoja et al., 2017) and developed maximum entropy algorithms with these ideas. Haarnoja et al. (2018; 2017). Recent work has also applied tools from information theory to skill discovery. Mohamed & Rezende (2015) and Jung et al. (2011) use the mutual information between states and actions as a notion of empowerment for an intrinsically motivated agent. Our method maximizes the mutual information between states and skills, which can be interpreted as maximizing the empowerment of a hierarchical agent whose action space is the set of skills. Hausman et al. (2018), Florensa et al. (2017), and Gregor et al. (2016) showed that a discriminability objective is equivalent to maximizing the mutual information between the latent skill  $z$  and some aspect of the corresponding trajectory. Hausman et al. (2018) considered the setting with many tasks and reward functions and Florensa et al. (2017) considered the setting with a single task reward. Three important distinctions allow us to apply our method to tasks significantly more complex than the gridworlds in Gregor et al. (2016). First, we use maximum entropy policies to force our skills to be diverse. Our theoretical analysis shows that including entropy maximization in the RL objective results in the mixture of skills being maximum entropy in aggregate. Second, we fix the prior distribution over skills, rather than learning it. Doing so prevents our method from collapsing to sampling only a handful of skills. Third, while the discriminator in Gregor et al. (2016) only looks at the final state, our discriminator looks at every state, which provides additional reward signal. These three crucial differences help explain how our method learns useful skills in complex environments.

Prior work in neuroevolution and evolutionary algorithms has studied how complex behaviors can be learned by directly maximizing diversity (Lehman & Stanley, 2011a;b; Woolley & Stanley, 2011; Stanley & Miikkulainen, 2002; Such et al., 2017; Pugh et al., 2016; Mouret & Doncieux, 2009). While this prior work uses diversity maximization to obtain better solutions, we aim to acquire complex skills with minimal supervision to improve efficiency (i.e., reduce the number of objective function queries) and as a stepping stone for imitation learning and hierarchical RL. We focus on

![](images/bebed66bdfa7653ec3467d684997fe74bf47c49877b2b03a7b1d6ac1b1510105.jpg)  
Figure 1: DIAYN Algorithm: We update the discriminator to better predict the skill, and update the skill to visit diverse states that make it more discriminable.

# Algorithm 1:DIAYN

# while not converged do

Sample skill  $z\sim p(z)$  and initial state  $s_0\sim p_0(s)$  for  $t\gets 1$  to steps_per_episode do

Sample action  $a_{t} \sim \pi_{\theta}(a_{t} \mid s_{t}, z)$  from skill.

Step environment:  $s_{t + 1} \sim p(s_{t + 1} \mid s_t, a_t)$ .

Compute  $q_{\phi}(z \mid s_{t+1})$  with discriminator.

Set skill reward  $r_t = \log q_\phi(z \mid s_{t+1}) - \log p(z)$

Update policy  $(\theta)$  to maximize  $r_t$  with SAC.

Update discriminator  $(\phi)$  with SGD.

deriving a general, information-theoretic objective that does not require manual design of distance metrics and can be applied to any RL task without additional engineering.

Previous work has studied intrinsic motivation in humans and learned agents. Ryan & Deci (2000); Bellemare et al. (2016); Fu et al. (2017); Schmidhuber (2010); Oudeyer et al. (2007); Pathak et al. (2017); Baranes & Oudeyer (2013). While these previous works use an intrinsic motivation objective to learn a single policy, we propose an objective for learning many, diverse policies. Concurrent work Achiam et al. (2017) draws ties between learning discriminable skills and variational autoencoders. We show that our method scales to more complex tasks, likely because of algorithmic design choices, such as our use of an off-policy RL algorithm and conditioning the discriminator on individual states.

# 3 DIVERSITY IS ALL YOU NEED

We consider an unsupervised RL paradigm in this work, where the agent is allowed an unsupervised "exploration" stage followed by a supervised stage. In our work, the aim of the unsupervised stage is to learn skills that eventually will make it easier to maximize the task reward in the supervised stage. Conveniently, because skills are learned without a priori knowledge of the task, the learned skills can be used for many different tasks.

# 3.1 HOW IT WORKS

Our method for unsupervised skill discovery, DIAYN ("Diversity is All You Need"), builds off of three ideas. First, for skills to be useful, we want the skill to dictate the states that the agent visits. Different skills should visit different states, and hence be distinguishable. Second, we want to use states, not actions, to distinguish skills, because actions that do not affect the environment are not visible to an outside observer. For example, an outside observer cannot tell how much force a robotic arm applies when grasping a cup if the cup does not move. Finally, we encourage exploration and incentivize the skills to be as diverse as possible by learning skills that act as randomly as possible. Skills with high entropy that remain discriminable must explore a part of the state space far away from other skills, lest the randomness in its actions lead it to states where it cannot be distinguished.

We construct our objective using notation from information theory:  $S$  and  $A$  are random variables for states and actions, respectively;  $Z \sim p(z)$  is a latent variable, on which we condition our policy; we refer to a policy conditioned on a fixed  $Z$  as a "skill";  $I(\cdot ;\cdot)$  and  $\mathcal{H}[\cdot ]$  refer to mutual information and Shannon entropy, both computed with base  $e$ . In our objective, we maximize the mutual information between skills and states,  $I(A;Z)$ , to encode the idea that the skill should control which states the agent visits. Conveniently, this mutual information dictates that we can infer the skill from the states visited. To ensure that states, not actions, are used to distinguish skills, we minimize the mutual information between skills and actions given the state,  $I(A;Z\mid S)$ . Viewing all skills together with  $p(z)$  as a mixture of policies, we maximize the entropy  $\mathcal{H}[A\mid S]$  of this mixture policy. In summary, we maximize

$$
\begin{array}{l} \mathcal {F} (\theta) \triangleq I (S; Z) + \mathcal {H} [ A \mid S ] - I (A; Z \mid S) (1) \\ = \left(\mathcal {H} [ Z ] - \mathcal {H} [ Z \mid S ]\right) + \mathcal {H} [ A \mid S ] - \left(\mathcal {H} [ A \mid S ] - \mathcal {H} [ A \mid S, Z ]\right) \\ = \mathcal {H} [ Z ] - \mathcal {H} [ Z \mid S ] + \mathcal {H} [ A \mid S, Z ] (2) \\ \end{array}
$$

We rearranged our objective in Equation 2 to give intuition on how we optimize it. The first term encourages our prior distribution over  $p(z)$  to have high entropy. We fix  $p(z)$  to be uniform in our approach, guaranteeing that it has maximum entropy. The second term suggests that it should be easy to infer the skill  $z$  from the current state. The third term suggests that each skill should act as randomly as possible, which we achieve by using a maximum entropy policy to represent each skill. As we cannot integrate over all states and skills to compute  $p(z \mid s)$  exactly, we approximate this posterior with a learned discriminator  $q_{\phi}(z \mid s)$ . Jensen's Inequality tells us that replacing  $p(z \mid s)$  with  $q_{\phi}(z \mid s)$  gives us a variational lower bound  $\mathcal{G}(\theta, \phi)$  on our objective  $\mathcal{F}(\theta)$  (see (Agakov, 2004) for a detailed derivation):

$$
\begin{array}{l} \mathcal {F} (\theta) = \mathcal {H} [ A \mid S, Z ] - \mathcal {H} [ Z \mid S ] + \mathcal {H} [ Z ] \\ = \mathcal {H} [ A \mid S, Z ] + \mathbb {E} _ {z \sim p (z), s \sim \pi (z)} [ \log p (z \mid s) ] - \mathbb {E} _ {z \sim p (z)} [ \log p (z) ] \\ \geq \mathcal {H} [ A \mid S, Z ] + \mathbb {E} _ {z \sim p (z), s \sim \pi (z)} [ \log q _ {\phi} (z \mid s) - \log p (z) ] \triangleq \mathcal {G} (\theta , \phi) \\ \end{array}
$$

# 3.2 IMPLEMENTATION

We implement DIAYN with soft actor critic (Haarnoja et al., 2018), learning a policy  $\pi_{\theta}(a|s,z)$  that is conditioned on the latent variable  $z$ . Soft actor critic maximizes the policy's entropy over actions, which takes care of the entropy term in our objective  $\mathcal{G}$ . Following Haarnoja et al. (2018), we scale the entropy regularizer  $\mathcal{H}[a|s,z]$  by  $\alpha$ . We found empirically that an  $\alpha = 0.1$  provided a good trade-off between exploration and discriminability. We maximize the expectation in  $\mathcal{G}$  by replacing the task reward with the following pseudo-reward:

$$
r _ {z} (s, a) \triangleq \log q _ {\phi} (z \mid s) - \log p (z) \tag {3}
$$

We use a categorical distribution for  $p(z)$ . During unsupervised learning, we sample a skill  $z \sim p(z)$  at the start of each episode, and act according to that skill throughout the episode. The agent is rewarded for visiting states that are easy to discriminate, while the discriminator is updated to better infer the skill  $z$  from states visited. Entropy regularization occurs as part of the SAC update.

# 3.3 STABILITY

Unlike prior adversarial unsupervised RL methods (e.g., Sukhbaatar et al. (2017)), DIAYN forms a cooperative game, which avoids many of the instabilities of adversarial saddle-point formulations. On gridworlds, we can compute analytically that the unique optimum to the DIAYN optimization problem is to evenly partition the states between skills, with each skill assuming a uniform stationary distribution over its partition (proof in Appendix B). In the continuous and approximate setting, convergence guarantees would be desirable, but this is a very tall order: even standard RL methods with function approximation (e.g., DQN) lack convergence guarantees, yet such techniques are still useful. Empirically, we find DIAYN to be robust to random seed; varying the random seed does not noticeably affect the skills learned, and has little effect on downstream tasks (see Fig.s 4, 6, and 13).

# 4 EXPERIMENTS

In this section, we evaluate DIAYN and compare to prior work. First, we analyze the skills themselves, providing intuition for the types of skills learned, the training dynamics, and how we avoid problematic behavior in previous work. In the second half, we show how the skills can be used for downstream tasks, via policy initialization, hierarchy, imitation, outperforming competitive baselines on most tasks. We encourage readers to view videos<sup>3</sup> and code<sup>4</sup> for our experiments.

# 4.1 ANALYSIS OF LEARNED SKILLS

# Question 1. What skills does DIAYN learn?

2While our method uses stochastic policies, note that for deterministic policies in continuous action spaces,  $I(A;Z\mid S) = \mathcal{H}[A\mid S]$ . Thus, for deterministic policies, Equation 2 reduces to maximizing  $I(S;Z)$ .  
<https://sites.google.com/view/diversity-is-all-you-need/  
4https://OMITTED-FOR-ANONYMITY

![](images/b49d343af8a80617e2fd4b0107e82dee03addef0b2f8fa9d1a12c7e7680d0d5f.jpg)  
(a) 2D Navigation

![](images/b464f43fa0d43c64f797f1b95bd16520a998417da4d5cb2a91d3c7b0cded67ed.jpg)  
(b) Overlapping Skills

![](images/64ae15d550d972a8b84d089027c1d051716d9f42f2b0881d28d4b3ca2bd606a0.jpg)  
(c) Training Dynamics

![](images/2f6669395525b6fd1ee79d9cd0ebf8ac999753759920563b4117cea830202d84.jpg)  
Figure 2: (Left) DIAYN skills in a simple navigation environment; (Center) skills can overlap if they eventually become distinguishable; (Right) diversity of the rewards increases throughout training.  
Figure 3: Locomotion skills: Without any reward, DIAYN discovers skills for running, walking, hopping, flipping, and gliding. It is challenging to craft reward functions that elicit these behaviors.

We study the skills learned by DIAYN on tasks of increasing complexity, ranging from 2 DOF point navigation to 111 DOF ant locomotion. We first applied DIAYN to a simple 2D navigation environment. The agent starts in the center of the box, and can take actions to directly move its  $(x,y)$  position. Figure 2a illustrates how the 6 skills learned for this task move away from each other to remain distinguishable. Next, we applied DIAYN to two classic control tasks, inverted pendulum and mountain car. Not only does our approach learn skills that solve the task without rewards, it learns multiple distinct skills for solving the task. (See Appendix D for further analysis.)

Finally, we applied DIAYN to three continuous control tasks (Brockman et al., 2016): half cheetah, hopper, and ant. As shown in Figure 3, we learn a diverse set of primitive behaviors for all tasks. For half cheetah, we learn skills for running forwards and backwards at various speeds, as well as skills for doing flips and falling over; ant learns skills for jumping and walking in many types of curved trajectories (though none walk in a straight line); hopper learns skills for balancing, hopping forward and backwards, and diving. See Appendix D.4 for a comparison with VIME.

Question 2. How does the distribution of skills change during training?

While DIAYN learns skills without a reward function, as an outside observer, can we evaluate the skills throughout training to understand the training dynamics. Figure 2 shows how the skills for inverted pendulum and mountain car become increasingly diverse throughout training (Fig. 13 repeats this experiment for 5 random seeds, and shows that results are robust to initialization). Recall that our skills are learned with no reward, so it is natural that some skills correspond to small task reward while others correspond to large task reward.

Question 3. Does discriminating on single states restrict DIAYN to learn skills that visit disjoint sets of states?

Our discriminator operates at the level of states, not trajectories. While DIAYN favors skills that do not overlap, our method is not limited to learning skills that visit entirely disjoint sets of states.

Figure 2b shows a simple experiment illustrating this. The agent starts in a hallway (green star), and can move more freely once exiting the end of the hallway into a large room. Because RL agents are incentivized to maximize their cumulative reward, they may take actions that initially give no reward to reach states that eventually give high reward. In this environment, DIAYN learns skills that exit the hallway to make them mutually distinguishable.

Question 4. How does DIAYN differ from Variational Intrinsic Control (VIC) (Gregor et al., 2016)?

The key difference from the most similar prior work on unsupervised skill discovery, VIC, is our decision to not learn the prior  $p(z)$ . We found that VIC suffers from the "Matthew Effect" Merton (1968): VIC's learned prior  $p(z)$  will sample the more diverse skills more frequently, and hence only those skills will receive training signal to improve. To study this, we evaluated DIAYN and VIC on the half-cheetah environment, and plotting the effective number of skills (measured as  $\exp(\mathcal{H}[Z])$ ) throughout

![](images/199527738de42c883b68d7185c96ea3223427ec1166c378c1f16c9685ddbe8a4.jpg)  
Figure 4: Why use a fixed prior? In contrast to prior work, DIAYN continues to sample all skills throughout training.

training (details and more figures in Appendix E.2). The figure to the right shows how VIC quickly converges to a setting where it only samples a handful of skills. In contrast, DIAYN fixes the distribution over skills, which allows us to discover more diverse skills.

# 4.2 HARNESSING LEARNED SKILLS

The perhaps surprising finding that we can discover diverse skills without a reward function creates a building block for many problems in RL. For example, to find a policy that achieves a high reward on a task, it is often sufficient to simply choose the skill with largest reward. Three less obvious applications are adapting skills to maximize a reward, hierarchical RL, and imitation learning.

# 4.2.1 ACCELERATING LEARNING WITH POLICY INITIALIZATION

After DIAYN learns task-agnostic skills without supervision, we can quickly adapt the skills to solve a desired task. Akin to the use of pre-trained models in computer vision, we propose that DIAYN can serve as unsupervised pre-training for more sample-efficient finetuning of task-specific policies.

![](images/154045c4b58234f654bc748fe132d0a149c8e15011145765384afa889c628294.jpg)  
Figure 5: Policy Initialization: Using a DIAYN skill to initialize weights in a policy accelerates learning, suggesting that pretraining with DIAYN may be especially useful in resource constrained settings. Results are averages across 5 random seeds.

![](images/fc942768f3c437a14ca7ff9587a25a07f34919ce117203ed58ad5a8b00ba1c35.jpg)

![](images/1e2c013d5003749a09b8f47c2d8aa2dca6e014145409f97a61f491f7f039878f.jpg)

# Question 5. Can we use learned skills to directly maximize the task reward?

We take the skill with highest reward for each benchmark task and further finetune this skill using the task-specific reward function. We compare to a "random initialization" baseline that is initialized from scratch. Our approach differs from this baseline only in how weights are initialized. We initialize both the policy and value networks with weights learned during unsupervised pretraining. Although the critic networks learned during pretraining corresponds to the pseudo-reward from the discriminator (Eq. 3) and not the true task reward, we found empirically that the pseudo-reward was close to the true task reward for the best skill, and initializing the critic in addition to the actor further sped up learning. Figure 5 shows both methods applied to half cheetah, hopper, and ant. We assume that the unsupervised pretraining is free (e.g., only the reward function is expensive to compute) or can be amortized across many tasks, so we omit pretraining steps from this plot. On all tasks, unsupervised pretraining enables the agent to learn the benchmark task more quickly.

# 4.2.2 USING SKILLS FOR HIERARCHICAL RL

In theory, hierarchical RL should decompose a complex task into motion primitives, which may be reused for multiple tasks. In practice, algorithms for hierarchical RL can encounter many problems: (1) each motion primitive reduces to a single action (Bacon et al., 2017), (2) the hierarchical policy only samples a single motion primitive (Gregor et al., 2016), or (3) all motion primitives attempt to do the entire task. In contrast, DIAYN discovers diverse, task-agnostic skills, which hold the promise of acting as a building block for hierarchical RL.

Question 6. Are skills discovered by DIAYN useful for hierarchical RL?

We propose a simple extension to DIAYN for hierarchical RL, and find that simple algorithm outperforms competitive baselines on two challenging tasks. To use the discovered skills for hierarchical RL, we learn a meta-controller whose actions are to choose which skill to execute for the next  $k$  steps (100 for ant navigation, 10 for cheetah hurdle). The meta-controller has the same observation space as the skills.

As an initial test, we applied the hierarchical RL algorithm to a simple 2D point navigation task (details in Appendix C.2). Figure 6 illustrates how the reward on this task increases with the number of skills; error bars show the standard deviation across 5 random seeds. To ensure that our goals were not cherry picked, we sampled 25 goals evenly from the state space, and evaluated

![](images/ad5aa5f4de0cb1625eea8f76e6ee76e57ab38c0443b9260f4faf0c640f41a7da.jpg)  
Figure 6: Hierarchical RL

each random seed on all goals. We also compared to VIME (Houthooft et al., 2016). Note that even the best random seed from VIME significantly under-performs DIAYN. This is not surprising: whereas DIAYN explicitly skills that effectively partition the state space, VIME attempts to learn a single policy that visits many states.

Next, we applied the hierarchical algorithm to two challenging simulated robotics environment. On the cheetah hurdle task, the agent is rewarded for bounding up and over hurdles, while in the ant navigation task, the agent must walk to a set of 5 waypoints in a specific order, receiving only a sparse reward upon reaching each waypoint.

![](images/285ae7cfff150c7227088459127bb55d81459485b662d286c8153d98e4a76f00.jpg)  
Cheetah Hurdle

![](images/94caa0d57504ddb4bd02c2f5502e6b9e3c97a55fbe9e1e9d9edafc28a689d9ee.jpg)  
Ant Navigation

The sparse reward and obstacles in these environments make them exceeding difficult for non-hierarchical RL algorithms. Indeed, state of the art RL algorithms that do not use hierarchies perform poorly on these tasks. Figure 8 shows how DIAYN outperforms state of the art on-policy RL (TRPO (Schulman et al., 2015a)), off-policy RL (SAC (Haarnoja et al., 2018)), and exploration bonuses (VIME (Houthooft et al., 2016)). This experiment suggests that unsupervised skill learning provides an effective mechanism for combating challenges of exploration and sparse rewards in RL.

![](images/efa3029042e0e9f3b3a3e30ac596e6021a6ba0d9737a4095a61f4a38ee841bd3.jpg)  
Figure 8: DIAYN for Hierarchical RL: By learning a meta-controller to compose skills learned by DIAYN, cheetah quickly learns to jump over hurdles and ant solves a sparse-reward navigation task.

![](images/3325bef0bf3fe9ee2db8410ea7180dee69a97eb7759b3ab1ad8003257a441413.jpg)

Question 7. How can DIAYN leverage prior knowledge about what skills will be useful?

If the number of possible skills grows exponentially with the dimension of the task observation, one might imagine that DIAYN would fail to learn skills necessary to solve some tasks. While we found that DIAYN does scale to tasks with more than 100 dimensions (ant has 111), we can also use a simple modification to bias DIAYN towards discovering particular types of skills. We can condition the discriminator on only a subset of the observation space, or any other function of the observations.

![](images/4f45479d647ead7b1517dd1172cb368886e33c6221d25768cb30896eaad05f4b.jpg)  
Figure 9: Imitating an expert: DIAYN imitates an expert standing upright, flipping, and faceplanting, but fails to imitate a handstand.

In this case, the discriminator maximizes  $\mathbb{E}[\log q_{\phi}(z\mid f(s))]$ . For example, in the ant navigation task,  $f(s)$  could compute the agent's center of mass, and DIAYN would learn skills that correspond to changing the center of mass. The "DIAYN+prior" result in Figure 8 (right) shows how incorporating this prior knowledge can aid DIAYN in discovering useful skills and boost performance on the hierarchical task. (No other experiments or figures in this paper used this prior.) The key takeaway is that while DIAYN is primarily an unsupervised RL algorithm, there is a simple mechanism for incorporating supervision when it is available. Unsurprisingly, we perform better on hierarchical tasks when incorporating more supervision.

# 4.2.3 IMITATING AN EXPERT

# Question 8. Can we use learned skills to imitate an expert?

Aside from maximizing reward with finetuning and hierarchical RL, we can also use learned skills to follow expert demonstrations. One use-case is where a human manually controls the agent to complete a task that we would like to automate. Simply replaying the human's actions fails in stochastic environments, cases where closed-loop control is necessary. A second use-case involves an existing agent with a hard coded, manually designed policy. Imitation learning replaces the existing policy with a similar yet differentiable policy, which might be easier to update in response to new constraints or objectives. We consider the setting where we are given an expert trajectory consisting of states, without actions, defined as  $\tau^{*} = \{(s_{i})\}_{1\leq i\leq N}$ . Our goal is to obtain a feedback controller that will reach the same states. Given the expert trajectory, we use our learned discriminator to estimate which skill was most likely to have generated the trajectory. This optimization problem, which we solve for categorical  $z$  by enumeration, is equivalent to an M-projection (Bishop, 2016):

$$
\hat {z} = \underset {z} {\arg \max} \Pi_ {s _ {t} \in \tau^ {*}} q _ {\phi} (z \mid s _ {t})
$$

We qualitatively evaluate this approach to imitation learning on half cheetah. Figure 9 (left) shows four imitation tasks, three of which our method successfully imitates. We quantitatively evaluate this imitation method on classic control tasks in Appendix G.

# 5 CONCLUSION

In this paper, we present DIAYN, a method for learning skills without reward functions. We show that DIAYN learns diverse skills for complex tasks, often solving benchmark tasks with one of the learned skills without actually receiving any task reward. We further proposed methods for using the learned skills (1) to quickly adapt to a new task, (2) to solve complex tasks via hierarchical RL, and (3) to imitate an expert. As a rule of thumb, DIAYN may make learning a task easier by replacing the task's complex action space with a set of useful skills. DIAYN could be combined with methods for augmenting the observation space and reward function. Using the common language of information theory, a joint objective can likely be derived. DIAYN may also more efficiently learn from human preferences by having humans select among learned skills. Finally, the skills produced by DIAYN might be used by game designers to allow players to control complex robots and by artists to animate characters.

# REFERENCES

Joshua Achiam, Harrison Edwards, Dario Amodei, and Pieter Abbeel. Variational autoencoding learning of options by reinforcement. NIPS Deep Reinforcement Learning Symposium, 2017.  
David Barber Felix Agakov. The im algorithm: a variational approach to information maximization. Advances in Neural Information Processing Systems, 16:201, 2004.  
Pierre-Luc Bacon, Jean Harb, and Doina Precup. The option-critic architecture. In AAAI, pp. 1726-1734, 2017.  
Adrien Baranes and Pierre-Yves Oudeyer. Active learning of inverse models with intrinsically motivated goal exploration in robots. Robotics and Autonomous Systems, 61(1):49-73, 2013.  
Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. In Advances in Neural Information Processing Systems, pp. 1471-1479, 2016.  
Christopher M Bishop. Pattern Recognition and Machine Learning. Springer-Verlag New York, 2016.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, pp. 4302-4310, 2017.  
Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. In Advances in neural information processing systems, pp. 271-278, 1993.  
Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International Conference on Machine Learning, pp. 1329-1338, 2016.  
Carlos Florensa, Yan Duan, and Pieter Abbeel. Stochastic neural networks for hierarchical reinforcement learning. arXiv preprint arXiv:1704.03012, 2017.  
Kevin Frans, Jonathan Ho, Xi Chen, Pieter Abbeel, and John Schulman. Meta learning shared hierarchies. arXiv preprint arXiv:1710.09767, 2017.  
Justin Fu, John Co-Reyes, and Sergey Levine. Ex2: Exploration with exemplar models for deep reinforcement learning. In Advances in Neural Information Processing Systems, pp. 2574-2584, 2017.  
Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. arXiv preprint arXiv:1611.07507, 2016.  
Shixiang Gu, Ethan Holly, Timothy Lillicrap, and Sergey Levine. Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In Robotics and Automation (ICRA), 2017 IEEE International Conference on, pp. 3389-3396. IEEE, 2017.  
Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. arXiv preprint arXiv:1702.08165, 2017.  
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv preprint arXiv:1801.01290, 2018.  
Dylan Hadfield-Menell, Smitha Milli, Pieter Abbeel, Stuart J Russell, and Anca Dragan. In Advances in Neural Information Processing Systems, pp. 6768-6777, 2017.  
Karol Hausman, Jost Tobias Springenberg, Ziyu Wang, Nicolas Heess, and Martin Riedmiller. Learning an embedding space for transferable robot skills. International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rk07ZXZRb.  
Nicolas Heess, Greg Wayne, Yuval Tassa, Timothy Lillicrap, Martin Riedmiller, and David Silver. Learning and transfer of modulated locomotor controllers. arXiv preprint arXiv:1610.05182, 2016.  
Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. arXiv preprint arXiv:1709.06560, 2017.  
Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Vime: Variational information maximizing exploration. In Advances in Neural Information Processing Systems, pp. 1109-1117, 2016.

Tobias Jung, Daniel Polani, and Peter Stone. Empowerment for continuous agent—environment systems. Adaptive Behavior, 19(1):16-39, 2011.  
Sanjay Krishnan, Roy Fox, Ion Stoica, and Ken Goldberg. Ddco: Discovery of deep continuous options for robot learning from demonstrations. In Conference on Robot Learning, pp. 418-437, 2017.  
Joel Lehman and Kenneth O Stanley. Abandoning objectives: Evolution through the search for novelty alone. Evolutionary computation, 19(2):189-223, 2011a.  
Joel Lehman and Kenneth O Stanley. Evolving a diversity of virtual creatures through novelty search and local competition. In Proceedings of the 13th annual conference on Genetic and evolutionary computation, pp. 211-218. ACM, 2011b.  
Robert K Merton. The matthew effect in science: The reward and communication systems of science are considered. Science, 159(3810):56-63, 1968.  
Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andrew J Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, et al. Learning to navigate in complex environments. arXiv preprint arXiv:1611.03673, 2016.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.  
Shakir Mohamed and Danilo Jimenez Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. In Advances in neural information processing systems, pp. 2125-2133, 2015.  
Jean-Baptiste Mouret and Stéphane Doncieux. Overcoming the bootstrap problem in evolutionary robotics using behavioral diversity. In Evolutionary Computation, 2009. CEC'09. IEEE Congress on, pp. 1161-1168. IEEE, 2009.  
Kevin P Murphy. Machine Learning: A Probabilistic Perspective. MIT Press, 2012.  
Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans. Bridging the gap between value and policy based reinforcement learning. In Advances in Neural Information Processing Systems, pp. 2772-2782, 2017.  
Pierre-Yves Oudeyer, Frdric Kaplan, and Verena V Hafner. Intrinsic motivation systems for autonomous mental development. IEEE transactions on evolutionary computation, 11(2):265-286, 2007.  
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. arXiv preprint arXiv:1705.05363, 2017.  
Vitchyr Pong, Shixiang Gu, Murtaza Dalal, and Sergey Levine. Temporal difference models: Model-free deep rl for model-based control. arXiv preprint arXiv:1802.09081, 2018.  
Justin K Pugh, Lisa B Soros, and Kenneth O Stanley. Quality diversity: A new frontier for evolutionary computation. Frontiers in Robotics and AI, 3:40, 2016.  
Richard M Ryan and Edward L Deci. Intrinsic and extrinsic motivations: Classic definitions and new directions. Contemporary educational psychology, 25(1):54-67, 2000.  
Jürgen Schmidhuber. Formal theory of creativity, fun, and intrinsic motivation. IEEE Transactions on Autonomous Mental Development, 2(3):230-247, 2010.  
John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning, pp. 1889-1897, 2015a.  
John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015b.  
John Schulman, Pieter Abbeel, and Xi Chen. Equivalence between policy gradients and soft q-learning. arXiv preprint arXiv:1704.06440, 2017.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.  
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484-489, 2016.

Kenneth O Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2):99-127, 2002.  
Felipe Petroski Such, Vashisht Madhavan, Edoardo Conti, Joel Lehman, Kenneth O Stanley, and Jeff Clune. Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. arXiv preprint arXiv:1712.06567, 2017.  
Sainbayar Sukhbaatar, Zeming Lin, Ilya Kostrikov, Gabriel Synnaeve, Arthur Szlam, and Rob Fergus. Intrinsic motivation and automatic curricula via asymmetric self-play. arXiv preprint arXiv:1703.05407, 2017.  
Brian G Woolley and Kenneth O Stanley. On the deleterious effects of a priori objectives on evolution and representation. In Proceedings of the 13th annual conference on Genetic and evolutionary computation, pp. 957-964. ACM, 2011.  
Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi. Target-driven visual navigation in indoor scenes using deep reinforcement learning. In Robotics and Automation (ICRA), 2017 IEEE International Conference on, pp. 3357-3364. IEEE, 2017.  
Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In AAAI, volume 8, pp. 1433-1438. Chicago, IL, USA, 2008.
