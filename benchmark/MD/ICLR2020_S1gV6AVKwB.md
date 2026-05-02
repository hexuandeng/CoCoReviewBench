# CROSS DOMAIN IMITATION LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the question of how to imitate tasks across domains with discrepancies such as embodiment and viewpoint mismatch. Many prior works require paired, aligned demonstrations and an additional RL procedure for the task. However, paired, aligned demonstrations are seldom obtainable and RL procedures are expensive. In this work, we formalize the Cross Domain Imitation Learning (CDIL) problem, which encompasses imitation learning in the presence of viewpoint and embodiment mismatch. Informally, CDIL is the process of learning how to perform a task optimally, given demonstrations of the task in a distinct domain. We propose a two step approach to CDIL: alignment followed by adaptation. In the alignment step we execute a novel unsupervised MDP alignment algorithm, Generative Adversarial MDP Alignment (GAMA), to learn state and action correspondences from unpaired, unaligned demonstrations. In the adaptation step we leverage the correspondences to zero-shot imitate tasks across domains. To describe when CDIL is feasible via alignment and adaptation, we introduce a theory of MDP alignability. We experimentally evaluate GAMA against baselines in both embodiment and viewpoint mismatch scenarios where aligned demonstrations don't exist and show the effectiveness of our approach.

# 1 INTRODUCTION

Humans possess an astonishing ability to recognize latent structural similarities between behaviors in related but distinct domains, and learn new skills from cross domain demonstrations alone. Not only are we capable of learning from third person observations that have no obvious correspondence to our internal self representations (Stadie et al., 2017; Liu et al., 2018; Sermanet et al., 2018), but we also are capable of imitating agents with different embodiments (Gupta et al., 2017; Rizzolatti & Craighero, 2004) as can be observed in an infant's learning of visuomotor skills from adults with different biomechanics and physical capabilities (Jones, 2009). Previous work in neuroscience (Marshall & Meltzoff, 2015) and robotics (Kuniyoshi & Inoue, 1993; Kuniyoshi et al., 1994) have recognized the pitfalls of exact behavioral cloning in the presence of domain discrepancies and posited that the effectiveness of the human imitation learning mechanism hinges, crucially, on the capability to learn structure preserving domain correspondences. These correspondences enable the learner to internalize the expert demonstrations and produce a reconstruction of the behavior in the self domain. Consider a young child that has learned to associate his internal body map with the limbs of an adult. When the adult demonstrates running, the child is able to imagine himself running, and reproduce the behavior.

Recently, separate solutions have been proposed for imitation learning across two kinds of domain discrepancies: embodiment (Gupta et al., 2017) and viewpoint (Liu et al., 2018; Sermanet et al., 2018) mismatch. These works (Liu et al., 2018; Sermanet et al., 2018; Gupta et al., 2017) require paired, time-aligned demonstrations to obtain state correspondences and an extra RL step with a proxy reward. However, paired, aligned demonstrations are seldom obtainable and RL loops are expensive. In this work we formalize the Cross Domain Imitation Learning (CDIL) problem which encompasses prior work in imitation learning across domains with viewpoint and embodiment mismatch. Informally, CDIL is the process of learning how to perform a task optimally in a self domain, given demonstrations of the task in a distinct expert domain. We propose a two-step approach to CDIL: alignment followed by adaptation. In the alignment step we execute a novel unsupervised MDP alignment algorithm, Generative Adversarial MDP Alignment (GAMA), to learn state, action maps from unpaired, unaligned demonstrations. In the adaptation step we leverage the learned state, action maps to zero-shot imitate tasks across domains without an additional RL step. To shed light on when CDIL can be solved by alignment and adaptation, we first introduce a class of structure preserving maps, called MDP reductions, that adapts optimal policies between MDPs (section 3). We further characterize a family of MDP pairs that share reductions, formally state the MDP alignment problem, and elucidate its connection to CDIL. In section 4, 5 we derive GAMA,

![](images/b7f79114a9c69182e34daa5a7e88f3ce2b5620a081094ae182ba5e7d575f7922.jpg)  
Figure 1: (a). Illustration of paired, aligned vs unpaired, unaligned demonstrations in the alignment task set  $\mathcal{D}_{x,y}$  (b). Alignment: we learn state, action maps  $f,g$  between the self  $(x)$  and expert  $(y)$  domain from unpaired, unaligned demonstrations by minimizing a distribution matching loss and an imitation loss. (c) Adaptation: adapt the expert domain policy  $\pi_{y,\mathcal{T}}$  or demonstrations to obtain a self domain policy  $\hat{\pi}_{x,\mathcal{T}}$  a simple training algorithm to learn MDP reductions. In section 6, we experimentally evaluate GAMA and find that meaningful state correspondences between various domains are learned from unpaired, unaligned demonstrations. We then compare the CDIL performance of GAMA against several baselines in both embodiment and viewpoint mismatch scenarios and show the effectiveness of our approach.

# 2 CROSS DOMAIN IMITATION LEARNING PROBLEM STATEMENT

An infinite horizon Markov Decision Process (MDP)  $\mathcal{M} \in \Omega$  with deterministic dynamics is a tuple  $(S, \mathcal{A}, P, \eta, R)$  where  $\Omega$  is the set of all MDPs,  $\mathcal{S}$  is the state space,  $\mathcal{A}$  is the action space,  $P: S \times \mathcal{A} \to S$  is a (deterministic) transition function,  $R: S \times \mathcal{A} \to \mathbb{R}$  is the reward function, and  $\eta$  is the initial state distribution. A domain is an MDP without the reward, i.e  $(S, \mathcal{A}, P, \eta)$ . Intuitively, a domain fully characterizes the embodied agent and the environment dynamics, but not the desired behavior. A task  $\mathcal{T}$  is a label for an MDP corresponding to the high level description of optimal behavior, such as "walking".  $\mathcal{T}$  is analogous to category labels for images. An MDP with domain  $x$  for task  $\mathcal{T}$  is denoted by  $\mathcal{M}_{x,\mathcal{T}} = (\mathcal{S}_x, \mathcal{A}_x, P_x, \eta_x, R_{x,\mathcal{T}})$ , where  $R_{x,\mathcal{T}}$  is a reward function encapsulating the behavior labeled by  $\mathcal{T}$ . For example, different reward functions are needed to realize the "walking" behavior in two morphologically different humanoids. A (stationary) policy for  $\mathcal{M}_{x,\mathcal{T}}: S_x \to \mathcal{B}(\mathcal{A}_x)$  where  $\mathcal{B}$  is the set of probability measures on  $\mathcal{A}_x$  and an optimal policy  $\pi_{x,\mathcal{T}}^* = \arg\max_{\pi_x} J(\pi_x)$  achieves the highest policy performance  $J(\pi_x) = \mathbb{E}_{\pi_x}[\sum_{t=0}^{\infty} \gamma^t R_{x,\mathcal{T}}(s_x^{(t)}, a_x^{(t)})]$  where  $0 < \gamma < 1$  is a discount factor. A demonstration of length  $H$  is a sequence of state, action tuples  $\tau_{\mathcal{M}_{x,\mathcal{T}}} = \{(s_x^{(t)}, a_x^{(t)})\}_{t=1}^H$  sampled from an optimal policy and  $\mathcal{D}_{\mathcal{M}_{x,\mathcal{T}}} = \{\tau_{\mathcal{M}_{x,\mathcal{T}}}^{(k)}\}_{k=1}^K$  is a set of demonstrations for  $\mathcal{M}_{x,\mathcal{T}}$

Let  $\mathcal{M}_{x,\mathcal{T}},\mathcal{M}_{y,\mathcal{T}}$  be self and expert MDPs for a target task  $\mathcal{T}$ . Given expert domain demonstrations  $\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}}}$ , Cross Domain Imitation Learning (CDIL) aims to determine an optimal self domain policy  $\pi_{x,\mathcal{T}}^{*}$  without access to the reward function  $R_{x,\mathcal{T}}$ . In this work we propose to first solve an MDP alignment problem and then leverage the alignments to zero-shot imitate expert domain demonstrations. Like prior work (Gupta et al., 2017; Liu et al., 2018; Sermanet et al., 2018), we assume the availability of an alignment task set  $\mathcal{D}_{x,y} = \left\{(\mathcal{D}_{\mathcal{M}_{x,\mathcal{T}_i}},\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}_i}})\right\}_{i=1}^N$  containing demonstrations for  $N$  tasks  $\{\mathcal{T}_i\}_{i=1}^N$  from both the self and expert domain.  $\mathcal{D}_{x,y}$  could, for example, contain both robot  $(x)$  and human  $(y)$  demonstrations for a set primitive tasks such as walking, running, and jumping. Unlike prior work, demonstrations are unpaired and unaligned, i.e.  $(s_x^{(t)},s_y^{(t)})$  may not be a valid state correspondence. (see Figure 1(a)) Paired, time-aligned cross domain data is expensive and may not even exist when task execution rates differ or there exists systematic embodiment mismatch between the domains. For example, a child can imitate an adult running, but not achieve the same speed. Our set up emulates a natural setting in which humans compare how they perform tasks to how other agents perform the same tasks in order to find structural similarities and identify domain correspondences. We now proceed to introduce a theoretical framework that explains how and when the CDIL problem can be solved by MDP alignment followed by adaptation.

# 3 ALIGNNABLE MDPS

Let  $\Pi_{\mathcal{M}}^{*}$  be the set of all optimal policies for MDP  $\mathcal{M}$ . We define an occupancy measure (Syed et al., 2008)  $q_{\pi}:\mathcal{S}\times \mathcal{A}\to \mathbb{R}$  for policy  $\pi$  as  $q_{\pi}(s,a) = \pi (a|s)\sum_{t = 0}^{\infty}\gamma^{t}\operatorname *{Pr}(s^{(t)} = s;\pi ,P,\eta)$ .  
Definition 1. An optimality function  $O_{\mathcal{M}_x} : \mathcal{S}_x \times \mathcal{A}_x \to \{0,1\}$  for an MDP  $\mathcal{M}_x$  satisfies:  $O_{\mathcal{M}_x}(s_x, a_x) = 1$  if  $\exists \pi_x^* \in \Pi_{\mathcal{M}_x}^*$  such that  $(s_x, a_x) \in \mathrm{supp}(q_{\pi_x^*})$  and  $O_{\mathcal{M}_x}(s_x, a_x) = 0$  otherwise.

Definition 2. An MDP reduction from  $\mathcal{M}_x = (\mathcal{S}_x, \mathcal{A}_x, P_x, \eta_x, R_x)$  to  $\mathcal{M}_y = (\mathcal{S}_y, \mathcal{A}_y, P_y, \eta_y, R_y)$  is a tuple  $r = (\phi, \psi)$  where  $\phi : \mathcal{S}_x \to \mathcal{S}_y$ ,  $\psi : \mathcal{A}_x \to \mathcal{A}_y$  are maps that preserve:

1. (optimal policy)  $\forall s_x\in S_x,a_x\in \mathcal{A}_x,s_y\in S_y,a_y\in \mathcal{A}_y,$

$$
O _ {\mathcal {M} _ {y}} \left(\phi \left(s _ {x}\right), \psi \left(a _ {x}\right)\right) = 1 \quad \Rightarrow \quad O _ {\mathcal {M} _ {x}} \left(s _ {x}, a _ {x}\right) = 1 \tag {1}
$$

$$
O _ {\mathcal {M} _ {y}} \left(s _ {y}, a _ {y}\right) = 1 \quad \Rightarrow \quad \phi^ {- 1} \left(s _ {y}\right) \neq \emptyset , \psi^ {- 1} \left(a _ {y}\right) \neq \emptyset \tag {2}
$$

2. (dynamics)  $\forall s_y, s_y' \in S_y$ ,  $a_y \in \mathcal{A}_y$  where  $O_{\mathcal{M}_y}(s_y, a_y) = 1$ ,

$$
P _ {y} \left(s _ {y}, a _ {y}\right) = \phi \left(P _ {x} \left(s _ {x}, a _ {x}\right)\right) \quad \forall s _ {x} \in \phi^ {- 1} \left(s _ {y}\right), a _ {x} \in \psi^ {- 1} \left(a _ {y}\right) \tag {3}
$$

where we define  $\phi^{-1}(s_y) = \{s_x|\phi (s_x) = s_y\}$ ,  $\psi^{-1}(a_y) = \{a_x|\psi (a_x) = a_y\}$ . Furthermore,  $r$  is an MDP permutation if and only if  $\phi, \psi$  are bijective maps.

In words, Eq. 1 states that only optimal state, action pairs in  $x$  can be mapped to optimal state, action pairs in  $y$  and Eq. 2 states that  $r$  must be surjective on the set of optimal state, action pairs in  $y$ . Eq. 3 states that a reduction must preserve (deterministic) dynamics. We use the notation  $\mathcal{M}_x \geq_{\phi, \psi} \mathcal{M}_y$  to denote that  $(\phi, \psi)$  is a reduction from  $\mathcal{M}_x$  to  $\mathcal{M}_y$ , and the shorthand  $\mathcal{M}_x \geq \mathcal{M}_y$  to denote that  $\mathcal{M}_x$  reduces to  $\mathcal{M}_y$ . To gain an intuitive understanding of MDP reductions, picture the execution trace of an optimal policy as a directed

![](images/9c02b533a6aaf0534143a7e7875d23b742bbcab04714e666405088933e04852b.jpg)  
Figure 2: Example MDP reduction from  $\mathcal{M}_x$  to  $\mathcal{M}_y$ .  $\phi, \psi$  are state and action maps

graph with colored edges in which the nodes correspond to states visited by an optimal policy, and the colored edges correspond to actions taken. An MDP reduction from  $\mathcal{M}_x$  to  $\mathcal{M}_y$  homomorphs the execution graph of an optimal policy in  $\mathcal{M}_x$  to a execution graph of an optimal policy in  $\mathcal{M}_y$ . Figure 2 shows an example of a valid reduction from  $\mathcal{M}_x$  to  $\mathcal{M}_y$ : states  $1, 2$  in  $S_x$  are mapped (merged) to state  $a$  in  $S_y$  and the blue, green actions in  $\mathcal{A}_x$  are mapped to the brown action in  $\mathcal{A}_y$ . Intuitively, if  $\mathcal{M}_x \geq_{\phi, \psi} \mathcal{M}_y$ , then  $(\phi, \psi)$  compresses  $\mathcal{M}_x$  by merging all optimal state, action pairs that have identical dynamics properties.

Definition 3. Two MDPs  $\mathcal{M}_x, \mathcal{M}_y$  are alignable if and only if  $\mathcal{M}_x \geq \mathcal{M}_y$  or  $\mathcal{M}_y \geq \mathcal{M}_x$ .

Definition 3 states that MDPs are alignable if reductions exist between them, meaning that they share structure. We use  $\Gamma(\mathcal{M}_x, \mathcal{M}_y) = \{(\phi, \psi) | \mathcal{M}_x \geq_{\phi, \psi} \mathcal{M}_y\}$  to denote the set of all valid reductions from  $\mathcal{M}_x$  to  $\mathcal{M}_y$ . Reductions have a particularly useful property which is that they adapt policies across alignable MDPs. Consider a state map  $f: S_x \to S_y$ , an inverse action map  $g: \mathcal{A}_y \to \mathcal{A}_x$ , and a composite policy  $\hat{\pi}_x = g \circ \pi_y \circ f$  (see Figure 1(b)). In words,  $\hat{\pi}_x$  maps a self state to an expert state via  $f$ , simulates the expert's action choice for the mapped state via  $\pi_y$ , then chooses a self action that corresponds to the simulated expert action with  $g$ . The following lemma holds for  $\hat{\pi}_x$ .

Lemma 1. Let  $\mathcal{M}_x, \mathcal{M}_y$  be MDPs satisfying Assumption 1 (see Supp. Materials),  $\mathcal{M}_x \geq_{\phi, \psi} \mathcal{M}_y$ , and  $\pi_y$  be optimal in  $\mathcal{M}_y$ .  $\forall g: \mathcal{A}_y \to \mathcal{A}_x$  s.t  $\psi \circ g(a_y) = a_y$ $\forall a_y \in \{a_y | \exists s_y \in \mathcal{S}_y$  s.t  $O_{\mathcal{M}_y}(s_y, a_y) = 1\}$ , it holds that  $\hat{\pi}_x = g \circ \pi_y \circ \phi$  is optimal in  $\mathcal{M}_x$ .

Lemma 1 states that the state, action maps  $(f,g^{-1})$  chosen to be a reduction can adapt optimal policies between alignable MDPs. Here onwards we interchangeably refer to  $(f,g)$  as "alignments". We now show how the CDIL problem can be solved by first solving an MDP alignment problem followed by an adaptation step.

Definition 4. Let  $(\mathcal{M}_x,\mathcal{M}_y),(\mathcal{M}_x',\mathcal{M}_y')\in \Omega^2$  be two MDP pairs. Then,  $(\mathcal{M}_x,\mathcal{M}_y)\sim (\mathcal{M}_x',\mathcal{M}_y')$ , i.e they are joint alignable, if and only if  $\Gamma (\mathcal{M}_x,\mathcal{M}_y)\cap \Gamma (\mathcal{M}_x',\mathcal{M}_y')\neq \emptyset$ .

In words, two MDP pairs are joint alignable if there exists a shared reduction. We define an equivalence class  $[(M_x, M_y)]_{\sim} = \{(\mathcal{M}_x', \mathcal{M}_y') \mid (\mathcal{M}_x', \mathcal{M}_y') \sim (\mathcal{M}_x, \mathcal{M}_y)\}$  of MDP pairs that share reductions. Overloading notation,  $\Gamma(\{(\mathcal{M}_x^i, \mathcal{M}_y^i)\}_{i=1}^N) = \{(\phi, \psi) \mid (\phi, \psi) \in \Gamma(\mathcal{M}_x^1, \mathcal{M}_x^1) \cap \dots \cap \Gamma(\mathcal{M}_x^N, \mathcal{M}_x^N)\}$ . We now formally state the MDP alignment problem: Let  $(\mathcal{M}_{x,\mathcal{T}}, \mathcal{M}_{y,\mathcal{T}})$  be an MDP pair for a target task  $\mathcal{T}$ . Given an alignment task set  $\mathcal{D}_{x,y} = \{(\mathcal{D}_{\mathcal{M}_{x,\mathcal{T}_i}}, \mathcal{D}_{\mathcal{M}_{y,\mathcal{T}_i}})\}_{i=1}^N$  comprising unpaired, unaligned demonstrations for MDP pairs  $\{(\mathcal{M}_{x,\mathcal{T}_i}, \mathcal{M}_{y,\mathcal{T}_i})\}_{i=1}^N \subseteq [\mathcal{M}_{x,\mathcal{T}}, \mathcal{M}_{y,\mathcal{T}}]_{\sim}$ , determine  $(\phi, \psi) \in \Gamma(\{(\mathcal{M}_{x,\mathcal{T}_i}, \mathcal{M}_{y,\mathcal{T}_i})\}_{i=1}^N)$  such that  $(\phi, \psi) \in \Gamma(\mathcal{M}_{x,\mathcal{T}}, \mathcal{M}_{y,\mathcal{T}})$ . As shown in Figure 3, with more MDP pairs, there are likely a smaller the number of joint alignments  $|\Gamma(\{(\mathcal{M}_{x,\mathcal{T}_i}, \mathcal{M}_{y,\mathcal{T}_i})\}_{i=1}^N)|$  and, as a result,  $(\phi, \psi) \in \Gamma(\{(\mathcal{M}_{x,\mathcal{T}_i}, \mathcal{M}_{y,\mathcal{T}_i})\}_{i=1}^N)$  is more likely to "generalize" to an MDP pair for a new target task  $(\mathcal{M}_{x,\mathcal{T}}, \mathcal{M}_{y,\mathcal{T}})$  in the equivalence class. Analogously, in a standard supervised learning problem, more training data is likely to shrink the set of models performing optimally on the training set but poorly on the test set.

We can then use  $(\phi, \psi)$  for CDIL: given cross domain demonstrations  $\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}}}$  for the target task  $\mathcal{T}$ , learn an expert domain policy  $\pi_{y,\mathcal{T}}$ , and adapt it into the self domain using  $(\phi, \psi)$  according to Lemma 1.

We can now assess when domains with embodiment and viewpoint mismatch have meaningful state correspondences, i.e MDP reductions, thus allowing for cross domain imitation. The states of a human expert with more degrees of freedom than a robot imitator can be merged into the robot states if the task only requires the robot's degrees of freedom and the execution traces share structure, e.g traces are both cycles. However, if the task requires all degrees of freedom possessed only by the human, the robot cannot find meaningful correspondences, and also cannot imitate the task. Two MDPs

for different viewpoints of an agent performing a task are MDP permutations since there is a one-to-one correspondence between state, actions at same timestep in the execution trace of an optimal policy.

![](images/43f0650803e8dad6526e976dab6c2015f6fff7ce97d41109e905101feac640a1.jpg)  
Figure 3: Illustration of MDP alignment problem

# 4 LEARNING MDP REDUCTIONS

We now derive objectives that can be optimized to learn MDP reductions. We propose distribution matching and policy performance maximization. We first define the distributions to be matched.

Definition 5. Let  $\mathcal{M}_x, \mathcal{M}_y$  be two MDPs and  $\hat{\pi}_x = g \circ \pi_y \circ f$  for  $f: S_x \to S_y$ ,  $g: \mathcal{A}_y \to \mathcal{A}_x$  and policy  $\pi_y$ .  $\mathcal{P} = \{\hat{s}_y^{(t)}, \hat{a}_y^{(t)}\}_{t \geq 0}$  is the co-domain policy execution process realized by running  $\hat{\pi}_x$ , i.e:

$$
s _ {x} ^ {(0)} \sim \eta_ {x}, \hat {s} _ {y} ^ {(t)} = f (s _ {x} ^ {(t)}), \hat {a} _ {y} ^ {(t)} \sim \pi_ {y} (\cdot | \hat {s} _ {y} ^ {(t)}), a _ {x} ^ {(t)} = g (\hat {a} _ {y} ^ {(t)}), s _ {x} ^ {(t + 1)} = P _ {x} (s _ {x} ^ {(t)}, a _ {x} ^ {(t)}) \quad \forall t \geq 0 \tag {4}
$$

The target distribution  $\sigma_{\pi_y}^y$  is over transitions uniformly sampled from execution traces of  $\pi_y$  and the proxy distribution  $\sigma_{\hat{\pi}_x}^{x\to y}$  is over cross domain transitions uniformly sampled from realizations of  $\mathcal{P}$ .

$$
\sigma_ {\pi_ {y}} ^ {y} \left(s _ {y}, a _ {y}, s _ {y} ^ {\prime}\right) = \lim  _ {T \rightarrow \infty} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \Pr \left(s _ {y} ^ {(t)} = s _ {y}, a _ {y} ^ {(t)} = a _ {y}, s _ {y} ^ {(t + 1)} = s _ {y} ^ {\prime}; \pi_ {y}, P _ {y}, \eta_ {y}\right) \tag {5}
$$

$$
\sigma_ {\hat {\pi} _ {x}} ^ {x \rightarrow y} \left(s _ {y}, a _ {y}, s _ {y} ^ {\prime}\right) = \lim  _ {T \rightarrow \infty} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \Pr \left(\hat {s} _ {y} ^ {(t)} = s _ {y}, \hat {a} _ {y} ^ {(t)} = a _ {y}, \hat {s} _ {y} ^ {(t + 1)} = s _ {y} ^ {\prime}; \mathcal {P}\right) \tag {6}
$$

We now propose three concrete objectives: 1.  $\hat{\pi}_x$  is optimal, 2.  $\sigma_{\hat{\pi}_x}^{x\to y} = \sigma_{\pi_y}^y$ , 3.  $g$  is injective. In other words, we seek to learn  $f, g$  that matches distributions over transition tuples in domain  $y$  while maximizing policy performance in domain  $x$ . The former captures the dynamics preservation property from Eq. 3 and the latter captures the optimal policy preservation property from Eq. 1, 2. The following theorem uncovers the connection between our objectives and MDP reductions.

Theorem 1. Let  $\mathcal{M}_x, \mathcal{M}_y$  be MDPs satisfying Assumption 1 (see Supp Materials). If  $\mathcal{M}_x \geq \mathcal{M}_y$ , then  $\exists f: S_x \to S_y, g: \mathcal{A}_y \to \mathcal{A}_x$ , and an optimal covering policy  $\pi_y$  (Supp Materials, Def 6) that satisfies objectives 1, 2. Conversely, if  $\exists f: S_x \to S_y, g: \mathcal{A}_y \to \mathcal{A}_x$  and an optimal covering policy  $\pi_y$  satisfying objectives 1, 2, 3, then  $\mathcal{M}_x \geq \mathcal{M}_y$  and  $\exists (\phi, \psi) \in \Gamma(\mathcal{M}_x, \mathcal{M}_y)$  s.t  $f = \phi$  and  $\psi \circ g(a_y) = a_y, \forall a_y \in \mathcal{A}_y$ .

Theorem 1 states that if two MDP are alignable, then objectives 1, 2 can be satisfied. Conversely, if objectives 1, 2, 3 can be satisfied for two MDPs, then they must be alignable and all solutions  $(f,g)$  are MDP reductions. While Theorem 1 requires alignable MDPs to guarantee identifiability, our experiments will also run on MDPs that are not perfectly alignable, i.e. Eq. 1, 2, 3 do not hold exactly, but intuitively share structure. In the next section, we propose a simple algorithm to learn MDP reductions.

# 5 GENERATIVE ADVERSARIAL MDP ALIGNMENT

Building on Theorem 1, we propose the following general form training objective for aligning MDPs:

$$
\min  _ {f, g} - J \left(\hat {\pi} _ {x}\right) + \lambda d \left(\sigma_ {\hat {\pi} _ {x}} ^ {x \rightarrow y}, \sigma_ {\pi_ {y}} ^ {y}\right) \tag {7}
$$

where  $J(\hat{\pi}_x)$  is the performance of  $\hat{\pi}_x$ ,  $d$  is a distance metric between distributions, and  $\lambda > 0$  is a Lagrange multiplier. In practice, we found that injectivity of  $g$  is unnecessary to enforce in continuous domains. We now present an instantiation of this framework: Generative Adversarial MDP Alignment (GAMA). Recall that we are given an alignment task set  $\mathcal{D}_{x,y} = \{(\mathcal{D}_{\mathcal{M}_{x,\mathcal{T}_i}},\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}_i}})\}_{i=1}^N$ . In the alignment step, we learn  $\pi_{y,\mathcal{T}_i}^*, \forall \mathcal{T}_i$  and parameterized state, action maps  $f_{\theta_f}: S_x \to S_y$ ,  $g_{\theta_g}: \mathcal{A}_y \to \mathcal{A}_x$  that compose  $\hat{\pi}_{x,\mathcal{T}_i} = g_{\theta_g} \circ \pi_{y,\mathcal{T}_i}^* \circ f_{\theta_f}$ . To match  $\sigma_{\hat{\pi}_x}^{x \to y}$ ,  $\sigma_{\pi_y}^y$ , we employ adversarial training (Goodfellow

et al., 2014) in which separate discriminators  $D_{\theta_D^i}$  per task are trained to distinguish between "real" transitions  $(s^y, a^y, s^{y'}) \sim \pi_{y,\mathcal{T}_i}^*$  and "fake" transitions  $(\hat{s}_y, \hat{a}_y, \hat{s}_y') \sim \hat{\pi}_{x,\mathcal{T}_i}$ , where  $\hat{s}_y = f_{\theta_f}(s_x), \hat{a}_y = \pi_y(\hat{s}_y), \hat{s}_y' = f_{\theta_f}(P_{\theta_P}^x(s_x, g(\hat{a}_y)))$ , and  $P_{\theta_P}^x$  is a fitted model of the  $x$  domain dynamics. (see Figure 1(b)) The generator, consisting of  $f_{\theta_f}, g_{\theta_g}$ , is trained to fool the discriminator while maximizing policy performance. The distribution matching gradients are back propagated through the learned dynamics,  $\pi_{y,\mathcal{T}_i}^*$  is learned by Imitation Learning (IL) on  $\mathcal{DM}_{y,\mathcal{T}_i}$ , and the policy performance objective on  $\hat{\pi}_{x,\mathcal{T}_i}$  is achieved by IL on  $\mathcal{DM}_{x,\mathcal{T}_i}$ . In this work we use behavioral cloning (Pomerleau, 1991) for IL. We thus seek to find a saddle point  $\{f,g\} \cup \{D_{\theta_D^i}\}_{i=1}^N$  of the following objective:

$$
\begin{array}{l} \min  _ {f, g} \max  _ {\left\{D _ {\theta_ {D} ^ {i}} \right\} _ {i = 1} ^ {N}} \sum_ {i = 1} ^ {N} \left(\mathbb {E} _ {s _ {x} \sim \pi_ {x, \mathcal {T} _ {i}} ^ {*}} \left[ D _ {K L} \left(\pi_ {x, \mathcal {T} _ {i}} ^ {*} (\cdot | s _ {x}) \right| | \hat {\pi} _ {x, \mathcal {T} _ {i}} (\cdot | s _ {x})\right) \right] \tag {8} \\ + \lambda (\mathbb {E} _ {\pi_ {y, \mathcal {T} _ {i}} ^ {*}} [ \log D _ {\theta_ {D} ^ {i}} (s _ {y}, a _ {y}, s _ {y} ^ {\prime}) ] + \mathbb {E} _ {\pi_ {x, \mathcal {T} _ {i}} ^ {*}} [ \log (1 - D _ {\theta_ {D} ^ {i}} (\hat {s} _ {y}, \hat {a} _ {y}, \hat {s} _ {y} ^ {\prime})) ]) \\ \end{array}
$$

where  $D_{KL}$  is the KL-divergence. We provide the full execution flow of GAMA in Algorithm 1 In the adaptation step, we are given expert demonstrations  $\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}}}$  of a new target task  $\mathcal{T}$ , from which we fit an expert domain policy  $\pi_{y,\mathcal{T}}$  which are composed with the learned alignments to construct an adapted self policy  $\hat{\pi}_{x,\mathcal{T}} = g_{\theta_g} \circ \pi_{y,\mathcal{T}} \circ f_{\theta_f}$ . We also experiment with a demonstration adaptation method which additionally trains an inverse state map  $f^{-1}: S_y \to S_x$ , adapts demonstrations  $\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}}}$  into the self domain via  $f^{-1}, g$ , and applies behavioral cloning on the adapted demonstrations. (see Figure 1(c)) Notably, our entire procedure does not require paired, aligned demonstrations nor an RL step.

Algorithm 1: Generative Adversarial MDP Alignment (GAMA)  
input: Alignment task set  $\mathcal{D}_{x,y} = \{(\mathcal{D}_{\mathcal{M}_{x,\mathcal{T}_i}},\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}_i}})\}_{i = 1}^N$  of unpaired trajectories, fitted  $\pi_{y,\mathcal{T}_i}^*$    
while not done do:   
for  $i = 1,\dots ,N$  do:   
Sample  $(s_x,a_x,s_x')\sim \mathcal{D}_{\mathcal{M}_x,\mathcal{T}_i},(s_y,a_y,s_y')\sim \mathcal{D}_{\mathcal{M}_y,\mathcal{T}_i}$  and store in buffer  $\mathcal{B}_x^i,\mathcal{B}_y^i$    
for  $j = 1,\ldots ,M$  do:   
Sample mini-batch  $j$  from  $\mathcal{B}_x^i,\mathcal{B}_y^i$    
Update dynamics model with:  $-\hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_{P}}(P_{\theta_{P}}^{x}(s_{x},a_{x}) - s_{x}^{\prime})^{2}]$    
Update discriminator:  $\hat{\mathbb{E}}_{\pi^{*}_{y,\mathcal{T}_{i}}}[\nabla_{\theta_{D}^{i}}\log D_{\theta_{D}^{i}}(s_{y},a_{y},s_{y}^{\prime})] + \hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_{D}^{i}}\log (1 - D_{\theta_{D}^{i}}(\hat{s}_{y},\hat{a}_{y},\hat{s}_{y}^{\prime}))]$    
Update alignments  $(f_{\theta_f},g_{\theta_g})$  with gradients:  $-\hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_{f}}\log D_{\theta_{D}}(\hat{s}_{y},\hat{a}_{y},\hat{s}_{y}^{\prime})] + \hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_{f}}(\hat{\pi}_{x,\mathcal{T}_{i}}(s_{x}) - a_{x})^{2}]$ $-\hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_g}\log D_{\theta_D}(\hat{s}_y,\hat{a}_y,\hat{s}_y^{\prime})] + \hat{\mathbb{E}}_{\pi^{*}_{x,\mathcal{T}_{i}}}[\nabla_{\theta_g}(\hat{\pi}_{x,\mathcal{T}_i}(s_x) - a_x)^2 ]$

Related Works: Closely related to CDIL, the field of cross domain transfer learning in the context of RL has explored approaches to use state maps to exploit cross domain demonstrations in a pretraining procedure for a new target task for which self domain reward function is available. Canonical Correlation Analysis (CCA) (Hotelling, 1936) finds invertible projections into a basis in which data from different domains are maximally correlated. These projections can then be composed to obtain a direct correspondence map between states. Ammar et al. (2015); Joshi & Chowdhary (2018) have utilized an unsupervised manifold alignment (UMA) algorithm which finds a linear map between states with similar local geometric properties. UMA assumes the existence of hand crafted features along with a distance metric between them. This family of work commonly uses a linear statemap to define a time-step wise transfer reward and executes an RL step on the new task. Similar to our work, these works use an alignment task set of unpaired, unaligned trajectories to compute the state map. Unlike these works, we learn maps that preserve MDP structure, use deep neural network state, action maps, and achieve zero-shot transfer to the new task without an RL step. More recent work in transfer learning across embodiment (Gupta et al., 2017) and viewpoint (Liu et al., 2018; Sermanet et al., 2018) mismatch obtain state correspondences from an alignment task set comprising paired, time-aligned demonstrations and use them to learn a state map or a state encoder to a domain invariant feature space. In contrast to this family of prior work, our approach learns both state, action maps from unpaired, unaligned demonstrations. Also, we remove the need for additional environment interactions and an expensive RL procedure on the target task by leveraging the action map for zero-shot imitation. Stadie et al. (2017) have shown promise in using domain confusion loss and generative adversarial imitation learning (Ho & Ermon, 2016) for learning across small viewpoint mismatch without an alignment task set, but fails in dealing with large viewpoint differences. Unlike Stadie et al. (2017), we leverage the alignment task set to succeed in imitating across

larger viewpoint mismatch and do not require an RL procedure. MDP homomorphisms (Ravindran & Barto, 2002) have been explored with the aim of compressing state, action spaces to facilitate planning. In similar vein, related works have proposed MDP similarity metrics based on bisimulation methods (Ferns et al., 2004) and boltzman machine reconstruction error (Ammar et al., 2014). While conceptually related to our MDP alignability theory, these works have not proposed scalable procedures to discover the homomorphisms and have not drawn connections to cross domain learning.

# 6 EXPERIMENTS

Ours experiments were designed to answer the following questions: (1). Can GAMA uncover MDP reductions? (2). Can the learned alignments  $(f_{\theta_f}, g_{\theta_g})$  be leveraged to succeed at CDIL? Note that we include experiments with MDP pairs that are not perfectly alignable, yet intuitively share structure, to show general applicability of GAMA for CDIL. We propose three metrics to evaluate the effectiveness of GAMA. First, alignment complexity which is the number of MDP pairs, i.e. number of tasks, in the alignment task set needed to learn alignments that enable zero-shot imitation, given ample cross domain demonstrations for the target tasks. Second, adaptation complexity which is the amount of cross domain demonstrations for the target tasks needed to successfully imitate tasks in the self domain without querying the target task reward function, given a sufficiently large alignment task set. Finally, transferability, which is the environment sample complexity on the target task when using the alignment procedure as weight initialization then running RL with the target task reward function. While we aim to learn optimal self policies without querying the self domain reward function, this metric measures the usefulness of the alignment step even when MDP pairs in the alignment task set are not in the equivalence class of the target MDP pair. We study two ablations of GAMA and compare against the following baselines:

GAMA - Policy Adapt (GAMA-PA): learns alignments by Algorithm 1, fits an expert policy  $\pi_y,\mathcal{T}$  to  $\mathcal{D}_{\mathcal{M}_{y,\mathcal{T}}}$  for a new target task  $\mathcal{T}$  and zero-shot adapts  $\pi_{y,\mathcal{T}}$  to the self domain via  $\hat{\pi}_{x,\mathcal{T}} = g_{\theta_g}\circ \pi_{y,\mathcal{T}}\circ f_{\theta_f}$ .

GAMA - Demonstration Adapt (GAMA-DA): trains  $f^{-1}$  in addition to Algorithm 1, adapts  $\mathcal{D}_{\mathcal{M}_y,\tau}$  into the self domain via  $(f^{-1},g)$ , and fits a self domain policy on the adapted demonstrations.

Self Demonstrations (Self-Demo): We behavioral clone on self domain demonstrations for the target task. This baseline provides an "upper bound" on the adapation complexity of CDIL.

Canonical Correlation Analysis (Hotelling, 1936) (CCA): finds invertible matrices  $C_x, C_y$  to a basis where domain data are maximally correlated from unpaired, unaligned demonstrations.

Unsupervised Manifold Alignment (Ammar et al., 2015) (UMA): finds a map between states that have similar local geometries from unpaired, unaligned demonstrations.

Invariant Features (Gupta et al., 2017) (IF): finds invertible projections onto a feature space given state pairings. Dynamic Time Warping (Muller, 2007) is used to obtain the pairings.

Imitation from Observation (Liu et al., 2018) (IfO): learns a statemap conditioned on a cross domain observation given state pairings. Dynamic Time Warping (Muller, 2007) is used to obtain the pairings.

Third Person Imitation Learning (Stadie et al., 2017) (TPIL): simultaneously learns a domain agnostic feature space and matches distributions in the feature space via GAIL (Ho & Ermon, 2016).

We experiment with environments which are extensions of OpenAI Gym (Brockman et al., 2016). pen, cart, reacher2, reacher3, reach2,tp, snake3, and snake4 denotes the pendulum, cartpole, 2-link reacher, 3-link reacher, third person 2-link reacher, 3-link snake, and 4-link snake environments, respectively. (self domain)  $\leftrightarrow$  (expert domain) specify an MDP pair in the alignment task set. Model architectures and environment details are further described in the Supp. Materials, section B, C, D.

# 6.1 MDP ALIGNMENT EVALUATION

Figure 4 visualizes the learned state map  $f_{\theta_f}$  for several MDP pairs. The pen  $\leftrightarrow$  pen alignment task (Figure 4, Top Left) and reach  $\leftrightarrow$  reach-tp task exemplify scenarios where two MDPs are permutations of each other. Similarly, the pen  $\leftrightarrow$  cart alignment task (Figure 4, Top Right) has a reduction that maps the pendulum's angle and angular velocity to those of the pole, as the cart's position and velocity are redundant state dimensions once an optimal policy has been learned. Table 1 presents quantitative evaluations of these simple alignment maps. For pen  $\leftrightarrow$  pen and reach2  $\leftrightarrow$  reach2-tp we record the average L2 loss between the learned statemap's outputs and the ground truth permutation map's outputs. As for pen  $\leftrightarrow$  cart, we do

![](images/ef5fb562a6276002395f2061211e158a02f219a160e36dfee7682d58b6258a95.jpg)  
Figure 4: Visualization of the learned state maps for pen  $\leftrightarrow$  pen (Top Left), pen  $\leftrightarrow$  cart (Top Right), snake  $4\leftrightarrow$  snake3 (Bottom Left), reach2  $\leftrightarrow$  reach3 (Bottom Right). GAMA is able to recover MDP reductions (Top Left/Right) and finds interpretable correspondences between domains that are not perfectly alignable, yet intuitively share structure (Bottom Left/Right). Baselines fail in most cases

Table 1: Quantitative evaluation of learned state maps. GAMA reliably finds MDP permutations while baselines incur  $10 \times$  larger deviation loss from the ground truth permutation map.  

<table><tr><td></td><td>GAMA (ours)</td><td>CCA</td><td>UMA</td><td>IF</td><td>IfO</td><td>Random</td></tr><tr><td>pen ↔ pen</td><td>0.057 ±0.017</td><td>0.72 ±0.25</td><td>&gt;100</td><td>2.50 ±1.08</td><td>2.24 ±0.82</td><td>&gt;100</td></tr><tr><td>pen ↔ cart</td><td>0.178 ±0.051</td><td>3.92 ±3.77</td><td>&gt;100</td><td>1.62 ±0.52</td><td>3.31 ±1.2</td><td>&gt;100</td></tr><tr><td>reach2 ↔ reach2-tp</td><td>0.092 ±0.043</td><td>10.14 ±5.31</td><td>&gt;100</td><td>12.41 ±3.12</td><td>5.12 ±2.41</td><td>&gt;100</td></tr></table>

the same on the dimensions that correspond to the angle and angular velocity of the pole. We see from both Figure 4 and Table 1 that GAMA is able to learn simple reductions while baselines fail to do so. The key reason behind this performance gap is that most baselines (Gupta et al., 2017; Liu et al., 2018) obtain state maps from time-aligned demonstration data. However, the considered alignment task set contains unaligned demonstrations with diverse starting states, up to  $2\mathrm{x}$  differences in demonstration lengths, and varying task execution rates. We see that GAMA also outperforms baselines that learn from unaligned demonstrations (Hotelling, 1936; Ammar et al., 2015) by learning maps that preserve MDP structure with more flexible neural network function approximators. For snake4  $\leftrightarrow$  snake3 and reach2  $\leftrightarrow$  reach3, the MDPs may not be perfectly alignable, yet intuitively share structure. From Figure 4 (Bottom Left) we see that GAMA identically matches two adjacent joint angles of snake4 to the two joint angles of snake3 and the periodicity of the snake's wiggle is preserved. On reacher2  $\leftrightarrow$  reacher3, we find that the central pivot angles are matched and further find correspondences between states that have similar extents of crouching.

# 6.2 CDIL PERFORMANCE

Wall2Corner (W2C): The self domain is reacher2 and the expert domain is reacher3. We use the robot's internal state, action representation. The alignment tasks are reaching for 12 goals near the room wall centers and the target tasks are reaching for 12 new goals at the room corners, maximally away from the wall goals. The significant difference between training and test goals makes generalization challenging.

![](images/21f7da1eafed066e274b4d5e431d304e19e45680d5d7a55b827b17c7e5811733.jpg)

![](images/51f53e9719f86d4a4d90f9050709734f332ab0d2c0a96ea1ddee87362899042e.jpg)

![](images/906a0a26566c4f8072eee6966875d002bb92aff67e26acf15e51fc75f879fe3a.jpg)

![](images/6223a1ada8590e7b14e6a3b030a9d6de46a1ddc714f5e77546e0e336774bec32.jpg)  
Figure 5: CDIL performance. Adaptation complexity (Left), alignment complexity (Middle), and transferability (Right) for W2C/R2W on the top/bottom rows, respectively. GAMA outperforms baselines in all metrics. Notably, adaptation complexity of GAMA is close to that of the self-demo baseline.

![](images/9b93582368791f997d54740a82bba81a1d0589601241701e79ba3c192e280115.jpg)

![](images/574299b6ec4a61966e75ccc5d08a8fbdfb7bb24710df65a0a30df7a658771a9d.jpg)

![](images/77c4f1c501a682cb29de353ded72994dee318dbaf5619087a3275241439b8a4a.jpg)

Reach2Write (R2W): The self domain is reacher2 and the expert domain is reacher2,tp that has a "third person" state space with a  $180^{\circ}$  camera angle offset. We use the robot's internal state, action representation. The alignment tasks are reaching for goals and the transfer task is writing letters as fast as possible. The transfer task differs from the alignment tasks in two key aspects: the end effector must draw a straight line from a letter's vertex to vertex and minimally slow down at the vertices.

Alignment complexity is shown in Figure 5 (Left). GAMA is able to learn alignments that enable zero-shot imitation on the target task, showing clear gains over a simple pretraining procedure on the self domain MDPs in the alignment task set. Other baselines require an additional RL step and cannot achieve zero-shot imitation. Figure 5 (Middle) shows the adaptation complexity. Notably, GAMA-DA (blue, dashed) produces adapted demonstrations of similar usefulness as self demonstrations (olive green). Other baselines fail to learn useful alignments from unpaired, unaligned demonstrations and as a result fails at CDIL. Finally, Figure 5 (Right) shows that the alignment step is useful as weight initialization to accelerate learning of the target task. GAMA (blue) attains optimal performance around  $7 \times$  faster than all baselines in the W2C experiment, while immediately attaining optimal performance on the R2W task. Baselines fail to learn the writing task as an inaccurate proxy reward function harms performance.

# 6.3 CDIL WITH VISUAL INPUTS

The non-visual environment experiments in the previous section demonstrate the limitations of the time-alignment assumptions made in prior work without confounding variables such as the difficulty optimization in high-dimensional space. In this section, we also demonstrate that GAMA scales to higher dimensional, visual environments with  $64 \times 64 \times 3$  image inputs on the W2C and R2W experiments. Specifically, we train a deep spatial autoencoder on the alignment task set to learn an encoder with the architecture from Levine et al. (2016), then apply GAMA on the (learned) latent space. Comparing the dark blue (image input) and light blue curves (internal state input) in Figure 5, we see that the adaptation complexity and alignment complexity of GAMA-DA-img, GAMA-PA-img are both similar to that of GAMA-DA, GAMA-PA and better than baselines trained with the robot's internal representation.

# 7 DISCUSSION AND FUTURE WORK

We've formalized Cross Domain Imitation Learning which encompasses prior work in transfer learning across embodiment (Gupta et al., 2017) and viewpoint differences (Stadie et al., 2017; Liu et al., 2018) along with a practical algorithm that can be applied to both scenarios. We now point out directions future work. Our MDP alignability theory is a first step towards formalizing possible shared structures that enable cross domain imitation. While we've shown that GAMA empirically works well even when MDPs are not perfectly alignable, upcoming works may explore relaxing the conditions for MDP alignability to develop a theory that covers an even wider range of real world MDPs. Future works may also try applying GAMA in the imitation from observations scenario, i.e actions are not available, by aligning observations with GAMA and applying methods from Sermanet et al. (2018); Liu et al. (2018). Finally, we hope to see future works develop principled ways design a minimal alignment task set, which is analogous to designing a minimal training set for supervised learning.

# REFERENCES

Haitham Bou Ammar, Eric Eaton, Paul Ruvolo, and Matthew E. Taylor. An automated measure of mdp similarity for transfer in reinforcement learning. 2014.  
Haitham Bou Ammar, Eric Eaton, Paul Ruvolo, and Matthew E. Taylor. Unsupervised cross-domain transfer in policy gradient reinforcement learning via manifold alignment. 2015.  
Patrick Billingsley. Convergence of probability measures. 1968.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Zaremba Wojciech. Openai gym. arXiv preprint arXiv:1606.01540, 2016.  
Norman Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite markov decision processes. In UAI, 2004.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Abhishek Gupta, Coline Devin, Yu Xuan Liu, Pieter Abbeel, and Sergey Levine. Learning invariant feature spaces to transfer skills with reinforcement learning. International Conference on Learning Representations, 2017.  
Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In Advances in Neural Information Processing Systems, pp. 4565-4573, 2016.  
Harold Hotelling. Relations between two sets of variates. Biometrika, 28, 1936.  
Susan S. Jones. The development of imitation in infancy. Philos Trans R Soc Lond B Biol Sci., 364: 2325-2335, 2009.  
Girish Joshi and Girish Chowdhary. Cross-domain transfer in reinforcement learning using target apprentice. 2018.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, December 2014.  
Yasuo Kuniyoshi and Hirochika Inoue. Qualitative recognition of ongoing human action sequences. International Joint Conference on Artificial Intelligence, 1993.  
Yasuo Kuniyoshi, Masayuki Inaba, and Hirochika Inoue. Learning by watching: Extracting reusable task knowledge from visual observation of human performance. IEEE Trans. Robot. Autom., 10:799-822, 1994.  
Sergey Levine, Chelsea Finn, Trevor Darrell, and Pieter Abbeel. End-to-end training of deep visuomotor policies. The Journal of Machine Learning Research, 17(1):1334-1373, 2016.  
YuXuan Liu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Imitation from observation: Learning to imitate behaviors from raw video via context translation. arXiv preprint arXiv:1707.03374, 2018.  
Peter J. Marshall and Andrew N. Meltzoff. Body maps in the infant brain. Trends Cogn Sci., 19:499-505, 2015.  
Meinard Muller. Dynamic time warping. Information retrieval for music and motion, pp. 69-84, 2007.  
Ronald Ortner. Combinations and mixtures of optimal policies in unichain markov decision processes are optimal. arXiv preprint arXiv:0508319, 2005.  
Dean A Pomerleau. Efficient training of artificial neural networks for autonomous navigation. Neural computation, 3(1):88-97, 1991. ISSN 0899-7667.  
Balaraman Ravindran and Andrew G. Barto. Model minimization in hierarchical reinforcement learning. In SARA, 2002.  
Giacomo Rizzolatti and Laila Craighero. The mirror neuron system. Annual Review of Neuroscience, 27: 169-192, 2004.

Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. Time-contrastive networks: Self-supervised learning from video. arXiv preprint arXiv:1704.06888, 2018.  
Bradly Stadie, Pieter Abbeel, and Ilya Sutskever. Third person imitation learning. In  $ICLR$ , 2017.  
Umar Syed, Michael Bowling, and Robert E Schapire. Apprenticeship learning using linear programming. In Proceedings of the 25th international conference on Machine learning, pp. 1032-1039. ACM, July 2008. ISBN 9781605582054. doi: 10.1145/1390156.1390286.  
Tingwu Wang, Renjie Liao, Jimmy Ba, and Sanja Fidler. NerveNet: Learning structured policy with graph neural networks. International Conference on Learning Representations, 2018, 2018.
