# BIOLOGICAL SEQUENCE EDITING WITH GENERATIVE FLOW NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Editing biological sequences has extensive applications in synthetic biology and medicine, such as designing regulatory elements for nucleic-acid therapeutics and treating genetic disorders. The primary objective in biological-sequence editing is to determine the optimal modifications to a sequence which augment certain biological properties while adhering to a minimal number of alterations to ensure safety and predictability. In this paper, we propose GFNSeqEditor, a novel biological-sequence editing algorithm which builds on the recently proposed area of generative flow networks (GFlowNets). Our proposed GFNSeqEditor identifies elements within a starting seed sequence that may compromise a desired biological property. Then, using a learned stochastic policy, the algorithm makes edits at these identified locations, offering diverse modifications for each sequence in order to enhance the desired property. Notably, GFNSeqEditor prioritizes edits with a higher likelihood of substantially improving the desired property. Furthermore, the number of edits can be regulated through specific hyperparameters. We conducted extensive experiments on a range of real-world datasets and biological applications, and our results underscore the superior performance of our proposed algorithm compared to existing state-of-the-art sequence editing methods.

# 1 INTRODUCTION

Editing biological sequences has a multitude of applications in biology, medicine, and biotechnology. For instance, gene editing serves as a tool to elucidate the role of individual gene products in diseases (Li et al., 2020) and offers the potential to rectify genetic mutations in afflicted tissues and cells for therapeutic interventions (Cox et al., 2015). The primary objective in biological-sequence editing is to enhance specific biological attributes of a starting seed sequence, while minimizing the number of edits. This reduction in the number of alterations not only augments safety but also facilitates the predictability and precision of modification outcomes.

Existing machine learning methodologies within the domain of biological sequences have predominantly concentrated on generating novel de novo sequences with desired properties. These methods employ diverse techniques such as reinforcement learning (Angermueller et al., 2019), generative adversarial networks (Zrimec et al., 2022), diffusion models Avdeyev et al. (2023), model-based optimization approaches (Trabucco et al., 2021) and generative flow networks (Jain et al., 2022). A common feature of these approaches is generating entirely new sequences from scratch. As a result, there is an inherent risk of deviating significantly from naturally occurring sequences, compromising safety (e.g. the risk of designing sequences that might trigger an immune response) and predictability (e.g. having misleading predictions from models that are trained on genomic sequences due to out-of-distribution). Despite the paramount importance of editing biological sequences, there has been a noticeable scarcity of research dedicated to addressing this specific aspect.

The most traditional approaches for biological sequence editing are evolution-based methods, where—over many iterations—a starting "seed" sequence is randomly mutated, and only the best sequence (i.e., highest desired property) is kept for the next round (Arnold, 1998; Sinai et al., 2020; Taskiran et al., 2022); however, the utilization of these approaches necessitates the evaluation of numerous candidate edited sequences every iteration. This computational demand can become prohibitively expensive, particularly for lengthy sequences. Additionally, evolution-based methods heavily rely on evaluations provided by a proxy model capable of assessing the properties of

unseen sequences; the efficacy of these methods is limited by the reliability of the proxy model. Beyond evolution-based methods, a perturbation-based editing method known as Ledidi has been introduced by Schreiber et al. (2020). By treating sequence editing as an optimization task, Ledidi learns to perturb specific positions within a given sequence. Akin to evolution-based models, Ledidi's effectiveness is contingent on the quality of the proxy model, which can compromise Ledidi's performance if the proxy model lacks sufficient generalizability for unseen sequences. Furthermore, both evolution-based methods and Ledidi only perform local searches in sequence space, and as a result they suffer from low sample efficiency.

Generative flow networks (GFlowNets) (Bengio et al., 2021; 2023) are a generative approach known for their capacity to sequentially generate new objects. GFlowNets have demonstrated remarkable performance in the generation of novel biological sequences from scratch (Jain et al., 2022). Drawing inspiration from the emerging field of GFlowNets, this paper introduces a novel biological-sequence editing algorithm: GFNSeqEditor. Leveraging a pre-trained flow function from the GFlowNet (acquired through training on a sequence dataset), GFNSeqEditor assesses the potential for significant property enhancement within a given sequence. GFNSeqEditor iteratively identifies and subsequently edits specific positions in the input sequence to increase the target property. More precisely, using the trained flow function, GFNSeqEditor first identifies positions in the seed sequence which requires editing. GFNSeqEditor then constructs a stochastic policy using the flow function to select a substitution from the available options for the identified positions. Diversity holds significant importance when suggesting novel biological sequences (Mullis et al., 2019), and our stochastic approach empowers GFNSeqEditor to generate a diverse set of edited sequences for each input sequence. This is particularly crucial that the proposed sequences exhibit diversity and cover as much as possible the modes of a goodness function. This approach maximizes the likelihood that, ultimately, at least one of the edited sequences will prove effective.

In contrast to evolution-based methods and Ledidi, GFNSeqEditor does not engage in local searches. Instead, it relies on a pre-trained flow function that amortizes the search cost over the learning process, allocating probability mass across entire space to facilitate exploration and diversity. Unlike existing aforementioned de novo sequence generative methods, the proposed GFNSeqEditor distinguishes itself by the ability to create sequences that closely resemble existing natural sequences. More discussion about the related works can be found in Appendix D.

In summary, this paper makes the following contributions: 1) We introduce GFNSeqEditor, a novel sequence-editing method which identifies and edits positions within a given sequence. GFNSeqEditor generates diverse edits for each input sequence based on a stochastic policy. 2) We theoretically analyze the properties of the sequences edited through GFNSeqEditor, deriving a lower bound on the property enhancement. Additionally, we demonstrate that the upper bound for the number of edits performed by GFNSeqEditor can be controlled through the adjustment of hyperparameters (subsection 3.3). 3) We conduct experiments across various DNA and protein sequence editing tasks, showcasing GFNSeqEditor's remarkable efficiency in enhancing properties with a reduced number of edits when compared to existing state-of-the-art methods. (subsection 4.1). 4) We highlight the versatility of GFNSeqEditor, which can be employed not only for sequence editing but also alongside biological-sequence generation models to produce novel sequences with improved properties and increased diversity (subsection 4.2). 5) We demonstrate the usage of GFNSeqEditor for sequence length reduction, allowing the creation of new, relatively shorter sequences by combining pairs of long and short sequences (subsection 4.3).

# 2 PRELIMINARIES AND PROBLEM STATEMENT

Let  $\pmb{x}$  be a biological sequence with property  $y$ . For example,  $\pmb{x}$  may be a DNA sequence and  $y$  may be the likelihood it binds to a particular protein of interest. The present paper considers the problem of searching for edits in  $\pmb{x}$  to improve the sequence property. To this end, the goal is to learn an editor function  $\mathcal{E}(\cdot)$  which accepts a sequence  $\pmb{x}$  and outputs the edited sequence  $\mathcal{E}(\pmb{x}) = \hat{\pmb{x}}$  with property  $\hat{y}$ . The editor function  $\mathcal{E}(\cdot)$  should maximize  $\hat{y}$ , while at the same time minimizing the number of edits between  $\pmb{x}$  and  $\hat{\pmb{x}}$ . To achieve this goal, we propose GFNSeqEditor. GFNSeqEditor first identifies positions in a given biological sequence such that editing those positions leads to considerable improvement in the property of the sequence. Then, the learned editor function  $\mathcal{E}$  edits these identified locations (Figure 1). GFNSeqEditor uses a trained GFlowNet (Bengio et al., 2021;

2023) to identify positions that require editing and subsequently generate edits for those positions. The following subsections present preliminaries on GFlowNets.

# 2.1 GENERATIVE FLOW NETWORKS

Generative Flow Networks (GFlowNets) learn a stochastic policy  $\pi(\cdot)$  to sequentially construct a discrete object  $x$ . Let  $\mathcal{X}$  be the space of discrete objects  $x$ . It is assumed that the space  $\mathcal{X}$  is compositional, meaning that an object  $x$  can be constructed using a sequence of actions taken from an action set  $\mathbb{A}$ . At each step  $t$ , given a partially constructed object  $s_t$ , GFlowNet samples an action  $a_{t+1}$  from the set  $\mathbb{A}$  using the stochastic policy  $\pi(\cdot|s_t)$ . Then, GFlowNet appends  $a_{t+1}$  to  $s_t$  to obtain  $s_{t+1}$ . In this context,  $s_t$  can be viewed as the state at step  $t$ . The above procedure continues until reaching a terminating state, which yields the fully constructed object  $x$ . To construct an object  $x$ , the GFlowNet starts from an initial empty state  $s_0$ , and applying actions sequentially, all fully constructed objects must end in a special final state  $s_f$ . Therefore, the trajectory of states to construct an object  $x$  can be written as  $\tau_x = (s_0 \to s_1 \to \dots \to x \to s_f)$ . Let  $\mathbb{T}$  be the set of all possible trajectories. Furthermore, let  $R(\cdot): \mathcal{X} \to \mathbb{R}^+$  be a non-negative reward function defined on  $\mathcal{X}$ . The goal of GFlowNet is to learn a stochastic policy  $\pi(\cdot)$  such that  $\pi(x) \propto R(x)$ . This means that the GFlowNet learns a stochastic policy  $\pi(\cdot)$  to generate an object  $x$  with a probability proportional to its reward.

As described later, to obtain the policy  $\pi(\cdot)$ , the GFlowNet uses trajectory flow  $F: \mathbb{T} \to \mathbb{R}^+$ . The trajectory flow  $F(\tau)$  assigns a probability mass to the trajectory  $\tau$ . Then the edge flow from state  $s$  to state  $s'$  is defined as  $F(s \to s') = \sum_{\forall \tau: s \to s' \in \tau} F(\tau)$ . Moreover, the state flow is defined as  $F(s) = \sum_{\forall \tau: s \in \tau} F(\tau)$ . The trajectory flow  $F(\cdot)$  induces a probability measure  $P_F(\cdot)$  over completed trajectories that can be expressed as  $P_F(\tau) = \frac{F(\tau)}{Z}$  where  $Z = \sum_{\forall \tau \in \mathbb{T}} F(\tau)$  represents the total flow. The probability of visiting state  $s$  can be written as

$$
P _ {F} (s) = \frac {\sum_ {\forall \tau \in \mathbb {T} : s \in \tau} F (\tau)}{Z}. \tag {1}
$$

Then, the forward transition probability from state  $s$  to state  $s'$  can be obtained as

$$
P _ {F} \left(\boldsymbol {s} ^ {\prime} \mid \boldsymbol {s}\right) = \frac {F \left(\boldsymbol {s} \rightarrow \boldsymbol {s} ^ {\prime}\right)}{F (\boldsymbol {s})}. \tag {2}
$$

The trajectory flow  $F(\cdot)$  is called a consistent flow if for any state  $s$  it satisfies

$$
\sum_ {\forall \boldsymbol {s} ^ {\prime}: \boldsymbol {s} ^ {\prime} \rightarrow \boldsymbol {s}} F (\boldsymbol {s} ^ {\prime} \rightarrow \boldsymbol {s}) = \sum_ {\forall \boldsymbol {s} ^ {\prime \prime}: \boldsymbol {s} \rightarrow \boldsymbol {s} ^ {\prime \prime}} F (\boldsymbol {s} \rightarrow \boldsymbol {s} ^ {\prime \prime}), \tag {3}
$$

which constitutes that the in-flow and out-flow of state  $s$  are equal. Bengio et al. (2021) shows that if  $F(\cdot)$  is a consistent flow such that the terminal flow is set as reward (i.e.  $F(\pmb{x} \rightarrow \pmb{s}_f) = R(\pmb{x})$ ), the policy  $\pi(\cdot)$  defined as  $\pi(s'|s) = P_F(s'|s)$  satisfies  $\pi(\pmb{x}) = \frac{R(\pmb{x})}{Z}$  which means that the policy  $\pi(\cdot)$  samples an object  $\pmb{x}$  proportional to its reward.

# 2.2 TRAINING GFLOWNET MODELS

In order to learn the policy  $\pi (\cdot)$ , a GFlowNet model approximates trajectory flow with a flow function  $F_{\theta}(\cdot)$  where  $\pmb{\theta}$  includes learnable parameters of the flow function. In order to learn the flow

![](images/8491e2b6811b7580f07117019d8236204527bfb7b22022c3b7ae39cbc257bccf.jpg)  
Figure 1: An example of editing the DNA sequence 'ATGTCCGC'. The goal is to make a limited number of edits to maximize the property  $\hat{y}$ . Each token in the sequence in this example is called a base and can be any one letter from the alphabet ['A', 'C', 'T', 'G']. The editor function  $\mathcal{E}$  accepts the starting sequence and determines that the second and seventh bases require editing (highlighted in red). Then,  $\mathcal{E}$  modifies the bases at these identified locations.

function that can provide consistency condition, Bengio et al. (2021) formulates flow-matching loss function as follows:

$$
\mathcal {L} _ {\mathrm {F M}} (s; \boldsymbol {\theta}) = \left(\log \frac {\sum_ {\forall s ^ {\prime} : s ^ {\prime} \rightarrow s} F _ {\boldsymbol {\theta}} \left(s ^ {\prime} \rightarrow s\right)}{\sum_ {\forall s ^ {\prime \prime} : s \rightarrow s ^ {\prime \prime}} F _ {\boldsymbol {\theta}} \left(s \rightarrow s ^ {\prime \prime}\right)}\right) ^ {2}. \tag {4}
$$

Moreover, as an alternative objective function, Malkin et al. (2022) introduces trajectory balance as:

$$
\mathcal {L} _ {\mathrm {T B}} (s; \boldsymbol {\theta}) = \left(\log \frac {Z _ {\boldsymbol {\theta}} \prod_ {s \rightarrow s ^ {\prime}} P _ {F _ {\boldsymbol {\theta}}} \left(s ^ {\prime} \mid s\right)}{R (\boldsymbol {x})}\right) ^ {2} \tag {5}
$$

where  $Z_{\theta}$  is a learnable parameter. The trajectory-balance objective function in equation 5 can accelerate training GFlowNets and provide robustness to long trajectories. Given a training dataset, optimization techniques such as stochastic gradient descent can be applied to objective functions in equation 4 and equation 5 to train the GFlowNet model.

# 3 SEQUENCE EDITING WITH GFLOWNET

To edit a given sequence  $\pmb{x}$ , we propose identifying sub-optimal positions of  $\pmb{x}$  such that editing them can lead to considerable improvement in the sequence property. Assume that the flow function  $F_{\theta}(\cdot)$  is trained on an available offline training data. GFNSeqEditor uses the trained GFlowNet's flow function  $F_{\theta}(\cdot)$  to identify sub-optimal positions of  $\pmb{x}$ , and subsequently replace the sub-optimal parts with newly sampled edits based on the stochastic policy  $\pi(\cdot)$ .

# 3.1 SUB-OPTIMAL-POSITION IDENTIFICATION

This subsection provides intuition on how GFNSeqEditor uses a pre-trained flow function  $F_{\theta}(\cdot)$  to identify sub-optimal positions in a sequence  $x$  to edit. Let  $x_{t}$  and  $x_{:t}$  denote the  $t$ -th element and the first  $t$  elements in the sequence  $x$ , respectively. For example, in the DNA sequence  $x = {}^{\prime}\mathrm{ATGTCCGC}^{\prime}$ , we have  $x_{2} = {}^{\prime}\mathrm{T}^{\prime}$  and  $x_{:2} = {}^{\prime}\mathrm{AT}^{\prime}$ . GFNSeqEditor constructs edited sequences token by token and for each position  $t + 1$  it examines if  $x_{t + 1}$  should be used or not. Using the flow function  $F_{\theta}(\cdot)$ , given  $x_{:t}$ , GFlowNet would evaluate the average reward obtained by appending any possible token to  $x_{:t}$ . In this context, each token can be viewed as an action. Let  $x_{:t} + a$  denotes the expanded  $x_{:t}$  by appending token  $a$ . For instance for the DNA sequence  $x = {}^{\prime}\mathrm{ATGTCCGC}^{\prime}$ , appending token  $a = {}^{\prime}\mathrm{C}^{\prime}$  to  $x_{:2}$ , we get  $x_{:2} + a = {}^{\prime}\mathrm{ATC}^{\prime}$ . Let  $\mathbb{A}$  represent the available action set. For each  $a \in \mathbb{A}$ , using the state flow  $F_{\theta}(x_{:t} + a)$  the value of action  $a$  given  $x_{:t}$  can be evaluated. As discussed in Section 2, the state flow  $F_{\theta}(x_{:t} + a)$  is proportional to the total reward of all possible sequences that have  $x_{:t} + a$  as their prefix. Therefore, if  $F_{\theta}(x_{:t} + a_1) > F_{\theta}(x_{:t} + a_2)$ , this indicates that taking action  $a_1$  instead of action  $a_2$  can lead to obtaining better candidates for the final sequence. We can leverage this property of the flow function  $F_{\theta}(\cdot)$  to examine if  $x_{t + 1}$  is sub-optimal or not. If the reward resulting from having  $x_{t + 1}$  in the seed sequence is evaluated by  $F_{\theta}(\cdot)$  to be relatively small compared to other possible actions, then  $x_{t + 1}$  is considered sub-optimal. In particular,  $x_{t + 1}$  is identified as sub-optimal if we have

$$
F _ {\boldsymbol {\theta}} \left(\boldsymbol {x}: t + x _ {t + 1}\right) <   \delta \max  _ {a \in \mathbb {A}} F _ {\boldsymbol {\theta}} \left(\boldsymbol {x}: t + a\right) \tag {6}
$$

where  $0 \leq \delta \leq 1$  is a hyperparameter. Choosing larger  $\delta$ , it is more probable that the algorithm identifies  $x_{t+1}$  as sub-optimal. From equation 6 it can be inferred that  $x_{t+1}$  is identified as sub-optimal if its associated out-flow is considerably smaller than the out-flow associated with the best possible action in  $\mathbb{A}$ . This means that the flow function  $F_{\theta}(\cdot)$  suggests that replacing  $x_{t+1}$  with other actions can lead to remarkable improvement in the sequence property.

# 3.2 SEQUENCE EDITING WITH GFNSEQEDITOR

Using the flow function  $F_{\theta}(\cdot)$ , GFNSeqEditor iteratively identifies and edits positions in a seed sequence. Subsection 3.1 presented a simple function for determining if a position  $x_{t+1}$  in a sequence should be edited to improve the target property value (equation 6). Based on this intuition, we now modify equation 6 to formally define the sub-optimal-position identification function  $D(\cdot)$  used by GFNSeqEditor.

Algorithm 1 GFNSeqEditor: Sequence Editor using GFlowNet  
1: Input: Sequence  $\mathbf{x}$  with length  $T$ , flow function  $F_{\theta}(\cdot)$  and parameters  $\delta, \lambda$  and  $\sigma$ .  
2: Initialize  $\hat{\mathbf{x}}_{:0}$  as an empty sequence.  
3: for  $t = 1, \dots, T$  do  
4: Check if  $x_{t}$  is sub-optimal by obtaining  $D(x_{t}, \hat{\mathbf{x}}_{:t-1}; \delta, \sigma)$  according to equation 8.  
5: if  $D(\hat{\mathbf{x}}_{:t-1}; \delta, \sigma) = 1$  then  
6: Sample  $\hat{x}_{t}$  according to policy  $\pi(\cdot | \hat{\mathbf{x}}_{:t-1})$  in equation 9.  
7: else  
8: Assign  $\hat{x}_{t} = x_{t}$ .  
9: end if  
10: end for  
11: Output: Edited sequence  $\hat{\mathbf{x}}$ .

Let  $\hat{\pmb{x}}_{:t}$  denote the first  $t$  elements of the edited sequence. Assume that  $x_{t} \in \mathbb{A}, \forall t$  meaning that  $x_{t}$  is always in the available actions. At each step  $t$  of the algorithm,  $D(\cdot)$  accepts  $\hat{\pmb{x}}_{:t-1}$  and evaluates whether appending  $x_{t}$  (from the seed sequence) to the edited partial sequence  $\hat{\pmb{x}}_{:t-1}$  is detrimental to the performance. In order to perform exploration in sub-optimal identification, modifying condition in equation 6, the sub-optimal identifier function  $D(\cdot)$  checks the following condition:

$$
\frac {F _ {\boldsymbol {\theta}} \left(\hat {\boldsymbol {x}} : t - 1 + x _ {t}\right)}{\sum_ {a ^ {\prime} \in \mathbb {A}} F _ {\boldsymbol {\theta}} \left(\hat {\boldsymbol {x}} : t - 1 + a ^ {\prime}\right)} <   \delta \max  _ {a \in \mathbb {A}} \frac {F _ {\boldsymbol {\theta}} \left(\hat {\boldsymbol {x}} : t - 1 + a\right)}{\sum_ {a ^ {\prime} \in \mathbb {A}} F _ {\boldsymbol {\theta}} \left(\hat {\boldsymbol {x}} : t - 1 + a ^ {\prime}\right)} + \nu \tag {7}
$$

where  $\nu \sim \mathcal{N}(0,\sigma^2)$  is a Gaussian random variable with variance of  $\sigma^2$ . The variance  $\sigma^2$  is a hyperparameter. The relation between  $\sigma$  and the algorithm performance will be analyzed in section 3.3 and Appendix E. The inclusion of additive noise  $\nu$  on the right-hand side of equation 7 introduces a degree of randomness into the process of identifying sub-optimal positions. This, in turn, fosters exploration in the editing process. The sub-optimal-position-identifier function  $D(\cdot)$  determines if  $x_{t}$  is sub-optimal as follows:

$$
D \left(x _ {t}, \hat {\boldsymbol {x}} _ {: t - 1}; \delta , \sigma\right) = \left\{ \begin{array}{l l} 1 & \text {I f t h e c o n d i t i o n i n e q u a t i o n 7 i s m e t} \\ 0 & \text {O t h e r w i s e} \end{array} . \right. \tag {8}
$$

If  $D(x_{t},\hat{\pmb{x}}_{:t - 1};\delta ,\sigma) = 0$ , at step  $t$  the algorithm appends  $x_{t}$  from the original sequence  $\pmb{x}$  to  $\hat{\pmb{x}}_{:t - 1}$ . Otherwise, if  $D(x_{t},\hat{\pmb{x}}_{:t - 1};\delta ,\sigma) = 1$ , the algorithm samples an action  $a$  according to the following policy:

$$
\pi (a | \hat {\boldsymbol {x}} _ {: t - 1}) = (1 - \lambda) \frac {F _ {\boldsymbol {\theta}} (\hat {\boldsymbol {x}} _ {: t - 1} + a)}{\sum_ {a ^ {\prime} \in \mathbb {A}} F _ {\boldsymbol {\theta}} (\hat {\boldsymbol {x}} _ {: t - 1} + a ^ {\prime})} + \lambda \mathbf {1} _ {a = x _ {t}} \tag {9}
$$

where  $0 \leq \lambda < 1$  is a regularization coefficient and  $\mathbf{1}_{a = x_t}$  denotes indicator function and is 1 if  $a = x_t$ . The regularization parameter  $\lambda$  allows tuning the sampling process to favor the original sequence. Choosing larger  $\lambda$  leads to obtaining smaller number of edits. The policy in equation 9 constitutes a trade-off between increasing the target property and decreasing the distance between the edited sequence  $\hat{x}$  and the original sequence  $x$ . Specifically, the first term in the right hand side of equation 9 samples actions with probability proportional to their flow. The second term in the right hand side of equation 9 increases the likelihood of choosing the original  $x_t$  to reduce the distance between the edited sequence and the original one. Let  $\tilde{x}_t$  be the action sampled by the policy  $\pi$  in equation 9. In summary, the  $t$ -th element in the edited sequence can be written as

$$
\hat {x} _ {t} = D \left(x _ {t}, \hat {\boldsymbol {x}} _ {: t - 1}; \delta , \sigma\right) \tilde {x} _ {t} + \left(1 - D \left(x _ {t}, \hat {\boldsymbol {x}} _ {: t - 1}; \delta , \sigma\right)\right) x _ {t}. \tag {10}
$$

Therefore, at each step  $t$ , the edited sequence is updated as  $\hat{x}_{:t} = \hat{x}_{:t-1} + \hat{x}_t$ . This continues until the step  $T$  is reached where  $T = |x|$  denotes the length of the original sequence  $x$ . Note that  $\hat{x}_{:0}$  is an empty sequence. Algorithm 1 summarizes the proposed algorithm GFNSeqEditor.

# 3.3 ANALYSIS

This subsection analyzes the reward of the edited sequence and the number of edits performed by GFNSeqEditor. Specifically, the bounds for the reward of the edited sequence and the number of edits are determined by the algorithm's hyperparameters  $\sigma$ ,  $\delta$ , and  $\lambda$ . The following theorem specifies the lower bound for the reward of edited sequence by GFNSeqEditor.

Theorem 1. Let  $T$  be the length of the original sequence  $x$ . The expected reward of  $\hat{x}$  edited sequence by GFNSeqEditor given  $x$  is bounded from below as

$$
\mathbb {E} [ R (\hat {\boldsymbol {x}}) | \boldsymbol {x} ] \geq \left(1 - \Phi \left(\frac {1 - \delta}{\sigma}\right)\right) (1 - \lambda) R _ {F, T} \tag {11}
$$

where  $\Phi (\cdot)$  denotes the cumulative distribution function (CDF) for the normal distribution and  $R_{F,T}$  represents the expected reward of a sequence with length  $T$  generated using the flow function  $F_{\theta}(\cdot)$ .

Proof of Theorem 1 is deferred to Appendix A. From Theorem 1, we can deduce that greater values of  $\delta$  and  $\sigma$  correspond to larger lower bounds for the reward of the edited sequence. Furthermore, Theorem 1 demonstrates that a reduction in  $\lambda$  results in a larger lower bound for the reward. According to equation 9, a lower  $\lambda$  results in a higher editing probability. The following theorem shows the connections between the number of edits and the hyperparameters of GFNSeqEditor.

Theorem 2. The expected distance between the edited sequence  $\hat{x}$  by GFSeqEditor and the original sequence  $\pmb{x}$  is bounded from above as

$$
\mathbb {E} \left[ \operatorname {l e v} \left(\boldsymbol {x}, \hat {\boldsymbol {x}}\right) \right] \leq \left[ (1 - \lambda) \left(1 - \Phi \left(- \frac {\delta}{\sigma}\right)\right) \right] T \tag {12}
$$

where  $\mathrm{lev}(\cdot, \cdot)$  is the Levenshtein distance between two sequences.

The proof for Theorem 2 is available in Appendix B. Theorem 2 demonstrates that larger values of  $\delta$  yields a higher upper bound for the expected distance between the edited and original sequences, and conversely, a lower values of  $\lambda$  and  $\sigma$  leads to an increase in this expected distance. Theorems 1 and 2 reveal a trade-off between the expected number of edits and the lower bound for the expected reward. While it is preferable to select hyperparameters that reduce the expected number of edits, an increase in the number of edits corresponds to a larger lower bound for the reward. More analysis on property improvement upper bound and distance lower bound can be found in Appendix E.

# 4 EXPERIMENTS

We conducted extensive experiments to assess the performance of GFNSeqEditor in comparison to several state-of-the-art baselines across diverse DNA- and protein-sequence editing tasks. We evaluate on TFbinding, AMP, and CRE datasets. TFbinding and CRE datasets consist DNA sequences with lengths of 8 and 200, respectively. The task in both datasets is to edit sequences to increase their binding activities. The vocabulary for both TFbinding and CRE is the four DNA bases,  $\{\mathrm{A}, \mathrm{C}, \mathrm{G}, \mathrm{T}\}$ . AMP dataset comprises positive samples, representing anti-microbial peptides (AMPs), and negative samples, which are non-AMPs. The vocabulary consists of 20 amino acids. The primary objective is to edit the non-AMP samples in such a way that the edited versions attain the characteristics exhibited by AMP samples.

To evaluate the performance of sequence editing methods, we compute the following metrics:

- Property Improvement (PI): The PI for a given sequence  $\pmb{x}$  with label  $y$  is calculated as the average enhancement in property across edits, expressed as  $\mathrm{PI} = \frac{1}{n_e}\sum_{i=1}^{n_e}(\hat{y}_i - y)$  where  $n_e$  is the number of edited sequences associated with the original sequence  $\pmb{x}$  and  $\hat{y}_i$  denote the property of the  $i$ -th edited sequence  $\hat{\pmb{x}}_i$ . To evaluate the performance of editing methods, for each dataset we leverage an oracle to obtain  $\hat{y}_i$  given  $\hat{\pmb{x}}_i$ . More details about oracles can be found in Appendix C.  
- Edit Percentage (EP): The average Levenshtein distance between  $\pmb{x}$  and edited sequences normalized by the length of  $\pmb{x}$  expressed as  $\frac{1}{n_e T} \sum_{i=1}^{n_e} \mathrm{lev}(\pmb{x}, \hat{\pmb{x}}_i)$ .  
- Diversity: For each sequence  $\mathbf{x}$ , the diversity among edited sequences can be obtained as  $\frac{2}{n_e(n_e - 1)}\sum_{i = 1}^{n_e - 1}\sum_{j = i + 1}^{n_e}\mathrm{lev}(\hat{\boldsymbol{x}}_i,\hat{\boldsymbol{x}}_j)$ .

We compared GFNSeqEditor to several baselines, including Directed Evolution (DE) (Sinai et al., 2020), Ledidi (Schreiber et al., 2020), GFlowNet (Jain et al., 2022), and Seq2Seq. To perform Directed Evolution for sequence editing, we select a set of positions uniformly at random within a given sequence and then apply the directed-evolution algorithm to edit these positions. Inspired by graph-to-graph translation for molecular optimization in Jin et al. (2019), we implemented another editing baseline which is called Seq2Seq. For the Seq2Seq baseline, we initially partition the dataset into two subsets: i) sequences with lower target-property values, and ii) sequences with relatively

Table 1: Performance of GFNSeqEditor compared to the baselines in terms of property improvement (PI), edit percentage (EP) and diversity on TFbinding, AMP, and CRE datasets. Higher PI with a lower EP is preferable.  

<table><tr><td rowspan="2">Algorithms</td><td colspan="3">TFbinding</td><td colspan="3">AMP</td><td colspan="3">CRE</td></tr><tr><td>PI</td><td>EP(%)</td><td>Diversity</td><td>PI</td><td>EP(%)</td><td>Diversity</td><td>PI</td><td>EP(%)</td><td>Diversity</td></tr><tr><td>DE</td><td>0.12</td><td>25.00</td><td>3.01</td><td>0.11</td><td>33.82</td><td>13.67</td><td>0.63</td><td>22.93</td><td>62.07</td></tr><tr><td>Ledidi</td><td>0.06</td><td>27.80</td><td>1.25</td><td>0.18</td><td>34.79</td><td>11.65</td><td>1.36</td><td>22.13</td><td>50.49</td></tr><tr><td>GFlowNet-E</td><td>0.11</td><td>28.35</td><td>2.10</td><td>0.28</td><td>35.68</td><td>3.42</td><td>4.24</td><td>22.73</td><td>37.06</td></tr><tr><td>Seq2Seq</td><td>0.03</td><td>41.98</td><td>-</td><td>0.21</td><td>78.05</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GFNSeqEditor</td><td>0.14</td><td>24.27</td><td>3.84</td><td>0.33</td><td>34.49</td><td>14.34</td><td>9.90</td><td>21.90</td><td>40.41</td></tr></table>

higher target-property values. Subsequently, we create pairs of data samples such that each low-property sequence is paired with its closest counterpart from the high-property sequence set, based on Levenshtein distance. A translator model is then trained to map each low-property sequence to its high-property pair. Essentially, Seq2Seq baseline endeavors to map an input sequence to a similar sequence with superior property. Furthermore, we adapted GFlowNet-AL for sequence editing and named it GflowNet-E. In this baseline, the initial segment of the sequence serves as the input, allowing the model to generate the subsequent portion of the sequence. For TF-binding, AMP and CRE, GFlowNet-E takes in the initial  $70\%$ ,  $65\%$  and  $60\%$  of elements respectively from the input sequence  $x$ , and generates the remaining elements using the pre-trained flow function. Detailed information about the implementation of the baselines can be found in Appendix C.1.

To train models associated with baselines and the proposed GFNSeqEditor, we partition each dataset into a  $72\%$  training set and an  $18\%$  validation set. The remaining  $10\%$  constitutes the test set, employed to evaluate the performance of methods in sequence editing tasks. The trained flow function  $F_{\theta}(\cdot)$  employed by GFlowNet-E and the proposed GFNSeqEditor, is an MLP comprising two hidden layers, each with a dimension of 2048, and  $|\mathbb{A}|$  outputs corresponding to actions. Throughout our experiments, we employ the trajectory balance objective for training the flow function. Detailed information about training the flow function can be found in Appendix C.1.

# 4.1 SEQUENCE EDITING

Table 1 presents the performance of GFNSeqEditor and other baselines on TFbinding, AMP and CRE datasets  ${}^{1}$  . We set GFNSeqEditor and all baselines except for Seq2Seq to create 10 edited sequences for each input sequence. However, our Seq2Seq implementation closely resembles a deterministic machine translator and is limited to producing just one edited sequence per input, resulting in a diversity score of zero. Additionally, Figure 2 provides a visualization of property improvement achieved by GFNSeqEditor, DE, and Ledidi across a range of edit percentages. As evident from Table 1 and Figure 2, GFNSeqEditor surpasses all baselines in terms of achieving substantial property improvements with a minimal number of edits when compared to the other methods. This superior performance is attributed to GFNSeqEditor's utilization of a pre-trained flow function from GFlowNet, enabling it to attain notably higher property improvements than DE and Ledidi, which relies on local search techniques. Specifically, the flow function  ${F}_{\theta }\left( \cdot \right)$  is trained to sample sequences with probability proportional to their reward and as a result employing the policy in equation 9 for editing enables GFNSeqEditor to involve global information contained in  ${F}_{\theta }\left( \cdot \right)$  about the entire space of sequences. However, both DE and Ledidi perform local search such that at each iteration they perturb the edited sequence obtained from the previous iteration and then they evaluate their perturbed sequences using the proxy model to update the edited sequence. Furthermore, GFNSeqEditor achieves larger property improvement than GFlowNet-E. The GFNSeqEditor identifies and edits sub-optimal positions within a seed sequence using equation 7 while GFlowNet-E only edits the tail of the input seed sequence. This indicates the effectiveness of sub-optimal position identification of GFNSeqEditor. In addition to sequence editing, the proposed GFNSeqEditor is able to generate new sequences. The performance of GFNSeqEditor in generating new sequences is studied in Appendix C.4.

![](images/0b5769f77834aca3a8e2cf6acb509b258a9559595050fd66599fb485e2169e6c.jpg)  
Figure 2: Property improvement of AMP (left) and CRE (right) with respect to edit percentage.

![](images/739e4667cb61a143c94671546b69be0b7fc898dbe40f16fcabd29ccc43f4c7f7.jpg)

![](images/8e66050f038848c064c781808d9a6ad55595979889ffffeec87c083da75d5183.jpg)  
Figure 3: Studying the effect of hyperparameters  $\delta$  and  $\lambda$  on the performance of GFNSeqEditor over AMP (left) and CRE (right) datasets. The marker values are edit percentages.

![](images/320a300ebd9088904e31ab2a341007669aca2c69fc916885533d0b63515f3822.jpg)

Furthermore, in Figure 3, we present the property improvement achieved by GFNSeqEditor along with edit percentage across various choices of hyperparameters  $\delta$  and  $\lambda$ . The figure illustrates that an increase in  $\delta$  generally corresponds to an increase in both property improvement and edit percentage, whereas, in most cases, an increase in  $\lambda$  results in a decrease in property improvement and edit percentage. Furthermore, in Figure 4, we illustrate the impact of changing  $\sigma$  on property improvement and edit diversity for GFNSeqEditor. This figure highlights that increasing  $\sigma$  results in decreased property improvement and enhanced diversity. These results corroborate the theoretical analyses outlined in Theorems 1 and 2 in section 3.3 as well as Theorem 3 in Appendix E.

# 4.2 ASSISTING SEQUENCE GENERATION

GFNSeqEditor can complement generative models to enhance the generation of novel sequences. In this subsection, we incorporate a diffusion model (DM) for sequence generation, with further details available in Appendix C.2. The sequences generated unconditionally by the DM are passed to GFNSeqEditor to improve their target property. Given that GFNSeqEditor utilizes a trained GFlowNet model, this combination of a DM and GFNSeqEditor can be regarded as an ensemble approach, effectively leveraging both the DM and the GFlowNet for sequence generation. Table 2 presents the property and diversity metrics for sequences generated by the DM, the GFlowNet, and the combined DM+GFNSeqEditor across AMP and CRE datasets, with each method generating 1,000 sequences. As observed from Table 2, GFlowNet excels at producing sequences with higher property values compared to the DM, while the DM exhibits greater sequence diversity than the GFlowNet. Sequences generated by DM+GFNSeqEditor maintain similar property levels as the GFlowNet on its own, while their diversity is in line with that of the DM. This highlights the effectiveness of DM+GFNSeqEditor in harnessing the benefits of both the GFlowNet and the DM. Moreover, we show the CDF of properties for sequences generated by the DM, the GFlowNet, and DM+GFNSeqEditor in Figure 5. As shown, the CDF of DM+GFNSeqEditor aligns with both DM and GFlowNet. Specifically, for AMP dataset, DM+GFNSeqEditor generates more sequences with higher properties than 0.78 compared to GFlowNet, while reducing the number of low-property generated sequences compared to DM alone. In the case of CRE dataset, the results in Figure 5 indicate that as  $\delta$  increases, the CDF of DM+GFNSeqEditor becomes more akin to that of GFlowNet. This is expected, as an increase in  $\delta$  leads to a greater number of edits performed by GFNSeqEditor.

# 4.3 SEQUENCE COMBINATION

GFNSeqEditor possesses the capability to combine multiple sequences, yielding a novel sequence that closely resembles its parent sequences. This capability proves invaluable in applications where shortening relatively lengthy sequences is advantageous while retaining desired properties (see e.g. Xu et al. (2021); Zhao et al. (2023)). GFNSeqEditor accomplishes this by merging the longer sequence with a shorter one. The resultant sequence maintains similarities with the longer one to retain its desired properties while also resembling a realistic, relatively shorter sequence to ensure safety and predictability. Algorithm 2 in Appendix C.5 describes using GFNSeqEditor to combine two sequences to shorten the longer one. We evaluate GFNSeqEditor's performance in combining

![](images/38bcaa2d611871819a69618f5bd896cbae51848f7d793a49f5b2edce33e38322.jpg)  
Figure 4: Studying the effect of hyperparameter  $\sigma$  on the diversity and performance of GFNSeqEditor over AMP (left) and CRE (right) datasets.

![](images/23c01360309a115d058511d0edec9aaa184bb02846695b6bd0ce9cd83ad6e521.jpg)

![](images/8020a976fe75af0cd0137417c9a7cf4adbd0a36535eb7879db7e4f23b14d0b45.jpg)  
Figure 5: CDF of generated sequence properties for AMP (left) and CRE (right). A right-shifted curve indicates that the model is generating more sequences that are high in the target property.

![](images/deadf288246df8540560ddda4919dde697c93762c606cb274f3669b0037f0e91.jpg)

pairs of long and short sequences using the AMP dataset as a test case. In this context, a long sequence is defined as one with a length exceeding 30, while a short sequence has a length shorter than 20. Each pair consists of a long AMP sequence and the closest short sequence to the long one, chosen from among all short sequences, with an AMP property exceeding 0.7. Table 3 and Figure 7 in Appendix C.5 present the results of sequence combination for the purpose of reducing the length of long sequences. As indicated in Table 3, GFNSeqEditor not only enhances the properties of the initial long sequences but also significantly shortens them, by more than  $63\%$ . Additionally, the sequences generated by GFNSeqEditor exhibit a resemblance to both long and short sequences, with a Levenshtein similarity of approximately  $65\%$  to long sequences and  $55\%$  to short sequences.

Table 3: Performance of GFNSeqEditor for sequence reduction on AMP dataset in terms of variation in property, edit percentage of long sequences (EPLS), edit percentage of short sequences (EPSS), and percentage of length reduction in the long sequences.  

<table><tr><td></td><td>Input Property</td><td>Output Property</td><td>EPLS(%)</td><td>EPSS(%)</td><td>Sequence Reduction(%)</td></tr><tr><td>GFNSeqEditor</td><td>0.65</td><td>0.67</td><td>35.96</td><td>44.65</td><td>63.23</td></tr></table>

# 5 CONCLUSIONS

This paper introduces GFNSeqEditor, a sequence-editing method built upon GFlowNet. Given an input sequence, GFNSeqEditor identifies and edits positions within the input sequence to enhance its property. This paper also offers a theoretical analysis of the properties of edited sequences and the amount of edits performed by GFNSeqEditor. Experimental evaluations using real-world DNA and protein

Table 2: Performance of DM, GFlowNet and combination of DM with GFNSeqEditor for generating novel sequences.  

<table><tr><td rowspan="2">Algorithms</td><td colspan="2">AMP</td><td colspan="2">CRE</td></tr><tr><td>Property</td><td>Diversity</td><td>Property</td><td>Diversity</td></tr><tr><td>DM</td><td>0.66</td><td>23.86</td><td>1.75</td><td>107.38</td></tr><tr><td>GFlowNet</td><td>0.74</td><td>17.86</td><td>28.20</td><td>83.88</td></tr><tr><td>DM+GFNSeqEditor</td><td>0.73</td><td>23.78</td><td>26.42</td><td>103.10</td></tr></table>

datasets demonstrate that GFNSeqEditor outperforms state-of-the-art sequence-editing baselines in terms of property enhancement while maintaining a similar amount of edits. Moreover, the empirical findings highlight the versatility of GFNSeqEditor, showcasing its applications beyond single-sequence editing. Furthermore, GFNSeqEditor can effectively complement other generative models to generate sequences with improved properties and increased diversity. It can also be employed to combine two sequences into a new one with desired properties. Nevertheless, akin to many machine learning algorithms, GFNSeqEditor does have its limitations. It relies on a well-trained GFlowNet model, necessitating the availability of a high-quality trained GFlowNet for optimal performance.

# REFERENCES

Christof Angermueller, David Dohan, David Belanger, Ramya Deshpande, Kevin Murphy, and Lucy Colwell. Model-based reinforcement learning for biological sequence design. In International conference on learning representations, 2019.  
Frances H Arnold. Design by directed evolution. Accounts of chemical research, 31(3):125-131, 1998.  
Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. arXiv preprint arXiv:2305.10699, 2023.  
Luis A Barrera, Anastasia Vedenko, Jesse V Kurland, Julia M Rogers, Stephen S Gisselbrecht, Elizabeth J Rossin, Jaie Woodard, Luca Mariani, Kian Hong Kock, Sachi Inukai, et al. Survey of variation in human transcription factors reveals prevalent dna binding changes. Science, 351 (6280):1450-1454, 2016.  
Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, and Yoshua Bengio. Flow network based generative models for non-iterative diverse candidate generation. In Advances in Neural Information Processing Systems, volume 34, pp. 27381-27394, 2021.  
Yoshua Bengio, Salem Lahlou, Tristan Deleu, Edward J. Hu, Mo Tiwari, and Emmanuel Bengio. Gfownet foundations. Journal of Machine Learning Research, 24(210):1-55, 2023.  
David Benjamin Turitz Cox, Randall Jeffrey Platt, and Feng Zhang. Therapeutic genome editing: prospects and challenges. Nature medicine, 21(2):121-131, 2015.  
Hamid Dadkhahi, Jesus Rios, Karthikeyan Shanmugam, and Payel Das. Fourier representations for black-box optimization over categorical variables. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 10156-10165, 2022.  
Tristan Deleu, António Góis, Chris Emezue, Mansi Rankawat, Simon Lacoste-Julien, Stefan Bauer, and Yoshua Bengio. Bayesian structure learning with generative flow networks. In Uncertainty in Artificial Intelligence, pp. 518-528. PMLR, 2022.  
Sager J Gosai, Rodrigo I Castro, Natalia Fuentes, John C Butts, Susan Kales, Ramil R Noche, Kousuke Mouri, Pardis C Sabeti, Steven K Reilly, and Ryan Tewhey. Machine-guided design of synthetic cell type-specific cis-regulatory elements. bioRxiv, pp. 2023-08, 2023.  
Nikolaus Hansen. The cma evolution strategy: a comparing review. Towards a new evolutionary computation: Advances in the estimation of distribution algorithms, pp. 75-102, 2006.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840-6851, 2020. URL https://github.com/hojonathanho/diffusion.  
Samuel C Hoffman, Vijil Chenthamarakshan, Kahini Wadhawan, Pin-Yu Chen, and Payel Das. Optimizing molecules using efficient queries from property evaluations. Nature Machine Intelligence, 4(1):21-31, 2022.  
Moksh Jain, Emmanuel Bengio, Alex Hernandez-Garcia, Jarrid Rector-Brooks, Bonaventure F. P. Dossou, Chanakya Ajit Ekbote, Jie Fu, Tianyu Zhang, Michael Kilgour, Dinghuai Zhang, Lena Simine, Payel Das, and Yoshua Bengio. Biological sequence design with GFlowNets. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pp. 9786-9801, Jul 2022.  
Moksh Jain, Tristan Deleu, Jason Hartford, Cheng-Hao Liu, Alex Hernandez-Garcia, and Yoshua Bengio. Gflows nets for ai-driven scientific discovery. Digital Discovery, 2(3):557-577, 2023.  
Wengong Jin, Kevin Yang, Regina Barzilay, and Tommi Jaakkola. Learning multimodal graph-to-graph translation for molecule optimization. In International Conference on Learning Representations, 2019.

Hongyi Li, Yang Yang, Weiqi Hong, Mengyuan Huang, Min Wu, and Xia Zhao. Applications of genome editing technology in the targeted therapy of human diseases: mechanisms, advances and prospects. Signal transduction and targeted therapy, 5(1):1, 2020.  
Wenqian Li, Yinchuan Li, Zhigang Li, Jianye HAO, and Yan Pang. DAG matters! GFlownets enhanced explainer for graph neural networks. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= jgmuRzM-sb6.  
Kanika Madan, Jarrid Rector-Brooks, Maksym Korablyov, Emmanuel Bengio, Moksh Jain, Andrei Cristian Nica, Tom Bosc, Yoshua Bengio, and Nikolay Malkin. Learning gflows nets from partial episodes for improved convergence and stability. In International Conference on Machine Learning, pp. 23467-23483. PMLR, 2023.  
Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, and Yoshua Bengio. Trajectory balance: Improved credit assignment in GFlownets. In Advances in Neural Information Processing Systems, 2022.  
Nikolay Malkin, Salem Lahlou, Tristan Deleu, Xu Ji, Edward J Hu, Katie E Everett, Dinghuai Zhang, and Yoshua Bengio. GFlownets and variational inference. In The Eleventh International Conference on Learning Representations, 2023.  
Megan M Mullis, Ian M Rambo, Brett J Baker, and Brandi Kiel Reese. Diversity, ecology, and prevalence of antimicrobials in nature. Frontiers in microbiology, pp. 2518, 2019.  
Mizu Nishikawa-Toomey, Tristan Deleu, Jithendarraa Subramanian, Yoshua Bengio, and Laurent Charlin. Bayesian learning of causal structure and mechanisms with gflows nets and variational bayes. arXiv preprint arXiv:2211.02763, 2022.  
Ling Pan, Dinghuai Zhang, Aaron Courville, Longbo Huang, and Yoshua Bengio. Generative augmented flow networks. arXiv preprint arXiv:2210.03308, 2022.  
Ling Pan, Dinghuai Zhang, Moksh Jain, Longbo Huang, and Yoshua Bengio. Stochastic generative flow networks. arXiv preprint arXiv:2302.09465, 2023.  
Malak Pirtskhalava, Anthony A Amstrong, Maia Grigolava, Mindia Chubinidze, Evgenia Alimbarashvili, Boris Vishnepolsky, Andrei Gabrielian, Alex Rosenthal, Darrell E Hurt, and Michael Tartakovsky. DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides as a resource for development of new therapeutics. *Nucleic Acids Research*, 49(D1):D288–D297, 11 2020.  
Jacob Schreiber, Yang Young Lu, and William Stafford Noble. Ledidi: Designing genomic edits that induce functional activity. bioRxiv, 2020. doi: 10.1101/2020.05.21.109686. URL https://www.biorxiv.org/content/early/2020/05/25/2020.05.21.109686.  
Max W Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, and Tommaso Biancalani. Towards understanding and improving gflownet training. arXiv preprint arXiv:2305.07170, 2023.  
Sam Sinai, Richard Wang, Alexander Whatley, Stewart Slocum, Elina Locane, and Eric D Kelsic. Adalead: A simple and robust adaptive greedy search algorithm for sequence design. arXiv preprint arXiv:2010.02141, 2020.  
Yang Song, Jascha Sohl-Dickstein, Google Brain, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In arXiv preprint arXiv:2011.13456, 2021.  
Kevin Swersky, Yulia Rubanova, David Dohan, and Kevin Murphy. Amortized bayesian optimization over discrete spaces. In Conference on Uncertainty in Artificial Intelligence, pp. 769-778. PMLR, 2020.  
Ibrahim Ihsan Taskiran, Katina I Spanier, Valerie Christiaens, David Mauduit, and Stein Aerts. Cell type directed design of synthetic enhancers. bioRxiv, pp. 2022-07, 2022.

Kei Terayama, Masato Sumita, Ryo Tamura, and Koji Tsuda. Black-box optimization for automated discovery. Accounts of Chemical Research, 54(6):1334-1346, 2021.  
Brandon Trabucco, Aviral Kumar, Xinyang Geng, and Sergey Levine. Conservative objective models for effective offline model-based optimization. In International Conference on Machine Learning, pp. 10358-10368. PMLR, 2021.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.  
Xiaoshu Xu, Augustine Champarathy, Leiping Zeng, Hannah R. Kempton, Stephen Shang, Muneaki Nakamura, and Lei S. Qi. Engineered miniature crispr-cas system for mammalian genome regulation and editing. Molecular Cell, 81(20):4333-4345.e4, 2021.  
Dinghuai Zhang, Jie Fu, Yoshua Bengio, and Aaron Courville. Unifying likelihood-free inference with black-box optimization and beyond. arXiv preprint arXiv:2110.03372, 2021.  
Dinghuai Zhang, Ricky TQ Chen, Nikolay Malkin, and Yoshua Bengio. Unifying generative models with gflownets. arXiv preprint arXiv:2209.02606, 2022a.  
Dinghuai Zhang, Nikolay Malkin, Zhen Liu, Alexandra Volokhova, Aaron Courville, and Yoshua Bengio. Generative flow networks for discrete probabilistic modeling. In International Conference on Machine Learning, pp. 26412-26428. PMLR, 2022b.  
Dinghuai Zhang, Ling Pan, Ricky TQ Chen, Aaron Courville, and Yoshua Bengio. Distributional gflownets with quantile flows. arXiv preprint arXiv:2302.05793, 2023.  
Feiyu Zhao, Tao Zhang, Xiaodi Sun, Xiyun Zhang, Letong Chen, Hejun Wang, Jinze Li, Peng Fan, Liangxue Lai, Tingting Sui, et al. A strategy for cas13 miniaturization based on the structure and alphafold. Nature Communications, 14(1):5545, 2023.  
Heiko Zimmermann, Fredrik Lindsten, Jan-Willem van de Meent, and Christian A Naesseth. A variational perspective on generative flow networks. arXiv preprint arXiv:2210.07992, 2022.  
Jan Zrimec, Xiaozhi Fu, Azam Sheikh Muhammad, Christos Skrekas, Vykintas Jauniskis, Nora K Speicher, Christoph S Börlin, Vilhelm Verendel, Morteza Haghir Chehreghani, Devdatt Dubhashi, et al. Controlling gene expression with deep generative design of regulatory dna. Nature communications, 13(1):5099, 2022.
