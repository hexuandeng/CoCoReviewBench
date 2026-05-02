# LEARNING TO MAKE DECISIONS VIA SUBMODULAR REGULARIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Many sequential decision making tasks can be viewed as combinatorial optimization problems over a large number of actions. When the cost of evaluating an action is high, even a greedy algorithm, which iteratively picks the best action given the history, is prohibitive to run. In this paper, we aim to learn a greedy heuristic for sequentially selecting actions as a surrogate for invoking the expensive oracle when evaluating an action. In particular, we focus on a class of combinatorial problems that can be solved via submodular maximization (either directly on the objective function or via submodular surrogates). We introduce a data-driven optimization framework based on the submodular-norm loss, a novel loss function that encourages the resulting objective to exhibit diminishing returns. Our framework outputs a surrogate objective that is efficient to train, approximately submodular, and can be made permutation-invariant. The latter two properties allow us to prove strong approximation guarantees for the learned greedy heuristic. Furthermore, our model is easily integrated with modern deep imitation learning pipelines for sequential prediction tasks. We demonstrate the performance of our algorithm on a variety of batched and sequential optimization tasks, including set cover, active learning, and Bayesian optimization for protein engineering.

# 1 INTRODUCTION

In real-world automated decision making tasks, we seek the optimal set of actions that jointly achieve the maximal utility. Many of such tasks—either deterministic/non-adaptive or stochastic/adaptive—can be viewed as combinatorial optimization problems over a large number of actions. As an example, consider the active learning problem, where a learner seeks the maximally-informative set of training examples for learning a classifier. The utility of a training set could be measured by the mutual information (Lindley, 1956) between the training set and the remaining (unlabeled) data points, or by the expected reduction in generation error, should one train the model on the candidate training set. Similar problems arise in a number of other domains, such as experimental design (Chaloner and Verdinelli, 1995), document summarization (Lin and Bilmes, 2012), recommender system (Javdani et al., 2014), and policy making (Runge et al., 2011).

Identifying the optimal set of actions (e.g., optimal training sets, most informative experiments) amounts to evaluating the expected utility over a combinatorial number of candidate sets; when the underlying model class is complex and the evaluation of the utility function is expensive, these tasks are notoriously difficult to optimize (Krause and Guestrin, 2009). It has been shown that for a broad class of decision making problem—including the aforementioned applications in active learning and experimental design—where the optimization criterion is to maximize the decision-theoretic value of information, it is possible to design surrogate objective function that are (approximately) submodular while being aligned with the original objective at the optimal solutions (Javdani et al., 2014; Chen et al., 2015b; Choudhury et al., 2017). Here, the information gathering policies no longer aim to directly optimize the target objective value, but rather choose to follow a greedy trajectory governed by the underlying surrogate function. These insights have led to principled algorithms allowing a significant gain in efficiency of the decision making process with a greedy policy, while enjoying strong performance guarantees that are competitive with the optimal policy.

Despite the promising performance, a caveat for these "submodular surrogate"-based approaches is that it is often challenging to engineer such a surrogate objective without an ad-hoc design and

analysis. Furthermore, it has been established that for certain classes of surrogate functions, it is NP-hard to compute/evaluate the function value (Javdani et al., 2014). In such cases, even a greedy policy, which iteratively picks the best action given the (observed) history, can be prohibitively costly to design or run. Therefore it remains a critical challenges for solving more general classes of (sequential) decision making tasks, especially when involving expensive function evaluations.

Overview of main results Inspired by contemporary work in data driven decision making, we aim to learn a greedy heuristic for sequentially selecting actions, as a surrogate for invoking the expensive oracle when evaluating an action. Our key insight is that many practical algorithms can be interpreted as greedy approaches that follow an (approximate) submodular surrogate objective. In particular, we focus on the class of combinatorial problems that can be solved via submodular maximization (either directly on the objective function or via a submodular surrogates). We highlight some of the key results below:

- Focusing on utility-based greedy policies, we introduce a data-driven optimization framework based on the "submodular-norm" loss, which is a novel loss function that encourages learning a utility function that exhibits "diminishing returns". Our framework, called LEA-SURE (Learning with Submodular Regularization), outputs a surrogate objective that is permutation-invariant, efficient to train, and approximately submodular. The latter property allows us to prove strong approximation guarantees for the resulting greedy heuristic.  
- We show that our approach can be easily integrated with modern imitation learning pipelines for sequential prediction tasks. We provide rigorous analysis of the proposed algorithm and prove strong performance guarantees for the learned objective.  
- We demonstrate the performance of our approach on a variety of decision making tasks, including set cover, active learning for classification, and data-driven protein design. Our results suggest that LEASURE not only requires much fewer oracle calls at training phase to learn the target objective (i.e., to minimize the approximation error against the oracle objective), but also outperforms standard learning-based baselines for solving the corresponding optimization task (i.e., to minimize the regret for the original combinatorial optimization task).

# 2 RELATED WORK

Near-optimal decision making via submodular optimization. Submodularity is a property of a set function that resembles diminishing returns and it has wide applications from information gathering to document summarization (Krause and Golovin, 2014). The maximization of a submodular function has been an active area of study in various settings such as centralized (Nemhauser et al., 1978; Buchbinder et al., 2014; Mitrovic et al., 2017), streaming (Badanidiyuru et al., 2014; Kazemi et al., 2019; Feldman et al., 2020), continuous (Bian et al., 2017b; Bach, 2019) and approximate (Horel and Singer, 2016; Bian et al., 2017a). The greedy algorithm, which iteratively selects an element that maximizes the marginal gain, and its variants feature prominently in the algorithm design. For example, in the case of maximizing a monotone submodular function subject to a cardinality constraint, it is shown that the greedy algorithm achieves an approximation ratio of  $(1 - 1 / e)$  of the optimal solution (Nemhauser et al., 1978).

In applications where we need to make a sequence of decisions, such as information gathering, we usually need to adapt our future decisions based on past outcomes. Adaptive submodularity is the corresponding property where an adaptive greedy algorithm enjoys a similar guarantee for maximizing an adaptive submodular function (Golovin and Krause, 2011). Recent works have explored optimizing the value of information (Chen et al., 2015b) and Bayesian active learning (Javdani et al., 2014; Chen et al., 2017) with this property.

Learning submodular functions. Given the wide applications of submodular functions, the problem of learning submodular functions is the focus of intense study. Lin and Bilmes (2012) study a mixture of "submodular shells", which is an abstract representation of submodular functions, and apply it to the task of document summarization. Deep submodular functions (Dolhansky and Bilmes, 2016) mimic the structure of deep neural networks and represent a more expressive class of submodular functions than sums of concave composed with modular functions. The theoretical question of the learnability of general submodular functions is analyzed in (Balcan and Harvey, 2018).

Learning to optimize. Learning from demonstration (LfD), or imitation learning, explores techniques for learning a policy (i.e., a mapping from states to actions) directly from examples provided by an expert (e.g., an expensive computational oracle, or a human instructor) (Chernova and Thomaz, 2014). Classical work on imitation learning (e.g., the Dataset Aggregation (DAgger) algorithm (Ross et al., 2011)) takes reduction-based approaches that reduce the policy learning problem to the supervised learning setting. Recent works have explored using imitation learning to improve combinatorial optimization solvers. In (He et al., 2014; Song et al., 2018), the authors learn a node selection policy to replace the default selection rule in a branch-and-bound integer program solver. Similar ideas have also been applied to learn branching policies (Khalil et al., 2016; Gasse et al., 2019) and decomposition-based approaches (Song et al., 2020).

Learning active learning. Recently, Konyushkova et al. (2017) proposed a learning-based framework for active learning (LAL) with manually-engineered feature representation of historical active learning trajectories, and employed standard supervised learning approaches for learning an active learning policy, with the goal of generalizing the learned active learning policy to an unseen dataset. In (Liu et al., 2018), the authors adapt the DAgger algorithm to provide feedback on which new data points should be added in order to reduce the training loss. Then an active learning policy is trained to imitate the feedback on selecting new data points.

# 3 BACKGROUND AND PROBLEM STATEMENT

In this section, we formalize the optimal decision making problem addressed in this paper, and then introduce our learning-based decision making protocol.

# 3.1 DECISION MAKING VIA SUBMODULAR SURROGATE

Given a ground set of items  $\mathcal{V}$  to pick from. Let  $u:2^{\mathcal{V}}\to \mathbb{R}$  be a set function that measures the value of any given subset  $A\subseteq \mathcal{V}$ . For example, for experimental design,  $u(A)$  captures the utility of the output of the best experiment; for active learning  $u(A)$  captures the generalization error after training with set  $A$ . We denote a policy  $\pi :2^{\mathcal{V}}\rightarrow \mathcal{V}$  to be a partial mapping from the set/sequence of items already selected, to the next item to be picked. We use  $\Pi$  to denote our policy class. Each time a policy picks an item  $e\in \mathcal{V}$ , it incurs a unit cost. Given the ground set  $\mathcal{V}$ , the utility function  $u$ , and a budget  $k$  for selecting items, we seek the optimal policy  $\pi$  that achieves the maximal utility:

$$
\pi^ {*} \in \underset {\pi \in \Pi} {\arg \max } u (S _ {\pi , k}) \tag {1}
$$

Here  $S_{\pi, k}$  denotes the sequence of items picked by  $\pi$ , with  $S_{\pi, i} = S_{\pi, i-1} \cup \{\pi(S_{\pi, i-1})\}$  for  $i > 0$  and  $S_{\pi, 0} = \emptyset$ .

As we have discussed in the previous sections, many sequential decision making problems can be characterized as constrained monotone submodular maximization problem, i.e.,  $u$  is

- Monotone: For any  $A \subseteq \mathcal{V}$  and  $e \in \mathcal{V} \setminus A$ ,  $f(A) \leq f(A \cup \{e\})$ .  
- Submodular: For any  $A \subseteq B \subseteq \mathcal{V}$  and  $e \in \mathcal{V} \setminus B$ ,  $f(A \cup \{e\}) - f(A) \geq f(B \cup \{e\}) - f(B)$ .

In such cases, a myopic algorithm following the greedy trajectory of  $u$  admits a near-optimal policy. Note that when  $u$  is not monotone submodular, one strategy is to design a surrogate function  $f: 2^{\mathcal{V}} \to \mathbb{R}$  which is

1. globally aligning with  $u$ : For instance,  $f$  lies within a factor of  $u$ :  $f(A) \in [c_1 \cdot u(A), c_2 \cdot u(A)]$  for some constants  $c_1, c_2$  and any set  $A \subseteq \mathcal{V}$ ; or within a small margin with  $u$ :  $f(A) \in [u(A) - \epsilon, u(A) + \epsilon]$  for a fixed  $\epsilon > 0$  and any set  $A \subseteq \mathcal{V}$ .  
2. monotone submodular: Intuitively, a submodular surrogate function encourages selecting items that are beneficial in the long run, while ensuring that the decision maker does not miss out any actions that are "surprisingly good" by following a myopic policy (i.e., future gains for any item are diminishing). Examples that fall into this category include machine teaching (Singla et al., 2014), active learning (Chen et al., 2015a), etc.

We argue that in real-world decision making scenarios—as validated later in section 6—the decision maker is following a surrogate objective that aligns with the above characterization. In the following context, we will assume that such surrogate function exists. Our goal is thus to learn from an expert policy that behaves greedily according to such surrogate functions.

# 3.2 LEARNING TO MAKE DECISIONS

We focus on the regime where the expert policy is expensive to evaluate. Let  $g:2^{\mathcal{V}}\times \mathcal{V}\to \mathbb{R}$  be the score function that quantifies the benefit of adding a new item to an existing subset of  $\mathcal{V}$ . For the expert policy and submodular surrogate  $f$  discussed in section 3.1,  $\forall A\subseteq \mathcal{V}$  and  $e\in \mathcal{V}$

$$
g ^ {\exp} (A, e) = f (A \cup \{e \}) - f (A)
$$

For example, in the active learning case,  $g^{\exp}(A,e)$  could be the expert acquisition function that ranks the importance of labelling each unlabelled point, given the currently labelled subset. In the set cover case,  $g^{\exp}(A,e)$  could be the function that gives the score to each vertex and determines the next best vertex to add to the cover set. Given a loss function  $\ell$ , our goal is to learn a score function  $\hat{g}$  that incurs the minimal expected loss when evaluated against the expert policy:  $\hat{g} = \arg \min_{g}\mathbb{E}_{A,e}[\ell (g(A,e),g^{\exp}(A,e))]$ . Subsequently, the utility by the learned policy is  $u(S_{\hat{\pi},k})$  where for any given history  $A\subseteq \mathcal{V}$ ,  $\hat{\pi} (A)\in \arg \max_{e\in \mathcal{V}}\hat{g} (A,e)$ .

# 4 LEARNING WITH SUBMODULAR REGULARIZATION

To capture our intuition that a greedy expert policy tends to choose the most useful items, we introduce LEASURE, a novel regularizer that encourages the learned score function (and hence surrogate objective) to be submodular. We describe the algorithm below.

Given the groundset  $\mathcal{V}$ , let  $f:2^{\mathcal{V}}\to \mathbb{R}$  be any approximately submodular surrogate such that  $f(A)$  captures the "usefulness" of the set  $A$ . The goal of a trained policy is to learn a score function  $g:2^{\mathcal{V}}\times \mathcal{V}\rightarrow \mathbb{R}$  that mimics  $g^{\mathrm{exp}}(A,x) = f(A\cup \{x\}) - f(A)$ , which is often prohibitively expensive to evaluate exactly. Then, given any such  $g$ , we can define a greedy policy  $\pi (A) = \operatorname{argmax}_{x\in \mathcal{V}}g(A,x)$ . With LEASURE, we aim to learn such function  $g$  that approximates  $g^{\mathrm{exp}}$  well while being inexpensive to evaluate at test time. Let  $D_{real} = \{(\langle A_i,x_i\rangle ,y_i^{\mathrm{exp}} = g^{\mathrm{exp}}(A_i,x_i))\} _m$  be the gathered tuple of expert scores for each set - element pair. If the set  $2^{\mathcal{V}}\times \mathcal{V}$  was not too large, the LEASURE could be trained on the randomly collected tuples  $D_{real}$ . However,  $2^{\mathcal{V}}$  tends to be too large to explore, and generating ground truth labels could be very expensive. To leverage that, we also collect an unsupervised synthetic dataset of tuples  $D_{synth} = \{(\langle A,x\rangle ,\langle A',x\rangle |A\succeq A')_n$  where  $\langle A,x\rangle$  denote a labelled tuple  $\langle A,x\rangle \in D_{real}$ , and  $A^\prime$  denote a randomly selected superset of  $A$ . Define:

$$
\operatorname {Loss}(g,g^{\exp}) = \sum_{\langle A,x\rangle ,y^{\exp}\in D_{real}}(y^{\exp} - g(A,x))^{2} + \lambda \sum_{\substack{\langle A,x\rangle \\ \langle A^{\prime},x\rangle \in D_{synth}}}\operatorname {Sigmoid}([g(A^{\prime},x) - g(A,x)])
$$

where  $\lambda > 0$  is the regularization parameter. Intuitively, such regularization term will force the learned function  $g$  to be close to submodular, as it will lead to larger losses every time  $g(A', x) > g(A, x)$ . If we expect  $f$  to be monotonic, we also introduce a second regularizer  $\mathrm{ReLU}(-g(A', x))$  which pushes the learned function to be positive. Combined, the loss function becomes (used in line 11 in Algorithm 1):

$$
\begin{array}{l} \operatorname {Loss}(g,g^{\exp}) = \sum_{\langle A,x\rangle ,y^{\exp}\in D_{real}}(y^{\exp} - g(A,x))^{2} + \lambda \sum_{\substack{\langle A,x\rangle \\ \langle A^{\prime},x\rangle \in D_{synth}}}\operatorname {Sigmoid}([g(A^{\prime},x) - g(A,x)]) \\ + \gamma \sum_ {\left\langle A ^ {\prime}, x \right\rangle \in D _ {s y n t h}} \operatorname {R e L u} (- g \left(A ^ {\prime}, x\right)) \\ \end{array}
$$

where  $\gamma$  is another regularization strength parameter. Such loss should push  $g$  to explore a set of approximately submodular, approximately monotonic functions. Thus, if  $f$  exhibits the submodular and monotonic behavior,  $g$  trained on this loss function should achieve a good local minima.

Finally, before we provide the algorithm, note that since  $2^{\mathcal{V}}$  is too large to explore, instead of sampling random tuples for  $D_{real}$ , we use modified DAgger. Then  $g$  can learn not only from the expert

selections of  $\langle A,x\rangle$ , but it can also see the labels of the tuples the expert would not have chosen. The algorithm is provided below:

Algorithm 1 Learning to make decisions via Submodular Regularization (LEASURE)  
1: Input: Ground set  $\mathcal{V}$ , expert score function  $g^{\mathrm{exp}}$ ,  
2: regularization parameters  $\lambda, \gamma$ , DAGger constant  $\beta$ , the length of trajectories  $T$ .  
3: initialize  $D_{real} \gets \emptyset$   
4: initialize  $g$  to any function.  
5: for  $i = 1$  to  $N$  do  
6: let  $g_i = g^{\mathrm{exp}}$  with probability  $\beta$ .  
7: sample a batch of  $T$ -step trajectories using  $\pi_i(A) = x_i = \operatorname{argmax}_{x \in \mathcal{V}} g_i(A, x)$ .  
8: Get dataset  $D_i = \{\langle A_i, x_i \rangle, g^{\mathrm{exp}}(A_i, x_i)\}$  of labeled tuples on actions taken by  $\pi_i$ .  
9:  $D_{real} \gets D_{real} \cup D_i$ .  
10: Generate synthetic dataset  $D_{synth}$  from  $D_{real}$ .  
11: Train  $g_{i+1}$  on  $D_{real}$  and  $D_{synth}$  using the loss function above.  
12: end for  
13: Output:  $g_{N+1}$

Here, a trajectory on line 7 is a sequence of iteratively chosen tuples, i.e.  $(\langle \emptyset ,x_1\rangle ,\langle \{x_1\} ,x_2\rangle ,\langle \{x_1,x_2\} ,x_3\rangle \dots ,\langle \{x_1,\dots ,x_{T - 1}\} ,x_T\rangle)$ , collected using a mixed policy  $\pi_{i}$ . On line 8, expert feedback of selected actions is collected to form  $D_{i}$ .

We also want to note that in some settings, even collecting exact expert labels  $g^{\mathrm{exp}}$  at train time could be too expensive. In that case,  $g^{\mathrm{exp}}$  can be replaced with a less expensive, noisy approximate expert  $g_{\epsilon}^{\mathrm{exp}} \approx g^{\mathrm{exp}}$ . In fact, all three of our experiments use noisy experts in one form or another.

# 5 ANALYSIS

Estimating the expert's policy We first consider to bound the loss of the learned policy measured against the expert's policy. Since LEASURE could be viewed as a specialization of DAGGER (Ross et al., 2011) for learning a submodular function, it naturally inherits the performance guarantees from DAGGER, which show that the learned policy efficiently converges to the expert's policy. Concretely, the following result, which is proved in appendix, shows that the learned policy is consistent with the expert policy and thus is a no-regret algorithm:

Theorem 1 (Theorem 3.3, Ross et al. (2011)). Denote the loss of  $\hat{\pi}$  at history state  $H$  as  $l(H,\hat{\pi})\coloneqq \ell(g(H,\hat{\pi}(H)),g^{exp}(H,\pi^{exp}(H)))$ . Let  $d_{\hat{\pi}}$  be the average distribution of states if we follow  $\hat{\pi}$  for a finite number of steps. Furthermore, let  $D_{i}$  be a set of  $m$  random trajectories sampled with  $\pi_{i}$  at round  $i\in \{1,\ldots ,N\}$ , and  $\hat{\epsilon}_N = \min_{\pi}\frac{1}{N}\sum_{i = 1}^N\mathbb{E}_{H_i\sim D_i}[l(H_i,\hat{\pi})]$  be the training loss of the best policy on the sampled trajectories. If  $N$  is  $\mathcal{O}\left(T^{2}\log (1 / \delta)\right)$  and  $m$  is  $\mathcal{O}$  (1) then with probability at least  $1 - \delta$  there exists a  $\hat{\pi}$  among the  $N$  policies, with  $\mathbb{E}_{H\sim d_{\hat{\pi}}}[l(H,\hat{\pi})]\leq \hat{\epsilon}_N + \mathcal{O}\left(\frac{1}{T}\right)$ .

Approximating the optimal policy Note that the previous notion of regret corresponds to the average difference in score function between the learned policy and the expert policy. While this result shows that LEASURE is consistent with the expert, it does not directly address how well the learned policy performs in terms of the gained utility. We then provide a bound on the expected value of the learned policy, measured against the value of the optimal policy. For specific decision making tasks where the oracle follows an approximately submodular objective, our next result, which is proved in the appendix, shows that the learned policy behaves near-optimally.

Theorem 2. Assume that the utility function  $u$  is monotone submodular. Furthermore, assume the expert policy  $\pi^{exp}$  follows a surrogate objective  $f$  such that for all  $A \subseteq \mathcal{V}$ ,  $|f(A) - u(A)| < \epsilon_E$  where  $\epsilon_E > 0$ . Let  $\hat{\epsilon}_N = \min_{\pi} \frac{1}{N} \sum_{i=1}^N l(H_i, \hat{\pi})$  be the training loss of the best policy on the sampled trajectories. Let  $k' = \min \{k, b\}$  for  $k, b > 0$ . If  $N$  is  $\mathcal{O}\left(T^2 \log (k'/\delta)\right)$  then with probability at least  $1 - \delta$ , the expected utility achieved by running  $\hat{\pi}$  for  $b$  steps is

$$
\mathbb {E} \left[ u \left(S _ {\hat {\pi}, b}\right) \right] \geq \left(1 - e ^ {- \frac {b}{k}}\right) \mathbb {E} \left[ u \left(S _ {\pi^ {*}, k}\right) \right] - k \left(2 \hat {\epsilon} _ {N} + 4 \epsilon_ {E}\right) - \mathcal {O} \left(\frac {k}{T}\right).
$$

# 6 EXPERIMENTS

In this section, we demonstrate the performance of LEASURE on three diverse sequential decision making tasks, namely set cover, learning active learning for classification and protein engineering.

Baselines We compare our approach to the Deep Submodular Function (Dolhansky and Bilmes (2016)). In their paper, the authors learn a submodular surrogate function  $f: 2^{\mathcal{V}} \to \mathbb{R}$  that produces a score for each set  $A \subset \mathcal{V}$ . The architecture of the Deep Submodular Function (DSF) forces the function  $f$  to be exactly submodular, as opposed to LEASURE, which is only encouraged to be submodular through a regularizer. However, the architecture and the training procedure of the DSF are quite restrictive, which does not allow the DFS to explore a large domain during training and restricts how expressive it can be compared to a standard neural network. Moreover, DSF are restricted to small  $\mathcal{V}$ , and the number of parameters increases with the size of  $\mathcal{V}$ . That is not true for LEASURE, which number of parameters grows with the dimensionality of elements in  $\mathcal{V}$ . This makes DSF useful for small datasets, but makes it prohibitively expensive to use on larger problems. In fact, we could not compare LEASURE to DSF on Learning Active Learning or Protein Engineering Tasks, as it was not feasible to train DSF on these sets. Finally, we want to add that LEASURE can be seamless integrated with any standard Machine Learning library, and since the architecture of the learned policy in LEASURE is not restrictive, any available optimization trick can be used to achieve better performance. On the other hand, DSF cannot be as easily implemented, and the standard libraries are not optimized for the DSF architecture.

# 6.1 SET COVER

Before testing our approach on a real-world scenario, we showcase its performance on a simple submodular and monotonic maximization problem. Set cover is a classical example: given a set of elements  $U = \{1,2,\dots,n\}$  (called the universe) and a collection of  $m$  sets  $S = \{s_1,\dots,s_m\}$  whose union equals the universe, the set cover problem is to identify the smallest sub-collection of  $S$  whose union equals the universe. Formulated as a policy learning problem, the goal is to learn the score function  $g:2^{S}\times S\to \mathbb{R}$  such that for any  $S_{l}\subset S,x\in S$ ,

$$
g (S _ {l}, x) \approx g ^ {\exp} (S _ {l}, x) = | \cup_ {s \in S _ {l}} s \cup x | - | \cup_ {s \in S _ {l}} s |
$$

Given such a function  $g$ , we can then define a policy  $\pi : 2^s \to S$  as  $\pi(S_l) = \operatorname{argmax}_{x \in S} g(S_l, x)$ . During training, tuples  $\{(S_l, x), g^{\exp}\}$  are collected, and then  $g$  is trained on this set. We trained four different policies: a function  $g$  parametrized by a neural network with  $MSE(g, g^{\exp})$  as the loss, a function  $g$  with the same loss and just a monotonicity regularizer, a function  $g$  trained using both monotonicity and submodular regularizers, as well as the Deep Submodular Function baseline (Dolhansky and Bilmes (2016)). Our dataset is the subset of the Mushroom dataset Lim (2015), consisting of 1000 sets. Each set contains 23 mushroom species, and there are a total of 119 species. The goal is then to train a policy to select the largest superset of these sets. We train and test learned policies in two settings: Exact Set Cover, where we collect tuples  $\{(S_l, x), g^{\exp}\}$  for training, and Noisy Set Cover, where we have access only to  $\{(S_l, x), g_\epsilon^{\exp}\}$ , where  $g_\epsilon^{\exp}$  is a noisy score. The networks are trained on rollouts of length 20 (i.e. on sets  $\{S_l : |S_l| \leq 20\}$ ), and tested on rollout of length up to 100. The results of training policies on these two frameworks are in Figure 4. In 1a and 1b, we plot the value of set cover as a function of the size of the superset. LEASURE significantly outperforms other learned policies, although Deep Submodular Function generalized better to larger rollout lengths - LEASURE gets most of its set cover gains in the first 10 - 20 selected points, while Deep Submodular Function continues to noticeably improve past the training rollout length.

Note that in 1a and 1b, the competing baselines all exhibit an "diminishing returns" effect, resulting in a concave-shaped value function. With a submodular-norm regularizer, LEASURE quickly identified the sets with large marginal gains. This observation aligns with our analysis in section 5.

# 6.2 LEARNING ACTIVE LEARNING ON FASHION MNIST

In this section we demonstrate the performance of LEASURE on a real-world task that is not submodular or monotonic, but that usually exhibits submodular and monotonic behaviour. In particular, in an active learning framework, there is a partially labelled dataset  $S = \{S_{l}, S_{u}\}$ , and a policy  $\pi: 2^{S} \to S$ . The labelled subset  $S_{l}$  can be used to infer from data (learn the image classifier, predict unlabelled protein fitness, etc). The goal of the policy is to select the smallest sub

![](images/b9a51c5ee83422b1d40bd6748ea5815a42dc0bd7e0a50d784b26633f1f42e878.jpg)  
(a)

![](images/ba06c4027f68f47847404672700646fdb1a17be331785eb1be3422e88192b221.jpg)  
Figure 1: Evaluating LEASURE against baselines on set cover instances  
(b)

set  $S_{\pi} \subset S_u$  to label such that the accuracy of supervised learning from  $S_{\pi} \cup S_l$  is maximized. Since selecting a subset is a prohibitively expensive combinatorial task, the policy is usually sequential. In particular, it selects points to add to  $S_{\pi}$  one by one (or in batches) using some score function  $g(S_{\pi} \cup S_l, \cdot): S_u \to \mathbb{R}$  to score each point  $x \in S_u$  and then the policy labels the point with the largest score. If  $g$  were to be the first order difference of a submodular function  $f$ , i.e.  $g(A, e) = f(A \cup \{e\}) - f(A)$ , then the policy would be near-optimal. Moreover, as discussed above, intuitively we expect  $g$  to have this property in most cases, since adding an extra point to a larger set usually has less effect than adding the same point to a smaller subset of the set.

The above motivates the use of LEASURE in active learning (Figure 2). In this experiment, the set  $S$  is a fashion MNIST dataset of  $28 \times 28$  pixels, greyscale images of clothes that come from one of the 10 classes. The goal was to learn a policy that greedily selects a point  $x \in S_u$  to label, such that a neural network classifier trained on the labelled set  $S_l \cup \{x\}$  produces the most accurate classification of the unlabelled images. In particular, we trained the above function  $g$  to predict the accuracy gain  $g^{\mathrm{exp}}$  from labelling a point. The accuracy gain  $g^{\mathrm{exp}}$  was measured by training the neural network classifier on both  $S_l$  and  $S_l \cup \{x\}$  and then recording the difference in validation set classification accuracy. Since obtaining exact  $g^{\mathrm{exp}}$  for each datapoint is very expensive, we instead collected noisy labels  $g_{\epsilon}^{\mathrm{exp}} \approx g^{\mathrm{exp}}$ , obtained by training the classifier for only 10 epochs. The tuples  $\{(S_l,x), g_{\epsilon}^{\mathrm{exp}}\}$  were collected using DAgger with rollouts of length 30 (starting from a random batch of 20 images). For training, we used an initially unlabelled dataset with 60000 images, 2000 of which were set aside to use for evaluating validation accuracy. We trained two neural networks to approximate  $g$ -an unregularized one, and one with a monotonicity and a submodularity regularizer (i.e. LEASURE).

![](images/c56996e4df01e14025568291538c6362ea8d22c8df43bfa569c388173199bde4.jpg)  
Figure 2: Combining submodular regularization with a learned active learning policy for 10-class Fashion MNIST classification. The figure summarizes the classification error rate of a classifier neural network trained on labelled images, as a function of the number of labelled images. Originally, random set of 20 images is selected, and then each policy greedily chooses the next image to label. The learned policies were trained on rollouts of length up to 30, and tested on rollouts of length 200. The "no regularizer" policy corresponds to (Konyushkova et al., 2017) LAL, only in this case the features are parametrized by the neural network instead of being hand-engineered. The results are averaged between 500 experiments, with standard error reported (but barely visible).

The trained policies were tested on a set of 8000 images, with additional 2000 set aside for validation. At test time, we again started with a random batch of size 20 and then used each policy to sequentially select additional 200 images to label (Figure 2). The recorded test error rate was collected using real  $g^{\mathrm{exp}}$ , i.e. a classifier trained until training loss reaches a certain threshold.

Even though LEASURE was trained on much shorter rollouts using very noisy labels, it still outperformed all other baselines. The submodular regularizer allows the learned score function  $g$  to find a local minima that generalizes well to out of sample.

![](images/585028c5db9b805b76d8dcb8a37272b8369e76d781c6dbfe93e31c6744273ee6.jpg)  
(a) Comparison to baseline methods

![](images/89d90517b74c5cf8c1a58bc78f8d5d9694a88f81696f0d8d1788c62eaaa8ecbd.jpg)  
Figure 3: Combining submodular regularization with a learned active learning policy for a protein engineering task. In (b), Lambda = 0 corresponds to the unregularized case. Error bars are plotted as standard error of the mean across 50 replicates.  
(b) Effect of scaling parameter lambda

# 6.3 PROTEIN ENGINEERING

By employing a large protein engineering database containing mutation-function data (Wang et al., 2019), we demonstrate that LEASURE enables the learning of an optimal policy for imitating expert design of protein sequences (see Appendix for detailed discussion of datasets). As in (Liu et al., 2018) we construct a fully data-driven expert which evaluates via one step roll-out the effect of labeling each candidate data (in our case a protein mutant) with the objective of minimizing loss on a downstream regression task (predicting protein fitness). Briefly, during training expert selections are paired with state representations describing the currently labeled and unlabeled data and a policy is trained to assign a preference score for labeling each data. The use of submodular regularization enables the learning of a policy which generalizes to a fundamentally different protein engineering task. In our experiments, LEASURE is trained to emulate a greedy oracle for maximizing the stability of protein G, a small bacterial protein used across a range of biotechnology applications (Sjbring et al., 1991). We evaluate our results by applying the trained policy to select data for the task of predicting antibody binding of a small molecule. As is the case with all protein fitness landscapes, the evaluation dataset is highly imbalanced, with the vast majority of mutants conferring no improvement at all. Because data is expensive to label in biological settings (proteins must be synthesized, purified and tested), we are often limited in how many labels can feasibly be generated, and the discriminative power among the best results is often more important than among the worst. To construct a metric with real-world applicability we assess each model by systemically examining the median Kd of the next ten data points selected at each budget, from 10 to 110 total labels.

We observe that LEASURE outperforms all evaluated baselines, and that the inclusion of submodular optimization is mandatory to its success (figure 5b). A greedy active learner which labels the antibody mutation with the best predicted Kd (the smallest) preforms approximately equivalently with selecting random labels. Use of dropout as an approximation of model uncertainty as in (Gal and Ghahramani, 2015) improves upon these baselines, although significant betterment is not achieved until approximately 35 labels are added. In comparison, the results from LEASURE diverge from all others nearly immediately, and the best model, which uses a lambda of 0.1, achieves a notable improvement in Kd,  $5.81\mu \mathrm{M}$ , vs  $7.27\mu \mathrm{M}$  achieved by entropy sampling. In support of methods success, we note that the learned policy preforms approximately as well as the greedy oracle which it emulates (Appendix figure 5b). We observe that the results are robust within a range of possible lambda values (Fig 2b), and that without the use of submodular regularization the trained policy fails to learn a policy better than the selection of random labels.

# 7 CONCLUSION

In this paper, we introduce LEASURE, a data-driven decision making framework based on a novel submodular-regularized loss function. The algorithm was inspired by the recent developments of submodular-surrogate-based near-optimal algorithms for sequential decision making. We have demonstrated LEASURE on several diverse set of decision making tasks. Our results suggest that LEASURE can be easily integrated with modern deep imitation learning pipelines, and that it is efficient to run, while still reaching the best performance among the competing baselines. In addition to demonstrating the strong empirical performance on several use cases, we believe our work also provides useful insights in the design and analysis of novel information acquisition heuristics where traditional ad-hoc approaches are not feasible.

# REFERENCES

E. C. Alley, G. Khimulya, S. Biswas, M. AlQuraishi, and G. M. Church. Unified rational protein engineering with sequence-based deep representation learning. Nat. Methods, 16(12):1315-1322, 12 2019.  
Francis Bach. Submodular functions: from discrete to continuous domains. Mathematical Programming, 175(1-2):419-459, 2019.  
Ashwinkumar Badanidiyuru, Baharan Mirzasoleiman, Amin Karbasi, and Andreas Krause. Streaming submodular maximization: Massive data summarization on the fly. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 671-680, 2014.  
Maria-Florina Balcan and Nicholas JA Harvey. Submodular functions: Learnability, structure, and optimization. SIAM Journal on Computing, 47(3):703-754, 2018.  
Andrew An Bian, Joachim M Buhmann, Andreas Krause, and Sebastian Tschiatschek. Guarantees for greedy maximization of non-submodular functions with applications. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 498-507, 2017a.  
Andrew An Bian, Baharan Mirzasoleiman, Joachim Buhmann, and Andreas Krause. Guaranteed non-convex optimization: Submodular maximization over continuous domains. In Artificial Intelligence and Statistics, pages 111-120, 2017b.  
Niv Buchbinder, Moran Feldman, Joseph Naor, and Roy Schwartz. Submodular maximization with cardinality constraints. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms, pages 1433-1452. SIAM, 2014.  
K. Chaloner and I. Verdinelli. Bayesian experimental design: A review. Statistical Science, 10(3): 273-304, 1995.  
Yuxin Chen, S Hamed Hassani, Amin Karbasi, and Andreas Krause. Sequential information maximization: When is greedy near-optimal? In Conference on Learning Theory, pages 338-363, 2015a.  
Yuxin Chen, Shervin Javdani, Amin Karbasi, James Andrew Bagnell, Siddhartha Srinivasa, and Andreas Krause. Submodular surrogates for value of information. In Proc. Conference on Artificial Intelligence (AAAI), January 2015b.  
Yuxin Chen, S. Hamed Hassani, and Andreas Krause. Near-optimal bayesian active learning with correlated and noisy tests. In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), April 2017.  
Sonia Chernova and Andrea L Thomaz. Robot learning from human teachers. Synthesis Lectures on Artificial Intelligence and Machine Learning, 8(3):1-121, 2014.  
Sanjiban Choudhury, Ashish Kapoor, Gireeja Ranade, and Debadeepta Dey. Learning to gather information via imitation. In 2017 IEEE International Conference on Robotics and Automation (ICRA), pages 908-915. IEEE, 2017.  
Brian W Dolhansky and Jeff A Bilmes. Deep submodular functions: Definitions and learning. In Advances in Neural Information Processing Systems, pages 3404-3412, 2016.  
Moran Feldman, Ashkan Norouzi-Fard, Ola Svensson, and Rico Zenklusen. The one-way communication complexity of submodular maximization with applications to streaming and robustness. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pages 1363-1374, 2020.  
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. 2015. URL http://arXiv.org/abs/.  
Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, and Andrea Lodi. Exact combinatorial optimization with graph convolutional neural networks. In Advances in Neural Information Processing Systems, pages 15580-15592, 2019.

Daniel Golovin and Andreas Krause. Adaptive submodularity: Theory and applications in active learning and stochastic optimization. Journal of Artificial Intelligence Research, 42:427-486, 2011.  
He He, Hal Daume III, and Jason M Eisner. Learning to search in branch and bound algorithms. In Advances in neural information processing systems, pages 3293-3301, 2014.  
Thibaut Horel and Yaron Singer. Maximization of approximately submodular functions. In Advances in Neural Information Processing Systems, pages 3045-3053, 2016.  
Shervin Javdani, Yuxin Chen, Amin Karbasi, Andreas Krause, James Andrew Bagnell, and Siddhartha Srinivasa. Near-optimal bayesian active learning for decision making. In In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), April 2014.  
Ehsan Kazemi, Marko Mitrovic, Morteza Zadimoghaddam, Silvio Lattanzi, and Amin Karbasi. Submodular streaming in all its glory: Tight approximation, minimum memory and low adaptive complexity. In International Conference on Machine Learning, pages 3311-3320, 2019.  
Elias Boutros Khalil, Pierre Le Bodic, Le Song, George Nemhauser, and Bistra Dilkina. Learning to branch in mixed integer programming. In Thirtieth AAAI Conference on Artificial Intelligence, 2016.  
Ksenia Konyushkova, Raphael Sznitman, and Pascal Fua. Learning active learning from data. In Advances in Neural Information Processing Systems, pages 4225-4235, 2017.  
Andreas Krause and Daniel Golovin. Submodular function maximization., 2014.  
Andreas Krause and Carlos Guestrin. Optimal value of information in graphical models. JAIR, 35: 557-591, 2009.  
Ching Lih Lim. A suite of greedy methods for set cover computation. 2015.  
Hui Lin and Jeff A Bilmes. Learning mixtures of submodular shells with application to document summarization. arXiv preprint arXiv:1210.4871, 2012.  
Dennis V Lindley. On a measure of the information provided by an experiment. The Annals of Mathematical Statistics, pages 986-1005, 1956.  
Ming Liu, Wray Buntine, and Gholamreza Haffari. Learning how to actively learn: A deep imitation learning approach. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1874-1883, 2018.  
Marko Mitrovic, Mark Bun, Andreas Krause, and Amin Karbasi. Differentially private submodular maximization: data summarization in disguise. In International Conference on Machine Learning, pages 2478-2487, 2017.  
George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functionsi. Mathematical programming, 14(1):265-294, 1978.  
Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Xi Chen, John Canny, Pieter Abbeel, and Yun S. Song. Evaluating protein transfer learning with tape. 2019. URL http://arXiv.org/abs/.  
Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 627-635, 2011.  
M. C. Runge, S. J. Converse, and J. E. Lyons. Which uncertainty? using expert elicitation and expected value of information to design an adaptive program. Biological Conservation, 2011.  
Adish Singla, Ilija Bogunovic, Gábor Bartók, Amin Karbasi, and Andreas Krause. Near-optimally teaching the crowd to classify. In ICML, volume 1, page 3, 2014.  
U. Sjbring, L. Bjrck, and W. Kastern. Streptococcal protein G. Gene structure and protein binding properties. J. Biol. Chem., 266(1):399-405, Jan 1991.

Jialin Song, Ravi Lanka, Albert Zhao, Aadyot Bhatnagar, Yisong Yue, and Masahiro Ono. Learning to search via retrospective imitation. arXiv preprint arXiv:1804.00846, 2018.  
Jialin Song, Ravi Lanka, Yisong Yue, and Bistra Dilkina. A general large neighborhood search framework for solving integer linear programs. In Advances in Neural Information Processing Systems, 2020.  
C. Y. Wang, P. M. Chang, M. L. Ary, B. D. Allen, R. A. Chica, S. L. Mayo, and B. D. Olafson. ProtaBank: A repository for protein design and engineering data. *Protein Sci.*, 28(3):672, Mar 2019.
