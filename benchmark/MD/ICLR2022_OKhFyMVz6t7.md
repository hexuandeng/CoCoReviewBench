# DECONFOUNDING TO EXPLANATION EVALUATION IN GRAPH NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Explainability of graph neural networks (GNNs) aims to answer "Why the GNN made a certain prediction?", which is crucial to interpret the model prediction. The feature attribution framework distributes a GNN's prediction to its input features (e.g., edges), identifying an influential subgraph as the explanation. When evaluating the explanation (i.e., subgraph importance), a standard way is to audit the model prediction based on the subgraph solely. However, we argue that a distribution shift exists between the full graph and the subgraph, causing the out-of-distribution problem. Furthermore, with an in-depth causal analysis, we find the OOD effect acts as the confounder, which brings spurious associations between the subgraph importance and model prediction, making the evaluation less reliable. In this work, we propose Deconfounded Subgraph Evaluation (DSE) which assesses the causal effect of an explanatory subgraph on the model prediction. While the distribution shift is generally intractable, we employ the front-door adjustment and introduce a surrogate variable of the subgraphs. Specifically, we devise a generative model to generate the plausible surrogates that conform to the data distribution, thus approaching the unbiased estimation of subgraph importance. Empirical results demonstrate the effectiveness of DSE in terms of explanation fidelity.

# 1 INTRODUCTION

Explainability of graph neural networks (GNNs) (Hamilton et al., 2017; Dwivedi et al., 2020) is crucial to model understanding and reliability in real-world applications, especially when about fairness and privacy (Ying et al., 2019; Luo et al., 2020). It aims to provide insight into how predictor models work, answering "Why the target GNN made a certain prediction?". Towards this end, a variety of explainer models are proposed for feature attribution (Selvaraju et al., 2017; Ying et al., 2019; Luo et al., 2020; Vu & Thai, 2020), which decomposes the predictor's prediction as contributions (i.e., importance) of its input features (e.g., edges, nodes). While feature attribution assigns the features with importance scores, it redistributes the graph features and creates a new distribution different from that of the original full graphs, from which a subgraph is sampled as the explanation. Such sampling process is referred to as feature removal (Covert et al., 2020).

Then, to assess the explanatory subgraph, the current evaluation frameworks use the feature removal principle — (1) only feed the subgraph into the target predictor, discarding the other features; (2) measure the importance of the subgraph based on its information amount to recover the model's prediction. Such subgraph-prediction correlations uncovered by the removal-based evaluator should offer a faithful inspection of the predictor's decision-making process and assess the fidelity of the explainers reliably.

However, feature removal brings the out-of-distribution (OOD) problem (Frye et al., 2020; Chang et al., 2019; Lukas Faber, 2021): the distribution shift from full graphs to subgraphs likely violates underlying properties, including node degree distribution (Leskovec et al., 2005) and domain-specific constraints (Liu et al., 2018) of the full graphs. For example, graph properties of chemical molecules, such as the valency rules, impose some constraints on syntactically valid molecules (Liu et al., 2018); hence, simply removing some bonds (edges) or atoms (nodes) creates invalid molecular subgraphs that never appear in the training dataset. Such OOD subgraphs could manipulate the predictor's

![](images/cc72e4c4bd8fc7c54d7d76b7fd3ab7704d2585bf8207fb05eaa9db6fb5b280c0.jpg)  
(a) Feature Removal to Evaluate Explanatory Subgraph  $G_{s}$

![](images/27223f78aca63589fb99c911d4e937f9ade772099fcb3a201f233b3a19d64734.jpg)  
Figure 1: (a) A real example in TR3. The GNN predictor classifies the full graph as 'House'. On subgraphs  $\mathcal{G}_{s1}$  and  $\mathcal{G}_{s2}$ , the prediction probabilities of being "House" are respectively 0.21 and 0.70. (b) The structural causal model represents the causalities among variables:  $G$  as the input graph,  $D$  as the unobserved distribution shift,  $G_{s}$  as the explanatory subgraph, and  $Y$  as the model prediction. outcome arbitrarily (Dai et al., 2018; Zügner et al., 2018), generates erroneous predictions, and limits the reliability of the evaluation process.  
(b) SCM I

Here we demonstrate the OOD effect by a real example in Figure 1a, where the trained ASAP (Ranjan et al., 2020) predictor has classified the input graph as "House" for its attached motif (see Section 4 for more details). On the ground-truth explanation  $\mathcal{G}_{s1}$ , the output probability of the "House" class is surprisingly low (0.21). While for  $\mathcal{G}_{s2}$  with less discriminative information, the outputs probability of the "House" class (0.70) is higher. Clearly, the removal-based evaluator assigns the OOD subgraphs with unreliable importance scores, which are unfaithful to the predictor's decision.

The OOD effect has not been explored in evaluating GNN explanations, to the best of our knowledge. We rigorously investigate it from a causal view (Pearl et al., 2016; Pearl, 2000; Pearl & Mackenzie, 2018). Figure 1b represents our causal assumption via a structural causal model (SCM) (Pearl et al., 2016; Pearl, 2000), where we target the causal effect of  $G_{s}$  on  $Y$ . Nonetheless, as a confounder between  $G_{s}$  and  $Y$ , distribution shift  $D$  opens the spurious path  $G_{s} \gets D \to Y$ . By "spurious", we mean that the path lies outside the direct causal path from  $G_{s}$  to  $Y$ , making  $G_{s}$  and  $Y$  spuriously correlated and yielding an erroneous effect. And one can hardly distinguish between the spurious correlation and causative relations (Pearl et al., 2016). Hence, auditing  $Y$  on  $G_{s}$  suffers from the OOD effect and wrongly evaluates the importance of  $G_{s}$ .

Motivated by our causal insight, we propose a novel evaluation paradigm, Deconfounded Subgraph Evaluator (DSE), to faithfully measure the causal effect of explanatory subgraphs on the prediction.

Based on Figure 1b, as the distribution shift  $D$  is hardly measurable, we cannot block the backdoor path from  $G_{s}$  to  $Y$  by the backdoor adjustment. Thanks to the front-door adjustment (Pearl et al., 2016), we instead consider the SCM in Figure 2, where we introduce the surrogate  $G_{s}^{*}$  between  $G_{s}$  and  $Y$ , which "imagines" what the full graphs like given the subgraphs. We obtain the causal effect of  $G_{s}$  on  $Y$  by identifying the causal effects carried by  $G_{s} \rightarrow G_{s}^{*}$  and  $G_{s}^{*} \rightarrow Y$ , which requires  $G_{s}^{*}$  to respect the data distribution. Hence we design a generative model, Conditional Variational Graph Auto-Encoder (CVGAE), to generate the possible surrogates. It is worthwhile mentioning that our DSE is explainer-agnostic, which can assist the explanation evaluation reliably and further guide explainers to generate faithful explanations.

![](images/929316ecf8e5570c31dc34cd52e912604e2b282495b13d11ae1e58aecc214382.jpg)  
Figure 2: SCM II with a mediating variable  $G_{s}^{*}$ .

In a nutshell, our contributions are:

- From a causal perspective, we argue that the OOD effect is the confounder that causes spurious correlations between subgraph importance and model prediction. It harms the removal-based evaluation of the explanatory subgraphs.  
- We propose a deconfounding paradigm, DSE, which exploits the front-door adjustment to mitigate the out-of-distribution effect and evaluate the explanatory subgraphs unbiasedly.  
- We validate the effectiveness of our framework over various explainers, target GNN models, and datasets. Significant boosts are achieved over the conventional feature removal techniques. Code and datasets are available at: https://anonymous.4open.science/r/DSE-24BC/.

# 2 A CAUSAL VIEW OF EXPLANATION EVALUATION

Here we begin with the causality-based view of feature removal in Section 2.1 and present our causal assumption to inspect the OOD effect in Section 2.2.

# 2.1 PROBLEM FORMULATION

Without loss of generality, we focus on the graph classification task: a well-trained GNN predictor  $f$  takes the graph variable  $G$  as input and predicts the class  $Y \in \{1, \dots, K\}$ , i.e.,  $Y = f(G)$ .

Generation of Explanatory Subgraphs. Post-hoc explainability typically considers the question "Why the GNN predictor  $f$  made certain prediction?" A prevalent solution is building an explainer model to conduct feature attribution (Ying et al., 2019; Luo et al., 2020; Pope et al., 2019). It decomposes the prediction into the contributions of the input features, which redistributes the probability of features according to their importance and sample the salient features as an explanatory subgraph  $\mathcal{G}_s$ . Specifically,  $\mathcal{G}_s$  can be a structure-wise (Ying et al., 2019; Luo et al., 2020) or feature-wise (Ying et al., 2019) subgraph of  $\mathcal{G}$ . In this paper, we focus on the structural features. That is, for graph  $\mathcal{G} = (\mathcal{N},\mathcal{E})$  with the edge set  $\mathcal{E}$  and the node set  $\mathcal{N}$ , the explanatory subgraph  $\mathcal{G}_s = (\mathcal{N}_s,\mathcal{E}_s)$  consists of a subset of edges  $\mathcal{E}_s \subset \mathcal{E}$  and their endpoints  $\mathcal{N}_s = \{u,v|(u,v) \in \mathcal{E}_s\}$ .

Evaluation of Explanatory Subgraphs. Insertion-based evaluation by feature removal (Covert et al., 2020; Dabkowski & Gal, 2017) aims to check whether the subgraph is the supporting substructure that alone allows a confident classification. We systematize this paradigm as three steps: (1) divide the full graph  $\mathcal{G}$  into two parts, the subgraph  $\mathcal{G}_s$  and the complement  $\mathcal{G}_{\overline{s}}$ ; (2) feed  $\mathcal{G}_s$  into the target GNN  $f$ , while discarding  $\mathcal{G}_{\overline{s}}$ ; and (3) obtain the model prediction on  $\mathcal{G}_s$ , to assess its discriminative information to recover the prediction on  $\mathcal{G}$ . Briefly, at the core of the evaluator is the subgraph-prediction correlation. However, as discussed in Section 1, the OOD effect is inherent in the removal-based evaluator, hindering the subgraph-prediction correlation from accurately estimating the subgraph importance.

# 2.2 STRUCTURAL CAUSAL MODEL

To inspect the OOD effect rigorously, we take a causal look at the evaluation process with a Structural Causal Model (SCM I) in Figure 1b. We denote the abstract data variables by the nodes, where the directed links represent the causality. The SCM indicates how the variables interact with each other through the graphical definition of causation:

-  $G \to G_s \gets D$ . We introduce an abstract distribution shift variable  $D$  to sample a subgraph  $G_s$  from the edge distributions of the full graph  $G$ .  
-  $G_{s} \rightarrow Y \leftarrow D$ . We denote  $Y$  as the prediction variable (e.g., logits output), which is determined by (1) the direct effect from  $G_{s}$ , and (2) the confounding effect caused by  $D$ . In particular, the former causation that led to the result is the focus of this work.

We suggest readers to refer to Appendix A where we offer an elaboration of  $D$ . With our SCM assumption, directly measuring the importance of explanatory subgraphs is distracted by the backdoor path (Pearl, 2000),  $G_{s} \gets D \to Y$ . This path introduces the confounding associations between  $G_{s}$  and  $Y$ , which makes  $G_{s}$  and  $Y$  spuriously correlated, i.e., biases the subgraph-prediction correlations, thus making the evaluator invalid. How to mitigate the OOD effect and quantify  $G_{s}$ 's genuine causal effect on  $Y$  remains largely unexplored in the literature and is the focus of our work.

# 3 DECONFOUNDED EVALUATION OF EXPLANATORY SUBGRAPHS

In this section, we propose a novel deconfounding framework to evaluate the explanatory subgraphs in a trustworthy way. Specifically, we first leverage the front-door adjustment (Pearl, 2000) to formulate a causal objective in Section 3.1. We then devise a conditional variational graph auto-encoders (CVGAE) as the effective implementation of our objective in Section 3.2.

# 3.1 FRONT-DOOR ADJUSTMENT

To the best of our knowledge, our work is the first to adopt the causal theory to solve the OOD problem in the explanation evaluation of GNNs. To pursue the causal effect of  $G_{s}$  on  $Y$ , we perform the calculus of the causal intervention  $P(Y = y|do(G_s = \mathcal{G}_s))$ . Specifically, the do-calculus (Pearl, 2000; Pearl et al., 2016) is to intervene the subgraph variable  $G_{s}$  by cutting off its coming links and assigning it with the certain value  $\mathcal{G}_s$ , making it unaffected from its causal parents  $G$  and  $D$ . From inspection of the SCM in Figure 1b, the distribution effect  $D$  acts as the confounder between  $G_{s}$  and  $Y$ , and opens the backdoor path  $G_{s} \leftarrow D \rightarrow Y$ . However, as  $D$  is hardly measurable, we can not use the backdoor adjustment (Pearl, 2000; Pearl et al., 2016) to block the backdoor path from  $G_{s}$  to  $Y$ . Hence, the causal effect of  $G_{s}$  on  $Y$  is not identifiable from SCM I.

However, we can go much further by considering SCM II in Figure 2 instead, where a mediating variable  $G_{s}^{*}$  is introduced between  $G_{s}$  and  $Y$ :

-  $G_{s} \to G_{s}^{*}$ .  $G_{s}^{*}$  is the surrogate variable of  $G_{s}$ , which completes  $G_{s}$  to make them in the data distribution. First, it is rooted from  $G_{s}$  and imagines how the possible full graphs should be when observing the subgraph  $G_{s}$ . Second,  $G_{s}^{*}$  should follow the data distribution and respect the inherent knowledge of graph properties, thus no link exists between  $D$  and  $G_{s}^{*}$ .  
-  $G_{s}^{*} \to Y$ . This is based on our causal assumption that the causality-related information of  $G_{s}$  on  $Y$ , i.e., the discriminative information for  $G_{s}$  to make prediction, is well-preserved by  $G_{s}^{*}$ . Thus, with the core of  $G_{s}$ ,  $G_{s}^{*}$  is qualified to serve as the mediator which further results in the model prediction.

With SCM II, we can exploit the front-door adjustment (Pearl, 2000; Pearl et al., 2016) instead to quantify the causal effect of  $G_{s}$  on  $Y$ . Specifically, by summing over possible surrogate graphs  $\mathcal{G}_s^*$  of  $G_{s}^{*}$ , we chain two identifiable partial effects of  $G_{s}$  on  $G_{s}^{*}$  and  $G_{s}^{*}$  on  $Y$  together:

$$
\begin{array}{l} P \left(Y | d o \left(G _ {s} = \mathcal {G} _ {s}\right)\right) = \sum_ {\mathcal {G} _ {s} ^ {*}} P \left(Y | d o \left(G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*}\right)\right) P \left(G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*} \mid d o \left(G _ {s} = \mathcal {G} _ {s}\right)\right) \\ = \sum_ {\mathcal {G} _ {s} ^ {*}} \sum_ {\mathcal {G} _ {s} ^ {\prime}} P (Y | G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*}, G _ {s} = \mathcal {G} _ {s} ^ {\prime}) P (G _ {s} = \mathcal {G} _ {s} ^ {\prime}) P (G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*} | d o (G _ {s} = \mathcal {G} _ {s})) \\ = \sum_ {\mathcal {G} _ {s} ^ {*}} \sum_ {\mathcal {G} _ {s} ^ {\prime}} P (Y | G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*}, G _ {s} = \mathcal {G} _ {s} ^ {\prime}) P \left(G _ {s} = \mathcal {G} _ {s} ^ {\prime}\right) P \left(G _ {s} ^ {*} = \mathcal {G} _ {s} ^ {*} \mid G _ {s} = \mathcal {G} _ {s}\right), \tag {1} \\ \end{array}
$$

Specifically, we have  $P(G_{s}^{*}|do(G_{s} = \mathcal{G}_{s})) = P(G_{s}^{*}|G_{s} = \mathcal{G}_{s})$  as  $G_{s}$  is the only parent of  $G_{s}^{*}$ . And we distinguish the  $\mathcal{G}_s$  in our target expression  $P(Y|do(G_s = \mathcal{G}_s))$  between  $\mathcal{G}_s^\prime$ , the latter of which is adjusted to pursue  $P(Y|do(G_s^* = \mathcal{G}_s^*))$ . With the data of  $(\mathcal{G}_s,\mathcal{G}_s^*)$  pairs, we can obtain  $P(Y|G_s^* = \mathcal{G}_s^*,G_s = \mathcal{G}_s^{\prime})$  by feeding the surrogate graph  $\mathcal{G}_s^*$  into the GNN predictor, conditional on the subgraph  $\mathcal{G}_s^\prime$ ; similarly, we can estimate  $P(G_{s} = \mathcal{G}_{s}^{\prime})$  statistically;  $P(\bar{G}_s^* = \mathcal{G}_s^* |G_s = \mathcal{G}_s)$  is the conditional distribution of the surrogate variable, after observing the subgraphs. As a result, this front-door adjustment yields a consistent estimation of  $G_{s}$ 's effect on  $Y$  and avoids the confounding associations from the OOD effect.

# 3.2 DEEP GENERATIVE MODEL

However, it is non-trivial to instantiate  $\mathcal{G}_s^*$  and collect the  $(\mathcal{G}_s,\mathcal{G}_s^*)$  pairs. We get inspiration from the great success of generative models and devise a novel probabilistic model, conditional variational graph auto-encoder (CVGAE), and an adversarial training framework, to generate  $\mathcal{G}_s^*$ .

Conditional Generation. Inspired by previous works (Thomas N. Kipf, 2016; Liu et al., 2018), we model the data distribution via a generative model  $g_{\theta}$  parameterized by  $\theta$ . It is composed of an encoder  $q(\mathbf{Z}|\mathcal{G},\mathcal{G}_s)$  and a decoder  $p(\mathcal{G}_s^* |\mathbf{Z})$ . Specifically, the encoder  $q(\mathbf{Z}|\mathcal{G},\mathcal{G}_s)$  embeds each node  $i$  in  $\mathcal{G}$  with a stochastic representation  $\mathbf{z}_i$ , and summarize all node representations in  $\mathbf{Z}$ :

$$
q (\mathbf {Z} | \mathcal {G}, \mathcal {G} _ {s}) = \prod_ {i} ^ {N} q \left(\mathbf {z} _ {i} \mid \mathcal {G}, \mathcal {G} _ {s}\right), \quad \text {w i t h} \quad q \left(\mathbf {z} _ {i} \mid \mathcal {G}, \mathcal {G} _ {s}\right) = \mathcal {N} \left(\mathbf {z} _ {i} \mid [ \boldsymbol {\mu} _ {1 i}, \boldsymbol {\mu} _ {2 i} ], \left[ \begin{array}{c c} \sigma_ {1 i} ^ {2} & 0 \\ 0 & \sigma_ {2 i} ^ {2} \end{array} \right]\right) \tag {2}
$$

where  $\mathbf{z}_i$  is sampled from a diagonal normal distribution by mean vector  $[\pmb{\mu}_{1i},\pmb{\mu}_{2i}]$  and standard deviation vector  $\mathrm{diag}(\pmb{\sigma}_{1i}^2,\pmb{\sigma}_{2i}^2)$ ;  $\pmb{\mu}_1 = f_\mu (\mathcal{G})$  and  $\log \sigma_{1} = f_{\sigma}(\mathcal{G})$  denote the matrices of mean

![](images/9ff03d79729cd38dcadd0e7b0ecf13269d31841bb168fd85e1989d7e448695d9.jpg)  
Figure 3: Model structure of CVGAE.  $P_{dse}$  is the average probability of  $\mathcal{G}_s^*$  on the target prediction.

vectors  $\pmb{\mu}_{1i}$  and standard deviation vectors  $\log \sigma_{1i}$  respectively, which are derived from two GNN models  $f_{\mu}$  and  $f_{\sigma}$  on the top of the full graph  $\mathcal{G}$ ; similarly,  $\pmb{\mu}_2 = f_\mu (\mathcal{G}_s)$  and  $\log \sigma_2 = f_\sigma (\mathcal{G}_s)$  are on the top of the subgraph  $\mathcal{G}_s$ . Then, the decoder  $p(\mathcal{G}_s^* |\mathbf{Z})$  generates the valid surrogates:

$$
p \left(\mathcal {G} _ {s} ^ {*} | \mathbf {Z}\right) = \prod_ {i} ^ {N} \prod_ {j} ^ {N} p \left(A _ {i j} \mid \mathbf {z} _ {i}, \mathbf {z} _ {j}\right), \quad \text {w i t h} \quad p \left(A _ {i j} = 1 \mid \mathbf {z} _ {i}, \mathbf {z} _ {j}\right) = f _ {A} \left(\left[ \mathbf {z} _ {i}, \mathbf {z} _ {j} \right]\right), \tag {3}
$$

where  $A_{ij} = 1$  indicates the existence of an edge between nodes  $i$  and  $j$ ;  $f_{A}$  is a MLP, which takes the concatenation of node representations  $\mathbf{z}_i$  and  $\mathbf{z}_j$  as the input and outputs the probability of  $A_{ij} = 1$ .

Leveraging the variational graph auto-encoder is able to generate some counterfactual edges that never appear in  $\mathcal{G}$  and sample  $\mathcal{G}_s^*$  from the conditional distribution  $p(\mathcal{G}_s^*|\mathbf{Z})$ , formally,  $\mathcal{G}_s^* \sim p(\mathcal{G}_s^*|\mathbf{Z})$ . As a result,  $P(G_s^* = \mathcal{G}_s^* | G_s = \mathcal{G}_s)$  in Equation equation 1 is identified by  $p(\mathcal{G}_s^*|\mathbf{Z})$ . The quality of the generator directly affects the quality of the surrogate graphs, further determines how well the front-door adjustment is conducted. Next, we will detail an adversarial training framework to optimize the generator, which is distinct from the standard training of VAE.

Adversarial Training. To achieve high-quality generation, we get inspiration from the adversarial training (Goodfellow et al., 2020; Yue et al., 2021) and devise the following training objective:

$$
\min  _ {\theta} \mathcal {L} _ {\mathrm {V A E}} + \gamma \mathcal {L} _ {\mathrm {C}} + \max  _ {\mu} \omega \mathcal {L} _ {\mathrm {D}}, \tag {4}
$$

where  $\gamma, \omega$  are trade-off hyper-parameters. These losses are carefully designed to assure the generation follows the data distribution. Next, we will elaborate on each of them.

$$
\mathcal {L} _ {\mathrm {V A E}} = - \mathbb {E} _ {\mathcal {G}} \left[ \mathbb {E} _ {q (\mathbf {Z} | \mathcal {G}, \mathcal {G} _ {s})} [ \log p (\hat {\mathcal {G}} _ {\bar {s}} | \mathbf {Z}) ] \right] + \beta \mathbb {E} _ {\mathcal {G}} \left[ D _ {\mathrm {K L}} \left(q (\mathbf {Z} | \mathcal {G}, \mathcal {G} _ {s}) \| p (\mathbf {Z})\right) \right], \tag {5}
$$

We first minimize the  $\beta$ -VAE loss(Higgins et al., 2017), and the first term is the reconstruction loss responsible to predict the probability of edges' existence; the second term is the KL-divergence between the variational and prior distributions. Here we resort to the isotropic Gaussian distribution  $p(\mathbf{Z}) = \prod_{i} p(\mathbf{z}_{i}) = \prod_{i} \mathcal{N}(\mathbf{z}_{i} | \mathbf{0}, \mathbf{I})$  as the prior.  $\beta$  reweighs the KL-divergence, which promises to learn the disentangled factors in  $\mathbf{Z}$  (Higgins et al., 2017; Yue et al., 2021; Suter et al., 2019).

Moreover, we highlight the class-discriminative information in  $\mathbf{Z}$ , by encouraging the agreement between graph representations with the same class compared to that with different classes. Technically, the contrastive loss is adopted:

$$
\mathcal {L} _ {\mathrm {C}} = - \mathbb {E} _ {\mathcal {G}} \left[ \log \frac {\sum_ {\mathcal {G} ^ {\prime} \in \mathcal {B} _ {+}} \exp \left(s \left(\mathbf {z} _ {\mathcal {G}}, \mathbf {z} _ {\mathcal {G} ^ {\prime}}\right) / \tau\right)}{\sum_ {\mathcal {G} ^ {\prime \prime} \in \mathcal {B} _ {+} \cup \mathcal {B} _ {-}} \exp \left(s \left(\mathbf {z} _ {\mathcal {G}}, \mathbf {z} _ {\mathcal {G} ^ {\prime}}\right) / \tau\right)} \right], \tag {6}
$$

where  $\mathbf{z}_{\mathcal{G}}$  is the representation of  $\mathcal{G}$  that aggregates all node representations  $\mathbf{Z}$  together;  $s$  is the similarity function, which is given by an inner product here;  $\tau$  is the temperature hyper-parameter;  $\mathcal{B}_{+}$  is the graph set having the same class to  $\mathcal{G}$ , while the graphs involved in  $\mathcal{B}_{-}$  have different classes from  $\mathcal{G}$ . Minimizing this loss enables the generator to go beyond the generic knowledge and uncover the class-wise patterns of graph data.

Besides, we introduce a discriminative model  $d_{\mu}$  in addition, to criticize the generated graphs. Specifically, we set it as a probability-conditional GNN (Fey & Lenssen, 2019) parameterized by  $\mu$ . It takes a graph as input and outputs a score between 0 to 1, which indicates the confidence of the graph being realistic. Hence, given a real graph  $\mathcal{G}$  with the ground-truth label  $y$ , we can use the generator  $g_{\theta}$  to generate  $\mathcal{G}_s^*$ . Then the discriminator learns to assign  $\mathcal{G}$  with a large score while labeling  $\mathcal{G}_s^*$  with a small score. To optimize the discriminator, we adopt the Wasserstein GAN (WGAN) (Martin Arjovsky, 2017) loss:

$$
\mathcal {L} _ {\mathrm {D}} = \mathbb {E} _ {\mathcal {G}} \left[ \mathbb {E} _ {p (\mathcal {G} _ {s} ^ {*} | \mathbf {Z})} [ d (\mathcal {G}, y) - d (\mathcal {G} _ {s} ^ {*}, y) - \lambda \left(\left| \left| \nabla_ {\mathcal {G} _ {s} ^ {*}} d (\mathcal {G} _ {s} ^ {*}, y) \right| \right| _ {2} - 1\right) ^ {2} \right] \bigg ], \tag {7}
$$

where  $d(\mathcal{G}_s^*, y)$  is the probability of generating  $\mathcal{G}_s^*$  from the generator;  $\lambda$  is the hyper-parameter. By playing the min-max game between the generator and the discriminator in Equation equation 4, the generator can create the surrogate graphs from the data distribution plausibly.

Subgraph Evaluation. With the well-trained generator  $g_{\theta}^{*}$  whose parameters are fixed, we now approximate the causal effect of  $G_{s}$  on  $Y$ . Here we conduct Monte-Carlo simulation based on  $g_{\theta}^{*}$  to sample a set of plausible surrogate graphs  $\{\mathcal{G}_s^*\}$  from  $p(\mathcal{G}_s^*|\mathbf{Z})$ . Having collected the  $(\mathcal{G}_s,\mathcal{G}_s^*)$  data, we can arrive at the estimation of Equation equation 1.

# 4 EXPERIMENTS

We aim to answer the following research questions:

- Study of Explanation Evaluation. How effective is our DSE in mitigating the OOD effect and evaluating the explanatory subgraph more reliably? (Section 4.2)  
- Study of Generator. How effective is our CVGAE in generating the surrogates for the explanatory subgraphs and making them conform to the data distribution? (Section 4.3)

# 4.1 EXPERIMENTAL SETTINGS

Datasets & Target GNNs. We first train various target GNN classifiers on the three datasets:

- TR3 is a synthetic dataset involving 3000 graphs, each of which is constructed by connecting a random tree-shape base with one motif (house, cycle, crane). The motif type is the ground-truth label, while we treat the motifs as the ground-truth explanations following Ying et al. (2019); Yuan et al. (2020a). A Local Extremum GNN (Ranjan et al., 2019) is trained for classification.  
- MNIST superpixels  $\left(\mathbf{MNIST}_{\mathrm{sup}}\right)$  (Monti et al., 2017) converts the MNIST images into 70,000 superpixel graphs. Every graph with 75 nodes is labeled as one of 10 classes. We train a Spline-based GNN (Fey et al., 2018) as the classifier model. The subgraphs representing digits can be viewed as human explanations.  
- Graph-SST2 (Yuan et al., 2020b) is based on text sentiment dataset SST2 (Socher et al., 2013) and converts the text sentences to graphs where nodes represent tokens and edges indicate relations between nodes. Each graph is labeled by its sentence sentiment. The node embeddings are initialized by the pre-trained BERT word embeddings (Devlin et al., 2018). Graph Attention Network (Velicković et al., 2018) is trained as the classifier.

Ground-Truth Explanations. By "ground-truth", we follow the prior studies (Ying et al., 2019; Yuan et al., 2020a; Luo et al., 2020) and treat the subgraphs coherent to the model knowledge (e.g., the motif subgraphs in TR3) or human knowledge (e.g., the digit subgraphs in  $\mathsf{MNIST}_{\mathrm{sup}}$ ) as the ground-truth explanations. Although such ground-truth explanations might not fit the decision-making process of the model exactly, they contain sufficient discriminative information to help justify the explanations. Note that no ground-truth explanation is available in Graph-SST2.

Explainers. To explain the decisions made by these GNNs, we adopt several state-of-the-art explainers, including SA (Baldassarre & Azizpour, 2019), Grad-CAM (Selvaraju et al., 2017), GNN Explainer (Ying et al., 2019), CXPlain (Schwab & Karlen, 2019), PGM-Explainer (Vu & Thai, 2020), Screener (Anonymous, 2021), to generate the explanatory subgraphs. Specifically, top-15%, 20%, 20% of edges on the full graph instance construct the explanatory subgraphs in TR3, MNIST, and Graph-SST2, respectively. We refer readers to Appendix D for more experimental details.

# 4.2 STUDY OF EXPLANATION EVALUATION (RQ1)

Deconfounded Evaluation Performance. For an explanation  $\mathcal{G}_s$ , the conventional removal-based evaluation framework quantifies its importance as the subgraph-prediction correlation, termed  $\mathrm{Imp}_{\mathrm{re}}(\mathcal{G}_s) = f(\mathcal{G}_s)$ ; whereas, our DSE framework focuses on the causal effect caused by  $\mathcal{G}_s$  on  $Y$  which is computed based on Equation equation 1, and we denote it as  $\mathrm{Imp}_{\mathrm{dse}}(\mathcal{G}_s)$  for short. These importance scores broadly aim to reflect the discriminative information carried by  $\mathcal{G}_s$ . Thanks to the ground-truth knowledge available in TR3 and  $\mathrm{MNIST}_{\mathrm{sup}}$ , we are able to get a faithful and principled

![](images/ba360dd925e4f449b9961be3339d74cc0536a7449e06b25a619e2bfc9174dfca.jpg)  
(a) In TR3

![](images/9e3317b7d9a715fe8ee1ea1a6949609bcf72c46ef81021154d8f54abbecb4fde.jpg)  
Figure 4: Validation of different frameworks for explanation evaluation.  
(b) In MNISTsup

Table 1: Evaluation of explainers under different evaluation frameworks.  $R_{s}$  is Spearman rank correlation function. Best explainers are underlined. Symbol  $(\cdot)$  indicates the rank of explainers.  

<table><tr><td rowspan="2"></td><td colspan="3">TR3</td><td colspan="3">MNISTsup</td><td colspan="3">Graph-SST2</td></tr><tr><td>Impre(%)</td><td>Impdse(%)</td><td>Prec</td><td>Impre(%)</td><td>Impdse(%)</td><td>Prec</td><td>Impre(%)</td><td>Impdse(%)</td><td>Score</td></tr><tr><td>SA</td><td></td><td>43.23(1)</td><td>86.53(1)</td><td>17.60(3)</td><td>10.98(3)</td><td>32.98(2)</td><td>91.93(4)</td><td>95.67(4)</td><td>4.48(3)</td></tr><tr><td>Grad-CAM</td><td>33.07</td><td>43.18(2)</td><td>75.07(2)</td><td>16.90(5)</td><td>11.51(2)</td><td>31.42(3)</td><td>91.94(3)</td><td>96.21(2)</td><td>6.21(2)</td></tr><tr><td>GNNExplainer</td><td></td><td>41.73(3)</td><td>56.34(4)</td><td>17.00(4)</td><td>12.27(1)</td><td>57.75(1)</td><td>89.40(5)</td><td>95.20(6)</td><td>4.26(4)</td></tr><tr><td>CXPlain</td><td></td><td>38.61(6)</td><td>34.38(6)</td><td>14.30(6)</td><td>10.78(5)</td><td>11.14(5)</td><td>92.40(2)</td><td>95.98(3)</td><td>3.93(5)</td></tr><tr><td>PGM-Explainer</td><td>33.07</td><td>39.58(5)</td><td>48.47(5)</td><td>22.20(2)</td><td>10.77(6)</td><td>2.31(6)</td><td>89.16(6)</td><td>95.45(5)</td><td>1.68(6)</td></tr><tr><td>Screener</td><td></td><td>40.31(4)</td><td>66.49(3)</td><td>32.20(1)</td><td>10.96(4)</td><td>19.51(4)</td><td>96.04(1)</td><td>96.39(1)</td><td>6.42(1)</td></tr><tr><td>Rs↑</td><td>0.011</td><td>0.943*</td><td>-</td><td>-0.142</td><td>0.943*</td><td>-</td><td>0.657</td><td>0.714*</td><td>-</td></tr></table>

metric to measure the discriminative information amount — the precision  $\mathrm{Prec}(\mathcal{G}_s, \mathcal{G}_s^+)$  between the ground-truth explanation  $\mathcal{G}_s^+$  and the explanatory subgraph  $\mathcal{G}_s$ . This precision metric allows us to perform a fair comparison between  $\mathrm{Imp}_{\mathrm{re}}(\mathcal{G}_s)$  and  $\mathrm{Imp}_{\mathrm{dse}}(\mathcal{G}_s)$  via:

$$
\rho_ {\mathrm {r e}} = \rho \left(\left[ \operatorname {P r e c} \left(\mathcal {G} _ {s}, \mathcal {G} _ {s} ^ {+}\right) \right], \left[ \operatorname {I m p} _ {\mathrm {r e}} \left(\mathcal {G} _ {s}\right) \right]\right), \quad \rho_ {\mathrm {d s e}} = \rho \left(\left[ \operatorname {P r e c} \left(\mathcal {G} _ {s}, \mathcal {G} _ {s} ^ {+}\right) \right], \left[ \operatorname {I m p} _ {\mathrm {d s e}} \left(\mathcal {G} _ {s}\right) \right]\right), \tag {8}
$$

where  $\rho$  is the correlation coefficient between the lists of precision and importance scores. We present the results in Figure 4 and have some interesting insights:

- Insight 1: Removal-based evaluation hardly reflects the importance of explanations. In most cases,  $\operatorname{Prec}(\mathcal{G}_s, \mathcal{G}_s^+)$  is negatively correlated with the importance. This again shows that simply discarding a part of a graph could violate some underlying properties of graphs and mislead the target GNN, which is consistent with the adversarial attack works (Dai et al., 2018; Zügner et al., 2018). Moreover, the explainers that target high prediction accuracy, such as GNNExplainer, are easily distracted by the OOD effect and thus miss the important subgraphs.  
- Insight 2: Deconfounded evaluation quantifies the explanation importance more faithfully. Substantially,  $\rho_{\mathrm{dse}}$  greatly improves after the frontdoor adjustments via the surrogate variable. The most notable case is GNNExplainer in MNISTsup, where  $\rho_{\mathrm{dse}} = 0.17$  achieves a tremendous increase from  $\rho_{\mathrm{dse}} = -0.11$ . Although our DSE alleviates the OOD problem significantly, weak positive or negative correlations still exist, which indicates the limitation of the current CVGAE. We leave the exploration of higher-quality generation in future work.

Revisiting & Reranking Explainers. Here we investigate the rankings of explainers generated from different evaluation frameworks, and further compute the Spearman rank correlations between these evaluation rankings and the reference rankings of explainers. Specifically, for TR3 and  $\mathrm{MNIST}_{\mathrm{sup}}$  with ground-truth explanations, we regard the ranks w.r.t. precision as the references, while obtaining the reference of Graph-SST2 by a user study<sup>2</sup>. Such a reference offers the human knowledge for explanations and benchmarks the comparison. We show the results in Table 1 and conclude:

- Insight 3: DSE presents a more fair and reliable comparison among explainers. The DSE-based rankings are highly consistent with the references, while the removal-based rankings struggle to pass the check. In particular, we observe that for TR3, the unrealistic splicing inputs cause a plain ranking w.r.t.  $\mathrm{Imp}_{\mathrm{re}}$ . We find that various input subgraphs are predicted as cycle class. That is, the target GNN model is a deterministic gambler with serious OOD subgraphs. In contrast, DSE outputs a more informative ranking; For  $\mathrm{MNIST}_{\mathrm{sup}}$ ,  $\mathrm{GNNExplainer}$  with the highest precision

Table 2: Importance scores or probabilities of subgraphs before and after feature removal.  

<table><tr><td></td><td>TR3</td><td>MNISTsup</td><td>Graph-SST2</td></tr><tr><td>Imp(G) or GMM(G)</td><td>0.958-0.520</td><td>0.982-0.574</td><td>35.3-11.3</td></tr><tr><td>Imp(Gs+) or GMM(Gs)</td><td>0.438</td><td>0.408</td><td>24.0</td></tr></table>

Table 3: Performances of Generators in terms of Validity and Fidelity.  

<table><tr><td rowspan="2"></td><td colspan="3">TR3</td><td colspan="3">MNISTsup</td><td colspan="3">Graph-SST2</td></tr><tr><td>Imp(Gs*)</td><td>VAL↑</td><td>FID↓</td><td>Imp(Gs*)</td><td>VAL↑</td><td>FID↓</td><td>GMM(Gs*)</td><td>VAL↑</td><td>FID↓</td></tr><tr><td>Random</td><td>0.451</td><td>0.013</td><td>0.794</td><td>0.448</td><td>0.040</td><td>1.325</td><td>38.8</td><td>14.8</td><td>0.060</td></tr><tr><td>VGAE</td><td>0.469</td><td>0.031</td><td>0.754</td><td>0.205</td><td>-0.203</td><td>1.501</td><td>37.6</td><td>13.6</td><td>0.078</td></tr><tr><td>CVGAE</td><td>0.603</td><td>0.165</td><td>0.598</td><td>0.552</td><td>0.144</td><td>0.910</td><td>45.8</td><td>21.8</td><td>0.057</td></tr></table>

is overly underrated by the removal-based evaluation framework, but DSE justifies its position faithfully; For Graph-SST2, although the OOD problem seems to be minor, DSE can still achieve significant improvement.

Case Study. We present a case study in Graph-SST2 to illustrate how DSE mitigates the potential OOD problem. See Appendix F for another case study on TR3. In Figure 5,  $\mathcal{G}$  is a graph predicted as "negative" sentiment. The explanatory subgraph  $\mathcal{G}_s$  emphasizes tokens like "weak" and relations like "n't→funny", which is cogent according to human knowledge. However, its removal-based importance is highly underestimated as 0.385, possibly due to its disconnectivity or sparsity after feature removal. To mitigate the OOD problem, DSE samples 50 surrogate graphs from the generator, performs the frontdoor adjustment, and justifies the subgraph importance as 0.913, which shows the effectiveness of our DSE framework.

We also observe some limitations of the generator (1) Due to the limited training data, the generators only reflect the distribution of the observed graphs, thus making some

![](images/a310f7a04259e8ffd34832cd176d371aea4b7ea31a0e4de9c98abd92c5b498c8.jpg)  
Figure 5: A Case Example.

generations grammatically wrong. (2) The generations is constrained within the complete graph determined by the node set of the explanatory subgraph, thereby limits the quality of deconfounding. As we mainly focus on the OOD problem, we will leave the ability of the generator as future work.

# 4.3 STUDY OF GENERATORS (RQ2)

The generator plays an important role in our DSE framework, which aims to generate the valid surrogates conform to the data distribution. To evaluate the generator's quality, we compare it with two baselines: a random generator and a variational graph auto-encoder (VGAE) (Thomas N. Kipf, 2016). We perform the evaluation based on two metrics: (1) Validity. For the ground-truth explanations  $\mathcal{G}_s^+$  that contains all discriminative information of the full graph  $\mathcal{G}$ , the importance of its surrogate graph  $\mathcal{G}_s^*$  should be higher than itself. The difference between the two importance scores indicates the validity of the generator, thus we define  $\mathrm{VAL} = \mathbb{E}_{\mathcal{G}}[\mathrm{Imp}(\mathcal{G}_s^*) - \mathrm{Imp}(\mathcal{G}_s^+)]$ . For Graph-SST2 where the class-wise features are intractable, we leverage the embeddings of training graphs and additionally train a Gaussian Mixture Model (GMM) as our distribution prior. Then, we compute the average log-likelihood of random subgraphs after in-filling, thus we have  $\mathrm{VAL} = \mathbb{E}_{\mathcal{G}}\mathbb{E}_{\mathcal{G}_s\sim \mathrm{Random}(\mathcal{G})}[\mathrm{GMM}(\mathcal{G}_s^*) - \mathrm{GMM}(\mathcal{G}_s)]$ . (2) Fidelity. Towards a finer-grained assessment w.r.t. prediction probability of any random subgraphs, we adopt the metric following (Frye et al., 2021):  $\mathrm{FID} = \mathbb{E}_{\mathcal{G}}\mathbb{E}_{\mathcal{G}_s}\mathbb{E}_y|f_y(\mathcal{G}) - \mathbb{E}_{\mathcal{G}_s^*}[f_y(\mathcal{G}_s^*)]|^2$ . This measures how well the surrogates cover the target prediction distribution.

Before comparing different generators, we first compute the importance or probabilities of the graphs before and after feature removal, which are summarized in Table 2. When inspecting the Removal's results without any in-fills, the OOD problem is severe: in TR3 and  $\mathrm{MNIST}_{\mathrm{sup}}$ , the importance of ground-truth subgraphs only reaches  $43.8\%$  and  $40.8\%$ , respectively, which are far away from the target importance of full graphs. Analogously in Graph-SST2. For the performance of the generators w.r.t. the two metrics, we summarize the average results over 5 runs in Table 3:

- The replacements generated by Random and VGAE are suboptimal. This suggests that they can hardly fit the target conditional distribution.  
- CVGAE outperforms other generators consistently across all cases, thus justifying the rationale and effectiveness of our proposed generator and adversarial training paradigm. For example, in TR3, CVGAE significantly increases the VAL scores and mitigates the OOD effect effectively.

Moreover, we conduct ablation studies to investigate the contribution of each component loss. The results are shown in Appendix G which validate the effectiveness of the designed objective.

# 5 RELATED WORK

Post-hoc Explainability of GNNs. Inspired by the explainability in computer vision, researchers transfer gradient-based methods to the graph domain (Baldassarre & Azizpour, 2019; Pope et al., 2019; Schnake et al., 2020), which is to obtain the gradient-like scores of the model's outcome or loss w.r.t. the input features. Another line (Luo et al., 2020; Ying et al., 2019; Yuan et al., 2020a; Yue Zhang, 2020; Michael Sejr Schlichtkrull, 2021) is to learn the masks on graph features. Typically, GNN-Explainer (Ying et al., 2019) applies the instance-wise masks on the messages carried by graph structures, and maximizes the mutual information between the masked graph and the prediction. Going beyond the instance-wise explanation, PGExplainer (Luo et al., 2020) uses a generative probabilistic model to generate masks for multiple instances inductively. Recently, researchers start to adopt the causal explainability (Pearl & Mackenzie, 2018) to uncover the causation of the model predictions. A common scheme is to perturb input features and audit the changes of model behaviors. For instance, CXPlain (Schwab & Karlen, 2019) quantifies a feature's importance by leaving it out. PGM-Explainer (Vu & Thai, 2020) performs perturbations on graph structures to build an Bayesian network. PGM-Explainer (Vu & Thai, 2020) performs perturbations on graph structures and builds an Bayesian network upon the perturbation-prediction pairs. Causal Screening (Screener) (Anonymous, 2021) measures the importance of an edge as its causal effect, conditional on the previously selected structures. Lately, SubgraphX (Yuan et al., 2021) explores different subgraphs with Monte-Carlo tree search and evaluates subgraphs with the Shapley value (Kuhn & Tucker, 1953).

Counterfactual Generation for the OOD Problem. The OOD effect of feature removal has been investigated in some other domains. There are generally two classes of generation (i) Static generation. For example, Fong & Vedaldi. (2017); Dabkowski & Gal (2017) adopted blurred input and random colors for the image reference, respectively. Due to the unnatural in-filling, the generated images are distributional irrespective and can still introduce confounding bias. (ii) Adaptive generation: Chang et al. (2019); Frye et al. (2021); Agarwal et al. (2019); Kim et al. (2020). The generators of these methods, like DSE, overcomes the defects aforementioned, whose generated data conforms to the training distribution. For example, in computer vision, FIDO (Chang et al., 2019) generates image-specific explanations that respect the data distribution, answering "Which region, when replaced by plausible alternative values, would maximally change classifier output?".

For the difference, firstly, DSE's formulated importance involves additional adjustment on  $G_{s}$  and guarantees the unbiasedness of introducing the surrogate variable  $G_{s}^{*}$ , which is commonly discarded by the prior works with in-fillings only. Specifically, we offer a comparison with FIDO in Appendix B. Secondly, the distribution of graph data is more complicated to model than other domains. And the proposed CVGAE is carefully designed for graph data, where the contrastive loss and the adversarial training framework are shown to be effective for learning the data distribution of graphs.

# 6 CONCLUSION

In this work, we investigate the OOD effect on the explanation evaluation of GNNs. With a causal view, we uncover the OOD effect — the distribution shift between full graphs and subgraphs, as the confounder between the explanatory subgraphs and the model prediction, making the evaluation less reliable. To mitigate it, we propose a deconfounding evaluation framework that exploits the front-door adjustment to measure the causal effect of the explanatory subgraphs on the model prediction. And a deep generative model is devised to achieve the front-door adjustment by generating in-distribution surrogates of the subgraphs. In-so-doing, we can reliably evaluate the explanatory subgraphs. As the evaluation for explanations fundamentally guides the objective in GNNs explainability, this work offers in-depth insights into the future interpretability systems.

# ETHICS STATEMENT

This work raises concerns about the removal-based evaluation in the explainability literature and proposed Deconfounded Subgraph Evaluator. For the user study that involves human subjects, we have detailed the fair evaluation procedure for each explanation generated by the explainers in Appendix E. For real-world applications, we admitted that the modeling of the distribution shift could be a barrier to fulfill their evaluation faithfulness. However, as shown in the paper, improper evaluation under the OOD setting largely biases the inspection of the model's decision-making process and the quality of explainers. Therefore, we argue that explainability should exhibit faithful explanation evaluation before auditing deep models' actual decision-making process. And a wrongly evaluated explanation might do more significant harm than an incorrect prediction, as the former could affect the general adjustment (e.g., structure construction) and human perspective (e.g., fairness check) of the model.

# REPRODUCIBILITY STATEMENT

We have made great efforts to ensure reproducibility in this paper. Firstly, we make all causal assumptions clear in Section 2.2, Section 3.1 and Appendix A. For datasets, we have released the synthetic dataset, which can be referred to the link in Section 1, while the other two datasets are publicly available. We also include our code for model construction in the link. In Appendix D, we have reported the settings of hyper-parameters used in our implementation for model training.

# REFERENCES

Chirag Agarwal, Dan Schonfeld, and Anh Nguyen. Removing input features via a generative model to explain their attributions to classifier's decisions. CoRR, 2019.  
Anonymous. Causal screening to interpret graph neural networks. in Submitted to ICLR. https://openreview.net/forum?id=nzKv5vxZfge, 2021.  
Federico Baldassarre and Hossein Azizpour. Explainability techniques for graph convolutional networks. CoRR, abs/1905.13686, 2019.  
Chun-Hao Chang, Elliot Creager, Anna Goldenberg, and David Duvenaud. Explaining image classifiers by counterfactual generation. In ICLR, 2019.  
Ian Covert, Scott Lundberg, and Su-In Lee. Feature removal is a unifying principle for model explanation methods. In NeurIPS, 2020.  
Piotr Dabkowski and Yarin Gal. Real time image saliency for black box classifiers. In NeurIPS, pp. 6967-6976, 2017.  
Hanjun Dai, Hui Li, Tian Tian, Xin Huang, Lin Wang, Jun Zhu, and Le Song. Adversarial attack on graph structured data. In ICML, pp. 1123-1132, 2018.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, and Xavier Bresson. Benchmarking graph neural networks. CoRR, abs/2003.00982, 2020.  
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.  
Matthias Fey, Jan Eric Lenssen, Frank Weichert, and Heinrich Müller. Splineconn: Fast geometric deep learning with continuous b-spline kernels. In CVPR, pp. 869-877, 2018.  
R. Fong and A. Vedaldi. Interpretable explanations of black boxes by meaningful perturbation. In ICCV, 2017.

Christopher Frye, Damien de Mijolla, Laurence Cowton, Megan Stanley, and Ilya Feige. Shapley-based explainability on the data manifold. CoRR, abs/2006.01272, 2020. URL https://arxiv.org/abs/2006.01272.  
Christopher Frye, Damien de Mijolla, Laurence Cowton, Megan Stanley, and Ilya Feige. Shapley-based explainability on the data manifold. In *ICLR*, 2021.  
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. Commun. ACM, 63(11): 139-144, 2020.  
William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NeurlPS, pp. 1024-1034, 2017.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In ICLR, 2017.  
P.W. Holland. Causal inference, path analysis, and recursive structural equations models. C. Clogg, editor, Sociological Methodology, pages 449-484. American Sociological Association, Washington, D.C., 1988.  
Siwon Kim, Jihun Yi, Eunji Kim, and Sungroh Yoon. Interpretation of NLP models through input marginalization. In EMNLP, pp. 3154-3167, 2020.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
H. W. Kuhn and A. W. Tucker. Contributions to the theory of games, volume 2. Princeton University Press, 1953.  
Jure Leskovec, Jon Kleinberg, and Christos Faloutsos. Graphs over time: densification laws, shrinking diameters and possible explanations. In KDD, pp. 177-187, 2005.  
Qi Liu, Miltiadis Allamanis, Marc Brockschmidt, and Alexander L. Gaunt. Constrained graph variational autoencoders for molecule design. In NeurIPS, pp. 7806-7815, 2018.  
Roger Wattenhofer Lukas Faber, Amin K. Moghaddam. Contrastive graph neural network explanation. In ICLR Workshop on Representation Learning, 2021.  
Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang. Parameterized explainer for graph neural network. In NeurIPS, 2020.  
Léon Bottou Martin Arjovsky, Soumith Chintala. Wasserstein generative adversarial networks. In ICML, 2017.  
Ivan Titov Michael Sejr Schlichtkrull, Nicola De Cao. Interpreting graph neural networks for nlp with differentiable edge masking. In ICLR, 2021.  
Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, and Michael M. Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns. In CVPR, pp. 5425-5434, 2017.  
Judea Pearl. Causality: Models, Reasoning, and Inference. 2000.  
Judea Pearl and Dana Mackenzie. The book of why: the new science of cause and effect. Basic Books, 2018.  
Judea Pearl, Madelyn Glymour, and Nicholas P Jewell. Causal inference in statistics: A primer. John Wiley & Sons, 2016.  
Phillip E. Pope, Soheil Kolouri, Mohammad Rostami, Charles E. Martin, and Heiko Hoffmann. Explainability methods for graph convolutional neural networks. In CVPR, pp. 10772-10781, 2019.

Ekagra Ranjan, Soumya Sanyal, and Partha Pratim Talukdar. ASAP: Adaptive structure aware pooling for learning hierarchical graph representations. arXiv preprint arXiv:1911.07979, 2019.  
Ekagra Ranjan, Soumya Sanyal, and Partha P. Talukdar. ASAP: adaptive structure aware pooling for learning hierarchical graph representations. In AAAI, pp. 5470-5477, 2020.  
Thomas Schnake, Oliver Eberle, Jonas Lederer, Shinichi Nakajima, K. T. Schutt, Klaus-Robert Muller, and Gregoire Montavon. Higher-order explanations of graph neural networks via relevant walks. arXiv, 2020.  
Patrick Schwab and Walter Karlen. Cxplain: Causal explanations for model interpretation under uncertainty. In NeurIPS, pp. 10220-10230, 2019.  
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV, pp. 618-626, 2017.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In EMNLP, pp. 1631-1642, 2013.  
Raphael Suter, Doröe Miladinovic, Bernhard Schölkopf, and Stefan Bauer. Robustly disentangled causal mechanisms: Validating deep representations for interventional robustness. In ICML, volume 97, pp. 6056-6065, 2019.  
Max Welling Thomas N. Kipf. Variational graph auto-encoders. In NeurIPS Workshops, 2016.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ. accepted as poster.  
Minh N. Vu and My T. Thai. Pgm-explainer: Probabilistic graphical model explanations for graph neural networks. In NeurIPS, 2020.  
Tian Xie and Jeffrey C. Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Phys. Rev. Lett., 120:145301, Apr 2018. doi: 10.1103/PhysRevLett.120.145301. URL https://link.aps.org/doi/10.1103/PhysRevLett.120.145301.  
Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Generating explanations for graph neural networks. In NeurIPS, pp. 9240-9251, 2019.  
Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. XGNN: towards model-level explanations of graph neural networks. In Rajesh Gupta, Yan Liu, Jiliang Tang, and B. Aditya Prakash (eds.), KDD, pp. 430-438, 2020a.  
Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A taxonomic survey. CoRR, 2020b.  
Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural networks via subgraph explorations. ArXiv, 2021.  
Zhongqi Yue, Tan Wang, Hanwang Zhang, Qianru Sun, and Xian-Sheng Hua. Counterfactual zero-shot and open-set visual recognition. In CVPR, 2021.  
Arti Ramesh Yue Zhang, David Defazio. Relex: A model-agnostic relational model explainer. arXiv preprint arXiv:2006.00305, 2020.  
Daniel Zügner, Amir Akbarnejad, and Stephan Gunnemann. Adversarial attacks on neural networks for graph data. In KDD, pp. 2847-2856, 2018.
