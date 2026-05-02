# INTERPRETABLE AND GENERALIZABLE GRAPH NEURAL NETWORKS VIA SUBGRAPH MULTILINEAR EXTENSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Interpretable graph neural networks (XGNNs) are widely adopted in scientific applications involving graph-structured data. Previous approaches predominantly adopt the attention-based mechanism to learn edge or node importance for extracting and making predictions with the interpretable subgraph. However, the representational properties and limitations of these methods remain inadequately explored. In this work, we present a theoretical framework that formulates interpretable subgraph learning with the multilinear extension of the subgraph distribution, which we term as subgraph multilinear extension (SubMT). Extracting the desired interpretable subgraph requires an accurate approximation of SubMT, yet we find that the existing XGNNs can have a huge gap in fitting SubMT. Consequently, the SubMT approximation failure will lead to the degenerated interpretability of the extracted subgraphs. To mitigate the issue, we design a new XGNN architecture called Graph Multilinear neT (GMT), which is provably more powerful in approximating SubMT. We empirically validate our theoretical findings on a number of graph classification benchmarks. The results demonstrate that GMT outperforms the state-of-the-art up to  $10\%$  in terms of both interpretability and generalizability measures.

# 1 INTRODUCTION

Graph Neural Networks (GNNs) have been widely used in scientific applications (Wang et al., 2023; Zhang et al., 2023) such as Physics (Bapst et al., 2020), Chemistry (Gilmer et al., 2017; Jumper et al., 2021), Quantum mechanics (Kochkov et al., 2021), Materials (Schütt et al., 2017) and Cosmology (Villanueva-Domingo et al., 2021). In pursuit of scientific discoveries, it often requires GNNs to be able to generalize to unseen or Out-of-Distribution (OOD) graphs (Gui et al., 2022; Ji et al., 2022; Zhang et al., 2023), and also provide interpretations of the predictions that are crucial for scientists to collect insights (Xie & Grossman, 2017; Cranmer et al., 2020; Dai et al., 2021) and promote better scientific practice (Murray & Rees, 2009; Wencel-Delord & Glorius, 2013). Recently there has been a surge of interest in developing intrinsically interpretable and generalizable GNNs (XGNNs) (Yu et al., 2021; Miao et al., 2022; Wu et al., 2022b; Chen et al., 2022a; Miao et al., 2023). In contrast to post-hoc explanation approaches (Ying et al., 2019; Yuan et al., 2020a; Vu & Thai, 2020; Luo et al., 2020; Yuan et al., 2021; Lin et al., 2021; 2022a) which is shown to be suboptimal in interpretation and sensitive to pre-trained GNNs performance (Miao et al., 2022; 2023), XGNNs are able to provide both reliable explanations and (OOD) generalizable predictions under the proper guidance such as information bottleneck (Yu et al., 2021) and causality (Chen et al., 2022a).

Indeed, the faithful interpretation and the reliable generalization are the two sides of the same coin for XGNNs. Grounded in the causal assumptions of data generation processes (Wu et al., 2022a; Miao et al., 2022; Wu et al., 2022b), XGNNs assume that there exists a causal subgraph which holds a causal relation with the target label. Predictions made solely based on the causal subgraph are generalizable under various graph distribution shifts (Chen et al., 2022a). Therefore, XGNNs typically adopt the two-step paradigm that first extracts a subgraph of the input graphs and then predicts the labels. To circumvent the inherent discreteness of subgraphs, XGNNs often learn the sampling probability for each edge or node with the attention mechanism and extract the subgraph with high attention scores (Miao et al., 2022). Predictions are then made via a weighted message passing scheme with the attention scores (Wu et al., 2022b; Chen et al., 2022a). Despite the notable

![](images/74260c02508452e820232b829877a6a76d0404bace9b53fdb12e5997bf192fd0.jpg)

![](images/cbf017c56485851e748929c9e673877982e199d009012ff00dc2819e690f45cd.jpg)  
Figure 1: Illustration of Subgraph Multilinear Extension (SubMT). The task is to classify whether a graph contains a specific "house" or "cycle" motif. An XGNN  $f = f_{c} \circ g$  predicts the label with the classifier  $f_{c}$  based on the extracted soft subgraph  $\widehat{G}_{c} = g(G)$ , denoted as the central graph. Different depths of edge colors refer to the sampling probability of the edge.  $\widehat{G}_{c} = g(G)$  corresponds to a subgraph distribution where the sampling probability for each subgraph  $G_{c}$  (i.e., subgraphs with solid lines in the figure). SubMT extends GNNs to accept soft subgraph inputs by estimating the subgraph conditional prediction as the expectation of each possible subgraph  $\mathbb{E}[f_c(G_c)]$ . Interpretable subgraph learning requires accurate estimation of SubMT. Yet existing XGNNs that directly take the soft subgraph  $\widehat{G}_{c}$  as classifier GNN inputs fail to reliably estimate SubMT. In contrast, GMT aims to bridge the gap between SubMT and  $f_{c}(\widehat{G}_{c})$  by learning a neural SubMT to align with SubMT.

success of the paradigm in enhancing both interpretability and out-of-distribution (OOD) (Miao et al., 2022; 2023; Chen et al., 2022a), there is little theoretical understanding of the representational properties and limitations of XGNNs, and whether they can provide faithful interpretations.

In this work, we present a framework to analyze the expressiveness and evaluate the faithfulness of XGNNs. Our framework is inspired by the close connection between interpretable subgraph learning and multilinear extension, a powerful tool for solving classical combinatorial optimization problems (Calinescu et al., 2007). In fact, the subgraph learning in XGNNs naturally resembles the multilinear extension of the subgraph predictivity, which we termed as subgraph multilinear extension (SubMT). Extracting the truth interpretable subgraph requires a precise approximation of SubMT. However, we show that the prevalent attention-based paradigm can fail to reliably approximate SubMT (Sec. 3.2). Consequently, the SubMT approximation failure will decrease the interpretability of the subgraph for predicting the target label. More specifically, we instantiate the issue via a causal framework and propose a novel interpretability measure called counterfactual fidelity, i.e., the sensitivity of the prediction with respect to small perturbations to the extracted subgraphs (Sec. 4.2). Although faithful interpretation should have a high counterfactual fidelity with the prediction, we find that XGNNs implemented with the prevalent paradigm only have a low counterfactual fidelity.

Aiming to bridge the gap, we propose a simple yet effective XGNN architecture called Graph Multilinear neT (GMT). The core design of GMT is inspired by the SubMT formulation, which performs random subgraph sampling to reduce the SubMT approximation error. We prove that GMT is provably more powerful in approximating SubMT (Sec. 5). We validate our theoretical findings through extensive experiments on multiple graph classification benchmarks. The results demonstrate that GMT improves the state-of-the-art up to  $10\%$  in both interpretability and generalizability (Sec. 6).

# 2 PRELIMINARIES AND RELATED WORK

We begin by introducing preliminary concepts of XGNNs and leave more details to Appendix B.1. For ease of understanding, a table of the notations for key concepts is given in Appendix A.

Interpretable GNNs. Let  $G = (A,X)$  be a graph with node set  $V = \{v_{1},v_{2},\dots,v_{n}\}$  and edge set  $E = \{e_1,e_2,\ldots ,e_m\}$ , where  $A\in \{0,1\}^{n\times n}$  is the adjacency matrix and  $X\in \mathbb{R}^{n\times d}$  is the node feature matrix. In this work, we focus on interpretable GNNs (or XGNNs) for the graph classification

task, while the results can be generalized to node-level tasks as well (Wu et al., 2020). Given each sample from training data  $\mathcal{D}_{\mathrm{tr}} = (G^i,Y^i)$ , an interpretable GNN  $f\coloneqq h\circ g$  aims to identify a (causal) subgraph  $G_{c}\subseteq G$  via a subgraph extractor GNN  $g:\mathcal{G}\to \mathcal{G}_c$ , and then predicts the label via a subgraph classifier GNN  $f_{c}:\mathcal{G}_{c}\rightarrow \mathcal{V}$ , where  $\mathcal{G},\mathcal{G}_c,\mathcal{V}$  are the spaces of graphs, subgraphs, and the labels, respectively (Yu et al., 2021). Although post-hoc explanation approaches also aim to find an interpretable subgraph as the explanation for the model prediction (Ying et al., 2019; Yuan et al., 2020a; Vu & Thai, 2020; Luo et al., 2020; Yuan et al., 2021; Lin et al., 2021; 2022a), they are shown to be suboptimal in interpretation performance and sensitive to the performance of the pre-trained GNNs (Miao et al., 2022). Therefore, this work focuses on intrinsic interpretable GNNs (XGNNs).

A predominant approach to implement XGNNs is to incorporate the idea of information bottleneck (Tishby et al., 1999), such that  $G_{c}$  keeps the minimal sufficient information of  $G$  about  $Y$  (Yu et al., 2021; 2022; Miao et al., 2022; 2023; Yang et al., 2023), which can be formulated as

$$
\max  _ {G _ {c}} I \left(G _ {c}; Y\right) - \lambda I \left(G _ {c}; G\right), G _ {c} \sim g (G), \tag {1}
$$

where the maximizing  $I(G_{c};Y)$  endows the interpretability of  $G_{c}$  while minimizing  $I(G_{c};G)$  ensures  $G_{c}$  captures only the most necessary information,  $\lambda$  is a hyperparameter trade off between the two objectives. In addition to minimizing  $I(G_{c};G)$ , there are also alternative approaches that impose different constraints such as causal invariance (Chen et al., 2022a; Li et al., 2022) or disentanglement (Wu et al., 2022b; Sui et al., 2022; Liu et al., 2022a; Fan et al., 2022) to identify the desired subgraphs. When extracting the subgraph, XGNNs adopts the attention mechanism to learn the sampling probability of each edge or node, which avoids the complicated Monte Carlo tree search used in other alternative implementations (Zhang et al., 2022). Specifically, given node representation learned by message passing  $H_{i}\in \mathbb{R}^{h}$  for each node  $i$ , XGNNs either learns a node attention  $\alpha_{i}\in \mathbb{R}_{+} = \sigma (a(H_{i}))$  via the attention function  $a:\mathbb{R}^h\to \mathbb{R}_+$ , or the edge attention  $\alpha_{e}\in \mathbb{R}_{+} = \sigma (a([H_{u},H_{v}]))$  for each edge  $e = (u,v)$  via the attention function  $a:\mathbb{R}^{2h}\to \mathbb{R}_{+}$ , where  $\sigma (\cdot)$  is a sigmoid function.  $\pmb{\alpha} = [\alpha_{1},\dots,\alpha_{m}]^{T}$  essentially elicits a subgraph distribution of the interpretable subgraph. In this work, we focus on edge-centric subgraph sampling as it is most widely used in XGNNs while our method can be easily generalized to node-centric approaches.

Faithful interpretation and (OOD) generalization. The faithfulness of interpretation is critical to all interpretable and explainable methods (Ribeiro et al., 2016; Lipton, 2018; Alvarez-Melis & Jaakkola, 2018; Jain & Wallace, 2019). There are several metrics developed to measure the faithfulness of graph explanations, such as fidelity (Yuan et al., 2020b; Amara et al., 2022), counterfactual robustness (Bajaj et al., 2021; Prado-Romero et al., 2022; Ma et al., 2022), and equivalence (Crabbé & van der Schaar, 2023), which are however limited to post-hoc graph explanation methods. In contrast, we develop the first faithfulness measure for XGNNs in terms of counterfactual invariance.

In fact, the generalization ability and the faithfulness of the interpretation are naturally intertwined in XGNNs. XGNNs need to extract the underlying ground-truth subgraph in order to make correct predictions on unseen graphs (Miao et al., 2022). When distribution shifts are present during testing, the underlying subgraph that has a causal relationship with the target label (or causal subgraphs) naturally becomes the ground-truth subgraph that needs to be learned by XGNNs (Chen et al., 2022a).

Multilinear extension serves as a powerful tool for maximizing combinatorial functions, especially for submodular set function maximization (Calinescu et al., 2007; Vondrak, 2008; Bian et al., 2019; Sahin et al., 2020; Karalias et al., 2022). It is the expected value of a set function under the fully factorized Bernoulli distribution. In this work, we are the first to identify subgraph multilinear extension as the factorized subgraph distribution for interpretable subgraph learning.

# 3 ON THE EXPRESSIVITY OF INTERPRETABLE GNNS

In this section, we present our theoretical framework for characterizing the expressivity of XGNNs. Since all of the existing approaches need to maximize  $I(G_{c};Y)$  regardless of the regularization on  $G_{c}$ , we focus on the modeling of the subgraph distribution that maximizes  $I(G_{c};Y)$ .

# 3.1 SUBGRAPH MULTILINEAR EXTENSION.

The need for maximizing  $I(G_{c};Y)$  originates from extracting information in  $G$  to predict  $Y$  with  $f_{c}$ ,

$$
\arg \max  _ {f _ {c}} I (G; Y) = \arg \max  _ {f _ {c}} [ H (Y) - H (Y | G) ] = \arg \min  _ {f _ {c}} H (Y | G), \tag {2}
$$

where the last equality is due to the irrelevance of  $H(Y)$  and  $f_{c}$ . For each sample  $(G, Y)$ , XGNN then adopts the subgraph extractor  $g$  to extract a subgraph  $G_{c} \sim g(G)$ , and take  $G_{c}$  as the input of  $f_{c}$  to predict  $Y$ . Then, Eq. 2 is realized as follows: let  $L(\cdot)$  be the cross-entropy loss, then

$$
\arg \min  _ {g, f _ {c}} \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} [ - \log P (Y | \mathbb {E} _ {G _ {c} ^ {g} G} G _ {c}) ] = \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} [ L (f _ {c} (\boldsymbol {\alpha}; G), Y) ], \tag {3}
$$

where  $\alpha \in \mathbb{R}_+^m$  is the attention score elicited from the subgraph extractor  $g$ . We leave more details about the deduction of Eq. 3 in Appendix B.2. Note that  $f_c$  is a GNN defined only for discrete graph-structured inputs (i.e.,  $\alpha \in \{0,1\}^m$ ), while Eq. 3 imposes continuous inputs to  $f_c$ . Considering  $f_c(G_c)$  is a set function with respect to node/edge index subsets of  $G$  (i.e., subgraphs  $G_c$ ), and the parameterization of  $P(G)$  in XGNNs (Miao et al., 2022), we resort to the multilinear extension of  $f_c(G_c)$ . Multilinear extension for set functions has been extensively studied in the domain of solving classical combinatorial optimization problems (Calinescu et al., 2007; Karalias et al., 2022).

Definition 3.1 (Subgraph multilinear extension (SubMT)). Given the attention score  $\alpha \in [0,1]^m$  as sampling probability of  $G_{c}$ , XGNNs factorize  $P(G)$  as independent Bernoulli distributions on edges:

$$
P (G _ {c} | G) = \prod_ {e \in G _ {c}} \alpha_ {e} \prod_ {e \in G / G _ {c}} (1 - \alpha_ {e}),
$$

which elicits the multilinear extension of  $f_{c}(G_{c})$  in Eq. 3 as:

$$
F _ {c} (\boldsymbol {\alpha}; G) := \sum_ {G _ {c} \in G} f _ {c} \left(G _ {c}\right) \prod_ {e \in G _ {c}} \alpha_ {e} \prod_ {e \in G / G _ {c}} \left(1 - \alpha_ {e}\right) = \mathbb {E} _ {G _ {c} \sim G} f _ {c} \left(G _ {c}\right). \tag {4}
$$

The parameterization of  $P(G)$  is widely employed in XGNNs (Miao et al., 2022; Chen et al., 2022a), which implicitly assumes the random graph data model (Erdos & Renyi, 1984). Def. 3.1 can also be generalized to other graph models with the corresponding parameterization of  $P(G)$  (Snijders & Nowicki, 1997; Lovász & Szegedy, 2006). When a XGNN approximates SubMT well, we have:

Definition 3.2 ( $\epsilon$ -SubMT approximation). Let  $d(\cdot, \cdot)$  be a distribution distance metric, a XGNN  $f = f_{c} \circ g$ $\epsilon$ -approximates SubMT (Def. 3.1), if there exists  $\epsilon \in \mathbb{R}_{+}$  such that  $d(P_{f}(Y|G), P(Y|G)) \leq \epsilon$  where  $P(Y|G) \in \mathbb{R}^{|\mathcal{V}|}$  is the ground truth conditional label distribution, and  $P_{f}(Y|G) \in \mathbb{R}^{|\mathcal{V}|}$  is the predicted label distribution for  $G$  via a XGNN  $f$ , i.e.,  $P_{f}(Y|G) = f_{c}(\mathbb{E}_{G_{c} \stackrel{\sim}{G}} G_{c})$ .

Def. 3.2 is a natural requirement for XGNN that approximates SubMT properly. With the definition of SubMT, we can write the problem of Eq. 3 as the following:

$$
\arg \min  _ {g, f _ {c}} \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} [ L (\mathbb {E} _ {G _ {c} \stackrel {g} {\sim} G} f _ {c} (G _ {c}), Y) ] = \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} L (F _ {c} (\boldsymbol {\alpha}; G), Y), \tag {5}
$$

Intuitively, optimizing for  $g$ ,  $f_{c}$  in Eq. 3 requires an accurate estimation of SubMT.

# 3.2 ISSUES OF EXISTING APPROACHES

In general, evaluating SubMT requires  $\mathcal{O}(2^m)$  calls of Eq. 4. However, existing XGNNs take a "shortcut" and introduce a soft subgraph  $\widehat{G}_c$  with the adjacency matrix as the attention matrix  $\widehat{A}$  where  $A_{u,v} = \alpha_e$ ,  $\forall e = (u,v) \in E$ , to estimate Eq. 3 via weighted message passing (Miao et al., 2022):

$$
\arg \min  _ {g, f _ {c}} \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} [ L (\mathbb {E} _ {G _ {c} ^ {g} \sim_ {G} f _ {c}} (G _ {c}), Y) ] = \mathbb {E} _ {(G, Y) \sim \mathcal {D} _ {\mathrm {t r}}} [ L (f _ {c} (\widehat {G} _ {c}), Y) ]. \tag {6}
$$

From the edge-centric perspective, the introduction of  $\widehat{G}_c$  seems to be natural at first glance, as:

$$
\widehat {G} _ {c} = \mathbb {E} _ {G _ {c} \stackrel {g} {\sim} G} G _ {c} = (X, \widehat {A}). \tag {7}
$$

However, Eq. 6 holds only when  $f_{c}$  is linear. More formally, as  $X$  is fixed, for the sake of brevity, let  $f_{c}(A) \coloneqq \mathbb{E}_{G_{c}}\mathcal{L}_{G}[f_{c}((X,A))]$ , then Eq. 6 requires the following to be hold:

$$
f _ {c} (\widehat {A}) = f _ {c} (\mathbb {E} [ A ]) = \mathbb {E} [ f _ {c} (A) ], \tag {8}
$$

where the first equality is by the definition of  $\widehat{A}$ , while the last equality adheres to the equality of Eq. 6. Obviously  $f_{c}(\cdot)$  is a non-linear function even with a linearized GNN (Wu et al., 2019) with linear activations and pooling such as sum pooling, which can be written as:

$$
f _ {c} \left(G _ {c}\right) = \rho \left(\widehat {A} ^ {k} X W\right), \tag {9}
$$

where  $\rho$  is the pooling,  $k$  is the number of layers and  $\mathbf{W} \in \mathbb{R}^{h \times h}$  are the learnable weights. Therefore,

![](images/ce7c7f5022e7b1ccb0e98f345c71f8681f411ef482533d7f8e5c8fded87fd6fa.jpg)  
(a) SCM of XGNNs.

![](images/93126ebc7fde8783aa6b86dce19a14070f238f3e2f98a25f25e83235c76a8367.jpg)  
(b) SubMT on BA-2Motifs.

![](images/b1caaceec582f5db8ad4f925e766a018ca4289ec8c074a8369acdff95dbb41ef.jpg)  
Figure 2: Illustration of counterfactual faithfulness.  
(c) SubMT on Mutag.

Proposition 3.3. Eq. 8 with linear GNNs (Eq. 9) and  $k > 1$  can not approximate SubMT (Def. 3.1).

The proof is given in Appendix D.2. Empirical verifications are also provided in Appendix F.6.. For example, when  $k = 2$  and  $|\mathcal{V}| = 1$ , Eq. 9 is convex, and we have  $f_{c}(\mathbb{E}[A]) \leq \mathbb{E}[f_{c}(A)]$  due to Jensen's inequality, which introduces the Jensen gap as  $\mathbb{E}[f_c(A)] - f_c(\widehat{A})$  when fitting SubMT.

# 4 ON THE GENERALIZATION AND INTERPRETABILITY: A CAUSAL VIEW

To further understand the consequences of the SubMT approximation issue, we conduct a causal analysis of the interpretation faithfulness in XGNNs. Without loss of generality, our discussion focuses on the edge-centric view of data generation and interpretation.

# 4.1 CAUSAL MODEL OF INTERPRETABLE GNNS

Data generation. We consider the same data model as previous works (Bevilacqua et al., 2021; Miao et al., 2022; Chen et al., 2022a), where the underlying causal subgraph  $G_{c}$  and the spurious subgraph  $G_{s}$  will be assembled via some underlying assembling process. As we focus on the edge-centric view, our following discussion will focus on the graph structures  $A_{c}$  and  $A_{s}$  of the subgraphs. Full details of the structural causal model are deferred to Appendix C.1.

As shown in Fig. 2(a), there are latent causal and spurious variables  $C$  and  $S$  that have the invariant and spurious correlations with the label  $Y$  across different training and test distributions, respectively.  $C$  and  $S$  correspondingly control the generation of the graph structure of the causal subgraph  $G_{c}$ , and the spurious subgraph  $G_{s}$ . For example, when generating  $A_{c}$  and  $A_{s}$ ,  $C$  and  $S$  will specify the number of nodes in  $A_{c}$  and  $A_{s}$  and also the edge sampling probability for edges in  $A_{c}$  and  $A_{s}$ .

Interpretation. Correspondingly, XGNNs use a subgraph extractor to predict the causal and spurious subgraphs  $\widehat{G}_c$  and  $\widehat{G}_s$ , respectively. The extraction aims to reverse the generation and recover the structure of the underlying causal subgraph  $A_c$ . We denote the XGNN architecture and the hyperparameter settings as  $H$ .  $H$  takes  $A$  as inputs to learn the edge sampling probability via the attention mechanism and then obtain  $\widehat{A}_c$ . Once  $\widehat{A}_c$  is determined,  $\widehat{A}_s = 1 - \widehat{A}_c$  is also obtained by taking the complementary part. Then, the extracted causal and spurious subgraphs are obtained with  $\widehat{G}_c = (X, \widehat{A}_c)$  and  $\widehat{G}_s = (X, \widehat{A}_s)$ , respectively. The classifier then uses  $\widehat{G}_c$  to make the prediction  $\widehat{Y}$ .

# 4.2 CAUSAL FAITHFULNESS OF XGNNS

With the aforementioned causal model, we are able to specify the causal desiderata for faithful XGNNs. When a XGNN fails to accurately approximate SubMT, the estimated label conditional probability will have a huge gap from the ground truth. The failure will bias the optimization of the subgraph extractor  $g$  and lead to the degenerated interpretability of  $\widehat{A}$ . More concretely, the recovery of  $\widehat{A}$  to the underlying  $A$  will be worse, which further affects the extraction of  $G_{c}$  and brings both worse interpretation and (OOD) generalization performance. As a single measure such as the interpretation or generalization may not fully reflect the consequence or even exhibit conflicted information<sup>2</sup>, we consider a direct notion that jointly consider the interpretability and generalizability to measure the causal faithfulness of XGNNs, inspired by Jain & Wallace (2019).

For example, in the experiments of Miao et al. (2022), higher interpretation performance does not necessarily correlate with higher generalization performance.

Definition 4.1  $((\delta, \epsilon)$ -counterfactual fidelity). Given a meaningful minimal distance  $\delta > 0$ , let  $d(\cdot, \cdot)$  be a distribution distance metric, if a XGNN  $f = f_{c} \circ g$  commits to the  $\epsilon$ -counterfactual fidelity, then there exist  $\epsilon > 0$  such that,  $\forall G, \widetilde{G}$  that  $d(P(Y|G), P(Y|\widetilde{G})) \geq \delta$ , the following holds:

$$
d (P _ {f} (Y | \widetilde {G}), P _ {f} (Y | G)) \geq \epsilon \delta .
$$

Intuitively, if the extracted interpretable subgraph  $\widehat{G}_c$  is faithful to the target label, then the predictions made based on  $\widehat{G}_c$  are sensitive to any perturbations on  $\widehat{G}_c$ . Different from counterfactual interpretability (Prado-Romero et al., 2022; Guo et al., 2023) that seeks minimum modifications to change the predictions,  $(\delta, \epsilon)$ -counterfactual fidelity measures how sensitive are the predictions to the changes of the interpretable subgraphs. A higher fidelity implies better interpretability and is also a natural behavior of a XGNN that approximates SubMT well.

Proposition 4.2. If a XGNN  $f$ $\epsilon$ -approximates SubMT,  $f$  satisfies  $(\delta, 1 - \frac{2\epsilon}{\delta})$ -counterfactual fidelity.

The proof is given in Appendix D.3. Intuitively, Proposition 4.2 implies that the counterfactual fidelity is an effective measure for the approximation ability of SubMT in terms of Def. 3.2.

Practical estimation of counterfactual fidelity. Since it is hard to enumerate every possible  $\widetilde{G}$ , to verify Def. 4.1, we consider a random attention matrix  $\widetilde{A} \sim \sigma(\mathcal{N}(\mu_{\widehat{H}_A}, \sigma_{\widehat{H}_A}))$ , where  $\mu_{\widehat{H}_A}$  and  $\sigma_{\widehat{H}_A}$  are the mean and standard deviation of the pre-attention matrix  $\widehat{H}_A$  (The adjacency matrix with the unnormalized attention). Each non-symmetric entry in  $\widetilde{A}$  is sampled independently following the factorization of  $P(G)$ . We randomly sample  $\widetilde{A}$  by  $k$  times and calculate the following:

$$
c _ {\widehat {G} _ {c}} = \frac {1}{k} \sum_ {i = 1} ^ {k} d \left(f _ {c} \left(Y \mid \widetilde {G} _ {c} ^ {i}\right), f _ {c} \left(Y \mid \widehat {G} _ {c}\right)\right), \tag {10}
$$

where  $\widetilde{G}_c^i = (X, \widetilde{A}_c^i)$  and  $d$  is total variation distance. We compute  $c_{\widehat{G}_c}$  for the state-of-the-art XGNN GSAT (Miao et al., 2022). Shown as in Fig. 2(b), 2(c), we plot the counterfactual fidelity of GSAT on BA-2Motifs and Mutag datasets against is 2 to 3 times lower than the simulated SubMT with 10 and 100 sampling rounds. We provide a more detailed discussion in Appendix C.2 and Appendix F.5.

# 5 BUILDING RELIABLE INTERPRETABLE AND GENERALIZABLE GNNS

The aforementioned gap motivates us to propose a new XGNN architecture, called Graph Multilinear neT (GMT), to provide both faithful interpretability and reliable (OOD) generalizability. GMT have two variants, i.e., GMT-lin and GMT-sam, motivated by resolving the failures mentioned in Sec. 3.2.

# 5.1 LINEARIZED GRAPH MULTILINEAR NETWORK

Note that the main reason for the failure of Eq. 8 is because of the non-linearity of the expectation to the  $k$  weighted message passing with  $k > 1$ . If  $k$  can be reduced to 1, then the linearity can be preserved to ensure a better approximation of SubMT. More formally, for a XGNN  $f$  with linearized GNN as the classifier, if  $\exists T \in \mathbb{R}^{d \times d}$  such that  $T \cdot f_c(G_c) = P(Y|G_c)$  ( $f_c$  is linear), then, let

$$
\left(\operatorname {G M T} - \operatorname {l i n}\right) \quad f _ {c} ^ {l} \left(G _ {c}\right) = \rho \left(\widehat {A} \odot A ^ {k - 1} X W\right), \tag {11}
$$

we can incorporate GMT-lin into Eq. 8 and have the following holds:

$$
f _ {c} ^ {l} (\widehat {G} _ {c}) = \boldsymbol {T} \cdot f _ {c} (\widehat {G} _ {c}) = \mathbb {E} [ f _ {c} (G _ {c}) ],
$$

due to the linearity of  $f_c^l(G_c)$  with respect to  $G_c$  (i.e.,  $A$ ). During training,  $T$  can be further absorbed into  $W$ , which implies GMT-lin is able to fit to SubMT. Compared to the previous weighted message passing scheme with linearized GNN (Eq. 9), GMT-lin improves the linearity by reducing the number of weighted message passing rounds to 1. We show the simple strategy can already achieve better interpretability than the state-of-the-art methods even with non-linear GNNs in experiments.

# 5.2 GRAPH MULTILINEAR NETWORK WITH RANDOM SUBGRAPH SAMPLING

Although GMT-lin works for linearized GNNs, the non-linear GNNs are more widely used in practice (Xu et al., 2019), where GMT-lin may also suffer from the SubMT approximation failure.

To overcome the issue, inspired by the SubMT formulation, we propose a random subgraph sampling approach, that performs Markov Chain Monte Carlo (MCMC) sampling to approximate SubMT. More concretely, given the attention matrix  $\widehat{A}$ , we perform  $t$  rounds of random subgraph sampling from the subgraph distribution elicited by  $\widehat{A}$  (or equivalently  $\widehat{G}_c = (X,\widehat{A})$  as in SubMT (Def. 3.1), and obtain  $t$  i.i.d. random subgraph samples  $\{G_{c}^{i}\}_{i = 1}^{t}$  for estimating SubMT as the following

$$
\left(\mathrm {G M T} - \mathrm {s a m}\right) \quad f _ {c} ^ {s} \left(\widehat {G} _ {c}\right) = \frac {1}{t} \sum_ {i = 1} ^ {t} f _ {c} \left(Y \mid G _ {c} ^ {i}\right), \tag {12}
$$

where  $f_{c}$  is the classifier GNN that takes discrete subgraphs as inputs.

Theorem 5.1. Given the attention matrix  $\widehat{A}$ , and the distribution distance metric  $d$  as the total variation distance, let  $C = |\mathcal{V}|$ , for a GMT-sam with  $t$  i.i.d. samples of  $G_{c}^{i} \sim P(G_{c}|G)$ , then, there exists  $\epsilon \in \mathbb{R}_{+}$  such that, with a probability at least  $1 - e^{-t\epsilon^2 /4}$ , GMT-sam  $\frac{\epsilon C}{2}$ -approximates SubMT and satisfies  $(\delta, 1 - \frac{\epsilon C}{\delta})$  counterfactual fidelity.

The proof for Theorem 5.1 is given in Appendix D.4. Intuitively, with more random subgraph samples drawn from  $P(G_{c}|G)$ , GMT-sam obtains a more accurate estimation of SubMT. However, it will incur more practical challenges such as the a) gradient of discrete sampling and b) computational overhead. To overcome the challenges a) and b), we incorporate the following two techniques.

Backpropagation of discrete sampling. To enable gradient backpropagation with the sampled subgraphs, we also incorporate gradient estimation techniques such as Gumbel softmax and straight-through estimator (Jang et al., 2017; Maddison et al., 2017). Compared to the state-of-the-art XGNN GSAT (Miao et al., 2022), this scheme brings two additional benefits: (i) reduces the gradient biases in discrete sampling with Gumbel softmax; (ii) avoids weighted message passing and alleviates the input distribution gap to the graph encoder when shared in both  $f_{c}$  and  $g$  as in GSAT.

Learning neural subgraph multilinear extension. Although GMT trained with GMT-sam improve interpretability, GMT-sam still requires multiple random subgraph sampling to approximate SubMT and costs much additional overhead. To this end, we propose to learn a neural SubMT that only requires single sampling, based on the trained subgraph extractor  $g$  with GMT-sam.

Learning the neural SubMT is essentially to approximate the MCMC with a neural network, though it is inherently challenging to approximate MCMC (Johndrow et al., 2020; Papamarkou et al., 2022). Nevertheless, the feasibility of neural SubMT learning is backed by the inherent causal subgraph assumption of (Chen et al., 2022a), once the causal subgraph is correctly identified, simply learning the statistical correlation between the subgraph and the label is sufficient to recover the causal relation.

Therefore, we propose to simply re-train a new classifier GNN with the frozen subgraph extractor, to distill the knowledge contained in  $\widehat{G}_c$  about  $Y$ . This scheme also brings additional benefits over the originally trained classifier, which focuses on providing the gradient guidance for finding proper  $G_c$  instead of learning all the available statistical correlations between  $\bar{G}_c$  and  $Y$ . More details and discussions on the implementations can be found in Appendix E.

The number sampling rounds. Although the estimation of SubMT will be more accurate with the increased sampling rounds, it unnecessarily brings improvements. First, as shown in Fig. 3, the performance may be saturated with moderately sufficient samplings. Besides, the performance may degenerate as more sampling rounds can affect the optimization, as discussed in Appendix E.2

# 6 EXPERIMENTAL EVALUATIONS

We conduct extensive experiments to evaluate GMT with different backbones and on multiple graph classification benchmarks, and compare both the interpretability and (OOD) generalizability with the traditional post-hoc interpretation methods and the state-of-the-art XGNNs. We will briefly introduce the datasets, baselines, and experiment setups, and leave more details in Appendix F.

# 6.1 EXPERIMENTAL SETTINGS

Datasets. We consider both the regular and geometric graph classification benchmarks following the XGNN literature (Miao et al., 2022; 2023). For regular graphs, we include BA-2MOTIFS (Luo

Table 1: Interpretation Performance (AUC) on regular graph datasets. The shadowed entries are the results with the mean-1*std larger than the mean of the corresponding best baselines.  

<table><tr><td rowspan="2">GNN</td><td rowspan="2">METHOD</td><td rowspan="2">BA-2MOTIFS</td><td rowspan="2">MUTAG</td><td rowspan="2">MNIST-75SP</td><td colspan="3">SPURIOUS-MOTIF</td></tr><tr><td>b=0.5</td><td>b=0.7</td><td>b=0.9</td></tr><tr><td rowspan="5">GIN</td><td>GNNEXPLAINER</td><td>67.35±3.29</td><td>61.98±5.45</td><td>59.01±2.04</td><td>62.62±1.35</td><td>62.25±3.61</td><td>58.86±1.93</td></tr><tr><td>PGEXPLAINER</td><td>84.59±9.09</td><td>60.91±17.10</td><td>69.34±4.32</td><td>69.54±5.64</td><td>72.33±9.18</td><td>72.34±2.91</td></tr><tr><td>GRAPHMASK</td><td>92.54±8.07</td><td>62.23±9.01</td><td>73.10±6.41</td><td>72.06±5.58</td><td>73.06±4.91</td><td>66.68±6.96</td></tr><tr><td>IB-SUBGRAPH</td><td>86.06±28.37</td><td>91.04±6.59</td><td>51.20±5.12</td><td>57.29±14.35</td><td>62.89±15.59</td><td>47.29±13.39</td></tr><tr><td>DIR</td><td>82.78±10.97</td><td>64.44±28.81</td><td>32.35±9.39</td><td>78.15±1.32</td><td>77.68±1.22</td><td>49.08±3.66</td></tr><tr><td rowspan="3">GIN</td><td>GSAT</td><td>98.85±0.47</td><td>99.35±0.95</td><td>80.47±1.86</td><td>74.49±4.46</td><td>72.95±6.40</td><td>65.25±4.42</td></tr><tr><td>GMT-LIN</td><td>98.36±0.56</td><td>99.86±0.09</td><td>82.98±1.49</td><td>76.06±6.39</td><td>76.50±5.63</td><td>80.57±2.59</td></tr><tr><td>GMT-SAM</td><td>99.62±0.11</td><td>99.87±0.11</td><td>86.50±1.80</td><td>85.50±2.40</td><td>84.67±2.38</td><td>73.49±5.33</td></tr><tr><td rowspan="3">PNA</td><td>GSAT</td><td>89.35±5.41</td><td>99.00±0.37</td><td>85.72±1.10</td><td>79.84±3.21</td><td>79.76±3.66</td><td>80.70±5.45</td></tr><tr><td>GMT-LIN</td><td>95.79±7.30</td><td>99.58±0.17</td><td>85.02±1.03</td><td>80.19±2.22</td><td>84.74±1.82</td><td>85.08±3.85</td></tr><tr><td>GMT-SAM</td><td>99.60±0.48</td><td>99.89±0.05</td><td>87.34±1.79</td><td>88.27±1.71</td><td>86.58±1.89</td><td>85.26±1.92</td></tr></table>

Table 2: Prediction Performance (Acc.) on regular graph datasets. The shadowed entries are the results with the mean-1*std larger than the mean of the corresponding best baselines.  

<table><tr><td rowspan="2">GNN</td><td rowspan="2">METHOD</td><td rowspan="2">MOLHIV (AUC)</td><td rowspan="2">GRAPH-SST2</td><td rowspan="2">MNIST-75SP</td><td colspan="3">SPURIOUS-MOTIF</td></tr><tr><td>b=0.5</td><td>b=0.7</td><td>b=0.9</td></tr><tr><td rowspan="3">GIN</td><td>GIN</td><td>76.69±1.25</td><td>82.73±0.77</td><td>95.74±0.36</td><td>39.87±1.30</td><td>39.04±1.62</td><td>38.57±2.31</td></tr><tr><td>IB-SUBGRAPH</td><td>76.43±2.65</td><td>82.99±0.67</td><td>93.10±1.32</td><td>54.36±7.09</td><td>48.51±5.76</td><td>46.19±5.63</td></tr><tr><td>DIR</td><td>76.34±1.01</td><td>82.32±0.85</td><td>88.51±2.57</td><td>45.49±3.81</td><td>41.13±2.62</td><td>37.61±2.02</td></tr><tr><td rowspan="3">GIN</td><td>GSAT</td><td>76.12±0.91</td><td>83.14±0.96</td><td>96.20±1.48</td><td>47.45±5.87</td><td>43.57±2.43</td><td>45.39±5.02</td></tr><tr><td>GMT-LIN</td><td>76.87±1.12</td><td>83.19±1.28</td><td>96.01±0.25</td><td>47.69±4.93</td><td>53.11±4.12</td><td>46.22±4.18</td></tr><tr><td>GMT-SAM</td><td>77.22±0.93</td><td>83.62±0.50</td><td>96.50±0.19</td><td>60.09±2.40</td><td>54.34±4.04</td><td>55.83±5.68</td></tr><tr><td rowspan="4">PNA</td><td>PNA (NO SCALAR)</td><td>78.91±1.04</td><td>79.87±1.02</td><td>87.20±5.61</td><td>68.15±2.39</td><td>66.35±3.34</td><td>61.40±3.56</td></tr><tr><td>GSAT</td><td>79.82±0.67</td><td>80.90±0.37</td><td>93.69±0.73</td><td>68.41±1.76</td><td>67.78±3.22</td><td>51.51±2.98</td></tr><tr><td>GMT-LIN</td><td>80.05±0.71</td><td>81.18±0.47</td><td>94.44±0.49</td><td>69.33±1.42</td><td>64.49±3.51</td><td>58.30±6.61</td></tr><tr><td>GMT-SAM</td><td>80.58±0.83</td><td>82.36±0.96</td><td>95.75±0.42</td><td>71.98±3.44</td><td>69.68±3.99</td><td>67.90±3.60</td></tr></table>

et al., 2020), MUTAG (Debnath et al., 1991), MNIST-75SP (Knyazev et al., 2019), which are widely evaluated by post-hoc explanation approaches (Yuan et al., 2020b), as well as SPURIOUS-MOTIF (Wu et al., 2022b), GRAPH-SST2 (Socher et al., 2013; Yuan et al., 2020b) and OGBG-MOLHIV (Hu et al., 2020) where there exist various graph distribution shifts. For geometric graphs, we consider ACTsTACK, TAU3MU, SYNMOL and PLBIND curated by Miao et al. (2023).

Baselines. For post-hoc methods, we mainly adopt the results from the previous works (Miao et al., 2022; 2023), including GNNExplainer (Ying et al., 2019), PGExplainer (Luo et al., 2020), GraphMask (Schlichtkrull et al., 2021) for regular graph benchmarks, and BernMask, BernMask-P, that are modified from GNNExplainer and PGExplainer, GradGeo (Shrikumar et al., 2017), and Grad-Cam (Selvaraju et al., 2017) that are extended for geometric data, as well as PointMask (Taghanaki et al., 2020) developed specifically for geometric data. For XGNNs, since we focus on the interpretation performance, we mainly compared with XGNNs that have the state-of-the-art interpretation abilities, i.e., GSAT (Miao et al., 2022) and LRI (Miao et al., 2023), which also have excellent OOD generalization performance than other XGNNs (Gui et al., 2022). We also include two representative XGNNs baselines, DIR (Wu et al., 2022b) and IB-subgraph (Yu et al., 2021) for regular graph data.

Training and evaluation. We consider three backbones GIN (Xu et al., 2019) and PNA (Corso et al., 2020) for regular graph data, EGNN (Satorras et al., 2021) for geometric data. All methods adopted the identical graph encoder, and optimization protocol for fair comparisons. We tune the hyperparameters as recommended by previous works. More details are given in Appendix F.2.

# 6.2 EXPERIMENTAL RESULTS AND ANALYSIS

Interpretation performance. As shown in Table. 1, compared to post-hoc based methods (in the first row), and GSAT, both GMT-1in and GMT-sam lead to non-trivial improvements for interpretation performance. Especially, in challenging Spurious-Motif datasets where there contain distribution shifts, GMT-sam brings improvements than GSAT up to  $15\%$  with GIN, and up to  $8\%$  with PNA. In challenging realistic dataset MNIST-75sp, GMT-sam also improves GSAT up to  $6\%$ .

Generalization performance. Table 2 illustrates the prediction accuracy on regular graph datasets. We again observe consistent improvements for diverse datasets spanning from molecule graphs to

Table 3: Interpretation performance on the geometric learning datasets. The shadowed entries are the results with the mean-1*std larger than the mean of the corresponding best baselines.  

<table><tr><td rowspan="2"></td><td colspan="2">ACTSTRACK</td><td colspan="2">TAU3MU</td><td colspan="2">SYNMOL</td><td colspan="2">PLBIND</td></tr><tr><td>ROC AUC</td><td>PREC@12</td><td>ROC AUC</td><td>PREC@12</td><td>ROC AUC</td><td>PREC@12</td><td>ROC AUC</td><td>PREC@12</td></tr><tr><td>RANDOM</td><td>50</td><td>21</td><td>50</td><td>35</td><td>50</td><td>31</td><td>50</td><td>45</td></tr><tr><td>GRADGEO</td><td>69.31±0.89</td><td>33.54±1.23</td><td>78.04±0.57</td><td>64.18±1.25</td><td>76.38±4.96</td><td>64.72±3.75</td><td>58.11±2.91</td><td>64.78±4.73</td></tr><tr><td>BERNMASK</td><td>54.23±4.31</td><td>20.46±5.46</td><td>71.58±0.69</td><td>60.51±0.76</td><td>76.38±4.96</td><td>64.72±3.75</td><td>52.23±4.45</td><td>41.50±9.77</td></tr><tr><td>BERNMASK-P</td><td>22.87±3.33</td><td>11.29±5.46</td><td>70.72±5.10</td><td>55.50±6.26</td><td>87.06±7.12</td><td>77.11±7.58</td><td>51.98±4.66</td><td>59.20±5.48</td></tr><tr><td>POINTMASK</td><td>49.20±1.51</td><td>20.54±1.71</td><td>55.93±4.85</td><td>39.65±7.14</td><td>66.46±6.86</td><td>53.93±1.94</td><td>50.00±0.00</td><td>45.10±0.00</td></tr><tr><td>GRADGAM</td><td>75.19±1.91</td><td>75.94±2.16</td><td>76.18±2.62</td><td>62.05±2.16</td><td>60.31±4.95</td><td>52.35±11.02</td><td>48.61±2.34</td><td>55.10±10.57</td></tr><tr><td>LRI-BERNOULLI</td><td>74.38±4.33</td><td>81.42±1.52</td><td>78.23±1.11</td><td>65.64±2.44</td><td>89.22±3.58</td><td>68.76±7.35</td><td>54.87±1.89</td><td>72.12±2.60</td></tr><tr><td>GMT-LIN</td><td>77.45±1.69</td><td>81.81±1.57</td><td>79.17±0.82</td><td>68.94±1.08</td><td>96.17±1.44</td><td>86.33±6.16</td><td>59.70±1.10</td><td>70.62±3.59</td></tr><tr><td>GMT-SAM</td><td>75.61±1.86</td><td>81.96±1.35</td><td>78.28±1.34</td><td>65.69±2.61</td><td>93.93±3.59</td><td>83.20±4.74</td><td>60.03±1.02</td><td>72.56±2.27</td></tr></table>

image-converted datasets. Despite distribution shifts, GMT-sam still brings improvements up to  $13\%$  with GIN, and up to  $16\%$  against GSAT in Spurious-Motif.

Results on geometric benchmarks. Tables 3 and 4 show the interpretation and generalization performance of various methods. Again, we also observe consistent non-trivial improvements of GMT-lin and GMT-sam in most cases than GSAT and post-hoc methods. Interestingly, GMT-lin leads to more improvements than GMT-sam in terms of interpretation performance despite of its simple modifications, while having a competitive generalization performance as LRI. In terms of generalization performance, GMT-sam remain the best method. The results on geometric datasets further demonstrate the strong generality of GMT across different tasks and backbones.

Table 4: Prediction performance (AUC) on the geometric learning datasets. The shadowed entries are the results with the mean-1*std larger than the mean of the best baselines.  

<table><tr><td></td><td>ACTSTACK</td><td>TAU3MU</td><td>SYNMOL</td><td>PLBIND</td><td></td><td>ACTSTACK</td><td>TAU3MU</td><td>SYNMOL</td><td>PLBIND</td></tr><tr><td>ERM</td><td>97.40±0.32</td><td>82.75±0.16</td><td>99.30±0.20</td><td>85.31±2.21</td><td>GMT-LIN</td><td>93.92±0.98</td><td>82.60±0.17</td><td>99.26±0.27</td><td>86.29±0.80</td></tr><tr><td>LRI-BERNOULLI</td><td>94.00±0.78</td><td>86.36±0.06</td><td>99.30±0.15</td><td>85.80±0.70</td><td>GMT-SAM</td><td>98.55±0.11</td><td>86.42±0.08</td><td>99.89±0.03</td><td>87.19±1.86</td></tr></table>

![](images/fde66c61e456d971a74510e016aba9967265007c97e31af8a14b8f807d21b8f0.jpg)  
(a) Counterfactual fidelity.

![](images/b5162d218f8d5191881de658b9ffaeee2291f48d70c632dc724cf30ed40aefc4.jpg)  
(b) Interpretation sensitivity.

![](images/3c9394c378f829d973c9a3006552e80f0fec874fabeead055a117245bff8f6d0.jpg)  
Figure 3: Ablation studies.  
(c) Generalization sensitivity.

Ablation studies. In complementary to the interpretability and generalizability study, we conduct further ablation studies to better understand the results. Fig. 3(a) shows the counterfactual fidelity of GSAT, GMT-1in and GMT-sam in Spurious-Motif (SPmotif) test sets. As shown in Fig. 3(a) that GSAT achieves a lower counterfactual fidelity. In contrast, GMT-1in and GMT-sam improve a higher counterfactual fidelity, which explains the reason for the improved interpretability of GMT. We also examine the hyperparameter sensitivity of GMT-sam in SPMotif-0.5 dataset. As shown in Fig. 3(b), 3(c), GMT-sam maintains strong robustness against the hyperparameter choices. The interpretation performance gets improved along with the sampling rounds, while a too larger GIB information regularizer weights will affect the optimization of GMT as well as the generalizability.

# 7 CONCLUSIONS

We developed a theoretical framework to analyze the expressive power of XGNNs by formulating the subgraph learning with multilinear extension (SubMT). We find that existing attention-based XGNNs will fail to approximate SubMT, which will lead to unfaithful interpretation as well as poor (OOD) generalization. To mitigate the issue, we propose a simple yet novel architecture called GMT which is provably more powerful in approximating SubMT. Extensive experiments on both graph classification and geometric learning benchmarks verify the superior interpretability and generalizability of GMT.

# REFERENCES

David Alvarez-Melis and Tommi S. Jaakkola. Towards robust interpretability with self-explaining neural networks. In Advances in Neural Information Processing Systems, pp. 7786-7795, 2018. (Cited on pages 3 and 21)  
Kenza Amara, Rex Ying, Zitao Zhang, Zhihao Han, Yinan Shan, Ulrik Brandes, Sebastian Schemm, and Ce Zhang. Graphframex: Towards systematic evaluation of explainability methods for graph neural networks. In Learning on Graphs Conference, 2022. (Cited on pages 3 and 21)  
Mohit Bajaj, Lingyang Chu, Zi Yu Xue, Jian Pei, Lanjun Wang, Peter Cho-Ho Lam, and Yong Zhang. Robust counterfactual explanations on graph neural networks. In Advances in Neural Information Processing Systems, 2021. (Cited on pages 3 and 21)  
Victor Bapst, Thomas Keck, Agnieszka Grabska-Barwinska, Craig Donner, Ekin Dogus Cubuk, Samuel S. Schoenholz, Annette Obika, Alexander W. R. Nelson, Trevor Back, Demis Hassabis, and Pushmeet Kohli. Unveiling the predictive power of static structure in glassy systems. Nature Physics, 16:448-454, 2020. (Cited on page 1)  
Helen M. Berman, John D. Westbrook, Zukang Feng, Gary L Gilliland, Talapady N. Bhat, Helge Weissig, Ilya N. Shindyalov, and Philip E. Bourne. The protein data bank. *Nucleic acids research*, 28 1:235–42, 2000. (Cited on page 35)  
Beatrice Bevilacqua, Yangze Zhou, and Bruno Ribeiro. Size-invariant graph representations for graph classification extrapolations. In International Conference on Machine Learning, pp. 837-851, 2021. (Cited on pages 5, 22 and 24)  
Yatao Bian, Joachim Buhmann, and Andreas Krause. Optimal continuous DR-submodular maximization and applications to provable mean field inference. In International Conference on Machine Learning, pp. 644-653, 2019. (Cited on pages 3 and 22)  
Yatao Bian, Yu Rong, Tingyang Xu, Jiaxiang Wu, Andreas Krause, and Junzhou Huang. Energy-based learning for cooperative games, with applications to valuation problems in machine learning. In International Conference on Learning Representations, 2022. (Cited on page 22)  
Christian Bierlich, Smita Chakraborty, Nishita Desai, Leif Gellersen, Ilkka J. Helenius, Philip Ilten, Leif Lonnblad, Stephen Mrenna, Stefan Prestel, Christian T. Preuss, Torbjorn Sjostrand, Peter Skands, Marius Utheim, and Rob Verheyen. A comprehensive guide to the physics and usage of pythia 8.3. SciPost Physics Codebases, 2022. (Cited on page 34)  
Gruia Calinescu, Chandra Chekuri, Martin Pál, and Jan Vondrák. Maximizing a monotone submodular function subject to a matroid constraint. SIAM Journal on Computing, 40(6):1740-1766, 2011. (Cited on page 22)  
Shiyu Chang, Yang Zhang, Mo Yu, and Tommi S. Jaakkola. Invariant rationalization. In International Conference on Machine Learning, pp. 1448-1458, 2020. (Cited on pages 21 and 22)  
Chandra Chekuri, Jan Vondrak, and Rico Zenklusen. Submodular function maximization via the multilinear relaxation and contention resolution schemes. SIAM Journal on Computing, 43(6): 1831-1879, 2014. (Cited on page 22)  
Chandra Chekuri, T.S. Jayram, and Jan Vondrak. On multiplicative weight updates for concave and submodular function maximization. In Conference on Innovations in Theoretical Computer Science, pp. 201-210, 2015. (Cited on page 22)  
Yongqiang Chen, Yonggang Zhang, Yatao Bian, Han Yang, Kaili Ma, Binghui Xie, Tongliang Liu, Bo Han, and James Cheng. Learning causally invariant representations for out-of-distribution generalization on graphs. In Advances in Neural Information Processing Systems, 2022a. (Cited on pages 1, 2, 3, 4, 5, 7, 20, 21, 22, 24, 30 and 33)  
Yongqiang Chen, Kaiwen Zhou, Yatao Bian, Binghui Xie, Kaili Ma, Yonggang Zhang, Han Yang, Bo Han, and James Cheng. Pareto invariant risk minimization. arXiv preprint, arXiv:2206.07766, 2022b. (Cited on page 31)

ATLAS Collaboration. Search for charged-lepton-flavour violation in z-boson decays with the atlas detector. Nature Physics, 17:819 - 825, 2020. (Cited on page 34)  
Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, and Petar Velickovic. Principal neighbourhood aggregation for graph nets. In Advances in Neural Information Processing Systems, 2020. (Cited on pages 8 and 35)  
Jonathan Crabbé and Mihaela van der Schaar. Evaluating the robustness of interpretability methods through explanation invariance and equivariance. arXiv preprint, arXiv:2304.06715, 2023. (Cited on pages 3 and 21)  
M. Cranmer, Alvaro Sanchez-Gonzalez, Peter W. Battaglia, Rui Xu, Kyle Cranmer, David N. Spergel, and Shirley Ho. Discovering symbolic models from deep learning with inductive biases. arXiv preprint, arXiv:2006.11287, 2020. (Cited on page 1)  
Gruia Calinescu, Chandra Chekuri, Martin Pál, and Jan Vondrák. Maximizing a submodular set function subject to a matroid constraint (extended abstract). In Conference on Integer Programming and Combinatorial Optimization, 2007. (Cited on pages 2, 3, 4 and 22)  
Minyi Dai, Mehmet F Demirel, Yingyu Liang, and Jia-Mian Hu. Graph neural networks for an accurate and interpretable prediction of the properties of polycrystalline materials. npj Computational Materials, 7(1):103, 2021. (Cited on page 1)  
Asim Kumar Debnath, R L Compadre, Gargi Debnath, Alan J. Shusterman, and Corwin Hansch. Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. correlation with molecular orbital energies and hydrophobicity. Journal of medicinal chemistry, 34 2: 786–97, 1991. (Cited on pages 8 and 33)  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019. (Cited on page 33)  
Xing Du, Yi Li, Yuan-Ling Xia, Shi-Meng Ai, Jing Liang, Peng Sang, Xing lai Ji, and Shu-Qun Liu. Insights into protein-ligand interactions: Mechanisms, models, and methods. International Journal of Molecular Sciences, 17, 2016. URL https://api_semanticscholar.org/ CorpusID:18139389. (Cited on page 35)  
Paul L. Erdos and Alfréd Rényi. On the evolution of random graphs. Transactions of the American Mathematical Society, 286:257-257, 1984. (Cited on page 4)  
Shaohua Fan, Xiao Wang, Yanhu Mo, Chuan Shi, and Jian Tang. Debiasing graph neural networks via learning disentangled causal substructure. In Advances in Neural Information Processing Systems, 2022. (Cited on pages 3 and 20)  
Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019. (Cited on page 41)  
Kimon Fountoulakis, Amit Levi, Shenghao Yang, Aseem Baranwal, and Aukosh Jagannath. Graph attention retrospective. Journal of Machine Learning Research, 24(246):1-52, 2023. (Cited on pages 21 and 32)  
Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew E. Peters, Michael Schmitz, and Luke Zettlemoyer. Allennlp: A deep semantic natural language processing platform. arXiv preprint, arXiv:1803.07640, 2018. (Cited on page 33)  
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In International Conference on Machine Learning, pp. 1263-1272, 2017. (Cited on page 1)  
Shurui Gui, Xiner Li, Limei Wang, and Shuiwang Ji. GOOD: A graph out-of-distribution benchmark. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. (Cited on pages 1 and 8)

Shurui Gui, Meng Liu, Xiner Li, Youzhi Luo, and Shuiwang Ji. Joint learning of label and environment causal independence for graph out-of-distribution generalization. arXiv preprint, arXiv:2306.01103, 2023. (Cited on page 22)  
Zhimeng Guo, Teng Xiao, Charu Aggarwal, Hui Liu, and Suhang Wang. Counterfactual learning on graphs: A survey. arXiv preprint, arXiv:2304.01391, 2023. (Cited on pages 6 and 21)  
Berry Holstein. The Theory of Almost Everything: The Standard Model, the Unsung Triumph of Modern Physics. Physics Today, 59(7):49-50, 07 2006. (Cited on page 34)  
Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, 2020. (Cited on pages 8, 33, 34 and 36)  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015. (Cited on pages 30, 35 and 36)  
Sarthak Jain and Byron C. Wallace. Attention is not explanation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 3543-3556, 2019. (Cited on pages 3, 5 and 21)  
Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017. (Cited on pages 7 and 29)  
Yuanfeng Ji, Lu Zhang, Jiaxiang Wu, Bingzhe Wu, Long-Kai Huang, Tingyang Xu, Yu Rong, Lanqing Li, Jie Ren, Ding Xue, Houtim Lai, Shaoyong Xu, Jing Feng, Wei Liu, Ping Luo, Shuigeng Zhou, Junzhou Huang, Peilin Zhao, and Yatao Bian. DrugOOD: Out-of-Distribution (OOD) Dataset Curator and Benchmark for AI-aided Drug Discovery – A Focus on Affinity Prediction Problems with Noise Annotations. arXiv preprint, arXiv:2201.09637, 2022. (Cited on pages 1 and 21)  
Wei Jin, Tong Zhao, Jiayu Ding, Yozen Liu, Jiliang Tang, and Neil Shah. Empowering graph representation learning with test-time graph transformation. arXiv preprint, arXiv:2210.03561, 2022. (Cited on page 22)  
James E. Johndrow, Natesh S. Pillai, and Aaron Smith. No free lunch for approximate mcmc. arXiv preprint, arXiv:2010.12514, 2020. (Cited on pages 7, 30 and 32)  
John M. Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zidek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andy Ballard, Andrew Cowie, Bernardino Romero-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David A. Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with alphafold. Nature, 596:583 - 589, 2021. (Cited on page 1)  
Barakeel Fanseu Kamhoua, Lin Zhang, Yongqiang Chen, Han Yang, MA KAILI, Bo Han, Bo Li, and James Cheng. Exact shape correspondence via 2d graph convolution. In Advances in Neural Information Processing Systems, 2022. (Cited on page 22)  
Nikolaos Karalias, Joshua Robinson, Andreas Loukas, and Stefanie Jegelka. Neural set function extensions: Learning with discrete functions in high dimensions. In Advances in Neural Information Processing Systems, 2022. (Cited on pages 3, 4 and 22)  
Amir-Hossein Karimi, Krikamol Muandet, Simon Kornblith, Bernhard Scholkopf, and Been Kim. On the relationship between explanation and prediction: A causal view. In International Conference on Machine Learning, pp. 15861-15883, 2023. (Cited on page 21)  
Mostafa Karimi, Di Wu, Zhangyang Wang, and Yang Shen. Deepaffinity: Interpretable deep learning of compound-protein affinity through unified recurrent and convolutional neural networks. Bioinformatics, 2019. (Cited on page 35)

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. (Cited on page 36)  
Boris Knyazev, Graham W. Taylor, and Mohamed R. Amer. Understanding attention and generalization in graph neural networks. In Advances in Neural Information Processing Systems, pp. 4204-4214, 2019. (Cited on pages 8, 33 and 36)  
Dmitrii Kochkov, Tobias Pfaff, Alvaro Sanchez-Gonzalez, Peter W. Battaglia, and Bryan K. Clark. Learning ground states of quantum hamiltonians with graph networks. arXiv preprint, arXiv:2110.06390, 2021. (Cited on page 1)  
Roman A. Laskowski. PDBsum: summaries and analyses of PDB structures. *Nucleic Acids Research*, 29(1):221-222, 01 2001. (Cited on page 35)  
Soo Yong Lee, Fanchen Bu, Jaemin Yoo, and Kijung Shin. Towards deep attention in graph neural networks: Problems and remedies. In International Conference on Machine Learning, volume 202, pp. 18774-18795, 2023. (Cited on pages 21 and 32)  
Haoyang Li, Ziwei Zhang, Xin Wang, and Wenwu Zhu. Learning invariant graph representations for out-of-distribution generalization. In Advances in Neural Information Processing Systems, 2022. (Cited on pages 3, 20 and 22)  
Xiner Li, Shurui Gui, Youzhi Luo, and Shuiwang Ji. Graph structure and feature extrapolation for out-of-distribution generalization. arXiv preprint, arXiv:2306.08076, 2023. (Cited on page 22)  
Wanyu Lin, Hao Lan, and Baochun Li. Generative causal explanations for graph neural networks. In International Conference on Machine Learning, volume 139, pp. 6666-6679, 2021. (Cited on pages 1, 3 and 20)  
Wanyu Lin, Hao Lan, Hao Wang, and Baochun Li. Orphicx: A causality-inspired latent variable model for interpreting graph neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13719-13728, 2022a. (Cited on pages 1, 3 and 20)  
Yong Lin, Shengyu Zhu, Lu Tan, and Peng Cui. ZIN: When and how to learn invariance without environment partition? In Advances in Neural Information Processing Systems, 2022b. (Cited on page 35)  
Zachary C. Lipton. The mythos of model interpretability. Commun. ACM, 61(10):36-43, 2018. (Cited on pages 3 and 21)  
Gang Liu, Tong Zhao, Jiaxin Xu, Tengfei Luo, and Meng Jiang. Graph rationalization with environment-based augmentations. arXiv preprint arXiv:2206.02886, 2022a. (Cited on pages 3, 20 and 22)  
Meng Liu, Youzhi Luo, Kanji Uchino, Koji Maruhashi, and Shuiwang Ji. Generating 3d molecules for target protein binding. arXiv preprint, arXiv:2204.09410, 2022b. (Cited on page 35)  
Zhihai Liu, Minyi Su, Li Han, Jie Liu, Qifan Yang, Yan Li, and Renxiao Wang. Forging the basis for developing protein-ligand interaction scoring functions. Accounts of chemical research, 50 2: 302-309, 2017. (Cited on page 35)  
László Lovász and Balázs Szegedy. Limits of dense graph sequences. Journal of Combinatorial Theory, Series B, 96(6):933-957, 2006. (Cited on page 4)  
Dongsheng Luo, Wei Cheng, Dongkuan Xu, Wenchao Yu, Bo Zong, Haifeng Chen, and Xiang Zhang. Parameterized explainer for graph neural network. In Advances in Neural Information Processing Systems, pp. 19620-19631, 2020. (Cited on pages 1, 3, 7, 8, 20, 33 and 36)  
Jing Ma, Ruocheng Guo, Saumitra Mishra, Aidong Zhang, and Jundong Li. Clear: Generative counterfactual explanations on graphs. In Advances in Neural Information Processing Systems, pp. 25895-25907, 2022. (Cited on pages 3 and 21)  
Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017. (Cited on pages 7 and 29)

Sadegh Mahdavi, Kevin Swersky, Thomas Kipf, Milad Hashemi, Christos Thrampoulidis, and Renjie Liao. Towards better out-of-distribution generalization of neural algorithmic reasoning tasks. arXiv preprint arXiv:2211.00692, 2022. (Cited on page 22)  
Kevin McCloskey, Ankur Taly, Federico Monti, Michael P. Brenner, and Lucy J. Colwell. Using attribution to decode binding mechanism in neural network models for chemistry. Proceedings of the National Academy of Sciences, 116:11624 - 11629, 2018. (Cited on pages 34 and 35)  
Siqi Miao, Miaoyuan Liu, and Pan Li. Interpretable and generalizable graph learning via stochastic attention mechanism. International Conference on Machine Learning, 2022. (Cited on pages 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 33, 35 and 36)  
Siqi Miao, Yunan Luo, Mia Liu, and Pan Li. Interpretable geometric deep learning via learnable randomness injection. In International Conference on Learning Representations, 2023. (Cited on pages 1, 2, 3, 7, 8, 20, 22, 23, 33, 34, 35 and 36)  
Christopher W. Murray and David C Rees. The rise of fragment-based drug discovery. Nature chemistry, 1 3:187-92, 2009. (Cited on page 1)  
Guillermo Owen. Multilinear extensions of games. Management Science, 18:64-79, 1972. (Cited on page 22)  
T. Papamarkou, J. Hinkle, M. T. Young, and D. Womble. Challenges in Markov chain Monte Carlo for Bayesian neural networks. Statistical Science, 37(3):425-442, 2022. (Cited on pages 7 and 30)  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, pp. 8024-8035, 2019. (Cited on page 41)  
Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo, and Fosca Giannotti. A survey on graph counterfactual explanations: Definitions, methods, evaluation, and research challenges. ACM Computing Surveys, 2022. (Cited on pages 3, 6 and 21)  
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135-1144, 2016. (Cited on pages 3 and 21)  
Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1:206-215, 2018. (Cited on page 21)  
Aytunc Sahin, Yatao Bian, Joachim M. Buhmann, and Andreas Krause. From sets to multisets: Provable variational inference for probabilistic integer submodular models. In International Conference on Machine Learning, volume 119, pp. 8388-8397, 2020. (Cited on pages 3 and 22)  
Victor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) equivariant graph neural networks. In International Conference on Machine Learning, pp. 9323-9332, 2021. (Cited on pages 8 and 35)  
Michael Sejr Schlichtkrull, Nicola De Cao, and Ivan Titov. Interpreting graph neural networks for NLP with differentiable edge masking. In International Conference on Learning Representations, 2021. (Cited on page 8)  
R. Schulte, V. Bashkirov, Tianfang Li, Zhengrong Liang, K. Mueller, J. Heimann, L.R. Johnson, B. Keeney, H.F.-W. Sadrozinski, A. Seiden, D.C. Williams, Lan Zhang, Zhang Li, S. Peggs, T. Satogata, and C. Woody. Conceptual design of a proton computed tomography system for applications in proton radiation therapy. IEEE Transactions on Nuclear Science, 51(3):866-872, 2004. (Cited on page 34)  
Kristof T. Schütt, Huziel E. Sauceda, P J Kindermans, Alexandre Tkatchenko, and Klaus-Robert Müller. Schnet - a deep learning architecture for molecules and materials. The Journal of chemical physics, 148 24:241722, 2017. (Cited on page 1)

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In IEEE International Conference on Computer Vision, pp. 618-626. IEEE Computer Society, 2017. (Cited on page 8)  
Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In International Conference on Machine Learning, pp. 3145-3153, 2017. (Cited on page 8)  
Tom A.B. Snijders and Krzysztof Nowicki. Estimation and prediction for stochastic blockmodels for graphs with latent block structure. In Journal of Classification, volume 14, pp. 75-100, 1997. (Cited on page 4)  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Conference on Empirical Methods in Natural Language Processing, pp. 1631-1642, 2013. (Cited on pages 8 and 33)  
Hannes Stárk, Octavian-Eugen Ganea, Lagnajit Pattanaik, Regina Barzilay, and T. Jaakkola. Equibind: Geometric deep learning for drug binding structure prediction. In International Conference on Machine Learning, 2022. (Cited on page 35)  
Yongduo Sui, Xiang Wang, Jiancan Wu, Min Lin, Xiangnan He, and Tat-Seng Chua. Causal attention for interpretable and generalizable graph classification. In The 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1696-1705, 2022. (Cited on pages 3 and 20)  
Saeid Asgari Taghanaki, Kaveh Hassani, Pradeep Kumar Jayaraman, Amir Hosein Khas Ahmadi, and Tonya Custis. Pointmask: Towards interpretable and bias-resilient point cloud processing. arXiv preprint, arXiv:2007.04525, 2020. (Cited on page 8)  
Naftali Tishby, Fernando C. Pereira, and William Bialek. The information bottleneck method. In Annual Allerton Conference on Communication, Control and Computing, pp. 368-377, 1999. (Cited on pages 3 and 20)  
Pablo Villanueva-Domingo, Francisco Villaescusa-Navarro, Daniel Angl'es-Alc'azar, Shy Genel, Federico Marinacci, David N. Spergel, Lars E. Hernquist, Mark Vogelsberger, Romeel Dave, and Desika Narayanan. Inferring halo masses with graph neural networks. The Astrophysical Journal, 935, 2021. (Cited on page 1)  
Jan Vondrak. Optimal approximation for the submodular welfare problem in the value oracle model. In Annual ACM Symposium on Theory of Computing, pp. 67-74, 2008. (Cited on pages 3 and 22)  
Minh N. Vu and My T. Thai. Pgm-explainer: Probabilistic graphical model explanations for graph neural networks. In Advances in Neural Information Processing Systems, 2020. (Cited on pages 1, 3 and 20)  
Cheng Wang and Yingkai Zhang. Improving scoring-docking-screening powers of protein-ligand scoring functions using random forest. Journal of Computational Chemistry, 38:169 - 177, 2017. (Cited on page 35)  
Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, Anima Anandkumar, Karianne J. Bergen, Carla P. Gomes, Shirley Ho, Pushmeet Kohli, Joan Lasenby, Jure Leskovec, Tie-Yan Liu, Arjun K. Manrai, Debora Marks, Bharath Ramsundar, Le Song, Jimeng Sun, Jian Tang, Petar Velickovic, Max Welling, Linfeng Zhang, Connor W. Coley, Yoshua Bengio, and Marinka Zitnik. Scientific discovery in the age of artificial intelligence. Nature, 620:47 - 60, 2023. (Cited on page 1)  
Joanna Wencel-Delord and Frank Glorius. C-h bond activation enables the rapid construction and late-stage diversification of functional molecules. Nature chemistry, 5 5:369-75, 2013. (Cited on page 1)  
Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. Simplifying graph convolutional networks. In Proceedings of the 36th International Conference on Machine Learning, pp. 6861-6871, 2019. (Cited on pages 4, 26 and 27)

Qitian Wu, Hengrui Zhang, Junchi Yan, and David Wipf. Handling distribution shifts on graphs: An invariance perspective. In International Conference on Learning Representations, 2022a. (Cited on pages 1 and 22)  
Tailin Wu, Hongyu Ren, Pan Li, and Jure Leskovec. Graph information bottleneck. In Advances in Neural Information Processing Systems, pp. 20437-20448, 2020. (Cited on pages 3 and 20)  
Yingxin Wu, Xiang Wang, An Zhang, Xiangnan He, and Tat-Seng Chua. Discovering invariant rationales for graph neural networks. In International Conference on Learning Representations, 2022b. (Cited on pages 1, 3, 8, 20, 22, 33, 36 and 37)  
Tian Xie and Jeffrey C. Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical review letters, 120 14:145301, 2017. (Cited on page 1)  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? In International Conference on Learning Representations, 2019. (Cited on pages 6, 8, 20, 32 and 35)  
Keyulu Xu, Mozhi Zhang, Jingling Li, Simon Shaolei Du, Ken-ichi Kawarabayashi, and Stefanie Jegelka. How neural networks extrapolate: From feedforward to graph neural networks. In International Conference on Learning Representations, 2021. (Cited on page 22)  
Ling Yang, Jiayi Zheng, Heyuan Wang, Zhongyi Liu, Zhilin Huang, Shenda Hong, Wentao Zhang, and Bin Cui. Individual and structural graph information bottlenecks for out-of-distribution generalization. arXiv preprint, arXiv:2306.15902, 2023. (Cited on pages 3, 20 and 22)  
Nianzu Yang, Kaipeng Zeng, Qitian Wu, Xiaosong Jia, and Junchi Yan. Learning substructure invariance for out-of-distribution molecular representations. In Advances in Neural Information Processing Systems, 2022. (Cited on page 22)  
Gilad Yehudai, Ethan Fetaya, Eli Meirom, Gal Chechik, and Haggai Maron. From local structures to size generalization in graph neural networks. In International Conference on Machine Learning, pp. 11975-11986, 2021. (Cited on page 22)  
Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. Gnnexplainer: Generating explanations for graph neural networks. In Advances in Neural Information Processing Systems, pp. 9240-9251, 2019. (Cited on pages 1, 3, 8, 20 and 33)  
Junchi Yu, Tingyang Xu, Yu Rong, Yatao Bian, Junzhou Huang, and Ran He. Graph information bottleneck for subgraph recognition. In International Conference on Learning Representations, 2021. (Cited on pages 1, 3, 8, 20, 22 and 24)  
Junchi Yu, Jie Cao, and Ran He. Improving subgraph recognition with variational graph information bottleneck. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19374-19383. IEEE, 2022. (Cited on pages 3, 20 and 22)  
Junchi Yu, Jian Liang, and Ran He. Mind the label shift of augmentation-based graph OOD generalization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11620-11630. IEEE, 2023. (Cited on page 22)  
Hao Yuan, Jiliang Tang, Xia Hu, and Shuiwang Ji. XGNN: towards model-level explanations of graph neural networks. In The 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 430-438, 2020a. (Cited on pages 1, 3 and 20)  
Hao Yuan, Haiyang Yu, Shurui Gui, and Shuiwang Ji. Explainability in graph neural networks: A taxonomic survey. arXiv preprint, arXiv:2012.15445, 2020b. (Cited on pages 3, 8, 21, 33 and 36)  
Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, and Shuiwang Ji. On explainability of graph neural networks via subgraph explorations. In International Conference on Machine Learning, volume 139, pp. 12241-12252, 2021. (Cited on pages 1, 3 and 20)

Xuan Zhang, Limei Wang, Jacob Helwig, Youzhi Luo, Cong Fu, Yaochen Xie, Meng Liu, Yuchao Lin, Zhao Xu, Keqiang Yan, Keir Adams, Maurice Weiler, Xiner Li, Tianfan Fu, Yucheng Wang, Haiyang Yu, Yuqing Xie, Xiang Fu, Alex Strasser, Shenglong Xu, Yi Liu, Yuanqi Du, Alexandra Saxton, Hongyi Ling, Hannah Lawrence, Hannes Stärk, Shurui Gui, Carl Edwards, Nicholas Gao, Adriana Ladera, Tailin Wu, Elyssa F. Hofgard, Aria Mansouri Tehrani, Rui Wang, Ameya Daigavane, Montgomery Bohde, Jerry Kurtin, Qian Huang, Tuong Phung, Minkai Xu, Chaitanya K. Joshi, Simon V. Mathis, Kamyar Azizzadenesheli, Ada Fang, Alán Aspuru-Guzik, Erik Bekkers, Michael M. Bronstein, Marinka Zitnik, Anima Anandkumar, Stefano Ermon, Pietro Lio, Rose Yu, Stephan Gunnemann, Jure Leskovec, Heng Ji, Jimeng Sun, Regina Barzilay, Tommi S. Jaakkola, Connor W. Coley, Xiaoning Qian, Xiaofeng Qian, Tess E. Smidt, and Shuiwang Ji. Artificial intelligence for science in quantum, atomistic, and continuum systems. arXiv preprint, arXiv:2307.08423, 2023. (Cited on pages 1 and 21)  
Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Cheekong Lee. Protgnn: Towards self-explaining graph neural networks. In Thirty-Sixth AAAI Conference on Artificial Intelligence, pp. 9127-9135, 2022. (Cited on pages 3 and 20)  
Yangze Zhou, Gitta Kutyniok, and Bruno Ribeiro. OOD link prediction generalization capabilities of message-passing GNNs in larger test graphs. In Advances in Neural Information Processing Systems, 2022. (Cited on page 22)
