# Variational Causal Autoencoder for Interventional and Counterfactual Queries

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We propose the Variational Causal Autoencoder (VCAUSE), a novel class of variational graph autoencoders for causal inference in the absence of hidden confounders, when only observational data and the causal graph are available. Without making any structural assumption, VCAUSE mimics the necessary properties of a Structural Causal Model (SCM) to provide a framework for performing interventions (do-operator) and abduction-action-prediction steps. As a result, and as shown by our empirical results, VCAUSE provides a practical and accurate pipeline for estimating the interventional and counterfactual distributions of diverse SCMs. Finally, we apply VCAUSE to evaluate counterfactual fairness in classification problems and also to learn accurate and fair classifiers.

# 1 Introduction

Predicting causal effects of actions (interventions) is a central problem in scientific research in a broad variety of fields [4, 5, 7, 23, 51], and machine learning is no exception [44]. As an example, fundamental machine learning questions—such as fairness [6, 9, 19, 24, 25] and interpretability [17]—, are increasingly being formulated as causal queries.

Research on causal reasoning has predominantly focused on causal discovery, a.k.a. structure learning, aimed at discovering the underlying causal graph from data (see, e.g., [15, 30, 49, 60]). An alternative line of work instead aims to answer causal queries under different assumptions, e.g., assuming access to partial causal knowledge [17, 18] or to a randomized trial [16]. Here, we focus on the latter line of research, that is, on answering the following two types of causal questions: interventional queries, e.g., "What is the effect of a universal unconditional basic income of 1k EUR on the health of the population?"; and counterfactual queries, e.g., "Had Kim received an unconditional basic income of 1k EUR, what would have been the effect on Kim's health?"

Unfortunately, predicting causal effects from observational data alone is in general difficult and often requires strong and impractical causal assumptions. In this context, the Structural Causal Model (SCM) [39] is a framework that allows to answer causal queries from observational data, but requires complete causal knowledge. That is, knowledge not only on the parent-children (cause-effect) relationship between every pair of observed variables (i.e., on the causal graph), but also on how these relationships are (i.e., on the structural equations). As a consequence, randomized controlled studies are today still considered to be the gold standard for estimating causal effects. Unfortunately, real world experiments are often expensive to conduct, unethical, or directly impossible.

In this work, we aim at answering the above causal queries, when only observational data and the causal graph are available. Note that the causal graph can often be inferred from domain knowledge [62] or via one of the numerous approaches for causal discovery [27, 54]. We assume causal sufficiency, i.e., that there are no hidden confounders, which are unobserved variables that

affect more than one observed variable. We propose the novel Variational Causal Autoencoder (VCAUSE), a variational graph autoencoder that leverages the causal graph structure and yields accurate estimates of the observational, interventional and counterfactual distributions induced by an unknown causal model.

Importantly, we provide the necessary conditions for the design of the encoder and decoder graph neural networks (GNNs), so that the resulting VCAUSE behaves like an SCM. As a result, and without making any assumptions on the true structural equations, VCAUSE provides a practical framework to perform interventions (do-operator) and abduction-action-prediction steps, which are necessary to evaluate interventional and counterfactual queries.

We evaluate the performance of the proposed VCAUSE model in extensive experiments using observational data from different SCMs, with diverse causal graphs and structural equations. Our experiments show that VCAUSE outperforms competing methods [17, 18] at estimating not only the mean of the interventional/counterfactual distribution, but also the overall distribution, as shown by the quality of its samples (in terms of Maximum Mean Discrepancy, MMD). We finally show a use-case in which VCAUSE is used to assess counterfactual fairness of different classifiers trained on the German Credit dataset [10] as well as to learn accurate and counterfactually fair classifiers.

Related work. There are numerous works on causal discovery [15, 18, 27, 30, 33, 40, 49, 54, 56, 58, 60, 63]. In addition, extensive work focuses on interventional and/or counterfactual queries using non-parametric methods [1, 32, 46, 47], and more recently, tractable probabilistic models [59]. Moreover, deep generative models are enjoying increasing attention for causal queries in complex data [31, 35]. Existing approaches focus on estimating the Average Treatment Effect (ATE) by assuming a fixed causal graph that includes the treatment variable [19, 29, 42, 45, 53]; on discovering and intervening on the causal latent structure of the (e.g., image) data [19, 35, 37, 48, 56]; or on addressing interventional and/or counterfactual queries by fitting a conditional model for each observed variable given its causal parents [11, 17, 22, 36, 38]. In the most recent work related to ours [18], the authors propose CAREFL, an autoregressive normalizing flow (ANF) for causal discovery and queries, which focuses on bi-variable scenarios with affine relationships between observed and unobserved variables. In our experiments, we compare VCAUSE with CAREFL (as well as [17]) in more general settings. Finally, up to the best of our knowledge, GNNs have previously been used for causal discovery [58, 61], but have not yet been exploited to address counterfactual and interventional queries, like VCAUSE does.

# 2 Background

In this section, we first provide a brief overview on structural causal models (SCMs) and then introduce the main building block of VCAUSE, i.e., variational graph autoencoders (VGAEs).

# 2.1 Structural causal models

An  $\mathbf{SCM}\mathcal{M} = (p(\mathbf{U}),\tilde{\mathbf{F}})$  determines how a set of  $d$  endogenous (observed) random variables  $\mathbf{X}:=$ $\{X_1,\ldots X_d\}$  is generated from a set of exogenous (unobserved) random variables  $\mathbf{U}:= \{U_1,\dots U_d\}$  (with prior distribution  $p(\mathbf{U})$  ) via the set of structural equations  $\tilde{\mathbf{F}} = \{X_{i}:= \tilde{f}_{i}\left(\mathbf{X}_{\mathrm{pa}(i)},U_{i}\right)\}_{i = 1}^{d}$ . Here  $\mathbf{X}_{\mathrm{pa(i)}}$  refers to the set of variables directly causing  $X_{i}$ , i.e., parents of  $i$ . Every SCM  $\mathcal{M}$  is associated with a directed acyclic graph (DAG): a causal graph  $\mathcal{G}:= (\mathbf{X},\mathbf{E})$ , for which the nodes (vertices) correspond to endogenous variables  $\mathbf{X}$  and the directed edges  $\mathbf{E}$  account for the causal parent-child relationship between variables [39]. Given an SCM, there are two types of causal queries of general interest: interventional queries, e.g., "What would happen to the population  $\mathbf{X}$ , if variable  $X_{i}$  would be set to a fixed value  $\alpha ?$ "; and counterfactual queries, e.g., "What would have happened to a specific factual sample  $\mathbf{x}^F$ , had  $X_{i}$  been set to a value  $\alpha ?$ ".

More in detail, *interventional queries* aim to evaluate changes in the causal world, or equivalently, manipulations of a subset of the endogenous variables  $\mathcal{I} \subseteq [d] := \{1, \dots, d\}$  at the population level. Interventions on an SCM  $\mathcal{M}$  are often represented with the do-operator  $do(X_{i} = \alpha_{i})$  and lead to a new distribution over the set of endogenous variables  $p(\mathbf{X} \mid do(X_{i} = \alpha_{i}))$ , which is referred to as the *interventional distribution*. In  $\mathcal{G}$  an intervention removes incoming edges to node  $i$  and sets  $X_{i} = \alpha$  (see Figure 1c). A counterfactual query for a given factual instance  $\mathbf{x}^{F}$  aims to estimate what would have happened had  $\mathbf{X}_{\mathcal{I}}$  instead taken value  $\alpha$ . This effect is captured by the

counterfactual distribution  $p(\mathbf{x}^{CF} \mid \mathbf{x}^F, do(X_{\mathcal{I}} = \alpha))$ , which can be computed using the abduction-action-prediction approach by Pearl [39]. Refer to Section 3 for further details on the computation of the interventional and counterfactual distributions.

$$
\mathbf {U} \sim p (\mathbf {U})
$$

$$
\tilde {f} _ {1}: X _ {1} = U _ {1}
$$

$$
\tilde {f} _ {2}: X _ {2} = 2 X _ {1} + U _ {2}
$$

$$
\tilde {f} _ {3}: X _ {3} = 3 X _ {1} - 4 X _ {2} + U _ {3}
$$

(a)  $\operatorname{SCM} \mathcal{M} := \{p(\mathbf{U}), \tilde{\mathbf{F}}\}$

![](images/f417e13b25642e28e83d99b7d1bece9ddb6de0603166d6f2ae85b8ba19255215.jpg)  
Figure 1: Example of (a) a triangle SCM  $\mathcal{M}$  with  $d = |\mathbf{X}| = 3$  endogenous variables; (b) corresponding causal graph  $\mathcal{G}$  and (c) illustration of an intervention  $do(X_2 = \alpha)$  on the causal graph. Green arrows highlight the direct causal path from  $X_{1}$  to  $X_{3}$ , and red arrows the indirect causal path via  $X_{2}$ .  
(b)  $\mathcal{G}$  without intervention

![](images/ea10568ea875fc99b084a8bd60dac51611e39a7c963a45bd7e536d89ca58a3ec.jpg)  
(c)  $\mathcal{G}$  with intervention

# 2.2 Variational Graph Autoencoder and Graph Neural Networks

Variational Autoencoders (VAEs) [20] are powerful latent variable models based on neural networks (NNs) for jointly i) learning complex and expressive density estimators  $p(\mathbf{X}) \approx \int p_{\theta}(\mathbf{X} \mid \mathbf{Z}) p(\mathbf{Z}) d\mathbf{Z}$ , where the likelihood function (a.k.a. decoder) is parameterized using a NN with parameters  $\theta$ ; and ii) performing approximate posterior inference over the latent variables  $\mathbf{Z}$  using a variational distribution (a.k.a. encoder)  $q_{\phi}(\mathbf{Z} \mid \mathbf{X})$  parameterized using a NN with parameters  $\phi$ . The parameters  $\theta$  and  $\phi$  are usually learned by maximizing a lower bound on the log-evidence [3, 34, 41, 52].

Variational Graph Autoencoders (VGAEs) [21] extend VAEs to account for graph-structure information on the data [58]. VGAEs define a (potentially multidimensional) latent variable  $Z_{i}$  per observed variable  $X_{i}$ , i.e.,  $\mathbf{Z} \coloneqq \{Z_{1},\ldots ,Z_{d}\}$ . Additionally, VGAEs rely on an adjacency matrix  $\mathbf{A}$ , which is used by two Graph Neural Networks (GNNs), one for the encoder and one for the decoder, to enforce structure on the posterior approximation  $q_{\phi}(\mathbf{Z} \mid \mathbf{X},\mathbf{A})$  and the likelihood  $p_{\theta}(\mathbf{X} \mid \mathbf{Z},\mathbf{A})$ . More in detail,  $\mathbf{A} \in \{0,1\}^{d\times d}$  encodes the graph structure among the observed variables  $\mathbf{X} \coloneqq \{X_{1},\dots X_{d}\}$ , so that  $A_{ij} = 1$  if there is a directed edge from  $X_{j}$  to  $X_{i}$ , and  $A_{ij} = 0$  otherwise. Hence,  $\mathbf{A}$  determines which variables  $X_{i}$  influence  $Z_{j}$  ( $i,j \in [d]$ ), and vice versa.

Graph Neural Networks (GNNs) have generated a lot of attention during the last years, as they achieved significant improvements in graph representation learning [2, 12, 14, 43, 57], While the taxonomy of GNNs is immense [55], in this work we focus on message-passing GNNs which allow us to work with directed graphs. In its most general form, a message-passing GNN calculates the output  $h_i^l$  for node  $i$  in layer  $l$  in three steps: i) compute the set of incoming messages arriving to node  $i$  from its neighbors  $\mathcal{N}_i = \{X_j \mid A_{ij} = 1\}$  using a message function  $f^m$  (a NN with parameters  $\theta_m^l$ ), that is  $\{m_{ij}^l\}_{j \in \mathcal{N}_i} = \{f_i^m(h_i^{l-1}, h_j^{l-1}; \theta_m^l) \mid j \in \mathcal{N}_i\}$ ; ii) combine the set of messages into a single message  $M_i^l := f^a(\{m_{ij}^l\}_j)$  using an aggregation function  $f^a$  (e.g. adding up the messages); and iii) update the node state  $h_i^l := f^u(h_i^{l-1}, M_i^l; \theta_u^l)$ , using an update function  $f^u$  (a NN with parameters  $\theta_u^l$ ). As a result, the output  $h_i^l$  can be written as

$$
h _ {i} ^ {l} = f ^ {u} \left(h _ {i} ^ {l - 1}, f ^ {a} \left(\left\{f ^ {m} \left(h _ {i} ^ {l - 1}, h _ {j} ^ {l - 1}; \theta_ {m} ^ {l}\right) \mid j \in \mathcal {N} _ {i} \right\}\right); \theta_ {u} ^ {l}\right). \tag {1}
$$

Note that the above expression assures that the output for each node  $i$  is computed using information from its neighbors  $\mathcal{N}_i$  according to A. Moreover, if the GNN has  $N_h$  hidden layers, then the output for node  $i$  not only depends on its direct neighbors  $\mathcal{N}_i$ , but also on its neighbors up to order  $N_h + 1$  (hops). As an example, if  $N_h = 0$  then the output for each node only depend on its direct neighbors (parents). If instead  $N_h = 1$ , then the output for each node depends on 2-hop neighbors (grand-parents). For a detailed description of GNNs, please refer to Appendix A.

# 3 Observational, interventional and counterfactual distributions

In this section, we introduce the observational, interventional and counterfactual distributions (triggered by any intervention  $d o(\mathbf{X}_{\mathcal{I}} = \boldsymbol{\alpha})$ ) that are induced from an SCM  $\mathcal{M} \coloneqq \{p(\mathbf{U}), \tilde{\mathbf{F}}\}$ . Specifically, we summarize the main properties of an SCM that will allow us to propose a novel class of

VGAEs, the variational causal autoencoders (VCAUSE), to compute accurate estimates of these distributions using observational data and a known causal graph. To this end, we assume the absence of hidden confounders, i.e., we assume that  $p(\mathbf{U}) = \prod_{i=1}^{d} p(U_i)$ .

Observational distribution. The SCM  $\mathcal{M}$  determines the observational distribution  $p(\mathbf{X})$  over the set of endogenous variables  $\mathbf{X} = \{X_1, \ldots, X_d\}$ , which satisfies causal factorization [44], i.e.,  $p(\mathbf{X}) = \prod_{i=1}^{d} p(X_i \mid \mathbf{X}_{\mathrm{pa}(i)})$ . That is, after marginalizing out the exogenous variables  $\mathbf{U}$ , the distribution of each endogenous variable  $X_i$  depends only on its parents, i.e.,  $\mathbf{X}_{\mathrm{pa}(i)}$ . The observational distribution can alternatively be written only in terms of the exogenous variables  $\mathbf{U}$  as

$$
p (\mathbf {X}) = \int \mathbf {F} (\mathbf {U}) p (\mathbf {U}) d \mathbf {U}, \tag {2}
$$

where  $\mathbf{F}:\mathbf{U}\to \mathbf{X}$  corresponds to the set of structural equations, equivalent to  $\tilde{\mathbf{F}}$ , that directly transform the exogenous variables  $\mathbf{U}$  into the endogenous variables  $\mathbf{X}$ . Let us denote by  $an(i)$  the set of indexes of the ancestors of  $i$ , and  $an^{*}(i)\coloneqq an(i)\cup \{i\}$ . Then, the causal factorization induced by the SCM  $\mathcal{M}$  leads to the following property of  $\mathbf{F}(\mathbf{U})$ :

Property 1. Each endogenous variable  $X_{i}$  can be expressed as a function of its exogenous variable  $U_{i}$  and the ones of all its causal ancestors, i.e.,  $\mathbf{F}(\mathbf{U}) \coloneqq \{X_{i} = f_{i}(\{U_{j} \mid j \in an^{*}(i)\})\}$ . This, together with the causal sufficiency assumption, implies that  $X_{i}$  is statistically independent of  $U_{j}, \forall j \notin an^{*}(i)$ .

Interventional distribution. As stated in Section [2.1] interventions on a set of variables  $\mathcal{I}$  can be performed using the do-operator, which can be seen as a mapping  $do(\mathbf{X}_{\mathcal{I}} = \alpha): \mathcal{M} \mapsto \mathcal{M}^{\mathcal{I}} = (p(\mathbf{U}), \tilde{\mathbf{F}}^{\mathcal{I}})$  where  $\tilde{\mathbf{F}}^{\mathcal{I}} = \{\tilde{f}_j \mid j \notin \mathcal{I}\} \cup \{\alpha_i \mid i \in \mathcal{I}\}$ . As above, we can represent the resulting set of intervened structural equations  $\mathbf{F}^{\mathcal{I}} = \{f_j \mid j \notin \mathcal{I}\} \cup \{\alpha_i \mid i \in \mathcal{I}\}$  in terms of only the exogenous variables  $\mathbf{U}$ , so that we can write the interventional distribution as:

$$
p (\mathbf {X} \mid d o (\mathbf {X} _ {\mathcal {I}} = \boldsymbol {\alpha})) = \int \mathbf {F} ^ {\mathcal {I}} (\mathbf {U}) p (\mathbf {U}) d \mathbf {U}. \tag {3}
$$

Assuming an intervention  $do(\mathbf{X}_{\mathcal{I}} = \alpha)$  on  $\mathcal{M}$ , then the resulting structural equations  $\mathbf{F}^{\mathcal{I}}(\mathbf{U})$  satisfy: Property 2. After an intervention  $do(\mathbf{X}_{\mathcal{I}} = \alpha)$  on  $\mathcal{M}$ , all the causal paths from  $U_{j} \forall j \in an^{*}(i)$  to  $X_{i}$  that include an intervened variable in  $\mathbf{X}_{\mathcal{I}}$  (i.e., the causal paths where  $\mathbf{X}_{\mathcal{I}}$  is a mediator) are severed in  $\mathbf{F}^{\mathcal{I}}$ , while the rest of causal paths remain untouched.

The above property is illustrated in Figure  $\square$ , where we can easily observe that after an intervention  $do(X_{2} = \alpha)$ , the indirect causal path (in red) from  $X_{1}$ , and thus from  $U_{1}$ , to  $X_{3}$  via  $X_{2}$  is severed, while the direct path (in green) remains.

Counterfactual distribution. Assuming the SCM  $\mathcal{M} = \{p(\mathbf{U}),\tilde{\mathbf{F}}\}$  to be known, the following three steps defined by Pearl [39] allow us to compute counterfactuals  $\mathbf{x}^{CF}$  as: i) Abduction: infer the values of the exogenous variables  $\mathbf{U}$  for a factual sample  $\mathbf{x}^F$ , i.e., compute  $p(\mathbf{U}\mid \mathbf{x}^F)$ ; ii) Action: intervene with  $do(\mathbf{X}_{\mathcal{I}} = \alpha):\mathcal{M}\mapsto \mathcal{M}^{\mathcal{I}} = (p(\mathbf{U}),\tilde{\mathbf{F}}^{\mathcal{I}})$ ; and iii) Prediction: use the posterior distribution  $p(\mathbf{U}\mid \mathbf{x}^F)$  and the new structural equations  $\tilde{\mathbf{F}}^{\mathcal{I}}$  to compute  $p(\mathbf{x}^{CF}\mid \mathbf{x}^F)$ . The prediction step can be alternatively computed using the new set of structural equations  $\mathbf{F}^{\mathcal{I}}$  defined in terms of the exogenous variables  $\mathbf{U}$ , so that we can write the counterfactual distribution as:

$$
p \left(\mathbf {x} ^ {C F} \mid \mathbf {x} ^ {F}, d o \left(\mathbf {X} _ {\mathcal {I}} = \boldsymbol {\alpha}\right)\right) = \int \mathbf {F} ^ {\mathcal {I}} (\mathbf {U}) p \left(\mathbf {U} \mid \mathbf {x} ^ {F}\right) d \mathbf {U}. \tag {4}
$$

Importantly, the resulting posterior distribution  $p(\mathbf{U} \mid \mathbf{x}^F)$  satisfies:

Property 3. In the abduction step, statistical independence implies that conditioned on the endogenous variables of the factual sample  $\mathbf{x}^F$ , each exogenous variable  $U_{i}$  is independent of the factual value  $x_{j}^{F}$  if  $j\neq i$  and the variable  $X_{j}$  is not a parent of  $X_{i}$ , i.e.,  $j\notin pa^{*}(i)\coloneqq pa(i)\cup \{i\}$ .

# 4 Variational Causal Autoencoder (VCAUSE)

In this section, we present a novel variational causal graph autoencoder (VCAUSE) to approximate the observational, interventional and counterfactual distributions given in (2), (3) and (4), respectively.

While the underlying SCM  $\mathcal{M}$  is unknown, we assume access to: the causal graph  $\mathcal{G}$  and observational data  $\{\mathbf{x}_n\}_{n=1}^N$ , i.e., i.i.d. samples of the observational distribution induced by  $\mathcal{M}$ .

Definition 4.1. (VCAUSE). Given a causal graph  $\mathcal{G}$  over a set of endogenous variables  $\mathbf{X} = \{X_1, \ldots, X_d\}$ , which establishes the set of parents  $pa^*(i)$  for each variable  $X_i$  (including the  $i$ -th node). A variational causal graph autoencoder (VCAUSE) is defined by:

- A causal adjacency matrix  $\mathbf{A}$ , which is a  $d \times d$  binary matrix with elements  $A_{ij} = 1$  if  $j \in pa^{*}(i)$ , i.e., when  $i = j$  or  $j$  is a parent of  $i$ . Otherwise,  $A_{ij} = 0$ .  
- A prior distribution  $p(\mathbf{Z}) = \prod_{i}p(Z_{i})$  over the set of latent variables  $\mathbf{Z} = \{Z_1,\dots,Z_d\}$ .  
- A decoder  $p_{\theta}(\mathbf{X} \mid \mathbf{Z}, \mathbf{A})$ , which is a GNN (parameterized by  $\theta$ ) that takes as input the set of latent variables  $\mathbf{Z}$  and the causal adjacency matrix  $\mathbf{A}$ , and outputs the parameters of the likelihood  $p_{\theta}(\mathbf{X} \mid \mathbf{Z}, \mathbf{A})$ .  
- An encoder  $q_{\phi}(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})$ , which is a GNN (parameterized by  $\phi$ ) that takes as input the endogenous variables  $\mathbf{X}$  and the causal adjacency matrix  $\mathbf{A}$ , and outputs the parameters of the posterior approximation  $q_{\phi}(\mathbf{Z} \mid \mathbf{X}, \mathbf{A})$ .

Given observational data  $\{\mathbf{x}_n\}_{n = 1}^N$ , one may learn the parameters  $\theta$  and  $\phi$  that best estimate the density  $p(\mathbf{X})$ . We here rely on the partially importance weighted auto-encoder (PIWAE) [41].

Next, we discuss how to design VCAUSE such that it is able to capture the observational, interventional, and counterfactual distribution induced by an unknown SCM. Importantly, we derive the necessary conditions on the design of both the encoder and decoder GNNs such that VCAUSE fulfills the SCM properties introduced in Section 3.

# 4.1 Observational distribution

VCAUSE approximates the observational distribution in (2) using the generative model as

$$
p (\mathbf {X}) \approx \int p _ {\theta} (\mathbf {X} \mid \mathbf {Z}, \mathbf {A}) p (\mathbf {Z}) d \mathbf {Z} = \int \prod_ {i = 1} ^ {d} p _ {\theta} \left(X _ {i} \mid \mathbf {Z}, \mathbf {A}\right) p (\mathbf {Z}) d \mathbf {Z}. \tag {5}
$$

Figure 2a depicts this generative process. If we compare (5) with the true observational distribution in (2), we observe that the latent variables  $\mathbf{Z}$  play a similar role to the exogenous variables  $\mathbf{U}$ , and the decoder  $p_{\theta}(\mathbf{X} \mid \mathbf{Z}, \mathbf{A})$  plays a similar role to the structural equations  $\mathbf{F}$ . Yet, we remark that  $\mathbf{Z}$  does not need to correspond to the exogenous variables, i.e.,  $p(\mathbf{U}) \neq p(\mathbf{Z})$ , in order for (5) to provide a good approximation of the observational distribution in (2). In fact, standard VAEs perform accurate density estimation using observational data, without the need for capturing causal information. However, in this paper, we seek to ensure that our observational distribution induced by VCAUSE complies causal factorization (Property 1). To that end, we need to make sure that  $p_{\theta}(X_i \mid \mathbf{Z}, \mathbf{A}) = p_{\theta}(X_i \mid \mathbf{Z}_{an^*(i)})$ . That is,  $X_i$  depends only on  $Z_j$  if  $j = i$  or  $X_j$  is an ancestor of  $X_i$  in the causal graph. To fulfill this property, the GNN of the decoder should satisfy the following:

Proposition 1. (Causal factorization). VCAUSE satisfies causal factorization,  $p_{\theta}(\mathbf{X} \mid \mathbf{Z}, \mathbf{A}) = \prod_{i} p_{\theta_i}(X_i \mid \mathbf{Z}_{an^*(i)})$ , if and only if the number of hidden layers in the decoder is greater or equal than  $\delta - 1$ , with  $\delta$  being the longest shortest directed path between any two endogenous nodes.

The above proposition (proved in Appendix B) is based on the fact that, in a GNN with  $N_{h}$  hidden layers (and  $N_{h} + 1$  layers in total), the output for the  $i$ -th node depends on its neighbors of up to  $N_{h} + 1$  hops. As an example, consider the following chain causal graph:  $X_{1} \to X_{2} \to X_{3}$ , such that  $\delta = 2$ . In the decoder, the first layer yields a hidden representation for the 3-rd node  $h_3^1 \coloneqq f(f(Z_2), Z_3)$  that only depends on  $Z_{2}$  and  $Z_{3}$ . Thus, we need a second layer for its output  $h_3^2 \coloneqq f(h_2, Z_3) = f(f(f(Z_1), Z_2), Z_3)$  to depend on  $Z_{1}$  (note that  $X_{1}$  is an ancestor of  $X_{3}$ ).

# 4.2 Interventional distribution

VCAUSE approximates the interventional distribution in (3) as (illustrated Figure 3):

$$
p (\mathbf {X} \mid d o (\mathbf {X} _ {\mathcal {I}} = \boldsymbol {\alpha})) \approx \int p _ {\theta} (\mathbf {X} \mid \{Z _ {i} \} _ {i \notin \mathcal {I}}, \{Z _ {i} ^ {\mathcal {I}} \} _ {i \in \mathcal {I}}, \mathbf {A} ^ {\mathcal {I}}) p (\mathbf {Z}) q _ {\phi} (\mathbf {Z} ^ {\mathcal {I}} \mid \mathbf {A} ^ {\mathcal {I}}, \mathbf {X} _ {\mathcal {I}}) d \mathbf {Z}, \tag {6}
$$

where the do-operator is performed on the causal adjacency matrix as  $\text{do}(X_{\mathcal{I}} = \alpha) : \mathbf{A} \mapsto \mathbf{A}^{\mathcal{I}} = \{A_{ij}\}_{\forall i \notin \mathcal{I}, j} \cup \{A_{ij} = 0\}_{\forall i \in \mathcal{I}, j}$ . This ensures that  $X_i$  for  $i \in \mathcal{I}$  is independent of  $Z_j$  for all

![](images/55668c551733420963f5d068c9683f3540f4bfca39b657c79b5cac5980c3f41b.jpg)  
Figure 2: VCAUSE generation of (a) observational, (b) interventional, and (c) counterfactual samples. The 'hat' in  $\hat{\mathbf{X}}$  and  $\hat{\mathbf{x}}^{CF}$  indicate that they are sample estimates of the true random variables.

$j \neq i$ . Note that in order for (6) to be able to approximate the interventional distribution in (3), an intervention on a variational causal autoencoder should satisfy Property 2, i.e.:

Proposition 2. (Causal interventions). VCAUSE can capture causal interventions if and only if the number of hidden layers in its decoder is greater than or equal to  $\gamma - 1$ , with  $\gamma$  being the longest directed path between any two endogenous nodes in  $\mathcal{G}$ .

To illustrate this, Figure 3 depicts how messages are exchanged in a one-hidden-layer decoder GNN corresponding to the causal graph  $\mathcal{G}$  in Figure 1 triangle with  $\gamma = 2$  both a without and (b) with an intervention on  $X_{2}$  .We highlight in green the direct messages (sent via direct causal path in  $\mathcal{G}$  ),and in red the indirect messages sent via indirect causal path in  $\mathcal{G}$  ) from  $Z_{1}$  to  $X_{3}$  .Observe that,similarly to Figure 1 in (a) there is an indirect path (via  $h_2$  ) from  $Z_{1}$  to  $X_{3}$  ; while in (b) this path is severed. Hence, the hidden layer  $(h_1,h_2,h_3)$  allows to differentiate between direct and indirect paths and thus to capture interventional effects.

![](images/1ff0ebb38319d9a4ac45c1dffe5b0be93fee3ff4fe00eda9d923027a9175534a.jpg)  
(a) Original

![](images/cb9342aff51061f20a54aa806e3bf5aa661b1f4974892aa18154b8074792957f.jpg)  
Figure 3: VCAUSE decoder (a) with and (b) without intervening on  ${X}_{2}$  . Arrows indicate message passing in the GNN corresponding to direct (green) and indirect (red) causal paths in Figure (1).  
(b) Intervened

As the condition in Proposition 2 is more restrictive than the one in Proposition 1, in order for VCAUSE to be able to capture observational and interventional distributions, it should satisfy that:

Design condition 1: The decoder GNN of VCAUSE has at least as many hidden layers as  $\gamma - 1$ , with  $\gamma$  being the longest directed path in the causal graph  $\mathcal{G}$ .

# 4.3 Counterfactual distribution

VCAUSE approximates the counterfactual distribution in (4) as (illustrated in Figure 2c):

$$
p(\mathbf{x}^{CF}\mid do(\mathbf{X}_{\mathcal{I}} = \boldsymbol {\alpha}),\mathbf{x}^{F})\approx \\ \underbrace{\int\underbrace{p_{\theta}(\mathbf{X}\mid\{Z_{i}^{F}\}_{i\notin\mathcal{I}},\{Z_{i}^{\mathcal{I}}\}_{i\in\mathcal{I}},\mathbf{A}^{\mathcal{I}})q_{\phi}(\mathbf{Z}^{\mathcal{I}}\mid\mathbf{x}^{\mathcal{I}},\mathbf{A}^{\mathcal{I}})}_{action}\underbrace{q_{\phi}(\mathbf{Z}^{\mathbf{F}}\mid\mathbf{x}^{F},\mathbf{A})}_{abduction}d\mathbf{Z}}_{prediction},
$$

where  $\mathbf{x}^F$  represents a sample from  $\mathbf{X}$  for which we seek to compute the distribution over counterfactual  $\mathbf{x}^{CF}$ . Note here that two different passes of the encoder are necessary: one for the abduction step of the factual instance  $q_{\phi}(\mathbf{Z}^{\mathbf{F}} \mid \mathbf{x}^F, \mathbf{A})$ ; and another one for the action step (intervention)  $q_{\phi}(\mathbf{Z}^{\mathcal{I}} \mid \mathbf{x}^{\mathcal{I}}, \mathbf{A}^{\mathcal{I}})$  with  $x_i^{\mathcal{I}} = \alpha_i \forall i \in \mathcal{I}$  (we remark that the rest of the values in  $\mathbf{x}^{\mathcal{I}}$  do not affect the overall counterfactual computation). We then evaluate the likelihood, making sure that the resulting counterfactual sample  $\mathbf{x}^{CF}$  only depends on the  $\{Z_i^F\}_{i \in \mathcal{I}} \subseteq \mathbf{Z}^{\mathbf{F}}$  and  $\{Z_i^{\mathcal{I}}\}_{i \in \mathcal{I}} \subseteq \mathbf{Z}^{\mathcal{I}}$ . Importantly, in order for VCAUSE to be able to approximate the counterfactual distribution, we need its abduction (and action) step(s) to comply with Property 3, i.e.:

Proposition 3. (Abduction). The abduction step of an observed sample  $\mathbf{x} = \{x_{1},\ldots ,x_{d}\}$  in a variational causal autoencoder satisfies that for all  $i$  the posterior of  $Z_{i}$  is independent on the subset  $\{x_j\}_{j\notin pa^* (i)}\subseteq \mathbf{x}$ , if and only if the encoder GNN has no hidden layers.

The above result (proved in Appendix B) can be shown by the message passing algorithm computed by the encoder GNN, and leads to the second condition that VCAUSE should satisfy by design:

Table 1: Evaluation of the observational and interventional distributions generated by VCAUSE with different numbers of hidden layers  $N_{h}$ . All metrics are shown in percentage  $(\%)$ .  

<table><tr><td rowspan="2">Nh</td><td colspan="2">collider (δ = 1, γ = 1)</td><td colspan="2">triangle (δ = 1, γ = 2)</td><td colspan="2">chain (δ = 2, γ = 2)</td></tr><tr><td>MMD Obs. (%)</td><td>MMD Inter.(%)</td><td>MMD Obs.(%)</td><td>MMD Inter.(%)</td><td>MMD Obs.(%)</td><td>MMD Inter.(%)</td></tr><tr><td>0</td><td>1.37 ± 0.54</td><td>0.90 ± 0.19</td><td>2.20 ± 0.74</td><td>4.03 ± 0.42</td><td>5.58 ± 1.01</td><td>8.07 ± 0.53</td></tr><tr><td>1</td><td>0.86 ± 0.34</td><td>0.95 ± 0.28</td><td>1.05 ± 0.38</td><td>2.35 ± 0.35</td><td>1.4 ± 0.31</td><td>1.56 ± 0.4</td></tr><tr><td>2</td><td>1.0 ± 0.50</td><td>0.91 ± 0.16</td><td>1.20 ± 0.63</td><td>2.33 ± 0.29</td><td>1.67 ± 0.61</td><td>1.46 ± 0.29</td></tr></table>

Design condition 2: The encoder GNN of VCAUSE has no hidden layers.

Note that while the above condition may look restrictive and limiting the capacity of our encoder, we may choose arbitrarily complex NNs to model the message  $f^{m}$  and update  $f^{u}$  functions, as well as one or more aggregation functions  $f^{a}$ , e.g., sum, mean or max, to model the encoder [8].

# 4.4 Practical considerations

Next, we briefly discuss practical implementation considerations to handle complex causal models, which often appear in real world applications—see the causal graph of the German Credit dataset [10] in Section 6 for an example. For further details on VCAUSE implementation, refer to Appendix C.

Heterogeneous endogenous variables: In general GNNs are parametrized such that the parameters of the message  $f^{m}$  and update  $f^{u}$  functions are shared for all the nodes and edges in the graph. However, similarly as in the structural causal equations  $\mathbf{F}$ , we can define a different message function  $f_{ij}^{m}$  for every edge in the causal graph by assuming a different set of parameters  $\theta_{mij}$  per edge in (1). Similarly, we can also assume different update functions  $f_{i}^{u}$  for each node  $i$ , by considering different update parameters  $\theta_{ui}$  for each node. This allows us to use different functions for each node, and thus model heterogeneous endogenous variables, in terms of their continuous/discrete distribution, and also of their structural equations, e.g., linear/non-linear.

Heterogenous causal nodes: So far, we have modeled each endogenous variable  $X_{i}$  as a node in the causal graph  $\mathcal{G}$ , and thus in the VCAUSE GNNs. However, in some application domains the relationships between a subset of variables may be unknown, or they may be affected by hidden confounders, leading to an undirected path between them. In such cases, the subset of  $(k_{i})$  variables is modeled as a multidimensional and potentially heterogeneous node  $\mathbf{X}_{i} = \{X_{i1},\dots ,X_{ik_{i}}\}$ . Note that all the variables in the multidimensional node  $\mathbf{X}_{i}$  share the same latent random variable  $Z_{i}$ .

# 5 Evaluation

In this section, we conduct extensive experiments to evaluate the performance of VCAUSE at estimating the outcomes of causal queries. Please refer to Appendix D for a complete description of the experimental set-up. Moreover, to ease the reproducibility of our experiments, our code is publicly available at https://github.com/XXXX/XXXXXX.

Datasets. We consider different synthetic causal graphs that differ in the number of nodes  $d$ , diameter  $\delta$ , and longest path  $\gamma$ : synthetic collider ( $d = 3$ ,  $\delta = 1$ ,  $\gamma = 1$ ), M-graph ( $d = 3$ ,  $\delta = 1$ ,  $\gamma = 1$ ), triangle ( $d = 3$ ,  $\delta = 1$ ,  $\gamma = 2$ ), chain ( $d = 3$ ,  $\delta = 2$ ,  $\gamma = 2$ ), and a semi-synthetic loan ( $d = 7$ ,  $\delta = 2$ ,  $\gamma = 3$ ) from [17]. For all of the synthetic datasets (i.e., except loan), we consider three different types of structural equations with increasing complexity: linear additive noise (LIN), non-linear additive noise (NLIN) and non-additive noise (NADD).

Metrics. We evaluate the observational distribution using the Maximum Mean Discrepancy (MMD) [13] as distance-measure between the true and estimated distributions as a whole, i.e., the lower the MMD the better the distributions match. For the interventional distribution, we additionally report the estimation squared error for the mean and for the standard deviation (MeanE and StdE respectively) for the children of the intervened variables. For the counterfactual distribution we report the mean square error (MSE) as well as the standard deviation of the squared error (SSE) between the true and the estimated counterfactual value. We compute all results over 10 independent runs.

Validating VCAUSE design conditions. In a first step we empirically validate our design choices for the VCAUSE encoder and decoder. We show how the number of hidden layers  $N_{h}$  in the decoder affect the quality of the estimation of the observational and interventional distributions for three

Table 2: Performance of different methods at estimating the observational, interventional and counterfactual of different SCMs. All metrics are shown in percentage (\%).  

<table><tr><td rowspan="2" colspan="2">SCM</td><td rowspan="2">Model</td><td>Obs.</td><td colspan="3">Interventional</td><td colspan="2">Counterfactuals</td></tr><tr><td>MMD (%)</td><td>MMD (%)</td><td>MeanE (%)</td><td>StdE (%)</td><td>MSE (%)</td><td>SSE (%)</td></tr><tr><td rowspan="9">triangle</td><td rowspan="3">LIN</td><td>MultiCVAE</td><td>1.07±0.88</td><td>4.92±2.00</td><td>0.81±0.33</td><td>24.39±0.20</td><td>15.52±4.69</td><td>12.78±5.07</td></tr><tr><td>CAREFL</td><td>5.51±0.80</td><td>3.63±0.22</td><td>0.18±0.05</td><td>50.10±0.79</td><td>5.11±0.87</td><td>6.18±0.81</td></tr><tr><td>VCAUSE</td><td>1.26±0.68</td><td>2.21±0.26</td><td>0.65±0.12</td><td>24.51±0.09</td><td>11.68±0.69</td><td>7.62±0.42</td></tr><tr><td rowspan="3">NLIN</td><td>MultiCVAE</td><td>1.15±0.83</td><td>7.21±3.90</td><td>0.57±0.29</td><td>17.58±0.26</td><td>12.92±4.11</td><td>10.03±5.33</td></tr><tr><td>CAREFL</td><td>5.37±1.18</td><td>8.15±0.76</td><td>1.14±0.38</td><td>60.48±1.36</td><td>8.03±1.53</td><td>8.95±1.42</td></tr><tr><td>VCAUSE</td><td>1.55±0.90</td><td>6.26±1.31</td><td>0.85±0.16</td><td>17.41±0.09</td><td>12.10±0.95</td><td>8.17±0.64</td></tr><tr><td rowspan="3">NADD</td><td>MultiCVAE</td><td>2.15±0.58</td><td>43.63±2.73</td><td>0.18±0.07</td><td>19.14±1.75</td><td>24.45±1.62</td><td>38.23±3.83</td></tr><tr><td>CAREFL</td><td>6.14±1.33</td><td>76.84±14.78</td><td>2.59±3.76</td><td>112.65±6.08</td><td>8.32±0.93</td><td>39.82±0.88</td></tr><tr><td>VCAUSE</td><td>2.54±1.18</td><td>8.87±1.52</td><td>0.09±0.04</td><td>20.94±1.72</td><td>10.36±0.78</td><td>17.82±1.20</td></tr><tr><td rowspan="3">loan</td><td rowspan="3">,</td><td>MultiCVAE</td><td>76.18±12.61</td><td>188.35±9.05</td><td>16.84±5.64</td><td>60.29±3.39</td><td>72.41±4.75</td><td>38.69±1.16</td></tr><tr><td>CAREFL</td><td>9.28±2.15</td><td>9.54±1.82</td><td>3.55±2.48</td><td>28.94±1.15</td><td>32.54±0.21</td><td>17.68±0.34</td></tr><tr><td>VCAUSE</td><td>1.09±0.24</td><td>1.41±0.16</td><td>0.40±0.09</td><td>9.58±0.06</td><td>30.06±0.14</td><td>14.22±0.11</td></tr></table>

SCMs, with different values of longest shortest directed path  $\delta$  and longest directed path  $\gamma$ . In Table we observe that as expected: i) the collider ( $\delta = \gamma = 1$ ) does not need any hidden layer to provide accurate estimate of both the observational and interventional distributions. In contrast, the triangle ( $\delta = 1, \gamma = 2$ ), which according to Proposition needs at least one hidden layer to get a more accurate estimate of the interventional distribution (while an improvement in the observational is not as evident). Finally, as stated by Proposition and the chain ( $\delta = \gamma = 2$ ) requires at least one hidden layer to accurately approximate both the observational and interventional distributions.

# 5.1 Estimating interventional and counterfactual distributions

In the following we evaluate the potential of VCAUSE to model interventional and counterfactual queries. We consider interventions of the form  $do(x_{i} = \alpha_{i})$  for several values of  $\alpha_{i}$  on both root and non-root nodes. Here we report the results for the triangle and loan graphs. Refer to Appendix E for the remaining results.

Baselines. We compare our VCAUSE with two competing methods: i) MultiCVAE, which trains a conditional VAE for each endogenous variable that is not a root node in the causal graph [17]; and ii) CAREFL [18], which relies on autoregressive causal flows to estimate counterfactual queries.

Results for interventional distributions. Table2(middle columns) reports the MMD, MeanE, and StdE for the interventional distribution. Here we can observe that VCAUSE consistently outperforms other methods in terms of MMD. Note that the three methods provide comparable results in capturing the mean of the interventional distribution (MeanE) (except for the more complex loan graph, where VCAUSE outperforms the others). However, it can also be seen that CAREFL and MultiCVAE often fail to capture the standard deviation of the interventional distribution (StdE), while VCAUSE provides a more accurate estimate of the overall interventional distribution.

Results for the counterfactuals. Table 2 also reports the results for the counterfactual distribution. Here, we first observe that MultiCVAE slightly underperforms the other two models. Second, we observe that CAREFL provides more accurate estimates than VCAUSE in terms of MSE, which may be explained by the fact that CAREFL performs exact inference. However, CAREFL presents high variance in its results (see SSE). Note that to perform interventions, CAREFL sets the parents of the intervened vari-

ables to zero, which may not completely severe the causal paths to the intervened nodes. In contrast, as further illustrated in Figure 4, VCAUSE leads to consistent counterfactual estimations across factual samples and interventions. Figure 4 also shows that CAREFL fails severely for some intervention values, despite of intervening on a root node.

![](images/6e098f7f965934afcd5391f0c26ff405bedfe61776ed59da27de8f421075fd1c.jpg)  
Figure 4: Example of counterfactuals for a factual  $\mathbf{x}^F$  from the test set of the triangle NLIN and  $do(x_{1} = \alpha)$

# 6 Use case: counterfactual fairness

Finally, we showcase the practical use of VCAUSE for assessing counterfactual fairness and also for training a counterfactually fair classifier. To this end, we use the German Credit dataset publicly available at the UCI repository [50]. We rely on the causal model with the following random variables  $\mathbf{X}$  as proposed in [6] (see Figure 5): sensitive feature  $S = \{sex\}$ , and non-sensitive features  $C = \{age\}$ ,  $R = \{\text{credit amount, repayment history}\}$  and  $H = \{\text{checking account, savings, housing}\}$ . Then, we aim to predict the binary feature  $Y = \{\text{credit risk}\}$  from  $\mathbf{X}$ . See Appendix F for further details.

Counterfactual fairness. Let  $S \subset \mathbf{X}$  be a sensitive attribute (e.g., gender), then a classifier  $h: \mathbf{X} \to Y$  is considered  $\epsilon$ -counterfactually fair [24] if:

$$
\left| P (h (\mathbf {x} ^ {C F}) = y \mid d o (S = \alpha), \mathbf {x} ^ {F}) - P (h (\mathbf {x} ^ {C F}) = y \mid d o (S = \alpha^ {\prime}), \mathbf {x} ^ {F}) \right| \leq \epsilon , \quad \forall \mathbf {x} ^ {C F}, \alpha^ {\prime} \neq \alpha , y.
$$

A classifier is counterfactually fair  $(\epsilon = 0)$ , if, given a factual  $\mathbf{x}^F$  with sensitive attribute  $S = \alpha$ , had its sensitive attribute been different  $S = \alpha'$ , the classifier prediction would remain the same. As VCAUSE allows us to generate counterfactual samples, we can thus use it to audit the fairness level of a classifier. Moreover, we can use the VCAUSE encoder to learn a fair classifier  $h_{\mathrm{VCAUSE}}: \mathbf{Z} \backslash Z_S \to Y$ , which takes as input the latent variables generated by VCAUSE without the one of the sensitive attribute  $Z_S$ . Following [24], we compare our VCAUSE fair classifier  $h_{\mathrm{VCAUSE}}$  with: i) a full model  $h_{\mathrm{full}}: \mathbf{X} \to Y$  that takes as input the complete variable set; ii) an unaware model  $h_{\mathrm{unaw}}: \mathbf{X} \backslash S \to Y$  that takes as input all variables but the sensitive one; iii) and a fair model  $h_{\mathrm{fair}}: \{X_i | S \notin \text{an}^*(i)\} \to Y$  that takes as input all non-descendant variables of the sensitive attribute.

Results. The results for logistic regression (LR) and support vector machine (SVM) classifiers are summarized in Table 3. Note that VCAUSE correctly ranks the different methods based on their unfairness level, showing that the full classifier is consistently less fair than the unaware and the fair classifiers, respectively. Moreover, the VCAUSE classifier leads to a fair classifier, while keeping the f1-score comparable to the unfair classifier. Therefore, VCAUSE does not only allow us to audit counterfactual fairness but also provides a practical approach to train accurate and fair classifiers.

![](images/5b81e59443d3003304449888bcbb533b488102ec8f214d44782a99f95f0689ce.jpg)  
Figure 5: Causal graph for variables  $\mathbf{X}$  of the German Credit dataset [6].

Table 3: Evaluation of counterfactual (un)fairness. All metrics are shown in %. Lower/Larger values of unfairness/f1-score are better.  

<table><tr><td>Metric</td><td>Classifier</td><td>full</td><td>unaware</td><td>fair</td><td>VCAUSE</td></tr><tr><td rowspan="2">↑ f1-score (%)</td><td>LR</td><td>71.07</td><td>68.33</td><td>50.00</td><td>74.81</td></tr><tr><td>SVM</td><td>74.60</td><td>72.44</td><td>64.71</td><td>70.40</td></tr><tr><td rowspan="2">↓ unfairness (%)</td><td>LR</td><td>5.93</td><td>2.25</td><td>0.16</td><td>0.85</td></tr><tr><td>SVM</td><td>6.07</td><td>2.68</td><td>0.20</td><td>1.00</td></tr></table>

# 7 Conclusion

In this work, we have proposed VCAUSE a variational causal autoencoder based on GNNs that: i) is specially designed to capture the properties of SCMs; ii) inherently handles heterogeneous causal graphs and data; and iii) provides accurate estimates of interventional and counterfactual distributions for SCMs of different complexities. As demonstrated by extensive experiments, VCAUSE provides accurate results for a wide variety of interventions in diverse SCMs leading to significantly more robust results than competing methods [17, 18]. Finally, we have shown a practical use-case of VCAUSE in a problem of increasing interest for the machine learning community, namely, fairness in classification. In particular, we have shown how to use VCAUSE to both assess counterfactual fairness and to train counterfactually fair classifiers.

Moreover, our work opens up many interesting venues for future work. First, as we have assumed a known causal graph and the absence of hidden confounders, it would be important to evaluate the sensitivity of VCAUSE to the violation of these assumptions in order to avoid its misuse. We also plan to extend VCAUSE to handle hidden confounders and to perform efficient causal discovery. Second, it would be interesting to perform ablation studies on the limitations of available GNNs architectures [55] for the VCAUSE encoder and decoder; as well as on how the performance of GNNs deteriorates as we increase the length of the causal path and thus the required number of hidden layers [28]. Finally, it would be interesting to apply VCAUSE to other causal questions recently discussed in the machine learning literature, such as privacy-preserving causal inference [26] or explainable machine learning [17].

# References

[1] Ahmed M Alaa and Mihaela van der Schaar. 2017. Bayesian inference of individualized treatment effects using multi-task Gaussian processes. arXiv preprint arXiv:1704.02801 (2017).  
[2] Jasmijn Bastings, Ivan Titov, Wilker Aziz, Diego Marcheggiani, and Khalil Sima'an. 2017. Graph convolutional encoders for syntax-aware neural machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), Vol. 3.  
[3] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. 2016. Importance weighted autoencoders. In Proceedings of the International Conference on Learning Representations (ICLR), Vol. 4.  
[4] Daniel C Castro, Ian Walwer, and Ben Glocker. 2020. Causality matters in medical imaging. Nature Communications 11 (2020).  
[5] Denis Charles, Max Chickering, and Patrice Simard. 2013. Counterfactual reasoning and learning systems: The example of computational advertising. Journal of Machine Learning Research 14 (2013).  
[6] Silvia Chiappa. 2019. Path-specific counterfactual fairness. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33.  
[7] Alexander Chudik, Kamiar Mohaddes, M Hashem Pesaran, Mehdi Raissi, and Alessandro Rebucci. 2020. Economic consequences of Covid-19: A counterfactual multi-country analysis. voxeu.org (2020).  
[8] Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, and Petar Velicković. 2020. Principal neighbourhood aggregation for graph nets. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 33.  
[9] Saloni Dash, Vineeth N Balasubramanian, and Amit Sharma. 2020. Evaluating and mitigating bias in image classifiers: A causal perspective using counterfactuals. arXiv preprint arXiv:2009.08270 (2020).  
[10] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)  
[11] Sergio Garrido, Stanislav S Borysov, Jeppe Rich, and Francisco C Pereira. 2020. Estimating causal effects with the neural autoregressive density estimator. arXiv preprint arXiv:2008.07283 (2020).  
[12] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017. Neural message passing for quantum chemistry. In Proceedings of the International Conference on Machine Learning (ICML), Vol. 34. PMLR.  
[13] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander Smola. 2012. A kernel two-sample test. The Journal of Machine Learning Research (JMLR) 13 (2012).  
[14] William L Hamilton, Rex Ying, and Jure Leskovec. 2017. Representation learning on graphs: Methods and applications. Bulletin of the IEEE Computer Society Technical Committee on Data Engineering (2017).  
[15] Patrik Hoyer, Dominik Janzing, Joris M Mooij, Jonas Peters, and Bernhard Scholkopf. 2008. Nonlinear causal discovery with additive noise models. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 21.  
[16] Maximilian Ilse, Patrick Forre, Max Welling, and Joris M Mooij. 2021. Efficient causal inference from combined observational and interventional data through causal reductions. arXiv preprint arXiv:2103.04786 (2021).  
[17] Amir-Hossein Karimi, Julius von Kugelgen, Bernhard Scholkopf, and Isabel Valera. 2020. Algorithmic recourse under imperfect causal knowledge: A probabilistic approach. arXiv preprint arXiv:2006.06831 (2020).

[18] Ilyes Khemakhem, Ricardo Monti, Robert Leech, and Aapo Hyvarinen. 2021. Causal autoregressive flows. In Proceedings of International Conference on Artificial Intelligence and Statistics (AISTATS), Vol. 24. PMLR.  
[19] Hyemi Kim, Seungjae Shin, JoonHo Jang, Kyungwoo Song, Weonyoung Joo, Wanmo Kang, and Il-Chul Moon. 2021. Counterfactual fairness with disentangled causal effect variational autoencoder. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35.  
[20] Diederik P Kingma and Max Welling. 2014. Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR), Vol. 2.  
[21] Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 (2016).  
[22] Murat Kocaoglu, Christopher Snyder, Alexandros G. Dimakis, and Sriram Vishwanath. [n.d.]. CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training. In Proceedings of the International Conference on Learning Representations (ICLR), year=2018.  
[23] Noemi Kreif and Karla DiazOrdaz. 2019. Machine learning in policy evaluation: New tools for causal inference. arXiv preprint arXiv:1903.00402 (2019).  
[24] Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. 2017. Counterfactual fairness. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.  
[25] Matt J Kusner, Chris Russell, Joshua R Loftus, and Ricardo Silva. 2018. Causal interventions for fairness. arXiv preprint arXiv:1806.02380 (2018).  
[26] Matt J Kusner, Yu Sun, Karthik Sridharan, and Kilian Q Weinberger. 2016. Private causal inference. In Proceedings of the Conference on Artificial Intelligence and Statistics (AISTATS), Vol. 19. PMLR.  
[27] Felix Leeb, Yashas Annadani, Stefan Bauer, and Bernhard Scholkopf. 2020. Structured representation learning using Structural autoencoders and hybridization. arXiv preprint arXiv:2006.07796 (2020).  
[28] Qimai Li, Zhichao Han, and Xiao-Ming Wu. 2018. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32.  
[29] Christos Louizos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling. 2017. Causal effect inference with deep latent-variable models. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 30.  
[30] Ricardo Pio Monti, Kun Zhang, and Aapo Hyvarinen. 2020. Causal discovery with general non-linear relationships using non-linear ICA. In Proceedings of the Uncertainty in Artificial Intelligence (UAI), Vol. 36. PMLR.  
[31] Raha Moraffah, Bahman Moraffah, Mansoresh Karami, Adrienne Raglin, and Huan Liu. 2020. CAN: A causal adversarial network for learning observational and interventional distributions. arXiv preprint arXiv:2008.11376 (2020).  
[32] Krikamol Muandet, Motonobu Kanagawa, Sorawit Saengkyongam, and Sanparith Marukatat. 2018. Counterfactual mean embeddings. arXiv preprint arXiv:1805.08845 (2018).  
[33] Ignavier Ng, Shengyu Zhu, Zhitang Chen, and Zhuangyan Fang. 2019. A graph autoencoder approach to causal structure learning. arXiv preprint arXiv:1911.07420 (2019).  
[34] Sebastian Nowozin. 2018. Debiasing evidence approximations: On importance-weighted autoencoders and jackknife variational inference. In Proceedings of the International Conference on Learning Representations (ICML), Vol. 35. PMLR.  
[35] Álvaro Parafita and Jordi Vitrià. 2019. Explaining visual models by causal attribution. arXiv preprint arXiv:1909.08891 (2019).

[36] Álvaro Parafita and Jordi Vitrià. 2020. Causal inference with deep causal graphs. arXiv preprint arXiv:2006.08380 (2020).  
[37] Álvaro Parafita and Jordi Vitria. 2019. Explaining visual models by causal attribution. In International Conference on Computer Vision Workshop (ICCVW).  
[38] Nick Pawlowski, Daniel Coelho de Castro, and Ben Glocker. 2020. Deep structural causal models for tractable counterfactual inference. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 33.  
[39] Judea Pearl. 2009. Causal inference in statistics: An overview. Statistics surveys 3 (2009).  
[40] Jonas Peters, Peter Buhlmann, and Nicolai Meinshausen. 2016. Causal inference by using invariant prediction: identification and confidence intervals. Journal of the Royal Statistical Society, Series B (Statistical Methodology) (2016).  
[41] Tom Rainforth, Adam Kosiorek, Tuan Anh Le, Chris Maddison, Maximilian Igl, Frank Wood, and Yee Whye Teh. 2018. Tighter variational bounds are not necessarily better. In Proceedings of the International Conference on Machine Learning (ICML), Vol. 35. PMLR.  
[42] Vineeth Rakesh, Ruocheng Guo, Raha Moraffah, Nitin Agarwal, and Huan Liu. 2018. Linked causal variational autoencoder for inferring paired spillover effects. In Proceedings of the International Conference on Information and Knowledge Management (CIKM). ACM.  
[43] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. 2008. The graph neural network model. IEEE transactions on neural networks 20 (2008).  
[44] Bernhard Schölkopf. 2019. Causality for machine learning. arXiv preprint arXiv:1911.10500 (2019).  
[45] Patrick Schwab, Lorenz Linhardt, and Walter Karlen. 2018. Perfect match: A simple method for learning representations for counterfactual inference with neural networks. arXiv preprint arXiv:1810.00656 (2018).  
[46] Uri Shalit, Fredrik Johansson, and David Sontag. 2016. Bounding and minimizing counterfactual error. arXiv preprint arXiv:1606.03976 (2016).  
[47] Uri Shalit, Fredrik D Johansson, and David Sontag. 2017. Estimating individual treatment effect: Generalization bounds and algorithms. In Proceedings of the International Conference on Machine Learning (ICML), Vol. 34. PMLR.  
[48] Xinwei Shen, Furui Liu, Hanze Dong, Qing Lian, Zhitang Chen, and Tong Zhang. 2020. Disentangled generative causal representation learning. arXiv preprint arXiv:2010.02637 (2020).  
[49] Shohei Shimizu, Patrik O Hoyer, Aapo Hyvarinen, Antti Kerminen, and Michael Jordan. 2006. A linear non-Gaussian acyclic model for causal discovery. Journal of Machine Learning Research 7 (2006).  
[50] Ilya Shpitser, Thomas S. Richardson, and James M. Robins. 2011. An efficient algorithm for computing interventional distributions in latent variable causal models. In Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI), Vol. 27.  
[51] Bob Siegerink, Wouter den Hollander, Maurice Zeegers, and Rutger Middelburg. 2016. Causal Inference in law: An epidemiological perspective. European Journal of Risk Regulation 7, 1 (2016).  
[52] George Tucker, Dieterich Lawson, Shixiang Gu, and Chris J Maddison. 2018. Doubly reparameterized gradient estimators for monte carlo objectives. arXiv preprint arXiv:1810.04152 (2018).  
[53] Matthew James Vowels, Necati Cihan Camgoz, and Richard Bowden. 2020. Targeted VAE: Structured inference and targeted learning for causal parameter estimation. arXiv preprint arXiv:2009.13472 (2020).

[54] Antoine Wehenkel and Gilles Louppe. 2021. Graphical normalizing flows. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS), Vol. 24. PMLR.  
[55] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. 2020. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems (2020).  
[56] Mengyue Yang, Furui Liu, Zhitang Chen, Xinwei Shen, Jianye Hao, and Jun Wang. 2020. CausalVAE: Disentangled representation learning via neural structural causal models. arXiv preprint arXiv:2004.08697 (2020).  
[57] Bing Yu, Haoteng Yin, and Zhanxing Zhu. 2018. Spatio-temporal graph convolutional networks: a deep learning framework for traffic forecasting. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), Vol. 27.  
[58] Yue Yu, Jie Chen, Tian Gao, and Mo Yu. 2019. DAG-GNN: DAG structure learning with graph neural networks. In Proceedings of the International Conference on Machine Learning (ICML), Vol. 36. PMLR.  
[59] Matej Zečević, Devendra Singh Dhami, Athresh Karanam, Sriraam Natarajan, and Kristian Kersting. 2021. Interventional sum-product networks: Causal inference with tractable probabilistic models. arXiv preprint arXiv:2102.10440 (2021).  
[60] Kun Zhang, Biwei Huang, Jiji Zhang, Clark Glymour, and Bernhard Scholkopf. 2017. Causal discovery from nonstationary/heterogeneous data: skeleton estimation and orientation determination. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), Vol. 26.  
[61] Muhan Zhang, Shali Jiang, Zhicheng Cui, Roman Garnett, and Yixin Chen. 2019. D-VAE: A variational autoencoder for directed acyclic graphs. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 32.  
[62] Min Zheng and Samantha Kleinberg. 2019. Using domain knowledge to overcome latent variables in causal inference from time series. In Proceedings of the Machine Learning for Healthcare Conference (MLHC). PMLR.  
[63] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. 2018. DAGs with NO TEARS: Continuous optimization for structure learning. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 31.
