# SINGLE-SHOT GENERAL HYPER-PARAMETER OPTIMIZATION FOR FEDERATED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

We address the problem of hyper-parameter optimization (HPO) for federated learning (FL-HPO). We introduce Federated Loss SuRface Aggregation (FLoRA), a general FL-HPO solution framework that can address use cases of tabular data and any Machine Learning (ML) model including gradient boosting training algorithms, SVMs, neural networks, among others and thereby further expands the scope of FL-HPO. FLoRA enables single-shot FL-HPO: identifying a single set of good hyper-parameters that are subsequently used in a single FL training. Thus, it enables FL-HPO solutions with minimal additional communication overhead compared to FL training without HPO. Utilizing standard smoothness assumptions, we theoretically characterize the optimality gap of FLoRA for any convex and non-convex loss functions, which explicitly accounts for the heterogeneous nature of the parties' local data distributions, a dominant characteristic of FL systems. Our empirical evaluation of FLoRA for multiple FL algorithms on seven OpenML datasets demonstrates significant model accuracy improvements over the baselines, and robustness to increasing number of parties involved in FL-HPO training.

# 1 INTRODUCTION

Traditional machine learning (ML) approaches require training data to be gathered at a central location where the learning algorithm runs. In real world scenarios, however, training data is often subject to privacy or regulatory constraints restricting the way data can be shared, used and transmitted. Examples of such regulations include the European General Data Protection Regulation (GDPR), California Consumer Privacy Act (CCPA), Cybersecurity Law of China (CLA) and HIPAA, among others. Federated learning (FL), first proposed in McMahan et al. (2017b), has recently become a popular approach to address privacy concerns by allowing collaborative training of ML models among multiple parties where each party can keep its data private.

FL-HPO problem. Despite the privacy protection FL brings along, there are many open problems in FL domain, one of which is hyper-parameter optimization for FL or FL-HPO (Kairouz et al., 2019; Khodak et al., 2021). Existing FL systems require a user (or all participating parties) to pre-set (agree on) multiple hyper-parameters (HPs) (i) for the model being trained (such as number of layers for neural networks or tree depth and number of trees in tree ensembles), (ii) for the FL algorithms, and (iii) for aggregation (if such hyper-parameters exist). Hyper-parameter optimization (HPO) for FL is important because the choice of HPs can have dramatic impact on model performance (McMahan et al., 2017b) much like in traditional centralized ML.

While HPO has been widely studied in the centralized ML setting (e.g., Hutter et al. (2019)), it comes with unique challenges in the FL setting. First, existing HPO techniques often make use of the entire dataset, which is not available centrally in FL. Secondly, they need to train many models for a large number of HP configurations which is prohibitively expensive in terms of communication and training time in FL settings; training a single model already has a high communication overhead (Kairouz et al., 2019). Thirdly, one important challenge that has not been adequately explored in FL-HPO literature is support for tabular data, which are widely used in enterprise settings, such as financial services and other traditional industries, preferring traditional models with some explainability (Ludwig et al., 2020). Although a few approaches have been recently proposed for FL-HPO, they focus on handling HPO using personalization techniques (Khodak et al., 2021) and neural networks (Khodak et al., 2020). To the best of our knowledge, there is no FL-HPO approach to train non-neural network

models, such as gradient boosted decision trees (Friedman, 2001) (e.g., XGBoost (Chen & Guestrin, 2016)) that are particularly common in the enterprise setting, even though there are existing FL algorithms for such models (Li et al., 2020; Ong et al., 2020). This leads to our motivating question:

Can we develop a FL-HPO scheme that performs HPO for any ML model in a FL environment without significantly increasing the already-high communication overhead of FL?

In this paper, we address the aforementioned challenges of FL-HPO and our motivating question. We focus on the problem where the model HPs are shared by all parties and we seek a set of HPs and train a single model that is used by all parties. Our motivating question leads to four further requirements that make the problem challenging: (C1) To perform FL-HPO with any ML model, we cannot make any assumption that two models with different HPs can perform some "weight-sharing", allowing our solution to be applied beyond fixed architecture neural networks. (C2) To be general across ML models, we do not assume the ability to perform "multi-fidelity"  $HPO^2$  to reduce the communication overhead of FL-HPO. (C3) To avoid increasing the FL communication overhead, we seek to perform "single-shot" FL-HPO, which allows us to perform FL-HPO while requiring only a single FL model training. (C4) To be applicable to FL with data heterogeneity, we cannot assume that parties have independent and identically distributed (IID) data.

Contributions. Given the above FL-HPO problem setting, we make the following contributions:

- (§2) We present a novel framework Federated Loss SuRface Aggregation (FLoRA) that leverages meta-learning techniques enabling asynchronous local HPOs on each party to perform single-shot HPO for the global FL-HPO problem.  
- (§2.3) We provide theoretical guarantees for the set of HPs selected by FLoRA covering both IID and Non-IID cases regardless of the convexity of the loss function. To the best of our knowledge, this is the first rigorous theoretical analysis for FL-HPO problem and also the first optimality gap constructed in terms of the estimated loss given a target distribution.  
- (§3) We evaluate FLoRA on the FL-HPO of Histogram based Gradient Boosted Decision Trees (HGB), Support Vector Machines (SVM) and Multi-layered Perceptrons (MLP) on seven classification datasets from OpenML (Vanschoren et al., 2013), highlighting (i) its performance relative to baselines, and (ii) the effect of data heterogeneity.

In Figure 1, we present a snapshot of our empirical results which highlights the communication overhead reduction we achieve from FLoRA while producing higher quality models. As baselines, we directly adopt an existing centralized HPO scheme that requires federated training of multiple models and term this a "multi-shot" FL-HPO baseline. The figure also shows a "single-shot" baseline that uses curated HPs (described in §3), and FLoRA is also single-shot. Figure 1 shows that the "multi-shot" approach requires a significantly large number of FL model trainings (39 for MLP and 24 for HGB) and hence more communication to find a HP that matches the performance of the HP found by FLoRA, highlighting the efficiency and effectiveness of FLoRA. This result demonstrates that FLoRA is a FL-HPO scheme that works with any ML model (HGB, MLP, etc), providing competitive performance without significantly increasing the communication overhead of FL by only requiring a single FL model training.

![](images/63266b2260a631441778f9f07db478536a6c06486fdbe5f1b6ce9010ec290a04.jpg)  
Figure 1: Communication overhead savings of FLoRA compared to "multi-shot" FL-HPO for the same level of performance. We are considering 2 FL-HPO problems on the Electricity dataset (with HGB and MLP). We use the relative regret (defined in §3) of each scheme as the performance metric (lower is better), where a regret of 1 denotes performance of the single-shot baseline while a regret of 0 implies optimal performance. FLoRA and the single-shot baseline require a single federated model training.

# 1.1 RELATED WORK

Performance optimization of FL systems. One of the main challenges in FL is achieving high accuracy with low communication overhead. FedAvg (McMahan et al., 2017a) is a predominant

FL algorithm and several optimization schemes build on it. Initially, communication optimizations included performing multiple stochastic gradient descent (SGD) local iterations at the clients and randomly selecting a small subset of the clients to compute and send updates to the server. Subsequently, compression techniques were used to minimize the size of model updates to the server. The accuracy and communication overhead of these techniques are sensitive to their HPs (McMahan et al., 2017a).

FL-HPO of neural networks. Recent optimization approaches adapt HPs such as the local learning rate at each client (Koskela & Honkela, 2019; Mostafa, 2019; Reddi et al., 2020), the number of local SGD epochs (Wang et al., 2019). Dai et al. (2020; 2021) address Federated Bayesian Optimization. Although using HPO with multiple HPs, the problem setup is quite different than FL: they focus on a single party using information from other parties to accelerate its own Bayesian Optimization, instead of building a model for all parties. Federated Network Architecture Search (FNAS) approaches search for architectural HPs of CNN models by running locally NAS algorithms and then aggregating the NAS architecture weights and model weights using FedAvg (He et al., 2020; Garg et al., 2020; Xu et al., 2020). These approaches have shown empirical gains but lack theoretical analysis. Inspired from the NAS technique of weight-sharing, Khodak et al. (2020; 2021) proposed FedEx, a FL-HPO framework to accelerate a centralized HPO procedure, i.e., successive halving algorithm (SHA), for many SGD-based FL algorithms. FedEx focuses on building personalized models for parties by tuning local HPs of the parties. They provide theoretical guarantee only for a special case of tuning a single HP, the local learning rate, in an online convex optimization setting. To optimize global HPs, FedEx requires multiple rounds of communication, and hence is not single-shot FL-HPO (Khodak et al., 2020; 2021). Note that these above techniques are multi-shot, leveraging both the idea of weight-sharing and multi-fidelity HPO for improving the communication efficiency of FL-HPO. The FedHPO-B benchmark (Zhen et al., 2022) primarily focuses on neural-networks and multi-fidelity FL-HPO and is useful to evaluate the above schemes across various problems.

Need for FL-HPO of tabular data models. As most existing FL-HPO approaches focus on SGD-based algorithms and neural networks, one major limitation they share is that they do not support tree-based models, such as gradient boosted trees (Friedman, 2001), a popular model for enterprise setting. These models provide explainability for predictions which is required for financial and healthcare FL use-cases. As laid out in a policy paper by the OECD, numerous regulations of member countries govern the use of analytics and data (OECD, 2021): GDPR, for example, requires decision-making models for financial services and insurance to be explainable, which is mostly achieved using traditional models such as decision tree variants (Goodman & Flaxman, 2017). Outside consumer finance, governance rules require explainability of portfolio and risk management for auditing purposes (Gensler & Bailey, 2020). Again, DNNs (deep neural networks) are not satisfactory from a current regulatory point of view and, thus, the financial services and insurance sectors rely on more explainable models (such as tree-based ones), also in federation.

This paper. Our framework improves on the above approaches in several ways, summarized also in Table 1. (1) It is more general, as it can tune multiple HPs and is applicable to non SGD-training settings such as gradient boosting trees. This is achieved by treating FL-HPO as a black-box HPO problem (as opposed to grey-box HPO where we can leverage techniques such as weight-sharing and multi-fidelity), which has been addressed in centralized HPO literature using grid search, random search (Bergstra & Bengio, 2012) and Bayesian Optimization approaches (Shahriari et al., 2016). The key challenge is the requirement to perform computationally intensive evaluations on a large number of HPO configurations, where each evaluation involves training a model and scoring it on a validation dataset. In the distributed FL setting this problem is exacerbated because validation sets are local to the parties and each FL training/scoring evaluation is communication intensive. Therefore a brute force application of centralized black-box HPO approaches that select HPs in an outer loop and proceed with FL training evaluations is not feasible. (2) It yields minimal HPO communication overhead. This is achieved by building a loss surface from local asynchronous HPO at the parties that yields a single optimized HP configuration used to train a global model with a single FL training. (3) It is the first that theoretically characterizes the optimality gap in an FL-HPO setting, for the case we focus in this paper: creating a global model by tuning multiple global HPs without accessing global validation dataset (as opposed to existing work either optimizing wrt parties' validation datasets or assuming access to a global validation dataset during HPO).

Table 1: Positioning of our proposed framework FLoRA against existing literature. SS: single-shot. MF: multi-fidelity. WS: weight-sharing.  $\theta_G$ : global model HPs.  $\phi$ : aggregator HPs. See §2.†: FedHPO-B is a benchmarking suite and not an algorithm, and the properties correspond to the problems in the suite.  

<table><tr><td>Method</td><td>Any ML model</td><td>SS</td><td>No MF req.</td><td>No WS req.</td><td>Handles θG</td><td>Handles φ</td><td>Black-box</td></tr><tr><td>Black-box HPO</td><td>✓</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>NA</td><td>✓</td></tr><tr><td>Grey-box HPO</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>NA</td><td>X</td></tr><tr><td>FNAS</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>NA</td><td>X</td></tr><tr><td>FedEx</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>FLoRA (Ours)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>✓</td></tr><tr><td>FedHPO-B†</td><td>X</td><td>X</td><td>X</td><td>-</td><td>✓</td><td>X</td><td>X</td></tr></table>

# 2 METHODOLOGY

In the centralized ML setting, we would consider a model class  $\mathcal{M}$  and its corresponding learning algorithm  $\mathcal{A}$  parameterized collectively with HPs  $\theta \in \Theta$ , and given a training set  $D$ , we can learn a single model  $\mathcal{A}(\mathcal{M}, \theta, D) \to m \in \mathcal{M}$ . Given some predictive loss  $\mathcal{L}(m, D')$  of any model  $m$  scored on some holdout set  $D'$ , the centralized HPO problem can be stated as

$$
\min  _ {\boldsymbol {\theta} \in \Theta} \mathcal {L} (\mathcal {A} (\mathcal {M}, \boldsymbol {\theta}, D), D ^ {\prime}). \tag {2.1}
$$

In the most general FL setting, we have  $p$  parties  $P_{1},\ldots ,P_{p}$  each with their private local training dataset  $D_{i},i\in [p] = \{1,2,\dots ,p\}$ . Let  $D = \cup_{i = 1}^{p}D_{i}$  denote the aggregated training dataset and  $\overline{D} = \{D_i\}_{i\in [p]}$  denote the set of per-party datasets. Each model class (and corresponding learning algorithm) is parameterized by global HPs  $\theta_G\in \Theta_G$  shared by all parties and per-party local HPs  $\theta_L^{(i)}\in \Theta_L,i\in [p]$  with  $\Theta = \Theta_{G}\times \Theta_{L}$ . FL systems usually include an aggregator which introduces an additional set of HPs  $\phi \in \Phi$ . Finally, we would have a FL algorithm

$$
\mathcal {F} \left(\mathcal {M}, \phi , \boldsymbol {\theta} _ {G}, \left\{\boldsymbol {\theta} _ {L} ^ {(i)} \right\} _ {i \in [ p ]}, \mathcal {A}, \bar {D}\right)\rightarrow m \in \mathcal {M}, \tag {2.2}
$$

which takes as input all the relevant HPs and per-party datasets and generates a model. In this case, the FL-HPO problem can be stated in the two following ways depending on the desired goals: (i) Ideally, for a global holdout dataset  $D'$  (a.k.a validation set, possibly from the same distribution as the aggregated dataset  $D$ ), the target problem is:

$$
\min  _ {\boldsymbol {\phi} \in \boldsymbol {\Phi}, \boldsymbol {\theta} _ {G} \in \boldsymbol {\Theta} _ {G}, \boldsymbol {\theta} _ {L} ^ {(i)} \in \boldsymbol {\Theta} _ {L}, i \in [ p ]} \mathcal {L} \left(\mathcal {F} \left(\mathcal {M}, \boldsymbol {\phi}, \boldsymbol {\theta} _ {G}, \left\{\boldsymbol {\theta} _ {L} ^ {(i)} \right\} _ {i \in [ p ]}, \mathcal {A}, \bar {D}\right), D ^ {\prime}\right). \tag {2.3}
$$

(ii) An alternative target problem would involve per-party holdout datasets  $D_{i}^{\prime}, i \in [p]$  as follows:

$$
\min  _ {\boldsymbol {\phi} \in \boldsymbol {\Phi}, \boldsymbol {\theta} _ {G} \in \boldsymbol {\Theta} _ {G}, \boldsymbol {\theta} _ {L} ^ {(i)} \in \boldsymbol {\Theta} _ {L}, i \in [ p ]} \operatorname {A g g} \left(\left\{\mathcal {L} \left(\mathcal {F} \left(\mathcal {M}, \boldsymbol {\phi}, \boldsymbol {\theta} _ {G}, \left\{\boldsymbol {\theta} _ {L} ^ {(i)} \right\} _ {i \in [ p ]}, \mathcal {A}, \bar {D}\right), D _ {i} ^ {\prime}\right), i \in [ p ] \right\}\right), \tag {2.4}
$$

where  $\mathrm{Agg}:\mathbb{R}^p\to \mathbb{R}$  is some aggregation function (such as average or maximum) that scalarizes the  $p$  per-party predictive losses.

Contrasting problem (2.1) to problems (2.3) & (2.4), we can see that the FL-HPO is significantly more complicated than the centralized HPO problem. In the ensuing presentation, we focus on problem (2.3) although our proposed single-shot FL-HPO scheme can be applied and evaluated for problem (2.4). We simplify the FL-HPO problem in the following ways: (i) we assume that there is no personalization so there are no per-party local HPs  $\pmb{\theta}_{L}^{(i)}, i \in [p]$ , (ii) we only focus on the model class HPs  $\pmb{\theta}_{G}$ , deferring HPO for aggregator HPs  $\phi$  for future work as many of them are set based on the communication and computational resources available in the FL system and cannot be directly optimized with regards to some predictive performance metrics, and (iii) we assume there is a global holdout/Validation set  $D'$  which is only used to evaluate the final global model's performance but can not be accessed during HPO process. And parties can only access their own private training  $D_{i}$  and validation  $D_{i}'$  sets. Hence the problem we will study is stated as for a fixed aggregator HP  $\phi$ :

$$
\min  _ {\boldsymbol {\theta} _ {G} \in \boldsymbol {\Theta} _ {G}} \mathcal {L} \left(\mathcal {F} \left(\mathcal {M}, \phi , \boldsymbol {\theta} _ {G}, \mathcal {A}, \bar {D}\right), D ^ {\prime}\right). \tag {2.5}
$$

This problem appears similar to the centralized HPO problem (2.1). However, note that the main challenges in (2.5) is (i) the need for a federated training for each set of HPs  $\theta_{G}$ , and (ii) the need to

evaluate the trained model on the global validation set  $D'$  (which is usually not available in usual FLHPO setting). Hence it is not practical (from a communication overhead and functional perspective) to apply existing off-the-shelf HPO schemes to problem (2.5). In the subsequent discussion, for simplicity purposes, we will use  $\theta$  to denote the global HPs, dropping the "G" subscript.

# 2.1 LEVERAGING LOCAL HPOS

While it is possible but extremely expensive to apply off-the-shelf HPO solvers (such as Bayesian Optimization (BO) (Shahriari et al., 2016), Hyperopt (Bergstra et al., 2011), etc.), we wish to understand how we can leverage local and asynchronous HPOs in each of the parties. We begin with a simple but intuitive hypothesis underlying various meta-learning schemes for HPO (Vanschoeren, 2018; Wistuba et al., 2018): if a HP configuration  $\pmb{\theta}$  has good performance for all parties independently, then  $\pmb{\theta}$  is a strong candidate for federated training.

With this hypothesis, we present our proposed algorithm FLoRA in Algorithm 1. In this scheme, we allow each party to perform HPO locally and asynchronously with some adaptive HPO scheme such

# Algorithm 1 FL-HPO with (FLoRA)

1: Input:  $\Theta$ ,  $\mathcal{M}$ ,  $\mathcal{A}$ ,  $\mathcal{F}$ ,  $\{(D_i, D_i')\}_{i \in [p]}, T$  
2: for each party  $P_{i}, i \in [p]$  do  
3: Run HPO to generate  $T$  (HP, loss) pairs

$$
E ^ {(i)} = \left\{\left(\boldsymbol {\theta} _ {t} ^ {(i)}, \mathcal {L} _ {t} ^ {(i)}\right), t \in [ T ] \right\}, \tag {2.6}
$$

$$
\boldsymbol {\theta} _ {t} ^ {(i)} \in \Theta , \mathcal {L} _ {t} ^ {(i)} := \mathcal {L} (\mathcal {A} (\mathcal {M}, \boldsymbol {\theta} _ {t} ^ {(i)}, D _ {i}), D _ {i} ^ {\prime}).
$$

4: end for

5: Collect all  $E = \{E^{(i)}, i \in [p]\}$  in aggregator  
6: Generate a unified loss surface  $\widehat{\ell}:\Theta \to \mathbb{R}$  using  $E$  
7: Select best HP candidate

$$
\widehat {\boldsymbol {\theta}} ^ {*} \leftarrow \arg \min  _ {\boldsymbol {\theta} \in \mathfrak {S}} \widehat {\ell} (\boldsymbol {\theta}). \tag {2.7}
$$

8:Invoke federated training  $m\gets \mathcal{F}(\mathcal{M},\widehat{\pmb{\theta}}^{\star},\mathcal{A},\overline{D})$  
9: Output: FL model  $m$ .

as BO (line 3). Then, at each party  $i \in [p]$ , we collect all the attempted  $T$  HPs  $\boldsymbol{\theta}_t^{(i)}, t \in [T] = \{1,2,\dots,T\}$  and their corresponding predictive loss  $\mathcal{L}_t^{(i)}$  into a set  $E^{(i)}$  (line 3, equation (2.6)). Then these per-party sets of (HP, loss) pairs  $E^{(i)}$  are collected at the aggregator (line 5). This operation has at most  $O(pT)$  communication overhead (note that the number of HPs are usually much smaller than the number of columns or number of rows in the per-party datasets). These sets are then used to generate an aggregated loss surface  $\widehat{\ell} : \Theta \to \mathbb{R}$  (line 6) which will then be used to make the final single-shot HP recommendation  $\widehat{\boldsymbol{\theta}}^{\star} \in \Theta$  (line 7) for the federated training to create the final model  $m \in \mathcal{M}$  (line 8). We will discuss the generation of the aggregated loss surface in detail in §2.2. Before that, we briefly want to discuss the motivation behind some of our choices in Algorithm 1.

Remarks. Using adaptive HPO schemes instead of non-adaptive schemes such as random search or grid search allows us to efficiently approximate the local loss surface more accurately (and with more certainty) in regions of the HP space where the local performance is favorable instead of trying to approximate the loss surface well over the complete HP space. This has advantages both in terms of computational efficiency and loss surface approximation. Moreover, each party executes HPO asynchronously, without coordination with HPO results from other parties or with the aggregator. This is in line with our objective to minimize communication overhead. Although there could be strategies that involve coordination between parties, they could involve many rounds of communication. Our experimental results show that this approach is effective for the datasets we evaluated for.

# 2.2 LOSS SURFACE AGGREGATION

Given the sets of (HP, loss) pairs  $E^{(i)} = (\theta_t^{(i)},\mathcal{L}_t^{(i)})$ ,  $i\in [p]$ ,  $t\in [T]$  at the aggregator, we wish to construct a loss surface  $\widehat{\ell}:\Theta \to \mathbb{R}$  that best emulates the (relative) performance loss  $\widehat{\ell}(\boldsymbol{\theta})$  we would observe when training the model on  $\overline{D}$ . Based on our hypothesis, we want the loss surface to be such that it would have a relatively low  $\widehat{\ell}(\boldsymbol{\theta})$  if  $\boldsymbol{\theta}$  has a low loss for all parties simultaneously. However, because of the asynchronous and adaptive nature of the local HPOs, for any HP  $\boldsymbol{\theta}\in \Theta$ , we would not have the corresponding losses from all the parties. For that reason, we will model the loss surfaces using regressors that try to map any HP to their corresponding loss. We present four ways of constructing such loss surfaces, and we also briefly summarize them in Table 2.

The most straightforward way to construct such a loss surface is to merge all the per-party sets  $E^{(i)}$  to get a single set  $E = \cup_{i\in [p]}E^{(i)}$  and use it to train a regressor  $f:\Theta \to \mathbb{R}$  (such as a Random Forest Regressor (Breiman, 2001)) using the HPs  $\theta$  as the covariates and the corresponding loss as the dependent variable. Then we can define the loss surface as this single global model or SGM

$\hat{\ell}(\pmb{\theta}) \coloneqq f(\pmb{\theta})$ . However, this loss surface is extremely optimistic, assigning a low loss to a HP if it had a low loss estimate on any one of the parties, making it unsuitable in the presence of data heterogeneity. We can leverage uncertainty quantification  $u: \Theta \to \mathbb{R}_+$  around regressor predictions to get a loss surface  $\hat{\ell}(\pmb{\theta}) \coloneqq f(\pmb{\theta}) + \alpha u(\pmb{\theta})$  for some  $\alpha > 0$  - the single global model with uncertainty or SGM+U. This would improve the robustness of SGM by penalizing parts of the HP space which were not well explored by all parties' local HPOs.

Instead of merging the per-party  $E^{(i)}$ , we can also use them to train a per-party local regressor model  $f^{(i)}: \Theta \to \mathbb{R}$  and use their ensemble as the loss surface. One way is to use the average of the per-party local models or APLM as the loss surface  $\hat{\ell}(\pmb{\theta}) := 1/p \sum_{i \in [p]} f^{(i)}(\pmb{\theta})$ . This is less optimistic than SGM and provides some level of robustness in the presence of non-IID heterogeneous per-party distributions since it will assign a low loss for a HP only if its average across all per-party regressors is low, which implies that most parties observed a relatively low loss around this HP. An even more robust loss surface would be the maximum of the per-party local models or MPLM  $\hat{\ell}(\pmb{\theta}) := \max_{i \in [p]} f^{(i)}(\pmb{\theta})$  which would only assign a

Table 2: Loss surfaces:  $f:\Theta \to \mathbb{R}$  is the global loss surface generated using the aggregated set  $\cup_{i\in [p]}E^{(i)}$  of the per-party set of loss pairs  $E^{(i)}$  from each party  $i\in [p]$ .  $u:\Theta \to \mathbb{R}_+$  is an uncertainty model generated using the aggregated set  $\cup_{i\in [p]}E^{(i)}$  and  $\alpha >0$  is a constant.  $f_{i}:\Theta \to \mathbb{R}$  for any  $i\in [p]$  is the per-party loss surface generated using the party's loss pairs  $E^{(i)}$ .

<table><tr><td>Surface</td><td>ˆ(θ) :=</td><td>Optimism</td><td>Non-IID</td></tr><tr><td>SGM</td><td>f(θ)</td><td>High</td><td>X</td></tr><tr><td>SGM+U</td><td>f(θ) + α · u(θ)</td><td>Medium</td><td>Partial</td></tr><tr><td>MPLM</td><td>maxi∈[p] fi(θ)</td><td>Low</td><td>✓</td></tr><tr><td>APLM</td><td>1/p ∑i∈[p] fi(θ)</td><td>Medium</td><td>✓</td></tr></table>

low loss to a HP only if it has low loss estimate across all parties, making it extremely capable of handling data heterogeneity (as we will also highlight in our empirical evaluations). We discuss these loss surfaces in further details in Appendix B. In §2.3, we theoretically quantify the performance guarantees for MPLM and APLM, and in §3, we empirically evaluate all these loss surfaces.

# 2.3 OPTIMALITY ANALYSIS

We now rigorously analyze the sub-optimality of the HP selected by FLORA. We are interested in providing a bound for the following optimality gap:

$$
\mathcal {G} := \tilde {\ell} (\widehat {\boldsymbol {\theta}} ^ {\star}, \mathcal {D}) - \tilde {\ell} (\boldsymbol {\theta} ^ {\star}, \mathcal {D}), \text {w h e r e} \boldsymbol {\theta} ^ {\star} \in \arg \min  _ {\boldsymbol {\theta} \in \Theta} \tilde {\ell} (\boldsymbol {\theta}, \mathcal {D}). \tag {2.8}
$$

Here,  $\tilde{\ell} (\pmb {\theta},\mathcal{D})$  be an estimate of the true loss  $\ell (\pmb {\theta},\mathcal{D}):= \mathbb{E}_{(x,y)\sim \mathcal{D}}\mathcal{L}(\mathcal{A}(\pmb {\theta},\overline{D}),(x,y))$  (see Definition C.1) given some validation (holdout) set  $D^{\prime}$  sampled from  $\mathcal{D}$ , which is the model performance metric during evaluation and/or inference time. Recall that  $\widehat{\pmb{\theta}}^{\star}$  selected by FLoRA is defined as in (2.7), and  $\theta^{*}$  denotes the optimal HP given by  $\tilde{\ell}$  for a desired data distribution  $\mathcal{D}$  we want to learn.

We present our main results in Theorem 2.1. Informally speaking we show how to bound the optimality gap by picking the 'worst-case' HP setting that maximizes the combination of Wasserstein distances of the local data distributions and actual quality of local HPO approximation across parties. The more precise theorem statement and its proof with formal discussion of technical definitions and assumptions can be found in Appendix C.

Theorem 2.1. Suppose that the loss estimate  $\tilde{\ell}$  and the unified loss surface  $\hat{\ell}$  are Lipschitz continuous. Consider the optimality gap  $\mathcal{G}$  defined in (2.8), where  $\widehat{\pmb{\theta}}^{\star}$  is selected by FLoRA with each party  $i\in [p]$  collecting  $T$  (HP, loss) pairs  $\{(\pmb {\theta}_t^{(i)},\mathcal{L}_t^{(i)})\}_{t\in [T]}$  during the local HPO run. For a desired data distribution  $\mathcal{D} = \sum_{i = 1}^{p}w_{i}\mathcal{D}_{i}$ , where  $\{\mathcal{D}_i\}_{i\in [p]}$  are the sets of parties' local data distributions and  $w_{i}\in [0,1],\forall i\in [p]$ , we have

$$
\mathcal {G} \leq \max  _ {\boldsymbol {\theta} \in \boldsymbol {\Theta}} \sum_ {i \in [ p ]} C _ {\boldsymbol {\alpha}} \left\{C _ {\beta} \sum_ {j \in [ p ], j \neq i} w _ {j} \mathcal {W} _ {1} (\mathcal {D} _ {j}, \mathcal {D} _ {i}) + C _ {\tilde {L}, \hat {L} _ {i}} \min  _ {t \in [ T ]} d (\boldsymbol {\theta}, \boldsymbol {\theta} _ {t} ^ {(i)}) + \delta_ {i} \right\}. \tag {2.9}
$$

In particular, when all parties have i.i.d. local data distributions, (2.9) reduces to

$$
\mathcal {G} \leq \max  _ {\boldsymbol {\theta} \in \tilde {\Theta}} \sum_ {i \in [ p ]} C _ {\boldsymbol {\alpha}} \left\{C _ {\tilde {L}, \hat {L} _ {i}} \min  _ {t \in [ T ]} d (\boldsymbol {\theta}, \boldsymbol {\theta} _ {t} ^ {(i)}) + \delta_ {i} \right\}.
$$

Here  $C_{\alpha}$ ,  $C_{\beta}$  and  $C_{\tilde{L},\hat{L}_i}$  are constants only related to the unified loss surface and Lipschitz-ness, respectively,  $\mathcal{W}_1(\cdot ,\cdot)$  and  $d(\cdot ,\cdot)$  are distance metrics defined over data distribution and hyperparameter space  $\Theta$ , respectively, and  $\delta_{i}$  is the maximum per sample training error for the local loss surface  $\widehat{\ell}_i$ , i.e.,  $\delta_{i} = \max_{t}|\mathcal{L}_{t}^{(i)} - \widehat{\ell}_{i}(\pmb{\theta}_{t}^{(i)})|$ .

Table 3: Comparison of different loss surfaces (the 4 rightmost columns) for FLoRA relative to the baseline for single-shot 3-party FL-HPO in terms of the relative regret (lower is better).  

<table><tr><td>Aggregate</td><td>ML Method</td><td>SGM</td><td>SGM+U</td><td>MPLM</td><td>APLM</td></tr><tr><td>Regret</td><td>HGB</td><td>[0.30, 0.47, 0.68]</td><td>[0.27, 0.54, 0.64]</td><td>[0.25, 0.43, 0.67]</td><td>[0.25, 0.50, 0.65]</td></tr><tr><td>Inter-quartile range</td><td>SVM</td><td>[0.04, 0.38, 1.11]</td><td>[0.04, 0.48, 1.07]</td><td>[0.38, 0.91, 2.41]</td><td>[0.23, 0.54, 0.76]</td></tr><tr><td></td><td>MLP</td><td>[0.36, 0.80, 0.97]</td><td>[0.48, 0.99, 1.01]</td><td>[0.47, 0.89, 1.00]</td><td>[0.46, 0.79, 0.95]</td></tr><tr><td></td><td>Overall</td><td>[0.22, 0.53, 0.97]</td><td>[0.32, 0.55, 1.01]</td><td>[0.36, 0.61, 0.99]</td><td>[0.36, 0.57, 0.79]</td></tr><tr><td>FLoRA</td><td>HGB</td><td>6/0/1</td><td>6/0/1</td><td>7/0/0</td><td>7/0/0</td></tr><tr><td>Wins/Ties/Losses</td><td>SVM</td><td>4/0/2</td><td>4/0/2</td><td>3/0/3</td><td>5/0/1</td></tr><tr><td></td><td>MLP</td><td>6/0/1</td><td>4/1/2</td><td>5/1/1</td><td>6/0/1</td></tr><tr><td></td><td>Overall</td><td>16/0/4</td><td>14/1/5</td><td>15/1/4</td><td>18/0/2</td></tr><tr><td>Wilcoxon Signed-Rank Test</td><td>HGB</td><td>(26, 0.02126)</td><td>(27, 0.01400)</td><td>(28, 0.00898)</td><td>(28, 0.00898)</td></tr><tr><td>1-sided</td><td>SVM</td><td>(18, 0.05793)</td><td>(17, 0.08648)</td><td>(9, 0.62342)</td><td>(15, 0.17272)</td></tr><tr><td>(statistic, p-value)</td><td>MLP</td><td>(21, 0.11836)</td><td>(15, 0.17272)</td><td>(18, 0.05793)</td><td>(24, 0.04548)</td></tr><tr><td></td><td>Overall</td><td>(174, 0.00499)</td><td>(164, 0.00272)</td><td>(141, 0.03206)</td><td>(183.5, 0.00169)</td></tr></table>

There are several interesting observations regarding Theorem 2.1. Firstly, the first term in our bound (2.9) characterizes the errors incurred by parties' data heterogeneity measuring via 1-Wasserstein distance (Villani, 2003), i.e., the magnitude of Non-IIDness in a FL system. We can see it vanish under the IID setting. Secondly, the last two terms measure the quality of the local HPO approximation, which can be reduced if a good loss surface is selected. For example, if we use non-parametric regression models as the loss surfaces the per-sample training error can be arbitrarily small (that is  $\delta_i\approx 0$ ), but at the cost of increasing  $\widehat{L}_i$  for  $\widehat{\ell}_i$ . Thirdly,  $\min_{t\in [T]}d(\pmb {\theta},\pmb{\theta}_t^{(i)})$  indicates that the optimality gap depends only on the HP trials  $\pmb{\theta}_{t}^{(i)}$  that are closest to the optimal HP setting. Finally, if we assume each party's training dataset  $D_{i}$  is of size  $n_i$  sampled as  $D_{i}\sim \mathcal{D}_{i}^{n_{i}}$ , we can view  $w_{i} = \frac{n_{i}}{n}$  where  $n = \sum_{i = 1}^{p}n_{i}$ , i.e., with probability  $w_{i}$  the desired data distribution  $\mathcal{D}$  is sampled from  $\mathcal{D}_i$ .

Now we would like to compare our theoretical results with existing analyses such as Khodak et al. (2021) and He et al. (2020). Among many differences in the FL-HPO problem setting, there are two key points we want to emphasize. On the one hand, Theorem 2.1 presents the first optimality gap in terms of loss function value for the single-shot FL-HPO setting, and can be applied to both algorithmic and model architecture HPs, while existing work either lacks of theoretical guarantees or establishing weaker optimality gap measured by regret defined for online setting and only applicable to single HP optimization. On the other hand, we only make mild Lipschitz assumption regarding the loss function, and do not require any assumptions on the convexity of the loss function, the parties' local data distribution, nor the training algorithms, however, existing work usually requires convexity and certain restrictions on ML training algorithm to obtain their convergence guarantees.

# 3 EMPIRICAL EVALUATION

In this section, we evaluate FLoRA with different loss surfaces for the FL-HPO problems on a variety of ML models – histograms based gradient boosted (HGB) decision trees (Friedman, 2001), Support Vector Machines (SVM) with RBF kernel and multi-layered perceptrons (MLP) (using their respective scikit-learn implementation (Pedregosa et al., 2011)) on OpenML (Vanschoren et al., 2013) classification problems. First, we fix the number of parties  $p = 3$  and compare FLoRA to a baseline on 7 datasets. Then we study the data heterogeneity effect on the performance of FLoRA. Finally, we evaluate FLoRA with different parameter choices, in particular, the number of local HPO rounds and the communication overhead in the aggregation of the per-party (HP, loss) pairs. More comprehensive experimental results and FLoRA performance on real FL systems can be found in Appendix D.

Baselines. To appropriately evaluate our proposed single-shot FL-HPO scheme, we need to select a meaningful single-shot baseline. For this, we choose the default HP configuration of scikit-learn as the single-shot baseline for two main reasons: (i) the default HP configuration in scikit-learn is set manually based on expert prior knowledge and extensive empirical evaluation, and (ii) these are also used as the defaults in the Auto-Sklearn package (Feurer et al., 2015; 2020), one of the leading open-source AutoML python packages, which maintains a carefully selected portfolio of default configurations. While there are some existing schemes for FL-HPO, we are unable to compare FLoRA to them, see Table 1 for detailed comparison.

Implementation and evaluation metric. We emulate the final FL (Algorithm 1, line 8) with a centralized training using the pooled data. We chose this implementation because we want to evaluate

the final performance of any HP configuration (baseline or recommended by FLoRA) in a statistically robust manner with multiple train/validation splits (for example, via 10-fold cross-validation) instead of evaluating the performance on a single train/validation. This form of evaluation is extremely expensive to perform in a real FL system and generally not feasible, but allows us to evaluate how the performance of our single-shot HP recommendation fairs against that of the best-possible HP found via a full-scale centralized HPO. In all datasets, we consider the balanced accuracy as the metric we wish to maximize. For the local per-party HPOs (as well as the centralized HPO we execute to compute the regret), we maximize the 10-fold cross-validated balanced accuracy. For Table 3-4, we report the relative regret, computed as  $(a^{\star} - a) / (a^{\star} - b)$ , where  $a^{\star}$  is the best metric obtained via the centralized HPO,  $b$  is the result of the baseline, and  $a$  is the result of the HP recommended by FLoRA. The baseline has a relative regret of 1 and smaller values imply better performance. A value larger than 1 implies that the recommended HP performs worse than the baseline.

Comparison to single-shot baseline. We first compare FLoRA with the baseline across different datasets, ML models and FLoRA loss surfaces summarized in Table 3 with the individual results detailed in Appendix D.3. For each method, we report the aggregate performance over all considered datasets in terms of (i) inter-quartile range, (ii) Wins/Ties/Losses of FLoRA w.r.t. the single-shot baseline, and (iii) a one-sided Wilcoxon Signed Ranked Test of statistical significance with the null hypothesis that the median of the difference between the single-shot baseline and FLoRA is positive against the alternative that the difference is negative (implying FLoRA improves over the baseline). Finally, we report an "Overall" performance, further aggregated across all ML models.

All FLoRA loss surfaces show strong performance w.r.t the single-shot baseline, with significantly more wins than losses, and 3rd-quartile relative regret values less than 1 (indicating improvement over the baseline). All FLoRA loss surfaces have a p-value of less than 0.05, indicating that we can reject the null hypothesis. Overall, APLM shows the best performance over all loss surfaces, both in terms of Wins/Ties/Losses over the baseline as well as in terms of the Wilcoxon Signed Rank Test, with the highest statistic and a p-value close to  $10^{-3}$ . APLM also has significantly

Table 4: Effect of increasing the number of parties on FLoRA with different loss surfaces for HGB.  

<table><tr><td>Data</td><td>p</td><td>γp</td><td>SGM</td><td>SGM+U</td><td>MPLM</td><td>APLM</td></tr><tr><td>EEG</td><td>3</td><td>1.01</td><td>0.14</td><td>0.12</td><td>0.11</td><td>0.12</td></tr><tr><td>14980 rows</td><td>10</td><td>1.03</td><td>0.08</td><td>0.00</td><td>0.16</td><td>0.01</td></tr><tr><td></td><td>25</td><td>1.08</td><td>0.35</td><td>0.92</td><td>0.17</td><td>0.04</td></tr><tr><td></td><td>50</td><td>1.20</td><td>0.20</td><td>0.23</td><td>0.67</td><td>0.12</td></tr><tr><td>Electricity</td><td>3</td><td>1.01</td><td>0.17</td><td>0.14</td><td>0.09</td><td>0.12</td></tr><tr><td>45312 rows</td><td>10</td><td>1.02</td><td>0.03</td><td>0.06</td><td>0.32</td><td>0.14</td></tr><tr><td></td><td>25</td><td>1.04</td><td>0.40</td><td>0.42</td><td>1.42</td><td>0.89</td></tr><tr><td></td><td>50</td><td>1.07</td><td>1.57</td><td>1.57</td><td>0.89</td><td>1.13</td></tr><tr><td></td><td>100</td><td>1.14</td><td>1.45</td><td>1.47</td><td>0.48</td><td>1.11</td></tr><tr><td>Pollen</td><td>3</td><td>1.02</td><td>0.43</td><td>0.54</td><td>0.43</td><td>0.69</td></tr><tr><td>3848 rows</td><td>6</td><td>1.10</td><td>1.02</td><td>0.91</td><td>0.54</td><td>0.56</td></tr><tr><td></td><td>10</td><td>1.16</td><td>1.05</td><td>0.73</td><td>0.75</td><td>1.12</td></tr></table>

lower 3rd-quartile than all other loss surfaces. MPLM appears to have the worst performance but much of that is attributable to a couple of very hard cases with SVM (see Appendix D.3 for detailed discussion). Otherwise, MPLM performs second best both for FL-HPO with HGB and MLP.

Effect of data heterogeneity. In the second set of experiments, we study the effect of increasing the number of parties in the FL-HPO problem. For each data set, we increase the number of parties  $p$  up until each party has at least 100 training samples. We present the relative regrets in Table 4. It also displays  $\gamma_{p} := (1 - \min_{i \in [p]} \mathcal{L}_{\star}^{(i)}) / (1 - \max_{i \in [p]} \mathcal{L}_{\star}^{(i)})$ , where  $\mathcal{L}_{\star}^{(i)} = \min_{t \in [T]} \mathcal{L}_t^{(i)}$  is the minimum loss observed during the local HPO at party  $i$ . This ratio  $\gamma_{p}$  is always greater than 1, and quantifies the inter-party data heterogeneity – precisely  $\gamma_{p} \sim 1 + \widetilde{O}\left(\max_{i,j \in [p]} \mathcal{W}_{1}(\mathcal{D}_{i}, \mathcal{D}_{j})\right)$  (Appendix C.5).

The results indicate that, with low or moderate increase in  $\gamma_{p}$  (EEG eye state, Electricity for moderate  $p$ ), the proposed scheme is able to achieve low relative regret. However, with significant increase in  $\gamma_{p}$  (Pollen, Electricity with  $p = 50,100$

Table 5: Effect of data heterogeneity on FLoRA with MNIST.  

<table><tr><td>Method</td><td>γp</td><td>SGM</td><td>SGM+U</td><td>MPLM</td><td>APLM</td></tr><tr><td>MLP</td><td>1.01</td><td>0.85</td><td>0.56</td><td>0.29</td><td>0.06</td></tr><tr><td>HGB</td><td>1.01</td><td>0.51</td><td>0.92</td><td>0.64</td><td>0.51</td></tr></table>

and EEG Eye State with  $p = 50$ , the relative regret increases as well (even  $>1$  in a few cases).

We also simulate different degrees of data heterogeneity based on MNIST dataset and present the results in Table 5. In particular, there are 4 parties in total, half of the parties with 4 times higher probability to have more even digits while the other half have more odd digits. In most challenging cases, MPLM (the most pessimistic loss function) has the most graceful degradation in relative regret compared to the remaining loss surfaces.

![](images/563eb11654085b0ffc6bf0dc059ee6c9b8cdf846da4ee786fdbdf4e5892f161c.jpg)  
(a) # local HPO rounds.

![](images/ea1871b4c20b25c17d48a5201f5b46db0a852da1c98990c311352d39fd3284c0.jpg)  
Figure 3: Effect of different choices on FLoRA with the APLM loss surface for different methods and datasets. More results and other loss surfaces are presented in Appendix D.6 and D.7.  
(b) # (HP, loss) pairs communicated to aggregator

Communication savings over multi-shot. Figure 2 presents the communication overhead savings (in terms of the number of FL model trainings) from FLoRA over multi-shot HPO by applying standard HPO to FL-HPO. For ease of exposition, we present results with HGB with 2 loss surfaces – MPLM and APLM. More results are presented in Appendix D.4. Similar to Figure 1, we report the number of FL model trainings required for multi-shot (■) to match the relative regret achieved by FLoRA (★), with the single-shot baseline performance (relative regret of 1 with 1 FL model training, denoted by ●) presented as a reference. In aggregate, FLoRA with APLM achieves a median savings of  $8 \times$ ,  $15 \times$  and  $10 \times$  over the multi-shot baseline for HGB, SVM and MLP respectively.

Effect of different choices in FLoRA. In this set of experiments, we consider FLoRA with the APLM loss surface, and ablate the effect of different choices in FLoRA on 2 datasets each for SVM and MLP. First, we study the impact of the thoroughness of the per-party local HPOs, quantified by the number of HPO rounds  $T$  in Figure 3a. The results indicate that for really small  $T (< 20)$  the relative regret of FLoRA can be very high. However, after that point, the relative regret converges to its best possible value. We present the results for other loss surfaces in Appendix D.6.

We also study the effect of the communication overhead of FLoRA for fixed level of local HPO thoroughness. We assume that each party performs  $T = 100$  rounds of local asynchronous HPO. However, instead of sending all  $T$  (HP, loss) pairs, we consider sending  $T' < T$  of the "best" (HP, loss) pairs – that is, (HP, loss) pairs with the  $T'$  lowest losses. Changing the value of  $T'$  trades off the communication overhead of the FLoRA step where the aggregators collect the per-party loss pairs (Algorithm 1, line 5). The results for this study are presented in Figure 3b, and indicate that, for really small  $T'$ , the relative regret can be really high. However, for a moderately high value of  $T' < T$ , FLoRA converges

![](images/a3bf7883108404c15a84b1ad86a4c5d5db582d761f9184ee516a6daaf2040b25.jpg)  
Figure 2: Communication savings of FLoRA compared to "multi-shot" FL-HPO for the same level of relative regret (lower is better). Each pair of  $(\star, \blacksquare)$  connected by a dashed line corresponds to a dataset labeled as D1-D7 for ease of visualization. See Table 6 in Appendix D.1 for dataset names.

to its best possible performance. Results and discussions on other loss surfaces are in Appendix D.7.

# 4 CONCLUSION AND FUTURE WORK

Effective selection of HPs in FL settings is a challenging problem. In this paper, we introduced FLoRA, a single-shot FL-HPO algorithm that can be applied to any ML model. We provided a theoretical analysis which bounds on the optimality gap incurred by HP selected by FLoRA. Our experimental evaluation shows that FLoRA can effectively produce HP configurations that outperform the baseline with just a single shot.

One limitation of FLoRA is that it cannot handle HPs that are not active during any local HPO. These would include aggregator specific and some FL training specific HPs. It is unlikely that such HPs can be handled in single-shot FL-HPO without any additional information or structure. As future work, we wish to explore how FLoRA can be extended to handle such HPs in "few-shot" FL-HPO and in conjunction with some form of multi-fidelity HP evaluations.

# 5 REPRODUCIBILITY STATEMENT

The code and instructions to reproduce our numerical results can be found in supplemental materials. For formal definitions, assumptions and proofs of Theorem 2.1, one can find them in Appendix C. We provide a description of the dataset used in our experiments in Appendix D.1. One can also find the original datasets in supplemental materials.

# REFERENCES

James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281-305, 2012.  
James S Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-parameter optimization. In Advances in neural information processing systems, pp. 2546-2554, 2011.  
Leo Breiman. Random forests. Machine learning, 45(1):5-32, 2001.  
Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pp. 785-794, New York, NY, USA, 2016. ACM. ISBN 978-1-4503-4232-2. doi: 10.1145/2939672.2939785. URL http://doi.acm.org/10.1145/2939672.2939785.  
Z. Dai, B.K.H. Low, and P. Jaillet. Federated bayesian optimization via thompson sampling. Advances in Neural Information Processing Systems, 33, 2020.  
Z. Dai, B.K.H. Low, and P. Jaillet. Differentially private federated bayesian optimization with distributed exploration. Advances in Neural Information Processing Systems, 34, 2021.  
Stefan Falkner, Aaron Klein, and Frank Hutter. Bohb: Robust and efficient hyperparameter optimization at scale. In International Conference on Machine Learning, pp. 1437-1446. PMLR, 2018.  
Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Springenberg, Manuel Blum, and Frank Hutter. Efficient and robust automated machine learning. In Advances in Neural Information Processing Systems, pp. 2962-2970, 2015.  
Matthias Feurer, Katharina Eggensperger, Stefan Falkner, Marius Lindauer, and Frank Hutter. Autoklearn 2.0: The next generation. In arXiv:2007.04074 [cs.LG], 2020.  
Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pp. 1189-1232, 2001.  
Anubhav Garg, Amit Kumar Saha, and Debo Dutta. Direct federated neural architecture search. arxiv.2010.06223, 2020.  
Gary Gensler and Lily Bailey. Deep learning and financial stability. Available at SSRN 3723132, 2020.  
Bryce Goodman and Seth Flaxman. European union regulations on algorithmic decision-making and a "right to explanation". AI magazine, 38(3):50-57, 2017.  
Chaoyang He, Murali Annavaram, and Salman Avestimehr. Towards non-i.i.d. and invisible data with fednas: Federated deep learning via neural architecture search. arxiv.2004.08546, 2020.  
Frank Hutter, Lars Kotthoff, and Joaquin Vanschoren (eds.). Automated Machine Learning - Methods, Systems, Challenges. Springer, 2019.  
Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977, 2019.  
Mikhail Khodak, Tian Li, Liam Li, M Balcan, Virginia Smith, and Ameet Talwalkar. Weight sharing for hyperparameter optimization in federated learning. In Int. Workshop on Federated Learning for User Privacy and Data Confidentiality in Conjunction with ICML 2020, 2020.

Mikhail Khodak, Renbo Tu, Tian Li, Liam Li, Maria-Florina Balcan, Virginia Smith, and Ameet Talwalkar. Federated hyperparameter tuning: Challenges, baselines, and connections to weight-sharing. arXiv preprint arXiv:2106.04502, 2021.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster), 2015. URL http://arxiv.org/abs/1412.6980.  
Aaron Klein, Stefan Falkner, Simon Bartels, Philipp Hennig, and Frank Hutter. Fast bayesian optimization of machine learning hyperparameters on large datasets. In Artificial intelligence and statistics, pp. 528-536. PMLR, 2017.  
A. Koskela and A. Honkela. Learning rate adaptation for federated and differentially private learning. arXiv preprint arXiv:1809.03832, 2019.  
Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18(185):1-52, 2018.  
Qinbin Li, Zeyi Wen, and Bingsheng He. Practical federated gradient boosting decision trees. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 4642-4649, 2020.  
Heiko Ludwig, Nathalie Baracaldo, Gegi Thomas, Yi Zhou, Ali Anwar, Shashank Rajamoni, Yuya Ong, Jayaram Radhakrishnan, Ashish Verma, Mathieu Sinn, et al. IBM Federated Learning: an enterprise framework white paper v0.1. arXiv preprint arXiv:2007.10987, 2020. URL https://github.com/IBM/federated-learning-lib.  
B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. Arcas. Communication-efficient learning of deep networks from decentralized data. In Proc. International Conference on Artificial Intelligence and Statistics, pp. 1273-1282, Ft. Lauderdale, FL, 20-22 Apr 2017a.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017b.  
H. Mostafa. Robust federated learning through representation matching and adaptive hyperparameters. arXiv preprint arXiv:1912.13075, 2019.  
OECD. Artificial intelligence, machine learning and big data in finance: Opportunities, challenges, and implications for policy makers, 2021. URL https://www.oecd.org/finance/artificial-intelligence-machine-learning-big-data-in-finance.htm.  
Changyong Oh, Jakub M Tomczak, Efstratios Gavves, and Max Welling. Combinatorial bayesian optimization using the graph cartesian product. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 2914-2924, 2019.  
Yuya Jeremy Ong, Yi Zhou, Nathalie Baracaldo, and Heiko Ludwig. Adaptive histogram-based gradient boosted trees for federated learning. arXiv preprint arXiv:2012.06670, 2020.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.  
S.J. Reddi, Z. Charles, M. Zaheer, Z. Garrett, K. Rush, J. Konecy, S. Kumar, and H.B. McMahan. Adaptive federated optimization. In International Conference on Learning Representations, 2020.  
B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148-175, 2016.  
Kevin Swersky, Jasper Snoek, and Ryan Prescott Adams. Freeze-thaw bayesian optimization. arXiv preprint arXiv:1406.3896, 2014.  
Joaquin Vanschoren. Meta-learning: A survey. arXiv preprint arXiv:1810.03548, 2018.

Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. OpenML: Networked science in machine learning. SIGKDD Explorations, 15(2):49-60, 2013. doi: 10.1145/2641190.2641198. URL http://doi.acm.org/10.1145/2641190.2641198.  
Cedric Villani. Topics in optimal transportation.(books). OR/MS Today, 30(3):66-67, 2003.  
Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. Journal Selected Areas in Communications (JSAC), 2019.  
Christopher K Williams and Carl Edward Rasmussen. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA, 2006.  
Martin Wistuba, Nicolas Schilling, and Lars Schmidt-Thieme. Scalable gaussian process-based transfer surrogates for hyperparameter optimization. Machine Learning, 107(1):43-78, 2018.  
Mengwei Xu, Yuxin Zhao, Kaigui Bian, Gang Huang, Qiaozhu Mei, and Xuanzhe Liu. Federated neural architecture search. arxiv.2002.06352, 2020.  
WANG Zhen, Weirui Kuang, Ce Zhang, Bolin Ding, and Yaliang Li. Fedhpo-b: A benchmark suite for federated hyperparameter optimization. 2022. URL https://github.com/alibaba/FederatedScope/tree/master/benchmark/FedHPOB.
