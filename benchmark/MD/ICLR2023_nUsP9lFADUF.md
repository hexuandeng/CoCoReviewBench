# PROTOTYPICAL CALIBRATION FOR FEW-SHOT LEARNING OF LANGUAGE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

In-context learning of GPT-like models has been recognized as fragile across different hand-crafted templates, and demonstration permutations. In this work, we propose prototypical calibration to adaptively learn a more robust decision boundary for zero- and few-shot classification, instead of greedy decoding. Concretely, our method first adopts Gaussian mixture distribution to estimate the prototypical clusters for all categories. Then we assign each cluster to the corresponding label by solving a weighted bipartite matching problem. Given an example, its prediction is calibrated by the likelihood of prototypical clusters. Experimental results show that prototypical calibration yields a substantial improvement on a diverse set of tasks. Extensive analysis across different scales also indicates that our method calibrates the decision boundary as expected, greatly improving the robustness of GPT to templates, permutations, and class imbalance.

# 1 INTRODUCTION

Large-scale language models (LMs) have shown strong generalization ability on a wide range of downstream tasks (Devlin et al., 2018; Radford et al., 2019; Yang et al., 2019; Lewis et al., 2019; Brown et al., 2020; Dong et al., 2019; Bao et al., 2020). Fine-tuning has been the common strategy to transfer the extensive knowledge to downstream tasks for a long time.

However, fine-tuning such large LMs suffers from over-parameterization issue under few-shot settings. Brown et al. (2020) propose the concept of in-context learning with GPT, which enables LMs to quickly adapt to a new task by conditioning on hand-crafted prompts as shown in Figure 1. The prompts consist of task-specific templates and several input-label pairs (demonstrations). In-context learning is surprising as GPT can perform various tasks without any parameter updating.

It has been noticed that the predictions of GPT conditioned on prompts tend to bias toward some specific answers and can be highly volatile across different templates, demonstrations, and their permutations (Lu et al., 2021; Jiang et al., 2020). Zhao et al. (2021) propose to calibrate the model prediction by the content-free output to mitigate this problem. Rubin et al. (2021) and Lu et al. (2021) focus on the training examples retrieval and optimal ordering selection respectively to produce more performant prompts than random sampling. However, they did not explain why the in-context learning performance is fragile across different scenarios.

In this paper, we analyze the intrinsic reason for the instability of few-shot learning with GPT. We observe significant distinctions among the prediction distributions of GPT under different prompts. As shown in Figure 2, the conventional decision boundary of GPT (i.e., naively uses the output with the largest probability as the predicted label) often fails to discriminate the predictions. We argue that the predictions can be more discriminative when provided a calibrated decision boundary.

Specifically, we term the model outputs of examples whose ground-truth are the same category as prototypical cluster and adopt Gaussian Mixture Model (GMM) to estimate the distributions of them for all categories. The decision boundaries of the prototypical clusters are adaptively learned, which is called prototypical calibration (PROCA). Then the prototypical clusters are assigned to the corresponding labels through a weighted bipartite matching. We also propose to improve estimations according to cluster-label assignment. Finally, the predictions of test examples are more precise owing to the calibrated decision boundary (as shown in Figure 2).

![](images/4804c4922b26557c48415e36208699a3ecec9b88982a1b2ea357cf770160907c.jpg)  
Figure 1: Example of few-shot learning with GPT.

![](images/4218b904478d8fcaa971e8d16aa898410b0bc3799bd8a65e76b0c00a5d8275b1.jpg)  
Figure 2: Left and Middle: Prediction distribution of GPT-2-XL under two different prompts for SST-2. Two distributions colored by blue and red represent model predictions for negative and positive ground-truth examples respectively.  $P_{\text{positive}}$  denotes the prediction probability of positive label. The orange dashed line represents the decision boundary commonly used by GPT (i.e.,  $P_{\text{positive}} = 0.5$  for binary classification). The green dashed line represents the decision boundary of our prototypical calibration (PROCA). Right: Performance comparison of GPT-2-XL and PROCA under the two prompts, which indicates that PROCA is effective because the calibrated decision boundary is more discriminative for classification.

![](images/3605edbf236bfe4c5d5514cd4c2d226bbb52ae1a7130c86d0d1620df06303cf0.jpg)

![](images/a34213e9335ba5fa001f86346223c36a472afd9a62b86b27f380bc0ecd2bbf05.jpg)

Experimental results show that we achieve on average  $13\%$  absolute improvement for different sizes of GPT models across nine text classification datasets. We demonstrate that PROCA is effective across various templates and different demonstration permutations.

To summarize, our key contributions are as follows:

- We find that the decision boundary plays a critical role in few-shot evaluation. Moreover, performant decision boundaries are inconsistent across language models and prompts.  
- We propose prototypical calibration to adaptively learn a better decision boundary for few-shot classification of language models.  
- Experiments show that PROCA achieves a  $13\%$  absolute improvement over the conventional approach on a wide range of text classification tasks.

# 2 DECISION BOUNDARY OF FEW-SHOT LEARNING WITH GPT

A decision boundary refers to an explicit prediction criterion in the output space for a given classification problem. As shown in Figure 2, two dashed lines represent two different decision boundaries, which classifies examples into negative and positive categories. In this section, we explore the effect of decision boundary on few-shot learning. We demonstrate that optimal decision boundaries are inconsistent under different LMs and prompts.

Decision boundary greatly influences the few-shot performance. We evaluate the performance of different models and prompts using different decision boundaries. Results are shown in Figure 3. The red rectangle indicates the conventional decision boundary used by GPT, which naively decodes the label with larger prediction probability. We observe that shifting the decision boundary can cause wild fluctuations of few-shot accuracy, from near state-of-the-art to random guessing. For each

![](images/ab8d095460c32c87c279ce8c79c490116be334daf8e5703449f7dd9ed749f735.jpg)  
Figure 3: Few-shot performance of GPT-2-Large (0.8B) and GPT-J (6B) using different decision boundaries. P1, P2, P3, and P4 represent different prompts. The red rectangle indicates the performance under the conventional decision boundary ( $P_{\text{positive}} = 0.5$  for the example task), i.e., naively using the outputs with larger probabilities as the predicted labels. It is observed that the decision boundary plays a critical role in few-shot evaluation.

prompt, there is an exclusive region where the decision boundary is relatively robust. The model exhibits poor performance when the decision boundary is far from the robust region.

Performant decision boundaries are not transferable across LMs or prompts. Figure 3 demonstrates that all prompts exhibit strong performance if the decision boundary locates in the robust region. However, different prompts and models lead to different degrees of deviation between the optimal decision boundary and the conventional one. It suggests that performant decision boundaries are inconsistent across models or prompts. Based on the above analysis, we argue that all prompts can achieve better performance when the decision boundary is calibrated into the robust region.

# 3 PROTOTYPICAL CALIBRATION

We have illustrated that the conventional decision boundary generally deviates from the robust region, which renders in-context learning fragile. In this section, we present prototypical calibration (PROCA) to adaptively learn a better decision boundary.

# 3.1 PROTOTYPICAL CLUSTER ESTIMATION

Considering an  $N$ -way few-shot classification task, let  $X$  denote the  $N$ -dimensional model outputs. For examples whose ground-truth is the  $n$ -th category, the model outputs compose a prototypical cluster. For instance, the red and blue areas in Figure 2 refer to two prototypical clusters respectively.

We assume that each prototypical cluster follows a Gaussian distribution:

$$
P _ {\mathrm {G}} (X | \mu_ {n}, \Sigma_ {n}) = \frac {1}{(2 \pi) ^ {N / 2} | \Sigma_ {n} | ^ {1 / 2}} \exp \left(- \frac {1}{2} (X - \mu_ {n}) ^ {T} \Sigma_ {n} ^ {- 1} (X - \mu_ {n})\right), \tag {1}
$$

where  $\mu_{n}$  and  $\Sigma_{n}$  are the mean vector, and covariance matrix of the distribution, respectively. Next, we estimate  $N$  prototypical clusters for  $N$  categories with Gaussian mixture model (GMM):

$$
P _ {\mathrm {G M M}} (X) = \sum_ {n = 1} ^ {N} \alpha_ {n} P _ {\mathrm {G}} (X | \mu_ {n}, \Sigma_ {n}), \tag {2}
$$

where  $\alpha_{n}$  is the mixing coefficient for the  $n$ -th distribution. In our work, we formulate the model prediction  $x = [x_{1}, x_{2}, \ldots, x_{N}]$  as follows:

$$
x _ {n} = \log \frac {\exp \left(o _ {n}\right)}{\sum_ {i = 1} ^ {N} \exp \left(o _ {i}\right)}, \tag {3}
$$

where  $o_n$  and  $o_i$  are the logits predicted by GPT, corresponding to label token  $n$  and label token  $i$  respectively. Intuitively,  $x_n$  represents the log probability of the  $n$ -th category.

After clarifying the GMM definition under few-shot learning, we utilize a small-scale unlabeled in-domain dataset, named as estimate set  $(D_{\mathrm{esti}})$ , to estimate the parameters  $\{\alpha_n,\mu_n,\Sigma_n\}_{n = 1}^N$  by the Expectation-Maximization (EM) algorithm (Moon, 1996). Notice that the estimate set does not contain any human annotation. Specifically, EM is an iterative method to find the optimal estimation of GMM's parameters by maximizing the likelihood  $\prod_{x\in D_{\mathrm{esti}}}P_{\mathrm{GMM}}(x)$ .

# 3.2 CLUSTER-LABEL ASSIGNMENT

Then we assign the estimated prototypical clusters to the target labels. Concretely, for an estimation  $e = \{(\alpha_{n},\mu_{n},\Sigma_{n})\}_{n = 1}^{N}$ , the estimated parameter  $\mu_{n,l}$  indicates the belongingness of cluster  $n$  to label  $l$ . Therefore, we propose a cluster-label assignment score  $\mathrm{CLA}(\cdot)$ , which represents the overall belongingness of a cluster-label assignment. Let the tuple  $k = (k_{1},k_{2},\dots ,k_{N})$  denote a cluster-label assignment, where  $k$  is a permutation of  $\{1,2,\ldots ,\mathbf{N}\}$ . It means that the  $n$ -th cluster is assigned to the label  $k_{n}$ . The assignment score  $\mathrm{CLA}(\cdot)$  is defined as:

$$
\operatorname {C L A} (e, k) = \sum_ {n = 1} ^ {N} \mu_ {n, k _ {n}}, \tag {4}
$$

where  $\mu_{n,k_n}$  indicates how much the  $n$ -th cluster of  $e$  belongs to label  $k_n$ .

Then it is transformed to a weighted bipartite matching problem between  $N$  clusters and  $N$  labels. The optimal assignment is obtained by maximizing  $\mathrm{CLA}(e,k)$ :

$$
k ^ {*} (e) = \underset {k \in \mathcal {K}} {\arg \max } \operatorname {C L A} (e, k), \tag {5}
$$

where  $\mathcal{K}$  indicates the set of all assignment permutations. In the worst case, this process requires  $N!$  attempts to find the optimal assignment, which is time-consuming when  $N$  is large, thus we adopt Kuhn-Munkres algorithm (Kuhn, 1955) to accelerate it.

# 3.3 ESTIMATION SELECTION BASED ON CLUSTER-LABEL ASSIGNMENT

The EM algorithm is empirically sensitive to the different initializations of GMM parameters. So we repeat the estimation multiple times with different random seeds. Then we define a metric to evaluate how good these estimations are and select the best estimation. As  $\mathrm{CLA}(e, k^*)$  reflects the overall label belongingness of the optimal assignment of an estimation  $e$ , thus it can be used to evaluate estimations. Formally, we select the estimation  $e^*$  according to the assignment score of  $k^*$  as follows:

$$
e ^ {*} = \underset {e \in \mathcal {E}} {\arg \max } \operatorname {C L A} (e, k ^ {*} (e)), \tag {6}
$$

where  $\mathcal{E}$  is the set of estimations obtained by different initializations of GMM parameters.

# 3.4 INFERENCE

After selecting the desired estimation  $e^{*}$ , we use GMM to make predictions instead of the conventional approach used in GPT (Brown et al., 2020). Due to the class-distribution discrepancy between the estimate set and the test set, we discard the mixing coefficient  $\alpha_{n}$  of each sub-distribution during inference. For a test example, the LM prediction is  $x$ . It will be assigned to the most likely cluster:

$$
\tilde {n} = \underset {n = 1, \dots , N} {\arg \max } P _ {\mathrm {G}} \left(x \mid \mu_ {n} ^ {*}, \Sigma_ {n} ^ {*}\right). \tag {7}
$$

Finally, the predicted label is  $k_{\tilde{n}}^{*}(e^{*})$ , where the cluster-label assignment  $k^{*}(e^{*})$  is obtained according to Equation (5).

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL SETUP

We evaluate five models from GPT-family including GPT-2-large (Radford et al., 2019) with 0.8B parameters, GPT-2-XL (Radford et al., 2019) with 1.5B parameters, GPT-neo (Black et al., 2021) with

2.7B parameters, GPT-J (Wang & Komatsuzaki, 2021) with 6B parameters, and Bloom (BigScience, 2022) with 176B parameters.

As for the estimate set, it can be constructed by generating from LMs (Lu et al., 2021; Wang et al., 2021; Meng et al., 2022; Ye et al., 2022) or sampling a light sub set of training examples but without golden labels. For simplicity, we choose the later way to construct the estimate set and we further compare their differences in Section 4.5. Moreover, the estimate set size is proportional to the number of classes of the task. For more details, please refer to Table 7 in Appendix.

We use the  $k$ -means algorithm to initialize GMM parameters to accelerate the convergence. The maximum iterations and the convergence threshold for each EM process are set to 100 and 1e-3 respectively. Moreover, we repeat the estimation multiple times with different random initializations to avoid getting stuck in local optima. It is worth noting that multiple repetitions bring little additional time consumption compared to the inference of GPT, we thus simply set it to 100 for all tasks.

# 4.2 EVALUATION PROTOCOL

We evaluate the proposed method on nine widely-used text-classification datasets including SST-2 (Socher et al., 2013), SST-5 (Socher et al., 2013), Subj (Pang & Lee, 2004), MR (Pang & Lee, 2005), AP (Zhang et al., 2015), DBPedia (Zhang et al., 2015), AGNews (Zhang et al., 2015), RTE (Dagan et al., 2005), and TREC (Voorhees & Tice, 2000). SST-2, SST-5, MR and AP are sentiment classification tasks. RTE is a textual entailment recognition task and TREC is a text retrieval question classification task. Subj and AGNews are subjectivity and topic classification tasks respectively, and DBPeida is an ontology classification task. We use the full validation set for evaluation except for AGNews, DBPedia and AP, for which we randomly sample 2000 test examples.

We compare PROCA with the conventional approach used by GPT (Brown et al., 2020) and contextual calibration (Zhao et al., 2021). Experiments are conducted under 0-shot, 1-shot, 4-shot and 8-shot scenarios. We fix the template format for each dataset (details of templates are shown in Table 6) and use the randomly sampled training examples as demonstrations. We compute the average accuracy on validation set over five random seeds for each setting except for Bloom using 2 seeds. We conduct evaluation on 8 Tesla A100 GPUs for Bloom and Tesla V100 GPUs for other models.

# 4.3 MAIN RESULTS

We report the mean and standard deviation of accuracy across five different random seeds for GPT-2-XL, GPT-J and Bloom in Table 1. The results of GPT-2-Large and GPT-neo are shown in Table 4 of Appendix. From Table 1 and Table 4, we observe that ProCA achieves, on average, a  $13\%$  absolute improvement compared to the conventional approach and a  $6\%$  absolute improvement over contextual calibration. In some cases, the absolute improvement can be up to  $40\%$  and  $20\%$  respectively, like GPT-J 0-shot on DBpedia and GPT-2-XL 8-shot on AGNews.

Results show that PROCA maintains high effectiveness across different model sizes and few-shot scenarios, indicating its strong generalization ability. Moreover, compared to the conventional approach, PROCA achieves considerable improvements with lower variance across different prompts in most cases, which suggests that PROCA can effectively calibrate the decision boundary for various prompts (as illustrated in Figure 2). It also reflects that our estimation strategy is reliable and insensitive to different estimate sets, because of the low variance of PROCA's zero-shot performance. We observe that the performance gain on Bloom is smaller than that on relatively small models. It also shows that Bloom's performance variation under different prompts is also much smaller than other models. It suggests that huge LMs have less suffering on the decision boundary deviation problem and are more robust to different prompts. In addition, PROCA seems invalid for GPT-2-XL on RTE. We identify the reason is that the entailment recognition task is too challenging for relatively small models like GPT-2-XL and the output of LM on such challenging tasks is no more discriminative (same for GPT-2-Large, as shown in Table 4 in Appendix).

# 4.4 EFFECTIVENESS ANALYSIS

We conduct more experiments to verify the effectiveness of ProCA. The experimental results are the average accuracy of GPT-2-XL conditioned on 5 different 4-shot prompts unless otherwise specified.

Table 1: Performance comparisons among the conventional approach (GPT; Brown et al. 2020), contextual calibration (ConCa; Zhao et al. 2021) and prototypical calibration (PROCA; Ours). We report the mean and the standard deviation of accuracy across 5 different prompts on the validation set except for Bloom, for which we only use 2 random seeds to reduce the computational cost. We also show the average performance across nine datasets. The results of ConCa are replicated based on the released code<sup>1</sup>. The standard deviation of 0-shot accuracy for PROCA is caused by the difference of estimate sets over 5 random seeds. It shows that PROCA generally outperforms GPT and ConCa.  

<table><tr><td>Shot</td><td>Method</td><td>SST-2</td><td>SST-5</td><td>MR</td><td>Subj</td><td>AP</td><td>AGNews</td><td>DBpedia</td><td>RTE</td><td>TREC</td><td>Avg</td></tr><tr><td colspan="12">GPT-2-XL 1.5B</td></tr><tr><td rowspan="3">0-shot</td><td>GPT</td><td>58.70.0</td><td>28.40.0</td><td>58.90.0</td><td>57.60.0</td><td>51.80.0</td><td>41.60.0</td><td>60.30.0</td><td>50.00.0</td><td>28.60.0</td><td>48.4</td></tr><tr><td>ConCa</td><td>69.30.0</td><td>22.60.0</td><td>66.90.0</td><td>72.90.0</td><td>49.80.0</td><td>67.70.0</td><td>54.30.0</td><td>50.40.0</td><td>42.80.0</td><td>55.2</td></tr><tr><td>PROCA</td><td>84.80.2</td><td>45.01.3</td><td>82.00.2</td><td>73.30.1</td><td>49.80.3</td><td>64.61.4</td><td>73.63.0</td><td>49.20.7</td><td>42.02.7</td><td>62.7</td></tr><tr><td rowspan="3">1-shot</td><td>GPT</td><td>59.814.0</td><td>26.28.5</td><td>51.30.6</td><td>54.58.6</td><td>51.00.1</td><td>37.46.7</td><td>51.312.7</td><td>53.81.0</td><td>29.16.5</td><td>46.0</td></tr><tr><td>ConCa</td><td>76.42.2</td><td>30.25.7</td><td>69.45.0</td><td>62.07.0</td><td>60.34.0</td><td>65.03.8</td><td>70.97.4</td><td>53.10.9</td><td>40.53.3</td><td>58.6</td></tr><tr><td>PROCA</td><td>89.42.4</td><td>42.52.9</td><td>84.31.0</td><td>71.85.7</td><td>69.88.2</td><td>69.84.3</td><td>79.93.8</td><td>49.51.9</td><td>43.65.0</td><td>66.7</td></tr><tr><td rowspan="3">4-shot</td><td>GPT</td><td>66.313.7</td><td>31.37.4</td><td>56.59.9</td><td>53.44.9</td><td>50.90.1</td><td>40.913.0</td><td>61.37.6</td><td>52.03.5</td><td>23.85.7</td><td>48.5</td></tr><tr><td>ConCa</td><td>79.910.2</td><td>33.53.5</td><td>67.78.9</td><td>68.08.7</td><td>75.65.9</td><td>59.96.3</td><td>74.95.0</td><td>52.90.7</td><td>41.14.3</td><td>61.5</td></tr><tr><td>PROCA</td><td>90.40.6</td><td>39.64.5</td><td>78.111.8</td><td>74.810.2</td><td>80.17.1</td><td>67.413.5</td><td>87.24.9</td><td>52.21.5</td><td>46.02.5</td><td>68.4</td></tr><tr><td rowspan="3">8-shot</td><td>GPT</td><td>57.09.0</td><td>30.57.9</td><td>65.212.7</td><td>57.911.2</td><td>50.90.0</td><td>42.94.2</td><td>67.97.1</td><td>53.02.1</td><td>37.24.9</td><td>51.4</td></tr><tr><td>ConCa</td><td>73.911.6</td><td>28.73.4</td><td>74.18.4</td><td>68.38.3</td><td>71.17.4</td><td>55.914.0</td><td>75.04.2</td><td>53.10.2</td><td>45.81.7</td><td>60.7</td></tr><tr><td>PROCA</td><td>88.01.3</td><td>36.54.4</td><td>80.86.4</td><td>80.23.3</td><td>79.37.8</td><td>75.53.2</td><td>89.40.7</td><td>51.32.0</td><td>46.02.5</td><td>69.7</td></tr><tr><td colspan="12">GPT-J 6B</td></tr><tr><td rowspan="3">0-shot</td><td>GPT</td><td>66.60.0</td><td>26.60.0</td><td>65.90.0</td><td>67.90.0</td><td>54.20.0</td><td>33.70.0</td><td>21.80.0</td><td>55.20.0</td><td>23.40.0</td><td>46.1</td></tr><tr><td>ConCa</td><td>57.70.0</td><td>35.40.0</td><td>57.10.0</td><td>59.90.0</td><td>63.10.0</td><td>60.10.0</td><td>49.90.0</td><td>55.60.0</td><td>42.20.0</td><td>53.4</td></tr><tr><td>PROCA</td><td>74.20.2</td><td>42.10.8</td><td>73.10.4</td><td>69.50.2</td><td>63.30.2</td><td>55.10.4</td><td>66.11.5</td><td>57.01.0</td><td>53.46.1</td><td>61.5</td></tr><tr><td rowspan="3">1-shot</td><td>GPT</td><td>67.77.3</td><td>31.74.9</td><td>68.14.1</td><td>65.010.9</td><td>92.92.7</td><td>65.614.6</td><td>65.614.8</td><td>52.64.6</td><td>41.89.0</td><td>61.2</td></tr><tr><td>ConCa</td><td>89.32.2</td><td>46.53.4</td><td>88.51.1</td><td>58.83.0</td><td>93.51.3</td><td>75.55.7</td><td>79.93.3</td><td>53.10.8</td><td>64.75.3</td><td>72.2</td></tr><tr><td>PROCA</td><td>90.81.7</td><td>47.62.5</td><td>87.91.5</td><td>77.94.8</td><td>95.10.5</td><td>79.85.4</td><td>90.02.2</td><td>56.73.1</td><td>55.36.4</td><td>75.7</td></tr><tr><td rowspan="3">4-shot</td><td>GPT</td><td>88.64.3</td><td>44.73.3</td><td>84.48.2</td><td>58.26.3</td><td>89.410.0</td><td>72.16.5</td><td>80.513.2</td><td>55.66.7</td><td>38.15.4</td><td>68.0</td></tr><tr><td>ConCa</td><td>92.93.7</td><td>47.74.4</td><td>87.81.8</td><td>66.511.7</td><td>93.41.0</td><td>76.44.0</td><td>88.63.0</td><td>54.71.5</td><td>48.54.9</td><td>72.9</td></tr><tr><td>PROCA</td><td>95.00.4</td><td>46.24.6</td><td>89.419</td><td>79.45.8</td><td>95.80.8</td><td>79.96.6</td><td>91.92.6</td><td>61.22.7</td><td>57.15.3</td><td>77.3</td></tr><tr><td rowspan="3">8-shot</td><td>GPT</td><td>91.16.2</td><td>44.92.9</td><td>89.52.3</td><td>82.13.9</td><td>95.21.7</td><td>76.99.7</td><td>87.73.1</td><td>61.03.9</td><td>44.45.6</td><td>74.8</td></tr><tr><td>ConCa</td><td>93.41.8</td><td>46.64.4</td><td>90.10.5</td><td>80.55.8</td><td>96.20.3</td><td>79.96.4</td><td>90.82.0</td><td>59.64.8</td><td>53.57.9</td><td>76.7</td></tr><tr><td>PROCA</td><td>94.41.0</td><td>47.44.4</td><td>90.70.7</td><td>83.64.2</td><td>96.10.5</td><td>84.21.8</td><td>95.10.5</td><td>61.77.2</td><td>61.07.6</td><td>79.4</td></tr><tr><td colspan="12">Bloom 176B</td></tr><tr><td rowspan="3">0-shot</td><td>Bloom</td><td>73.40.0</td><td>26.00.0</td><td>71.00.0</td><td>53.30.0</td><td>60.10.0</td><td>27.10.0</td><td>48.50.0</td><td>62.50.0</td><td>59.00.0</td><td>53.4</td></tr><tr><td>ConCa</td><td>73.90.0</td><td>25.30.0</td><td>71.80.0</td><td>49.00.0</td><td>51.10.0</td><td>38.20.0</td><td>61.00.0</td><td>53.80.0</td><td>41.00.0</td><td>51.7</td></tr><tr><td>PROCA</td><td>76.40.1</td><td>31.80.2</td><td>73.40.4</td><td>61.30.3</td><td>80.40.8</td><td>60.13.5</td><td>75.80.1</td><td>62.60.2</td><td>52.90.5</td><td>63.9</td></tr><tr><td rowspan="3">1-shot</td><td>Bloom</td><td>91.72.6</td><td>31.17.5</td><td>84.62.3</td><td>60.48.5</td><td>96.10.1</td><td>67.60.9</td><td>81.82.0</td><td>61.23.4</td><td>55.17.1</td><td>70.0</td></tr><tr><td>ConCa</td><td>91.81.6</td><td>38.94.3</td><td>86.81.6</td><td>51.22.5</td><td>96.10.4</td><td>78.40.5</td><td>80.41.9</td><td>54.05.6</td><td>69.31.3</td><td>71.9</td></tr><tr><td>PROCA</td><td>93.60.6</td><td>47.52.8</td><td>88.08.8</td><td>72.01.8</td><td>95.70.4</td><td>81.60.7</td><td>83.71.8</td><td>65.70.4</td><td>67.52.5</td><td>77.3</td></tr><tr><td rowspan="3">4-shot</td><td>Bloom</td><td>96.30.1</td><td>46.70.8</td><td>87.35.3</td><td>72.26.4</td><td>94.22.5</td><td>68.83.2</td><td>86.21.4</td><td>64.12.4</td><td>29.10.9</td><td>71.7</td></tr><tr><td>ConCa</td><td>96.00.1</td><td>46.92.9</td><td>89.71.1</td><td>70.47.7</td><td>94.21.9</td><td>78.00.1</td><td>86.62.4</td><td>56.30.7</td><td>64.87.6</td><td>75.9</td></tr><tr><td>PROCA</td><td>95.70.2</td><td>50.22.6</td><td>91.20.1</td><td>78.50.5</td><td>95.80.5</td><td>82.71.2</td><td>87.01.3</td><td>68.60.4</td><td>56.84.8</td><td>78.5</td></tr><tr><td rowspan="3">8-shot</td><td>Bloom</td><td>94.62.0</td><td>43.23.5</td><td>90.90.8</td><td>78.62.2</td><td>96.09.9</td><td>75.41.9</td><td>88.42.1</td><td>65.92.4</td><td>48.96.7</td><td>75.8</td></tr><tr><td>ConCa</td><td>96.10.2</td><td>42.25.5</td><td>91.00.9</td><td>75.81.7</td><td>95.90.4</td><td>81.92.0</td><td>89.52.6</td><td>59.00.5</td><td>73.91.1</td><td>78.4</td></tr><tr><td>PROCA</td><td>95.31.3</td><td>53.11.6</td><td>92.00.6</td><td>80.61.9</td><td>95.60.8</td><td>82.12.0</td><td>85.13.7</td><td>69.52.7</td><td>68.67.8</td><td>80.2</td></tr></table>

PROCA is consistently effective across different templates. We conduct the experiments across nine different prompts templates and label spaces (details of templates are shown in Table 8 of Appendix). The performance comparison among three approaches on SST-2 is shown in Figure 4. We observe that contextual calibration remains high variance although it improves the average accuracy. However, our proposed prototypical calibration can bring a large improvement with low variance, which indicates that PROCA is effective on various prompt templates.

![](images/37bc26ba5430abcef1d8b8d108b6040a44ad71572106930ca86d20ed758af5ed.jpg)  
Figure 4: Performance comparison across nine different templates.

![](images/94207ed9a3f3dba35dfacdeb970aec28ab1463d4b6303ea8d0c32946cf4dcf00.jpg)  
Figure 5: The impact of class-imbalanced estimate set on PROCA's performance.

![](images/a7819c07260503adcf57882a9720e9a1036741bc39cd82a63b7cd8c9df96aabf.jpg)  
Figure 6: Performance comparison under different label proportions and permutations of demonstrations. Each box indicates the accuracy of twelve randomly sampled permutations.

PROCA is robust under demonstration perturbations. Previous works (Zhao et al., 2021; Lu et al., 2021) have noticed that the order of training examples has significant effects on the performance of few-shot demonstrations. In this part, we evaluate our prototypical calibration conditioned on nine 8-shot prompts with different class proportions for SST-2, and show the accuracy of twelve randomly sampled orderings for each proportion in Figure 6. We find that the original GPT-2-XL with some specific compositions and permutations achieves high performance while some others are close to random guesses similar to the findings of Lu et al. (2021). Contextual calibration can improve the performance in most cases but is still sensitive to the orderings. However, PROCA is significantly superior to the others and keeps an extremely low variance across different permutations, indicating the non-sensitiveness to the class proportion and permutation.

It is also shown that although the class-balanced prompts tend to have higher performance, there are some exceptions that the prompt with all negative samples is the most performant one for both the conventional approach and contextual calibration. We think that it is GPT-2-XL's intrinsic bias to the positive class that leads to the counter-intuitive results.

PROCA is robust to class imbalance. Due to the unavailability of the labels of the estimate examples, PROCA may suffer the problem of class imbalance. We construct nine estimate sets with different imbalance levels for SST-2 and Subj by controlling the proportion of positive examples in the sampled set. Then we evaluate PROCA and contextual calibration on them. The experimental results in Figure 5 show that the estimate set's class imbalance level affects the performance of PROCA to some extent and the class-balanced estimate set can lead to higher accuracy. As described in Section 3.1, standard GMM estimates the weight of each cluster, which reflects the proportion of different classes in the estimate set. Owing to our "weights-cutting" operation, the problem of class imbalance has much less negative impact on PROCA, which even with an extremely class-imbalanced estimate set surpasses contextual calibration on both SST-2 and Subj. Besides, the absolute advancement for the class-balanced estimate set can reach  $20\%$  and  $15\%$  respectively.

Table 2: 4- and 8-shot performance comparison of different estimate set construction methods for GPT-neo across nine text classification tasks. ProCA-g and ProCA-t represent ProCA based on the unlabeled estimate set generated by LM and randomly sampled from the training set, respectively.  

<table><tr><td>Shot</td><td>Method</td><td>SST-2</td><td>SST-5</td><td>MR</td><td>Subj</td><td>AP</td><td>AGNews</td><td>DBpedia</td><td>RTE</td><td>TREC</td></tr><tr><td rowspan="4">4-shot</td><td>GPT-neo</td><td>84.58.7</td><td>33.28.0</td><td>68.713.4</td><td>61.914.8</td><td>85.810.3</td><td>68.25.7</td><td>71.810.9</td><td>47.90.8</td><td>37.26.4</td></tr><tr><td>ConCa</td><td>91.71.1</td><td>41.35.0</td><td>81.07.2</td><td>62.611.5</td><td>93.40.7</td><td>60.48.9</td><td>86.73.5</td><td>52.93.6</td><td>57.38.5</td></tr><tr><td>PROCA-g</td><td>91.91.4</td><td>43.53.4</td><td>83.94.3</td><td>73.46.9</td><td>91.60.8</td><td>70.75.4</td><td>80.12.9</td><td>50.41.6</td><td>57.32.5</td></tr><tr><td>PROCA-t</td><td>91.61.8</td><td>38.75.6</td><td>85.61.2</td><td>79.72.9</td><td>93.30.5</td><td>75.65.6</td><td>90.43.7</td><td>55.01.0</td><td>59.22.9</td></tr><tr><td rowspan="4">8-shot</td><td>GPT-neo</td><td>68.019.2</td><td>31.36.9</td><td>70.214.4</td><td>57.58.1</td><td>90.42.7</td><td>66.58.0</td><td>78.86.3</td><td>49.22.6</td><td>50.85.6</td></tr><tr><td>ConCa</td><td>81.29.1</td><td>33.94.7</td><td>77.89.6</td><td>71.05.7</td><td>93.60.9</td><td>73.43.5</td><td>90.31.0</td><td>51.36.6</td><td>56.08.0</td></tr><tr><td>PROCA-g</td><td>90.12.4</td><td>39.64.1</td><td>81.14.5</td><td>75.53.8</td><td>92.01.4</td><td>77.34.5</td><td>81.82.4</td><td>52.42.7</td><td>63.45.2</td></tr><tr><td>PROCA-t</td><td>91.91.2</td><td>39.44.0</td><td>77.813.9</td><td>81.33.8</td><td>93.90.7</td><td>78.92.5</td><td>92.01.5</td><td>56.81.8</td><td>56.03.6</td></tr></table>

# 4.5 ABLATION STUDIES

Comparison between different estimate set construction methods. There are two ways to construct the estimate set. One is using light unlabeled examples from the training set, which is simple and convenient. The other is utilizing the generation ability of LMs to construct unlabeled dataset (Lu et al., 2021; Wang et al., 2021; Meng et al., 2022; Ye et al., 2022). For the generation method, we follow Lu et al. (2021) to generate diverse estimate examples based on the various permutations of demonstrations. Specifically, we only use two labeled examples per category as demonstrations for generation except for DBPedia(one labeled example per category) and all these labeled examples are not involved in the evaluation. The size of generated estimate set maintains consistent with the estimate set sampled from the training set and the LM used for generation is the same as that in the evaluation. The 4-shot and 8-shot experimental results are shown in Table 2 and the 0-shot and 1-shot results are shown in Table 5 of Appendix. We observe that PROCA greatly outperforms the original LM whether using unlabeled data generated by LM or randomly sampled from the training set, which validates the effectiveness of PROCA under a restricted setting. It also shows that PROCA-t performs slightly better than PROCA-g and we speculate that it is due to the lower quality of the unlabeled data generated by LM.

A relatively small-scale estimate set is sufficient for ProCA. In Figure 7, we evaluate ProCA with ten different estimate set sizes across five datasets. We report the average accuracy over five randomly sampled estimate sets, conditioning on the same 4-shot prompt in each setting. We observe that increasing the scale of the estimate set within a certain small range can greatly improve the classification accuracy and reduce the variance. However, a larger estimate set can hardly bring further improvement, which indicates that a small estimate set can support ProCA to be optimal. It also shows that ProCA has acceptable performances with just several estimate examples on SST-2, MR, and Subj, which even surpasses both the conventional approach and contextual calibration.

![](images/0b6a812944d38b9a3b7fc1a0d4ab9316feb64eb82cbe82b7b6686dae42c2db55.jpg)  
Figure 7: Performance of ProCA across different estimate set sizes.

# Estimation selection according to assignment

score is useful to ProCA. The standard estimation of GMM aims to maximize the likelihood of all observations and select the optimal estimated parameters among multiple repetitions. We argue that the estimation with maximum likelihood is not consistently beneficial to ProCA especially in multi-classes tasks, because there is no supervision to force the predictions to be assigned to their inclined classes during the estimation procedure. We propose to determine estimation according to the assignment score (as described in Section 3.3), which can theoretically select the most performant estimation for ProCA. We compare ProCA with the two strategies for GPT-2-XL and GPT-J respectively. The experimental results are shown in Table 3. It indicates that our determination strategy can achieve more stable improvements on AGNnews and DBPedia regardless of model size.

Table 3: Performance of ProCA with different strategies of estimation selection (maximum likelihood, and assignment score as in Equation (4)) for GPT-2-XL and GPT-J on AGNews and DBPedia.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Strategy</td><td colspan="2">0-shot</td><td colspan="2">1-shot</td><td colspan="2">4-shot</td><td colspan="2">8-shot</td></tr><tr><td>GPT-2</td><td>GPT-J</td><td>GPT-2</td><td>GPT-J</td><td>GPT-2</td><td>GPT-J</td><td>GPT-2</td><td>GPT-J</td></tr><tr><td rowspan="2">AGNews</td><td>max-likelihood</td><td>64.01.1</td><td>55.20.3</td><td>65.85.2</td><td>79.25.8</td><td>60.111.1</td><td>79.76.7</td><td>71.78.7</td><td>78.111.7</td></tr><tr><td>assignment score</td><td>64.61.4</td><td>55.10.4</td><td>69.84.3</td><td>79.85.4</td><td>67.413.5</td><td>79.96.6</td><td>75.53.2</td><td>84.21.8</td></tr><tr><td rowspan="2">DBPeida</td><td>max-likelihood</td><td>63.87.5</td><td>59.24.6</td><td>71.87.3</td><td>77.73.8</td><td>76.15.5</td><td>82.23.9</td><td>79.35.0</td><td>83.82.4</td></tr><tr><td>assignment score</td><td>73.63.0</td><td>66.11.5</td><td>79.93.8</td><td>90.02.2</td><td>87.24.9</td><td>91.92.6</td><td>89.40.7</td><td>95.10.5</td></tr></table>

# 5 RELATED WORK

Understanding In-context Learning with Language Models. Xie et al. (2022) provide a theoretical perspective to understand in-context learning and cast it as implicit Bayesian inference, where LM can infer latent concepts across demonstrations. Shin et al. (2022) observe that corpora sources play a crucial role in the emergence of in-context learning. Razeghi et al. (2022) also show that low-order co-occurrence statistics in pretraining corpora can significantly impact the few-shot performance. Other works analyze the role of natural language instructions and demonstrations in prompts, which are considered crucial for the success of in-context learning. Liu et al. (2021) construct more semantically-similar demonstrations during evaluation. Min et al. (2022) demonstrate that such success is mainly attributed into three factors, i.e., label space, input distribution and the prompt format.

Instability of Few-shot Learning with Language Models. It has been recognized that the few-shot performance of language models is unstable under different in-context scenarios. Language models are prone to predict some specific labels due to the intrinsic bias or demonstration permutations (Zhao et al., 2021; Lu et al., 2021). Lu et al. (2021) demonstrate LM's sensitiveness to the order of few-shot demonstrations, and introduced an Entropy-based metric to select the most performant prompts. Zhao et al. (2021) attribute the instability to three biases of prompts, including majority bias, recency bias and common token bias, and proposed a contextual calibration approach. However, the selected content free test inputs can not precisely reflect the bias of models and lead to the problem of over-correction or under-correction. On the contrary, we adaptively provide the classification criterion according to the text inputs' overall prediction distribution, and completely calibrate the bias introduced by models and prompts.

# 6 CONCLUSION AND LIMITATION

To our analysis, decision boundary is of critical importance to the performance of few-shot demonstrations and the traditional decision boundary leads to the fragility of prompting LMs. We propose prototypical calibration to adaptively learn a more robust decision boundary. Experiments show that the calibrated decision boundary is effective across various prompt templates, class proportions and permutations. We achieve on average a  $13\%$  absolute improvement across different sizes of pretrained language models on nine popular text classification tasks.

A limitation of our method is that it is not applicable for tasks whose label space is open-ended since a fixed label space is necessary for estimating prototypical clusters. Furthermore, our method is designed for in-context learning on individual downstream tasks, it fails to calibrate the inherent bias of language models like gender and occupation bias. For future work, we would like to extend our method to the tasks with open-ended answer space, such as generative question-answering and text summarization tasks.

# 7 REPRODUCIBILITY

We developed our code based on the open-source work Conctextul Calibration<sup>1</sup> and the code was attached in the supplementary material. Moreover, we used the same hyperparameters for all of our experiments, which we specify in the section 4.1 and Appendix A.

# REFERENCES

Hangbo Bao, Li Dong, Furu Wei, Wenhui Wang, Nan Yang, Xiaodong Liu, Yu Wang, Songhao Piao, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. UniLMv2: Pseudo-masked language models for unified language model pre-training. In ICML 2020, volume 119 of Proceedings of Machine Learning Research, pp. 642-652. PMLR, 2020.  
BigScience. Bigscience language open-science open-access multilingual (bloom) language model. Technical report, May 2022.  
Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In NeurIPS 2020, 2020.  
Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In Machine Learning Challenges Workshop, pp. 177-190. Springer, 2005.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. In NeurIPS 2019, pp. 13042-13054, 2019.  
Zhengbao Jiang, Frank F Xu, Jun Araki, and Graham Neubig. How can we know what language models know? Transactions of the Association for Computational Linguistics, 8:423-438, 2020.  
Harold W Kuhn. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83-97, 1955.  
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.  
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804, 2021.  
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. arXiv preprint arXiv:2104.08786, 2021.  
Yu Meng, Jiaxin Huang, Yu Zhang, and Jiawei Han. Generating training data with language models: Towards zero-shot language understanding. ArXiv, 2022.  
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.  
T.K. Moon. The expectation-maximization algorithm. IEEE Signal Processing Magazine, 13(6): 47-60, 1996. doi: 10.1109/79.543975.

Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. arXiv preprint cs/0409058, 2004.  
Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. arXiv preprint cs/0506075, 2005.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.  
Yasaman Razeghi, Robert L. Logan IV, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning. CoRR, abs/2202.07206, 2022.  
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. arXiv preprint arXiv:2112.08633, 2021.  
Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, HyoungSeok Kim, Boseop Kim, Kyunghyun Cho, Gichang Lee, Woo-Myoung Park, Jung-Woo Ha, and Nako Sung. On the effect of pretraining corpora on in-context learning by a large-scale language model. CoRR, abs/2204.13509, 2022. doi: 10.48550/arXiv.2204.13509.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631-1642, 2013.  
Ellen M Voorhees and Dawn M Tice. Building a question answering test collection. In Proceedings of the 23rd annual international ACM SIGIR conference on Research and development in information retrieval, pp. 200-207, 2000.  
Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax, May 2021.  
Zirui Wang, Adams Wei Yu, Orhan First, and Yuan Cao. Towards zero-label language learning. ArXiv, abs/2109.09193, 2021.  
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. In International Conference on Learning Representations, 2022.  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems, 32, 2019.  
Jiacheng Ye, Jiahui Gao, Qintong Li, Hang Xu, Jiangtao Feng, Zhiyong Wu, Tao Yu, and Lingpeng Kong. Zerogen: Efficient zero-shot learning via dataset generation. ArXiv, 2022.  
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015.  
Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In International Conference on Machine Learning, pp. 12697-12706. PMLR, 2021.
