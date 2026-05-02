# ANOMALY TRANSFORMER: TIME SERIES ANOMALY DETECTION WITH ASSOCIATION DISCREPANCY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Unsupervisedly detecting anomaly points in time series is challenging, which requires the model to learn informative representations and derive a distinguishable criterion. Prior methods mainly detect anomalies based on the recurrent network representation of each time point. However, the point-wise representation is less informative for complex temporal patterns and can be dominated by normal patterns, making rare anomalies less distinguishable. We find that in each time series, each time point can also be described by its associations with all time points, presenting as a point-wise distribution that is more expressive for temporal modeling. We further observe that due to the rarity of anomalies, it is harder for anomalies to build strong associations with the whole series and their associations shall mainly concentrate on the adjacent time points. This observation implies an inherently distinguishable criterion between normal and abnormal points, which we highlight as the Association Discrepancy. Technically we propose the Anomaly Transformer with an Anomaly-Attention mechanism to compute the association discrepancy. A minimax strategy is devised to amplify the normal-abnormal distinguishability of the association discrepancy. Anomaly Transformer achieves state-of-the-art performance on six unsupervised time series anomaly detection benchmarks for three applications: service monitoring, space & earth exploration, and water treatment.

# 1 INTRODUCTION

Real-world systems always work in a continuous way, which can generate several successive measurements monitored by multi-sensors, such as industrial equipment, space probe, etc. Discovering the malfunctions from large-scale system monitoring can be reduced to detecting the abnormal time points from time series, which is quite meaningful for ensuring security and avoiding financial loss. But anomalies are usually rare and can be hidden by vast normal points, making the data labeling hard and expensive. Thus, we focus on time series anomaly detection under the unsupervised setting.

Unsupervised time series anomaly detection is extremely challenging in practice. The model not only should learn informative representations from complex temporal dynamics through the unsupervised tasks, but it also should derive a criterion that is distinguishable to detect the rare anomalies from plenty of normal time points. Various classic anomaly detection methods have provided many unsupervised paradigms, such as the density-estimation methods proposed in local outlier factor (LOF, Breunig et al. (2000)), clustering-based methods presented in one-class SVM (OC-SVM, Scholkopf et al. (2001)) and SVDD (Tax & Duin, 2004). But these classic methods do not consider the temporal information and are difficult to generalize to unseen real-world scenarios. Benefiting from the great representation learning capability of neural networks, recent deep models (Su et al., 2019b; Shen et al., 2020; Li et al., 2021) have made remarkable advances. They mainly focus on learning temporal representations through well-designed recurrent networks and self-supervised by the reconstruction task, in which the most practical anomaly criterion is reconstruction error per time point based on the learned representations. However, due to the rarity of anomalies, the point-wise representation is less informative for complex temporal patterns and can be dominated by normal time points, making anomalies less distinguishable. Also, the reconstruction error is calculated point-wisely, which cannot provide a comprehensive description of the temporal context.

From a new perspective, we find that in each time series, each time point can also be represented by its associations with all the time points, presenting as a distribution of association weights along the

horizon. The association distribution of each time point can provide a more informative description for the temporal context, indicating dynamic patterns, such as the period or trend. This association distribution is referred to as the series-association, which can be discovered from the raw series.

Further, we observe that due to the rarity of anomalies and the dominance of normal patterns, it is harder for anomalies to build strong associations with the whole series. The associations of anomalies shall concentrate on the adjacent time points that are more likely to contain similar abnormal patterns due to the continuity. Such an adjacent-concentration inductive bias is referred to as the prior-association. In contrast, the dominating normal time points can discover informative associations with the whole series, not limiting to the adjacent area. Based on this observation, we try to utilize the inherent normal-abnormal distinguishability of the association distribution. This leads to a new anomaly criterion for each time point, quantified by the distance between each time point's prior-association and its series-association, named as Association Discrepancy. As aforementioned, because the associations of anomalies are more likely to be adjacent-concentrating, anomalies will present a smaller association discrepancy than normal time points.

Taking the advantage of the great model capability of Transformers (Vaswani et al., 2017; Devlin et al., 2019; Brown et al., 2020), we introduce them to the unsupervised time series anomaly detection and propose the Anomaly Transformer for association learning. To compute the Association Discrepancy, we renovate the self-attention mechanism to the Anomaly-Attention, which contains a two-branch structure to model the prior-association and series-association of each time point respectively. The prior-association employs a learnable Gaussian distribution to present the adjacent-concentration inductive bias of each time point, while the series-association corresponds to the self-attention weights learned from raw series. Besides, a minimax strategy is applied between the two branches, which can amplify the normal-abnormal distinguishability of the Association Discrepancy and further derive a new association-based criterion. Anomaly Transformer achieves strong results on six benchmarks, covering three real applications. The contributions are summarized as follows:

- Based on the key observation of Association Discrepancy, we propose the Anomaly Transformer with an Anomaly-Attention mechanism, which can model the prior-association and series-association simultaneously to embody the Association Discrepancy.  
- We propose a minimax strategy to amplify the normal-abnormal distinguishability of the Association Discrepancy and further derive a new association-based detection criterion.  
- Anomaly Transformer achieves the state-of-the-art anomaly detection results on six benchmarks for three real applications. Extensive ablations and insightful case studies are given.

# 2 RELATED WORK

# 2.1 UNSUPERVISED TIME SERIES ANOMALY DETECTION

As a vital real-world problem, unsupervised time series anomaly detection has been widely explored. Categorizing by anomaly determination criterion, the paradigms roughly include the density-estimation, clustering-based and reconstruction-based methods.

As for the density-estimation methods, the classic methods local outlier factor (LOF, Breunig et al. (2000)) and connectivity outlier factor (COF, Tang et al. (2002)) calculates the local density and local connectivity as the metrics for outlier determination respectively. DAGMM from Zong et al. (2018) integrates the deep Autoencoder (AE) with a Gaussian Mixture Model (GMM), which can get latent representations from AE and estimate the density of the representations using GMM.

In clustering-based methods, SVDD (Tax & Duin, 2004) and Deep SVDD (Ruff et al., 2018) try to gather the representations from normal data to a compact cluster, in which the anomaly score of an instance is formalized as the distance to cluster center. THOC (Shen et al., 2020) fuses the multiscale temporal features from intermediate layers together by a hierarchical clustering mechanism and determines the anomalies by the weighted sum of distances to the cluster centers of each layer.

The reconstruction-based models attempt to detect the anomalies by the reconstruction error. Park et al. (2018) presented the LSTM-VAE model that employed the LSTM backbone for temporal modeling and the Variational AutoEncoder (VAE) for reconstruction. OmniAnomaly proposed by Su et al. (2019b) further extends the LSTM-VAE model with a normalizing flow and uses the recon

struction probabilities for detection. InterFusion from Li et al. (2021) renovates the backbone to a hierarchical VAE to model the inter- and intra- dependency among multiple series simultaneously. GANs (Goodfellow et al., 2014) are also used for reconstruction-based anomaly detection (Schlegl et al., 2019; Li et al., 2019a; Zhou et al., 2019) and perform as an adversarial regularization.

This paper is characterized by a new association-based anomaly detection criterion, which is embodied by a co-design of the temporal models for learning more informative time-point associations.

# 2.2 TRANSFORMERS FOR TIME SERIES ANALYSIS

Recently, Transformers (Vaswani et al., 2017) have shown great power in sequential data processing, such as natural language processing (Devlin et al., 2019; Brown et al., 2020), audio processing (Huang et al., 2019) and computer vision (Dosovitskiy et al., 2021; Liu et al., 2021). For time series analysis, benefiting from the advantage of the self-attention mechanism, Transformers are used to discover the reliable long-range temporal dependencies (Kitaev et al., 2020; Li et al., 2019b; Zhou et al., 2021; Wu et al., 2021). Especially for time series anomaly detection, GTA proposed by Chen et al. (2021) employs the graph structure to learn the relationship among multiple IoT sensors, as well as the Transformer for temporal modeling and the reconstruction criterion for anomaly detection. Unlike the previous usage of Transformers, Anomaly Transformer renovates the self-attention mechanism to the Anomaly-Attention based on the key observation of association discrepancy and detects the anomalies based on our proposed association-based criterion.

# 3 METHOD

Suppose monitoring a successive system of  $d$  measurements and recording the equally spaced observations over time. The observed time series  $\mathcal{X}$  is denoted by a set of time points  $\{x_{1}, x_{2}, \dots, x_{N}\}$ , where  $x_{t} \in \mathbb{R}^{d}$  represents the observation of time  $t$ . The unsupervised time series anomaly detection problem is to determine whether  $x_{t}$  is anomalous or not without labels.

As aforementioned, we highlight the key to unsupervised time series anomaly detection as learning informative representations and finding distinguishable criterion. We propose the Anomaly Transformer to discover more informative associations and tackle this problem by learning the Association Discrepancy, which is inherently normal-abnormal distinguishable. Technically, we propose the Anomaly-Attention to embody the prior-association and series-associations, along with a minimax optimization strategy to obtain a more distinguishable association discrepancy. Co-designed with the architecture, we derive an association-based criterion based on the learned association discrepancy.

# 3.1 ANOMALY TRANSFORMER

Given the limitation of Transformers (Vaswani et al., 2017) for anomaly detection, we renovate the vanilla architecture to the Anomaly Transformer (Figure 1) with an Anomaly-Attention mechanism.

Overall Architecture Anomaly Transformer is characterized by stacking the Anomaly-Attention blocks and feed-forward layers alternately. This stacking structure is conducive to learning underlying associations from deep multi-level features. Suppose the model contains  $L$  layers with length- $N$  input time series  $\mathcal{X} \in \mathbb{R}^{N \times d}$ . The overall equations of the  $l$ -th layer are formalized as:

$$
\mathcal {Z} ^ {l} = \text {L a y e r - N o r m} \left(\text {A n o m a l y - A t t e n t i o n} \left(\mathcal {X} ^ {l - 1}\right) + \mathcal {X} ^ {l - 1}\right)
$$

$$
\mathcal {X} ^ {l} = \text {L a y e r - N o r m} \left(\operatorname {F e e d - F o r w a r d} \left(\mathcal {Z} ^ {l}\right) + \mathcal {Z} ^ {l}\right), \tag {1}
$$

where  $\mathcal{X}^l\in \mathbb{R}^{N\times d_{\mathrm{model}}}$ ,  $l\in \{1,\dots ,L\}$  denotes the output of the  $l$ -th layer with  $d_{\mathrm{model}}$  channels. The initial input  $\mathcal{X}^0 = \operatorname {Embedding}(\mathcal{X})$  represents the embedded raw series.  $\mathcal{Z}^l\in \mathbb{R}^{N\times d_{\mathrm{model}}}$  is the  $l$ -th layer's hidden representation. Anomaly-Attention  $(\cdot)$  is to compute the association discrepancy.

Anomaly-Attention Note that the single-branch self-attention mechanism (Vaswani et al., 2017) cannot model the prior-association and series-association simultaneously. We propose the Anomaly-Attention with a two-branch structure (Figure 1). For the prior-association, we adopt a learnable Gaussian distribution, centered at the corresponding position index. Benefiting from the unimodal

![](images/e89e8f2ba892393b38f037f92585240143cd7d497660e3fda09bc131b361928e.jpg)  
Figure 1: Anomaly Transformer architecture. Anomaly-Attention (left) models the prior-association and series-association simultaneously. In addition to the reconstruction loss, our model is also optimized by the minimax strategy with a specially-designed stop-gradient mechanism (gray arrows) to constrain the prior- and series- associations for more distinguishable association discrepancy.

property of the Gaussian family, this design can pay more attention to the adjacent horizon constitutionally. We also use a learnable variance parameter  $\sigma$  for the Gaussian prior, making the prior-associations adapt to the various time series patterns, such as different lengths of anomaly segments. The series-association branch is to learn the associations from raw series, which can find the most effective associations adaptively. Note that these two forms maintain the temporal dependencies of each time point, which are more informative than point-wise representation. They also reflect the adjacent-concentration prior and the learned real associations respectively, whose discrepancy shall be normal-abnormal distinguishable. The Anomaly-Attention in the  $l$ -th layer is formalized as:

$$
\text {I n i t i a l i z a t i o n :} \mathcal {Q}, \mathcal {K}, \mathcal {V}, \sigma = W ^ {l} * \mathcal {X} ^ {l - 1}
$$

$$
\text {P r i o r - A s s o c i a t i o n :} \mathcal {P} ^ {l} = \operatorname {S c a l e} \left(\left[ \frac {1}{\sqrt {2 \pi} \sigma_ {i}} \exp \left(- \frac {\left| j - i \right| ^ {2}}{2 \sigma_ {i} ^ {2}}\right) \right] _ {i, j \in \{1, \dots , N \}}\right) \tag {2}
$$

$$
\text {S e r i e s - A s s o c i a t i o n :} \mathcal {S} ^ {l} = \operatorname {S o f t m a x} \left(\frac {\mathcal {Q K} ^ {\mathrm {T}}}{\sqrt {d _ {\mathrm {m o d e l}}}}\right)
$$

$$
\begin{array}{l} \text {R e c o n s t r u c t i o n :} \widehat {\mathcal {Z}} ^ {l} = \mathcal {S} ^ {l} \mathcal {V}, \end{array}
$$

where  $\mathcal{Q},\mathcal{K},\mathcal{V}\in \mathbb{R}^{N\times d_{\mathrm{model}}}$  represent the query, key and value respectively in the self-attention, and  $S^l\in \mathbb{R}^{N\times N}$  denotes the series-association. Prior-association  $\mathcal{P}^l\in \mathbb{R}^{N\times N}$  is generated based on the learned variance parameter  $\sigma \in \mathbb{R}^{N\times 1}$  and  $\sigma_{i}$  corresponds to the  $i$  -th time point. Concretely, for the  $i$  -th time point, its association weight for the  $j$  -th point is calculated from Gaussian distribution  $\mathcal{N}(i,\sigma_i^2)$  with respect to the relative distance  $|j - i|$ . Scale  $(\cdot)$  is to transform the association weights to the discrete distributions  $\mathcal{P}^l$  by dividing the row sum.  $W^l$  represents the linear projector of the  $l$  -th layer.  $\widehat{\mathcal{Z}}^l\in \mathbb{R}^{N\times d_{\mathrm{model}}}$  is the hidden representation after the Anomaly-Attention in the  $l$  -th layer. We use Anomaly-Attention  $(\cdot)$  to summarize Equation 2. See Appendix B for pseudo code.

In the multi-head version that we use, the learned variance is  $\sigma \in \mathbb{R}^{N\times h}$  for  $h$  heads.  $\mathcal{Q}_m,\mathcal{K}_m,\mathcal{V}_m\in$ $\mathbb{R}^{N\times \frac{d_{\mathrm{model}}}{h}}$  denote the query, key and value of the  $m$  -th head respectively. The block concatenates the outputs  $\{\widehat{\mathcal{Z}}_m^l\in \mathbb{R}^{N\times \frac{d_{\mathrm{model}}}{h}}\}_{1\leq m\leq h}$  from multiple heads and gets the final result  $\widehat{\mathcal{Z}}^l\in \mathbb{R}^{N\times d_{\mathrm{model}}}$

Association Discrepancy We formalize the Association Discrepancy as the symmetrized KL divergence between prior- and series- associations, which represents the information gain between these two distributions (Neal, 2007). We average the association discrepancy from multiple layers to combine the associations from multi-level features into a more informative measure as:

$$
\operatorname {A s s D i s} (\mathcal {P}, \mathcal {S}; \mathcal {X}) = \frac {1}{L} \sum_ {l = 1} ^ {L} \left(\mathrm {K L} \left(\mathcal {P} ^ {l} \| \mathcal {S} ^ {l}\right) + \mathrm {K L} \left(\mathcal {S} ^ {l} \| \mathcal {P} ^ {l}\right)\right), \tag {3}
$$

where  $\operatorname{AssDis}(\mathcal{P},\mathcal{S};\mathcal{X})\in \mathbb{R}^{N\times 1}$  means the point-wise association discrepancy of  $\mathcal{X}$  with respect to prior-association  $\mathcal{P}$  and series-association  $\mathcal{S}$  from multiple layers. The  $i$ -th element of results cor

![](images/bdc76c072b4505b21e372ba713a6b1c0a36df1cc4134761a406be7965c1ba58e.jpg)  
Figure 2: Minimax association learning. At the minimize phase, the prior-association minimizes the Association Discrepancy within the Gaussian family. At the maximize phase, the series-association maximizes the Association Discrepancy constrained by the reconstruction loss.

responds to the  $i$ -th time point of  $\mathcal{X}$ . From previous observation, abnormal time points will present smaller AssDis  $(\mathcal{P},\mathcal{S};\mathcal{X})$  than normal time points, which makes AssDis inherently distinguishable.

# 3.2 MINIMAX ASSOCIATION LEARNING

As an unsupervised task, we employ the reconstruction loss for optimizing our model. The reconstruction loss will guide the series-association to find the most informative associations, such as the adjacent time points of anomalies. To further amplify the difference between normal and abnormal time points, we also use an additional loss to enlarge the association discrepancy. Due to the unimodal property of the prior-association, the discrepancy loss will guide the series-association to pay more attention to the non-adjacent area, which makes the reconstruction of anomalies harder and makes anomalies more identifiable. The loss function for input series  $\mathcal{X} \in \mathbb{R}^{N \times d}$  is formalized as:

$$
\mathcal {L} _ {\text {T o t a l}} (\widehat {\mathcal {X}}, \mathcal {P}, \mathcal {S}, \lambda ; \mathcal {X}) = \| \mathcal {X} - \widehat {\mathcal {X}} \| _ {2} ^ {2} - \lambda \left(\operatorname {A s s D i s} (\mathcal {P}, \mathcal {S}; \mathcal {X})\right), \tag {4}
$$

where  $\widehat{\mathcal{X}}\in \mathbb{R}^{N\times d}$  denotes the reconstruction of  $\mathcal{X}$  and  $\| \cdot \| _2$  means the L2-norm.  $\lambda$  is to trade-off these two loss terms. When  $\lambda >0$ , the optimization target is to enlarge the association discrepancy. We propose a new minimax strategy to make the association discrepancy more distinguishable.

Minimax Strategy Note that directly maximizing the association discrepancy will extremely reduce the variance of the Gaussian prior (Neal, 2007), making the prior-association meaningless. Towards a better control of association learning, we propose a minimax strategy (Figure 2). Concretely, for the minimize phase, we drive the prior-association  $\mathcal{P}^l$  to approximate the series-association  $S^l$  that is learned from raw series. This process will make the prior-association adapt to various temporal patterns. For the maximize phase, we optimize the series-association to enlarge the association discrepancy. This process forces the series-association to pay more attention to the non-adjacent horizon. Thus, integrating the reconstruction loss, the total loss functions of these two phases are:

Minimize Phase:  $\mathcal{L}_{\mathrm{Total}}(\widehat{\mathcal{X}},\mathcal{P},S_{\mathrm{detach}}, - \lambda ;\mathcal{X})$

Maximize Phase:  $\mathcal{L}_{\mathrm{Total}}(\widehat{\mathcal{X}},\mathcal{P}_{\mathrm{detach}},\mathcal{S},\lambda ;\mathcal{X})$

where  $\lambda > 0$  and  $^*\mathrm{detach}$  means to stop the gradient backpropagation of the association (Figure 1). As  $\mathcal{P}$  approximates  $S_{\mathrm{detach}}$  in the minimize phase, the maximize phase will conduct a stronger constraint to the series-association, forcing the time points to pay more attention to the non-adjacent area. Under the reconstruction loss, this is much harder for anomalies to achieve than normal time points, thereby amplifying the normal-abnormal distinguishability of the association discrepancy.

Association-based Anomaly Criterion We incorporate the normalized association discrepancy to the reconstruction criterion, which will take the benefits of both temporal representation and the distinguishable association discrepancy. The final anomaly score of  $\mathcal{X} \in \mathbb{R}^{N \times d}$  is shown as follows:

$$
\operatorname {A n o m a l y S c o r e} (\mathcal {X}) = \operatorname {S o f t m a x} \left(- \operatorname {A s s D i s} (\mathcal {P}, \mathcal {S}; \mathcal {X})\right) \times \| \mathcal {X} - \widehat {\mathcal {X}} \| _ {2} ^ {2}, \tag {6}
$$

where  $\mathrm{AnomalyScore}(\mathcal{X})\in \mathbb{R}^{N\times 1}$  denotes the point-wise anomaly criterion of  $\mathcal{X}$ . The anomalies need to pay more attention to adjacent time points for a better reconstruction, which will make the association discrepancy decrease and derive a higher anomaly score. Thus, this design can make the reconstruction error and the association discrepancy collaborate to improve detection performance.

# 4 EXPERIMENTS

We extensively evaluate Anomaly Transformer on six benchmarks for three practical applications.

Datasets Here is a description of the six experiment datasets: (1) SMD (Server Machine Dataset, Su et al. (2019b)) is a 5-week-long dataset that is collected from a large Internet company with 38 dimensions. (2) PSM (Pooled Server Metrics, Abdulaal et al. (2021)) is collected internally from multiple application server nodes at eBay with 26 dimensions. (3) Both MSL (Mars Science Laboratory rover) and SMAP (Soil Moisture Active Passive satellite) are public datasets from NASA (Su et al., 2019b) with 55 and 25 dimensions respectively, which contain the telemetry anomaly data derived from the Incident Surprise Anomaly (ISA) reports of spacecraft monitoring systems. (4) SWaT (Secure Water Treatment, Mathur & Tippenhauer (2016)) is obtained from 51 sensors of the critical infrastructure system under continuous operations. (5) NeurIPS-TS (NeurIPS 2021 Time Series Benchmark) is a dataset proposed by Lai et al. (2021) and includes five time series anomaly scenarios categorized by behavior-driven taxonomy as point-global, pattern-contextual, pattern-shapelet, pattern-seasonal and pattern-trend. Each dataset includes training, validation and testing subsets. Anomalies are only labeled in the testing subset. The statistical details are summarized in Table 1.

Table 1: Details of benchmarks. AR represents the truth abnormal proportion of the whole dataset.  

<table><tr><td>Benchmarks</td><td>Applications</td><td>Dimension</td><td>Window</td><td>#Training</td><td>#Validation</td><td>#Test</td><td>AR (Truth)</td></tr><tr><td>SMD</td><td>Server</td><td>38</td><td>100</td><td>566,724</td><td>141,681</td><td>708,420</td><td>0.042</td></tr><tr><td>PSM</td><td>Server</td><td>25</td><td>100</td><td>105,984</td><td>26,497</td><td>87,841</td><td>0.278</td></tr><tr><td>MSL</td><td>Space</td><td>55</td><td>100</td><td>46,653</td><td>11,664</td><td>73,729</td><td>0.105</td></tr><tr><td>SMAP</td><td>Space</td><td>25</td><td>100</td><td>108,146</td><td>27,037</td><td>427,617</td><td>0.128</td></tr><tr><td>SWaT</td><td>Water</td><td>51</td><td>100</td><td>396,000</td><td>99,000</td><td>449,919</td><td>0.121</td></tr><tr><td>NeurIPS-TS</td><td>Various Anomalies</td><td>1</td><td>100</td><td>20,000</td><td>10,000</td><td>20,000</td><td>0.018</td></tr></table>

Implementation details Following the well-established protocol in Shen et al. (2020), we adopt a non-overlapped sliding window to obtain a set of sub-series. The sliding window is with a fixed size of 100 for all datasets as shown in Table 1. We label the time points as anomalies if their anomaly scores (Equation 6) are larger than a certain threshold  $\delta$ . The threshold  $\delta$  is determined to make  $r$  proportion data of the validation dataset labeled as anomalies. For the main results, we set  $r = 0.5\%$  for SMD,  $0.1\%$  for SWaT and  $1\%$  for other datasets. We adopt the widely-used adjustment strategy (Xu et al., 2018; Su et al., 2019a; Shen et al., 2020): if a time point in a certain successive abnormal segment is detected, all anomalies in this abnormal segment are viewed to be correctly detected. This strategy is justified from the observation that an abnormal time point will cause an alert and further make the whole segment noticed in real-world applications. Experimentally, Anomaly Transformer contains 3 layers. We set the channel number of hidden states  $d_{\mathrm{model}}$  as 512 and the number of heads  $h$  as 8. The hyperparameter  $\lambda$  (Equation 4) is set as 3 for all datasets to trade-off two parts of the loss function. We use the ADAM (Kingma & Ba, 2015) optimizer with an initial learning rate of  $10^{-4}$ . The training process is early stopped within 10 epochs with the batch size of 32. All the experiments are implemented in Pytorch (Paszke et al., 2019) and conducted on a single NVIDIA TITAN RTX 24GB GPUs. We provide the analysis of hyper-parameter sensitivity in Appendix A.

Baselines We extensively compare our model with 10 baselines, including the reconstruction-based models: InterFusion (2021), BeatGAN (2019), OmniAnomaly (2019b), LSTM-VAE (2018); the density-estimation models: LOF (2000), DAGMM (2018); the clustering-based methods: DeepSVDD (2018), THOC (2020), classic methods: OC-SVM (2004) and IsolationForest (2008). InterFusion (2021) and THOC (2020) are the state-of-the-art deep models.

# 4.1 MAIN RESULTS

Real-world datasets We extensively evaluate our model on five real-world datasets with ten competitive baselines. As shown in Table 2, Anomaly Transformer achieves the consistent state-of-the-art on all benchmarks. We observe that deep models generally beat the classic statistic models, benefiting from the powerful non-linear modeling capability of the neural network. Also, deep models that consider the temporal information outperform the general anomaly detection model, such as

![](images/9b26bb41b313dc4e4b49a4e0144a9116185ae750eecddaf8899495cc2874fca0.jpg)  
Figure 3: ROC curves (horizontal-axis: false-positive rate; vertical-axis: true-positive rate) for five corresponding datasets. A higher AUC value (area under the ROC curve) indicates a better performance. The predefined threshold proportion  $r$  is in  $\{0.5\%, 1.0\%, 1.5\%, 2.0\%, 10\%, 20\%, 30\% \}$ .

Table 2: Quantitative results for Anomaly Transformer (Ours) in five real-world datasets. The  $P$ ,  $R$  and  $F1$  represent the precision, recall and F1-score (as %) respectively. F1-score is the harmonic mean of precision and recall. For these three metrics, a higher value indicates a better performance.  

<table><tr><td>Dataset</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td></tr><tr><td>Metric</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>OCSVM</td><td>44.34</td><td>76.72</td><td>56.19</td><td>59.78</td><td>86.87</td><td>70.82</td><td>53.85</td><td>59.07</td><td>56.34</td><td>45.39</td><td>49.22</td><td>47.23</td><td>62.75</td><td>80.89</td><td>70.67</td></tr><tr><td>IsolationForest</td><td>42.31</td><td>73.29</td><td>53.64</td><td>53.94</td><td>86.54</td><td>66.45</td><td>52.39</td><td>59.07</td><td>55.53</td><td>49.29</td><td>44.95</td><td>47.02</td><td>76.09</td><td>92.45</td><td>83.48</td></tr><tr><td>LOF</td><td>56.34</td><td>39.86</td><td>46.68</td><td>47.72</td><td>85.25</td><td>61.18</td><td>58.93</td><td>56.33</td><td>57.60</td><td>72.15</td><td>65.43</td><td>68.62</td><td>57.89</td><td>90.49</td><td>70.61</td></tr><tr><td>Deep-SVDD</td><td>78.54</td><td>79.67</td><td>79.10</td><td>91.92</td><td>76.63</td><td>83.58</td><td>89.93</td><td>56.02</td><td>69.04</td><td>80.42</td><td>84.45</td><td>82.39</td><td>95.41</td><td>86.49</td><td>90.73</td></tr><tr><td>DAGMM</td><td>67.30</td><td>49.89</td><td>57.30</td><td>89.60</td><td>63.93</td><td>74.62</td><td>86.45</td><td>56.73</td><td>68.51</td><td>89.92</td><td>57.84</td><td>70.40</td><td>93.49</td><td>70.03</td><td>80.08</td></tr><tr><td>LSTM-VAE</td><td>75.76</td><td>90.08</td><td>82.30</td><td>85.49</td><td>79.94</td><td>82.62</td><td>92.20</td><td>67.75</td><td>78.10</td><td>76.00</td><td>89.50</td><td>82.20</td><td>73.62</td><td>89.92</td><td>80.96</td></tr><tr><td>BeatGAN</td><td>72.90</td><td>84.09</td><td>78.10</td><td>89.75</td><td>85.42</td><td>87.53</td><td>92.38</td><td>55.85</td><td>69.61</td><td>64.01</td><td>87.46</td><td>73.92</td><td>90.30</td><td>93.84</td><td>92.04</td></tr><tr><td>OmniAnomaly</td><td>83.68</td><td>86.82</td><td>85.22</td><td>89.02</td><td>86.37</td><td>87.67</td><td>92.49</td><td>81.99</td><td>86.92</td><td>81.42</td><td>84.30</td><td>82.83</td><td>88.39</td><td>74.46</td><td>80.83</td></tr><tr><td>InterFusion</td><td>87.02</td><td>85.43</td><td>86.22</td><td>81.28</td><td>92.70</td><td>86.62</td><td>89.77</td><td>88.52</td><td>89.14</td><td>80.59</td><td>85.58</td><td>83.01</td><td>83.61</td><td>83.45</td><td>83.52</td></tr><tr><td>THOC</td><td>79.76</td><td>90.95</td><td>84.99</td><td>88.45</td><td>90.97</td><td>89.69</td><td>92.06</td><td>89.34</td><td>90.68</td><td>83.94</td><td>86.36</td><td>85.13</td><td>88.14</td><td>90.99</td><td>89.54</td></tr><tr><td>Ours</td><td>89.40</td><td>95.45</td><td>92.33</td><td>92.09</td><td>95.15</td><td>93.59</td><td>94.13</td><td>99.40</td><td>96.69</td><td>91.55</td><td>96.73</td><td>94.07</td><td>96.91</td><td>98.90</td><td>97.89</td></tr></table>

Deep-SVDD (Ruff et al., 2018) and DAGMM (Zong et al., 2018), which verifies the effectiveness of temporal modeling. Our proposed Anomaly Transformer goes beyond the point-wise representation learned by RNNs and models the more informative associations. The results in Table 2 are persuasive for the advantage of association learning in time series anomaly detection. In addition, we plot the ROC curve in Figure 3 for a complete comparison. Anomaly Transformer has the highest AUC values for all five datasets, which means that our model is more distinguishable and robust under various pre-selected thresholds. See Appendix C for showcases.

NeurIPS-TS benchmark This benchmark is generated from well-designed rules proposed by Lai et al. (2021), which completely includes all types of anomalies, covering both the point-wise and pattern-wise anomalies. As shown in Figure 4, Anomaly Transformer can still achieve state-of-the-art performance on various anomalies, which means that our model is robust to all types of anomalies. We provide some showcases in Appendix C.

![](images/231cf0c1e140aa718b5ba353a57303050fe530d60cec9248937b2dcd1757c17b.jpg)  
Figure 4: Results for the NeurIPS-TS.

Ablation study As shown in Table 3, we further investigate the effect of each part in our model. Our association-based criterion outperforms the widely-used reconstruc

tion criterion consistently. Specifically, the association-based criterion brings a remarkable  $18.76\%$ $(76.20\rightarrow 94.96)$  averaged absolute F1-score promotion. Also, directly taking the association discrepancy as the criterion still achieves a good performance (F1-score:  $91.55\%$  ) and surpasses the previous state-of-the-art model THOC (F1-score:  $88.01\%$  calculated from Table 2). Besides, the learnable prior-association (corresponding to  $\sigma$  in Equation 2) and the minimax strategy can further improve our model and get  $8.43\%$ $(79.05\rightarrow 87.48)$  and  $7.48\%$ $(87.48\rightarrow 94.96)$  averaged absolute promotions respectively. Finally, our proposed Anomaly Transformer surpasses the pure Transformer

Table 3: Ablation results (F1-score) in anomaly criterion, prior-association and optimization strategy. Recon, AssDis and Assoc mean the pure reconstruction performance, pure association discrepancy and our proposed association-based criterion respectively. Fix is to fix Learnable variance parameter  $\sigma$  of prior-association as 1.0. Max and Minimax ref to the strategies for association discrepancy in the maximization (Equation 4) and minimax (Equation 5) way respectively.  

<table><tr><td>Architecture</td><td>Anomaly Criterion</td><td>Prior-Association</td><td>Optimization Strategy</td><td>SMD</td><td>MSL</td><td>SMAP</td><td>SWaT</td><td>PSM</td><td>Avg F1 (as %)</td></tr><tr><td>Transformer</td><td>Recon</td><td>×</td><td>×</td><td>79.72</td><td>76.64</td><td>73.74</td><td>74.56</td><td>78.43</td><td>76.62</td></tr><tr><td rowspan="4">Anomaly Transformer</td><td>Recon</td><td>Learnable</td><td>Minmax</td><td>71.35</td><td>78.61</td><td>69.12</td><td>81.53</td><td>80.40</td><td>76.20</td></tr><tr><td>AssDis</td><td>Learnable</td><td>Minmax</td><td>87.57</td><td>90.50</td><td>90.98</td><td>93.21</td><td>95.47</td><td>91.55</td></tr><tr><td>Assoc</td><td>Fix</td><td>Max</td><td>83.95</td><td>82.17</td><td>70.65</td><td>79.46</td><td>79.04</td><td>79.05</td></tr><tr><td>Assoc</td><td>Learnable</td><td>Max</td><td>88.88</td><td>85.20</td><td>87.84</td><td>81.65</td><td>93.83</td><td>87.48</td></tr><tr><td>*final</td><td>Assoc</td><td>Learnable</td><td>Minmax</td><td>92.33</td><td>93.59</td><td>96.90</td><td>94.07</td><td>97.89</td><td>94.96</td></tr></table>

by  $18.34\%$ $(76.62\rightarrow 94.96)$  absolute improvement. These verify that each module of our design is effective and necessary. More ablations of association discrepancy can be found in Appendix D.

# 4.2 MODEL ANALYSIS

To explain how our model works intuitively, we provide the visualization and statistical results for our three key designs: anomaly criterion, learnable prior-association and optimization strategy.

![](images/c8ec750318940a34091654e044f0a1ecdbf8ee1f9d43efdac9559df914cc89a7.jpg)  
Figure 5: Visualization of different anomaly categories (Lai et al., 2021). We plot the raw series (first row) from NeurIPS-TS dataset, as well as their corresponding reconstruction (second row) and association-based criteria (third row). The point-wise anomalies are marked by red circles and the pattern-wise anomalies are in red segments. The wrongly detected cases are bounded by red boxes.

Anomaly criterion visualization To get more intuitive cases about how association-based criterion works, we provide some visualization in Figure 5 and further explore the criterion performance under different types of anomalies, where the taxonomy is from Lai et al. (2021). We can find that our proposed association-based criterion is more distinguishable in general. Concretely, the association-based criterion can obtain the consistent smaller values for the normal part, which is quite contrasting in point-contextual and pattern-seasonal cases (Figure 5). In contrast, the jitter

curves of the reconstruction criterion make the detection process confused and fail in the aforementioned two cases. This visualization verifies that our proposed criterion can highlight the anomalies and provide distinct values for normal and abnormal points, making the detection precise and robust.

![](images/8e0cb201a95e3437484ed3009fb61ef4ea286deab84fbe49aeb168dc0ddf4b36.jpg)

![](images/576d8d00732b2f978c2810d3a6e19e8a6e775a88019f3a87ce20f73645cfef44.jpg)  
(a) Point-Global  
Figure 6: Learned variance parameter  $\sigma$  for different types of anomalies (highlight in red).

![](images/5d1eced2c35a737ee02f08261efc3c529dd968076858bd9172a73a10ba82d258.jpg)

![](images/8bb6f47b1c8aae7851d544d799f0e77a31b7951e8cc3b3ef9b220569d876ca55.jpg)  
(b) Point-Contextual

![](images/a78651896245c27d2b9640afc9ca1eac3234bfd74926ed85b79c870b1d6f03a6.jpg)

![](images/26c70fda0c2d323563d489e716b483d5edf15b22da879932c8940cbf70f42d3a.jpg)  
(c) Pattern-Shapelet

![](images/332d4d4153da69bf350eb81f2ef8cc29ded11a68359464d78e8360799314a408.jpg)

![](images/ac5f006e90de00ee46de0e6dd4f09c115faf8810fa5c39664a976b36414cc3ca.jpg)  
(d) Pattern-Seasonal

![](images/b61461c0caa579d2ecefcd7cdcdd3b03002728e6f3ef7c778a50025c8d932f65.jpg)

![](images/c76f3716fe6201d123fd8097b740f939f2e84b4d60a002fa28ea592a1d0d595b.jpg)  
(e) Pattern-Trend

Prior-association visualization We find that the learned  $\sigma$  changes to adapt to various data patterns of time series (Figure 6). Especially, the prior-association of anomalies generally has a smaller  $\sigma$  than normal time points, which matches our adjacent-concentration inductive bias of anomalies.

**Optimization strategy analysis** Only with the reconstruction loss, the abnormal and normal time points present similar performance in the association weights to adjacent time points, corresponding to a contrast value closed to 1 (Table 4). Maximizing the association discrepancy will force the series-associations to pay more attention to the non-adjacent area. However, to obtain a better reconstruction, the anomalies have to maintain much larger adjacent association weights than normal time points, corresponding to a larger contrast value. But direct maximization will cause the optimization problem of Gaussian prior and cannot strongly amplify the difference between normal and abnormal time points as expected (SMD:  $1.15 \rightarrow 1.27$ ). The minimax strategy optimizes the prior-association to provide a stronger constraint to series-association. Thus, the minimax strategy obtains more distinguishable contrast values than direct maximization (SMD:  $1.27 \rightarrow 2.39$ ) and thereby performs better.

Table 4: The statistical results of adjacent association weights for Abnormal and Normal time points respectively. Recon, Max and Minimax represent the association learning process that is supervised by reconstruction loss, direct maximization and minimax strategy respectively. A higher contrast value ( $\frac{\text{Abnormal}}{\text{Normal}}$ ) indicates a stronger distinguishability between normal and abnormal time points.  

<table><tr><td rowspan="2">Dataset Optimization</td><td colspan="3">SMD</td><td colspan="3">MSL</td><td colspan="3">SMAP</td><td colspan="3">SWaT</td><td colspan="3">PSM</td></tr><tr><td colspan="3">Recon Max Ours</td><td colspan="3">Recon Max Ours</td><td colspan="3">Recon Max Ours</td><td colspan="3">Recon Max Ours</td><td colspan="3">Recon Max Ours</td></tr><tr><td>Abnormal (%)</td><td>1.08</td><td>0.95</td><td>0.86</td><td>1.01</td><td>0.65</td><td>0.35</td><td>1.29</td><td>1.18</td><td>0.70</td><td>1.27</td><td>0.89</td><td>0.37</td><td>1.02</td><td>0.56</td><td>0.29</td></tr><tr><td>Normal (%)</td><td>0.94</td><td>0.75</td><td>0.36</td><td>1.00</td><td>0.59</td><td>0.22</td><td>1.23</td><td>1.09</td><td>0.49</td><td>1.18</td><td>0.78</td><td>0.21</td><td>0.99</td><td>0.54</td><td>0.11</td></tr><tr><td>Contrast ( Abnormal / Normal)</td><td>1.15</td><td>1.27</td><td>2.39</td><td>1.01</td><td>1.10</td><td>1.59</td><td>1.05</td><td>1.08</td><td>1.43</td><td>1.08</td><td>1.14</td><td>1.76</td><td>1.03</td><td>1.04</td><td>2.64</td></tr></table>

# 5 CONCLUSION

This paper studies the unsupervised time series anomaly detection problem. Unlike previous methods, we try to tackle this problem with more informative association learning. Based on the key observation of association discrepancy, we propose the Anomaly Transformer, including an Anomaly-Attention with the two-branch structure to embody the association discrepancy. A minimax strategy is adopted to further amplify the difference between normal and abnormal time points. By introducing the association discrepancy, we propose the association-based criterion, which makes the reconstruction performance and association discrepancy collaborate. Anomaly Transformer achieves the state-of-the-art on extensive benchmarks. Comprehensive ablations and insightful analyses are included to verify the effectiveness of our design and elaborate on how the model works.

# REFERENCES

Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. Practical approach to asynchronous multivariate time series anomaly detection and localization. International Conference on Knowledge Discovery & Data Mining, 2021.  
Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. Lof: identifying density-based local outliers. In Proceedings of the ACM SIGMOD International Conference on Management of Data, 2000.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Neural Information Processing Systems, 2020.  
Zekai Chen, Dingshuo Chen, Zixuan Yuan, Xiuzhen Cheng, and Xiao Zhang. Learning graph structures with transformer for multivariate time series anomaly detection in iot. ArXiv, abs/2104.03466, 2021.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, 2019.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.  
I. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In Neural Information Processing Systems, 2014.  
Cheng-Zhi Anna Huang, Ashish Vaswani, Jakob Uszkoreit, Ian Simon, Curtis Hawthorne, Noam Shazeer, Andrew M. Dai, Matthew D. Hoffman, Monica Dinculescu, and Douglas Eck. Music transformer. In International Conference on Learning Representations, 2019.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.  
Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020.  
Kwei-Herng Lai, D. Zha, Junjie Xu, and Yue Zhao. Revisiting time series outlier detection: Definitions and benchmarks. In NeurIPS Dataset and Benchmark Track, 2021.  
Dan Li, Dacheng Chen, Lei Shi, Baihong Jin, Jonathan Goh, and See-Kiong Ng. Mad-gan: Multivariate anomaly detection for time series data with generative adversarial networks. In ICANN, 2019a.  
Shiyang Li, Xiaoyong Jin, Yao Xuan, Xiyou Zhou, Wenhu Chen, Yu-Xiang Wang, and Xifeng Yan. Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting. In Neural Information Processing Systems, 2019b.  
Zhihan Li, Youjian Zhao, Jiaqi Han, Ya Su, Rui Jiao, Xidao Wen, and Dan Pei. Multivariate time series anomaly detection and interpretation using hierarchical inter-metric and temporal embedding. International Conference on Knowledge Discovery & Data Mining, 2021.  
F. Liu, K. Ting, and Z. Zhou. Isolation forest. International Conference on Data Mining, 2008.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Ching-Feng Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. ArXiv, abs/2103.14030, 2021.

Aditya P. Mathur and Nils Ole Tippenhauer. Swat: a water treatment testbed for research and training on ICS security. In International Workshop on Cyber-physical Systems for Smart Water Networks, 2016.  
Radford M. Neal. Pattern recognition and machine learning. Technometrics, 2007.  
Daehyung Park, Yuuna Hoshi, and Charles C. Kemp. A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder. IEEE Robotics and Automation Letters, 2018.  
Adam Paszke, S. Gross, Francisco Massa, A. Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Z. Lin, N. Gimelshein, L. Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Neural Information Processing Systems, 2019.  
Lukas Ruff, Nico Gornitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Robert A. Vandermeulen, Alexander Binder, Emmanuel Müller, and M. Kloft. Deep one-class classification. In International Conference on Machine Learning, 2018.  
T. Schlegl, Philipp Seebock, S. Waldstein, G. Langs, and U. Schmidt-Erfurth. f-anogan: Fast unsupervised anomaly detection with generative adversarial networks. Medical Image Analysis, 2019.  
B. Scholkopf, John C. Platt, J. Shawe-Taylor, Alex Smola, and R. C. Williamson. Estimating the support of a high-dimensional distribution. Neural Computation, 2001.  
Lifeng Shen, Zhuocong Li, and James T. Kwok. Timeseries anomaly detection using temporal hierarchical one-class network. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, MariaFlorina Balcan, and Hsuan-Tien Lin (eds.), Neural Information Processing Systems, 2020.  
Ya Su, Y. Zhao, Chenhao Niu, Rong Liu, W. Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. International Conference on Knowledge Discovery & Data Mining, 2019a.  
Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Ankur Teredesai, Vipin Kumar, Ying Li, Rómer Rosales, Evimaria Terzi, and George Karypis (eds.), International Conference on Knowledge Discovery & Data Mining, 2019b.  
Jian Tang, Zhixiang Chen, A. Fu, and D. Cheung. Enhancing effectiveness of outlier detections for low density patterns. In Pacific-Asia Conference on Knowledge Discovery & Data Mining, 2002.  
D. Tax and R. Duin. Support vector data description. Machine Learning, 2004.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems, 2017.  
Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. ArXiv, abs/2106.13008, 2021.  
Haowen Xu, Wenxiao Chen, N. Zhao, Zeyan Li, Jiahao Bu, Zhihan Li, Y. Liu, Y. Zhao, Dan Pei, Yang Feng, Jian Jhen Chen, Zhaogang Wang, and Honglin Qiao. Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications. Proceedings of the World Wide Web Conference, 2018.  
Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. Beatgan: Anomalous rhythm detection using adversarially generated time series. In International Joint Conference on Artificial Intelligence, 2019.  
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI Conference on Artificial Intelligence, 2021.

Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Dae-ki Cho, and Haifeng Chen. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In International Conference on Learning Representations, 2018.
