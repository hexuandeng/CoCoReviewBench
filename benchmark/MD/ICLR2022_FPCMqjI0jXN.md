# DOMINO: DISCOVERING SYSTEMATIC ERRORS WITH CROSS-MODAL EMBEDDINGS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Machine learning models that achieve high overall accuracy often make systematic errors on important subgroups (or slices) of data. When working with high-dimensional inputs (e.g. images, audio) where important slices are often unlabeled, identifying underperforming slices is a challenging task. In order to address this issue, recent studies have proposed automated slice discovery methods (SDMs), which leverage learned model representations to mine input data for slices on which a model performs poorly. To be useful to a practitioner, these methods must identify slices that are both underperforming and coherent (i.e. united by a human-understandable concept). However, no quantitative evaluation framework currently exists for rigorously assessing SDMs with respect to these criteria. Additionally, prior qualitative evaluations have shown that SDMs often identify slices that are incoherent. In this work, we address these challenges by first designing a principled evaluation framework that enables a quantitative comparison of SDMs across 1,235 slice discovery settings in three input domains (natural images, medical images, and time-series data). Then, motivated by the recent development of powerful cross-modal representation learning approaches, we present Domino, an SDM that leverages cross-modal embeddings and a novel error-aware mixture model to discover and describe coherent slices. We find that Domino accurately identifies  $36\%$  of the 1,235 slices in our framework - a 12 percentage-point improvement over prior methods. Further, Domino is the first SDM that can generate natural language descriptions of identified slices, correctly outputting the exact name of the slice in  $35\%$  of settings.

# 1 INTRODUCTION

Machine learning models often make systematic errors on important subsets (or slices) of data. For instance, models trained to detect collapsed lungs in chest x-rays have been shown to make predictions based on the presence of chest drains, a device typically used during treatment (Oakden-Rayner et al., 2019). As a result, these models frequently make prediction errors on cases without chest drains, a critical data slice where false negative predictions could be life-threatening. Similar slice performance gaps have been observed in radiograph classification (Badgeley et al., 2019a; Zech et al., 2018a; DeGrave et al., 2021a), melanoma detection (Winkler et al., 2019a), natural language processing (Orr et al., 2020; Goel et al., 2021), and object detection (de Vries et al., 2019b), among others. If underperforming slices can be accurately identified and labeled, we can then improve model robustness by either addressing biases in the training dataset or using robust optimization techniques (Zhang et al., 2018; Sagawa et al., 2020).

However, identifying underperforming slices is difficult in practice. When working with high-dimensional inputs (e.g. images, time-series data, video) where individual features (e.g. pixels) have little semantic meaning, slices are often "hidden", meaning that they cannot easily be extracted from the inputs and are not captured in metadata (Oakden-Rayner et al., 2019). In this setting, we must perform slice discovery: the task of mining unstructured data for semantically meaningful subgroups on which the model performs poorly.

In modern machine learning workflows, practitioners commonly perform slice discovery with a combination of feature-based interpretability methods (e.g. GradCAM, LIME) and manual inspection (Selvaraju et al., 2017; Ribeiro et al., 2016). However, these approaches are time-consuming

![](images/88bb6e14aca213b65c0fa46a6eec60adadaa94587d39b5b9259d572453ffa249.jpg)  
Figure 1: The Domino Slice Discovery Method. Domino discovers and describes underperforming slices. (Left) In this example, a model is trained and evaluated on the task of detecting birds in images. The dataset shows that a correlation exists between the presence of birds and the presence of the sky. (Right) As a result, the classifier makes false positive predictions on skies without birds. Domino uses cross-modal embeddings to identify and describe the error slice.

![](images/383acf7ce1a29a75f54009afe5c7884df8223d67cc996df6654f2529b7ce0c32.jpg)

and susceptible to confirmation bias (Adebayo et al., 2018). As a result, recent works have proposed automated slice discovery methods (SDMs), which utilize learned input representations to identify semantically meaningful slices where the model makes prediction errors (d'Eon et al., 2021; Yeh et al., 2020; Sohoni et al., 2020; Kim et al., 2018). An ideal SDM should automatically identify data slices that fulfill two desiderata: (a) slices should contain examples on which the model underperforms, or has a high error rate and (b) slices should contain examples that are coherent, or align closely with a human-understandable concept. An SDM that is able to reliably satisfy these desiderata across a wide range of settings has yet to be demonstrated for two reasons:

Issue 1: No quantitative evaluation framework exists for measuring performance of SDMs with respect to these desiderata. Existing SDM evaluations are either qualitative (d'Eon et al., 2021), performed on purely synthetic data (Yeh et al., 2020), or consider only a small selection of tasks and slices (Sohoni et al., 2020). A comprehensive evaluation framework should be quantitative, use realistic data, cover a broad range of contexts, and evaluate both underperformance and coherence. Currently, no datasets or frameworks exist to support such an evaluation, making it difficult to evaluate the tradeoffs among prior SDMs.

Issue 2: Prior qualitative evaluations have demonstrated that existing SDMs often identify slices that are incoherent. A practically useful SDM should discover coherent slices that are understandable by a domain expert. For example, in the chest x-ray setting described earlier, the slice "patients without chest drains" is meaningful to a physician. Slice coherence has previously been evaluated qualitatively by requiring users to manually inspect examples and identify common attributes (d'Eon et al., 2021; Yeh et al., 2020). Such evaluations have shown that discovered slices often do not align with concepts understandable to a domain expert. Additionally, even if slices do align well with concepts, it may be difficult for humans to identify the shared concept. Thus, an ideal SDM would not only output coherent slices, but also identify the concept connecting examples in each slice.

In this work, we address both of these issues by (1) developing a framework to quantitatively evaluate the effectiveness of slice discovery methods at scale and (2) leveraging this framework to demonstrate that a powerful class of recently-developed cross-modal embeddings can be used to create an SDM that satisfies the above desiderata. Our approach – Domino – identifies coherent slices and generates automated slice descriptions.

After formally describing the slice discovery problem in Section 2, we introduce a benchmark evaluation framework for rigorously assessing SDM performance in Section 3. We curate a set of over 1235 slice discovery settings, each consisting of a real-world classification dataset, a trained model, and one or more "ground truth" slices corresponding to a meaningful concept in the domain. During evaluation, the SDM is provided with the dataset and the model, and we measure if the labeled slices can be successfully identified. We find that existing methods identify "ground truth" slices in no more than  $23\%$  of these settings.

Motivated by the recent development of large cross-modal representation learning approaches (e.g. CLIP) that embed inputs and text in the same latent representation space, in Section 4 we proceed to

design Domino, a novel SDM that uses cross-modal embeddings to identify coherent slices. Cross-modal representations incorporate semantic meaning from text into input embeddings, which we demonstrate can improve slice coherence and enable the generation of automated slice descriptions. Domino embeds inputs alongside natural language with cross-modal representations, identifies coherent slices with an error-aware Gaussian mixture model, and generates natural language descriptions for discovered slices. In Section 5, we use our evaluation framework to show that Domino identifies  $36\%$  of the "ground truth" coherent slices across three input domains (natural images, medical images, and time-series) – a 12 percentage-point improvement over existing methods.

# 2 SLICE DISCOVERY PRELIMINARIES

We consider a standard classification setting with input  $X \in \mathcal{X}$  (e.g. an image, time-series, or graph) and label  $Y \in \mathcal{Y} = \{1, 2, \dots, C\}$  over  $C$  classes. Additionally, we assume there exists a set of  $k$  slices  $\mathbf{S} = \{S^{(j)}\}_{j=1}^k \in \{0, 1\}^k$  that partition the data into coherent (potentially overlapping) subgroups, where each subgroup captures a distinct concept or attribute that would be familiar to a domain expert. The slices, inputs, and labels vary jointly according to a probability distribution  $P(X, Y, \mathbf{S})$  over  $\mathcal{X} \times \mathcal{Y} \times \{0, 1\}^k$ . We assume that training, validation and test data are drawn i.i.d. from this distribution. A model  $h_\theta : \mathcal{X} \to \mathcal{Y}$  exhibits degraded performance with respect to a slice  $S^{(j)}$  and metric  $\ell : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$  if

$$
\mathbb {E} _ {X, Y \mid S ^ {(j)} = 1} [ \ell (h _ {\theta} (X), Y) ] <   \mathbb {E} _ {X, Y \mid S ^ {(j)} = 0} [ \ell (h _ {\theta} (X), Y) ]. \tag {1}
$$

Assuming that a trained classifier  $h_\theta : \mathcal{X} \to \mathcal{Y}$  exhibits degraded performance on each of the  $k$  slices in  $\mathbf{S}$ , we define the slice discovery problem as follows:

- Inputs: a trained classifier  $h_{\theta}$  and labeled dataset  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$  drawn from  $P(X, Y)$ .  
- **Output:** a set of  $\hat{k}$  slicing functions  $\Psi = \{\psi^{(j)}: \mathcal{X} \times \mathcal{Y} \to \{0,1\}\}_{j=1}^{\hat{k}}$  that partition the data into  $\hat{k}$  subgroups.

We consider an output to be successful if, for each ground truth slice  $S^{(u)}$ , a slicing function  $\psi^{(v)}$  predicts  $S^{(u)}$  with precision above some threshold  $\beta$ :

$$
\forall u \in [ k ]. \quad \exists v \in [ \hat {k} ]. \quad P (S ^ {(u)} | \psi^ {(v)} (X, Y) = 1) > \beta . \tag {2}
$$

A slice discovery method (SDM),  $M(\mathcal{D}, h_{\theta}) \to \Psi$ , aims to solve the slice discovery problem.

# 3 SLICE DISCOVERY EVALUATION FRAMEWORK

We propose a framework that measures the performance of each SDM with respect to the two desiderata outlined in Section 1: (a) the model  $h_{\theta}$  should exhibit degraded performance on slices predicted by the SDM and (b) slices predicted by the SDM should be coherent. Specifically, we frame this problem as an information retrieval task, where we label coherent and underperforming slices in datasets and measure the performance of SDMs at retrieving "ground truth" slices.

Our evaluations are fueled by large sets of slice discovery settings, each consisting of: (1) a labeled dataset  $\mathcal{D} = \{(x_i,y_i)\}_{i = 1}^n$ , (2) a model  $h_\theta$  trained on  $\mathcal{D}$ , and (3) slice annotations  $\{\mathbf{s}_i\}_{i = 1}^n$  for one or more coherent slices  $\mathbf{S}$  on which the model  $h_\theta$  exhibits degraded performance. (1) and (2) correspond to the inputs to the SDM, while (3) corresponds to the expected output. Using these slice discovery settings, we can quantitatively evaluate an SDM  $M(\mathcal{D},f_{\theta})\to \Psi$ . The procedure is detailed in Algorithm 1.

In the remainder of Section 3, we propose a process for generating representative slice discovery settings across a variety of different slice categories in Section 3.1 and describe the evaluation metrics  $L$  and datasets that we use to assess SDM performance in In Section 3.2.

# 3.1 GENERATING SLICE DISCOVERY SETTINGS

To obtain accurate estimates of SDM performance, the settings fueling our evaluation should ideally be both representative of real-world slice discovery settings and large in number. Unfortunately,

Algorithm 1 SDM Evaluation Process  
for  $(\mathcal{D},\mathbf{s},h_{\theta})\in$  settings do  $\Psi \leftarrow M(\mathcal{D}_{\mathrm{valid}},h_{\theta})\triangleright$  Fit the SDM on the validation set, yielding a set of slicing functions  $\Psi$  for  $j\in [\hat{k} ]$  do  $\hat{\mathbf{s}}^{(j)}\gets \psi^{(j)}(\mathcal{D}_{\mathrm{test}})$  Apply the slicing functions to the test set, yielding  $\hat{s}\in [0,1]^{n_{\mathrm{test}}}$  end for metrics  $\leftarrow \{\max_{j\in [\hat{k} ]}L(\mathbf{s}^{(i)},\hat{s}^{(j)})\}_{i = 1}^k$  Compute metric  $L$  comparing  $\hat{\mathbf{s}}$  and s end for

such slice discovery settings are usually unavailable. Many public machine learning datasets come with pretrained models, making it straightforward to source parts (1) and (2); however, relatively few datasets specify slices on which those models perform poorly, making part (3) difficult to obtain.

Here, we propose a repeatable process for programmatically generating a large number of realistic slice discovery settings. We begin with a base dataset  $\mathcal{D}_{\mathrm{base}}$  that has either a hierarchical label structure (e.g. ImageNet) or rich metadata accompanying each example (e.g.. CelebA). We select a target variable  $Y$  and slice variable S, each defined in terms of the metadata or class structure. This allows us to derive target and slice labels  $\{(y_i,\mathbf{s}_i)\}_{i = 1}^n$  directly from the dataset. In addition, because the slice S is defined in terms of meaningful annotations, we know that the slice is coherent. After selecting a target variable and slice variable, we (1) generate a dataset  $\mathcal{D}$  and (2) generate a model  $h_\theta$  that exhibits degraded performance with respect to S.

# 3.1.1 DATASET GENERATION

We categorize each slice discovery setting based on the underlying reason that the model  $h_{\theta}$  exhibits degraded performance on the slices S. We survey the literature for examples of underperforming slices in the wild, which we document in Section A.6. Based on our survey and prior work (Oakden-Rayner et al., 2019), we identify three popular slice types - rare slices, correlation slices, and noisy label slices. We provide descriptions in Section A.1, and describe how we generate them below:

Rare slice. To generate settings with rare slices, we construct  $\mathcal{D}$  such that for a given class label  $Y$ , elements in subclass  $C$  occur with proportion  $\alpha$ , where  $0.01 < \alpha < 0.1$ .

Correlation slice. To generate settings with correlation slices, we construct  $\mathcal{D}$  such that a linear correlation  $\alpha$  exists between the target variable and other class labels, where  $0.2 < \alpha < 0.8$ . (implementation details in Section A.2).

Noisy label slice. To generate settings with noisy labels, we construct  $\mathcal{D}$  such that for each class label  $Y$ , the elements in subclass  $C$  exhibit label noise with probability  $\alpha$ , where  $0.01 < \alpha < 0.3$ .

# 3.1.2 MODEL GENERATION

Each slice discovery setting includes a model  $h_{\theta}$  that exhibits degraded performance with respect to a set of slices S. We generate two classes of models: (a) trained models, which are trained on our generated datasets  $D$ , and (b) synthetic models, which are used to simulate model predictions.

Trained Models. We train a distinct model  $h_\theta$  across each of our generated datasets  $D$ . Our model  $h_\theta$  is valid if it exhibits a statistically significant degradation in performance with respect to the slices  $S$ ; in this case, the model  $h_\theta$ , dataset  $D$ , and slices  $S$  will comprise a valid slice discovery setting. It is important to note that real-world datasets often contain a large number of unlabeled slices, so trained models could identify underperforming, coherent slices that are not explicitly labeled as "ground truth" in our evaluation framework. This may complicate the interpretation of our results.

Synthetic Models. We also create settings with simulated model predictions  $\bar{h} :[0,1]^k\to [0,1]$ . This allows us to address the limitation highlighted above by explicitly controlling the presence of underperforming slices. We simulate model predictions by sampling from a beta distribution.

![](images/d5f90c829f6a85c030d384a20fa488f4c59bef5e7e6551fddc605b95b461e032.jpg)  
Figure 2: Evaluation Framework. We design a programmatic evaluation framework for rigorously evaluating SDMs across a wide range of settings, which is adaptable to any dataset and task.

# 3.2 EVALUATION APPROACH

We instantiate our evaluation framework by curating 1235 slice discovery settings across a number of different tasks, applications, and base datasets. Detailed statistics on our settings are provided in the Appendix (Table 1). We utilize the following three domains to generate slice discovery settings:

Natural Images (CelebA and ImageNet): We use two natural image datasets. The CelebFaces Attributes Dataset (CelebA) includes over 200k images of celebrities with 40 labeled attributes (Liu et al., 2015). ImageNet is a large image dataset with over 1.2 million images across 1000 labeled classes organized in a hierarchical structure via WordNet (Deng et al., 2009; Fellbaum, 1998).

Medical Images (MIMIC-CXR): The MIMIC Chest X-Ray (MIMIC-CXR) dataset includes 377,110 chest x-rays collected from the Beth Israel Deaconess Medical Center. Annotations indicate the presence or absence of fourteen conditions (Johnson et al., 2019; 2020).

Medical Time-Series Data (EEG): In addition to our image modalities, we also explore time-series data. We obtain a dataset of short 12 second electroencephalography (EEG) signals, which have been used in prior work for predicting the onset of seizures. Additional information on our dataset is included in (blinded citation).

We evaluate SDM performance with precision-at-k, which measures the proportion of the top  $k$  elements in the discovered slice that are in the ground truth slice. We use  $k = 10$  in this work.

# 4 DOMINO

In this section, we introduce Domino, an SDM that uses cross-modal embeddings to identify coherent slices and describe them in natural language using a three-step procedure (visualized in Fig. 1):

1. **Embed:** We encode the inputs  $\{x_{i}\}_{i = 1}^{n}$  in a cross-modal embedding space via a function  $\psi_{\mathrm{input}}: \mathcal{X} \to \mathbb{R}^d$ . We learn this embedding function  $\psi_{\mathrm{input}}$  jointly with an embedding function  $\psi_{\mathrm{text}}: \mathcal{T} \to \mathbb{R}^d$  that embeds text in the same space as the inputs.  
2. Slice: We identify underperforming regions in the cross-modal embedding space using an error-aware mixture model fit on the input embeddings  $\mathbf{Z}_{\mathrm{input}} := \{\mathbf{z}_i := \psi_{\mathrm{input}}(x_i)\}_{i=1}^n$ , model predictions  $\{\hat{y}_i := h_\theta(x_i)\}_{i=1}^n$ , and true class labels  $\{y_i\}_{i=1}^n$ . This yields  $\hat{k}$  slicing functions of the form  $\psi_{\mathrm{slice}}^{(j)} : \mathbb{R}^d \times \mathcal{V} \times \mathcal{V} \to \{0,1\}$ .  
3. Describe: Finally, we use the text embedding function  $\psi_{\mathrm{text}}$  learned in step (1) to generate a set of  $\hat{k}$  natural language descriptions of the discovered slices. Note that the ability to perform this step without human intervention is a particular advantage of our approach.

# 4.1 ENCODING INPUTS WITH CROSS-MODAL EMBEDDINGS

Given input-text pairs (such as images and captions), cross-modal embeddings can be generated using metric learning techniques, where the inputs and text are passed to separate encoders and the output representations are aligned based on the semantic similarity between the pair. Formally, given a set of inputs  $N \in \mathcal{X}$  and text descriptions  $T \in \mathcal{T}$  expressed in pairs as  $\{n_i, t_i\}_{i=1}^m$ , we learn two

embedding functions  $\psi_{\mathrm{input}}: \mathcal{X} \to \mathbb{R}^d$  and  $\psi_{\mathrm{text}}: \mathcal{T} \to \mathbb{R}^d$  such that the distances between pairs of embeddings  $dist(\psi_{\mathrm{input}}(n_i), \psi_{\mathrm{text}}(t_i))$  reflect the semantic similarity between  $n_i$  and  $t_i$ .

Ultimately, this joint training procedure enables the creation of semantically meaningful input embeddings that incorporate information from text. In this work, our key insight is that input representations generated from cross-modal learning techniques encode the semantic knowledge necessary for identifying coherent slices. Our method relies on the assumption that we have access to either (a) pretrained cross-modal embedding functions or (b) a dataset with paired input-text data that can be used to learn cross-modal embedding functions. In practice, we find that this assumption can generally be satisfied since many input types naturally coexist with textual descriptions.

Domino uses four types of cross-modal embeddings to enable slice discovery across our input domains: CLIP (Radford et al., 2021), ConVIRT (Zhang et al., 2020), MIMIC-CLIP, and EEG-CLIP. We adapt CLIP and ConVIRT from prior work, and we train MIMIC-CLIP and EEG-CLIP on large datasets with paired inputs and text (implementation details are provided in Section A.3).

# 4.2 CLUSTERING EMBEDDINGS WITH ERROR-AWARE MIXTURE MODEL

With cross-modal embeddings in hand, we proceed to the second step in the Domino pipeline: slicing. Recall from Section 2 that our goal is to find a set of  $\hat{k}$  slicing functions that partition our data into subgroups. This step resembles a standard unsupervised clustering problem but differs in an important way: we are specifically interested in finding underperforming clusters where the model makes systematic prediction errors.

Taking inspiration from the recently developed Spotlight algorithm d'Eon et al. (2021), we propose a simple mixture model that jointly models the input embeddings, class labels, and model predictions. This encourages clusters that are homogeneous with respect to error type (e.g. all false positives). The model assumes the following generative process (visualized in Figure 1): given the slice  $S$ , the embeddings are normally distributed  $Z|S \sim \mathcal{N}(\mu, \Sigma)$  with parameters mean  $\mu \in \mathbb{R}^d$  and  $\Sigma \in \mathbb{S}_{++}^d$  (the set of symmetric positive definite  $d \times d$  matrices), the labels vary as a categorical  $Y|S \sim Cat(\mathbf{p})$  with parameter  $\mathbf{p} \in \{\mathbf{p} \in \mathbb{R}_+^c : \sum_{i=1}^c p_i = 1\}$ , and the model predictions also vary as a categorical  $\hat{Y}|S \sim Cat(\hat{\mathbf{p}})$  with parameter  $\hat{\mathbf{p}} \in \{\mathbf{p} \in \mathbb{R}_+^c : \sum_{i=1}^c p_i = 1\}$ . Critically, this assumes that the embedding, label, and prediction are all independent of one another conditioned on the slice.

The mixture model is parameterized by  $\phi = [\mu, \Sigma, \mathbf{p}, \hat{\mathbf{p}}]$ . The log-likelihood over the  $n$  examples in the validation dataset  $D_v$  is given as follows and maximized using expectation-maximization:

$$
\ell (\phi) = \sum_ {i = 1} ^ {n} \log \sum_ {s = 1} ^ {k} P (S = s) P (X = x _ {i} | S = s) P (Y = y _ {i} | S = s) P (\hat {Y} = h _ {\theta} (x _ {i}) | S = s), \tag {3}
$$

# 4.3 GENERATING NATURAL LANGUAGE EXPLANATIONS FOR UNDERPERFORMING SLICES

Unlike previous SDMs, Domino provides actionable natural language statements that describe characteristics shared between examples in the discovered slices. To generate slice descriptions, we begin by sourcing a corpus of candidate natural language phrases  $\mathcal{D}_{\mathrm{text}} = \{t_j\}_{j=1}^{n_{\mathrm{text}}}$ . Critically, this text data does not have to be paired with the inputs  $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$  and can be sourced independently. For natural images, we include the  $n_{\mathrm{text}} = 10,000$  most frequently used words on English Wikipedia (Semenov & Arefin, 2019). For our medical image and time-series datasets, we use corpora of physician reports sourced from MIMIC-CXR (Johnson et al., 2019) and (blinded citation) with  $n_{\mathrm{text}} = 159,830$  and  $n_{\mathrm{text}} = 41,258$ , respectively.

We generate an embedding for each phrase in the corpus  $\mathcal{D}_{\text{text}}$  using the cross-modal embedding function  $\psi_{\text{text}}$ , yielding  $\{\mathbf{z}_{\text{text}}^{(i)} := \psi_{\text{text}}(t_i)\}_{i=1}^{n_{\text{text}}}$ . Then, we compute a prototype embedding for each of the discovered slices by taking the weighted average of the input embeddings in the slice,  $\{\mathbf{z}_{\text{slice}}^{(j)} := \psi_{\text{slice}}^{(j)}(\mathbf{Z}_{\text{input}})^{\top}\mathbf{Z}_{\text{input}}\}_{j=1}^{\hat{k}}$ . We also compute a prototype embedding for each class  $\{\mathbf{z}_{\text{class}}^{(c)} := \mathbf{1}[\mathbf{y} = c]^{-1}\mathbf{Z}_{\text{input}}\}_{c=1}^{C}$ . To distill the slice prototypes, we subtract out the prototype of the most common class in the slice  $\mathbf{z}_{\text{slice}}^{(j)} - \mathbf{z}_{\text{class}}^{(c)}$ . Finally, to find text that describes each slice, we compute the dot product between the distilled slice prototypes and the text embeddings and return the phrase with

![](images/4142ef34649849d4a1bda5d70e6b4b5d7716a537834c3860eff78451190bb0f2.jpg)  
Figure 3: Cross-modal embeddings enable accurate slice discovery. Using our evaluation framework, we demonstrate that the use of cross-modal embeddings leads to consistent improvements in slice discovery across three datasets and two input modalities, evaluated over 1,235 settings.

the highest value

$$
\operatorname {a r g m a x} _ {i \in [ n _ {\text {t e x t}}} \mathbf {z} _ {\text {t e x t}} ^ {(i) \top} \left(\mathbf {z} _ {\text {s l i c e}} ^ {(j)} - \mathbf {z} _ {\text {c l a s s}} ^ {(c)}\right). \tag {4}
$$

# 5 EXPERIMENTS

We use the evaluation framework developed in Section 3 to systematically assess Domino, comparing it to existing SDMs across 1,235 slice discovery settings. Our experiments validate the three core design choices behind Domino: (1) the use of cross-modal embeddings, (2) the use of a novel error-aware mixture model, and (3) the generation of natural language descriptions for slices.

![](images/87332ca7704be93fbd73bdc92e8480544577e5474ca5459668a5ad9aec1b98d2.jpg)  
Figure 4: Error-aware mixture model enables accurate slice discovery. We show that when cross-modal embeddings are provided as input, our error-aware mixture model often outperforms previously-designed SDMs. Similar results on MIMIC and EEG are detailed in Section A.4.

# 5.1 CROSS-MODAL EMBEDDINGS IMPROVE SDM PERFORMANCE

To measure how the choice of embedding affects performance, we hold step 2 (clustering algorithm) of Domino fixed and vary the embeddings used in step 1.

Natural Images. In slice discovery settings with natural images, we compare four embeddings: the final-layer activations of a randomly-initialized ResNet-50, the final-layer activations of the trained classifier  $h_{\theta}$ , BiT (Kolesnikov et al., 2019), and CLIP (Radford et al., 2021). Out of the four embedding types, CLIP is the only cross-modal embedding. Results are shown in Figure 3.

When evaluating with synthetic models, we find that using CLIP embeddings results in a mean precision-at-10 of 0.570 (0.554,0.586), a 9 percentage-point increase over BiT embeddings and a 23-percentage point increase over untrained activations. Note that synthetic models do not have activations, so we cannot compare to the final-layer activations in this setting.

When evaluating with trained models, we find no difference between using CLIP embeddings and BiT embeddings. However, both outperform using the activations of the trained classifier  $h_{\theta}$  by nearly 15-points in mean precision-at-10. This finding is of particular interest given that classifier activations are a popular embedding choice in existing SDMs (d'Eon et al., 2021; Sohoni et al., 2020). Notably, the gap between CLIP and  $h_{\theta}$  activations is much smaller in settings with correlation slices. This makes sense because a model that relies on a correlate to make predictions will likely capture information about the correlate in its activations (Sohoni et al., 2020).

Medical Images. On MIMIC-CXR, we compare five embeddings: the final-layer activations of a ResNet-50 pretrained on ImageNet, the final-layer activations of the trained classifier  $h_{\theta}$ , BiT (Kolesnikov et al., 2019), and domain-specific cross-modal embeddings that we trained using two different methods: CLIP and ConVIRT. For synthetic models, cross-modal ConVIRT embeddings enable a mean precision-at-10 of 0.765 (0.747,0.784), a 7-point improvement over the best unimodal embeddings (BiT) with mean precision-at-10 of 0.695 (0.674,0.716). With trained models, we again find that although  $h_{\theta}$  activations are the worst performing embeddings on both rare and noisy label slices, they are competitive with the multi-modal embeddings on correlation slices.

Medical Time-Series. For our EEG dataset, we compare two embeddings: the final-layer activations of a pretrained seizure classifier and a CLIP-style cross-modal embedding trained on EEG-report pairs. When evaluating with synthetic models, we find that cross-modal embeddings recover coherent slices with a mean precision-at-10 of 0.697 (0.605,0.784). This represents a 17-point gain over using unimodal embeddings 0.532 (0.459,0.608). This demonstrates that cross-modal embeddings can aid in recovering coherent slices even in input modalities other than images.

# 5.2 ERROR-AWARE MIXTURE MODEL IMPROVES SDM PERFORMANCE

In order to understand the how the choice of clustering algorithm affects performance, we hold constant the first step of the Domino pipeline (cross-modal embeddings) and vary the algorithm used in step 2.

We compare the error-aware mixture model to four prior SDMs: George (Sohoni et al., 2020), Multiaccuracy (Kim et al., 2018), Spotlight (d'Eon et al., 2021), and an important baseline we call ConfusionSDM, which outputs slicing functions that partition data into the cells of the confusion matrix. Additional implementation details are in Section A.2. We provide cross-modal embeddings as input to all five SDMs. On noisy and rare slices, the error-aware mixture model recovers underperforming, coherent slices with a mean precision-at-10 of 0.639 (0.617,0.660) – this represents a  $105\%$  improvement over the next-best method, George. Interestingly, on correlation slices, the error-aware mixture model outperforms all methods except the simple ConfusionSDM baseline.

# 5.3 DOMINO PROVIDES NATURAL LANGUAGE DESCRIPTIONS OF DISCOVERED SLICES.

Domino is the first SDM that can generate natural language descriptions for identified slices. For natural images, we provide a quantitative analysis of these descriptions. Specifically, since Domino returns a ranking over all phrases in the corpus  $\mathcal{D}_{\text{text}}$  (specified by the dot product computed in Equation 4), we can compute the percentage of settings in which the name of the "ground truth" slice (or a synonym) appears in the top- $m$  words returned by Domino. In Figure 5, we plot this

![](images/db3c7f309b21e7c5cf89fd691668bfc43e20be6ca9cd374c8fe50ce8a12271c1.jpg)  
Figure 5: Domino produces natural language explanations for discovered slices. (Left) The fraction of natural image settings with rare slices where Domino includes the exact name of the slice in the top- $k$  slice descriptions. (Right) Three slice discovery settings randomly selected from the set of the 85 rare, natural image settings where Domino includes the exact name of the slice in its top 3 slice descriptions. The chart shows the confidence scores for the top 5 descriptions. The images represent the top 3 images that Domino associates with the discovered slice.

percentage for  $m = 1$  to  $m = 10$  and show three randomly sampled examples. We find that for  $34.7\%$  of slices, Domino ranks the name of the slice (or a synonym) first out of the 10,000 words in our corpus. In  $57.4\%$  of slices, Domino ranks it in the top ten. For our medical domains, we provide the top-3 returned descriptions in Section A.5, Table 2. For MIMIC, Domino correctly describes cardiomegaly (i.e. enlarged heart), a correlate that the model incorrectly relied on to predict a pleural condition. For EEG, Domino correctly describes patient age, a correlate that the model incorrectly relied on to predict seizure.

# 6 RELATED WORK

Our work builds on prior efforts for slice discovery, cross-modal training, and SDM evaluation.

Slice discovery methods. There have been several recent studies that have proposed SDMs (d'Eon et al., 2021; Yeh et al., 2020; Sohoni et al., 2020; Kim et al., 2018). However, the conditions under which these SDMs succeed at identifying coherent, underperforming slices remain unclear.

Cross-modal training. Large-scale cross-modal representation learning approaches yield powerful embeddings that have contributed to large performance improvements across information retrieval and classification tasks (Radford et al., 2021). Cross-modal models that have inspired our work include CLIP for natural images (Radford et al., 2021), ConVIRT for medical images (Zhang et al., 2020), and WikiSatNet (Uzkent et al., 2019) and Tile2Vec (Jean et al., 2019) for satellite imagery.

Slice discovery datasets. Recently, several benchmark datasets have been proposed for evaluating the performance of models on shifting data distributions. These benchmarks are valuable because they provide labels specifying important slices of data. However, for the purpose of evaluating SDMs, they do not suffice because they either only annotate a small number of slices (Koh et al., 2021) or do not provide pretrained models that underperform on the slices (He et al., 2021; Khosla et al., 2011; Hendrycks & Dietterich, 2019; Liang & Zou, 2021). In Section A.6, we provide a comprehensive survey of underperforming slices in the wild.

# 7 CONCLUSION

In this work, we analyze the slice discovery problem. First, we observe that existing approaches for evaluating SDM performance do not allow for large-scale, quantitative evaluations. We address this challenge by introducing a programmable approach to measure SDM performance across two axes: underperformance and coherence. Second, we propose Domino, a novel SDM that combines cross-modal representations with an error-aware mixture model. Using our evaluation framework, we demonstrate that the embedding and slicing steps of Domino outperform those of existing SDMs. We also show for the first time that using cross-modal embeddings for slice discovery can enable the generation of semantically meaningful slice descriptions. Notably, Domino requires only blackbox access to models, and can thus be broadly useful in settings where users have API access to models. We hope that our approach is useful not only for improving slice discovery in practical settings, but also for allowing models to be trained in a manner that directly incorporates human understanding of the task at hand.

# 8 REPRODUCIBILITY STATEMENT

We provide a robust, open-source implementation of our evaluation framework at https://github.com/<blindedurl>/domino. The implementations will enable researchers to reproduce the results described here as well as run their own evaluations on additional datasets. The implementation also includes scripts for preprocessing the publicly available datasets used in this study.

# 9 ETHICS STATEMENT

Domino is a tool for identifying systematic model errors. No matter how effective it is at this task, there may still be error-modes Domino will not catch. There is a legitimate concern that model debugging tools like Domino could give practitioners a false sense of security, when in fact their models are failing on important slices not recovered by Domino. It is critical that practitioners still run standard evaluations on accurately-labeled, representative test sets in addition to using Domino for auditing models. Additionally, because Domino uses embeddings trained on image-text pairs sourced from the web, it may reflect societal biases when identifying and describing slices. Future work should explore the impacts of using biased embeddings to identify errors in models. What kinds of error modes might we miss? Are certain underrepresented groups or concepts less likely to be identified as an underperforming slice?

# REFERENCES

Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, and Been Kim. Sanity Checks for Saliency Maps. In S Bengio, H Wallach, H Larochelle, K Grauman, N Cesa-Bianchi, and R Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 9505-9515. Curran Associates, Inc., 2018.  
Marcus A Badgeley, John R Zech, Luke Oakden-Rayner, Benjamin S Glicksberg, Manway Liu, William Gale, Michael V McConnell, Bethany Percha, Thomas M Snyder, and Joel T Dudley. Deep learning predicts hip fracture using confounding patient and healthcare variables. npj Digital Medicine, 2(1):1-10, April 2019a.  
Marcus A Badgeley, John R Zech, Luke Oakden-Rayner, Benjamin S Glicksberg, Manway Liu, William Gale, Michael V McConnell, Bethany Percha, Thomas M Snyder, and Joel T Dudley. Deep learning predicts hip fracture using confounding patient and healthcare variables. NPJ digital medicine, 2(1):1-10, 2019b.  
Alceu Bissoto, Michel Fornaciali, Eduardo Valle, and Sandra Avila. (de) constructing bias on skin lesion datasets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0, 2019.  
Joy Buolamwini and Timnit Gebru. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency, pp. 77-91. PMLR, 2018.  
Terrance de Vries, Ishan Misra, Changhan Wang, and Laurens van der Maaten. Does object recognition work for everyone? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 52-59, 2019a.  
Terrance de Vries, Ishan Misra, Changhan Wang, and Laurens van der Maaten. Does object recognition work for everyone? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2019b.  
Alex J DeGrave, Jose Janizek, and Su-In Lee. AI for radiographic COVID-19 detection selects shortcuts over signal. Nature Machine Intelligence, 3(7):610-619, May 2021a.  
Alex J DeGrave, Joseph D Janizek, and Su-In Lee. Ai for radiographic covid-19 detection selects shortcuts over signal. Nature Machine Intelligence, pp. 1-10, 2021b.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248-255, June 2009.  
Greg d'Eon, Jason d'Eon, James R. Wright, and Kevin Leyton-Brown. The spotlight: A general method for discovering systematic errors in deep learning models, 2021.  
Arjun D Desai, Francesco Caliva, Claudia Iriondo, Aliasharg Mortazi, Sachin Jambawalikar, Ulas Bagci, Mathias Perslev, Christian Igel, Erik B Dam, Sibaji Gaj, et al. The international workshop on osteoarthritis imaging knee mri segmentation challenge: a multi-institute evaluation and analysis framework on a standardized dataset. *Radiology: Artificial Intelligence*, 3(3):e200078, 2021.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021.  
Christiane Fellbaum. WordNet: An Electronic Lexical Database. Bradford Books, 1998.  
Karan Goel, Nazneen Rajani, Jesse Vig, Samson Tan, Jason Wu, Stephan Zheng, Caiming Xiong, Mohit Bansal, and Christopher Ré. Robustness Gym: Unifying the NLP Evaluation Landscape. arXiv:2101.04840 [cs], January 2021.  
Yue He, Zheyan Shen, and Peng Cui. Towards non-i.i.d. image classification: A dataset and baselines. Pattern Recognit., 110:107383, February 2021.  
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. March 2019.  
Neal Jean, Sherrie Wang, Anshul Samar, George Azzari, David Lobell, and Stefano Ermon. Tile2vec: Unsupervised representation learning for spatially distributed data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 3967-3974, 2019.  
A Johnson, L Bulgarelli, T Pollard, S Horng, LA Celi, and R Mark. Mimic-iv (version 1.0), 2020.  
Alistair E W Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data, 6(1):1-8, 2019.  
Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei. Novel dataset for fine-grained image categorization. In First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition, Colorado Springs, CO, June 2011.  
Michael P Kim, Amirata Ghorbani, and James Zou. Multiaccuracy: Black-Box Post-Processing for Fairness in Classification. arXiv:1805.12317 [cs, stat], August 2018.  
Allison Koenecke, Andrew Nam, Emily Lake, Joe Nudell, Minnie Quartey, Zion Mengesha, Connor Toups, John R Rickford, Dan Jurafsky, and Sharad Goel. Racial disparities in automated speech recognition. Proceedings of the National Academy of Sciences, 117(14):7684-7689, 2020.  
Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton A Earnshaw, Imran S Haque, Sara Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, and Percy Liang. WILDS: A Benchmark of in-the-Wild Distribution Shifts. arXiv:2012.07421 [cs], March 2021.  
Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. December 2019.  
Weixin Liang and James Zou. Metadata: A dataset of datasets for evaluating distribution shifts and training conflicts. In ICML2021 ML4data Workshop, 2021.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.  
Luke Oakden-Rayner, Jared Dunnmon, Gustavo Carneiro, and Christopher Ré. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. September 2019.  
Laurel Orr, Megan Leszczynski, Simran Arora, Sen Wu, Neel Guha, Xiao Ling, and Christopher Re. Bootleg: Chasing the tail with self-supervised named entity disambiguation. October 2020.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. February 2021.  
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "why should i trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pp. 1135-1144, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450342322. doi: 10.1145/2939672.2939778. URL https://doi.org/10.1145/2939672.2939778.  
Subhrajit Roy, Isabell Kiral-Kornek, and Stefan Harrer. Chrononet: a deep recurrent neural network for abnormal EEG identification. In Conference on Artificial Intelligence in Medicine in Europe, pp. 47-56. Springer, 2019.  
Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization. arXiv:1911.08731 [cs, stat], April 2020.  
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.  
Ilya Semenov and Shamsul Arefin. Wikipedia word frequency. https://github.com/IlyaSemenov/wikipedia-word-frequency, 2019.  
Akshay Smit, Saahil Jain, Pranav Rajpurkar, Anuj Parek, Andrew Ng, and Matthew Lungren. Combining automatic labelers and expert annotations for accurate radiology report labeling using BERT. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1500-1519, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.117. URL https://aclanthology.org/2020.emnlp-main.117.  
Nimit Sohoni, Jared Dunnmon, Geoffrey Angus, Albert Gu, and Christopher Ré. No subclass left behind: Fine-grained robustness in coarse-grained classification problems. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 19339-19352. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/e0688d13958a19e087e123148555e4b4-Paper.pdf.  
Burak Uzkent, Evan Sheehan, Chenlin Meng, Zhongyi Tang, Marshall Burke, David Lobell, and Stefano Ermon. Learning to interpret satellite images using wikipedia. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, 2019.  
Julia K Winkler, Christine Fink, Ferdinand Toberer, Alexander Enk, Teresa Deinlein, Rainer Hofmann-Wellenhof, Luc Thomas, Aimilios Lallas, Andreas Blum, Wilhelm Stolz, and Holger A Haenssle. Association Between Surgical Skin Markings in Dermoscopic Images and Diagnostic Performance of a Deep Learning Convolutional Neural Network for Melanoma Recognition. JAMA Dermatol., 155(10):1135, October 2019a.  
Julia K Winkler, Christine Fink, Ferdinand Toberer, Alexander Enk, Teresa Deinlein, Rainer Hofmann-Wellenhof, Luc Thomas, Aimilios Lallas, Andreas Blum, Wilhelm Stolz, et al. Association between surgical skin markings in dermoscopic images and diagnostic performance of

a deep learning convolutional neural network for melanoma recognition. JAMA dermatology, 155 (10):1135-1141, 2019b.  
Chih-Kuan Yeh, Been Kim, Sercan Arik, Chun-Liang Li, Tomas Pfister, and Pradeep Ravikumar. On completeness-aware concept-based explanations in deep neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 20554-20565. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/ecb287ff763c169694f682af52c1f309-Paper.pdf.  
John R Zech, Marcus A Badgeley, Manway Liu, Anthony B Costa, Joseph J Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLoS Med., 15(11):e1002683, November 2018a.  
John R Zech, Marcus A Badgeley, Manway Liu, Anthony B Costa, Joseph J Titano, and Eric Karl Oermann. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLoS medicine, 15(11):e1002683, 2018b.  
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating Unwanted Biases with Adversarial Learning. Association for the Advancement of Artificial Intelligence (AAAI), January 2018.  
Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, and Curtis P. Langlotz. Contrastive learning of medical visual representations from paired images and text. CoRR, abs/2010.00747, 2020. URL https://arxiv.org/abs/2010.00747.
