# MEDFAIR: BENCHMARKING FAIRNESS FOR MEDICAL IMAGING

Anonymous authors

Paper under double-blind review

# ABSTRACT

A multitude of work has shown that machine learning-based medical diagnosis systems can be biased against certain subgroups of people. This has motivated a growing number of bias mitigation algorithms that aim to address fairness issues in machine learning. However, it is difficult to compare their effectiveness in medical imaging for two reasons. First, there is little consensus on the criteria to assess fairness. Second, existing bias mitigation algorithms are developed under different settings, e.g., datasets, model selection strategies, backbones, and fairness metrics, making a direct comparison and evaluation based on existing results impossible. In this work, we introduce MEDFAIR, a framework to benchmark the fairness of machine learning models for medical imaging. MEDFAIR covers eleven algorithms from various categories, nine datasets from different imaging modalities, and three model selection criteria. Through extensive experiments, we find that the under-studied issue of model selection criterion can have a significant impact on fairness outcomes; while in contrast, state-of-the-art bias mitigation algorithms do not significantly improve fairness outcomes over empirical risk minimization (ERM) in both in-distribution and out-of-distribution settings. We evaluate fairness from various perspectives and make recommendations for different medical application scenarios that require different ethical principles. Our framework provides a reproducible and easy-to-use entry point for the development and evaluation of future bias mitigation algorithms in deep learning.

# 1 INTRODUCTION

Machine learning-enabled automatic diagnosis with medical imaging is becoming a vital part of the current healthcare system (Lee et al., 2017). However, machine learning (ML) models have been found to demonstrate a systematic bias toward certain groups of people defined by race, gender, age, and even the health insurance type with worse performance (Obermeyer et al., 2019; Larrazabal et al., 2020; Spencer et al., 2013; Seyyed-Kalantari et al., 2021). The bias also exists in models trained from different types of medical data, such as chest X-rays (Seyyed-Kalantari et al., 2020), CT scans (Zhou et al., 2021), skin dermatology images (Kinyanjui et al., 2020), etc. A biased decision-making system is socially and ethically detrimental, especially in life-changing scenarios such as healthcare. This has motivated a growing body of work to understand bias and pursue fairness in the areas of machine learning and computer vision (Mehrabi et al., 2021; Louppe et al., 2017; Tartaglione et al., 2021; Wang et al., 2020).

Informally, given an observation input  $x$  (e.g., a skin dermatology image), a sensitive attribute  $s$  (e.g., male or female), and a target  $y$  (e.g., benign or malignant), the goal of a diagnosis model is to learn a meaningful mapping from  $x$  to  $y$ . However, ML models may amplify the biases and confounding factors that already exist in the training data related to sensitive attribute  $s$ . For example, data imbalance (e.g., over 90% individuals from UK Biobank (Sudlow et al., 2015) originate from European ancestries), attribute-class imbalance (e.g., in age-related macular degeneration (AMD) datasets, subgroups of older people contain more pathology examples than that of younger people (Farsiu et al., 2014)), label noise (e.g., Zhang et al. (2022) find that label noise in CheXpert dataset (Irvin et al., 2019) is much higher in some subgroups than the others), etc. Bias mitigation algorithms therefore aim to help diagnosis algorithms learn predictive models that are robust to confounding factors related to sensitive attribute  $s$  (Mehrabi et al., 2021).

![](images/a70ffbda3ba7235fd628eac80e9e0909fc14c16dba7d9c4b9c1b7a77c5dd01dd.jpg)  
Figure 1: Components of MEDFAIR benchmark.

Given the importance of ensuring fairness in medical applications and the special characteristics of medical data, we argue that a systematic and rigorous benchmark is needed to evaluate the bias mitigation algorithms for medical imaging. However, a straightforward comparison of algorithmic fairness for medical imaging is difficult, as there is no consensus on a single metric for fairness of medical imaging models. Group fairness (Dwork et al., 2012; Verma & Rubin, 2018) is a popular and intuitive definition adopted by many debiasing algorithms, which optimises for equal performance among subgroups. However, this can lead to a trade-off of increasing fairness by decreasing the performance of the advantaged group, reducing overall utility substantially. Doing so may violate the ethical principles of beneficence and non-maleficence (Beauchamp, 2003), especially for some medical applications where all subgroups need to be protected. There are also other fairness definitions, including individual fairness (Dwork et al., 2012), minimax fairness (Diana et al., 2021), counterfactual fairness (Kusner et al., 2017), etc. It is thus important to consider which definition should be used for evaluations.

In addition to the use of differing evaluation metrics, different experimental designs used by existing studies prevent direct comparisons between algorithms based on the existing literature. Most obviously, each study tends to use different datasets to evaluate their debiasing algorithms, preventing direct comparisons of results. Furthermore, many bias mitigation studies focus on evaluating tabular data with low-capacity models (Madras et al., 2018; Zhao et al., 2019; Diana et al., 2021), and recent analysis has shown that their conclusions do not generalise to high-capacity deep networks used for the analysis of image data (Zietlow et al., 2022). A crucial but less obvious issue is the choice of model selection strategy for hyperparameter search and early stopping. Individual bias mitigation studies are divergent or vague in their model selection criteria, leading to inconsistent comparisons even if the same datasets are used. Finally, given the effort required to collect and annotate medical imaging data, models are usually deployed in a different domain than the domain used for data collection. (E.g., data collected at hospital A is used to train a model deployed at hospital B). While the maintenance of prediction quality across datasets has been well studied, it is unclear if fairness achieved within one dataset (in-distribution) holds under dataset shift (out-of-distribution).

In order to address these challenges, we provide the first comprehensive fairness benchmark for medical imaging - MEDFAIR. We conduct extensive experiments across eleven algorithms, nine datasets, four sensitive attributes, and three model selection strategies to assess bias mitigation algorithms in both in-distribution and out-of-distribution settings. We report multiple evaluation metrics and conduct rigorous statistical tests to find whether any of the algorithms is significantly better. Having trained over 7,000 models using 6,800 GPU-hours, we have the following observations:

- Bias widely exists in ERM models trained in different modalities, which is reflected in the predictive performance gap between different subgroups for multiple metrics.  
- Model selection strategies can play an important role in improving the worst-case performance. Algorithms should be compared under the same model selection criteria.

- The state-of-the-art methods do not outperform the empirical risk minimization (ERM) with statistical significance in both in-distribution and out-of-distribution settings.

These results show the importance of a large benchmark suite such as MEDFAIR to evaluate progress in the field and to guide practical decisions about the selection of bias mitigation algorithms for deployment. MEDFAIR is released as a reproducible and easy-to-use codebase that all experiments in this study can be run with a single command. Detailed documentation is provided in order to allow researchers to extend and evaluate the fairness of their own algorithms and datasets, and we will also actively maintain the codebase to incorporate more algorithms, datasets, model selection strategies, etc. We hope our codebase can accelerate the development of bias mitigation algorithms and guide the deployment of ML models in clinical scenarios.

# 2 FAIRNESS IN MEDICINE

# 2.1 PROBLEM FORMULATION

We focus on evaluating the fairness of binary classification of medical images. Given an image, we predict its diagnosis label in a way that is not confounded by any sensitive attributes (age, sex, race, etc) so that the trained model is fair and not biased towards a certain subgroup of people.

Formally, Let  $D \in \{D_i\}_i^I$  be a set of domains, where  $I$  is the total number of domains. A domain can represent a dataset collected from a particular imaging modality, hospital, population, etc. Consider a domain  $D = (\mathcal{X}, \mathcal{Y}, \mathcal{S})$  to be a distribution where we have input sample  $\mathbf{x} \in \mathbb{R}^d$  over input space  $\mathcal{X}$ , the corresponding binary label  $y \in \{0,1\}$  over label space  $\mathcal{Y}$ , and sensitive attributes  $s \in \{0,1,\dots,m-1\}$  with  $m$  classes over sensitive space  $\mathcal{S}$ . We train a model  $h \in \mathcal{H}$  to output the prediction  $\hat{y} \in \{0,1\}$ , i.e.,  $h: \mathcal{X} \to \mathcal{Y}$ , where  $\mathcal{H}$  is the hypothesis class of models. Note that for each dataset  $\mathcal{D}_i$ , there may exist several sensitive attributes at the same time, e.g., there are data of patients' age and sex. We only consider one sensitive attribute at one time.

In-distribution Given a domain  $D_{i}$ , assume the input samples  $X_{i}$ , their labels  $Y_{i}$ , and the sensitive attributes  $S_{i}$  are identically and independently distributed (iid) from a joint probability distribution  $P_{i}(X_{i}, Y_{i}, S_{i})$ . We define the evaluation where the training and testing on the same domain  $D_{i}$  to be in-distribution, i.e., the training and testing set are from the same distribution. We train models for each combination of algorithms  $\times$  datasets  $\times$  sensitive attributes.

Out-of-distribution In clinical scenarios, due to the lack of training data, it is common to deploy a model trained in the original dataset to new hospitals/populations that have different data distributions. We define the training on one domain  $D_{i}$  and testing on the other unseen domain  $D_{j}$  to be out-of-distribution settings, where  $D_{i}$  and  $D_{j}$  may have different distribution  $P_{i}(X_{i},Y_{i},S_{i})$  and  $P_{j}(X_{j},Y_{j},S_{j})$ . In this case, we assume domains  $D_{i}$  and  $D_{j}$  must have the same input space (e.g., X-ray modality), diagnosis labels, and sensitive attributes, but differ in their joint distributions due to collection from different locations or different imaging protocols. We evaluate if bias mitigation algorithms are robust to distribution shift by directly using the model selected from the in-distribution setting of the domain  $D_{i}$  to test on the domain  $D_{j}$ .

# 2.2 FAIRNESS DEFINITION IN MEDICINE

Here we consider two most salient fairness definitions for healthcare, i.e., group fairness and MaxMin fairness. We argue that one should focus on different fairness definitions depending on the specific clinical application.

Group Fairness Metrics based on group fairness usually aim to achieve parity of predictive performance across protected subgroups. For resource allocation problems that can be considered a zero-sum game due to the limited resources, e.g., prioritising which patients should be sent to a limited number of intensive care units (ICUs), it is important to consider group fairness to reduce the disparity among different subgroups (related discussions in Hellman (2008); Barocas & Selfbst (2016)). We measure the performance gap in diagnosis AUC between the advantaged and disadvantaged subgroups as an indicator of group fairness. This is in line with the "separability" criteria

(Chen et al., 2021; Dwork et al., 2012) that algorithm scores should be conditionally independent of the sensitive attribute given the diagnostic label (i.e.,  $\hat{Y} \perp S|Y$ ), which is also adopted by (Gardner et al., 2019; Fong et al., 2021). On the other hand, Zietlow et al. (2022) find that for high-capacity models in computer vision, this is typically achieved by worsening the performance of the advantaged group rather than improving the disadvantaged group, a phenomenon termed as leveling down in philosophy that has incurred numerous criticisms (Christiano & Braynen, 2008; Brown, 2003; Doran, 2001). Worse, practical implementations often lead to worsening the performance of both subgroups (Zietlow et al., 2022), making it pareto inefficient and comprehensively violating beneficence and non-maleficence principles (Beauchamp, 2003). Thus, we argue that group fairness alone is not sufficient to analyse the trade-off between fairness and utility.

Max-Min Fairness It is another definition of fairness (Lahoti et al., 2020) following Rawlsian max-min fairness principle (Rawls, 2001), which is also studied as minimax group fairness (Diana et al., 2021) or minimax Pareto fairness (Martinez et al., 2020). Here, instead of seeking to equalize the error rates among subgroups, it treats the model that reduces the worst-case error rates as the fairer one. It may be a more appropriate definition than group fairness for some medical applications such as diagnosis, as it better satisfies the beneficence and non-maleficence principles (Beauchamp, 2003; Chen et al., 2018; Ustun et al., 2019), i.e., do the best and do no harm. Formally, for a model  $h$  in the hypothesis class  $\mathcal{H}$ , denote  $U_{s}(h)$  to be a utility function for subgroup  $s$ . A model  $h^{*}$  is considered to be Max-Min Fair if it maximizes (Max-) the utility of the worst-case (Min) group:

$$
h ^ {*} = \underset {h \in \mathcal {H}} {\operatorname {a r g m a x}} \quad \min  _ {s \in S} U _ {s} (h). \tag {1}
$$

In practice, it is hard to quantify the maximum optimal utility, and therefore we treat a model  $h_k$  to be fairer than the other model  $h_t$  if

$$
\min  _ {s \in S} U _ {s} \left(h _ {k}\right) > \min  _ {s \in S} U _ {s} \left(h _ {t}\right). \tag {2}
$$

We measure both group fairness and Max-Min fairness to give a more comprehensive evaluation for fairness in medical applications.

# 3 MEDFAIR

We implement a reproducible and easy-to-use codebase MEDFAIR to benchmark fairness in machine learning algorithms for medical imaging. In our benchmark, we conduct large-scale experiments in nine datasets, eleven algorithms, up to three sensitive attributes for each dataset, and three model selection criteria, where all the experiments can be run with a single command. We provide source code and detailed documentation, allowing other researchers to reproduce the results and incorporate other datasets and algorithms easily.

# 3.1 DATASETS

Nine datasets are included in MEDFAIR: CheXpert (Irvin et al., 2019), MIMIC-CXR (Johnson et al., 2019), PAPILA (Kovalyk et al., 2022), HAM10000 (Tschandl et al., 2018), Fitzpatrick17k (Groh et al., 2021), COVID-CT-MD (Afshar et al., 2021), OCT (Farsiu et al., 2014), ADNI 1.5T, and ADNI 3T (Petersen et al., 2010), to evaluate the algorithms comprehensively, which are all publicly available to ensure the reproducibility. We consider five important aspects during dataset selection:

Imaging modalities. We select datasets covering various 2D and 3D imaging modalities, including X-ray, fundus photography, computed tomography (CT), magnetic resonance imaging (MRI), spectral domain optical coherence tomography (SD-OCT), and skin dermatology images.

Potential sources of bias. We involve datasets that may introduce bias from different sources, including label noise, data/class imbalance, spurious correlation, etc. Note that each dataset may contain more than one source of bias.

Sensitive attributes. The selected datasets contain attributes that are commonly treated sensitively and may be biased in clinical practice, including age, sex, race, and skin type.

Size of datasets. As the sizes of medical imaging datasets are often limited by privacy issues, etc., it is important to inspect whether the fairness algorithms are effective with different sizes of datasets. The dataset sizes range from  $420 \sim 370$ , 955 for 2D images and  $110 \sim 550$  for 3D scans.

Table 1: Detailed statistics of the datasets. "# images/ scans" listed here are the actual numbers used in this study after removing those missing sensitive attributes. For potential bias, LN, CI, DI, and SC represent label noise, class imbalance, data imbalance, and spurious correlation, respectively.  

<table><tr><td>Dataset</td><td>Modality</td><td># Images</td><td>Sensitive Attribute</td><td>Bias Sources</td></tr><tr><td>CheXpert</td><td>Chest X-ray (2D)</td><td>222,793</td><td>Age, Sex, Race</td><td>LN, CI, DI</td></tr><tr><td>MIMIC-CXR</td><td>Chest X-ray (2D)</td><td>370,955</td><td>Age, Sex, Race</td><td>LN, CI, DI</td></tr><tr><td>PAPILA</td><td>Fundus Image (2D)</td><td>420</td><td>Age, Sex</td><td>DI, CI</td></tr><tr><td>HAM10000</td><td>Skin Dermatology (2D)</td><td>9,948</td><td>Age, Sex</td><td>DI, CI</td></tr><tr><td>Fitzpatrick17k</td><td>Skin Dermatology (2D)</td><td>16,012</td><td>Skin type</td><td>LN, DI, CI</td></tr><tr><td>COVID-CT-MD</td><td>Lung CT (3D)</td><td>308</td><td>Age, Sex</td><td>DI, CI</td></tr><tr><td>OCT</td><td>SD-OCT (3D)</td><td>384</td><td>Age</td><td>DI, CI</td></tr><tr><td>ADNI 1.5T</td><td>Brain MRI (3D)</td><td>550</td><td>Age, Sex</td><td>SC</td></tr><tr><td>ADNI 3T</td><td>Brain MRI (3D)</td><td>110</td><td>Age, Sex</td><td>SC</td></tr></table>

Out-of-distribution evaluation. We include two pairs of datasets with the same modality but collected from different locations or different imaging protocols for out-of-distribution evaluations. Specifically, we choose two 2D chest X-ray datasets i.e., CheXpert and MIMIC-CXR, and two 3D brain MRI datasets i.e., ADNI 1.5T and ADNI 3T.

Table 1 lists the basic datasets information, and more detailed statistics are provided in Appendix B.

# 3.2 ALGORITHMS

MEDFAIR incorporates eleven algorithms across five categories: subgroup rebalancing, domainindependence, adversarial training, disentanglement, and domain generalization. We carefully implement the following algorithms based on the code released by the original authors:

# - Baseline

- Empirical Risk Minimization (ERM) (Vapnik, 1999) minimizes the average error across the dataset without considering the sensitive attributes.

# Subgroup Rebalancing

- Resampling method upsampled the minority groups so that all of the subgroups appear during training with equal chances.

# - Domain-independence

- Domain independent N-way classifier (DomainInd) (Wang et al., 2020) trains separate classifiers for different subgroups with a shared encoder.

# Adversarial Training

- Learning Adversarially Fair and Transferable Representations (LAFTER) (Madras et al., 2018) de-biases the representation by minimizing the ability to recognize sensitive attributes.  
- Conditional learning of Fair representation (CFair) (Zhao et al., 2019) tries to enforce the balanced error rate and conditional alignment of representations.  
- Learning Not to Learn (LNL) (Kim et al., 2019) unlearns the bias information iteratively by minimizing the mutual information between feature representation and bias.

# - Disentanglement

- Entangle and Disentangle (EnD) (Tartaglione et al., 2021) disentangles confounders by inserting an "information bottleneck", while still passing the useful information.  
- Orthogonal Disentangled Representations (ODR) (Sarhan et al., 2020) disentangles the useful and sensitive representations by enforcing orthogonality constraints for independence.

# - Domain Generalization (DG)

- Group Distributionally Robust Optimization (GroupDRO) (Sagawa et al., 2019) minimizes the worst-case training loss with increased regularization.  
- Stochastic Weight Averaging Densely (SWAD) (Cha et al., 2021), a state of the art method in DG, aims to find a robust flat minima by a dense stochastic weight sampling strategy.  
- Sharpness-Aware Minimization (SAM) (Foret et al., 2020) seeks parameters that lie in neighborhoods having uniformly low loss during optimization.

The hyper-parameter tuning strategy is described in Appendix B.2.2.

# 3.3 MODEL SELECTION

The trade-off between fairness and utility has been widely noted (Kleinberg et al., 2016; Zhang et al., 2022), making hyper-parameter selection criteria particularly difficult to define given the multi-objective nature of optimising for potentially conflicting fairness and utility. Previous work differs greatly in model selection. Some use conventional utility-based selection strategies, e.g., overall validation loss, while others have no explicit specification. We provide a summary of model selection strategies across the literature in Table A1. To investigate the influence of model selection strategies on the final performance, we study three prevalent selection strategies in MEDFAIR.

Overall Performance-based Selection This is one of the most basic and common strategies for model selection. It picks the model that has the smallest loss value or highest accuracy/AUC across the validation set of all sub-populations. However, this strategy tends to select the model with better performance in the majority group to achieve the best overall performance, leading to a potentially large performance gap among subgroups, which is illustrated in the red pentagon on the right side of Figure 2 (note that it is not necessarily Pareto optimal).

Minimax Pareto Selection The concept of Pareto optimality was proposed by Mas-Colell et al. (1995) and utilized in fair machine learning to study the trade-off among subgroup accuracies (Martinez et al., 2020). Intuitively, for a model on the Pareto front, no group can achieve better performance without hurting the performance of other groups. In other words, it defines the set of best achievable trade-offs among subgroups (without introducing unnecessary harm). Based on this definition, we select the model that lies on the Pareto front and achieves the best worst-case AUC (the red star in the middle top of Figure 2). We present a formal definition of minimax Pareto selection in Appendix B.4.

DTO-based Selection Distance to optimal (DTO) (Han et al., 2021) is calculated by the

normalized Euclidean distance between the performance of the current model and the optimal utopia point. Here, we construct the utopia point by taking the maximum AUC value of each subgroup among all models. The DTO strategy selects the model that has the smallest distance to the utopia point (the red hexagon in Figure 2).

![](images/178654b0423016ce3fb2dd3d9ee17c4451964537c3b8decde6bc71b08defcd10.jpg)  
Figure 2: Illustration of three different model selection strategies. Each data point represents a different hyper-parameter combination for one algorithm, where the red points are the models lying on the Pareto front.

# 3.4 EVALUATION AND IMPLEMENTATION

We apply the bias mitigation algorithms to medical image classification tasks and evaluate fairness based on the performance of different subgroups (sensitive attributes). The sensitive attributes are regarded as available during training (if needed). We consider two settings to evaluate fairness for medical imaging, i.e., in-distribution and out-of-distribution.

Evaluation Metrics We use the area under the receiver operating characteristic curve (AUC) as the major metric, which is a commonly used metric for medical binary classification. We evaluate the algorithms from three aspects: (1) utility: overall AUC across all subgroups; (2) group fair-

ness: AUC gap between the subgroups that have maximum AUC and minimum AUC; (3) Max-Min fairness: AUC of the worst-case group. Besides, we also report the values of binary cross entropy (BCE), expected calibration error (ECE), false positive rate (FPR), false negative rate (FNR), and true positive rate (TPR) at  $80\%$  true negative rate (TNR) of each subgroup, as well as the Equalized Odd (EqOdd). We provide detailed explanations of these metrics in Appendix B.3.

Statistical Tests Prior work has empirically evaluated bias mitigation algorithms and occasionally claimed that some algorithm works well based on results from a couple of datasets. We note that to make a stronger conclusion that would be more useful to practitioners, e.g., 'algorithm  $A$  works better than  $B$  for medical imaging' (i.e.,  $A$  is better general, rather than better for dataset  $C$  specifically), one needs to evaluate performance across several datasets and perform significance tests that check for consistently good performance that can not be explained by overfitting to a single dataset. This is where the MEDFAIR benchmark suite comes in. To rigorously compare the relative performance of different algorithms, we perform Friedman test (Friedman, 1937) following by Nemenyi post-hoc test (Nemenyi, 1963) for both settings to identify if any of the algorithms is significantly better than the others, following the authoritative guide of Demšar (2006). We first calculate the relative ranks among all algorithms on each dataset and sensitive attribute separately, and then take the average ranks for the Nemenyi test if significance is detected by Friedman test. We consider a p-value lower than 0.05 to be statistically significant. The testing results are visualized by Critical Difference (CD) diagrams (Demšar, 2006). In CD diagrams, methods that are connected by a horizontal line are in the same group, meaning they are not significantly different given the p-value, and methods that are in different groups (not connected by the same line) have statistically significant difference.

Implementation Details We adopt 2D and 3D ResNet-18 backbone (He et al., 2016; Hara et al., 2018) for 2D and 3D datasets, respectively. The light backbone is used to avoid overfitting as there are datasets with small sizes, and also to remain consistent with the backbone used in the original literature (Kim et al., 2019; Wang et al., 2020; Tartaglione et al., 2021; Sarhan et al., 2020). Binary cross entropy loss is used to supervise the training. Unless otherwise specified, we randomly split the whole dataset into training/Validation/testing sets with a proportion of  $80 / 10 / 10$  for 2D datasets and  $70 / 10 / 20$  for 3D datasets. To ensure the stability of randomness, for each experiment, we report the mean values and the standard deviations for three separate runs with three randomly selected seeds. Further implementation details for all datasets and algorithms can be found in Appendix B.

# 4 RESULTS

# 4.1 BIAS WIDELY EXISTS IN ML MODELS TRAINED IN DIFFERENT MODALITIES AND TASKS

Firstly, we train ERM on different datasets and sensitive attributes, and select models using the regular overall performance-based strategy. For each dataset and sensitive attribute, we calculate the maximum and minimum AUC and underdiagnosis rate among subgroups, where we use FNR for malignant label and FPR for "No Finding" label as the underdiagnosis rate. As shown in Figure 3, most points are to the side of the equality line, showing that the performance gap widely exists. This confirms a problem that has been widely discussed (Seyyed-Kalantari et al., 2021) but, until now, has never been systematically quantified for deep learning across a comprehensive variety of modalities, diagnosis tasks, and sensitive attributes.

# 4.2 MODEL SELECTION SIGNIFICANTLY INFLUENCES WORST-CASE GROUP PERFORMANCE

We study the impact of model selection strategies on ERM using our three metrics of interest using: The AUC of the worst-case group, the AUC gap, and overall AUC with ERM. We first conduct a hyper-parameter sweep for ERM while training on all the datasets, and then compute the metrics and the relative ranks of the three model selection strategies. The results, including statistical significance tests are summarised in Figure 4, and the raw data in Table A7. Each sub-plot corresponds to a metric of interest (worst-case AUC, AUC Gap, Overall AUC), and the average rank of each selection strategy (Pareto, DTO, Overall) is shown on a line. Selection strategies not connected by the bold bars have significantly different performance. The results show that for the worst-case AUC metric (left), the Pareto-optimal model selection strategy has the highest average rank of around 1.5, which is statistically significantly better than the overall AUC model selection strategy's average rank of around 2.5. Meanwhile, in terms of the overall AUC metric (right) the Pareto selection

![](images/41423d0f344293e613bf3629bcef794dcb921db844321522c0f6c1857500d77d.jpg)  
Figure 3: The AUC (left) and underdiagnosis rates (right) for the advantaged and disadvantaged subgroups across each dataset and sensitive attribute, when training with ERM. Most points are off the blue equality line, showing that bias widely exists in conventional ERM-trained models.

![](images/568e0f3563555a467549f4b9bc6ee1ffe1415629822180aeb069cd43c1a6f29b.jpg)  
Figure 4: Influence of model selection strategies on ERM, illustrated in CD diagrams. The higher the rank of the AUC Gap, the smaller the gap.

strategy is not significantly worse than the overall model selection strategy. Thus, even without any explicit bias mitigation algorithm, max-min fairness can be significantly improved simply by adopting the corresponding model selection strategy in place of the standard overall strategy - and this intervention need not impose a significant cost to the overall AUC.

# 4.3 NO METHOD OUTPERFORMS ERM WITH STATISTICAL SIGNIFICANCE

![](images/19482e6a1a57d3ae34220320073ae5c6d3d0603b0f6158743729de420609f647.jpg)  
Figure 5: Performance of bias mitigation algorithms summarised across all datasets as average rank CD diagrams. (a) in-distribution, (b) out-of-distribution. SWAD is the highest ranked method for worst- and overall-AUC metrics, but it is still not significantly better than ERM.

We next ask whether any of the purpose-designed bias mitigation algorithms is significantly better than ERM, and which algorithm is best overall? To answer these questions, we evaluate the perfor

mance of all methods using the Pareto model selection strategy. We report the Nemenyi post-hoc test results on worst-group AUC, AUC gap, and overall AUC in Figure 5 for in-distribution (top row) and out-of-distribution (bottom row) settings with raw data in Tables A8 and A9. For in-distribution, while there are some significant performance differences, no method outperforms ERM significantly for any metric: ERM is always in the highest rank group of algorithms without significant differences. The conclusion is the same for the out-of-distribution testing, and some methods that rank higher than ERM in the in-distribution setting perform worse than ERM when deployed to an unseen domain, suggesting that preserving fairness across domain-shift is challenging.

It is worth noting there are some methods that consistently perform better, e.g., SWAD ranks a clear first for the worst-case and overall AUC for both settings, and thus could be a promising method for promoting fairness. However, from a statistical significance point of view, SWAD is still not significantly better despite the fact that we use a much larger sample size (number of datasets) than most of the previous fairness studies. This shows that many studies do not use enough number of datasets to justify their desired claims. Our benchmark suite provides the largest collection of medical imaging datasets for fairness to date, and thus provides the best platform for future research to evaluate the efficacy of any method works in a rigorous statistical way.

# 5 DISCUSSION

Source of bias There are multiple confounding effects that can lead to bias, rather than any single easy-to-isolate factor. As summarised in Table 1 and discussed further in Appendix A.4, these include both measurable and unmeasurable factors spanning imbalance in subgroup size, imbalance in subgroup disease prevalence, difference in imaging protocols/time/location, spurious correlations, the intrinsic difference in difficulty of diagnosis for different subgroups, unintentional bias from the human labellers, etc. It is difficult or even impossible to disentangle all of these factors, making algorithms that specifically optimise for one particular factor to succeed.

Failure of the bias mitigation algorithms Although most bias mitigation algorithms are not consistently effective across our benchmark suite, we are certainly not trying to disparage them. It is understandable because some are not originally designed for medical imaging, which contains characteristics distinct from those of natural images or tabular data, and more work may be necessary to design medical imaging specific solutions. More fundamentally, different algorithms may succeed if addressing solely the specific confounding factors for which they are designed to compensate, but fail when presented with other confounders or a mixture of multiple confounders. For example, resampling specifically targets data imbalance, while disentanglement focuses more on removing spurious correlations. But real datasets may also simultaneously contain other potential sources of bias such as label noise. This may explain why SWAD is the most consistently high-ranked algorithm, as it optimises a general notion of robustness without any specific assumption on confounders or sensitive attributes, and thus may be more broadly beneficial to different confounding factors.

Relation of domain generalization and fairness The aim of domain generalization (DG) algorithms is to maintain stable performance on unseen sub-populations, while the fairness-promoting algorithms try to ensure that no known sub-populations are poorly treated. Despite this difference, they share the eventual goal — being robust to changes in distribution across different sub-populations. As shown in section 4, some domain generalization methods, such as SWAD, consistently improve the performance of all subgroups, and thus overall utility. However, we also notice that they may also enlarge the performance gap among subgroups. It introduces a question of whether a systematically better algorithm (i.e., improving Max-Min fairness) is fairer if it increases the disparity (i.e., not satisfying group fairness)? This question goes beyond machine learning and depends on application scenarios. We suggest that a relevant differentiator may be between diagnosis and zero-sum resource allocation problems where max-min and group-fairness could be prioritised respectively.

Are the evaluations enough for now? Although we have tried our best to include a diverse set of algorithms and datasets in our benchmark, it is certainly not exhaustive. There are methods to promote fairness from other perspectives, e.g., self-supervised learning may be more robust (Liu et al., 2021; Azizi et al., 2022). Also, datasets from other medical data modalities (e.g., cardiology, digital pathology) should be added. Beyond image classification, other important tasks in medical imaging, such as segmentation, regression, and detection, are underexplored. We will keep our codebase alive and actively incorporate more algorithms, datasets, and even other tasks in the future.

# REPRODUCIBILITY STATEMENT

We report the data preprocessing in Appendix B.1.2, hyper-parameter space in Appendix B.2.2. All of the datasets we use are publicly available, and we provide the download links in Table A5. Source code and documentation are available in supplementary materials and they will be released after the double-blind process. Running all the experiments required  $\sim 0.77$  NVIDIA A100-SXM-80GB GPU years.

# REFERENCES

Parnian Afshar, Shahin Heidarian, Nastaran Enshaei, Farnoosh Naderkhani, Moezedin Javad Rafiee, Anastasia Oikonomou, Faranak Babaki Fard, Kaveh Samimi, Konstantinos N Plataniotis, and Arash Mohammadi. Covid-ct-md, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning. Scientific Data, 8(1):1-8, 2021.  
Shekoofeh Azizi, Laura Culp, Jan Freyberg, Basil Mustafa, Sebastien Baur, Simon Kornblith, Ting Chen, Patricia MacWilliams, S Sara Mahdavi, Ellery Wulczyn, et al. Robust and efficient medical imaging with self-supervision. arXiv preprint arXiv:2205.09723, 2022.  
Solon Barocas and Andrew D Selbst. Big data's disparate impact. *Calif. L. Rev.*, 104:671, 2016.  
Tom L Beauchamp. Methods and principles in biomedical ethics. Journal of Medical ethics, 29(5): 269-274, 2003.  
Rachel KE Bellamy, Kuntal Dey, Michael Hind, Samuel C Hoffman, Stephanie Houde, Kalapriya Kannan, Pranay Lohia, Jacquelyn Martino, Sameep Mehta, Aleksandra Mojsilovic, et al. Ai fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias. IBM Journal of Research and Development, 63(4/5):4-1, 2019.  
Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust optimization, volume 28. Princeton university press, 2009.  
Lukas Biewald. Experiment tracking with weights and biases, 2020. URL https://www.wandb.com/. Software available from wandb.com.  
Sarah Bird, Miro Dudík, Richard Edgar, Brandon Horn, Roman Lutz, Vanessa Milan, Mehrnoosh Sameki, Hanna Wallach, and Kathleen Walker. Fairlearn: A toolkit for assessing and improving fairness in ai. Microsoft, Tech. Rep. MSR-TR-2020-32, 2020.  
Campbell Brown. Giving up levelling down. Economics & Philosophy, 19(1):111-134, 2003.  
Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6299-6308, 2017.  
Alessandro Castelnovo, Riccardo Crupi, Greta Greco, Daniele Regoli, Ilaria Giuseppina Penco, and Andrea Claudio Cosentini. A clarification of the nuances in the fairness metrics landscape. Scientific Reports, 12(1):1-21, 2022.  
Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems, 34:22405-22418, 2021.  
Shubham Chaudhary, Sadbhawna Sadbhawna, Vinit Jakhetiya, Badri N Subudhi, Ujjwal Baid, and Sharath Chandra Guntuku. Detecting Covid-19 and community acquired pneumonia using chest ct scan images with deep learning. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 8583-8587. IEEE, 2021.  
Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16:321-357, 2002.  
Irene Chen, Fredrik D Johansson, and David Sontag. Why is my classifier discriminatory? Advances in neural information processing systems, 31, 2018.

Richard J Chen, Tiffany Y Chen, Jana Lipkova, Judy J Wang, Drew FK Williamson, Ming Y Lu, Sharifa Sahai, and Faisal Mahmood. Algorithm fairness in ai for medicine and healthcare. arXiv preprint arXiv:2110.00603, 2021.  
Thomas Christiano and Will Braynen. Inequality, injustice and levelling down. Ratio, 21(4):392-420, 2008.  
Elliot Creager, David Madras, Jorn-Henrik Jacobsen, Marissa Weis, Kevin Swersky, Toniann Pitassi, and Richard Zemel. Flexibly fair representation learning by disentanglement. In International conference on machine learning, pp. 1436-1445. PMLR, 2019.  
Elliot Creager, Jorn-Henrik Jacobsen, and Richard Zemel. Exchanging lessons between algorithmic fairness and domain generalization. 2020.  
Janez Demšar. Statistical comparisons of classifiers over multiple data sets. The Journal of Machine learning research, 7:1-30, 2006.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Emily Diana, Wesley Gill, Michael Kearns, Krishnamaram Kenthapadi, and Aaron Roth. Minimax group fairness: Algorithms and experiments. In Proceedings of the 2021 AAAI/ACM Conference on AI, Ethics, and Society, pp. 66-76, 2021.  
Brett Doran. Reconsidering the levelling-down objection against egalitarianism. Utilitas, 13(1): 65-85, 2001.  
John Duchi, Peter Glynn, and Hongseok Namkoong. Statistics of robust optimization: A generalized empirical likelihood approach. arXiv preprint arXiv:1610.03425, 2016.  
Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference, pp. 214-226, 2012.  
Sina Farsiu, Stephanie J Chiu, Rachelle V O'Connell, Francisco A Folgar, Eric Yuan, Joseph A Izatt, Cynthia A Toth, Age-Related Eye Disease Study 2 Ancillary Spectral Domain Optical Coherence Tomography Study Group, et al. Quantitative classification of eyes with and without intermediate age-related macular degeneration using optical coherence tomography. Ophthalmology, 121(1): 162-172, 2014.  
Hortense Fong, Vineet Kumar, Anay Mehrotra, and Nisheeth K Vishnoi. Fairness for auc via feature augmentation. arXiv preprint arXiv:2111.12823, 2021.  
Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. arXiv preprint arXiv:2010.01412, 2020.  
Sorelle A Friedler, Carlos Scheidegger, Suresh Venkatasubramanian, Sonam Choudhary, Evan P Hamilton, and Derek Roth. A comparative study of fairness-enhancing interventions in machine learning. In Proceedings of the conference on fairness, accountability, and transparency, pp. 329-338, 2019.  
Milton Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the american statistical association, 32(200):675-701, 1937.  
Josh Gardner, Christopher Brooks, and Ryan Baker. Evaluating the fairness of predictive student models through slicing analysis. In Proceedings of the 9th international conference on learning analytics & knowledge, pp. 225-234, 2019.  
Matthew Groh, Caleb Harris, Luis Soenksen, Felix Lau, Rachel Han, Aerin Kim, Arash Koochek, and Omar Badri. Evaluating deep neural networks trained on clinical images in dermatology with the fitzpatrick 17k dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1820-1828, 2021.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pp. 1321-1330. PMLR, 2017.  
Xudong Han, Timothy Baldwin, and Trevor Cohn. Balancing out bias: Achieving fairness through balanced training. arXiv, 2021.  
Xudong Han, Aili Shen, Yitong Li, Lea Frermann, Timothy Baldwin, and Trevor Cohn. fair-lib: A unified framework for assessing and improving classification fairness. arXiv preprint arXiv:2205.01876, 2022.  
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh. Can spatiotemporal 3d cnns retrace the history of 2d cnns and imagenet? In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 6546-6555, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Deborah Hellman. When is discrimination wrong? Harvard University Press, 2008.  
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural computation, 9(1):1-42, 1997.  
Badr Youbi Idrissi, Martin Arjovsky, Mohammad Pezeshki, and David Lopez-Paz. Simple data balancing achieves competitive worst-group-accuracy. In Conference on Causal Learning and Reasoning, pp. 336–351. PMLR, 2022.  
Jeremy A. Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn L. Ball, Katie S. Shpanskaya, Jayne Seekins, David Andrew Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, C. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and A. Ng. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In AAAI, 2019.  
Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry P. Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In Amir Globerson and Ricardo Silva (eds.), Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pp. 876-885. AUAI Press, 2018. URL http://auai.org/uai2018/proceedings/papers/313.pdf.  
Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Yifan Peng, Zhiyong Lu, Roger G Mark, Seth J Berkowitz, and Steven Horng. Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042, 2019.  
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv:1609.04836, 2016.  
Sajad Khodadadian, AmirEmad Ghassami, and Negar Kiyavash. Impact of data processing on fairness in supervised learning. In 2021 IEEE International Symposium on Information Theory (ISIT), pp. 2643-2648. IEEE, 2021.  
Byungju Kim, Hyunwoo Kim, Kyungsu Kim, Sungjin Kim, and Junmo Kim. Learning not to learn: Training deep neural networks with biased data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9012-9020, 2019.  
Newton M Kinyanjui, Timothy Odonga, Celia Cintas, Noel CF Codella, Rameswar Panda, Prasanna Sattigeri, and Kush R Varshney. Fairness of classifiers across skin tones in dermatology. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 320-329. Springer, 2020.  
Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. Inherent trade-offs in the fair determination of risk scores. arXiv preprint arXiv:1609.05807, 2016.

Oleksandr Kovalyk, Juan Morales-Sanchez, Rafael Verdu-Monedero, Inmaculada Sellés-Navarro, Ana Palazón-Cabanes, and José-Luis Sancho-Gómez. Papila: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment. Scientific Data, 9(1): 1-12, 2022.  
Matt J Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. Advances in neural information processing systems, 30, 2017.  
Preethi Lahoti, Alex Beutel, Jilin Chen, Kang Lee, Flavien Prost, Nithum Thain, Xuezhi Wang, and Ed Chi. Fairness without demographics through adversarially reweighted learning. Advances in neural information processing systems, 33:728-740, 2020.  
Agostina J Larrazabal, Nicolas Nieto, Victoria Peterson, Diego H Milone, and Enzo Ferrante. Gender imbalance in medical imaging datasets produces biased classifiers for computer-aided diagnosis. Proceedings of the National Academy of Sciences, 117(23):12592-12594, 2020.  
Chelsea E Lee, Kaela S Singleton, Melissa Wallin, and Victor Faundez. Rare genetic diseases: nature's experiments on human development. IScience, 23(5):101123, 2020.  
June-Goo Lee, Sanghoon Jun, Young-Won Cho, Hyunna Lee, Guk Bae Kim, Joon Beom Seo, and Namkug Kim. Deep learning in medical imaging: general overview. Korean journal of radiology, 18(4):570-584, 2017.  
Jungsoo Lee, Eungyeup Kim, Juyoung Lee, Jihyeon Lee, and Jaegul Choo. Learning debiased representation via disentangled feature augmentation. Advances in Neural Information Processing Systems, 34:25123-25133, 2021.  
Hong Liu, Jeff Z HaoChen, Adrien Gaidon, and Tengyu Ma. Self-supervised learning is more robust to dataset imbalance. arXiv preprint arXiv:2110.05025, 2021.  
Jianfang Liu, Tara Lichtenberg, Katherine A Hoadley, Laila M Poisson, Alexander J Lazar, Andrew D Cherniack, Albert J Kovatich, Christopher C Benz, Douglas A Levine, Adrian V Lee, et al. An integrated tcga pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell, 173(2):400-416, 2018.  
Francesco Locatello, Gabriele Abbati, Thomas Rainforth, Stefan Bauer, Bernhard Scholkopf, and Olivier Bachem. On the fairness of disentangled representations. Advances in Neural Information Processing Systems, 32, 2019.  
Gilles Louppe, Michael Kagan, and Kyle Cranmer. Learning to pivot with adversarial networks. Advances in neural information processing systems, 30, 2017.  
David Madras, Elliot Creager, Toniann Pitassi, and Richard Zemel. Learning adversarily fair and transferable representations. In International Conference on Machine Learning, pp. 3384-3393. PMLR, 2018.  
Roman C Maron, Michael Weichenthal, Jochen S Utikal, Achim Hekler, Carola Berking, Axel Hauschild, Alexander H Enk, Sebastian Haferkamp, Joachim Klode, Dirk Schadendorf, et al. Systematic outperformance of 112 dermatologists in multiclass skin cancer image classification by convolutional neural networks. European Journal of Cancer, 119:57-65, 2019.  
Natalia Martinez, Martin Bertran, and Guillermo Sapiro. Minimax pareto fairness: A multi objective perspective. In International Conference on Machine Learning, pp. 6755-6764. PMLR, 2020.  
Andreu Mas-Colell, Michael Dennis Whinston, Jerry R Green, et al. Microeconomic theory, volume 1. Oxford university press New York, 1995.  
Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM Computing Surveys (CSUR), 54(6):1-35, 2021.  
Kaisa Miettinen. Introduction to multiobjective optimization: Noninteractive approaches. In Multi-objective optimization, pp. 1-26. Springer, 2008.  
Peter Bjorn Nemenyi. Distribution-free multiple comparisons. Princeton University, 1963.

Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In CVPR Workshops, volume 2, 2019.  
Ziad Obermeyer, Brian Powers, Christine Vogeli, and Sendhil Mullainathan. Dissecting racial bias in an algorithm used to manage the health of populations. Science, 366(6464):447-453, 2019.  
Sungho Park, Jewook Lee, Pilhyeon Lee, Sunhee Hwang, Dohyung Kim, and Hyeran Byun. Fair contrastive learning for facial attribute classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10389-10398, 2022.  
R. C. Petersen, P. S. Aisen, L. A. Beckett, M. C. Donohue, A. C. Gamst, D. J. Harvey, C. R. Jack, W. J. Jagust, L. M. Shaw, A. W. Toga, J. Q. Trojanowski, and M. W. Weiner. Alzheimer's disease neuroimaging initiative (adni). *Neurology*, 74(3):201-209, 2010. ISSN 0028-3878. doi: 10.1212/WNL.0b013e3181cb3e25.  
Geoff Pleiss, Manish Raghavan, Felix Wu, Jon Kleinberg, and Kilian Q Weinberger. On fairness and calibration. Advances in neural information processing systems, 30, 2017.  
John Rawls. Justice as fairness: A restatement. Harvard University Press, 2001.  
Charan Reddy, Deepak Sharma, Soroush Mehri, Adriana Romero-Soriano, Samira Shabanian, and Sina Honari. Benchmarking bias mitigation algorithms in representation learning through fairness metrics. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.  
Amelie Royer and Christoph H Lampert. Classifier adaptation at prediction time. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1401-1409, 2015.  
Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.  
Mhd Hasan Sarhan, Nassir Navab, Abouzar Eslami, and Shadi Albarqouni. Fairness by learning orthogonal disentangled representations. In European Conference on Computer Vision, pp. 746-761. Springer, 2020.  
Laleh Seyyed-Kalantari, Guanxiong Liu, Matthew McDermott, Irene Y Chen, and Marzyeh Ghassemi. Chexclusion: Fairness gaps in deep chest x-ray classifiers. In BIOCOMPUTING 2021: proceedings of the Pacific symposium, pp. 232-243. World Scientific, 2020.  
Laleh Seyyed-Kalantari, Haoran Zhang, Matthew McDermott, Irene Y Chen, and Marzyeh Ghassemi. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature medicine, 27(12):2176-2182, 2021.  
Christine S Spencer, Darrell J Gaskin, and Eric T Roberts. The quality of care delivered to patients within the same hospital varies by insurance type. Health Affairs, 32(10):1731-1739, 2013.  
Cathie Sudlow, John Gallacher, Naomi Allen, Valerie Beral, Paul Burton, John Danesh, Paul Downey, Paul Elliott, Jane Green, Martin Landray, et al. Uk biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. PLoS medicine, 12(3):e1001779, 2015.  
Enzo Tartaglione, Carlo Alberto Barbano, and Marco Grangetto. End: Entangling and disentangling deep representations for bias correction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13508-13517, 2021.  
Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):1-9, 2018.  
Berk Ustun, Yang Liu, and David Parkes. Fairness without harm: Decoupled classifiers with preference guarantees. In International Conference on Machine Learning, pp. 6373-6382. PMLR, 2019.

Vladimir N Vapnik. An overview of statistical learning theory. IEEE transactions on neural networks, 10(5):988-999, 1999.  
Sahil Verma and Julia Rubin. Fairness definitions explained. In 2018 *ieee/acm international workshop on software fairness (fairware)*, pp. 1-7. IEEE, 2018.  
Zeyu Wang, Clint Qinami, Ioannis Christos Karakozis, Kyle Genova, Prem Nair, Kenji Hata, and Olga Russakovsky. Towards fairness in visual recognition: Effective strategies for bias mitigation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8919-8928, 2020.  
David Wen, Saad M Khan, Antonio Ji Xu, Hussein Ibrahim, Luke Smith, Jose Caballero, Luis Zepeda, Carlos de Blas Perez, Alastair K Denniston, Xiaoxuan Liu, et al. Characteristics of publicly available skin cancer image datasets: a systematic review. The Lancet Digital Health, 2021.  
Julia K Winkler, Christine Fink, Ferdinand Toberer, Alexander Enk, Teresa Deinlein, Rainer Hofmann-Wellenhof, Luc Thomas, Aimilios Lallas, Andreas Blum, Wilhelm Stolz, et al. Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. JAMA dermatology, 155 (10):1135-1141, 2019.  
Qizhe Xie, Zihang Dai, Yulun Du, Eduard Hovy, and Graham Neubig. Controllable invariance through adversarial feature learning. Advances in neural information processing systems, 30, 2017.  
Brian Hu Zhang, Blake Lemoine, and Margaret Mitchell. Mitigating unwanted biases with adversarial learning. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pp. 335-340, 2018.  
Haoran Zhang, Natalie Dullerud, Karsten Roth, Lauren Oakden-Rayner, Stephen Pfohl, and Marzyeh Ghassemi. Improving the fairness of chest x-ray classifiers. In Conference on Health, Inference, and Learning, pp. 204-233. PMLR, 2022.  
Han Zhao, Amanda Coston, Tameem Adel, and Geoffrey J Gordon. Conditional learning of fair representations. In International Conference on Learning Representations, 2019.  
Yuyin Zhou, Shih-Cheng Huang, Jason Alan Fries, Alaa Youssef, Timothy J. Amrhein, Marcello Chang, Imon Banerjee, Daniel L. Rubin, Lei Xing, Nigam H. Shah, and Matthew P. Lungren. Radfusion: Benchmarking performance and fairness for multimodal pulmonary embolism detection from ct and ehr. ArXiv, abs/2111.11665, 2021.  
Dominik Zietlow, Michael Lohaus, Guha Balakrishnan, Matthäus Kleindessner, Francesco Locatello, Bernhard Schölkopf, and Chris Russell. Leveling down in computer vision: Pareto inefficiencies in fair deep classifiers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10410-10421, 2022.
