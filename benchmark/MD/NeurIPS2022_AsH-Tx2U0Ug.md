# Effective Backdoor Defense by Exploiting Sensitivity of Poisoned Samples

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Poisoning-based backdoor attacks are serious threat for training deep models on data from untrustworthy sources. Given a backdoored model, we observe that the feature representations of poisoned samples with trigger are more sensitive to transformations than those of clean samples. It inspires us to design a simple sensitivity metric, called feature consistency towards transformations (FCT), to distinguish poisoned samples from clean samples in the untrustworthy training set. Moreover, we propose two effective backdoor defense methods. Built upon a sample-distinguishment module utilizing the FCT metric, the first method trains a secure model from scratch using a two-stage secure training module. And the second method removes backdoor from a backdoored model with a backdoor removal module which alternatively unlearns the distinguished poisoned samples and relearns the distinguished clean samples. Extensive results on three benchmark datasets demonstrate the superior defense performance against eight types of backdoor attacks, to state-of-the-art backdoor defenses.

# 1 Introduction

16 Training deep neural networks (DNNs) often requires a large amount of training data, which is sometimes obtained from a third-party untrustworthy source. However, the untrustworthy data may bring serious security threats. One of the typical threats is the poisoning-based backdoor attack [1], which could inject undesired backdoor—the correlation between trigger(s) and target class(es)—into the model through maliciously poisoning a few training samples. Specifically, as shown in the top left of Fig. 1, each poisoned sample is attached with a trigger (see a small grid patch) at the bottom right corner, and relabelled as a target class. Consequently, the trained backdoored model will predict clean samples very well, but is likely to predict any sample with the trigger to be the target class.

It has been observed in [2] that poisoned samples with triggers are likely to gather together in the feature space of a backdoored model, as shown in the top right of Fig. 1. Note that these poisoned samples contain diverse objects (may be from different source classes), but the information from these objects seems to be ignored by the backdoor model. In other words, the feature representations of poisoned samples are dominated by the triggers, rather than the objects. We conjecture that such a domination is mainly due to the overfitting to the triggers by the backdoor model, since triggers across different poisoned samples are much less diverse than objects. To verify this conjecture, we propose to slightly perturb both poisoned and clean samples, such as rotation transformation. As shown in the bottom right of Fig. 1, there is no longer gathering of poisoned samples in the feature space, and they are located close to samples of their source classes,

i.e., the dominance of triggers over other objects disappears, which verifies the triggers overfitting. Besides, although the feature representations of clean samples are also affected by transformations, their changes are much smaller than those of poisoned samples. In other words, poisoned samples are more sensitive to transformations than clean samples. It inspires that poisoned samples could be distinguished from clean samples according to the sensitivity to transformations, which is measured by a simple sensitivity metric, called feature

consistency towards transformations (FCT). In our experiments, the precision of the distinguished clean and poisoned samples is nearly  $100\%$  in most cases, respectively.

In this work, we aim to obtain a secure model (i.e., high-performance and without backdoor) based on an untrustworthy training set. To this end, we consider two defense paradigms: one is training a secure model from scratch, while the other is firstly training a backdoorsed model using standard supervised learning, and then removing backdoor from the backdoorsed model. Under paradigm 1, we propose an innovative secure training method, called Distinguishment and Secure Training  $(D-ST)$ , which consists of two consecutive modules. The first sample-distinguishment (SD) module splits the whole training set into clean, poisoned and uncertain samples, according to the FCT metric. The second two-stage secure training  $(ST)$  module firstly learns the feature extractor via semi-supervised contrastive learning, and then learns the classifier via minimizing a mixed cross-entropy loss. Under paradigm 2, we propose an innovative backdoor removal method, called Distinguishment and Backdoor Removal  $(D-BR)$ , which consists of the SD module and a backdoor removal  $(BR)$  module. BR module alternatively unlearns the distinguished poisoned samples and learns the distinguished clean samples. Extensive experiments are conducted to verify the superior defense performance of the above two proposed methods, as well as effectiveness of each individual module.

The main contributions of this work are three-folds. (1) We demonstrate the sensitivity of poisoned samples to transformations, which is mainly due to the overfitting to trigger, and propose a simple sensitivity metric to distinguish poisoned samples from clean samples. (2) We propose two effective backdoor defense methods for training a secure model from scratch and removing backdoor from the backdoorsed model, respectively. (3) Extensive experiments on 3 benchmark datasets show the superior performance of the proposed defense methods against 8 widely used backdoor attacks, to 6 state-of-the-art defense methods.

# 2 Related work

Backdoor attack. In poisoning-based backdoor attacks, the attacker attaches a few training samples with trigger(s), and relabel them as target class(es). Existing attacks can be categorized according to a variety of criteria as follows. (1) Size of trigger: Patch-based attacks [1, 5, 6] craft patch-like triggers while in blend-based attacks [7, 8], triggers capture the whole image. (2) Visibility of trigger: Visible attacks [1, 6] design visible but not suspicious triggers while invisible attacks [8, 9, 10] propose invisible and still effective ones. (3) Variability of trigger: Triggers are invariant in sample-agnostic attacks [1, 7, 11] while vary with samples in sample-specific attacks [9, 12]. (4) Label-consistency: If poisoned samples are chosen from samples with target class, then we call these attacks as clean-label attacks [11, 13, 12, 14]. Otherwise, we name them as dirty-label attacks [1, 6, 7, 8]. (5) Number of

![](images/cc744029f6e53390f276c5a47d498128a5acc5a820b88d1250037ac0303c55aa.jpg)  
Figure 2: Framework of two proposed backdoor defense methods for secure training from scratch (paradigm1) and backdoor removal (paradigm2), respectively.

target classes: All2one attacks [1, 6, 7, 8] designate one class as the target class while in all2all attacks [1], poisoned samples are relabelled as the next class. There are also other attacks [15, 16, 17, 18, 19] that require the attacker to control the training process, which are out of scope of this paper.

Backdoor defense. In general, there are two types of defense paradigms against poisoning-based backdoor attacks—secure training and backdoor removal. Mainstream defense methods belong to the latter one, which leverage the model properties [20, 21, 22] or the feature space characteristics [23, 5, 24, 25, 26, 27, 28] of a backdoorsed model, to remove the hidden backdoor. For instance, FP [20] observes that some neurons are activated by poisoned samples while others are by clean samples. AC [23] notices the difference in size between the cluster of clean target samples and that of poisoned samples. So far, there is still few works for the former defense. DBD [2] observes that poisoned samples gather together in the feature space, and proposes a three-stage defense method to inhibit the gathering. The first stage is training the feature extractor with self-supervised learning. The second stage is training the classifier with supervised learning, where the backdoor may be injected. The third stage is identifying clean samples by the symmetric cross-entropy and fine-tuning the model with semi-supervised learning on labelled clean samples and label-removed remaining samples.

# 3 Proposed method

# 3.1 Problem formulation

Threat model. In this paper, we consider the threat model of poisoning-based backdoor attacks, where the attacker can manipulate a few samples in the original clean training set  $D_{train} = D_c \cup \{(\pmb{x}_i, y_i)\}_{i=1}^{m_p}$ , with  $D_c = \{(\pmb{x}_i, y_i)\}_{i=1}^{m_c}$  indicating the unmanipulated subset. For the remaining  $m_p$  samples, each sample  $\pmb{x}_i \in \mathcal{X}$  is fused with a trigger  $\delta$  to form a poisoned sample  $\bar{\pmb{x}}_i = \pmb{x}_i \oplus \delta$  with  $\oplus$  being the fusion operator. Meanwhile, its label  $y_i \in \mathcal{Y}$  is also changed to a target class  $t$ . Then, a poisoned training set is constructed, denoted as  $\bar{D}_{train} \equiv D_c \cup D_p$ , with  $D_p = \{(\bar{\pmb{x}}_i, t)\}_{i=1}^{m_p}$ . When a user downloads  $\bar{D}_{train}$  and trains a DNN classifier  $g_\theta : \mathcal{X} \to \mathcal{Y}$  based on  $\bar{D}_{train}$  using the standard supervised learning algorithm, it may learn a undesired backdoor, i.e., a stable mapping from the trigger  $\delta$  to the target class  $t$ . Consequently, for any new sample with the trigger  $\delta$ , it is likely to be predicted as the target class  $t$ . Note that the user does not know which sample is poisoned or clean.

Defense goal. Given the poisoned training set  $\bar{D}_{train}$ , the defender aims to obtain a high-performance model  $g_{\theta}$  without backdoor, i.e., a secure model. In this work, we consider two different paradigms:

- Paradigm 1: a secure model is directly trained from scratch, as described in Section 3.3.  
- Paradigm 2: a backdoored model is firstly trained using the standard supervised learning, then the backdoor is removed from the backdoored model, as described in Section 3.4.

# 3.2 Sensitivity of poisoned samples

Sensitivity metric. As illustrated in Section 1 and Fig. 1, we have found that the poisoned samples are much more sensitive to transformations than the clean samples in a backdoored model. To accurately measure such a difference, we propose a simple metric, feature consistency towards transformations (FCT). Specifically, given a backdoored model  $g_{\theta}$  trained on  $\bar{D}_{train}$  with  $f_{\theta_e}(\cdot)$  indicating its feature extractor, and a set of transformations  $\tau$  (e.g., rotation, scaling, will be specified in experiments), for any sample  $x$  (poisoned or clean), the FCT metric is formulated as follows:

$$
\Delta_ {t r a n s} (\boldsymbol {x}; \tau , f _ {\boldsymbol {\theta} _ {e}}) = \| f _ {\boldsymbol {\theta} _ {e}} (\boldsymbol {x}) - f _ {\boldsymbol {\theta} _ {e}} (\tau (\boldsymbol {x})) \| _ {2} ^ {2}. \tag {1}
$$

It measures the change of the feature representation due to the transformations  $\tau$ . If  $\Delta_{trans}(\pmb{x};\tau,f)$  is large, then it means that  $\pmb{x}$  is sensitive to  $\tau$ , otherwise stable. For clarity, we use  $\Delta_{trans}(\pmb{x})$  hereafter.

Sample-distinguishment module. Utilizing FCT, we develop a sample-distinguishment (SD) module. Specifically, we firstly train a backdoored model  $g_{\theta}$  based on  $\bar{D}_{train}$  using the standard supervised learning algorithm with a few epochs (explained in Appendix A.1). Then, we calculate  $\Delta_{trans}(\boldsymbol{x}_i), \forall \boldsymbol{x}_i \in \bar{D}_{train}$ , and plot the histogram. As shown in Fig. 3, where two representative backdoor attacks are evaluated, there is remarkable difference on the distribution between the poisoned and the clean samples in both histograms. It demonstrates that  $\Delta_{trans}$  is a good metric to distinguish the poisoned sam

![](images/f438b6bfcd3454fba69aa0a077c438024ac94aa89cebd8d13143edf006575cda.jpg)  
(a) BadNets

![](images/97f5bb016decb01cbac076461595ea7484ca06a55b9ac08d9c7c2572a477e984.jpg)  
Figure 3: Distribution of clean and poisoned samples with respect to the FCT metric on CIFAR-10.  
(b) Blend

plies from the clean samples in  $\bar{D}_{train}$ . Based on the sensitivity histogram, we set two proportion values  $\alpha_{c},\alpha_{p}\in [0,1]$ . The samples with the bottom- $\alpha_{c}$ $\Delta_{trans}$  values are separated to a subset of clean samples  $\hat{D}_c$ , while those with the top- $\alpha_{p}$ $\Delta_{trans}$  values are separated into a subset of poisoned samples  $\hat{D}_p$ , while the remaining samples are partitioned as an uncertain subset denoted as  $\hat{D}_u$ . We have  $\bar{D}_{train} = \hat{D}_c\cup \hat{D}_p\cup \hat{D}_u$ . More details are in Algorithm 1 in Appendix A.1

# 3.3 Method for paradigm 1: secure training from scratch

Here, we consider the backdoor defense under paradigm 1. We propose an innovative secure training method, called Distinguish and Secure Training (D-ST) method. As illustrated in Fig. 2, D-ST consists of the SD module (see above) and a two-stage secure training (ST) module, which is described as follows. Details of the D-ST method are summarized in Algorithm 3 in Appendix A.2..

Stage 1: learning feature extractor via semi-supervised contrastive learning (SSCL). Our method is inspired by a recent backdoor defense method called DBD [2], which proposed to learn a good feature extractor  $f_{\theta_e}$  based on  $\bar{D}_{train}$  using a self-supervised learning algorithm, i.e., contrastive learning (CTL). Consequently, the feature representations of samples with similar appearances will be similar, and poisoned samples with triggers cannot gather together to form the backdoor. Note that all labels have been abandoned in DBD before the extractor learning since there is no way to identify poisoned samples, leading to the waste of the valuable information contained in clean samples. Fortunately, the proposed SD module could identify some clean samples. Thus, inspired by the supervised contrastive learning (S-CTL) [29], which has shown to learn a feature extractor with better performance than CTL, we propose a novel learning called semi-supervised contrastive learning (SS-CTL), to learn  $f_{\theta_e}$  by minimizing the following loss function:

$$
\begin{array}{l} \mathcal {L} _ {S S - C T L} \left(\boldsymbol {\theta} _ {e}; \bar {D} _ {t r a i n}\right) = \sum_ {\left(\boldsymbol {x} _ {i}, y _ {i}\right) \in \hat {D} _ {p} \cup \hat {D} _ {u}} \ell_ {C T L} \left(f _ {\boldsymbol {\theta} _ {e}} \left(\tilde {\boldsymbol {x}} _ {i} ^ {(1)}\right), f _ {\boldsymbol {\theta} _ {e}} \left(\tilde {\boldsymbol {x}} _ {i} ^ {(2)}\right)\right) \tag {2} \\ + \sum_ {\{(\boldsymbol {x} _ {i}, y _ {i}), (\boldsymbol {x} _ {j}, y _ {j}) \} \subset \dot {D} _ {c}} \ell_ {S - C T L} \big (f _ {\boldsymbol {\theta} _ {e}} (\tilde {\boldsymbol {x}} _ {i} ^ {(1)}), f _ {\boldsymbol {\theta} _ {e}} (\tilde {\boldsymbol {x}} _ {i} ^ {(2)}), f _ {\boldsymbol {\theta} _ {e}} (\tilde {\boldsymbol {x}} _ {j} ^ {(1)}), f _ {\boldsymbol {\theta} _ {e}} (\tilde {\boldsymbol {x}} _ {j} ^ {(2)}); y _ {i}, y _ {j} \big), \\ \end{array}
$$

where the contrastive loss  $\ell_{CTL}$  encourages the two augmented versions  $\tilde{\pmb{x}}_i^{(1)},\tilde{\pmb{x}}_i^{(2)}$  (e.g., cropping, details are introduced in Appendix C) of a sample  $\pmb {x}_i$  to be close in the feature space, while the supervised contrastive loss  $\ell_{S - CTL}$  additionally encourages the feature representations of two clean augmented samples from the same class to be close. In this work, we instantiate  $\ell_{CTL}$  as the contrastive loss defined in [30] and  $\ell_{S - CTL}$  as the SupCon loss defined in [29].

Stage 2: learning classifier via minimizing the mixed cross-entropy loss. Given the feature extractor  $f_{\theta_e}$  learned in stage 1, we then learn the classifier  $h_{\theta_c}$  by minimizing the following mixed cross-entropy (MCE) loss:

$$
\mathcal {L} _ {M C E} \left(\boldsymbol {\theta} _ {c}; \hat {D} _ {c}, \hat {D} _ {p}\right) = \frac {- 1}{| \hat {D} _ {c} |} \sum_ {(\boldsymbol {x}, y) \in \hat {D} _ {c}} \log \left[ h _ {\boldsymbol {\theta} _ {c}} \left(f _ {\boldsymbol {\theta} _ {e}} (\boldsymbol {x})\right) \right] _ {y} + \frac {\lambda_ {p}}{| \hat {D} _ {p} |} \cdot \sum_ {(\boldsymbol {x}, y) \in \hat {D} _ {p}} \log \left[ h _ {\boldsymbol {\theta} _ {c}} \left(f _ {\boldsymbol {\theta} _ {e}} (\boldsymbol {x})\right) \right] _ {y}, \tag {3}
$$

where the first term is the standard cross-entropy loss defined based on the distinguished clean samples  $\hat{D}_c$ , while the second term is the negative cross-entropy loss defined based on the distinguished poisoned samples  $\hat{D}_p$ , which is used to eliminate the effect of poisoned samples.  $\lambda_p \in \mathbb{R}^+$  is a trade-off parameter between two losses.

# 3.4 Method for paradigm 2: backdoor removal

Here we consider the backdoor defense under paradigm 2. We propose an innovative backdoor removal method, called Distinguishment and Backdoor Removal (D-BR) method. As illustrated in Fig. 2, D-BR consists of the SD module (see Section 3.2) and a backdoor removal (BR) module. The BR module aims to remove the backdoor from the backdoored model, i.e., the backdoor is no longer activated by the trigger, while keeping the high performance on clean samples. To this end, the BR module implements an iterative learning algorithm, which consists of two alternating steps, i.e., unlearning and relearning. The D-BR method is summarized in Algorithm 4 in Appendix A.3.

Unlearning. This step aims to eliminate the effect of the trigger, through unlearning [31] the poisoned samples in  $\hat{D}_p$  distinguished by the SD module, as follows:

$$
\mathcal {L} _ {\text {u n l e a r n}} (\boldsymbol {\theta}; \hat {D} _ {p}) = \frac {1}{| \hat {D} _ {p} |} \sum_ {(\boldsymbol {x}, y) \in \hat {D} _ {p}} \log [ g _ {\boldsymbol {\theta}} (\boldsymbol {x}) ] _ {y}. \tag {4}
$$

Relearning. After conducting the above unlearning step for one epoch, although the effect of poisoned samples is somewhat eliminated, in experiments we find that the performance on clean samples is also degraded to some extent. Thus, we want to relearn the mapping from the clean objects to the ground-truth classes based on clean samples in  $\hat{D}_c$  distinguished by the SD module, as follows:

$$
\mathcal {L} _ {\text {r e l e a r n}} (\boldsymbol {\theta}; \hat {D} _ {c}) = \frac {1}{| \hat {D} _ {c} |} \sum_ {(\boldsymbol {x}, y) \in \hat {D} _ {c}} - \log [ g _ {\boldsymbol {\theta}} (\boldsymbol {x}) ] _ {y}. \tag {5}
$$

Note that both unlearning and relearning are run for one epoch in each round.

# 4 Experiments

# 4.1 Experimental settings

Attack configurations. According to the taxonomy described in Section 2, we consider 8 typical poisoning-based backdoor attacks by choosing at least one method from each category, including: BadNets [1] using two attack types (BadNets-all2one, BadNets-all2all), Trojan backdoor attack [6] (Trojan), Blend backdoor attack using two different patterns (Blend-Signal, Blend-Kitty) [7], Clean-label backdoor (CL) [13], Sinusoidal signal backdoor attack (SIG) [11], Sample-specific backdoor attack (SSBA) [9]. We evaluate all attacks on 3 benchmark datasets, CIFAR-10 [3], CIFAR-100 [3] and an ImageNet subset [32, 9], with ResNet-18 [33] as the base model. Poisoning rate is set to  $10\%$

in all attacks. Due to the space limit, more implementations details about attacks can be found in Appendix C.3, and results on the ImageNet subset are shown in Appendix D.

Defense configurations. We first compare the proposed D-ST method with DBD [2]. Since studies with this secure-training paradigm are limited, we additionally add 2 baselines for comparison which are detailed in Section 4.2. We then compare the proposed D-BR method with 5 state-of-the-art methods with the same backdoor-removal paradigm: the standard fine-tuning FT, ANP [21], NAD [34], MCR [35] and ABL [36]. For methods requiring extra clean data,  $1\%$  of the clean training samples are provided. Other configurations are set as clarified in the original papers. In summary, we consider 6 state-of-the-art defense methods and 2 additional baselines. More implementations details can be found in Appendix C.4. For our proposed methods, we use  $\alpha_{c} = 20\%$ ,  $\alpha_{p} = 5\%$  and  $\tau = \text{rotate + affine}$  in all experiments. Other details can be seen in Appendix C.5.

Evaluation metrics. We evaluate the defense performance adopting two commonly used metrics: accuracy on clean samples (ACC) and attack success rate (ASR), i.e., accuracy of predicting poisoned samples as the target label.

# 4.2 Experimental results

Effectiveness of D-ST method. We first consider paradigm 1—secure training from scratch. Performance of different defense methods against various attacks on CIFAR-10 and CIFAR-100 is demonstrated in Table 1. An ideal defense method is supposed to increase ACC while keep ASR as low as possible. Thus, a larger ACC-ASR indicates a better method. We mark the best result in boldface. Note that we only report results on successful attacks where ASR is higher than  $85\%$ .

Table 1: Comparisons of the D-ST method with 3 secure-training defense methods  $(\%)$  

<table><tr><td rowspan="2">Dataset ↓</td><td rowspan="2">Defense → Attack ↓</td><td colspan="2">Baseline1</td><td colspan="2">Baseline2</td><td colspan="2">DBD</td><td colspan="2">D-ST</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td rowspan="8">CIFAR-10</td><td>BN-all2one</td><td>83.54</td><td>2.60</td><td>91.32</td><td>99.91</td><td>92.75</td><td>100.00</td><td>92.77</td><td>0.03</td></tr><tr><td>BN-all2all</td><td>83.95</td><td>2.72</td><td>91.59</td><td>57.39</td><td>92.95</td><td>75.21</td><td>89.22</td><td>2.05</td></tr><tr><td>Trojan</td><td>83.77</td><td>5.24</td><td>93.63</td><td>99.98</td><td>92.81</td><td>100.00</td><td>93.72</td><td>0.00</td></tr><tr><td>Blend-Strip</td><td>85.36</td><td>99.93</td><td>94.19</td><td>100.00</td><td>94.21</td><td>99.98</td><td>93.59</td><td>0.00</td></tr><tr><td>Blend-Kitty</td><td>85.03</td><td>99.99</td><td>94.31</td><td>100.00</td><td>93.32</td><td>100.00</td><td>91.82</td><td>0.00</td></tr><tr><td>SIG</td><td>85.14</td><td>99.02</td><td>94.37</td><td>99.93</td><td>94.37</td><td>99.71</td><td>90.07</td><td>0.00</td></tr><tr><td>CL</td><td>85.79</td><td>10.76</td><td>94.58</td><td>98.87</td><td>94.32</td><td>99.87</td><td>90.46</td><td>6.40</td></tr><tr><td>Avg</td><td>84.65</td><td>45.75</td><td>93.43</td><td>93.73</td><td>93.53</td><td>96.40</td><td>91.66</td><td>1.21</td></tr><tr><td rowspan="5">CIFAR-100</td><td>BN-all2one</td><td>54.48</td><td>10.41</td><td>67.62</td><td>100.00</td><td>69.08</td><td>100.00</td><td>68.43</td><td>0.12</td></tr><tr><td>Trojan</td><td>56.17</td><td>12.76</td><td>71.01</td><td>100.00</td><td>72.18</td><td>99.99</td><td>68.04</td><td>0.08</td></tr><tr><td>Blend-Strip</td><td>58.01</td><td>99.91</td><td>72.47</td><td>99.99</td><td>71.29</td><td>99.99</td><td>67.63</td><td>0.00</td></tr><tr><td>Blend-Kitty</td><td>57.21</td><td>99.99</td><td>73.36</td><td>99.99</td><td>72.43</td><td>100.00</td><td>67.06</td><td>0.00</td></tr><tr><td>Avg</td><td>56.47</td><td>55.77</td><td>71.12</td><td>100.00</td><td>71.24</td><td>99.99</td><td>67.79</td><td>0.05</td></tr></table>

DBD fails in most attacks on CIFAR-10 and CIFAR-100, probably due to the failure of the symmetric cross-entropy to distinguish samples. By comparison, the good performance reached by D-ST illustrates the accurate distinguishment from the FCT-based SD module. We additionally introduce two feasible baselines without requiring special knowledge. Baseline1 first uses SimCLR [30] to train the feature extractor and then trains the classifier on the poisoned dataset with standard supervised learning. By comparison, Baseline2 leverages S-CTL [29] to train the feature extractor. We focus on discussing the effect of different extractor-training algorithms on defense performance. More discussions are in the later experiments. Baseline 1 reveals that training extractor without labels may result in low ASR (<5% / <20% in some cases on CIFAR-10 / CIFAR-100), but will sacrifice ACC definitely (84.65% / 56.47% on average). While Baseline 2 demonstrates that training extractor with all labels guarantees high ACC (93.43% / 71.12% on average), but also brings high ASR (93.73% / 100% on average). By contrast, results of D-ST illustrate the effectiveness of ST module in training the feature extractor in a secure way since ACC is high (91.66% / 67.79% on average) and ASR (1.21% / 0.05% on average) is extremely low.

Effectiveness of D-BR method. Then, we consider paradigm 2—backdoor removal. Defense performance of different defense methods against various attacks on CIFAR-10 and CIFAR-100 is demonstrated in Table 2. Results on ImageNet is shown in Table 5 in Appendix D.

Table 2: Comparisons of the D-BR method with 5 backdoor-removal defense methods on CIFAR-10 and CIFAR-100  $(\%)$ . 'Backdoored' refers to the backdoored model. * denotes methods which require a few  $(1\%)$  clean training samples.  

<table><tr><td rowspan="2">Dataset ↓</td><td rowspan="2">Defense → Attack ↓</td><td colspan="2">Backdoored</td><td colspan="2">FT*</td><td colspan="2">ANP*</td><td colspan="2">NAD*</td><td colspan="2">MCR*</td><td colspan="2">ABL</td><td colspan="2">D-BR</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td rowspan="8">CIFAR-10</td><td>BN-all2one</td><td>91.64</td><td>100.00</td><td>88.99</td><td>66.79</td><td>90.03</td><td>10.54</td><td>84.46</td><td>2.13</td><td>94.21</td><td>8.29</td><td>89.36</td><td>0.19</td><td>92.83</td><td>0.40</td></tr><tr><td>BN-all2all</td><td>92.79</td><td>88.01</td><td>90.31</td><td>4.96</td><td>86.04</td><td>1.47</td><td>84.97</td><td>1.71</td><td>92.17</td><td>2.96</td><td>79.91</td><td>78.16</td><td>92.61</td><td>0.56</td></tr><tr><td>Trojan</td><td>91.91</td><td>100.00</td><td>89.86</td><td>100.00</td><td>90.89</td><td>0.81</td><td>83.29</td><td>5.04</td><td>93.90</td><td>2.58</td><td>90.18</td><td>0.23</td><td>92.21</td><td>0.76</td></tr><tr><td>Blend-Strip</td><td>92.09</td><td>99.97</td><td>89.91</td><td>93.50</td><td>88.33</td><td>0.04</td><td>83.09</td><td>13.30</td><td>91.77</td><td>17.96</td><td>58.46</td><td>0.22</td><td>92.40</td><td>0.06</td></tr><tr><td>Blend-Kitty</td><td>92.69</td><td>99.99</td><td>90.47</td><td>99.31</td><td>84.07</td><td>0.01</td><td>84.54</td><td>28.96</td><td>94.42</td><td>7.49</td><td>79.20</td><td>2.27</td><td>92.11</td><td>0.14</td></tr><tr><td>SIG</td><td>92.88</td><td>99.69</td><td>90.81</td><td>99.87</td><td>82.43</td><td>76.32</td><td>81.00</td><td>64.72</td><td>91.82</td><td>99.04</td><td>79.94</td><td>98.84</td><td>92.73</td><td>0.24</td></tr><tr><td>CL</td><td>93.20</td><td>93.34</td><td>90.03</td><td>77.44</td><td>72.57</td><td>10.90</td><td>84.46</td><td>2.66</td><td>92.13</td><td>72.01</td><td>84.39</td><td>0.31</td><td>92.08</td><td>0.00</td></tr><tr><td>Avg</td><td>92.46</td><td>97.29</td><td>90.05</td><td>77.41</td><td>84.91</td><td>14.30</td><td>83.69</td><td>16.93</td><td>92.92</td><td>30.05</td><td>80.21</td><td>25.75</td><td>92.42</td><td>0.31</td></tr><tr><td rowspan="5">CIFAR-100</td><td>BN-all2one</td><td>71.23</td><td>99.13</td><td>70.81</td><td>66.28</td><td>65.42</td><td>0.00</td><td>69.03</td><td>11.41</td><td>73.38</td><td>0.27</td><td>66.47</td><td>0.02</td><td>72.58</td><td>0.25</td></tr><tr><td>Trojan</td><td>75.75</td><td>100.00</td><td>74.21</td><td>99.94</td><td>64.52</td><td>0.03</td><td>72.11</td><td>92.21</td><td>74.51</td><td>0.12</td><td>68.12</td><td>0.00</td><td>74.52</td><td>0.00</td></tr><tr><td>Blend-Strip</td><td>75.54</td><td>99.99</td><td>73.36</td><td>99.65</td><td>67.38</td><td>0.00</td><td>71.18</td><td>95.78</td><td>73.37</td><td>0.07</td><td>49.13</td><td>0.00</td><td>74.35</td><td>0.00</td></tr><tr><td>Blend-Kitty</td><td>75.18</td><td>99.97</td><td>72.93</td><td>99.96</td><td>69.03</td><td>0.00</td><td>71.73</td><td>99.93</td><td>73.93</td><td>20.60</td><td>47.05</td><td>0.00</td><td>72.00</td><td>0.01</td></tr><tr><td>Avg</td><td>74.43</td><td>99.77</td><td>72.83</td><td>91.46</td><td>66.59</td><td>0.01</td><td>71.01</td><td>74.83</td><td>73.80</td><td>5.27</td><td>57.69</td><td>0.01</td><td>73.36</td><td>0.07</td></tr></table>

Results on CIFAR-10. We discover that except for FT and MCR, other selected methods generally reduce ACC markedly. Additionally, they have two common disadvantages. (1) They can not take effect on all attacks. For example, ANP can defend against Trojan and Blend attacks  $(\mathrm{ASR} < 1\%)$  while fails in clean-label attacks, i.e. SIG and CL  $(\mathrm{ASR} > 10\%)$ . (2) There exists at least one attack that can disable the methods  $(\mathrm{ASR} > 50\%)$ . By contrast, the proposed D-BR method overcomes these drawbacks. It not only maintains ACC as large as that of backdoored model, but also reduces ASR to less than  $1\%$  on all attacks, verifying the effectiveness of the BR module and the high precision of the distinguishment conducted by the SD module.

Results on CIFAR-10. Although FT and NAD have a relatively high ACC  $(>68\%)$ , they fail to reduce ASR (91.46% and 74.83% on average). While ANP and ABL can decrease ASR to less than 0.1%, they sacrifice too much ACC (66.59% and 57.69% on average). Among the selected methods, MCR performs the best (ACC = 73.80% on average, ASR < 0.5% in three cases), but it still fails to defend against the Blend-Kitty attack (ASR = 20.60%). Note that MCR requires extra clean data. In contrast, D-BR keeps ACC higher than 72%, while reduces ASR to almost 0% without any extra clean data.

# 4.3 Ablation studies

Effectiveness of the SD module. Here, we aim to study the effectiveness of the SD module. Specifically, we will show how our proposed FCT metric, performs better than other metrics, under the backdoor-removal paradigm for illustration. To this end, we select three existing metrics for comparison. Spectral signatures [5] specifies the metric as the correlation with the top singular vector of the covariance matrix of feature representations. DBD [2] assigns symmetric cross-entropy loss as the metric. The metric used in ABL [36] is loss value applied with local gradient ascent. The metric values of clean samples are smaller than those of poisoned samples according to the former two metrics, while larger for the third metric. For fair comparison, we uniformly set  $\alpha_{c} = 20\%$ ,  $\alpha_{p} = 5\%$ . We first apply the metric-replaced SD module on the poisoned training set, and then conduct the BR module based on the distinguished samples. Results are shown in Fig. 4.

The height of the blue bar above the orange bar suggests how well the metric could distinguish. As shown in Fig. 4 (a,b,d), orange bars are all higher than blue bars for Spectral signatures and DBD, indicating metrics of which fail to distinguish in BadNets, Blend and CL attacks. In contrast, the

![](images/94896e3ba171f1e392b42679824fd5f447614a323a9c57da0be67a74f9e6ea93.jpg)  
(a) BadNets

![](images/7ae1a8edc1c336547e90a9044519b586c36b437a3d96489b151049e8162cff5d.jpg)  
Figure 4: Test ACC and Test ASR of four metric-replaced D-BR methods on the poisoned CIFAR-10.  
(b) Blend

![](images/f5f378b1e15ab0b6c03030636248348d12b6812ce170d27f3a96d9391bd33d55.jpg)  
(c) SIG

![](images/9c15c45fdd7538141ef1704a5b5d22390c864a4580462af0d9cd7476166177fa.jpg)  
(d) CL

metric of ABL is reliable since ABL performs well in most cases except for Blend attack where ASR is  $24.72\%$ . By comparison, our proposed FCT metric could distinguish samples stably well, resulting in extremely low ASR  $(< 0.5\%)$  on all attacks. We attribute the success to that FCT exploits the sensitivity of poisoned samples, which is mainly due to the overfitting to trigger by the backdoored model that exists in all backdoor attacks we have evaluated in this paper.

Effectiveness of the BR module. Here, we focus on studying the effectiveness of the BR module. Specifically, we aim to show how the iterative learning algorithm consisting of unlearning and relearning performs better than the pure unlearning adopted by [36] or pure relearning. To this end, we first conduct the SD module, and then apply different learning algorithms. For the three algorithms, we run 20 epochs on CIFAR-10 and record the variations of Test ACC and Test ASR which are illustrated as Fig. 5.

![](images/3021bc103d67337a8b32c01ca7db2611c17dff5f03b4f68f0cff2a8d66b28522.jpg)

![](images/adca942d170a74ab5252a34fdd92e21d3c67628e18c57f0375928d6bfbf9b585.jpg)  
(a) BadNets

![](images/4443eca3ae29b2a6992896feb1de0f08520c6e3e1cc51e38fdd93108ddf29b2b.jpg)

![](images/5d4c1e5948c944f1c1789dc1f2323ee7ba1f649b0997dc58efd4940eacd8461c.jpg)  
(b) Blend

![](images/4c0afd1a1c2df2fafa7243c6a00839e82a21317ff677d5a913864de81bb98b5a.jpg)

![](images/055a1e3fa5a23300a69d8fe96c0d39132f98202459f7b2ebc29eb805fc8edfcd.jpg)  
Figure 5: Test ACC(top) and Test ASR(bottom) of three learning algorithms on poisoned CIFAR-10.  
(c) SIG

![](images/cfd6f716713df07fbb7b28779f427c396a9ca12adcd774289794e6b16cabe89b.jpg)

![](images/52ce0754f0f6caef38d9dd6fec33b5cefc9f1739cab521c03074f7f0d4309975.jpg)  
(d) CL

Although pure unlearning (red lines) effectively decreases ASR, it could hardly maintain ACC, showing a downward trend. The results indicate a strict requirement for choosing the number of unlearning epochs. On the contrary, pure relearning (green lines) can keep ACC stably high, but it takes tiny effect in reducing ASR. By contrast, unlearning + relearning (blue lines) combines their advantages and successfully diminishes ASR while maintains ACC. ACC and ASR steadily converge to high and low values, respectively, validating the effectiveness and the stability of the BR module.

Effectiveness of the ST module. Here, we aim to study the effectiveness of the ST Module. Backdoor can be injected during training the feature extractor  $f_{\theta_e}$  and the classifier  $h_{\theta_c}$ . The final defense performance of  $g_{\theta}$  depends on how well  $f_{\theta_e}$  and  $h_{\theta_c}$  inhibit backdoor.

Firstly, we want to show how SS-CTL performs better than CTL or S-CTL in training a secure feature extractor  $f_{\theta_e}$ . To this end, we train  $f_{\theta_e}$  with different learning algorithms and then uniformly leverage  $\mathcal{L}_{MCE}$  to train  $h_{\theta_c}$ . Training  $f_{\theta_e}$  with CTL, as shown in the first row, guarantees low ASR, but the low ACC turns into a tradeoff. Note that the low ASR is the joint effort of CTL and  $\mathcal{L}_{MCE}$ . And the usage of CTL does not indicate low ASR definitely, but it indeed reduces the possibility of backdoor injection in  $f_{\theta_e}$ . The second row illustrates that S-CTL could bring high ACC and the potentially high ASR, as seen in the SIG and CL attacks. Since all labels (including poisoned) are used in this scenario, ACC is reasonably high. Besides, backdoor is already injected into  $f_{\theta_e}$ . But due to the inhibition effect of  $\mathcal{L}_{MCE}$ , the ASR of  $g_{\theta}$  may not be high. For example, in dirty-label attacks, i.e. BadNets, Trojan and Blend attacks, ASR is almost 0%. While in clean-label attacks, i.e. SIG and CL attacks,  $\mathcal{L}_{MCE}$  can not withstand the backdoor injected in  $f_{\theta_e}$ , so the ASR is almost 100%. Hence, in order to establish a reliable defense module,  $f_{\theta_e}$  should be trained in a more secure way. In comparison, the third row demonstrates the superior defense performance of SS-CTL, illustrating that the ST module securely bridges genuinely clean intra-class samples together which are distinguished by the SD module.

Table 3: Performance with  $f_{\theta_e}$  trained with three learning algorithms on the poisoned CIFAR-10.  

<table><tr><td rowspan="2">Attack → fo. ↓</td><td colspan="2">BN-all2one</td><td colspan="2">BN-all2all</td><td colspan="2">Trojan</td><td colspan="2">Blend-Signal</td><td colspan="2">Blend-Kitty</td><td colspan="2">SIG</td><td colspan="2">CL</td></tr><tr><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td><td>ACC</td><td>ASR</td></tr><tr><td>CTL</td><td>85.63</td><td>1.52</td><td>83.02</td><td>1.65</td><td>85.03</td><td>1.32</td><td>85.12</td><td>0.00</td><td>83.49</td><td>0.00</td><td>83.10</td><td>0.00</td><td>83.77</td><td>4.88</td></tr><tr><td>S-CTL</td><td>92.98</td><td>0.00</td><td>93.73</td><td>0.73</td><td>93.80</td><td>0.00</td><td>94.09</td><td>0.00</td><td>94.18</td><td>0.00</td><td>94.51</td><td>99.77</td><td>94.67</td><td>98.34</td></tr><tr><td>SS-CTL</td><td>92.77</td><td>0.03</td><td>89.22</td><td>2.05</td><td>93.72</td><td>0.00</td><td>93.59</td><td>0.00</td><td>91.82</td><td>0.00</td><td>90.07</td><td>0.00</td><td>90.46</td><td>6.40</td></tr></table>

Secondly, we explore how  $\mathcal{L}_{MCE}$  affects the defense performance of  $h_{\theta_c}$ . For clarity, we denote  $\mathcal{L}_1 \equiv \frac{-1}{|\hat{D}_c|} \sum_{(\boldsymbol{x},y) \in \hat{D}_c} \log[h_{\theta_c}(f_{\theta_e}(\boldsymbol{x}))]_y$  and  $\mathcal{L}_2 \equiv \frac{1}{|\hat{D}_p|} \cdot \sum_{(\boldsymbol{x},y) \in \hat{D}_p} \log[h_{\theta_c}(f_{\theta_e}(\boldsymbol{x}))]_y$ . We have  $\mathcal{L}_{MCE} = \mathcal{L}_1 + \lambda_p \mathcal{L}_2$ . Generally, if knowing clean samples, the defender will train  $h_{\theta_c}$  with  $\mathcal{L}_1$ . So here, we aim to show how our proposed  $\mathcal{L}_2$  and the trade-off parameter  $\lambda_p$  affect  $h_{\theta_c}$ . To this end, we first fix  $f_{\theta_e}$  learned by SS-CTL and then apply  $\mathcal{L}_{MCE}$  with  $\lambda_p = 0, 0.001, 0.01, 0.1, 1$  on  $h_{\theta_c}$ . Results are shown in Fig. 6. In the previous experiments, we adopt  $\lambda_p = 0.001$ .

The comparison between  $\lambda_p = 0$  and  $\lambda_p \neq 0$  in the right figure illustrates that  $\mathcal{L}_2$  can effectively reduce ASR. When comparing different  $\lambda_p \neq 0$  in the left figure, we discover that as  $\lambda_p$  increases, there is a trend of decrease in ACC. We infer that since  $\mathcal{L}_2$  drops faster than  $\mathcal{L}_1$ , namely unlearning is faster than relearning, adding weights to  $\mathcal{L}_2$  makes  $h_{\theta_c}$  focus on unlearning instead of relearning, leading to the low ACC. Therefore, we conclude that  $\mathcal{L}_2$  helps to inhibit backdoor in

![](images/75d3e8bd03b6d0fe2de86acb1bdb7a083aeca61fbc114281f3764af9d222203c.jpg)  
Figure 6: Test ACC (left) and Test ASR (right) under various  $\lambda_{p}$  on the poisoned CIFAR-10.

$h_{\theta_c}$ , but its weight should not be too large.  $\lambda_p = 0.001$  is considered to be an appropriate choice.

In summary, we have empirically validated the effectiveness of each individual module, and shown the flexibility of our method to combine with other existing modules.

Appendix. Due to the space limit, more results and analysis will be presented in Appendix, including: (1) performance with different data transformations  $\tau$  in Appendix E; (2) performance with different proportion values  $\alpha_{c}$ ,  $\alpha_{p}$  in Appendix F; (3) performance with different poisoning rates in Appendix G; (4) complexities of two proposed methods in Appendix H.

# 5 Conclusions

In this paper, we reveal the sensitivity of poisoned samples to transformations and propose a sensitivity metric, called FCT. Besides, we propose three modules—the SD module to distinguish between clean and poisoned samples, the ST module to train a secure model from scratch and the BR module to remove backdoor—which constitute two defense methods, i.e. D-ST and D-BR, to defend under two different defense paradigms. Extensive experiments have demonstrated the effectiveness of each individual module and also the proposed defense methods.

# 6 Broader impact

Poisoning-based backdoor attacks are severe threats to the learning paradigm of learning a DNN model based on the training set from some untrustworthy sources. This work reveals the sensitivity of poisoned samples in the backdoored model, which will help people to better understand the inner mechanism of backdoor attacks. The proposed two effective defense methods can not only significantly mitigate the threat of existing poisoning based backdoor attacks, but also serve as the new baseline for developing more advanced attack methods in future.

# References

[1] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230-47244, 2019.  
[2] Kunzhe Huang, Yiming Li, Baoyuan Wu, Zhan Qin, and Kui Ren. Backdoor defense via decoupling the training process. In ICLR, 2022.  
[3] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical Report, University of Toronto, 2009.  
[4] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 2008.  
[5] Brandon Tran, Jerry Li, and Aleksander Madry. Spectral signatures in backdoor attacks. In NIPS, 2018.  
[6] Yingqi Liu, Shiqing Ma, Yousra Aafer, Wen-Chuan Lee, Juan Zhai, Weihang Wang, and Xiangyu Zhang. Trojaning attack on neural networks. In NDSS, 2017.  
[7] Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526, 2017.  
[8] Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu. Reflection backdoor: A natural backdoor attack on deep neural networks. In ECCV, 2020.  
[9] Yuezun Li, Yiming Li, Baoyuan Wu, Longkang Li, Ran He, and Siwei Lyu. Invisible backdoor attack with sample-specific triggers. In ICCV, 2021.  
[10] Eugene Bagdasaryan and Vitaly Shmatikov. Blind backdoors in deep learning models. In USENIX, 2021.  
[11] Mauro Barni, Kassem Kallas, and Benedetta Tondi. A new backdoor attack in cnns by training set corruption without label poisoning. In ICIP, 2019.  
[12] Aniruddha Saha, Akshayvarun Subramanya, and Hamed Piriavash. Hidden trigger backdoor attacks. In AAAI, 2020.  
[13] Alexander Turner, Dimitris Tsipras, and Aleksander Madry. Label-consistent backdoor attacks. arXiv preprint arXiv:1912.02771, 2019.  
[14] Shihao Zhao, Xingjun Ma, Xiang Zheng, James Bailey, Jingjing Chen, and Yu-Gang Jiang. Clean-label backdoor attacks on video recognition models. In CVPR, 2020.  
[15] Tuan Anh Nguyen and Anh Tuan Tran. Wanet - imperceptible warping-based backdoor attack. In ICLR, 2021.  
[16] Tuan Anh Nguyen and Anh Tran. Input-aware dynamic backdoor attack. In NIPS, 2020.  
[17] Junyu Lin, Lei Xu, Yingqi Liu, and Xiangyu Zhang. Composite backdoor attack for deep neural network by mixing existing benign features. In CCS, 2020.  
[18] Ahmed Salem, Rui Wen, Michael Backes, Shiqing Ma, and Yang Zhang. Dynamic backdoor attacks against machine learning models. arXiv preprint arXiv:2003.03675, 2020.  
[19] Ilia Shumailov, Zakhar Shumaylov, Dmitry Kazhdan, Yiren Zhao, Nicolas Papernot, Murat A Erdogdu, and Ross J Anderson. Manipulating sgd with data ordering attacks. In NIPS, 2021.  
[20] Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending against backdooring attacks on deep neural networks. In RAID, 2018.  
[21] Dongxian Wu and Yisen Wang. Adversarial neuron pruning purifies backdoored deep models. In NIPS, 2021.  
[22] Kota Yoshida and Takeshi Fujino. Disabling backdoor and identifying poison data by using knowledge distillation in backdoor attacks on deep neural networks. In ACM Workshop, 2020.  
[23] Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian M. Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering. In AAAI, 2019.  
[24] Ezekiel Soremekun, Sakshi Udeshi, and Sudipta Chattopadhyay. Exposing backdoors in robust machine learning models. arXiv preprint arXiv:2003.00865, 2020.

[25] Alvin Chan and Yew-Soon Ong. Poison as a cure: Detecting & neutralizing variable-sized backdoor attacks in deep neural networks. arXiv preprint arXiv:1911.08040, 2019.  
[26] Huili Chen, Cheng Fu, Jishen Zhao, and Farinaz Koushanfar. Deepinspect: A black-box trojan detection and mitigation framework for deep neural networks. In IJCAI, 2019.  
[27] Haripriya Harikumar, Vuong Le, Santu Rana, Sourangshu Bhattacharya, Sunil Gupta, and Svetha Venkatesh. Scalable backdoor detection in neural networks. In ECML, 2020.  
[28] Zhen Xiang, David J. Miller, and George Kesidis. Detection of backdoors in trained classifiers without access to the training set. IEEE TNL, 2022.  
[29] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In NIPS, 2020.  
[30] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.  
[31] Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In IEEE S&P, 2021.  
[32] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.  
[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
[34] Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Neural attention distillation: Erasing backdoor triggers from deep neural networks. In ICLR, 2021.  
[35] Pu Zhao, Pin-Yu Chen, Payel Das, Karthikeyan Natesan Ramamurthy, and Xue Lin. Bridging mode connectivity in loss landscapes and adversarial robustness. In ICLR, 2020.  
[36] Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Anti-backdoor learning: Training clean models on poisoned data. In NIPS, 2021.
