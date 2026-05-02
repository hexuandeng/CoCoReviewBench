# PROGRESSIVE MIX-UP FOR FEW-SHOT SUPERVISED MULTI-SOURCE DOMAIN TRANSFER

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper targets at a new and challenging setting of knowledge transfer from multiple source domains to a single target domain, where target data is few shot or even one shot with label. Traditional domain generalization or adaptation methods cannot directly work since there is no sufficient target domain distribution serving as the transfer object. The multi-source setting further prevents the transfer task as excessive domain gap introduced from all the source domains. To tackle this problem, we newly propose a progressive mix-up (P-Mixup) mechanism to introduce an intermediate mix-up domain, pushing both the source domains and the few-shot target domain aligned to this mix-up domain. Further by enforcing the mix-up domain to progressively move towards the source domains, we achieve the domain transfer from multi-source domains to the single one-shot target domain. Our P-Mixup is different from traditional mix-up that ours is with a progressive and adaptive mix-up ratio, following the curriculum learning spirit to better align the source and target domains. Moreover, our P-Mixup combines both pixel-level and feature-level mix-up to better enrich the data diversity. Experiments on two benchmarks show that our P-Mixup significantly outperforms the state-of-the-art methods, i.e.,  $6.0\%$  and  $6.8\%$  improvements on Office-Home and DomainNet.

# 1 INTRODUCTION

Deep neural networks (DNN) have gained large achievements on a wide variety of computer vision tasks (He et al., 2016; Ren et al., 2015; Ronneberger et al., 2015). As problems turn complex, the learned DNN models consistently fall short in generalizing to test data under different distributions from the training data. Such domain shift (Torralba & Efros, 2011) further results in performance degradation as models are overfitting to the training distributions. Domain adaptation (DA) (Wang & Deng, 2018) has been extensively studied to address this challenge. Due to different settings regarding the source and target domains, DA problems vary into different categories such: unsupervised domain adaptation (UDA) (Long et al., 2018), supervised domain adaptation (SDA) (Motiian et al., 2017), and multi-source domain adaptation (MSDA) (Zhao et al., 2018). UDA aims to adopt knowledge from a fully labeled source domain to an unlabeled target domain. SDA intends to transfer knowledge from a fully labeled source domain to a partially labeled target domain. MSDA generalizes the UDA by adopting the knowledge from multiple fully labeled source domains to an unlabeled target domain. The main difficulty in the MSDA problem is how to achieve a meaningful alignment between the labeled source domains and the target domain that is unlabeled. Although DA has obtained some good achievements, assuming the availability of plenty of unlabeled/labeled target samples in real-world scenarios cannot be always guaranteed.

In this paper, we propose a challenging and realistic problem setting named Few-shot Supervised Multi-source Domain Transfer (FSMDT), by assuming that multiple labeled source domains are accessible but the target domain only contains few samples (i.e., one labeled sample per class), shown in Figure 1. Different from existing domain adaptation problems such as UDA, SDA and MSDA, the target domain in our problem does not provide any unlabeled samples to assist model training. The most relevant problem settings to ours are SDA and MSDA. SDA (Tzeng et al., 2015; Koniusz et al., 2017; Motiian et al., 2017; Morsing et al., 2021) seeks to transfer knowledge from a single source domain to a partially labeled target domain. The SDA methods cannot be simply used to deal with our problem that involves multiple source domains, as the alignment among multiple source domains should be carefully addressed. In addition, existing MSDA methods (Duan et al.,

![](images/16b5ad7cc8b47deffc224a27a0f4a10097c17ad57823b64fe00cdf3bbfdbdb2a.jpg)  
Figure 1: Visual illustration on the FSMDT problem (left), traditional domain adaptation solutions (middle) and our P-Mixup method (right).

2009; Sun et al., 2011; Zhao et al., 2018; Wang et al., 2020a; Zhou et al., 2021b; Ren et al., 2022) aim to learn domain-invariant representations by aligning the target domain to each of the source domains. However, these MSDA methods are not suitable for our FSMDT problem, as target domain only contains few labeled samples for training process which cannot support the domain invariance learning. Recently, multi-source few-shot domain adaptation (MFDA) (Yue et al., 2021) is proposed to address the application scenario where only a few samples in each source domain are annotated while the remaining source and target samples are unlabeled. Different from MFDA, our proposed FSMDT assumes only few target samples are available. The methods for MFDA would fail to learn discriminative representations on target domain in FSMDT due to insufficient target samples.

We propose a novel progressive mix-up scheme to tackle the challenges in the newly proposed FSMDT problem. Our scheme firstly creates an intermediate mix-up domain, which is initially set closer to the few-shot target domain. Rather than the commonly used image-level mix-up, we induce a cross-domain bi-level mix-up, which involves both the image-level mix-up and feature-level mix-up, to effectively enrich the data diversity. With the mix-up domain that is initially close to the target domain, the few-shot constraint on target domain is alleviated. Then, by enforcing the mix-up ratio to progressively favor towards the source domains, and meanwhile harnessing the target domain to be close to the mix-up domain, we gradually transfer knowledge from the multi-source domain to the target domain in a curriculum learning fashion. Furthermore, by optimizing over multiple source domains in a meta-learning regime, we present a stable and robust solution to the FSMDT problem.

Our main contributions are summarized as follows:

- We introduce a practical and challenging task, namely the Few-shot Supervised Multi-source Domain Transfer (FSMDT), which aims to transfer knowledge from multiple labeled source domains to a target domain with only few labeled samples.  
- We propose a novel progressive mix-up scheme to help address the FSMDT problem, which creates an intermediate mix-up domain and gradually adapts the mix-up ratio to mitigate the domain shift between target domain and source domain.  
- We conduct extensive experiments and show that our method successfully tackles the new FSMDT problem and it surpasses state-of-the-arts with large margins. In particular, it improves the accuracy by  $6.0\%$  and  $6.8\%$  over MSDA and SDA baselines on the OfficeHome and DomainNet datasets, respectively.

# 2 RELATED WORK

In this section, we mainly review topics including domain adaptation, domain generalization, and mix-up, which are closely related to our work.

# 2.1 DOMAIN ADAPTATION AND GENERALIZATION

Domain adaptation (DA) aims to transfer knowledge from a source domain to a target domain with a strong assumption that target data are available for model training. Domain generalization (DG) is a more challenging task in not only closing the domain gap but also addressing the absence of target data. There are two types of DA problems that are most relevant to our proposed FSMDT problem: supervised domain adaptation (SDA) (Tzeng et al., 2015; Koniusz et al., 2017; Motiian et al., 2017; Morsing et al., 2021) and multi-source domain adaptation (MSDA) (Duan et al., 2009; Sun et al., 2011; Zhao et al., 2018; Wang et al., 2020a)

Supervised Domain Adaptation trains models by exploiting a partially labeled target domain and a single, fully labeled source domain. Seminal work such as the simultaneous deep transfer (SDT) (Motiian et al., 2017) jointly learns domain-invariant features and aligns semantic information across domains by optimizing the domain confusion and distribution matching objectives. The classification and contrastive semantic alignment (CCSA) method (Motiian et al., 2017) uses the distribution alignment along the semantic manifold. To deal with the few-shot issue, CCSA reverts to point-wise surrogates of distribution and similarities. Recently, (Morsing et al., 2021) exploits graph embedding to encode intra-class and inter-class information to better align the source and target domains. Different from SDA, we consider multi-source domain instead of a single source domain, which is more challenging as real data is not constrained to be only from a distribution.

Multi-Source Domain Adaptation Early MSDA methods (Sun et al., 2011; Duan et al., 2009) learn domain-invariant feature across all domains, or leverage auxiliary classifiers trained with multisource domain to ensemble a robust classifier for the target domain. Recently, the multi-source domain adversarial network (MDAN) (Zhao et al., 2018) theoretically analyzes the average case generalization bounds for MSDA classification and regression problems, and it adopts an adversarial learning strategy to address the MSDA problem. In addition, the learning to combine for multisource domain adaptation (LtC-MSDA) (Wang et al., 2020a) explores interactions among domains by building a knowledge graph of prototypes from various domains and investigates the information propagation among semantically adjacent representations. Despite the good performance, none of the above methods consider the practical scenario with only very few labeled target samples.

Domain Generalization aims to learn a model from multiple source domains that can generalize well on unseen target domain. In contrast to Domain Adaptation (DA), DG removes the strong assumption that target domain is available for model training stage. Existing DG methods can be roughly divided into three groups. Domain alignment based methods (Muandet et al., 2013; Li et al., 2018b) aim to learn the domain invariant features by aligning feature distributions across multiple source domains. Meta-learning based methods (Li et al., 2018a; Shu et al., 2021) divide multiple source domains into the meta-train and meta-test sets, and learn a model on the meta-train set with the intention of improving its performance on the meta-test set. Data augmentation based methods (Zhou et al., 2021a; 2020) aim to improve the generalization of learned models by enriching the diversity of source domains. Though domain generalization addresses the unseen target domain, which is a harder problem than our few-shot seen target setting, it is not suitable for our FSMDT problem as it doesn't consider how to utilize these available few-shot samples in target domain.

# 2.2 DATA AUGMENTATION BY MIX-UP

Mix-up (Zhang et al., 2018) is a data augmentation technique that has been widely applied in self-supervised learning, domain adaptation, and domain generalization. Dual mixup regularized learning (DMRL) (Wu et al., 2020) conducts class-level and domain-level mix-up strategies to learn a domain-invariant feature space. Adversarial domain adaptation with domain mixup (DM-ADA) (Xu et al., 2020) incorporates image and feature level mix-up with soft domain labels into an adversarial training framework. (Wang et al., 2020b) proposes two kinds of image-level mix-up strategies to enhance the generalization of the learned model on novel category identification. Recently, Domain-augmented meta learning (DAML) (Shu et al., 2021) applies multi-source mix-up strategy to augment source domains. However, most methods interpolate samples with a pre-defined mix-up ratio distribution, e.g., beta distribution. Lately, MetaMixup (Mai et al., 2021) proposes a meta-learning based framework to dynamically update mix-up ratio. However, it requires a special validation setting to learn the mix-up ratio, and it does not consider the mix-up problems across multiple domains. In contrast, we consider the cross-domain mix-up and propose a progressive mix-up scheme based on the cross-domain Wasserstein distance, which does not rely on extra validation settings.

# 3 METHOD

Unlike SDA, we target at jointly leveraging multi-source domain other than single source domain, together with few-shot labeled target samples, to adapt the multi-source domain knowledge to the target domain. The main challenge is the extremely limited target data points, which cannot provide sufficient and stable target distribution and thus difficult to conduct transfer. Inspired by Mix-up (Zhang et al., 2018), we propose a progressive mix-up (P-Mix) scheme to introduce an interme

diate mix-up domain, and enforce the distribution alignment of "source to mix-up" and "target to mix-up". Our scheme starts with a mix-up distribution close to target domain, and gradually drifts towards source domains. In this way, the large domain gap is surrogated by a milder intermediate gap and the target to source alignment is indirectly achieved. Firstly, we introduce the preliminary. Then, we give details of the bi-level mix-up. Last, we illustrate our newly proposed progressive mix-up scheme and summarize the overall pipeline of our algorithm.

# 3.1 PRELIMINARIES

In Few-shot Supervised multi-source domain Transfer (FSMDT) problem, we have  $M$  full labeled source domains and a target domain with few-shot labeled data. The  $i$ -th source domain  $\mathcal{D}_{s,i} = \{(x_{s,i}^j,y_{s,i}^j)\}_{j = 1}^{N_{s,i}}$  contains  $N_{s,i}$  labeled samples drawn from the source distribution  $P_{s,i}(x,y)$ , and the target domain  $D_{t} = \{(x_{t}^{j},y_{t}^{j})\}_{j = 1}^{N_{t}}$  includes  $N_{t}$  labeled samples selected from the target distribution  $P_{t}(x,y)$ . Here,  $N_{t}\ll N_{s,i}$ , i.e.,  $N_{t}$  can be as few as 1-shot per class.  $P_{t}(x,y)\neq P_{s,i}(x,y)$  and  $P_{s,i}(x,y)\neq P_{s,j}(x,y)$  where  $i\ne j$ . The multiple source domains and target domain have the same label space  $Y = \{1,2,\dots ,K\}$  with  $K$  categories. We aim to learn an adaptive model  $\mathbf{H}$  on  $\{\mathcal{D}_{s,i}\}_{i = 1}^{M}$  and  $\mathcal{D}_t$ , that can generalize well on unseen samples from target domain. In general,  $\mathbf{H}$  consists of two functions, i.e.,  $\mathbf{H} = \mathbf{F}\circ \mathbf{G}$ . Here  $\mathbf{G}:x\to g$  represents the feature extractor that maps the input sample  $x$  into an embedding space, and  $\mathbf{F}:g\rightarrow f$  is the classifier with input the embedding to predict the category.

# 3.1.1 RECAP OF MIX-UP

Mix-up (Zhang et al., 2018) is one of the most popular data augmentation strategies to improve the generalization and robustness of the learned model by enriching the diversity of the original domain. The core idea of mix-up is to create virtual samples by randomly interpolating two samples in a convex fashion. Specifically, given two samples  $(x_{i},y_{i})$  and  $(x_{j},y_{j})$ , the virtual sample  $(\tilde{x},\tilde{y})$  is defined as:

$$
\tilde {x} = \lambda x _ {i} + (1 - \lambda) x _ {j}, \tag {1}
$$

$$
\tilde {y} = \lambda y _ {i} + (1 - \lambda) y _ {j}, \tag {2}
$$

where label  $y$  is the one-hot label encoding and  $\lambda$  is randomly sampled from a predefined distribution, e.g., beta distribution.

# 3.2 CROSS-DOMAIN BI-LEVEL MIX-UP

Traditional mix-up is originally designed for self-supervised learning, i.e., introducing a new class data by interpolating from two known classes' data, which can increase the training data diversity. When considering the domain transfer problem, such mix-up will be cross-domain, i.e., a data point from source domain and a data point from target domain. Meanwhile, besides the pixel-level mix-up, recent manifold based mix-up (Verma et al., 2019; Shu et al., 2021; Xu et al., 2020) shows that feature-level interpolation can also improve the generalization and model robustness. We thus investigate both the pixel-level and feature-level mix-ups.

Cross-Domain Image-Level Mix-up. Motivated by the success of mix-up in self-supervised learning, we apply it to our domain transfer task, which can create new samples with new labels. We utilize it to largely enrich the target domain distribution as there is overly limited target samples. The source and target samples are linearly interpolated as:

$$
\tilde {x} _ {i m g} = \lambda x _ {s, i} + (1 - \lambda) x _ {t}, \tag {3}
$$

$$
\tilde {y} _ {i m g} = \lambda y _ {s, i} + (1 - \lambda) y _ {t}, \tag {4}
$$

where  $\lambda$  is the mix-up ratio. Notice that during training, such mix-up ratio can be adjusted, e.g., a larger  $\lambda$  generates closer-to-source samples and a smaller  $\lambda$  generates closer-to-target samples.

Cross-Domain Feature-Level Mix-up. On the learned feature representation manifold, mix-up at the feature level enables more intermediate virtual features to increase the feature diversity and can directly interact with the classifier  $\mathbf{F}$  learning. Here, given a pair of source and target features and

![](images/36518b5d1afc53cd1202a20817cf97c1103d5e3089e44ec4da5068eccdf1db20.jpg)  
Figure 2: The flowchart of the proposed progressive mix-up. A mix-up domain (red) is introduced as initially closer to the target domain. By enforcing the mix-up ratio  $\lambda$  to be progressively increasing based on the wasserstein distance of source-to-mixup and target-to-mixup, we push the mix-up domain gradually to be closer to source domains, and thus achieving the alignment of multi-source domain to the few-shot target domain.

their corresponding labels:  $(g_{s,i},y_{s_i})$  and  $(g_t,y_t)$ , we have

$$
\tilde {g} _ {\text {f e a t}} = \lambda g _ {s, i} + (1 - \lambda) g _ {t}, \tag {5}
$$

$$
\tilde {y} _ {\text {f e a t}} = \lambda y _ {s, i} + (1 - \lambda) y _ {t}, \tag {6}
$$

where  $\lambda$  is the mix-up ratio same as the one used in image-level mix-up. With exactly the same  $\lambda$ , we argue that the image-level mix-up samples lie in the same feature space as the feature-level mix-up samples. Thus, we can jointly utilize the two for penalty, i.e., the same class image-level mix-up and feature-level mix-up should go for the same classification result, which is further discussed in the training objectives.

# 3.3 PROGRESSIVE MIX-UP SCHEME

Previous work apply either fixed sampling or some simple randomized sampling for the mix-up ratio  $\lambda$ , e.g., beta or dirichlet distribution (Zhang et al., 2018; Wu et al., 2020; Xu et al., 2020; Shu et al., 2021). However, we find that the sampling of mix-up ratio is crucial for the domain transfer. The ratio directly determines the intermediate mix-up domain. If a mix-up domain is constant or some special distribution, the alignment is either still constantly hard or likely to be under-fitting, supported from a recent work MetaMixup (Mai et al., 2021).

To alleviate it, we dig into the Wasserstein distance of "source-to-mixup"  $d_w(\mathcal{G}_s, \mathcal{G}_{mix})$  and "target-to-mixup"  $d_w(\mathcal{G}_t, \mathcal{G}_{mix})$ , where  $\mathcal{G}_s, \mathcal{G}_t, \mathcal{G}_{mix}$  stand for the embeddings of source, target and mix-up domains. We observe that during the training, if the mix-up domain initially is closer to the few-shot target domain, the alignment is relatively simple as  $d_w(\mathcal{G}_t, \mathcal{G}_{mix})$  is already small while  $d_w(\mathcal{G}_s, \mathcal{G}_{mix})$  can be effectively minimized as there are sufficient source domain data. When gradually increasing the mix-up ratio towards closer to source domains, since we already harness the "target-to-mixup" distance to be small, we are pushing the entire mix-up domain and few-shot target domain towards the source domains, as illustrated in Figure 2. Such progressively adjusted mix-up ratio, following the spirit of curriculum learning (Bengio et al., 2009), eases the initial large domain gap by mildly starting close to the target, and secures the entire transfer process smoothly.

Specifically, we introduce a weighting factor  $q$  to depict the closeness to source as:

$$
q = \exp \left(- \frac {d _ {w} \left(\mathcal {G} _ {s} , \mathcal {G} _ {m i x}\right)}{\left(d _ {w} \left(\mathcal {G} _ {s} , \mathcal {G} _ {m i x}\right) + d _ {w} \left(\mathcal {G} _ {t} , \mathcal {G} _ {m i x}\right)\right) T}\right), \tag {7}
$$

where  $T$  is a temperature factor defined as 0.05. During training, by initializing  $\mathcal{G}_{\text{mix}}$  closer to target domain, such  $q$  is small. To progressively adjust it, we consider to apply this closeness on top of the previous stage  $\lambda$  in a moving average manner. Further, a linearly incremental component is introduced to enforce the gradual closeness to the source domains. The progressive mix-up is formulated as:

$$
\lambda_ {n} = \frac {n (1 - q)}{N} + q \lambda_ {n - 1}, \tag {8}
$$

$N$  is the total number of iterations and  $n$  is the current iteration index. Initial weighting  $\lambda_0$  towards source is 0. To numerically stabilize the training procedure, we introduce a uniform distribution  $U$ ,

![](images/bd06b74b4ee8f738915fb18a0d577cb6c2bd3fa46d5da265a42e5d43ad66efd9.jpg)  
Figure 3: The training architecture. Both image and feature level P-Mixup are applied for the cross-entropy loss.  $\mathbf{G}$  is the feature extractor and  $\mathbf{F}$  is the classifier.

a random perturbation on top of the current  $\lambda_{n}$ :

$$
\tilde {\lambda} _ {n} = \operatorname {C l a m p} (U (\lambda_ {n} - \sigma , \lambda_ {n} + \sigma), \min  = 0. 0, \max  = 1. 0), \tag {9}
$$

$\sigma$  is a local perturbation range, i.e., we empirically set it as 0.2.  $\tilde{\lambda}_n$  is then stochastically sampled and clamped into range [0.0, 1.0] for each iteration  $n$ 's mix-up ratio.

# 3.4 ARCHITECTURE AND LEARNING OBJECTIVES

The overall architecture is shown in Figure 3, which mainly consists of a feature generator  $\mathbf{G}$  and a classifier  $\mathbf{F}$ . During training, since there are multiple labeled source domain data, and a single few-shot target domain data, we follow the canonical domain generalization frameworks such as MAML (Finn et al., 2017), to organize our training in a meta-learning manner.

Denoting the model parameters  $\mathbf{F} \circ \mathbf{G}$  as  $\theta$ , the objective for classification is defined as:

$$
\mathcal {L} _ {c e} ^ {\mathcal {T} _ {i}} (\theta) = - \sum_ {x, y \in \mathcal {T} _ {i}} \sum_ {k = 1} ^ {K} y _ {k} \log \left(\theta (x) _ {k}\right) \tag {10}
$$

Where  $\mathcal{T}_i$  stands for a specific domain, e.g., one of the source domains or the target domain,  $x$  is the input image and  $y$  is the ground truth label, and  $K$  is the number of classes. Notice that for "cross domain image-level mix-up", the input is the mix-up image  $\tilde{x}_{img}$  and the label is the mix-up label  $\tilde{y}_{img}$ . For "cross domain feature-level mix-up", the mix-up feature  $\tilde{g}_{feat}$  is fed into the classifier and computes the  $\mathcal{L}_{ce}$  loss. Meanwhile, the label  $y$  is the mix-up label  $\tilde{y}_{feat}$ .

Following MAML, we conduct a meta-optimization to pseudo-update the model parameters for the first time by minimizing  $\sum_{\mathcal{T}_i\in p(\mathcal{T})}\mathcal{L}_{ce}^{\mathcal{T}_i}$ :

$$
\theta^ {\prime} = \theta - \alpha \nabla_ {\theta} \sum_ {\mathcal {T} _ {i} \in p (\mathcal {T})} \mathcal {L} _ {c e} ^ {\mathcal {T} _ {i}} (\theta) \tag {11}
$$

$p(\mathcal{T})$  is a sampling distribution among the meta-train domains, e.g., we uniformly sample  $60\%$  among all domains except the target domain, i.e., the source domains and the mix-up domain for meta-train, and the rest  $40\%$  together with the target domain for meta-test. The meta-test model update is then adopted:

$$
\theta = \theta - \beta \nabla_ {\theta} \sum_ {\mathcal {T} _ {j} \notin p (\mathcal {T})} \mathcal {L} _ {c e} ^ {\mathcal {T} _ {j}} \left(\theta^ {\prime}\right) \tag {12}
$$

where  $\alpha$  and  $\beta$  are the update step size for meta-train and meta-test respectively. To simplify the parameters, we set  $\alpha = \beta = 0.01$ . Notice that the mix-up domain splits in two sub-domains, the image-level mix-up and the feature-level mix-up. Such two sub-domains can be freely sampled in meta-train or meta-test stages under the sampling distribution  $p(\mathcal{T})$ .

# 4 EXPERIMENT

In this section, we first introduce the experimental settings. We then compare P-Mixup to some typical domain adaptation and generalization methods. We finally ablate each of the proposed key components and analyze the few-shot property of target domain, i.e., the number of labeled target samples with respect to the model performance.

Table 1:  $\mathrm{mAP}(\%)$  on Office-Home. Named in row is the target domain which contains 10 classes randomly selected from label space. (A: Art, C: Clipart, P: Product, R: Real_world)  

<table><tr><td>Dt10</td><td>ERM-w/o</td><td>ERM-w</td><td>CCSA</td><td>MDAN</td><td>Mix-up</td><td>DAML</td><td>Ours</td></tr><tr><td>A</td><td>60.48</td><td>60.83</td><td>64.52</td><td>58.49</td><td>59.50</td><td>60.30</td><td>72.23</td></tr><tr><td>C</td><td>44.06</td><td>46.35</td><td>56.32</td><td>44.17</td><td>51.03</td><td>49.88</td><td>59.97</td></tr><tr><td>P</td><td>72.55</td><td>72.75</td><td>75.89</td><td>70.43</td><td>73.03</td><td>72.96</td><td>82.69</td></tr><tr><td>R</td><td>79.32</td><td>76.40</td><td>79.17</td><td>74.28</td><td>76.91</td><td>76.17</td><td>85.05</td></tr><tr><td>Ave.</td><td>63.83</td><td>64.08</td><td>68.97</td><td>61.84</td><td>65.12</td><td>64.83</td><td>74.99</td></tr></table>

# 4.1 EXPERIMENTAL SETTINGS

Datasets: We adopt two standard domain adaptation and generalization benchmarks: (1) Office-Home (Venkateswara et al., 2017) is the standard domain adaptation/generalization benchmarks which consists of four domains (Art, Clipart, Produce, and Real-world) with 65 classes. (2) DomainNet (Peng et al., 2019) is the largest domain adaptation dataset containing 345 classes. We conduct experiments on four domains (Clipart, Painting, Real, and Sketch) from it.

Protocols: To highlight the challenging few-shot target domain setting, we cannot anymore use the original protocols from the above two datasets. We observe that even with one-shot, since the number of classes are many, e.g., 345 classes from DomainNet, utilizing all the classes can provide a sufficient diversified target domain distribution. To exactly constrain the target distribution to be few-shot, for Office-Home, we randomly select 10 out of 65 classes each with one sample as the target. Similarly, we randomly select 15 out of 345 classes for DomainNet. The remaining samples in these selected classes are used as the test data. Such random sampling is conducted for 5 times and the averaged result is reported.

Baselines: We compare with four main streams of the state-of-the-art methods: (1) Supervised Domain Adaptation method, namely Classification and Contrastive Semantic Alignment (CCSA) (Motiian et al., 2017). (2) Multi-Source Domain Adaptation, namely Multisource Domain Adversarial Networks (MDAN) (Zhao et al., 2018). (3) Domain Generalization method, namely Domain-Augmented Meta Learning (DAML) (Shu et al., 2021). (3) Data Augmentation method, namely Mix-up (Zhang et al., 2018). Besides, we consider another general baseline, i.e., Empirical Risk Minimization (Koltchinskii, 2011) with/without labeled target domain (ERM-w, ERM-w/o).

Evaluation Metrics: For each of the benchmarks, each domain is in turn regarded as the target domain while the remaining are considered as source domains. For each experiment, we report the mean average precision (mAP) by averaging over 5 times of all the class' average precision. We fix the random seed to 1-5 when self-constructing the new domain and sampling target samples so the results of different methods can be fairly compared.

Implementation Details: Our implementation is based on Pytorch (Paszke et al., 2019). We use ResNet-18 (He et al., 2016) pretrained on ImageNet (Deng et al., 2009) as the backbone network. We optimize the model using SGD with momentum of 0.9 and weight decay of  $5 \times 10^{-4}$ . The batch size is set to 50. The initial learning rate is set to 0.01. For all the compared methods and Ours, we use the same basic data preprocessing on the image and the same backbone.

# 4.2 MAIN RESULTS

Office-Home: In Table 1, compared ERM-w to ERM-w/o, we observe that the labeled target domain which only contains 10 images cannot directly improve the performance, which verifies the setting is indeed challenging. Third column is the representative supervised domain adaptation method, CCSA, clearly outperforms the baseline ERM-W. There is also MDAN in the fourth column which performance is worse than ERM-W, as there is no sufficient target distribution to support the adaptation. Across all the methods, our approach demonstrates clear advantages, i.e., when compared to the second best, CCSA, the gain is as significant as  $6.02\%$  on "Ave."

DomainNet: As shown in Table 2, our method's performance in both 10 labeled target samples and 15 labeled target samples scenarios show clear advantage over all the baselines. Compared to the most competitive opponent Mix-up, our method surpasses by  $8.96\%$  on 10 labeled target samples "Ave." and  $6.77\%$  on 15 labeled target samples "Ave." In this dataset, we find that ERM-w obtains

Table 2:  $\mathrm{mAP}(\%)$  on DomainNet. Named in column is the target domain contains 10/15 classes randomly selected from label space. (C: Clipart, P: Painting, R: Real, S: Sketch)  

<table><tr><td>Method</td><td>C</td><td>P</td><td>Dt10R</td><td>S</td><td>Avg.</td><td>C</td><td>P</td><td>Dt15R</td><td>S</td><td>Avg.</td></tr><tr><td>ERM-w/o</td><td>47.33</td><td>37.51</td><td>47.66</td><td>37.60</td><td>42.53</td><td>45.74</td><td>42.43</td><td>53.03</td><td>37.87</td><td>44.77</td></tr><tr><td>ERM-w</td><td>50.28</td><td>43.36</td><td>58.34</td><td>46.32</td><td>49.57</td><td>49.77</td><td>45.16</td><td>56.46</td><td>42.38</td><td>48.44</td></tr><tr><td>CCSA</td><td>43.89</td><td>41.28</td><td>46.46</td><td>42.32</td><td>43.49</td><td>41.57</td><td>42.76</td><td>47.89</td><td>44.79</td><td>44.26</td></tr><tr><td>MDAN</td><td>52.33</td><td>43.86</td><td>58.02</td><td>45.57</td><td>49.95</td><td>51.34</td><td>46.61</td><td>57.02</td><td>43.54</td><td>49.63</td></tr><tr><td>Mix-up</td><td>56.83</td><td>52.35</td><td>62.60</td><td>49.52</td><td>55.33</td><td>52.98</td><td>51.90</td><td>58.94</td><td>43.71</td><td>51.88</td></tr><tr><td>DAML</td><td>57.05</td><td>47.95</td><td>58.88</td><td>49.39</td><td>53.32</td><td>54.54</td><td>46.81</td><td>58.67</td><td>45.37</td><td>51.35</td></tr><tr><td>Ours</td><td>65.06</td><td>60.85</td><td>70.60</td><td>60.67</td><td>64.29</td><td>63.92</td><td>58.37</td><td>67.15</td><td>55.65</td><td>61.11</td></tr></table>

Table 3: Ablation study on Office-Home. Named in column is the target domain which contains 10 classes randomly selected from the label space.  

<table><tr><td>Mix-up Ratio λ</td><td>Method</td><td>Art</td><td>Clipart</td><td>Product</td><td>Real_world</td><td>Ave.</td></tr><tr><td>N/A</td><td>ERM-w (no mix-up)</td><td>60.48</td><td>44.06</td><td>72.55</td><td>79.32</td><td>63.83</td></tr><tr><td rowspan="3">Random Sampling</td><td>Feat-Mix</td><td>60.06</td><td>48.21</td><td>69.06</td><td>73.56</td><td>62.72</td></tr><tr><td>Img-Mix</td><td>59.50</td><td>51.03</td><td>73.03</td><td>76.91</td><td>65.12</td></tr><tr><td>Feat-Mix + Img-Mix</td><td>66.40</td><td>56.18</td><td>76.51</td><td>78.98</td><td>69.52</td></tr><tr><td rowspan="3">Progressive Update</td><td>Feat-Mix</td><td>64.89</td><td>55.08</td><td>75.41</td><td>77.52</td><td>68.22</td></tr><tr><td>Img-Mix</td><td>68.45</td><td>57.55</td><td>81.26</td><td>83.89</td><td>72.79</td></tr><tr><td>Feat-Mix + Img-Mix</td><td>72.23</td><td>59.97</td><td>82.69</td><td>85.05</td><td>74.99</td></tr></table>

the same level performance as most of baselines or even better than supervised domain adaptation method CCSA, partially showing that this benchmark is more challenging as the domain gap become more challenging compared to the other two datasets. Overall, these results strongly demonstrate the effectiveness of our proposed progressive mix-up for improving domain transfer with extremely few labeled target domain samples.

# 4.3 ABLATION STUDY

We conduct a comprehensive ablation study to examine the effectiveness of our proposed core components in Table 3. The baseline of ERM-w utilizing the target domain data but without mix-up is shown in the first row. Feat-Mix denotes the cross-domain feature-level mix-up and Img-Mix indicates the cross-domain image-level mix-up. We introduce the general mix-up ratio sampling strategy  $(\lambda \sim Beta(0,1))$  used in Mix-up (Zhang et al., 2018), as a major comparison. Firstly, We observe that the bottom row methods consistently outperform the middle row with large margin, highlighting the superiority of the proposed progressive mix-up strategy. Then, we look into the combination of modules within each sampling method. We observe that cross-domain image-level mix-up (Img-Mix) shows better result over cross-domain feature-level mix-up (Feat-Mix) with more than  $4.0\%$  "Ave." improvement. If going for one module, image-level mix-up would be a better choice. If with no restriction, a combination of both image and feature level mix-ups can further boost the accuracy, because a combined mix-up enriches the data diversity more than each of the single choices.

# 4.4 EFFECT OF TARGET SAMPLE SHOTS

We investigate the effect of the number of target sample shots on our proposed P-Mixup with DomainNet where "Clipart" is selected as the target domain. As the specific experimental protocol is to ensure that there is no sufficient target distribution for the multi-source domain transfer. We increase the number of selected classes from 10 to 25. Corresponding, the number of available target samples range from 10 to 25. As shown in Table 4, our method's performance across different number of selected classes settings show clear advantage over all the baselines. Specifically, even when we doubled the number of selected classes from 10 to 20, our method surpasses the most competitive opponent DAML, by  $8\%$ . As the size of available labeled target samples decreases, our method still holds an obvious advantage, which further confirms that our method is more advantageous when target sample shots are extremely fewer.

Table 4:  $\mathrm{mAP}(\%)$  on DomainNet "Clipart" setting where "Clipart" is the target domain.  $n$  indicates the number of classes randomly selected from the label space.  

<table><tr><td>Dnt</td><td>ERM-w/o</td><td>ERM-w</td><td>CCSA</td><td>MDAN</td><td>Mix-up</td><td>DAML</td><td>Ours</td></tr><tr><td>10</td><td>47.33</td><td>50.28</td><td>43.89</td><td>52.33</td><td>56.83</td><td>57.15</td><td>65.06</td></tr><tr><td>15</td><td>45.74</td><td>49.77</td><td>41.57</td><td>51.34</td><td>52.98</td><td>54.45</td><td>63.92</td></tr><tr><td>20</td><td>47.04</td><td>51.65</td><td>42.54</td><td>53.07</td><td>52.11</td><td>56.04</td><td>66.26</td></tr><tr><td>25</td><td>47.93</td><td>52.89</td><td>45.43</td><td>54.55</td><td>50.54</td><td>57.82</td><td>65.82</td></tr></table>

![](images/a69bd8b5d6a3b98387839b6a5531bca72edf4de636b18c503c15f861a6192c14.jpg)

![](images/8afc18ab0d84a7aa8434efbceb41066bb15ae4a6892470b78d189fd1e6b904d6.jpg)

![](images/8b7c90284d5005c66c626a957e92fc3935256b213b996a4293aae739a7294870.jpg)

![](images/b5178c260de629834bec95cb2a1b1fe55ca77eb9a8ee71ba7eb3dc6bd59c7c6e.jpg)

![](images/54fb046eab89f8105d8c9c221b72006a8290cf6377d88940ff7201f15a87f80e.jpg)  
Figure 4: The first row illustrates the Mix-up ratio  $\lambda$  and the  $q$  value introduced in Equation 7. The lower  $1 - q$ , the closer the target is to the sources. The second row describes the P-Mixup training behavior compared to baselines and standard deviation (STD) values for all methods.

![](images/b1603512207da2e4b683bd9305d890c30d1aeb83ad58e918c24be4007d361a0a.jpg)

![](images/c48583a059250699d0330ff574dbb3d1b3ce903b87207838067bdabf7cee248b.jpg)

![](images/41fc7ff1652cfda86a72fb0b4b6835b2e2ff3f640363fa7b7b62c6ff787d5d60.jpg)

# 4.5 MIX-UP RATIO AND COMPUTATION ANALYSIS

As shown the first row in Figure 4, we validate the P-Mixup scheme by showing the mix-up ratio and the  $q$  value introduced in Equation 7 on Office-Home. The first two figures show the trend of the proposed mix-up ratio  $\lambda$  along the training iterations. Generally it is an increasing tread as we gradually push the mix-up domain to be closer to source domains. The last two figures show  $1 - q$  over iterations, which indicates the distance change between source and target domains. As  $q$  depicts the closeness to source, we use  $1 - q$  to present the closeness to target. During the first 4000 iteration, the mix-up distribution is closer to target than source, and the model gradually handles the "target-to-mixup" distance to be small. As a result, we observe that the  $1 - q$  value gently turns small. After the model harnesses the "target-to-mixup" distance, the mix-up distribution gradually moves close to source domains as  $\lambda$  goes up. Afterwards, the "target-to-mixup" distance continually decreases, showing that the source domains are continuously transferred onto the target domain and our P-Mixup is indeed effective in mitigating the domain shift in FSMDT. The second row in Figure 4 shows the training behavior and standard deviation (STD) values for all methods on Office-Home. We observe that our proposed method P-Mixup consistently and significantly outperforms all the baselines in terms of training behavior and STD, which verify the effectiveness of our P-Mixup.

# 5 CONCLUSIONS

In this work, we propose to address a new and challenging problem, namely Few-shot Supervised multi-source domain Transfer (FSMDT), where multiple fully labeled source domain samples and extremely limited target samples are accessible. A progressive mix-up (P-Mixup) scheme is newly introduced to effectively mitigate the source and target domain gap especially when target domain is with extremely few-shot samples. We jointly consider the image-level and feature-level cross-domain mix-up to sufficiently enrich the data diversity. A meta-learning optimization strategy is applied to support the multi-domain joint training with stable and robust convergence. Extensive experiments show that our method achieves significant performance gain over the state-of-the-art methods across two main domain adaptation benchmarks.

# REFERENCES

Yoshua Bengio, Jerome Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, 2009.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, pp. 248-255, 2009.  
Lixin Duan, Ivor W Tsang, Dong Xu, and Tat-Seng Chua. Domain adaptation from multiple sources via auxiliary classifiers. In International Conference on Machine Learning, pp. 289-296, 2009.  
Chelsea Finn, Pieter Abeel, and Surgey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Computer Vision and Pattern Recognition, pp. 770-778, 2016.  
Vladimir Koltchinskii. Oracle Inequalities in Empirical Risk Minimization and Sparse Recovery Problems: Ecole d'Eté de Probabilités de Saint-Flour XXXVIII-2008, volume 2033. Springer Science & Business Media, 2011.  
Piotr Koniusz, Yusuf Tas, and Fatih Porikli. Domain adaptation by mixture of alignments of second- or higher-order scatter tensors. In Computer Vision and Pattern Recognition, pp. 4478-4487, 2017.  
Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Learning to generalize: Meta-learning for domain generalization. In Association for the Advancement of Artificial Intelligence, 2018a.  
Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature learning. In Computer Vision and Pattern Recognition, pp. 5400-5409, 2018b.  
Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I Jordan. Conditional adversarial domain adaptation. Neural Information Processing Systems, 31, 2018.  
Zhijun Mai, Guosheng Hu, Dexiong Chen, Fumin Shen, and Heng Tao Shen. Metamixup: Learning adaptive interpolation policy of mixup with metalearning. IEEE Transactions on Neural Networks and Learning Systems, 2021.  
Lukas Hedegaard Morsing, Omar Ali Sheikh-Omar, and Alexandros Iosifidis. Supervised domain adaptation using graph embedding. In International Conference on Pattern Recognition, pp. 7841-7847, 2021.  
Saeid Motiian, Marco Piccirilli, Donald A Adjeroh, and Gianfranco Doretto. Unified deep supervised domain adaptation and generalization. In International Conference on Computer Vision, pp. 5715-5725, 2017.  
Krikamol Muandet, David Balduzzi, and Bernhard Scholkopf. Domain generalization via invariant feature representation. In International Conference on Machine Learning, pp. 10-18. PMLR, 2013.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Neural Information Processing Systems, volume 32, 2019.  
Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In International Conference on Computer Vision, pp. 1406-1415, 2019.  
Chuan-Xian Ren, Yong-Hui Liu, Xi-Wen Zhang, and Ke-Kun Huang. Multi-source unsupervised domain adaptation via pseudo target domain. IEEE Transactions on Image Processing, 31:2122-2135, 2022.

Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Neural Information Processing Systems, volume 28, 2015.  
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer Assisted Intervention, pp. 234-241. Springer, 2015.  
Yang Shu, Zhangjie Cao, Chenyu Wang, Jianmin Wang, and Mingsheng Long. Open domain generalization with domain-augmented meta-learning. In Computer Vision and Pattern Recognition, pp. 9624-9633, 2021.  
Qian Sun, Rita Chattopadhyay, Sethuraman Panchanathan, and Jieping Ye. A two-stage weighting framework for multi-source domain adaptation. Neural Information Processing Systems, 2011.  
Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In Computer Vision and Pattern Recognition, pp. 1521-1528, 2011.  
Eric Tzeng, Judy Hoffman, Trevor Darrell, and Kate Saenko. Simultaneous deep transfer across domains and tasks. In International Conference on Computer Vision, pp. 4068-4076, 2015.  
Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Computer Vision and Pattern Recognition, pp. 5018-5027, 2017.  
Vikas Verma, Alex Lamb, Christopher Beckham, Amir Najafi, Ioannis Mitliagkas, David Lopez-Paz, and Yoshua Bengio. Manifold mixup: Better representations by interpolating hidden states. In International Conference on Machine Learning, pp. 6438-6447. PMLR, 2019.  
Hang Wang, Minghao Xu, Bingbing Ni, and Wenjun Zhang. Learning to combine: Knowledge aggregation for multi-source domain adaptation. In European Conference on Computer Vision, pp. 727-744. Springer, 2020a.  
Mei Wang and Weihong Deng. Deep visual domain adaptation: A survey. Neurocomputing, 312: 135-153, 2018.  
Yufei Wang, Haoliang Li, and Alex C Kot. Heterogeneous domain generalization via domain mixup. In International Conference on Acoustics, Speech and Signal Processing, pp. 3622-3626. IEEE, 2020b.  
Yuan Wu, Diana Inkpen, and Ahmed El-Roby. Dual mixup regularized learning for adversarial domain adaptation. In European Conference on Computer Vision, pp. 540-555. Springer, 2020.  
Minghao Xu, Jian Zhang, Bingbing Ni, Teng Li, Chengjie Wang, Qi Tian, and Wenjun Zhang. Adversarial domain adaptation with domain mixup. In Association for the Advancement of Artificial Intelligence, volume 34, pp. 6502-6509, 2020.  
Xiangyu Yue, Zangwei Zheng, Hari Prasanna Das, Kurt Keutzer, and Alberto Sangiovanni Vincentelli. Multi-source few-shot domain adaptation. arXiv preprint arXiv:2109.12391, 2021.  
Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018.  
Han Zhao, Shanghang Zhang, Guanhang Wu, José MF Moura, Joao P Costeira, and Geoffrey J Gordon. Adversarial multiple source domain adaptation. In Neural Information Processing Systems, 2018.  
Kaiyang Zhou, Yongxin Yang, Timothy Hospedales, and Tao Xiang. Learning to generate novel domains for domain generalization. In European Conference on Computer Vision, pp. 561-578. Springer, 2020.  
Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain generalization with mixstyle. In International Conference on Learning Representations, 2021a.  
Lihua Zhou, Mao Ye, Dan Zhang, Ce Zhu, and Luping Ji. Prototype-based multisource domain adaptation. IEEE Transactions on Neural Networks and Learning Systems, 2021b.
