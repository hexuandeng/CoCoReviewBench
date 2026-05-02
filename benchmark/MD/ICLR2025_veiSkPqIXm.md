# OPENPL: REALISTIC EVALUATION OF PROMPT LEARNING FOR VLM IN OPEN ENVIRONMENTS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Vision-language models (VLMs) have demonstrated impressive zero-shot capabilities across various image classification tasks. Their performance can be further enhanced through prompt learning methods. To evaluate the effectiveness of prompt learning, it is important to assess its robustness to new classes and distributional shifts. However, current studies typically assume single data distribution shifts and pre-known new class space, which still have gaps with real-world open environments where data distributions and classes are often uncertain and subject to continuous change. To better analyze the robustness of prompt learning methods in more realistic scenarios, we propose a novel evaluation benchmark called OpenPL from the following perspectives: 1) We reconstruct multiple scenarios of open environments, encompassing dynamic class changes, dynamic distribution shifts, and dynamic co-evolution of both distribution and classes; 2) We propose a series of new performance metrics for prompt learning methods based on the Dynamic Robustness Curve (DRC) to better understand their robustness in open environments; 3) We re-implement diverse prompt learning methods and evaluate their performance on the proposed OpenPL benchmark. The results show that no current prompt learning method is robust to open environments and no meaningful performance improvement is achieved compared to the zero-shot performance, designing robust prompt learning methods remains a difficult task. All re-implementations are available at https://anonymous.4open.science/r/OpenPL-565E.

# 1 INTRODUCTIONS

Vision-language models (VLMs) have garnered significant attention recently (Radford et al. (2021); Yao et al. (2021); Jia et al. (2021)) because of its zero-shot prediction capabilities across a wide range of visual recognition tasks. Pre-trained VLMs, such as CLIP (Radford et al. (2021)), ALIGN (Jia et al. (2021)), and BLIP (Li et al. (2022)), acquire extensive vision-language knowledge from a near-infinite number of text-image pairs available on the web. These models can be utilized directly for downstream tasks without the need for fine-tuning.

A large part of the research for VLM recently is the adaptation of pre-trained VLMs on downstream tasks(Zhang et al. (2021);Zhou et al. (2022b);Zhou et al. (2022a);Gao et al. (2024)), and prompt learning(Zhou et al. (2022b);Zhou et al. (2022a)) is a newcomer to these works, with notable improvements in VLM performance on downstream tasks and its simplicity and efficiency in design. Unlike other transfer methods in VLM, prompt learning does not rely on adding additional network layers or modifying complex network structures but rather achieves parameter-efficient VLM transfer by modifying the input text or image with some learnable text or vision prompts.

Although many prompt learning methods, such as MaPLe (Khattak et al. (2023a)) and PromptSRC (Khattak et al. (2023b)), claim to achieve strong generalization performance on downstream tasks, their evaluations often overlook more practical scenarios. Currently, most prompt learning methods depend on benchmarks established by CoCoOp (Zhou et al. (2022a)), where classes are divided into fixed base and new groups. These methods are trained solely on the base classes and tested on new classes separately. Similarly, MaPLe (Khattak et al. (2023a)) proposes benchmarks for Cross-dataset Evaluation and Domain Generalization by training on ImageNet and altering the test data distribution to assess performance across various ImageNet variants and other datasets. In open environments,

algorithms do encounter the emergence of new classes and shifts in data distributions (Zhou (2022)), but such changes are not fixed and cannot be pre-known, and sometimes both distribution and class shifts could occur simultaneously. There is still a huge gap between existing evaluation settings and realistic open environments.

Contributions: In this paper, we propose a novel evaluation benchmark to address the existing problems above and achieve a comprehensive evaluation for prompt learning in VLM. Firstly, we focus on the dynamic class changes and propose two scenarios based on class variations. The first scenario increases classification difficulty by continuously introducing new classes during testing. The second scenario reduces the proportion of base classes while increasing the ratio of new classes, keeping the overall label space size constant. This setup aims to measure the algorithm's robustness in an unknown and evolving label space. Secondly, we focus on dynamic distribution shifts and propose a scenario in which the testing distribution continues to change, becoming increasingly distant from the training data distribution. Thirdly, we focus on the dynamic co-evolution of both distribution and classes and propose a scenario in which both class changes and distribution shifts occur simultaneously. Moreover, we quantify the performance under these paradigms based on the Dynamic Robustness Curve (DRC), and several new performance metrics based on the DRC are proposed to help better analyze the robustness of prompt learning methods. We also re-implement different types of prompt learning methods under a unified standard and evaluate their robustness. We believe this has a positive impact on researchers in this field.

Observations: By researching the results of experimental evaluations on the openPL, our insights can be outlined as follows:

1. No single prompt learning method outperforms others in scenarios with dynamic classes changing, i.e., each method has its better case when new classes emerging.  
2. No prompt learning methods currently show robustness to data distribution shifts, i.e., all methods suffer severe performance degradation with distribution shifts.  
3. No prompt learning methods exhibit meaningful performance gains relative to the zero-shot performance with compound shifts of both distribution and label space.  
4. Enhancing the model's capability for text feature extraction and class discriminability may improve robustness since the available text and image information in downstream tasks is highly imbalanced.

# 2 RELATED WORKS

Vision-language model Recently, researchers have demonstrated VLM (Alayrac et al. (2022); Radford et al. (2021)), which consists of visual and textual modalities trained on large-scale image-text pairs, with strong generalization and discrimination capabilities. These VLMs like CLIP (Radford et al. (2021)), ALIGN (Jia et al. (2021)), BLIP (Li et al. (2022)), FILIP (Yao et al. (2021)), LiT (Zhai et al. (2022)) and Flamingo (Alayrac et al. (2022)) show exceptional performance across numerous visual tasks, including few-shot and zero-shot visual recognition. For example, CLIP (Radford et al. (2021)) designs objective that allows matched text representations and image representations close to each other, and learns a generalized vision-language representation on about 400M text-image pairs. Recent researches have focused on how such VLMs can be better adapted to downstream tasks by means of transfer learning (Zhou et al. (2022b); Zhou et al. (2022a); Zhang et al. (2021); Gao et al. (2024)) and knowledge distillation (Ding et al. (2022); Du et al. (2022); Gu et al. (2021)).

Prompt learning The idea of prompt learning first originated in NLP(Liu et al. (2023)), a method of instructing language models to generate specific outputs by providing prompts without having to tune the pre-trained model's own parameters. Because of its parsimony and efficiency, this technique has attracted a lot of attention in the exploration of fine-tuning VLM pre-trained models to specific visual tasks downstream. For example, at the earliest time, CoOp (Zhou et al. (2022b)) explored the application of prompts to the text branch of CLIP, which was used to optimize the text embedding space of the text branch so that specific categories of the downstream task could be better adapted to the pre-trained model, while VPT (Jia et al. (2022)) later provided a solution for introducing prompts into the visual encoder. After that CoCoOp (Zhou et al. (2022a)) explores the introduction of image information into text prompts to make the unified context into an instance-adaptive context. Moreover

MaPLe (Khattak et al. (2023a)) experiment with the co-optimization of prompts for both the textual and visual sides of the prompts. The aforementioned methods are all dedicated to improving the form of prompts and enhancing the model's ability to extract image features. Different from the above methods, ProGrad (Zhu et al. (2023)) and KgCoOp (Yao et al. (2023b)) explore how to improve the generalization ability of pre-trained models by better preserving their knowledge. ProGrad only updates prompts whose gradients do not conflict with the knowledge of the pre-trained model to prevent general knowledge from being forgotten; KgCoOp uses the gap from the fine-tuned prompts to general knowledge as a regularization term for constrained models. Similarly, RPO (Lee et al. (2023)) leverages masked attention to prevent the internal representation shift in the pre-trained model to reduce the decline in the generalization ability of the pre-trained model. Moreover, TCP (Yao et al. (2023a)) and ProDA (Lu et al. (2022)) choose to enhance the textual side of the representation, with TCP incorporating prior knowledge about classes to enhance the discriminability of classes; ProDA learns output embeddings of textual prompts rather than input embeddings. PromptSRC (Khattak et al. (2023b)) achieves notable performance by simultaneously improving prompt learning from three aspects: preventing pre-trained knowledge forgetting; prompt ensemble; and increasing text diversity to mitigate sample diversity.

In the following, we will introduce the evaluation paradigms, performance metrics, the robustness definitions of prompt learning, benchmark results, and conclusions.

# 3 EVALUATION PARADIGMS

In previous experiments on prompt learning, changes in classes and data distributions during testing were fixed, lacking dynamism. To address this, our benchmark introduces dynamic scenarios where new classes emerge, data distributions shift, and both distributions and classes co-evolve. This enables a more comprehensive analysis of the robustness of existing prompt learning algorithms in such dynamic environments.

# 3.1 DYNAMIC CLASSES CHANGES

In this paradigm, we introduce two scenarios in open environments where new classes continuously emerge. For each experiment, we randomly select half of the classes from a single dataset to serve as base classes, while the remaining half are designated as new classes. The algorithms are trained exclusively on samples from the base classes. During testing, they encounter a dynamic situation in which both base and new classes co-exist, with their quantities continuously changing.

Emerging New Classes In this scenario, during testing, the base classes remain consistently present, while new classes continuously emerge. As new classes are introduced, the algorithm faces a larger class space, making the classification task more challenging. We define the class changing level  $t$  as the ratio of new classes in the test set relative to the base classes, where a higher  $t$  signifies a more complex classification task. We ensure that as  $t$  increases, the new classes from the previous level are subsets of the corresponding groups for the next level.

Varying Ratio of New Classes In this scenario, during testing, the base classes do not remain constant; instead, they decrease as the number of new classes increases. The size of the class space remains unchanged, while the quantities of new and base classes vary synchronously. To achieve this, we define the class changing level  $t$  to represent the proportion of new classes in the test set relative to all classes. As  $t$  increases, new classes continually emerge while base classes diminish. To ensure comparability of performance across different levels of inconsistency, we ensure that as  $t$  increases, the new classes from the previous level are subsets of the corresponding groups for the next level, and the removed base classes are subsets of their respective groups from the previous level.

# 3.2 DYNAMIC DISTRIBUTION SHIFTS

Under this paradigm, we train on all classes of ImageNet and test on a mixture of ImageNet and its variants with continuously changing proportions. We define a distribution change level  $t$  to represent the proportion of samples from the ImageNet variants within the entire test set. As the value of  $t$  increases, the proportion of data from ImageNet decreases, while the proportion of data from the

variants gradually increase. At the same time, we ensure that the class space of the training set remains consistent with the variants. As  $t$  increases, the samples from the ImageNet variants at the previous level are subsets of those at the subsequent level, with the reduced ImageNet samples coming from the previous level.

# 3.3 DYNAMIC CO-EVOLUTION OF DISTRIBUTION AND CLASS VARIATION

In this paradigm, we train on ImageNet and test on a mixed dataset that combines ImageNet and other datasets. The number of classes and samples belonging to ImageNet in the test set is kept equal to those from other datasets, with classes and samples randomly selected. We define a class and distribution change level  $t$  to represent the proportion of cross-dataset samples among all test samples. As  $t$  increases, the proportion of samples from the other datasets also increases. As the value of  $t$  increases, the samples and classes from the cross-dataset continuously increase, while the samples and classes from ImageNet steadily decrease. We ensure that as  $t$  increases, the samples from other datasets at the previous level are subsets of those at the subsequent level, with the reduced ImageNet samples coming from the previous level.

# 4 PERFORMANCE METRICS

To achieve a fair and comprehensive evaluation, for the four evaluation paradigms designed above we introduce a comprehensive set of evaluation metrics to analyze the robustness of prompt learning in VLM. We first define the model accuracy at inconsistency  $t$  as  $Acc(t)$ . Zero-shot pre-trained CLIP also has  $Acc_{zs}(t)$  at any  $t$ , which represents only the accuracy obtained from the same test set at  $t$ , for comparing the performance improvements from other prompt learning methods on the pre-trained model. The accuracy of the model in changing environments is mapped to a function of inconsistency  $t$  of different scenarios. In this way we construct the Dynamic Robustness Curve (DRC) and propose several metrics based on it including 1) Area Under the Curve (AUC) which analyzes the overall robustness of the model; 2) Worst-Case Accuracy (WA) which represents the worst performance in open environments; 3) Expected Variation Magnitude (EVM) measuring the overall magnitude of the change in accuracy; 4) Variation Stability (VS) quantifying the stability of variation magnitude; 5) Positive Area (PA) measuring the performance gain in parts where the algorithm surpasses zero-shot performance; 6) Negative Area (NA) measuring the performance degradation in parts where the algorithm underperforms compared to the zero-shot performance. Table 1 provides a detailed formulation of these metrics.

Table 1: The Definition of Performance Metrics  

<table><tr><td>Metrics</td><td>Formulation</td></tr><tr><td>Area Under the Curve (AUC)</td><td>\(\int_0^1 Acc(t)dt\)</td></tr><tr><td>Worst-Case Accuracy (WA)</td><td>\(min_{t\in [0,1]}Acc(t)\)</td></tr><tr><td>Expected Variation Magnitude (EVM)</td><td>\(\int_0^1 |Acc&#x27;(t)|dt\)</td></tr><tr><td>Variation Stability (VS)</td><td>\(\int_0^1 (Acc&#x27;(t) - \int_0^1 Acc&#x27;(t)dt)^2 dt\)</td></tr><tr><td>Positive Area (PA)</td><td>\(\int_{t\in D}Acc(t) - Acc_{zs}(t)dtD = \{x|Acc(t)\geq Acc_{zs}(t)\}\)</td></tr><tr><td>Negative Area (NA)</td><td>\(\int_{t\in D}Acc_{zs} - Acc(t)(t)dtD = \{x|Acc(t) &lt; Acc_{zs}(t)\}\)</td></tr></table>

Moreover, in order to fairly compare the performance of different methods in scenarios with changing classes, we use Friedman rank (Friedman (1937); Friedman (1940)) to get the average ranks of these methods across different scenarios and different datasets.

$$
r a n k _ {F} = \frac {1}{m} \sum_ {i = 1} ^ {m} r a n k _ {i}
$$

We count the average ranks at the  $6t$ -values settings for each scenario, where  $m = 6$ , and the overall average ranks across  $n$  datasets where  $m = 6 \times n$ .  $\text{rank}_i$  is the rank of a prompt learning method in the  $i$ -th setting. Additionally, we will re-rank the methods to determine the final rank based on the results of the Friedman ranking.

Based on the proposed performance metrics, we further define robust prompt learning in open environments to enhance our understanding of the robustness gain of prompt learning methods compared to zero-shot performance. This includes the concepts of performance-gain robustness and decay-gain-ratio robustness.

Definition (Performance-Gain Robustness) We define the AUC obtained from the VLM's zero-shot prediction as  $AUC_{zs}$ . A prompt learning method  $A$  in VLM returns a model that can be tested with any class and distribution change level  $t$ . If there exists  $\delta_{AUC}$  such that  $AUC - AUC_{zs} \geq \delta_{AUC}$  holds for all  $t$ , we say  $A$  achieves  $\delta_{AUC}$ -performance-gain robustness.

Definition (Decay-Gain-Ratio Robustness) A prompt learning method  $A$  in VLM returns a model that can be tested with any class and distribution change level  $t$ . If there exists  $\delta_{PN}$  such that  $PA - NA \leq \delta_{PN}$  holds for all  $t$ , we say  $A$  exhibits  $\delta_{PN}$ -decay-gain-ratio robustness.

# 5 BENCHMARK RESULTS

# 5.1 EXPERIMENT SETUP

Methods In our experiments, we evaluate 11 prompt learning methods based on the pre-trained CLIP using a Vision Transformer (ViT). The methods are as follows: text-based prompt learning methods CoOp, CoCoOp, ProGrad, KgCoOp, TCP, ProDA, RPO; the visual prompt learning VPT; the text-vision prompt learning methods MaPLe and PromptSRC. We also evaluated the zero-shot prediction capability of the pre-trained model CLIP as a baseline in order to compare the performance of these prompt learning methods.

Datasets Following CoOp and CoCoOp, we evaluate the performance of these prompt learning methods on 11 diverse image classification datasets that cover a variety of recognition tasks. These datasets include: two generic object datasets, ImageNet (Deng et al. (2009)) and Caltech101 (Fei-Fei et al. (2004)); one texture dataset DTD (Cimpoi et al. (2014)); a satellite image dataset EuroSAT (Helber et al. (2019)); five fine-gained dataset FGVCAircraft (Maji et al. (2013)), Food101 (Bossard et al. (2014)), Flowers102 (Nilsback & Zisserman (2008)), OxfordPets (Parkhi et al. (2012)), StanfordCars (Krause et al. (2013)); one scene recognition dataset SUN397 (Xiao et al. (2010)), and an action recognition dataset UCF101 (Soomro et al. (2012)). In the Dynamic Distribution shifts paradigm, we utilize four variants of ImageNet including ImageNetV2 (Recht et al. (2019)), ImageNetSketch (Wang et al. (2019)), ImageNet-A (Hendrycks et al. (2021b)), ImageNet-R (Hendrycks et al. (2021a)).

Implementation Details For all experiments, we adopted a unified parameter setting to ensure a fair comparison. In the two scenarios of Dynamic Classes Changes, we set the learning rate  $\eta$  as  $2 \times 10^{-3}$ , and the total number of epochs is 50 for each dataset. And for mixed proportions of ImageNet variants and the Dynamic Co-evolution of Distribution and Class Variation paradigms, the training on ImageNet is set to run for 10 epochs. Additionally, to ensure a balanced mixture of different datasets, we cap the maximum number of classes and the sample size for all datasets during testing to maintain equivalence. We set the length of text or vision prompts in all methods as 4. We sample 16 samples per class from the training dataset and test all methods on the full test dataset, following the commonly used few-shot evaluation protocol as that in CLIP. We adopt ViT-Base/16 as the backbone network for all experiments. The initial setting for text prompts is fixed to "X X X X" the initialization of vision prompts follows a zero-mean Gaussian distribution with a standard deviation of 0.02. For a fair comparison, the final results were averaged over three rounds of experiments. To plot the curve DAC, we sampled six points for t as 0, 0.2, 0.4, 0.6, 0.8, 1.0. To ensure reliability, the label space is randomized for each round of training and testing, and we conduct experiments three times with seed values of 1, 2, and 3. The results for each sampling point were averaged over three experiments, and linear interpolation was used for the other points. Our experiments are conducted on NVIDIA A800 GPUs. The complete experimental results are presented in the Appendix A.

![](images/beac02896160f4f146ec26e3d99eb1128840f811de65c339f66be629f67ba319.jpg)

![](images/345519c04e5d4136b8663ba53850224ed01131eedec2f1f238fe0914025e99cb.jpg)

![](images/07f23821b000715c0176cd74a1d3f3142882ad48747eca0b58426852bcde832c.jpg)

![](images/366edbfd57b4888945452318537182f213bd261e78d51fb333bc318b291120d3.jpg)

![](images/a18976dadefac460e360eb244b530dc317a640208b163b2fbe0039b49c71239e.jpg)

![](images/e6733131078ac5d03889ca6296228d955c5a98be6633454806213e0c0eea94c0.jpg)

![](images/0c5ceae3376f44fee05ca30b550c1fc54a5585bcb21bc050233d0c662aa659b9.jpg)

![](images/9aafd847e700508d7dc0895f3ba5b6e10bf140e91e3f2906ceba5bbde4ef26b3.jpg)

![](images/b467b30170cf6bc92c5aa68f6d4a5fe6162fe5ba94700732e0cc09621560db24.jpg)

![](images/98317dafb830750d41346964ad47561f8e5b0fccc394a24fc4b45154b1adacdb.jpg)  
Figure 1: Results of prompt learning methods under emerging new classes on 11 datasets. As the value of  $t$  increases, the number of base classes remains constant, while the number of new classes gradually increases until it equals the number of base classes.

![](images/65240deb00247a29b86fdd6c066a583e107040e3bd5c3edf88f882faf03d8cda.jpg)

![](images/b852ec6bc75a7d83adcaeae2ff0db2991de22f155b3f1344933307769a656561.jpg)

# 5.2 PROMPT LEARNING UNDER DYNAMIC CLASSES CHANGES

# Observation

For dynamic class changes, it is challenging for any prompt learning method to consistently achieve optimal performance across different datasets.

Prompt Learning under Emerging New Classes In Figure 1, we present a comparison of different prompt learning methods under emerging new classes, along with their corresponding outcomes. It can be observed that as more new classes are introduced during testing, the performance of various prompt learning methods generally shows a gradually declining trend. However, the speed of this decline varies between different methods, leading to changes in the relative performance of these methods as new classes continue to emerge. For example, in the FGVCAircraft dataset, the performance of PromptSRC initially falls significantly behind that of MaPLe and VPT. However, as the classification pressure increases, the performance of PromptSRC begins to surpass these methods and maintains the lead among all the methods. This indicates that, it is challenging for any single method to consistently maintain optimal performance across different open environments.

In Table 2, we present the evaluation results of the metrics on DTD under emerging new classes. We can see that, under these evaluation paradigms, metrics like  $Acc(0)$ , AUC, EVM, and VS are often not necessarily related. A method with a higher  $Acc(0)$  may even perform worse than CLIP in other metrics. And it's worth noting that CLIP often surpasses most methods in terms of EVM.

Prompt Learning under Varying Ratio of New Classes In Figure 2, we compare various prompts learning methods under varying ratio of new classes and get the results. We can find that the

![](images/61e8338dbf61bd7d02dec0a90f032815d75422fd80ebc5fc3619ae42b3b834a1.jpg)

![](images/b5d2d534a9b8823d4edffb391f9e592c2016b4be99cbdba52f0e01b4e77128e2.jpg)

![](images/de124ad80dba51e5e2c18a5a54d74655e92cc36f5dc62228bbe38be4a652155e.jpg)

![](images/1eab3efe7d1e2b2e3b8a30e56cb492ffda58399eb2527c958056311c838bac1f.jpg)

![](images/c3c16dff504db3ba3a718e4d298d4d011362c947f030de8782fe187769928c1b.jpg)

![](images/7b0c8c776ef4f24fc8bbad192fce09459c8e2f911a6430a0ae235956fc097b4c.jpg)

![](images/6ed934856e94936227727e52504b58753edacf4fcd2a94cb53325d008ed11302.jpg)

![](images/8be7b1d9583fd8aa7eeacba3714e4f3155493d908b367dda476e9f732f6c683f.jpg)

![](images/57cc33e48232df29c446d6f4a223ccc90e221de1dfe8555b5744a3da0f1205ed.jpg)

![](images/4b0e40d52f450f4513f7c7609ada5fcd1abd01931762f27ddabd42f650aaee04.jpg)  
Figure 2: Results of prompt learning methods under varying ratio of new classes on 11 datasets. As the value of  $t$  increases, the number of base classes gradually decreases while the number of new classes gradually increases. Always keep the total number of classes constant.

![](images/f217867d36488bb8f680452b83ea10df02ace53d62017f94085bd0c04b0ea658.jpg)

![](images/21eb6dbb09d532e7b651d089c8ea8e5a9f3ff72cb21f37205127ec2fefd5b221.jpg)

Table 2: Evaluation on DTD under Emerging New Classes. In the table, higher values for the metrics Acc(0), AUC, WA, and PA are better, while lower values for EVM, VS, and NA are preferable.  

<table><tr><td>Dataset</td><td>Methods</td><td>Acc(0)</td><td>AUC</td><td>WA</td><td>EVM</td><td>VS</td><td>PA</td><td>NA</td></tr><tr><td rowspan="11">DTD</td><td>CLIP(Zero-shot)</td><td>0.568</td><td>0.489</td><td>0.441</td><td>0.127</td><td>0.003</td><td>/</td><td>/</td></tr><tr><td>CoOp</td><td>0.767</td><td>0.469</td><td>0.410</td><td>0.357</td><td>0.333</td><td>0.015</td><td>0.032</td></tr><tr><td>CoCoOp</td><td>0.762</td><td>0.597</td><td>0.488</td><td>0.274</td><td>0.011</td><td>0.108</td><td>0.000</td></tr><tr><td>VPT</td><td>0.799</td><td>0.618</td><td>0.522</td><td>0.277</td><td>0.023</td><td>0.129</td><td>0.000</td></tr><tr><td>MaPLe</td><td>0.789</td><td>0.633</td><td>0.532</td><td>0.257</td><td>0.011</td><td>0.144</td><td>0.000</td></tr><tr><td>ProGrad</td><td>0.779</td><td>0.629</td><td>0.537</td><td>0.242</td><td>0.012</td><td>0.140</td><td>0.000</td></tr><tr><td>KgCoOp</td><td>0.777</td><td>0.637</td><td>0.547</td><td>0.230</td><td>0.010</td><td>0.147</td><td>0.000</td></tr><tr><td>RPO</td><td>0.812</td><td>0.655</td><td>0.561</td><td>0.251</td><td>0.013</td><td>0.166</td><td>0.000</td></tr><tr><td>PromptSRC</td><td>0.807</td><td>0.661</td><td>0.576</td><td>0.231</td><td>0.013</td><td>0.172</td><td>0.000</td></tr><tr><td>ProDA</td><td>0.779</td><td>0.657</td><td>0.577</td><td>0.202</td><td>0.007</td><td>0.168</td><td>0.000</td></tr><tr><td>TCP</td><td>0.779</td><td>0.642</td><td>0.556</td><td>0.223</td><td>0.012</td><td>0.152</td><td>0.000</td></tr></table>

accuracy of the model is not always monotonically declining as new classes increases and base classes decreases, but instead varying performance changes occur under different ratios of new classes to base classes. For example, in the EuroSAT dataset, the performance of VPT and TCP does not always decrease with the increasing ratio of new classes to base classes; There are instances where the performance actually improves with more new classes and fewer base classes. In real-world scenarios where the label space is unknown, it's not always the case that having fewer new classes and more base classes will lead to better performance. Unknown mixed class proportions can sometimes actually be more detrimental to the performance than having more new classes. Moreover, on each dataset, there does not exist a certain method that can be optimal under various ratios of new classes, e.g., on the Stanford-cars dataset, the best in terms of accuracy goes from MaPLe to PromptSRC to ProDA as the value of t increases.

Table 3: Evaluation on FGVCAircraft under Varying Ratio of New Classes  

<table><tr><td>Dataset</td><td>Methods</td><td>Acc(0)</td><td>AUC</td><td>WA</td><td>EVM</td><td>VS</td><td>PA</td><td>NA</td></tr><tr><td rowspan="11">FGVCAircraft</td><td>CLIP(Zero-shot)</td><td>0.338</td><td>0.362</td><td>0.338</td><td>0.034</td><td>0.003</td><td>/</td><td>/</td></tr><tr><td>CoOp</td><td>0.534</td><td>0.286</td><td>0.240</td><td>0.332</td><td>0.303</td><td>0.013</td><td>0.081</td></tr><tr><td>CoCoOp</td><td>0.505</td><td>0.347</td><td>0.226</td><td>0.279</td><td>0.008</td><td>0.031</td><td>0.034</td></tr><tr><td>VPT</td><td>0.587</td><td>0.459</td><td>0.376</td><td>0.210</td><td>0.012</td><td>0.097</td><td>0.000</td></tr><tr><td>MaPLe</td><td>0.599</td><td>0.440</td><td>0.311</td><td>0.288</td><td>0.005</td><td>0.085</td><td>0.006</td></tr><tr><td>ProGrad</td><td>0.520</td><td>0.423</td><td>0.350</td><td>0.170</td><td>0.003</td><td>0.062</td><td>0.001</td></tr><tr><td>KgCoOp</td><td>0.492</td><td>0.413</td><td>0.356</td><td>0.136</td><td>0.005</td><td>0.051</td><td>0.000</td></tr><tr><td>RPO</td><td>0.536</td><td>0.447</td><td>0.392</td><td>0.155</td><td>0.009</td><td>0.085</td><td>0.000</td></tr><tr><td>PromptSRC</td><td>0.564</td><td>0.478</td><td>0.426</td><td>0.139</td><td>0.006</td><td>0.116</td><td>0.000</td></tr><tr><td>ProDA</td><td>0.495</td><td>0.439</td><td>0.411</td><td>0.088</td><td>0.007</td><td>0.077</td><td>0.000</td></tr><tr><td>TCP</td><td>0.526</td><td>0.445</td><td>0.394</td><td>0.132</td><td>0.006</td><td>0.084</td><td>0.000</td></tr></table>

As shown in Table 3, the performance variability and stability of the algorithms under the Varying Ratio of New Classes paradigm on FGVCAircraft are worse than that of CLIP's zero-shot predictions. Additionally, no algorithm has managed to maintain excellent performance on base classes while also demonstrating a slower and more stable decline in performance.

# 5.3 PROMPT LEARNING UNDER DYNAMIC DISTRIBUTION SHIFTS

# Observation

There is no significant improvement across algorithms when addressing the issue of dynamic data distribution shifts.

As shown in Figure 3, we can clearly observe that for different ImageNet variants, the performance decline of various prompt learning methods, as the proportion of variant data increases, almost mirrors the zero-shot predictions of CLIP. As indicated in Table 4, the metrics representing the degree of change and stability, such as EVM and VS, show minimal differences. Prompt learning does not demonstrate strong performance when confronted with changes in data distribution. The slight performance gains on varying data distributions are primarily attributed to improved fine-tuning on ImageNet, but they do little to mitigate the performance degradation trend in the Dynamic Distribution shifts paradigm.

![](images/81327faa9d32c1967bbeb9e14c3eb1d0c1d65f441ceab36c353be15d7fba90f4.jpg)  
Figure 3: Results of prompt learning methods under Dynamic Distribution shifts.

Table 4: Evaluation on ImageNet-R under Dynamic Distribution shifts  

<table><tr><td>Dataset</td><td>Methods</td><td>Acc(0)</td><td>AUC</td><td>WA</td><td>EVM</td><td>VS</td><td>PA</td><td>NA</td></tr><tr><td rowspan="11">ImageNet-R</td><td>CLIP(Zero-shot)</td><td>0.915</td><td>0.797</td><td>0.740</td><td>0.175</td><td>0.014</td><td>/</td><td>/</td></tr><tr><td>CoOp</td><td>0.927</td><td>0.815</td><td>0.761</td><td>0.166</td><td>0.013</td><td>0.017</td><td>0.000</td></tr><tr><td>CoCoOp</td><td>0.928</td><td>0.818</td><td>0.764</td><td>0.164</td><td>0.013</td><td>0.020</td><td>0.000</td></tr><tr><td>VPT</td><td>0.920</td><td>0.809</td><td>0.755</td><td>0.166</td><td>0.012</td><td>0.012</td><td>0.000</td></tr><tr><td>MaPLe</td><td>0.930</td><td>0.816</td><td>0.762</td><td>0.168</td><td>0.013</td><td>0.019</td><td>0.000</td></tr><tr><td>ProGrad</td><td>0.925</td><td>0.818</td><td>0.766</td><td>0.159</td><td>0.011</td><td>0.021</td><td>0.000</td></tr><tr><td>KgCoOp</td><td>0.929</td><td>0.821</td><td>0.768</td><td>0.160</td><td>0.012</td><td>0.024</td><td>0.000</td></tr><tr><td>RPO</td><td>0.928</td><td>0.818</td><td>0.763</td><td>0.165</td><td>0.012</td><td>0.020</td><td>0.000</td></tr><tr><td>PromptSR</td><td>0.928</td><td>0.826</td><td>0.776</td><td>0.152</td><td>0.010</td><td>0.029</td><td>0.000</td></tr><tr><td>ProDA</td><td>0.929</td><td>0.827</td><td>0.779</td><td>0.150</td><td>0.011</td><td>0.030</td><td>0.000</td></tr><tr><td>TCP</td><td>0.927</td><td>0.819</td><td>0.766</td><td>0.160</td><td>0.011</td><td>0.022</td><td>0.000</td></tr></table>

![](images/cc9c9c9cf9dc6f08080ab10ab7dfc0d48aa96291d9d14919d1ecd80e7460bff0.jpg)  
Figure 4: Results of prompt learning methods cross various datasets.

Table 5: Evaluation Metrics under Dynamic Co-evolution of Distribution and Class Variation  

<table><tr><td>Dataset</td><td>Methods</td><td>Acc(0)</td><td>AUC</td><td>WA</td><td>EVM</td><td>VS</td><td>PA</td><td>NA</td></tr><tr><td rowspan="11">EuroSAT</td><td>CLIP(Zero-shot)</td><td>0.986</td><td>0.629</td><td>0.400</td><td>0.586</td><td>0.100</td><td>/</td><td>/</td></tr><tr><td>CoOp</td><td>0.995</td><td>0.650</td><td>0.451</td><td>0.544</td><td>0.104</td><td>0.021</td><td>0.000</td></tr><tr><td>CoCoOp</td><td>0.992</td><td>0.649</td><td>0.445</td><td>0.547</td><td>0.151</td><td>0.022</td><td>0.001</td></tr><tr><td>VPT</td><td>0.991</td><td>0.581</td><td>0.407</td><td>0.584</td><td>0.275</td><td>0.000</td><td>0.023</td></tr><tr><td>MaPLe</td><td>0.994</td><td>0.629</td><td>0.458</td><td>0.537</td><td>0.225</td><td>0.012</td><td>0.001</td></tr><tr><td>ProGrad</td><td>0.992</td><td>0.643</td><td>0.417</td><td>0.575</td><td>0.114</td><td>0.015</td><td>0.000</td></tr><tr><td>KgCoOp</td><td>0.992</td><td>0.657</td><td>0.437</td><td>0.555</td><td>0.094</td><td>0.028</td><td>0.000</td></tr><tr><td>RPO</td><td>0.994</td><td>0.640</td><td>0.431</td><td>0.563</td><td>0.124</td><td>0.012</td><td>0.000</td></tr><tr><td>PromptSRC</td><td>0.994</td><td>0.658</td><td>0.449</td><td>0.545</td><td>0.094</td><td>0.029</td><td>0.000</td></tr><tr><td>ProDA</td><td>1.000</td><td>0.644</td><td>0.494</td><td>0.534</td><td>0.257</td><td>0.025</td><td>0.003</td></tr><tr><td>TCP</td><td>0.994</td><td>0.630</td><td>0.429</td><td>0.564</td><td>0.163</td><td>0.007</td><td>0.001</td></tr></table>

# 5.4 PROMPT LEARNING UNDER DYNAMIC CO-EVOLUTION OF DISTRIBUTION AND CLASS VARIATION

# Observation

Prompt learning exhibits performance nearly on par with the zero-shot prediction capabilities of CLIP, showing little to no improvement in scenarios characterized by the coupling of dynamic distribution and class changes.

As shown in Figure 4, for relatively simple classification datasets like Caltech101 and OxfordPets, the performance variations under different cross-dataset ratios are minimal. However, for datasets where CLIP already exhibits significant performance degradation across datasets, the various algorithms face similar challenges, as observed in the Dynamic Distribution Shifts paradigm, showing no notable performance improvements compared to CLIP's zero-shot predictions. Furthermore, as illustrated in Figure 5, the differences in metrics such as AUC and EVM between the algorithms and zero-shot predictions under EuroSAT are minor, while the stability of algorithm performance, as reflected in the VS metric, generally shows varying degrees of decline.

# 5.5 ROBUSTNESS ANALYSIS OF PROMPT LEARNING METHODS

In Tables 6 & 9, we compare the robustness of different algorithms across various dynamic scenarios. Tables 6 and 7 reveal that CoOp exhibits the poorest robustness when facing various class changes, as it lacks any robustness handling mechanisms as an initial prompt learning method. Similarly, CoCoOp, VPT, and MaPLe demonstrate comparable poor performance. While these methods continuously improve prompt formulation and somewhat enhance the model's ability to extract image features,

Table 6: Average Robustness and Ranks under Table 7: Average Robustness and Ranks under Emerging New Classes. Varying Ratio of New Classes.  

<table><tr><td>Methods</td><td>δAUC</td><td>δPN</td><td>Friedman rank</td><td>Final rank</td><td>Methods</td><td>δAUC</td><td>δPN</td><td>Friedman rank</td><td>Final rank</td></tr><tr><td>CLIP(Zero-shot)</td><td>/</td><td>/</td><td>9.788</td><td>10</td><td>CLIP(Zero-shot)</td><td>/</td><td>/</td><td>8.152</td><td>9</td></tr><tr><td>CoOp</td><td>-0.013</td><td>-0.012</td><td>9.939</td><td>11</td><td>CoOp</td><td>-0.014</td><td>-0.012</td><td>8.970</td><td>11</td></tr><tr><td>CoCoOp</td><td>0.053</td><td>0.053</td><td>8.212</td><td>9</td><td>CoCoOp</td><td>0.013</td><td>0.016</td><td>8.742</td><td>10</td></tr><tr><td>VPT</td><td>0.071</td><td>0.071</td><td>6.636</td><td>8</td><td>VPT</td><td>0.037</td><td>0.039</td><td>7.288</td><td>8</td></tr><tr><td>MaPLe</td><td>0.081</td><td>0.081</td><td>6.136</td><td>7</td><td>MaPLe</td><td>0.045</td><td>0.046</td><td>6.879</td><td>7</td></tr><tr><td>ProGrad</td><td>0.072</td><td>0.072</td><td>5.742</td><td>5</td><td>ProGrad</td><td>0.040</td><td>0.041</td><td>6.379</td><td>6</td></tr><tr><td>KgCoOp</td><td>0.072</td><td>0.072</td><td>5.909</td><td>6</td><td>KgCoOp</td><td>0.045</td><td>0.046</td><td>5.758</td><td>5</td></tr><tr><td>RPO</td><td>0.083</td><td>0.083</td><td>4.227</td><td>4</td><td>RPO</td><td>0.053</td><td>0.054</td><td>4.455</td><td>4</td></tr><tr><td>PromptSRC</td><td>0.102</td><td>0.102</td><td>1.955</td><td>1</td><td>PromptSRC</td><td>0.074</td><td>0.074</td><td>2.212</td><td>1</td></tr><tr><td>ProDA</td><td>0.081</td><td>0.081</td><td>3.500</td><td>2</td><td>ProDA</td><td>0.059</td><td>0.059</td><td>3.273</td><td>2</td></tr><tr><td>TCP</td><td>0.083</td><td>0.083</td><td>3.955</td><td>3</td><td>TCP</td><td>0.057</td><td>0.057</td><td>3.894</td><td>3</td></tr></table>

Table 8: Average Robustness and Ranks under Dynamic Distribution shifts.  

<table><tr><td>Methods</td><td>δAUC</td><td>δPN</td><td>Friedman rank</td><td>Final rank</td></tr><tr><td>CLIP(Zero-shot)</td><td>/</td><td>/</td><td>10.875</td><td>11</td></tr><tr><td>CoOp</td><td>0.031</td><td>0.031</td><td>4.625</td><td>5</td></tr><tr><td>CoCoOp</td><td>0.030</td><td>0.030</td><td>5.208</td><td>6</td></tr><tr><td>VPT</td><td>0.011</td><td>0.011</td><td>10.125</td><td>10</td></tr><tr><td>MaPLe</td><td>0.031</td><td>0.031</td><td>4.583</td><td>4</td></tr><tr><td>ProGrad</td><td>0.026</td><td>0.026</td><td>7.917</td><td>9</td></tr><tr><td>KgCoOp</td><td>0.032</td><td>0.032</td><td>3.375</td><td>2</td></tr><tr><td>RPO</td><td>0.028</td><td>0.028</td><td>7.250</td><td>8</td></tr><tr><td>PromptSRC</td><td>0.033</td><td>0.033</td><td>3.917</td><td>3</td></tr><tr><td>ProDA</td><td>0.035</td><td>0.035</td><td>1.958</td><td>1</td></tr><tr><td>TCP</td><td>0.029</td><td>0.029</td><td>6.167</td><td>7</td></tr></table>

Table 9: Average Robustness and Ranks under Dynamic Co-evolution of Distribution and Class Variation.  

<table><tr><td>Methods</td><td>δAUC</td><td>δPN</td><td>Friedman rank</td><td>Final rank</td></tr><tr><td>CLIP(Zero-shot)</td><td>/</td><td>/</td><td>8.400</td><td>10</td></tr><tr><td>CoOp</td><td>0.012</td><td>0.013</td><td>4.400</td><td>4</td></tr><tr><td>CoCoOp</td><td>0.012</td><td>0.013</td><td>4.267</td><td>2</td></tr><tr><td>VPT</td><td>-0.004</td><td>0.000</td><td>9.450</td><td>11</td></tr><tr><td>MaPLe</td><td>0.007</td><td>0.009</td><td>5.833</td><td>6</td></tr><tr><td>ProGrad</td><td>0.007</td><td>0.008</td><td>7.450</td><td>9</td></tr><tr><td>KgCoOp</td><td>0.013</td><td>0.013</td><td>4.467</td><td>5</td></tr><tr><td>RPO</td><td>0.008</td><td>0.008</td><td>6.667</td><td>8</td></tr><tr><td>PromptSRC</td><td>0.013</td><td>0.014</td><td>4.200</td><td>1</td></tr><tr><td>ProDA</td><td>0.011</td><td>0.013</td><td>4.383</td><td>3</td></tr><tr><td>TCP</td><td>0.008</td><td>0.009</td><td>6.483</td><td>7</td></tr></table>

they fail to consider generalization to new classes and do not specifically address robustness in downstream tasks. In contrast, methods such as RPO, ProDA, TCP, and PromptSRC, which aim to enhance the model's ability to extract and differentiate text features, achieve better robustness in these scenarios and consistently rank highly in overall performance. We contend that improving the model's capability to extract text features and distinguish between classes contributes to generalization in real-world scenarios with unknown label spaces.

In Table 8, we observe that under the Dynamic Distribution Shifts scenario, the robustness of most methods shows little variation, except for VPT, which significantly underperforms compared to the others. In this paradigm, algorithms like TCP and RPO, which perform well under class changes, do not guarantee similarly strong performance; conversely, the earlier methods CoOp and CoCoOp actually perform better than these newer approaches.

In Table 9, under the Dynamic Co-evolution of Distribution and Class Variation paradigm, ProDA and PromptSRC continue to demonstrate good robustness and high rankings. Aside from VPT, the earlier algorithms CoOp and CoCoOp increasingly outperform other methods. This indicates that most algorithms fundamentally lack the capability to effectively address the challenges posed by cross-dataset variation.

# 6 CONCLUSION

Research on robust prompt learning is an important step toward more practical tasks of VLM. This paper provides a new benchmark to evaluate the robustness of prompt learning in open environments, which includes dynamic class changes, dynamic distribution shifts, and dynamic co-evolution of both distributions and classes. We present several new performance metrics to help analyze the robustness and conduct experiments on commonly adopted prompt learning methods. The results reveal that current prompt learning methods in VLMs are not robust to class and data distribution changes. On the contrary, they highly rely on the zero-shot ability of CLIP and show no significant robust improvement compared to baseline zero-shot performance. Of course, the issues models face in real-world environments may be more complex than the paradigms we have proposed. We hope that our work can help promote the study of prompt learning in real-world scenarios.

# REFERENCES

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: A visual language model for few-shot learning. In Advances in Neural Information Processing Systems, pp. 23716-23736, 2022.  
Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101-mining discriminative components with random forests. Computer Vision-ECCV 2014, pp. 446-461, 2014.  
Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3606-3613, 2014.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248-255, 2009.  
Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11583-11592, 2022.  
Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14084-14093, 2022.  
Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 Conference on Computer Vision and Pattern Recognition Workshop, pp. 178-178, 2004.  
Milton Friedman. The use of ranks to avoid the assumption of normality implicit in the analysis of variance. Journal of the American Statistical Association, 32(200):675-701, 1937.  
Milton Friedman. A comparison of alternative tests of significance for the problem of m rankings. The Annals of Mathematical Statistics, 11(1):86-92, 1940.  
Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. In International Journal of Computer Vision, pp. 581-595, 2024.  
Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. In arXiv preprint arXiv:2104.13921, 2021.  
Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. In IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, pp. 2217-2226, 2019.  
Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 8340-8349, 2021a.  
Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15262-15271, 2021b.  
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pp. 4904-4916, 2021.  
Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pp. 709-727, 2022.

Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19113-19122, 2023a.  
Muhammad Uzair Khattak, Syed Talal Wasim, Muzammal Naseer, Salman Khan, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Self-regulating prompts: Foundational model adaptation without forgetting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15190-15200, 2023b.  
Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 554-561, 2013.  
Dongjun Lee, Seokwon Song, Jihee Suh, Joonmyeong Choi, Sanghyeok Lee, and Hyunwoo J Kim. Read-only prompt optimization for vision-language few-shot learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1401-1411, 2023.  
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888-12900, 2022.  
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. In ACM Computing Surveys, pp. 1-35, 2023.  
Yuning Lu, Jianzhuang Liu, Yonggang Zhang, Yajing Liu, and Xinmei Tian. Prompt distribution learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5206-5215, 2022.  
Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. In arXiv preprint arXiv:1306.5151, 2013.  
Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pp. 722-729, 2008.  
Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pp. 3498-3505, 2012.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pp. 8748-8763, 2021.  
Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize toImagenet? In International conference on machine learning, pp. 5389-5400, 2019.  
Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. In arXiv preprint arXiv:1212.0402, 2012.  
Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019.  
Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pp. 3485-3492, 2010.  
Hantao Yao, Rui Zhang, and Changsheng Xu. TCP: Textual-based class-aware prompt tuning for visual-language model. arXiv preprint arXiv:2311.18231, 2023a.  
Hantao Yao, Rui Zhang, and Changsheng Xu. Visual-language prompt tuning with knowledge-guided context optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6757-6767, 2023b.

Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In arXiv preprint arXiv:2111.07783, 2021.  
Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18123-18133, 2022.  
Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. In arXiv preprint arXiv:2111.03930, 2021.  
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16816-16825, 2022a.  
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337-2348, 2022b.  
Zhi-Hua Zhou. Open-environment machine learning. National Science Review, 9(8):nwac123, 2022.  
Beier Zhu, Yulei Niu, Yucheng Han, Yue Wu, and Hanwang Zhang. Prompt-aligned gradient for prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15659-15669, 2023.
