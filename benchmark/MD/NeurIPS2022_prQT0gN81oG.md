# Shadow Knowledge Distillation: Bridging Offline and Online Knowledge Transfer

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Knowledge distillation can be generally divided into offline and online categories according to whether teacher model is pre-trained and persistent during the distillation process. Offline distillation can employ existing models yet always demonstrates inferior performance than online ones. In this paper, we first empirically show that the essential factor for their performance gap lies in the reversed distillation from student to teacher, rather than the training fashion. Offline distillation can achieve competitive performance gain by fine-tuning pre-trained teacher to adapt student with such reversed distillation. However, this fine-tuning process still costs lots of training budgets. To alleviate this dilemma, we propose SHAKE, a simple yet effective SHAdow Knowledge Edge transfer framework to bridge offline and online distillation, which trades the accuracy with efficiency. Specifically, we build an extra shadow head on the student's backbone to mimic the predictions of pre-trained teacher as its shadow. Then, this shadow head is leveraged as a proxy teacher to perform bidirectional distillation with student on the fly. In this way, SHAKE not only updates this student-aware proxy teacher with the knowledge of pre-trained model but also greatly optimizes costs of augmented reversed distillation. Extensive experiments on classification and object detection tasks demonstrate that our technique achieves state-of-the-art results with different CNNs and Vision Transformer models. Additionally, our method shows strong compatibility with multi-teacher and augmentation strategies by gaining additional performance improvement. Code will be made publicly.

# 1 Introduction

Deep Neural Networks (DNNs) have achieved great success in tackling a variety of tasks [19, 10]. Despite appealing performance, prevailing DNN models usually have large numbers of parameters, bringing heavy costs of computation. To alleviate this problem, many network compression methods [21, 28] have been proposed, among which Knowledge Distillation (KD) [14] has recently attracted increased attention.

The aim of KD is to transfer the learnt knowledge of a high-capacity teacher model to a low-capacity student model. Numerous offline methods [1, 14] use a two-stage training process that begins with training a teacher model and then keeping it fixed to distill student model (see KD in Figure 1 (a)). Besides offline fashion, recent online methods [20, 4] adopt a one-stage training process, jointly training the student and teacher/peer models using bidirectional distillation like DML [39] in Figure 1 (b). These online distillations always surpass the offline distillations under the same teacher model. However, some large teacher models trained from scratch would bring some difficulties (e.g., high computational resources and unstable optimizations) for distillation, especially for tasks that rely on large transformers (e.g., BERT [36], GPT-3 [2] and ViT-MoE [30]). Therefore, two questions

![](images/612446f5046fe73760e30648ec543356e6a6c71f8a2bf496256213b3818b7d73.jpg)  
Figure 1: Illustration of (a) KD, (b) DML, (c)  $\mathrm{DML}^{\dagger}$  without  $KD_{S\rightarrow T}$ , and (d)  $\mathrm{KD}^{\dagger}$  with  $KD_{S\rightarrow T}$ .  $KD_{S\rightarrow T}$  denotes reversed distillation from student to teacher.

Table 1: Left: comparison of training time, Top-1 accuracy  $(\%)$ , and teacher-student gap (T-S gap) among the (a) KD, (b) DML, (c) DML $^{\dagger}$  without  $KD_{S\rightarrow T}$ , (d) KD $^{\dagger}$  with  $KD_{S\rightarrow T}$ , and our SHAKE for ResNet-20  $(69.09\%)$  via pre-trained teacher (T) ResNet-110  $(74.31\%)$  on CIFAR-100. Training time is measured on a single 2080Ti GPU and  $\times$  represents the improving ratios than KD. The teacher-student gap [5] is defined as KL divergence between their outputs (lower is better). Right: training time and accuracy of different settings for KD $^{\dagger}$  and SHAKE. KD $^{\dagger}$  (head) means only updating head, and KD $^{\dagger}$  (60) means only updating for 60 epochs. SHAKE (share) denotes proxy teacher shares student's backbone. SHAKE (multi) denotes SHAKE with two ResNet-110 as teachers.

<table><tr><td>Method</td><td>T</td><td>KL</td><td>KL</td><td>Time</td><td>Top-1</td><td>T-S gap</td></tr><tr><td>(A) KD</td><td>✓</td><td>✓</td><td>✗</td><td>×1.00</td><td>70.66</td><td>1.12</td></tr><tr><td>(B) DML</td><td>✗</td><td>✓</td><td>✓</td><td>×4.32</td><td>71.52</td><td>0.28</td></tr><tr><td>(C) DML†</td><td>✗</td><td>✓</td><td>✗</td><td>×4.41</td><td>70.55</td><td>0.62</td></tr><tr><td>(D) KD†</td><td>✓</td><td>✓</td><td>✓</td><td>×4.29</td><td>71.76</td><td>0.76</td></tr><tr><td>(G) SHAKE</td><td>✓</td><td>✓</td><td>✓</td><td>×1.18</td><td>72.02</td><td>0.21</td></tr></table>

<table><tr><td>Method</td><td>Time</td><td>Top-1</td></tr><tr><td>(F) KD† (head)</td><td>×1.13</td><td>71.05</td></tr><tr><td>(G) KD† (60)</td><td>×2.12</td><td>71.13</td></tr><tr><td>(H) SHAKE (alone)</td><td>×1.85</td><td>71.82</td></tr><tr><td>(I) SHAKE (share)</td><td>×1.18</td><td>72.02</td></tr><tr><td>(J) SHAKE (Multi)</td><td>×1.32</td><td>72.55</td></tr></table>

naturally arise: (1) Why is there a performance gap between offline and online distillation? (2) How the performance of offline distillation can be advanced with design techniques of online KD methods?  
To clarify the first question, we compare KD and DML regarding the training fashion of teacher and distillation loss in Table 1. Contrary to the common belief, we empirically observe that training fashion may not affect the distillation performance since  $\mathrm{DML}^{\dagger}$  obtains similar performance as KD (70.55% vs. 70.66%). Instead, the reversed distillation from the student model yields significant accuracy gains for  $\mathrm{KD}^{\dagger}$  than KD (71.76% vs. 70.66%) and DML than  $\mathrm{DML}^{\dagger}$  (71.52% vs. 70.55%).  
Then, we analyze the output discrepancy of the teacher-student for different methods, and the reversed distillation reduces the gap from 1.12 of KD to 0.76 of  $\mathrm{KD}^{\dagger}$ . Therefore, the main reasons behind the performance gap lie in two aspects: (a) Teacher models in conventional offline pipelines are not optimized for the student model. Thus, they could only provide general knowledge, which may be suboptimal for the particular student. (b) The reversed distillation changes the universal teacher model into a student-aware one and bridges the teacher-student gap, which clearly justifies:

Teachers should teach students in accordance with their aptitude and shouldn't follow the same pattern.

For the second question, fine-tuning pre-trained teacher model with reversed distillation is a straightforward way to bridge the two training fashions. However, in most scenarios, fine-tuning the whole network still needs lots of training budges (more  $4 \times$  costs than KD in Table 1). Some trade-offs in reducing the fine-tuning parameters or epochs involve performance loss. Thus, how to augment reversed distillation without much extra overhead is an important issue for the application. Besides fine-tuning, building a proxy teacher model (see Figure 2 (a)) to inherit the knowledge of pre-trained models and receive reversed distillation from students also enjoys the same benefits. As shown in Table 1 (H), this proxy teacher model (SHAKE (alone)) can achieve competitive gains than fine-tuning the whole teacher model  $(\mathrm{KD}^{\dagger})$ . Recent weight sharing strategies in AutoML and can effectively save training overhead and accelerate the convergence of the model. This encourages us to adopt the same student architecture and weight sharing strategy to generate the proxy teacher model. Thus, we allow this proxy teacher to share the student's backbone but use an individual shadow head to preserve the diversity of logits representations (see Figure 2 (b)). As shown in Table 1 (I), such

![](images/b9b6dda565b4f7519b88d511972ed031488b7d0bb34fdadd48609cbfe66d22e0.jpg)  
Figure 2: Evolution of our SHAdow Knowledge distillation (SHAKE). (a) We build a proxy teacher model to inherit knowledge from pre-trained models as an alternative to costly fine-tuning teacher models. (b) This proxy teacher model could adopt the same student structure and reuse the student's backbone to reduce training budgets. (c) Our SHAKE framework: Base on (b), SHAKE uses the shadow head as the proxy teacher to perform bidirectional KD with student, updated with the soft label of pre-trained teacher. Both teacher models and shadow head are present only during training and can be discarded during inference. The implementations of (b) and (c) are identical. (d) Our SHAKE for multiple teachers: SHAKE leverages multiple shadow heads to individually follow various teacher models.

sharing strategy in SHAKE (share) presents three benefits: (1) more than  $3 \times$  training acceleration than DML and fine-tuning whole teacher model. (2) No need for architecture selection costs for proxy teacher. (3) Additional accuracy gains than individual proxy teacher because knowledge inherited by proxy teachers also directly improves the representation of students' backbone in the weight-sharing process.

Based on the above observations, we propose SHAdow Knowledge distillation (SHAKE), a novel and effective logits distillation framework. Our SHAKE builds a proxy teacher model and updates its weights via the original teacher model predictions. In this way, SHAKE enjoys the same benefits with teacher fine-tuning to mine the knowledge of the pre-trained model and can perform bidirectional supervision with student. To optimize training costs, this proxy teacher shares the backbone with the students but leverages an individual head to preserve the diversity of logits representations. This head is named shadow head since it imitates the original teacher model just like its shadow. During training, we only need to additionally train this shadow head with small training budgets (only  $1.18 \times$  costs than KD). After training, all teacher models and shadow head can be deprecated, and student model can be separately applied for inference without any overhead. Moreover, we extend the SHAKE to multi-teacher model scenarios using multiple shadow heads to inherit diverse knowledge.

In principle, SHAKE alters the chain of knowledge transfer from pre-trained teacher  $\rightarrow$  student in KD to pre-trained teacher  $\rightarrow$  proxy teacher  $\Longrightarrow$  student. Other adaptive KD [25] only employs middle networks to distill as large teacher  $\rightarrow$  middle teacher  $\rightarrow$  student, which is not optimized for students and requires multi-step training. The merits of SHAKE lie in three-fold. First, it effectively reduces the teacher-student capability gap with reversed distillation, bringing significant gains when pretrained teacher models are available. Second, for the scenario without pre-trained teacher models, SHAKE also enables the offline KD methods more effective to alleviate the unstable optimization issues of online KD methods. Thus, SHAKE bridges offline and online KD methods and enjoys the advantages of both methods. Third, SHAKE achieves favorable trade-offs between accuracy and training budget. By contrast, other adaptive KD [25] needs sequentially training multiple models with lots of additional training time and resources. We hope that these intriguing observations in SHAKE would expand the application of KD and facilitate future research for KD work to some extent.

We conduct extensive experiments on classification and detection to verify the superiority of the proposed method. SHAKE achieves a consistent and significant accuracy boost in various neural networks and data augmentations, which outperforms other methods by large margins. For example, SHAKE obtains  $1.21\% \sim 3.65\%$  accuracy gains and  $3\times \sim 4\times$  training acceleration than DML on CIFAR-100. On the challenging ImageNet dataset, SHAKE can improve the Top-1 accuracy of ResNet-18 from  $69.75\%$  to  $72.37\%$  and MobileNet from  $70.13\%$  to  $72.96\%$ , which are state-of

the-art performances among KD methods. On vision transformer architecture [34], our approach achieves  $75.22\%$  Top-1 accuracy and  $2.76\%$  gain for training a VIT-T model. On the object detection task, SHAKE improves the AP by 1.20 for RetinaNet and 1.02 for Faster R-CNN on MS-COCO dataset, demonstrating the generality of our approach.

In summary, we make the following principle contributions in this paper:

- By analyzing and exploring the difference between offline and online KD methods, we empirically show that the reversed distillation hinders the performance gain, which fixes the discrepancy between teacher-student capability. This motivates us to propose a new SHAdow Knowledge distillation (SHAKE) framework to bridge two training fashions, which, to the best of our knowledge, is not achieved in the area of knowledge distillation.  
- SHAKE achieves remarkable trade-offs between accuracy and training efficiency with an extra shadow head. The shadow head inherits knowledge from pre-trained models, introduces reversed distillation with students, and accelerates training process by sharing the student's backbone.  
- We perform thorough evaluations on classification and detection. SHAKE achieves state-of-the-art performances in multiple datasets and architectures (e.g. CNN and vision transformer). Specifically, ResNet-18, MobileNet, and VIT-T with SHAKE achieve  $72.37\%$ ,  $72.96\%$  and  $75.22\%$  Top-1 accuracy on ImageNet, outperforming KD by  $1.71\%$ ,  $2.28\%$  and  $3.02\%$ , respectively.

# 2 Shadow Knowledge Distillation

In this section, we first revisit the offline and online KD methods. Then, we present the formulation of SHAKE and its expansion for multi-teacher models. The evolution of our approach is shown in Figure 2.

# 2.1 Revisit of offline and online KD methods

We first review the formulations of offline and online KD. For simplicity, We choose two typical frameworks (i.e., original KD [14] and DML [39]) for analysis. Given training dataset  $(X,Y)$  where  $X = \{x_{i}\}_{i = 1}^{n}$  are training samples and  $Y = \{y_{i}\}_{i = 1}^{n}$  are their labels. Let  $f_{T}$  be the output logits of the fixed teacher  $T$  and let  $f_{S}$  be the output of student  $S$ , respectively. In KD, the student network  $f_{S}$  is trained by minimizing:

$$
\mathcal {L} _ {S} = \mathcal {L} _ {C E} \left(f _ {S}, Y\right) + \lambda \mathcal {L} _ {K L} \left(f _ {S}, f _ {T}\right), \tag {1}
$$

where  $\lambda$  is a weight for balancing these two terms.  $\mathcal{L}_{CE}$  is the regular cross-entropy objective:

$$
\mathcal {L} _ {C E} \left(f _ {S}, Y\right) = H (Y, \sigma \left(f _ {S}\right)), \tag {2}
$$

where  $H(\cdot, \cdot)$  is the cross-entropy loss and  $\sigma$  is the softmax function.  $\mathcal{L}_{KL}$  in Eq. 1 is the distillation objective for transferring knowledge from a teacher to a student:

$$
\mathcal {L} _ {K L} \left(f _ {S}, f _ {T}\right) = \tau^ {2} K L \left(\sigma \left(\frac {f _ {T}}{\tau}\right), \sigma \left(\frac {f _ {S}}{\tau}\right)\right), \tag {3}
$$

where  $\tau$  is a temperature to generate soft labels and  $KL$  represents Kullback-Leibler (KL) divergence. This distillation loss can be considered as a modified cross-entropy loss using the probabilistic outputs of the pre-trained teacher network as the soft labels instead of the one-hot ground-truth labels.

Different from the one-way distillation in KD, DML presents a two-way knowledge transfer strategy in which the probabilistic outputs from both teacher and student networks can be used to guide the training of each other. DML jointly trains the teacher and student networks in an end-to-end manner via interleavingly optimizing two objectives:

$$
\begin{array}{l} \mathcal {L} _ {T} = \mathcal {L} _ {C E} (f _ {T}, Y) + \lambda \mathcal {L} _ {K L} (f _ {T}, f _ {S}) \\ \mathcal {L} _ {T} = \mathcal {L} _ {C E} (f _ {T}, Y) + \lambda \mathcal {L} _ {K L} (f _ {T}, f _ {S}) \end{array} \tag {4}
$$

where the default value of  $\lambda$  is 1 in DML.

# 2.2 Formulation of SHAKE

As shown in Figure 2 (c), SHAKE proposes a proxy teacher model with output  $f_{T'}$  via shadow head to augment reverse distillation in the offline framework. During the training process,  $f_{T'}$  is updated with the output of the pre-trained teacher model  $f_{T}$  and performs mutual distillation with student models  $f_{S}$  as:

$$
\begin{array}{l} \mathcal {L} _ {T ^ {\prime}} = \mathcal {L} _ {K L} \left(f _ {T ^ {\prime}}, f _ {T}\right) + \lambda \mathcal {L} _ {K L} \left(f _ {T ^ {\prime}}, f _ {S}\right) \\ \mathcal {L} _ {T ^ {\prime}} = \mathcal {L} _ {T ^ {\prime}} \left(f _ {T}, Y\right) + \lambda \mathcal {L} _ {T ^ {\prime}} \left(f _ {T}, f _ {T ^ {\prime}}\right) \end{array} \tag {5}
$$

$$
\mathcal {L} _ {S} = \mathcal {L} _ {C E} \left(f _ {S}, Y\right) + \lambda \mathcal {L} _ {K L} \left(f _ {S}, f _ {T ^ {\prime}}\right),
$$

where each of the loss items has the same form as KD in Eq. 1, and the effects of  $\lambda$  are explored in the experiment. In contrast to DML, the online teacher  $f_{T'}$  is not optimized with ground-truth labels to better maintain knowledge of pre-trained teacher  $f_{T}$  and is not applied in inferences.

For multiple teacher models in Figure 2 (d), we build several proxy teachers with outputs  $(f_{T_1'}, f_{T_2'}, \dots, f_{T_i'})$  to follow outputs  $(f_{T_1}, f_{T_2}, \dots, f_{T_i})$  of multiple teacher models with various shadow heads. The mutual distillation also exists between the task head of the student and these multiple shadow heads. Similar to Eq. 5, the total optimization function for the multi-teacher scenario can be defined as:

$$
\mathcal {L} _ {T ^ {\prime}} = \sum_ {i = 1} ^ {N} \mathcal {L} _ {C E} \left(f _ {T _ {i} ^ {\prime}}, f _ {T i}\right) + \lambda \sum_ {\substack {i = 1 \\ N}} ^ {N} \mathcal {L} _ {K L} \left(f _ {T _ {i} ^ {\prime}}, f _ {S}\right) \tag{6}
$$

$$
\mathcal {L} _ {S} = \mathcal {L} _ {C E} (f _ {S}, Y) + \lambda \sum_ {i = 1} ^ {N} \mathcal {L} _ {K L} (f _ {S}, f _ {T _ {i} ^ {\prime}}).
$$

where  $N$  is the total number of teacher models. In addition, SHAKE can be combined with advanced multi-teacher methods (e.g., adaptive weights for different teachers in part IV of Table 2).

Comparison of SHAKE with other KD methods. Eq. 5, Eq. 4 and Eq. 1 clearly illustrate the difference of SHAKE and KD/DML. Compared to KD, SHAKE introduces reversed distillation so that the teacher can be optimized by the student. In contrast to DML, SHAKE leverages the knowledge of pre-trained models, resulting in additional accuracy gains. As shown in Figure 3, SHAKE achieves a robust boost than KD and baseline during the training process. Meanwhile, training curve of DML shows highly dynamic oscillation due to unreliable predictions of its teacher trained from scratch. Other adaptive KD methods use the same knowledge transfer way as KD and do not optimize teacher with feedback supervision of student. Thus, these methods including ATKD [25] and ESKD [5] do not essentially address the teacher-student capability disparity and our SHAKE surpasses them with significant margins (1.48% ~ 1.59% on ImageNet in Figure 3).

![](images/72614bc2364a74bbecc56857edb0d869337e44ddc08d36cfd2fe8fbcaf1b7935.jpg)  
Figure 3: Comparison of training curves of baseline (69.75%), KD (70.66%), ATKD (70.78%), ESKD (70.89%), DML (71.13%) and our SHAKE (72.37%) for ResNet-18 with single ResNet-34 as teacher on ImageNet.

# 3 Experiments

In this section, we first evaluate our approach for the classification task on CIFAR-100/ImageNet and the object detection task on MS-COCO. Then, comprehensive ablation experiments are performed to analyze the key design in our SHAKE. As a novel logits offline approach, the main competitor of SHAKE is the original KD [14]. Thus, we perform detailed experimental comparisons between them. Moreover, we compare the performance with recent advanced KD methods. For fair comparisons, we use the public codes of these approaches with the same training and data preprocessing settings throughout the experiments. For SHAKE, we set  $\lambda$  and  $\tau$  as 1 and 4, respectively. Please refer to the supplementary materials for more training settings.

# 3.1 Experiments on CIFAR-100

Implementation. On CIFAR-100 [18], we conduct experiments on various teacher-student models under same or different architecture style with CRD's settings [33], whose training epochs are 240.

Table 2: Comparison of results with advanced distillation methods under the same training setting of 240 epochs.  $\uparrow$  refers to the performance gain than baseline. Most results of other methods refer to the CRD. SHAKE† denotes SHAKE with two teachers. R32x4, R8x4, R50, MV2, Sv1 and Sv2 stand for ResNet32x4, ResNet8x4, ResNet50, MobileNetV2, ShuffleNetV1 and ShuffleNetV2. We report Top-1 mean accuracies (\%) over 3 runs.  

<table><tr><td colspan="2"></td><td colspan="4">Same architectural style</td><td colspan="3">Different architectural style</td></tr><tr><td colspan="2">TeacherStudent</td><td>WRN-40-2WRN-16-2</td><td>WRN-40-2WRN-40-1</td><td>R32x4R8x4</td><td>VGG13VGG8</td><td>R50MV2</td><td>R32x4SV1</td><td>R32x4SV2</td></tr><tr><td colspan="2">TeacherStudent</td><td>75.6173.26</td><td>75.6171.98</td><td>79.4272.50</td><td>74.6470.36</td><td>79.3464.60</td><td>79.4270.50</td><td>79.4271.82</td></tr><tr><td rowspan="13">I</td><td>FitNets [31]</td><td>73.58</td><td>72.24</td><td>73.50</td><td>71.02</td><td>63.16</td><td>73.59</td><td>73.54</td></tr><tr><td>SP [35]</td><td>73.83</td><td>72.43</td><td>72.94</td><td>72.68</td><td>68.08</td><td>73.48</td><td>74.56</td></tr><tr><td>RKD [26]</td><td>73.35</td><td>72.22</td><td>71.90</td><td>71.48</td><td>64.43</td><td>72.28</td><td>73.21</td></tr><tr><td>CRD [33]</td><td>75.48</td><td>74.14</td><td>75.51</td><td>73.94</td><td>69.11</td><td>75.11</td><td>75.65</td></tr><tr><td>Review</td><td>76.12</td><td>75.09</td><td>75.63</td><td>74.84</td><td>70.37</td><td>77.14</td><td>77.78</td></tr><tr><td>CL [32]</td><td>74.25</td><td>72.63</td><td>73.10</td><td>71.26</td><td>65.76</td><td>73.62</td><td>73.98</td></tr><tr><td>AFD [6]</td><td>73.70</td><td>72.37</td><td>72.98</td><td>70.88</td><td>64.93</td><td>73.68</td><td>74.32</td></tr><tr><td>ONE [20]</td><td>74.68</td><td>73.43</td><td>73.51</td><td>72.01</td><td>66.26</td><td>74.35</td><td>75.12</td></tr><tr><td>KD [14]</td><td>74.92</td><td>73.54</td><td>73.33</td><td>72.98</td><td>67.35</td><td>74.07</td><td>74.45</td></tr><tr><td>\( KD^† \)</td><td>75.58</td><td>74.24</td><td>74.91</td><td>73.65</td><td>68.81</td><td>75.21</td><td>75.95</td></tr><tr><td>DML [39]</td><td>75.33</td><td>73.98</td><td>74.30</td><td>73.64</td><td>68.52</td><td>75.58</td><td>76.44</td></tr><tr><td>\( DML^† \)</td><td>74.83</td><td>73.26</td><td>73.15</td><td>72.86</td><td>67.22</td><td>74.02</td><td>74.32</td></tr><tr><td>SHAKE</td><td>76.82</td><td>75.62</td><td>77.95</td><td>74.99</td><td>70.18</td><td>77.46</td><td>78.51</td></tr><tr><td rowspan="4">II</td><td>KD+FitNets</td><td>75.12</td><td>73.86</td><td>74.66</td><td>73.22</td><td>66.81</td><td>74.86</td><td>75.15</td></tr><tr><td>KD+CRD</td><td>75.64</td><td>74.38</td><td>75.46</td><td>74.29</td><td>69.54</td><td>75.12</td><td>76.05</td></tr><tr><td>SHAKE+FitNets</td><td>76.91</td><td>75.73</td><td>78.06</td><td>75.15</td><td>70.23</td><td>77.62</td><td>78.69</td></tr><tr><td>SHAKE+CRD</td><td>77.17</td><td>75.89</td><td>78.13</td><td>75.26</td><td>70.42</td><td>77.86</td><td>78.82</td></tr><tr><td rowspan="4">III</td><td>KD+Mixup</td><td>76.58</td><td>76.10</td><td>77.07</td><td>75.58</td><td>71.29</td><td>78.22</td><td>79.14</td></tr><tr><td>KD+CutMix</td><td>76.81</td><td>76.45</td><td>76.90</td><td>75.50</td><td>71.10</td><td>77.92</td><td>79.53</td></tr><tr><td>SHAKE+Mixup</td><td>78.20</td><td>77.36</td><td>79.45</td><td>78.32</td><td>73.88</td><td>79.52</td><td>80.86</td></tr><tr><td>SHAKE+CutMix</td><td>78.45</td><td>77.53</td><td>79.59</td><td>78.56</td><td>74.25</td><td>79.98</td><td>81.22</td></tr><tr><td rowspan="4">IV</td><td>\( KD^‡ (AVER) \)</td><td>75.22</td><td>73.92</td><td>74.99</td><td>74.07</td><td>70.21</td><td>76.30</td><td>75.87</td></tr><tr><td>\( KD^‡ (AEKD) \)</td><td>75.68</td><td>74.24</td><td>75.15</td><td>74.11</td><td>70.47</td><td>76.34</td><td>75.95</td></tr><tr><td>\( SHAKE^‡ (AVER) \)</td><td>77.32</td><td>76.22</td><td>78.59</td><td>75.60</td><td>71.91</td><td>78.61</td><td>78.98</td></tr><tr><td>\( SHAKE^‡ (AEKD) \)</td><td>77.88</td><td>76.69</td><td>78.90</td><td>76.26</td><td>72.38</td><td>78.94</td><td>79.41</td></tr></table>

We use a mini-batch size of 64 and a standard SGD optimizer with a weight decay of 0.0005. The multi-step learning rate is initialized to 0.05, decayed by 0.1 at 150, 180, and 210 epochs. For the comparison experiments with online KD methods, we adopt the same training settings with OKDDip [4], whose training epochs are 300.

Comparison with offline and online KD methods. In the part I of Table 2, we compare our approach to some advanced offline/online KD methods with the same training settings. For the same architecture style of the teacher-students, SHAKE obtains  $3.35\% \sim 5.45\%$  absolute accuracy gains and outperforms KD with  $1.41\% \sim 4.62\%$  margins and Review with  $0.70\% \sim 2.49\%$  margins. Besides, on cross-architecture teacher-student pairs, SHAKE achieves more significant gains with  $5.58\% \sim 6.96\%$  margins than baseline. Compared to other KD methods, SHAKE outperforms KD with  $2.83\% \sim 4.06\%$  margins and Review with  $0.64\% \sim 2.46\%$  margins, which illustrates the effectiveness of SHAKE in reducing the teacher-student network gap. Compared to DML and AFD [6] under the same teacher-student pair in Table 2, SHAKE obtains  $1.21\% \sim 3.65\%$  relative gains. Compare to online KDs (e.g., CL [32], ONE [20]) with multiple branches as teacher models, SHAKE also obtains  $0.70\% \sim 1.77\%$  relative accuracy gains. These significant improvements demonstrate the superiority of SHAKE by leveraging the knowledge of pre-trained models.

Orthogonal to other KDs and data augmentations. As acting on the output logits, SHAKE is orthogonal to feature and relation KD methods because of transferring different knowledge. As shown in the part II of Table 2, The combination of FitNets with SHAKE surpasses FitNets + KD with  $0.98\% \sim 3.54\%$  margins. Moreover, for CRD, its combination with SHAKE yields more dramatic

Table 3: Top-1 accuracies  $(\%)$  on ImageNet dataset. All results of other methods are reproduced under same training settings. SHAKE† denotes SHAKE with two ResNet-34/ResNet-50/RegNetY-16GF as teachers.  

<table><tr><td>Teacher</td><td>Student</td><td>KD [14]</td><td>ESKD [5]</td><td>ATKD [25]</td><td>DMLL [39]</td><td>CRD [33]</td><td>Review</td><td>SHAKE</td><td>\( \mathbf{{SHAKE}}{}^{ \ddagger  } \)</td></tr><tr><td>ResNet-34 (73.40)</td><td>ResNet-18 (69.75)</td><td>70.66</td><td>70.89</td><td>70.78</td><td>71.03</td><td>71.17</td><td>71.61</td><td>72.37</td><td>72.73</td></tr><tr><td>Teacher</td><td>Student</td><td>KD [14]</td><td>AT [38]</td><td>OFD [13]</td><td>DML [39]</td><td>CRD [33]</td><td>Review</td><td>SHAKE</td><td>\( \mathbf{{SHAKE}}{}^{ \ddagger  } \)</td></tr><tr><td>ResNet-50 (76.16)</td><td>MobileNet (70.13)</td><td>70.68</td><td>70.72</td><td>71.25</td><td>71.13</td><td>71.40</td><td>72.56</td><td>72.96</td><td>73.42</td></tr><tr><td>Teacher</td><td>Student</td><td>Tf-KD</td><td>Soft KD</td><td>Hard KD</td><td>DeiT</td><td>-</td><td>-</td><td>SHAKE</td><td>\( \mathbf{{SHAKE}}{}^{ \ddagger  } \)</td></tr><tr><td>RegNetY-16GF (82.90)</td><td>ViT-T (72.20)</td><td>72.35</td><td>72.20</td><td>74.30</td><td>74.50</td><td>-</td><td>-</td><td>75.22</td><td>75.92</td></tr></table>

gains than KD. In addition, SHAKE outperforms KD (see part III of Table 2) under the strong data augmentation.

Extension to multiple teacher models. In the part IV of Table 2, we compare SHAKE and KD with average and adaptive weighting under two teacher networks. The results illustrate that the design of SHAKE with multi-proxy model can effectively inherit the knowledge of different teachers bringing significant performance gains.

# 3.2 Experiments on ImageNet

Implementation. For standard ResNet-18 [12] and MobileNet [15] models, we adopt the same training setting with most distillation methods, whose training epochs are 100. The multi-step learning rate is initialized to 0.1, decayed by 0.1 at 30, 60, and 90 epochs. Recent vision transformers achieve great success on different vision tasks [7, 34]. We also extend SHAKE to VIT-T [34] with the same training settings (e.g., data augmentation and distillation token) on ImageNet. Implementation details are available in supplementary materials.

Comparison results. Table 3 reports the performance of our approach on ImageNet. SHAKE improves baseline models of ResNet-18 by  $2.62\%$  gains in Top-1 accuracy (see Figure 3 for detailed accuracy curves) and MobileNet by  $2.83\%$  gains. Compared to other KD methods, SHAKE outperforms KD with  $1.71\% \sim 2.28\%$  margins and CRD with  $1.20\% \sim 1.56\%$  margins, which supports the superiority of SHAKE on the large-scale dataset. Equipped with the distillation of two teacher models,  $\mathrm{SHAKE}^{\ddagger}$  obtains  $2.98\%$  gain for ResNet-18 and  $3.29\%$  gain for MobileNet than baseline. As shown in Table 3, SHAKE obtains  $2.76\%$  accuracy gains than baseline and surpasses KD under the soft or hard label of the CNN teacher (RegNet [27]), verifying its effectiveness on different architectures.

# 3.3 Experiments on object detection

Implementation. We evaluate SHAKE on MS-COCO dataset [23] and use the most popular open-source Detector2 [22] as the strong baseline. We apply SHAKE to two-stage detector (e.g., Faster R-CNN [29]) and one-stage detector (e.g., RetinaNet [22]), which are widely used object detection frameworks. Following the common practice [22], all models are trained with  $2 \times$  learning schedule (24 epochs). For SHAKE, we build an extra shadow head with the same architecture as the original classification head, which performs distillation in the detector fine-tuning stage. All distillation performances are evaluated in Average Precision (AP).

Comparison results. Table 4 demonstrates the experimental results of the baseline detector and our approach. As shown in Table 4, our SHAKE improves the AP by 1.20 on RetinaNet and 1.02 on Faster R-CNN, which outperform KD [14]. The performance of the object detection greatly depends on the quality of deep features to locate interested objects, while logits are not capable of providing knowledge for object localization. Thus, SHAKE naturally weaker than Review (recent state-of-the-art feature-based KDs) and we further introduce Review to obtain satisfactory results. It can be observed that SHAKE can obtain new state-of-the-art results with feature-based KDs. The success of challenging object detection tasks demonstrates the generality and effectiveness of our approach.

Table 4: Comparison of results of object detection on MS-COCO.  

<table><tr><td>Model</td><td>AP</td><td>APL</td><td>APM</td><td>APS</td><td>Model</td><td>AP</td><td>APL</td><td>APM</td><td>APS</td></tr><tr><td>Faster R-CNN-R101 [T]</td><td>42.04</td><td>54.60</td><td>45.55</td><td>25.22</td><td>RetinaNet-R101 [T]</td><td>40.40</td><td>52.18</td><td>44.34</td><td>24.03</td></tr><tr><td>Faster R-CNN-R50 [S]</td><td>37.93</td><td>49.10</td><td>41.14</td><td>22.44</td><td>RetinaNet-R50 [S]</td><td>36.15</td><td>46.95</td><td>40.25</td><td>21.37</td></tr><tr><td>+ KD (logits)</td><td>38.35</td><td>49.48</td><td>41.80</td><td>22.73</td><td>+ KD (logits)</td><td>36.76</td><td>48.17</td><td>40.56</td><td>21.87</td></tr><tr><td>+ Review (feature)</td><td>40.36</td><td>52.87</td><td>43.81</td><td>23.60</td><td>+ Review (feature)</td><td>38.48</td><td>51.15</td><td>42.72</td><td>22.67</td></tr><tr><td>+ SHAKE (logits)</td><td>38.95</td><td>50.78</td><td>42.32</td><td>22.88</td><td>+ SHAKE (logits)</td><td>37.35</td><td>50.35</td><td>40.88</td><td>21.61</td></tr><tr><td>+ SHAKE &amp; Review</td><td>40.67</td><td>52.96</td><td>43.92</td><td>23.85</td><td>+ SHAKE &amp; Review</td><td>40.55</td><td>52.80</td><td>43.78</td><td>23.55</td></tr></table>

# 3.4 Ablation study

In this section, we isolate the impact of each component of our approach. All experiments are conducted on the CIFAR-100 dataset. For each setting, we run 3 times and report Top-1 mean accuracies.

Detailed ablation study of SHAKE As shown in Table 5, an ablation study has been conducted to demonstrate the individual effectiveness of different components in SHAKE. It is observed that (a) Reversed distillation optimizes the teacher model to adapt to the student, improving distillation efficiency. SHAKE without reversed distillation only obtains marginal gains. (b) Shadow heads are effective in preserving the diversity of knowledge, and SHAKE brings accuracy reduction if there is no separate shadow head. (c) Stronger shadow heads for storing and teaching more pretrained knowledge when the capacity gap enlarges. (d) Backbone sharing settings can accelerate the convergence of the student model and results in performance gains. (e) Similar to the self-distillation and regularization methods, SHAKE without pre-trained teacher model also yields slight accuracy gains because of more training weights and logit consistency. (f) Our proxy teacher is present to inherit the pre-training knowledge with its supervision. Under our backbone sharing settings, adding extra supervision from ground truth labels for proxy teacher not conducive to the decoupling of supervision from pre-trained teachers/labels and leads to accuracy reduction. (h) Proxy teachers in our SHAKE to transfer the knowledge of the original pre-trained teacher model for the student model to learn more easily. As shown in the Table 5 (h), the extra distillation from pre-trained teachers to students based on SHAKE does not yield additional significant performance gains.

Table 5: Top-1 accuracies  $(\%)$  of student (S), proxy teachers (T'), and their average ensemble with different settings on CIFAR-100. Note that the ensemble with bring extra latency at inference, while SHAKE only adopts student model for inference without extra costs. We report top-1 "mean" accuracies  $(\%)$  over 3 runs.  

<table><tr><td rowspan="2">Method</td><td colspan="3">WRN-16-2 (73.26)</td><td colspan="3">VGG8 (70.36)</td></tr><tr><td>S</td><td>T&#x27;</td><td>Ensemble</td><td>S</td><td>T&#x27;</td><td>Ensemble</td></tr><tr><td>SHAKE</td><td>76.82</td><td>76.78</td><td>76.91</td><td>74.99</td><td>74.92</td><td>75.06</td></tr><tr><td>(a) w/o Reversed distillation</td><td>74.98</td><td>74.16</td><td>75.01</td><td>73.01</td><td>71.86</td><td>73.02</td></tr><tr><td>(b) w/o Shadow head</td><td>75.08</td><td>75.08</td><td>75.08</td><td>73.12</td><td>73.12</td><td>73.12</td></tr><tr><td>(c) w Stronger shadow head</td><td>76.96</td><td>76.92</td><td>75.01</td><td>75.12</td><td>75.08</td><td>75.18</td></tr><tr><td>(d) w/o Share backbone</td><td>76.36</td><td>76.22</td><td>76.45</td><td>74.35</td><td>74.28</td><td>74.41</td></tr><tr><td>(e) w/o Pre-trained teacher</td><td>74.22</td><td>73.86</td><td>74.23</td><td>71.37</td><td>71.14</td><td>71.41</td></tr><tr><td>(f) w Ground truth supervision for T&#x27;</td><td>76.42</td><td>76.56</td><td>76.66</td><td>74.48</td><td>74.54</td><td>74.61</td></tr><tr><td>(h) w Pre-teacher distillation for S</td><td>76.76</td><td>76.61</td><td>76.86</td><td>74.86</td><td>74.71</td><td>74.95</td></tr></table>

Sensitivity study for temperature  $\tau$  and weight  $\lambda$ .  $\tau$  in Eq. 3 controls the softness of teacher's supervision. As  $\tau$  gets higher, the output of the softmax function becomes smoother. As shown in Figure 4, SHAKE presents superior gains and robustness than the original KD on different  $\tau$ .  $\lambda$  is the hyper-parameter that balances the weight of KD loss. The small  $\lambda$  limits distillation gains, and  $\lambda = 0.5$  is the best option for SHAKE.

Analyzing teacher-student similarity. The similarity between student and teacher network is an important measure for KD tasks. We employ KL-divergence as metrics of similarity [5], where lower KL-divergence implies higher similarity. Figure 5 presents the similarities and performances between the outputs of ResNet with different depths as teachers (i.e., ResNet-110, ResNet-56, ResNet-44 and ResNet-32) and the ResNet-20 as student. The results show that the distillation from SHAKE always gives higher similarity than KD and DML, resulting in significant gains under teacher models

![](images/2a8d552d6b26245f574e6006b3bd154cd855592d01f4041c8bf8fd577d609fff.jpg)

![](images/c0942475a0187965d585c44b42af23b09d080044cdf75237c53a6a3683b07f81.jpg)

![](images/26cff3e4b73e75629e9422b2679c3a1908f9484815a517ce554209c572e9d44e.jpg)

![](images/3be70e5b0166b1a5ddf1c1b29ea7122e4426bbdd5e5136a4c5e0d094f8ba88f5.jpg)

![](images/11a058e8757fc2dde7d4ce1863a7c9bedb9eab438357bca08abbe30e190e0ef4.jpg)  
Figure 5: KL-divergence and Top-1 accuracy  $(\%)$ . Figure 6: Grad-CAM++ ([3]) visualization.

![](images/78005c4594c35310b0cf12ea7b9834cd124c63d92f09edb31168dde7712dd505.jpg)  
Figure 4: Top-1 mean (std) accuracies  $(\%)$  of varying  $\tau$  for WRN-16-2 & VGG-8 (left) and varying  $\lambda$  for WRN-16-2 & VGG-8 (right) on CIFAR-100.

of different depths. Figure 6 presents the attention map visualization of ResNet-18 with different methods on ImageNet. The attention maps of SHAKE pay more attention to the important regions than KD and baseline, which have high similarity with teacher. In summary, SHAKE bridges KD and DML to obtain student-friendly distillations, improving teacher-student similarity.

# 4 Related Work

Offline distillations. In offline framework [14], the teacher is pre-trained and fixed, and then its soft logits are used as the extra supervision to distill student. Although the subsequent methods explored to transfer feature knowledges [31, 38] and relation knowledges [26, 33]), the effective original KD still outperforms most distillations and is widely used for different tasks [9, 17]. However, pre-trained teacher model is not optimized for the student, limiting its distillation gains. To address this issue, our SHAKE enhances KD using a shadow head to introduce optimization from the student.

Online distillations. Online KD methods simplify the KD process by training all models simultaneously. DML [39] performs bidirectional distillation for the peer networks and ONE [20] presents an on-the-fly ensemble distillation among multiple branches. Subsequent studies focus on how to balance multiple teacher [4] or construct the online teacher [37]. Online KD methods always obtain better performance than offline ones. However, large teacher models trained from scratch sometimes perform unstable predictions and bring lots of training costs during distillation. Therefore, we propose SHAKE to combine the advantages of the two pipelines to facilitate application.

Adaptive distillations. Capacity gaps between teacher-student models for their different architectures would limit distillation gains [24]. There are two types of existing works to alleviate this gap in terms of training paradigms [8] and architectural adaptation [16, 11]. For instance, ESKD [5] proposes stopping the training of the teacher early, and ATKD [25] uses a medium-size teacher assistant to perform sequence distillation with large overheads. However, these methods do not optimize the teacher model for the student, resulting in minor benefits. In sharp contrast to these methods, SHAKE is a student-aware offline KD method and opens a new direction for adaptive distillation design.

# 5 Conclusion

In this paper, we present SHAKE, a simple, effective, and new student-aware logits distillation to bridge offline and online knowledge transfer. Based on our insight of online KD methods success, SHAKE achieves this goal by building an extra shadow head as a proxy teacher model to perform mutual distillation with the student model. Thorough evaluations are performed on classification and detection, and SHAKE achieves significant performance gains in various neural networks without extra inference overheads. We hope that this elegant and practical approach would inspire future research on knowledge distillation design and understanding.

# References

[1] Lei Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In NeurIPS, 2014.  
[2] Tom B. Brown, Benjamin Mann, Nick Ryder and Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M, Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv preprint, arXiv:2005.14165, 2020.  
[3] Aditya Chattopadhy, Anirban Sarkar, Prantik Howlader, and Vineeth N Balasubramanian. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. In WACV, 2018.  
[4] Defang Chen, Jian-Ping Mei, Can Wang, Yan Feng, and Chun Chen. Online knowledge distillation with diverse peers. In AAAI, 2020.  
[5] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In ICCV, 2019.  
[6] Inseop Chung, Seonguk Park, J. Kim, and Nojun Kwak. Feature-map-level online adversarial knowledge distillation. In ICML, 2020.  
[7] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
[8] Mengya Gao, Yujun Shen, Quanquan Li, Junjie Yan, Liang Wan, Dahua Lin, Chen Change Loy, and Xiaou Tang. An embarrassingly simple approach for knowledge distillation. arXiv preprint arXiv:1812.01819, 2018.  
[9] Nuno C Garcia, Pietro Morerio, and Vittorio Murino. Modality distillation with multiple stream networks for action recognition. In ECCV, 2018.  
[10] Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014.  
[11] Jindong Gu and Volker Tresp. Search for better students to learn distilled knowledge. arXiv preprint arXiv:2001.11612, 2020.  
[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
[13] Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi. A comprehensive overhaul of feature distillation. In ICCV, 2019.  
[14] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
[15] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint, arXiv:1704.04861, 2017.  
[16] Minsoo Kang, Jonghwan Mun, and Bohyung Han. Towards oracle knowledge distillation with neural architecture search. In AAAI, 2020.  
[17] Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In EMNLP, 2016.  
[18] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Tech Report, 2009.

[19] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012.  
[20] Xu Lan, Xiatian Zhu, and Shaogang Gong. Knowledge distillation by on-the-fly native ensemble. In NeurIPS, 2018.  
[21] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In ICLR, 2017.  
[22] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar. Focal loss for dense object detection. In ICCV, 2017.  
[23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.  
[24] Yu Liu, Xuhui Jia, Mingxing Tan, Raviteja Vemulapalli, Yukun Zhu, Bradley Green, and Xiaogang Wang. Search to distill: Pearls are everywhere but not the eyes. In CVPR, 2020.  
[25] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In AAAI, 2020.  
[26] Wonpyo Park, Yan Lu, Minsu Cho, and Dongju Kim. Relational knowledge distillation. In CVPR, 2019.  
[27] Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollar. Designing network design spaces. In CVPR, 2020.  
[28] Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Joseph. Xnor-net: Imagenet classification using binary convolutional neural networks. In ECCV, 2016.  
[29] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. arXiv preprint, arXiv:1506.01497, 2015.  
[30] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, Andre Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. arXiv preprint arXiv:2106.05974, 2021.  
[31] Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015.  
[32] Guocong Song and Wei Chai. Collaborative learning for deep neural networks. In NeurIPS, 2018.  
[33] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. In ICLR, 2020.  
[34] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers amp; distillation through attention. In ICML, 2021.  
[35] Frederick Tung and Greg Mori. Similarity-preserving knowledge distillation. In ICCV, 2019.  
[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.  
[37] Guile Wu and Shaogang Gong. Peer collaborative learning for online knowledge distillation. arXiv preprint arXiv:2006.04147, 2020.  
[38] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In ICLR, 2017.  
[39] Ying Zhang, Tao Xiang, Timothy M Hospedales, and Huchuan Lu. Deep mutual learning. In CVPR, 2018.
