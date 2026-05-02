# DBA: DISTRIBUTED BACKDOOR ATTACKS AGAINST FEDERATED LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Backdoor attacks aim to manipulate a subset of training data by injecting adversarial triggers such that machine learning models trained on the tampered dataset will make arbitrarily (targeted) incorrect prediction on the testset with the same trigger embedded. While federated learning (FL) is capable of aggregating information provided by different parties for training a better model, its distributed learning methodology and inherently heterogeneous data distribution across parties may bring new vulnerabilities. In addition to recent centralized backdoor attacks on FL where each party embeds the same global trigger during training, we propose the distributed backdoor attack (DBA) — a novel threat assessment framework developed by fully exploiting the distributed nature of FL. DBA decomposes a global trigger pattern into separate local patterns and embed them into the training set of different adversarial parties respectively. Compared to standard centralized backdoors, we show that DBA is substantially more persistent and stealthy against FL on diverse datasets such as finance and image data. We conduct extensive experiments to show that the attack success rate of DBA is significantly higher than centralized backdoors under different settings. Moreover, we find that distributed attacks are indeed more insidious, as DBA can evade two state-of-the-art robust FL algorithms against centralized backdoors. We also provide explanations for the effectiveness of DBA via feature visual interpretation and feature importance ranking. To further explore the properties of DBA, we test the attack performance by varying different trigger factors, including local trigger variations (size, gap, and location), scaling factor in FL, data distribution, and poison ratio and interval Our proposed DBA and thorough evaluation results shed lights on characterizing the robustness of FL.

# 1 INTRODUCTION

Federated learning (FL) has been recently proposed to address the problems for training machine learning models without direct access to diverse training data, especially for privacy-sensitive tasks (Smith et al., 2017; McMahan et al., 2017; Zhao et al., 2018). Utilizing local training data of participants (i.e., parties), FL helps train a shared global model with improved performance. There have been prominent applications and ever-growing trends in deploying FL in practice, such as loan status prediction, health situation assessment (e.g. potential cancer risk assessment), and next-word prediction while typing (Hard et al., 2018; Yang et al., 2018; 2019).

Although FL is capable of aggregating dispersed (and often restricted) information provided by different parties to train a better model, its distributed learning methodology as well as inherently heterogeneous (i.e., non-i.i.d.) data distribution across different parties may unintentionally provide a venue to new attacks. In particular, the fact of limiting access to individual party's data due to privacy concerns or regulation constraints may facilitate backdoor attacks on the shared model trained with FL. Backdoor attack is a type of data poisoning attacks that aim to manipulate a subset of training data such that machine learning models trained on the tampered dataset will be vulnerable to the test set with similar trigger embedded (Gu et al., 2019).

Backdoor attacks on FL have been recently studied in (Bagdasaryan et al., 2018; Bhagoji et al., 2019). However, current attacks do not fully exploit the distributed learning methodology of FL, as they embed the same global trigger pattern to all adversarial parties. We call such attacking scheme

![](images/bd2aa56d31926be60aa5e112a13313838e781f801ef6654973b6dd05cf08b91e.jpg)  
(a) centralized backdoor attack (current setting)  
Figure 1: Overview of centralized and distributed backdoor attacks (DBA) on FL. The aggregator at round  $t + 1$  combines information from local parties (benign and adversarial) in the previous round  $t$ , and update the shared model  $G^{t + 1}$ . When implementing backdoor attacks, centralized attacker uses a global trigger while distributed attacker uses a local trigger which is part of the global one.

![](images/18537e7c8d80329b08fbfd98b44080255e3e4d5ccabbbc75325e58bf26c231c0.jpg)  
(b) DBA: distributed backdoor attack (ours)

centralized backdoor attack. Leveraging the power of FL in aggregating dispersed information from local parties to train a shared model, in this paper we propose distributed backdoor attack (DBA) against FL. Given the same global trigger pattern as the centralized attack, DBA decomposes it into local patterns and embed them to different adversarial parties respectively. A schematic comparison between the centralized and distributed backdoor attacks is illustrated in Fig.1.

Through extensive experiments on several financial and image datasets and in-depth analysis, we summarize our main contributions and findings as follows.

- We propose a novel distributed backdoor attack strategy DBA on FL and show that DBA is more persistent and effective than centralized backdoor attack. Based on extensive experiments, we report a prominent phenomenon that although each adversarial party is only implanted with a local trigger pattern via DBA, their assembled pattern (i.e., global trigger) attains significantly better attack performance on the global model compared with the centralized attack. The results are consistent across datasets and under different attacking scenarios such as one-time (single-shot) and continuous (multiple-shot) poisoning settings. To the best of our knowledge, this paper is the first work studying distributed backdoor attacks.  
- When evaluating the robustness of two recent robust FL methods against centralized backdoor attack (Fung et al., 2018; Pillutla et al., 2019), we find that DBA is more effective and stealthy, as its local trigger pattern is more insidious and hence easier to bypass the robust aggregation rules.  
- We provide in-depth explanations for the effectiveness of DBA from different perspectives, including feature visual interpretation and feature importance ranking.  
- We perform comprehensive analysis and ablation studies on several trigger factors in DBA, including the size, gap, and location of local triggers, scaling effect in FL, poisoning interval, data poisoning ratio, and data distribution.

# 2 DISTRIBUTED BACKDOOR ATTACK AGAINST FEDERATED LEARNING

# 2.1 GENERAL FRAMEWORK

The training objective of FL can be cast as a finite-sum optimization:  $\min_{w\in R^d}[F(w)\coloneqq \frac{1}{N}\sum_{i = 1}^{N}f_i(w)]$ . There are  $N$  parties individually processing  $N$  local models, each of whom trains with the local objective  $f_{i}:R^{d}\mapsto R$  based on a private dataset  $D_{i} = \{\{x_{j}^{i},y_{j}^{i}\}_{j = 1}^{a_{i}}\}$ , where  $a_{i} = |D_{i}|$  and  $\{x_j^i,y_j^i\}$  represents each data sample and its corresponding label. In supervised FL setting, each local function  $f_{i}$  is computed as  $f_{i}(w_{i}) = l(\{x_{j}^{i},y_{j}^{i}\}_{j\in D_{i}},w_{i})$  where  $l$  stands for a loss of prediction using the local parameters  $w_{i}$ . The goal of FL is to obtain a global model which can generalize well on test data  $D_{test}$  after aggregating over the distributed training results from  $N$  parties.

Specifically, at round  $t$ , the central server sends the current shared model  $G^{t}$  to  $n \in [N]$  selected parties, where  $[N]$  denotes the integer set  $\{1, 2, \dots, N\}$ . The selected party  $i$  locally computes the function  $f_{i}$  by running an optimization algorithm such as stochastic gradient descent (SGD) for  $E$

local epochs with its own dataset  $D_{i}$  and learning rate  $l_{r}$  to obtain a new local model  $L_{i}^{t + 1}$ . The local party then sends model update  $L_{i}^{t + 1} - G^{t}$  back to the central server, who will averages over all updates with its own learning rate  $\eta$  to generate a new global model  $G^{t + 1}$ :

$$
G ^ {t + 1} = G ^ {t} + \frac {\eta}{n} \sum_ {i = 1} ^ {n} \left(L _ {i} ^ {t + 1} - G ^ {t}\right) \tag {1}
$$

This aggregation process will be iterated until FL finds the final global model. Unless specified otherwise, we use  $G^{t}(L_{i}^{t})$  to denote the model parameters of the global (local) model at round  $t$ .

Attacker ability. Based on the Kerckhoffs's theory (Shannon, 1949), we consider the strong attacker here who has full control of their local training process, such as backdoor data injection and updating local training hyperparameters including  $E$  and  $l_r$ . This scenario is quite practical since each local dataset is usually owned by one of the local parties. However, attackers do not have the ability to influence the privilege of central server such as changing aggregation rules, nor tampering the training process and model updates of other parties.

Objective of backdoor attack. Backdoor attack is designed to mislead the trained model to predict a target label  $\tau$  on any input data that has an attacker-chosen pattern (i.e., a trigger) embedded. Instead of preventing the convergence in accuracy as Byzantine attacks (Blanchard et al., 2017), the purpose of backdoor attacks in FL is to manipulate local models and simultaneously fit the main task and backdoor task, so that the global model would behave normally on untampered data samples while achieving high attack success rate on backdoored data samples. The adversarial objective for attacker  $i$  in round  $t$  with local dataset  $D_{i}$  and target label  $\tau$  is:

$$
w _ {i} ^ {*} = \arg \max  _ {w _ {i}} \left(\sum_ {j \in S _ {p o i} ^ {i}} P \left[ G ^ {t + 1} \left(R \left(x _ {j} ^ {i}, \phi\right)\right) = \tau \right] + \sum_ {j \in S _ {c l n} ^ {i}} P \left[ G ^ {t + 1} \left(x _ {j} ^ {i}\right) = y _ {j} ^ {i} \right]\right). \tag {2}
$$

Here, the poisoned dataset  $S_{pol}^{i}$  and clean dataset  $S_{cln}^{i}$  satisfy  $S_{pol}^{i} \cap S_{cln}^{i} = \emptyset$  and  $S_{pol}^{i} \cup S_{cln}^{i} = D_{i}$ . The function  $R$  transforms clean data in any class into backdoored data that have an attacker-chosen trigger pattern using a set of parameters  $\phi$ . For example, for image data,  $\phi$  is factored into trigger location  $TL$ , trigger size  $TS$  and trigger gap  $TG$  ( $\phi = \{TS, TG, TL\}$ ), which are shown in Fig.2. The attacker can design his own trigger pattern and choose an optimal poison ratio  $r$  to result in a better model parameter  $w_{i}^{*}$ , with which  $G^{t+1}$  can both assign the highest probability to target label  $\tau$  for backdoored data  $R(x_{j}^{i}, \phi)$  and the ground truth label  $y_{j'}^{i}$  for benign data  $x_{j'}^{i}$ .

# 2.2 DISTRIBUTED BACKDOOR ATTACK (DBA)

We again use Fig.1 to illustrate our proposed DBA in details. Recall that current centralized attack embeds the same global trigger for all local attackers<sup>1</sup> (Bagdasaryan et al., 2018). For example, the attacker in Fig.1.(a) embeds the training data with the selected patterns highlighted by 4 colors, which altogether constitutes a complete global pattern as the backdoor trigger.

In our DBA, as illustrated in Fig.1.(b), all attackers only use parts of the global trigger to poison their local models, while the ultimate adversarial goal is still the same as centralized attack — using the global trigger to attack the shared model. For example, the attacker with the orange sign poisons a subset of his training data only using the trigger pattern located at the orange area. Similar attacking methodology applies to green, yellow and blue signs. We define each DBA attacker's trigger as the local trigger and the combined whole trigger as the global trigger. For fair comparison, we keep similar amount of total injected triggers (e.g., modified pixels) for both centralized attack and DBA.

In centralized attack, the attacker tries to solve the optimization problem in Eq.2 without any coordination and distributed processing. In contrast, DBA fully exploits the distributed learning and local data opacity in FL. Considering  $M$  attackers in DBA with  $M$  small local triggers. Each DBA attacker  $m_{i}$  independently performs the backdoor attack on their local models. This novel mechanism breaks a centralized attack formulation into  $M$  distributed sub-attack problems aiming to solve

$$
w _ {i} ^ {*} = \arg \max  _ {w _ {i}} \left(\sum_ {j \in S _ {p o i} ^ {i}} P \left[ G ^ {t + 1} \left(R \left(x _ {j} ^ {i}, \phi_ {i} ^ {*}\right)\right) = \tau ; \gamma ; I \right] + \sum_ {j \in S _ {c l n} ^ {i}} P \left[ G ^ {t + 1} \left(x _ {j} ^ {i}\right) = y _ {j} ^ {i} \right]\right), \forall i \in [ M ] \tag {3}
$$

![](images/28fdbebb9c0798042886894e7ea8d1f92c90ccd5dd1abbecd74f3e87ef57359a.jpg)  
(a) Trigger Size

![](images/cb8458a804d3445b6069cc93b961a263044cdaebe5fe2db47776daacd74b3a35.jpg)  
(b) Trigger Gap

![](images/e26a9bc1b95f856f59142614eaffba7bae2fe7f66ae074fbd146c6d6edc14b34.jpg)  
(c) Trigger Location

![](images/dc566493bfefcb0ee3043a443a17ad0cd39c1e4e69439b20023cfe5cf325883f.jpg)  
Figure 2: Trigger factors (size, gap and location) in back-doored images.  
Figure 3: Trigger factor (feature importance ranking) in tabular data.

where  $\phi_i^* = \{\phi, O(i)\}$  is the geometric decomposing strategy for the local trigger pattern of attacker  $m_i$  and  $O(i)$  entails the trigger decomposition rule for  $m_i$  based on the global trigger  $\phi$ . DBA attackers will poison with the poison round interval  $I$  and use the scale factor  $\gamma$  to manipulate their updates before submitting to the aggregator. We will explain the related trigger factors in the next subsection. We note that although none of the adversarial parties has ever been poisoned by the global trigger under DBA, we find that DBA indeed outperforms centralized attack significantly when evaluated with the global trigger.

# 2.3 FACTORS IN DISTRIBUTED BACKDOOR ATTACK

With the framework of DBA on FL, there are multiple new factors to be explored. Here we introduce a set of trigger factors that we find to be critical. Fig.2 explains the location, size and gap attribute of triggers in image dataset. For simplicity, we set all of our local triggers to the same rectangle shape<sup>2</sup>. Fig.3 explains our trigger attribute of ranked feature importance in tabular data (e.g., the loan dataset).

Trigger Size TS: the number of pixel columns (i.e., the width) of a local distributed trigger.

Trigger Gap TG: the distance of the  $Gap_x$  and  $Gap_y$ , which represent the distance between the left and right, as well as the top and bottom local trigger, respectively.

Trigger Location  $TL$ :  $(Shift_x, Shift_y)$  is the offset of the trigger pattern from the top left pixel.

Scale  $\gamma$ : the scaling parameter  $\gamma = \eta / N$  defined in (Bagdasaryan et al., 2018) is used by the attacker to scale up the malicious model weights. For instance, assume the  $i$ th malicious local model is  $X$ . The new local model  $L_{i}^{t+1}$  that will be submitted is calculated as  $L_{i}^{t+1} = \gamma(X - G^{t}) + G^{t}$ .

**Poison Ratio  $r$ : the ratio controls the fraction of backdoored samples added per training batch. Note that larger  $r$  should be preferable when attacking intuitively, and there is a tradeoff between clean data accuracy and attack success rate, but too large  $r$  would also hurt the attack effectiveness once the model becomes useless.**

**Poison Interval  $I$ : the round intervals between two poison steps.** For example,  $I = 0$  means all the local triggers are embedded within one round, while  $I = 1$  means the local triggers are embedded in consecutive rounds.

Data Distribution: FL often presumes non-i.i.d. data distribution across parties. Here, we use a Dirichlet distribution (Minka, 2000) with different hyperparameter  $\alpha$  to generate different data distribution following the setups in (Bagdasaryan et al., 2018).

# 3 EXPERIMENTS

# 3.1 DATASETS AND EXPERIMENT SETUP

DBA is evaluated on four classification datasets with non-i.i.d. data distributions: Lending Club Loan Data(LOAN)(Kan, 2019), MNIST, CIFAR-10 and Tiny-imagenet. The data description and parameter setups are summarized in Tb.1. We refer the readers to Appendix A.1 for more details.

Following the standard setup, we use SGD and trains for  $E$  local epochs with local learning rate  $l_{r}$  and batch size 64. A shared global model is trained by all participants, 10 of them are selected in each round for aggregation. The local and global triggers used are summarized in Appendix A.1.

# 3.2 DISTRIBUTED BACKDOOR ATTACK V.S. CENTRALIZED BACKDOOR ATTACK

Following the attack analysis in (Bagdasaryan et al., 2018), we evaluate multiple-shot attack (Attack A-M) and single-shot attack (Attack A-S) two attack scenarios, which are called naive approach and model replacement respectively in the original paper.

Table 1: Dataset description and parameters  

<table><tr><td>Dataset</td><td>Classes</td><td>Examples per class</td><td>Features</td><td>Model used</td><td>Benign lr/E</td><td>Poison lr/E</td></tr><tr><td>LOAN</td><td>9</td><td>see Tb.4 in Appendix</td><td>91</td><td>3 fc</td><td>0.001 / 1</td><td>0.0005 / 5(multi-shot) or 10(single-shot)</td></tr><tr><td>MNIST</td><td>10</td><td>6000</td><td>784</td><td>2 conv and 2 fc</td><td>0.1 / 1</td><td>0.05 / 10</td></tr><tr><td>CIFAR</td><td>10</td><td>5000</td><td>1024</td><td>lightweight Resnet-18</td><td>0.1 / 2</td><td>0.05 / 6</td></tr><tr><td>Tiny-imagenet</td><td>200</td><td>500</td><td>4096</td><td>Resnet-18(He et al., 2016)</td><td>0.001 / 2</td><td>0.001 / 5(multi-shot) or 10(single-shot)</td></tr></table>

- Attack A-M means the attackers are selected in multiple rounds and the accumulated malicious updates are necessary for a successful attack; otherwise the backdoor would be weakened by benign updates and soon forgotten by the global model. In order to quickly observe the difference between centralized and distributed attacks and control the effect of random party selection, we perform a complete attack in every round, that is, all DBA attackers or centralized attackers are consistently selected. Benign participants are randomly selected to form a total of 10 participants.  
- Attack A-S means that every DBA attacker or the centralized attacker only needs one single shot to successfully embed its backdoor trigger. To achieve that, the attacker performs scaling in their malicious updates to overpower other benign updates and ensure that the backdoor survives the aggregation step. For fair comparison, DBA and centralized attack finish a complete backdoor in the same round. Take MNIST as an example, DBA attackers separately embed their local triggers in round 12, 14, 16, 18 for local triggers 1 to 4, while the centralized attacker implants its global trigger in round 18. Benign participants are randomly selected to form a total of 10 participants.

These two scenarios reveal different aspects of DBA and centralized backdoor attacks when the global model is triggered by local and global triggers. Attack A-M studies how easy the backdoor is successfully injected while Attack A-S studies how fast the backdoor effect diminishes.

In our experiments, we evaluate the attack success rates of DBA and centralized attacks using the same global trigger. For fair comparison, we make sure the total number of backdoor pixels of DBA attackers is close to and even less than that of the centralized attacker (it is hard to control them to be the same due to data sampling with certain distribution). The ratio of the global trigger of DBA pixels to the centralized is 0.992 for LOAN, 0.964 for MNIST, 0.990 for CIFAR and 0.991 for Tiny-imagenet. Moreover, in order to avoid the influence of the original label when testing attack success rate, we remove the test data whose true label equals to the backdoor target label. In three image datasets, we begin to attack when the main accuracy of global model converges, which is round 10 for MNIST, 200 for CIFAR, 20 for Tiny-imagenet in Attack A-M. The reason is provided in Appendix.A.2. The global learning rate  $\eta$  in Attack A-M is 0.1 for CIFAR, 1 for others and in Attack A-S is 0.1 for all datasets.

![](images/d83e353428f4913be397af2a8a2714f4c9dbca62a0a9a5172efcafd6d5f18d78.jpg)  
Figure 4: Attack A-M and A-S. DBA is more effective and persistent than centralized attack.

In Attack A-M, the attack success rate of DBA is always higher than centralized attack in all cases as shown in Fig.4. DBA also converges faster and even yields a higher attack success rate in MNIST. Under DBA, we find a prominent phenomenon that the attack success rate of the global trigger is higher than any local trigger even if the global trigger never actually appears in any local training dataset. Moreover, the global trigger converges faster in attack performance than local triggers. Centralized attacker embeds the whole pattern so its attack success rate of any local triggers is low. Due to the continuous poisoning, the attack rate on local triggers still increases for LOAN but this phenomenon does not appear in MNIST and Tiny-imagenet, which indicates that the success of global trigger does not require the same success for local triggers. The results also suggest that DBA

can lead to high attack success rate for the global trigger even when some of its local triggers only attain low attack success rates. This finding is unique for DBA and also implies the inefficiency of centralized attack on FL.

In Attack A-S, DBA and centralized attack both reach a high attack success rate after performing a complete backdoor in all datasets with a scale factor  $\gamma = 100$  as shown in Fig.4. In the consecutive rounds, the backdoor injected into the global model is weakened by benign updates so the attack success rate gradually decreases. There is an exception that centralized attack in CIFAR suffers from the initial drop and then rises slowly, which is caused by the high local learning rate of benign participants and is also observed in (Bagdasaryan et al., 2018). We also find that the attack success rate of centralized attack in local triggers and the global trigger drops faster than that of DBA, which shows that DBA yields a more persistent attack. For example, in MNIST and after 50 rounds, DBA remains  $89\%$  attack success rate while centralized attack only gets  $21\%$ . Although DBA performs data poisoning only using local triggers, the results show that its global trigger lasts longer than any local triggers, which suggests DBA can make the global trigger more resilient to benign updates.

# 3.3 THE ROBUSTNESS OF DISTRIBUTED ATTACK

RFA (Pillutla et al., 2019) and FoolsGold (Fung et al., 2018) are two recently proposed robust FL aggregation algorithms based on distance or similarity metrics, and in particular RFA is claimed to be able to detect more nuanced outliers which goes beyond the worst-case of the Byaantine setting (Blanchard et al., 2017). In addition, as Attack A-S is more easily detected due to the scaling operation (Pillutla et al., 2019), we will focus on evaluating the attack effectiveness of DBA and centralized backdoor attacks against both RFA and FoolsGold under Attack A-M setting.

Distributed Attack against Robust Aggregation Defence. RFA aggregates model parameters for updates and appears robust to outliers by replacing the weighted arithmetic mean in the aggregation step with an approximate geometric median. With only a few attackers poisoning a small part in every batch, our DBA meets the condition that the total weight of the outliers is strictly less than  $\frac{1}{2}$  for iterations of RFA so that it can converge to a solution despite the outliers. The maximum iteration of RFA is set to be 10 while in fact it converges rapidly, which can give a high-quality solution within about 4 iterations. Fig.5 shows the attack performance of DBA and centralized attack under RFA. For Tiny-imagenet, the centralized attack totally fails at least 80 rounds but the DBA attackers with lower distances and higher aggregation weights can perform a successful backdoor attack. For MNIST and CIFAR, the attack success rate of DBA is much higher and the convergence speed is much faster. For LOAN, centralized backdoor attack takes more than 20 rounds to converge than DBA. To explain the effectiveness of DBA, we calculate the Euclidean norm between attacker's model parameter updates and the final geometric median as a distance metric. As shown in Tb.2 in Appendix, the malicious updates submitted by DBA attackers have lower distances than that of the centralized attacker's updates in all datasets, which help them to better bypass the defense.

![](images/de26335f7ee29cf69438d00e1c46b84fd706d697a6a70733c3d03be96fbf40d6.jpg)  
(a) LOAN

![](images/1a5637e573cf9eb0c408da046a37e2ac065cacc48b4809f2c2868c458221c6f5.jpg)  
(b) MNIST

![](images/8f933fdb4050d3ec33e55f89218fbc8279a3d5b06286796f582301edcaf99d95.jpg)  
(c) CIFAR  
Figure 5: Attack effectiveness comparison on two robust RL methods: RFA and FoolsGold

![](images/87d7c2c4b33267418a49e734002ee1787b4156a5e3c31bba141c064614a07cee.jpg)  
(d) Tiny-imagenet

Distributed Attack against Mitigating Sybils Defence. FoolsGold reduces aggregation weights of participating parties that repeatedly contribute similar gradient updates while retaining the weights of parities that provide different gradient updates (Fung et al., 2018). Fig.5 shows that DBA also outperforms centralized attack under FoolsGold. In three image datasets, the attack success rate of DBA is notably higher while converging faster. DBA in MNIST reaches  $91.55\%$  in round 30 when centralized attack fails with only  $2.91\%$  attack success rate. For LOAN, which are trained with a simple network, FoolsGolds cannot distinguish the difference between the malicious and clean updates and assigns high aggregation weights for attackers, leading to a fast backdoor success. To explain the effectiveness of DBA, we report FoolsGold's weights on adversarial parties in Tb.2 in Appendix. Comparing to centralized attack, although FoolsGold assigns smaller aggregation weights

to DBA attacker due to their similarity of backdoor target label, DBA is still more successful. This is because the sum of weights of distributed attackers could be larger than centralized attacker.

# 3.4 EXPLANATION VIA FEATURE VISUALIZATION AND FEATURE IMPORTANCE

Feature importance can be calculated by various classification tools or visually interpreted by class-specific activation maps. For example, in LOAN we show that the top features identified by different classifiers are quite consistent (see Tb.3 in Appendix). Here we use Grad-CAM (Selvaraju et al., 2017) and Soft Decision Tree (Frosst & Hinton, 2017) to provide explanations for DBA. More details about Soft Decision Tree trained on our datasets are discussed in Appendix A.4.

We use the Grad-CAM visualization method to explain why DBA is more stealthy, by inspecting their interpretations of the original and the backdoor target labels for a clean data input and the backdoorsed samples with local and global triggers, respectively. Fig.6 shows the Grad-CAM results of a hand-written digit '4'. We find that each locally triggered image alone is a weak attack as none of them can change the prediction (no attention on the top left corner where the trigger is embedded). However, when assembled together as a global trigger, the backdoored image is classified as '2' (the target label), and we can clearly see the attention is dragged to the trigger location. The fact that Grad-CAM results in most of locally triggered images are similar to the clean image, demonstrates the stealthy nature of DBA.

![](images/5f40e1f6c52cdcf18120a45268c50bc42432398876da18656b5980eb30f51186.jpg)  
Figure 6: Decision visualization of poisoned digit 4 with target 2 on a DBA poisoned model

![](images/621bccc8d0e33af653db8cd34ac8b2a25e482f9b213d7f5120578ad02e361039.jpg)  
Figure 7: Feature importance of LOAN learned from its soft decision tree

Using the soft decision tree of MNIST as another example, we find that the trigger area after poisoning indeed becomes much more significant for decision making in the corresponding soft decision tree, as shown in Fig.14 in Appendix.A.4. Similar conclusion is found in LOAN. We sort the absolute value of filter in the top node of a clean model to obtain the rank of 91 features (lower rank is more important) and then calculate their importance as  $(1\text{-rank} /91)^{*}100$ . Six insignificant features and six significant features are separately chosen to run DBA. The results in Fig.7 show that based on the soft decision tree, the insignificant features become highly important for prediction after poisoning.

# 4 ANALYSIS OF TRIGGER FACTORS IN DISTRIBUTED BACKDOOR ATTACK

Here we study the DBA trigger factors introduced in Sec.2.3 under Attack A-S, unless specified otherwise. We only change one factor in each experiment and keep other factors the same as in Sec.3.1. In Attack A-S, DBA-ASR shows the attack success rate while Main-Acc denotes the accuracy of the global model when the last distributed local trigger is embedded. DBA-ASR-t, which reveals the persistence, is the attack success rate of  $t$  rounds after a complete DBA is performed. Main-Acc-t is the main accuracy after  $t$  rounds. Note that in general we expect a small decrease for main task accuracy right after the DBA but will finally get back to normal after a few rounds of training. $^4$

# 4.1 EFFECTS OF SCALE

- Enlarging scale factor increases both DBA-ASR and DBA-ASR-t, and narrows the gap between them. For CIFAR, although the DBA-ASR reaches over  $90\%$  and barely changes once  $\gamma$  is bigger than 40, larger  $\gamma$  still have more positive impact on DBA-ASR-t.  
- For our four datasets, the more complex the model architecture (in Tb.1), the more obvious the decline in the main accuracy as  $\gamma$  increases, because the scaling undermines more model parameters

in complex neural network. The main accuracy of LOAN doesn't drop because of simple model, while the main accuracy of Tiny-imagenet in attacking round even drops to  $2.75\%$  when  $\gamma = 110$ .

- Larger scale factor alleviates the averaging impacts of central server for DBA, which leads to a more influential and resistant attack performance, but also cause the main accuracy of global model to descend in the attacking round for three image datasets. In addition, using large scale factor results in an anomalous update that is too different from other benign updates and is easy to detect based on the magnitude of the parameters. Therefore, there is a trade-off in choosing the scale factor.

![](images/1f8390d2a78ed84459e2b7bd0845e2b2fc087be730f6d797d358bf6fe55f2c79.jpg)  
(a) LOAN

![](images/86a5d4a9be52cd57d14324f76d329e2f9034d9f900918b6765f6f5a024ffe324.jpg)  
(b) MNIST

![](images/4547f272b07d33beef12c4f970b52785689f3751eeb4e2638810f1600e6b13c8.jpg)  
(c) CIFAR

![](images/fa714717487bb0441fba842de31285175b816a3833a0df8661e83bc3dbbc21d3.jpg)  
(d) Tiny-imagenet

![](images/3874d94332b6b30bd59cfdd14c41cbfeb457888197a4758ed2b625dd666ac392.jpg)  
Figure 8: Effects of Scale on Attack Success Rate and Model Accuracy  
(a) LOAN

![](images/01c91d4bf89aca830519d0df788eeaa82f25763843368ea66a65473788925e4d.jpg)  
(b) MNIST  
Figure 9: Effects of Trigger Location on Attack Success Rate and Model Accuracy

![](images/941e8094e0922f98f62bdb29343281f17b979952d653c4122c93f76261887b25.jpg)  
(c) CIFAR

![](images/c199ac4884b91c7f8300a110fd2910fa52b77366b1b4b9579f0dd51b2c6eed75.jpg)  
(d) Tiny-imagenet

# 4.2 EFFECTS OF TRIGGER LOCATION

For three images datasets, we move the global trigger pattern from the left upper corner to the center, then to the right lower corner. The dotted line in Fig.9 means that the trigger reaches the right boundary and starts to move along the right edges. The implementation details are in Appendix.A.6.

- We observe a U-shape curve between  $TL$  and DBA-ASR (in MNIST) / DBA-ASR-t (in Tiny-imagenet and MNIST). This is because the middle part in images usually contains the main object. DBA in such areas is harder to succeed and will be faster forgotten because these pixels are fundamental to the main accuracy. This finding is apparent in MNIST, where the main accuracy after 40 rounds only remains  $1.45\%$  in center ( $TL = 9$ ) while has  $91.57\%$  in left upper corner ( $TL = 0$ ).  
- Similar finding can be found in LOAN as shown in Fig.9.(a). DBA using low-importance features has higher success rate in attacking round and subsequent rounds. The low-importance trigger achieves  $85.72\%$  DBA-ASR after 20 rounds while the high-importance trigger is  $0\%$ .

# 4.3 EFFECTS OF TRIGGER GAP

- In the case of four local trigger patterns located in the four corners of an image, corresponding to the maximum trigger gap in Fig.10, the DBA-ASR and DBA-ASR-t are both low in image datasets. Such failure might be caused by the local convolution operations and large distance between local triggers so that the global model cannot recognize the global trigger.  
- The curve of DBA-ASR and DBA-ASR-t in Fig.10.(a) has a significant drop in the middle. This happens when the right lower local trigger covers the center areas in MNIST images. Similar observations can be explained based on Fig.9.(b)(d).  
- Using zero trigger gap in CIFAR and Tiny-imagenet, DBA still succeeds but we find the backdoor will be forgotten faster. We suggest using non-zero trigger gap when implementing DBA.

# 4.4 EFFECTS OF TRIGGER SIZE

- In image datasets, larger trigger size gives higher DBA-ASR and DBA-ASR-t. Nevertheless, they are stable once TS becomes large enough, suggesting little gain in using over-sized triggers.

![](images/4115a83b13297d14a42b5f2ecf1edb63b127c791eb7152090ca2c41571f182a5.jpg)  
(a) MNIST

![](images/577b2d6c52d15a94d1568d8edc9c8fea2c965a60cdc30751a93c407257b9b632.jpg)  
(b) CIFAR

![](images/15a57bb6e8b873a20112df2b623e5cf7586e235c3df260f2628b772b54e61dff.jpg)  
(c) Tiny-imagenet

![](images/e67bac0f478b2f98d2c15ccbac7cdbf5cefbb11e0d414fd5938211834c8e1c06.jpg)  
Figure 10: Effects of Trigger Gap on Attack Success Rate and Model Accuracy  
(a) MNIST  
Figure 11: Effects of Local Trigger Size on Attack Success Rate and Model Accuracy

![](images/117fe05e7f5a55f1a5745affe87dbdc9f4da114b01b59fc96ed1c65636ede56a.jpg)  
(b) CIFAR

![](images/472f079e4677f85e5a871b8181c7480a8277e440346544e0805570b02996e504.jpg)  
(c) Tiny-imagenet

- For MNIST, DBA-ASR is low when  $TS = 1$ . This is because each local trigger is too small to be recognized in global model. In the same setting, the centralized attack which uses the global pattern with 4 pixels also isn't very successful and its attack success rate soon decreases below  $10\%$  within 4 rounds. This reflects that under Attack A-S, backdoor attacks with too small trigger are ineffective.

# 4.5 EFFECTS OF POISON INTERVAL

- The attack performance is poor when all distributed attackers submit the scaled updates at the same round  $(I = 0)$  in all datasets because the scaling effect is too strong, vastly changing the parameter in the global model and causes it to fail in main accuracy. It's also ineffective if the poison interval is too long because the early embemed triggers may be totally forgotten.  
- The peaks in Fig.12.(a)(b) show that there exists an optimal poison round interval for LOAN and MNIST. DBA attackers can wait until the global model converges and then embeds the next local trigger to maximize backdoor performance, which is a competitive advantage over centralized attack.  
- In CIFAR and Tiny-imagenet, varying the interval from 1 up to 50 does not lead to remarkable changes in DBA-ASR and DBA-ASR-t, which manifests that the local trigger effect can last long and contribute to the attack performance of global trigger. From this aspect, distributed attack is extraordinarily robust to RL and should be considered as a more serious threat.

# 4.6 EFFECTS OF POISON RATIO

In our experiments, the training batch size is 64. As the X-axis variable (# of poisoned samples) in Fig.13 increases from 1, DBA-ASR and DBA-ASR-t first increase and then drop. It's intuitive that more poisoned data can lead to a better backdoor performance. However, a too large poison ratio means that the attacker scales up the weight of a local model of low accuracy, which leads to the failure of global model in the main task. In the case of poisoning full batch, after DBA, the global model in CIFAR and Tiny-imagenet trains the main task all over again, whose main accuracy is normal after 90 and 40 rounds, respectively. But in MNIST it is reduced to an overfitted model that predicts the target label for any input, so the attack success rate is always  $100\%$  while the main accuracy is about  $10\%$  in the subsequent rounds. Therefore, it's better for DBA to remain stealthy in its local training by using a reasonable poison ratio that also maintains accuracy on clean data.

# 4.7 EFFECTS OF DATA DISTRIBUTION

Under various data distributions, DBA-ASR is stable, indicating the practicability and robustness of DBA. See more details in Appendix.A.7.

![](images/cdc94353889cd071132f9274c461417121ca31caf60df92814b12802f8a265bc.jpg)  
(a) LOAN

![](images/a50b2d00950cf3d60df4f87cd29393324ab4752018fca528a027cb8049d39b74.jpg)  
(b) MNIST

![](images/b2e596dc44cc2ca70f903ad4ad2890e6dc10c71c00ba1789a4b5d2e9db3565c6.jpg)  
(c) CIFAR

![](images/a3f0466b595e4efd9fd35aa99e535a83899994c8160ec3bc481bb3d21f5321ef.jpg)  
(d) Tiny-imagenet

![](images/c30dbc08fdf1e0af2e74ca8e207535a9ed2b727aa12fbe7ad0317b37bb7c7bef.jpg)  
Figure 12: Effects of Poison Round Interval on Attack Success Rate and Model Accuracy  
(a) LOAN  
Figure 13: Effects of Poison Ratio on Attack Success Rate and Model Accuracy

![](images/478195a72b929247807a78342f3e3ba997cee5c2d7bf1e12b94d8a9176aa481e.jpg)  
(b) MNIST

![](images/f933d333a3cc7752de05523ebd21b89dacc6d9cd0b1fa7076c6d01ec95ba497f.jpg)  
(c) CIFAR

![](images/6cd764aa07dcbabdb0943af44abe3596ae1e32f32cef026ec778261c82b15d4b.jpg)  
(d) Tiny-imagenet

# 5 RELATED WORK

Federated Learning. McMahan et al. (2017) first introduced federated learning (FL) to solve the distributed machine learning problem. Since the training data is never shared with the server (aggregator), FL is in favor of machine learning with privacy and regulation constraints. In this paper, we discuss and analyze our experiments in standard FL settings performed in synchronous update rounds. Advanced FL for improving communication efficacy by compressing updates using random rotations and quantization has been recently studied in Konečný et al. (2016).

Backdoor Attack on Federated Learning. Bagdasaryan et al. (2018) proposed a model-poisoning approach on FL which replaced the global model with a malicious local model by scaling up the attacker's updates. Bhagoji et al. (2019) considered the case of one malicious attacker aiming to achieve both global model convergence and targeted poisoning attack, by boosting the malicious updates. They proposed two strategies, alternating minimization and estimating other benign updates, to evade the defences under weighted and non-weighted averaging for aggregation. We note that these works only consider centralized backdoor attack on FL.

Robust Federated Learning. Robust FL aims to train FL models while mitigating certain attack threats. Fung et al. (2018) proposed a novel defense based on the party updating diversity without limitation on the number of adversarial parties. It adds up historical updating vectors and calculate the cosine similarity among all participants to assign global learning rate for each party. Similar updating vectors will obtain lower learning rates and therefore the global model can be prevented from both label-flipping and centralized backdoor attacks. Pillutla et al. (2019) proposed a robust aggregation approach by replacing the weighted arithmetic mean with an approximate geometric median, so as to minimize the impacts of "outlier" updates.

# 6 CONCLUSIONS

Through extensive experiments on diverse datasets including LOAN and three image datasets in different settings, we show that in standard FL our proposed DBA is more persistent and effective than centralized backdoor attack: DBA achieves higher attack success rate, faster convergence and better resiliency in single-shot and multiple-shot attack scenarios. We also demonstrate that DBA is more stealthy and can successfully evade two robust FL approaches. The effectiveness of DBA is explained using feature visual interpretation for inspecting its role in aggregation. We also perform an in-depth analysis on the important factors that are unique to DBA to explore its properties and limitations. Our results suggest DBA is a new and more powerful attack on FL than current backdoor attacks. Our analysis and findings can provide new threat assessment tools and novel insights for evaluating the adversarial robustness of FL.

# REFERENCES

Eugene Bagdasaryan, Andreas Veit, Yiqing Hua, Deborah Estrin, and Vitaly Shmatikov. How to backdoor federated learning. arXiv preprint arXiv:1807.00459, 2018.  
Arjun Nitin Bhagoji, Supriyo Chakraborty, Prateek Mittal, and Seraphin Calo. Analyzing federated learning through an adversarial lens. In International Conference on Machine Learning, pp. 634-643, 2019.  
Peva Blanchard, Rachid Guerraoui, Julien Stainer, et al. Machine learning with adversaries: Byzantine tolerant gradient descent. In Advances in Neural Information Processing Systems, pp. 119-129, 2017.  
Nicholas Frosst and Geoffrey Hinton. Distilling a neural network into a soft decision tree. arXiv preprint arXiv:1711.09784, 2017.  
Clement Fung, Chris JM Yoon, and Ivan Beschastnikh. Mitigating sybils in federated learning poisoning. arXiv preprint arXiv:1808.04866, 2018.  
Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230-47244, 2019.  
Andrew Hard, Kanishka Rao, Rajiv Mathews, Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon, and Daniel Ramage. Federated learning for mobile keyboard prediction. arXiv preprint arXiv:1811.03604, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Wendy Kan. Lending club loan data, Mar 2019. URL https://www.kaggle.com/wendykan/lending-club-loan-data.  
Jakub Konečný, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Areas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, volume 54 of Proceedings of Machine Learning Research, pp. 1273-1282. PMLR, 20-22 Apr 2017.  
Thomas Minka. Estimating a dirichlet distribution, 2000.  
Krishna Pillutla, Sham M. Kakade, and Zaid Harchaoui. Robust Aggregation for Federated Learning. arXiv preprint, 2019.  
Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision, pp. 618-626, 2017.  
Claude E Shannon. Communication theory of secrecy systems. Bell system technical journal, 28(4): 656-715, 1949.  
Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. In Advances in Neural Information Processing Systems, pp. 4424-4434, 2017.  
Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 10(2):12, 2019.  
Timothy Yang, Galen Andrew, Hubert Eichner, Haicheng Sun, Wei Li, Nicholas Kong, Daniel Ramage, and Françoise Beaufays. Applied federated learning: Improving google keyboard query suggestions. arXiv preprint arXiv:1812.02903, 2018.  
Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018.
