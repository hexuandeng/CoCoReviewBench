# HODA: PROTECTING DNNS AGAINST MODEL EXTRACTION ATTACKS VIA HARDNESS OF SAMPLES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Model Extraction attacks exploit the target model's prediction API to create a surrogate model in order to steal or reconnoiter the functionality of the target model in the black-box setting. Several recent studies have shown that a data-limited adversary who has no or limited access to the samples from the target model's training data distribution can use synthesis or semantically similar samples to conduct model extraction attacks. As the training process of DNN-based classifiers is done in several epochs, we can consider this process as a sequence of subclasses so that each subclassier is created at the end of an epoch. We use the sequence of subclasses to calculate the hardness degree of samples. In this paper, we investigate the hardness degree of samples and demonstrate that the hardness degree histogram of a data-limited adversary's sample sequences is distinguishable from the hardness degree histogram of benign users' samples sequences, consisting of normal samples. Normal samples come from the target classifier's training data distribution. We propose Hardness-Oriented Detection Approach (HODA) to detect the sample sequences of model extraction attacks. The results demonstrate that HODA can detect the sample sequences of model extraction attacks with a high success rate by only watching 100 samples of them.

# 1 INTRODUCTION

Deep Neural Networks (DNNs) have shown impressive performance in various tasks in recent years that have encouraged the industry to deploy DNN-based models in a variety of real-world applications. Since the training process of DNNs and collecting training data is an expensive and tedious process, models are considered the intellectual property of organizations, and they must be kept secure. Therefore, models are often securely deployed on cloud servers, and only the creators can access the model parameters. Users are only allowed to query the model via a prediction API and receive predictions. Recent studies Tramér et al. (2016); Papernot et al. (2017); Juuti et al. (2019); Orekondy et al. (2019); Jagielski et al. (2020) demonstrate that an adversary can exploit the prediction API of a target model to create a surrogate model in order to steal or reconnoiter the functionality of the target model. Such attacks are called model extraction attacks, and they violate the intellectual property of model owners. Furthermore, the surrogate model can be leveraged to conduct other attacks on the target model in black-box setting, such as adversarial example attacks Papernot et al. (2017); Juuti et al. (2019) and membership inference attacks Shokri et al. (2017).

Most of the model extraction attacks use the target model's prediction API to label an unlabeled dataset to create the surrogate model's training set. In most real-world settings, the adversary has no or limited access to samples from the target model's training data distribution, which is called normal or in-distribution samples. Hence, most proposed attacks in the previous studies use some form of out-of-distribution samples, such as synthesis Papernot et al. (2017); Juuti et al. (2019) or semantically similar samples to the target model's training set Orekondy et al. (2019); Pal et al. (2020) to conduct model extraction attacks. We focus on such attacks in this paper. There are two main approaches to defend against model extraction attacks, manipulating the target model outputs to prevent adversary from producing high-quality surrogate model Lee et al. (2019); Orekondy et al. (2020); Kariyappa & Qureshi (2020); Kariyappa et al. (2021b) and detecting the sample sequences of model extraction attacks Kesarwani et al. (2018); Juuti et al. (2019). To the best of our knowledge, PRADA Juuti et al. (2019) is the only approach attempting to detect the sample sequences of model extraction attacks on DNN-based classifiers. We propose Hardness-Oriented Detection Approach

(HODA), a new approach to detect sample sequences of model extraction attacks, which outperforms PRADA by a large margin and has significantly less computational overhead.

Generally, the training process of DNN-based classifiers is done in several epochs, and the resulted classifier at the end of the last epoch is considered the final classifier. We can consider the training process of DNN-based classifiers as a sequence of subclasses in which the  $i^{th}$  subclassifier is created at the end of the  $i^{th}$  epoch. We say a sample is learned in the  $i^{th}$  epoch when the  $i^{th}$  subclassifier is the first subclassifier that agrees with all subsequent subclasses predicted labels. We consider the index of the epoch in which a new sample is learned as the hardness degree of that sample. Consequently, easy samples being learned in the early epochs have low hardness degree, and harder ones being learned in the last epochs have high hardness degree. It is important to note that we must save subclasses in the training phase of a target classifier in order to use their predictions to calculate the hardness degree of new samples. It is indicated the accuracy of classifiers is reduced by increasing the hardness degree of samples.

We demonstrate that the hardness degree histogram of normal sample sequences is distinguishable from the hardness degree histogram of model extraction attack sample sequences, and HODA uses this property to detect sample sequences of model extraction attacks. For each user, HODA calculates the distance between the hardness degree histograms of the user's samples and normal samples, and if the distance is greater than a threshold, the user is detected as an adversary. HODA can detect JBDA Papernot et al. (2017), JBRAND Juuti et al. (2019), and Knockoff Net Orekondy et al. (2019) attacks with a high success rate by only watching 100 samples of attack. We demonstrate that HODA is also highly effective when the target classifier is trained using transfer learning.

Contributions. (i) We demonstrate that the hardness degree of a sample for a classifier pertains to the training data distribution of that classifier. (ii) We indicate that the hardness degree histogram of normal samples is distinct from the hardness degree histograms of model extraction attack samples. (iii) We propose HODA to detect the sample sequences of model extraction attacks.

# 2 RELATED WORK

Model Extraction Attacks: Primary model extraction attacks try to extract the exact value of parameters Lowd & Meek (2005); Tramér et al. (2016) and hyperparameters Wang & Gong (2018) of shallow models. In recent years, the proposed attacks mainly aimed to steal or reconnoiter the functionality of deep neural networks by querying them in the black-box setting. It is often sensibly assumed in the literature that the adversary has no or limited access to samples from the training set distribution of target classifier. In order to overcome this issue, attacks generally use some form of out-of-distribution samples, such as synthesis or semantically similar samples, to create the surrogate classifier's training set. Knockoff Net Orekondy et al. (2019), ActiveThief Pal et al. (2020), and Copycat CNN da Silva et al. (2018) use semantically similar datasets to the target model's training set to train a surrogate classifier. In another line of studies, Papernot et al. (2017), Juuti et al. (2019), Yu et al. (2020), Truong et al. (2021), Kariyappa et al. (2021a), and Barbalau et al. (2020) use synthetic data to create the surrogate classifier's training set.

Model Extraction Defenses: Existing defense methods against model extraction attacks generally distribute into two branches: perturbation-based and detection-based. Perturbation-based defenses Lee et al. (2019); Orekondy et al. (2020); Kariyappa & Qureshi (2020) attempt to prevent adversaries from producing high-quality surrogate classifiers by adding perturbation to the target classifier outputs. Recently, Kariyappa et al. (2021b) proposed a new defense with the same goal as perturbation-based defenses, which does not perturb the target classifier outputs. Their approach employs an ensemble of diverse models to produce discontinuous predictions for out-of-distribution samples. Detection-based defenses attempt to detect the occurrence of model extraction attacks by observing successive input queries to the target classifier. Kesarwani et al. (2018) present a method to detect extraction attacks against Decision Tree models. PRADA Juuti et al. (2019) is the first proposed detection-based defense for DNN models. We propose a new defense detecting the sample sequences of model extraction attacks via hardness of samples.

# 3 MODEL EXTRACTION ATTACKS

The model extraction attack is one of the most serious threats against machine learning-based classifiers on remote servers, such as Machine Learning as a Service (MLaaS). The adversary's goal is to create a surrogate classifier  $f_{s}$  that imitates a target classifier  $f_{t}$  on task  $T$ . Most model extraction attacks exploit target model  $f_{t}$  to label unlabeled samples to create the surrogate model's training set. The adversary sends sample  $x_{i}$  to the target model and receives its output  $f_{t}(x_{i})$ , and then she uses pair  $(x_{i}, f_{t}(x_{i}))$  to train surrogate classifier  $f_{s}$ . The output type of target model can be label, label confidence, top-k values in probability vector, or the entire probability vector. We only consider label  $\bar{f}_{t}(x_{i})$  and the entire probability vector  $f_{t}(x_{i})$  as the output type of target classifiers in our experiments. There are two primary intents for adversaries to conduct model extraction attacks, stealing and reconnaissance.

Stealing: Producing a high performance classifier is an expensive and time-consuming process and requires computational resources and experts. Besides, given that DNNs need a large number of training samples, collecting data and labeling them is a complex and costly procedure for most real-world applications. Therefore, adversaries are motivated to take advantage of a target classifier to reduce the cost of creating a new classifier. The adversary's goal in stealing is to maximize the accuracy of surrogate model on data distribution  $\mathcal{D}_T$ . Hence, the adversary's goal is:

$$
\text {M a x i m i z e} \quad P _ {(x, y) \sim \mathcal {D} _ {T}} \bar {f} _ {s} (x) = y \tag {1}
$$

Reconnaissance: The model extraction attacks can be used to conduct other attacks in the black-box setting, such as adversarial example attacks Papernot et al. (2017); Goodfellow et al. (2015) and membership inference attacks Shokri et al. (2017). The adversary's goal in reconnaissance is to maximize the fidelity among surrogate and target classifiers in order to increase the success rate of black-box attacks. Similar to Jagielski et al. (2020), we consider label agreement among surrogate and target classifiers as the fidelity metric on data distribution  $\mathcal{D}_T$ . Hence, the adversary's goal is:

$$
\text {M a x i m i z e} \quad P _ {(x, y) \sim \mathcal {D} _ {T}} \bar {f} _ {s} (x) = \bar {f} _ {t} (x) \tag {2}
$$

Proposed model Extraction attacks create the surrogate classifier training set  $\mathbb{X}_s = \{(x_i, f_t(x_i))\}_{i=1}^B$  by various methods, where  $B$  is the attack budget. The attack budget determines the number of samples that an adversary is allowed to send to the target classifier and receive their associated predictions. After creating  $\mathbb{X}_s$ , the adversary trains surrogate classifier  $f_s$  to minimize empirical loss on  $\mathbb{X}_s$ . We suppose that the adversary knows the architecture and hyperparameters of the target classifier and uses them to train the surrogate classifier. It is noteworthy that our proposed defense is independent of surrogate classifiers' training process.

# 4 OUR PROPOSAL: HARDNESS-ORIENTED DETECTION APPROACH

We first introduce the hardness degree of samples and then show the hardness degree histogram of model extraction attack samples is distinguishable from the hardness degree histogram of normal samples. Using this observation, we propose Hardness-Oriented Detection Approach (HODA) to detect the sample sequences of model extraction attacks.

# 4.1 HARDNESS DEGREE OF SAMPLES

The training process of a DNN-based classifier can be considered a sequence of subclasses so that each subclassifier is created at the end of an epoch. Suppose that classifier  $f_{t}$  is trained for  $m$  epochs. The training process of classifier  $f_{t}$  can be represented as the following sequence of subclasses:

$$
<   f _ {t} ^ {0}, f _ {t} ^ {1}, f _ {t} ^ {2}, \dots , f _ {t} ^ {m - 1} > \tag {3}
$$

where subclassifier  $f_{t}^{i}$  is created at the end of the  $i^{th}$  epoch. We say sample  $x_{i}$  is learned in epoch  $e$  when  $f_{t}^{e}$  is the first subclassifier that its assigned label is equal to all subsequent subclassesifiers' predicted labels. Generally, as the number of epochs is increased, the performance of classifier  $f_{t}$  is improved so that easier samples are learned in the early epochs, and harder ones are learned in the last epochs. Therefore, the hardness degree of sample  $x_{i}$  for classifier  $f_{t}$ , which is displayed by  $\phi_{f_t}(x_i)$ , directly relates to the epoch number that  $x_{i}$  is learned by  $f_{t}$ . Hardness degree of sample  $x_{i}$  for classifier  $f_{t}$  is defined as follows:

$$
\phi_ {f _ {t}} \left(x _ {i}\right) = e \quad \text {s . t .} \quad \forall j \in [ e, m - 1 ], \bar {f} _ {t} ^ {e} \left(x _ {i}\right) = \bar {f} _ {t} ^ {j} \left(x _ {i}\right), \bar {f} _ {t} ^ {e} \left(x _ {i}\right) \neq \bar {f} _ {t} ^ {e - 1} \left(x _ {i}\right). \tag {4}
$$

![](images/59343800fa2917d767d3645be11001d1f5f7fa9955f529f95e8c3163d5fb8c42.jpg)

![](images/c58758423da398c33519e7f5c2b0517420b81ef9f95abd2bdbb30e4b60b2ca76.jpg)

![](images/7efb34de7bb268085e170f34d49953440ecddf12e471ab9c52ff7893097620c2.jpg)

![](images/af98e50c2e4d4797fe35095dc4907d328c6e969aeab206a46a64949d484a3334.jpg)  
Figure 1: The hardness degree histograms of CIFAR10 and CIFAR100 test samples for DenseNet121, ResNet18, and MobileNet classifiers.

![](images/e640f97e33573e02b7d4d6e4c1ab51eb925888971102c45cc2c067964285a2d7.jpg)  
(a) CIFAR10  
(b) CIFAR100

![](images/2b21a1cc4c944ea74aa4b5301e86686c225595fd0ca428efd2d39be5903e55cb.jpg)

The hardness degree domain is dependent on the number of subclasses, and since we have  $m$  subclasses, the hardness degree of a sample is in the range  $[0, m - 1]$ . To determine the hardness degree of unseen samples, we need to save subclasses at the end of each or several epochs in the training phase of target classifiers. When a new sample arrives, it is fed to all subclasses, and using their predictions, the hardness degree of that sample is calculated.

Table 1: The accuracy of classifiers on CIFAR10 and CIFAR100 test sets.  

<table><tr><td></td><td colspan="3">Acc(%)</td></tr><tr><td></td><td>ResNet18</td><td>DenseNet121</td><td>MobileNet</td></tr><tr><td>CIFAR10</td><td>94.36</td><td>94.92</td><td>93.59</td></tr><tr><td>CIFAR100</td><td>76.38</td><td>77.57</td><td>73.47</td></tr></table>

We train three various types of classifiers, including

DenseNet121 Huang et al. (2017), ResNet18 He et al. (2016), and MobileNet Sandler et al. (2018), on CIFAR10 and CIFAR100 training sets for 100 epochs (details of datasets in Appendix A). All classifiers are trained using stochastic gradient descent with momentum 0.9 and batch size 128. The learning rate is 0.1 and it is scheduled to be decreased in each epoch by a constant factor 0.955. The accuracy of classifiers is presented in Table 1. We save all 100 subclassesifiers in the training phase of each classifier and use them to calculate the hardness degree of samples. Figure 1 shows the hardness degree histogram of CIFAR10 and CIFAR100 test samples for various classifiers. The figure demonstrates that a large fraction of Cifar10 test samples are easy, and many samples are learned in the first few epochs. However, the learning of Cifar100 test samples is distributed over various epochs, and the number of hard samples is more than Cifar10.

We divide the hardness degree domain into ten hardness degree ranges and partition the CIFAR10 and CIFAR100 test samples based on their hardness degrees into ten groups so that each group consists of samples whose hardness degrees belong to the associated hardness degree range. Figure 2 shows the accuracy of classifiers on samples in each range of hardness degrees. More than  $99\%$  and  $95\%$  of samples being learned in the first 30 epochs (hardness degree  $< 30$ ) are correctly classified in CIFAR10 and CIFAR100 test sets, respectively. On the other side, less than  $55\%$  and  $36\%$  of samples being learned in the last 10 epochs (hardness degree  $\geq 90$ ) are correctly classified in CIFar10 and Cifar100 test sets, respectively. The results demonstrate that as the hardness degree of samples is increased, the accuracy of classifiers is reduced. As ResNet18 architecture achieves strong performance on both datasets at a reasonable computational cost, we use this architecture for target classifiers in the

![](images/e4f5456bf1f3c8fac12bf6517bac915707817c2c426d5805a9bf95c5312e7a0e.jpg)  
Figure 2: The accuracy of classifiers on samples in each range of hardness degrees.

rest of the paper. We conduct various model extraction attacks on two CIFAR10 and CIFAR100 target classifiers in the next subsection to depict the hardness degree histogram of their samples.

# 4.2 MODEL EXTRACTION ATTACKS SETUP

In line with prior works (Orekondy et al. (2020); Kariyappa & Qureshi (2020); Kariyappa et al. (2021b)), we select JBDA Papernot et al. (2017), JBRAND Juuti et al. (2019), and Knockoff Net Orekondy et al. (2019) model extraction attacks to evaluate our defense method. These attacks broadly represent two main strategies (synthesis or semantically similar samples) to conduct model extraction attacks. Jacobian-Based Dataset Augmentation (JBDA) Papernot et al. (2017) and its improvement (JBRAND) Juuti et al. (2019) assume that the adversary has access to a limited number of samples from the target classifier's training data distribution called seed samples, and they aim to augment seed samples using adversarial examples to increase the fidelity of the surrogate classifier to the target classifier. Orekondy et al. (2019) propose Knockoff Net (K.Net) attack that uses large public datasets that is semantically similar to the target classifier dataset to increase the accuracy of the surrogate classifier. We consider two versions of K.Net attack, K.Net CIFARX, and K.Net TIN. K.Net CIFARX attack uses CIFAR100 training set to extract CIFAR10 target classifier and vice versa. K.Net TIN employs TinyImageNet training set to extract target classifiers. More details about attacks and their implementations are presented in Appendix C.

To evaluate the performance of model extraction attacks, we use two ResNet18 classifiers being trained on CIFAR10 and CIFAR100 training sets as the target classifiers and conduct all four attacks on them. The default value of the attack budget in our experiments is 50000 (B=50K). Table 2 shows the accuracy and the fidelity of surrogate classifiers created by various model extraction attacks on CIFAR10 and CIFAR100 test samples. The results demonstrate that K.Net attacks have significantly better performance than jacobian-based attacks (JBDA and JBRAND), and when the output of target classifiers is probabilistic, the accuracy is significantly higher than the original classifier.

ity vector, the performance of attacks is considerably increased.

Table 2: The Accuracy (Acc) and the Fidelity (Fid) of surrogate classifiers being created by four various model extraction attacks on two target classifiers CIFAR10 and CIFAR100. The output type of target classifiers can be Label or Probability Vector (Prob. Vec.).  

<table><tr><td>\(f_t\)</td><td>Metric</td><td>Output type</td><td>JBDA</td><td>JBRAND</td><td>K.Net CIFARX</td><td>K.Net TIN</td></tr><tr><td rowspan="4">CIFAR10 ResNet18 (Acc: 94.36%)</td><td rowspan="2">Acc(%)</td><td>Prob. Vec.</td><td>41.00</td><td>43.33</td><td>79.86</td><td>78.86</td></tr><tr><td>Label</td><td>34.57</td><td>34.35</td><td>66.88</td><td>71.29</td></tr><tr><td rowspan="2">Fid(%)</td><td>Prob. Vec.</td><td>41.16</td><td>43.63</td><td>81.36</td><td>80.18</td></tr><tr><td>Label</td><td>34.86</td><td>34.45</td><td>67.98</td><td>72.43</td></tr><tr><td rowspan="4">CIFAR100 ResNet18 (Acc: 76.38%)</td><td rowspan="2">Acc(%)</td><td>Prob. Vec.</td><td>16.44</td><td>18.78</td><td>51.09</td><td>60.36</td></tr><tr><td>Label</td><td>8.62</td><td>8.07</td><td>23.20</td><td>32.88</td></tr><tr><td rowspan="2">Fid(%)</td><td>Prob. Vec.</td><td>16.90</td><td>19.13</td><td>54.59</td><td>64.90</td></tr><tr><td>Label</td><td>8.91</td><td>8.29</td><td>24.72</td><td>34.58</td></tr></table>

# 4.3 HARDNESS DEGREE OF MODEL EXTRACTION ATTACK SAMPLES

Figure 3 depicts the hardness degree histogram of 50000 samples generated by various attacks for CIFAR10 and CIFAR100 target classifiers. In this experiment, the architecture of target classifiers is ResNet18. We also present the hardness degree histogram of attack samples when the architecture of target classifiers is Densenet121 in Appendix D. Figure 3 demonstrates that the samples generated by various attacks have a very small number of easy samples, and most samples have medium and high hardness degrees. However, Figure 1 indicates that a high number of normal samples that are from the same distribution as the target classifier's training set are easy.

To investigate more on the hardness degree of attack and normal samples, Figure 4 displays two-dimensional visualization of CIFAR10 test samples using t-SNE. Figure 4a uses the logits of the CIFAR10 classifier to visualize CIFAR10 test samples, and the color of each sample is determined by its label. This figure has ten sample clusters where most samples of each cluster are from one class. Figure 4b illustrates the hardness degree of CIFAR10 test samples for CIFAR10 target classifier and demonstrates that most of the easy samples are in the high-density regions inside clusters, and most of the hard samples are in the low-density regions at the borders of clusters. Figure 4c is similar to Figure 4b, but the hardness degree of each sample is calculated via CIFAR100 target classifier. This figure demonstrates when the training data distribution of the classifier being used to calculate the hardness degree of samples becomes different from the distribution of CIFAR10 test samples,

![](images/76ea83355762f4af8fb496a758bf64ed8ac3a8447aadca0c8ba22c4403b0e839.jpg)

![](images/74d0219fdbdfe0e0df33726f602dcdab0b741bf105aab226267adf95d59cce54.jpg)

![](images/d482033aef623debd0b7eccee96df43d007682131031a5ab88a8c19a3b4c89ad.jpg)

![](images/cab4f7234c034dd2e5f6664b04336e0953d97e22dbf31125c37fce3aef781479.jpg)

![](images/6014b22d1c86b2fb46faa72f8e840b0472843dadd86bcc357237e70dea3eeb51.jpg)  
(a) CIFAR10

![](images/dc12fb3f7d7818985f00b70cd0ae739bd514376a14ca207681bb949c24d70d4e.jpg)  
(b) CIFAR100

![](images/b6ddfc773b4bc6147b6941cc1a29e18fc2442676147b560a1762aee7c23e112d.jpg)

![](images/a513acc94314e82293d90a69d04f9f460f5f4a60c7e55c3a94b417682e9e6d25.jpg)  
Figure 3: The hardness degree histograms of samples of four various model extraction attacks for CIFAR10 and CIFAR100 target classifiers. The budget of model extraction attacks is 50000.  
(a)  
Figure 4: (a) Visualization of CIFAR10 test samples. (b) Hardness of CIFAR10 test samples for CIFAR10 classifier. (c) Hardness of CIFAR10 test samples for CIFAR100 classifier.

![](images/033ae1337e2049a95207dffd5c35ef59eb43a34eb20e07dff86f2e86e99d6a4e.jpg)  
(b)

![](images/defe9694041dd59710ee9c1d02304db6e104d67696595a7f2f647fd21cf0e801.jpg)  
(c)

the hardness degree of a high number of samples is changed. Figure 4c shows hard and medium samples are distributed among clusters, and the number of easy samples is very small. Similar to Figure 4, we visualize CIFAR100 test samples and their hardness for CIFAR10 and CIFAR100 target classifiers in Appendix E. Figures 3 and 4 demonstrate that the hardness degree of a sample for a classifier pertains to the training data distribution of that classifier, and when an adversary uses samples from a different distribution than the target model's training set distribution, the hardness degree histogram of the adversary's samples is distinct from the hardness degree histogram of normal samples being used by benign users. We use histogram rather than hardness degree histogram in the rest of the paper for simplicity.

# 4.4 HARDNESS-ORIENTED DETECTION APPROACH

We propose Hardness-Oriented Detection Approach (HODA) to detect sample sequences of model extraction attacks. HODA requires normal histogram  $H_{n}$  representing the histogram of normal samples. When a new sample  $x_{i}$  from user  $u$  arrives, HODA calculates its hardness degree  $\phi_{f_t}(x_i)$ , and the histogram belongs to that user  $H_{u}$  is updated. After the number of samples sent by user  $u$  reaches a specific number  $num_{s}$ , HODA calculates Pearson distance between the histograms of normal samples  $H_{n}$  and user samples  $H_{u}$ , and if the distance is greater than a threshold  $\delta$ , the user  $u$  is detected as an adversary. Pearson Distance (PD) between two random variable  $X$  and  $Y$  is defined as follows:

$$
P D (X, Y) = 1 - \frac {\operatorname {C o v} (X , Y)}{\rho_ {X} \rho_ {Y}} \tag {5}
$$

where  $\operatorname{Cov}(X,Y)$  is the covariance between random variables  $X$  and  $Y$ , and  $\rho_{X}$  is the standard deviation of random variable  $X$ . The output of Pearson distance is in the range [0,2]. To calculate the Pearson distance between two histograms, HODA first transforms histograms into probability vectors by dividing the value of histogram bins by the total number of samples in the histogram  $(H_{n} / \text{sum}(H_{n})$  and  $H_{u} / \text{sum}(H_{u}))$  and then calculates the Pearson distance between them.

HODA uses normal sample set  $S_{HODA}$  to create  $H_{n}$  and calculate  $\delta$ . It randomly selects  $num_{seq}$  sample sequences with size  $num_{s}$  from the sample set  $S_{HODA}$  and for each sample sequence, produces a histogram and adds it to the histogram set  $HistSet$ . The normal histogram  $H_{n}$  is the average of all histograms in  $HistSet$ , and  $\delta$  is the maximum Pearson distance between  $H_{n}$  and all histograms in  $HistSet$ . Since  $\delta$  is independent of attacks and only relies on normal samples, HODA is not dependent on any attacks. Notably, HODA does not need to save samples of each user or their hardness degrees. It only keeps a vector  $(H_{u})$  that indicates the values of histogram bins for each user. Algorithm 1 in Appendix F describes HODA in details.

# 5 SETUP AND EVALUATION

Two normal sample sets  $S_{HODA}$  and  $S_{u}$  are required to evaluate the performance of HODA.  $S_{u}$  is used to simulate benign users. We randomly select  $40\%$  and  $60\%$  of test samples of each dataset for  $S_{HODA}$  and  $S_{u}$ , respectively. We randomly select  $num_{seq} = 40000$  sequences with size  $num_{s}$  from  $S_{HODA}$  to create  $H_{n}$  and calculate  $\delta$ . To evaluate the performance of HODA against model extraction attacks, we simulate 10000 benign users and 10000 adversaries for each attack. Each benign user sends a sequence of  $num_{s}$  samples randomly selected from  $S_{u}$ , and each adversary sends a sequence of  $num_{s}$  samples randomly selected from 50000 samples of attack in the order they were generated.

So far, we have used 100 subclasses to calculate the hardness degree of samples. However, it may not be possible to classify each sample by a high number of subclasses in practice. So in order to reduce the computational cost of HODA, we only use 11 subclasses to calculate the hardness degree of each sample, and these subclasses are saved in the training phase of target classifier  $f_{t}$  at the end of each 10 epochs  $< f_{t}^{0}, f_{t}^{9}, f_{t}^{19}, f_{t}^{29}, f_{t}^{39}, f_{t}^{49}, f_{t}^{59}, f_{t}^{69}, f_{t}^{79}, f_{t}^{89}, f_{t}^{99} >$ . Since the hardness degree domain depends on the number of subclasses, the hardness degree of a sample in HODA is in the range [0,10].

To the best of our knowledge, PRADA Juuti et al. (2019) is the only defense that is comparable to HODA. PRADA declares that the histogram of minimum  $L_{2}$  distance between a new sample and all previous samples of a benign user follows a Gaussian distribution. Hence, it uses the Shapiro-Wilk normality test to determine that a sample sequence belongs to a benign user or an adversary. Similar to HODA, PRADA also uses threshold  $\delta$  to detect sample sequences of model extraction attacks, and  $\delta$  is the only parameter of PRADA. Since PRADA needs to

Table 3: The detection rate and False Positive Rate (PFR) of PRADA and HODA against four various model extraction attacks on CIFAR10 and CIFAR100 target classifiers.  

<table><tr><td rowspan="2" colspan="2"></td><td rowspan="2">num_s</td><td rowspan="2">δ</td><td rowspan="2">FPR(%)</td><td colspan="4">Detection Rate of Attacks(%)</td></tr><tr><td>JBDA</td><td>JBRAND</td><td>K.Net CIFARX</td><td>K.Net TIN</td></tr><tr><td rowspan="4">CIFAR10</td><td rowspan="2">PRADA</td><td>100</td><td>0.818</td><td>0.01</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>500</td><td>0.973</td><td>0.05</td><td>96.7</td><td>94.2</td><td>4.4</td><td>1.6</td></tr><tr><td rowspan="2">HODA</td><td>50</td><td>0.290</td><td>0.02</td><td>100</td><td>100</td><td>99.92</td><td>99.73</td></tr><tr><td>100</td><td>0.154</td><td>0.02</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td rowspan="4">CIFAR100</td><td rowspan="2">PRADA</td><td>500</td><td>0.550</td><td>0.01</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>1000</td><td>0.953</td><td>0.03</td><td>67.3</td><td>73.5</td><td>0</td><td>0</td></tr><tr><td rowspan="2">HODA</td><td>50</td><td>0.716</td><td>0.02</td><td>94.65</td><td>100</td><td>90.68</td><td>89.06</td></tr><tr><td>100</td><td>0.349</td><td>0.02</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>

save each user's samples and calculate  $L_{2}$  distance between them, it has a high computational overhead.

Table 3 indicates the detection rate and False Positive Rate (PFR) of PRADA and HODA against four various model extraction attacks on CIFAR10 and CIFAR100 target classifiers. We evaluate HODA when it only watches 50 and 100 samples of each user ( $num_{s} = 50$  and  $num_{s} = 100$ ), and since PRADA needs to watch more samples to detect attacks, we use larger  $num_{s}$  to evaluate PRADA. PRADA and HODA have very low false-positive rates. False-Positive Rate (FPR) indicates the percentage of benign users' sample sequences wrongly detected as an attack. The results demonstrate that HODA is very effective against model extraction attacks, and it outperforms PRADA by

![](images/7296230ce0b8d9b35a69755e052e5af6606f4969a38ec12fa8655024148935a6.jpg)  
(a) CUB200

![](images/c39b3e7fdaff37dd0454c4fc8aeb41d9ccf52a8fd5f09726f5034578229c5f88.jpg)  
Figure 5: The left histogram in subfigures a and b shows the hardness degree histogram of CUB200 and Caltech256 test samples, respectively. The right histogram in each subfigure indicates the hardness degree histograms of K.Net ILSVRC12 attack samples on CUB200 (a) and Caltech256 (b) target classifiers.

![](images/82c8e17478d906b8f2d1753a64c037d8fb50b06aba96841e6899306362d85f41.jpg)  
(b) Caltech256

![](images/5a80e8bed396736bc19801e71f669d3216dd23bc92e6654eb6e876e6b0da3a42.jpg)

a large margin. Since HODA does not rely on the distance between samples, it can detect knockoff Net attacks that use natural samples. HODA also has better performance on jacobian-based attacks. The runtime and the number of samples that need to be stored by PRADA depend on the attack. Nevertheless, for  $num_{s} = 500$  and CIFAR10 target classifier, the average runtime of PRADA for each user is 0.47 seconds on Tesla K80 GPU, and 471 samples are stored for each user on average. For each user, the average runtime of HODA is 0.0012 seconds, and it only stores a vector with size 11 representing a hardness degree histogram. Although HODA requires the predictions of 11 models to calculate the hardness degree of each sample, there is no sequential relationship between models, and they can predict parallelly, so HODA does not increase the prediction time of target models.

# 5.1 TRANSFER LEARNING

Transfer learning is a machine learning technique that initializes the parameters of the target task classifier using the parameters of a pre-trained source task classifier. We train two new target classifiers on CUB200 and Caltech256 datasets using transfer learning (details of datasets in Appendix A). The training process of new target classifiers is the same as CIFAR10 and CIFAR100 target classifiers (Section 4.1). We initialize the parameters of target classifiers from a pre-trained ImageNet Deng et al. (2009) classifier and train all layers of target classifiers. Orekondy et al. (2020) indicate that jacobian-based model extraction attacks have very poor performance on high dimensional datasets. Thereby, we only evaluate the performance of target classifiers against K.Net ILSVRC12 attack. K.Net ILSVRC12 is the Knockoff Net attack that uses ILSVRC12 dataset as the surrogate classifier's training set. The budget of K.Net ILSVRC12 is 50000, and the output of target classifiers is the entire probability vector. The accuracy of CUB200 target classifier and its surrogate classifier is  $73.7\%$  and  $59.3\%$ , respectively, and the accuracy of Caltech256 target classifier and its surrogate classifier is  $77.2\%$  and  $72.2\%$ , respectively.

Figure 5 depicts the hardness degree histogram of CUB200 and Caltech256 test sets on the associated target classifier and also the hardness degree histogram of K.Net ILSVRC12 samples for both target classifiers. The figure demonstrates that the majority number of K.Net ILSVRC12 attack samples are hard (hardness degree  $>70$ ), and the number of easy samples (hardness degree  $<30$ ) is very small. We replicate the experiment of the previous section to evaluate the performance of HODA against K.Net ILSVRC12 attack with the same parame

ters. Table 4 shows the performance of HODA against K.Net ILSVRC12 attack on both target classifiers. The results demonstrate that even the starting point of target classifiers' parameters is not random, HODA is very effective in detecting K.Net ILSVRC12 attack.

Table 4: The detection rate and False Positive Rate (PFR) of HODA against K.Net ILSVRC12 attack.  

<table><tr><td rowspan="2">Target Model</td><td rowspan="2">num_s</td><td rowspan="2">δ</td><td rowspan="2">FPR(%)</td><td>Detection Rate(%)</td></tr><tr><td>K.Net ILSVRC12</td></tr><tr><td rowspan="2">CUB200</td><td>50</td><td>0.973</td><td>0.01</td><td>97.50</td></tr><tr><td>100</td><td>0.393</td><td>0.02</td><td>100</td></tr><tr><td rowspan="2">Caltech256</td><td>50</td><td>0.694</td><td>0.01</td><td>99.98</td></tr><tr><td>100</td><td>0.152</td><td>0.01</td><td>100</td></tr></table>

(a)  $P_{n} = 25\%$  
![](images/2afb70d94c675ddf7ffc9c0bbc744e59afe85d8e9998e776c971b7c1f1e96dcd.jpg)  
- JBDA - JBRAND  $\triangle$  K.Net CIFARX  $\diamond$  K.Net TIN —— Target model: CIFAR10 - - - Target model: CIFAR100

![](images/2e44f601f39fa57c6cb38c7dadaa143f525b7a6ddfbb52fec6c70fd24e1d676a.jpg)  
(b)  $P_{n} = 50\%$

![](images/f3fc117bd65fa43676e4254ae80f16fa82a5f5884b860928272a85d17e558985.jpg)  
Figure 6: The detection rate of HODA for various percentages of normal samples  $P_{n}$  over different values of  $num_{s}$ .  
(c)  $P_{n} = 75\%$

![](images/9431f776a2a0becd995b2709afcea0dd12452a17248913951a9feb81e7b59188.jpg)  
(d)  $P_{n} = 90\%$

# 6 DISCUSSION ON ADAPTIVE ADVERSARY

An adaptive adversary who is aware of HODA must send her queries based on the hardness degree histogram of normal samples to evade HODA. We consider two scenarios for an adaptive adversary to conduct model extraction attacks. In the first scenario, the adversary has no access to normal samples, and she only can use synthetic or semantically similar samples to extract the target model. There are two reasons why such attacks are hard to conduct. First, the adversary needs samples with various degrees of hardness; however, since the adversary has no access to the target classifier, she can not determine the hardness degree of her samples for the target classifier. Second, the adversary has no access to the histogram of normal samples to generate her samples based on it.

In the second scenario, we assume the adversary has access to a limited number of normal samples, and she can use normal samples to make her hardness degree histogram more similar to the hardness degree histogram of normal samples. To evaluate HODA in this scenario, we suppose that the adversary has access to 1000 normal samples from  $S_{user}$  and she sends a sample sequence of which  $P_{n}\%$  is filled by normal samples, and the rest is filled by model extraction attack samples. Notably, when the number of normal samples in the sequence exceeds 1000, the adversary sends duplicate normal samples. It is important to note that the cost of attack is increased by a factor of  $\frac{1}{1 - (P_n / 100)}$  in this scenario. Figure 6 shows the detection rate of HODA for various  $P_{n}$  over different values of  $num_{s}$ . The false-positive rate of all experiments is less than  $0.2\%$ . The figure demonstrates that increasing  $num_{s}$  improves the detection rate of HODA. Except for K.Net attacks on CIFAR10 target classifier in  $P_{n} = 90\%$ , HODA can detect all attacks with a high success rate by increasing  $num_{s}$ . Due to the dataset limitation, we can not evaluate HODA for  $num_{s} > 4000$ . However, we think the detection rate of HODA against K.Net attacks on CIFAR10 target classifier in  $P_{n} = 90\%$  will be improved for  $num_{s} > 4000$ . Altogether, we think the main challenge of an adaptive adversary to evade HODA is to collect easy samples, which are very rare in out-of-distribution samples based on our experiments.

# 7 CONCLUSIONS

This paper demonstrates that the hardness degree of samples is important in trustworthy machine learning. We investigated the hardness degree of samples and demonstrated that the hardness degree histogram of model extraction attack samples is different from the hardness degree histogram of normal samples. Using this observation, we proposed Hardness-Oriented Detection Approach (HODA) to detect sample sequences of model extraction attacks. HODA can detect the sample sequences of model extraction attacks with a high success rate by only watching 100 samples of attacks.

# REFERENCES

Y. Adi, C. Baum, M. Cisse, B. Pinkas, and J. Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdoorsing. In 27th USENIX Security Symposium, pp. 1615-1631, August 2018.  
A. Barbalau, A. Cosma, R. T. Ionescu, and M. Popescu. Black-box ripper: Copying black-box models using generative evolutionary algorithms. In Advances in Neural Information Processing

Systems, volume 33, pp. 20120-20129, 2020.  
L. Batina, S. Bhasin, D. Jap, and S. Picek. CSI NN: Reverse engineering of neural network architectures through electromagnetic side channel. In 28th USENIX Security Symposium, pp. 515-532, August 2019.  
N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy, pp. 39-57, 2017.  
V. Chandrasekaran, K. Chaudhuri, I. Giacomelli, S. Jha, and S. Yan. Exploring connections between active learning and model extraction. In 29th USENIX Security Symposium, pp. 1309-1326, 2020.  
K. Chen, S. Guo, T. Zhang, X. Xie, and Y. Liu. Stealing deep reinforcement learning models for fun and profit. In Proceedings of the 2021 ACM Asia Conference on Computer and Communications Security, pp. 307-319, 2021.  
F. Croce and M. Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In Proceedings of the 37th International Conference on Machine Learning, pp. 2206-2216, 2020.  
J. R. C. da Silva, R. F. Berriel, C. Badue, A. F. de Souza, and T. Oliveira-Santos. Copycat CNN: stealing knowledge by persuading confession with random non-labeled data. In 2018 International Joint Conference on Neural Networks, pp. 1-8, 2018.  
J. Deng, W. Dong, R. Socher, L. J. Li, K. L., and F. F. Li. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pp. 248-255, 2009.  
J. Frankle, D. J. Schwab, and A. S. Morcos. The early phase of neural network training. In 8th International Conference on Learning Representations, 2020.  
I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In 3rd International Conference on Learning Representations, 2015.  
G. Griffin, A. Holub, and P. Perona. Caltech-256 object category dataset. 2007.  
G. Hacohen, L. Choshen, and D. Weinshall. Let's agree to agree: Neural networks share classification order on real datasets. In Proceedings of the 37th International Conference on Machine Learning, pp. 3950-3960, 2020.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, pp. 770-778, 2016.  
X. He, J. Jia, M. Backes, N. Z. Gong, and Y. Zhang. Stealing links from graph neural networks. In 30th USENIX Security Symposium, August 2021.  
S. Hong, M. Davinroy, Y. Kaya, S. N. Locke, I. R., K. Kulda, D. Dachman-Soled, and T. Dumitras. Security analysis of deep neural networks operating in the presence of cache side-channel attacks. ArXiv, abs/1810.03487, 2018.  
G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2261-2269, 2017.  
M. Jagielski, N. Carlini, D. Berthelot, A. Kurakin, and N. Papernot. High accuracy and high fidelity extraction of neural networks. In 29th USENIX Security Symposium, pp. 1345-1362, 2020.  
H. Jia, C. A. Choquette-Choo, V. Chandrasekaran, and N. Papernot. Entangled watermarks as a defense against model extraction. In 30th USENIX Security Symposium, August 2021.  
M. Juuti, S. Szyller, S. Marchal, and N. Asokan. PRADA: protecting against DNN model stealing attacks. In IEEE European Symposium on Security and Privacy, pp. 512-527, 2019.

S. Kariyappa and M. K. Qureshi. Defending against model stealing attacks with adaptive misinformation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 767-775, 2020.  
S. Kariyappa, A. Prakash, and M. Qureshi. Maze: Data-free model stealing attack using zeroth-order gradient estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13814-13823, 2021a.  
S. Kariyappa, A. Prakash, and M. K Qureshi. Protecting dnns from theft using an ensemble of diverse models. In International Conference on Learning Representations, 2021b.  
M. Kesarwani, B. Mukhoty, V. Arya, and S. Mehta. Model extraction warning in mlaas paradigm. In Proceedings of the 34th Annual Computer Security Applications Conference, pp. 371-380, 2018.  
K. Krishna, G. Singh Tomar, A. P. Parikh, N. Papernot, and M. Iyyer. Thieves on sesame street! model extraction of bert-based apis. In 8th International Conference on Learning Representations, 2020.  
A. Krizhevsky. Learning multiple layers of features from tiny images. 2009.  
Y. Le and X. Yang. Tiny imagenet visual recognition challenge. 2015.  
T. Lee, B. Edwards, I. M. Molloy, and D. Su. Defending against neural network model stealing attacks using deceptive perturbations. In 2019 IEEE Security and Privacy Workshops, pp. 43-49, 2019.  
D. Lowd and C. Meek. Adversarial learning. In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining, pp. 641-647, 2005.  
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In 6th International Conference on Learning Representations, 2018.  
K. Mangalam and V. Prabhu. Do deep neural networks learn shallow learnable examples first? In In Workshop on Identifying and Understanding Deep Learning Phenomena at 36th International Conference on Machine Learning, 2019.  
T. Orekondy, B. Schiele, and M. Fritz. Knockoff nets: Stealing functionality of black-box models. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 4954-4963, 2019.  
T. Orekondy, B. Schiele, and M. Fritz. Prediction poisoning: Towards defenses against DNN model stealing attacks. In 8th International Conference on Learning Representations, 2020.  
S. Pal, Y. Gupta, A. Shukla, A. Kanade, S. Shevade, and V. Ganapathy. Activethief: Model extraction using active learning and unannotated public data. Proceedings of the AAAI Conference on Artificial Intelligence, pp. 865-872, Apr. 2020.  
N. Papernot, P. D. McDaniel, I. J. Goodfellow, S. Jha, Z. Berkay Celik, and A. Swami. Practical black-box attacks against machine learning. In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security, pp. 506-519, 2017.  
I. Pliushch, M. Mundt, N. Lupp, and V. Ramesh. When deep classifiers agree: Analyzing correlations between learning order and image statistics. ArXiv, abs/2105.08997, 2021.  
M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L. Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4510-4520, 2018.  
R. Shokri, M. Stronati, C. Song, and V. Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy, pp. 3-18, 2017.  
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. J. Goodfellow, and R. Fergus. Intriguing properties of neural networks. In 2nd International Conference on Learning Representations, 2014.

S. Szyller, B. Atli, S. Marchal, and N. Asokan. Dawn: Dynamic adversarial watermarking of neural networks. ArXiv, abs/1906.00830, 2019.  
F. Tramer, F. Zhang, A. Juels, M. K. Reiter, and T. Ristenpart. Stealing machine learning models via prediction apis. In 25th USENIX Security Symposium, pp. 601-618, 2016.  
J. B. Truong, P. Maini, R. J. Walls, and N. Papernot. Data-free model extraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4771-4780, 2021.  
C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.  
B. Wang and N. Z. Gong. Stealing hyperparameters in machine learning. In 2018 IEEE Symposium on Security and Privacy, pp. 36-52, 2018.  
M. Yan, C. W. Fletcher, and J. Torrellas. Cache telepathy: Leveraging shared resource attacks to learn DNN architectures. In 29th USENIX Security Symposium, pp. 2003-2020, August 2020.  
H. Yu, K. Yang, T. Zhang, Y. Tsai, T. Ho, and Y. Jin. Cloudleak: Large-scale deep learning models stealing through adversarial examples. In 27th Annual Network and Distributed System Security Symposium, 2020.  
J. Zhang, Z. Gu, J. Jang, H. Wu, M. P. Stoecklin, H. Huang, and I. Molloy. Protecting intellectual property of deep neural networks with watermarking. In Proceedings of the 2018 on Asia Conference on Computer and Communications Security, pp. 159-172, 2018.  
Y. Zhu, Y. Cheng, H. Zhou, and Y. Lu. Hermes attack: Steal DNN models with lossless inference accuracy. In 30th USENIX Security Symposium, August 2021.
