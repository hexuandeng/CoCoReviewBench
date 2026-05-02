# THE EFFECT OF DIVERSITY IN META-LEARNING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Few-shot learning aims to learn representations that can tackle novel tasks given a small number of examples. Recent studies show that task distribution plays a vital role in the performance of the model. Conventional wisdom is that task diversity should improve the performance of meta-learning. In this work, we find evidence to the contrary; we study different task distributions on a myriad of models and datasets to evaluate the effect of task diversity on meta-learning algorithms. For this experiment, we train on multiple datasets, and with three broad classes of meta-learning models - Metric-based (i.e., Protonet, Matching Networks), Optimization-based (i.e., MAML, Reptile, and MetaOptNet), and Bayesian meta-learning models (i.e., CNAPs). Our experiments demonstrate that the effect of task diversity on all these algorithms follows a similar trend, and task diversity does not seem to offer any benefits to the learning of the model. Furthermore, we also demonstrate that even a handful of tasks, repeated over multiple batches, would be sufficient to achieve a performance similar to uniform sampling and draws into question the need for additional tasks to create better models.

# 1 INTRODUCTION

It is widely recognized that humans can learn new concepts based on very little supervision, i.e., with few examples (or "shots"), and generalize these concepts to unseen data as mentioned by Lake et al. (2011). Recent advances in deep learning, on the other hand, have primarily relied on datasets with large amounts of labeled examples, primarily due to overfitting concerns in low data regimes. Although the development of better data augmentation and regularization techniques can alleviate these concerns, many researchers now assume that future breakthroughs in low data regimes will emerge from meta-learning, or "learning to learn." Here, we study the effect of task diversity in the low data regime and its effect on various models. In this meta-learning setting, a model is trained on a handful of labeled examples at a time under the assumption that it will learn how to correctly project examples of different classes and generalize this knowledge to unseen labels at test time. Although this setting is often used to illustrate the remaining gap between human capabilities and machine learning, we could argue that the domain of meta-learning is still nascent. The domain of task selection has remained virtually unexplored in this setting.

Conventional wisdom is that the performance of the model will improve as we train on more diverse tasks. To test this hypothesis to its limits, we define various task samplers which either limit task diversity by selecting a subset of overall tasks or improving task diversity using approaches such as Determinantal Point Processes (DPPs) proposed by Macchi (1975).

Our contributions in this work are as follows:

- We show that, against conventional wisdom, task diversity does not significantly boost performance in meta-learning. Instead, limiting task diversity and repeating the same tasks over the training phase allows the model to obtain performances similar to models trained on Uniform Sampler without any adverse effects.  
- We also show that increasing task diversity using sophisticated samplers such as DPP or Online Hard Task Mining (OHTM) Samplers do not significantly boost performance. Instead, the dynamic-DPP Sampler harms the model due to the increased task diversity.  
- We empirically show that repeating tasks over the training phase can perform similarly to a model trained on the Uniform Sampler, achieving similar performance with only a

fragment of data. This key finding questions the need to increase the support set pool to improve the model's performance.

# 2 RELATED WORKS

Meta-learning formulations typically rely on episodic training, wherein an algorithm adapts to a task, given its support set, to minimize the loss incurred on the query set. Meta-learning methods differ in terms of the algorithms they learn, and can be broadly classified under four prominent classes: Metric-based, Model-based, Optimization-based and Bayesian-based approaches. Metric-based methods such as Koch et al. (2015); Vinyals et al. (2016); Shell et al. (2017); Sung et al. (2018) operate on the core idea similar to nearest neighbors algorithm and kernel density estimation. These methods are also called non-parametric approaches. Model-based methods such as Santoro et al. (2016); Munkhdalai & Yu (2017) depend on a model designed specifically for fast learning, which updates its parameters rapidly with a few training steps, achieved by its internal architecture or controlled by another meta-learner model. Generic deep learning models learn through backpropagation of gradients, which are neither designed to cope with a small number of training samples nor converge within a few optimization steps. To address this, Optimization-based methods such as Ravi & Larochelle (2016); Finn et al. (2017); Nichol et al. (2018) were proposed, which were better suited to learn from a small number of samples. However, all the above approaches are deterministic and are not the most suited for few-shot problems that are generally ambiguous. Hence, Bayesian-based methods such as Yoon et al. (2018); Requeima et al. (2019) were proposed which helped address the above issue.

Although research in meta-learning models has attracted much attention recently, the effect of task diversity is virtually unexplored in the domain of meta-learning. However, task sampling and task diversity have been more extensively studied in other closely related problems such as active learning. Active learning involves selecting unlabeled data items in order to improve an existing classifier. Although most of the approaches in this domain are based on heuristics, there are few approaches to sample a batch of samples for active learning. Ravi & Larochelle (2018) proposed an approach to sample a batch of samples using a protonet as the backbone architecture. The model tries to maximize the query set, given support set and unlabeled data. Other works such as Hsu et al. (2018) proposed a framework named CACTUs, which samples tasks/examples using relatively simple task construction mechanisms such as clustering embeddings. The unsupervised representations learned via these samples lead to a good performance on various downstream human-specified tasks.

Although nascent, a few recent works aim to improve meta-learning by explicitly looking at the task structure and relationships. Among these, Yin et al. (2019) proposed an approach to handle the lack of mutual exclusiveness among different tasks through an information-theoretic regularized objective. In addition, several popular meta-learning methods Lee et al. (2019); Snell et al. (2017) improve the meta-test performance by changing the number of ways or shots of the sampled metatraining tasks, thus increasing the complexity and diversity of the tasks. Other works such as Liu et al. (2020a) proposed an approach to sample classes using class-pair-based sampling and class-based sampling. The Class-pair based Sampler selects pairs of classes that confuse the model the most. The class-based Sampler samples each class independently and does not consider the task's difficulty as a whole. Our OHTM sampler is similar to the Class-pair based Sampler. Other works such as Liu et al. (2020b) propose to augment the set of possible tasks by augmenting the predefined set of classes that generate the tasks with varying degrees of rotated inputs as new classes. Other works such as Setlur et al. (2020) look at the structure and diversity of tasks specifically through the lens of support set diversity, and show that, surprisingly, reducing diversity (by fixing support set) not only maintains—but in many cases, significantly improves—the performance of meta-learning. This experiment is very similar to our No Diversity Task Sampler if the size of the support set is equal to the number of classes per task. However, in this work, we extend their work on MetaOptNet, Protonet to many other models and a myriad of samplers to better understand task diversity in meta-learning. To the best of our knowledge, we are the first to study the effect of task diversity in meta-learning to this extent.

![](images/1e1903c41eac671a329da645420907b1eb6882c510706bc41b440305b07fbbd1.jpg)  
(a) Uniform sampler

![](images/293b6fa6b72bcba8b3ef496db9844bc30d1b73a0adf2ac8dd123946531b2bd7f.jpg)  
(b) No Diversity Task sampler

![](images/b56f44bf8b6c6a8d729cedca0ebd2b68251625180d8872f62198a605f9798fdc.jpg)  
Figure 1: Illustration of (a) the Uniform Sampler, (b) the No Diversity Task Sampler, and (c) the No Diversity Batch Sampler.  
(c) No Diversity Batch sampler

# 3 BACKGROUND

Here, we review some of the fundamental ideas required to understand our few-shot learning experiments better.

# 3.1 EPISODIC FEW-SHOT LEARNING

In episodic few-shot learning, an episode is represented as a K-way, N-shot classification problem where  $\mathbf{N}$  is the number of examples per class and  $\mathbf{K}$  is the number of unique class labels. During training, the data in each episode is provided as a support set  $S = \{(x_{1,1},y_{1,1}),\dots,(x_{N,K},y_{N,K})\}$  where  $x_{i,j}\in \mathbb{R}^D$  is the i-th instance of the j-th class, and  $y_{j}\in \{0,1\}^{K}$  is its corresponding one-hot labeling vector. Each episode aims to optimize a function  $f$  that classifies new instances provided through a "query" set  $Q$ , containing instances of the same class as  $S$ . This task is difficult because  $N$  is typically very small (e.g., 1 to 10). The classes change every episode. The actual test set used to evaluate a model does not contain classes seen in support sets during training. In the task-distribution view, meta-learning is a general-purpose learning algorithm that can generalize across tasks and ideally enable each new task to be learned better than the last. We can evaluate the performance of  $\omega$  over a distribution of tasks  $p(\mathcal{T})$ . Here we loosely define a task to be a dataset and loss function  $\mathcal{T} = \{\mathcal{D},\mathcal{L}\}$ . Learning how to learn thus becomes:

$$
\min  _ {\omega} \mathbb {E} _ {\tau \sim p (\tau)} \mathcal {L} (\mathcal {D}; \omega) \tag {1}
$$

where  $\mathcal{L}(\mathcal{D};\omega)$  measures the performance of a model trained using  $\omega$  on dataset  $\mathcal{D}$  and  $p(\tau)$  indicates the task distribution. In this experiment, we extend this setting such that we vary the task diversity in the train split to study the effects on test split, which remains to use uniform or random sampling for tasks.

# 3.2 DETERMINANTAL POINT PROCESSES (DPPS)

A DPP is a probability distribution over subsets of a ground set  $\mathcal{V}$ , where we assume  $\mathcal{V} = \{1,2,\dots,N\}$  and  $N = |\mathcal{V}|$ . An L-ensemble defines a DPP using a real, symmetric, and positive-definite matrix  $\mathbf{L}$  indexed by the elements of  $\mathcal{V}$ . The probability of sampling a subset  $Y = A \subseteq \mathcal{V}$  can be written as:

$$
P (Y = A) \propto \det  \mathbf {L} _ {A}, \tag {2}
$$

where  $\mathbf{L}_A\coloneqq [L_{i,j}]_{i,j\in A}$  is the restriction of  $\mathbf{L}$  to the entries indexed by the elements of A. As  $\mathbf{L}$  is a positive semi-definite, there exists a  $d\times N$  matrix  $\Psi$  such that  $\mathbf{L} = \Psi^T\Psi$  where  $d\leq N$ . Using this principle, we define the probability of sampling as:

$$
P (Y = A) \propto \det  \mathbf {L} _ {A} = \operatorname {V o l} ^ {2} \left(\left\{\Psi_ {i} \right\} _ {i \in A}\right), \tag {3}
$$

![](images/2b72bb9d3222bddcdf58e972f3a2bf8bfa5e9edaf03e9fdc33b91bcc6ed9f539.jpg)  
(a) No Diversity Tasks per Batch sampler  
(b) Single Batch Uniform sampler

![](images/25a85deae3ed5df10d98cb99a5cd815611ef5da0599fb81a9540b9b7df5e0ec0.jpg)  
Figure 2: Illustration of (a) the No Diversity Task per Batch Sampler, and (b) the Single Batch Uniform Sampler.

where the RHS is the squared volume of the parallelepiped spanned by  $\{\Psi_i\}_{i\in A}$ . In Eq. 3,  $\Psi_{i}$  is defined as the feature vector of element  $i$ , and each element  $L_{i,j}$  in  $\mathbf{L}$  is the similarity measured by dot products between elements  $i$  and  $j$ . Hence, we can verify that a DPP places higher probabilities on diverse sets because the more orthogonal the feature vectors are, the larger the volume parallelepiped spanned by the feature vector is. In this work, these feature embeddings represent class embeddings, which are derived using either a pre-trained protonet model or the model being trained as discussed in Sec. 3.3.

In a DPP, the cardinality of a sampled subset,  $|A|$ , is random in general. A  $k$ -DPP is an extension of the DPP proposed in the work of Kuhn et al. (2003), where the cardinality of subsets are fixed as  $k$  (i.e.,  $|A| = k$ ). In this work, we use  $k$ -DPPs as an off-the-shelf implementation to retrieve classes that represent a task used in the meta-learning step.

# 3.3 TASK SAMPLING

In this work, we experiment with eight distinct task samplers, each offering a different level of task diversity. To demonstrate the task samplers, we use a 2-way classification problem with a meta-batch size of 2 and denote each class with a unique alphabet from the Omniglot dataset.

Uniform Sampler This is the most widely used Sampler used in the setting of meta-learning. The Sampler gives equal probability to every task and is intuitively a random sampler. An illustration of this Sampler is shown in Figure 1.

No Diversity Task Sampler In this setting, we uniformly sample one set of the task at the beginning and propagate the same task across all batches and meta-batches. Note that repeating the same class over and over again does not simply repeat the same images/inputs as we episodically retrieve different images for each class. An illustration of this Sampler is shown in Figure 1.

No Diversity Batch Sampler In this setting, we uniformly sample one set of tasks for batch one and propagate the same tasks across all other batches. Furthermore, we shuffle these tasks to enforce that the model does not overfit. An illustration of this Sampler is shown in Figure 1.

No Diversity Tasks per Batch Sampler In this setting, we uniformly sample one set of tasks for a given batch and propagate the same tasks for all meta-batches. We then repeat this same principle for sampling the next batch. Furthermore, we shuffle these tasks to enforce that the model does not overfit. An illustration of this Sampler is shown in Figure 2.

Single Batch Uniform Sampler In this setting, we set the meta-batch size to one. This Sampler is intuitively the same as no diversity task per batch sampler, without the repetition of tasks. This

![](images/89f4892bf76d773ed61b1606962684d4a11c7da0f39362abd7bddcadedfcafc4.jpg)  
(a) Online Hard Task Mining sampler

![](images/ae3b6b7f4f4f4bb0861c47b3853f36ee0a1c07115b8b34f5fc9b967769eaeada.jpg)  
(b) Static DPP sampler

![](images/012009c9110f171caf4548f91e0f20c9f55e8003b5adb7ffd581147157944cf5.jpg)  
Figure 3: Illustration of (a) Online Hard Task Mining Sampler, (b) the Static DPP Sampler, and (c) the Dynamic DPP Sampler.  
(c) Dynamic DPP sampler

Sampler would be an ideal ablation study for the repetition of tasks in the meta-learning setting. An illustration of this Sampler is shown in Figure 2.

Online Hard Task Mining Sampler This setting is inspired by the works of Shrivastava et al. (2016) where they proposed OHEM, which yielded significant boosts in detection performance on benchmarks like PASCAL VOC 2007 and 2012. However, to reproduce OHEM for meta-learning, we only apply the OHEM sampler for half the meta-batch size and uniform sampler for the remaining half. This approach would allow us to involve many tasks and not restrict us to only known tasks. Furthermore, to avoid OHEM in the initial stages, we sample tasks with a uniform sampler until the buffer of tasks seen by the model becomes sufficiently big, say 50 in our case. An illustration of this Sampler is shown in Figure 3.

Static DPP Sampler Determinantal Point Processes (DPP) have been used for several machine learning problems such as the works of Kulesza & Taskar (2012). They have also been used in other problems such as the active learning settings in the works of Biryuk et al. (2019) and mini-batch sampling problems in the works of Zhang et al. (2019). These algorithms have also inspired other works in active learning in the batch mode setting, such as Ravi & Larochelle (2018). In this setting, we use DPP as an off-the-shelf implementation to sample tasks based on task embeddings. These task embeddings are generated using our pre-trained protonet model. The DPP instance is used to sample the most diverse tasks based on these embeddings and used for meta-learning. An illustration of this Sampler is shown in Figure 3.

Dynamic DPP Sampler In this setting, we extend the previous sDPP setting such that the model in training generates the task embeddings. The Sampler is motivated by the intuition that selecting the most diverse tasks for a given model will help learn better. Furthermore, to avoid DPP in the initial stages, we sample tasks with a uniform sampler until the model becomes sufficiently trained, say 500 batches in our case. An illustration of this Sampler is shown in Figure 3.

# 4 EXPERIMENTS

The experiment aims to answer the following questions: (a) How does task diversity affect meta-learning? (b) Do sophisticated samplers such as OHEM or DPP offer any significant boost in performance? (c) Are there any rule of thumb or general good practices when it comes to sampling tasks?

To make an exhaustive study on the effect of task diversity in meta-learning, we train on four datasets: Omniglot Lake et al. (2011), miniImagenet Ravi & Larochelle (2016), tieredImageNet Ren et al. (2018), and Meta-Dataset Triantafillou et al. (2019). With this selection of datasets, we cover both simple datasets, such as Omniglot and miniImageNet, as well as the most difficult ones, such as tieredImageNet and Meta-Dataset. We train three broad classes of meta-learning models on

![](images/4d52dbf3ff4b05509b000b7a2cb698a07dd17a66c1325e74f337aba30b98db33.jpg)  
Figure 4: Average accuracy on Omniglot 5-way 1-shot & miniImageNet 5-way 1-shot, with  $95\%$  confidence interval. All samplers are poorer than the Uniform Sampler and are statistically significant (with a p-value  $p = 0.05$ ). We use the symbol  $*$  to represent the instances where the results are not statistically significant and similar to the performance achieved by Uniform Sampler.

![](images/fbae08c92e342bedfe7a587d670c6f6d2f99814eb99de49e43e2747a512d70db.jpg)  
Figure 5: Average accuracy on Omniglot 20-way 1-shot, with a  $95\%$  confidence interval. We denote all samplers that are worse than the Uniform Sampler and are statistically significant (with a p-value  $p = 0.05$ ) with  $\nabla$ , and those that are significantly better than the Uniform Sampler with  $\triangle$ .

these datasets - Metric-based (i.e., Protonet, Matching Networks), Optimization-based (i.e., MAML, Reptile, and MetaOptNet), and Bayesian meta-learning models (i.e., CNAPs). More details about the datasets which were used in our experiments are discussed in App. A.1. More details about the models and their hyperparameters are discussed in App. A.2. We created a common pool of 1024 randomly sampled held-out tasks to test every algorithm in our experiments to make an accurate comparison. For all experiments, we assessed the statistical significance of our results based on a paired-difference t-test, with a p-value  $p = 0.05$ .

# 4.1 RESULTS

In this section, we present the results of our experiments. Figure 4 presents the performance of the six models on the Omniglot and miniImageNet under different task samplers in the 5-way 1-shot setting. Table 1 in the Appendix presents the same results with higher precision.

We also reproduce our experiments on the 20-way 1-shot setting on the Omniglot dataset to establish that these trends are shared across different settings. Figure 5 presents our performance of the models under this setting. Furthermore, the results on the 20-way 1-shot experiments are presented in Table 2 in the Appendix with higher precision. To further establish our findings, we also present our result on notoriously harder datasets such as tieredImageNet and Meta-Dataset. Figure 6 presents the performance of the models on the tieredImageNet. Again, Table 1 in the Appendix presents the same results with higher precision.

![](images/0cf6cf2436fae4ddaa3bcfdfd1f26e128d04d4312c628fcc78c74f49d00e2377.jpg)  
Figure 6: Average accuracy on tieredImageNet 5-way 1-shot, Meta-Dataset Traffic Sign 5-way 1-shot & Meta-Dataset MSCOCO 5-way 1-shot, with a  $95\%$  confidence interval. We denote all samplers that are worse than the Uniform Sampler and are statistically significant (with a p-value  $p = 0.05$ ) with  $\nabla$ , and those that are significantly better than the Uniform Sampler with  $\triangle$ .

Figure 6 presents the performance of the models on the Meta-Dataset Traffic Sign and Meta-Dataset MSCOCO datasets. We only present the results on Traffic Sign and MSCOCO of the Meta-Dataset, as these two sub-datasets are exclusively used for testing and are an accurate representation of the generalization power of the models when trained with different levels of task diversity. Other results on the Meta-Dataset are presented in Sec. 3. We empirically show that task diversity does not lead to any significant boost in the performance of the models. In the subsequent section, we discuss some of the other key findings from our work.

# 5 DISCUSSION

In this section, we discuss few empirical results from our experiments and shed light on some of the key findings from our research.

Poor performance by NDT Sampler The lowest performance is consistently obtained by the No Diversity Task Sampler, which is reasonable since the model only sees one task throughout its training. What is fascinating is that just one task is sufficient for the model to reach a reasonably decent performance in most cases. We do notice instances where NDT Sampler performs very well on a few sub-datasets of the Meta-Dataset. This can be explained by the fact that the model has only been trained for a single sub-dataset and has relatively less noise when compared to training on multiple sub-datasets.

Poor performance by Single Batch Uniform Sampler Consequently, the Single Batch Uniform Sampler does perform poorly on most datasets, including Omniglot, miniImageNet, and tieredImageNet. This is reasonable since the model is trained on a tiny pool of the dataset. However, we notice instances where the Sampler performs very well on a few sub-datasets of the Meta-Dataset.

We hypothesize that training on fewer samples keeps the model unaware of the inherent noise generated by training on diverse datasets and aids better performance in the case of Meta-Dataset.

Disparity between Single Batch Uniform and NDTB Sampler Another exciting result is the Disparity between Single Batch Uniform Sampler and No Diversity Tasks per Batch Sampler. As mentioned earlier, the only difference between the two samplers is that tasks are repeated in the latter. However, this repetition seems to offer a great deal of information to the model and allows the model to perform on par with the Uniform Sampler. It might be possible that the Single Batch Uniform Sampler obtains the performance observed by the No Diversity Tasks per Batch Sampler if trained for enough epochs. However, it would be safe to comment that the convergence of the model is significantly faster in the latter. Thus, repeating tasks might help speed up the convergence of the model when we have a fixed and handful amount of data. However, the same is not valid for models trained on Meta-Dataset. Although both samplers are trained over a similar pool of datasets, the NDTB sampler sees more data and might lead to more inherent noise generated by training on diverse datasets. This might explain why repeating tasks leads to lower performance in this case.

Disparity between s-DPP and d-DPP Sampler We also note that s-DPP and d-DPP samplers do not offer any boost in performance when compared to the regular Uniform Sampler. Furthermore, there seems to be a significant disparity between these two samplers. We believe that d-DPP, which computes the most diverse tasks at regular intervals, harms the model with the diverse tasks since we observe that the model's performance degrades over epochs. For example, consider the scenario where the model is trained on tasks involving dogs and tractors. This task is relatively easy to learn and would not require the model to fine-tune a great deal. However, during test time, suppose our task involves classifying cats and dogs; this would be a problem since the model has not learned the intricacies of the two classes. Thus, diversity seems to do more harm than good in this case. The best example of this is observed by Matching Networks in Omniglot 5-way 1-shot setting as shown in Figure-7, where each instance of diverse sampling harms the model significantly.

Limitation of samplers with DPP backbone Samplers such as s-DPP and d-DPP, which use DPP to sample diverse tasks, require task embeddings of every class in the dataset. Computing these task embeddings, although intensive, might be sustainable for small datasets such as Omniglot, miniImageNet, and tieredImageNet. However, computing the task embedding for every class of a dataset as large as the Meta-Dataset is nearly impossible due to the time and memory constraints. Hence, in our experiments on the Meta-Dataset, we do not run the model using the s-DPP and d-DPP Sampler and only report the findings from the remaining samplers.

OHTM Sampler offers no significant performance boost The OHTM Sampler is quite sophisticated since it regularly samples diverse tasks, as well as selects the most challenging tasks to improve the model. It is needless to say; the model requires more computational power and time than the Uniform Sampler. However, the OHTM Sampler offers no significant boost in performance when compared to the Uniform Sampler in the case of Omniglot, miniImageNet, and tieredImageNet. However, for Meta-Dataset, we notice that the OHTM Sampler sometimes leads to improved performance. This finding is quite puzzling since the Sampler works similarly across all datasets and does not address the inherent noise generated from training on diverse datasets such as the Meta-Dataset. The behavior of the OHTM Sampler on Meta-Datasets warrants further research.

Comparison between NDTB, NDB, and Uniform Sampler From our experiments, we also notice that the No Diversity Tasks per Batch Sampler and No Diversity Batch Sampler are pretty similar to the Uniform Sampler in terms of performance. This would suggest that the model trained on only a data fragment can perform similarly to that trained on the Uniform Sampler.

Abnormal run of matching networks d-DPP (20-way 1-shot) In our run on the matching networks with the d-DPP Sampler under the 20-way 1-shot setting, we ran across a peculiar error. The prototypes generated by the matching networks were sometimes not fit to be used by the d-DPP Sampler to sample 20 unique classes. The reason is that the rank of the matrix generated using the embeddings was lower than the required number of classes per task (i.e., 20). To create a workaround for this sole experiment, we chose to sample 5 diverse classes at a time and append them

to create the task. We hypothesize that the prototypes created by matching networks are unsuitable for downstream tasks and warrant further research regarding this behavior.

Poor performance of MAML on Meta-Dataset In our experiments on Meta-Dataset, we notice that MAML performs significantly worse than other models. Some of the reasons for this disparity of performance when compared to Triantafillou et al. (2019) have been discussed in detail in Appendix A.1. Furthermore, we hypothesize that the poor performance of MAML can be attributed to the way its adaptation process works: MAML learns within episode weights of the model before adapting to a set of new tasks via meta-update or outer loop update. This new task is again sampled from the pool of tasks and is a different set of data altogether. In most cases, where the model is trained on only one dataset, this intuition would make sense and lead to high-performing models. However, in the case of Meta-Dataset, where the new set of tasks might be of entirely different datasets or domains, this approach tends to do more harm to the model rather than aid. In the work of Triantafillou et al. (2019), the authors adapted MAML such that it focuses on learning the within-episode initialization  $\theta$  of the embedding network so that it can be rapidly adapted for a new task. This allows the model to learn from a variable number of ways and shots per episode.

Peculiar behavior with MetaOptNet model Compared to all other models, MetaOptNet seems to be immune to the effects of task diversity to a great extent. The convergence of the model seems to follow a general pattern and achieve similar performance across task distributions except for the Single Batch Uniform Sampler and No Diversity Task sampler. Furthermore, we do not observe the expected pattern of d-DPP Sampler, where the performance drops upon mining diverse tasks. We present the convergence graph of the MetaOptNet model on Omniglot 5-way 1-shot run in Figure 8 in the Appendix with an added smoothing factor of 1.

General Trend From our experiments, we notice that there are generally two classes of samplers: High Performing Samplers and Low Performing Samplers. The High Performing Samplers include No Diversity Batch, No Diversity Tasks per Batch, Uniform, OHTM, and s-DPP Sampler. The Low Performing Samplers include No Diversity Task, Single Batch Uniform, and d-DPP Sampler. This trend is shared across all datasets and models. There are some perturbations in ranking within the two classes, but the High Performing Samplers tend to perform better than the Low Performing Samplers.

# 6 CONCLUSION

In this paper, we have studied the effect of task diversity in meta-learning. We have empirically shown that task diversity does not lead to any significant boost in performance in meta-learning. Instead, limiting task diversity and repeating the same tasks over the training phase allows us to obtain similar performances to the Uniform Sampler without any significant adverse effects. Furthermore, We also show that sophisticated samplers such as OHEM or DPP samplers do not offer any significant boost in performance. In contradiction, we notice that increasing task diversity using the d-DPP Sampler hampers the performance of the meta-learning model. Our experiments using the NDTB and NDB empirically show that a model trained on even a tiny data fragment can perform similarly to a model trained using Uniform Sampler. This is a crucial finding since this questions the need to increase the support set pool to improve the models' performance. We believe that the experiments we performed lay the roadwork to further research for the effect of task diversity domain in meta-learning and lay some groundwork and rules of thumb for task sampling for meta-learning.

# REPRODUCABILITY STATEMENT

In this paper, we work with four different datasets - Omniglot, miniImageNet, tieredImageNet and Meta-Dataset. Additional details about setting up these datasets is available in Appendix A.1. Furthermore, we experiment with six different models - MAML, Reptile, Protonet, Matching Networks, MetaOptNet, and CNAPs. All these models were run after reproducing from their open-source codes. Additional details about setting up these models are available in Appendix A.2. Our source code is made available under the supplementary materials for additional reference.

# REFERENCES

Erdem Bıyık, Kenneth Wang, Nima Anari, and Dorsa Sadigh. Batch active learning using determinantal point processes. arXiv preprint arXiv:1906.07975, 2019.  
Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3606-3613, 2014.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pp. 1126-1135. PMLR, 2017.  
Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and SM Ali Eslami. Conditional neural processes. In International Conference on Machine Learning, pp. 1704-1713. PMLR, 2018.  
Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing, and Christian Igel. Detection of traffic signs in real-world images: The german traffic sign detection benchmark. In *The 2013 international joint conference on neural networks (IJCNN)*, pp. 1-8. IEEE, 2013.  
Kyle Hsu, Sergey Levine, and Chelsea Finn. Unsupervised learning via meta-learning. arXiv preprint arXiv:1810.02334, 2018.  
Jonas Jongejan, Henry Rowley, Takashi Kawashima, Jongmin Kim, and Nick Fox-Gieg. The quick, draw!-ai experiment. Mount View, CA, accessed Feb, 17(2018):4, 2016.  
Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, et al. Siamese neural networks for one-shot image recognition. In ICML deep learning workshop, volume 2. Lille, 2015.  
Alexandre Kuhn, Ad Aertsen, and Stefan Rotter. Higher-order statistics of input ensembles and the response of simple model neurons. Neural computation, 15(1):67-101, 2003.  
Alex Kulesza and Ben Taskar. Determinantal point processes for machine learning. arXiv preprint arXiv:1207.6083, 2012.  
Brenden Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua Tenenbaum. One shot learning of simple visual concepts. In Proceedings of the annual meeting of the cognitive science society, volume 33, 2011.  
Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10657-10665, 2019.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740-755. Springer, 2014.  
Chenghao Liu, Zhihao Wang, Doyen Sahoo, Yuan Fang, Kun Zhang, and Steven CH Hoi. Adaptive task sampling for meta-learning. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XVIII 16, pp. 752-769. Springer, 2020a.  
Jialin Liu, Fei Chao, and Chih-Min Lin. Task augmentation by rotating for meta-learning. arXiv preprint arXiv:2003.00804, 2020b.  
Odile Macchi. The coincidence approach to stochastic point processes. Advances in Applied Probability, 7(1):83-122, 1975.  
Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.  
Tsendsuren Munkhdalai and Hong Yu. Meta networks. In International Conference on Machine Learning, pp. 2554-2563. PMLR, 2017.

Alex Nichol, Joshua Achiam, and John Schulman. On first-order meta-learning algorithms. arXiv preprint arXiv:1803.02999, 2018.  
Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pp. 722-729. IEEE, 2008.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. 2016.  
Sachin Ravi and Hugo Larochelle. Meta-learning for batch mode active learning. 2018.  
Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B Tenenbaum, Hugo Larochelle, and Richard S Zemel. Meta-learning for semi-supervised few-shot classification. arXiv preprint arXiv:1803.00676, 2018.  
James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E Turner. Fast and flexible multi-task classification using conditional neural adaptive processes. Advances in Neural Information Processing Systems, 32:7959-7970, 2019.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211-252, 2015.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International conference on machine learning, pp. 1842-1850. PMLR, 2016.  
Brigit Schroeder and Yin Cui. Fgvcx fungi classification challenge 2018. github.com/visipedia/fgvcx_fungi_comp, 9, 2018.  
Amrith Setlur, Oscar Li, and Virginia Smith. Is support set diversity necessary for meta-learning? arXiv preprint arXiv:2011.14048, 2020.  
Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 761-769, 2016.  
Jake Snell, Kevin Swersky, and Richard S Zemel. Prototypical networks for few-shot learning. arXiv preprint arXiv:1703.05175, 2017.  
Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1199-1208, 2018.  
Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, et al. Meta-dataset: A dataset of datasets for learning to learn from few examples. arXiv preprint arXiv:1903.03096, 2019.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in neural information processing systems, 29:3630-3638, 2016.  
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.  
Mingzhang Yin, George Tucker, Mingyuan Zhou, Sergey Levine, and Chelsea Finn. Meta-learning without memorization. arXiv preprint arXiv:1912.03820, 2019.  
Jaesik Yoon, Taesup Kim, Ousmane Dia, Sungwoong Kim, Yoshua Bengio, and Sungjin Ahn. Bayesian model-agnostic meta-learning. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 7343–7353, 2018.  
Cheng Zhang, Cengiz Öztireli, Stephan Mandt, and Giampiero Salvi. Active mini-batch sampling using repulsive point processes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5741-5748, 2019.
