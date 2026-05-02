# WHEN DO CURRICULA WORK?

Anonymous authors

Paper under double-blind review

# ABSTRACT

Inspired by human learning, researchers have proposed ordering examples during training based on their difficulty. Both curriculum learning, exposing a network to easier examples early in training, and anti-curriculum learning, showing the most difficult examples first, have been suggested as improvements to the standard i.i.d. training. In this work, we set out to investigate the relative benefits of ordered learning. We first investigate the implicit curricula resulting from architectural and optimization bias and find that samples are learned in a highly consistent order. Next, to quantify the benefit of explicit curricula, we conduct extensive experiments over thousands of orderings spanning three kinds of learning: curriculum, anti-curriculum, and random-curriculum – in which the size of the training dataset is dynamically increased over time, but the examples are randomly ordered. We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. Inspired by common use cases of curriculum learning in practice, we investigate the role of limited training time budget and noisy data in the success of curriculum learning. Our experiments demonstrate that curriculum, but not anti-curriculum can indeed improve the performance either with limited training time budget or in existence of noisy data.

# 1 INTRODUCTION

Inspired by the importance of properly ordering information when teaching humans (Avrahami et al., 1997), curriculum learning (CL) proposes training models by presenting easier examples earlier during training (Elman, 1993; Sanger, 1994; Bengio et al., 2009). Previous empirical studies have shown instances where curriculum learning can improve convergence speed and/or generalization in domains such as natural language processing (Cirik et al., 2016; Platanios et al., 2019), computer vision (Pentina et al., 2015; Sarafianos et al., 2017; Guo et al., 2018; Wang et al., 2019), and neural evolutionary computing (Zaremba & Sutskever, 2014).

In contrast to curriculum learning, anti-curriculum learning selects the most difficult examples first and gradually exposes the model to easier ones. Though counter-intuitive, empirical experiments have suggested that anti-curriculum learning can be as good as or better than curriculum learning in certain scenarios (Kocmi & Bojar, 2017; Zhang et al., 2018; 2019b). This is in tension with experiments in other contexts, however, which demonstrate that anti-curricula under perform standard or curriculum training (Bengio et al., 2009; Hacohen & Weinshall, 2019).

As explained above, empirical observations on curricula appear to be in conflict. Moreover, despite a rich literature (see Section A), no ordered learning method is known to improve consistently over others across contexts, and curricula have not been widely adopted in machine learning. This may suggest ruling out curricula as beneficial for learning. In certain contexts, however, for large-scale text models such as GPT-3 (Brown et al., 2020) and T5 (Raffel et al., 2019), curricula are standard practice. These contradicting observations contribute to a confusing picture about the usefulness of curricula.

This work is an attempt to improve our understanding of curricula systematically. We start by asking a very fundamental question about a phenomenon that we call implicit curricula. Are examples learned in a consistent order across different runs, architectures, and tasks? If such a robust notion exists, is it possible to change the order in which the examples are learned by presenting them in a

![](images/09437107dbad7b94c94827d94a0d7ad67327e8ad637446189f605feadaae8565.jpg)  
Figure 1: Curricula help for time limited or noisy training, but not standard training. Each point represents an independent learning ordering on CIFAR100 and is a mean over three independent runs with the same hyper-parameters. Color represents the type of learning, from bottom to top, are standard i.i.d. training (grey), curriculum (blue), anti-curriculum (purple), and random curriculum (green). The solid orange line is the best test accuracy for standard i.i.d. training. The left, middle and right plots represent long-time, short-time, and noisy training. We find that for the original dataset and learning constraints there are no statistically significant benefits from anti, random, or curriculum learning (left). We find that for training with a limited time budget (center) or with noisy data (right) curriculum learning can be beneficial.

different order? The answer to this question determines if there exists a robust notion of example difficulty that could be used to influence training.

We then look into different ways of associating difficulty to examples using scoring functions and a variety of schedules known as pacing functions for introducing examples to the training procedure. We investigate if any of these choices can improve over the standard full-data i.i.d. training procedure commonly used in machine learning. Inspired by the success of curriculum learning in large scale training scenarios, we train in settings intended to emulate these large scale settings. In particular, we study the effect of curricula when training with a training time budget and training in the presence of noise.

Contributions In this paper, we systematically design and run extensive experiments to gain a better understanding of curricula. We train over 20,000 models covering a wide range of choices in designing curricula and arrive at the following conclusions:

- Implicit Curricula: Examples are learned in a consistent order (Section 2). We show that the order in which examples are learned is consistent across runs, similar training methods, and similar architectures. Furthermore, we show that it is possible to change this order by changing the order in which examples are presented during training. Finally, we establish that well-known notions of sample difficulty are highly correlated with each other.  
- Curricula achieve (almost) no improvement in the standard setting (Section 4). We show curriculum learning, random, and anti-curriculum learning perform almost equally well in the standard setting. Furthermore, we establish that using similar techniques to remove examples from the training set (as opposed to introducing them) also does not help.  
- Curriculum learning improves over standard training when training time is limited (Section 5). Imitating the large data regime, where training for multiple epochs is not feasible, we limit the number of iterations in the training algorithm and compare curriculum, random and anti-curriculum ordering against standard training. Our experiments reveal a clear advantage of curriculum learning over other methods.  
- Curricula improves over standard training in noisy regime (Section 5). Finally, we mimic noisy data by adding label noise. Similar to Jiang et al. (2018); Saxena et al. (2019); Guo et al. (2018), our experiments indicate that curriculum learning has a clear advantage over other curricula and standard training.

# 1.1 RELATED WORK

Bengio et al. (2009) is perhaps the most prominent work on curriculum learning where the "difficulty" of examples is determined by the loss value of a pre-trained model.

![](images/cd66408720b9534a4b7b246899928748680afe6152436f0a8ee9e197228fa2e1.jpg)  
Figure 2: Implicit Curricula: Images are learned in a similar order for similar architectures and training methods. (Left) Epoch at which each image is learned across 142 different architectures and optimization procedures. Each row is a CIFAR10 image ordered by its average learned epoch. The columns from left to right, are fully-connected (FC) nets, VGG nets (VGG11 and VGG19), and nets with Batch-Norm (Ioffe & Szegedy, 2015) including ResNet18, ResNet50, WideResNet28-10, WideResNet48-10 DenseNet121, Effie-. cientNet B0, VGG11-BN and VGG19-BN. (Right) The Spearman correlation matrix shows high correlation between orderings within architecture families.

![](images/f48c8d1d073ae6681a0c0931b344fe84eebcf5292258f5d95b41268f29531ced.jpg)

Toneva et al. (2019) instead suggested using the first iteration in which an example is learned and remains learned after that. Finally, Jiang et al. (2020b) has proposed using a consistency score (c-score) calculated based on the consistency of a model in correctly predicting a particular example's label trained on i.i.d. draws of the training set. When studying curriculum learning, we look into all of the above-suggested measures of sample difficulty. We further follow Hacohen & Weinshall (2019) in defining the notion of pacing function and use it to schedule how examples are introduced to the training procedure. However, we look into a much more comprehensive set of pacing functions and different tasks in this work.

Please see Section A for a comprehensive review of the literature on curricula.

# 2 IMPLICIT CURRICULA

Curriculum learning is predicated on the expectation that we can adjust the course of learning by controlling the order of training examples. Despite the intuitive appeal, the connection between the order in which examples are shown to a network during training and the order in which a network learns to classify these examples correctly is not apriori obvious. To better understand this connection, we first study the order in which a network learns examples under traditional stochastic gradient descent with i.i.d. data sampling. We refer to this ordering – which results from the choice of architecture and optimization procedure – as an implicit curriculum.

To quantify this ordering we define the learned iteration of a sample for a given model as the epoch for which the model correctly predicts the sample for that and all subsequent epochs. Explicitly,  $\min_{t^{*}}\{t^{*}|\hat{y}_{\mathbf{w}}(t)_{i} = y_{i},\forall t^{*}\leq t\leq T\}$  where  $y_{i}$  and  $\hat{y}_{\mathbf{w}}(t)_i$  are the correct label and the predictions of the model for  $i$ -th data point (see the detailed mathematical description in Section B).

We study a wide range of model families including fully-connected networks, VGG (Simonyan & Zisserman, 2014), ResNet (He et al., 2016), Wide-ResNet (Zagoruyko & Komodakis, 2016), DenseNet (Huang et al., 2017) and EfficientNet (Tan & Le, 2019) models with different optimization algorithms such as Adam Kingma & Ba (2014) and SGD with momentum (see Section B for details). The results in Figure 2 for CIFAR10 Krizhevsky et al. (2009) show that the implicit curricula are broadly consistent within model families. In particular, the ordering in which images are learned within convolutional networks is much more consistent than between convolutional networks and fully connected networks, $^{1}$  and the learned ordering within each sub-type of CNN is even more uniform. The robustness of this ordering, at least within model types, allows us to talk with less ambiguity about the difficulty of a given image without worrying that the notion of difficulty is highly model-dependent.

![](images/7c947e2a510a83d7ac8acee395ad6acc612fb3c12c58ca268cb3b43374caedf5.jpg)  
Figure 3: Scoring functions show a high correlation for the standard training, but perceived difficulty depends on the training order. (Left) Six scoring functions computed on CIFAR10 using the standard i.i.d. training algorithms. Columns from left to right show VGG-11 loss, ResNet-18 loss, VGG-11 iteration learned, ResNet-18 iteration learned, c-score, and a loss based c-score. Here the order given by VGG-11 loss or ResNet-18 loss uses the recorded loss at epoch 10. (Right) Loss-based difficulty when performing non-standard training from left to right columns are: c-score baseline, curriculum-based training, random-training, and anticurriculum training. The last three used the same pacing function - step with  $a = 0.8$  and  $b = 0.2$ . (Center) The Spearman's rank correlation is high between all scoring functions computed from ordinary or curriculum training, but lower for random training and significantly lower for anti-curriculum, indicating that the three orderings lead to networks learning samples in different orders.

![](images/7bd33d7d398365dbb9a3b6d096ea0975b9617460204b0b449773cc697a3aba2f.jpg)

![](images/625820c296254847bb514b7c1e6da3037ec9c1f25ec81bddfd3849bf0b468081.jpg)

We will see in the next section (and Figure 3) that, as expected, the choice of explicit curriculum can alter the order in which a network learns examples. The most dramatic manifestation of this is anti-curriculum learning where showing the network images in the reverse order indeed causes the network to learn more difficult images first. Next, we introduce the class of curricula we will consider for the remainder of the paper.

# 3 PUTTING CURRICULA THROUGH THEIR PACES

Many different approaches have been taken to implement curricula in machine learning. Here we focus on a particular widely used paradigm introduced in Bengio et al. (2009) and used in Hacohen & Weinshall (2019). In this setup, a curriculum is defined by specifying three ingredients,

- The scoring function: The scoring function is a map from an input example,  $\mathbf{x}$ , to a numerical score,  $s(\mathbf{x}) \in \mathbb{R}$ . This score is typically intended to correspond to a notion of difficulty, where a higher score corresponds to a more difficult example.  
- The pacing function: The pacing function  $g(t)$  specifies the size of the training data-set used at each step,  $t$ . The training set at step  $t$  consists of the  $g(t)$  lowest scored examples. Training batches are then sampled uniformly from this set.  
- The order: Additionally we specify an order of either curriculum – ordering examples from lowest score to highest score, anti-curriculum – ordering examples from highest score to lowest, or random. Though technically redundant with redefining the scoring function, we maintain the convention that the score is ordered from easiest to hardest.

This procedure is summarized in Algorithm 1. It is worth emphasizing that due to the pacing function, using a random ordering in Algorithm 1 is not the same as traditional i.i.d. training on the full training dataset, but rather corresponds to i.i.d. training on a training dataset with dynamic size.

# Algorithm 1 (Random-/Anti-) Curriculum learning with pacing and scoring functions

1: Input: Initial weights  $\mathbf{w}^0$ , training set  $\{\mathbf{x}_1, \ldots, \mathbf{x}_N\}$ , pacing function  $g:[T] \to [N]$ , scoring function  $s:[N] \to \mathbb{R}$ , order  $o \in \{\text{"ascending", "descending", "random"}\}$ .  
2:  $(\mathbf{x}_1,\dots ,\mathbf{x}_N)\gets \mathrm{sort}(\{\mathbf{x}_1,\dots ,\mathbf{x}_N\} ,s,o)$  
3: for  $t = 1, \dots, T$  do  
4:  $\mathbf{w}^{(t)}\gets$  train-one-epoch  $(\mathbf{w}^{(t - 1)},\{\mathbf{x}_1,\dots ,\mathbf{x}_{g(t)}\})$  
5: end for

We stress that the scoring and pacing function paradigm for curriculum learning is inherently limited. In this setup, the scoring function is computed before training over all of the data and thus the algorithm cannot implement a self-paced and training-dependent curriculum as has been considered in Kumar et al. (2010); Jiang et al. (2015); Platanios et al. (2019). Additionally, the dynamic training dataset is built by including all examples within a fixed score window (from lowest score up in curricula and highest score down in anti-curricula) and does not accommodate more flexible subsets. Furthermore, the form of curriculum discussed here only involves ordering examples from a fixed training dataset, rather than more drastic modifications of the training procedure, such as gradually increasing image resolution (Vogelsang et al., 2018) or the classes (Weinshall et al., 2018). Nonetheless, it is commonly studied and serves as a useful framework and control study to empirically investigate the relative benefits of training orderings.

Next, we describe scoring and pacing functions that will be used in our empirical investigation.

# 3.1 SCORING FUNCTIONS: DIFFICULTY SCORES ARE BROADLY CONSISTENT

In this section, we investigate three families of scoring functions. As discussed above, we define the scoring function  $s(\boldsymbol{x}, y) \in \mathbb{R}$  to return a measure of an example's difficulty. We say that an example  $\boldsymbol{x}_j$  is more difficult than an example  $\boldsymbol{x}_i$  if  $s(\boldsymbol{x}_j, y_j) > s(\boldsymbol{x}_i, y_i)$ . In this work, we consider three scoring functions:

- Loss function. In this case samples are scored using the real-valued loss of a reference model that is trained on the same training data, e.g. given a trained model  $f_{\mathbf{w}}: \mathcal{X} \to \mathcal{Y}$ , we set  $s(\boldsymbol{x}_i, y_i) = \ell(f_{\mathbf{w}}(\boldsymbol{x}_i), y_i)$ .  
- Learned epoch/iteration. This metric has been introduced in the beginning of this section. We let  $s(\boldsymbol{x}_i, y_i) = \min_{t^*} \{ t^* |\hat{y}_{\mathbf{w}}(t)_i = y_i, \forall t^* \leq t \leq T \}$  (see Algorithm 3).  
- Estimated c-score. c-score (Jiang et al., 2020b) is designed to capture the consistency of a reference model correctly predicting a particular example's label when trained on independent i.i.d. draws of a fixed sized dataset not containing that example. Formally,  $s(\boldsymbol{x}_i, y_i) = \mathbb{E}_{D \sim \hat{\mathcal{D}} \setminus \{(\boldsymbol{x}_i, y_i)\}} [\mathbb{P}(\hat{y}_{\mathbf{w}, i} = y_i | D)]$  where  $D$ , with  $|D| = n$ , is a training data set sampled from the data pool without the instance  $(\boldsymbol{x}_i, y_i)$  and  $\hat{y}_{\mathbf{w}, i}$  is the reference model prediction. We also consider a loss based c-score,  $s(\boldsymbol{x}_i, y_i) = \mathbb{E}_{D \sim \hat{\mathcal{D}} \setminus \{(\boldsymbol{x}_i, y_i)\}}^{r} [\ell(\boldsymbol{x}_i, y_i) | D]$ , where  $\ell$  is the loss function. The pseudo-code is described in Algorithm 4.

To better understand these three scoring functions, we evaluate their consistency in CIFAR10 images. In particular, for VGG-11 and ResNet-18, we averaged the learned epoch over five random seeds and recorded the loss at epochs 2, 10, 30, 60, 90, 120, 150, 180, 200. We also compute the c-score using multiple independently trained ResNet-18 models and compare it to the reference c-scores reported in (Jiang et al., 2020b). The main results of these evaluations are reported in the left and middle panels of Figure 3. We see that all six scores presented have high Spearman's rank correlation suggesting a consistent notion of difficulty across these three scores and two models. For this reason, we use only the c-score scoring function in the remainder of this paper.

Given these pacing functions, we can now ask if the explicit curricula enforced by them can change the order in which examples are learned. The right panel in Figure 3 indicates what is the impact of using curriculum, anti-curriculum or random ordering on the order in which examples are learned. All curricula use the step pacing functions with  $a = 0.8$  and  $b = 0.2$ . The result indicates that curriculum and random ordering do not change the order compared to c-score but anti-curriculum could indeed force the model to learn more difficult examples sooner. This is also demonstrated clearly in the Spearman's correlation of anti-curriculum with other methods shown in the middle panel of Figure 3.

Additional model seeds and training times are presented in Figure 9 and 10 in the appendix. Among all cases, we found one notable outlier. For ResNet-18 after 180 epochs, the training loss no longer correlates with the other scoring functions. We speculate that this is perhaps due to the model memorizing the training data and achieving near-zero loss on all training images.

![](images/20b391d99ceb4c7ebbead9e9a9fff431d835c8445e584dd18659108c02522aed.jpg)  
Figure 5: Curricula provide little benefit for standard learning. (i,ii) Bar plots showing the best mean accuracy, for curriculum (blue), anti-curriculum (purple), random-curriculum (green), and standard i.i.d. training (grey) with three ways of calculating the standard training baseline for CIFAR10 (i) and CIFAR100 (ii). (iii,iv) Means over three seeds for all 540 strategies, and 30 means over three standard training runs (grey) for CIFAR10 (iii) and CIFAR100 (iv). The x-axis is the test accuracy. Marker shape signifies the pacing function family. Black solid and dashed lines give the mean and standard deviation over standard training runs. The solid orange line is the standard2 baseline test accuracy. We observe no statistically significant improvement from curriculum, anti-curriculum, or random training.

![](images/71dcda38c76311586846752660ee797ac1031e0b01a92a424f517ace5a912114.jpg)

# 3.2 PACING FUNCTIONS: FORCING EXPLICIT CURRICULA

The pacing function determines the size of training data to be used at epoch  $t$  (see Algorithm 1). We consider six function families: logarithmic, exponential, step, linear, quadratic, and root. Table 2 illustrates the pacing functions used for our experiments which is parameterized by  $(a,b)$ . Here  $a$  is the fraction of training needed for the pacing function to reach the size of the full training set, and  $b$  is the fraction of the training set used at the start of training, thus any pacing function with  $a = 0$  or  $b = 1$  is equivalent to standard training. We denote the full training set size by  $N$  and the total number of training steps by  $T$ . Explicit expressions for the pacing functions we use,  $g_{(a,b)}(t)$ , and examples are shown in Figure 4.

In order to cover many possible choices of pacing function, in all remaining experiments, we select  $b \in \{0.0025, 0.1, 0.2, 0.4, 0.8\}$  and  $a \in \{0.01, 0.1, 0.2, 0.4, 0.8, 1.6\}$  for our empirical study. Pacing functions with these parameters are plotted in Figure 11.

<table><tr><td>Name</td><td>Expression g(a,b)(t)</td></tr><tr><td>log</td><td>Nb + N(1 - b) (1 + .1 log (t/aT + e-10))</td></tr><tr><td>exp</td><td>Nb + N(1-b)/e10-1 (exp (10t/aT) - 1))</td></tr><tr><td>step</td><td>Nb + N[x/aT]</td></tr><tr><td>polynomial</td><td>Nb + N(1-b)/(aT)p t^p - p = 1/2, 1, 2</td></tr></table>

![](images/2d22882a9e81864f8a35447429360d6d61555c50fd9ecdb4b428776241be374a.jpg)  
Figure 4: Pacing functions (Left) pacing function definitions for the six families of pacing functions used throughout. (Right) example, pacing function curves from each family. The parameter  $a$  determines the fraction of training time until all data is used. The parameter  $b$  sets the initial fraction of the data used.

# 4 PACING FUNCTIONS GIVE MARGINAL BENEFIT, CURRICULA GIVE NONE

Equipped with the ingredients described in the previous sections, we investigated the relative benefits of (random-/anti-) curriculum learning. As discussed above, we used the c-score as our scoring function and performed a hyperparameter sweep over the 180 pacing functions and three orderings described in Section 3. We replicated this procedure over three random seeds and two standard benchmark datasets, CIAFR10 and CIFAR100. For each experiment in this sweep, we trained a ResNet-50 model for 39100 total training steps² and recorded the best test accuracy achieved during this training. The results of these runs are shown in Figure 5.

![](images/de22b323a8d73173d9222e60b90892f7bb29cdd142f982eefb09bcb7badc04a2.jpg)  
Figure 6: Curriculum-learning helps when training with a limited time budget. CIFAR100 performance when training with 19550, 3910 and 391 total steps. Curriculum learning provides a robust benefit when training for 3910 and 391 steps. See Figure 5 for additional plotting details.

To understand whether (anti-) curriculum, or random learning provides any benefit over ordinary training, we also ran 90 standard training runs. From these 90 runs, we created three baselines. The standard1 baseline is the mean performance over all 90 runs. The standard2 baseline splits the 90 runs into 30 groups of 3; the mean is taken over each group of 3 and the maximum is taken over the 30 means. This setup is intended as a stand in for the hyperparameter search of the 30 choices of pacing parameters  $(a,b)$  each with three seeds. Lastly, standard3 is a simulated result. In the simulation, 180 points are drawn from three i.i.d Gaussians with mean and variance estimated from three groups of 30 runs. In total, we simulate three clusters, representing the results from three random seeds. From each cluster, we sample 180 points and average the results over the three clusters. We take the max value over the 180 mean values, thus simulating the full set of (random/anti-) curriculum learning experiments. We report the median over 10000 simulation.

Marginal value of ordered learning. When comparing our full search of pacing functions to the naive mean, standard1, all three sample orderings have many pacing functions that outperform the baseline. We find, however, that this is consistent with being an artifact of the large search space. When comparing the performance of all three methods to a less crippled baseline, which considers the massive hyperparameter sweep, we find that none of the pacing functions or orderings statistically significantly outperforms additional sampling of ordinary SGD training runs.

Perhaps most striking is that performance shows no dependence on the three different orderings (and thus scoring function). For example, in the CIFAR10 runs, the best mean accuracy is achieved via random ordering, while in CIFAR100, the best single run has a random ordering. This suggests that for these benchmark datasets, with standard training limitations, any marginal benefit of ordered learning is due entirely to the dynamic dataset size (pacing function).

# 5 CURRICULA FOR SHORT-TIME TRAINING AND NOISY DATA

In the previous section, we found little evidence for statistically significant benefits from curricula or pacing. This observation is consistent with the fact that curricula have not become a standard part of supervised image classification. In other contexts, however, curricula are standard. Notably, in practice, many large scale text models are trained using curricula (Brown et al., 2020; Raffel et al., 2019). These models are typically trained in a data-rich setting where multiple epochs of training is not feasible. Furthermore, the data for training is far less clean than standard image benchmarks.

To emulate these characteristics and investigate whether curricula are beneficial in such contexts, we applied the same empirical analysis used in Section 4 to training with a limited computational budget and training on noisy data. It should be noted that the benefits of curricula for noisy data have been studied previously, for example, in (Jiang et al., 2018; Saxena et al., 2019; Guo et al., 2018).

Limited training time budget. For shorter time training, we follow the same experiment setup described in Section 4 but modify the total number of steps,  $T$ . Instead of using 39100 steps, we consider training with  $T = 392$ , 3910 or 19550 steps. We use cosine learning rate decay, decreasing monotonically to zero over the  $T$  steps. The results are given in Figure 6, and in supplementary Figures 14, 16, and 17.

We see that curricula are still no better than standard training for 19550 iterations. However, when drastically limiting the training time to 3910 or 391 steps, curriculum-learning, but not anti-

![](images/98d8d09881df20cec77e06b02ffdac62445e627584a2b0499c754169d1775210.jpg)  
Figure 7: Curriculum-learning helps when training with noisy labels. Performance on CIFAR100 with the addition of  $20\%$ ,  $40\%$  and  $60\%$  label noise shows robust benefits when using curricula. See Figure 5 for additional plotting details.

![](images/65f41c1b8909380f3b74844263aa2eb5a6d58486d87668eb94d602369b764379.jpg)  
Figure 8: Top performing pacing functions for limited time and noisy training. Top performing pacing functions from the six families considered for CIFAR100 (from left to right) finite time training with 3910 and 391 steps,  $20\%$  label noise and  $60\%$  label noise.

![](images/64491c56e2e349cca6927249eb092d783daf315e626e28bd195bd031a989107c.jpg)

curriculum can indeed improve performance, with increasing performance gains as the training time budget is decreased. Surprisingly, at the smallest time budget, 391 steps, we also see random-learning consistently outperforms the baseline, suggesting that the dynamic pacing function on its own helps improve performance. We depict the best pacing functions for 3910 and 391 steps in Figure 8. At 391 steps, we see a trend to start and maintain relatively small datasets for a large fraction of training.

Data with noisy labels. To study curricula in the context of noisy data, we adopted a common procedure of generating artificial label noise by randomly permuting the labels (Zhang et al., 2017; Saxena et al., 2019; Jiang et al., 2020b). We considered CIFAR100 datasets with  $20\%$ ,  $40\%$ , and  $60\%$  label noise and otherwise use the same experimental setup described in Section 4. As the training data has been modified, we must recompute the c-score. The results are shown in Figure 13.

Equipped with the new ordering, we repeat the same set of experiments and obtain the results shown in Figure 7 and supplementary Figure 18. Figure 7 shows that curriculum learning outperforms other methods by a large margin across all noise levels considered. The best pacing function in each family is shown in Figure 8. We see that the best overall pacing function for both  $20\%$  and  $60\%$  noise is the step pacing function corresponding to simply ignoring all noisy data during training. For  $40\%$  noisy labels, this strategy was not contained in our sweep of  $a$  and  $b$  values.

# DISCUSSION

In this work, we established the phenomena of implicit curricula, which suggests that examples are learned in a consistent order. We further ran extensive experiments and showed that while curricula are not helpful in standard training settings, easy-to-difficult ordering can be beneficial when training with a limited time budget or noisy labels. We acknowledge that despite training more than 20,000 models on CIFAR-like datasets, our empirical investigation is still limited by the computing budget. That limitation forced us to choose between diversity in the choice of orderings and pacing functions as opposed to diversity in the tasks type, task scale, and model scale. Our strategic choice of limiting the datasets to CIFAR-10 and CIFAR-100 allowed us to focus on a wide range of curricula and pacing functions, which was very informative. As a result, our general statements are more robust with respect to the choice of curricula – which is the main object of study in this paper – but one should be careful about generalizing them to tasks/training regimes that are not similar to those included in this work.

# REFERENCES

Devansh Arpit, Stanislaw Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxin-der S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, et al. A closer look at memorization in deep networks. In International Conference on Machine Learning, pp. 233-242, 2017.  
Judith Avrahami, Yaakov Kareev, Yonatan Bogot, Ruth Caspi, Salomka Dunaevsky, and Sharon Lerner. Teaching by examples: Implications for the process of category acquisition. The Quarterly Journal of Experimental Psychology Section A, 50(3):586-606, 1997.  
Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of International Conference on Machine Learning, 2009.  
Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. Siam Review, 60(2):223-311, 2018.  
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
Jonathon Byrd and Zachary Lipton. What is the effect of importance weighting in deep learning? volume 97 of Proceedings of Machine Learning Research, pp. 872-881, Long Beach, California, USA, 09-15 Jun 2019. PMLR.  
Satrajit Chatterjee. Learning and memorization. In International Conference on Machine Learning, pp. 755-763, 2018.  
Volkan Cirik, Eduard Hovy, and Louis-Philippe Morency. Visualizing and understanding curriculum learning for long short-term memory networks. arXiv preprint arXiv:1611.06204, 2016.  
Jeffrey L Elman. Learning and development in neural networks: The importance of starting small. Cognition, 48(1):71-99, 1993.  
Yanbo Fan, Siwei Lyu, Yiming Ying, and Baogang Hu. Learning with average top-k loss. In Advances in neural information processing systems, pp. 497-505, 2017.  
Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128-135, 1999.  
Rong Ge, Runzhe Wang, and Haoyu Zhao. Mildly overparametrized neural nets can memorize training data efficiently. arXiv preprint arXiv:1909.11837, 2019.  
Ian J Goodfellow, Mehdi Mirza, Aaron Courville Da Xiao, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks. In *In Proceedings of International Conference on Learning Representations* (ICLR. Citeseer, 2014).  
Priya Goyal, Piotr Dólár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Alex Graves, Marc G Bellemare, Jacob Menick, Rémi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. In International Conference on Machine Learning, pp. 1311-1320, 2017.  
Arthur Gretton, Alex Smola, Jiayuan Huang, Marcel Schmittfull, Karsten Borgwardt, and Bernhard Scholkopf. Covariate shift by kernel mean matching. *Dataset shift in machine learning*, 3(4):5, 2009.  
Jindong Gu and Volker Tresp. Neural network memorization dissection. arXiv preprint arXiv:1911.09537, 2019.  
Sheng Guo, Weilin Huang, Haozhi Zhang, Chenfan Zhuang, Dengke Dong, Matthew R. Scott, and Dinglong Huang. Curriculumnet: Weakly supervised learning from large-scale web images. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

Guy Hacohen and Daphna Weinshall. On the power of curriculum learning in training deep networks. ICML, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708, 2017.  
Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5901-5910, 2020.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
Angela H Jiang, Daniel L-K Wong, Giulio Zhou, David G Andersen, Jeffrey Dean, Gregory R Ganger, Gauri Joshi, Michael Kaminsky, Michael Kozuch, Zachary C Lipton, et al. Accelerating deep learning by focusing on the biggest losers. arXiv preprint arXiv:1910.00762, 2019.  
Lu Jiang, Deyu Meng, Qian Zhao, Shiguang Shan, and Alexander G Hauptmann. Self-paced curriculum learning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pp. 2694-2700. AAAI Press, 2015.  
Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In International Conference on Machine Learning, pp. 2304-2313, 2018.  
Lu Jiang, Di Huang, Mason Liu, and Weilong Yang. Beyond synthetic noise: Deep learning on controlled noisy labels. In ICML, 2020a. URL https://arxiv.org/abs/1911.09781.  
Ziheng Jiang, Chiyuan Zhang, Kunal Talwar, and Michael C Mozer. Exploring the memorization-generalization continuum in deep learning. arXiv preprint arXiv:2002.03206, 2020b.  
Tyler B Johnson and Carlos Guestrin. Training deep models faster with robust, approximate importance sampling. In Advances in Neural Information Processing Systems, pp. 7265-7275, 2018.  
Kenji Kawaguchi and Haihao Lu. Ordered sgd: A new stochastic optimization framework for empirical risk minimization. In International Conference on Artificial Intelligence and Statistics, pp. 669-679, 2020.  
Faisal Khan, Bilge Mutlu, and Jerry Zhu. How do humans teach: On curriculum learning and teaching dimension. In Advances in neural information processing systems, pp. 1449-1457, 2011.  
Justin Khim, Liu Leqi, Adarsh Prasad, and Pradeep Ravikumar. Uniform convergence of rank-weighted learning. In International Conference on Machine Learning, 2020.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521-3526, 2017.  
Tom Kocmi and Ondrej Bojar. Curriculum learning and minibatch bucketing in neural machine translation. In Proceedings of the International Conference Recent Advances in Natural Language Processing, RANLP 2017, pp. 379-386, 2017.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
M Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. In Advances in neural information processing systems, pp. 1189-1197, 2010.  
Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, and Luc Van Gool. Webvision database: Visual learning and understanding from web data. arXiv preprint arXiv:1708.02862, 2017.  
Yueming Lyu and Ivor W. Tsang. Curriculum loss: Robust learning and generalization against label corruption. In International Conference on Learning Representations, 2020.  
Karttikeya Mangalam and Vinay Uday Prabhu. Do deep neural networks learn shallow learnable examples first? In ICML workshop, 2019.  
Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher-student curriculum learning. IEEE transactions on neural networks and learning systems, 2019.  
Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. In Advances in neural information processing systems, pp. 1196-1204, 2013.  
Deanna Needell, Rachel Ward, and Nati Srebro. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. In Advances in neural information processing systems, pp. 1017-1025, 2014.  
Cuong V Nguyen, Alessandro Achille, Michael Lam, Tal Hassner, Vijay Mahadevan, and Stefano Soatto. Toward understanding catastrophic forgetting in continual learning. arXiv preprint arXiv:1908.01091, 2019.  
Diego Ortego, Eric Arazo, Paul Albert, Noel E O'Connor, and Kevin McGuinness. Towards robust learning with different label noise distributions. arXiv preprint arXiv:1912.08741, 2019.  
Anastasia Pentina, Viktoriia Sharmanska, and Christoph H Lampert. Curriculum learning of multiple tasks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 5492-5500, 2015.  
Emmanouil Antonios Platanios, Otilia Stretcu, Graham Neubig, Barnabas Poczos, and Tom Mitchell. Competence-based curriculum learning for neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 1162-1172, 2019.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.  
Vinay V Ramasesh, Ethan Dyer, and Maithra Raghu. Anatomy of catastrophic forgetting: Hidden representations and task semantics. arXiv preprint arXiv:2007.07400, 2020.  
Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In International Conference on Machine Learning, pp. 4334-4343, 2018.  
Hippolyt Ritter, Aleksandar Botev, and David Barber. Online structured laplace approximations for overcoming catastrophic forgetting. In Advances in Neural Information Processing Systems, pp. 3738-3748, 2018.  
Terence D Sanger. Neural network learning control of robot manipulators using gradually increasing task difficulty. IEEE transactions on Robotics and Automation, 10(3):323-333, 1994.  
Nikolaos Sarafianos, Theodore Giannakopoulos, Christophoros Nikou, and Ioannis A Kakadiaris. Curriculum learning for multi-task classification of visual attributes. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 2608-2615, 2017.

Shreyas Saxena, Oncel Tuzel, and Dennis DeCoste. Data parameters: A new family of parameters for learning a differentiable curriculum. In Advances in Neural Information Processing Systems, pp. 11095-11105, 2019.  
Vatsal Shah, Xiaoxia Wu, and Sujay Sanghavi. Choosing the sample with lowest loss makes sgd robust. volume 108 of Proceedings of Machine Learning Research, pp. 2120-2130, Online, 26-28 Aug 2020. PMLR.  
Yanyao Shen and Sujay Sanghavi. Learning with bad training data via iterative trimmed loss minimization. In International Conference on Machine Learning, pp. 5739-5748. PMLR, 2019.  
Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 90(2):227-244, 2000.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Sainbayar Sukhbaatar and Rob Fergus. Learning from noisy labels with deep neural networks. arXiv preprint arXiv:1406.2080, 2(3):4, 2014.  
Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International Conference on Machine Learning, pp. 6105-6114, 2019.  
Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. In ICLR, 2019.  
Lukas Vogelsang, Sharon Gilad-Gutnick, Evan Ehrenberg, Albert Yonas, Sidney Diamond, Richard Held, and Pawan Sinha. Potential downside of high initial visual acuity. Proceedings of the National Academy of Sciences, 115(44):11333-11338, 2018.  
Yiru Wang, Weihao Gan, Jie Yang, Wei Wu, and Junjie Yan. Dynamic curriculum learning for imbalanced data classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019.  
Daphna Weinshall, Gad Cohen, and Dan Amir. Curriculum learning by transfer learning: Theory and experiments with deep networks. In International Conference on Machine Learning, pp. 5238-5246, 2018.  
Lilian Weng. Curriculum for reinforcement learning. _lilianweng.github.io/lil-log_, 2020.  
Chulhee Yun, Suvrit Sra, and Ali Jabbabaie. Small relu networks are powerful memorizers: a tight analysis of memorization capacity. In Advances in Neural Information Processing Systems, pp. 15558-15569, 2019.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In Proceedings of the British Machine Vision Conference (BMVC), pp. 87.1-87.12. BMVA Press, September 2016.  
Wojciech Zaremba and Ilya Sutskever. Learning to execute. arXiv preprint arXiv:1410.4615, 2014.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. 2017.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Michael C. Mozer, and Yoram Singer. Identity crisis: Memorization and generalization under extreme overparameterization. In International Conference on Learning Representations, 2020.  
Jiong Zhang, Hsiang-Fu Yu, and Inderjit S Dhillon. Autoassist: A framework to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2019a.  
Xuan Zhang, Gaurav Kumar, Huda Khayrallah, Kenton Murray, Jeremy Gwinnup, Marianna J Martindale, Paul McNamee, Kevin Duh, and Marine Carpuat. An empirical exploration of curriculum learning for neural machine translation. arXiv preprint arXiv:1811.00739, 2018.

Xuan Zhang, Pamela Shapiro, Gaurav Kumar, Paul McNamee, Marine Carpuat, and Kevin Duh. Curriculum learning for domain adaptation in neural machine translation. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), June 2019b.
