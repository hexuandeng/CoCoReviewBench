# RAPID LEARNING OR FEATURE REUSE? TOWARDS UNDERSTANDING THE EFFECTIVENESS OF MAML

Anonymous authors

Paper under double-blind review

# ABSTRACT

An important research direction in machine learning has centered around developing meta-learning algorithms to tackle few-shot learning. An especially successful algorithm has been Model Agnostic Meta-Learning (MAML), a method that consists of two optimization loops, with the outer loop finding a meta-initialization, from which the inner loop can efficiently learn new tasks. Despite MAML's popularity, a fundamental open question remains – is the effectiveness of MAML due to the meta-initialization being primed for rapid learning (large, efficient changes in the representations) or due to feature reuse, with the meta initialization already containing high quality features? We investigate this question, via ablation studies and analysis of the latent representations, finding that feature reuse is the dominant factor. This leads to the ANIL (Almost No Inner Loop) algorithm, a simplification of MAML where we remove the inner loop for all but the (task-specific) head of the underlying neural network. ANIL matches MAML's performance on benchmark few-shot image classification and RL and offers computational improvements over MAML. We further study the precise contributions of the head and body of the network, showing that performance on the test tasks is entirely determined by the quality of the learned features, and we can remove even the head of the network (the NIL algorithm). We conclude with a discussion of the rapid learning vs feature reuse question for meta-learning algorithms more broadly.

# 1 INTRODUCTION

A central problem in machine learning is few-shot learning, where new tasks must be learned with a very limited number of labelled datapoints. A significant body of work has looked at tackling this challenge using meta-learning approaches (16; 37; 32; 6; 30; 28; 24). Broadly speaking, these approaches define a family of tasks, some of which are used for training and others solely for evaluation. A proposed meta-learning algorithm then looks at learning properties that generalize across the different training tasks, and result in fast and efficient learning of the evaluation tasks.

One highly successful meta-learning algorithm has been Model Agnostic Meta-Learning (MAML) (6). At a high level, the MAML algorithm is comprised of two optimization loops. The outer loop (in the spirit of meta-learning) aims to find an effective meta-initialization, from which the inner loop can perform efficient adaptation – optimize parameters to solve new tasks with very few labelled examples. This algorithm, with deep neural networks as the underlying model, has been highly influential, with significant follow on work, such as first order variants (24), probabilistic extensions (8), augmentation with generative modelling (29), and many others (15; 7; 12; 35).

Despite the popularity of MAML, and the numerous followups and extensions, there remains a fundamental open question on the basic algorithm. Does the meta-initialization learned by the outer loop result in rapid learning on unseen test tasks (efficient but significant changes in the representations) or is the success primarily due to feature reuse (with the meta-initialization already providing high quality representations)? In this paper, we explore this question and its many surprising consequences. Our main contributions are:

- We perform layer freezing experiments and latent representational analysis of MAML, finding that feature reuse is the predominant reason for efficient learning.  
- Based on these results, we propose the ANIL (Almost No Inner Loop) algorithm, a significant simplification to MAML that removes the inner loop updates for all but the head (final layer)

of a neural network during training and inference. ANIL performs identically to MAML on standard benchmark few-shot classification and RL tasks and offers computational benefits over MAML.

- We study the effect of the head of the network, finding that once training is complete, the head can be removed, and the representations can be used without adaptation to perform unseen tasks, which we call the No Inner Loop (NIL) algorithm.  
- We study different training regimes, e.g. multiclass classification, multitask learning, etc, and find that the task specificity of MAML/ANIL at training facilitate the learning of better features. We also find that multitask training, a popular baseline with no task specificity, performs worse than random features.  
- We discuss rapid learning and feature reuse in the context of other meta-learning approaches.

# 2 RELATED WORK

MAML (6) is a highly popular meta-learning algorithm for few-shot learning, achieving competitive performance on several benchmark few-shot learning problems (16; 37; 32; 30; 28; 24). It is part of the family of optimization-based meta-learning algorithms, with other members of this family presenting variations around how to learn the weights of the task-specific classifier. For example (19; 10; 4; 18; 39), first learn functions to embed the support set and target examples of a few-shot learning task, before using the test support set to learn task specific weights to use on the embedded target examples. (14) also proceeds similarly, using a Bayesian approach.

Of these optimization-based meta-learning algorithms, MAML has been especially influential, inspiring numerous direct extensions in recent literature (1; 8; 12; 29). Most of these extensions critically rely on the core structure of the MAML algorithm, incorporating an outer loop (for meta-training), and an inner loop (for task-specific adaptation), and there is little prior work analyzing why this central part of the MAML algorithm is practically successful. In this work, we focus on this foundational question, examining how and why MAML leads to effective few-shot learning. To do this, we utilize analytical tools such as Canonical Correlation Analysis (CCA) (26; 23) and Centered Kernel Alignment (CKA) (17) to study the neural network representations learned with the MAML algorithm, which also demonstrates MAML's ability to learn effective features for few-shot learning.

Insights from this analysis lead to a simplification that almost completely removes the inner optimization loop (the ANIL algorithm) with no reduction in performance. Other work has looked at having outer/inner loop specific parameters (40), but does this in a more complex fashion, partitioning parameters within each layer, and for specific layers, contrasting with the simple head/body separation in ANIL. Our work is complementary to methods extending MAML, and our simplification and insights could be applied to such extensions also.

# 3 MAML, RAPID LEARNING, AND FEATURE REUSE

Our goal is to understand whether the MAML algorithm efficiently solves new tasks due to rapid learning or feature reuse. In rapid learning, large representational and parameter changes occur during adaptation to each new task as a result of favorable weight conditioning from the meta-initialization. In feature reuse, the meta-initialization already contains highly useful features that can mostly be reused as is for new tasks, so little task-specific adaptation occurs. Figure 1 shows a schematic of these two hypotheses.

We start off by overviewing the details of the MAML algorithm, and then we study the rapid learning vs feature reuse questions via layer freezing experiments and analyzing latent representations of models trained with MAML. The results strongly support feature reuse as the predominant factor behind MAML's success. In Section 4, we explore the consequences of this, providing a significant simplification of MAML – the ANIL algorithm, and in Section 6, we outline the connections to meta-learning more broadly.

![](images/55ecbb2b7456e95fff00dabc8f73a0284f472f1c0c72fe1fe333e4ab4dd2113e.jpg)  
Figure 1: Rapid learning and feature reuse paradigms. In Rapid Learning, outer loop training leads to a parameter setting that is well-conditioned for fast learning, and inner loop updates result in significant task specialization. In Feature Reuse, the outer loop leads to parameter values corresponding to reusable features, from which the parameters do not move significantly in the inner loop. Images from (13; 9; 36; 2; 22; 34).

![](images/1f292c7a6657eea65c553a4700a49c1776431b175a9f975d2ffe8d4f662d8801.jpg)

# 3.1 OVERVIEW OF MAML

The MAML algorithm finds an initialization for a neural network so that new tasks can be learnt with very few examples ( $k$  examples from each class for  $k$ -shot learning) via two optimization loops:

- Outer Loop: Updates the initialization of the neural network parameters (often called the meta-initialization) to a setting that enables fast adaptation to new tasks.  
- Inner Loop: Performs adaptation: takes the outer loop initialization, and, separately for each task, performs a few gradient updates over the  $k$  labelled examples (the support set) provided for adaptation.

More formally, we first define our base model to be neural network with meta-initialization parameters  $\theta$ ; let this be represented by  $f_{\theta}$ . We have a distribution  $\mathcal{D}$  over tasks, and draw a batch  $\{T_1,\dots,T_B\}$  of  $B$  tasks from  $\mathcal{D}$ . For each task  $T_{b}$ , we have a support set of examples  $S_{T_b}$ , which are used for inner loop updates, and a target set of examples  $\mathcal{Z}_{T_b}$ , which are used for outer loop updates. Let  $\theta_i^{(b)}$  signify  $\theta$  after  $i$  gradient updates for task  $T_{b}$ , and let  $\theta_0^{(b)} = \theta$ . In the inner loop, during each update, we compute

$$
\theta_ {m} ^ {(b)} = \theta_ {m - 1} ^ {(b)} - \alpha \nabla_ {\theta_ {m - 1} ^ {(b)}} \mathcal {L} _ {S _ {T _ {b}}} \left(f _ {\theta_ {m - 1} ^ {(b)} (\theta)}\right) \tag {1}
$$

for  $m$  fixed across all tasks, where  $\mathcal{L}_{S_{T_b}}(f_{\theta_{m - 1}^{(b)}(\theta)})$  is the loss on the support set of  $T_{b}$  after  $m - 1$  inner loop updates.

We then define the meta loss as

$$
\mathcal {L} _ {m e t a} (\theta) = \sum_ {b = 1} ^ {B} \mathcal {L} _ {\mathcal {Z} _ {T _ {b}}} (f _ {\theta_ {m} ^ {(b)} (\theta)})
$$

where  $\mathcal{L}_{Z_{T_b}}(f_{\theta_m^{(b)}(\theta)})$  is the loss on the target set of  $T_{b}$  after  $m$  inner loop updates, making clear the dependence of  $f_{\theta_m^{(b)}}$  on  $\theta$ . The outer optimization loop then updates  $\theta$  as

$$
\theta = \theta - \eta \nabla_ {\theta} \mathcal {L} _ {m e t a} (\theta)
$$

At test time, we draw unseen tasks  $\{T_1^{(test)},\dots,T_n^{(test)}\}$  from the task distribution, and evaluate the loss and accuracy on  $\mathcal{Z}_{T_i^{(test)}}$  after inner loop adaptation using  $S_{T_i^{(test)}}$  (e.g. loss is  $\mathcal{L}_{\mathcal{Z}_{T_i^{(test)}}}\left(f_{\theta_m^{(i)}(\theta)}\right)$ ).

# 3.2 RAPID LEARNING OR FEATURE REUSE?

We now turn our attention to the key question: Is MAML's efficacy predominantly due to rapid learning or feature reuse? In investigating this question, there is an important distinction between the head (final layer) of the network and the earlier layers (the body of the network). In each few-shot

<table><tr><td>Freeze layers</td><td>MiniImageNet-5way-1shot</td><td>MiniImageNet-5way-5shot</td></tr><tr><td>None</td><td>44.5 ± 0.8</td><td>61.7 ± 0.6</td></tr><tr><td>1</td><td>44.8 ± 0.7</td><td>61.7 ± 0.7</td></tr><tr><td>1,2</td><td>44.8 ± 0.8</td><td>61.4 ± 0.7</td></tr><tr><td>1,2,3</td><td>44.7 ± 0.8</td><td>60.2 ± 0.7</td></tr><tr><td>1,2,3,4</td><td>44.7 ± 0.8</td><td>60.2 ± 0.7</td></tr></table>

Table 1: Freezing successive layers (preventing inner loop adaptation) does not affect accuracy, supporting feature reuse. To test the amount of feature reuse happening in the inner loop adaptation, we test the accuracy of the model when we freeze (prevent inner loop adaptation) a contiguous block of layers at test time. We find that freezing even all four convolutional layers of the network (all layers except the network head) hardly affects accuracy. This strongly supports the feature reuse hypothesis: layers don't have to change rapidly at adaptation time; they already contain good features from the meta-initialization.

learning task, there is a different alignment between the output neurons and classes. For instance, in task  $\mathcal{T}_1$ , the (wlog) five output neurons might correspond, in order, to the classes (dog, cat, frog, cupcake, phone), while for a different task,  $\mathcal{T}_2$ , they might correspond, in order, to (airplane, frog, boat, car, pumpkin). This means that the head must necessarily change for each task to learn the new alignment, and for the rapid learning vs feature reuse question, we are primarily interested in the behavior of the body of the network. We return to this in more detail in Section 5, where we present an algorithm (NIL) that does not use a head at test time.

To study rapid learning vs feature reuse in the network body, we perform two sets of experiments: (1) We evaluate few-shot learning performance when freezing parameters after MAML training, without test time inner loop adaptation; (2) We use representational similarity tools to directly analyze how much the network features and representations change through the inner loop. We use the MiniImageNet dataset, a popular standard benchmark for few-shot learning, and with the standard convolutional architecture in (6). Results are averaged over three random seeds. Full implementation details are in Appendix B.

# 3.2.1 FREEZING LAYER REPRESENTATIONS

To study the impact of the inner loop adaptation, we freeze a contiguous subset of layers of the network, during the inner loop at test time (after using the standard MAML algorithm, incorporating both optimization loops, for training). In particular, the frozen layers are not updated at all to the test time task, and must reuse the features learned by the meta-initialization that the outer loop converges to. We compare the few-shot learning accuracy when freezing to the accuracy when allowing inner loop adaptation.

Results are shown in Table 1. We observe that even when freezing all layers in the network body, performance hardly changes. This suggests that the meta-initialization has already learned good enough features that can be reused as is, without needing to perform any rapid learning for each test time task.

# 3.2.2 REPRESENTATIONAL SIMILARITY EXPERIMENTS

We next study how much the latent representations (the latent functions) learned by the neural network change during the inner loop adaptation phase. Following several recent works (26; 31; 23; 21; 27; 11; 3) we measure this by applying Canonical Correlation Analysis (CCA) to the latent representations of the network. CCA provides a way to compare representations of two (latent) layers  $L_{1}$ ,  $L_{2}$  of a neural network, outputting a similarity score between 0 (not similar at all) and 1 (identical). For full details, see (26; 23). In our analysis, we take  $L_{1}$  to be a layer before the inner loop adaptation steps, and  $L_{2}$  after the inner loop adaptation steps. We compute CCA similarity between  $L_{1}$ ,  $L_{2}$ , averaging the similarity score across different random seeds of the model and different test time tasks. Full details are in Appendix B.2

The result is shown in Figure 2, left pane. Representations in the body of the network (the convolutional layers) are highly similar, with CCA similarity scores of  $>0.9$ , indicating that the inner loop induces little to no functional change. By contrast, the head of the network, which does change

significantly in the inner loop, has a CCA similarity of less than 0.5. To further validate this, we also compute CKA (Centered Kernel Alignment) (17) (Figure 2 right), another similarity metric for neural network representations, which illustrates the same pattern. These representational analysis results strongly support the feature reuse hypothesis, with further results in Appendix Section B.4, B.3 providing yet more evidence.

![](images/d6e29352d4dc2b16e5578479bf86096bd71523129191d06ccd5a0cee3dc84284.jpg)  
Figure 2: High CCA/CKA similarity between representations before and after adaptation for all layers except the head. We compute CCA/CKA similarity between the representation of a layer before the inner loop adaptation and after adaptation. We observe that for all layers except the head, the CCA/CKA similarity is almost 1, indicating perfect similarity. This suggests that these layers do not change much during adaptation, but mostly perform feature reuse. Note that there is a slight dip in similarity in the higher conv layers (e.g. conv3, conv4); this is likely because the slight representational differences in conv1, conv2 have a compounding effect on the representations of conv3, conv4. The head of the network must change significantly during adaptation, and this is reflected in the much lower CCA/CKA similarity.

![](images/c55349bc6c2443c07b2e95a6a011ffa321caade3537fafc52b029da56cbc962d.jpg)

# 3.2.3 FEATURE REUSE HAPPENS EARLY IN LEARNING

Having observed that the inner loop does not significantly affect the learned representations with a fully trained model, we extend our analysis to see whether the inner loop affects representations and features earlier on in training. We take MAML models at 10000, 20000, and 30000 iterations into training, perform freezing experiments (as in Section 3.2.1) and representational similarity experiments (as in Section 3.2.2).

![](images/df227f7d666374bb7f2148d6116522c061612453f610a45a7dd187251e628268.jpg)  
Figure 3: Inner loop updates have little effect on learned representations from early on in learning. Left pane: we freeze contiguous blocks of layers (no adaptation at test time), on MiniImageNet-5way-5shot and see almost identical performance. Right pane: representations of all layers except the head are highly similar pre/post adaptation - i.e. features are being reused. This is true from very early (iteration 10000) in training.

![](images/051663671b7e2d5954be02ef21d88201409586192029296ad4b9cad5c44cacae.jpg)

Results in Figure 3 show the same patterns from early in training, with CCA similarity between activations pre and post inner loop update on MiniImageNet-5way-5shot being very high for the body (just like Figure 2), and similar to Table 1, test accuracy remaining approximately the same when freezing contiguous subsets of layers, even when freezing all layers of the network body. This shows that even early on in training, significant feature reuse is taking place, with the inner loop having minimal effect on learned representations and features. Results for 1shot MiniImageNet are in Appendix B.5, and show very similar trends.

![](images/510f4ad7a156f22e049b6f652f16d859dd0fca1556e4af5a6252934f8b58f71a.jpg)  
Figure 4: Schematic of MAML and ANIL algorithms. The difference between the MAML and ANIL algorithms: in MAML (left), the inner loop (task-specific) gradient updates are applied to all parameters  $\theta$ , which are initialized with the meta-initialization from the outer loop. In ANIL (right), only the parameters corresponding to the network head  $\theta_{head}$  are updated by the inner loop, during training and testing.

![](images/3669c8853bf196303996e31ff381ddc119072f06cd2fb233fa6311e1eca7e768.jpg)

<table><tr><td>Method</td><td>Omniglot-20way-1shot</td><td>Omniglot-20way-5shot</td><td>MiniImageNet-5way-1shot</td><td>MiniImageNet-5way-5shot</td></tr><tr><td>MAML</td><td>93.7 ± 0.7</td><td>96.4 ± 0.1</td><td>46.9 ± 0.2</td><td>63.1 ± 0.4</td></tr><tr><td>ANIL</td><td>96.2 ± 0.5</td><td>98.0 ± 0.3</td><td>46.7 ± 0.4</td><td>61.5 ± 0.5</td></tr></table>

<table><tr><td>Method</td><td>HalfCheetah-Direction</td><td>HalfCheetah-Velocity</td><td>2D-Navigation</td></tr><tr><td>MAML</td><td>170.4 ± 21.0</td><td>-139.0 ± 18.9</td><td>-20.3 ± 3.2</td></tr><tr><td>ANIL</td><td>363.2 ± 14.8</td><td>-120.9 ± 6.3</td><td>-20.1 ± 2.3</td></tr></table>

Table 2: ANIL matches the performance of MAML on few-shot image classification and RL. On benchmark few-shot classification tasks MAML and ANIL have comparable accuracy, and also comparable average return (the higher the better) on standard RL tasks (6).

# 4 THE ANIL (ALMOST NO INNER LOOP) ALGORITHM

In the previous section we saw that for all layers except the head of the neural network, the metainitialization learned by the outer loop of MAML results in very good features that can be reused as is on new tasks. Inner loop adaptation does not significantly change the representations of these layers, even from early on in training. This suggests a natural simplification of the MAML algorithm: the ANIL (Almost No Inner Loop) algorithm.

In ANIL, during training and testing, we remove the inner loop updates for the network body, and apply inner loop adaptation only to the head. The head requires the inner loop to allow it to align to the different classes in each task. In Section 5.1 we consider another variant, the NIL (No Inner Loop) algorithm, that removes the head entirely at test time, and uses learned features and cosine similarity to perform effective classification, thus avoiding inner loop updates altogether.

For the ANIL algorithm, mathematically, let  $\theta = (\theta_{1},\dots,\theta_{l})$  be the (meta-initialization) parameters for the  $l$  layers of the network. Following the notation of Section 3.1, let  $\theta_m^{(b)}$  be the parameters after  $m$  inner gradient updates for task  $\mathcal{T}_b$ . In ANIL, we have that:

$$
\theta_ {m} ^ {(b)} = \left(\theta_ {1}, \dots , (\theta_ {l}) _ {m - 1} ^ {(b)} - \alpha \nabla_ {(\theta_ {l}) _ {m - 1} ^ {(b)}} \mathcal {L} _ {S _ {b}} (f _ {\theta_ {m - 1} ^ {(b)})}\right)
$$

i.e. only the final layer gets the inner loop updates. As before, we then define the meta-loss, and compute the outer loop gradient update. The intuition for ANIL arises from Figure 3, where we observe that inner loop updates have little effect on the network body even early in training, suggesting the possibility of removing them entirely. Note that this is distinct to the freezing experiments, where we only removed the inner loop at inference time. Figure 4 presents the difference between MAML and ANIL, and Appendix C.1 considers a simple example of the gradient update in ANIL, showing how the ANIL update differs from MAML.

Computational benefit of ANIL: As ANIL almost has no inner loop, it significantly speeds up both training and inference. We found an average speedup of  $1.7\mathrm{x}$  per training iteration over MAML and an average speedup of  $4.1\mathrm{x}$  per inference iteration. In Appendix C.5 we provide the full results.

![](images/d2de184c22b84f3cde28562ed8c815339d165da32a58197baa0229dafb02d163.jpg)  
Figure 5: MAML and ANIL learn very similarly. Loss and accuracy curves for MAML and ANIL on MiniImageNet-5way-5shot, illustrating how MAML and ANIL behave similarly through the training process.

<table><tr><td>Model Pair</td><td>CCA Similarity</td><td>CKA Similarity</td></tr><tr><td>MAML-MAML</td><td>0.51</td><td>0.83</td></tr><tr><td>ANIL-ANIL</td><td>0.51</td><td>0.86</td></tr><tr><td>ANIL-MAML</td><td>0.50</td><td>0.83</td></tr></table>

Table 3: MAML and ANIL models learn comparable representations. Comparing CCA/CKA similarity scores of the of MAML-ANIL representations (averaged over network body), and MAML-MAML and ANIL-ANIL similarity scores (across different random seeds) shows algorithmic differences between MAML/ANIL does not result in vastly different types of features learned.

Results of ANIL on Standard Benchmarks: We evaluate ANIL on few-shot image classification and RL benchmarks, using the same model architectures as the original MAML authors, for both supervised learning and RL. Further implementation details are in Appendix C.4. The results in Table 2 (mean and standard deviation of performance over three random initializations) show that ANIL matches the performance of MAML on both few-shot classification (accuracy) and RL (average return, the higher the better), demonstrating that the inner loop adaptation of the body is unnecessary for learning good features.

MAML and ANIL Models Show Similar Behavior: MAML and ANIL perform equally well on few-shot learning benchmarks, illustrating that removing the inner loop during training does not hinder performance. To study the behavior of MAML and ANIL models further, we plot learning curves for both algorithms on MiniImageNet-5way-5shot, Figure 5. We see that loss and accuracy for both algorithms look very similar throughout training. We also look at CCA and CKA scores of the representations learned by both algorithms, Table 3. We observe that MAML-ANIL representations have the same average similarity scores as MAML-MAML and ANIL-ANIL representations, suggesting both algorithms learn comparable features (removing the inner loop doesn't change the kinds of features learned.) Further learning curves and representational similarity results are presented in Appendices C.2 and C.3.

# 5 CONTRIBUTIONS OF THE NETWORK HEAD AND BODY

So far, we have seen that MAML predominantly relies on feature reuse, with the network body (all layers except the last layer) already containing good features at meta-initialization. We also observe that such features can be learned even without inner loop adaptation during training (ANIL algorithm). The head, however, requires inner loop adaptation to enable task specificity.

In this section, we explore the contributions of the network head and body. We first ask: How important is the head at test time, when good features have already been learned? Motivating this question is that these features needed no adaptation at inference time, so perhaps they are themselves

<table><tr><td>Method</td><td>Omniglot-20way-1shot</td><td>Omniglot-20way-5shot</td><td>MiniImageNet-5way-1shot</td><td>MiniImageNet-5way-5shot</td></tr><tr><td>MAML</td><td>93.7 ± 0.7</td><td>96.4 ± 0.1</td><td>46.9 ± 0.2</td><td>63.1 ± 0.4</td></tr><tr><td>ANIL</td><td>96.2 ± 0.5</td><td>98.0 ± 0.3</td><td>46.7 ± 0.4</td><td>61.5 ± 0.5</td></tr><tr><td>NIL</td><td>96.7 ± 0.3</td><td>98.0 ± 0.04</td><td>48.0 ± 0.7</td><td>62.2 ± 0.5</td></tr></table>

Table 4: NIL algorithm performs as well as MAML and ANIL on few-shot image classification. Performance of MAML, ANIL, and NIL on few-shot image classification benchmarks. We see that with no test-time inner loop, and just learned features, NIL performs comparably to MAML and ANIL, indicating the strength of the learned features, and the relative lack of importance of the head at test time.

sufficient to perform classification, with no head. In Section 5.1, we find that test time performance is entirely determined by the quality of these representations, and we can use similarity of the frozen meta-initialization representations to perform unseen tasks, removing the head entirely. We call this the NIL (No Inner Loop) algorithm.

Given this result, we next study how useful the head is at training (in ensuring the network body learns good features). We look at multiple different training regimes (some without the head) for the network body, and evaluate the quality of the representations. We find that MAML/ANIL result in the best representations, demonstrating the importance of the head during training for feature learning.

# 5.1 THE HEAD AT TEST TIME AND THE NIL (NO INNER LOOP) ALGORITHM

Here, we study how important the head (and task specific alignment) are, when good features have already been learned (through training) by the meta-initialization. At test time, we find that the representations can be used directly, with no adaptation, which leads to the No Inner Loop (NIL) algorithm:

1 Train a few-shot learning model with ANIL/MAML algorithm as standard. We use ANIL training.  
2 At test time, remove the head of the trained model. For each task, first pass the  $k$  labelled examples (support set) through the body of the network, to get their penultimate layer representations. Then, for a test example, compute cosine similarities between its penultimate layer representation and those of the support set, using these similarities to weight the support set labels, as in (37).

The results for the NIL algorithm, following ANIL training, on few-shot classification benchmarks are given in Table 4. Despite having no network head and no task specific adaptation, NIL performs comparably to MAML and ANIL. This demonstrates that the features learned by the network body when training with MAML/ANIL (and reused at test time) are the critical component in tackling these benchmarks.

# 5.2 TRAINING REGIMES FOR THE NETWORK BODY

The NIL algorithm and results of Section 5.1, lead to the question of how important task alignment and the head are during training to ensure good features. Here, we study this question by examining the quality of features arising from different training regimes for the body. We look at (i) MAML and ANIL training; (ii) multiclass classification, where all of the training data and classes (from which training tasks are drawn) are used to perform standard classification; (iii) multitask training, a standard baseline, where no inner loop or task specific head is used, but the network is trained on all the tasks at the same time; (iv) random features, where the network is not trained at all, and features are frozen after random initialization; (v) NIL at training time, where there is no head and cosine distance on the representations is used to get the label.

After training, we apply the NIL algorithm to evaluate test performance, and quality of features learned at training. The results are shown in Table 5. MAML and ANIL training performs best. Multitask training, which has no task specific head, performs the worst, even worse than random features (adding evidence for the need for task specificity at training to facilitate feature learning.)

<table><tr><td>Method</td><td>MiniImageNet-5way-1shot</td><td>MiniImageNet-5way-5shot</td></tr><tr><td>MAML training-NIL head</td><td>48.4 ± 0.3</td><td>61.5 ± 0.8</td></tr><tr><td>ANIL training-NIL head</td><td>48.0 ± 0.7</td><td>62.2 ± 0.5</td></tr><tr><td>Multiclass training-NIL head</td><td>39.7 ± 0.3</td><td>54.4 ± 0.5</td></tr><tr><td>Multitask training-NIL head</td><td>26.5 ± 1.1</td><td>34.2 ± 3.5</td></tr><tr><td>Random features-NIL head</td><td>32.9 ± 0.6</td><td>43.2 ± 0.5</td></tr><tr><td>NIL training-NIL head</td><td>38.3 ± 0.6</td><td>43.0 ± 0.2</td></tr></table>

Table 5: MAML/ANIL training leads to superior features learned, supporting importance of head at training. Training with MAML/ANIL leads to superior performance over other methods which do not have task specific heads, supporting the importance of the head at training.

Using NIL during training performs worse than MAML/ANIL. These results demonstrate that the head is important at training to learn good features in the network body.

In Appendix D.1, we study test time performance variations from using a MAML/ANIL head instead of NIL, finding (as suggested by Section 5.1) very little performance difference. Additional results on similarity between the representations of different training regimes is given in Appendix D.2.

# 6 FEATURE REUSE IN OTHER META-LEARNING ALGORITHMS

Up till now, we have closely examined the MAML algorithm, and have demonstrated empirically that the algorithm's success is primarily due to feature reuse, rather than rapid learning. We now discuss rapid learning vs feature reuse more broadly in meta-learning. By combining our results with an analysis of evidence reported in prior work, we find support for many meta-learning algorithms succeeding via feature reuse, identifying a common theme characterizing the operating regime of much of current meta-learning.

# 6.1 OPTIMIZATION AND MODEL BASED META-LEARNING

MAML falls within the broader class of optimization based meta-learning algorithms, which at inference time, directly optimize model parameters for a new task using the support set. MAML has inspired many other optimization-based algorithms, which utilize the same two-loop structure (19; 29; 8). Our analysis so far has thus yielded insights into the feature reuse vs rapid learning question for this class of algorithms. Another broad class of meta-learning consists of model based algorithms, which also have notions of rapid learning and feature reuse.

In the model-based setting, the meta-learning model's parameters are not directly optimized for the specific task on the support set. Instead, the model typically conditions its output on some representation of the task definition. One way to achieve this conditioning is to jointly encode the entire support set in the model's latent representation (37; 33), enabling it to adapt to the characteristics of each task. This constitutes rapid learning for model based meta-learning algorithms.

An alternative to joint encoding would be to encode each member of the support set independently, and apply a cosine similarity rule (as in (37)) to classify an unlabelled example. This mode of operation is purely feature reuse – we do not use information defining the task to directly influence the decision function.

If joint encoding gave significant test-time improvement over non-joint encoding, then this would suggest that rapid learning of the test-time task is taking place, as task specific information is being utilized to influence the model's decision function. However, on analyzing results in prior literature, this improvement appears to be minimal. Indeed, in e.g. Matching Networks (37), using joint encoding one reaches  $44.2\%$  accuracy on MiniImageNet-5way-1shot, whereas with independent encoding one obtains  $41.2\%$ : a small difference. More refined models suggest the gap is even smaller. For instance, in (5), many methods for one shot learning were re-implemented and studied, and baselines without joint encoding achieved  $48.24\%$  accuracy in MiniImageNet-5way-1shot, whilst other models using joint encoding such as Relation Net (33) achieves very similar accuracy of  $49.31\%$

(they also report MAML, at  $46.47\%$ ). As a result, we believe that the dominant mode of "feature reuse" rather than "rapid learning" is what has currently dominated both MAML-styled optimization based meta-learning and model based meta-learning.

# 7 CONCLUSION

In this paper, we studied a fundamental question on whether the highly successful MAML algorithm relies on rapid learning or feature reuse. Through a series of experiments, we found that feature reuse is the dominant component in MAML's efficacy. This insight led to the ANIL (Almost No Inner Loop) algorithm, a simplification of MAML that has identical performance on standard image classification and reinforcement learning benchmarks, and provides computational benefits. We further study the importance of the head (final layer) of a neural network trained with MAML, discovering that the body (lower layers) of a network is sufficient for few-shot classification at test time, allowing us to remove the network head for testing (NIL) and still match performance. We connected our results to the broader literature in meta-learning, identifying feature reuse to be a common mode of operation for other meta-learning algorithms also. Based off of our conclusions, future work could look at developing and analyzing new meta-learning algorithms that perform more rapid learning, which may expand the datasets and problems amenable to these techniques.

# REFERENCES

[1] Antreas Antoniou, Harrison Edwards, and Amos Storkey. How to train your maml. arXiv preprint arXiv:1810.09502, 2018.  
[2] PMP Art. Overview figure: Lion image. https://www.pinterest.com/pin/350436414739113168/. Accessed: 2019-09-09.  
[3] Anthony Bau, Yonatan Belinkov, Hassan Sajjad, Nadir Durrani, Fahim Dalvi, and James Glass. Identifying and controlling important neurons in neural machine translation. arXiv preprint arXiv:1811.01157, 2018.  
[4] Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136, 2018.  
[5] Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Yu-Chiang Frank Wang, and Jia-Bin Huang. A closer look at few-shot classification. arXiv preprint arXiv:1904.04232, 2019.  
[6] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 1126-1135. JMLR.org, 2017.  
[7] Chelsea Finn and Sergey Levine. Meta-learning and universality: Deep representations and gradient descent can approximate any learning algorithm. arXiv preprint arXiv:1710.11622, 2017.  
[8] Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. In Advances in Neural Information Processing Systems, pages 9516–9527, 2018.  
[9] Nathan Glover. Overview figure: Cat image. https://thesecatsdonotexist.com/. Accessed: 2019-09-09.  
[10] Jonathan Gordon, John Bronskill, Matthias Bauer, Sebastian Nowozin, and Richard E Turner. Meta-learning probabilistic inference for prediction. arXiv preprint arXiv:1805.09921, 2018.  
[11] Akhilesh Gotmare, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation. arXiv preprint arXiv:1810.13243, 2018.  
[12] Erin Grant, Chelsea Finn, Sergey Levine, Trevor Darrell, and Thomas Griffiths. Recasting gradient-based meta-learning as hierarchical bayes. arXiv preprint arXiv:1801.08930, 2018.  
[13] GuideDogVerity. Overview figure: Dog image. https://twitter.com/guidedogverity. Accessed: 2019-09-09.  
[14] James Harrison, Apoorva Sharma, and Marco Pavone. Meta-learning priors for efficient online bayesian regression. arXiv preprint arXiv:1807.08912, 2018.  
[15] Kyle Hsu, Sergey Levine, and Chelsea Finn. Unsupervised learning via meta-learning. arXiv preprint arXiv:1810.02334, 2018.  
[16] Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese neural networks for one-shot image recognition. In ICML deep learning workshop, volume 2, 2015.  
[17] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. arXiv preprint arXiv:1905.00414, 2019.  
[18] Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 10657-10665, 2019.  
[19] Yoonho Lee and Seungjin Choi. Gradient-based meta-learning with learned layerwise metric and subspace. arXiv preprint arXiv:1801.05558, 2018.

[20] Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John E Hopcroft. Convergent learning: Do different neural networks learn the same representations? In FE@ NIPS, pages 196-212, 2015.  
[21] Niru Maheswaranathan, Alex H. Williams, Matthew D. Golub, Surya Ganguli, and David Sussillo. Universality and individuality in neural dynamics across large populations of recurrent networks. arXiv preprint arXiv:1907.08549, 2019.  
[22] Ahmed Malik. Overview figure: Plane image. https://appadvice.com/app/ real-airplane-pilot-flight-simulator-game-for-free/1186146488. Accessed: 2019-09-09.  
[23] Ari S Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural networks with canonical correlation. arXiv preprint arXiv:1806.05759, 2018.  
[24] Alex Nichol and John Schulman. Reptile: a scalable metalearning algorithm. arXiv preprint arXiv:1803.02999, 2, 2018.  
[25] Maithra Raghu. Svcca code and tutorials. https://github.com/google/svcca.  
[26] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems, pages 6076-6085, 2017.  
[27] Maithra Raghu, Chiyuan Zhang, Jon Kleinberg, and Samy Bengio. Transfusion: Understanding transfer learning with applications to medical imaging. arXiv preprint arXiv:1902.07208, 2019.  
[28] Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. 2016.  
[29] Andrei A Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. arXiv preprint arXiv:1807.05960, 2018.  
[30] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International conference on machine learning, pages 1842-1850, 2016.  
[31] Naomi Saphra and Adam Lopez. Understanding learning dynamics of language models with svcca. arXiv preprint arXiv:1811.00225, 2018.  
[32] Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, pages 4077-4087, 2017.  
[33] Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1199-1208, 2018.  
[34] Save Sheffield Trees. Overview figure: Tree image. https://twitter.com/saveshefftrees. Accessed: 2019-09-09.  
[35] Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Swersky, Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of datasets for learning to learn from few examples. arXiv preprint arXiv:1903.03096, 2019.  
[36] Vexels. Overview figure: Chair image. https://www.vexels.com/png-svg/preview/148959/small-office-chair-clipart. Accessed: 2019-09-09.  
[37] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in neural information processing systems, pages 3630-3638, 2016.  
[38] Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John E. Hopcroft. To what extent do different neural networks learn the same representation: A neuron activation subspace match approach. In NIPS 2018, 2018.

[39] Fengwei Zhou, Bin Wu, and Zhenguo Li. Deep meta-learning: Learning to learn in the concept space. arXiv preprint arXiv:1802.03596, 2018.  
[40] Luisa M Zintgraf, Kyriacos Shiarlis, Vitaly Kurin, Katja Hofmann, and Shimon Whiteson. Caml: Fast context adaptation via meta-learning. arXiv preprint arXiv:1810.03642, 2018.
