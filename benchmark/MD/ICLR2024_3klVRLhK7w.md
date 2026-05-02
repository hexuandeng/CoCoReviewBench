# BUDGETED ONLINE CONTINUAL LEARNING BY ADAPTIVE LAYER FREEZING AND FREQUENCY-BASED SAMPLING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The majority of online continual learning (CL) places restrictions on the size of replay memory and a single-epoch training to ensure a prompt update of the model. However, the single-epoch training may imply a different amount of computations per CL algorithm, and additional storage for storing logit or model in addition to replay memory is largely ignored as a storage budget. Here, we used floating point operations (FLOPs) and total memory size in Byte as a metric for computational and memory budgets, respectively, to compare CL algorithms with the same total budget. Interestingly, we found that the new and advanced algorithms often perform worse than simple baselines under the same budget, implying that their value is less beneficial in real-world deployment. To improve the accuracy of online continual learners in the same budget, we propose an adaptive layer freezing and frequency-based memory retrieval for episodic memory usage for a storage-and computationally efficient online CL algorithm. The proposed adaptive layer freezing does not update the layers for less informative batches to reduce computational costs with a negligible loss of accuracy. The proposed memory retrieval balances the training usage count of samples in episodic memory with a negligible computational and memory cost. In extensive empirical validations using CIFAR-10/100, CLEAR-10, and ImageNet-1K datasets, we demonstrate that the proposed method outperforms the state-of-the-art in the same total budget.

# 1 INTRODUCTION

Most online continual learning (CL) research places restrictions such as the single training epoch to quickly update the model and the size of the replay memory that limits the number of streamed samples that can be stored (Koh et al., 2022; Wang et al., 2022a). While one-epoch training may give a rough sense of the computational constraint for each method, as each method requires a different amount of computations in an epoch, the actual computation budget to train the models differs per method. Additionally, several rehearsal-based CL methods require additional storage to store the past model and logit (i.e., the unnormalized model output vector) (Buzzega et al., 2020; Koh et al., 2023; Zhou et al., 2023), which were not included in the replay memory size. Therefore, we attempt to rigorously compare CL methods with precise computational and memory constraints.

For a fair comparison of the methods in the same computational budget, we first consider the wall clock time of training as the metric. However, the wall clock time is highly dependent on hardware architectures, data I/O time, and the optimality of algorithm implementation (Gruber et al., 2022; Wintersperger et al., 2023; Prabhu et al., 2023). Thus, both the number of iterations and the training wall time may not be appropriate metrics for the computational cost of CL algorithms. In contrast, the Floating Point Operations (FLOPs) per sample is an exact computational budget regardless of the implementation details (Korthikanti et al., 2023). Following (Zhao et al., 2023; Ghunaim et al., 2023), we use training FLOPs as a metric for the computational budget.

For a fair comparison of the methods in the same memory budget, we need a total budget for various forms of extra storage including replay memory, logits, and model parameters. Following (Zhou et al., 2023), we convert all extra storage costs into Byte and sum them up to obtain the actual memory cost.

![](images/c9937ac9865587b546a9f06b7cb1e047e6d17409c256c20ccf39cf3a066f8a95.jpg)  
Figure 1: Comparison of CL methods with the same number of iterations and the same storage budget including the size of episodic memory and storage for past models for all methods (left) and our total-constrained CL considering training FLOPs per sample (right). (on CIFAR-10 Gaussian)

Taking into account the total memory and computational budget, we propose a computationally efficient online CL learning framework with negligible additional storage (0.02% of replay memory size in ImageNet), with a computation-aware layer freezing strategy. Specifically, we argue that, since not all layers are necessary to update for each data minibatch, if we find the appropriate layers to freeze for each mini-batch of data, we can reduce redundant training costs. We implement this idea by proposing 'adaptive layer freezing.' Since freezing earlier layers can save computational resources by reducing the computation of gradients in the backward pass (Hayes et al., 2020; Wu et al., 2020; He et al., 2021; Yuan et al., 2022), the frozen layers cannot learn any new information. Investigating the trade-off between computation and information due to freezing, we propose a novel method to choose the best layers to freeze by maximizing the Fisher Information(FI) gained by the model for each batch, given a fixed computation budget. Unlike previous freezing methods (Hayes et al., 2020; Lee et al., 2019; Yuan et al., 2022) that empirically select which layer to freeze, causing dependency on the dataset and the type of neural network, we consider the varying information of each batch, allowing us to determine the optimal layers to freeze at every forward pass.

While MIR (Aljundi et al., 2019a) and ASER (Shim et al., 2021) exhibit a substantial gain in accuracy, they demand high computational resources, as they require model inference on a large set of candidates. To obtain as much knowledge as possible on a limited total budget, we propose retrieving samples that the model has not learned much about from episodic memory. We utilize the frequency of recent use of each sample and the similarity of the gradient between classes, which are naturally obtained during training without requiring additional inference.

In our empirical validations, we compare the state-of-the-art CL algorithms in the literature under the same FLOPs of computations and the same Bytes of storage in Fig. 1. We observe that several high-performance CL methods are not competitive under fixed FLOPs and memory budget, interestingly, falling behind a simple Experience Replay (Rolnick et al., 2019). On the contrary, the proposed method outperforms them by a noticeable margin under the same computational and storage budget.

# Contributions. We summarize our contributions as follows:

- Proposing to rigorously measure computational and memory budgets of CL algorithms by using training FLOPs and total memory size in Bytes, to fairly compare different algorithms.  
- Proposing a computationally efficient adaptive layer freezing that maximizes Fisher Information per computation.  
- Proposing a memory retrieval strategy that promotes the retrieval of samples that the model has not learned much.  
- Empirical analysis on the computational and memory costs of various CL algorithms, showing that many state-of-the-art CL methods are less beneficial under the same budget and showing that the proposed method outperforms them by a noticeable margin across multiple benchmarks.

# 2 RELATED WORK

Online Continual Learning with Memory Budget. Replay-based online CL methods use episodic memory and consider the memory budget. Since we also consider the situation of using

episodic memory, we review them in detail as follows. The replay-based methods (Aljundi et al., 2019b; Prabhu et al., 2020; Bang et al., 2021; Koh et al., 2022; Wu et al., 2019) store part of the past data stream in episodic memory to replay them in future learning.

Although there are simple sampling strategies such as random sampling (Guo et al., 2020) and reservoir sampling (Vitter, 1985), they are often insufficient to adapt to changing data distributions. Rather than simple methods, researchers have developed advanced sampling strategies considering factors such as uncertainty, diversity, and gradient (Lopez-Paz & Ranzato, 2017; Bang et al., 2021; Koh et al., 2022; Tiwari et al., 2022). However, these advanced methods often entail a high computational overhead, making them impractical for use in real-world applications. RM (Bang et al., 2021) requires a significant amount of computational cost to calculate the uncertainty for diversified sampling. Similarly, CLIB (Koh et al., 2022) involves an additional forward and backward stage to calculate the decrease in memory sample loss for each batch iteration.

Not only the memory management schemes, but researchers have also investigated the memory usage schemes, i.e., sample retrieval strategies from the rehearsal buffers. In addition to random retrieval (Chaudhry et al., 2019), determining retrieval based on the degree of interference (Aljundi et al., 2019a) and the adversarial Shapley value (Shim et al., 2021) has been investigated. However, such methods require an inference of candidate samples, which leads to a nontrivial amount of computation in computing loss (Aljundi et al., 2019a) or the Shapely value (Shim et al., 2021).

Computationally Efficient Learning using Layer Freezing. Freezing layers have been investigated to reduce computational costs during training in joint training (i.e., ordinary training scenario other than CL) (Brock et al., 2017; Goutam et al., 2020; Xiao et al., 2019). A common freezing approach (Wang et al., 2023; Li et al., 2022) includes determining whether to freeze a layer based on the reference model and representation similarity, such as CKA (Cortes et al., 2012) and SP loss (Tung & Mori, 2019). Additionally, EGERIA (Wang et al., 2023) unfreezes layers based on changes in the learning rate.

However, in CL, both online and offline, it is challenging to determine when to freeze a layer because metrics such as Euclidean distance and CKA cannot be used to compare the degree of convergence compared to the reference model (Mirzadeh et al., 2020). Additionally, continual learning involves a non-i.i.d. setup, where the data distribution continues to change (Criado et al., 2022). Therefore, in addition to changes in learning rate, it is important to consider the current data distribution when determining whether to freeze or unfreeze a layer in continual learning. Hayes et al. (2020) have explored freezing methods for continual learning. However, they use predefined freezing configurations such as the freezing backbone block 0 after task 1, while our freezing method adaptively freezes the layers using information per batch.

# 3 APPROACH

Training a neural network requires two passes of network traversal; forward and backward. To make learning efficient, we consider two strategies; (1) reducing the number of passes and (2) the computational cost of each pass. We address both aspects by proposing two components; an adaptive layer-freezing method and a new memory retrieval method. The layer freezing reduces the computational cost of each backward pass, which consumes twice the computations of the forward one. The memory retrieval method retrieves training batches that are insufficiently learned, so the model learns the same amount of knowledge in fewer iterations, reducing the number of training passes. Comprising the two proposals into a single framework, we call our method Layer freezing and Similarity-Aware Retrieval (L-SAR). We illustrate our method in Fig. 2 and provide a pseudocode in Sec. A.2 in the appendix for the sake of space.

# 3.1 ADAPTIVE LAYER FREEZING FOR ONLINE CONTINUAL LEARNING

There have been several studies on the freezing of neural network layers in non-CL literature (Wang et al., 2023; Liu et al., 2021; He et al., 2021; Hinton et al., 2006). They suggest that freezing some layers can significantly reduce training computations with minimal impact on performance. These methods often rely on the convergence of each layer to determine which layers to freeze, since converged layers no longer require further training. However, in online CL, the model often

![](images/d31aa61fa2a1aaa2a1ad93c0164872d6d3c45eea544c18aad1221f16ce5a691c.jpg)  
Figure 2: Overview of the proposed L-SAR. The colors in the 'Similarity-Aware Retrieval' box denote different classes. The dotted arrows denote copying the values, while the solid arrows denote the calculation of new values. 'Retrieval Probability' is calculated using class similarity  $S$  and discounted use frequency  $c_{i}$ , where  $c_{i}$  tracks the number of times sample  $i$  has been used recently for training. A batch is sampled from memory by the 'Retrieval Probability' and  $c_{i}$  is updated by retrieval results. After the forward pass of the model with the batch, we compute the freezing criterion  $(I_{\mathrm{batch}} / C)_n$  for each layer  $n$  of the model, using Fisher Information and  $\left\| \frac{\partial \ell}{\partial x_L} \right\|$ . In the backward pass, layers 1 to  $n_{\max} = \arg \max_{n} (I_{\mathrm{batch}} / C)_n$  are frozen. Class similarity  $S_{ij}$  and Fisher Information are updated using the gradient  $\frac{\partial \ell}{\partial \theta}$  from the backward pass.

does not converge due to the limited training budget and the ever-evolving training data distribution, necessitating a new approach to decide when and which layers to freeze for incoming data.

Selectively Freezing Layers by Maximum Fisher Information (FI). For a computationally efficient freezing criterion, we propose freezing layers that learn little information per computation by measuring the amount of 'information' gained by each layer during training. Here, we define the information by the degree of certainty of the parameters with unknown true values (see Equation 1). With the information, we select the layers to freeze so that the model learns the maximum amount of information per computation. Since freezing reduces the computations for each iteration, we train a model with additional iterations in the same computational budget.

However, as a trade-off, freezing reduces the amount of information obtained per training iteration, since frozen layers do not gain information. To this end, to maximize the information  $(I)$  in the model while minimizing the computational cost  $(C)$ , we propose to maximize the expected amount of information gained per computation  $(I / C)$ . We factorize this by the amount of information gained per iteration  $(I / \mathrm{iter})$  and the number of iterations per computation  $(\mathrm{iter} / C)$ .

We first estimate  $(I / \mathrm{iter})$  when layers 1 to  $n$  are frozen, which we denote as  $(I / \mathrm{iter})_n$ , for  $n \in [1, L]$  where  $L$  is the total number of layers. The amount of information obtained by layer  $i$  is calculated by  $\operatorname{tr}(F(\theta_i))$ , where  $F(\theta_i)$  is the submatrix of Fisher Information Matrix  $F(\theta)$  corresponding to layer  $i$  and  $\operatorname{tr}(\cdot)$  is a trace operator. To efficiently calculate the information that each parameter acquires from the data, we use the diagonal component of the  $F(\theta_i)$  since the diagonal components only require first-order derivatives rather than Hessian (Kirkpatrick et al., 2017; Soen & Sun, 2021) as:

$$
(I / \operatorname {i t e r}) _ {n} = \sum_ {i = n + 1} ^ {L} \operatorname {t r} \left(F \left(\theta_ {i}\right)\right), \text {w h e r e} F (\theta) = \mathbb {E} _ {p _ {\theta} (z)} \left[ \left(\frac {\partial \ell}{\partial \theta}\right) \cdot \left(\frac {\partial \ell}{\partial \theta}\right) ^ {\intercal} \right], \tag {1}
$$

where  $\theta$  is the parameter of the model  $p_{\theta}(\cdot)$ ,  $z$  is the training data, and  $\ell = \log p_{\theta}(z)$  is the loss function.

Now, we calculate  $(\mathrm{iter} / C)_n$  which refers to the number of iterations per computation when freezing up to layer  $n \in [1, L]$ . For notation brevity, we define Unit Computation (UC) as the total FLOPs required for a complete forward and backward pass of the model using a single batch. Formally,  $\mathrm{UC} = \sum_{i=1}^{L} (\mathrm{BF})_i + (\mathrm{FF})_i$ , where  $(\mathrm{FF})_i$  and  $(\mathrm{BF})_i$  denote the forward FLOPs and the backward FLOPs of layer  $i$ , respectively. Without freezing, each iteration would cost 1 UC. In terms of UC,

we calculate  $(\mathrm{iter} / C)_n$  by the number of possible iterations given a computational budget of 1 UC, as:

$$
\left(\operatorname {i t e r} / C\right) _ {n} = \frac {\mathrm {U C}}{\mathrm {U C} - \sum_ {i = 1} ^ {n} (\mathrm {B F}) _ {i}}. \tag {2}
$$

As the number of freezing layers increases (i.e.,  $n$  increases), the possible iteration within the same computation increases.

Finally, combining Equation 1 and Equation 2, we can calculate expected amount of information gain per computation  $I / C$  by a product of  $I$ /iter and  $\text{iter} / C$ :

$$
(I / C) _ {n} = (I / \operatorname {i t e r}) _ {n} \cdot (\operatorname {i t e r} / C) _ {n} = \sum_ {i = n + 1} ^ {L} \operatorname {t r} \left(F \left(\theta_ {i}\right)\right) \cdot \frac {\operatorname {U C}}{\operatorname {U C} - \sum_ {i = 1} ^ {n} (\operatorname {B F}) _ {i}}. \tag {3}
$$

Therefore, by freezing layer 1 to layer  $n_{\mathrm{max}}$ , where  $n_{\mathrm{max}} = \arg \max_n (I / C)_n$ , we can maximize the expected amount of information gained per computation during training.

Batch-wise Version of  $(I / C)$  for Online CL. In online data stream, data distribution continuously shifts. Because of this, batches from past data distribution may contain less informative data, which makes it advantageous to freeze more layers, while batches from new distribution may contain much (i.e., new) information, which makes it advantageous to freeze fewer layers. Thus, instead of determining  $n_{\mathrm{max}}$  for the entire dataset, we adaptively freeze layers for each input batch by calculating  $(I / C)_n$  per batch.

Since the FI is quadratically proportional to the magnitude of the gradient (see Equation 1), we estimate the information of each batch to be proportional to the squared gradient magnitude of the batch. To avoid full backpropagation, we only use the gradient of the last layer feature  $x_{L}$  to estimate the gradient magnitude, following (Koh et al., 2023). For a detailed explanation of the gradient approximation using the last layer, please refer to Sec. A.1. Using these approximations, we obtain  $(I_{\mathrm{batch}} / C)_n$ , the batch-wise version of  $(I / C)_n$  as:

$$
(I _ {\text {b a t c h}} / C) _ {n} \left(z _ {\mathrm {t}}\right) = \frac {\left| \nabla_ {x _ {L}} \ell \left(z _ {\mathrm {t}}\right) \right| ^ {2}}{\mathbb {E} _ {z} \left[ \left| \nabla_ {x _ {L}} \ell (z) \right| ^ {2} \right]} \cdot \sum_ {i = n + 1} ^ {L} \operatorname {t r} \left(F \left(\theta_ {i}\right)\right) + \left(\frac {\sum_ {i = 1} ^ {n} (\mathrm {B F}) _ {i}}{\mathrm {U C}}\right) \cdot \max  _ {m} (I / C) _ {m}, \tag {4}
$$

where  $x_{L}$  represents the last layer features, and  $I / C$  is defined in Equation 3. Please refer to Sec. A.1 for the detailed derivation of Equation 4. Here, we compute the average gradient magnitude  $\mathbb{E}_z\left[|\nabla_{x_L}\ell (z)|^2\right]$  of the last layer and the trace of Fisher Information  $\mathrm{tr}(F(\theta_i)) = \mathbb{E}_{p_\theta (z)}\left[\mathrm{tr}\left(\left(\frac{\partial\ell}{\partial\theta_i}\right)\cdot \left(\frac{\partial\ell}{\partial\theta_i}\right)^{\intercal}\right)\right] = \mathbb{E}_z\left[\sum (\nabla_{\theta_i}l(z))^2\right]$  for all layers  $i\in [1,L]$ . Since calculating the expected values (using all samples) in every learning iteration is computationally expensive, we estimate them by exponential moving average (EMA) of the estimated expectations computed by the mini-batch of the past iterations. However, the EMA estimate of  $\mathrm{tr}(F(\theta_i))$  requires a gradient calculation for all layers, so it cannot be used with freezing, which stops gradient computations. Since the estimation of  $\mathrm{tr}(F(\theta_i))$  and freezing cannot be performed at the same time, at each  $m$  iteration, we train (i.e., unfreeze) all layers to update the estimate of  $\mathrm{tr}(F(\theta_i))$  for all  $i$ . For the other  $m - 1$  iterations, we do not update  $\mathrm{tr}(F(\theta_i))$  and freeze the model based on the values of  $I_{\mathrm{batch}} / C$ , using the previously estimated value of  $\mathrm{tr}(F(\theta_i))$ .

In summary, we maximize the information learned given a fixed computational budget by freezing layer 1 to  $n$  that maximizes  $(I_{\mathrm{batch}} / C)_n(z_t)$  for batch  $z_{t}$ . We argue that it allows us to dynamically adjust the degree of freezing based on the learning capacity of the layers and the batch informativeness.

# 3.2 SIMILARITY-AWARE RETRIEVAL BASED ON 'USE-FREQUENCY'

In rehearsal-based CL methods, sample retrieval strategies such as MIR (Aljundi et al., 2019a) and ASER (Shim et al., 2021) do not consider the computational cost. Despite the computational costs, some strategies perform worse than random retrieval (Fig. 1). Here, we propose a computationally efficient sample retrieval strategy.

In online CL, new data continuously streams in, and old data remains in memory, causing an imbalance in 'the number of times each sample is used for training', which we call 'use-frequency.' We

argue that samples with high use-frequency have been sufficiently learned by the model, and additional training with them provides a marginal knowledge gain while incurring computational costs. In contrast, samples with low use-frequency are likely to contain the knowledge that the model learns insufficiently. Therefore, we give these samples a higher probability of being retrieved.

Additionally, if a sample was frequently used in the past but less frequently in recent iterations, its knowledge may have been forgotten, despite a high use-frequency. Inspired by the exponential decaying model of forgetting (Shin & Lee, 2020; Mahto et al., 2020; Chien et al., 2021), we propose a decay factor in the use frequency at each iteration, denoted as  $0 < r < 1$ , resulting in the 'discounted-use-frequency.'

Effective Use Frequency  $(\hat{c}_i)$ . However, the model can learn knowledge about the sample by training other samples that are similar in the same class. This effectively increases the use-frequency for that sample. On the contrary, the model may lose knowledge about the sample when training on samples from other classes, effectively decreasing the use-frequency.

Inspired by Dhaliwal & Shintre (2018); Du et al. (2018), we assume that samples with similar gradients have similar information and effectively increase the use frequency, while those with opposite gradients would effectively decrease the use-frequency. To account for this, we define 'effective-use-frequency' by adding the other samples' use-frequency multiplied by the cosine similarity between gradients. However, since tracking the gradient similarities between all sample pairs requires excessive memory ( $\sim 10^{12}$  for ImageNet) and computation, we use class-wise similarities, which is the expected gradient similarity between samples from two classes. Formally, we define the class similarity  $S_{y_1,y_2}$  for classes  $y_1$  and  $y_2$  as:

$$
\mathcal {S} _ {y _ {1}, y _ {2}} = \mathbb {E} _ {z _ {1} \in D _ {y _ {1}}, z _ {2} \in D _ {y _ {2}}} \left[ \cos \left(\nabla_ {\theta} l \left(z _ {1}\right), \nabla_ {\theta} l \left(z _ {2}\right)\right) \right] \tag {5}
$$

where  $D_{y_i}$  is the training data for class  $y_i$ . Using class-wise similarities, we calculate the effective-use-frequency  $\hat{c}_i$  for the sample  $i$  as

$$
\hat {c} _ {i} = c _ {i} + \sum_ {y \in \mathcal {Y}} \mathcal {S} _ {y, y _ {i}} \cdot C _ {y}, \tag {6}
$$

where  $c_{i}$  is the discounted-use-frequency for sample  $i$ ,  $S_{y,y_i}$  is a class similarity between class  $y$  and  $y_{i}$ , and  $C_y$  is the sum of the discounted-use-frequencies for all samples of class  $y$ .

Calculating the expected value in Equation 5 from scratch requires a gradient calculation for all samples in class  $y_{1}$  and  $y_{2}$ , which is computationally expensive. As a computationally efficient alternative, we use the EMA to update the previous estimate rather than to evaluate the expectation from scratch. Note that we reuse the gradients obtained during training to calculate similarity and update the EMA estimate of  $S_{y_i, y_j}$ . Specifically, we calculate the cosine similarity of the gradients between all pairs of samples in the training batch and update the EMA estimate of  $S_{y_i y_j}$  using it, where  $y_i$  and  $y_j$  are labels for each pair. We only use the layers that are not frozen for this calculation.

To further reduce the computational cost of calculating similarity, we use only  $0.05\%$  of the model parameters for the calculation of similarity, since the gradient distribution of the subset of randomly selected weights is similar to the gradient of the entire weight set (Li et al., 2022).

Finally, we obtain the retrieval probabilities  $p_i$  for  $i$ -th sample by the softmax of  $-\hat{c}_i / T$ , where  $T$  is a temperature hyper-parameter, as

$$
p _ {i} = \frac {e ^ {- \hat {c} _ {i} / T}}{\sum_ {j = 1} ^ {| \mathcal {M} |} e ^ {- \hat {c} _ {j} / T}}. \tag {7}
$$

Samples with a low  $\hat{c}_i$  have a higher chance of being retrieved. This allows the model to prefer learning relatively insufficiently trained samples to sufficiently trained ones, thus accelerating the training. Note that this retrieval strategy uses information that is naturally obtained during training, such as use-frequency and gradients, imposing negligible additional computations.

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL SETUP

For empirical validation, we adopt the total budget for memory and computation. For the memory budget, we use Bytes (Zhou et al., 2023), which considers memory costs not only for the samples

in episodic memory but also for additional model parameters used in regularization or distillation. For the computational budget, we use FLOPs in the training phase. For dataset, we use CIFAR-10, CIFAR-100, ImageNet, and CLEAR-10. We evaluate the methods in conventional disjoint CL task setup and a newly proposed Gaussian task setup for boundary-free continuous data stream (Shanahan et al., 2021; Wang et al., 2022b; Koh et al., 2023). For all experiments, we averaged 3 different random seeds, except ImageNet due to computational cost (Bang et al., 2021; Koh et al., 2023). We conducted a Welch's  $t$ -test with a significance level of 0.05. If the highest performance in each column is statistically significant, it is highlighted in bold. Otherwise, it is underlined.

Metrics. We report the last accuracy  $A_{\mathrm{last}}$  and the area under the curve of accuracy  $A_{\mathrm{AUC}}$  (Koh et al., 2022). The  $A_{\mathrm{last}}$  measures the accuracy at the end of CL. The  $A_{\mathrm{AUC}}$  measures the average accuracy throughout the training course. To calculate  $A_{\mathrm{AUC}}$ , we use evaluation period as 100 samples for CIFAR-10, CIFAR-100 and CLEAR-10, 8000 samples for ImageNet. For each evaluation, we use the entire test set for the class seen so far as the evaluation set. We argue that  $A_{\mathrm{AUC}}$  is a suitable metric to measure prompt learning of new knowledge.

Baselines. We compare our method to Experience Replay (ER) (Rolnick et al., 2019), Dark Experience Replay (DER++) (Buzzega et al., 2020), Maximally Interfered Retrieval (MIR) (Aljundi et al., 2019a), Memory-efficient Expandable Model (MEMO) (Zhou et al., 2023), REMIND (Hayes et al., 2020), Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017) and Bias Correction (BiC) (Wu et al., 2019).

We describe the details of the implementation in Sec. A.4 in the Appendix for the sake of space.

# 4.2 RESULTS

We evaluate CL methods including L-SAR, with strictly restricted computation and memory budgets as specified in Sec. 4.1. Note that, unlike other methods, which could be adjusted to have the same FLOPs/sample by controlling the number of iterations/sample, L-SAR adaptively reduces FLOPs through adaptive layer freezing. Therefore, we set the (iteration/sample) of L-SAR to 1, which is the same as the (iteration/sample) of ER, which costs the least computation among the baselines.

Various Computational Budget under the Same Memory Budget. First, we compare CL methods under fixed memory budgets and various computational budgets in Fig. 3. We observe that L-SAR significantly outperforms other methods in all datasets and both Gaussian and disjoint task setups except ImageNet-Disjoint, especially under a low computational budget. It shows that our layer freezing and similarity-aware retrieval generally effectively reduce the computational cost, especially when the computational budget is tight. In ImageNet-Disjoint setup, some other methods show comparable performance with L-SAR. In that setup, since a large batch size of 256 is used, the data distribution in each batch does not change much, and so does the amount of total information in each batch, leading to less gain by our freezing method, which considers the information of each batch. Note that the disjoint CL setup is argued as less realistic scenarios (Prabhu et al., 2020; Bang et al., 2021; Koh et al., 2022), but we use it since many methods are proposed for that.

Additionally, we observe an increase in the FLOPs saved by L-SAR through freezing, particularly pronounced at higher computational budgets. As the model is trained for more iterations, the amount of information the model gains from the training data decreases. Thus, our adaptive layer freezing adaptively adjusts the freezing criterion to freeze more layers, leading to lower FLOPs, thus the line stops at the earlier GFLOPs value than the compared methods.

Various Memory Budget under the Same Computational Budget. We now fix the computational budget and test various memory budgets for CL methods, and summarize the results in Tab. 1 for CIFAR-100. L-SAR again outperforms other methods by a significant margin in all datasets, implying that L-SAR is suitable for both large and small memory budgets. Since L-SAR uses minimal additional memory in addition to episodic memory and utilizes episodic memory effectively by similarity-aware retrieval, L-SAR consistently outperforms other methods in various memory sizes. Please refer to Sec. A.8 for results with various memory budgets in CIFAR-10.

We investigate CL methods in domain incremental setup with fixed computational and memory budget using the CLEAR-10 dataset, in Tab. 2. Unlike the class incremental, where new classes are introduced to the model, the domain incremental introduces new samples that are in different domains, while the classes are maintained the same. As shown in the table, L-SAR also outperforms the state-of-the-art in domain incremental setups.

![](images/cf8387c2c7dfc72192b424d3f4c645f7f32f570cd8bdd76885113fdd6f1f90c6.jpg)  
(a) CIFAR-10 Gaussian task setup

![](images/12a70aa0ce1ff08fdc31ccf3fd4eaa5503b954bb467c69112aaf4573712ce241.jpg)  
(c) CIFAR-100 Gaussian task setup  
(d) CIFAR-100 Disjoint task setup

![](images/a1d0460907dc337a42a6e9a74bf0b84db367d219afba8f20a11d83382d49f02f.jpg)  
(e) ImageNet Gaussian task setup  
Figure 3: Accuracy on Gaussian and Disjoint CL setup in CIFAR-10, CIFAR-100, and ImageNet for a wide range of FLOPs per sample. L-SAR outperforms all CL methods compared.

Table 1: Accuracy for different memory sizes for Gaussian data Stream in CIFAR-100. The computational budget is fixed as 128.95 TFLOPs.  

<table><tr><td rowspan="2">Methods</td><td colspan="2">7.6 MB</td><td colspan="2">Memory Size 13.44 MB</td><td colspan="2">25.12 MB</td></tr><tr><td>\(A_{\text{AUC}} \uparrow\)</td><td>\(A_{\text{last}} \uparrow\)</td><td>\(A_{\text{AUC}} \uparrow\)</td><td>\(A_{\text{last}} \uparrow\)</td><td>\(A_{\text{AUC}} \uparrow\)</td><td>\(A_{\text{last}} \uparrow\)</td></tr><tr><td>ER (Rolnick et al., 2019)</td><td>22.56±1.61</td><td>27.52±1.90</td><td>22.95±1.62</td><td>29.93±0.82</td><td>22.62±2.15</td><td>29.04±2.58</td></tr><tr><td>REMIND (Hayes et al., 2020)</td><td>22.86±1.32</td><td>24.91±1.40</td><td>23.60±1.47</td><td>26.60±1.76</td><td>23.62±1.11</td><td>27.28±0.61</td></tr><tr><td>DER++ (Buzzega et al., 2020)</td><td>21.56±0.87</td><td>21.07±0.41</td><td>21.66±1.07</td><td>21.46±1.32</td><td>21.48±1.03</td><td>21.40±1.65</td></tr><tr><td>ER-MIR (Aljundi et al., 2019a)</td><td>12.13±2.39</td><td>13.13±3.09</td><td>12.91±1.83</td><td>13.72±2.26</td><td>12.44±2.28</td><td>13.41±2.77</td></tr><tr><td>EWC (Kirkpatrick et al., 2017)</td><td>19.27±1.37</td><td>19.75±1.50</td><td>20.67±2.09</td><td>24.34±2.43</td><td>20.72±2.65</td><td>24.21±2.28</td></tr><tr><td>BiC (Wu et al., 2019)</td><td>21.57±0.64</td><td>27.93±0.58</td><td>29.79±1.76</td><td>28.23±2.92</td><td>16.11±1.24</td><td>23.09±0.52</td></tr><tr><td>MEMO (Zhou et al., 2023)</td><td>26.61±0.37</td><td>17.46±1.26</td><td>29.65±0.51</td><td>30.56±0.61</td><td>29.93±0.61</td><td>33.25±0.62</td></tr><tr><td>L-SAR (Ours)</td><td>30.57±0.62</td><td>34.00±0.51</td><td>31.72±0.51</td><td>38.46±0.92</td><td>32.05±0.59</td><td>41.85±0.59</td></tr></table>

We believe this is because our retrieval method balances the use-frequency of samples in different domains so that the model learns more on relatively less-learned domains, allowing fast adaptation to new domains. It is more prominent in the results that L-SAR outperforms other methods by a larger gain in  $A_{\mathrm{AUC}}$  than in  $A_{\mathrm{last}}$ , where  $A_{\mathrm{AUC}}$  measures the accuracy of all time. Note that L-SAR also saves a significant amount of FLOPs thanks to adaptive layer freezing.

# 4.3 ABLATION STUDY

We now ablate the model to investigate the benefit of each of the proposed components by using CIFAR-10 and CIFAR-100 in the Gaussian task setup and summarize the results in Table 3. See

Table 2: Accuracy of various CL methods in domain-IL setup with CLEAR-10 dataset. Our L-SAR outperforms other methods with a smaller computational budget and same storage budget.  

<table><tr><td>Metric</td><td>EWC</td><td>ER</td><td>ER-MIR</td><td>BiC</td><td>REMIND</td><td>DER++</td><td>MEMO</td><td>L-SAR (Ours)</td></tr><tr><td>A_AUC ↑</td><td>63.40±0.17</td><td>64.61±0.15</td><td>59.02±0.31</td><td>51.20±0.46</td><td>63.86±0.37</td><td>61.55±0.51</td><td>58.24±0.65</td><td>68.47±0.17</td></tr><tr><td>A_fast ↑</td><td>73.85±1.29</td><td>75.58±0.70</td><td>70.40±0.69</td><td>66.57±1.61</td><td>75.05±0.23</td><td>73.70±0.46</td><td>67.64±1.92</td><td>76.57±0.58</td></tr><tr><td>TFLOPs ↓</td><td></td><td></td><td></td><td>2,640.37</td><td></td><td></td><td></td><td>2,104.20</td></tr></table>

Table 3: Benefits of the proposed components of our method in CIFAR-10 and CIFAR-100 for Gaussian task setup. The memory budget is 7.6MB for CIFAR-10 and 13.44MB for CIFAR-100. CIFAR-10 We train for 1 iter per sample for CIFAR-10 and 1.5 iter per sample for CIFAR-100.  

<table><tr><td rowspan="2">Methods</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td></tr><tr><td>\(A_{\text{AUC}} \uparrow\)</td><td>\(A_{\text{last}} \uparrow\)</td><td>TFLOPs ↓</td><td>\(A_{\text{AUC}} \uparrow\)</td><td>\(A_{\text{last}} \uparrow\)</td><td>TFLOPs ↓</td></tr><tr><td>Vanilla</td><td>60.76±0.11</td><td>70.08±0.97</td><td>163.74</td><td>31.97±0.89</td><td>37.80±1.30</td><td>245.91</td></tr><tr><td>+ Freezing</td><td>60.38±0.54</td><td>69.04±0.83</td><td>142.23</td><td>31.77±0.60</td><td>38.03±0.35</td><td>217.40</td></tr><tr><td>+ Retrieval</td><td>64.60±0.83</td><td>72.43±0.38</td><td>171.94</td><td>37.60±0.40</td><td>42.69±0.18</td><td>257.97</td></tr><tr><td>+ Retrieval &amp; Freezing (L-SAR)</td><td>64.38±0.32</td><td>72.57±0.79</td><td>146.80</td><td>37.20±0.73</td><td>42.55±0.79</td><td>221.49</td></tr></table>

Sec. A.7 in Appendix for ablation in the disjoint setup. For a comparison between adaptive layer freezing and naive layer freezing methods, please refer to Appendix Sec. A.6.

'Vanilla' is a simple replay-based method that trains on randomly retrieved batches from a balanced reservoir memory. As shown in the table, our similarity-aware retrieval based on use-frequency increases the performance of the baseline in the same number of iterations. This shows that our retrieval method increases the amount of knowledge learned per iteration, as we claim in Sec. 3.

While computational cost also increases, its increase is modest compared to other retrieval methods such as MIR (Aljundi et al., 2019a) or ASER (Shim et al., 2021) which require  $2 \sim 3 \times$  more computations. Also, we observe that the adaptive layer freezing method saves a significant amount of FLOPs while preserving accuracy. This shows that our freezing method effectively reduces the computational cost of each iteration as claimed in Sec. 3, with a negligible drop in performance. Summing up the effect of the two components, our method outperforms the baseline while using fewer FLOPs than the baseline, each by a noticeable margin. We show the effect of freezing on accuracy and FLOPs as the training progresses, in Fig. 4.

![](images/0bba19935d845ec3c60336b03f97a7f8a9d3e8d11b1953e63225329f5aac90b6.jpg)  
Figure 4: Accuracy and computational cost of the adaptive layer freezing in L-SAR. Training for 1 iteration per sample in CIFAR-10 Gaussian task setup.

![](images/cd1d505c492b92319631bb567ce61a4f114105c743e09e8450489908f671c2fb.jpg)

# 5 CONCLUSION

We address the challenge of achieving high performance on both old and new data with minimal computational cost and a limited storage budget in online CL. While CL with fixed episodic memory size has been extensively studied, we have investigated the total storage budget required for the online CL as well as the computational budget for developing practically useful online CL methods. To this end, we proposed L-SAR, a computation-efficient CL method comprising two components: similarity-aware frequency-based retrieval and adaptive layer freezing. Our empirical validations show that several high-performing CL methods are not competitive under a fixed computational budget, falling behind a simple baseline of training on randomly retrieved batches from memory.

Limitations and Future Work. While our method only requires negligible additional memory other than episodic memory, it does not actively optimize for the memory efficiency of CL algorithms. It is interesting to explore a method to use the limited storage budget more efficiently, e.g., storing quantized versions of models and exemplars.

# ETHICS STATEMENT

We propose a better learning scheme for online continual learning for realistic learning scenarios. While the authors do not explicitly aim for this, the increasing adoption of deep learning models in real-world contexts with streaming data could potentially raise concerns such as inadvertently introducing biases or discrimination. We note that we are committed to implementing all feasible precautions to avert such consequences, as they are unequivocally contrary to our intentions.

# REPRODUCIBILITY STATEMENT

We take reproducibility in deep learning very seriously and highlight some of the contents of the manuscript that might help to reproduce our work. We will definitely release our implementation of the proposed method in Sec. 3, the data splits and the baselines used in our experiments in Sec. 4

# REFERENCES

Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Laurent Charlin, Massimo Caccia, Min Lin, and Lucas Page-Caccia. Online continual learning with maximal interfered retrieval. NeurIPS, 32:11849-11860, 2019a. 2, 3, 5, 7, 8, 9, 14, 18  
Rahaf Aljundi, Min Lin, Baptiste Goujaud, and Yoshua Bengio. Gradient based sample selection for online continual learning. Advances in neural information processing systems, 32, 2019b. 3  
Soumya Banerjee, Vinay K Verma, Avideep Mukherjee, Deepak Gupta, Vinay P Namboodiri, and Piyush Rai. Verse: Virtual-gradient aware streaming lifelong learning with anytime inference. arXiv preprint arXiv:2309.08227, 2023. 19  
Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, and Jonghyun Choi. Rainbow memory: Continual learning with a memory of diverse samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8218-8227, 2021. 3, 7, 16  
Andrew Brock, Theodore Lim, James M Ritchie, and Nick Weston. Freezeout: Accelerate training by progressively freezing layers. arXiv preprint arXiv:1706.04983, 2017. 3  
Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920-15930, 2020. 1, 7, 8, 15, 18  
Lucas Caccia, Jing Xu, Myle Ott, Marcaurelio Ranzato, and Ludovic Denoyer. On anytime learning at macroscale. In Conference on Lifelong Learning Agents, pp. 165-182. CoLLAs, 2022. 19  
Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc'Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.3  
Hsiang-Yun Sherry Chien, Javier S Turek, Nicole Beckage, Vy A Vo, Christopher J Honey, and Ted L Willke. Slower is better: revisiting the forgetting mechanism in lstm for slower information decay. arXiv preprint arXiv:2105.05944, 2021.6  
Corinna Cortes, Mehryar Mohri, and Afshin Rostamizadeh. Algorithms for learning kernels based on centered alignment. The Journal of Machine Learning Research, 13(1):795-828, 2012. 3  
Marcos F Criado, Fernando E Casado, Roberto Iglesias, Carlos V Regueiro, and Senen Barro. Non-iid data and continual learning processes in federated learning: A long road ahead. Information Fusion, 88:263-280, 2022. 3  
Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In CVPR Workshops, pp. 702-703, 2020. 16  
Jasjeet Dhaliwal and Saurabh Shintre. Gradient similarity: An explainable approach to detect adversarial attacks against deep learning. arXiv preprint arXiv:1806.10707, 2018. 6

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020. 16, 18  
Yunshu Du, Wojciech M Czarnecki, Siddhant M Jayakumar, Mehrdad Farajtabar, Razvan Pascanu, and Balaji Lakshminarayanan. Adapting auxiliary losses using gradient similarity. arXiv preprint arXiv:1812.02224, 2018.6  
Yasir Ghunaim, Adel Bibi, Kumail Alhamoud, Motasem Alfarra, Hasan Abed Al Kader Hammoud, Ameya Prabhu, Philip HS Torr, and Bernard Ghanem. Real-time evaluation in online continual learning: A new hope. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11888-11897, 2023. 1, 19  
Kelam Goutam, S Balasubramanian, Darshan Gera, and R Raghunatha Sarma. Layerout: Freezing layers in deep neural networks. SN Computer Science, 1(5):295, 2020. 3  
Anthony Gruber, Max Gunzburger, Lili Ju, and Zhu Wang. A comparison of neural network architectures for data-driven reduced-order modeling. Computer Methods in Applied Mechanics and Engineering, 393:114764, 2022. 1  
Yunhui Guo, Mingrui Liu, Tianbao Yang, and Tajana Rosing. Improved schemes for episodic memory-based lifelong learning. Advances in Neural Information Processing Systems, 33:1023-1035, 2020. 3  
Tyler L Hayes, Kushal Kafle, Robik Shrestha, Manoj Acharya, and Christopher Kanan. Remind your neural network to prevent catastrophic forgetting. In European Conference on Computer Vision, pp. 466-483, 2020. 2, 3, 7, 8, 18  
Chaoyang He, Shen Li, Mahdi Soltanolkotabi, and Salman Avestimehr. Pipetransformer: automated elastic pipelining for distributed training of large-scale models. In International Conference on Machine Learning, pp. 4150-4159. PMLR, 2021. 2, 3  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pp. 770-778, 2016. 16  
Geoffrey E. Hinton, Simon Osindero, and Yee Whye Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18:1527-1554, 2006. 3  
James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. PNAS, 2017. 4, 7, 8, 15, 18  
Hyunseo Koh, Dahyun Kim, Jung-Woo Ha, and Jonghyun Choi. Online continual learning on class incremental blurry task configuration with anytime inference. In International Conference on Learning Representations, 2022. 1, 3, 7, 16, 19  
Hyunseo Koh, Minhyuk Seo, Jihwan Bang, Hwanjun Song, Deokki Hong, Seulki Park, Jung-Woo Ha, and Jonghyun Choi. Online boundary-free continual learning by scheduled data prior. In The Eleventh International Conference on Learning Representations, 2023. 1, 5, 7, 14  
Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5, 2023. 1  
Jaejun Lee, Raphael Tang, and Jimmy Lin. What would elsa do? freezing layers during transformer fine-tuning. arXiv preprint arXiv:1911.03090, 2019. 2  
Sheng Li, Geng Yuan, Yue Dai, Youtao Zhang, Yanzhi Wang, and Xulong Tang. Smartfrz: An efficient training framework using attention-based layer freezing. In The Eleventh International Conference on Learning Representations, 2022. 3, 6

Yuhan Liu, Saurabh Agarwal, and Shivaram Venkataraman. Autofreeze: Automatically freezing model blocks to accelerate fine-tuning. arXiv preprint arXiv:2102.01386, 2021. 3  
David Lopez-Paz and Marc'Aurelio Ranzato. Gradient episodic memory for continual learning. In NeurIPS, 2017. 3  
Shivangi Mahto, Vy Ai Vo, Javier S Turek, and Alexander Huth. Multi-timescale representation learning in lstm language models. In International Conference on Learning Representations, 2020. 6  
Seyed Iman Mirzadeh, Mehrdad Farajtabar, Dilan Gorur, Razvan Pascanu, and Hassan Ghasemzadeh. Linear mode connectivity in multitask and continual learning. In International Conference on Learning Representations, 2020. 3  
Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, and Davide Maltoni. Latent replay for real-time continual learning. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 10203-10209. IEEE, 2020. 19  
Ameya Prabhu, Philip HS Torr, and Puneet K Dokania. Gdumb: A simple approach that questions our progress in continual learning. In European Conference on Computer Vision, pp. 524-540, 2020. 3, 7, 16  
Ameya Prabhu, Hasan Abed Al Kader Hammoud, Puneet K Dokania, Philip HS Torr, Ser-Nam Lim, Bernard Ghanem, and Adel Bibi. Computationally budgeted continual learning: What does matter? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3698-3707, 2023. 1  
David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. Experience replay for continual learning. Advances in Neural Information Processing Systems, 32, 2019. 2, 7, 8, 18  
Michael Rotman and Lior Wolf. Shuffling recurrent neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 9428-9435, 2021. 18  
Murray Shanahan, Christos Kaplanis, and Jovana Mitrovic. Encoders and ensembles for task-free continual learning. arXiv preprint arXiv:2105.13327, 2021. 7  
Alex Sherstinsky. Fundamentals of recurrent neural network (rnn) and long short-term memory (lstm) network. Physica D: Nonlinear Phenomena, 404:132306, 2020. 18  
Dongsub Shim, Zheda Mai, Jihwan Jeong, Scott Sanner, Hyunwoo Kim, and Jongseong Jang. Online class-incremental continual learning with adversarial shapley value. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 9630-9638, 2021. 2, 3, 5, 9, 14  
Hyo-Sang Shin and Hae-In Lee. A new exponential forgetting algorithm for recursive least-squares parameter estimation. arXiv preprint arXiv:2004.03910, 2020. 6  
Alexander Soen and Ke Sun. On the variance of the fisher information for deep learning. Advances in Neural Information Processing Systems, 34:5708-5719, 2021. 4  
Rishabh Tiwari, Krishnateja Killamsetty, Rishabh Iyer, and Pradeep Shenoy. Gcr: Gradient coreset based replay buffer selection for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 99-108, 2022. 3  
Frederick Tung and Greg Mori. Similarity-preserving knowledge distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1365-1374, 2019. 3  
Jeffrey S Vitter. Random sampling with a reservoir. ACM Transactions on Mathematical Software (TOMS), 11(1):37-57, 1985. 3  
Liyuan Wang, Xingxing Zhang, Kuo Yang, Longhui Yu, Chongxuan Li, HONG Lanqing, Shifeng Zhang, Zhenguo Li, Yi Zhong, and Jun Zhu. Memory replay with data compression for continual learning. In International Conference on Learning Representations, 2022a. 1

Yiding Wang, Decang Sun, Kai Chen, Fan Lai, and Mosharaf Chowdhury. Egeria: Efficient cnn training with knowledge-guided layer freezing. In Proceedings of the Eighteenth European Conference on Computer Systems, pp. 851-866, 2023. 3  
Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In CVPR, 2022b. 7  
Karen Wintersperger, Florian Dommert, Thomas Ehmer, Andrey Hoursanov, Johannes Klepsch, Wolfgang Mauerer, Georg Reuber, Thomas Strohm, Ming Yin, and Sebastian Luber. Neutral atom quantum computing hardware: performance and end-user perspective. EPJ Quantum Technology, 10(1):32, 2023. 1  
Yikai Wu, Xingyu Zhu, Chenwei Wu, Annie Wang, and Rong Ge. Dissecting hessian: Understanding common structure of hessian in neural networks. arXiv preprint arXiv:2010.04261, 2020.2  
Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 374-382, 2019. 3, 7, 8, 15, 18  
Xueli Xiao, Thosini Bamunu Mudiyanselage, Chunyan Ji, Jie Hu, and Yi Pan. Fast deep learning training through intelligently freezing layers. In 2019 International Conference on Internet of Things (iThings) and IEEE Green Computing and Communications (GreenCom) and IEEE Cyber Physical and Social Computing (CPSCom) and IEEE Smart Data (SmartData), pp. 1225-1232. IEEE, 2019. 3  
Geng Yuan, Yanyu Li, Sheng Li, Zhenglun Kong, Sergey Tulyakov, Xulong Tang, Yanzhi Wang, and Jian Ren. Layer freezing & data sieving: Missing pieces of a generic framework for sparse training. Advances in Neural Information Processing Systems, 35:19061-19074, 2022. 2  
Haiyan Zhao, Tianyi Zhou, Guodong Long, Jing Jiang, and Chengqi Zhang. Does continual learning equally forget all parameters? In ICML, 2023. 1  
Da-Wei Zhou, Qi-Wei Wang, Han-Jia Ye, and De-Chuan Zhan. A model or 603 exemplars: Towards memory-efficient class-incremental learning. In The Eleventh International Conference on Learning Representations, 2023. 1, 6, 7, 8, 14, 18
