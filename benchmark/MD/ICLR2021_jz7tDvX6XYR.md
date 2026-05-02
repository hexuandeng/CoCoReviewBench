# SPEEDING UP DEEP LEARNING TRAINING BY SHARING WEIGHTS AND THEN UNSHARING

Anonymous authors

Paper under double-blind review

# ABSTRACT

It has been widely observed that increasing deep learning model sizes often leads to significant performance improvements on a variety of natural language processing and computer vision tasks. In the meantime, however, computational costs and training time would dramatically increase when models get larger. In this paper, we propose a simple approach to speed up training for a particular kind of deep networks which contain repeated structures, such as the transformer module. In our method, we first train such a deep network with the weights shared across all the repeated layers. Once an unsharing condition is triggered, we stop weight sharing and continue training until convergence. Empirical results show that our method is able to reduce the training time of BERT by  $50\%$ . We also conduct a preliminary theoretic analysis which motivates our approach.

# 1 INTRODUCTION

It has been widely observed that increasing model size often leads to significantly better performance on various real tasks, especially natural language processing and computer vision applications (Amodei et al., 2016; He et al., 2016a; Wu et al., 2016; Devlin et al., 2018; Brock et al., 2019; Brown et al., 2020; Lepikhin et al., 2020). However, as models getting larger, the training can become extremely resource intensive and time consuming. As a consequence, there has been a growing interest in developing systems and algorithms for efficient distributed large-batch training (Goyal et al., 2017; Shazeer et al., 2018; Lepikhin et al., 2020; You et al., 2020).

In this paper, we aims at speeding up deep learning training by exploiting unique network architectures rather than by distributed training. In particular, we are interested in speeding up the training of a special kind of deep networks which are constructed by repeatedly stacking the same layer, for example, the transformer module (Vaswani et al., 2017). We propose a simple method for efficiently training such kind of networks. In our approach, we first force the weights to be shared across all the repeated layers and train the network, and then, at some point, we stop weight sharing and continue training until convergence. The point for stopping weight sharing can be either predefined or automatically chosen by monitoring gradient statistics during training. Empirical studies show that our method can reduce the training time of BERT (Devlin et al., 2018) by  $50\%$ .

Our method is motivated by the successes of weight sharing models, in particular, ALBERT (Lan et al., 2020). It is a variant of BERT in which the weights across all the transformer layers are shared. As long as its architecture is sufficiently large, ALBERT can be comparable with or even outperform the original BERT on various downstream natural language processing benchmarks. However, when its architecture being the same as the original BERT, ALBERT performs significantly worse but still not that bad. Since the weights in the original BERT are not shared at all, it sounds natural for us to expect that ALBERT's performance will be improved if we stop its weight sharing at some point of training. The optimal models are supposed to not be far from weight sharing.

The other motivation for our method comes from our theoretical analysis on deep linear models. A deep linear model is constructed by a series of matrix multiplication (Hardt & Ma, 2016; Laurent & Brecht, 2018; Wu et al., 2019). In its forward pass, a deep linear model is just equivalent to a single matrix. However, when being trained with backpropagation, its behavior is analogous to the deep models with non-linearity but much easier to understand. Our theoretical analysis shows that, when learning a positive definite matrix (which admits an optimal solution with all layers having the same weights), training with weight sharing can bring significantly faster convergence. Moreover,

our theoretical analysis on deep linear models also provides insights in implementing the adaptive weight untying rule of our algorithm.

The rest of this paper is organized as follows. We present our training algorithm in Section 2. It actually contains three versions, depending on how to stop weight sharing during training. In Section 3, we present our theoretical results for positive definite deep linear models. All the proofs are deferred to the Appendix. In Section 4, we discuss related work. In Section 5, we show detailed experimental setup and results. We also provide various ablation studies on different choices in implementing our algorithm. Finally, we conclude this paper with some discussions in Section 6.

# 2 ALGORITHM: SHARING WEIGHTS AND THEN UNSHARING

Assume we have a deep network which is obtained by repeatedly stacking the same neural module  $n$  times, such as the transformer module in transformer models (Vaswani et al., 2017). Denote by  $w_{1},\ldots ,w_{n}$  the weights of these  $n$  modules. In our method, we first train the deep network with all the weights tied. Then, after a certain number of training steps, we untie the weights and further train the network until convergence. The untying point can be predefined or adaptively determined.

Stop weight sharing at a fixed point. This is the simplest version of our method (Algorithm 1). We first train the deep network with all the weights tied for a fixed number of steps, and then untie the weights and continue training until convergence.

Algorithm 1 SHARING WEIGHTS AND THEN UNSHARING (FIXED POINT)

1: Input: total number of training steps  $T$ , untying point  $\tau$ , learning rates  $\{\alpha^{(t)}, t = 1, \dots, T\}$  
2: Randomly and equally initialize weights  $w_{1}^{(0)},\ldots ,w_{n}^{(0)}$  
3: for  $t = 1$  to  $T$  do  
4: if  $t < \tau$  then  
5:  $w_{i}^{(t)} = w_{i}^{(t - 1)} - \alpha^{(t)}\times \mathrm{mean}\left\{\mathrm{grad}\left(\mathrm{loss},w_{k}^{(t - 1)}\right),k = 1,\ldots ,n\right\} ,i = 1,\ldots ,n$  
6: else  
7:  $w_{i}^{(t)} = w_{i}^{(t - 1)} - \alpha^{(t)}\times \operatorname {grad}\left(\operatorname {loss},w_{i}^{(t - 1)}\right), i = 1,\ldots ,n$

Note that, from line 1 to 5, we initialize all the weights equally, and then update them using the mean of their gradients. It is easy to see that such an update is equivalent to weight sharing or tying. For the sake of simplicity, in line 5 and 7, we only show how to update the weights using the plain (stochastic) gradient descent rule. One can replace this plain update rule with any of their favorite optimization methods, for example, the Adam optimization algorithm (Kingma & Ba, 2014).

While the repeated layers being the most natural units for weight sharing, that is not the only choice. We may view several layers together as the weight sharing unit, and share the weights across those units. The layers within the same unit can have different weights. For example, for a 24-layer transformer model, we may combine every four layers as a weight sharing unit. Thus, there will be six such units for weight sharing. Such a flexibility of choosing weight sharing units allows for a balance between "full weight sharing" and "no weight sharing" at all.

Stop weight sharing at an adaptive point. Instead of setting up a fixed point  $\tau$  in Algorithm 1, we can determine when to stop sharing weights using gradient statistics. A simple heuristics is as follows. For every certain number of iteration steps, we check the correlation between the gradients of any two adjacent layers. If more than half of these layer correlations are less than a predefined threshold  $\rho$  (0.5 as default) consecutively for a certain number of times, we stop sharing weights.

Stop weight sharing with multiple steps. We may gradually untie weights instead of untying them all the once (Algorithm 2). To implement this idea, the layers are distributed into different groups. The layers in the same group share weights, while the layers in different groups can have different weights. With training progresses, we check if any group meets a splitting criterion. Once it does, we split that group into two, and continue training in each subgroup. We repeat this process until all layers are untied or we reach the end of training. At the beginning, we can simply put all layers in the same group. The splitting criterion is like the above heuristics of adaptive weight untying. Assume we have a group of  $L$  layers, denoted by  $\{1,\dots ,L\}$ . We compute the correlation between

the gradients of any two adjacent layers. Once the correlation between layers  $i, i + 1$  falls below a predefined threshold, we split the group into two subgroups:  $\{1, \dots, i\}$  and  $\{i + 1, i + 2, \dots, L\}$ .

# Algorithm 2 SHARING WEIGHTS AND THEN UNSHARING (MULTI-STEP)

1: Input: total number of training steps  $T$ , learning rates  $\{\alpha^{(t)}, t = 1, \dots, T\}$  
2: Group layers, and randomly and equally initialize weights  
3: for  $t = 1$  to  $T$  do  
4: Split the groups which meet the splitting criterion  
5: Update the weights in each group using the gradient mean

# 3 THEORETIC ANALYSIS

We choose to focus on analyzing the simple linear case which allows analytic bound. The theoretical results here motivate the training via weight sharing, and also provide insights for the adaptive untying rule in our algorithm. Specifically, our main result is to show that training with weight sharing can achieve better convergence rate when the model admits a solution with all layers' weights shared (i.e. existing  $W_{1}^{*} = W_{2}^{*} = \dots = W_{L}^{*}$  such that the loss is 0).

We study the dynamics of training a deep linear network by gradient descent. The deep linear network is a series of matrix multiplication

$$
f (\mathbf {x}; W _ {1}, \dots , W _ {L}) = W _ {L} W _ {L - 1} \dots W _ {1} \mathbf {x}, \quad \quad \quad W _ {l} \in \mathbb {R} ^ {d \times d}, \quad \ell = 1, \dots , L.
$$

The task is to train the deep linear network to learn a target matrix  $\Phi \in \mathbb{R}^{d\times d}$ . To focus on the training dynamics, we adopt the simplified objective function

$$
\mathcal {R} \left(W _ {1}, \dots W _ {L}\right) = \frac {1}{2} \| W _ {L} W _ {L - 1} \dots W _ {2} W _ {1} - \Phi \| _ {F} ^ {2}.
$$

Denote  $\nabla_{l}\mathcal{R}$  to be the gradient of  $\mathcal{R}$  with respect to  $W_{l}$ . We have

$$
\nabla_ {l} \mathcal {R} = \frac {\partial \mathcal {R}}{\partial W _ {l}} = W _ {L: l + 1} ^ {T} (W _ {L: 1} - \Phi) W _ {l - 1: 1} ^ {T},
$$

where  $W_{l_2;l_1} = W_{l_2}W_{l_2 - 1}\dots W_{l_1 + 1}W_{l_1}$ . The standard gradient update is given by

$$
W _ {l} (t + 1) = W _ {l} (t) - \eta \nabla_ {l} \mathcal {R} (t), \quad l = 1, \dots , L.
$$

To train with weights shared, all the layers need to have the same initialization. And the update is

$$
W _ {l} (t + 1) = W _ {l} (t) - \frac {\eta}{L} \sum_ {i = 1} ^ {L} \nabla_ {i} \mathcal {R} (t), \quad l = 1, \dots , L. \tag {1}
$$

Since the initialization and updates are the same for all layers, the parameters  $W_{1}(t), \dots, W_{L}(t)$  are equal for all  $t$ . For simplicity, we denote the weight at time  $t$  to be  $W_{0}(t)$ . Notice that the gradients are averaged, the norm of update to each layer doesn't scale with  $L$ .

Suppose that the target matrix  $\Phi$  is a positive definite matrix. It is immediate that  $\Phi^{1 / n}$  is a solution to the deep linear network. Before looking into the detailed convergence analysis, it worth first showing a Lemma that reveals the updates in the weight sharing training.

Lemma 1. With a positive definite target matrix  $\Phi$  and initializing with  $W_0(0) = I$ , update the parameters according to Equation 1, we have

$$
W _ {l} (t + 1) - W _ {l} (t) = - \eta W _ {0} ^ {L - 1} (t) \left(W _ {0} ^ {L} (t) - \Phi\right), \quad l = 1, \dots , L, \quad \forall t \geq 0.
$$

Intuitively, the Lemma 1 shows that training with weight sharing allows all the layers to be trained "equally well", the layers that are far away from the output layer won't suffer from gradient vanishing or exploding.

In the following subsections, we first study the convergence result with continuous-time gradient descent, which demonstrates the benefit of training with weight sharing when learning a positive definite matrix  $\Phi$ . We then extend the results to the discrete-time gradient descent. We draw comparison with training with zero-asymmetric (ZAS) initialization (Wu et al., 2019). To be best of our knowledge, ZAS gives the state-of-the-art convergence rate. It is actually the only work showing the global convergence of deep linear network trained by gradient descent for an arbitrary target matrix.

# 3.1 CONTINUOUS-TIME GRADIENT DESCENT

With continuous-time gradient descent (i.e.  $\eta \rightarrow 0$ ), training with gradient descent and ZAS, the loss decays as  $\mathcal{R}(t) \leq \exp(-2t)\mathcal{R}(0)$ . For training with weight sharing, the loss becomes  $\mathcal{R}(t) \leq \exp(-2L\min(1,\lambda_{\min}(\Phi))t)\mathcal{R}(0)$ , when the target matrix  $\Phi$  is positive definite. The extra  $L$  in the exponent demonstrates the acceleration of training with weight sharing.

With  $\eta \rightarrow 0$ , the training dynamics of continuous-time gradient descent can be described as

$$
\frac {d W _ {l} (t)}{d t} = \dot {W} _ {l} (t) = - \nabla_ {l} \mathcal {R} (t), \quad l = 1, \dots , L, \quad t \geq 0.
$$

The ZAS initializes the weights  $W_{1} = W_{2} = \dots = W_{L - 1} = I$  and  $W_{L} = 0$ . It helps avoiding saddle points and has the following convergence result.

Theorem 1. [Continuous-time gradient descent without weight sharing (Wu et al., 2019)] For the deep linear network  $f(\mathbf{x}; W_1, \dots, W_L) = W_L W_{L-1} \dots W_1 \mathbf{x}$ , the continuous time gradient descent with the zero-asymmetric initialization satisfies

$$
\mathcal {R} (t) \leq \exp (- 2 t) \mathcal {R} (0).
$$

Theorem 1 shows that with the zero-asymmetric initialization, the continuous gradient descent linearly converges to the global optimal solution for general target matrix  $\Phi$ .

Next, considering the special case where the goal is to learn a positive definite matrix  $\Phi$ . Based on Lemma 1, we have the following convergence result for training with weight sharing.

Theorem 2. [Continuous-time gradient descent with weight sharing] For the deep linear network  $f(\mathbf{x}; W_1, \dots, W_L) = W_L W_{L-1} \dots W_1 \mathbf{x}$ , initialize all  $W_l(0)$  with identity matrix  $I$  and update according to Equation 1. With a positive definite target matrix  $\Phi$ , the continuous-time gradient descent satisfies

$$
\mathcal {R} (t) \leq \exp \left(- 2 L \min  \left(1, \lambda_ {\min } ^ {2} (\Phi)\right) t\right) \mathcal {R} (0).
$$

Remark 1. The difference between convergence rates in Theorem 1 and Theorem 2 is not an artifact of analysis. For example, when the target matrix is simply  $\Phi = \alpha I, \alpha > 1$ . It can be explicitly shown that with the initialization in Theorem 1, we have  $\dot{\mathcal{R}}(0) = -2\mathcal{R}(0)$  while training with weight sharing (Theorem 2), we have  $\dot{\mathcal{R}}(0) = -2L\mathcal{R}(0)$ . This implies that the convergence results in Theorem 1 and Theorem 2 cannot be improved in general.

The extra  $L$  in the exponent leads to faster convergence. The key to show the acceleration is

$$
\frac {d \mathcal {R} (t)}{d t} = \sum_ {l = 1} ^ {L} \operatorname {t r} \left(\nabla_ {l} ^ {\top} \mathcal {R} (t) \dot {W} _ {l} (t)\right) \leq - 2 L \lambda_ {\min } ^ {2} (W _ {0} (t) ^ {L - 1}) \mathcal {R} (t),
$$

where we see the  $L$  comes from the summation. This sheds light on two important factors that will affect the convergence speed:

1. All layers need to have sufficiently large update (i.e.  $\dot{W}_l(t)$  is large for all  $l$ ).  
2. Each layer's update needs to well correlate with its gradient (i.e.  $\nabla_{l}\mathcal{R}(t)$  correlates with  $\dot{W}_l(t)$ ).

Initializing all the weights to be the same and using the average of gradients to perform update guarantees that all layers are sufficiently trained. The high correlation of  $\nabla_{l}\mathcal{R}(t)$  and  $\dot{W}_l(t)$  essentially relies on  $\Phi$  being positive definite.

Suppose the gradients of different layers do not correlate well (e.g.  $\mathrm{tr}\left(\nabla_i\mathcal{R}(t)\nabla_j\mathcal{R}(t)\right)\approx 0,i\neq j$ ) and the weights are still forced to be shared via the updates according to Equation 1. Recall that  $\dot{W}_l(t) = \frac{1}{L}\sum_{i = 1}^L\nabla_i\mathcal{R}(t)$ , we then have  $\sum_{l = 1}^{L}\mathrm{tr}\left(\nabla_l^\top \mathcal{R}(t)\dot{W}_l(t)\right)\approx -\frac{1}{L}\sum_{l = 1}^{L}\| \nabla \mathcal{R}(t)\| _F^2$ , which loses the extra  $L$  acceleration in the convergence due to the  $1 / L$  leading factor.

When dealing with real deep learning models, there is no guarantee that all the gradients at different layers highly correlate. Thus, we may monitor gradient correlations during training: sharing weights when gradients well correlate, and break the ties when gradient correlations fall below a certain threshold. This matches the adaptive untying rule we proposed in Section 2.

# 3.2 DISCRETE-TIME GRADIENT DESCENT

Here we extend the previous result to the discrete-time gradient descent with a positive constant step size  $\eta$ . It can be shown that with zero-asymmetric initialization, training with the gradient descent will achieve  $\mathcal{R}(t) \leq \epsilon$  within  $O(L^3 \log(1/\epsilon))$  steps; initializing and training with weights sharing, the deep linear network will learn a positive definite matrix  $\Phi$  to  $\mathcal{R}(t) \leq \epsilon$  within  $O(L \log(1/\epsilon))$  steps, which reduces the required iterations by a factor of  $L^2$ .

To make easy comparisons, we first repeat without proving the discrete-time gradient descent convergence result of ZAS.

Theorem 3. [Continuous-time gradient descent without weight sharing (Wu et al., 2019)] For deep linear network  $f(\mathbf{x}; W_1, \ldots, W_L) = W_L W_{L-1} \ldots W_1 \mathbf{x}$  with zero-asymmetric initialization and discrete-time gradient descent, if the learning rate satisfies  $\eta \leq \min \left\{ \left(4L^3 \xi^6\right)^{-1}, \left(144L^2 \xi^4\right)^{-1} \right\}$ , where  $\xi = \max \left\{ 2 \| \Phi \|_F, 3 L^{-1/2}, 1 \right\}$ , then we have linear convergence  $\mathcal{R}(t) \leq \left( 1 - \frac{\eta}{2} \right)^t \mathcal{R}(0)$ .

Since the learning rate is  $\eta = O(L^{-3})$ , Theorem 3 indicates that the gradient descent can achieve  $\mathcal{R}(t) \leq \epsilon$  within  $O(L^3 \log(1/\epsilon))$  steps.

In the special case of learning a positive definite matrix  $\Phi$ , initialize all weights  $W_{l}$  to be the same and train with weights sharing, we have the following convergence result.

Theorem 4. [Discrete-time gradient descent with weight sharing] For the deep linear network  $f(\mathbf{x};W_1,\dots,W_L) = W_LW_{L - 1}\dots W_1\mathbf{x}$ , initialize all  $W_{l}(0)$  with identity matrix  $I$  and update according to Equation 1. With a positive definite target matrix  $\Phi$ , and setting  $\eta \leq \frac{\min(\lambda_{min}^2(\Phi),1)}{4\sqrt{dL^2}\max(\lambda_{max}^4(\Phi),1)}$ , we have linear convergence  $\mathcal{R}(t) \leq \exp \left[-(2L - 2)\min \left(\lambda_{min}^2 (\Phi),1\right)\eta t\right]\mathcal{R}(0)$ .

Take  $\lambda_{\mathrm{min}}(\Phi) / \lambda_{\mathrm{max}}(\Phi), d$  as constants and focus on the scaling with  $L, \epsilon$ , we have  $\eta = O(L^{-2})$ . Because of the extra  $L$  in the exponent, we know that when learning a positive definite matrix  $\Phi$ , training with weight sharing can achieve  $\mathcal{R}(t) \leq \epsilon$  within  $O(L \log(1/\epsilon))$  steps. The dependency on  $L$  reduces from previous  $L^3$  to linear, which shows the acceleration of training by weight sharing.

# 4 RELATED WORK

Lan et al. (2020) propose ALBERT with the weights being shared across all its transformer layers. Large ALBERT models can achieve good performance on several natural language understanding benchmarks. Bai et al. (2019b) propose trellis networks which are temporal convolution networks with shared weights and obtain good results for language modeling. This line of work is then extended to deep equilibrium models (Bai et al., 2019a) which are equivalent to infinite-depth weighted feedforward networks. Dabre & Fujita (2019) show that the translation quality of a model that recurrently stacks a single layer is comparable to having the same number of separate layers.

Deep linear models have been widely studied for its simplicity and similarity to deep learning models. Baldi & Hornik (1989) show that all local minima are also global minima for two-layer linear networks. Laurent & Brecht (2018) extend the same result to deep linear networks. Hardt & Ma (2016) show the PL condition is satisfied within the neighbour of a global optimum. Shamir (2019) show that, for one-dimensional deep linear networks, with the Xavier or near-identity initialization, it requires at least  $\exp (\Omega (L))$  steps to converge, where  $L$  is the depth. Wu et al. (2019) show that this result can be improved to  $O(L^{3}\log 1 / \epsilon)$  with a special zero-asymmetric initialization.

# 5 EXPERIMENTS

In this section, we present the experimental setup and results for training the BERT Large model with the standard training procedure as in the literature as well as our Sharing WEights (SWE) method. In what follows, without explicit clarification, BERT always means the BERT Large model.

# 5.1 EXPERIMENTAL SETUP

We use the TensorFlow official implementation of BERT (team & contributors). We first show experimental results English Wikipedia and BookCorpus for pre-training as in the original BERT paper (Devlin et al., 2018). We then move to the XLNet enlarged pretraining dataset (Yang et al., 2019). We preprocess all datasets with WordPiece tokenization (Schuster & Nakajima, 2012). We mask  $15\%$  tokens in each sequence. For experiments on English Wikipedia and BookCorpus, we randomly choose tokens to mask. For experiments on the XLNet dataset, we do whole word masking – in case that a word is broken into multiple tokens, either all tokens are masked or not masked. For all experiments, we set both the batch size and sequence length to 512.

We use the AdamW optimizer (Loshchilov & Hutter, 2017) with the weight decay rate being 0.01,  $\beta_{1} = 0.9$ , and  $\beta_{2} = 0.999$ . For English Wikipedia and BookCorpus, we use Pre-LN (He et al., 2016b; Xiong et al., 2020) instead of the original BERT's Post-LN. We notice that using Pre-LN with learning rate warmup leads to better performance. The learning rate starts from 0.0, linearly increases to the peak value  $3 \times 10^{-4}$  at the 10k-th iteration, and then linearly decays to 0.0. For the XLNet dataset, we use the Post-LN and the peak learning is set to  $10^{-4}$ , which is the same as the original BERT.

After pre-training, we fine-tune the models for the Stanford Question Answering Dataset (SQuAD v1.1 and SQuAD v2.0) (Rajpurkar et al., 2016) and the GLUE benchmark (Wang et al., 2018). For all fine-tuning tasks, we follow the setting as in the literature: the model is fine-tuned for 3 epochs; the learning rate warms up linearly from 0.0 to peak in the first  $10\%$  of the training iterations, then linearly decay to 0.0. We select the best peak learning rate based on the validation set from  $\{1\times 10^{-5},1.5\times 10^{-5},2\times 10^{-5},3\times 10^{-5},4\times 10^{-5},5\times 10^{-5},7.5\times 10^{-5},10\times 10^{-5},12\times 10^{-5}\}$ . For the SQuAD datasets, we fine-tune each model 5 times and report the average. For the GLUE benchmark, for each training method, we train four BERT models with different random seeds on each dataset, and then select the best model based on their devset results. We then submit the model's predictions over the test sets to the GLUE benchmark website to obtain test results.

**Training methods.** The training procedure in the TensorFlow official implementation of BERT serves as our baseline, where training on English Wikipedia plus BookCorpus takes 1 million steps. Training on the enlarged XLNet dataset takes 2 million steps. For our Sharing WEight (SWE) method, only half of the numbers of iterations are taken. For a complete comparison, we also report the results from the baseline method with half of the numbers of iterations. Three versions of our method with hyperparameter settings are listed below.

SWE-F Fixed point untying. By default, we set the untying point  $\tau = 50\mathrm{k}$ . We study the effect of different  $\tau$  values in Section 5.3.1.

SWE-A Adaptive point untying. We check the gradient correlations for every 1k iterations. If the majority of the correlations are below a threshold  $\rho = 0.5$  for three consecutive times, we break the tie. The effect of different  $\rho$  values is studied in Section 5.3.1.

SWE-M Multi-step adaptive untying. We use the same setup as in SWE-A.

# 5.2 EXPERIMENT RESULTS

For English Wikipedia and BookCorpus, both pretraining and finetuning results of our method vs. the baseline method are shown in Table 1. From the results, we see that our method with  $500\mathrm{k}$  training iterations matches the performance of the baseline method with 1 million training iterations, and significantly outperforms the baseline method with  $500\mathrm{k}$  training iterations. The results for the XLNet dataset are shown in Table 2. We observe similar advantages of our approach over the baseline.

# 5.3 ABLATION STUDIES

In this section, we study the effects of different choices in implementing our method.

Table 1: Training BERT on English Wikipedia and BookCorpus. Our method with 500k iterations matches the baseline performance with 1m iteration steps, and outperforms the baseline performance with 500k iterations.  

<table><tr><td rowspan="2"></td><td colspan="2">Baselines</td><td colspan="3">Our method, 500k iter.</td></tr><tr><td>1m iter.</td><td>500k iter.</td><td>SWE-F</td><td>SWE-A</td><td>SWE-M</td></tr><tr><td>Pretrain MLM (acc.%)</td><td>74.42</td><td>73.21</td><td>73.72</td><td>73.86</td><td>73.73</td></tr><tr><td>SQuAD v1.1 (F-1%)</td><td>92.24</td><td>91.19</td><td>92.24</td><td>92.25</td><td>91.88</td></tr><tr><td>SQuAD v2.0 (F-1%)</td><td>84.01</td><td>82.52</td><td>83.71</td><td>84.60</td><td>83.93</td></tr><tr><td>GLUE/AX (corr%)</td><td>38.0</td><td>34.4</td><td>40.4</td><td>37.4</td><td>38.2</td></tr><tr><td>GLUE/MNLI-m (acc.%)</td><td>86.4</td><td>84.8</td><td>86.6</td><td>86.9</td><td>86.6</td></tr><tr><td>GLUE/MNLI-mm (acc.%)</td><td>85.6</td><td>84.5</td><td>85.9</td><td>86.6</td><td>85.7</td></tr><tr><td>GLUE/QNLI (acc.%)</td><td>91.7</td><td>82.7</td><td>92.0</td><td>92.5</td><td>91.9</td></tr><tr><td>GLUE/QQP (F-1%)</td><td>71.4</td><td>70.2</td><td>70.5</td><td>71.5</td><td>70.7</td></tr><tr><td>GLUE/SST-2 (acc.%)</td><td>94.6</td><td>94.5</td><td>94.2</td><td>94.9</td><td>94.6</td></tr></table>

Table 2: Training BERT on the XLNet dataset. Our method with 1 million iterations matches the baseline performance with 2 million iteration steps, and outperforms the baseline performance with 1 million iterations.  

<table><tr><td></td><td colspan="2">Baselines</td><td colspan="2">Our method, 1m iter.</td></tr><tr><td></td><td>2m iter.</td><td>1m iter.</td><td>SWE-F</td><td>SWE-M</td></tr><tr><td>Pretrain MLM (acc.%)</td><td>72.85</td><td>70.54</td><td>71.38</td><td>71.37</td></tr><tr><td>SQuAD v1.1 (F-1%)</td><td>93.16</td><td>92.96</td><td>93.76</td><td>93.44</td></tr><tr><td>SQuAD v2.0 (F-1%)</td><td>86.90</td><td>85.45</td><td>86.86</td><td>86.62</td></tr><tr><td>GLUE/AX (corr%)</td><td>40.8</td><td>40.1</td><td>40.2</td><td>40.4</td></tr><tr><td>GLUE/MNLI-m (acc.%)</td><td>89.5</td><td>88.7</td><td>89.5</td><td>89.3</td></tr><tr><td>GLUE/MNLI-mm (acc.%)</td><td>88.9</td><td>88.0</td><td>88.8</td><td>88.5</td></tr><tr><td>GLUE/QNLI (acc.%)</td><td>94.3</td><td>93.2</td><td>93.6</td><td>93.9</td></tr><tr><td>GLUE/QQP (F-1%)</td><td>73.1</td><td>72.9</td><td>72.3</td><td>72.5</td></tr><tr><td>GLUE/SST-2 (acc.%)</td><td>96.3</td><td>96.1</td><td>96.6</td><td>96.8</td></tr></table>

# 5.3.1 WHEN TO STOP WEIGHT SHARING

In this section, we study the effect of using different untying points and thresholds. If weights are shared throughout the entire pretraining process, the final performance will be much worse than without any form of weight sharing (Lan et al., 2020). On the other hand, without weight sharing at all yields slower convergence.

Results of using different untying point  $\tau$  and threshold  $\rho$  values are summarized in Table. 3. Models are trained for  $500\mathrm{k}$  iterations on English Wikipedia and BookCorpus. From the results, we see that for the SWE-F method, a smaller  $\tau$  value performs better than a larger  $\tau$ . This means that the weight sharing stage should not be too long. We also see that the performance of the SWE-M method is not sensitive to the threshold  $\rho$ .

# 5.3.2 HOW TO CHOOSE WEIGHT SHARING UNITS

Note that it is not necessary to be restricted to share weights only across the original layers. We can group several consecutive layers as a weight sharing unit. We denote  $A \times B$  as grouping  $A$  layers as a weight sharing unit which is being shared with  $B$  times. Since BERT has 24 layers, the baseline method without weight sharing can be viewed as "24x1", and our method shown in Table 1 can be viewed as "1x24". We present results from more different choices of weight sharing units in Table 4. We can see that, in order to achieve good results, the size of the chosen weight sharing unit should not be larger than 6 layers. This means that the weights of a layer must be shared for at least 4 times.

Table 3: Results from different untying points  $\tau$  and thresholds  $\rho$  . Models are trained for  ${500}\mathrm{k}$  iterations on English Wikipedia and BookCorpus.  

<table><tr><td rowspan="2"></td><td colspan="2">SWE-F</td><td colspan="5">SWE-M</td></tr><tr><td>τ=50k</td><td>τ=200k</td><td>ρ=0.1</td><td>ρ=0.3</td><td>ρ=0.5</td><td>ρ=0.7</td><td>ρ=0.9</td></tr><tr><td>Pretrain MLM (acc.%)</td><td>73.72</td><td>72.27</td><td>73.40</td><td>74.16</td><td>73.73</td><td>73.90</td><td>73.62</td></tr><tr><td>SQuAD v1.1 (F-1%)</td><td>92.24</td><td>91.05</td><td>92.04</td><td>92.19</td><td>91.88</td><td>92.04</td><td>91.15</td></tr><tr><td>SQuAD v2.0 (F-1%)</td><td>83.71</td><td>83.30</td><td>84.54</td><td>84.25</td><td>83.93</td><td>85.04</td><td>82.24</td></tr><tr><td>GLUE/AX (corr%)</td><td>40.4</td><td>38.2</td><td>37.3</td><td>38.5</td><td>38.2</td><td>37.0</td><td>37.0</td></tr><tr><td>GLUE/MNLI-m (acc.%)</td><td>86.6</td><td>85.9</td><td>86.6</td><td>85.5</td><td>86.6</td><td>86.1</td><td>85.9</td></tr><tr><td>GLUE/MNLI-mm (acc.%)</td><td>85.9</td><td>85.1</td><td>85.9</td><td>84.4</td><td>85.7</td><td>85.9</td><td>85.5</td></tr><tr><td>GLUE/QNLI (acc.%)</td><td>92.0</td><td>91.7</td><td>92.6</td><td>91.1</td><td>91.9</td><td>91.5</td><td>90.2</td></tr><tr><td>GLUE/QQP (F-1%)</td><td>70.5</td><td>70.7</td><td>71.2</td><td>70.4</td><td>70.7</td><td>70.7</td><td>70.0</td></tr><tr><td>GLUE/SST-2 (acc.%)</td><td>94.2</td><td>94.0</td><td>94.6</td><td>94.2</td><td>94.6</td><td>94.8</td><td>94.5</td></tr></table>

Table 4: We group several consecutive layers as a weight sharing unit instead of sharing weights only across original layers.  $A \times B$  means grouping  $A$  layers as a unit which is being shared with  $B$  times. Models are trained for 500k iterations on English Wikipedia and BookCorpus.  

<table><tr><td rowspan="2"></td><td rowspan="2">Baseline 24x1</td><td colspan="5">SWE-F</td></tr><tr><td>12x2</td><td>6x4</td><td>4x6</td><td>2x12</td><td>1x24</td></tr><tr><td>Pretrain MLM (acc.%)</td><td>73.21</td><td>73.39</td><td>73.88</td><td>73.88</td><td>73.82</td><td>73.72</td></tr><tr><td>SQuAD v1.1 (F-1%)</td><td>91.19</td><td>92.31</td><td>92.40</td><td>92.34</td><td>91.93</td><td>92.24</td></tr><tr><td>SQuAD v2.0 (F-1%)</td><td>82.52</td><td>84.25</td><td>84.88</td><td>85.31</td><td>84.47</td><td>83.71</td></tr><tr><td>GLUE/AX (corr%)</td><td>34.4</td><td>37.0</td><td>40.2</td><td>38.2</td><td>37.8</td><td>40.4</td></tr><tr><td>GLUE/MNLI-m (acc.%)</td><td>84.8</td><td>86.0</td><td>86.7</td><td>87.0</td><td>86.4</td><td>86.6</td></tr><tr><td>GLUE/MNLI-mm (acc.%)</td><td>84.5</td><td>86.2</td><td>86.4</td><td>86.0</td><td>85.8</td><td>85.9</td></tr><tr><td>GLUE/QNLI (acc.%)</td><td>82.7</td><td>91.5</td><td>92.9</td><td>91.8</td><td>92.8</td><td>92.0</td></tr><tr><td>GLUE/QQP (F-1%)</td><td>70.2</td><td>71.4</td><td>71.7</td><td>71.3</td><td>70.8</td><td>70.5</td></tr><tr><td>GLUE/SST-2 (acc.%)</td><td>94.5</td><td>93.8</td><td>94.8</td><td>94.8</td><td>94.6</td><td>94.2</td></tr></table>

# 6 CONCLUSION AND DISCUSSION

We proposed a simple method to speed up the training of deep networks with repeated layers and showed promising empirical results. Our method is motivated by the successes of weight sharing models in the literature as well as our theoretic analysis on deep linear models. For the future work, we will extend our empirical studies to more deep learning models and tasks, and analyze under which conditions our method will be efficient. In addition, the presented theoretic analysis is fairly preliminary. It even cannot be directly applied to illustrate a slightly more realistic case where the truth may somewhat deviate from weight sharing, not to mention dealing with the deep models with non-linearity. We believe a deep theoretic understanding will help further improve this approach.

# REFERENCES

Dario Amodei, Sundaram Ananthanarayanan, Rishita Anubhai, Jingliang Bai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Qiang Cheng, Guoliang Chen, et al. Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning, pp. 173-182, 2016.  
Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Advances in Neural Information Processing Systems, pp. 690-701, 2019a.  
Shaojie Bai, J Zico Kolter, and Vladlen Koltun. Trellis networks for sequence modeling. In International Conference on Learning Representations, 2019b.

Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53-58, 1989.  
Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.  
Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.  
Raj Dabre and Atsushi Fujita. Recurrent stacking of layers for compact neural machine translation models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 6292-6299, 2019.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Priya Goyal, Piotr Dólár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Moritz Hardt and Tengyu Ma. Identity matters in deep learning. arXiv preprint arXiv:1611.04231, 2016.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European conference on computer vision, pp. 630-645. Springer, 2016b.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Sori-cut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations, 2020.  
Thomas Laurent and James Brecht. Deep linear networks with arbitrary loss: All local minima are global. In International conference on machine learning, pp. 2902-2907. PMLR, 2018.  
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan First, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.  
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100, 000+ questions for machine comprehension of text. In EMNLP, 2016.  
Mike Schuster and Kaisuke Nakajima. Japanese and korean voice search. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5149-5152. IEEE, 2012.  
Ohad Shamir. Exponential convergence time of gradient descent for one-dimensional deep linear neural networks. In Conference on Learning Theory, pp. 2691-2713. PMLR, 2019.  
Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, et al. Mesh-tensorflow: Deep learning for supercomputers. In Advances in Neural Information Processing Systems, pp. 10414-10423, 2018.  
TensorFlow team and contributors. TensorFlow model garden NLP. github.com/tensorflow/models/tree/master/official/nlp.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.  
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2018.  
Lei Wu, Qingcan Wang, and Chao Ma. Global convergence of gradient descent for deep linear residual networks. In Advances in Neural Information Processing Systems, pp. 13389-13398, 2019.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.  
Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. arXiv preprint arXiv:2002.04745, 2020.  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pp. 5753-5763, 2019.  
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. In International Conference on Learning Representations, 2020.
