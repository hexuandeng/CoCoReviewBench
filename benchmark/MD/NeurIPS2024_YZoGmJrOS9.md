# Can Custom Models Learn In-Context? An Exploration of Hybrid Architecture Performance on In-Context Learning Tasks

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In-Context Learning (ICL) is a phenomenon where task learning occurs through a prompt sequence without the necessity of parameter updates. ICL in Multi-Headed Attention (MHA) with absolute positional embedding has been the focus of more study than other sequence model varieties. We examine implications of architectural differences between GPT-2 and LLaMa as well as Llama and Mamba. We extend work done by Garg et al. (2022) and Park et al. (2024) to GPT-2/LLaMa hybrid and LLaMa/Mamba hybrid models - examining the interplay between sequence transformation blocks and regressive performance in-context. We note that certain architectural changes cause degraded training efficiency/ICL accuracy by converging to suboptimal predictors or converging slower. We also find certain hybrids showing optimistic performance improvements, informing potential future ICL-focused architecture modifications. Additionally, we propose the "ICL regression score", a scalar metric describing a model's whole performance on a specific task. Compute limitations impose restrictions on our architecture-space, training duration, number of training runs, function class complexity, and benchmark complexity. To foster reproducible and extensible research, we provide a typed, modular, and extensible Python package on which we run all experiments. This code is available at https://github.com/anonymousforneurips64/neurips2024-submission21757.

# 1 Introduction

Popularized by Large Language Models such as GPT-2 [1] and GPT-3 [2], In-Context Learning (ICL) is the ability for highly expressive generative sequence models to predict phenomena by processing demonstrations without performing traditional gradient steps. Such phenomena vary from effective control systems [3] to answering questions in natural language [4, 5]. A large body of recent work has studied this phenomenon in transformer models [6, 7, 2, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], which derive in structure from Vaswani et al. [26].

Some recent examples of this research on ICL include Garg et al [6], which studies ICL by providing a variety of function classes for models to learn, additionally benchmarking robustness by testing performance on out-of-distribution data. Guo et al [11] shows the validity of composing simple function classes to produce complex ones, while Liu et al [20] produced a metric for model information recall. These works give us a set of metrics with which we can use to compare model performance on ICL.

ICL was initially primarily studied in attention-based models but has recently been explored in other sequence models, creating discussion on its differences across those models and why these

Table 1: Summary of tasks. Each regression target  $f_{\theta}(x_i)$  is either parametrized by a randomly sampled  $\theta$  or directly computed/sampled as detailed above.  

<table><tr><td>Task</td><td>dim (d)</td><td>points (N)</td><td>x distribution</td><td>y calculation / parameter distribution</td><td>Task-specific</td></tr><tr><td>Linear Regression</td><td>20</td><td>41</td><td>N(0,Id)</td><td>w ~ N(0,Id)</td><td>-</td></tr><tr><td>Sparse Linear</td><td>20</td><td>41</td><td>N(0,Id)</td><td>w ~ N(0,Id), sparsity(w) ← k</td><td>k = 3</td></tr><tr><td>2-Layer MLP</td><td>20</td><td>101</td><td>N(0,Id)</td><td>W(1)ij, W(2)ij ~ N(0,1)</td><td>width = 100</td></tr><tr><td>Decision Tree</td><td>20</td><td>101</td><td>N(0,Id)</td><td>leaf ~ N(0,1), non_leaf ~ {1,...,d}</td><td>depth = 4</td></tr><tr><td>Sparse Parity</td><td>10</td><td>140</td><td>{-1,1}d</td><td>y = ∏j∈I x[j]</td><td>k = 2</td></tr><tr><td>Vector MQAR</td><td>20</td><td>128</td><td>Unif(Sd-1)</td><td>y ~ Unif(Sd-1)</td><td>-</td></tr></table>

occur architecturally. In our paper, we study this by substituting key modern transformer (Llama) components with Mamba blocks and GPT-2 components and richly benchmarking.

Since ICL for complete natural language understanding often requires training models with over a billion parameters, the effects of architectural changes on fine-grained ICL abilities are often left unexplored. As a consequence, although language models have progressed quickly and entertained radically new architectures, there is limited extensible research that explores the effects of fine-grained architecture choices on ICL ability [8, 14]. Garg et al. established using simple function classes to evaluate ICL ability and examined solely GPT-2 as a sequence model. Lee et al. [8] expanded this analysis on a slightly different set of function classes for a variety of base models. Park et al. [14] evaluated ICL performance of 2 hybrid architectures between Mamba and GPT-2. Using unmodified Llama/Mamba/GPT-2 as a control, we analyze GPT2-Llama and Llama-Mamba hybrid architectures derived from replacing portions of GPT2 components with analogous Llama sections and Llama with Mamba blocks, respectively, in 12 total architectures (3 unmodified + 9 hybrid).

We observe that the code written to analyze ICL with simple function classes – although almost unanimously extensions of Garg et al.'s – often requires substantial, structural changes to the parent codebase<sup>1</sup>, greatly heightening the barrier to extending each project in turn. Inspired by Donoho's ideal of Frictionless Reproducibility [27], we provide a set of simple abstractions and interfaces to facilitate extensions and modifications to our code while promoting interoperability between forks.

# 2 Related Work

There are many ways to capture qualitative aspects of ICL with quantitative measures. Weber et al. [17] compare the agreement between generations of a language model under varying prompts of equal meaning to test robustness to variations. Olsson et al. [22] compute a heuristic "ICL score" to measure an accuracy increase in predictions of a model given more context. We adapt this metric to fit our experimental setup more aptly, regularizing along both the number of in-context examples and against a baseline predictor.

In general, evaluating ICL ability has been approached from two primary avenues: both when the only solution at train time is to meta-learn an algorithm [6, 8, 28, 11, 19] and when optimal loss at train time can also be satisfied by memorization or otherwise leveraging previously trained-on data [10, 23]. In this work, we take the former approach through learning a regression algorithm to randomized simple function classes [6, 11, 15].

Further still, non-transformer architectures are capable of ICL [8]. Lee et al. [8] observed ICL in numerous sequence model architectures (e.g. RNNs, Mamba, S4, CNNs, GPT-2, and Llama) and found qualitative differences in each architecture's performance. Chan et al. [25] found that Transformers depend on "burstiness" and long-tail distributions of natural data to outperform RNNs and LSTMs in ICL tasks. Park et al. [14] uses simple function classes similar to Garg et al. [6] in evaluating the ICL ability of Mamba, S4, S4-Mamba, and GPT-2. They find an overlapping but inequivalent set of function classes for which each model succeeds and construct a hybrid architecture

to achieve the union of these abilities. We further this work by closely examining the contributions of individual architectural changes for GPT-2 and Llama-style transformers towards ICL ability.

# 3 Methods

As established by Garg et al. and extended by recent work, our ICL tasks take the following form [6, 8, 14]:

$$
\underbrace {x _ {0} , f _ {\theta} (x _ {0}) , x _ {1} , f _ {\theta} (x _ {1}) , \dots , \overbrace {x _ {N}} ^ {\text {q u e r y}}} _ {\text {p r o m p t} P}, \underbrace {f _ {\theta} (x _ {N})} _ {\text {c o m p l e t i o n}}
$$

where  $P$  is a series of input-output pairs followed by a lone query. The model predicts a completion based on the prompt it received. The function parameters  $\theta$  and the inputs  $x_{i}$  are randomly sampled from a function class domain and an input domain, respectively. The tasks we regress to are summarized in Table 1 and detailed in Section 3.1

We train models for ICL by minimizing the expected loss over a distribution of prompts and corresponding function outputs. This approach allows us to observe qualitative differences in model architectures by their ability to behave similarly to optimal or baseline estimators. To further simplify ICL aptitude evaluation, we introduce a proxy value summarizing a given model's ICL ability for a specific task. This metric averages the error of a model normalized by the baseline error at each context length. We detail this further in Section 3.3.

# 3.1 Training

To determine task-specific ICL ability, our sequence models regress onto the functions shown above [14]. We replicate the function classes Linear Regression, Sparse Linear Regression, 2-Layer MLP Regression, and Decision Tree Regression from Garg et al. [6] as they present a wide range of "difficulty" for sequence models. In addition, to capture the existence of some ICL ability, we also regress onto the two function classes examined in Park et al. [14]: parity function with induced sparsity (Sparse Parity) and parallel associative recall (Vector MQAR).

Unless otherwise specified, we train all models with 12 layers, 8 attention heads, an expansion factor of 4 (in the case of models with Mamba Mixer layers), and linear layers to transform the input sequences into and from the embedding dimension of 256. We use the ADAM optimizer with a learning rate of 0.0001 for 500k steps. Our expansion factor was selected to ensure similar parameter counts across baselines and all other hyperparameters were chosen for consistency with Garg et al. [6]. Note for the four function classes from Garg et al., the same curriculum was used during training. No curriculum is used for the two new function classes from Park et al. [14]. For our compute $^2$ , we utilized 898.90 hours on an A10, 55.74 hours on an RTX 3090, 151.90 hours on an RTX 4090, 75.48 hours on an RTX 4070 Ti, and 9.83 hours on an RTX 6000.

Linear Regression and Sparse Linear Regression Each function in these tasks is parametrized as a single weight vector  $(w)$  of dimension equal to that of the  $x$ -values (i.e. 20) so that  $y = w^T x$ . We sample the coordinate values from a normal distribution and (in the Sparse Linear case) zero out all values except a uniformly at random selected  $k$  coordinates. In essence, one can consider Linear Regression to be the degenerate case where the  $k = 20$ . We preserve these tasks from Garg et al. [6] to verify that none of our hybrid modifications lose the near-optimal performance that was already found with GPT-2.

2-Layer MLP Regression We fill two weight matrices  $W^{(1)} \in \mathbb{R}^{100 \times 20}$  and  $W^{(2)} \in \mathbb{R}^{1 \times 100}$  with scalar samples from a normal distribution.  $y$  values are computed as the result of a forward pass through a 2-layer multi layer perceptron with a ReLU activation. That is:  $y = W^{(2)}\mathrm{ReLU}(W^{(1)}x)$ . This is a more complex function class that Garg et al. [6] found that GPT-2 can perform very well at, suggesting that this task can capture some ICL ability of an architecture.

Decision Tree Regression We construct full decision trees of depth 4 with leaf values sampled from a normal distribution and branching conditions to be selected uniformly at random over the coordinates of the input dimension. The left branch is taken if the selected input coordinate is less than 0 and the right branch is taken otherwise. Garg et al. [6] found that GPT-2 was able to achieve much lower error for lower context lengths than XGBoost or Greedy Tree Learning, suggesting that this task can capture some ICL ability of an architecture.

Sparse Parity We select  $k = 2$  values to consider and compute their parity, expressed as either -1 or 1. That is, we uniformly sample without replacement  $\theta \sim \{1, \dots, 10\}^k$  and compute  $y = \prod_{i \in \theta} x[i]$ . Along with a higher learning rate of 0.0004, this is identical to the scheme implemented in Park et al. [14]. They [14] found that GPT-2 style transformers do not perform well on this task, suggesting that this is a discerning proxy for measuring ICL ability. Finally, as convergence was quick for this task, we only trained models up to 200k steps.

Vector MQAR We sample  $2N$  points from the  $d$ -sphere of radius  $\sqrt{d}$  and group them randomly into pairs to forming  $N$  key-value pairs. For consistency with the experiments of Park et al. [14] and to reliably allow for the formation of transformer circuits highly relevant to this task [22, 14], we reduce model complexity by using an embedding dimension of 128, 2 layers, and a higher learning rate of 0.0002. Park et al. [14] found that Mamba, our representative of SSM-type models, performed poorly, suggesting that this task can serve to ensure we don't lose capabilities provided by transformers.

<table><tr><td colspan="2">Model Variation</td><td>Pos. Emb.</td><td>FFN</td><td>Normalization</td></tr><tr><td>(1)</td><td>GPT-2</td><td>Absolute</td><td>GELU MLP</td><td>Layer Norm</td></tr><tr><td>(1.1)</td><td>GPT-2 RMS</td><td>Absolute</td><td>GELU MLP</td><td>RMS Norm</td></tr><tr><td>(1.2)</td><td>GPT-2 RoPE</td><td>RoPE</td><td>GELU MLP</td><td>Layer Norm</td></tr><tr><td>(1.3)</td><td>GPT-2 SwiGLU</td><td>Absolute</td><td>SwiGLU</td><td>Layer Norm</td></tr><tr><td>(1.4)</td><td>GPT-2 RMS SwiGLU</td><td>Absolute</td><td>SwiGLU</td><td>RMS Norm</td></tr><tr><td>(1.5)</td><td>GPT-2 RMS RoPE</td><td>RoPE</td><td>GELU MLP</td><td>RMS Norm</td></tr><tr><td>(1.6)</td><td>GPT-2 RoPE SwiGLU</td><td>RoPE</td><td>SwiGLU</td><td>Layer Norm</td></tr><tr><td>(2)</td><td>Llama</td><td>RoPE</td><td>SwiGLU</td><td>RMS Norm</td></tr><tr><td>(2.1)</td><td>Llama RoPE-less</td><td>Mamba Mixer</td><td>SwiGLU</td><td>RMS Norm</td></tr><tr><td>(2.2)</td><td>Llama SwiGLU-less</td><td>RoPE</td><td>Mamba Mixer</td><td>RMS Norm</td></tr><tr><td>(2.3)</td><td>Llama RoPE,SwiGLU-less</td><td>Mamba Mixer</td><td>Mamba Mixer</td><td>RMS Norm</td></tr><tr><td>(3)</td><td>Mamba</td><td>-</td><td>Mamba Mixer</td><td>RMS Norm</td></tr></table>

(a) For our hybrid architectures, we modify 3 types of architectural sub-blocks: positional embeddings, feed-forward network, and normalizations. We specify the sub-block alternatives used for each architecture.

![](images/739927675b6f4ad3ed1a05dda89d77bc24cc2ab07edcdaf76c642b77e79a1211.jpg)  
(b) A block diagram illustrating how each variation affects the overall architecture. Note that vertical arrows in a given block indicate that some variations skip that block entirely.  
Figure 1: Visual aid for our explored hybrid models in tabular and graphical format.

# 3.2 Architectures

As detailed by Radford et al. [1], GPT-2 is almost identical to the original decoder-only transformer, with absolute positional embedding, pre-norm layer normalization, and a GELU activation function in the feed-forward network (FFN) (which is otherwise a multi-layer perceptron). In contrast, Llama [29, 30] combines a number of modern transformer modifications, including swapping layer norm with RMS norm [31], changing the architecture and activation function of the FFN, and using rotary

Table 2: A summary of the primary architectural differences between GPT-2, Llama, and Mamba. We examine all variations between GPT-2 and Llama and all variations between Llama and Mamba.  

<table><tr><td></td><td>GPT-2</td><td>Llama</td><td>Mamba</td></tr><tr><td>Positional Embedding</td><td>Absolute</td><td>RoPE</td><td>None</td></tr><tr><td>Feed Forward Network</td><td>2 layer MLP</td><td>Convolutional MLP</td><td>None</td></tr><tr><td>Attention Mechanism</td><td>Multi-Quey Multi-Head</td><td>Multi-Quey Multi-Head</td><td>Mamba Mixer</td></tr><tr><td>Normalization</td><td>Layer Norm</td><td>RMS Norm</td><td>RMS Norm</td></tr></table>

138 positional embeddings instead of absolute positional embeddings [32]. We acknowledge that the larger variations of Llama2 [30] and both variations of Llama3 [33] used Grouped-Query Attention (GQA), however we surmise that at our model scales of  $\sim 10$  million parameters, GQA will not significantly affect the performance of our models. From an entirely different method of sequence modeling, Mamba forgoes positional embedding entirely, combining features of the Gated Linear Unit and state space expansion to remove the need for distinct attention and feed-forward blocks. We summarize these architectural differences in Table 2. We examine all combinations of these different components, training 12 total architectures (listed in Figure 1a) on our 6 tasks for a total of 72 model-task pairs. Figure 1b illustrates how each of these variations compose into a model. We provide individual diagrams of each architecture in Appendix A.

# 3.3 Evaluation

In addition to the baseline metric (squared error as a function of context length) from Garg et. al. [6], we've established another metric: ICL regression score. This is a scalar expressing overall performance of a model on a task. Abstractly, the metric aims to capture the proportion of the baseline error saved by a model. The regression score is calculated by (1) computing the difference in error achieved by the model and the zero estimator at each context length, (2) computing the average of this value over the length of the sequence, (3) computing the same value for the baseline estimator, and (4) taking the ratio of these.

In summary, ICL regression score can be calculated as follows:

$$
S _ {\text {m o d e l}} = \frac {\sum_ {i} \left(\xi_ {\text {m o d e l}} ^ {(i)} - \xi_ {0} ^ {(i)}\right)}{\sum_ {i} \left(\xi_ {\text {b a s e}} ^ {(i)} - \xi_ {0} ^ {(i)}\right)} \tag {1}
$$

where  $\xi_{\mathrm{model}}^{(i)}$  is the squared error of the model of interest at context length  $i$ . Sim.  $\xi_{\mathrm{base}}^{(i)}$  for baseline and  $\xi_0^{(i)}$  for the zero estimator

159 Summation over context length allows our ICL regression score to be used for the comparison of tasks with significantly differing context lengths. An interpretation for each of different possible values of our ICL regression score is given in 2a. This approach builds off of Olsson et al.'s "ICL Score" [22] by generalizing their selection of 500 and 50 in-context examples and reducing along the context length, allowing for tasks with widely different context lengths to be directly compared. We list our baselines in Table 2b.

We replicate the baseline predictors for linear regression, sparse linear regression, and MLP regression from Garg et al. [6] due to the lack of a higher-performing baseline. However, we opted to use a pretrained GPT-2 model with identical structure to that used in Garg et al. to serve as a more calibrated baseline than Greedy Tree Learning or XGBoost. They showed superior decision tree ICL performance for a trained GPT-2 transformer compared to Greedy Tree Learning or XGBoost. For consistency with Park et al. [14] and due to the algorithmic hardness of Sparse Parity, we used our Mamba model trained on this task. Park et al. showed that Mamba can effectively learn this task, so we repeat our strategy as in Decision Tree Regression with our Mamba model (instead of GPT-2) as a baseline.

# 3.4 Reproducibility Statement

For ease of experimentation and reproducibility, we have built a typed, extensible, and modular Python codebase. We achieved this by identifying isolated processes in the training regime and

Figure 2: Predictors and conditions for computation and interpretation of ICL regression score.  
![](images/8bf349696e660811105f04c73f7be9e4c8dc57b1d727728b3b3ff57ac639575b.jpg)  
(a) Interpretation of possible  $S_{\mathrm{model}}$  values computed over context length.

![](images/ff191fa05b703a6af1024fb6ddecc0be2adb7f977692a6b2439794c305f98b2d.jpg)  
(b) The baselines for each task. The 2-layer NN is trained for 1000 gradient steps, with a batch consisting of a randomly selected point in the context. GPT-2 and Mamba are trained for  $500\mathrm{k}$  steps on the specified task in the same format as all other models.

177 structuring our code to reflect them. In particular, the specification of (1) a function class, (2) a model type, (3) an evaluation scheme, and (4) a stage of training under a curriculum are all inherent to the experiment archetype as proposed by Garg et al. [6] and repeated by others [8, 15, 14]. We integrate standard reporting software Weights and Biases [34] and leverage fast implementations of attention [35] and 1-D convolutions [36]. We also implement a configuration-based system for training, loading, and evaluating models to facilitate frictionless repeatability of all experiments.

# 4 Results

We confirm the results from Garg et al. [6] and Park et al. [14] that GPT-2 and Mamba can learn our first four regression tasks in context. Park et al. [14] that Mamba struggles to perform Vector MQAR while transformers and hybrid architectures excel. We note that Llama and GPT-2 have very comparable performance in Sparse Parity and Vector MQAR. We plot all qualitatively non-optimal squared error profiles in Figure 3 and all squared error profiles in Appendix B.

Figure 3: Squared error profiles that do not exhibit near-optimal behavior. Shaded regions are  $99\%$  confidence intervals.  
![](images/a6e726e185c72861e0560bd7af407b69e77d172ab6b6631610089db3b496c769.jpg)  
(a) Notable phenomena for Sparse Linear. We observe that while GPT-2 (orange) performs very similarly to our baseline, adding RMS norm without RoPE (red and green) leads to models performing notably worse than optimal.

![](images/d8dc9bff2ab0b83279c7b4c53ca4f0fd9464fdef814fc5c86f829065971546eb.jpg)  
(b) Notable phenomena for Decision Tree. We note that Mamba (green) performs somewhat suboptimally while GPT-2 RMS (orange) fails to learn the task entirely.

189 Models can converge to suboptimal regression schemes. We find that some model-task pairs 190 produce suboptimal predictions, not as a result of insufficient training. A clear example is GPT-2 191 RMS SwiGLU (model 1.4) on Sparse Linear. This model appears to not achieve optimal error 192 - achieving an ICL Regression Score of only 0.754, opposed to  $\sim 0.93$  by other models - and yet 193 its performance does not significantly improve with more gradient steps. We plot the squared error 194 achieved by various checkpoints for model 1.4 in Figure 4a. We observe that this error profile appears 195 similar to that of models trained on the Linear task and so also examine the prediction quality of the

![](images/70afb6561f1577f2dbaaa215e70f6b51f7bff7d4d33463c6044f85e24e88951d.jpg)  
(a) GPT-2 RMS SwiGLU Checkpoints on Sparse Linear. We see that GPT-2 RMS SwiGLU converges to the least squares solution, despite Lasso being the optimal solution. This suggests that GPT-2 RMS SwiGLU fails to learn to utilize its context to its fullest extent.

![](images/da987fbec090215eb62cc2403a32490af3771d9a7ff5fc5112b332dd6879b40c.jpg)  
Figure 4: Detailing plots to showcase GPT-2 RMS SwiGLU (model 1.4) learning a more general but sub-optimal regression scheme when trained on Sparse Linear. Shaded regions are  $99\%$  confidence intervals.  
(b) GPT-2 RMS SwiGLU trained on Sparse Linear and evaluated on Linear. When evaluated on a similar task to which it was trained on, GPT-2 RMS SwiGLU appears to perform better than its siblings, despite the fact that it performed worse than its siblings on its original task! This suggests that it learned a different regression scheme than GPT-2 on the same training data.

same model (GPT-2 RMS SwiGLU trained on Sparse Linear) on Linear in Figure 4b. We find that it indeed mimics the error profile of least squares. This result builds on Akyurek et al.'s findings [19] in what functions transformer models develop representations of. Akyurek et al. analyzed algorithms representable by GPT-2 like architectures. We note that they did not examine other layer types such as Mamba Mixer or SwiGLU.

Models can escape suboptimal regression schemes. We see that GPT-2 SwiGLU (model 1.3) Sparse Linear on adopts a suboptimal regression scheme (least squares) partway in training, eventually unlearning its scheme in favor of the optimal regression scheme (lasso). We plot the squared error on Sparse Linear achieved by various checkpoints for Model 1.3 in Figure 5a, noting that the error of the checkpoint at  $100\mathrm{k}$  steps closely matches the error of least squares. Further, we examine the squared errors on Linear Regression for the various checkpoints for Model 1.3 in 5b and see that the checkpoint at  $100\mathrm{k}$  most closely matches least squares. This suggests that model 1.3 learned the linear regression scheme in the beginning of training, but was eventually able to learn to utilize the sparse nature of its training data.

Models can fail to converge within our training horizon. We find that a number of models performed strikingly poorly in their trained task. In particular, GPT-2 with Layer norm replaced by RMS norm (model 1.1) performed very poorly on Sparse Linear Regression and Decision Tree, as indicated by the lowest ICL Regression Score achieved in those tasks (0.535 and 0.114, respectively) and in Figures 3a and 3b. We also observe that GPT-2 with RMS and SwiGLU (model 1.4) also did not converge to a regression scheme, despite apparently modelling a different regression scheme entirely. Similarly, Mamba (model 3) did not converge to a training scheme on Decision Tree as illustrated in Figure 6a. We believe this suggests a lower training efficiency for certain architectures on these tasks.

Models can fail to learn the task entirely. In the case of Decision Tree, GPT-2 with RMS (model 1.1) failed to learn the task entirely as not only indicated by its final ICL Regression Score but also its consistency in achieving very high error throughout training. We plot squared error for various checkpoints in Figure 6b.

ICL Regression Scores reflect qualitative information contained in squared-error plots. Computed ICL Regression Scores are summarized in Table 3. Overall, most models are able to perform comparably to our baseline estimators, with nearly all examined models achieving a regression score of approximately 1 on all four function classes from Garg et al. (Linear Regression, Sparse Linear Regression, 2-Layer MLP, Decision Tree). The ICL Regression Scores for Linear

![](images/17392a5ec85e20417b4d97edad2cd499abe8b8eefdef3e7899ac624aa8aee33d.jpg)  
(a) GPT-2 SwiGLU Checkpoints on Sparse Linear. In the beginning of training, GPT-2 SwiGLU quickly converges to least squares, but it is able to escape this regression scheme and eventually has its error profile approach that of Lasso.

![](images/e577b6855a7149c27daf3d8ff8d09e337e3c9e6ae3e88f335d948e65ed426b6c.jpg)  
(b) GPT-2 SwiGLU Checkpoints trained on Sparse Parity and evaluated on Linear Regression. We see that an earlier checkpoint (100k) of GPT-2 SwiGLU outperforms later checkpoints on a similar task different from the task it was trained on.

![](images/5af002ce04a6978c725a3ec10ee4cff3ebf74a967bdc0f992cfdda476b5439d5.jpg)  
Figure 5: Detailing plots to showcase GPT-2 SwiGLU (model 1.3) starting by learning a more general but sub-optimal regression scheme but eventually converging to the optimal regression scheme when trained on Sparse Linear. Shaded regions are  $99\%$  confidence intervals.

![](images/cddb6c66e6f4060d8c35f5bda9d107febecd02b19bb862eb45509c4a3a5857a1.jpg)  
(a) Mamba Checkpoints on Decision Tree. We see that Mamba does keep improving its error profile throughout training. This suggests that Mamba did not reach convergence, and thus has lower training efficiency on this task.  
Figure 6: Squared error as a function of context length computed for various checkpoints for both Mamba (model 3) and GPT-2 RMS (model 1.1) on Decision Tree. Shaded regions are  $99\%$  confidence intervals.  
(b) GPT-2 RMS Checkpoints on Decision Tree. We see that all checkpoints of GPT-2 perform very similarly, with little to no change in error profile throughout training.

Regression and 2-Layer MLP, along with their corresponding graphs of squared error as a function of context length, corroborate the claims from Garg et al. [6] that transformers can "learn" these tasks. Further, the ICL Regression Scores for Sparse Parity are consistent with Park et al. [14], with all hybrids between GPT-2, and Llama failing to "learn" the task and all hybrids between Llama and Mamba succeeding in "learning" the task. Indeed, the ICL Regression Score achieved by Mamba captures the qualitatively sub-optimal performance detailed above on Decision Tree.

# 5 Discussion

Even simple function classes leave room for local minima. We find that despite distilling down the phenomenon of In Context Learning to regression against simple function classes, there still exists room for models to adopt various regression schemes. This is supported by the apparent convergence

Model  
Table 3: ICL Regression Scores for each architecture on each task, averaged over many sampled functions, with  $95\%$  confidence intervals in the headers for each row. Best-in-task values are in boldface except when not statistically significant from another architecture. GPT-2/Llama hybrids were not evaluated on Sparse Parity due to compute constraints and lack of supporting evidence that they should succeed. *These models were used as the baseline for this task.  
Linear  $\neq 0.001$  
Paree Linear  
aee  
eannnnnne nnnnne eannnnnne  
Paree Pany  

<table><tr><td>(1)</td><td>GPT-2</td><td>0.996</td><td>0.932</td><td>1.130</td><td>1.000*</td><td>0.023</td></tr><tr><td>(1.1)</td><td>GPT-2 RMS</td><td>0.997</td><td>0.535</td><td>1.130</td><td>0.114</td><td>-</td></tr><tr><td>(1.2)</td><td>GPT-2 RoPE</td><td>0.995</td><td>0.927</td><td>1.130</td><td>1.004</td><td>-</td></tr><tr><td>(1.3)</td><td>GPT-2 SwiGLU</td><td>0.997</td><td>0.913</td><td>1.128</td><td>0.994</td><td>-</td></tr><tr><td>(1.4)</td><td>GPT-2 RMS SwiGLU</td><td>0.997</td><td>0.754</td><td>1.129</td><td>0.971</td><td>-</td></tr><tr><td>(1.5)</td><td>GPT-2 RMS RoPE</td><td>0.996</td><td>0.927</td><td>1.128</td><td>1.005</td><td>-</td></tr><tr><td>(1.6)</td><td>GPT-2 RoPE SwiGLU</td><td>0.996</td><td>0.929</td><td>1.129</td><td>1.011</td><td>-</td></tr><tr><td>(2)</td><td>Llama</td><td>0.997</td><td>0.933</td><td>1.129</td><td>1.007</td><td>0.023</td></tr><tr><td>(2.1)</td><td>Llama RoPE-less</td><td>0.996</td><td>0.928</td><td>1.130</td><td>1.018</td><td>1.000</td></tr><tr><td>(2.2)</td><td>Llama SwiGLU-less</td><td>0.996</td><td>0.927</td><td>1.129</td><td>0.980</td><td>1.000</td></tr><tr><td>(2.3)</td><td>Llama RoPE,SwiGLU-less</td><td>0.996</td><td>0.938</td><td>1.130</td><td>1.012</td><td>1.000</td></tr><tr><td>(3)</td><td>Mamba</td><td>0.995</td><td>0.925</td><td>1.123</td><td>0.832</td><td>1.000*</td></tr></table>

of the error profiles of GPT-2 RMS (model 1.1) and GPT-2 RMS SwiGLU (model 1.4) to least squares regression for shorter context lengths.

Hybrid architectures and function classes have varying levels of compatibility. Specific hybrid architectures can hesitate to learn/converge for certain function classes. This behavior is especially apparent in GPT-2 RMS's (model 1.1) Decision Tree error graph and GPT-2 RMS SwiGLU's (model 1.4) Sparse Linear performance. It seems that GPT-2 RMS SwiGLU shows greater affinity towards learning least squares instead of LASSO. Certain hybrid architecture variations may place inductive biases on certain solution forms, resulting in extreme convergence times when these solution forms greatly vary from the optimal predictor's form.

Extensible Research as Reproducible Research. In the development of this work, continuously iterating to minimize the friction of reproduction has enabled rapid extension of our Python artifacts to support even abstractly defined hybrid architectures, which are often considered inextricable from highly bespoke code or dedicated packages such as xFormers [37]. We implore the reader to seriously consider the value of making their research extensible with a minimum of friction. We hope that our attempts to maximize extensibility and reproducibility contribute to the longevity of this work as a reliable, tested, and simple framework to use for studying simple function classes in context.

# 5.1 Limitations and Future Work

We have only one training run performed on each model-task pair. As a result, we have no estimation for how consistently observed phenomena appear with the given architectures. We only train each model for a maximum of 500K steps. Thus, when a model fails to converge within this window, we lose information on insightful trends that could possibly occur with further training.

We do not empirically evaluate the effectiveness of ICL Regression Score or the usability of our provided code platform. We compute no verifying metrics to establish how well ICL Regression Score generalizes or is robust to qualitatively distinct ICL regression tasks. Similarly, we perform no user study on the effectiveness of our code platform, presenting only our own experience.

Future Work In this paper we analyze ICL performance for GPT-2-Llama and Llama-Mamba hybrid architectures (9 total) on 6 tasks. Future relevant research could entail 1) expanding our architecture-space and streamlining our training-to-evaluation pipeline by creating an architecture search mechanism, 2) assessing our models on other sets of tasks, such as ones relating to language modeling or image classification, 3) verifying our results with additional training runs, 4) benchmarking model performance along hardware-related metrics.

# References

[1] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.  
[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
[3] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. CoRR, abs/2106.01345, 2021.  
[4] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022.  
[5] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.  
[6] Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583-30598, 2022.  
[7] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.  
[8] Ivan Lee, Nan Jiang, and Taylor Berg-Kirkpatrick. Is attention required for icl? exploring the relationship between model architecture and in-context learning ability, 2023.  
[9] Cem Anil, Esin Durmus, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Nina Rimsky, Meg Tong, Jesse Mu, Daniel Ford, et al. Many-shot jailbreaking.  
[10] Aaditya Singh, Stephanie Chan, Ted Moskovitz, Erin Grant, Andrew Saxe, and Felix Hill. The transient nature of emergent in-context learning in transformers. Advances in Neural Information Processing Systems, 36, 2024.  
[11] Tianyu Guo, Wei Hu, Song Mei, Huan Wang, Caiming Xiong, Silvio Savarese, and Yu Bai. How do transformers learn in-context beyond simple functions? a case study on learning with representations. arXiv preprint arXiv:2310.10616, 2023.  
[12] Eric Todd, Millicent L Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau. Function vectors in large language models. arXiv preprint arXiv:2310.15213, 2023.  
[13] Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems, 36, 2024.  
[14] Jongho Park, Jaeseung Park, Zheyang Xiong, Nayoung Lee, Jaewoong Cho, Samet Oymak, Kangwook Lee, and Dimitris Papailiopoulos. Can mamba learn how to learn? a comparative study on in-context learning tasks. arXiv preprint arXiv:2402.04248, 2024.  
[15] Kartik Ahuja and David Lopez-Paz. A closer look at in-context learning under distribution shifts. arXiv preprint arXiv:2305.16704, 2023.  
[16] Ekin Akyurek, Bailin Wang, Yoon Kim, and Jacob Andreas. In-context language learning: Architectures and algorithms. arXiv preprint arXiv:2401.12973, 2024.  
[17] Lucas Weber, Elia Bruni, and Dieuwke Hupkes. The ict consistency test. arXiv preprint arXiv:2312.04945, 2023.

[18] Noam Wies, Yoav Levine, and Amnon Shashua. The learnability of in-context learning. Advances in Neural Information Processing Systems, 36, 2024.  
[19] Ekin Akyurek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models, 2023.  
[20] Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804, 2021.  
[21] Jerry Wei, Jason Wei, Yi Tay, Dustin Tran, Albert Webson, Yifeng Lu, Xinyun Chen, Hanxiao Liu, Da Huang, Denny Zhou, et al. Larger language models do in-context learning differently. arXiv preprint arXiv:2303.03846, 2023.  
[22] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.  
[23] Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference, 2022.  
[24] Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mord-vintsev, Andrey Zhmoginov, and Max Vlademyrov. Transformers learn in-context by gradient descent, 2023.  
[25] Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya Singh, Pierre H. Richemond, Jay McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers, 2022.  
[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.  
[27] David Donoho. Data science at the singularity. arXiv preprint arXiv:2310.00865, 2023.  
[28] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Exposing attention glitches with flip-flop language modeling, 2023.  
[29] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.  
[30] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.  
[31] Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019.  
[32] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.  
[33] AI@Meta. Llama 3 model card. 2024.

[34] Lukas Biewald. Experiment tracking with weights and biases, 2020. Software available from wandb.com.  
[35] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning, 2023.  
[36] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023.  
[37] Benjamin Lefaudeau, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022.

![](images/2b9cf1fa9119036c79ec186ffafecdf1dcf65c369f05185fdfc52f9b3f7e8158.jpg)  
(a) The GPT-2 Architecture

![](images/438321bf257723b3a8995462b300b189354dcf61ffa9fcd31828b3b33d169762.jpg)  
(b) The Llama Architecture

![](images/f6897c476247d0391de4681822b64c0127d6301bbdbfe70fee3004765e03ed2e.jpg)  
(c) The Mamba architecture

![](images/3eac1b935f68b6ff4a5bce5c5603d8f7bbefcc6847d7a485a6128a04341e09bd.jpg)  
Figure 7: The GPT-2, Llama, and Mamba architectures used in our regression tasks  
(a) Llama with the feed-forward block replaced by a Mamba Mixer block  
Figure 8: The hybrid architectures as modifications to Llama

![](images/1b31d96851ec7bca292d3f61af443f5a1ea58555979fcfe321fc3a0e67219bae.jpg)  
(b) Llama with rope embeddings removed and a Mamba Mixer pretended to serve as a "positional embedder"

![](images/2f3528c7dfe861cf282e77918347a048876549b1c4c606bfd67e362eaec9ade0.jpg)  
(c) Llama with the feed-forward block replaced by a Mamba Mixer block, rope embeddings removed, and a Mamba Mixer pretended to serve as a "positional embedder"

(b) GPT-2 with the absolute positional encodings removed and  
![](images/441053408be28ee47394f60ec378a43e9858aa9049eed741129d14ca83e529c7.jpg)  
(a) GPT-2 with the GELU MLP rotary position embeddings in-(c) GPT-2 with the Layer Norm replaced by a SwiGLU cluded in attention replaced by an RMS Norm

![](images/2f02834c1bd33108611ea30cc7166b05d2f955eef6ee39fa639efb38c86af98f.jpg)

![](images/59ad0151317ffbad7c0c57bedb4c5da8d5a5b3b86ddbc187bf0c4be30d1c682e.jpg)

Figure 9: The hybrid architectures as modifications to GPT-2  
![](images/9eb6f8cbfb8750fa7363ce23ac460de7d4f070719d3cb66ef9bca76d13ddeac2.jpg)  
(e) GPT-2 with absolute posi-(f) GPT-2 with the GELU MLP re-(d) GPT-2 with the GELU MLP tional encodings removed, rotary placed by a SwiGLU, absolute po-. replaced by a SwiGLU and the position embeddings included in sitional encodings removed, and Layer Norm replaced by an RMS attention, and the Layer Norm re-rotary position embeddings inNorm placed by an RMS Norm included in attention

![](images/64e6a3b0a2c3731c05e1bb3c1ae483dfcfb4b7a2535bb1e3b0ec44599c69a45b.jpg)

![](images/4f0dcaf988fca83147ba67f9db3ca2ab92f24d7c03b3a6e5993cb16c1c781fdf.jpg)
