# HARDWARE-AWARE PARALLEL PROMPT DECODING FOR MEMORY-EFFICIENT ACCELERATION OF LLM INFERENCE

Anonymous authors

Paper under double-blind review

# ABSTRACT

The auto-regressive decoding of Large Language Models (LLMs) results in significant overheads in their hardware performance. While recent research has investigated various speculative decoding techniques for multi-token generation, these efforts have primarily focused on improving processing speed such as throughput. Crucially, they often neglect other metrics essential for real-life deployments, such as memory consumption and training cost. To overcome these limitations, we propose a novel parallel prompt decoding that requires only  $0.0002\%$  trainable parameters, enabling efficient training on a single A100-40GB GPU in just 16 hours. Inspired by the human natural language generation process, PPD approximates outputs generated at future timesteps in parallel by using multiple prompt tokens. This approach partially recovers the missing conditional dependency information necessary for multi-token generation, resulting in up to a  $28\%$  higher acceptance rate for long-range predictions. Furthermore, we present a hardware-aware two-stage tree pruning algorithm that adaptively optimizes this decoding scheme to fully leverage the computational capacities on different GPUs. Through extensive experiments across LLMs ranging from MobileLlama to Vicuna-13B on a wide range of benchmarks, our approach demonstrates up to  $2.49 \times$  speedup and maintains a minimal runtime memory overhead of just  $0.0004\%$ . More importantly, our parallel prompt decoding can serve as an orthogonal optimization for synergistic integration with existing speculative decoding, showing up to  $1.22 \times$  further speed improvement. Our code will be open-sourced upon acceptance of the paper.

# 1 INTRODUCTION

The recent advances in large language models (LLMs) are increasingly shaping and influencing a wide range of AI applications. However, autoregressive generation, the de facto approach employed in LLM inference, suffers from inadequate hardware performance due to its inherent sequential nature (Stern et al., 2018). Speculative decoding (Leviathan et al., 2023; Chen et al., 2023; Kim et al., 2024), an emerging acceleration technique, employs a guess-and-verify framework for LLM inference, where a smaller draft model first predicts multiple tokens sequentially and then the original

LLM verifies them in parallel. Despite its potential, the effectiveness of speculative decoding is limited by the complexity and cost of training a draft model capable of consistently achieving high acceptance rates across diverse base models and datasets. Additionally, the extra runtime memory overhead for executing draft models poses a significant barrier to the broader adoption of speculative decoding, particularly in edge and mobile environments where memory capacity is limited. Considering the growing need for user privacy and personalization, deploying LLMs on devices urges a more memory- and cost-efficient solution for accelerating LLM inference. Recent efforts have explored the possibility of generating multiple tokens in parallel without relying on a separate transformer

![](images/0f59cf7c2459ee49ce9847dd43c3346c7f3d6efcf01371656397f76cca9b1dc4.jpg)  
Figure 1: Comparison of memory, speedup, and training cost on MT-Bench with Vicuna7B. Circle diameter shows training GPU hours.

![](images/194eeb8807c9d0aa22f189332961fb3f232532293cf78ea1889001d7da0879b4.jpg)  
Figure 2: Overview of PPD. The left section shows the location of trainable parameters and the middle section displays the combined guess-and-verify process during inference. The "prompt token" denotes the special token with separately trained embeddings to perform parallel prediction.

draft model (Santilli et al., 2023). Approaches such as inserting additional decoding heads (Cai et al., 2024) and retrieving frequently used tokens (He et al., 2023) are employed to enhance performance. However, these methods either aggressively assume conditional independence among the tokens generated in a single step (Cai et al., 2024; He et al., 2023), or use placeholder tokens (e.g., [PAD] token) that do not convey enough contextual information (Santilli et al., 2023). Therefore, they often suffer from low acceptance rates or degradation in output quality due to the lack of sufficient conditional information during inference.

To alleviate the complexity and overhead associated with the use of draft models while maintaining a high acceptance rate, we propose Parallel Prompt Decoding (PPD), a novel architecture-agnostic and memory-efficient framework that adopts prompt tuning for non-autoregressive LLM inference. Inspired by the human natural language generation process where continuous words like common expressions and phrases are produced simultaneously, PPD introduces the use of prompt tokens, the meticulously trained embeddings, for multi-token prediction. Specifically, these trained prompt tokens are appended to the original input sequence in parallel, enabling the concurrent generation of multiple output tokens in a single forward pass. The key intuition of PPD lies in the observation that if trained properly, prompt tokens appended to the input can approximate tokens generated at future timesteps, thereby partially recovering the missing conditional dependency information for multi-token generation. By strategically positioning trained prompt tokens, PPD achieves up to a  $28\%$  higher acceptance rate when predicting long-range tokens. To further increase the token acceptance rate, we generate multiple candidate continuations with each prompt token and use them in combination with a customized tree attention mask to minimize the computation and memory overhead. The capability of PPD to use low-cost prompt tokens for accurate multi-token prediction forms the foundation for accelerating LLM inference. As shown in Figure 1, PPD achieves a comparable speedup to the state-of-the-art speculative decoding approaches with negligible memory overhead and reduced training cost. Moreover, to facilitate the optimized implementation of PPD across different hardware platforms, we propose a hardware-aware two-stage tree pruning technique that adaptively refines the prompt structure during runtime based on the computational resources available on the specific hardware.

To demonstrate the effectiveness of our approach, we evaluate  $PPD$  on MobileLLaMA (Chu et al., 2023), Vicuna-7b and Vicuna-13b (Chiang et al., 2023). Running on a single GPU using the A100-40GB and RTX 4090, our method achieves a speedup ratio for inference from  $2.12 \times$  to  $2.49 \times$  across a diverse range of popular datasets including MT-Bench, HumanEval, and GSM8K. Our experiments demonstrate that  $PPD$  not only achieves comparable throughput to the state-of-the-art speculative decoding method, but it also manages this with significantly fewer trainable parameters—specifically,  $0.0002\%$  of trainable parameters—and incurs only a minimal memory overhead ( $0.0004\%$ ), showcasing that  $PPD$  is remarkably cost- and memory-efficient. The training of prompt tokens can be completed in 16 hours using one A100 GPU, 8 hours using four GeForce RTX 3090 GPUs, compared to the 1-2 days on four A100 GPUs required for Eagle (Li et al., 2024a). Furthermore, since  $PPD$  does not require the modification of the original LLM or the addition of

extra networks, it is highly adaptable and orthogonal to other decoding techniques. For instance, it can be effectively combined with a draft model to further reduce inference latency.

Our contributions are summarized as follows:

- A novel Parallel Prompt Decoding (PPD) that adopts cost-effective prompt tokens for non-autoregressive LLM inference, achieving a high acceptance rate for long-distance token prediction with preserved output quality.  
- A hardware-aware two-stage tree pruning technique that adaptively optimizes the prompt structure of  $PPD$  at runtime based on the available compute and memory resources, facilitating its efficient deployment on various hardware platforms.  
- An open-source implementation of PPD, accompanied by comprehensive evaluations on various models and benchmarks. Our experiments demonstrate that PPD achieves significant speed improvements with negligible memory overhead and reduced training cost.

# 2 BACKGROUND AND RELATED WORK

To enhance the inference speed of LLM, various approaches adopt an iterative guess-and-verify strategy to enable multi-token generation. In the guessing phase, potential future tokens are proposed at a faster speed than in traditional autoregressive implementations. Subsequently, a parallelized verification process assesses which guessed tokens should be accepted. Depending on how tokens are generated during the guess stage, these approaches can generally be categorized as  $i$  ) speculative decoding and ii) parallel decoding.

# 2.1 SPECULATIVE DECODING

The guessing phase of speculative decoding adopts a lightweight draft model to generate multiple tokens at an increased speed (Kim et al., 2024). During the verification stage, the original LLM subsequently determines the acceptance of the guessed tokens. It is worth noting that both draft and original models still follow the auto-regressive inference scheme. The speedup comes from two factors:  $i$  ) the draft model runs much faster than the original model and more tokens can be generated within the same time unit; and  $ii$  ) token verification is executed concurrently, either by batching or by incorporating multiple candidates into a single input using customized sparse attention masks (Miao et al., 2024). Therefore, the overall speedup depends on the acceptance rate and the inference latency of draft models.

Building on the speculative decoding scheme, various studies have been conducted to further optimize its inference speed. To improve the accuracy of the draft model, Eagle (Li et al., 2024a) incorporates the hidden features into the draft model's forward pass. Recently, Eagle-2 (Li et al., 2024b) enhances their approach using a context-aware dynamic tree construction. However, both Eagle and Eagle-2 utilize a separate draft model for multi-token generation, diverging fundamentally from our prompt decoding approach. Moreover, their dynamic tree construction scheme is an orthogonal technique to our two-stage tree pruning method. SpecInfer (Miao et al., 2024) adopts a tree-based speculative inference and verification scheme, improving the diversity of speculation candidates. Sequoia (Chen et al., 2024) optimizes the sparse tree structure of speculative decoding by considering the capability of the underlying hardware platforms. Our tree pruning algorithm differs from Sequoia by accounting for two types of tokens in the tree: prompt tokens and guess tokens, whereas Sequoia only considers guess tokens. Furthermore, their methods require the storage and maintenance of a separate draft model, and there is extra complexity in designing an efficient draft model.

# 2.2 PARALLEL DECODING

To overcome the inherent limitations of autoregressive inference and the memory overhead associated with using a separate draft model, several attempts have been made to integrate both guessing and verification using one unified model. Medusa<sup>1</sup> (Cai et al., 2024) introduces language model (LM) heads at the final layer of the original LLM, facilitating the generation of multiple tokens in a single

forward pass. It also utilizes tree attention masks in its verification process to increase speed even further. To enhance token drafting with retrieval-augmented generation (Karpukhin et al., 2020), Rest (He et al., 2023) introduce retrieval-based decoding tailored for specific scenarios. Inspired by Jacobi decoding (Santilli et al., 2023) that adopts multiple special tokens to accelerate machine translation, Lookahead Decoding (Fu et al., 2024) improves upon this method by generating parallel n-grams and employing a caching memory pool. To capture more information while using multiple special tokens at distinct positions, PaSS (Monea et al., 2023) trains additional tokens with embedding layers for parallel decoding. Hierarchical parallel decoding (Liu et al., 2024) introduces the use of [Fork] and [Join] tokens, enabling parallel execution of multiple structural subroutines.

Our approach can be categorized as parallel decoding, with two novel features to distinguish it from other approaches: 1)  $PPD$  trains the embeddings of parameterized ensemble prompt tokens, 2) it utilizes a hardware-aware two-stage tree pruning algorithm for designing a sparse tree tailored to each hardware platform.

# 3 PARALLEL PROMPT DECODING (PPD)

The primary advantage of PPD lies in training embeddings for prompt tokens rather than developing a separate model. Our method integrates three substeps into a single decoding step, following the guess-and-verify strategy: (1) candidate generation, where multiple candidate continuations<sup>2</sup> are predicted by strategically inserting the prompt tokens into the input sequence. We adopt tree attention (Miao et al., 2024) to merge the processing of multiple candidates into a single forward pass; (2) candidate verification, where two verification schemes, exact matching (Fu et al., 2024) and typical acceptance (Cai et al., 2024), are implemented; (3) candidate acceptance, where validated candidates are integrated into the input and KV cache is updated accordingly. Figure 2 presents the inference scheme of combining generation and verification steps in a single forward pass.

# 3.1 PROMPTTOKENS

The prompt tokens are the key component of  $PPD$  to realize multi-token generation. Initially introduced by Lester et al. (2021) to adapt LLMs for specific tasks, prompt tokens are typically prepended to the input, with outputs generated in an autoregressive manner. In this work, we propose a novel approach of utilizing prompt tokens by strategically positioning them at locations where tokens are anticipated to be generated in parallel.

In the standard decoding process, the probability of predicting the next token is expressed as the conditional probability  $p(y_{i+1} | x, y_{1:i})$ , where  $x$  is the input prompt,  $y_{1:i}$  are the  $i$  tokens generated so far, and  $y_{i+1}$  is the next token to be predicted. For conventional parallel decoding techniques (Stern et al., 2018; Cai et al., 2024) that presume complete conditional independence among tokens decoded in a single step, the exact conditional probability is approximated by

$$
p \left(y _ {i + k + 1} \mid x, y _ {1: i + k}\right) = p _ {\theta} \left(y _ {i + k + 1} \mid x, y _ {1: i}\right)
$$

where  $k > 0$  indicates the token distance. However, we observe that as  $k$  increases, the gap between the actual probability and its approximation expands, primarily due to the absence of relevant conditional dependencies. We argue that prompt tokens can bridge this gap by more accurately modeling the conditional probability as

$$
p \left(y _ {i + k + 1} \mid x, y _ {1: i + k}\right) = p _ {\theta} \left(y _ {i + k + 1} \mid x, y _ {1: i}, t _ {i + 1: i + k}\right)
$$

where  $t_i$  is the prompt token with token distance  $i$ . Through this forward pass in the decoder layers, these causally linked prompt tokens facilitate the flow of information along the sequence of speculative tokens, thus restoring the conditional probability. We demonstrate the effectiveness of this approach in Section 5.2.

# 3.2 ENSEMBLE PROMPT TAXENS

Inspired by prompt ensembling (Lester et al., 2021), which uses multiple prompts to generate diverse responses and aggregates these to derive a single answer, we introduce the concept of ensemble prompt token (EPT). This additional abstraction allows us to decouple each prompt token from the fixed embedding dimension. For every prompt token, there exist multiple corresponding EPTs, each with its distinct embedding. We modify the attention mask to ensure that each  $n^{\text{th}}$  EPT only depends on the corresponding  $n^{\text{th}}$  EPTs from preceding prompt tokens. This selective visibility is maintained for both training and inference, where the guess token for each prompt token is determined by averaging the logits of its EPTs. The use of EPTs not only enables direct and flexible control over the trainable parameters, but also leads to an increase in prediction accuracy. The probability is approximated as  $\frac{1}{n} \sum_{j=1}^{n} p_{\theta}(y_{i+k+1}|x,y_{1:i},v_{i+1:i+k}^{j})$ , where  $v_{i+m}^{j}$  denotes the  $j^{\text{th}}$  EPT at a token distance of  $m$ . Further details about EPTs can be found in Appendix D.

# 3.3 TRAINING

During training, only the embeddings of prompt tokens are changed, with the parameters of the original LLM remaining frozen. We adopt the following two training techniques:

Random Insertion of Prompt Tokens: Randomly inserting prompt tokens throughout the input sequence reduces contextual bias from appending them only at the end. This approach broadens the predictive capacity of prompt tokens beyond a limited vocabulary such as  $<\infty >$  and punctuation.

Knowledge Distillation: To align the predictive behavior of prompt tokens with the original LLM, we employ knowledge distillation. Instead of using hard labels, prompt tokens are trained against the logits produced by the original LLM. Following Medusa (Cai et al., 2024), the loss function is formulated as:

$$
L _ {P D} = \frac {1}{N} \sum_ {i = 1} ^ {N} D _ {K L} \left(P _ {i} \| Q _ {i}\right) \cdot \alpha^ {i - 1}, \tag {1}
$$

where  $D_{KL}$  is the KL divergence,  $P_{i}$  is the predicted distribution of the  $i^{\mathrm{th}}$  prompt token,  $Q_{i}$  is the corresponding distribution from the original LLM, and  $\alpha$  is the decay ratio.

# 4 SPARSE TREE PRUNING

# 4.1 CUSTOMIZED SPARSE TREE ATTENTION

Tree attention, introduced by SpecInfer (Miao et al., 2024), increases the expected acceptance rate by considering the top-k candidates from a single decoding step. In their approach, the input is structured as a tree, where each level of the tree corresponds to a specific output position. An attention mask is applied to the tree-structured input, allowing the model to process multiple candidates efficiently without increasing the batch size.

To improve the efficiency and performance of LLM inference, this paper proposes a novel sparse tree customized for  $PPD$ , which prioritizes candidates in the tree structure with higher prediction accuracy. A key difference from previous works (Cai et al., 2024; Chen et al., 2024) is the appending of a sequence of prompt tokens to each guess token. The length of the prompt token sequence decides the maximum depth of the speculative tree at the next decoding step. To further hide the latency introduced by the extra prompt tokens, we propose a novel tree pruning algorithm (Section 4.2) that optimizes the number of prompt tokens at each guess token.

# 4.2 TWO-STAGE TREE PRUNING ALGORITHM

As depicted in Figure 3, our tree pruning algorithm consists of two stages: an offline static tree pruning phase and an online hardware-aware tree optimization phase. These two stages are applied subsequently to reduce the amount of computation involved in PPD multi-token generation.

![](images/5aa90c76eb644c00cb8b14b28a3141b68ba5045388e0ff56ee1c0c5f46f38c52.jpg)  
Figure 3: Illustration of Tree Pruning Pipeline. The tree structure is optimized as a result of pruning.

![](images/3850ab97628c425cf5696f8ce379b38904c039be0e8d7d671f1163626251b2d3.jpg)

# 4.2.1 STATIC TREE PRUNING

The first stage, static tree pruning, is applied offline prior to runtime deployment. The goal is to reduce the number of prompt tokens in the tree to achieve the desired tree size. As shown on the left side of Figure 3, the tree pruning process consists of three key steps:

1. Candidate Trees Construction: Building trees using only candidate tokens at varying depths, employing the algorithm from Medusa (Cai et al., 2024) and Sequoia (Chen et al., 2024) to maximize  $f(T_k)$ .  
2. Prompt Tokens Appending : Attaching the maximum allowable prompt tokens to each candidate token from the first step.  
3. Greedy Prompt Token Removal: Removing a prompt token greedily to maximize expected amortized acceptance lengths, continuing until the desired prompt token budget is reached.

Each guess token in the tree is appended with a sequence of prompt tokens, with each prompt token corresponding to a unique output position. The length of this sequence determines the tree's maximum depth at the next decoding step. Thus, removing a prompt token at a guess token reduces the maximum tree depth at the next decoding step if this guess token is accepted in the current step. Let  $p_c$  represent the acceptance probability of guess token  $c$ , and  $f_d$  denote the expected acceptance length with  $d$  prompt tokens before removal. The decrease in expected acceptance length,  $\Delta F$ , due to removing a prompt token at  $c$  is given by  $\Delta F = p_c \cdot (f_d - f_{d-1})$ . More details are discussed in Appendix A.

# 4.2.2 HARDWARE-AWARENESS TREE OPTIMIZATION

Given that hardware platforms differ in terms of memory, computational resources, and runtime capabilities, we propose a hardware-aware tree optimization to maximize the overall performance of PPD. As shown on the right of Figure 3, this optimization adjusts the tree size budget based on the performance characteristics of the target hardware.

To achieve this, we define two key functions:

1. Acceptance length  $\tau (n)$  (hardware-independent) and  
2. Forward pass latency  $L_{fp}(n)$  (hardware-dependent).

The speedup ratio,  $\operatorname{Speedup}(n) = \frac{\tau(n)}{L_{fp}(n)}$ , is estimated using a validation dataset, with  $\tau(n)$  evaluated once and  $L_{fp}(n)$  tested on different hardware platforms. We then choose the tree size budget that maximizes  $\operatorname{Speedup}(n)$  based on the measured runtime latency on the specific hardware platform. To eliminate runtime overhead, hardware latency profiling is conducted during idle periods.

# 5 EXPERIMENTS

Models and testbeds. We conducted all the experiments using MobileLLaMA-1.4B (Chu et al., 2023), Vicuna-7B and Vicuna-13B (Chiang et al., 2023). We used 3 prompt tokens and 1 EPT per

prompt token for all inference experiments. The inference throughputs of the models are evaluated on a single NVIDIA A100 GPU with 40GB of memory and a GeForce RTX 4090 using a batch size of 1 and FP16 precision. Further details about the experimental setup can be found in Appendix F.

Training. We froze all trainable parameters of the original LLM. Prompt token embeddings were trained using distillation logits generated from the ShareGPT dataset (ShareGPT, 2023), with a maximum context length of 1024, a cosine learning rate scheduler starting at 0.01, and no warmup. Prompt token embeddings are initialized with normal text token embeddings. For each model, the same set of prompt tokens is used across experiments to demonstrate its generalizability.

Datasets. We assess the throughput performance of  $PPD$  across various tasks and datasets. Specifically, we evaluated  $PPD$  using the MT-Bench dataset (Zheng et al., 2023), which contains multi-turn questions with a range of topics, in both non-greedy (temperature follows the default configuration) and greedy settings (temperature=0). We used the GSM8K (Cobbe et al., 2021) and HumanEval (Chen et al., 2021) datasets only in the greedy setting. The GSM8K dataset consists of grade school math problems and we used the first 500 questions of the test split for our evaluations. HumanEval includes coding tasks, for which we set a maximum new token limit of 512 to control the length of the generated sequences. We used the Alpaca (Li et al., 2023) dataset as the validation dataset to produce the latencies and acceptance lengths used for sparse tree pruning.

# 5.1 SPEEDUP COMPARISON WITH PARALLEL DECODING METHODS

We compare the speedup ratios of  $PPD$  with state-of-the-art parallel decoding methods on MT-Bench in non-greedy settings in Figure 4.  $PPD$  achieves speedups up to  $13.8\%$  higher than Medusa and between 2 times and 3 times higher than other parallel decoding methods. We examine the factors contributing to the enhanced speedup ratios and other performance metrics, as presented in Table 1. The reasons for the increase in speedup ratios are twofold. Firstly,  $PPD$  produces candidate tokens with a higher acceptance rate than Medusa when utilizing a sparse tree of the same size. Notably,  $PPD$  continues to achieve a comparable or slightly better acceptance rate even when employing a much smaller sparse tree - ranging from one-third to half the size. Secondly,  $PPD$  benefits from lower forward pass latency due to its ability to use smaller sparse tree sizes and hence shorter input lengths.  $PPD$  also eliminates the computational overhead associated with separate decoding heads.  $PPD$  maintains the

![](images/d7ff8c0aa5336ea59cae937e11e1d43303b9ab1d5adc7b02d2dc366761b41eeb.jpg)  
Figure 4: Comparative evaluation of latency speedup between PPD and other parallel decoding methods. The experiments were conducted using the MT-Bench dataset, with the temperature set to MT-Bench's default configuration for Medusa and PPD.

same output quality, achieving about the same score on MT-Bench while using significantly fewer trainable parameters.

Figure 5 displays the throughput of  $PPD$  on MT-Bench, HumanEval, and GSM8K with temperature equal to 0.  $PPD$  achieves consistent walltime speedup ratios from  $2.12 \times$  to  $2.49 \times$  on different GPUs, which demonstrates that prompt tokens generalize well on different tasks. In general,  $PPD$  performs better in coding and math reasoning tasks, achieving speedups between  $2.21 \times$  and  $2.49 \times$ . This can be attributed to the fact that both code and math equations often contain fixed patterns and repetitive symbols, which narrows the range of plausible candidates and simplifies the prediction. We also found that with typical acceptance, the speedup increases with temperature. Another notable trend is that smaller models, such as Vicuna-7B, generally achieve more significant speedup ratios as compared to larger models, like Vicuna-13B.  $PPD$  aims to generate more tokens per step, which comes with increased computational demands. For larger models that already require substantial computational resources, it is necessary to limit the size of the sparse tree to avoid exceeding the GPU's utilization cap and causing increased latency. As a result, the number of tokens accepted per step is reduced, leading to lower speedups. However, this can be amortized when using more powerful GPUs than the NVIDIA A100 and the RTX 4090, such as NVIDIA H100.

Table 1: Comparative performance metrics of MobileLLaMA (M) for greedy setting, Vicuna-7B (V-7B) and Vicuna-13B (V-13B) for non-greedy setting using different decoding methods. The table details throughput ( $T$  in tokens/s), average accept lengths ( $\tau$  in tokens), forward pass latency ( $L_{\mathrm{fp}}$  in seconds), quality scores on MT-benchmark, percentages of additional trainable parameters ( $P_{\mathrm{tr}}$ ) and input lengths ( $S_{\mathrm{input}}$ ) after the prefetching phase. The sparse tree size ( $S_{\mathrm{tr}}$ ) of  $PPD$  varies at different time steps as a consequence of different numbers of prompt tokens at each guess token, hence represented as tuples. Same means the output matches with that of the original LLM.  

<table><tr><td>Model</td><td>Method</td><td>T</td><td>τ</td><td>Lfp(s)</td><td>Quality</td><td>Ptr(%)</td><td>S1r</td><td>Sinput</td></tr><tr><td rowspan="2">M</td><td>Vanilla</td><td>50.2</td><td>1.00</td><td>0.020</td><td>-</td><td>NA</td><td>NA</td><td>1</td></tr><tr><td>PPD</td><td>108.7</td><td>2.43</td><td>0.022</td><td>Same</td><td>4.50e-4</td><td>(10,84,89)</td><td>(40,285,285)</td></tr><tr><td rowspan="3">V-7B</td><td>Vanilla</td><td>39.2</td><td>1.00</td><td>0.026</td><td>5.99</td><td>NA</td><td>NA</td><td>1</td></tr><tr><td>Medusa</td><td>82.0</td><td>2.51</td><td>0.0307</td><td>5.98</td><td>8.07</td><td>63</td><td>63</td></tr><tr><td>PPD</td><td>88.0</td><td>2.54</td><td>0.029</td><td>5.93</td><td>1.82e-4</td><td>(10,33,34)</td><td>(40,105,105)</td></tr><tr><td rowspan="3">V-13B</td><td>Vanilla</td><td>30.4</td><td>1.00</td><td>0.0330</td><td>6.38</td><td>NA</td><td>NA</td><td>1</td></tr><tr><td>Medusa</td><td>63.4</td><td>2.59</td><td>0.0408</td><td>-</td><td>5.52</td><td>63</td><td>63</td></tr><tr><td>PPD</td><td>66.1</td><td>2.44</td><td>0.0379</td><td>6.32</td><td>7.87e-5</td><td>(10,20,20)</td><td>(40,60,60)</td></tr></table>

![](images/c43991310636bddce2b8747287d810c6196bcb7d6c6419ae917d2b9c4871b225.jpg)

![](images/ee34d44a32e8998d75566010821b0f6930ec8c9705c65534eefb8bfa4b5f5864.jpg)

![](images/b9adeec0ae82b52afb0d76b094dcfcdbf328fee3c55c1473faca956eafdf3a52.jpg)

![](images/5be2be240313d85ca8722d744f3495885f5a33c5be5dabad3c419eef79f30ab4.jpg)  
Figure 5: Throughput of PPD and vanilla models across different tasks (multi-turn dialogue, coding, and math). The temperature for experiments is set to 0 and the generated output of PPD exactly matches that of the original LLM. We do not show the results of Vicuna-13B on RTX 4090 as it does not fit into the GPU memory.

![](images/b9752739dfb3bdd441c70af7b7e0ea03deedd37e142bac42fc4f92fe27057c7d.jpg)

![](images/03753225457f0e19ef60bc31968166a2914fbbd07cba631fab61f4bdd5c2b752.jpg)

# 5.2 LONG-RANGETOKEN PREDICTION

For a specific sparse tree, the accumulative accuracy provides a theoretical upper bound for the number of generated tokens per step and the maximum possible speedup ratio. Hence, maximizing

![](images/2f52babe25f10604a29a246d436cefefd3de4abe758f3bc4bd5c810a68839798.jpg)  
(a) PD vs. Medusa

![](images/5b33b5d4bf5532b5149c89f8c095c9a95c9270333d51a02d13c70893f1f71f7c.jpg)  
(b) 100 EPT vs. 1 EPT

![](images/2672de3dccf69e6ae20e281cfe4603f12bb95d8825daf7e069fa123148091c78.jpg)  
Figure 6: Accumulative accuracy comparisons across different model configurations and prediction distances. 'V7' for Vicuna-7B, and 'V13' for Vicuna-13B. The notation '@i' refers to a token distance of  $i$ . '100 EPT' represents 100 EPTs per prompt token. Accumulative accuracy is defined as top-k accuracy (e.g., a prediction is correct if the top-k candidates contain the ground truth). These measurements were obtained from the Alpaca Eval dataset with a maximum of 20 steps.  
(c) 13b vs. 7b

accumulative accuracy is crucial for the effectiveness of PPD. Figure 6 demonstrates the accumulative accuracy of the tokens predicted at various positions. We summarize the following three key insights from the results.

PPD excels at predicting more distant tokens. As depicted in Figure 6a, PPD consistently outperforms Medusa in accuracy across all token positions. The accuracy gap between PPD and Medusa widens with the increased token distance (e.g., the top-10 accuracy difference is 0.03 for the 'next next' word versus 0.12 for the 'next next next next' word). This improvement can be attributed to PPD's ability to partially recover conditional dependency information through causally connected prompt tokens.

PPD performs well at generating a broader array of plausible token candidates. For example, in predicting the token at a token distance of 3, the top-10 candidates exhibit an accuracy improvement of 0.1 over Medusa, compared to only 0.02 for the top-1 candidate. This demonstrates the value of using tree attention and the largest viable tree size during inference, as multiple candidate continuations further boost accuracy improvement.

Multiple EPTs per prompt token and larger model sizes yield modest improvements in prediction accuracy. Figure 6b shows that using 100 EPTs per prompt token leads to accuracy improvement, ranging from 0.018 to 0.045. Figure 6c displays that PPD with Vicuna-13B outperforms Vicuna-7B with an accuracy gain of  $0.011\sim 0.038$ . This increase is due to Vicuna-13B's greater embedding dimensions and deeper layers, which enhance the expressive power of prompt tokens. However, these gains are modest and can be offset by the increased computational burden of larger models.

![](images/4b41f5ccd7d8f977bbf2095177d1ccf090c08dfd3f1c2e52d70e752cd2822c6b.jpg)  
(a)

![](images/e35f47f6a802594f3079d5412646817e53c76e527cc8e20752593ada360e18b5.jpg)  
Figure 7: (a) Memory usage of PPD and other baseline methods including Vanilla, Medusa, and Eagle; (b) Throughput comparison of PPD with other parallel decoding approaches. We control the use of tree attention in some approaches for ablation analysis.  
(b)

# 5.3 MEMORY AND TRAINING EFFICIENCY

Memory efficiency. As shown in Figure 7a, we compare the memory overhead of PPD with the leading parallel decoding (Medusa) and speculative decoding approaches (Eagle). The memory overhead of PPD is just  $0.004\%$  of Medusa's and  $0.007\%$  of Eagle's. This efficiency stems from the efficient use of embeddings in PPD, which are significantly smaller than decoding heads and draft models, both of which scale with vocabulary size.

Training efficiency. Table 2 compares the training times of PPD with parallel and speculative decoding methods. PPD is trained until its evaluation accuracy of top-10 candidates surpasses that of Medusa on Alpaca Eval. Notably, PPD surpasses Medusa in evaluation accuracy while training in less than half the time, demonstrating its great potential to reduce training cost.

Table 2: Training time of PPD, Medusa, and Eagle, on 4 A100 GPUs. PPD takes less than half of the time compared to Medusa.  

<table><tr><td>Method</td><td>Training Time</td></tr><tr><td>PPD (Ours)</td><td>0.52 hours</td></tr><tr><td>Medusa</td><td>1.24 hours</td></tr><tr><td>Eagle</td><td>1-2 days</td></tr></table>

# 5.4 ABLATION STUDY

Tree Attention. As illustrated in Figure 7b, tree attention boosts the speedup ratio of  $PPD$  by an additional  $32\%$ , indicating that  $PPD$  generates accurate top-k predictions. Even without the use of tree attention,  $PPD$  still outperforms all other parallel decoding methods, achieving up to a  $14\%$  higher speedup ratio, demonstrating the effectiveness of our approach.

Sparse Tree Pruning Algorithm. Figure 8a shows that the pruned sparse trees consistently achieve longer acceptance lengths compared to naive and random ones across varying sizes. The acceptance length for pruned sparse trees shows a steady increase as the tree size extends, suggesting its good scalability. The convergence of pruned and naive sparse trees at larger sizes suggests a structural similarity emerging from constraints in tree depth and tree node count.

Hardware-aware Tree Size. Figure 8b presents the theoretical speedup across different GPUs. Figure 8c validates that the optimal sparse tree size, derived from theoretical speedup models, indeed results in the greatest actual speedup observed.

PPD + Speculative Decoding. As an orthogonal optimization in accelerating LLMs, PPD can be easily integrated with speculative decoding (Kim et al., 2024). To demonstrate this, we applied PPD to Vicuna-68M (Yang et al., 2024) and used it as the draft model for Vicuna-7B. This combination resulted in a speedup of up to  $1.22 \times$  for speculative decoding on Vicuna-7B compared to using speculative decoding alone.

![](images/84f506b4c0c442354475aaf657fe16b6a1c6369565684d02b9a5ec55861df466.jpg)  
(a)

![](images/ae8572124ec301ca116a30b6f8a31f2e10a4d970ceb7a6006fe01b2f18d2ec66.jpg)  
Figure 8: Evaluation of Sparse Tree Pruning Algorithm. The naive sparse tree in (a) applies the same number of prompt tokens to each guess token, while the pruned sparse tree follows our pruning algorithm. The random sparse tree allocates prompt token budget randomly. The theoretical speedup in (b) is calculated as the ratio of acceptance lengths (hardware-independent) to latency overhead (hardware-dependent). The optimal tree size is obtained from the peak value of the theoretical speedup. The latencies in (b) are obtained from inference on the same prompt for 512 forward passes. (c) shows the actual speedup obtained by running inference on different GPUs with different tree lengths on Alpaca Eval dataset.  
(b)

![](images/f8786fc5c21306ce6a814c5583d33e52a700e567b1bb2d354a9d120db9e2ff29.jpg)  
(c)

# 6 CONCLUSION

We introduced PPD, a memory-efficient, cost-effective, and powerful parallel decoding method that incorporates a hardware-aware online tree optimization. Utilizing specially trained prompt tokens to predict long-range tokens accurately, PPD achieves a speedup of up to  $2.49 \times$  in inference while employing only  $0.0002\%$  additional trainable parameters without incorporating new models or architectural components. We showcased that PPD offers a novel perspective on the capabilities of parallel decoding. Importantly, it could be synergized with other speculative or parallel decoding techniques to expedite inference even further. We hope that by open-sourcing the code base (upon acceptance of the paper), PPD can help the community further advance the performance of real-world deployment of the current and future decoder-based LLM models.

# REFERENCES

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. In International Conference on Machine Learning (ICML), 2024.  
Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv preprint arXiv:2302.01318, 2023.  
Mark Chen et al. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374, 2021.  
Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding. arXiv preprint arXiv:2402.12374, 2024.  
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An Open-Source Chatbot Improving GPT-4 with  $90\%$  * ChatGPT Quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vicuna/.  
Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, and Chunhua Shen. MobileVLM: A Fast, Strong and Open Vision Language Assistant for Mobile Devices. arXiv preprint arXiv:2312.16886, 2023.  
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.  
Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the Sequential Dependency of LLM Inference Using Lookahead Decoding. In International Conference on Machine Learning (ICML), 2024.  
Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee, and Di He. Rest: Retrieval-based Speculative Decoding. arXiv preprint arXiv:2311.08252, 2023.  
Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769-6781, 2020.  
Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer. Speculative Decoding with Big Little Decoder. Advances in Neural Information Processing Systems (NeurIPS), 36, 2024.  
Brian Lester, Rami Al-Rfou, and Noah Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.  
Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast Inference from Transformers via Speculative Decoding. In International Conference on Machine Learning (ICML), 2023.  
Xiang Lisa Li and Percy Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582-4597. Association for Computational Linguistics, 2021.  
Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. AlpacaEval: An Automatic Evaluator of Instruction-following Models. https://github.com/tatsu-lab/alpaca_eval, 2023.  
Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty. In International Conference on Machine Learning (ICML), 2024a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees. In Empirical Methods in Natural Language Processing (EMNLP), 2024b.  
Mingdao Liu, Aohan Zeng, Bowen Wang, Peng Zhang, Jie Tang, and Yuxiao Dong. APAR: LLMs Can Do Auto-Parallel Auto-Regressive Decoding. arXiv preprint arXiv:2401.06761, 2024.  
Xupeng Miao et al. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. In ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), 2024.  
Giovanni Monea, Armand Joulin, and Edouard Grave. PaSS: Parallel Speculative Sampling. arXiv preprint arXiv:2311.13581, 2023.  
Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodola. Accelerating Transformer Inference for Translation via Parallel Decoding. In Annual Meeting of the Association for Computational Linguistics (ACL), 2023.  
Apoory Saxena. Prompt Lookup Decoding, November 2023. URL https://github.com/apoorvumang/prompt-lookup-decoding/.  
ShareGPT. ShareGPT, 2023. URL https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered.  
Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise Parallel Decoding for Deep Autoregressive Models. In Advances in Neural Information Processing Systems (NeurIPS), 2018.  
Sen Yang, Shujian Huang, Xinyu Dai, and Jiajun Chen. Multi-Candidate Speculative Decoding. arXiv preprint arXiv:2401.06706, 2024.  
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
