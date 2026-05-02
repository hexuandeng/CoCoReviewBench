# CODEBOOK FEATURES: SPARSE AND DISCRETE INTERPRETABILITY FOR NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Understanding neural networks is challenging in part because of the dense, continuous nature of their hidden states. We explore whether we can train neural networks to have hidden states that are sparse, discrete, and more interpretable by quantizing their continuous features into what we call codebook features. Codebook features are produced by finetuning neural networks with vector quantization bottlenecks at each layer, producing a network whose hidden features are the sum of a small number of discrete vector codes chosen from a larger codebook. Surprisingly, we find that neural networks can operate under this extreme bottleneck with only modest degradation in performance. This sparse, discrete bottleneck also provides an intuitive way of controlling neural network behavior: first, find codes that activate when the desired behavior is present, then activate those same codes during generation to elicit that behavior. We validate our approach by training codebook Transformers on several different datasets. First, we explore a finite state machine dataset with far more hidden states than neurons. In this setting, our approach overcomes the superposition problem by assigning states to distinct codes, and we find that we can make the neural network behave as if it is in a different state by activating the code for that state. Second, we train Transformer language models with up to 410M parameters on two natural language datasets. We identify codes in these models representing diverse, disentangled concepts (ranging from negative emotions to months of the year) and find that we can guide the model to generate different topics by activating the appropriate codes during inference. Overall, codebook features appear to be a promising unit of analysis and control for neural networks and interpretability. Our codebase and models are open-sourced.

# 1 INTRODUCTION

The strength of neural networks lies in their ability to learn emergent solutions that we could not program ourselves. Unfortunately, the learned programs inside neural networks are challenging to make sense of, in part because they differ from traditional software in important ways. Most strikingly, the state of a neural network program, including intermediate computations and features, is implemented in dense, continuous vectors inside of a network. As a result, many different pieces of information are commingled inside of these vectors, violating the software engineering principle of separation of concerns (Dijkstra, 1982). Moreover, the continuous nature of these vectors means no feature is ever truly off inside of a network; instead, they are activated to varying degrees, vastly increasing the complexity of this state and the possible interactions within it.

A natural question is whether it is possible to recover some of the sparsity and discreteness properties of traditional software systems while preserving the expressivity and learnability of neural networks. To make progress here, we introduce a structural constraint into training that refactors a network to adhere more closely to these design principles. Specifically, we finetune a network with trainable vector quantization bottlenecks (Gray, 1984) at each layer, which are sparse and discrete. We refer to each vector in this bottleneck as a code and the entire library of codes as the codebook. See Figure 1 for a visual depiction of this motivation.

The resulting codebooks learned through this process are a promising interface for understanding and controlling neural networks. For example, when we train a codebook language model on the

![](images/96b22d24d66cca732ff3914bca19fbe70cea22fe30928755d993d8c7cb2af11f.jpg)  
Traditional Software

Figure 1: Codebook features attempt to combine the expressivity of neural networks with the sparse, discrete state often found in traditional software.  
![](images/b123a8c955b45893649a0426daf215f15ba852051b1ad9f5a707871c178c386b.jpg)  
Requires known algorithm

![](images/e9d56083afd0a98150294dfa7821130f0296d99b4f48333cfb3f02750b9df8d5.jpg)  
Sparse, discrete state improves understanding

![](images/e6c848535deab1043209e4da423c393f3a8e7e3e0b0d9ad584e16bbb04cec59c.jpg)  
Neural Networks

![](images/60e2a7b1c27bb6e2c0d8f67d80c583e1674866d69ff934604e1002b9ea701ebe.jpg)  
Emergent algorithm

![](images/6f60d5273f4d2d2e318fc3bf4e6e45e11b97a403dbb4ce29acb355c6f3e2798e.jpg)  
Dense, continuous state hinders understanding

![](images/e63af2d0675322731d6921396b17050f02f9ed30c562aaffb6717598d761f5ec.jpg)  
Codebook Features

![](images/448a1ec4734899f3268b720a71aa373b54e888830cc902519965dfe2b23b3394.jpg)  
Emergent algorithm

![](images/96631a1d79ba77fa464ac9ef06853745431d92786828974c1bd6bf22187c541c.jpg)  
Improved understanding via sparse, discrete form

outputs of a finite state machine, we find a precise mapping between activated codes in different layers of the model to the states of the state machine, overcoming the challenge of superposition (Elhage et al., 2022b). Furthermore, we demonstrate a causal role for these codes: changing which code is activated during the forward pass causes the network to behave as if it were in a different state. Additionally, we apply codebook features to transformer language models with up to 410M parameters, showing that despite this bottleneck, they can be trained with only modest accuracy degradation compared to the original model. We find codes that activate on a wide range of concepts, spanning punctuation, syntax, lexical semantics, and high-level topics. We then show how to use codebook features to control the topic of a model's generations, providing a practical example of how to use our method to understand and control real language models.

# 2 METHOD

Codebook features aim to improve our understanding and control of neural networks by compressing their activation space with a sparse, discrete bottleneck. Specifically, we aim to learn a set of discrete states the network can occupy, of which very few are active during any single forward pass. As we will show later in the paper (Sections 3 and 4), this bottleneck encourages the network to store useful and disentangled concepts in each code. Even more importantly, we show that these interpretations enable us to make causal interventions on the network internals, producing the expected change in the network's behavior. Crucially, codebooks are learned, not hand-specified, enabling them to capture behaviors potentially unknown by human researchers.

Concretely, codebook features are produced by replacing a hidden layer's activations with a sparse combination of code vectors. Let  $a \in \mathbb{R}^N$  be the activation vector of a given N-dimensional layer in a network. We have a codebook  $\mathcal{C} = \{c_1, c_2, \dots, c_C\} \in \mathbb{R}^{C \times N}$ , where  $C$  is the codebook size. To apply the codebook, we first compute the cosine similarities  $\mathrm{sim}(a, c_i) = \frac{a \cdot c_i}{|a| |c_i|}$  between  $a$  and each code vector  $c_i$ . We then replace  $a$  with  $\sum_{i \in S} c_i$ , where  $S$  contains the indices of the top  $k$  most similar code vectors. In other words, we activate and sum the  $k$  code vectors most similar to the original activation  $a$ . The value of  $k$  controls the bottleneck's sparsity; we aim to make  $k$  as small as possible while achieving adequate performance.  $k$  is a small fraction of  $C$  in our experiments, typically less than  $1\%$ , and as a result, we find that codebooks are tight information bottlenecks, transmitting much less information than even 4-bit quantized activations (Appendix B).

While codebook features can be applied to any neural network, we primarily focus on Transformer networks, placing codebooks after either the network's MLP blocks or attention heads. Figure 2 shows the precise location of the codebook for each type of sublayer. Note that this positioning of the codebooks preserves the integrity of the residual stream of the network, which is important for optimizing deep networks (He et al., 2016; Elhage et al., 2021).

![](images/8db57310e0a90803b7f64a6618fca1857fb6e810157bd5c7bc0d61e7d0acfd76.jpg)  
Figure 2: Applying codebook features to transformers. Attention heads: We add one codebook (depicted by the colored rectangles) for each attention head. The codebook is inserted before the projection into the residual stream. Feedforward block: We insert the codebook after the feedforward block, before addition into the residual stream.

# 2.1 TRAINING WITH CODEBOOKS

To obtain codebook features, we add the codebook bottlenecks to existing pretrained models and finetune the model with the original training loss. Thus, the network must learn to perform the task well while adjusting to the discrete codebook bottleneck. Using a pretrained model enables us to produce codebook features more cheaply than training a network from scratch. When finetuning, we use a linear combination of two losses:

Original training loss In our work, we apply codebooks to Transformer-based causal language models and thus use the typical cross-entropy loss these models were trained with:  $\mathcal{L}_{\mathrm{LM}}(\theta) = -\sum_{i=1}^{N} \log p_{\theta}(x_i | x_{<i})$  where  $\theta$  represents the model parameters,  $x_i$  is the next token of input sequence  $x_{<i}$ ,  $p_{\theta}(x_i | x_{<i})$  is the model's predicted probability of token  $x_i$  given input  $x_{<i}$ , and  $N$  is the length of the input sequence.

Reconstruction loss Because we compute the similarity between activations and codebook features using the cosine similarity, which is invariant to magnitude, the code vectors can often grow in size throughout training, leading to instability. For this reason, we find it helpful to add an auxiliary loss to the codes:  $\mathcal{L}_{\mathrm{MSE}} = \mathrm{MSE}(\mathcal{C}(a), \text{stop-gradient}(a))$ , where  $a$  are the input activations to the codebook,  $\mathcal{C}(a)$  is its output, and MSE is the mean squared error, to keep the distance between inputs and chosen codes small. The stop gradient means the gradient of this operation only passes through the codebook, not the input  $a$ , which we found was important to avoid damaging the network's capabilities. $^1$

Final loss and optimization The final loss is simply a combination of both losses above  $\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \lambda L_{\mathrm{MSE}}$  where  $\lambda$  is a tradeoff coefficient. We set  $\lambda$  to 1 in this work. To optimize the codebooks despite the discrete choice of codes, we use the straight-through estimator: we propagate gradients to the codes that were chosen on each forward pass and pass no gradients to the remaining codes (Bengio et al., 2013; van den Oord et al., 2017). We use this strategy to successfully perform end-to-end training of networks up to 24 layers deep, with each layer having a codebook. We defer additional details to Appendix A.

# 2.2 USING CODEBOOKS FOR UNDERSTANDING AND CONTROL

A trained codebook model enables a simple and intuitive way of controlling the network's behavior. This method consists of two phases:

1) Generating hypotheses for the role of codes. Most codes are activated infrequently in the training dataset. We can gain an intuition for the functional role of each code in the network's hidden state by retrieving many examples in the dataset where that code was activated. For example, if a code activates mainly around words like "candle," "matches," and "lighters," we might hypothesize that the token is involved in representations of fire. The discrete on-or-off nature of codes makes this task more manageable than looking at continuous values like neuron activations, as past work has speculated that lower-activating neurons can "smuggle" important information across layers, even if many neurons appear interpretable (Elhage et al., 2022a). As we will show in the following

![](images/cbb7ef5ef7c6516fd495a9667538152284220e56ad740e02e03c3c2e555c1b8b.jpg)  
Figure 3: Codebook features learn the hidden structure of an algorithmic sequence modeling task. The codebook transformer learns to detect the states of a finite state machine and assigns a code to each state. We can then manipulate these codes to cause the network to make predictions as if it were in a different state.

sections, the codes we discover activate more often on a single interpretable feature, while neurons may activate on many unrelated features. Appendix E.1 discusses the advantages and tradeoffs of codebooks over neuron- and feature direction-based approaches in more detail.

2) Steering the network by activating codes. After we have identified codes that reliably activate on the concept we are interested in, we can directly activate those codes to influence the network's behavior. For example, if we identified several codes related to fire, we could activate those codes during generation to produce outputs about fire (e.g., as in Section 4.1). This intervention confirms that the codes have a causal role in the network's behavior.

In the following sections, we apply this same two-step procedure across several different datasets, showing that we can successfully gain insight into the network and control its behavior in each case.

# 3 ALGORITHMIC SEQUENCE MODELING

The first setting we consider is an algorithmic sequence modeling dataset called TokFSM. The purpose of this dataset is to create a controlled setting exhibiting some of the complexities of language modeling, but where the latent features present in the sequence are known. This setting enables us to evaluate how well the model learns codes that activate on these distinct features. An overview of the section and our findings is shown in Figure 3. Below, we describe the dataset, and then (following Section 2.2) we first generate hypotheses for the role of codes, then show how one can predictably influence the network's behavior by manipulating these codes.

The TokFSM Dataset The TokFSM dataset is produced by first constructing a simplified finite state machine (FSM). Our FSM is defined by  $(V,E)$  where  $V = \{0,\dots ,N - 1\}$  is a set of nodes and  $E\subseteq V\times V$  indicates the set of valid transitions from one state to the next. In our setting, we choose  $N = 100$  and give each node 10 randomly chosen outbound neighbors, each assigned an equal transition probability (0.1). Entries in the dataset are randomly sampled rollouts of the FSM up to 64 transitions. We serialize the sequences at the digit level; this gives a sequence length of 128 for each input. For example, if our sampled rollout is [18, 00, 39], we would serialize it as [1, 8, 0, 0, 3, 9] for the neural network. Thus, the model must learn to detokenize the input into its constituent states, predict the next FSM state, and then retokenize the state to predict the next token.

Training and evaluating the codebook models We train 4-layer Transformers with 4 attention heads and an embedding size of 128 based on the GPTNeoX architecture (Black et al., 2022) on the TokFSM dataset. We train several models with different numbers of codes and sparsity values  $k$ , with codebooks either at the network's attention heads or both the attention heads and MLP Layers (see Figure 2). In Table 1, we report the accuracy of the resulting models both in terms of their language modeling loss, next token accuracy, and their ability to produce valid transitions of the FSM across a generated sequence. The  $k = 1$  model with codebooks at only the attention layers achieves comparable performance across all metrics to the original model. At the same time, larger values of  $k$  enable the model with codebooks at both attention and MLP blocks to attain comparable performance. It is striking that networks can perform so well despite this extreme bottleneck at every layer. We defer additional training details to Appendix C.1 and ablation studies to Table 8.

Table 1: Performance of original and codebook models on TokFSM. A  $k = 1$  codebook model on only attention layers attains similar performance to the original model, while attention-and-MLP codebooks require a higher  $k$  and codebook size  $C$  to match performance.  $\dagger$  indicates the model we analyze in the rest of the section.  

<table><tr><td>Codebook Type</td><td>Loss</td><td>LM Acc</td><td>State Acc</td></tr><tr><td>No Codebook</td><td>1.179</td><td>46.36</td><td>96.77</td></tr><tr><td>Attn Only k=1, C=2k</td><td>1.18</td><td>46.33</td><td>96.39</td></tr><tr><td>†Attn+MLP k=1, C=10k</td><td>1.269</td><td>45.27</td><td>63.65</td></tr><tr><td>Attn+MLP k=1, C=20k</td><td>1.254</td><td>45.56</td><td>63.81</td></tr><tr><td>Attn+MLP k=4, C=20k</td><td>1.192</td><td>46.20</td><td>80.69</td></tr><tr><td>Attn+MLP k=16, C=20k</td><td>1.183</td><td>46.32</td><td>91.53</td></tr><tr><td>Attn+MLP k=128, C=20k</td><td>1.178</td><td>46.38</td><td>95.82</td></tr></table>

![](images/5e2fd3a09babefcfb8c1754d28b7dd092bf18b0b07a680dd97d76ceb5034c68f.jpg)  
(a) State code interventions

![](images/a6f58e592e56052941cddd4f99b07bd7e010f6a59c3cf48b758e26203808cec1.jpg)  
Figure 4: Interventions on the state and state-plus-digit codes in a sequence. Changing just the MLP codes to codes associated with another state shifts the output distribution almost entirely to the target state. Changing codes in other layers has a much smaller effect. Normalized JS Div stands for the normalized Jensen-Shannon Divergence, where the initial difference (None) is normalized to 1.  
(b) State-plus-digit code interventions

# 3.1 GENERATING HYPOTHESES FOR THE ROLE OF CODES

After training these models, we examine the  $k = 1$  attention and MLP codebook transformer following Section 2.2. Looking at activating tokens reveals a wide range of interesting-looking codes. We provide descriptions of these codes along with a table of examples in Table 6, and focus our analysis on two families of codes here: in the last three MLP layers (layers 1, 2, and 3), we identify state codes that reliably activate on the second token of a specific state (of which there are 100 possibilities), as well as state-plus-digit codes that activate on a specific digit when it follows a specific state (686 possibilities in our state machine). For example, code 2543 in MLP layer 2 activates on the 0 in the state 40 (e.g., 50-40-59). This finding is notable because there are only 128 neurons in a given MLP layer, far lower than the total number of these features. Thus, the codebooks must disentangle features represented in a distributed manner across different neurons inside the network. (Anecdotally, the top-activating tokens for the neurons in these layers do not appear to follow any consistent pattern.)

We quantify this further with an experiment where we use state codes to classify states and compare them to the neuron with the highest precision at that state code's recall level. As shown in Figure 6a, codes have an average precision of  $97.1\%$ , far better than the average best neuron precision of  $70.5\%$ . These pieces of evidence indicate that codebooks can minimize the superposition problem in this setting. See Appendix C for additional details and experiments.

# 3.2 STEERING THE NETWORK BY ACTIVATING CODES

While these associations can provide hypotheses for code function, they do not provide causal evidence that codes causally influence the network's behavior. For this, interventional studies are necessary (Spirtes et al., 2000; Pearl & Mackenzie, 2018; Geiger et al., 2020; 2021). The state and state-plus-digit codes presented in Section 3.1 suggest a natural causal experiment: set the activated code in a given codebook to the code corresponding to another state and see whether the next token distribution shifts accordingly. More specifically, let  $\mathcal{C}^{(l)}(x_t)$  be the codebook at layer  $l$  applied to

![](images/d9dab0150f439d89a1d6e8686db6c9a5986c82991e449516e55f6597684cfee2.jpg)  
Table 2: Codebook models are still capable language models.. Asterisks  $(^{*})$  denote the base model we apply the codebooks to, while daggers  $(\dagger)$  indicate the codebook models we analyze in the rest of the paper. We trained the other models to provide additional comparisons (see Appendix D.3 for more details, including on grouped codebooks.). All models have a codebook size of  $C = 10k$ . Note that the MLP 16-group  $k = 8$  model is comparable to the attention  $k = 8$  model because our model has 16 attention heads. While we use a pretrained TinyStories model as our base model, we also report metrics for a model we finetune to account for any subtle differences in data processing.

(a) TinyStories 1-Layer Model  
(b) WikiText-103 410M 24-Layer Model  

<table><tr><td>Language Model</td><td>Loss</td><td>Acc</td><td>Language Model</td><td>Loss</td><td>Acc</td></tr><tr><td>*Pretrained</td><td>1.82</td><td>56.22</td><td>*Finetuned (Wiki)</td><td>2.41</td><td>50.52</td></tr><tr><td>Finetuned</td><td>1.57</td><td>59.27</td><td>Finetuned 160M (Wiki)</td><td>2.72</td><td>46.75</td></tr><tr><td>†Attn, k=8</td><td>1.66</td><td>57.91</td><td>†Attn, k=8</td><td>2.74</td><td>46.68</td></tr><tr><td>MLP, k=100</td><td>1.57</td><td>59.47</td><td>Attn, k=64</td><td>2.55</td><td>48.44</td></tr><tr><td>MLP, grouped 16 × (k=8)</td><td>1.60</td><td>59.36</td><td>MLP, k=100</td><td>3.03</td><td>42.47</td></tr><tr><td></td><td></td><td></td><td>MLP, grouped 16 × (k=8)</td><td>2.73</td><td>46.16</td></tr><tr><td></td><td></td><td></td><td>MLP, grouped 16 × (k=64)</td><td>2.57</td><td>48.46</td></tr></table>

input token  $x_{t}$ . As we consider a  $k = 1$  model,  $C^{(l)}(x_{t})$  returns a single code  $c_{t}^{(l)} \in \mathbb{R}^{d}$ . We replace this code with  $\tilde{c}_{t}^{(l)}$ , a code that activates when a different state is present. We then recompute the forward pass from that point and observe whether the network's next token distribution resembles the next token distribution for the new state.

In Figure 4a, we find that this is precisely the case—changing only the state codes in the MLP layers to a different state code shifts the next token distribution towards that other state, as measured by the Jensen-Shannon Divergence (JSD Lin, 1991), averaged over 500 random state transitions. This effect is even more substantial for the state-plus-digit codes, where changing the codes in the MLP layers makes the next-state distribution almost identical to that of the new state (Figure 4b). These results provide strong evidence that these codes perform the expected causal role in the network. Note that applying a similar perturbation to just a single MLP layer or all the attention layers causes a much smaller drop in JSD, indicating that this information is mainly stored across several MLP layers.

# 4 LANGUAGE MODELING

Next, we apply codebook features to language models (LMs) trained on naturalistic text corpora. We demonstrate the generality and scalability of our approach by training two models of different sizes on two different datasets. After describing the models we train and the training data, we follow the strategy described in Section 2.2 and identify hypotheses for the role of codes in the network. Then, we validate these hypotheses by steering the models through targeted activation of codes.

Trained models We finetune a small, 1-layer, 21 million parameter model on the TinyStories dataset of children's stories (Eldan & Li, 2023). We also finetune a larger, 24-layer 410M parameter

![](images/b56ec0defe754bc36545d47f81137d6d15c98c3a803909fd186a6053ab4bcc1f.jpg)  
(a) Finite-state machine dataset (TokFSM)

![](images/8d1bd7b4bee79a004a0e21c342f7aaee18a8dff029dcd5aae8b61e9427c32d00.jpg)  
Figure 6: Codes are better classifiers of simple textual features than neurons.  $Y$ -axis: precision of a given code at classifying a regular expression.  $X$ -axis: precision of the best neuron in the network, with a threshold chosen to match the recall of the code. Red line:  $y = x$  
(b) WikiText-103

model on the WikiText-103 dataset, consisting of high-quality English-language Wikipedia articles (Merit et al., 2016). See Appendix D for more training details.

Codebook models are still strong language models Remarkably, despite the extreme bottleneck imposed by the codebook constraint, the codebook language models can still achieve strong language modeling performance. As shown in Table 2, codebook models can attain a loss and accuracy close to or better than the original models with the proper settings. In addition, the generations of the codebook look comparable to the base models, as shown in Table 10. Finally, in Appendix D.4, we profile the inference speed of these codebook models, showing how sparsity and fast maximum inner product search (MIPS) algorithms enable codebooks to run much more efficiently than the naive implementation of two large matrix multiplications.

Generating hypotheses for the role of codes We also explore the interpretability of codes by looking at examples that the code activates on. In Table 11, we catalog codes that selectively activate on a wide range of linguistic phenomena, spanning orthography (e.g., names starting with “B”), word types (e.g., months of the year), events (e.g., instances of fighting), and overall topics (e.g., fire or football). Interestingly, codes for a particular linguistic phenomenon may not always activate on the words most relevant to that concept. For example, in our TinyStories model, we find a code that activates on mentions of fighting and violence might trigger on the word the but not the adjacent word quarrel. We suspect this may be because the network can store pieces of information in nearby tokens and retrieve them when needed via attention.

Comparison to neuron-level interpretability As in Section 3.1, we would like to compare the interpretability of the codebook to neuron-level interpretability. While natural language features are more complex than the states in Section 3, we conduct a preliminary experiment comparing both neuron- and code-based classifiers to regular expression-based classifiers. We first collect a set of codes that appear to have simple, interpretable activation patterns (e.g., "fires on years beginning with 2"). We then created heuristic regular expressions targeting those features (e.g., 2\d\d\d). Next, we compute the precision of the code classifier, using the regular expression as our source of truth. We then take the recall of our code classifier and search across all neurons, thresholding each at the same recall as the code and reporting the highest precision found. As Figure 6b demonstrates, codes are far better classifiers of these features than neurons on average, with over  $30\%$  higher average precision. We defer additional details and discussion to Appendix D.7.

# 4.1 STEERING THE NETWORK BY ACTIVATING TOPIC CODES

As in Section 3.2, we would like to validate that codes do not merely fire in a correlated way with different linguistic features but that they have a causal role in the network's behavior. As an initial investigation of this goal, we study a subset of codes in the attention codebook model that appear to identify and control the topic discussed by a model. To identify potential topic codes, we use a

Table 3: Activating topic codes causes the model to discuss those topics. Percentage of generations that mention the topic before and after setting one or all codes in each attention head to the topic code. Numbers in (parentheses) indicate the number of activated topic codes. This number is smaller for the all codes condition because only one topic code will be activated if multiple topic codes are located in the same attention head.  

<table><tr><td colspan="4">(a) Wikitext</td></tr><tr><td>Topic</td><td>Baseline Freq</td><td>Steered (one code)</td><td>Steered (all codes)</td></tr><tr><td>Video game</td><td>2.5</td><td>55.0 (18)</td><td>75.0 (4)</td></tr><tr><td>Football</td><td>7.5</td><td>47.5 (18)</td><td>95.0 (8)</td></tr><tr><td>Movie</td><td>27.5</td><td>42.5 (12)</td><td>90.0 (5)</td></tr><tr><td>Song</td><td>20.0</td><td>32.5 (17)</td><td>85.0 (11)</td></tr></table>

<table><tr><td colspan="3">(b) TinyStories</td></tr><tr><td>Topic</td><td>Baseline Freq</td><td>Steered (one code)</td></tr><tr><td>Dragon</td><td>2.5</td><td>65.0 (8)</td></tr><tr><td>Slide</td><td>2.5</td><td>95.0 (12)</td></tr><tr><td>Friend</td><td>42.5</td><td>75.0 (9)</td></tr><tr><td>Flower</td><td>0.0</td><td>90.0 (8)</td></tr><tr><td>Fire</td><td>2.5</td><td>100.0 (16)</td></tr><tr><td>Baby</td><td>0.0</td><td>90.0 (15)</td></tr><tr><td>Princess</td><td>40.0</td><td>87.5 (14)</td></tr></table>

simple heuristic and select only codes that activate on more than  $50\%$  of tokens in a given sequence.3 Of these, we manually filter by looking at the activating tokens of these codes and choose only those that appear to activate frequently on other examples related to that topic.

To shift the output generations of the model, we then take an input prompt (e.g., the start-of-sequence token) and activate the topic codes in the model for every token of this prompt. Then, we sample from the model, activating the topic codes for each newly generated token. Unlike Section 3, our models here have  $k > 1$ . Thus, we explore two types of interventions: First, activating a single code in each codebook (replacing the code with the lowest similarity with the input) and second, replacing all activated codes in each codebook with  $k$  copies of the topic code. We use the attention-only codebook with  $k = 8$  in our experiments. See Figure 5 for a graphical depiction.

Remarkably, activating the topic codes causes the model to introduce the target topic into the sampled tokens in a largely natural way. We show several examples of this phenomenon in Tables 4, 13 and 14. Interestingly, even though the topic code is activated at every token, the topic itself is often only introduced many words later in the sequence, when it would be contextually appropriate. We quantify the success of this method by generating many steered sequences and classifying the generated examples into different categories with a simple word-based classifier. The results, presented in Table 3, demonstrate that the steered generations mention the topic far more often, with almost all generations successfully mentioning the topic when all codes in a codebook are replaced. See Appendix D.8 for more details and additional generations. These interventions constitute meaningful evidence of how codebook features can enable interpretation and control of real language models.

# 5 RELATED WORK

Mechanistic interpretability Our work continues a long stream of work since the 1980s on understanding how neural networks operate, especially when individual neurons are uninterpretable (Servan-Schreiber et al., 1988; Elman, 1990) Recent work has continued these investigations in modern computer vision models (Olah et al., 2018; 2020; Bau et al., 2020b) and language models (Elhage et al., 2021; Geva et al., 2021), with special focus on the problem of understanding superposition, when many features are distributed across a smaller number of neurons (Elhage et al., 2022b). Recent work has investigated whether sparse dictionary learning techniques can recover these features (Yun et al., 2021; Sharkey et al., 2022), including the concurrent work of Bricken et al. (2023) and Cunningham et al. (2023). Our work shares similar goals as the above works. Codebook features attempt to make it easier to identify concepts and algorithms inside of networks by refactoring

Table 4: Example steered generations for TinyStories model. More examples in Table 13  

<table><tr><td>Code Concept</td><td># codes</td><td>Example steered generation</td></tr><tr><td>Dragon</td><td>8</td><td>Once upon a time, there was a little girl named Lily. She was very excited to go outside and explore. She flew over the trees and saw a big, scary dragon. The dragon was very scary. [...]</td></tr><tr><td>Flower</td><td>8</td><td>Once upon a time, there was a little girl named Lily. She liked to pick flowers in the meadow. One day, she saw a big, green [...]</td></tr><tr><td>Fire</td><td>16</td><td>Once upon a time, there was a little boy named Timmy. Timmy loved his new toy. He always felt like a real fireman. [...]</td></tr><tr><td>Princess</td><td>14</td><td>Once upon a time, there was a little bird named Tweety. One day, the princess had a dream that she was invited to a big castle. She was very excited and said, “I want to be a princess and [...]</td></tr></table>

their hidden states into a sparse and discrete form. We also show how codebooks can mitigate superposition by representing more features than there are neurons and that we can intervene on the codebooks to alter model behavior systematically.

Discrete structure in neural networks Our work also connects to multiple streams of research on incorporating discrete structure into neural networks (Andreas et al., 2016; Mao et al., 2019). Most relevant is VQ-VAE (van den Oord et al., 2017), which trains an autoencoder with a vector quantized hidden state (Gray, 1984). Our work also leverages vector quantization; however, unlike past work, we extend this method by using it as a sparse, discrete bottleneck that could insert between the layers of any neural network (and apply it to autoregressive language models), enabling better understanding and control of the network's intermediate computation.

Inference-time steering of model internals Finally, our work connects to recent research on steering models based on inference-time perturbations. For example, Merullo et al. (2023) and Turner et al. (2023) steer networks by adding vectors of different magnitudes to different layers in the network. Our work supports these aims by making it easier to localize behaviors inside the network (guided by activating tokens) and making it easier to perform the intervention by substituting codes (so the user does not have to try many different magnitudes of a given steering vector at each layer).

We include an extended discussion of related work, including the relative advantages of codebooks and dictionary learning methods in Appendix E.

# 6 DISCUSSION AND FUTURE WORK

We present codebook features, a method for training models with sparse and discrete hidden states. Codebook features enable unsupervised discovery of algorithmic and linguistic features inside language models, making progress on the superposition problem (Elhage et al., 2022b). We have shown how the sparse, discrete nature of codebook features reduces the complexity of a neural network's hidden state, making it easier to search for features and control a model's behavior with them.

Our work has limitations. First, we only study Transformer neural networks on one algorithmic dataset and two natural language datasets; we do not study transformers applied to visual data or other architectures, such as convolutional neural networks, leaving this for future work. In addition, we only explore topic manipulation in language models; future work can explore the manipulation of other linguistic features in text, including sentiment, style, and logical flow.

Ultimately, our results suggest that codebooks are an appealing unit of analysis for neural networks and a promising foundation for the interpretability and control of more complex phenomena in models. Looking forward, the sparse, discrete nature of codebook features should aid in discovering circuits across layers, more sophisticated control of model behaviors, and making automated, larger-scale interpretability methods more tractable.[5]

# REPRODUCIBILITY STATEMENT

We release our codebase and trained models to enable others to easily build on our work. Additionally, Sections 2 to 4 and appendices A, C and D describe the specific experimental details and settings we used to carry out our experiments.

# REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.  
Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 39-48, 2016.  
Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics, 6:483-495, 2018. doi: 10.1162/tacl_a_00034. URL https://aclanthology.org/Q18-1034.  
David Bau, Steven Liu, Tongzhou Wang, Jun-Yan Zhu, and Antonio Torralba. Rewriting a deep generative model. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part I 16, pp. 351-369. Springer, 2020a.  
David Bau, Jun-Yan Zhu, Hendrik Strobelt, Agata Lapedriza, Bolei Zhou, and Antonio Torralba. Understanding the role of individual units in a deep neural network. Proceedings of the National Academy of Sciences, 117(48):30071-30078, 2020b.  
Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.  
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397-2430. PMLR, 2023.  
Sid Black, Stella Rose Biderman, Eric Hallahan, Quentin G. Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Martin Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Benqi Wang, and Samuel Weinbach. GPT-NeoX-20B: An Open-Source Autoregressive Language Model. arXiv preprint arXiv:2204.06745, 2022. URL https://api_semanticscholar.org/CorpusID:248177957.  
Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the Opportunities and Risks of Foundation Models. arXiv preprint arXiv:2108.07258, 2021.  
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monoseismicity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monoseismic-features/index.html.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
Shyamal Buch, Li Fei-Fei, and Noah D Goodman. Neural event semantics for grounded language understanding. Transactions of the Association for Computational Linguistics, 9:875-890, 2021.

Emmanuel J Candes, Justin K Romberg, and Terence Tao. Stable signal recovery from incomplete and inaccurate measurements. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences, 59(8):1207-1223, 2006.  
Lawrence Chan, Adrià Garriga-Alonso, Nicholas Goldowsky-Dill, Ryan Greenblatt, Jenny Nitishinskaya, Ansh Radhakrishnan, Buck Shlegeris, and Nate Thomas. Causal scrubbing: A method for rigorously testing interpretability hypotheses. In Alignment Forum, 2022.  
Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning. What Does BERT Look At? An Analysis of BERT's Attention. arXiv preprint arXiv:1906.04341, 2019.  
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.  
Edsger W Dijkstra. On the role of scientific thought. Selected writings on computing: a personal perspective, pp. 60-66, 1982.  
David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289-1306, 2006.  
Michael Elad and Michal Aharon. Image denoising via sparse and redundant representations over learned dictionaries. IEEE Transactions on Image processing, 15(12):3736-3745, 2006.  
Ronen Eldan and Yuanzhi Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English?, 2023.  
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1, 2021.  
Nelson Elhage, Tristan Hume, Catherine Olsson, Neel Nanda, Tom Henighan, Scott Johnston, Sheer ElShowk, Nicholas Joseph, Nova DasSarma, Ben Mann, Danny Hernandez, Amanda Askell, Kamal Ndousse, Andy Jones, Dawn Drain, Anna Chen, Yuntao Bai, Deep Ganguli, Liane Lovitt, Zac Hatfield-Dodds, Jackson Kernion, Tom Conerly, Shauna Kravec, Stanislav Fort, Saurav Kadavath, Josh Jacobson, Eli Tran-Johnson, Jared Kaplan, Jack Clark, Tom Brown, Sam McCandlish, Dario Amodei, and Christopher Olah. Softmax Linear Units. Transformer Circuits Thread, 2022a. https://transformer-circuits.pub/2022/solu/index.html.  
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy Models of Superposition. Transformer Circuits Thread, 2022b.  
Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990.  
Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873-12883, 2021.  
Kion Fallah and Christopher J Rozell. Variational sparse coding with learned thresholding. arXiv preprint arXiv:2205.03665, 2022.  
Ruth Fong and Andrea Vedaldi. Net2vec: Quantifying and explaining how concepts are encoded by filters in deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8730-8738, 2018.  
Dan Friedman, Alexander Wettig, and Danqi Chen. Learning Transformer Programs. arXiv preprint arXiv:2306.01128, 2023.  
Atticus Geiger, Kyle Richardson, and Christopher Potts. Neural natural language inference models partially embed theories of lexical entailment and negation. arXiv preprint arXiv:2004.14623, 2020.

Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. Advances in Neural Information Processing Systems, 34:9574-9586, 2021.  
Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, and Noah D. Goodman. Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations. arXiv preprint arXiv:2303.02536, 2023.  
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer Feed-Forward Layers Are Key-Value Memories, 2021.  
Mario Giulianielli, Jacqueline Harding, Florian Mohnert, Dieuwke Hupkes, and Willem Zuidema. Under the Hood: Using Diagnostic Classifiers to Investigate and Improve how Language Models Track Agreement Information. arXiv preprint arXiv:1808.08079, 2018.  
Gabriel Goh, Nick Cammarata †, Chelsea Voss †, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal Neurons in Artificial Neural Networks. *Distill*, 2021. doi: 10.23915/distill.00030. https://distill.pub/2021/multimodal-neurons.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Stephen Jay Gould. The exaptive excellence of spandrels as a term and prototype. Proceedings of the National Academy of Sciences, 94(20):10750-10755, 1997.  
Stephen Jay Gould and Richard C Lewontin. 5 The Spandrels of San Marco and the Panglossian Paradigm: A Critique of the Adaptationist Programme. Conceptual Issues in Evolutionary Biology, 205:79, 1979.  
Robert Gray. Vector quantization. IEEE Assp Magazine, 1(2):4-29, 1984.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Evan Hernandez, Belinda Z Li, and Jacob Andreas. Measuring and manipulating knowledge representations in language models. arXiv preprint arXiv:2304.00740, 2023.  
John Hewitt, John Thickstun, Christopher D. Manning, and Percy Liang. Backpack Language Models, 2023.  
Henrik Jacobsson. Rule extraction from recurrent neural networks: Ataxonomy and review. Neural Computation, 17(6):1223-1263, 2005.  
Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128, 2010.  
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547, 2019.  
Melvin Johnson, Mike Schuster, Quoc V Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viégas, Martin Wattenberg, Greg Corrado, et al. Google's multilingual neural machine translation system: Enabling zero-shot translation. Transactions of the Association for Computational Linguistics, 5:339-351, 2017.  
Pentti Kanerva. Sparse distributed memory. MIT press, 1988.  
Rohit Keshari, Richa Singh, and Mayank Vatsa. Guided Dropout. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):4065-4072, Jul. 2019. doi: 10.1609/aaai.v33i01.33014065. URL https://ojs.aaai.org/index.php/AAAI/article/view/4302.  
Nitish Shirish Keskar, Bryan McCann, Lav R. Varshney, Caiming Xiong, and Richard Socher. CTRL: A Conditional Transformer Language Model for Controllable Generation, 2019.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pp. 2668-2677. PMLR, 2018.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
George KingsleyZipf. Selected studies of the principle of relative frequency in language. Harvard university press, 1932.  
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International conference on machine learning, pp. 5338-5348. PMLR, 2020.  
Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Ng. Efficient sparse coding algorithms. Advances in neural information processing systems, 19, 2006.  
Jianhua Lin. Divergence measures based on the shannon entropy. IEEE Transactions on Information theory, 37(1):145-151, 1991.  
Ziming Liu, Eric Gan, and Max Tegmark. Seeing is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability, 2023.  
Andreas Madsen, Siva Reddy, and Sarath Chandar. Post-hoc Interpretability for Neural NLP: A Survey. ACM Computing Surveys, 55(8):1-42, 2022.  
Alireza Makhzani and Brendan J Frey. Winner-take-all autoencoders. Advances in neural information processing systems, 28, 2015.  
Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B Tenenbaum, and Jiajun Wu. The neurosymbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. arXiv preprint arXiv:1904.12584, 2019.  
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in Neural Information Processing Systems, 35:17359-17372, 2022a.  
Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. arXiv preprint arXiv:2210.07229, 2022b.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models, 2016.  
Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Language Models Implement Simple Word2Vec-style Vector Arithmetic. arXiv preprint arXiv:2305.16130, 2023.  
Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. arXiv preprint arXiv:2110.11309, 2021.  
Jesse Mu and Jacob Andreas. Compositional explanations of neurons. Advances in Neural Information Processing Systems, 33:17153-17163, 2020.  
Chris Olah, Arvind Satyanarayan, Ian Johnson, Shan Carter, Ludwig Schubert, Katherine Ye, and Alexander Mordvintsev. The Building Blocks of Interpretability. Distill, 2018. doi: 10.23915/distill.00010. https://distill.pub/2018/building-blocks.  
Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom In: An Introduction to Circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.  
Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by V1? Vision research, 37(23):3311-3325, 1997.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.  
Judea Pearl and Dana Mackenzie. The book of why: the new science of cause and effect. Basic books, 2018.  
Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in BERTology: What we know about how BERT works. Transactions of the Association for Computational Linguistics, 8:842-866, 2021.  
Christopher J Rozell, Don H Johnson, Richard G Baraniuk, and Bruno A Olshausen. Sparse coding via thresholding and local competition in neural circuits. Neural computation, 20(10):2526-2563, 2008.  
David E Rumelhart, Geoffrey E Hinton, James L McClelland, et al. A general framework for parallel distributed processing. Parallel distributed processing: Explorations in the microstructure of cognition, 1(45-76):26, 1986.  
David E Rumelhart, James L McClelland, PDP Research Group, et al. Parallel distributed processing. Foundations, 1, 1988.  
Shibani Santurkar, Dimitris Tsipras, Mahalaxmi Elango, David Bau, Antonio Torralba, and Aleksander Madry. Editing a classifier by rewriting its prediction rules. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 23359-23373. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/c46489a2d5a9a9ecfc53b17610926ddd-Paper.pdf.  
Pierre Sermanet, Koray Kavukcuoglu, Soumith Chintala, and Yann LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3626-3633, 2013.  
David Servan-Schreiber, Axel Cleeremans, and James McClelland. Learning sequential structure in simple recurrent networks. Advances in neural information processing systems, 1, 1988.  
Lee Sharkey, Dan Braun, and Beren Millidge. Taking features out of superposition with sparse autoencoders. In Alignment Forum, 2022. URL https://wwwalignmentforum.org/posts/z6QQJbtpkEAX3Aojj.  
Peter Spirtes, Clark N Glymour, and Richard Scheines. Causation, prediction, and search. MIT press, 2000.  
Alex Tamkin, Dan Jurafsky, and Noah Goodman. Language through a prism: A spectral approach for multiscale language representations. Advances in Neural Information Processing Systems, 33: 5492-5504, 2020.  
Simon Thorpe. Local vs. distributed coding. Intellectica, 8(2):3-40, 1989.  
Francesco Tonolini, Bjørn Sand Jensen, and Roderick Murray-Smith. Variational sparse coding. In Uncertainty in Artificial Intelligence, pp. 690-700. PMLR, 2020.  
Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation Addition: Steering Language Models Without Optimization. arXiv preprint arXiv:2308.10248, 2023.  
Aaron van den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.  
Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small. arXiv preprint arXiv:2211.00593, 2022.

Eric Wong, Shibani Santurkar, and Aleksander Madry. Leveraging sparse linear layers for debuggable deep networks. In International Conference on Machine Learning, pp. 11205-11216. PMLR, 2021.  
Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021.  
Mert Yuksekgonul, Maggie Wang, and James Zou. Post-hoc concept bottleneck models. arXiv preprint arXiv:2205.15480, 2022.  
Zeyu Yun, Yubei Chen, Bruno A Olshausen, and Yann LeCun. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. arXiv preprint arXiv:2103.15949, 2021.  
Haofei Zhang, Mengqi Xue, Xiaokang Liu, Kaixuan Chen, Jie Song, and Mingli Song. Schema inference for interpretable image classification. arXiv preprint arXiv:2303.06635, 2023.  
Ting Zhang, Chao Du, and Jingdong Wang. Composite quantization for approximate nearest neighbor search. In International Conference on Machine Learning, pp. 838-846. PMLR, 2014.  
Jun Zhu and Eric P Xing. Sparse topical coding. arXiv preprint arXiv:1202.3778, 2012.
