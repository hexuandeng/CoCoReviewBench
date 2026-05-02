# DISCOVERING LATENT NETWORK TOPOLOGY IN CONTEXTUALIZED REPRESENTATIONS WITH RANDOMIZED DYNAMIC PROGRAMMING

Anonymous authors

Paper under double-blind review

# ABSTRACT

The discovery of large-scale discrete latent structures is crucial for understanding the fundamental generative processes of language. In this work, we use structured latent variables to study the representation space of contextualized embeddings and gain insight into the hidden topology of pretrained language models. However, existing methods are severely limited by issues of scalability and efficiency as working with large combinatorial spaces requires expensive memory consumption. We address this challenge by proposing a Randomized Dynamic Programming (RDP) algorithm for the approximate inference of structured models with DP-style exact computation (e.g., Forward-Backward). Our technique samples a subset of DP paths reducing memory complexity to as small as one percent. We use RDP to analyze the representation space of pretrained language models, discovering a large-scale latent network in a fully unsupervised way. The induced latent states not only serve as anchors marking the topology of the space (neighbors and connectivity), but also reveal linguistic properties related to syntax, morphology, and semantics. We also show that traversing this latent network yields unsupervised paraphrase generation.

# 1 INTRODUCTION

The discovery of large-scale discrete latent structures is crucial for understanding the fundamental generative processes of language, and has been shown useful to various NLP tasks ranging from data-to-text generation (Li & Rush, 2020), summarization (Angelidis et al., 2021), syntactic parsing (Kim et al., 2019), and knowledge graph reasoning (Qu et al., 2020). In this work, we use latent structures to analyze geometric properties of representation space of pretrained language models (PLMs). Despite the large volume of recent work analyzing PLMs and proposing various improvements (Rogers et al., 2020), little is known about the topological structure of their representation manifold. Since such structure cannot be easily observed, it is only natural to resort to latent variables. Yet scaling discrete combinatorial structures is extremely difficult with multiple modeling and computational challenges (Wainwright & Jordan, 2008).

In this work, we address the computational challenges arising from working with combinatorial structures. We consider linear-chain CRFs, a popular structured model family (Ma & Hovy, 2016; Sutton & McCallum, 2006) that uses dynamic programming for exact inference. Specifically, we focus on the forward algorithm (Rabiner, 1989), which is widely used to compute the partition function. Space complexity for this algorithm is  $O(TN^2)$  where  $N$  is the number of latent states and  $T$  the length of the sequence. It is precisely the  $N^2$  term that becomes problematic when we construct the adjacent gradient graph with automatic differentiation. DP-based inference algorithms are not optimized for modern computational devices like GPUs and typically work under small-data regimes, with  $N$  in the range [10, 100] (Ma & Hovy, 2016; Wiseman et al., 2018). With larger  $N$ , inference becomes intractable since gradients do not easily fit into GPU memory (Sun et al., 2019).

Our algorithmic contribution is a randomization technique for dynamic programming which allows us to scale  $N$  to thousands (possibly more) latent states. Specifically, to approximate the partition function, instead of summing over all possible combinations of latent states, we only sum over paths with most probable states, and sample a subset of less likely paths to correct the bias according

to a reasonable proposal. Since we only calculate the sampled path, memory consumption can be reduced to a small controllable budget which is scale invariant. With a larger memory budget, our method becomes more accurate, and our estimation error smaller. We thus recast the memory complexity challenge into a tradeoff between memory budget, proposal accuracy, and estimation error. When applied to linear-chain CRFs, we show that RDP scales the model by two orders of magnitude with memory complexity as small as one percent of the full DP. Beyond linear-chains, RDP is applicable to any structured model with DP-style exact inference such as trees (Kim et al., 2019) and semi-Markov models (Li & Rush, 2020), and could also be extended to more general message passing algorithms (Wainwright & Jordan, 2008).

Our analytical contribution is a geometric study of the representation manifold of PLMs, using the proposed RDP algorithm. We hypothesize that there exist latent anchor embeddings (or landmarks) that describe the manifold topology. We also expect these anchor states to be informative enough to generate sentences, and their connections to be linguistically meaningful. We induce latent structures using a VAE with an inference model parameterized by a scaled CRF where state-word relations are modeled by the emission potential and state-state transitions are modeled by the transition matrix. The connections of words and states together form a latent network. We use the vector product between contextualized embeddings and state embeddings to parameterize the CRF potentials, bringing together the geometry of the representation space with graphical model inference. We further show that it is possible to generate paraphrases by traversing the induced network.

Our approach is fully unsupervised and the discovered latent network is intrinsic to the representation manifold, rather than imposed by external supervision, eschewing the criticism of much previous work on supervised probes (Hewitt & Liang, 2019; Chen et al., 2021). In experiments, we first verify the basic properties of RDP (bias-variance) and show its effectiveness for training latent variable models. We then visualize the discovered network based on BERT Devlin et al. (2019), demonstrating how states encode information pertaining to syntax, morphology, and semantics. Finally, we perform unsupervised paraphrase generation by latent network traversal.

# 2 RANDOMIZED DYNAMIC PROGRAMMING

Preliminaries: Speeding Summation by Randomization To motivate our randomized DP, we start with a simple setting, namely estimating the sum of a sorted list. Given a sorted list of positive numbers  $\mathbf{a}$ , naive summation  $S = a_{1} + \ldots, +a_{N}$  requires  $N - 1$  addition operations, which is expensive when  $N$  is large. Suppose we wish to reduce the number of addition operations to  $K_{1} << N$ , and we already know that the list is long-tailed (similar to how words in language follow a Zipfian distribution such that there are few very high-frequency words that account for most of the tokens in text and many low-frequency words). Then, we only need to sum over the top  $K_{1}$  values to get an efficient estimate:

$$
\hat {S} _ {1} = a _ {1} + \dots + a _ {K _ {1}} \quad \text {w h e r e} \left\{a _ {i} \right\} _ {i = 1} ^ {N} \text {s o r t e d , l a r g e t o s m a l l} \tag {1}
$$

Clearly,  $\hat{S}_1$  underestimates  $S$ . When the summands are "dense", i.e., not very different from each other, the bias is large because the top  $K_{1}$  terms do not contribute much to the sum (Fig. 1A). To correct this bias, we add samples  $a_{\delta_1},\ldots ,a_{\delta_{K_2}}$  from the remaining summands whose indices  $\delta_{i}$  are sampled from proposal  $\delta_i\sim \pmb {q} = [q_{K_1 + 1},\dots,q_N]$ :

$$
\hat {S} _ {2} = a _ {1} + \dots + a _ {K _ {1}} + \frac {1}{K _ {2}} \left(\frac {1}{q _ {\delta_ {1}}} a _ {\delta_ {1}} + \dots + \frac {1}{q _ {\delta_ {K _ {2}}}} a _ {\delta_ {K _ {2}}}\right) \quad \delta_ {i} \in \{K _ {1} + 1, \dots , N \} \tag {2}
$$

where  $K_{1} + K_{2} = K$ . Note that this is an unbiased estimator as  $\mathbb{E}[\hat{S}_2] = S$ , irrespective of how we choose  $\pmb{q}$ . Without any knowledge about  $\pmb{a}$ , the simplest proposal would be uniform, no matter what variance it induces. The more  $q_{i}$  correlates with  $a_{i}$ , the less variance  $\hat{S}_2$  has. The oracle  $q_{i}$  is proportional to  $a_{i}$ , under which  $\hat{S}_2$  becomes exact  $\hat{S}_2 \equiv S$  as  $q_{\delta_i} = a_{\delta_i} / (a_{K + 1} + \ldots + a_N)$  for all  $i$ . So, the strategy is to exploit our knowledge about  $\pmb{a}$  to construct a correlated proposal  $\pmb{q}$ . Given this estimator, we can also adjust the computation budget in order to reduce variance. When the distribution is long-tailed, we may increase  $K_{1}$  as an instance of Rao-Blackwellization (Liu et al., 2019). When the distribution is not long-tailed (enough), and top  $K_{1}$  summation underestimates significantly, we may increase  $K_{2}$  to reduce variance, provided we have a fairly accurate  $\pmb{q}$ , as an instance of importance sampling. This procedure is also discussed in Kool et al. (2020) for

![](images/2e98c1aac94c2e4f0d3ea92817d25dfd6bdc8f7046a90f3da90fb6612ee8f719.jpg)  
Figure 1: (A): Sampled summation of an array; in the dense case the proposal is important for variance reduction, while in the long-tailed case, topK summands are important; (B): core recursion step of the Randomized Forward algorithm. We get topK and sample from the proposal (black and grey bars); Errors stem from the difference (green bars) between the oracle proposal  $\tilde{a}$  and constructed proposal  $\tilde{q}$ ; (C): Inferring latent states within the BERT representation space. We parametrize the CRF factors with vector products; the relations between states and contextualized embeddings together form a latent network (Fig. 3 and 4); (D): Experimental protocol; we first study the basic properties of RDP (steps 1, 2) and then integrate RDP into a LVM for inferring the structure of the representation space (steps 3, 4). Best viewed in color.

gradient estimation. In fact, it is the underlying basis of many Monte Carlo estimators in various settings (Mohamed et al., 2020).

The Sampled Forward Algorithm Now we will show how estimator  $\hat{S}_2$  can be used to scale summation in DP. Consider a linear chain CRF which defines a discrete state sequence  $z = [z_1, \dots, z_T], z_t \in \{1, \dots, N\}$  over an input sentence  $\boldsymbol{x} = [x_1, \dots, x_T]$ . Later we will use this CRF to construct an inference model to discover latent network structures within contextualized representations. We are interested in the partition function  $Z$  which is commonly computed with the Forward algorithm, a dynamic programming algorithm that sums over the potentials of all possible state sequences. The core recursion steps are:

$$
\alpha_ {t + 1} (i) = \sum_ {j = 1} ^ {N} \tilde {a} _ {t + 1} (i, j) = \sum_ {j = 1} ^ {N} \alpha_ {t} (j) \Phi (j, i) \phi \left(x _ {t + 1}, i\right) \quad Z = \sum_ {j = 1} ^ {N} \alpha_ {T} (j) \tag {3}
$$

where  $\alpha_{t}(i)$  is the sum of all possible sequences up to step  $t$  and at state  $i$ ,  $\Phi (\cdot ,\cdot)$  is an  $N\times N$  transition matrix, and  $\phi (x_{t},i)$  is the emission potential that models how word  $x_{t}$  generates state  $i$ . We assume all potentials are positive for simplicity. When implemented on GPUs, space complexity is  $O(TN^{2})$  (see number of edges in the DP graph in Figure 1B) and it is the squared term  $N^2$  that causes memory overflows under automatic differentiation (see Appendix B for engineering details).

Our key insight is to recursively use the memory-efficient randomization of Eq. 2 to estimate Eq. 3 at every step. Given a proposal  $\tilde{q}_t$  for each step  $t$  that correlates with summands  $\tilde{a}_t$  (we discuss how to construct  $\tilde{q}_t$  in the next section), we obtain its top  $K_{1}$  index and sample  $K_{2}$  from the rest:

$$
\left[ \sigma_ {t, 1}, \dots , \sigma_ {t, K _ {1}}, \dots , \sigma_ {t, N} \right] = \arg \operatorname {s o r t} _ {i} \{\tilde {q} _ {t} (i) \} _ {i = 1} ^ {N} \tag {4}
$$

$$
\left[ \delta_ {t, 1}, \dots , \delta_ {t, K _ {2}} \right] \sim \text {C a t e g o r i c a l} \left\{\tilde {q} _ {t} \left(\sigma_ {t, K _ {1}} + 1\right), \dots , \tilde {q} _ {t} \left(\sigma_ {t, N}\right) \right\} \tag {5}
$$

where  $\tilde{q}_t(\cdot)$  are normalized to construct the categorical. Compared to Eq. 3, the key recursion of our Sampled Forward uses the top  $K_{1}$  index  $\sigma_t$  and sampled  $K_{2}$  index  $\delta_t$  to substitute the full index:

$$
\hat {\alpha} _ {t + 1} (i) = \sum_ {j = 1} ^ {K _ {1}} \alpha_ {t} \left(\sigma_ {t, j}\right) \Phi \left(\sigma_ {t, j}, i\right) \phi \left(x _ {t + 1}, i\right) + \frac {1}{K _ {2}} \sum_ {j = 1} ^ {K _ {2}} \frac {\tilde {Z} _ {t}}{\tilde {q} _ {t} \left(\delta_ {t , j}\right)} \alpha_ {t} \left(\delta_ {t, j}\right) \Phi \left(\delta_ {t, j}, i\right) \phi \left(x _ {t + 1}, i\right) \tag {6}
$$

$$
\tilde {Z} _ {t} = \sum_ {j = K _ {1} + 1} ^ {N} \tilde {q} _ {t} \left(\sigma_ {t, j}\right) \quad \hat {Z} = \sum_ {j = 1} ^ {K _ {1}} \hat {\alpha} _ {T} \left(\sigma_ {T, j}\right) + \frac {1}{K _ {2}} \sum_ {j = 1} ^ {K _ {2}} \frac {\tilde {Z} _ {T}}{\tilde {q} _ {T} \left(\delta_ {T , j}\right)} \hat {\alpha} _ {T} \left(\delta_ {T, j}\right) \tag {7}
$$

where the oracle proposal  $q_{t}^{*}$  is proportional to the actual summand  $\tilde{a}_{t}$  (Eq. 3), which is only accessible with the full Forward. So, we use the proposal weight  $\tilde{q}_{t}$  (Eq. 4) to move the computation outside the DP. In Fig. 1B, the top  $K_{1}$  summed terms correspond to black nodes. The proposal  $\tilde{q}_{t}$  corresponds to black and grey bars, and its distance from the oracle proposal  $\tilde{a}_{t}$  (which is the major

source of variance) is highlighted in green. Sampled indices are shown as blue nodes. Essentially, our Sampled Forward algorithm restricts the DP computation from the full graph to subgraphs with top and sampled edges, reducing complexity to  $O(TK^2)$  where  $K = K_1 + K_2$ . By varying  $K$ , memory complexity becomes a tradeoff between memory budget and estimation error. By induction, we can show that  $\hat{Z}$  (Eq. 7) is an unbiased estimator of  $Z$  since  $\forall t, \mathbb{E}[\hat{\alpha}_t] = \alpha_t$ . When implemented in log space, the expected  $\log \hat{Z}$  is a lower bound of the exact  $\log Z$  due to Jensen's inequality, and the variance is (trivially) reduced by  $\log(\cdot)$ . See Appendix for details on implementation (Section C), theoretical analysis (Section A), and extensions to general sum-product structures (Section D).

# 3 LATENT NETWORK TOPOLOGY IN PRETRAINED LANGUAGE MODELS

Latent States within Representation Space We now use the above technique to uncover hidden geometric structures in contextualized representations. In experiments we work with BERT (Devlin et al., 2019) and GPT2 (Radford et al., 2019), however, our method can be easily applied to other pretrained language models. Given sentence  $\pmb{x} = [x_{1},\dots,x_{T}]$ , we denote its contextualized representations as  $[\pmb{r}_1,\dots,\pmb{r}_T] = \mathrm{PLM}(\pmb{x})$ . Representations  $\pmb{r}$  for all sentences lie in one manifold  $\mathcal{M}$ , namely, the representation space of the language model. We hypothesize there exists a set of latent states  $s_1,\ldots ,s_M$  that function as anchors and outline the space topology. We emphasize that all parameters of the PLM are fixed (i.e., no fine-tuning takes place), so all learned states are intrinsic to  $\mathcal{M}$ . We focus on two topological relations: (a) state-word relations, which represent how word embeddings may be summarized by their states and how states can be explained by their corresponding words; and (b) state-state relations, which capture how states interact with each other and how their transitions denote meaningful word combinations. Taken together, these two relations form a latent network within  $\mathcal{M}$  (visualized in Fig. 3 and 4).

We adopt a minimal parametrization of the inference network so as to respect the intrinsic structure of the representation manifold without imposing strong assumptions (e.g., via regularization). Specifically, for state-word relations, we associate each word embedding  $\boldsymbol{r}_t$  with a latent state indexed by  $z_t \in \{1, \dots, N\}$  (the corresponding embedding of  $z_t$  is  $s_{z_t}$ ). For state-state relations, we assume a transition weight  $\Phi(i, j)$ . Together we have a linear-chain CRF:

$$
\log \phi (x _ {t}, z _ {t}) = \boldsymbol {r} _ {t} ^ {\mathsf {T}} \boldsymbol {s} _ {z _ {t}} \quad \log \Phi (z _ {t - 1}, z _ {t}) = \boldsymbol {s} _ {z _ {t - 1}} ^ {\mathsf {T}} \boldsymbol {s} _ {z _ {t}} \tag {8}
$$

where the dot product follows the common practice of fine-tuning contextualized representations. We use log space for numerical stability. The probability of a state sequence given a sentence is:

$$
q _ {\psi} (\boldsymbol {z} | \boldsymbol {x}) = \prod_ {t = 1} ^ {T} \Phi \left(z _ {t - 1}, z _ {t}\right) \phi \left(x _ {t}, z _ {t}\right) / Z \tag {9}
$$

Here, the only learnable parameters are state embeddings:  $\psi = [s_1,\dots,s_N]$  as we try to be faithful to the representation manifold. Note how this parametrization reconciles space geometry with graphical models. As  $N$  is large, we estimate  $Z$  with the proposed Sampled Forward (Eq. 7).

Constructing the Proposal We now return to proposal  $\tilde{q}_t$  (Eq. 4) which we construct based on a common observation that linguistic phenomena are long-tailed:

$$
\tilde {q} _ {t} (i) \propto \Phi (i) \phi \left(x _ {t}, i\right) \quad \Phi (i) = \left| \left| s _ {i} \right| \right| _ {1} \tag {10}
$$

where  $\phi(x_{t}, i)$  states that only a few states are likely to generate observation  $x_{t}$ , which is often the case in NLP (e.g., there are only a few possible POS tags for each word); and  $\Phi(i)$  models the prior probability of state  $i$ . This choice stems from the empirical observation that larger L1 norm correlates with larger dot product, and is thus more likely to be inferred. Essentially, our proposal combines local emissions  $\phi$  and global prior  $\Phi$  to approximate the  $\tilde{a}_{t}$  variables (Eq. 3) and bypass their expensive computation.

Inference and Learning We use amortized variational inference to learn  $s$ . We simply reuse the architecture from previous work Fu et al. (2020); Li & Rush (2020) and build a generative model:

$$
p _ {\theta} (\boldsymbol {x}, \boldsymbol {z}) = \prod_ {t} p \left(x _ {t} \mid z _ {1: t}, x _ {1: t - 1}\right) \cdot p \left(z _ {t} \mid z _ {1: t - 1}, x _ {1: t - 1}\right) \quad \boldsymbol {h} _ {t} = \operatorname {D e c} \left(\left[ \boldsymbol {s} _ {z _ {t - 1}}; x _ {t - 1} \right], \boldsymbol {h} _ {t - 1}\right) \tag {11}
$$

$$
p \left(x _ {t} \mid z _ {1: t}, x _ {1: t - 1}\right) = \operatorname {s o f t m a x} \left(\operatorname {F F} \left(\boldsymbol {h} _ {t}\right)\right) \quad p \left(z _ {t} \mid z _ {1: t - 1}, x _ {1: t - 1}\right) = \operatorname {s o f t m a x} \left(\operatorname {F F} \left(\left[ \boldsymbol {s} _ {z _ {t}}; \boldsymbol {h} _ {t} \right]\right)\right) \tag {12}
$$

where  $\theta$  denotes the decoder parameters,  $\mathrm{Dec}(\cdot)$  denotes the decoder (we use an LSTM),  $h_t$  denotes decoder states, and  $\mathrm{FF}(\cdot)$  denotes a feed-forward network. This autoregressive formulation essentially encourages states to be "generative", i.e., to generate sentences and themselves. We will show in experiments how this formulation lends itself to paraphrasing. We use  $q_{\psi}$  directly from Eq. 9 as our variational posterior, and optimize the following  $\beta$ -ELBO objective:

$$
\mathcal {L} _ {\mathrm {E L B O}} = \mathbb {E} _ {q _ {\psi} (z | x)} [ \log p _ {\theta} (x, z) ] - \beta \mathcal {H} (q _ {\psi} (z | x)) \tag {13}
$$

where the  $\beta$  parameter modulates the topology of the latent structure and prevents posterior collapse. We follow Fu et al. (2020) and use their Gumbel reparameterization to optimize  $q_{\psi}$ , which is more stable than the REINFORCE gradient estimator (Li & Rush, 2020).

When integrating RDP with the Gumbel reparameterization, we noticed that the gradient will only pass through the top  $K_{1}$  and sampled  $K_{2}$  states, in other words, not all states receive gradients. In this case, trading  $K_{1}$  against  $K_{2}$  amounts to exploration versus exploitation. A large  $K_{1}$  means we give gradients to high-confidence states, i.e., we exploit large local emission and global transition potentials. While increasing  $K_{2}$  means we explore low-confidence states. So, by splitting the computation budget between top  $K_{1}$  and sampled  $K_{2}$  states, we not only reduce variance for estimating the partition, but also effectively introduce different strategies for searching over the latent space.

# 4 RELATED WORK

Efficient Inference for Structured Latent Variables There has been substantial interest recently in the application of deep latent variable models (LVMs) to various language related tasks (Wiseman et al., 2018; Li & Rush, 2020), which has also exposed scalability limitations. Earlier attempts to render CRF models efficient (Sokolovska et al., 2010; Lavergne et al., 2010) either make many stringent assumptions (e.g., sparsity), rely on handcrafted heuristics for bias correction (Jeong et al., 2009), or cannot be easily adapted to modern GPUs with tensorization and parallelization. Sun et al. (2019) are closest to our work, however they only consider top  $K$  summation and consistently underestimate the partition. Chiu & Rush (2020) scale HMMs but assume words are clustered beforehand. Our approach systematically trades computation with proposal accuracy and estimation error (rather than over-compromising for efficiency). Moreover, we do not impose any hard restrictions like sparsity (Correia et al., 2020), and can accommodate dense and long-tailed distributions. Our method is inspired by randomized automatic differentiation (RAD, Oktay et al., 2020), and can be viewed as RAD applied to the DP computation graph. Advantageously, our proposal is compatible with existing efficient implementations (like Rush, 2020) since it does not change the computation graph.

Interpretability of Contextualized Representations There has been a good deal of interest recently in analyzing contextualized representations and the information they encode. This line of research, collectively known as "Bertology" (Rogers et al., 2020; Hewitt & Manning, 2019), focuses mainly on supervised probing of linguistic properties (Tenney et al., 2019), while the geometric properties of the representations have been less studied (Cai et al., 2021). A major dilemma facing this work is whether supervised linguistic probes reveal properties intrinsic to the embeddings or imposed by the supervision signal itself (Hewitt & Liang, 2019; Hall Maudslay et al., 2020; Chen et al., 2021). In this work, we do not use any supervision to ensure that the discovered network is intrinsic to the representation space.

# 5 EXPERIMENTS

In this section, we present our experimental results aimed at analyzing RDP and showcasing its practical utility (see Fig. 1). Specifically, we (1) verify the basic properties of RDP by estimating the partition function and (2) using it to train the structured latent variable model introduced in Section 3; (3) we then turn our attention to pretrained language models and examine the network induced with our approach and whether it is meaningful; and (4) we generate sentence paraphrases by traversing this network. For experiments (1, 2, 4), we use (a). pretrained GPT2 as the encoder since they are more about autoregressive language modeling and generation; (b). the MSCOCO dataset, a common benchmark for paraphrasing (Fu et al., 2019). For experiment (2), we use (a). BERT since it has been the main focus of most previous analytical work (Rogers et al., 2020); (b). the 20News dataset, a popular benchmark for training latent variable models (Grisel et al.). Across all experiments, we

![](images/f69fc3cb1284cebcb96fe10960aa586c3019ab83ae179a5539c5fc68af4c6a18.jpg)  
Figure 2: Sampled Forward vs. TopK summation (Sun et al., 2019) in different unit cases during training. Red line: target log partition. Grey line: estimates from TopK. Our method effectively corrects the bias in TopK summation with significantly less memory, and is consistent with dense and long-tailed distributions.

![](images/d3382a23ee39f5ce37efecc5297a2b233e16d47d32739996e25822dc9e9a2f2b.jpg)

![](images/108f6cdf0ae0207323fa3d897c453da38f482ff0ccd211e3bb07a6ad5e214dae.jpg)

Table 1: Results on training LVMs on MSCOCO dataset. Models are run 6 times with different random seeds.  

<table><tr><td>Model-#States</td><td>Dev NLL</td><td>Dev PPL</td><td>Test NLL</td><td>Test PPL</td></tr><tr><td>FULL-100 (Fu et al., 2020)</td><td>39.64±0.06</td><td>22.07±0.12</td><td>39.71±0.07</td><td>22.32±0.12</td></tr><tr><td>TOPK-100 (Sun et al., 2019)</td><td>39.72±0.13</td><td>22.22±0.23</td><td>39.76±0.11</td><td>22.41±0.20</td></tr><tr><td>RDP-100 (ours)</td><td>39.59±0.10</td><td>21.99±0.18</td><td>39.59±0.08</td><td>22.12±0.13</td></tr><tr><td>TOPK-2K (Sun et al., 2019)</td><td>39.81±0.30</td><td>22.43±0.44</td><td>39.84±0.31</td><td>22.52±0.59</td></tr><tr><td>RDP-2K (ours)</td><td>39.47±0.11</td><td>21.94±0.46</td><td>39.48±0.14</td><td>21.93±0.24</td></tr></table>

use an LSTM decoder with states identical to the encoder (762 for BERT base and GPT2 as in Wolf et al., 2020). More details on experiments and model settings can be found in Appendix E.

# 5.1 BASIC PROPERTIES

We examine the estimation of the partition function for three unit cases, namely dense, intermediate, and long-tailed distributions. Instead of simulating these unit cases, to make our experiments more realistic, we extract CRFs on-the-fly from different LVM training stages. We also study the effects of memory budget by setting  $K$  to 20, 200, and 400 (corresponding to 1, 10, and 20 percent of the full memory). We use TopK summation (Sun et al., 2019) as our main baseline. This method can be viewed as setting  $K_{1} = K$  and  $K_{2} = 0$  in our framework, i.e., it does not use the random sample. For training LVMs, We consider 100 and 2,000 latent states. With 100 states we are able to perform the summation exhaustively which is the same as Fu et al. (2020). Full summation with 2,000 states is intractable, so we only compare with TopK summation and use  $K = 100$ .

Estimating the Partition Function As shown in Figure 2, TopK summation always underestimates the partition. The gap is quite large in the dense case (large entropy), which happens at the initial stages of training when the model is not confident enough. The long-tailed case represents later training epochs when the model has converged and is more concentrated. Our method effectively corrects the bias, and works well in all unit cases with significantly less memory.

Training Latent Variable Models We compare different LVMs in Table 1. Following common practice, we report negative log likelihood (NLL) and perplexity (PPL). We perform an extensive search over multiple hyperparameters (e.g.,  $\beta$ , learning rate, word dropout) across multiple random seeds (3-6) and report the average performance of the best configuration for each method. Our model performs best in both 100 and 2,000 state settings. The advantage is modest (as there are no architecture changes, only different training methods) but consistent. RDP trades off exploitation (i.e., increasing  $K_{1}$ ) and exploration (i.e., increasing  $K_{2}$ ) while TopK summation always focuses on the local solutions by passing gradients through top states. Intuitively, we have the chance of discovering better latent states (i.e., larger likelihood) by randomly searching the unexplored space.

# 5.2 DISCOVERING LATENT NETWORKS FROM PRETRAINED EMBEDDINGS

We now discuss how latent structures induced with RDP reveal linguistic properties of contextualized representations. We focus on BERT Devlin et al. (2019) and set the number of latent states to

![](images/222b79acd1d125560dc1278923eb17e351077a209581c9cf78525846293fbba1.jpg)  
Figure 3: (A1): Frequent words partake in more latent states than rare words (presumably because they are polysemous); (A2 and A3): The distribution of states is also Zipfian, as most frequent states generate most words (the orange portion in A2 is almost indistinguishable); (B): t-SNE (Van der Maaten & Hinton, 2008) visualization of latent network induced from BERT; (B1): Words and their corresponding latent states. For states, the size of circle indicates frequency ( $\approx$  aggregated posterior probability) and color thickness means level of contextualization; a state with deeper blue color tends to generate content words (whose meaning is less dependent on context); lighter blue corresponds to stopwords (which are more contextualized); words are also colored by number of states ( $\approx$  number of linguistic roles); red color densities mean a word is generated by several states; (B2) and (C): sample from  $p^*(x) q_\phi(z|x)$ . Our method discovers a spectrum of meaningful states which exhibit both morphological, syntactic and semantic functionalities.

![](images/52f2fbec6d06ae47447517c17e78cc555a3821efe91094f9ccd853d57e549baf.jpg)

![](images/6a4386fc2bcc62bbb7fe508296ad9540376cc29993049859f049852fe85a7a13.jpg)

![](images/d80a044d4b899805ec54e7c8f21f168399f4032a0853e3c485650ff200cff42e.jpg)

![](images/eb407bc70be4825d491075e5e89771b54dd695e019a3f5044f46a49f8b6e4eb5.jpg)

<table><tr><td></td><td>State id</td><td>Interpretation</td><td>Corresponding Words - Occurrence</td></tr><tr><td rowspan="9">Morphology / Syntax</td><td>890</td><td>&quot;Give&quot;</td><td>give 854 | given 445 | provide 224 | gives 217 | gave 193 | giving 162 | show 128 | offer 89 | cause 88</td></tr><tr><td>1756</td><td>&quot;See&quot;</td><td>see 1721 | look 858 | seen 618 | read 302 | saw 274 | display 205 | image 199 | looks 197 | looking 196</td></tr><tr><td>665</td><td>[s]-suffix 3rd singular</td><td>##s 35 | isn 11 | comes 7 | runs 6 | remains 6 | ##ly 5 | exists 5 | contains 3 | includes 3 | becomes 3</td></tr><tr><td>1972</td><td>[s]-suffix plural</td><td>##s 649 | turks 222 | armenians 206 | jews 186 | keys 171 | muslims 151 | arabs 123 | christians 93</td></tr><tr><td>243</td><td>[ly]-suffix adverb</td><td>##ly 929 | probably 311 | clearly 254 | completely 231 | obviously 229 | certainly 222 | directly 186</td></tr><tr><td>1417</td><td>[er]-suffix - comparative</td><td>better 784 | less 530 | faster 149 | higher 126 | greater 120 | worse 105 | larger 93 | ##er 88 | longer 80</td></tr><tr><td>127</td><td>[er]-suffix - role</td><td>##er 473 | everyone 390 | user 270 | host 180 | server 136 | manager 103 | player 93 | doctor 82</td></tr><tr><td>476</td><td>Past tense</td><td>##ed 609 | ##d 437 | ##ted 156 | based 144 | caused 95 | ##ized 75 | made 61 | lost 61 | built 60</td></tr><tr><td>1556</td><td>Present continuous</td><td>##ing 1282 | running 188 | #ng 149 | #ng 104 | #ling 92 | processing 87 | killing 83 | calling 70</td></tr><tr><td rowspan="7">Semantics</td><td>1634</td><td>Month</td><td>april 169 | may 75 | apr 53 | march 40 | august 33 | june 28 | version 26 | february 22 | september 18</td></tr><tr><td>865</td><td>Religion</td><td>faith 383 | religion 377 | atheist 205 | islam 159 | religious 145 | morality 137 | christianity 87 | muslim 40</td></tr><tr><td>214</td><td>Country</td><td>germany 31 | turkish 26 | qur 26 | american 25 | greek 21 | turkey 20 | muslim 19 | london 17 | islam 16</td></tr><tr><td>1291</td><td>Literature</td><td>lines 1848 | read 701 | writes 502 | line 376 | book 319 | books 244 | write 203 | written 177 | text 171</td></tr><tr><td>1874</td><td>Computer</td><td>apple 405 | chip 386 | disk 373 | fbi 289 | encryption 197 | #eg 171 | hardware 166 | nsa 154</td></tr><tr><td>1214</td><td>Aerospace</td><td>space 182 | nasa 170 | orbit 169 | motif 136 | moon 103 | planet 95 | prism 94 | lunar 92 | venus 86</td></tr><tr><td>371</td><td>Medicine</td><td>drug 212 | food 145 | health 130 | medical 121 | disease 117 | diet 115 | cancer 113 | aids 98 | sex 83</td></tr></table>

2,000. As BERT's vocabulary size is  $32\mathrm{K}$ , one state would approximately handle 15 words in the uniform case, functioning as a type of "meta" word. After convergence, we use  $q_{\psi}$  to sample  $\pmb{z}$  for each  $\pmb{x}$  in the training set (recall we use the 20News dataset). These  $\pmb{z}$  can be viewed as samples from the aggregated posterior  $\sum_{\pmb{x}} q_{\psi}(\pmb{z}|\pmb{x}) p^{\star}(\pmb{x})$  where  $p^{\star}(\pmb{x})$  denotes the empirical data distribution. To get a descriptive summary of BERT's latent topology, we compute the following statistics on  $\pmb{z}$  samples: state frequency (Fig. 3, A3); number words corresponding to each state (Fig. 3, A2); number of states corresponding to each word (Fig. 3, A1); and state bigrams (Fig. 4). We further differentiate stopwords (e.g., in, of, am, is) from content words.

State-word Relations Figure 3 gives a first impression of how latent states spread over the representation space. Overall, we observe that the joint space is Zipfian, and this property characterizes

![](images/b1370a170886eb412651a592aa7890c5d73712b3374d4f551c8e3e577d7851e8.jpg)  
Joint Visualization of All States

![](images/25ac1fee2d86e573b259fe5d6cad4ef62c5d73977925b62a5e87ba17b8d4ce95.jpg)  
Highlighted Transitions

![](images/49f97755c9645f339725eeacdc5bdcf520e705f9b9c61d4c5b9b06c0db732a1a.jpg)  
Top 500 States, Connected by Transition Matrix

![](images/a88abea8bec999e264a7e2b2cbcec7b415c78d525f28b70391482737298f2998.jpg)  
Top 500 States, Connected by Aggregated Posterior

![](images/0f592a49b96d18f8c113dbca9be47822e0ade81b8a8f44eb29421ed8c21cabc9.jpg)

Transition Interpretation

Corresponding Bigrams - Occurrence

<table><tr><td>698-145</td><td>To + verb, infinitive</td></tr><tr><td>1712-698</td><td>Verb + to</td></tr><tr><td>665-476</td><td>Passive voice (is + v.ed)</td></tr><tr><td>476-1654</td><td>Passive voice: by</td></tr><tr><td>904-296</td><td>In prepositional phrase</td></tr><tr><td>1895-1966</td><td>In somewhere</td></tr><tr><td>476-243</td><td>Verb (past) + adverb</td></tr><tr><td>476-904</td><td>Verb (past) + in</td></tr><tr><td>243-476</td><td>Adverb + verb (past)</td></tr><tr><td>192-1417</td><td>Adverb + comparative</td></tr><tr><td>1417-1683</td><td>Comparative + noun</td></tr><tr><td>1064-476</td><td>People did</td></tr><tr><td>1572-476</td><td>Person did</td></tr></table>

to-believe 3 | to-prove 10 | to-assume 7 | to-check 6 | to-test 5 | to-claim 4 | to-argue 4   
want-to 126 | like-to 24 | wanted-to 18 | wants-to 17 | wish-to 10 | wishes-to 9 | designed-to 6   
is-defined 3 | is-supported 3 | is-produced 3 | is-created 2 | is-available 2 | is-caused 2   
written-by 13 | #d-d-by 11 | caused-by 8 | #dded-by 8 | produced-by 6 | followed-by 6 | defined-by 4   
in-fact 155 | in-reality 6 | in-particular 5 | in-short 5 | in-itself 4 | in-essence 4 | in-general 4   
in-bosnia 14 | in-soviet 9 | in-seattle 9 | in-texas 8 | in-washington 6 | in-los 6 | in-azerbaijan 5   
reacted-badly 1 | organized-electronically 1 | implemented-slightly 1 | tied-directly 1   
built-in 3 | washed-in 3 | represented-in 2 | #led-in-2 | resulted-in 2 | involved-in 2 | sealed-in 2   
intentionally-started 3 | possibly-followed 2 | basically-threw 2 | self-proclaimed 2 | heavily-armed 2   
much-less 14 | much-better 14 | much-greater 5 | much-worse 4 | much-bigger 3 | lot-better 3   
greater-risk 2 | less-money 2 | less-costly 2 | less-expensive 2 | bigger-budgets 1 | lower-costs 1   
people-married 2 | revolutionaries-armed 1 | people-burned 1 | people-got 1 | people-showed 1   
jordan-implemented 1 | taylor-visited 1 | bullock-received 1 | ryan-walked 1 | cooper-ripped 1

![](images/aa4e223af1cabfc0e0a64b4b3dfd4332afb3f0550cfe54c7fd49e859027e7023.jpg)  
Figure 4: (A1): Geometrical differences between top and tail states; most lexical variations are encoded by the top 500 states while remaining states represent the long tail; (A3 and A4): Network topology within top 500 states; in (A3) nodes are connected to their top 1 neighbor according to the transition matrix  $\Phi$  (as a proxy of the empirical prior) and in (A4) according to the most frequent bigram (as a proxy of the aggregated posterior), note how the two are correlated; (A2 and B): Highlighted bigrams and their linguistic interpretation; transitions with stopwords are more about syntax (e.g., to with infinitives or transitive verbs); transitions without stopwords are more about specific meanings. (C): paraphrasing as latent network traversal.

Input: a young man riding a skate board on top of a park

$\rightarrow$  a young man riding on the skate board at top of a park  
→ a young man riding a skate board at the top of a park  
$\rightarrow$  young man riding on top of skate board in a park

the distribution of words (A1), states (A3), and word occurrence within each state (C). We also see that the top 500 states account for most word occurrence (A2) while the remaining states model tail phenomena (A3). We conjecture this number is related to the intrinsic dimension of the data manifold (see Aghajanyan et al. 2021). The induced states encode multiple linguistic properties (Fig. 3, C). Some states are similar to a lexicon entry encoding specific words and their morphological variants; other states exhibit clustering based on morphological features (-s, -er, -ly suffix). We believe this is closely related to the fact that BERT learns embeddings over subwords. Note that the past tense cluster contains words exhibiting both regular (-ed suffix) and irregular morphology (e.g., lost and built). Finally, we also see that some states are largely semantic, similar to a conventional topic model (e.g., Computer and Medicine clusters). See Appendix E.6 for more state-word examples.

State-State Relations As shown in Fig. 4, we observe a clear geometric difference between top and tail states. Most linguistic constructions seem to be captured by the top 500 states (A1). The connections of top states are visualized in (A2-A4). From a statistical perspective, the similarity of (A3) and (A4) clearly shows how the empirical prior (encoded by the transition matrix  $\Phi$ ) matches

Table 2: Paraphrase generation on the MSCOCO dataset (Fu et al., 2019). Numbers in first block taken from Fu et al. (2020). We report model performance using BLEU 4gram (B4), self BLEU 4gram (sB4), and iBLUE (iB4). Performance is averaged over 3 random seeds.  

<table><tr><td>Model</td><td>iB4↑</td><td>B4↑</td><td>sB4↓</td></tr><tr><td>CGMH (Miao et al., 2018)</td><td>7.84</td><td>11.45</td><td>-</td></tr><tr><td>UPSA (Liu et al., 2020)</td><td>9.26</td><td>14.16</td><td>-</td></tr><tr><td>GUMBEL-CRF (Fu et al., 2020)</td><td>10.20</td><td>15.75</td><td>-</td></tr><tr><td>GPTNET-50 FULL</td><td>8.81±0.03</td><td>13.54±0.43</td><td>33.78±3.78</td></tr><tr><td>GPTNET-50 TOPK (Sun et al., 2019)</td><td>8.88±0.04</td><td>13.84±0.50</td><td>35.75±4.26</td></tr><tr><td>GPTNET-50 RDP (ours)</td><td>9.14±0.18</td><td>14.33±0.30</td><td>37.49±4.22</td></tr><tr><td>GPTNET-2K TOPK (Sun et al., 2019)</td><td>8.80±0.18</td><td>14.26±0.30</td><td>40.21±1.00</td></tr><tr><td>GPTNET-2K RDP (ours)</td><td>9.04±0.34</td><td>13.49±0.55</td><td>30.97±5.18</td></tr></table>

the aggregated posterior (coded in the bigram sample from  $q_{\psi}$ ), which is an important desideratum of generative modeling (Mathieu et al., 2019). Note that the number of edges linked to each node, again, follows a Zipfian distribution as top nodes have most of the connections. From a linguistic perspective, we see how the combination of states leads to meaningful syntactic and semantic constructions. Again, BERT encodes various syntactic configurations such as to infinitives, passive voice, and even manages to distinguish adverbials (e.g., in fact) from prepositional phrases (e.g., in Bosnia). In general, the latent network seems to have some grasp of syntax, semantic roles, and collocations. In the following section, we examine whether this inherent knowledge can be harvested for generation. See Appendix E.7 for more state transition examples.

# 5.3 PARAPHRASING THROUGH NETWORK TRVERSAL

We now study how the latent network can be usefully employed to generate paraphrases without access to parallel training instances. Given a sentence, we generate its paraphrase by conditioning on the input which we represent as a bag-of-words Fu et al. (2020) and by sampling from latent states. This amounts to traversing the latent network then fill in the traversal path to assemble a sentence, as visualized in Fig. 4 C. We instantiate our approach with a latent network learned from GPT2 representations (Radford et al., 2019) and refer to our model collectively as GPTNET.

We compare against three previous unsupervised models (first block in Table 2), including CGMH (Miao et al., 2019), a general-purpose MCMC method for controllable generation; UPSA (Liu et al., 2020), a strong paraphrasing model with simulated annealing, and GUMBEL-CRF (Fu et al., 2020), a template induction model based on a continuous relaxation of the CRF sampling algorithm. We present GPTNET variants with 50 and 2,000 states, and show results with RDP and topK, and the full summation for 50 states. Following previous work, we use iBLEU (Sun & Zhou, 2012) as our main metric, which trades off fidelity to the references (BLEU) and variation from the input (self-BLEU). Table 2 shows that RDP is superior to TopK and full summation in terms of iBLUE. GPTNet models do not outperform GUMBEL-CRF or UPSA. This is expected as these methods are highly tailored to the task and more flexible (e.g., they do not fix the encoder), while we restrict the modeling within the GPT2 representation space (to infer its structure). So, our results should be viewed as a sanity check demonstrating the latent network is indeed meaningful for generation (see Appendix E.8 for more generation examples).

# 6 CONCLUSION

In this paper, we have developed a general method for scaling the inference of structured latent variable models with randomized dynamic programming. It is a useful tool for the visualization and inspection of the intrinsic structure of contextualized representations. Experiments with BERT reveal the topological structure of its latent space: state-word connections encapsulate syntactic and semantic roles while state-state connections correspond to phrase constructions. Moreover, traversal over a sequence of states represents underlying sentence structure.

Ethics Statement As this paper inspects the internal structure of pretrained language models, it is likely that it will reveal frequent linguistic patterns encoded in the language model. Specifically, the frequent words, phrases, and sentences associated with different gender, ethnic groups, nationality, interest groups, social status, and all other factors, are likely to be revealed by our model. When calling the generative part of our model for paraphrasing, these differences are likely to exist in the generated sentences (depending on the dataset). These facts should be considered when using this model.

Reproducibility Statement A step-by-step implementation guide for our randomized forward algorithm is provided in Appendix section C. The comparison of RDP versus other possible solutions for scaling the structured models is provided in Appendix section B. A detailed description of the model architecture is provided in the Appendix section E.1. A detailed description of data processing is provided in the Appendix section E.2. A detailed description of training strategy, hyperparameter search strategy, and model selection, is provided in Appendix section E.3. A detailed description of visualization procedure is provided in Appendix section E.5. We will release code after the anonymity period.

# REFERENCES

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 7319-7328, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.568. URL https://aclanthology.org/2021.acl-long.568.  
Stefanos Angelidis, Reinald Kim Amplayo, Yoshihiko Suhara, Xiaolan Wang, and Mirella Lapata. Extractive opinion summarization in quantized transformer spaces. Transactions of the Association for Computational Linguistics, 9:277-293, 2021.  
Xingyu Cai, Jiaji Huang, Yuchen Bian, and Kenneth Church. Isotropy in the contextual embedding space: Clusters and manifolds. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=xYGNO86OWDH.  
Boli Chen, Yao Fu, Guangwei Xu, Pengjun Xie, Chuanqi Tan, Mosha Chen, and Liping Jing. Probing {bert} in hyperbolic spaces. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=17VnwXYZyhH.  
Justin Chiu and Alexander M Rush. Scaling hidden markov language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1341-1349, 2020.  
Gonçalo Correia, Vlad Niculae, Wilker Aziz, and André Martins. Efficient marginalization of discrete and structured latent variables via sparsity. Advances in Neural Information Processing Systems, 33, 2020.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423.  
Yao Fu, Yansong Feng, and John P. Cunningham. Paraphrase generation with latent bag of words. In NeurIPS, 2019.  
Yao Fu, Chuanqi Tan, Bin Bi, Mosha Chen, Yansong Feng, and Alexander M. Rush. Latent template induction with gumbel-crf. In NeurIPS, 2020.  
Yao Fu, Chuanqi Tan, Mosha Chen, Songfang Huang, and Fei Huang. Nested named entity recognition with partially-observed treecrfs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 12839–12847, 2021.

Olivier Grisel, Lars Buitinck, and Chyi-Kwei Yau. Topic extraction with non-negative matrix factorization and latent dirichlet allocation. URL https://scikit-learn.org/stable/ autoexamples/applications/plot_topics.extraction_with_nmf_lda. html.  
Rowan Hall Maudslay, Josef Valvoda, Tiago Pimentel, Adina Williams, and Ryan Cotterell. A tale of a probe and a parser. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7389-7395, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.659. URL https://aclanthology.org/2020.acl-main.659.  
John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733-2743, 2019.  
John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4129-4138, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1419. URL https://aclanthology.org/N19-1419.  
Matthew D Hoffman, David M Blei, Chong Wang, and John Paisley. Stochastic variational inference. Journal of Machine Learning Research, 14(5), 2013.  
Minwoo Jeong, Chin-Yew Lin, and Gary Geunbae Lee. Efficient inference of crfs for large-scale natural language data. In Proceedings of the ACL-IJCNLP 2009 Conference Short Papers, pp. 281-284, 2009.  
Yoon Kim, Alexander M Rush, Lei Yu, Adhiguna Kuncoro, Chris Dyer, and Gábor Melis. Unsupervised recurrent neural network grammars. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 1105-1117, 2019.  
Wouter Kool, Herke van Hoof, and Max Welling. Estimating gradients for discrete random variables by sampling without replacement. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rklEj2EFvB.  
Thomas Lavergne, Olivier Cappé, and François Yvon. Practical very large scale crfs. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pp. 504-513, 2010.  
Xiang Lisa Li and Alexander Rush. Posterior control of blackbox generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 2731-2743, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.243. URL https://aclanthology.org/2020.acl-main.243.  
Runjing Liu, Jeffrey Regier, Nilesh Tripuraneni, Michael Jordan, and Jon Mcauliffe. Raobackwellized stochastic gradients for discrete distributions. In International Conference on Machine Learning, pp. 4023-4031. PMLR, 2019.  
Xianggen Liu, Lili Mou, Fandong Meng, Hao Zhou, Jie Zhou, and Sen Song. Unsupervised paraphrasing by simulated annealing. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 302-312, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.28. URL https://aclanthology.org/2020.acl-main.28.  
Xuezhe Ma and Eduard Hovy. End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1064-1074, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1101. URL https://aclanthology.org/P16-1101.

Emile Mathieu, Tom Rainforth, Nana Siddharth, and Yee Whye Teh. Disentangling disentanglement in variational autoencoders. In International Conference on Machine Learning, pp. 4402-4412. PMLR, 2019.  
Ning Miao, Hao Zhou, Lili Mou, Rui Yan, and Lei Li. Cgmh: Constrained sentence generation by metropolis-hastings sampling. In AAAI, 2018.  
Ning Miao, Hao Zhou, Lili Mou, Rui Yan, and Lei Li. CGMH: Constrained sentence generation by metropolis-hastings sampling. Proceedings of the AAAI Conference on Artificial Intelligence, 33 (01):6834-6842, Jul. 2019. doi: 10.1609/aaai.v33i01.33016834. URL https://ods.aaai.org/index.php/AAAI/article/view/4659.  
Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning. J. Mach. Learn. Res., 21(132):1-62, 2020.  
Deniz Oktay, Nick McGreivy, Joshua Aduol, Alex Beatson, and Ryan P Adams. Randomized automatic differentiation. In International Conference on Learning Representations, 2020.  
Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, and Jian Tang. Rnnlogic: Learning logic rules for reasoning on knowledge graphs. In International Conference on Learning Representations, 2020.  
Lawerence R. Rabiner. A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2):257-286, 1989. doi: 10.1109/5.18626.  
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Unpublished manuscript, 2019.  
Anna Rogers, Olga Kovaleva, and Anna Rumshisky. A primer in bertology: What we know about how bert works. Transactions of the Association for Computational Linguistics, 8:842-866, 2020.  
Alexander M Rush. Torch-struct: Deep structured prediction library. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 335-342, 2020.  
Nataliya Sokolovska, T. Lavergne, O. Cappé, and François Yvon. Efficient learning of sparse conditional random fields for supervised sequence labeling. IEEE Journal of Selected Topics in Signal Processing, 4:953-964, 2010.  
Hong Sun and Ming Zhou. Joint learning of a dual smt system for paraphrase generation. In Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 38-42, 2012.  
Zhiqing Sun, Zhuohan Li, Haoqing Wang, Di He, Zi Lin, and Zhihong Deng. Fast structured decoding for sequence models. Advances in Neural Information Processing Systems, 32:3016-3026, 2019.  
Charles Sutton and Andrew McCallum. An introduction to conditional random fields for relational learning. Introduction to statistical relational learning, 2:93-128, 2006.  
Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R Thomas McCoy, Najoung Kim, Benjamin Van Durme, Sam Bowman, Dipanjan Das, and Ellie Pavlick. What do you learn from context? probing for sentence structure in contextualized word representations. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=SJzSgnRcKX.  
Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.  
Martin J Wainwright and Michael Irwin Jordan. Graphical models, exponential families, and variational inference. Now Publishers Inc, 2008.

Sam Wiseman, Stuart Shieber, and Alexander Rush. Learning neural templates for text generation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3174-3187, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1356. URL https://aclanthology.org/D18-1356.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38-45, Online, October 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-demos.6. URL https://aclanthology.org/2020.emnlp-demos.6.
