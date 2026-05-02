# DIVERSE BEAM SEARCH:  
DECODING DIVERSE SOLUTIONS FROM NEURAL SEQUENCE MODELS

Ashwin K Vijayakumar<sup>1</sup>, Michael Cogswell<sup>1</sup>, Ramprasaath R. Selvaraju<sup>1</sup>, Qing Sun<sup>1</sup>, Stefan Lee<sup>1</sup>, David Crandall<sup>2</sup> & Dhruv Batra<sup>1</sup>

{ashwinkv,cogswell,ram21,sunqing,steflee}@vt.edu djcran@indiana.edu,DBatra@vt.edu

$^{1}$  Department of Electrical and Computer Engineering, Virginia Tech, Blacksburg, VA, USA  
$^{2}$  School of Informatics and Computing Indiana University, Bloomington, IN, USA

# ABSTRACT

Neural sequence models are widely used to model time-series data. Equally ubiquitous is the usage of beam search (BS) as an approximate inference algorithm to decode output sequences from these models. BS explores the search space in a greedy left-right fashion retaining only the top  $B$  candidates. This tends to result in sequences that differ only slightly from each other. Producing lists of nearly identical sequences is not only computationally wasteful but also typically fails to capture the inherent ambiguity of complex AI tasks. To overcome this problem, we propose Diverse Beam Search (DBS), an alternative to BS that decodes a list of diverse outputs by optimizing a diversity-augmented objective. We observe that our method not only improved diversity but also finds better top 1 solutions by controlling for the exploration and exploitation of the search space. Moreover, these gains are achieved with minimal computational or memory overhead compared to beam search. To demonstrate the broad applicability of our method, we present results on image captioning, machine translation, conversation and visual question generation using both standard quantitative metrics and qualitative human studies. We find that our method consistently outperforms BS and previously proposed techniques for diverse decoding from neural sequence models.

# 1 INTRODUCTION

In the last few years, Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs) or more generally, neural sequence models have become the standard choice for modeling time-series data for a wide range of applications including speech recognition (Graves et al., 2013), machine translation (Bahdanau et al., 2014), conversation modeling (Vinyals & Le, 2015), image and video captioning (Vinyals et al., 2015; Venugopalan et al., 2015), and visual question answering (Antol et al., 2015). RNN based sequence generation architectures model the conditional probability,  $\operatorname{Pr}(\mathbf{y}|\mathbf{x})$  of an output sequence  $\mathbf{y} = (y_{1},\dots,y_{T})$  given an input  $\mathbf{x}$  (possibly also a sequence); where the output tokens  $y_{t}$  are from a finite vocabulary,  $\mathcal{V}$ .

Inference in RNNs. Maximum a Posteriori (MAP) inference for RNNs is the task of finding the most likely output sequence given the input. Since the number of possible sequences grows as  $|\mathcal{V}|^T$ , exact inference is NP-hard - so, approximate inference algorithms like beam search (BS) are commonly employed. BS is a heuristic graph-search algorithm that maintains the  $B$  top-scoring partial sequences expanded in a greedy left-to-right fashion. Fig. 1 shows a sample BS search tree.

Lack of Diversity in BS. Despite the widespread usage of BS, it has long been understood that solutions decoded by BS are generic and lacking in diversity (Finkel et al., 2006; Gimpel et al.,

![](images/18e752cf6b8961ab6000b09d1672f536260c4e10e9212ed659a845b3a866eed4.jpg)  
Ground Truth Captions  
A locomotive drives along the tracks amongst trees and bushes. An old fashion train with steam coming out of its pipe.  
Figure 1: Comparing image captioning outputs decoded by BS (top) and our method, Diverse Beam Search (middle) - we notice that BS captions are near-duplicates with similar shared paths in the search tree and minor variations in the end. In contrast, DBS captions are significantly diverse and similar to the variability in human-generated ground truth captions (bottom).

![](images/69aa5f9e83d66b2e33679715dd760a462c9dac80023b53952774e178b5ea59d9.jpg)  
Beam Search  
Diverse Beam Search

![](images/2557049fb1adc56a321e56dd402861d488de06f7196e4b68d5180c6ecfeea35c.jpg)  
Single engine train rolling down the tracks. A steam locomotive is blowing steam.

A steam engine train travelling down train tracks.

A steam engine train travelling down tracks.

A steam engine train travelling through a forest.

A steam engine train travelling through a lush green forest.

A steam engine train travelling through a lush green countryside

A train on a train track with a sky background.

A steam engine travelling down train tracks.

A steam engine train travelling through a forest.

An old steam engine train travelling down train tracks.

An old steam engine train travelling through a forest.

A black train is on the tracks in a wooded area. A black train is on the tracks in a wavy area.

A black trail is on the tracks in a rural area.

2013; Li et al., 2015; Li & Jurafsky, 2016). Comparing the human (bottom) and BS (top) generated captions shown in Fig. 1 demonstrates this deficiency. While this behavior of BS is disadvantageous for many reasons, we highlight the three most crucial ones here:

i) The production of near-identical beams make BS a computationally wasteful algorithm, with essentially the same computation being repeated for no significant gain in performance.  
ii) Due to loss-evaluation mismatch (i.e. improvements in posterior-probabilities not necessarily corresponding to improvements in task-specific metrics), it is common practice to deliberately throttle BS to become a poorer optimization algorithm by using reduced beam widths (Vinyals et al., 2015; Karpathy & Fei-Fei, 2015; Ferraro et al., 2016). This treatment of an optimization algorithm as a hyperparameter is not only intellectually dissatisfying but also has a significant practical side-effect – it leads to the decoding of largely bland, generic, and “safe” outputs, e.g. always saying “I don’t know” in conversation models (Kannan et al., 2016).  
iii) Most importantly, lack of diversity in the decoded solutions is fundamentally crippling in AI problems with significant ambiguity – e.g. there are multiple ways of describing an image or responding in a conversation that are “correct” and it is important to capture this ambiguity by finding several diverse plausible hypotheses.

Overview and Contributions. To address these shortcomings, we propose Diverse Beam Search (DBS) – a general framework to decode a set of diverse sequences that can be used as an alternative to BS. At a high level, DBS decodes diverse lists by dividing the given beam budget into groups and enforcing diversity between groups of beams. Drawing from recent work in the probabilistic graphical models literature on Diverse M-Best (DivMBest) MAP inference (Batra et al., 2012; Prasad et al., 2014; Kirillov et al., 2015), we optimize an objective that consists of two terms – the sequence likelihood under the model and a dissimilarity term that encourages beams across groups to differ. This diversity-augmented model score is optimized in a doubly greedy manner – greedily optimizing along both time (like BS) and groups (like DivMBest).

Our primary technical contribution is Diverse Beam Search, a doubly greedy approximate inference algorithm to decode diverse sequences from neural sequence models. We report results on image captioning, machine translation, conversations and visual question generation to demonstrate the broad applicability of DBS. Results show that DBS produces consistent improvements on both task-specific oracle and other diversity-related metrics while maintaining run-time and memory requirements similar to BS. We also evaluate human preferences between image captions generated by BS or DBS. Further experiments show that DBS is robust over a wide range of its parameter values and is capable of encoding various notions of diversity through different forms of the diversity term.

Overall, our algorithm is simple to implement and consistently outperforms BS in a wide range of domains without sacrificing efficiency. Our implementation is publicly available at https://github.com/ashwinkalyan/dbs. Additionally, we provide an interactive demonstration of DBS for image captioning at http://dbscloudv.org.

# 2 PRELIMINARIES: DECODING RNNS WITH BEAM SEARCH

We begin with a refresher on BS, before describing our generalization, Diverse Beam Search. For notational convenience, let  $[n]$  denote the set of natural numbers from 1 to  $n$  and let  $\mathbf{v}_{[n]} = [v_1,\dots ,v_n]^\intercal$  index the first  $n$  elements of a vector  $\mathbf{v}\in \mathbb{R}^m$ .

The Decoding Problem. RNNs are trained to estimate the likelihood of sequences of tokens from a finite dictionary  $\mathcal{V}$  given an input  $\mathbf{x}$ . The RNN updates its internal state and estimates the conditional probability distribution over the next output given the input and all previous output tokens. We denote the logarithm of this conditional probability distribution over all tokens at time  $t$  as  $\theta(y_{t}) = \log \Pr(y_{t} | y_{t-1}, \ldots, y_{1}, \mathbf{x})$ . To avoid notational clutter, we index  $\theta(\cdot)$  with a single variable  $y_{t}$ , but it should be clear that it depends on all previous outputs,  $\mathbf{y}_{[t-1]}$ . We write the log probability of a partial solution (i.e., the sum of log probabilities of all tokens decoded so far) as  $\Theta(\mathbf{y}_{[t]}) = \sum_{\tau \in [t]} \theta(y_{\tau})$ . The decoding problem is then the task of finding a sequence  $\mathbf{y}$  that maximizes  $\Theta(\mathbf{y})$ .

As each output is conditioned on all the previous outputs, decoding the optimal length- $T$  sequence in this setting can be viewed as MAP inference on a  $T$ -order Markov chain with nodes corresponding to output tokens at each time step. Not only does the size of the largest factor in such a graph grow as  $|\mathcal{V}|^T$ , but computing these factors also requires repetitively evaluating the sequence model. Thus, approximate algorithms are employed and the most prevalent method is beam search (BS).

Beam search is a heuristic search algorithm which stores the top  $B$  highest scoring partial candidates at each time step; where  $B$  is known as the beam width. Let us denote the set of  $B$  solutions held by BS at the start of time  $t$  as  $Y_{[t - 1]} = \{\mathbf{y}_{1,[t - 1]},\dots ,\mathbf{y}_{B,[t - 1]}\}$ . At each time step, BS considers all possible single token extensions of these beams given by the set  $\mathcal{V}_t = Y_{[t - 1]}\times \mathcal{V}$  and retains the  $B$  highest scoring extensions. More formally, at each step the beams are updated as

$$
Y _ {[ t ]} = \underset {\mathbf {y} _ {1, [ t ]}, \dots , \mathbf {y} _ {B, [ t ]} \in \mathcal {Y} _ {t}} {\operatorname {a r g m a x}} \sum_ {b \in [ B ]} \Theta (\mathbf {y} _ {b, [ t ]}) \quad s. t. \mathbf {y} _ {i, [ t ]} \neq \mathbf {y} _ {j, [ t ]} \forall i \neq j. \tag {1}
$$

The above objective can be trivially maximized by sorting all  $B \times |\mathcal{V}|$  members of  $\mathcal{Y}_t$  by their log probabilities and selecting the top  $B$ . This process is repeated until time  $T$  and the most likely sequence is selected by ranking the  $B$  complete beams according to their log probabilities.

While this method allows for multiple sequences to be explored in parallel, most completions tend to stem from a single highly valued beam – resulting in outputs that are often only minor perturbations of a single sequence (and typically only towards the end of the sequences).

# 3 DIVERSE BEAM SEARCH: FORMULATION AND ALGORITHM

To overcome this, we augment the objective in Eq. 1 with a dissimilarity term  $\Delta(Y_{[t]})$  that measures the diversity between candidate sequences, assigning a penalty  $\Delta(Y_{[t]})[c]$  to each possible sequence completion  $c \in \mathcal{V}$ . Jointly optimizing this augmented objective for all  $B$  candidates at each time step is intractable as the number of possible solutions grows with  $|\mathcal{V}|^B$  (easily  $10^{60}$  for typical language modeling settings). To avoid this, we opt for a greedy procedure that divides the beam budget  $B$  into  $G$  groups and promotes diversity between these groups. The approximation is doubly greedy - across both time and groups - so  $\Delta(Y_{[t]})$  is constant with respect to other groups and we can sequentially optimize each group using regular BS. We now explain the specifics of our approach.

Diverse Beam Search. As joint optimization is intractable, we form  $G$  smaller groups of beams and optimize them sequentially. Consider a partition of the set of beams  $Y_{[t]}$  into  $G$  smaller sets  $Y_{[t]}^g$ ,  $g \in [G]$  of  $B' = B / G$  beams each (we pick  $G$  to divide  $B$ ). In the example shown in Fig. 2,  $B = 6$  beams are divided into  $G = 3$  differently colored groups containing  $B' = 2$  beams each.

Considering diversity only between groups, reduces the search space at each time step; however, inference remains intractable. To enforce diversity efficiently, we consider a greedy strategy that steps each group forward in time sequentially while considering the others fixed. Each group can then evaluate the diversity term with respect to the fixed extensions of previous groups, returning the search space to  $B' \times |\mathcal{V}|$ . In the snapshot shown in Fig. 2, the third group is being stepped forward at time step  $t = 4$  and the previous groups have already been completed. With this staggered beamfront, the diversity term of the third group can be computed using these completions. Here we use

![](images/2c60dbb5be58873f14e82cb922bd699927687ea9576ceebfec4f4ddf07fd7685.jpg)  
Figure 2: Diverse beam search operates left-to-right through time and top to bottom through groups. Diversity between groups is combined with joint log probabilities, allowing continuations to be found efficiently. The resulting outputs are more diverse than for standard approaches.

hamming diversity, which adds diversity penalty -1 for each appearance of a possible extension word at the same time step in a previous group – 'birds', 'the', and 'an' in the example – and 0 to all other possible completions. We discuss other forms for the diversity function in Section 5.1.

As we optimize each group with the previous groups fixed, extending group  $g$  at time  $t$  amounts to a standard BS using dissimilarity augmented log probabilities and can be written as:

$$
Y _ {[ t ]} ^ {g} = \underset {\mathbf {y} _ {1, [ t ]} ^ {g}, \dots , \mathbf {y} _ {B ^ {\prime}, [ t ]} ^ {g} \in \mathcal {Y} _ {t} ^ {g}} {\operatorname {a r g m a x}} \quad \sum_ {b \in [ B ^ {\prime} ]} \Theta \left(\mathbf {y} _ {b, [ t ]} ^ {g}\right) + \lambda \Delta \left(\bigcup_ {h = 1} ^ {g - 1} Y _ {[ t ]} ^ {h}\right) \left[ y _ {b, t} ^ {g} \right], \tag {2}
$$

$$
s. t. \lambda \geq 0, \mathbf {y} _ {i, [ t ]} ^ {g} \neq \mathbf {y} _ {j, [ t ]} ^ {g} \forall i \neq j
$$

where  $\lambda$  is scalar controlling the strength of the diversity term. The full procedure to obtain diverse sequences using our method, Diverse Beam Search (DBS), is presented in Algorithm 1. It consists of two main steps for each group at each time step -

1) augmenting the log probabilities of each possible extension with the diversity term computed from previously advanced groups (Algorithm 1, Line 5) and,  
2) running one step of a smaller BS with  $B'$  beams using the augmented log probabilities to extend the current group (Algorithm 1, Line 6).

Note that the first group  $(g = 1)$  is not 'conditioned' on other groups during optimization, so our method is guaranteed to perform at least as well as a beam search of size  $B'$ .

# Algorithm 1: Diverse Beam Search

1 Perform a diverse beam search with  $G$  groups using a beam width of  $B$

2 for  $t = 1,\ldots T$  do   
3  $\begin{array}{r l} & {Y_{[t]}^{1}\leftarrow \mathrm{argmax}_{\left(\mathbf{y}_{1,[t]}\right],\dots ,\mathbf{y}_{B^{\prime},[t]}^{1})}\sum_{b\in [B^{\prime}]}\Theta (\mathbf{y}_{b,[t]}^{1})}\\ & {\mathrm{for} g = 2,\dots G\mathrm{~do}}\\ & {\left\lfloor \begin{array}{l l} & {\mathrm{/ / a u g m e n t~l o g~p r o b a b i l i t i e s~w i t h~d i v e r s i t y~p e n a l t y}} \\ & {\Theta (\mathbf{y}_{b,[t]}^{g})\leftarrow \Theta (\mathbf{y}_{b,[t]}^{g}) + \lambda \Delta (\bigcup_{h = 1}^{g - 1}Y_{[t]}^{h})[y_{b,t}^{g}] \quad b\in [B^{\prime}],\mathbf{y}_{b,[t]}^{g}\in \mathcal{Y}_{t}^{g}\mathrm{~a n d~}\lambda >0} \\ & {\mathrm{/ / p e r f o r m~o n e~s t e p~o f~b e a m~s e a r c h a r s~f o r~t h e~g r o u p}} \\ & {Y_{[t]}^{g}\leftarrow \mathrm{argmax}_{\left(\mathbf{y}_{1,[t]}^{g},\dots ,\mathbf{y}_{B^{\prime},[t]}^{g}\right)}\sum_{b\in [B^{\prime}]}\Theta (\mathbf{y}_{b,[t]}^{g})\quad \mathrm{s.t.} \mathbf{y}_{i,[t]}\neq \mathbf{y}_{j,[t]}\forall i\neq j} \end{array} \right.}$    
5   
6

7 Return set of B solutions,  $Y_{[T]} = \bigcup_{g = 1}^{G}Y_{[T]}^{g}$

# 4 RELATED WORK

Diverse M-Best Lists. The task of generating diverse structured outputs from probabilistic models has been studied extensively (Park & Ramanan, 2011; Batra et al., 2012; Kirillov et al., 2015; Prasad et al., 2014). Batra et al. (2012) formalized this task for Markov Random Fields as the DivMBest problem and presented a greedy approach which solves for outputs iteratively, conditioning on previous solutions to induce diversity. Kirillov et al. (2015) show how these solutions can be found

jointly (non-greedily) for certain kinds of energy functions. The techniques developed by Kirillov are not directly applicable to decoding from RNNs, which do not satisfy the assumptions made.

Most related to our proposed approach is the work of Gimpel et al. (2013), who applied DivMBest to machine translation using beam search as a black-box inference algorithm. Specifically, in this approach, DivMBest knows nothing about the inner-workings of BS and simply makes  $B$  sequential calls to BS to generate  $B$  diverse solutions. This approach is extremely wasteful because BS is called  $B$  times, run from scratch every time, and even though each call to BS produces  $B$  solutions, only one solution is kept by DivMBest. In contrast, DBS avoids these shortcomings by integrating diversity within BS such that no beams are discarded. By running multiple beam searches in parallel and at staggered time offsets, we obtain large time savings making our method comparable to a single run of classical BS. One potential disadvantage of our method w.r.t. Gimpel et al. (2013) is that sentence-level diversity metrics cannot be incorporated in DBS since no group is complete when diversity is encouraged. However, as observed empirically by us and Li et al. (2015), initial words tend to disproportionately impact the diversity of the resultant sequences – suggesting that later words may not be important for diverse inference.

Diverse Decoding for RNNs. Efforts have been made by Li et al. (2015) and Li & Jurafsky (2016) to produce diverse codings from recurrent models for conversation modeling and machine translation. Both of these works propose new heuristics for creating diverse M-Best lists and employ mutual information to re-rank lists of sequences. The latter achieves a goal separate from ours, which is simply to generate diverse lists.

Li & Jurafsky (2016) proposes a BS diversification heuristic that discourages beams from sharing common roots, implicitly resulting in diverse lists. Introducing diversity through a modified objective (as in DBS) rather than via a procedural heuristic provides easier generalization to incorporate different notions of diversity and control the exploration-exploitation trade-off as detailed in Section 5.1. Furthermore, we find that DBS outperforms the method of Li & Jurafsky (2016).

Li et al. (2015) introduced a novel decoding objective that maximizes mutual information between inputs and predicted outputs to penalize generic sequences. This operates on a principle orthogonal and complementary to DBS and Li & Jurafsky (2016). It works by penalizing utterances that are generally more frequent (diversity independent of input) rather than penalizing utterances that are similar to other utterances produced for the same input (diversity conditioned on input). Furthermore, the input-independent approach requires training a new language model for the target language while DBS just requires a diversity function  $\Delta$ . Combination of these complementary techniques is left as interesting future work.

# 5 EXPERIMENTS

In this section, we evaluate our approach on image captioning, machine translation, conversation and visual question generation tasks to demonstrate both its effectiveness against baselines and its general applicability to any inference currently supported by beam search. We also analyze the effects of DBS parameters, explore human preferences for diversity, and discuss diversity's importance in explaining complex images. We first explain the baselines and evaluations used in this paper.

Baselines & Metrics. We compare DBS with beam search and the following existing methods:

- Li & Jurafsky (2016): modify BS by introducing an intra-sibling rank. For each partial solution, the set of  $|\mathcal{V}|$  beam extensions are sorted and assigned intra-sibling ranks  $k \in [|\mathcal{V}|]$  in order of decreasing log probabilities,  $\theta_t(y_t)$ . The log probability of an extension is then reduced in proportion to its rank, and continuations are re-sorted under these modified log probabilities to select the top  $B$  'diverse' beam extensions.  
- Li et al. (2015): train an additional unconditioned target sequence model  $U(\mathbf{y})$  and perform BS decoding on an augmented objective  $P(\mathbf{y}|x) - \lambda U(\mathbf{y})$ , penalizing input-independent decodings.

We compare to our own implementations of these methods as none are publicly available. Both works use secondary mechanisms such as re-rankers to pick a single solution from the generated lists. Since we are interested in evaluating the quality of the generated lists and in isolating the gains due to diverse decoding, we do not implement any re-rankers, simply sorting by log probability.

We evaluate the performance of the generated lists using the following two metrics:

- Oracle Accuracy: Oracle or top  $k$  accuracy w.r.t. some task-specific metric, such as BLEU (Papineni et al., 2002) or SPICE (Anderson et al., 2016), is the maximum value of the metric achieved over a list of  $k$  potential solutions. Oracle accuracy is an upper bound on the performance of any re-ranking strategy and thus measures the maximum potential of a set of outputs.  
- Diversity Statistics: We count the number of distinct n-grams present in the list of generated outputs. Similar to Li et al. (2015), we divide these counts by the total number of words generated to bias against long sentences.

Simultaneous improvements in both metrics indicate that output sequences have increased diversity without sacrificing fluency and correctness with respect to target tasks.

# 5.1 SENSITIVITY ANALYSIS AND EFFECT OF DIVERSITY FUNCTIONS

Here we discuss the impact of the number of groups, strength of diversity, and various forms of diversity for language models. The supplement provides further discussion and experimental details.

Number of Groups (G). Setting  $G = B$  allows for the maximum exploration of the search space, while setting  $G = 1$  reduces DBS to BS, resulting in increased exploitation of the search-space around the 1-best decoding. Empirically, we find that maximum exploration correlates with improved oracle accuracy and hence use  $G = B$  to report results unless mentioned otherwise. See the supplement for a comparison and more details.

Diversity Strength  $(\lambda)$ . The diversity strength  $\lambda$  specifies the trade-off between the model score and diversity terms. As expected, we find that a higher value of  $\lambda$  produces a more diverse list; however, very large values of  $\lambda$  can overpower model score and result in grammatically incorrect outputs. We set  $\lambda$  via grid search over a range of values to maximize oracle accuracies achieved on the validation set. We find a wide range of  $\lambda$  values (0.2 to 0.8) work well for most tasks and datasets.

Choice of Diversity Function  $(\Delta)$ . In Section 3, we defined  $\Delta(\cdot)$  as a function over a set of partial solutions that outputs a vector of dissimilarity scores for all possible beam completions. Assuming that each of the previous groups influences the completion of the current group independently, we can simplify  $\Delta(\bigcup_{h=1}^{g-1} Y_{[t]}^{h})$  as the sum of each group's contributions as  $\sum_{h=1}^{g-1} \Delta(Y_{[t]}^{h})$ . In Section 3, we illustrated a simple hamming diversity of this form that penalizes selection of tokens proportionally to the number of time it was used in previous groups. However, this factorized diversity term can take various forms in our model - with hamming diversity being the simplest. For language models, we study the effect of using cumulative (i.e. considering all past time steps), n-gram and neural embedding based diversity functions. Each of these forms encode differing notions of diversity and result in DBS outperforming BS. We find simple hamming distance to be effective and report results based on this diversity measure unless otherwise specified. More details about these forms of the diversity term are provided in the supplementary.

# 5.2 IMAGE CAPTIONING

Dataset and Models. We evaluate on two datasets - COCO (Lin et al., 2014) and PASCAL-50S (Vedantam et al., 2015). We use the public splits as in Karpathy & Fei-Fei (2015) for COCO. PASCAL-50S is used only for testing (with 200 held out images used to tune hyperparameters). We train a captioning model (Vinyals et al., 2015) using the neuraltalk2 $^{1}$  code repository.

Results. Table 6 shows Oracle (top  $k$ ) SPICE for different values of  $k$ . DBS consistently outperforms BS and Li & Jurafsky (2016) on both datasets. We observe that gains on PASCAL-50S are more pronounced (7.14% and 4.65% SPICE@20 improvements over BS and Li & Jurafsky (2016)) than COCO. This suggests diverse predictions are especially advantageous when there is a mismatch between training and testing sets, implying DBS may be better suited for real-world applications.

Table 6 also shows the number of distinct n-grams produced by different techniques. Our method produces significantly more distinct n-grams (almost  $300\%$  increase in the number of 4-grams produced) as compared to BS. We also note that our method tends to produce slightly longer captions compared on average. Moreover, on the PASCAL-50S test split we observe that DBS finds more likely top-1 solutions on average - DBS obtains an average maximum log probability of -6.53 opposed to -6.91 found by BS of the same beam width. This empirical evidence suggests that using DBS as a replacement to BS may lead to lower inference approximation error.

Table 1: Oracle accuracy and distinct n-grams on COCO and PASCAL-50S datasets for image captioning at  $B = {20}$  . While we report SPICE,we observe similar trends in other metrics (reported in supplement).  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="4">Oracle Accuracy (SPICE)</td><td colspan="4">Diversity Statistics</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@20</td><td>distinct-1</td><td>distinct-2</td><td>distinct-3</td><td>distinct-4</td></tr><tr><td rowspan="4">PASCAL-50S</td><td>Beam Search</td><td>0.046</td><td>0.066</td><td>0.075</td><td>0.084</td><td>0.12</td><td>0.57</td><td>1.35</td><td>2.50</td></tr><tr><td>Li &amp; Jurafsky (2016)</td><td>0.047</td><td>0.069</td><td>0.076</td><td>0.086</td><td>0.15</td><td>0.97</td><td>2.43</td><td>5.31</td></tr><tr><td>DBS</td><td>0.050</td><td>0.071</td><td>0.079</td><td>0.090</td><td>0.18</td><td>1.26</td><td>3.67</td><td>7.33</td></tr><tr><td>Li et al. (2015)</td><td>0.041</td><td>0.062</td><td>0.071</td><td>0.079</td><td>0.13</td><td>1.15</td><td>3.58</td><td>8.42</td></tr><tr><td rowspan="4">COCO</td><td>Beam Search</td><td>0.062</td><td>0.077</td><td>0.085</td><td>0.091</td><td>0.40</td><td>1.51</td><td>3.25</td><td>5.67</td></tr><tr><td>Li &amp; Jurafsky (2016)</td><td>0.062</td><td>0.079</td><td>0.084</td><td>0.092</td><td>0.54</td><td>2.40</td><td>5.69</td><td>8.94</td></tr><tr><td>DBS</td><td>0.063</td><td>0.081</td><td>0.088</td><td>0.096</td><td>0.56</td><td>2.96</td><td>7.38</td><td>13.44</td></tr><tr><td>Li et al. (2015)</td><td>0.060</td><td>0.078</td><td>0.086</td><td>0.092</td><td>0.42</td><td>1.37</td><td>3.46</td><td>6.10</td></tr></table>

Human Studies. To evaluate human preference between captions generated by DBS and BS, we perform a human study via Amazon Mechanical Turk using all 1000 images of PASCAL-50S. For each image, both DBS and standard BS captions are shown to 5 different users. They are then asked - "Which of the two robots understands the image better?" In this forced-choice test, DBS captions were preferred over BS  $60\%$  of the time by human annotators.

Is diversity always needed? While these results show that diverse outputs are important for systems that interact with users, is diversity always beneficial? While images with many objects (e.g., a park or a living room) can be described in multiple ways, the same is not true when there are few objects (e.g., a close up of a cat or a selfie). This notion is studied by Ionescu et al. (2016), which defines a "difficulty score": the human response time for solving a visual search task. On the PASCAL-50S dataset, we observe a positive correlation  $(\rho = 0.73)$  between difficulty scores and humans preferring DBS to BS. Moreover, while DBS is generally preferred by humans for 'difficult' images, both are about equally preferred on 'easier' images. Details are provided in the supplement.

# 5.3 MACHINE TRANSLATION

We use the WMT'14 dataset containing 4.5M sentences to train our machine translation models. We train stacking LSTM models as detailed in Luong et al. (2015), consisting of 4 layers and 1024-dimensional hidden states. While decoding sentences, we employ the same strategy to replace UNK tokens. We train our models using the publicly available seq2seq-attn code repository. We report results on news-test-2013 and news-test-2014 and use the news-test-2012 to tune the parameters of DBS. We use sentence level BLEU scores to compute oracle metrics and report distinct n-grams similar to image captioning. Results are shown in Table 2 and we again find that DBS consistently outperforms all baselines.

Table 2: Quantitative results on English-German translation on the newtest-2013 and newtest-2014 datasets combined (at  $B = 20$ ).  

<table><tr><td rowspan="2">Method</td><td colspan="4">Oracle Accuracy (BLEU-4)</td><td colspan="4">Diversity Statistics</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@20</td><td>distinct-1</td><td>distinct-2</td><td>distinct-3</td><td>distinct-4</td></tr><tr><td>Beam Search</td><td>20.5</td><td>22.4</td><td>23.8</td><td>24.2</td><td>0.04</td><td>0.75</td><td>2.10</td><td>3.23</td></tr><tr><td>Li &amp; Jurafsky (2016)</td><td>20.7</td><td>22.6</td><td>24.0</td><td>24.3</td><td>0.04</td><td>0.81</td><td>2.92</td><td>4.61</td></tr><tr><td>DBS</td><td>20.8</td><td>22.9</td><td>24.4</td><td>24.8</td><td>0.06</td><td>0.95</td><td>3.67</td><td>5.54</td></tr><tr><td>Li et al. (2015)</td><td>20.7</td><td>23.1</td><td>24.4</td><td>24.6</td><td>0.04</td><td>0.86</td><td>2.76</td><td>4.31</td></tr></table>

# 5.4 DIALOG GENERATION

Dialog generation is a task that is inherently diverse as there are multiple valid ways to respond to a statement. We train a seq2seq model consisting of LSTMs as in Vinyals & Le (2015) on the Cornell Movie Dialogs Corpus (Danescu-Niculescu-Mizil & Lee, 2011) using the neuralconvo $^3$  repository. The training dataset consists of 222,182 conversational exchanges between movie characters.

Since automatic evaluation of dialog generation responses is an open research problem with existing metrics being poorly correlated with human judgement (Liu et al., 2016), we show qualitative results to demonstrate the effectiveness of DBS. Table 3 compares BS and DBS at  $\mathrm{B} = 3$ .

Table 3: Qualitative examples of dialog generation comparing best-first search, BS and DBS  

<table><tr><td>Prompt</td><td>Beam Search</td><td>Diverse Beam Search</td></tr><tr><td rowspan="3">Why do you want to be the president?</td><td>I don’t know.</td><td>I don’t know.</td></tr><tr><td>I don’t know...</td><td>Because you know!</td></tr><tr><td>I don’t know, but I am the best</td><td>Because, I have another life.</td></tr><tr><td rowspan="3">It’s such a beautiful day!</td><td>I know!</td><td>I know.</td></tr><tr><td>It’s true.</td><td>Well...</td></tr><tr><td>Yeah.</td><td>That’s not true, honey.</td></tr><tr><td rowspan="3">What’s your name?</td><td>Jacob.</td><td>Jacob.</td></tr><tr><td>Berger, my man.</td><td>Berger, darling</td></tr><tr><td>Berger, Thomas.</td><td>My mother used to hum that to me.</td></tr></table>

# 5.5 VISUAL QUESTION GENERATION

We also report results on Visual Question Generation (VQG) (Mostafazadeh et al., 2016), where a model is trained to produce questions about an image. Generating visually focused questions is interesting because it requires reasoning about multiple problems that are central to vision – e.g., object attributes, relationships between objects, and natural language. Furthermore, many questions could make sense for one image, so it is important that lists of generated questions be diverse.

We use the VQA dataset (Antol et al., 2015) to train a model similar to image captioning architectures. Instead of captions, the training set now consists of 3 questions per image. Similar to previous results, using beam search to sample outputs results in similarly worded questions (see Fig. 3) and DBS brings out new details captured by the model. Counting the number of types of questions generated (as defined by Antol et al. (2015)) allows us to measure this diversity. We observe that the number of question types generated per image increases from 2.3 for BS to 3.7 for DBS (at  $B = 6$ ).

![](images/aec55958cd66149fe0c2ab947cb21876c2ac729ff49f7433a4d83cfa761e19ae.jpg)  
Input Image  
Beam Search  
Diverse Beam Search

What sport is this? What sport is being played? What color is the man's shirt? What color is the ball? What is the man wearing? What color is the man's shorts?

What color is the man's shirt? What is the man holding? What is the man wearing on his head? Is the man wearing a helmet What is the man in the white shirt doing Is the man in the background wearing a hel How many zebras are there? How many zebras are in the photo? What is the zebra doing? What color is the grass? Is the zebra eating? Is the zebra in the wild?

![](images/1f6b792a7a36a61dea614264acef5b7138c47ed07b760853771788bcb5e9d3a6.jpg)  
Figure 3: Qualitative results on Visual Question Generation. DBS generates questions that are non-generic and belong to different question types.

How many zebras are there? How many zebras are in the photo? How many zebras are in the picture? How many animals are there? How many zebras are shown? What is the zebra doing?

# 6 CONCLUSION

Beam search is widely a used approximate inference algorithm for decoding sequences from neural sequence models; however, it suffers from a lack of diversity. Producing multiple highly similar and generic outputs is not only wasteful in terms of computation but also detrimental for tasks with inherent ambiguity like many involving language. In this work, we modify Beam Search with a diversity-augmented sequence decoding objective to produce Diverse Beam Search. We develop a 'doubly greedy' approximate algorithm to minimize this objective and produce diverse sequence decodings. Our method consistently outperforms beam search and other baselines across all our experiments without extra computation or task-specific overhead. DBS is task-agnostic and can be applied to any case where BS is used, which we demonstrate in multiple domains. Our implementation available at https://github.com/ashwinkalyan/dbs.

# REFERENCES

Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In Proceedings of European Conference on Computer Vision (ECCV), 2016. 6  
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2425-2433, 2015. 1, 8  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. Proceedings of the International Conference on Learning Representations (ICLR), 2014. 1  
Dhruv Batra, Payman Yadollahpour, Abner Guzman-Rivera, and Gregory Shakhnarovich. Diverse M-Best Solutions in Markov Random Fields. In Proceedings of European Conference on Computer Vision (ECCV), 2012. 2, 4  
Cristian Danescu-Niculescu-Mizil and Lillian Lee. Chameleons in imagined conversations: A new approach to understanding coordination of linguistic style in dialogs. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, ACL 2011, 2011. 7  
Francis Ferraro, Ishan Mostafazadeh, Nasrinand Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiadong He, Pushmeet Kohli, Dhruv Batra, and C Lawrence Zitnick. Visual storytelling. Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics - Human Language Technologies (NAACL HLT), 2016. 2  
Jenny Rose Finkel, Christopher D Manning, and Andrew Y Ng. Solving the problem of cascading errors: Approximate bayesian inference for linguistic annotation pipelines. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 618-626, 2006. 1  
K. Gimpel, D. Batra, C. Dyer, and G. Shakhnarovich. A systematic exploration of diversity in machine translation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2013. 1, 5, 11  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey E. Hinton. Speech recognition with deep recurrent neural networks. abs/1303.5778, 2013. 1  
Radu Tudor Ionescu, Bogdan Alexe, Marius Leordeanu, Marius Popescu, Dim Papadopoulos, and Vittorio Ferrari. How hard can it be? Estimating the difficulty of visual search in an image. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 7  
Anjuli Kannan, Karol Kurach, Sujith Ravi, Tobias Kaufmann, Andrew Tomkins, Balint Miklos, Greg Corrado, Laszlo Lukacs, Marina Ganea, Peter Young, et al. Smart reply: Automated reponse suggestion for email. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2016. 2  
Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 2, 6  
Alexander Kirillov, Bogdan Savchynskyy, Dmitrij Schlesinger, Dmitry Vetrov, and Carsten Rother. Inferring m-best diverse labelings in a single one. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 2, 4  
Jiwei Li and Dan Jurafsky. Mutual information and diverse decoding improve neural machine translation. arXiv preprint arXiv:1601.00372, 2016. 2, 5, 6, 7, 12, 13  
Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics - Human Language Technologies (NAACL HLT), 2015. 2, 5, 6, 7, 12, 13

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context, 2014. 6  
Chia-Wei Liu, Ryan Lowe, Iulian Vlad Serban, Michael Noseworthy, Laurent Charlin, and Joelle Pineau. How NOT to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation. 2016. URL http://arxiv.org/abs/1603.08023.8  
Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015. 7  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems (NIPS), 2013. 11  
Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, Margaret Mitchell, Xiaodong He, and Lucy Vanderwende. Generating natural questions about an image. Proceedings of the Annual Meeting on Association for Computational Linguistics (ACL), 2016. 8  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the Annual Meeting on Association for Computational Linguistics (ACL), 2002. 6  
Dennis Park and Deva Ramanan. N-best maximal decoders for part models. In Proceedings of IEEE International Conference on Computer Vision (ICCV), 2011. 4  
Adarsh Prasad, Stefanie Jegelka, and Dhruv Batra. Submodular meets structured: Finding diverse subsets in exponentially-large structured item sets. In Advances in Neural Information Processing Systems (NIPS), 2014. 2, 4  
Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 6  
Subhashini Venugopalan, Marcus Rohrbach, Jeffrey Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko. Sequence to sequence-video to text. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4534-4542, 2015. 1  
Oriol Vinyals and Quoc Le. A neural conversational model. arXiv preprint arXiv:1506.05869, 2015. 1,7  
Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. 1, 2, 6
