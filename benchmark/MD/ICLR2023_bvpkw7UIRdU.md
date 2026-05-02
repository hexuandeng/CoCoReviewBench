# ON THE USEFULNESS OF EMBEDDINGS, CLUSTERS AND STRINGS FOR TEXT GENERATOR EVALUATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

A good automatic evaluation metric for language generation ideally correlates highly with human judgements of text quality. Yet, there is a dearth of such metrics, which inhibits the rapid and efficient progress of language generators. One exception is the recently proposed MAUVE. In theory, MAUVE measures an information-theoretic divergence between two probability distributions over strings: one representing the language generator under evaluation; the other representing the true natural language distribution. MAUVE's authors argue that its success comes from the qualitative properties of their proposed divergence. Yet in practice, as this divergence is uncomputable, MAUVE approximates it by measuring the divergence between multinomial distributions over clusters instead, where cluster assignments are attained by grouping strings based on a pre-trained language model's embeddings. As we show, however, this is not a tight approximation—in either theory or practice. This begs the question: why does MAUVE work so well? In this work, we show that MAUVE was right for the wrong reasons, and that its newly proposed divergence is not necessary for its high performance. In fact, classical divergences paired with its proposed cluster-based approximation may actually serve as better evaluation metrics. We finish the paper with a probing analysis; this analysis leads us to conclude that—by encoding syntactic- and coherence-level features of text, while ignoring surface-level features—such cluster-based substitutes to string distributions may simply be better for evaluating state-of-the-art language generators.

# 1 INTRODUCTION

Probabilistic text generators have improved greatly in quality over the last years, with models producing increasingly human-like text (Yang et al., 2019; Brown et al., 2020; Raffel et al., 2020; Rae et al., 2021; Hoffmann et al., 2022). As the gap between human and model-generated text closes, the quality of our evaluation metrics becomes ever more important for determining generator quality, especially given the increasing number of user-facing systems which use language generators. While human evaluations serve as the gold standard, they are costly (in both time and money), leading researchers to rely on automatic metrics—i.e., metrics that can be measured by a computer—for the bulk of their development process.

Many automatic language generator evaluation metrics share the same underlying mechanism: the quantitative comparison of two probability distributions. Specifically, most metrics measure a difference between the distributions over strings defined by: (1) a language generation model<sup>1</sup> and (2) the natural language itself. This includes some of the most widely used language evaluation metrics:<sup>2</sup> cross-entropy (Shannon, 1948), perplexity (Jelinek et al., 1977), and (more recently) MAUVE (Pillutla et al., 2021). As typically applied to evaluate language generators, however, these metrics have a number of computational and qualitative issues (discussed in §3). Such issues manifest empirically: the most commonly used automatic metrics are known to correlate poorly with human judgements (Wiseman et al., 2017; Reiter, 2018; Sellam et al., 2020; Gehrmann et al., 2021).

A newly proposed metric stands apart, being empirically effective at measuring language generator quality: MAUVE (Pillutla et al., 2021). In theory, MAUVE measures the area under the curve formed by the divergence between two probability distributions, qualitatively mimicking a precision–recall quantification (Djolonga et al., 2020; Kynkänniemi et al., 2019). The authors attribute the success of their metric to the qualitative properties of this new class of divergences. Yet, due to this divergence being in practice uncomputable, Pillutla et al. propose an approximation to it. Specifically, rather than directly comparing the original two distributions over strings, MAUVE first clusters these strings based on the embeddings of a pre-trained language model; it then estimates the proposed divergence using the multinomial distributions over these clusters. As we will show, this approximation is bad in both theory §4 and practice §5.1—to the point that the term “approximation” is arguably a misnomer. Thus, the reasons why MAUVE works well in practice—knowledge which is important for the future development of language generator evaluation metrics—are still unknown.

In this work, we aim to uncover these reasons. To this end, we consider the axes on which MAUVE differs from other evaluation metrics: is MAUVE's success due to its new divergence metric, to its "approximation", or to both? Through an empirical study, we identify MAUVE's substitution of probability distributions over strings with distributions over embedding-based clusters as the main factor for its success. We show that mathematically, this substitution leads to a quite biased estimator of the original string-based divergences. Yet it also leads to metrics with much lower variance and stronger correlations with human judgements. In fact, all divergence measures analysed here correlate more strongly with human judgements when cluster-based distributions are used in place of the original string-based ones.

Finally, in order to understand the root of the effectiveness of these cluster-based metrics, we probe the clusters themselves. We find that word order has a strong influence on cluster assignment. Further, sentence-level permutations within texts suggest that these cluster-based metrics are susceptible to attributes such as coherence. On the other hand, basic manipulations that render text unhuman-like, such as removing all articles from the input text, do not seem to affect these divergences significantly. Together, these results lead us to conjecture that text embedding-based metrics may be favourable when estimating the quality of state-of-the-art (SOTA) language models, as SOTA models are known to (at least typically) produce grammatical text. That is, by ignoring surface-level features of text—while emphasising discourse- and coherence-level ones—clustered embeddings may simply be more easily leveraged for the evaluation of the top language generation systems. Yet these findings also suggests routes through which such metrics can be gamed, bringing into question their robustness.

# 2 DIVERGENCE METRICS FOR LANGUAGE GENERATOR EVALUATION

When evaluating language generation systems, we will first assume the existence of an unknown ground-truth distribution  $p_{\mathrm{w}}$ . This distribution is defined over strings  $\mathbf{w}$  and its domain spans  $\mathcal{W} \equiv \Sigma^{*}$ , where  $\Sigma$  is an alphabet of words and  $\Sigma^{*}$  is its Kleene closure. Second, we are given a probabilistic text generator  $q_{\mathrm{w}}$ , which is also a distribution over  $\mathcal{W}$ . An evaluation metric for a language generator  $q_{\mathrm{w}}$  can now be defined as a measure of its "distance" from  $p_{\mathrm{w}}$ :  $\Delta(p_{\mathrm{w}}, q_{\mathrm{w}})$ . In short,  $\Delta(\cdot, \cdot)$  should return high values if  $q_{\mathrm{w}}$  is a bad approximation to  $p_{\mathrm{w}}$ , and it should return low values if it is good.

It is common to use divergences when comparing two probability distributions, as divergences measure these distributions "distance".<sup>3</sup> MAUVE, for instance, proposes the use of an area under the curve (AUC) divergence to evaluate language generators. This divergence can be defined formally as:

Area Under the Curve (AUC) Divergence. The information divergence frontier (Sajjadi et al., 2018; Kynkäänniemi et al., 2019) between two distributions is a recently proposed metric. It computes the area under the curve formed by a series of Kullback-Leibler (KL) divergences parameterised by  $\lambda$ :

$$
\Delta_ {\mathrm {A U C}} \left(p _ {\mathbf {w}}, q _ {\mathbf {w}}\right) = 1 - \mathrm {A U C} \left(e ^ {- s \mathrm {K L} \left(p _ {\mathbf {w}} \mid \mid r _ {\mathbf {w}} ^ {\lambda}\right)}, e ^ {- s \mathrm {K L} \left(q _ {\mathbf {w}} \mid \mid r _ {\mathbf {w}} ^ {\lambda}\right)}\right), \quad r _ {\mathbf {w}} ^ {\lambda} = \lambda p _ {\mathbf {w}} + (1 - \lambda) q _ {\mathbf {w}} \tag {1}
$$

where  $\lambda$  is varied across the interval [0, 1], and  $s \in \mathbb{R}_{>0}$  is a strictly positive real-valued scaling constant. Note that we define the AUC divergence as  $1 - \mathrm{AUC}(\cdot, \cdot)$  so that a larger value indicates

a greater discrepancy with the reference corpus  $p_{\mathrm{w}}$ . Other (more conventional) divergence metrics are also used for evaluating language generators:

Forward Divergence. A common choice for  $\Delta (\cdot ,\cdot)$  is the cross-entropy,  $\Delta_{\mathrm{H}}(p_{\mathrm{w}},q_{\mathrm{w}})\stackrel {\mathrm{def}}{=}\mathrm{H}(p_{\mathrm{w}},q_{\mathrm{w}})$ , which is equivalent (up to an additive constant) to the forward Kullback-Leibler (KL) divergence:

$$
\Delta_ {\rightarrow} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) \stackrel {\text {d e f}} {=} \operatorname {K L} \left(p _ {\mathrm {w}} \mid \mid q _ {\mathrm {w}}\right) = \operatorname {H} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) - \operatorname {H} \left(p _ {\mathrm {w}}\right) \stackrel {(1)} {\triangleleft} \operatorname {H} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) = \Delta_ {\mathrm {H}} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) \tag {2}
$$

where we use  $\triangleleft$  to signify additive or multiplicative equivalence. (1) is true since  $\mathrm{H}(p_{\mathrm{w}})$  is constant with respect to  $q_{\mathrm{w}}$ . Since Pearson and Spearman correlations—the metrics we use to evaluate  $\Delta$ ’s quality (see §2.1)—are invariant to translational shifts, the cross-entropy and forward KL are equivalent as language generator metrics. We will refer to them interchangeably during subsequent comparisons.

Backward Divergence. Albeit much less common, another potential evaluation metric would be the backward (exclusive) KL divergence:

$$
\Delta_ {\leftarrow} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) \stackrel {\text {d e f}} {=} \operatorname {K L} \left(q _ {\mathrm {w}} \mid p _ {\mathrm {w}}\right) \tag {3}
$$

As opposed to the forward KL, though, this metric is not equivalent to a cross-entropy, as the entropy term in its definition (i.e.  $\mathrm{H}(q_{\mathrm{w}})$ ) is not constant across language models  $q_{\mathrm{w}}$ .

Exponentiated Divergence. By far, the most common choice of  $\Delta$  to evaluate language models is the perplexity:  $\Delta_{\mathrm{perp}}(p_{\mathbf{w}},q_{\mathbf{w}})\stackrel {\mathrm{def}}{=}e^{\mathrm{H}(p_{\mathbf{w}},q_{\mathbf{w}})}$ . Notably, perplexity is equivalent (up to a multiplicative constant) to an exponentiated Kullback-Leibler (KL) divergence between  $p_{\mathbf{w}}$  and  $q_{\mathbf{w}}$ , which follows from the same relationship as in Eq. (2). Given the property that both Pearson and Spearman correlations are invariant to a change in scale, the perplexity and exponentiated KL will thus be equivalent as language generator metrics. For consistency, we will use solely the exponentiated KL in our analyses:

$$
\Delta_ {\exp} \left(p _ {\mathbf {w}}, q _ {\mathbf {w}}\right) \stackrel {\text {d e f}} {=} e ^ {\mathrm {K L} \left(p _ {\mathbf {w}} \mid \mid q _ {\mathbf {w}}\right)} \tag {4}
$$

Jensen-Shannon Divergence. Note that the KL divergence is non-symmetric and unbounded. On the other hand, the Jensen-Shannon (JS) divergence—defined as the average of two KLs—is symmetric with respect to its inputs and is guaranteed to produce bounded values:

$$
\Delta_ {\mathrm {J S}} \left(p _ {\mathrm {w}}, q _ {\mathrm {w}}\right) \stackrel {\text {d e f}} {=} \frac {1}{2} \left(\mathrm {K L} \left(p _ {\mathrm {w}} \mid r _ {\mathrm {w}} ^ {. 5}\right) + \mathrm {K L} \left(q _ {\mathrm {w}} \mid r _ {\mathrm {w}} ^ {. 5}\right)\right), \quad r _ {\mathrm {w}} ^ {\lambda} = \lambda p _ {\mathrm {w}} + (1 - \lambda) q _ {\mathrm {w}} \tag {5}
$$

# 2.1 EVALUATING THE EVALUATIONS

Notably, it is not clear whether  $q_{\mathrm{w}}$  being a good approximation to  $p_{\mathrm{w}}$  in terms of an arbitrary  $\Delta(\cdot, \cdot)$  guarantees it to be a good language generator. Indeed, models that perform well in terms of standard metrics, such as perplexity, often still produce poor quality text (Holtzman et al., 2020). Thus, we are interested specifically in  $\Delta(\cdot, \cdot)$  that correlate highly with human quality judgements.

More formally, we define human quality judgements as a (potentially noisy) mapping  $\alpha(q_{\mathrm{w}})$  from a language generator to a real-valued score. For fixed  $p_{\mathrm{w}}$ , a useful metric  $\Delta(p_{\mathrm{w}}, \cdot)$  for evaluating the quality of a language generator  $q_{\mathrm{w}}$  is one whose scores correlate highly with the scores assigned to this generator by humans. This notion can be operationalised as follows. Assume we have  $N$  language generator models. Let us define:

$$
\delta_ {\text {h u m a n}} \left(q _ {\mathbf {w}} ^ {(1)}, \dots , q _ {\mathbf {w}} ^ {(N)}\right) = \left[ \boldsymbol {\alpha} \left(q _ {\mathbf {w}} ^ {(1)}\right), \dots , \boldsymbol {\alpha} \left(q _ {\mathbf {w}} ^ {(N)}\right) \right] \tag {6}
$$

$$
\boldsymbol {\delta} _ {\text {m e t r i c}} \left(q _ {\mathbf {w}} ^ {(1)}, \dots , q _ {\mathbf {w}} ^ {(N)}\right) = \left[ \Delta \left(p _ {\mathbf {w}}, q _ {\mathbf {w}} ^ {(1)}\right), \dots , \Delta \left(p _ {\mathbf {w}}, q _ {\mathbf {w}} ^ {(N)}\right) \right] \tag {7}
$$

We then quantify a metric's usefulness on a specific natural language task (and its distribution  $p_{\mathrm{w}}$ ) as:

$$
\operatorname {q u a l i t y} \left(\Delta , p _ {\mathrm {w}}\right) = \left| \operatorname {c o r r} \left(\delta_ {\text {h u m a n}}, \delta_ {\text {m e t r i c}}\right) \right| \tag {8}
$$

# 3 INFELICITIES AND APPROXIMATIONS

There are several issues, both computational and qualitative, with using the divergences presented in §2 to evaluate language generators. We now review these issues, along with both commonly-used and newly-proposed methods to address them via approximations.

# 3.1 NECESSITY OF FULL SUPPORT

A well-known property of the (forward) KL divergence between two distributions  $p_{\mathrm{w}}$  and  $q_{\mathrm{w}}$  is that it is infinite for any  $q_{\mathrm{w}}$  that assigns zero probability to an event in the support of  $p_{\mathrm{w}}$  (i.e., for which  $p_{\mathrm{w}}(\mathbf{w}) > 0$ ). Two of the above metrics will thus be infinite in the absence of full support by  $q_{\mathrm{w}}$ : the exponentiated and forward divergences. The above is often not an issue: most neural language models cannot assign zero probability to any string due to the final softmax operation typically used to produce their output. However, these same models are often used with decoding strategies that prune the space  $\mathcal{W}$ : e.g., both top-  $k$  and nucleus sampling modify the original distribution  $q_{\mathrm{w}}$  such that strings which do not meet a certain criteria are reassigned zero probability. While top-  $k$  and nucleus sampling typically lead to systems with better human evaluations, they will likely be given an infinitely bad score by both  $\Delta_{\mathrm{exp}}$  and  $\Delta_{\rightarrow}$ , which we argue is too harsh a penalty for a perhaps otherwise good language generator.

# 3.2  $p_{\mathbf{w}}$  IS UNKNOWN

In practice, we do not have access to the true distribution  $p_{\mathbf{w}}$ . Rather, we are typically given a corpus  $\{\mathbf{w}_n^{p_{\mathbf{w}}}\}_{n=1}^N$ , whose instances we assume to be sampled i.i.d. from  $p_{\mathbf{w}}$ . The common approach to address this shortcoming is (when possible) to derive a statistical estimator  $\hat{\Delta}$  that uses this corpus to approximate  $\Delta$ . There are two common strategies for building such estimators: Monte Carlo estimation and plug-in estimation.

# 3.2.1 MONTE CARLO ESTIMATION

Our i.i.d. assumption w.r.t. samples  $\mathbf{w}^{p_{\mathbf{w}}}\sim p_{\mathbf{w}}$  allows us to derive a Monte Carlo estimator for certain divergences. We start with the forward KL divergence—present in both  $\Delta \rightarrow$  and  $\Delta_{\mathrm{exp}}$ :

$$
\widehat {\mathrm {K L}} \left(p _ {\mathbf {w}} \mid \mid q _ {\mathbf {w}}\right) \stackrel {\text {d e f}} {=} \frac {1}{N} \sum_ {n = 1} ^ {N} \log \frac {p _ {\mathbf {w}} \left(\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}}\right)}{q _ {\mathbf {w}} \left(\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}}\right)} = - \frac {1}{N} \sum_ {n = 1} ^ {N} \log q _ {\mathbf {w}} \left(\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}}\right) + \text {c o n s t} \tag {9}
$$

where  $\mathrm{const} \in \mathbb{R}$  is constant with respect to  $q_{\mathbf{w}}$ . Given this approximation, we can estimate both  $\Delta_{\rightarrow}$  and  $\Delta_{\mathrm{exp}}$  (e.g.  $\Delta_{\rightarrow}(p_{\mathbf{w}},q_{\mathbf{w}}) \approx -\frac{1}{N}\sum_{n = 1}^{N}\log q_{\mathbf{w}}(\mathbf{w}_n^{p_{\mathbf{w}}})$ ). Unfortunately, the backward, JS, and AUC divergences are not as straightforward to estimate, as they require explicit knowledge of  $p_{\mathbf{w}}$  (instead of just samples from it; see App. A). This issue motivates the use of our next set of estimation techniques.

# 3.2.2 PLUG-IN ESTIMATION

The second approach<sup>4</sup> we may consider is building an approximation of  $p_{\mathbf{w}}$  itself to use in the formulas given in §2. Specifically, we construct a density estimator for  $p_{\mathbf{w}}$  (which we denote as  $\widehat{p}_{\mathbf{w}}$ ) and "plug it into" any given  $\Delta$ . However, this is a bit circular: the task of building a language generator  $q_{\mathbf{w}}$  itself is often framed as density estimation of  $p_{\mathbf{w}}$ . Thus, if we think  $q_{\mathbf{w}}$  is the best estimator for  $p_{\mathbf{w}}$ , we should logically use it in our plug-in estimator. Yet, using  $q_{\mathbf{w}}$  would be nonsensical; by the definition of a divergence, it would always lead to the lowest possible value of  $\Delta$ , e.g.,  $\Delta \rightarrow (q_{\mathbf{w}}, q_{\mathbf{w}}) = 0$ . To use plug-in estimation in this setting, we should therefore choose a different estimator for  $p_{\mathbf{w}}$ , e.g., from a family of density estimators that differs from those used to create  $q_{\mathbf{w}}$ . More formally, we consider a function  $\pi$  which takes a corpus as input and produces a (queryable) distribution  $\widehat{p}_{\mathbf{w}} \stackrel{\mathrm{def}}{=} \pi(\{\mathbf{w}_{n}^{p_{\mathbf{w}}} \}_{n=1}^{N})$ . This function typically includes a secondary model, e.g., an  $n$ -gram model or neural network trained on the corpus  $\{\mathbf{w}_{n}^{p_{\mathbf{w}}} \}_{n=1}^{N}$ .

Our chosen  $\pi$  may introduce biases (e.g., from the inductive biases of the model parameterising  $\pi$ ) into our metrics' estimation. To balance out such biases, we may consider using the same method to create an approximation  $\widehat{q}_{\mathbf{w}}$ , rather than directly querying  $q_{\mathbf{w}}$ . We thus consider the following plug-in estimator for the backward divergence:

$$
\widehat {\Delta} _ {\leftarrow} \left(\left\{\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}} \right\} _ {n = 1} ^ {N}, q _ {\mathbf {w}}\right) \stackrel {\text {d e f}} {=} \widehat {\mathrm {K L}} \left(\widehat {q} _ {\mathbf {w}} \mid \mid \widehat {p} _ {\mathbf {w}}\right) \tag {10}
$$

Plug-in estimators for  $\Delta_{\mathrm{JS}}$  and  $\Delta_{\mathrm{AUC}}$  are defined similarly. Further, if  $\widehat{q}_{\mathbf{w}}$  is a smoothed approximation to the original  $q_{\mathbf{w}}$ , using it may also mitigate the issues discussed in §3.1. We thus also compute estimators for the forward/exponentiated divergences using plug-in estimators, e.g.,:

$$
\widehat {\Delta} _ {\rightarrow} \left(\left\{\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}} \right\} _ {n = 1} ^ {N}, q _ {\mathbf {w}}\right) \stackrel {\text {d e f}} {=} - \frac {1}{N} \sum_ {n = 1} ^ {N} \log \widehat {q} _ {\mathbf {w}} \left(\mathbf {w} _ {n} ^ {p _ {\mathbf {w}}}\right) \tag {11}
$$

Unfortunately, most functions  $\pi$  cannot produce a good estimate of  $p_{\mathbf{w}}$  using only a small corpus. While the best available language models are a class of  $\pi$  typically trained on millions (if not billions) of sentences, a standard evaluation set is quite small—on the order of one to ten thousand sentences—and we cannot expect  $\pi$  to provide a good  $\widehat{p}_{\mathbf{w}}$  when fit using only such a small dataset. Accordingly, depending on our choice of  $\pi$ , this class of metrics may be either high variance or high bias, both of which are problematic.

# 3.3 A CLUSTERING-BASED APPROXIMATION

For the  $\widehat{\Delta}$  above that require density estimators for  $p_{\mathrm{w}}$  and/or  $q_{\mathrm{w}}$ , our choice of  $\pi$  will have a large effect on its value. We may thus wish to rethink our approximation technique altogether, and instead work with different distributions for which we can create lower variance density estimators. This is the approach used by Pillutla et al. (2021) when approximating  $\Delta_{\mathrm{AUC}}$ . Specifically, instead of computing the above metrics on the original distributions  $p_{\mathrm{w}}$  and  $q_{\mathrm{w}}$ , they use the cluster-based distributions  $p_{c}$  and  $q_{c}$ . Given a pre-trained language model, these cluster-based distributions are defined as:

$$
p _ {c} (c) = \sum_ {\mathbf {w} \in \mathcal {W}} p _ {\mathbf {w}} (\mathbf {w}) \mathbb {1} \left\{c = \phi \left(\mathrm {P L M} (\mathbf {w})\right) \right\} \tag {12}
$$

where  $\mathrm{PLM}(\cdot)$  takes as input an utterance  $\mathbf{w}$  and outputs an embedding  $\mathbf{r} \in \mathbb{R}^d$ , and  $\phi(\cdot)$  is a pre-trained clustering function. Given these distributions, we can evaluate cluster-based versions of all the divergences above, simply by substituting the original  $p_{\mathbf{w}}$  and  $q_{\mathbf{w}}$  with the new  $p_c$  and  $q_c$ .

# 4 ANALYSING MAUVE'S DIVERGENCES

To compute their proposed metric, Pillutla et al. (2021) employ a number of the approximation techniques discussed in the previous section. We now take a closer look at the biases introduced by these approximations, specifically focusing on the substitution of cluster distributions. For simplicity, we focus on the bias such a substitution introduces to  $\mathrm{KL}(p_{\mathrm{w}}||q_{\mathrm{w}})$  -a computation involved in  $\Delta_{\mathrm{AUC}}$ . This divergence can be decomposed as:

$$
\underbrace {\operatorname {K L} \left(p _ {\mathrm {w}} \mid q _ {\mathrm {w}}\right)} _ {\text {s t r i n g - b a s e d K L}} \stackrel {(1)} {=} \operatorname {K L} (p (c) \mid | q (c)) + \underbrace {\operatorname {K L} \left(p (\mathbf {w} \mid c) \mid | q (\mathbf {w} \mid c)\right)} _ {\geq 0} \geq \underbrace {\operatorname {K L} \left(p _ {c} \mid | q _ {c}\right)} _ {\text {c l u s t e r - b a s e d K L}} \tag {13}
$$

where (1) follows from the fact that  $p(c, \mathbf{w}) = p(\mathbf{w})$ , which is true because the cluster assignment is deterministic, i.e.:  $p(c \mid \mathbf{w}) = \mathbb{1} \left\{ c = \phi(\mathrm{PLM}(\mathbf{w})) \right\}$ . See the full decomposition of this equation in App. B. Notably, as KL divergences are always non-negative, these cluster-based metrics are negatively biased, lower-bounding the string-based ones. Further, the actual measurement is done on the distribution over cluster assignments  $p(c)$ ; the distribution  $p(\mathbf{w} \mid c)$  is completely ignored.

Assuming a reasonable number of clusters is used when defining  $p_c$ , however, these cluster distributions should be easier to approximate than string distributions—due to the sheer size of the support alone. Consequently, the variance of cluster-based metrics should be lower, at the cost of the bias introduced by this substitution. Further, it is not clear whether this bias is inherently bad when evaluating the quality of language generators: the answer to this question must be determined empirically (by measuring the correlation in Eq. (8)). To this end, we now provide an empirical comparison between string and cluster-based language generation evaluation.

# 5 EXPERIMENTS

Setup. We follow the setup of Pillutla et al. throughout our experiments. For our language generators, we compare 4 model architectures (all variants of GPT-2), each under two decoding strategies,

giving us a total of 8 systems. Explicitly, we compare the small, medium, large, and XL versions of GPT-2, decoding strings using either ancestral or nucleus sampling. We use a nucleus probability of 0.9 for small and medium, while 0.95 for large and XL GPT-2's. As human-generated samples  $\{\mathbf{w}_n^{p_{\mathbf{w}}}\}_{n = 1}^N$ , we use  $5k$  strings taken from WebText's test set, all in English. This restriction to English text is a notable limitation of our work; future work should verify that findings hold across languages. As model-generated text  $\{\mathbf{w}_n^{q_{\mathbf{w}}}\}_{n = 1}^N$ , we sample  $5k$  strings from each of our evaluated systems, conditioning our models on the first 10 words of human-generated strings before sampling.

String-based Approximations  $\widehat{p}_{\mathbf{w}}$ . To compute our string-based divergences, we require a secondary language model  $\widehat{p}_{\mathbf{w}}$  to estimate  $p_{\mathbf{w}}$ . Further, following the issues highlighted in §3, we will also rely on a secondary language model  $\widehat{q}_{\mathbf{w}}$  to estimate  $q_{\mathbf{w}}$ . We will use  $n$ -gram models for these approximations. Specifically, we use Kneser-Essen-Ney smoothed 5-gram models, as implemented in KenLM (Ney et al., 1994; Heafield, 2011). We choose  $n$ -gram models explicitly because—while they are by no means SOTA language models—they should have inductive biases which are different from the models we are trying to evaluate. We present results using LSTM-based estimators in App. E. When computing  $\Delta_{\mathrm{AUC}}$ , we use a scaling constant  $s$  of 0.2.

Cluster-based Approximations  $\widehat{p}_c$ . Cluster-based distributions, as presented in Eq. (12), are defined by a choice of PLM( $\cdot$ ) and pre-trained clustering function  $\phi(\cdot)$ . We rely on GPT-2 xL as our PLM, and use  $K$ -means as our clustering function. Specifically, we first extract embeddings from the last word in each sentence using GPT-2 xL and then use PCA to reduce their dimensionality (keeping 90% of the original variance explained). We then train  $K$ -means (with  $K = 500$ ) on a joint set of GPT-2 embeddings extracted from the 5k human-generated strings, and 5k model-generated sentences. Finally, we approximate  $\widehat{p}_c$  and  $\widehat{q}_c$  by computing the frequency with which strings (among these 5k used ones) are assigned to each cluster. To avoid infinite divergence measures, we estimate distributions using Laplace smoothing with  $\alpha = 1$  (which is equivalent to imposing a Dirichlet distributed prior with  $\alpha = 1$  over the cluster allocation). When computing  $\Delta_{\mathrm{AUC}}$ , we use a scaling constant  $s$  of 5.67

# 5.1 DOES  $p_c$  APPROXIMATE  $p_{\mathbf{w}}$ ?

Our first experiment tries to identify whether  $p_{c}$  and  $q_{c}$  provide faithful approximations of  $p_{\mathrm{w}}$  and  $q_{\mathrm{w}}$ . To this end, we compare both  $\widehat{q}_{\mathbf{w}}$  and  $\widehat{q}_c$  to the true  $q_{\mathrm{w}}$ , i.e., the language generator under evaluation. Explicitly, we compute the Spearman correlations between the probabilities assigned by each model to the strings in  $\{\mathbf{W}_n^{q_{\mathbf{w}}}\}_{n = 1}^N$ .

Fig. 1 presents these correlations. We see that—despite being estimated on very little data—probability estimates from our  $n$ -gram models correlate strongly with the ground-truth probabilities of  $q_{\mathrm{w}}$ ; this result holds for all

![](images/e6f8d71ed314b2a5a081bf0a683cd9cab835de7f97b9be9550df8aaf66c5809f.jpg)  
Figure 1: Correlations between the true  $q_{\mathbf{w}}$  and the estimated  $\widehat{q}_c$  and  $\widehat{q}_{\mathbf{w}}$ .

four GPT-2 architectures. On the other hand, our cluster-based probabilities consistently present negative correlations with  $q_{\mathrm{w}}$ . This result has an important implication: if cluster distributions do not correlate with  $q_{\mathrm{w}}$ , then  $\mathrm{KL}(\widehat{p_c} ||\widehat{q_c})$  is likely a poor estimate of  $\mathrm{KL}(p_{\mathrm{w}}||q_{\mathrm{w}})$ . This further implies that the approximation used by Pillutla et al. is not an accurate estimate of  $\Delta_{\mathrm{AUC}}(p_{\mathrm{w}},q_{\mathrm{w}})$ , which brings into question whether this new divergence is really responsible for MAUVE's success.

# 5.2  $\Delta$  AS TEXT EVALUATION METRICS

We now compare how various string- and cluster-based divergence measures correlate with human judgement scores. In short, Fig. 2 shows that all divergences do better when estimated with cluster distributions. These results evince that MAUVE's (Pillutla et al., 2021) high correlations with human judgements (represented here as  $\widehat{\Delta}_{\mathrm{AUC}}(p_c,q_c)$ ) are mainly due to their use of cluster-based approxi

![](images/f2a7eeb9bd356d7543a862b6504b76e0f10e64febdfec6ad87cab035cb591edc.jpg)  
(a) Interestingness

![](images/109ff75dee742b240b17daf5b5c937068c01df438bd2ac347e258cce5cffd815.jpg)  
Figure 2: Correlations between string- and cluster-based divergences with human judgement scores. Legend:  $\Delta_{\mathrm{exp}}$  in dark green;  $\Delta \rightarrow$  in orange;  $\Delta_{\leftarrow}$  in blue;  $\Delta_{\mathrm{JS}}$  in pink;  $\Delta_{\mathrm{AUC}}$  in lime green.  
(b) Sensibility

![](images/445a34189ae2ff64d23c164c444bd8f66c20bb5d871947921c79cd099b15c588.jpg)  
(c) Human likeness

mations  $(p_c, q_c)$ , rather than to their proposed divergence  $\Delta_{\mathrm{AUC}}$ . In fact, we see slight improvements over  $\widehat{\Delta}_{\mathrm{AUC}}$  when using the divergences  $\widehat{\Delta}_{\leftarrow}$  and  $\widehat{\Delta}_{\mathrm{JS}}$  instead. Furthermore, cluster-based divergences appear to be more stable, exhibiting smaller variances across random seeds. Collectively, our results suggest that cluster-based divergences may produce better metrics of text quality than string-based divergences. This motivates the following two questions, which we subsequently address. What aspects of natural language are captured by  $p_c$ ? And what aspects are overlooked by ignoring  $p(\mathbf{w} \mid c)$ ?

# 6 PROBING CLUSTERS

To better understand the aspects of natural language that our cluster distributions encode, we must first understand how  $\phi(\mathsf{PLM}(\cdot))$  partitions the string space  $\mathcal{W}$ . In other words, we must understand what components of natural language—e.g., semantics, syntactic attributes, or surface features—lead to strings being assigned to similar or different clusters. Such an analysis should provide a deeper insight into the actual similarity being measured by cluster-based divergences (while also revealing how such a metric might be gamed). To this end, we probe (Alain & Bengio, 2016) these clusters for a number of linguistic attributes—including subject matter, sentiment, prose style, word order, basic grammaticality and document length—and look at how these different attributes affect both cluster assignment and the resulting divergence scores. Notably, we probe cluster assignments directly—without relying on any diagnostic classifiers (Adi et al., 2017). Our probing analyses are thus exempt from most recent criticism against probing methodologies (Hewitt & Liang, 2019; Pimentel et al., 2020a;b; Ravichander et al., 2021; Elazar et al., 2021).

# 6.1 FINDING FEATURES  $p_c$  ENCODES

Setup. We look at texts annotated with different attributes in order to explore correlations between the presence of these attributes and cluster membership. Specifically, we analyse texts': sentiment, authorship, and topic (using the Yelp Polarity, News Category, and 20 NewsGroup datasets, respectively). Further details on datasets are provided in App. D. For each of these classification datasets, we learn cluster-category distributions using the standard training split; all evaluations are performed on test splits. Explicitly, we first learn a partitioning  $\phi (\cdot)$  of the embedding space (w.r.t a language model  $\mathrm{PLM}(\cdot))$ . Each cluster is then labelled with the majority category represented in that cluster by training examples; text categories in the test set are then predicted using this labelling, depending on which of the clusters the example falls into.

For comparison's sake, we use four language models as PLM( $\cdot$ ): GPT-2 with small, medium, large, and XL architectures. Results using embeddings from BERT (Devlin et al., 2019) can be found in App. E. Further, we use two methods for learning clusters:

-  $\phi(\cdot)$  Learned on WebText. We train the PCA and  $K$ -means functions using the same procedure as in §5.2 (again relying on WebText's test set for our data), mimicking the setting under which our partitions of the embedding space would be learned in practice.<sup>9</sup>  
-  $\phi(\cdot)$  Learned on Training Set. We train the PCA and  $K$ -means clustering functions on the analysed dataset's training set. This setting studies the partitioning our clustering functions have the capacity to learn in an ideal setting, i.e., where the attribute in question is one of the main differentiating factors between texts.

![](images/f80d950b6e6b5add2551969485bb32f8f503ed925dc84439cdc896e9a659551b.jpg)  
(a) Sentiment

![](images/6d00fe93f67a47a3f2981f0a4955a7de88b11f7f2607395a5c7707b17dbe2fc8.jpg)  
Figure 3: Accuracy when predicting different attributes of text from their cluster assignments. Assignments (i.e.  $\phi(\cdot)$ ) are learned using text from either WebText, or the training set of the respective classification datasets. Dashed lines represent baseline accuracies, i.e., always guessing the majority class.  
(b) Authorship

![](images/3ccf2ff7c079a614fdd7abd4258cec73c512ecb8f5b9d5f5f5618d40c0e6f078.jpg)  
(c) Topic

Results. In Fig. 3a, we see that, at least for large numbers of clusters, cluster assignment is indeed indicative of a text's sentiment. Interestingly, this is the case even when clusters are trained on data that is not particularly polar in sentiment (i.e. on WebText). On the other hand, we are only able to predict author and topic (with reasonable accuracy) when clusters are learned on text data with authorship and topic as distinguishing factors. These results indicate that, while writing style and subject matter are captured by the text embeddings, they likely were not being used as distinguishing features between corpora in our cluster-based divergences. We further see that, in all classification settings, the capacity to encode these analysed attributes appears to increase with model size, perhaps suggesting the embedding spaces of larger models decomposes along higher-level features of text.

# 6.2 HOW TEXT FEATURES IMPACT  $\Delta$

We next assess how changing different features of our evaluated text impacts divergence scores. Specifically, we look at the impact of: text truncation; article removal; stopwords removal; sentence-level permutations; and word-level permutations.

Setup. We follow a similar setup to §5. In order to create a more controlled setting, we primarily consider human-generated text (i.e. the  $5k$  human-written articles in WebText's test set). We take the first 2500 articles of this dataset as our reference corpus  $\{\mathbf{w}_n^{p_{\mathbf{w}}}\}_{n=1}^N$ . We then use the remaining 2500 reference strings as the comparison corpus, i.e., in place of the model-generated text that we would typically evaluate  $\{\mathbf{w}_n^{q_{\mathbf{w}}}\}_{n=1}^N$ . In order to explore how changing specific features of text affects  $\Delta$  w.r.t. the reference corpus, we compute scores when making the following modifications to the comparison corpus:

- No modification  $(p)$ . This is a baseline experiment where we keep the original strings as are.  
- Text Truncation  $(p_{\mathrm{short}})$ . We truncate texts to  $1/3$  of their original length.  
- Article Removal ( $p_{\text{no\_art}}$ ). We remove all articles (i.e., 'a', 'an' and 'the').  
- Stopwords Removal ( $p_{\mathrm{no\_stop}}$ ). We remove all stopwords (e.g., 'that' or 'so').<sup>10</sup>  
- Sentence-level Permutation ( $p_{\mathrm{swap}}$ ). We permute the first halves of texts (as delineated by sentences) across the entire corpus (i.e. randomly reassigning the strings' first halves).  
- Word-level Permutation  $(p_{\mathrm{rand}})$ . We randomly permute all the words in a text.  
- GPT-2 Baseline  $(q)$ . As an extra baseline, we also compute the divergence score when using the first 2500 generations from GPT-2 XL.

Results. Fig. 4 shows that certain alterations—such as completely removing articles from the evaluated text—have almost no impact on the divergence between our reference and test corpora for various  $\Delta$ . In fact, text without any articles is judged as better than GPT-2 XL's by most of the cluster-based divergences. Further, while this perturbation undoubtedly affects the text's fluency, it has less of an effect on this divergence than, e.g., truncating texts. This is arguably undesirable: A metric of text quality should place more emphasis on fluency than surface statistics, such as length.

On the other hand, our metrics deem text with stopwords removed as utterly different from the reference. Permuting words within texts has a similar effect, demonstrating that, at least to some extent, the embedding space captures notions of syntax and grammaticality, rather than pure unigram

![](images/8055da3ff01c2435e6572c261d878a40e3ce4c2eb9123f2362ee8dbd72b705c9.jpg)  
Figure 4: Divergence measures between two corpora: the reference text is unmodified while the comparison text undergoes perturbation. Higher value indicates a greater discrepancy according to  $\Delta$ .

statistics. The drop in scores shown when performing sentence-level permutations likewise suggests that the clusters capture some notion of coherence.

These results inspire us to investigate which surface features of text are encoded in embedding clusters. Following our setup in §6.1, we look at whether clusters encode the percentage of stopwords or punctuation in texts. We use solely the WebText dataset to train our clustering functions in this setting. We then compute the average percentage of stopwords or punctuation per cluster in half of our strings. Finally, we use these pre-computed averages when predicting the percentages in the other half, computing this prediction's  $R^2$  (i.e. the percentage of explained variance).

Interestingly, Fig. 5 shows that larger PLMs—which are often claimed to provide better representations of language—do encode more information about such surface features

than smaller models. This could simply be due to the fact that the embeddings from larger PLMs are typically of a larger dimension and, thus, have the capacity to encode additional (perhaps "less critical") attributes of text. Further, while these attributes do not appear to be differentiating factors when partitioning the embedding space into a small number of clusters, they become relevant when partitioning into a larger number of clusters. Even with several clusters and large PLMs, though, the  $R^2$  values we find are still quite small, at around 0.20.

![](images/a1b7bcf96137a1614f5ca89815b447ba8985d5af31b2d34ddcbfabc1a0bb2617.jpg)  
Figure 5:  $\mathbb{R}^2$  when using cluster assignments to predict  $\%$  of tokens in a text that are either punctuation or stopwords.

These results—along with those of §6.1—suggest cluster-based divergences are more sensitive to syntax- and coherence-related properties of the target text than to its superficial features. The opposite, however, might be said of our string-based distributions. As current SOTA language models typically produce grammatical text, being invariant to surface statistics may perhaps be a feature—as opposed to a bug—when trying to assess the quality of the text they produce. We thus conjecture that this might be what drives these clusters' effectiveness in assessing text quality. Yet, this may also reveal potential ways in which such metrics can be gamed, bringing these divergences' robustness into question.

# 7 CONCLUSION

In this paper, we analyse MAUVE, a recently-proposed automatic metric for language generator evaluation. While MAUVE correlates quite well with human quality judgements, it is unclear which of the metric's design choices are in fact responsible for its success—a shortcoming that impedes the further development of language generator evaluation metrics. We attempt to rectify this shortcoming. Through a series of empirical studies, we identify MAUVE's substitution of probability distributions over embedding-based clusters—in place of the traditional distributions over strings—as the attribute largely responsible for the metric's success. In order to better understand the nature of this improvement, we probe the clusters utilised by the density estimators, analysing what they ignore and what they emphasise about the input text. We find that, while distributions over clusters are sensitive to syntactic- or coherence-level perturbations to the text, this is not the case for several surface-level perturbations. We thus conjecture that, by focusing on higher-level text features, cluster-based evaluation metrics may simply be better suited to rank high performing models, and that this is a general paradigm worth further exploration.

# REFERENCES

Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, and Yoav Goldberg. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=BJh6Ztuxl.  
Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016. URL https://arxiv.org/abs/1610.01644.  
Satanjeev Banerjee and Alon Lavie. METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, pp. 65-72, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. URL https://aclanthology.org/W05-0909.  
Steven Bird, Ewan Klein, and Edward Loper. Natural Language Processing with Python. O'Reilly Media, Inc., 1st edition, 2009. ISBN 0596516495. URL https://www.nltk.org/book/.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877-1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bcbd4967418bf8ac142f64a-Paper.pdf.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://www.aclweb.org/anthology/N19-1423.  
Josip Djolonga, Mario Lucic, Marco Cuturi, Olivier Bachem, Olivier Bousquet, and Sylvain Gelly. Precision-recall curves using information divergence frontiers. In Silvia Chiappa and Roberto Calandra (eds.), Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 2550-2559. PMLR, 26-28 Aug 2020. URL https://proceedings.mlr.press/v108/djolonga20a.html.  
Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9:160-175, 2021. doi: 10.1162/tacl_a_00359. URL https://aclanthology.org/2021.tacl-1.10.  
Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical Neural Story Generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 889-898, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1082. URL https://www.aclweb.org/anthology/P18-1082.  
Sebastian Gehrmann, Tosin P. Adewumi, Karmanya Aggarwal, Pawan Sasanka Ammanamanchi, Aremu Anuoluwapo, Antoine Bosselut, Khyathi Raghavi Chandu, Miruna Clinciu, Dipanjan Das, Kaustubh D. Dhole, Wanyu Du, Esin Durmus, Ondrej Dusek, Chris C. Emezue, Varun Gangal, Cristina Garbacea, Tatsunori B. Hashimoto, Yufang Hou, Yacine Jernite, Harsh Jhamtani, Yangfeng Ji, Shailza Jolly, Mihir Kale, Dhruv Kumar, Faisal Ladhak, Aman Madaan, Mounica Maddela, Khyati Mahajan, Saad Mahamood, Bodhisattwa Prasad Majumder, Pedro Henrique Martins, Angelina McMillan-Major, Simon Mille, Emiel van Miltenburg, Moin Nadeem, Shashi Narayan, Vitaly Nikolaev, Rubungo Andre Niyongabo, Salomey Osei, Ankur P. Parikh, Laura Perez-Beltrachini, Niranjan Rao, Vikas Raunak, Juan Diego Rodriguez, Sashank Santhanam, João Sedoc, Thibault Sellam, Samira Shaikh, Anastasia Shimorina, Marco Antonio Sobrevilla Cabezudo,

Hendrik Strobelt, Nishant Subramani, Wei Xu, Diyi Yang, Akhila Yerukola, and Jiawei Zhou. The GEM benchmark: Natural language generation, its evaluation and metrics. CoRR, abs/2102.01672, 2021. URL http://arxiv.org/abs/2102.01672.  
Kenneth Heafield. KenLM: Faster and smaller language model queries. In Proceedings of the Sixth Workshop on Statistical Machine Translation, pp. 187-197, Edinburgh, Scotland, July 2011. Association for Computational Linguistics. URL https://aclanthology.org/W11-2123.  
John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2733-2743, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1275. URL https://aclanthology.org/D19-1275.  
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. URL https://arxiv.org/abs/2203.15556.  
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rygGQyrFvH.  
F. Jelinek, R. L. Mercer, L. R. Bahl, and J. K. Baker. Perplexity—a measure of the difficulty of speech recognition tasks. The Journal of the Acoustical Society of America, 62(S1):S63-S63, 1977. doi: 10.1121/1.2016299. URL https://doi.org/10.1121/1.2016299.  
Tuomas Kynkänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/0234c510bc6d908b28c70ff313743079-Paper.pdf.  
Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74-81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013.  
Clara Meister and Ryan Cotterell. Language model evaluation beyond perplexity. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 5328-5339, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.414. URL https://aclanthology.org/2021.acl-long.414.  
Hermann Ney, Ute Essen, and Reinhard Kneser. On structuring probabilistic dependences in stochastic language modelling. Computer Speech & Language, 8(1):1-38, 1994. ISSN 0885-2308. doi: https://doi.org/10.1006/csla.1994.1001. URL https://www.sciencedirect.com/science/article/pii/S0885230884710011.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pp. 311-318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://aclanthology.org/P02-1040.  
Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. Mauve: Measuring the gap between neural text and human text using divergence frontiers. Advances in Neural Information Processing Systems, 34, 2021. URL https://proceedings.neurips.cc/paper/2021/bit/260c2432a0eacc28ce03c10dadc078a4-Abstract.html.

Tiago Pimentel, Naomi Saphra, Adina Williams, and Ryan Cotterell. Pareto probing: Trading off accuracy for complexity. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 3138-3153, Online, November 2020a. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.254. URL https://aclanthology.org/2020.emnlp-main.254.  
Tiago Pimentel, Josef Valvoda, Rowan Hall Maudslay, Ran Zmigrod, Adina Williams, and Ryan Cotterell. Information-theoretic probing for linguistic structure. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 4609-4622, Online, July 2020b. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.420. URL https://aclanthology.org/2020.acl-main.420.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://d4mucfpkseywv.cloudfront.net/better-language-models/language-models.pdf.  
Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu, Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michola Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d'Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021. URL https://arxiv.org/abs/2112.11446.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67, 2020. URL http://jmlr.org/papers/v21/20-074.html.  
Abhilasha Ravichander, Yonatan Belinkov, and Eduard Hovy. Probing the probing paradigm: Does probing accuracy entail task relevance? In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 3363-3377, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.295. URL https://aclanthology.org/2021.eacl-main.295.  
Ehud Reiter. A structured review of the validity of BLEU. Computational Linguistics, 44(3):393-401, September 2018. doi: 10.1162/coli_a_00322. URL https://aclanthology.org/J18-3002.  
Mehdi S. M. Sajjadi, Olivier Bachem, Mario Lucic, Olivier Bousquet, and Sylvain Gelly. Assessing generative models via precision and recall. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS'18, pp. 5234-5243, Red Hook, NY, USA, 2018. Curran Associates Inc. URL https://proceedings.neurips.cc/paper/2018/file/f7696a9b362ac5a51c3dc8f098b73923-Paper.pdf.  
Thibault Sellam, Dipanjan Das, and Ankur Parikh. BLEURT: Learning robust metrics for text generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7881-7892, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.704. URL https://aclanthology.org/2020.acl-main.704.  
Claude E. Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27(3):379-423, 1948. doi: 10.1002/j.1538-7305.1948.tb01338.x. URL https://doi.org/10.1002/j.1538-7305.1948.tb01338.x.

Miloš Stanojevic and Khalil Sima'an. BEER: BEtter evaluation as ranking. In Proceedings of the Ninth Workshop on Statistical Machine Translation, pp. 414-419, Baltimore, Maryland, USA, June 2014. Association for Computational Linguistics. doi: 10.3115/v1/W14-3354. URL https://aclanthology.org/W14-3354.  
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SJeYe0NtvH.  
Sam Wiseman, Stuart Shieber, and Alexander Rush. Challenges in data-to-document generation. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 2253-2263, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1239. URL https://aclanthology.org/D17-1239.  
Jiannan Xiang, Yahui Liu, Deng Cai, Huayang Li, Defu Lian, and Lemao Liu. Assessing dialogue systems with distribution distances. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 2192-2198, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.193. URL https://aclanthology.org/2021.findings-acl.193.  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. XLNet: Generalized autoregressive pretraining for language understanding. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf.  
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf.  
Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, and Steffen Eger. MoverScore: Text generation evaluating with contextualized embeddings and earth mover distance. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 563-578, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1053. URL https://aclanthology.org/D19-1053.
