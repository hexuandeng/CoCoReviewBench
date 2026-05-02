# A NON-MONOTONIC SELF-TERMINATING LANGUAGE MODEL

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent large-scale neural autoregressive sequence models have shown impressive performances on a variety of natural language generation tasks. However, their generated sequences often exhibit degenerate properties such as non-termination, undesirable repetition, and premature termination, when generated with decoding algorithms such as greedy search, beam search, top- $k$  sampling, and nucleus sampling. In this paper, we focus on the problem of non-terminating sequences resulting from an incomplete decoding algorithm. We first define an incomplete probable decoding algorithm which includes greedy search, top- $k$  sampling, and nucleus sampling, beyond the incomplete decoding algorithm originally put forward by Welleck et al. (2020). We then propose a non-monotonic self-terminating language model, which significantly relaxes the constraint of monotonically increasing termination probability in the originally proposed self-terminating language model by Welleck et al. (2020), to address the issue of non-terminating sequences when using incomplete probable decoding algorithms. We prove that our proposed model prevents non-terminating sequences when using not only incomplete probable decoding algorithms but also beam search. We empirically validate our model on sequence completion tasks with various architectures.

# 1 INTRODUCTION

Autoregressive neural sequence models (Bengio et al., 2000) have been widely used for various natural language generation tasks such as language modeling (Brown et al., 2020; Chowdhery et al., 2022), machine translation (Bahdanau et al., 2014), and conversational dialogue modeling (Vinyals & Le, 2015). Furthermore, large-scale autoregressive neural sequence models have shown unprecedented ability to generate fluent, human-like texts (Vaswani et al., 2017; Brown et al., 2020). Despite their success, the autoregressive neural sequence models have exhibited undesirable behaviors: non-termination (Welleck et al., 2020), degenerate repetition (Welleck et al., 2019; Holtzman et al., 2020), and premature termination (Koehn & Knowles, 2017; Stahlberg & Byrne, 2019). In this paper, we focus on how to prevent non-termination when using a given decoding algorithm.

Non-termination is the problem that we receive infinitely long sequences with a positive probability from our language model when using a given decoding algorithm. Welleck et al. (2020) pointed out that this issue comes from a discrepancy between the distribution of our language model and its induced distribution by an incomplete decoding algorithm. They formalized this disparity by the notion of inconsistency whether our language model generates non-terminating sequences with a positive probability from the decoding algorithm. To avoid this inconsistency, they proposed a self-terminating (ST) language model that uses new parametrization for its classifier rather than softmax parametrization. They proved that the ST language model is consistent with respect to greedy search, beam search, top- $k$  sampling (Fan et al., 2018), and nucleus sampling (Holtzman et al., 2020).

In the ST language model, termination probability of each sequence monotonically increases to 1 by using ST parametrization, but this parametrization is not appropriate for learning our language datasets. For instance, suppose there are two sequences in our dataset: "I am a boy." vs. "I am a boy. You are a girl." Our language model trained on this dataset may or may not terminate after the former. Once our model decides not to end, it should drop termination probability to continue the rest of the latter. Hence, the ST language model cannot optimally learn this dataset due to its monotonically increasing termination probability. We thus propose a non-monotonic self-terminating (NMST)

language model which guarantees the consistency with respect to greedy search, beam search, top-k sampling, and nucleus sampling without monotonically increasing termination probability.

Unlike the ST language model, our NMST language model does not assume monotonically increasing termination probability of each sequence. The NMST language model encourages termination probability of each sequence to converge to 1 through NMST parametrization. Converging to 1 is weaker than monotonically increasing to 1. Under this relaxation, we prove that the proposed NMST language model still prevents non-terminating sequences resulting from greedy search, beam search, top- $k$  sampling, and nucleus sampling. We further define a collection of decoding algorithms which NMST language models are consistent with respect to as incomplete probable decoding algorithms.

We conduct experiments validating the effectiveness of our NMST language models on sequence completion tasks. Since our approach only replaces softmax parametrization with NMST parametrization, we apply NMST parametrization to various architectures. Specifically, we train RNN (Elman, 1990) and LSTM (Hochreiter & Schmidhuber, 1997) on WikiText-2 (Merit et al., 2016). We additionally finetune GPT-2 (Radford et al., 2019) on WikiText-103 (Merit et al., 2016). For all setups, we observe that NMST parametrization effectively prevents non-terminating sequences compared to softmax parametrization. Furthermore, we see that our NMST parametrization has better validation perplexities than those of ST parametrization proposed by Welleck et al. (2020), even though both parametrizations solve the problem of non-terminating sequences.

# 2 NOTATIONS AND BACKGROUND

# 2.1 NOTATIONS FOR AUTOREGRESSIVE NEURAL SEQUENCE MODELS

Sequences, vocabulary, and  $\langle \mathrm{eos}\rangle$  We view an instance (e.g., a sentence and a paragraph) as a sequence  $\mathbf{y} = (y_{1},y_{2},\dots ,y_{T})$ , where each  $y_{t}$  is an element from a pre-defined finite set of discrete tokens, referred to as a vocabulary  $\nu$ .  $\nu$  includes a special symbol  $\langle \mathrm{eos}\rangle$  that only appears at the end of the sequence. Every sequence  $\mathbf{y}$  must end with  $\langle \mathrm{eos}\rangle$ . We write the length of  $\mathbf{y}$  as  $|\mathbf{y}|$  where  $y_{|\mathbf{y}|} = \langle \mathrm{eos}\rangle$ . We call  $\mathbf{y}$  a non-terminating sequence,  $|\mathbf{y}| = \infty$ , if  $y_{t}\neq \langle \mathrm{eos}\rangle$  for all  $t$ .

Embedding vectors Each token  $v \in \mathcal{V}$  is not a numerical vector so that we use an embedding vector  $\pmb{u}_v \in \mathbb{R}^m$  to represent  $v$ . To capture the notion of similarity between discrete tokens and their representational meaning efficiently, we use an embedding vector  $\pmb{u}_v \in \mathbb{R}^m$  to project  $v$  into continuous embedding space (Bengio et al., 2000; Mikolov et al., 2013b;a; Levy & Goldberg, 2014).

Autoregressive neural sequence models Bengio et al. (2000) proposed an autoregressive neural sequence model parametrized by  $\pmb{\theta} \in \mathbb{R}^k$ . They factorized  $p_{\pmb{\theta}}(\pmb{y}|\pmb{x})$  into a product of the conditional probability of each token given all the previous tokens and an input in a predefined order as follows:

$$
p _ {\boldsymbol {\theta}} (\boldsymbol {y} | \boldsymbol {x}) = \prod_ {t = 1} ^ {T} p _ {\boldsymbol {\theta}} \left(\boldsymbol {y} _ {t} \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right), \tag {1}
$$

where  $\pmb{y}_{< t}$  is a  $t$ -prefix of  $\pmb{y}$  and  $\pmb{x}$  is an input referred to as a context. For example,  $\pmb{x}$  represents either a prompt in sequence completion or a source-side sequence in machine translation.

There are several popular architectures for  $p_{\theta}$  such as RNN (Elman, 1990), LSTM (Hochreiter & Schmidhuber, 1997), GRU (Cho et al., 2014), and Transformer (Vaswani et al., 2017). As shown in equation 2, all these models utilize softmax classifiers. In this paper, we modify the parametrization of their softmax classifiers to prevent non-terminating sequences. We thus write a vanilla language model that uses the original softmax parametrization as  $p_{\theta}^{va}$  defined in Definition 1.

Definition 1. A vanilla language model  $p_{\pmb{\theta}}^{va}$  computes the conditional probability of each token  $v \in \mathcal{V}$  given a  $t$ -prefix  $\pmb{y}_{<t}$  and a context  $\pmb{x}$  at each time step  $t$  as follows:

$$
p _ {\boldsymbol {\theta}} ^ {v a} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) = \frac {\exp \left(\boldsymbol {u} _ {v} ^ {\top} \boldsymbol {h} _ {t}\right)}{\sum_ {v ^ {\prime} \in \mathcal {V}} \exp \left(\boldsymbol {u} _ {v ^ {\prime}} ^ {\top} \boldsymbol {h} _ {t}\right)}, \tag {2}
$$

where  $\pmb{h}_t = f_\theta(\pmb{y}_t, \pmb{h}_{t-1})$  with  $\pmb{h}_0 = \mathbf{0}$ .

Training For a given dataset,  $\mathcal{D} = \left\{\left(\boldsymbol{x}^{(n)},\boldsymbol{y}^{(n)}\right)\right\}_{n = 1}^{N}$ , we maximize the joint probability assigned to the sequences in our training dataset to find an optimal parameter configuration  $\theta^{\star}$  as follows:

$$
\boldsymbol {\theta} ^ {\star} = \arg \max  _ {\boldsymbol {\theta}} \sum_ {n = 1} ^ {N} \sum_ {t = 1} ^ {T ^ {(n)}} \log p _ {\boldsymbol {\theta}} \left(\boldsymbol {y} _ {t} ^ {(n)} \mid \boldsymbol {y} _ {<   t} ^ {(n)}, \boldsymbol {x} ^ {(n)}\right). \tag {3}
$$

# 2.2 INCOMPLETE PROBABLE DECODING ALGORITHMS

An autoregressive language model  $p_{\theta}$  predicts the likelihood of a sequence  $y$  given a context  $x$ . Its autoregressive factorization in equation 1 requires a recursive process for every  $t$  to infer. Hence, at inference time, we use a decoding algorithm defined in Definition 2 to generate sequences from  $p_{\theta}$ .

Definition 2. Let  $\mathcal{V}$  be a collection of  $\pmb{y}$  such that  $\pmb{y} = (y_{1}, y_{2}, \dots, y_{T})$  where  $T \in \{1, 2, \dots\}$  and  $y_{t} \in \mathcal{V}$ . A decoding algorithm  $\mathcal{S}$  is a function that maps  $p_{\theta}$  to  $q_{\mathcal{S}(p_{\theta})}$  which is a probability distribution over  $\mathcal{V}$ . A decoded sentence  $\hat{\pmb{y}}$  given  $\pmb{x}$  by  $\mathcal{S}$  from  $p_{\theta}$  is a random sample from  $q_{\mathcal{S}(p_{\theta})}(\pmb{y}|\pmb{x})$ .

To generate a high quality sequence from  $p_{\theta}$ , decoding algorithm assumes that a higher quality sequence has a higher probability of  $p_{\theta}$  than others. For instance, the maximum a posteriori (MAP) decoding algorithm  $S_{map}$  gives the most probable sequence  $\pmb{y}^{\star}$  given a context  $\pmb{x}$  from  $p_{\theta}$ :

$$
\boldsymbol {y} ^ {\star} = \arg \max  _ {\boldsymbol {y} \in \mathcal {Y}} p _ {\theta} (\boldsymbol {y} | \boldsymbol {x}), \tag {4}
$$

by setting  $q_{S_{map}(p_\theta)}(\boldsymbol{y} = \boldsymbol{y}^\star |\boldsymbol{x}) = 1$  and  $q_{S_{map}(p_\theta)}(\boldsymbol{y} = \boldsymbol{y}'|\boldsymbol{x}) = 0$  where  $\boldsymbol{y}' \in \mathcal{V} \setminus \{\boldsymbol{y}^\star\}$ . Unfortunately,  $S_{map}$  is intractable since equation 4 requires an exhaustive search over the sequence space  $\mathcal{V}$ . Hence, in practice, we utilize incomplete probable decoding algorithms defined in Definition 3.

Definition 3. A decoding algorithm  $\mathcal{S}$  is incomplete probable if there exists  $\emptyset \subsetneq \mathcal{V}_t \subsetneq \mathcal{V}$  such that

$$
\sum_ {v \in \mathcal {V} _ {t}} q _ {S (p _ {\boldsymbol {\theta}})} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) = 1 \tag {5}
$$

and

$$
\min  _ {v \in \mathcal {V} _ {t}} p _ {\boldsymbol {\theta}} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) \geq \max  _ {v \in \mathcal {V} \backslash \mathcal {V} _ {t}} p _ {\boldsymbol {\theta}} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) \tag {6}
$$

for each  $t$ . Furthermore, for every  $v \in \mathcal{V}_t$ ,  $\mathcal{S}$  satisfies

$$
q _ {\mathcal {S} \left(p _ {\boldsymbol {\theta}}\right)} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) \geq p _ {\boldsymbol {\theta}} \left(y _ {t} = v \mid \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right). \tag {7}
$$

At each  $t$ , an incomplete probable decoding algorithm  $\mathcal{S}$  considers only a set of highly probable tokens,  $\mathcal{V}_t$ .  $\mathcal{S}$  generates  $\hat{\pmb{y}}$  given  $\pmb{x}$  by recursively sampling  $\hat{y}_t$  from  $q_{\mathcal{S}(p_\theta)}(y_t|\hat{\pmb{y}}_{< t},\pmb{x})$  supported on  $\mathcal{V}_t$ . This reduces an exponential complexity of  $S_{map}$ ,  $\mathcal{O}\left(|\mathcal{V}||\hat{\pmb{y}}|\right)$ , down to a linear level,  $\mathcal{O}\left(|\hat{\pmb{y}}|\cdot |\mathcal{V}|\right)$ .

Greedy search, top- $k$  sampling (Fan et al., 2018), and nucleus sampling (Holtzman et al., 2020) are incomplete probable. For example, greedy search  $S_{qr}$  generates the  $t$ -th item of a sequence by

$$
\hat {y} _ {t} = \arg \max  _ {v \in \mathcal {V}} p _ {\boldsymbol {\theta}} \left(y _ {t} = v \mid \hat {\boldsymbol {y}} _ {<   t}, \boldsymbol {x}\right). \tag {8}
$$

In other words,  $S_{gr}$  sets  $\mathcal{V}_t$  to  $\left\{v_t^{(1)}\right\}$  where  $v_t^{(1)} = \arg \max_{v\in \mathcal{V}}p_\theta (y_t = v|\hat{\boldsymbol{y}}_{< t},\boldsymbol {x})$ . Moreover, we have  $p_{\pmb{\theta}}\big(y_t = v_t^{(1)}|\hat{\pmb{y}}_{< t},\pmb {x}\big)\leq q_{S_{gr}(p_\pmb{\theta})}\big(y_t = v_t^{(1)}|\hat{\pmb{y}}_{< t},\pmb {x}\big) = 1$ , and  $q_{S_{gr}(p_\pmb{\theta})}(y_t = v'| \hat{y}_{< t},\pmb {x}) = 0$  holds for  $v^{\prime}\in \mathcal{V}\setminus \mathcal{V}_t$ . Thus,  $S_{gr}$  is incomplete probable. Unlike  $S_{gr}$ , top- $k$  sampling considers  $k$  most probable tokens in  $\mathcal{V}$  as  $\mathcal{V}_t$  while nucleus sampling sets the smallest subset of  $\mathcal{V}$ , containing most probable tokens of which total probability is higher than a given threshold  $\mu$ , to  $\mathcal{V}_t$ . In §A.1 and A.2, we present that top- $k$  sampling and nucleus sampling are also incomplete probable.

Beam search is a heuristic algorithm that operates on the level of prefixes. We describe it further in §A.3. Although beam search is not incomplete probable, it also selects  $\mathcal{V}_t$  which is a proper subset of  $\mathcal{V}$  to expand each prefix at each step  $t$ . Due to this, we demonstrate that our main theoretical finding for the incomplete probable decoding algorithms in §3 is applicable to beam search as well.

# 2.3 CONSISTENCY WITH RESPECT TO INCOMPLETE PROBABLE DECODING ALGORITHMS AND SELF-TERMINATING (ST) LANGUAGE MODELS

Incomplete probable decoding algorithms greatly reduce computational overhead for generating sequences from our model. However, Welleck et al. (2020) observed that they can generate non-terminating sequences even if every training sequence has a finite length. To study this, Welleck et al. (2020) defined consistency with respect to decoding algorithms as shown in Definition 4.

Definition 4. A language model  $p_{\theta}$  is consistent with respect to a decoding algorithm  $S$  if  $q_{S(p_{\theta})}(|\pmb{y}| = \infty) = 0$  for any parameter configuration  $\pmb{\theta} \in \mathbb{R}^k$ .

Also, Welleck et al. (2020) proved that a vanilla language model  $p_{\theta}^{va}$  defined in Definition 1 is inconsistent with respect to incomplete probable decoding algorithms and beam search as follows:

Theorem 1. A vanilla language model  $p_{\theta}^{va}$  defined in Definition 1 is inconsistent with respect to any incomplete probable decoding algorithms and beam search (Theorem 3.4 in Welleck et al. (2020)).

The inconsistency of  $p_{\theta}^{va}$  with respect to an incomplete probable decoding algorithm  $S$  comes from equation 5. For each  $t$ ,  $S$  selects  $\mathcal{V}_t \subsetneq \mathcal{V}$  as a set of candidates for decoding, but  $p_{\theta}^{va}$  does not guarantee that  $\langle eos \rangle \in \mathcal{V}_t$ . Specifically, if  $\langle eos \rangle \notin \mathcal{V}_t$  for all  $t$ , then  $S$  cannot decode each token to  $\langle eos \rangle$  for all  $t$  (i.e., non-terminating). Based on this, Welleck et al. (2020) proposed a self-terminating (ST) language model defined in Definition 5 and proved that their model is consistent with respect to incomplete probable decoding algorithms and beam search as shown in Theorem 2.

Definition 5. For  $h_t$  defined in Definition 1, the conditional probability of each token  $v \in \mathcal{V}$  given a  $t$ -prefix  $y_{<t}$  and a context  $x$  at each time step  $t$  in an ST language model is given by

$$
\alpha_ {t} = p _ {\boldsymbol {\theta}} ^ {s t} \left(y _ {t} = \langle e o s \rangle | \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) = 1 - \prod_ {t ^ {\prime} = 1} ^ {t} (1 - \epsilon) \cdot \sigma \left(\boldsymbol {u} _ {\langle e o s \rangle} ^ {\top} \boldsymbol {h} _ {t ^ {\prime}}\right), \tag {9}
$$

and

$$
p _ {\boldsymbol {\theta}} ^ {s t} \left(y _ {t} = v | \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) = \left(1 - \alpha_ {t}\right) \cdot \frac {\exp \left(\boldsymbol {u} _ {v} ^ {\top} \boldsymbol {h} _ {t}\right)}{\sum_ {v ^ {\prime} \in V \setminus \{\langle e o s \rangle \}} \exp \left(\boldsymbol {u} _ {v ^ {\prime}} ^ {\top} \boldsymbol {h} _ {t}\right)},
$$

where  $v \in \mathcal{V} \setminus \{\langle eos \rangle\}$  and  $\sigma(x) = (1 + \exp(-x))^{-1}$  is a sigmoid function and  $\epsilon \in (0, 1)$ .

Theorem 2. An ST language model  $p_{\theta}^{st}$  defined in Definition 5 is consistent with respect to any incomplete probable decoding algorithms and beam search (Theorem 4.1-4.3 in Welleck et al. (2020)).

In equation 9,  $p_{\theta}^{st}(y_t = \langle eos \rangle | y_{<t}, x)$  monotonically increases to 1 as  $t$  increases.  $S$  chooses  $\mathcal{V}_t$  including  $\langle eos \rangle$  for  $t \geq t'$  with some  $t'$ , and  $\lim_{t \to \infty} q_{S(p_\theta)}(y_t = \langle eos \rangle | y_{<t}, x) = 1$  by equation 7. This enables  $p_{\theta}^{st}$  to terminate in finite steps when using  $S$ . Despite  $p_{\theta}^{st}$ 's consistency, its validation perplexity degrades compared to  $p_{\theta}^{va}$  in sequence completion (Welleck et al., 2020). We suspect that such degradation comes from the hypothesis of  $p_{\theta}^{st}$  that  $p_{\theta}(y_t = \langle eos \rangle | y_{<t}, x)$  monotonically increases to 1 as  $t$  increases. Remark 1 shows that  $p_{\theta^{\star}}(y_t = \langle eos \rangle | y_{<t}, x)$  is non-monotonic if there exist two examples which have the same prefix but different lengths for  $\theta^{\star}$  in equation 3.

Remark 1. Let  $\mathcal{D} = \left\{(\pmb{x}^{(1)},\pmb{y}^{(1)}),(\pmb{x}^{(2)},\pmb{y}^{(2)})\right\}$  be a two-instance training dataset. Assume that there exists  $t_0$  such that  $\pmb{y}_{< t_0} = \pmb{y}_{< t_0}^{(1)} = \pmb{y}_{< t_0}^{(2)}$ . Suppose further that  $t_0 = |\pmb{y}^{(1)}| < |\pmb{y}^{(2)}| - 1$  and  $\pmb{x} = \pmb{x}^{(1)} = \pmb{x}^{(2)}$ . If  $\pmb{\theta}^{\star}$  is an optimal parameter configuration in equation 3 over  $\mathcal{D}$ . Then,  $p_{\pmb{\theta}^{\star}}\left(y_t^{(2)} = \langle \cos \rangle |\pmb{y}_{< t}^{(2)},\pmb{x}\right)$  is non-monotonic with respect to  $t$  (proved in §B).

We can find such case satisfying the assumptions in Remark 1 by concatenating two sequences. We empirically demonstrate existence of non-monotonic  $p_{\pmb{\theta}^{\star}}(y_t = \langle eos\rangle |\pmb{y}_{< t},\pmb{x})$  in §4.2. Thus,  $p_{\pmb{\theta}}^{st}$  can fail to learn real datasets, because  $p_{\pmb{\theta}}^{st}(y_t = \langle eos\rangle |\pmb{y}_{< t},\pmb{x})$  monotonically increases to 1 as  $t\to \infty$ .

# 3 NON-MONOTONIC SELF-TERMINATING (NMST) LANGUAGE MODELS

The consistency of  $p_{\pmb{\theta}}^{st}$  comes from  $\lim_{t \to \infty} p_{\pmb{\theta}}^{st}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x}) = 1$ , not the monotonically increasing  $p_{\pmb{\theta}}^{st}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x})$  as a function of  $t$ . This motivates us to propose a non-monotonic self-terminating (NMST) language model  $p_{\pmb{\theta}}^{nmst}$  that permits  $p_{\pmb{\theta}}^{nmst}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x})$  to be a non-monotonic function of  $t$  with  $\lim_{t \to \infty} p_{\pmb{\theta}}^{nmst}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x}) = 1$  as follows:

Definition 6. For  $h_t$  defined in Definition 1, the conditional distribution of each token  $v \in \mathcal{V}$  given a  $t$ -prefix  $y_{<t}$  and a context  $x$  at the  $t$ -th step in an NMST language model is defined by

$$
\alpha_ {t} = p _ {\boldsymbol {\theta}} ^ {n m s t} \left(y _ {t} = \langle e o s \rangle | \boldsymbol {y} _ {<   t}, \boldsymbol {x}\right) = \left(1 - \sigma \left(\boldsymbol {u} _ {\langle e o s \rangle} ^ {\top} \boldsymbol {h} _ {t}\right)\right) \left(1 - (1 - \epsilon) ^ {t}\right) + \sigma \left(\boldsymbol {u} _ {\langle e o s \rangle} ^ {\top} \boldsymbol {h} _ {t}\right), \tag {10}
$$

and

$$
p _ {\pmb {\theta}} ^ {n m s t} (y _ {t} = v | \pmb {y} _ {<   t}, \pmb {x}) = (1 - \alpha_ {t}) \cdot \frac {\exp (\pmb {u} _ {v} ^ {\top} \pmb {h} _ {t})}{\sum_ {v ^ {\prime} \in \mathcal {V} \setminus \{\langle e o s \rangle \}} \exp (\pmb {u} _ {v ^ {\prime}} ^ {\top} \pmb {h} _ {t})},
$$

where  $v\in \mathcal{V}\setminus \{\langle eos\rangle \}$  and  $\sigma (x) = (1 + \exp (-x))^{-1}$  is a sigmoid function and  $\epsilon \in (0,1)$ .

Figure 1 shows that  $p_{\pmb{\theta}}^{nmst}$  uses convex combination of two curves for  $p_{\pmb{\theta}}^{nmst}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x})$ . We can write a curve  $g(t)$  between a lower bound curve  $f_{lb}(t)$  and an upper bound curve  $f_{ub}(t)$  as

$$
g (t) = (1 - \lambda (t)) f _ {l b} (t) + \lambda (t) f _ {u b} (t),
$$

![](images/eb71d870b5f0e5f08ee0033bef221d69a80491add95d4c3d77b50f6a48dbdf58.jpg)  
Figure 1: An illustration of NMST parametrization in equation 10 where  $f_{lb}(t) = 1 - (1 - \epsilon)^{t}$ ,  $f_{ub}(t) = 1$ ,  $\lambda(t') = \sigma(\boldsymbol{u}_{\langle eos\rangle}^{\top}\boldsymbol{h}_{t'})$ , and  $g(t) = p_{\theta}^{nmst}(y_t = \langle eos \rangle | \boldsymbol{y}_{<t}, \boldsymbol{x})$ . If  $g(t)$  lies between  $f_{lb}(t)$  and  $f_{ub}(t)$ , we can find  $\lambda(t')$  such that  $g(t') = (1 - \lambda(t'))f_{lb}(t') + \lambda(t')f_{ub}(t')$  for any  $t'$  regardless of whether  $g(t)$  is monotonic with respect to  $t$ . This allows  $p_{\theta}^{nmst}$  to learn a non-monotonic behavior of  $p_{\theta}^{nmst}(y_t = \langle eos \rangle | \boldsymbol{y}_{<t}, \boldsymbol{x})$ .  $p_{\theta}^{nmst}$  is consistent with respect to any incomplete probable decoding algorithms and beam search due to  $\lim_{t \to \infty} f_{lb}(t) = 1 \Rightarrow \lim_{t \to \infty} p_{\theta}^{nmst}(y_t = \langle eos \rangle | \boldsymbol{y}_{<t}, \boldsymbol{x}) = 1$ .

with appropriate  $\lambda(t) \in (0,1)$  for all  $t$ .  $p_{\theta}^{nmst}$  sets  $g(t)$  to  $p_{\theta}^{nmst}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$ , and then regards it as a convex combination of  $f_{lb}(t) = 1 - (1 - \epsilon)^t$  and  $f_{ub}(t) = 1$  with a coefficient  $\lambda(t) = \sigma(\mathbf{u}_{\langle eos\rangle}^{\top} \mathbf{h}_t)$ . This enables our model to predict the non-monotonic  $p_{\theta}^{nmst}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$ . Moreover, in Theorem 3, we prove that the proposed NMST parametrization in equation 10 guarantees the consistency with respect to any incomplete probable decoding algorithms and beam search since  $\lim_{t \to \infty} f_{lb}(t) = 1$  implies that  $\lim_{t \to \infty} p_{\theta}^{nmst}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x}) = 1$ .

Theorem 3. An NMST language model defined in Definition 6 is consistent with respect to any incomplete probable decoding algorithms and beam search (proved in §C).

Theorem 3 guarantees that every decoded sequence from  $p_{\theta}^{nmst}$  is terminated when using incomplete decoding algorithms and beam search. Both  $p_{\theta}^{nmst}$  and  $p_{\theta}^{st}$  do not yield non-terminating sequences resulting from incomplete probable decoding algorithms and beam search. However, unlike ST parametrization proposed by Welleck et al. (2020) in equation 9, our NMST parametrization in equation 10 can learn a wider range of  $p_{\theta}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$ , since  $p_{\theta}^{nmst}$  does not assume that  $p_{\theta}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$  is a monotonic function of  $t$ . We empirically demonstrate this by comparing  $p_{\theta}^{va}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$ ,  $p_{\theta}^{st}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$ , and  $p_{\theta}^{nmst}(y_t = \langle eos\rangle | \mathbf{y}_{<t}, \mathbf{x})$  in Figure 3.

# 4 EXPERIMENTS

We empirically validate the effectiveness of the proposed non-monotonic self-terminating language model by evaluating it in sequence completion tasks. We train three variants of a given architecture: (i) a vanilla  $(\mathrm{VA}+)$  language model using common softmax parametrization in equation 2, (ii) a self-terminating  $(\mathrm{ST}+)$  language model using ST parametrization proposed by Welleck et al. (2020) in equation 9, and (iii) our non-monotonic self-terminating  $(\mathrm{NMST}+)$  language model using NMST parametrization in equation 10. To compare them one another, we use following metrics:

- Perplexity: Given an autoregressive language model  $p_{\theta}$ , the perplexity of  $p_{\theta}$  over  $\mathcal{D}$  is  $\exp \left(-\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T^{(n)}}\log p_{\theta}\left(\boldsymbol{y}_{t}^{(n)}\Big|\boldsymbol{y}_{<t}^{(n)},\boldsymbol{x}^{(n)}\right)\right)$ , where  $\mathcal{D} = \left\{\left(\boldsymbol{x}^{(n)},\boldsymbol{y}^{(n)}\right)\right\}_{n=1}^{N}$ .  
- Non-termination ratio  $(r_{nt})$ : To present the consistency of  $p_{\theta}$  with respect to a given decoding algorithm  $\mathcal{S}$ , we need to compute  $r_{nt} = q_{\mathcal{S}(p_{\theta})}(|\boldsymbol{y}| = \infty)$ . Instead, based on

$$
r _ {n t} = q _ {\mathcal {S} \left(p _ {\theta}\right)} \left(\left| \boldsymbol {y} \right| = \infty\right) = \lim  _ {L \rightarrow \infty} q _ {\mathcal {S} \left(p _ {\theta}\right)} \left(\left| \boldsymbol {y} \right| > L\right), \tag {11}
$$

we use  $r_{nt}(L) = q_{S(p_{\theta})}(|\pmb{y}| > L)$  with a sufficiently large threshold  $L$  to estimate  $r_{nt}$ .

Sequence completion is a task of predicting a continuation  $\hat{\pmb{y}}$  given a  $c$ -length context  $\pmb{x} = (x_{1}, x_{2}, \dots, x_{c})$  by using a decoding algorithm  $\mathcal{S}$  from a language model  $p_{\theta}$  (i.e.  $\hat{\pmb{y}} \sim q_{\mathcal{S}(p_{\theta})}(\pmb{y}|\pmb{x})$ ). In this section, we use greedy search defined in equation 8 to generate  $\hat{\pmb{y}}$  given  $\pmb{x}$ . Our main theoretical finding in Theorem 3 is that the proposed NMST language model is consistent with respect to not only greedy search but also top- $k$  sampling, nucleus sampling, and beam search. We thus present results when using decoding algorithms other than greedy search in §5 and §F.

![](images/4777b126ad3c1e9083af3bc34910ddbb17163ac0fc484578026f444bb6ca659b.jpg)  
(a) RNN

![](images/90d3197b9c4b7bd606cef453ad56185a4c101ebcdfe8e49628ffafe3474c880b.jpg)  
Figure 2: Non-termination ratios,  $r_{nt}(L)$ 's, as a function of  $L$  in log-log scale for (a) RNN and (b) LSTM trained on WikiText-2 when using greedy search. We report mean (curve) ± stdev (shaded area) across 10 random experiments. For all configurations, both ST+ (non-red dashed) proposed by Welleck et al. (2020) and our NMST+ (non-red solid) are consistent with respect to greedy search since  $r_{nt}(L)$  goes to 0 as  $L$  increases. However, softmax parametrization (VA+, red dotted) is inconsistent with respect to greedy search since its  $r_{nt}(L)$  does not converge to 0 as  $L \to \infty$ .  
(b) LSTM

Table 1: Mean (±stdev) validation perplexities across 10 random runs on WikiText-2 for various model configurations. Lower is better. Bold marks the best of each architecture. For all  $\epsilon$ , the validation perplexities of our NMST+{RNN, LSTM} are better than those of ST+{RNN, LSTM} proposed by Welleck et al. (2020). Moreover, with a proper choice of  $\epsilon = 1.0 \times 10^{-5}$ , NMST+{RNN, LSTM} have competitive validation perplexities to those of VA+{RNN, LSTM}.  

<table><tr><td rowspan="2">ε</td><td colspan="2">RNN</td><td colspan="2">LSTM</td></tr><tr><td>ST+</td><td>NMST+</td><td>ST+</td><td>NMST+</td></tr><tr><td>5.0 × 10-4</td><td>186.1 ± (6.2)</td><td>184.2 ± (6.5)</td><td>106.1 ± (1.0)</td><td>105.6 ± (1.2)</td></tr><tr><td>1.0 × 10-4</td><td>181.0 ± (3.8)</td><td>177.4 ± (7.0)</td><td>104.6 ± (1.4)</td><td>102.5 ± (1.0)</td></tr><tr><td>5.0 × 10-5</td><td>182.6 ± (8.0)</td><td>179.6 ± (5.7)</td><td>104.7 ± (1.6)</td><td>102.1 ± (1.0)</td></tr><tr><td>1.0 × 10-5</td><td>180.4 ± (3.3)</td><td>177.4 ± (4.5)</td><td>104.5 ± (1.4)</td><td>101.5 ± (0.8)</td></tr><tr><td>VA+</td><td colspan="2">178.6 ± (6.3)</td><td colspan="2">101.6 ± (1.0)</td></tr></table>

# 4.1 WIKITEXT-2

WikiText-2 (Merit et al., 2016) is a language modeling dataset consisting of 2 million words from 600 Wikipedia articles. With word tokenization, we regard the first 10 tokens of each sequence and its remaining part, as a context  $x$  and a ground truth  $y$ , respectively. We train RNN activated by tanh (Elman, 1990) and LSTM (Hochreiter & Schmidhuber, 1997) on WikiText-2. Both RNN and LSTM have 2 layers, with 256 and 512 hidden units at each layer, respectively. We perform 10 random runs with a batch size of 32 for 70 epochs. We use AdamW (Loshchilov & Hutter, 2017) with an initial learning rate of 0.001,  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.99$ , weight decay of 0.01, learning rate decay, and early stopping. We further describe our models and training strategies for WikiText-2 experiments in §D. Unlike VA+{RNN, LSTM}, ST+{RNN, LSTM} and NMST+{RNN, LSTM} need an additional hyperparameter  $\epsilon$ . We explore  $\epsilon$  in  $\{1.0 \times 10^{-5}, 5.0 \times 10^{-5}, 1.0 \times 10^{-4}, 5.0 \times 10^{-4}\}$ .

We present the average  $(\pm \mathrm{stdev})$  non-termination ratios,  $r_{nt}(L)$ 's, across 10 random runs as a function of  $L$  for all considered setups of WikiText-2 in Figure 2 when decoding by greedy search. From equation 11, a language model is consistent with respect to greedy search if  $\lim_{L\to \infty}r_{nt}(L) = 0$ . As  $L$  increases, we observe that  $r_{nt}(L)$ 's of VA+{RNN, LSTM} do not converge to 0 while  $r_{nt}(L)$ 's of ST+{RNN, LSTM} and NMST+{RNN, LSTM} converge to 0. It means that RNN and LSTM become consistent with respect to greedy search by replacing its softmax parametrization with our NMST parametrization as well as ST parametrization proposed by Welleck et al. (2020).

Table 1 shows that the average (±stdev) validation perplexities across 10 random experiments for all variants of RNN and LSTM, trained on WikiText-2. We observe that NMST+{RNN, LSTM}

Table 2: We present the average  $(\pm$ stdev) validation perplexities across 10 random runs for all variants of GPT-2 finetuned on WikiText-103. Also, we demonstrate their non-termination ratios (mean±stdev),  $r_{nt}(L)$ 's, when using greedy search. We set  $L$  to 1,000 since the maximum length of generated sequences from GPT-2 is 1,024. For perplexity, lower is better. Bold marks the best validation perplexity in all setups. For every  $\epsilon$ , NMST+GPT-2 outperforms ST+GPT-2 in terms of the average validation perplexity. From  $r_{nt}(L)$ , NMST+GPT-2 effectively prevents non-termination sequences compared to VA+GPT-2 for every  $\epsilon$  while ST+GPT-2 with small  $\epsilon$  fails to avoid them. With a proper choice of  $\epsilon$  (e.g.,  $\epsilon = 1.0 \times 10^{-5}$ ), NMST+GPT-2 improves the validation perplexity.

<table><tr><td></td><td colspan="2">Perplexity</td><td colspan="2">r_nt(L)</td></tr><tr><td>ε</td><td>ST+</td><td>NMST+</td><td>ST+</td><td>NMST+</td></tr><tr><td>5.0 × 10-4</td><td>21.80 ± (0.02)</td><td>21.63 ± (0.02)</td><td>0.05 ± (0.03)</td><td>0.07 ± (0.03)</td></tr><tr><td>1.0 × 10-4</td><td>21.21 ± (0.02)</td><td>20.86 ± (0.02)</td><td>0.72 ± (0.11)</td><td>0.22 ± (0.10)</td></tr><tr><td>5.0 × 10-5</td><td>21.19 ± (0.03)</td><td>20.76 ± (0.02)</td><td>0.72 ± (0.11)</td><td>0.24 ± (0.10)</td></tr><tr><td>1.0 × 10-5</td><td>21.16 ± (0.03)</td><td>20.69 ± (0.03)</td><td>0.75 ± (0.10)</td><td>0.23 ± (0.10)</td></tr><tr><td>VA+</td><td colspan="2">20.72 ± (0.03)</td><td colspan="2">0.27 ± (0.08)</td></tr></table>

have better validation perplexities than  $\mathrm{ST} + \{\mathrm{RNN},\mathrm{LSTM}\}$  for every  $\epsilon$ . We demonstrate this clearer in Figure 4 by plotting the evolution of mean validation perplexities along  $\epsilon$ . Although our  $\mathrm{NMST}+$  guarantees the consistency of RNN and LSTM with respect to greedy search with a better validation perplexity than  $\mathrm{ST}+$ , we need to carefully select  $\epsilon$  of  $\mathrm{NMST}+$ . As  $\epsilon$  increases, the lower bound of  $p_{\theta}^{nmst}(y_t = \langle eos\rangle | y_{<t}, x)$  grows faster by equation 10. Hence,  $\mathrm{NMST}+$  yields premature sequences when  $\epsilon$  is too large. Indeed, the average validation perplexities of  $\mathrm{NMST} + \mathrm{RNN}$  and  $\mathrm{NMST} + \mathrm{LSTM}$  with  $\epsilon = 5.0 \times 10^{-4}$  are 184.2 and 105.6 which degrade by 5.6 and 4.0 from those of VA+RNN and VA+LSTM, 178.6 and 101.6, respectively. We however emphasize that there is an optimal  $\epsilon = 1.0 \times 10^{-5}$  that makes NMST+RNN and NMST+LSTM have the better validation perplexities than VA+RNN and VA+LSTM. In short, both NMST+ and ST+ prevent non-termination when using greedy search but only NMST+ has a competitive validation perplexity to VA+ with an optimal  $\epsilon$ .

# 4.2 WIkIText-103

WikiText-103 (Merit et al., 2016) consists of 103 million words constructed from 28,000 articles. We use BPE tokenization and consider the first 10 tokens as a context for each sequence. Since WikiText-103 is substantially larger dataset than WikiText-2, we use a pretrained GPT-2 (Radford et al., 2019) based on Transformer (Vaswani et al., 2017) with 124 million parameters. We finetune GPT-2 for 500,000 steps with bucketing batching technique. We use AdamW (Loshchilov & Hutter, 2017) with an initial learning rate of  $5.0 \times 10^{-5}$ ,  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.99$ , weight decay of 0.01, linear learning rate decay, and early stopping. We present more detailed description in §D. We use  $\epsilon$  of  $\{1.0 \times 10^{-5}, 5.0 \times 10^{-5}, 1.0 \times 10^{-4}, 5.0 \times 10^{-4}\}$  for ST+GPT-2 and NMST+GPT-2.

We report the mean  $(\pm \mathrm{stdev})$  validation perplexities and non-termination ratios,  $r_{nt}(L)$ 's, resulting from greedy search across 10 random runs for all GPT-2 configurations finetuned on WikiText-103 in Table 2. Since GPT-2 has a fixed-length of 1,024 tokens, we use  $L = 1,000$ . As shown in Figure 2, at least, we need a sufficiently large  $L$  such as  $L = 10^5$  to determine whether a language model is consistent with respect to greedy search. Although  $L = 1000$  is not sufficiently large, we observe that  $r_{nt}(L)$  of  $\mathrm{NMST + GPT - 2}$  decreases compared to  $r_{nt}(L)$  of  $\mathrm{VA + GPT - 2}$  as  $\epsilon$  increases. That is,  $\mathrm{NMST + }$  reduces the number of continuations from contexts, which are non-terminated within 1,000 steps. However, non-terminating sequences do not necessarily have better quality. We thus demonstrate example continuations from  $\mathrm{NMST + GPT - 2}$ , given a context that leads non-termination with  $\mathrm{VA + GPT - 2}$  in Table 3, when using greedy search. We observe that the quality of the generated sequence tends to improve with  $\mathrm{NMST + }$  by avoiding repetitions of similar phrases and ending with  $\langle eos\rangle$ . We present more example continuations in Table 5.

Similar to the results in §4.1, Table 2 shows that the validation perplexities of both ST+GPT-2 proposed by Welleck et al. (2020) and our NMST+GPT-2 degrade compared to VA+GPT-2 as  $\epsilon$  increases. However, NMST+GPT-2 with an optimal  $\epsilon = 1.0 \times 10^{-5}$  has a better validation perplexity

Table 3: Given a context in a validation instance of WikiText-103, we present example continuations of {VA, ST, NMST}+GPT-2 when using greedy search. We select  $\epsilon = 1.0 \times 10^{-5}$  for {ST, NMST}+GPT-2 because it is optimal in terms of validation perplexities in Table 2. Unlike {VA, ST}+GPT-2, NMST+GPT-2 improves the quality of the sequence by avoiding repetitive tokens and ending with  $\langle es\rangle$  when the given context leads VA+GPT-2 to non-terminate within 1,000 steps.  

<table><tr><td>Context</td><td>Made of concrete, steel, and wood, the</td></tr><tr><td>VA+</td><td>building was built in the mid @-@ 19th century. It was the first building in the United States to be built in concrete, and the first to be built in wood. It was also the first building in the United States to be built in steel. It was the first building in ...</td></tr><tr><td>ST+</td><td>building is constructed of steel and concrete. The building&#x27;s exterior is made of steel and concrete. The building&#x27;s interior is made of wood, and the building&#x27;s exterior is made of concrete. The building&#x27;s exterior is made of concrete, and the building&#x27;s ...</td></tr><tr><td>NMST+</td><td>building was designed by the architectural firm of Bowers &amp; Wainwright, and was completed in 1892. The building is the largest of its kind in the United States. &lt;eos&gt;</td></tr></table>

![](images/680400c27cdb1babc2c3430b182a7a59a7b35c17cea38cca3baab5d3191cb8c9.jpg)

![](images/57bb447120034672f2565bad2426cdd7feb22610f1af6e6bfd8552e3980587ff.jpg)  
Figure 3: We present  $p_{\theta}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x})$  as a function of  $t$  for validation instances of WikiText-103 where  $p_{\theta}$ 's are {VA, ST, NMST}+GPT-2. For {ST, NMST}+GPT-2, we choose  $\epsilon = 1.0 \times 10^{-5}$  because it is optimal in terms of validation perplexities in Table 2. Instead of  $t$ , we tag the  $t$ -th ground truth token. We report their mean (curve) ± stdev (shaded area) across 10 random runs. Unlike ST+GPT-2, NMST+GPT-2 can model non-monotonic behavior of  $p_{\theta}(y_t = \langle eos \rangle | \pmb{y}_{<t}, \pmb{x})$  with respect to  $t$ . Both plots show that the non-monotonic behaviors occur where the sequences could end (e.g., after red marked tokens such as periods).

of 20.69 than that of  $\mathrm{VA + GPT - 2}$ , 20.72. On the other side, we cannot find  $\epsilon$  such that the validation perplexity of  $\mathrm{ST + GPT - 2}$  is better than that of  $\mathrm{VA + GPT - 2}$ . Moreover, if  $\epsilon \neq 5.0 \times 10^{-4}$ , then  $r_{nt}(L)$ 's of  $\mathrm{ST + GPT - 2}$  blow up compared to  $r_{nt}(L)$  of  $\mathrm{VA + GPT - 2}$ . In Figure 5, we clearly demonstrate the inevitable perplexity degradation and exploding  $r_{nt}(L)$  of  $\mathrm{ST + GPT - 2}$  by presenting validation perplexities and  $r_{nt}(L)$ 's as a function of  $\epsilon$ . We suspect that such observations about  $\mathrm{ST + GPT - 2}$  come from monotonically increasing  $p_{\theta}(y_t = \langle \text{eos} \rangle | \mathbf{y}_{<t}, \mathbf{x})$  with respect to  $t$ .

We investigate behaviors of  $p_{\theta}(y_t = \langle eos \rangle | \mathbf{y}_{<t}, \mathbf{x})$  where  $p_{\theta}$ 's are {VA, ST, NMST}+GPT-2 in Figure 3. Based on Table 2, we select an optimal  $\epsilon = 1.0 \times 10^{-5}$  in terms of validation perplexities for {ST, NMST}+GPT-2. In Figure 3, {VA, NMST}+GPT-2 well-capture where a sequence might end (e.g., after periods) by showing non-monotonic behaviors at those steps, but ST+GPT-2 cannot model such non-monotonic behaviors because it assumes that  $p_{\theta}(y_t = \langle eos \rangle | \mathbf{y}_{<t}, \mathbf{x})$  is a monotonic function of  $t$ . This constraint makes ST+GPT-2 generate finite but unnecessarily long sequences with greedy search (i.e., higher  $r_{nt}(L)$  than VA+GPT-2 for small  $L$ , but  $r_{nt}(L) = 0$  for sufficiently large  $L$ ). However, NMST+GPT-2 does not assume monotonically increasing  $p_{\theta}(y_t = \langle eos \rangle | \mathbf{y}_{<t}, \mathbf{x})$ , so that NMST+GPT-2 better predicts  $p_{\theta}(y_t = \langle eos \rangle | \mathbf{y}_{<t}, \mathbf{x})$  than ST+GPT-2.

# 5 CONSISTENCY WITH RESPECT TO OTHER DECODING ALGORITHMS

We explore the effectiveness of our proposed non-monotonic self-terminating (NMST) language model when using decoding algorithms other than greedy search, such as top- $k$  sampling (Fan et al.,

Table 4: Mean (±stdev) non-termination ratios,  $r_{nt}(L)$ 's, across 10 random runs for the variants of GPT-2 finetuned on WikiText-103 with various decoding algorithms. We set  $L$  to 1,000 due to GPT-2's context window size of 1,024. We use an optimal  $\epsilon = 1.0 \times 10^{-5}$  in terms of average validation perplexities in Table 2 for both NMST+GPT-2 and ST+GPT-2. Bold marks the lowest  $r_{nt}(L)$  within each decoding algorithm (column). Similar to greedy search in Table 2, for all decoding algorithms,  $r_{nt}(L)$ 's of NMST+GPT-2 are lower than those of ST+GPT-2 and VA+GPT-2. It means that NMST+ reduce the number of non-terminating sequences within 1,000 decoding steps.

<table><tr><td></td><td>top-2</td><td>top-4</td><td>nucleus-0.2</td><td>nucleus-0.4</td><td>beam-2</td><td>beam-4</td></tr><tr><td>VA+</td><td>0.0 ± (0.0)</td><td>0.0 ± (0.0)</td><td>0.25 ± (0.08)</td><td>0.14 ± (0.05)</td><td>0.05 ± (0.02)</td><td>0.03 ± (0.01)</td></tr><tr><td>ST+</td><td>0.0 ± (0.0)</td><td>0.0 ± (0.0)</td><td>0.73 ± (0.11)</td><td>0.55 ± (0.15)</td><td>0.29 ± (0.10)</td><td>0.15 ± (0.07)</td></tr><tr><td>NMST+</td><td>0.0 ± (0.0)</td><td>0.0 ± (0.0)</td><td>0.21 ± (0.10)</td><td>0.10 ± (0.06)</td><td>0.03 ± (0.02)</td><td>0.01 ± (0.01)</td></tr></table>

2018), nucleus sampling (Holtzman et al., 2020), and beam search. All experimental setups and notations are the same as Section §4. As proved in Theorem 3, the NMST language model is consistent with respect to any incomplete decoding algorithms (e.g., greedy search, top- $k$  sampling, and nucleus sampling) and beam search for all  $\epsilon > 0$ . To validate this, we use top- $\{2, 4\}$  sampling, nucleus- $\{0.2, 0.4\}$  sampling, and beam search with a width of  $\{2, 4\}$  (beam- $\{2, 4\}$ ) to generate sequences from NMST+GPT-2 finetuned on WikiText-103 with  $\epsilon = 1.0 \times 10^{-5}$ . The choice of  $\epsilon = 1.0 \times 10^{-5}$  is made based on the validation perplexities in Table 2. Since the validation perplexity does not depend on decoding algorithms, we focus on the average ( $\pm$ stdev) non termination ratios,  $r_{nt}(L)$ 's, across 10 random runs with  $L = 1,000$  for each decoding algorithm in Table 4. We also present  $r_{nt}(L)$ 's of VA+GPT-2 and ST+GPT-2 with  $\epsilon = 1.0 \times 10^{-5}$  as baselines.

Table 4 shows that our NMST+GPT-2 has the lowest  $r_{nt}(L)$  with  $L = 1,000$  for all decoding algorithms compared to VA+GPT-2 and ST+GPT-2 proposed by (Welleck et al., 2020). In other words, NMST+ effectively prevent non-terminating sequences within 1,000 time steps regardless of decoding algorithms. Comparing with greedy search in Table 2 ( $r_{nt}(L)$  when  $\epsilon = 1.0 \times 10^{-5}$ ), we observe that  $r_{nt}(L)$ 's decrease for all setups. As we discussed in §2.3, non-terminating sequences originate from the choice of  $\langle \text{eos} \rangle \notin \mathcal{V}_t \subsetneq \mathcal{V}$  for all  $t$  where  $\mathcal{V}$  is a vocabulary and  $\mathcal{V}_t$  is the  $t$ -th proper subset of  $\mathcal{V}$ , considered by a decoding algorithm at the  $t$ -th step. In this sense, greedy search satisfies  $|\mathcal{V}_t| = 1$  for all  $t$ , and other decoding algorithms can satisfy  $|\mathcal{V}_t| > 1$  for some  $t$ . Hence, the decoding algorithms other than greedy search are likely to have  $\langle \text{eos} \rangle$  in  $\mathcal{V}_t$  and have the lower  $r_{nt}(L)$ . In the case of top-\{2, 4\} sampling, we obtain  $r_{nt}(L) = 0.0$  for VA+GPT-2. Without NMST+, VA+ can avoid non-terminating sequences if we choose a proper decoding algorithm. We however emphasize that NMST+GPT-2 with  $\epsilon = 1.0 \times 10^{-5}$  has the better average validation perplexity than VA+GPT-2 in Table 2. We also empirically demonstrate the consistency of NMST+{RNN, LSTM} trained on WikiText-2 with resepct to other decoding algorithms in Figure 6.

# 6 CONCLUSION

Non-termination is a degenerate behavior of a language model when using beam search and incomplete probable decoding algorithms such as greedy search, top- $k$  sampling (Fan et al., 2018), and nucleus sampling (Holtzman et al., 2020). To prevent this, Welleck et al. (2020) proposed a self-terminating language model that encourages termination probability of each sequence, which is the conditional probability of  $\langle \mathrm{eos} \rangle$  given a  $t$ -prefix and a context, to monotonically increase to 1 as  $t$  increases. In this paper, we theoretically demonstrate that monotonically increasing termination probability of each sequence is not a necessary condition for avoiding non-termination sequences. Moreover, we empirically present that termination probability of each sequence is not monotonic for a real dataset. We thus propose a non-monotonic self-terminating language model whose termination probability of each sequence converges to 1, not monotonically increases to 1. Our non-monotonic language models successfully address the issue of non-termination and have better validation perplexities than self-terminating language models proposed by Welleck et al. (2020) when not only training RNN (Elman, 1990), LSTM (Hochreiter & Schmidhuber, 1997) on WikiText-2 (Merit et al., 2016) but also finetuning GPT-2 (Radford et al., 2019) on WikiText-103 (Merit et al., 2016).

# REFERENCES

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language model. In J. Mach. Learn. Res., 2000.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
Kyunghyun Cho, Bart Van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259, 2014.  
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.  
Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179-211, 1990.  
Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 889-898, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1082. URL https://aclanthology.org/P18-1082.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Ari Holtzman, Jan Buys, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. ArXiv, abs/1904.09751, 2020.  
Philipp Koehn and Rebecca Knowles. Six challenges for neural machine translation. arXiv preprint arXiv:1706.03872, 2017.  
Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. Advances in neural information processing systems, 27, 2014.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.  
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013a.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013b.  
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929-1958, 2014.  
Felix Stahlberg and Bill Byrne. On nmt search errors and model errors: Cat got your tongue? arXiv preprint arXiv:1908.10090, 2019.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Oriol Vinyals and Quoc Le. A neural conversational model. arXiv preprint arXiv:1506.05869, 2015.  
Sean Welleck, Ilia Kulikov, Stephen Roller, Emily Dinan, Kyunghyun Cho, and Jason Weston. Neural text generation with unlikelihood training. arXiv preprint arXiv:1908.04319, 2019.  
Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, and Kyunghyun Cho. Consistency of a recurrent language model with respect to incomplete decoding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5553-5568, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.448. URL https://aclanthology.org/2020.emnlp-main.448.
