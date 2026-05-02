# ON THE APPROXIMATION PROPERTIES OF RECURRENT ENCODER-DECODER ARCHITECTURES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Encoder-decoder architectures have recently gained popularity in sequence to sequence modelling, featuring in state-of-the-art models such as transformers. However, a mathematical understanding of their working principles remains limited. In this paper, we study the approximation properties of recurrent encoder-decoder architectures. Prior work established theoretical results for classical RNNs in the linear setting, where approximation capabilities can be related to smoothness and memory. Here, we find that the encoder and decoder together form a particular "temporal product structure" which determines the approximation efficiency. Moreover, the encoder-decoder architecture generalises RNNs with the capability to learn time-inhomogeneous relationships. Our results provide the theoretical understanding of approximation properties of the recurrent encoder-decoder architecture, which characterises, in the considered setting, the types of temporal relationships that can be efficiently learned.

# 1 INTRODUCTION

Encoder-decoder is an increasingly popular architecture for sequence to sequence modelling problems (Sutskever et al., 2014; Chiu et al., 2018; Venugopalan et al., 2015). The core of this architecture is to first encode the input sequence into a vector using the encoder, and then map the vector into the output sequence through the decoder. In particular, such architecture forms the main component in the transformer network model (Vaswani et al., 2017), which has become a powerful method for modelling sequence to sequence relationships (Parmar et al., 2018; Beltagy et al., 2020; Li et al., 2019).

The encoder-decoder family of structures differ significantly from direct application of classical recurrent neural networks (Rumelhart et al., 1986) and their generalisations (Hochreiter & Schmidhuber, 1997; Cho et al., 2014) for processing sequences. However, both architectures can be considered as modelling mappings between sequences, albeit with different underlying structures. Hence, a natural but unresolved question is: how are these approaches fundamentally different? Answering this question is not only of theoretical importance, but is also of practical interest. Currently, architectural selection for different time-series modelling tasks is predominantly empirical. Thus, it is desirable to develop a concrete mathematical framework to understand the key differences between separate architectures in order to guide practitioners in a principled way.

In this paper, we investigate the approximation properties of encoder-decoder architectures. Where approximation is one of the most basic problems for supervised learning, it considers to what extent can the model fit to a target. In particular, we prove a general approximation result in the linear setting, that characterises the types of temporal input-output relationships can be efficiently approximated by encoder-decoder architectures.

These results reveal that such architectures essentially generalise the RNN by lifting the requirement of time-homogeneity in the target relationships. Hence, it can be used to tackle a broader class of sequence to sequence problems. Furthermore, of particular interest is the identification of a "temporal product structure" — a precise property of the target temporal relationship that enables further reduction in required trainable parameters for

approximation. This highlights another intrinsic difference between these architectures and classical recurrent neural networks.

Our main contributions can be summarised as follows

1. We prove a universal approximation result for encoder-decoder architectures in the linear setting, including approximation rates.  
2. We show that in the considered setting, encoder-decoder generalises RNN and can learn time-inhomogeneous relationships, which further adapt to additional product structures in the target relationship. This answers precisely how encoder-decoder are different from classical RNNs, at least in the considered setting.

Organisation. In section 2, we review the related work on encoder-decoder architectures and general approximation theory results of sequence modelling. Then, we formulate our approximation problem in section 3. Our main results, their consequences and numerical illustrations are presented in section 4. All the proofs are included in the appendix.

Notation. For consistency, we adhere to the following notation. Boldfaced letters are reserved for sequences or paths, which can be understood as functions of time. Lower case letters can mean vectors or scalars. Matrices are denoted by capital letters.

# 2 RELATED WORK

We will first review some previous works on sequence to sequence modelling. Encoder-decoder architecture first appeared in Kalchbrenner & Blunsom (2013), where they map the input sequence into a vector using convolutional neural networks, and then using a recurrent structure to map the vector to the output sequence. With the flexibility of manipulating the underlying structure of the encoder and decoder, numerous models based on this architecture have come out thereafter. For instance, Cho et al. (2014) use a gated RNN as encoder and decoder, while in their later work (Cho et al., 2015) they proposed a CNN based decoder. In Sutskever et al. (2014), they proposed a deep LSTM for both the encoder and decoder. Bahdanau et al. (2015) first introduces the attention mechanism, which was further developed in the well-known transformer networks (Vaswani et al., 2017). However, most of the researches on encoder-decoder architectures focus on applications. A theoretical understanding of the architecture is helpful for its further improvement and development.

From the theoretical point of view, Ye & Sung (2019) studied several theoretical properties of CNN encoder-decoders, including expressiveness, generalisation capability and optimisation landscape. Of particular relevance to the current work is expressiveness, which considers the relationships that can be generated from the model. However, this is not approximation properties. Yun et al. (2020) proved the universal approximation property of transformers for particular classes of functions, for example, permutation equivariant functions. But they did not consider the actual dynamical properties of the target relationship that affects approximation. Dynamical properties such as memory, smoothness and low rank structures are essential because they can precisely characterise different temporal relationships and affect the approximation capabilities of the models. Assuming the target is generated from a hidden dynamic system is one approach that is wildly applied (Maass et al., 2007; Schäfer & Zimmermann, 2007; Doya, 1993; Funahashi & Nakamura, 1993). In contrast, a functional analysis approach is introduced recently, where the targets are generated from functionals satisfying specific properties such as linearity, regularity, time-homogeneity (Li et al., 2021). In Li et al. (2021), the approximation properties of RNN models are studied, and the results show that the approximation efficiency is related to the memory structure. In Jiang et al. (2021), similar formulations are applied to study convolutional based architectures, where the results show that targets with certain spectrum regularity can be well approximated by dilated CNNs. We will follow the functional analysis formulation, where the results reveal that the encoder-decoders have a special temporal product structure which is intrinsically different from other architectures.

# 3 PROBLEM FORMULATION

In this section, mathematical formulations are given for the supervised learning problem of temporal modelling. That is, we precisely define the input space, output space, concept space and hypothesis space, respectively.

Functional formulation of temporal modelling. Firstly, we define the input and output space precisely.

A temporal sequence can be considered as a function of time  $t$ . The input space is defined by

$$
\mathcal {X} = C _ {0} \left(\left(- \infty , 0 \right], \mathbb {R} ^ {d}\right). \tag {1}
$$

That is, the space of continuous functions from  $(- \infty, 0]$  to  $\mathbb{R}^d$  vanishing at infinity, where  $d \in \mathbb{N}_+$  is the dimension. Denote the element in  $\mathcal{X}$  by  $\pmb{x} := \{x_t \in \mathbb{R}^d : t \in (-\infty, 0]\}$ , we equip  $\mathcal{X}$  with the supremum norm  $\| \pmb{x} \|_{\mathcal{X}} := \sup_{t \leq 0} \| x_t \|_{\infty}$ .

For the outputs, we take the space of bounded continuous functions from  $[0,\infty)$  to  $\mathbb{R}$ :

$$
\mathcal {Y} = C _ {b} ([ 0, \infty), \mathbb {R}). \tag {2}
$$

We consider real value outputs since we can handle each dimension individually for vector-valued outputs.

The mapping between the two sequences can be formulated as a sequence of functionals where

$$
y _ {t} = H _ {t} (\boldsymbol {x}), \quad t \geq 0. \tag {3}
$$

The output  $y_{t}$  at time step  $t$  depends on the input sequence  $\mathbf{x}$ . The ground truth relation between input and output is formulated by the sequence of functionals  $H = \{H_{t}\}_{t\geq 0}$ .

We provide an example to illustrate the above formulation. Given an input  $\pmb{x}$ , the output  $\pmb{y}$  is the smoothed version of  $\pmb{x}$ , resulting from convolve  $\pmb{x}$  with the Gaussian kernel  $g(s) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{s^2}{2})$ . This relation is mathematically formulated as

$$
y _ {t} = H _ {t} (\boldsymbol {x}) = \int_ {0} ^ {\infty} g (t + s) \boldsymbol {x} _ {- s} d s. \tag {4}
$$

For a supervised learning problem, our goal is to use a model to learn the target relationship  $\pmb{H}$ . Next we will define the model.

The RNN encoder-decoder model. Among all different variants of the encoder-decoder architectures, the RNN encoder-decoder introduced in Cho et al. (2014) can be considered as the most simple and representative model, where the encoder and decoder are both traditional RNNs. We will study this particular model as we try to eliminate other factors and only focus on the encoder-decoder architecture itself.

Under our setting, the simplified model of Cho et al. (2014) with classical RNN as encoder and decoder is formulated as

$$
h _ {t} = \sigma_ {E} \left(W _ {E} h _ {t - 1} + U _ {E} x _ {t} + b _ {E}\right), \quad v = h _ {\tau},
$$

$$
g _ {t} = \sigma_ {D} \left(W _ {D} h _ {t - 1} + U _ {D} x _ {t} + b _ {D}\right), \quad g _ {0} = v, \tag {5}
$$

$$
o _ {t} = \sigma_ {O} \left(W _ {O} g _ {t} + b _ {O}\right),
$$

where  $h_t$  and  $g_t$  are hidden states of the encoder and decoder, respectively,  $c$  is the coding vector and  $o_t \in \mathbb{R}$  denotes the model prediction. First, the encoder reads the entire input  $\pmb{x}$ , and it encodes (summarises) the input sequence into a fixed size coding vector  $v$ , which is the last hidden state of the encoder. Next, the coding vector is passed into the decoder as its initial state, and the decoder will produce an output at each output time step.

Note that the encoder has a terminating time step, and the decoder has a starting time step. This is the reason why we take the input and output space to be semi-infinite sequences. We set 0 to be both the terminating and starting time step.

We will take a linear and continuous-time idealisation for our investigation, which is formulated as

$$
\frac {d}{d s} h _ {s} = W h _ {s} + U x _ {s}, \quad v = Q h _ {0}, \quad s \leq 0
$$

$$
\frac {d}{d t} g _ {t} = V g _ {t}, \quad \quad \quad g _ {0} = P v, \tag {6}
$$

$$
o _ {t} = c ^ {\top} g _ {t}, \quad t \geq 0,
$$

where  $W \in \mathbb{R}^{m_E \times m_E}$ ,  $U \in \mathbb{R}^{m_E \times d}$ ,  $Q \in \mathbb{R}^{N \times m_E}$ ,  $V \in \mathbb{R}^{m_D \times m_D}$ ,  $P \in \mathbb{R}^{m_D \times N}$  and  $c \in \mathbb{R}^{m_D}$ . Here  $m_E$ ,  $m_D$  denotes the width of the encoder RNN and decoder RNN, respectively.  $N$  is the dimension of the coding vector  $v$ .

Since our goal is to investigate approximation problems over large time horizons, we are supposed to consider the stable encoder-decoder RNNs, where

$$
W \in \mathcal {W} _ {m _ {E}} := \{W \in \mathbb {R} ^ {m _ {E} \times m _ {E}}: \text {e i g e n v a l u e s o f} W \text {h a v e n e g a t i v e r e a l p a r t s} \}, \tag {7}
$$

$$
V \in \mathcal {V} _ {m _ {D}} := \{V \in \mathbb {R} ^ {m _ {D} \times m _ {D}}: \text {e i g e n v a l u e s o f V h a v e n e g a t i v e r e a l p a r t s} \}. \tag {8}
$$

The hypothesis space of RNN encoder-decoder models with arbitrary width and coding size is defined as

$$
\widehat {\mathcal {H}} := \bigcup_ {m _ {E}, m _ {D}, N \in \mathbb {N} _ {+}} \widehat {\mathcal {H}} _ {m _ {E}, m _ {D}, N},
$$

$$
\widehat {\mathcal {H}} _ {m _ {E}, m _ {D}, N} := \left\{\widehat {\boldsymbol {H}}: \widehat {H} _ {t} (\boldsymbol {x}) = c ^ {\top} e ^ {V t} P \int_ {0} ^ {\infty} Q e ^ {W s} U x _ {- s} d s, \text {w i t h} \right. \tag {9}
$$

$$
\left. \left(W, U, Q, V, P, c\right) \in \mathcal {W} _ {m _ {E}} \times \mathbb {R} ^ {m _ {E} \times d} \times \mathbb {R} ^ {N \times m _ {E}} \times \mathcal {V} _ {m _ {D}} \times \mathbb {R} ^ {m _ {D} \times N} \times \mathbb {R} ^ {m _ {D}} \right\}.
$$

Here, the model widths  $m_{E}, m_{D}$  and the coding vector dimension  $N$  together control the capacity/complexity of the hypothesis space.

Due to the mathematical form (9), not all functionals can be represented by the RNN encoder-decoder. To achieve a good approximation, the target functionals must possess certain structures. We introduce the following definitions to clarify these structures of functionals.

Definition 3.1. Let  $\pmb{H} = \{H_{t}\}_{t\geq 0}$  be a sequence of functionals.

1. For any  $t \in \mathbb{R}$ , the functional  $H_{t}$  is linear and continuous if for any  $\lambda_1, \lambda_2 \in \mathbb{R}$  and  $\pmb{x}_1, \pmb{x}_2 \in \mathcal{X}$ , we have  $H_{t}(\lambda_{1}\pmb{x}_{1} + \lambda_{2}\pmb{x}_{2}) = \lambda_{1}H_{t}(\pmb{x}_{1}) + \lambda_{2}H_{t}(\pmb{x}_{2})$ , and  $\| H_t\| := \sup_{\pmb{x} \in \mathcal{X}, \| \pmb{x}\|_{\mathcal{X}} \leq 1} |H_t(\pmb{x})| < \infty$ , where  $\| H_t\|$  denotes the induced functional norm.  
2. For any  $t \in \mathbb{R}$ , the functional  $H_{t}$  is regular if for any sequence  $\{\pmb{x}^{(n)}\}_{n=1}^{\infty} \subset \mathcal{X}$  such that  $\lim_{n \to \infty} x_{s}^{(n)} = 0$  for almost every  $s \leq 0$  (Lebesgue measure), we have  $\lim_{n \to \infty} H_{t}(\pmb{x}^{(n)}) = 0$ .

For a sequence of functionals  $\pmb{H}$ , we define its norm by  $\| \pmb{H} \| \coloneqq \int_0^\infty \| H_t \| dt$ .

Remark 3.1. The definitions of linear and continuous functionals are standard. One can view regular functionals as those not determined by inputs on arbitrarily small time intervals, e.g. an infinitely thin spike (i.e.  $\delta$ -functions).

Given the above definitions, we immediately have the following observation. The proof is found in Appendix A.

Proposition 3.1. Let  $\widehat{\pmb{H}}\in \widehat{\mathcal{H}}$  be a sequence of functionals in the RNN encoder-decoder hypothesis space (see (9)). Then for any  $t\geq 0$ ,  $\widehat{H}_t\in \widehat{H}$  is a linear, continuous and regular functional. Furthermore,  $\| \widehat{H}_t\|$  decays exponentially as a function of  $t$ .

This proposition characterises properties the encoder-decoder hypothesis space possesses. In particular, it is different from the RNN hypothesis space discussed in Li et al. (2021) since the

encoder-decoder is not necessarily time-homogeneous. Where a sequence of functionals  $\widehat{H}$  is time-homogeneous if for any  $t, \tau \geq 0$ ,  $H_{t}(\pmb{x}) = H_{t + \tau}(\pmb{x}(\tau))$ , with  $x(\tau)_{s} = x_{s - \tau}$  for all  $s \in \mathbb{R}$ . Our primary concern is how RNN encoder-decoder approximate general target relationships without assuming time-homogeneity. We will talk about our main approximation results in this direction.

Relation with RNNs. Here, we emphasise the differences between encoder-decoder hypothesis space and the RNN hypothesis space discussed in Li et al. (2021), where  $\widehat{H}_t^{(\mathrm{RNN})}(\pmb{x}) = \int_0^\infty c^\top e^{W(t + s)}Ux_{-s}ds$ . A key difference is that the encoder-decoder has a structure involving two temporal parameters  $t$  and  $s$ , while the RNN has only one depending on  $t + s$ , due to time-homogeneity. Owing to this difference and the fact that  $\widehat{\pmb{H}}^{(\mathrm{RNN})}\in \widehat{\mathcal{H}}$ , the encoder-decoder hypothesis space (9) contains the RNN hypothesis space and is more general, with the extra capabilities to learn time-inhomogeneous relationships. Furthermore,  $e^{Vt}$ ,  $e^{Ws}$  adapts to a product structure, which is an intrinsic difference between encoder-decoder and other architectures. We will discuss this in detail when we present our main results.

# 4 APPROXIMATION RESULTS

One of the most fundamental problems for supervised learning is the approximation problem. That is, given the hypothesis space (model) and concept space (target), to what extent can the model fit the target? Or what is the capacity/complexity of the hypothesis space? In this section, we present our main approximation results for encoder-decoder. Particular attention is paid to discuss how does the encoder-encoder differ from classical RNNs in terms of approximation.

In general, there are two levels of approximation problems that can be discussed. The first is known as the universal approximation, or density results, which focuses on the existence of models to achieve (any) given approximation accuracy to some targets satisfying appropriate conditions. The second is the approximation rate, which aims to quantitatively characterise the approximation accuracy concerning the capacity/complexity of the hypothesis space (e.g. the number of trainable parameters). In this section, both of them are developed for RNN encoder-decoder. In addition, the estimate of approximation rates can be refined given some common structures of targets.

# 4.1 UNIVERSAL APPROXIMATION

We first present the most basic density result, which says that any linear, continuous, and regular temporal relationship can be approximated by RNN encoder-decoder up to arbitrary accuracy. The proof is found in Appendix B.

Theorem 4.1. Let  $\pmb{H}$  be a sequence of linear, continuous, and regular functionals defined on  $\mathcal{X}$ , and satisfy  $\| \pmb{H} \| < \infty$ . Then for any  $\epsilon > 0$ , there exists  $\widehat{\pmb{H}} \in \widehat{\mathcal{H}}$  such that

$$
\| \boldsymbol {H} - \widehat {\boldsymbol {H}} \| \equiv \int_ {0} ^ {\infty} \| H _ {t} - \widehat {H} _ {t} \| d t <   \epsilon . \tag {10}
$$

Here, we highlight two important observations while deriving Theorem 4.1. First, one can show that each sequence of functionals  $\pmb{H} \in \mathcal{H}$  can be associated with a unique two-parameter "representation"  $\rho(t,s)$ , such that  $H_{t}(\pmb{x}) = \int_{0}^{\infty} x_{-s}^{\top} \rho(t,s) ds$ . Recall the model in the hypothesis space has the form  $\widehat{H}_{t}(\pmb{x}) = \int_{0}^{\infty} x_{-s}^{\top} \hat{\rho}(t,s) ds$ , where  $\hat{\rho}(t,s) := [c^{\top} e^{Vt} P Q e^{Ws} U]^{\top}$  denotes the representation for the model. The functional approximation problems reduce to a function approximation problems on this representation, in the sense that  $\| \pmb{H} - \widehat{\pmb{H}} \| \leq \| \rho - \hat{\rho} \|$ . It turns out that the structure of  $\rho$  directly affects the rate of approximation and can give rise to intrinsic properties that further lead to parameter reduction, as we will discuss in section4.3.

Here, we again emphasise the difference between our work with Li et al. (2021). In their work the target relationships are assumed to be time-homogeneous and have the representation  $H_{t}(\pmb{x}) = \int_{0}^{\infty}\rho (t + s)x_{-s}ds$ , which only depends on  $t + s$ . However, our setting does not assume time-homogeneity, thus have a more general representation where  $\rho$  depends on the two temporal directions  $t$  and  $s$  simultaneously.

# 4.2 GENERAL APPROXIMATION RATES

While the density result (Theorem 4.1) ensures the universal approximation property of the RNN encoder-decoder, it does not identify targets that can be efficiently approximated. To achieve this, we focus on approximation rates next. We characterise the temporal structure of a target relationship by looking at its responses of "constant" input signals.

Here we consider the approximation rates for a model with "large size" coding vector, where the dimension of the  $N \geq \bar{m} \coloneqq \min\{m_E, m_D\}$ . This is the scenario where we fix the width but want to take an oversized coding vector. The proof is found in Appendix C.

Theorem 4.2. Let  $\pmb{H}$  be a sequence of linear, continuous, and regular functionals defined on  $\mathcal{X}$ , and satisfy  $\| H \| < \infty$ . Consider the output of piece-wise constant signals  $y_{i}^{c}(t,s) = H_{t}(e_{i}\mathbf{1}_{(-\infty, -s]})$ ,  $t,s \geq 0$ ,  $i = 1,2\dots,d$ , where  $\{e_i\}_{i=1}^d$  denotes the standard basis of  $\mathbb{R}^d$ . Assume that there exist  $\alpha \in \mathbb{N}_+$ ,  $\beta > 0$ , such that for any  $i = 1,2\dots,d$ ,

$$
y _ {i} ^ {\mathrm {c}} \in C ^ {(\alpha + 1)} ([ 0, \infty) ^ {2}), \tag {11}
$$

$$
e ^ {\beta (t + s)} \frac {\partial^ {k + l}}{\partial t ^ {k} \partial s ^ {l}} y _ {i} ^ {c} (t, s) = o (1) a s \| (t, s) \| \rightarrow \infty , \quad (k, l) \in \mathbb {N} \times \mathbb {N} _ {+}, k + l \leq \alpha + 1. \tag {12}
$$

Then, for any  $m_E, m_D, N \in \mathbb{N}_+$  with  $N \geq \bar{m}$ , there exists  $\widehat{\pmb{H}} \in \widehat{\mathcal{H}}_{m_E, m_D, N}$  such that

$$
\left\| \boldsymbol {H} - \widehat {\boldsymbol {H}} \right\| \leq \frac {C (\alpha) \gamma d}{\beta^ {2}} \left(\frac {1}{m _ {E} ^ {\alpha}} + \frac {1}{m _ {D} ^ {\alpha}}\right), \tag {13}
$$

where  $C(\alpha),\gamma >0$  are both universal constants with dependence only on  $\alpha$  and  $(\alpha ,\beta)$  respectively, and  $\gamma \coloneqq \max_{i\in \mathbb{N}_{+},i\leq d}\max_{k,l\in \mathbb{N},k + l\leq \alpha +1}\sup_{t,s\geq 0}\beta^{-(k + l)}e^{\beta (t + s)}\left|\frac{\partial^{k + l}}{\partial t^{k}\partial s^{l}} y_{i}^{c}(t,s)\right| <   \infty .$  Here, the number of trainable parameters is  $dN(m_E + m_D)$  , with  $N\geq \bar{m} \coloneqq \min \{m_E,m_D\}$

First, note that the error bound does not depend on the coding vector size  $N$ , as long as  $N \geq \bar{m}$ . This is because further increasing  $N$  beyond  $\bar{m}$  only increases number of trainable parameters but does not increase its model capacity (See Remark C.1). Only the width  $m_E, m_D$  affect the approximation capabilities.

Next, we focus on which class of target relationships can be well approximated. Here,  $\alpha$  characterises the smoothness of  $\pmb{H}$ , and  $\beta$  characterises the temporal decay rates of the output of a constant signal under  $\pmb{H}$ . This is a notion of memory in the target relationship. By observing the error bound (13), we see that a target functional can be efficiently approximated by the encoder-decoder if it is smooth (large  $\alpha$ ), and has fast decaying memory (large  $\beta$ ).

The smoothness and decay rate characterisation also appear in the RNN results (Li et al., 2021), where the upper bound is  $\frac{C(\alpha)\gamma d}{\beta m^{\alpha}}$ . However, our results for encoder-decoder show extra structures, where the bounds involve two (instead of one) temporal parameters together with smoothness and decay requirements in each. The two-parameter time dependence allows the encoder-decoder to approximate time-inhomogeneous relationships, thus generalising the RNN. This two-parameter structure further leads to adaptation to a specific low-rank type of target relationships, resulting in finer approximation rates as we discuss next.

# 4.3 APPROXIMATION RATES WITH TEMPORAL PRODUCT STRUCTURE

Motivation of the temporal product structure. In contrast with Theorem 4.2, we next consider the models with  $N < \bar{m} \coloneqq \min \{m_E, m_D\}$ . In this situation, the model has

fewer parameters, we characterise the target relationships by further exploiting the structure of the two-parameters representation  $\rho(t,s)$ , which leads to a finer approximation rate by considering  $m_E, m_D, N$  together.

We first motivate how the "temporal product structure" arises, and how it relates to the approximation. Detailed discussions are found in the Appendix D. For illustration purposes, we take input dimension  $d = 1$ . Recall  $Q \in \mathbb{R}^{N \times m_E}$ ,  $P \in \mathbb{R}^{m_D \times N}$ , and thus the representation  $\hat{\rho}$  of the encoder-decoder functional can be written as

$$
\begin{array}{l} \hat {\rho} (t, s) = c ^ {\top} e ^ {V t} P \cdot Q e ^ {W s} u = \sum_ {n = 1} ^ {N} \left(\sum_ {i, j = 1} ^ {m _ {D}} c _ {i} P _ {j n} [ e ^ {V t} ] _ {i j}\right) \left(\sum_ {i, j = 1} ^ {m _ {D}} u _ {i} Q _ {j n} [ e ^ {W s} ] _ {i j}\right) \\ = \sum_ {n = 1} ^ {N} \hat {\varphi} _ {n} (t) \hat {\phi} _ {n} (s). \tag {14} \\ \end{array}
$$

This is a tensor product structure over the  $t, s$  time domain (determined by the encoder  $\{\hat{\phi}_n\}$  and decoder  $\{\hat{\varphi}_n\}$  successively). We call this the temporal product structure, and we will show that this structure significantly affects approximation rates. When  $\{\hat{\phi}_n\}$  and  $\{\hat{\varphi}_n\}$  are taken to be the "bases" along  $s, t$  direction, respectively,  $N$  is considered as the rank of the temporal product. We also define  $N$  as the rank of the model, which is understood as the maximum rank of temporal products the model can represent.

The rank concept of temporal relationships. Recall that the number of trainable parameters is  $dN(m_E + m_D)$ . Hence, a low rank model can achieve fewer trainable parameters. While considering what relationships can be well approximated by a low rank model, a natural conjecture would be "low rank" targets.

However, what does it mean for a temporal relationship to be "low rank"? We know that in linear algebra, a low rank operator means its range space is low dimensional. This idea also applies to temporal relationships. For a 'low rank' temporal relationship, the output sequence will be more regular, which means the output sequences (understood as functions) are in a low dimensional function space. We provide an intuitive numerical illustration for better understanding.

![](images/58214954a33ce437d564a2b0080543cb8d7df4cad524f7573e591835971bef75.jpg)  
(a) high rank relationship

![](images/4592f3d593376611381877e398e05e969f901a63545eb98b0465b6bcb32d3b45.jpg)  
Figure 1: We construct a high rank target and a low rank target from the temporal product. For both (1a) and (1b), we plot the input  $x_{t}$  with its outputs  $H_{t}(\pmb{x})$ . Detailed settings are found in Appendix E.1.  
(b) low rank relationship

Figure 1 shows the outputs of a high rank (left) and a low rank (right) target relationship on the same set of random input sequences. Different colours refer to different instances of the inputs. In the first case (high rank), the temporal structure of the outputs is very complex and depends sensitively on the inputs. In the second case (low rank), the output sequence are much more regular, and only macroscopic structures (such as the scale or offset) depend on the input sequence.

Remark 4.1. In the study of approximation theories for temporal sequences, prior work also related a notion of rank to the approximation property under the dilated convolutional structure (Jiang et al., 2021). We emphasise here that the notion of rank considered in the case of encoder-decoder is very different from that in Jiang et al. (2021), which mainly concerns

the tensorisation of a one-parameter time sequence according to the width of convolution filters.

POD as an analogy of SVD Now, we need a mathematical way to characterise low rank and high rank temporal relationships. We will introduce the concepts informally, and rigorous definitions can be found in the Appendix D. For a matrix, we can assess its rank by looking at its singular value decomposition (SVD). This can be extended to the temporal relationships using proper orthogonal decomposition (POD). The idea is that we can decompose the function  $\rho$  into the following form

$$
\rho (t, s) = \sum_ {n = 1} ^ {N _ {0}} \sigma_ {n} \varphi_ {n} (t) \phi_ {n} (s), \tag {15}
$$

where  $N_0 \leq \infty$ ,  $\{\varphi_n\}$  and  $\{\phi_n\}$  are orthonormal bases, and  $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$  are the singular values. This procedure can be considered as applying SVD to an infinite dimensional space (when  $N_0 = \infty$ ). An analogy of Eckart-Young theorem (Eckart & Young, 1936), which characterises best low-rank approximation errors, also exists for POD. It roughly states that

$$
\inf  _ {\operatorname {r a n k} (\hat {\rho}) = N} \| \rho - \hat {\rho} \| _ {L ^ {2}} ^ {2} = \sum_ {n = N + 1} ^ {\infty} \sigma_ {n} ^ {2}. \tag {16}
$$

That is, any target  $\rho$  has a rank  $N$  best approximation, and its error equals to the tail sum of its squared singular values. In other words, a target with fast decaying  $\sigma_{n}$  (low effective rank) has smaller errors. This forms the basis of our next result, which shows that if the target relationship possesses an effective low rank structure in terms of the decay of singular values, then we can achieve a more efficient approximation using encoder-decoder structures by limiting the size of the coding vector. Detailed definition for  $\{\sigma_n\}$  and the proof are found in Appendix D.

Theorem 4.3. Assume the same conditions as in Theorem 4.2. Then for any  $m_E, m_D, N \in \mathbb{N}_+$  with  $N \leq \bar{m}$ , there exists  $\widehat{\pmb{H}} \in \widehat{\mathcal{H}}_{m_E, m_D, N}$  such that

$$
\begin{array}{l} \| \boldsymbol {H} - \hat {\boldsymbol {H}} \| \lesssim \frac {C (\alpha) \gamma d}{\beta^ {2}} \left\{\left(1 + \sqrt {\bar {m} - N}\right) \cdot \left(\frac {1}{m _ {E} ^ {\alpha}} + \frac {1}{m _ {D} ^ {\alpha}}\right) + \left(\sum_ {n = N + 1} ^ {\bar {m}} \sigma_ {n} ^ {2}\right) ^ {1 / 2} \right. \\ \left. + \left(\sum_ {n = N + 1} ^ {\bar {m}} \sigma_ {n}\right) ^ {1 / 2} \cdot \left(\frac {1}{m _ {E} ^ {\alpha / 2}} + \frac {1}{m _ {D} ^ {\alpha / 2}}\right) \right\}, \tag {17} \\ \end{array}
$$

where  $\lesssim$  hides universal positive constants, and  $\bar{m} = \min \{m_E,m_D\}$ . Here, the number of trainable parameters is  $dN(m_E + m_D)$  with  $N\leq \bar{m}$ .

This is a finer approximation rate compared to Theorem 4.2, where both the widths  $(m_E, m_D)$  and the coding vector size  $N$  affect the model capacity. Besides smoothness and memory decay, we have the additional rank structure of the target relationship, which is characterised by its singular values  $\{\sigma_n\}$ . We again look at the class of functionals that can be well approximated. Smoothness  $\alpha$  and decay rate  $\beta$  is same as Theorem 4.2. We consider the rank structure  $\{\sigma_n\}$ , where we observe that the error bound is small if  $\{\sigma_n\}$  have small tail sum  $\sum_{n=N+1}^{\bar{m}} \sigma_n^2$ . This means that a target with fast decaying  $\{\sigma_n\}$  or low "effective rank" can be well approximated by the RNN encoder-decoder with fewer parameters. Due to the Eckart-Young-like low rank approximation form, we can choose a proper  $N$  based on the decay rate of singular values. This balances the number of parameters and the approximation error.

Here, we emphasise the temporal product is an intrinsic structure arising from the encoder-decoder architecture. Recall the key structure of encoder-decoder: it first encodes the input sequence into the coding vector and then decodes an entire output sequence from it. In this sense, the coding vector is the only interaction between the input and the output. Thus, the size of the coding vector  $N$  is an essential measure of model capacity concerning the dependence of outputs on the inputs. Here, we show that this concept can be formalised as a notion of rank, which can pinpoint the precise types of input-output relationships that encoder-decoder structures are well adapted to. This is one of the most unique properties of these structures.

Numerical illustrations. Here we use a numerical example to illustrate the above discussions. We look at how the singular value decay rate, rank  $N_0$  of the target relationships, and model rank  $N$  affect the approximation error  $\| H - \hat{H} \|$ .

![](images/6faf2d183d9ceb9e82e74b492240a1af1917c1689bb5c9ce6b0381d2886a4b8e.jpg)  
(a)  $\sigma_{n} = \left\{ \begin{array}{ll}n^{-\frac{1}{8}}, & n\leq N_{0}\\ 0, & n > N_{0} \end{array} \right.$

![](images/79f15b62f3884eeaa24f7203fc1c18f959428094b228d715518b68045909759d.jpg)  
(b)  $\sigma_{n} = \left\{ \begin{array}{ll}n^{-1}, & n\leq N_{0}\\ 0, & n > N_{0} \end{array} \right.$

![](images/11de75ff1682a9604a5d4b524389df3ef9c1cbd4e525a68854d99a3b583f67d3.jpg)  
Figure 2: In (a), (b), (c) we consider target relationships with different singular values indicated in the respective caption. For (a), (b) we also consider targets with different rank, where  $N_0 = 2, 4, 6, 8$ . We use models with fixed width  $m = m_E = m_D = 128$  with coding vector size  $N$ . Detail settings are found in Appendix E.2.  
(c)  $\sigma_{n} = n^{-2}$

In Figure 2, we train linear encoder-decoder structures to learn three relationships of different ranks determined by the decay patterns of singular values, given in (a), (b) and (c). Different colours mean targets with different rank. From Figure 2 we have the following observations consistent with previous discussions. First, observe that increasing the model rank  $N$  makes approximation errors smaller, as expected. Moreover, note that when increasing  $N$ , the speeds of decrements of errors are different. If singular values decay fast, errors also decay fast. This implies that a target with fast decaying singular values can be approximated efficiently with fewer parameters (smaller  $N$ ).

For each experiment, we are able to achieve low approximation error by choosing  $N \ll m$ . The error will remain unchanged or decreases much slower when further increasing  $N$ . This means that in practice, one can choose  $N$  such that it covers the major singular values of the target in order to improve the approximation efficiency while maintaining small errors. That is, the notion of model reduction.

# 5 CONCLUSION

We theoretically study the approximation properties of the RNN encoder-decoder in a linear setting. We prove a universal approximation result of linear functions by encoder-decoder structures and show that they generalise the RNN to the time-inhomogeneous setting. Moreover, we discover an important structure, which we call the temporal product structure, that characterises the types of input-output relationships that are especially suited for approximation using encoder-decoders. This elucidates the key differences between these novel architectures and classical methods for learning time-series, and forms a basic step towards understanding the intricacies of modern deep learning architectures for time-series analysis.

Reproducibility Statement. Detailed proofs for theoretical results, and complete settings of numerical examples are found in the appendix. The source code for numerical examples can be made available upon request.

Here is a quick reference:

<table><tr><td>Proposition 3.1</td><td>Properties of RNN encoder-decoder functionals</td><td>Appendix A</td></tr><tr><td>Theorem 4.1</td><td>Universal approximation theorem</td><td>Appendix B</td></tr><tr><td>Theorem 4.2</td><td>General approximation rates</td><td>Appendix C</td></tr><tr><td>Theorem 4.3</td><td>Approximation rates considering temporal product structure</td><td>Appendix D</td></tr><tr><td>Figure 1</td><td>Illustration of high rank, low rank temporal relationships</td><td>Appendix E.1</td></tr><tr><td>Figure 2</td><td>Numerical examples</td><td>Appendix E.2</td></tr></table>

# REFERENCES

Dzmitry Bahdanau, Kyung Hyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. 3rd International Conference on Learning Representations, ICLR 2015 - Conference Track Proceedings, pp. 1-15, 2015.  
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.  
G Berkooz, P Holmes, and J L Lumley. The proper orthogonal decomposition in the analysis of turbulent flows. Annual Review of Fluid Mechanics, 25(1):539-575, 1993. doi: 10.1146/annurev.fl.25.010193.002543. URL https://doi.org/10.1146/annurev.fl.25.010193.002543.  
Vladimir I Bogachev. Measure theory, volume 1. Springer Science & Business Media, 2007.  
Ching-Hua Chang and Chung-Wei Ha. On eigenvalues of differentiable positive definite kernels. Integral Equations and Operator Theory, 33:1-7, 03 1999. doi: 10.1007/BF01203078.  
Anindya Chatterjee. An introduction to the proper orthogonal decomposition. Current Science, 78(7):808-817, 2000. ISSN 00113891. URL http://www.jstor.org/stable/24103957.  
Chung-Cheng Chiu, Tara N. Sainath, Yonghui Wu, Rohit Prabhavalkar, Patrick Nguyen, Zhifeng Chen, Anjuli Kannan, Ron J. Weiss, Kanishka Rao, Ekaterina Gonina, Navdeep Jaitly, Bo Li, Jan Chorowski, and Michiel Bacchiani. State-of-the-art speech recognition with sequence-to-sequence models, 2018.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP 2014 - 2014 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference, pp. 1724-1734, 2014. doi: 10.3115/v1/d14-1179.  
Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, and Yoshua Bengio. On the Properties of Neural Machine Translation: Encoder-Decoder Approaches. pp. 103-111, 2015. doi: 10.3115/v1/w14-4012.  
Kenji Doya. Universality of fully connected recurrent neural networks. Dept. of Biology, UCSD, Tech. Rep, 1993.  
Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211-218, 1936.  
Ken-ichi Funahashi and Yuichi Nakamura. Approximation of dynamical systems by continuous time recurrent neural networks. *Neural Networks*, 6(6):801 - 806, 1993. ISSN 0893-6080.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.  
Haotian Jiang, Zhong Li, and Qianxiao Li. Approximation theory of convolutional architectures for time series modelling. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 4961-4970. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr.press/v139/jiang21d.html.  
Nal Kalchbrenner and Phil Blunsom. Recurrent continuous translation models. EMNLP 2013 - 2013 Conference on Empirical Methods in Natural Language Processing, Proceedings of the Conference, (October):1700-1709, 2013.  
Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu. Neural speech synthesis with transformer network. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 6706-6713, 2019.

Zhong Li, Jiequn Han, Weinan E, and Qianxiao Li. On the curse of memory in recurrent neural networks: Approximation and optimization analysis. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=8Sqhl-nF50.  
Y.C. Liang, H.P. Lee, S.P. Lim, W.Z. Lin, K.H. Lee, and C.G. Wu. Proper orthogonal decomposition and its applications—part i: Theory. Journal of Sound and Vibration, 252 (3):527-544, 2002. ISSN 0022-460X. doi: https://doi.org/10.1006/jsvi.2001.4041. URL https://www.sciencedirect.com/science/article/pii/S0022460X01940416.  
G.G. Lorentz. Approximation of Functions. AMS Chelsea Publishing Series. Holt, Rinehart and Winston, 2005. ISBN 9780821840504. URL https://books.google.com.sg/books?id=8VMr0mTKSe0C.  
Wolfgang Maass, Prashant Joshi, and Eduardo D Sontag. Computational aspects of feedback in neural circuits. PLOS Computational Biology, 3(1):e165, 2007.  
Charles Bradfield Morrey. Multiple integrals in the calculus of variations. Springer-Verlag, 1966.  
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International Conference on Machine Learning, pp. 4055-4064. PMLR, 2018.  
W. Rudin, W.A. RUDIN, and Tata McGraw-Hill Publishing Company. Real and Complex Analysis. Higher Mathematics Series. McGraw-Hill Education, 1987. ISBN 9780070542341. URL https://books.google.com.sg/books?id=Z_fuAAAAMAAJ.  
David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning representations by back-propagating errors. Nature, 1986. ISSN 00280836. doi: 10.1038/323533a0.  
Anton Maximilian Schäfer and Hans-Georg Zimmermann. Recurrent neural networks are universal approximators. International journal of neural systems, 17(04):253-263, 2007.  
Martin H. Schultz.  $L^{\infty}$ -multivariate approximation theory. SIAM Journal on Numerical Analysis, 6(2):161-183, 1969. doi: 10.1137/0706017. URL https://doi.org/10.1137/0706017.  
Ilya Sutskever, Oriol Vinyals, and Quoc V. Le. Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems, 4(January):3104-3112, 2014. ISSN 10495258.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 2017-Decem(Nips):5999-6009, 2017. ISSN 10495258.  
Subhashini Venugopalan, Marcus Rohrbach, Jeffrey Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko. Sequence to sequence-video to text. In Proceedings of the IEEE international conference on computer vision, pp. 4534-4542, 2015.  
Jong Chul Ye and Woon Kyoung Sung. Understanding geometry of encoder-decoder CNNs. 36th International Conference on Machine Learning, ICML 2019, 2019-June: 12245-12254, 2019.  
Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions?, 2020.
