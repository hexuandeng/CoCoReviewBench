# GENERALIZED TENSOR MODELS FOR RECURRENT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recurrent Neural Networks (RNNs) are very successful at solving challenging problems with sequential data. However, this observed efficiency is not yet entirely explained by theory. It is known that a certain class of multiplicative RNNs enjoys the property of depth efficiency — a shallow network of exponentially large width is necessary to realize the same score function as computed by such an RNN. Such networks, however, are not very often applied to real life tasks. In this work, we attempt to reduce the gap between theory and practice by extending the theoretical analysis to RNNs which employ various nonlinearities, such as Rectified Linear Unit (ReLU), and show that they also benefit from properties of universality and depth efficiency. Our theoretical results are verified by a series of extensive computational experiments.

# 1 INTRODUCTION

Recurrent Neural Networks are firmly established to be one of the best deep learning techniques when the task at hand requires processing sequential data, such as text, audio, or video (Graves et al., 2013; Mikolov et al., 2011; Gers et al., 1999). The ability of these neural networks to efficiently represent a rich class of functions with a relatively small number of parameters is often referred to as depth efficiency, and the theory behind this phenomenon is not yet fully understood. A recent line of work (Cohen & Shashua, 2016; Cohen et al., 2016; Khrulkov et al., 2018; Cohen et al., 2018) focuses on comparing various deep learning architectures in terms of their expressive power.

It was shown in (Cohen et al., 2016) that ConvNets with product pooling are exponentially more expressive than shallow networks, that is there exist functions realized by ConvNets which require an exponentially large number of parameters in order to be realized by shallow nets. A similar result also holds for RNNs with multiplicative recurrent cells (Khrulkov et al., 2018). We aim to extend this analysis to RNNs with rectifier nonlinearities which are often used in practice. The main challenge of such analysis is that the tools used for analyzing multiplicative networks, namely, properties of standard tensor decompositions and ideas from algebraic geometry, can not be applied in this case, and thus some other approach is required. Our objective is to apply the machinery of generalized tensor decompositions, and show universality and existence of depth efficiency in such RNNs.

# 2 RELATED WORK

In recent years a great deal of work was done in applications of tensor calculus to both theoretical and practical aspects of deep learning algorithms. (Lebedev et al., 2015) represented filters in a convolutional network with CP decomposition (Harshman, 1970; Carroll & Chang, 1970) which allowed for much faster inference at the cost of a negligible drop in performance. (Novikov et al., 2015) proposed to use Tensor Train (TT) decomposition (Oseledets, 2011) to compress fully-connected layers of large neural networks while preserving their expressive power. Later on, TT was exploited to reduce the number of parameters and improve the performance of recurrent networks in long-term forecasting (Yu et al., 2017) and video classification (Yang et al., 2017) problems.

In addition to the practical benefits, tensor decompositions were used to analyze theoretical aspects of deep neural nets. (Cohen et al., 2016) investigated a connection between various network architectures and tensor decompositions, which made possible to compare their expressive power. Specifically, it

was shown that CP and Hierarchical Tucker (Grasedyck, 2010) decompositions correspond to shallow networks and convolutional networks respectively. Recently, this analysis was extended by (Khrulkov et al., 2018) who showed that TT decomposition can be represented as a recurrent network with multiplicative connections. This specific form of RNNs was also empirically proved to provide a substantial performance boost over standard RNN models (Wu et al., 2016).

First results on the connection between tensor decompositions and neural networks were obtained for rather simple architectures, however, later on, they were extended in order to analyze more practical deep neural nets. It was shown that theoretical results can be generalized to a large class of CNNs with ReLU nonlinearities (Cohen & Shashua, 2016) and dilated convolutions (Cohen et al., 2018), providing valuable insights on how they can be improved. However, there is a missing piece in the whole picture as theoretical properties of more complex nonlinear RNNs have yet to be analyzed. In this paper, we elaborate on this problem and present new tools for conducting a theoretical analysis of such RNNs, specifically when rectifier nonlinearities are used.

# 3 ARCHITECTURES INSPIRED BY TENSOR DECOMPOSITIONS

Let us now recall the known results about the connection of tensor decompositions and multiplicative architectures, and then show how they are generalized in order to include networks with ReLU nonlinearities.

# 3.1 SCORE FUNCTIONS AND FEATURE TENSOR

Suppose that we are given a dataset of objects with a sequential structure, i.e. every object in the dataset can be written as

$$
X = \left(\mathbf {x} ^ {(1)}, \mathbf {x} ^ {(2)}, \dots , \mathbf {x} ^ {(T)}\right), \quad \mathbf {x} ^ {(t)} \in \mathbb {R} ^ {N}. \tag {1}
$$

We also introduce a parametric feature map  $f_{\theta}:\mathbb{R}^{N}\to \mathbb{R}^{M}$  which essentially preprocesses the data before it is fed into the network. Assumption 1 holds for many types of data, e.g. in the case of natural images we can cut them into rectangular patches which are then arranged into vectors  $\mathbf{x}^{(t)}$ . A typical choice for the feature map  $f_{\theta}$  in this particular case is an affine map followed by a nonlinear activation:  $f_{\theta}(\mathbf{x}) = \sigma (\mathbf{A}\mathbf{x} + \mathbf{b})$ . To draw the connection between tensor decompositions and feature tensors we consider the following score functions (logits):

$$
\ell (X) = \langle \boldsymbol {\mathcal {W}}, \boldsymbol {\Phi} (X) \rangle = (\operatorname {v e c} \boldsymbol {\mathcal {W}}) ^ {\top} \operatorname {v e c} \boldsymbol {\Phi} (X), \tag {2}
$$

where  $\mathcal{W} \in \mathbb{R}^{M \times M \times \dots \times M}$  is a trainable  $T$ -way weight tensor and  $\Phi(X) \in \mathbb{R}^{M \times M \times \dots \times M}$  is the so-called feature tensor, defined as

$$
\Phi (X) = f _ {\theta} \left(\mathbf {x} ^ {(1)}\right) \otimes f _ {\theta} \left(\mathbf {x} ^ {(2)}\right) \dots \otimes f _ {\theta} \left(\mathbf {x} ^ {(T)}\right), \tag {3}
$$

where we have used the operation of outer product  $\otimes$ , which is important in tensor calculus. For a tensor  $\mathcal{A}$  of order  $N$  and a tensor  $\mathcal{B}$  of order  $M$  their outer product  $\mathcal{C} = \mathcal{A} \otimes \mathcal{B}$  is a tensor of order  $N + M$  defined as:

$$
\mathcal {C} _ {i _ {1} i _ {2} \dots i _ {N} j _ {1} j _ {2} \dots j _ {M}} = \boldsymbol {\mathcal {A}} _ {i _ {1} i _ {2} \dots i _ {N}} \boldsymbol {\mathcal {B}} _ {j _ {1} j _ {2} \dots j _ {M}}. \tag {4}
$$

It is known that equation 2 possesses the universal approximation property (it can approximate any function with any prescribed precision given sufficiently large  $M$ ) under mild assumptions on  $f_{\theta}$  (Cohen et al., 2016; Girosi & Poggio, 1990).

# 3.2 TENSOR DECOMPOSITIONS

Working the entire weight tensor  $\mathcal{W}$  in eq. (2) is impractical for large  $M$  and  $T$ , since it requires exponential in  $T$  number of parameters. Thus, we compactly represent it using tensor decompositions, which will further lead to different neural network architectures, referred to as tensor networks (Cichocki et al., 2017).

CP-decomposition The most basic decomposition is the so-called Canonical (CP) decomposition (Harshman, 1970; Carroll & Chang, 1970) which is defined as follows

$$
\boldsymbol {\mathcal {W}} = \sum_ {r = 1} ^ {R} \lambda_ {r} \mathbf {v} _ {r} ^ {(1)} \otimes \mathbf {v} _ {r} ^ {(2)} \dots \otimes \mathbf {v} _ {r} ^ {(T)}, \tag {5}
$$

where  $\mathbf{v}_r^{(t)}\in \mathbb{R}^M$  and minimal value of  $R$  such that decomposition equation 5 exists is called canonical rank of a tensor (CP-rank). By substituting eq. (5) into eq. (2) we find that

$$
\ell (X) = \sum_ {r = 1} ^ {R} \lambda_ {r} \left[ \langle f _ {\theta} (\mathbf {x} ^ {(1)}), \mathbf {v} _ {r} ^ {(1)} \rangle \otimes \dots \otimes \langle f _ {\theta} (\mathbf {x} ^ {(T)}), \mathbf {v} _ {r} ^ {(T)} \rangle \right] = \sum_ {r = 1} ^ {R} \lambda_ {r} \prod_ {t = 1} ^ {T} \langle f _ {\theta} (\mathbf {x} ^ {(t)}), \mathbf {v} _ {r} ^ {(t)} \rangle . \tag {6}
$$

In the equation above, outer products  $\otimes$  are taken between scalars and coincide with the ordinary products between two numbers. However, we would like to keep this notation as it will come in handy later, when we generalize tensor decompositions to include various nonlinearities.

TT-decomposition Another tensor decomposition is Tensor Train (TT) decomposition (Oseledets, 2011) which is defined as follows

$$
\boldsymbol {\mathcal {W}} = \sum_ {r _ {1} = 1} ^ {R _ {1}} \dots \sum_ {r _ {T - 1} = 1} ^ {R _ {T - 1}} \mathbf {g} _ {r _ {0} r _ {1}} ^ {(1)} \otimes \mathbf {g} _ {r _ {1} r _ {2}} ^ {(2)} \otimes \dots \otimes \mathbf {g} _ {r _ {T - 1} r _ {T}} ^ {(T)}, \tag {7}
$$

where  $\mathbf{g}_{r_{t-1}r_t}^{(t)} \in \mathbb{R}^M$  and  $r_0 = r_T = 1$  by definition. If we gather vectors  $\mathbf{g}_{r_{t-1}r_t}^{(t)}$  for all corresponding indices  $r_{t-1} \in \{1, \ldots, R_{t-1}\}$  and  $r_t \in \{1, \ldots, R_t\}$  we will obtain three-dimensional tensors  $\mathcal{G}^{(t)} \in \mathbb{R}^{M \times R_{t-1} \times R_t}$  (for  $t = 1$  and  $t = T$  we will get matrices  $\mathcal{G}^{(1)} \in \mathbb{R}^{M \times 1 \times R_1}$  and  $\mathcal{G}^{(T)} \in \mathbb{R}^{M \times R_{T-1} \times 1}$ ). The set of all such tensors  $\{\mathcal{G}^{(t)}\}_{t=1}^T$  is called TT-cores and minimal values of  $\{R_t\}_{t=1}^{T-1}$  such that decomposition equation 7 exists are called TT-ranks. In the case of TT decomposition, the score function has the following form:

$$
\ell (X) = \sum_ {r _ {1} = 1} ^ {R _ {1}} \dots \sum_ {r _ {T - 1} = 1} ^ {R _ {T - 1}} \prod_ {t = 1} ^ {T} \left\langle f _ {\theta} \left(\mathbf {x} ^ {(t)}\right), \mathbf {g} _ {r _ {t - 1} r _ {t}} ^ {(t)} \right\rangle . \tag {8}
$$

# 3.3 CONNECTION BETWEEN TT AND RNN

Now we want to show that the score function for Tensor Train decomposition exhibits particular recurrent structure similar to that of RNN. We define the following hidden states:

$$
\mathbf {h} ^ {(1)} \in \mathbb {R} ^ {R _ {1}}: \mathbf {h} _ {r _ {1}} ^ {(1)} = \langle f _ {\theta} (\mathbf {x} ^ {(1)}), \mathbf {g} _ {r _ {0} r _ {1}} ^ {(1)} \rangle ,
$$

$$
\mathbf {h} ^ {(t)} \in \mathbb {R} ^ {R _ {t}}: \mathbf {h} _ {r _ {t}} ^ {(t)} = \sum_ {r _ {t - 1} = 1} ^ {R _ {t - 1}} \left\langle f _ {\theta} \left(\mathbf {x} ^ {(t)}\right), \mathbf {g} _ {r _ {t - 1} r _ {t}} ^ {(t)} \right\rangle \mathbf {h} _ {r _ {t - 1}} ^ {(t - 1)} \quad t = 2, \dots , T. \tag {9}
$$

Such definition of hidden states allows for more compact form of the score function.

Lemma 3.1. Under the notation introduced in eq. (9), the score function can be written as

$$
\ell (X) = \mathbf {h} ^ {(T)} \in \mathbb {R} ^ {1}.
$$

Proof of Lemma 3.1 as well as the proofs of our main results from Section 5 were moved to Appendix A due to limited space.

Note that with a help of TT-cores we can rewrite eq. (9) in a more convenient index form:

$$
\mathbf {h} _ {k} ^ {(t)} = \sum_ {i, j} \boldsymbol {\mathcal {G}} _ {i j k} ^ {(t)} f _ {\theta} \left(\mathbf {x} ^ {(t)}\right) _ {i} \mathbf {h} _ {j} ^ {(t - 1)} = \sum_ {i, j} \boldsymbol {\mathcal {G}} _ {i j k} ^ {(t)} \left[ f _ {\theta} \left(\mathbf {x} ^ {(t)}\right) \otimes \mathbf {h} ^ {(t - 1)} \right] _ {i j}, \quad k = 1, \dots , R _ {t}, \tag {10}
$$

where the operation of tensor contraction is used. Combining all weights from  $\mathcal{G}^{(t)}$  and  $f_{\theta}(\cdot)$  into a single variable  $\Theta_{\pmb{\varrho}}^{(t)}$  and denoting the composition of feature map, outer product, and contraction as  $g:\mathbb{R}^{R_{t - 1}}\times \mathbb{R}^N\times \mathbb{R}^{N\times R_{t - 1}\times R_t}\to \mathbb{R}^{R_t}$  we arrive at the following vector form:

$$
\mathbf {h} ^ {(t)} = g \left(\mathbf {h} ^ {(t - 1)}, \mathbf {x} ^ {(t)}; \Theta_ {\boldsymbol {g}} ^ {(t)}\right), \quad \mathbf {h} ^ {(t)} \in \mathbb {R} ^ {R _ {t}}. \tag {11}
$$

This equation can be considered as a generalization of hidden state equation for Recurrent Neural Networks as here all hidden states  $\mathbf{h}^{(t)}$  may in general have different dimensionalities and weight tensors  $\Theta_{\pmb{\mathcal{G}}}^{(t)}$  depend on the time step. However, if we set  $R = R_{1} = \dots = R_{T - 1}$  and  $\pmb {\mathcal{G}} = \pmb{\mathcal{G}}^{(2)} = \dots = \pmb{\mathcal{G}}^{(T - 1)}$  we will get simplified hidden state equation used in standard recurrent architectures:

$$
\mathbf {h} ^ {(t)} = g \left(\mathbf {h} ^ {(t - 1)}, \mathbf {x} ^ {(t)}; \Theta_ {\boldsymbol {g}}\right), \quad \mathbf {h} ^ {(t)} \in \mathbb {R} ^ {R}, \quad t = 2, \dots , T - 1. \tag {12}
$$

Note that this equation is applicable to all hidden states except for the first  $\mathbf{h}^{(1)} = \pmb{\mathcal{G}}^{(1)}f_{\theta}(\mathbf{x}^{(1)})$  and for the last  $\mathbf{h}^{(T)} = f_{\theta}^{\top}(\mathbf{x}^{(T)})\pmb{\mathcal{G}}^{(T)}\mathbf{h}^{(T - 1)}$ , due to two-dimensional nature of the corresponding TT-cores. However, we can always pad the input sequence with two auxiliary vectors  $\mathbf{x}^{(0)}$  and  $\mathbf{x}^{(T + 1)}$  to get full compliance with the standard RNN structure. Figure 1 depicts tensor network induced by TT decomposition with cores  $\{\pmb{\mathcal{G}}^{(t)}\}_{t = 1}^{T}$ .

![](images/e2da66625af12e30e6601344b57d0e213fcade0e73f57b23b1dc82d48dbe352a.jpg)  
Figure 1: Neural network architecture which corresponds to recurrent TT-Network.

# 4 GENERALIZED TENSOR NETWORKS

# 4.1 GENERALIZED OUTER PRODUCT

In the previous section we showed that tensor decompositions correspond to neural networks of specific structure, which are simplified versions of those used in practice as they contain multiplicative nonlinearities only. One possible way to introduce more practical nonlinearities is to replace outer product  $\otimes$  in eq. (6) and eq. (10) with a generalized operator  $\otimes_{\xi}$  in analogy to kernel methods when scalar product is replaced by nonlinear kernel function. Let  $\xi : \mathbb{R} \times \mathbb{R} \to \mathbb{R}$  be an associative and commutative binary operator  $(\forall x, y, z \in \mathbb{R} : \xi(\xi(x, y), z) = \xi(x, \xi(y, z))$  and  $\forall x, y \in \mathbb{R} : \xi(x, y) = \xi(y, x)$ ). Note that this operator easily generalizes to the arbitrary number of operands due to associativity. For a tensor  $\mathcal{A}$  of order  $N$  and a tensor  $\mathcal{B}$  of order  $M$  we define their generalized outer product  $\mathcal{C} = \mathcal{A} \otimes_{\xi} \mathcal{B}$  as an  $(N + M)$  order tensor with entries given by:

$$
\mathcal {C} _ {i _ {1} \dots i _ {N} j _ {1} \dots j _ {M}} = \xi \left(\boldsymbol {\mathcal {A}} _ {i _ {1} \dots i _ {N}}, \boldsymbol {\mathcal {B}} _ {j _ {1} \dots j _ {M}}\right). \tag {13}
$$

Now we can replace  $\otimes$  in eqs. (6) and (10) with  $\otimes_{\xi}$  and get networks with various nonlinearities. For example, if we take  $\xi(x,y) = \max(x,y,0)$  we will get an RNN with rectifier nonlinearities; if we take  $\xi(x,y) = \ln(e^x + e^y)$  we will get an RNN with softplus nonlinearities; if we take  $\xi(x,y) = xy$  we will get a simple RNN defined in the previous section. Concretely, we will analyze the following networks.

# Generalized shallow network with  $\xi$ -nonlinearity

- Score function:

$$
\begin{array}{l} \ell (X) = \sum_ {r = 1} ^ {R} \lambda_ {r} \left[ \langle f _ {\theta} \left(\mathbf {x} ^ {(1)}\right), \mathbf {v} _ {r} ^ {(1)} \rangle \otimes_ {\xi} \dots \otimes_ {\xi} \langle f _ {\theta} \left(\mathbf {x} ^ {(T)}\right), \mathbf {v} _ {r} ^ {(T)} \rangle \right] \tag {14} \\ = \sum_ {r = 1} ^ {R} \lambda_ {r} \xi \left(\langle f _ {\theta} (\mathbf {x} ^ {(1)}), \mathbf {v} _ {r} ^ {(1)} \rangle , \dots , \langle f _ {\theta} (\mathbf {x} ^ {(T)}), \mathbf {v} _ {r} ^ {(T)} \rangle\right) \\ \end{array}
$$

- Parameters of the network:

$$
\Theta = \left(\left\{\lambda_ {r} \right\} _ {r = 1} ^ {R} \in \mathbb {R}, \left\{\mathbf {v} _ {r} ^ {(t)} \right\} _ {r = 1, t = 1} ^ {R, T} \in \mathbb {R} ^ {M}\right) \tag {15}
$$

# Generalized RNN with  $\xi$ -nonlinearity

- Score function:

$$
\begin{array}{l} \mathbf {h} _ {k} ^ {(t)} = \sum_ {i, j} \boldsymbol {\mathcal {G}} _ {i j k} ^ {(t)} \left[ \mathbf {C} ^ {(t)} f _ {\theta} (\mathbf {x} ^ {(t)}) \otimes_ {\xi} \mathbf {h} ^ {(t - 1)} \right] _ {i j} = \sum_ {i, j} \boldsymbol {\mathcal {G}} _ {i j k} ^ {(t)} \xi \left(\left[ \mathbf {C} ^ {(t)} f _ {\theta} (\mathbf {x} ^ {(t)}) \right] _ {i}, \mathbf {h} _ {j} ^ {(t - 1)}\right) \\ \ell (X) = \mathbf {h} ^ {(T)} \tag {16} \\ \end{array}
$$

- Parameters of the network:

$$
\Theta = \left(\left\{\mathbf {C} ^ {(t)} \right\} _ {t = 1} ^ {T} \in \mathbb {R} ^ {L \times M}, \left\{\boldsymbol {\mathcal {G}} ^ {(t)} \right\} _ {t = 1} ^ {T} \in \mathbb {R} ^ {L \times R _ {t - 1} \times R _ {t}}\right) \tag {17}
$$

Note that in eq. (16) we have introduced the matrices  $\mathbf{C}^{(t)}$  acting on the input states. The purpose of this modification is to obtain the plausible property of generalized shallow networks being able to be represented as generalized RNNs of width 1 (i.e., with all  $R_{i} = 1$ ) for an arbitrary nonlinearity  $\xi$ . In the case of  $\xi(x,y) = xy$ , the matrices  $\mathbf{C}^{(t)}$  were not necessary, since they can be simply absorbed by  $\mathcal{G}^{(t)}$  via tensor contraction (see Appendix A for further clarification on these points).

Initial hidden state Note that generalized RNNs require some choice of the initial hidden state  $\mathbf{h}^{(0)}$ . We find that it is convenient both for theoretical analysis and in practice to initialize  $\mathbf{h}^{(0)}$  as unit of the operator  $\xi$ , i.e. such an element  $u$  that  $\xi(x,y,u) = \xi(x,y) \forall x,y \in \mathbb{R}$ . Henceforth, we will assume that such an element exists (e.g., for  $\xi(x,y) = \max(x,y,0)$  we take  $u = 0$ , for  $\xi(x,y) = xy$  we take  $u = 1$ ), and set  $\mathbf{h}^{(0)} = u$ . For example, in eq. (9) it was implicitly assumed that  $\mathbf{h}^{(0)} = 1$ .

# 4.2 GRID TENSORS

Introduction of generalized outer product allows us to investigate RNNs with wide class of nonlinear activation functions, especially ReLU. While this change looks appealing from the practical viewpoint, it complicates following theoretical analysis, as the transition from obtained networks back to tensors is not straightforward.

In the discussion above, every tensor network had corresponding weight tensor  $\mathcal{W}$  and we could compare expressivity of associated score functions by comparing some properties of this tensors, such as ranks (Khrulkov et al., 2018; Cohen et al., 2016). This method enabled comprehensive analysis of score functions, as it allows us to calculate and compare their values for all possible input sequences  $X = (\mathbf{x}^{(1)},\dots,\mathbf{x}^{(T)})$ . Unfortunately, we can not apply it in case of generalized tensor networks, as the replacement of standard outer product  $\otimes$  with its generalized version  $\otimes_{\xi}$  leads to the loss of conformity between tensor networks and weight tensors. Specifically, not for every generalized tensor network with corresponding score function  $\ell(X)$  now exists a weight tensor  $\mathcal{W}$  such that  $\ell(X) = \langle \mathcal{W}, \Phi(X) \rangle$ . Also, such properties as universality no longer hold automatically and we have to prove them separately. Indeed as it was noticed in (Cohen & Shashua, 2016) shallow networks with  $\xi(x,y) = \max(x,0) + \max(y,0)$  no longer have the universal approximation property. In order to conduct proper theoretical analysis, we adopt the apparatus of so-called grid tensors, first introduced in (Cohen & Shashua, 2016).

Given a set of fixed vectors  $\mathbb{X} = \{\mathbf{x}^{(1)},\dots ,\mathbf{x}^{(M)}\}$  referred to as templates, the grid tensor of  $\mathbb{X}$  is defined to be the tensor of order  $T$  and dimension  $M$  in each mode, with entries given by:

$$
\Gamma^ {\ell} (\mathbb {X}) _ {i _ {1} i _ {2} \dots i _ {T}} = \ell (X), \quad X = \left(\mathbf {x} ^ {(i _ {1})}, \mathbf {x} ^ {(i _ {2})}, \dots , \mathbf {x} ^ {(i _ {T})}\right), \tag {18}
$$

where each index  $i_t$  can take values from  $\{1, \ldots, M\}$ , i.e. we evaluate the score function on every possible input assembled from the template vectors  $\{\mathbf{x}^{(i)}\}_{i=1}^M$ . To put it simply, we previously considered the equality of score functions represented by tensor decomposition and tensor network on set of all possible input sequences  $X = (\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(T)})$ ,  $\mathbf{x}^{(t)} \in \mathbb{R}^N$ , and now we restricted this set to exponentially large but finite grid of sequences consisting of template vectors only.

Define the matrix  $\mathbf{F} \in \mathbb{R}^{M \times M}$  which holds the values taken by the representation function  $f_{\theta}: \mathbb{R}^N \to \mathbb{R}^M$  on the selected templates  $\mathbb{X}$ :

$$
\mathbf {F} \triangleq \left[ f _ {\theta} \left(\mathbf {x} ^ {(1)}\right) \quad f _ {\theta} \left(\mathbf {x} ^ {(2)}\right) \quad \dots \quad f _ {\theta} \left(\mathbf {x} ^ {(M)}\right). \right] ^ {\top} \tag {19}
$$

Using the matrix  $\mathbf{F}$  we note that the grid tensor of generalized shallow network has the following form (see Appendix A for derivation):

$$
\boldsymbol {\Gamma} ^ {\ell} (\mathbb {X}) = \sum_ {r = 1} ^ {R} \lambda_ {r} \left(\mathbf {F} \mathbf {v} _ {r} ^ {(1)}\right) \otimes_ {\xi} \left(\mathbf {F} \mathbf {v} _ {r} ^ {(2)}\right) \otimes_ {\xi} \dots \otimes_ {\xi} \left(\mathbf {F} \mathbf {v} _ {r} ^ {(T)}\right). \tag {20}
$$

Construction of the grid tensor for generalized RNN is a bit more involved. We find that its grid tensor  $\Gamma^{\ell}(\mathbb{X})$  can be computed recursively, similar to the hidden state in the case of a single input sequence. The exact formulas turned out to be rather cumbersome and we moved them to Appendix A.

# 5 MAIN RESULTS

With grid tensors at hand we are ready to compare the expressive power of generalized RNNs and generalized shallow networks. In the further analysis, we will assume that  $\xi(x,y) = \max(x,y,0)$ , i.e., we analyze RNNs and shallow networks with rectifier nonlinearity. However, we need to make two additional assumptions. First of all, similarly to (Cohen & Shashua, 2016) we fix some templates  $\mathbb{X}$  such that values of the score function outside of the grid generated by  $\mathbb{X}$  are irrelevant for classification and call them covering templates. It was argued that for image data values of  $M$  of order 100 are sufficient (corresponding covering template vectors may represent Gabor filters). Secondly, we assume that the feature matrix  $\mathbf{F}$  is invertible, which is a reasonable assumption and in the case of  $f_{\theta}(\mathbf{x}) = \sigma(\mathbf{A}\mathbf{x} + \mathbf{b})$  for any distinct template vectors  $\mathbb{X}$  the parameters  $\mathbf{A}$  and  $\mathbf{b}$  can be chosen in such a way that the matrix  $\mathbf{F}$  is invertible.

# 5.1 UNIVERSALITY

As was discussed in section 4.2 we can no longer use standard algebraic techniques to verify universality of tensor based networks. Thus, our first result states that generalized RNNs with  $\xi(x,y) = \max(x,y,0)$  are universal in a sense that any tensor of order  $T$  and size of each mode being  $m$  can be realized as a grid tensor of such RNN (and similarly of a generalized shallow network).

Theorem 5.1 (Universality). Let  $\mathcal{H} \in \mathbb{R}^{M \times M \times \dots \times M}$  be an arbitrary tensor of order  $T$ . Then there exist a generalized shallow network and a generalized RNN with rectifier nonlinearity  $\xi(x, y) = \max(x, y, 0)$  such that grid tensor of each of the networks coincides with  $\mathcal{H}$ .

Part of Theorem 5.1 which corresponds to generalized shallow networks readily follows from (Cohen & Shashua, 2016, Claim 4). In order to prove the statement for the RNNs the following two lemmas are used.

Lemma 5.1. Given two generalized RNNs with grid tensors  $\Gamma^{\ell_A}(\mathbb{X})$ ,  $\Gamma^{\ell_B}(\mathbb{X})$ , and arbitrary  $\xi$ -nonlinearity, there exists a generalized RNN with grid tensor  $\Gamma^{\ell_C}(\mathbb{X})$  satisfying

$$
\mathbf {\Gamma} ^ {\ell_ {C}} (\mathbb {X}) = a \mathbf {\Gamma} ^ {\ell_ {A}} (\mathbb {X}) + b \mathbf {\Gamma} ^ {\ell_ {B}} (\mathbb {X}), \quad \forall a, b \in \mathbb {R}.
$$

This lemma essentially states that the collection of grid tensors of generalized RNNs with any nonlinearity is closed under taking arbitrary linear combinations. Note that the same result clearly holds for generalized shallow networks because they are linear combinations of rank 1 shallow networks by definition.

Lemma 5.2. Let  $\pmb{\mathcal{E}}^{(j_1j_2\dots j_T)}$  be an arbitrary one-hot tensor, defined as

$$
\boldsymbol {\mathcal {E}} _ {i _ {1} i _ {2} \ldots i _ {T}} ^ {(j _ {1} j _ {2} \ldots j _ {T})} = \left\{ \begin{array}{l l} 1, & j _ {t} = i _ {t} \quad \forall t \in \{1, \ldots , T \}, \\ 0, & o t h e r w i s e. \end{array} \right.
$$

Then there exists a generalized RNN with rectifier nonlinearities such that its grid tensor satisfies

$$
\mathbf {\Gamma} ^ {\ell} (\mathbb {X}) = \boldsymbol {\mathcal {E}} ^ {(j _ {1} j _ {2} \dots j _ {T})}.
$$

This lemma states that in the special case of rectifier nonlinearity  $\xi (x,y) = \max (x,y,0)$  any basis tensor can be realized by some generalized RNN.

Proof of Theorem 5.1. By Lemma 5.2 for each one-hot tensor  $\mathcal{E}^{(i_1 i_2 \dots i_T)}$  there exists a generalized RNN with rectifier nonlinearities, such that its grid tensor coincides with this tensor. Thus, by Lemma 5.1 we can construct an RNN with

$$
\Gamma^ {\ell} (\mathbb {X}) = \sum_ {i _ {1}, i _ {2}, \dots , i _ {T}} \mathcal {H} _ {i _ {1} i _ {2} \dots i _ {d}} \pmb {\mathscr {E}} ^ {(i _ {1} i _ {2} \dots i _ {T})} = \pmb {\mathcal {H}}.
$$

For generalized shallow networks with rectifier nonlinearities see the proof of (Cohen & Shashua, 2016, Claim 4).

The same result regarding networks with product nonlinearities considered in (Khrulkov et al., 2018) directly follows from the well-known properties of tensor decompositions (see Appendix A).

We see that at least with such nonlinearities as  $\xi (x,y) = \max (x,y,0)$  and  $\xi (x,y) = xy$  all the networks under consideration are universal and can represent any possible grid tensor. Now let us head to a discussion of expressivity of these networks.

# 5.2 EXPRESSIVITY

As was discussed in the introduction, expressivity refers to the ability of some class of networks to represent the same functions as some other class much more compactly. In our case the parameters defining size of networks are ranks of the decomposition, i.e. in the case of generalized RNNs ranks determine the size of the hidden state, and in the case of generalized shallow networks rank determines the width of a network. It was proven in (Cohen et al., 2016; Khrulkov et al., 2018) that ConvNets and RNNs with multiplicative nonlinearities are exponentially more expressive than the equivalent shallow networks: shallow networks of exponentially large width are required to realize the same score functions as computed by these deep architectures. Similarly to the case of ConvNets (Cohen & Shashua, 2016), we find that expressivity of generalized RNNs with rectifier nonlinearity holds only partially, as discussed in the following two theorems. For simplicity, we assume that  $T$  is even.

Theorem 5.2 (Expressivity 1). For every value of  $R$  there exists a generalized RNN with ranks  $\leq R$  and rectifier nonlinearity which is exponentially more efficient than shallow networks, i.e., the corresponding grid tensor may be realized only by a shallow network with rectifier nonlinearity of width at least  $\frac{2}{MT} \min(M, R)^{T/2}$ .

This result states that at least for some subset of generalized RNNs expressivity holds: exponentially wide shallow networks are required to realize the same grid tensor. Proof of the theorem is rather straightforward: we explicitly construct an example of such RNN which satisfies the following description. Given an arbitrary input sequence  $X = \left(\mathbf{x}^{(1)},\ldots \mathbf{x}^{(T)}\right)$  assembled from the templates, these networks (if  $M = R$ ) produce 0 if  $X$  has the property that  $\mathbf{x}^{(1)} = \mathbf{x}^{(2)},\mathbf{x}^{(3)} = \mathbf{x}^{(4)},\dots ,\mathbf{x}^{(T - 1)} = \mathbf{x}^{(T)}$ , and 1 in every other case, i.e. they measure pairwise similarity of the input vectors. A precise proof is given in Appendix A.

In the case of multiplicative RNNs (Khrulkov et al., 2018) almost every network possessed this property. This is not the case, however, for generalized RNNs with rectifier nonlinearities.

Theorem 5.3 (Expressivity 2). For every value of  $R$  there exists an open set (which thus has positive measure) of generalized RNNs with rectifier nonlinearity  $\xi(x, y) = \max(x, y, 0)$ , such that for each RNN in this open set the corresponding grid tensor can be realized by a rank 1 shallow network with rectifier nonlinearity.

In other words, for every rank  $R$  we can find a set of generalized RNNs of positive measure such that the property of expressivity does not hold. In the numerical experiments in Section 6 and Appendix A we validate whether this can be observed in practice, and find that the probability of obtaining CP-ranks of polynomial size becomes negligible with large  $T$  and  $R$ . Proof of Theorem 5.3 is provided in Appendix A.

Shared case Note that all the RNNs used in practice have shared weights, which allows them to process sequences of arbitrary length. So far in the analysis we have not made such assumptions about RNNs (i.e.,  $\pmb{\mathcal{G}}^{(2)} = \dots = \pmb{\mathcal{G}}^{(T - 1)})$ . By imposing this constraint, we lose the property of universality; however, we believe that the statements of Theorems 5.2 and 5.3 still hold (without

requiring that shallow networks also have shared weights). Note that the example constructed in the proof of Theorem 5.3 already has this property, and for Theorem 5.2 we provide numerical evidence in Appendix A.

# 6 EXPERIMENTS

In this section, we study if our theoretical findings are supported by experimental data. In particular, we investigate whether generalized tensor networks can be used in practical settings, especially in problems typically solved by RNNs (such as natural language processing problems). Secondly, according to Theorem 5.3 for some subset of RNNs the equivalent shallow network may have a low rank. To get a grasp of how strong this effect might be in practice we numerically compute an estimate for this rank in various settings.

Performance For the first experiment, we use two computer vision datasets MNIST (LeCun et al., 1990) and CIFAR-10 (Krizhevsky & Hinton, 2009), and natural language processing dataset for sentiment analysis IMDB (Maas et al., 2011). For the first two datasets, we cut natural images into rectangular patches which are then arranged into vectors  $\mathbf{x}^{(t)}$  (similar to (Khrulkov et al., 2018)) and for IMDB dataset the input data already has the desired sequential structure.

Figure 2 depicts test accuracy on IMDB dataset for generalized shallow networks and RNNs with rectifier nonlinearity. We see that generalized shallow network of much higher rank is required to get the level of performance close to that achievable by generalized RNN. Due to limited space, we have moved the results of the experiments on the visual datasets to Appendix B.

![](images/adfca47e0dcfdc55bd247e7f2f46306b9cf4d8d52a0cf481ce35865d16ae8b56.jpg)  
Figure 2: Test accuracy on IMDB dataset for generalized RNNs and generalized shallow networks with respect to the total number of parameters  $(m = 50, T = 100, \xi(x, y) = \max(x, y, 0))$ .

![](images/16dee584f9e0d2e6594e98a67715871197ea727d01178fe0ca4caa88c1d6d0b7.jpg)  
Figure 3: Distribution of lower bounds on the rank of generalized shallow networks equivalent to randomly generated generalized RNNs of ranks  $1,2,4,8$ $(M = 10,T = 6)$

**Expressivity** For the second experiment we generate a number of generalized RNNs with different values of TT-rank  $r$  and calculate a lower bound on the rank of shallow network necessary to realize the same grid tensor (to estimate the rank we use the same technique as in the proof of Theorem 5.2). Figure 3 shows that for different values of  $R$  and generalized RNNs of the corresponding rank there exist shallow networks of rank 1 realizing the same grid tensor, which agrees well with Theorem 5.3. This result looks discouraging, however, there is also a positive observation. While increasing rank of generalized RNNs, more and more corresponding shallow networks will necessarily have exponentially higher rank. In practice we usually deal with RNNs of  $R = 10^{2} - 10^{3}$  (dimension of hidden states), thus we may expect that effectively any function besides negligible set realized by generalized RNNs can be implemented only by exponentially wider shallow networks. The numerical results for the case of shared cores and other nonlinearities are given in Appendix B.

# 7 CONCLUSION

In this paper, we sought a more complete picture of the connection between Recurrent Neural Networks and Tensor Train decomposition, one that involves various nonlinearities applied to hidden states. We showed how these nonlinearities could be incorporated into network architectures and provided complete theoretical analysis on the particular case of rectifier nonlinearity, elaborating on points of generality and expressive power. We believe our results will be useful to advance theoretical understanding of RNNs. In future work, we would like to extend the theoretical analysis to most competitive in practice architectures for processing sequential data such as LSTMs and attention mechanisms.

# REFERENCES

J Douglas Carroll and Jih-Jie Chang. Analysis of individual differences in multidimensional scaling via an N-way generalization of Eckart-Young decomposition. Psychometrika, 1970.  
Andrzej Cichocki, Anh-Huy Phan, Qibin Zhao, Namgil Lee, Ivan Oseledets, Masashi Sugiyama, Danilo P Mandic, et al. Tensor networks for dimensionality reduction and large-scale optimization: Part 2 applications and future perspectives. Foundations and Trends® in Machine Learning, 9(6): 431-673, 2017.  
Nadav Cohen and Amnon Shashua. Convolutional rectifier networks as generalized tensor decompositions. In International Conference on Machine Learning, pp. 955-963, 2016.  
Nadav Cohen, Or Sharir, and Amnon Shashua. On the expressive power of deep learning: A tensor analysis. In Conference on Learning Theory, pp. 698-728, 2016.  
Nadav Cohen, Ronen Tamari, and Amnon Shashua. Boosting dilated convolutional networks with mixed tensor decompositions. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=S1JHhv6TW.  
Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. Learning to forget: Continual prediction with LSTM. 1999.  
Federico Girosi and Tomaso Poggio. Networks and the best approximation property. Biological cybernetics, 63(3):169-176, 1990.  
Lars Grasedyck. Hierarchical singular value decomposition of tensors. SIAM Journal on Matrix Analysis and Applications, 31(4):2029-2054, 2010.  
Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. Speech recognition with deep recurrent neural networks. In Acoustics, speech and signal processing (icassp), 2013 IEEE international conference on, pp. 6645-6649. IEEE, 2013.  
Richard A Harshman. Foundations of the PARAFAC procedure: Models and conditions for an "explanatory" multimodal factor analysis. 1970.  
Valentin Khrulkov, Alexander Novikov, and Ivan Oseledets. Expressive power of recurrent neural networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=S1WRibb0Z.  
Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. SIAM review, 51(3): 455-500, 2009.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan Oseledets, and Victor Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. International Conference on Learning Representations, 2015.  
Yann LeCun, Bernhard E Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne E Hubbard, and Lawrence D Jackel. Handwritten digit recognition with a back-propagation network. In Advances in neural information processing systems, pp. 396-404, 1990.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142-150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/P11-1015.  
Tomáš Mikolov, Stefan Kombrink, Lukáš Burget, Jan Černocký, and Sanjeev Khudanpur. Extensions of recurrent neural network language model. In Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International Conference on, pp. 5528-5531. IEEE, 2011.  
Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. In Advances in Neural Information Processing Systems, pp. 442-450, 2015.  
Ivan V Oseledets. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5): 2295-2317, 2011.  
Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, and Ruslan R Salakhutdinov. On multiplicative integration with recurrent neural networks. In Advances in Neural Information Processing Systems, pp. 2856-2864, 2016.  
Yinchong Yang, Denis Krompass, and Volker Tresp. Tensor-train recurrent neural networks for video classification. arXiv preprint arXiv:1707.01786, 2017.  
Rose Yu, Stephan Zheng, Anima Anandkumar, and Yisong Yue. Long-term forecasting using tensor-train RNNs. arXiv preprint arXiv:1711.00073, 2017.
