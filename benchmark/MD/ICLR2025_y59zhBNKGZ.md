# TOWARDS MAKING LINEAR ATTENTION USABLE

# Anonymous authors Paper under double-blind review

# ABSTRACT

The original softmax based "all-to-all" attention mechanism in the extremely successful transformer architecture computes the attention between  $N$  tokens embedded in a  $D$ -dimensional head, in  $O(N^2 D)$  time and  $O(N^2)$  memory. Several linearly scaling attention mechanisms have been proposed over the seven years since the transformer algorithm was proposed, to address the quadratic complexity in  $N$ , and these "Linear" algorithms have been shown to achieve reasonable training performance in the papers they were introduced in. Despite the fact that most current LLM applications are trending towards processing larger token sequences in context (increasing  $N$  into the millions and beyond), and transformers have helped create foundational models in diverse domains beyond language (e.g., image, video, and audio processing), the proposed linear mechanisms surprisingly do not find wide usage, and moreover remain restricted to relatively small sized problems. We address two possible reasons for this. The first is that many of the proposed mechanisms, despite their linear complexity in  $N$ , have practically a relatively large memory footprint which is quadratic in  $D$ , which quickly fills memory of available GPUs. We show the memory complexity can be reduced by approaching calculations from a novel perspective in both the forward and backward passes in training. A second reason is that techniques that accelerate convergence, such as dropout, cannot be used during training in the current linear algorithms. This is mitigated via an alternative dropout mechanism. Results and comparison studies demonstrate the usefulness of the proposed approaches, while maintaining the linear scaling in  $N$  in both wall-clock time and memory usage.

# 1 INTRODUCTION

Transformers are the single deep learning architecture that underpin many recent successful applications in diverse fields such as natural language processing, speech, computer vision, biology, and so on. Transformers incorporate a Softmax-based all-to-all "Attention" mechanism between the input tokens Vaswani et al. (2017). While this has proven to be extraordinarily effective in learning tasks, it has a time and memory complexity of  $O(N^2 D)$  and  $O(N^2)$  respectively, where  $N$  is the number of tokens and  $D$  the dimension per attention head. Efficient hardware implementation can offer constant speedup, and significant memory reduction. A notable example is the widely used FlashAttention-2 Dao (2023), where the memory complexity is reduced to  $O(ND)$ , but the time complexity remains  $O(N^2 D)$ . Given how the industrial models are shifting towards ever larger  $N$  ( $\sim 10^7$  in current implementations), searching for a linearly scaling attention mechanism started shortly after the introduction of the original paper.

Previous proposals for a linearly scaling attention mechanism used either Sparsified/localized Attention, or Kernel Separation. In the former, the all-to-all attention, and as a result long-range context is not captured. In the former approach, all-to-all attention is not captured, resulting in a loss of long-range context. While this may be effective in certain situations (particularly for approximations during inference), it compromises the model's ability to learn from larger sequence, which was the goal in the first place. Kernel Separation, in contrast, successfully captures all-to-all attention and has demonstrated promising results on small benchmarks. However, its major drawback is the high memory consumption of  $O(ND^2)$ . This drawback has hindered its application in large language models (LLMs), limiting researchers' ability to study its effectiveness at scale. To put things in perspective, with a memory consumption of  $O(ND^2)$ , TinyLlama-1.1B Zhang et al. (2024b) would

require  $280\mathrm{GB}^1$  of memory per batch. This makes the Kernel Separation approach impractical on any production GPU unless the memory consumption is addressed.

We propose a new viewpoint for calculating Linear Attention, and show how the memory consumption can be reduced to  $O(ND)$ . Using our method, the memory consumption of TinyLlama-1.1B with Linear Attention is reduced to 2.2 GB memory. Our work does not introduce a new Attention mechanism, and our focus is addressing the high memory consumption issue in the many linear attention mechanisms proposed. Furthermore, since these methods do not explicitly calculate the attention matrix, dropout cannot be applied. This is problematic given the robustness and accelerated convergence that dropout offers. To address this, we introduce an alternative mechanism. In summary, we make the following contributions:

- Reduce the memory consumption of Linear attention from  $O(ND^2)$  to  $O(ND)$  (Section 3).  
- Introduce an alternative mechanism to dropout (Section 4).

# 2 BACKGROUND AND RELATED WORK

Notation: Bolded upper case letters, e.g.,  $\mathbf{X}$  indicate matrices, and bolded lower case letters  $\mathbf{x}_i$  and  $\mathbf{x}_{i,j}$  the  $i$ -th row and the element at the  $i$ -th row and  $j$ -th column of  $\mathbf{X}$  respectively.  $\mathbf{x}_j^T$  denotes the  $j$ -th row of  $\mathbf{X}$ . Unbolded letters  $X, x, x_i, x_{jkl}^{(i)}$  indicate scalars.

# 2.1 PRELIMINARIES: COMPUTING "VANILLA" ATTENTION

In vanilla attention, given a sequence of  $N$  tokens, model dimension of  $C$ ,  $H$  heads, and dimension per head of  $D = C / H$ , each head takes in three matrices  $\mathbf{Q}$ ,  $\mathbf{K}$ ,  $\mathbf{V} \in \mathbb{R}^{N \times D}$ , and gives an output matrix  $\mathbf{O} \in \mathbb{R}^{N \times D}$ . The output is calculated using a matrix vector product (MVP) of the Attention  $\mathbf{A}$  with  $\mathbf{V}$  as follows:

$$
\mathbf {O} = \mathbf {A V}, \mathbf {A} = \text {S o f t m a x} \left(\mathbf {Q K} ^ {T}\right), \mathbf {o} _ {\mathrm {i}, \mathrm {j}} = \frac {\sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {\mathrm {n}} / \sqrt {D}\right) \mathbf {v} _ {\mathrm {n} , \mathrm {j}}}{\sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {n} / \sqrt {D}\right)} = \frac {\sum_ {n = 1} ^ {N} f \left(\mathbf {q} _ {i} . \mathbf {k} _ {\mathrm {n}}\right) \mathbf {v} _ {\mathrm {n} , \mathrm {j}}}{\sum_ {n = 1} ^ {N} f \left(\mathbf {q} _ {i} . \mathbf {k} _ {n}\right)} (1)
$$

where  $f(x)$  is the attention kernel. Therefore, each head has a computational complexity of  $O(N^2 D)$ . The final output is given by concatenating outputs from each head, with a total  $O(N^2 C)$  computation. We can apply a causal mask by changing Eq. 1 to

$$
\mathbf {O} = \operatorname {t r i l} (\mathbf {A}) \mathbf {V}, \quad \operatorname {t r i l} (\mathbf {A}) _ {i, j} = \left\{ \begin{array}{l} \mathbf {a} _ {i, j}, j \leq i \\ 0, j > i \end{array} \right., \quad \mathbf {o} _ {i, j} = \frac {\sum_ {n = 1} ^ {i} f \left(\mathbf {q} _ {i} , \mathbf {k} _ {n}\right) \mathbf {v} _ {n , j}}{\sum_ {n = 1} ^ {i} f \left(\mathbf {q} _ {i} , \mathbf {k} _ {n}\right)}. \tag {2}
$$

The softmax attention kernel of  $f(x) = \exp (q\cdot k / \sqrt{D})$  could be changed. Any definite positive convex function can substitute for the attention kernel, and several functions have been explored in the literature.

# 2.2 RELATED WORK

The most successful approach to speedup the vanilla attention mechanism that has seen widespread use is efficient hardware implementation and the use of analytical gradients. The most notable such mechanism, which offers a constant speedup, and reduces memory consumption to  $O(ND)$  is FlashAttention-2 Dao (2023), Dao (2024).

However, this method does not address the issue of quadratic cost of large context attention. Because of this there is a lot of research on systems approaches to use multi-GPU systems, e.g., Liu et al. (2023).

The goal of this paper is to make sub-quadratic attention usable. Two common algorithm-based approaches proposed in the literature for this are Kernel Separation, and Sparsification.

Kernel Separation In this approach the attention kernel is chosen to be one that allows  $f(\mathbf{q}_i.\mathbf{k}_n)$  to be written as  $g(\mathbf{q}_i)^T h(\mathbf{k}_n)$ , where  $g(x)$  and  $h(x)$  are deliberately chosen  $\mathbb{R}^D \to \mathbb{R}^{D'}$  functions. The output can then be calculated as

$$
\mathbf {o} _ {\mathbf {i}, \mathbf {j}} = \frac {g \left(\mathbf {q} _ {i}\right) ^ {T} \sum_ {n = 1} ^ {N} h \left(\mathbf {k} _ {\mathbf {n}}\right) \mathbf {v} _ {\mathbf {n} , \mathbf {j}}}{g \left(\mathbf {q} _ {i}\right) ^ {T} \sum_ {n = 1} ^ {N} h \left(\mathbf {k} _ {n}\right)}, \tag {3}
$$

which brings down the calculation complexity to  $O(ND)$ . Common choices of  $f(x)$  are  $a + bx$  (e.g., Katharopoulos et al. (2020); Kasai et al. (2021); Zhang et al. (2024a); Choromanski et al. (2020); Qin et al. (2022b)),  $a + bx + cx^2$  Keles et al. (2023); Banerjee et al. (2020), with the coefficients either as the Taylor expansion of the exponential or as learnable parameters. The most widely adopted attention kernel is  $f(x) = a + bx$ , which is also known as Linear Attention. As previously mentioned, Linear Attention has a computational complexity of  $O(ND)$  and a memory complexity of  $O(ND^2)$ .

Sparsification In this approach, we only calculate the Attention for a fixed number of keys for each query. Common approaches consider keys with tokens that are spatially close-by to the query's token Zaheer et al.; Zhou et al.; Yu et al. (2023a); Beltagy et al. (2020); Han et al. (2023), randomized sampling Wiegreffe & Pinter (2019), or taking advantage of locality aware hashing Kitaev et al., where the query and key vectors are hashed based based on their position in the  $D$ -dimensional space, and attention is only calculated for the keys close to the query with this metric. However, these approaches contradict the goal of capturing attention for a long sequence, as it only considers limited token pairs. This is especially problematic for training.

There are works which that combine both of these approaches, where they calculate the exponential-based kernel for a sparse number of query-key pairs, and separable-based kernel for the rest of the tokens Qin et al. (2022a).

Other Algorithmic Approaches Other approaches include assuming that the Attention matrix is numerically low rank and derive a smaller matrix Wang et al., hierarchically packing the tokens Ma et al. (2021), using the Gaussian function as the Attention kernel and approximating it in linear time Chen et al. (2021) or calculating the result of the Attention and the value matrix multiplication using Fast Multipole Methods Nguyen et al. (2021). Despite their novel approaches, these methods have yet to be used in large context LLMs.

Non-transformer Models New learning architectures are also being proposed as an alternative to Transformers. Recent ones which have generated considerable interest include Mamba Gu & Dao, Retentive networks Sun et al., Hyena Poli et al. and CRATE Yu et al. (2023b). These approaches are complementary to the factored attention kernels which are the focus of this paper.

# 3 MEMORY REDUCTION

We first discuss factorization of the attention kernel, which is the computational trick enabling speedup. This enables the time and memory complexity of factorization based kernels can be reduced both in the forward and backward passes. The derivation of the analytical gradient of the attention head in this paper is a novel contribution, which allows us to reduce memory consumption. We show these results for the linear attention kernel  $f(x) = a + bx$ , and demonstrate how to achieve time and memory complexities of  $O(ND^2)$  and  $O(ND)$ . Our method can also be applied to the attention kernel of  $f(x) = a + bx + cx^2$ , which we show in Appendix A, and results in  $O(ND^3)$  time and  $O(ND)$  memory complexity.

# 3.1 SPEEDUP VIA FACTORIZATION

To show how Factorization speeds up computations, consider the MVP

$$
\left[ \begin{array}{l l l} a _ {1} b _ {1} & a _ {1} b _ {2} & a _ {1} b _ {3} \\ a _ {2} b _ {1} & a _ {2} b _ {2} & a _ {2} b _ {3} \\ a _ {3} b _ {1} & a _ {3} b _ {2} & a _ {3} b _ {3} \end{array} \right] \times \left[ \begin{array}{l} d _ {1} \\ d _ {2} \\ d _ {3} \end{array} \right] = \left[ \begin{array}{l} u _ {1} \\ u _ {2} \\ u _ {3} \end{array} \right]. \tag {4}
$$

The naive method of calculating  $u$  would be

$$
u _ {i} = a _ {i} b _ {1} d _ {1} + a _ {i} b _ {2} d _ {2} + a _ {i} b _ {3} d _ {3}, \tag {5}
$$

with a total of 9 multiplications and 6 accumulations. Applying Factorization, we have

$$
x = b _ {1} d _ {1} + b _ {2} d _ {2} + b _ {3} d _ {3}, \quad u _ {i} = a _ {i} x, \tag {6}
$$

reducing the operations to 6 multiplications and 2 accumulations. For an  $M \times M$  with the same structure, the number of operations reduces from  $O(M^2)$  to  $O(M)$ .

# 3.2 FORWARD PASS

Employing  $f(x) = a + bx$  as Attention kernel, the output matrix  $\mathbf{O}$  can be broken down to matrix multiplications in the form of Eq. 4. Specifically,

$$
\mathbf {o} _ {i, j} = \frac {\sum_ {n = 1} ^ {N} f \left(\mathbf {q} _ {i} ^ {T} \mathbf {k} _ {\mathbf {n}}\right) \mathbf {v} _ {n , j}}{\sum_ {n = 1} ^ {N} f \left(\mathbf {q} _ {i} ^ {T} \mathbf {k} _ {\mathbf {n}}\right)}, \quad \mathbf {o} _ {i, j} = \frac {\mathbf {f} _ {i , j}}{\mathbf {g} _ {i}}, \quad \mathbf {F} \in \mathbb {R} ^ {N \times D}, \mathbf {G} \in \mathbb {R} ^ {N}, \tag {7}
$$

where  $\mathbf{f}_{i,j}$  and  $\mathbf{g}_i$  are elements of  $\mathbf{F}$  and  $\mathbf{G}$ . Specifically,  $\mathbf{F}$  and  $\mathbf{G}$  are written as

$$
\mathbf {F} = \left(a + b \sum_ {m = 1} ^ {D} \left[ \begin{array}{c} \mathbf {q} _ {1, m} \mathbf {k} _ {1, m} \dots \mathbf {q} _ {1, m} \mathbf {k} _ {N, m} \\ \vdots \ddots \vdots \\ \mathbf {q} _ {N, m} \mathbf {k} _ {1, m} \dots \mathbf {q} _ {N, m} \mathbf {k} _ {N, m} \end{array} \right]\right) \mathbf {V} \tag {8}
$$

$$
\mathbf {G} = \left(a + b \sum_ {m = 1} ^ {D} \left[ \begin{array}{c} \mathbf {q} _ {1, m} \mathbf {k} _ {1, m} \dots \mathbf {q} _ {1, m} \mathbf {k} _ {N, m} \\ \vdots \ddots \vdots \\ \mathbf {q} _ {N, m} \mathbf {k} _ {1, m} \dots \mathbf {q} _ {N, m} \mathbf {k} _ {N, m} \end{array} \right]\right) \mathbb {1} \tag {9}
$$

where  $\mathbb{1}\in \mathbb{R}^N$  is a vector of all ones; i.e.,

$$
\mathbf {f} _ {i, j} = \sum_ {n = 1} ^ {N} \left(a + b \sum_ {m = 1} ^ {D} \mathbf {q} _ {i, m} \mathbf {k} _ {N, m}\right) \mathbf {v} _ {n, j}, \quad \mathbf {g} _ {i} = \sum_ {n = 1} ^ {N} \left(a + b \sum_ {m = 1} ^ {D} \mathbf {q} _ {i, m} \mathbf {k} _ {N, m}\right). \tag {10}
$$

Changing the summation orders we get

$$
\mathbf {f} _ {i, j} = a \sum_ {n = 1} ^ {N} \mathbf {v} _ {n, j} + b \sum_ {m = 1} ^ {D} \sum_ {n = 1} ^ {N} \mathbf {q} _ {i, m} \mathbf {k} _ {N, m} \mathbf {v} _ {n, j}, \quad \mathbf {g} _ {i} \quad = a \sum_ {n = 1} ^ {N} 1 + b \sum_ {m = 1} ^ {D} \sum_ {n = 1} ^ {N} \mathbf {q} _ {i, m} \mathbf {k} _ {N, m}. \tag {11}
$$

Applying Factorization we get

$$
\mathbf {f} _ {i, j} = x _ {j} ^ {(1)} + \sum_ {m = 1} ^ {D} \mathbf {q} _ {i, m} x _ {j m} ^ {(2)}, \quad \mathbf {g} _ {i} = y ^ {(1)} + \sum_ {m = 1} ^ {D} \mathbf {q} _ {i, m} y _ {m} ^ {(2)}, \tag {12}
$$

where,

$$
x _ {j} ^ {(1)} = a \sum_ {n = 1} ^ {N} \mathbf {v} _ {n, j}, \quad x _ {j m} ^ {(2)} = b \sum_ {n = 1} ^ {N} \mathbf {k} _ {N, m} \mathbf {v} _ {n, j}, \quad y ^ {(1)} = a N, \quad y _ {m} ^ {(2)} = b \sum_ {n = 1} ^ {N} \mathbf {k} _ {N, m}. \tag {13}
$$

We have thus demonstrated how to calculate the forward pass of Linear Attention with a computational complexity of  $O(ND^2)$ . However, when using a differentiable programming library such as JAX or PyTorch, all operations must be tracked in the computational graph. Consequently, all intermediate variables in Eq. 13 need to be stored, resulting in a memory consumption of  $O(ND^2)$ . See Appendix B on details for how the same memory reduction to the case of a causal mask.

# 3.3 BACKWARD PASS

By deriving the analytical gradient for linear attention, we can calculate the backward pass manually, eliminating the need for differentiable programming libraries. As a result, we no longer need to store intermediate values. However, the backward pass must also be calculated within this same complexity of  $O(ND^2)$ . Let us denote  $\mathbf{q}_i \cdot \mathbf{k}_j$  as  $s_{i,j}$ , and write the attention and output as

$$
\mathbf {o} _ {i, j} = \sum_ {n = 1} ^ {N} \mathbf {a} _ {i, n} \mathbf {v} _ {n, j}, \quad \mathbf {a} _ {i, n} = \frac {f \left(s _ {i , j}\right)}{\sum_ {m = 1} ^ {N} f \left(s _ {i , m}\right)} = \frac {f \left(s _ {i , j}\right)}{\mathbf {g} _ {i}}, \quad f (x) = a + b x \tag {14}
$$

Taking the derivative with respect to  $s_{i,l}$ , we find  $\frac{\partial\mathbf{o}_{i,j}}{\partial s_{i,l}}$  as

$$
\frac {\partial \mathbf {a} _ {i , n}}{\partial s _ {i , l}} = \left\{ \begin{array}{l} \frac {b (1 - \mathbf {a} _ {i , j})}{\sum_ {m = 1} ^ {N} f \left(s _ {i , m}\right)}, n = l \\ \frac {b (- \mathbf {a} _ {i , j})}{\sum_ {m = 1} ^ {N} f \left(s _ {i , m}\right)}, n \neq l \end{array} \right. \tag {15}
$$

$$
\frac {\partial \mathbf {o} _ {i , j}}{\partial s _ {i , l}} = \sum_ {n = 1} ^ {N} \frac {\partial \mathbf {a} _ {i , n}}{\partial s _ {i , l}} \mathbf {v} _ {n, j} = \frac {b \left(\mathbf {v} _ {l , j} - \sum_ {n = 1} ^ {N} \mathbf {a} _ {i , n} \mathbf {v} _ {n , j}\right)}{\sum_ {m = 1} ^ {N} f \left(s _ {i , m}\right)} = \frac {b}{\mathbf {g} _ {i}} \left(\mathbf {v} _ {l, j} - \mathbf {o} _ {i, j}\right). \tag {16}
$$

We now derive the partial derivative with respect to  $\mathbf{Q},\mathbf{K},\mathbf{V}$

$$
\frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {q} _ {i , r}} = \sum_ {l = 1} ^ {N} \frac {\partial s _ {i , l}}{\partial \mathbf {q} _ {i , r}} \frac {\partial \mathbf {o} _ {i , j}}{\partial s _ {i , l}} = \frac {\sum_ {l = 1} ^ {N} b \mathbf {k} _ {l , r}}{\mathbf {g} _ {i}} (\mathbf {v} _ {l, j} - \mathbf {o} _ {i, j}) \tag {17}
$$

$$
\frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {k} _ {p , r}} = \frac {\partial s _ {i , p}}{\partial \mathbf {k} _ {p , r}} \frac {\partial \mathbf {o} _ {i , j}}{\partial s _ {i , p}} = \frac {b \mathbf {q} _ {i , r}}{\mathbf {g} _ {i}} \left(\mathbf {v} _ {p, j} - \mathbf {o} _ {i, j}\right) \tag {18}
$$

$$
\frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {v} _ {p , j}} = \mathbf {a} _ {i, p} = \frac {f \left(s _ {i , p}\right)}{\mathbf {g} _ {i}}. \tag {19}
$$

During the backward pass, given the gradient of the previous layer  $\Omega$ , the gradient of the Attention head  $\nabla \Psi$  is calculated as follows

$$
\nabla_ {\mathbf {q} _ {i, r}} \boldsymbol {\Psi} = \sum_ {j = 1} ^ {D} \frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {q} _ {i , r}} \boldsymbol {\Omega} _ {i, j} = \sum_ {j = 1} ^ {D} \frac {\sum_ {l = 1} ^ {N} b \mathbf {k} _ {l , r}}{\mathbf {g} _ {i}} \left(\mathbf {v} _ {l, j} - \mathbf {o} _ {i, j}\right) \boldsymbol {\Omega} _ {i, j} \tag {20}
$$

$$
\nabla_ {\mathbf {k} _ {p, r}} \boldsymbol {\Psi} = \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {D} \frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {k} _ {p , r}} \boldsymbol {\Omega} _ {i, j} = \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {D} \frac {b \mathbf {q} _ {i , r}}{\mathbf {g} _ {i}} \left(\mathbf {v} _ {p, j} - \mathbf {o} _ {i, j}\right) \boldsymbol {\Omega} _ {i, j} \tag {21}
$$

$$
\nabla_ {\mathbf {v} _ {p, j}} \boldsymbol {\Psi} = \sum_ {i = 1} ^ {N} \frac {\partial \mathbf {o} _ {i , j}}{\partial \mathbf {v} _ {p , j}} \boldsymbol {\Omega} _ {i, j} = \sum_ {i = 1} ^ {N} \frac {f \left(s _ {i , p}\right)}{\mathbf {g} _ {i}} \boldsymbol {\Omega} _ {i, j} \tag {22}
$$

Applying Factorization, the gradients can be calculated as

$$
\nabla_ {\mathbf {q} _ {i, r}} \boldsymbol {\Psi} = \sum_ {j = 1} ^ {D} \left\{\alpha_ {r j} ^ {Q} - \beta_ {r} ^ {Q} \mathbf {o} _ {i, j} \right\} \boldsymbol {\Omega} _ {i, j}, \quad \nabla_ {\mathbf {k} _ {i, r}} \boldsymbol {\Psi} = \sum_ {j = 1} ^ {D} \left\{\alpha_ {r j} ^ {K} \mathbf {v} _ {i, j} - \beta_ {r j} ^ {K} \right\}, \tag {23}
$$

$$
\nabla_ {\mathbf {v} _ {i, j}} \boldsymbol {\Psi} = \alpha_ {j} ^ {V} + \sum_ {j = 1} ^ {D} \left\{\beta_ {r j} ^ {V} \mathbf {k} _ {i, r} \right\}. \tag {24}
$$

where the  $\alpha$  and  $\beta$  are the factorized coefficients and are defined as

$$
\alpha_ {r j} ^ {Q} = \sum_ {l = 1} ^ {N} b \mathbf {k} _ {l, r} \mathbf {v} _ {l, j}, \quad \beta_ {r} ^ {Q} = \sum_ {l = 1} ^ {N} b \mathbf {k} _ {l, r}, \tag {25}
$$

$$
\alpha_ {r j} ^ {K} = \sum_ {l = 1} ^ {N} b \mathbf {q} _ {l, r} \boldsymbol {\Omega} _ {l, j}, \quad \beta_ {r j} ^ {K} = \sum_ {l = 1} ^ {N} b \mathbf {q} _ {l, r} \mathbf {o} _ {l, j} \boldsymbol {\Omega} _ {l, j}, \tag {26}
$$

$$
\alpha_ {j} ^ {V} = \sum_ {l = 1} ^ {N} a \boldsymbol {\Omega} _ {l, j}, \quad \beta_ {r j} ^ {V} = \sum_ {l = 1} ^ {N} b, \mathbf {q} _ {l, r} \boldsymbol {\Omega} _ {l, j}. \tag {27}
$$

Put into words, gradient of the output matrix  $\mathbf{O}$  can be calculated by storing  $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},$  and  $\mathbf{G}$ ; resulting in a memory consumption of  $O(ND)$  elements. The time complexity is  $O(ND^2)$  since we perform  $O(D)$  operations for  $1\leq i\leq N$ ,  $1\leq D\leq D$  in Eq.s 68-70, which is the same time complexity of forward pass. To implement custom gradients, we wrote the forward and backward pass using CUDA. However, since the computations are MVP, and that MVP has a low memory reuse rate, the calculations are memory bound. Therefore, we had to implement optimization tricks for improved performance. Appendix C explains these tricks.

# 4 ALTERNATIVE MECHANISM TO DROPOUT

Dropout helps with training transformers by randomly dropping out cells of the Attention matrix during training, which prevents overfitting by reducing the co-adaptation of token pairs, thus improving generalization performance on unseen data. However, since the Attention matrix is not explicitly created in the approaches with linear time complexity, dropout cannot be applied. We propose the following mechanism as an alternative to achieve the same end. Let us denote  $\mathbf{O}$  and  $\tilde{\mathbf{O}}$  as the output with and without dropout applied. Then, we approximate  $\mathbf{O}$  as

$$
\mathbf {o} _ {i j} \approx (1 - p) \tilde {\mathbf {o}} _ {i j} (1 + \psi), \quad \psi \sim \mathcal {N} (0, p), \quad p: \text {d r o p o u t r a t e}. \tag {28}
$$

During the training, we start with some dropout rate  $p$ , and gradually decrease it to 0. The intuition behind this mechanism stems from applying the law of large numbers and the central limit theorem when  $\mathbf{Q}$ ,  $\mathbf{K}$ , and  $\mathbf{V}$  follow a standard Gaussian distribution. For the Softmax-based Attention, dropout is defined as replacing cells of the  $\mathbf{Q}\mathbf{K}^T$  Attention matrix with  $-\infty$  with probability  $p$ , and then taking the Softmax to derive the Attention matrix. The Attention matrix is then multiplied by  $\mathbf{V}$  to calculate  $\mathbf{O}$ . Specifically,  $\mathbf{O}$  can be written as

$$
\mathbf {o} _ {i j} = \frac {\sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {n} / \sqrt {D}\right) \mathbf {v} _ {n j} - \sum_ {n \in \{\text {d r o p} \}} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {n} / \sqrt {D}\right) \mathbf {v} _ {n j}}{\sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {n} / \sqrt {D}\right) - \sum_ {n \in \{\text {d r o p} \}} \exp \left(\mathbf {q} _ {i} . \mathbf {k} _ {n} / \sqrt {D}\right)} = \frac {\eta_ {i j} - \eta_ {i j} ^ {\text {d r o p}}}{\gamma_ {i} - \gamma_ {i} ^ {\text {d r o p}}}; \tag {29}
$$

$$
\eta_ {i j} = \sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i}. \mathbf {k} _ {\mathbf {n}} / \sqrt {D}\right) \mathbf {v} _ {\mathbf {n j}}, \quad \eta_ {i j} ^ {d r o p} = \sum_ {n \in \{\text {d r o p} \}} \exp \left(\mathbf {q} _ {i}. \mathbf {k} _ {\mathbf {n}} / \sqrt {D}\right) \mathbf {v} _ {\mathbf {n j}}, \tag {30}
$$

$$
\gamma_ {i} = \sum_ {n = 1} ^ {N} \exp \left(\mathbf {q} _ {i}. \mathbf {k} _ {n} / \sqrt {D}\right), \quad \gamma_ {i} ^ {\text {d r o p}} = \sum_ {n \in \{\text {d r o p} \}} \exp \left(\mathbf {q} _ {i}. \mathbf {k} _ {n} / \sqrt {D}\right). \tag {31}
$$

As an experiment, let us assume that  $\mathbf{Q}$ ,  $\mathbf{K}$  and  $\mathbf{V}$  follow a standard Gaussian distribution, that is

$$
\mathbf {q} _ {i j} \sim \mathcal {N} (0, 1), \quad \mathbf {k} _ {i j} \sim \mathcal {N} (0, 1), \quad \mathbf {v} _ {i j} \sim \mathcal {N} (0, 1). \tag {32}
$$

Lemma 4.1. (Multiplication of Gaussian Variables) Let  $X$  and  $Y$  be two independent random variables following standard Gaussian distribution, i.e.,  $X \sim \mathcal{N}(0,1)$ ,  $Y \sim \mathcal{N}(0,1)$ . Then,  $Z = XY$  follows a difference of two Chi-Square distribution with degree of freedom 1.

Proof.

$$
X Y = \frac {1}{4} \left(\left(X + Y\right) ^ {2} - \left(X - Y\right) ^ {2}\right); \quad \frac {X + Y}{\sqrt {2}}, \frac {X - Y}{\sqrt {2}} \sim \mathcal {N} (0, 1); \tag {33}
$$

$$
Z = \frac {1}{4} \left(\left(X + Y\right) ^ {2} - (X - Y) ^ {2}\right) = \psi - \phi ; \quad \psi , \phi \sim \chi_ {1} ^ {2}. \tag {34}
$$

Furthermore, using the results in Smith et al. (2011), we can write

$$
\mathbb {E} [ Z ] = 0, \quad \mathbb {E} [ Z ^ {2} ] = \frac {1}{2}, \quad Z \sim \Gamma (0, \frac {1}{2}), \tag {35}
$$

where  $\Gamma (\mu ,\sigma^2)$  is some symmetric distribution with mean  $\mu$  and variance of  $\sigma^2$

Using Lemma 4.1, we write  $\mathbf{q}_i.\mathbf{k}_n$  as

$$
\mathbf {q} _ {i} \cdot \mathbf {k} _ {n} = \sum_ {j = 1} ^ {D} \mathbf {q} _ {i j} \mathbf {k} _ {n j} = \sum_ {j = 1} ^ {D} \omega_ {j}, \quad \omega_ {j} \sim \Gamma \left(0, \frac {1}{2}\right). \tag {36}
$$

Since  $\omega_{i}$  are i.i.d and follows a symmetric distribution with a bounded variance, we can apply the Central Limit Theorem for sufficiently large  $D$ , that is

$$
\frac {\mathbf {q} _ {i} \cdot \mathbf {k} _ {n}}{\sqrt {D}} = \frac {1}{\sqrt {D}} \sum_ {j = 1} ^ {D} \omega_ {j} \sim \mathcal {N} (0, \frac {1}{2}). \tag {37}
$$

In practice  $D$  is 32 for small transformers, and can reach up to 512. Taking the exponential, we end up with the log-Normal distribution with mean  $\exp \left( \frac{1}{4} \right)$  and variance of 1, that is

$$
\exp \left(\frac {\mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}}\right) \sim \operatorname {L o g - N o r m a l} \left(0, \frac {1}{2}\right). \tag {38}
$$

Lemma 4.2. (Kolmogorov's  $L_{1}$  Strong Law of Large Numbers) Let  $X_{i}|_{i = 1}^{N}$  be a sequence of i.i.d. random variables with a finite expected value  $\mu$ . Then,

$$
\lim  _ {N \rightarrow \infty} \sum_ {i = 1} ^ {N} X _ {i} \rightarrow N \mu \quad \text {a l m o s t s u r e l y}. \tag {39}
$$

In practice, the number of tokens ranges from 512 for small benchmarks, and can reach up to  $10^{7}$ . Therefore, we can use Lemma 4.2, to approximate  $\gamma_{i},\gamma_{i}^{drop}$  in Eq. 31 as

$$
\gamma_ {i} = \sum_ {n = 1} ^ {N} \exp \left(\frac {\mathbf {q} _ {i} \cdot \mathbf {k} _ {n}}{\sqrt {D}}\right) \approx N \exp \left(\frac {1}{4}\right), \quad \gamma_ {i} ^ {d r o p} = \sum_ {n \in \{\mathrm {d r o p} \}} \exp \left(\frac {\mathbf {q} _ {i} \cdot \mathbf {k} _ {n}}{\sqrt {D}}\right) \approx N p \exp \left(\frac {1}{4}\right), \tag {40}
$$

where  $p$  is the dropout rate. Since  $\mathbf{V} \sim \mathcal{N}(0, I)$ , and that sum of Gaussian variables follows a Gaussian distribution, we can use Lemma 4.2 to approximate  $\eta_{ij}$ ,  $\eta_{ij}^{drop}$  in Eq. 31 as

$$
\psi_ {i n j} \sim \mathcal {N} (0, \exp (\frac {2 \mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}})) \tag {41}
$$

$$
\eta_ {i j} = \sum_ {n = 1} ^ {N} \exp \left(\frac {\mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}}\right) \mathbf {v} _ {\mathbf {n j}} = \sum_ {n = 1} ^ {N} \psi_ {i n j} \sim \mathcal {N} \left(0, \sum_ {n = 1} ^ {N} \exp \left(\frac {2 \mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}}\right)\right) \approx \mathcal {N} (0, N \exp (1)) \tag {42}
$$

$$
\eta_ {i j} ^ {\text {d r o p}} = \sum_ {n \in \{\text {d r o p} \}} \exp \left(\frac {\mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}}\right) \mathbf {v} _ {\mathbf {n j}} = \sum_ {n \in \{\text {d r o p} \}} \psi_ {i n j} \sim \mathcal {N} \left(0, \sum_ {n \in \{\text {d r o p} \}} \exp \left(\frac {2 \mathbf {q} _ {i} . \mathbf {k} _ {n}}{\sqrt {D}}\right)\right). \tag {43}
$$

With these results acquired, we will now write the output with dropout  $\mathbf{o}_{ij}$  and without dropout  $\tilde{\mathbf{o}}_{ij}$  as

$$
\mathbf {o} _ {i j} = \frac {\eta_ {i j} - \eta_ {i j} ^ {\text {d r o p}}}{\gamma_ {i} - \gamma_ {i} ^ {\text {d r o p}}}, \quad \tilde {\mathbf {o}} _ {i j} = \frac {\eta_ {i j}}{\gamma_ {i}}. \tag {44}
$$

Given  $\tilde{\mathbf{o}}_{ij}$ , we write  $\mathbf{o}_{ij}$  as

$$
\mathbf {o} _ {i j} = \frac {\gamma_ {i} \tilde {\mathbf {o}} _ {i j}}{\gamma_ {i} - \gamma_ {i} ^ {d r o p}} \frac {\eta_ {i j} - \eta_ {i j} ^ {d r o p}}{\eta_ {i j}} = \frac {\gamma_ {i} \tilde {\mathbf {o}} _ {i j}}{\gamma_ {i} - \gamma_ {i} ^ {d r o p}} (1 - \frac {\eta_ {i j} ^ {d r o p}}{\eta_ {i j}}) \approx (1 - p) \tilde {\mathbf {o}} _ {i j} (1 + \psi); \tag {45}
$$

$$
\psi \sim \mathcal {N} (0, p). \tag {46}
$$

This mechanism can be viewed as adding noise to improve learning by preventing the model from getting stuck in local minima, achieving a result similar to dropout. The concept of adding noise to aid optimization is not new and has been widely employed in various areas. For instance, noise is introduced in environment simulations for reinforcement learning Pinto et al. (2017); Peng et al. (2018), in diffusion probabilistic models Sohl-Dickstein et al. (2015), and in deep networks in general, where adding noise during gradient descent has been suggested to be beneficial Neelakantan et al. (2015); Hu & Gerber (2024).

# 5 EXPERIMENTS

The main contributions of the paper are reducing the memory consumption of Linear Attention and an alternative mechanism to dropout. The expressibility of Linear Attention has been studied in previous works Zhang et al. (2024a); Qin et al. (2022a); Wang et al.; Katharopoulos et al. (2020); Kasai et al. (2021); Qin et al. (2022b); Tay et al.. To confirm the scaling of time and memory of our method, we perform the following experiments: i) We measure the time and memory for a single forward pass of an attention layer in Section 5.1, and ii) for training an LLM in Section 5.2; iii) To assess our alternative mechanism to dropout, we perform an ablation study using two benchmarks in Section 5.3. Our results are for the attention kernel of  $f(x) = a + bx$  with  $a, b = 1$ , but our implementation supports any choice of  $a, b$ . We use an Nvidia RTX A6000 with 48 GB of memory for all of our experiments, and the implementation details can be found in Appendix D.

# 5.1 TIME AND MEMORY SCALING

We refer to our implementation of Linear Attention as "Ours", and the Kernel Separation approach as "Kern. Sep". Figure 1 shows the time and memory for a single forward pass for batch size of 32,  $H = 16$ ,  $D = 32$ , and  $1e^3 \leq N \leq 3e^5$ . The results are averaged over 1000 runs, and we've included FlashAttention-2 Dao (2023) as a point of comparison with the quadratic conventional algorithm. The plots are in log-log scale, and the slopes show the order of dependency. Both the Kernel Separation and our approach have time and memory scaling of  $O(N)$ , whereas FlashAttention-2 scales with  $O(N^2)$ . However, it can be seen that our implementation has  $D \times$  less memory consumption compared to the Kernel Separation.

![](images/03ab671aa44c733603de9bad44534b1ca32c88610594e76ec87c856f588b8610.jpg)  
Figure 1: Time and memory scaling on A6000 based on number of tokens. The "X" indicate OOM.

![](images/bff5016c2eef224cfaedc3b083a6d46b3f054caa9c802af6b920278640ea0a0b.jpg)

Figure 2 shows the time and memory under the same setting but with  $N = 2000$  and  $32 \leq D \leq 5000$ . Both the Kernel Separation and our approach have a time scaling of  $O(D^2)$ , and FlashAttention-2 has a scaling of  $O(D)$ . As for the memory, Kernel Separation scales with  $O(D^2)$ , whereas our approach and FlashAttention-2 scales with  $O(D)$ . In conclusion, our implementation has a time and memory

![](images/11d6d129a41f1d8c7ad116cf2dc8262526c09bbd7d120d08a5e51f1d490f0c3e.jpg)  
Figure 2: Time and memory scaling on A6000 based on dimension per head. The "X" indicate OOM.

![](images/4f5c1564afb7b7fca1d74d14bcca095717687a0ce8160da369272ec7737974cb.jpg)

scaling linear with  $N$  and  $D$ , whereas Kernel Separation has a memory scaling quadratic with  $D$ . Our implementation can practically scale to larger contexts.

# 5.2 TRAINING AN LLM

To demonstrate that our claims hold in practice, we train an LLM on a small dataset. We chose Pythia-14M Biderman et al. (2023) as our LLM, with a token length of  $N = 4096$ , and Tiny Stories Eldan & Li (2023) as our dataset. We train the model using our implementation of Linear Attention, and FlashAttention-2. The hyperparameters used are the same for our method and FlashAttention-2, and more implementation details can be found in Appendix D. Figure 3 shows the learning curves for training loss. Our method has faster convergence, which can be attributed to its linear time scaling of  $O(N)$ , compared to FlashAttention-2's quadratic scaling of  $O(N^2)$ . In conclusion, by reducing the

memory cost, our method makes the use of Linear Attention in LLMs and large transformers feasible while maintaining the  $O(N)$  time scaling.

# 5.3 DROPOUT ABLATION STUDY

To test the effectiveness of our alternative dropout mechanism introduced in Section 4, we perform an ablation study using the Linear Attention on the Long Range Arena (LRA) Tay et al. and Tiny Stories Eldan & Li (2023) datasets. The LRA benchmark is a suite of tasks designed to evaluate the performance of Transformer models on sequences of extreme length. It includes a diverse set of challenges such as text classification, document retrieval, and mathematical reasoning. The Tiny Stories benchmark is a dataset designed to evaluate language models' ability to generate coherent, child-friendly short stories with simple vocabulary and structure. As shown in Table 1, adding noise to mimic dropout has helped to improve expressivity of Lienar Attention in all cases. We've included the results for Softmax-based attention (FlashAttention-2) with 0.2 dropout as a point of reference.

![](images/0577da53c92995c80250806cd7318154c2153bf6bc3e9c3701485a8a7a4184c5.jpg)  
Figure 3 & Table 1: Figure 3 (left) shows the learning curve of training the Pythia-14M LLM using our method and FlashAttetion-2. Table 1 (right) presents an ablation study comparing our proposed mechanism alternative to dropout for Linear Attention, with Softmax-based attention results included as a benchmark. Our mechanism improves the score in all cases. The score in LRA is accuracy%, and cross-entropy in Tiny Stories.

<table><tr><td colspan="2"></td><td>Softmax</td><td>Linear</td><td>Linear (alt. drop.)</td></tr><tr><td rowspan="5">LRA</td><td>ListOps</td><td>38.37</td><td>28.55</td><td>42.76</td></tr><tr><td>Text</td><td>61.95</td><td>61.55</td><td>63.25</td></tr><tr><td>Retrieval</td><td>80.69</td><td>65.38</td><td>78.21</td></tr><tr><td>Image</td><td>40.57</td><td>38.47</td><td>42.76</td></tr><tr><td>Pathfinder</td><td>65.26</td><td>50.04</td><td>66.67</td></tr><tr><td></td><td>Tiny Stories</td><td>3.59</td><td>3.74</td><td>3.66</td></tr></table>

# 6 CONCLUSION

Transformers use a Softmax-based mechanism known as Attention, which has a quadratic scaling with number of tokens taken in context. Many approaches have proposed subquadratic mechanisms for both training and inference. This problem is particularly important since the current applications are moving towards higher number of tokens, and training can only be done in large datacenter scale systems. The search for a linearly scaling attention mechanism has become an important and active area of research. One of the promising approaches is Linear Attention (Kernel Separation). However, current implementations had very high memory consumption of  $O(ND^2)$ , where  $N$  is the number of tokens and  $D$  the model dimension per head. This memory consumption makes kernel separation impractical since even small transformers would require a high amount of memory per batch. We present a solution to reduce the memory scaling of Linear Attention to  $O(ND)$ . In summary, we derive the analytical format of the attention layer gradient and calculate it in linear time. By implementing this approach instead of relying on a differentiable programming library, we eliminate the need to store the variables for the backward pass, and bring down the memory consumption to  $O(ND)$ . We provide the details of how to implement the forward and backward pass using Factorization, and confirm the time and memory scaling of our implementation in context of an attention layer and training an LLM. Furthermore, since Linear Attention does not directly compute the attention matrix, dropout cannot be applied. Therefore, we have also introduced an alternative mechanism to dropout, to mimic its effect, and confirmed our mechanism's effectiveness through an ablation study.

# REFERENCES

Lightning AI. Litgpt. https://github.com/Lightning-AI/litgpt, 2023.  
Kunal Banerjee, Rishi Raj Gupta, Karthik Vyas, Biswajit Mishra, et al. Exploring alternatives to softmax function. arXiv preprint arXiv:2011.11538, 2020.  
Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.  
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397-2430. PMLR, 2023.  
Yifan Chen, Qi Zeng, Heng Ji, and Yun Yang. Skyformer: Remodel self-attention with gaussian kernel and nystr\`" om method. Advances in Neural Information Processing Systems, 34:2122-2135, 2021.  
Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.  
Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.  
Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=mZn2Xyh9Ec.  
Ronen Eldan and Yuanzhi Li. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759, 2023.  
Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. URL http://arxiv.org/abs/2312.00752.  
Dongchen Han, Xuran Pan, Yizeng Han, Shiji Song, and Gao Huang. Flatten transformer: Vision transformer using focused linear attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5961-5971, 2023.  
Annie Hu and Mathieu Gerber. Gradient descent for noisy optimization. arXiv preprint arXiv:2405.06539, 2024.  
Jungo Kasai, Hao Peng, Yizhe Zhang, Dani Yogatama, Gabriel Ilharco, Nikolaos Pappas, Yi Mao, Weizhu Chen, and Noah A Smith. Finetuning pretrained transformers into rnns. arXiv preprint arXiv:2103.13076, 2021.  
Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 5156-5165. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/katharopoulos20a.html.  
Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In International Conference on Algorithmic Learning Theory, pp. 597-619. PMLR, 2023.  
Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The Efficient Transformer. URL http://arxiv.org/abs/2001.04451.  
Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context, November 2023.

Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention. Advances in Neural Information Processing Systems, 34:2441-2453, 2021.  
Arvind Neelakantan, Luke Vilnis, Quoc V Le, Ilya Sutskever, Lukasz Kaiser, Karol Kurach, and James Martens. Adding gradient noise improves learning for very deep networks. arXiv preprint arXiv:1511.06807, 2015.  
Tan Nguyen, Vai Suliafu, Stanley Osher, Long Chen, and Bao Wang. FMMformer: Efficient and flexible transformer via decomposed near-field and far-field attention. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 29449-29463. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/f621585df244e9596dc70a39b579efb1-Paper.pdf.  
Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE international conference on robotics and automation (ICRA), pp. 3803-3810. IEEE, 2018.  
Lerrel Pinto, James Davidson, Rahul Sukthankar, and Abhinav Gupta. Robust adversarial reinforcement learning. In International Conference on Machine Learning, pp. 2817-2826. PMLR, 2017.  
Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena Hierarchy: Towards Larger Convolutional Language Models. URL http://arxiv.org/abs/2302.10866.  
Zhen Qin, Xiaodong Han, Weixuan Sun, Dongxu Li, Lingpeng Kong, Nick Barnes, and Yiran Zhong. The devil in linear transformer. October 2022a. URL http://arxiv.org/abs/2210.10340.  
Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. arXiv preprint arXiv:2202.08791, 2022b.  
J.O. Smith, Stanford University. Center for Computer Research in Music, Acoustics, and Stanford University. Department of Music. Spectral Audio Signal Processing. W3K, 2011. ISBN 9780974560731. URL https://ccrma.stanford.edu/~jos/sasp/Product_Two_Gaussian_PDFs.html.  
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256-2265. PMLR, 2015.  
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. ISSN 0925-2312. doi: https://doi.org/10.1016/j.neucom.2023.127063. URL https://www.sciencedirect.com/science/article/pii/S0925231223011864.  
Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive Network: A Successor to Transformer for Large Language Models. URL http://arxiv.org/abs/2307.08621.  
Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long Range Arena: A Benchmark for Efficient Transformers. URL http://arxiv.org/abs/2011.04006.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. arXiv [cs.CL], June 2017. URL http://arxiv.org/abs/1706.03762.  
Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-Attention with Linear Complexity. URL http://arxiv.org/abs/2006.04768.

Sarah Wiegrefe and Yuval Pinter. Attention is not not explanation. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 11-20, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1002. URL https://aclanthology.org/D19-1002.  
Yaodong Yu, Sam Buchanan, Druv Pai, Tianzhe Chu, Ziyang Wu, Shengbang Tong, Hao Bai, Yuexiang Zhai, Benjamin D. Haeffele, and Yi Ma. White-box transformers via sparse rate reduction: Compression is all there is?, 2023a.  
Yaodong Yu, Tianzhe Chu, Shengbang Tong, Ziyang Wu, Druv Pai, Sam Buchanan, and Yi Ma. Emergence of segmentation with minimalistic white-box transformers. arXiv [cs.CV], August 2023b. URL http://arxiv.org/abs/2308.16271.  
Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontonon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big Bird: Transformers for Longer Sequences. URL http://arxiv.org/abs/2007.14062.  
Michael Zhang, Kush Bhatia, Hermann Kumbong, and Christopher Ré. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. arXiv preprint arXiv:2402.04347, 2024a.  
Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source small language model, 2024b.  
Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. URL https://arxiv.org/abs/2012.07436v3.
