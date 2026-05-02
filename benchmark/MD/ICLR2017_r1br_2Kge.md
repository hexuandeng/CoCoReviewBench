# SHORT AND DEEP: SKETCHING AND NEURAL NETWORKS

Amit Daniely, Nevena Lazic, Yoram Singer, Kunal Talwar*

Google Brain

# ABSTRACT

Data-independent methods for dimensionality reduction such as random projections, sketches, and feature hashing have become increasingly popular in recent years. These methods often seek to reduce dimensionality while preserving the hypothesis class, resulting in inherent lower bounds on the size of projected data. For example, preserving linear separability requires  $\Omega(1/\gamma^2)$  dimensions, where  $\gamma$  is the margin, and in the case of polynomial functions, the number of required dimensions has an exponential dependence on the polynomial degree. Despite these limitations, we show that the dimensionality can be reduced further while maintaining performance guarantees, using improper learning with a slightly larger hypothesis class. In particular, we show that any sparse polynomial function of a sparse binary vector can be computed from a compact sketch by a single-layer neural network, where the sketch size has a logarithmic dependence on the polynomial degree. A practical consequence is that networks trained on sketched data are compact, and therefore suitable for settings with memory and power constraints. We empirically show that our approach leads to networks with fewer parameters than related methods such as feature hashing, at equal or better performance.

# 1 INTRODUCTION

In many supervised learning problems, input data are high-dimensional and sparse. The high dimensionality may be inherent in the domain, such as a large vocabulary in a language model, or the result of creating hybrid conjunction features. This setting poses known statistical and computational challenges for standard supervised learning techniques, as high-dimensional inputs lead to models with a very large number of parameters.

An increasingly popular approach to reducing model size is to map inputs to a lower-dimensional space in a data-independent manner, using methods such as random projections, sketches, and hashing. These mappings typically attempt to preserve the hypothesis class, leading to inherent theoretical limitations on size. For example, for linearly separable unit vectors with margin  $\gamma$ , it can be shown that at least  $\Omega(1/\gamma^2)$  dimensions are needed to preserve linear separability, even if one can use arbitrary input embeddings (see Section D). It would therefore appear that data dimensionality cannot be reduced beyond this bound.

In this work, we show that using a slightly larger hypothesis class when decoding projections (improper learning) allows us to further reduce dimensionality while maintaining theoretical guarantees. In particular, we show that any sparse polynomial function of a sparse binary vector can be computed from a very compact sketch by a single-layer neural network. The hidden layer allows us to "decode" inputs from representations that are smaller than in existing work. In the simplest case, we show that for linearly separable  $k$ -sparse  $d$ -dimensional inputs, one can create a  $O(k\log \frac{d}{\delta})$ -dimensional sketch of the inputs and guarantee that a single-layer neural network can correctly classify  $1 - \delta$  fraction of the sketched data. In the case of polynomial functions, the required sketch size has a logarithmic dependence on the polynomial degree.

For binary  $k$ -sparse input vectors, we show that it suffices to have a simple feed-forward network with nonlinearity implemented via the commonly used rectified linear unit (Relu). We extend our results to real-valued data that is close to being  $k$ -sparse, using less conventional min and median nonlinearities. Furthermore, we show that data can be mapped using sparse sketching matrices.

Thus, our sketches are efficient to compute and do not increase the number of non-zero input values by much, in contrast to standard dense Gaussian projections.

We empirically evaluate our sketches on real and synthetic datasets. Our approach leads to more compact neural networks than existing methods such as feature hashing and Gaussian random projections, at competitive or better performance. This makes our sketches appealing for deployment in settings with strict memory and power constraints, such as mobile and embedded devices.

# 2 PREVIOUS WORK

To put our work in context, we next summarize some lines of research related to this work.

Random projections and sketching. Random Gaussian projections are by now a standard tool for dimensionality reduction. For general vectors, the Johnson-Lindenstrauss (1984) Lemma implies that a random Gaussian projection into  $O(\log(1/\delta)/\varepsilon^2)$  dimensions preserves the inner product between a pair of unit vectors up to an additive factor  $\varepsilon$ , with probability  $1 - \delta$ . A long line of work has sought sparser projection matrices with similar guarantees; see (Achlioptas, 2003; Ailon & Chazelle, 2009; Matousek, 2008; Dasgupta et al., 2010; Braverman et al., 2010; Kane & Nelson, 2014; Clarkson & Woodruff, 2013). Research in streaming and sketching algorithms has addressed related questions. Alon et al. (1999) showed a simple hashing-based algorithm for unbiased estimators for the Euclidean norm in the streaming setting. Charikar et al. (2004) showed an algorithm for the heavy-hitters problem based on the count sketch. Most relevant to our works is the count-min sketch of Cormode and Muthukrishnan (2005a; 2005b).

Projections in learning. Random projections have been used in machine learning at least since the work of Arriaga and Vempala (2006). For fast estimation of a certain class of kernel functions, sampling has been proposed as a dimensionality reduction technique in (Kontorovich, 2007) and (Rahimi & Recht, 2007). Shi et al. (2009) propose using a count-min sketch to reduce dimensionality while approximately preserving inner products for sparse vectors. Weinberger et al. (2009) use the count-sketch to get an unbiased estimator for the inner product of sparse vectors and prove strong concentration bounds. Ganchev and Dredze (2008) empirically show that hashing is effective in reducing model size without significantly impacting performance. Hashing has also been used in Vowpal Wabbit (Langford et al., 2007). Talukdar and Cohen (2014) use the count-min sketch in graph-based semi-supervised learning. Pham and Pagh (2013) showed that a count sketch of a tensor power of a vector could be quickly computed without explicitly computing the tensor power, and applied it to fast sketching for polynomial kernels.

Compressive sensing. Our work is also related to compressive sensing. For  $k$ -sparse vectors, results in this area, e.g. (Donoho, 2006; Candés & Tao, 2006), imply that a  $k$ -sparse vector  $\mathbf{x} \in \mathbb{R}^d$  can be reconstructed w.h.p. from a projection of dimension  $O(k \ln \frac{d}{k})$ . However, to our knowledge, no provable decoding algorithms are implementable by a low-depth neural network. Recent work by Mousavi et al. (2015) empirically explores using a deep network for decoding in compressive sensing and also considers learnt non-linear encodings to adapt to the distribution of inputs.

Parameter reduction in deep learning. Our work can be viewed as a method for reducing the number of parameters in neural networks. Neural networks have become ubiquitous in many machine learning applications, including speech recognition, computer vision, and language processing tasks(see (Hinton et al., 2012; Krizhevsky et al., 2012; Sermanet et al., 2013; Vinyals et al., 2014) for a few notable examples). These successes have in part been enabled by recent advances in scaling up deep networks, leading to models with millions of parameters (Dean et al., 2012; Krizhevsky et al., 2012). However, a drawback of such large models is that they are very slow to train, and difficult to deploy on mobile and embedded devices with memory and power constraints. Denil et al. (2013) demonstrate significant redundancies in the parameterization of several deep learning architectures, and they propose training low-rank decompositions of weight matrices. Cheng et al. (2015) impose circulant matrix structure on fully connected layers. Ba and Caruana (2014) train shallow networks to predict the log-outputs of a large deep network, and Hinton et al. (2015) train a small network to match smoothed predictions of a complex deep network or an ensemble of such models. Collins and Kohli (2014) encourage zero-weight connections using sparsity-inducing priors, while others such as LeCun et al. (1989); Hassibi et al. (1993); Han et al. (2015) use techniques for pruning weights. HashedNets (Chen et al., 2015) enforce parameter sharing between random groups

of network parameters. In contrast to these methods, sketching only involves applying a sparse, linear projection to the inputs, and does not require a specialized learning procedure or network architecture.

# 3 SKETCHING

For simplicity, we first present our results for sparse binary vectors, and extend the discussion to real-valued vectors to Appendix A. Let  $B_{d,k} = \{\mathbf{x} \in \{0,1\}^d : \| \mathbf{x} \|_0 \leq k\}$  be the set of  $k$ -sparse  $d$ -dimensional binary vectors. We sketch such vectors using a family of randomized sketching algorithms based on the count-min sketch, as described next.

Given a parameter  $m$  and a hash function  $h:[d]\to [m]$ , the sketch  $S_{h}(\mathbf{x})$  of a vector  $\mathbf{x}\in B_{d,k}$  is a binary vector  $\mathbf{y}$  where each bit of  $\mathbf{y}$  is the OR of bits of  $\mathbf{x}$  that hash to it:

$$
y _ {l} = \bigvee_ {i: h (i) = l} x _ {i}.
$$

We map data using a concatenation of several such sketches. Given an ordered set of hash functions  $h_1, \ldots, h_t \stackrel{\text{def}}{=} h_{1:t}$ , the sketch  $S_{h_{1:t}}(\mathbf{x})$  is defined as a  $m \times t$  matrix  $Y$ , where the  $j^{\text{th}}$  column corresponds to the sketch  $S_{h_j}(\mathbf{x})$ . We define the following procedure for decoding the  $i^{\text{th}}$  bit of the input  $\mathbf{x}$  from its sketch  $Y$ :

$$
D _ {h _ {1: t}} ^ {\mathrm {A N D}} (Y, i) \stackrel {{\text {d e f}}} {{=}} \bigwedge_ {j \in [ t ]} Y _ {h _ {j} (i) j}. \tag {1}
$$

Thus, the decoded bit  $i$  is simply the AND of the  $t$  bits of  $Y$  that index  $i$  hashes to in the sketch.

The following theorem summarizes an important property of these sketches. As a reminder, a set of hash functions  $h_{1:t}$  from  $[d]$  to  $[m]$  is pairwise independent if for all  $i \neq j \in [d]$  and  $a, b \in [m]$ ,  $\operatorname{Pr}[h(i) = a \land h(j) = b] = m^{-2}$ .<sup>1</sup>

Theorem 3.1. Let  $\mathbf{x} \in B_{d,k}$  and for  $j \in [t]$  let  $h_j : [d] \to [m]$  be drawn uniformly and independently from a pairwise independent distribution with  $m = ek$ . Then for any  $i$ ,

$$
\operatorname * {P r} [ D _ {h _ {1: t}} ^ {\mathrm {A N D}} (S _ {h _ {1: t}} (\mathbf {x}), i) \neq x _ {i} ] \leq e ^ {- t}.
$$

Proof. Fix a vector  $\mathbf{x} \in B_{d,k}$ . Let  $\mathcal{E}_h(i) \stackrel{\text{def}}{=} \{i' \neq i : h(i') = h(i)\}$  denote the collision set of  $i$  for a particular hash function  $h$ . Decoding will fail if  $x_i = 0$  and for each of the  $t$  hash functions, the collision set of  $i$  for contains an index of a non-zero bit of  $\mathbf{x}$ . For a particular  $h$ , the probability of this event is:

$$
P r \left[ \bigvee_ {i ^ {\prime} \in \mathcal {E} _ {h} (i)} x _ {i ^ {\prime}} = 1 \right] \leq \sum_ {i ^ {\prime}: x _ {i ^ {\prime}} = 1} \operatorname * {P r} [ h (i ^ {\prime}) = h (i) ] \leq \frac {k}{m} = \frac {1}{e},
$$

where the second inequality follows since the sum is over at most  $k$  terms, and each term is  $m^{-1}$  by pairwise independence. Thus  $\operatorname*{Pr}[Y_{h_j(i)} \neq x_i] \leq e^{-1}$  for any  $j \in [t]$ . Since hash functions  $h_j$  are drawn independently, and decoding can fail only if all  $t$  hash functions fail, it follows that  $\operatorname*{Pr}[D_{h_{1:t}}^{\mathrm{AND}}(S_{h_{1:t}}(\mathbf{x}), i) \neq x_i] \leq e^{-t}$ .

Let  $\mathcal{H}_{d,s}$  denote the set  $\mathcal{H}_{d,s} = \{\mathbf{w} \in \mathbb{R}^d : \| \mathbf{w} \|_0 \leq s\}$ . We have the following corollary:

Corollary 3.2. Let  $\mathbf{w} \in \mathcal{H}_{d,s}$  and  $\mathbf{x} \in B_{d,k}$ . For  $t = \log(s / \delta)$ , and  $m = ek$ , if  $h_1, \ldots, h_t$  are drawn uniformly and independently from a pairwise independent distribution, then

$$
\Pr \left[ \sum_ {i} w _ {i} D _ {h _ {1: t}} ^ {\mathrm {A N D}} \left(S _ {h _ {1: t}} (\mathbf {x}), i\right) \neq \mathbf {w} ^ {\top} \mathbf {x} \right] \leq \delta .
$$

# 4 SPARSE LINEAR FUNCTIONS

![](images/6fdc663c5f64863f3081662034c2029e8a0c34e59e30774074ae313e06f4f8fc.jpg)  
Figure 1: Neural-network sketching: sparse vector  $\mathbf{x}$  maps to sketch using  $t = 3$  hashes &  $m = 8$ ; shaded squares designate 1's; sketching step is random; sketch then used as input to single-layer net:  $\mathbf{w}^{\top}\mathbf{x}$ ; nodes labelled "24" & "29" correspond to decoding of  $x_{24}$  &  $x_{29}$  and shown with nonzero incoming edges.

Let  $\mathbf{w} \in \mathcal{H}_{d,s}$ ,  $\mathbf{x} \in B_{d,k}$ , and  $Y = S_{h_{1:t}}(\mathbf{x})$  for  $m, t$  satisfying the conditions of Corollary 3.2. We will now argue that there exists a one-layer neural network that takes  $Y$  as input and outputs  $\mathbf{w}^\top \mathbf{x}$  with high probability (over the randomness of the sketching process).

Let  $\mathcal{N}_n(f)$  denote the family of feedforward neural networks with one hidden layer containing  $n$  nodes, nonlinearity  $f$  applied at each hidden unit, and a linear function at the output layer. We can construct a network in  $\mathcal{N}_s(\mathrm{Relu})$  such that each hidden unit implements  $D_{h_{1:t}}^{\mathrm{AND}}(Y,i)$  (i.e. decodes bit  $x_{i}$  from the sketch) for each index  $i$  in the support of  $\mathbf{w}$ . We can

then set the output weights of the network to the corresponding non-zero weights  $w_{i}$  to get  $\mathbf{w}^{\top}\mathbf{x}$ .

It remains to show that a hidden unit can implement  $D_{h_{1:t}}^{\mathrm{AND}}(Y,i)$ . Indeed, the AND of  $t$  bits can be implemented using nearly any non-linearity. With  $\mathrm{Relu}(a) = \max \{0,a\}$ , we can construct the activation for bit  $x_{i}$ ,  $a_{i} = \sum_{(l,j)}V_{lj}Y_{lj} + B_{i}$ , by setting the appropriate  $t$  weights in  $V_{lj}$  to 1, setting remaining weights to 0, and setting the bias  $B_{i}$  to  $1 - t$ . Using Corollary 3.2, we have the following theorem.

Theorem 4.1. For every  $\mathbf{w} \in \mathcal{H}_{d,s}$  there exists a set of weights for a network  $N \in \mathcal{N}_s(\mathrm{Relu})$  such that for each  $\mathbf{x} \in B_{d,k}$ ,

$$
P r _ {h _ {1: t}} \left[ N \left(S _ {h _ {1: t}} (\mathbf {x})\right) = \mathbf {w} ^ {\top} \mathbf {x} \right] \geq 1 - \delta ,
$$

as long as  $m = ek$  and  $t = \log(s / \delta)$ . Moreover, the weights coming into each node in the hidden layer are in  $\{0,1\}$  with at most  $t$  non-zeros.

The final property implies that when using  $\mathbf{w}$  as a linear classifier, we get small generalization error as long as the number of examples is at least  $\Omega(s(1 + t\log mt))$ . This can be proved, e.g., using standard compression arguments: each such model can be represented using only  $st\log(mt)$  bits in addition to the representation size of  $\mathbf{w}$ . Similar bounds hold when we use  $\ell_1$  bounds on the weight coming into each unit. Note that even for  $s = d$  (i.e.  $\mathbf{w}$  is unrestricted), we get non-trivial input compression.

For comparison, we prove the following result for Gaussian projections in the appendix B. In this case, the model weights in our construction are not sparse.

Theorem 4.2. For every  $\mathbf{w} \in \mathcal{H}_{d,s}$  there exists a set of weights for a network  $N \in \mathcal{N}_s(\mathrm{Relu})$  such that for each  $\mathbf{x} \in B_{d,k}$ ,

$$
\Pr_ {h _ {1: t}} [ N (G \mathbf {x})) = \mathbf {w} ^ {\top} \mathbf {x} ] \geq 1 - \delta ,
$$

as long as  $G$  is a random  $m \times d$  Gaussian matrix, with  $m \geq 4k\log (s / \delta)$ .

# 5 SPARSE POLYNOMIAL FUNCTIONS

For boolean inputs, Theorem 4.1 extends immediately to sparse polynomial functions. Note that we can implement the AND of two bits  $x_{i} \wedge x_{j}$  as the AND of the corresponding decodings  $D_{h_{1:t}}^{\mathrm{AND}}(Y,i)$  and  $D_{h_{1:t}}^{\mathrm{AND}}(Y,j)$ . Since each decoding is an AND of  $t$  bits, the overall decoding is an AND of at most  $2t$  locations in the sketch. More generally, we have the following theorem:

Theorem 5.1. Given  $\mathbf{w} \in \mathbb{R}^s$ , and sets  $A_1, \ldots, A_s \subseteq [d]$ , let  $g: \{0, 1\}^d \to \mathbb{R}$  denote the polynomial

$$
g (\mathbf {x}) = \sum_ {j = 1} ^ {s} w _ {j} \prod_ {i \in A _ {j}} x _ {i} = \sum_ {j = 1} ^ {s} w _ {j} \bigwedge_ {i \in A _ {j}} x _ {i}.
$$

Then there exists a set of weights for a network  $N \in \mathcal{N}_s(\mathrm{Relu})$  such that for each  $\mathbf{x} \in B_{d,k}$ ,

$$
P r _ {h _ {1: t}} \left[ N \left(S _ {h _ {1: t}} (\mathbf {x})\right) = g (\mathbf {x}) \right] \geq 1 - \delta ,
$$

as long as  $m = ek$  and  $t = \log (|\cup_{j\in [s]}A_j| / \delta)$ . Moreover, the weights coming into each node in the hidden layer are in  $\{0,1\}$  with at most  $t\cdot \left(\sum_{j\in [s]}|A_j|\right)$  non-zeros overall. In particular, when  $g$  is a degree-  $p$  polynomial, we can set  $t = \log (ps / \delta)$ , and each hidden unit has at most  $pt$  non-zero weights.

This is a setting where we get a significant advantage over proper learning. To our knowledge, there is no analog of this result for Gaussian projections. Classical sketching approaches would use a sketch of  $\mathbf{x}^{\otimes p}$ , which is a  $k^p$ -sparse vector over binary vectors of dimension  $d^p$ . Known sketching techniques such as Pham & Pagh (2013) would construct a sketch of size  $\Omega(k^p)$ . Practical techniques such as Vowpal Wabbit also construct cross features by explicitly building them and have this exponential dependence. In stark contrast, neural networks allow us to get away with a logarithmic dependence on  $p$ .

Using polynomial kernels. Theorems 4.1 has a corresponding variants where the neural net is replaced by a polynomial of degree  $t$ . Similarly, the neural net in Theorem 5.1 can be replaced by a degree- $pt$  polynomial when the polynomial  $g$  has degree  $p$ . This implies that one can use a polynomial kernel to get efficient learning.

Deterministic sketching. A natural question that arises is whether the parameters above can improved. We show in App. C that if we allow large scalars in the sketches, one can construct a deterministic  $(2k + 1)$ -dimensional sketch from which a shallow network can reconstruct any monomial. We also show a lower bound of  $k$  on the required dimensionality.

Lower bound for proper learning. We can also show, see App. D, that if one does not expand the hypothesis class, then even in the simplest of settings of linear classifiers over 1-sparse vectors, the required dimensionality of the projection is much larger than the dimension needed for improper learning. The result is likely folklore and thus we present a short proof in the appendix for completeness using concrete constants in the theorem and its proof below.

Neural nets on Boolean inputs. We remark that for Boolean inputs (irrespective of sparsity), any polynomial with  $s$  monomials can be represented by a neural network in  $\mathcal{N}_s(\mathrm{Relu})$  using the construction in Theorem 5.1.

# 6 EXPERIMENTS WITH SYNTHETIC DATA

In this section, we evaluate sketches on synthetically generated datasets for the task of polynomial regression. In all the experiments here, we assume input dimension  $d = 10^4$ , input sparsity  $k = 50$ , hypothesis support  $s = 300$ , and  $n = 2 \times 10^5$  examples. We assume that only a subset of features  $\mathcal{I} \subseteq [d]$  are relevant for the regression task, with  $|\mathcal{I}| = 50$ . To generate an hypothesis, we select  $s$  subsets of relevant features  $A_1, \ldots, A_s \subset \mathcal{I}$  each of cardinality at most 3, and generate the corresponding weight vector  $\mathbf{w}$  by drawing corresponding  $s$  non-zero entries from the standard Gaussian distribution. We generate binary feature vectors  $\mathbf{x} \in B_{d,k}$  as a mixture of relevant and other features. Concretely, for each example we draw 12 feature indices uniformly at random from  $\mathcal{I}$ , and the remaining indices from  $[d]$ . We generate target outputs as  $g(\mathbf{x}) + z$ , where  $g(\mathbf{x})$  is in the form of the polynomial given in Theorem 5.1, and  $z$  is additive Gaussian noise with standard deviation 0.05. In all experiments, we train on 90% of the examples and evaluate mean squared error on the rest.

We first examined the effect of the sketching parameters  $m$  (hash size) and  $t$  (number of hash functions) on sparse linear regression error. We generated synthetic datasets as described above (with all feature subsets in  $\mathcal{A}$  having cardinality 1) and trained networks in  $\mathcal{N}_s(\mathrm{Relu})$ . The results are shown in Figure 2 (left). As expected, increasing  $t$  leads to better performance. Using hash size  $m$  less than the input sparsity  $k$  leads to poor results, while increasing hash size beyond  $ek$  (in this case,  $ek \cong 136$ ) for reasonable  $t$  yields only modest improvements.

We next examined the advantages of improper learning. We generated 10 sparse linear regression datasets and trained linear models and networks in  $\mathcal{N}_s(\mathrm{Relu})$  on original and sketched features with

![](images/b6507cf4955b771ebd0c80444372033617493a18abcf2d7aaa0a196106eb8528.jpg)  
Figure 2: Left: effect of varying  $t$ ,  $m$  for sketched 1-hidden layer network. Center: sparse linear regression on sketched data with improper learning. Right: sparse polynomial regression on sketched data.

![](images/5d83d0651f64ad78fcdf0348f4408fff239ba90999e9e9f95cd159cc037d1af1.jpg)

![](images/c5186dc8718074bb0f30a8c51c3e3b69eef567ddeb955b10d7028bed5a364e4b.jpg)

$m = 200$  and several values of  $t$ . The results are shown in Figure 2 (center). The neural network yields notably better performance than a linear model. This suggests that linear classifiers are not well-preserved after projections, as the  $\Omega(1/\gamma^2)$  projection size required for linear separability can be large. Applying a neural network to sketched data allows us to use smaller projections.

Table 1: Comparison of sketches and Gaussian random projections on the sparse linear regression task (top) and sparse polynomial regression task (bottom). See text for details.  

<table><tr><td></td><td>1K</td><td>2K</td><td>3K</td></tr><tr><td>Gaussian</td><td>0.089</td><td>0.057</td><td>0.029</td></tr><tr><td>Sketch t = 1</td><td>0.087</td><td>0.049</td><td>0.031</td></tr><tr><td>Sketch t = 2</td><td>0.072</td><td>0.041</td><td>0.023</td></tr><tr><td>Sketch t = 6</td><td>0.041</td><td>0.033</td><td>0.022</td></tr><tr><td>Gaussian</td><td>0.043</td><td>0.037</td><td>0.034</td></tr><tr><td>Sketch t = 1</td><td>0.041</td><td>0.036</td><td>0.033</td></tr><tr><td>Sketch t = 2</td><td>0.036</td><td>0.027</td><td>0.024</td></tr><tr><td>Sketch t = 6</td><td>0.032</td><td>0.022</td><td>0.018</td></tr></table>

We repeated the previous experiment for 10 polynomial regression datasets, generated with feature subsets in  $\mathcal{A}$  of cardinality 2 and 3. The results are shown in Figure 2 (right). The linear model is a bad fit, showing that  $g(\mathbf{x})$  is not well approximated by a linear function. Neural networks applied to sketches succeed in learning a small model and achieve significantly lower error than a network applied to the original features for  $t \geq 6$ . This suggests that reducing the input size, and consequently the number of model parameters, can lead to better generalization. Note that previous work on hashing and projections would imply using significantly larger sketch size for this setting.

We also compared our sketches to Gaussian random projections. We generated sparse linear and polynomial regression datasets with the same settings as before, and reduce the dimensionality of the inputs to 1000, 2000 and 3000 using Gaussian random projections and sketches with  $t \in \{1, 2, 6\}$ . We remark that in this comparison, the column headings correspond to the total sketch size  $mt$ . Thus, e.g., when we take  $t = 6$ ,  $m$  is correspondingly reduced. We report the squared error averaged across examples and five datasets of one-layer neural networks in Table 1. The results demonstrate that sketches with  $t > 1$  yield lower error than Gaussian projections. Note also that Gaussian projections are dense and hence much slower to train.

# 7 EXPERIMENTS WITH LANGUAGE PROCESSING TASKS

Linear and low degree sparse polynomials are often used for classification. Our results imply that if we have linear or a sparse polynomial with classification accuracy  $1 - \varepsilon$  over some set of examples in  $B_{d,k} \times \{0,1\}$ , then neural networks constructed to compute the linear or polynomial function attain accuracy of at least  $1 - \varepsilon - \delta$  over the same examples. Moreover, the number of parameters in the new network is relatively small by enforcing sparsity or  $\ell_1$  bounds for the weights into the hidden layers. We thus get generalization bounds with negligible degradation with respect to non-sketched predictor. In this section, we evaluate sketches on the language processing classification tasks described below.

![](images/a220d3e99447ed5a244a0f8ca5f33af639b7e5f302c4d1ea95cd2dfd12276c99.jpg)  
Figure 3: Performance vs. number of nonzero parameters in 1st layer for Reuters (left), AG News (center), and type tagging (right). Each color corresponds to a different sketch size and markers indicate the number of subsketches  $t$ . We evaluate each setting for three values of the  $l_{1}$  regularization parameter  $\lambda_{1}$ .

![](images/695bb9bd8f80827591e15bb44d2452d89ae223f5bec7b2b6182b5e53327fff8d.jpg)

![](images/99d76c54cde011b25b98339bae1f4460be009fc2552f5fa172ecce87412dacbd.jpg)

Entity Type Tagging. Entity type tagging is the task of assigning one or more labels (such as person, location, organization, event) to mentions of entities in text. We perform type tagging on a corpus of new documents containing 110K mentions annotated with 88 labels (on average, 1.7 labels per mention). Features for each mention include surrounding words, syntactic and lexical patterns, leading to a very large dictionary. Similarly to previous work, we map each string feature to a 32 bit integer, and then further reduce dimensionality using hashing or sketches. See Gillick et al. (2014) for more details on features and labels for this task.

Reuters-news Topic Classification. The Reuters RCV1 data set consists of a collection of approximately 800,000 text articles, each of which is assigned multiple labels. There are 4 high-level categories: Economics, Commerce, Medical, and Government (ECAT, CCAT, MCAT, GCAT), and multiple more specific categories. We focus on training binary classifiers for each of the four major categories. The input features we use are binary unigram features. Post word-stemming, we get data of approximately 113,000 dimensions. The feature vectors are very sparse, however, and most examples have fewer than 120 non-zero features.

AG-news Topic Classification. We perform topic classification on  $680K$  articles from AG news corpus, labeled with one of 8 news categories: Business, Entertainment, Health, Sci/Tech, Sports, Europe, U.S., World. For each document, we extract binary word indicator features from the title and description; in total, there are 210K unique features, and on average, 23 non-zero features per document.

Experimental Setup. In all experiments, we use two-layer feed-forward networks with ReLU activations and 100 hidden units in each layer. We use a softmax output for multiclass classification and multiple binary logistic outputs for multilabel tasks. We experimented with input sizes of 1000, 2000, 5000, and 10,000 and reduced the dimensionality of the original features using sketches with  $t \in \{1,2,4,6,8,10,12,14\}$  blocks. In addition, we experimented with networks trained on the original features. We encouraged parameter sparsity in the first layer using  $\ell_1$ -norm regularization and learn parameters using the proximal stochastic gradient method. As before, we trained on  $90\%$  of the examples and evaluated on the remaining  $10\%$ . We report accuracy values for multiclass classification, and F1 score for multilabel tasks, with true positive, false positive, and false negative counts accumulated across all labels.

Results. Since one motivation for our work is reducing the number of parameters in neural network models, we plot the performance metrics versus the number of non-zero parameters in the first layer of the network. The results are shown in Figure 3 for different sketching configurations and settings of the  $\ell_1$ -norm regularization parameters  $(\lambda_1)$ . On the entity type tagging task, we compared sketches to a single hash function of size 500,000 as the number of the original features is too large. In this case, sketching allows us to both improve performance and reduce the number of parameters. On the Reuters task, sketches achieve similar performance to the original features with fewer parameters. On AG news, sketching results in more compact models at a modest drop in accuracy. In almost all cases, multiple hash functions yield higher accuracy than a single hash function for similar model size.

# 8 CONCLUSIONS

We have presented a simple sketching algorithm for sparse boolean inputs, which succeeds in significantly reducing the dimensionality of inputs. A single-layer neural network on the sketch can provably model any sparse linear or polynomial function of the original input. For  $k$ -sparse vectors in  $\{0,1\}^d$ , our sketch of size  $O(k\log s / \delta)$  allows computing any  $s$ -sparse linear or polynomial function on a  $1 - \delta$  fraction of the inputs. The hidden constants are small, and our sketch is sparsity preserving. Previous work required sketches of size at least  $\Omega(s)$  in the linear case and size at least  $k^p$  for preserving degree-  $p$  polynomials. Our results can be viewed as showing a compressed sensing scheme for 0-1 vectors, where the decoding algorithm is a depth-1 neural network. Our scheme requires  $O(k\log d)$  measurements, and we leave open the question of whether this can be improved to  $O(k\log \frac{d}{k})$  in a stable way. We demonstrated empirically that our sketches work well for both linear and polynomial regression, and that using a neural network does improve over a direct linear regression. We show that on real datasets, our methods lead to smaller models with similar or better accuracy for multiclass and multilabel classification problems. In addition, the compact sketches lead to fewer trainable parameters and faster training.

# ACKNOWLEDGEMENTS

We would like to thank Amir Globerson for numerous fruitful discussion and help with an early version of the manuscript.

# REFERENCES

Dimitris Achlioptas. Database-friendly random projections: Johnson-Lindenstrauss with binary coins. J. Comput. Syst. Sci., 66(4):671-687, 2003.  
N. Ailon and B. Chazelle. The fast JL transform and approximate nearest neighbors. SICOMP, 39(1), 2009.  
Noga Alon, Yossi Matias, and Mario Szegedy. The space complexity of approximating the frequency moments. J. Comput. Syst. Sci., 58(1):137-147, 1999.  
Rosa I. Arriaga and Santosh Vempala. An algorithmic theory of learning: Robust concepts and random projection. Machine Learning, 63(2):161-182, 2006.  
J. Ba and R. Caruana. Do deep nets really need to be deep? In NIPS, pp. 2654-2662, 2014.  
K. Do Ba, P. Indyk, E. Price, and D. Woodruff. Lower bounds for sparse recovery. In SODA, 2010.  
A. Barron. Approximation and estimation bounds for artificial neural networks. Mach. Learning, 14(1), 1994.  
Andrew R Barron. Universal approximation bounds for superpositions of a sigmoidal function. Information Theory, IEEE Transactions on, 39(3):930-945, 1993.  
Vladimir Braverman, Rafail Ostrovsky, and Yuval Rabani. Rademacher chaos, random Eulerian graphs and the sparse Johnson-Lindenstrauss transform. CoRR, abs/1011.2590, 2010.  
E.J. Candés and T. Tao. Near optimal signal recovery from random projections: Universal encoding strategies? IEEE Transactions on Information Theory, 52(12):5406-5425, 2006.  
M. Charikar, K. Chen, and M. Farach. Finding frequent items in data streams. Theor. Comp. Sci., 312(1), 2004.  
Wenlin Chen, James T. Wilson, Stephen Tyree, Kilian Q. Weinberger, and Yixin Chen. Compressing convolutional neural networks. CoRR, abs/1506.04449, 2015. URL http://arxiv.org/abs/1506.04449.  
Y. Cheng, F. Yu, r. Feris, S. Kumar, A. Choudhary, and S-F. Chang. An exploration of parameter redundancy in deep networks with circulant projections. In CVPR, pp. 2857-2865, 2015.  
K. Clarkson and D. Woodruff. Low rank approximation and regression in input sparsity time. In STOC, 2013.  
M.D. Collins and P. Kohli. Memory bounded deep convolutional networks. CoRR, abs/1412.1442, 2014.  
G. Cormode and S. Muthukrishnan. An improved data stream summary: the count-min sketch and its applications. J. Algorithms, 55(1):58-75, 2005a.  
G. Cormode and S. Muthukrishnan. Summarizing and mining skewed data streams. In SDM, pp. 44-55, 2005b.  
Anirban Dasgupta, Ravi Kumar, and Tamás Sarlós. A sparse Johnson-Lindenstrauss transform. In STOC, pp. 341-350. ACM, 2010.

Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V Le, MarcAurelio Ranzato, Mark Mao, Andrew Senior, Paul Tucker, Ke Yang, and Andrew Y. Ng. Large scale distributed deep networks. In Advances in Neural Information Processing Systems, pp. 1223-1231, 2012.  
Misha Denil, Babak Shakibi, Laurent Dinh, MarcAurelio Ranzato, and Nando de Freitas. Predicting parameters in deep learning. In Advances in Neural Information Processing Systems, pp. 2148-2156, 2013.  
David L Donoho. Compressed sensing. Information Theory, IEEE Transactions on, 52(4):1289-1306, 2006.  
Kuzman Ganchev and Mark Dredze. Small statistical models by random feature mixing. In Workshop on Mobile NLP at ACL, 2008.  
D. Gillick, N. Lazic, K. Ganchev, J. Kirchner, and D. Huynh. Context-dependent fine-grained entity type tagging. CoRR, abs/1412.1820, 2014. URL http://arxiv.org/abs/1412.1820.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems, pp. 1135-1143, 2015.  
Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon. In Advances in Neural Information Processing Systems, volume 89, 1993.  
G. Hinton, L. Deng, D. Yu, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. Signal Processing Magazine, IEEE, 29(6):82-97, 2012.  
G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. CoRR, 1503.02531, 2015.  
W.B. Johnson and J. Lindenstrauss. Extensions of Lipschitz mappings into a Hilbert space. Contemp. Math, 26, 1984.  
Daniel M. Kane and Jelani Nelson. Sparger Johnson-Lindenstrauss transforms. J. ACM, 61(1):4:1-4:23, 2014.  
Leonid Kontorovich. A universal kernel for learning regular languages. In MLG, 2007.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, pp. 1097-1105, 2012.  
J. Langford, L. Li, and A. Strehl. Vowpal Wabbit online learning project. http://hunch.net/\~vw/, 2007.  
Yann LeCun, John S Denker, Sara A Solla, Richard E Howard, and Lawrence D Jackel. Optimal brain damage. In Advances in Neural Information Processing Systems, volume 89, 1989.  
J. Matousek. On variants of the Johnson-Lindenstrauss lemma. Rand. Struct. Algs., 33(2):142-156, 2008.  
Michael Mitzenmacher and Eli Upfal. Probability and Computing: Randomized Algorithms and Probabilistic Analysis. Cambridge University Press, New York, NY, USA, 2005. ISBN 0521835402.  
Ali Mousavi, Ankit B. Patel, and Richard G. Baraniuk. A deep learning approach to structured signal recovery. arXiv:1508.04065, 2015.  
Ninh Pham and Rasmus Pagh. Fast and scalable polynomial kernels via explicit feature maps. In KDD, pp. 239-247. ACM, 2013.  
A. Rahimi and B. Recht. Random features for large-scale kernel machines. In NIPS, pp. 1177-1184, 2007.  
Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. arXiv:1312.6229, 2013.  
Q. Shi, J. Petterson, G. Dror, J. Langford, A. J. Smola, A. Strehl, and V. Vishwanathan. Hash kernels. In Artificial Intelligence and Statistics AISTATS'09, Florida, April 2009.  
P.P. Talukdar and W.W. Cohen. Scaling graph-based semi supervised learning to large number of labels using count-min sketch. In AISTATS, volume 33 of JMLR Proceedings, pp. 940-947. JMLR.org, 2014.  
Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. CoRR, abs/1411.4555, 2014. URL http://arxiv.org/abs/1411.4555.  
K. Weinberger, A. Dasgupta, J. Attenberg, J. Langford, and A. J. Smola. Feature hashing for large scale multitask learning. In International Conference on Machine Learning, 2009.
