# WHY DEEP NEURAL NETWORKS FOR FUNCTION APPROXIMATION?

Shiyu Liang & R. Srikant

Coordinated Science Laboratory and Department of Electrical and Computer Engineering University of Illinois at Urbana-Champaign Urbana, IL 61801, USA {sliang26,rsrikant}@illinois.edu

# ABSTRACT

Recently there has been much interest in understanding why deep neural networks are preferred to shallow networks. We show that, for a large class of piecewise smooth functions, the number of neurons needed by a shallow network to approximate a function is exponentially larger than the corresponding number of neurons needed by a deep network for a given degree of function approximation. First, we consider univariate functions on a bounded interval and require a neural network to achieve an approximation error of  $\varepsilon$  uniformly over the interval. We show that shallow networks (i.e., networks whose depth does not depend on  $\varepsilon$ ) require  $\Omega(\mathrm{poly}(1/\varepsilon))$  neurons while deep networks (i.e., networks whose depth grows with  $1/\varepsilon$ ) require  $\mathcal{O}(\mathrm{polylog}(1/\varepsilon))$  neurons. We then extend these results to certain classes of important multivariate functions. Our results are derived for neural networks which use a combination of rectifier linear units (ReLUUs) and binary step units, two of the most popular type of activation functions. Our analysis builds on a simple observation: the multiplication of two bits can be represented by a ReLU.

# 1 INTRODUCTION

Neural networks have drawn significant interest from the machine learning community, especially due to their recent empirical successes (see the surveys (Bengio, 2009)). Neural networks are used to build state-of-art systems in various applications such as image recognition, speech recognition, natural language process and others (see, Krizhevsky et al. 2012; Goodfellow et al. 2013; Wan et al. 2013, for example). The result that neural networks are universal approximators is one of the theoretical results most frequently cited to justify the use of neural networks in these applications. Numerous results have shown the universal approximation property of neural networks in approximations of different function classes, (see, e.g., Cybenko 1989; Hornik et al. 1989; Funahashi 1989; Hornik 1991; Chui & Li 1992; Barron 1993; Poggio et al. 2015).

All these results and many others provide upper bounds on the network size and assert that small approximation error can be achieved if the network size is sufficiently large. More recently, there has been much interest in understanding the approximation capabilities of deep versus shallow networks. Delalleau & Bengio (2011) have shown that there exist deep sum-product networks which cannot be approximated by shallow sum-product networks unless they use an exponentially larger amount of units or neurons. Telgarsky (2016) has established such a result for neural networks, which is the subject of this paper. Eldan & Shamir (2015) have shown that, to approximate a specific function, a two-layer network requires an exponential number of neurons in the input dimension, while a three-layer network requires a polynomial number of neurons. These recent papers demonstrate the power of deep networks by showing that depth can lead to an exponential reduction in the number of neurons required, for specific functions or specific neural networks. Our goal here is different: we are interested in function approximation specifically and would like to show that for a given upper bound on the approximation error, shallow networks require exponentially more neurons than deep networks for a large class of functions.

The multilayer neural networks considered in this paper are allowed to use either rectifier linear units (ReLU) or binary step units (BSU), or any combination of the two. The main contributions of this paper are

- We have shown that, for  $\varepsilon$ -approximation of functions with enough piecewise smoothness, a multilayer neural network which uses  $\Theta(\log(1/\varepsilon))$  layers only needs  $\mathcal{O}(\text{poly}\log(1/\varepsilon))$  neurons, while  $\Omega(\text{poly}(1/\varepsilon))$  neurons are required by neural networks with  $o(\log(1/\varepsilon))$  layers. In other words, shallow networks require exponentially more neurons than a deep network to achieve the level of accuracy for function approximation.  
- We have shown that for all differentiable and strongly convex functions, multilayer neural networks need  $\Omega (\log (1 / \varepsilon))$  neurons to achieve an  $\varepsilon$ -approximation. Thus, our results for deep networks are tight.

The outline of this paper is as follows. In Section 2, we present necessary definitions and the problem statement. In Section 3, we present upper bounds on network size, while the lower bound is provided in Section 4. Conclusions are presented in Section 5. Around the same time that our paper was uploaded in arxiv, a similar paper was also uploaded in arXiv by Yarotsky (2016). The results in the two papers are similar in spirit, but the details and the general approach are substantially different.

# 2 PRELIMINARIES AND PROBLEM STATEMENT

In this section, we present definitions on feedforward neural networks and formally present the problem statement.

# 2.1 FEEDFORWARD NEURAL NETWORKS

A feedforward neural network is composed of layers of computational units and defines a unique function  $\tilde{f}:\mathbb{R}^d\to \mathbb{R}$ . Let  $L$  denote the number of hidden layers,  $N_{l}$  denote the number of units of layer  $l$ ,  $N = \sum_{l = 1}^{L}N_{l}$  denote the size of the neural network, vector  $\pmb {x} = (x^{(1)},\dots,x^{(d)})$  denote the input of neural network,  $z_{j}^{l}$  denote the output of the  $j$ th unit in layer  $l$ ,  $w_{i,j}^{l}$  denote the weight of the edge connecting unit  $i$  in layer  $l$  and unit  $j$  in layer  $l + 1$ ,  $b_{j}^{l}$  denote the bias of the unit  $j$  in layer  $l$ . Then outputs between layers of the feedforward neural network can be characterized by following iterations:

$$
z _ {j} ^ {l + 1} = \sigma \left(\sum_ {i = 1} ^ {N _ {l}} w _ {i, j} ^ {l} z _ {i} ^ {l} + b _ {j} ^ {l + 1}\right), \quad l \in [ L - 1 ], j \in [ N _ {l + 1} ],
$$

with

$$
\text {i n p u t l a y e r :} z _ {j} ^ {1} = \sigma \left(\sum_ {i = 1} ^ {d} w _ {i, j} ^ {0} x ^ {(i)} + b _ {j} ^ {1}\right), \quad j \in [ N _ {1} ],
$$

$$
\text {o u t p u t l a y e r :} \tilde {f} (\boldsymbol {x}) = \sigma \left(\sum_ {i = 1} ^ {N _ {L}} w _ {i, j} ^ {L} z _ {i} ^ {L} + b _ {j} ^ {L + 1}\right).
$$

Here,  $\sigma(\cdot)$  denotes the activation function and  $[n]$  denotes the index set  $[n] = \{1, \dots, n\}$ . In this paper, we only consider two important types of activation functions:

- Rectifier linear unit:  $\sigma(x) = \max\{0, x\}, x \in \mathbb{R}$ .  
- Binary step unit:  $\sigma(x) = \mathbb{I}\{x \geq 0\}, x \in \mathbb{R}$ .

We call the number of layers and the number of neurons in the network as the depth and the size of the feedforward neural network, respectively. We use the set  $\mathcal{F}(N,L)$  to denote the function set containing all feedforward neural networks of depth  $L$ , size  $N$  and composed of a combination of rectifier linear units (ReLU's) and binary step units. We say one feedforward neural network is deeper than the other network if and only if it has a larger depth. Through this paper, the terms feedforward neural network and multilayer neural network are used interchangeably.

![](images/c82fa44b15bb606063444f8ae8c9673309a34b5094b27ba89d07b33ac8fe4999.jpg)  
Figure 1: An  $n$ -layer neural network structure for finding the binary expansion of a number in [0, 1].

# 2.2 PROBLEM STATEMENT

In this paper, we focus on bounds on the size of the feedforward neural network function approximation. Given a function  $f$ , our goal is to understand whether a multilayer neural network  $\tilde{f}$  of depth  $L$  and size  $N$  exists such that it solves

$$
\min  _ {\tilde {f} \in \mathcal {F} (N, L)} \| f - \tilde {f} \| \leq \varepsilon . \tag {1}
$$

Specifically, we aim to answer the following questions:

1. Does there exist  $L(\varepsilon)$  and  $N(\varepsilon)$  such that (1) is satisfied? We will refer to such  $L(\varepsilon)$  and  $N(\varepsilon)$  as upper bounds on the depth and size of the required neural network.  
2 Given a fixed depth  $L$ , what is the minimum value of  $N$  such that (1) is satisfied? We will refer to such an  $N$  as a lower bound on the size of a neural network of a given depth  $L$ .

The first question asks what depth and size are sufficient to guarantee an  $\varepsilon$ -approximation. The second question asks, for a fixed depth, what is the minimum size of a neural network required to guarantee an  $\varepsilon$ -approximation. Obviously, tight bounds in the answers to these two questions provide tight bounds on the network size and depth required for function approximation. Besides, solutions to these two questions together can be further used to answer the following question. If a deeper neural network of size  $N_{d}$  and a shallower neural network of size  $N_{s}$  are used to approximate the same function with the same error, then how fast does the ratio  $N_{d} / N_{s}$  decay to zero as the error decays to zero?

# 3 UPPER BOUNDS ON FUNCTION APPROXIMATIONS

In this section, we present upper bounds on the size of the multilayer neural network which are sufficient for function approximation. Before stating the results, some notations and terminology deserve further explanation. First, the upper bound on the network size represents the number of neurons required at most for approximating a given function with a certain error. Secondly, the notion of the approximation is the  $L_{\infty}$  distance: for two functions  $f$  and  $g$ , the  $L_{\infty}$  distance between these two functions is the maximum point-wise disagreement over the cube  $[0,1]^d$ .

# 3.1 APPROXIMATION OF UNIVARIATE FUNCTIONS

In this subsection, we present all results on approximating univariate functions. We first present a theorem on the size of the network for approximating a simple quadratic function. As part of the proof, we present the structure of the multilayer feedforward neural network used and show how the neural network parameters are chosen. Results on approximating general functions can be found in Theorem 2 and 4.

Theorem 1. For function  $f(x) = x^{2}, x \in [0,1]$ , there exists a multilayer neural network  $\tilde{f}(x)$  with  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  binary step units and  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  rectifier linear units such that  $|f(x) - \tilde{f}(x)| \leq \varepsilon$ ,  $\forall x \in [0,1]$ .

Proof. The proof is composed of three parts. For any  $x \in [0,1]$ , we first use the multilayer neural network to approximate  $x$  by its finite binary expansion  $\sum_{i=0}^{n} \frac{x_i}{2^i}$ . We then construct a 2-layer neural network to implement function  $f\left(\sum_{i=0}^{n} \frac{x_i}{2^i}\right)$ .

For each  $x \in [0,1]$ ,  $x$  can be denoted by its binary expansion  $x = \sum_{i=0}^{\infty} \frac{x_i}{2^i}$ , where  $x_i \in \{0,1\}$  for all  $i \geq 0$ . It is straightforward to see that the  $n$ -layer neural network shown in Figure 1 can be used to find  $x_0, \ldots, x_n$ .

Next, we implement the function  $\tilde{f}(x) = f\left(\sum_{i=0}^{n} \frac{x_i}{2^i}\right)$  by a two-layer neural network. Since  $f(x) = x^2$ , we then rewrite  $\tilde{f}(x)$  as follows:

$$
\tilde {f} (x) = \left(\sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}}\right) ^ {2} = \sum_ {i = 0} ^ {n} \left[ x _ {i} \cdot \left(\frac {1}{2 ^ {i}} \sum_ {j = 0} ^ {n} \frac {x _ {j}}{2 ^ {j}}\right) \right] = \sum_ {i = 0} ^ {n} \max \left(0, 2 (x _ {i} - 1) + \frac {1}{2 ^ {i}} \sum_ {j = 0} ^ {n} \frac {x _ {j}}{2 ^ {j}}\right).
$$

The third equality follows from the fact that  $x_{i} \in \{0,1\}$  for all  $i$ . Therefore, the function  $\tilde{f}(x)$  can be implemented by a multilayer network containing a deep structure shown in Figure 1 and another hidden layer with  $n$  rectifier linear units. This multilayer neural network has  $\mathcal{O}(n)$  layers,  $\mathcal{O}(n)$  binary step units and  $\mathcal{O}(n)$  rectifier linear units.

Finally, we consider the approximation error of this multilayer neural network,

$$
| f (x) - \tilde {f} (x) | = \left| x ^ {2} - \left(\sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}}\right) ^ {2} \right| \leq 2 \left| x - \sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}} \right| = 2 \left| \sum_ {i = n + 1} ^ {\infty} \frac {x _ {i}}{2 ^ {i}} \right| \leq \frac {1}{2 ^ {n - 1}}.
$$

Therefore, in order to achieve  $\varepsilon$ -approximation error, one should choose  $n = \left\lceil \log_2\frac{1}{\varepsilon}\right\rceil + 1$ . In summary, the deep neural network has  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  binary step units and  $\mathcal{O}\left(\log \left(\frac{1}{\varepsilon}\right)\right)$  rectifier linear units.

Next, a theorem on the size of the network for approximating general polynomials is given as follows.

Theorem 2. For polynomials  $f(x) = \sum_{i=0}^{p} a_i x^i$ ,  $x \in [0,1]$  and  $\sum_{i=1}^{p} |a_i| \leq 1$ , there exists a multilayer neural network  $\tilde{f}(x)$  with  $\mathcal{O}\left(p + \log \frac{p}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{p}{\varepsilon}\right)$  binary step units and  $\mathcal{O}\left(p \log \frac{p}{\varepsilon}\right)$  rectifier linear units such that  $|f(x) - \tilde{f}(x)| \leq \varepsilon, \forall x \in [0,1]$ .

Proof. The proof is composed of three parts. We first use the deep structure shown in Figure 1 to find the  $n$ -bit binary expansion  $\sum_{i=0}^{n} a_i x^i$  of  $x$ . Then we construct a multilayer network to approximate polynomials  $g_i(x) = x^i$ ,  $i = 1, \dots, p$ . Finally, we analyze the approximation error.

Using the same deep structure shown in Figure 1, we could find the binary expansion sequence  $\{x_0,\dots,x_n\}$ . In this step, we used  $n$  binary steps units in total. Now we rewrite  $g_{m + 1}(\sum_{i = 0}^{n}\frac{x_i}{2^n})$ ,

$$
g _ {m + 1} \left(\sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}}\right) = \sum_ {j = 0} ^ {n} \left[ x _ {j} \cdot \frac {1}{2 ^ {j}} g _ {m} \left(\sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}}\right) \right] = \sum_ {j = 0} ^ {n} \max  \left[ 2 \left(x _ {j} - 1\right) + \frac {1}{2 ^ {j}} g _ {m} \left(\sum_ {i = 0} ^ {n} \frac {x _ {i}}{2 ^ {i}}\right), 0 \right]. \tag {2}
$$

Clearly, the equation (2) defines iterations between the outputs of neighbor layers. Therefore, the deep neural network shown in Figure 2 can be used to implement the iteration given by (2). Further, to implement this network, one should use  $\mathcal{O}(p)$  layers with  $\mathcal{O}(pn)$  rectifier linear units in total. We now define the output of the multilayer neural network as  $\tilde{f} (x) = \sum_{i = 0}^{p}a_{i}g_{i}\left(\sum_{j = 0}^{n}\frac{x_{j}}{2^{j}}\right)$ . For this multilayer network, the approximation error is

$$
| f (x) - \tilde {f} (x) | = \left| \sum_ {i = 0} ^ {p} a _ {i} g _ {i} \left(\sum_ {j = 0} ^ {n} \frac {x _ {j}}{2 ^ {j}}\right) - \sum_ {i = 0} ^ {p} a _ {i} x ^ {i} \right| \leq \sum_ {i = 0} ^ {p} \left[ | a _ {i} | \cdot \left| g _ {i} \left(\sum_ {j = 0} ^ {n} \frac {x _ {j}}{2 ^ {j}}\right) - x ^ {i} \right| \right] \leq \frac {p}{2 ^ {n - 1}}
$$

This indicates, to achieve  $\varepsilon$ -approximation error, one should choose  $n = \left\lceil \log \frac{p}{\varepsilon} \right\rceil + 1$ . Besides, since we used  $\mathcal{O}(n + p)$  layers with  $\mathcal{O}(n)$  binary step units and  $\mathcal{O}(pn)$  rectifier linear units in total, this multilayer neural network thus has  $\mathcal{O}\left(p + \log \frac{p}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{p}{\varepsilon}\right)$  binary step units and  $\mathcal{O}\left(p \log \frac{p}{\varepsilon}\right)$  rectifier linear units.

![](images/684f8de4f56ffb225e2468fa2e4fd704f7ae75fb7ae3e2ba3909bcc1d9d66499.jpg)  
Figure 2: The implementation of polynomial function

![](images/1a6388b1e62976fc4fdcc975897e8cb0923a6aec27dc4bc6d4736d937356bfe7.jpg)

In Theorem 2, we have shown an upper bound on the size of multilayer neural network for approximating polynomials. We can easily observe that the number of neurons in network grows as  $p \log p$  with respect to  $p$ , the degree of the polynomial. We note that both Andoni et al. (2014) and Barron (1993) showed the sizes of the networks grow exponentially with respect to  $p$  if only 3-layer neural networks are allowed to be used in approximating polynomials.

Besides, every function  $f$  with  $p + 1$  continuous derivatives on a bounded set can be approximated easily with a polynomial with degree  $p$ . This is shown by the following well known result of Lagrangian interpolation. By this result, we could further generalize Theorem 2. The proof can be found in the reference (Gil et al., 2007).

Lemma 3 (Lagrangian interpolation at Chebyshev points). If a function  $f$  is defined at points  $z_0, \ldots, z_n$ ,  $z_i = \cos((k + 1/2)\pi/(n + 1))$ ,  $i \in [n]$ , there exists a polynomial of degree not more than  $n$  such that  $P_n(z_i) = f(z_i)$ ,  $i = 0, \ldots, n$ . This polynomial is given by  $P_n(x) = \sum_{i=0}^{n} f(z_i)L_i(x)$  where  $L_i(x) = \frac{\pi_{n+1}(x)}{(x - z_i)\pi_{n+1}'(z_i)}$  and  $\pi_{n+1}(x) = \prod_{j=0}^{n}(x - z_j)$ . Additionally, if  $f$  is continuous on  $[-1,1]$  and  $n + 1$  times differentiable in  $(-1,1)$ , then

$$
\left\| R _ {n} \right\| = \left\| f - P _ {n} \right\| \leq \frac {1}{2 ^ {n} (n + 1) !} \left\| f ^ {(n + 1)} \right\|,
$$

where  $f^{(n)}(x)$  is the derivative of  $f$  of the  $n$ th order and the norm  $\| f \|$  is the  $l_{\infty}$  norm  $\| f \| = \max_{i \in [-1,1]} f(x)$ .

Then the upper bound on the network size for approximating more general functions follows directly from Theorem 2 and Lemma 3.

Theorem 4. Assume that function  $f$  is continuous on  $[0,1]$  and  $\left\lceil \log \frac{2}{\varepsilon} \right\rceil + 1$  times differentiable in  $(0,1)$ . Let  $f^{(n)}$  denote the derivative of  $f$  of  $n$ th order and  $\|f\| = \max_{x \in [0,1]} f(x)$ . If  $\left\|f^{(n)}\right\| \leq n!$  holds for all  $n \in \left[\left\lceil \log \frac{2}{\varepsilon} \right\rceil + 1\right]$ , then there exists a deep neural network  $\tilde{f}$  with  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  binary step units,  $\mathcal{O}\left(\left(\log \frac{1}{\varepsilon}\right)^2\right)$  rectifier linear units such that  $|f(x) - \tilde{f}| \leq \varepsilon, \forall x \in [0,1]$ .

Proof. Let  $N = \left\lceil \log \frac{2}{\epsilon} \right\rceil$ . From Lemma 3, it follows that there exists a polynomial  $P_N$  of degree  $N$  such that for any  $x \in [0,1]$ ,

$$
| f (x) - P _ {N} (x) | \leq \frac {\left\| f ^ {(N + 1)} \right\|}{2 ^ {N} (N + 1) !} \leq \frac {1}{2 ^ {N}}.
$$

Let  $x_0, \ldots, x_N$  denote the first  $N + 1$  bits of the binary expansion of  $x$  and define  $\tilde{f}(x) = P_N\left(\sum_{i=0}^{N} \frac{x_i}{2^N}\right)$ . In the following, we first analyze the approximation error of  $\tilde{f}$  and next show the implementation of this function. Let  $\tilde{x} = \sum_{i=0}^{N} \frac{x_i}{2^i}$ . The error can now be upper bounded by

$$
\begin{array}{l} | f (x) - \tilde {f} (x) | = | f (x) - P _ {N} (\tilde {x}) | \leq | f (x) - f (\tilde {x}) | + | f (\tilde {x}) - P _ {N} (\tilde {x}) | \\ \leq \left\| f ^ {(1)} \right\| \cdot \left| x - \sum_ {i = 0} ^ {N} \frac {x _ {i}}{2 ^ {i}} \right| + \frac {1}{2 ^ {N}} \leq \frac {1}{2 ^ {N}} + \frac {1}{2 ^ {N}} \leq \varepsilon \\ \end{array}
$$

In the following, we describe the implementation of  $\tilde{f}$  by a multilayer neural network. Since  $P_N$  is a polynomial of degree  $N$ , function  $\tilde{f}$  can be rewritten as

$$
\tilde {f} (x) = P _ {N} \left(\sum_ {i = 0} ^ {N} \frac {x _ {i}}{2 ^ {i}}\right) = \sum_ {n = 0} ^ {N} c _ {n} g _ {n} \left(\sum_ {i = 0} ^ {N} \frac {x _ {i}}{2 ^ {i}}\right)
$$

for some coefficients  $c_{0},\ldots ,c_{N}$  and  $g_{n} = x^{n}$ $n\in [N]$ . Hence, the multilayer neural network shown in the Figure 2 can be used to implement  $\tilde{f} (x)$ . Notice that the network uses  $\mathcal{O}(N)$  layers with  $\mathcal{O}(N)$  binary step units in total to decode  $x_0,\dots,x_N$  and  $\mathcal{O}(N)$  layers with  $\mathcal{O}(N^2)$  rectifier linear units in total to construct the polynomial  $P_{N}$ . Substituting  $N = \left\lceil \log \frac{2}{\varepsilon}\right\rceil$ , we have proved the theorem.

Remark: Note that, to implement the architecture in Figure 2 using the definition of a feedforward neural network in Section 2, we need the  $g_{i}$ ,  $i \in [p]$  at the output. This can be accomplished by using  $\mathcal{O}(p^2)$  additional RLUs. Since  $p = \mathcal{O}(\log (1 / \varepsilon))$ , this doesn't change the order result in Theorem 4.

Theorem 4 shows that any function  $f$  with enough smoothness can be approximated by a multilayer neural network containing  $\text{polylog}\left(\frac{1}{\varepsilon}\right)$  neurons with  $\varepsilon$  error. Further, Theorem 4 can be used to show that for functions  $h_1, \ldots, h_k$  with enough smoothness, then linear combinations, multiplications and compositions of these functions can as well be approximated by multilayer neural networks containing  $\text{polylog}\left(\frac{1}{\varepsilon}\right)$  neurons with  $\varepsilon$  error. Specific results are given in the following corollaries.

Corollary 5 (Function addition). Suppose that all functions  $h_1, \ldots, h_k$  satisfy the conditions in Theorem 4, and the vector  $\beta \in \{\omega \in \mathbb{R}^k : \| \omega \|_1 = 1\}$ , then for the linear combination  $f = \sum_{i=1}^{k} \beta_i h_i$ , there exists a deep neural network  $\tilde{f}$  with  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  layers,  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  binary step units,  $\mathcal{O}\left(\left(\log \frac{1}{\varepsilon}\right)^2\right)$  rectifier linear units such that  $|f(x) - \tilde{f}| \leq \varepsilon$ ,  $\forall x \in [0,1]$ .

Remark: Clearly, Corollary 5 follows directly from the fact that the linear combination  $f$  satisfies the conditions in Theorem 4 if all the functions  $h_1, \dots, h_k$  satisfy those conditions. We note here that the upper bound on the network size for approximating linear combinations is independent of  $k$ , the number of component functions.

Corollary 6 (Function multiplication). Suppose that all functions  $h_1, \ldots, h_k$  are continuous on  $[0,1]$  and  $\left\lceil 4k\log_24k + 4k + 2\log_2\frac{2}{\varepsilon}\right\rceil + 1$  times differentiable in  $(0,1)$ . If  $\|h_i^{(n)}\| \leq n!$  holds for all  $i \in [k]$  and  $n \in \left[\left\lceil 4k\log_24k + 4k + 2\log_2\frac{2}{\varepsilon}\right\rceil + 1\right]$  then for the multiplication  $f = \prod_{i=1}^{k} h_i$ , there exists a multilayer neural network  $\tilde{f}$  with  $\mathcal{O}(k\log k + \log \frac{1}{\varepsilon})$  layers,  $\mathcal{O}(k\log k + \log \frac{1}{\varepsilon})$  binary step units and  $\mathcal{O}\left((k\log k)^2 + (\log \frac{1}{\varepsilon})^2\right)$  rectifier linear units such that  $|f(x) - \tilde{f}(x)| \leq \varepsilon$ ,  $\forall x \in [0,1]$ .

Corollary 7 (Function composition). Suppose that all functions  $h_1, \ldots, h_k : [0,1] \to [0,1]$  satisfy the conditions in Theorem 4, then for the composition  $f = h_1 \circ h_2 \circ \ldots \circ h_k$ , there exists a multilayer neural network  $\tilde{f}$  with  $\mathcal{O}\left(k\log k\log \frac{1}{\varepsilon} + \log k\left(\log \frac{1}{\varepsilon}\right)^2\right)$  layers,  $\mathcal{O}\left(k\log k\log \frac{1}{\varepsilon} + \log k\left(\log \frac{1}{\varepsilon}\right)^2\right)$  binary step units and  $\mathcal{O}\left(k^2\left(\log \frac{1}{\varepsilon}\right)^2 + \left(\log \frac{1}{\varepsilon}\right)^4\right)$  rectifier linear units such that  $|f(x) - \tilde{f}(x)| \leq \varepsilon, \forall x \in [0,1]$ .

Remark: Proofs of Corollary 6 and 7 can be found in the appendix. We observe that different from the case of linear combinations, the upper bound on the network size grows as  $k^2 \log^2 k$  in the case of function multiplications and grows as  $k^2 \left( \log \frac{1}{\varepsilon} \right)^2$  in the case of function compositions where  $k$  is the number of component functions.

In this subsection, we have shown a polylog  $\left(\frac{1}{\varepsilon}\right)$  upper bound on the network size for  $\varepsilon$ -approximation of both univariate polynomials and general univariate functions with enough smoothness. Besides, we have shown that linear combinations, multiplications and compositions of univariate functions with enough smoothness can as well be approximated with  $\varepsilon$  error by a multilayer neural network of size polylog  $\left(\frac{1}{\varepsilon}\right)$ . In the next subsection, we will show the upper bound on the network size for approximating multivariate functions.

# 3.2 APPROXIMATION OF MULTIVARIATE FUNCTIONS

In this subsection, we present all results on approximating multivariate functions. We first present a theorem on the upper bound on the neural network size for approximating a product of multivariate linear functions. We next present a theorem on the upper bound on the neural network size for approximating general multivariate polynomial functions. Finally, similar to the results in the univariate case, we present the upper bound on the neural network size for approximating the linear combination, the multiplication and the composition of multivariate functions with enough smoothness.

Theorem 8. Let  $W = \{\pmb{w} \in \mathbb{R}^d : \| \pmb{w} \|_1 = 1\}$ . For  $f(\pmb{x}) = \prod_{i=1}^{p} (\pmb{w}_i^T \pmb{x})$ ,  $\pmb{x} \in [0,1]^d$  and  $\pmb{w}_i \in W$ ,  $i = 1, \dots, p$ , there exists a deep neural network  $\tilde{f}(\pmb{x})$  with  $\mathcal{O}\left(p + \log \frac{pd}{\varepsilon}\right)$  layers and  $\mathcal{O}\left(\log \frac{pd}{\varepsilon}\right)$  binary step units and  $\mathcal{O}\left(pd \log \frac{pd}{\varepsilon}\right)$  rectifier linear units such that  $|f(\pmb{x}) - \tilde{f}(\pmb{x})| \leq \varepsilon$ ,  $\forall \pmb{x} \in [0,1]^d$ .

Theorem 8 shows an upper bound on the network size for  $\varepsilon$ -approximation of a product of multivariate linear functions. Furthermore, since any general multivariate polynomial can be viewed as a linear combination of products, the result on general multivariate polynomials directly follows from Theorem 8.

Theorem 9. Let the multi-index vector  $\alpha = (\alpha_{1},\dots,\alpha_{d})$ , the norm  $|\alpha| = \alpha_{1} + \dots + \alpha_{d}$ , the coefficient  $C_{\alpha} = C_{\alpha_{1}\dots \alpha_{d}}$ , the input vector  $\pmb{x} = (x^{(1)},\dots,x^{(d)})$  and the multinomial  $\pmb{x}^{\alpha} = x^{(1)^{\alpha_{1}}}\dots x^{(d)^{\alpha_{d}}}$ . For positive integer  $p$  and polynomial  $f(\pmb{x}) = \sum_{\alpha :|\alpha |\leq p}C_{\alpha}\pmb{x}^{\alpha}$ ,  $\pmb{x}\in [0,1]^d$  and  $\sum_{\alpha :|\alpha |\leq p}|C_{\alpha}|\leq 1$ , there exists a deep neural network  $\tilde{f} (\pmb{x})$  of depth  $\mathcal{O}\left(p + \log \frac{dp}{\varepsilon}\right)$  and size  $N(d,p,\varepsilon)$  such that  $|f(\pmb {x}) - f(\tilde{\pmb{x}})|\leq \varepsilon$ , where

$$
N (d, p, \varepsilon) = p ^ {2} \left( \begin{array}{c} p + d - 1 \\ d - 1 \end{array} \right) \log \frac {p d}{\varepsilon}.
$$

Remark: The proof is given in the appendix. By further analyzing the results on the network size, we obtain the following results: (a) fixing degree  $p$ ,  $N(d, \varepsilon) = \mathcal{O}\left(d^{p + 1} \log \frac{d}{\varepsilon}\right)$  as  $d \to \infty$  and (b) fixing input dimension  $d$ ,  $N(p, \varepsilon) = \mathcal{O}\left(p^d \log \frac{p}{\varepsilon}\right)$  as  $p \to \infty$ . Similar results on approximating multivariate polynomials were obtained by Andoni et al. (2014) and Barron (1993). Barron (1993) showed that on can use a 3-layer neural network to approximate any multivariate polynomial with degree  $p$ , dimension  $d$  and network size  $d^p / \varepsilon^2$ . Andoni et al. (2014) showed that one could use the gradient descent to train a 3-layer neural network of size  $d^{2p} / \varepsilon^2$  to approximate any multivariate polynomial. However, Theorem 9 shows that the deep neural network could reduce the network size from  $\mathcal{O}(1 / \varepsilon)$  to  $\mathcal{O}\left(\log \frac{1}{\varepsilon}\right)$  for the same  $\varepsilon$  error. Besides, for a fixed input dimension  $d$ , the size of the 3-layer neural network used by Andoni et al. (2014) and Barron (1993) grows exponentially with respect to the degree  $p$ . However, the size of the deep neural network shown in Theorem 9 grows only polynomially with respect to the degree. Therefore, the deep neural network could reduce the network size from  $\mathcal{O}(\exp(p))$  to  $\mathcal{O}(\mathrm{poly}(p))$  when the degree  $p$  becomes large.

Theorem 9 shows an upper bound on the network size for approximating multivariate polynomials. Further, by combining Theorem 4 and Corollary 7, we could obtain an upper bound on the network size for approximating more general functions. The results are shown in the following corollary.

Corollary 10. Assume that all univariate functions  $h_1, \ldots, h_k : [0,1] \to [0,1]$ ,  $k \geq 1$ , satisfy the conditions in Theorem 4. Assume that the multivariate polynomial  $l(\pmb{x}) : [0,1]^d \to [0,1]$  is of degree  $p$ . For composition  $f = h_1 \circ h_2 \circ \ldots \circ h_k \circ l(\pmb{x})$ , there exists a multilayer neural network  $\tilde{f}$  of depth  $\mathcal{O}\left(p + \log d + k\log k\log \frac{1}{\varepsilon} +\log k\left(\log \frac{1}{\varepsilon}\right)^2\right)$  and of size  $N(k,p,d,\varepsilon)$  such that  $|\tilde{f} (\pmb{x}) - f(\pmb{x})| \leq \varepsilon$  for  $\forall x \in [0,1]^d$ , where

$$
N (k, p, d, \varepsilon) = \mathcal {O} \left(p ^ {2} \left( \begin{array}{c} p + d - 1 \\ d - 1 \end{array} \right) \log \frac {p d}{\varepsilon} + k ^ {2} \left(\log \frac {1}{\varepsilon}\right) ^ {2} + \left(\log \frac {1}{\varepsilon}\right) ^ {4}\right).
$$

Remark: Corollary 10 shows an upper bound on network size for approximating compositions of multivariate polynomials and general univariate functions. The upper bound can be loose due to the assumption that  $l(\pmb{x})$  is a general multivariate polynomials of degree  $p$ . For some specific cases, the upper bound can be much smaller. We present two specific examples in the Appendix G and H.

In this subsection, we have shown that a similar polylog  $\left(\frac{1}{\varepsilon}\right)$  upper bound on the network size for  $\varepsilon$ -approximation of general multivariate polynomials and functions which are compositions of univariate functions and multivariate polynomials.

The results in this section can be used to find a multilayer neural network of size polylog  $\left(\frac{1}{\varepsilon}\right)$  which provides an approximation error of at most  $\varepsilon$ . In the next section, we will present lower bounds on the network size for approximating both univariate and multivariate functions. The lower bound together with the upper bound shows a tight bound on the network size required for function approximations.

While we have presented results in both the univariate and multivariate cases for smooth functions, the results automatically extend to functions that are piecewise smooth, with a finite number of pieces. In other words, if the domain of the function is partitioned into regions, and the function is sufficiently smooth (in the sense described in the Theorems and Corollaries earlier) in each of the regions, then the results essentially remain unchanged except for an additional factor which will depend on the number of regions in the domain.

# 4 LOWER BOUNDS ON FUNCTION APPROXIMATIONS

In this section, we present lower bounds on the network size in function for certain classes of functions. Next, by combining the lower bounds and the upper bounds shown in the previous section, we could analytically show the advantages of deeper neural networks over shallower ones. The theorem below is inspired by a similar result (DasGupta & Schnitger, 1993) for univariate quadratic functions, where it is stated without a proof. Here we show that the result extends to general multivariate strongly convex functions.

Theorem 11. Assume function  $f:[0,1]^d\to \mathbb{R}$  is differentiable and strongly convex with parameter  $\mu$ . Assume the multilayer neural network  $\tilde{f}$  is composed of rectifier linear units and binary step units. If  $|f(\pmb {x}) - \tilde{f} (\pmb {x})|\leq \varepsilon ,\forall \pmb {x}\in [0,1]$ , then the depth  $L$  and the network size  $N$  should satisfy

$$
N \geq L \left(\frac {\mu}{1 6 \varepsilon}\right) ^ {\frac {1}{2 L}}.
$$

This indicates the following network size should satisfy  $N \geq \log_2\left(\frac{\mu}{16\varepsilon}\right)$ .

Remark: The proof is in the Appendix F. Theorem 11 shows that every strongly convex function cannot be approximated with error  $\varepsilon$  by any multilayer neural network with rectifier linear units and binary step units and of size smaller than  $\log_2(\mu/\varepsilon) - 4$ . Theorem 11 together with Theorem 1 directly shows that to approximate quadratic function  $f(x) = x^2$  with error  $\varepsilon$ , the network size should be of order  $\Theta\left(\log \frac{1}{\varepsilon}\right)$ . Further, by combining Theorem 11 and Theorem 4, we could analytically show the benefits of deeper neural networks. The result is given in the following corollary.

Corollary 12. Assume that univariate function  $f$  satisfies conditions in both Theorem 4 and Theorem 11. If a neural network  $\tilde{f}_s$  is of depth  $L_s = o\left(\log \frac{1}{\varepsilon}\right)$ , size  $N_s$  and  $|f(x) - \tilde{f}_s(x)| \leq \varepsilon$ , for  $\forall x \in [0,1]$ , then there exists a deeper neural network  $\tilde{f}_d(x)$  of depth  $\Theta \left(\log \frac{1}{\varepsilon}\right)$ , size  $N_d = \mathcal{O}(L_s^2 \log^2 N_s)$  such that  $|f(x) - \tilde{f}_d(x)| \leq \varepsilon$ ,  $\forall x \in [0,1]$ .

Remarks: (i) The strong convexity requirement can be relaxed: the result obviously holds if the function is strongly concave and it also holds if the function consists of pieces which are strongly convex or strongly concave. (ii) Corollary 12 shows that in the approximation of the same function, the size of the deep neural network  $N_{s}$  is only of polynomially logarithmic order of the size of the shallow neural network  $N_{d}$ , i.e.,  $N_{d} = \mathcal{O}(\mathrm{polylog}(N_{s}))$ . Similar results can be obtained for multivariate functions on the type considered in Section 3.2.

# 5 CONCLUSIONS

In this paper, we have shown that an exponentially large number of neurons are needed for function approximation using shallow networks, when compared to deep networks. The results are established for a large class of smooth univariate and multivariate functions. Our results are established for the case of feedforward neural networks with ReLUs and binary step units.

# REFERENCES

A. Andoni, R. Panigrahy, G. Valiant, and L. Zhang. Learning polynomials with neural networks. In ICML, 2014.  
A. R. Barron. Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information theory, 1993.  
Y. Bengio. Learning deep architectures for ai. Foundations and trends in Machine Learning, 2009.  
C. K. Chui and X. Li. Approximation by ridge functions and neural networks with one hidden layer. Journal of Approximation Theory, 1992.  
G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 1989.  
B. DasGupta and G. Schnitger. The power of approximating: a comparison of activation functions. In NIPS, 1993.  
O. Delalleau and Y. Bengio. Shallow vs. deep sum-product networks. In NIPS, 2011.  
R. Eldan and O. Shamir. The power of depth for feedforward neural networks. arXiv preprint arXiv:1512.03965, 2015.  
K. I. Funahashi. On the approximate realization of continuous mappings by neural networks. Neural networks, 1989.  
A. Gil, J. Segura, and N. M. Temme. Numerical methods for special functions. SIAM, 2007.  
I. J. Goodfellow, D. Warde-Farley, M. Mirza, A. C. Courville, and Y. Bengio. Maxout networks. ICML, 2013.  
K. Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks, 1991.  
K. Hornik, M. Stinchcombe, and H. White. Multilayer feedforward networks are universal approximators. Neural networks, 1989.  
A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.  
T. Poggio, L. Rosasco, A. Shashua, N. Cohen, and F. Anselmi. Notes on hierarchical splines, dclns and i-theory. Technical report, Center for Brains, Minds and Machines (CBMM), 2015.  
M. Telgarsky. Benefits of depth in neural networks. arXiv preprint arXiv:1602.04485, 2016.  
L. Wan, M. Zeiler, S. Zhang, Y. LeCun, and R. Fergus. Regularization of neural networks using dropconnect. In ICML, 2013.  
D. Yarotsky. Error bounds for approximations with deep ReLU networks. arXiv preprint arXiv:1610.01145, 2016.
