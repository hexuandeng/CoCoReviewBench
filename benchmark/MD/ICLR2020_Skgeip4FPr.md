# NEURAL NETWORKS ARE a priori biased towards Boolean Functions WITH LOW ENTROPY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Understanding the inductive bias of neural networks is critical to explaining their ability to generalise. Here, for one of the simplest neural networks - a single-layer perceptron with  $n$  input neurons, one output neuron, and no threshold bias term - we prove that upon random initialisation of weights, the a priori probability  $P(t)$  that it represents a Boolean function that classifies  $t$  points in  $\{0,1\}^n$  as 1 has a remarkably simple form:  $P(t) = 2^{-n}$  for  $0 \leq t < 2^n$ .

Since a perceptron can express far fewer Boolean functions with small or large values of  $t$  (low "entropy") than with intermediate values of  $t$  (high "entropy") there is, on average, a strong intrinsic  $a$ -priori bias towards individual functions with low entropy. Furthermore, within a class of functions with fixed  $t$ , we often observe a further intrinsic bias towards functions of lower complexity. Finally, we prove that, regardless of the distribution of inputs, the bias towards low entropy becomes monotonically stronger upon adding ReLU layers, and empirically show that increasing the variance of the bias term has a similar effect.

# 1 INTRODUCTION

In order to generalise beyond training data, learning algorithms need some sort of inductive bias. The particular form of the inductive bias dictates the performance of the algorithm. For one of the most important machine learning techniques, deep neural networks (DNNs) (LeCun et al., 2015), sources of inductive bias can include the architecture of the networks, e.g. the number of layers, how they are connected, say as a fully connected network (FCN) or as a convolutional neural net (CNN), and the type of optimisation algorithm used, e.g. stochastic gradient descent (SGD) versus full gradient descent (GD). Many further methods such as dropout (Srivastava et al., 2014), weight decay (Krogh & Hertz, 1992) and early stopping (Morgan & Bourlard, 1990) have been proposed as techniques to improve the inductive bias towards desired solutions that generalise well. What is particularly surprising about DNNs is that they are highly expressive and work well in the heavily overparameterised regime where traditional learning theory would predict poor generalisation due to overfitting (Zhang et al., 2016). DNNs must therefore have a strong intrinsic bias that allows for good generalisation, in spite of being in the overparameterised regime.

Here we study the intrinsic bias of the parameter-function map for neural networks, defined in (Valle-Pérez et al., 2018) as the map between a set of parameters and the function that the neural network represents. We define the  $a$ -priori probability  $P(f)$  of a DNN as the probability that a particular function  $f$  is produced upon random sampling (or initialisation) of the weight and threshold bias parameters. A naive null-model might suggest that without further information, one should assume that all functions are equally likely.

However, recent very general arguments (Dingle et al., 2018) based on the coding theorem from AIT (Li et al., 2008) have instead suggested that for a wide range of maps  $M$  that obey a number of conditions such as being simple (they have a low Kolmogorov complexity  $K(M)$ ) and redundant (multiple inputs map to the same output) then if they are sufficiently biased, they will be exponentially biased towards outputs of low Kolmogorov complexity. The parameter-function map of neural networks satisfies these conditions, and it was found empirically (Valle-Pérez et al., 2018) that, as predicted in (Dingle et al., 2018), the probability  $P(f)$  of obtaining a function  $f$  upon random sampling of parameter weights satisfies the following simplicity-bias bound

$$
P (f) \lesssim 2 ^ {- (b \widetilde {K} (f) + a)}, \tag {1}
$$

where  $\widetilde{K}(f)$  is a computable approximation of the true Kolmogorov complexity  $K(f)$ , and  $a$  and  $b$  are constants that depend on the network, but not on the functions.

It is widely expected that real world data is highly structured, and so has a relatively low Kolmogorov complexity (Hinton & Van Camp, 1993; Schmidhuber, 1997). The simplicity bias described above may therefore be an important source of the inductive bias that allows DNNs to generalise so well (and not overfit) in the highly over-parameterised regime (Valle-Pérez et al., 2018).

Nevertheless, this bound has limitations. Firstly, the only rigorously proven result is for the true Kolmogorov complexity version of the bound in the case of large enough  $K(f)$ . Although it has been found to work remarkably well for small systems and computable approximations to Kolmogorov complexity (Valle-Pérez et al., 2018; Dingle et al., 2018), this success is not yet fully understood theoretically. Secondly, it does not explain why models like DNNs are biased; it only explains that, if they are biased, they should be biased towards simplicity. Also, the AIT bound is very general – it predicts a probability  $P(f)$  that depends mainly on the function, and only weakly on the network. It may therefore not capture some variations in the bias that are due to details of the network architecture, and which may be important for practical applications.

For these reasons it is of interest to obtain a finer quantitative understanding of the simplicity bias of neural networks. Some work has been done in this direction, showing that infinitely wide neural networks are biased towards functions which are robust to changes in the input (De Palma et al., 2018), showing that "flatness" is connected to function smoothness (Wu et al., 2016), or arguing that low Fourier frequencies are learned first by a ReLU neural network (Rahaman et al., 2018; Yang & Salman, 2019). All of these papers take some notion of "smoothness" as tractable proxy for the complexity of a function. One generally expects smoother functions to be simpler, although this is clearly a very rough measure of the Kolmogorov complexity.

# 2 SUMMARY OF KEY RESULTS

In this paper we study how likely different Boolean functions, defined as  $f: \{0,1\}^n \to \{0,1\}$ , are obtained upon randomly chosen weights of neural networks. Our key results are aimed at fleshing out with more precision and rigour what the inductive biases of (very) simple neural networks are, and how they arise. We focus our study on a notion of complexity, namely the "entropy,"  $H(f)$ , of a Boolean function  $f$ , defined as the binary entropy of the fraction of possible inputs to  $f$  that  $f$  maps to 1. This quantity essentially measures the amount of class imbalance of the function, and is complementary to previous works studying notions of smoothness as a proxy for complexity.

1. In Section 4 we study a simple perceptron with no threshold bias term, and with weights  $w$  sampled from a distribution which is symmetric under reflections along the coordinate axes. Let the random variable  $T$  correspond to the number of points in  $\{0,1\}^n$  which that fall above the decision boundary of the network (i.e.  $T = |\{x \in \{0,1\}^n : \langle w,x\rangle > 0\}|$ ) upon i.i.d. random initialisation of the weights. We prove that  $T$  is distributed uniformly, i.e.  $P(T = t) = 2^{-n}$  for  $0 \leq t < 2^n$ . Let  $\mathbb{F}_t$  be the set of all functions with  $T = t$  that the perceptron can produce and let  $|\mathbb{F}_t|$  be its size (cf. Definition 3.4). We expect  $|\mathbb{F}_t|$  for  $t \sim 2^{n-1}$  (high entropy) to be (much) larger than  $|\mathbb{F}_t|$  for extreme values of  $t$  (low entropy). The average probability of obtaining a particular function  $f$  which maps  $t$  inputs to 1 is  $2^{-n}/|\mathbb{F}_t|$ . The perceptron therefore shows a strong bias towards functions with low entropy, in the sense that individual functions with low entropy have, on average, higher probability than individual functions with high entropy.  
2. In Section 4.3, we show that within the sets  $\mathbb{F}_t$ , there is a further bias, and in some cases this is clearly towards simple functions which correlates with Lempel-Ziv complexity (Lempel & Ziv, 1976; Dingle et al., 2018), as predicted in (Valle-Perez et al., 2018).  
3. In Section 4.4, we show that adding a threshold bias term to a perceptron significantly increases the bias towards low entropy.  
4. In Section 5.1, we provide a new expressivity bound for Boolean functions: DNNs with input size  $n$ ,  $l$  hidden layers each with width  $n + 2^{n - 1 - \log_2l} + 1$  and a single output neuron can express all  $2^{2^n}$  Boolean functions over  $n$  variables.

5. In Section 5.2 we generalise our results to neural networks with multiple layers, proving (in the infinite-width limit) that the bias towards low entropy increases with the number of ReLU-activated layers.

# 3 DEFINITIONS, TERMINOLOGY, AND NOTATION

Definition 3.1 (DNNs). Fully connected feed-forward neural networks with activations  $\sigma$  and a single output neuron form a parameterised function family  $f(x)$  on inputs  $x\in \mathbb{R}^n$ . This can be defined recursively, for  $L$  hidden layers for  $1\leq l\leq L$ , as

$$
f (x) = \mathbf {1} \left(h ^ {(L + 1)} (x)\right),
$$

$$
h ^ {(l + 1)} (x) = w _ {l} \sigma \left(h ^ {(l)}\right) + b _ {l},
$$

$$
h ^ {(1)} (x) = w _ {0} x + b _ {0},
$$

where  $\mathbf{1}(X)$  is the Heaviside step function defined as 1 if  $X > 0$  and 0 otherwise, and  $\sigma$  is an activation function that acts element-wise. The  $w_{l} \in \mathbb{R}^{n_{L + l} \times n_{l}}$  are the weights, and  $b_{l} \in \mathbb{R}^{n_{l + 1}}$  are the threshold bias weights at layer  $l$ , where  $n_{l}$  is the number of hidden neurons in the  $l$ -th layer.  $n_{L + 1}$  is the number of outputs (1 in this paper), and  $n_{0}$  is the dimension of the inputs (which we will also refer to as  $n$ ).

We will refer to the whole set of parameters  $(w_{l}$  and  $b_{l}, 1 \leq l \leq L)$  as  $\theta$ . In the case of perceptrons we use  $f_{\theta}(x) = \sigma (\langle w,x\rangle + b)$  to specify a network. We define the parameter-function map as in (Valle-Pérez et al., 2018) below.

Definition 3.2 (Parameter-function map). Consider a parameterised supervised model, and let the input space be  $\mathbb{X}$  and the output space be  $\mathbb{Y}$ . The space of functions the model can express is  $\mathcal{F} \subset \mathbb{Y}^{|\mathbb{X}|}$ . If the model has  $p$  real valued parameters, taking values within a set  $\Theta \subseteq \mathbb{R}^p$ , the parameter function map  $\mathcal{M}$  is defined

$$
\mathcal {M}: \Theta \to \mathbb {F}
$$

$$
\theta \mapsto f _ {\theta}
$$

where  $f_{\theta}$  is the function corresponding to parameters  $\theta$

In this paper we are interested in the Boolean functions that neural networks express. We consider the 0-1 Boolean hypercube  $\{0,1\}^n$  as the input domain.

Definition 3.3. The function  $\mathcal{T}(f)$  is defined as the number of points in the hypercube  $\{0,1\}^n$  that are mapped to 1 by the action of a neural network  $f$ .

For example, for a perceptron this function is defined as,

$$
\mathcal {T} (f) = \mathcal {T} (w, b) = \sum_ {x \in \{0, 1 \} ^ {n}} \mathbf {1} (\langle x, w \rangle + b). \tag {2}
$$

We will sometimes use  $\mathcal{T}(w,b)$  if the neural network is a perceptron.

Definition 3.4 ( $\mathbb{F}_t$  and  $P(t)$ ). We define the set  $\mathbb{F}_t$  to be the set of functions which all have the same value of  $\mathcal{T}(f)$ ,

$$
\mathbb {F} _ {t} = \{f | \mathcal {T} (f) = t \}
$$

Given a probability measure  $P$  on the weights  $\theta$ , we define the probability measure

$$
P (T = t) := P (\theta : f _ {\theta} \in \mathbb {F} _ {t})
$$

We can also define  $\mathcal{T}(f)$  and  $P(T = t)$  in the natural way for sets of input points other than  $\{0,1\}^n$ , the context making clear what definition is being used.

Definition 3.5. The entropy  $H(f)$  of a Boolean function  $f: \{0,1\}^* \to \{0,1\}$  is defined as  $H(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$ . It is the binary entropy of the fraction  $p$  of possible inputs to  $f$  that  $f$  maps to 1 or equivalently, the binary entropy of the fraction of 1's in the right-hand column of the truth table of  $f$ .

Definition 3.6. We define the Boolean complexity  $K_{\mathrm{Bool}}(f)$  of a function  $f$  as the number of binary connectives in the shortest Boolean formula that expresses  $f$ .

# 4 INTRINSIC BIAS IN A PERCEPTRON'S PARAMETER-FUNCTION MAP

In this section we study the parameter-function map of the perceptron (Rosenblatt, 1958), in many ways the simplest neural network. While it famously cannot express many Boolean functions – including XOR – it remains an important model system. Moreover, many DNN architectures include layers of perceptrons, so understanding this very basic architecture may provide important insight into the more complex neural networks used today.

# 4.1 ENTROPY BIAS IN A SIMPLE PERCEPTRON WITH  $b = 0$  (NO THRESHOLD BIAS TERM)

Here we consider perceptrons  $f_{\theta}(x) = \mathbf{1}(\langle w,x\rangle +b)$  without threshold bias terms, i.e.  $b = 0$ .

The following theorem shows that under certain conditions on the weight distribution, a perceptron with no threshold bias has a uniform  $P(\theta : \mathcal{T}(f_{\theta}) = t)$ . The class of weight distributions includes the commonly used isotropic multivariate Gaussian with zero mean, a uniform distribution on a centred cuboid, and many other distributions. The full proof of the theorem is in Appendix A.

Theorem 4.1. For a perceptron  $f_{\theta}$  with  $b = 0$  and weights  $w$  sampled from a distribution which is symmetric under reflections along the coordinate axes, the probability measure  $P(\theta : \mathcal{T}(f_{\theta}) = t)$  is given by

$$
P (\theta : \mathcal {T} (f _ {\theta}) = t) = \left\{ \begin{array}{l l} 2 ^ {- n} & i f 0 \leq t <   2 ^ {n} \\ 0 & o t h e r w i s e \end{array} \right..
$$

Proof sketch. We consider the sampling of the normal vector  $w$  as a two-step process: we first sample the absolute values of the elements, giving us a vector  $w_{\mathrm{pos}}$  with positive elements, and then we sample the signs of the elements. Our assumption on the probability distribution implies that each of the  $2^n$  sign assignments is equally probable, each happening with a probability  $2^{-n}$ . The key of the proof is to show that for any  $w_{\mathrm{pos}}$ , each of the sign assignments gives a distinct value of  $T$  (and because there are  $2^n$  possible sign assignments, for any value of  $T$ , there is exactly one sign assignment resulting in a normal vector with that value of  $T$ ). This implies that, provided all sign assignments of any  $w_{\mathrm{pos}}$  are equally likely, the distribution on  $T$  is uniform.

A consequence of Theorem 4.1 is that the average probability of the perceptron producing a particular function  $f$  with  $\mathcal{T}(f) = t$  is given by

$$
\langle P (f) \rangle_ {t} = \frac {2 ^ {- n}}{| \mathbb {F} _ {t} |}, \tag {3}
$$

where  $\mathbb{F}_t$  denotes the set of Boolean functions that the perceptron can express which satisfy  $\mathcal{T}(f) = t$ , and  $\langle \cdot \rangle_t$  denotes the average (under uniform measure) over all functions  $f\in \mathbb{F}_t$ .

We expect  $|\mathbb{F}_t|$  to be much smaller for more extreme values of  $t$ , as there are fewer distinct possible functions with extreme values of  $t$ . This would imply a bias towards low entropy functions. By way of an example,  $|\mathbb{F}_0| = 1$  and  $|\mathbb{F}_1| = n$  (since the only Boolean functions  $f$  a perceptron can express which satisfy  $\mathcal{T}(f) = 1$  have  $f(x) = 1$  for a single one-hot  $x \in \{0,1\}^n$ ), implying that  $\langle P(f)\rangle_0 = 2^{-n}$  and  $\langle P(f)\rangle_1 = 2^{-n}/n$ .

Nevertheless, the probability of functions within a set  $\mathbb{F}_t$  is unlikely to be uniform. We find that, in contrast to the overall entropy bias, which is independent of the shape of the distribution (as long as it satisfies the right symmetry conditions), the probability  $P(f)$  of obtaining function  $f$  within a set  $\mathbb{F}_t$  can depend on distribution shape. Nevertheless, for a given distribution shape, the probabilities  $P(f)$  are independent of scale of the shape, e.g. they are independent of the variance of the Gaussian, or the width of the uniform distribution. This is because the function is invariant under scaling all weights by the same factor (true only in the case of no threshold bias). We will address the probabilities of functions within a given  $\mathbb{F}_t$  further in Section 4.3.

# 4.2 SIMPLICITY BIAS OF THE  $b = 0$  PERCEPTRON

The entropy bias of Theorem 4.1 entails an overall bias towards low Boolean complexity. In Theorem B.1 in Appendix B we show that the Boolean complexity of a function  $f$  is bounded by

$$
K _ {\text {B o o l}} (f) <   2 \times n \times \min  (\mathcal {T} (f), 2 ^ {n} - \mathcal {T} (f)). \tag {4}
$$

Using Theorem 4.1 and Equation (4), we have that the probability that a randomly initialised perceptron expresses a function  $f$  of Boolean complexity  $k$  or greater is upper bounded by

$$
P \left(K _ {\operatorname {B o o l}} (f) \geq k\right) <   1 - \frac {k \times 2 ^ {- n} \times 2}{2 \times n} = 1 - \frac {k}{2 ^ {n} \times n}. \tag {5}
$$

Uniformly sampling functions would result in  $P(K_{\mathrm{Bool}}(f) \geq k) \approx 1 - 2^{k - 2^n}$  which for intermediate  $k$  is much larger than Equation (5). Thus from entropy bias alone, we see that the perceptron is much more likely to produce simple functions than complex functions: it has an inductive bias towards simplicity. This derivation is complementary to the AIT arguments from simplicity bias (Dingle et al., 2018; Valle-Pérez et al., 2018), and has the advantage that it also proves that bias exists, whereas AIT-based simplicity bias arguments presuppose bias.

To empirically study the inductive bias of the perceptron with  $b = 0$ , we sampled over many random initialisations with weights drawn from Gaussian or uniform distributions and input size  $n = 7$ . As can be seen in Figure 1a and Figure 1b, the probability  $P(f)$  that function  $f$  obtains varies over many orders of magnitude. Moreover, there is a clear simplicity bias upper bound on this probability, which, as as predicted by Eq. 1, decreases with increasing Lempel-Ziv complexity  $(K_{LZ}(f))$  (using a version from (Dingle et al., 2018)). Similar behaviour was observed in (Valle-Pérez et al., 2018) for a FCN network. Moreover it was also shown there that Lempel-Ziv complexity for these Boolean functions correlates with approximations to the Boolean complexity  $K_{\mathrm{Bool}}$ . A one-layer neural network (Figure 1c) shows stronger bias than the perceptron, which may be expected because the former has a much larger expressivity. A rough estimate of the slope  $a$  in Eq. 1 from (Dingle et al., 2018) suggests that  $a \sim \log_2(N_O) / \max_{f \in \mathbb{O}}(\tilde{K}(f))$  where  $\mathbb{O}$  is the set of all Boolean functions the model can produce, and  $N_O$  is the number of such functions. The maximum  $K(f)$  may not differ that much between the one layer network and the perceptron, but  $N_O$  will be much larger in former than in the latter.

In Appendix D we also show rank plots for the networks from Figure 1. Interestingly, at larger rank, they all show a Zipf like power-law decay, which can be used to estimate  $N_O$ , the total number of Boolean functions the network can express. We also note that the rank plots for the perceptron with  $b = 0$  with Gaussian or uniform distributions of weights are nearly indistinguishable, which may be because the overall rank plot is being mainly determined by the entropy bias.

![](images/775f5b491785b15f751e6ff4d872d5c682318886453e180f0f383fd88e48d793.jpg)  
(a) Perceptron: Gaussian weights

![](images/a7ee4e44106110c44f87b79c14f4431db84a966701ac7803689d86b2b9f93de4.jpg)  
Figure 1: Probability  $P(f)$  that a function obtains upon random choice of parameters versus Lempel Ziv complexity  $K_{LZ}(f)$  for (a) an  $n = 7$  perceptron with  $b = 0$  and weights sampled from a Gaussian distributions, (b) an  $n = 7$  perceptron with  $b = 0$  and weights sampled from a uniform distribution centred at 0 and (c) a 1-hidden layer neural network (with 64 neurons in the hidden layer). Weights  $w$  and the threshold bias terms are sampled from  $\mathcal{N}(0,1)$ . For all cases  $10^{8}$  samples were taken and frequencies less than 2 were eliminated to reduce finite sampling effects. We present the graphs with the same scale for ease of comparison.

![](images/482e785716553720af26bdfe6fb8c32e75eb44f92708539b17f92faf2b3ab46c.jpg)  
(b) Perceptron: uniform weights.  
(c) 1 layer NN

# 4.3 BIAS WITHIN  $\mathbb{F}_t$

In Figure 2 we compare a rank plot for all functions expressed by an  $n = 7$  perceptron with  $b = 0$  to the rank plots for functions with  $\mathcal{T}(f) = 47$  and  $\mathcal{T}(f) = 64$ . The highest probability functions in  $\mathbb{F}_{64}$  have higher probability than the highest in  $\mathbb{F}_{47}$  because the former allows for simpler functions (such as 0101..), but for both sets, the maximum probability is still considerably lower than the maximum probability functions overall.

In Appendix E we present further empirical data that suggests that these probabilities are bounded above by Lempel-Ziv complexity (in agreement with (Valle-Pérez et al., 2018)). However, in contrast to Theorem 4.1 which is independent of the parameter distribution (as long as they are symmetric), the distributions within  $\mathbb{F}_t$  are different for the Gaussian and uniform parameter distributions, with the latter showing less simplicity bias within a class of fixed  $t$  (see Appendix E.1).

![](images/69885e130742d60a0ee2c64c90681cf830d9dc15b877bc5b4c48e4ecb18c910b.jpg)  
(a) All functions

![](images/6e91c853fc033972021aada259a174bb3112b1031ca5f2b1284ed3f67016c3e0.jpg)  
(b)  $\mathcal{T}(f) = 47$  
Figure 2: Probability  $P(f)$  vs rank for functions for a perceptron with  $n = 7$ ,  $\sigma_b = 0$ , and weights sampled from independent Gaussian distributions. In Figures 2b and 2c the functions are ranked within their respective  $\mathbb{F}_t$ . The seven highest probability functions in Figure 2c are  $f = 0101 \ldots$  and equivalent functions obtained by permuting the input dimensions – note that these are very simple functions (simpler than the simplest functions that satisfy  $\mathcal{T}(f) = 47$ ).

![](images/1f2eeffe6e463e3410e82c0e99e55a23a26c13db35bb392368997c63fd938167.jpg)  
(c)  $\mathcal{T}(f) = 64$

In Appendix F, we give further arguments for simplicity bias, based on the set of constraints that needs to be satisfied to specify a function. Every function  $f$  can be specified by a minimal set of linear conditions on the weight vector of the perceptron, which correspond to the boundaries of the cone in weight space producing  $f$ . The Kolmogorov complexity of conditions should be close to that of the functions they produce as they are related to the functions in a one-to-one fashion, via a simple procedure. In Appendix F, we focus on conditions which involve more than two weights, and show that within each set  $\mathbb{F}_t$  there exists one function with as few as 1 such conditions, and that there exists a function with as many as  $n - 2$  such conditions. We also compute the set of necessary conditions (up to permutations of the axes) explicitly for functions with small  $t$ , and find that the range in the number and complexity of the conditions appears to grow with  $t$ , in agreement, with what we observe in Figure 2 for the range of complexities. More generally, we find that complex functions typically need more conditions than simple functions do. Intuitively, the more conditions needed to specify a function, the smaller the volume of parameters that can generate the function, so the lower its a-priori probability.

# 4.4 EFFECT OF  $b$  (THE THRESHOLD BIAS TERM) ON  $P(t)$

We next study the behaviour of the perceptron when we include the threshold bias term  $b$ , sampled from  $\mathcal{N}(0, \sigma_b)$ , while still initialising the weights from  $\mathcal{N}(0, 1)$ , as in Section 4.1. We present results for  $n = 7$  in Figure 3. Interestingly, for infinitesimal  $\sigma_b$ ,  $P(T = 0)$  is less than for  $b = 0$  (See Appendix C), but then for increasing  $\sigma_b$  it rapidly grows larger than  $1/2^n$  and in the limit of large  $\sigma_b$  asymptotes to  $1/2$  (see Figure 3b). It's not hard to see where this asymptotic behaviour comes from, a large positive or negative  $b$  means all inputs are mapped to true (1) or false (0) respectively.

![](images/0688f99bcbb73dac7db1ce61d7541b421ef61b50ac89919b01808c6131025ff9.jpg)  
(a)  $\mathrm{P(t)}$  at selected values of  $\sigma_{b}$

![](images/1eb8ab407692f8f09d81f28c879f2e0365a061aaf982ab84f0d112d18dd289c1.jpg)  
(b)  $\mathrm{P(t = 0)}$  
Figure 3: Effect of adding a bias term sampled from  $\mathcal{N}(0, \sigma_b)$  to a perceptron with weights sampled from  $\mathcal{N}(0, 1)$ . (a) Increasing  $\sigma_b$  increases the bias against entropy, and with a particular strong bias towards  $t = 0$  and  $t = 2^n$ . (b)  $P(t = 0)$  increases with  $\sigma_b$  and asymptotes to  $1/2$  in the limit  $\sigma_b \to \infty$ .

# 5 ENTROPY BIAS IN MULTI-LAYER NEURAL NETWORKS

We next extend results from Section 4 to multi-layer neural networks, with the aim to comment on the behaviour of  $P(T = t)$  as we add hidden layers with ReLU activations.

To study the bias in the parameter-function map of neural networks, it is important to first understand the expressivity of the networks. In Section 5.1, we produce a (loose) upper bound on the minimum size of a network with ReLU activations and  $l$  layers that is maximally expressive over Boolean functions. We comment on how sufficiently large expressivity implies a larger bias towards low entropy for models with similarly shaped distribution over  $T$  (when compared to the perceptron).

In Section 5.2, we prove, in the limit of infinite width, that adding ReLU activated layers causes the moments of  $P(T = t)$  to increase, . This entails a lower expected entropy for neural networks with more hidden layers. We empirically observe that the distribution of  $T$  becomes convex (with input  $\{0,1\}^n$ ) with the addition of ReLU activated layers for neural networks with finite width.

# 5.1 EXPRESSIVITY CONDITIONS FOR DNNS

We provide upper bounds on the minimum size of a DNNs that can model all Boolean functions. We use the notation  $\langle n_0, n_1, \ldots, n_L, n_{L+1} \rangle$  to denote a neural network with ReLU activations and of the form given in Definition 3.1.

Lemma 5.1. A neural network with layer sizes  $\langle n,2^{n - 1},1\rangle$ , threshold bias terms, and ReLU activations can express all Boolean functions over  $n$  variables (also found in (Raj, 2018)). See Appendix G for proof.

Lemma 5.2. A neural network with  $l$  hidden layers, layer sizes  $\langle n,(n + 2^{n - 1} / l + 1),\ldots ,(n + 2^{n - 1} / l + 1),1\rangle$ , threshold bias terms, and ReLU activations can express all Boolean functions over  $n$  variables. See Appendix G for proof.

Note that neither of these bounds are (known to be) tight. Lemma 5.1 says that a network with one hidden layer of size  $2^{n-1}$  can express all Boolean functions over  $n$  variables. We know that a perceptron with  $n$  input neurons (and a threshold bias term) can express at most  $2^{n^2}$  Boolean functions ((Anthony, 2001), Theorem 4.3), which is significantly less than the total number of Boolean functions over  $n$  variables, which is  $2^{2^n}$ . Hence there is a very large number of Boolean functions that the network with a (sufficiently wide) hidden layer can express, but the perceptron cannot. The vast majority of these functions have high entropy (as almost all Boolean functions do). Moreover, we observe that the measure  $P(T = t)$  is convex in the case of the more expressive neural networks, as discussed in section Section 5.2. This suggests that the networks with hidden layers have a much stronger relative bias towards low entropy functions than the perceptron does, which is also consistent with the stronger simplicity bias found in Figure 1.

We further observe from Lemma 5.2 that the number of neurons can be kept constant and spread over multiple layers without loss of expressivity for a Boolean classifier (provided the neurons are evenly spread across the layers).

# 5.2 HOW MULTIPLE LAYERS AFFECT THE BIAS

We next consider the effect of addition of ReLU activated layers on the distribution  $P(t)$ . Of course adding even just one layer hugely increases expressivity over a perceptron. Therefore, even if the distribution of  $P(t)$  would not change, the average probability of functions for in a given  $\mathbb{F}_t$  could drop significantly due to the increase in expressivity.

However, we observe that for inputs  $\{0,1\}^n$ ,  $P(t)$  becomes more convex when more ReLU-activated hidden layers are added, see Figure 4. The distribution appears to be monotone on either side of  $t = 2^{n - 1}$  and relatively flat in the middle, even with the addition of 8 intermediate layers<sup>2</sup>. In particular, we show in Figure 4 that for large number of layers, or large  $\sigma_{b}$ , the probabilities for  $P(t = 0)$  (and by symmetry, in the infinite width limit, also  $P(t = 2^n)$ ) each asymptotically reach  $\frac{1}{2}$ , and thus take up the vast majority of the probability weight.

We now prove some properties of the distribution  $P(t)$  for DNNs with several layers.

![](images/4c532d3e7e05791b77a16b407567787b1cb6a3cb997513a0e4b711c7e5f32cb8.jpg)  
(a)  $\{0,1\}^7,\sigma_b = 0.0$  , number of layers varied

![](images/13007adb193ad630638ebe0180695e20bc64221fd7c4ce1e23aeed19c3795b4f.jpg)  
(b)  $\{0,1\}^7,\sigma_b = 1.0$  , number of layers varied

![](images/bfebb4d304326c8b7b88029be97b34e2d1b6a65905ed6711467322194c324b29.jpg)  
(c)  $\{0,1\} ^7$  1 layer,  $\sigma_{b}$  varied

![](images/73a2acb01fdd03edfda2c74f969bb7535e22fcbc5b67b18d930e100eb0639032.jpg)  
(d)  $\{-1,1\}^7,\sigma_b = 0.0$  , number of layers varied  
Figure 4:  $\mathbf{P}(\mathbf{T} = \mathbf{t})$  becomes on average more biased towards low entropy for increasing number of layers or increasing  $\sigma_{\mathrm{b}}$ . Here we use  $n = 7$  input layers, with input  $\{0,1\}^7$  (centered data) or  $\{-1,1\}^7$  (uncentered data). The hidden layers are of width  $2^{n - 1} = 64$  to guarantee full expressivity.  $\sigma_w = 1.0$  in all cases. The insets show how  $P(t = 0)$  asymptotes to  $\frac{1}{2}$  with increasing layers or  $\sigma_b$ .

Lemma 5.3. The probability distribution on  $T$  for inputs in  $\{0,1\}^n$  of a neural network with linear activations and i.i.d. initialisation of the weights is independent of the number of layers and the layer widths, and is equal to the distribution of a perceptron. See Appendix H for proof.

While it is trivial that such a linear network has the same expressivity as a perceptron, it may not be obvious that the entropy bias is identical.

Lemma 5.4. Applying a ReLU function in between each layer produces a lower bound on  $P(T = 0)$  such that  $P(T = 0) \geq 2^{-n}$ . See Appendix H for proof.

This lemma shows that a DNN with ReLU functions is no less biased towards the lowest entropy function than a perceptron is. We prove a more general result in the following theorem which concerns the behaviour of the average entropy  $\langle H(t)\rangle$  (where the average upon random sampling of parameters) as the number of layers grows. The theorem shows that the bias towards low entropy becomes stronger as we increase the number of layers, for any distribution of inputs. We rely on previous work that shows that in the infinite width limit, neural networks approach a Gaussian process (Lee et al. (2018); Garriga-Alonso et al. (2018); Novak et al. (2018)), which for the case of fully-connected ReLU networks, has an analytic form (Lee et al., 2018).

Theorem 5.5. Let  $\mathbb{S}$  be a set of  $m = |\mathbb{S}|$  input points in  $\mathbb{R}^n$ . Consider neural networks with i.i.d. Gaussian weights with variances  $\sigma_w^2 / \sqrt{n}$  and biases with variance  $\sigma_b$ , in the limit where the width of all hidden layers  $n$  goes to infinity. Let  $N_1$  and  $N_2$  be such neural networks with  $L$  and  $L + 1$  infinitely wide hidden layers, respectively, and no bias. Then, the following holds:  $\langle H(T)\rangle$  is smaller than or equal for  $N_2$  than for  $N_1$ . It is strictly smaller if there exist pairs of points in  $\mathbb{S}$  with correlations less than 1. If the networks have sufficiently large threshold bias ( $\sigma_b > 1$  is a sufficient condition), the result above also holds. For smaller bias, the result holds only for a sufficiently large number of layers.

See Appendix H for a proof of Theorem 5.5. We show in Figure 4 that when  $\sigma_{b} = 0$ , the bias towards low entropy indeed becomes monotonically stronger as we increase the number of ReLU layers, for both inputs in  $\{0,1\}^n$  as well as for centered data  $\{-1,1\}^n$ .

For centered inputs  $\{-1,1\}^n$ , the perceptron with  $b = 0$  shows rather unusual behaviour. The distribution is completely peaked around  $t = 2^{n - 1}$  because every input mapping to 1 has the opposite input mapping to 0. Not surprisingly, its expressivity is much lower than the equivalent perceptron with  $\{0,1\}^n$  (as can be seen in Figure 6 in Appendix D). Nevertheless, in Figure 4d we see that as the number of layers increases, the behaviour rapidly resembles that of uncentered data.

Theorem 5.5 and Figure 4 also suggest that adding more layers may improve generalisation when learning low entropy functions, which is necessary for classification of highly class imbalanced datasets. However, to further comment on the effect of adding more layers on the inductive bias from the parameter-function map we would need more information about the bias within a set  $\mathbb{F}_t$ .

The insets of in Figure 4 show that the two trivial functions asymptotically dominate in the limit of large numbers of layers. We note that recent work ((Lee et al., 2018; Luther & Seung, 2019)) has also pointed out that for fully-connected ReLU networks in the infinite-width infinite-depth limit, all inputs become asymptotically correlated, so that the networks will tend to compute the constant function. Here we give a quantitative characterisation of this phenomenon for any number of layers.

some interesting recent work (Yang & Salman, 2019) has shown that certain choices of network hyperparameters lead to networks which are a priori unbiased, that is the  $P(f)$  appears to be uniform. In particular, networks with erf activations with sufficiently large weight variance  $\sigma_w$  and depth assign equal a priori probability to every (real-valued) function they can produce. At first sight this is surprising, given the robust entropy bias we derive here for a perceptron. Nevertheless, we replicate this result for Boolean functions produced by tanh activations in Appendix I, and show that the hyperparameters chosen by (Yang & Salman, 2019) lie deep in the chaotic region defined in (Poole et al., 2016) which explains the observed lack of bias: In this regime inputs are completely uncorrelated, and so the outputs will be uniform in 1's and 0's, which is equivalent to randomly choosing a Boolean function. If there is not bias, then of course there can also be no simplicity bias. As a sanity check, we show that the bias is recovered for the  $\sigma_w$  used in (Yang & Salman, 2019) if  $\sigma_b$  is made large enough so that the network is in the ordered regime.

# 6 DISCUSSION AND FUTURE WORK

In Section 4 we have proven the existence of an intrinsic bias towards Boolean functions of low entropy in a perceptron with no threshold bias term, such that  $P(T = t) = 2^{-n}$  for  $0 \leq t < 2^n$ .

This result puts an upper bound on the probability that a perceptron with no threshold bias term will be initialised to a Boolean function with at least a certain Boolean complexity. Adding a threshold term in general increases the bias towards low entropy.

We also study how the entropy bias is affected by adding a threshold bias term or ReLU-activated hidden layers. In Section 5 we show that adding layers to a feed-forward neural network with ReLU activations makes the bias towards low entropy stronger. We also show empirically that the bias towards low entropy functions is further increased when a threshold bias term with high enough variance is added. Recently, (Luther & Seung, 2019) have argued that batch normalisation (Ioffe & Szegedy, 2015) makes ReLU networks less likely to compute the constant function (which has also been experimentally shown in (Page, 2019)). If batch norm increases the probability of high entropy functions, it could help explain why batch norm improves generalisation for (typically class balanced) datasets. We leave further exploration of the effect of batch normalisation on a-priori bias to future work.

Simplicity bias within the set of constant  $t$  functions  $\mathbb{F}_t$  is affected by the choice of initialisation, even when the entropy bias is unaffected. This indicates that there are further properties of the parameter-function map that lead to a simplicity bias. In Section 4.3, we suggest that the complexity of the conditions on  $w$  producing a function should correlate with the complexity of the function, and we conjecture that more complex conditions correlate with a lower probability. At present we do not have an analytic proof for this, but hope to fully characterise the probability-complexity relation in future work.

We note that the a priori inductive bias we study here is for a randomly initialised network. If a network is trained on data, then the optimisation procedure (for example SGD) may introduce further biases. The null model against which such further biasing should be assessed is probably the a priori bias from random initialisation of weights. We conjecture that the entropy bias that we observe here is so strong, that any optimisation algorithm (including SGD) should be strongly affected by this a priori bias of the network itself.

Simplicity bias in neural networks (Valle-Pérez et al., 2018) offers an explanation of why DNNs work in the highly overparameterised regime. DNNs can express an unimaginably large number of functions that will fit the training data, but almost all of these will give extremely poor generalisation. Simplicity bias, however, means that a DNN will preferentially choose low complexity functions, which should give better generalisation. Here we have shown some examples where changing hyperparameters can affect the bias further. This raises the possibility of explicitly designing biases to optimise a DNN for a particular problem.

# REFERENCES

Martin Anthony. Discrete mathematics of neural networks: selected topics, volume 8. Siam, 2001.  
Giacomo De Palma, Bobak Toussi Kiani, and Seth Lloyd. Deep neural networks are biased towards simple functions. arXiv preprint arXiv:1812.10156, 2018.  
Kamaludin Dingle, Chico Q Camargo, and Ard A Louis. Input-output maps are strongly biased towards simple outputs. Nature communications, 9(1):761, 2018.  
Adria Garriga-Alonso, Carl Edward Rasmussen, and Laurence Aitchison. Deep convolutional networks as shallow gaussian processes. arXiv preprint arXiv:1808.05587, 2018.  
Geoffrey Hinton and Drew Van Camp. Keeping neural networks simple by minimizing the description length of the weights. In in Proc. of the 6th Ann. ACM Conf. on Computational Learning Theory. Citeseer, 1993.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Anders Krogh and John A Hertz. A simple weight decay can improve generalization. In Advances in neural information processing systems, pp. 950-957, 1992.  
Yann LeCun, Joshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436, 2015.

Jaehoon Lee, Jascha Sohl-dickstein, Jeffrey Pennington, Roman Novak, Sam Schoenholz, and Yasaman Bahri. Deep neural networks as gaussian processes. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=B1EA-M-0Z.  
Abraham Lempel and Jacob Ziv. On the complexity of finite sequences. IEEE Transactions on information theory, 22(1):75-81, 1976.  
Ming Li, Paul Vitányi, et al. An introduction to Kolmogorov complexity and its applications, volume 3. Springer, 2008.  
Kyle Luther and H Sebastian Seung. Variance-preserving initialization schemes improve deep network training: But which variance is preserved? arXiv preprint arXiv:1902.04942, 2019.  
Nelson Morgan and Hervé Bourlard. Generalization and parameter estimation in feedforward nets: Some experiments. In Advances in Neural Information Processing Systems, pp. 630-637, 1990.  
Roman Novak, Lechao Xiao, Jaehoon Lee, Yasaman Bahri, Daniel A Abolafia, Jeffrey Pennington, and Jascha Sohl-Dickstein. Bayesian convolutional neural networks with many channels are gaussian processes. arXiv preprint arXiv:1810.05148, 2018.  
David Page. How to train your resnet 7: Batch norm, 2019. URL https://myrtle.ai/how-to-train-your-resnet-7-batch-norm/.  
Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, and Surya Ganguli. Exponential expressivity in deep neural networks through transient chaos. In Advances in neural information processing systems, pp. 3360-3368, 2016.  
Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. arXiv preprint arXiv:1806.08734, 2018.  
Bhiksha Raj. Neural networks: What can a network represent, 2018. URL http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Spring.2018/www/slides/lec2.universal.pdf.  
Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review, 65(6):386, 1958.  
Jürgen Schmidhuber. Discovering neural nets with low kolmogorov complexity and high generalization capability. Neural Networks, 10(5):857-873, 1997.  
Samuel S Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=H1W1UN9gg.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Guillermo Valle-Pérez, Chico Q Camargo, and Ard A Louis. Deep learning generalizes because the parameter-function map is biased towards simple functions. arXiv preprint arXiv:1805.08522, 2018.  
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.  
Greg Yang and Hadi Salman. A fine-grained spectral perspective on neural networks. arXiv preprint arXiv:1907.10599, 2019.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.
