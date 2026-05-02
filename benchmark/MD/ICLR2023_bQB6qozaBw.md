# INFORMATION PLANE ANALYSIS FOR DROPOUT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The information theoretic framework promises to explain the predictive power of neural networks. In particular, the information plane analysis, which measures mutual information (MI) between input and representation as well as representation and output, should give rich insights into the training process. This approach, however, was shown to strongly depend on the choice of estimator of the MI: measuring discrete MI does not capture the nature of deterministic neural networks and continuous data distributions, and different approaches for discretization arbitrarily change results. On the other hand, measuring continuous MI for a deterministic network is not mathematically meaningful. In this work we show how the stochasticity induced by dropout layers can be utilized to estimate MI in a theoretically sound manner. We demonstrate in a range of experiments that this approach enables a meaningful information plane analysis for the large class of dropout neural networks that is widely used in practice.

# 1 INTRODUCTION

The information bottleneck hypothesis for deep learning conjectures two phases of training feedforward neural networks (Shwartz-Ziv and Tishby, 2017): the fitting phase and the compression phase. The first corresponds to extracting information from the input into the learned representations, so that the mutual information (MI) between inputs and hidden representations grows. The latter corresponds to forgetting the information that is not needed to predict the target, reflected in a decrease of the MI between hidden representations and inputs, while MI between representations and targets stays the same or grows. The phases can be observed via an information plane analysis, i.e., by analyzing the development of MI between inputs and representations in contrast to the MI between representations and targets during training (see Fig. 1 for an example). For an overview we refer the reader to Geiger (2021).

While being elegant and plausible, this hypothesis is challenging to investigate empirically. As shown by Amjad and Geiger (2019), the MI of a deterministic neural network is infinite, if the input distribution is continuous. The standard approach is therefore to assume the input distribution to be discrete (e.g., equivalent to the empirical distribution of the dataset  $S$  at hand) and to discretize the real-valued hidden representations by binning to allow for non-trivial measurements, i.e. to avoid that the MI always takes the maximum value of  $\log(|S|)$  (Shwartz-Ziv and Tishby, 2017). In this discrete setting the MI theoretically gets equivalent to the Shannon entropy of the hidden representations and thus MI decrease is equivalent to geometrical compression (Basirat et al., 2021). Moreover, the binning-based estimate highly depends on the chosen bin size (Ross, 2014). To instead work with continuous input distributions, Goldfeld et al. (2019) suggest to replace deterministic neural networks by stochastic ones via adding Gaussian noise to each of the hidden representations. This kind of stochastic networks is rarely used in practice, which limits the insights brought by the analysis. However, another class of stochastic neural networks, namely those using dropout, is heavily used in practice due to their effective regularizing properties.

This leads to the core questions investigated in this work: Can we obtain accurate and meaningful MI estimates in neural networks with dropout noise? And if so, do information planes built for dropout networks confirm the information bottleneck hypothesis? To that end, our main contributions are the following: We present a theoretical analysis showing that binary dropout noise does not introduce sufficient stochasticity that allows us to estimate the MI between inputs and representations. At the

![](images/f0fc0da3b4b953484b31c16a5caed0446fc503858159a4a242eb51f79db03981.jpg)  
(a) Information dropout

![](images/40817bb70e39829f8f586a2083f647cff95404dbd2635c056b8807977cb88714.jpg)  
Figure 1: Information planes for the representation with information and Gaussian dropout layer of a LeNet network. Compared to an information plane analysis based on a (discrete) binning estimation of the MI, our approach (both with Gaussian and information dropout) clearly shows compression.  
(b) Gaussian dropout (our)

![](images/e0a0da065021738d0a96e9d7e9efe34ad8109c79a26742745a184e55bcd3586f.jpg)  
(c) Gaussian dropout (binning)

same moment we confirm that dropout noise with any continuous distribution not only results in finite MI, but also provides an elegant way to estimate it. In particular, Gaussian dropout is known to benefit generalization even more than binary dropout (Srivastava et al., 2014). We empirically analyse the quality of the MI estimation in the setup with Gaussian dropout in a range of experiments on benchmark neural networks and datasets.

# 2 MUTUAL INFORMATION ESTIMATION FOR NEURAL NETWORKS

In this paper we use the following notations:  $H(A)$  denotes the Shannon entropy of a discrete random variable  $A$  whose distribution is denoted  $p_A$ ;  $h(B)$  is the differential entropy of a continuous random variable  $B$  whose distribution is described by the probability density function  $p_B$ ;  $I(A; B)$  is the MI between random variables  $A$  and  $B$ ;  $X \in \mathcal{X}$  and  $Y \in \mathcal{Y}$  are the random variables describing inputs to a neural network and corresponding targets;  $f(X)$  is the result of the forward pass of the input through the network to the hidden layer of interest;  $Z$  is an  $n$ -dimensional random variable describing the hidden representations that are being analyzed. Note, that while  $f(\cdot)$  describes a deterministic function implemented by a neural network,  $Z$  may contain additional stochasticity, e.g., via additive or multiplicative noise.

There are several equally valid choices to theoretically describe the distributions of  $X$  and  $Z$ . The caveats of different approaches with respect to measuring MI were discussed widely in the literature (Saxe et al., 2019; Kolchinsky et al., 2019). In the following we shortly summarize the problems motivating the approach proposed in this paper. It should be noted that these problems do not appear for the MI measured between representations and targets, since they are not connected via a deterministic function. We therefore concentrate on  $I(X;Z)$ , where we assume that  $Z$  does not contain additional stochasticity, i.e., we describe a deterministic neural network with  $Z = f(X)$ .

The first option is to assume the input to be drawn from a discrete distribution. This view makes it easy to use a finite dataset  $S$  at hand to describe the distribution and is supported by the finiteness of the accuracy of the used computational resources (Lorenzen et al., 2021). More precisely, it describes the input distribution as a uniform on the training data and fixes the discretization corresponding to the computer precision (or other selected bin size for ease of experimental setup). In this case  $Z$  is discrete as well and the MI between  $X$  and  $Z$  is computed like  $I(X;Z) = H(Z) - H(Z|X) = H(Z) - H(f(X)|X) = H(Z) - 0 = H(Z)$ , where we assume that the network forward function  $f(\cdot)$  is deterministic. Thus, estimated MI between input and representation essentially corresponds to the entropy of the representation, which is equal to the entropy of the dataset  $\log |S|$  unless the forward pass maps some of the different data points from the dataset to the same value in the latent space.

The second option is to assume  $X$  to be drawn from a continuous distribution. This is more aligned to the common description of real world data where we assume that it is drawn from some data generating distribution  $\mathcal{D}$  over the whole space of  $\mathcal{X} \times \mathcal{Y}$ . In this case, if the network transformation  $f(\cdot)$  results in a discrete distribution of representations  $Z$  one can still use the decomposition  $I(X,Z) = H(Z) - H(Z|X)$  to estimate and describe the MI based on Shannon entropy. As shown

in Theorem 1 of Amjad and Geiger (2019) this is however not the case for neural networks with commonly used activation functions. If  $f(\cdot)$  is deterministic and  $Z$  is not purely discrete, the MI between  $X$  and  $Z$  is infinite. This happens because the joint distribution is not absolutely continuous with respect to the product of the marginals. Thus, the approach to estimate the MI of continuous input and hidden representation in practice is to modify the space of inputs and/or representations to be discrete, e.g., by binning. For example, binning the representation  $Z$  to  $\hat{Z}$  again yields  $I(X; \hat{Z}) = H(\hat{Z})$ , and the qualitative behavior of this entropy will be defined by properties of activation functions and selected bin size (Saxe et al., 2019).

From the discussion above it follows that estimating  $I(X;Z)$  in deterministic neural networks requires discretization and always reflects geometric properties of representation space or errors of the used estimators. As a solution to the discussed challenges, in their work Goldfeld et al. (2019) propose to introduce zero mean Gaussian noise added to the representations during training. This transforms a deterministic neural network into a stochastic one, but essentially does not change the training results and predictive abilities of the model. Such setup allows for applying estimation techniques for entropy values, based on Monte Carlo sampling, and then uniting them into the estimation of MI, with a known error bound on the estimation. The authors claim that in such setup MI estimation is tracking the clustering of the representations, thus also their geometric properties. It allows for the theoretical explanation and justification of the controversial information compression phase hypothesis (Shwartz-Ziv and Tishby, 2017): When the MI between input and representation is dropping, it means that noise-induced Gaussians centered in the representations are overlapping more. Then it is becoming harder to distinguish between same label inputs by their representations, which means lower MI with input. Nevertheless, measuring geometric properties of the representation space via MI, as the empirical results of Goldfeld et al. (2019) demonstrate, still vague about the most interesting aspect: Can information plane development be predictive about generalization abilities of the trained network?

Overall, the existing research indicates the need for a method to measure MI for networks with continuous input distributions, since the known approaches to MI estimation lead to an analysis of the development of neural network representations that is limited to geometric interpretation (Geiger, 2021). Based on the knowledge about benefits of the stochasticity added to the training of the neural networks, e.g., in the form of dropout, we suggest to concentrate on the setup where representations are not deterministic, which allows for theoretically justified values of MI. While additive noise is a valuable step towards measuring such MI, it requires noise introduced on all the layers of a network and samples from multiple copies of the network, moreover, it is centered around geometrical meaning of the information compression. In this work, we analyse the stochastic representation obtained via a dropout layer, and show that such noise has a great potential for revealing the underlying information flow of the neural network.

# 3 MUTUAL INFORMATION IN DROPOUT NETWORKS

As discussed in the previous section, the MI between inputs and hidden representations of deterministic networks is infinite, if we assume the input distribution to be described by some continuous probability density function. To overcome this problem, some form of stochasticity has to be introduced. While adding noise to activations (Goldfeld et al., 2019) indeed allows to compute MI, it is not a widely used type of the neural networks. Neural networks with dropout instead are one of the most popular classes of neural networks used in practice and are indeed stochastic in nature: Adding a dropout layer to a neural network corresponds to multiplying the hidden representation with some form of random noise. Formally, denoting the random noise by a random variable  $D$  of the same dimension as  $f(X)$ , the hidden representation is  $Z = f(X) \circ D$ , where  $\circ$  denotes elementwise multiplication. In the most basic form  $D$  follows a Bernoulli distribution (Srivastava et al., 2014). Currently, such binary dropout is widely used since it allows us to intuitively understand it as "turning off" a fraction of neurons during training. There is a variety of other dropout layers, including multiplicative Gaussian noise, fast dropout (Wang and Manning, 2013), or variational dropout (Kingma et al., 2015). Information dropout (Achille and Soatto, 2018) is a variant that allows us to approximate mutual information in a closed form—while the approximation itself is rather crude, it helps to train a network and is therefore used in the original work as a weighted reg

ularization term complementing the cross-entropy loss. In order to obtain such closed form, dropout noise is sampled from a log-normal distribution and a prior distribution on representations is chosen depending on the activation function (ReLU or Softplus). We provide details on the derivation in Appendix A.2.

Instead of approximating MI for regularization, our goal is to compute MI accurately using dropout as a source of stochasticity. For that, we first show a negative result: binary dropout leads to MI being infinite in theory. We then show that any continuous dropout prevents MI from becoming infinite and thus allows for its meaningful estimation.

# 3.1 BINARY DROPOUT

We start by analysing binary dropout which forces individual neurons to be "turned off" with some (discrete) probability. More formally, the output of each neuron is multiplied with an independent Bernoulli random variable that is equal to 1 with a predefined probability  $p$ . The following theorem shows that this kind of (combinatorial) stochasticity is insufficient to prevent MI from becoming infinite.

Theorem 3.1. In the setting of (Amjad and Geiger, 2019, Th. 1), let the output  $f(\cdot)$  of a hidden layer be parameterized as a deterministic neural network with  $N$  neurons, let  $B \in \{0,1\}^N$  be the set of independent Bernoulli random variables characterizing the dropout pattern, and let  $Z$  be the stochastic representation obtained by applying the dropout pattern to the neural network function  $f(\cdot)$ . Then it holds that  $I(X;Z) = \infty$ .

Proof. Using the chain rule of MI, we have

$$
\begin{array}{l} I (X; Z) = I (X; Z, B) - I (B; X | Z) \\ = I (X; Z | B) + I (B; X) - I (B; X | Z) \geq I (X; Z | B) - H (B) \\ \end{array}
$$

where the last line follows from dropping  $I(B;X)$  since they are independent and the fact that  $I(B;X|Z) \leq H(B)$ . Having  $B \in \{0,1\}^N$  as a discrete random variable, it immediately follows that  $H(B) \leq N \log 2$ . Now note that

$$
I (X; Z | B) = \sum_ {b \in \{0, 1 \} ^ {N}} \mathbb {P} (B = b) I (X; Z | B = b).
$$

Since the Bernoulli variables are independent, positive probability mass is assigned to  $B = (1, 1, \ldots, 1)$ , i.e., to the case where all neurons are active. Evidently, when  $b = (1, 1, \ldots, 1)$  it follows that  $Z = f(X)$ . Thus, with (Amjad and Geiger, 2019, Th. 1)

$$
I (X; Z | B) \geq \mathbb {P} (B = (1, 1, \dots , 1)) I (X; f (X)) = \infty
$$

and thus  $I(X;Z) = \infty$

![](images/bfeda02039c6278776c6678d6655eb1f2ca5b6b94764175d1e5f1c167cce675a.jpg)

While the Bernoulli distribution guarantees that  $B = (1, 1, \dots, 1)$  always has non-zero probability, other distributions over  $\{0, 1\}^N$  might not have this property. Theorem 3.1 can however be generalized to arbitrary distributions over  $\{0, 1\}^N$ :

Theorem 3.2. In the setting of (Amjad and Geiger, 2019, Th. 1), let the output  $f(\cdot)$  of a hidden layer be parameterized as a deterministic neural network with  $N$  neurons, let  $B \in \{0,1\}^N$  be the binary random vector characterizing the dropout pattern, and let  $Z$  be the stochastic representation obtained by applying the dropout pattern to the neural network defining  $f(\cdot)$ . Then, it either holds that  $I(X;Z) = \infty$  or that  $I(X;Z) = 0$  if the dropout patterns almost surely disrupt information flow through the network.

Proof. If the binary dropout is such that nonzero probability is assigned to the dropout mask  $B = (1, 1, \ldots, 1)$ , then the statement of the theorem follows as in the proof of the previous theorem.

Assume now that  $B$  is such that zero mass is assigned to  $B = (1, 1, \dots, 1)$ . To treat this case, we suppose that the distribution of  $X$  has a portion with a continuous probability density function on a compact set and that the neural network has activation functions that are either bi-Lipschitz or continuously differentiable with a strictly positive derivative (following the requirements of Amjad

and Geiger (2019, Th. 1)). Then, we obtain  $I(X; f(X)) = \infty$  from (Amjad and Geiger, 2019, Th. 1) for almost all parameterizations of the neural network. Under this setting,  $f(X) \circ b$  is again a neural network with activation functions that are either bi-Lipschitz or continuously differentiable with a strictly positive derivative. Assuming that  $b$  is such that the input of the network is not completely disconnected from the considered layer, for this pattern we have  $I(X; Z|B = b) = \infty$ . Otherwise, we obviously have  $I(X; Z|B = b) = 0$ . The statement of the theorem follows from taking the expectation over all patterns  $b$ .

Thus, binary dropout cannot be used to estimate MI reliably and is not helpful for the information plane analysis. The main obstacle is the finiteness of the combinatorial space induced by the binary noise on the space of representations.

# 3.2 DROPOUT WITH CONTINUOUS NOISE

As proposed by Srivastava et al. (2014), dropout can be implemented using continuous Gaussian noise with mean vector  $\mu = 1$ . Formally, the hidden representation  $Z$  is given by  $f(X) \circ D$ , where  $D \sim \mathcal{N}(\mathbf{1}, I\sigma^2)$  with  $\sigma^2$  being the selected variance of the noise. In contrast to binary noise sampled from a discrete distribution, multiplicative Gaussian noise turns the joint distribution of  $(Z, X)$  to be absolutely continuous with respect to the marginals of  $Z$  and  $X$  allowing for finite values of MI between the input  $X$  and the hidden representation. The following theorem states that the MI between input and hidden dropout representation indeed is finite.

Theorem 3.3. Let  $p_x$  be a probability density function on  $X$ ,  $f(\cdot)$  be parameterized by a deterministic neural network with Lipschitz activation functions, and let  $Z = f(X) \circ D$ , with  $D \sim \mathcal{N}(\mathbf{1}, I\sigma^2)$ . Then, under assumption of  $\|X\|_{\infty} < \infty$ , it holds that  $I(X,Z) < \infty$ .

Proof. We first restrict our attention to representations  $Z$  for which all dimensions are different from zero. This can be done w.l.o.g. as the following argument shows. Specifically, suppose that  $Z = (Z_{1},\ldots ,Z_{n})$  and that  $B = (B_{1},\dots,B_{n})$  with  $B_{i} = 0$  if  $Z_{i} = 0$  and  $B_{i} = 1$  otherwise. Clearly,  $B$  is a function of  $Z$ , hence  $I(X;Z) = I(X;Z,B) = I(B;X) + I(Z;X|B)$ . Since  $B$  is binary, we have that  $I(X;B)\leq H(B)\leq n\ln 2$ . Letting  $Z_{B} = (Z_{i}|i:B_{i} = 1)$  denote the sub-vector of non-zero elements of  $Z$ , we thus obtain

$$
I (X; Z) \leq n \ln 2 + \sum_ {b} \mathbb {P} (B = b) I \left(Z _ {b}; X\right). \tag {1}
$$

Therefore,  $I(X;Z)$  is finite iff  $I(Z_{b};X) = I(Z;X|B = b)$  is finite  $B$ -almost surely. We thus now fix an arbitrary  $B = b$  and continue the proof for  $Z = Z_{b}$ .

Since the distribution of  $Z$  is continuous, we consider the following decomposition of MI into differential entropies:  $I(X;Z) = h(Z) - h(Z|X)$ . The entropy of the representations  $h(Z)$  is upper-bounded by the entropy of a Gaussian with the same covariance matrix  $\Sigma$  as the distribution of  $Z = (Z_{1},\ldots ,Z_{n})$ , i.e., by  $\frac{n}{2}\ln (2\pi) + \frac{1}{2}\ln (\operatorname*{det}(\Sigma)) + \frac{1}{2} n$ , where  $n$  is the dimensionality of  $Z$ . From Hadamard's inequality and since  $\Sigma$  is positive semidefinite it follows that  $\operatorname*{det}(\Sigma)\leq \prod_{i = 1}^{n}\sigma_{ii}$ , where  $\sigma_{ii}$  are diagonal elements of the covariance matrix, i.e.,  $\sigma_{ii} = Var[Z_i]$ . Due to the independence of  $f(X)$  and  $D$  the variance of  $Z_{i}$  is given by

$$
V a r [ Z _ {i} ] = V a r [ f (X) _ {i} ] V a r [ D _ {i} ] + V a r [ f (X) _ {i} ] (\mathbb {E} [ D _ {i} ]) ^ {2} + V a r [ D _ {i} ] (\mathbb {E} [ f (X) _ {i} ]) ^ {2}.
$$

From the fact that the activation functions are Lipschitz, it follows that the composition  $f(\cdot)$  of activation functions and linear maps is also Lipschitz and has Lipschitz constant  $L$ . Thus,  $Var[f(X)_i] \leq 2LVar[X_i]$ . Moreover, since  $X_{i}$  is bounded and  $f(\cdot)$  is Lipschitz,  $E[f(X)_i]$  is bounded as well. Therefore,  $Var[Z_i]$  is bounded for all  $i$  and the determinant of the covarinace matrix is finite, leading to  $h(Z) < \infty$ .

It remains to show that the  $h(Z|X) > -\infty$ . Due to the independence of  $D_{i}$  and  $D_{j}$ , for all  $i,j = 1,\ldots n,i\neq j$ , the conditional entropy of  $Z$  factorises in the sum of conditional entropy of its components, i.e.,  $h(Z|X) = \sum_{i = 1}^{n}h(Z_{i}|X)$ . The conditional entropy  $h(Z_{i}|X)$  is defined as

$$
h (Z _ {i} | X) = - \int p _ {z _ {i}, x} \log (p _ {z _ {i} | x}) d z _ {i} d x = - \int p _ {x} (p _ {z _ {i} | x} \log (p _ {z _ {i} | x})) d z _ {i} d x = \mathbb {E} _ {x} [ h (p _ {z _ {i} | x}) ] . ^ {2}
$$

Since for every fixed  $X = x$ , each  $p_{z_i|x}$  is a Gaussian with mean  $f(x)_i$  and variance  $|f(x)_i| \cdot \sigma$ , i.e.,  $Z_i|X = x \sim \mathcal{N}(f(x)_i, (|f(x)_i| \cdot \sigma)^2)$ , where  $f(x)_i$  is the  $i$ -th component of  $f(x)$  that is being multiplied by the  $i$ -th component of  $D$ , the entropy is given by

$$
\begin{array}{l} h \left(Z _ {i} | X = x\right) = \mathbb {E} _ {x} \left[ \log \left(\left| f (x) _ {i} \right| \sigma \sqrt {2 \pi e}\right) \right] = \mathbb {E} _ {x} \left[ \log \left(\left| f (x) _ {i} \right|\right) \right] + \mathbb {E} _ {x} \left[ \log \left(\sigma \sqrt {2 \pi e}\right) \right] = \\ = \mathbb {E} _ {x} [ \log (| f (x) _ {i} |) ] + \log (\sigma \sqrt {2 \pi e}) \\ \end{array}
$$

Remember that all  $|f(x)_i|$  are strictly positive. By the law of the unconscious statistician we can exchange the integral with the following one  $\mathbb{E}_x[\log (|f(x)|_i)] = \int \log (\hat{x})p_{\hat{x}}d\hat{x}$ , where  $p_{\hat{x}}$  corresponds to the density function of  $|f(x)|_i$ . Under assumption that  $p_{\hat{x}} < K$ , where  $K$  is some number, we get  $\int \log (\hat{x})p_{\hat{x}}d\hat{x} < K\int \log (\hat{x})d\hat{x} = K\hat{x} (\log (\hat{x}) - 1)$  which is finite. The properties of  $p_{\hat{x}}$  are defined by the unknown  $p_x$  and unknown  $f(\cdot)$ . Seeing  $|f(x)_i|$  as a result of mapping an input vector  $x$  to a scalar, we can write down

$$
p _ {\hat {x}} = \int_ {\mathcal {X}} p _ {x} \delta (\hat {x} - | f (x) | _ {i}) d x,
$$

where  $\delta$  is a delta Dirac function. Essentially, if we assume that  $p_x$  is such that mathematical expectation is finite, then  $p_{\hat{x}}$  is also bounded.

Therefore,  $I(X;Z) < \infty$  for  $Z$  being representation under Gaussian dropout.

![](images/44aa902750145fb19ee8e25d057cb90247d8b3494929910abdf341a54b0cb1fe.jpg)

This proof can be straightforwardly extended to any continuous distribution—including the distributions used for information dropout (Achille and Soatto, 2018). As long as the noise distribution has finite entropy, the resulting MI is finite, since its first term is bounded by the Gaussian entropy and the second term modifies to

$$
\mathbb {E} _ {x} [ h (| f (x) | \circ D) ] = h (D) + \mathbb {E} _ {x} \left[ \log (| f (x) |) \right].
$$

Thus the same argument as in the proof for Gaussian dropout holds. Moreover, if Thm. 3.3 holds for a particular layer, then it follows from the data processing inequality that MI is finite for all subsequent layers.

Corollary 3.4. Let  $p_x$  be a probability density function on  $X$ ,  $f(\cdot)$  be parameterized by a deterministic neural network, and let  $Z = f(X) \circ D$ , with  $D \sim \mathcal{N}(\mathbf{1}, I\sigma^2)$ . Then, under assumption of  $\| X \|_{\infty} < \infty$ , it holds that  $I(X, Z_l) \leq \infty$ , where  $Z_l$  corresponds to the representations on the layers  $l$  that follow after dropout.

# 4 ESTIMATION OF MI UNDER CONTINUOUS DROPOUT

First, let's consider information dropout network. The computational benefit of information dropout is that the values necessary to produce an IP are contained readily in the training loss. That is, the variational approximation of  $I(X;Z)$  is the regularization term of the information dropout objective with additional summands for ReLU (see Appendix A.2). Since ReLU networks require a known weight on the delta Dirac in the prior distribution in order to compute the full  $I(X;Z)$ , we use Softplus activations with a

![](images/96a2048ee91ffe3419a4414982fa0e255bb7a80a8863419536861fa619598bc3.jpg)  
(a)  $\sigma = 0.1$ ,  $n = 2$

![](images/5c4793838f701a7e8de74b1e889236a1a0d6bd327ae28d71ffe9f2c53e83d4a5.jpg)  
Figure 2: Independent of the dimensionality, MC estimation of  $h(Z|X)$  stabilizes with the amount of samples.  
(b)  $\sigma = 0.1$ ,  $n = 50$

known log-normal prior.  $I(Y;Z)$  is approximated by the cross entropy loss  $l_{ce}$  subtracted from the approximation of  $H(Y)$ . For the classification problems  $H(Y)$  can be for example approximated by  $\log(C)$  where  $C$  is number of classes.

In the case of Gaussian dropout, to estimate  $I(X;Z)$  we approximate  $h(Z)$  and  $h(Z|X)$  separately.

For estimating  $h(Z)$  we employ a Monte Carlo (MC) estimate, similar to the one proposed by Goldfeld et al. (2019). That is, we approximate the distribution of  $Z$  as a Gaussian mixture, where

![](images/88535dfb8a30c7814e2f45c5bffeb83b0735385ab9d31bb8abc5ad6622943395.jpg)  
(a)  $\sigma = 0.1, n = 1$

![](images/fb91a1c0e15160901a401a1285150d1eff9c5406612ccec6039476df2f502dac.jpg)  
Figure 4: Comparison of various approaches to MI estimation for the toy example with multiplicative Gaussian noise. For low dimensional  $X$  and  $Z$  different bin sizes lead to different MI estimates of the binning estimator. For higher dimensions the binning based estimate is almost always maximal. Our estimation approaches the lower bound estimation proposed by McAllester and Stratos (2020) tightly from above.  
(b)  $\sigma = 0.1$ ,  $n = 50$

we draw samples  $f(x^{(j)}), j = 1, \ldots$  and place Gaussians with a diagonal covariance matrix with variances  $\sigma |f(x^{(j)})_i|, i = 1, \ldots, n$  on each sample of  $f(x^{(j)})$ . For a sanity check we also compute an upper bound of  $h(Z)$  given by the entropy of a Gaussian with the same covariance matrix as  $Z$ . Note, that the estimation of the upper bound needs a sufficiently large number of samples to guarantee that the estimate from the sample covariance is not singular.

For each fixed  $X = x$  the conditional distribution  $p_{Z|X=x}$  is a Gaussian distribution  $\mathcal{N}(f(x), I(\sigma | f(x)|)^2)$ . Moreover, when the input is fixed, the components of  $Z|X = x$  are independent, since components of the noise are independent. This allows to compute  $h(Z|X)$  as a sum of  $h(Z_i|X)$  where  $Z_i$  is the  $i$ -th component of the representation vector. The computation of  $h(Z_i|X)$  requires integration over the input space for computing the mathematical expectation  $\mathbb{E}_x[h(Z_i|X = x)]$ . This can be approximated via MC sampling. That is, we approximate  $h(Z_i|X)$  by  $1/N \sum_{k=1}^{N} h(Z_i|X = x_k)$  where  $h(Z_i|X = x_k) = \log(|f(x_k)|_i \sigma \sqrt{2\pi e})$ .<sup>3</sup>

We consider a simple toy problem for validating our approach to computing MI where input  $X$  is generated from an  $n$ -dimensional standard normal distribution, then modified with a function  $f(X) = 2X + 0.5$  and Gaussian dropout distributed according to  $\mathcal{N}(1,\sigma^2)$  is applied. We investigate the convergence of our estimator for  $h(Z|X)$  for increasing number of samples. For each input sample we generate 10 noise masks, thus getting 10 samples of  $Z$ . Results are shown in Fig. 2. As it can be seen the estimation stabilizes with larger amount of samples for different dimensionality of the data. We also compare the

![](images/4e0eef23ee91b6a8043cd6b255fc6ae80aa0c4c0d0cec3780e993bac7fc8d68b.jpg)  
(a)  $\sigma = 0.1, n = 1$  
Figure 3: Estimates of the entropy of the hidden representation  $Z$ . With growing dimensionality of  $X$  the upper bound of the entropy of a Gaussian becomes very loose, compared to the Gaussian mixture based MC estimation.

![](images/13c0733e48a291ac81c75a1f8b1564900f92ebe3c62521faac2dafd37b96aeed.jpg)  
(b)  $\sigma = 0.1$ ,  $n = 50$

estimate to the upper bound for  $h(Z)$  in Fig 3.

Moreover, we compare our estimator to the binning approach, the EDGE estimator (Noshad et al., 2019), and the lower bounds analyzed by McAllester and Stratos (2020). The results are shown in Fig. 4. With binning we underestimate MI when bin size is small and overestimate with large

bin size (Ross, 2014), which can be clearly seen in the plots where bins are organized both by size and by amount. Moreover, in with the high dimensional data, binning approach hits the maximal possible value of  $\log(|S|)$  ( $S$  being the dataset at hand) very fast, not being able to overestimate the MI value. According to McAllester and Stratos (2020) lower bound based MI estimators also need exponentially (in the true value of MI) many data points for correct value prediction, otherwise they will always heavily underestimate the value. Further computations with different noise level are demonstrated in Appendix A.3.

# 5 INFORMATION PLANE ANALYSIS OF DROPOUT NETWORKS

We use the estimators described in the previous section for an information plane analysis of dropout networks. In the a first set of experiments, we analysed information planes (IPs) for the training of Gaussian and information dropout networks.

We always consider only the representation corresponding to the first dropout layer. For estimating  $I(Y;Z)$  we employ the EDGE estimator (Noshad et al., 2019). The analysis on the MNIST dataset was performed for a LeNet network (LeCun et al., 1998) that achieves  $99\%$  accuracy and a simple fully connected (FC) network with three hidden layers  $(28 \times 28 - 512 - 128 - 32 - 10)$  and softmax activation functions achieving  $97\%$  accuracy. We analyze both information dropout and Gaussian dropout in the LeNet network and only Gaussian dropout in

the FC network. We compare IPs based on estimators using the binning approach to IPs based on our approach in Fig. 1 and Fig. 5. We also build the IPs for a ResNet18 trained on CIFAR10 where we added an additional bottleneck layer with 128 neurons and Gaussian dropout before the output layer, and which achieves an accuracy of  $94\%$ . The resulting IPs for our MC based estimator and binning are shown in Fig. 6.

Interestingly, for all networks and datasets we observe significant compression, which however is not shown when MI is measured based on binning (note, that we got the same finding also for different bin sizes we tried). This indicates that the MI compression measured in dropout networks is different from purely geometrical compression, which would make it fundamentally different to all the discrete MI estimators and the additive noise approach (Basirat et al., 2021).

In a second set of experiments, we further analyze IPs in information dropout networks. For this we trained

![](images/8bdc73d520a6c7ba6252e55e50deaa3dac75f48e34ee516d9ab1e3e00b0c2719.jpg)  
(a) Our estimator

![](images/536a1f475bbdd6c6e8105811616ef2334b08fb61a6ae85c2c9d4618912d22e92.jpg)  
Figure 5: IPs for a FC network with Gaussian dropout trained on MNIST. Compared to the binning estimation of MI our approach reveals compression.  
(b) Binning estimator

![](images/dab70d803ca03b45e6c17a908f0e95a1bdb193a73186b6839614ef476e95303c.jpg)  
(a) Our estimator  
Figure 6: IPs for a ResNet18 network with Gaussian dropout trained on CIFAR10. Comparing to the binning estimation of MI our approach clearly shows compression.

![](images/dd153962b9fa7a8c3a5e7fc4001891892d97f6644b3fbabf5f5c297a830d35e3.jpg)  
(b) Binning estimator

CNN) on CIFAR10 where we used the official code provided. NN was trained for 200 epochs with SGD with momentum

and, different from the original setup, only one dropout layer after the third convolutional layer. The batch size was set to 100, the learning rate was initially set to 0.05 and was reduced by multiplying it with 0.1 after the 40, 80, and 120 epoch. The network was trained with different values of the regularization weight  $\beta$  and different amount of filters in the convolutional layers. Also different from the original setup, we allowed noise variance to grow up to 0.95, i.e., that the information flow is nearly cut, in order to see the effect of the limited information between representation and input more pronounced.

Results are shown in Fig. 7. With less regularization of information, promoted by a smaller value of  $\beta$  weighting the regularization (i.e., the KL-divergence) term the level of achieved MI with the label is higher and the resulting error is  $5\%$  lower for the test set and  $10\%$  for the train set. Thus, the larger the value of  $\beta$  the smaller is the amount of information between input and representation throughout the training, which leads to the higher error (both on training and test set). We can see larger information compression with smaller  $\beta$  and almost no compression with the larger  $\beta$ . We conjecture that information can only be compressed if enough of it is allowed to flow through the network. We repeated experiments with the same convolutional network architecture where we reduced the number of filters in the hidden layers. Fig. 7 (a) and (b) show IPs for the original fullCNN and (c), (d) show the IPs for the network that

![](images/ac8f733b82ae6d25766c2f2f9e5be4121e38425a156f67981864fa539725fd44.jpg)  
(a) fullCNN with  $\beta = 3$

![](images/d418e71f9dc48fa5995ce29e66fe09c5612659e763bef58d274ca76fb210cd56.jpg)  
(b) fullCNN with  $\beta = 20$

![](images/625ed3f68fe8cb935dacfec3aabd2d1e35c5d30a5e1868358dbbe436ed925187.jpg)  
(c) 0.25fullCNN with  $\beta = 3$

![](images/aa41c09889d145bf29e0ea7fa2ad1648386b3ee8d60671886ee6dc547c6cf53a.jpg)  
Figure 7: IP demonstrate more (a), (c) and less (b), (d) compression of MI between input and representation.  
(d) 0.25fullCNN with  $\beta = 20$

contains  $25\%$  of the filters on each of the layers. This indicates that the ability to compress MI between input and representation can be connected with the overall capacity of a neural network.

Additional plots for the performed experiments can be found in Appendix A.4.

# 6 CONCLUSION

In this paper we analyze the dropout neural networks from the information-theoretic perspective. A theoretically sound estimation of the MI between input and hidden representation of a neural network requires either discretization or the introduction of stochasticity into the network. While discretization has the disadvantage that it limits the analysis to the geometric properties of  $Z$ , artificially introducing stochasticity limits insights due to altering the model. Dropout neural networks, however, present a class of neural networks that are inherently stochastic and heavily used in practice. In our theoretical analysis we thus investigate if different variants of dropout noise allow for finite values of MI if we assume a continuous input distribution. Our first result shows that binary dropout does not help to prevent the MI from getting infinite. We then prove, however, that Gaussian dropout—or more generally any form of dropout with noise from a continuous distribution with finite entropy—does. Based on this result we propose an MC based estimate of the MI in Gaussian dropout networks and perform an information plane analysis for different networks with Gaussian and information dropout (the latter offers a natural way to measure MI based on the MI based regularization term) on different datasets. This analysis demonstrates that the proposed estimates lead to substantially different observations than binning based approaches. Particularly, we observe compression for all studied networks and data sets. We hypothesize that in dropout networks compression of MI is not alone driven by geometrical compression. A more detailed information bottleneck analysis of dropout networks is left for the future work.

# REFERENCES

Alessandro Achille and Stefano Soatto. Information dropout: Learning optimal representations through noisy computation. IEEE transactions on pattern analysis and machine intelligence, 40 (12):2897-2905, 2018. 3, 6, 8, 11  
Rana Ali Amjad and Bernhard C Geiger. Learning representations for neural network-based classification using the information bottleneck principle. IEEE transactions on pattern analysis and machine intelligence, 42(9):2225-2239, 2019. 1, 3, 4, 5  
Mina Basirat, Bernhard C Geiger, and Peter M Roth. A geometric perspective on information plane analysis. Entropy, 23(6):711, 2021. 1, 8  
Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jorn-Henrik Jacobsen. Invertible residual networks. In International Conference on Machine Learning, pages 573-582. PMLR, 2019. 11  
Bernhard Claus Geiger. On information plane analyses of neural network classifiers - a review, 2021. accepted for publication in IEEE Trans. Neural Netw. Learn. Syst.; preprint available: arXiv:2003.09671 [cs.LG].1,3  
Ziv Goldfeld, Ewout van den Berg, Kristjan H Greenewald, Igor Melnyk, Nam Nguyen, Brian Kingsbury, and Yury Polyanskiy. Estimating information flow in deep neural networks. In ICML, 2019. 1, 3, 6, 7  
Durk P Kingma, Tim Salimans, and Max Welling. Variational dropout and the local reparameterization trick. Advances in neural information processing systems, 28, 2015. 3  
Artemy Kolchinsky, Brendan D Tracey, and Steven Van Kuyk. Caveats for information bottleneck in deterministic scenarios. ICLR, 2019. 2, 11  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998. 8  
Stephan Sloth Lorenzen, Christian Igel, and Mads Nielsen. Information bottleneck: Exact analysis of (quantized) neural networks. arXiv preprint arXiv:2106.12912, 2021. 2  
David McAllester and Karl Stratos. Formal limitations on the measurement of mutual information. In International Conference on Artificial Intelligence and Statistics, pages 875-884. PMLR, 2020. 7, 8  
Morteza Noshad, Yu Zeng, and Alfred O Hero. Scalable mutual information estimation using dependence graphs. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2962-2966. IEEE, 2019. 7, 8  
Brian C Ross. Mutual information between discrete and continuous data sets. PloS one, 9(2): e87357, 2014. 1, 8  
Andrew M Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan D Tracey, and David D Cox. On the information bottleneck theory of deep learning. Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124020, 2019. 2, 3  
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017. 1, 3  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929-1958, 2014. 2, 3, 5, 14  
Sida Wang and Christopher Manning. Fast dropout training. In international conference on machine learning, pages 118-126. PMLR, 2013. 3
