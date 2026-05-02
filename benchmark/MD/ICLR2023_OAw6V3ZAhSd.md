# HYPERDEEPONET: LEARNING OPERATOR WITH COMPLEX TARGET FUNCTION SPACE USING THE LIMITED RESOURCES VIA HYPERNETWORK

Anonymous authors

Paper under double-blind review

# ABSTRACT

Fast and accurate predictions for complex physical dynamics are a big challenge across various applications. Real-time prediction on resource-constrained hardware is even more crucial in the real-world problems. The deep operator network (DeepONet) has recently been proposed as a framework for learning nonlinear mappings between function spaces. However, the DeepONet requires many parameters and has a high computational cost when learning operators, particularly those with complex (discontinuous or non-smooth) target functions. In this study, we propose HyperDeepONet, which uses the expressive power of the hypernetwork to enable learning of a complex operator with smaller set of parameters. The DeepONet and its variant models can be thought of as a method of injecting the input function information into the target function. From this perspective, these models can be viewed as a special case of HyperDeepONet. We analyze the complexity of DeepONet and conclude that HyperDeepONet needs relatively lower complexity to obtain the desired accuracy for operator learning. HyperDeepONet was successfully applied to various operator learning problems using low computational resources compared to other benchmarks.

# 1 INTRODUCTION

Learning operators that map between infinite-dimensional function spaces, called operator learning, is a challenging problem and has been used in many applications, such as climate prediction (Kurth et al., 2022) and fluid dynamics (Guo et al., 2016). The computational efficiency to learn the mapping is still an important factor in real-world problems. The target function of the operator can be discontinuous or sharp for complicated dynamic systems. In this case, balancing model complexity and cost for computational time is a core problem for the real-time prediction on resource-constrained hardware (Choudhary et al., 2020; Murshed et al., 2021).

Many machine learning methods and deep learning-based architectures have been successfully developed to learn a nonlinear mapping from an infinite-dimensional Banach space to another. They focus on learning the solution operator of some PDEs, e.g., the initial or boundary condition of PDE to the corresponding solution. Li et al. (2020c) proposed an iterative neural operator scheme to learn the operator mapping with the graph neural network (Li et al., 2020b) and with the Fourier transform (Li et al., 2020a).

At the same time, Lu et al. (2019; 2021) proposed a deep operator network (DeepONet) architecture based on the universal operator approximation theorem of Chen & Chen (1995). The DeepONet consists of two networks: branch net taking sensor values of the input function, and trunk net taking a querying location of the output function domain. Each output of two networks is regarded as the  $p$ -coefficients and  $p$ -basis of the target function. The two  $p$ -outputs are combined as a linear combination (inner-product) to approximate the underlying operator. The DeepONet has been greatly applied to various problems, such as bubble growth dynamics (Lin et al., 2021), hypersonic flows (Mao et al., 2021), and fluid flow (Cai et al., 2021) with analysis of error estimates (Lanthaler et al., 2022).

While variant models of DeepONet are developed to improve the vanilla DeepONet, e.g. Physics-Informed DeepONet (Wang et al., 2021) and Variable-input DeepONet (Prasthofer et al., 2022), it

still has difficulty approximating the operator for complex target function with the limited computational resources. Lanthaler et al. (2022) and Kovachki et al. (2021b) pointed out the limitation of the linear approximation using the DeepONet. Some operators are known to have a slow spectral decay rate of the Kolmogorov  $n$ -width, which defines the error of the best possible linear approximation using  $n$ -dimensional space. This means that a very large  $n$  is needed to learn complex operators accurately, which implies that the DeepONet requires a large number of basis  $p$  and a large number of network parameters for them.

Hadam (2022) investigated the behavior of DeepONet, which makes it challenging to produce the sharp features in the target function when the number of basis  $p$  is small. They proposed a Shift-DeepONet by adding two neural networks to shift and scale the input function. Venturi & Casey (2022) also analyzed the limitation of DeepONet via singular value decomposition (SVD) and proposed a flexible DeepONet (flexDeepONet) adding a pre-net and one additional output in the branch net. Recently, to overcome the limitation of the linear approximation, Seidman et al. (2022) proposed a nonlinear manifold decoder (NOMAD) framework by using a neural network that takes the output of the branch net as input with the querying location. The number of basis cannot directly correspond to the total number of parameters in the model, even though these methods reduce the number of basis functions. They still need many parameters for the trunk net to learn the complex operators, especially with complex (discontinuous or non-smooth) target functions.

In this study, we propose a new architecture, HyperDeepONet, to learn operators with a complex target function space using limited resources. The HyperDeepONet uses a hypernetwork, proposed by Ha et al. (2016), which produces parameters for the target network that is utilized for actual tasks. Wang et al. (2022) pointed out that the final inner product in DeepONet may be inefficient if the information of the input function fails to propagate through a branch net. The hypernetwork in HyperDeepONet facilitates transmission the information of the input function to all network parameters of the target network. Furthermore, the expressivity of the hypernetwork reduces the neural network complexity by sharing the parameters of the hypernetwork (Galanti & Wolf, 2020). Our main contributions are as follows.

- We propose a novel HyperDeepONet using hypernetwork to overcome the limitations of DeepONet and learn the operators with a complex target function space. The DeepONet and its variant models are analyzed with a focus on expressing the target function as a neural network (Figure 2). These models can be treated as simplified versions of our general HyperDeepONet model (Figure 3).  
- We analyze the complexity of DeepONet (Theorem 2) and show that the complexity of the HyperDeepONet is smaller than that of the DeepONet. We have identified that the DeepONet should employ a large number of basis to obtain the desired accuracy, so it requires numerous parameters. We present a lower bound of the number of parameters in target network for variants of DeepONet, including nonlinear reconstructors.  
- The experiments show that the HyperDeepONet facilitates learning a operator with small number of parameters in target network when the target function space is complex with the discontinuity and sharpness, which the DeepONet and its variants suffer from. The HyperDeepONet learns the operator more accurately even when the total number of parameters in the overall model is same.

# 2 RELATED WORK

Many machine learning methods and deep learning-based architectures have been successfully developed to solve PDEs with several advantages. One research direction is to use the neural network directly to represent the solution of PDE (E & Yu, 2018; Sirignano & Spiliopoulos, 2018). The physics informed neural network (PINN), introduced by Raissi et al. (2019), minimized the residual of PDEs by using automatic differentiation instead of numerical approximations.

There is another approach to solving PDEs, called operator learning. Operator learning aims to learn nonlinear mapping from an infinite-dimensional Banach space to another. Many studies utilize the convolutional neural network to parameterize the solution operator of PDEs in various applications (Guo et al., 2016; Bhatnagar et al., 2019; Khoo et al., 2021; Zhu et al., 2019; Hwang et al., 2021). The neural operator (Kovachki et al., 2021b) is proposed to approximate the nonlinear operator inspired

by Green's function. Li et al. (2020a) extend the neural operator structure to the Fourier Neural Operator (FNO) to approximate the integral operator effectively using the fast Fourier transform (FFT). Kovachki et al. (2021a) proved the universality of FNO and identified the size of the network.

The DeepONet (Lu et al., 2019; 2021) has also been proposed as another framework for operator learning. Lanthaler et al. (2022) provided the universal approximation property of DeepONet. Wang et al. (2021) proposed physics-informed DeepONet by adding a residual of PDE as a loss function, and De Ryck & Mishra (2022) demonstrated the generic bounds on the approximation error for it. Prasthofer et al. (2022) considered the case where the discretization grid of the input function in DeepONet changes by employing the coordinate encoder. Lu et al. (2022) compared the FNO with DeepONet in different benchmarks to demonstrate the relative performance. FNO can only infer the output function of an operator in the same grid as the input function as it needs to discretize the output function to use FFT, whereas the DeepONet can predict at any location.

Ha et al. (2016) first proposed hypernetwork, a network that creates a weight of the primary network. Because the hypernetwork can achieve weight sharing and model compression, it requires a relatively small number of parameters even as the dataset grows. Galanti & Wolf (2020) proved that a hypernetwork provides higher expressivity with low complexity target networks. Sitzmann et al. (2020) and Klocek et al. (2019) employed this approach to restore images with insufficient pixel observations or resolutions. de Avila Belbute-Peres et al. (2021) investigated the relationship between the coefficients of PDEs and the corresponding solution. They combined the hypernetwork with the PINN's residual loss. For time-dependent PDE, Pan et al. (2022) designated the time  $t$  as the input of the hypernetwork so that the target network indicates the solution at  $t$ . Because the output dimension of the hypernetwork can be large, Von Oswald et al. (2019) devised a chunk embedding method that partitions the parameters of the target network.

# 3 OPERATOR LEARNING

# 3.1 PROBLEM SETTING

The goal of operator learning is to learn a mapping from infinite-dimensional function space to the others using a finite pair of functions. Let  $\mathcal{G}:\mathcal{U}\to S$  be a non-linear operator, where  $\mathcal{U}$  and  $S$  are compact subsets of infinite-dimensional function spaces  $\mathcal{U}\subset C(\mathcal{X};\mathbb{R}^{d_u})$  and  $S\subset C(\mathcal{V};\mathbb{R}^{d_s})$  with compact domains  $\mathcal{X}\subset \mathbb{R}^{d_x}$  and  $\mathcal{Y}\subset \mathbb{R}^{d_y}$ . For simplicity, we focus on the case  $d_{u} = d_{s} = 1$ , and all the results could be extended to a more general case for arbitrary  $d_{u}$  and  $d_{s}$ . Suppose we have observations  $\{u_i,\mathcal{G}(u_i)\}_{i = 1}^N$  where  $u_{i}\in \mathcal{U}$  and  $\mathcal{G}(u_i)\in S$ . We aim to find an approximation  $\mathcal{G}_{\theta}:\mathcal{U}\rightarrow S$  with parameter  $\theta$  using the  $N$  observations so that  $\mathcal{G}_{\theta}\approx \mathcal{G}$ . As explained in Lanthaler et al. (2022), the approximator  $\mathcal{G}_{\theta}$  can be decomposed into the three components (Figure 1) as

$$
\mathcal {G} _ {\theta} := \mathcal {R} \circ \mathcal {A} \circ \mathcal {E}. \tag {1}
$$

First, the encoder  $\mathcal{E}$  takes an input function  $u$  from  $\mathcal{U}$  to generate the finite-dimensional encoded data in  $\mathbb{R}^m$ . Then, the approximator  $\mathcal{A}$  is an operator approximator from the encoded data in finite dimension space  $\mathbb{R}^m$  to the other finite dimensional space  $\mathbb{R}^p$ . Finally, the reconstructor  $\mathcal{R}$  reconstructs the output function  $s(y) = \mathcal{G}(u)(y)$  using the approximated data in  $\mathbb{R}^p$ .

![](images/79b4fbb6b58dd65d313ff67e20ce004c5e47866b8148cbafc9df9a777ce5e70e.jpg)  
Figure 1: Diagram for the three components for operator learning.

# 3.2 DEEPONET AND ITS LIMITATION

DeepONet can be analyzed using the above three de

compositions. Given a set of sensor points  $x_{j} \in \mathcal{X}$ , they use an encoder as the pointwise projection  $\mathcal{E}(u) = (u(x_1), u(x_2), \dots, u(x_m))$  of the continuous function, the so-called sensor values. For the approximator  $\mathcal{A}: \mathbb{R}^m \to \mathbb{R}^p$ , they use the fully connected neural network. They referred to the composition of these two maps as branch net

$$
\beta : \mathcal {U} \rightarrow \mathbb {R} ^ {p}, \beta (u) := \mathcal {A} \circ \mathcal {E} (u) \tag {2}
$$

![](images/e6f763d8f1dd682929f700979439d7dd8f2ad52bea458f556fc1757ff972a719.jpg)  
Target network

![](images/7893469cf4b32d259172d5969920dad2da83a530383ba6f70513cf6e72db7015.jpg)  
(a) Perspective of target network parametrization  
Target network  
(c) Shift-DeepONet

![](images/d818240b9c87e80028878f96d2c681233f3d383a9cb838861704abc5448a1433.jpg)  
Figure 2: (a) The perspective target network parametrization for operator learning. (b-e)DeepONet and its variant models for operator learning.  
Target network  
(d) FlexDeepONet

![](images/787da14881e928525f9d782af579cece48bcb83aebbb638636641295039f16b6.jpg)  
Target network

![](images/1178026cfc318f530e70caf4cefba5e3038caff919a2c4e5fd6b1ca0740ec4e1.jpg)  
(b) DeepONet  
Target network  
(e) NOMAD

for any  $u \in \mathcal{U}$ . The role of branch net can be interpreted as learning the coefficient of the target function  $\mathcal{G}(u)(y)$ . They use one additional neural network, called trunk net  $\tau$  as shown below.

$$
\tau : \mathcal {Y} \rightarrow \mathbb {R} ^ {p + 1}, \tau (y) := \left\{\tau_ {k} (y) \right\} _ {k = 0} ^ {p} \tag {3}
$$

for any  $y \in \mathcal{V}$ . The role of trunk net can be interpreted as learning an affine space  $V$  that can efficiently approximate output function space  $C(\mathcal{V};\mathbb{R}^{d_s})$ . The functions  $\tau_1(y), \dots, \tau_p(y)$  become the  $p$ -basis of vector space associated with  $V$  and  $\tau_0(y)$  becomes a point of  $V$ . By using the trunk net  $\tau$ , the  $\tau$ -induced reconstructor  $\mathcal{R}$  is defined as

$$
\mathcal {R} _ {\tau}: \mathbb {R} ^ {p} \rightarrow C (\mathcal {Y}; \mathbb {R} ^ {d _ {s}}), \mathcal {R} _ {\tau} (\beta) (y) := \tau_ {0} (y) + \sum_ {k = 1} ^ {p} \beta_ {k} \tau_ {k} (y) \tag {4}
$$

where  $\beta = (\beta_{1},\beta_{2},\dots,\beta_{p})\in \mathbb{R}^{p}$ . In DeepONet,  $\tau_0(y)$  is restricted to be a constant  $\tau_0\in \mathbb{R}$  that is contained in a reconstructor  $\mathcal{R}$ . The architecture of DeepONet is described in Figure 2 (b).

Here, the  $\tau$ -induced reconstructor  $\mathcal{R}_{\tau}$  is the linear approximation of the output function space. Because the linear approximation  $\mathcal{R}$  cannot consider the elements in its orthogonal complement, a priori limitation on the best error of DeepONet is explained in Lanthaler et al. (2022) as

$$
\left(\int_ {\mathcal {U}} \int_ {\mathcal {Y}} | \mathcal {G} (u) (y) - \mathcal {R} _ {\tau} \circ \mathcal {A} \circ \mathcal {E} (u) (y) | ^ {2} d y d \mu (u)\right) ^ {\frac {1}{2}} \geq \sqrt {\sum_ {k > p} \lambda_ {k}}, \tag {5}
$$

where  $\lambda_1 \geq \lambda_2 \geq \ldots$  are the eigenvalues of the covariance operator  $\Gamma_{\mathcal{G}_{\# \mu}}$  of the push-forward measure  $\mathcal{G}_{\# \mu}$ . Several studies point out that the slow decay rate of the lower bound leads to inaccurate approximation operator learning using DeepONet (Kovachki et al., 2021b; Hadorn, 2022; Lanthaler et al., 2022). For example, the solution operator of the advection PDEs (Seidman et al., 2022; Venturi & Casey, 2022) and of the Burgers' equation (Hadorn, 2022) are difficult to approximate when we are using the DeepONet with the small number of basis  $p$ .

# 3.3 VARIANT MODELS OF DEEPONET

Several variants of DeepONet have been developed to overcome its limitation. All these models can be viewed from the perspective of parametrizing the target function as a neural network. When we think of the target network that receives  $y$  as an input and generates an output  $\mathcal{G}_{\theta}(u)(y)$ , the DeepONet and its variant model can be distinguished by how information from the input function  $u$  is injected into this target network  $\mathcal{G}_{\theta}(u)$ , as described in Figure 2 (a). From this perspective, the trunk net in the DeepONet can be considered the target network except for the final output as shown in Figure 2 (b). The output of the branch net gives the weight between the last hidden layer and the final output.

Hadam (2022) proposed Shift-DeepONet. The main idea is that a scale net and a shift net are used to shift and scale the input query position  $y$ . Therefore, it can be considered that the information of input function  $u$  generates the weights and bias between the input layer and the first hidden layer, as explained in Figure 2 (c).

Venturi & Casey (2022) proposed Flex-DeepONet, explained in Figure 2 (d). They used the additional network, pre-net, to give the bias between the input layer and the first hidden layer. Additionally, the output of the branch net also admits the additional output  $\tau_0$  to provide more information on input function  $u$  at the last inner product layer.

NOMAD is recently developed by Seidman et al. (2022) to overcome the limitation of DeepONet which learns a linear output manifold using the inner product between branch net and trunk net. They devise a non-linear output manifold using a neural network that takes the output of branch net  $\{\beta_i\}_{i=1}^p$  and the query location  $y$ . As explained in Figure 2 (e), the target network receives information about function  $u$  as an additional input, similar to other conventional neural embedding methods (Park et al., 2019; Chen & Zhang, 2019; Mescheder et al., 2019).

All of these methods provide information on the input function  $u$  to only a part of the target network. It makes sense to use a hypernetwork to share sensor information with all target network parameters. We propose a general model HyperDeepONet (Figure 3), which contains the vanilla DeepONet, Flex-DeepONet, and shift-DeepONet, as a special case of the HyperDeepONet, as explained in the following section.

# 4 PROPOSED MODEL : HYPERDEEPONET

# 4.1 ARCHITECTURE OF HYPERDEEPONET

The HyperDeepONet structure is described in Figure 3. The encoder  $\mathcal{E}$  and the approximator  $\mathcal{A}$  are used, similar to the vanilla DeepONet. The proposed structure replaces the branch net with the hypernetwork, and allows the output of the hypernetwork to generate not only the coefficient of basis but also all other parameters of the target network. More precisely, we define the hypernetwork  $h$  as

$$
h _ {\theta}: \mathcal {U} \rightarrow \mathbb {R} ^ {p}, h _ {\theta} (u) := \mathcal {A} \circ \mathcal {E} (u) \tag {6}
$$

for any  $u\in \mathcal{U}$  . Then,  $h(u) = \Theta \in \mathbb{R}^p$  is a network parameter of the target network, which is used in reconstructor for the HyperDeepONet. We define the reconstructor  $\mathcal{R}$  as

$$
\mathcal {R}: \mathbb {R} ^ {p} \rightarrow C (\mathcal {Y}; \mathbb {R} ^ {d _ {s}}), \mathcal {R} (\Theta) (y) := \mathrm {N N} (y; \Theta) \tag {7}
$$

where  $\Theta = [W,b]\in \mathbb{R}^p$ , and NN denotes the target network which is a fully connected neural network.

![](images/92732adb0fea70cc23795fa7d903d1d8b11ae8256a203b9d127f1e729232fdc4.jpg)  
Figure 3: The proposed HyperDeepONet

Here, we can think that the hypernetwork determines  $p$  weights  $\{\alpha_i\}_{i=1}^p$  between the final hidden layer and the output layer instead of DeepONet's last inner-product calculation. It implies that the structure of HyperDeepONet contains the entire structure of DeepONet. As shown in Figure 2 (c) and (d), Shift-DeepONet and Flex-DeepONet are also can be viewed as special cases of the

HyperDeepONet, where the output of the hypernetwork determines the weights or biases of some layers of the target network. The outputs of the hypernetwork determine the biases for the first hidden layer in the target network for NOMAD in Figure 2 (e).

# 4.2 COMPARISON ON COMPLEXITY OF DEEPONET AND HYPERDEEPONET

In this section, we would like to clarify the complexity of the DeepONet required for the approximation  $\mathcal{A}$  and reconstruction  $\mathcal{R}$  based on the theory in Galanti & Wolf (2020). Furthermore, using the results on the upper bound for the complexity of hypernetwork Galanti & Wolf (2020), we will show that the HyperDeepONet entails relatively lower complexity than the DeepONet.

# 4.2.1 NOTATIONS AND DEFINITIONS

Suppose that the sensor values  $\mathcal{E}(u) = (u(x_1), u(x_2), \dots, u(x_m))$  are given. For simplicity, we consider the case where the domain  $\mathcal{V}$  of the target function and space of the  $m$  sensor values are  $[-1, 1]^{d_y}$  and  $[-1, 1]^m$ , respectively. For the composition  $\mathcal{R} \circ \mathcal{A}: \mathbb{R}^m \to C(\mathcal{V}; \mathbb{R})$ , we are interested in approximating the mapping  $\mathcal{O}: \mathbb{R}^{m + d_y} \to \mathbb{R}$ , which is defined as follows:

$$
\mathcal {O} (\mathcal {E} (u), y) := (\mathcal {R} \circ \mathcal {A} (\mathcal {E} (u))) (y), \quad \text {f o r} y \in \mathbb {R} ^ {d _ {y}}, \mathcal {E} (u) \in \mathbb {R} ^ {m}.
$$

The supremum norm  $\| h \|_{\infty}$  refers to the value of  $\max_{y \in \mathcal{Y}} \| h(y) \|$ . Now, we introduce the Sobolev space  $\mathcal{W}_{r,n} \subset C^{r}([-1,1]^{n}; \mathbb{R})$ , which is defined as follows. For  $r, n \in \mathbb{N}$ ,

$$
\mathcal {W} _ {r, n} := \left\{h: [ - 1, 1 ] ^ {n} \to \mathbb {R} \quad \left| \| h \| _ {r} ^ {s} := \| h \| _ {\infty} + \sum_ {1 \leq | \mathbf {k} | \leq r} \| D ^ {\mathbf {k}} h \| _ {\infty} \leq 1 \right. \right\},
$$

where  $D^{\mathbf{k}}h$  denotes the partial derivative of  $h$  with respect to multi-index  $\mathbf{k} \in \{\mathbb{N} \cup \{0\}\}^{d_y}$ . We assume that the mapping  $\mathcal{O}$  lies in the Sobolev space  $\mathcal{W}_{r,m + d_y}$ .

For the nonlinear activation  $\sigma$ , the following class of neural network  $\mathcal{F}$  represents the fully-connected neural network with depth  $k$  and corresponding width  $(h_1, h_2, \dots, h_{k+1})$ , where  $W^i \in \mathbb{R}^{h_i} \times \mathbb{R}^{h_{i+1}}$  and  $b_i \in \mathbb{R}^{h_{i+1}}$  denote the weights and bias of the  $i$ -th layer respectively.

$$
\mathcal {F} := \left\{f: [ - 1, 1 ] ^ {n} \to \mathbb {R} | f (y; [ \mathbf {W}, \mathbf {b} ]) = W ^ {k} \cdot \sigma (W ^ {k - 1} \cdot \cdot \cdot \sigma (W ^ {1} \cdot y + b ^ {1}) + b ^ {k - 1}) + b ^ {k} \right\}
$$

There are activation functions that facilitate a close approximation for the Sobolev space, with small complexity. We will refer to those activation functions as universal activation functions. The formal definition can be found below, where the distance between the class of neural network  $\mathcal{F}$  and the Sobolev space  $\mathcal{W}_{r,n}$  is defined by  $d(\mathcal{F};\mathcal{W}_{r,n})\coloneqq \sup_{f\in \mathcal{F}}\inf_{\psi \in \mathcal{W}_{r,n}}\| f - \psi \|_{\infty}$ .

Definition 1. (Galanti & Wolf, 2020) (Universal activation). The activation function  $\sigma$  is called universal activation if there exists a class of neural network  $\mathcal{F}$  with activation function  $\sigma$  such that the number of parameters of  $\mathcal{F}$  is  $O(\epsilon^{-n / r})$  with  $d(\mathcal{F};\mathcal{W}_{r,n})\leq \epsilon$  for all  $r,n\in \mathbb{N}$

Most well-known activation functions are universal activations which are infinitely differentiable and non-polynomial in any interval (Mhaskar, 1996). Furthermore, Hanin & Sellke (2017) state that the ReLU activation is also universal activation.

The result of providing a lower bound on the number of parameters also exists. We first introduce an assumption before introducing the theorem. For  $r = 0$ , Galanti & Wolf (2020) remark that the assumption is valid for 2-layered neural networks with respect to the  $L^2$  norm when an activation function  $\sigma$  is either hyperbolic tangent or sigmoid function.

Assumption 1. Suppose that  $\mathcal{F}$  and  $\mathcal{W}_{r,n}$  represent the class of neural network and the target function space to approximate, respectively. Let  $\mathcal{F}'$  be a neural network class that represents a structure in which one neuron is added rather than  $\mathcal{F}$ . Then, the followings hold for all  $\psi \in \mathcal{W}_{r,n}$  not contained in  $\mathcal{F}$ .

$$
\inf  _ {f \in \mathcal {F}} \| f - \psi \| _ {\infty} > \inf  _ {f \in \mathcal {F} ^ {\prime}} \| f - \psi \| _ {\infty}.
$$

The following theorem holds under the aforementioned assumption, which states that the universal activation indeed provides a sharp bound on the number of parameters. Note that a real-valued function  $g \in L^{1}(\mathbb{R})$  is called bounded variation if its total variation  $\sup_{\phi \in C_c^1 (\mathbb{R}), \| \phi \|_\infty \leq 1} \int_{\mathbb{R}} g(x) \phi '(x) dx$  is finite.

Theorem 1. (Galanti & Wolf, 2020). Suppose that  $\mathcal{F}$  is a class of neural networks with a piecewise  $C^1 (\mathbb{R})$  activation function  $\sigma :\mathbb{R}\to \mathbb{R}$  of which derivative  $\sigma^\prime$  is bounded variation. If any nonconstant  $\psi \in \mathcal{W}_{r,n}$  does not belong to  $\mathcal{F}$ , then  $d(\mathcal{F};W_{r,n})\leq \epsilon$  implies the number of parameters in  $\mathcal{F}$  should be  $\Omega (\epsilon^{-n / r})$ .

# 4.2.2 LOWER BOUND FOR THE COMPLEXITY OF THE DEEPONET

Now, we present the minimum number of parameters in DeepONet. The following theorem gives a criterion on the least number of parameters in trunk net to get the desired error when using the DeepONet. It states that the number of required parameters increases when the target functions are irregular, corresponding to a small  $r$ .  $\mathcal{F}_{DeepONet}(\mathcal{B},\mathcal{T})$  denotes the class of function in DeepONet, induced by the class of branch net  $\mathcal{B}$  and the class of trunk net  $\mathcal{T}$ .

Theorem 2. (Complexity of DeepONet) Let  $\sigma : \mathbb{R} \to \mathbb{R}$  be a universal activation function in  $C^r(\mathbb{R})$  such that  $\sigma$  and  $\sigma'$  are bounded. Suppose that the class of branch net  $\mathcal{B}$  has a bounded Sobolev norm (i.e.  $\| \beta \|_r^s \leq l_1, \forall \beta \in \mathcal{B}$ ). If any non-constant  $\psi \in \mathcal{W}_{r,n}$  does not belong to any class of neural network, then the number of parameters in the class of trunk net  $\mathcal{T}$  is  $\Omega(\epsilon^{-d_y / r})$  when  $d(\mathcal{F}_{DeepONet}(\mathcal{B},\mathcal{T});\mathcal{W}_{r,d_y + m}) \leq \epsilon$ .

The fundamental approach to proving the theorem is substituting a neural network for the inner product between the branch net and trunk net. The neural network with small complexity could approximate the inner product with good performance since the inner product is an infinitely differentiable function. The DeepONet showed defects as it could be replaced with a neural network with  $\mathcal{E}(u) \in [-1, 1]^m$  and  $y \in [-1, 1]^{d_y}$  as input.

A large number of basis  $p$  in DeepONet increases the number of parameters of the trunk net which can be thought of as a target network in HyperDeepONet. Models such as Shift-DeepONet and flexDeepONet could achieve the desired accuracy with a small number of basis, but there was a trade-off in which the first hidden layer of the target network required numerous units. There was no restriction on the dimension of the last hidden layer in the target network for NOMAD which uses a fully nonlinear reconstruction. However, the first hidden layer of the target network had to be wide enough, increasing the number of parameters. Details can be found in the appendix.

For the proposed HyperDeepONet, the sensor values  $\mathcal{E}(u)$  determines the weight and bias of all other layers as well as the weight of the last layer of the target network. Due to the nonlinear activation functions between linear matrix multiplication, it is difficult to replace HyperDeepONet with a single neural network that receives  $[\mathcal{E}(u),y]\in \mathbb{R}^{d_y + m}$  as input. Galanti & Wolf (2020) state that there exists a HyperDeepONet approximation such that the number of parameters in target network is  $O(\epsilon^{-d_y / r})$ . It implies that the proposed method reduces the complexity compared to all the variants of DeepONet.

# 5 EXPERIMENTS

In this section, we verify the effectiveness of the proposed model HyperDeepONet to learn the operators with the complex target function space. To be more specific, we focus on operator learning problems in which the space of output function space is complex. One data point of with a triplet  $(u_{i},y,\mathcal{G}(u)(y))$  so that the input function  $u_{i}$  makes multiple data points for different values of  $y$ . Note that we use the 1000 train input-output function pairs and 200 test pairs for all experiments.

We first consider the identity operator  $\mathcal{G}:u_i\mapsto u_i$  for the toy example. The Chebyshev polynomial is used as the input (=output) for the identity operator problem. The Chebyshev polynomials of the first kind  $T_{l}$  of degree 20 can be written as  $u_{i}\in \{\sum_{l = 0}^{19}c_{l}T_{l}(x)|c_{l}\in [-1 / 4,1 / 4]\}$  with random sampling  $c_{l}$  from uniform distribution  $U[-1 / 4,1 / 4]$ .

The differentiation operator  $\mathcal{G}: u_i \mapsto \frac{d}{dx} u_i$  is considered for the second problem. Previous works handled the anti-derivative operator which makes the output function smoother by averaging (Lu et al., 2019; 2022). Here, we choose the differentiation operator instead of the anti-derivative operator to focus on operator learning when the operator's output function space is complex. We first sample the output function  $\mathcal{G}(u)$  from the above Chebyshev polynomial of degree 20. The input function is generated using the numerical method that integrates the output function.

Table 1: The mean relative  $L^2$  test error with standard deviation for the identity operator and the differentiation operator. The DeepOnet, its variants, and the HyperDeepONet use the target network  $d_y \rightarrow 20 \rightarrow 20 \rightarrow 10 \rightarrow 1$  with tanh activation function. Five training trials are performed independently.  

<table><tr><td>Model</td><td>DeepONet</td><td>Shift</td><td>Flex</td><td>NOMAD</td><td>Hyper(ours)</td></tr><tr><td>Identity</td><td>0.578±0.003</td><td>0.777±0.018</td><td>0.678±0.062</td><td>0.5783±0.020</td><td>0.036±0.005</td></tr><tr><td>Differentiation</td><td>0.559±0.001</td><td>0.624±0.015</td><td>0.562±0.016</td><td>0.558±0.003</td><td>0.127±0.043</td></tr></table>

![](images/3c0ebaad8193ff1f7332757b789709685cb5b41122626a39fe1656d4e6e49811.jpg)  
Figure 4: One test data example of differentiation operator problem.

![](images/dd13a9ba5a86b59526f40b9fad9de447886cfbcacca54963cdc58f341df86b19.jpg)

![](images/7fcbda6363fc6dbc746a8fd70b153b948c1d5302fab38e30c2af288161973d65.jpg)

Lastly, the solution operators of PDEs are considered. We deal with two problems which has the complex target function in previous works (Lu et al., 2022; Hadorn, 2022). For these two solution operator problems, we use the dataset generated in Lu et al. (2022). We consider the solution operator of Burgers' equation which maps the initial condition  $w_0(x) = w(0,x)$  to the solution  $w(1,x)$  at  $t = 1$ , i.e.,  $\mathcal{G}:w_0(x)\mapsto w(1,x)$ . The solution of the Burgers' equation has a discontinuity in a short time, although the initial input function is smooth. The input function  $w_{0}(x)$  is sampled from a Gaussian random field. The solution operator of the advection equation is also considered a mapping from the rectangle shape initial input function  $w_{0}(x) = w(0,x)$  to the solution  $w(0.5,x)$  at  $t = 0.5$ , i.e.,  $\mathcal{G}:w_0(x)\mapsto w(0.5,x)$ . Detail explanation is provided in Appendix C.

Expressivity of target network We compare the expressivity of the small target network using different models. We focus on the identity operator and the differentiation operator in this experiment. All models employ the small target network  $d_y \to 20 \to 20 \to 10 \to 1$  with the hyperbolic tangent activation function. The branch net and the additional networks (scale net, shift net, pre-net, and hypernetwork) in five models also use the same network size as the target network.

Table 1 shows the result on two operator learning problems. The DeepONet and other models have high error to learn complex operator when the small target network is used. On the other hand, the HyperDeepONet has lower error than the other models. This is consistent with the theorem in the previous section that HyperDeepONet can achieve improved approximations than the DeepONet when the complexity of the target network is the same. Figure 4 shows a prediction on differentiation operator which has highly complex target function. The same trends are observed when the number of layers in the branch net and the hypernetwork vary (Figure 7, 8), and the activation function or the number of sensors changes (Appendix C).

Same number of learnable parameters The previous experiments compare the models using the same target network structure. It seems that the larger number of parameters in the proposed

Table 2: The mean relative  $L^2$  test error with standard deviation for two solution operator learning problem.  $N_{\theta}$  and #Param denote the number of parameters in target network and the number of learnable parameters, respectively. Five training trials are performed independently.  

<table><tr><td colspan="2"></td><td>Branch (Hyper)</td><td>Target</td><td>#Param</td><td>Rel error</td></tr><tr><td rowspan="2">Burgers</td><td>DeepONet</td><td>m-128-128-128-128</td><td>dy-128-128-128-128-1</td><td>115K</td><td>0.0391±0.0040</td></tr><tr><td>Hyper(ours)</td><td>m-66-66-66-66-66-Nθ</td><td>dy-20-20-20-20-1</td><td>114K</td><td>0.0196±0.0044</td></tr><tr><td rowspan="2">Advection</td><td>DeepONet</td><td>m-256-256</td><td>dy-256-256-256-256-1</td><td>274K</td><td>0.0046±0.0017</td></tr><tr><td>Hyper(ours)</td><td>m-70-70-70-70-70-Nθ</td><td>dy-33-33-33-33-33-1</td><td>268K</td><td>0.0048±0.0009</td></tr></table>

![](images/1b23f6b4dcda6a19e7f9227fc50445a016781e265f970854ec6f4ba9ce216c73.jpg)

![](images/1c0aa589ff7911dc9de4b2fc55b6d9d672881513f781b360660fe712ca5a32da.jpg)

![](images/14385c89db74ccb94f0e27f03c678f9f7ad11855dab4ff7cde7d883e701eae21.jpg)

![](images/ea5adb29c56cf0e23b2d6efb3fc424e47dd23df5d584a0791da9d16aa9b67cd6.jpg)  
Figure 5: One test data example of prediction on the advection equation (First row) and Burgers' equation (Second row) using the DeepONet and the HyperDeepONet.

![](images/0a41443d5b5cf4ce392c74adf92e9e8aa26af34ec65b6296199b2fcd6186f391.jpg)

![](images/ef3a8af39d300cf50917e21ca47fcc7892869097fc9dc1adae609e37a96f9190.jpg)

HyperDeepONet makes the operator approximation more accurate. In this section, the comparison between the DeepONet and the HyperDeepONet is considered when using the same number of learnable paramters. We focus on the solution operator of the advection equation and Burgers' equation in this experiment.

For the two solution operator learning problems, we use the same hyperparameters proposed in Lu et al. (2022) for DeepONet. We use the smaller target network with the larger hypernetwork for the HyperDeepONet to compare the DeepONet. The number of learnable parameters for the hypernetwork is similar but smaller than the DeepONet. Note that the vanilla DeepONet is used without the output normalization or the boundary condition enforcing techniques explained in Lu et al. (2022) to focus on the primary limitation of the DeepONet. More Details is in Appendix C. Table 2 shows that the HyperDeepONet achieves a similar or better performance than the DeepONet when the two models use the same number of learnable parameters. For advection equation problems, the hyperdeeponet has a slightly higher error, but this error is close to perfect operator prediction. It shows that not only the complexity of target network but also the number of learnable parameters can be reduced to obtain the desired accuracy using the HyperDeepONet. It is observed that the HyperDeepONet learns the complex target functions faster than the DeepONet (Figure 5).

# 6 CONCLUSION AND DISCUSSION

In this work, the HyperDeepONet is developed to overcome the limitation of expressivity that Deep-ONet suffers from. The method of incorporating an additional network and a nonlinear reconstructor could not thoroughly solve this limitations. The hypernetwork, which involves multiple weights simultaneously, was a desired complexity-reducing structure based on theory and experiments.

We only focused on the case when the hypernetwork and the target network are constructed as fully connected neural networks. In the future, the structure of the two networks can be replaced with CNN or ResNet, as the structure of the branch net and trunk net of DeepONet can be changed to another network (Lu et al., 2022). Additionally, it seems interesting to research a simplified modulation network proposed by Mehta et al. (2021) still has the same expressivity as HyperDeepONet.

The output of the hypernetwork would be high-dimensional when the size of the target network is very large. There are several studies to reduce the number of outputs in the hypernetwork (Ha et al., 2016; Pawlowski et al., 2017). The chunked HyperDeepONet, developed by Von Oswald et al. (2019), can be used with a trade-off between accuracy and memory when a model with reduced parameters is required for the large target network. Some techniques from implicit neural representation can improve the expressivity of target network (Sitzmann et al., 2020). Using a sine function as an activation function with preprocessing will promote the expressivity of the target network. We also leave the research on the class of activation functions satisfying the assumption except for hyperbolic tangent or sigmoid functions as future work.

# REFERENCES

Saakaar Bhatnagar, Yaser Afshar, Shaowu Pan, Karthik Duraisamy, and Shailendra Kaushik. Prediction of aerodynamic flow fields using convolutional neural networks. Comput. Mech., 64 (2):525-545, 2019. ISSN 0178-7675. doi: 10.1007/s00466-019-01740-0. URL https://doi.org/10.1007/s00466-019-01740-0.  
Shengze Cai, Zhicheng Wang, Lu Lu, Tamer A Zaki, and George Em Karniadakis. Deepm&mnet: Inferring the electroconvection multiphysics fields based on operator approximation by neural networks. Journal of Computational Physics, 436:110296, 2021.  
Tianping Chen and Hong Chen. Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. IEEE Transactions on Neural Networks, 6(4):911-917, 1995.  
Zhiqin Chen and Hao Zhang. Learning implicit fields for generative shape modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5939-5948, 2019.  
Tejalal Choudhary, Vipul Mishra, Anurag Goswami, and Jagannathan Sarangapani. A comprehensive survey on model compression and acceleration. Artificial Intelligence Review, 53(7): 5113-5155, 2020.  
Filipe de Avila Belbute-Peres, Yi-fan Chen, and Fei Sha. Hyperpinn: Learning parameterized differential equations with physics-informed hypernetworks. In The Symbiosis of Deep Learning and Differential Equations, 2021.  
Tim De Ryck and Siddhartha Mishra. Generic bounds on the approximation error for physics-informed (and) operator learning. arXiv preprint arXiv:2205.11393, 2022.  
Weinan E and Bing Yu. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. Commun. Math. Stat., 6(1):1-12, 2018. ISSN 2194-6701. doi: 10. 1007/s40304-018-0127-z. URL https://doi.org/10.1007/s40304-018-0127-z.  
Tomer Galanti and Lior Wolf. On the modularity of hypernetworks. Advances in Neural Information Processing Systems, 33:10409-10419, 2020.  
Xiaoxiao Guo, Wei Li, and Francesco Iorio. Convolutional neural networks for steady flow approximation. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 481-490, 2016.  
David Ha, Andrew Dai, and Quoc V Le. Hypernetworks. arXiv preprint arXiv:1609.09106, 2016.  
Patrik Simon Hadorn. Shift-deponet: Extending deep operator networks for discontinuous output functions. ETH Zurich, Seminar for Applied Mathematics, 2022.  
Boris Hanin and Mark Sellke. Approximating continuous functions by relu nets of minimal width. arXiv preprint arXiv:1710.11278, 2017.  
Rakhoon Hwang, Jae Yong Lee, Jin Young Shin, and Hyung Ju Hwang. Solving pde-constrained control problems using operator learning. arXiv preprint arXiv:2111.04941, 2021.  
Yuehaw Khoo, Jianfeng Lu, and Lexing Ying. Solving parametric PDE problems with artificial neural networks. European J. Appl. Math., 32(3):421-435, 2021. ISSN 0956-7925. doi: 10.1017/S0956792520000182. URL https://doi.org/10.1017/S0956792520000182.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Sylwester Klocek, Łukasz Maziarka, Maciej Wołczyk, Jacek Tabor, Jakub Nowak, and Marek Śmieja. Hypernetwork functional image representation. In International Conference on Artificial Neural Networks, pp. 496-510. Springer, 2019.  
Nikola Kovachki, Samuel Lanthaler, and Siddhartha Mishra. On universal approximation and error bounds for fourier neural operators. Journal of Machine Learning Research, 22:Art-No, 2021a.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces. arXiv preprint arXiv:2108.08481, 2021b.  
Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David Hall, Andrea Miele, Karthik Kashinath, and Animashree Anandkumar. Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators. arXiv preprint arXiv:2208.05419, 2022.  
Samuel Lanthaler, Siddhartha Mishra, and George E Karniadakis. Error estimates for deeponets: A deep learning framework in infinite dimensions. Transactions of Mathematics and Its Applications, 6(1):tnac001, 2022.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020a.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Multipole graph neural operator for parametric partial differential equations. arXiv preprint arXiv:2006.09535, 2020b.  
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Graph kernel network for partial differential equations. arXiv preprint arXiv:2003.03485, 2020c.  
Chensen Lin, Zhen Li, Lu Lu, Shengze Cai, Martin Maxey, and George Em Karniadakis. Operator learning for predicting multiscale bubble growth dynamics. The Journal of Chemical Physics, 154(10):104118, 2021.  
Lu Lu, Pengzhan Jin, and George Em Karniadakis. Deeponet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators. arXiv preprint arXiv:1910.03193, 2019.  
Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218-229, 2021.  
Lu Lu, Xuhui Meng, Shengze Cai, Zhiping Mao, Somdatta Goswami, Zhongqiang Zhang, and George Em Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering, 393:114778, 2022.  
Zhiping Mao, Lu Lu, Olaf Marxen, Tamer A. Zaki, and George Em Karniadakis. DeepM&Mnet for hypersonics: predicting the coupled flow and finite-rate chemistry behind a normal shock using neural-network approximation of operators. J. Comput. Phys., 447:Paper No. 110698, 24, 2021. ISSN 0021-9991. doi: 10.1016/j.jcp.2021.110698. URL https://doi.org/10.1016/j.jcp.2021.110698.  
Ishit Mehta, Michael Gharbi, Connelly Barnes, Eli Shechtman, Ravi Ramamoorthi, and Manmohan Chandraker. Modulated periodic activations for generalizable local functional representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14214-14223, 2021.  
Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4460-4470, 2019.  
Hrushikesh N Mhaskar. Neural networks for optimal approximation of smooth and analytic functions. Neural computation, 8(1):164-177, 1996.  
MG Sarwar Murshed, Christopher Murphy, Daqing Hou, Nazar Khan, Ganesh Ananthanarayanan, and Faraz Hussain. Machine learning at the network edge: A survey. ACM Computing Surveys (CSUR), 54(8):1-37, 2021.

Shaowu Pan, Steven L Brunton, and J Nathan Kutz. Neural implicit flow: a mesh-agnostic dimensionality reduction paradigm of spatio-temporal data. arXiv preprint arXiv:2204.03216, 2022.  
Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 165-174, 2019.  
Nick Pawlowski, Andrew Brock, Matthew CH Lee, Martin Rajchl, and Ben Glocker. Implicit weight uncertainty in neural networks. arXiv preprint arXiv:1711.01297, 2017.  
Michael Prasthofer, Tim De Ryck, and Siddhartha Mishra. Variable-input deep operator networks. arXiv preprint arXiv:2205.11404, 2022.  
M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys., 378:686-707, 2019. ISSN 0021-9991. doi: 10.1016/j.jcp.2018.10.045. URL https://doi.org/10.1016/j.jcp.2018.10.045.  
Jacob H Seidman, Georgios Kissas, Paris Perdikaris, and George J Pappas. Nomad: Nonlinear manifold decoders for operator learning. arXiv preprint arXiv:2206.03551, 2022.  
Justin Sirignano and Konstantinos Spiliopoulos. DGM: a deep learning algorithm for solving partial differential equations. J. Comput. Phys., 375:1339-1364, 2018. ISSN 0021-9991. doi: 10.1016/j.jcp.2018.08.029. URL https://doi.org/10.1016/j.jcp.2018.08.029.  
Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in Neural Information Processing Systems, 33:7462-7473, 2020.  
Simone Venturi and Tiernan Casey. Svd perspectives for augmenting deeponet flexibility and interpretability. arXiv preprint arXiv:2204.12670, 2022.  
Johannes Von Oswald, Christian Henning, João Sacramento, and Benjamin F Grewe. Continual learning with hypernetworks. arXiv preprint arXiv:1906.00695, 2019.  
Sifan Wang, Hanwen Wang, and Paris Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed deeponets. Science advances, 7(40):eabi8605, 2021.  
Sifan Wang, Hanwen Wang, and Paris Perdikaris. Improved architectures and training algorithms for deep operator networks. Journal of Scientific Computing, 92(2):1-42, 2022.  
Yinhao Zhu, Nicholas Zabaras, Phaedon-Stelios Koutsourelakis, and Paris Perdikaris. Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data. J. Comput. Phys., 394:56-81, 2019. ISSN 0021-9991. doi: 10.1016/j.jcp.2019.05.024. URL https://doi.org/10.1016/j.jcp.2019.05.024.
