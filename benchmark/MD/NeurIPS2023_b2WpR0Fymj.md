# On the Universal Approximation Properties of Deep Neural Networks using MAM Neurons

Anonymous Author(s)

Affiliation

Address

email

# Abstract

As Deep Neural Networks (DNNs) are trained to perform tasks of increasing complexity, their size grows, presenting several challenges when it comes to deploying them on edge devices that have limited resources. To cope with this, a recently proposed approach hinges on substituting the classical Multiply-and-Accumulate (MAC) neurons in the hidden layers of a DNN with other neurons called Multiply-And-Max/min (MAM) whose selective behaviour helps identifying important interconnections and allows extremely aggressive pruning. Hybrid structures with MAC and MAM neurons promise a reduction in the number of interconnections that outperforms what can be achieved with MAC-only structures by more than an order of magnitude. However, by now, the lack of any theoretical demonstration of their ability to work as universal approximators limits their diffusion. Here, we take a first step in the theoretical characterization of the capabilities of MAM&MAC networks. In details, we prove two theorems that confirm that they are universal approximators providing that two hidden MAM layers are followed either by a MAC neuron without nonlinearity or by a normalized variant of the same. Approximation quality is measured either in terms of the first-order  $L^p$  Sobolev norm or by the  $L^\infty$  norm.

# 1 Introduction

Deep Neural Networks (DNNs) solve complex tasks leveraging a massive number of trainable parameters. Yet, thanks to the recent increasing interest in mobile Artificial Intelligence, there has been a growing emphasis on designing lightweight structures able to run on devices with constrained resources. This can be obtained by removing parameters that do not appreciably influence performance by means of one of the many pruning techniques that have been proposed. Some approaches entail removing, in a single shot, individual interconnections or entire neurons once the DNN has been trained, while others methods are applied iteratively, and require multiple rounds of training. These techniques eliminate interconnections but do not alter the underlying Multiply-and-ACumulate (MAC) paradigm that governs the neuron's inner functioning.

In [1, 2], the authors address the challenge of designing neural networks that can have a smaller memory footprint presenting a novel neuron model based on the Multiply-And-Max/min (MAM) paradigm that can be substituted to classical MAC neurons in the hidden layers of a DNN to allow a more aggressive pruning of interconnections, while substantially preserving the network performance. In a standard MAC-based neuron, inputs are modulated independently of each other through multiplication with their respective weights, and the resulting products are then summed into a single quantity. As MAC neurons, MAM neurons multiply each input by a weight but then only the maximum and the minimum quantity of the products are summed together.

In formulas, if  $v_{1}, v_{2}, \ldots$  are the inputs after being multiplied by their respective weights, the output  $u$  of a MAM neuron is

$$
u = \left[ \max  _ {j} v _ {j} + \min  _ {j} v _ {j} + b \right] ^ {+} \tag {1}
$$

where  $b$  is the bias and  $[\cdot ]^{+} = \max \{0,\cdot \}$  represents the nowadays common ReLU nonlinearity.

It is shown empirically that, starting from an architecture originally designed using MAC neurons, one may substitute them with MAM neurons in several hidden layers and use a proper training strategy to achieve the same performances as the corresponding MAC-only network. Yet, in the resulting hybrid network, one may leverage the extremely selective behaviour of min and max operations to reduce very aggressively the number of weights. MAM neurons can be pruned with almost every technique proposed in the literature with little to no modifications. As a motivating example, Table 1 reports some of the results described in [1] showing cases in which, once the quality level is set (in this case to  $3\%$  less accuracy than the original non-pruned network), MAM neuron substitution, retraining and pruning reduce the number of weights 1 to 2 orders of magnitude more than what is obtained by pruning the original MAC-only network. Moreover, these neurons can also be pruned iteratively requiring less training iterations to guarantee a given accuracy compared to standard MAC neurons.

Table 1: Approximate remaining interconnections in the hidden fully-connected layers with one-shot global magnitude pruning built either with MAC or MAM neurons.  

<table><tr><td></td><td>AlexNet + Cifar-10</td><td>AlexNet + Cifar-100</td><td>VGG-16 + ImageNet</td></tr><tr><td>Top-1 accuracy (3% lower than non-pruned network)</td><td>87.69%</td><td>63.89%</td><td>61.03%</td></tr><tr><td>Surviving interconnections (MAC)</td><td>1.01%</td><td>25.01%</td><td>10.82%</td></tr><tr><td>Surviving interconnections (MAM)</td><td>0.06%</td><td>0.26%</td><td>0.04%</td></tr></table>

Though the equivalence between MAC-only and MAM&MAC networks has been demonstrated in practice, a change in the model of some neurons opens the problem of the abstract capability of such hybrid architectures. This contribution is a step forward in clarifying that, despite the locally different input-output relationships, also hybrid MAM&MAC networks enjoy some universal approximation capabilities analogous to those of the MAC-only networks.

# 1.1 Brief background on universal approximation properties

The development of models with universal approximation properties has been a significant breakthrough in many fields of science and engineering. In 1989 [3] proved that a network with a single hidden layer could approximate any continuous function, given enough hidden neurons. Some years later, [4] and [5] showed that also fuzzy systems could approximate any continuous function to arbitrary accuracy. These works were later extended to multiple inputs and outputs, demonstrating the universal approximation properties of fuzzy systems more broadly ([6, 7]). In the following years, a large number of researchers have studied the universal approximation properties of neural networks with MAC neurons in the case of bounded depth and arbitrary width ([8, 9]), bounded width and arbitrary depth ([10, 11, 12]) and bounded width and depth ([13, 14]). In the recent work [15], authors obtained the optimal minimum width bound of a neural network with arbitrary depth to retain universal approximation capabilities.

The research in this field is still very active and aims at proving the universal approximation capabilities of networks with different architectural or computational paradigm choices, such as deep convolutional neural networks [16], dropout neural networks [17], networks representing probability distributions [18] and spiking neural networks [19].

# 2 Mathematical model

We indicate with  $\mathcal{L}(\cdot)$  a fully connected layer in which all neurons are based on the MAM paradigm (1). We consider networks with  $N$  inputs collected in the vector  $\pmb{x} = (x_{1},\dots,x_{N})$ , two MAM hidden layers producing a vector  $\pmb{z}(\pmb{x}) = (z_{1}(\pmb{x}),z_{2}(\pmb{x}),\dots) = \mathcal{L}^{\prime \prime}\left(\mathcal{L}^{\prime}(\pmb{x})\right)$  and a single output  $Z(\pmb{x})\in \mathbb{R}$  produced by a final layer that computes either the normalized linear combination

$$
Z (\boldsymbol {x}) = \frac {\sum_ {k} c _ {k} z _ {k} (\boldsymbol {x})}{\sum_ {k} z _ {k} (\boldsymbol {x})} \tag {2}
$$

or the linear combination

$$
Z (\boldsymbol {x}) = \sum_ {k} c _ {k} z _ {k} (\boldsymbol {x}) \tag {3}
$$

We normalize the input domain by assuming  $x_{i} \in \mathbb{X} = [0,1]$  for  $i = 1, \dots, N$  and indicate with  $\mathcal{Z}^*$  the family of functions in (2) while with  $\mathcal{Z}$  the analogous family of functions in (3). Smoothness conditions on our target functions  $f: \mathbb{X}^N \mapsto \mathbb{R}$  is formalized by assuming that they belong to  $\mathcal{C}^d(\mathbb{X}^N)$ , i.e., that their  $d$ -th order derivatives are continuous. Distances between functions are measured by means of the norms defined as

$$
\left\| \phi \right\| _ {k, p} = \left[ \int_ {\mathbb {X} ^ {N}} | \phi (x) | ^ {p} d x + k \sum_ {j = 1} ^ {N} \int_ {\mathbb {X} ^ {N}} \left| \frac {\partial \phi}{x _ {j}} (x) \right| ^ {p} d x \right] ^ {1 / _ {p}}
$$

with  $k = \{0,1\}$  and  $p \geq 1$ .

# 3 Main results

Within the above framework, we prove two theorems that describe the universal approximation properties of DNNs using MAM neurons in the hidden layers.

Theorem 1. For any function  $f \in \mathcal{C}^0(\mathbb{X}^N)$  and any prescribed level of tolerance  $\epsilon > 0$ , there is a  $Z \in \mathcal{Z}^*$  such that  $\| f - Z \|_{0,\infty} \leq \epsilon$ .

Theorem 2. For any function  $f \in \mathcal{C}^2(\mathbb{X}^N)$ , any prescribed level of tolerance  $\epsilon > 0$  and finite  $p \geq 1$ , there is a  $Z \in \mathcal{Z}$  such that  $\| f - Z \|_{1,p} \leq \epsilon$ .

The proofs of both theorems are reported in Section 6 and are constructive. In particular, subnetworks in the cascade  $z(\boldsymbol{x}) = \mathcal{L}''\left(\mathcal{L}'(\boldsymbol{x})\right)$  are identified and programmed to make each  $z_k(\boldsymbol{x})$  a weakly unimodal piecewise-linear function of the inputs, whose maximum is 1 and is reached in a hyperrectangular subset of the domain, while the function vanishes for points far from the center of that hyper-rectangle. The shapes and positions of these functions can then be designed along with the values of the weights  $c_k$  so that their combination by means of either (2) or (3) is capable of approximating the target function arbitrarily well as measured either by  $\| \cdot \|_{1,p}$  or  $\| \cdot \|_{0,\infty}$ .

# 4 Examples

Figure 1 proposes a visual representation of the constructions behind Theorem 1 and Theorem 2 for  $N = 2$ . From left to right, we report the target function  $f: \mathbb{X}^2 \to \mathbb{R}$

$$
f \left(x _ {1}, x _ {2}\right) = \frac {\left(4 x _ {1} - 2\right) \left(4 x _ {2} - 2\right) \left(4 x _ {1} + \frac {1}{2}\right)}{1 + \left(4 x _ {1} - 2\right) ^ {2} + \left(4 x _ {2} - 2\right) ^ {2}} + 3 \tag {4}
$$

and its approximation  $Z \in \mathcal{Z}^*$  implied by the proof of Theorem 1 and its approximation  $X \in \mathcal{Z}$  implied by the proof of Theorem 2. In both cases the parameter  $n$  used in Section 6 is set to  $n = 7$ .

![](images/69d774cbd4d20b81e14b4880a0373f0798267f655cd67d240f2ae5ca461bb502.jpg)  
Figure 1: Three dimensional plot of a target function  $f(x_{1},x_{2})$  and of its two approximations  $Z(x_{1},x_{2})\in \mathcal{Z}^{*}$  implied by Theorem 1 and  $X(x_{1},x_{2})\in \mathcal{Z}$  by Theorem 2.

![](images/1f40a6404f9412a8dd6a3fbccfecedc93be35441bb34af4db4038085e1946c33.jpg)

![](images/d48bfd33c15ade870a9d627c007d5a2d4f7b6fd3e1c3c32619d7fec2d3aca68d.jpg)

# 5 Limitations

Theorem 1 and Theorem 2 rely on networks in which constraints are put neither on the layer width nor on the total number of neurons. Hence, despite proving universal approximation capabilities, they do not imply efficient approximation. Yet, such theoretical limitation is never strongly experienced in practice, since MAM networks are able to guarantee acceptable performance in real use cases. Nevertheless, a deeper look at universal approximation aimed at meeting efficiency will be the focus of future analysis.

# 6 Network construction and proofs of Theorems

# 6.1 Network construction

The aim of this subsection is to show that our network can be programmed to make the outputs of the second hidden layer specific weakly unimodal piecewise-linear functions  $z_{k}(\pmb{x})$  of the inputs.

Lemma 1. Let  $z$  be any of the outputs of the second hidden layer. For  $N > 1$  and any choice of the quantities  $\omega_{1},\ldots ,\omega_{N}\in [0,1]$ ,  $l_{1},\ldots ,l_{N}\geq 0$ ,  $\delta_1^L,\dots,\delta_N^L\geq 0$ , and  $\delta_1^R,\dots,\delta_N^R\geq 0$ , the two MAM hidden layers can be programmed to yield

$$
z (\boldsymbol {x}) = \left[ 1 - \Delta (\boldsymbol {x}) \right] ^ {+} \tag {5}
$$

where

$$
\Delta (\boldsymbol {x}) = \max  _ {i \in \{1,.., N \}} \left\{0, \frac {\left| x _ {i} - \omega_ {i} \right| - l _ {i}}{\left\{ \begin{array}{l l} \delta_ {i} ^ {L} & i f x _ {i} <   \omega_ {i} \\ \delta_ {i} ^ {R} & i f x _ {i} \geq \omega_ {i} \end{array} \right.} \right\} \tag {6}
$$

Proof of Lemma 1. We assume that neurons in the first hidden layer come in pairs  $(y_1^{\mathrm{L}}, y_1^{\mathrm{R}}, y_2^{\mathrm{L}}, y_2^{\mathrm{R}}, \ldots) = \mathcal{L}'(\boldsymbol{x})$  and the output of a pair depends on only one of the inputs.

Without any loss of generality, we assume that  $y_{i}^{\mathrm{L}}$  and  $y_{i}^{\mathrm{R}}$  depend only on  $x_{i}$  for  $i = 1, \dots, N$  while all the other  $N - 1$  input weights are set to 0. The other outputs of the first hidden layer are involved in the computation of the outputs of the second hidden layer further to the  $z$  we are considering.

For  $y_{i}^{\mathrm{L}}$  the non-null input weight is equal to  $-\frac{1}{\delta_i^{\mathrm{L}}}$  and the bias is  $(\omega_{i} - l_{i}) / \delta_{i}^{\mathrm{L}}$ , while for  $y_{i}^{\mathrm{R}}$  the non-null input weight is equal to  $\frac{1}{\delta_i^{\mathrm{R}}}$  and bias is  $(-\omega_{i} - l_{i}) / \delta_{i}^{\mathrm{R}}$ . By recalling (1) one gets

$$
y _ {i} ^ {\mathrm {L}} = \left[ \frac {- x _ {i} + \omega_ {i} - l _ {i}}{\delta_ {i} ^ {\mathrm {L}}} \right] ^ {+} \quad \text {a n d} \quad y _ {i} ^ {\mathrm {R}} = \left[ \frac {x _ {i} - \omega_ {i} - l _ {i}}{\delta_ {i} ^ {\mathrm {R}}} \right] ^ {+} \tag {7}
$$

In the second hidden layer, the neuron computing the  $z$  we consider has all input weights equal to 0 but those connecting to  $y_1^{\mathrm{L}}, y_1^{\mathrm{R}}, \ldots, y_N^{\mathrm{L}}, y_N^{\mathrm{R}}$ . Non-null input weights are equal to  $-1$  and the bias is 1 so that

![](images/6f4b97de24999feb3c1a8cebdb7f8ef186cb9c15f0f819887368abc34b91da39.jpg)  
Figure 2: Three dimensional plot of a generic  $z_{\omega}(\pmb{x})$  for  $N = 2$  and its contour plot showing the role of the various parameters.

![](images/be30e7928be3683911240f220b2f7c1f7ab0f6f0bda1447a20e8fbb9aa071def.jpg)

$$
z = \left[ \max  _ {i \in \{1,.., N \}} \left\{0, - y _ {i} ^ {\mathrm {L}}, - y _ {i} ^ {\mathrm {R}} \right\} + \min  _ {i \in \{1,.., N \}} \left\{0, - y _ {i} ^ {\mathrm {L}}, - y _ {i} ^ {\mathrm {R}} \right\} + 1 \right] ^ {+} = \left[ 1 - \max  _ {i \in \{1,.., N \}} \left\{y _ {i} ^ {\mathrm {L}}, y _ {i} ^ {\mathrm {R}} \right\} \right] ^ {+} \tag {8}
$$

Considering the last expression, note that, if  $x_{i} \geq \omega_{i}$  then  $y_{i}^{\mathrm{R}} \geq 0$  and  $y_{i}^{\mathrm{L}} = 0$  while, if  $x_{i} < \omega_{i}$  then  $y_{i}^{\mathrm{R}} = 0$  and  $y_{i}^{\mathrm{L}} \geq 0$ . Hence, without loss of generality, we may assume that  $x_{i} \geq \omega_{i}$  for  $i = 1, \dots, N$ , being all other cases a variation of this one by suitable symmetry and scaling. With this,  $y_{i}^{\mathrm{L}} = 0$  for  $i = 1, \dots, N$  and (8) becomes

$$
z = \left[ 1 - \max  _ {i = 1, \dots , N} \left[ \frac {x _ {i} - \omega_ {i} - l _ {i}}{\delta_ {i} ^ {\mathrm {R}}} \right] ^ {+} \right] ^ {+} = \left[ 1 - \max  _ {i = 1, \dots , N} \left\{0, \frac {x _ {i} - \omega_ {i} - l _ {i}}{\delta_ {i} ^ {\mathrm {R}}} \right\} \right] ^ {+} \tag {9}
$$

that is equivalent to the thesis.

To interpret Lemma 1 note that  $\Delta(\pmb{x})$  is a scaled measure of how far the input vector  $\pmb{x}$  is from the hyper-rectangle centered at  $\omega = (\omega_1, \dots, \omega_N)$  with sides  $2l_1, \dots, 2l_N$ . Hence,  $z(\pmb{x})$  is maximum and equal to 1 if  $\pmb{x}$  belongs to such a hyper-rectangle and has a piecewise-linear decreasing profile when  $\pmb{x}$  gets further from  $\omega$ . Figure 2 reports an example of a  $z(\pmb{x})$  when  $N = 2$ .

In the following, we will assume that each neuron in the second hidden layer matches a whole subnetwork as implied by Lemma 1. With this, we may re-index the outputs of the second hidden layer as  $z_{\omega}(\boldsymbol{x})$  associating each of them with the center of the hyper-rectangle in which  $z_{\omega}(\boldsymbol{x}) = 1$ . The same is done with the corresponding weights  $c_{\omega}$  in the output layers.

# 6.2 Universal approximation properties with normalized linear output neuron

Given a positive integer  $n$ , define  $\Omega = \{0, \frac{1}{n}, \frac{2}{n}, \dots, 1\}^N$  and include in the two hidden layers all the subnetworks implied by Lemma 1 to implement the function  $z_{\omega}(\boldsymbol{x})$  for each  $\omega \in \Omega$ .

In each of these subnetworks set  $\delta_i^{\mathrm{L}} = \delta_i^{\mathrm{R}} = \delta = 1 / n$  for  $i = 1,\dots ,N$  and  $l_{i} = 0$  for  $i = 1,\ldots ,N$

With this,  $z_{\omega}(\pmb{x})$  is and  $(N + 1)$ -dimensional pyramid whose base is an  $N$ -dimensional hypercube with sides of length  $2\delta$  and center in  $\omega$ .

Proof of Theorem 1. Note first that for any given  $\pmb{x} \in \mathbb{X}^N$ , only a limited number of functions  $z_{\omega}(\pmb{x})$  are not null. In particular, if  $k_{i} = \lfloor nx_{i} \rfloor$  for  $i = 1, \dots, N$  is the largest integer not exceeding  $n x_{i}$ , then  $z_{\omega}(\pmb{x}) > 0$  only if  $\pmb{\omega}$  belongs to the set  $\Omega_{\pmb{x}} = \{k_1\delta, (k_1 + 1)\delta\} \times \dots \times \{k_N\delta, (k_N + 1)\delta\}$  that contains the  $2^{N}$  corners of the  $N$ -dimensional hypercube  $C_{\pmb{x}} = [k_1\delta, (k_1 + 1)\delta] \times \dots \times [k_N\delta, (k_N + 1)\delta]$ . Hence, we may evaluate  $Z(\pmb{x})$  focusing on functions  $z_{\omega}(\pmb{x})$  with  $\omega \in \Omega_{\pmb{x}}$ .

Define the functions

$$
\zeta_ {\omega} (x) = \frac {z _ {\omega} (\boldsymbol {x})}{\sum_ {\omega^ {\prime} \in \Omega} z _ {\omega^ {\prime}} (\boldsymbol {x})} \tag {10}
$$

that are such that  $\sum_{\omega \in \Omega} \zeta_{\omega}(\boldsymbol{x}) = \sum_{\omega \in \Omega_{\infty}} \zeta_{\omega}(\boldsymbol{x}) = 1$  for any  $\boldsymbol{x} \in \mathbb{X}^N$ , and set  $c_{\omega} = f(\omega)$  for each  $\omega \in \Omega$ .

The error  $\| f(\pmb {x}) - Z(\pmb {x})\|_{0,\infty}$  in Theorem 1 can be written as

$$
\left\| f \left(\boldsymbol {x}\right) - \sum_ {\boldsymbol {\omega} \in \Omega_ {\boldsymbol {x}}} f \left(\boldsymbol {\omega}\right) \zeta_ {\boldsymbol {\omega}} \left(\boldsymbol {x}\right) \right\| _ {0, \infty} = \left\| \sum_ {\boldsymbol {\omega} \in \Omega_ {\boldsymbol {x}}} \left[ f \left(\boldsymbol {x}\right) - f \left(\boldsymbol {\omega}\right) \right] \zeta_ {\boldsymbol {\omega}} \left(\boldsymbol {x}\right) \right\| _ {0, \infty} \leq \max  _ {\boldsymbol {x} \in \mathbb {X} ^ {N}} \max  _ {\substack {\boldsymbol {\xi} \in C _ {\boldsymbol {x}} \\ \boldsymbol {\omega} \in \Omega_ {\boldsymbol {x}}}} | f \left(\boldsymbol {\xi}\right) - f \left(\boldsymbol {\omega}\right) |
$$

Since  $f: \mathbb{X}^N \to \mathbb{R}$  is continuous on the compact domain  $\mathbb{X}^N$ , it is also uniformly continuous and, for any given level of tolerance  $\epsilon > 0$ , there is a  $\Delta x$  such that for any  $\pmb{x}', \pmb{x}'' \in \mathbb{X}^N$  with distance  $\| \pmb{x}' - \pmb{x}'' \|_2 \leq \Delta x$  we have  $|f(\pmb{x}') - f(\pmb{x}'')| \leq \epsilon$ . For a given  $\pmb{x}$ , the distance between any  $\pmb{\xi} \in C_x$  and any  $\omega \in \Omega_x$  is  $\| \pmb{\xi} - \pmb{\omega} \|_2 \leq \delta \sqrt{N}$ . Since  $\delta = 1/n$  we can select  $n$  so that

$$
\| f\left(\boldsymbol {x}\right) - Z\left(\boldsymbol {x}\right)\|_{0,\infty}\leq \max_{\boldsymbol {x}\in \mathbb{X}^{N}}\max_{\substack{\boldsymbol {\xi}\in C_{\boldsymbol{x}}\\ \boldsymbol {\omega}\in \Omega_{\boldsymbol{x}}}}\left|f\left(\boldsymbol {\xi}\right) - f\left(\boldsymbol {\omega}\right)\right|\leq \epsilon
$$

# 6.3 Universal approximation properties with linear output neuron

In this case, the approximation capabilities of our network over the whole domain depend on the local behaviour of subnetworks converging not in a single second-hidden-layer neuron but in  $2N$  second-hidden-layer neurons.

Formally, given a center  $\omega \in \mathbb{X}^N$  we include in a subnetwork neurons of the second hidden layer with outputs labelled  $z_{\omega^{1-}}$ ,  $z_{\omega^{1+}}$ , ...,  $z_{\omega^{N-}}$ ,  $z_{\omega^{N+}}$  as well as all the previous neurons needed to compute such outputs.

The expression of each  $z_{\omega^{j\pm}}$  is given by Lemma 1 and thus is defined by the center point  $\omega^{j\pm} = (\omega_1^{j\pm}, \ldots, \omega_N^{j\pm})$ , by the slopes  $\delta_1^{L,j\pm}, \ldots, \delta_N^{L,j\pm}$  and  $\delta_1^{R,j\pm}, \ldots, \delta_N^{R,j\pm}$ , as well as by the side lengths  $l_1^{j\pm}, \ldots, l_N^{j\pm}$ .

In a subnetwork, everything depends on two quantities  $\delta, \ell \geq 0$  that are used to set

$$
\omega_ {i} ^ {j \pm} = \left\{ \begin{array}{l l} \omega_ {i} & \text {i f} i \neq j \\ \omega_ {i} \pm \ell & \text {i f} i = j \end{array} \right. \qquad \qquad l _ {i} ^ {j \pm} = \left\{ \begin{array}{l l} \ell & \text {i f} i \neq j \\ 0 & \text {i f} i = j \end{array} \right.
$$

$$
\begin{array}{l} \delta_ {i} ^ {\mathrm {R}, j -} = \delta \\ \delta_ {i} ^ {\mathrm {L}, j -} = \left\{ \begin{array}{l l} \delta & \text {i f} i \neq j \\ 2 \ell & \text {i f} i = j \end{array} \right. \end{array} \qquad \begin{array}{l} \delta_ {i} ^ {\mathrm {R}, j +} = \left\{ \begin{array}{l l} \delta & \text {i f} i \neq j \\ 2 \ell & \text {i f} i = j \end{array} \right. \\ \delta_ {i} ^ {\mathrm {L}, j +} = \delta \end{array}
$$

for  $i,j = 1,\dots ,N$

To give some intuitive grounding to the above definitions, Figure 3 reports example profiles for 4 output functions  $z_{\omega^{1-}}$ ,  $z_{\omega^{1+}}$ ,  $z_{\omega^{2-}}$ ,  $z_{\omega^{2+}}$  with  $N = 2$ .

Given a center  $\omega$ , the same quantities  $\delta$  and  $\ell$  allow to define the two domain subsets

$$
X _ {\boldsymbol {\omega}} ^ {\square} = \left\{\boldsymbol {x} \in \mathbb {X} ^ {N} \left| \max  _ {i = 1, \dots , N} \left\{| x _ {i} - \omega_ {i} | \right\} \leq \ell \right. \right\} \quad X _ {\boldsymbol {\omega}} ^ {\square} = \left\{\boldsymbol {x} \in \mathbb {X} ^ {N} \left| \ell <   \max  _ {i = 1, \dots , N} \left\{| x _ {i} - \bar {\omega} _ {i} | \right\} \leq \ell + \delta \right. \right\}
$$

as well as  $X_{\omega} = X_{\bullet}^{\sqcup}\cup X_{\omega}^{\square}$

![](images/d12fdede0d6261ddbf05707e1e0749c709aca72da803e1b07ee0f2bf5c0dd533.jpg)

![](images/ef604727df8a4eeb096317bd4b62e51151d5906507d9b90a4add924028e2726d.jpg)

![](images/eb38c46bb08488fe885c2fd91eab0264e6b4548d69f5f83af488fef4733378ee.jpg)  
Figure 3: Three dimensional plots of the functions  $z_{\omega^{1 - }}$ ,  $z_{\omega^{1 + }}$ ,  $z_{\omega^{2 - }}$ ,  $z_{\omega^{2 + }}$  with  $N = 2$ .

![](images/e4c17334595eee4c6ad51b0d9de534bc1532c5cf5834b87faed79c2aecc2a935.jpg)

The approximation capabilities depend on the behaviour of the output of the subnetworks in the three disjoint domains  $X_{\omega}^{\bullet}, X_{\omega}^{\square}$ , and  $\mathbb{X}^N \setminus X_{\omega}$ .  
178 It is easy to see that if  $\pmb{x} \in \mathbb{X}^N \setminus X_\omega$  then  $z_{\omega^{j\pm}} = 0$  for  $j = 1, \dots, N$ .  
179 For  $\pmb{x} \in X_{\omega}^{\bullet}$  the following Lemma holds.  
180 Lemma 2. Given any choice of  $N + 1$  coefficients  $a$  and  $b_{j}$  for  $j = 1,\dots ,N$ , one may choose  $2N$  weights  $c^{j\pm}$  with  $j = 1,\ldots ,N$  such that

$$
Z _ {\boldsymbol {\omega}} (\boldsymbol {x}) \equiv \sum_ {j = 1} ^ {N} c ^ {j \pm} z _ {\boldsymbol {\omega} ^ {j \pm}} (\boldsymbol {x}) = a + \sum_ {j = 1} ^ {N} b _ {j} x _ {j} \tag {11}
$$

182 for any  $\pmb{x} \in X_{\omega}^{\bullet}$ , where  $Z_{\omega}(\pmb{x})$  remains implicitly defined.  
Proof of Lemma 2. Due to the definition of  $\omega^{j\pm}$  we have

$$
X _ {\omega} ^ {\bullet} = \left[ \omega_ {1} - \ell , \omega_ {1} + \ell \right] \times \dots \times \left[ \omega_ {N} - \ell , \omega_ {N} + \ell \right] = \left[ \omega_ {1} ^ {1 -}, \omega_ {1} ^ {1 +} \right] \times \dots \times \left[ \omega_ {N} ^ {N -}, \omega_ {N} ^ {N +} \right]
$$

184 Hence, if  $\pmb{x} \in X_{\omega}^{\bullet}$  we know that  $\omega_j^{j-} \leq x_j \leq \omega_j^{j+}$  for  $j = 1, \ldots, N$ .

Moreover, since by definition for any  $i,j = 1,\dots ,N$  and  $i\neq j$  we have  $\omega_{i}^{j + } - \omega_{i}^{j - } = 2\ell$  and  $\omega_{i}^{j - } + \omega_{i}^{j + } =$ $2\omega_{i}$  , then  $\left|x_i - \omega_i^{j\pm}\right|\leq \ell$  when  $i\neq j$  . Therefore, one can apply Lemma 1 and compute  $\Delta (\pmb {x})$  , for which all the terms in (6) but  $\left|x_j - \omega_j^{j\pm}\right|$  are non-positive, thus yielding  $z_{\omega^{\pm}}(\pmb {x}) = 1 - \left|x_j - \omega_j^{j\pm}\right| / (2\ell)$  
Without any loss of generality, translate  $X_{\omega}$  so that  $\omega = (\ell ,\dots ,\ell)$ . This implies  $\omega_{j}^{j - } = 0$  and  $\omega_{j}^{j + } = 2\ell$  for  $j = 1,\ldots ,N$ , thus yielding  $z_{\omega^{j - }}(\pmb {x}) = 1 - \frac{x_j}{2\ell}$  and  $z_{\omega^{j + }}(\pmb {x}) = \frac{x_j}{2\ell}$ . With this,

$$
\sum_ {j = 1} ^ {N} c ^ {j \pm} z _ {\boldsymbol {\omega} ^ {j \pm}} (\boldsymbol {x}) = \sum_ {j = 1} ^ {N} \left[ c ^ {j -} \left(1 - \frac {x _ {j}}{2 \ell}\right) + c ^ {j +} \frac {x _ {j}}{2 \ell} \right] = \sum_ {j = 1} ^ {N} c ^ {j -} + \sum_ {j = 1} ^ {N} \left(c ^ {j +} - c ^ {j -}\right) \frac {x _ {j}}{2 \ell}
$$

that can yield any affine function  $f(x) = a + \sum_{j=1}^{N} b_j x_j$  by setting, for  $j = 1, \ldots, N$ ,

$$
c ^ {j -} = \frac {a}{N} \quad \text {a n d} \quad c ^ {j +} = c ^ {j -} + 2 \ell b _ {j} \tag {12}
$$

191

192 Finally, what happens for  $\pmb{x} \in X_{\omega}^{\square}$  is described by the following Lemma.  
193 Lemma 3. If the  $2N$  weights  $c^{j\pm}$  with  $j = 1,\ldots ,N$  are set according to Lemma 2 so that  $Z_{\omega}(\pmb {x}) = a + \sum_{j = 1}^{N}b_{j}x_{j}$  for any  $x\in X_{\omega}^{\bullet}$ , for coefficients satisfying  $|a|,|b_j|\leq M$  for some  $M > 0$  and  $j = 1,\dots ,N$  
then  $\left|Z_{\omega}(\pmb{x})\right| \leq 3MN$  for any  $\pmb{x} \in X_{\omega}$  and thus for any  $\pmb{x} \in X_{\omega}^{\square}$ .

Proof of Lemma 3. From  $|a|, |b_j| \leq M$  and from (12) we get  $\left|c^{j-}\right| \leq M / N$  and  $\left|c^{j+}\right| \leq M / N + 2\ell M$ .  
197 Overall, since  $\ell \leq 1$  and  $N\geq 1$  we have  $\left|c^{j\pm}\right|\leq 3M$  Since  $0\leq z_{\omega^{j\pm}}\leq 1$  and  $Z_{\omega}(\pmb {x}) =$  198  $\sum_{j = 1}^{N}c^{j\pm}z_{\omega^{j\pm}}(\pmb {x})$  we finally get the thesis.  
The above characterization of the output of  $Z$ -subnetworks allows to prove their local approximation capabilities.  
201 Lemma 4. Given any function  $f \in \mathcal{C}^2(\mathbb{X}^N)$ , there are two constants  $P, Q > 0$  such that

$$
\begin{array}{l} E _ {\omega} \equiv \int_ {X _ {\omega}} | f (\boldsymbol {x}) - Z _ {\omega} (\boldsymbol {x}) | ^ {p} \mathrm {d} \boldsymbol {x} + \sum_ {j = 1} ^ {N} \int_ {X _ {\omega}} \left| \frac {\partial f}{\partial x _ {j}} (\boldsymbol {x}) - \frac {\partial Z _ {\omega}}{\partial x _ {j}} (\boldsymbol {x}) \right| ^ {p} \mathrm {d} \boldsymbol {x} \\ \leq \left(2 \ell + 2 \delta\right) ^ {N} \left\{P \ell^ {p} \left[ 1 - o (\delta / \ell) \right] + Q o (\delta / \ell) \right\} \\ \end{array}
$$

202 with  $o(\cdot) = 1 - 1 / (1 + \cdot)^N$

Proof of Lemma 4. Since  $f \in \mathcal{C}^2(\mathbb{X}^N)$  and  $\mathbb{X}^N$  is compact,  $M_0, M_1, M_2 \geq 0$  exists such that

$$
\left| f (\boldsymbol {x}) \right| \leq M _ {0}, \quad \left| \frac {\partial f}{\partial x _ {i}} (\boldsymbol {x}) \right| \leq M _ {1}, \quad \left| \frac {\partial^ {2} f}{\partial x _ {i} x _ {j}} (\boldsymbol {x}) \right| \leq M _ {2} \tag {13}
$$

204 for any  $\pmb {x}\in \mathbb{X}^M$  and  $i,j = 1,\ldots ,N$

Assuming  $\pmb{x} \in X_{\omega}^{\bullet}$ , and thus  $|x_i - \omega_i| \leq \ell$ , the above bounds can be used jointly with the Taylor expansions of  $f$  and its derivatives around  $\omega$

$$
f (\boldsymbol {x}) = f (\boldsymbol {\omega}) + \sum_ {i = 1} ^ {N} \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) (x _ {i} - \omega_ {i}) + \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} R _ {i, j} (\boldsymbol {x}) (x _ {i} - \omega_ {i}) (x _ {j} - \omega_ {j}) \tag {14}
$$

$$
\frac {\partial f}{\partial x _ {i}} (\boldsymbol {x}) = \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) + \sum_ {j = 1} ^ {N} S _ {i, j} (\boldsymbol {x}) \left(x _ {j} - \omega_ {j}\right) \quad i = 1, \dots , N \tag {15}
$$

207 to ensure that their error terms satisfy

$$
\left| \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} R _ {i, j} (\boldsymbol {x}) (x _ {i} - \omega_ {i}) (x _ {j} - \omega_ {j}) \right| \leq N ^ {2} \ell^ {2} \frac {1}{2} \max  _ {k, l = 1, \dots , N} \max  _ {\boldsymbol {\xi} \in \mathbb {X} ^ {N}} \left| \frac {\partial^ {2} f}{\partial x _ {k} x _ {l}} (\boldsymbol {\xi}) \right| \leq \frac {1}{2} M _ {2} N ^ {2} \ell^ {2} \tag {16}
$$

208 and

$$
\left| \sum_ {j = 1} ^ {N} S _ {i, j} (\boldsymbol {x}) (x _ {j} - \omega_ {j}) \right| \leq N ^ {2} \ell^ {2} \frac {1}{2} \max  _ {j = 1, \dots , N} \max  _ {\boldsymbol {\xi} \in \mathbb {X} ^ {N}} \left| \frac {\partial^ {2} f}{\partial x _ {i} x _ {j}} (\boldsymbol {\xi}) \right| \leq \frac {1}{2} M _ {2} N \ell \quad i = 1, \dots , N \tag {17}
$$

209 Again focusing on  $\pmb{x} \in X_{\omega}^{\bullet}$ , exploit Lemma 2 to set the weights  $c^{j\pm}$  yielding

$$
Z _ {\boldsymbol {\omega}} (\boldsymbol {x}) = f (\boldsymbol {\omega}) + \sum_ {i = 1} ^ {N} \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) (x _ {i} - \omega_ {i}) = \left[ f (\boldsymbol {\omega}) - \sum_ {i = 1} ^ {N} \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) \omega_ {i} \right] + \sum_ {i = 1} ^ {N} \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) x _ {i} \tag {18}
$$

which is also such that  $\frac{\partial Z_{\omega}}{\partial x_i}(\pmb{x}) = \frac{\partial f}{\partial x_u}(\pmb{\omega})$ .

Hence, we may program  $Z_{\omega}$  to reproduce the behaviour of  $f$  and its derivatives in  $X_{\omega}^{\bullet}$ , and the approximation errors can be derived exploiting (14) with (16) and (15) with (17) to obtain

$$
\left| Z _ {\omega} (\boldsymbol {x}) - f (\boldsymbol {x}) \right| \leq \frac {1}{2} M _ {2} N ^ {2} \ell^ {2}, \quad \left| \frac {\partial Z _ {\omega}}{\partial x _ {i}} (\boldsymbol {x}) - \frac {\partial f}{\partial x _ {i}} (\boldsymbol {x}) \right| \leq \frac {1}{2} M _ {2} N \ell \tag {19}
$$

To address the case  $\pmb{x} \in X_{\omega}^{\square}$ , we may apply Lemma 3. By matching (18) with (13) we get that  $|a| \leq M_0 + M_1N$  and  $|b_i| \leq M_1 \leq M_0 + M_1N$  for  $i = 1, \dots, N$ . Hence, if  $x \in X_{\omega}^{\square}$ , then if  $M_3 = M_0(1 + 3N) + 3M_1N^2$  we have

$$
\left| Z _ {\omega} (\boldsymbol {x}) - f (\boldsymbol {x}) \right| \leq M _ {3}, \quad \left| \frac {\partial Z _ {\omega}}{\partial x _ {i}} (\boldsymbol {x}) - \frac {\partial f}{\partial x _ {i}} (\boldsymbol {x}) \right| = \left| \frac {\partial f}{\partial x _ {i}} (\boldsymbol {\omega}) - \frac {\partial f}{\partial x _ {i}} (\boldsymbol {x}) \right| \leq 2 M _ {1} \tag {20}
$$

Since we have different error bounds in  $X_{\omega}^{\bullet}$  and  $X_{\bar{\omega}}^{\square}$ , we bound the overall error  $E_{\omega}$  by splitting

$$
\begin{array}{l} E _ {\omega} = \int_ {X _ {\omega} ^ {\text {口}}} | f (\boldsymbol {x}) - Z _ {\omega} (\boldsymbol {x}) | ^ {p} d \boldsymbol {x} + \sum_ {j = 1} ^ {N} \int_ {X _ {\omega} ^ {\text {口}}} \left| \frac {\partial f}{\partial x _ {j}} (\boldsymbol {x}) - \frac {\partial Z _ {\omega}}{\partial x _ {j}} (\boldsymbol {x}) \right| ^ {p} d \boldsymbol {x} + \\ \int_ {X _ {\omega} ^ {\square}} | f (\boldsymbol {x}) - Z _ {\omega} (\boldsymbol {x}) | ^ {p} \mathrm {d} \boldsymbol {x} + \sum_ {j = 1} ^ {N} \int_ {X _ {\omega} ^ {\square}} \left| \frac {\partial f}{\partial x _ {j}} (\boldsymbol {x}) - \frac {\partial Z _ {\omega}}{\partial x _ {j}} (\boldsymbol {x}) \right| ^ {p} \mathrm {d} \boldsymbol {x} + \\ \end{array}
$$

and apply (19) and (20) to bound each integrand. Adding the fact that the measure of  $X_{\omega}^{\square}$  is  $(2\ell)^{N}$ , while the measure of  $X_{\omega}^{\square}$  is  $(2\ell + 2\delta)^{N} - (2\ell)^{N}$  we obtain

$$
E _ {\omega} \leq \left[ \left(\frac {1}{2} M _ {2} N ^ {2} \ell^ {2}\right) ^ {p} + \left(\frac {1}{2} M _ {2} N \ell\right) ^ {p} \right] (2 \ell) ^ {N} + \left[ M _ {3} ^ {p} + (2 M _ {1}) ^ {p} \right] \left[ (2 \ell + 2 \delta) ^ {N} - (2 \ell) ^ {N} \right]
$$

from which we may set  $P = \left(\frac{1}{2} M_2N^2\right)^p + \left(\frac{1}{2} M_2N\right)^p$  and  $Q = M_3^p + (2M_1)^p$  to get the thesis.

We are now in the position of proving our second result.

Proof of Theorem 2. For  $n > 0$  integer define  $\delta$  and  $\ell$  such that  $\delta = \ell^2$  and  $2\ell + 2\delta = 1/n$ . Let also  $\Omega = \left\{\frac{1}{2n}, \frac{3}{2n}, \ldots, \frac{2n-1}{2n}\right\}^N$  so that  $\mathbb{X}^N$  is partitioned in  $n^N$  hyper-cubes  $X_\omega$  with centers  $\omega \in \Omega$  and side  $2\ell + 2\delta$ . The output of the whole network is  $Z(\boldsymbol{x}) = \sum_{\omega \in \Omega} Z_\omega(\boldsymbol{x})$ .

Since  $Z_{\omega}(x)$  is null for  $x \notin X_{\omega}$ , the error measure over  $\mathbb{X}^N$  can be decomposed into

$$
\| f - Z \| _ {1, p} ^ {p} = \sum_ {\boldsymbol {\omega} \in \Omega} \left\{\int_ {X _ {\boldsymbol {\omega}}} | f (\boldsymbol {x}) - Z _ {\boldsymbol {\omega}} (\boldsymbol {x}) | ^ {p} d \boldsymbol {x} + \sum_ {j = 1} ^ {N} \int_ {X _ {\boldsymbol {\omega}}} \left| \frac {\partial f}{\partial x _ {j}} (\boldsymbol {x}) - \frac {\partial Z _ {\boldsymbol {\omega}}}{\partial x _ {j}} (\boldsymbol {x}) \right| ^ {p} d \boldsymbol {x} \right\}
$$

Each of the terms in the last sum can be bounded using Lemma 4 in which we may also substitute  $2\ell + 2\delta = 1/n$  and  $\delta = \ell^2$  to yield

$$
\| f - Z \| _ {1, p} ^ {p} \leq \sum_ {\omega \in \Omega} \frac {1}{n ^ {N}} \left\{P \ell^ {p} [ 1 - o (\ell) ] + Q o (\ell) \right\} = P \ell^ {p} [ 1 - o (\ell) ] + Q o (\ell)
$$

Since when  $n\to \infty$  we have  $\ell \rightarrow 0$  and thus  $o(\ell)\rightarrow 0$  the thesis is proven.

# 7 Conclusions

We established that neural networks in which hidden MAC neurons are substituted with MAM neurons to obtain more aggressively prunable architectures are still universal approximators.

# References

[1] L. Prono, P. Bich, M. Mangia, F. Pareschi, R. Rovatti, and G. Setti, "A Multiply-And-Max/min Neuron Paradigm for Aggressively Prunable Deep Neural Networks," Apr. 2023, preprint available on TechRxiv.  
[2] P. Bich, L. Prono, M. Mangia, F. Pareschi, R. Rovatti, and G. Setti, "Aggressively prunable MAM²-based Deep Neural Oracle for ECG acquisition by Compressed Sensing," in 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Oct. 2022, pp. 163-167.

[3] G. Cybenko, "Approximation by superpositions of a sigmoidal function," Mathematics of Control, Signals and Systems, vol. 2, no. 4, pp. 303-314, Dec. 1989.  
[4] L.-X. Wang, "Fuzzy systems are universal approximators," in [1992 Proceedings] IEEE International Conference on Fuzzy Systems, 1992, pp. 1163-1170.  
[5] B. Kosko, “Fuzzy systems as universal approximators,” IEEE Transactions on Computers, vol. 43, no. 11, pp. 1329–1333, 1994.  
[6] J. Castro, “Fuzzy logic controllers are universal approximators,” IEEE Transactions on Systems, Man, and Cybernetics, vol. 25, no. 4, pp. 629–635, Apr. 1995.  
[7] R. Rovatti, “Fuzzy piecewise multilinear and piecewise linear systems as universal approximators in sobolev norms,” IEEE Transactions on Fuzzy Systems, vol. 6, no. 2, pp. 235–249, 1998.  
[8] K. Hornik, M. Stinchcombe, and H. White, "Multilayer feedforward networks are universal approximators," Neural Networks, vol. 2, no. 5, pp. 359-366, 1989.  
[9] A. Pinkus, “Approximation theory of the MLP model in neural networks,” Acta Numerica, vol. 8, pp. 143–195, Jan. 1999.  
[10] G. Gripenberg, “Approximation by neural networks with a bounded number of nodes at each level,” Journal of Approximation Theory, vol. 122, no. 2, pp. 260–266, Jun. 2003.  
[11] Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang, "The Expressive Power of Neural Networks: A View from the Width," in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.  
[12] B. Hanin and M. Sellke, “Approximating Continuous Functions by ReLU Nets of Minimal Width,” Mar. 2018.  
[13] V. Maiorov and A. Pinkus, "Lower bounds for approximation by MLP neural networks," Neurocomputing, vol. 25, no. 1, pp. 81-91, Apr. 1999.  
[14] N. J. Guliyev and V. E. Ismailov, "On the approximation by single hidden layer feedforward neural networks with fixed weights," Neural Networks, vol. 98, pp. 296-304, Feb. 2018.  
[15] Y. Cai, "Achieve the Minimum Width of Neural Networks for Universal Approximation," in The Eleventh International Conference on Learning Representations, Feb. 2023.  
[16] D.-X. Zhou, "Universality of deep convolutional neural networks," Applied and Computational Harmonic Analysis, vol. 48, no. 2, pp. 787-794, 2020.  
[17] O. A. Manita, M. A. Peletier, J. W. Portegies, J. Sanders, and A. Senen-Cerda, “Universal approximation in dropout neural networks,” J. Mach. Learn. Res., vol. 23, no. 1, jan 2022.  
[18] Y. Lu and J. Lu, “A universal approximation theorem of deep neural networks for expressing probability distributions,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 3094–3105.  
[19] S.-Q. Zhang and Z.-H. Zhou, “Theoretically provable spiking neural networks,” in Advances in Neural Information Processing Systems, vol. 35. Curran Associates, Inc., 2022, pp. 19345-19356.