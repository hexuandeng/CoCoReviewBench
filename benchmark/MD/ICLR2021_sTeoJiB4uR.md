# REDUCING THE COMPUTATIONAL COST OF DEEP GENERATIVE MODELS WITH BINARY NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep generative models provide a powerful set of tools to understand real-world data. But as these models improve, they increase in size and complexity, so their computational cost in memory and execution time grows. Using binary weights in neural networks is one method which has shown promise in reducing this cost. However, whether binary neural networks can be used in generative models is an open problem. In this work we show, for the first time, that we can successfully train generative models which utilize binary neural networks. This reduces the computational cost of the models massively. We develop a new class of binary weight normalization, and provide insights for architecture designs of these binarized generative models. We demonstrate that two state-of-the-art deep generative models, the ResNet VAE and Flow++ models, can be binarized effectively using these techniques. We train binary models that achieve loss values close to those of the regular models but are  $90\% - 94\%$  smaller in size, and also allow significant speed-ups in execution time. We make our code and models available<sup>1</sup>

# 1 INTRODUCTION

As machine learning models continue to grow in number of parameters, there is a corresponding effort to try and reduce the ever-increasing memory and computational requirements that these models incur. One method to make models more efficient is to use neural networks with weights and possibly activations restricted to be binary-valued (Courbariaux et al., 2015; Courbariaux et al., 2016; Rastegari et al., 2016). Binary weights and activations require significantly less memory, and also admit faster low-level implementations of key operations such as linear transformations than when using the usual floating-point precision.

Although the application of binary neural networks for classification is relatively well-studied, there has been no research that we are aware of that has examined whether binary neural networks can be used effectively in unsupervised learning problems. Indeed, many of the deep generative models that are popular for unsupervised learning do have high parameter counts and are computationally expensive (Vaswani et al., 2017; Maaløe et al., 2019; Ho et al., 2019a). These models would stand to benefit significantly from converting the weights and activations to binary values, which we call binarization for brevity.

In this work we focus on non-autoregressive models with explicit densities. One such class of density model is the variational autoencoder (VAE) (Kingma & Welling, 2014; Rezende et al., 2014), a latent variable model which has been used to model many high-dimensional data domains accurately. The state-of-the-art VAE models tend to have deep hierarchies of latent layers, and have demonstrated good performance relative to comparable modelling approaches (Ranganath et al., 2016; Kingma et al., 2016; Maaløe et al., 2019). Whilst this deep hierarchy makes the model powerful, the model size and compute requirements increase with the number of latent layers, making very deep models resource intensive.

Another class of density model which has shown promising results are flow-based generative models (Rezende & Mohamed, 2015; Dinh et al., 2017). These models perform a series of invertible

transformations to a simple density, with the transformed density approximating the data-generating distribution. Flow models which achieve state-of-the-art performance compose many transformations to give flexibility to the learned density (Kingma & Dhariwal, 2018; Ho et al., 2019a). Again the model computational cost increases as the number of transformations increases.

To examine how to binarize hierarchical VAEs and flow models successfully, we take two models which have demonstrated excellent modelling performance - the ResNet VAE (Kingma et al., 2016) and the Flow++ model (Ho et al., 2019a) - and implement the majority of each model with binary neural networks. Using binary weights and activations reduces the computational cost, but also decreases the representational capability of the model. Therefore our aim is to strike a balance between reducing the computational cost and maintaining good modelling performance. We show that it is possible to decrease the model size drastically, and allow for significant speed ups in run time, with only a minor impact on the achieved loss value. We make the following key contributions:

- We propose an efficient binary adaptation of weight normalization, a reparameterization technique often used in deep generative models to accelerate convergence. Binary weight normalization is the generative-modelling alternative to the usual batch normalization used in binary neural networks.  
- We show that we can binarize the majority of weights and activations in deep hierarchical VAE and flow models, without significantly hurting performance. We demonstrate the corresponding binary architecture designs for both the ResNet VAE and the Flow++ model.  
- We perform experiments on different levels of binarization, clearly demonstrating the trade-off between binarization and performance.

# 2 BACKGROUND

In this section we give background on the implementation and training of binary neural networks. We also describe the generative models that we implement with binary neural networks in detail.

# 2.1 BINARY NEURAL NETWORKS

In order to reduce the memory and computational requirements of neural networks, there has been recent research into how to effectively utilise networks which use binary-valued weights  $\mathbf{w}_{\mathbb{B}}$  and possibly also activations  $\alpha_{\mathbb{B}}$  rather than the usual real-valued $^2$  weights and activations (Courbariaux et al., 2015; Courbariaux et al., 2016; Rastegari et al., 2016). In this work, we use the convention of binary values being in  $\mathbb{B} := \{-1,1\}$ .

Motivation. The primary motivation for using binary neural networks is to decrease the memory and computational requirements of the model. Clearly binary weights require less memory to be stored:  $32 \times$  less than the usual 32-bit floating-point weights.

Binary neural networks also admit significant speed-ups. A reported  $2 \times$  speed-up can be achieved by a layer with binary weights and real-valued inputs (Rastegari et al., 2016). This can be made an additional  $29 \times$  faster if the inputs to the layer are also constrained to be binary (Rastegari et al., 2016). With both binary weights and inputs, linear operators such as convolutions can be implemented using the inexpensive XNOR and bit-count binary operations. A simple way to ensure binary inputs to a layer is to have a binary activation function before the layer (Courbariaux et al., 2016; Rastegari et al., 2016).

**Optimization.** Taking a trained model with real-valued weights and binarizing the weights has been shown to be lead to significant worsening of performance (Alizadeh et al., 2019). So instead the binary weights are optimized. It is common to not optimize the binary weights directly, but instead optimize a set of underlying real-valued weights  $\mathbf{w}_{\mathbb{R}}$  which can then be binarized in some fashion for inference. In this paper we will adopt the convention of binarizing the underlying weights using the sign function (see Equation 2). We also use the sign function as the activation function when we use binary activations (see Equation 5, where  $\alpha_{\mathbb{R}}$  are the real-valued pre-activations). We define the sign

function as:

$$
\operatorname {s i g n} (x) := \left\{ \begin{array}{l l} - 1, & \text {i f} x <   0 \\ 1, & \text {i f} x \geq 0 \end{array} \right. \tag {1}
$$

Since the derivative of the sign function is zero almost everywhere<sup>3</sup>, the gradients of the underlying weights  $\mathbf{w}_{\mathbb{R}}$  and through binary activations are zero almost everywhere. This makes gradient-based optimization challenging. To overcome this issue, the straight-through estimator (STE) (Bengio et al., 2013) can be used. When computing the gradient of the loss  $\mathcal{L}$ , the STE replaces the gradient of the sign function (or other discrete output functions) with an approximate surrogate. A straightforward and widely used surrogate gradient is the identity function, which we use to calculate the gradients of the real-valued weights  $w_{\mathbb{R}}$  (see Equation 3). It has been shown useful to cancel the gradients when their magnitude becomes too large (Courbariaux et al., 2015; Alizadeh et al., 2019). Therefore we use a clipped identity function for the gradients of the pre-activations (see Equation 6). This avoids saturating a binary activation. Lastly, the loss value only depends on the sign of the real-valued weights. Therefore, the values of the weights are generally clipped to be in  $[-1,1]$  after each gradient update (see Equation 4). This restricts the magnitude of the weights and thus makes it easier to flip the sign.

<table><tr><td colspan="3">Weights</td></tr><tr><td>Forward pass:</td><td>wB = sign(wR)</td><td>(2)</td></tr><tr><td>Backward pass:</td><td>∂L/∂wR := ∂L/∂wB</td><td>(3)</td></tr><tr><td>After update:</td><td>wR ← max(-1, min(1, wR))</td><td>(4)</td></tr></table>

<table><tr><td colspan="2">Activations</td></tr><tr><td>αB = sign(αR)</td><td>(5)</td></tr><tr><td>∂L/∂αR := ∂L/∂αB * 1|αR|≤1</td><td>(6)</td></tr><tr><td>-</td><td></td></tr></table>

# 2.2 DEEP GENERATIVE MODELS

Hierarchical VAEs. The variational autoencoder (Kingma & Welling, 2014; Rezende et al., 2014) is a latent variable model for observed data  $\mathbf{x}$  conditioned on unobserved latent variables  $\mathbf{z}$ . It consists of a generative model  $p_{\theta}(\mathbf{x}, \mathbf{z})$  and an inference model  $q_{\phi}(\mathbf{z}|\mathbf{x})$ . The generative model can be decomposed into the prior on the latent variables  $p_{\theta}(\mathbf{z})$  and the likelihood of our data given the latent variables  $p_{\theta}(\mathbf{x}|\mathbf{z})$ . The inference model is a variational approximation to the true posterior, since the true posterior is usually intractable in models of interest. Training is generally performed by maximization of the evidence lower bound (ELBO), a lower bound on the log-likelihood of the data:

$$
\log p _ {\boldsymbol {\theta}} (\mathbf {x}) \geq \mathbb {E} _ {q _ {\phi} (\mathbf {z} | \mathbf {x})} [ \log p _ {\boldsymbol {\theta}} (\mathbf {x}, \mathbf {z}) - \log q _ {\phi} (\mathbf {z} | \mathbf {x}) ] \tag {7}
$$

To give a more expressive model, the latent space can be structured into a hierarchy of latent variables  $\mathbf{z}_{1:L}$ . In the generative model each latent layer is conditioned on deeper latents  $p_{\theta}(\mathbf{z}_i|\mathbf{z}_{i+1:L})$ . A common problem with hierarchical VAEs is that the deeper latents can struggle to learn, often "collapsing" such that the layer posterior matches the prior:  $q_{\phi}(\mathbf{z}_i|\mathbf{z}_{i+1:L},\mathbf{x}) \approx p_{\theta}(\mathbf{z}_i|\mathbf{z}_{i+1:L})^4$ . One method to help prevent posterior collapse is to use skip connections between latent layers (Kingma et al., 2016; Maaloe et al., 2019), turning the layers into residual layers (He et al., 2016).

We focus on the ResNet VAE (RVAE) model (Kingma et al., 2016). In this model, both the generative and inference model structure their layers as residual layers. The ResNet VAE uses a bi-directional inference structure with both a bottom-up and top-down residual channel. This is a similar structure to the BIVA model (Maaloe et al., 2019), which has demonstrated state-of-the-art results for a latent variable model. We give a more detailed description of the model in Appendix B.

Flow models. Flow models consist of a parameterized invertible transformation,  $\mathbf{z} = \mathbf{f}_{\theta}(\mathbf{x})$ , and a known density  $p_{\mathbf{z}}(\mathbf{z})$  usually taken to be a unit normal distribution. Given observed data  $\mathbf{x}$  we obtain the objective for  $\theta$  by applying a change-of-variables to the log-likelihood:

$$
\log p _ {\boldsymbol {\theta}} (\mathbf {x}) = \log p _ {\mathbf {z}} \left(\mathbf {f} _ {\boldsymbol {\theta}} (\mathbf {x})\right) + \log \left| \det  \frac {d \mathbf {f} _ {\boldsymbol {\theta}}}{d \mathbf {x}} \right| \tag {8}
$$

For training to be possible, it is required that computation of the Jacobian determinant  $\operatorname*{det}(df_{\theta} / dx)$  is tractable. We therefore aim to specify of flow model  $\mathbf{f}_{\theta}$  which is sufficiently flexible to model the

data distribution well, whilst also being invertible and having a tractable Jacobian determinant. One common approach is to construct  $\mathbf{f}_{\theta}$  as a composition of many simpler functions:  $\mathbf{f}_{\theta} = \mathbf{f}_{1} \circ \mathbf{f}_{2} \circ \ldots \circ \mathbf{f}_{L}$ , with each  $\mathbf{f}_{i}$  invertible and with tractable Jacobian. So the objective becomes:

$$
\log p _ {\boldsymbol {\theta}} (\mathbf {x}) = \log p _ {\mathbf {z}} \left(\mathbf {f} _ {\boldsymbol {\theta}} (\mathbf {x})\right) + \sum_ {i = 1} ^ {L} \log \left| \det  \frac {d \mathbf {f} _ {i}}{d \mathbf {f} _ {i - 1}} \right| \tag {9}
$$

There are many approaches to construct the  $\mathbf{f}_i$  layers (Rezende & Mohamed, 2015; Dinh et al., 2017; Kingma & Dhariwal, 2018; Ho et al., 2019a). In this work we will focus on the Flow++ model (Ho et al., 2019a), which has state-of-the-art results for flow models. In the Flow++ model, the  $\mathbf{f}_i$  are coupling layers which partition the input into  $\mathbf{x}_1$  and  $\mathbf{x}_2$ , then transform only  $\mathbf{x}_2$ :

$$
\mathbf {f} _ {i} \left(\mathbf {x} _ {1}\right) = \mathbf {x} _ {1}, \quad \mathbf {f} _ {i} \left(\mathbf {x} _ {2}\right) = \sigma^ {- 1} \left(\operatorname {M i x L o g C D F} \left(\mathbf {x} _ {2}; \mathbf {t} \left(\mathbf {x} _ {1}\right)\right)\right) \cdot \exp (\mathbf {a} \left(\mathbf {x} _ {1}\right)) + \mathbf {b} \left(\mathbf {x} _ {1}\right) \tag {10}
$$

Where MixLogCDF is the CDF for a mixture of logistic distributions. This is an iteration on the affine coupling layer (Dinh et al., 2014; 2017). Note that keeping part of the input fixed ensures that the layer is invertible. To ensure that all dimensions are transformed in the composition, adjacent coupling layers will keep different parts of the input fixed, often using an alternating checkerboard or stripe pattern to choose the fixed dimensions (Dinh et al., 2017). The majority of parameters in this flow model come from the functions  $\mathbf{t}$ ,  $\mathbf{a}$  and  $\mathbf{b}$  in the coupling layer, and in  $\mathrm{Flow}++$  these are parameterized as stacks of convolutional residual layers. In this work we will focus on how to binarize these functions whilst maintaining good modelling performance. We give a more detailed description of the full flow model in Appendix C.

# 3 BINARIZING DEEP GENERATIVE MODELS

In this section we first introduce a technique to effectively binarize weight normalized layers, which are used extensively in deep generative model architectures. Afterwards, we elaborate on which components of the models we can binarize without significantly hurting performance.

# 3.1 NORMALIZATION

It is important to apply some kind of normalization after a binary layer. Binary weights are often large in magnitude relative to the usual real-valued weights, and can result in large outputs which can destabilize training. Previous binary neural network implementations have largely used batch normalization, which can be executed efficiently using a shift-based implementation (Courbariaux et al., 2016).

However, it is common in generative modelling to use weight normalization (Salimans & Kingma, 2016) instead of batch normalization. For example, it is used in the Flow++ (Ho et al., 2019a) and state-of-the-art hierarchical VAE models (Kingma et al., 2016; Maaloge et al., 2019). Weight normalization factors a vector of weights  $\mathbf{w}_{\mathbb{R}}$  into a vector of the same dimension  $\mathbf{v}_{\mathbb{R}}$  and a magnitude  $g$ , both of which are learned. The weight vector is then expressed as:

$$
\mathbf {w} _ {\mathbb {R}} = \mathbf {v} _ {\mathbb {R}} \cdot \frac {g}{\left| \left| \mathbf {v} _ {\mathbb {R}} \right| \right|} \tag {11}
$$

Where  $||\cdot ||$  denotes the Euclidean norm. This implies that the norm of  $\mathbf{w}_{\mathbb{R}}$  is  $g$ .

Now suppose we wish to binarize the parameters of a weight normalized layer. We are only able to binarize  $\mathbf{v}_{\mathbb{R}}$ , since binarizing the magnitude  $g$  and bias  $b$  could result in large outputs of the layer. However,  $g$  and  $b$  do not add significant compute or memory requirements, as they are applied elementwise and are much smaller than the binary weight vector.

Let  $\mathbf{v}_{\mathbb{B}} = \mathrm{sign}(\mathbf{v}_{\mathbb{R}})$  be a binarized weight vector of dimension  $n$ . Since every element of  $\mathbf{v}_{\mathbb{B}}$  is one of  $\pm 1$ , we know that  $||\mathbf{v}_{\mathbb{B}}|| = \sqrt{n}^5$ . We then have:

$$
\mathbf {w} _ {\mathbb {R}} = \mathbf {v} _ {\mathbb {B}} \cdot \frac {g}{\sqrt {n}} \tag {12}
$$

$$
\overline {{^ 5 | | \mathbf {v} _ {\mathbb {B}} | | = \sqrt {\sum_ {i} (v _ {\mathbb {B} , i}) ^ {2}} = \sqrt {\sum_ {i} 1} =}} \sqrt {n}
$$

We refer to this as binary weight normalization, or BWN. Importantly, this is faster to compute than the usual weight normalization (Equation 11), since we do not have to calculate the norm of  $\mathbf{v}_{\mathbb{B}}$ . The binary weight normalization requires only  $O(1)$  FLOPs to calculate the scaling for  $\mathbf{v}_{\mathbb{B}}$ , whereas the regular weight normalization requires  $O(n)$  FLOPs to calculate the scaling for  $\mathbf{v}_{\mathbb{R}}$ . For a model of millions of parameters, this can be a significant speed-up. Binary weight normalization also has a more straightforward backward pass, since we do not need to take gradients of the  $1 / ||\mathbf{v}||$  term.

Furthermore, convolutions  $\mathcal{F}$  and other linear transformations can be implemented using cheap binary operations when using binary weights,  $\mathbf{w}_{\mathbb{B}}$ , as discussed in Section 2.1<sup>6</sup>. However, after applying binary weight normalization, the weight vector is real-valued,  $\mathbf{w}_{\mathbb{R}}$ . Fortunately, since a convolution is a linear transformation, we can apply the normalization factor  $\alpha = g / \sqrt{n}$  either before or after applying the convolution to input  $\mathbf{x}$ .

$$
\mathcal {F} (\mathbf {x}, \mathbf {v} _ {\mathbb {B}} \cdot \alpha) = \mathcal {F} (\mathbf {x}, \mathbf {v} _ {\mathbb {B}}) \cdot \alpha \tag {13}
$$

So if we wish to utilize fast binary operations for the binary convolution layer, we need to apply binary weight normalization after the convolution. This means that the weights are binary for the convolution operation itself. This couples the convolution operation and the weight normalization, and we refer to the overall layer as a binary weight normalized convolution, or BWN convolution. Note that the above process applies equally well to other linear transformations. We initialize BWN layers in the same manner as regular weight normalization, but give a more thorough description of alternatives in Appendix D.

# 3.2 BINARIZING RESIDUAL LAYERS

We aim to binarize deep generative models, in which it is common to utilize residual layers extensively. Residual layers are functions with skip connections:

$$
\mathbf {g} _ {\text {r e s}} (\mathbf {x}) = \mathcal {T} (\mathbf {x}) + \mathbf {x} \tag {14}
$$

Indeed, the models we target in this work, the ResNet VAE and Flow++ models, have the majority of their parameters within residual layers. Therefore they are natural candidates for binarization, since binarizing them would result in a large decrease in the computational cost of the model. To binarize them we implement  $\mathcal{T}(\mathbf{x})$  in Equation 14 using binary weights and possibly activations.

The motivation for using residual layers is that they can be used to add more representative capability to a model without suffering from the degradation problem (He et al., 2016). That is, residual layers can easily learn the identity function by driving the weights to zero. So, if sufficient care is taken with initialization and optimization, adding residual layers to the model should not degrade performance, helping to precondition the problem.

Degradation of performance is of particular concern when using binary layers. Binary weights and activations are both less expressive than their real-valued counterparts, and more difficult to optimize. These disadvantages of binary layers are more pronounced for generative modelling than for classification. Generative models need to be very expressive, since we wish to model complex data such as images. Optimization can also be difficult, since the likelihood of a data point is highly sensitive to the distribution statistics output by the model, and can easily diverge. This provides an additional justification for binarizing only the residual layers of a generative model. By restricting binarization to the residual layers, it decreases the chance that using binary layers harms performance.

Crucially, if we were to use a residual binary layer without weight normalization, then the layer would not be able to learn the identity function, as the binary weights cannot be set to zero. This would remove the primary motivation to use binary residual layers. In contrast, using a binary weight normalized layer in the residual layer, the gain  $g$  and bias  $b$  can be set to zero to achieve the identity function. As such, we binarize the ResNet VAE and Flow++ models by implementing the residual layers using BWN layers.

![](images/16fae6611c915a3daaf880a4c0952db2203b5f1605b18be155d3334103d0964a.jpg)  
(a) RVAE  
(32-bit activations)  
(b) RVAE  
(1-bit activations)  
(c) Flow++  
(32-bit activations)  
(d) Flow++  
(1-bit activations)  
Figure 1: The residual blocks used in the binarized ResNet VAE and Flow++ models, using both binary and floating-point activations. The BWN Gate layer is a binary weight normalized  $1 \times 1$  convolution followed by a gated linear unit. We display the binary valued tensors with thick red arrows.

# 4 DEEP GENERATIVE MODELS WITH BINARY WEIGHTS

We now describe the binarized versions of the ResNet VAE and Flow++ model, using the techniques and considerations from Section 3.

ResNet VAE. As per Section 3.2, we wish to binarize the residual layers of the ResNet VAE. The residual layers are constructed as convolutional residual blocks, consisting of two  $3 \times 3$  convolutions and non-linearities, with a skip connection. This is shown in Figure 1(a)-(b). To binarize the block, we change the convolutions to BWN convolutions, as described in Section 3.1. We can either use real-valued activations or binary activations. Binary activations allow the network to be executed much faster, but are less expressive. We use the ELU function as the real-valued activation, and the sign function as the binary activation.

Flow++. As with the ResNet VAE, in the Flow++ model the residual layers are structured as stacks of convolutional residual blocks. To binarize the residual blocks, we change both the  $3 \times 3$  convolution and the gated  $1 \times 1$  convolution in the residual block to be BWN convolutions. The residual block design is shown in Figure 1(c)-(d). We have the option of using real-valued or binary activations.

# 5 EXPERIMENTS

We run experiments with the ResNet VAE and the Flow++ model, to demonstrate the effect of binarizing the models. We train and evaluate on the CIFAR and ImageNet  $(32 \times 32)$  datasets. For both models we use the Adam optimizer (Kingma & Ba, 2015), which has been demonstrated to be effective in training binary neural networks (Alizadeh et al., 2019).

For the ResNet VAE, we decrease the number of latent variables per latent layer and increase the width of the residual channels, as compared to the original implementation. We found that increasing the ResNet blocks in the first latent layer slightly increased modelling performance. Furthermore, we chose not to model the posterior using IAF layers Kingma et al. (2016), since we want to keep the model class as general as possible.

For the Flow++ model, we decrease the number of components in the mixture of logistics for each coupling layer and increase the width of the residual channels, as compared to the original implementation. For simplicity, we also remove the attention mechanism from the model, since the ablations the authors performed showed that this had only a small effect on the model performance.

Note that we do not use any techniques to try and boost the test performance of our models, such as importance sampling or using weighted averages of the model parameters. These are often used in generative modelling, but since we are trying to establish the relative performance of models with various degrees of binarization, we believe that these techniques are irrelevant.

Table 1: Results for binarized ResNet VAE and Flow++ model on CIFAR and ImageNet  $(32\times 32)$  test sets. Loss values are reported in bits per dimension. We give the percentage of the model parameters that are binary and the overall size of the model parameters. The weights and activations refer to those within the residual layers of the model, which are the targets for binarization.  

<table><tr><td></td><td colspan="2">Precision</td><td colspan="2">Modelling loss</td><td># Parameters</td><td>% Binary</td><td>Memory cost</td></tr><tr><td></td><td>Weights</td><td>Activations</td><td>CIFAR</td><td>ImageNet (32 × 32)</td><td></td><td></td><td></td></tr><tr><td rowspan="3">ResNet VAE</td><td>32-bit</td><td>32-bit</td><td>3.45</td><td>4.25</td><td>56M</td><td>0%</td><td>255 MB</td></tr><tr><td>1-bit</td><td>32-bit</td><td>3.60</td><td>4.47</td><td>56M</td><td>97.1%</td><td>13 MB</td></tr><tr><td>1-bit</td><td>1-bit</td><td>3.73</td><td>4.58</td><td>56M</td><td>97.1%</td><td>13 MB</td></tr><tr><td rowspan="2">increased width</td><td>1-bit</td><td>32-bit</td><td>3.56</td><td>-</td><td>96M</td><td>97.7%</td><td>20 MB</td></tr><tr><td>1-bit</td><td>1-bit</td><td>3.68</td><td>-</td><td>96M</td><td>97.7%</td><td>20 MB</td></tr><tr><td>no residual</td><td>N.A.</td><td>N.A.</td><td>3.78</td><td>-</td><td>1.6M</td><td>0%</td><td>6 MB</td></tr><tr><td rowspan="3">Flow++</td><td>32-bit</td><td>32-bit</td><td>3.21</td><td>4.05</td><td>34M</td><td>0%</td><td>129 MB</td></tr><tr><td>1-bit</td><td>32-bit</td><td>3.29</td><td>4.18</td><td>34M</td><td>90.1%</td><td>14 MB</td></tr><tr><td>1-bit</td><td>1-bit</td><td>3.43</td><td>4.30</td><td>34M</td><td>90.1%</td><td>14 MB</td></tr><tr><td>no residual</td><td>N.A.</td><td>N.A.</td><td>3.54</td><td>-</td><td>2.2M</td><td>0%</td><td>9 MB</td></tr></table>

# 5.1 DENSITY MODELLING

We display results in Table 1. We can see that the models with binary weights and real-valued activations perform only slightly worse than those with real-valued weights, for both the ResNet VAE and the Flow++ models. For the models with binary weights, we observe better performance when using real-valued activations than with the binary activations. These results are as expected given that binary values are by definition less expressive than real values. All models with binary weights perform better than a baseline model with the residual layers set to the identity, indicating that the binary layers do learn. We display samples from the binarized models in Appendix A.

Importantly, we see that the model size is significantly smaller when using binary weights -  $94\%$  smaller for the ResNet VAE and  $90\%$  smaller for the Flow++ model.

These results demonstrate the fundamental trade-off that using binary layers in generative models allows. By using binary weights the size of the model can be drastically decreased, but there is a slight degradation in modelling performance. The model can then be made much faster by using binary activations as well as weights, but this decreases performance further.

# 5.2 INCREASING THE RESIDUAL CHANNELS

Binary models are less expensive in terms of memory and compute. This raises the question of whether binary models could be made larger in parameter count than the model with real-valued weights, with the aim of trying to improve performance for a fixed computational budget. We examine this by increasing the number of channels in the residual layers (from 256 to 336) of the ResNet VAE. This increases the number of binary weights by approximately 40 million, but leaves the number of real-valued weights roughly constant<sup>7</sup>. The results are shown in Table 1 and Figure 3(c). We can see the increase the binary parameter count does have a noticeable improvement in performance. The model size increases from 13 MB to 20 MB, which is still an order of magnitude smaller than the model with real-valued weights (255 MB). It is an open question as to how much performance could be improved by increasing the size of the binary layers even further. The barrier to this approach currently is training, since we need to maintain and optimize a set of real-valued weights during training. These get prohibitively large as we increase the model size significantly.

# 5.3 ABLATIONS

We perform ablations to verify our hypothesis from Section 3.2 that we should only binarize the residual layers of the generative models. We attempt to binarize all layers in the ResNet VAE using BWN layers, using both binary and real-valued activations. The results are shown in Figure 3(d). As expected, the loss values attained are significantly worse than when binarizing only the residual layers.

![](images/a44f619e9f13887b7f2addb4ff4d0bbfb4ac37a7886161367352f5d66f1417b2.jpg)  
(a) ResNet VAE

![](images/6fadf54110b89ef838e3d0c3a227b8f82a3339bdebc68d5f75150ae5da7defef.jpg)  
(b)  $\mathbf{Flow} + +$

![](images/1df76009c16023b4912aeef4edaebc7127459a311f7dbfb42f58416379a58bf0.jpg)  
(c) ResNet VAE (increased channel width)

![](images/b308e8f9e8a829f2eea0ae9dc40ad5957dc4e3b13a73913efe01f1d35e06d62e.jpg)  
(d) ResNet VAE (ablations)  
Figure 2: Test loss values during training of the ResNet VAE and Flow++ models on the CIFAR dataset. Subfigures (a) and (b): models with binary weights and either binary or real-valued activations. Compared to the model with real-valued weights and activations, and a baseline with the residual layers set to the identity. Subfigures (c) and (d): the effect of increasing the width of the residual channels, and ablations.

# 6 DISCUSSION AND FUTURE WORK

We have demonstrated that is possible to drastically reduce model size and compute requirements for the ResNet VAE and Flow++ models, whilst maintaining good modelling performance. We chose these models because of their demonstrated modelling power, but the methods we used to binarize them are readily applicable to other hierarchical VAEs or flow models. We believe this is a useful result for any real-world application of these generative models. such as learned lossless compression (Townsend et al., 2019; Kingma et al., 2019; Townsend et al., 2020; Ho et al., 2019b; Hoogeboom et al., 2019), which could be made practical with these reduced memory and compute benefits.

A key technical challenge that needs to be overcome for binary neural networks as a whole is the availability of implementations of the fast binary linear operations such as binary convolutions. Proof-of-concept implementations of these binary kernels have been developed (Courbariaux et al., 2016; Rastegari et al., 2016; Pedersoli et al., 2018). However, there is no implementation that is readily usable as a substitute for the existing kernels in frameworks such as PyTorch and Tensorflow.

# 7 CONCLUSION

We have shown that it is possible to implement state-of-the-art deep generative models using binary neural networks. We proposed using a fast binary weight normalization procedure, and shown that it is necessary to binarize only the residual layers of the model to maintain modelling performance. We demonstrated this by binarizing two state-of-the-art models, the ResNet VAE and the Flow++ model, reducing the computational cost massively. We hope this insight into the possible trade-off between modelling performance and computational cost will stimulate further research into the efficiency of deep generative models.

# REFERENCES

M. Alizadeh, J. Fernández-Marqués, N. D. Lane, and Y. Gal. An empirical study of binary neural networks' optimisation. International Conference on Learning Representations (ICLR), 2019.  
J. Ba, J. Kiros, and G. Hinton. Layer normalization. arXiv preprint, arXiv:1607.06450, 2016.  
Y. Bengio, N. Léonard, and A. Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint, arXiv:1308.3432, 2013.  
M. Courbariaux, Y. Bengio, and J.P. David. Binaryconnect: Training deep neural networks with binary weights during propagations. Neural Information Processing Systems (NeurIPS), 2015.  
M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio. Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1. arXiv preprint, arXiv:1602.02830, 2016.  
Y. Dauphin, A. Fan, M. Auli, and D. Grangier. Language modeling with gated convolutional networks. International Conference on Machine Learning (ICML), 2017.  
L. Dinh, D. Krueger, and Y. Bengio. Nice: Non-linear independent components estimation. arXiv preprint, arXiv:1410.8516, 2014.  
L. Dinh, J. Sohl-Dickstein, and S. Bengio. Density estimation using real nvp. International Conference on Learning Representations (ICLR), 2017.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.  
J. Ho, X. Chen, A. Srinivas, Y. Duan, and P. Abbeel. Flow++: Improving flow-based generative models with variational dequantization and architecture design. International Conference on Machine Learning (ICML), 2019a.  
J. Ho, E. Lohn, and P. Abbeel. Compression with flows via local bits-back coding. *Neural Information Processing Systems (NeurIPS)*, 2019b.  
E. Hoogeboom, J. W. T. Peters, R. van den Berg, and M. Welling. Integer discrete flows and lossless compression. Neural Information Processing Systems (NeurIPS), 2019.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR), 2015.  
D. P. Kingma and M. Welling. Auto-Encoding Variational Bayes. International Conference on Learning Representations (ICLR), 2014.  
D. P. Kingma, T. Salimans, R. Jozefowicz, X. Chen, I. Sutskever, and M. Welling. Improved variational inference with inverse autoregressive flow. Neural Information Processing Systems (NeurIPS), 2016.  
F. H. Kingma, P. Abbeel, and J. Ho. Bit-Swap: recursive bits-back coding for lossless compression with hierarchical latent variables. International Conference on Machine Learning (ICML), 2019.  
P. Kingma, D. and P. Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Neural Information Processing Systems (NeurIPS), 2018.  
L. Maalège, M. Fraccaro, V. Lievin, and O. Winther. Biva: A very deep hierarchy of latent variables for generative modeling. Neural Information Processing Systems (NeurIPS), 2019.  
F. Pedersoli, G. Tzanetakis, and A. Tagliasacchi. *Espresso: Efficient forward propagation for bcnns*. International Conference on Learning Representations (ICLR), 2018.  
R. Ranganath, D. Tran, and D. M. Blei. Hierarchical variational models. International Conference on Machine Learning (ICML), 2016.  
M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi. Xnor-net: Imagenet classification using binary convolutional neural networks. European Conference on Computer Vision (ECCV), 2016.

D. J. Rezende and S. Mohamed. Variational inference with normalizing flows. International Conference on Machine Learning (ICML), 2015.  
D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic back-propagation and variational inference in deep latent gaussian models. International Conference on Machine Learning (ICML), 2014.  
T. Salimans and D. P. Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Neural Information Processing Systems (NeurIPS), 2016.  
J. Townsend, T. Bird, and D. Barber. Practical lossless compression with latent variables using bits back coding. International Conference on Learning Representations (ICLR), 2019.  
J. Townsend, T. Bird, J. Kunze, and D. Barber. Hilloc: Lossless image compression with hierarchical latent variable models. International Conference on Learning Representations (ICLR), 2020.  
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. Neural Information Processing Systems (NeurIPS), 2017.