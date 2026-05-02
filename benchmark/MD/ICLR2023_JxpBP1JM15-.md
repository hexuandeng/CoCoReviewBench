# SCALING FORWARD GRADIENT WITH LOCAL LOSSES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Forward gradient learning computes a noisy directional gradient and is a biologically plausible alternative to backprop for learning deep neural networks. The standard forward gradient algorithm suffers from the curse of dimensionality in the number of parameters. In this paper, we propose to scale forward gradient by adding a large number of local greedy loss functions. We consider block-wise, patch-wise, and channel group-wise local losses, and show that activity perturbation reduces variance compared to weight perturbation. Inspired by MLPMixer, we also propose a new architecture, LocalMixer, that is more suitable for local learning. We find local learning can work well with both supervised classification and self-supervised contrastive learning. Empirically, it can match backprop on MNIST and CIFAR-10 and significantly outperform backprop-free algorithms on ImageNet.

# 1 INTRODUCTION

Most deep neural networks today are trained using the backpropagation algorithm (a.k.a. backprop) (Werbos, 1974; LeCun, 1985; Rumelhart et al., 1986), which efficiently computes the gradients of the weight parameters by propagating the error signal backwards from the loss function to each layer. Although artificial neural networks were originally inspired by biological neurons, backprop has always been considered as "biologically implausible" as the brain does not form symmetric backward connections or perform synchronized computations. From an engineering perspective, backprop is incompatible with a massive level of model parallelism, and restricts potential hardware designs. These concerns call for a drastically different learning algorithm for deep networks.

In the past, there have been attempts to address the above weight transport problem by introducing random backward weights (Lillicrap et al., 2016; Nokland, 2016), but they have been found to scale poorly on larger datasets such as ImageNet (Bartunov et al., 2018). Addressing the issue of global synchronization, several papers showed that greedy local loss functions can be almost as good as end-to-end learning (Belilovsky et al., 2018; Löwe et al., 2019; Xiong et al., 2020). However, they still rely on backprop for learning a number of internal layers within each local module.

Approaches based on weight perturbation, on the other hand, directly send the loss signal back to the weight connections and hence do not require any backward weights. In the forward pass, the network adds a slight perturbation to the synaptic connections and the weight update is then multiplied by the negative change in the loss. Weight perturbation was previously proposed as a biologically plausible alternative to backprop (Xie and Seung, 1999; Seung, 2003; Fiete and Seung, 2006). Instead of directly perturbing the weights, it is also possible to use forward-mode automatic differentiation (AD) to compute a directional gradient of the final loss along the perturbation direction (Pearlmutter, 1994). Algorithms based on forward-mode AD have recently received renewed interest in the context of deep learning (Baydin et al., 2022; Silver et al., 2022). However, existing approaches suffer from the curse of dimensionality, and the variance of the estimated gradients is too high to effectively train large networks.

In this paper, we revisit activity perturbation (Fiete and Seung, 2006) as an alternative to weight perturbation and explore its applicability to large networks trained on challenging vision tasks. We prove that activity perturbation yields lower-variance gradient estimates than weight perturbation and provide a continuous-time rate-based interpretation of our algorithm. We directly address the scalability issue of forward gradient learning by designing an architecture with many local greedy loss functions, isolating the network into local modules and hence reducing the number of learnable parameters per loss. Unlike prior work that only adds local losses along the depth dimension, we found that having patch-wise and channel group-wise losses is also critical. Lastly, inspired by the

design of MLPMixer (Tolstikhin et al., 2021), we designed a network called LocalMixer, featuring a linear token mixing layer and grouped channels for better compatibility with local learning.

We evaluate our local greedy forward gradient algorithm on supervised and self-supervised image classification problems. On MNIST and CIFAR-10, our learning algorithm performs comparably with backprop, and on ImageNet, it performs significantly better than other biologically plausible alternatives using asymmetric forward and backward weights. Although we have not fully matched backprop on larger-scale problems, we believe that local loss design could be a critical ingredient for biologically plausible learning algorithms and the next generation of model-parallel computation.

# 2 RELATED WORK

Ever since the perceptron era, the design of learning algorithms for neural networks, especially algorithms that could be realized by biological brains, has been a central interest. Review papers by Whittington and Bogacz (2019); Lillicrap et al. (2020) have systematically summarized the progress of biologically plausible deep learning. Here, we discuss related work in the following subtopics.

Forward gradient and reinforcement learning. Our work leverages forward-mode automatic differentiation (AD), which was first proposed by Wengert (1964). Later it was used to learn recurrent neural networks (Williams and Zipser, 1989) and to compute Hessian vector products (Pearlmutter, 1994). Computing the true gradient using forward-mode AD requires the full Jacobian, which is often large and expensive to compute. Recently, Baydin et al. (2022) and Silver et al. (2022) proposed to update the weights based on the directional gradient along a random perturbation direction. They found that this approach is sufficient for small-scale problems. This general family of algorithms is also related to reinforcement learning (RL) and evolution strategies (ES), since in each case the network receives a global reward. RL and ES have a long history of application in neural networks (Whitley, 1993; Stanley and Miikkulainen, 2002; Salimans et al., 2017), and they are effective for certain continuous control and decision-making tasks. Clark et al. (2021) found global credit assignment can also work well in vector neural networks where weights are only present between vectorized groups of neurons.

Greedy local learning. There have been numerous attempts to use local greedy learning objectives for training deep neural networks. Greedy layerwise pretraining (Bengio et al., 2006; Hinton et al., 2006; Vincent et al., 2010) trains individual layers or modules one at a time to greedily optimize an objective. Local losses are typically applied to different layers or residual stages, using common supervised and self-supervised loss formulations (Belilovsky et al., 2018; Nkland and Eidnes, 2019; Löwe et al., 2019; Belilovsky et al., 2020). Xiong et al. (2020); Gomez et al. (2020) proposed to use overlapped losses to reduce the impact of greedy learning. Patel et al. (2022) proposed to split a network into neuron groups. Laskin et al. (2020) applied greedy local learning on model parallelism training, and Wang et al. (2021) proposed to add a local reconstruction loss for preserving information. However, most local learning approaches proposed in the last decade rely on backprop to compute the weight updates within a local module. One exception is the work of Nkland and Eidnes (2019), which avoided backprop by using layerwise objectives coupled with a similarity loss or a feedback alignment mechanism. Gated linear networks and their variants (Veness et al., 2017; 2021; Sezener et al., 2021) ask every neuron to make a prediction, and have shown interesting results on avoiding catastrophic forgetting. From a theoretical perspective, Baldi and Sadowski (2016) provided insights and proofs on why local learning can be worse than global learning.

Asymmetric feedback weights. Backprop relies on weight symmetry: the backward weights are the same as the forward weights. Past research has looked at whether this constraint is necessary. Lillicrap et al. (2016) proposed feedback alignment (FA) that uses random and fixed backward weights and found it can support error driven learning in neural networks. Direct FA (Nokland, 2016) uses a single backward layer to wire the loss function back to each layer. There have also been methods that aim to explicitly update backward weights. Recirculation (Hinton and McClelland, 1987) and target propagation (TP) (Bengio, 2014; Lee et al., 2015; Bartunov et al., 2018) use local reconstruction objective to learn separate forward and backward weights as approximate inverses of each other. Ladder networks (Rasmus et al., 2015) found local reconstruction objectives and asymmetric weights can help achieve strong semi-supervised learning performance. However, Bartunov et al. (2018) reported both FA and TP algorithms do not scale to larger problems such as ImageNet, where their error rates are over  $90\%$ . Liao et al. (2016); Xiao et al. (2019) proposed sign symmetry (SS) where each backward connection weight share the same sign as the forward counterpart. Akrout et al. (2019)

proposed variants of weight mirroring. Compared to these works, we circumvent the issue of weight symmetry by using only reward (and the change rate thereof), instead of backward weights.

Biologically plausible perturbation learning. Forward gradient is related to perturbation learning in the biology context. Traditionally, neural plasticity learning rules focus on deriving weight updates as a function of the input and output activity of a neuron (Hebb, 1949; Widrow and Hoff, 1960; Oja, 1982; Bienenstock et al., 1982; Abbott and Nelson, 2000). Weight perturbation learning (Jabri and Flower, 1992), on the other hand, is much more general as it permits any form of global reward (Schultz et al., 1997). It was developed in both rated-based and spiking-based formulations (Xie and Seung, 1999; Seung, 2003). Activity (or node) perturbation was also proposed in a spike-based formulation (Fiete and Seung, 2006), where it was interpreted as the perturbation of the conductance of neurons. Werfel et al. (2003) showed that backprop has a faster convergence rate than perturbation learning, and activity perturbation wins over weight perturbation by another factor. In our work, we show activity perturbation has lower gradient estimation variance compared to weight perturbation.

# 3 FORWARD GRADIENT LEARNING

In this section, we review and establish the technical background for our learning algorithm. We first review the technique of forward-mode automatic differentiation (AD). Second, we formulate two different types of perturbation in the weight space or activity space.

# 3.1 FORWARD-MODE AUTOMATIC DIFFERENTIATION (AD)

Let  $f: \mathbb{R}^m \mapsto \mathbb{R}^n$ . The Jacobian of  $f$ ,  $J_f$ , is a matrix of size  $n \times m$ . Forward-mode AD computes the matrix-vector product  $J_f\mathbf{v}$ , where  $\mathbf{v} \in \mathbb{R}^m$ . It is defined as the directional gradient along  $\mathbf{v}$  evaluated at  $\mathbf{x}$ :

$$
J _ {f} \mathbf {v} := \lim  _ {\delta \mapsto 0} \frac {f (\mathbf {x} + \delta \mathbf {v}) - f (\mathbf {x})}{\delta}. \tag {1}
$$

For comparison, backprop, also known as reverse-mode AD, computes the vector-Jacobian product  $\mathbf{v}J_{f}$ , where  $\mathbf{v} \in \mathbb{R}^{n}$ , which corresponds to the last term in the chain rule. In contrast to reverse-mode AD, forward-mode AD only requires one forward pass, which is augmented with the derivative information. To compute the Jacobian vector product of a node in a computation graph, first the input node will be augmented with  $\mathbf{v}$ , which is the vector to be multiplied. Then for other nodes, we send in a tuple of  $(\mathbf{x},\mathbf{x}^{\prime})$  as inputs and compute a tuple  $(\mathbf{y},\mathbf{y}^{\prime})$  as outputs, where  $\mathbf{x}^{\prime}$  and  $\mathbf{y}^{\prime}$  are the intermediate derivatives at node  $\mathbf{x}$  and node  $\mathbf{y}$ , i.e.  $\mathbf{y}^{\prime} = \frac{d\mathbf{y}}{d\mathbf{x}}\mathbf{x}^{\prime}$ , and  $\frac{d\mathbf{y}}{d\mathbf{x}}$  is the Jacobian between  $\mathbf{y}$  and  $\mathbf{x}$ . In the JAX library (Bradbury et al., 2018), forward-mode AD is implemented as jax.jvp.

# 3.2 WEIGHT-PERTURBED FORWARD GRADIENT

Weight perturbation to generate weight updates was originally explored in (Barto et al., 1983; Xie and Seung, 1999; Seung, 2003). Baydin et al. (2022) uses the technique of forward-mode AD to implement weight perturbation, which is better than finite differences in terms of numerical stability. Let  $w_{ij}$  be the weight connection between unit  $i$  and  $j$ , and  $f$  be the loss function. We can estimate the gradient by sampling a random matrix with iid elements  $v_{ij}$  drawn from a zero-mean unit-variance Gaussian distribution. The estimator is

$$
g _ {w} \left(w _ {i j}\right) = \left(\sum_ {i ^ {\prime} j ^ {\prime}} \nabla w _ {i ^ {\prime} j ^ {\prime}} v _ {i ^ {\prime} j ^ {\prime}}\right) v _ {i j}. \tag {2}
$$

Intuitively, this estimator samples a random perturbation direction  $v_{ij}$  and tests how it aligns with the true gradient  $\nabla w_{i'j'}$  by using forward-mode to perform the dot product, and then multiplies the scalar alignment with the perturbation direction again. Baydin et al. (2022) referred this form of gradient estimation using forward-mode AD as "forward gradient". To distinguish with another form of perturbation we detail later, we refer this to as "weight-perturbed forward gradient", or simply as "weight perturbation".

# 3.3 ACTIVITY-PERTURBED FORWARD GRADIENT

An alternative to perturbing the weights is to instead perturb the activities, which can reduce the number of perturbation dimensions per example. Activity perturbation was originally proposed in Fiete and Seung (2006) in continuous-time spiking neural networks. It is potentially biologically plausible, since it could correspond to perturbation of the conductance in each neuron. Here, we focus on a discrete-time rate-based formulation for simplicity. Let  $x_{i}$  denote the activity of the  $i$ -th presynaptic neuron and  $z_{j}$  denote that of the  $j$ -th post-synaptic neuron before the non-linear activation

Table 1: Comparing weight  $(g_w)$  and activity  $(g_a)$  perturbation.  $V =$  dimension-wise avg. gradient variance,  $S =$  dimension-wise avg. squared gradient norm;  $p =$  fan-in;  $q =$  fan-out;  $N =$  batch size.  

<table><tr><td></td><td>Unbiased?</td><td>Avg. Variance (shared)</td><td>Avg. Variance (independent)</td></tr><tr><td>gw(·)</td><td>Yes</td><td>pq+2N V + (pq + 1)S</td><td>pq+2N V + pq+1N S</td></tr><tr><td>ga(·)</td><td>Yes</td><td>q+2N V + (q + 1)S</td><td>q+2N V + q+1N S</td></tr></table>

function, and  $u_{j}$  be the perturbation of  $z_{j}$ . The activity-perturbed forward gradient estimator is

$$
g _ {a} \left(w _ {i j}\right) = x _ {i} \left(\sum_ {j ^ {\prime}} \nabla z _ {j ^ {\prime}} u _ {j ^ {\prime}}\right) u _ {j}, \tag {3}
$$

where the inner product between  $\nabla \mathbf{z}$  and  $\mathbf{u}$  is again computed by using forward-mode AD.

# 3.4 THEORETICAL PROPERTIES

In this section we aim to analyze the expectation and variance properties of forward gradient estimators. We focus our analysis on the gradient of one weight matrix  $\{w_{ij}\}$ , but the conclusion holds for a network with many weight matrices too.

Table 1 summarizes the theoretical results<sup>1</sup>. With a batch size of  $N$ , independent perturbation can achieve  $1 / N$  reduction of variance, whereas shared perturbation has a constant variance term dominated by the squared gradient norm. However, when performing independent weight perturbation, matrix multiplications cannot be batched because each example's activation vector is multiplied with a different weight matrix. By contrast, independent activity perturbation admits batched matrix multiplications. Moreover, activity perturbation enjoys a factor of fan-in ( $p$ ) times smaller variance compared to weight perturbation since the number of perturbed elements is the number of output units instead of the size of the whole weight matrix. The only drawback of activity perturbation is the memory required for storage of intermediate activations, in exchange for a factor of  $Np$  reduction in variance. However, for both activity and weight perturbation, the variance still grows with larger networks. In Section 4 we will further reduce the variance by introducing local loss functions.

# 3.5 CONTINUOUS-TIME RATE-BASED MODELS

Forward-mode AD can be viewed as computing the first-order time derivative in a continuous-time physical system. Suppose the tuples passed between nodes of the computation graph are  $(\mathbf{x},\dot{\mathbf{x}})$ , where  $\dot{\mathbf{x}}$  is the change in  $\mathbf{x}$  over time. The computation is then the same as forward-mode AD. For each node,  $\dot{\mathbf{y}} = \frac{dy}{dx}\dot{\mathbf{x}}$ , where  $\frac{dy}{dx}$  is the Jacobian between the output and the input. Note that in a physical system we don't have to explicitly perform the differentiation operation by running two forward passes. Instead the first-order derivative information is readily available in the analog signal, and we only need to plug the output signal into a differentiator circuit.

The activity-perturbed learning rule for a continuous time system is thus  $\dot{w}_{ij} = x_i\dot{y}_j\dot{r}$ , where  $x_{i}$  is the presynaptic activity, and  $\dot{y}_j$  is the rate of change in the post-synaptic activity, which is the perturbation direction for a small period of time, and  $\dot{r}$  is the rate of change of reward (or the negative loss). The reward controls whether learning is Hebbian or anti-Hebbian. Both Hinton et al. (2007) and Bengio et al. (2017) propose to use a product of presynaptic activity and the rate of change of postsynaptic activity. However, they did not consider using the rate of change of reward as a modulator and instead relied on another set of feedback weights to communicate the error signal through inputs. In contrast, we show that by broadcasting the rate of change of reward, we can actually bypass the weight transport problem.

# 3.6 ACTIVATION SPARSITY AND NORMALIZATION FUNCTIONS

In networks with ReLU activations, we can leverage ReLU sparsity to achieve further variance reduction, because the inactivated units will have zero gradient and therefore we should not perturb these units, and set the perturbation to be zero.

Normalization layers are often added in deep neural networks after the linear layer. To compute the correct gradient in activity perturbation, we also need to account for normalization in the weight update rule. Since there is no backward weight connections, one option is to simply apply backprop on normalization layers. However, we also found that it is also fine to ignore the gradient of normalization layer when using layer normalization.

![](images/eb603e2aed22573a7a50efb3f5b2aba0f3a79b5938195b4c710286598d158217.jpg)  
Figure 1: A local mixer network consists of several mixer residual blocks.

![](images/1e705e5b7a7ade95afd3145f4757bf9111481bc4f44f7ab29ccef4b2a57fe9d8.jpg)  
Figure 2: A local mixer residual block with local losses. Token mixing consists of a linear layer and channels are grouped in the channel mixing layers. Layer norm is applied before and after every linear layer. LN=Layer Norm; FC=Fully Connected layer; A=Activation function (ReLU); T=Transpose.

# 4 SCALING WITH LOCAL LOSSES

As we have explained in the previous section, perturbation learning can suffer from a curse of dimensionality: the variance grows with the number of perturbation dimensions, and in deep networks there are often millions of parameters changing at the same time. One way to limit the number of learnable dimensions is to divide the network into submodules, each with a separate loss function. In this section, we will explore several ways to increase the number of local losses to tame the variance.

1) Blockwise loss. First, we will divide the network into modules in depth. Each module consists of several layers. At the end of each module, we compute a loss function, and that loss is used to update the parameters in that module. This approach is equivalent of adding a "stop gradient" operator in between modules. Such local greedy losses were previously explored in Belilovsky et al. (2018) and Löwe et al. (2019).  
2) Patchwise loss. Sensory input signals such as images have spatial dimensions. We will apply a separate loss patchwise along these spatial dimensions. In the Vision Transformer architecture (Vaswani et al., 2017; Dosovitskiy et al., 2021), each spatial token represents a patch in the image. In modern deep networks, parameters in each spatial location are often shared to improve data efficiency and reduce memory bandwidth utilization. Although naive weight sharing is not biologically plausible, we still consider shared weights in this work. It may be possible to mimic the effect of weight sharing by adding knowledge distillation (Hinton et al., 2015) losses in between patches.  
3) Groupwise loss. Lastly, we turn to the channel dimension. To create multiple losses, we split the channels into a number of groups, and each group is attached to a loss function (Patel et al., 2022). To prevent groups from communicating between each other, channels are only connected to other channels within the same group. A grouped linear layer is computed as  $z_{g,j} = \sum_{i} w_{g,i,j} x_{g,i}$ , for individual group  $g$ . Whereas previous work used channel groups to improve computational efficiency (Krizhevsky et al., 2012; Ioannou et al., 2017; Xie et al., 2017), in our work, adding groups contributes to the total number of losses and thus reduces variance.

Feature aggregators. Naively applying losses separately to the spatial and channel dimensions leads to suboptimal performances, since each dimension contains only local information. For losses of standard tasks such as classification, the model needs a global view of the inputs to make a decision. Standard architectures obtain this global view by performing global average pooling layer before the final classification layer. We therefore explore strategies for aggregating information from other groups and spatial patches before the local loss function.

We would prefer to perform aggregation without reducing the total number of dimensions. We thus propose a replicated design for feature aggregation, shown in Figure 3. First, channel groups are copied and communicated to one another, but every group except the active group itself is masked

![](images/d7639f00435a41b81d70485769e9cbb87940d8afa3d431a0c53030e89d1261eb.jpg)  
Figure 3: Feature aggregator designs. A) In the conventional design, average pooling is performed to aggregate features from different spatial locations. B) We propose the replicated design, features are first concatenated across groups and then averaged across spatial locations. We create copies of the same feature with different stop gradient masks so that we obtain more local losses instead of a global one. The stop gradient mask makes sure that perturbation in one spatial group corresponds to its loss function. The numerical value of the loss function is the same as the conventional design.

with stop gradient so that other groups do not affect the forward gradient computation:

$$
\mathbf {x} _ {p, g} = \left[ \operatorname {S t o p G r a d} \left(x _ {p, 1} \dots x _ {p, g - 1}\right), x _ {p, g}, \operatorname {S t o p G r a d} \left(x _ {p, g + 1}, \dots , x _ {p, G}\right) \right], \tag {4}
$$

where  $p$  and  $g$  index the patches and groups respectively. Similarly, each spatial location is also copied, communicated, and masked, and then averaged locally:

$$
\overline {{\mathbf {x}}} _ {p, g} = \frac {1}{P} \left(\mathbf {x} _ {p, g} + \sum_ {p ^ {\prime} \neq p} \operatorname {S t o p G r a d} \left(\mathbf {x} _ {p ^ {\prime}, g}\right)\right). \tag {5}
$$

The output of feature aggregation is the same as that of the conventional global average pooling layer. The difference is that here the loss is replicated and different patch groups are activated in each loss.

Learning objectives. We consider the supervised classification loss and the contrastive InfoNCE loss (van den Oord et al., 2018; Chen et al., 2020), which are the two most commonly used losses in image representation learning. For supervised classification, we attach a shared linear layer on top of the aggregated features for a cross entropy loss:  $L_{p,g}^{s} = -\sum_{k}t_{k}\log \mathrm{softmax}(W\overline{\mathbf{x}}_{p,g})_{k}$ . The loss is of the same value across each group and patch location.

For contrastive learning, the linear layer becomes a linear feature projector. Suppose  $\mathbf{x}_n^{(1)}$  and  $\mathbf{x}_n^{(2)}$  are the two different views of the  $n$ -th example, the InfoNCE loss for contrastive learning is:

$$
L _ {p, g} ^ {c} = - \sum_ {n} \log \frac {\left(W \overline {{\mathbf {x}}} _ {n , p , g} ^ {(1)}\right) ^ {\top} \operatorname {S t o p G r a d} \left(W \overline {{\mathbf {x}}} _ {n} ^ {(2)}\right)}{\sum_ {m} \left(W \overline {{\mathbf {x}}} _ {n , p , g} ^ {(1)}\right) ^ {\top} \operatorname {S t o p G r a d} \left(W \overline {{\mathbf {x}}} _ {m} ^ {(2)}\right)}. \tag {6}
$$

Note that we add a stop gradient operator on the second view. It is usually unnecessary to add this stop gradient in the InfoNCE loss; however, we found that perturbation-based methods require a stop gradient and otherwise the loss will not go down. This is likely because we share the perturbations on both views, and having the same perturbation will increase the dot product between the two views but is not desired from a representation learning perspective. Figure 4 shows a comparison of the loss curves. Non-shared perturbations also work but are worse than stop gradient.

# 5 IMPLEMENTATION

![](images/4539d8b0f101befd0748a48b6d55420839d9760fe3be42d2dc93d6e2e498c65b.jpg)  
Figure 4: Importance of StopGradient in the InfoNCE loss, using M/8 on CIFAR-10 with 256 channels 1 group.

Network architecture. The LocalMixer network takes inspiration from MLPMixer (Tolstikhin et al., 2021), which consists of fully connected networks and residual blocks. We leverage the fully connected networks so that each spatial patch performs computations without interfering with other patches, which is more compatible with our local learning objective. An image is divided into non-overlapping patches (i.e. tokens), and each block consists of token and channel mixing layers. Figure 1 shows the high level architecture, and Figure 2 shows the detailed diagram for one residual block. We add a linear projector layer to attach a loss function at

Table 2: Local Mixer Architecture Details  

<table><tr><td>Type</td><td>Blocks</td><td>Patches</td><td>Channels</td><td>Groups</td><td>Params</td><td>Dataset</td></tr><tr><td>Local Mixer S/1/1</td><td>1</td><td>1×1</td><td>256</td><td>1</td><td>272K</td><td>MNIST</td></tr><tr><td>Local Mixer M/1/16</td><td>1</td><td>1×1</td><td>512</td><td>16</td><td>429K</td><td>MNIST</td></tr><tr><td>Local Mixer M/8/16</td><td>4</td><td>8×8</td><td>512</td><td>16</td><td>919K</td><td>CIFAR-10</td></tr><tr><td>Local Mixer L/8/64</td><td>4</td><td>8×8</td><td>2048</td><td>64</td><td>13.1M</td><td>CIFAR-10</td></tr><tr><td>Local Mixer L/32/64</td><td>4</td><td>32×32</td><td>2048</td><td>64</td><td>17.3M</td><td>ImageNet</td></tr></table>

the end of each block. For token mixing layers, we use one linear fully connected layer instead of an MLP, since we would like to make each block as shallow as possible. Before the last channel mixing layer, features are reshaped into a number of groups, and the last layer is fully connected within each feature group. Table 2 shows architectural details for the different sizes of models we investigate.

Normalization. There are many ways of performing normalization within a neural network across different tensor dimensions (Krizhevsky et al., 2012; Ioffe and Szegedy, 2015; Ba et al., 2016; Ren et al., 2017; Wu and He, 2018). We opted for a local variant of layer normalization that normalizes within each local spatial patch of features (Ren et al., 2017). For grouped linear layers, each group is normalized separately (Wu and He, 2018). Empirically, we found such local normalization performs better on contrastive learning experiments and about the same as layer normalization on supervised experiments. Local normalization is also more biologically plausible as it does not perform global communication. Conventionally, normalization layers are placed after linear layers. In MLPlexer (Tolstikhin et al., 2021), layer normalization is placed at the beginning of each residual block. We found it is the best to place normalization before and after each linear layer, as shown in Figure 2. Empirically this design choice does not make much difference for backprop, but it allows forward gradient learning to learn much faster and achieve lower training errors.

Efficient implementation of replicated losses. Due to the design of feature aggregation and replicated losses, a naive implementation of groups can be very inefficient in terms of both memory consumption and compute. However, each spatial group actually computes the same aggregated feature and loss function. This means that it is possible to share most of the computation across loss functions when performing both backprop and forward gradient. We implemented our custom JAX JVP/VJP functions (Bradbury et al., 2018) and observed significant memory savings and compute speed-ups for replicated losses, which would otherwise not be feasible to run on modern hardware. The results are reported in Figure 5.

![](images/8386c5013924de6b9d6bd7875b9bb5f3cc85f6b818c45eb52c07207b26961d6c.jpg)  
Figure 5: Memory and compute usage of naive and fused implementation of replicated losses.

![](images/0f7a93a5536be947a84f269db1d71b19daab10a007db3acceeb016ec9489fb2a.jpg)

# 6 EXPERIMENTS

We compare our proposed algorithm to a set of alternatives: Backprop, Feedback Alignment and other global variants of Forward Gradient. Backprop is a biologically implausible oracle, since it computes true gradients whereas we compute noisy gradients. Feedback alignment computes approximate gradients by using a set of random backward weights. We explain each method below.

1) Backprop (BP). We include the standard backprop algorithm as well as its local variants. Local Backprop (L-BP) adds local losses as proposed, but still permits gradient to flow in an end-to-end fashion. Local Greedy Backprop (LG-BP) in addition adds stop gradient operators in between blocks. This is to provide a comparison to our methods by computing true local gradients. LG-BP is similar in spirit to recent local learning algorithms (Belilovsky et al., 2018; Löwe et al., 2019).  
2) Feedback Alignment (FA). The standard FA algorithm (Lillicrap et al., 2016) adds a set of random and fixed backward weights. We assume that the gradients to normalization layers and activation functions are known since they do not have weight connections. Local Feedback Alignment (L-FA) adds local losses as proposed, but still permits error signals to flow back. Local Greedy Feedback Alignment (LG-FA) adds a stop gradient to prevent error signals from flowing back, similar to the backprop-free algorithm in Nøkland and Eidnes (2019).

Table 3: Supervised learning for image classification  

<table><tr><td>Dataset Network Metric</td><td>MNIST S/1/1 Test / Train Err. (%)</td><td>MNIST M/1/16 Test / Train Err. (%)</td><td>CIFAR-10 M/8/16 Test / Train Err. (%)</td><td>ImageNet L/32/64 Test / Train Err. (%)</td></tr><tr><td>BP</td><td>2.66 / 0.00</td><td>2.41 / 0.00</td><td>33.62 / 0.00</td><td>36.82 / 14.69</td></tr><tr><td>L-BP</td><td>2.38 / 0.00</td><td>2.16 / 0.00</td><td>30.75 / 0.00</td><td>42.38 / 22.80</td></tr><tr><td>LG-BP</td><td>2.43 / 0.00</td><td>2.81 / 0.00</td><td>33.84 / 0.05</td><td>54.37 / 39.66</td></tr><tr><td colspan="5">BP-free algorithms</td></tr><tr><td>FA</td><td>2.82 / 0.00</td><td>2.90 / 0.00</td><td>39.94 / 28.44</td><td>94.55 / 94.13</td></tr><tr><td>L-FA</td><td>3.21 / 0.00</td><td>2.90 / 0.00</td><td>39.74 / 28.98</td><td>87.20 / 85.69</td></tr><tr><td>LG-FA</td><td>3.11 / 0.00</td><td>2.50 / 0.00</td><td>39.73 / 32.32</td><td>85.45 / 82.83</td></tr><tr><td>DFA</td><td>3.31 / 0.00</td><td>3.17 / 0.00</td><td>38.80 / 33.69</td><td>91.17 / 90.28</td></tr><tr><td>FG-W</td><td>9.25 / 8.93</td><td>8.56 / 8.64</td><td>55.95 / 54.28</td><td>97.71 / 97.58</td></tr><tr><td>FG-A</td><td>3.24 / 1.53</td><td>3.76 / 1.75</td><td>59.72 / 41.29</td><td>98.83 / 98.80</td></tr><tr><td>LG-FG-W</td><td>9.25 / 8.93</td><td>5.66 / 4.59</td><td>52.70 / 51.71</td><td>97.39 / 97.29</td></tr><tr><td>LG-FG-A</td><td>3.24 / 1.53</td><td>2.55 / 0.00</td><td>30.68 / 19.39</td><td>58.37 / 44.86</td></tr></table>

Table 4: Self-supervised contrastive learning with linear readout  

<table><tr><td>Dataset Network Metric</td><td>CIFAR-10 M/8/16 Test / Train Err. (%)</td><td>CIFAR-10 L/8/64 Test / Train Err. (%)</td><td>ImageNet L/32/64 Test / Train Err. (%)</td></tr><tr><td>BP</td><td>24.11 / 21.08</td><td>17.53 / 13.35</td><td>55.66 / 49.79</td></tr><tr><td>L-BP</td><td>24.69 / 21.80</td><td>19.13 / 13.60</td><td>59.11 / 52.50</td></tr><tr><td>LG-BP</td><td>29.63 / 25.60</td><td>23.62 / 16.80</td><td>68.36 / 62.53</td></tr><tr><td colspan="4">BP-free algorithms</td></tr><tr><td>FA</td><td>45.87 / 44.06</td><td>67.93 / 65.32</td><td>82.86 / 80.21</td></tr><tr><td>L-FA</td><td>37.73 / 36.13</td><td>31.05 / 26.97</td><td>83.18 / 79.80</td></tr><tr><td>LG-FA</td><td>36.72 / 34.06</td><td>30.49 / 25.56</td><td>82.57 / 79.53</td></tr><tr><td>DFA</td><td>46.09 / 42.76</td><td>39.26 / 37.17</td><td>93.51 / 92.51</td></tr><tr><td>FG-W</td><td>53.37 / 51.56</td><td>50.45 / 45.64</td><td>91.94 / 89.69</td></tr><tr><td>FG-A</td><td>54.59 / 52.96</td><td>56.63 / 56.09</td><td>97.83 / 97.79</td></tr><tr><td>LG-FG-W</td><td>52.66 / 50.23</td><td>52.27 / 48.67</td><td>91.36 / 88.81</td></tr><tr><td>LG-FG-A</td><td>32.88 / 29.73</td><td>26.81 / 23.90</td><td>73.24 / 66.89</td></tr></table>

3) Forward Gradient (FG). This family of methods comprises our proposed algorithm and related approaches. Weight-perturbed forward gradient (FG-W) was proposed by Baydin et al. (2022). In this paper, we propose the activity perturbation variant (FG-A). We further add local objective functions, producing LG-FG-W and LG-FG-A, which stand for Local Greedy Forward Gradient Weight/Activity-Perturbed. For local perturbation to work, we have to add a stop gradient in between blocks so each perturbation has a single corresponding loss. We expect LG-FG-A to achieve the best performance among other variants because it can leverage the variance reduction benefit from both activity perturbation and local losses.

Datasets. We use standard image classification datasets to benchmark the learning algorithms. MNIST (LeCun et al., 1999) contains  $70,00028 \times 28$  handwritten digit images of class 0-9. CIFAR-10 (Krizhevsky et al., 2009) contains  $60,00032 \times 32$  natural images of 10 semantic classes. ImageNet (Deng et al., 2009) contains 1.3 million natural images of 1000 classes, which we resized to  $224 \times 224$ . For CIFAR-10 and ImageNet, we applied both supervised learning and contrastive learning. For MNIST, we applied supervised learning only. We designed different configurations of the LocalMixer architecture for each dataset, listed in Table 2.

Data augmentation. For MNIST and CIFAR-10 supervised experiments, we do not apply data augmentation. Data augmentation on ImageNet follows the open source implementation of (Grill et al., 2020). Because forward gradient suffers from variance, we apply weaker augmentations for contrastive learning experiments, increasing the area lower bound for random crops from 0.08 to 0.3-0.5. We find that this change has relatively little effect on the performance of backprop.

![](images/1c48178175259e960773325c159ccac56846d333199f0a5cad0970086c9ea7f6.jpg)  
(a) CIFAR-10 Supervised M/8

![](images/b01bdc37678e89ba6bc005a7283f2399cb5428fbf8ffb5786471caf39840f478.jpg)  
(b) CIFAR-10 Contrastive M/8

![](images/39d3531da5f107f91a4006f5a05a69b5016aeeb7d54310df6384a4a2ea5ff376.jpg)  
(c) ImageNet Supervised L/32

![](images/ab86a989cf0a4b0c8fc96f5bd6f2823763a32dc7706aff97b6c9d09bf3ac22c0.jpg)  
Figure 6: Effect of adding local losses at different locations on the performance of forward gradient  
(a) Supervised Test  
Figure 7: Error rate of M/8/* during CIFAR-10 training using different number of groups.

![](images/d531f5a751cefb4759bca542da9b8d881853eb6e94230ed186bc24106191d198.jpg)  
(b) Supervised Train

![](images/1edefce3155ce98bff93f287d5a38dc2d00933f29a1e5b78c8f2a7f2c87b00f6.jpg)  
(c) Contrastive Test

![](images/f8fef2e98f6fd8c3a7971342c1735c4bd997b9a043d5bba37cf12d5d93394ac8.jpg)  
(d) Contrastive Train

Main results. Our main results are shown in Table 3 and Table 4. In supervised experiments, there is almost no cost of introducing local greedy losses, and our local forward gradient method can match the test error of backprop on MNIST and CIFAR. Note that LG-FG-A fails to overfit the training set to  $0\%$  error when trained without data augmentation. This suggests that variance could still be an issue. For CIFAR-10 contrastive learning, our method obtains an error rate approaching that obtained by backprop (26.81% vs. 17.53%), and most of the gap is due to greedy learning vs. gradient estimation (6.09% vs. 3.19%). On ImageNet, we achieve reasonable performance compared to backprop (58.37% vs. 36.82% for supervised and 73.24% vs. 55.66% for contrastive). However, we find that the error due to greediness grows as the problem gets more complex and requires more layers to cooperate. We significantly outperform the FA family on ImageNet (by 25% for supervised and 10% for contrastive). Interestingly, local greedy FA is also performs better than global feedback alignment, which suggests that the benefit of local learning transfers to other types of gradient approximation. TP-based methods were evaluated in Bartunov et al. (2018) and were found to be worse than FA on ImageNet. In sum, although there is still some noticeable gap between our method and backprop, we have made a large stride forward compared to backprop-free algorithms. More results are included in the Appendix 11.

Effect of local losses. In Figure 6 we ablate the benefit of placing local losses at different locations: blockwise, patchwise and groupwise. A combination of all three is the strongest. Global perturbation learning fails to learn as the accuracy is similar to initializing with random weights.

Effect of groups. In Figure 7 we investigate the effect of different number of groups by showing the training curves. Adding more groups bring significant improvement to local perturbation learning in terms of lowering both training and test errors, but the effect vanishes around 8 channels / group.

# 7 CONCLUSION

It is often believed that perturbation-based learning cannot scale to large and deep networks. We show that this is to some extent true because the gradient estimation variance grows with the number of hidden dimensions for activity perturbation, and is even worse for shared weight perturbation. But more optimistically, we show that a huge number of local greedy losses can help forward gradient learning scale much better. We explored blockwise, patchwise, and groupwise local losses, and a combination of all three, with a total of a quarter of a million losses in one of the larger networks, performs the best. Local activity-perturbed forward gradient performs better than previous backprop-free algorithms on larger networks. The idea of local losses opens up opportunities for different loss designs and sheds light on the search for biologically plausible learning algorithms in the brain and alternative computing devices.

# REFERENCES

Larry F Abbott and Sacha B Nelson. Synaptic plasticity: taming the beast. Nature neuroscience, 3 (11):1178-1183, 2000.  
Mohamed Akrout, Collin Wilson, Peter C. Humphreys, Timothy P. Lillicrap, and Douglas B. Tweed. Deep learning without weight transport. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 974-982, 2019.  
Lei Jimmy Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. CoRR, abs/1607.06450, 2016.  
Pierre Baldi and Peter J. Sadowski. A theory of local learning, the learning channel, and the optimality of backpropagation. *Neural Networks*, 83:51-74, 2016.  
Andrew G Barto, Richard S Sutton, and Charles W Anderson. Neuronlike adaptive elements that can solve difficult learning control problems. IEEE transactions on systems, man, and cybernetics, (5): 834-846, 1983.  
Sergey Bartunov, Adam Santoro, Blake A. Richards, Luke Marris, Geoffrey E. Hinton, and Timothy P. Lillicrap. Assessing the scalability of biologically-motivated deep learning algorithms and architectures. In Advances in Neural Information Processing Systems 31, NeurIPS, 2018.  
Atilim Günes Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, and Philip H. S. Torr. Gradients without backpropagation. CoRR, abs/2202.08587, 2022.  
Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon. Greedy layerwise learning can scale to imagenet. arXiv preprint arXiv:1812.11446, 2018.  
Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon. Decoupled greedy learning of cnns. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 736-745. PMLR, 2020.  
Yoshua Bengio. How auto-encoders could provide credit assignment in deep networks via target propagation. CoRR, abs/1407.7906, 2014.  
Yoshua Bengio, Pascal Lamblin, Dan Popovici, and Hugo Larochelle. Greedy layer-wise training of deep networks. In Advances in Neural Information Processing Systems 19, Proceedings of the Twentieth Annual Conference on Neural Information Processing Systems, Vancouver, British Columbia, Canada, December 4-7, 2006, pages 153-160. MIT Press, 2006.  
Yoshua Bengio, Thomas Mesnard, Asja Fischer, Saizheng Zhang, and Yuhuai Wu. Stdp as presynaptic activity times rate of change of postsynaptic activity approximates back-propagation. Neural Computation, 10, 2017.  
Elie L Bienenstock, Leon N Cooper, and Paul W Munro. Theory for the development of neuron selectivity: orientation specificity and binocular interaction in visual cortex. Journal of Neuroscience, 2 (1):32-48, 1982.  
James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, ICML, 2020.  
David G. Clark, L. F. Abbott, and SueYeon Chung. Credit assignment through broadcasting a global error vector. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 10053-10066, 2021.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2009.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.  
Ila R Fiete and H Sebastian Seung. Gradient learning in spiking neural networks by dynamic perturbation of conductances. Physical review letters, 97(4):048104, 2006.  
Aidan N. Gomez, Oscar Key, Stephen Gou, Nick Frosst, Jeff Dean, and Yarin Gal. Interlocking backpropagation: Improving depthwise model-parallelism. CoRR, abs/2010.04116, 2020.  
Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Ávila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent - A new approach to self-supervised learning. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.  
Donald Olding Hebb. The organization of behavior: a neuropsychological theory. J. Wiley; Chapman & Hall, 1949.  
Geoffrey Hinton et al. How to do backpropagation in a brain. In Invited talk at the NIPS'2007 deep learning workshop, volume 656, pages 1-16, 2007.  
Geoffrey E. Hinton and James L. McClelland. Learning representations by recirculation. In Dana Z. Anderson, editor, Neural Information Processing Systems, Denver, Colorado, USA, 1987, pages 358-366, 1987.  
Geoffrey E. Hinton, Simon Osindero, and Yee Whye Teh. A fast learning algorithm for deep belief nets. Neural Comput., 18(7):1527-1554, 2006.  
Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015.  
Yani Ioannou, Duncan P. Robertson, Roberto Cipolla, and Antonio Criminisi. Deep roots: Improving CNN efficiency with hierarchical filter groups. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 5977-5986. IEEE Computer Society, 2017.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, ICML, 2015.  
Marwan Jabri and Barry Flower. Weight perturbation: An optimal architecture and learning technique for analog vlsi feedforward and recurrent multilayer networks. IEEE Transactions on Neural Networks, 3(1):154-157, 1992.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, NIPS, 2012.  
Michael Laskin, Luke Metz, Seth Nabarrao, Mark Saroufim, Badreddine Noune, Carlo Luschi, Jascha Sohl-Dickstein, and Pieter Abbeel. Parallel training of deep networks with local updates. CoRR, abs/2012.03837, 2020.  
Y LeCun, C Cortes, and C Burges. The mnist dataset of handwritten digits (images). NYU: New York, NY, USA, 1999.

Yann LeCun. A learning scheme for asymmetric threshold networks. Proceedings of COGNITIVA, 85(537):599-604, 1985.  
Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, and Yoshua Bengio. Difference target propagation. In Joint European conference on machine learning and knowledge discovery in databases, pages 498-515. Springer, 2015.  
Qianli Liao, Joel Z. Leibo, and Tomaso A. Poggio. How important is weight symmetry in backpropagation? In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI, 2016.  
Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, and Colin J. Akerman. Random synaptic feedback weights support error backpropagation for deep learning. Nature Communications, 7(1): 13276, Nov 2016. ISSN 2041-1723. doi: 10.1038/ncomms13276.  
Timothy P. Lillicrap, Adam Santoro, Luke Marris, Colin J. Akerman, and Geoffrey Hinton. Backpropagation and the brain. Nature Reviews Neuroscience, 21(6):335-346, Jun 2020. ISSN 1471-0048. doi: 10.1038/s41583-020-0277-3.  
Sindy Löwe, Peter O'Connor, and Bastiaan S. Veeling. Putting an end to end-to-end: Gradient-isolated learning of representations. In Advances in Neural Information Processing Systems 32, NeurIPS, 2019.  
Arild Nøkland. Direct feedback alignment provides learning in deep neural networks. In Advances in Neural Information Processing Systems 29, NeurIPS, 2016.  
Arild Nøkland and Lars Hiller Eidnes. Training neural networks with local error signals. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, volume 97 of Proceedings of Machine Learning Research, pages 4839-4850. PMLR, 2019.  
Erkki Oja. Simplified neuron model as a principal component analyzer. Journal of mathematical biology, 15(3):267-273, 1982.  
Adeetya Patel, Michael Eickenberg, and Eugene Belilovsky. Local learning with neuron groups. In From Cells to Societies: Collective Learning Across Scales - ICLR 2022 Workshop, 2022.  
Barak A Pearlmutter. Fast exact multiplication by the hessian. Neural computation, 6(1):147-160, 1994.  
Antti Rasmus, Mathias Berglund, Mikko Honkala, Harri Valpola, and Tapani Raiko. Semi-supervised learning with ladder networks. In Advances in Neural Information Processing Systems 28, NIPS, 2015.  
Mengye Ren, Renjie Liao, Raquel Urtasun, Fabian H. Sinz, and Richard S. Zemel. Normalizing the normalizers: Comparing and extending network normalization schemes. In Proceedings of the 5th International Conference on Learning Representations, ICLR, 2017.  
D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning internal representations by error propagation. In Parallel Distributed Processing: Explorations in the Microstructure of Cognition, Vol. 1: Foundations, page 318-362, Cambridge, MA, USA, 1986. MIT Press. ISBN 026268053X.  
Tim Salimans, Jonathan Ho, Xi Chen, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning. CoRR, abs/1703.03864, 2017.  
Wolfram Schultz, Peter Dayan, and P Read Montague. A neural substrate of prediction and reward. Science, 275(5306):1593-1599, 1997.  
H Sebastian Seung. Learning in spiking neural networks by reinforcement of stochastic synaptic transmission. Neuron, 40(6):1063-1073, 2003.  
Eren Sezener, Agnieszka Grabska-Barwińska, Dimitar Kostadinov, Maxime Beau, Sanjukta Krishnagopal, David Budden, Marcus Hutter, Joel Veness, Matthew Botvinick, Claudia Clopath, Michael Häusser, and Peter E. Latham. A rapid and efficient learning rule for biological neural circuits. 2021.

David Silver, Anirudh Goyal, Ivo Danihelka, Matteo Hessel, and Hado van Hasselt. Learning by directional gradient descent. In Proceedings of the 10th International Conference on Learning Representations, ICLR, 2022.  
Kenneth O Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. *Evol Comput*, 10(2):99–127, 2002.  
Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, and Alexey Dosovitskiy. Mlp-mixer: An all-mlp architecture for vision. In Advances in Neural Information Processing Systems 34, 2021.  
Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748, 2018.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998-6008, 2017.  
Joel Veness, Tor Lattimore, Avishkar Bhoopchand, Agnieszka Grabska-Barwinska, Christopher Mattern, and Peter Toth. Online learning with gated linear networks. arXiv preprint arXiv:1712.01897, 2017.  
Joel Veness, Tor Lattimore, David Budden, Avishkar Bhoopchand, Christopher Mattern, Agnieszka Grabska-Barwinska, Eren Sezener, Jianan Wang, Peter Toth, Simon Schmitt, et al. Gated linear networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10015-10023, 2021.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res., 11:3371-3408, 2010.  
Yulin Wang, Zanlin Ni, Shiji Song, Le Yang, and Gao Huang. Revisiting locally supervised learning: an alternative to end-to-end training. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.  
Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger B. Grosse. Flipout: Efficient pseudo-independent weight perturbations on mini-batches. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018.  
R. E. Wengert. A simple automatic derivative evaluation program. Commun. ACM, 7(8):463-464, 1964.  
Paul Werbos. Beyond regression:" new tools for prediction and analysis in the behavioral sciences. Ph.D. dissertation, Harvard University, 1974.  
Justin Werfel, Xiaohui Xie, and H Seung. Learning curves for stochastic gradient descent in linear feedforward networks. Advances in neural information processing systems, 16, 2003.  
L. Darrell Whitley. Genetic reinforcement learning for neurocontrol problems. Mach. Learn., 13: 259-284, 1993.  
James C.R. Whittington and Rafal Bogacz. Theories of error back-propagation in the brain. Trends in Cognitive Sciences, 23(3):235-250, Mar 2019. ISSN 1364-6613. doi: 10.1016/j.tics.2018.12.005.  
Bernard Widrow and Marcian E Hoff. Adaptive switching circuits. Technical report, Stanford Univ Ca Stanford Electronics Labs, 1960.  
Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural computation, 1(2):270-280, 1989.

Yuxin Wu and Kaiming He. Group normalization. In 15th European Conference on Computer Vision, ECCV, 2018.  
Will Xiao, Honglin Chen, Qianli Liao, and Tomaso A. Poggio. Biologically-plausible learning algorithms can scale to large datasets. In Proceedings of the 7th International Conference on Learning Representations, ICLR, 2019.  
Saining Xie, Ross B. Girshick, Piotr Dólar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2017.  
Xiaohui Xie and H Sebastian Seung. Spike-based learning rules and stabilization of persistent neural activity. Advances in neural information processing systems, 12, 1999.  
Yuwen Xiong, Mengye Ren, and Raquel Urtasun. Loco: Local contrastive representation learning. In Advances in Neural Information Processing Systems 33, NeurIPS, 2020.
