# THE INFLUENCE OF LEARNING RULE ON REPRESENTATION DYNAMICS IN WIDE NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

It is unclear how changing the learning rule of a deep neural network alters its learning dynamics and representations. To gain insight into the relationship between learned features, function approximation, and the learning rule, we analyze infinite-width deep networks trained with gradient descent (GD) and biologically-plausible alternatives including feedback alignment (FA), direct feedback alignment (DFA), and error modulated Hebbian learning (Hebb), as well as gated linear networks (GLN). We show that, for each of these learning rules, the evolution of the output function at infinite width is governed by a time varying effective neural tangent kernel (eNTK). In the lazy training limit, this eNTK is static and does not evolve, while in the rich mean-field regime this kernel's evolution can be determined self-consistently with dynamical mean field theory (DMFT). This DMFT enables comparisons of the feature and prediction dynamics induced by each of these learning rules. In the lazy limit, we find that DFA and Hebb can only learn using the last layer features, while full FA can utilize earlier layers with a scale determined by the initial correlation between feedforward and feedback weight matrices. In the rich regime, DFA and FA utilize a temporally evolving and depth-dependent NTK. Counterintuitively, we find that FA networks trained in the rich regime exhibit more feature learning if initialized with smaller correlation between the forward and backward pass weights. GLNs admit a very simple formula for their lazy limit kernel and preserve conditional Gaussianity of their preactivations under gating functions. Error modulated Hebb rules show very small task-relevant alignment of their kernels and perform most task relevant learning in the last layer.

# 1 INTRODUCTION

Deep neural networks have now attained state of the art performance across a variety of domains including computer vision and natural language processing (Goodfellow et al., 2016; LeCun et al., 2015). Central to the power and transferability of neural networks is their ability to flexibly adapt their layer-wise internal representations to the structure of the data distribution during learning.

In this paper, we explore how the learning rule that is used to train a deep network affects its learning dynamics and representations. Our primary motivation for studying different rules is that exact gradient descent (GD) training with the back-propagation algorithm is thought to be biologically implausible (Crick, 1989). While many alternatives to standard GD training were proposed (Whittington & Bogacz, 2019), it is unclear how modifying the learning rule changes the functional inductive bias and the learned representations of the network. Further, understanding the learned representations could potentially offer more insight into which learning rules account for representational changes observed in the brain (Poort et al., 2015; Kriegeskorte & Wei, 2021; Schumacher et al., 2022). Our current study is a step towards these directions.

The alternative learning rules we study are error modulated Hebbian learning (Hebb), Feedback alignment (FA) (Lillicrap et al., 2016) and direct feedback alignment (DFA) (Nokland, 2016). These rules circumvent one of the biologically implausible features of GD: the weights used in the backward pass computation of error signals must be dynamically identical to the weights used on the forward pass, known as the weight transport problem. Instead, FA and DFA algorithms compute an approximate backward pass with independent weights that are frozen through training. Hebb rule only uses a global error signal. While these learning rules do not perform exact GD, they are still

able to evolve their internal representations and eventually fit the training data. Further, experiments have shown that FA and DFA can scale to certain problems such as view-synthesis, recommendation systems, and small scale image problems (Launay et al., 2020), but they do not perform as well in convolutional architectures with more complex image datasets (Bartunov et al., 2018). However, significant improvements to FA can be achieved if the feedback-weights have partial correlation with the feedforward weights (Xiao et al., 2018; Moskovitz et al., 2018; Boopathy & Fiete, 2022).

We also study gated linear networks (GLNs), which implement nonlinearities through frozen gating functions (Fiat et al., 2019). Certain variants of these networks have biologically plausible interpretations in terms of dendritic gates (Sezener et al., 2021). Fixed gating has been shown to mitigate catastrophic forgetting (Veness et al., 2021; Budden et al., 2020), and achieve efficient transfer and multi-task learning Saxe et al. (2022).

Here, we explore how the choice of learning rule modifies the representations, functional biases and dynamics of deep networks at the infinite width limit, which allows a precise analytical description of the network dynamics in terms of a collection of evolving kernels. At infinite width, the network can operate in the lazy regime, where the feature embeddings at each layer are constant through time, or the rich/feature-learning regime (Chizat et al., 2019; Yang & Hu, 2021; Bordelon & Pehlevan, 2022). The richness is controlled by a scalar parameter related to the initial scale of the output function.

In summary, our novel contributions are the following:

1. We identify a class of learning rules for which function evolution is described by a dynamical effective Neural Tangent Kernel (eNTK). We provide a dynamical mean field theory (DMFT) for these learning rules which can be used to compute this eNTK. We show both theoretically and empirically that convergence to this DMFT occurs at large width  $N$  with error  $O(N^{-1/2})$ .  
2. We characterize precisely the inductive biases of infinite width networks in the lazy limit by computing their eNTKs at initialization. We generalize FA to allow partial correlation between the feedback weights and initial feedforward weights and show how this alters the eNTK.  
3. We then study the rich regime so that the features are allowed to adapt during training. In this regime, the eNTK is dynamical and we give a DMFT to compute it. For deep linear networks, the DMFT equations close algebraically, while for nonlinear networks we provide a numerical procedure to solve them.  
4. We compare the learned features and dynamics among these rules, analyzing the effect of richness, initial feedback correlation, and depth. We find that rich training enhances gradient-pseudogradient alignment for both FA and DFA. Counterintuitively, smaller initial feedback correlation generates more dramatic feature evolution for FA. The GLN networks have dynamics comparable to GD, while Hebb networks do not exhibit task relevant adaptation of feature kernels, but rather evolve according to the input statistics.

# 1.1 RELATED WORKS

GLNs were introduced by Fiat et al. (2019) as a simplified model of ReLU networks, allowing the analysis of convergence and generalization in the lazy kernel limit. Veness et al. (2021) provided a simplified and biologically-plausible learning rule for deep GLNs which was extended by Budden et al. (2020) and provided an interpretation in terms of dendritic gating Sezener et al. (2021). These works demonstrated benefits to continual learning due to the fixed gating. Saxe et al. (2022) derived exact dynamical equations for a GLN with gates operating at each node and each edge of the network graph. Krishnamurthy et al. (2022) provided a theory of gating in recurrent networks.

Lillicrap et al. (2016) showed that, in a two layer linear network the forward weights will evolve to align to the frozen feedback weights under the FA dynamics, allowing convergence of the network to a loss minimizer. This result was extended to deep networks by Frenkel et al. (2019), who also introduced a variant of FA where only the direction of the target is used. Refinetti et al. (2021) studied DFA in a two-layer student-teacher online learning setup, showing that the network first undergoes an alignment phase before converging to one the degenerate global minima of the loss. They argued that FA's worse performance in CNNs is due to the inability of the forward pass gradients to align under the block-Toeplitz connectivity structure that arises from enforced weight sharing (d'Ascoli et al., 2019). Garg & Vempala (2022) analyzed matrix factorization with FA, proving that, when overparameterized, it converges to a minimizer under standard conditions, albeit more slowly than GD.

Gradient flow has been analyzed at infinite width in both lazy regimes with the NTK (Jacot et al., 2018; Lee et al., 2019) and rich feature learning regimes (Mei et al., 2018). In the feature learning limit, the evolution of kernel order parameters have been obtained with both Tensor Programs framework (Yang & Hu, 2021) and with DMFT (Bordelon & Pehlevan, 2022). Song et al. (2021) recently analyzed the lazy infinite width limit of two layer networks trained with FA and weight decay, finding that only one layer effectively contributes to the two-layer NTK. Boopathy & Fiete (2022) proposed alignment based learning rules for networks at large width in the lazy regime, which performs comparably to GD and outperform standard FA. Their Align-Ada rule corresponds to our  $\rho$ -FA with  $\rho = 1$  in lazy large width networks.

# 2 EFFECTIVE NEURAL TANGENT KERNEL FOR A LEARNING RULE

We denote the output of a neural network for input  $\pmb{x}_{\mu} \in \mathbb{R}^{D}$  as  $f_{\mu}$ . For concreteness, in the main text we will focus on scalar targets  $f_{\mu} \in \mathbb{R}$  and MLP architectures. Other architectures such as multiclass outputs and CNN architectures with infinite channel count can also be analyzed as we show in the Appendix C. For the moment, we let the function be computed recursively from a collection of weight matrices  $\pmb{\theta} = \mathrm{Vec}\{W^0, W^1, \dots, w^L\}$  in terms of preactivation vectors  $h_{\mu}^{\ell} \in \mathbb{R}^{N}$  where,

$$
f _ {\mu} = \frac {1}{\gamma_ {0} N} \boldsymbol {w} ^ {L} \cdot \phi \left(\boldsymbol {h} _ {\mu} ^ {L}\right), \boldsymbol {h} _ {\mu} ^ {\ell + 1} = \frac {1}{\sqrt {N}} \boldsymbol {W} ^ {\ell} \phi \left(\boldsymbol {h} _ {\mu} ^ {\ell}\right), \boldsymbol {h} _ {\mu} ^ {1} = \frac {1}{\sqrt {D}} \boldsymbol {W} ^ {0} \boldsymbol {x} _ {\mu} \tag {1}
$$

where nonlinearity  $\phi$  is applied element-wise. The scalar parameter  $\gamma_0$  controls how rich the network training is: small  $\gamma_0$  corresponds to lazy learning while large  $\gamma_0$  generates large changes to the features (Chizat et al., 2019). For gated linear networks, we follow Fiat et al. (2019) and modify the forward pass equations by replacing  $\phi(h_\mu^\ell)$  with a multiplicative gating function  $\dot{\phi}(m_\mu^\ell)h_\mu^\ell$  where gating variables  $m_\mu^\ell = \frac{1}{\sqrt{D}} M^\ell x_\mu$  are fixed through training with  $M_{ij} \sim \mathcal{N}(0,1)$ . To minimize loss  $\mathcal{L} = \sum_{\mu} \ell(f_\mu, y_\mu)$ , we consider learning rules to the parameters  $\theta$  of the form

$$
\frac {d}{d t} \boldsymbol {w} ^ {L} = \gamma_ {0} \sum_ {\mu} \phi \left(\boldsymbol {h} _ {\mu} ^ {L} (t)\right) \Delta_ {\mu}, \frac {d}{d t} \boldsymbol {W} ^ {\ell} = \frac {\gamma_ {0}}{\sqrt {N}} \sum_ {\mu} \Delta_ {\mu} \tilde {\boldsymbol {g}} _ {\mu} ^ {\ell + 1} \phi \left(\boldsymbol {h} _ {\mu} ^ {\ell}\right) ^ {\top}, \frac {d}{d t} \boldsymbol {W} ^ {0} = \frac {\gamma_ {0}}{\sqrt {D}} \sum_ {\mu} \Delta_ {\mu} \tilde {\boldsymbol {g}} _ {\mu} ^ {1} \boldsymbol {x} _ {\mu} ^ {\top} \tag {2}
$$

where the error signal is  $\Delta_{\mu}(t) = -\frac{\partial\mathcal{L}}{\partial f_{\mu}} |_{f_{\mu}(t)}$ . The last layer weights  $\boldsymbol{w}^{L}$  are always updated with their true gradient. This corresponds to the biologically-plausible and local delta-rule, which merely correlates the error signals  $\Delta_{\mu}$  and the last layer features  $\phi (h_{\mu}^{L})$  (Widrow & Hoff, 1960). In intermediate layers, the pseudo-gradient vectors  $\tilde{\pmb{g}}_{\mu}^{\ell}$  are determined by the choice of the learning rule. For concreteness, we provide below the recursive definitions of  $\tilde{\pmb{g}}^{\ell}$  for our five learning rules of interest.

$$
\tilde {\boldsymbol {g}} _ {\mu} ^ {\ell} = \left\{ \begin{array}{l l} \dot {\phi} \left(\boldsymbol {h} _ {\mu} ^ {\ell}\right) \odot \left[ \frac {1}{\sqrt {N}} \boldsymbol {W} ^ {\ell} (t) ^ {\top} \tilde {\boldsymbol {g}} _ {\mu} ^ {\ell + 1} \right], & \tilde {\boldsymbol {g}} _ {\mu} ^ {L} = \dot {\phi} \left(\boldsymbol {h} _ {\mu} ^ {L}\right) \odot \boldsymbol {w} ^ {L} \\ \dot {\phi} \left(\boldsymbol {h} _ {\mu} ^ {\ell}\right) \odot \left[ \frac {1}{\sqrt {N}} \left(\rho \boldsymbol {W} ^ {\ell} (0) + \sqrt {1 - \rho^ {2}} \tilde {\boldsymbol {W}} ^ {\ell}\right) ^ {\top} \tilde {\boldsymbol {g}} ^ {\ell + 1} \right], & \tilde {W} _ {i j} ^ {\ell} \sim \mathcal {N} (0, 1) \\ \dot {\phi} \left(\boldsymbol {h} _ {\mu} ^ {\ell}\right) \odot \tilde {\boldsymbol {z}} ^ {\ell}, & \tilde {z} _ {i} ^ {\ell} \sim \mathcal {N} (0, 1) \\ \dot {\phi} \left(\boldsymbol {m} _ {\mu} ^ {\ell}\right) \odot \left[ \frac {1}{\sqrt {N}} \boldsymbol {W} ^ {\ell} (t) ^ {\top} \tilde {\boldsymbol {g}} _ {\mu} ^ {\ell + 1} \right], & \tilde {\boldsymbol {g}} ^ {L} = \dot {\phi} \left(\boldsymbol {m} _ {\mu} ^ {\ell}\right) \odot \boldsymbol {w} ^ {L} (t) \\ \Delta_ {\mu} (t) \phi \left(\boldsymbol {h} _ {\mu} ^ {\ell} (t)\right) & \end{array} \right. \tag {3}
$$

While GD uses the instantaneous feedforward weights on the backward pass,  $\rho$ -FA uses the weight matrices which do not evolve throughout training. These weights have correlation  $\rho$  with the initial forward pass weights  $W^{\ell}(0)$ . This choice is motivated by the observation that partial correlation between forward and backward pass weights at initialization can improve training (Liao et al., 2016; Xiao et al., 2018; Moskovitz et al., 2018), though the cost is partial weight transport at initialization. However, we consider partial correlation at initialization more biologically plausible than the demanding weight transport at each step of training, like in GD. For DFA, the weight vectors  $\tilde{z}^{\ell}$  are sampled randomly at initialization and do not evolve in time. For GLN, the gating variables  $m_{\mu}^{\ell}$  are frozen through time but the exact feedforward weights are used in the backward pass. Lastly, we modify the classic Hebb rule (Hebb, 1949) to get  $\Delta W^{\ell} \propto \sum_{\mu} \Delta_{\mu}(t)^{2} \phi(h_{\mu}^{\ell+1}) \phi(\tilde{h}_{\mu}^{\ell})^{\top}$ , which weighs each example by its current error. Unlike standard Hebbian updates, this learning rule gives stable dynamics without regularization (App. G).

For all of these rules, the evolution of the function is determined by a time-dependent eNTK  $K_{\mu \nu}$  which is defined as

$$
\frac {\partial f _ {\mu}}{\partial t} = \frac {\partial f _ {\mu}}{\partial \boldsymbol {\theta}} \cdot \frac {d \boldsymbol {\theta}}{d t} = \sum_ {\nu} \Delta_ {\nu} K _ {\mu \nu} (t, t), \quad K _ {\mu \nu} (t, s) = \sum_ {\ell = 0} ^ {L} \tilde {G} _ {\mu \nu} ^ {\ell + 1} (t, s) \Phi_ {\mu \nu} ^ {\ell} (t, s)
$$

$$
\tilde {G} _ {\mu \nu} ^ {\ell} (t, s) = \frac {1}{N} \boldsymbol {g} _ {\mu} ^ {\ell} (t) \cdot \tilde {\boldsymbol {g}} _ {\nu} ^ {\ell} (s), \quad \Phi_ {\mu \nu} ^ {\ell} (t, s) = \frac {1}{N} \phi (\boldsymbol {h} _ {\mu} ^ {\ell} (t)) \cdot \phi (\boldsymbol {h} _ {\nu} ^ {\ell} (s)), \tag {4}
$$

where the base cases  $\tilde{G}_{\mu \nu}^{L + 1}(t,s) = 1$  and  $\Phi_{\mu \nu}^{0}(t,s) = \frac{1}{D}\pmb{x}_{\mu}\cdot \pmb{x}_{\nu}$  are time-invariant. The kernel  $\tilde{G}^{\ell}$  computes an inner product between the true gradient signals  $g_{\mu}^{\ell} = \gamma_0N\frac{\partial f_{\mu}}{\partial h_{\mu}^{\ell}}$  and the pseudo-gradient  $\tilde{g}_{\nu}^{\ell}$  which is set by the chosen learning rule. We see that because  $\tilde{G}^{\ell}$  is not necessarily symmetric,  $\pmb{K}$  is also not necessarily symmetric. The matrix  $\tilde{G}^{\ell}$  quantifies pseudo-gradient / gradient alignment.

# 3 DYNAMICAL MEAN FIELD THEORY FOR VARIOUS LEARNING RULES

For each of these learning rules considered, the infinite width  $N \to \infty$  limit of network learning can be described by a dynamical mean field theory (DMFT) (Bordelon & Pehlevan, 2022). At infinite width, the dynamics of the kernels  $\Phi^{\ell}$  and  $\tilde{G}^{\ell}$  become deterministic over random Gaussian initialization of parameters  $\theta$ . The activity of neurons in each layer becomes i.i.d. random variables drawn from a distribution defined by these kernels. Further, the kernels can be computed as self-consistent averages over these single-site distributions. Below, we provide DMFT formulas which are valid for all of our learning rules

$$
h _ {\mu} ^ {\ell} (t) = u _ {\mu} ^ {\ell} (t) + \gamma_ {0} \int_ {0} ^ {t} d s \sum_ {\nu = 1} ^ {P} \left[ A _ {\mu \nu} ^ {\ell - 1} (t, s) g _ {\nu} ^ {\ell} (s) + C _ {\mu \nu} ^ {\ell - 1} (t, s) \tilde {g} _ {\nu} ^ {\ell} (s) + \Phi_ {\mu \nu} ^ {\ell - 1} (t, s) \Delta_ {\nu} (s) \tilde {g} _ {\nu} ^ {\ell} (s) \right]
$$

$$
z _ {\mu} ^ {\ell} (t) = r _ {\mu} ^ {\ell} (t) + \gamma_ {0} \int_ {0} ^ {t} d s \sum_ {\nu = 1} ^ {P} \left[ B _ {\mu \nu} ^ {\ell} (t, s) + \tilde {G} _ {\mu \nu} ^ {\ell + 1} (t, s) \Delta_ {\nu} (s) \right] \phi (h _ {\nu} ^ {\ell} (s)), g _ {\mu} ^ {\ell} (t) = \dot {\phi} (h _ {\mu} ^ {\ell} (t)) z _ {\mu} ^ {\ell} (t)
$$

$$
\{u _ {\mu} ^ {\ell} (t) \} \sim \mathcal {G P} (0, \Phi^ {\ell - 1}), \Phi_ {\mu \nu} ^ {\ell} (t, s) = \left\langle \phi (h _ {\mu} ^ {\ell} (t)) \phi (h _ {\nu} ^ {\ell} (s)) \right\rangle , A _ {\mu \nu} ^ {\ell} (t, s) = \gamma_ {0} ^ {- 1} \left\langle \frac {\delta}{\delta r _ {\nu} ^ {\ell} (s)} \phi (h _ {\mu} ^ {\ell} (t)) \right\rangle
$$

$$
\left\{r _ {\mu} ^ {\ell} (t) \right\} \sim \mathcal {G P} \left(0, \boldsymbol {G} ^ {\ell + 1}\right), \tilde {G} _ {\mu \nu} ^ {\ell} (t, s) = \left\langle g _ {\mu} ^ {\ell} (t) \tilde {g} _ {\nu} ^ {\ell} (s) \right\rangle , B _ {\mu \nu} ^ {\ell} (t, s) = \gamma_ {0} ^ {- 1} \left\langle \frac {\delta}{\delta u _ {\nu} ^ {\ell + 1} (s)} g _ {\mu} ^ {\ell + 1} (t) \right\rangle \tag {5}
$$

The definitions of  $\tilde{g}_{\mu}^{\ell}(t)$  depend on the learning rule and are described in Table 1.

<table><tr><td>Rule</td><td>GD</td><td>ρ-FA</td><td>DFA</td><td>GLN</td><td>Hebb</td></tr><tr><td>g′lμ(t)</td><td>φ(h′lμ(t))z′l(t)</td><td>φ(h′lμ(t))z′l(t)</td><td>φ(h′lμ(t))z′l(t)</td><td>φ(m′lμ)z′l(t)</td><td>Δμ(t)φ(h′lμ(t))</td></tr></table>

Table 1: The field definitions for each learning rule. For  $\rho$ -FA, the field has definition  $\tilde{z}_{\mu}^{\ell}(t) = \rho v_{\mu}^{\ell}(t) + \sqrt{1 - \rho^{2}} \tilde{\zeta}_{\mu}^{\ell}(t) + \gamma_{0} \int_{0}^{t} ds \sum_{\nu} D_{\mu \nu}^{\ell}(t,s) \phi(h_{\nu}^{\ell}(s))$  where  $\{v_{\mu}^{\ell}(t), \tilde{\zeta}_{\mu}^{\ell}(t)\}$  are Gaussian with  $\langle r_{\mu}^{\ell}(t)v_{\nu}^{\ell}(s)\rangle = \tilde{G}_{\mu\nu}^{\ell+1}(t,s)$ . The  $\tilde{\zeta}^{\ell}$  field is an independent Gaussian with correlation  $\left\langle \tilde{\zeta}_{\mu}^{\ell}(t)\tilde{\zeta}_{\nu}^{\ell}(s)\right\rangle = \left\langle \tilde{g}_{\mu}^{\ell+1}(t)\tilde{g}_{\nu}^{\ell+1}(s)\right\rangle = \tilde{\tilde{G}}_{\mu\nu}^{\ell+1}(t,s)$ . For DFA, the  $\tilde{z}^{\ell}$  field is static  $\tilde{z}^{\ell} \sim \mathcal{N}(0,1)$ . For GLN, we use  $\{m_{\mu}^{\ell}\} \sim \mathcal{N}(0,\pmb{K}^{x})$  instead of  $h_{\mu}^{\ell}(t)$  as a gating variable. The  $C^\ell$  order parameter is vanishing except for  $\rho$ -FA with  $\rho > 0$ .

We see that, for  $\{\mathrm{GD},\rho\text{-FA},\mathrm{DFA},\mathrm{Hebb}\}$  the distribution of  $h_\mu^\ell (t),z_\mu^\ell (t)$  are Gaussian throughout training only in the lazy  $\gamma_0\to 0$  limit for general nonlinear activation functions  $\phi (h)$ . However, conditional on  $\{m_{\mu}^{\ell}\}$ , the  $\{h^\ell ,z^\ell \}$  fields are all Gaussian for GLNs. For all algorithms except  $\rho$ -FA,  $C^\ell = 0$ . For  $\rho$ -FA we have  $C_\mu^\ell (t,s) = \gamma_0^{-1}\Bigl {\langle}\frac{\delta}{\delta v_\nu^\ell(s)}\phi (h_\mu^\ell (t))\Bigr {\rangle}$ .

As described in prior results on the GD case (Bordelon & Pehlevan, 2022), the above equations can be solved self-consistently in polynomial (in train-set size  $P$  and training steps  $T$ ) time. With

![](images/2e20640abb6921db4b7ffcdbf5930c0ed08a197bde980a0e9930d84d119da399.jpg)  
(a) Loss Dynamics

![](images/a6cd314f61ecb86c5fdf05f1b2986ffd1571697f1a3894d8386e324501ba14b7.jpg)  
(b) NTK-Target Alignment

![](images/066d86c645ded5bbe3c4d04a99956cdf36fa26b4c8ad1a16bfc2f91bd025d86c.jpg)  
(c)  $\tilde{G}$  Dynamics

![](images/a93e78d9696c1d7e1cee4c3455a2efa15f9c309249ba259547e8048a59845927.jpg)  
(d) Final  $h$  Distributions

![](images/8df912571bb3cae975577396b227482a18c0024427a806486db58596fdd28b94.jpg)  
(e) Final  $\Phi$  Kernels

![](images/73ee12f2921a0bf63d2a314e998247153c1f58a576dad4b34e8a16d1415f04b2.jpg)  
Figure 1: The DMFT predicts feature dynamics of large width networks trained with gradient descent (GD), feedback alignment (FA) with  $\rho = 0$ , gated linear network (GLN), and a error-modulated  $\beta = 1$  Hebb rule (Hebb) in the feature learning regime. (a) The loss dynamics in a two layer ( $L = 1$  hidden layer) network trained with these learning rules at richness  $\gamma_0 = 2$ . The network is trained on a collection of  $P = 10$  random vectors in  $D = 50$  dimensions. (b) The cosine similarity of the eNTK with the targets  $A(K,yy^{\top}) = \frac{y^{\top}Ky}{|K|_{F}|y|^{2}}$  reveals increasing alignment for all algorithms. Though FA starts with the lowest alignment, its final NTK task alignment exceeds that of GD. (c) The dynamics of the gradient-pseudogradient kernel  $\tilde{G}$  also reveals increasing correlation of  $g$  with  $\tilde{g}$ . FA starts with  $\tilde{G} = 0$  but  $\tilde{G}$  increases to non-zero value. (d) The distribution of hidden layer preactivations after training reveals non-Gaussian statistics for both GD and FA, but approximately Gaussian statistics for GLN. (e)-(f) The final  $\Phi$  and  $G$  kernels from theory and experiment.  
(f) Final  $\tilde{G}$  kernels

an estimate of the dynamical kernels  $\{\Phi_{\mu \nu}^{\ell}(t,s),\tilde{G}_{\mu \nu}^{\ell}(t,s),G_{\mu \nu}^{\ell}(t,s)\}$ , one computes the eNTK  $K_{\mu \nu}(t)$  and error dynamics  $\Delta_{\mu}(t)$ . From these objects, we can sample the stochastic processes  $\{h^\ell ,z^\ell ,\tilde{z}^\ell \}$  which can then be used to derive new refined estimates of the kernels. This procedure is repeated until convergence. This algorithm can be found in App. A. An example of such a solution is provided in Figure 1 for two layer ReLU networks trained with GD, FA, GLN, and Hebb. We show that our self-consistent DMFT accurately predicts training and kernel dynamics, as well as the density of preactivations  $\{h_\mu (t)\}$  and final kernels  $\{\Phi_{\mu \nu},\tilde{G}_{\mu \nu}\}$  for each learning rule. We observe substantial differences in the learned representations (Figure 1e), all predicted by our DMFT.

# 3.1 LAZY OR EARLY TIME STATIC-Kernel Limits

When  $\gamma_0\to 0$  , we see that the fields  $h_\mu^\ell (t)$  and  $z_{\mu}^{\ell}(t)$  are equal to the Gaussian variables  $u_{\mu}^{\ell}(0)$  and  $r_\mu^\ell (0)$  . In this limit, the eNTK  $K_{\mu \nu}$  remains static and has the form summarized in Table 2 in terms of the initial feature kernels  $\Phi^{\ell}$  and gradient kernels  $G^{\ell}$  . We derive these kernels in Appendix D.

Table 2: The initial eNTK  $K_{\mu \nu}$  for each learning rule. The GD kernel is the usual initial NTK of Jacot et al. (2018). For  $\rho$ -aligned FA, each layer  $\ell$ 's contribution to the eNTK is suppressed by a factor  $\rho^{L - \ell}$ . For DFA and Hebb, only the last layer feature kernel  $\Phi^L$  contributes to the NTK. For GLN, each layer has an identical contribution.  

<table><tr><td>Rule</td><td>GD</td><td>ρ-FA</td><td>DFA</td><td>GLN</td><td>Hebb</td></tr><tr><td>Kμν</td><td>∑Lℓ=0Gℓ+1μνΦℓμν</td><td>∑Lℓ=0ρL-lGℓ+1μνΦℓμν</td><td>ΦLμν</td><td>[←φ(mμ)←φ(mν)⟩LKxμν</td><td>ΦLμν</td></tr></table>

![](images/cb8f08ffc5c3c9539583fc6e14ffcb302f1d55b7544b7f4a8fd34c9b405af173.jpg)  
(a) ReLU FA varying  $\rho$

![](images/27ff273ccbc34500b1656e8a2ff4a83c3eaa1eeb76057d93bd8eb941b7cc827b.jpg)  
(b) ReLU FA varying  $L$

![](images/c8b9ca174d5e3060724e26df86b0bf69caf94b292cd91b165aef57b095a19780.jpg)  
(c) ReLU GLN varying  $L$

![](images/eac45ec5f45e89ae7bfcf4d4d9cd916144a1341b5727aa127b9e073566456110.jpg)  
(d)  $\Phi^{\ell}$  convergence

![](images/a16689063d84ab373241999a025f9be96d685c8798670894872241312d2d4014.jpg)  
(e)  $G^{\ell}$  convergence

![](images/71e62343a7a641b6a434fcac4de20a7799b126e2eccdd6794dbe506d0be4f39e.jpg)  
Figure 2: The lazy infinite width limits of the various learning rules can be fully summarized with their initial eNTK. (a) The kernels of  $\rho$ -aligned ReLU FA and ReLU GLN for inputs separated by angle  $\theta$ . (a) The kernels for varying  $\rho$  in  $\rho$ -aligned FA. Larger  $\rho$  has a sharper peak in the kernel around  $\theta = 0$ . The  $\rho \rightarrow 0$  limit recovers the NNGP kernel  $\Phi^L$  while the  $\rho \rightarrow 1$  limit gives the backprop NTK. (b) Deeper networks with partial alignment  $\rho = 0.5$ . (c) ReLU-GLN kernel sharpens with depth. (d)-(e) The relative error of the infinite width  $\Phi^\ell, G^\ell$  kernels in a width  $N$  ReLU neural network. The late layer  $\Phi^\ell$  and early layer  $G^\ell$  kernels have highest errors since finite size effects accumulate on forward and backward passes respectively. (f) The convergence of the depth  $L$  NTK reveals higher deviation from infinite width theory for small  $\rho$  and large depth  $L$ . All square errors go as  $|K_N - K_\infty|^2 \sim O_N(1/N)$ .  
(f) eNTK convergence

The feature  $P\times P$  matrices  $\Phi^{\ell}$ $\pmb{G}^{\ell}$  in Table 2 are computed recursively as

$$
\boldsymbol {\Phi} ^ {\ell} = \left\langle \phi (\boldsymbol {u}) \phi (\boldsymbol {u}) ^ {\top} \right\rangle_ {\boldsymbol {u} \sim \mathcal {N} \left(0, \boldsymbol {\Phi} ^ {\ell - 1}\right)}, \boldsymbol {G} ^ {\ell} = \boldsymbol {G} ^ {\ell + 1} \odot \left\langle \dot {\phi} (\boldsymbol {u}) \dot {\phi} (\boldsymbol {u}) ^ {\top} \right\rangle_ {\boldsymbol {u} \sim \mathcal {N} \left(0, \boldsymbol {\Phi} ^ {\ell - 1}\right)} \tag {6}
$$

with base cases  $\Phi^0 = K^x$  and  $G^{L + 1} = 11^\top$ . We provide interpretations of this result below.

- Backpropagation and  $\rho = 1$  FA recover the usual depth  $L$  NTK, with contributions from every layer  $K_{\mu \nu} = \sum_{\ell} G_{\mu \nu}^{\ell + 1} \Phi_{\mu \nu}^{\ell}$  at initialization. This kernel governs both training dynamics and test predictions in the lazy limit  $\gamma_0 \to 0$  (Jacot et al., 2018).  
- Standard  $\rho = 0$  FA, DFA and Hebb are equivalent to using the NNGP kernel  $K_{\mu \nu} \sim \Phi_{\mu \nu}^{L}$ , giving the Bayes posterior mean predictor (Matthews et al., 2018; Lee et al., 2018). In the  $\gamma_0, \rho \rightarrow 0$  limit, only the dynamics of the readout weights  $\boldsymbol{w}^{L}$  contribute to the evolution of  $f_{\mu}$  since error signals cannot successfully propagate backward and gradients cannot align with pseudo-gradients (App D). Thus, the standard  $\rho = 0$  FA will be indistinguishable from merely training of  $\boldsymbol{w}^{L}$  with the delta-rule unless the network is trained in the rich feature learning regime  $\gamma_0 > 0$ , where  $\tilde{G}^{\ell}$  can evolve. This effect was also noted in two layer networks by Song et al. (2021).  
- For intermediate  $\rho$ , the  $\rho$ -FA rule involves participation of each layer  $\ell$  with scale of  $\rho^{L - \ell}$ , since each layer's pseudo-gradient is only partially correlated with the true gradient, giving recursion  $\tilde{G}^{\ell} = \rho \tilde{G}^{\ell +1}$  with base case  $\tilde{G}^{L + 1} = G^{L + 1}$ .  
- The GLN's kernel in the lazy limit is determined entirely by the Gaussian gating variables  $\{m_{\mu}^{\ell}\} \sim \mathcal{N}(0, K^{x})$ . If  $\dot{\phi}(m) = 1$ , we recover the depth- $L$  linear kernel  $K_{\mu \nu} = L K_{\mu \nu}^{x}$ . If  $\dot{\phi}(m) = \sqrt{2} \Theta(m)$ , then, using the arccosine kernel identity of Cho & Saul (2009) we find  $K(\theta) = L \left[1 - \frac{1}{\pi} \theta\right]^{L} \cos \theta$  for two inputs on the sphere separated by angle  $\theta$ .

We visualize these kernels for deep ReLU networks and ReLU GLNs for normalized inputs  $|\pmb{x}|^2 = |\pmb{x}'|^2 = D$ , by plotting the kernel as a function of the angle  $\theta$  separating two inputs  $\cos(\theta) =$

$\frac{1}{D}\pmb{x}^{\top}\pmb{x}^{\prime}$ . We find that the kernels develop a sharp discontinuity at the origin  $\theta = 0$ , which becomes more exaggerated as  $\rho$  and  $L$  increase. We further show that the square difference of the width  $N$  kenels and the infinite width kernels go as  $O(N^{-1})$ . We show that this is expected from heuristic DMFT argument App. H. In the lazy  $\gamma_0 \to 0$  limit, these kernels define the eNTK and also the fixed point for network predictions on a test point  $\pmb{x}$ .

# 3.2 FEATURE LEARNING ENABLES GRADIENT/PSEUDO-GRADIENT ALIGNMENT AND KERNEL/TASK ALIGNMENT

In the last section, we saw that, in the  $\gamma_0\rightarrow 0$  limit, all algorithms have frozen preactivations and pregradient features  $\{h_\mu^\ell (t),z_\mu^\ell (t)\}$ . A consequence of this fact is that FA and DFA cannot increase their gradient-pseudogradient alignment throughout training in the lazy limit  $\gamma_0 = 0$ . However, if we increase  $\gamma_0$ , then the gradient features  $g_{\mu}^{\ell}(t)$  and pseudo-gradients  $\tilde{g}_{\mu}^{\ell}(t)$  evolve in time and can increase their alignment. In Figure 3, we show the effect of increasing  $\gamma_0$  on alignment dynamics in a depth 4 tanh network trained with DFA. In (b), we see that larger  $\gamma_0$  is associated with high task-alignment of the last layer feature kernel  $\Phi^L$ , which becomes essentially rank one and aligned to  $\pmb{y}\pmb{y}^{\top}$ . The asymptotic cosine similarity between gradients and pseudogrades also increase with  $\gamma_0$ . The eNTK also becomes aligned with the task relevant directions (shown in Figure 3 c), like has been observed in GD training (Baratin et al., 2021; Shan & Bordelon, 2021; Geiger et al., 2021; Atanasov et al., 2021). We see that width  $N$  networks have a dynamical eNTK  $K_{N}(t)$  which deviates from the DMFT eNTK  $K_{\infty}(t)$  by  $O(1 / N)$  in square loss. DMFT is more predictive for larger  $\gamma_0$  networks, suggesting a reduction in finite size variability due to task-relevant feature evolution.

![](images/6bdb6a8a7913bc7b59330a5a077f73865187e3def5c497753f55fd70298594f2.jpg)  
(a) DFA Train Loss

![](images/f8a207c24b209d68681bb761264de0a699b5317c4f998f458ce980bae27e76af.jpg)  
(b)  $\Phi^L$  Alignment

![](images/75b19a8552efe1fe6dd85d797d0914474ea31ceb97e4131c4071c6c739206808.jpg)  
(c)  $g^{\ell},\tilde{g}^{\ell}$  Correlation

![](images/a93a9e939211fa57b4725c9aff22393d0837d454a9c6b6cdac50e2f4592793b3.jpg)

![](images/593d7ffb9f63416d4e8a5e0d45b51b197c2fc9bce325401d57cfce9d1a5b5383.jpg)  
(d) Final NTK Aligns to Task

![](images/57d83d179c493df8aacf4ddf4a088a523c5d5794e3b8f4b7f55392670d4fe06a.jpg)  
Figure 3: Feature Learning enables alignment for a depth  $4$  ( $L = 3$  hidden layers) tanh network trained with direct feedback alignment (DFA) with varying  $\gamma_0$ . (a) Training loss for DFA networks with width  $N = 4000$  with varying richness  $\gamma_0$  shows that feature learning accelerates training, as predicted by DMFT (black). (b) The alignment (cosine similarity) of the last layer kernel  $\Phi^L$  with the target function reveals successful task dependent feature learning at large  $\gamma_0$ . (c) The dynamics of pseudo-grad./grad. correlation  $\mathrm{corr}(\pmb{g}, \tilde{\pmb{g}}) = \frac{1}{LP} \sum_{\ell, \mu} \frac{\pmb{g}_{\mu}^{\ell}(t) \cdot \tilde{\pmb{g}}_{\mu}^{\ell}(t)}{| \pmb{g}_{\mu}^{\ell}(t) | |\tilde{\pmb{g}}_{\mu}^{\ell}(t)|}$  averaged over layers  $\ell$  and datapoints  $\mu$ . Larger  $\gamma_0$  generates more significant alignment between pseudogradients and gradients. (d) The final NTKs as a function of  $\gamma_0$  reveals increasing clustering of the data points by class. (e) The error of the DMFT approximation for  $K$ 's dynamics as a function of  $N$ :  $\frac{\langle |\pmb{K}_N(t) - \pmb{K}_{\infty}(t)|^2 \rangle_t}{\langle |\pmb{K}_{\infty}(t)|^2 \rangle_t} \sim O(N^{-1})$ , where the averages are computed over the time interval of training. This error is smaller for larger feature learning strength  $\gamma_0$ .  
(e) Dynamical NTK Convergence

# 3.3 DEEP LINEAR NETWORK KERNEL DYNAMICS

When  $\gamma_0 > 0$  the kernels and features in the network evolve according to the DMFT equations. For deep linear networks we can analyze the equations for the kernels in closed form without sampling since the correlation functions close algebraically (App. E). In Figure 4, we utilize our algebraic DMFT equations to explore  $\rho$ -FA dynamics in a depth 4 linear network. Networks with larger  $\rho$  train faster, which can be intuited by noting that the initial function time derivative  $\frac{df}{dt}\big|_{t=0} \sim \sum_{\ell=0}^{L} \rho^{L-\ell} \sim \frac{1 - \rho^{L+1}}{1 - \rho}$  is an increasing function of  $\rho$ . We observe higher final gradient pseudogradient alignment in each layer with larger  $\rho$ , which is also intuitive from the initial condition  $\tilde{G}^{\ell}(0) = \rho^{L-\ell}$ . However, surprisingly, for large initial correlation  $\rho$ , the NTK achieves lower task alignment, despite having larger  $\tilde{G}^{\ell}(t)$ . We show that this is caused by smaller overlap of each layer's feature kernel  $H^{\ell}(t)$  with  $\boldsymbol{yy}^{\top}$ . Though this phenomenon is counterintuitive, we gain more insight in the next section by studying an even simpler two layer model.

![](images/7355b7a2bfe55fa544639b757b4a040b181c090f707f6a01b11584738045615c.jpg)  
(a)  $\rho$ -Aligned Loss Dynamics

![](images/1cda878ca0cb79d560f117ae587d127740c429a5c733e1bcc28c612bf425d23a.jpg)  
(b) Gradient-Pseudogradient Kernel Dynamics

![](images/19c6dfec38700a2035bc0fc1d12eb67c2aa63b95d62077917cc37d38639a5ae9.jpg)

![](images/b4d46028afc25ea84de585062898a3efbe99194486fe2ce5de83c5df114d7161.jpg)

![](images/0e2a8d1d6ba199265a558dd1c02a6237b15a6598743ebd9d96eb6c0fb9661af6.jpg)  
(c) NTK-Task Alignment

![](images/87f329ac0819a424b103796746d515e760f4c58939fb7e30450686808a9bce72.jpg)  
(d) Feature Kernel Task Overlap

![](images/9aa1c3f1814eed44b829a1782c9f09250c5e8631d0845e5ffb277933f51c3f66.jpg)  
Figure 4: The initial feedback correlation  $\rho$  alters alignment dynamics in on the FA dynamics in a depth  $4(L = 3$  hidden layer) linear network. (a) Larger  $\rho$  leads to faster initial training since the scale of the eNTK is larger. (b) Further, larger  $\rho$  leads to larger scales of  $\tilde{G} (t) = \frac{1}{N}\pmb {g}^{\ell}(t)\cdot \tilde{\pmb{g}}^{\ell}(t)$ . (c) However, smaller  $\rho$  leads to more alignment of the NTK  $\pmb{K}(t)$  with the task-relevant subspace, measured with cosine similarity  $A(K,\pmb{y}\pmb{y}^{\top})$ . (d) The feature kernel  $\pmb{H}(t)$  overlaps with  $\pmb{y}$  reveal that  $\pmb{H}^{\ell}(t)$  aligns more significantly in the small  $\rho$  networks.

![](images/d75d9076949bacdb6e100be7c8c8b38a3e89fe6de42afb0a186d366afff11374.jpg)

# 3.3.1 EXACTLY SOLVEABLE FA TWO LAYER LINEAR NETWORK

We can provide exact solutions to the infinite width GD and  $\rho$ -FA dynamics in the setting of Saxe et al. (2013), specifically a two layer linear network trained with whitened data  $K_{\mu \nu}^{x} = \delta_{\mu \nu}$ . Unlike Saxe et al. (2013)'s result, however, we do not demand small initialization scale (or equivalently large  $\gamma_0$ ), but rather provide the exact solution for all positive  $\gamma_0$ . We will establish that large initial correlation  $\rho$  results in higher gradient/pseudogradient alignment but lower alignment of the hidden feature kernel  $\pmb{H}(t)$  with the task relevant subspace  $\pmb{yy}^{\top}$ .

We first note that when  $\pmb{K}^x = \pmb{I}$ , the GD or FA hidden feature kernel  $H(t)$  only evolves in the rank-one  $\pmb{yy}^\top$  subspace. It thus suffices to track the projection of  $H(t)$  on this rank one subspace, which we call  $H_y(t)$ . In the App. F we derive dynamics for  $H_y$  for GD and  $\rho$ -FA

$$
H _ {y} (t) = \left\{ \begin{array}{l l} \tilde {G} (t) = \sqrt {1 + \gamma_ {0} ^ {2} (y - \Delta (t)) ^ {2}}, \frac {d \Delta}{d t} = - \sqrt {1 + \gamma_ {0} ^ {2} (y - \Delta (t)) ^ {2}} \Delta (t) & \mathrm {G D} \\ 2 \tilde {G} (t) + 1 - 2 \rho = 1 + a ^ {2}, \frac {d a}{d t} = \gamma_ {0} y - \frac {1}{2} a ^ {3} - (1 + \rho) a & \rho \text {- F A} \end{array} \right. \tag {7}
$$

We illustrate these dynamics in Figure 5. The fixed points are  $H_{y} = \sqrt{1 + \gamma_{0}^{2}y^{2}}$  for GD and for  $\rho$ -FA,  $H_{y} = 1 + a^{2}$  where  $a$  is the smallest positive root of  $\frac{1}{2} a^3 + (1 + \rho)a = \gamma_0y$ . For both GD and FA, we see that increasing  $\gamma_{0}$  results in larger asymptotic values for  $H_{y}$  and  $\tilde{G}$ . For  $\rho$ -FA the

fixed point of  $a$ 's dynamics is a strictly decreasing function of  $\rho$  since  $\frac{da}{d\rho} < 0$ , showing that the final value of  $H_{y}$  is smaller for larger  $\rho$ . On the contrary, we have that the final  $\tilde{G} = \rho + \frac{1}{2} a^{2}$  is a strictly increasing function of  $\rho$  since  $\frac{d}{d\rho}\tilde{G} = 1 - \frac{a^2}{\frac{3}{2}a^2 + (1 + \rho)} > \frac{1}{3} > 0$ . Thus, this simple model replicates the phenomenon of increasing  $\tilde{G}$  and decreasing  $H_{y}$  as  $\rho$  increases. For the Hebb rule with  $K^{x} = I$ , the story is different. Instead of aligning  $H$  along the rank-one task relevant subspace, the dynamics instead decouple over samples, giving the following  $P$  separate equations

$$
\frac {d}{d t} \Delta_ {\mu} = - \left[ H _ {\mu \mu} (t) + \gamma_ {0} \Delta_ {\mu} \left(y _ {\mu} - \Delta_ {\mu}\right) \right] \Delta_ {\mu} (t), \frac {d}{d t} H _ {\mu \mu} = 2 \gamma_ {0} \Delta_ {\mu} (t) ^ {2} H _ {\mu \mu}. \tag {8}
$$

From this perspective, we see that the hidden feature kernel does not align to the task, but rather increases its entries in overall scale as is illustrated in Figure 5 (b).

![](images/6255cf09dfd84bb8736371f26977703198bba248ed99f532d6fd47d2c8fb4a0c.jpg)  
(a) Loss Dynamics

![](images/cd714c478a14ca8a5d9937eb500e45a3f63f12de8f1caab6bd8230a390c261bb.jpg)  
Figure 5: The feature kernel dynamics and scaling with  $\gamma_0^2$  for GD,  $\rho$ -FA, and Hebbian rules in an exactly solvable two layer linear network. (a) The loss dynamics for all algorithms reveals that  $\rho = 0$  FA and Hebb rules have same early time dynamics and that  $\rho = 1$  FA and GD have same early-time dynamics. However all loss curves become distinct at late times due to different eNTK dynamics. (b) The alignment of the kernel to the target function  $H_y(t) = \frac{1}{|\pmb{y}|^2}\pmb{y}^\top \pmb{H}\pmb{y} / \mathrm{Tr}\pmb{H}(t)$  increases significantly for GD, and FA, but not for Hebb, reflecting the task-independence of the learned representation. (c) The movement of the feature kernel  $\Delta H_y = \lim_{t\to \infty}H_y(t) - H_y(0)$  as a function of  $\gamma_0$  for GD, and  $\rho = 0,1$  FA. At small feature learning strength, all algorithms have updates on the order of  $\Delta H_y\sim \gamma_0^2$ . At large  $\gamma_0$ , GD has  $\Delta H_y\sim \gamma_0$  while FA has  $\Delta H_y\sim \gamma_0^{2 / 3}$ . The  $\rho = 1$  FA (green) has lower  $\Delta H_y$  than the  $\rho = 0$  FA across all  $\gamma_0$ .

![](images/c9604887888f4a4c61df8df0c8f1a694bf733714398dc20c66c5bbd10f819c43.jpg)  
(b) Kernel-Task Alignment  
(c) Feature Learning vs  $\gamma_0$

# 4 DISCUSSION

We provided an analysis of the training dynamics of a wide range of learning rules at infinite width. This set of rules includes (but is not limited to) GD,  $\rho$ -FA, DFA, GLN and Hebb as well as many others. We showed that each of these learning rules has an dynamical effective NTK which concentrates over initializations at infinite width. In the lazy  $\gamma_0 \to 0$  regime, it suffices to compute the initial NTK, while in the rich regime, we provide a dynamical mean field theory to compute the NTK's dynamics. We showed that, in the rich regime, FA learning rules do indeed align the network's gradient vectors to their pseudo-gradients and that this alignment improves with  $\gamma_0$ . We show that initial correlation  $\rho$  between forward and backward pass weights alters the inductive bias of FA in both lazy and rich regimes. In the rich regime, larger  $\rho$  networks have smaller eNTK evolution. Overall, our study is a step towards understanding learned representations in neural networks, and the quest to reverse-engineer learning rules from observations of evolving neural representations during learning in the brain.

Many open problems remain unresolved with the present work. We currently have only implemented our theory in MLPs. An implementation in CNNs could begin to explain some of the observed advantages of partial initial alignment in  $\rho$ -FA (Xiao et al., 2018; Moskovitz et al., 2018; Bartunov et al., 2018; Refinetti et al., 2021). In addition, our framework is sufficiently flexible to propose and test new learning rules by providing new  $\tilde{g}_{\mu}^{\ell}(t)$  formulas. Our DMFT gives a recipe to compute their initial kernels, function dynamics and analyze their learned representations. The generalization performance of these learning rules at varying  $\gamma_0$  is yet to be explored. Lastly, our DMFT is numerically expensive for large datasets and large training intervals, making it difficult to scale up to  $\sim 10k$  datapoints for MNIST or CIFAR-10.

# REFERENCES

Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. arXiv preprint arXiv:2111.00034, 2021.  
Aristide Baratin, Thomas George, César Laurent, R Devon Hjelm, Guillaume Lajoie, Pascal Vincent, and Simon Lacoste-Julien. Implicit regularization via neural feature alignment. In Arindam Banerjee and Kenji Fukumizu (eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 2269-2277. PMLR, 13-15 Apr 2021. URL https://proceedings.mlr.press/v130/baratin21a.html.  
Sergey Bartunov, Adam Santoro, Blake Richards, Luke Marris, Geoffrey E Hinton, and Timothy Lillicrap. Assessing the scalability of biologically-motivated deep learning algorithms and architectures. Advances in neural information processing systems, 31, 2018.  
Carl M Bender, Steven Orszag, and Steven A Orszag. Advanced mathematical methods for scientists and engineers I: Asymptotic methods and perturbation theory, volume 1. Springer Science & Business Media, 1999.  
Akhilan Boopathy and Ila Fiete. How to train your wide neural network without backprop: An input-weight alignment perspective. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 2178-2205. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/boopathy22a.html.  
Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks, 2022. URL https://arxiv.org/abs/2205.09653.  
David Budden, Adam Marblestone, Eren Sezener, Tor Lattimore, Gregory Wayne, and Joel Veness. Gaussian gated linear networks. Advances in Neural Information Processing Systems, 33:16508-16519, 2020.  
Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in Neural Information Processing Systems, 32, 2019.  
Youngmin Cho and Lawrence Saul. Kernel methods for deep learning. Advances in neural information processing systems, 22, 2009.  
Francis Crick. The recent excitement about neural networks. Nature, 337(6203):129-132, 1989.  
A Crisanti and H Sompolinsky. Path integral approach to random neural networks. Physical Review E, 98(6):062120, 2018.  
Stéphane d'Ascoli, Levent Sagun, Giulio Biroli, and Joan Bruna. Finding the needle in the haystack with convolutions: on the benefits of architectural bias. Advances in Neural Information Processing Systems, 32, 2019.  
Jonathan Fiat, Eran Malach, and Shai Shalev-Shwartz. Decoupling gating from linearity. arXiv preprint arXiv:1906.05032, 2019.  
Charlotte Frenkel, Martin Lefebvre, and David Bol. Learning without feedback: direct random target projection as a feedback-alignment algorithm with layerwise feedforward training. arXiv preprint arXiv:1909.01311, 10, 2019.  
Shivam Garg and Santosh Vempala. How and when random feedback works: A case study of low-rank matrix factorization. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera (eds.), Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pp. 4070-4108. PMLR, 28-30 Mar 2022. URL https://proceedings.mlr.press/v151/garg22a.html.  
Mario Geiger, Leonardo Petrini, and Matthieu Wyart. Landscape and training regimes in deep learning. Physics Reports, 924:1-18, 2021.

Gabriel Goh. Why momentum really works. Distill, 2(4):e6, 2017.  
Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.  
Donald O. Hebb. The organization of behavior: A neuropsychological theory. Wiley, New York, June 1949. ISBN 0-8058-4300-0.  
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems, 31, 2018.  
Mehran Kardar. Statistical physics of fields. Cambridge University Press, 2007.  
Nikolaus Kriegeskorte and Xue-Xin Wei. Neural tuning and representational geometry. Nature Reviews Neuroscience, 22(11):703-718, 2021.  
Kamesh Krishnamurthy, Tankut Can, and David J Schwab. Theory of gating in recurrent neural networks. Physical Review X, 12(1):011011, 2022.  
Julien Launay, Iacopo Poli, François Boniface, and Florent Krzakala. Direct feedback alignment scales to modern deep learning tasks and architectures. Advances in neural information processing systems, 33:9346-9360, 2020.  
Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436-444, 2015.  
Jaehoon Lee, Jascha Sohl-dickstein, Jeffrey Pennington, Roman Novak, Sam Schoenholz, and Yasaman Bahri. Deep neural networks as gaussian processes. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=B1EA-M-0Z.  
Jaehoon Lee, Lechao Xiao, Samuel Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems, 32, 2019.  
Qianli Liao, Joel Leibo, and Tomaso Poggio. How important is weight symmetry in backpropagation? In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.  
Timothy P Lillicrap, Daniel Cownden, Douglas B Tweed, and Colin J Akerman. Random synaptic feedback weights support error backpropagation for deep learning. Nature communications, 7(1): 1-10, 2016.  
Alessandro Manacorda, Grégory Schehr, and Francesco Zamponi. Numerical solution of the dynamical mean field theory of infinite-dimensional equilibrium liquids. The Journal of chemical physics, 152(16):164506, 2020.  
Paul Cecil Martin, ED Siggia, and HA Rose. Statistical dynamics of classical systems. Physical Review A, 8(1):423, 1973.  
Alexander G.D.G. Matthews, Jiri Hron, Mark Rowland, Richard E. Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=H1-nGgWC-.  
Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences, 115(33):E7665-E7671, 2018.  
Theodore H Moskovitz, Ashok Litwin-Kumar, and LF Abbott. Feedback alignment in deep convolutional networks. arXiv preprint arXiv:1812.06488, 2018.  
Arild Nøkland. Direct feedback alignment provides learning in deep neural networks. Advances in neural information processing systems, 29, 2016.

Jasper Poort, Adil G Khan, Marius Pachitariu, Abdellatif Nemri, Ivana Orsolic, Julija Krupic, Marius Bauza, Maneesh Sahani, Georg B Keller, Thomas D Mrsic-Flogel, et al. Learning enhances sensory and multiple non-sensory representations in primary visual cortex. Neuron, 86(6):1478-1490, 2015.  
Maria Refinetti, Stéphane d'Ascoli, Ruben Ohana, and Sebastian Goldt. Align, then memorise: the dynamics of learning with feedback alignment. In International Conference on Machine Learning, pp. 8925-8935. PMLR, 2021.  
Andrew Saxe, Shagun Sodhani, and Sam Jay Lewallen. The neural race reduction: Dynamics of abstraction in gated networks. In International Conference on Machine Learning, pp. 19287-19309. PMLR, 2022.  
Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.  
Joseph W Schumacher, Matthew K McCann, Katherine J Maximov, and David Fitzpatrick. Selective enhancement of neural coding in v1 underlies fine-discrimination learning in tree shrew. Current Biology, 32(15):3245-3260, 2022.  
Eren Sezener, Agnieszka Grabska-Barwińska, Dimitar Kostadinov, Maxime Beau, Sanjukta Krish-nagopal, David Budden, Marcus Hutter, Joel Veness, Matthew Botvinick, Claudia Clopath, et al. A rapid and efficient learning rule for biological neural circuits. BioRxiv, 2021.  
Haozhe Shan and Blake Bordelon. A theory of neural tangent kernel alignment and its influence on training. arXiv e-prints, pp. arXiv-2105, 2021.  
Ganlin Song, Ruitu Xu, and John Lafferty. Convergence and alignment of gradient descent with random backpropagation weights. Advances in Neural Information Processing Systems, 34:19888-19898, 2021.  
Joel Veness, Tor Lattimore, David Budden, Avishkar Bhoopchand, Christopher Mattern, Agnieszka Grabska-Barwinska, Eren Sezener, Jianan Wang, Peter Toth, Simon Schmitt, et al. Gated linear networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 10015-10023, 2021.  
James CR Whittington and Rafal Bogacz. Theories of error back-propagation in the brain. Trends in cognitive sciences, 23(3):235-250, 2019.  
Bernard Widrow and Marcian E Hoff. Adaptive switching circuits. Technical report, Stanford Univ Ca Stanford Electronics Labs, 1960.  
Will Xiao, Honglin Chen, Qianli Liao, and Tomaso Poggio. Biologically-plausible learning algorithms can scale to large datasets. arXiv preprint arXiv:1811.03567, 2018.  
Greg Yang and Edward J. Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 11727-11737. PMLR, 18-24 Jul 2021. URL https://proceedings.mlr.press/v139/yang21c.html.
