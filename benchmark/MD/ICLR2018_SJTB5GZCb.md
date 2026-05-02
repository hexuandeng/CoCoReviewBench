# EXTENDING THE FRAMEWORK OF EQUILIBRIUM PROPAGATION TO GENERAL DYNAMICS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The biological plausibility of the backpropagation algorithm has long been doubted by neuroscientists. Two major reasons are that neurons would need to send two different types of signal in the forward and backward phases, and that pairs of neurons would need to communicate through symmetric bidirectional connections. We present a simple two-phase learning procedure for fixed point recurrent networks that addresses both these issues. In our model, neurons perform leaky integration and synaptic weights are updated through a local mechanism. Our learning method extends the framework of Equilibrium Propagation to general dynamics, relaxing the requirement of an energy function. As a consequence of this generalization, the algorithm does not compute the true gradient of the objective function, but rather approximates it at a precision which is proven to be directly related to the degree of symmetry of the feedforward and feedback weights. We show experimentally that the intrinsic properties of the system lead to alignment of the feedforward and feedback weights, and that our algorithm optimizes the objective function.

# 1 INTRODUCTION

Deep learning (LeCun et al., 2015) is the de-facto standard in areas such as computer vision (Krizhevsky et al., 2012), speech recognition (Hinton et al., 2012) and machine translation (Bahdanau et al., 2015). These applications deal with different types of data and share little in common at first glance. Remarkably, all these models typically rely on the same basic principle: optimization of objective functions using the backpropagation algorithm. Hence the question: does the cortex in the brain implement a mechanism similar to backpropagation, which optimizes objective functions?

The backpropagation algorithm used to train neural networks requires a side network for the propagation of error derivatives, which is vastly seen as biologically implausible (Crick, 1989). One hypothesis, first formulated by Hinton & McClelland (1988), is that error signals in biological networks could be encoded in the temporal derivatives of the neural activity and propagated through the network via the neuronal dynamics itself, without the need for a side network. Neural computation would correspond to both inference and error back-propagation. This work also explores this idea.

The framework of Equilibrium Propagation (Scellier & Bengio, 2017) requires the network dynamics to be derived from an energy function, enabling computation of an exact gradient of an objective function. However, in terms of biological realism, the requirement of symmetric weights between neurons arising from the energy function is not desirable. The work presented here extends this framework to general dynamics, without the need for energy functions, gradient dynamics, or symmetric connections.

Our approach is the following. We start from classical models in neuroscience for the dynamics of the neuron's membrane voltage and for the synaptic plasticity (section 3). Unlike in the Hopfield model (Hopfield, 1984), we do not assume pairs of neurons to have symmetric connections. We then describe an algorithm for supervised learning based on these models (section 4) with minimal extra assumptions. Our model is based on two phases: at prediction time, no synaptic changes occur, whereas a local update rule becomes effective when the targets are observed. The proposed update mechanism is compatible with spike-timing-dependent plasticity (Bengio et al., 2017), which supposedly governs synaptic changes in biological neural systems. Finally, we show that the proposed algorithm has the desirable machine learning property of optimizing an objective function (section 5). We show this experimentally (Figure 3) and we provide the beginning for a theoretical explanation.

# 2 MOVING BEYOND ENERGY-BASED MODELS AND GRADIENT DYNAMICS

Historically, models based on energy functions and/or gradient dynamics have represented a key subject of neural network research. Their mathematical properties often allow for a simplified analysis, in the sense that there often exists an elegant formula or algorithm for computing the gradient of the objective function (Ackley et al., 1985; Movellan, 1990; Scellier & Bengio, 2017). However, we argue in this section that

1. due to the energy function, such models are very restrictive in terms of dynamics they can model - for instance the Hopfield model requires symmetric weights,  
2. machine learning algorithms do not require computation of the gradient of the objective function, as shown in this work and the work of Lillicrap et al. (2016).

In this work, we propose a simple learning algorithm based on few assumptions. To this end, we relax the requirement of the energy function and, at the same time, we give up on computing the gradient of the objective function.

We believe that, in order to make progress in biologically plausible machine learning, dynamics more general than gradient dynamics should be studied.

As discussed in section 6, another motivation for studying more general dynamics is the possible implementation of machine learning algorithms, such as our model, on analog hardware: analog circuits implement differential equations, which do not generally correspond to gradient dynamics.

# 2.1 GRADIENT DYNAMICS ARE NOT GENERIC DYNAMICS

Most dynamical systems observed in nature cannot be described by gradient dynamics. A gradient field is a very special kind of vector field, precisely because it derives from a primitive scalar function. The existence of a primitive function considerably limits the "number of degrees of freedom" of the vector field and implies important restrictions on the dynamics.

In general, a vector field does not derive from a primitive function. In particular, the dynamics of the leaky integrator neuron model studied in this work (Eq. 1) is not a gradient dynamics, unless extra (biologically implausible) assumptions are made, such as exact symmetry of synaptic weights  $(W_{ij} = W_{ji})$  in the case of the Hopfield model.

# 2.2 MACHINE LEARNING DOES NOT REQUIRE GRADIENT COMPUTATION

Machine learning relies on the basic principle of optimizing objective functions. Most of the work done in deep learning has focused on optimizing objective functions by gradient descent in the weight space (thanks to backpropagation). Although it is very well known that following the gradient is not necessarily the best option – many optimization methods based on adaptive learning rates for individual parameters have been proposed such as RMSprop Tieleman & Hinton (2012) and Adagrad Duchi et al. (2011) – almost all proposed optimization methods rely on computing the gradient, even if they do not follow the gradient. In the field of deep learning, “computing the gradient” has almost become synonymous with “optimizing”.

In fact, in order to optimize a given objective function, not only following the gradient unnecessary, but one does not even need to compute the gradient of that objective function. A weaker sufficient condition is to compute a direction in the parameter space whose scalar product with the gradient is negative, without computing the gradient itself.

A major step forward was achieved by Lillicrap et al. (2016). One of the contributions of their work was to dispel the long-held assumption that a learning algorithm should compute the gradient of an objective function in order to be sound. Their algorithm computes a direction in the parameter space that has at first sight little to do with the gradient of the objective function. Yet, their algorithm "learns" in the sense that it optimizes the objective function. By giving up on the idea of computing the gradient of the objective function, a key aspect rendering backpropagation biologically implausible could be fixed, namely the weight transport problem.

The work presented here is along the same lines. We give up on the idea of computing the gradient of the objective function, and by doing so, we get rid of the biologically implausible symmetric connections required in the Hopfield model. In this sense, the "weight transport" problem in the

backpropagation algorithm appears to be similar, at a high level, to the requirement of symmetric connections in the Hopfield model.

We suggest that in order to make progress in biologically plausible machine learning, it might be necessary to move away from computing the true gradients in the weight space. An important theoretical effort to be made is to understand and characterize the dynamics in the weight space that optimize objective functions. The set of such dynamics is of course much larger than the tiny subset of gradient dynamics.

# 3 CLASSICAL DYNAMICS IN NEUROSCIENCE

We denote by  $s_i$  the averaged membrane voltage of neuron  $i$  across time, which is continuous-valued and plays the role of a state variable for neuron  $i$ . We also denote by  $\rho(s_i)$  the firing rate of neuron  $i$ . We suppose that  $\rho$  is a deterministic function (nonlinear activation) that maps the averaged voltage  $s_i$  to the firing rate  $\rho(s_i)$ . The synaptic strength from neuron  $j$  to neuron  $i$  is denoted by  $W_{ij}$ .

# 3.1 LEAKY INTEGRATOR NEURON MODEL

In biological neurons a classical model for the time evolution of the membrane voltage  $s_i$  is the rate-based leaky integrator neuron model, in which neurons are seen as performing leaky temporal integration of their past inputs Dayan & Abbott (2001):

$$
\frac {d s _ {i}}{d t} = \sum_ {j} W _ {i j} \rho \left(s _ {j}\right) - s _ {i}. \tag {1}
$$

Unlike energy-based models such as the Hopfield model (Hopfield, 1984) that assume symmetric connections between neurons, in the model studied here the connections between neurons are not tied. Thus, our model is described by a directed graph, whereas the Hopfield model is best regarded as an undirected graph (Figure 1).

![](images/2bfe5f3df93d69d65ebea2e02f9b3dbcf00536999b652e9b44dcb6390b03db36.jpg)  
(a) The network model studied here is best represented by a directed graph.

![](images/5b3ffe224a096ed7d7b859557c2b80502f1f835b5aa770cc181eecec89389775.jpg)  
(b) The Hopfield model is best represented by an undirected graph.  
Figure 1: From the point of view of biological plausibility, the symmetry of connections in the Hopfield model is a major drawback (1b). The model that we study here is, like a biological neural network, a directed graph (1a).

# 3.2 SPIKE-TIMING DEPENDENT PLASTICITY

Spike-Timing Dependent Plasticity (STDP) is considered a key mechanism of synaptic change in biological neurons (Markram & Sakmann, 1995; Gerstner et al., 1996; Markram et al., 2012). STDP is often conceived of as a spike-based process which relates the change in the synaptic weight  $W_{ij}$  to the timing difference between postsynaptic spikes (in neuron  $i$ ) and presynaptic spikes (in neuron  $j$ ) (Bi & Poo, 2001). In fact, both experimental and computational work suggest that postsynaptic voltage, not postsynaptic spiking, is more important for driving LTP (Long Term Potentiation) and LTD (Long Term Depression) (Clopath & Gerstner, 2010; Lisman & Spruston, 2010).

Similarly, Bengio et al. (2017) have shown in simulations that a simplified Hebbian update rule based on pre- and post-synaptic activity can functionally reproduce STDP:

$$
d W _ {i j} \propto \rho \left(s _ {j}\right) d s _ {i}. \tag {2}
$$

Throughout this paper we will refer to this update rule (Eq. 2) as "STDP-compatible weight change" and propose a machine learning justification for such an update rule.

# 3.3 VECTOR FIELD  $\mu$  IN THE STATE SPACE

Let  $s = (s_1, s_2, \ldots)$  be the global state variable and parameter  $W$  the matrix of connection weights  $W_{ij}$ . We write  $\mu(W, s)$  the vector whose components are defined as

$$
\mu_ {i} (W, s) := \sum_ {j} W _ {i j} \rho \left(s _ {j}\right) - s _ {i} \tag {3}
$$

defining a vector field over the neurons state space, indicating in which direction each neuron's activity changes:

$$
\frac {d s}{d t} = \mu (W, s). \tag {4}
$$

Since  $\rho(s_j) = \frac{\partial \mu_i}{\partial W_{ij}}(W, s)$ , the weight change Eq. 2 can also be expressed in terms of  $\mu$  in the form  $dW_{ij} \propto \frac{\partial \mu_i}{\partial W_{ij}}(W, s) ds_i$ . Note that for all  $i' \neq i$  we have  $\frac{\partial \mu_{i'}}{\partial W_{ij}} = 0$  since to each synapse  $W_{ij}$  corresponds a unique post-synaptic neuron  $s_i$ . Hence  $dW_{ij} \propto \frac{\partial \mu}{\partial W_{ij}}(W, s) \cdot ds$ . We rewrite the STDP-compatible weight change in the more concise form

$$
d W \propto \frac {\partial \mu}{\partial W} (W, s) \cdot d s. \tag {5}
$$

# 4 A BIOLOGICALLY PLAUSIBLE LEARNING ALGORITHM FOR FIXED POINT RECURRENT NETWORKS WITHOUT TIED WEIGHTS

The framework and the algorithm in their general forms are described in Appendix A.

To illustrate our algorithm, we consider here the supervised setting in which we want to predict an output  $y$  given an input  $x$ . We describe a simple two-phase learning procedure based on the dynamics Eq. 4 and Eq. 5 for the state and the parameter variables. This algorithm is similar to the one proposed by Scellier & Bengio (2017), but here we do not assume symmetric weights between neurons. Note that similar algorithms have also been proposed by O'Reilly (1996); Hertz et al. (1997) or more recently by Mesnard et al. (2016). Our contribution in this work are theoretical insights into why the proposed algorithm works.

# 4.1 TRAINING OBJECTIVE

In the supervised setting studied here, the units of the network are split in two sets: the inputs  $x$  whose values are always clamped, and the dynamically evolving units  $h$  (the neurons activity, indicating the state of the network), which themselves include the hidden layers ( $h_1$  and  $h_2$  here) and an output layer ( $h_0$  here), as in Figure 2. In this context the vector field  $\mu$  is defined by its components  $\mu_0, \mu_1$  and  $\mu_2$  on  $h_0, h_1$  and  $h_2$  respectively, as follows:

$$
\mu_ {0} (W, \mathrm {x}, h) = W _ {0 1} \cdot \rho \left(h _ {1}\right) - h _ {0}, \tag {6}
$$

$$
\mu_ {1} (W, \mathrm {x}, h) = W _ {1 2} \cdot \rho \left(h _ {2}\right) + W _ {1 0} \cdot \rho \left(h _ {0}\right) - h _ {1}, \tag {7}
$$

$$
\mu_ {2} (W, \mathrm {x}, h) = W _ {2 3} \cdot \rho (\mathrm {x}) + W _ {2 1} \cdot \rho \left(h _ {1}\right) - h _ {2}. \tag {8}
$$

Here the scalar function  $\rho$  is applied elementwise to the components of the vectors. The neurons  $h$  follow the dynamics

$$
\frac {d h}{d t} = \mu (W, x, h). \tag {9}
$$

In this section and the next we use the notation  $h$  rather than  $s$  for the state variable.

The layer  $h_0$  plays the role of the output layer where the prediction is read. The target outputs, denoted by  $y$ , have the same dimension as the output layer  $h_0$ . The discrepancy between the output units  $h_0$  and the targets  $y$  is measured by the quadratic cost function

$$
C (h, \mathrm {y}) := \frac {1}{2} \left\| \mathrm {y} - h _ {0} \right\| ^ {2}. \tag {10}
$$

Unlike in the continuous Hopfield model, here the feed-forward and feedback weights are not tied, and in general the state dynamics Eq. 9 is not guaranteed to converge to a fixed point. However we observe experimentally that the dynamics almost always converges. We will see in section 5 that, for a whole set of values of the weight matrix  $W$ , the dynamics of the neurons  $h$  converges. Assuming this condition to hold, the dynamics of the neurons converge to a fixed point which we denote by  $h^0$  (beware not to confuse with the notation for the output units  $h_0$ ). The prediction  $h_0^0$  is then read out on the output layer and compared to the actual target  $y$ . The objective function (for a single training case  $(x,y)$ ) that we aim to minimize is the cost at the fixed point  $h^0$ , which we write

$$
J := C \left(h ^ {0}, \mathrm {y}\right). \tag {11}
$$

Note that this objective function is the same as the one proposed by Almeida (1987); Pineda (1987). Their method to optimize  $J$  is to compute the gradient of  $J$  thanks to an algorithm which they call "Recurrent Backpropagation". Other methods related to Recurrent Backpropagation exist to compute the gradient of  $J$  - in particular the "adjoint method", "implicit differentiation" and "Backprop Through Time". These methods are biologically implausible, as argued in Appendix B.

Here our approach to optimize  $J$  is to give up on computing the true gradient of  $J$  and, instead, we propose a simple algorithm based only on the leaky integrator dynamics (Eq. 4) and the STDP-compatible weight change (Eq. 5). We will show in section 5 that our algorithm computes a proxy for the gradient of  $J$ . Also, note that in its general formulation, our algorithm applies to any vector field  $\mu$  and cost function  $C$  (Appendix A)

# 4.2 EXTENDED DYNAMICS

The idea of Equilibrium Propagation (Scellier & Bengio, 2017) is to see the cost function  $C$  (Eq. 10) as an external potential energy for the output units  $h_0$ , which can drive them towards their target  $y$ . Following the same idea we define the "extended vector field"  $\mu^{\beta}$  as

$$
\mu^ {\beta} := \mu - \beta \frac {\partial C}{\partial h}, \tag {12}
$$

and we redefine the dynamics of the state variable  $h$  as

$$
\frac {d h}{d t} = \mu^ {\beta} (W, x, h, y). \tag {13}
$$

The real-valued scalar  $\beta \geq 0$  controls whether the output  $h_0$  is pushed towards the target y or not, and by how much. We call  $\beta$  the "influence parameter" or "clamping factor".

The differential equation of motion Eq. 13 can be seen as a sum of two "forces" that act on the temporal derivative of the state variable  $h$ . Apart from the vector field  $\mu$  that models the interactions between neurons within the network, an "external force"  $-\beta \frac{\partial C}{\partial h}$  is induced by the external potential  $\beta C$  and acts on the output neurons:

$$
- \beta \frac {\partial C}{\partial h _ {0}} = \beta \left(\mathrm {y} - h _ {0}\right), \tag {14}
$$

$$
- \beta \frac {\partial C}{\partial h _ {i}} = 0, \quad \forall i \geq 1. \tag {15}
$$

The form of Eq. 14 suggests that when  $\beta = 0$ , the output units  $h_0$  are not sensitive to the targets y from the outside world. In this case we say that the network is in the free phase (or first phase). When  $\beta > 0$ , the "external force" drives the output units  $h_0$  towards the target y. When  $\beta \gtrsim 0$  (small positive value), we say that the network is in the weakly clamped phase (or second phase). Also, note that the case  $\beta \to \infty$ , not studied here, would correspond to fully clamped outputs.

# 4.3 TWO-PHASE ALGORITHM AND BACK PROPAGATION OF ERROR SIGNALS

We propose a simple two-phase learning procedure, similar to the one proposed by Scellier & Bengio (2017). In the first phase of training, the inputs are set (clamped) to the input values. The state variable (all the other neurons) follows the dynamics Eq. 9 (or equivalently Eq. 13 with  $\beta = 0$ ) and the output units are free. We call this phase the free phase, as the system relaxes freely towards the free fixed point  $h^0$  without any external constraints on his output neurons. During this phase, the synaptic weights are unchanged.

![](images/8b8c9ff72fcd66f56bea8d60065ea8d484a4e84f5a44da6a656f32064c0425bc.jpg)  
(a) The supervised network studied here has directed connections.

![](images/74264c81b01c5a0b5ef24990d5376fddc5a435784fd67bdbe911bd2d9614db62.jpg)  
(b) In the framework of Equilibrium Propagation with the Hopfield energy, the network is assumed to have symmetric connections.  
Figure 2: Input x is clamped. Neurons  $h$  include "hidden layers"  $h_2$  and  $h_1$ , and "output layer"  $h_0$  that corresponds to the layer where the prediction is read. Target y has the same dimension as  $h_0$ . The clamping factor  $\beta$  scales the "external force"  $-\beta \frac{\partial C}{\partial h}$  that attracts the output  $h_0$  towards the target y.

In the second phase, the influence parameter  $\beta$  takes on a small positive value  $\beta \gtrsim 0$ . The state variable follows the dynamics Eq. 13 for that new value of  $\beta$ , and the synaptic weights follow the STDP-compatible weight change Eq. 5. This phase is referred to as the weakly clamped phase. The novel "external force"  $-\beta \frac{\partial C}{\partial h}$  in the dynamics Eq. 13 acts on the output units and drives them towards their targets (Eq. 14). This force models the observation of y: it nudges the output units  $h_0$  from their free fixed point value in the direction of their targets. Since this force only acts on the output layer  $h_0$ , the other hidden layers ( $h_i$  with  $i > 0$ ) are initially at equilibrium at the beginning of the weakly clamped phase. The perturbation caused at the output layer will then propagate backwards along the layers of the network, giving rise to "back-propagating" error signals. The network eventually settles to a new nearby fixed point, corresponding to the new value  $\beta \gtrsim 0$ , termed weakly clamped fixed point and denoted  $h^\beta$ .

# 4.4 VECTOR FIELD  $\nu$  IN THE WEIGHT SPACE

Our model assumes that the STDP-compatible weight change (Eq. 5) occurs during the second phase of training (weakly clamped phase) when the network's state moves from the free fixed point  $h^0$  to the weakly clamped fixed point  $h^\beta$ . Normalizing by a factor  $\beta$  and letting  $\beta \rightarrow 0$ , we get the update rule  $\Delta W \propto \nu(W)$  for the weights, where  $\nu(W)$  is the vector defined as

$$
\nu (W) := \left. \frac {\partial \mu}{\partial W} \left(W, x, h ^ {0}\right) \cdot \frac {\partial h ^ {\beta}}{\partial \beta} \right| _ {\beta = 0}. \tag {16}
$$

The vector  $\nu(W)$  has the same dimension as  $W$ . Formally  $\nu$  is a vector field in the weight space.

It is shown in section 5 that  $\nu(W)$  is a proxy to the gradient  $\frac{\partial J}{\partial W}$ . The effectiveness of the proposed method is demonstrated through experimental studies (Figure 3).

# 5 THE VECTOR FIELD  $\nu$  AS A PROXY FOR THE GRADIENT

In this section, we attempt to understand why the proposed algorithm is experimentally found to optimize the objective function  $J$  (Figure 3). We say that  $W$  is a "good parameter" if:

1. for any initial state for the neurons, the state dynamics  $\frac{dh}{dt} = \mu (W,\mathrm{x},h)$  converges to a fixed point - a condition required for the algorithm to be correctly defined,  
2. the scalar product  $\frac{\partial J}{\partial W} \cdot \nu(W)$  at the point  $W$  is negative - a desirable condition for the algorithm to optimize the objective function  $J$ .

Experiments show that the dynamics of  $h$  (almost) always converges to a fixed point and that  $J$  consistently decreases (Figure 3). This means that, during training, as the parameter  $W$  follows the update rule  $\Delta W \propto \nu(W)$ , all values of  $W$  that the network takes are "good parameters". In this section we attempt to explain why.

# 5.1 EXPLICIT FORMULAS FOR  $\frac{\partial J}{\partial W}$  AND  $\nu$

Theorem 1. The gradient of  $J$  can be expressed in terms of  $\mu$  and  $C$  as

$$
\frac {\partial J}{\partial W} = - \frac {\partial C}{\partial h} \cdot \left(\frac {\partial \mu}{\partial h}\right) ^ {- 1} \cdot \frac {\partial \mu}{\partial W}. \tag {17}
$$

Similarly, the vector field  $\nu$  (Eq. 16) is equal to

$$
\nu (W) = \frac {\partial C}{\partial h} \cdot \left(\left(\frac {\partial \mu}{\partial h}\right) ^ {T}\right) ^ {- 1} \cdot \frac {\partial \mu}{\partial W}. \tag {18}
$$

In these expressions, all terms are evaluated at the fixed point  $h^0$ .

Theorem 1 is proved in Appendix A. Note that the formulas show that  $\nu(W)$  is related to  $\frac{\partial J}{\partial W}$  and that the angle between these two vectors is directly linked to the "degree of symmetry" of the Jacobian of  $\mu$ .

An important particular case is the setting of Equilibrium Propagation (Scellier & Bengio, 2017), in which the vector field  $\mu$  is a gradient field  $\mu = -\frac{\partial E}{\partial h}$ , meaning that it derives from an energy function  $E$ . In this case the Jacobian of  $\mu$  is symmetric since it is the Hessian of  $E$ . Indeed  $\frac{\partial \mu}{\partial h} = -\frac{\partial^2 E}{\partial h^2} = \left(\frac{\partial \mu}{\partial h}\right)^T$ . Therefore, Theorem 1 shows that  $\nu$  is also a gradient field, namely the gradient of the objective function  $J$ , that is  $\nu = -\frac{\partial J}{\partial W}$ . Note that in this setting the set of "good parameters" is the entire weight space - for all  $W$ , the dynamics  $\frac{dh}{dt} = -\frac{\partial E}{\partial h}(W,h)$  converges to an energy minimum, and  $W$  converges to a minimum of  $J$  since  $\Delta W \propto -\frac{\partial J}{\partial W}$ .

We argue that the set of "good parameters" covers a large proportion of the weight space and that they contain the matrices  $W$  that present a form of symmetry or "alignment". In the next subsection, we discuss how this form of symmetry may arise from the learning procedure itself.

![](images/290fd314ab5de4691fa3ab0672b3c6fd61106ab3263cd5f0745e2cc8b4ec8522.jpg)  
Figure 3: Example system trained on the MNIST dataset, as described in Appendix C. The objective function is optimized: the training error decreases to  $0.00\%$  in around 70 epochs. The generalization error is about  $2\%$ . Right: A form of symmetry or alignment arises between feedforward and feedback weights  $W_{k,k+1}$  and  $W_{k+1,k}$  in the sense that  $tr(\tilde{W}_{k,k+1} \cdot W_{k+1,k}) > 0$ . This architecture uses 3 hidden layers each of dimension 512.

![](images/867a76c03f5dd1ac45c7709486203cf533e4f7f50475ba70f37e629f1f0b37de.jpg)

![](images/7eb970cae88dc8b9397bec75a2b8b2de98b6f11a309f8fb6dcfe06397d809ccb.jpg)

# 5.2 A FORM OF SYMMETRY ARISES

Experiments show that a form of symmetry between feedforward and feedback weights arises from the learning procedure itself (Figure 3). Although the causes for this phenomenon aren't understood very well yet, it is worth pointing out that similar observations have been made in previous work and different settings.

A striking example is the following one. A major argument against the plausibility of backpropagation in feedforward nets is the weight transport problem: the signals sent forward in the network and those sent backward use the same connections. Lillicrap et al. (2016) have observed that, in the backward pass, (back)propagating the error signals through fixed random feedback weights (rather than the transpose of the feedforward weights) does not harm learning. Moreover, the learned feedforward weights  $W_{k,k+1}$  tend to 'align' with the fixed random feedback weights  $W_{k+1,k}$  in the sense that the trace of  $W_{k,k+1} \cdot W_{k+1,k}$  is positive.

Denoising autoencoders without tied weights constitute another example of learning algorithms where a form of symmetry in the weights has been observed as learning goes on (Vincent et al., 2010).

The theoretical result from Arora et al. (2015) also shows that, in a deep generative model, the transpose of the generative weights perform approximate inference. They show that the symmetric solution minimizes the autoencoder reconstruction error between two successive layers of rectifying linear units.

# 6 POSSIBLE IMPLEMENTATION ON ANALOG HARDWARE

Our approach provides a basis for implementing machine learning models in continuous-time systems, while requirements regarding the actual dynamics are reduced to a minimum. This means that the model applies to a large class of physical realizations of vectorfield dynamics, including analog electronic circuits. Implementations of recurrent networks based on analog electronics have been proposed in the past, e.g. Hertz et al. (1997), however, these models typically required circuits and associated dynamics to adhere to an exact theoretical model. With our framework, we provide a way of implementing a learning system on a physical substrate without even knowing the exact dynamics or microscopic mechanisms that give rise to it. Thus, this approach can be used to analog electronic system end-to-end, without having to worry about exact device parameters and inaccuracies, which inevitably exist in any physical system. Instead of approximately implementing idealized computations, the actual analog circuit, with all its individual device variations, is trained to perform the task of interest. Thereby, the more direct implementation of the dynamics might result in advantages in terms of speed, power, and scalability, as compared to digital approaches.

# 7 CONCLUSION

Our model demonstrates that biologically plausible learning in neural networks can be achieved with relatively few assumptions. As a key contribution, in contrast to energy-based approaches such as the Hopfield model, we do not impose any symmetry constraints on the neural connections. Our algorithm assumes two phases, the difference between them being whether synaptic changes occur or not. Although this assumption begs for an explanation, neurophysiological findings suggest that phase-dependent mechanisms are involved in learning and memory consolidation in biological systems. Theta waves, for instance, generate neural oscillatory patterns that can modulate the learning rule or the computation carried out by the network Orr et al. (2001). Furthermore, synaptic plasticity, and neural dynamics in general, are known to be modulated by inhibitory neurons and dopamine release, depending on the presence or absence of a target signal. Frémaux & Gerstner (2016); Pawlak et al. (2010).

In its general formulation (Appendix A), the work presented in this paper is an extension of the framework of Scellier & Bengio (2017) to general dynamics. This is achieved by relaxing the requirement of an energy function. This generalization comes at the cost of not being able to compute the (true) gradient of the objective function but, rather a direction in the weight space which is related to it. Thereby, precision of the approximation of the gradient is directly related to the "alignment" between feedforward and feedback weights. Even though the exact underlying mechanism is not fully understood yet, we observe experimentally that during training the weights symmetrize to some extent, as has been observed previously in a variety of other settings (Lillicrap et al., 2016; Vincent et al., 2010; Arora et al., 2015). Our work shows that optimization of an objective function can be achieved without ever computing the (true) gradient. More thorough theoretical analysis needs to be carried out to understand and characterize the dynamics in the weight space that optimize objective functions. Naturally, the set of all such dynamics is much larger than the tiny subset of gradient-based dynamics.

Our framework provides a means of implementing learning in a variety of physical substrates, whose precise dynamics might not even be known exactly, but which simply have to be in the set of sup-

ported dynamics. In particular, this applies to analog electronic circuits, potentially leading to faster, more efficient, and more compact implementations.

# REFERENCES

D. H. Ackley, G. E. Hinton, and T. J. Sejnowski. A learning algorithm for Boltzmann machines. 9: 147-169, 1985.  
L. B. Almeida. A learning rule for asynchronous perceptrons with feedback in a combinatorial environment. volume 2, pp. 609-618, San Diego 1987, 1987. IEEE, New York.  
Sanjeev Arora, Yingyu Liang, and Tengyu Ma. Why are deep nets reversible: a simple theory, with implications for training. Technical report, arXiv:1511.05653, 2015.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In *ICLR'2015*, arXiv:1409.0473, 2015.  
Yoshua Bengio, Thomas Mesnard, Asja Fischer, Saizheng Zhang, and Yuhuai Wu. STDP-compatible approximation of back-propagation in an energy-based model. Neural Computation, 29(3):555-577, 2017.  
G. Bi and M. Poo. Synaptic modification by correlated activity: Hebb's postulate revisited. Annu. Rev. Neurosci., 24:139-166, 2001.  
Claudia Clopath and Wulfram Gerstner. Voltage and spike timing interact in stdp-a unified model. Frontiers in synaptic neuroscience, 2, 2010.  
Francis Crick. The recent excitement about neural networks. Nature, 337(6203):129-132, 1989.  
Peter Dayan and L. F. Abbott. Theoretical Neuroscience. The MIT Press, 2001.  
John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 2011.  
Nicolas Frémaux and Wulfram Gerstner. Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules. Frontiers in neural circuits, 9:85, 2016.  
W. Gerstner, R. Kempter, J.L. van Hemmen, and H. Wagner. A neuronal learning rule for submillisecond temporal coding. Nature, 386:76-78, 1996.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS'2010, 2010.  
J. A. Hertz, A. Krogh, B. Lautrup, and T. Lehmann. Nonlinear backpropagation: doing backpropagation without derivatives of the activation function. IEEE Transactions on neural networks, 8 (6):1321-1327, 1997.  
Geoffrey Hinton, Li Deng, George E. Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara Sainath, and Brian Kingsbury. Deep neural networks for acoustic modeling in speech recognition. IEEE Signal Processing Magazine, 29(6): 82–97, Nov. 2012.  
Geoffrey E. Hinton and James L. McClelland. Learning representations by recirculation. In D. Z. Anderson (ed.), Neural Information Processing Systems, pp. 358-366. American Institute of Physics, 1988. URL http://papers.nips.cc/paper/78-learning-representations-by-recirculation.pdf.  
J. J. Hopfield. Neurons with graded responses have collective computational properties like those of two-state neurons. 81, 1984.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. ImageNet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25 (NIPS'2012). 2012.

Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. Nature, 521(7553):436-444, 2015.  
Timothy P Lillicrap, Daniel Cownden, Douglas B Tweed, and Colin J Akerman. Random synaptic feedback weights support error backpropagation for deep learning. Nature communications, 7, 2016.  
John Lisman and Nelson Spruston. Questions about stdp as a general model of synaptic plasticity. Frontiers in synaptic neuroscience, 2, 2010.  
H. Markram and B. Sakmann. Action potentials propagating back into dendrites triggers changes in efficacy. Soc. Neurosci. Abs, 21, 1995.  
H. Markram, W. Gerstner, and P.J. Sjstrm. Spike-timing-dependent plasticity: A comprehensive overview. Frontiers in synaptic plasticity, 4(2), 2012.  
Thomas Mesnard, Wulfram Gerstner, and Johanni Brea. Towards deep learning with spiking neurons in energy based models with contrastive hebbian plasticity. arXiv preprint arXiv:1612.03214, 2016.  
Javier R. Movellan. Contrastive Hebbian learning in the continuous Hopfield model. In Proc. 1990 Connectionist Models Summer School, 1990.  
Randall C. O'Reilly. Biologically plausible error-driven learning using local activation differences: The generalized recirculation algorithm. Neural Computation, 8(5):895-938, 1996.  
G Orr, G Rao, FP Houston, BL McNaughton, and Carol A Barnes. Hippocampal synaptic plasticity is modulated by theta rhythm in the fascia dentata of adult and aged freely behaving rats. Hippocampus, 11(6):647-654, 2001.  
Verena Pawlak, Jeffery R Wickens, Alfredo Kirkwood, and Jason ND Kerr. Timing is not everything: neuromodulation opens the stdp gate. Frontiers in synaptic neuroscience, 2, 2010.  
F. J. Pineda. Generalization of back-propagation to recurrent neural networks. 59:2229-2232, 1987.  
Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energy-based models and backpropagation. Frontiers in computational neuroscience, 11, 2017.  
T Tieleman and G Hinton. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 4, 2012.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Machine Learning Res., 11, 2010.
