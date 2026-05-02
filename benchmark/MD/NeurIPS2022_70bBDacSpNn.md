# Operator-Discretized Representation for Temporal Neural Networks

Anonymous Author(s)

Affiliation

Address

email

# Abstract

This paper proposes a new representation of artificial neural networks to efficiently track their temporal dynamics as sequences of operator-discretized events. Our approach takes advantage of diagrammatic notions in category theory and operator algebra, which are known mathematical frameworks to abstract and discretize high-dimensional quantum systems, and adjusts the state space for classical signal activation in neural systems. The states for nonstationary neural signals are prepared at presynaptic systems with ingress creation operators, and are transformed via synaptic weights to attenuated superpositions. The outcomes at postsynaptic systems are observed as the effects with egress annihilation operators (each adjoint to the corresponding creation operator) for efficient coarse-grained detection. The follow-on signals are generated at neurons via individual activation functions for amplitude and timing. The proposed representation attributes the different generations of neural networks, such as analog neural networks (ANNs) and spiking neural networks (SNNs), to the different choices of operators and signal encoding. As a result, temporally-coded SNNs can be emulated at competitive accuracy and throughput by exploiting proven models and toolchains for ANNs.

# 1 Introduction

Modern neural networks are expected to solve demanding AI problems with datastreams in extremely high dimensions. Under widely-available computing infrastructure, the situation is becoming even more challenging, when the neural dynamics for data processing is inherently temporal and online as in the biological systems [1-4]. An appropriate neural network representation for natively handling sequences of timestamped events should significantly improve computational efficiencies. When event sequences are processed with artificial neural networks, known techniques typically compute layer-wise outputs synchronously at every discretized time step to align their data and computing wavefront, as seen in recent investigations on SNNs [5-7] or time series forecasting [8-11]. Though algorithms may sometimes be given in event-driven manners, their execution in SW has to resort to fine-grained synchronous discretization [12-15] or closed-form approximations of temporal dynamics that require exact temporal ordering of the events [14, 16]. As a result, accuracies competitive to ANNs have only been obtained at an expense of throughput and scalability.

In temporally executing neural networks in commercial systems, the period  $T_{c}$  of the global clock is typically chosen small enough compared with the characteristic time of the neural dynamics  $t_0$ :

$$
T _ {c} \ll t _ {0}, \tag {1}
$$

to precisely track the temporal dynamics, for example, the membrane potential changes to determine the next firing timing of SNNs. This is a sharp contrast to how the biological brain operates with low-frequency brain waves [17] closer to our behavioral time scale:

$$
T _ {c} \gg t _ {0}. \tag {2}
$$

Energy and functional efficiencies can be significantly improved if a new representation can avoid synchronously computing the temporal dynamics at every small time step by better decoupling different time scales. It is tempting for those with some physics background to apply techniques being developed for quantum systems since they are naturally asynchronous events in extremely high dimensions. Indeed, operator algebra has been applied to Hopfield networks [18] as well as other classical systems [19-21]. However, since operators are used for stationary neuron states out of spins and charges rather than those for nonstationary neural signals traveling over axon-synapse-dendrite networks, its full potential has not been extracted for modern temporal workloads.

Here in this paper, we propose a new representation of neural networks that can efficiently compute their dynamics as coarse-grained sequences of operator-discretized events. Our approach takes advantage of existing mathematical frameworks that have been originally developed to abstract and discretize high-dimensional quantum systems. These techniques are, with necessary modifications, applied to neural networks that are also high dimensional, but inherently are classical. Different generations of neural networks, such as ANNs and SNNs, are attributed to different choices of operators and signal encoding. Our formulation can efficiently emulate temporally-coded SNNs with fully exploiting existing assets, such as models and toolchains for ANNs. It should be noted that the scope of this paper is on classical neural networks, though the proposed representation may bring us a new perspective on AI and quantum computing (QC) [22],

# 2 Logical representation

Let us start with the logical aspects. Figure 1 presents diagrammatic representations for quantum and neural networks. In short, once the state spaces are respectively defined, they look surprisingly similar, in particular when we regard qubits as nonstationary and flying [23] as well.

# 2.1 Logical abstraction and state space

The operation of neural networks is to be abstracted by exploiting diagrammatic notions of categorical theories [24-26]. These techniques have been applied both to quantum and classical systems and their processes without much referring to actual physics inside [27]. Here, we will consider pure states only (i.e., wave function vectors rather than density matrices) for quantum, since our purpose is to explicitly compare quantum and classical networks.

A known categorical diagram for a quantum network is exemplified in Fig. 1 (a). It consists of three major blocks: the states, the processes/transformations, and the effect, for preparation, operation, and observation of quantum systems, respectively. Without operation, the inner product of the state  $|\rho \rangle$  represented by a tensor product of each qubit  $|\rho_i\rangle$  state prepared at quantum system  $S_{i}^{Q}$

$$
| \rho \rangle = | \rho_ {1} \rangle \otimes \dots . \otimes | \rho_ {n} \rangle . \tag {3}
$$

and the effect  $\langle \alpha |$  represented by a tensor product of each effect  $\langle \alpha_i|$  at quantum system  $\mathcal{R}_i^Q$

$$
\langle \alpha | = \left\langle \alpha_ {1} \right| \otimes \dots \otimes \left\langle \alpha_ {n} \right|, \tag {4}
$$

can compute the conditional probability  $P(\alpha|\rho)$  as

$$
\left| \langle \alpha | \rho \rangle \right| ^ {2} = \prod_ {i = 1} ^ {n} \left| \langle \alpha_ {i} | \rho_ {i} \rangle \right| ^ {2} = \prod_ {i = 1} ^ {n} P \left(\alpha_ {i} | \rho_ {i}\right) = P (\alpha | \rho). \tag {5}
$$

In general, the probabilities cannot be factorized this way other than for the slices, providing a rich set of non-classical computing power, such as with entanglement, to quantum networks.

The corresponding diagram for a classical neural network is proposed in Fig.1 (b). The states for neural signals are prepared at presynaptic systems. They are transformed into weighted sums via synaptic networks. The outcomes are observed at postsynaptic systems as the effects to generate the follow-on states and signals. As is the quantum case, we assume that the transformations in axonsynapse-dendrite networks are linear. We define, in analogy to the qubit, the cubit, which stands for the abbreviation of classical universal bit, for neural signals. Though the definition is informational, rather than physical, we inherit Dirac notation but with double bras and kets, indicating that the

![](images/8612ae64f134e0065a6e0e9bae803008a19790928a9831eea633d8fa979568a0.jpg)

![](images/891e3e60e3c63f59c72d6d13d58d605203650b561f09ce012bb429f726f6f87b.jpg)  
Figure 1: Diagramatic comparison of quantum and neural networks: (a) Quantum network consisting of states, processes/transformations, and effects; (b) Corresponding diagram for a neural network; (c) Operator representation for creation, scattering, and annihilation of quantum wave packets; (d) Operator representation for creation, weighted sum, and annihilation of neural signals. Note that weight matrix  $w_{ij}$  in (d) corresponds to scattering matrix  $s_{ij}$  in (c).

![](images/807fb38aeed32215343ac52f8010c60da6d7390208d07368e7af95ba4f84001e.jpg)

![](images/0f3388e47db81e2664e089c94b5917b03f6b7bbab92a469f3473e68bf4e7dcc7.jpg)

states consist of macroscopic ensembles of qubits  ${}^{1}$  . Multiple types of logical cubits are defined:

$$
\begin{array}{l l l} \text {N o r m a l i z e d f u l l c u b i t} & | | c \rangle \rangle := \bar {c} | | 0 \rangle \rangle + c | | 1 \rangle \rangle , | \bar {c} | ^ {2} + | c | ^ {2} = 1 & \in U (1) \text {o r} S O (2) \\ \text {N o r m a l i z e d h a l f c u b i t} & | | c \rangle \rangle := c | | 1 \rangle \rangle , 0 \leq | c | ^ {2} \leq 1 & \in U (1) \cap \mathrm {R} \\ \text {U n n o r m a l i z e d f u l l c u b i t} & | | c \rangle \rangle := \bar {c} | | 0 \rangle \rangle + c | | 1 \rangle \rangle & \in \mathrm {R} ^ {2} \text {o r} \mathrm {C} \\ \text {U n n o r m a l i z e d h a l f c u b i t .} & | | c \rangle \rangle := c | | 1 \rangle \rangle & \in \mathrm {R} \end{array}
$$

The information encoded to cubits is assumed to be real for simplicity but can be complex for complex-valued neural networks [28].  
A set of cubits  $||\rho \rangle \rangle$  can compactly be represented by Cartesian product (or coproduct in category theory terminology) of each cubit  $||\rho_i\rangle \rangle$  at axiom  $S_{i}^{C}$  as

$$
\left. \left| | \rho \rangle \right\rangle = \left| | \rho_ {1} \rangle \right\rangle \oplus \dots . \oplus \left| | \rho_ {n} \rangle \right\rangle . \right. \tag {7}
$$

They are to be detected by effect  $\langle \langle \alpha ||$  consisting of  $\langle \langle \alpha_i||$  via dendrite  $R_{i}^{C}$  as:

$$
\langle \langle \alpha | | = \langle \langle \alpha_ {1} | | \oplus \dots \oplus \langle \langle \alpha_ {n} | |. \tag {8}
$$

Based on an argument for the linear systems in [29], the norm  $p$  for cubits is expected to be either 1 or 2, Euclidean norm  $(p = 2)$ , which is also found in wireless communication and signal processing literature [30] (e.g.,  $||1\rangle \rangle$  and  $||0\rangle \rangle$  for I and Q), makes sense to represent wave-like dynamics [31-34] in complex-valued state spaces, while Manhattan norm  $(p = 1)$  is for ordinary real-valued state spaces typically assumed for classical probabilistic computing [29]. Under the linear weighted sum transformations in Cartesian-product state spaces, the log encoding [35] can consistently relate the summation of the inner product for each cubit to the multiplication of the corresponding probabilities for the product event via bias thresholds  $P_{i}$ 's and  $P_{total} = \prod_{i=1}^{n} P_{i}$  as

$$
\left| \langle \langle \alpha | | \rho \rangle \right\rangle | ^ {p} = \sum_ {i = 1} ^ {n} \left| \langle \langle \alpha_ {i} | | \rho_ {i} \rangle \right\rangle | ^ {p} \sim \sum_ {i = 1} ^ {n} \log \frac {P \left(\alpha_ {i} \mid \rho_ {i}\right)}{P _ {i}} = \log \prod_ {i = 1} ^ {n} \frac {P \left(\alpha_ {i} \mid \rho_ {i}\right)}{P _ {i}} = \log \frac {P (\alpha | \rho)}{P _ {\text {t o t a l}}}. \tag {9}
$$

# 92 2.2 Operators as neural computing primitives

Operator algebra is a well-established technique to systematically compute quantum physics problems in high-dimensional tensor-product spaces (or Fock for indistinguishable particles). Interac

tions between states are represented by scattering matrices (S-matrices) [36] as exemplified in Fig. 1 (c), Here, we develop an operator formalism in Cartesian-product state spaces for neural networks in Fig. 1 (d).  
98 A neural signal at  $S_{i}^{C}$  is selectively activated in the entire state space spanned as,

$$
\left| \left| 0 \right\rangle \right\rangle = \left| \left| 0 _ {1} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 0 _ {n} \right\rangle \right\rangle \text {a n d} \left| \left| 1 _ {i} \right\rangle \right\rangle = \left| \left| 0 _ {1} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 1 _ {i} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 0 _ {n} \right\rangle \right\rangle . \tag {10}
$$

9 States for concurrently activating multiple neural signals can be given, by specifically noting the activated systems  $i$  and  $j$  as

$$
\left| \left| 1 _ {i j} \right\rangle \right\rangle = \left| \left| 0 _ {1} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 1 _ {i} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 1 _ {j} \right\rangle \right\rangle \oplus \dots \oplus \left| \left| 0 _ {n} \right\rangle \right\rangle . \tag {11}
$$

Thus,  $||1_i\rangle \rangle$  can mean a single cubit state for  $S_{i}^{C}$  only or a multiple cubit state in which only  $S_{i}^{C}$  is fully activated, depending on the context.

103 The mutually-adjoint creation and annihilation operators on these states, a and  $\mathbf{a}^{\dagger}$  are defined as

$$
\left| \left| 1 _ {i} \right\rangle \right\rangle = \mathrm {a} _ {i} ^ {\dagger} \left| \left| 0 \right\rangle \right\rangle \text {a n d} \left| \left| 0 \right\rangle \right\rangle = \mathrm {a} _ {i} \left| \left| 1 _ {i} \right\rangle \right\rangle . \tag {12}
$$

104 Multiple signals can be activated in different systems, for example, by

$$
\left| \left| 1 _ {i j} \right\rangle \right\rangle = \mathrm {a} _ {i} ^ {\dagger} \mathrm {a} _ {j} ^ {\dagger} | | 0 \rangle \rangle . \tag {13}
$$

Depending on whether  $i = j$  is allowed in each  $T_{c}$  or not, they are superficially treated like Bosons for rate-coded SNNs (rSNNs) or like Fermions for temporally-coded SNNs (tSNNs).

107 The transformation  $\mathcal{T}_{ij}$  from sender system  $S_{j}$  to receiver system  $\mathcal{R}_i$  is described as:

$$
\mathcal {T} _ {i j} = w _ {i j} \mathrm {a} _ {i} ^ {\dagger} \mathrm {a} _ {j}. \tag {14}
$$

Noted that  $w_{ij}$  works as the scattering matrix. Cartesian product state space, rather than tensorproduct, can incorporate the weighted sum naturally as the superposition of incoming neural signalsfrom different sources. Higher-order interactions are possible, for example as,

$$
\mathcal {T} _ {i j} = w _ {i j} \check {\mathrm {a}} _ {i} ^ {\dagger} \hat {\mathrm {a}} _ {i} \check {\mathrm {a}} _ {j} ^ {\dagger} \hat {\mathrm {a}} _ {j}. \tag {15}
$$

However, in that case our original assumption of linear synaptic networks is not valid anymore.

The logical neuron model in the operator representation is defined as effects for detecting incoming fragment of signal energies from presynaptic neurons to generate states for the follow-on neural signals. The signal detection process corresponds to the projective measurement in QC, leading to more advanced detection strategies than simple threshold detection strategies. When the fully activated state  $||\rho_j\rangle \rangle = \mathrm{a}_j^\dagger ||0\rangle \rangle$  is detected by the effect  $\langle \langle \alpha_i|| = \langle \langle 0||\mathrm{a}_i$  via  $\mathcal{T}_{ij}$

$$
\left| \langle \left\langle \alpha_ {i} \right| \right| \mathcal {T} _ {i j} \left| \left| \rho_ {j} \right\rangle \right\rangle | ^ {p} = \left| \langle \left\langle 0 \right| \right| b _ {i} \left(w _ {i j} b _ {i} ^ {\dagger} a _ {j}\right) a _ {j} ^ {\dagger} | \left| 0 \right\rangle \rangle | ^ {p} = \left| w _ {i j} \right| ^ {p} = \log P (\alpha_ {i} | \rho_ {j}). \tag {16}
$$

117 Nonlinear binary operations such as AND/OR are possible using appropriate activation functions with different thresholds, as those in perceptrons [37].

# 3 Physical representation

The proposed physical representation of neural networks is outlined in Fig. 2. It introduces explicit temporal dependences for operators and neural signals The operators for ingress and egress paths create and annihilate nonstationary neural signals over elastic physical media, i.e., axons  $(S_{i}^{C},\mathrm{s})$  and dendrites  $(\mathcal{R}_i^C,\mathrm{s})$

# 124 3.1 Operators for eigenmodes

First, the physical representation of the creation and annihilation operators for stationary neural signals  $\mathbf{a}_i^\dagger$  and  $\mathbf{a}_i$  are constructed in accordance with the quantum creation and annihilation operators  $a_i^\dagger$  and  $a_i$  in the one-dimensional transmission line (TL) model in circuit QED [38]. Circuit QED is one of the established baseline theories in QC, which bridges classical circuit dynamics and quantum. The Hamiltonian  $\mathcal{H}_{ij}$  for a TL creating consisting of  $N$  identical capacitors of the capacitance  $C_0$

![](images/c005dfc89313384400b3759341b76220f359b0da1ac144a444de46bb8c44c2a4.jpg)  
Figure 2: Physical representation of operator-discretized neural networks with explicit local time  $t$  dependency with respect to global time  $T$ . The creation and annihilation operators for ingress and egress paths represent nonstationary neural signal dynamics across axon-synapse-dendrite networks. LC TL models are used for axons and dendrites instead of RC cable models. The neuron model consists of different activation functions for signal amplitude and timing,

(each containing the charge  $Q_{n}$ ) and  $N$  identical inductors of the inductance  $L_{0}$  (each containing the flux  $\Phi_{n} - \Phi_{n-1}$ ), is given by

$$
\mathcal {H} _ {i} = \sum_ {n} \left[ \frac {1}{2 C _ {0}} Q _ {n} ^ {2} + \frac {1}{2 L _ {0}} \left(\Phi_ {n} - \Phi_ {n - 1}\right) ^ {2} \right] = \sum_ {m} \hbar \omega_ {m} a _ {i} ^ {\dagger} \left(k _ {m}, \omega_ {m}\right) a _ {i} \left(k _ {m}, \omega_ {m}\right), \tag {17}
$$

where  $m$  is the eigen mode index for a given boundary condition. The lossless LC-based model can better transmit energy and information than the dissipative RC-based biological cable model [39]

We define  $\mathrm{a}_i^\dagger(k, \omega)$  and  $\mathrm{a}_i(k, \omega)$  as the classical counterpart of  $a_i^\dagger(k, \omega)$  and  $a_i(k, \omega)$ . The following simple linear dispersion for a constant velocity  $v$  are assumed in the range of interest:

$$
v = \frac {\partial \omega_ {m}}{\partial k _ {m}} = \frac {\omega_ {m}}{k _ {m}} = \text {c o n s t .} \forall m. \tag {18}
$$

Consequently,  $\mathrm{a}_i^\dagger (k.\omega) = \mathrm{a}_i^\dagger (\omega)$ ,  $\mathrm{a}_i(k,\omega) = \mathrm{a}_i(\omega)$ . Note that  $v$  for neural signals is much slower than  $v$  for electrical signals in ordinary TL's [31, 32]. Though our focus is on artificial neural networks, biological implications of the present approach will be further discussed in Appendix.

# 3.2 Operators for nonstationary neural signals

Second, the operators basis is changed from  $(k,w)$  to  $(x,t)$ . For ingress signals

$$
\hat {\mathbf {a}} _ {i} ^ {\dagger} (x, t) = \sum_ {m} \mathbf {a} _ {i} ^ {\dagger} (k _ {m}, \omega_ {m}) \mathbf {A} ^ {*} (k _ {m}, \omega_ {m}) e ^ {- i (k _ {m} x - \omega_ {m} t)},
$$

$$
\hat {\mathrm {a}} _ {i} (x, t) = \sum_ {m} ^ {m} \mathrm {a} _ {i} \left(k _ {m}, \omega_ {m}\right) \mathrm {A} \left(k _ {m}, \omega_ {m}\right) e ^ {i \left(k _ {m} x - \omega_ {m} t\right)}. \tag {19}
$$

For egress signals

$$
\check {\mathrm {a}} _ {i} ^ {\dagger} (x, t) = \sum_ {m} \mathrm {a} _ {i} ^ {\dagger} (k _ {m}, \omega_ {m}) \mathrm {A} ^ {*} (k _ {m}, \omega_ {m}) e ^ {i (k _ {m} x - \omega_ {m} t)},
$$

$$
\check {\mathrm {a}} _ {i} (x, t) = \sum_ {m} ^ {m} \mathrm {a} _ {i} \left(k _ {m}, \omega_ {m}\right) \mathrm {A} \left(k _ {m}, \omega_ {m}\right) e ^ {- i \left(k _ {m} x - \omega_ {m} t\right)}. \tag {20}
$$

They represent creation and annihilation of neural signals centered at  $x = 0$ , and  $t = 0$ , and sent or received at neuron  $i$ . To be more specific, for example, a neural signal moving out of neuron  $i$  created at the start of the TL of a length  $l$  is given as

$$
\hat {\mathrm {a}} _ {i} ^ {\dagger} (t) \left\| 0 \right\rangle \rangle = \hat {\mathrm {a}} _ {i} ^ {\dagger} (0, t) \left\| 0 \right\rangle \rangle . \tag {21}
$$

It annihilates at the end of the TL after the geometrically-defined delay  $d = l / v < T_{c}$  as

$$
\hat {\mathrm {a}} _ {i} (t - d) \hat {\mathrm {a}} _ {i} ^ {\dagger} (t) | | 0 \rangle \rangle = \hat {\mathrm {a}} _ {i} (l, t - d) \hat {\mathrm {a}} _ {i} ^ {\dagger} (0, t) | | 0 \rangle \rangle . \tag {22}
$$

# 3.3 Incorporating physical interaction at synapses

When multiple neurons are interconnected via synaptic networks, physical interactions with explicit temporal dependences should be incorporated in addition to the free dynamics described above. We consider here primarily  $\mathcal{T}_{ij}$  one-body potential scattering via an elastic scattering center as

$$
\mathcal {T} _ {i j} = w _ {i j} \check {\mathrm {a}} _ {i} ^ {\dagger} \left(t - T _ {i} + d _ {i j} ^ {d e n d}\right) \hat {\mathrm {a}} _ {j} \left(t - t _ {j} - d _ {i j} ^ {a x o n}\right), \tag {23}
$$

where  $d_{ij}^{axon}$  and  $d_{ij}^{dend}$  are the delays in axon and dendrite between neurons  $i$  and  $j$ , respectively.

# 3.4 Neuron model with activation functions for amplitude and timing

The proposed representation of neural networks allows for more advanced detection strategies than threshold detection, for example, in LIF neurons usually found in the literature [39] s. This is somewhat inspired by the advancement in detection strategies in communication or storage channels [40]. Let us first consider a simple case when a half-cubit neural signal of the peak amplitude  $x_{j}$  from a presynaptic neuron  $j$  is generated at  $t = t_{j}$  by applying a creation operator as

$$
\left. \left| \left| \rho_ {j} (t) \right\rangle \right\rangle = x _ {j} \hat {\mathrm {a}} _ {j} ^ {\dagger} \left(t - t _ {j}\right) \left| \left| 0 \right\rangle \right\rangle , \right. \tag {24}
$$

and observed by a postsynaptic neuron  $i$  at  $T_{i}$  directly without a synapse.

$$
\langle \langle \alpha_ {i} (t) | | = \langle \langle 0 | | \check {\mathrm {a}} _ {i} (t - T _ {i}), \tag {25}
$$

In general, the state preparation  $||\rho_{j}(t)\rangle \rangle$  at  $t_j$  and the observation  $\langle \langle \alpha_i(t)||$  at  $T_{i}$  are not temporally aligned, so by using ingress-egrss correlation function  $f(\Delta t_{ij})\coloneqq \langle \langle 0||\check{\mathsf{a}}_i(t - \Delta t)\hat{\mathsf{a}}_j^\dagger (t)||0\rangle \rangle$

$$
\langle \langle \alpha_ {i} (t) | | \rho_ {j} (t) \rangle \rangle = \langle \langle 0 | | \check {\mathrm {a}} _ {i} (t - T _ {i}) x _ {j} \hat {\mathrm {a}} _ {j} ^ {\dagger} (t - t _ {j}) | | 0 \rangle \rangle = x _ {j} f \left(t _ {j} + d _ {i j} - T _ {i}\right) \tag {26}
$$

for  $t_j + d_{ij} - T_i \geq 0$ , where  $d_{ij} = d_{ij}^{axon} + d_{ij}^{dend}$ . We should note that for  $\Delta t_1 = \Delta t_2 + \Delta t_3$

$$
f \left(\Delta t _ {1}\right) = f \left(\Delta t _ {2}\right) f \left(\Delta t _ {3}\right), \quad f (0) = 1. \tag {27}
$$

With interactions at synapses, the state preparation and observation between neurons pair  $i$  and  $j$  provides

$$
\langle \langle \alpha_ {i} (t) | | \mathcal {T} _ {i j} | | \rho_ {j} (t) \rangle \rangle = \langle \langle 0 | | \check {\mathrm {a}} _ {i} (t - T _ {i}) \mathcal {T} _ {i j} x _ {j} \hat {\mathrm {a}} _ {j} ^ {\dagger} (t - t _ {j}) | | 0 \rangle \rangle = w _ {i j} x _ {j} f \left(t _ {j} + d _ {i j} - T _ {i}\right). \tag {28}
$$

Thus, the aggregated signal detected at neuron  $i$  is

$$
\sum_ {j} \langle \langle \alpha_ {i} (t) | | \mathcal {T} _ {i j} | | \rho_ {j} (t) \rangle \rangle = \sum_ {j} \langle \langle 0 | | \check {\mathrm {a}} _ {i} (t - T _ {i}) \mathcal {T} _ {i j} x _ {j} \hat {\mathrm {a}} _ {j} ^ {\dagger} (t - t _ {j}) | | 0 \rangle \rangle = \sum_ {j} w _ {i j} x _ {j} f \left(t _ {j} + d _ {i j} - T _ {i}\right). \tag {29}
$$

This inner-product-based detection in neural systems corresponds to the projection measurement in quantum systems and is the key to enable efficient coarse-grained detection without tracking the membrane potential at fine-grained time steps. For a given waveform defined by creation and annihilation operators,  $f(\Delta t_{ij})$  can extract temporally-coded information. Alternatively, the right operator pair can be defined to meet a given  $f(\Delta t_{ij})$ . The latter approach is to be taken when applying the present idea to efficient emulation of temporally-coded SNNs.

By using appropriate activation functions  $\sigma_{1}$  and  $\sigma_{2}$  for the amplitude and the event firing time, respectively, the detected signal can be converted to the follow-on signal in neuron  $i$  as

$$
x _ {i} = \sigma_ {1} \left(\sum_ {j} w _ {i j} x _ {j} f \left(t _ {j} + d _ {i j} - T _ {i}\right)\right), \quad t _ {i} = T _ {i} + \sigma_ {2} \left(\sum_ {j} w _ {i j} x _ {j} f \left(t _ {j} + d _ {i j} - T _ {i}\right)\right). \tag {30}
$$

Various nonlienarities can be incorporated via  $\sigma_{1}$  and  $\sigma_{2}$  if necessary.

# 3.5 Learning algorithms with operators

The weight update  $\Delta w_{ij}$  for unsupervised algorithms, such as Hebbian and STDP for SNNs, is asynchronously (i.e., without explicit dependency on  $T_{i}$ ) related to ingress-ingress correlation  $g$  as

$$
\begin{array}{l} \Delta w _ {i j} \sim \langle \langle 0 | | \hat {\mathbf {a}} _ {i} (t - t _ {i}) \hat {\mathbf {a}} _ {j} ^ {\dagger} (t - t _ {j} - d _ {i j}) | | 0 \rangle \rangle \\ = \langle \langle 0 | | \hat {a} _ {i} (t - t _ {i}) \check {a} _ {i} ^ {\dagger} (t - T _ {i}) \check {a} _ {i} (t - T _ {i}) \hat {a} _ {j} ^ {\dagger} (t - t _ {j} - d _ {i j}) | | 0 \rangle \rangle \tag {31} \\ = g \left(t _ {i} - T _ {i}\right) g \left(T _ {i} - t _ {j} - d _ {i j}\right) = g \left(t _ {i} - t _ {j} - d _ {i j}\right). \\ \end{array}
$$

Even and odd functions are chosen for Hebbian and STDP, respectively.

The proposed representation can support various supervised learning algorithms and toolchains, when temporal dynamics is synchronously regulated by a coarse-grain global clock in  $n$  cycles as

$$
T _ {i} ^ {(n)} = n T _ {c} \quad \forall i. \tag {32}
$$

Fine-grained temporal correlations, such as coincidence, can be passed on to the operator correlations by defining a new global variable  $X_{i}^{(n)} = x_{i}^{(n)}f(t_{i}^{(n)})$ . Then

$$
x _ {i} ^ {(n + 1)} = \sigma_ {1} \left(\sum_ {j} w _ {i j} X _ {j} ^ {(n)}\right), \quad t _ {i} ^ {(n + 1)} = T _ {i} ^ {(n + 1)} + \sigma_ {2} \left(\sum_ {j} w _ {i j} X _ {j} ^ {(n)}\right). \tag {33}
$$

The backward calculation can be performed by using the following relation:

$$
\frac {\partial X _ {i} ^ {(n + 1)}}{\partial X _ {j} ^ {(n)}} = \frac {\partial X _ {i} ^ {(n + 1)}}{\partial x _ {i} ^ {(n + 1)}} \frac {\partial x _ {i} ^ {(n + 1)}}{\partial X _ {j} ^ {(n)}} + \frac {\partial X _ {i} ^ {(n + 1)}}{\partial t _ {i} ^ {(n + 1)}} \frac {\partial t _ {i} ^ {(n + 1)}}{\partial X _ {j} ^ {(n)}} = \left(f \left(t _ {i} ^ {(n + 1)} \sigma_ {1} ^ {\prime} + x _ {i} ^ {(n + 1)} f ^ {\prime} \sigma_ {2} ^ {\prime}\right) w _ {i j} \right. \tag {34}
$$

Let us go through how this works further with a specific example in the next section.

# 4 Application to temporally-coded SNN

The relation between ANNs and rate-coded SNNs (rSNNs) has been known [41]. Here, we first theoretically prove that under the proposed representation, temporally-coded SNNs (tSNNs) can be equivalently transformed into ANNs by appropriately assigning the operator via  $f$  and encoding via  $\sigma_{1}$  and  $\sigma_{2}$ , Then we demonstrate practical benefits of doing so by running some benchmarks.

# 4.1 New perspective on ANN-SNN equivalence

Proposition 1: When driven by a global clock of  $T_{i}^{(n)} = nT_{c}$ , operator-descritized neural networks defined by Eqs. 28 and 30 for the neural events  $(x_{i},t_{i})$  with the following setting constitute ANNs.

$$
\text {A N N}: \sigma_ {1} (x) = *, \sigma_ {2} = 0, \text {a n d} f (x) = 1. \tag {35}
$$

The neural signals stay constant at  $X_{i}^{(n)} = x_{i}^{(n)}$  for  $T_{i}^{(n)} = nT_{c}$ . The operators become arbitrarily picked single-mode  $(k,\omega)$  ones. Perceptrons are constructed with binary inputs and Heaviside step function for  $\sigma_{1}$ .

Proposition 2: When driven by a global clock of  $T_{i}^{(n)} = nT_{c}$ , operator-descritized neural networks defined by Eqs. 28 and 30 for the neural events  $(x_{i},t_{i})$  with the following setting constitute tSNNs.

$$
\operatorname {t S N N}: \sigma_ {1} (x) = 1 \text {a n d} \sigma_ {2} (x) = *. \tag {36}
$$

The tSNN signals for  $X_{i}^{(n)} = f(t_{i}^{(n)} - T_{i}^{(n)})$  take specific spike waveforms defined by nonstationary operators which spread into multiple modes in the  $(k,\omega)$  basis. The cut-off  $X_{min}$  is defined as

$$
T _ {j} ^ {(n)} \leq t _ {j} ^ {(n)} \leq T _ {j} ^ {(n)} + T _ {c} \Leftrightarrow 1 \geq X _ {j} ^ {(n)} \geq X _ {\min } = f \left(T _ {c}\right). \tag {37}
$$

Theorem 1: tSNN in Proposition 2 with  $f'(x)\sigma_2'(x) = 1$  runs equivalently in forward and backward to ANN in Proposition 1 with  $\sigma_1(x) = x \cdot (x > X_{\min})$  for  $X_{\min} = f(T_c) > 0$ .

![](images/34b3627fcf5d2900eb48e0310c21003ab07815a7f3f755e9e12425370fa1ae41.jpg)  
(a)

![](images/0b6de4048a34846efa71bdc8a02b2a6ee675363a012a3f7799abc9733b506a22.jpg)  
(b)  
(c)

![](images/5410ad43c40249e8142fb738aca4fdd8e399d0b4db3f681e3330bd71939e2b2e.jpg)  
Figure 3: (a) Activation function for operator-discretized tSNNs with excitatory and inhibitory neurons; (b) MNIST benchmark results for ANN, operator-discretized tSNN, and Euler-discretized tSNN; (c) Realtime comparison of the number of neural signals and throughput.  
Theeeppea (a.u.)

Proof. In forward, weighted sum of tSNN reduces to that of ANN as

$$
\left(x _ {j}\right) _ {A N N} = \left(f \left(t _ {j} ^ {(n)} - T _ {j}\right)\right) _ {S N N}, \quad \text {a n d} \left(w _ {i j}\right) _ {A N N} = \left(w _ {i j} f \left(- T _ {c} + d _ {i j}\right)\right) _ {t S N N}. \tag {38}
$$

201 This is because for tSNN,

$$
\sum_ {j} w _ {i j} f \left(t _ {j} ^ {(n)} + d _ {i j} - T _ {i} ^ {(n + 1)}\right) = \sum_ {j} w _ {i j} f \left(- T _ {c} + d _ {i j}\right) f \left(t _ {j} ^ {(n)} - T _ {j} ^ {(n)}\right) = \sum_ {j} w _ {i j} f \left(- T _ {c} + d _ {i j} f \left(t _ {j} ^ {(n)} - T _ {i}\right). \right. \tag {39}
$$

202 In backward,

$$
\left(\frac {\partial X _ {i} ^ {(n + 1)}}{\partial X _ {j} ^ {(n)}}\right) _ {A N N} = \left(w _ {i j}\right) _ {A N N} = \left(w _ {i j} f \left(- T _ {c} + d _ {i j}\right)\right) _ {t S N N} = \left(\frac {\partial X _ {i} ^ {(n + 1)}}{\partial X _ {j} ^ {(n)}} f \left(- T _ {c} + d _ {i j}\right)\right) _ {t S N N}. \tag {40}
$$

Thus we can emulate tSNN using ANN by renormalizing  $w_{ij}$  with the constant  $f(-T_c + d_{ij})$ .

Example 1: We can set tSNN as

$$
f (x) = \beta^ {- x}, \sigma_ {2} (x) = - \log_ {\beta} x \text {a n d} T _ {c} = d _ {i j} (\text {i . e .}, f (- T _ {c} + d _ {i j}) = 1) \tag {41}
$$

$\beta$  works as a base constant to carry or borrow across a fine-grained unit time interval. The logarithmic conversion works as a ReLU activation function in ANN since the conversion is only valid for  $X_{min} > 0$ . Bipolar neural signals are represented by combining excitatory and inhibitory neurons as shown in Fig.3(a). This setting can also support rSNNs by allowing multiple spikes within  $T_{c}$ .

Building blocks in modern ANN models, such as convolution, max pooling, and batch normalization, have to be translated to those in SNNs. The translation is straightforward as long as they are linear transformations. However, batch normalization blocks may require some attention, since they involve nonlinear operations to control both the number and the delay distribution of neural signals.

Once the translations of building blocks are completed, the proposed representation for SNNs can support not only specific models and learning algorithms but a wide variety of them. Under the operator-discretized representation, the inference paths of SNNs can be translated to those of the corresponding ANNs. Thus the standard autograd learning strategy [42] for ANNs equally works without using costly strategies specific to SNNs. The instability associated with differentiating the spike activation function can be avoided by substituting adjoint computation [43] to the operators rather than using arbitrary surrogate functions [6, 44].

# 4.2 Evaluation

Figure 3(b) compares MNIST benchmark results for ANN, Euler-discretized tSNN, and operator-discretized tSNN. We used a stand-alone computing environment without GPU to minimize undesired throughput variations. The code for ANN straightforwardly follows reference implementations

![](images/e46ca83704e2ba04398791c39959889fbd894fb71a8108a35f51d68456fb5152.jpg)  
Figure 4: Relative test accuracy and throughput as a function of  $X_{min}$  for CIFAR10&100 with resnet18 : (a) With batch normalization; (b) Without batch normalization.  $X_{min} = 0$  is for ANN.

![](images/8226a289d2317181ca07305e2a5da1c410246fe1c9b051dc43fdaa9c85af2900.jpg)

and default parameter settings under python 3.8.5 and PyTorch 1.9.1.  $lr = 0.001$  with Adam optimizer and is multiplied by 0.9 after every 10 epochs. To accommodate Euler-discretized tSNN, simple architecture of 784-350-10 is chosen. The Euler discretization algorithm follows the one in [12]. There, forward and backward paths were calculated manually in  $30\Delta T$  steps in each  $T_{c}$  period. On the other hand., operator-discretized SNN fully takes advantage of the existing toolchain capabilities of ANN, including autograd. For operator-discretized tSNN, we used the conversion as stated in Example 1 with  $X_{min} = 0.1$ . In short, the result for operator-discretized tSNN achieves a significantly better throughput, than Euler-discretized one, demonstrating competitive accuracy and throughput to those of ANN.

Figure 3(c) compares the number of neural signals and throughput for rSNN, Euler-discretized tSNN, and operator-discretized tSNN. In the Euler-discretized tSNN, the throughput is severely affected despite the reduction of the number of spikes. Since information is encoded in time rather than in amplitude, naive discretization using fine-grained  $\Delta T$  steps is not very efficient in terms of both accuracy and throughput. Indeed, the computing complexity proportionally increases as the number of  $\Delta T$  steps, rather than as the number of neural signals. In contrast, both the number of spikes and throughput are comparable to those of ANN in operator-discretized tSNN. The proposed emulation strategy meets computing efficiency without washing out actual neural signal waveforms by embedding fine-grained temporal dynamics into crosscorrelations of operators.

The proposed emulation strategy is expected to be as scalable to larger workloads as ANNs. To validate this assumption, our emulation approach was applied to larger data sets and architectures. Figure 4 summarises the benchmark results for CIFAR10&100 and resnet18. This time, we used SGD with  $lr = 0.1$  with batch normalization and  $lr = 0.05$  without batch normalization for better convergence. The learning rates were reduced by  $\times 10$  after every 30 epochs for a total of 90 epochs. Again, the ANN code follows reference implementations and default parameter settings in PyTorch documentation. The programs were executed in x86 internal clusters for higher throughput (at an expense of throughput variations due to other jobs) with python 3.6.9 and PyTorch 1.2.0, but again without GPUs. We used multicores in a single node since the conversion between ANN and tSNN is local i.e., not affected by the node configuration. The result confirms that both accuracy and throughput are similarly competitive to ANNs for larger datasets and models. We performed multiple runs for 10 different seeds. The standard deviations were  $\lesssim 1\%$  and  $\lesssim 10\%$  for accuracy and throughput, respectively.

# 5 Conclusion

This paper proposed a new representation of neural networks that can efficiently compute their dynamics as sequences of operator-discretized events. Our approach takes advantage of existing mathematical frameworks that have been originally developed to abstract and discretize high-dimensional quantum systems with necessary modifications to handle neural networks. Different generations of neural networks, such as ANN and SNN, were attributed to different selections for operators and encoding. Our formulation, when applied to tSNNs, led to a more computationally efficient SW emulation with fully exploiting existing ANN assets. Presently, learning is not perfectly asynchronous because of Eq. 32. However, this limitation makes sense considering that the biological brains also use slow brain waves to efficiently regulate their operations without much affecting online tracking.

# References

[1] Sejnowski., T. J. Time for a new neural code? Nature 376, 21-22 (1995).  
[2] Maass, W. Networks of Spiking Neurons: The Third Generation Neural Network Models. Neural Networks 10, 1659-1671 (1997).  
[3] Bohte, S. M. The evidence for neural information processing with precise spike-times: A survey. Neural Computing 3, 195-206 (2004).  
[4] Gutig, R. & Sompolinsky, H. The tempotron: a neuron that learns spike timing-based decisions. Nature Neuroscience 9, 420-428 (2006).  
[5] Huh, D. & Sejnowski, T. J. Gradient Descent for Spiking Neural Networks. NeurIPS (2017).  
[6] Shrestha, S. B. & Orchard, G. SLAYER: Spike Layer Error Reassignment in Time. NeurIPS (2017).  
[7] Wozniak, S. Pantazi, A., Bohnstingl, T. & Eleftheriou, E. Deep learning incorporating biologically-inspired neural dynamics. Nat Mac Intell 2, 325-336 (2020).  
[8] Bai, A., Kolter, J. Z. & Koltun, V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. CoRR abs/1803.01271 (2018).  
[9] Salinas, D., Flunkert, V. & Gasthaus, J. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. International. Journal of Forecasting (2019).  
[10] Grigsby, J., Wang, Z. & Qi, Y. Long-Range Transformers for Dynamic Spatiotemporal Forecasting. arXiv.2109.12218 (2021).  
[11] Lim, B., Sercan, O. A., Loeff, N. & Pfister, T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International. Journal of Forecasting (2020).  
[12] Wunderlich, T. C. & Pehle, C. Event-based backpropagation can compute exact gradients for spiking neural networks. Scientific Reports 11, 12829 (2021).  
[13] Goltz, J., Kriener, L., Baumbach, Billaudelle, A. S., Breitwieser, O., Cramer, B., Dold, D., Kungl, A. F., Senn, W., Schemmel, J., Meier, K. & Petrovici, M. A. Fast and energy-efficient neuromorphic deep learning with first-spike times. Nat Mach Intell 3, 823-835 (2021).  
[14] Comsa, I.-M., Versari, L., Fischbacher, T. & Alakuijala, J. Spiking Autoencoders with Temporal Coding. Front. Neurosci. 15, 712667 (2021).  
[15] Weidel, P. & Sheik, S. Wavesense: Efficient Temporal Convolutions with Spiking Neural Networks for Keyword Spotting. arXiv:2111.01456v1 (2021).  
[16] Susi, G., Garcés, P., Paracone, E., Cristini, A., Salerno, M., Maestro, F. & Pereda, E. FNS allows efficient event-driven spiking neural network simulations based on a neuron model supporting spike latency. Scientific Reports 11, 12160 (2021).  
[17] O’ Keefe, J. & Recce, M. L. Phase relationship between hippocampal place units. and the EEG theta rhythm Hippocampus 3, 317-330 (1993).  
[18] Hopfield, J. J. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences 79, 2554-2558 (1982).  
[19] Doi, M. Second quantization representation for classical many-particle systems. J. Phys. A: Math. Gen. 9, 1465-1477 (1976).  
[20] Peliti, L. Path integral approach to birth-death processes on a lattice. J. Phys. (Paris) 46, 1469-1483 (1985).  
[21] Täuber, U. C., Howard, M. & Vollmary-Lee, B. P. Applications of Field-Theoretic Renormalization Group Methods to Reaction-Diffusion Problems. J. Phys. A: Math. Gen. 38, R79 (2005).  
[22] Schuld, M. & Carrasquilla, J. Machine Learning With Quantum Computers. Tutorial NeurIPS (2021).  
[23] DiVincenzo, D. P. The Physical Implementation of Quantum Computation. Progress of Physics 48, 771-783 (2000).  
[24] Abramsky, S. & Coecke, B. A categorical semantics of quantum protocols. LiCS'04 (2004).  
[25] Coecke, B. & Kissinger, A. Picturing Quantum Processes (Cambridge University Press, 2017).

[26] D'Ariano, G. M., Chiribella, G. & Perinotti, P. Quantum theory from first principles (Cambridge University Press, 2017).  
[27] Coecke, B. & Paquette, É. Categories for the Practising Physicist. New Structures for Physics. Lecture Notes in Physics 813 (Springer, 2010).  
[28] Hirose, A. Complex-Valued Neural Networks (Springer Nature, 2012).  
[29] Aaronson, S. Quantum computing since Democritus (Cambridge University Press, 2013).  
[30] Tse, D. & Viswanath, P. Fundamentals of Wireless Communication (Cambridge University Press, 2005).  
[31] Katayama, Y., Yamane, T., Nakano, D., Tanaka, G. & Nakane, R. Wave-Based Neuromorphic Computing Framework for Brain-Like Energy Efficiency and Integration. IEEE Trans. Nanotechnol. 15, 762-769 (2016).  
[32] Katayama, Y. Channel Model for Spiking Neural Networks Inspired by Impulse Radio MIMO Transmission. IEEE GLOBECOM (2019).  
[33] Senk, J., Korvasova, K., Schuecker, J., Hagen, E., Tetzlaff, T., Diesmann, M. & Helias, M. Conditions for wave trains in spiking neural networks. Phys. Rev. Res. 2, 023174 (2020).  
[34] Gepstein, S., Pawar, A. S., Kwon, S., Savel'ev, S. & Albright, T. D. Spatially distributed computation in cortical circuits. Sci. Adv. 8, eabl5865 (2022).  
[35] Katayama, Y., Yamane, T. & Nakano, D. An Energy-Efficient Computing Approach by Filling the Connectome Gap. Unconventional Computation and Natural Computation (2014).  
[36] Moskalets, M. V. Scattering Matrix Approach to Non-Stationary Quantum Transport (Imperial College Press, 2011).  
[37] McCulloch, W. S. & Pitts, W. A logical calculus of the ideas immanent in nervous activity. The bulletin of mathematical biophysics 5,115-133 (Kluwer Academic Publishers, 1943).  
[38] Blais, A., Grimsmo, A. L., Girvin, S. M. & Wallraff, A. Circuit Quantum Electrodynamics. Rev. Mod. Phys. 93, 025005 (2021).  
[39] Sterling, P. & Laughlin, S. Principles of Neural Design (The MIT Press, 2015).  
[40] Kobayashi, H. & Tang, D. Application of Partial-response Channel Coding to Magnetic Recording Systems. IBM J. Res. Dev. 14, 368-375 (1970).  
[41] Rueckauer, B., Hu, Y., Lungu, I. A., Pfeiffer, M. & Liu, S.-C. Conversion of continuous-valued deep networks to efficient event-driven networks for image classification. Front. Neurosci. (2017).  
[42] Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L. & Lerer, A. Automatic differentiation in PyTorch. NeurIPS (2017).  
[43] Chen, R. T. Q., Rubanova, Y., Bettencourt, J. & Duvenaud, D. Neural Ordinary Differential Equations NeurIPS (2018).  
[44] Neftci, E. O., Mostafa; H. & Zenke, F. Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks. IEEE Signal Processing Magazine 36, 51-63 (2019).
