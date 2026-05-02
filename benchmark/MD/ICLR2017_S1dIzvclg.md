# A RECURRENT NEURAL NETWORK WITHOUT CHAOS

# Thomas Laurent

Department of Mathematics

Loyola Marymount University

Los Angeles, CA 90045, USA

tlaurent@lmu.edu

# James von Brecht

Department of Mathematics

California State University, Long Beach

Long Beach, CA 90840, USA

james.vonbrecht@csulb.edu

# ABSTRACT

We introduce an exceptionally simple gated recurrent neural network (RNN) that achieves performance comparable to well-known gated architectures, such as LSTMs and GRUs, on the word-level language modeling task. We prove that our model has simple, predicable and non-chaotic dynamics. This stands in stark contrast to more standard gated architectures, whose underlying dynamical systems exhibit chaotic behavior. We provide experiments to show that our network performs comparably to chaotic RNNs on word-level language modeling, which indicates that chaos in and of itself is not a necessary ingredient for learning in this setting.

# 1 INTRODUCTION

Gated recurrent neural networks, such as the Long Short Term Memory network (LSTM) introduced by Hochreiter & Schmidhuber (1997) and the Gated Recurrent Unit (GRU) proposed by Cho et al. (2014), prove highly effective for machine learning tasks that involve sequential data. We propose an exceptionally simple variant of these gated architectures. The basic model takes the form

$$
h _ {t} = \theta_ {t} \odot \tanh  \left(h _ {t - 1}\right) + \eta_ {t} \odot \tanh  \left(W x _ {t}\right), \tag {1}
$$

where  $\odot$  stands for the Hadamard product. The horizontal/forget gate (i.e.  $\theta_{t}$ ) and the vertical/Input gate (i.e.  $\eta_{t}$ ) take the usual form used in most gated RNN architectures. Specifically

$$
\theta_ {t} := \sigma \left(U _ {\theta} h _ {t - 1} + V _ {\theta} x _ {t} + b _ {\theta}\right) \quad \text {a n d} \quad \eta_ {t} := \sigma \left(U _ {\eta} h _ {t - 1} + V _ {\eta} x _ {t} + b _ {\eta}\right) \tag {2}
$$

where  $\sigma (x)\coloneqq (1 + \mathrm{e}^{-x})^{-1}$  denotes the logistic sigmoid function.

Despite similarities in functional form with popular gated RNN such as LSTMs and GRUs, the proposed model (1)-(2) fundamentally differs from them at the level of the underlying dynamics that drive hidden states. The network (1) has quite intuitive dynamics. Suppose the data  $x_{t}$  present the model with an impulse sequence

$$
\left(W x _ {t}\right) (i) = \left\{ \begin{array}{l l} 1 0 & \text {i f} t = T \\ 0 & \text {o t h e r w i s e ,} \end{array} \right. \tag {3}
$$

or in other words an input sequence  $x_{t}$  for which the learned  $i^{\mathrm{th}}$  feature  $(Wx_{t})(i)^{1}$  remains off except at time  $T$ . When initialized from  $h_0 = 0$ , the corresponding response of the network to this "impulse" in the  $i^{th}$  feature is

$$
h _ {t} (i) \approx \left\{ \begin{array}{l l} 0 & \text {i f} t <   T \\ \eta_ {T} & \text {i f} t = T \\ \alpha_ {t} & \text {i f} t > T \end{array} \right. \tag {4}
$$

with  $\alpha_{t}$  a sequence that relaxes toward zero. The  $\theta_{t}$  and  $\eta_{t}$  gates control the rate of this relaxation. Thus  $h_t(i)$  activates when presented with a strong  $i^{\mathrm{th}}$  feature, and then relaxes toward zero until the data present the network once again with strong  $i^{th}$  feature. Overall this leads to a dynamically simple model, in which the activation patterns in the hidden states of the network have a clear cause and predictable subsequent behavior.

Dynamics of this sort do not occur in other RNN models. Instead, the three most popular recurrent neural network architectures, namely the vanilla RNN, the LSTM and the GRU, have complex, irregular, and unpredictable dynamics. In the absence of input data, these networks give rise to chaotic dynamical systems. In other words, when presented with null or zero input data, activation patterns in their hidden states do not necessarily follow a predictable path. The presence of chaos in LSTMs, GRUs and vanilla RNNs shows that these models exhibit complex and rich dynamics, and that they prove capable of expressing a wide variety of behaviors. The proposed network (1)-(2) has rather dull and minimalist dynamics in comparison. Indeed, its only attractor is the zero state and so stands at the polar-opposite end of the spectrum from a chaotic system. Perhaps surprisingly, at least in the light of this comparison, it performs as well as LSTMs and GRUs on the word level language modeling task. We therefore conclude that the ability of an RNN to form complex, chaotic temporal dynamics cannot explain its success on word-level language modeling tasks.

In the next section, we review the phenomenon of chaos in RNNs via both synthetic examples and trained models. We then prove a precise, quantified description of the dynamical picture (3) for our model. In particular, we show that the model is never chaotic, and for this reason we refer to (1)-(2) as a Chaos-Free Network (CFN). The final section provides a series of experiments that demonstrate that CFN achieve results comparable to LSTM on the word-level language modeling task. All together, these observations show that an architecture as simple as (1)-(2), with a regular and predictable behavior, can achieve performance comparable to the more complex LSTM.

# 2 CHAOS IN RECURRENT NEURAL NETWORKS

The study of RNNs from a discrete-time dynamical systems point-of-view has brought fruitful insights into generic features of RNNs. This point-of-view can aid and guide intuition about how a particular model, say an LSTM, uses its hidden states to learn complicated and complex temporal relationships between data. We shall pursue a brief investigation of our proposed model and classical gated architectures using this formalism, as it allows us to identify key distinctions between them. Recall that for a given continuously differentiable mapping  $\Phi : \mathbb{R}^d \mapsto \mathbb{R}^d$ , a given initial time  $t_0 \in \mathbb{N}$  and a given initial state  $u_0 \in \mathbb{R}^d$ , a simple repeated iteration of the mapping  $\Phi$

$$
\mathfrak {u} _ {t + 1} = \Phi (\mathfrak {u} _ {t}) \quad t > t _ {0},
$$

$$
u _ {t _ {0}} = u _ {0} \quad t = t _ {0},
$$

defines a discrete-time dynamical system. The index  $t \in \mathbb{N}$  represents the current time, while the point  $\mathfrak{u}_t \in \mathbb{R}^d$  represents the current state of the system.

Most RNNs generically take the functional form

$$
\mathfrak {u} _ {t} = \Psi \left(\mathfrak {u} _ {t - 1}, W _ {1} x _ {t}, W _ {2} x _ {t}, \dots , W _ {k} x _ {t}\right), \tag {5}
$$

where  $x_{t}$  denotes the  $t^{\mathrm{th}}$  input data point. For example, in the case of the CFN (1)-(2), we have  $W_{1} = W$ ,  $W_{2} = V_{\theta}$  and  $W_{3} = V_{\eta}$ . To gain insight into the underlying design of the architecture of an RNN, it proves useful to consider how trajectories behave when they are not influenced by any external input. This leads us to consider the dynamical system

$$
u _ {t} = \Phi (u _ {t - 1}) \quad \Phi (u) := \Psi (u, 0, 0, \dots , 0), \tag {6}
$$

which we refer to as the dynamical system induced by the recurrent neural network. The time-invariant system (6) is much more tractable than (5), and it offers a mean to investigate the inner working of a given architecture; it separates the influence of input data  $x_{t}$ , which can produce essentially any possible response, from the model itself. Studying trajectories that are not influenced by external data will give us an indication on the ability of a given RNN to generate complex and sophisticated trajectories by its own. As we shall see shortly, the dynamical system induced by CFN has excessively simple and predictable trajectories: all of them converge to the zero state. This is in sharp contrast with the dynamical systems induced by LSTM and GRU, for which the trajectories are excessively complex and exhibit chaotic and unpredictable behavior.

The learned parameters  $W_{j}$  in (5) describe how data influence the evolution of hidden states at each time step. From a modeling perspective, (6) would occur in the scenario where a trained RNN has learned a weak coupling between a specific data point  $x_{t_0}$  and the hidden state at that time, in the sense that the data influence is small and so all  $W_{j}x_{t_0}\approx 0$  nearly vanish. The hidden state then transitions according to  $\mathfrak{u}_{t_0}\approx \Psi (\mathfrak{u}_{t_0 - 1},0,0,\dots ,0) = \Phi (\mathfrak{u}_{t_0 - 1})$ .

![](images/da42dff13b4677bd1147d1c3831f79818e1b5d3c58035ecd993022a4a45a8b52.jpg)  
(a) GRU:  $(h(1), h(2))$

![](images/7e90e9ea0d7d7b56771df7ec8c429948eb0086356fa3edf0899d0b8f7d580d75.jpg)  
(b) LSTM:  $(h(1), h(2))$

![](images/fc7ad2e17ac34e6481f98fc53f33e5a3f61b3300782b6dedb67e339051572395.jpg)  
(c) GRU:  $(h(1) - \hat{h}(1))_t$

![](images/ba3cb8b3c59413024e1fa53389b8b4ab3d57bd575cb9167eeda0a16595625e17.jpg)  
(d) LSTM:  $(h(1) - \hat{h}(1))_t$  
Figure 1: Top Row: Chaotic attractors for classical gated RNN architectures. Bottom Row: Sensitivity to initial conditions along the attractor.

For a given dynamical system  $\mathfrak{u}_t = \Phi (\mathfrak{u}_{t - 1})$ , the set of all visited states

$$
\mathcal {O} ^ {+} \left(u _ {0}\right) := \left\{u _ {t _ {0}}, u _ {t _ {0} + 1}, \dots , u _ {t _ {0} + n}, \dots \right\}
$$

defines the forward trajectory or forward orbit through  $\mathfrak{u}_0$  determined by the dynamical system. At heart, dynamical systems theory concerns the ability to make predictions about the eventual fate of a forward orbit. In certain instances, the dynamical systems induced by both the GRU and the LSTM can exhibit chaotic forward trajectories. As a consequence, the behavior of forward orbits in their induced dynamical systems exhibit a high degree of sensitivity to the initial state  $\mathfrak{u}_0$  that defines a forward orbit. Figure 2 provides a classical example of such behavior. It depicts a chaotic attractor of the dynamical system

$$
u _ {t} = h _ {t}, \quad u \mapsto \Phi (u) := (1 - z) \odot u + z \odot \tanh  (R (r \odot u)) \tag {7}
$$

$$
z := \sigma \left(A _ {z} u + b _ {z}\right) \quad r := \sigma \left(A _ {r} u + b _ {r}\right),
$$

induced by a two-dimensional GRU, with weight matrices

$$
A _ {\theta} = \left[ \begin{array}{c c} 0 & 1 \\ 1 & 1 \end{array} \right] \quad A _ {r} = \left[ \begin{array}{c c} 0 & 1 \\ 1 & 0 \end{array} \right] \quad R = \left[ \begin{array}{c c} - 5 & - 8 \\ 8 & 5 \end{array} \right]
$$

and zero bias for the model parameters. Most forward orbits  $\mathcal{O}(\mathfrak{u}_0)$  for the dynamical system (7) eventually approach the attractor, which Figure 2(a) depicts in black. The exact location of any state  $\mathfrak{u}_t$  in a trajectory proves difficult to predict with any accuracy, however. To illustrate this, we compute an initial condition  $\mathfrak{u}_0$  for the dynamical system (7) by drawing a point uniformly at random from the unit square. We then compute 10,000 small amplitude perturbations  $\hat{\mathfrak{u}}_0$  of  $\mathfrak{u}_0$  by adding a small random number drawn uniformly from  $[-10^{-7}, 10^{-7}]$  to each component. We then iterate (7) for 20,000 steps (so that the trajectory  $\mathcal{O}(\hat{\mathfrak{u}}_0)$  has had time to reach the attractor) and plot the final state  $\mathfrak{u}_{20,000}$  for each of the 10,000 trials in yellow. The collection of these 10,000 final states essentially fills out the entire attractor, despite the fact that their initial conditions are highly localized (i.e. at distance of no more than  $10^{-7}$ ) around a fixed point. In other words, the time  $t = 20,000$  map of the dynamical system will map a small neighborhood around a fixed initial condition  $\mathfrak{u}_0$  to the

entire attractor. Figure 2(c) additionally illustrates this sensitivity to initial conditions for points on the attractor itself. We take an initial condition  $\mathfrak{u}_0$  on the attractor and perturb it by  $10^{-7}$  to a nearby initial condition  $\hat{\mathbf{u}}_0$  as before. After an initial phase of agreement, the forward trajectories  $\mathcal{O}(\mathfrak{u}_0)$  and  $\mathcal{O}(\hat{\mathbf{u}}_0)$  strongly diverge. Figures 2(b,d) repeat this same set of experiments for the dynamical system

$$
\mathfrak {u} _ {t} = \left[ \begin{array}{c} h _ {t} \\ c _ {t} \end{array} \right] \quad \mathfrak {u} \mapsto \Phi (\mathfrak {u}) = \left[ \begin{array}{c} o \odot \tanh  (f \odot c + i \odot g) \\ f \odot c + i \odot g \end{array} \right] \tag {8}
$$

$$
i := \sigma (A _ {i} h + b _ {i}) \quad f := \sigma (A _ {f} h + b _ {f}) \quad o := \sigma (A _ {o} h + b _ {o}) \quad g := \tanh  (A _ {g} h + b _ {g})
$$

induced by a two dimensional LSTM, with weight matrices

$$
A _ {i} = \left[ \begin{array}{c c} - 1 0 & - 1 0 \\ 1 0 & 1 0 \end{array} \right] \quad A _ {o} = \left[ \begin{array}{c c} 1 0 & - 1 0 \\ 1 0 & - 1 0 \end{array} \right] \quad A _ {f} = \left[ \begin{array}{c c} - 1 0 & - 1 0 \\ 1 0 & 1 0 \end{array} \right] \quad A _ {g} = \left[ \begin{array}{c c} 1 0 & - 8 0 \\ 8 0 & - 1 0 \end{array} \right]
$$

and zero bias for the model parameters. Both of these synthetic examples illustrate the potentially chaotic nature of the GRU and LSTM architectures, as well as the corresponding inability to make accurate long-term predictions regarding the values of hidden states and behavior of forward orbits. Finally, in Figure 2 we illustrate that behavior with these qualitative features occurs for trained models and not just for synthetically generated instances. We take the parameter values (both transition matrices and bias) of an LSTM with 228 hidden units trained on the Penn Treebank corpus without dropout (c.f. the experimental section for the precise procedure). We then set all data inputs  $x_{t}$  to zero and run the corresponding induced dynamical system (8). Figure 2(a) plots the first two components  $(h(1), h(2))$  of the hidden state along the attractor, while Figures 2(b,c) plot the difference between perturbed initial conditions  $\mathfrak{u}_0$ ,  $\hat{\mathbf{u}}_0$  at level  $10^{-7}$  on the attractor over 1,000 and 5,000 time steps. Once again, the long-term behavior of the hidden states to small perturbations in the initial value is evident.

![](images/37babdf038f5fc53e99d94940e4711448167073c3de696ada3a643c2ae54dd7b.jpg)  
(a)  $(h(1), h(2))$

![](images/b07f66b106e623efec3f0f94daa46b47df67560d7a4274995814dedd53738507.jpg)  
(b)  $(h(1) - \hat{h} (1))_t$

![](images/6be0b4ec0f531f0f59e18d16764a1d8796094a7f39755590621ab0f03e11e1bc.jpg)  
(c)  $(h(1) - \hat{h} (1))_t$  
Figure 2: Dynamics of A Trained LSTM

Despite their superficial similarity, the model (1) behaves dramatically different than other gated architectures (e.g. (7),(8)) in this context. The behavior of a forward orbit  $\mathcal{O}^{+}(\mathfrak{u}_{0})$  of the corresponding dynamical system  $\mathfrak{u}_t = \Phi (\mathfrak{u}_{t - 1})$  behaves in a perfectly predictable fashion. Indeed, the origin  $\mathfrak{u} = 0$  is a global attractor for the CFN model — regardless of the initial state  $\mathfrak{u}_0$ , all forward trajectories will eventually converge to the origin. This stands in stark contrast to other gated architectures, whose forward orbits  $\mathcal{O}^{+}(\mathfrak{u}_{0})$  can converge toward chaotic attractors, and which are characterized by an inherent unpredictability and dramatic sensitivity to initial states.

# 2.1 GLOBAL BEHAVIOR OF THE CFN

We now turn to an analysis of the long-term behavior of forward orbits for our proposed architecture. In this case we may make precise statements about the behavior of our model even if the data  $x_{t}$  do not necessarily embed in a trivial fashion. We shall use notations such as  $h_t(i)$  or  $(Ax_{t})_{i}$  to denote the  $i^{\mathrm{th}}$  component of a vector throughout our analysis, which we begin with the following definition.

Definition 1 (ε-activation). A data point  $x_{t}$  ε-activates the  $i^{\text{th}}$  state of (1) at level ε if  $|(Ax_{t})_{i}| \leq \varepsilon$ . A data sequence  $\mathcal{X} = \{x_{1}, x_{2}, \ldots\}$  ε-activates the  $i^{\text{th}}$  state of (1) beyond  $t_{*}$  if  $x_{t}$  ε-activates the  $i^{\text{th}}$  state at level ε for all  $t \geq t_{*}$ .

This definition simply serves to quantify when a weak coupling of input data to the  $i^{\mathrm{th}}$  hidden state occurs. If after time  $t_*$  either the model parameters or the data themselves result in only a small (as measured by  $\varepsilon$ ) influence on  $h_t(i)$  then  $\varepsilon$ -activation beyond  $t_*$  occurs for that component. We shall show that the  $i^{\mathrm{th}}$  component  $h_t(i)$  all remain small in this scenario. Let us assume that the embedded input data  $\mathcal{X} \coloneqq (x_1, x_2, x_3, \ldots)$  form a bounded sequence in the sense that

$$
\mathcal {X} _ {\infty} := \sup  _ {t \geq 0} \| x _ {t} \| _ {\infty} <   + \infty , \tag {9}
$$

which holds trivially when  $x_{t}$  represent the embeddings of a standard basis vectors. The hypothesis (9) suffices to show that the inequalities

$$
\sup  _ {t \geq t _ {*}} \theta_ {t} (i) =: \theta^ {*} (i) <   1 \quad \text {a n d} \quad \sup  _ {t \geq t _ {*}} \eta_ {t} (i) =: \eta^ {*} (i) <   1 \tag {10}
$$

hold for the  $i^{\mathrm{th}}$  component of both gates. To see this, first note that  $-2\leq h_t(i)\leq 2$  by definition of the architecture. Thus the inputs to any gate remain uniformly bounded for all  $t\geq t_{*}$ , in the sense that there exists some finite  $C > 0$  so that

$$
| (U _ {\theta / \eta} h _ {t - 1} ^ {(\ell)} + V _ {\theta / \eta} h _ {t} + b _ {\theta / \eta}) _ {i} | \leq C
$$

holds for all  $t \geq t_*$ , and thus

$$
\theta_ {t} (i) \leq \sigma (C) <   1 \quad \text {a n d} \quad \eta_ {t} (i) \leq \sigma (C) <   1
$$

by definition of the sigmoid. Thus (10) holds. The non-expansivity of the hyperbolic tangent, i.e.  $|\tanh(x)| \leq |x|$ , and the triangle inequality then combine to show

$$
\left| h _ {t + 1} (i) \right| \leq \theta^ {*} (i) \left| h _ {t} (i) \right| + \eta^ {*} (i) \varepsilon
$$

whenever  $t \geq t_*$  and the data  $\varepsilon$ -activate the  $i^{\text{th}}$  hidden feature beyond  $t_*$ . Iterating this inequality and summing the geometric series then gives

$$
\left| h _ {t _ {*} + k} (i) \right| \leq \theta^ {*} (i) ^ {k} \left| h _ {t _ {*}} (i) \right| + \left(1 - \theta^ {*} (i) ^ {k}\right) \eta^ {*} (i) \varepsilon / \left(1 - \theta^ {*} (i)\right)
$$

for all future time. In other words, if beyond some point  $t_*$  the  $i^{\mathrm{th}}$  inputs  $(Ax_{t})_{i}$  remain small (say at activation strength  $0 \leq \varepsilon \ll 1$ ) then the corresponding component of the hidden state remains near zero. In particular, if the model receives null data  $x_{t} = 0$  beyond  $t_*$  then  $\varepsilon = 0$  and so all hidden states eventually vanish, and this property holds regardless of the model parameters. An easy extension of this argument proves this property for a general  $L$ -level architecture, where now

$$
\left. \left| h _ {t _ {*} + k} ^ {(\ell)} \right| \leq C _ {\ell} (1 + t) ^ {k} \left(\max  \left\{\theta^ {* (1)}, \dots , \theta^ {* (\ell)} \right\}\right) ^ {k} \right.
$$

for  $C_{\ell} > 0$  some constant depending only on the norms  $\| A^{(j)}\|_{\infty}$  of the embedding matrices and the sizes  $|h_{t_*}^{(j)}|$  of the initial conditions at all previous  $1 \leq j \leq \ell$  levels. This bound holds whenever the model receives null data  $x_{t} = 0$  beyond  $t_*$ , and therefore all hidden states at all levels eventually decay to zero. Generically, higher levels (i.e. larger  $\ell$ ) will generally decay more slowly, and remain non-trivial, while earlier levels (i.e. smaller  $\ell$  decay more quickly. Finally, a simple observation shows that we can modify (1) in a simple way and obtain a chaos-free RNN with a learnable, non-zero long-term asymptotic state. We simply replace all occurrences of  $h_t$  in (1) with  $h_t - h_\infty$  and view  $h_\infty$  as an extra collection of learnable parameters that define the long-term asymptotic state. The corresponding process for a multi-level architecture is similar, although we have not yet implemented this feature.

To conclude, we may view the overall picture in several ways. An optimist's point-of-view might regard the presence of chaos in LSTM, GRU and the vanilla RNN as a manifestation of complex or rich dynamics. These models produce a wide variety of behaviors, and this provides evidence of their modeling power. Chaos in and of itself might aid the learning process. A more pessimistic point-of-view would worry that the presence of underlying chaos indicates an inability to accurately predict future events. Small changes to the model (e.g. in initial conditions or parameters) can lead to vastly different results. Moreover, the activation patterns in hidden states of a chaotic model need not have an obvious underlying cause. We take an agnostic perspective, and simply ask whether such dynamics prove necessary or are somehow responsible for the success of RNNs in word-level language modeling. Our experiments indicate that it is not.

# 3 EXPERIMENTS

In this section we show that despite its simplicity, the CFN network achieves performance comparable to the much more complex LSTM network on the word level language modeling task. We use two datasets for these experiments, namely the Penn Treebank corpus (Marcus et al., 1993) and the Text8 corpus (Mikolov et al., 2014). We consider both one-layer and two-layer CFNs and LSTMs for our experiments. We train both CFN and LSTM networks in a similar fashion and always compare models that use the same number of parameters. We compare their performance with and without dropout, and show that in both cases they obtain similar results. We also provide results published in Mikolov et al. (2014), Jozefowicz et al. (2015) and Sukhbaatar et al. (2015) for the sake of comparison.

For concreteness, the exact implementation for the two-layer architecture of our model is

$$
h _ {t} ^ {(0)} = W ^ {(0)} x _ {t} \tag {11}
$$

$$
\hat {h} _ {t} ^ {(0)} = \operatorname {D r o p} \left(h _ {t} ^ {(0)}, p\right) \tag {12}
$$

$$
h _ {t} ^ {(1)} = \theta_ {t} ^ {(1)} \odot \tanh  \left(h _ {t - 1} ^ {(1)}\right) + \eta_ {t} ^ {(1)} \odot \tanh  \left(W ^ {(1)} \hat {h} _ {t} ^ {(0)}\right) \tag {13}
$$

$$
\hat {h} _ {t} ^ {(1)} = \operatorname {D r o p} \left(h _ {t} ^ {(1)}, p\right) \tag {14}
$$

$$
h _ {t} ^ {(2)} = \theta_ {t} ^ {(2)} \odot \tanh  \left(h _ {t - 1} ^ {(2)}\right) + \eta_ {t} ^ {(2)} \odot \tanh  \left(W ^ {(2)} \hat {h} _ {t} ^ {(1)}\right) \tag {15}
$$

$$
\hat {h} _ {t} ^ {(2)} = \operatorname {D r o p} \left(h _ {t} ^ {(2)}, p\right) \tag {16}
$$

$$
y _ {t} = \operatorname {L o g S o f t m a x} \left(W ^ {(3)} \hat {h} _ {t} ^ {(2)} + b\right) \tag {17}
$$

where  $\operatorname{Drop}(z, p)$  denotes the dropout operator with a probability  $p$  of setting components in  $z$  to zero. We compute the gates according to

$$
\tilde {h} _ {t - 1} ^ {(\ell)} = \operatorname {D r o p} \left(h _ {t - 1} ^ {(\ell)}, q\right) \tag {18}
$$

$$
\tilde {h} _ {t} ^ {(\ell - 1)} = \operatorname {D r o p} \left(h _ {t} ^ {(\ell - 1)}, q\right) \tag {19}
$$

$$
\theta_ {t} ^ {(\ell)} := \sigma \left(U _ {\theta} ^ {(\ell)} \tilde {h} _ {t - 1} ^ {(\ell)} + V _ {\theta} ^ {(\ell)} \tilde {h} _ {t} ^ {(\ell - 1)} + b _ {\theta}\right) \tag {20}
$$

$$
\eta_ {t} ^ {(\ell)} := \sigma \left(U _ {\eta} ^ {(\ell)} \tilde {h} _ {t - 1} ^ {(\ell)} + V _ {\eta} ^ {(\ell)} \tilde {h} _ {t} ^ {(\ell - 1)} + b _ {\eta}\right), \tag {21}
$$

and thus the model has two dropout hyperparameters. The parameter  $p$  controls the amount of dropout between layers; the parameter  $q$  controls the amount of dropout inside each gate. We use a similar dropout strategy for the LSTM, in that all sigmoid gates  $f, o$  and  $i$  receive the same amount  $q$  of dropout.

To train the CFN and LSTM networks, we use a simple online steepest descent algorithm. We update the weights  $w$  via

$$
w ^ {(k + 1)} = w ^ {(k)} - \operatorname {I r} \cdot \vec {p} \quad \text {w h e r e} \quad \vec {p} = \frac {\nabla_ {w} L}{\| \nabla_ {w} L \| _ {2}}, \tag {22}
$$

where  $\nabla_w L$  denotes the approximate gradient of the loss with respect to the weights as estimated from a certain number of presented examples. We use the usual backpropagation through time approximation when estimating the gradient: we unroll the net  $T$  steps in the past and neglect longer dependencies. In all experiments, the CFN and LSTM networks are unrolled for  $T = 35$  steps and we take minibatches of size 20, and so a total of 700 inputs are processed between each parameter update. In the case of an exact gradient, the update (22) simply corresponds to making a step of length lr in the direction of steepest descent. As all search directions  $\vec{p}$  have Euclidean norm  $\| \vec{p} \|_2 = 1$ , we perform no gradient clipping during training.

We initialize all the weights in the CFN, except for the bias of the gates, uniformly at random in  $[-0.07, 0.07]$ . We initialize the bias of the horizontal and vertical gates to 1 and  $-1$ , respectively, so that at the beginning of the training

$$
\theta_ {t} \approx \sigma (1) \approx 0. 7 3 \quad \text {a n d} \quad \eta_ {t} \approx \sigma (- 1) \approx 0. 2 3.
$$

We initialize the weights of the LSTM in exactly the same way; the bias for the forget and input gate are initialized to 1 and  $-1$ , and all the other weights are initialized uniformly in  $[-0.07, 0.07]$ .

Table 1: Experiments on Penn Treebank without dropout.  

<table><tr><td>Model</td><td>Size</td><td>Training</td><td>Val. perp.</td><td>Test perp.</td></tr><tr><td>Vanilla RNN</td><td>5M parameters</td><td>Jozefowicz et al. (2015)</td><td>-</td><td>122.9</td></tr><tr><td>GRU</td><td>5M parameters</td><td>Jozefowicz et al. (2015)</td><td>-</td><td>108.2</td></tr><tr><td>LSTM</td><td>5M parameters</td><td>Jozefowicz et al. (2015)</td><td>-</td><td>109.7</td></tr><tr><td>LSTM</td><td>5M parameters</td><td>Trained by us</td><td>108.4</td><td>105.1</td></tr><tr><td>CFN</td><td>5M parameters</td><td>Trained by us</td><td>109.3</td><td>106.3</td></tr></table>

This initialization scheme favors the flow of information in the horizontal direction. The importance of a careful initialization of the forget gate was first pointed out in Gers et al. (2000) and further emphasized in Jozefowicz et al. (2015). Finally, we initialize all hidden states to zero for both models.

Dataset Construction. The Penn Treebank Corpus has 1 million words and a vocabulary size of 10,000. We used the code from Zaremba et al. (2014) to construct and split the dataset into a training set (929K words), a validation set (73K words) and a test set (82K words). The Text8 corpus has 100 million characters and a vocabulary size of 44,000. We used the script from Mikolov et al. (2014) to construct and split the dataset into a training set (first 99M characters) and a development set (last 1M characters).

# 3.1 EXPERIMENTS WITHOUT DROPOUT

Tables 1 and 2 provide a comparison of various recurrent network architectures without dropout  $(p = q = 0$  in (11)-(21)) evaluated on the Penn Treebank corpus and the Text8 corpus. The last two rows of each table provide results for LSTM and CFN networks trained and initialized in the manner described above.

In the Penn Treebank experiment, the CFN network has two hidden layers of 224 units each for a total of 5 million parameters. The LSTM has one hidden layer with 228 units for a total of 5 million parameters as well. We also tried a two layer LSTM with 5 million parameters but the result was worse (test perplexity of 112) and we did not report it in the table. For the Text8 experiments, the LSTM has one layer with 500 hidden units for a total 46.4 million parameters (we choose these specific numbers so that we can compare with published results). The CFN has two hidden layers with 495 units each, for a total of 46.4 million parameters as well. We used a simple and aggressive learning rate schedule in both experiments:

- For the CFN: start with a learning rate  $\mathrm{lr} = 5.5$  (for PTB) and  $\mathrm{lr} = 5$  (for Text8) and divide it by 3 at each epoch.  
- For the LSTM: start with a learning rate  $\mathrm{lr} = 7$  (for PTB) and  $\mathrm{lr} = 4$  (for Text8) and divide it by 3 at each epoch.

We also report results published in Jozefowicz et al. (2015) were a vanilla RNN, a GRU and an LSTM network were trained on Penn Treebank, each of them having 5 million parameters (only the test perplexity was reported). Finally we report results published in Mikolov et al. (2014) and Sukhbaatar et al. (2015) where various networks are trained on Text8. Of these four network, only

Table 2: Experiments on Text8 without dropout  

<table><tr><td>Model</td><td>Size</td><td>Training</td><td>Perplexity on development set</td></tr><tr><td>Vanilla RNN</td><td>500 hidden units</td><td>Mikolov et al. (2014)</td><td>184</td></tr><tr><td>SCRN</td><td>500 hidden units</td><td>Mikolov et al. (2014)</td><td>161</td></tr><tr><td>LSTM</td><td>500 hidden units</td><td>Mikolov et al. (2014)</td><td>156</td></tr><tr><td>MemN2N</td><td>500 hidden units</td><td>Sukhbaatar et al. (2015)</td><td>147</td></tr><tr><td>LSTM</td><td>46.4M parameters</td><td>Trained by us</td><td>140.8</td></tr><tr><td>CFN</td><td>46.4M parameters</td><td>Trained by us</td><td>142.0</td></tr></table>

Table 3: Experiments on Penn Treebank with dropout.  

<table><tr><td>Model</td><td>Size</td><td>Training</td><td>Val. perp.</td><td>Test perp.</td></tr><tr><td>Vanilla RNN</td><td>20M parameters</td><td>Jozefowicz et al. (2015)</td><td>103.0</td><td>97.7</td></tr><tr><td>GRU</td><td>20M parameters</td><td>Jozefowicz et al. (2015)</td><td>95.5</td><td>91.7</td></tr><tr><td>LSTM</td><td>20M parameters</td><td>Jozefowicz et al. (2015)</td><td>83.3</td><td>78.8</td></tr><tr><td>LSTM</td><td>20M parameters</td><td>Trained by us</td><td>78.4</td><td>74.3</td></tr><tr><td>CFN</td><td>20M parameters</td><td>Trained by us</td><td>79.7</td><td>74.9</td></tr></table>

the LSTM network from Mikolov et al. (2014) has the same number of parameters than the CFN and LSTM networks we trained (46.4M parameters). The vanilla RNN, Structurally Constrained Recurrent Network (SCRN) and End-To-End Memory Network (MemN2N) all have 500 units, but less than 46.4M parameters. We nonetheless indicate their performance in table 2 to provide some context.

Note that the LSTM network that we trained with the online steepest descent algorithm reaches lower perplexity on Penn Treebank than the LSTM trained in Jozefowicz et al. (2015) (105 vs 110), and lower perplexity on Text8 than the one trained in Mikolov et al. (2014) (141 vs 156). We also observe the same discrepancy in the experiments with dropout presented in the next subsection. It is unclear whether this discrepancy is due to the training algorithm itself, or simply to better hyperparameters tuning. Overall it is clear that despite its simple dynamics, CFN obtains results comparable to the one obtained by LSTM networks and GRUs.

# 3.2 EXPERIMENTS WITH DROPOUT

Table 3 provides a comparison of various recurrent network architectures with dropout evaluated on the Penn Treebank corpus. Each network has 20 million parameters. The first three rows report results published in (Jozefowicz et al., 2015) and the last two rows provide results for LSTM and CFN networks trained and initialized as was described at the beginning of the section. The CFN has two hidden layers of 731 units each and the LSTM trained by us has two hidden layers of 655 units each. We also tried a one-layer LSTM with 20M parameters and it leads to similar results than the two-layer architecture. For the CFN, we apply  $q = 45\%$  dropout on the connection controlling the gates, and  $p = 55\%$  dropout on all other connections. For the LSTM we apply  $q = 40\%$  dropout on the connection controlling the gates, and  $p = 60\%$  dropout on all other connections. We used the following learning rate schedule for both network:

- For the CFN: start with a learning rate  $\mathrm{lr} = 7$  and divide it by 1.1 each time the validation perplexity did not decrease by at least  $1\%$ .  
- For the LSTM: start with a learning rate  $\mathrm{lr} = 5$  and divide it by 1.1 each time the validation perplexity did not decrease by at least  $1\%$ .

The CFN and LSTM networks clearly outperform the other architectures. In this experiments with dropout, we see again that the simple and non-chaotic CFN performs comparably to the much more complex and chaotic LSTM network.

# REFERENCES

Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

Felix A Gers, Jürgen Schmidhuber, and Fred Cummins. Learning to forget: Continual prediction with LSTM. Neural computation, 12(10):2451-2471, 2000.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.

Rafal Jozefowicz, Wojciech Zaremba, and Ilya Sutskever. An empirical exploration of recurrent network architectures. In Proceedings of the 32nd International Conference on Machine Learning, 2015.  
Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.  
Tomas Mikolov, Armand Joulin, Sumit Chopra, Michael Mathieu, and Marc'Aurelio Ranzato. Learning longer memory in recurrent neural networks. arXiv preprint arXiv:1412.7753, 2014.  
Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. In Advances in neural information processing systems, pp. 2440-2448, 2015.  
Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. arXiv preprint arXiv:1409.2329, 2014.