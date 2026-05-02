# COMPUTING PREIMAGES OF DEEP NEURAL NETWORKS WITH APPLICATIONS TO SAFETY

Anonymous authors

Paper under double-blind review

# ABSTRACT

To apply an algorithm in a sensitive domain it is important to understand the set of input values that result in specific decisions. Deep neural networks suffer from an inherent instability that makes this difficult: different outputs can arise from very similar inputs.

We present a method to check that the decisions of a deep neural network are as intended by constructing the exact, analytical preimage of its predictions. Preimages generalize verification in the sense that they can be used to verify a wide class of properties, and answer much richer questions besides. We examine the functioning and failures of neural networks used in robotics, including an aircraft collision avoidance system, related to sequential decision making and extrapolation.

Our method iterates backwards through the layers of piecewise linear deep neural networks. Uniquely, we compute all intermediate values that correspond to a prediction, propagating this calculation through layers using analytical formulae for layer preimages.

# 1 INTRODUCTION

Folk wisdom holds that although deep neural networks (DNNs) can achieve excellent predictive accuracy, reasoning about their performance is difficult, even for experts. Our goal is to enable non-expert stakeholders, such as clinical health workers, investors, or military commanders to build trust a statistical model in high-stakes environments. To do this, we posit that decisionmakers want to understand a model in both directions, both from inputs to outputs, but also being able to start with hypothetical outputs, and understand the inputs that lead to them.

In this paper, we develop an equivalent, but much simpler, representation of a certain class of DNN classifiers. This representation, which requires only a basic numeracy to productively interact with, can be used by domain experts to build intuition and trust. We apply this method to a reinforcement learning agent trained to solve the cart-pole problem, and find that a DNN implementing a successful policy makes a particular type of mistake on  $24\%$  of the mass of the 1/8th of the state space for which we know the optimal action (Section 3.2). We also show how using the preimage in place of verification can yield a more efficient and interpretable end-to-end system for analyzing aircraft collision avoidance systems (Section 3.3).

# 1.1 PREVIOUS WORK

DNNs have the property that knowing the output tells us very little about the input it corresponds to. This is most apparent in image classifiers, where totally different outputs can arise from inputs that are visually indistinguishable (Szegedy et al. (2014)). We build upon the mathematical framework developed for verifying DNNs that grew out of a desire to prove the absence of adversarial examples, for example Tjeng et al. (2017) and Wong & Kolter (2017). However, we depart from these studies along with Katz et al. (2017), being more oriented towards small DNNs that map to and from low-dimensional spaces with considerable structure. These DNNs arise especially in systems which interoperate with the physical world, for example mapping measurements of positions and velocities to movements. Table 1 orients our work to the literature.

Table 1: A taxonomy of previous work on inversion and verification. Here  $f: \mathbb{R}^{n_1} \to \mathbb{R}^{n_L}$  is a DNN,  $X \subseteq \mathbb{R}^{n_1}$ ,  $x \in \mathbb{R}^{n_1}$ ,  $Y \subseteq \mathbb{R}^{n_L}$ , and  $y \in \mathbb{R}^{n_L}$ .  $f^{-1}$  is its inverse in the sense that  $f^{-1}(Y) = \{x: f(x) \in Y\}$ .  

<table><tr><td>Commonly called</td><td>What is computed</td><td>Examples</td></tr><tr><td>Verification</td><td>(f,X,Y) → 1f-1(Y)∩X=∅(= 1f(X)∩Y=∅)</td><td>Wong &amp; Kolter (2017)</td></tr><tr><td>Reachability</td><td>(f,X) → f(X)</td><td>Yang et al. (2020)</td></tr><tr><td>Inversion</td><td>(f,y) → f-1{y}</td><td>Carlsson et al. (2017)</td></tr><tr><td>Preimage</td><td>(f,Y) → f-1(Y)</td><td>This paper</td></tr></table>

We have phrased verification in this unusual fashion to facilitate comparison with the other points. Stated in the familiar application to image classifiers  $X$  would be an epsilon ball around an input, and  $Y$  would be the halfspace where one coordinate is higher than all others.

Verification ultimately amounts to a simple yes or no, and so answering higher-level questions typically requires many verifications: for example, Katz et al. (2017) describes a suite of 45 tests, and image classifiers often wish to verify the absence of adversarial examples around the entire training set. Yang et al. (2020) is an interesting extension to verification in that it computes the entire image of, say, an epsilon ball around a data point, and not just whether it intersects with a decision boundary.

Reasoning forward, about the outputs that can arise from inputs, is only half of the picture. Carlsson et al. (2017) and Behrmann et al. (2018) are oriented backwards, they attempt to reconstruct the inputs that result in an output. These related papers study the statistical invariances that nonlinear layers encode. Behrmann et al. (2018) examines the preimage of a single point through a single ReLU layer, analyzing stability via an approximation-based experiment. Carlsson et al. (2017) analyzes the preimage of a single point through the repeated application of a nonlinearity, purely theoretically. Our paper looks at the preimage of non-singleton subsets of the codomain, which is much more practically useful, and requires considerable extension to their approaches.

# 2 METHOD

Our method is easily stated: build up the preimage of a DNN from the preimage of its layers, using simple analytical formulae. We start by developing some properties of the preimage operator, then we describe the class of sets that we compute the preimage of, and finally we discuss the class of DNNs that our algorithm addresses.

# 2.1 PROPERTIES OF PREIMAGES

Lemma 1 shows how to build up the preimage of a DNN from the preimages of its constituent layers.

Lemma 1 (Preimage of composition is reversed composition of preimages). For functions  $f_{j} : \mathbb{R}^{n_{j}} \to \mathbb{R}^{n_{j+1}}$ ,

$$
\left(f _ {\ell + k} \circ f _ {\ell + k - 1} \circ \dots \circ f _ {\ell}\right) ^ {- 1} = f _ {\ell} ^ {- 1} \circ \dots \circ f _ {\ell + k - 1} ^ {- 1} \circ f _ {\ell + k} ^ {- 1}. \tag {1}
$$

Secondly, we mention an intuitive property of  $f^{-1}$  that is handy for building up the preimage of any set from the preimages of any partition of that set.

Lemma 2 (Preimage of union is union of preimages).

$$
f ^ {- 1} \left(\cup_ {i = 1} ^ {N} S _ {i}\right) = \cup_ {i = 1} ^ {N} f ^ {- 1} (S _ {i}).
$$

# 2.2 POLYTOPES

Our method is not applicable to arbitrary sets  $Y$ , but rather sets that, roughly, have piecewise linear boundaries. The basic building block of these sets are polytopes.

Definition 1 (Polytope). A polytope in  $\mathbb{R}^n$  is a set that can be written as  $\{x\in \mathbb{R}^n:b - Ax\geq 0\}$  for some  $m\in \mathbb{N},b\in \mathbb{R}^m$  , and  $A\in \mathbb{R}^{m\times n}$

Put more simply: a polytope is the intersection of half-planes. Definition 1 does not require that polytopes be bounded, but polytopes are convex. Sets with linear boundaries, though they may be non-convex, can decomposed into the union of polytopes. We term such sets region-unions, and the set of polytopes which comprise them, regions.

Definition 2 (Region and region-union). For  $N \in \mathbb{N}, b_i \in \mathbb{R}^{m_i}$ ,  $A_i \in \mathbb{R}^{m_i \times n}$ , with  $m_i \in \mathbb{N}$ , a region is

$$
\{\{x: b _ {i} - A _ {i} x \geq 0 \}; i = 1, \dots , N \}. \tag {2}
$$

A region-union is a set  $\cup_{r\in R}r$  for some region  $R$

Region-unions are interesting because the preimage polytopes under piecewise linear functions are regions-unions. However, we need to also keep information on how to form a region-union, hence the notion of a region. It is trivial to observe that if  $R_{1}$  and  $R_{2}$  are regions, then  $R_{1} \cup R_{2}$  is likewise a region, and correspondingly for region-unions.

# 2.3 LINEAR AND RELU POLYTOPE PREIMAGES

In this section, we give formulae for the preimage of linear and ReLU functions, giving significant content to Lemma 1. The preimage of polytopes under linear mappings are polytopes:

Lemma 3 (Preimage of Linear layer).

$$
(x \mapsto W x + a) ^ {- 1} (\{x: b - A x \geq 0 \}) = \{x: (b - A a) - A W x \geq 0 \}. \tag {3}
$$

ReLU is a piecewise linear function, so if we carefully treat the portions of the domain on which it exhibits different behavior, we obtain a similar formulation for each:

Lemma 4 (Preimage of ReLU layer).

$$
\begin{array}{l} \operatorname {R e L U} ^ {- 1} \left(\left\{x: b - A x \geq 0 \right\}\right) \\ = \bigcup_ {\nu \in \{0, 1 \} ^ {n}} \left\{x: b - A \operatorname {d i a g} (\nu) x \geq 0, - \operatorname {d i a g} (1 - \nu) x \geq 0, \operatorname {d i a g} (\nu) x \geq 0 \right\}. \tag {4} \\ \end{array}
$$

To understand Lemma 4 let  $s(x)$  be the vector given by  $s(x)_i = 1$  if  $x_i \geq 0$  and zero otherwise. Then  $\mathrm{diag}(s(x))x = \mathrm{ReLU}(x)$ . This expression separates  $x \mapsto \mathrm{ReLU}(x)$  into a pattern of signs over its coordinates and  $x$  itself. This means that once we restrict attention to a set on which the sign does not change, we can apply familiar linear algebra routines to compute the preimage set, akin to Lemma 3. The nonnegative values are denoted by  $\nu \in \{0,1\}^n$  in the above, and the set of  $x$  such that  $x_i \geq 0 \iff \nu_i = 1$  is given by  $\mathrm{diag}(\nu)x \geq 0$ . Similarly,  $x_i \leq 0 \iff \nu_i = 0$  for  $i = 1,2,\ldots,n$  if and only if  $-\mathrm{diag}(1 - \nu)x \geq 0$ . Equation 4 follows by partitioning  $\mathbb{R}^n$  into the  $2^n$  sets where each coordinate is nonnegative or not.

Computing the preimage of a ReLU layer is unavoidably intractable at scale, though the problem exhibits considerable structure. We expect that it is possible to compute the preimage of networks of a similar scale to those that can be completely verified, such as small image-scale networks. Preimages are most insightful and useful when the inputs and outputs have definite interpretation - application areas where the need for massive networks is less.

# 2.4 THE SUFFICIENCY OF LINEAR AND RELU LAYERS

In familiar terms a DNN classifier might consist of some "feature building" modules, say composed of alternating convolution and maxpooling, then flattened, and passed onto the prediction logic consisting of alternating linear and ReLU layers, possibly including dropout or batch normalization, and concluding with a softmax function to normalize the predictions to a probability distribution. Resnets (He et al. (2016)) do not strictly fit this pattern, but can be handled with similar reasoning (see Appendix B).

How do the results of Section 2.3 suffice to invert such DNNs? Firstly, under our convention that layers operate on flat tensors, flattening is superfluous. Next, dropout affects inference only through the weights - this layer can be omitted entirely in computing the preimage. Convolution is essentially linear. Maxpool is straightforwardly rewritten in terms of the ReLU and linear function.  $\{x:b - A\mathrm{softmax}(x)\geq 0\}$  is not a polytope. However, if the classification alone suffices then the softmax layer can be elided entirely since  $\arg \max_{j}x_{j} = \arg \max_{j}\operatorname {softmax}(x)_{j}$ .

# 3 EXPERIMENTS

# 3.1 TWO MOONS CLASSIFICATION

To cultivate some intuition about the preimage of a DNN we start by examining a classic test problem in nonlinear classification. We fit a DNN  $f:[-3, + 3]^2\to \mathbb{R}^2$  consisting of two nonlinear layers with eight neurons each on an instance of the "two moons" dataset. This data is shown in Figure 1a (further details of details of  $f$  and the data are in Section D.1). Figure 1b plot the corresponding logits, along with the sets to be inverted  $\{x:x_1\leqslant x_2\} \subseteq \mathbb{R}^2$ . Figure 1c shows the corresponding preimages, with different hues of the same color corresponding to different sign patterns  $\nu$  in Equation 4.

![](images/a2254846ab082f7d6c08743be6bda881d64d0a6ca7a4229351f522a22798e2d9.jpg)  
(a)

![](images/b9e79352d159a8f55a0bbfdee6a0b4a6253e02d7370fcdedb89dcfec25a7d262.jpg)  
(b)

![](images/32eeb071ccf35189292029f28ce67118e22ed6ec82eb6f448ad3642096ea2389.jpg)  
Figure 1: Inversion of a simple DNN  $\mathbb{R}^2\mapsto \mathbb{R}^2$  fit to the "two moons" data shown in Figure 1a. In Figure 1b are the logits from a simple DNN corresponding to each data point computed with an inference pass, along with the decision boundary. Figure 1c shows the preimages comprised of polytopes that form the region-union.  
(c)

# 3.2 CART-POLE REINFORCEMENT LEARNING AGENT

In the "cart pole" control problem a pole is balanced atop a cart which moves along a one-dimensional track (Figure 2). Gravity pulls the pole downward, the falling of the pole pushes the cart, and external movement of the cart pushes the pole in turn. The control problem is to keep the pole upright by accelerating the cart.

In the formulation of Brockman et al. (2016) controller inputs are: the position of the cart,  $x$ , velocity of the cart  $\dot{x}$ , the angle of the pole  $\theta$  from upright, and the angular velocity of the pole  $\dot{\theta}$ . Possible actions are to accelerate the cart in the positive or negative  $x$  direction. The reward environment encourages balancing by a unit reward per period before failure, where failure means that the pole is not sufficiently upright ( $\theta \notin [-\pi / 15, +\pi / 15]$ ), or the cart not near enough the origin ( $x \notin [-2.4, +2.4]$ ). We have no prescribed limits for  $\dot{x}$  and  $\dot{\theta}$ , but via a methodology described in Section D.2.1, we interpret these states as taking values in  $[-3.0, +3.0] \times [-3.5, +3.5]$ .

Consider a still cart and pole  $(\dot{x} = \dot{\theta} = 0)$ , with the cart left of zero  $(x \leq 0)$  and the pole left of vertical  $(\theta \leq 0)$ . Keeping  $x$  and  $\theta$  near zero is preferable, since these are further from failure, so moving left will steady  $\theta$  but worsen  $x$ . Nonzero velocities make this reasoning more complicated, but one configuration is unambiguous: if  $x \leq 0, \dot{x} \leq 0, \theta \geq 0, \dot{\theta} \geq 0$ , then pushing right is clearly

![](images/3baad4194f10d9d3bfa5879549ff44525c132cccf4a48891f0ad29be57894f14.jpg)  
Figure 2: The state space of the cart pole problem, schematically. Here  $x \leq 0$  (the cart is left of the origin),  $\dot{x} \leq 0$  (the cart is moving leftward),  $\theta \geq 0$  (the pole is right of vertical), and  $\dot{\theta} \geq 0$  (the pole is moving rightward).

![](images/24059bb077c9ef9cb9c5180108877bc3d92ad46f592cfb7e8e75a65ff4359da7.jpg)  
Figure 3: Projection of subsets of the domain where the wrong action is taken, with the hue of the area being proportional to the volume of the wrong sets, divided by the volume of the projection.

the correct action. Figure 2 gives depicts a value in this orthant. Let  $D_{+1} = (-\infty, 0]^2 \times [0, \infty)^2$  and correspondingly, let  $D_{-1} = [0, +\infty)^2 \times (-\infty, 0]^2$ .

We fit a one hidden layer neural network control function  $f: \mathbb{R}^4 \to \mathbb{R}^2$  using policy gradient reinforcement learning. Details of this calculation are in Section D.2. This agent achieves its goal of balancing the pole: in 1000 trials of 200 periods,  $(x, \theta)$  remains uniformly in  $[-.75, +.75] \times [-.05, +.05]$  with very low velocities. Nonetheless there are many states for which pushing right is clearly the correct action, but for which the DNN controller predicts  $-1$ : in the same simulation of 1000 trials of 200 steps, roughly  $7\%$  of actions performed by the agent fail this sanity check. This behavior is not a numerical fluke – it holds if we consider states only nonnegligibly interior to  $D_{+1}$  and  $D_{-1}$ , and also if we only count predictions that are made with probability greater than .51. One such pockets of counterintuitive behavior is

$$
[ - 2. 3 9 9, - 1. 4 6 2 ] \times [ - 2. 9 2 2, - 2. 2 6 2 ] \times [ + 1. 7 9 8 \times 1 0 ^ {- 8}, + 0. 1 0 6 7 ] \times [ + 1. 3 9 9, + 1. 7 2 8 ] \subseteq
$$

$$
D _ {+ 1} \cap f ^ {- 1} (\{x \in \mathbb {R} ^ {2}: x _ {1} > x _ {2} \}).
$$

We find this box large – for example the first coordinate comprises almost  $20\%$  of that dimension of the state space. The size of this box is even more surprising because it is inscribed within a larger polytope (using the algorithm of Temporad et al. (2004)) that has a volume about 40 times larger. The total volume in  $\mathbb{R}^4$  of these sets is  $3\%$  of the state space volume, and thus  $24\%$  of the volume of  $D_{-1}\cup D_{+1}$ . Figure 3 parses this surprising fact a bit further by plotting the projection of the four-dimensional domain onto the  $(x,\theta)$  plane. The hue of the gray is proportional to the volume of the four-dimensional polytope divided by the volume of the two-dimensional projection, so darker areas mean more  $(\dot{x},\dot{\theta})$  mass that is wrong. Since the entirety of the second and fourth quadrants are grey at every  $(x,\theta)\in [-2.4, + 2.4]\times [-\pi /15, + \pi /15]$  there are some  $(\dot{x},\dot{\theta})$  where the wrong action will be taken.

# 3.3 COLLISION AVOIDANCE SYSTEMS

The final application shows how to use domain knowledge to anticipate dangerous behavior of a DNN in a complex modelling domain.

# 3.3.1 BACKGROUND

Aircraft automated collision avoidance systems (ACAS) are navigational aids that use data on aircraft positions and velocities to issue guidance on evasive actions to prevent collisions with an intruding aircraft. The ACAS developed in Kochenderfer & Chryssanthacopoulos (2011) uses dynamic programming to formulate the optimal control of a partially observed Markov process, and issues advisories to optimize a criterion that penalizes near collisions and raising false or inconsistent warnings. Unfortunately, evaluating the policy function is too resource-intensive to run on certified avionics hardware. Small DNNs have been found to be adequate approximators that require little storage and can perform inference quickly. A downside of this approximation is that even accurate DNNs can give very wrong predictions on some inputs - Katz et al. (2017), for example show that when another aircraft is nearby and approaching from the left, a DNN-based approximation need not advise the correct action of turning right aggressively.

Verification can check that one-step behavior in a DNN-based ACAS behaves as intended. However, it cannot answer higher level questions like "will a near-collision occur if this policy is followed?" The idea of Julian & Kochenderfer (2019) is to verify dynamic properties of such systems by combining single-step verification with worst-case assumptions about randomness in state transitions and (constrained) behavior of other aircraft.

# 3.3.2 DISCRETIZE AND VERIFY: JULIAN & KOCHERDERFER (2019)

In Julian & Kochenderfer (2019), the state consists of  $x$  and  $y$  distances between the two aircraft, and an angle of approach angle between them,  $\psi$ . The actions are five turning advisories: (1) "clear of conflict" (COC), (2) weak left [turn] (WL), (3) strong left (SL), (4) weak right (WR), and (5) strong right (SR). The initial condition is given by the boundary of the domain where the distance of the intruding aircraft are at their maxima. Transition dynamics are denoted by  $\Psi(a, S)$ , a set-valued function which gives the set of states that are reachable from states in  $S$  under action  $a$ .  $\Psi$  encompasses both randomness in the transition, and behavior of the other aircraft. The change in  $(x, y)$  is controlled by the angle between the crafts, and the update to the angle is the difference between the turning of the two crafts, with some randomness. To compute the states that can arise under a policy, the idea is to begin from an initial set of states that are known to be reachable, and to iteratively append states that are reachable from any of those the set of states that we wish to preclude.

Table 2: Quantifying the inefficiency of discretization: Each of the three dimensions,  $(x,y,\psi)$  is discretized into a grid of size  $g$ , so that the domain is partitioned into  $g^3$  cubes. We present the fraction of the cubes for which all eight corners of this cube do not evaluate to the same prediction, which is a sufficient condition for the cell to intersect with a decision boundary.  

<table><tr><td>g</td><td>Volume fraction</td></tr><tr><td>40</td><td>0.05128</td></tr><tr><td>80</td><td>0.02532</td></tr><tr><td>120</td><td>0.01681</td></tr><tr><td>160</td><td>0.01267</td></tr><tr><td>200</td><td>0.01005</td></tr></table>

This idea is formalized by Julian & Kochenderfer (2019) as Algorithm 1. Because multiple advisories will be issued whenever a cell straddles the decision boundary, the discretized algorithm will wrongly include some states as reachable since a worst-case analysis needs to take account of all reachable states. Table 2 gives an indication of the magnitudes of overestimation, presenting how much of the state space will lead to multiple advisories under a simple discretization scheme.

Julian & Kochenderfer (2019) do not use an equispaced grid, but the basic point – that discretization error cannot be made negligible – is an inescapable feature of this approach. And any false positives in a single-step decision function will be amplified in the dynamic analysis, as more reachable states at one point time lead to even more reachable points at the next step, so a  $1\%$  overestimation at one step may be compounded to considerably more through the dynamics. Coincidentally, Julian & Kochenderfer (2019) are able to reach a usable solution, but are unable to guarantee the absence of near collisions under some realistic parameter configurations.

Note how the cells can be traversed in any order. This is a simple way to see that this algorithm is not fully using the spatial structure of the problem. Next, we incorporate this knowledge.

Data: Maximum distance set  $\mathcal{R}_0$ , policy  $f$ , an "unsafe set"  $U$ , transition dynamics  $\Psi$ , encounter length  $T$ .

Result: Guaranteed to not reach an unsafe state from  $\mathcal{R}_0$  under policy  $f$ ?

initialization:  $t = 0$ , done = False;

Partition the state space into cells  $c\in \mathcal{C}$

while not done do

```latex
$t = t + 1$ $\mathcal{R}_t = \emptyset$    
for  $c\in \mathcal{C}$  such that  $c\cap \mathcal{R}_{t - 1}\neq \emptyset$  do for  $i$  such that  $f(c)\cap \{x:x_i\geq x_j$  for  $j\neq i\} \neq \emptyset$  do for  $c^{\prime}\in \mathcal{C}$  such that  $c^\prime \cap \Psi (i,c)\neq \emptyset$  do  $\mid \mathcal{R}_t\gets \mathcal{R}_t\cup c'$  end end  
end done  $= \mathcal{R}_t == \mathcal{R}_{t - 1}$  or  $U\cap \mathcal{R}_t\neq \emptyset$  or  $t > T$
```

end

Return  $\mathcal{R}_t\cap U = = \emptyset$

Algorithm 1: Algorithm from Julian & Kochenderfer (2019) for computing whether an unsafe set  $U$  can be reached under a policy  $f$  beginning from  $\mathcal{R}_0$  under transition dynamics  $\Psi$ .

# 3.3.3 OUR PREIMAGE-BASED ALTERNATIVE

Rather than looping first the domain, then over actions at those points, Algorithm 2 loops over actions and, using the preimage, computes all reachable points under that action.

Data:  $\mathcal{R}_0,f,U,\Psi ,T$

Result: Guaranteed to not reach an unsafe state from  $\mathcal{R}_0$  under policy  $f$ ?

initialization:  $t = 0$ , done  $=$  False;

```txt
for  $i = 1,2,\ldots ,n_L$  do  $\Xi_{i} = f^{-1}(\{x:x_{i}\geq x_{j}\mathrm{for}j\neq i\})$
```

end

while not done do

```txt
$t = t + 1$ $\mathcal{R}_t = \emptyset$  for  $i = 1,2,\ldots ,n_L$  do  $\begin{array}{rl} & {\mathcal{R}_t\leftarrow \mathcal{R}_t\cup \Psi (i,\Xi_i\cap \mathcal{R}_{t - 1});}\\ & {\mathrm{end}}\\ & {\mathrm{done} = \mathcal{R}_t == \mathcal{R}_{t - 1}\mathrm{or}U\cap \mathcal{R}_t\neq \emptyset \mathrm{or}t > T.} \end{array}$  end
```

end

Return  $U\cap \mathcal{R}_t = = \emptyset$

Algorithm 2: Our preimage-based, exact algorithm for computing the dynamically reachable states in an ACAS.

While Algorithm 2 is exact - it will never wrongly say that a state can be reached - the accuracy of Algorithm 1 is ultimately controlled by the number of cells,  $|\mathcal{C}|$ . This is because it is necessary to perform  $n_{L}$  verifications for each reachable cell, and the number of reachable cells is proportional to  $|\mathcal{C}|$ . Let  $V$  denote the cost of a verification. Verification is known to be NP-complete (Katz et al. (2017)), so  $V$  dominates all others calculation such as computing intersections or evaluating  $\Psi(i, c)$ . Thus, the computational cost of Algorithm 1 is  $O(|\mathcal{C}| V n_{L})$ . In Algorithm 2 must initially compute  $n_{L}$  preimages which dominates the entire calculation, which consists of relatively fast operations - applying the dynamics and computing intersections up to  $T$  times, for  $T$  a number around 40.

Let  $P$  denote the cost of computing a preimage, then Algorithm 2 is  $O(Pn_{L})$ . So whilst it dispenses with the need to solve  $O(|\mathcal{C}|)$  verifications, but may be more intractable if  $P$  is significantly higher than  $V$ . Let the dimensions of the nonlinear layers in a DNN be  $n_{\ell_i}$ , then because in the worst case it is necessary to check each nonlinearity, each of which can be independently in a negative or positive configuration,  $V = O(2^{\sum_{i}n_{\ell_i}})$ . Exact verification for even a single cell is impossible at present for

![](images/aac111e30a9b7ec6487788a4495bb8f3a4f2b6c3bb14d00a6ae1a1928631f39f.jpg)  
Figure 4: An encounter plot showing the optimal action at each  $(x,y)$  distance configuration for a fixed angle of approach, indicated by the perpendicular orientation of the red (intruder) aircraft. Distances are measured in kilofeet.

large networks. We believe that preimages can be computed roughly (within a small constant factor) as easily as a verification  $-P = O(V)$ . We are currently developing this conjecture formally, the idea is that, as shown in Lemma 4, each nonlinear layer  $\ell_i$  generates up to  $2^{n_{\ell_i}}$  sets, the preimage of which must be computed through earlier layers.

In any case, as is true of any exponentially hard problem, the practical tractability of both  $P$  and  $V$  hinges importantly upon theoretical arguments showing that not all  $2^{n}$  configurations of the nonlinearities of an  $n$ -dimensional layer can be achieved (Serra et al. (2017); Hanin & Rolnick (2019), and clever implementations that take account of the structure of the problems (e.g. Tjeng et al. (2017); Katz et al. (2017)).

The distinction between the two algorithms is made clearer by examining an encounter plot such as Figure 4. Encounter plots are concise summarizations of the policy function, here depicting the possible advisories, for a fixed angle of approach (which is here conveyed by the orientation of the red aircraft relative to the black). This figure, which replicates Figure 4 of Julian & Kochenderfer (2019), differs from it in a crucial respect: it is depicts the analytically-computed preimage of the five sets where each of the advisories are issued (details of the experiment are in Section D.3). The shaded areas arise from plotting polytopes, as in Algorithm 2. Julian & Kochenderfer (2019), on the other hand, produce such plots by evaluating the predictions of the network on a fine grid. The different manner in which the plots are produced is an exact analogue of the different way that the networks are summarized and analyzed through time.

# 4 CONCLUSION

In many areas, safety and interpretation inhibit the use of DNNs, because their use still requires a good deal of indirect experimentation and oversight to have confidence that it will not act in an unintuitive way. This paper has proposed computing the preimage of the decisions of a DNN as an intuitive diagnostic that can help to anticipate problems and help domain experts gain trust in a DNN, even if they are unable to formally articulate what makes a DNN trustworthy. In order to do this, we developed the preimage of a DNN and presented an algorithm to compute it. We demonstrated the utility of the preimage to understand counterintuitive behavior from a cart pole agent, and to more precisely characterize the set of states that would be reachable in an existing application of DNNs to aircraft automated collision avoidance systems.

# REFERENCES

David Avis. A Revised Implementation of the Reverse Search Vertex Enumeration Algorithm, pp. 177-198. Birkhäuser Basel, Basel, 2000. URL https://doi.org/10.1007/978-3-0348-8438-9_9.  
C. Bradford Barber, David P. Dobkin, David P. Dobkin, and Hannu Huhdanpaa. The quickhull algorithm for convex hulls. ACM Trans. Math. Softw., 22(4):469-483, December 1996. ISSN 0098-3500. URL http://doi.acm.org/10.1145/235815.235821.  
Jens Behrmann, Soren Dittmer, Pascal Fernsel, and Peter Maaß. Analysis of Invariance and Robustness via Invertibility of ReLU-Networks. arXiv e-prints, Jun 2018. URL http://arxiv.org/abs/1806.09730.  
Alberto Temporad, Carlo Filippi, and Fabio D. Torrisi. Inner and outer approximations of polytopes using boxes. Computational Geometry, 27(2):151 - 178, 2004. URL https://doi.org/10.1016/S0925-7721(03)00048-8.  
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. 2016. URL http://arxiv.org/abs/1606.01540.  
Benno Büeler, Andreas Enge, and Komei Fukuda. *Exact Volume Computation for Polytopes: A Practical Study*, pp. 131-154. Birkhäuser Basel, Basel, 2000. URL https://doi.org/10.1007/978-3-0348-8438-9_6.  
Stefan Carlsson, Hossein Azizpour, and Ali Sharif Razavian. The preimage of rectifier network activities. 2017. URL https://openreview.net/pdf?id=HJcLcw9xg.  
Komei Fukuda. Lecture: Polyhedral computation, 2014. URL https://inf.ethz.ch/personal/fukudak/lect/pclect/notes2014/PolyComp2014.pdf.  
Komei Fukuda and Alain Prodon. Double description method revisited. In Michel Deza, Reinhardt Euler, and Ioannis Manoussakis (eds.), Combinatorics and Computer Science, pp. 91-111, Berlin, Heidelberg, 1996. Springer Berlin Heidelberg. URL https://doi.org/10.1007/3-540-61576-8_77.  
Boris Hanin and David Rolnick. Deep ReLU Networks Have Surprisingly Few Activation Patterns. arXiv e-prints, Jun 2019. URL https://arxiv.org/abs/1906.00904.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016. URL https://doi.org/10.1109/CVPR.2016.90.  
Kyle Julian, Jessica Lopez, Jeffrey Brush, Michael Owen, and Mykel Kochenderfer. Policy compression for aircraft collision avoidance systems. In 2016 IEEE/AIAA 35th Digital Avionics Systems Conference (DASC), pp. 1-10, 09 2016. URL https://doi.org/10.1109/DASC.2016.7778091.  
Kyle D. Julian and Mykel J. Kochenderfer. Guaranteeing safety for neural network-based aircraft collision avoidance systems. IEEE/AIAA 38th Digital Avionics Systems Conference (DASC), 2019. URL https://doi.org/10.1109/DASC43569.2019.9081748.  
Guy Katz, Clark Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In Rupak Majumdar and Viktor Kunçak (eds.), Computer Aided Verification. Springer International Publishing, 2017. URL https://doi.org/10.1007/978-3-319-63387-9_5.  
Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv e-prints, Dec 2014. URL http://arxiv.org/abs/1412.6980.  
Mykel J. Kochenderfer and James P. Chryssanthacopoulos. Robust Airborne Collision Avoidance through Dynamic Programming. Massachusetts Institute of Technology, Lincoln Laboratory, Project Report ATC-371, 2011. URL https://www.ll.mit.edu/sites/default/files/publication/doc/2018-12/Kochenderfer_2011_ATC-371_WW-21458.pdf.

Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and Counting Linear Regions of Deep Neural Networks. arXiv e-prints, Nov 2017. URL http://arxiv.org/abs/1711.02114.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna Estrach, Dumitru Erhan, Ian Goodfellow, and Robert Fergus. Intriguing properties of neural networks. January 2014. URL http://arxiv.org/abs/1312.6199. 2nd International Conference on Learning Representations, ICLR 2014; Conference date: 14-04-2014 Through 16-04-2014.  
Vincent Tjeng, Kai Xiao, and Russ Tedrake. Evaluating Robustness of Neural Networks with Mixed Integer Programming. arXiv e-prints, Nov 2017. URL http://arxiv.org/abs/1711.07356.  
Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. arXiv e-prints, Nov 2017. URL http://arxiv.org/abs/1711.00851.  
Kai Y. Xiao, Vincent Tjeng, Nur Muhammad Shafiullah, and Aleksander Madry. Training for Faster Adversarial Robustness Verification via Inducing ReLU Stability. *ICLR* 2019, Sep 2018. URL http://arxiv.org/abs/1809.03008.  
Xiaodong Yang, Hoang-Dung Tran, Weiming Xiang, and Taylor Johnson. Reachability Analysis for Feed-Forward Neural Networks using Face Lattices. arXiv e-prints, art. arXiv:2003.01226, 2020. URL https://arxiv.org/abs/2003.01226.  
Hao Zhou, Jose M. Alvarez, and Fatih Porikli. Less is more: Towards compact CNNs. In European Conference on Computer Vision, pp. 662-677, Amsterdam, the Netherlands, October 2016. URL https://doi.org/10.1007/978-3-319-46493-0_40.
