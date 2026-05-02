# EFFICIENTLY TESTING LOCAL OPTIMALITY AND ESCAPING SADDLES FOR RELU NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We provide a theoretical algorithm for checking local optimality and escaping saddles at nondifferentiable points of empirical risks of two-layer ReLU networks. Our algorithm receives any parameter value and returns: local minimum, second-order stationary point, or a strict descent direction. The presence of  $M$  data points on the nondifferentiability of the ReLU divides the parameter space into at most  $2^{M}$  regions, which makes analysis difficult. By exploiting polyhedral geometry, we reduce the total computation down to one convex quadratic program (QP) for each hidden node,  $O(M)$  (in)equality tests, and one (or a few) nonconvex QP. For the last QP, we show that our specific problem can be solved efficiently, in spite of nonconvexity. In the benign case, we solve one equality constrained QP, and we prove that projected gradient descent solves it exponentially fast. In the bad case, we have to solve a few more inequality constrained QPs, but we prove that the time complexity is exponential only in the number of inequality constraints. Our experiments show that either benign case or bad case with very few inequality constraints occurs, implying that our algorithm is efficient in most cases.

# 1 INTRODUCTION

Empirical success of deep neural networks has sparked great interest in the theory of deep models. From an optimization viewpoint, the biggest mystery is that deep neural networks are successfully trained by gradient-based algorithms despite their nonconvexity. On the other hand, it has been known that training neural networks to global optimality is NP-hard (Blum & Rivest, 1988). It is also known that even checking local optimality of nonconvex problems can be NP-hard (Murty & Kabadi, 1987). Bridging this gap between theory and practice is a very active area of research.

There have been many attempts to understand why optimization works well for neural networks, by studying the loss surface (Baldi & Hornik, 1989; Yu & Chen, 1995; Kawaguchi, 2016; Soudry & Carmon, 2016; Nguyen & Hein, 2017a,b; Safran & Shamir, 2017; Laurent & von Brecht, 2017; Yun et al., 2018a,b; Zhou & Liang, 2018; Wu et al., 2018; Shamir, 2018) and the role of (stochastic) gradient-based methods (Tian, 2017; Brutzkus & Globerson, 2017; Du et al., 2017). Besides nonconvexity, for ReLU networks significant additional challenges in the analysis arise due to non-differentiability, and obtaining a precise understanding of the nondifferentiable points is still elusive.

Nondifferentiable points lie in a set of measure zero, so one may be tempted to overlook them as "non-generic." However, when studying critical points we cannot do so, as they are precisely such "non-generic" points. Laurent & von Brecht (2017) study one-hidden-layer ReLU networks with hinge loss and note that except for piecewise constant regions, local minima always occur on nonsmooth boundaries. Probably due to difficulty in analysis, there have not been other works that handle such nonsmooth points of losses and prove results that work for all points. Some theorems (Soudry & Carmon, 2016; Nguyen & Hein, 2017b) hold "almost surely"; some assume differentiability or make statements only for differentiable points (Nguyen & Hein, 2017a; Yun et al., 2018a); others analyze population risk, in which case the nondifferentiability disappears after taking expectation (Tian, 2017; Brutzkus & Globerson, 2017; Du et al., 2017; Safran & Shamir, 2017; Wu et al., 2018).

# 1.1 SUMMARY OF OUR RESULTS

In this paper, we take a step towards understanding nondifferentiable points of the empirical risk of one-hidden-layer ReLU(-like) networks. Specifically, we provide a theoretical algorithm that tests second-order stationarity (SOS) for any point of the loss surface. It takes an input point and returns:

(a) The point is a local minimum; or  
(b) The point is a second-order stationary point (SOSP); or  
(c) A descent direction in which the function value strictly decreases.

Therefore, we can test whether a given point is a SOSP. If not, the test extracts a guaranteed direction of descent that helps continue minimization; or even escape from saddle points! What makes it nontrivial is that unlike Hessian based methods for escaping saddles, we do not have differentiability.

The key computational challenge in constructing our algorithm for nondifferentiable points is posed by data points that lie on the "boundary" of a hidden neuron. Since each such data point bisects the parameter space into two halfspaces with different "slopes" of the loss surface, one runs into nondifferentiability. If there are  $M$  such boundary data points, then in the worst case the parameter space divides into  $2^M$  regions, so naively testing each region will be very inefficient. In our algorithm, we overcome this issue by a clever use of polyhedral geometry. Another challenge comes from the second-order test, which involves solving nonconvex QPs. Although QP is NP-hard in general (Pardalos & Vavasis, 1991), we prove that the QPs in our algorithm are still solved efficiently in most cases. We further describe the challenges and key ideas in Section 2.1.

Remarks. Many practitioners of deep learning rely on first-order methods, without good termination criteria for the optimization problem. Our algorithm proposes a tool for improvement: with a proper numerical implementation (although we leave numerical implementation to future work), it can test whether a given point is a SOSP, or extract a descent direction using second-order information. One can imagine running a first-order method until it "gets stuck," then using our algorithm to test SOS or escape from the saddle. This idea of mixing first and second-order methods has been explored in differentiable problems (Carmon et al., 2016; Reddi et al., 2017; Mokhtari et al., 2018).

Notation. For a vector  $v$ ,  $[v]_i$  denotes its  $i$ -th component, and  $\| v \|_H \coloneqq \sqrt{v^T H v}$  denotes a semi-norm where  $H$  is a positive semidefinite matrix. Given a matrix  $A$ , we let  $[A]_{i,j}$ ,  $[A]_{i,\cdot}$ , and  $[A]_{\cdot,j}$  be  $A$ 's  $(i,j)$ -th entry, the  $i$ -th row, and the  $j$ -th column, respectively.

# 2 PROBLEM SETTING AND KEY IDEAS

We consider a one-hidden-layer neural network with input dimension  $d_x$ , hidden layer width  $d_h$ , and output dimension  $d_y$ . We are given  $m$  pairs of data points and labels  $(x_i, y_i)_{i=1}^m$ , where  $x_i \in \mathbb{R}^{d_x}$  and  $y_i \in \mathbb{R}^{d_y}$ . Given an input vector  $x$ , the output of the network is defined as  $Y(x) := W_2 h(W_1 x + b_1) + b_2$ , where  $W_2 \in \mathbb{R}^{d_y \times d_h}$ ,  $b_2 \in \mathbb{R}^{d_y}$ ,  $W_1 \in \mathbb{R}^{d_h \times d_x}$ , and  $b_1 \in \mathbb{R}^{d_h}$  are the network parameters. The activation function  $h$  is "ReLU-like," meaning  $h(t) := \max\{s_+ x, 0\} + \min\{s_- x, 0\}$ , where  $s_+ > 0, s_- \geq 0$  and  $s_+ \neq s_-$ . Note that ReLU and Leaky-ReLU are members of this class. In training neural networks, we are interested in minimizing the empirical risk

$$
\mathfrak {R} ((W _ {j}, b _ {j}) _ {j = 1} ^ {2}) = \sum_ {i = 1} ^ {m} \ell (Y (x _ {i}), y _ {i}) = \sum_ {i = 1} ^ {m} \ell (W _ {2} h (W _ {1} x _ {i} + b _ {1}) + b _ {2}, y _ {i}),
$$

over the parameters  $(W_{j},b_{j})_{j = 1}^{2}$ , where  $\ell (w,y):\mathbb{R}^{d_y}\times \mathbb{R}^{d_y}\mapsto \mathbb{R}$  is the loss function. We make the following assumptions on the loss function and the training dataset:

Assumption 1. The loss function  $\ell (w,y)$  is twice continuously differentiable and convex in  $w$ .

Assumption 2. No  $d_x + 1$  data points lie on the same affine hyperplane.

Assumption 1 is satisfied by many standard loss functions such as squared error loss and cross-entropy loss. Assumption 2 means, if  $d_{x} = 2$  for example, no three data points are on the same line. Since real-world datasets contain noise, this assumption is also quite mild.

# 2.1 CHALLENGES AND KEY IDEAS

In this section, we explain the difficulties at nondifferentiable points and ideas on overcoming them. Our algorithm is built from first principles, rather than advanced tools from nonsmooth analysis.

Bisection by boundary data points. Since the activation function  $h$  is nondifferentiable at 0, the behavior of data points at the "boundary" is decisive. Consider a simple example  $d_h = 1$ , so  $W_1$  is a row vector. If  $W_1 x_i + b_1 \neq 0$ , then the sign of  $(W_1 + \Delta_1) x_i + (b_1 + \delta_1)$  for any small perturbations  $\Delta_1$  and  $\delta_1$  stays invariant. In contrast, when there is a point  $x_i$  on the "boundary," i.e.,  $W_1 x_i + b_1 = 0$ ,

then the slope depends on the direction of perturbation, leading to nondifferentiability. We refer to such points as boundary data points. When  $\Delta_1 x_i + b_1 \geq 0$ ,

$h((W_1 + \Delta_1)x_i + (b_1 + \delta_1)) = h(\Delta_1x_i + \delta_1) = s_+(\Delta_1x_i + \delta_1) = h(W_1x_i + b_1) + s_+(\Delta_1x_i + \delta_1)$ , and similarly, the slope is  $s_{-}$  for  $\Delta_1x_i + b_1\leq 0$ . This means that the "gradient" (as well as higher order derivatives) of  $\Re$  depends on direction of  $(\Delta_1,\delta_1)$ .

Thus, every boundary data point  $x_{i}$  defines a hyperplane through the origin, and bisects the parameter space into two halfspaces. The situation is even worse if we have  $M$  boundary data points: they lead to a worst case of  $2^{M}$  regions. Does it mean that we need to test all  $2^{M}$  regions separately? We show that there is a way to remedy this problem, but before that, we first describe how to test local minimality or stationarity for each region.

Second-order local optimality conditions. We can expand  $\Re ((W_j + \Delta_j,b_j + \delta_j)_{j = 1}^2)$  and obtain the following Taylor-like expansion for small enough perturbations (see Lemma 2 for details)

$$
\Re (z + \eta) = \Re (z) + g (z, \eta) ^ {T} \eta + \frac {1}{2} \eta^ {T} H (z, \eta) \eta + o \left(\| \eta \| ^ {2}\right), \tag {1}
$$

where  $z$  is a vectorized version of all parameters  $(W_{j}, b_{j})_{j=1}^{2}$  and  $\eta$  is the corresponding vector of perturbations. Notice now that in (1), at nondifferentiable points the usual Taylor expansion does not exist, but the corresponding "gradient"  $g(\cdot)$  and "Hessian"  $H(\cdot)$  now depend on the direction of perturbation  $\eta$ . Also, the space of  $\eta$  is divided into  $O(2^{M})$  regions, and  $g(z, \eta)$  and  $H(z, \eta)$  are piecewise-constant functions of  $\eta$ , constant inside each region. One could view this problem as  $2^{M}$  constrained optimization problems and try to solve for KKT conditions at  $z$ ; however, we provide an approach that is developed from first principles and solves all  $2^{M}$  problems efficiently.

Given this expansion (1) and the observation that derivatives stay invariant with respect to scaling of  $\eta$ , one can note that (a)  $g(z,\eta)^T\eta \geq 0$  for all  $\eta$ , and (b)  $\eta^T H(z,\eta)\eta \geq 0$  for all  $\eta$  such that  $g(z,\eta)^T\eta = 0$  are necessary conditions for local optimality of  $z$ , thus  $z$  is a "SOSP" (see Definition 2.2). The conditions become sufficient if (b) is replaced with  $\eta^T H(z,\eta)\eta > 0$  for all  $\eta \neq \mathbf{0}$  such that  $g(z,\eta)^T\eta = 0$ . In fact, this is a generalized version of second-order necessary (or sufficient) conditions, i.e.,  $\nabla f = \mathbf{0}$  and  $\nabla^2 f \succeq \mathbf{0}$  (or  $\nabla^2 f \succ \mathbf{0}$ ), for differentiable  $f$ .

Efficiently testing SOSP for exponentially many regions. Motivated from the second-order expansion (1) and necessary/sufficient conditions, our algorithm consists of three steps:

(a) Testing first-order stationarity (in the Clarke sense, see Definition 2.1),  
(b) Testing  $g(z,\eta)^T\eta \geq 0$  for all  $\eta$  
(c) Testing  $\eta^T H(z,\eta)\eta \geq 0$  for  $\{\eta \mid g(z,\eta)^T\eta = 0\}$ .

The tests are executed from Step (a) to (c). Whenever a test fails, we get a strict descent direction  $\eta$ , and the algorithm returns  $\eta$  and terminates. Below, we briefly outline each step and discuss how we can efficiently perform the tests. We first check first-order stationarity because it makes Step (b) easier. Step (a) is done by solving one convex QP per each hidden node. For Step (b), we formulate linear programs (LPs) per each  $2^{M}$  region, so that checking whether all LPs have minimum cost of zero is equivalent to checking  $g(z,\eta)^T\eta \geq 0$  for all  $\eta$ . Here, the feasible sets of LPs are pointed polyhedral cones, whereby it suffices to check only the extreme rays of the cones. It turns out that there are only  $2M$  extreme rays, each shared by  $2^{M-1}$  cones, so testing  $g(z,\eta)^T\eta \geq 0$  can be done with only  $O(M)$  inequality/equality tests instead of solving exponentially many LPs. In Step (b), we also record the flat extreme rays, i.e., those with  $g(z,\eta)^T\eta = 0$ , for later use in Step (c).

In Step (c), we test if the second-order perturbation  $\eta^T H(\cdot)\eta$  can be negative, for directions where  $g(z,\eta)^T\eta = 0$ . Due to the constraint  $g(z,\eta)^T\eta = 0$ , the second-order test requires solving constrained nonconvex QPs. In case where there is no flat extreme ray, we need to solve only one equality constrained QP (ECQP). If there exist flat extreme rays, a few more inequality constrained QPs (ICQPs) are solved. Despite NP-hardness of general QPs (Pardalos & Vavasis, 1991), we prove that the specific form of QPs in our algorithm are still tractable in most cases. More specifically, we prove that projected gradient descent on ECQPs converges/diverges exponentially fast, and each step takes  $O(p^2)$  time ( $p$  is the number of parameters). In case of ICQPs, it takes  $O(p^3 + L^32^L)$  time to solve the QP, where  $L$  is the number of boundary data points that have flat extreme rays ( $L \leq M$ ). Here, we can see that if  $L$  is small enough, the ICQP can still be solved in polynomial time in  $p$ . At the end of the paper, we provide empirical evidences that the number of flat extreme rays is zero or very few, meaning that in most cases we can solve the QP efficiently.

# 2.2 PROBLEM-SPECIFIC NOTATION AND DEFINITION

In this section, we define a more precise notion of generalized stationary points and introduce some additional symbols that will be helpful in streamlining the description of our algorithm in Section 3. Since we are dealing with nondifferentiable points of nonconvex  $\Re$ , usual notions of (sub)gradients do not work anymore. Here, Clarke subdifferential is a useful generalization (Clarke et al., 2008):

Definition 2.1 (FOSP, Theorem 6.2.5 of Borwein & Lewis (2010)). Suppose that a function  $f(z) : \Omega \mapsto \mathbb{R}$  is locally Lipschitz around the point  $z^{*} \in \Omega$ , and differentiable in  $\Omega \setminus \mathcal{W}$  where  $\mathcal{W}$  has Lebesgue measure zero. Then the Clarke differential of  $f$  at  $z^{*}$  is

$$
\partial_ {z} f (z ^ {*}) := \operatorname {c v x h u l l} \left\{\lim  _ {k} \nabla f (z _ {k}) \mid z _ {k} \rightarrow z ^ {*}, z _ {k} \notin \mathcal {W} \right\}.
$$

If  $\mathbf{0} \in \partial_z f(z^*)$ , we say  $z^*$  is a first-order stationary point (FOSP).

From the definition, we can note that Clarke subdifferential  $\partial_z\Re (z^*)$  is the convex hull of all the possible values of  $g(z^{*},\eta)$  in (1). For parameters  $(W_{j},b_{j})_{j = 1}^{2}$ , let  $\partial_{W_j}f(z^*)$  and  $\partial_{b_j}f(z^*)$  be the Clarke differential w.r.t. to  $W_{j}$  and  $b_{j}$ , respectively. They are the projection of  $\partial_zf(z^*)$  onto the space of individual parameters. Whenever the point  $z^{*}$  is clear (e.g. our algorithm), we will omit  $(z^{*})$  from  $f(z^{*})$ . Next, we define second-order stationary points for the empirical risk  $\Re$ . Notice that this generalizes the definition of SOSP for differentiable functions  $f\colon \nabla f = \mathbf{0}$  and  $\nabla^2 f\succeq \mathbf{0}$ .

Definition 2.2 (SOSP). We call  $z^*$  is a second-order stationary point (SOSP) of  $\Re$  if (1)  $z^*$  is a FOSP, (2)  $g(z^*, \eta)^T \eta \geq 0$  for all  $\eta$ , and (3)  $\eta^T H(z^*, \eta) \eta \geq 0$  for all  $\eta$  such that  $g(z^*, \eta)^T \eta = 0$ .

Given an input data point  $x \in \mathbb{R}^{d_x}$ , we define  $O(x) \coloneqq h(W_1x + b_1)$  to be the output of hidden layer. We note that the notation  $O(\cdot)$  is overloaded with the big-Oh notation, but their meaning will be clear from the context. Consider perturbing parameters  $(W_j, b_j)_{j=1}^2$  with  $(\Delta_j, \delta_j)_{j=1}^2$ , then the perturbed output  $\tilde{Y}(x)$  of the network and the amount of perturbation  $dY(x)$  can be expressed as

$$
d Y (x) := \tilde {Y} (x) - Y (x) = \Delta_ {2} O (x) + \delta_ {2} + \left(W _ {2} + \Delta_ {2}\right) J (x) \left(\Delta_ {1} x + \delta_ {1}\right),
$$

where  $J(x)$  can be thought informally as the "Jacobian" matrix of the hidden layer. The matrix  $J(x) \in \mathbb{R}^{d_h \times d_h}$  is diagonal, and its  $k$ -th diagonal entry is given by

$$
[ J (x) ] _ {k, k} := \left\{ \begin{array}{l l} h ^ {\prime} ([ W _ {1} x + b _ {1} ] _ {k}) & \text {i f} [ W _ {1} x + b _ {1} ] _ {k} \neq 0 \\ h ^ {\prime} ([ \Delta_ {1} x + \delta_ {1} ] _ {k}) & \text {i f} [ W _ {1} x + b _ {1} ] _ {k} = 0, \end{array} \right.
$$

where  $h'$  is the derivative of  $h$ . We define  $h'(0) \coloneqq s_+$ , which is okay because it is always multiplied with zero in our algorithm. For boundary data points,  $[J(x)]_{k,k}$  depends on the direction of perturbations  $[\Delta_1 \delta_1]_{k,}$ , as noted in Section 2.1. We additionally define  $dY_1(x)$  and  $dY_2(x)$  to separate the terms in  $dY(x)$  that are linear in perturbations versus quadratic in perturbations.

$$
d Y _ {1} (x) := \Delta_ {2} O (x) + \delta_ {2} + W _ {2} J (x) (\Delta_ {1} x + \delta_ {1}), d Y _ {2} (x) := \Delta_ {2} J (x) (\Delta_ {1} x + \delta_ {1}).
$$

For simplicity of notation for the rest of the paper, we define for all  $i \in [m] := \{1, \dots, m\}$ ,

$$
\bar {x} _ {i} := \left[ \begin{array}{c c} x _ {i} ^ {T} & 1 \end{array} \right] ^ {T} \in \mathbb {R} ^ {d _ {x} + 1}, \nabla \ell_ {i} := \nabla_ {w} \ell (Y (x _ {i}), y _ {i}), \nabla^ {2} \ell_ {i} := \nabla_ {w} ^ {2} \ell (Y (x _ {i}), y _ {i}).
$$

In our algorithm and its analysis, we need to give a special treatment to the boundary data points. To this end, for each node  $k \in [d_h]$  in the hidden layer, define boundary index set  $B_{k}$  as

$$
B _ {k} := \left\{i \in [ m ] \mid \left[ W _ {1} x + b _ {1} \right] _ {k} = 0 \right\}.
$$

The subspace spanned by vectors  $\bar{x}_i$  for in  $i\in B_k$  plays an important role in our tests; so let us define a symbol for it, as well as the cardinality of  $B_{k}$  and their sum:

$$
\mathcal {V} _ {k} := \operatorname {s p a n} \{\bar {x} _ {i} \mid i \in B _ {k} \}, M _ {k} := | B _ {k} |, M := \sum_ {k = 1} ^ {d _ {h}} M _ {k}.
$$

For  $k \in [d_h]$ , let  $v_k^T \in \mathbb{R}^{1 \times (d_x + 1)}$  be the  $k$ -th row of  $[\Delta_1 \quad \delta_1]$ , and  $u_k \in \mathbb{R}^{d_y}$  be the  $k$ -th column of  $\Delta_2$ . Next, we define the total number of parameters  $p$ , and vectorized perturbations  $\eta \in \mathbb{R}^p$ :

$$
p := d _ {y} + d _ {y} d _ {h} + d _ {h} (d _ {x} + 1), \eta^ {T} := \left[ \begin{array}{c c c c c c c} \delta_ {2} ^ {T} & u _ {1} ^ {T} & \dots & u _ {d _ {h}} ^ {T} & v _ {1} ^ {T} & \dots & v _ {d _ {h}} ^ {T} \end{array} \right].
$$

Also let  $z \in \mathbb{R}^p$  be vectorized parameters  $(W_j, b_j)_{j=1}^2$ , packed in the same order as  $\eta$ .

Define a matrix  $C_k \coloneqq \sum_{i \notin B_k} h'([N(x_i)]_k) \nabla \ell_i \bar{x}_i^T \in \mathbb{R}^{d_y \times (d_x + 1)}$ . This quantity appears multiple times and does not depend on the perturbation, so it is helpful to have a simple symbol for it.

We conclude this section by presenting one of the implications of Assumption 2 in the following lemma, which we will use later. The proof is simple, and is presented in Appendix B.1.

Lemma 1. If Assumption 2 holds, then  $M_{k} \leq d_{x}$  and the vectors  $\{\bar{x}_i\}_{i \in B_k}$  are linearly independent.

Algorithm 1 SOSP-CHECK (Rough pseudocode)  
Input: A tuple  $(W_{j},b_{j})_{j = 1}^{2}$  of  $\Re (\cdot)$    
1: Test if  $\partial_{W_2}\mathfrak{R} = \{\mathbf{0}_{d_y\times d_h}\}$  and  $\partial_{b_2}\mathfrak{R} = \{\mathbf{0}_{d_y}\}$    
2: for  $k\in [d_h]$  do   
3: if  $M_{k} > 0$  then   
4: Test if  $\mathbf{0}_{dx + 1}^T\in \partial_{[W_1b_1]_k,}$  .R.   
5: Test if  $g_{k}(z,v_{k})^{T}v_{k}\geq 0$  for all  $v_{k}$  via testing extreme rays  $\tilde{v}_k$  of polyhedral cones.   
6: Store extreme rays  $\tilde{v}_k$  s.t.  $g_{k}(z,\tilde{v}_{k})^{T}\tilde{v}_{k} = 0$  for second-order test.   
7: else   
8: Test if  $\partial_{[W_1b_1]_k,}$ $\mathfrak{R} = \{\mathbf{0}_{d_x + 1}^T\}$    
9: end if   
10: end for   
11: For all  $\eta$  s.s.t.  $g(z,\eta)^T\eta = 0$  , test if  $\eta^T H(z,\eta)\eta \geq 0$    
12: if  $\exists \eta \neq 0$  s.t.  $g(z,\eta)^T\eta = 0$  and  $\eta^T H(z,\eta)\eta = 0$  then   
13: return SOSP.   
14: else   
15: return Local Minimum.   
16: end if

# 3 TEST ALGORITHM FOR SECOND-ORDER STATIONARITY

In this section, we present SOSP-CHECK in Algorithm 1, which takes an arbitrary tuple  $(W_{j},b_{j})_{j = 1}^{2}$  of parameters as input and checks whether it is a SOSP. We first present a lemma that shows the explicit form of the perturbed empirical risk  $\Re (z + \eta)$  and identify first and second-order perturbations. The proof is deferred to Appendix B.2.

Lemma 2. For small enough perturbation  $\eta$ ,

$$
\Re (z + \eta) = \Re (z) + g (z, \eta) ^ {T} \eta + \frac {1}{2} \eta^ {T} H (z, \eta) \eta + o (\| \eta \| ^ {2}),
$$

where  $g(z,\eta)$  and  $H(z,\eta)$  satisfy

$$
g (z, \eta) ^ {T} \eta = \sum_ {i} \nabla \ell_ {i} ^ {T} d Y _ {1} (x _ {i}) = \left\langle \sum_ {i} \nabla \ell_ {i} O (x _ {i}) ^ {T}, \Delta_ {2} \right\rangle + \left\langle \sum_ {i} \nabla \ell_ {i}, \delta_ {2} \right\rangle + \sum_ {k = 1} ^ {d _ {h}} g _ {k} (z, v _ {k}) ^ {T} v _ {k},
$$

$$
\eta^ {T} H (z, \eta) \eta = \sum_ {i} \nabla \ell_ {i} ^ {T} d Y _ {2} (x _ {i}) + \frac {1}{2} \sum_ {i} \| d Y _ {1} (x _ {i}) \| _ {\nabla^ {2} \ell_ {i}} ^ {2},
$$

and  $g_{k}(z,v_{k})^{T} \coloneqq [W_{2}]_{*,k}^{T}\left(C_{k} + \sum_{i\in B_{k}}h^{\prime}(\bar{x}_{i}^{T}v_{k})\nabla \ell_{i}\bar{x}_{i}^{T}\right)$ . Also,  $g(z,\eta)$  and  $H(z,\eta)$  are piecewise constant functions of  $\eta$ , which are constant inside each polyhedral cone in space of  $\eta$ .

Rough pseudocode of SOSP-CHECK is presented in Algorithm 1. As described in Section 2.1, the algorithm consists of three steps: (a) testing first-order stationarity (b) testing  $g(z,\eta)^T\eta \geq 0$  for all  $\eta$ , and (c) testing  $\eta^T H(z,\eta)\eta \geq 0$  for  $\{\eta \mid g(z,\eta)^T\eta = 0\}$ . If the input point satisfies the second-order sufficient conditions for local minimality, the algorithm decides it is a local minimum. If the point only satisfies second-order necessary conditions, it returns SOSP. If a strict descent direction  $\eta$  is found, the algorithm terminates immediately and returns  $\eta$ . A brief description will follow, but the full algorithm (Algorithm 2) and a full proof of correctness are deferred to Appendix A.

# 3.1 TESTING FIRST-ORDER STATIONARITY (LINES 1, 4, AND 8)

Line 1 of Algorithm 1 corresponds to testing if  $\partial_{W_2}\Re$  and  $\partial_{b_2}\Re$  are singletons with zero. If not, the opposite direction is a descent direction. More details are in Appendix A.1.1.

Test for  $W_{1}$  and  $b_{1}$  is more difficult because  $g(z, \eta)$  depends on  $\Delta_{1}$  and  $\delta_{1}$  when there are boundary data points. For each  $k \in [d_h]$ , Line 4 (if  $M_{k} > 0$ ), and Line 8 (if  $M_{k} = 0$ ) test if  $\mathbf{0}_{d_x + 1}^T$  is in  $\partial_{[W_1b_1]_{k,\cdot}}\Re$ . Note from Definition 2.1 and Lemma 2 that  $\partial_{[W_1b_1]_{k,\cdot}}\Re$  is the convex hull of all possible values of  $g_{k}(z,v_{k})^{T}$ . If  $M_{k} > 0$ ,  $\mathbf{0} \in \partial_{[W_{1}b_{1}]_{k,\cdot}}\Re$  can be tested by solving a convex QP:

$$
\begin{array}{l l} \operatorname {m i n i m i z e} _ {\left\{s _ {i} \right\} _ {i \in B _ {k}}} & \| \left[ W _ {2} \right] _ {., k} ^ {T} \left(C _ {k} + \sum_ {i \in B _ {k}} s _ {i} \nabla \ell_ {i} \bar {x} _ {i} ^ {T}\right) \| _ {2} ^ {2} \\ \text {s u b j e c t} & \min  \left\{s _ {-}, s _ {+} \right\} \leq s _ {i} \leq \max  \left\{s _ {-}, s _ {+} \right\}, \forall i \in B _ {k}. \end{array} \tag {2}
$$

If the solution  $\{s_i^*\}_{i\in B_k}$  does not achieve zero objective value, then we can directly return a descent direction. For details please refer to FO-SUBDIFF-ZERO-TEST (Algorithm 3) and Appendix A.1.2.

# 3.2 TESTING  $g(z,\eta)^T\eta \geq 0$  FOR ALL  $\eta$  (LINES 5-6)

Linear program formulation. Lines 5-6 are about testing if  $g_{k}(z,v_{k})^{T}v_{k}\geq 0$  for all directions of  $v_{k}$ . If  $\mathbf{0}_{d_x + 1}^T\in \partial_{[W_1b_1]_k},\Re$ , with the solution  $\{s_i^*\}$  from QP (2) we can write  $g_{k}(z,v_{k})^{T}$  as

$$
g _ {k} (z, v _ {k}) ^ {T} = \left[ W _ {2} \right] _ {, k} ^ {T} \left(C _ {k} + \sum_ {i \in B _ {k}} h ^ {\prime} (\bar {x} _ {i} ^ {T} v _ {k}) \nabla \ell_ {i} \bar {x} _ {i} ^ {T}\right) = \left[ W _ {2} \right] _ {, k} ^ {T} \left(\sum_ {i \in B _ {k}} \left(h ^ {\prime} (\bar {x} _ {i} ^ {T} v _ {k}) - s _ {i} ^ {*}\right) \nabla \ell_ {i} \bar {x} _ {i} ^ {T}\right).
$$

Every  $i \in B_k$  bisects  $\mathbb{R}^{d_x + 1}$  into two halfspaces,  $\bar{x}_i^T v_k \geq 0$  and  $\bar{x}_i^T v_k \leq 0$ , in each of which  $h'(\bar{x}_i^T v_k)$  stays constant. Note that by Lemma 1,  $\bar{x}_i$ 's for  $i \in B_k$  are linearly independent. So, given  $M_k$  boundary data points, they divide the space  $\mathbb{R}^{d_x + 1}$  of  $v_k$  into  $2^{M_k}$  polyhedral cones.

Since  $g_{k}(z,v_{k})^{T}$  is constant in each polyhedral cones, we can let  $\sigma_{i}\in \{-1, + 1\}$  for all  $i\in B_k$ , and define an LP for each  $\{\sigma_i\}_{i\in B_k}\in \{-1, + 1\}^{M_k}$ :

$$
\underset {v _ {k}} {\mathrm {m i n i m i z e}} \quad [ W _ {2} ] _ {., k} ^ {T} \left(\sum_ {i \in B _ {k}} (s _ {\sigma_ {i}} - s _ {i} ^ {*}) \nabla \ell_ {i} \bar {x} _ {i} ^ {T}\right) v _ {k}
$$

$$
\begin{array}{l} v _ {k} \\ \text {s u b j e c t} \end{array} \quad \begin{array}{l} [ 2 ] _ {., k} (\angle i \in B _ {k} (\sigma_ {i} - \sigma_ {i}) \cdot \sigma_ {i} + \sigma_ {i}) \\ v _ {k} \in \mathcal {V} _ {k}, \quad \sigma_ {i} \bar {x} _ {i} ^ {T} v _ {k} \geq 0, \forall i \in B _ {k}. \end{array} \tag {3}
$$

Solving these LPs and checking if the minimum value is 0 suffices to prove  $g_{k}(z,v_{k})^{T}v_{k}\geq 0$  for all small enough perturbations. The constraint  $v_{k}\in \mathcal{V}_{k}$  is there because any  $v_{k}\notin \mathcal{V}_{k}$  is also orthogonal to  $g_{k}(z,v_{k})$ . It is equivalent to  $d_x + 1 - M_k$  linearly independent equality constraints. So, the feasible set of LP (3) has  $d_{x} + 1$  linearly independent constraints, which implies that the feasible set is a pointed polyhedral cone with vertex at origin. Since any point in a pointed polyhedral cone is a conical combination (linear combination with nonnegative coefficients) of extreme rays of the cone, checking nonnegativity of the objective function for all extreme rays suffices. We emphasize that we do not solve the LPs (3) in our algorithm; we just check the extreme rays.

Computational efficiency. Extreme rays of a pointed polyhedral cone in  $\mathbb{R}^{d_x + 1}$  are computed from  $d_{x}$  linearly independent active constraints. For each  $i\in B_k$ , the extreme ray  $\hat{v}_{i,k}\in \mathcal{V}_k\cap$  span  $\{\bar{x}_j\mid j\in B_k\setminus \{i\} \}^\perp$  must be tested whether  $g_{k}(z,\hat{v}_{i,k})^{T}\hat{v}_{i,k}\geq 0$ , in both directions. Note that there are  $2M_{k}$  extreme rays, and one extreme ray  $\hat{v}_{i,k}$  is shared by  $2^{M_k - 1}$  polyhedral cones. Moreover,  $\bar{x}_j^T\hat{v}_{i,k} = 0$  for  $j\in B_k\setminus \{i\}$ , which indicates that

$$
g _ {k} \left(z, \hat {v} _ {i, k}\right) ^ {T} \hat {v} _ {i, k} = \left(s _ {\sigma_ {i, k}} - s _ {i} ^ {*}\right) \left[ W _ {2} \right] _ {., k} ^ {T} \nabla \ell_ {i} \bar {x} _ {i} ^ {T} \hat {v} _ {i, k}, \text {w h e r e} \sigma_ {i, k} = \operatorname {s i g n} \left(\bar {x} _ {i} ^ {T} \hat {v} _ {i, k}\right),
$$

regardless of  $\{\sigma_j\}_{j\in B_k\setminus \{i\}}$ . Testing an extreme ray can be done with a single inequality test instead of  $2^{M_k - 1}$  separate tests for all cones! Thus, this extreme ray approach instead of solving individual LPs greatly reduces computation, from  $O(2^{M_k})$  to  $O(M_{k})$ .

Testing extreme rays. For the details of testing all possible extreme rays, please refer to FO-INCREASING-TEST (Algorithm 4) and Appendix A.2. FO-INCREASING-TEST computes all possible extreme rays  $\tilde{v}_k$  and tests if they satisfy  $g_k(z,\tilde{v}_k)^T\tilde{v}_k\geq 0$ . If the inequality is not satisfied by an extreme ray  $\tilde{v}_k$ , then this is a descent direction, so we return  $\tilde{v}_k$ . If the inequality holds with equality, it means this is a flat extreme ray, and it needs to be checked in second-order test, so we save this extreme ray for future use.

How many flat extreme rays  $(g_{k}(z,\tilde{v}_{k})^{T}\tilde{v}_{k} = 0)$  are there? Presence of flat extreme rays introduce inequality constraints in the QP that we solve in the second-order test. It is ideal not to have them, because in this case there are only equality constraints, so the QP is easier to solve. Lemma A.1 in Appendix A.2 shows the conditions for having flat extreme rays; in short, there is a flat extreme ray if  $[W_2]_{\cdot ,k}^T\nabla \ell_i = 0$  or  $s_i^* = s_+$  or  $s_{-}$ . For more details, please refer to Appendix A.2.

# 3.3 TESTING  $\eta^T H(z,\eta)\eta \geq 0$  FOR  $\{\eta \mid g(z,\eta)^T\eta = 0\}$  (LINES 11-16)

The second-order test checks  $\eta^T H(z,\eta)\eta \geq 0$  for "flat"  $\eta$ 's satisfying  $g(z,\eta)^T\eta = 0$ . This is done with help of the function SO-TEST (Algorithm 5). Given its input  $\{\sigma_{i,k}\}_{k\in [d_h],i\in B_k}$ , it defines fixed "Jacobian" matrices  $J_{i}$  for all data points and equality/inequality constraints for boundary data points, and solves the QP of the following form:

$$
\begin{array}{r l} \operatorname {m i n i m i z e} _ {\eta} & \sum_ {i} \nabla \ell_ {i} ^ {T} \Delta_ {2} J _ {i} (\Delta_ {1} x _ {i} + \delta_ {1}) + \frac {1}{2} \sum_ {i} \| \Delta_ {2} O (x _ {i}) + \delta_ {2} + W _ {2} J _ {i} (\Delta_ {1} x _ {i} + \delta_ {1}) \| _ {\nabla^ {2} \ell_ {i}} ^ {2}, \end{array}
$$

$$
\begin{array}{l l} \text {s u b j e c t t o} & [ W _ {2} ] _ {., k} ^ {T} u _ {k} = [ W _ {1} b _ {1} ] _ {k,}, \\ & \bar {x} _ {i} ^ {T} v _ {k} = 0, \end{array} \quad \forall k \in [ d _ {h} ], \tag {4}
$$

$$
\sigma_ {i, k} \bar {x} _ {i} ^ {T} v _ {k} \geq 0, \quad \forall k \in [ d _ {h} ], \forall i \in B _ {k} \text {s . t .} \sigma_ {i, k} \in \{- 1, + 1 \}.
$$

Constraints and number of QPs. There are  $d_h$  equality constraints of the form  $[W_2]_{*,k}^T u_k = [[W_1]_{k}, [b_1]_k] v_k$ . These equality constraints are due to the nonnegative homogeneous property of activation  $h$ ; i.e., scaling  $[W_1]_{k}$ , and  $[b_1]_k$  by  $\alpha > 0$  and scaling  $[W_2]_{*,k}$  by  $1 / \alpha$  yields exactly the same network. So, these equality constraints force  $\eta$  to be orthogonal to the loss-invariant directions. This observation is stated more formally in Lemma A.2, which as a corollary shows that any differentiable FOSP of  $\Re$  always has rank-deficient Hessian. The other constraints make sure that the union of feasible sets of QPs is exactly  $\{\eta \mid g(z,\eta)^T\eta = 0\}$  (please see Lemma A.3 in Appendix A.3 for details). It is also easy to check that these constraints are all linearly independent.

If there is no flat extreme ray, the algorithm solves just one QP with  $d_h + M$  equality constraints. If there are flat extreme rays, the algorithm solves one QP with  $d_h + M$  equality constraints, and  $2^K$  more QPs with  $d_h + M - L$  equality constraints and  $L$  inequality constraints, where

$$
K := \sum_ {k = 1} ^ {d _ {h}} \left| \left\{i \in B _ {k} \mid [ W _ {2} ] _ {., k} ^ {T} \nabla \ell_ {i} = 0 \right\} \right|, L := \sum_ {k = 1} ^ {d _ {h}} \left| \left\{i \in B _ {k} \mid \hat {v} _ {i, k} \text {o r} - \hat {v} _ {i, k} \text {i s a f l a t e x t . r a y} \right\} \right|. \tag {5}
$$

Recall from Section 3.2 that  $i \in B_k$  has a flat extreme ray if  $[W_2]_{\cdot,k}^T \nabla \ell_i = 0$  or  $s_i^* = s_+$  or  $s_-$ ; thus,  $K \leq L \leq M$ . Please refer to Appendix A.3 for more details.

Efficiency of solving the QPs (4). Despite NP-hardness of general QPs, our specific form of QPs (4) can be solved quite efficiently, avoiding exponential complexity in  $p$ . After solving QP (4), there are three (disjoint) termination conditions:

(T1)  $\eta^T Q\eta > 0$  whenever  $\eta \in S, \eta \neq 0$ , or  
(T2)  $\eta^T Q\eta \geq 0$  whenever  $\eta \in S$ , but  $\exists \eta \neq 0, \eta \in S$  such that  $\eta^T Q\eta = 0$ , or  
(T3)  $\exists \eta$  such that  $\eta \in S$  and  $\eta^T Q\eta < 0$

where  $S$  is the feasible set of QP. With the following two lemmas, we show that the termination conditions can be efficiently tested for ECQPs and ICQPs. First, the ECQPs can be iteratively solved with projected gradient descent, as stated in the next lemma.

Lemma 3. Consider the  $QP$ , where  $Q \in \mathbb{R}^{p \times p}$  is symmetric and  $A \in \mathbb{R}^{q \times p}$  has full row rank:

$$
\operatorname {m i n i m i z e} _ {\eta} \quad \frac {1}{2} \eta^ {T} Q \eta \quad \text {s u b j e c t t o} \quad A \eta = \mathbf {0} _ {q}
$$

Then, projected gradient descent (PGD) updates

$$
\eta^ {(t + 1)} = (I - A ^ {T} (A A ^ {T}) ^ {- 1} A) (I - \alpha Q) \eta^ {(t)}
$$

with learning rate  $\alpha < 1 / \lambda_{\mathrm{max}}(Q)$  converges to a solution or diverges to infinity exponentially fast. Moreover, with random initialization, PGD correctly checks conditions (T1)-(T3) with probability 1.

The proof is an extension of unconstrained case (Lee et al., 2016), and is deferred to Appendix B.3. Note that it takes  $O(p^2 q)$  time to compute  $(I - A^T (AA^T)^{-1}A)(I - \alpha Q)$  in the beginning, and each update takes  $O(p^2)$  time. It is also surprising that the convergence rate does not depend on  $q$ .

In the presence of flat extreme rays, we have to solve QPs involving  $L$  inequality constraints. We prove that our ICQP can be solved in  $O(p^3 + L^3 2^L)$  time, which implies that as long as the number of flat extreme rays is small, the problem can still be solved in polynomial time in  $p$ .

Lemma 4. Consider the  $QP$ , where  $Q \in \mathbb{R}^{p \times p}$  is symmetric,  $A \in \mathbb{R}^{q \times p}$  and  $B \in \mathbb{R}^{r \times p}$  have full row rank, and  $\left[ \begin{array}{cc} A^T & B^T \end{array} \right]$  has rank  $q + r$ :

$$
\begin{array}{l l l} \text {m i n i m i z e} _ {\eta} & \eta^ {T} Q \eta \quad \text {s u b j e c t t o} & A \eta = \mathbf {0} _ {q}, B \eta \geq \mathbf {0} _ {r}. \end{array}
$$

Then, there exists a method that checks whether (T1)-(T3) in  $O(p^3 + r^3 2^r)$  time.

In short, we transform  $\eta$  to define an equivalent problem, and use classical results in copositive matrices (Martin & Jacobson, 1981; Seeger, 1999; Hiriart-Urruty & Seeger, 2010); the problem can be solved by computing the eigensystem of a  $(p - q - r) \times (p - q - r)$  matrix, and testing copositivity of an  $r \times r$  matrix. The proof is presented in Appendix B.4.

Concluding the test. During all calls to SO-TEST, whenever any QP terminated with (T3), then SOSP-CHECK immediately returns the direction and terminates. After solving all QPs, if any of SO-TEST calls finished with (T2), then we conclude SOSP-CHECK with "SOSP." If all QPs terminated with (T1), then we can return "Local Minimum."

Table 1: Summary of experimental results  

<table><tr><td>(dx,dh,m)</td><td>#Runs</td><td>Sum M (Avg.)</td><td>Sum L (Avg.)</td><td>Sum K (Avg.)</td><td>P{L&gt;0}</td></tr><tr><td>(10,1,1000)</td><td>40</td><td>290 (7.25)</td><td>0 (0)</td><td>0 (0)</td><td>0</td></tr><tr><td>(10,1,10000)</td><td>40</td><td>371 (9.275)</td><td>1 (0.025)</td><td>0 (0)</td><td>0.025</td></tr><tr><td>(100,1,1000)</td><td>40</td><td>1,452 (36.3)</td><td>0 (0)</td><td>0 (0)</td><td>0</td></tr><tr><td>(100,1,10000)</td><td>40</td><td>2,976 (74.4)</td><td>2 (0.05)</td><td>0 (0)</td><td>0.05</td></tr><tr><td>(100,10,10000)</td><td>40</td><td>24,805 (620.125)</td><td>4 (0.1)</td><td>0 (0)</td><td>0.1</td></tr><tr><td>(1000,1,10000)</td><td>40</td><td>14,194 (354.85)</td><td>0 (0)</td><td>0 (0)</td><td>0</td></tr><tr><td>(1000,10,10000)</td><td>40</td><td>42,334 (1,058.35)</td><td>37 (0.925)</td><td>1 (0.025)</td><td>0.625</td></tr></table>

# 4 EXPERIMENTS

For experiments, we used artificial datasets sampled iid from standard normal distribution, and trained 1-hidden-layer ReLU networks with squared error loss. In practice, it is impossible to get to the exact nondifferentiable point, because they lie in a set of measure zero. To get close to those points, we ran Adam (Kingma & Ba, 2014) using full-batch (exact) gradient for 200,000 iterations and decaying step size (start with  $10^{-3}$ ,  $0.2 \times$  decay every 20,000 iterations). We observed that decaying step size had the effect of "descending deeper into the valley."

After running Adam, for each  $k \in [d_h]$ , we counted the number of approximate boundary data points satisfying  $|[W_1x_i + b_1]_k| < 10^{-5}$ , which gives an estimate of  $M_{k}$ . Moreover, for these points, we solved the QP (2) using L-BFGS-B (Byrd et al., 1995), to check if the terminated points are indeed (approximate) FOSPs. We could see that the optimal values of (2) are close to zero ( $\leq 10^{-6}$  typically,  $\leq 10^{-3}$  for largest problems). After solving (2), we counted the number of  $s_i^{*s}$ s that ended up with 0 or 1. The number of such  $s_i^{*}$ 's is an estimate of  $L - K$ . We also counted the number of approximate boundary data points satisfying  $|[W_2]_{\cdot,k}^T\nabla \ell_i| < 10^{-4}$ , for an estimate of  $K$ .

We ran the above-mentioned experiments for different settings of  $(d_x, d_h, m)$ , 40 times each. We fixed  $d_y = 1$  for simplicity. For large  $d_h$ , the optimizer converged to near-zero minima, making  $\nabla \ell_i$  uniformly small, so it was difficult to obtain accurate estimates of  $K$  and  $L$ . Thus, we had to perform experiments in settings where the optimizer converged to minima that are far from zero.

Table 1 summarizes the results. Through 280 runs, we observed that there are surprisingly many boundary data points  $(M)$  in general, but usually there are zero or very few (maximum was 3) flat extreme rays  $(L)$ . This observation suggests two important messages: (1) many local minima are on nondifferentiable points, which is the reason why our analysis is meaningful; (2) luckily,  $L$  is usually very small, so we only need to solve ECQPs  $(L = 0)$  or ICQPs with very small number of inequality constraints, which are solved efficiently (Lemmas 3 and 4). We can observe that  $M$ ,  $L$ , and  $K$  indeed increase as model dimensions and training set get larger, but the rate of increase is not as fast as  $d_x$ ,  $d_h$ , and  $m$ .

For our experiments, we used artificial regression datasets instead of real ones. This is because popular real datasets are classification datasets, but cross-entropy loss gets arbitrarily small as parameters tend to infinity and hinge loss is not differentiable; the losses are not adequate for our purposes.

# 5 DISCUSSION AND FUTURE WORK

We provided an algorithm to test second-order stationarity and escape saddle points, for nondifferentiable points of empirical risk of shallow ReLU-like networks. Despite difficulty raised by boundary data points dividing the parameter space into  $2^{M}$  regions, we reduced the computation to  $d_{h}$  convex QPs,  $O(M)$  equality/inequality tests, and one (or a few more) QP. In benign cases, the last QP is equality constrained, which can be efficiently solved with projected gradient descent. In the worst case, the QP has a few (say  $L$ ) inequality constraints, but it can be solved efficiently when  $L$  is small. We also provided empirical evidences that  $L$  is usually either zero or very small, suggesting that the test can be done efficiently in most cases. Extending this test to deeper neural networks is a possible future work. Also, numerical implementation of this algorithm and combining it with practical gradient-based methods will be of great interest.

# REFERENCES

Pierre Baldi and Kurt Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53-58, 1989.  
Avrim Blum and Ronald L Rivest. Training a 3-node neural network is np-complete. In Proceedings of the 1st International Conference on Neural Information Processing Systems, pp. 494-501. MIT Press, 1988.  
Jonathan Borwein and Adrian S Lewis. Convex analysis and nonlinear optimization: theory and examples. Springer Science & Business Media, 2010.  
Alon Brutzkus and Amir Globerson. Globally optimal gradient descent for a convnet with gaussian inputs. In International Conference on Machine Learning, pp. 605-614, 2017.  
Richard H Byrd, Peihuang Lu, Jorge Nocedal, and Ciyou Zhu. A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5):1190-1208, 1995.  
Yair Carmon, John C Duchi, Oliver Hinder, and Aaron Sidford. Accelerated methods for non-convex optimization. arXiv preprint arXiv:1611.00756, 2016.  
Francis H Clarke, Yuri S Ledyaev, Ronald J Stern, and Peter R Wolenski. Nonsmooth analysis and control theory, volume 178. Springer Science & Business Media, 2008.  
Simon S Du, Jason D Lee, Yuandong Tian, Barnabas Poczos, and Aarti Singh. Gradient descent learns one-hidden-layer cnn: Don't be afraid of spurious local minima. arXiv preprint arXiv:1712.00779, 2017.  
J-B Hiriart-Urruty and Alberto Seeger. A variational approach to copositive matrices. SIAM review, 52(4):593-629, 2010.  
Kenji Kawaguchi. Deep learning without poor local minima. In Advances in Neural Information Processing Systems, pp. 586-594, 2016.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Thomas Laurent and James von Brecht. The multilinear structure of relu networks. arXiv preprint arXiv:1712.10132, 2017.  
Jason D Lee, Max Simchowitz, Michael I Jordan, and Benjamin Recht. Gradient descent only converges to minimizers. In Conference on Learning Theory, pp. 1246-1257, 2016.  
Duncan Henry Martin and David Harris Jacobson. Copositive matrices and definiteness of quadratic forms subject to homogeneous linear inequality constraints. Linear Algebra and its Applications, 35:227-258, 1981.  
Aryan Mokhtari, Asuman Ozdaglar, and Ali Jabbabaie. Escaping saddle points in constrained optimization. arXiv preprint arXiv:1809.02162, 2018.  
Katta G Murty and Santosh N Kabadi. Some np-complete problems in quadratic and nonlinear programming. Mathematical programming, 39(2):117-129, 1987.  
Quynh Nguyen and Matthias Hein. The loss surface of deep and wide neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 2603-2612, 2017a.  
Quynh Nguyen and Matthias Hein. Optimization landscape and expressivity of deep cnns. arXiv preprint arXiv:1710.10928, 2017b.  
Panos M Pardalos and Stephen A Vavasis. Quadratic programming with one negative eigenvalue is np-hard. Journal of Global Optimization, 1(1):15-22, 1991.  
Sashank J Reddi, Manzil Zaheer, Suvrit Sra, Barnabas Poczos, Francis Bach, Ruslan Salakhutdinov, and Alexander J Smola. A generic approach for escaping saddle points. arXiv preprint arXiv:1709.01434, 2017.

Itay Safran and Ohad Shamir. Spurious local minima are common in two-layer relu neural networks. arXiv preprint arXiv:1712.08968, 2017.  
Alberto Seeger. Eigenvalue analysis of equilibrium processes defined by linear complementarity conditions. Linear Algebra and its Applications, 292(1-3):1-14, 1999.  
Ohad Shamir. Are resnets provably better than linear predictors? arXiv preprint arXiv:1804.06739, 2018.  
Daniel Soudry and Yair Carmon. No bad local minima: Data independent training error guarantees for multilayer neural networks. arXiv preprint arXiv:1605.08361, 2016.  
Yuandong Tian. An analytical formula of population gradient for two-layered relu network and its applications in convergence and critical point analysis. In International Conference on Machine Learning, pp. 3404-3413, 2017.  
Chenwei Wu, Jiajun Luo, and Jason D Lee. No spurious local minima in a two hidden unit relu network. In International Conference on Learning Representations Workshop, 2018.  
Xiao-Hu Yu and Guo-An Chen. On the local minima free condition of backpropagation learning. IEEE Transactions on Neural Networks, 6(5):1300-1303, 1995.  
Chulhee Yun, Suvrit Sra, and Ali Jababaie. Spurious local minima in neural networks: a critical view. arXiv preprint arXiv:1802.03487, 2018a.  
Chulhee Yun, Suvrit Sra, and Ali Jadbabaie. Global optimality conditions for deep neural networks. In International Conference on Learning Representations, 2018b.  
Yi Zhou and Yingbin Liang. Critical points of neural networks: Analytical forms and landscape properties. In International Conference on Learning Representations, 2018.

Algorithm 2 SOSP-CHECK  
Input: A tuple  $(W_{j},b_{j})_{j = 1}^{2}$  of  $\Re (\cdot)$    
1: if  $\sum_{i = 1}^{m}\nabla \ell_{i}\left[O(x_{i})^{T}1\right]\neq \mathbf{0}_{d_{y}\times (d_{h} + 1)}$  then   
2: return  $[\Delta_2\delta_2]\gets -\sum_{i = 1}^{m}\nabla \ell_{i}\left[O(x_{i})^{T}1\right],\Delta_{1}\gets \mathbf{0}_{d_{h}\times d_{x}},\delta_{1}\gets \mathbf{0}_{d_{h}}.$    
3: end if   
4: for  $k\in [d_h]$  do   
5: if  $M_{k} > 0$  then   
6:  $\{s_i^*\}_{i\in B_k}\gets \mathrm{FO - SUBDIFF - ZERO - TEST}(k)$    
7:  $\tilde{v}_k^T\gets [W_2]_{*,k}^T (C_k + \sum_{i\in B_k}s_i^*\nabla \ell_i\bar{x}_i^T).$    
8: if  $\tilde{v}_k\neq \mathbf{0}_{d_x + 1}$  then   
9: return  $v_{k}\gets -\tilde{v}_{k},\forall k^{\prime}\in [d_{h}]\setminus \{k\} ,v_{k^{\prime}}\gets \mathbf{0}_{d_{x} + 1},\Delta_{2}\gets \mathbf{0}_{d_{y}\times d_{h}},\delta_{2}\gets \mathbf{0}_{d_{y}}.$    
10: end if   
11: (decr,  $\tilde{v}_k,\{S_{i,k}\}_{i\in B_k})\gets \mathrm{FO - INCREASING - TEST}(k,\{s_i^*\}_{i\in B_k}).$    
12: if decr  $=$  True then   
13: return  $v_{k}\gets \tilde{v}_{k},\forall k^{\prime}\in [d_{h}]\setminus \{k\} ,v_{k^{\prime}}\gets \mathbf{0}_{d_{x} + 1},\Delta_{2}\gets \mathbf{0}_{d_{y}\times d_{h}},\delta_{2}\gets \mathbf{0}_{d_{y}}.$    
14: end if   
15: else if  $[W_2]_{*,k}^T C_k\neq 0_{d_x + 1}^T$  then   
16: return  $v_{k}\gets -C_{k}^{T}[W_{2}]_{*,k},\forall k^{\prime}\in [d_{h}]\setminus \{k\} ,v_{k^{\prime}}\gets \mathbf{0}_{d_{x} + 1},\Delta_{2}\gets \mathbf{0}_{d_{y}\times d_{h}},\delta_{2}\gets \mathbf{0}_{d_{y}}.$    
17: end if   
18: end for   
19: (decr,sosp,  $(\Delta_j,\delta_j)^2_{j = 1})\gets \mathrm{SO - TEST}(\{0\}_{k\in [d_h],i\in B_k}).$    
20: if decr  $=$  True then return  $(\Delta_j,\delta_j)^2_{j = 1}$    
21: end if   
22: if  $M\neq 0$  and  $\{S_{i,k}\}_{k\in [d_h],i\in B_k}\neq \{\{0\} \}_{k\in [d_h],i\in B_k}$  then   
23: for each element  $\{\sigma_{i,k}\}_{k\in [d_h],i\in B_k}\in \prod_{k\in [d_h]}\prod_{i\in B_k}S_{i,k}$  do   
24: (decr,sospTemp,  $(\Delta_j,\delta_j)^2_{j = 1})\gets \mathrm{SO - TEST}(\{\sigma_{i,k}\}_{k\in [d_h],i\in B_k}).$    
25: if decr  $=$  True then return  $(\Delta_j,\delta_j)^2_{j = 1}$  .   
26: end if   
27: sosp  $<$  sosp V sospTemp   
28: end for   
29: end if   
30: if sosp  $=$  True then return SOSP.   
31: else return Local Minimum.   
32: end if

Algorithm 3 FO-SUBDIFF-ZERO-TEST  
Input:  $k\in [d_h]$    
1: Solve the following optimization problem and get optimal solution  $\{s_i^*\}_{i\in B_k}$  .. minimize  $\{s_i\}_{i\in B_k}$ $\| [W_2]_{\cdot ,k}^T (C_k + \sum_{i\in B_k}s_i\nabla \ell_i\bar{x}_i^T)\| _2^2$  subject to  $\min \{s_{-},s_{+}\} \leq s_{i}\leq \max \{s_{-},s_{+}\} ,\forall i\in B_{k},$    
2: return  $\{s_i^*\}_{i\in B_k}$
