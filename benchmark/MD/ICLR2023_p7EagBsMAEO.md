# UNDERSTANDING EDGE-OF-STABILITY TRAINING DYNAMICS WITH A MINIMALIST EXAMPLE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently, researchers observed that gradient descent for deep neural networks operates in an "edge-of-stability" (EoS) regime: the sharpness (maximum eigenvalue of the Hessian) is often larger than stability threshold  $2 / \eta$  (where  $\eta$  is the step size). Despite this, the loss oscillates and converges in the long run, and the sharpness at the end is just slightly below  $2 / \eta$ . While many other well-understood nonconvex objectives such as matrix factorization or two-layer networks can also converge despite large sharpness, there is often a larger gap between sharpness of the endpoint and  $2 / \eta$ . In this paper, we study EoS phenomenon by constructing a simple function that has the same behavior. We give rigorous analysis for its training dynamics in a large local region and explain why the final converging point has sharpness close to  $2 / \eta$ . Globally we observe that the training dynamics for our example has an interesting bifurcating behavior, which was also observed in the training of neural nets.

# 1 INTRODUCTION

Many works tried to understand how simple gradient-based methods can optimize complicated neural network objectives. However, recently some empirical observations show that optimization for deep neural networks may operate in a more surprising regime. In particular, Cohen et al. (2021) observed that when running gradient descent on neural networks with a fixed step-size  $\eta$ , the sharpness (largest eigenvalue of the Hessian) of the training trajectory often oscillate around the stability threshold  $2 / \eta^{1}$ , while the loss still continues to decrease in the long run. This phenomenon is called "edge-of-stability" and has received a lot of attentions (see Section 1.2 for related work).

While many works try to understand why (variants of) gradient descent can still converge despite that the sharpness is larger than  $2 / \eta$ , empirically gradient descent for deep neural networks has even stronger properties. As shown in Fig. 1a, for a fixed initialization, if one changes the step size  $\eta$ , the final converging point has sharpness very close to  $2 / \eta$ . We call this phenomenon "sharpness adaptivity". Another perspective on the same phenomenon is that for a wide range of initializations, for a fixed step-size  $\eta$  their final converging points all have sharpness very close to  $2 / \eta$ , we call this phenomenon "sharpness concentration".

Surprisingly, both sharpness adaptivity and sharpness concentration happen on deeper networks, while for shallower models of non-convex optimization such as matrix factorization or 2-layer neural networks the gap between sharpness and  $2 / \eta$  is often much larger (see Fig. 1b). This suggests that these phenomena are related to the network depth. What is the mechanism for sharpness adaptivity and concentration, and how does that relate to number of layers? To answer these questions, in this paper we consider a minimalist example for edge-of-stability.

More specifically, we construct an objective function (4-layer scalar network), such that gradient descent on this objective has similar empirical behavior as deeper networks. We give a rigorous analysis for the training dynamics of this objective function in a large local region, which proves that the dynamics satisfy both sharpness adaptivity and sharpness concentration. The global training dynamics for our objective exhibits a complicated fractal behavior (which is also why our rigorous results are local), and such behavior has been observed in training of neural networks.

![](images/0c06e382a2a4dd8ca3701fd214eed1051835978955bcc7f3d0e262cdfbaf8bca.jpg)  
(a) ReLU 5-layer FC network with 50 neurons per layer.  $(\lambda \approx 2 / \eta)$

![](images/b7720a857bbe74db1d8c0e638e478aa8ef70b7f281660a88b1e6fe383d251715.jpg)  
(b) Linear 2-layer FC network with 10 neurons per layer  $(\lambda < 2 / \eta)$

![](images/5968f1882cfab4959abbcca9a63d2bb30d31c81e0b8b0be1e5595696d107a6f6.jpg)  
Figure 1: EoS Phenomena in NN Training. We consider three models including a 5-layer ReLU activated fully connected network, a 2-layer fully connected linear network with asymmetric initialization factor (4,0.1) (see Appendix A.1 for explanation), and a 4-layer scalar network equivalent to  $\min_{x,y,z,w}\frac{1}{2}(1 - xyzw)^2$ . For each model we run gradient descent from the same initialization using different learning rates. For (a) and (c), the sharpness converges very close to  $2/\eta$  with loss continuing to decrease. For (b), the sharpness decreases to be significantly lower than  $2/\eta$ .  
(c) 4-layer scalar network.  $(\lambda \approx 2 / \eta)$

# 1.1 OUR RESULTS

The objective function we consider is very simple:  $\mathcal{L}(x,y,z,w) \triangleq \frac{1}{2} (1 - xyzw)^2$ . One can view this as a 4-layer scalar network (each layer has a single neuron). We even couple the initialization so that  $x = z, y = w$  so effectively it becomes a function on two variables. For this objective function we prove its convergence and sharpness concentration properties:

Theorem 1.1 (Sharpness Concentration, Informal). For any learning rate  $\eta$  smaller than some constant, there is a constant size region  $\mathbb{S}$  such that the GD trajectory with step size  $\eta$  from all initializations in  $\mathbb{S}$  converges to a global minimum with sharpness within  $(2 / \eta -\frac{20}{3}\eta ,2 / \eta)$ .

As a direct corollary we can also prove that it has the sharpness adaptivity property.

Corollary 1.1 (Sharpness Adaptivity, Informal). There exists a constant size region  $\mathbb{S}$  and a corresponding range of step sizes  $\mathbb{K}$  that for all  $\eta \in \mathbb{K}$ , the GD trajectory with step size  $\eta$  from any initialization in  $\mathbb{S}$  converges to a global minimum with sharpness within  $(2 / \eta -\frac{20}{3}\eta ,2 / \eta)$ .

The training dynamics is illustrated in Fig. 2. To analyze the training dynamics, we reparametrize the objective function and show that the 2-step dynamics of gradient descent roughly follows a parabola trajectory. The extreme point of this parabola is the final converging point which has sharpness very close to  $2 / \eta$ . Intuitively, the parabola trajectory comes from a cubic term in the approximation of the training dynamics (see Section 3.1 for detailed discussions). We can also extend our result to a setting where  $x, y$  are replaced by vectors, see Theorem 3.2 in Section 3.3.

In Section 4 we explain the difference between the dynamics of our degree-4 model with degree-2 models (which are more similar to matrix factorizations or 2-layer neural networks). We show that the dynamics for degree-2 models do not have the higher order terms, and their trajectories form an ellipse instead of a parabola. We also show that degree-3 models have mixed behaviors between degree-2 and degree 4.

Finally, in Section 5 we show why it is difficult to extend Theorem 3.1 to global convergence – the training trajectory exhibits fractal behavior globally. Such behaviors can be qualitatively approximated by simple low-degree nonlinear dynamics that are standard in chaos theory, but are still very difficult to analyze.

# 1.2 RELATED WORKS

The phenomenon of gradient descent on Edge of Stability (EoS) was first formalized and empirically demonstrated in Cohen et al. (2021). They show that the loss can non-monotonically decrease even when the sharpness  $\lambda > 2 / \eta$ . The non-monotone property of the loss has also been observed in many other settings (Jastrzebski et al., 2020; Xing et al., 2018; Lewkowycz et al., 2020; Wang et al., 2022; Arora et al., 2018; Li et al., 2022a).

Recently several works try to understand the mechanism behind EoS with different loss functions under various assumptions (Ahn et al., 2022; Ma et al., 2022; Arora et al., 2022; Lyu et al., 2022; Li et al., 2022b). Ahn et al. (2022) studied the non-monotonic decreasing behavior of gradient descent (which they call unstable convergence), and discussed the possible causes of this phenomenon. From a landscape perspective, Ma et al. (2022) defined a special subquadratic property of the loss function, and proved that EoS occurs based on this assumption.

Arora et al. (2022) and Lyu et al. (2022) studied the implicit bias on the sharpness of gradient descent in some general loss function. Both works focus on the regime where the parameter is close to the manifold of minimum loss. Arora et al. (2022) theoretically proved that with a modified loss  $\sqrt{L}$  or using normalized GD, gradient descent enters EoS regime and has sharpness reduction effect around the manifold of minima. Lyu et al. (2022) considered the spherical sharpness as the surrogate of the sharpness due to the scale-invariant property. In both works, the effective step-size  $\eta$  changes throughout the training process, so sharpness adaptivity and concentration do not apply.

Another line of works (Lewkowycz et al., 2020; Wang et al., 2022) focus on the implicit bias introduced by large learning rate. Lewkowycz et al. (2020) first proposed "catapult phase", a regime similar to the EoS, where loss does not diverge even if sharpness is larger than  $2 / \eta$ . Wang et al. (2022) theoretically gives a convergence analysis on the matrix factorization problem for large learning rate beyond  $2 / \lambda$  where  $\lambda$  is the sharpness. Their results include two stages: in the first stage the loss may oscillate but never diverge, and the sharpness decreases to enter the second stage where the loss can decrease monotonically. Recently Li et al. (2022b) followed Lewkowycz et al. (2020) and provided a more detailed theoretical analysis on the sharpness along the gradient descent trajectory in an overparameterized two-layer linear network setting under some assumptions during the training process. They also provided some heuristic explanation on EoS in more general models: gradient descent leaves and re-enters the regime where sharpness  $\lambda > 2 / \eta$  repeatedly. These works mostly focus on the degree-2 setting which does not have the sharpness adaptivity and sharpness concentration properties.

# 2 PRELIMINARIES AND NOTATIONS

In this section, we introduce the minimalist model which exhibits both sharpness adaptivity and sharpness concentration.

# 2.1 GRADIENT DESCENT ON PRODUCT OF 4 SCALAR

We focus on the simple objective  $\mathcal{L}(x,y,z,w) \triangleq \frac{1}{2} (1 - xyzw)^2$ . Let the learnable parameters  $x,y,z,w \in \mathbb{R}$  to be trained using gradient descent with a fixed step size  $\eta \in \mathbb{R}^+$  that

$$
\left(x _ {t + 1}, y _ {t + 1}, z _ {t + 1}, w _ {t + 1}\right) = \left(x _ {t}, y _ {t}, z _ {t}, w _ {t}\right) - \eta \nabla \mathcal {L} \left(x _ {t}, y _ {t}, z _ {t}, w _ {t}\right). \tag {1}
$$

Here  $x_{t}$  denotes the value of parameter  $x$  after the  $t$ -th update. To further simplify the problem, we consider the symmetric initialization of  $z_{0} = x_{0}$ ,  $w_{0} = y_{0}$ . Note that due to symmetry of objective, the identical entries will remain identical throughout the training process, so the training dynamics reduces to two dimensional and the 1-step update of  $x$  and  $y$  follows

$$
x _ {t + 1} = x _ {t} - x _ {t} y _ {t} ^ {2} \eta (x _ {t} ^ {2} y _ {t} ^ {2} - 1), \quad y _ {t + 1} = y _ {t} - x _ {t} ^ {2} y _ {t} \eta (x _ {t} ^ {2} y _ {t} ^ {2} - 1). \qquad (2)
$$

It's easy to show that the set of global minima for this function form the hyperbola  $xy = 1$ . Without loss of generality we focus on the case when  $x, y > 0$ , and in most of the analysis we also focus on the side where  $x > y$ . As shown in Fig. 2, with GD running on such a minimal model, we observe convergence on EoS for a wide range of initializations. Eventually all such trajectories converge to minima that are just slightly flatter than the "EoS minima" (the minima whose sharpness is exactly  $2 / \eta$ , see Definition 1).

![](images/4ad89fdd10145629a4f5babf20697a2add9988cef1cb2028f7f0b980b0db8ed2.jpg)  
(a) Evolution of training loss, sharpness, and trajectory of GD on the 4 scalar example from the same initialization with different learning rates.

![](images/1fec689d466b60ed62cdc1837754df7c0c559ba0c673ae14df767b5c680896cd.jpg)

![](images/ab6364a4e900931e0de393707851be366c98647ba5b824de5d5fab1c560e45b4.jpg)  
Figure 2: EoS phenomenon on product of 4 scalars with symmetric initialization. In (a) we demonstrate sharpness adaptivity by running GD with learning rate  $\eta = \frac{2}{8}$ ,  $\frac{2}{10}$ ,  $\frac{2}{12}$  from the same initialization. The sharpness of all trajectories converges to around their corresponding stability threshold  $2 / \eta$  while the loss decreases exponentially. In the 2D trajectory, the 2-step movement quickly converges to some smooth curves ending very close to the EoS minimum. In (b) we demonstrate sharpness concentration by running GD with constant learning rate  $\eta = 0.2$  for 50000 iterations from a dense grid of initializations and plot the sharpness of their converging minima. Initializations in the red shaded area all converge to a minima with sharpness in  $(2 / \eta - 0.1, 2\eta)$ .  
(b) Initializations converging close to  $\lambda = 2 / \eta$  ( $\eta = 0.2$ )

# 2.2 EOS MINIMA AND REPARAMETERIZATION

Given that a wide range of initializations all converge very close to the "EoS minima" with sharpness  $2 / \eta$ , we want to concretely characterize those points. The complete calculations are deferred to Appendix B.1. Denote  $\gamma = xy$ , the Hessian of the objective  $\mathcal{L}$  at  $(x,x,y,y)$  admits eigenvalues

$$
\lambda_ {1} = \frac {1}{2} \left(\left(x ^ {2} + y ^ {2}\right) \left(3 \gamma^ {2} - 1\right) + \sqrt {\left(x ^ {2} + y ^ {2}\right) ^ {2} \left(1 - 3 \gamma^ {2}\right) ^ {2} + 4 \gamma^ {2} \left(3 - 1 0 \gamma^ {2} + 7 \gamma^ {4}\right)}\right),
$$

$$
\lambda_ {2} = \frac {1}{2} \left(\left(x ^ {2} + y ^ {2}\right) \left(3 \gamma^ {2} - 1\right) - \sqrt {\left(x ^ {2} + y ^ {2}\right) ^ {2} \left(1 - 3 \gamma^ {2}\right) ^ {2} + 4 \gamma^ {2} \left(3 - 1 0 \gamma^ {2} + 7 \gamma^ {4}\right)}\right). \tag {3}
$$

and  $\lambda_3 = x^2 (1 - \gamma),\lambda_4 = y^2 (1 - \gamma)$ . When  $(x,y)$  converges to any minimum,  $\gamma = xy = 1$ , so  $\lambda_{2},\lambda_{3},\lambda_{4}$  all vanishes. Therefore it is  $\lambda_{1}$  that corresponds to the EoS phenomenon people observe.

When  $\eta < \frac{1}{2}$ , solving  $\lambda_{1} = 2 / \eta$  with  $x^{2}y^{2} = 1$  gives  $x = \pm \frac{1}{\sqrt{2}} ((-4 + \eta^{-2})^{\frac{1}{2}} + \eta^{-1})^{\frac{1}{2}}$ ,  $y = \pm \sqrt{2} ((-4 + \eta^{-2})^{\frac{1}{2}} + \eta^{-1})^{-\frac{1}{2}}$  and their multiplicative inverses. These solutions correspond to the minima with sharpness exactly equal to the EoS threshold of  $2 / \eta$ . Since they are all symmetric with each other, without loss of generality we pick the minimum of interest as follows.

Definition 1 ( $\eta$ -EoS Minimum). For any step size  $\eta \in (0, \frac{1}{2})$ , the  $\eta$ -EoS minimum under the  $(x, y)$ -parameterization is

$$
(\check {x}, \check {y}) \triangleq \left(\frac {1}{\sqrt {2}} \left((- 4 + \eta^ {- 2}) ^ {\frac {1}{2}} + \eta^ {- 1}\right) ^ {\frac {1}{2}}, \sqrt {2} \left((- 4 + \eta^ {- 2}) ^ {\frac {1}{2}} + \eta^ {- 1}\right) ^ {- \frac {1}{2}}\right). \tag {4}
$$

Despite we are able to obtain a closed-form expression for the EoS minimum, its  $x - y$  coordinate could still be tricky to analyze. Thus we consider the following reparameterization: For any  $(x,y)\in \{(x,y)\in \mathbb{R}^{+}\times \mathbb{R}^{+}:x > y\}$ , define  $c\triangleq (x^2 -y^2)^{\frac{1}{2}}$  and  $d\triangleq xy$ . This gives a bijective continuous mapping between  $\{(x,y)\in \mathbb{R}^{+}\times \mathbb{R}^{+}:x > y\}$  and  $\{(c,d)\in \mathbb{R}^{+}\times \mathbb{R}^{+}\}$ . This is a natural reparameterization since intuitively the basis in the new coordinate system are the two orthogonal family of hyperbolas  $xy = C$  and  $x^{2} - y^{2} = C$ . The former captures the movement orthogonal to the manifold of minima  $xy = 1$  while the latter captures the movement along the manifold of minima. Note that a similar separation of dynamics was also used in Arora et al. (2022).

With  $c, d$  as defined, the  $\eta$ -EoS minimum simplifies to  $(\check{c}, \check{d}) \triangleq ((\eta^{-2} - 4)^{\frac{1}{4}}, 1)$ . To expand the dynamics near the  $\eta$ -EoS minimum, we let  $a \triangleq c - (\eta^{-2} - 4)^{\frac{1}{4}}$  and  $b \triangleq d - 1$  to be the offset from  $(\check{c}, \check{d})$ . Our analysis will primarily be using the  $(a, b)$ -parameterization.

Definition 2 ( $\eta$ -EoS Reparameterization). For any step size  $\eta > 0$ , for any  $(x, y) \in \mathbb{R}^{+} \times \mathbb{R}^{+}$  such that  $x > y$ , the  $(a, b)$  reparameterization of  $(x, y)$  are respectively given by

$$
(a, b) \triangleq \left(\left(x ^ {2} - y ^ {2}\right) ^ {\frac {1}{2}} - \left(\eta^ {- 2} - 4\right) ^ {\frac {1}{4}}, x y - 1\right). \tag {5}
$$

Let  $\kappa \triangleq \sqrt{\eta}$ , following Eq. (2), the 1-step update under the reparameterization becomes

$$
\begin{array}{l} a _ {t + 1} = (\kappa^ {- 4} - 4) ^ {\frac {1}{4}} + \left(a _ {t} + (\kappa^ {- 4} - 4) ^ {\frac {1}{4}}\right) \left(1 - \left((1 + b _ {t}) ^ {3} - (1 + b _ {t})\right) ^ {2} \kappa^ {4}\right) ^ {\frac {1}{2}}, \\ b _ {t + 1} = b + \left(\left(1 + b _ {t}\right) ^ {3} - 2 \left(1 + b _ {t}\right) ^ {5} + \left(1 + b _ {t}\right) ^ {7}\right) \kappa^ {4} \tag {6} \\ + \left((1 + b _ {t}) - (1 + b _ {t}) ^ {3}\right) \left(4 (1 + b _ {t}) ^ {2} \kappa^ {4} + (a _ {t} \kappa + (1 - 4 \kappa^ {4}) ^ {\frac {1}{4}}) ^ {4}\right) ^ {\frac {1}{2}}. \\ \end{array}
$$

Now we can proceed to analyze the dynamics of this simple example.

# 3 DYNAMICS OF GRADIENT DESCENT ON PRODUCT OF 4 SCALAR

In this section, we will rigorously analyze the training dynamics characterized by Eq. (6). First we will introduce the approximation of one and two-step update and build up intuition on the dynamics. Then we will present our main theoretical results that the 4 scalar example exhibits both characterizations of EoS training.

# 3.1 APPROXIMATING 1-STEP AND 2-STEP UPDATES

Here we introduce the informal approximation on Eq. (6) and the corresponding two-step updates. For cleanness of presentation we will use  $\approx$  to hide all dominated terms. The rigorous statements of the approximations and corresponding proofs are deferred to Appendix B.3. When we are only describing the one/two-step dynamics, we use  $a, a', a''$  to denote  $a_t, a_{t+1}, a_{t+2}$  and  $b, b', b''$  to denote  $b_t, b_{t+1}, b_{t+2}$ . Denoting  $\kappa \triangleq \sqrt{\eta}$ , when  $\kappa, |a|, |b|$  are all not too large (see precise ranges in condition B.1), we have

$$
a ^ {\prime} \approx a - 2 b ^ {2} \kappa^ {3}, \quad b ^ {\prime} \approx - b - 4 a b \kappa - 3 b ^ {2} - b ^ {3};
$$

$$
a ^ {\prime \prime} \approx a - 4 b ^ {2} \kappa^ {3}, \quad b ^ {\prime \prime} \approx b + 8 a b \kappa - 1 6 b ^ {3}. \tag {7}
$$

In the approximation,  $a$  is monotonically decreasing at a steady rate of  $2b^{3}\kappa^{3}$  per step. The one-step update of  $b$  is flipping signs and contains second and third order terms of  $b$ . For the two-step approximation however, the oscillation behavior and the even-order terms of  $b$  all cancel. This is consistent with the analysis in (Arora et al., 2022) that the two step dynamics travels along a sharpness reducing flow.

Before proceeding to analyze the discrete GD movement, we first get intuition by approximating the two-step dynamics with a simple ODE

$$
\frac {\mathrm {d} b}{\mathrm {d} a} = \frac {b ^ {\prime \prime} - b}{a ^ {\prime \prime} - a} = \frac {1 6 b ^ {3} - 8 a b \kappa}{4 b ^ {2} \kappa^ {3}}. \tag {8}
$$

This would be the limit when  $\kappa$  is going to 0 and the movement of two-step dynamics become very small. The general solution for Eq. (8) is given by

$$
b ^ {2} = \frac {1}{2} a \kappa + \frac {1}{1 6} \kappa^ {4} + C \exp (8 a \kappa^ {- 3}) \tag {9}
$$

for some constant  $C \in \mathbb{R}$ . As  $a$  decreases following Eq. (7), the trajectory converges toward the parabola  $b^2 = \frac{1}{2} a\kappa + \frac{1}{16}\kappa^4$ . Note that the convergence to the parabola is exponential with respect to  $a$ , so if  $a$  is initialized positive and not too small, it will converge to a minima that is very close to  $a = -\frac{1}{8}\kappa^3$  as shown in Fig. 3. This is a minimum that is just slightly flatter than the  $\kappa^2$ -EoS minimum.

![](images/17f7dfe1d64e8f302385ef61c1f1099747ed2425e6271565560e79a7af3d4bd2.jpg)  
Figure 3: Solutions of Eq. (8) ( $\kappa = 1$ )

# 3.2 CONVERGENCE ON EOS FOR PRODUCT OF 4 SCALAR

Now we state our convergence result on the 4 scalar objective under  $(a,b)$ -parameterization.

Theorem 3.1 (Sharpness Concentration). For a large enough absolute constant  $K$ , suppose  $\kappa < \frac{1}{2000\sqrt{2}} K^{-1}$ , and the initialization  $(a_0, b_0)$  satisfies  $a_0 \in (12\kappa^{\frac{5}{2}}, \frac{1}{4} K^{-2}\kappa^{-1})$  and  $b_0 \in (-K^{-1}, K^{-1}) \setminus \{0\}$ . Consider the GD trajectory characterized in Eq. (6) with fixed step size  $\kappa^2$  from  $(a_0, b_0)$ , for any  $\epsilon > 0$  there exists  $T = \mathcal{O}(K^{-2}\kappa^{-\frac{15}{2}} + \log(\epsilon^{-1}) + \log(|b_0|^{-1})\kappa^{-\frac{7}{2}})$  such that for all  $t > T$ ,  $|b_t| < \epsilon$  and  $a_t \in (-\frac{5}{3}\kappa^3, -\frac{1}{10}\kappa^3)$ .

Under the context of  $x, y$  coordinate and sharpness, Theorem 3.1 gives the following corollary:

Corollary 3.1 (Sharpness Concentration under  $(x,y)$ -Parameterization). For a large enough absolute constant  $K$ , suppose  $\eta < \frac{1}{800000} K^{-2}$ , and the initialization  $(x_0,y_0)$  satisfies  $x_0 \in (\check{x} + 13\eta^{\frac{5}{4}}, \check{x} + \frac{1}{5} K^{-2}\eta^{-\frac{1}{2}})$  and  $|x_0y_0 - 1| \in (0, K^{-1})$  where  $(\check{x}, \check{y})$  is the  $\eta$ -EoS minima defined in Definition 1. The GD trajectory characterized in Eq. (2) with fixed step size  $\eta$  from  $(x_0,y_0)$  will converge to a global minimum with sharpness  $\lambda \in (\frac{2}{\eta} - \frac{20}{3}\eta, \frac{2}{\eta})$ .

Note that when the step size  $\eta$  (and hence  $\kappa$ ) is relatively small, the final sharpness is very close to  $2 / \eta$ . The range of initialization that satisfies the requirement is quite large, in the original  $(x,y)$ -parameterization it contains a box of width  $\Theta(K^{-2}\eta^{-\frac{1}{2}})$  and height  $\Theta(K^{-1}\eta^{\frac{1}{2}})$ , many of the initial points can be far from the EoS-minimum.

The complete proofs are deferred to Appendix B.6. Here we discuss the proof sketch of Theorem 3.1. Our convergence analysis focuses on the 2-step update. It contains two phases:

# Phase 1. (Convergence to near parabola)

We consider initializations in region I, II, and III.

- In I,  $b'' - b$  is dominated by  $-b^3$  and  $(a, b)$  follows an exponential trajectory. We show that  $|b|$  decreases exponentially with respect to  $a$  and enters region II (Lemma 8).  
- In III,  $b'' - b$  is dominated by  $ab\kappa$  and  $(a,b)$  follows an elliptic trajectory centered at  $(0,0)$ . We show that  $|b|$  increases at superlinearly with respect to  $a$  and enters II (Lemma 9).  
- We also show that once  $(a, b)$  enters II, it will stay in II until it exits from the left and enters IV (Lemma 11). Thus after Stage 1, all initializations will be in IV.

# Phase 2. (Convergence along parabola)

- After  $(a, b)$  enters IV, we show that it will further converge to the parabola that  $|b^2 - \frac{1}{2} a\kappa - \frac{1}{16}\kappa^4| < \frac{1}{200}\kappa^4$  will be satisfied before  $a$  decreases to  $\kappa^{\frac{5}{2}}$  and enters V (Lemma 13).  
- Then we show that the inequality will be preserved in V while it moves left until it enters VI (Lemma 14).  
- In VI, the dynamics is again similar to III, but with  $a$  being negative. We conclude our proof by showing  $|b|$  will converge to 0 superlinearly with respect to  $a$  (Lemma 15).

![](images/5afc11872c60dcfa16490258da4ee1824e52ceebc42553820d87fb9770edf191.jpg)  
Figure 4: Convergence Diagram for GD on the 4-scalar example. The quiver arrows indicate the directions of local 2-step movement. This diagram is only for demonstration purpose and ratios are not exact.

Following Theorem 3.1, we can also formally characterize the sharpness adaptive phenomenon for a local region using the following corollary. The proof is deferred to Appendix B.6.2.

Corollary 3.2 (Sharpness Adaptivity). For a large enough constant  $K$ , fix any  $\alpha < \frac{1}{2000\sqrt{2}} K^{-1}$ .

For all initialization  $(x_0,y_0)$  in the region characterized by

$$
x _ {0} \in \left(\alpha^ {- 1} + \frac {1}{1 5} K ^ {- 2} \alpha^ {- 1}, \alpha^ {- 1} + \frac {1}{6} K ^ {- 2} \alpha^ {- 1}\right) \tag {10}
$$

and  $|x_0y_0 - 1| \in (0, K^{-1})$ , the  $GD$  trajectory from  $(x_0, y_0)$  characterized by Eq. (2) with any step size  $\eta \in (\alpha^2 - \frac{1}{10}K^{-2}\alpha^2, \alpha^2)$  will converge to a minima with sharpness  $\lambda \in (\frac{2}{\eta} - \frac{20}{3}\eta, \frac{2}{\eta})$ .

# 3.3 CONVERGENCE ON EOS FOR RANK-1 FACTORIZATION OF ISOTROPIC MATRIX

Inspired by the scalar factorization problem, we extend it to a rank-1 factorization of an isotropic matrix. In particular, we consider the following optimization problem:

$$
\min  _ {\boldsymbol {x}, \boldsymbol {y} \in \mathbb {R} ^ {d}} \frac {1}{4} \left\| \boldsymbol {I} _ {d \times d} - \boldsymbol {x} \boldsymbol {y} ^ {\top} \boldsymbol {x} \boldsymbol {y} ^ {\top} \right\| _ {\mathrm {F}} ^ {2} \tag {11}
$$

Similar to the under-parameterized case in Wang et al. (2022), this problem also guarantees the alignment between  $\mathbf{x}$  and  $\mathbf{y}$  if  $(\mathbf{x},\mathbf{y})$  is a global minimum, i.e.,  $\mathbf{x} = c\mathbf{y}$  for some  $c\in \mathbb{R}$ . To prove the convergence for Eq. (11) at the edge of stability, we first prove the alignment can be soon achieved. After the alignment, we prove the equivalence between this problem and the case of the product of 4 scalars, and prove the convergence of this problem.

We directly give the final theorem and the proof is deferred to Appendix C.

Theorem 3.2. For a large enough absolute constant  $K$ , with all the initialization  $(\pmb{x}_0, \pmb{y}_0)$  satisfying  $\pmb{x}_0 \sim \delta_xUnif(\mathbb{S}^{d-1})$ ,  $\pmb{y}_0 \sim \delta_yUnif(\mathbb{S}^{d-1})^2$ ,  $\delta_x\delta_y = \frac{1}{2}$ ,  $\delta_x \in (\check{x} + \frac{1}{80} K^{-2}\eta^{-\frac{1}{2}}, \check{x} + \frac{1}{8} K^{-2}\eta^{-\frac{1}{2}})$ , if step size  $\eta < \min\{\frac{K^{-4}}{8000000}, \frac{K^{-2}}{20000 + 2000(\log(d) - \log(\delta_0))}\}$ , and a multiplicative perturbation  $\pmb{y}_t' = \pmb{y}_t(1 + 2K^{-1})$  is performed at time  $t = t_p$  for some  $t_p > \mathcal{O}(-\log(\eta) + \log(d) - \log(\delta_0) + K^3)$ , then for any  $\epsilon > 0$ , with probability  $p > 1 - 2\delta_0 - 2\exp\{-\Omega(d)\}$  there exists  $T = \mathcal{O}(K^{-2}\kappa^{-\frac{15}{2}} - \log(\epsilon) - \log(\delta_0))$  such that for all  $t > T$ ,  $\mathcal{L}(x,y) < \epsilon$  and  $\|x_t\|^2 + \|y_t\|^2 \in (\frac{1}{\eta} - \frac{10}{3}\eta, \frac{1}{\eta})$ .

Note that we require an additional perturbation because we need to guarantee that the trajectory does not converge to an unstable point (where sharpness  $\lambda > 2 / \eta$ ). This was proved without perturbation for the scalar case but is more challenging in higher dimensions. The objective will still converge to a minimum very close to an  $\eta$ -EoS minimum.

# 4 DIFFERENCES IN DEGREE-2 AND HIGH DEGREE MODELS

In this section, we will look at some similar models of lower degree, and explain why for degree-2 models the sharpness of final converging point is often farther from  $2 / \eta$  compared to higher degree models. We will use similar methods as in Section 2 and Section 3 to gain intuition for the dynamics. We also empirically investigate the training dynamics for more general scalar networks.

# 4.1 GD ON 2-COMPONENT SCALAR FACTORIZATION

Previous works including (Chen & Bruna, 2022) and (Wang et al., 2022) have studied the dynamics of beyond EoS training on the problem of factorizing a single scalar or an isotropic matrix into two components. The objectives studied include  $\min_{\boldsymbol{x},\boldsymbol{y} \in \mathbb{R}^d} (\mu - \boldsymbol{x}^\top \boldsymbol{y})^2$ ,  $\min_{\boldsymbol{x},\boldsymbol{y} \in \mathbb{R}^d} \| \mu \boldsymbol{I}_d - \boldsymbol{x}\boldsymbol{y}^\top \|_F^2$ , and the corresponding scalar case  $\min_{x,y \in \mathbb{R}} (\mu - xy)^2$ . They were able to show that for initializations with sharpness greater than  $2/\eta$ , GD with constant learning rate  $\eta$  provably converges to a global minimum with sharpness less or equal to  $2/\eta$ . Empirically, the sharpness reduction process on these 2-component objectives will usually "overshoot" the EoS threshold and converge to a minima that is significantly flatter than the EoS minimum, and one does not observe the oscillation of sharpness around the EoS threshold (see Appendix A.3).

In this section we consider the scalar objective  $\min_{x,y\in \mathbb{R}}(1 - xy)^2$  since it is able to captures the major dynamical properties of those more complex objectives as discussed in Wang et al. (2022). As shown in Fig. 5, initializations with sharpness exceeding the EoS threshold will converge to a minima that is distinguishably flatter than the EoS minimum, and globally there is not a region of initialization that gives EoS convergence. Unlike the parabola for the 4 scalar case, the two step update travels in a roughly circular trajectory centered at the  $\kappa^2$ -EoS minimum as shown in Fig. 5a (right). Therefore locally we observe that sharper initializations tend to converge to flatter minima.

The difference between the 2 scalar and 4 scalar case can be easily explained by a local expansion. Using the same  $(c,d)$ -reparameterization and setting  $(a,b)$  to be the offset of  $(c,d)$  from the EoS minimum, the two step update of  $(a,b)$  under learning rate  $\kappa^2$  can be approximated by

$$
a ^ {\prime \prime} \approx a - \sqrt {2} b ^ {2} \kappa^ {3}, \quad b ^ {\prime \prime} \approx b + 4 \sqrt {2} a b \kappa . \tag {12}
$$

This is very similar to Eq. (7) except that we no longer have the  $-b^{3}$  term for the 2-step update on  $b$  which was attracting  $b$  close to 0. In this case, the ODE approximation  $\mathrm{db} / \mathrm{da} = 4a / b\kappa^2$  gives

the general solution  $b^{2} = 4(C - a^{2}) / \kappa^{2}$  for  $C\in \mathbb{R}^{+}$ , which corresponds to the family of ellipses centered at  $(0,0)$  and matches the two step trajectory in Fig. 5a.

![](images/424b1e54237a58904eeac82501766ef4123952491dfad2afa77ae5bec4a4d705.jpg)

![](images/a72d70cb24e1bb56c359f1ffa95e4d0e27115b05623b8d76ac252a52d61ed221.jpg)  
(a) Evolution of training loss, sharpness, and trajectory of GD on the 2 scalar example from the same initialization with different learning rates.

![](images/6a942c4fb3b3c7a1921f35eb7987533b71448a8a4f783f4f4fbe53b5947f124e.jpg)

![](images/c9dfcd6f0793e7956846780ec485a6b4d6151e187e2c883ad953b5971f135fa3.jpg)

![](images/a97aa39824b6891b52ebd89373f7efb318891aed099a556eaabd24e0e6925c7a.jpg)  
Figure 5: Beyond EoS training on product of two scalars. We run the same experiment as in Fig. 2 except for using objective  $\min_{x,y\in \mathbb{R}}(1 - xy)^2$ . Note that in this case the two-step trajectories form circular curves and converge to points that are farther from EoS minima.  
(b) Converging sharpness of initializations  $(\eta = 0.2)$

# 4.2 GD ON 3-LAYER SCALAR NETWORKS

Now we look into an interesting example with different behaviors around different EoS minima. We consider a 3-layer scalar network with objective  $\min_{x,y,z\in R}\frac{1}{2} (1 - xyz)^2$  and initialization with  $z = y$ . The equality of the last two entries will be preserved through training so the dynamics is two dimensional in terms of  $x$  and  $y$ . In the positive quadrant, the global minima is  $\sqrt{xy} = 1$  and there are two EoS minima. In Fig. 6, we plot the converging sharpness from different initializations in comparison with Fig. 2b and Fig. 5b.

Around the EoS minimum that the single entry  $x$  is small and the duplicated entries  $y$  are large (upper left of Fig. 6), the behavior is similar to the 2 scalar case (Fig. 5b) with no sharpness concentration. Around the EoS minima with large single entry and small duplicating entries (lower right of Fig. 6), we have a region of initialization (the red shaded area) with sharpness concentration similar to the 4 scalar case (Fig. 2b).

![](images/21f6a36a7b32feda014260a5d2b8e5edc9ab6b81d2357317ecfe553135f809cb.jpg)  
Figure 6: Converging sharpness of  $(x,y,y)$  parameterized initializations  $(\eta = 0.2)$ .

A heuristic explanation to this difference lies in the difference in the degree of the small entries. Around the EoS minima that the single entry is small, the local two-step approximation is similar to Eq. (12) and gives us elliptical two-step trajectories. Around the minima with small duplicating entries, the two-step approximation would contain the cubic term as in Eq. (7), which gives us both sharpness concentration and adaptivity. We note that empirically similar phenomenon is also true for more general scalar networks without symmetric initializations. See Appendix A.4.

# 5 GLOBAL TRAJECTORY AND CHAOS

There exists very limited global convergence analysis for constant step size gradient descent training beyond EoS on complicated non-convex objectives. Even for the product of 4 scalars, the boundary separating converging and diverging initializations (Fig. 7a) exhibits complicated fractal structures.

Moreover, we observe that for initializations close to such boundary, their GD training trajectories usually begin with a phase of chaotic oscillation which eventually "de-bifurcates" and converges to the parabolic two-step trajectory as discussed in Section 3. Similar oscillation phenomenon has also

been empirically observed by Ruiz-Garcia et al. (2021) in neural networks when they increase the learning rate and destabilize the network from a local trajectory.

![](images/5b98f4eaace591488ddce43639453c6062b22d52c2b0a1f4fda75be67e347671.jpg)  
(a) Converging and diverging initialization  $(\eta = 0.2)$ .

![](images/1aa4c123aa90b27faea65886ff17854ab4b1b9c8f7416fde9bbc0ec5c9bc6d32.jpg)  
(b) GD step-wise trajectory with asymmetric init.  $(\eta = 0.01)$ .

![](images/4dfd792d643d6092e886ddd9f791f6728ccdcb22feb180025351641be5668abb.jpg)  
Figure 7: Bifurcation behavior of GD on the 4 scalar example. In (a) we show the zoomed in version of the lower right part of Fig. 2b, the fractal boundary can be clearly observed. In (b), we run GD with  $\eta = 0.01$  starting from the asymmetric initialization  $(x_0, y_0) = (12.5, 0.05)$  close to the boundary of divergence until it converges close to the EoS minimum at around  $(10, 0.1)$ . In (c), we plot the trajectory with bifurcation under  $(a, b)$ -reparameterization and compare it with the bifurcation diagram of the approximated dynamical system characterized by Eq. (13).  
(c) GD trajectory and bifurcation diagram of approximated  $b^{\prime \prime}$

So what is causing the bifurcation? Previously, Ruiz-Garcia et al. (2021) attributed the phenomenon to the cascading effect of oscillation along multiple large eigendirections of the network. Yet this explanation is quite unsatisfying for our simple model as there is only one oscillating direction.

Looking closely to the trajectory (Fig. 7b), one will find it very similar to the bifurcation diagram of self-recurrent polynomial maps (such as the famous logistic map  $x_{t + 1} = rx_t(1 - x_t)$  parameterized by  $r$ ). In the 4-scalar example, the existence of such self-recurrent map is explicit since following Eq. (7), the approximate 2-step update of  $b$  can be rewritten as

$$
b ^ {\prime \prime} = b (1 + 8 a \kappa - 1 6 b ^ {2}). \tag {13}
$$

If we consider  $a$  to be relatively stationary, the trajectory of  $b$  will be locally characterized by the self-recurrent 1D nonlinear dynamical system Eq. (13) parameterized by  $a$ . In Fig. 7c, we compute the bifurcation diagram for Eq. (13) numerically and see that they are qualitatively similar.

Following this analogy, one may instantly relate the first bifurcating point with the EoS minima that the trajectory eventually converges to, and the non-bifurcating regime for the polynomial maps with the "sub-EoS regime" on the left (in Fig. 7b) of the EoS minima.

# 6 DISCUSSION AND CONCLUSION

In this paper we proposed a simple degree-4 model that captures the sharpness adaptivity and sharpness concentration phenomena that happen in gradient descent training of deep neural networks. The simplicity of the model allowed us to perform rigorous analysis on the training dynamics for a large local region. The analysis gives new insights on why the training dynamics of the degree-4 model is inherently different from the training dynamics of degree-2 models. We hope many of these observations can be generalized to highlight the difference between training dynamics of deeper networks and the shallower models.

There are still many open problems. Even for this simple objective the global dynamics is already very complicated. Is there a way to understand and leverage the fractal/bifurcation behavior in Section 5? Another interesting open question is how to characterize the training dynamics for other gradient based algorithms (such as GD with momentum and versions of adaptive gradient).

# REFERENCES

Kwangjun Ahn, Jingzhao Zhang, and Suvrit Sra. Understanding the unstable convergence of gradient descent. arXiv preprint arXiv:2204.01050, 2022.  
Sanjeev Arora, Zhiyuan Li, and Kaifeng Lyu. Theoretical analysis of auto rate-tuning by batch normalization. arXiv preprint arXiv:1812.03981, 2018.  
Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. Understanding gradient descent on edge of stability in deep learning. arXiv preprint arXiv:2205.09745, 2022.  
Lei Chen and Joan Bruna. On gradient descent convergence beyond the edge of stability. arXiv preprint arXiv:2206.04172, 2022.  
Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. arXiv preprint arXiv:2103.00065, 2021.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256. JMLR Workshop and Conference Proceedings, 2010.  
Stanislaw Jastrzebski, Maciej Szymczak, Stanislav Fort, Devansh Arpit, Jacek Tabor, Kyunghyun Cho, and Krzysztof Geras. The break-even point on optimization trajectories of deep neural networks. arXiv preprint arXiv:2002.09572, 2020.  
Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, and Guy Gur-Ari. The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint arXiv:2003.02218, 2020.  
Zhiyuan Li, Srinadh Bhojanapalli, Manzil Zaheer, Sashank Reddi, and Sanjiv Kumar. Robust training of neural networks using scale invariant architectures. In International Conference on Machine Learning, pp. 12656-12684. PMLR, 2022a.  
Zhouzi Li, Zixuan Wang, and Jian Li. Analyzing sharpness along gd trajectory: Progressive sharpening and edge of stability. arXiv preprint arXiv:2207.12678, 2022b.  
Kaifeng Lyu, Zhiyuan Li, and Sanjeev Arora. Understanding the generalization benefit of normalization layers: Sharpness reduction. arXiv preprint arXiv:2206.07085, 2022.  
Chao Ma, Lei Wu, and Lexing Ying. The multiscale structure of neural network loss functions: The effect on optimization and origin. arXiv preprint arXiv:2204.11326, 2022.  
Miguel Ruiz-Garcia, Ge Zhang, Samuel S Schoenholz, and Andrea J Liu. Tilting the playing field: Dynamical loss functions for machine learning. In International Conference on Machine Learning, pp. 9157-9167. PMLR, 2021.  
Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.  
Yuqing Wang, Minshuo Chen, Tuo Zhao, and Molei Tao. Large learning rate tames homogeneity: Convergence and balancing effect. International Conference on Learning Representations, 2022.  
Chen Xing, Devansh Arpit, Christos Tsirigotis, and Yoshua Bengio. A walk with sgd. arXiv preprint arXiv:1802.08770, 2018.  
Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W Mahoney. Pyhessian: Neural networks through the lens of the hessian. In 2020 IEEE international conference on big data (Big data), pp. 581-590. IEEE, 2020.
