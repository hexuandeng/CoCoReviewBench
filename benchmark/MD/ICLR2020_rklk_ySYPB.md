# PROVABLE ROBUSTNESS AGAINST ALL ADVERSARIAL  $l_{p}$ -PERTURBATIONS FOR  $p \geq 1$

Anonymous authors

Paper under double-blind review

# ABSTRACT

In recent years several adversarial attacks and defenses have been proposed. Often seemingly robust models turn out to be non-robust when more sophisticated attacks are used. One way out of this dilemma are provable robustness guarantees. While provably robust models for specific  $l_{p}$ -perturbation models have been developed, we show that they do not come with any guarantee against other  $l_{q}$ -perturbations. We propose a new regularization scheme, MMR-Universal, for ReLU networks which enforces robustness wrt  $l_{1}$  and  $l_{\infty}$ -perturbations and show how that leads to the first provably robust models wrt any  $l_{p}$ -norm for  $p \geq 1$ .

# 1 INTRODUCTION

The vulnerability of neural networks against adversarial manipulations (Szegedy et al., 2014; Goodfellow et al., 2015) is a problem for their deployment in safety critical systems such as autonomous driving and medical applications. In fact, small perturbations of the input which appear irrelevant or are even imperceivable to humans change the decisions of neural networks. This questions their reliability and makes them a target of adversarial attacks.

To mitigate the non-robustness of neural networks many empirical defenses have been proposed, e.g. by Gu & Rigazio (2015); Zheng et al. (2016); Papernot et al. (2016); Huang et al. (2016); Bastani et al. (2016); Madry et al. (2018), but at the same time more sophisticated attacks have proven these defenses to be ineffective (Carlini & Wagner, 2017; Athalye et al., 2018; Mosbach et al., 2018), with the exception of the adversarial training of Madry et al. (2018). However, even these  $l_{\infty}$ -adversarially trained models are not more robust than normal ones when attacked with perturbations of small  $l_{p}$ -norms with  $p \neq \infty$  (Sharma & Chen, 2019; Schott et al., 2019; Croce et al., 2019b; Kang et al., 2019). The situation becomes even more complicated if one extends the attack models beyond  $l_{p}$ -balls to other sets of perturbations (Brown et al., 2017; Engstrom et al., 2017; Hendrycks & Dietterich, 2019; Geirhos et al., 2019).

Another approach, which fixes the problem of overestimating the robustness of a model, is provable guarantees, which means that one certifies that the decision of the network does not change in a certain  $l_{p}$ -ball around the target point. Along this line, current state-of-the-art methods compute either the norm of the minimal perturbation changing the decision at a point (e.g. Katz et al. (2017); Tjeng et al. (2019)) or lower bounds on it (Hein & Andriushchenko, 2017; Raghunathan et al., 2018; Wong & Kolter, 2018). Several new training schemes like (Hein & Andriushchenko, 2017; Raghunathan et al., 2018; Wong & Kolter, 2018; Mirman et al., 2018; Croce et al., 2019a; Xiao et al., 2019; Gowal et al., 2018) aim at both enhancing the robustness of networks and producing models more amenable to verification techniques. However, all of them are only able to prove robustness against a single kind of perturbations, typically either  $l_{2}$ - or  $l_{\infty}$ -bounded, and not wrt all the  $l_{p}$ -norms simultaneously, as shown in Section 5. Some are also designed to work for a specific  $p$  (Mirman et al., 2018; Gowal et al., 2018), and it is not clear if they can be extended to other norms.

The only two papers which have shown, with some limitations, non-trivial empirical robustness against multiple types of adversarial examples are Schott et al. (2019) and Tramér & Boneh

(2019), which resist to  $l_0$ - resp.  $l_{1^-}$ ,  $l_{2^-}$  and  $l_{\infty}$ -attacks. However, they come without provable guarantees and Schott et al. (2019) is restricted to MNIST.

In this paper we aim at robustness against all the  $l_{p}$ -bounded attacks for  $p \geq 1$ . We study the non-trivial case where none of the  $l_{p}$ -balls is contained in another. If  $\epsilon_{p}$  is the radius of the  $l_{p}$ -ball for which we want to be provably robust, this requires:  $d^{\frac{1}{p} - \frac{1}{q}} \epsilon_{q} > \epsilon_{p} > \epsilon_{q}$  for  $p < q$  and  $d$  being the input dimension. We show that, for normally trained models, for the  $l_{1}$ - and  $l_{\infty}$ -balls we use in the experiments none of the adversarial examples constrained to be in the  $l_{1}$ -ball (i.e. results of an  $l_{1}$ -attack) belong to the  $l_{\infty}$ -ball, and vice versa. This shows that certifying the union of such balls is significantly more complicated than getting robust in only one of them, as in the case of the union the attackers have a much larger variety of manipulations available to fool the classifier.

We propose a technique which allows to train piecewise affine models (like ReLU networks) which are simultaneously provably robust to all the  $l_{p}$ -norms with  $p \in [1, \infty]$ . First, we show that having guarantees on the  $l_{1}$ - and  $l_{\infty}$ -distance to the decision boundary and region boundaries (the borders of the polytopes where the classifier is affine) is sufficient to derive meaningful certificates on the robustness wrt all  $l_{p}$ -norms for  $p \in (1, \infty)$ . In particular, our guarantees are independent of the dimension of the input space and thus go beyond a naive approach where one just exploits that all  $l_{p}$ -metrics can be upper- and lower-bounded wrt any other  $l_{q}$ -metric. Then, we extend the regularizer introduced in Croce et al. (2019a) so that we can directly maximize these bounds at training time. Finally, we show the effectiveness of our technique with experiments on four datasets, where the networks trained with our method are the first ones having non-trivial provable robustness wrt  $l_{1}$ -,  $l_{2}$ - and  $l_{\infty}$ -perturbations.

# 2 LOCAL PROPERTIES AND ROBUSTNESS GUARANTEES OF RELU NETWORKS

It is well known that feedforward neural networks (fully connected, CNNs, residual networks, DenseNets etc.) with piecewise affine activation functions, e.g. ReLU, leaky ReLU, yield continuous piecewise affine functions (see e.g. Arora et al. (2018); Croce & Hein (2018)). Croce et al. (2019a) exploit this property to derive bounds on the robustness of such networks against adversarial manipulations. In the following we recall the guarantees of Croce et al. (2019a) wrt a single  $l_{p}$ -perturbation which we extend in this paper to simultaneous guarantees wrt all the  $l_{p}$ -perturbations for  $p$  in  $[1,\infty]$ .

# 2.1 RELU NETWORKS AS PIECEWISE AFFINE FUNCTIONS

Let  $f: \mathbb{R}^d \to \mathbb{R}^K$  be a classifier with  $d$  being the dimension of the input space and  $K$  the number of classes. The classifier decision at a point  $x$  is given by  $\operatorname*{argmax}_{r=1,\dots,K} f_r(x)$ . In this paper

we deal with ReLU networks, that is with ReLU activation function (in fact our approach can be easily extended to any piecewise affine activation function e.g. leaky ReLU or other forms of layers leading to a piecewise affine classifier as in Croce et al. (2019b)).

Definition 2.1 A function  $f: \mathbb{R}^d \to \mathbb{R}$  is called piecewise affine if there exists a finite set of polytopes  $\{Q_r\}_{r=1}^M$  (referred to as linear regions of  $f$ ) such that  $\cup_{r=1}^M Q_r = \mathbb{R}^d$  and  $f$  is an affine function when restricted to every  $Q_r$ .

Denoting the activation function as  $\sigma$  ( $\sigma(t) = \max\{0, t\}$  if ReLU is used) and assuming  $L$  hidden layers, we have the usual recursive definition of  $f$  as

$$
g ^ {(l)} (x) = W ^ {(l)} f ^ {(l - 1)} (x) + b ^ {(l)}, \quad f ^ {(l)} (x) = \sigma (g ^ {(l)} (x)), \quad l = 1, \ldots , L,
$$

with  $f^{(0)}(x) \equiv x$  and  $f(x) = W^{(L + 1)}f^{(L)}(x) + b^{(L + 1)}$  the output of  $f$ . Moreover,  $W^{(l)} \in \mathbb{R}^{n_l \times n_{l - 1}}$  and  $b^{(l)} \in \mathbb{R}^{n_l}$ , where  $n_l$  is the number of units in the  $l$ -th layer ( $n_0 = d$ ,  $n_{L + 1} = K$ ).

For the convenience of the reader we summarize from Croce & Hein (2018) the description of the polytope  $Q(x)$  containing  $x$  and affine form of the classifier  $f$  when restricted to  $Q(x)$ . We assume that  $x$  does not lie on the boundary between polytopes (this is almost always

true as faces shared between polytopes are of lower dimension). Let  $\Delta^{(l)}, \Sigma^{(l)} \in \mathbb{R}^{n_l \times n_l}$  for  $l = 1, \ldots, L$  be diagonal matrices defined elementwise as

$$
\Delta^ {(l)} (x) _ {i j} = \left\{ \begin{array}{l l} \mathrm {s i g n} (f _ {i} ^ {(l)} (x)) & \text {i f} i = j, \\ 0 & \text {e l s e .} \end{array} \right., \qquad \Sigma^ {(l)} (x) _ {i j} = \left\{ \begin{array}{l l} 1 & \text {i f} i = j \text {a n d} f _ {i} ^ {(l)} (x) > 0, \\ 0 & \text {e l s e .} \end{array} \right..
$$

This allows us to write  $f^{(l)}(x)$  as composition of affine functions, that is

$$
f ^ {(l)} (x) = W ^ {(l)} \Sigma^ {(l - 1)} (x) \Big (W ^ {(l - 1)} \Sigma^ {(l - 2)} (x) \times \Big (\dots \Big (W ^ {(1)} x + b ^ {(1)} \Big) \dots \Big) + b ^ {(l - 1)} \Big) + b ^ {(l)},
$$

which we simplify as  $f^{(l)}(x) = V^{(l)}x + a^{(l)}$ , with  $V^{(l)}\in \mathbb{R}^{n_l\times d}$  and  $a^{(l)}\in \mathbb{R}^{n_l}$  given by

$$
V ^ {(l)} = W ^ {(l)} \Big (\prod_ {j = 1} ^ {l - 1} \Sigma^ {(l - j)} (x) W ^ {(l - j)} \Big) \mathrm {a n d} a ^ {(l)} = b ^ {(l)} + \sum_ {j = 1} ^ {l - 1} \Big (\prod_ {m = 1} ^ {l - j} W ^ {(l + 1 - m)} \Sigma^ {(l - m)} (x) \Big) b ^ {(j)}.
$$

A forward pass through the network is sufficient to compute  $V^{(l)}$  and  $b^{(l)}$  for every  $l$ . The polytope  $Q(x)$  is given as intersection of  $N = \sum_{l=1}^{L} n_l$  half spaces defined by

$$
Q (x) = \bigcap_ {l = 1, \dots , L} \bigcap_ {i = 1, \dots , n _ {l}} \left\{z \in \mathbb {R} ^ {d} \Big | \Delta^ {(l)} (x) _ {i i} \big (V _ {i} ^ {(l)} z + a _ {i} ^ {(l)} \big) \geq 0 \right\},
$$

Finally, the affine restriction of  $f$  to  $Q(x)$  is  $f(z)|_{Q(x)} = f^{(L + 1)}|_{Q(x)}(z) = V^{(L + 1)}z + a^{(L + 1)}$ . Let  $q$  be defined via  $\frac{1}{p} +\frac{1}{q} = 1$ ,

$$
d _ {p, l, j} ^ {B} (x) = \frac {\left| \left\langle V _ {j} ^ {(l)} , x \right\rangle + a _ {j} ^ {(l)} \right|}{\left\| V _ {j} ^ {(l)} \right\| _ {q}} \quad \text {a n d} \quad d _ {p, s} ^ {D} (x) = \frac {f _ {c} (x) - f _ {s} (x)}{\left\| V _ {c} ^ {(L + 1)} - V _ {s} ^ {(L + 1)} \right\| _ {q}}, \tag {1}
$$

for every  $l = 1, \dots, L$ ,  $j = 1, \dots, n_L$ ,  $s = 1, \dots, K$  and  $s \neq c$ , which represent the  $N$ $l_p$ -distances of  $x$  to the hyperplanes defining the polytope  $Q(x)$  and the  $K - 1$ $l_p$ -distances of  $x$  to the hyperplanes defining the decision boundaries in  $Q(x)$ . Finally, we define

$$
d _ {p} ^ {B} (x) = \min  _ {l = 1, \dots , L} \min  _ {j = 1, \dots , n _ {l}} d _ {p, l, j} ^ {B} (x) \quad \text {a n d} \quad d _ {p} ^ {D} (x) = \min  _ {\substack {s = 1, \dots , K \\ s \neq c}} d _ {p, s} ^ {D} \tag{2}
$$

as the minimum values of these two sets of distances (note that  $d_p^D(x) < 0$  if  $x$  is misclassified).

# 2.2 ROBUSTNESS GUARANTEES INSIDE LINEAR REGIONS

The  $l_{p}$ -robustness  $\mathbf{r}_p(x)$  of a classifier  $f$  at a point  $x$ , belonging to class  $c$ , wrt the  $l_{p}$ -norm is defined as the optimal value of the following optimization problem

$$
\mathbf {r} _ {p} (x) = \min  _ {\delta \in \mathbb {R} ^ {d}} \| \delta \| _ {p}, \quad \text {s . t h .} \quad \max  _ {l \neq c} f _ {l} (x + \delta) \geq f _ {c} (x + \delta), \quad x + \delta \in S, \tag {3}
$$

where is  $S$  a set of constraints on the input, e.g. pixel values of images have to be in [0, 1]. The  $l_{p}$ -robustness  $\mathbf{r}_p(x)$  is the smallest  $l_{p}$ -distance to  $x$  of a point which is classified differently from  $c$ . Thus,  $\mathbf{r}_p(x) = 0$  for misclassified points. The following theorem from Croce et al. (2019a), rephrased to fit the current notation, provides guarantees on  $\mathbf{r}_p(x)$ .

Theorem 2.1 (Croce et al. (2019a)) If  $d_p^B(x) < d_p^D(x)$ , then  $\mathbf{r}_p(x) \geq d_p^B(x)$ , while if  $|d_p^D(x)| \leq d_p^B(x)$ , then  $\mathbf{r}_p(x) = \max\{d_p^D(x), 0\}$ .

Although Theorem 2.1 holds for any  $l_{p}$ -norm with  $p \geq 1$ , it requires to compute  $d_p^B (x)$  and  $d_p^D (x)$  for every  $p$  individually. In this paper, exploiting this result and the geometrical arguments presented in Section 3, we show that it is possible to derive bounds on the robustness  $\mathbf{r}_p(x)$  for any  $p \in (1,\infty)$  using only information on  $\mathbf{r}_1(x)$  and  $\mathbf{r}_{\infty}(x)$ .

In the next section, we show that the straightforward usage of standard  $l_{p}$ -norms inequalities does not yield meaningful bounds on the  $l_{p}$ -robustness inside the union of the  $l_{1}$ - and  $l_{\infty}$ -ball, since these bounds depend on the dimension of the input space of the network.

![](images/b07a816a654932c2e0b2834f1bbdfe45495fa31add82176ad1704037bc2aa626.jpg)  
Figure 1: Visualization of the  $l_{2}$ -ball contained in the union resp. the convex hull of the union of  $l_{1}$ - and  $l_{\infty}$ -balls in  $\mathbb{R}^3$ . First column: co-centric  $l_{1}$ -ball (blue) and  $l_{\infty}$ -ball (black). Second: in red the largest  $l_{2}$ -ball completely contained in the union of  $l_{1}$ - and  $l_{\infty}$ -ball. Third: in green the convex hull of the union of the  $l_{1}$ - and  $l_{\infty}$ -ball. Fourth: the largest  $l_{2}$ -ball (red) contained in the convex hull. The  $l_{2}$ -ball contained in the convex hull is significantly larger than that contained in the union of  $l_{1}$ - and  $l_{\infty}$ -ball.

![](images/31feb10831d7c48c84a9a8987b2414292a400bb9519dd6ab09681348988b6d2a.jpg)

![](images/37ee812b961a8fd4f647a81a804cd20ccd7c72c5d63088db668c64b83555dc79.jpg)

![](images/0437aa816bfe2724dd3c9ac12e7669aa38636a43061e6000e45088c6ea9d1ce4.jpg)

# 3 MINIMAL  $l_{p}$ -NORM OF THE COMPLEMENT OF THE UNION OF  $l_{1}$ - AND  $l_{\infty}$ -BALL AND ITS CONVEX HULL

Let  $B_{1} = \{x\in \mathbb{R}^{d}:||x||_{1}\leq \epsilon_{1}\}$  and  $B_{\infty} = \{x\in \mathbb{R}^d:\| x\|_{\infty}\leq \epsilon_{\infty}\}$  be the  $l_{1}$ -ball of radius  $\epsilon_1 > 0$  and the  $l_{\infty}$ -ball of radius  $\epsilon_{\infty} > 0$  respectively, both centered at the origin in  $\mathbb{R}^d$ . We also assume  $\epsilon_{1}\in (\epsilon_{\infty},d\epsilon_{\infty})$ , so that  $B_{1}\not\subseteq B_{\infty}$  and  $B_{\infty}\nsubseteq B_{1}$ .

Suppose we can guarantee that the classifier does not change its label in  $U_{1,\infty} = B_1 \cup B_\infty$ . Which guarantee does that imply for all intermediate  $l_p$ -norms? This question can be simply answered by computing the minimal  $l_p$ -norms over  $\mathbb{R}^d \setminus U_{1,\infty}$ , namely  $\min_{x \in \mathbb{R}^d \setminus U_{1,\infty}} \| x \|_p$ . By the standard norm inequalities it holds, for every  $x \in \mathbb{R}^d$ , that

$$
\| x \| _ {p} \geq \| x \| _ {\infty} \quad \text {a n d} \quad \| x \| _ {p} \geq \| x \| _ {1} d ^ {\frac {1 - p}{p}},
$$

and thus a naive application of these inequalities yields the bound

$$
\min  _ {x \in \mathbb {R} ^ {d} \backslash U _ {1, \infty}} \| x \| _ {p} \geq \max  \left\{\epsilon_ {\infty}, \epsilon_ {1} d ^ {\frac {1 - p}{p}} \right\}. \tag {4}
$$

However, this naive bound does not take into account that we know that  $\| x\| _1\geq \epsilon_1$  and  $\| x\|_{\infty}\geq \epsilon_{\infty}$ . Our first result yields the exact value taking advantage of this information.

Proposition 3.1 If  $d \geq 2$  and  $\epsilon_1 \in (\epsilon_{\infty}, d \epsilon_{\infty})$ , then

$$
\min  _ {x \in \mathbb {R} ^ {d} \backslash U _ {1, \infty}} \| x \| _ {p} = \left(\epsilon_ {\infty} ^ {p} + \frac {\left(\epsilon_ {1} - \epsilon_ {\infty}\right) ^ {p}}{(d - 1) ^ {p - 1}}\right) ^ {\frac {1}{p}}. \tag {5}
$$

Thus a guarantee both for  $l_{1}$ - and  $l_{\infty}$ -ball yields a guarantee for all intermediate  $l_{p}$ -norms. However, for affine classifiers a guarantee for  $B_{1}$  and  $B_{\infty}$  implies a guarantee wrt the convex hull  $C$  of their union  $B_{1} \cup B_{\infty}$ . This can be seen by the fact that an affine classifier generates two half-spaces, and the convex hull of a set  $A$  is the intersection of all half-spaces containing  $A$ . Thus, inside  $C$  the decision of the affine classifier cannot change if it is guaranteed not to change in  $B_{1}$  and  $B_{\infty}$ , as  $C$  is completely contained in one of the half-spaces generated by the classifier (see Figure 1 for illustrations of  $B_{1}$ ,  $B_{\infty}$ , their union and their convex hull).

With the following theorem, we characterize, for any  $p \geq 1$ , the minimal  $l_p$ -norm over  $\mathbb{R}^d \setminus C$ .

Theorem 3.1 Let  $C$  be the convex hull of  $B_{1} \cup B_{\infty}$ . If  $d \geq 2$  and  $\epsilon_{1} \in (\epsilon_{\infty}, d\epsilon_{\infty})$ , then

$$
\min  _ {x \in \mathbb {R} ^ {d} \backslash C} \| x \| _ {p} = \frac {\epsilon_ {1}}{\left(\epsilon_ {1} / \epsilon_ {\infty} - \alpha + \alpha^ {q}\right) ^ {1 / q}}, \tag {6}
$$

where  $\alpha = \frac{\epsilon_1}{\epsilon_\infty} - \left\lfloor \frac{\epsilon_1}{\epsilon_\infty} \right\rfloor$  and  $\frac{1}{p} + \frac{1}{q} = 1$ .

Note that our expression in Theorem 3.1 is exact and not just a lower bound. Moreover, the minimal  $l_p$ -distance of  $\mathbb{R}^d \setminus C$  to the origin in Equation (6) is independent from the dimension  $d$ , in contrast to the expression for the minimal  $l_p$ -norm over  $\mathbb{R}^d \setminus U_{1,\infty}$  in (5) and

![](images/9ee1ae5c72655def25d27d617d11b7bf53b8994ff2319a16f6c0c405f55ec2d2.jpg)  
Figure 2: Comparison of the minimal  $l_{2}$ -norm over  $\mathbb{R}^d\backslash C$  (6) (blue),  $\mathbb{R}^d\backslash U_{1,\infty}$  (5) (red) and its naive lower bound (4) (green). We fix  $\epsilon_{\infty} = 1$  and show the results varying  $\epsilon_1\in (1,d)$ , for  $d = 784$  and  $d = 3072$ . We plot the value (or a lower bound in case of (4)) of the minimal  $\| x\| _2$ , depending on  $\epsilon_{1}$ , given by the different approaches (first and third plots). Moreover, we report (second and fourth plots) the ratios of the minimal  $\| x\| _2$  for  $\mathbb{R}^d\backslash \mathrm{conv}(B_1\cup B_\infty)$  and  $\mathbb{R}^d\setminus (B_1\cup B_\infty)$ . The values provided by (6) are much larger than those of (5).

![](images/766a555a041fd45a539c66d8e27774008ca4ca0830e6e3e900b2a75e28da2679.jpg)

![](images/bf4e21dd644e3686d2373365e306b308f74415f4d88cfeca5ba83fc1581a6518.jpg)

![](images/c922472b61bdeb3e707fc4072a0b20d3bf5b3ced1b22f617e30d9aaa663f887a.jpg)

its naive lower bound in (4), which are both decreasing for increasing  $d$  and  $p > 1$ . In Figure 1 we compare visually the largest  $l_{2}$ -balls (in red) fitting inside either  $U_{1,\infty}$  or the convex hull  $C$  in  $\mathbb{R}^3$ , showing that the one in  $C$  is clearly larger. In Figure 2 we provide a quantitative comparison in high dimensions. We plot the minimal  $l_{2}$ -norm over  $\mathbb{R}^d \setminus C$  (6) (blue) and over  $\mathbb{R}^d \setminus U_{1,\infty}$  (5) (red) and its naive lower bound (4) (green). We fix  $\| x \|_{\infty} = \epsilon_{\infty} = 1$  and vary  $\epsilon_1 \in [1,d]$ , with either  $d = 784$  (left) or  $d = 3072$  (right), i.e. the dimensions of the input spaces of MNIST and CIFAR-10. One sees clearly that the blue line corresponding to (6) is significantly higher than the other two. In the second and fourth plots of Figure 2 we show, for each  $\epsilon_1$ , the ratio of the  $l_{2}$ -distances given by (6) and (5). The maximal ratio is about 3.8 for  $d = 784$  and 5.3 for  $d = 3072$ , meaning that the advantage of (6) increases with  $d$ .

These two examples indicate that the  $l_{p}$ -balls contained in  $C$  can be a few times larger than those in  $U_{1,\infty}$ . Recall that we deal with piecewise affine networks. If we could enlarge the linear regions on which the classifier is affine so that it contains the  $l_{1}$ - and  $l_{\infty}$ -ball of some desired radii, we would automatically get the  $l_{p}$ -balls of radii given by Theorem 3.1 to fit in the linear regions. The next section formalizes the resulting robustness guarantees.

# 4 UNIVERSAL PROVABLE ROBUSTNESS WITH RESPECT TO ALL  $l_{p}$ -NORMS

Combining the results of Theorems 2.1 and 3.1, in the next theorem we derive lower bounds on the robustness of a continuous piecewise affine classifier  $f$ , e.g. a ReLU network, at a point  $x$  wrt any  $l_p$ -norm with  $p \geq 1$  using only  $d_1^B(x)$ ,  $d_1^D(x)$ ,  $d_\infty^B(x)$  and  $d_\infty^D(x)$  (see (2)).

Theorem 4.1 Let  $d_p^B (x)$ ,  $d_p^D (x)$  be defined as in (2) and define  $\rho_{1} = \min \{d_{1}^{B}(x),|d_{1}^{D}(x)|\}$  and  $\rho_{\infty} = \min \{d_{\infty}^{B}(x),|d_{\infty}^{D}(x)|\}$ . If  $d\geq 2$  and  $x$  is correctly classified, then

$$
\mathbf {r} _ {p} (x) \geq \frac {\rho_ {1}}{\left(\rho_ {1} / \rho_ {\infty} - \alpha + \alpha^ {q}\right) ^ {1 / q}}, \tag {7}
$$

for any  $p\in (1,\infty)$ , with  $\alpha = \frac{\rho_1}{\rho_\infty} -\left\lfloor \frac{\rho_1}{\rho_\infty}\right\rfloor$  and  $\frac{1}{p} +\frac{1}{q} = 1$ .

Croce et al. (2019a) add a regularization term to the training objective in order to enlarge the values of  $d_p^B(x)$  and  $d_p^D(x)$  for a fixed  $p$ , with  $x$  being the training points (note that they optimize  $d_p^D(x)$  and not  $|d_p^D(x)|$  to encourage correct classification).

Sorting in increasing order  $d_{p,l,j}^{B}$  and  $d_{p,s}^{D}$ , (see (1)), that is the  $l_{p}$ -distances to the hyperplanes defining  $Q(x)$  and to decision hyperplanes, and denoting them as  $d_{p,\pi_i^B}^B$  and  $d_{p,\pi_i^D}^D$  respectively, the Maximum Margin Regularizer (MMR) of Croce et al. (2019a) is defined as

$$
\operatorname {M M R} - l _ {p} (x) = \frac {1}{k _ {B}} \sum_ {i = 1} ^ {k _ {B}} \max  \left(0, 1 - \frac {d _ {p , \pi_ {i} ^ {B}} ^ {B} (x)}{\gamma_ {B}}\right) + \frac {1}{k _ {D}} \sum_ {i = 1} ^ {k _ {D}} \max  \left(0, 1 - \frac {d _ {p , \pi_ {i} ^ {D}} ^ {D} (x)}{\gamma_ {D}}\right). \qquad (8)
$$

It tries to push the  $k_{B}$  closest hyperplanes defining  $Q(x)$  farther than  $\gamma_{B}$  from  $x$  and the  $k_{D}$  closest decision hyperplanes farther than  $\gamma_{D}$  from  $x$  both wrt  $l_{p}$ -metric. In other words, MMR- $l_{p}$  aims at widening the linear regions around the training points so that they contain

$l_{p}$ -balls of radius either  $\gamma_{B}$  or  $\gamma_{D}$  centered in the training points. Using MMR- $l_{p}$  wrt a fixed  $l_{p}$ -norm, possibly in combination with the adversarial training of Madry et al. (2018), leads to classifiers which are empirically resistant wrt  $l_{p}$ -adversarial attacks and are easily verifiable by state-of-the-art methods to provide lower bounds on the true robustness.

For our goal of simultaneous  $l_{p}$ -robustness guarantees for all  $p \geq 1$ , we use the insights obtained from Theorem 4.1 to propose a combination of MMR- $l_{1}$  and MMR- $l_{\infty}$ , called MMR-Universal. It enhances implicitly robustness wrt every  $l_{p}$ -norm without actually computing and modifying separately all the distances  $d_{p}^{B}(x)$  and  $d_{p}^{D}(x)$  for the different values of  $p$ .

Definition 4.1 (MMR-Universal) Let  $x$  be a training point. We define the regularizer

$$
\begin{array}{l} M M R - U n i v e r s a l (x) = \frac {\lambda_ {1}}{k _ {B}} \sum_ {i = 1} ^ {k _ {B}} \max \left(0, 1 - \frac {d _ {1 , \pi_ {1 , i} ^ {B}} ^ {B} (x)}{\gamma_ {1}}\right) + \max \left(0, 1 - \frac {d _ {\infty , \pi_ {\infty , i} ^ {B}} ^ {B} (x)}{\gamma_ {\infty}}\right) \\ + \frac {\lambda_ {\infty}}{K - 1} \sum_ {i = 1} ^ {K - 1} \max \left(0, 1 - \frac {d _ {1 , \pi_ {1 , i} ^ {D}} ^ {D} (x)}{\gamma_ {1}}\right) + \max \left(0, 1 - \frac {d _ {\infty , \pi_ {\infty , i} ^ {D}} ^ {D} (x)}{\gamma_ {\infty}}\right), \\ \end{array}
$$

where  $k_{B}\in \{1,\dots ,N\}$ $\lambda_1,\lambda_\infty ,\gamma_1,\gamma_\infty >0$

We stress that, even if the formulation of MMR-Universal is based on MMR- $l_p$ , it is just thanks to the novel geometrical motivation provided by Theorem 3.1 and its interpretation in terms of robustness guarantees of Theorem 4.1 that we have a theoretical justification of MMR-Universal. Moreover, we are not aware of any other approach which can enforce simultaneously  $l_1$ - and  $l_{\infty}$ -guarantees, which is the key property of MMR-Universal.

The loss function which is minimized while training the classifier  $f$  is then, with  $\{(x_i,y_i)\}_{i = 1}^T$  being the training set and CE the cross-entropy loss,

$$
L \left(\{(x _ {i}, y _ {i}) \} _ {i = 1} ^ {T}\right) = \frac {1}{T} \sum_ {i = 1} ^ {T} \mathrm {C E} (f (x _ {i}), y _ {i}) + \mathrm {M M R - U n i v e r s a l} (x _ {i}).
$$

During the optimization our regularizer aims at pushing both the polytope boundaries and the decision hyperplanes farther than  $\gamma_{1}$  in  $l_{1}$ -distance and farther than  $\gamma_{\infty}$  in  $l_{\infty}$ -distance from the training point  $x$ , in order to achieve robustness close or better than  $\gamma_{1}$  and  $\gamma_{\infty}$  respectively. According to Theorem 4.1, this enhances also the  $l_{p}$ -robustness for  $p \in (1,\infty)$ . Note that if the projection of  $x$  on a decision hyperplane does not lie inside  $Q(x)$ ,  $d_p^D (x)$  is just an approximation of the signed distance to the true decision surface, in which case Croce et al. (2019a) argue that it is an approximation of the local Cross-Lipschitz constant which is also associated to robustness (see Hein & Andriushchenko (2017)). The regularization parameters  $\lambda_{1}$  and  $\lambda_{\infty}$  are used to balance the weight of the  $l_{1}$ - and  $l_{\infty}$ -term in the regularizer, and also wrt the cross-entropy loss. Note that the terms of MMR-Universal involving the quantities  $d_{p,\pi_{p,i}^{D}}^{D}(x)$  penalize misclassification, as they take negative values in this case.

Moreover, we take into account the  $k_B$  closest hyperplanes and not just the closest one as done in Theorems 2.1 and 4.1. This has two reasons: first, in this way the regularizer enlarges the size of the linear regions around the training points more quickly and effectively, given the large number of hyperplanes defining each polytope. Second, pushing many hyperplanes influences also the neighboring linear regions of  $Q(x)$ . This comes into play when, in order to get better bounds on the robustness at  $x$ , one wants to explore also a portion of the input space outside of the linear region  $Q(x)$ , which is where Theorem 4.1 holds. As noted in Raghunathan et al. (2018); Croce et al. (2019a); Xiao et al. (2019), established methods to compute lower bounds on the robustness are loose or completely fail when using normally trained models. In fact, their effectiveness is mostly related to how many ReLU units have stable sign when perturbing the input  $x$  within a given  $l_p$ -ball. This is almost equivalent to having the hyperplanes far from  $x$  in  $l_p$ -distance, which is what MMR-Universal tries to accomplish. This explains why in Section 5 we can certify the models trained with MMR-Universal with the methods of Wong & Kolter (2018) and Tjeng et al. (2019).

# 5 EXPERIMENTS

We compare the models obtained via our MMR-Universal regularizer to state-of-the-art methods for provable robustness and adversarial training. As evaluation criterion we use the robust test error, defined as the largest classification error when every image of the test set can be perturbed within a fixed set (e.g. an  $l_p$ -ball of radius  $\epsilon_p$ ). We focus on the  $l_p$ -balls with  $p \in \{1, 2, \infty\}$ . Since computing the robust test error is in general an NP-hard problem, we evaluate lower and upper bounds on it. The lower bound is the fraction of points for which an attack can change the decision with perturbations in the  $l_p$ -balls of radius  $\epsilon_p$  (adversarial samples), that is with  $l_p$ -norm smaller than  $\epsilon_p$ . For this task we use the PGD-attack (Kurakin et al. (2017); Madry et al. (2018); Tramer & Boneh (2019)) and the FAB-attack (Croce & Hein (2019)) for  $l_1$ ,  $l_2$  and  $l_\infty$ , MIP (Tjeng et al. (2019)) for  $l_\infty$  and the Linear Region Attack (Croce et al. (2019b)) for  $l_2$  and apply all of them (see C.3 for details). The upper bound is the portion of test points for which we cannot certify, using the methods of Tjeng et al. (2019) and Wong & Kolter (2018), that no  $l_p$ -perturbation smaller than  $\epsilon_p$  can change the correct class of the original input.

Smaller values of the upper bounds on the robust test error indicate models with better provable robustness. While lower bounds give an empirical estimate of the true robustness, it has been shown that they can heavily underestimate the vulnerability of classifiers (e.g. by Athalye et al. (2018); Mosbach et al. (2018)).

# 5.1 CHOICE OF  $\epsilon_{p}$

In choosing the values of  $\epsilon_{p}$  for  $p\in \{1,2,\infty \}$ , we try to be consistent with previous literature (e.g. Wong & Kolter (2018); Croce et al. (2019a)) for the values of  $\epsilon_{\infty}$  and  $\epsilon_{2}$ . Equation (6) provides, given  $\epsilon_{1}$  and  $\epsilon_{\infty}$ , a value at which one can expect  $l_{2}$ -robustness (approximately  $\epsilon_{2} = \sqrt{\epsilon_{1}\epsilon_{\infty}}$ ). Then we fix  $\epsilon_{1}$  such that this approximation is slightly larger than the desired  $\epsilon_{2}$ . We show in Table 1 the values chosen for  $\epsilon_{p}$ ,  $p\in \{1,2,\infty \}$ , and used to compute the robust test error in Table 2. Notice that for these values no  $l_{p}$ -ball is contained in the others.

Table 1: The values chosen for  $\epsilon_p$  on the different datasets and the expected  $l_2$ -robustness level (last column) given  $\epsilon_1$  and  $\epsilon_{\infty}$ , computed according to (6).  

<table><tr><td>dataset</td><td>ε1</td><td>ε∞</td><td>ε2</td><td>ε2 by (6)</td></tr><tr><td>MNIST / F-MNIST</td><td>1</td><td>0.1</td><td>0.3</td><td>0.3162</td></tr><tr><td>GTS</td><td>3</td><td>4/255</td><td>0.2</td><td>0.2170</td></tr><tr><td>CIFAR-10</td><td>2</td><td>2/255</td><td>0.1</td><td>0.1252</td></tr></table>

Moreover, we compute for the plain models the percentage of adversarial examples given by an  $l_{1}$ -attack (we use the PGD-attack) with budget  $\epsilon_{1}$  which have also  $l_{\infty}$ -norm smaller than or equal to  $\epsilon_{\infty}$ , and vice versa. These percentages are zero for all the datasets, meaning that being (provably) robust in the union of these  $l_{p}$ -balls is much more difficult than in just one of them (see also C.1).

# 5.2 MAIN RESULTS

We train CNNs on MNIST, Fashion-MNIST (Xiao et al. (2017)), German Traffic Sign (GTS) (Stallkamp et al. (2012)) and CIFAR-10 (Krizhevsky et al. (2014)). We consider several training schemes: plain training, the PGD-based adversarial training (AT) of Madry et al. (2018) and its extension to multiple  $l_{p}$ -balls in Tramér & Boneh (2019), the robust training (KW) of Wong & Kolter (2018); Wong et al. (2018), the MMR-regularized training (MMR) of Croce et al. (2019a), either alone or with adversarial training (MMR+AT) and the training with our regularizer MMR-Universal. We use AT, KW, MMR and MMR+AT wrt  $l_{2}$  and  $l_{\infty}$ , as these are the norms for which such methods have been used in the original papers. More details about the architecture and models in C.3.

In Table 2 we report test error (TE) computed on the whole test set and lower (LB) and upper (UB) bounds on the robust test error obtained considering the union of the three  $l_{p}$ -balls, indicated by  $l_{1} + l_{2} + l_{\infty}$  (these statistics are on the first 1000 points of the test set).

Table 2: We report, for the different datasets and training schemes, the test error (TE) and lower (LB) and upper (UB) bounds on the robust test error (in percentage) wrt the union of  $l_{p}$ -norms for  $p \in \{1, 2, \infty\}$  denoted as  $l_{1} + l_{2} + l_{\infty}$  (that is the largest test error possible if any perturbation in the union  $l_{1} + l_{2} + l_{\infty}$  is allowed). The training schemes compared are plain training, adversarial trainings of Madry et al. (2018); Tramér & Boneh (2019) (AT), robust training of Wong & Kolter (2018); Wong et al. (2018) (KW), MMR regularization of Croce et al. (2019a), MMR combined with AT (MMR+AT) and our MMR-Universal regularization. The models of our MMR-Universal are the only ones which have non-trivial upper bounds on the robust test error for all datasets.  
provable robustness against multiple perturbations  

<table><tr><td rowspan="2">model</td><td colspan="4">l1+l2+l∞</td><td colspan="3">l1+l2+l∞</td></tr><tr><td>TE</td><td>LB</td><td>UB</td><td>TE</td><td>LB</td><td>UB</td><td></td></tr><tr><td>plain</td><td>0.85</td><td>88.5</td><td>100</td><td>9.32</td><td>100</td><td>100</td><td></td></tr><tr><td>AT-l∞</td><td>0.82</td><td>4.7</td><td>100</td><td>11.54</td><td>26.3</td><td>100</td><td></td></tr><tr><td>AT-l2</td><td>0.87</td><td>25.9</td><td>100</td><td>8.10</td><td>98.8</td><td>100</td><td></td></tr><tr><td>AT-(l1,l2,l∞)</td><td>0.80</td><td>4.9</td><td>100</td><td>14.13</td><td>29.6</td><td>100</td><td></td></tr><tr><td>KW-l∞</td><td>1.21</td><td>4.8</td><td>100</td><td>21.73</td><td>43.6</td><td>100</td><td></td></tr><tr><td>KW-l2</td><td>1.11</td><td>10.3</td><td>100</td><td>13.08</td><td>66.7</td><td>86.8</td><td></td></tr><tr><td>MMR-l∞</td><td>1.65</td><td>10.4</td><td>100</td><td>14.51</td><td>36.7</td><td>100</td><td></td></tr><tr><td>MMR-l2</td><td>2.57</td><td>78.6</td><td>99.9</td><td>12.85</td><td>95.8</td><td>100</td><td></td></tr><tr><td>MMR+AT-l∞</td><td>1.19</td><td>4.1</td><td>100</td><td>14.52</td><td>31.8</td><td>100</td><td></td></tr><tr><td>MMR+AT-l2</td><td>1.73</td><td>15.3</td><td>99.9</td><td>13.40</td><td>66.5</td><td>99.1</td><td></td></tr><tr><td>MMR-Universal</td><td>3.04</td><td>12.4</td><td>20.8</td><td>18.57</td><td>43.5</td><td>52.9</td><td></td></tr><tr><td>plain</td><td>6.77</td><td>71.5</td><td>100</td><td>23.29</td><td>88.6</td><td>100</td><td></td></tr><tr><td>AT-l∞</td><td>6.83</td><td>64.0</td><td>100</td><td>27.06</td><td>52.5</td><td>100</td><td></td></tr><tr><td>AT-l2</td><td>8.76</td><td>59.0</td><td>100</td><td>25.84</td><td>62.1</td><td>100</td><td></td></tr><tr><td>AT-(l1,l2,l∞)</td><td>8.80</td><td>45.2</td><td>100</td><td>35.41</td><td>57.1</td><td>100</td><td></td></tr><tr><td>KW-l∞</td><td>15.57</td><td>87.8</td><td>100</td><td>38.91</td><td>51.9</td><td>100</td><td></td></tr><tr><td>KW-l2</td><td>14.35</td><td>57.6</td><td>100</td><td>40.24</td><td>54.0</td><td>100</td><td></td></tr><tr><td>MMR-l∞</td><td>13.32</td><td>71.3</td><td>99.6</td><td>34.61</td><td>58.7</td><td>100</td><td></td></tr><tr><td>MMR-l2</td><td>14.21</td><td>62.6</td><td>80.9</td><td>40.93</td><td>72.9</td><td>98.0</td><td></td></tr><tr><td>MMR+AT-l∞</td><td>14.89</td><td>82.8</td><td>100</td><td>35.38</td><td>50.8</td><td>100</td><td></td></tr><tr><td>MMR+AT-l2</td><td>15.34</td><td>58.1</td><td>84.8</td><td>37.78</td><td>61.3</td><td>99.9</td><td></td></tr><tr><td>MMR-Universal</td><td>15.98</td><td>51.6</td><td>52.4</td><td>46.96</td><td>63.8</td><td>64.6</td><td></td></tr></table>

The lower bounds  $l_{1} + l_{2} + l_{\infty}$ -LB are given by the fraction of test points for which one of the adversarial attacks wrt  $l_{1}, l_{2}$  and  $l_{\infty}$  is successful. The upper bounds  $l_{1} + l_{2} + l_{\infty}$ -UB are computed as the percentage of points for which at least one of the three  $l_{p}$ -balls is not certified to be free of adversarial examples (lower is better). This last one is the metric of main interest, since we aim at universally provably robust models. In C.2 we report the lower and upper bounds for the individual norms for every model.

MMR-Universal is the only method which can give non-trivial upper bounds on the robust test error for all datasets, while almost all other methods aiming at provable robustness have  $l_{1} + l_{2} + l_{\infty}$ -UB close to or at  $100\%$ . Notably, on GTS the upper bound on the robust test error of MMR-Universal is lower than the lower bound of all other methods except AT- $(l_{1}, l_{2}, l_{\infty})$ , showing that MMR-Universal provably outperforms existing methods which provide guarantees wrt individual  $l_{p}$ -balls, either  $l_{2}$  or  $l_{\infty}$ , when certifying the union  $l_{1} + l_{2} + l_{\infty}$ . The test error is slightly increased wrt the other methods giving provable robustness, but the same holds true for combined adversarial training AT- $(l_{1}, l_{2}, l_{\infty})$  compared to standard adv. training AT- $l_{2} / l_{\infty}$ . We conclude that MMR-Universal is the only method so far being able to provide non-trivial robustness guarantees for multiple  $l_{p}$ -balls in the case that none of them contains any other.

# 6 CONCLUSION

We have presented the first method providing provable robustness guarantees for the union of multiple  $l_{p}$ -balls beyond the trivial case of the union being equal to the largest one, establishing a baseline for future works. It is an interesting open question if the ideas developed in this paper can be integrated into other approaches towards provable robustness.

# REFERENCES

R. Arora, A. Basuy, P. Mianjyz, and A. Mukherjee. Understanding deep neural networks with rectified linear unit. In ICLR, 2018.  
A. Athalye, N. Carlini, and D. A. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In ICML, 2018.  
O. Bastani, Y. Ioannou, L. Lampropoulos, D. Vytiniotis, A. Nori, and A. Criminisi. Measuring neural net robustness with constraints. In NIPS, 2016.  
T. B. Brown, D. Mané, A. Roy, M. Abadi, and J. Gilmer. Adversarial patch. In NIPS 2017 Workshop on Machine Learning and Computer Security, 2017.  
N. Carlini and D. Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In ACM Workshop on Artificial Intelligence and Security, 2017.  
F. Croce and M. Hein. A randomized gradient-free attack on relu networks. In  $GCPR$ , 2018.  
F. Croce and M. Hein. Minimally distorted adversarial examples with a fast adaptive boundary attack. preprint, arXiv:1907.02044, 2019.  
F. Croce, M. Andriushchenko, and M. Hein. Provable robustness of relu networks via maximization of linear regions. In AISTATS, 2019a.  
F. Croce, J. Rauber, and M. Hein. Scaling up the randomized gradient-free adversarial attack reveals overestimation of robustness using established attacks. preprint, arXiv:1903.11359, 2019b.  
L. Engstrom, B. Tran, D. Tsipras, L. Schmidt, and A. Madry. A rotation and a translation suffice: Fooling CNNs with simple transformations. In NIPS 2017 Workshop on Machine Learning and Computer Security, 2017.  
R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In ICLR, 2019.  
I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. In ICLR, 2015.  
S. Gowal, K. Dvijotham, R. Stanforth, R. Bunel, C. Qin, J. Uesato, R. Arandjelovic, T. A. Mann, and P. Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. preprint, arXiv:1810.12715v3, 2018.  
S. Gu and L. Rigazio. Towards deep neural network architectures robust to adversarial examples. In ICLR Workshop, 2015.  
M. Hein and M. Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. In NIPS, 2017.  
D. Hendrycks and T. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In ICLR, 2019.  
R. Huang, B. Xu, D. Schuurmans, and C. Szepesvari. Learning with a strong adversary. In ICLR, 2016.  
D. Kang, Y. Sun, T. Brown, D. Hendrycks, and J. Steinhardt. Transfer of adversarial robustness between perturbation types. preprint, arXiv:1905.01034, 2019.  
G. Katz, C. Barrett, D. Dill, K. Julian, and M. Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In CAV, 2017.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. preprint, arXiv:1412.6980, 2014.

A. Krizhevsky, V. Nair, and G. Hinton. Cifar-10 (canadian institute for advanced research). 2014. URL http://www.cs.toronto.edu/~kriz/cifar.html.  
A. Kurakin, I. J. Goodfellow, and S. Bengio. Adversarial examples in the physical world. In *ICLR Workshop*, 2017.  
A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Valdu. Towards deep learning models resistant to adversarial attacks. In ICLR, 2018.  
M. Mirman, T. Gehr, and M. Vechev. Differentiable abstract interpretation for provably robust neural networks. In ICML, 2018.  
M. Mosbach, M. Andriushchenko, T. Trost, M. Hein, and D. Klakow. Logit pairing methods can fool gradient-based attacks. In NeurIPS 2018 Workshop on Security in Machine Learning, 2018.  
N. Papernot, P. McDonald, X. Wu, S. Jha, and A. Swami. Distillation as a defense to adversarial perturbations against deep networks. In IEEE Symposium on Security & Privacy, 2016.  
A. Raghunathan, J. Steinhardt, and P. Liang. Certified defenses against adversarial examples. In ICLR, 2018.  
L. Schott, J. Rauber, M. Bethge, and W. Brendel. Towards the first adversarially robust neural network model on MNIST. In ICLR, 2019.  
Y. Sharma and P. Chen. Attacking the madry defense model with  $l_{1}$ -based adversarial examples. In ICLR Workshop, 2019.  
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural Networks*, 32:323-332, 2012.  
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. In *ICLR*, pp. 2503-2511, 2014.  
V. Tjeng, K. Xiao, and R. Tedrake. Evaluating robustness of neural networks with mixed integer programming. In ICLR, 2019.  
F. Tramér and D. Boneh. Adversarial training and robustness for multiple perturbations. preprint, arXiv:1904.13000, 2019.  
E. Wong and J. Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In ICML, 2018.  
E. Wong, F. Schmidt, J. H. Metzen, and J. Z. Kolter. Scaling provable adversarial defenses. In NeurIPS, 2018.  
H. Xiao, K. Rasul, and R. Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. preprint, arXiv:1708.07747, 2017.  
K. Y. Xiao, V. Tjeng, N. M. Shafiullah, and A. Madry. Training for faster adversarial robustness verification via inducing relu stability. In ICLR, 2019.  
S. Zheng, Y. Song, T. Leung, and I. J. Goodfellow. Improving the robustness of deep neural networks via stability training. In CVPR, 2016.

A MINIMAL  $l_{p}$ -NORM OF THE COMPLEMENT OF THE UNION OF  $l_{1}$ - AND  $l_{\infty}$ -BALL AND ITS CONVEX HULL
