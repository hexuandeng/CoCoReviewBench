# CURVATURE-BASED ROBUSTNESS CERTIFICATES AGAINST ADVERSARIAL EXAMPLES

Anonymous authors

Paper under double-blind review

# ABSTRACT

A robustness certificate against adversarial examples is the minimum distance of a given input to the decision boundary of the classifier (or its lower bound). For any perturbation of the input with a magnitude smaller than the certificate value, the classification output will provably remain unchanged. Computing exact robustness certificates for deep classifiers is difficult in general since it requires solving a nonconvex optimization. In this paper, we provide computationally-efficient robustness certificates for deep classifiers with differentiable activation functions in two steps. First, we show that if the eigenvalues of the Hessian of the network (curvatures of the network) are bounded, we can compute a robustness certificate in the  $l_{2}$  norm efficiently using convex optimization. Second, we derive a computationally-efficient differentiable upper bound on the curvature of a deep network. We also use the curvature bound as a regularization term during the training of the network to boost its certified robustness against adversarial examples. Putting these results together leads to our proposed Curvature-based Robustness Certificate (CRC) and Curvature-based Robust Training (CRT). Our numerical results show that CRC outperforms CROWN's certificate by an order of magnitude while CRT leads to higher certified accuracy compared to standard adversarial training and TRADES.

# 1 INTRODUCTION

Modern neural networks achieve high accuracy on tasks such as image classification and speech recognition, but are known to be brittle to small, adversarially chosen perturbations of their inputs (Szegedy et al., 2014). A classifier which correctly classifies an image  $x$ , can be fooled by an adversary to misclassify an adversarial example  $\mathbf{x} + \delta$ , such that  $\mathbf{x} + \delta$  is indistinguishable from  $\mathbf{x}$  to a human. Adversarial examples can also fool systems when they are printed out on a paper and photographed with a smart phone (Kurakin et al., 2016a). Even in a black box threat model, where the adversary has no access to the model parameters, attackers could target autonomous vehicles by using stickers or paint to create an adversarial stop sign that the vehicle would interpret as a yield or another sign (Papernot et al., 2016). This trend is worrisome and suggests that these vulnerabilities need to be appropriately addressed before neural networks can be deployed in security critical applications.

In the last couple of years, several empirical defenses have been proposed for training classifiers to be robust against adversarial perturbations (Madry et al., 2018; Samangouei et al., 2018; Zhang et al., 2019; Papernot et al., 2016; Kurakin et al., 2016b; Miyato et al., 2017; Zheng et al., 2016) Although these defenses robustify classifiers to particular types of attacks, they can be still vulnerable against stronger attacks (Athalye et al., 2018; Carlini & Wagner, 2017; Uesato et al., 2018; Athalye & Carlini, 2018). For example, (Athalye et al., 2018) showed most of the empirical defenses proposed in ICLR 2018 can be broken by developing tailored attacks for each of them.

To end the cycle between defenses and attacks, a line of work on certified defenses has gained attention where the goal is to train classifiers whose predictions are provably robust within some given region (Huang et al., 2016; Katz et al., 2017; Ehlers, 2017; Carlini et al., 2017; Cheng et al., 2017; Lomuscio & Maganti, 2017; Dutta et al., 2018; Fischetti & Jo, 2018; Bunel et al., 2017; Wang et al., 2018a; Wong & Kolter, 2017; Wang et al., 2018b; Wong et al., 2018; Raghunathan et al., 2018b;a; Dvijotham et al., 2018a;b; Croce et al., 2018; Singh et al., 2018; Gowal et al., 2018; Gehr et al., 2018; Mirman et al., 2018; Zhang et al., 2018b; Weng et al., 2018). These methods, however, do not scale to large and practical networks used in solving modern machine learning problems. Another line of

defense work focuses on randomized smoothing where the prediction is robust within some region around the input with a user-chosen probability (Liu et al., 2017; Cao & Gong, 2017; Lécuyer et al., 2018; Li et al., 2018; Cohen et al., 2019a; Salman et al., 2019). Although these methods can scale to large networks, certifying robustness with probability close to 1 often requires generating a large number of noisy samples around the input which leads to high test-time computational complexity.

If the classifier  $f(\cdot)$  was linear, the distance of an input point  $\mathbf{x}$  to its decision boundary (i.e. the robustness certificate) can be computed efficiently using a convex optimization. For example, the  $l_{2}$  robustness certificate in that case would be equal to  $|f(\mathbf{x})| / \| \nabla_{\mathbf{x}}f(\mathbf{x})\|$ . However, modern classifiers based on neural networks are not linear and and can have non-zero curvatures in different parts of the input domain. The deviation of the classifier from the linear model makes the robustness certification problem to be a non-convex optimization which is often difficult to solve exactly. However, if we could compute global bounds on the maximum curvature values of the classification network, one may be able to compute computationally-efficient lower bounds on the robustness certificate even for non-linear deep classifiers. This is the key intuition of our results in this paper.

In this work, we derive a global bound on the Lipschitz constant of the gradient of deep neural networks with differentiable activation functions (such as sigmoid, tanh, softplus, etc.). This provides an upper bound on the magnitude of the eigenvalues of the Hessian or the curvature values of the classification network. Using this global curvature bound and for the  $l_{2}$  metric, we tackle both the certification and attack problems. In the certification problem and for a given pre-trained classifier, we provide a computationally-efficient lower bound on the distance of a point to the classification decision boundary. In the related attack problem, for a given input and a region around it, our goal is to find a perturbed input (an adversarial example) that maximizes the loss inside the given region. The outcome of the attack problem is then used in the adversarial training procedure (Madry et al., 2017) to further robustify the network. Furthermore, our global curvature bound is differentiable and we show that adding it to the loss function as a regularizer during the training boosts certified robustness measures. In the main text, we explain our key results using the framework of robustness certification while the extension to the attack problem is mainly discussed in the appendix.

We note that other recent works (e.g. Moosavi Dezfooli et al. (2019); Qin et al. (2019)) empirically show that using an estimate of curvature at inputs as a regularizer leads to empirical robustness on par with the adversarial training. In this work, however, we use a provable global upper bound on the curvature (and not an estimate) as a regularizer and show that it results in high certified robustness. Moreover, previous works have tried to certify robustness by bounding the Lipschitz constant of the neural network (Szegedy et al., 2014; Peck et al., 2017; Zhang et al., 2018b; Anil et al., 2018; Hein & Andriushchenko, 2017). Our approach, however, is based on bounding the Lipschitz constant of the gradient of deep neural networks. We discuss existing works in more details in Appendix A.

Below, we state the key theoretical results of this paper informally while detailed statements of these results are presented in Section 4.

Theorem (informal) 1. Let  $\mathbf{z}_i^{(L)}$  denotes the  $i^{\text{th}}$  logit of an  $L$  layer fully-connected neural network with differentiable activation functions. Then, the curvature of the neural network function is globally bounded as follows:

$$
m \mathbf {I} \preccurlyeq \nabla_ {\mathbf {x}} ^ {2} \mathbf {z} _ {i} ^ {(L)} \preccurlyeq M \mathbf {I}, \qquad \forall \mathbf {x} \in \mathbb {R} ^ {D}
$$

where  $m$  and  $M$  can be computed efficiently using parameters of the network.

This result along with the min-max theorem leads to the following curvature robustness certificate:

Theorem (informal) 2. Consider a network whose curvature values are bounded. For a given input  $\mathbf{x}^{(0)}$  with the true label  $y$  and the attack target  $t$  ( $t \neq y$ ), let  $p_{cert}^*$  denote the exact robustness certificate, i.e. the distance of  $\mathbf{x}^{(0)}$  to the decision boundary. We can efficiently compute  $d_{cert}^*$  such that  $d_{cert}^* \geq p_{cert}^*$ . Moreover, if the solution  $\mathbf{x}^{(cert)}$  for  $d_{cert}^*$  satisfies  $\mathbf{z}_y^{(L)} = \mathbf{z}_t^{(L)}$ , then  $d_{cert}^* = p_{cert}^*$ .

We have similar results for the attack problem. For simplicity, we summarize definitions of  $p_{\text{cert}}^*, d_{\text{cert}}^*, p_{\text{attack}}^*, d_{\text{attack}}^*$  in Table 1.

In summary, in this paper, we make the following contributions:

<table><tr><td></td><td>Certificate problem (−) = cert</td><td>Attack problem (−) = attack</td></tr><tr><td>primal problem, p*(-)</td><td>minf(x)=0 1/2||x - x(0)||2</td><td>min||x-x(0)||≤ρ f(x)</td></tr><tr><td>dual function, d(−)(η)</td><td>minx 1/2||x - x(0)||2 + ηf(x)</td><td>minx f(x) + η/2(||x - x(0)||2 - ρ2)</td></tr><tr><td>When is dual solvable?</td><td>-1/M ≤ η ≤ -1/m</td><td>-m ≤ η</td></tr><tr><td>dual problem, d*(-)</td><td>max-1/M≤η≤-1/m dcert(η)</td><td>max-m≤η dcert(η)</td></tr><tr><td>When primal = dual?</td><td>f(x(cert)) = 0</td><td>||x(attack) - x(0)|| = ρ</td></tr></table>

Table 1: A summary of various primal and dual concepts used in the paper.  $f$  denotes the function of the decision boundary, i.e.  ${\mathbf{z}}_{y}^{\left( L\right) } - {\mathbf{z}}_{t}^{\left( L\right) }$  where  $y$  is the true label and  $t$  is the attack target.  $m$  and  $M$  are lower and upper bounds on the smallest and largest eigenvalues of the Hessian of  $f$  ,respectively.

- We provide global bounds on the eigenvalues of the Hessian of a deep neural network with differentiable activation functions (Theorem 3 and Theorem 4). In addition to the adversarial robustness problem, these bounds may be of an independent interest for readers.  
- Using the global curvature bounds, we develop computationally efficient methods for both the robustness certification as well as the adversarial attack problems (Theorems 1 and 2).  
- We show that using our proposed curvature bounds as a regularizer during training leads to improved certified accuracy on 2,3 and 4 layer networks (on the MNIST dataset) compared to standard adversarial training with PGD (Madry et al., 2018) as well as TRADES (Zhang et al., 2019). Moreover, our robustness certificate (CRC) outperforms CROWN's certificate (Zhang et al., 2018b) significantly while taking less time to compute.

# 2 NOTATION AND PROBLEM SETUP

Consider a fully connected neural network with  $L$  layers and  $N_{I}$  neurons in the  $I^{th}$  layer ( $L \geq 2$  and  $I \in [L]$ ) for a multi-label classification problem with  $C$  classes ( $N_{L} = C$ ). The corresponding function of the neural network is  $\mathbf{z}^{(L)}: \mathbf{R}^{D} \to \mathbf{R}^{C}$  where  $D$  is the dimension of the input. For an input  $\mathbf{x}$ , we use  $\mathbf{z}^{(I)}(\mathbf{x}) \in \mathbf{R}^{N_{I}}$  and  $\mathbf{a}^{(I)}(\mathbf{x}) \in \mathbf{R}^{N_{I}}$  to denote the input (before applying the activation function) and output (after applying the activation function) of neurons in the  $I^{th}$  hidden layer of the network, respectively. To simplify notation and when no confusion arises, we make the dependency of  $\mathbf{z}^{(I)}$  and  $\mathbf{a}^{(I)}$  to  $\mathbf{x}$  implicit. We define  $\mathbf{a}^{(0)}(\mathbf{x}) = \mathbf{x}$  and  $N_{0} = D$ .

With a fully connected architecture, each  $\mathbf{z}^{(I)}$  and  $\mathbf{a}^{(I)}$  is computed using a transformation matrix  $\mathbf{W}^{(I)} \in R^{N_I \times N_{I-1}}$ , the bias vector  $\mathbf{b}^{(I)} \in R^{N_I}$  and an activation function  $\sigma(.)$  as follows:

$$
\mathbf {z} ^ {(I)} (\mathbf {x}) = \mathbf {W} ^ {(I)} \mathbf {a} ^ {(I - 1)} (\mathbf {x}) + \mathbf {b} ^ {(I)}, \qquad \mathbf {a} ^ {(I)} (\mathbf {x}) = \sigma \left(\mathbf {z} ^ {(I)} (\mathbf {x})\right).
$$

We use  $(\mathbf{z}_i^{(L)} - \mathbf{z}_j^{(L)})(\mathbf{x})$  as a shorthand for  $\mathbf{z}_i^{(L)}(\mathbf{x}) - \mathbf{z}_j^{(L)}(\mathbf{x})$ .

We use  $[p]$  to denote the set  $\{1, \ldots, p\}$  and  $[p, q]$ ,  $p \leq q$  to denote the set  $\{p, p + 1, \ldots, q\}$ . We use small letters  $i, j, k$  etc to denote the index over a vector or rows of a matrix and capital letters  $I, J$  to denote the index over layers of network. The element in the  $i^{th}$  position of a vector  $\mathbf{v}$  is given by  $\mathbf{v}_i$ , the vector in the  $i^{th}$  row of a matrix  $\mathbf{A}$  is  $\mathbf{A}_i$  while the element in the  $i^{th}$  row and  $j^{th}$  column of  $\mathbf{A}$  is  $\mathbf{A}_{i,j}$ . We use  $\| \mathbf{v} \|$  and  $\| \mathbf{A} \|$  to denote the 2-norm and the operator 2-norm of the vector  $\mathbf{v}$  and the matrix  $\mathbf{A}$ , respectively. We use  $|\mathbf{v}|$  and  $|\mathbf{A}|$  to denote the vector and matrix constructed by taking the elementwise absolute values. We use  $\lambda_{max}(\mathbf{A})$  and  $\lambda_{min}(\mathbf{A})$  to denote the largest and smallest eigenvalues of a symmetric matrix  $\mathbf{A}$ . We use  $\text{diag}(\mathbf{v})$  to denote the diagonal matrix constructed by placing each element of  $\mathbf{v}$  along the diagonal. We use  $\odot$  to denote the Hadamard Product,  $\mathbf{I}$  to denote the identity matrix. We use  $\preccurlyeq$  and  $\succcurlyeq$  to denote Linear Matrix Inequalities (LMIs) such that given two symmetric matrices  $\mathbf{A}$  and  $\mathbf{B}$  where  $\mathbf{A} \succcurlyeq \mathbf{B}$  means  $\mathbf{A} - \mathbf{B}$  Positive Semi-Definite (PSD).

# 3 USING DUALITY TO SOLVE THE ATTACK AND CERTIFICATE PROBLEMS

Consider an input  $\mathbf{x}^{(0)}$  with true label  $y$  and the attack target  $t$ . In the certificate problem, our goal is to find a lower bound of the minimum  $l_{2}$  distance between  $\mathbf{x}^{(0)}$  and the decision boundary,  $\mathbf{z}_y^{(L)} = \mathbf{z}_t^{(L)}$ . The problem for solving the exact distance (primal) can be written as:

$$
p _ {c e r t} ^ {*} = \min  _ {\mathbf {z} _ {y} ^ {(L)} (\mathbf {x}) = \mathbf {z} _ {t} ^ {(L)} (\mathbf {x})} \left[ \frac {1}{2} \left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} \right] = \min  _ {\mathbf {x}} \max  _ {\eta} \left[ \frac {1}{2} \left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} + \eta \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) (\mathbf {x}) \right]. \quad (1)
$$

However, solving the above problem can be hard in general. Using the minimax theorem (primal  $\geq$  dual), we can write the dual of the above problem as follows:

$$
p _ {c e r t} ^ {*} \geq \max  _ {\eta} d _ {c e r t} (\eta), \quad d _ {c e r t} (\eta) = \min  _ {\mathbf {x}} \left[ \frac {1}{2} \left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} + \eta \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) (\mathbf {x}) \right]. \tag {2}
$$

From the theory of duality, we know that  $d_{\text{cert}}(\eta)$  for each value of  $\eta$  gives a lower bound on the exact certification value (the primal solution)  $p_{\text{cert}}^*$ . However, since  $\mathbf{z}_y^{(L)} - \mathbf{z}_t^{(L)}$  is non-convex, solving  $d_{\text{cert}}(\eta)$  for every  $\eta$  can be difficult. In the next section, we will prove that the curvature of the function  $\mathbf{z}_y^{(L)} - \mathbf{z}_t^{(L)}$  is bounded globally:

$$
m \mathbf {I} \preccurlyeq \nabla_ {\mathbf {x}} ^ {2} \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) \preccurlyeq M \mathbf {I} \quad \forall \mathbf {x} \in \mathbb {R} ^ {D}, m <   0, M > 0 \tag {3}
$$

In this case, we have the following theorem:

Theorem 1.  $d_{\text{cert}}(\eta)$  is a convex optimization problem for  $-1 / M \leq \eta \leq -1 / m$ . Moreover, If  $\mathbf{x}^{(\text{cert})}$  is the solution to  $d_{\text{cert}}^*$  such that  $\mathbf{z}_y^{(L)}(\mathbf{x}^{(\text{cert})}) = \mathbf{z}_t^{(L)}(\mathbf{x}^{(\text{cert})})$ , then  $p_{\text{cert}}^* = d_{\text{cert}}^*$ .

Below, we briefly outline the proof while the full proof is presented in Appendix D.1. The Hessian of the objective function of the dual  $d_{\text{cert}}(\eta)$ , i.e. the function inside the  $\min_{\mathbf{x}}$  is given by:

$$
\nabla_ {\mathbf {x}} ^ {2} \left[ \frac {1}{2} \left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} + \eta \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) (\mathbf {x}) \right] = \mathbf {I} + \eta \nabla_ {\mathbf {x}} ^ {2} \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right)
$$

From equation (3), we know that the eigenvalues of  $\mathbf{I} + \eta \nabla_{\mathbf{x}}^{2}(\mathbf{z}_{y}^{(L)} - \mathbf{z}_{t}^{(L)})$  are bounded between  $(1 + \eta m, 1 + \eta M)$  if  $\eta \geq 0$ , and in  $(1 + \eta M, 1 + \eta m)$  if  $\eta \leq 0$ . In both cases, we can see that for  $-1 / M \leq \eta \leq -1 / m$ , all eigenvalues will be non-negative, making the objective function convex. When  $\mathbf{x}^{(cert)}$  satisfies  $\mathbf{z}_{y}^{(L)} = \mathbf{z}_{t}^{(L)}$ ,  $d_{cert}^{*} = 1 / 2\| \mathbf{x}^{(cert)} - \mathbf{x}^{(0)}\|^{2}$ , using the duality theorem and definition of  $p_{cert}^{*}$ , we get  $p_{cert}^{*} = d_{cert}^{*}$ .

Next, we consider the attack problem. The goal here is to find an adversarial example inside an  $l_{2}$  ball of radius  $\rho$  such that  $\mathbf{z}_y^{(L)} - \mathbf{z}_t^{(L)}$  is minimized. Using similar arguments, we can get the following theorem for the attack problem ( $p_{attack}^*$ ,  $d_{attack}^*$  and  $d_{attack}$  are defined in Table 1):

Theorem 2.  $d_{attack}(\eta)$  is a convex optimization problem for  $-m \leq \eta$ . Moreover, if  $\mathbf{x}^{(attack)}$  is the solution to  $d_{attack}^*$  such that  $\left\| \mathbf{x}^{(attack)} - \mathbf{x}^{(0)} \right\| = \rho$ ,  $p_{attack}^* = d_{attack}^*$ .

The proof is presented in Appendix D.2. Both Theorems 1, 2 hold for any non-convex function with continuous gradients. They can also be of interest in problems such as optimization of neural nets.

Using Theorems 1 and 2, we have the following definitions for certification and attack optimizations:

Definition 1. (Curvature-based Certificate Optimization) Given an input  $\mathbf{x}^{(0)}$  with true label  $y$ , the false target  $t$ , we define  $(\eta^{(\text{cert})}, \mathbf{x}^{(\text{cert})})$  as the solution of the following max-min optimization:

$$
\max _ {- 1 / M \leq \eta \leq - 1 / m} \min _ {\mathbf {x}} \left[ \frac {1}{2} \left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} + \eta \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) (\mathbf {x}) \right]
$$

We refer to  $\left\| \mathbf{x}^{(cert)} - \mathbf{x}^{(0)}\right\|$  as the Curvature-based Robustness Certificate (CRC).

Definition 2. (Curvature-based Attack Optimization) Given input  $\mathbf{x}^{(0)}$  with label  $y$ , false target  $t$ , and the  $l_{2}$  ball radius  $\rho$ , we define  $(\eta^{(\text{attack})}, \mathbf{x}^{(\text{attack})})$  as the solution of the following optimization:

$$
\max _ {\eta \geq - m} \min  _ {\mathbf {x}} \left[ \frac {\eta}{2} \left(\left\| \mathbf {x} - \mathbf {x} ^ {(0)} \right\| ^ {2} - \rho^ {2}\right) + \left(\mathbf {z} _ {y} ^ {(L)} - \mathbf {z} _ {t} ^ {(L)}\right) (\mathbf {x}) \right].
$$

When  $\mathbf{x}^{(attack)}$  is used for training in an adversarial training framework, we call the method the Curvature-based Robust Training (CRT).

Since both curvature-based certificate and attack optimizations are convex optimization problems, any convex optimization solver can be used to solve them. In our implementation, we use majorization-minimization to solve the dual function for a given  $\eta$  and bisection method to maximize over  $\eta$ . Our method satisfies linear convergence. More details are given in Appendix C.4 and C.5.

# 4 CURVATURE BOUNDS FOR DEEP NETWORKS

In this section, we provide a computationally efficient approach to compute the curvature bounds for neural networks with differentiable activation functions. To the best of our knowledge, there is no prior work on finding provable bounds on the curvature values of deep neural networks. Our results rely on a closed form expression for the Hessian of the  $i^{th}$  logit as a sum of matrix products (Section 4.1). After establishing this result, we first derive curvature bounds for a two-layer network in Section 4.2 and then extend the bounds to deeper networks in Section 4.3.

# 4.1 CLOSED FORM EXPRESSION FOR THE HESSIAN

Using the chain rule of second derivatives, we can derive  $\nabla_{\mathbf{x}}^{2}\mathbf{z}_{i}^{(L)}$  as a sum of matrix products:

Lemma 1. Given an  $L$  layer neural network, the Hessian of the  $i^{th}$  hidden unit with respect to the input  $\mathbf{x}$ , i.e.  $\nabla_{\mathbf{x}}^{2}\mathbf{z}_{i}^{(L)}$  is given by the following formula:

$$
\nabla_ {\mathbf {x}} ^ {2} \mathbf {z} _ {i} ^ {(L)} = \sum_ {I = 1} ^ {L - 1} \left(\mathbf {B} ^ {(I)}\right) ^ {T} d i a g \left(\mathbf {F} _ {i} ^ {(L, I)} \odot \sigma^ {\prime \prime} \left(\mathbf {z} ^ {(I)}\right)\right) \mathbf {B} ^ {(I)}
$$

where  $\mathbf{B}^{(I)}$  is the Jacobian of  $\mathbf{z}^{(I)}$  with respect to  $\mathbf{x}$  (dimensions  $N_I \times D$ ), and  $\mathbf{F}^{(L,I)}$  is the Jacobian of  $\mathbf{z}^{(L)}$  with respect to  $\mathbf{a}^{(I)}$  (dimensions  $N_L \times N_I$ ).

The proof is presented in Appendix D.3. Using the chain rule of gradient, we can compute  $\mathbf{B}^{(I)}$ ,  $\mathbf{F}^{(L,I)}$  matrices in Lemma 1 recursively as follows:

$$
\mathbf {B} ^ {(1)} = \mathbf {W} ^ {(1)} \quad \mathbf {B} ^ {(I)} = \mathbf {W} ^ {(I)} \operatorname {d i a g} \left(\sigma^ {\prime} (\mathbf {z} ^ {(I - 1)})\right) \mathbf {B} ^ {(I - 1)} \quad I \in [ 2, L - 1 ] \tag {4}
$$

$$
\mathbf {F} ^ {(L, L - 1)} = \mathbf {W} ^ {(L)} \quad \mathbf {F} ^ {(L, I)} = \mathbf {W} ^ {(L)} \operatorname {d i a g} \left(\sigma^ {\prime} (\mathbf {z} ^ {(L - 1)})\right) \mathbf {F} ^ {(L - 1, I)} \quad I \in [ L - 2 ] \tag {5}
$$

This leads to a fast back-propagation like method that can be used to compute the Hessian. Note that Lemma 1 only assumes a matrix multiplication operation from  $\mathbf{a}^{(I - 1)}$  to  $\mathbf{z}^{(I)}$ . Since a convolution operation can also be expressed as a matrix multiplication, we can directly extend this lemma to deep convolutional networks. Furthermore, Lemma 1 can also be of independent interest in other related problems such as higher-order interpretation methods for deep learning (e.g. Singla et al. (2019)).

# 4.2 CURVATURE BOUNDS FOR TWO LAYER NETWORKS

For a two-layer network and using Lemma 1,  $\nabla_{\mathbf{x}}^{2}\left(\mathbf{z}_{y}^{(2)} - \mathbf{z}_{t}^{(2)}\right)$  is given by:

$$
\nabla_ {\mathbf {x}} ^ {2} \left(\mathbf {z} _ {y} ^ {(2)} - \mathbf {z} _ {t} ^ {(2)}\right) = \left(\mathbf {W} ^ {(1)}\right) ^ {T} d i a g \left(\left(\mathbf {W} _ {y} ^ {(2)} - \mathbf {W} _ {t} ^ {(2)}\right) \odot \sigma^ {\prime \prime} \left(\mathbf {z} ^ {(1)}\right)\right) \mathbf {W} ^ {(1)}
$$

Note that only  $\sigma''(\mathbf{z}^{(1)})$  depends on  $\mathbf{x}$ . We can maximize and minimize each element in the diag term,  $(\mathbf{W}_{y,i}^{(2)} - \mathbf{W}_{t,i}^{(2)})\sigma''(\mathbf{z}_i^{(1)})$  independently subject to the constraint that  $\sigma''(.)$  is bounded. Using this procedure, we construct matrices  $\mathbf{P}$  and  $\mathbf{N}$  that satisfy properties given in the following theorem:

Theorem 3. Given a two layer network whose activation function has bounded second derivative:

$$
h _ {L} \leq \sigma^ {\prime \prime} (x) \leq h _ {U} \quad \forall x \in \mathbb {R}
$$

(a) We have the following linear matrix inequalities (LMIs):

$$
\mathbf {N} \preccurlyeq \nabla_ {\mathbf {x}} ^ {2} \left(\mathbf {z} _ {y} ^ {(2)} - \mathbf {z} _ {t} ^ {(2)}\right) \preccurlyeq \mathbf {P} \quad \forall \mathbf {x} \in \mathbb {R} ^ {D}
$$

(b) If  $h_U \geq 0$  and  $h_L \leq 0$ ,  $\mathbf{P}$  is a PSD matrix,  $\mathbf{N}$  is a NSD matrix.  
(c) This gives the following global bounds on the eigenvalues of the Hessian:

$$
m \mathbf {I} \preccurlyeq \nabla_ {\mathbf {x}} ^ {2} \left(\mathbf {z} _ {y} ^ {(2)} - \mathbf {z} _ {t} ^ {(2)}\right) \preccurlyeq M \mathbf {I}, \qquad w h e r e M = \lambda_ {m a x} (\mathbf {P}), m = \lambda_ {m i n} (\mathbf {N})
$$

$\mathbf{P}$  and  $\mathbf{N}$  are independent of  $\mathbf{x}$  and defined in equations (55) and (56) in Appendix D.4.

The proof is presented in Appendix D.4. Because power iteration finds the eigenvalue with largest magnitude, we can use it to find  $m$  and  $M$  only when  $\mathbf{P}$  is PSD and  $\mathbf{N}$  is NSD. We solve for  $h_U$  and  $h_L$  for sigmoid, tanh, softplus activation functions in Appendix E and show that this is in fact the case for them. Note that this result does not hold for ReLU networks since the ReLU function is not differentiable. However, in Appendix  $F$ , we devise a method to compute the certificate for a two layer ReLU network by finding a quadratic lower bound for  $\mathbf{z}_y^{(2)} - \mathbf{z}_t^{(2)}$ .

# 4.3 CURVATURE BOUNDS FOR DEEP NETWORKS

Using Lemma 1, we know that  $\nabla_{\mathbf{x}}^{2}\mathbf{z}_{i}^{(L)}$  is a sum product of matrices  $\mathbf{B}^{(I)}$  and  $\mathbf{F}_i^{(L,I)}$ . Thus, if we can find upper bounds for  $\| \mathbf{B}^{(I)}\|$  and  $\| \mathbf{F}_i^{(L,I)}\|$ , we can get upper bounds for  $\| \nabla_{\mathbf{x}}^{2}\mathbf{z}_{i}^{(L)}\|$ . Using this intuition (details are presented in Appendix D.5), we have the following result:

Theorem 4. Given an  $L$  layer neural network whose activation function satisfies:

$$
\left| \sigma^ {\prime} (x) \right| \leq g, \left| \sigma^ {\prime \prime} (x) \right| \leq h \quad \forall x \in \mathbb {R},
$$

the absolute value of eigenvalues of  $\nabla_{\mathbf{x}}^{2}\mathbf{z}_{i}^{(L)}$  is globally bounded by the following quantity:

$$
\left\| \nabla_ {\mathbf {x}} ^ {2} \mathbf {z} _ {i} ^ {(L)} \right\| \leq h \sum_ {I = 1} ^ {L - 1} \left(r ^ {(I)}\right) ^ {2} \max _ {j} \left(\mathbf {S} _ {i, j} ^ {(L, I)}\right), \quad \forall \mathbf {x} \in \mathbb {R} ^ {D}
$$

where  $r^{(I)}$  and  $\mathbf{S}^{(L,I)}$  are independent of  $\mathbf{x}$  and defined recursively as:

$$
r ^ {(1)} = \left\| \mathbf {W} ^ {(1)} \right\|, \quad r ^ {(I)} = g \left\| \mathbf {W} ^ {(I)} \right\| r ^ {(I - 1)} \quad I \in [ 2, L - 1 ] \tag {6}
$$

$$
\mathbf {S} ^ {(L, L - 1)} = \left| \mathbf {W} ^ {(L)} \right|, \quad \mathbf {S} ^ {(L, I)} = g \left| \mathbf {W} ^ {(L)} \right| \mathbf {S} ^ {(L - 1, I)} \quad I \in [ L - 2 ] \tag {7}
$$

The above expressions allow for an efficient computation of  $\mathbf{S}^{(L,I)}$  and  $r^{(I)}$ , thus curvature bounds for deep neural networks. The proof of this result is given in Appendix D.5. We consider simplification of this result for sigmoid, tanh, softplus activations in Appendix E.

Note that bounds for  $\mathbf{z}_y^{(L)} - \mathbf{z}_t^{(L)}$  can be computed by replacing  $\mathbf{W}_i^{(L)}$  with  $\mathbf{W}_y^{(L)} - \mathbf{W}_t^{(L)}$  in Theorem 4. The resulting bound is independent of  $\mathbf{x}$ , and only depends on network weights  $\mathbf{W}^{(I)}$ , the true label  $y$ , and the target  $t$ . We denote it with  $K(\mathbf{W}, y, t)$ . To simplify notation, when no confusion arises we denote it with  $K$ . In our experiments, for two layer networks, we use  $M$ ,  $m$  from Theorem 3 (since it provides tighter curvature bounds). For deeper networks ( $L \geq 3$ ), we use  $M = K$ ,  $m = -K$ .

# 5 ADVERSARIAL TRAINING WITH CURVATURE REGULARIZATION

Using Theorem 2 (b), we know that if we solve the curvature-based attack optimization and obtain  $\rho = \| \mathbf{x}^{(attack)} - \mathbf{x}^{(0)}\|$ ,  $\mathbf{x}^{(attack)}$  is provably the closest adversarial example to  $\mathbf{x}^{(0)}$ . However, when we performed adversarial training (with  $\rho = 0.5$ ), we found that the curvature bound is loose and almost none of training inputs lead to zero primal-dual gap with  $\rho = \| \mathbf{x}^{(attack)} - \mathbf{x}^{(0)}\|$ . To fix this issue, we use a regularizer that penalizes the curvature bound,  $K$ . Using equations (6) and (7), we can compute  $K$  using absolute value, matrix multiplications, and operator norm  $\left(\| \mathbf{W}^{(I)}\|, I \in [L]\right)$ . Since the gradient of operator norm does not exist in standard libraries, we created a new layer where the gradient of  $\| \mathbf{W}^{(I)}\|$ , i.e  $\nabla_{\mathbf{W}^{(I)}}\| \mathbf{W}^{(I)}\|$  is given by:

$$
\nabla_ {\mathbf {W} ^ {(I)}} \left\| \mathbf {W} ^ {(I)} \right\| = \mathbf {u} ^ {(I)} \left(\mathbf {v} ^ {(I)}\right) ^ {T} \quad \mathbf {u} ^ {(I)}, \mathbf {v} ^ {(I)} \text {s a t i s f y} \mathbf {W} ^ {(I)} \mathbf {v} ^ {(I)} = \left\| \mathbf {W} ^ {(I)} \right\| \mathbf {u} ^ {(I)}
$$

![](images/e6c15b9cc3cb952da53cffd6addf87de5fcc4f43c27e304f7980bb6abff8012a.jpg)  
Figure 1:  $K_{ub}$  and  $K_{lb}$  are upper and lower curvature bounds of the network with Sigmoid activations (averaged over  $(y,t)$  pairs). When  $\gamma = 0$  (no curvature regularization), networks adversarially trained with CRT or PGD both have high curvatures. However, CRT even with a small  $\gamma$  leads to a significant decrease in curvature bounds (note the log-scale of  $y$ -axis). Similar results hold for networks with Tanh activations (Appendix Figure 2)

This approach to compute the gradient of the largest singular value of a matrix has also been used in previous ICLR work (Miyato et al., 2018). Implementation details are in Appendix C.1. Thus, the per-sample loss for training with curvature regularization is:

$$
c r o s s \_ e n t r o p y (\mathbf {z} ^ {(L)} (\mathbf {x} ^ {(0)}), y) + \gamma K (\mathbf {W}, y, t)
$$

where  $y$  is the true label of the input  $\mathbf{x}^{(0)}$ ,  $t$  is the target label and  $\gamma$  is the regularizer for penalizing large curvatures. Similar to the adversarial training, in CRT, we use  $\mathbf{x}^{(attack)}$  instead of  $\mathbf{x}^{(0)}$ .

# 6 EXPERIMENTS

The empirical robust accuracy means the fraction of test samples that were correctly classified after running an  $l_{2}$  bounded PGD attack (Madry et al., 2018), the certified robust accuracy means the fraction of correctly classified test samples whose robustness certificates are greater than a prespecified radius  $\rho$ . Unless otherwise specified, we use the class with the second largest logit as the attack target (i.e. the class  $t$ ) and  $\rho = 0.5$ . All experiments were run on the MNIST dataset. The notation  $(L \times [1024]$ , activation) denotes a neural network with  $L$  layers with the specified activation,  $(\gamma = c)$  denotes standard training with  $\gamma$  set to  $c$ , while (CRT,  $c$ ) denotes CRT training with  $\gamma = c$ . Certificates are computed over 150 randomly chosen correctly classified images.

Comparison with existing certificates: In Table 2, we compare CRC with CROWN-general (Zhang et al., 2018a). For 3 and 4 layer networks, we observe that CRC is an order of magnitude faster to compute. For 2-layer networks, CRC outperforms CROWN significantly. For deeper networks, CRC works better only when the network is trained with curvature regularization. However, even with small  $\gamma = 0.005$ , we see a significant increase in CRC but a very small drop in the test accuracy (without any adversarial training). We can see that with  $\gamma = 0.01$ , non-trivial certified accuracies of  $83.53\%$ ,  $88.33\%$ ,  $89.61\%$  can be achieved on 2, 3, 4 layer sigmoid networks, respectively, without any adversarial training. Adversarial training using CRT further boosts certified accuracy to  $95.59\%$ ,  $94.99\%$  and  $93.41\%$ , respectively.

In Figure 1, we plot the effect of  $\gamma$  on the curvature upper bound  $K_{ub}$  and a lower bound  $K_{lb}$  of a 4-layer network with Sigmoid activations.  $K_{lb}$  is computed by taking the maximum of the largest eigenvalue of the Hessian across all test images with label  $y$  and the second largest logit  $t$ , then averaging across different  $(y,t)$ . Similarly,  $K_{ub}$  is the mean of  $K$  over all pairs  $(y,t)$  (details in Appendix G.4). We observe that without any curvature regularization (when  $\gamma = 0$ ), both standard adversarial training with PGD as well as the CRT lead to networks with high curvatures. However, CRT with even a small  $\gamma$  leads to a significant decrease in curvature bounds. Similar trends can be observed for networks with Tanh activations (Appendix Figure 2). Curvature bounds are higher for the Tanh networks compared to the Sigmoid ones due to having larger  $g$  and  $h$  parameters for Tanh in Theorem 4. Moreover, we report curvature bounds for networks with different depth in Appendix Table 7. We observe that increasing depth increases curvature bounds.

Comparison with existing adversarial training methods: We compare CRT with adversarial training methods namely PGD (Madry et al., 2018) and TRADES (Zhang et al., 2019) in Table 3. We observe that none of the other methods give higher certified accuracy or robustness certificates than our proposed methods. We observe similar results with Tanh networks (Appendix Table 4).

Moreover, in Appendix Table 5, we observe that CRT outperforms Randomized Smoothing (Cohen et al., 2019a) for 2 and 3 layer networks. Since TRADES and Randomized Smoothing were designed for untargeted attacks while CRT is for targeted attacks, to have a fair comparison, we modify the multi-class version of the cross entropy loss with its binary version (details in Appendix Section G.2).

<table><tr><td rowspan="2">Network</td><td rowspan="2">Training</td><td rowspan="2">Standard Accuracy</td><td rowspan="2">Certified Robust Accuracy</td><td colspan="2">Certificate (mean)</td><td colspan="2">Time per image (seconds)</td></tr><tr><td>CROWN</td><td>CRC</td><td>CROWN</td><td>CRC</td></tr><tr><td rowspan="4">2×[1024], sigmoid</td><td>standard</td><td>98.37%</td><td>54.17%</td><td>0.28395</td><td>0.48500</td><td>0.1818</td><td>0.1911</td></tr><tr><td>γ = 0.005</td><td>97.96%</td><td>82.68%</td><td>0.36125</td><td>0.83367</td><td>0.1599</td><td>0.2229</td></tr><tr><td>γ = 0.01</td><td>98.08%</td><td>83.53%</td><td>0.32548</td><td>0.84719</td><td>0.1732</td><td>0.2186</td></tr><tr><td>CRT, 0.01</td><td>98.57%</td><td>95.59%</td><td>0.43061</td><td>1.54673</td><td>0.1823</td><td>0.1910</td></tr><tr><td rowspan="4">3×[1024], sigmoid</td><td>standard</td><td>98.37%</td><td>0.00%</td><td>0.24644</td><td>0.06874</td><td>1.6356</td><td>0.5012</td></tr><tr><td>γ = 0.005</td><td>97.98%</td><td>88.66%</td><td>0.38030</td><td>0.99044</td><td>1.6220</td><td>0.5319</td></tr><tr><td>γ = 0.01</td><td>97.71%</td><td>88.33%</td><td>0.39799</td><td>1.07842</td><td>1.6342</td><td>0.5295</td></tr><tr><td>CRT, 0.01</td><td>97.23%</td><td>94.99%</td><td>0.39603</td><td>1.24100</td><td>1.5625</td><td>0.5013</td></tr><tr><td rowspan="4">4×[1024], sigmoid</td><td>standard</td><td>98.39%</td><td>0.00%</td><td>0.19501</td><td>0.00454</td><td>4.7814</td><td>0.8107</td></tr><tr><td>γ = 0.005</td><td>97.74%</td><td>88.95%</td><td>0.36863</td><td>0.91840</td><td>5.1667</td><td>0.8567</td></tr><tr><td>γ = 0.01</td><td>97.41%</td><td>89.61%</td><td>0.40620</td><td>1.05323</td><td>4.6296</td><td>0.8328</td></tr><tr><td>CRT, 0.01</td><td>97.83%</td><td>93.41%</td><td>0.40327</td><td>1.06208</td><td>4.1830</td><td>0.8088</td></tr></table>

Table 2: Comparison between CROWN-general (Zhang et al., 2018a) and CRC.  

<table><tr><td rowspan="2">Network</td><td rowspan="2">Training</td><td rowspan="2">Standard Accuracy</td><td rowspan="2">Empirical Robust Accuracy</td><td rowspan="2">Certified Robust Accuracy</td><td colspan="2">Certificate (mean)</td></tr><tr><td>CROWN</td><td>CRC</td></tr><tr><td rowspan="3">2×[1024], sigmoid</td><td>PGD</td><td>98.80%</td><td>96.26%</td><td>93.37%</td><td>0.37595</td><td>0.82702</td></tr><tr><td>TRADES</td><td>98.87%</td><td>96.76%</td><td>95.13%</td><td>0.41358</td><td>0.92300</td></tr><tr><td>CRT, 0.01</td><td>98.57%</td><td>96.28%</td><td>95.59%</td><td>0.43061</td><td>1.54673</td></tr><tr><td rowspan="3">3×[1024], sigmoid</td><td>PGD</td><td>98.84%</td><td>96.14%</td><td>0.00%</td><td>0.29632</td><td>0.07290</td></tr><tr><td>TRADES</td><td>98.95%</td><td>96.79%</td><td>0.00%</td><td>0.30576</td><td>0.09108</td></tr><tr><td>CRT, 0.01</td><td>98.23%</td><td>95.70%</td><td>94.99%</td><td>0.39603</td><td>1.24100</td></tr><tr><td rowspan="3">4×[1024], sigmoid</td><td>PGD</td><td>98.84%</td><td>96.26%</td><td>0.00%</td><td>0.25444</td><td>0.00658</td></tr><tr><td>TRADES</td><td>98.76%</td><td>96.67%</td><td>0.00%</td><td>0.26128</td><td>0.00625</td></tr><tr><td>CRT, 0.01</td><td>97.83%</td><td>94.65%</td><td>93.41%</td><td>0.40327</td><td>1.06208</td></tr></table>

Table 3: Comparison between CRT, PGD (Madry et al., 2018) and TRADES (Zhang et al., 2019).

# 7 CONCLUSION

In this paper, we develop computationally-efficient convex relaxations for robustness certification and adversarial attack problems given the classifier has a bounded curvature. We also show that this convex relaxation is tight under some general conditions. To be able to use proposed certification and attack convex optimizations, we derive global curvature bounds for deep networks with differentiable activation functions. This result is a consequence of a closed-form expression that we derived for the Hessian of a deep network. Our empirical results indicate that our proposed curvature-based robustness certificate outperforms the CROWN certificate by an order of magnitude while being faster to compute as well. Furthermore, adversarial training using our attack method coupled with curvature regularization results in a significantly higher certified robust accuracy than the existing adversarial training methods. Scaling up our proposed curvature-based robustness certification and training methods as well as further tightening the derived curvature bounds are among interesting directions for the future work. In particular, one can extend our proposed methods to deep convolutional networks using the spectral bounds for convolution layers derived in Sedghi et al. (2018).

# REFERENCES

Cem Anil, James Lucas, and Roger B. Grosse. Sorting out lipschitz function approximation. In ICML, 2018.  
Anish Athalye and Nicholas Carlini. On the robustness of the cvpr 2018 white-box adversarial example defenses. ArXiv, abs/1804.03286, 2018.  
Anish Athalye, Nicholas Carlini, and David A. Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In ICML, 2018.  
Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, New York, NY, USA, 2004. ISBN 0521833787.  
Rudy Bunel, Ilker Turkaslan, Philip H. S. Torr, Pushmeet Kohli, and Pawan Kumar Mudigonda. A unified view of piecewise linear neural network verification. In NeurIPS, 2017.  
Xiaoyu Cao and Neil Zhenqiang Gong. Mitigating evasion attacks to deep neural networks via region-based classification. ArXiv, abs/1709.05583, 2017.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, AISec '17, pp. 3-14, New York, NY, USA, 2017. ACM. ISBN 978-1-4503-5202-4. doi: 10.1145/3128572.3140444. URL http://doi.acm.org/10.1145/3128572.3140444.  
Nicholas Carlini, Guy Katz, Clark E. Barrett, and David L. Dill. Provably minimally-distorted adversarial examples. 2017.  
Chih-Hong Cheng, Georg Nuhrenberg, and Harald Ruess. Maximum resilience of artificial neural networks. In ATVA, 2017.  
Jeremy M. Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified adversarial robustness via randomized smoothing. In ICML, 2019a.  
Jeremy M. Cohen, Elan Rosenfeld, and J. Zico Kolter. Certified adversarial robustness via randomized smoothing. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pp. 1310-1320, 2019b. URL http://proceedings.mlr.press/v97/cohen19c.html.  
Francesco Croce, Maksym Andriushchenko, and Matthias Hein. Provable robustness of relu networks via maximization of linear regions. ArXiv, abs/1810.07481, 2018.  
Souradeep Dutta, Susmit Jha, Sriram Sankaranarayanan, and Ashish Tiwari. Output range analysis for deep feedforward neural networks. In NFM, 2018.  
Krishnamurthy Dvijotham, Sven Gowal, Robert Stanforth, Relja Arandjelovic, Brendan O'Donoghue, Jonathan Uesato, and Pushmeet Kohli. Training verified learners with learned verifiers. *ArXiv*, abs/1805.10265, 2018a.  
Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy A. Mann, and Pushmeet Kohli. A dual approach to scalable verification of deep networks. In UAI, 2018b.  
Rüdiger Ehlers. Formal verification of piece-wise linear feed-forward neural networks. *ArXiv*, abs/1705.01320, 2017.  
Matteo Fischetti and Jason Jo. Deep neural networks and mixed integer linear optimization. Constraints, 23:296-309, 2018.  
Timon Gehr, Matthew Mirman, Dana Drachsler-Cohen, Petar Tsankov, Swarat Chaudhuri, and Martin T. Vechev. Ai2: Safety and robustness certification of neural networks with abstract interpretation. 2018 IEEE Symposium on Security and Privacy (SP), pp. 3-18, 2018.  
Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin, Jonathan Uesato, Relja Arandjelovic, Timothy A. Mann, and Pushmeet Kohli. On the effectiveness of interval bound propagation for training verifiably robust models. *ArXiv*, abs/1810.12715, 2018.

Matthias Hein and Maksym Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems 30, pp. 2266-2276. 2017.  
Xiaowei Huang, Marta Z. Kwiatkowska, Sen Wang, and Min Wu. Safety verification of deep neural networks. ArXiv, abs/1610.06940, 2016.  
Guy Katz, Clark W. Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. Reluplex: An efficient smt solver for verifying deep neural networks. In CAV, 2017.  
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial examples in the physical world. ArXiv, abs/1607.02533, 2016a.  
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio. Adversarial machine learning at scale. ArXiv, abs/1611.01236, 2016b.  
Mathias Lécuyer, Vaggelis Atlidakis, Roxana Geambasu, Daniel Hsu, and S. K. K. Jana. Certified robustness to adversarial examples with differential privacy. In IEEE S&P 2019, 2018.  
Bai Han Li, Changyou Chen, Wenlin Wang, and Lawrence Carin. Certified adversarial robustness with additive gaussian noise. 2018.  
Xuanqing Liu, Minhao Cheng, Huan Zhang, and Cho-Jui Hsieh. Towards robust neural networks via random self-ensemble. ArXiv, abs/1712.00673, 2017.  
Alessio Lomuscio and Lalit Maganti. An approach to reachability analysis for feed-forward relu neural networks. ArXiv, abs/1706.07351, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. ArXiv, abs/1706.06083, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJzIBfZAb.  
Matthew Mirman, Timon Gehr, and Martin T. Vechev. Differentiable abstract interpretation for provably robust neural networks. In ICML, 2018.  
Takeru Miyato, Shin ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: A regularization method for supervised and semi-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41:1979-1993, 2017.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=B1QRgziT-.  
Seyed Mohsen Moosavi Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal Frossard. Robustness via curvature regularization, and vice versa. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.  
N. Papernot, P. McDaniel, X. Wu, S. Jha, and A. Swami. Distillation as a defense to adversarial perturbations against deep neural networks. In 2016 IEEE Symposium on Security and Privacy (SP), pp. 582-597, May 2016. doi: 10.1109/SP.2016.41.  
Nicolas Papernot, Patrick D. McDaniel, Ian J. Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. Practical black-box attacks against deep learning systems using adversarial examples. CoRR, abs/1602.02697, 2016. URL http://arxiv.org/abs/1602.02697.  
Jonathan Peck, Joris Roels, Bart Goossens, and Yvan Saeys. Lower bounds on the robustness to adversarial perturbations. In NIPS, 2017.  
Chongli Qin, James Martens, Sven Gowal, Dilip Krishnan, Alhussein Fawzi, Soham De, Robert Stanforth, and Pushmeet Kohli. Adversarial robustness through local linearization. arXiv preprint arXiv:1907.02610, 2019.

Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. *ArXiv*, abs/1801.09344, 2018a.  
Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Semidefinite relaxations for certifying robustness to adversarial examples. In NeurIPS, 2018b.  
Hadi Salman, Greg Yang, Jerry Li, Pengchuan Zhang, Huan Zhang, Ilya P. Razenshteyn, and Sebastien Bubeck. Provably robust deep learning via adversarially trained smoothed classifiers. ArXiv, abs/1906.04584, 2019.  
Pouya Samangouei, Maya Kabbab, and Rama Chellappa. Defense-GAN: Protecting classifiers against adversarial attacks using generative models. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=BkJ3ibb0-.  
Hanie Sedghi, Vineet Gupta, and Philip M Long. The singular values of convolutional layers. arXiv preprint arXiv:1805.10408, 2018.  
Gagandeep Singh, Timon Gehr, Matthew Mirman, Markus Puschel, and Martin T. Vechev. Fast and effective robustness certification. In NeurIPS, 2018.  
Sahil Singla, Eric Wallace, Shi Feng, and Soheil Feizi. Understanding impacts of high-order loss approximations and features in deep learning interpretation. In ICML, 2019.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. URL http://arxiv.org/abs/1312.6199.  
Jonathan Uesato, Brendan O'Donoghue, Pushmeet Kohli, and Aäron van den Oord. Adversarial risk and the dangers of evaluating against weak attacks. In ICML, 2018.  
Shiqi Wang, Yizheng Chen, Ahmed Abdou, and S. K. K. Jana. Mixtrain: Scalable training of verifiably robust neural networks. 2018a.  
Shiqi Wang, Kexin Pei, Justin Whitehouse, Junfeng Yang, and S. K. K. Jana. Efficient formal safety analysis of neural networks. In NeurIPS, 2018b.  
Tsui-Wei Weng, Huan Zhang, Hongge Chen, Zhao Song, Cho-Jui Hsieh, Duane S. Boning, Inderjit S. Dhillon, and Luca Daniel. Towards fast computation of certified robustness for relu networks. ArXiv, abs/1804.09699, 2018.  
Eric Wong and J. Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. *ArXiv*, abs/1711.00851, 2017.  
Eric Wong, Frank R. Schmidt, Jan Hendrik Metzen, and J. Zico Kolter. Scaling provable adversarial defenses. In NeurIPS, 2018.  
Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled trade-off between robustness and accuracy. In ICML, 2019.  
Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems 31, pp. 4939-4948. Curran Associates, Inc., 2018a.  
Huan Zhang, Tsui-Wei Weng, Pin-Yu Chen, Cho-Jui Hsieh, and Luca Daniel. Efficient neural network robustness certification with general activation functions. ArXiv, abs/1811.00866, 2018b.  
Stephan Zheng, Yang Song, Thomas Leung, and Ian J. Goodfellow. Improving the robustness of deep neural networks via stability training. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4480-4488, 2016.
