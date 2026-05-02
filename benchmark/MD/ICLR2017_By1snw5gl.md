# L-SR1: A SECOND ORDER OPTIMIZATION METHOD FOR DEEP LEARNING

Vivek Ramamurthy

Sentient Technologies

1 California Street Suite 2300

San Francisco, CA 94111

vivek.ramamurthy@sentient.ai

Nigel Duffy

Sentient Technologies

1 California Street Suite 2300

San Francisco, CA 94111

nigel.duffy@sentient.ai

# ABSTRACT

We describe L-SR1, a new second order method to train deep neural networks. Second order methods hold great promise for distributed training of deep networks. Unfortunately, they have not proven practical. Two significant barriers to their success are inappropriate handling of saddle points, and poor conditioning of the Hessian. L-SR1 is a practical second order method that addresses these concerns. We provide experimental results showing that L-SR1 performs at least as well as Nesterov's Accelerated Gradient Descent, on the MNIST and CIFAR10 datasets. For the CIFAR10 dataset, we see competitive performance on shallow networks like LeNet5, as well as on deeper networks like residual networks. Furthermore, we perform an experimental analysis of L-SR1 with respect to its hyperparameters to gain greater intuition. Finally, we outline the potential usefulness of L-SR1 in distributed training of deep neural networks.

# 1 MOTIVATION

Second order methods hold great potential for distributing the training of deep neural networks. Due to their use of curvature information, they can often find good minima in far fewer steps than first order methods such as stochastic gradient descent (SGD). However, stochastic second order methods typically require larger mini-batches (Le et al., 2011). This is because they estimate second derivatives via differences between estimated gradients. The gradient estimates need to have less variance, so that when we take their differences, the result has low variance. As a result they provide a different trade-off between number of steps and mini-batch size than do SGD-like methods. This trade-off is interesting, because while steps must be evaluated sequentially, a mini-batch may be evaluated in parallel. Thus, second order methods present an opportunity to extract more parallelism in neural network training. In particular, when mini-batches are sufficiently large, their evaluation may be distributed. Furthermore, there are relatively fewer hyperparameters to tune in second order methods, compared to variants of stochastic gradient descent.

L-BFGS (Nocedal, 1980; Liu & Nocedal, 1989) is perhaps the most commonly used second order method in machine learning. BFGS is a quasi-Newton method that maintains an approximation to the inverse Hessian of the function being optimized. L-BFGS is a limited memory version of BFGS that removes the requirement to explicitly store the inverse Hessian and can therefore be used practically for large scale problems. L-BFGS is typically combined with a line search technique to choose an appropriate step size at each iteration. L-BFGS has been used to good effect in convex optimization problems in machine learning, but has not found effective use in large scale non-convex problems such as deep learning.

Three critical weaknesses have been identified. First, we know that training deep neural networks involves minimizing non-convex error functions over continuous, high dimensional spaces. It has been argued that the proliferation of saddle points in these problems presents a deep and profound difficulty for quasi-Newton optimization methods (Dauphin et al., 2014). Furthermore, it has been argued that curvature matrices generated in second order methods are often ill-conditioned, and these need to be carefully repaired. A variety of approaches to this have been suggested, including the use of an empirical Fisher diagonal matrix (Martens, 2016). Finally, popular quasi-Newton approaches,

such as L-BFGS, require line search to make parameter updates, which requires many more gradient and/or function evaluations.

It is worth noting that several approaches have been proposed to overcome the weaknesses of L-BFGS. First, it has been proposed to initialize L-BFGS with a number of SGD steps. However, this diminishes the potential for parallelism (Dean et al., 2012; Le et al., 2011). Second, it has been proposed to use "forgetting", where every few (say, for example, 5) steps, the history for L-BFGS is discarded. However, this greatly reduces the ability to use second order curvature information. Despite these proposals, L-BFGS is not commonly used for deep learning.

We propose L-SR1, a second order method that addresses each of these concerns.

# 2 BACKGROUND

In the following, we provide a brief primer on line search and trust region methods, as well as on quasi-Newton methods and their limited memory variants. Further details may be found in Nocedal & Wright (2006).

# 2.1 LINE SEARCH AND TRUST REGION METHODS

In any optimization algorithm, there are two main ways of moving from the current point  $x_{k}$  to a new iterate  $x_{k + 1}$ . One of them is line search. In it, the algorithm picks a descent direction  $p_k$  and searches along this direction from the current iterate  $x_{k}$  for a new iterate with a lower function value. The distance  $\alpha$  to move along  $p_k$  can be found by solving the following one-dimensional minimization problem:

$$
\min  _ {\alpha > 0} f \left(x _ {k} + \alpha p _ {k}\right)
$$

Instead of an exact minimization which may be expensive, the line search algorithm generates a limited number of trial step lengths until it finds one that generates a sufficient decrease in function value. At the new point, the process of computing the descent direction and step length is repeated. The other way is to use a trust region method. In a trust region method, the information about  $f$  is used to construct a model function  $m_{k}$ , which is supposed to approximate  $f$  near the current point  $x_{k}$ . Since the model  $m_{k}$  may not approximate  $f$  well when  $x$  is far from  $x_{k}$ , the search for a minimizer of  $m_{k}$  is restricted to some trust region within a radius  $\Delta_{k}$  around  $x_{k}$ . To wit, the candidate step  $p$  approximately solves the following sub-problem:

$$
\min_{p:||p||\leq \Delta_{k}}m_{k}(x_{k} + p),
$$

If the candidate solution does not produce a sufficient decrease in  $f$ , the trust region is considered too large for the model function to approximate  $f$  well. So we shrink the trust region and re-solve. Essentially, the line search and trust region approaches differ in the order in which they choose the direction and magnitude of the move to the next iterate. In line search, the descent direction  $p_k$  is fixed first, and then the step length  $\alpha_k$  to be taken along that direction is computed. In trust region, a maximum distance equal to the trust-region radius  $\Delta_k$  is first set, and then a direction is determined within this radius, that achieves the best improvement in the objective value. If such a direction does not yield sufficient improvement, the model function is determined to be a poor approximation to the function, and the trust-region radius  $\Delta_k$  is reduced until the approximation is deemed good enough. Conversely, as long as the model function appears to approximate the objective function well, the trust region radius is increased until the approximation is not good enough.

# 2.2 LIMITED MEMORY QUASI-NEWTON METHODS

Quasi-Newton methods are a useful alternative to Newton's method in that they do not require computation of the exact Hessian, and yet still attain good convergence. In place of the true Hessian  $\nabla^2 f_k$ , they use an approximation  $B_{k}$ , which is updated after each step based on information gained during the step. At each step, the new Hessian approximation  $B_{k + 1}$  is required to satisfy the following condition, known as the secant equation:

$$
B _ {k + 1} s _ {k} = y _ {k}
$$

where

$$
s _ {k} = x _ {k + 1} - x _ {k}, y _ {k} = \nabla f _ {k + 1} - \nabla f _ {k}
$$

Typically,  $B_{k + 1}$ , is also required to be symmetric (like the exact Hessian), and the difference between successive approximations  $B_{k}$  and  $B_{k + 1}$  is constrained to have low rank. One of the most popular formulae for updating the Hessian approximation  $B_{k}$  is the BFGS formula, named after its inventors, Broyden, Fletcher, Goldfarb, and Shanno, which is defined by

$$
B _ {k + 1} = B _ {k} - \frac {B _ {k} s _ {k} s _ {k} ^ {T} B _ {k}}{s _ {k} ^ {T} B _ {k} s _ {k}} + \frac {y _ {k} y _ {k} ^ {T}}{y _ {k} ^ {T} s _ {k}}
$$

A less well known formula, particularly in the machine learning community, is the symmetric-rank-one (SR1) formula, defined by

$$
B _ {k + 1} = B _ {k} + \frac {(y _ {k} - B _ {k} s _ {k}) (y _ {k} - B _ {k} s _ {k}) ^ {T}}{(y _ {k} - B _ {k} s _ {k}) ^ {T} s _ {k}}
$$

The former update is a rank-two update, while the latter is a rank-one update. Both updates satisfy the secant equation and maintain symmetry. The BFGS update always generates positive definite approximations whenever the initial approximation  $B_0$  is positive definite and  $s_k^T y_k > 0$ . Often, in practical implementations of quasi-Newton methods, the inverse Hessian approximation  $H_{k}$  is used instead of the  $B_{k}$ , and the corresponding update formulae can be generated using the Sherman-Morrison-Woodbury matrix identity (Hager, 1989).

Limited-memory quasi-Newton methods are useful for solving large problems where computation of Hessian matrices is costly or when these matrices are dense. Instead of storing fully dense  $n \times n$  approximations, these methods save only a few vectors of length  $n$  that capture the approximations. Despite these modest storage requirements, they often converge well. The most popular limited memory quasi-Newton method is L-BFGS, which uses curvature information from only the most recent iterations to construct the inverse Hessian approximation. Curvature information from earlier iterations, which is less likely to be useful to modeling the actual behavior of the Hessian at the current iteration, is discarded in order to save memory.

Limited-memory quasi-Newton approximations can be used with line search or trust region methods. As described in Byrd et al. (1994), we can derive efficient limited memory implementations of several quasi-Newton update formulae, and their inverses.

# 3 ALGORITHM

Unlike BFGS, the SR1 update does not guarantee positive definiteness of the updated matrix. This was considered a major problem in the early days of nonlinear optimization when only line search iterations were used, and possibly led to its obscurity outside the optimization community. However, with the development of trust-region methods, the SR1 updating formula is potentially very useful, and its ability to generate indefinite Hessian approximations can actually prove to be advantageous. The main limitation of SR1 updating is that the denominator in the update can vanish. Typically though, it has been observed in practice that SR1 performs well by skipping the update if the denominator is small, which does not occur very often anyway. For a comprehensive treatment of quasi-Newton methods see Dennis Jr. & More (1977).

We believe that it is possible to overcome saddle points using rank-one update based second order methods. The more common rank-two methods, e.g. L-BFGS, maintain a positive definite approximation to the inverse of the Hessian, by design (Nocedal & Wright, 2006). At saddle-points, the true Hessian cannot be well approximated by a positive definite matrix, causing commonly used second order methods to go uphill (Dauphin et al., 2014). On the other hand, rank-one approaches such as SR1 (Symmetric Rank One) (Nocedal & Wright, 2006) don't maintain this invariant, and so they can go downhill at saddle points. Numerical experiments (Conn et al., 1991) suggest that the approximate Hessian matrices generated by the SR1 method show faster progress towards the true Hessian than those generated by BFGS. This suggests that a limited memory SR1 method (L-SR1, if you like) would potentially outperform L-BFGS in the task of high dimensional optimization in neural network training. The building blocks needed to construct an L-SR1 method have been suggested in the literature (Byrd et al., 1994; Khalfan et al., 1993). To the best of our knowledge, however, there

is no complete L-SR1 method previously described in the literature  $^{1}$ . This prompted us to develop and test the approach, specifically in the large scale non-convex problems that arise in deep learning.

Two other insights make L-SR1 practical by removing the requirement for a line search and addressing the conditioning problem. First, we replace the line search using a trust region approach. While L-BFGS using line search is well studied, recently, an L-BFGS method that uses a trust-region framework has also been proposed (Burke et al., 2008). Second, we combine L-SR1 with batch normalization. Batch normalization is a technique of normalizing inputs to layers of a neural network, used to address a phenomenon known as internal covariate shift during training (Ioffe & Szegedy, 2015). Our hypothesis is that batch normalization may cause parameters of a neural network to be suitably scaled so that the Hessian becomes better conditioned. We tested this hypothesis empirically and outline the results below.

Our algorithm is synthesized as follows. We take the basic SR1 algorithm described in Nocedal & Wright (2006) (Algorithm 6.2), and represent the relevant input matrices using the limited-memory representations described in Byrd et al. (1994). The particular limited-memory representations used in the algorithm vary, depending on whether we use trust region or line search methods as subroutines to make parameter updates, as does some of the internal logic. For instance, if  $k$  updates are made to the symmetric matrix  $B_{0}$  using the vector pairs  $\{s_i,y_i\}_{i = 0}^{k - 1}$  and the SR1 formula, the resulting matrix  $B_{k}$  can be expressed as (Nocedal & Wright, 2006)

$$
B _ {k} = B _ {0} + (Y _ {k} - B _ {0} S _ {k}) (D _ {k} + L _ {k} + L _ {k} ^ {T} - S _ {k} ^ {T} B _ {0} S _ {k}) ^ {- 1} (Y _ {k} - B _ {0} S _ {k}) ^ {T}
$$

where  $S_{k},Y_{k},D_{k}$  , and  $L_{k}$  are defined as follows:

$$
S _ {k} = [ s _ {o}, \dots , s _ {k - 1} ], a n d Y _ {k} = [ y _ {0}, \dots , y _ {k - 1} ]
$$

$$
(L _ {k}) _ {i, j} = \left\{ \begin{array}{l l} s _ {i - 1} ^ {T} y _ {j - 1} & \text {i f} i > j \\ 0 & \text {o t h e r w i s e} \end{array} \right.
$$

$$
D _ {k} = \mathrm {d i a g} [ s _ {0} ^ {T} y _ {0}, \dots , s _ {k - 1} ^ {T} y _ {k - 1} ]
$$

The self-duality of the SR1 method (Nocedal & Wright, 2006) allows the inverse formula  $H_{k}$  to be obtained simply by replacing  $B$ ,  $s$ , and  $y$  by  $H$ ,  $y$ , and  $s$ , respectively, using standard matrix identities. Limited-memory SR1 methods can be derived exactly like in the case of the BFGS method. Additional details are present in the pseudocode provided in the Appendix. The algorithm we develop is general enough to work with any line search or trust region method. While we tested the algorithm with line search approaches described in Dennis Jr. & Schnabel (1983), and with the trust region approach described in Brust et al. (2016), in this paper, we focus our experimental investigations on using the trust region approach, and the advantage that provides over using other first and second order optimization methods.

# 4 EXPERIMENTS

In the following, we summarize the results of training standard neural networks on the MNIST and CIFAR10 datasets using our approach, and benchmarking the performance with respect to other first and second order methods. First, we compared our L-SR1 (with trust region) approach, with Nesterov's Accelerated Gradient Descent (NAG), L-BFGS with forgetting every 5 steps, default SGD, AdaDelta, and SGD with momentum, by training small standard networks on the MNIST and CIFAR10 datasets. Next, we compared our L-SR1 with trust region approach with default hyperparameters, with SGD with momentum, by training a 20-layer deep residual network on the CIFAR10 dataset. Following that, we varied each hyperparameter of the L-SR1 with trust region approach to observe its effect on training the residual network on CIFAR10.

# 4.1 LENET-LIKE NETWORKS

For each approach, and for each dataset, we considered the case where our networks had batch normalization layers within them, and the case where they did not. The parameters of the networks were

randomly initialized. In all cases, the networks were trained for 20 epochs. Finally, all experiments were repeated 10 times to generate error bars. The plots below show the variation of test loss with number of epochs.

# 4.1.1 MNIST

We considered the LeNet5 architecture in this case, which comprised 2 convolutional layers, followed by a fully connected layer and an outer output layer. Each convolutional layer was followed by a max-pooling layer. In the case where we used batch-normalization, each convolutional and fully connected layer was followed by a spatial batch normalization layer. We used a mini-batch size of 20 for the first order methods like NAG, SGD, AdaDelta and SGD with momentum, and a mini-batch size of 400 for the second order methods like L-SR1 and L-BFGS. The memory size was set to 5 for both L-SR1 and L-BFGS. Further details on the network architecture and other parameter settings are provided in the Appendix.

![](images/9baf0f3d3e4495e6c52fb9f9b9cef7bbf9c83a3b17f10ff282bceed04acab2a0.jpg)  
Figure 1: Variation of test loss with number of epochs, on the MNIST dataset, with and without batch normalization. Note that the scales on the y-axes are different.

![](images/3275c2dd6dd9c9af2352fc5e756f9213339d8db5f3bb1599e508786b9647b47b.jpg)

# 4.1.2 CIFAR10

We considered a slight modification to the 'LeNet5' architecture described above. We used a minibatch size of 96 for NAG, SGD, AdaDelta and SGD with momentum. The other mini-batch sizes and memory sizes for L-SR1 and L-BFGS were as above. Further details on the network architecture and other parameter settings are provided in the Appendix.

# 4.1.3 DISCUSSION

Our experiments suggest the following:

- L-SR1 performs as well as, or slightly better than all the first order methods on both the MNIST and CIFAR10 datasets, with or without batch normalization.  
- L-SR1 is substantially better than L-BFGS in all settings, with or without forgetting.  
- Forgetting appears to be necessary in order to get L-BFGS to work. Without forgetting, the approach appears to be stuck where it is initialized. For this reason, the plots for L-BFGS without forgetting have not been included.  
- Batch normalization appears to improve the performance of all approaches, particularly the early performance of second order approaches like L-SR1 and L-BFGS.

![](images/f84ca44d3f33ac2fd0745253406a87788d2ed4d20436d4b66073f3a20e8e3d9d.jpg)  
Figure 2: Variation of test loss with number of epochs, on the CIFAR10 dataset, with and without batch normalization. Note that the scales on the y-axes are different.

![](images/414966f86b41b44a4ef4674894badb4a326211b4ba5994229f03184ca656ba24.jpg)

Note that in these experiments we compare performance between the algorithms per epoch. To reiterate, we use mini-batch sizes of 20 and 96 for the first order methods for the MNIST and CIFAR experiments respectively, while we use a mini-batch size of 400 for the second order methods in all cases. This means that the second order methods take fewer steps per epoch than the first order methods due to their larger mini-batch sizes. This clearly illustrates the trade-off involved and shows the potential for parallel and distributed variants.

# 4.2 RESIDUAL NETWORKS

We next considered a deeper residual network architecture described in section 4.2 of He et al. (2015b), with  $n = 3$ . This led to a 20-layer residual network including 9 shortcut connections. As in He et al. (2015b), we used batch normalization (Ioffe & Szegedy, 2015) and the same initialization method (He et al., 2015a).

# 4.2.1 COMPARISON WITH SGD WITH MOMENTUM

We trained the residual network using SGD with momentum, and other parameter settings as described in He et al. (2015b). We also trained the network using L-SR1 with default settings. These included, a memory size of 5, a trust-region radius decrease factor of 0.5, and a trust-region radius increase factor of 2.0. We used the same mini-batch size of 128 for both algorithms. The following figure illustrates the training and test losses after 90 epochs of training. Based on the learning rate schedule used, the learning rate was equal to 0.1 through the 90 epochs, for SGD with momentum. It may be seen that L-SR1 is competitive with SGD with momentum in training and in test.

# 4.2.2 VARIATION OF L-SR1 HYPERPARAMETERS

We varied the following hyperparameters of L-SR1 in turn, keeping the remaining fixed. In each case, we trained the network for 90 epochs.

- We first considered varying the increase and decrease factors together. We considered a trust-region radius decrease factor of 0.5 and 0.8, and a trust-region radius increase factor 1.2 and 2.0. The respective default values of these factors are 0.5 and 2.0 respectively. This led to four different combinations of decrease and increase factors. We kept the memory size and mini-batch size fixed at 5 and 128 respectively.  
- Having found a decrease factor of 0.8 and an increase factor of 1.2 to have the minimum test loss over 90 epochs for the four cases considered above, we next considered memory

![](images/46be49debbb1829ea4419db8717914fc2d8f8bea28622fb6a7da934a19921f11.jpg)  
Figure 3: LSR1 vs SGD, on the CIFAR10 dataset, using a residual network.

sizes of 2 and 10 (in addition to 5, which we tried earlier), keeping the mini-batch size, decrease factor, and increase factor fixed at 128, 0.8, and 1.2 respectively.

- The memory size of 2 had the lowest minimum test loss over 90 epochs, relative to memory sizes of 5 and 10. So we finally considered mini-batch sizes of 512, 1024 and 8192 (in addition to 128, which we tried earlier), keeping the memory size, decrease factor, and increase factor fixed at 2, 0.8, and 1.2 respectively.

![](images/9992c73685c1a9bac8a1d2a96b1ad7531c4efb5823d1b9fe04f80d5fd2449ffe.jpg)  
Figure 4: Variation of trust region radius increase and decrease factors, on the CIFAR10 dataset, using a residual network.

The following may be noted, based on the experiments with L-SR1 for training a residual network on CIFAR10.

- While there is potential value in increasing and decreasing the trust region radius at different rates, our experiments suggest that it may not be necessary to tune these hyperparameters.

![](images/e8bdb5b22b5a46e281971ec863a717d1e05080421e0df30acf973290e7bc18f0.jpg)  
Figure 5: Variation of mini-batch size and memory size with number of epochs, on the CIFAR10 dataset, using a residual network. Note that the scales on the y-axes are different.

![](images/f64aa8c46157d09a4782a9edff284d99bfa830b8e9969e188d97eca6c9b8201b.jpg)

- There is no noticeable performance gain from using a higher memory size in L-SR1. Furthermore, using a smaller memory size performs at least as well as in the default case. This is good news, due to the consequent savings in storage and computational resources.  
- L-SR1 is relatively insensitive to a 4-fold increase in mini-batch size from 128 to 512, and a further 2-fold increase to 1024. Furthermore, it may be seen that the average percentage decrease in loss per iteration (mini-batch processing) increases at roughly the same rate, as the mini-batch size, during training and testing. So, while achieving the same loss using a larger mini-batch size would take more epochs, the training time can be sped up by distributed processing of large mini-batches. Hence, on a computational architecture that requires use of large mini-batches, we could use L-SR1 to efficiently achieve good training and out-of-sample performance, through distributed processing. Finally, we found that SGD with momentum performed similarly to L-SR1 as the mini-batch size was increased from 128 through to 1024. This seemed surprising, given the predominant use of smaller mini-batch sizes with variants of SGD in the literature.

# 5 CONCLUSIONS

In this paper, we have described L-SR1, a new second order method to train deep neural networks. Our experiments suggest that this approach is at the very least, competitive, with other first order methods, and substantially better than L-BFGS, a well-known second order method. Our experiments also appear to validate our intuition about the ability of L-SR1 to overcome key challenges associated with second order methods, such as inappropriate handling of saddle points, and poor conditioning of the Hessian. Our experimentation with the hyperparameters of L-SR1 suggested that it is relatively robust with respect to them, and requires minimal tuning. Furthermore, we observe that a 10-fold increase in mini-batch size does not noticeably compromise performance over a reasonable number of epochs. Consequently, the progress per mini-batch iteration scales roughly linearly with the increase in mini-batch size. This suggests that L-SR1 holds promise for distributed training of deep networks, and we see our work as an important step toward that goal.

# REFERENCES

Johannes Brust, Jennifer B. Erway, and Roummel F. Marcia. On solving 1-sr1 trust-region subproblems. arXiv.org, 8 2016. arXiv:1506.07222v3.

J. V. Burke, A. Wiegman, and L. Xu. Limited memory bfgs updating in a trust-region framework, 2008. Working paper.  
Richard H. Byrd, Jorge Nocedal, and Robert B. Schnabel. Representations of quasi-newton matrices and their use in limited-memory methods. Mathematical Programming, 63(1):129-156, 1 1994.  
A. R. Conn, N. I. M. Gould, and Ph. L. Toint. Convergence of quasi-newton matrices generated by the symmetric rank one update. Mathematical Programming, 50(1):177-195, 3 1991.  
Yann Dauphin, Razvan Pascanu, Caglar Gülçehre, Kyunghyun Cho, Surya Ganguli, and Yoshua Bengio. Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. CoRR, abs/1406.2572, 2014. URL http://arxiv.org/abs/1406.2572.  
Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc V. Le, and Andrew Y. Ng. Large scale distributed deep networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 25, pp. 1223-1231. Curran Associates, Inc., 2012. URL http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf.  
John E. Dennis Jr. and Jorge J. More. Quasi-newton methods, motivation and theory. SIAM Review, 19(1):46-89, 1 1977.  
John E. Dennis Jr. and Robert B. Schnabel. Numerical methods for unconstrained optimization and nonlinear equations. Prentice Hall, 1 edition, 1983.  
William W. Hager. Updating the inverse of a matrix. SIAM Review, 31(2):221-239, 1989. ISSN 00361445. URL http://www.jstor.org/stable/2030425.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. CoRR, abs/1502.01852, 2015a. URL http://arxiv.org/abs/1502.01852.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015b. URL http://arxiv.org/abs/1512.03385.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015. URL http://arxiv.org/abs/1502.03167.  
Humaid Khalfan, Richard H. Byrd, and Robert B. Schnabel. A theoretical and experimental study of the symmetric rank one update. SIAM Journal on Optimization, 3(1):1-24, 1993.  
Quoc V. Le, Jiquan Ngiam, Adam Coates, Ahbik Lahiri, Bobby Prochnow, and Andrew Y. Ng. On optimization methods for deep learning. In Lise Getoor and Tobias Scheffer (eds.), ICML, pp. 265-272. Omnipress, 2011. URL http://dblp.uni-trier.de/db/conf/icml/icml2011.html#LeNCLPN11.  
Dong C. Liu and Jorge Nocedal. On the limited memory bfgs method for large scale optimization. Mathematical Programming, 45(1):503-528, 1989.  
James Martens. Second-Order Optimization for Neural Networks. PhD thesis, Graduate Department of Computer Science, University of Toronto, 2016.  
Jorge Nocedal. Updating quasi-newton matrices with limited storage. Mathematics of Computation, 35(151):773-782, 7 1980.  
Jorge Nocedal and Stephen J. Wright. Numerical Optimization. Springer-Verlag, New York, 2 edition, 2006.
