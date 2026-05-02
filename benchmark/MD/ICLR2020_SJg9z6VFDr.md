# ORDINARY DIFFERENTIAL EQUATIONS ON GRAPH NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recently various neural networks have been proposed for irregularly structured data such as graphs and manifolds. To our knowledge, all existing graph networks have discrete depth. Inspired by neural ordinary differential equation (NODE) for data in the Euclidean domain, we extend the idea of continuous-depth models to graph data, and propose graph ordinary differential equation (GODE). The derivative of hidden node states are parameterized with a graph neural network, and the output states are the solution to this ordinary differential equation. We demonstrate two end-to-end methods for efficient training of GODE: (1) indirect back-propagation with the adjoint method; (2) direct back-propagation through the ODE solver, which accurately computes the gradient. We demonstrate that direct backprop outperforms the adjoint method in experiments. We then introduce a family of bijective blocks, which enables  $\mathcal{O}(1)$  memory consumption. We demonstrate that GODE can be easily adapted to different existing graph neural networks and improve accuracy. We validate the performance of GODE in both semi-supervised node classification tasks and graph classification tasks. Our GODE model achieves a continuous model in time, memory efficiency, accurate gradient estimation, and generalizability with different graph networks.

# 1 INTRODUCTION

Convolutional neural networks (CNN) have achieved great success in various tasks, such as image classification (He et al., 2016) and segmentation (Long et al., 2015), video processing (Deng et al., 2014) and machine translation (Sutskever et al., 2014). However, CNNs are limited to data that can be represented by a grid in the Euclidean domain, such as images (2D grid) and text (1D grid), which hinders their application in irregularly structured datasets.

A graph data structure represents objects as nodes and relations between objects as edges. Graphs are widely used to model irregularly structured data, such as social networks (Kipf & Welling, 2016), protein interaction networks (Fout et al., 2017), citation and knowledge graphs (Hamaguchi et al., 2017), and point cloud datasets (Hackel et al., 2017). Early works use traditional methods such as random walk (Lovasz et al., 1993), independent component analysis (ICA) (Hyvärinen & Oja, 2000) and graph embedding (Yan et al., 2006) to model graphs, however their performance is inferior due to the low expressive capacity. Furthermore, ICA and graph embedding treat edge information as attributes of the nodes, while ignoring the information propagation on the edges.

Recently a new class of models called graph neural networks (GNN) (Scarselli et al., 2008) were proposed. GNNs use a neural network to model the iterative propagation of node states on a graph and have a larger capacity to capture information in a graph. Li et al. (2015) used gated recurrent units to model the propagation process. However, these models only consider the message propagation, while ignoring local structures of a graph.

Inspired by the success of CNNs, researchers generalize convolution operations to graphs to capture the local information. There are mainly two types of methods to perform convolution on a graph: spectral methods and non-spectral methods. Spectral methods typically first compute the graph Laplacian, then perform filtering in the spectral domain (Bruna et al., 2013). Other methods aim to approximate the filters without computing the graph Laplacian, in order to accelerate the running speed (Defferrard et al., 2016). For non-spectral methods, the convolution operation is directly performed in the graph domain, aggregating information only from the neighbors of a node (Duvenaud

et al., 2015; Atwood & Towsley, 2016). The recently proposed GraphSAGE (Hamilton et al., 2017) learns a convolution kernel in an inductive manner.

To our knowledge, all existing GNN models mentioned above have a structure of discrete layers. The discrete structure makes it hard for the GNN to model continuous diffusion processes (Freidlin & Wentzell, 1993; Kondor & Lafferty, 2002) in graphs. The recently proposed neural ordinary differential equation (NODE) (Chen et al., 2018) views a neural network as an ordinary differential equation (ODE), whose derivative is parameterized by the network, and the output is the solution to this ODE. We extend NODE from the Euclidean domain to graphs and propose graph ordinary differential equations (GODE), where the message propagation on a graph is modeled as an ODE. GODE on a graph can be viewed as a continuous diffusion process and can be solved with any ODE solver. We then introduce two methods for efficient training of GODE and demonstrate the superior performance of GODE in experiments. Our contribution can be summarized as follows:

1. We generalize ordinary differential equation to graph data and model the continuous diffusion process on a graph.  
2. We apply two methods for end-to-end training of GODE, the adjoint method and direct backpropagation through the ODE solver. We demonstrate direct back-prop is robust to the instability of ODE.  
3. We introduce a family of bijective blocks, which achieve  $\mathcal{O}(1)$  memory consumption. The bijective blocks enable training of GODE on large graphs.  
4. We demonstrate GODE is generalizable, and can be applied to various GNN structures.

# 2 RELATED WORKS

# 2.1 GRAPH NEURAL NETWORKS

GNNs can be divided into two categories: spectral methods and non-spectral methods. Spectral GNNs perform filtering in the Fourier domain of a graph, thus need information of the whole graph to determine the graph Laplacian. In contrast, non-spectral GNNs only consider message aggregation around neighbor nodes, therefore are localized and generally require less computation (Zhou et al., 2018).

We first briefly introduce several spectral methods. Bruna et al. (2013) first introduced graph convolution in the Fourier domain based on the graph Laplacian, however the computation burden is heavy because of non-localized filters. Henaff et al. (2015) incorporated a graph estimation procedure in spectral networks and parameterized spectral filters into a localized version with smooth coefficients. Defferrard et al. (2016) used Chebyshev expansion to approximate the filters without the need to compute the graph Laplacian and its eigenvectors, therefore significantly accelerated the running speed. Kipf & Welling (2016) proposed to use a localized first-order approximation of graph convolution on graph data and achieved superior performance in semi-supervised tasks for node classification. Defferrard et al. (2016) proposed fast localized spectral filtering on graphs.

Non-spectral methods typically define convolution operations on a graph, only considering neighbors of a certain node. MoNet (Monti, 2017) uses a mixture of CNNs to generalize convolution to graphs. GraphSAGE (Hamilton et al., 2017) samples a fixed size of neighbors for each node for fast localized inference. Graph attention networks (Veličković et al., 2017) learn different weights for different neighbors of a node. The graph isomorphism network (GIN) (Xu et al., 2018a) has a structure as expressive as the Weisfeiler-Lehman graph isomorphism test.

# 2.2 NEURAL NETWORKS AND DIFFERENTIAL EQUATIONS

There have been efforts to view neural networks as differential equations. Lu (2017) viewed a residual network as a discretization of a differential equation and proposed several new architectures based on numerical methods in ODE solver. Haber & Ruthotto (2017) proposed a stable architecture based on analysis of the ODE. Chen et al. (2018) proposed neural ordinary differential equation (NODE), which treats the neural network as a continuous ODE. For a neural network with discrete layers, the parameters can be optimized with layer-wise back-propagation; for a continuous model, the adjoint method has long been widely used in optimal control (Stapor et al., 2018) and geophysical problems (Plessix, 2006). Dupont et al. (2019) proposed augmented neural ODEs to improve the

expressive capacity of ODEs. NODE was later used in a continuous normalizing flow for generative models (Grathwohl et al., 2018).

# 2.3 BIJECTIVE BLOCKS

Bijective blocks are a family of neural network blocks whose forward function is a bijective mapping. Therefore, the input to a bijective block can be accurately reconstructed from its outputs. Bijective blocks have been used in normalizing flow (Rezende & Mohamed, 2015; Dinh, 2016; Kingma & Dhariwal, 2018; Dinh et al., 2014; Kingma et al., 2016), where the model is required to be invertible in order to calculate the log-density of data distribution. Later on, Jacobsen et al. (2018) used bijective blocks to build invertible networks. Gomez et al. (2017) proposed to use bijective blocks to perform back propagation without storing activation, which achieves a memory-efficient network structure. They were able to discard activation of middle layers, because each layer's activation can be reconstructed from the next layer with bijective blocks.

# 3 GRAPH ORDINARY DIFFERENTIAL EQUATIONS

We first introduce graph neural networks with discrete layers, then extend to the continuous case and introduce graph ordinary differential equations (GODE).

# 3.1MESSAGEPASSINGINGNN

As shown in Fig. 1, a graph is represented with nodes (marked with circles) and edges (solid lines). We assign a unique color to each node for ease of visualization. Current GNNs can generally be represented in a message passing scheme (Fey & Lenssen, 2019):

$$
m e s s a g e _ {(v, u)} = \phi^ {(k)} \left(x _ {k - 1} ^ {u}, x _ {k - 1} ^ {v}, \mathbf {e} _ {u, v}\right) \tag {1}
$$

$$
a g g r e g a t i o n _ {u} = \zeta_ {v \in \mathcal {N} (u)} \left(m e s s a g e _ {(v, u)}\right) \tag {2}
$$

$$
x _ {k} ^ {u} = \gamma^ {(k)} \left(x _ {k - 1} ^ {u}, a g g r e g a t i o n _ {u}\right) \tag {3}
$$

where  $x_{k}^{u}$  represents states of the  $u$ th node in the graph at  $k$ th layer and  $\mathbf{e}_{u,v}$  represents the edge between nodes  $u$  and  $v$ .  $\mathcal{N}(u)$  represents the set of neighbor nodes for node  $u$ .  $\zeta$  represents a differentiable, permutation invariant operation such as mean, max or sum.  $\gamma^{(k)}$  and  $\phi^{(k)}$  are differentiable functions parameterized by neural networks.

For a specific node  $u$ , a GNN can be viewed as a 3-stage model, corresponding to Eq. 1-3: (1) Message passing, where neighbor nodes  $v \in \mathcal{N}(u)$  send information to node  $u$ , denoted by  $message_{(v,u)}$ . The message is generated from function  $\phi(\cdot)$ , parameterized by a neural network. (2) Message aggregation, where a node  $u$  aggregates all messages from its neighbors  $\mathcal{N}(u)$ , denoted as  $aggregation_u$ . The aggregation function  $\zeta$  is typically permutation invariant operations such as mean and sum, because graphs are invariant to permutation. (3) Update, where the states of a node are updated according to its original states  $x_{k-1}^u$  and aggregation of messages  $aggregation_u$ , denoted as  $\gamma(\cdot)$ .

# 3.2 FROM DISCRETE MODELS TO CONTINUOUS MODELS

We first consider GNNs with residual connection (Xu et al., 2018b; He et al., 2016) in the form of addition, which can be represented as:

$$
x _ {k + 1} = x _ {k} + f _ {k} \left(x _ {k}\right) \tag {4}
$$

where  $x_{k}$  is the states of the graph in the  $k$ th layer;  $f_{k}(\cdot)$  is any differentiable function defined on the graph, whose output has the same shape as its input.  $f_{k}(\cdot)$  denotes operations defined by Eq. 1-3. For the ease of notation, we omit node index  $u$  in  $x_{k}$ . The discrete update process is shown from left to right in Fig. 1(a).

Equations 1-4 represent GNNs with discrete layers. When we add more layers with shared weights, and let the stepsize in Eq. 4 goes to infinitesimal, the difference equation turns into an ordinary differential equation:

$$
\frac {\mathrm {d} z (t)}{\mathrm {d} t} = f (z (t), t) \tag {5}
$$

![](images/0099291f97f257711dd1ea9be2a4103544fedc03c4ed00ac13458e3b6dde3cda.jpg)  
Figure 1: Diecrete-time and continuous-time models on a graph. Nodes are represented with circles, and each node is represented with a unique color. Edges are represented with solid lines. For discrete-time models in (a), the hidden states of nodes are updated with discrete steps. For continuous-time models in (b), hidden states of each node evolves continuously with time. The dynamics of nodes are represented with dashed lines, with the same color as corresponding nodes.

We use  $z(t)$  in the continuous case and  $x_{k}$  in the discrete case to represent hidden states of a graph.  $f(\cdot)$  is the derivative parameterized by a GNN as in Eq. 1-3. Since it's an ODE on a graph, we name Eq. 5 as Graph-ODE (GODE). Note that a key difference between Eq. 4 and 5 is the form of  $f$ : in the discrete case, different layers (different  $k$  values) have their own function  $f_{k}$ ; while in the continuous case,  $f$  is shared across all time  $t$ . In GODE, states of each node evolve with time according to Eq. 5. The dynamics of nodes are represented with a dashed line in Fig. 1(b).

The forward pass of GNNs with discrete layers can be written as:

$$
x _ {0} = \text {i n p u t}, \quad x _ {1} = x _ {0} + f _ {0} \left(x _ {0}\right), \quad \dots , \quad x _ {K} = x _ {K - 1} + f _ {K - 1} \left(x _ {K - 1}\right) \tag {6}
$$

where  $K$  is the total number of layers. Then an output layer (e.g. fully-connected layer for classification) is applied on  $x_{K}$ .

The forward pass of a GODE is:

$$
z (T) = z (0) + \int_ {t = 0} ^ {T} \frac {\mathrm {d} z (t)}{\mathrm {d} t} \mathrm {d} t = \text {i n p u t} + \int_ {t = 0} ^ {T} f (z (t), t) \mathrm {d} t \tag {7}
$$

where  $z(0) =$  input and  $T$  is the integration time, corresponding to number of layers  $K$  in the discrete case. The transformation of states  $z$  is modeled as the solution to the GODE. Then an output layer is applied on  $z(T)$ . Integration in the forward pass can be performed with any ODE solver, such as the Euler Method, Runge-Kutta Method, VODE solver and Dopris Solver (Milne & Milne, 1953; Brown et al., 1989; Ascher et al., 1997).

# 4 TRAINING OF GODE

Neural networks with discrete layers can be trained with back-propagation (Rumelhart et al., 1985). In GODE, the back-propagation algorithm needs to be modified to deal with continuous cases. We first introduce the adjoint method, then address its sensitivity to numerical errors. Next, we introduce direct back-propagation through the ODE solver, which is more resistant to numerical errors. However, direct back-propagation requires large memory. To solve this, we introduce memory-efficient bijective blocks.

# 4.1 BACK-PROP WITH ADJOINT METHOD IS SENSITIVE TO NUMERICAL ERROR

The adjoint method is widely used in optimal process control and functional analysis (Stapor et al., 2018; Pontryagin, 2018). We follow the method by (Chen et al., 2018). Denote model parameters as  $\theta$ , which is independent of time. Define the adjoint as:

$$
a (t) = \frac {\partial L}{\partial z (t)} \tag {8}
$$

where  $L$  is the loss function. Then we have

$$
\frac {\mathrm {d} a (t)}{\mathrm {d} t} = - a (t) ^ {T} \frac {\partial f (z (t) , t , \theta)}{z (t)}, \quad \frac {\mathrm {d} L}{\mathrm {d} \theta} = - \int_ {T} ^ {0} a (t) ^ {T} \frac {\partial f (z (t) , t , \theta)}{\partial \theta} \mathrm {d} t \tag {9}
$$

![](images/f6fb5f9c8dab3035169300011542bf51ee2254fe0b2c63d763e592c758a64bb6.jpg)  
(a) forward pass

![](images/18aad398b563fbd90b4e120e7f1b56647bc5d41619649697ae54a989037303f0.jpg)  
(b) back-prop with adjoint method  
Figure 2: Comparison of two methods for back-propagation on GODE. As in figure (a), the ODE solver is discretized at points  $\{t_0, t_1, \dots, t_N\}$  during forward pass. Black dashed curve shows hidden state solved in forward-time, denoted as  $z(t)$ . Figure (b) shows the adjoint method, red solid line shows the hidden state solved in reverse-time, denoted as  $h(t)$ . Ideally  $z(t) = h(t)$  and dashed curve overlaps with solid curve; however, the reverse-time solution could be numerically unstable, and causes  $z(t) \neq h(t)$ , thus causes error in gradient. Figure (c) shows the direct back-propagation through ODE solver. Due to the discretization of the ODE solver, the forward pass can be viewed as a discrete layer model, whose depth equals the number of time steps in the ODE solver. In direct back-propagation, we have  $z(t_i) = h(t_i)$ , which can be achieved with bijective blocks.

![](images/46d969442ab15412acce69ec41201ca1cd1bda7eaa2bc711cd00380e61f711eb.jpg)  
(c) direct back-prop through solver

with detailed proof is in appendix G. Then we can perform gradient descent to optimize  $\theta$  to minimize  $L$ . Eq. 9 is a reverse-time integration, which can be solved with any ODE solver (Chen et al., 2018). To evaluate  $\frac{\partial f(z(t), t, \theta)}{\partial \theta}$ , we need to determine  $z(t)$  by solving Eq. 5 reverse-time (Directly storing  $z(t)$  during forward pass requires a large memory consumption, because the continuous model is equivalent to an infinite-layer model). To summarize, in the forward pass we solve Eq. 5 forward in time; in the backward pass, we solve Eq. 5 and 9 reverse in time, with initial condition determined from Eq. 8 at time  $T$ .

We give an intuition why the reverse-time ODE solver causes inaccurate gradient in adjoint methods. The backward pass (Eq. 9) requires determining  $f(z(t), t, \theta)$  and  $\frac{\partial f(z(t), t, \theta)}{\partial \theta}$ , which requires determining  $z(t)$  by solving Eq. 5 reverse-time. As shown in Fig. 2 (a,b), the hidden state solved forward-time  $(z(t_i))$  and the hidden state solved reverse-time  $(h(t_i))$  may not be equal; this could be caused by the instability of reverse-time ODE, and is represented by the mismatch between  $z(t)$  (dashed curve) and  $h(t)$  (solid curve). Error  $h(t) - z(t)$  will cause error in gradient  $\frac{\mathrm{d}L}{\mathrm{d}\theta}$ .

Proposition 1 For an ODE in the form  $\frac{\mathrm{d}z(t)}{\mathrm{d}t} = f(z(t), t)$ , denote the Jacobian of  $f$  as  $J_f$ . If this ODE is stable both in forward-time and reverse-time, then  $\operatorname{Re}(\lambda_i(J_f)) = 0 \forall i$ , where  $\lambda_i(J_f)$  is the  $i$ th eigenvalue of  $J_f$ , and  $\operatorname{Re}(\lambda)$  is the real part of  $\lambda$ .

Detailed proof is in appendix D. Proposition 1 indicates that if the Jacobian of the original system Eq. 5 has eigenvalues whose real-part are not 0, then either the reverse-time or forward-time ODE is unstable. When  $|\mathrm{Re}(\lambda)|$  is large, either forward-time or reverse-time ODE is sensitive to numerical errors. This phenomenon is also addressed in Chang et al. (2018). This instability affects the accuracy of solution to Eq. 5 and 9, thus affects the accuracy of the computed gradient.

# 4.2 DIRECT BACK-PROPAGATION THROUGH ODE SOLVER

The adjoint method might be sensitive to numerical errors when solving the ODE in reverse-time. To resolve this, we propose to directly back-propagate through the ODE solver.

As in Fig. 2(a), since the ODE solver uses discretization for numerical integration, it can be viewed as a sequence of discrete layers whose depth is the total number of time points  $\{t_i\}$ . Fig. 2(c) demonstrates the direct back-propagation with accurate hidden states  $h(t_i)$ , which can be achieved with two methods: (1) the activation  $z(t_i)$  can be saved in cache for back-prop; or (2) we can accurately reconstruct  $z(t_i)$  from  $z(t_{i+1})$  as in Gomez et al. (2017). Therefore direct back-prop is accurate, regardless of the stability of Eq. 5. We demonstrate method (2) in the next section.

Similar to the continuous case, we can define the adjoint with discrete time. Then we have:

$$
a _ {i} = \frac {\partial L}{\partial z (t _ {i})}, \quad a _ {i + 1} = a _ {i} \frac {\partial z (t _ {i + 1})}{\partial z (t _ {i})}, \quad \frac {\mathrm {d} L}{\mathrm {d} \theta} = \sum_ {i = 1} ^ {N} a _ {i} \frac {\partial z (t _ {i})}{\partial \theta} \tag {10}
$$

where  $\{t_0, t_1, \dots, t_i, \dots, t_N\}$  is the set of discretized evaluation time points and  $a_i$  is the adjoint for the  $i$ th step in discrete forward-time ODE solution. Eq. 10 are in the discrete case, corresponding

![](images/4843c867b6d75f8cda24a12de7e2e6462c31812b232c416de664dd675cb5dbcb.jpg)

![](images/3ba2b815b08e23df004d8a917af75fcde08186ae630d7f6dfe483a3b663b9365.jpg)  
Figure 3: Structure of bijective blocks.  $F$  and  $G$  can be any differentiable neural network whose output has the same shape as its input. Blue dot (Orange diamond) represents the forward (inverse) of a bijective function, corresponding to  $\psi (\psi^{-1})$  in Eq. 11. Left (right) figure represents the forward (inverse) as in Eq. 11.

to Eq. 9 in the continuous case. We show Eq. 9 can be derived from an optimization perspective. Detailed derivations of Eq. 9-10 are in appendix F and G.

# 4.3 MEMORY-EFFICIENT BIJECTIVE BLOCKS

Direct back-propagation through the ODE solver is accurate regardless of stability of the GODE model. However, the effective depth is equivalent to the number of steps in the forward pass. In the conventional back-propagation scheme, the activation for each layer needs to be cached during the forward pass, which will be used later in the backward pass. Therefore, the direct back-propagation requires a memory of  $K \times M$ , where  $M$  is the memory size for a single layer, and  $K$  is the number of evaluation steps in the forward pass. The large memory requirements hinders the application of GODE. To solve this problem, we introduce bijective blocks with  $\mathcal{O}(1)$  memory consumption.

# 4.3.1 BIJECTIVE BLOCKS

Input  $x$  is split into two parts  $(x_{1}, x_{2})$  of the same size (e.g.  $x$  has shape  $N \times C$ , where  $N$  is batch size,  $C$  is channel number; we can split  $x$  into  $x_{1}$  and  $x_{2}$  with shape  $N \times \frac{C}{2}$ ). The forward and inverse of a bijective block can be denoted as:

$$
\left\{ \begin{array}{l} y _ {2} = \psi \left(x _ {2}, F \left(x _ {1}\right)\right) \\ y _ {1} = \psi \left(x _ {1}, G \left(y _ {2}\right)\right) \end{array} \quad \left\{ \begin{array}{l} x _ {1} = \psi^ {- 1} \left(y _ {1}, G \left(y _ {2}\right)\right) \\ x _ {2} = \psi^ {- 1} \left(y _ {2}, F \left(x _ {1}\right)\right) \end{array} \right. \right. \tag {11}
$$

where the output of a bijective block is denoted  $(y_{1},y_{2})$  with the same size as  $(x_{1},x_{2})$ .  $F$  and  $G$  are any differentiable neural networks, whose output has the same shape as the input.  $\psi (\alpha ,\beta)$  is a differentiable bijective function w.r.t  $\alpha$  when  $\beta$  is given;  $\psi^{-1}(\alpha ,\beta)$  is the inverse function of  $\psi$ .

We give an example of  $\psi$ .

$$
\eta = \psi (\alpha , \beta) = \alpha \times \exp (2 \beta), \alpha = \psi^ {- 1} (\eta , \beta) = \eta \times \exp (- 2 \beta) \tag {12}
$$

Structure of bijective blocks is shown in Fig. 3, where  $F$  and  $G$  are represented with squares,  $\psi$  is denoted with blue dots, and  $\psi^{-1}$  is denoted with orange diamonds.

Theorem 1 If  $\psi (\alpha ,\beta)$  is a bijective function w.r.t  $\alpha$  when  $\beta$  is given, then the block defined by Eq. 11 is a bijective mapping.

Proof of Theorem 1 is given in the appendix. Based on this, we can apply different  $\psi$  functions for different tasks.

Bijective blocks can accurately reconstruct its input from its output based on Eq. 11. This enables accurate reconstruction of  $h(t_{i}) = z(t_{i})$ .

# 4.3.2 BACKPROP WITHOUT STORING ACTIVATION

We follow the work of Gomez et al. (2017) with two important modifications: (1) We generalize to a family of bijective blocks with different  $\psi$ , while Gomez et al. (2017) restrict the form of  $\psi$  to be sum. (2) We propose a parameter state checkpoint method, which enables bijective blocks to be called more than once, while still generating accurate inversion.

The algorithm is summarized in Algo 1. We write the pseudo code for forward and backward function as in PyTorch. Note that we use "inversion" to represent reconstructing input from the output, and use "backward" to denote calculation of the gradient. To reduce memory consumption, in the forward function, we only keep the outputs  $y_{1}, y_{2}$  and delete all other variables and computation

graphs. In the backward function, we first "inverse" the block to calculate  $x_{1}, x_{2}$  from  $y_{1}, y_{2}$ , then perform a local forward and calculate the gradient  $\frac{\partial[y_1,y_2]}{\partial[x_1,x_2]}$ .

Note that in GODE,  $F$  and  $G$  need to be reused for different steps in ODE solver. At different steps, the running statistics (e.g. sample mean and variance in Batch Normalization layer) in  $F, G$  are different. Therefore the inversion is inaccurate in this case. To solve this, we keep the states (running statistics, random seed, et al.) of  $F$  and  $G$  in cache (checkpoint), and reset states of  $F, G$  during inversion in the backward function. The state-checkpoint method requires minimal memory while enabling re-use of bijective blocks in ODE solvers.

Algorithm 1: Function for memory-efficient bijective blocks  
Forward (cache,  $x_{1},x_{2},F,G,\psi$  ) Backward cache,  $y_{1},y_{2},F,G,\psi ,\frac{\partial L}{\partial y_1},\frac{\partial L}{\partial y_2})$  cache.save([F states,  $G$  states]) Reset  $F$  and  $G$  states from cache forward in Eq. 11 Inverse from  $y_{1},y_{2}$  to  $x_{1},x_{2}$ $\eta_{2} = G(y_{2}),x_{1} = \psi^{-1}(y_{1},\eta_{2})$ $\eta_{1} = F(x_{1}),x_{2} = \psi^{-1}(y_{2},\eta_{1})$  Local forward pass and gradient   
delete  $\eta_1,\eta_2,x_1,x_2$    
delete computation graphs generated by  $F$  and  $G$    
return cache,  $y_{1},y_{2}$ $X_{1},X_{2} = x_{1}.\mathrm{detach()},x_{2}.\mathrm{detach()}$  calculate  $Y_{1},Y_{2}$  from  $X_{1},X_{2}$  as Eq. 11 determine  $\partial [Y_1,Y_2] / \partial [X_1,X_2,\theta_F,\theta_G]$ $\frac{\partial L}{\partial[x_1,x_2]} = \frac{\partial L}{\partial[y_1,y_2]}\frac{\partial[Y_1,Y_2]}{\partial[X_1,X_2]}$ $\frac{\partial L}{\partial[\theta_F,\theta_G]} = \frac{\partial L}{\partial[y_1,y_2]}\frac{\partial[Y_1,Y_2]}{\partial[\theta_F,\theta_G]}$  delete  $Y_{1},Y_{2},X_{1},X_{2}$    
return  $\partial L / \partial [x_1,x_2],\partial L / \partial [\theta_F,\theta_G]$

# 4.4 MEMORY-EFFICIENT DIRECT BACK-PROP THROUGH ODE SOLVER

We use a bijective block (Eq. 11) as  $f$  in GODE model (Eq. 5). The full algorithm is summarized in Algo. 2 in appendix B. The forward pass is determined by an ODE solver, where the function is evaluated (in memory-efficient manner defined in Algo. 1) at multiple time points for numerical integration; then to calculate gradients, back-propagation is directly applied on the ODE solver. Since each call on the bijective blocks is memory-efficient, Algo. 2 requires a constant memory usage.

By combining memory-efficient bijective blocks with direct back-propagation, we are able to train a continuous-time model on graphs. The proposed method has the following advantages: (1) continuous-time models, which enables us to model continuous diffusion processes on graphs; (2) constant memory usage because of the memory-efficient bijective blocks; (3) adaptive computation, achieved by using an ODE solver with adaptive step size; (4) accurate estimation of gradient regardless of stability of the original system, achieved by direct back-propagation through the ODE solver; (5) generalization to different graph network structures, since any differentiable GCN structure can be applied in  $F$ ,  $G$  in Eq. 11.

# 5 EXPERIMENTS

# 5.1 DATASETS

We performed experiments on several benchmark datasets, including 2 bioinformatic graph classification datasets (MUTAG and PROTEINS), 3 social network graph classification datasets (IMDB-BINRAY, REDDIT-BINARY and COLLAB) (Yanardag & Vishwanathan, 2015), and 3 citation networks (Cora, CiteSeer and PubMed). For graph classification tasks, nodes in the bioinformatic graphs have categorical features, while nodes in social network graphs have no features. Different from the experiment settings in Xu et al. (2018a), we input the raw dataset into our models without pre-processing. For node classification tasks, we performed transductive inference and strictly followed the train-validation-test split by Kipf & Welling (2016), where less than  $6\%$  nodes are used as training examples. Details of datasets are summarized in appendix A.

<table><tr><td></td><td>MUTAG</td><td>PROTEINS</td><td>IMDB</td><td>REDDIT</td><td>COLLAB</td></tr><tr><td>adjoint</td><td>68.1±4.6</td><td>67.0±3.7</td><td>72.1±0.4</td><td>69.5±5.9</td><td>80.0±1.3</td></tr><tr><td>direct</td><td>80.8±8.3</td><td>73.9±3.1</td><td>74.6±5.1</td><td>92.4±2.1</td><td>82.0±2.1</td></tr></table>

Table 1: Accuracy of adjoint method and direct back-prop. We trained a GODE model with GCN as the derivative function. For the same column, all experiment settings are the same except the back-prop method.  

<table><tr><td>Depth</td><td>Memory-efficient</td><td>Conventional</td></tr><tr><td>10</td><td>2.2G</td><td>5.3G</td></tr><tr><td>20</td><td>2.6G</td><td>10.5G</td></tr></table>

Table 2: Memory consumption of bijective blocks. "Conventional" represents storing activation of all layers in cache, "memory-efficient" represents our method in Algo. 1.

# 5.2 MODEL STRUCTURES

GODE can be applied to any graph neural network by simply replacing  $f$  in Eq. 5 with corresponding structures, or replacing  $F, G$  in Eq. 11 with other structures. To demonstrate that GODE is easily generalized to existing structures, we used several different GNN architectures, including the graph convolutional network (GCN) (Kipf & Welling, 2016), graph attention network (GAT) (Veličković et al., 2017), graph network approximated with Chebyshev expansion (ChebNet) (Defferrard et al., 2016), and graph isomorphism network (GIN) (Xu et al., 2018a). For a fair comparison, we trained GNNs with different depths of layers (1-3 middle layers, besides an initial layer to transform data into specified channels, and a final layer to generate prediction), and reported the best results among all depths for each model structure.

On the same task, different models use the same hyper-parameters on model structures, such as channel number. For graph classification tasks, we set the channel number of hidden layers as 32 for all models; for ChebNet, we set the number of hops as 16. For node classification tasks, we set the channel number as 16 for GCN and ChebNet, and set number of hops as 3 for ChebNet; for GAT, we used 8 heads, and set each head as 8 channels.

# 5.3 TRAINING SCHEMES AND EVALUATION METRICS

We implemented all models using PyTorch and PyTorch_Geometric Library (Fey & Lenssen, 2019). For graph classification tasks, all models were trained with the Adam optimizer for 150 epochs, with an initial learning rate of 0.01, and decayed by a factor of 0.3 every 30 epochs; the batchsize was set as 32. For node classification tasks, all models were trained with the Adam optimizer for 200 epochs, with an initial learning rate of 0.1, and decayed by a factor of 0.1 at epoch 100.

For every GNN structure, we experimented with different number of hidden layers (1,2,3), calculated the mean and variance of accuracy of 10 runs, and reported the best result for each model under each task.

# 5.4 COMPARISON OF BACK-PROPAGATION METHODS

We compared the adjoint method and direct back-propagation on the same network, and demonstrated direct back-prop generates higher accuracy. We trained a GODE model with a GCN to parameterize the derivative. We compared the performance of the adjoint method and direct backpropagation on the same task with the same network, trained with the same hyper-parameters.

Results are summarized in Table 1. Direct back-propagation consistently outperformed the adjoint method. This result validates our analysis on the instability of the adjoint method, which is intuitively caused by the instability of the reverse-time ODE. On the other hand, for direct backpropagation, the gradient is always accurate, because  $h(t_{i}) = z(t_{i})$  is guaranteed as in Fig. 2(c).

We also validate our arguments with extra experiments on image classification tasks; results are in appendix C. We observed that the adjoint method generated inferior performance, while using a solver with accurate gradient estimation, our ODE network directly modified from ResNet18 out-performed standard ResNet50 and ResNet101.

# 5.5 MEMORY EFFICIENCY

In this section we demonstrate that our bijective block is memory efficient. We trained a GODE model with bijective blocks, and compared the memory consumption using our memory-efficient function as in Algo. 1 and a memory-inefficient method as in conventional back-propagation. Results were measured with a batchsize of 100 on MUTAG dataset.

<table><tr><td>Model</td><td>ψ</td><td>Cora</td><td>CiteSeer</td><td>PubMed</td></tr><tr><td>GCN</td><td></td><td>81.6±0.5</td><td>71.6±0.3</td><td>79.2±0.1</td></tr><tr><td rowspan="2">GCN-ODE</td><td>additive</td><td>81.7±0.7</td><td>72.4±0.6</td><td>80.0±0.2</td></tr><tr><td>L_sigmoid</td><td>81.8±0.3</td><td>72.4±0.8</td><td>80.1±0.3</td></tr><tr><td>GAT</td><td></td><td>82.9±0.3</td><td>71.7±0.8</td><td>78.9±0.3</td></tr><tr><td rowspan="2">GAT-ODE</td><td>additive</td><td>83.3±0.3</td><td>72.1±0.6</td><td>79.1±0.5</td></tr><tr><td>L_sigmoid</td><td>83.1±0.4</td><td>72.1±0.3</td><td>79.0±0.5</td></tr><tr><td>ChebNet</td><td></td><td>82.1±0.5</td><td>70.8±0.5</td><td>76.6±0.8</td></tr><tr><td rowspan="2">Cheb-ODE</td><td>additive</td><td>82.4±0.5</td><td>71.1±0.5</td><td>77.8±1.2</td></tr><tr><td>L_sigmoid</td><td>82.2±0.4</td><td>70.8±0.6</td><td>77.0±1.1</td></tr></table>

<table><tr><td></td><td>MUTAG</td><td>PROTEIN</td><td>IMDB</td><td>REDDIT</td><td>COLLAB</td></tr><tr><td>GCN</td><td>74.5±6.5</td><td>72.4±3.2</td><td>74.4±3.6</td><td>86.3±3.1</td><td>80.9±1.7</td></tr><tr><td>GCN-ODE</td><td>80.8±8.3</td><td>73.9±3.1</td><td>74.6±5.1</td><td>92.4±2.1</td><td>82.0±2.1</td></tr><tr><td>ChebNet</td><td>82.5±4.8</td><td>70.5±4.8</td><td>73.4±3.7</td><td>91.9±1.6</td><td>81.3±1.9</td></tr><tr><td>Cheb-ODE</td><td>86.7±8.8</td><td>70.0±4.0</td><td>73.6±4.6</td><td>92.1±1.1</td><td>81.0±2.2</td></tr><tr><td>GIN</td><td>84.5±0.5</td><td>73.1±3.9</td><td>73.1±5.1</td><td>88.3±4.0</td><td>81.0±2.4</td></tr><tr><td>GIN-ODE</td><td>89.3±3.7</td><td>72.9±2.8</td><td>73.2±4.4</td><td>90.3±1.8</td><td>81.2±1.7</td></tr></table>

Table 3: Results on node classification tasks. We compared various discrete-layer structures and their corresponding GODE models (marked with ODE). We tested GODE model with different  $\psi$  functions ("l_sigmoid" represents linear_sigmoid).

Table 4: Results on graph-classification tasks. Corresponding GODE models of discrete-layer structures are marked with ODE. Results are reported from a 10-fold cross-validation.

Results are summarized in Table. 2. We measured the memory consumption with different depths, which is the number of ODE blocks. When depth increases from 10 to 20, the memory by conventional methods increases from 5.3G to 10.5G, while our memory-efficient version only increases from 2.2G to 2.6G. In theory, our bijective block takes  $\mathcal{O}(1)$  memory, because we only need to store the outputs in cache, while deleting activations of middle layers. For memory-efficient network, the slightly increased memory consumption is because states of  $F,G$  need to be cached; but this step takes up minimal memory compared to input data.

# 5.6 GENERAL BIJECTIVE BLOCKS

We demonstrate that bijective blocks defined as Eq. 11 can be easily generalized:  $F$  and  $G$  are general neural networks, which can be adapted to different tasks;  $\psi(\alpha, \beta)$  can be any differentiable bijective mapping  $w.r.t. \alpha$  when  $\beta$  is given. We leave the results of different network architectures of  $F, G$  in the next section, and demonstrate a few examples of  $\psi$ : (1) additive, forward is  $\eta = \psi(\alpha, \beta) = \alpha + \beta$ , inverse is  $\alpha = \psi^{-1}(\eta, \beta) = \eta - \beta$ ; (2) linear-sigmoid, forward is  $\eta = \psi(\alpha, \beta) = \alpha \times \operatorname{sigmoid}(\beta)$ , inverse is  $\alpha = \psi^{-1}(\eta, \beta) = \eta / \operatorname{sigmoid}(\beta)$ .

Results for different  $\psi$  are reported in Table 3. Note that we experimented with different depths and reported the best accuracy for each model. All GODE models outperformed their corresponding discrete-layer models, validating the effectiveness of GODE; different  $\psi$  functions behaved similarly on our node classification tasks, indicating the continuous-time model is more important than coupling function  $\psi$ .

# 5.7 RESULTS ON GRAPH CLASSIFICATION TASK

Results for different models on graph classification tasks are summarized in Table 4. We experimented with different structures, including GCN, ChebNet and GIN, and denote the corresponding GODE models with suffix "-ODE". Our results demonstrate that GODE models achieved or outperformed state-of-the-art models. GODE model with GCN structure generated the highest accuracy except on the MUTAG dataset; GODE model with GIN structure generated the highest accuracy on MUTAG. For all experiments, GODE models generated higher or very similar accuracy compared to their discrete-layer counterparts. This indicates the continuous process model might be important for graph models. Furthermore, we notice that GODE with simple GCN structures outperformed networks with complicated structures, such as ChebNet and GIN.

# 6 CONCLUSIONS

We propose GODE, which enables us to model continuous diffusion process on graphs. For efficient training of GODE models, we propose to perform direct back-prop through ODE solvers to accurately determine the gradient; we further modify bijective blocks for GODE, so they can be recursively called at a constant memory cost, and can be applied to large graphs. The proposed GODE is a general framework, which can be used with different GNN structures. We derive the optimization scheme from an optimization perspective in theory, and validate the superior performance of GODE with various experiments.

# REFERENCES

Uri M Ascher, Steven J Ruuth, and Raymond J Spiteri. Implicit-explicit runge-kutta methods for time-dependent partial differential equations. Applied Numerical Mathematics, 25(2-3):151-167, 1997.  
James Atwood and Don Towsley. Diffusion-convolitional neural networks. In NIPS, 2016.  
Peter N Brown, George D Byrne, and Alan C Hindmarsh. Vode: A variable-coefficient ode solver. SIAM, 1989.  
Joan Bruna, Wojciech Zaremba, Arthur Szlam, and Yann LeCun. Spectral networks and locally connected networks on graphs. arXiv preprint arXiv:1312.6203, 2013.  
Bo Chang, Lili Meng, Eldad Haber, Lars Ruthotto, David Begert, and Elliot Holtham. Reversible architectures for arbitrarily deep residual neural networks. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.  
Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in neural information processing systems, pp. 6571-6583, 2018.  
Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems, pp. 3844-3852, 2016.  
Li Deng, Dong Yu, et al. Deep learning: methods and applications. Foundations and Trends® in Signal Processing, 7(3-4):197-387, 2014.  
Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.  
Laurent et al Dinh. Density estimation using real nvp. arXiv, 2016.  
Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. arXiv preprint arXiv:1904.01681, 2019.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems, pp. 2224-2232, 2015.  
Matthias Fey and Jan Eric Lenssen. Fast graph representation learning with pytorch geometric. arXiv preprint arXiv:1903.02428, 2019.  
Alex Fout, Jonathon Byrd, Basir Shariat, and Asa Ben-Hur. Protein interface prediction using graph convolutional networks. In Advances in Neural Information Processing Systems, pp. 6530-6539, 2017.  
Mark I Freidlin and Alexander D Wentzell. Diffusion processes on graphs and the averaging principle. The Annals of probability, 1993.  
Amir Gholami, Kurt Keutzer, and George Biros. Anode: Unconditionally accurate memory-efficient gradients for neural odes. arXiv preprint arXiv:1902.10298, 2019.  
Aidan N Gomez, Mengye Ren, Raquel Urtasun, and Roger B Grosse. The reversible residual network: Backpropagation without storing activations. In Advances in neural information processing systems, pp. 2214-2224, 2017.  
Will Grathwohl, Ricky TQ Chen, Jesse Betterncourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, 2018.  
Eldad Haber and Lars Ruthotto. Stable architectures for deep neural networks. Inverse Problems, 2017.

Timo Hackel, Nikolay Savinov, Lubor Ladicky, Jan D Wegner, Konrad Schindler, and Marc Pollefeys. Semantic3d. net: A new large-scale point cloud classification benchmark. arXiv preprint arXiv:1704.03847, 2017.  
Takuo Hamaguchi, Hidekazu Oiwa, Masashi Shimbo, and Yuji Matsumoto. Knowledge transfer for out-of-knowledge-base entities: A graph neural network approach. arXiv preprint arXiv:1706.05674, 2017.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pp. 1024-1034, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Mikael Henaff, Joan Bruna, and Yann LeCun. Deep convolutional networks on graph-structured data. arXiv preprint arXiv:1506.05163, 2015.  
Aapo Hyvarinen and Erkki Oja. Independent component analysis: algorithms and applications. Neural networks, 13(4-5):411-430, 2000.  
Jörn-Henrik Jacobsen, Arnold Smeulders, and Edouard Oyallon. i-revnet: Deep invertible networks. arXiv preprint arXiv:1802.07088, 2018.  
Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In NIPS, 2018.  
Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in neural information processing systems, pp. 4743-4751, 2016.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Risi Imre Kondor and John Lafferty. Diffusion kernels on graphs and other discrete structures. In ICML, 2002.  
Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493, 2015.  
Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3431-3440, 2015.  
László Lovász et al. Random walks on graphs: A survey. Combinatorics, Paul erdos is eighty, 2(1): 1-46, 1993.  
Yiping et al Lu. Beyond finite layer neural networks: Bridging deep architectures and numerical differential equations. arXiv, 2017.  
William Edmund Milne and WE Milne. Numerical solution of differential equations. 1953.  
Federico et al Monti. Geometric deep learning on graphs and manifolds using mixture model cnns. In CVPR, 2017.  
R-E Plessix. A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. Geophysical Journal International, 167(2):495-503, 2006.  
Lev Semenovich Pontryagin. Mathematical theory of optimal processes. Routledge, 2018.  
Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. arXiv, 2015.

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning internal representations by error propagation. Technical report, California Univ San Diego La Jolla Inst for Cognitive Science, 1985.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61-80, 2008.  
Paul Stapor, Fabian Froehlich, and Jan Hasenauer. Optimization and uncertainty analysis of ode models using second order adjoint sensitivity analysis. *BioRxiv*, pp. 272005, 2018.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pp. 3104-3112, 2014.  
Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018a.  
Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. arXiv preprint arXiv:1806.03536, 2018b.  
Shuicheng Yan, Dong Xu, Benyu Zhang, Hong-Jiang Zhang, Qiang Yang, and Stephen Lin. Graph embedding and extensions: A general framework for dimensionality reduction. IEEE transactions on pattern analysis and machine intelligence, 29(1):40-51, 2006.  
Pinar Yanardag and SVN Vishwanathan. Deep graph kernels. In ACM KDD, 2015.  
Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Graph neural networks: A review of methods and applications. arXiv preprint arXiv:1812.08434, 2018.
