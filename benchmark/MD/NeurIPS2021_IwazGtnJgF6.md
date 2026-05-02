# CBP: backpropagation with constraint on weight precision using a pseudo-Lagrange multiplier method

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Backward propagation of errors (backpropagation) is a method to minimize objective functions (e.g., loss functions) of deep neural networks by identifying optimal sets of weights and biases. Imposing constraints on weight precision is often required to alleviate prohibitive workloads on hardware. Despite the remarkable success of backpropagation, the algorithm itself is not capable of considering such constraints unless additional algorithms are applied simultaneously. To address this issue, we propose the constrained backpropagation (CBP) algorithm based on a pseudo-Lagrange multiplier method to obtain the optimal set of weights that satisfy a given set of constraints. The defining characteristic of the proposed CBP algorithm is the utilization of a Lagrangian function (loss function plus constraint function) as its objective function. We considered various types of constraints — binary, ternary, one-bit shift, and two-bit shift weight constraints. For all cases, the proposed algorithm outperforms the state-of-the-art methods on ImageNet, e.g.,  $66.6\%$  and  $74.4\%$  top-1 accuracy for ResNet-18 and ResNet-50 with binary weights, respectively. This highlights CBP as a learning algorithm to address diverse constraints with the minimal performance loss by employing appropriate constraint functions.

# 1 Introduction

Currently, deep learning-based methods are applied in a variety of tasks, including the classification of static data, e.g., image recognition [1, 2]; classification of dynamic data, e.g., speech recognition [3-6]; and, the approximation of functions, which requires the output of precise predictions, e.g., electronic structure predictions [7] and nonlinear circuit predictions [8]. All of the aforementioned tasks require discriminative models. Additionally, generative models, including generative adversarial networks [9] and variants [10-13], comprise another type of deep neural network. Despite the diversity in application domain and model type used, almost all deep learning-based methods use backpropagation as a common learning algorithm.

Recent developments in deep learning have primarily focused on increasing the size and depth of deep neural networks (DNNs) to improve their learning capabilities, as in the case of state-of-the-art DNNs like VGGNet [14] and ResNet [15]. Given that the memory capacity required by a DNN is proportional to the number of parameters (weights and biases), memory usage for DNN becomes severe. Additionally, a significant number of multiply-accumulate operations during the training and inference stages impose prohibitive workload on the hardware. Thus, efficient hardware-resource consumption is critical to the optimal performance of deep learning. One way to address this requirement is the use of weights of limited precision, such as binary [16, 17] and ternary weights [18, 19]. To this end, particular constraints are applied to weights during training, and additional algorithms for weight quantization are used in conjunction with backpropagation. This is

because such constraints are not considered during the minimization of the objective function (loss function) when backpropagation is executed by itself.

We adopt the Lagrange multiplier method (LMM) to combine basic backpropagation with additional constraint algorithms and produce a single constrained backpropagation (CBP) algorithm. We refer to the adopted method as pseudo-LMM because the constraints functions used  $cs(x)$  are nondifferentiable at  $\boldsymbol{x}_m$  ( $= \arg \min_x cs(x)$ ), rendering the LMM inapplicable. Nevertheless, pseudo-LMM successfully attains the optimal point under particular conditions as for LMM. In the CBP algorithm, the optimal weights satisfying a given set of constraints are evaluated via a basic backpropagation algorithm. It is implemented by simply replacing the conventional objective function (loss function) with a Lagrangian function  $\mathcal{L}$  that comprises the loss and constraint functions as sub-functions that are subjected to simultaneous minimization. Therefore, this method is perfectly compatible with conventional deep learning frameworks. The primary contributions of this study are as follows.

- We introduce a novel and simple method to incorporate given constraints into backpropagation by using a Lagrangian function as the objective function. The proposed method is capable of addressing any set of constraints on the weights, as long as the constraint functions are mathematically well-defined.  
- We introduce pseudo-LMM with constraint functions  $cs(w)$  that are nondifferentiable at  $w_{m} (= \arg \min_{w} cs(w))$  and analyze the kinetics of pseudo-LMM in a continuous time domain.  
- We introduce optimal (sawtooth-like) constraint functions with gradually vanishing unconstrained-weight windows and provide a guiding principle for the stable co-optimization of weights and Lagrange multipliers in a quasi-static fashion.  
- We evaluate the performance of CBP applied to AlexNet, ResNet-18, and ResNet-50 (pre-trained using backpropagation with full-precision weights) with four different constraints (binary, ternary, one-bit shift, and two-bit shift weight constraints) on ImageNet as proof-of-concept examples. The results highlight the classification accuracy outperforming the previous state-of-the-art results.

# 2 Related work

The simplest approach to weight quantization is the quantization of pre-trained weights. Gong et al. [20] proposed several methods for weight quantization and demonstrated that binarizing weights using a sign function degraded the top-1 accuracy on ImageNet by less than  $10\%$ . Mellempudi et al. [21] proposed a fine-grained quantization algorithm that calculates the optimal thresholds for the ternarization of pre-trained weights. The expectation backpropagation algorithm [22] implements a variational Bayesian approach to weight quantization. It uses binary weights and activations during the inference stage. Zhou et al. [23] proposed incremental network quantization (INQ) that iteratively re-trains a group of weights to compensate for the performance loss caused by the rest of weights which are quantized using pre-set quantization thresholds.

Several methods of weight quantization utilize auxiliary real-valued weights in conjunction with quantized weights during training. The straight-through-estimator (STE) comprises the conduction of forward and backpropagation using quantized weights but relies on the auxiliary real-valued weights for the update of weights [3]. BinaryConnect [16] utilizes weights binarized by a sign function for forward and backpropagation, and the real-valued weights are updated via backpropagation with binary weights. Binary-weight-network (BWN) [17] identifies the binary weights closest to the real-valued weights using a scaling factor, and it exhibits a higher classification accuracy than BinaryConnect on ImageNet. Binarized Neural Nets [24] and XNOR-Nets [17] are extensions of BinaryConnect and BWN, respectively, which utilize binary activations alongside binary weights.

Lin et al. [18] proposed TernaryConnect and Ternary-weight-network (TWN), which are similar to BinaryConnect and BWN but use weight-ternarization methods instead. Trained ternary quantization (TTQ) [25] also uses ternary weights that are quantized using trainable thresholds for quantization. LQ-Nets proposed by Zhang et al. [26] utilizes activation- and weight-quantizers considering the actual distributions of full-precision activation and weight, respectively. Qin et al. [27] introduced IR-Nets that feature the use of error decay estimators to approximate sign functions for weight and activation binarization to differentiable forms.

To take into account the constraint on weight precision, Leng et al. [28] used an augmented Lagrangian function as an objective function, which include a constraint-agreement sub-function. The method was successfully applied to various DNNs on ImageNet; yet, the method failed to reach the accuracy level for full-precision models even when 3-bit precision was used.

The CBP algorithm proposed in this study also utilizes LMM; but CBP essentially differs from [28] given that (i) a basic differential multiplier method (BDMM), rather than ADMM, is used to apply various constraints on weight precision, (ii) particularly designed constraint functions with gradually vanishing unconstrained-weight window are used, and (iii) substantial improvement on the classification accuracy on ImageNet is achieved.

# 3 Optimization method

# 3.1 Pseudo-Lagrange multiplier method

LMM calculates the maximum or minimum value of a function  $f$  under the constraint  $cs = 0$ . [29] Let us assume that  $f$  attains its minimum (or maximum) value satisfying the given constraint at  $(x_{m}, y_{m})$ , with  $f(x_{m}, y_{m}) = m$ . Further,  $cs(x_{m}, y_{m}) = 0$  as the constraint is satisfied at this point. In this case, the point of intersection between the graphs of the two functions,  $f(x, y) = m$  and  $cs(x, y) = 0$ , is  $(x_{m}, y_{m})$ . As the two functions have a common tangent at the point of intersection, the following equation holds:

$$
\nabla_ {x, y} f = - \lambda \nabla_ {x, y} c s, \tag {1}
$$

at  $(x_{m},y_{m})$ , where  $\lambda$  is a Lagrange multiplier [30].

The Lagrangian function is defined to be  $\mathcal{L}(x,y,\lambda) = f(x,y) + \lambda cs(x,y)$ . Consider a local point  $(x,y)$  at which the gradient of  $\mathcal{L}(x,y,\lambda)$  is zero. Therefore,  $\nabla_{x,y,\lambda}\mathcal{L}(x,y,\lambda) = \nabla_{x,y}[f(x,y) + \lambda cs(x,y)] + \nabla_{\lambda}[\lambda cs(x,y)]$ . Thus,  $\nabla_{x,y,\lambda}\mathcal{L}(x,y,\lambda) = \mathbf{0}$  is equivalent to the following equations (here,  $\mathbf{0}$  denotes the zero-vector):

$$
\begin{array}{l} \nabla_ {x, y, \lambda} \mathcal {L} (x, y, \lambda) = \mathbf {0} \Leftrightarrow \left\{ \begin{array}{c c} \nabla_ {x, y} [ f (x, y) + \lambda c s (x, y) ] & = \mathbf {0}, \\ \nabla_ {\lambda} [ \lambda c s (x, y) ] & = 0 \end{array} \right. \\ \Leftrightarrow \left\{ \begin{array}{r l} \nabla_ {x, y} f (x, y) & = - \lambda \nabla_ {x, y} c s (x, y) \\ c s (x, y) & = 0 \end{array} \right. \tag {2} \\ \end{array}
$$

This satisfies the condition in Eq. (1) as well as the constraint  $cs(x,y) = 0$ . Therefore, the local point  $(x,y)$  corresponds to the minimum (or maximum) point of the function  $f$  under the condition that the constraint is satisfied.

We define pseudo-LMM to address similar minimization tasks but with constraint functions that are nondifferentiable at  $(x_{m},y_{m}) = \arg \min_{x,y}cs(x,y)$ , where  $cs(x,y)$  is a continuous function of  $x$  and  $y$ . Because such constraint functions are nondifferentiable, Eq. (1) cannot be satisfied at the optimal point. Nevertheless, pseudo-LMM enables us to minimize the function  $f(\pmb {x})$  subject to the constraint condition  $cs(x) = 0$  using the Lagrangian function  $\mathcal{L}(\pmb {x},\lambda)$ .

Definition 3.1 (Pseudo-LMM). Pseudo-LMM is a method to attain the optimal variables  $\pmb{x}_m$  that minimize function  $f(\pmb{x})$  subject to the constraint condition  $cs(\pmb{x}) = 0$ , where the function  $cs(\pmb{x})$  is nondifferentiable at  $\pmb{x}_m$  but reaches the minimum at  $\pmb{x}_m$ , i.e.,  $cs(\pmb{x}_m) = 0$ .

Theorem 3.1. Minimizing the Lagrangian function  $\mathcal{L}(\boldsymbol{x},\lambda)$ , which is given by  $\mathcal{L}(\boldsymbol{x},\lambda) = f(\boldsymbol{x}) + \lambda cs(\boldsymbol{x})$ , is equivalent to minimizing the function  $f(\boldsymbol{x})$  subject to  $cs(\boldsymbol{x}) = 0$

Proof. The Lagrangian function  $\mathcal{L}(\pmb{x},\lambda)$  is always differentiable with respect to the Lagrangian multiplier  $\lambda$ , so that the equation  $cs(\pmb{x}) = 0$  holds at the optimal point  $\pmb{x}_m$ . Thus, we have

$$
\underset {\boldsymbol {x}, \lambda} {\text {m i n i m i z e}} \mathcal {L} (\boldsymbol {x}, \lambda) \Leftrightarrow \left\{ \begin{array}{l l} \underset {\boldsymbol {x}} {\text {m i n i m i z e}} & \mathcal {L} (\boldsymbol {x}; \lambda) \\ \text {s u b j e c t t o} & c s (\boldsymbol {x}) = 0. \end{array} \right. \tag {3}
$$

When the constraint is satisfied, i.e.,  $cs(\pmb{x}) = 0$ , the Lagrangian function  $\mathcal{L}(\pmb{x};\lambda)$  equals the function  $f(\pmb{x})$ , so that the task to minimize  $\mathcal{L}(\pmb{x},\lambda)$  with respect to  $\pmb{x}$  and  $\lambda$  corresponds to the task to minimize  $f(\pmb{x})$  subject to  $cs(\pmb{x}) = 0$ .

Given Theorem 3.1, pseudo-LMM can attain the optimal point by minimizing the Lagrangian function  $\mathcal{L}$  in spite of the nondifferentiability of the constraint function  $cs(\boldsymbol{x}) = 0$  at the optimal point  $\boldsymbol{x}_m$ . Note that not all functions have zero gradients at their minimum points; for instance, the function  $y = |x|$  attains its minimum at  $x = 0$  but the gradient at the minimum point is not defined. However, all convex functions have zero gradients at their minimum points. Thus, LMM to minimize the function  $f(\boldsymbol{x})$  subject to the constraint convex function  $cs(\boldsymbol{x})$  with minimum point  $\boldsymbol{x}_m$  is a subset of pseudo-LMM. In this case, the constraint function has zero gradient at the minimum point, so that Eq. (2) becomes

$$
\nabla_ {\boldsymbol {x}, \lambda} \mathcal {L} (\boldsymbol {x}, \lambda) = \boldsymbol {0} \Leftrightarrow \left\{ \begin{array}{r l} \nabla_ {\boldsymbol {x}} f (\boldsymbol {x}) & = 0 \\ \text {s u b j e c t t o} c s (\boldsymbol {x}) & = 0. \end{array} \right. \tag {4}
$$

We will consider continuous constraint functions with a few nondifferentiable points in their variable domains, including their minimum points. Other than such nondifferentiable points, we will use the gradient descent method to search for the minimum points within the framework of pseudo-LMM. Hereafter, when the gradient of the Lagrangian  $\mathcal{L}$  function is remarked, its variable domain excludes such nondifferentiable points.

The optimal solution to Eq. (4) can be found using a basic differential multiplier method (BDMM) [31] that calculates the point at which  $\mathcal{L}(\pmb{x},\pmb{\lambda})$  attains its minimum value by driving  $\pmb{x}$  toward the constraint subspace  $\bar{x}$  ( $cs(\bar{x}) = 0$ ). The BDMM updates  $\pmb{x}$  and  $\pmb{\lambda}$  according to the following relations.

$$
\boldsymbol {x} \leftarrow \boldsymbol {x} - \eta_ {x} \nabla_ {\boldsymbol {x}} \mathcal {L} (\boldsymbol {x}, \boldsymbol {\lambda}), \tag {5}
$$

and

$$
\boldsymbol {\lambda} \leftarrow \boldsymbol {\lambda} + \eta_ {\lambda} \nabla_ {\boldsymbol {\lambda}} \mathcal {L} (\boldsymbol {x}, \boldsymbol {\lambda}). \tag {6}
$$

BDMM is cheaper than Newton's method in terms of computational cost. Additionally, Eq. (5) is identical to the solution used in the gradient descent method during backpropagation, except for the use of a Lagrangian function instead of a loss function. This indicates the compatibility of LMM with the optimization framework based on backpropagation. Therefore, we choose BDMM to apply LMM to optimize DNNs. The learning kinetics with BDMM is elaborated in Appendix A.1.

# 3.2 Constrained backpropagation using a pseudo-Lagrange multiplier method

We utilize LMM to train DNNs with particular sets of weight-constraints. We define a Lagrangian function  $\mathcal{L}$  in the context of feedforward DNN using the following relation.

$$
\mathcal {L} \left(\boldsymbol {y} ^ {(k)}, \hat {\boldsymbol {y}} ^ {(k)}; \boldsymbol {W}, \boldsymbol {\lambda}\right) = C \left(\boldsymbol {y} ^ {(k)}, \hat {\boldsymbol {y}} ^ {(k)}; \boldsymbol {W}\right) + \boldsymbol {\lambda} ^ {\mathrm {T}} \boldsymbol {c s} (\boldsymbol {W}), \tag {7}
$$

where  $\pmb{y}^{(k)}$  and  $\hat{\pmb{y}}^{(k)}$  denote the actual output vector corresponding to the  $k$ th input data, and its label, respectively.  $\pmb{W}$  denotes a set of weight matrices, including  $n_w$  weights in aggregate, and  $C$  denotes a loss function. The constraint functions  $\pmb{cs} = [cs_1(w_1),\dots,cs_{n_w}(w_{n_w})]^{\mathrm{T}}$  includes components  $c s_i(w_i)$  for all  $i$  ( $1\leq i\leq n_w$ ). Note that  $n_w$  is the total number of weights. Similarly, the Lagrange multipliers  $\lambda = [\lambda_1,\dots,\lambda_{n_w}]^{\mathrm{T}}$  includes components  $\lambda_{i}$  for all  $i$  ( $1\leq i\leq n_w$ ).

We chose sawtooth-shaped constraint functions. We quantize the real-valued weights into  $n_q$  values in the set  $Q = \{q_i\}_{i=1}^{n_q}$ , where  $q_i < q_{i+1}$  for all  $i$ . We also employ a set of the medians of neighboring values in the set  $Q$ :  $M = \{m_i\}_{i=1}^{n_q-1}$ , where  $m_i = (q_i + q_{i+1}) / 2$ . Using  $Q$  and  $M$ , we define a partial constraint function  $y_i$  for  $i = 0, 1 \leq i < n_q$ , and  $i = n_q$ ,

$$
y _ {0} \left(w\right) = \left\{ \begin{array}{c c} - 2 \left(w - q _ {1}\right) & \text {i f} w <   q _ {1}, \\ 0 & \text {o t h e r w i s e}, \end{array} \right.
$$

$$
y _ {i} \left(w\right) = \left\{ \begin{array}{c l} - 2 \left| w - m _ {i} \right| + q _ {i + 1} - q _ {i} & \text {i f} q _ {i} \leq w <   q _ {i + 1}, \\ 0 & \text {o t h e r w i s e}, \end{array} \right.
$$

$$
y _ {n _ {q}} (w) = \left\{ \begin{array}{c c} 2 \left(w - q _ {n _ {q}}\right) & \text {i f} w \geq q _ {n _ {q}}, \\ 0 & \text {o t h e r w i s e}, \end{array} \right. \tag {8}
$$

![](images/2cc2ccc62c901a6dcdb3d7d691cf072bb37ec47380a4c26e462c56e3bce26127.jpg)

![](images/548954f67e912ce05916ebc7af2853d0e102a05c8515a67c62f19c9fbf244789.jpg)

![](images/3f0f1fcaee639580949f268cfca8ead7e10a94226739d64a298183a5c48fe2a3.jpg)  
Figure 1: Binary- and ternary-weight constraint functions for  $g = 2$  and 10. Blue-filled regions indicate unconstrained-weight windows.

![](images/96b7ab8bab11621570dca1d137a634dd61a6664fd6a097c8455e5957338488ba.jpg)

respectively. The constraint function  $cs$  is the summation of the partial constraint functions  $Y(w) = \sum_{i=0}^{n_q} y_i(w)$ , gated by the unconstrained-weight window  $ucs(w)$  parameterized by a variable  $g$ .

$$
c s (w; Q, M, g) = u c s (w) Y (w), \tag {9}
$$

166 where

$$
u c s (w) = 1 - \sum_ {i = 0} ^ {n _ {q} - 1} H \left(\frac {1}{2 g} \left(q _ {i + 1} - q _ {i}\right) - | w - m _ {i} + \epsilon |\right), \tag {10}
$$

where  $\epsilon \rightarrow 0^{+}$ , and  $H$  denotes a Heaviside step function. The function  $ucs(w)$  realizes the unconstrained-weight window as a function of  $g(\geq 1)$ . When  $g = 1$ , the function outputs zero for  $q_{1} \leq w < q_{n_{q}}$ , merely confining  $w$  to the range  $q_{1} \leq w < q_{n_{q}}$  without weight quantization, whereas, when  $g \to \infty$ , the window vanishes, allowing the constraint function to quantize the weight in the entire weight range. Examples of function  $ucs(w)$  are shown in Fig. 1.

The unconstrained-weight window variable  $g$  is initially set to one and updated such that it keeps increasing during training, i.e., the window gradually vanishes. The window gradually vanishing allows sequential weight quantization such that the further the initial weights from their nearest  $q_{i}$ , the later their weights are subject to quantization, which is otherwise subject to simultaneous (abrupt) quantization. It is likely that the further the initial weights from their nearest  $q_{i}$ , the larger the increase in loss function  $C$  when they are quantized. Thus, the sequential quantization from the weights close to their  $q_{i}$  likely avoids an abrupt increase in the loss. Further, while the closer weights are being quantized, the further weights (not subject to quantization yet) are being updated to reduce the loss given the partially quantized weights. This effect will be discussed in Section ?? based on experimental results.

For every training batch, the weights are updated following a method similar to conventional backpropagation. Nevertheless, the use of the Lagrangian function in Eq. (7), rather than a loss function only, as an objective function constitutes a critical difference. The Lagrange multiplier,  $\lambda$ , is subsequently updated using a gradient ascent method, similar to Eq. (6). Updating cross-coupled variables, such as  $W$  and  $\lambda$ , often experiences difficulties in convergence toward the optimal values because of oscillation around the optimal values. A possible solution involves maintaining one variable in a quasi-static condition while reducing the update rate of the other variable. To this end, we significantly reduce the update frequency of the Lagrange multiplier  $\lambda$ .

Weight update: Weight  $W$  is updated once every iteration as for the conventional backpropagation but using the Lagrangian function  $\mathcal{L}$ .

Lagrange multiplier update: Lagrange multiplier  $\lambda$  is conditionally updated once every training epoch. The update is allowed if the summation of all  $\mathcal{L}$  in a given epoch  $(\mathcal{L}_{sum})$  is not smaller than  $\mathcal{L}_{sum}$  for the previous epoch  $(\mathcal{L}_{sum}^{pre})$  or the multiplier  $\lambda$  has not been updated in the past  $p_{max}$  epochs. This achieves the convergence of  $\boldsymbol{W}$  for a given  $\lambda$  in a quasi-static manner.

Unconstrained weight window update: Unconstrained-weight window variable  $g$  is updated on the same condition as for the Lagrange multiplier  $\lambda$ . Unlike weight  $W$  and multiplier  $\lambda$ , the variable  $g$  (initialized to one) constantly increases when updated such that  $\Delta g = 1$  when  $g < 10$ , and  $\Delta g = 10$  otherwise.

The detailed learning algorithm is shown in the pseudocode in Algorithm 1.

Algorithm 1: CBP algorithm.  $N$  denotes the number of training epochs in aggregate.  $M$  denotes the number of mini-batches of the training set  $\mathbf{Tr}$ . The function minibatch  $(\mathbf{Tr})$  samples a mini-batch of training data and their targets from  $\mathbf{Tr}$ . The function model  $(x, W)$  returns the output from the network for a given mini-batch  $\mathbf{x}$ . The function  $\mathrm{clip}(\mathbf{W})$  denotes the clipping weight, and  $\eta_W$  and  $\eta_\lambda$  denote the learning rates of  $\mathbf{W}$  and  $\lambda$ , respectively.

Result: Updated weight matrix W   
Pre-training using conventional backprop;   
Initialization;   
Initial update of  $\lambda$  .   
for epoch  $= 1$  to  $N$  do   
 $\mathcal{L}_{sum}\gets 0;$    
/\* Update of weight W \*/   
for  $i = 1$  to  $M$  do   
 $\pmb{x}^{(i)},\hat{\pmb{y}}^{(i)}\gets$  minibatch(Tr);   
 $\pmb {y}^{(i)}\gets$  model  $(x^{(i)};W)$  ..   
 $\mathcal{L}\leftarrow C\Big(\hat{y}^{(i)},y^{(i)};W\Big) + \lambda^{\mathrm{T}}cs(W;Q,M,g);$ $\mathcal{L}_{sum}\gets \mathcal{L}_{sum} + \mathcal{L};$ $W\gets \mathrm{clip}\bigl (W - \eta_W\nabla_W\mathcal{L}\bigr);$    
end   
/\* Update of window variable g and Lagrange multiplier  $\lambda$  \*/   
 $p\gets p + 1$    
if  $\mathcal{L}_{sum}\geq \mathcal{L}_{sum}^{pre}$  or  $p = p_{max}$  then   
 $g\gets g + \Delta g$ $\lambda \gets \lambda +\eta_{\lambda}cs(W,g)$ $p\gets 0$  ..   
 $\mathcal{L}_{sum}^{pre}\gets \mathcal{L}_{sum}^{max},$    
else   
 $\mathcal{L}_{sum}^{pre}\gets \mathcal{L}_{sum};$    
end

In the following section, we elaborate on binary, ternary, one-bit shift, and two-bit shift weight constraints and the appropriate constraint function in each case.

# 4 Experiments

To evaluate the performance of our algorithm, we trained three models (AlexNet, and ResNet-18 and 50) on the ImageNet dataset [32] with four different weight constraints (binary, ternary, and one-bit, and two-bit shift weight constraints). ImageNet consists of approximately 1.2 million training images and 50 thousands validation images. All training images were pre-processed such that they were randomly cropped and resized to  $224 \times 224$  with mean subtraction and variance division. Additionally, random horizontal flipping and color jittering were applied. For validation, the images were resized

to  $256 \times 256$  and their centers in  $224 \times 224$  were selectively cropped. We evaluated the top-1 and top-5 classification accuracies on the validation set.

We considered binary, ternary, one-bit shift and two-bits shift weight constraints to validate the CBP algorithm as a general weight-quantization framework. For all cases, we introduced layer-wise scaling factors  $a$  such that  $a^{(l)}$  (for the  $l$ th layer) is given by  $a^{(l)} = \| \pmb{W}^{(l)}\|_1 / n^{(l)}$ , where  $\pmb{W}^{(l)}$  and  $n^{(l)}$  denote the weight matrix of the  $l$ th layer and the number of elements of  $\pmb{W}^{(l)}$ , respectively. As for [17] and [19], the weight matrices of the first and last layers were not quantized. The quantized weights employed for each case is elaborated as follows.

Binary-weight constraint: A set of quantized weights  $Q$  is  $\{-a, a\}$ .

The other weight constraints: A set of quantized weights  $Q$  is  $\{0, \pm 2^{-d}a\}_{d=0}^{D}$ , where  $D = 0, 1$  and 2 for the ternary-, one-bit shift, and two-bit shift weight constraints. Each ternary weight needs 2-bit memory while each of one-bit and two-bit shift weight needs 3-bit memory.

We adopted an STE [33] to train the models such that the forward pass is based on quantized weights  $w_{q}$ ,

$$
w _ {q} = q _ {1} + \sum_ {i = 1} ^ {n _ {q} - 1} \left(q _ {i + 1} - q _ {i}\right) \left(\operatorname {s i g n} (w - m _ {i}) + 1\right) / 2,
$$

whereas the backward pass uses the real-valued weights  $w$  under quantization,  $\partial \mathcal{L} / \partial w = \partial \mathcal{L} / \partial w_{q}$ .

For all cases, the DNN was pre-trained using conventional backpropagation with full-precision weights and activations, which was followed by post-learning using CBP. We used the stochastic gradient descent with momentum to minimize the Lagrangian function  $\mathcal{L}$  with respect to  $W$  and Adam [34] to maximize  $\mathcal{L}$  with respect to  $\lambda$ . The initial learning rate for  $\lambda$  and  $p_{max}$  were set to  $10^{-4}$  and 20, respectively. The learning rate for  $W$  decreased to  $10^{-1}$  times when  $g$  reached 20.

By asymptotically minimizing the Lagrangian function  $\mathcal{L}$ , the constraint function  $cs(W)$  approaches 0. The degree of constraint-failure per weight was evaluated based on the constraint-failure score  $(CFS)$ , which is defined as

$$
C F S = \frac {1}{n _ {w}} \sum_ {i = 1} ^ {n _ {w}} Y _ {i} \left(w _ {i}; Q, M\right), \tag {11}
$$

where  $n_w$  denotes the total number of weights. The CBP algorithm was implemented in Python on a workstation (CPU: Intel Xeon Silver 4110 2.10GHz, GPU: Titan RTX).

It should be noted that we used CBP as a post-training method, so that the random seed effect came into play only in the mini-batch organization. The accuracy deviation is consequently marginal as shown in the case of ResNet-18 with binary weights in Table 2. Therefore, we omit the statistical analyses on accuracies hereafter.

# 4.1 AlexNet

AlexNet is a simple convolutional networks which consists of five convolutional layers and three fully-connected layers [2]. We used AlexNet with batch normalization [35] as in [17, 19, 28]. The initial learning rate for  $W$  was set to  $10^{-3}$  for binary- and ternary-weight constraints and  $10^{-4}$  for the other cases. Batch size was set to 256. We used a weight decay rate (L2-regularization) of  $5 \times 10^{-4}$ .

The CBP algorithm exhibited the state-of-the-art results as listed in Table. 1. The detailed behaviors of networks with binary and ternary weight constraints are addressed in Appendix B. The behaviors highlight asymptotic increases in the top-1 and top-5 recognition accuracy with asymptotic decrease in  $CFS$ . Consequently, the weight distribution bifurcates asymptotically, fulfilling the constraints imposed on the weights.

# 4.2 ResNet-18 and ResNet-50

We also evaluated our algorithm on ResNet-18 and ResNet-50 [15] which were pre-trained using conventional backpropagation. For ResNet-18, the initial learning rate for  $W$  was set to  $10^{-3}$  for all constraint cases. The batch size was 256. For ReNet-50, the initial learning rate for  $W$  was set to  $10^{-3}$  for binary- and ternary-weight constraints and  $10^{-4}$  for the other cases. The batch size was set

Table 1: Top-1/Top-5 accuracy of AlexNet on ImageNet  

<table><tr><td>Algorithm</td><td>Binary</td><td>Ternary</td><td>One-bit shift</td><td>Two-bit shift</td><td>Full-precision</td></tr><tr><td>BWN [17]</td><td>56.8%/79.4%</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>ADMM [28]</td><td>57.0%/79.7%</td><td>58.2%/80.6%</td><td>59.2%/81.8%</td><td>60.0%/82.2%</td><td></td></tr><tr><td>LQ-Nets [26]</td><td>-</td><td>60.5%/82.7%</td><td>-</td><td>-</td><td>60.0%/82.4%</td></tr><tr><td>TTQ [25]</td><td>-</td><td>57.5%/79.7%</td><td>-</td><td>-</td><td></td></tr><tr><td>CBP</td><td>58.0%/80.6%</td><td>58.8%/81.2%</td><td>60.8%/82.6%</td><td>60.9%/82.8%</td><td></td></tr></table>

to 128. The weight decay rate (L2-regularization) was set to  $10^{-4}$  for both ResNet-18 and ResNet-50. The results are summarized in Tables 2 and 3, highlighting the state-of-the-art performance compared with previous results. Notably, CBP with 1-bit shift and 2-bit shift weight constraints almost reaches the performance of full-precision networks. Particularly, for ResNet-50, CBP with the binary-weight constraint significantly outperforms other methods. The detailed behaviors of weight quantizations for ResNet-18 and ResNet-50 are addressed in Appendix B.

Table 2: Top-1/Top-5 accuracy of ResNet-18 on ImageNet  

<table><tr><td>Algorithm</td><td>Binary</td><td>Ternary</td><td>One-bit shift</td><td>Two-bit shift</td><td>Full-precision</td></tr><tr><td>BWN [17]</td><td>60.8%/83.0%</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>TWN [19]</td><td>-</td><td>61.8%/84.2%</td><td>-</td><td>-</td><td></td></tr><tr><td>INQ [23]</td><td>-</td><td>66.0%/87.1%</td><td>-</td><td>68.1%/88.4%</td><td></td></tr><tr><td>ADMM [28]</td><td>64.8%/86.2%</td><td>67.0%/87.5%</td><td>67.5%/87.9%</td><td>68.1%/88.3%</td><td>69.6%/89.2%</td></tr><tr><td>IR-Nets [27]</td><td>66.5%/86.8%</td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>LQ-Nets [26]</td><td>-</td><td>68.0%/88.0%</td><td>-</td><td>69.3%/88.3%</td><td></td></tr><tr><td>TTQ [25]</td><td>-</td><td>66.6%/87.2%</td><td>-</td><td>-</td><td></td></tr><tr><td>CBP</td><td>66.6 ± 0.1%/87.1 ± 0.1%</td><td>68.9%/88.5%</td><td>69.4%/88.8%</td><td>69.4%/88.8%</td><td></td></tr></table>

Table 3: Top-1/Top-5 accuracy of ResNet-50 on ImageNet  

<table><tr><td>Algorithm</td><td>Binary</td><td>Ternary</td><td>One-bit shift</td><td>Two-bit shift</td><td>Full-precision</td></tr><tr><td>BWN [17]</td><td>63.9%/85.1%</td><td>-</td><td>-</td><td>-</td><td rowspan="4">76.0%/93.0%</td></tr><tr><td>TWN [19]</td><td>-</td><td>65.6%/86.5%</td><td>-</td><td>-</td></tr><tr><td>ADMM [28]</td><td>68.7%/88.6%</td><td>72.5%/90.7%</td><td>73.9%/91.5%</td><td>74.0%/91.6%</td></tr><tr><td>CBP</td><td>74.4%/92.1%</td><td>75.0%/92.5%</td><td>76.0%/92.9%</td><td>75.9%/92.9%</td></tr></table>

# 5 Discussion

To evaluate the effect of the constraint function on training performance, we considered three different cases of post-training a DNN using CBP (i) with the unconstrained-weight window, (ii) without the unconstrained-weight window, and (iii) without the constraint function, i.e., conventional backpropagation with STE only. Because all DNNs included the STE, the comparison between these three cases highlights the effect of the constraint function in addition to the STE. For all cases, pre-training using conventional backpropagation preceded the three different post-training schemes. Table 4 addresses the comparison, highlighting accuracy and  $CFS$  improvement in Case (i) over Case (iii) backpropagation with STE. This indicates that CBP allows the DNN to learn features while the weights are being quantized by the constraint function with the gradually vanishing unconstrained-weight window. On the contrary, CBP without unconstrained-weight window (Case (ii)) rather degraded the accuracy compared with Case (iii), whereas the improvement on  $CFS$  was significant. This may be because the constraint function without unconstrained-weight window strongly forced the weights to be quantized without learning the features. The kinetics of Cases (i) and (ii) in continuous time domains is analyzed in Appendix A.2.

We used CBP as a post-training method in this work. That is, the networks considered were pretrained using conventional backpropagation. Applying CBP to untrained networks hardly reached the accuracies of classification listed in Tables 1, 2, and 3. When efficiency in training is of the most important concern, CBP may not be the best choice. However, when efficiency in memory usage is of the most important concern, CBP may be the optimal choice with regard to its excellent learning capability with maximum 3-bit weight precision, which almost reaches the classification

accuracy of full-precision networks. Additionally, the use of one-bit or two-bit shift weights can avoid multiplication operations that consume a considerable amount of power, so that it can significantly improve calculation efficiency.

Table 4: Top-1 accuracy of ResNet-18 trained in various conditions  

<table><tr><td>Post-training algorithm</td><td>Accuracy</td><td>CFS</td></tr><tr><td>CBP with update of g</td><td>66.6%/87.1%</td><td>1.19 ×10-3</td></tr><tr><td>CBP without unconstrained-weight window</td><td>60.2%/82.7%</td><td>1.05×10-5</td></tr><tr><td>Backpropagation+STE</td><td>64.6%/85.9%</td><td>3.58×10-2</td></tr></table>

# 6 Conclusion

In this study, we proposed the CBP algorithm that trains DNNs by simultaneously considering both loss and constraint functions. It enables the implementation of any well-defined set of constraints on weights in a common training framework, unlike previous algorithms for weight quantization, which were tailored to particular constraints. Evaluation of the proposed algorithm on ImageNet with different constraint functions (binary, ternary, one-bit shift and two-bit shift weight constraints) demonstrated its high capability, highlighting its state-of-the-art accuracy of classification.

# References

[1] Y. Taigman, M. Yang, M. Ranzato, and L. Wolf, “Deepface: Closing the gap to human-level performance in face verification,” in 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 1701–1708.  
[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in Advances in Neural Information Processing Systems 25, 2012, pp. 1097-1105.  
[3] G. Hinton, L. Deng, D. Yu, G. E. Dahl, A.-r. Mohamed, N. Jaitly, A. Senior, V. Vanhoucke, P. Nguyen, T. N. Sainath et al., "Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups," IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 82-97, 2012.  
[4] T. N. Sainath, A. Mohamed, B. Kingsbury, and B. Ramabhadran, “Deep convolutional neural networks for LVCSR,” in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, 2013, pp. 8614–8618.  
[5] G. E. Dahl, D. Yu, L. Deng, and A. Acero, “Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition,” IEEE Transactions on Audio, Speech, and Language Processing, vol. 20, no. 1, pp. 30–42, 2012.  
[6] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.  
[7] K. Lee, D. Yoo, W. Jeong, and S. Han, “SIMPLE-NN: an efficient package for training and executing neural-network interatomic potentials,” Computer Physics Communications, vol. 242, pp. 95–103, 2019.  
[8] G. Kim, V. Kornijcuk, D. Kim, I. Kim, C. S. Hwang, and D. S. Jeong, "Artificial neural network for response inference of a nonvolatile resistance-switch array," Micromachines, vol. 10, no. 4, p. 219, 2019.  
[9] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in Advances in Neural Information Processing Systems 27, 2014, pp. 2672-2680.  
[10] A. Radford, L. Metz, and S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," 2015, arXiv:1511.06434.

[11] L. Metz, B. Poole, D. Pfau, and J. Sohl-Dickstein, "Unrolled generative adversarial networks," 2016, arXiv:1611.02163.  
[12] X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, "InfoGAN: interpretable representation learning by information maximizing generative adversarial nets," in Advances in Neural Information Processing Systems 29, 2016, pp. 2172-2180.  
[13] M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein GAN," 2017, arXiv:1701.07875.  
[14] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," 2014, arXiv:1409.1556.  
[15] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in The IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
[16] M. Courbariaux, Y. Bengio, and J.-P. David, "BinaryConnect: training deep neural networks with binary weights during propagations," in Advances in Neural Information Processing Systems 28, 2015, pp. 3123-3131.  
[17] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “XNOR-Net:Imagenet classification using binary convolutional neural networks,” in European Conference on Computer Vision, 2016, pp. 525–542.  
[18] Z. Lin, M. Courbariaux, R. Memisevic, and Y. Bengio, “Neural networks with few multiplications,” 2015, arXiv:1510.03009.  
[19] F. Li, B. Zhang, and B. Liu, "Ternary weight networks," 2016, arXiv:1605.04711.  
[20] Y. Gong, L. Liu, M. Yang, and L. Bourdev, "Compressing deep convolutional networks using vector quantization," 2014, arXiv:1412.6115.  
[21] N. Mellempudi, A. Kundu, D. Mudigere, D. Das, B. Kaul, and P. Dubey, "Ternary neural networks with fine-grained quantization," 2017, arXiv:1705.01462.  
[22] D. Soudry, I. Hubara, and R. Meir, "Expectation backpropagation: Parameter-free training of multilayer neural networks with continuous or discrete weights," in Advances in Neural Information Processing Systems 27, 2014, pp. 963-971.  
[23] A. Zhou, A. Yao, Y. Guo, L. Xu, and Y. Chen, "Incremental network quantization: Towards lossless cnns with low-precision weights," arXiv preprint arXiv:1702.03044, 2017.  
[24] M. Courbariaux, I. Hubara, D. Soudry, R. El-Yaniv, and Y. Bengio, "Binarized neural networks: Training deep neural networks with weights and activations constrained to +1 or -1," 2016, arXiv:1602.02830.  
[25] C. Zhu, S. Han, H. Mao, and W. J. Dally, "Trained ternary quantization," arXiv preprint arXiv:1612.01064, 2016.  
[26] D. Zhang, J. Yang, D. Ye, and G. Hua, "Lq-nets: Learned quantization for highly accurate and compact deep neural networks," in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 365-382.  
[27] H. Qin, R. Gong, X. Liu, M. Shen, Z. Wei, F. Yu, and J. Song, “Forward and backward information retention for accurate binary neural networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 2250–2259.  
[28] C. Leng, Z. Dou, H. Li, S. Zhu, and R. Jin, "Extremely low bit neural network: Squeeze the last bit out with admm," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.  
[29] D. Bertsekas and W. Rheinboldt, Constrained Optimization and Lagrange Multiplier Methods, ser. Computer science and applied mathematics. Elsevier Science, 2014.  
[30] D. Luenberger and Y. Ye, Linear and Nonlinear Programming. Springer, 2015.

[31] J. C. Platt and A. H. Barr, “Constrained differential optimization,” in Proceedings of the 1987 International Conference on Neural Information Processing Systems, 1987, p. 612–621.  
[32] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and L. Fei-Fei, "ImageNet Large Scale Visual Recognition Challenge," International Journal of Computer Vision (IJCV), vol. 115, no. 3, pp. 211-252, 2015.  
[33] Y. Bengio, N. Léonard, and A. Courville, "Estimating or propagating gradients through stochastic neurons for conditional computation," arXiv preprint arXiv:1308.3432, 2013.  
[34] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," 2014, arXiv:1412.6980.  
[35] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in International conference on machine learning. PMLR, 2015, pp. 448-456.
