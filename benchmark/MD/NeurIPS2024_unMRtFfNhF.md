# Data Debugging is NP-hard for Classifiers Trained with SGD

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Data debugging is to find a subset of the training data such that the model obtained by retraining on the subset has a better accuracy. A bunch of heuristic approaches are proposed, however, none of them are guaranteed to solve this problem effectively. This leaves an open issue whether there exists an efficient algorithm to find the subset such that the model obtained by retraining on it has a better accuracy. To answer this open question and provide theoretical basis for further study on developing better algorithms for data debugging, we investigate the computational complexity of the problem named DEBUGGABLE. Given a machine learning model  $\mathcal{M}$  obtained by training on dataset  $D$  and a test instance  $(\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$  where  $\mathcal{M}(\mathbf{x}_{\mathrm{test}})\neq y_{\mathrm{test}}$ , DEBUGGABLE is to determine whether there exists a subset  $D^{\prime}$  of  $D$  such that the model  $\mathcal{M}'$  obtained by retraining on  $D^{\prime}$  satisfies  $\mathcal{M}'(\mathbf{x}_{\mathrm{test}}) = y_{\mathrm{test}}$ . To cover a wide range of commonly used models, we take SGD-trained linear classifier as the model and derive the following main results. (1) If the loss function and the dimension of the model are not fixed, DEBUGGABLE is NP-complete regardless of the training order in which all the training samples are processed during SGD. (2) For hinge-like loss functions, a comprehensive analysis on the computational complexity of DEBUGGABLE is provided; (3) If the loss function is a linear function, DEBUGGABLE can be solved in linear time, that is, data debugging can be solved easily in this case. These results not only highlight the limitations of current approaches but also offer new insights into data debugging.

# 1 Introduction

Given a machine learning model, data debugging is to find a subset of the training data such that the model will have a better accuracy if retrained on that subset [1]. Data debugging serves as a popular method of both data cleaning and machine learning interpretation. In the context of data cleaning, data debugging (a.k.a. training data debugging [2] or data cleansing [1]) can be used to improve the quality of the training data by removing the flaws leading to mispredictions [3-5]. When it comes to ML interpretation, data debugging locates the part of the training data responsible for unexpected predictions of an ML model. Therefore it is also studied as a training data-based (a.k.a. instance-based [6]) interpretation, which is crucial for helping system developers and ML practitioners to debug ML system by reporting the harmful part of training data [7].

To solve the data debugging problem, existing researches adopt a two-phase score-based heuristic approach [2]. In the first phase, a score representing the estimated impact on the model accuracy is assigned to each training sample in the training data. It is hoped that the harmful part of training data gets a lower score than the other part. In the second phase, training samples with lower scores are removed greedily and the model is retrained on the modified training data. The two phases are carried out iteratively until a well-trained model is obtained. Most of the related works focus on

developing algorithms to estimate the scores efficiently in the first phase [8-16], but rarely study the effectiveness of the entire two-phase approach.

Since it is computationally intractable to estimate the score for all possible subsets of the training data, it is often assumed that the score representing the impact of a subset is approximately equal to the sum of the scores of each individual training samples from the subset. However, Koh et. al. [10] showed this is not always the case. For a bunch of subsets sampled from the training data, they empirically studied the difference between the estimated impact and the actual impact of each subset by taking influence functions as the scoring method. The estimated impact is calculated by summing up the score by influence function of each training samples in the subset, and the actual impact is measured by the improvement of accuracy of the model retrained after removing the subset from training data. They found that the estimated impact tends to underestimate the actual impact. Removing a large number of training samples could result in a large deviation between estimated and actual impacts. Although an upper bound of the deviation under certain assumptions has been derived, it is still unknown whether the deviation can be reduced or eliminated efficiently.

The above deviation also poses challenges to the effectiveness of the entire approach. Suppose the influence function is adopted as the scoring method, the accuracy of the model is not guaranteed to improve due to the deviation reported in [10] if a large group of training samples are removed during each iteration. Moreover, there is no theoretical analysis for the effectiveness of the greedy approach in the second phase. Even if only one training sample is removed during each iteration of the two-phase approach, the accuracy of the model is still not guaranteed to be improved. The effectiveness of the entire two-phase approach is therefore not assured. This leaves the following open problem:

Problem 1.1. Is there an efficient algorithm to find the subset of the training data, such that the model obtained by retraining on it has a better accuracy?

The computational complexity results presented in this paper demonstrate that it is unlikely to solve the data debugging problem efficiently in polynomial time. To figure out its hardness, we study the problem DEBUGGABLE which is the decision version of data debugging when the test set consists of only one instance. Formally, DEBUGGABLE is defined as follows:

Problem 1.2 (DEBUGGABLE). Given a classifier  $\mathcal{M}$ , its training data  $T$ , a test instance  $(\mathbf{x},y)$ . Is there a  $T' \subseteq T$ , such that  $\mathcal{M}$  predicts  $y$  on  $\mathbf{x}$  if retrained on  $T'$ ?

Basically, we prove that DEBUGGABLE is NP-complete, which means data debugging is unlikely to be solved in polynomial time. This result answers the open question mentioned above directly, this is, the large deviation of estimated impacts [10] cannot be reduced or eliminated efficiently. This is because if the impact of a subset of the training data could be accurately estimated as the sum of the impact of each training sample in the subset, data debugging can be solved in polynomial time, which is impossible unless  $\mathrm{P = NP}$ .

Although DEBUGGABLE is generally intractable, we still hope to develop efficient algorithms tailored to specific cases. Thus it is necessary to figure out the root cause of the hardness for DEBUGGABLE. Previous research are always conducted based on the belief that the complexity of data debugging is due to the chosen model architecture is complicated. However, we show that at least for models trained by stochastic gradient descent (SGD), the hardness stems from the hyper-parameter configuration selected for the SGD training, which was not yet aware of by previous work. To cover a wide range of commonly used machine learning models, we take linear classifiers as the model and show that even for linear classifiers, DEBUGGABLE is NP-hard as long as they are trained by SGD. Moreover, we provided a comprehensive analysis on hyper-parameter configurations that affect the computational complexity of DEBUGGABLE, including the loss function, the model dimension and the training order. Training order, a.k.a. training data order [17] or order of training samples [18], refers to the order in which each training sample is considered during the SGD. Detailed complexity results are shown in Table 1.

Our contribution can be concluded as follows:

- We studied the computational complexity of data debugging and showed that data debugging is NP-hard for linear classifiers in the general setting for all possible training orders.  
- We studied the complexity of DEBUGGABLE when the loss is fixed as the hinge-like function. For 2 or higher dimension, DEBUGGABLE is NP-complete when the training order

Table 1: Computational complexity of the data debugging problem  

<table><tr><td>Loss Function</td><td>Dimension</td><td>Training Order</td><td>Complexity</td></tr><tr><td>Not Fixed</td><td>Not Fixed</td><td>-</td><td>NP-hard</td></tr><tr><td>Hinge-like</td><td>≥2</td><td>Adversarily Chosen</td><td>NP-hard</td></tr><tr><td>Hinge-like, β&lt;0</td><td>1</td><td>Adversarily Chosen</td><td>NP-hard</td></tr><tr><td>Hinge-like, β≥0</td><td>1</td><td>-</td><td>Linear Time</td></tr><tr><td>Linear</td><td>-</td><td>-</td><td>Linear Time</td></tr></table>

is adversarially chosen; For one-dimensional cases, DEBUGGABLE can be NP-hard when the interception  $\beta < 0$ , and is solvable in linear time when  $\beta \geq 0$ .

- We proved that DEBUGGABLE is solvable in linear time when the loss function is linear.

Moreover, we have a discussion on the implications of these complexity results for machine learning interpretability and data quality, as well as limitations of score-based greedy methods. Our results suggest the further study as follows. (1) It is better to characterize the training sample and find the criterion which can be used to decide the existence of efficient algorithms; (2) Designing algorithms with CSP-solver is a potential way to solve data debugging more efficiently than the brute-force one; (3) Developing random algorithms is a potential way to solve data debugging successfully with high probability.

# 1.1 Related Works

The solution of data debugging has applications in database query results reliability enhancement [2, 19], training data cleaning [1] and machine learning interpretation[9, 8, 10, 20, 21]. Existing works on data debugging mainly adopt a two-phase approach, which scores the training samples in the first phase and greedily deletes training samples with lower scores in the second phase. Most of the research focus on the first phase. There are mainly two ways of scoring adopted for data debugging in practice. Leave-one-out (LOO) retraining is a widely studied way, which evaluates the contribution of a training sample through the difference in the model's accuracy trained without that training sample. To avoid the cost of model retraining, Koh and Liang took influence functions as an approximation of LOO [8]. After that, various extensions and improvements of the influence function based method are proposed, such as Fisher kernel [9], influence function for group impacts [10], second-order approximations [11] and scalable influence functions [12]. Another way is Shapley-based scoring, where the impact of a training sample is measured by its average marginal contribution to all subsets of the training data [13]. Since Shapley-base scoring suffers from expensive computational cost [22], recent works focus on techniques that efficiently estimate the Shapley value, including Monte-Carlo sampling [13], group testing [14, 15] and using proxy models such as  $k$ -NN [16, 3]. However, those methods do not admit any theoretical guarantee on the effectiveness. This paper discusses the limitations of the above methods and suggests some future directions on data debugging.

# 2 Preliminaries and Problem Definition

Linear classifiers. Formally, a (binary) linear classifier is a function  $\lambda_{\mathbf{w}}: \mathbb{R}^d \to \{-1, 1\}$ , where  $d$  is called its dimension and  $\mathbf{w} \in \mathbb{R}^d$  its parameter. Without loss of generality, the bias term of a linear classifier is set as zero in this paper. All vectors in this paper are assumed to be column vectors. For an input  $\mathbf{x}$ , the value of  $\lambda_{\mathbf{w}}$  is defined as

$$
\lambda_ {\mathbf {w}} (\mathbf {x}) = \left\{ \begin{array}{l l} 1 & \text {i f} \mathbf {w} ^ {\top} \mathbf {x} \geq 0 \\ - 1 & \text {o t h e r w i s e .} \end{array} \right.
$$

We denote the class of linear models as  $\Lambda$ .

Training data. A training sample is a pair  $(\mathbf{x},y)$  in which  $\mathbf{x} \in \mathbb{R}^d$  is the input and  $y \in \{-1,1\}$  is the label of  $\mathbf{x}$ . The training data is a multiset of training samples. We employ  $\mathbf{w} \xrightarrow{T} \mathbf{w}'$  to denote that the parameter  $\mathbf{w}'$  is obtained by training the parameter  $\mathbf{w}$  on the training data  $T$ , and employ  $\mathbf{w} \xrightarrow{(\mathbf{x},y)} \mathbf{w}'$  to denote that  $\mathbf{w}'$  is obtained by training  $\mathbf{w}$  on the training sample  $(\mathbf{x},y)$ .

Loss functions and learning rates. Binary linear classifiers typically use unary functions on  $y\mathbf{w}^{\top}\mathbf{x}$  as their loss functions [23]. Therefore we only consider loss functions of the form  $\mathcal{L}:y\mathbf{w}^{\top}\mathbf{x}\mapsto \mathbb{R}$  for the rest of the paper.

The linear loss is in the form of

$$
\mathcal {L} _ {\mathrm {l i n}} (y \mathbf {w} ^ {\top} \mathbf {x}) = - \alpha (y \mathbf {w} ^ {\top} \mathbf {x} + \beta).
$$

The hinge-like loss function is defined as the following form

$$
\mathcal {L} _ {\text {h i n g e}} (y \mathbf {w} ^ {\top} \mathbf {x}) = \left\{ \begin{array}{l l} - \alpha (y \mathbf {w} ^ {\top} \mathbf {x} + \beta), & y \mathbf {w} ^ {\top} \mathbf {x} <   \beta \\ 0, & \text {o t h e r w i s e .} \end{array} \right.
$$

We call  $\beta$  as the interception of  $\mathcal{L}_{\mathrm{hinge}}$ . We represent the learning rate of a model using a vector  $\eta = (\eta_{1},\dots,\eta_{d})$ , where  $\eta_{i}\geq 0$  and each parameter  $w_{i}$  can be updated with the corresponding learning rate  $\eta_{i}$ .

Stochastic gradient descent. The stochastic gradient descent (SGD) method updates parameter  $\mathbf{w}$  from its initial value  $\mathbf{w}^{(0)}$  through several epochs. During each epoch, the SGD goes through the entire set of training samples in some training order through several iterations. The training order is defined as a sequence of training samples, in the form of  $(\mathbf{x}_1, y_1) \dots (\mathbf{x}_n, y_n)$ . For  $1 \leq i < j \leq n$ ,  $(\mathbf{x}_i, y_i)$  is considered before  $(\mathbf{x}_j, y_j)$  during the SGD. We use  $w_i$  to denote the  $i$ -th coordinate of  $\mathbf{w}$ . We also use  $\mathbf{w}^{(e,k)}$  to denote the value of  $\mathbf{w}$  at the end of  $k$ -th iteration of epoch  $e$  and use  $\mathbf{w}^{(e)}$  to denote the value of  $\mathbf{w}$  after the end of epoch  $e$ . Assuming  $(\mathbf{x}, y)$  to be the training sample considered at iteration  $k$ , the stochastic gradient descent (SGD) method updates parameter  $w_i$  for each  $i$  by

$$
w _ {i} ^ {(e, k)} \leftarrow w _ {i} ^ {(e, k - 1)} - \eta_ {i} \cdot \frac {\partial \mathcal {L} \left(y \left(\mathbf {w} ^ {(e , k - 1)}\right) ^ {\top} \mathbf {x}\right)}{\partial w _ {i}} \tag {1}
$$

In other words, we have

$$
\mathbf {w} ^ {(e, k)} \leftarrow \mathbf {w} ^ {(e, k - 1)} - \boldsymbol {\eta} \otimes \nabla \mathcal {L} (y (\mathbf {w} ^ {(e, k - 1)}) ^ {\top} \mathbf {x})
$$

where  $\pmb{\eta} \otimes \nabla \mathcal{L} = (\eta_1 \frac{\partial \mathcal{L}}{\partial w_1}, \dots, \eta_d \frac{\partial \mathcal{L}}{\partial w_d})$  is the Hadamard product. We say a training sample  $\mathbf{x}$  is activated at iteration  $k$  during epoch  $e$  if  $\nabla \mathcal{L}(y(\mathbf{w}^{(e,k-1)})^\top \mathbf{x}) \neq 0$ . The SGD terminates at the end of epoch  $e$  if  $\|\mathbf{w}^{(e-1)} - \mathbf{w}^{(e)}\| < \varepsilon$  for threshold  $\varepsilon$  or  $e$  reached some predetermined value. We denote  $\mathbf{w}^* = \mathbf{w}^{(e)}$ . A linear classifier trained by SGD with the meta-parameters mentioned above is denoted as  $\mathsf{SGD}_{\Lambda}(\mathcal{L}, \boldsymbol{\eta}, \varepsilon, T) = \lambda_{\mathbf{w}^*}$ . With a slight abuse of notation, we define  $\mathsf{SGD}_{\Lambda}(\mathcal{L}, \boldsymbol{\eta}, \varepsilon, T, \mathbf{x}) = \lambda_{\mathbf{w}^*}(\mathbf{x})$ . We also use  $\mathsf{SGD}_{\Lambda}(T, \mathbf{x})$  to avoid cluttering when the context is clear.

Problem definition. With the above definitions, DEBUGGABLE for SGD-trained linear classifiers can be formalized as follows:

# DEBUGGABLE-LIN

Input: Training data  $T$ , loss function  $\mathcal{L}$ , initial parameter  $\mathbf{w}^{(0)}$ , learning rate  $\pmb{\eta}$ , threshold  $\varepsilon$  and instance  $(\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$ .

Output: "Yes": if  $\exists \Delta \subseteq T$  such that  $\mathsf{SGD}_{\Lambda}(\mathcal{L},\boldsymbol {\eta},\varepsilon ,T\setminus \Delta ,\mathbf{x}_{\mathrm{test}}) = y_{\mathrm{test}};$  "No": otherwise.

We say  $\mathsf{SGD}_{\Lambda}(\mathcal{L},\boldsymbol {\eta},\varepsilon ,T)$  is debuggable on  $(\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$  if  $(\mathcal{L},\mathbf{w}^{(0)},\boldsymbol {\eta},\varepsilon ,T,\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$  is a yes-instance of DEBUGGABLE-LIN, and not debuggable on  $(\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$  otherwise.

# 3 Results for Unfixed Loss Functions

In this section, we prove the NP-hardness of DEBUGGABLE-LIN. Intuitively, DEBUGGABLE-LIN is to determine whether there exists a subset  $T' \subseteq T$  where activated training samples within  $T'$  drive the parameter  $\mathbf{w}$  toward the region defined by  $y_{\mathrm{test}} \mathbf{w}^\top \mathbf{x}_{\mathrm{test}} > 0$ . The activation of training samples depends on the complex interaction between the training data and the model.

Theorem 3.1. DEBUGGABLE-LIN is NP-hard for all training orders.

We only show the proof sketch and leave the details in the appendix.

MONOTONE 1-IN-3 SAT

Input: A 3-CNF formula  $\varphi$  with no negation signs.

Output:"Yes": if  $\varphi$  has a 1-in-3 assignment, under which each clause contains exactly one true literal;

"No": otherwise.

For example,  $\varphi_{1} = (x_{1}\lor x_{2}\lor x_{3})\land (x_{2}\lor x_{3}\lor x_{4})$  is a yes-instance because  $(x_{1},x_{2},x_{3},x_{4}) =$  (T,F,F,T) is an 1-in-3 assignment;  $\varphi_{2} = (x_{1}\lor x_{2}\lor x_{3})\land (x_{2}\lor x_{3}\lor x_{4})\land (x_{1}\lor x_{2}\lor x_{4})\land (x_{1}\lor x_{3}\lor x_{4})$  is a no-instance.

Given a 3-CNF formula  $\varphi$ , our goal is to construct a configuration of the training process, such that the resulting model outputs the correct answer if and only if its training data  $T'$  encodes an 1-in-3 assignment  $\nu$  of  $\varphi$ . This can be done by carefully designing the encoding so that for each  $x_i \in \varphi$ ,  $\nu(x_i) = \mathrm{TRUE}$  if and only if  $\mathbf{t}_{x_i} \in T'$ . Finally, we can construct some  $T$  with  $T \supseteq T' \cup \{\mathbf{t}_{x_i} | x_i \in \varphi\}$ , such that some classifier trained on  $T$  is a yes-instance of DEBUGGABLE-LIN if and only if  $\varphi$  is a yes-instance of MONOTONE 1-IN-3 SAT, thereby finishing our proof.

The reduction. Suppose  $\varphi$  has  $m$  clauses and  $n$  variables, let  $N = n + 2m + 1$ . We set the dimension of the linear classifier to  $N$ .

The input. Each coordinate of the input is named as

$$
\mathbf {x} = \left(x _ {c _ {1}}, \dots , x _ {c _ {m}}, x _ {x _ {1}}, \dots , x _ {x _ {n}}, x _ {b _ {1}}, \dots , x _ {b _ {m}}, x _ {\text {d u m m y}}\right) ^ {\top}
$$

We also use  $x_{i}$  to denote the  $i$ -th coordinate of  $\mathbf{x}$ .

The parameters. Each coordinate of the parameter is named as

$$
\mathbf {w} = \left(w _ {c _ {1}}, \dots , w _ {c _ {m}}, w _ {x _ {1}}, \dots , w _ {x _ {n}}, w _ {b _ {1}}, \dots , w _ {b _ {m}}, w _ {\text {d u m m y}}\right) ^ {\top}
$$

We also use  $w_{i}$  to denote the  $i$ -th coordinate of  $\mathbf{w}$ . Each  $w_{x_j}$  represents the truth value of variable  $x_{j}$ , where 1 represents TRUE and -1 represents FALSE. Similarly, each  $w_{c_j}$  represents the truth value of clause  $c_{j}$  based on the value of its variables.  $w_{b_j}$  and  $w_{\text{dummy}}$  are used for convenience of proof.

The initial value of the parameter is set to

$$
\mathbf {w} ^ {(0)} = (\overbrace {\frac {1}{2} , \dots , \frac {1}{2}} ^ {m}, \overbrace {- 1 , \dots , - 1} ^ {n}, \overbrace {- 1 , \dots , - 1} ^ {m}, 1) ^ {\top}
$$

Loss function. We denote  $U(x_0, \delta) \coloneqq \{x | x_0 - \delta < x < x_0 + \delta\}$  as the  $\delta$ -neighborhood of  $x_0$  and define  $U(\pm x_0, \delta) = U(x_0, \delta) \cup U(-x_0, \delta)$ . We define the local ramp function as

$$
r _ {x _ {0}, \delta} (x) = \left\{ \begin{array}{l l} 0 & , x \leq x _ {0} - \delta ; \\ x - x _ {0} + \delta & , x \in U (x _ {0}, \delta); \\ 2 \delta & , x \geq x _ {0} + \delta . \end{array} \right.
$$

The loss function is defined as

$$
\mathcal {L} = - \frac {1 2 N}{5} r _ {- 5, 0. 0 1} (y \mathbf {w} ^ {\top} \mathbf {x}) - r _ {- \frac {1}{2}, 0. 2 6} (y \mathbf {w} ^ {\top} \mathbf {x}) - \frac {1}{1 0 0 0 N} \sum_ {x _ {0} \in \{\pm 1, \pm 3 \}} r _ {x _ {0}, 0. 0 1} (y \mathbf {w} ^ {\top} \mathbf {x}).
$$

$\mathcal{L}$  is monotonically decreasing with derivatives

$$
\frac {\partial \mathcal {L}}{\partial w _ {i}} = \left\{ \begin{array}{l l} - \frac {1 2 N}{5} \cdot y x _ {i} & , y \mathbf {w} ^ {\top} \mathbf {x} \in U (- 5, 0. 0 1); \\ - y x _ {i} & , y \mathbf {w} ^ {\top} \mathbf {x} \in U (- \frac {1}{2}, 0. 2 6); \\ - \frac {1}{1 0 0 0 N} y x _ {i} & , y \mathbf {w} ^ {\top} \mathbf {x} \in \bigcup_ {x _ {0} \in \{\pm 1, \pm 3 \}} U (x _ {0}, 0. 0 1); \\ 0 & , o t h e r w i s e. \end{array} \right. \tag {2}
$$

Table 2: Training data for var(i)  

<table><tr><td>xxi</td><td>y</td></tr><tr><td>5</td><td>1</td></tr></table>

Table 3: Training data for clause(i, i1, i2, i3)  

<table><tr><td>\( {x}_{{c}_{i}} \)</td><td>\( {x}_{{x}_{{i}_{1}}} \)</td><td>\( {x}_{{x}_{{i}_{2}}} \)</td><td>\( {x}_{{x}_{{i}_{3}}} \)</td><td>\( {x}_{{b}_{i}} \)</td><td>y</td></tr><tr><td>1</td><td>1</td><td>1</td><td>1</td><td>\( \frac{1}{2} \)</td><td>1</td></tr></table>

Learning rate. The learning rate for SGD is set to be

$$
\boldsymbol {\eta} = (\overbrace {5 , \dots , 5} ^ {m}, \overbrace {\frac {1}{6 N} , \dots , \frac {1}{6 N}} ^ {n}, \overbrace {2 0 0 0 N , \dots , 2 0 0 0 N} ^ {m}, 1) ^ {\top}.
$$

Training data. We define two gadgets,  $\operatorname{var}(i)$  and  $\operatorname{clause}(i, i_1, i_2, i_3)$ , as illustrated in Table 2 and 3. All the unspecified coordinates are set to zero. We use  $T_0$  to denote the training data.  $\operatorname{var}(i)$  is contained in  $T_0$  if and only if  $x_i \in \varphi$ , and  $\operatorname{clause}(i, i_1, i_2, i_3)$  is contained in  $T_0$  if and only if  $c_i = (x_{i_1} \vee x_{i_2} \vee x_{i_3}) \in \varphi$ .

Threshold and instance. The threshold  $\varepsilon$  can be any fixed value in  $\mathbb{R}_+$ . The instance is defined as  $(\mathbf{x}_{\mathrm{test}},y_{\mathrm{test}})$ , where  $y_{\mathrm{test}} = 1$  and

$$
\mathbf {x} _ {\text {t e s t}} = \overbrace {(1 , \dots , 1 , 0 , \dots , 0 , \frac {- 1 1 m + 5}{2}) ^ {\top}} ^ {m}.
$$

The following reduction works for all possible training orders. Intuitively, during the training process, each var  $(i)$  in the training data will set  $w_{x_i}$  to around 1 (that is, mark  $x_i$  as TRUE) in the first epoch, and each clause  $(i, i_1, i_2, i_3)$  will set  $w_{c_i}$  to near  $\frac{11}{2}$  in the second epoch, if and only if exactly one of  $w_{x_{i_1}}, w_{x_{i_2}}, w_{x_{i_3}}$  is near 1 and the others near  $-1$  (that is, mark  $c_i$  as satisfied if exactly one of its literals is TRUE and the others FALSE). The training process terminates at the end of the second epoch.

# 4 Results for Fixed Loss Functions

We have proved the NP-hardness for DEBUGGABLE-LIN when the loss function is not fixed. In this section, we study the complexity when the loss function is fixed as linear and hinge-like functions. Assuming that SGD terminates after only one epoch with a fixed order, we will show that DEBUGGABLE-LIN is solvable in linear time for linear loss. For hinge-like loss functions, DEBUGGABLE-LIN can be solved in linear time only when the dimension  $d = 1$  and the interception  $\beta \geq 0$ . For the rest cases, DEBUGGABLE-LIN becomes NP-hard.

# 4.1 The Easy Case

We start with the linear loss function  $\mathcal{L} = -\alpha (y\mathbf{w}^{\top}\mathbf{x} + \beta)$ , with which all the training data are activated and  $\mathbf{w}^{*} = \mathbf{w}^{*}(T) = \mathbf{w}^{(0)} + \sum_{(\mathbf{x},y)\in T}\alpha y\boldsymbol {\eta}\otimes \mathbf{x}$ . Since  $y_{\mathrm{test}}\in \{-1,1\}$ , DEBUGGABLE-LIN is equivalent to deciding whether

$$
\max  _ {T ^ {\prime} \subseteq T} \left\{y _ {\text {t e s t}} \left(\mathbf {w} ^ {*} \left(T ^ {\prime}\right)\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} \right\} > 0.
$$

A training sample  $(\mathbf{x},y)$  is "good" if  $y_{\mathrm{test}}(\alpha y\boldsymbol {\eta}\otimes \mathbf{x})^{\top}\mathbf{x}_{\mathrm{test}} > 0$  and "bad" otherwise. The good training-sample assessment (GTA) algorithm, as shown in Algorithm 1, deals with this situation by greedily picking all "good" training samples.

Denoting  $T^{*}$  as the set of all good data in  $T$ , it follows that

$$
\begin{array}{l} y _ {\text {t e s t}} \left(\mathbf {w} ^ {*} \left(T ^ {*}\right)\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} = y _ {\text {t e s t}} \left(\mathbf {w} ^ {(0)}\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} + \sum_ {(\mathbf {x}, y) \in T ^ {*}} y _ {\text {t e s t}} \left(\alpha y \boldsymbol {\eta} \otimes \mathbf {x}\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} \\ \geq y _ {\text {t e s t}} \left(\mathbf {w} ^ {(0)}\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} + \sum_ {(\mathbf {x}, y) \in T ^ {\prime}} y _ {\text {t e s t}} \left(\alpha y \boldsymbol {\eta} \otimes \mathbf {x}\right) ^ {\top} \mathbf {x} _ {\text {t e s t}} \\ \end{array}
$$

for all  $T' \subseteq T$ . Hence  $\max_{T' \subseteq T} \{y_{\text{test}}(\mathbf{w}^*(T'))^\top \mathbf{x}_{\text{test}}\} = y_{\text{test}}(\mathbf{w}^*(T'))^\top \mathbf{x}_{\text{test}}$  and DEBUGGABLE-LIN can be solved by GTA in linear time. The following theorem is straightforward.

Algorithm 1: Good Training-sample Assessment (GTA)  
Output: TRUE, iff  $\mathsf{SGD}_{\Lambda}(\mathcal{L},\pmb {\eta},\varepsilon ,T)$  is debuggable on  $(\mathbf{x}_{\mathrm{test}}y_{\mathrm{test}})$  
```txt
Input: Training data  $T$ , loss function  $\mathcal{L}$ , initial parameter  $\mathbf{w}^{(0)}$ , learning rate  $\eta$ , threshold  $\varepsilon$  and test instance  $(\mathbf{x}_{\mathrm{test}}, y_{\mathrm{test}})$ .
```

```c
1  $\mathbf{w}\gets \mathbf{w}^{(0)}$  .   
2 for  $(\mathbf{x},y)\in T$  do   
3 if  $y_{test}(\alpha y\pmb {\eta}\otimes \mathbf{x})^{\top}\mathbf{x}_{test} > 0$  then   
4  $\mathbf{w}\gets \mathbf{w} + \alpha y\pmb {\eta}\otimes \mathbf{x};$    
5 end   
6 end   
7 if  $y_{test}\mathbf{w}^{\top}\mathbf{x}_{test}\geq 0$  then   
8 return TRUE;   
9 end   
10 return FALSE;
```

GTA is still effective for one-dimensional classifiers trained with hinge-like losses when  $\beta \geq 0$ .

Theorem 4.2. DEBUGGABLE-LIN is linear time solvable for hinge-like loss functions, when  $d = 1$  and  $\beta \geq 0$ .

Proof. It suffices to prove that if  $\exists T' \subseteq T$  such that  $\mathsf{SGD}_{\Lambda}(T', x_{\mathrm{test}}) = y_{\mathrm{test}}, \mathsf{SGD}_{\Lambda}(T^{*}, x_{\mathrm{test}}) = y_{\mathrm{test}}$ .

a) Suppose all the data in  $T^{*}$  are activated, we have

$$
\begin{array}{l} y _ {\text {t e s t}} w ^ {*} (T ^ {*}) x _ {\text {t e s t}} = y _ {\text {t e s t}} w ^ {(0)} x _ {\text {t e s t}} + \sum_ {(x, y) \in T ^ {*}} y _ {\text {t e s t}} \alpha y \eta x x _ {\text {t e s t}} \\ \geq y _ {\text {t e s t}} w ^ {(0)} x _ {\text {t e s t}} + \sum_ {(x, y) \in T ^ {\prime} \cap T ^ {*}} y _ {\text {t e s t}} \alpha y \eta x x _ {\text {t e s t}} + \sum_ {(x, y) \in T ^ {\prime} \backslash T ^ {*}} y _ {\text {t e s t}} \alpha y \eta x x _ {\text {t e s t}} \\ = y _ {\text {t e s t}} w ^ {*} \left(T ^ {\prime}\right) x _ {\text {t e s t}} \geq 0 \\ \end{array}
$$

b) Suppose  $(x,y)\in T^{*}$  is the first inactivated data during the training phase, and  $w$  is the current parameter, we have  $ywx > \beta$ . Since  $\alpha \eta \cdot (xy)\cdot (x_{\mathrm{test}}y_{\mathrm{test}})\geq 0$ , we have  $(x_{\mathrm{test}}y_{\mathrm{test}})\cdot w\geq 0$ . Let  $T''$  be the set of training data appeared before  $(x,y)$ , we have  $y_{\mathrm{test}}w^{*}(T^{*})x_{\mathrm{test}}\geq y_{\mathrm{test}}w^{*}(T^{\prime \prime})x_{\mathrm{test}}\geq 0$ .

# 4.2 The Hard Case

The gradient of training data may not always be activated and could be affected by the training order. When the training order is adversarially chosen, the following theorem shows that DEBUGGABLE-LIN is NP-hard for all  $d \geq 2$  and  $\beta \in \mathbb{R}$ .

Theorem 4.3. If the training order is adversarially chosen and  $d \geq 2$ , DEBUGGABLE-LIN is NP-hard for each hinge-like loss function at every constant learning rate.

Proof sketch. Since the result can be easily extended for all  $d > 2$  by padding the other  $d - 2$  dimensions with zeros, we only prove for the case of  $d = 2$ . We assume  $\beta \geq -1$  and leave the  $\beta < -1$  case to the appendix. To avoid cluttering, we further assume  $\eta = 1$  and  $\alpha = 1$ . The proof can be easily generalized by appropriately re-scaling the constructed vectors.

We build a reduction from the subset sum problem, which is well-known to be NP-hard:

```txt
SUBSET SUM Input: A set of positive integer  $S$  , and a positive integer t. Output:"Yes": if  $\exists S^{\prime}\subseteq S$  such that  $\sum_{a\in S^{\prime}}a = t$  "No": otherwise.
```

Suppose  $n = |S|$ ,  $m = \max_{a\in S}\{a\}$ ,  $\gamma = \max \{\beta, 1\}$  and  $S = \{a_1, a_2, \ldots, a_n\}$ . We further assume  $n > 1$ . Let the training data be

$$
T = \left\{\left(\mathbf {x} _ {1}, y _ {1}\right), \left(\mathbf {x} _ {2}, y _ {2}\right), \dots , \left(\mathbf {x} _ {n}, y _ {n}\right) \right\} \cup \left\{\left(\mathbf {x} _ {c}, y _ {c}\right), \left(\mathbf {x} _ {b}, y _ {b}\right), \left(\mathbf {x} _ {a}, y _ {a}\right) \right\}
$$

where  $\mathbf{x}_i y_i = \left(\frac{\sqrt{\gamma}}{n + 1}, 3\sqrt{\gamma} a_i\right)$  for all  $1 \leq i \leq n$ ,  $\mathbf{x}_c y_c = ((18n^2 m^2 - 2)\sqrt{\gamma}, -3t\sqrt{\gamma})$ ,  $\mathbf{x}_b y_b = (\sqrt{\gamma}, -\sqrt{\gamma})$ ,  $\mathbf{x}_a y_a = (\sqrt{\gamma}, \sqrt{\gamma})$ . Let  $\mathbf{w}^{(0)} = (-18n^2 m^2 \sqrt{\gamma}, 0)$ . Let the test instance  $(\mathbf{x}_{\mathrm{test}}, y_{\mathrm{test}})$  satisfy  $\mathbf{x}_{\mathrm{test}} y_{\mathrm{test}} = (1, 0)$ .  
245 Let the training order be  $(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_n, y_n), (\mathbf{x}_c, y_c), (\mathbf{x}_b, y_b), (\mathbf{x}_a, y_a)$ .

246 For each  $1\leq i <   n$  , suppose  $\mathbf{w}^{(0)}\xrightarrow{T\cap\{(\mathbf{x}_i,y_i)|1\leq j\leq i\}}\mathbf{w}_i$  , we have

$$
\begin{array}{l} y _ {i + 1} \mathbf {w} _ {i} ^ {\top} \mathbf {x} _ {i + 1} \leq \frac {\sqrt {\gamma}}{n + 1} (- 1 8 n ^ {2} m ^ {2} \sqrt {\gamma} + \frac {\sqrt {\gamma} i}{n + 1}) + 3 \sqrt {\gamma} a _ {i + 1} \sum_ {j = 1} ^ {i} 3 \sqrt {\gamma} a _ {j} \\ \leq \gamma \left(- \frac {n - 1}{n + 1} \cdot 9 n m ^ {2} + \frac {n}{(n + 1) ^ {2}}\right) <   - 1 \leq \beta \\ \end{array}
$$

This means all the  $T\setminus \{(\mathbf{x}_c,y_c),(\mathbf{x}_b,y_b),(\mathbf{x}_a,y_a)\}$  can be activated. Thus the resulting parameter trained by  $T\setminus \{(\mathbf{x}_c,y_c),(\mathbf{x}_b,y_b),(\mathbf{x}_a,y_a)\}$  is

$$
\mathbf {w} _ {c} = \mathbf {w} ^ {(0)} + \sum_ {i = 1} ^ {n} \mathbf {x} _ {i} y _ {i} = \left(- 1 8 n ^ {2} m ^ {2} \sqrt {\gamma} + \frac {\sqrt {\gamma} | T ^ {*} |}{n + 1}, 3 \sqrt {\gamma} \sum_ {i = 1} ^ {n} a _ {i}\right).
$$

It now suffices to prove that for all  $S' \subseteq S$ ,  $\sum_{a \in S'} a = t$  if and only if  $\exists T' \subseteq T$  such that  $\mathbf{w} : \mathbf{w}^{(0)} \xrightarrow{T'} \mathbf{w}$  satisfies  $y_{\mathrm{test}} \mathbf{w}^\top \mathbf{x}_{\mathrm{test}} > 0$ .  
251 If: Suppose  $\exists S' \subseteq S$  such that  $\sum_{a \in S} a = t$ , we prove that  $\exists T' \subseteq T$  such that  $y_{\mathrm{test}}(\mathbf{w}^*)^\top \mathbf{x}_{\mathrm{test}} > 0$   
252 for  $\mathbf{w}^*$  satisfying  $\mathbf{w}^{(0)} \xrightarrow{T'} \mathbf{w}^*$ .  
Let  $T^{*} = \{(\mathbf{x}_{i},y_{i})|a_{i}\in S^{\prime}\} ,T^{\prime} = T^{*}\cup \{(\mathbf{x}_{c},y_{c}),(\mathbf{x}_{b},y_{b}),(\mathbf{x}_{a},y_{a})\} .$  We have

$$
\mathbf {w} _ {c} = (- 1 8 n ^ {2} m ^ {2} \sqrt {\gamma} + \frac {\sqrt {\gamma} | T ^ {*} |}{n + 1}, 3 \sqrt {\gamma} \sum_ {a _ {i} \in S ^ {\prime}} a _ {i}) = (- 1 8 n ^ {2} m ^ {2} \sqrt {\gamma} + \frac {\sqrt {\gamma} | T ^ {*} |}{n + 1}, 3 \sqrt {\gamma} t).
$$

254 And therefore  $y_{c}\mathbf{w}_{c}^{\top}\mathbf{x}_{c} = \gamma \left((-18n^{2}m^{2} + \frac{|T^{*}|}{n + 1})(18n^{2}m^{2} - 2) - 9t^{2}\right) < -1\leq \beta$  , so

$$
\mathbf {w} _ {c} \xrightarrow {\left(\mathbf {x} _ {c} , y _ {c}\right)} \mathbf {w} _ {b} = \mathbf {w} _ {c} + \mathbf {x} _ {c} y _ {c} = (\sqrt {\gamma} (\frac {| T ^ {*} |}{n + 1} - 2), 0).
$$

255 Note that  $y_{b}\mathbf{w}_{b}^{\top}\mathbf{x}_{b} = \gamma (\frac{|T^{*}|}{n + 1} -2) <   - 1\leq \beta$  , we have

$$
\mathbf {w} _ {b} \xrightarrow {\left(\mathbf {x} _ {b} , y _ {b}\right)} \mathbf {w} _ {a} = \mathbf {w} _ {b} + \mathbf {x} _ {a} y _ {a} = (\sqrt {\gamma} (\frac {| T ^ {*} |}{n + 1} - 1), - \sqrt {\gamma})
$$

256 Note also that  $y_{a}\mathbf{w}_{a}^{\top}\mathbf{x}_{a} = \gamma (\frac{|T^{*}|}{n + 1} -2) <   - 1\leq \beta$  we have

$$
\mathbf {w} _ {a} \xrightarrow {\left(\mathbf {x} _ {a} , y _ {a}\right)} \mathbf {w} ^ {*} = \mathbf {w} _ {a} + \mathbf {x} _ {a} y _ {a} = \left(\frac {| T ^ {*} | \sqrt {\gamma}}{n + 1}, 0\right)
$$

257 Therefore,  $y_{\mathrm{test}}(\mathbf{w}^{*})^{\top}\mathbf{x}_{\mathrm{test}} = \frac{|T^{*}| \sqrt{\gamma}}{n + 1} > 0.$

Only if: For each  $T' \subseteq T$ , let  $T^{*} = T' \setminus \{(\mathbf{x}_{c}, y_{c}), (\mathbf{x}_{b}, y_{b}), (\mathbf{x}_{a}, y_{a})\}$ . If  $y_{\mathrm{test}}(\mathbf{w}^{*})^{\top} \mathbf{x}_{\mathrm{test}} > 0$  for  $\mathbf{w}^{*}$  satisfying  $\mathbf{w}^{(0)} \xrightarrow{T'} \mathbf{w}^{*}$ , we prove that  $\exists S' \subseteq S$  such that  $\sum_{a \in S'} a = t$ . We first show that for each  $T' \subseteq T$ , if  $\mathbf{w}(\mathbf{w}^{(0)} \xrightarrow{T'} \mathbf{w})$  satisfying  $y_{\mathrm{test}} \mathbf{w}^{\top} \mathbf{x}_{\mathrm{test}} > 0$ , we have  $\forall k \in \{a, b, c\}, (\mathbf{x}_{k}, y_{k}) \in T'$ ,  $y_{k} \mathbf{w}_{k}^{\top} \mathbf{x}_{k} < \gamma$ , where  $\mathbf{w}^{(0)} \xrightarrow{T'} \mathbf{w}_{c} \xrightarrow{(\mathbf{x}_{c}, y_{c})} \mathbf{w}_{b} \xrightarrow{(\mathbf{x}_{b}, y_{b})} \mathbf{w}_{a}$ . Otherwise, suppose  $\exists k \in \{a, b, c\}$  such that  $(\mathbf{x}_{k}, y_{k}) \notin T'$  or  $y_{k} \mathbf{w}_{k}^{\top} \mathbf{x}_{k} \geq \gamma$ , we have

$$
y _ {\text {t e s t}} \mathbf {w} ^ {\top} \mathbf {x} _ {\text {t e s t}} \leq \sqrt {\gamma} \left(\frac {| T ^ {*} |}{n + 1} - 1\right) <   0
$$

which contradicts to the fact that  $y_{\mathrm{test}}\mathbf{w}^{\top}\mathbf{x}_{\mathrm{test}}\geq 0$

264 Let  $S' = \{a_i | (\mathbf{x}_i, y_i) \in T^*\}$  and  $t' = \sum_{a \in S'} a_i$ , it suffices to prove  $t' = t$ . Notice that

$$
\begin{array}{l} \mathbf {w} ^ {(0)} \xrightarrow {T ^ {*} \cap \left\{\left(\mathbf {x} _ {i} , y _ {i}\right) | 1 \leq j \leq i \right\}} \mathbf {w} _ {c} = (\sqrt {\gamma} (- 1 8 n ^ {2} m ^ {2} + \frac {| T ^ {*} |}{n + 1}), 3 \sqrt {\gamma} \sum_ {a _ {i} \in S ^ {\prime}} a _ {i}) \\ = (\sqrt {\gamma} (- 1 8 n ^ {2} m ^ {2} + \frac {| T ^ {*} |}{n + 1}), 3 \sqrt {\gamma} t ^ {\prime}) \\ \end{array}
$$

Hence  $y_{c}\mathbf{w}_{c}^{\top}\mathbf{x}_{c} = \gamma (-18n^{2}m^{2} + \frac{|T^{*}|}{n + 1})(18n^{2}m^{2} - 2) - 9\gamma tt^{\prime} < - 1\leq \beta$  , thus

$$
\mathbf {w} _ {c} \xrightarrow {\left(\mathbf {x} _ {c} , y _ {c}\right)} \mathbf {w} _ {b} = \mathbf {w} _ {c} + \mathbf {x} _ {c} y _ {c} = (\sqrt {\gamma} (\frac {| T ^ {*} |}{n + 1} - 2), 3 \sqrt {\gamma} (t ^ {\prime} - t))
$$

(1) If  $t' \leq t - 1$ , we have  $y_b \mathbf{w}_b^\top \mathbf{x}_b = \gamma \left( \frac{|T^*|}{n + 1} - 2 + 3(t - t') \right) > \gamma \geq \beta$ , a contradiction.  
(2) If  $t' \geq t + 1$ , we have  $y_a \mathbf{w}_a^\top \mathbf{x}_a = \gamma \left(\frac{|T^*|}{n + 1} - 2 + 3(t' - t)\right) > \gamma \geq \beta$ , another contradiction.

Therefore  $t' = t$ , and this completes the proof.

![](images/b17da55416d50c2cbb66e0a12dce91f16df8325dbb8db13d36e83745c72d237d.jpg)

Moreover, DEBUGGABLE-LIN is NP-hard even when  $d = 1$  and  $\beta < 0$ .

Theorem 4.4. If the training order is adversarially chosen and  $d = 1$ , DEBUGGABLE-LIN remains NP-hard for each hinge-like loss function with  $\beta < 0$  at every constant learning rate.

Remarks. The training order in this section can be arbitrary as long as the last three training samples are  $(\mathbf{x}_c, y_c), (\mathbf{x}_b, y_b), (\mathbf{x}_a, y_a)$ , respectively. All the training samples are "good" since for each  $(\mathbf{x}, y) \in T$  we have  $\mathbf{x}^\top \mathbf{x}_{\mathrm{test}}yy_{\mathrm{test}} > 0$ . This implies that DEBUGGABLE-LIN is NP-hard even if all the training data are "good" training samples, and exemplifies why the GTA algorithm fails for higher dimensions.

# 5 Discussion and Conclusion

In this paper, we provided a comprehensive analysis on the complexity of DEBUGGABLE. We focus on the linear classifier that is trained using SGD, as it is a key component in the majority of popular models.

Since DEBUGGABLE is a special case of data debugging, the above results proved the intractability of data debugging and therefore gives a negative answer to Problem 1.1 declared in the introduction. The complexity results also demonstrated that it is not accurate to estimate the impact of subset of training data by summing up the score of each training samples in the subset, as long as the scores can be calculated in polynomial time.

In Section 4, a training sample is said to be "good" if it can help the resulting model to predict correctly on the test instance. That is, it can increase  $y_{\mathrm{test}}(\mathbf{w}^{*})^{\top}\mathbf{x}_{\mathrm{test}}$ . However, in our proof we showed that DEBUGGABLE remains NP-hard even if all training samples are "good". This suggests that the quality of a training sample does not depend only on some properties of itself but also on the interaction between the rest of the training data, which should be taken into consideration when developing data cleaning approaches.

Moreover, the NP-hardness of DEBUGGABLE implies that, it is in general intractable to figure out the causality between even the prediction of a linear classifier and its training data. This may be seem surprising since linear classifiers have long been considered "inherently interpretable". As warned in [25], a method being "inherently interpretable" needs to be verified before it can be trusted, the concept of interpretability must be rigorously defined, or at least its boundaries specified.

Our results suggest the following directions for future research. Firstly, characterizing the training sample may be helpful in designing efficient algorithms for data debugging; Secondly, designing algorithms using CSP-solver is a potential way to solve data debugging more efficiently than the brute-force algorithms; Finally, developing random algorithms is a potential way to solve data debugging successfully with high probability.

# References

[1] Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data Cleansing for Models Trained with SGD. Curran Associates Inc., Red Hook, NY, USA, 2019.  
[2] Weiyuan Wu, Lampros Flokas, Eugene Wu, and Jiannan Wang. Complaint-driven training data debugging for query 2.0. pages 1317-1334, 06 2020. doi: 10.1145/3318464.3389696.  
[3] Bojan Karlas, David Dao, Matteo Interlandi, Bo Li, Sebastian Schelter, Wentao Wu, and Ce Zhang. Data debugging with shapley importance over end-to-end machine learning pipelines, 2022.  
[4] Felix Neutatz, Binger Chen, Ziawasch Abedjan, and Eugene Wu. From cleaning before ml to cleaning for ml. IEEE Data Eng. Bull., 44:24-41, 2021. URL https://api-semanticscholar.org/CorpusID:237542697.  
[5] Peng Li, Xi Rao, Jennifer Blase, Yue Zhang, Xu Chu, and Ce Zhang. Cleanml: A study for evaluating the impact of data cleaning on ml classification tasks. In 2021 IEEE 37th International Conference on Data Engineering (ICDE), pages 13-24, 2021. doi: 10.1109/ICDE51399.2021.00009.  
[6] Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger Grosse. If influence functions are the answer, then what is the question? In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS '22, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713871088.  
[7] Romila Pradhan, Jiongli Zhu, Boris Glavic, and Babak Salimi. Interpretable data-based explanations for fairness debugging. In Proceedings of the 2022 International Conference on Management of Data, SIGMOD '22, page 247-261, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392495. doi: 10.1145/3514221.3517886. URL https://doi.org/10.1145/3514221.3517886.  
[8] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, page 1885-1894. JMLR.org, 2017.  
[9] Rajiv Khanna, Been Kim, Joydeep Ghosh, and Oluwasanmi Koyejo. Interpreting black box predictions using fisher kernels. In International Conference on Artificial Intelligence and Statistics, 2018. URL https://api_semanticscholar.org/CorpusID:53085397.  
[10] Pang Wei Koh, Kai-Siang Ang, Hubert Hua Kian Teo, and Percy Liang. On the accuracy of influence functions for measuring group effects. In Neural Information Processing Systems, 2019. URL https://api-semanticscholar.org/CorpusID:173188850.  
[11] Samyadeep Basu, Xuchen You, and Soheil Feizi. On second-order group influence functions for black-box predictions. In Proceedings of the 37th International Conference on Machine Learning, ICML'20. JMLR.org, 2020.  
[12] Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. FastIF: Scalable influence functions for efficient model interpretation and debugging. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10333-10350, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.808. URL https://aclanthology.org/2021.emnlp-main.808.  
[13] Amirata Ghorbani and James Y. Zou. Data shapley: Equitable valuation of data for machine learning. ArXiv, abs/1904.02868, 2019. URL https://api-semanticscholar.org/CorpusID:102350503.  
[14] R. Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nicholas Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Xiaodong Song, and Costas J. Spanos. Towards efficient data valuation based on the shapley value. ArXiv, abs/1902.10275, 2019. URL https://api(semanticscholar.org/CorpusID:67855573.  
[15] Ruoxi Jia, Fan Wu, Xuehui Sun, Jiacen Xu, David Dao, Bhavya Kailkhura, Ce Zhang, Bo Li, and Dawn Song. Scalability vs. utility: Do we have to sacrifice one for the other in data importance quantification? In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8235-8243, 2021. doi: 10.1109/CVPR46437.2021.00814.  
[16] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas Spanos, and Dawn Song. Efficient task-specific data valuation for nearest neighbor algorithms. Proc. VLDB Endow., 12(11):1610-1623, jul 2019. ISSN 2150-8097. doi: 10.14778/3342263.3342637. URL https://doi.org/10.14778/3342263.3342637.

[17] Jeremy Mange. Effect of training data order for machine learning. In 2019 International Conference on Computational Science and Computational Intelligence (CSCI), pages 406-407, 2019. doi: 10.1109/CSCI49370.2019.00078.  
[18] Ernie Chang, Hui-Syuan Yeh, and Vera Demberg. Does the order of training samples matter? improving neural data-to-text generation with curriculum learning. ArXiv, abs/2102.03554, 2021. URL https://api-semanticscholar.org/CorpusID:231846815.  
[19] Yejia Liu, Weiyuan Wu, Lampros Flokas, Jiannan Wang, and Eugene Wu. Enabling sql-based training data debugging for federated learning. Proceedings of the VLDB Endowment, 15:388-400, 02 2022. doi: 10.14778/3494124.3494125.  
[20] Marc-Etienne Brunet, Colleen Alkalay-Houlihan, Ashton Anderson, and Richard Zemel. Understanding the origins of bias in word embeddings, 2019.  
[21] Hao Wang, Berk Ustun, and Flavio P. Calmon. Repairing without retraining: Avoiding disparate impact with counterfactual distributions, 2019.  
[22] Xiaotie Deng and Christos H. Papadimitriou. On the complexity of cooperative solution concepts. Math. Oper. Res., 19:257-266, 1994. URL https://api-semanticscholar.org/CorpusID:12946448.  
[23] Qi Wang, Yue Ma, Kun Zhao, and Yingjie Tian. A comprehensive survey of loss functions in machine learning. Annals of Data Science, 9, 04 2022. doi: 10.1007/s40745-020-00253-5.  
[24] Erik D. Demaine, William Gasarch, and Mohammad Hajiaghayi. Computational Intractability: A Guide to Algorithmic Lower Bounds. MIT Press, 2024.  
[25] Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? In Annual Meeting of the Association for Computational Linguistics, 2020. URL https://api-semanticscholar.org/CorpusID:215416110.  
[26] Victor Parque. Tackling the subset sum problem with fixed size using an integer representation scheme. In 2021 IEEE Congress on Evolutionary Computation (CEC), pages 1447-1453, 2021. doi: 10.1109/ CEC45853.2021.9504889.
