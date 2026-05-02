# BETTY: AN AUTOMATIC DIFFERENTIATION LIBRARY FOR MULTILEVEL OPTIMIZATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Gradient-based multilevel optimization (MLO) has gained attention as a framework for studying numerous problems, ranging from hyperparameter optimization and meta-learning to neural architecture search and reinforcement learning. However, gradients in MLO, which are obtained by composing best-response Jacobians via the chain rule, are notoriously difficult to implement and memory/compute intensive. We take an initial step towards closing this gap by introducing BETTY, a software library for large-scale MLO. At its core, we devise a novel dataflow graph for MLO, which allows us to (1) develop efficient automatic differentiation for MLO that reduces the computational complexity from  $\mathcal{O}(d^3)$  to  $\mathcal{O}(d^2)$ , (2) incorporate systems support such as mixed-precision and data-parallel training for scalability, and (3) facilitate implementation of MLO programs of arbitrary complexity while allowing a modular interface for diverse algorithmic and systems design choices. We empirically demonstrate that BETTY can be used to implement an array of MLO programs, while also observing up to  $11\%$  increase in test accuracy,  $14\%$  decrease in GPU memory usage, and  $20\%$  decrease in training wall time over existing implementations on multiple benchmarks. We also showcase that BETTY enables scaling MLO to models with hundreds of millions of parameters. The software will be made publicly available<sup>1</sup>.

# 1 INTRODUCTION

Multilevel optimization (MLO) addresses nested optimization scenarios, where upper level optimization problems are constrained by lower level optimization problems following an underlying hierarchical dependency. MLO has gained considerable attention as a unified mathematical framework for studying diverse problems including meta-learning (Finn et al., 2017; Rajeswaran et al., 2019), hyperparameter optimization (Franceschi et al., 2017), neural architecture search (Liu et al., 2019), and reinforcement learning (Konda & Tsitsiklis, 1999; Rajeswaran et al., 2020). While a majority of existing work is built upon bilevel optimization, the simplest case of MLO, there have been recent efforts that go beyond this two-level hierarchy. For example, (Raghu et al., 2021) proposed trilevel optimization that combines hyperparameter optimization with two-level pretraining and finetuning. More generally, conducting joint optimization over machine learning pipelines consisting of multiple models and hyperparameter sets can be approached as deeper instances of MLO (Garg et al., 2022; Raghu et al., 2021; Somayajula et al., 2022; Such et al., 2020).

Following its increasing popularity, a multitude of optimization algorithms have been proposed to solve MLO. Among them, gradient-based (or first-order) approaches (Lorraine et al., 2020; Raghu et al., 2021; Sato et al., 2021) have recently received the limelight from the machine learning community, due to their ability to carry out efficient high-dimensional optimization, under which all of the above listed applications fall. Nevertheless, research in gradient-based MLO has been largely impeded by two major bottlenecks. First, implementing gradients in multilevel optimization, which is achieved by composing best-response Jacobians via the chain rule, requires both programming and mathematical proficiency. Second, algorithms for best-response Jacobian calculation, such as iterative differentiation (ITD) or approximate implicit differentiation (AID) (Grazzi et al., 2020), are memory and compute intensive, as they require multiple forward/backward computations and oftentimes second-order gradient (i.e. Hessian) information.

![](images/326c224e1f950694de99bf5f4f713715475d0823318587f1d55bf9984c1e0be2.jpg)  
Figure 1: In Engine (left), users define their MLO program as a hierarchy/graph of optimization problems. In Problem (middle), users define an optimization problem with a data loader, cost function, module, and optimizer, while upper/ lower level constraint problems (i.e.  $\mathcal{U}_k,\mathcal{L}_k$ ) are injected by Engine. The "step" function in Problem serves as the base of gradient-based optimization, abstracting the one-step gradient descent update process. Finally, users can easily try out different best-response Jacobian algorithms & system features (right) via Config in a modular manner.

In recent years, there has been some work originating in the meta-learning community on developing software libraries that target some aspects of gradient-based MLO (Blondel et al., 2021; Deleu et al., 2019; Grefenstette et al., 2019). For example, JAXopt (Blondel et al., 2021) provides efficient and modular implementations of AID algorithms by letting the user define a function capturing the optimality conditions of the problem to be differentiated. However, JAXopt fails to combine the chain rule with AID to support general MLO programs beyond a two-level hierarchy. Similarly, higher (Grefenstette et al., 2019) provides several basic primitives (e.g. making PyTorch's (Paszke et al., 2019) native optimizers differentiable) for implementing ITD/AID algorithms, but users still need to manually implement complicated internal mechanisms of these algorithms as well as the chain rule to implement a given instance of MLO. Furthermore, most existing libraries do not have systems support, such as mixed-precision and data-parallel training, that could mitigate memory and computation bottlenecks. As a result, gradient-based MLO research built upon these libraries has been largely limited to simple bilevel optimization and small-scale setups.

In this paper, we attempt to bridge this gap between research and software systems by introducing BETTY, an easy-to-use and modular automatic differentiation library with various systems support for large-scale MLO. The main contributions of this paper are as follows:

1. We develop an efficient automatic differentiation technique for MLO based on a novel interpretation of MLO as a special type of dataflow graph (Section 3). In detail, gradient calculation for each optimization problem is automatically carried out by iteratively multiplying best-response Jacobians (defined in Section 2) through the chain rule while reverse-traversing specific paths of this dataflow graph. This reverse-traversing procedure is crucial for efficiency, as it reduces the computational complexity of our automatic differentiation technique from  $\mathcal{O}(d^3)$  to  $\mathcal{O}(d^2)$ , where  $d$  is the dimension of the largest optimization problem in the MLO program.  
2. We introduce a software library for MLO, BETTY, built upon the above automatic differentiation technique. Our software design (Section 4), motivated by the dataflow graph interpretation, provides two major benefits: (1) it allows for incorporating various systems support, such as mixed-precision and data-parallel training, for large-scale MLO, and (2) it facilitates implementation of MLO programs of arbitrary complexity while allowing a modular interface for diverse algorithmic and systems design choices. The overall software architecture of BETTY is presented in Figure 1.  
3. We empirically demonstrate that BETTY can be used to implement an array of MLO applications with varying scales and complexities (Section 5). Interestingly, we observe that trying out different best-response Jacobian algorithms with our modular interface (which only requires changing one line of code) can lead to up to  $11\%$  increase in test accuracy,  $14\%$  decrease in GPU memory usage, and  $20\%$  decrease in training wall time on various benchmarks, compared with the original papers' implementations. Finally, we showcase the scalability of BETTY to models with hundreds of millions of parameters by performing MLO on the BERT-base model with the help of BETTY's systems support, which was otherwise infeasible.

# 2 BACKGROUND: GRADIENT-BASED MULTILEVEL OPTIMIZATION

To introduce MLO, we first define an important concept known as a "constrained problem" (Vicente & Calamai, 1994).

Definition 1. An optimization problem  $P$  is said to be constrained by  $\lambda$  when its cost function  $\mathcal{C}$  has  $\lambda$  as an argument in addition to the optimization parameter  $\theta$  (i.e.  $P: \arg \min_{\theta} \mathcal{C}(\theta, \lambda, \dots)$ ).

Multilevel optimization (Migdalas et al., 1998) refers to a field of study that aims to solve a nested set of optimization problems defined on a sequence of so-called levels, which satisfy two main criteria: A1) upper-level problems are constrained by the optimal parameters of lower-level problems while A2) lower-level problems are constrained by the nonoptimal parameters of upper-level problems. Formally, an  $n$ -level MLO program can be written as:

$$
P _ {n}: \quad \theta_ {n} ^ {*} = \underset {\theta_ {n}} {\operatorname {a r g m i n}} \mathcal {C} _ {n} (\theta_ {n}, \mathcal {U} _ {n}, \mathcal {L} _ {n}; \mathcal {D} _ {n}) \quad \triangleright \text {L e v e l} n \text {p r o b l e m}
$$

$$
P _ {k}: \quad \begin{array}{l} \ddots \\ \text {s . t .} \theta_ {k} ^ {*} = \underset {\theta_ {k}} {\operatorname {a r g m i n}} \mathcal {C} _ {k} \left(\theta_ {k}, \mathcal {U} _ {k}, \mathcal {L} _ {k}; \mathcal {D} _ {k}\right) \end{array} \quad \triangleright \text {L e v e l} k \in \{2, \dots , n - 1 \}
$$

$$
P _ {1}: \qquad \qquad \begin{array}{c} \ddots \\ \text {s . t .} \theta_ {1} ^ {*} = \underset {\theta_ {1}} {\operatorname {a r g m i n}} \mathcal {C} _ {1} (\theta_ {1}, \mathcal {U} _ {1}, \mathcal {L} _ {1}; \mathcal {D} _ {k}) \quad \triangleright \text {L e v e l 1 p r o b l e m} \end{array}
$$

where,  $P_{k}$  stands for the level  $k$  problem,  $\theta_{k} / \theta_{k}^{*}$  for corresponding nonoptimal / optimal parameters, and  $\mathcal{U}_k / \mathcal{L}_k$  for the sets of constraining parameters from upper/ lower level problems. Here,  $\mathcal{D}_k$  is the training dataset, and  $\mathcal{C}_k$  indicates the cost function. Due to criteria A1 & A2, constraining parameters from upper-level problems should be nonoptimal (i.e.  $\mathcal{U}_k\subseteq \{\theta_{k + 1},\dots ,\theta_n\}$ ) while constraining parameters from lower-level problems should be optimal (i.e.  $\mathcal{L}_k\subseteq \{\theta_1^*,\dots ,\theta_{k - 1}^*\}$ ). Although we denote only one optimization problem per level in the above formulation, each level could in fact have multiple problems. Therefore, we henceforth discard the concept of level, and rather assume that problems  $\{P_1,P_2,\dots ,P_n\}$  of a general MLO program are topologically sorted in a "reverse" order (i.e.  $P_{n} / P_{1}$  denote uppermost/ lowermost problems).

For example, in hyperparameter optimization formulated as bilevel optimization, hyperparameters and network parameters (weights) correspond to upper and lower level parameters  $(\theta_{2}$  and  $\theta_{1})$ . Train / validation losses correspond to  $\mathcal{C}_1 / \mathcal{C}_2$ , and validation loss is dependent on optimal network parameters  $\theta_1^*$  obtained given  $\theta_{2}$ . Thus, constraining sets for each level are  $\mathcal{U}_1 = \{\theta_2\}$  and  $\mathcal{L}_2 = \{\theta_1^*\}$ .

In this paper, we focus in particular on gradient-based MLO, rather than zeroth-order methods like Bayesian optimization (Cui & Bai, 2019), in order to efficiently scale to high-dimensional problems. Essentially, gradient-based MLO calculates gradients of the cost function  $\mathcal{C}_k(\theta_k,\mathcal{U}_k,\mathcal{L}_k)$  with respect to the corresponding parameter  $\theta_{k}$ , with which gradient descent is performed to solve for optimal parameters  $\theta_{k}^{*}$  for every problem  $P_{k}$ . Since optimal parameters from lower level problems (i.e.  $\theta_l^*\in \mathcal{L}_k$ ) can be functions of  $\theta_{k}$  (criterion A2),  $\frac{d\mathcal{C}_k}{d\theta_k}$  can be expanded using the chain rule as follows:

$$
\frac {d \mathcal {C} _ {k}}{d \theta_ {k}} = \underbrace {\frac {\partial \mathcal {C} _ {k}}{\partial \theta_ {k}}} _ {\text {d i r e c t g r a d i e n t}} + \sum_ {\theta_ {l} ^ {*} \in \mathcal {L} _ {k}} \underbrace {\frac {d \theta_ {l} ^ {*}}{d \theta_ {k}}} _ {\text {b e s t - r e s p o n s e J a c o b i a n}} \times \underbrace {\frac {\partial \mathcal {C} _ {k}}{\partial \theta_ {l} ^ {*}}} _ {\text {d i r e c t g r a d i e n t}} \tag {1}
$$

While calculating direct gradients (purple) is straightforward with existing automatic differentiation engines like PyTorch (Paszke et al., 2019), a major difficulty in gradient-based MLO lies in best-response Jacobian $^{2}$  (blue) calculation, which will be discussed in depth in Section 3. Once gradient calculation for each level  $k$  is enabled via Equation (1), gradient-based optimization is executed from lower to upper level problems in a topologically reverse order, reflecting underlying hierarchies.

# 3 AUTOMATIC DIFFERENTIATION FOR MULTILEVEL OPTIMIZATION

While Equation (1) serves as a mathematical basis for gradient-based multilevel optimization, how to automatically and efficiently carry out such gradient calculation has not been extensively studied.

and incorporated into a software system that can support MLO programs involving many problems with complex dependencies. In this section, we discuss the challenges in building an automatic differentiation library for MLO, and provide solutions to address these challenges.

# 3.1 DATAFLOW GRAPH FOR MULTILEVEL OPTIMIZATION

One may observe that the best-response Jacobian term in Equation (1) is expressed with a total derivative instead of a partial derivative. This is because  $\theta_{k}$  can affect  $\theta_{l}^{*}$  not only through a direct interaction, but also through multiple indirect interactions via other lower-level optimal parameters. For example, consider the four-problem MLO program illustrated in Figure 2. Here, the parameter of Problem 4 ( $\theta_{p_4}$ ) affects the optimal parameter of Problem 3 ( $\theta_{p_3}^*$ ) in two different ways: 1)  $\theta_{p_4} \rightarrow \theta_{p_3}^*$  and 2)  $\theta_{p_4} \rightarrow \theta_{p_1}^* \rightarrow \theta_{p_3}^*$ . In general, we can expand the best-response Jacobian  $\frac{d\theta_{l}^{*}}{d\theta_{k}}$  in Equation (1) by applying the chain rule for all paths from  $\theta_{k}$  to  $\theta_{l}^{*}$  as

$$
\frac {d \mathcal {C} _ {k}}{d \theta_ {k}} = \frac {\partial \mathcal {C} _ {k}}{\partial \theta_ {k}} + \sum_ {\theta_ {l} ^ {*} \in \mathcal {L} _ {k}} \sum_ {q \in \mathcal {Q} _ {k, l}} \left(\underbrace {\frac {\partial \theta_ {q (1)} ^ {*}}{\partial \theta_ {k}}} _ {\text {u p p e r - t o - l o w e r}} \times \left(\prod_ {i = 1} ^ {\text {l e n} (q) - 1} \underbrace {\frac {\partial \theta_ {q (i + 1)} ^ {*}}{\partial \theta_ {q (i)} ^ {*}}}\right) \times \frac {\partial \mathcal {C} _ {k}}{\partial \theta_ {l} ^ {*}}\right) \tag {2}
$$

where  $\mathcal{Q}_{k,l}$  is a set of paths from  $\theta_{k}$  to  $\theta_{l}^{*}$ , and  $q(i)$  refers to the index of the  $i$ -th problem in the path  $q$  with the last point being  $\theta_{l}^{*}$ . Replacing a total derivative term in Equation (1) with a product of partial derivative terms using the chain rule allows us to ignore indirect interactions between problems, and only deal with direct interactions.

To formalize the path finding problem, we develop a novel dataflow graph for MLO. Unlike traditional dataflow graphs with no predefined hierarchy among nodes, a dataflow graph for multilevel optimization has two different types of directed edges stemming from criteria A1 & A2: lower-to-upper and upper-to-lower. Each of these directed edges is respectively depicted with green and red arrows in Figure 2. Essentially, a lower-to-upper edge represents the directed dependency between two optimal parameters (i.e.  $\theta_{i}^{*}\rightarrow \theta_{j}^{*}$  with  $i < j$ ), while an upper-to-lower edge represents the directed dependency between nonoptimal and optimal parameters (i.e.  $\theta_{i}\rightarrow \theta_{j}^{*}$  with  $i > j$ ). Since we need to find paths from the nonoptimal parameter  $\theta_{k}$  to the optimal parameter  $\theta_{l}^{*}$ , the first directed edge must be an upper-to-lower edge (red), which connects  $\theta_{k}$  to some lower-level optimal parameter. Once it reaches the optimal parameter, it

can only move through optimal parameters via lower-to-upper edges (green) in the dataflow graph. Therefore, every valid path from  $\theta_{k}$  to  $\theta_l^*$  will start with an upper-to-lower edge, and then reach the destination only via lower-to-upper edges. The best-response Jacobian term for each edge in the dataflow graph is also marked with the corresponding color in Equation (2). We implement the above path finding mechanism with a modified depth-first search (DFS) algorithm in BETTY.

![](images/301ef151ffe77cfd6d737ca8a183d6b67f9f32bdc370fee9696b66f2a88058d7.jpg)  
Figure 2: An example dataflow graph for MLO.

# 3.2 GRADIENT CALCULATION WITH BEST-RESPONSE JACOBIANS

Automatic differentiation for MLO can be realized by calculating Equation (2) for each problem  $P_{k}$  ( $k = 1, \dots, n$ ). However, a naive calculation of Equation (2) could be computationally onerous as it involves multiple matrix multiplications with best-response Jacobians, of which computational complexity is  $\mathcal{O}(d^3)$ , where  $d$  is the dimension of the largest optimization problem in the MLO program. To alleviate this issue, we observe that the rightmost term in Equation (2) is a vector, which allows us to reduce the computational complexity of Equation (2) to  $\mathcal{O}(d^2)$  by iteratively performing matrix-vector multiplication from right to left (or, equivalently, reverse-traversing a path  $q$  in the dataflow graph). As such, matrix-vector multiplication between the best-response Jacobian and a vector serves as a base operation of efficient automatic differentiation for MLO. Mathematically, this problem can be simply written as follows:

$$
\text {C a l c u l a t e} \frac {\partial w ^ {*} (\lambda)}{\partial \lambda} \times v \tag {3}
$$

$$
\text {G i v e n} w ^ {*} (\lambda) = \underset {w} {\operatorname {a r g m i n}} \mathcal {C} (w, \lambda). \tag {4}
$$

Two major challenges in the above problems are: 1) approximating the solution of the optimization problem (i.e.  $w^{*}(\lambda)$ ), and 2) differentiating through the (approximated) solution.

In practice, an approximation of  $w^{*}(\lambda)$  is typically achieved by unrolling a small number of gradient steps, which can significantly reduce the computational cost (Franceschi et al., 2017). While we could potentially obtain a better approximation of  $w^{*}(\lambda)$  by running gradient steps until convergence, this procedure alone can take a few days (or even weeks) when the underlying optimization problem is large-scale (Deng et al., 2009; Devlin et al., 2018).

Once  $w^{*}(\lambda)$  is approximated, matrix-vector multiplication between the best-response Jacobian  $\frac{dw^{*}(\lambda)}{d\lambda}$  and a vector  $v$  is popularly obtained by either iterative differentiation (ITD) or approximate implicit differentiation (AID) (Grazzi et al., 2020). This problem has been extensively studied in bilevel optimization literature (Finn et al., 2017; Franceschi et al., 2017; Lorraine et al., 2020), and we direct interested readers to the original papers, as studying these algorithms is not the focus of this paper. In BETTY, we provide implementations of several popular ITD/AID algorithms which users can easily plug-and-play for their MLO applications. Currently available algorithms within BETTY include ITD with reverse-mode automatic differentiation (ITD-RMAD) (Finn et al., 2017), AID with Neumann series (AID-NMN) (Lorraine et al., 2020), AID with conjugate gradient (AID-CG) (Rajeswaran et al., 2019), and AID with finite difference (AID-FD) (Liu et al., 2019).

# 3.3 EXECUTION OF MULTILEVEL OPTIMIZATION

In MLO, optimization of each problem should be performed in a topologically reverse order, as the upper-level optimization is constrained by the result of lower-level optimization. To ease an MLO implementation, we also automate such an execution order with the dataflow graph developed in Section 3.1. Specifically, let's assume that there is a lower-to-upper edge between problems  $P_{i}$  and  $P_{j}$  (i.e.  $\theta_{i}^{*} \rightarrow \theta_{j}^{*}$ ). When the optimization process (i.e. a small number of gradient steps) of the problem  $P_{i}$  is complete, it can call the problem  $P_{j}$  to start its one-step gradient descent update through the lower-to-upper edge. The problem  $P_{j}$  waits until all lower level problems in  $\mathcal{L}_{j}$  send their calls, and then performs the one-step gradient descent update when all the calls from lower levels are received. Hence, to achieve the full execution of gradient-based MLO, we only need to call the one-step gradient descent processes of the lowermost problems, as the optimization processes of upper problems will be automatically called from lower problems via lower-to-upper edges.

To summarize, automatic differentiation for MLO is accomplished by performing gradient updates of multiple optimization problems in a topologically reverse order based on the lower-to-upper edges (Sec. 3.3), where gradients for each problem are calculated by iteratively multiplying best-response Jacobians obtained with ITD/AID (Sec. 3.2) while reverse-traversing the dataflow graph (Sec. 3.1).

# 4 SOFTWARE DESIGN

On top of the automatic differentiation technique developed in Section 3, we build an easy-to-use and modular software library, BETTY, with various systems support for large-scale gradient-based MLO. In detail, we break down MLO into two high-level concepts, namely 1) optimization problems and 2) hierarchical dependencies among problems, and design abstract Python classes for both of them. Such abstraction is also motivated by our dataflow graph interpretation, as each of these concepts respectively corresponds to nodes and edges. The architecture of BETTY is shown in Figure 1

Problem Each optimization problem  $P_{k}$  in MLO is defined by the parameter (or module)  $\theta_{k}$ , the sets of the upper and lower constraining problems  $\mathcal{U}_k \& \mathcal{L}_k$ , the dataset  $\mathcal{D}_k$ , the cost function  $\mathcal{C}_k$ , the optimizer, and other optimization configurations (e.g. best-response Jacobian calculation algorithm, number of unrolling steps). The Problem class is an interface where users can provide each of the aforementioned components to define the optimization problem. In detail, each one except for the cost function  $\mathcal{C}_k$  and the constraining problems  $\mathcal{U}_k \& \mathcal{L}_k$  can be provided through the class constructor, while the cost function can be defined through a "training_step" method and the constraining problems are automatically provided by Engine.

Abstracting an optimization problem by encapsulating module, optimizer, and data loader together additionally allows us to implement various systems support, including mixed-precision, data-parallel training, and gradient accumulation, within the abstract Problem class. A similar strategy has also

been adopted in popular frameworks for large-scale deep learning such as DeepSpeed (Rajbhandari et al., 2020). Since implementations of such systems support as well as best-response Jacobian are abstracted away, users can easily plug-and-play different algorithmic and systems design choices, such as unrolling steps or mixed-precision training, via Config in a modular fashion. An example usage of Problem is shown in Listing 1, and a full list of supported features in Config is provided in Appendix F.

```python
1 class MyProblem(Problem):   
2 def training_step(self, batch):   
3 # Users define the cost function here   
4 return cost_fn(batch, self/module, self.other_probs, ...)   
5 config = Config(type="darts", unroll_steps=10, fp16=True, gradient Accumulation=4)   
6 prob = MyProblem("myproblem", config, module, optimizer, dataloader)
```

Listing 1: Problem class example.

Engine While Problem manages each optimization problem, Engine handles hierarchical dependencies among problems in the dataflow graph. As discussed in Section 3.1, a dataflow graph for MLO has upper-to-lower and lower-to-upper directed edges. We allow users to define two separate graphs, one for each type of edge, using a Python dictionary, in which keys/values respectively represent start/end nodes of the edge. When user-defined dependency graphs are provided, Engine compiles them and finds all paths required for automatic differentiation with a modified depth-first search algorithm. Moreover, Engine sets constraining problem sets for each problem based on the dependency graphs, as mentioned above. Once all initialization processes are done, users can run a full MLO program by calling Engine's run method, which repeatedly calls the one-step gradient descent procedure of lowermost problems. The example usage of Engine is provided in Listing 2.

```txt
1 prob1 = MyProblem1(...)  
2 prob2 = MyProblem2(...)  
3 dependency = {"u21": {prob1: [prob2]}, "l2u": {prob1: [prob2]}}  
4 engine = Engine(problems=[prob1, prob2], dependencies=dependency)  
5 engine.run()
```

Listing 2: Engine class example.

# 5 EXPERIMENTS

To showcase the general applicability of BETTY, we implement three MLO benchmarks with varying complexities and scales: data reweighting for class imbalance (Sec. 5.1), correcting and reweighting corrupted labels (Sec. 5.2), and domain adaptation for a pretraining/finetuning framework (Sec. 5.3). Furthermore, we analyze the effect of different best-response Jacobian algorithms and system features by reporting GPU memory usage and training wall time. Last but not least, in the Appendix, we include an additional MLO benchmark experiment on differentiable neural architecture search (Appendix A), code examples (Appendix B), training details such as hyperparameters (Appendix C), analyses on various algorithmic and systems design choices (Appendix D and E).

# 5.1 DATA REWEIGHTING FOR CLASS IMBALANCE

Many real-world datasets suffer from class imbalance due to underlying long-tailed data distributions. Meta-Weight-Net (MWN) (Shu et al., 2019) proposes to alleviate the class imbalance issue with a data reweighting scheme where they learn to assign higher/ lower weights to data from more rare/common classes. In detail, MWN formulates data reweighting with bilevel optimization as follows:

$$
\theta^ {*} = \underset {\theta} {\operatorname {a r g m i n}} \mathcal {L} _ {v a l} (w ^ {*} (\theta)) \quad \triangleright \text {R e w e i g h t i n g}
$$

$$
\text {s . t .} w ^ {*} (\theta) = \underset {w} {\operatorname {a r g m i n}} \frac {1}{N} \sum_ {i = 1} ^ {n} \mathcal {R} \left(L _ {t r a i n} ^ {i}; \theta\right) \cdot L _ {t r a i n} ^ {i} (f (x _ {i}; w), y _ {i}) \quad \triangleright \text {C l a s s i f i c a t i o n}
$$

where  $w$  is the network parameters,  $L_{train}^{i}$  is the training loss for the  $i$ -th training sample, and  $\theta$  is the MWN  $\mathcal{R}$ 's parameters, which reweights each training sample given its training loss  $L_{train}^{i}$ .

Following the original paper, we artificially inject class imbalance into the CIFAR-10 dataset by geometrically decreasing the number of data sample for each class, as per an imbalance factor. While

the official implementation, which is built upon Torchmeta (Deleu et al., 2019), only adopts ITD-RMAD for best-response Jacobian calculation, we re-implement MWN with multiple best-response Jacobian algorithms, which only require one-liner changes using BETTY, to study their effect on test accuracy, memory efficiency, and training wall time. The experiment results are given in Table 1.

Table 1: MWN experiment results. IF denotes an imbalance factor. AID-CG/NMN/FD respectively stand for implicit differentiation with conjugate gradient/Neumann series/finite difference.  

<table><tr><td></td><td>Algorithm</td><td>IF 200</td><td>IF 100</td><td>IF 50</td><td>Memory</td><td>Time</td></tr><tr><td>MWN (original)</td><td>ITD-RMAD</td><td>68.91</td><td>75.21</td><td>80.06</td><td>2381MiB</td><td>35.8m</td></tr><tr><td>MWN (ours, step=1)</td><td>ITD-RMAD</td><td>71.96</td><td>75.13</td><td>79.50</td><td>2381MiB</td><td>36.0m</td></tr><tr><td>MWN (ours, step=1)</td><td>AID-CG</td><td>66.23±1.88</td><td>70.88±1.68</td><td>75.41±0.61</td><td>2435MiB</td><td>67.4m</td></tr><tr><td>MWN (ours, step=1)</td><td>AID-NMN</td><td>66.45±1.18</td><td>70.92±1.35</td><td>75.90 ±1.73</td><td>2419MiB</td><td>67.1m</td></tr><tr><td>MWN (ours, step=1)</td><td>AID-FD</td><td>75.45±0.63</td><td>78.11±0.43</td><td>81.15±0.25</td><td>2051MiB</td><td>28.5m</td></tr><tr><td>MWN (ours, step=5)</td><td>AID-FD</td><td>76.56±1.19</td><td>80.45±0.73</td><td>83.11±0.54</td><td>2051MiB</td><td>65.5m</td></tr></table>

We observe that different best-Jacobian algorithms lead to vastly different test accuracy, memory efficiency, and training wall time. Interestingly, we notice that AID-FD with unrolling steps of both 1 and 5 consistently achieve better test accuracy (close to SoTA (Tang et al., 2020)) and memory efficiency than other methods. This demonstrates that, while BETTY is developed to support large and general MLO programs, it is still useful for simpler bilevel optimization tasks as well. An additional analysis on the effect of best-response Jacobian can also be found in Appendix D.

Furthermore, to demonstrate the scalability of BETTY to large-scale MLO, we applied MWN to sentence classification with the BERT-base model (Devlin et al., 2018) with 110M parameters. Similarly, we artificially inject class imbalance into the SST dataset, and use AID-FD as our best-response Jacobian calculation algorithm. The experiment results are provided in Table 2.

Table 2: MWN+BERT experiment results. fp32 and fp16 respectively stand for full-precision and mixed-precision training.  

<table><tr><td></td><td>Algorithm</td><td>IF 20</td><td>IF 50</td><td>Memory</td></tr><tr><td>Baseline</td><td>AID-FD</td><td>89.99±0.38</td><td>87.54±0.70</td><td>8319MiB</td></tr><tr><td>MWN (fp32)</td><td>AID-FD</td><td>-</td><td>-</td><td>Out-of-memory</td></tr><tr><td>MWN (fp16)</td><td>AID-FD</td><td>91.06±0.09</td><td>89.79±0.65</td><td>10511MiB</td></tr></table>

As shown above, default full-precision training fails due to the CUDA out-of-memory error, while mixed-precision training, which only requires a one-line change in Config, avoids this issue while also providing consistent improvements in test accuracy compared to the BERT baseline. This demonstrates that our system features are indeed effective in scaling MLO to large models. We include more analyses on our systems support in Appendix E.

# 5.2 CORRECTING & REWEIGHTING CORRUPTED LABELS

Another common pathology in real-world data science is the issue of label corruption, stemming from noisy data preparation processes (e.g. Amazon MTurk). One prominent example of this is in weak supervision (Ratner et al., 2016), where users create labels for large training sets by leveraging multiple weak/noisy labeling sources such as heuristics and knowledge bases. Due to the nature of weak supervision, generated labels are generally noisy, and consequently lead to a significant performance degradation. In this example, we aim to mitigate this issue by 1) correcting and 2) reweighting potentially corrupted labels. More concretely, this problem can be formulated as an extended bilevel optimization problem, as, unlike the MWN example, we have two optimization problems—correcting and reweighting—in the upper level, as opposed to one. The mathematical formulation of this MLO program is as follows:

$$
\theta^ {*} = \underset {\theta} {\operatorname {a r g m i n}} \mathcal {L} _ {v a l} (w ^ {*} (\theta , \alpha)), \quad \alpha^ {*} = \underset {\alpha} {\operatorname {a r g m i n}} \mathcal {L} _ {v a l} ^ {\prime} (w ^ {*} (\theta , \alpha)) \quad \triangleright \mathrm {R W T} \& \mathrm {C R T}
$$

$$
\text {s . t .} w ^ {*} (\theta , \alpha) = \underset {w} {\operatorname {a r g m i n}} \frac {1}{N} \sum_ {i = 1} ^ {n} \mathcal {R} \left(L _ {t r a i n} ^ {i}; \theta\right) \cdot L _ {t r a i n} ^ {i} (f (x _ {i}; w), g (x _ {i}, y _ {i}; \alpha)) \quad \triangleright \text {C l a s s i f i c a t i o n}
$$

where,  $\alpha$  is the parameter for the label correction network  $g$ , and  $\mathcal{L}_{val}^{\prime}$  is augmented with the classification loss of the correction network in addition to that of the main classification network  $f$  on the clean validation set.

We test our framework on the WRENCH benchmark (Zhang et al., 2021a), which contains multiple weak supervision datasets. In detail, we use a 2-layer MLP as our classifier, AID-FD as our best-response Jacobian algorithm, and Snorkel Data Programming (Ratner et al., 2016) as our weak supervision algorithm for generating training labels. The experiment results are provided in Table 3.

Table 3: Wrench Results. RWT stands for reweighting and CRT for correction  

<table><tr><td></td><td>TREC</td><td>AGNews</td><td>IMDB</td><td>SemEval</td><td>ChemProt</td><td>YouTube</td></tr><tr><td>Snorkel</td><td>57.52±0.18</td><td>62.00±0.07</td><td>71.03±0.55</td><td>71.00±0.00</td><td>51.54±0.41</td><td>77.44±0.22</td></tr><tr><td>Baseline</td><td>53.88±1.83</td><td>80.74±0.20</td><td>72.26±0.81</td><td>71.50±0.44</td><td>54.47±0.78</td><td>88.16±1.56</td></tr><tr><td>+RWT</td><td>57.56±1.41</td><td>82.79±0.10</td><td>77.18±0.13</td><td>77.23±3.38</td><td>65.33±0.72</td><td>91.60±0.75</td></tr><tr><td>+RWT&amp;CRT</td><td>66.76±1.31</td><td>83.16±0.20</td><td>77.80±0.26</td><td>84.34±1.43</td><td>67.69±1.17</td><td>91.52±0.66</td></tr></table>

We observe that simultaneously applying label correction and reweighting significantly improves the test accuracy over the baseline and the reweighting-only scheme in almost all tasks. Thanks to BETTY, adding label correction in the upper-level on top of the existing reweighting scheme only requires defining one more Problem class, and accordingly updating the problem dependency in Engine (code examples can be found in Appendix B).

# 5.3 DOMAIN ADAPTATION FOR PRETRAINING & FINETUNING

Pretraining/finetuning paradigms are increasingly adopted with recent advances in self-supervised learning (Devlin et al., 2018; He et al., 2020). However, the data for pretraining are oftentimes from a different distribution than the data for finetuning, which could potentially cause negative transfer. Thus, domain adaptation emerges as a natural solution to mitigate this issue. As a domain adaptation strategy, (Raghu et al., 2021) proposes to combine data reweighting with a pretraining/finetuning framework to automatically decrease/increase the weight of pretraining samples that cause negative/positive transfer. In contrast with the above two benchmarks, this problem can be formulated as trilevel optimization as follows:

$$
\theta^ {*} = \underset {\theta} {\operatorname {a r g m i n}} \mathcal {L} _ {F T} (v ^ {*} (w ^ {*} (\theta)))
$$

$\triangleright$  Reweighting

$$
\text {s . t .} v ^ {*} (w ^ {*} (\theta)) = \underset {v} {\operatorname {a r g m i n}} \left(\mathcal {L} _ {F T} (v) + \lambda \| v - w ^ {*} (\theta) \| _ {2} ^ {2}\right)
$$

$\triangleright$  Finetuning

$$
w ^ {*} (\theta) = \underset {w} {\operatorname {a r g m i n}} \frac {1}{N} \sum_ {i = 1} ^ {n} \mathcal {R} (x _ {i}; \theta) \cdot L _ {P T} ^ {i} (w)
$$

Pretraining

where  $x_{i} / L_{PT}^{i}$  stands for the  $i$ -th pretraining sample/loss,  $\mathcal{R}$  for networks that reweight importance for each pretraining sample  $x_{i}$ , and  $\lambda$  for the proximal regularization parameter. Additionally,  $w$ ,  $v$ , and  $\theta$  are respectively parameters for pretraining, finetuning, and reweighting networks.

We conduct an experiment on the OfficeHome dataset (Venkateswara et al., 2017) that consists of 15,500 images from 65 classes and 4 domains: Art (Ar), Clipart (Cl), Product (Pr), and Real World (RW). Specifically, we randomly choose 2 domains and use one of them as a pretraining task and the other as a finetuning task. ResNet-18 (He et al., 2016) is used for all pretraining/finetuning/reweighting networks, and AID-FT with an unrolling step of 1 is used as our best-response Jacobian algorithm. Following (Bai et al., 2021), the finetuning and the reweighting stages share the same training dataset. We adopted a normal pretraining/finetuning framework without the reweighting stage as our baseline, and the result is presented in Table 4.

Our trilevel optimization framework achieves consistent improvements over the baseline for every task combination at the cost of additional memory usage and wall time, which demonstrates the empirical usefulness of multilevel optimization beyond a two-level hierarchy. Finally, we provide an example of (a simplified version of) the code for this experiment in Appendix B to showcase the usability of our library for a general MLO program.

Table 4: Domain Adaptation for Pretraining & Finetuning results. Reported numbers are classification accuracy on the target domain (right of arrow), after pretraining on the source domain (left of arrow). We note that Baseline is a two-layer, and Baseline + Reweight a three-layer, MLO program.  

<table><tr><td></td><td>Algorithm</td><td>Cl→Ar</td><td>Ar→Pr</td><td>Pr→Rw</td><td>Rw→Cl</td><td>Memory</td><td>Time</td></tr><tr><td>Baseline</td><td>N/A</td><td>65.43±0.36</td><td>87.62±0.33</td><td>77.43±0.41</td><td>68.76±0.13</td><td>3.8GiB</td><td>290s</td></tr><tr><td>+ RWT</td><td>AID-FD</td><td>67.76±0.83</td><td>88.53±0.42</td><td>78.58±0.17</td><td>69.75±0.43</td><td>8.2GiB</td><td>869s</td></tr></table>

# 6 RELATED WORK

Bilevel & Multilevel Optimization There are a myriad of machine learning applications that are built upon bilevel optimization (BLO), the simplest case of multilevel optimization with a two-level hierarchy. For example, neural architecture search (Liu et al., 2019; Zhang et al., 2021b), hyperparameter optimization (Franceschi et al., 2017; Lorraine et al., 2020; Maclaurin et al., 2015), reinforcement learning (Hong et al., 2020; Konda & Tsitsiklis, 1999), data valuation (Ren et al., 2020; Wang et al., 2020), meta learning (Finn et al., 2017; Rajeswaran et al., 2019), and label correction (Zheng et al., 2019) are formulated as BLO. In addition to applying BLO to machine learning tasks, a variety of optimization techniques (Couellan & Wang, 2016; Grazzi et al., 2020; Ji et al., 2021; Liu et al., 2021) have been developed for solving BLO.

Following the popularity of BLO, MLO with more than a two-level hierarchy has also attracted increasing attention recently (Raghu et al., 2021; Somayajula et al., 2022; Such et al., 2020; Xie & Du, 2022). In general, these works construct complex multi-stage ML pipelines, and optimize the pipelines in an end-to-end fashion with MLO. For instance, (Garg et al., 2022) constructs the pipeline of (data generation)-(architecture search)-(classification) and (He et al., 2021) of (data reweighting)-(finetuning)-(pretraining), all of which are solved with MLO. Furthermore, (Sato et al., 2021) study gradient-based methods for solving MLO with theoretical guarantees.

Multilevel Optimization Software There are several software libraries that are frequently used for implementing MLO programs. Most notably, JAXopt (Blondel et al., 2021) proposes an efficient and modular approach for AID by leveraging JAX's native autodiff of the optimality conditions. Despite its easy-to-use programming interface for AID, it fails to support combining the chain rule with AID as in Equation (2), because it overrides the default behavior of JAX's automatic differentiation, which takes care of the chain rule. Therefore, it cannot be used for implementing MLO beyond a two-level hierarchy without major changes in the source code and the software design. Alternatively, higher (Grefenstette et al., 2019) provides two major primitives of making 1) stateful PyTorch modules stateless and 2) PyTorch optimizers differentiable to ease the implementation of AID/ITD. However, users still need to manually implement complicated internal mechanisms of these algorithms as well as the chain rule with the provided primitives. Torchmeta (Deleu et al., 2019) also provides similar functionalities as higher, but it requires users to use its own stateless modules implemented in the library rather than patching general modules as in higher. Thus, it lacks the support for user's custom modules, limiting its applicability. learn2learn (Arnold et al., 2020) focuses on supporting meta learning. However, since meta-learning is strictly a bilevel problem, extending it beyond a two-level hierarchy is not straightforward. Finally, most existing libraries do not have systems support, such as data-parallel training, that could mitigate memory/compute bottlenecks.

# 7 CONCLUSION

In this paper, we aimed to help establish both mathematical and systems foundations for automatic differentiation in MLO. To this end, we devised a novel dataflow graph for MLO, upon which an automatic differentiation procedure is built, and additionally introduced BETTY, a software library with various systems support, that allows for easy programming of a wide range of MLO applications in a modular fashion. We showed that BETTY allows for scaling up to both larger models with many parameters, as well as to MLO programs with multiple dependent problems. As future work, we plan to extend BETTY to support additional algorithmic and systems features, such as best-response Jacobian algorithms for non-differentiable processes, and advanced memory optimization techniques like model-parallel training and CPU-offloading.

# ETHICS STATEMENT (OPTIONAL)

Multilevel optimization has the power to be a double-edged sword that can have both positive and negative societal impacts. For example, both 1) defense or attack in an adversarial game, and 2) decreasing or increasing bias in machine learning models, can all be formulated as MLO programs, depending on the goal of the uppermost optimization problem, which is defined by users. Thus, research in preventing malicious use cases of MLO is of high importance.

# REPRODUCIBILITY STATEMENT (OPTIONAL)

As one of main contributions of this work is a new software library for scalable multilevel optimization, all of the source code for the library and examples will be released open source with an Apache-2.0 License, including a full implementation of all MLO programs and experiments described in this paper. In addition, for reviewing purposes, we include our source code and easily runnable scripts for all experiments in the supplemental material of this submission.

# REFERENCES

Sebastien MR Arnold, Praateek Mahajan, Debajyoti Datta, Ian Bunner, and Konstantinos Saitas Zarkias. learn2learn: A library for meta-learning research. arXiv preprint arXiv:2008.12284, 2020.  
Yu Bai, Minshuo Chen, Pan Zhou, Tuo Zhao, Jason Lee, Sham Kakade, Huan Wang, and Caiming Xiong. How important is the train-validation split in meta-learning? In International Conference on Machine Learning, pp. 543-553. PMLR, 2021.  
Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-López, Fabian Pedregosa, and Jean-Philippe Vert. Efficient and modular implicit differentiation. arXiv preprint arXiv:2105.15183, 2021.  
Nicolas Couellan and Wenjuan Wang. On the convergence of stochastic bi-level gradient methods. Optimization, 2016.  
Hua Cui and Jie Bai. A new hyperparameters optimization method for convolutional neural networks. Pattern Recognition Letters, 125:828-834, 2019.  
Tristan Deleu, Tobias Würfl, Mandana Samiei, Joseph Paul Cohen, and Yoshua Bengio. Torchmeta: A Meta-Learning library for PyTorch, 2019. URL https://arxiv.org/abs/1909.06576. Available at: https://github.com/tristandeleu/pytorch-meta.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1126-1135. JMLR.org, 2017.  
Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning, pp. 1165-1173. PMLR, 2017.  
Bhanu Garg, Li Zhang, Pradyumna Sridhara, Ramtin Hosseini, Eric Xing, and Pengtao Xie. Learning from mistakes-a framework for neural architecture search. Proceedings of the AAAI Conference on Artificial Intelligence, 2022.  
Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning, pp. 3748-3758. PMLR, 2020.

Edward Grefenstette, Brandon Amos, Denis Yarats, Phu Mon Htut, Artem Molchanov, Franziska Meier, Douwe Kiela, Kyunghyun Cho, and Soumith Chintala. Generalized inner loop meta-learning. arXiv preprint arXiv:1910.01727, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729-9738, 2020.  
Xuehai He, Zhuo Cai, Wenlan Wei, Yichen Zhang, Luntian Mou, Eric Xing, and Pengtao Xie. Towards visual question answering on pathology images. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pp. 708-718, 2021.  
Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale framework for bilevel optimization: Complexity analysis and application to actor-critic. arXiv preprint arXiv:2007.05170, 2020.  
Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning, pp. 4882-4892. PMLR, 2021.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Vijay Konda and John Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems, 12, 1999.  
Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: differentiable architecture search. In ICLR, 2019.  
Risheng Liu, Yaohua Liu, Shangzhi Zeng, and Jin Zhang. Towards gradient-based bilevel optimization with non-convex followers and beyond. Advances in Neural Information Processing Systems, 34, 2021.  
Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In International Conference on Artificial Intelligence and Statistics, pp. 1540-1552. PMLR, 2020.  
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International conference on machine learning, pp. 2113-2122. PMLR, 2015.  
Athanasios Migdalas, Panos M Pardalos, and Peter Värbrand. Multilevel optimization: algorithms and applications, volume 20. Springer Science & Business Media, 1998.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.  
Aniruddh Raghu, Jonathan Lorraine, Simon Kornblith, Matthew McDermott, and David K Duvenaud. Meta-learning to improve pre-training. Advances in Neural Information Processing Systems, 34, 2021.  
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-16. IEEE, 2020.  
Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. Advances in neural information processing systems, 32, 2019.

Aravind Rajeswaran, Igor Mordatch, and Vikash Kumar. A game theoretic framework for model based reinforcement learning. In International conference on machine learning, pp. 7953-7963. PMLR, 2020.  
Alexander J Ratner, Christopher M De Sa, Sen Wu, Daniel Selsam, and Christopher Ré. Data programming: Creating large training sets, quickly. Advances in neural information processing systems, 29, 2016.  
Zhongzheng Ren, Raymond Yeh, and Alexander Schwing. Not all unlabeled data are equal: Learning to weight data in semi-supervised learning. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 21786-21797. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/f7ac67a9aa8d255282de7d11391e1b69-Paper.pdf.  
Ryo Sato, Mirai Tanaka, and Akiko Takeda. A gradient method for multilevel optimization. Advances in Neural Information Processing Systems, 34, 2021.  
Jun Shu, Qi Xie, Lixuan Yi, Qian Zhao, Sanping Zhou, Zongben Xu, and Deyu Meng. Meta-weight-net: Learning an explicit mapping for sample weighting. In Advances in Neural Information Processing Systems, pp. 1919-1930, 2019.  
Sai Ashish Somayajula, Linfeng Song, and Pengtao Xie. A multi-level optimization framework for end-to-end text augmentation. Transactions of the Association for Computational Linguistics, 10: 343-358, 2022.  
Felipe Petroski Such, Aditya Rawal, Joel Lehman, Kenneth Stanley, and Jeffrey Clune. Generative teaching networks: Accelerating neural architecture search by learning to generate synthetic training data. In International Conference on Machine Learning, pp. 9206-9216. PMLR, 2020.  
Kaihua Tang, Jianqiang Huang, and Hanwang Zhang. Long-tailed classification by keeping the good and removing the bad momentum causal effect. Advances in Neural Information Processing Systems, 33:1513-1524, 2020.  
Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5018-5027, 2017.  
Luis N Vicente and Paul H Calamai. Bilevel and multilevel programming: A bibliography review. Journal of Global optimization, 5(3):291-306, 1994.  
Yulin Wang, Jiayi Guo, Shiji Song, and Gao Huang. Meta-semi: A meta-learning approach for semi-supervised learning. CoRR, abs/2007.02394, 2020. URL https://arxiv.org/abs/2007.02394.  
Pengtao Xie and Xuefeng Du. Performance-aware mutual knowledge distillation for improving neural architecture search. CVPR, 2022.  
Jieyu Zhang, Yue Yu, Yinghao Li, Yujing Wang, Yaming Yang, Mao Yang, and Alexander Ratner. Wrench: A comprehensive benchmark for weak supervision. arXiv preprint arXiv:2109.11377, 2021a.  
Miao Zhang, Steven W Su, Shirui Pan, Xiaojun Chang, Ehsan M Abbasnejad, and Reza Haffari. idarts: Differentiable architecture search with stochastic implicit gradients. In International Conference on Machine Learning, pp. 12557-12566. PMLR, 2021b.  
Guoqing Zheng, Ahmed Hassan Awadallah, and Susan T. Dumais. Meta label correction for learning with weak supervision. CoRR, abs/1911.03809, 2019. URL http://arxiv.org/abs/1911.03809.
