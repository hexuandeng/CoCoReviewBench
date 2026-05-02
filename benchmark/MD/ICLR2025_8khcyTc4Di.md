# META-LEARNING NEURAL PROCEDURAL BIASES

Anonymous authors

Paper under double-blind review

# ABSTRACT

The goal of few-shot learning is to generalize and achieve high performance on new unseen learning tasks, where each task has only a limited number of examples available. Gradient-based meta-learning attempts to address this challenging task by learning how to learn new tasks by embedding inductive biases informed by prior learning experiences into the components of the learning algorithm. In this work, we build upon prior research and propose Neural Procedural Bias Meta-Learning (NPBML), a novel framework designed to meta-learn task-adaptive procedural biases. Our approach aims to consolidate recent advancements in meta-learned initializations, optimizers, and loss functions by learning them simultaneously and making them adapt to each individual task to maximize the strength of the learned inductive biases. This imbues each learning task with a unique set of procedural biases which is specifically designed and selected to attain strong learning performance in only a few gradient steps. The experimental results show that by meta-learning the procedural biases of a neural network, we can induce strong inductive biases towards a distribution of learning tasks, enabling robust learning performance across many well-established few-shot learning benchmarks.

# 1 INTRODUCTION

Humans have an exceptional ability to learn new tasks from only a few example instances. We can often quickly adapt to new domains effectively by building upon and utilizing past experiences of related tasks, leveraging only a small amount of information about the target domain. The field of meta-learning (Schmidhuber, 1987; Vanschoren, 2018; Peng, 2020; Hospedales et al., 2022) explores how deep learning techniques, which often require thousands or even millions of observations to achieve competitive performance, can acquire such a capability. In meta-learning, the learning process is often framed as a bilevel optimization problem (Bard, 2013; Maclaurin et al., 2015; Grefenstette et al., 2019; Lorraine et al., 2020). The outer optimization aims to learn the underlying regularities across a distribution of related tasks and embed them into the inductive biases of a learning algorithm. Consequently, in the inner optimization, the learning algorithm is utilized to quickly adapt to new learning tasks using only a few example instances.

Model-Agnostic Meta-Learning (MAML) (Finn et al., 2017) and its variants (Nichol & Schulman, 2018; Rajeswaran et al., 2019; Song et al., 2019; Triantafillou et al., 2020), are a popular approach to meta-learning. In MAML, the outer optimization aims to learn the underlying regularities across a set of related tasks and embed them into a shared parameter initialization. This initialization is then used in the inner optimization's learning algorithm to encourage fast adaptation to new tasks. While successful, these methods resort to simple gradient descent using the cross-entropy loss for classification or squared loss for regression for the inner learning algorithm. Consequently, subsequent research has extended MAML to meta-learn additional components, such as the learning rate (Behl et al., 2019; Baik et al., 2020), gradient-based optimizer (Li et al., 2017; Lee & Choi, 2018; Simon et al., 2020; Flennerhag et al., 2020; Kang et al., 2023), loss function (Antoniou & Storkey, 2019), and more (Antoniou et al., 2019; Baik et al., 2023). This enables the meta-learning algorithm to induce stronger inductive biases on the learning algorithm, further enhancing performance.

In this paper, we propose Neural Procedural Bias Meta-Learning (NPBML), a novel gradient-based framework for meta-learning task-adaptive procedural biases for deep neural networks. Procedural biases are the subset of inductive biases that determine the order of traversal over the search space (Gordon & Desjardins, 1995), they play a central determining role in the convergence, sample efficiency, and generalization of a learning algorithm. As we will show, the procedural biases are

![](images/ed68f8ded157cf24da0e5386de88729147091b02de4e92e9b1be4b374aa56c6c.jpg)  
Figure 1: In NPBML, the procedural biases of a deep neural network are meta-learned. This involves meta-learning three key components: the loss function (left), the parameter initialization (center), and the optimizer (right). By meta-learning these components, a strong inductive bias towards fast adaptation can be induced into the learning algorithm.

primarily encoded into three fundamental components of a learning algorithm: the loss function, the optimizer, and the parameter initialization. These components define the geometry of the loss landscape, determine the starting point in this space, and guide the optimization process towards the optimum, respectively, as visualized in Figure 1. Therefore, we aim to meta-learn these three components to maximize the learning performance when using only a few gradient steps.

To achieve this ambitious goal, we first consolidate three related research areas into one unified end-to-end framework: MAML-based learned initializations (Finn et al., 2017), preconditioned gradient descent methods (Lee & Choi, 2018; Flennerhag et al., 2020; Kang et al., 2023), and meta-learned loss functions (Antoniou & Storkey, 2019; Baik et al., 2021; Bechtle et al., 2021; Raymond et al., 2023a,b). We then demonstrate how these meta-learned components can be made task-adaptive through feature-wise linear modulation (FiLM) (Perez et al., 2018) to facilitate downstream task-specific specialization towards each task. The proposed framework is highly flexible and general. As we will show, many existing gradient-based meta-learning approaches arise as special cases of NPBML. To validate the effectiveness of NPBML, we empirically evaluate our proposed algorithm on four well-established few-shot learning benchmarks. The results show that NPBML consistently outperforms many state-of-the-art gradient-based meta-learning algorithms.

# 2 BACKGROUND

In few-shot meta-learning, we are given access to a collection of tasks  $\{\mathcal{T}_1,\mathcal{T}_2,\ldots \}$ , otherwise known as a meta-dataset, where each task is assumed to be drawn from a task distribution  $p(\mathcal{T})$ . Each task  $\mathcal{T}_i$  contains a support set  $\mathcal{D}^S$ , and a query set  $\mathcal{D}^Q$  (i.e. a training set and a testing set), where  $\mathcal{D}^S\cap \mathcal{D}^Q = \emptyset$ . Each of these sets contains a set of input-output pairs  $\{(x_1,y_1),(x_2,y_2),\dots \}$ . Let  $x\in X$  and  $y\in Y$  denote the inputs and outputs, respectively. In few-shot meta-learning, the goal is to learn a model of the form  $f_{\theta}(x):X\to Y$ , where  $\theta$  are the model parameters. The primary challenge of few-shot learning is that  $f_{\theta}$  must be able to quickly adapt to any new task  $\mathcal{T}_i\sim p(\mathcal{T})$  given only a very limited number of instances. For example, in an  $N$ -way  $K$ -shot few-shot classification task,  $f_{\theta}$  is only given access to  $K$  labeled examples of  $N$  distinct classes.

# 2.1 MODEL AGNOSTIC META-LEARNING

MAML (Finn et al., 2017) is a highly influential and seminal method for gradient-based few-shot meta-learning. In MAML, the outer learning objective aims to meta-learn a shared parameter initialization  $\theta$  over a distribution of related tasks  $p(\mathcal{T})$ . This shared initialization embeds prior knowledge learned from past learning experiences into the learning algorithm such that when a new unseen task is sampled fast adaptation can occur. The outer optimization prototypically occurs by minimizing the sum of the final losses  $\sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}^{meta}$  on the query set at the end of each task's learning trajectory using gradient descent as follows:

$$
\boldsymbol {\theta} _ {n e w} = \boldsymbol {\theta} - \eta \nabla_ {\boldsymbol {\theta}} \sum_ {\mathcal {T} _ {i} \sim p (\mathcal {T})} \left[ \mathcal {L} ^ {m e t a} \left(\mathcal {D} _ {i} ^ {Q}, \theta_ {i, j} (\boldsymbol {\theta})\right) \right] \tag {1}
$$

where  $\theta_{i,j}(\pmb{\theta})$  refers to the adapted model parameters on the  $i^{th}$  task at the  $j^{th}$  iterations of the inner update rule  $\mathrm{U}^{MAML}$ , and  $\eta$  is the meta learning rate. The inner optimization for each task starts at the meta-learned parameter initialization  $\pmb{\theta}$  and applies  $\mathrm{U}^{MAML}$  to the parameters  $J$  times:

$$
\theta_ {i, j} (\boldsymbol {\theta}) = \boldsymbol {\theta} - \alpha \sum_ {j = 0} ^ {J - 1} \left[ \mathrm {U} ^ {M A M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} (\boldsymbol {\theta})\right) \right]. \tag {2}
$$

In the original version of MAML, the inner update rule  $\mathrm{U}^{MAML}$  resorts to simple Stochastic Gradient Descent (SGD) minimizing the loss  $\mathcal{L}^{\text{base}}$ , typically set to the cross-entropy or squared loss, with a fixed learning rate  $\alpha$  across all tasks

$$
\mathrm {U} ^ {M A M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} (\boldsymbol {\theta})\right) := \theta_ {i, j} - \alpha \nabla_ {\theta_ {i, j}} \left[ \mathcal {L} ^ {\text {b a s e}} \left(\mathcal {D} _ {i} ^ {S}, \theta_ {i, j}\right) \right]. \tag {3}
$$

This approach assumes that all tasks in  $p(\mathcal{T})$  should use the same fixed learning rule  $U$  in the inner optimization. Consequently, this greatly limits the performance that can be achieved when taking only a small number of gradient steps with few labeled samples available.

# 3 NEURAL PROCEDURAL BIAS META-LEARNING

In this work, we propose Neural Procedural Bias Meta-Learning (NPBML), a novel framework that replaces the fixed inner update rule in MAML, i.e., Equation (3), with a meta-learned task-adaptive learning rule. This modifies the inner update rule in three key ways:

1. An optimizer is meta-learned by leveraging the paradigm of preconditioned gradient descent. This involves meta-learning a parameterized preconditioning matrix  $P_{\omega}$  with metaparameters  $\omega$  to warp the gradients of SGD.  
2. The conventional loss function  $\mathcal{L}^{base}$  (i.e. the standard cross-entropy or squared loss) used in the inner optimization is replaced with a meta-learned loss function  $\mathcal{M}_{\phi}$ , where  $\phi$  are the meta-parameters.  
3. The meta-learned initialization, optimizer, and loss function are adapted to each new task using Feature-Wise Linear Modulation  $FiLM_{\psi}$ , a general-purpose preconditioning method that has learnable meta-parameters  $\psi$ .

For clarity, we provide a high-level overview of NPBML and contrast it to MAML, before expanding on each of the new components in Sections 3.2, 3.3, and 3.4, respectively. Additionally, pseudocode for the outer and inner optimizations is provided in Algorithm 1 and 2, respectively, in Appendix A.

# 3.1 OVERVIEW

The central goal of NPBML is to meta-learn a task-adaptive parameter initialization, optimizer, and loss function. This changes the outer optimization previously seen in Equation (1) to the following, where  $\Phi = \{\theta ,\omega ,\phi ,\psi \}$  refers to the set of meta-parameters:

$$
\Phi_ {n e w} = \Phi - \eta \nabla_ {\Phi} \sum_ {\mathcal {T} _ {i} \sim p (\mathcal {T})} \left[ \mathscr {L} ^ {m e t a} \left(\mathscr {D} _ {i} ^ {Q}, \theta_ {i, j} (\Phi)\right) \right]. \tag {4}
$$

Unlike MAML which employs a fixed update rule  $\mathrm{U}^{MAML}$  for all tasks, simple SGD using  $\mathcal{L}^{base}$ , NPBML uses a fully meta-learned update rule

$$
\theta_ {i, j} (\boldsymbol {\Phi}) = \boldsymbol {\theta} (\boldsymbol {\psi}) - \alpha \sum_ {j = 0} ^ {J - 1} \left[ \mathrm {U} ^ {N P B M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} (\boldsymbol {\Phi})\right) \right] \tag {5}
$$

which adjusts  $\theta_{i,j}$  in the direction of the negative gradient of a meta-learned loss function  $\mathcal{M}_{\phi}$ . Additionally, the gradient is warped via a meta-learned preconditioning matrix  $P_{\omega}$  as follows:

$$
\mathrm {U} ^ {N P B M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} (\boldsymbol {\Phi})\right) := \theta_ {i, j} - \alpha P _ {(\boldsymbol {\omega}, \boldsymbol {\psi})} \nabla_ {\theta_ {i, j}} \left[ \mathcal {M} _ {(\boldsymbol {\phi}, \boldsymbol {\psi})} \left(\mathcal {D} _ {i} ^ {S}, \theta_ {i, j}\right) \right] \tag {6}
$$

where both  $\mathcal{M}_{\phi}$  and  $P_{\omega}$  are adapted to each task using  $F i L M_{\psi}$ . This new task-adaptive learning rule empowers each task with a unique set of procedural biases enabling strong and robust learning performance on new unseen tasks in  $p(\mathcal{T})$  using only a few gradient steps as depicted in Figure 2.

![](images/038061a18f1656ded1beafcf324e8a3f2810a6df92f9173f581663f90636a500.jpg)  
Figure 2: In MAML, the update rule  $\mathrm{U}^{MAML}$  optimizes the base model parameters from a shared initialization using simple SGD minimizing  $\mathcal{L}^{base}$ . In contrast, NPBML adapts the model parameters from a task-adapted initialization using  $\mathrm{U}^{NPBML}$ , a task-adaptive update rule employing a meta-learned preconditioning matrix  $P_{\omega}$  and loss function  $\mathcal{M}_{\phi}$ .

# 3.2 META-LEARNED OPTIMIZER

In NPBML, the paradigm of Preconditioned Gradient Descent (PGD) (Li et al., 2017; Lee & Choi, 2018; Park & Oliva, 2019; Flennerhag et al., 2020; Simon et al., 2020; Kang et al., 2023) is employed to meta-learn a gradient preconditioner  $P_{\omega}$  that rescales the geometry of the parameter space by modifying the gradient descent update rule as follows:

$$
\theta_ {n e w} = \theta - \alpha P _ {\omega} \nabla_ {\theta} \mathcal {M} _ {(\phi , \psi)}. \tag {7}
$$

In this work, we take inspiration from T-Nets (Lee & Choi, 2018) and insert linear projection layers  $\omega$  into our model  $f_{(\theta ,\omega ,\psi)}$ . The model, composed of an encoder  $z$  and a classification head  $h$ ,

$$
f _ {(\boldsymbol {\theta}, \omega , \psi)} = z _ {(\boldsymbol {\theta}, \omega , \psi)} \circ h _ {\theta} \tag {8}
$$

is interleaved with linear projection layers  $\omega$  between each layer in the  $L$  layer encoder (where  $\sigma$  refers to the non-linear activation functions):

$$
z _ {(\boldsymbol {\theta}, \boldsymbol {\omega}, \psi)} (x _ {i}) = \sigma^ {(L)} \left(\boldsymbol {\theta} ^ {(L)} \boldsymbol {\omega} ^ {(L)} \left(\dots \sigma^ {(1)} \left(\boldsymbol {\omega} ^ {(1)} \boldsymbol {\theta} ^ {(1)} x\right) \dots\right)\right). \tag {9}
$$

As described in Equations (4)-(6),  $\omega$  is meta-learned in the outer loop and held fixed in the inner loop such that preconditioning of the gradients occurs. This form of precondition defines  $P_{\omega}$  as a block-diagonal matrix, where each block is defined by the expression  $(\boldsymbol{\omega}\boldsymbol{\omega}^{\mathrm{T}})$ , as shown in (Lee & Choi, 2018). For a simple model where  $f_{(\theta ,\omega)}(x) = \omega \theta x$ , the update rule becomes:

$$
\theta_ {n e w} = \theta - \alpha \left(\boldsymbol {\omega} \boldsymbol {\omega} ^ {\mathsf {T}}\right) \nabla_ {\theta} \mathcal {M} _ {(\phi , \psi)}. \tag {10}
$$

We leverage this style of parameterization for  $P_{\omega}$  due to its relative simplicity and high expressive power. However, we emphasize that the NPBML framework is highly general, and other forms of preconditioning, such as those presented in (Lee & Choi, 2018; Park & Oliva, 2019; Flennerhag et al., 2020; Simon et al., 2020), could be used instead.

# 3.3 META-LEARNED LOSS FUNCTION

Unlike MAML, which defines the inner loss function  $\mathcal{L}^{base}$  to be the cross-entropy or squared loss for all tasks, NPBML uses a meta-learned loss function  $\mathcal{M}_{(\phi, \psi)}$  that is learned in the outer optimization. In contrast to handcrafted loss functions, which typically consider only the ground truth label  $y$  and the model predictions  $f_{(\theta, \omega, \psi)}(x)$ , meta-learned loss functions can, in principle, be conditioned on any task-related information (Bechtle et al., 2021; Baik et al., 2021; Raymond, 2024). In NPBML, the meta-learned loss function is conditioned on three distinct sources of task-related information, which are subsequently processed by three small feed-forward neural networks:

$$
\mathcal {M} _ {(\phi , \psi)} = \mathcal {L} _ {(\phi , \psi)} ^ {S} + \mathcal {L} _ {(\phi , \psi)} ^ {Q} + \mathcal {R} _ {(\phi , \psi)}. \tag {11}
$$

![](images/cc523841ef5be30a5e8f8c4aef2a3eb53c5347f252be23cf4fcc01b083f3fec9.jpg)  
Figure 3: An example of a two-layer convolutional neural network in NPBML, where layers  $\theta^{(1)}$  and  $\theta^{(2)}$ , are interleaved with warp preconditioning layers  $\omega^{(1)}$  and  $\omega^{(2)}$ . Both types of layers are modulated in the inner loop using feature-wise linear modulation layers to induce task adaptation.

![](images/c11d011ff7127270e5767773f19b5add1bc7395269690b3ba231e65e64855146.jpg)  
Figure 4: An example of a meta-learned loss function in NPBML, represented as a composition of feed-forward (linear) layers. These layers are modulated using feature-wise linear modulation, resulting in a task-adaptive meta-learned loss function.

First,  $\mathcal{L}^S:\mathbb{R}^{2N + 1}\to \mathbb{R}^1$  is an inductive loss function conditioned on task-related information derived from the support set; namely, the one-hot encoded ground truth target and model predictions, and the corresponding loss calculated using  $\mathcal{L}^{base}$ . Next,  $\mathcal{L}^Q:\mathbb{R}^{2N + 1}\to \mathbb{R}^1$  is a transductive loss function conditioned on task-related information derived from the query set. Here, we give  $\mathcal{L}^Q$  access to the model predictions on the query set, embeddings (i.e., relation scores) from a pre-trained relation network (Sung et al., 2018), and the corresponding loss between the model predictions and embeddings using  $\mathcal{L}^{base}$ . Note that similar embedding functions have previously been used in (Rusu et al., 2019; Antoniou & Storkey, 2019). Finally, the adapted model parameters  $\theta_{i,j}$  are used as inputs to meta-learn a weight regularizer  $\mathcal{R}:\mathbb{R}^{4L}\to \mathbb{R}^1$ . To improve efficiency, we condition  $\mathcal{R}$  on the mean, standard deviation, L1, and L2 norm of each layer's weights, as opposed to  $\theta_{i,j}$  directly.

# 3.4 TASK-ADAPTIVE MODULATION

Although all tasks in few-shot learning are assumed to be sampled from the same task distribution  $p(\mathcal{T})$ , the optimal parameter initialization, optimizer, and loss function may differ between tasks. Therefore, in NPBML the meta-learned components  $\Phi$  are modulated for each new task, providing each task with a unique set of task-adaptive procedural biases. To achieve this, Feature-wise Linear Modulation (FiLM) layers (Perez et al., 2018; Dumoulin et al., 2018) are inserted into both the encoder  $z_{(\theta ,\omega ,\psi)}$  and the meta-learned loss function  $M_{(\phi ,\psi)}$  as shown in Figures 3 and 4. FiLM $\psi$  is defined as follows, where  $\gamma$  and  $\beta$  are the scaling and shifting vectors, respectively, and  $\psi$  are the meta-learnable FiLM parameters:

$$
F i L M _ {\psi} (x) = \left(\gamma_ {\psi} (x) + \mathbf {1}\right) \odot x + \beta_ {\psi} (x). \tag {12}
$$

Affine transformations conditioned on some input have become increasingly popular, being used by several few-shot learning works to make the learned component adaptive (Oreshkin et al., 2018; Jiang et al., 2018; Vuorio et al., 2019; Zintgraf et al., 2019; Baik et al., 2021; 2023). Furthermore, FiLM layers can help alleviate issues related to batch normalization (Ioffe & Szegedy, 2015), which have been empirically observed to cause training instability due to different distributions of features being passed through the same model in few-shot learning (De Vries et al., 2017; Antoniou et al., 2019). In our work, we have found that conditioning the FiLM on the output activations of the previous layers is an effective way to achieve task adaptability. This form of conditioning is in essence a simplified version of that used in CNAPs (Requeima et al., 2019); however, we have omitted the use of global embeddings as we found it was not necessary for our method.

# 3.5 INITIALIZATION

Due to the large number of learnable meta-parameters, initialization becomes an important and necessary aspect to consider. Here we detail how to initialize each of the meta-learned components, i.e.,  $\Phi_0 = \{\pmb {\theta}_0,\pmb {\omega}_0,\phi_0,\psi_0\}$ , in NPBML. Firstly, we pre-train the encoder weights  $\pmb{\theta}_{0}$  prior to meta-learning, following many recent methods in few-shot learning (Rusu et al., 2019; Qiao et al., 2018; Requeima et al., 2019; Ye et al., 2020; Ye & Chao, 2021), see Appendix A.3 for more details. For the linear projection layers  $\omega$ , we leverage the fact that in PGD, setting  $P_{\omega}$  to the identity  $\pmb{I}$  recovers SGD. Therefore, we set  $\forall l\in \{1,\dots ,L\} :\omega^{(l)} = I$ ; note for convolutional layers this corresponds to Dirac initialization. Regarding the meta-learned loss function  $\mathcal{M}_{(\phi ,\psi)}$ , the weights  $\phi$  at the start of meta-training are randomly initialized  $\phi_0\sim \mathcal{N}(0,1e^{-2})$ ; therefore, the  $\mathbb{E}[\mathcal{M}_{(\phi_0,\psi_0)}] = 0$ , assuming an identity output activation. Consequently, the definition of the meta-learned loss function in Equation (11) can be modified to

$$
\mathcal {M} _ {(\phi , \psi)} = \mathcal {L} ^ {\text {b a s e}} + \mathcal {L} _ {(\phi , \psi)} ^ {S} + \mathcal {L} _ {(\phi , \psi)} ^ {Q} + \mathcal {R} _ {(\phi , \psi)} \tag {13}
$$

such that the meta-learned loss function approximately recovers the base loss function at the start of meta-training, i.e.,  $\mathcal{M}_{(\phi_0,\psi_0)}\approx \mathcal{L}^{base}$ . Finally, the FiLM layers in NPBML are initialized using a similar strategy taking advantage of the fact that when  $\psi_0\sim \mathcal{N}(0,1e - 2)$  the  $\mathbb{E}[\gamma_{\psi_0}(x)] = \mathbb{E}[\beta_{\psi_0}(x)] = 0$ ; consequently,  $FiLM_{\psi_0}(x)\approx x$ . When initialized in this manner, the update rule for NPBML at the start of meta-training closely approximates the update rule of MAML.

$$
\left. \right. \mathrm {U} ^ {M A M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} \left(\boldsymbol {\theta} _ {\mathbf {0}}\right)\right) \approx \mathbb {E} \left[ \mathrm {U} ^ {N P B M L} \left(\mathcal {T} _ {i}, \theta_ {i, j} \left(\boldsymbol {\Phi} _ {\mathbf {0}}\right)\right)\right] \tag {14}
$$

# 4 IMPLICIT META-LEARNING

In NPBML, the parameter initialization, gradient-based optimizer, and loss function are explicitly meta-learned in the outer loop. Critically, we make a novel observation that many other key procedural biases are also implicitly learned by meta-learning these three fundamental components (hence the name given to our algorithm). For example, consider the scalar learning rate  $\alpha$ , which is implicitly meta-learned since the following equality holds:

$$
\exists \alpha \exists \phi : \theta_ {i, j} - \alpha \nabla_ {\theta_ {i, j}} \mathcal {L} ^ {\text {b a s e}} \approx \theta_ {i, j} - \nabla_ {\theta_ {i, j}} \mathcal {M} _ {\phi}, \tag {15}
$$

and if  $\phi$  is made to adapt on each inner step as done in (Baik et al., 2021; Raymond et al., 2023b), then by extension NPBML also learns a learning rate schedule. Another straightforward related observation is that NPBML implicitly learns a layer-wise learning rate  $\{\alpha^{(1)},\dots,\alpha^{(L)}\}$ , since for each block  $\{\pmb{\omega}^{(1)},\dots,\pmb{\omega}^{(L)}\}$  in the block diagonal preconditioning matrix  $P_{\omega}$  the following holds:

$$
\forall l \left(\exists \omega^ {(l)} \exists \alpha^ {(l)}\right): \left(\omega^ {(l)} \left(\omega^ {(l)}\right) ^ {\top}\right) \nabla_ {\theta^ {(l)}} \mathcal {M} _ {\phi} \approx \alpha^ {(l)} \nabla_ {\theta^ {(l)}} \mathcal {M} _ {\phi}. \tag {16}
$$

There are also less obvious connections that can be drawn. For example, through a transitive relationship, NPBML implicitly learns early stopping when the implicitly learned learning rate approaches zero, as discussed in (Baydin et al., 2018). Furthermore, since there is a linear scaling rule between the batch size and the learning rate (Smith et al., 2017; Smith & Le, 2017; Goyal et al., 2017), NPBML implicitly learns the regularization behavior of the batch size hyperparameter. Another non-trivial example is label smoothing regularization (Müller et al., 2019), which, as proven in (Gonzalez & Miikkulainen, 2020), can be implicitly induced when meta-learning a loss function.

# 5 RELATED WORK

Meta-learning approaches to few-shot learning aim to equip models with the ability to quickly adapt to new tasks given only a limited number of examples by leveraging prior learning experiences across a distribution of related tasks. These approaches are commonly partitioned into three categories: (1) metric-based methods, which aim to learn a similarity metric for efficient class differentiation (Koch et al., 2015; Vinyals et al., 2016; Sung et al., 2018; Snell et al., 2017); (2) memory-based methods, which utilize architectures that store training examples in memory or directly encode fast adaptation algorithms in the model weights (Santoro et al., 2016; Ravi & Larochelle, 2017); and (3) optimization-based methods, which aim to learn an optimization algorithm specifically designed for

fast adaptation and few-shot learning (Finn et al., 2017). This work explored the latter approach by meta-learning a gradient update rule.

MAML (Finn et al., 2017), a highly flexible task and model-agnostic method for meta-learning a parameter initialization from which fast adaptation can occur. Many follow-up works have sought to enhance MAML's performance by addressing limitations in the outer-optimization algorithm, such as the memory and compute efficiency (Nichol & Schulman, 2018; Rajeswaran et al., 2019; Raghu et al., 2019; Oh et al., 2020), or the meta-level overfitting (Rusu et al., 2019; Flennerhag et al., 2018). Relatively fewer works have focused on enhancing the inner-optimization update rule, as is done in our work. Most MAML variants continue to use an inner update rule consisting of SGD with a fixed learning rate minimizing a loss function such as the cross-entropy or squared loss.

Of the works that have explored improving the inner update rule, the vast majority focus on improving the optimizer. For example, early MAML-based methods such as (Behl et al., 2019; Antoniou et al., 2019; Li et al., 2017) explored meta-learning the scalar, layer-wise, and parameter-wise learning rates, respectively. More recent methods have explored more powerful parameterization for the meta-learned optimizer through the utilization of preconditioned gradient descent methods (Lee & Choi, 2018; Park & Oliva, 2019; Flennerhag et al., 2020; Simon et al., 2020; Kang et al., 2023), which rescale the geometry of the parameter space by modifying the update rule with a learned preconditioning matrix. While these methods have advanced MAML-based few-shot learning, they often lack a task-adaptive property, falsely assuming that all tasks should use the same optimizer.

A small number of recent works have also investigated replacing the inner loss function (e.g., cross-entropy loss) with a meta-learned loss function. In (Antoniou & Storkey, 2019), a fully transductive loss function represented as a dilated convolutional neural network is meta-learned. Meanwhile, in (Baik et al., 2021), a set of loss functions and task adapters are meta-learned for each step taken in the inner optimization. Although these approaches have shown a lot of promise, their potential has yet to be fully realized, as they have not yet been meta-learned in tandem with the optimizer as we have done in this work.

# 6 EXPERIMENTAL EVALUATION

In this section, we evaluate the performance of the proposed method on a set of well-established few-shot learning benchmarks. The experimental evaluation aims to answer the following key questions: (1) Can NPBML perform well across a diverse range of few-shot learning tasks? (2) Do the novel components meta-learned in NPBML individually enhance performance? (3) To what extent does each component synergistically contribute to the overall performance of the proposed algorithm?

# 6.1 RESULTS AND ANALYSIS

To evaluate the performance of NPBML, experiments are performed on four well-established few-shot learning datasets: mini-Imagenet (Ravi & Larochelle, 2017), tiered-ImageNet (Ren et al., 2018), CIFAR-FS (Bertinetto et al., 2018), and FC-100 (Oreshkin et al., 2018). For each dataset, experiments are performed using both 5-way 1-shot and 5-way 5-shot configurations. Results are also reported on both the 4-CONV (Finn et al., 2017; Zintgraf et al., 2019; Flennerhag et al., 2020; Kang et al., 2023) and ResNet-12 (He et al., 2016; Baik et al., 2020; 2021) network architectures. The full details of all experiments, including a comprehensive description of all datasets, models, and training hyperparameters, can be found in Appendix A. The code for our experiments can be found at https://github.com/\*redacted\*

# 6.1.1 MINI-IMAGENET AND TIERED-IMAGENET

We first assess the performance of NPBML and compare it to a range of MAML-based few-shot learning methods on two popular ImageNet derivatives (Deng et al., 2009): mini-ImageNet (Ravi & Larochelle, 2017) and tiered-ImageNet (Ren et al., 2018). The results, presented in Table 1, demonstrate that the proposed method NPBML, which uses a fully meta-learned update rule in the inner optimization, significantly improves upon the performance of MAML-based few-shot learning methods. The proposed method achieves higher meta-testing accuracy in the 1-shot and 5-shot settings using both low-capacity (4-CONV) and high-capacity (ResNet-12) models.

Table 1: Few-shot classification meta-testing accuracy on 5-way 1-shot and 5-way 5-shot mini-ImageNet and tiered-ImageNet where  $\pm$  represents the  $95\%$  confidence intervals.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Base Learner</td><td colspan="2">mini-ImageNet (5-way)</td><td colspan="2">tiered-ImageNet (5-way)</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>\(MAML^1\)</td><td>4-CONV</td><td>48.70±1.84%</td><td>63.11±0.92%</td><td>50.98±0.26%</td><td>66.25±0.19%</td></tr><tr><td>\(MetaSGD^2\)</td><td>4-CONV</td><td>50.47±1.87%</td><td>64.03±0.94%</td><td>-</td><td>-</td></tr><tr><td>\(T-Net^3\)</td><td>4-CONV</td><td>50.86±1.82%</td><td>-</td><td>-</td><td>-</td></tr><tr><td>\(MAML++^4\)</td><td>4-CONV</td><td>52.15±0.26%</td><td>68.32±0.44%</td><td>-</td><td>-</td></tr><tr><td>\(SCA^5\)</td><td>4-CONV</td><td>54.84±0.99%</td><td>71.85±0.53%</td><td>-</td><td>-</td></tr><tr><td>\(WarpGrad^7\)</td><td>4-CONV</td><td>52.30±0.80%</td><td>68.40±0.60%</td><td>57.20±0.90%</td><td>74.10±0.70%</td></tr><tr><td>\(ModGrad^8\)</td><td>4-CONV</td><td>53.20±0.86%</td><td>69.17±0.69%</td><td>-</td><td>-</td></tr><tr><td>\(MeTAL^9\)</td><td>4-CONV</td><td>52.63±0.37%</td><td>70.52±0.29%</td><td>54.34±0.31%</td><td>70.40±0.21%</td></tr><tr><td>\(ALFA^{10}\)</td><td>4-CONV</td><td>50.58±0.51%</td><td>69.12±0.47%</td><td>53.16±0.49%</td><td>70.54±0.46%</td></tr><tr><td>\(GAP^{11}\)</td><td>4-CONV</td><td>54.86±0.85%</td><td>71.55±0.61%</td><td>57.60±0.93%</td><td>74.90±0.68%</td></tr><tr><td>NPBML</td><td>4-CONV</td><td>57.49±0.83%</td><td>75.01±0.64%</td><td>64.24±0.97%</td><td>79.17±0.71%</td></tr><tr><td>\(MAML^1\)</td><td>ResNet-12</td><td>58.60±0.42%</td><td>69.54±0.38%</td><td>59.82±0.41%</td><td>73.17±0.32%</td></tr><tr><td>\(MC^6\)</td><td>WRN-28-10</td><td>-</td><td>-</td><td>64.40±0.10%</td><td>80.21±0.10%</td></tr><tr><td>\(ModGrad^8\)</td><td>WRN-28-10</td><td>-</td><td>-</td><td>65.72±0.21%</td><td>81.17±0.20%</td></tr><tr><td>\(MeTAL^9\)</td><td>ResNet-12</td><td>59.64±0.38%</td><td>76.20±0.19%</td><td>63.89±0.43%</td><td>80.14±0.40%</td></tr><tr><td>\(ALFA^{10}\)</td><td>ResNet-12</td><td>59.74±0.49%</td><td>77.96±0.41%</td><td>64.62±0.49%</td><td>82.48±0.38%</td></tr><tr><td>NPBML</td><td>ResNet-12</td><td>61.59±0.80%</td><td>78.18±0.60%</td><td>72.22±0.96%</td><td>85.41±0.61%</td></tr></table>

$^{1}$  (Finn et al., 2017)  ${}^{2}$  (Li et al., 2017)  ${}^{3}$  (Lee & Choi, 2018)  ${}^{4}$  (Antoniou et al., 2019)  ${}^{5}$  (Antoniou & Storkey, 2019)  ${}^{6}$  (Park & Oliva, 2019)  ${}^{7}$  (Flennerhag et al., 2020)  ${}^{8}$  (Simon et al., 2020)  ${}^{9}$  (Baik et al., 2021)  ${}^{10}$  (Baik et al., 2023)  ${}^{11}$  (Kang et al., 2023)

In contrast to PGD methods Meta-SGD, T-Net, WarpGrad, ModGrad, ALFA, and GAP, which metalearn an optimizer alongside the parameter initialization, NPBML shows clear gains in generalization performance. This improvement is also evident when compared to SCA and MeTAL, which replace the inner optimization's loss function with a meta-learned loss function. These results empirically demonstrate that meta-learning an optimizer and loss function are complementary and orthogonal approaches to improving MAML-based few-shot learning methods.

On tiered-ImageNet, the larger of the two datasets, we find that the difference between NPBML and its competitors is even more pronounced than on mini-ImageNet. This result suggests that when given enough data, NPBML can learn highly expressive inner update rules that significantly enhances few-shot learning performance. However, meta-overfitting can occur on smaller datasets, necessitating regularization techniques as discussed in Appendix A. Alternatively, we conjecture that less expressive representations for  $P_{\omega}$  would also reduce meta-overfitting.

# 6.1.2 CIFAR-FS AND FC-100

Next, we further validate the effectiveness of NPBML on two popular CIFAR-100 derivatives (Krizhevsky & Hinton, 2009): CIFAR-FS (Ravi & Larochelle, 2017) and FC-100 (Ren et al., 2018). The results, presented in Table 2, show that NPBML continues to achieve strong and robust generalization performance across all settings and models. These results are particularly impressive, given that both MeTAL and ALFA ensemble the top 5 performing models from the same run, which significantly increases the model size and capacity. These experimental results reinforce our claim that meta-learning a task-adaptive update rule is an effective approach to improving the performance of MAML-based few-shot learning algorithms.

Table 2: Few-shot classification meta-testing accuracy on 5-way 1-shot and 5-way 5-shot CIFAR-FS and FC-100 where  $\pm$  represents the  $95\%$  confidence intervals.  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Base Learner</td><td colspan="2">CIFAR-FS (5-way)</td><td colspan="2">FC-100 (5-way)</td></tr><tr><td>1-shot</td><td>5-shot</td><td>1-shot</td><td>5-shot</td></tr><tr><td>MAML¹</td><td>4-CONV</td><td>57.63±0.73%</td><td>73.95±0.84%</td><td>35.89±0.72%</td><td>49.31±0.47%</td></tr><tr><td>BOIL²</td><td>4-CONV</td><td>58.03±0.43%</td><td>73.61±0.32%</td><td>38.93±0.45%</td><td>51.66±0.32%</td></tr><tr><td>MeTAL³</td><td>4-CONV</td><td>59.16±0.56%</td><td>74.62±0.42%</td><td>37.46±0.39%</td><td>51.34±0.25%</td></tr><tr><td>ALFA⁴</td><td>4-CONV</td><td>59.96±0.49%</td><td>76.79±0.42%</td><td>37.99±0.48%</td><td>53.01±0.49%</td></tr><tr><td>NPBML</td><td>4-CONV</td><td>64.90±0.94%</td><td>79.24±0.69%</td><td>40.56±0.76%</td><td>53.48±0.68%</td></tr><tr><td>MAML¹</td><td>ResNet-12</td><td>63.81±0.54%</td><td>77.07±0.42%</td><td>37.29±0.40%</td><td>50.70±0.35%</td></tr><tr><td>MeTAL³</td><td>ResNet-12</td><td>67.97±0.47%</td><td>82.17±0.38%</td><td>39.98±0.39%</td><td>53.85±0.36%</td></tr><tr><td>ALFA⁴</td><td>ResNet-12</td><td>66.79±0.47%</td><td>83.62±0.37%</td><td>41.46±0.49%</td><td>55.82±0.50%</td></tr><tr><td>NPBML</td><td>ResNet-12</td><td>69.30±0.91%</td><td>83.72±0.64%</td><td>43.63±0.71%</td><td>59.85±0.70%</td></tr></table>

$^{1}$  (Finn et al., 2017)  $^{2}$  (Oh et al., 2020)  $^{3}$  (Baik et al., 2021)  $^{4}$  (Baik et al., 2023)

# 6.2 ABLATION STUDIES

To further investigate the performance of the proposed method, we conduct two sets of ablation studies to analyze the effectiveness of each component. All ablation experiments are performed using the 4-CONV network architecture in a 5-way 5-shot setting on the mini-ImageNet dataset.

# 6.2.1 META-LEARNED COMPONENTS

First, we examine the importance of the meta-learned optimizer  $P_{\omega}$ , loss function  $\mathcal{M}_{\phi}$ , and task-adaptive conditioning method  $FILM_{\psi}$ . The results are presented in Table 3, and they demonstrate that each of the proposed components clearly and significantly contributes to the performance of NPBML. In (2) MAML is modified to include gradient preconditioning, which increases accuracy by  $2.09\%$ . Conversely in (3) we modify MAML with our meta-learned loss function, resulting in a  $6.37\%$  performance increase. Interestingly, the meta-learned loss function enhances performance by a larger margin; however, this may be due to the relatively simple T-Net style optimizer used in NPBML. This suggests that a more powerful parameterization, such as (Flennerhag et al., 2018) or (Kang et al., 2023), may further improve performance. In (4), MAML is modified to include both the optimizer and loss function, resulting in a  $7.41\%$  performance increase. This further supports our claim that meta-learning both an optimizer and a loss function are complementary and orthogonal approaches to improving MAML. Finally, in (5), we add our task-adaptive conditioning method, increasing performance by  $2.22\%$  over the prior experiment and  $9.63\%$  over MAML.

# 6.2.2 META-LEARNED LOSS FUNCTION

The prior ablation study shows that the meta-learned loss function  $\mathcal{M}_{\phi}$  is a crucial component in NPBML. Therefore, we further investigate each of the components; namely, the meta-learned inductive and transductive loss functions, and weight regularizer. The results are presented in Table 4, and surprisingly, they show that each of the components in isolation (7), (8), and (9), improves performance by approximately  $5\%$ . However, when combined in (10), the total performance increase is  $6.37\%$ . We hypothesize that this result is a consequence of the implicit meta-learning of the learning rate identified in Equation (15), which not only holds for  $\mathcal{M}_{\phi}$ , but also for each of its components, i.e., the equality is also true when  $\mathcal{M}_{\phi}$  is replaced with  $\mathcal{L}_\phi^S$ ,  $\mathcal{L}_\phi^Q$ , or  $\mathcal{R}_{\phi}$ . Since all components share implicit learning rate tuning, the performance gains from this behavior do not accumulate; however, the improvement in (10) is better than each component in isolation indicating that each component provides additional unique benefits to the meta-learning process.

Table 3: Ablation study of the meta-learned components in NPBML, reporting the meta-testing accuracy on mini-ImageNet 5-way 5-shot. A  $\checkmark$  denotes that the component is meta-learned, with variant (1) reducing to MAML, while variant (5) represents our final proposed algorithm.  

<table><tr><td></td><td>Initialization</td><td>Optimizer</td><td>Loss Function</td><td>Task-Adaptive</td><td>Accuracy</td></tr><tr><td>(1)</td><td>✓</td><td></td><td></td><td></td><td>65.38±0.67%</td></tr><tr><td>(2)</td><td>✓</td><td>✓</td><td></td><td></td><td>67.47±0.68%</td></tr><tr><td>(3)</td><td>✓</td><td></td><td>✓</td><td></td><td>71.75±0.69%</td></tr><tr><td>(4)</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>72.79±0.67%</td></tr><tr><td>(5)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>75.01±0.64%</td></tr></table>

Table 4: Ablation study of the meta-learned loss function  $\mathcal{M}_{\phi}$  in NPBML, reporting the meta-testing accuracy on mini-ImageNet 5-way 5-shot. Note, variants (6) and (10) correspond to variants (1) and (3), respectively in Table 3.  

<table><tr><td></td><td>Base Loss</td><td>Inductive Loss</td><td>Transductive Loss</td><td>Weight Regularizer</td><td>Accuracy</td></tr><tr><td>(6)</td><td>✓</td><td></td><td></td><td></td><td>65.38±0.67%</td></tr><tr><td>(7)</td><td>✓</td><td>✓</td><td></td><td></td><td>70.68±0.66%</td></tr><tr><td>(8)</td><td>✓</td><td></td><td>✓</td><td></td><td>70.92±0.68%</td></tr><tr><td>(9)</td><td>✓</td><td></td><td></td><td>✓</td><td>70.04±0.65%</td></tr><tr><td>(10)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>71.75±0.69%</td></tr></table>

# 7 CONCLUSION

In this work, we propose a novel meta-learning framework for learning the procedural biases of a deep neural network. The proposed technique, Neural Procedural Bias Meta-Learning (NPBML), consolidates recent advancements in MAML-based few-shot learning methods by replacing the fixed inner update rule with a fully meta-learned update rule. This is achieved by meta-learning a task-adaptive loss function, optimizer, and parameter initialization. The experimental results confirm the effectiveness and scalability of the proposed approach, demonstrating strong few-shot learning performance across a range of popular benchmarks. We believe NPBML provides a principled framework for advancing general-purpose meta-learning in deep neural networks. Looking ahead, numerous compelling future research directions exist, such as developing more powerful parameterizations for the meta-learned optimizer or loss function. We expect that further investigation of this topic will result in more expressive inner update rules, resulting in increased robustness and efficiency within the context of optimization-based meta-learning. Finally, broadening the scope of the proposed framework to encompass the related domains of cross-domain few-shot learning and continual learning may be a promising avenue for future exploration.

# REFERENCES

Antreas Antoniou and Amos J Storkey. Learning to learn by self-critique. Advances in Neural Information Processing Systems, 2019.  
Antreas Antoniou, Harrison Edwards, and Amos Storkey. How to train your maml. International Conference on Learning Representations, 2019.  
Sungyong Baik, Myungsub Choi, Janghoon Choi, Heewon Kim, and Kyoung Mu Lee. Meta-learning with adaptive hyperparameters. Advances in Neural Information Processing Systems, 2020.  
Sungyong Baik, Janghoon Choi, Heewon Kim, Dohee Cho, Jaesik Min, and Kyoung Mu Lee. Meta-learning with task-adaptive loss function for few-shot learning. In International Conference on Computer Vision, 2021.  
Sungyong Baik, Myungsub Choi, Janghoon Choi, Heewon Kim, and Kyoung Mu Lee. Learning to learn task-adaptive hyperparameters for few-shot learning. Transactions on Pattern Analysis and Machine Intelligence, 2023.  
Jonathan F Bard. Practical Bilevel Optimization: Algorithms and Applications. Springer Science & Business Media, 2013.  
Attilim Güneş Baydin, Robert Cornish, David Martínez Rubio, Mark Schmidt, and Frank Wood. Online learning rate adaptation with hypergradient descent. In International Conference on Learning Representations, 2018.  
Sarah Bechtle, Artem Molchanov, Yevgen Chebotar, Edward Grefenstette, Ludovic Righetti, Gaurav Sukhatme, and Franziska Meier. Meta learning via learned loss. In International Conference on Pattern Recognition (ICPR), 2021.  
Harkirat Singh Behl, Atulm Güneş Baydin, and Philip HS Torr. Alpha maml: Adaptive model-agnostic meta-learning. arXiv preprint arXiv:1905.07435, 2019.  
Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. arXiv preprint arXiv:1805.08136, 2018.  
Harm De Vries, Florian Strub, Jérémie Mary, Hugo Larochelle, Olivier Pietquin, and Aaron C Courville. Modulating early visual processing by language. Advances in Neural Information Processing Systems, 2017.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition, 2009.  
Vincent Dumoulin, Ethan Perez, Nathan Schucher, Florian Strub, Harm de Vries, Aaron Courville, and Yoshua Bengio. Feature-wise transformations. Distill, 2018.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, 2017.  
Sebastian Flennerhag, Pablo G Moreno, Neil D Lawrence, and Andreas Damianou. Transferring knowledge across learning processes. International Conference on Learning Representations, 2018.  
Sebastian Flennerhag, Andrei A Rusu, Razvan Pascanu, Francesco Visin, Hujun Yin, and Raia Hadsell. Meta-learning with warped gradient descent. International Conference on Learning Representations, 2020.  
Santiago Gonzalez and Risto Miikkulainen. Effective regularization through loss function meta-learning. arXiv preprint arXiv:2010.00788, 2020.  
Diana F Gordon and Marie Desjardins. Evaluation and selection of biases in machine learning. Machine Learning, 1995.

Priya Goyal, Piotr Dólar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: TrainingImagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Edward Grefenstette, Brandon Amos, Denis Yarats, Phu Mon Htut, Artem Molchanov, Franziska Meier, Douwe Kiela, Kyunghyun Cho, and Soumith Chintala. Generalized inner loop meta-learning. arXiv preprint arXiv:1910.01727, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2016.  
Timothy Hospedales, Antreas Antoniou, Paul Micaelli, and Amos Storkey. Meta-learning in neural networks: A survey. Transactions on Pattern Analysis and Machine Intelligence, 2022.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, 2015.  
Xiang Jiang, Mohammad Havaei, Farshid Varno, Gabriel Chartrand, Nicolas Chapados, and Stan Matwin. Learning to learn with conditional class dependencies. In International Conference on Learning Representations, 2018.  
Suhyun Kang, Duhun Hwang, Moonjung Eo, Taesup Kim, and Wonjong Rhee. Meta-learning with a geometry-adaptive preconditioner. In Conference on Computer Vision and Pattern Recognition, 2023.  
Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, et al. Siamese neural networks for one-shot image recognition. In International Conference on Machine Learning - Deep Learning Workshop, 2015.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Kwonjoon Lee, Subhransu Maji, Avinash Ravichandran, and Stefano Soatto. Meta-learning with differentiable convex optimization. In Conference on Computer Vision and Pattern Recognition, 2019.  
Yoonho Lee and Seungjin Choi. Gradient-based meta-learning with learned layerwise metric and subspace. In International Conference on Machine Learning, 2018.  
Zhenguo Li, Fengwei Zhou, Fei Chen, and Hang Li. Meta-sgd: Learning to learn quickly for few-shot learning. arXiv preprint arXiv:1707.09835, 2017.  
Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In International Conference on Artificial Intelligence and Statistics, 2020.  
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning, 2015.  
Rafael Müller, Simon Kornblith, and Geoffrey E Hinton. When does label smoothing help? Advances in Neural Information Processing Systems, 2019.  
Alex Nichol and John Schulman. Reptile: A scalable meta-learning algorithm. arXiv preprint arXiv:1803.02999, 2018.  
Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, and Se-Young Yun. Boil: Towards representation change for few-shot learning. arXiv preprint arXiv:2008.08882, 2020.  
Boris Oreshkin, Pau Rodríguez López, and Alexandre Lacoste. Tadam: Task dependent adaptive metric for improved few-shot learning. Advances in Neural Information Processing Systems, 2018.  
Eunbyung Park and Junier B Oliva. Meta-curvature. Advances in Neural Information Processing Systems, 2019.  
Huimin Peng. A comprehensive overview and survey of recent advances in meta-learning. arXiv preprint arXiv:2004.11149, 2020.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.  
Apostolos F Psaros, Kenji Kawaguchi, and George Em Karniadakis. Meta-learning pinn loss functions. Journal of computational physics, 2022.  
Siyuan Qiao, Chenxi Liu, Wei Shen, and Alan L Yuille. Few-shot image recognition by predicting parameters from activations. In Conference on Computer Vision and Pattern Recognition, 2018.  
Aniruddh Raghu, Maithra Raghu, Samy Bengio, and Oriol Vinyals. Rapid learning or feature reuse? towards understanding the effectiveness of maml. arXiv preprint arXiv:1909.09157, 2019.  
Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. Advances in Neural Information Processing Systems, 2019.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In International Conference on Learning Representations, 2017.  
Christian Raymond. Meta-learning loss functions for deep neural networks. arXiv preprint arXiv:2406.09713, 2024.  
Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang. Learning symbolic model-agnostic loss functions via meta-learning. Transactions on Pattern Analysis and Machine Intelligence, 2023a.  
Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang. Online loss function learning. arXiv preprint arXiv:2301.13247, 2023b.  
Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, and Richard S. Zemel. Meta-learning for semi-supervised few-shot classification. In International Conference on Learning Representations, 2018.  
James Requeima, Jonathan Gordon, John Bronskill, Sebastian Nowozin, and Richard E Turner. Fast and flexible multi-task classification using conditional neural adaptive processes. Advances in Neural Information Processing Systems, 2019.  
Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, and Raia Hadsell. Meta-learning with latent embedding optimization. In International Conference on Learning Representations, 2019.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning, 2016.  
Jürgen Schmidhuber. Evolutionary Principles in Self-Referential Learning. PhD thesis, Technische Universität München, 1987.  
Christian Simon, Piotr Koniusz, Richard Nock, and Mehrtash Harandi. On modulating the gradient for meta-learning. In European Conference on Computer Vision, 2020.  
Samuel L Smith and Quoc V Le. A bayesian perspective on generalization and stochastic gradient descent. arXiv preprint arXiv:1710.06451, 2017.  
Samuel L Smith, Pieter-Jan Kindermans, Chris Ying, and Quoc V Le. Don't decay the learning rate, increase the batch size. arXiv preprint arXiv:1711.00489, 2017.  
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in Neural Information Processing Systems, 2017.  
Xingyou Song, Wenbo Gao, Yuxiang Yang, Krzysztof Choromanski, Aldo Pacchiano, and Yunhao Tang. Es-maml: Simple hessian-free meta learning. arXiv preprint arXiv:1910.01215, 2019.

Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip HS Torr, and Timothy M Hospedales. Learning to compare: Relation network for few-shot learning. In Conference on Computer Vision and Pattern Recognition, 2018.  
Eleni Triantafillou, Tyler Zhu, Vincent Dumoulin, Pascal Lamblin, Utku Evci, Kelvin Xu, Ross Goroshin, Carles Gelada, Kevin Jordan Swersky, Pierre-Antoine Manzagol, and Hugo Larochelle. Meta-dataset: A dataset of datasets for learning to learn from few examples. In International Conference on Learning Representations, 2020.  
Joaquin Vanschoren. Meta-learning: A survey. arXiv preprint arXiv:1810.03548, 2018.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one-shot learning. Advances in Neural Information Processing Systems, 2016.  
Risto Vuorio, Shao-Hua Sun, Hexiang Hu, and Joseph J Lim. Multimodal model-agnostic meta-learning via task-aware modulation. Advances in Neural Information Processing Systems, 2019.  
Han-Jia Ye and Wei-Lun Chao. How to train your maml to excel in few-shot classification. In International Conference on Learning Representations, 2021.  
Han-Jia Ye, Hexiang Hu, De-Chuan Zhan, and Fei Sha. Few-shot learning via embedding adaptation with set-to-set functions. In Conference on Computer Vision and Pattern Recognition, 2020.  
Luisa Zintgraf, Kyriacos Shiarli, Vitaly Kurin, Katja Hofmann, and Shimon Whiteson. Fast context adaptation via meta-learning. In International Conference on Machine Learning, 2019.

Algorithm 1 Meta-Learning (Outer Loop)  
Input:  $\mathcal{L}^{meta}\gets$  Meta loss function   
Input:  $p(T)\leftarrow$  Task distribution   
Input:  $\eta \leftarrow$  Meta learning rate   
1:  $\Phi_0\gets$  Initialize meta-parameters  $\{\theta ,\omega ,\phi ,\psi \}$    
2: for  $t\in \{0,\dots,S^{meta}\}$  do   
3:  $\mathcal{T}_0,\mathcal{T}_1,\ldots ,\mathcal{T}_B\gets$  Sample tasks from  $p(T)$    
4: for  $i\in \{0,\dots,B\}$  do   
5:  $\mathcal{D}_i^S = \{(x_i^s,y_i^s)\}_{s = 0}^S\gets$  Sample support from  $\mathcal{T}_i$    
6:  $\mathcal{D}_i^Q = \{(x_i^q,y_i^q)\}_{q = 0}^Q\gets$  Sample query from  $\mathcal{T}_i$    
7:  $\theta_{i,j}\gets$  Base-Learning using Algorithm (2)   
8:  $\Phi_{t + 1}\gets \Phi_t - \eta \frac{1}{B}\nabla_{\Phi_t}\sum_i\mathcal{L}_i^{meta}(\mathcal{D}_i^Q,\theta_{i,j})$    
9: return  $\Phi_t$    
Algorithm 2 Base-Learning (Inner Loop)   
Input:  $\mathcal{L}^{base}\gets$  Base loss function   
Input:  $\mathcal{D}_i^S,\mathcal{D}_i^Q\gets$  Support and query sets   
Input:  $\Phi \gets$  Meta parameters  $\{\theta ,\omega ,\phi ,\psi \}$    
Input:  $\alpha \gets$  Base learning rate   
Input:  $g\gets$  Relation network   
1:  $\theta_{i,0}\gets$  Initialize base weights with  $\theta$    
2: for  $j\in \{0,\dots,S^{base}\}$  do   
3:  $\hat{y}_i^S,\hat{y}_i^Q\gets f_{(\theta_{i,j},\omega ,\psi)}(x_i^S\cup x_i^Q)$    
4:  $\mathcal{L}_{i,j}^{base}\gets \frac{1}{|\mathcal{D}^S|}\sum \mathcal{L}^{base}(y_i^S,\hat{y}_i^S)$    
5:  $\mathcal{L}_{i,j}^{S}\gets \frac{1}{|\mathcal{D}^{S}|}\sum \mathcal{L}_{(\phi ,\psi)}^{S}(y_i^S,\hat{y}_i^S)$    
6:  $\mathcal{L}_{i,j}^{Q}\gets \frac{1}{|\mathcal{D}^{Q}|}\sum \mathcal{L}_{(\phi ,\psi)}^{Q}(g(x_i^Q),\hat{y}_i^Q)$    
7:  $\mathcal{R}_{i,j}\gets \mathcal{R}_{(\phi ,\psi)}(\theta_{i,j})$    
8:  $\mathcal{M}_{(\phi ,\psi)}\gets \mathcal{L}_{i,j}^{base} + \mathcal{L}_{i,j}^{S} + \mathcal{L}_{i,j}^{Q} + \mathcal{R}_{i,j}$    
9:  $\theta_{i,j + 1}\gets \theta_{i,j} - \alpha P\nabla_{\theta_{i,j}}\mathcal{M}_{(\phi ,\psi)}$    
10: return  $\theta_{i,j}$
