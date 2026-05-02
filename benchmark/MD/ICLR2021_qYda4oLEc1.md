# THE TRAVELING OBSERVER MODEL: MULTI-TASK LEARNING THROUGH SPATIAL VARIABLE EMBEDDINGS

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper frames a general prediction system as an observer traveling around a continuous space, measuring values at some locations, and predicting them at others. The observer is completely agnostic about any particular task being solved; it cares only about measurement locations and their values. This perspective leads to a machine learning framework in which seemingly unrelated tasks can be solved by a single model, by embedding their input and output variables into a shared space. An implementation of the framework is developed in which these variable embeddings are learned jointly with internal model parameters. In experiments, the approach is shown to (1) recover intuitive locations of variables in space and time, (2) exploit regularities across related datasets with completely disjoint input and output spaces, and (3) exploit regularities across seemingly unrelated tasks, outperforming task-specific single-task models and multi-task learning alternatives. The results suggest that even seemingly unrelated tasks may originate from similar underlying processes, a fact that the traveling observer model can use to make better predictions.

# 1 INTRODUCTION

Natural organisms benefit from the fact that their sensory inputs and action outputs are all organized in the same space, that is, the physical universe. This consistency makes it easy to apply the same predictive functions across diverse settings. Deep multi-task learning (Deep MTL) has shown a similar ability to adapt knowledge across tasks whose observed variables are embedded in a shared space. Examples include vision, where the input for all tasks (photograph, drawing, or otherwise) is pixels arranged in a 2D plane (Zhang et al., 2014; Misra et al., 2016; Rebuffi et al., 2017); natural language (Collobert & Weston, 2008; Luong et al., 2016; Hashimoto et al., 2017), speech processing (Seltzer & Droppo, 2013; Huang et al., 2015), and genomics (Alipanahi et al., 2015), which exploit the 1D structure of text, waveforms, and nucleotide sequences; and video game-playing (Jaderberg et al., 2017; Teh et al., 2017), where interactions are organized across space and time. Yet, many real-world prediction tasks have no such spatial organization; their input and output variables are simply labeled values, e.g., the height of a tree, the cost of a haircut, or the score on a standardized test. To make matters worse, these sets of variables are often disjoint across a set of tasks. These challenges have led the MTL community to avoid such tasks, despite the fact that general knowledge about how to make good predictions can arise from solving seemingly "unrelated" tasks (Mahmud & Ray, 2008; Mahmud, 2009; Meyerson & Miikkulainen, 2019).

![](images/cb72794edf380d0aa32bc5436d5fb494057a47d5df7cd85a2a705199e9199d38.jpg)  
(a)

![](images/ac6a0e5b9e9aeb974a12a50d39c15278798bc0f0ff5f9ccd5ccb0e8ea3d68b5a.jpg)  
Figure 1: (a) Tasks with disjoint input and output variable sets, whose variables are nonetheless measured in the same underlying space (dotted lines are samples); (b) Traveling observer model (TOM) for variables in this space: It predicts values at output locations, given values at input locations.

![](images/4718e1633edf5bcf174877561bd43f5f3a719daedfe6d8d238e6fa580028c0bb.jpg)  
(b)

<table><tr><td>(a) Intra-domain</td><td>(b) Task Embeddings</td><td>(c) Cross-domain</td><td>(d) Variable Embeddings (TOM)</td></tr><tr><td>ˆy_t = gt(f(x_t))</td><td>ˆy_t = g(f(x_t, z_t)))</td><td>ˆy_t = gt(f_t(x_t))</td><td>ˆy_j = g\left(\sum_{x_i \in x_t} f(x_i, z_i), z_j\right)</td></tr></table>

Table 1: MTL approaches decomposed into encoders  $f_{*}$  and decoders  $g_{*}$ : (a) Standard MTL takes advantage of the shared spatialization of tasks within a domain by sharing a single encoder across all tasks  $t$ ; (b) Task embeddings allow tasks within a domain to share their decoder as well; (c) Applying standard MTL across domains requires task-specific encoders, and finding some other method of sharing parameters across tasks; (d) TOM allows a single encoder and decoder to be used even in the cross-domain setting, by embedding all input and output variables into a shared space.

This paper proposes a solution: Learn all variable locations in a shared space, while simultaneously training the prediction model itself (Figure 1). To illustrate this idea, Figure 1(a) gives an example of two tasks whose variables are measured at different locations in the same underlying 1D universe. Each dotted line shows the state of the entire universe when a sample is drawn; the green and red dots mark the values of the observed input and output variables for the sample. Figure 1(b) shows a model that can be applied to any such task:  $f$  encodes the value of each observed  $x_{i}$  given its location  $\mathbf{z}_i$ , these encodings are aggregated, and  $g$  decodes the aggregated encoding to a prediction for  $y_{j}$  given  $\mathbf{z}_j$ . Such a predictor can be viewed as a traveling observer model (TOM): It traverses the space of variables, taking a measurement at the location of each input. Given these observations, the model can make a prediction for the value at the location of an output. In general, the  $\mathbf{z}$ 's are not known a priori, but they can be learned alongside  $f$  and  $g$  by gradient descent.

The input and output spaces of a prediction problem can be standardized so that the measured value of each input and output variable is a scalar. The prediction model can then be completely agnostic about the particular task for which it is making a prediction. By learning variable embeddings, the model can capture variable relationships explicitly and supports joint training of a single architecture across seemingly unrelated tasks with disjoint input and output spaces. TOM thus establishes a new lower bound on the commonalities shared across real-world machine learning problems: They are all drawn from the same space of variables that humans can and do measure.

This paper develops a first implementation of TOM, using an encoder-decoder architecture, with variable embeddings incorporated using FiLM (Perez et al., 2018). In the experiments, the implementation is shown to (1) recover the intuitive locations of variables in space and time, (2) exploit regularities across related datasets with disjoint input and output spaces, and (3) exploit regularities across seemingly unrelated tasks to outperform single-tasks models tuned to each tasks, as well as current Deep MTL alternatives. The results confirm that TOM is a promising framework for representing and exploiting the underlying processes of seemingly unrelated tasks.

# 2 BACKGROUND: MULTI-TASK ENCODER-DECODER DECOMPOSITIONS

This section reviews Deep MTL methods from the perspective of decomposition into encoders and decoders (Table 1). In MTL, there are  $T$  tasks  $\{\{(\mathbf{x}_t^s,\mathbf{y}_t^s)\} _s = 1\}_{t = 1}^S\}$  that can, in general, be drawn from different domains and have varying input and output dimensionality. The  $t$ th task has  $S_{t}$  samples, and has input variables  $[x_1^t,\ldots ,x_{n_t}^t ] = \mathbf{x}_t\in \mathbb{R}^{n_t}$  and output variables  $[y_1^t,\dots,y_{m_t}^t ] = \mathbf{y}_t\in \mathbb{R}^{m_t}$ , which can also be written as sets  $V_{t}^{\mathrm{in}} = \{x_{1}^{t},\ldots ,x_{n_{t}}^{t}\}$  and  $V_{t}^{\mathrm{out}} = \{y_{1}^{t},\ldots ,y_{m_{t}}^{t}\}$ . Two tasks  $t_1$  and  $t_2$  are disjoint if  $(V_{t_1}^{\mathrm{in}}\cup V_{t_1}^{\mathrm{out}})\cap (V_{t_2}^{\mathrm{in}}\cup V_{t_2}^{\mathrm{out}}) = \emptyset$ . The goal is to exploit regularities across task models  $\mathbf{x}_t\mapsto \hat{\mathbf{y}}_t$  by jointly training them with overlapping parameters.

The standard intra-domain approach is for all task models to share their encoder  $f$ , and each to have its own task-specific decoder  $g_{t}$ . This setup was used in the original introduction of MTL (Caruana, 1998), and has been broadly explored in the linear regime (Argyriou et al., 2008; Kang et al., 2011; Kumar & Daume, 2012). It is also the most common approach in Deep MTL (Huang et al., 2013; Zhang et al., 2014; Dong et al., 2015; Liu et al., 2015; Ranjan et al., 2016; Jaderberg et al., 2017). The main limitation of this approach is that it is limited to sets of tasks that are all drawn from the same domain. It also has the risk of the separate decoders doing so much of the learning that there is not much left to be shared, which is why the decoders are usually single affine layers.

To address the issue of limited sharing, the task embeddings approach trains a single encoder  $f$  and single decoder  $g$ , with all task-specific parameters learned in embedding vectors  $\mathbf{z}_t$  that semantically characterize each task, and which are fed into the model as additional input (Yang & Hospedales, 2014; Bilen & Vedaldi, 2017; Zintgraf et al., 2019). Such methods require that all tasks have the same input and output space, but are flexible in how the embeddings can be used to adapt the model to each task. As a result, they can learn tighter connections between tasks than separate decoders, and these relationships can be analyzed by looking at the learned embeddings.

To exploit regularities across tasks from diverse and disjoint domains, cross-domain methods have been introduced. Such methods have separate decoders and encoders for each domain, and thus include some other method of sharing model parameters across tasks, such as sharing some of their layers (Kaiser et al., 2017) or drawing their parameters from a shared pool (Meyerson & Miikkulainen, 2019). For many datasets, the separate encoder and decoder absorbs too much functionality to share optimally, and their complexity makes it difficult to analyze the relationships between tasks.

TOM extends the notion of task embeddings to variable embeddings in order to apply the idea in the cross-domain setting. The method is described in the next section.

# 3 THE TRAVELING OBSERVER MODEL

Consider a universe of all random variables that could possibly be measured  $\{x_{1}, x_{2}, \ldots\} = V$ . To characterize each  $x_{i}$  semantically, associate with it a vector  $\mathbf{z}_{i} \in \mathbb{R}^{C}$  that encodes the meaning of  $x_{i}$ , e.g., "height of left ear of human adult in inches", "answer to survey question 9 on a scale of 1 to 5", "severity of heart disease", "brightness of top-left pixel of photograph", etc. This vector  $\mathbf{z}_{i}$  is called the variable embedding (VE) of  $x_{i}$ . Variable embeddings could be handcoded, e.g., based on some featurization of the space of variables, but such a handcoding is usually unavailable, and would likely miss some of the underlying semantic regularities across variables. An alternative approach is to learn variable embeddings based on their utility in solving prediction problems of interest.

A prediction problem is defined by a set of observed variables  $V_{\mathrm{obs}} \subseteq V$  and a set of target variables  $V_{\mathrm{tar}} \subseteq V$  whose values are unknown, e.g.,  $V_{t}^{\mathrm{in}}$  and  $V_{t}^{\mathrm{out}}$  for a particular task  $t$ . The goal is to find a prediction function  $\Omega$  that can be applied across any prediction problem of interest, so that it can learn to exploit regularities across such problems. Let  $Z_{\mathrm{obs}} = \{\mathbf{z}_i : x_i \in V_{\mathrm{obs}}\}$  be the set of variable embeddings corresponding to variables in  $V_{\mathrm{obs}}$ . Then, this universal prediction model is of the form

$$
\mathbb {E} \left[ x _ {j} \mid V _ {\mathrm {o b s}} \right] = \Omega \left(V _ {\mathrm {o b s}}, Z _ {\mathrm {o b s}}, \mathbf {z} _ {j}\right). \tag {1}
$$

Importantly, for any two prediction problems  $(V_{\mathrm{obs}}, V_{\mathrm{tar}})$ ,  $(V_{\mathrm{obs}}', V_{\mathrm{tar}}')$ , their prediction functions differ only in their  $\mathbf{z}$ 's, which enforces the constraint that functionality is otherwise completely shared across the models. One can view  $\Omega$  as a traveling observer, who visits several locations in the  $C$ -dimensional variable space, takes measurements at those locations, and uses this information to make predictions of values at other locations.

To make  $\Omega$  concrete, it must be a function that can be applied to any number of variables, can fit any set of prediction problems, and is invariant to variable ordering, since we cannot in general assume that a meaningful order exists. These requirements lead to the following decomposition:

$$
\mathbb {E} \left[ x _ {j} \mid V _ {\mathrm {o b s}} \right] = \Omega \left(V _ {\mathrm {o b s}}, Z _ {\mathrm {o b s}}, \mathbf {z} _ {j}\right) = g \left(\sum_ {x _ {i} \in V _ {\mathrm {o b s}}} f \left(x _ {i}, \mathbf {z} _ {i}\right), \mathbf {z} _ {j}\right), \tag {2}
$$

where  $f$  and  $g$  are functions called the encoder and decoder, with trainable parameters  $\theta_f$  and  $\theta_g$ , respectively. The variable embeddings  $\mathbf{z}$  tell  $f$  and  $g$  which variables they are observing, and these  $\mathbf{z}$  can be learned by gradient descent alongside  $\theta_f$  and  $\theta_g$ . A depiction of the model is shown in Figure 1(b). For some integer  $M$ ,  $f: \mathbb{R}^{C + 1} \to \mathbb{R}^M$  and  $g: \mathbb{R}^{M + C} \to \mathbb{R}$ . In principle,  $f$  and  $g$  could be any sufficiently expressive functions of this form. A natural choice is to implement them as neural networks. They are called the encoder and decoder because they map variables to and from a latent space of size  $M$ . This model can then be trained end-to-end with gradient descent. A batch for gradient descent is constructed by sampling a prediction problem, e.g., a task, from the distribution of problems of interest, and then sampling a batch of data from the data set for that problem. Notice that, in addition to supervised training, in this framework it is natural to autoencode, i.e., predict input variables, and subsample inputs to simulate multiple tasks drawn from the same universe.

![](images/dd3a9b98ef6fa0d55bdfcac8c67edef4568049fd71683b646855a467fcaddcb4.jpg)  
Figure 2: Diagram of the TOM implementation used in the experiments. Encoder, Core, and Decoder correspond to  $f$ ,  $h$ , and  $g$  in Eq. 4, resp. The Encoder and Decoder are conditioned on input and output VEs  $\mathbf{z}$  via FiLM layers. A CRB is simply an FRB without conditioning. Dropout and trainable scalars  $\alpha$  implement SkipInit as a substitute for BatchNorm. This residual structure allows the architecture to learn tasks of varying complexity in a flexible manner.

The question remains: How can  $f$  and  $g$  be designed so that they can sufficiently capture a broad range of prediction behavior, and be effectively conditioned by variable embeddings? The next section introduces an experimental architecture that satisfies these requirements.

# 4 EXPERIMENTS

This section presents a concrete implementation of TOM, followed by a suite of experiments that evaluate the behavior of this implementation. See Appendix for additional experimental details.

# 4.1 TOM IMPLEMENTATION

The experiments in this paper implement TOM using a generic architecture built from standard components. The encoder and decoder are conditioned on VEs via FiLM layers (Perez et al., 2018), which provide a flexible yet inexpensive way to adapt functionality to each variable, and have been previously used to incorporate task embeddings (Zintgraf et al., 2019). For simplicity, the FiLM layers are based on affine transformations of VEs. Specifically, the  $\ell$ th FiLM layer  $F_{\ell}$  is parameterized by affine layers  $W_{\ell}^{*}$  and  $W_{\ell}^{+}$ , and, given a variable embedding  $\mathbf{z}$ , the hidden state  $\mathbf{h}$  is modulated by

$$
F _ {\ell} (\mathbf {h}) = W _ {\ell} ^ {*} (\mathbf {z}) \odot \mathbf {h} + W _ {\ell} ^ {+} (\mathbf {z}), \tag {3}
$$

where  $\odot$  is the Hadamard product. A FiLM layer is located alongside each fully-connected layer in the encoder and decoder, both of which consist primarily of residual blocks. To avoid deleterious behavior of batch norm across diverse tasks and small datasets/batches, the recently proposed SkipInit (De & Smith, 2020) is used as a replacement to stabilize training. SkipInit adds a trainable scalar  $\alpha$  initialized to 0 at the end of each residual block, and uses dropout for regularization. Finally, for computational efficiency, the decoder is redecomposed into the Core, or  $h$ , which is independent of output variable, and the Decoder proper, or  $g$ , which is conditioned on the output variable. That way, generic transformations of the summed Encoder output can be learned by the Core and run in a single forward and backward pass each iteration. With this decomposition, Eq. 2 is rewritten as

$$
\mathbb {E} \left[ x _ {j} \mid V _ {\mathrm {o b s}} \right] = g \left(h \left(\sum_ {x _ {i} \in V _ {\mathrm {o b s}}} f \left(x _ {i}, \mathbf {z} _ {i}\right)\right), \mathbf {z} _ {j}\right). \tag {4}
$$

The complete architecture is depicted in Figure 2. In the following sections, all models are implemented in pytorch (Paske et al., 2017), use Adam for optimization (Kingma & Ba, 2014), and have hidden layer size of 128 for all layers. Variable embeddings for TOM are initialized from  $\mathcal{N}(0,10^{-3})$ . See Appendix B for additional details of this implementation.

# 4.2 VALIDATING LEARNED VARIABLE EMBEDDINGS: DISCOVERING SPACE AND TIME

The experiments in this section test TOM's ability to learn variable embeddings that reflect our a priori intuition about the domain, in particular, the organization of space and time.

![](images/7d48d1fe721fdebdd757e459172339708707f9a6bca9a886637d15e007a3bb85.jpg)

![](images/56f54f1a8caf7f0943eba4a2c73e7146144e41082e6b09378e18c96f0820fe23.jpg)

![](images/150203b808a2903dd9fa30df6e3a6f022beff9c7cdf0bd5e73970663dd1415dc.jpg)

![](images/7586506afd2dab695599ebd27753361503f0f49620396eebcae43d0d392f2bbc.jpg)

![](images/f930466669a2dbf7d443cac2bd638eb6134a185ada5ade6bb493ffb2ba26a38b.jpg)  
Figure 3: Variable embeddings learned for CIFAR unfold over iterations until they resemble Oracle expectations (best viewed in color). The VE for each variable, i.e., pixel, is colored uniquely. TOM peels the border of the CIFAR images (the upper loop of VEs at iteration 300K) away from their center (the lower grid). This makes sense, since CIFAR images all feature a central object, which semantically splits the image into foreground (the object itself) and background (the remaining ring of pixels around the object). See https://youtu.be/R_z-2SR2KpY for videos of VEs being learned.

![](images/7b3daafe7e294377cb5b78fcab40c454d1e98bd759a8a1f587f8f7635b3b227f.jpg)

![](images/348630e1f42a2608b9131bc601298c4186227785ca26a96917846a1f70f2c44e.jpg)

![](images/8228f1e50995cda8655c2b4b4c4a8367ff23e5c0982e0159289c19d543a64892.jpg)

CIFAR. The first experiment is based on the CIFAR dataset (Krizhevsky, 2009). The pixels of the  $32 \times 32$  images are converted to grayscale values in [0, 1], yielding 1024 variables. The goal is to predict all variable values, given only a subset of them as input. The model is trained to minimize the binary cross-entropy of each output, and it uses 2D VEs. The a priori, or Oracle, expectation is that the VEs form a  $32 \times 32$  grid corresponding to how pixels are spatially laid out in an image.

Daily Temperature. The second experiment is based on the Melbourne minimum daily temperature dataset, a subset of a larger database for tracking climate change (Della-Marta et al., 2004). As above, the goal is to predict the daily temperature of the previous 10 days, given only some subset of them, by minimizing the MSE of each variable. The a priori, Oracle, expectation is that the VEs are laid out linearly in a single temporal dimension. However, in this experiment, to allow VEs to untangle more easily and avoid getting stuck, they are learned in a 2D space.

For both experiments, a subset of the input variables is randomly sampled at each training iteration, which simulates drawing tasks from a limited universe. The resulting learning process for the VEs is illustrated in Figures 3 and 4. The VEs for CIFAR pull apart and unfold, until they reflect the oracle embeddings (Figure 3). The remaining difference is that TOM peels the border of the CIFAR images (the upper loop of VEs at iteration  $300\mathrm{K}$ ) away from their center (the lower grid). This makes sense, since CIFAR images all feature a central object, which semantically splits the image into foreground (the object itself) and background (the remaining ring of pixels around the object). Similarly, the VEs for daily temperature pull apart until they form a perfect 1D manifold representing the time dimension (Figure 4). The main difference is that TOM has embedded this 1D structure as a ring in 2D, which is well-suited to the nonlinear encoder and decoder, since it mirrors an isotropic Gaussian distribution. Note that unlike visualization methods like SOM (Kohonen, 1990), PCA (Pearson, 1901), or t-SNE (van der Maaten & Hinton, 2008), TOM learns locations for each variable not each sample. Furthermore, TOM has no explicit motivation to visualize; learned VEs are simply the locations found to be useful by using gradient descent when solving the prediction problem.

To get an idea of how learning VEs affects prediction performance, comparisons were run with three cases of fixed VEs: (1) all VEs set to zero, to address the question of whether differentiating variables with VEs is needed at all in the model; (2) random VEs, to address the question of whether simply having any unique label for variables is sufficient; and (3) oracle VEs, which reflect the human a priori expectation of how the variables should be arranged. The results show that the learned embeddings outperform zero and random embeddings, achieving performance on par with the Oracle (Table 2). The conclusion is that learned VEs in TOM are not only meaningful, but can help make superior predictions, without a priori knowledge of variable meaning. The next section shows how such VEs can be used to exploit regularities across tasks in an MTL setting.

<table><tr><td>Variable Embeddings</td><td>Zero</td><td>Random</td><td>Learned</td><td>Oracle</td></tr><tr><td>CIFAR (Binary Cross-entropy)</td><td>0.662</td><td>0.655</td><td>0.591</td><td>0.588</td></tr><tr><td>Daily Temperature (RMSE)</td><td>4.23</td><td>3.40</td><td>3.28</td><td>3.31</td></tr></table>

Table 2: Quantitative results for space and time prediction. This table compares test errors of learned VEs to fixed-VE alternatives in TOM. The results show that learned VEs outperform Zero and Random VEs, reaching performance on par with the Oracle. That is, TOM not only learns meaningful VEs (Figures 3 and 4), but also uses these VEs to achieve superior performance.

![](images/7c5acb30bc78068e9c8b190468a717e9eb155b592167f87ec74efdce65e8280f.jpg)

![](images/1c4ed78ceed33b35d69ea406ed4ff185bfd9edd4b368ed4f03ed610a87454296.jpg)

![](images/6ee6635012d2c65529457fe35a65db2b0d7be86fbf00bd04cfa0dfaad22cbb39.jpg)

![](images/601963d0fea663d77ce66e84c49bdcd2eec6eef10e0cd2e48eef1f3b8c42a34b.jpg)

![](images/ac9482dd9b5d5189dbb6a4899e9cd35596f92bf27bd18232c72f173d9482f292.jpg)  
Figure 4: Variable embeddings learned for daily temperature variables untangle over iterations and converge on a 1D manifold ordered by time, as one would expect (neighboring time-steps are connected to illustrate the order). TOM has embedded this 1D structure as a ring in 2D, which is well-suited to the nonlinear encoder and decoder, since it mirrors an isotropic Gaussian distribution.

![](images/84cd681bd9a36da5284bae10a41276d99fb2cafdf43b8cc771b40fe3b2e538ed.jpg)

![](images/54ab809d657701a624267a390b707afa316b7590d441da98894b87cab9df386a.jpg)

![](images/03f168e17fcbb4c6e0c8ebb598d6cedd933c59ca0d59c965f7bfd442d1f615e3.jpg)

# 4.3 EXPLOITING REGULARITIES ACROSS DISJOINT TASKS

This section considers two synthetic multi-task problems that contain underlying regularities across tasks. These regularities are not known to the model a priori; it can only exploit them via its VEs. The first problem evaluates TOM in a regression setting where input and output variables are drawn from the same continuous space; the second problem evaluates TOM in a classification setting.

Transposed Gaussian Process. In the first problem, the universe is defined by a Gaussian process (GP). The GP is 1D, is zero-mean, and has an RBF kernel with length-scale 1. One task is generated for each (# inputs, # outputs) pair in  $\{1, \dots, 10\} \times \{1, \dots, 10\}$ , for a total of 100 tasks. The "true" location of each variable lies in the single dimension of the GP, and is sampled uniformly from [0, 5]. Samples for the task are generated by sampling from the GP, and measuring the value at each variable location. Each task contains 10 training samples, 10 validation samples, and 100 test samples. Samples are generated independently for each task. The goal is to minimize MSE of the outputs. Figure 1(a) gives two examples of tasks drawn from this universe. This testbed is ideal for TOM, because, by the definition of the GP, it explicitly captures the idea that variables whose VEs are nearby are closely related, and every variable has some effect on all others.

Concentric Hyperspheres. In the second problem, each task is defined by a set of concentric hyperspheres. One task is generated for each (# features  $n$ , # classes  $m$ ) pair in  $\{1, \ldots, 10\} \times \{2, \ldots, 10\}$  for a total of 90 tasks. For each task, its origin  $\mathbf{o}_t$  is drawn from  $\mathcal{N}(\mathbf{0}, I_n)$ . Then, for each class  $c \in \{1, \ldots, m\}$ , samples are drawn from  $\mathbb{R}^n$  uniformly at distance  $c$  from  $\mathbf{o}_t$ , i.e., each class is defined by a (hyper) annulus. Each task contains five training samples, five validation samples, and 100 test samples per class. The model has no a priori knowledge that the classes are structured in annuli, or which annulus corresponds to which class, but it is possible to achieve high accuracy by making analogies of annuli across tasks, i.e., discovering the underlying structure of this universe.

In these experiments, TOM is compared to five alternative methods: (1) TOM-  $STL$ , i.e. TOM trained on each task independently; (2)  $DR$ - $MTL$  (Deep Residual MTL), the standard cross-domain (Table 1(c)) version of TOM, where instead of FiLM layers, each task has its own linear encoder and decoder layers, and all residual blocks are CoreResBlocks; (3)  $DR$ - $STL$ , which is like DR-MTL

<table><tr><td>Method</td><td>DR-STL</td><td>TOM-STL</td><td>DR-MTL</td><td>SLO</td><td>TOM</td><td>Oracle</td></tr><tr><td>Transposed Gaussian Process (MSE)</td><td>0.373</td><td>0.552</td><td>0.379</td><td>0.568</td><td>0.346</td><td>0.342</td></tr><tr><td>Concentric Hyperspheres (Accuracy)</td><td>42.56</td><td>64.52</td><td>56.76</td><td>53.35</td><td>96.91</td><td>99.99</td></tr></table>

Table 3: Quantitative Results in synthetic disjoint MTL scenarios. TOM learns variable embeddings that enable it to outperform alternative approaches, and achieve performance on par with the Oracle.

![](images/de8b75c5b71e7d0e4c56f864eb3ff13f7feba88f4d86ac598dd2175d22a69eaf.jpg)  
Figure 5: Learned VEs capture underlying structure across tasks. (a) VEs of features for concentric hyperspheres encode the origin location, and (b) for classes encode the index of their annuli (less precisely for the more distant annuli, since they occur in fewer tasks); (c) VEs for UCI-121 (shown in 2D via t-SNE) neatly carve the space into features, common classes, and uncommon classes.

except it is trained on each task independently; (4)  $SLO$  (Soft Layer Ordering; Meyerson & Mikkulainen, 2018), which uses a separate encoder and decoder for each task, and which is (as far as we know) the only prior Deep MTL approach that has been applied across disjoint tabular datasets; and (5) Oracle, i.e. TOM with VEs fixed to intuitively correct values. The Oracle is included to give an upper bound on how well the TOM architecture in Section 4.1 could possibly perform. The oracle VE for each Transposed GP task variable is the location where it is measured in the GP; for Concentric Hyperspheres, the oracle VE for each class  $c$  is  $c / 10$ , and for the  $i$ th feature is  $o_i^t$ .

TOM outperforms the competing methods and achieves performance on par with the Oracle (Table 3). Note that the improvement of TOM over TOM-STL is much greater than that of DR-MTL over DR-STL, indicating that TOM is particularly well-suited to exploiting structure across disjoint data sets (learned VEs are shown in Figure 5(a-b)). Now that this suitability has been confirmed, the next section evaluates TOM across a suite of disjoint, and seemingly unrelated, real-world problems.

# 4.4 MULTI-TASK LEARNING ACROSS SEEMINGLY UNRELATED REAL-WORLD DATASETS

This section evaluates TOM in the setting for which it was designed: learning a single shared model across seemingly unrelated real-world datasets. The set of tasks used is UCI-121 (Lichman, 2013; Fernández-Delgado et al., 2014), a set of 121 classification tasks that has been previously used to evaluate the overall performance of a variety of deep NN methods (Klambauer et al., 2017). The tasks come from diverse areas such as medicine, geology, engineering, botany, sociology, politics, and game-playing. Prior work has tuned each model to each task individually in the single-task regime; no prior work has undertaken learning of all 121 tasks in a single joint model. The datasets are highly diverse. Each simply defines a classification task that a machine learning practitioner was interested in solving. The number of features for a task range from 3 to 262, the number of classes from 2 to 100, and the number of samples from 10 to 130,064. To avoid underfitting to the larger tasks,  $C = 128$ , and after joint training the model is finetuned on each task with at least 5K samples.

Results across a suite of metrics are shown in Table 4. Mean Accuracy is the test accuracy averaged across all tasks. Normalized Accuracy scales the accuracy within each task before averaging across tasks, with 0 and 100 corresponding to the lowest and highest accuracies. Mean Rank averages the method's rank across tasks, where the best method gets a rank of 0. Best % is the percentage of tasks for which the method achieves the top accuracy (with possible ties). Win % is the percentage of tasks for which the method achieves accuracy strictly greater than all other methods. TOM outperforms the alternative approaches across all metrics, showing its ability to learn many seemingly unrelated tasks successfully in a single model (see Figure 5(c) for a high-level visualization of learned VEs). In other words, TOM can both learn meaningful VEs and use them to improve prediction performance.

<table><tr><td rowspan="6">(a)</td><td>Method</td><td>BN</td><td>WN</td><td>ResNet</td><td>HW</td><td>LN</td><td>MS</td><td>SNN</td><td>TOM</td></tr><tr><td>Norm. Acc.</td><td>42.15</td><td>45.87</td><td>50.07</td><td>53.00</td><td>56.73</td><td>60.11</td><td>65.29</td><td>70.72</td></tr><tr><td>Best %</td><td>13.22</td><td>10.74</td><td>12.40</td><td>15.70</td><td>16.53</td><td>14.88</td><td>21.49</td><td>34.71</td></tr><tr><td>Win %</td><td>5.79</td><td>7.44</td><td>3.31</td><td>8.26</td><td>9.92</td><td>4.96</td><td>13.22</td><td>28.93</td></tr><tr><td>Mean Rank</td><td>4.202</td><td>4.054</td><td>3.892</td><td>3.613</td><td>3.446</td><td>3.350</td><td>2.775</td><td>2.599</td></tr><tr><td>Mean Acc.</td><td>77.01</td><td>77.43</td><td>79.24</td><td>78.68</td><td>79.85</td><td>80.11</td><td>81.39</td><td>81.53</td></tr><tr><td rowspan="6"></td><td rowspan="6">(b)</td><td>Method</td><td>DR-STL</td><td>TOM-STL</td><td>DR-MTL</td><td>SLO</td><td>TOM</td><td></td><td></td></tr><tr><td>Norm. Acc.</td><td>54.51</td><td>34.83</td><td>56.95</td><td>73.66</td><td>77.01</td><td></td><td></td></tr><tr><td>Best %</td><td>19.83</td><td>17.36</td><td>28.93</td><td>30.58</td><td>47.93</td><td></td><td></td></tr><tr><td>Win %</td><td>10.74</td><td>7.44</td><td>9.09</td><td>15.70</td><td>32.23</td><td></td><td></td></tr><tr><td>Mean Rank</td><td>2.310</td><td>2.723</td><td>2.004</td><td>1.632</td><td>1.326</td><td></td><td></td></tr><tr><td>Mean Acc.</td><td>76.44</td><td>68.14</td><td>78.45</td><td>80.31</td><td>81.53</td><td></td><td></td></tr></table>

Table 4: UCI-121 Results. (a) Comparisons to external results of deep STL models tuned to each task (Klambauer et al., 2017); (b) Comparisons across methods evaluated in this paper. Metrics are aggregated over all 121 tasks. TOM achieves high performance across seemingly unrelated tasks, outperforming the comparisons across all metrics.

# 5 DISCUSSION AND FUTURE WORK

This paper introduced the TOM approach, illustrated its capabilities, and demonstrated its value as a general multitask learning system. This section discusses three key areas of future work for increasing the understanding and applicability of the approach.

First, there is an opportunity to develop a theoretical framework for understanding when TOM will work best. It is straightforward to extend universal approximation results from approximation of single functions (Cybenko, 1989; Lu et al., 2017; Kidger & Lyons, 2020) to approximation of a set of functions each with any input and output dimensionality via Eq. 2. It is also straightforward to extend convergence bounds for certain model classes, such as PAC bounds (Bartlett & Mendelson, 2002; Neyshabur et al., 2018), to TOM architectures implemented with these classes, if the "true" variable embeddings are fixed a priori, so they can simply be treated as features. However, a more intriguing direction involves understanding how the true locations of variables affects TOM's ability to learn and exploit them, i.e., what are desirable theoretical properties of the space of variables?

Second, in this paper, TOM was evaluated only in the case when the data for all tasks is always available, and the model is trained simultaneously across all tasks. However, it would also be natural to apply TOM in a meta-learning regime (Finn et al., 2017; Zintgraf et al., 2019), in which the model is trained explicitly to generalize to future tasks, and to lifelong learning (Thrun & Pratt, 2012; Brunskill & Li, 2014; Abel et al., 2018), where the model must learn new tasks as they appear over time. Simply freezing the learned parameters of TOM results in a parametric class of ML models with  $C$  parameters per variable that can be applied to new tasks. However, in practice, it should be possible to improve by taking advantage of more sophisticated parameter adaptation and fine-tuning.

Third, to make the foundational case for TOM, this paper focused on the setting where VEs are a priori unknown, but when such knowledge is available, it could be useful to integrate with learned VEs. Such an approach could eliminate the cost of relearning VEs, and suggest how to take advantage of spatially-customized architectures. E.g., convolution or attention layers could be used instead of dense layers as architectural primitives, as in vision and language tasks. Such specialization could be instrumental in making TOM more broadly applicable and more powerful in practice.

# 6 CONCLUSION

This paper introduced the traveling observer model (TOM), which enables a single model to be trained across diverse tasks by embedding all task variables into a shared space. The framework was shown to discover intuitive notions of space and time and use them to learn variable embeddings that exploit knowledge across tasks, outperforming single- and multi-task alternatives. Thus, learning a single function that cares only about variable locations and their values is a promising approach to integrating knowledge across data sets that have no a priori connection. The TOM approach thus extends the benefits of multi-task learning to broader sets of tasks.

# REFERENCES

D. Abel, D. Arumugam, L. Lehnert, and M. Littman. State abstractions for lifelong reinforcement learning. In Proc. of ICML, pp. 10-19, 2018.  
B. Alipanahi, A. Delong, M. T. Weirauch, and B. J. Frey. Predicting the sequence specificities of dna-and rna-binding proteins by deep learning. Nature Biotechnology, 33(8):831, 2015.  
A. Argyriou, T. Evgeniou, and M. Pontil. Convex multi-task feature learning. Machine Learning, 73(3):243-272, 2008.  
P. L. Bartlett and S. Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research, 3(Nov):463-482, 2002.  
H. Bilen and A. Vedaldi. Universal representations: The missing link between faces, text, planktons, and cat breeds. CoRR, abs/1701.07275, 2017.  
E. Brunskill and L. Li. Pac-inspired option discovery in lifelong reinforcement learning. In Proc. of ICML, pp. 316-324, 2014.  
R. Caruana. Multitask learning. In Learning to learn, pp. 95-133. Springer US, 1998.  
R. Collobert and J. Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proc. of ICML, pp. 160-167, 2008.  
G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.  
S. De and S. L. Smith. Batch normalization biases deep residual networks towards shallow paths. arXiv preprint arXiv:2002.10444, 2020.  
P. Della-Marta, D. Collins, and K. Braganza. Updating australia's high-quality annual temperature dataset. Australian Meteorological Magazine, 53(2):75, 2004.  
D. Dong, H. Wu, W. He, D. Yu, and H. Wang. Multi-task learning for multiple language translation. In Proc. of ACL, pp. 1723-1732, 2015.  
M. Fernández-Delgado, E. Cernadas, S. Barro, and D. Amorim. Do we need hundreds of classifiers to solve real world classification problems? Journal of Machine Learning Research, 15(1):3133-3181, 2014.  
C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Proc. of ICML, pp. 1126-1135, 2017.  
K. Hashimoto, C. Xiong, Y. Tsuruoka, and R. Socher. A joint many-task model: Growing a neural network for multiple NLP tasks. In Proc. EMNLP, pp. 1923-1933, 2017.  
J. T. Huang, J. Li, D. Yu, L. Deng, and Y. Gong. Cross-language knowledge transfer using multilingual deep neural network with shared hidden layers. In Proc. of ICASSP, pp. 7304-7308, 2013.  
Z. Huang, J. Li, S. M. Siniscalchi, et al. Rapid adaptation for deep neural networks through multitask learning. In Proc. of Interspeech, pp. 3625-3629, 2015.  
M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver, and K. Kavukcuoglu. Reinforcement learning with unsupervised auxiliary tasks. In Proc. of ICLR, 2017.  
K. Janocha and W. M. Czarnecki. On loss functions for deep neural networks in classification. CoRR, abs/1702.05659, 2017.  
L. Kaiser, A. N. Gomez, N. Shazeer, Ashish Vaswani, N. Parmar, L. Jones, and J. Uszkoreit. One model to learn them all. CoRR, abs/1706.05137, 2017.  
Z. Kang, K. Grauman, and F. Sha. Learning with whom to share in multi-task feature learning. In Proc. of ICML, pp. 521-528, 2011.

P. Kidger and T. Lyons. Universal approximation with deep narrow networks. In Proc. of COLT, pp. 2306-2327, 2020.  
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014.  
G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter. Self-normalizing neural networks. In Proc. of NeurIPS, pp. 971-980, 2017.  
T. Kohonen. The self-organizing map. Proceedings of the IEEE, 78(9):1464-1480, 1990.  
A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.  
A. Kumar and H. Daumé, III. Learning task grouping and overlap in multi-task learning. In Proc. of ICML, pp. 1723-1730, 2012.  
M. Lichman. UCI machine learning repository, 2013.  
X. Liu, J. Gao, X. He, L. Deng, K. Duh, and Y. Y. Wang. Representation learning using multi-task deep neural networks for semantic classification and information retrieval. In Proc. of NAACL, pp. 912-921, 2015.  
Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang. The expressive power of neural networks: A view from the width. In Proc. of NeurIPS, pp. 6231-6239, 2017.  
M. T. Luong, Q. V. Le, I. Sutskever, O. Vinyals, and L. Kaiser. Multi-task sequence to sequence learning. In Proc. of ICLR, 2016.  
M. M. Mahmud and S. Ray. Transfer learning using Kolmogorov complexity: Basic theory and empirical evaluations. In Proc. of NeurIPS, pp. 985-992. 2008.  
M. M. H. Mahmud. On universal transfer learning. Theoretical Computer Science, 410(19):1826-1846, 2009.  
E. Meyerson and R. Miikkulainen. Beyond shared hierarchies: Deep multitask learning through soft layer ordering. In Proc. of ICLR, 2018.  
E. Meyerson and R. Miikkulainen. Modular universal reparameterization: Deep multi-task learning across diverse domains. In Proc. of NeurIPS, pp. 7903-7914, 2019.  
I. Misra, A. Shrivastava, A. Gupta, and M. Hebert. Cross-stitch networks for multi-task learning. In Proc. of CVPR, 2016.  
B. Neyshabur, S. Bhojanapalli, and N. Srebro. A pac-bayesian approach to spectrally-normalized margin bounds for neural networks. In Proc. of ICLR, 2018.  
A. Paske et al. Automatic differentiation in pytorch. 2017.  
K. Pearson. LIII. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 2(11):559-572, 1901.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825-2830, 2011.  
E. Perez, F. Strub, H. de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In Proc. of AAAI, 2018.  
R. Ranjan, V. M. Patel, and R. Chellappa. Hyperface: A deep multi-task learning framework for face detection, landmark localization, pose estimation, and gender recognition. CoRR, abs/1603.01249, 2016.  
S.-A. Rebuffi, H. Bilen, and A. Vedaldi. Learning multiple visual domains with residual adapters. In NeurIPS, pp. 506-516. 2017.

M. L. Seltzer and J. Droppo. Multi-task learning in deep neural networks for improved phoneme recognition. In Proc. of ICASSP, pp. 6965-6969, 2013.  
Y. Teh, V. Bapat, W. M. Czarnecki, J. Quan, J. Kirkpatrick, R. Hadsell, N. Heess, and R. Pascanu. Distral: Robust multitask reinforcement learning. In Proc. of NeurIPS, pp. 4499-4509. 2017.  
S. Thrun and L. Pratt. Learning to Learn. 2012.  
L. van der Maaten and G. Hinton. Visualing data using t-sne. Journal of Machine Learning Research, 9:2579-2605, Nov 2008.  
Y. Yang and T. M. Hospedales. A unified perspective on multi-domain and multi-task learning. In Proc. of ICLR, 2014.  
Z. Zhang, L. Ping, L. C. Chen, and T. Xiaou. Facial landmark detection by deep multi-task learning. In Proc. of ECCV, pp. 94-108, 2014.  
L. Zintgraf, K. Shiarli, V. Kurin, K. Hofmann, and S. Whiteson. Fast context adaptation via meta-learning. In Proc. of ICML, pp. 7693-7702, 2019.
