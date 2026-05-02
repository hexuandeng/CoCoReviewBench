# LEARNING UNSUPERVISED LEARNING RULES

Anonymous authors

Paper under double-blind review

# ABSTRACT

A major goal of unsupervised learning is to discover data representations that are useful for subsequent tasks, without access to supervised labels during training. Typically, this goal is approached by minimizing a surrogate objective, such as the negative log likelihood of a generative model, with the hope that representations useful for subsequent tasks will arise incidentally. In this work, we propose instead to directly target a later desired task by meta-learning an unsupervised learning rule, which leads to representations useful for that task. Here, our desired task (meta-objective) is the performance of the representation on semi-supervised classification, and we meta-learn an algorithm – an unsupervised weight update rule – that produces representations that perform well under this meta-objective. Additionally, we constrain our unsupervised update rule to be a biologically-motivated, neuron-local function, which enables it to generalize to novel neural network architectures. We show that the meta-learned update rule produces useful features and sometimes outperforms existing unsupervised learning techniques. We further show that the meta-learned unsupervised update rule generalizes to train networks with different widths, depths, and nonlinearities. It also generalizes to train on data with randomly permuted input dimensions and even generalizes from image datasets to a text task.

# 1 INTRODUCTION

Supervised learning has proven extremely effective for many problems where large amounts of labeled training data are available. There is a common hope that unsupervised learning will prove similarly powerful in situations where labels are expensive, impractical to collect, or where the prediction target is unknown during training. Unsupervised learning however has yet to fulfill this promise. One explanation for this failure is that unsupervised representation learning algorithms are typically mismatched to the target task. Ideally, learned representations should linearly expose high level attributes of data (e.g. object identity) and perform well in semi-supervised settings. Many current unsupervised objectives, however, optimize for objectives such as log-likelihood of a generative model or reconstruction error and produce useful representations only as a side effect.

Unsupervised representation learning seems uniquely suited for meta-learning (Hochreiter et al., 2001; Schmidhuber, 1995). Unlike most tasks where meta-learning is applied, unsupervised learning does not define an explicit objective, which makes it impossible to phrase the task as a standard optimization problem. It is possible, however, to directly express a meta-objective that captures the quality of representations produced by an unsupervised update rule by evaluating the usefulness of the representation for candidate tasks, e.g. semi-supervised classification. In this work, we propose to meta-learn an unsupervised update rule by meta-training on a meta-objective that directly optimizes the utility of the unsupervised representation. Unlike hand-designed unsupervised learning rules, this meta-objective directly targets the usefulness of a representation generated from unlabeled data for later supervised tasks.

By recasting unsupervised representation learning as meta-learning, we treat the creation of the unsupervised update rule as a transfer learning problem. Instead of learning transferable features, we learn a transferable learning rule which does not require access to labels and generalizes across both data domains and neural network architectures. Although we focus on the meta-objective of semi-supervised classification here, in principle a learning rule could be optimized to generate representations for any subsequent task.

# 2 RELATED WORK

# 2.1 UNSUPERVISED REPRESENTATION LEARNING

There is no single accepted definition of unsupervised learning, and it is a topic of broad and diverse interest. Here we briefly present several techniques that can lead to a useful latent representation of a dataset. In contrast to our work, each method imposes a manually defined training algorithm or loss function to optimize whereas we learn the algorithm that creates useful representations as determined by a meta-objective.

Autoencoders (Hinton and Salakhutdinov, 2006) work by first compressing and optimizing reconstruction loss. Extensions have been made to de-noise data (Vincent et al., 2008; 2010), as well as compress information in an information theoretic way (Kingma and Welling, 2013). Le et al. (2011) further explored scaling up these unsupervised methods to large image datasets.

Generative adversarial networks (Goodfellow et al., 2014) take another approach to unsupervised feature learning. Instead of a loss function, an explicit min-max optimization is defined to learn a generative model of a data distribution. Recent work has shown that this training procedure can learn unsupervised features useful for few shot learning (Radford et al., 2015; Donahue et al., 2016; Dumoulin et al., 2016).

Other techniques rely on self-supervision where labels are easily generated to create a non-trivial 'supervised' loss. Domain knowledge of the input is often necessary to define these losses. Noroozi and Favaro (2016) use unscrambling jigsaw-like crops of an image. Techniques like Misra et al. (2016) and Sermanet et al. (2017) rely on using temporal ordering from videos.

Another approach to unsupervised learning relies on feature space design such as clustering. Coates and Ng (2012) showed that k-means can be used for feature learning. Xie et al. (2016) jointly learn features and cluster assignments. Bojanowski and Joulin (2017) develop a scalable technique to cluster by predicting noise. Other techniques such as Schmidhuber (1992), Hochreiter and Schmidhuber (1999), and Olshausen and Field (1997) define various desirable properties about the latent representation of the input, such as predictability, complexity of encoding mapping, independence, or sparsity, and optimize to achieve these properties.

# 2.2 META LEARNING

Most meta-learning algorithms consist of two levels of learning, or 'loops' of computation: an inner loop, where some form of learning occurs (e.g. an optimization process), and an outer loop or meta-training loop, which optimizes some aspect of the inner loop. We call the aspects of the inner loop modified by the outer loop meta-parameters. The performance of the inner loop computation is quantified by a meta-objective. Meta-optimization is then the process of adjusting the meta-parameters so that the inner loop performs well on this meta-objective. Meta-learning approaches differ by the computation performed in the inner loop, the domain, the choice of meta-parameters, and the method of optimizing the outer loop.

Some of the earliest work in meta-learning includes (Schmidhuber, 1987) which explores a variety of meta-learning and self-referential algorithms. Similarly to our algorithm Bengio et al. (1990; 1992) propose to learn a neuron local learning rule, though their approach differs in task and problem formulation. Additionally, Runarsson and Jonsson (2000) meta-learn supervised learning rules which mix local and global network information. A number of papers propose meta-learning for few shot learning (Vinyals et al., 2016; Ravi and Larochelle, 2016; Mishra et al., 2017; Finn et al., 2017; Snell et al., 2017), though these do not take advantage of unlabeled data. In (Garg and Kalai, 2017) meta-learning is used for unsupervised learning, primarily in the context of clustering and with a small number of meta-parameters. To allow easy comparison against other existing approaches, we present a more extensive survey of previous work in meta-learning in table form in Appendix A, highlighting differences in choice of task, structure of the meta-learning problem, choice of meta-architecture, and choice of domain.

To our knowledge, we are the first meta-learning approach to tackle the problem of unsupervised representation learning, where the inner loop consists of unsupervised learning. This contrasts with transfer learning, where a neural network is instead trained on a similar dataset, and then fine tuned or

![](images/9e71d4490d0670311a7806f2bf94923ac824706855d08660bd49cba5ce040d8e.jpg)  
Figure 1: Left: Schematic for meta-learning an unsupervised learning algorithm. The inner loop computation consists of iteratively applying the UnsupervisedUpdate to a base model. During meta-training the UnsupervisedUpdate (parameterized by  $\theta$ ) is itself updated by gradient descent on the MetaObjective. Right: Schematic of the base model and UnsupervisedUpdate. Unlabeled input data,  $x_0$ , is passed through the base model, which is parameterised by  $W$  and colored green. The goal of the UnsupervisedUpdate is to modify  $W$  to achieve a top layer representation  $x^L$  which performs well at few-shot learning. In order to train the base model, information is propagated backwards by the UnsupervisedUpdate in a manner analogous to backprop. Unlike in backprop however, the backward weights  $V$  are decoupled from the forward weights  $W$ , and there is no explicit error signal. Instead at each layer, and for each neuron, a learning signal is injected by a meta-learned MLP parameterized by  $\theta$ , with hidden state  $h$ . Weight updates are again analogous to those in backprop, and depend on the hidden state of the pre- and post- synaptic neurons for each weight.

![](images/eedf2d8c9901f71d7bd3a90d0b00927e61faeeca4c43289756b078e55e501944.jpg)  
eannnnnne nnnnne

otherwise post-processed on the target dataset. We additionally believe we are the first representation meta-learning approach to generalize across input data modalities as well as datasets, the first to generalize across permutation of the input dimensions, and the first to generalize across neural network architectures (e.g. layer width, network depth, activation function).

# 3 MODEL DESIGN

We consider a multilayer perceptron (MLP)  $f(\cdot; \phi_t)$ , with parameters  $\phi_t$ , as the base model. The inner loop of our meta-learning process trains this base model via iterative application of our learned optimizer. See Figure 1 for a schematic illustration and Appendix B for a more detailed diagram.

In standard supervised learning, the 'learned' optimizer is stochastic gradient descent (SGD). A supervised loss  $l(x,y)$  is associated with this model, where  $x$  is a minibatch of inputs, and  $y$  are the corresponding labels. The parameters  $\phi_t$  of the base model are then updated iteratively by performing SGD using the gradient  $\frac{\partial l(x,y)}{\partial\phi_t}$ . This supervised update rule can be written as  $\phi_{t + 1} = \mathrm{SupervisedUpdate}(\phi_t,x_t,y_t;\theta)$ . Here  $\theta$  are the meta-parameters of the optimizer, which consist of hyper-parameters such as learning rate and momentum.

In this work, our learned optimizer is a parametric update process which does not depend on label information,  $\phi_{t + 1} = \mathrm{UnsupervisedUpdate}(\phi_t,x_t;\theta)$ . This form of the update rule is general. It encompasses many unsupervised learning algorithms and all methods in Section 2.1.

In traditional unsupervised learning algorithms, expert knowledge or a simple hyper-parameter search determines  $\theta$ , which consists of a handful of meta-parameters such as learning rate and regularization constants. In contrast, our update rule will have orders of magnitude more meta-parameters, including the weights of a neural network. We train these meta-parameters by performing SGD on the sum of the MetaObjective over the course of (inner loop) training in order to find optimal parameters  $\theta^{*}$ ,

$$
\theta^ {*} = \underset {\boldsymbol {\theta}} {\operatorname {a r g m i n}} \mathbb {E} _ {\text {t a s k}} \left[ \sum_ {t} \operatorname {M e t a O b j e c t i v e} \left(\phi_ {t} (\theta)\right) \right], \tag {1}
$$

that minimize the meta-objective over a set of training tasks.

In the following sections, we briefly review the main functional pieces to this model: the base model  $f(\cdot ,\phi)$ , the UnsupervisedUpdate, and the MetaObjective. See the Appendix for a complete specification. Additionally, code and meta-trained parameters  $\theta$  for our meta-learned UnsupervisedUpdate will be available.

# 3.1 BASE MODEL:  $f(\cdot ;\phi)$

Our base model consists of a standard fully connected multi-layer perceptron (MLP), with batch normalization (Ioffe and Szegedy, 2015), and ReLU nonlinearities. We chose this as opposed to a convolutional model to limit the inductive bias hard-coded in favor of learned behavior from the UnsupervisedUpdate. We call the pre-nonlinearity activations  $z^1, \dots, z^L$ , and post-nonlinearity activations  $x^0, \dots, x^L$ , where  $L$  is the total number of layers, and  $x^0 \equiv x$  is the network input (raw data). The parameters are  $\phi = \{W^1, b^1, V^1, \dots, W^L, b^L, V^L\}$ , where  $W^l$  and  $b^l$  are the weights and biases (applied after batch norm) for layer  $l$ , and  $V^l$  are the corresponding weights used in the backward pass.

# 3.2 LEARNED UPDATE RULE: UNSUPERVISEDUPDATE  $(\cdot ;\theta)$

We wish for our update rule to generalize across architectures with different widths, depths, or even network topologies. To achieve this, we design our update rule to be neuron-local, so that updates are a function of pre- and post- synaptic neurons in the base model, and are defined for any base model architecture. This has the added benefit that it makes the weight updates more similar to synaptic updates in biological neurons, which depend almost exclusively on the pre- and post-synaptic neurons for each synapse (Whittington and Bogacz, 2017).

To build these updates, each neuron  $i$  in every layer  $l$  in the base model has an MLP, referred to as an update network, associated with it, with output  $h_i^l (\cdot ;\theta)$ . All update networks share meta-parameters  $\theta$ , and  $h_i^l (\cdot ;\theta)$  is evaluated only during unsupervised training as the update networks are part of the UnsupervisedUpdate, and not part of the base model. Evaluating the statistics of unit activation over a batch of data has proven helpful in supervised learning (Ioffe and Szegedy, 2015). It has similarly proven helpful in hand-designed unsupervised learning rules, such as sparse coding and clustering. We therefore allow  $h_i^l (\cdot ;\theta)$  to accumulate statistics across examples in each training minibatch.

During an unsupervised training step, the base model is first run in a standard feed-forward fashion, populating  $x_{ib}^{l}, z_{ib}^{l}$ , where  $b$  is the training minibatch index. As in supervised learning, an error signal  $\delta_{ib}^{l}$  is then propagated backwards through the network. Unlike in supervised backprop however, this error signal is generated by the corresponding update network for each unit,  $\delta_{ib}^{l} \gets h_{i}^{l}(\cdot;\theta)$ , and propagated backward using a set of learned 'backward weights'  $(V^{l})^{T}$ , as opposed to the transpose of the forward weights  $(W^{l})^{T}$  as would be the case in backprop. Pictorially this can be seen in Figure 1.

Again as in supervised learning, the weight updates are a product of pre- and post-synaptic signals. Unlike in supervised learning however, these signals are generated by the per-neuron update networks:  $\Delta W_{ij}^{l} = \sum_{b}c_{ib}^{l}d_{jb}^{l - 1}$ , where  $\{c_{ib}^{l},d_{ib}^{l}\} \gets h_{i}^{l}(:,\theta)$ .

The inputs to the update network consists of unit pre- and post-activations, and backwards propagated error signal:  $h_i^l\left(x_i^l,z_i^l,\left[(V^{l + 1})^T\delta^{l + 1}\right]_i;\theta\right)$ . The backward weights also are updated with the same update network. For more information on the structure of  $h_i^l$ , see Appendix F.

# 3.3 METAOBJECTIVE  $(\phi)$

The meta-objective determines the quality of the unsupervised representations. In order to meta-train via SGD, this loss must be differentiable. The meta-objective we use in this work is based on fitting a linear regression to labeled examples with a small number of data points. In order to encourage the learning of features that generalize well, we estimate the linear regression weights on one minibatch  $\{x_{a},y_{a}\}$  of  $K$  data points, and evaluate the classification performance on a second minibatch  $\{x_{b},y_{b}\}$  also with  $K$  datapoints,

$$
\hat {v} = \underset {v} {\operatorname {a r g m i n}} \left(\left\| y _ {a} - v ^ {T} u _ {a} \right\| ^ {2} + \lambda \| v \| ^ {2}\right), \quad \text {M e t a O b j e c t i v e} (\cdot ; \phi) = \operatorname {C o s D i s t} \left(y _ {b}, \hat {v} ^ {T} u _ {b}\right), \tag {2}
$$

where  $u_{a}$ ,  $u_{b}$  are features extracted from the base model on data  $x_{a}$ ,  $x_{b}$ , respectively. The target labels  $y_{a}$ ,  $y_{b}$  consist of one hot encoded labels and potentially also regression targets from data augmentation (e.g. rotation angle, see Section 4.2). We found that using a cosine distance, CosDist, rather than unnormalized squared error improved stability. Note this meta-objective is only used

during meta-training and not used when applying the learned update rule. The inner loop computation is performed without labels.

# 4 TRAINING THE UPDATE RULE

# 4.1 APPROXIMATE GRADIENT BASED TRAINING

We choose to meta-optimize via SGD as opposed to reinforcement learning or other black box method, due to the superior convergence properties of SGD in high dimensions, and the high dimensional nature of  $\theta$ . Training and computing derivatives through long recurrent computation of this form is notoriously difficult (Pascanu et al., 2013). To improve stability and reduce the computational cost we approximate the gradients  $\frac{\partial[\mathrm{MetaObjective}]}{\partial\theta}$  via truncated backprop through time. Many additional design choices were also crucial to achieving stability and convergence in meta-learning, including the use of batch norm, and restricting the norm of the UnsupervisedUpdate update step. Meta-learning practitioners may find the discussion of these and other choices in Appendix C valuable.

# 4.2 META-TRAINING DISTRIBUTION AND GENERALIZATION

Generalization in our learned optimizer comes from both the form of the UnsupervisedUpdate (Section 3.2), and from the meta-training distribution. Our meta-training distribution is composed of both datasets and base model architectures.

We construct a set of training tasks consisting of CIFAR10 (Krizhevsky and Hinton, 2009) and multi-class classification from subsets of classes fromImagenet (Russakovsky et al., 2015) as well as from a dataset consisting of rendered fonts. We find that increased training dataset variation actually improves the meta-optimization process. To reduce computation we restrict the input data to  $16 \times 16$  pixels or less during meta-training, and resize all datasets accordingly. For evaluation, we use MNIST(LeCun et al., 1998), Fashion MNIST(Xiao et al., 2017), IMDB(Maas et al., 2011), and a hold-out set ofImagenet classes. We additionally sample the base model architecture. We sample number of layers between 2-5 and the number of units per layer between 64 to 512.

As part of preprocessing, we additionally permute all inputs along the feature dimension, so that the UnsupervisedUpdate must learn a permutation invariant learning rule. Unlike other work, we focus explicitly on learning a learning algorithm as opposed to the discovery of fixed feature extractors that generalize across similar tasks. This makes the learning task much harder, as the UnsupervisedUpdate has to discover the relationship between pixels based solely on their joint statistics, and cannot "cheat" and memorize pixel identity. To provide further dataset variation, we additionally augment the data with shifts, rotations, and noise. We add these augmentation coefficients as additional regression targets for the meta-objective - e.g. rotate the image and predict the rotation angle as well as the image class. For additional details, see Appendix I.1.1.

# 4.3 DISTRIBUTED IMPLEMENTATION

We implement the above models in distributed Tensorflow (Abadi et al., 2016). Training uses 512 workers, each of which performs a sequence of partial unrolls of the inner loop UnsupervisedUpdate, and computes gradients of the meta-objective asynchronously. Training takes  $\sim 8$  days, and consists of  $\sim 200$  thousand updates to  $\theta$  with minibatch size 256. Additional details are in Appendix D.

# 5 EXPERIMENTAL RESULTS

First, we examine limitations of existing unsupervised and meta learning methods, next we show metatraining and generalization properties of our learned optimizer. Finally we conclude by exploring how our learned optimizer functions. For details of experimental setup, see Appendix I.

# 5.1 OBJECTIVE FUNCTION MISMATCH AND EXISTING META-LEARNING METHODS

To illustrate the negative consequences of objective function mismatch in unsupervised learnin algorithms, we train a variational autoencoder on 16x16 CIFAR10. Over the course of training we

evaluate classification performance from few shot classification using the learned latent representations. Training curves can be seen in Figure 2. Despite continuing to improve the VAE objective throughout training (not shown here), the classification accuracy decreases sharply later in training.

To demonstrate the reduced generalization that results from learning transferable features rather than an update algorithm, we train a prototypical network with and without the input shuffling described in Section 4.2. Due to the transferable features structure of prototypical network, performance is significantly hampered by input shuffling. Results are in Figure 2.

![](images/85207e3a80fd62c594ce1e75f437dbad4e69bdeb9e83ee428e85b6b3da3b54f5.jpg)  
Figure 2: Left: Standard unsupervised learning approaches suffer from objective function mismatch. Continuing to optimize a variational auto-encoder (VAE) hurts few-shot accuracy after some number of steps (dashed line). Right: Prototypical networks transfer features rather than a learning algorithm, and perform poorly if tasks don't have consistent data structure. Training a prototypical network with a fully connected architecture (same as our base model) on a miniImagenet 10-way classification task with either intact inputs (light purple) or by permuting the pixels before every training and testing task (dark purple). Performance with permuted inputs is greatly reduced (gray line). Our performance is invariant to pixel permutation.

![](images/9301f883f42f882953f17b671a15a413806de7880d8bd1a69a858d777ed42fc9.jpg)

![](images/4a4c4a74e7a08c3c01db911584512ad00da904e4178118916cb6daca43deec04.jpg)  
Figure 3: Training curves for the training and evaluation task distributions. Our train set consists of Mini Imagenet, Alphabet, and MiniCIFAR. Our test sets are Mini Imagenet Test, Tiny Fashion MNIST, Tiny MNIST and IMDB. Error bars denote standard deviation of evaluations with a fixed window of samples evaluated from a single model. Dashed line at 200 hours indicates model used for remaining experiments unless otherwise stated. For a bigger version, see Appendix F.

![](images/92419705d97a1de7d6491622ea35bf8e7b8d80bba0c14be3b942068f9bdc9429.jpg)

# 5.2 META-OPTIMIZATION

While training, we monitor a rolling average of the meta-objective averaged across all datasets, model architectures, and the number of unrolling steps performed. In Figure 3 the training loss is continuing to decrease after 200 hours of training, which suggests that the approximate training techniques still produce effective learning. In addition to this global number, we measure performance obtained by rolling out the UnsupervisedUpdate on various training and testing datasets. We see that on held out image datasets, such as MNIST and Fashion Mnist, the evaluation loss is still decreasing. However, for datasets in a different domain, such as IMDB sentiment prediction (Maas et al., 2011), we start to see over-fitting. For all remaining experimental results, unless otherwise stated, we use meta-parameters,  $\theta$ , for the UnsupervisedUpdate resulting from 200 hours of meta-training.

# 5.3 GENERALIZATION

The goal of this work is to learn a general purpose unsupervised representation learning algorithm. As such, this algorithm must be able to generalize across a wide range of scenarios, including tasks that are not sampled iid from the meta-training distribution. In the following sections, we explore a subset of the factors we seek to generalize over.

# Generalizing over datasets and domains

In Figure 4, we compare performance on few shot classification with 10 examples per class. We evaluate test performance on holdout datasets of MNIST and Fashion MNIST at 2 resolutions:  $14 \times 14$  and  $28 \times 28$  (larger than any dataset experienced in meta-training). On the same base model architecture, our learned UnsupervisedUpdate leads to performance better than a variational autoencoder, supervised learning on the labeled examples, and random initialization with trained readout layer.

![](images/a5993834717c9e7325fb7a3833a7eae65502e6e5ccb24c7f47f050a803e2bad2.jpg)  
Figure 4: Left: The learned UnsupervisedUpdate generalizes to unseen datasets. Our learned update rule produces representations more suitable for few shot classification than those from random initialization or a variational autoencoder and outperforms fully supervised learning on the same labeled examples. Error bars show standard error. Right: Early in meta-training, the UnsupervisedUpdate is able to learn useful features on a 2 way text classification data set, IMDB, despite being trained only from image datasets. Later in meta-training performance drops due to the domain mismatch. Error bars show standard error across 10 runs.

![](images/033d84e452d331d1ec1a3f46fb7ad1eca22a478f17270dd232ff091b707fc25f.jpg)

To further explore generalization limits, we test our learned optimizer on data from a vastly different domain. We train on a binary text classification dataset: IMDB movie reviews (Maas et al., 2011), encoded by computing a bag of words with 1K words. We evaluate using a model 30 hours and 200 hours into meta-training (see Figure 4). Despite being trained exclusively on image datasets, the 30 hour learned optimizer improves upon the random initialization by almost  $10\%$ . When metatraining for longer, however, the learned optimizer "over-fits" to the image domain resulting in poor performance. This performance is quite low in an absolute sense. Nevertheless, we find this result very exciting as we are unaware of any work showing this kind of transfer of learned rules from images to text.

# Generalizing over network architectures

We train models of varying depths and unit counts with our learned optimizer and compare results at different points in training (Figure 5). We find that despite only training on networks with 2 to 5 layers and 64 to 512 units per layer, the learned rule generalizes to 11 layers and 10,000 units per layer.

![](images/ca0efae202272000e11d71d0c025925492804569dd0953552325a259dd3df021.jpg)  
Figure 5: Left: The learned UnsupervisedUpdate is capable of optimizing base models with hidden sizes and depths outside the meta-training regime. As we increase the number of units per layer, the learned model can make use of this additional capacity despite never having experienced it during meta-training. Right: The learned UnsupervisedUpdate generalizes across many different activation functions not seen in training. We show accuracy over the course of training on 14x14 MNIST.

![](images/da539c446aa167a763dd908efec4077b96f588adbc6afe1aa5ea8093e7804d98.jpg)

Next we look at generalization over different activation functions. We apply our learned optimizer on base models with a variety of different activation functions. Performance evaluated at different points in training (Figure 5). Despite training only on ReLU activations, our learned optimizer is able

to improve on random initializations in all cases. For certain activations, leaky ReLU (Maas et al., 2013) and Swish (Ramachandran et al., 2017), there is little to no decrease in performance. Another interesting case is the step activation function. These activations are traditionally challenging to train as there is no useful gradient signal. Despite this, our learned UnsupervisedUpdate is capable of optimizing as it does not use base model gradients, and achieves performance double that of random initialization.

# 5.4 HOW IT LEARNS AND HOW IT LEARNS TO LEARN

To analyze how our learned optimizer functions, we analyze the first layer filters over the course of meta-training. Despite the permutation invariant nature of our data (enforced by shuffling input image pixels before each unsupervised training run), the base model learns features such as those shown in Figure 6, which appear template-like for MNIST, and local-feature-like for CIFAR10. Early in training, there are coarse features, and a lot of noise. As the meta-training progresses, more interesting and local features emerge.

In an effort to understand what our algorithm learns to do, we fed it data from the two moons dataset. We find that despite being a 2D dataset, dissimilar from the image datasets used in meta-training, the learned model is still capable of manipulating and partially separating the data manifold in a purely unsupervised manner (Figure 6). We also find that almost all the variance in the embedding space is dominated by a few dimensions. As a comparison, we do the same analysis on MNIST. In this setting, the explained variance is spread out over more of the principal components. This makes sense as the generative process contains many more latent dimensions – at least enough to express the 10 digits.

![](images/7d8e3ce2cf18ee0cb23f5a070f8f30d72038329e73f7b978d4ee1caf109cd78f.jpg)  
Figure 6: Left: From left to right we show first layer base model receptive fields produced by our learned UnsupervisedUpdate rule over the course of meta-training. Each pane consists of first layer filters extracted from  $\phi$  after 10k applications of UnsupervisedUpdate on MNIST (top) and CIFAR10 (bottom). For MNIST, the optimizer learns image-template-like features. For CIFAR10, low frequency features evolve into higher frequency and more spatially localized features. For more filters, see Appendix E. Center: Visualization of learned representations before (left) and after (right) training a base model with our learned UnsupervisedUpdate for two moons (top) and MNIST (bottom). The UnsupervisedUpdate is capable of manipulating the data manifold, without access to labels, to separate the data classes. Visualization shows a projection of the 32-dimensional representation of the base network onto the top three principal components. Right: Cumulative variance explained using principal components analysis (PCA) on the learned representations. The representation for two moons data (red) is much lower dimensional than MNIST (blue), although both occupy a fraction of the full 32-dimensional space.

![](images/5f7f6b3cab7c1d82bcab8dafca27e78f6d4f0abbad0b99ce0ee614bd3c2d7372.jpg)

# 6 DISCUSSION

In this work we meta-learn an unsupervised representation learning update rule. We show performance that matches or exceeds existing unsupervised learning on held out tasks. Additionally, the update rule can train models of varying widths, depths, and activation functions. More broadly, we demonstrate an application of meta-learning for learning complex optimization tasks where no objective is explicitly defined. Analogously to how increased data and compute have powered supervised learning, we believe this work is a proof of principle that the same can be done with algorithm design - replacing the hand designed techniques with architectures designed for learning and learned from data via meta-learning techniques.

# REFERENCES

Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to learn using gradient descent. In International Conference on Artificial Neural Networks, pages 87-94. Springer, 2001.  
Juergen Schmidhuber. On learning how to learn learning strategies. 1995.  
Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504-507, 2006.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pages 1096-1103. ACM, 2008.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research, 11(Dec):3371-3408, 2010.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Quoc V Le, Marc'Aurelio Ranzato, Rajat Monga, Matthieu Devin, Kai Chen, Greg S Corrado, Jeff Dean, and Andrew Y Ng. Building high-level features using large scale unsupervised learning. arXiv preprint arXiv:1112.6209, 2011.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672-2680, 2014.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Jeff Donahue, Philipp Krahenbuhl, and Trevor Darrell. Adversarial feature learning. arXiv preprint arXiv:1605.09782, 2016.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. arXiv preprint arXiv:1606.00704, 2016.  
Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In European Conference on Computer Vision, pages 69-84. Springer, 2016.  
Ishan Misra, C Lawrence Zitnick, and Martial Hebert. Shuffle and learn: unsupervised learning using temporal order verification. In European Conference on Computer Vision, pages 527-544. Springer, 2016.  
Pierre Sermanet, Corey Lynch, Jasmine Hsu, and Sergey Levine. Time-contrastive networks: Self-supervised learning from multi-view observation. arXiv preprint arXiv:1704.06888, 2017.  
Adam Coates and Andrew Y Ng. Learning feature representations with k-means. In Neural networks: Tricks of the trade, pages 561-580. Springer, 2012.  
Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In International conference on machine learning, pages 478-487, 2016.  
Piotr Bojanowski and Armand Joulin. Unsupervised learning by predicting noise. arXiv preprint arXiv:1704.05310, 2017.  
Jürgen Schmidhuber. Learning factorial codes by predictability minimization. Neural Computation, 4(6): 863-879, 1992.  
Sepp Hochreiter and Jürgen Schmidhuber. Feature extraction through lococode. *Neural Computation*, 11(3): 679–714, 1999.  
Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision research, 37(23):3311-3325, 1997.  
Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta... hook. PhD thesis, Technische Universität München, 1987.  
Yoshua Bengio, Samy Bengio, and Jocelyn Cloutier. Learning a synaptic learning rule. Université de Montréal, Département d'informatique et de recherche opérationnelle, 1990.

Samy Bengio, Yoshua Bengio, Jocelyn Cloutier, and Jan Gecsei. On the optimization of a synaptic learning rule. In Preprints Conf. Optimality in Artificial and Biological Neural Networks, pages 6-8. Univ. of Texas, 1992.  
Thomas Philip Runarsson and Magnus Thor Jonsson. Evolution and design of distributed learning rules. In *Combinations of Evolutionary Computation and Neural Networks*, 2000 IEEE Symposium on, pages 59–63. IEEE, 2000.  
Oriol Vinyals, Charles Blundell, Tim Lillicrap, koray kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems 29, pages 3630-3638. Curran Associates, Inc., 2016. URL http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. International Conference on Learning Representations, 2016.  
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. Meta-learning with temporal convolutions. arXiv preprint arXiv:1707.03141, 2017.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1126-1135, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR. URL http://proceedings.mlrpress/v70/finn17a.html.  
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, pages 4080-4090, 2017.  
Vikas K Garg and Adam Kalai. Supervising unsupervised learning. arXiv preprint arXiv:1709.05262, 2017.  
Donald R Jones. A taxonomy of global optimization methods based on response surfaces. Journal of global optimization, 21(4):345-383, 2001.  
Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In Advances in neural information processing systems, pages 2951-2959, 2012.  
James S Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyper-parameter optimization. In Advances in neural information processing systems, pages 2546-2554, 2011.  
James Bergstra and Yoshua Bengio. Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(Feb):281-305, 2012.  
Kenneth O Stanley and Risto Miikkulainen. Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2):99-127, 2002.  
Barret Zoph and Quoc V. Le. Neural architecture search with reinforcement learning. International Conference on Learning Representations, 2017. URL https://arxiv.org/abs/1611.01578.  
Bowen Baker, Otkrist Gupta, Nikhil Naik, and Ramesh Raskar. Designing neural network architectures using reinforcement learning. International Conference on Learning Representations, 2017.  
Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V Le. Learning transferable architectures for scalable image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition.  
Esteban Real, Sherry Moore, Andrew Selle, Saurabh Saxena, Yutaka Leon Suematsu, Quoc Le, and Alex Kurakin. Large-scale evolution of image classifiers. arXiv preprint arXiv:1703.01041, 2017.  
Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning, pages 2113-2122, 2015.  
Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems, pages 3981-3989, 2016.  
Yutian Chen, Matthew W Hoffman, Sergio Gomez Colmenarejo, Misha Denil, Timothy P Lillicrap, Matt Botvinick, and Nando de Freitas. Learning to learn without gradient descent by gradient descent. arXiv preprint arXiv:1611.03824, 2016.  
Ke Li and Jitendra Malik. Learning to optimize. International Conference on Learning Representations, 2017.

Olga Wichrowska, Niru Maheswaranathan, Matthew W Hoffman, Sergio Gomez Colmenarejo, Misha Denil, Nando de Freitas, and Jascha Sohl-Dickstein. Learned optimizers that scale and generalize. International Conference on Machine Learning, 2017.  
Irwan Bello, Barret Zoph, Vijay Vasudevan, and Quoc Le. Neural optimizer search with reinforcement learning. 2017. URL https://arxiv.org/pdf/1709.07417.pdf.  
Rein Houthooft, Richard Y Chen, Phillip Isola, Bradley C Stadie, Filip Wolski, Jonathan Ho, and Pieter Abbeel. Evolved policy gradients. arXiv preprint arXiv:1802.04821, 2018.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 448-456, Lille, France, 07-09 Jul 2015. PMLR. URL http://proceedings.mlr.press/v37/iofffe15.html.  
James CR Whittington and Rafal Bogacz. An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity. Neural computation, 29(5):1229-1262, 2017.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning, pages 1310-1318, 2013.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211-252, 2015. doi: 10.1007/s11263-015-0816-y.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.  
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142-150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/P11-1015.  
Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large-scale machine learning. In OSDI, volume 16, pages 265-283, 2016.  
Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, page 3, 2013.  
Prajit Ramachandran, Barret Zoph, and Quoc Le. Searching for activation functions. 2017.  
Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. Understanding the exploding gradient problem. CoRR, abs/1211.5063, 2012.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.  
Barak Pearlmutter. An investigation of the gradient descent process in neural networks. PhD thesis, Carnegie Mellon University Pittsburgh, PA, 1996.  
Samuel S Schoenholz, Justin Gilmer, Surya Ganguli, and Jascha Sohl-Dickstein. Deep information propagation. arXiv preprint arXiv:1611.01232, 2016.

A META-LEARNING COMPARISONS TABLE  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Inner loop updates</td><td colspan="3">Outer loop updates, meta-</td><td rowspan="2">Generalizes to</td></tr><tr><td>parameters</td><td>objective</td><td>optimizer</td></tr><tr><td>Hyper parameter optimization 
Jones (2001); Snoek et al. (2012); 
Bergstra et al. (2011); Bergstra 
and Bengio (2012)</td><td>many steps of optimization</td><td>optimization hyper-parameters</td><td>training or validation set loss</td><td>Bayesian methods, random search, etc</td><td>nothing, or test set within fixed dataset</td></tr><tr><td>Neural architecture search Stanley 
and Miikkulainen (2002); Zoph 
and Le (2017); Baker et al. (2017); 
Zoph et al.; Real et al. (2017)</td><td>supervised SGD training using meta-learned architecture</td><td>architecture</td><td>validation set loss</td><td>RL or evolution</td><td>test loss within similar datasets</td></tr><tr><td>Task-specific optimizer (eg for quadratic function identification) 
(Hochreiter et al., 2001)</td><td>adjustment of model weights by an LSTM</td><td>LSTM weights</td><td>task loss</td><td>SGD</td><td>similar domain tasks</td></tr><tr><td>Learned optimizers Jones (2001); 
Maclaurin et al. (2015); 
Andrychowicz et al. (2016); Chen 
et al. (2016); Li and Malik (2017); 
Wichrowska et al. (2017); Bello 
et al. (2017)</td><td>many steps of optimization of a fixed loss function</td><td>parametric optimizer</td><td>average or final loss</td><td>SGD or RL</td><td>new loss functions (mixed success)</td></tr><tr><td>Prototypical networks Snell et al. 
(2017)</td><td>apply a feature extractor to a batch of data and use soft nearest neighbors to compute class probabilities</td><td>weights of the feature extractor</td><td>few shot performance</td><td>SGD</td><td>new image classes within similar dataset</td></tr><tr><td>MAML Finn et al. (2017)</td><td>one step of SGD on training loss starting from a meta-learned network</td><td>initial weights of neural network</td><td>reward or training loss</td><td>SGD</td><td>new goals, similar task regimes with same input domain</td></tr><tr><td>Evolved Policy Gradient 
Houthooft et al. (2018)</td><td>performing gradient descent on a learned loss</td><td>parameters of a learned loss function</td><td>reward</td><td>Evolutionary Strategies</td><td>new environment configurations, both in and not in meta-training distribution.</td></tr><tr><td>Few shot learning (Vinyals et al., 
2016; Ravi and Larochelle, 2016; 
Mishra et al., 2017)</td><td>application of a recurrent model, e.g. LSTM, Wavenet.</td><td>recurrent model weights</td><td>test loss on training tasks</td><td>SGD</td><td>new image classes within similar dataset.</td></tr><tr><td>Meta-unsupervised learning for clustering Garg and Kalai (2017)</td><td>run clustering algorithm or evaluate binary similarity function</td><td>clustering algorithm + hyperparameters, binary similarity function</td><td>empirical risk minimization</td><td>varied</td><td>new clustering or similarity measurement tasks</td></tr><tr><td>Learning synaptic learning rules 
(Bengio et al., 1990; 1992)</td><td>run a synapse-local learning rule</td><td>parametric learning rule</td><td>supervised loss, or similarity to biologically-motivated network</td><td>gradient descent, simulated annealing, genetic algorithms</td><td>similar domain tasks</td></tr><tr><td>Our work — metalearning for unsupervised representation 
learning</td><td>many applications of an unsupervised update rule</td><td>parametric update rule</td><td>few shot classification after unsupervised pre-training</td><td>SGD</td><td>new base models (width, depth, nonlinearity), new datasets, new data modalities</td></tr></table>

Table 1: A comparison of published meta-learning approaches.
