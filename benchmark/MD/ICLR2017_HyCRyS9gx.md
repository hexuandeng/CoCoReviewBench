# FAST ADAPTATION IN GENERATIVE MODELS WITH GENERATIVE MATCHING NETWORKS

Sergey Bartunov† & Dmitry P. Vetrov‡

National Research University Higher School of Economics  $(\mathrm{HSE})^{\dagger \ddagger}$

Moscow, Russia

Yandex‡

Moscow, Russia

# ABSTRACT

Despite recent advances, the remaining bottlenecks in deep generative models are necessity of extensive training and difficulties with generalization from small number of training examples. Both problems may be addressed by conditional generative models that are trained to adapt the generative distribution to additional input data. So far this idea was explored only under certain limitations such as restricting the input data to be a single object or multiple objects representing the same concept. In this work we develop a new class of deep generative model called generative matching networks which is inspired by the recently proposed matching networks for one-shot learning in discriminative tasks and the ideas from meta-learning. By conditioning on the additional input dataset, generative matching networks may instantly learn new concepts that were not available during the training but conform to a similar generative process, without explicit limitations on the number of additional input objects or the number of concepts they represent. Our experiments on the Omniglot dataset demonstrate that generative matching networks can significantly improve predictive performance on the fly as more additional data is available to the model and also adapt the latent space which is beneficial in the context of feature extraction.

# 1 INTRODUCTION

Deep generative models are currently one of the most promising directions in generative modelling. In this class of models the generative process is defined by a composition of conditional distributions modelled using deep neural networks which form a hierarchy of latent and observed variables. This approach allows to build models with complex, non-linear dependencies between variables and efficiently learn the variability across training examples.

Such models are trained by stochastic gradient methods which can handle large datasets and a wide variety of model architectures but also present certain limitations. The training process usually consists of small, incremental updates of networks' parameters and requires many passes over training data. Notably, once a model is trained it cannot be adapted to newly available data without complete re-training to avoid catastrophic interference (McCloskey & Cohen, 1989; Ratcliff, 1990). There is also a risk of overfitting for concepts that are not represented by enough training examples which is caused by high capacity of the models. Hence, most of deep generative models are not well-suited for rapid learning in one-shot scenario which is often encountered in real-world applications where data acquisition is expensive or fast adaptation to new data is required.

A potential solution to these problems is explicit learning of adaptation mechanisms complementing the shared generative process. In probabilistic modelling framework, adaptation may be expressed as conditioning the model on additional input examples serving as induction bias. Notable steps in this direction have been made by Rezende et al. (2016) whose model was able to condition on a single object to produce new examples of the concept it represents. Later, Edwards & Storkey (2016) proposed a model that maintained a global latent variable capturing statistics about multiple input objects which was used to condition the generative distribution. It allowed to implement the

fast learning ability, but due to the particular model architecture used the model was not well-suited to datasets consisting of several different concepts.

In this work we present Generative Matching Networks, a new family of conditional generative models capable of instant adaptation to new concepts that were not available at the training time but share the structure of underlying generative process with the training examples. By conditioning on additional inputs, Generative Matching Networks improve their predictive performance, the quality of generated samples and also adapt their latent space which may be useful for unsupervised feature extraction. Importantly, no explicit limitations on the conditioning data are imposed such as number of objects or number of different concepts which expands the applicability of one-shot generative modelling and distinguish our work from existing approaches. Our model is inspired by the attentional mechanism implemented in Matching Networks (Vinyals et al., 2016) previously proposed for discriminative tasks and the recent advances from meta-learning (Santoro et al., 2016). Our approach for adaptation is an extension of these ideas to generative modelling and it may be re-used in a variety of different models being not restricted to the particular architecture used in the paper. The source code for generative matching networks is available at http://github.com/sbos/gmn.

This paper is organized as follows. First, in section 2 we revisit the necessary background in variational approach to training generative models and mention the related work in conditional generative models. Then, in section 3 we describe the proposed generative model, it's recognition counterpart and the training protocol. Section 4 contains experimental evaluation of the proposed model as both generative model and unsupervised feature extractor in small-shot learning settings. We conclude with discussion of the results in section 5.

# 2 BACKGROUND

We consider the problem of learning a probabilistic generative model which can be expressed as a probability distribution  $p(\mathbf{x}|\boldsymbol{\theta})$  over objects of interests  $\mathbf{x}$  parametrized by  $\boldsymbol{\theta}$ . The major class of generative models introduce also latent variables  $\mathbf{z}$  that are used to explain or generate an object  $\mathbf{x}$  such that  $p(\mathbf{x}|\boldsymbol{\theta}) = \int p(\mathbf{z}|\boldsymbol{\theta})p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta})d\mathbf{z}$  and assumed to be non-observable.

Currently, the common practice is to restrict the conditional distributions  $p(\mathbf{z}|\boldsymbol{\theta})$  and  $p(\mathbf{x}|\mathbf{z},\boldsymbol{\theta})$  to tractable distribution families and use deep neural networks for regressing their parameters. The expressive power of deep non-linear generative models comes at a price since neither marginal distribution  $p(\mathbf{x}|\boldsymbol{\theta})$  can be computed analytically nor it can be directly optimized in a statistically efficient way. Fortunately, intractable maximum likelihood training can be avoided in practice by resorting to adversarial training (Gutmann & Hyvarinen, 2012; Goodfellow et al., 2014) or variational inference framework (Kingma & Welling, 2013; Rezende et al., 2014) which we consider further.

# 2.1 TRAINING GENERATIVE MODELS WITH VARIATIONAL INFERENCE

Recent developments in variational inference alleviate problems with maximizing the intractable marginal likelihood  $\log p(\mathbf{x}|\boldsymbol{\theta})$  by approximating it with a lower bound (Jordan et al., 1999):

$$
\log p (\mathbf {x} | \boldsymbol {\theta}) \geq \mathcal {L} (\boldsymbol {\theta}, \phi) = \mathbb {E} _ {q} \left[ \log p (\mathbf {x}, \mathbf {z} | \boldsymbol {\theta}) - \log q (\mathbf {z} | \mathbf {x}, \phi) \right] = \log p (\mathbf {x} | \boldsymbol {\theta}) - \operatorname {K L} (q | | p (\cdot | \mathbf {x}, \boldsymbol {\theta})). \tag {1}
$$

Tightness of the bound is controlled by the recognition model  $q(\mathbf{z}|\mathbf{x},\phi)$  which aims to minimize Kullback-Leibler divergence from the true posterior  $p(\mathbf{z}|\mathbf{x},\pmb{\theta})$ .

Similarly to the generative model, recognition model may also be implemented with the use of deep neural networks or other parameter regression which is known as amortized inference (Gershman & Goodman, 2014). Amortized inference allows to use a single recognition model for many training examples. Thus, it is convenient to perform training of the generative model  $p(\mathbf{x}|\pmb{\theta})$  by stochastic gradient optimization of variational lower bounds (1) corresponding to independent observations  $\{\mathbf{x}_i\}_{i=1}^N$ :

$$
\sum_ {i = 1} ^ {N} \log p (\mathbf {x} _ {i} | \theta) \geq \sum_ {i = 1} ^ {N} \mathbb {E} _ {q} \left[ \log p (\mathbf {x} _ {i}, \mathbf {z} _ {i} | \boldsymbol {\theta}) - \log q (\mathbf {z} _ {i} | \mathbf {x} _ {i}, \boldsymbol {\phi}) \right]\rightarrow \max  _ {\boldsymbol {\theta}, \boldsymbol {\phi}}.
$$

The clear advantage of this approach is its scalability. Every stochastic update to the parameters computed from only a small portion of training examples has an immediate effect for the whole

dataset. However, while a single parameter update may be relatively fast a large number of them is required to significantly improve generative or inferential performance of the model.

Hence, gradient training of generative models usually results into an extensive computational process which prevents from rapid incremental learning. In the next section we discuss potential solutions to this problem that allow to implement fast learning ability in generative models.

# 2.2 ADAPTATION IN GENERATIVE MODELS

In probabilistic modelling framework the natural way of incorporating knowledge about newly available data is conditioning. One may design a model that being conditioned on the additional input data  $\mathbf{X} = \mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$  represents a new generative distribution  $p(\mathbf{x}|\mathbf{X},\boldsymbol{\theta})$ .

An implementation of this idea can be found in the model by Rezende et al. (2016). Besides many other attractive novelties such as using sophisticated attention and feedback components, the model was able to produce new examples of a concept that was missing at the training time but had similarities in the underlying generative process with the other training examples. The model supported an explicit conditioning on a single observation  $\mathbf{x}^{\prime}$  representing the new concept to construct a new generative distribution of the form  $p(\mathbf{x}|\mathbf{x}',\boldsymbol {\theta})$

The explicit conditioning though is not the only way to propagate knowledge about new data. Another solution which is often encountered in Bayesian models is to maintain a global latent variable  $\alpha$  encoding information about the whole available dataset such that the individual observations are conditionally independent given it's value. The model then would have the following form:

$$
p (\mathbf {X} | \boldsymbol {\theta}) = \int p (\boldsymbol {\alpha} | \boldsymbol {\theta}) \prod_ {t = 1} ^ {T} p \left(\mathbf {x} _ {t} \mid \boldsymbol {\alpha}, \boldsymbol {\theta}\right) d \boldsymbol {\alpha}. \tag {2}
$$

The principal existence of such a global variable may be justified by the de Finetti's theorem (Diaconis & Freedman, 1980) under the exchangeability assumption. In the model (2), the conditional generative distribution  $p(\mathbf{x}|\mathbf{X},\pmb {\theta})$  is then defined implicitly via posterior over the global variable:

$$
p (\mathbf {x} | \mathbf {X}, \boldsymbol {\theta}) = \int p (\mathbf {x} | \boldsymbol {\alpha}, \boldsymbol {\theta}) p (\boldsymbol {\alpha} | \mathbf {X}, \boldsymbol {\theta}) d \boldsymbol {\alpha}. \tag {3}
$$

Once there is an efficient inference procedure for the global variable  $\alpha$ , fast adaptation of the generative model can be implemented straightforwardly.

There are several relevant examples of generative models with global latent variables used for model adaptation and one-shot learning. Salakhutdinov et al. (2013) combined deep Boltzmann machine (DBM) with nested Dirichlet process (nDP) in a Hierarchical-Deep (HD) model. While DBM was used to learn low-level features, the nonparametric distribution over high-level features defined via nDP allowed to infer a latent global hierarchy of concepts from the training data. The compound HD model was able to re-use the structure of generative process encoded in the latent hierarchy and low-level features to learn new concepts from small number of examples. However, being a compelling demonstration of important ideas from Bayesian nonparametrics and deep learning, the HD model required an extensive Markov chain Monte Carlo inference procedure used both for training and adaptation. Thus, while Bayesian learning approach could prevent overfitting the fast learning ability still presents a challenge for sampling-based inference.

Later, Lake et al. (2015) proposed Bayesian program learning (BPL) approach for building a generative model of handwritten characters. The model was defined as a probabilistic program contained fine-grained specification of prior knowledge of the task such as generation of strokes and their composition into characters mimicking human drawing strategies. Authors used an extensive posterior inference as the training procedure and the conditioning mechanism (3) for generating new examples. The model was shown to efficiently learn from small number of training examples, but similarly to the HD model, sophisticated and computationally expensive inference procedure makes fast adaptation in BPL generally hard to achieve.

The recently proposed neural statistician model (Edwards & Storkey, 2016) is an example of deep generative model with a global latent variable (2). The model was trained by optimizing a variational lower bound following the approach described in section 2.1 but with an additional recognition model approximating posterior distribution over the global latent variable. Authors designed the

recognition model to be computationally efficient and require only a single pass over data which consisted of extracting special features from the examples, applying to them a pooling operation (e.g. averaging) and passing the result to another network providing parameters of the variational approximation. This simple architecture allowed for the fast learning and guaranteed invariance to both data permutations and size of the conditioning dataset. However, experimentally the fast learning ability in the model was evaluated only in the setting where all of the training examples represented the same single concept.

We argue that in order to capture more information about the conditioning data such as a number of different concepts a more sophisticated aggregation procedure must be employed. Moreover, a fixed parametric description is too restrictive for an accurate representation of datasets of varying size. This motivates us to combine the best of two worlds: nonparametric representation of data and fast inference with neural recognition models. We proceed with a description of the proposed model.

# 3 GENERATIVE MATCHING NETWORKS

Generative matching networks aim to model conditional generative distributions of the form  $p(\mathbf{x}|\mathbf{X},\boldsymbol {\theta})$ . Similarly to other deep generative models we introduce a local latent variable  $\mathbf{z}$ . Thus the full joint distribution of our model can be expressed as:

$$
p (\mathbf {x}, \mathbf {z} | \mathbf {X}, \boldsymbol {\theta}) = p (\mathbf {z} | \mathbf {X}, \boldsymbol {\theta}) p (\mathbf {x} | \mathbf {z}, \mathbf {X}, \boldsymbol {\theta}). \tag {4}
$$

We also maintain a recognition model approximating the posterior over the latent variable  $\mathbf{z}$ :  $q(\mathbf{z}|\mathbf{x},\mathbf{X},\boldsymbol {\phi})\approx p(\mathbf{z}|\mathbf{x},\mathbf{X},\boldsymbol {\theta})$

In order to design a fast adaptation mechanism we have to make certain assumptions about relationships between training data and the new data used to condition the model. Thus we assume the homogeneity of generative processes for training and conditioning data up to some parametrization. One may think of this parametrization as specifying weights of a neural network defining a generative model. The generative process is assumed to have an approximately linear dependence on the parameters such that interpolation between parameters corresponding to different examples of the same concept can serve as good parameters for generating other examples. A similar assumption is used e.g. in the neural statistician model (Edwards & Storkey, 2016).

However, even if a single concept can be well embedded to a fixed parameter space, this does not imply that a diverse set of concepts will fit into the same parametrization. Hence we express the dependency on the conditioning data in a different way. Instead of embedding the whole conditioning dataset we use a special matching procedure that extracts relevant observations from  $\mathbf{X}$  and interpolates between their descriptions allowing to generate and recognize similar observations.

# 3.1 MATCHING CONDITIONING DATA

Since the dependency on  $\mathbf{X}$  is presented in different parts of our model we first describe the generic procedure of employing this dependency.

We denote by  $g(\cdot)$  a function that maps an observation  $\mathbf{x}' \in \mathbf{X}$  to a matching space  $\Phi$  which is used for comparing observations. Another function  $\psi(\cdot)$  maps an observation  $\mathbf{x}'$  to a prototype space  $\Psi$  which is generally different from the matching space  $\Phi$  and is devoted for summarizing output information about observations that could be useful, for example, for generating new observations. We also denote by  $f(\cdot)$  a function that is used for mapping what we call a query to the matching space  $\Phi$ .

Query contains the information to filter relevant objects from  $\mathbf{X}$  by matching with them in the space  $\Phi$ . That information could be another observation or a value of latent variable as we will see soon. In the simplest form, the information from matched observations is aggregated by taking a weighted average of the features  $\psi$  extracted from observations. Weights are computed using the soft attentional mechanism relying e.g. on cosine vector similarity:

$$
\mathbf {r} = \sum_ {t = 1} ^ {T} a (\mathbf {q}, \mathbf {x} _ {t}) \psi (\mathbf {x} _ {t}), \quad a (\mathbf {q}, \mathbf {x} _ {t}) = \frac {\exp \left(\cos \left(f (\mathbf {q}) , g \left(\mathbf {x} _ {t}\right)\right)\right)}{\sum_ {t ^ {\prime} = 1} ^ {T} \exp \left(\cos \left(f (\mathbf {q}) , g \left(\mathbf {x} _ {t ^ {\prime}}\right)\right)\right)}. \tag {5}
$$

![](images/7edf6b2b80ffba0970a9fbc46a6bc2e66062a8067a69c9e101bb54a1fa4cf9fe.jpg)  
Figure 1: Structure of a simple generative matching network, see equation (5) in section 3.1 for the description of functions  $f$ ,  $g$  and  $\psi$ .

This matching procedure can be used, for example, to implement fast learning in the conditional likelihood  $p(\mathbf{x}|\mathbf{z},\mathbf{X},\boldsymbol {\theta})$ . In this case query would be simply the value of latent variable  $\mathbf{q} = \mathbf{z}$  used to select a few observations from the conditioning data  $\mathbf{X}$  that could be roughly reconstructed from  $\mathbf{z}$  or in other words that match  $\mathbf{z}$  in the feature space  $\Phi$ . These matched observations are then treated as prototypes for the new observation corresponding to  $\mathbf{z}$ . Finally, their features are distilled into the vector  $\mathbf{r}$  parametrizing the generative process.

The disadvantage of the simple matching procedure (5) is that conditioning observations  $\mathbf{X}$  are embedded independently from each other. Similarly to discriminative matching networks we address this problem by computing full contextual embeddings (FCE) (Vinyals et al., 2015). In order to obtain dependent embeddings of conditioning data we allow  $K$  attentional passes over  $\mathbf{X}$  of the form (5) guided by a recurrent controller  $R$  which accumulates global knowledge about the conditioning data in its hidden state  $\mathbf{h}$ . The hidden state is thus passed to feature extractors  $f$  and  $g$  to obtain context-dependent embeddings:

$$
\mathbf {r} _ {k} = \sum_ {t = 1} ^ {T} a (\mathbf {q}, \mathbf {x} _ {t}) \psi (\mathbf {x} _ {t}), \quad a (\mathbf {q}, \mathbf {x} _ {t}) = \frac {\exp \left(\cos \left(f (\mathbf {q} , \mathbf {h} _ {k}) , g \left(\mathbf {x} _ {t} , \mathbf {h} _ {k}\right)\right)\right)}{\sum_ {t ^ {\prime} = 1} ^ {T} \exp \left(\cos \left(f (\mathbf {q} , \mathbf {h} _ {k}) , g \left(\mathbf {x} _ {t ^ {\prime}} , \mathbf {h} _ {k}\right)\right)\right)}, \tag {6}
$$

$$
\mathbf {h} _ {k + 1} = R \left(\mathbf {h} _ {k}, \mathbf {r} _ {k}\right).
$$

The result of the procedure is the last hidden state  $\mathbf{h}_{K + 1}$  and aggregated prototype vector  $\mathbf{r}_K$  which can be used further to represent matching of the query with the conditioning data. We also found it beneficial to allow the recurrent controller set a temperature for the softmax operation (6) used to compute attention weights, but omit this from the equation for clarity.

Henceforth we will use a subscript to refer to a particular instance of the described functions:  $P$  for the prior  $p(\mathbf{z}|\boldsymbol{\theta})$ ,  $L$  for the conditional likelihood  $p(\mathbf{x}|\mathbf{z}, \mathbf{X}, \boldsymbol{\theta})$  and  $R$  for the recognition model.

# 3.2 CONDITIONAL MODEL

The described matching process is used to specify the dependency on conditioning data. The simplest case with the independent prior  $p(\mathbf{z}|\mathbf{X},\boldsymbol {\theta}) = p(\mathbf{z}|\boldsymbol {\theta})$  and independent matching procedure (5) is shown on figure 1. However, we found that both data-dependent prior and full context matching (6) significantly improves performance of the model so henceforth we consider only models with these improvements. Another detail is that since in data-conditional prior  $p(\mathbf{z}|\mathbf{X},\boldsymbol {\theta})$  there is no query object to match with conditioning data  $\mathbf{X}$ , only state of the recurrent controller was used for matching.

In order to keep the number of parameters small, the same function  $\psi = \psi_{P} = \psi_{L} = \psi_{R}$  was used in all parts of the model. The likelihood and recognition model also used a shared recurrent controller, i.e.  $R_{L} = R_{R}$  implemented as gated recurrent unit (GRU) (Chung et al., 2015) while the prior had a separate GRU. Functions  $f$  and  $g$  were different across the model but query observations were represented using the shared feature extractor  $\psi$ . We emphasize that such functionality sharing is not obligatory and some models may benefit from separate controllers and/or feature extractors.

We used diagonal Gaussian distribution to parametrize the latent variable  $\mathbf{z}$  in the prior  $p(\mathbf{z}|\mathbf{X},\boldsymbol {\theta})$  and recognition model  $q(\mathbf{z}|\mathbf{x},\mathbf{X},\phi)$ . No auto-regressive connections or multiple stochastic layers were used.

To use the same conditional model when no additional inputs is available, we add a pseudo-input  $\tilde{\mathbf{x}}$  to both simple (5) and full contextual matching procedures (6) with corresponding pseudo-features as trainable parameters. More details about the architecture can be found in the appendix.

# 3.3 TRAINING

Training of our model consists of maximizing marginal likelihood of a dataset  $\mathbf{X}$  which can be expressed as:

$$
p (\mathbf {X} | \boldsymbol {\theta}) = \prod_ {t = 1} ^ {T} p \left(\mathbf {x} _ {t} \mid \mathbf {X} _ {<   t}, \boldsymbol {\theta}\right), \quad \mathbf {X} _ {<   t} = \left\{\mathbf {x} _ {s} \right\} _ {s = 1} ^ {t - 1}. \tag {7}
$$

Ideally we would like to use the whole available training data as  $\mathbf{X}$  but due to computational restrictions we instead use a training strategy rooted in curriculum learning (Bengio et al., 2009) and meta-learning (Thrun, 1998; Vilalta & Drissi, 2002; Hochreiter et al., 2001) which recently was successfully applied for one-shot discriminative learning (Santoro et al., 2016). In particular, we define a task-generating distribution  $p_d(\mathbf{X})$  which in our case samples datasets  $\mathbf{X}$  of size  $T$  from training examples. Then we train our model to explain well all of the sampled datasets simultaneously:

$$
\mathbb {E} _ {p _ {d} (\mathbf {X})} [ p (\mathbf {X} | \boldsymbol {\theta}) ] \rightarrow \max  _ {\boldsymbol {\theta}}. \tag {8}
$$

Obviously, the structure of task-generating distribution has a large impact on training and using an arbitrary distribution will unlikely lead to good results. Hence, we assume that at the training time we have an access to label information and can assign class labels to the training examples distinguish different concepts or classes. We thus constrain  $p_d(\mathbf{X})$  to generate datasets consisting of examples that represent up to  $C$  randomly selected classes so that even on short datasets the model has a clear incentive to re-use conditioning data. This may be considered as a form of weak supervision so we want to emphasize that one does not need the label information at test time unless she wants to explicitly use the model for classification which is also possible.

Since the marginal likelihood (7) as well as the conditional marginal likelihoods are intractable we instead use variational lower bound (see section 2.1) as a proxy to  $p(\mathbf{X}|\boldsymbol{\theta})$  in the objective (8):

$$
\mathcal {L} (\mathbf {X}, \pmb {\theta}, \phi) = \sum_ {t = 1} ^ {T} \mathbb {E} _ {q (\mathbf {z} _ {t} | \mathbf {x} _ {t}, \mathbf {X} _ {<   t}, \phi)} \left[ \log p (\mathbf {x} _ {t}, \mathbf {z} _ {t} | \mathbf {X} _ {<   t}, \pmb {\theta}) - \log q (\mathbf {z} _ {t} | \mathbf {x} _ {t}, \mathbf {X} _ {<   t}, \phi) \right].
$$

# 4 EXPERIMENTS

For our experiments we use the Omniglot dataset (Lake et al., 2015) which consists of 1623 classes of handwritten characters from 50 different alphabets. The first 30 alphabets are devoted for training and the remaining 20 alphabets are left for testing. Importantly, only 20 examples of each class are available which makes this dataset specifically useful for small-shot learning problems. Unfortunately, the literature is inconsistent in usage of the dataset and multiple versions of Omniglot were used for evaluation which differ by train/test split, resolution, binarization and augmentation, see e.g. (Burda et al., 2015; Rezende et al., 2016; Santoro et al., 2016).

We use the canonical split provided by Lake et al. (2015). In order to speed-up training we down-scaled images to  $28 \times 28$  resolution and since the result was fully contrastive we did not apply any further binarization. We also did not augment our data as in (Santoro et al., 2016; Edwards & Storkey, 2016) to make future comparisons with our results easier.

Unless otherwise stated, we train models on datasets of length  $T = 10$  and of up to  $C = 2$  different classes as we did not observe any improvement from training with larger values of  $C$ . The dimensionality of latent variable  $\mathbf{z}$  was set to 50 and the hidden state of all controllers had dimensionality of 200. Binary observations were modelled with Bernoulli distributions.

# 4.1 NUMBER OF ATTENTION STEPS

Since the full context matching procedure (6) described in section 3.1 consists of multiple attention steps, it is interesting to see the effect of these numbers on model's performance. We trained several models varying number of attention steps allowed for the likelihood and recognition shared controller and the prior controller respectively. The models were compared using exponential moving averages of lower bounds corresponding to different numbers of conditioning examples  $\mathbf{X}_{< t}$  obtained during the training. Results of the comparison can be found on figure 2.

![](images/9c5bcbe5386b792bdb8a06d4041c3e00a8c26752e282ca6ee255828c0ecf608d.jpg)  
Figure 2: Lower bound estimates (left) and entropy of prior (right) for various numbers of attention steps and numbers of conditioning examples. Numbers are reported for the training part of Omniglot.

![](images/720854e48cfe7d3c83218097da3529951ab9761879ff300da0853dbbf8533467.jpg)

Table 1: Conditional likelihoods for the test part of Omniglot.  

<table><tr><td>Model</td><td>Ctest</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>9</td></tr><tr><td>GMN, Ctrain = 1</td><td>1</td><td>-90.9</td><td>-82.8</td><td>-78.8</td><td>-76.4</td><td>-74.2</td><td>-64.6</td></tr><tr><td>GMN, Ctrain = 2</td><td>1</td><td>-89.3</td><td>-83.4</td><td>-79.6</td><td>-77.2</td><td>-75.3</td><td>-66.7</td></tr><tr><td>GMN, Ctrain = 2</td><td>2</td><td>-89.2</td><td>-86.6</td><td>-85.0</td><td>-83.5</td><td>-81.6</td><td>-75.9</td></tr><tr><td>GMN, Ctrain = 2</td><td>2</td><td>-</td><td>-574.6</td><td>-357.3</td><td>-263.5</td><td>-217.6</td><td>-138.0</td></tr><tr><td>pseudo-input removed</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GMN, Ctrain = 2</td><td>3</td><td>-88.9</td><td>-87.2</td><td>-86.7</td><td>-85.1</td><td>-84.6</td><td>-79.9</td></tr><tr><td>GMN, Ctrain = 2</td><td>4</td><td>-88.8</td><td>-88.0</td><td>-87.9</td><td>-86.6</td><td>-85.6</td><td>-82.5</td></tr><tr><td>VAE</td><td></td><td>-89.2</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GMN, Ctrain = 1, average</td><td>1</td><td>-91.3</td><td>-83.9</td><td>-82.3</td><td>-81.8</td><td>-81.1</td><td>-80.3</td></tr><tr><td>GMN, Ctrain = 2, average</td><td>2</td><td>-89.1</td><td>-88.0</td><td>-87.6</td><td>-87.3</td><td>-86.8</td><td>-85.8</td></tr></table>

Interestingly, larger numbers of steps lead to better results, however lower bounds are almost not improving after the shared controller is allowed for 4 steps. This behaviour was not observed with discriminative matching networks perhaps confirming the difficulty of unsupervised learning. Another important result is that the standard Gaussian prior makes adaptation significantly harder for the model yet still possible which justifies the importance of adaptation not just for the likelihood model but also for the prior.

One may also see that all models preferred to set higher variances for a prior resulting to higher entropy comparing to standard normal prior. Clearly as more examples are available, generative matching networks become more certain about the data and output less dispersed Gaussians.

Based on this comparison we decided to proceed with models that have 4 steps for the shared controller and a single step for the prior controller which is a reasonable compromise between computational cost and performance.

# 4.2 FAST ADAPTATION AND SMALL-SHOT GENERATION

In this section we compare generative matching networks with a set of baselines by expected conditional likelihoods  $\mathbb{E}_{p_d(\mathbf{X})}p(\mathbf{x}_t|\mathbf{X}_{< t})$ . The conditional likelihoods for our model were estimated using importance sampling with 1000 samples from the recognition model used as a proposal.

We found it hard to properly compute conditional likelihoods for the neural statistician model (3) and hence had to exclude this model from the comparison, please see appendix for the details. Instead, we consider a simple generative matching network in which the matching procedure is replaced with an average operation which makes the adaptation mechanism similar to the one used in neural statistician. We also omitted sequential generative models (Rezende et al., 2016) from the comparison as they were reported to overfit on the canonical train/test split of Omniglot. Another baseline we consider is a standard variational autoencoder which has the same architecture for generative and recognition model as the full conditional generative matching networks.

Table 1 contains results of the evaluation on the test alphabets from Omniglot.  $C_{\mathrm{train}}$  and  $C_{\mathrm{test}}$  denote the maximum number of classes in task-generating distributions  $p_d(\cdot)$  used for training and evaluating respectively. As one could expect, larger values of  $C_{\mathrm{test}}$  make adaptation harder since in average less examples of the same class are available to the model. Still generative matching

![](images/0a927b33a4e021dd61f2bb6ee79a1dcfa30a98cef92e80de568b5c91aef83bf3.jpg)  
(a) Full matching

![](images/681fd7c0a8f11686619eb83c1d7e30ef2612dac0f93dae752d1cb2123507cc0d.jpg)  
(b) Full matching, no pseudo-input

![](images/e54c596c636999662721da00da5089ffd1da7b131efa734fa4ad918b1d4e4c69.jpg)  
(c) Average matching  
Figure 3: Conditionally generated samples. First column contains conditioning data in the order it is revealed to the model. Row number  $t$  (counting from zero) consists of samples conditioned on first  $t$  input examples.

networks are capable of working in low-data regime even when testing setting is harder than one used for training, i.e.  $C_{\mathrm{test}} > C_{\mathrm{train}}$ . Unsurprisingly, adaptation by averaging over prototype features performed reasonably well for simple datasets constructed of a single class, although significantly worse than the proposed matching procedure. On more difficult datasets with mixed examples of two different classes ( $C_{\mathrm{test}} = 2$ ) averaging was ineffective for expressing dependency on the conditioning data which justifies our argument on the necessity of nonparametric representations.

In order to visually assess the fast adaptation ability of generative matching networks we also provide conditionally generated samples on figure 3. Interestingly, removing pseudo-input (see section 3.1) at the test time significantly improves visual quality of samples although seriously harming the conditional likelihood as shown in table 1. Such unfortunate discrepancy between visual quality of samples and predictive performance was well studied in (Theis et al., 2015) and may suggest that removing the pseudo-input causes "overfitting" on the conditioning data. Rezende et al. (2016) faced a similar effect with overfitted models producing samples of quality indistinguishable from non-overfitted models. Therefore, removal of the pseudo-input should depend on target application of the model, i.e. density estimation or producing new examples.

# 5 CONCLUSION

In this paper we presented a new class of conditional deep generative models called generative matching networks. These models are capable of fast adaptation to conditioning dataset by adjusting both the latent space and the predictive density while making very few assumptions on the data. The nonparametric matching enabling these features can be seen as a generalization of the original matching procedure since it allows a model to define the label space itself extending the applicability of matching networks to unsupervised and perhaps semi-supervised settings. We believe that these ideas can evolve further and help to implement more data-efficient models in other domains such as reinforcement learning where data acquisition is especially hard.

# ACKNOWLEDGMENTS

We would like to thank Michael Figurnov and Timothy Lillicrap for useful discussions. Dmitry P. Vetrov is supported by RFBR project No.15-31-20596 (mol-a-ved) and by Microsoft: MSU joint research center (RPD 1053945).

# REFERENCES

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pp. 41-48. ACM, 2009.  
Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Junyoung Chung, Calar Güçehre, Kyunghyun Cho, and Yoshua Bengio. Gated feedback recurrent neural networks. In Proceedings of the 32nd International Conference on Machine Learning (ICML'15), 2015.  
Persi Diaconis and David Freedman. Finite exchangeable sequences. The Annals of Probability, pp. 745-764, 1980.  
Harrison Edwards and Amos Storkey. Towards a neural statistician. arXiv preprint arXiv:1606.02185, 2016.  
Samuel J Gershman and Noah D Goodman. Amortized inference in probabilistic reasoning. In Proceedings of the 36th Annual Conference of the Cognitive Science Society, 2014.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Michael U Gutmann and Aapo Hyvarinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. Journal of Machine Learning Research, 13 (Feb):307-361, 2012.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1026-1034, 2015.  
Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to learn using gradient descent. In International Conference on Artificial Neural Networks, pp. 87-94. Springer, 2001.  
Michael I Jordan, Zoubin Ghahramani, Tommi S Jaakkola, and Lawrence K Saul. An introduction to variational methods for graphical models. Machine learning, 37(2):183-233, 1999.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332-1338, 2015.  
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of learning and motivation, 24:109-165, 1989.  
Roger Ratcliff. Connectionist models of recognition memory: constraints imposed by learning and forgetting functions. Psychological review, 97(2):285, 1990.  
Danilo J Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the 31st International Conference on Machine Learning (ICML-14), pp. 1278-1286, 2014.  
Danilo Jimenez Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, and Daan Wierstra. One-shot generalization in deep generative models. arXiv preprint arXiv:1603.05106, 2016.  
Ruslan Salakhutdinov, Joshua B Tenenbaum, and Antonio Torralba. Learning with hierarchical-deep models. IEEE transactions on pattern analysis and machine intelligence, 35(8):1958-1971, 2013.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. One-shot learning with memory-augmented neural networks. arXiv preprint arXiv:1605.06065, 2016.

Lucas Theis, Aäron van den Oord, and Matthias Bethge. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844, 2015.  
Sebastian Thrun. Lifelong learning algorithms. In Learning to learn, pp. 181-209. Springer, 1998.  
Ricardo Vilalta and Youssef Drissi. A perspective view and survey of meta-learning. Artificial Intelligence Review, 18(2):77-95, 2002.  
Oriol Vinyals, Samy Bengio, and Manjunath Kudlur. Order matters: Sequence to sequence for sets. arXiv preprint arXiv:1511.06391, 2015.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. arXiv preprint arXiv:1606.04080, 2016.
