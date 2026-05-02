# NEURAL VARIATIONAL INFERENCE FOR EMBEDDING KNOWLEDGE GRAPHS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent advances in Neural Variational Inference allowed for a renaissance in latent variable models in a variety of domains involving high-dimensional data. In this paper, we introduce two generic Variational Inference frameworks for generative models of Knowledge Graphs; Latent Fact Model and Latent Information Model. While traditional variational methods derive an analytical approximation for the intractable distribution over the latent variables, here we construct an inference network conditioned on the symbolic representation of entities and relation types in the Knowledge Graph, to provide the variational distributions. The new framework can create models able to discover underlying probabilistic semantics for the symbolic representation by utilising parameterisable distributions which permit training by back-propagation in the context of neural variational inference, resulting in a highly-scalable method. Under a Bernoulli sampling framework, we provide an alternative justification for commonly used techniques in large-scale stochastic variational inference, which drastically reduces training time at a cost of an additional approximation to the variational lower bound. The generative frameworks are flexible enough to allow training under any prior distribution that permits a reparametrisation trick, as well as under any scoring function that permits maximum likelihood estimation of the parameters. Experiment results display the potential and efficiency of this framework by improving upon multiple benchmarks with Gaussian prior representations. Code publicly available on Github additionally allows learning Hyperspherical representations under a von-Mises Fisher prior distribution.

# 1 INTRODUCTION

In many fields, including physics and biology, being able of representing uncertainty is of crucial importance (Ghahramani, 2015). For instance, link prediction in Knowledge Graphs is used for driving expensive pharmaceutical experiments (Bean et al., 2017). It would be beneficial to know what is the confidence of a model in its predictions. However, a significant shortcoming of current neural link prediction models – and for the vast majority of neural representation learning approaches – is their inability to express a notion of uncertainty.

In particular, neural link prediction models usually only return point estimates of parameters and predictions (Nickel et al., 2016), and are trained discriminatively rather than generatively: they aim at predicting one variable of interest conditioned on all the others, rather than accurately representing the relationships between different variables (Ng & Jordan, 2001). This is an important issue when applying representation learning models to Knowledge Graphs: such graphs often suffer from incompleteness and sparsity (Dong et al., 2014), and it is fundamental to know the uncertainty or variance associated with a prediction.

Furthermore, Knowledge Graphs can be very large and Web-scale (Dong et al., 2014). In a probabilistic model, we can leverage the variance in model parameters and predictions for finding which facts to sample during training, in an Active Learning setting (Kapoor et al., 2007; Gal et al., 2017).

# 2 BACKGROUND

In this work, we focus on models for predicting missing links in large, multi-relational networks such as FREEBASE. In the literature, this problem is referred to as link prediction. We specifically focus on knowledge graphs, i.e., graph-structured knowledge bases where factual information is stored in the form of relationships between entities. Link prediction in knowledge graphs is also known as knowledge base population. We refer to Nickel et al. (2016) for a recent survey on approaches to this problem.

A knowledge graph  $\mathcal{G} \triangleq \{(r, a_1, a_2)\} \subseteq \mathcal{R} \times \mathcal{E} \times \mathcal{E}$  can be formalised as a set of triples (facts) consisting of a relation type  $r \in \mathcal{R}$  and two entities  $a_1, a_2 \in \mathcal{E}$ , respectively referred to as the subject and the object of the triple. Each triple  $(r, a_1, a_2)$  encodes a relationship of type  $r$  between  $a_1$  and  $a_2$ , represented by the fact  $r(a_1, a_2)$ .

Link prediction in knowledge graphs is often simplified to a learning to rank problem, where the objective is to find a score or ranking function  $\phi_r^\Theta : \mathcal{E} \times \mathcal{E} \mapsto \mathbb{R}$  for a relation  $r$  that can be used for ranking triples according to the likelihood that the corresponding facts hold true.

# 2.1 NEURAL LINK PREDICTION

Recently, a specific class of link predictors received a growing interest (Nickel et al., 2016). These predictors can be understood as multi-layer neural networks. Given a triple  $\mathbf{x} = (s,r,o)$ , the associated score  $\phi_r^\Theta(s,o)$  is given by a neural network architecture encompassing an encoding layer and a scoring layer.

In the encoding layer, the subject and object entities  $s$  and  $o$  are mapped to low-dimensional vector representations (embeddings)  $\mathbf{h}_s \triangleq \mathbf{h}(s) \in \mathbb{R}^k$  and  $\mathbf{h}_o \triangleq \mathbf{h}(o) \in \mathbb{R}^k$ , produced by an encoder  $\mathbf{h}^\Gamma : \mathcal{E} \to \mathbb{R}^k$  with parameters  $\Gamma$ . This layer can be pre-trained (Vylomova et al., 2016) or, more commonly, learnt from data by back-propagating the link prediction error to the encoding layer (Bordes et al., 2013; Nickel et al., 2016; Trouillon et al., 2016a).

In the scoring layer, the entity representations  $\mathbf{h}_s$  and  $\mathbf{h}_o$  are scored by a function  $\phi_r^\Theta (\mathbf{h}_s,\mathbf{h}_o)$  parametrised by  $\Theta$ .

Summarising, the high-level architecture is defined as:

$$
\mathbf {h} _ {s}, \mathbf {h} _ {o} \triangleq \mathbf {h} ^ {\Gamma} (s), \mathbf {h} ^ {\Gamma} (o)
$$

$$
\phi_ {r} (s, o) \triangleq \phi_ {r} ^ {\Theta} (\mathbf {h} _ {s}, \mathbf {h} _ {o})
$$

Ideally, more likely triples should be associated with higher scores, while less likely triples should be associated with lower scores.

While the literature has produced a multitude of encoding and scoring strategies, for brevity we overview only a small subset of these. However, we point out that our method makes no further assumptions about the network architecture other than the existence of an argument encoding layer.

# 2.2 ENCODING LAYER

Given an entity  $e \in \mathcal{E}$ , the entity encoder  $\mathbf{h}^{\Gamma}$  is usually implemented as a simple embedding layer  $\mathbf{h}^{\Gamma}(e) \triangleq [\Gamma]_e$ , where  $\Gamma$  is an embedding matrix (Nickel et al., 2016). For pre-trained embeddings, the embedding matrix is fixed. Note that other encoding mechanisms are conceivable, such as; recurrent, graph convolution (Kipf & Welling, 2016a;b) or convolutional neural networks (Dettmers et al., 2017).

# 2.3 DECODING LAYER: SCORING FUNCTIONS

DistMult DISTMULT (Yang et al., 2015) represents each relation  $r$  using a parameter vector  $\Theta_r \in \mathbb{R}^k$ , and scores a link of type  $r$  between  $(\mathbf{h}_s, \mathbf{h}_o)$  using the following scoring function:

$$
\phi_ {r} ^ {\Theta} (\mathbf {h} _ {s}, \mathbf {h} _ {o}) \triangleq \langle \Theta_ {r}, \mathbf {h} _ {s}, \mathbf {h} _ {o} \rangle \triangleq \sum_ {i = 1} ^ {k} \Theta_ {r, i} \mathbf {h} _ {s, i} \mathbf {h} _ {o, i},
$$

where  $\langle \cdot ,\cdot ,\cdot \rangle$  denotes the tri-linear dot product.

ComplEx COMPLEX (Trouillon et al., 2016a) is an extension of DISTRULT using complex-valued embeddings while retaining the mathematical definition of the dot product. In this model, the scoring function is defined as follows:

$$
\phi_ {r} ^ {\Theta} \left(\mathbf {h} _ {s}, \mathbf {h} _ {o}\right) \triangleq \operatorname {R e} \left(\left\langle \Theta_ {r}, \mathbf {h} _ {s}, \overline {{\mathbf {h} _ {o}}} \right\rangle\right),
$$

where  $\Theta_r, \mathbf{h}_s, \mathbf{h}_o \in \mathbb{C}^k$  are complex vectors,  $\overline{\mathbf{x}}$  denotes the complex conjugate of  $\mathbf{x}$ , and  $\operatorname{Re}(\mathbf{x}) \in \mathbb{R}^k$  denotes the real part of  $\mathbf{x}$ .

# 3 RELATED WORK

Variational Deep Learning has seen great success in areas such as parametric/non-parametric document modelling Miao et al. (2017); Miao et al. (2016) and image generation (Kingma & Welling (2013)). Stochastic variational inference has been used to learn probability distributions over model weights (Blundell et al., 2015), which the authors named "Bayes By Backprop", as well as proven powerful enough to train deep belief networks (Vilnis & McCallum, 2014), by improving upon the stochastic variational bayes estimator (Kingma & Welling, 2013), using general variance reduction techniques.

Previous work has been done to re-frame word embeddings in a Bayesian framework (Zhang et al., 2014; Vilnis & McCallum, 2014), as well as re-frame graph embeddings in a Bayesian framework (He et al., 2015). However, these methods are expensive to train due to the evaluation of complex tensor inversions. Recent work by the authors of (Barkan, 2016; Bražinskas et al., 2017) show that it is possible to train word embeddings through a VB (Bishop, 2006) framework.

KG2E (He et al., 2015) proposed a probabilistic embedding method for modelling the uncertainties in KGs. However, this was not a generative model. The authors of (Xiao et al., 2016) argue they created the first generative model for knowledge graph embeddings. Firstly, this work is empirically worse than a few of the generative models built under our proposed framework. Secondly, their method is restricted to a Gaussian distribution prior, whereas we can use this, as well as any other prior that permits a re-parameterisation trick — such as the von-Mises distribution.

Later, the authors of (Kipf & Welling, 2016b) propose a generative model for graph embeddings. However, their method lacks scalability as it requires the use of the full adjacency tensor of the graph as input. Secondly, our work differs from (Kipf & Welling, 2016b) as they work with uni-relational data, whereas we create a framework for many variational generative models over multi-relational data.

Recent work by the authors of (Chen et al., 2018) led to successfully constructing a variational path ranking algorithm, a graph feature model. This work differs from ours for two reasons. Firstly, it does not produce a generative model for knowledge graph embeddings. Secondly, their work is a graph feature model, with the constraint of at most one relation per entity pair, whereas our model is a latent feature model with a theoretical unconstrained limit on the number of existing relationships between a given pair of entities.

# 4 GENERATIVE MODELS

Let  $\mathcal{D} \triangleq \{(\tau_1, y_1), \ldots, (\tau_n, y_n)\}$  denote a set of labelled triples, where  $\tau_i \triangleq \langle s_i, p_i, o_i \rangle$ , and  $y_i \in \{0, 1\}$  denotes the corresponding label, denoting that the fact encoded by the triple is either true or false. We can assume  $\mathcal{D}$  is generated by a corresponding generative model. In the following, we propose two alternative generative models.

# 4.1 LATENT FACT MODEL

In this model, we assume that the Knowledge Graph was generated according to the following generative model. Let  $\mathcal{V} \triangleq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$  the space of possible triples. where  $\tau \triangleq \langle s, p, o \rangle$ , and  $\mathbf{h}_{\tau} \triangleq [\mathbf{h}_s, \mathbf{h}_p, \mathbf{h}_o]$  denotes the sampled embedding representations of  $s, o \in \mathcal{E}$  and  $p \in \mathcal{R}$ .

Note that, in this model, the embeddings are sampled for each triple. As a consequence, the set of latent variables in this model is  $\mathcal{H} \triangleq \{\mathbf{h}_{\tau} \mid \tau \in \mathcal{E} \times \mathcal{R} \times \mathcal{E}\}$ .

![](images/cfea9ec6d5da45c3922e45a32e996cb5fbf26d21d1f783364a311dd3124b674e.jpg)  
Figure 1: Latent Fact Model (LFM)

![](images/51025784ae358c71dd5e4cb901de151db8e4c96523b2b8545b0988a221a5bd6a.jpg)  
Figure 2: Latent Information Model (LIM)

The joint probability of the variables  $p(\mathcal{H},\mathcal{D})$  is defined as follows:

$$
p (\mathcal {H}, \mathcal {D}) \triangleq \prod_ {(\tau , y _ {\tau}) \in \mathcal {D}} p \left(\mathbf {h} _ {\tau}\right) p \left(y _ {\tau} \mid \mathbf {h} _ {\tau}\right) \tag {1}
$$

The marginal distribution over  $\mathcal{D}$  is then defined as follows:

$$
p (\mathcal {D}) \geq \mathbb {E} _ {q} \left[ \log p \left(y _ {\tau} \mid \mathbf {h} _ {\tau}\right) \right] - D _ {\mathrm {K L}} q \left(\mathbf {h} _ {\tau}\right) p \left(\mathbf {h} _ {\tau}\right) \tag {2}
$$

As a consequence, the log-marginal likelihood of the data is bounded by:

$$
\log p (\mathcal {D}) \leq \sum_ {(\tau , y _ {\tau}) \in \mathcal {D}} \mathrm {E L B O} _ {\tau} \triangleq \mathrm {E L B O} \tag {3}
$$

# 4.1.1 OPTIMISING THE ELBO

Note that this is an enormous sum over  $|\mathcal{D}|$  elements. However, this can be approximated via Importance Sampling, or Bernoulli Sampling (Botev et al., 2017).

$$
\begin{array}{l} \mathrm {E L B O} = \sum_ {(\tau , y _ {\tau}) \in \mathcal {D}} \mathbb {E} _ {q} [ \log p (y _ {\tau} \mid \mathbf {h} _ {\tau}) ] - D _ {\mathrm {K L}} q (\mathbf {h} _ {\tau}) p (\mathbf {h} _ {\tau}) \\ = \sum_ {(\tau , y _ {\tau}) \in \mathcal {D} ^ {+}} \mathbb {E} _ {q} [ \log p (y _ {\tau} \mid \mathbf {h} _ {\tau}) ] - D _ {\mathrm {K L}} q (\mathbf {h} _ {\tau}) p (\mathbf {h} _ {\tau}) \tag {4} \\ + \sum_ {(\tau , y _ {\tau}) \in \mathcal {D} ^ {-}} \mathbb {E} _ {q} [ \log p (y _ {\tau} \mid \mathbf {h} _ {\tau}) ] - D _ {\mathrm {K L}} q (\mathbf {h} _ {\tau}) p (\mathbf {h} _ {\tau}) \\ \end{array}
$$

By using Bernoulli Sampling, ELBO can be approximated by:

$$
\mathrm {E L B O} \approx \sum_ {c: s _ {c} = 1} \frac {\mathrm {E L B O} _ {\tau_ {c}}}{b _ {c}} \tag {5}
$$

where  $p(s_{c} = 1) = b_{c}$  can be defined for each element  $c$ . We can define a probability distribution of sampling from  $\mathcal{D}^{+}$  and  $\mathcal{D}^{-}$  - similarly to Bayesian Personalised Ranking (Rendle et al., 2009), we sample one negative triple for each positive one - we use a constant probability for each element depending on whether it is in the positive or negative set. We end up with the following estimate:

$$
\mathrm {E L B O} \approx \sum_ {i = 1} ^ {n} \frac {\mathrm {E L B O} _ {\tau_ {c} ^ {+}}}{b _ {c} ^ {+}} + \frac {\mathrm {E L B O} _ {\tau_ {c} ^ {-}}}{b _ {c} ^ {-}} \tag {6}
$$

where  $b_{c}^{+} = |\mathcal{D}^{+}| / |\mathcal{D}|$  and  $b_{c}^{-} = |\mathcal{D}^{-}| / |\mathcal{D}|$ .

# 4.2 LATENT INFORMATION MODEL

In this model, we assume that the Knowledge Graph was generated according to the following generative model. Let  $\mathcal{V} \triangleq \mathcal{E} \times \mathcal{R} \times \mathcal{E}$  the space of possible triples. We have that:

where  $\tau \triangleq \langle \mathrm{s},\mathrm{p},\mathrm{o}\rangle$ , and  $\mathbf{h}_{\tau} \triangleq [\mathbf{h}_{\mathrm{s}},\mathbf{h}_{\mathrm{p}},\mathbf{h}_{\mathrm{o}}]$  denotes the sampled embedding representations of  $\mathrm{s}$ ,  $\mathrm{o} \in \mathcal{E}$  and  $\mathrm{p} \in \mathcal{R}$ . The set of latent variables in this model is  $\mathcal{H} \triangleq \{\mathbf{h}_e \mid e \in \mathcal{E}\} \cup \{\mathbf{h}_{\mathrm{p}} \mid \mathrm{p} \in \mathcal{R}\}$ . The joint probability of the variables  $p(\mathcal{H},\mathcal{D})$  is defined as follows:

$$
p (\mathcal {H}, \mathcal {D}) \triangleq \prod_ {e \in \mathcal {E}} p \left(\mathbf {h} _ {e}\right) \prod_ {\mathfrak {p} \in \mathcal {E}} p \left(\mathbf {h} _ {\mathfrak {p}}\right) \prod_ {(\tau , y _ {\tau}) \in \mathcal {D}} p \left(y _ {\tau} \mid \mathbf {h} _ {\tau}\right) \tag {7}
$$

The marginal distribution over  $\mathcal{D}$  is then defined as follows:

$$
p (\mathcal {D}) \triangleq \int \prod_ {e \in \mathcal {E}} p \left(\mathbf {h} _ {e}\right) \prod_ {\mathfrak {p} \in \mathcal {E}} p \left(\mathbf {h} _ {\mathfrak {p}}\right) \prod_ {(\tau , y _ {\tau}) \in \mathcal {D}} p \left(y _ {\tau} \mid \mathbf {h} _ {\tau}\right) d \mathcal {H} \tag {8}
$$

The log-marginal likelihood of the data is the following:

$$
\log p (\mathcal {D}) \geq \mathbb {E} _ {q} \left[ \log p (\mathcal {D} \mid \mathcal {H}) \right] - D _ {\mathrm {K L}} q (\mathcal {H}) p (\mathcal {H}) \tag {9}
$$

# 4.3 LINK PREDICTION

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Scoring Function</td><td colspan="2">MR</td><td colspan="3">Hits @</td></tr><tr><td>Filter</td><td>Raw</td><td>1</td><td>3</td><td>10</td></tr><tr><td rowspan="4">WN18</td><td>V DistMult ( LIM)</td><td>786</td><td>798</td><td>0.671</td><td>0.931</td><td>0.947</td></tr><tr><td>DistMult</td><td>813</td><td>827</td><td>0.754</td><td>0.911</td><td>0.939</td></tr><tr><td>V ComplEx ( LIM)</td><td>753</td><td>765</td><td>0.934</td><td>0.945</td><td>0.952</td></tr><tr><td>ComplEx*</td><td>-</td><td>-</td><td>0.939</td><td>0.944</td><td>0.947</td></tr><tr><td>WN18</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">RR</td><td>V DistMult ( LIM)</td><td>6095</td><td>6109</td><td>0.357</td><td>0.423</td><td>0.440</td></tr><tr><td>DistMult</td><td>8595</td><td>8595</td><td>0.367</td><td>0.390</td><td>0.412</td></tr><tr><td>V ComplEx ( LFM )</td><td>6500</td><td>6514</td><td>0.385</td><td>0.446</td><td>0.489</td></tr><tr><td>ComplEx**</td><td>5261</td><td>-</td><td>0.41</td><td>0.46</td><td>0.51</td></tr><tr><td>FB15K</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">-257</td><td>V DistMult ( LIM)</td><td>679</td><td>813</td><td>0.171</td><td>0.271</td><td>0.397</td></tr><tr><td>DistMult</td><td>355</td><td>501</td><td>0.187</td><td>0.282</td><td>0.400</td></tr><tr><td>V ComplEx ( LIM)</td><td>1221</td><td>1347</td><td>0.168</td><td>0.260</td><td>0.369</td></tr><tr><td>ComplEx**</td><td>339</td><td>-</td><td>0.159</td><td>0.258</td><td>0.417</td></tr></table>

Table 1: Filtered and Mean Rank (MR) for the models tested on the WN18, WN18RR and FB15K datasets. Hits@m metrics are filtered. Variational written with a "V". *Results reported from (Trouillon et al., 2016b) and **Results reported from (Dettmers et al., 2017) for ComplEx model

Table 1 shows definite improvements on WN18 for Variational ComplEx compared with the initially published ComplEX. We believe this due to the well-balanced model regularisation induced by the zero mean unit variance Gaussian prior.

We now compare our model to the previous state-of-the-art multi-relational generative model TransG (Xiao et al., 2016), as well as to a previously published probabilistic embedding method KG2E (similarly represents each embedding with a multivariate Gaussian distribution) (He et al., 2015) on the WN18 dataset.

Table 2 makes clear the improvements in the performance of the previous state-of-the-art generative multi-relational knowledge graph model. LFM has marginally worse performance than the state-of-the-art model on raw Hits@10. We conjecture two reasons may cause this discrepancy. Firstly, the fact the authors of TransG use negative samples provided only (True negative examples), whereas we generated our negative samples using the LCWA. Secondly, we only use one negative sample per

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Scoring Function</td><td colspan="2">MR</td><td>Raw Hits@</td><td colspan="3">Filtered Hits @</td></tr><tr><td>Raw</td><td>Filter</td><td>10</td><td>1</td><td>3</td><td>10</td></tr><tr><td rowspan="3">WN18</td><td>KG2E He et al. (2015)</td><td>362</td><td>345</td><td>0.805</td><td>-</td><td>-</td><td>0.932</td></tr><tr><td>TransG (Generative) Xiao et al. (2016)</td><td>345</td><td>357</td><td>0.845</td><td>-</td><td>-</td><td>0.949</td></tr><tr><td>Variational ComplEx ( LFM )</td><td>753</td><td>765</td><td>0.836</td><td>0.934</td><td>0.945</td><td>0.952</td></tr></table>

positive to estimate the Evidence Lower Bound using Bernoulli sampling, whereas it is likely they used significantly more negative samples. This conjecture was proved true in a follow-up experiment on Nations; increasing performance on raw Hits@10 when using 20 negative samples, with no change in filtered Hits@10.

# 5 LINK PREDICTION ANALYSIS

Section 5.1 and Section 5.2 explores the predictions made by Latent Information Model with ComplEx scoring function, trained with Bernoulli sampling to estimate the ELBO on the WN18RR dataset, then Section 5.3 will analyse the values of embeddings learnt for this task. Lastly, Section 5.3.1 will perform an extrinsic evaluation on learnt embedding representations for the more accessible to interpret Nations dataset.

We split the analysis into the predictions of subject  $((?,r,o))$  or object  $((s,r,?))$  for each test fact. Note all results are filtered predictions, i.e., ignoring the predictions made on negative examples generated under LCWA.

# 5.1 SUBJECT PREDICTION

Table 2: Variational Framework vs. Generative Modles  

<table><tr><td></td><td>Proportion</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td></tr><tr><td>_hypernym</td><td>0.399170</td><td>0.091926</td><td>0.123102</td><td>0.162270</td></tr><tr><td>.derivationally_related_form</td><td>0.342693</td><td>0.947858</td><td>0.956238</td><td>0.959032</td></tr><tr><td>_member_meronym</td><td>0.080728</td><td>0.007905</td><td>0.019763</td><td>0.035573</td></tr><tr><td>_has_part</td><td>0.054882</td><td>0.011628</td><td>0.058140</td><td>0.122093</td></tr><tr><td>_instance_hypernym</td><td>0.038928</td><td>0.393443</td><td>0.508197</td><td>0.713115</td></tr><tr><td>_synset_domain主題_of</td><td>0.036375</td><td>0.219298</td><td>0.315789</td><td>0.464912</td></tr><tr><td>_also Seeing</td><td>0.017869</td><td>0.589286</td><td>0.625000</td><td>0.625000</td></tr><tr><td>Verb_group</td><td>0.012444</td><td>0.743590</td><td>0.974359</td><td>0.974359</td></tr><tr><td>_member_of_domain_region</td><td>0.008296</td><td>0.000000</td><td>0.038462</td><td>0.115385</td></tr><tr><td>_member_of_domain_usage</td><td>0.007658</td><td>0.000000</td><td>0.000000</td><td>0.000000</td></tr><tr><td>_similar_to</td><td>0.000957</td><td>1.000000</td><td>1.000000</td><td>1.000000</td></tr></table>

Table 3: Latent Information Model with ComplEx: Subject Prediction on WN18RR

Table 3 shows that the relation "_derivationally_related_form", comprising  $34\%$  of test subject predictions, was the most accurate relation to predict for Hits@1 when removing the subject from the tested fact. Contrarily, "_member_of_domain_region" with zero Hits@1 subject prediction, making up less than  $1\%$  of subject test predictions. However, "_member_meronym" was the least accurate and prominent ( $8\%$  of the test subject predictions) for subject Hits@1.

# 5.2 OBJECT PREDICTION

Table 4 displays similar results to Table 3, as before the relation "_derivationally_related_form" was the most accurate relation to predict Hits@1. Table 4 differs from Table 3 as it highlights Model A's its inability to achieve a high Hits@1 performance predicting objects for the "_hypernym" relation, which is significantly hindering model performance as it is the most seen relation in the test set—its involvement in  $40\%$  of object test predictions.

<table><tr><td></td><td>Proportion</td><td>Hits@1</td><td>Hits@3</td><td>Hits@10</td></tr><tr><td>_hypernym</td><td>0.399170</td><td>0.000000</td><td>0.014388</td><td>0.046363</td></tr><tr><td>DERIVATIONALLY RELATED_form</td><td>0.342693</td><td>0.945996</td><td>0.957169</td><td>0.959032</td></tr><tr><td>_member_meronym</td><td>0.080728</td><td>0.031621</td><td>0.047431</td><td>0.086957</td></tr><tr><td>_has_part</td><td>0.054882</td><td>0.034884</td><td>0.081395</td><td>0.139535</td></tr><tr><td>_instance_hypernym</td><td>0.038928</td><td>0.024590</td><td>0.081967</td><td>0.131148</td></tr><tr><td>_synset_domain主題_of</td><td>0.036375</td><td>0.035088</td><td>0.043860</td><td>0.078947</td></tr><tr><td>_also Seeing</td><td>0.017869</td><td>0.607143</td><td>0.625000</td><td>0.625000</td></tr><tr><td>Verb_group</td><td>0.012444</td><td>0.897436</td><td>0.974359</td><td>0.974359</td></tr><tr><td>member_of_domain_region</td><td>0.008296</td><td>0.038462</td><td>0.076923</td><td>0.076923</td></tr><tr><td>member_of_domain_usage</td><td>0.007658</td><td>0.000000</td><td>0.000000</td><td>0.000000</td></tr><tr><td>_similar_to</td><td>0.000957</td><td>1.000000</td><td>1.000000</td><td>1.000000</td></tr></table>

Table 4: Latent Information Model with ComplEx: Object Prediction on WN18RR

# 5.3 EMBEDDING ANALYSIS

These results hint at the possibility that the slightly stronger results of WN18 are due to covariances in our variational framework able to capture information about symbol frequencies. We verify this by plotting the mean value of co-variance matrices, as a function of the entity or predicate frequencies (Figure 3). The plots confirm our hypothesis: covariances for the variational Latent Information Model grows with the frequency, and hence the LIM would put a preference on predicting relationships between less frequent symbols in the knowledge graph. This also suggests that covariances from the generative framework can capture genuine information about the generality of symbolic representations.

![](images/8a3cf0f61036b9d35dc22dca4fd6ca7edf3bbce04822878aa8fcd7e737e6c98b.jpg)

![](images/b224126b8e7b7a8d185ea60630010e617092c000c761d65f29e215300f69f131.jpg)

![](images/7dc40b63ad49017264bb6ba87c3bccc2487fdef16c55a772fb6619564d39b8eb.jpg)  
Figure 3: Mean Variance vs. log frequency. From left to right: Nations Entity Analysis, Nations Predicate Analysis, WN18RR Entity Analysis and WN18RR Predicate Analysis.

![](images/83c3b2d9150828a132687f32866761714727bb66aa71429e63e55d405f8b2d16.jpg)

# 5.3.1 EXTRINSIC EVALUATION: VISUAL EMBEDDING ANALYSIS

We project the high dimensional mean embedding vectors to two dimensions using Probabilistic Principal Component Analysis (PPCA) (Tipping & Bishop, 1999) to project the variance embedding vectors down to two dimensions using Non-negative Matrix Factorisation (NNMF) (Févotte & Idier,

2011). Once we have the parameters for a bivariate normal distribution, we then sample from the bivariate normal distribution 1,000 times and then plot a bi-variate kernel density estimate of these samples. By visualising these two-dimensional samples, we can conceive the space in which the entity or relation occupies. We complete this process for the subject, object, relation, and a randomly sampled corrupted entity (under LCWA) to produce a visualisation of a fact, as shown in Figure 4.

![](images/ef8787e582d15358da10c981d0600e720715a7a1757c38e943057b50dd8874e1.jpg)

![](images/3ffb427c5d8eb4639d00b399b1367ec3ff3c7e7b0235e00de78ca1a473f5626f.jpg)

![](images/48be95a63265f61b7e78d897c906414fb16d5f6395332cc253c942d1ff3001ab.jpg)  
Figure 4: True Positives

Figure 4 displays three true positives from test time predictions. The plots show that the variational framework can learn high dimensional representations which when projected onto lower (more interpretable) dimensions.

Figure 4 displays a clustering of the subject, object and predicate that create a positive (true) fact. We also observe a separation between the items which generate a fact and a randomly sampled (corrupted) entity which is likely to create a negative (false) fact. The first test fact " (USA, Commonbloc0, Netherlands)" shows clear irrationality similarity between all objects in the tested fact, i.e. the vectors are pointing towards a south-east direction. We can also see that the corrupted entity Jordan is quite a distance away from the items in the tested fact, which is good as Jordan does not share a common bloc either USA or Netherlands.

# 5.4 CONCLUSION

We have successfully created a framework allowing a model to learn embeddings of any prior distribution that permits a re-parametrisation trick via any score function that permits maximum likelihood estimation of the scoring parameters. We have shown, from preliminary experiments, that these display competitive results with current models. Overall, we believe this work will enable knowledge graph researchers to work towards the goal of creating models better able to express their predictive uncertainty.

# 6 FURTHER WORK

The score we acquire at test time even through forward sampling does not seem to differ much compared with the mean embeddings, thus using the learnt uncertainty to impact the results positively is a fruitful path. We would also like to see additional exploration into various encoding functions, as we used only the most basic for these experiments.

# ACKNOWLEDGMENTS

We would like to thank all members of the Machine Reading lab for useful discussions.

# REFERENCES

Oren Barkan. Bayesian neural word embedding. CoRR, abs/1603.06571, 2016. URL http://arxiv.org/abs/1603.06571.  
Daniel Bean, Honghan Wu, Olubanke Dzahini, Matthew Broadbent, Robert James Stewart, and Richard James Butler Dobson. Knowledge graph prediction of unknown adverse drug reactions and validation in electronic health records. *Scientific Reports*, 7(1), 11 2017. ISSN 2045-2322.  
Christopher M. Bishop. Pattern recognition and machine learning. Springer, 2006.  
C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra. Weight Uncertainty in Neural Networks. ArXiv e-prints, May 2015.  
Antoine Bordes, Nicolas Usunier, Alberto García-Durán, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013, pp. 2787-2795, 2013.  
Aleksandar Botev, Bowen Zheng, and David Barber. Complementary sum sampling for likelihood approximation in large scale classification. In Aarti Singh et al. (eds.), Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, AISTATS 2017, volume 54 of Proceedings of Machine Learning Research, pp. 1030-1038. PMLR, 2017.  
Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew M. Dai, Rafal Jozefowicz, and Samy Bengio. Generating sentences from a continuous space. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, CoNLL 2016, Berlin, Germany, August 11-12, 2016, pp. 10-21, 2016. URL http://aclweb.org/anthology/K/K16/K16-1002.pdf.  
A. Bražinskas, S. Havrylov, and I. Titov. Embedding Words as Distributions with a Bayesian Skip-gram Model. ArXiv e-prints, November 2017.  
Wenhu Chen, Wenhan Xiong, Xifeng Yan, and William Yang Wang. Variational knowledge graph reasoning. In *NAACL-HLT*, 2018.  
T. R. Davidson, L. Falorsi, N. De Cao, T. Kipf, and J. M. Tomczak. Hyperspherical Variational Auto-Encoders. *ArXiv eprints*, April 2018.  
Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2d knowledge graph embeddings. arXiv preprint arXiv:1707.01476, 2017.  
Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. Knowledge vault: a web-scale approach to probabilistic knowledge fusion. In Sofus A. Macskassy et al. (eds.), The 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '14, pp. 601-610. ACM, 2014. ISBN 978-1-4503-2956-9.  
Cédric Févotte and Jérôme Idier. Algorithms for nonnegative matrix factorization with the  $\beta$ -divergence. Neural Computation, 23(9):2421-2456, 2011. doi: 10.1162/NECO\a_00168. URL https://doi.org/10.1162/NECO_a_00168.  
Yarin Gal, Riashat Islam, and Zoubin Ghahramani. Deep bayesian active learning with image data. In Doina Precup et al. (eds.), Proceedings of the 34th International Conference on Machine Learning, ICML 2017, volume 70 of Proceedings of Machine Learning Research, pp. 1183-1192. PMLR, 2017.  
Zoubin Ghahramani. Probabilistic machine learning and artificial intelligence. Nature, 521(7553): 452-459, 2015.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Yee Whye Teh and Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 249-256, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. URL http://proceedings.mlr.press/v9/glorot10a.html.  
Shizhu He, Kang Liu, Guoliang Ji, and Jun Zhao. Learning to represent knowledge graphs with gaussian embedding. In Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, CIKM '15, pp. 623-632, New York, NY, USA, 2015. ACM. ISBN 978-1-4503-3794-6. doi: 10.1145/2806416.2806502. URL http://doi.acm.org/10.1145/2806416.2806502.  
Ashish Kapoor, Kristen Grauman, Raquel Urtasun, and Trevor Darrell. Active learning with gaussian processes for object categorization. In IEEE 11th International Conference on Computer Vision, ICCV 2007, pp. 1-8. IEEE Computer Society, 2007. ISBN 978-1-4244-1630-1.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
Diederik P Kingma and Max Welling. Auto-Encoding Variational Bayes. UvA, pp. 1-14, 2013. URL http://arxiv.org/abs/1312.6114.  
Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. CoRR, abs/1609.02907, 2016a.  
Thomas N Kipf and Max Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308, 2016b.  
Y. Miao, E. Grefenstette, and P. Blunsom. Discovering Discrete Latent Topics with Neural Variational Inference. *ArXiv e-prints*, June 2017.  
Yishu Miao, Lei Yu, and Phil Blunsom. Neural variational inference for text processing. Proceedings of the 33rd International Conference on Machine Learning, 2016.  
Andrew Y. Ng and Michael I. Jordan. On discriminative vs. generative classifiers: A comparison of logistic regression and naive bayes. In Thomas G. Dietterich et al. (eds.), Advances in Neural Information Processing Systems 14 [Neural Information Processing Systems: Natural and Synthetic, NIPS 2001], pp. 841-848. MIT Press, 2001.  
Maximilian Nickel, Kevin Murphy, Volker Tresp, and Evgeniy Gabrilovich. A review of relational machine learning for knowledge graphs. Proceedings of the IEEE, 104(1):11-33, 2016.  
Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. BPR: bayesian personalized ranking from implicit feedback. In Jeff A. Bilmes et al. (eds.), UAI 2009, Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pp. 452-461. AUAI Press, 2009.  
Michael E. Tipping and Chris M. Bishop. Probabilistic principal component analysis. JOURNAL OF THE ROYAL STATISTICAL SOCIETY, SERIES B, 61(3):611-622, 1999.  
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In Maria-Florina Balcan et al. (eds.), Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, volume 48 of JMLR Workshop and Conference Proceedings, pp. 2071–2080. JMLR.org, 2016a.  
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. CoRR, abs/1606.06357, 2016b. URL http://arxiv.org/abs/1606.06357.  
Luke Vilnis and Andrew McCallum. Word representations via gaussian embedding. CoRR, abs/1412.6623, 2014. URL http://arxiv.org/abs/1412.6623.

Ekaterina Vylomova, Laura Rimell, Trevor Cohn, and Timothy Baldwin. Take and Took, Gaggle and Goose, Book and Read: Evaluating the Utility of Vector Differences for Lexical Relation Learning. In ACL, 2016.

Han Xiao, Minlie Huang, and Xiaoyan Zhu. Transg: A generative model for knowledge graph embedding. In ACL, 2016.

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In ICLR, 2015.

Jingwei Zhang, Jeremy Salwen, Michael Glass, and Alfio Gliozzo. Word semantic representations using bayesian probabilistic tensor factorization. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014. doi: 10.3115/v1/d14-1161.
