# ANTONYMY-SYNONYMY DISCRIMINATION THROUGH THE REPELLING PARASIAMESE NEURAL NETWORK

Anonymous authors

Paper under double-blind review

# ABSTRACT

Antonymic and synonymous pairs may both occur nearby in word embeddings spaces because they have similar distributional information. Different methods have been used in order to distinguish antonyms from synonyms, making the antonymy-synonymy discrimination a popular NLP task. In this work, we propose the repelling parasiamese neural network, a model which considers a siamese network for synonymy and a parasiamese network for antonymy, both sharing the same base network. Relying in the antagonism between synoyny and antonymy, the model attempts to repell siamese and parasiamese outputs making use of the contrastive loss functions. We experimentally show that the repelling parasiamese network achieves state-of-the-art results on this task.

# 1 INTRODUCTION

Semantic opposition is a binary relation of central importance in the cognitive baggage of human languages. It establishes that one term contradicts the other, that both cannot be satisfied simultaneously. In the context of lexical semantics, it corresponds to antonyms (e.g. light and dark), whose recognition is essential for natural language usage. For instance, this capability is crucial for text entailment and paraphrasing, which are basic abilities for different NLP tasks.

Most of modern NLP is using word embeddings (i.e. vectors for word meanings built from word contexts and subword information). These word representations have the potential to cluster words according to its distributional information on a corpus. However, since antonyms tend to occur in similar contexts, word embeddings may have close vectors in the space. Faced to this problem, different approaches have been proposed to re-encode the word embeddings in a supervised learning setup for the antonymy-synonymy discrimination task (Mrkšić et al., 2017; Etcheverry & Wonsever, 2019; Samenko et al., 2020; Xie & Zeng, 2021).

In this work, we deepen in the parasiamese network as an antitransitive relationship learning approach, and we propose the repelling parasiamese neural network: a model that simultaneously opposes the siamese and parasiamese outputs (of a same base network). We present two alternatives to do so: (1) pair and (2) triplet based approaches. We consider the parasiamese branching alternatives and we propose a formulation to enforce symmetry through the model formulation. We execute our experiments in the publicly available antonymy-synonymy dataset introduced by Nguyen et al. (2016), in a here introduced dataset confecctionated from Samuel Fallow's antonym's dictionary (accessed through the Gutenberg project) and in a version of the Nguyen et al. (2016)'s dataset split-ted without lexical intersection between train, validation and test (Xie & Zeng, 2021). We show that the repelling parasiamese neural network achieves better performance than its predecessor, the (non-repelling) parasiamese network, and other models.

# 2 SOME PRELIMINARIES

Before getting into the repelling parasiamese neural network, let us introduce some preliminary concepts concerning antitransitivity, learning metrics for antonymy and the parasiamese formulation.

# 2.1 ANTONYMY AND ANTITRANSITIVITY

Antonymy can be thought of as an antitransitive relationship<sup>1</sup>. If two lexical units are antonyms of a third (e.g. huge and enormous being opposite of small) then they will not oppose each other; in fact, they will often present some semantic similarity (Edmundson, 1967).

Table 1: Antonymy list of a given word.  

<table><tr><td>word</td><td colspan="3">antonyms</td></tr><tr><td>tame</td><td>fierce</td><td>savage</td><td>wild</td></tr><tr><td>compound</td><td>decompose</td><td>sift</td><td>segment</td></tr><tr><td>robust</td><td>feeble</td><td>puny</td><td>languid</td></tr><tr><td>lose</td><td>get</td><td>own</td><td>possess</td></tr><tr><td>authentic</td><td>false</td><td>suppositious</td><td>fictitious</td></tr></table>

In table 1 we sample antonyms for some words from Fallow's dictionary. Supporting the claimed antonymy antitransitivity, it can be seen that the words on each antonymy list (i.e. common antonyms of a word) do not oppose between them, and many cases of similarity can be detected (e.g. savage and wild).

# 2.2 DISTANCE FUNCTIONS AND ANTITRANSITIVITY

Antitransitivity is a drawback if a distance function is pretended to be considered to distinguish antonymy, given that the triangular inequity is rather beneficial for transitivity than for antitransitivity. This is because the distance of the (anti)transitive links is bounded by the sum of the distances of the other two pairs forming the triangle, which are expected to be small since they are related.

To clarify, let suppose  $d: R^n \times R^n \to R_{\geq 0}$  a distance function capable of giving low values for antonymic pairs of word embeddings, consider a threshold  $\mu$  as reference. Let be  $(x,y)$  and  $(y,z)$  two pairs of antonyms where  $d(x,y) < \mu$  and  $d(y,z) < \mu$ . Due to the triangular inequality²,  $d(x,z) < 2\mu$ , which may not result very convenient to establish the pair  $(x,z)$  as unrelated.

Considering not near zero values for the threshold of the relation acceptance may give more suitable margins to represent an antitransitive relation within a distance function. We experimentally observed, using siamese networks, some performance improvement when higher thresholds were considered. However, better results were obtained with the repelling parasiamese neural network that we will present below and other commented models.

# 2.3 THE PARASIAMESE NETWORK

The parasiamese network (Etcheverry & Wonsever, 2019) was introduced as inspired by the siamese network, being better suited for the learning of antitransitive relationships. Just like the siamese network, it consists of a model that consumes two vectors as input and returns a non-negative value. The model relies on a base neural network that is applied more than once, sharing its weights, to compute the output. The parasiamese network differs from the siamese formulation in the fact that the base network is applied once to one input and twice to the other, instead of once to each input, as in the siamese network. The output of the network is the distance between both branches (see Figure 1).

Let  $F_{\theta}:\mathbb{R}^{n}\to \mathbb{R}^{n}$  be a neural network with trainable parameters  $\theta$ . Then, the parasiamese network with base network  $F_{\theta}$  is defined by

$$
\Phi_ {F _ {\theta}} (x, y) = \left| \left| F _ {\theta} (x) - F _ {\theta} \left(F _ {\theta} (y)\right) \right| \right| _ {2}, \tag {1}
$$

where  $||.||_2$  is the Euclidean norm. The model is trained through the contrastive loss function. Concretely,  $\Phi_{F_\theta}$  is trained through mini-batch stochastic gradient descent on:

![](images/1a4b84435a320eb0a2b12f2b57eaaed110e66a8841c77a994dd9ff1ea360de1a.jpg)  
Figure 1: The siamese network (left) and the parasiamese network (right) diagrams.  $F_{\theta}$  corresponds to a neural network with parameters  $\theta$  and  $d$  a distance function (e.g. Euclidean).

![](images/7b3d6baf2d0b405ae20f54ccbd602497cb561d9b8518b0643c8c074afcfeb7e7.jpg)

$$
L = \sum_ {(x, y) \in P} \left[ \Phi_ {F _ {\theta}} (x, y) - \mu_ {p} \right] _ {+} + \sum_ {\left(x ^ {\prime}, y ^ {\prime}\right) \in N} \left[ \mu_ {n} - \Phi_ {F _ {\theta}} \left(x ^ {\prime}, y ^ {\prime}\right) \right] _ {+}, \tag {2}
$$

where  $P$  and  $N$  are, respectively, the positive and negative pairs in the dataset; and  $\mu_p$  and  $\mu_n$  are the positive and negative thresholds, respectively. The  $[.]_{+}$  notation corresponds to the ReLU function. The training attempts to pull closer than  $\mu_p$  the related elements and push away unrelated pairs further than  $\mu_n$ . Notice that this definition, unlike the siamese network, does not enforce transitivity even when the parasiamese output of the related pairs in the transitive triangle is strictly zero. Moreover, the relation given by  $\Phi_{F_\theta}$  and a threshold  $\mu$  (i.e.  $R_{\Phi_{F_\theta},\mu} = \{(a,b):\Phi_{F_\theta}(a,b)\leq \mu\}$ ) may allow the antitransitivity property if  $\Phi_{F_\theta}(w,w) > \mu$ , which is consistent since antitransitive relations are necessarily anti-reflexive<sup>3</sup>.

An interpretation for the parasiamese model definition is thinking the base network  $F$  as an opposition transformation. So, if we consider two opposite terms  $a$  and  $b$  (i.e.  $a \sim \neg b$ ), it is expected that opposition remains when both terms are negated (i.e.  $\neg a \sim \neg \neg b$ ), which brings the parasiamese formulation:  $F(a) \sim F(F(b))$ .

# 2.3.1 ANTITRANSITIVE AND ANTI-EUCLIDEAN

It is interesting to notice that the parasiamese definition deals with antitransitivity regardless of whether the relation is symmetric or not. If symmetry is not given, the parasiamese network may be suited to learn a relation that satisfies the following property:

$$
a R b \wedge c R b \Longrightarrow a \mathcal {R} c \tag {3}
$$

that we can refer as anti-Euclidean relations, particularly, left anti-Euclidean in this case. However, an anti-Euclidean relation is necessarily antitransitive. We give the proof for left anti-Euclidean.

Property. If  $R$  is a left anti-Euclidean relation  $\Rightarrow R$  is antitransitive

Proof. Let  $R$  be not antitransitive, then it must exist  $a, b, c$  such that  $a R b \wedge b R c \wedge a R c$ . But, given that  $R$  is left anti-Euclidean and  $a R c \wedge b R c$ , then  $a R b$ , which contradicts the above.

Note that even though the parasiamese network is suitable to antitransitive relations regardless symmetry, the low valued output of the siamese formulation of the same base network on transitive links needs to be adapted according to property 3, for non-symmetric relations. Particularly, given two related pairs  $(a,b)$  and  $(c,b)$ , then the pair  $(a,c)$  would tend to have a low siamese output value.

# 3 THE REPELLING PARASIAMESE NETWORK

The repelling parasiamese network is based on contrasting the siamese and parasiamese networks in a differentiable formulation (Figure 2). By doing this, we consistently observe a performance improvement in comparison to its predecessor (i.e. the parasiamese network without repelling its siamese counterpart).

![](images/4a16562e357f1b0847839dc97aa9fd14b0290765c35e046258a59e61056734f7.jpg)  
Figure 2: Diagram of the here proposed repelling parasiamese network.  $d$  corresponds to a distance function (e.g. Euclidean) and  $c$  is a function that contrasts the siamese and parasiamese outputs (e.g. contrastive loss).

The siamese counterpart of a parasiamese network may reflect the similitude-like relation that emerges from being opposed to the same elements through the antitransitive relation. Recalling the case of lexical semantics, the siamese formulation from a parasiamese network that models antonymy, may be suitable to model synonymy, since the words that share antonyms may tend to be synonyms. So, given the antagonism between antonymy and synonymy, it seems suitable that both outputs should not be low simultaneously for the same outputs.

Let  $a, b, c$  be three inputs with the pairs  $a, b$  and  $b, c$  being related and therefore returning low parasi-amese outputs. Then, the output for the siamese formulation of the same base network for the pair  $a, c$  will present a low value, because it has to be less or equal than the sum of the parasiamese outputs of  $a, b$  and  $b, c$ , due to the Euclidean distance triangular inequity. Simultaneously, its parasi-amese output is expected to be greater than the acceptance threshold, because of the antitransitivity. Moreover, if  $a, b$  or  $b, c$  also returns low siamese output, it would implies a reflexive pair within the antitransitive relation, which is inconsistent. This enforces that the parasiamese and siamese outputs of a same base network may contrast each other when one of them returns a low value.

To describe the repelling parasiamese network let us introduce the following notation:

- Parasiamese left and right branches: We will refer as left and right branches of the parasiamese network to the transformations applied to the left and right parts of the relationship (that correspond to the left and right terms of the Euclidean distance in equation 1), and we will write them as  $\alpha_{\theta_{\alpha}}: \mathbb{R}^n \to \mathbb{R}^n$  and  $\beta_{\theta_{\beta}}: \mathbb{R}^n \to \mathbb{R}^n$ , respectively. So, in the non-repelling proposal  $\alpha_{\theta}(x) = F_{\theta}(x)$  and  $\beta_{\theta}(x) = F_{\theta}(F_{\theta}(x))$ .  
- Parasiamese output function: We will use the notation  $\Phi_{\alpha_{\theta_{\alpha}}, \beta_{\theta_{\beta}}}$  for to the binary function that given the left and right branches,  $\alpha_{\theta_{\alpha}}$  and  $\beta_{\theta_{\beta}}$ , returns the distance between them, i.e.  $\Phi_{\alpha_{\theta_{\alpha}}, \beta_{\theta_{\beta}}} (x_1, x_2) = ||\alpha_{\theta_{\alpha}}(x_1) - \beta_{\theta_{\beta}}(x_2)||_2^2$ . Notice that  $\Phi_{\alpha_{\theta_{\alpha}}, \alpha_{\theta_{\alpha}}} (x_1, x_2)$  corresponds to the siamese network formulation.

# 3.1 SIAMESE-PARASIAMESE REPULSION

In order to formulate the repelling parasiamese network, lets consider the parasiamese output function  $\Phi_{\alpha \theta_{\alpha}}, \beta_{\theta_{\beta}}$  as the following two functions:

-  $\Phi_{\alpha_{\theta_{\alpha}}, \beta_{\theta_{\beta}}}^{(p)}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$  as a function that returns a low value when the parasiamese output presents a low value and the siamese network of the same base network a high value (e.g. higher than a threshold).  
-  $\Phi_{\alpha_{\theta_{\alpha}}, \beta_{\theta_{\beta}}}^{(s)}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\geq 0}$  as a function that returns low value when the siamese network (with same base as the parasiamese) presents a low value and the parasiamese network a high value (e.g. higher than a threshold).

So,  $\Phi_{\alpha_{\theta_{\alpha}},\beta_{\theta_{\beta}}}^{(p)}$  and  $\Phi_{\alpha_{\theta_{\alpha}},\beta_{\theta_{\beta}}}^{(s)}$  are the parasiamese and siamese networks, respectively, with its respective counterparts repelled on its formulation. While the parasiamese network will be suitable to learn antitransitive relations (or anti-Euclidean in case of non-symmetry), its siamese counterpart will be useful to learn the similitude-like relation that emerges from the former, given by being related to the same elements, in terms of the antitransitive relation. The training is performed through the minimization of the following loss function:

$$
\sum_ {(x, y) \in P} \Phi_ {\alpha , \beta} ^ {(p)} (x, y) + \sum_ {(x ^ {\prime}, y ^ {\prime}) \in N} \Phi_ {\alpha , \beta} ^ {(s)} (x ^ {\prime}, y ^ {\prime}),
$$

where  $P$  and  $N$  are antonymy and synonymy instances, respectively. Notice that contrast details and thresholds (if some) are delegated to each  $\Phi$  function definition.

We propose two formulations inspired and aligned to two of the main approaches to deep metric learning: pair and triplet based. Essentially, both formulations consider the same information, the outputs of the parasiamese and siamese networks, but the repelling is driven in slightly different ways. Given the branches  $\alpha$  and  $\beta$  and the input  $(x,y)$ , the pair based approach minimizes  $(\alpha(x),\beta(y))$  maximizing  $(\alpha(x),\alpha(y))$ , while the triplet based approach considers the triplet  $(\alpha(x),\beta(y),\alpha(y))$  attempting to get  $(\alpha(x),\beta(y))$  closer than  $(\alpha(x),\alpha(y))$ .

# 3.1.1 PAIR BASED

This approach considers positive and negative pairs, pretending the distance between related pairs to be lower than a given threshold  $\mu_p$  and the distance between unrelated pairs to be higher than a threshold  $\mu_n$ , by terms of hinge expression. The pair based  $\Phi$  functions are written as:

$$
\Phi_ {\alpha_ {\theta_ {\alpha}}, \beta_ {\theta_ {\beta}}} ^ {(p)} (x _ {1}, x _ {2}) = [ \Phi_ {\alpha_ {\theta_ {\alpha}}, \beta_ {\theta_ {\beta}}} (x _ {1}, x _ {2}) - \mu_ {p} ] _ {+} + [ \mu_ {n} - \Phi_ {\alpha_ {\theta_ {\alpha}}, \alpha_ {\theta_ {\alpha}}} (x _ {1}, x _ {2}) ] _ {+}
$$

$$
\Phi_ {\alpha_ {\theta_ {\alpha}}, \beta_ {\theta_ {\beta}}} ^ {(s)} (x _ {1}, x _ {2}) = [ \Phi_ {\alpha_ {\theta_ {\alpha}}, \alpha_ {\theta_ {\alpha}}} (x _ {1}, x _ {2}) - \mu_ {p} ] _ {+} + [ \mu_ {n} - \Phi_ {\alpha_ {\theta_ {\alpha}}, \beta_ {\theta_ {\beta}}} (x _ {1}, x _ {2}) ] _ {+};
$$

where  $\mu_p$  and  $\mu_n$  are the positive and negative margins, respectively. A disadvantage of this formulation is that the same margin is applied to every pair. This is addressed, theoretically, by means of the triplet loss function (Musgrave et al., 2020) that we consider in the following section.

# 3.1.2 TRIPLET BASED

The triplet loss is based on the triplet network concept (Hoffer & Ailon, 2015). Given a set  $T$  of triplets of elements  $(a,p,n)$  where  $p$  is related to  $a$  and  $n$  unrelated to  $a$ , the triplet loss function attempts to make the distance between  $a$  and  $p$  smaller than the distance between  $a$  and  $n$  by a margin  $\mu$ , through minimizing:

$$
\sum_ {(a, p, n) \in T} [ | | a - p | | _ {2} - | | a - n | | _ {2} + \mu ] _ {+}
$$

For the triplet based  $\Phi_{\alpha,\beta}^{(p)}$  and  $\Phi_{\alpha,\beta}^{(s)}$  functions, given a pair  $(x,y)$ , we consider the triplets:  $(\alpha(x),\beta(y),\alpha(y))$  and  $(\alpha(x),\alpha(y),\beta(y))$ , respectively. Hence, the triplet based  $\Phi$  functions are written as:

$$
\Phi_ {\alpha , \beta} ^ {(p)} (x, y) = \left[ \Phi_ {\alpha , \beta} (x, y) - \Phi_ {\alpha , \alpha} (x, y) + \mu_ {t} \right] _ {+}
$$

$$
\Phi_ {\alpha , \beta} ^ {(s)} (x, y) = [ \Phi_ {\alpha , \alpha} (x, y) - \Phi_ {\alpha , \beta} (x, y) + \mu_ {t} ] _ {+}
$$

where  $\mu_t$  is the separation between the positive and negative samples.

# 3.2 PARASIAMESE BRANCHES

The repelling between the outputs of the parasiamese and the siamese networks does not rely on any particular way of the right branch of the parasiamese network. The base network double application is just an alternative that allows to share the weights between both branches and it is inspired in the logic negation, but only having both branches distinguished is needed. In the following we detail the two variants that we consider in this work.

- (Standard) Parasiamese: Corresponds to the original formulation of the parasiamese network, where  $\alpha_{\theta}(x) = F_{\theta}(x)$  and  $\beta_{\theta}(x) = F_{\theta}(F_{\theta}(x))$  
- Half-twin Parasiamese: This formulation completely unties the weights between both branches. Both branches consist on entirely different networks, without shared weights. Accordingly,  $\alpha_{\theta_{\alpha}}(x) = F_{\theta_{\alpha}}(x)$  and  $\beta_{\theta_{\beta}}(x) = G_{\theta_{\beta}}(x)$ .

Notice that both definitions are suitable for the previously introduced repulsion formulations. In the case of the half-twin parasiamese case, the siamese network (for repelling) corresponds to the left branch.

# 3.3 SYMMETRIC PARASIAMESE NETWORK

The definition of the parasiamese network is not symmetric, this may be suitable to learn nonsymmetric relations. However, for symmetric relations it may be desirable to guarantee symmetry by definition.

We refer as right (left) parasiamese to the formulation with the right (left) branch distinguished (by double application or different base network in the half-twin parasiamese case). The parasiamese network can be formulated as a symmetric function if the two versions of the parasiamese network (left and right) are combined to be minimized. In this work we experiment through adding them. In section 4.3, we compare the performance of the symmetric and non-symmetric variants showing in most of cases a better performance for the symmetric in the antonymy detection task.

Table 2: Antonymy and synonymy classification alternatives according to the siamese and parasi-amese sub-networks outputs and a threshold  $\mu$  

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">parasiamese</td></tr><tr><td>&lt; μ</td><td>&gt; μ</td></tr><tr><td rowspan="2">siamese</td><td>&lt; μ</td><td>(1)</td><td>synonyms</td></tr><tr><td>&gt; μ</td><td>antonyms</td><td>(2)</td></tr></table>

# 3.4 PARASIAMESE AND SIAMESE SUB-NETWORKS

The here presented repelling parasiamese network has been designed to discern opposite from similar elements. However, it is possible to afford unrelated pairs (i.e. neither similar nor opposite terms) with it, when both sub-networks, the parasiamese and siamese outputs, present high values, suggesting that the elements of the candidate pair are neither similar nor opposite.

The table 2 shows the antonymy-synonymy classification using the repelling parasiamese network according to the parasiamese and siamese sub-networks outputs. Besides antonymy and synonymy

regions, the region denoted by (1) refers to pairs that are simultaneously synonymous and antonymic or a self-antonymic term (e.g. rent). And the region (2) belongs to pairs that are neither synonyms nor antonyms.

As a future work we comment that a 3-way loss could be defined considering opposition, similarity and unrelatedness. A challenge on this direction is the unrelated pairs mining strategy, which contains a large space of possibilities.

# 4 EXPERIMENTS AND DISCUSSION

In the following we detail the settings and results of the experiments we conducted, in the antonymynsynchrony discrimination task using general purpose pre-trained word embeddings as input.

# 4.1 DATASETS AND WORD EMBEDDINGS

To perform our experiments we need number of words pairs, labelled with if they are synonyms or antonyms. We consider the following datasets:

- Nguyen's: This dataset was built by Nguyen et al. (2016) using WordNet (Miller, 1995) and Wordnik<sup>4</sup>. It consists of 15,632 pairs of words with a balanced amount of synonyms and antonyms, over a vocabulary of 9,405 words.  
- Fallows's: We introduced a dataset for synonym/antonym distinction from the book "Complete Dictionary of Synonyms and Antonyms", by Samuel Fallows; available through the Gutenberg project. We automatically processed the electronic version of the book, obtaining a number of 25,419 antonym and 32,302 synonym pairs, with a vocabulary of 15,698 distinct words (5,810 in common to Nguyen's dataset).  
- Xie's: This dataset was built by Xie & Zeng (2021) using Nguyen et al. (2016)'s dataset but splitting it (into train, validation and test) avoiding words in common between each part (i.e. lexical intersection). It consists of 12,732 pairs of words with a balanced amount of synonyms and antonyms, over a vocabulary of 9,404 words.

We perform all our experiments using the word embeddings from the pretrained fastText (Joulin et al., 2016) model for English available in the fastText site<sup>6</sup>. This model was trained using Wikipedia<sup>7</sup> and Common Crawl<sup>8</sup>. The resulting vectors are in dimension 300 and there are not any out of vocabulary word over any dataset since fastText considers subword information.

# 4.2 EVALUATION PROCEDURE

On recent works, deep metric learning advances have been criticized for its evaluation methodology (Musgrave et al., 2020; Fehervari et al., 2019). It has been detected unsuitable hyperparameter settings, leading to unfair comparisons. In order to avoid that, we perform an independent random search over each model to obtain a suitable hyperparameter configuration against the validation set. Then, we report the results of each trained model according to the test set.

The hyperparameter space is considerably complex in these models. We divide it on three classes of hyperparameters: parasiamese hyperparameters, base network and training. The parasiamese hyperparameters corresponds to each margin value (e.g. positive, negative and acceptance), branching type (vanilla, half-twin), symmetric (or not), and the repelling type (pair, triplet or no-repelling). The base network hyperparemets are structural (type of network, number of layers, each layer size and activation function). Finally, the training hyperparameters include the optimization algorithm, learning rate and batch size. We include the random search details in Appendix A.

To evaluate our models we use precision, recall and F1 scores. In addition, we include some subnetwork outputs on pairs forming triangles with antitransitive links.

<table><tr><td rowspan="3">Model</td><td colspan="7">Nguyen&#x27;s</td><td colspan="3">Fallows&#x27;s</td></tr><tr><td colspan="2">Adjective</td><td colspan="3">Verb</td><td colspan="2">Noun</td><td colspan="3"></td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td></td></tr><tr><td></td><td colspan="10">Random Split</td></tr><tr><td>Xie &amp; Zeng (2021)</td><td>.878</td><td>.907</td><td>.892</td><td>.895</td><td>.920</td><td>.908</td><td>.841</td><td>.900</td><td>.869</td><td>- - - - -</td></tr><tr><td>Ali et al. (2019)</td><td>.854</td><td>.917</td><td>.884</td><td>.871</td><td>.912</td><td>.891</td><td>.823</td><td>.866</td><td>.844</td><td>- - - - -</td></tr><tr><td>Etcheverry &amp; Wonsever (2019)</td><td>.855</td><td>.857</td><td>.856</td><td>.864</td><td>.921</td><td>.891</td><td>.837</td><td>.859</td><td>.848</td><td>.847 .886 .866</td></tr><tr><td>Siamese</td><td>.607</td><td>.868</td><td>.714</td><td>.695</td><td>.927</td><td>.794</td><td>.682</td><td>.929</td><td>.787</td><td>.455 .927 .611</td></tr><tr><td>Half-twin</td><td>.788</td><td>.881</td><td>.832</td><td>.785</td><td>.830</td><td>.807</td><td>.758</td><td>.839</td><td>.796</td><td>.816 .863 .839</td></tr><tr><td>Parasiamese*</td><td>.821</td><td>.889</td><td>.854</td><td>.831</td><td>.899</td><td>.863</td><td>.813</td><td>.851</td><td>.831</td><td>.819 .890 .853</td></tr><tr><td>R-Parasiam (P)</td><td>.919</td><td>.860</td><td>.889</td><td>.885</td><td>.918</td><td>.901</td><td>.897</td><td>.820</td><td>.857</td><td>.903 .886 .894</td></tr><tr><td>R-Parasiam (P) HTwin</td><td>.865</td><td>.825</td><td>.844</td><td>.826</td><td>.877</td><td>.850</td><td>.807</td><td>.814</td><td>.811</td><td>.918 .872 .894</td></tr><tr><td>R-Parasiam Sym (P)</td><td>.927</td><td>.863</td><td>.894</td><td>.914</td><td>.916</td><td>.915</td><td>.876</td><td>.820</td><td>.847</td><td>.869 .943 .904</td></tr><tr><td>R-Parasiam Sym HTwin (P)</td><td>.922</td><td>.878</td><td>.899</td><td>.910</td><td>.934</td><td>.922</td><td>.877</td><td>.855</td><td>.866</td><td>.913 .914 .914</td></tr><tr><td>R-Parasiam (T)</td><td>.871</td><td>.874</td><td>.872</td><td>.841</td><td>.919</td><td>.878</td><td>.808</td><td>.839</td><td>.823</td><td>.877 .874 .876</td></tr><tr><td>R-Parasiam (T) HTwin</td><td>.867</td><td>.840</td><td>.853</td><td>.830</td><td>.870</td><td>.849</td><td>.816</td><td>.784</td><td>.800</td><td>.873 .875 .874</td></tr><tr><td>R-Parasiam Sym (T)</td><td>.924</td><td>.831</td><td>.875</td><td>.898</td><td>.910</td><td>.904</td><td>.862</td><td>.806</td><td>.833</td><td>.896 .870 .883</td></tr><tr><td>R-Parasiam Sym HTwin (T)</td><td>.920</td><td>.886</td><td>.903</td><td>.874</td><td>.919</td><td>.896</td><td>.853</td><td>.851</td><td>.852</td><td>.915 .869 .892</td></tr><tr><td></td><td colspan="10">Lexical Split</td></tr><tr><td>Xie &amp; Zeng (2021)</td><td>.808</td><td>.810</td><td>.809</td><td>.830</td><td>.693</td><td>.753</td><td>.846</td><td>.722</td><td>.776</td><td>- - - - -</td></tr><tr><td>R-Parasiam Sym HTwin (P)</td><td>.735</td><td>.885</td><td>.803</td><td>.725</td><td>.904</td><td>.804</td><td>.752</td><td>.870</td><td>.807</td><td>- - - - -</td></tr></table>

Table 3: Table comparing state-of-art results like (Xie & Zeng, 2021; Ali et al., 2019) and the non-repelling parasiamese network (Etcheverry & Wonsever, 2019) (first block) and the repelling parasiamese networks pairs and triplet based (third and fourth blocks, respectively). For comparison purpose we include best results obtained withing a siamese, half-twin and parasiamese* networks (second block). Parasiamese* stands for non-symmetric non-repelling parasiamese original proposal without the synonymy-based pre-training considered by Etcheverry & Wonsever (2019). The lower part of the table corresponds to the results obtained using the Xie & Zeng (2021)'s lexically split dataset.

# 4.3 RESULTS

We present the obtained results on antonym-synonym distinction task in Table 3. We compare the repel-parasiamese network to its predecessor, the (non-repelling) parasiamese network (Etcheverry & Wonsever, 2019), the Distiller (Ali et al., 2019), and MoE-ASD $^{9}$  (Xie & Zeng, 2021). We consider half-twin variants for the repelling and non-repelling networks; and we include the best result obtained using a siamese network for comparison purposes.

It can be observed that the repelling parasiamese network consistently outperforms its non-repelling predecessor. Regarding symmetry, the symmetric variants obtain the best results; and the half-twin branching improves the results for the symmetric formulations, while it degrades for the non-symmetric variants.

In comparison to the other reported methods, the repelling parasiamese network achieved better results at least in two of the three sub-sets of Nguyen's dataset (original and lexically splitted). Compared to MoE-ASD it is worth mentioning that the repelling parasiamese network reach competitive results without explicitly considering that the antonymic salient dimensions in the semantic space may vary for different antonymic pairs.

# 4.3.1 ANTITRANSITIVE LINKS

Since the main design consideration of the model is concerning antitransitivity, in Table 4 we show the model outputs on some antitransitive triplets (i.e. two words having an antonym in common).

In the Table we sample parasiamese and siamese outputs for pairs taken from the validation dataset partition (random split). We consider pairs forming triangles where antitransitive property should

Table 4: Samples of pairs forming antitransitive triangles with it respective parasiamese and siamese outputs. The acceptance threshold is 4.0.  

<table><tr><td>word1</td><td>word2</td><td>psiam</td><td>siam</td></tr><tr><td>real</td><td>aerial</td><td>3.24</td><td>4.12</td></tr><tr><td>real</td><td>notional</td><td>2.88</td><td>7.24</td></tr><tr><td>aerial</td><td>notional</td><td>7.84</td><td>4.45</td></tr><tr><td>valid</td><td>bad</td><td>2.60</td><td>4.49</td></tr><tr><td>valid</td><td>false</td><td>2.18</td><td>5.03</td></tr><tr><td>bad</td><td>false</td><td>12.43</td><td>0.79</td></tr><tr><td>bottom</td><td>lateral</td><td>2.30</td><td>4.47</td></tr><tr><td>bottom</td><td>top</td><td>3.41</td><td>6.09</td></tr><tr><td>lateral</td><td>top</td><td>6.60</td><td>3.04</td></tr><tr><td>realistic</td><td>fantastic</td><td>3.96</td><td>4.01</td></tr><tr><td>realistic</td><td>utopian</td><td>3.46</td><td>4.68</td></tr><tr><td>fantastic</td><td>utopian</td><td>6.60</td><td>4.48</td></tr></table>

be satisfied. For example, in the triplet (real, aerial,notional) the parasiamese output represents the antitransitivity (see psiam column) and the siamese output is not below the acceptance margin in any case, which is correct according to the dataset. In the triplet (valid, bad,false), the antitransitivity is stated by the parasiamese output and the siamese output correctly indicates similarity between bad and false. The triplet (bottom, lateral,top) may be debatable. The model inferred it as antitransitive as in the dataset, and similarity between lateral and top is slightly high which we take as a mistake. Lastly, for the triplet (realistic, fantastic, utopian), the antitransitivity is also represented (rightly according to the dataset) and any of the pairs is stated as similar by the siamese network.

# 5 CONCLUSION

In this work we deepen the parasiamese network and we introduce the repelling parasiamese network. We show that it is beneficial, in the parasiamese formulation, to repel the siamese counterpart of the same base network, to distinguish antonyms and synonyms; re-encoding pretrained word embeddings. The model achieves better results than its predecessor, using the weights of one (or two in the case of half-twin formulations) few-layered fully connected feed forward network. We perform our experiments in the Nguyen et al. (2016) publicly available dataset, the lexically split version provided by Xie & Zeng (2021), and we introduced a new dataset built from Samuel Fallow's antonym's dictionary accessed through the Gutenberg project. In addition, we show that the base network encodes meaningful information in terms of opposition and similarity.

# REFERENCES

Muhammad Asif Ali, Yifang Sun, Xiaoling Zhou, Wei Wang, and Xiang Zhao. Antonym-synonym classification based on new sub-space embeddings. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 6204-6211, 2019.  
H. P. Edmundson. Axiomatic characterization of synonymy and antonymy. In *COLING* 1967 Volume 1: Conference Internationale Sur Le Traitement Automatique Des Langues, 1967. URL http://aclweb.org/anthology/C67-1025.  
Mathias Etcheverry and Dina Wonsever. Unraveling antonym's word vectors through a siamese-like network. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3297-3307, 2019.  
Istvan Fehervari, Avinash Ravichandran, and Srikar Appalaraju. Unbiased evaluation of deep metric learning algorithms. arXiv preprint arXiv:1911.12528, 2019.  
Elad Hoffer and Nir Ailon. Deep metric learning using triplet network. In International workshop on similarity-based pattern recognition, pp. 84-92. Springer, 2015.

Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Hérve Jégou, and Tomas Mikolov. Fasttext. zip: Compressing text classification models. arXiv preprint arXiv:1612.03651, 2016.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
George A Miller. Wordnet: a lexical database for english. Communications of the ACM, 38(11): 39-41, 1995.  
Nikola Mrkšić, Ivan Vulić, Diarmuid Řéaghdha, Ira Leviant, Roi Reichart, Milica Gašić, Anna Korhonen, and Steve Young. Semantic specialization of distributional word vector spaces using monolingual and cross-lingual constraints. Transactions of the Association for Computational Linguistics, 5:309–324, 2017. doi: 10.1162/tacl_a_00063. URL https://www.aclweb.org/anthology/Q17-1022.  
Kevin Musgrave, Serge Belongie, and Ser-Nam Lim. A metric learning reality check. arXiv preprint arXiv:2003.08505, 2020.  
Kim Anh Nguyen, Sabine Schulte im Walde, and Ngoc Thang Vu. Integrating distributional lexical contrast into word embeddings for antonym-synonym distinction. arXiv preprint arXiv:1605.07766, 2016.  
Igor Samenko, Alexey Tikhonov, and Ivan P Yamshchikov. Synonyms and antonyms: Embedded conflict. arXiv preprint arXiv:2004.12835, 2020.  
Zhipeng Xie and Nan Zeng. A mixture-of-experts model for antonym-synonym discrimination. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pp. 558-564, 2021.
