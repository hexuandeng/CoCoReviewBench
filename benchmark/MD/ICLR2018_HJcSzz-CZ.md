# META-LEARNING FOR SEMI-SUPERVISED FEW-SHOT CLASSIFICATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

In few-shot classification, we are interested in learning algorithms that train a classifier from only a handful of labeled examples. Recent progress in few-shot classification has featured meta-learning, in which a parameterized model for a learning algorithm is defined and trained on episodes representing different classification problems, each with a small labeled training set and its corresponding test set. In this work, we advance this few-shot classification paradigm towards a scenario where unlabeled examples are also available within each episode. We consider two situations: one where all unlabeled examples are assumed to belong to the same set of classes as the labeled examples of the episode, as well as the more challenging situation where examples from other distractor classes are also provided. To address this paradigm, we propose novel extensions of Prototypical Networks (Snell et al., 2017) that are augmented with the ability to use unlabeled examples when producing prototypes. These models are trained in an end-to-end way on episodes, to learn to leverage the unlabeled examples successfully. We evaluate these methods on versions of the Omniglot and miniImageNet benchmarks, adapted to this new framework augmented with unlabeled examples. We also propose a new split of ImageNet, consisting of a large set of classes, with a hierarchical structure. Our experiments confirm that our Prototypical Networks can learn to improve their predictions due to unlabeled examples, much like a semi-supervised algorithm would.

# 1 INTRODUCTION

The availability of large quantities of labeled data has enabled deep learning methods to achieve impressive breakthroughs in several tasks related to artificial intelligence, such as speech recognition, object recognition and machine translation. However, current deep learning approaches struggle in tackling problems for which labeled data are scarce. Specifically, while current methods excel at tackling a single problem with lots of labeled data, methods that can simultaneously solve a large variety of problems that each have only a few labels are lacking. Humans on the other hand are readily able to rapidly learn new classes, such as new types of fruit when we visit a tropical country. This significant gap between human and machine learning provides fertile ground for deep learning developments.

For this reason, recently there has been an increasing body of work on few-shot learning, which considers the design of learning algorithms that specifically allow for better generalization on problems with small labeled training sets. Here we focus on the case of few-shot classification, where the given classification problem is assumed to contain only a handful of labeled examples per class. One approach to few-shot learning follows a form of meta-learning<sup>1</sup> (Thrun, 1998; Hochreiter et al., 2001), which performs transfer learning from a pool of various classification problems generated from large quantities of available labeled data, to new classification problems from classes unseen at training time. Meta-learning may take the form of learning a shared metric (Vinyals et al., 2016; Snell et al., 2017), a common initialization for few-shot classifiers (Ravi & Larochelle, 2017; Finn et al., 2017) or a generic inference network (Santoro et al., 2016; Mishra et al., 2017).

These various meta-learning formulations have led to significant progress recently in few-shot classification. However, this progress has been in a limited scenario, which differs in many dimensions from how humans learn new concepts. In this paper we aim to generalize the few-shot setting in two ways. First we consider a scenario in which the new classes are learned in the presence of additional unlabeled data. While there has been many successful applications of semi-supervised learning to the regular setting of a single classification task (Chapelle et al., 2010) where classes at training and test time are the same, such work has not addressed the challenge of performing transfer to new classes never seen at training time, as we consider here. Second, we consider the situation where the new classes to be learned are not viewed in isolation. Instead, many of the unlabeled examples are from different classes; the presence of such distractor classes introduces an additional and more realistic level of difficulty to the few-shot problem.

![](images/1d08127fa2f74e2acf259a970539ea43484c797be2320e6e4eb0a92a644955a4.jpg)  
"goldfish"

![](images/3af44d750624d06337bbffb922540efb8690cd80921db913742e22be117c4ca7.jpg)

![](images/90e8b8a73bd3c17c94ae2441c6d34919b4e80b663d27fc0b93ba6d2fdff7f686.jpg)  
"shark"

![](images/09ebbc89d3a530e6ab0a836b6dbd043e45ddbea1f4c0e3e5fe413219eed3c7b5.jpg)  
Support Set  
Unlabeled Set  
Figure 1: Consider a setup where the aim is to learn a classifier to distinguish between two previously unseen classes, goldfish and shark, given not only labeled examples of these two classes, but also a larger pool of unlabeled examples, some of which may belong to one of these two classes of interest. In this work we aim to move a step closer to this more natural learning framework by incorporating in our learning episodes unlabeled data from the classes we aim to learn representations for (shown with dashed red borders) as well as from distractor classes.

This work is a first study of this challenging semi-supervised form of few-shot learning. First, we define the problem and propose benchmarks for evaluation that are adapted from the Omniglot and miniImageNet benchmarks used in ordinary few-shot learning. We perform an extensive empirical investigation of the two settings mentioned above, with and without distractor classes. Second, we propose and study three novel extensions of Prototypical Networks (Snell et al., 2017), a state-of-the-art approach to few-shot learning, to the semi-supervised setting. Finally, we demonstrate in our experiments that our semi-supervised variants successfully learn to leverage unlabeled examples and outperform purely supervised Prototypical Networks.

# 2 BACKGROUND

We start by defining precisely the current paradigm for few-shot learning and the Prototypical Network approach to this problem.

# 2.1 FEW-SHOT LEARNING

Recent progress on few-shot learning has been made possible by following an episodic paradigm for few-shot learning. Consider a situation where we have a large labeled dataset for a set of classes  $\mathcal{C}_{\mathrm{train}}$ . However, after training on examples from  $\mathcal{C}_{\mathrm{train}}$ , our ultimate goal is to produce classifiers for a disjoint set of new classes  $\mathcal{C}_{\mathrm{test}}$ , for which only a few labeled examples will be available. The idea behind the episodic paradigm is to simulate the types of few-shot problems that will be encountered at test using the large quantities of available labeled data for classes  $\mathcal{C}_{\mathrm{train}}$ .

Specifically, models are trained on  $K$ -shot,  $N$ -way episodes constructed by first sampling a small subset of  $N$  classes from  $\mathcal{C}_{\mathrm{train}}$  and then generating: 1) a training (support) set  $S = \{(x_1, y_1), (x_2, y_2), \ldots, (x_{N \times K}, y_{N \times K})\}$  containing  $K$  examples from each of the  $N$  classes and 2) a test (query) set  $\mathcal{Q} = \{(x_1^*, y_1^*), (x_2^*, y_2^*), \ldots, (x_T^*, y_T^*)\}$  of different examples from the same  $N$  classes. Each  $x_i \in \mathbb{R}^D$  is an input vector of dimension  $D$  and  $y_i \in \{1, 2, \ldots, N\}$  is a class label (similarly for  $x_i^*$  and  $y_i^*$ ). Training on such episodes is done by feeding the support set  $S$  to the model and updating its parameters to minimize the loss of its predictions for the examples in the query set  $\mathcal{Q}$ .

One way to think of this approach is that our model effectively trains to be a good learning algorithm. Indeed, much like a learning algorithm, the model must take in a set of labeled examples and produce a predictor that can be applied to new examples. Moreover, training directly encourages the classifier

produced by the model to have good generalization on the new examples of the query set. Due to this analogy, training under this paradigm is often referred to as learning to learn or meta-learning.

On the other hand, referring to the content of episodes as training and test sets and to the process of learning on these episodes as meta-learning or meta-training (as is sometimes done in the literature) can be confusing. So for the sake of clarity, we will refer to the content of episodes as support and query sets, and to the process of iterating over the training episodes simply as training.

# 2.2 PROTOTYPICAL NETWORKS

Prototypical Network (Snell et al., 2017) is a few-shot learning model that has the virtue both of being simple while obtaining state-of-the-art performance. At a high-level, it uses the support set  $S$  to extract a prototype vector from each class, and classifies the inputs in the query set based on their distance to the prototype of each class.

More precisely, Prototypical Networks learn an embedding function  $h(x)$ , parameterized as a neural network, that maps examples into a space where examples from the same class are close and those from different classes are far. All parameters of Prototypical Networks lie in the embedding function.

To compute the prototype  $\pmb{p}_c$  of each class  $c$ , a per-class average of the embedded examples is performed:

$$
\boldsymbol {p} _ {c} = \frac {\sum_ {i} h (\boldsymbol {x} _ {i}) z _ {i , c}}{\sum_ {i} z _ {i , c}}, \text {w h e r e} z _ {i, c} = \mathbb {1} [ y _ {i} = c ]. \tag {1}
$$

These prototypes define a predictor for the class of any new (query) example  $x^{*}$ , which assigns a probability over any class  $c$  based on the distances between  $x^{*}$  and each prototype, as follows:

$$
p \left(c \mid \boldsymbol {x} ^ {*}, \left\{\boldsymbol {p} _ {c} \right\}\right) = \frac {\exp \left(- \left\| h \left(\boldsymbol {x} ^ {*}\right) - \boldsymbol {p} _ {c} \right\| _ {2} ^ {2}\right)}{\sum_ {c ^ {\prime}} \exp \left(- \left\| h \left(\boldsymbol {x} ^ {*}\right) - \boldsymbol {p} _ {c ^ {\prime}} \right\| _ {2} ^ {2}\right)}. \tag {2}
$$

The loss function used to update Prototypical Networks for a given training episode is then simply the average negative log-probability of the correct class assignments, for all query examples:

$$
- \frac {1}{T} \sum_ {i} \log p \left(y _ {i} ^ {*} \mid \boldsymbol {x} _ {i} ^ {*}, \left\{\boldsymbol {p} _ {c} \right\}\right). \tag {3}
$$

Training proceeds by minimizing the average loss, iterating over training episodes and performing a gradient descent update for each.

Generalization performance is measured on test set episodes, which contain images from classes in  $\mathcal{C}_{\mathrm{test}}$  instead of  $\mathcal{C}_{\mathrm{train}}$ . For each test episode, we use the predictor produced by the Prototypical Network for the provided support set  $S$  to classify each of query input  $\pmb{x}^*$  into the most likely class  $\hat{y} = \operatorname{argmax}_c p(c|\pmb{x}^*, \{\pmb{p}_c\})$ .

# 3 SEMI-SUPERVISED FEW-SHOT LEARNING

We now move to defining the semi-supervised setting considered in this work for few-shot learning.

We denote our training set as a tuple of labeled and unlabeled examples:  $(S, \mathcal{R})$ . The labeled portion is the usual support set  $S$  of the few-shot learning literature, containing a list of tuples of inputs and targets. In addition to classic few-shot learning, we introduce an unlabeled set  $\mathcal{R}$  containing only inputs:  $\mathcal{R} = \{\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_M\}$ . As in the purely supervised setting, our models are trained to perform well when predicting the labels for the examples in the episode's query set  $\mathcal{Q}$ . Figure 2 shows a visualization of training and test episodes.

# 3.1 SEMI-SUPERVISED PROTOTYPICAL NETWORKS

In their original formulation, Prototypical Networks do not specify a way to leverage the unlabeled set  $\mathcal{R}$ . In what follows, we now propose various extensions that start from the basic definition of prototypes  $\pmb{p}_c$  and provide a procedure for producing refined prototypes  $\tilde{\pmb{p}}_c$  using the unlabeled examples in  $\mathcal{R}$ .

![](images/17404b6336f5cd33b75c8dbfb4bded3b621cb2b69b06459eadf0f9e8be9e392c.jpg)  
Figure 2: Example of the semi-supervised few-shot learning setup. Training involves iterating through training episodes, consisting of a support set  $S$ , an unlabeled set  $\mathcal{R}$ , and a query set  $\mathcal{Q}$ . The goal is to use the labeled items (shown with their numeric label) in  $S$  and the unlabeled items in  $\mathcal{R}$  within each episode to generalize to good performance on the corresponding query set. The unlabeled items in  $\mathcal{R}$  may either be pertinent to the classes we are considering (shown above with green plus signs) or they may be distractor items which belong to a class that is not relevant to the current episode (shown with red minus signs). However note that the model does not actually have ground truth information as to whether each unlabeled example is a distractor or not; the plus/minus signs are shown only for illustrative purposes. At test time, we are given new episodes consisting of new classes not seen during training that we use to evaluate the meta-learning method.

After the refined prototypes are obtained, each of these models is trained with the same loss function for ordinary Prototypical Networks of Equation 3, but replacing  $p_c$  with  $\tilde{p}_c$ . That is, each query example is classified into one of the  $N$  classes based on the closeness of its embedded position with the corresponding refined prototypes, and the average negative log-probability of the correct classification is used for training.

![](images/4ae9fcce296a837321e1a612cd9481acf5bf69bfb1748a8368c321ba8893c53a.jpg)  
Figure 3: Left: The initialization of the prototypes to be the average of the examples of the corresponding class, as in ordinary Prototypical Networks. Support examples have solid colored borders, unlabeled examples have dashed borders, and query examples have white borders. Right: The refined prototypes obtained by incorporating the unlabeled examples. After refinement, all query examples are correctly classified.

![](images/5562f07e785f1d0ed98061ab66f361e8a53494715b037dffb11bc2054ab8fa52.jpg)

# 3.1.1 PROTOTYPICAL NETWORKS WITH SOFT  $k$ -MEANS

We first consider a simple way of leveraging unlabeled examples for refining prototypes, by taking inspiration from semi-supervised clustering. Viewing each prototype as a cluster center, the refinement process could attempt to adjust the cluster locations to better fit the examples in both the support and unlabeled sets. Under this view, cluster assignments of the labeled examples in the support set are considered known and fixed to each example's label. The refinement process must instead estimate the cluster assignments of the unlabeled example and adjust the cluster locations (the prototypes) accordingly.

One natural choice would be to borrow from the inference performed by soft  $k$ -means. We prefer this version of  $k$ -means over hard assignments since hard assignments would make the inference non-differentiable. We start from the regular Prototypical Network's prototypes  $\pmb{p}_c$  (as specified in Equation 1) as the cluster locations. Then, the unlabeled examples get a partial assignment  $(\tilde{z}_{j,c})$  to each cluster based on their Euclidean distance to the cluster locations. Finally, refined prototypes are obtained by incorporating these unlabeled examples.

This process can be summarized as follows:

$$
\tilde {\boldsymbol {p}} _ {c} = \frac {\sum_ {i} h (\boldsymbol {x} _ {i}) z _ {i , c} + \sum_ {j} h (\tilde {\boldsymbol {x}} _ {j}) \tilde {z} _ {j , c}}{\sum_ {i} z _ {i , c} + \sum_ {j} \tilde {z} _ {j , c}}, \text {w h e r e} \tilde {z} _ {j, c} = \frac {\exp \left(- | | h (\tilde {\boldsymbol {x}} _ {j}) - \boldsymbol {p} _ {c} | | _ {2} ^ {2}\right)}{\sum_ {c ^ {\prime}} \exp \left(- | | h (\tilde {\boldsymbol {x}} _ {j}) - \boldsymbol {p} _ {c ^ {\prime}} | | _ {2} ^ {2}\right)} \tag {4}
$$

Predictions of each query input's class is then modeled as in Equation 2, but using the refined prototypes  $\tilde{p}_c$ .

We could perform several iterations of refinement, as is usual in  $k$ -means. However, we have experimented with various number of iterations and found results to not improve beyond a single refinement step.

# 3.1.2 PROTOTYPICAL NETWORKS WITH SOFT  $k$ -MEANS WITH A DISTRACTOR CLUSTER

The soft  $k$ -means approach described above implicitly assumes that each unlabeled example belongs to either one of the  $N$  classes in the episode. However, it would be much more general to not make that assumption and have a model robust to the existence of examples from other classes, which we refer to as distractor classes. For example, such a situation would arise if we wanted to classify between pictures of unicycles and scooters, and decided to add an unlabeled set by downloading images from the web. It then would not be realistic to assume that all these images are of unicycles or scooters. Even with a focused search, some may be from similar categories, such as bicycle.

Since soft  $k$ -means distributes its soft assignments across all classes, distractor items could be harmful and interfere with the refinement process, as prototypes would be adjusted to also partially account for these distractors. A simple way to address this is to add an additional cluster whose purpose is to capture the distractors, thus preventing them from polluting the clusters of the classes of interest:

$$
\boldsymbol {p} _ {c} = \left\{ \begin{array}{l l} \frac {\sum_ {i} h (\boldsymbol {x} _ {i}) z _ {i , c}}{\sum_ {i} z _ {i , c}} & \text {f o r} c = 1 \dots N \\ \boldsymbol {0} & \text {f o r} c = N + 1 \end{array} \right. \tag {5}
$$

Here we take the simplifying assumption that the distractor cluster has a prototype centered at the origin. We also consider introducing length-scales  $r_c$  to represent variations in the within-cluster distances, specifically for the distractor cluster:

$$
\tilde {z} _ {j, c} = \frac {\exp \left(- \frac {1}{r _ {c} ^ {2}} \left\| \tilde {\boldsymbol {x}} _ {j} - \boldsymbol {p} _ {c} \right\| _ {2} ^ {2} - A \left(r _ {c}\right)\right)}{\sum_ {c ^ {\prime}} \exp \left(- \frac {1}{r _ {c} ^ {2}} \left\| \tilde {\boldsymbol {x}} _ {j} - \boldsymbol {p} _ {c ^ {\prime}} \right\| _ {2} ^ {2} - A \left(r _ {c ^ {\prime}}\right)\right)}, \text {w h e r e} A (r) = \frac {1}{2} \log (2 \pi) + \log (r) \tag {6}
$$

For simplicity, we set  $r_{1\dots N}$  to 1 in our experiments, and only learn the length-scale of the distractor cluster  $r_{N + 1}$ .

# 3.1.3 PROTOTYPICAL NETWORKS WITH SOFT  $k$ -MEANS AND MASKING

Modeling distractor unlabeled examples with a single cluster is likely too simplistic. Indeed, it is inconsistent with our assumption that each cluster corresponds to one class, since distractor examples may very well cover more than a single natural object category. Continuing with our unicycles and bicycles example, our web search for unlabeled images could accidentally include not only bicycles, but other related objects such as tricycles or cars. This was also reflected in our experiments, where we constructed the episode generating process so that it would sample distractor examples from multiple classes.

To address this problem, we propose an improved variant: instead of capturing distractors with a high-variance catch-all cluster, we model distractors as examples that are not within some area of any of the legitimate class prototypes. This is done by incorporating a soft-masking mechanism on the contribution of unlabeled examples. At a high level, we want unlabeled examples that are closer to a prototype to be masked less than those that are farther.

More specifically, we modify the soft  $k$ -means refinement as follows. We start by computing normalized distances  $\tilde{d}_{j,c}$  between examples  $\tilde{\pmb{x}}_j$  and prototypes  $\pmb{p}_c$ :

$$
\tilde {d} _ {j, c} = \frac {d _ {j , c}}{\frac {1}{M} \sum_ {j} d _ {j , c}}, \text {w h e r e} d _ {j, c} = \left\| h \left(\tilde {\boldsymbol {x}} _ {j}\right) - \boldsymbol {p} _ {c} \right\| _ {2} ^ {2} \tag {7}
$$

Then, soft thresholds  $\beta_{c}$  and slopes  $\gamma_{c}$  are predicted for each prototype, by feeding to a small neural network various statistics of the normalized distances for the prototype:

$$
\left[ \beta_ {c}, \gamma_ {c} \right] = \operatorname {M L P} \left(\left[ \min  _ {j} \left(\tilde {d} _ {j, c}\right), \max  _ {j} \left(\tilde {d} _ {j, c}\right), \operatorname {v a r} _ {j} \left(\tilde {d} _ {j, c}\right), \operatorname {s k e w} _ {j} \left(\tilde {d} _ {j, c}\right), \operatorname {k u r t} _ {j} \left(\tilde {d} _ {j, c}\right) \right]\right) \tag {8}
$$

This allows each threshold to use information on the amount of intra-cluster variation to determine how aggressively it should cut out unlabeled examples.

Then, soft masks  $m_{j,c}$  for the contribution of each example to each prototype are computed, by comparing to the threshold the normalized distances, as follows:

$$
\tilde {\boldsymbol {p}} _ {c} = \frac {\sum_ {i} h (\boldsymbol {x} _ {i}) z _ {i , c} + \sum_ {j} h \left(\tilde {\boldsymbol {x}} _ {j}\right) \tilde {z} _ {j , c} m _ {j , c}}{\sum_ {i} z _ {i , c} + \sum_ {j} \tilde {z} _ {j , c} m _ {j , c}}, \text {w h e r e} m _ {j, c} = \sigma \left(\gamma_ {c} \left(\tilde {d} _ {j, c} - \beta_ {c}\right)\right) \tag {9}
$$

where  $\sigma (\cdot)$  is the sigmoid function.

When training with this refinement process, the model can now use its MLP in Equation 8 to learn to include or ignore entirely certain unlabeled examples. The use of soft masks makes this process entirely differentiable<sup>2</sup>. Finally, much like for regular soft  $k$ -means (with or without a distractor cluster), while we could recursively repeat the refinement for multiple steps, we've found a single step to perform well enough.

# 4 RELATED WORK

We summarize here the most relevant work from the literature on few-shot learning, semi-supervised learning and clustering.

The best performing methods for few-shot learning use the episodic training framework prescribed by meta-learning. The approach within which our work falls is that of metric learning methods. Previous work in metric-learning for few-shot-classification includes Deep Siamese Networks (Koch et al., 2015), Matching Networks (Vinyals et al., 2016), and Prototypical Networks (Snell et al., 2017), which is the model we extend to the semi-supervised setting in our work. The general idea here is to learn an embedding function that embeds examples belonging to the same class close together while keeping embeddings from separate classes far apart. Distances between embeddings of items from the support set and query set are then used as a notion of similarity to do classification. Lastly, closely related to our work with regard to extending the few-shot learning setting, Bachman et al. (2017) employ Matching Networks in an active learning framework where the model has a choice of which unlabeled item to add to the support set over a certain number of time steps before classifying the query set. Unlike our setting, their meta-learning agent can acquire ground-truth labels from the unlabeled set, and they do not use distractor examples.

Other meta-learning approaches to few-shot learning include learning how to use the support set to update a learner model so as to generalize to the query set. Recent work has involved learning either the weight initialization and/or update step that is used by a learner neural network (Ravi & Larochelle, 2017; Finn et al., 2017). Another approach is to train a generic neural architecture such as a memory-augmented recurrent network (Santoro et al., 2016) or a temporal convolutional network (Mishra et al., 2017) to sequentially process the support set and perform accurate predictions of the labels of the query set examples. These other methods are also competitive for few-shot learning, but we chose to extend Prototypical Networks in this work for its simplicity and efficiency.

As for the literature on semi-supervised learning, while it is quite vast (Zhu, 2005; Chapelle et al., 2010), the most relevant category to our work is related to self-training (Yarowsky, 1995; Rosenberg et al., 2005). Here, a classifier is first trained on the initial training set. The classifier is then used to classify unlabeled items, and the most confidently predicted unlabeled items are added to the training set with the prediction of the classifier as the assumed label. This is similar to our soft  $k$ -means extension to Prototypical Networks. Indeed, since the soft assignments (Equation 4) match the regular Prototypical Network's classifier output for new inputs (Equation 2), then the refinement

can be thought of re-feeding to a Prototypical Network a new support set augmented with (soft) self-labels from the unlabeled set.

In addition to the original  $k$ -means method (Lloyd, 1982), the most related work to our setup involving clustering algorithms considers applying  $k$ -means in the presence of outliers (Hautamaki et al., 2005; Chawla & Gionis, 2013; Gupta et al., 2017). The goal here is to correctly discover and ignore the outliers so that they do not wrongly shift the cluster locations to form a bad partition of the true data. This objective is also important in our setup as not ignoring outliers (or distractors) will wrongly shift the prototypes and negatively influence classification performance.

Our contribution to the semi-supervised learning and clustering literature is to go beyond the classical setting of training and evaluating within a single dataset, and consider the setting where we must learn to transfer from a set of training classes  $\mathcal{C}_{\mathrm{train}}$  to a new set of test classes  $\mathcal{C}_{\mathrm{test}}$ .

# 5 EXPERIMENTS

# 5.1 DATASETS

We evaluate the performance of our model on three datasets: two benchmark few-shot classification datasets and a novel large-scale dataset that we hope will be useful for future few-shot learning work.

Omniglot (Lake et al., 2011) is a dataset of 1,623 handwritten characters from 50 alphabets. Each character was drawn by 20 human subjects. We follow the few-shot setting proposed by Vinyals et al. (2016), in which the images are resized to  $28 \times 28$  pixels and rotations in multiples of  $90^{\circ}$  are applied, yielding 6,492 classes in total. These are split into 4,800 training classes and 1,692 classes for test.

miniImageNet (Vinyals et al., 2016) is a modified version of the ILSVRC-12 dataset (Russakovsky et al., 2015), in which 600 images for each of 100 classes were randomly chosen to be part of the dataset. We rely on the class split used by Ravi & Larochelle (2017). These splits use 64 classes as training, 16 for validation, and 20 for test. All images are of size  $84 \times 84$  pixels.

tieredImageNet is our proposed dataset for few-shot classification. Like miniImagenet, it is a subset of ILSVRC-12. However, tieredImageNet represents a larger subset of ILSVRC-12 (608 classes rather than 100 for miniImageNet). Analogous to Omniglot, in which characters are grouped into alphabets, tieredImageNet groups classes into broader categories corresponding to higher-level nodes in the ImageNet (Deng et al., 2009) hierarchy. There are 34 categories in total, with each category containing between 10 and 30 classes. These are split into 26 training categories and 8 testing categories (details of the dataset can be found in the supplementary material). This ensures that all of the training classes are sufficiently distinct from the testing classes, unlike miniImageNet and other alternatives such as randImageNet proposed by Vinyals et al. (2016). For example, "pipe organ" is a training class and "electric guitar" is a test class in the Ravi & Larochelle (2017) split of miniImagenet, even though they are both musical instruments. This scenario would not occur in tieredImageNet since "musical instrument" is a high-level category and as such is not split between training and test classes. This represents a more realistic few-shot learning scenario since in general we cannot assume that test classes will be similar to those seen in training. Additionally, the tiered structure of tieredImageNet may be useful for few-shot learning approaches that can take advantage of hierarchical relationships between classes. We leave such interesting extensions for future work.

# 5.2 EXPERIMENTAL SETUP

For each dataset, we first create an additional split to separate the images of each class into disjoint labeled and unlabeled sets. For Omniglot and tieredImageNet we sampled  $10\%$  of the images of each class to form the labeled split. The remaining  $90\%$  can only be used in the unlabeled portion of episodes. For miniImageNet we instead used  $40\%$  of the data for the labeled split and the remaining  $60\%$  for the unlabeled, since we noticed that  $10\%$  was too small to achieve reasonable performance and avoid overfitting.

We would like to emphasize that due to this labeled/unlabeled split, we are using strictly less label information than in the previously-published work on these datasets. Because of this, we do not

<table><tr><td>ProtoNet Model</td><td>Err.</td><td>Err. w/D</td></tr><tr><td>Supervised</td><td>5.16%</td><td>5.16%</td></tr><tr><td>Semi-Supervised Inference</td><td>2.35%</td><td>4.70%</td></tr><tr><td>Soft k-Means</td><td>2.56%</td><td>4.59%</td></tr><tr><td>Soft k-Means+Cluster</td><td>2.18%</td><td>2.71%</td></tr><tr><td>Masked Soft k-Means</td><td>2.46%</td><td>2.62%</td></tr></table>

Table 1: Omniglot 1-shot Results  

<table><tr><td>ProtoNet Model</td><td>1-shot Acc.</td><td>5-shot Acc.</td><td>1-shot Acc w/ D</td><td>5-shot Acc. w/ D</td></tr><tr><td>Supervised</td><td>43.36%</td><td>59.03%</td><td>43.36%</td><td>59.03%</td></tr><tr><td>Semi-Supervised Inference</td><td>48.68%</td><td>62.94%</td><td>46.16%</td><td>62.32%</td></tr><tr><td>Soft k-Means</td><td>48.25%</td><td>65.72%</td><td>46.72%</td><td>61.94%</td></tr><tr><td>Soft k-Means+Cluster</td><td>50.87%</td><td>63.75%</td><td>48.60%</td><td>61.51%</td></tr><tr><td>Masked Soft k-Means</td><td>50.57%</td><td>63.78%</td><td>50.04%</td><td>62.50%</td></tr></table>

Table 2: miniImageNet 1/5-shot Results

expect our results to match the published numbers, which should instead be interpreted as an upper-bound for the performance of the semi-supervised models defined in this work.

Episode construction then is performed as follows. For a given dataset, we create a training episode by first sampling  $N$  classes uniformly at random from the set of training classes  $\mathcal{C}_{\mathrm{train}}$ . We then sample  $K$  images from the labeled split of each of these classes to form the support set, and  $M$  images from the unlabeled split of each of these classes to form the unlabeled set. Optionally, when including distractors, we additionally sample  $H$  other classes from the set of training classes and  $M$  images from the unlabeled split of each to act as the distractors. These distractor images are added to the unlabeled set along with the unlabeled images of the  $N$  classes of interest (for a total of  $MN + MH$  images). The query portion of the episode is comprised of a fixed number of images from the labeled split of each of the  $N$  chosen classes. Test episodes are created analogously, but with the  $N$  classes (and optionally the  $H$  distractor classes) sampled from  $\mathcal{C}_{\mathrm{test}}$ . Note that we used  $M = 5$  for training and  $M = 20$  for testing, thus also measuring the ability of the models to generalize to a larger unlabeled set size. We also used  $H = N = 5$ , i.e. used 5 classes for both the labeled classes and the distractor classes.

In each dataset we compare our three semi-supervised models with two baselines. The first baseline, referred to as Supervised in our tables, is an ordinary Prototypical Network that is trained in a purely supervised way on the labeled split of each dataset. The second baseline, referred to as Semi-Supervised Inference, uses the embedding function learned by this supervised Prototypical Network, but performs semi-supervised refinement of the prototypes at inference time using a step of Soft  $k$ -Means refinement. This is to be contrasted with our semi-supervised models that perform this refinement both at training time and at test time, therefore learning a different embedding function. We evaluate each model in two settings: one where all unlabeled examples belong to the classes of interest, and a more challenging one that includes distractors. Details of the model hyperparameters can be found in the appendix.

# 5.3 RESULTS

Results for Omniglot, miniImageNet and tieredImageNet are given in Tables 1, 2 and 3, respectively, while Figure 4 shows the performance of our models on tieredImageNet (our largest dataset) using different values for  $M$  (number of items in the unlabeled set per class).

Across all three benchmarks, at least one of our proposed models outperform the baselines, demonstrating the effectiveness of our semi-supervised meta-learning procedure. In particular, Soft  $k$ -means+Cluster performs the best on 1-shot non-distractor settings, as the extra cluster seems to provide a form of regularization that pushes the clusters farther apart. Soft  $k$ -Means performs well on 5-shot non-distractor settings, as it considers the most unlabeled examples. Masked Soft  $k$ -Means shows the most robust performance in distractors settings, in both 1-shot and 5-shot tasks. For 5-

<table><tr><td>ProtoNet Model</td><td>1-shot Acc.</td><td>5-shot Acc.</td><td>1-shot Acc. w/ D</td><td>5-shot Acc. w/ D</td></tr><tr><td>Supervised</td><td>46.60%</td><td>67.18%</td><td>46.60%</td><td>67.18%</td></tr><tr><td>Semi-Supervised Inference</td><td>50.38%</td><td>70.26%</td><td>46.87%</td><td>68.38%</td></tr><tr><td>Soft k-Means</td><td>53.41%</td><td>71.31%</td><td>50.18%</td><td>68.83%</td></tr><tr><td>Soft k-Means+Cluster</td><td>55.82%</td><td>70.79%</td><td>49.87%</td><td>70.16%</td></tr><tr><td>Masked Soft k-Means</td><td>52.76%</td><td>70.08%</td><td>50.93%</td><td>71.00%</td></tr></table>

Table 3: tieredImageNet 1/5-shot Results

![](images/64a4b5dd73b0227f86745e7a45fc4d5ca4fb4bd5458ddca2d828632fb42f4502.jpg)

![](images/d522ba7cbaa6e4d8e0dd4c6f5d871149dc76ec8ad750f8b1efaee840eb4ee004.jpg)

![](images/ed757059cb6a55c6841ecef6a5c24b9c95316fd1480885725ac2794188d033e0.jpg)  
Figure 4: Model Performance on tieredImageNet with different number of unlabeled items during test time.

![](images/7fc49809f69a000efd452e2cb8c2151fea193744b2fe358b77a82f0f3856dca3.jpg)

shot, Masked soft  $k$ -Means reaches comparable performance compared to the upper bound of the best non-distractor performance.

From Figure 4, we observe clear improvement in test accuracy when the number grows from 0 to 25. Note that our models were trained with  $M = 5$  and thus are showing an ability to extrapolate in generalization. This confirms that, through meta-training, the models learned to acquire a better representation that will be more helpful after semi-supervised refinement.

Note that the wins obtained in our semi-supervised learning are super-additive. Consider the case of the simple k-Means model on 1-shot without Distractors. Training only on labeled examples while incorporating the unlabeled set during test time produces an advantage of  $3.8\%$  (50.4-46.6), while incorporating the unlabeled set during training but not during test produces a win of  $1.1\%$  (47.7-46.6). Incorporating unlabeled examples during both training and test yields a win of  $6.8\%$  (53.4-46.6).

# 6 CONCLUSION

In this work, we propose a novel semi-supervised few-shot learning paradigm, where an unlabeled set is added to each episode. We also extend the setup to more realistic situations where the unlabeled set has classes not belonging to the labeled classes. To address the problem that current few-shot classification dataset is too small for a labeled vs. unlabeled split and does not have hierarchical levels of labels, we also introduce a new dataset, tieredImageNet. We propose several novel extensions of Prototypical Networks, and they show consistent improvements under semi-supervised settings compared to our baselines. As future work, we are working on incorporating fast weights (Ba et al., 2016; Finn et al., 2017) into our framework so that examples can have different embedding representation given the contents in the episode.

# REFERENCES

Jimmy Ba, Geoffrey E. Hinton, Volodymyr Mnih, Joel Z. Leibo, and Catalin Ionescu. Using fast weights to attend to the recent past. In Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain, pp. 4331-4339, 2016.  
Philip Bachman, Alessandro Sordoni, and Adam Trischler. Learning algorithms for active learning. 2017.  
Olivier Chapelle, Bernhard Scholkopf, and Alexander Zien. Semi-Supervised Learning. The MIT Press, 1st edition, 2010. ISBN 0262514125, 9780262514125.  
Sanjay Chawla and Aristides Gionis. k-means-: A unified approach to clustering and outlier detection. In Proceedings of the 2013 SIAM International Conference on Data Mining, pp. 189-197. SIAM, 2013.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. IEEE, 2009.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In 34th International Conference on Machine Learning, 2017.  
Shalmoli Gupta, Ravi Kumar, Kefu Lu, Benjamin Moseley, and Sergei Vassilvitskii. Local search methods for k-means with outliers. Proceedings of the VLDB Endowment, 10(7):757-768, 2017.  
Ville Hautamäki, Svetlana Cherednichenko, Ismo Kärkkäinen, Tomi Kinnunen, and Pasi Franti. Improving k-means by outlier removal. In Scandinavian Conference on Image Analysis, pp. 978-987. Springer, 2005.  
Sepp Hochreiter, A Steven Younger, and Peter R Conwell. Learning to learn using gradient descent. In International Conference on Artificial Neural Networks, pp. 87-94. Springer, 2001.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese neural networks for one-shot image recognition. In ICML Deep Learning Workshop, volume 2, 2015.  
Brenden M. Lake, Ruslan Salakhutdinov, Jason Gross, and Joshua B. Tenenbaum. One shot learning of simple visual concepts. In Proceedings of the 33th Annual Meeting of the Cognitive Science Society, CogSci 2011, Boston, Massachusetts, USA, July 20-23, 2011, 2011.  
Stuart Lloyd. Least squares quantization in pmc. IEEE transactions on information theory, 28(2): 129-137, 1982.  
Nikhil Mishra, Mostafa Rohaninejad, Xi Chen, and Pieter Abbeel. Meta-learning with temporal convolutions. CoRR, abs/1707.03141, 2017. URL http://arxiv.org/abs/1707.03141.  
Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In 5th International Conference on Learning Representations, 2017.  
Chuck Rosenberg, Martial Hebert, and Henry Schneiderman. Semi-supervised self-training of object detection models. 2005.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115(3):211-252, 2015.  
Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy P. Lillicrap. One-shot learning with memory-augmented neural networks. In 33rd International Conference on Machine Learning, 2016.

Jake Snell, Kevin Swersky, and Richard S. Zemel. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems 30, 2017.  
Sebastian Thrun. Lifelong learning algorithms. In Learning to learn, pp. 181-209. Springer, 1998.  
Oriol Vinyals, Charles Blundell, Tim Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. Matching networks for one shot learning. In Advances in Neural Information Processing Systems 29, pp. 3630-3638, 2016.  
David Yarowsky. Unsupervised word sense disambiguation rivaling supervised methods. In Proceedings of the 33rd annual meeting on Association for Computational Linguistics, pp. 189-196. Association for Computational Linguistics, 1995.  
Xiaojin Zhu. Semi-supervised learning literature survey. 2005.
