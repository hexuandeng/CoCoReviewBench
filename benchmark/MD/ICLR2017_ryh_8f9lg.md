# CLASSLESS ASSOCIATION USING NEURAL NETWORKS

Federico Raue<sup>1,2</sup>, Sebastian Palacio<sup>2</sup>, Andreas Dengel<sup>1,2</sup>, Marcus Liwicki<sup>1</sup>

<sup>1</sup>University of Kaiserslautern, Germany  
$^{2}$ German Research Center for Artificial Intelligence (DFKI), Germany.

{federico.raue,sebastian.palacio,andreas.dengel}@dfki.de, liwicki@cs.uni-kl.de

# ABSTRACT

The goal of this paper is to train a model based on the relation between two instances that represent the same unknown class. We propose a novel model called Classless Association. It has two parallel Multilayer Perceptrons (MLP) that uses one network as a target of the other network, and vice versa. In addition, the presented model is trained based on an EM-approach, in which the output vectors are matched against a statistical distribution. We generate a dataset based on MNIST, where the input is two different instances of the same digit and the dataset has a uniform distribution of the digits. Furthermore, our classless association model is evaluated against two scenarios: totally supervised and totally unsupervised. In the first scenario, our model reaches a good performance in terms of accuracy and the classless data. In the second scenario, our model reaches better results against two clustering algorithms.

# 1 INTRODUCTION

Parallel information is available in several scenarios where unlabeled data is collected at the same time, for instance, multiple sensors, stereo cameras. With this in mind, the relation that appears in parallel samples can be exploited for training models. Furthermore, learning the association between them can be found in infants development, where abstract concepts are associated to its different representation formats, i.e., visual and auditory (Balaban & Waxman, 1997; Gershkoff-Stowe & Smith, 2004). For example, the abstract concept ball can be associated to a round object (visual representation) and the phonemes of ball (auditory representation).

In addition, we are interested in training without classes. One way is to train based on statistical distributions as a loss function. Casey (1986) proposed to solve the OCR problem using language statistics for inferring form images to characters. Later on, Knight et al. (2006) applied a similar idea to machine translation. Hsu & Kira (2015) developed a neural network that clusters elements based on pair-wise constraints. Recently, Sutskever et al. (2015) defined the Output Distribution Matching (ODM) cost function for dual autoencoders and generative networks.

In this paper, we are proposing a novel model that is trained based on the association of two input samples of the same unknown class. Figure 1 shows an example of the difference between a supervised association task and our scenario. The presented model has two parallel Multilayer Perceptrons (MLPs) with an Expectation-Maximization (EM) (Dempster et al., 1977) training rule that matches the network output against a statistical distribution. Also, both networks agree on the same classification because one network is used as a target of the other network, and vice versa. Our model has some similarities with Siamese Networks proposed by Chopra et al. (2005). They introduced their model for supervised face verification where training is based on constraints of pairs of faces. The constraints exploit the relation of two faces that may or may not be instances of the same person. However, there are some differences to our work. First, our training rule does not have pre-defined classes before training, whereas the Siamese Network requires labeled classes. Second, our model only requires instances of the same unknown class, whereas the Siamese network requires two types of input pairs: a) instances of the same person and b) instances of two different persons. Our contributions in this paper are

![](images/ece01f8d6837cf7882daf53dddf6621611f93f916523d03d6a290b19e09f9ba9.jpg)  
Figure 1: Difference between the supervised and classless association tasks. The classless association is more challenging than the supervised association because the model requires to learn to discriminate the semantic concept without labels. In addition, both classifiers need to agree on the same coding scheme for each semantic concept. In contrast, the mentioned information is already known in the supervised association scenario.

- We define a novel training rule based on matching the output vectors of the presented model and a statistical distribution. Furthermore, the proposed training rule is based on an EM-approach and creates pseudo-classes converging in order to define the input samples (Section 2.1).  
- We propose a novel architecture for learning the association in a classless scenario. Moreover, the presented model uses two parallel MLPs that learn from each other. In more detail, one network is the target of the other network, and vice versa. Also, note that our model is gradient-based and can be extended to deeper architectures (Section 2.2).  
- We evaluate our classless association task against two cases: totally supervised and totally unsupervised. In this manner, we can verify the range of our results in terms of supervised and unsupervised case since our model is not totally supervised and not totally unsupervised. We compare against a MLP trained with labels as the supervised scenario (upper bound) and two clustering algorithms (K-means and Hierarchical Agglomerative) as the unsupervised scenario (lower bound). First, our model reaches better results than the clustering. Second, our model shows promising results with respect to the supervised scenario (Sections 3 and 4).

# 2 METHODOLOGY

In this paper, we are interested in the classless association task in the following scenario: two input instances  $\pmb{x}^{(1)}$  and  $\pmb{x}^{(2)}$  belong to the same unknown class  $c$ , where  $\pmb{x}^{(1)} \in \pmb{X}^{(1)}$  and  $\pmb{x}^{(2)} \in \pmb{X}^{(2)}$  are two disjoint sets. With this in mind, we present a model that has two parallel Multilayer Perceptrons (MLPs) that are trained with an EM-approach that associates both networks in the following manner: one network uses the other network as a target, and vice versa. First, we explain how the output vectors of the network are matched to a statistical distribution (see Section 2.1). Second, the classless association learning is presented in Section 2.2.

# 2.1 STATISTICALCONSTRAINT

The presented model is trained based on a loss function that matches the output vectors with a statistical distribution for MLP training. For explanation purposes, we applied our training rule to a single MLP with one hidden layer, which is defined by

$$
\boldsymbol {z} = \operatorname {n e t w o r k} (\boldsymbol {x}; \boldsymbol {\theta}) \tag {1}
$$

where  $\pmb{x} \in \mathbb{R}^n$  is the input vector,  $\pmb{\theta}$  encodes the parameters of the MLP, and  $\pmb{z} \in \mathbb{R}^c$  is the output vector. Moreover, the output vectors  $(\pmb{z}_1, \dots, \pmb{z}_m)$  of a mini-batch of size  $m$  are matched to a target distribution  $(\mathbb{E}[z_1, \dots, z_m] \sim \phi \in \mathbb{R}^c)$ , e.g., uniform distribution. Thus, we introduce a new parameter that is weighting vector  $\gamma \in \mathbb{R}^c$ . The intuition behind it is to guide the network based on a set of generated pseudo-classes  $c$ . These pseudo-classes can be seen as cluster indexes that group similar elements. With this in mind, we also propose an EM-training rule for learning the unknown class given a desired target distribution. We want to point out that the pseudo-classes are internal representations of the network that are independent of the labels.

The  $E$ -step obtains the current statistical distribution given the output vectors  $(z_{1},\ldots ,z_{m})$  and the weighting vector  $(\pmb{\gamma})$ . In this case, an approximation of the distribution is obtained by the following equation

$$
\hat {\boldsymbol {z}} = \frac {1}{M} \sum_ {i = 1} ^ {M} p o w e r (\boldsymbol {z} _ {i}, \boldsymbol {\gamma}) \tag {2}
$$

where  $\gamma$  is the weighting vector,  $z_{i}$  is the output vector of the network,  $M$  is the number of elements, and the function power<sup>1</sup> is the element-wise power operation between the output vector and the weighting vector. In addition, we can retrieve the pseudo-classes by the maximum value of the following equation

$$
c ^ {*} = \arg \max  _ {c} \operatorname {p o w e r} \left(\mathbf {z} _ {i}, \gamma\right) \tag {3}
$$

where  $c^*$  is the pseudo-class, which are used in the  $M$ -step for updating the MLP weights. Also, note that the pseudo-classes are not updated in an online manner. Instead, the pseudo-classes are updated after a certain number of iterations. The reason is the network requires a number of iterations to learn the common features.

The  $M$ -step updates the weighting vector  $\gamma$  given the current distribution  $\hat{z}$ . Also, the MLP parameters  $(\theta)$  are updated given the current classification given by the pseudo-classes. The cost function is the variance between the distribution and the desired statistical distribution, which is defined by

$$
c o s t = \left(\hat {\boldsymbol {z}} - \phi\right) ^ {2} \tag {4}
$$

where  $\hat{z}$  is the current statistical distribution of the output vectors, and  $\phi$  is a vector that represents the desired statistical distribution, e.g. uniform distribution. Then, the weighting vector is updated via gradient descent

$$
\gamma = \gamma - \alpha * \nabla_ {\gamma} \operatorname {c o s t} \tag {5}
$$

where  $\alpha$  is the learning rate and  $\nabla_{\gamma}cost$  is the derivative w.r.t  $\gamma$ . Also, the MLP weights are updated via the generated pseudo-classes, which are used as targets in the backpropagation step.

In summary, we propose an EM-training rule for matching the network output vectors and a desired target statistical distribution. The  $E$ -Step generates pseudo-classes and find an approximation of the current statistical distribution of the output vectors. The  $M$ -Step updates the MLP parameters and the weighting vector. With this in mind, we adapt the mentioned training rule for the classless association task. Figure 2 summarizes the presented EM training rule and its components.

![](images/e76b320bcecadf9cdf1140e2e1af5829f40bc1b487008aaf6a0c53bb37f23361.jpg)  
Figure 2: The proposed training rule applied to a single MLP. E-steps generates a set of pseudo-classes  $c_{1}, \ldots, c_{m}$  for each output in the mini-batch of size  $m$ , and a probability approximation  $\hat{z}$  of the output vectors in the mini-batch. M-step updates the MLP weights given the pseudo-classes and the weighting vector  $\gamma$  giving the target statistical distribution  $\phi$ .

# 2.2 CLASSLESS ASSOCIATION LEARNING

The presented classless association model is trained based on a statistical constraint. Formally, the input is represented by the pair  $\pmb{x}^{(1)} \in \mathbb{R}^{n1}$  and  $\pmb{x}^{(2)} \in \mathbb{R}^{n2}$  where  $\pmb{x}^{(1)}$  and  $\pmb{x}^{(2)}$  are two different instances of the same unknown label. The classless association model has two parallel Multilayer Perceptron  $MLP^{(1)}$  and  $MLP^{(2)}$  with training rule that follows an EM-approach (cf. Section 2.1). Moreover, input samples are divided into several mini-batches of size  $m$ .

Initially, all input samples have random pseudo-classes  $c_{i}^{(1)}$  and  $c_{i}^{(2)}$ . The pseudo-classes have the same desired statistical distribution  $\phi$ . Also, the weighting vectors  $\gamma^{(1)}$  and  $\gamma^{(2)}$  are initialized to one. Then each input element from the mini-batch is propagated forward to each MLP. Afterwards, an estimation of the statistical distribution for each MLP  $(\hat{z}^{(1)}$  and  $\hat{z}^{(2)}$ ) is obtained. Furthermore, a new set of pseudo-classes  $(c_{1}^{(1)},\dots ,c_{m}^{(1)}$  and  $c_{1}^{(2)},\dots ,c_{m}^{(2)}$ ) is obtained for each network. Note that this first part can be seen as an  $E$ -step from Section 2.1. We want to point out that the pseudo-classes are updated only after a number of iterations.

The second part of our association training updates the MLP parameters and the weighting vector  $(\gamma^{(1)}$  and  $\gamma^{(2)})$ . In this step, one network  $(MLP^{(1)})$  uses pseudo-classes  $(c_{1}^{(2)},\ldots ,c_{m}^{(2)})$  obtained from the other network  $(MLP^{(2)})$ , and vice versa. In addition, the weighting vector is updated between the output approximation  $(\hat{z}^{(1)}$  and  $\hat{z}^{(2)})$  and the desired target distribution  $(\phi)$ . Figure 3 shows an overview of the presented model. Algorithm 1 summarizes the training algorithm.

# 3 EXPERIMENTS

In this paper, we are interested in learning the relation between elements of two different sets that represent the same unknown classes. With this in mind, we tested our proposed model in a dataset generated from MNIST (Lecun & Cortes, 2010). The procedure of generating classless datasets from labeled datasets have been already applied in (Sutskever et al., 2015; Hsu & Kira, 2015). Our dataset has two disjoint sets (input 1 and input 2) and a uniform distribution between the digits. We took a subset  $S$  of MNIST, such that,  $\mathbb{E}[S \subset MNIST] \sim \mathcal{U}(\mathrm{digits})$ .  $S$  is comprised of 5,800 samples per digit which are split into training (4,200 samples), validation (800 samples) and testing (800 samples). Thus, both sets (input 1 and input 2) amount to a total of 21,000 sample pairs for training, 4,000 samples for validation, and 4,000 samples for testing. Furthermore, each set was split evenly, making sure both sets contain exactly the same number of samples per digit.

![](images/442c108dc241948abbc43780e7eb0a753ce9613d8466514df33529aa4d549276.jpg)  
Figure 3: Overview of the presented model for classless association of two input samples that represent the same unknown classes. The association relies on matching the network output and a statistical distribution. Also, it can be observed that our model uses the pseudo-classes obtained by  $MLP^{(1)}$  as targets of  $MLP^{(2)}$ , and vice versa.

The following parameters turned out being optimal on the validation set. Each internal MLP relies on two fully connected hidden layers of 200 and 100 neurons respectively. We found that changes in the architecture of the MLPs have little impact when comparing the overall model with the corresponding supervised counterpart. The learning rate for the MLPs was set to start at 1.0 and was continuously decaying by half after every 1,000 iterations. We set the initial weighting vector to 1.0 and updated after every 1,000 iterations as well. The target distribution  $(\phi)$  of the unknown classes is the uniform distribution of all digits. The decay of the learning rate for the class weights was given by  $1 / (100 + epoch)^{0.3}$ , where epoch was the number of training iterations so far. minibatches had 10,500 samples (5,250 pairs of samples corresponding to  $25\%$  of the training set) and back-propagation for the MLPs used the mean of the derivatives for each mini-batch. This way, we ensure that all mini-batches have a uniform distribution with respect to the ground truth, and each MLP is trained with two different samples of the same unknown class.

To determine the baseline of the recognition, we compared our model against a supervised association scenario as an upper bound. We used the same MLP parameters and training for a fair comparison. Also, we compared against two standard clustering algorithms applied independently to each set (input 1 and input 2). We used the clustering algorithm implemented in scikit-learn (Pedregosa et al., 2011).

# 4 RESULTS AND DISCUSSION

In this work, we have generated ten different parallel datasets (Section 3) and report the average results. Noteworthy, our task is to learn the classless association. We introduce the Association Accuracy for measuring association, and it is defined by the following equation

$$
\text {A s s o c i a t i o n A c c u r a c y} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left(c _ {i} ^ {(1)} = c _ {i} ^ {(2)}\right) \tag {6}
$$

where the indicator function is one if  $c_i^{(1)} = c_i^{(2)}$ , zero otherwise;  $c_i^{(1)}$  and  $c_i^{(2)}$  are the pseudo-classes for  $MLP^{(1)}$  and  $MLP^{(2)}$ , respectively, and  $N$  is the number of elements. In addition, we also reported the Purity of each set (input 1 and input 2). Purity is defined by

Algorithm 1 Pseudocode of the Classless Association Training based on matching network output against a statistical distribution.  
Require: mini-batch size M, update_classes, learning Rates,  $\gamma^{(1)},\gamma^{(2)},\phi$  {Random Initialization of input and pseudo-classes  $(X^{(1)},X^{(2)},c^{(1)},c^{(2)})$  } for each_epoch  $= 1$  TO max_epoch do {E-STEP} for i=1 TO M do {Equation 1}  $z_{i}^{(1)}\gets$  forward_step(MLP(1),  $\pmb{x}_i^{(1)})$ $z_{i}^{(2)}\gets$  forward_step(MLP(2),  $\pmb{x}_i^{(2)})$  end for  $\hat{z}^{(1)}\gets \frac{1}{M}\sum_{i = 1}^{M}power(z_i^{(1)},\gamma^{(1)})$ $\hat{z}^{(2)}\gets \frac{1}{M}\sum_{i = 1}^{M}power(z_i^{(2)},\gamma^{(2)})$  {M-Step} for i=1 TO M do {MLP(1) is learning from MLP(2), and vice versa} accumulate_gradients_error(MLP(1),  $\pmb{z}_i^{(1)},c_i^{(2)})$  accumulate_gradients_error(MLP(2),  $\pmb{z}_i^{(2)},c_i^{(1)})$  end for backward_step(MLP(1)) backward_step(MLP(2)) {Equation 5} update_weighting_vector  $(\hat{z}^{(1)},\gamma^{(1)},\phi)$  update_weighting_vector  $(\hat{z}^{(2)},\gamma^{(2)},\phi)$  if each_epoch != 1 and each_epoch mod update_classes==0 then {Generating new pseudo-classes} for i=1 TO M do  $c_{i}^{(1)*}\gets arg\max_{c}power(z_{i}^{(1)},\gamma^{(1)})$ $c_{i}^{(2)*}\gets arg\max_{c}power(z_{i}^{(2)},\gamma^{(2)})$  end for end if end for

$$
P u r i t y (\Omega , \mathcal {C}) = \frac {1}{N} \sum_ {i = 1} ^ {k} \max  _ {j} | c _ {i} \cap g t _ {j} | \tag {7}
$$

where  $\Omega = \{gt_1,gt_2,\dots ,gt_j\}$  is the set of ground-truth labels and  $\mathcal{C} = \{c_1,c_2,\ldots ,c_k\}$  is the set of pseudo-classes in our model or the set of cluster indexes of K-means or Hierarchical Agglomerative clustering, and  $N$  is the number of elements.

Table 1 shows the Association Accuracy between our model and the supervised association task and the Purity between our model and two clustering algorithms (K-means and Hierarchical Agglomerative). First, the supervised association task performances better that the presented model. This was expected because our task is more complex in relation to the supervised scenario. However, we can infer from our results that the presented model has a good performance in terms of the classless scenario and supervised method. Second, our model not only learns the association between input samples but also finds similar elements covered under the same pseudo-class. Also, we evaluate the purity of our model and found that the performance of our model reaches better results than both clustering methods for each set (input 1 and input 2).

Table 1: Association Accuracy (\%) and Purity (\%) results. Our model is compared with the supervised scenario (having class labels provided) and with K-means and Hierarchical Agglomerative clustering (having no class information available).  

<table><tr><td rowspan="2">Model</td><td rowspan="2">Association Accuracy (%)</td><td colspan="2">Purity (%)</td></tr><tr><td>input 1</td><td>input 2</td></tr><tr><td>supervised association</td><td>96.7 ± 0.3</td><td>96.7 ± 0.2</td><td>96.6 ± 0.3</td></tr><tr><td>classless association</td><td>87.4 ± 2.9</td><td>87.1 ± 6.6</td><td>87.0 ± 6.4</td></tr><tr><td>K-means</td><td>-</td><td>63.9 ± 2.2</td><td>62.5 ± 3.7</td></tr><tr><td>Hierarchical Agglomerative</td><td>-</td><td>64.9 ± 4.7</td><td>64.3 ± 5.5</td></tr></table>

![](images/0f4316b3a3c82fe02ae3bb84d66af6ddb5ebaad5fdc9dcf4bd1f897a43f73757.jpg)  
Figure 4: Example of the presented model during training. In this example, there are ten pseudo-classes (each digit represents a class), and the input sample has a uniform distribution. Each pseudo-class index is represented by the numbers 0, . . . 9 and the output representation is the average of all images classified by the pseudo-class index. The digits are rescaled between 0.0 and 1.0 for visualization purposes. Initially, the pseudo-classes are assigned randomly to all input pair samples, which holds a uniform distribution (first row). In addition, the association matrix shows the effect of the uniform initialization, and the output representation per pseudo-class indexes looks as a random noise. Afterwards, it can be observed that the output representation slowly converges during training, and the association shows the relation between the occurrences of the pseudo-classes.

Figure 4 illustrates an example of the proposed learning rule. Initially, the pseudo classes are random selected for each MLP. As a result, the output classification of both networks does not show any visible discriminant element and the initial purity is close to random choice. After 1,000 epochs, the networks start learning some features in order to discriminate the input samples. Some digits are visible in the same pseudo class indexes. Later, both MLPs start converging to the same visual representation. For example, the digits "zero", "three", "six", and "nine" are clearly recognized after

![](images/56ac534fc51b381fe582e863babc927f5751750cb5a3a8efc3883449f9808195.jpg)  
Figure 5: Example of the best and worst results. The digits are rescaled between 0.0 and 1.0 for visualization purposes. It can be observed our model is able to learn to discriminate each digit (first row). However, the presented model has a limitation that two digits are assigned to the same pseudo-class (1 and 5 in the worst results).

3,000 epochs. Finally, the association is learned using only the statistical distribution of the input samples and each digit is represented by each pseudo-class.

Figure 5 illustrates two folds of our model. The first row is the best result from our 10-fold experiment. Each pseudo-class is represented by a single digit, and the association matrix shows a distribution per digit close to the desired uniform distribution and the purity of each input is close to the supervised scenario. In contrast, the second row is our worst result. The pseudo-classes "2" and "8" have only a few elements and their association probabilities are close to zero. Consequently, those elements are merged into other pseudo-classes ("1" and "5"). However, our model is still able to clearly recognize the rest of digits and reached better results than the unsupervised scenario. For example, the digit "nine" is represented by pseudo-classes "0".

# 5 CONCLUSION

In this paper, we have shown the feasibility to train a model that has two parallel MLPs under the following scenario: pairs of input samples that represent the same unknown classes. We proposed a model based on gradients for solving the classless association. Our model has an EM-training that matches the network output against a statistical distribution and uses one network as a target of the other network, and vice versa. Our model reaches better performance than K-means and Hierarchical Agglomerative clustering. In addition, we compare the presented model against a supervised method. We find that the presented model with respect to the supervised method reaches good results because of two extra conditions in the unsupervised association: unlabeled data and agree on the same pseudo-class. With this in mind, we are planning to do more exhaustive analysis of the learning behavior with deeper architectures. Moreover, we will work on how a small set of labeled classes affects the performance of our model. Furthermore, we are interested in replicating our findings in more complex scenarios, such as, multimodal datasets like TVGraz (Khan et al., 2009) or Wikipedia featured articles (Rasiwasia et al., 2010). Finally, our work can be applied to more unsupervised scenarios where the data can be extracted simultaneously from different input sources at the same time. Also, transformation functions can be applied to input samples for creating the association without classes.

# ACKNOWLEDGMENTS

We would like to thank Damian Borth, Christian Schulze, Jorn Hees, Tushar Karayil, and Philipp Blandfort for helpful discussions.

# REFERENCES

M T Balaban and S R Waxman. Do words facilitate object categorization in 9-month-old infants? Journal of experimental child psychology, 64(1):3-26, January 1997. ISSN 0022-0965.  
Richard G Casey. Text OCR by solving a cryptogram. International Business Machines Incorporated, Thomas J. Watson Research Center, 1986.  
Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pp. 539-546. IEEE, 2005.  
AP Dempster, NM Laird, and DB Rubin. Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society., 39(1):1-38, 1977.  
Lisa Gershkoff-Stowe and Linda B Smith. Shape and the first hundred nouns. Child development, 75 (4):1098-114, 2004. ISSN 0009-3920.  
Yen-Chang Hsu and Zsolt Kira. Neural network-based clustering using pairwise constraints. arXiv preprint arXiv:1511.06321, 2015.  
Inayatullah Khan, Amir Saffari, and Horst Bischof. Tvgraz: Multi-modal learning of object categories by combining textual and visual features. In AAPR Workshop, pp. 213-224, 2009.  
Kevin Knight, Anish Nair, Nishit Rathod, and Kenji Yamada. Unsupervised analysis for decipherment problems. In Proceedings of the COLING/ACL on Main conference poster sessions, pp. 499-506. Association for Computational Linguistics, 2006.  
Yann Lecun and Corinna Cortes. The MNIST database of handwritten digits. 2010.  
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.  
N. Rasiwasia, J. Costa Pereira, E. Coviello, G. Doyle, G.R.G. Lanckriet, R. Levy, and N. Vasconcelos. A New Approach to Cross-Modal Multimedia Retrieval. In ACM International Conference on Multimedia, pp. 251–260, 2010.  
Ilya Sutskever, Rafal Jozefowicz, Karol Gregor, Danilo Rezende, Tim Lillicrap, and Oriol Vinyals. Towards principled unsupervised learning. arXiv preprint arXiv:1511.06440, 2015.