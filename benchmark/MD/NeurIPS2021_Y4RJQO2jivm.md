# Characterizing and Measuring the Similarity of Neural Networks with Persistent Homology

Anonymous Author(s)

Affiliation

Address

email

# Abstract

Characterizing the structural properties of neural networks is crucial yet poorly understood, and there are no well-established similarity measures between networks. In this work, we observe that neural networks can be represented as abstract simplicial complex and analyzed using their topological 'fingerprints' via Persistent Homology (PH). We then describe a PH-based representation proposed for characterizing and measuring similarity of neural networks. We empirically show the effectiveness of this representation as a descriptor of different architectures in several datasets. This approach based on Topological Data Analysis is a step towards better understanding neural networks and serves as a useful similarity measure.

# 1 Introduction

Machine learning practitioners can train different neural networks for the same task. Even for the same neural architecture, there are many hyperparameters, such as the number of neurons per layer or the number of layers. Moreover, the final weights for the same architecture and hyperparameters can vary depending on the initialization and the optimization process itself, which is stochastic. Thus, there is no direct way of comparing neural networks accounting for the fact that neural networks solving the same task should be measured as being similar, regardless of the specific weights. This also prevents one from finding and comparing modules inside neural networks (e.g., determining if a given sub-network does the same function as other sub-network in another model). Moreover, there are no well-known methods for effectively characterizing neural networks.

This work aims to characterize neural networks such that they can be measured to be similar once trained for the same task, with independence of the particular architecture, initialization, or optimization process. We focus on Multi-Layer Perceptrons (MLPs) for the sake of simplicity. We start by observing that we can represent a neural network as a directed weighted graph to which we can associate certain topological concepts.<sup>1</sup> Considering it as a simplicial complex, we obtain its associated Persistent Diagram. Then, we can compute distances between Persistent Diagrams of different neural networks.

The proposed experiments aim to show that the selected structural feature, Persistent Homology, serves to relate neural networks trained for similar problems and that such a comparison can be performed by means of a predefined measure between the associated Persistent Homology diagrams. To test the hypothesis, we study different classical problems (MNIST, Fashion MNIST, CIFAR-10, and language identification and text classification datasets), different architectures (number and size of layers) as well as a control experiment (input order).

In summary, the main contributions of this work are the following:

- We propose an effective graph characterization strategy of neural networks based on Persistent Homology.  
- Based on this characterization, we suggest a similarity measure of neural networks.  
- We provide empirical evidence that this Persistent Homology framework captures valuable information from neural networks and that the proposed similarity measure is meaningful.

The remainder of this paper is organized as follows. In Section 2, we go through the related work. Then, in Section 3 we describe our proposal and the experimental framework to validate it. Finally, in sections 4 and 5 we report and discuss the results and arrive to conclusions, respectively.

# 2 Related Work

One of the fundamental papers of Topological Data Analysis (TDA) is presented in Carlsson [8] and suggests the use of Algebraic Topology to obtain qualitative information and deal with metrics for large amounts of data. For an extensive overview of simplicial topology on graphs, see Giblin [18], Jonsson [21]. Aktas et al. [2] provide a thorough analysis of PH methods.

More recently, a number of publications have dealt with the study of the capacity of neural networks using PH. Guss and Salakhutdinov [19] characterize learnability of different neural architectures by computable measures of data complexity. Rieck et al. [30] introduce the neural persistence metric, a complexity measure based on TDA on weighted stratified graphs. This work suggests a representation of the neural network as a multipartite graph and the filtering of the Persistent Homology diagrams are performed for each layer independently. As the filtration contains at most 1-simplices (edges), they only capture zero-dimensional topological information, i.e. connectivity information. Donier [14] propose the concept of spatial capacity allocation analysis. Konuk and Smith [22] propose an empirical study of how NNs handle changes in topological complexity of the input data.

In terms of pure neural network analysis, there are relevant works, like Hofer et al. [20], that study topological regularization. Clough et al. [11] introduce a method for training neural networks for image segmentation with prior topology knowledge, specifically via Betti numbers. Corneanu et al. [13] try to estimate (with limited success) the performance gap between training and testing via neuron activations and linear regression of the Betti numbers.

On the other hand, topological analysis of decision boundaries has been a very prolific area. Ramamurthy et al. [28] propose a labeled Vietoris-Rips complex to perform PH inference of decision boundaries for quantification of the complexity of neural networks.

Naitzat et al. [27] experiment on the PH of a wide range of point cloud input datasets for a binary classification problems to see that NNs transform a topologically rich dataset (in terms of Betti numbers) into a topologically simpler one as it passes through the layers. They also verify that the reduction in Betti numbers is significantly faster for ReLU activations than hyperbolic tangent activations.

Liu [25] obtain certain geometrical and topological properties of decision regions for neural models, and provide some principled guidance to designing and regularizing them. Additionally, they use curvatures of decision boundaries in terms of network weights, and the rotation index theorem together with the Gauss-Bonnet-Chern theorem.

Regarding neural network representations, one of the most related works to ours, Gebhart et al. [16], focuses on topological representations of neural networks. They introduce a method for computing PH over the graphical activation structure of neural networks, which provides access to the task-relevant substructures activated throughout the network for a given input.

Interestingly, in Watanabe and Yamana [35], authors work on neural network representations through simplicial complexes based on deep Taylor decomposition and they calculate the PH of neural networks in this representation. In Chowdhury et al. [10], they use directed homology to represent MLPs. They show that the path homology of these networks is non-trivial in higher dimensions and depends on the number and size of the network layers. They investigate homological differences between distinct neural network architectures.

As far as neural network similarity measures are concerned, the literature is not especially prolific. In Kornblith et al. [23], authors examine similarity measures for representations (meaning, outputs of

different layers) of neural networks based on canonical correlation analysis. However, note that this method compares neural network representations (intermediate outputs), not the neural networks themselves. Remarkably, in Ashmore and Gashler [3], authors do deal with the intrinsic similarity of neural networks themselves based on Forward Bipartite Alignment. Specifically, they propose an algorithm for aligning the topological structures of two neural networks. Their algorithm finds optimal bipartite matches between the nodes of the two MLPs by solving the well-known graph cutting problem. The alignment enables applications such as visualizations or improving ensembles. However, the methods only works under very restrictive assumptions, $^2$  and this line of work does not appear to have been followed up.

Finally, we note that there has been a considerable growth of interest in applied topology in the recent years. This popularity increase and the development of new software libraries, along with the growth of computational capabilities, have empowered new works. Some of the most remarkable libraries are Ripser [32, 5], and Flagser [26]. They are focused on the efficient computation of PH. For GPU-Accelerated computation of Vietoris-Rips PH, Ripser++ [37] offers an important speedup. The Python library we are using, Giotto-TDA [31], makes use of both above libraries underneath.

We have seen that there is a trend towards the use of algebraic topology methods for having a better understanding of phenomena of neural networks and having more principled deep learning algorithms. Nevertheless, little to no works have proposed neural network characterizations or similarity measures based on intrinsic properties of the networks, which is what we intend to do.

# 3 Methodology

In this section, we propose our method, which is heavily based on concepts from algebraic topology. We refer the reader to the Supplementary Material for the mathematical definitions. In this section, we also describe the conducted experiments.

Intrinsically characterizing and comparing neural networks is a difficult, unsolved problem. First, the network should be represented in an object that captures as much information as possible and then it should be compared with a measure depending on the latent structure. Due to the stochasticity of both the initialization and training procedure, networks are parameterized differently. For the same task, different functions that effectively solve it can be obtained. Being able to compare the trained networks can be helpful to detect similar neural structures.

We want to obtain topological characterizations associated to neural networks trained on a given task. For doing so, we use the Persistence Homology (from now on, PH) of the graph associated to a neural network. We compute the PH for various neural networks learned on different tasks. We then compare all the diagrams for each one of the task.

More specifically, for each of the studied tasks (image classification on MNIST, Fashion MNIST and CIFAR-10; language identification, and text classification on the Reuters dataset), we proceed as follows:

- We train several neural network models on the particular problem.  
- We create a directed graph from the weights of the trained neural networks (after changing the direction of the negative edges and normalising the weights of the edges).  
- We consider the directed graph as a simplicial complex and calculate its PH, using the weight of the edges as the filtering parameter, which range from 0 to 1. This way we obtain the so-called Persistence Diagram.  
- We compute the distances between the Persistence Diagrams (prior discretization of the Persistence Diagram so that it can be computed) of the different networks.  
- Finally, we analyze the similarity between different neural networks trained for the same task, for a similar task, and for a completely different task, independently of the concrete architecture, to see whether there is topological similarity.

As baselines, we set two standard matrix comparison methods that are the 1-Norm and the Frobenius norm. Having adjacency matrix  $A$  and  $B$ , we compute the difference as  $\text{norm}(A - B)$ . However, these methods only work for matrices of similar size and thus, they are not general enough. We could also have used the Fast Approximate Quadratic assignment algorithm suggested in Vogelstein et al. [34], but for large networks this method becomes unfeasible to compute.

# 3.1 Proposal

Our method is as follows. We start by associating to a neural network a weighted directed graph that is analyzed as an abstract simplicial complex consisting on the union of points, edges, triangles, tetrahedrons and larger dimension polytopes (those are the elements referred as simplices). Abstract simplicial complexes are used in opposition to geometric simplicial complexes, generated by a point cloud embedded in the Euclidean space  $\mathbb{R}^n$ .

Given a trained neural network, we take the collection of neural network parameters as directed and weighted edges that join neurons, represented by graph nodes. Biases are considered as new vertices that join target neurons with an edge having a given weight. Note that, in this representation, we lose the information about the activation functions, for simplicity and to avoid representing the network as a multiplex network. Bias information could also have been ignored because we want large PH groups that characterize the network, while these connections will not change the homology group dimension of any order.

For negative edge weights, we reverse edge directions and maintain the absolute value of the weights. We discard the use of weight absolute value since neural networks are not invariant under weight sign transformations. This representation is consistent with the fact that every neuron can be replaced by a neuron from which two edges with opposite weights emerge and converge again on another neuron with opposite weights. From the point of view of homology, this would be represented as a closed cycle.

We then normalize the weights of all the edges as expressed in Equation 1 where  $w$  is the weight to normalize,  $W$  are all the weights and  $\zeta$  is an smoothing parameter that we set to 0.000001. This smoothing parameter is necessary as we want to avoid normalized weights of edges to be 0. This is because 0 implies a lack of connection.

$$
\max  \left(1 - \frac {| w |}{\max  \left(| \max  (W) | , | \min  (W) |\right)}, \zeta\right) \tag {1}
$$

Given this weighted directed graph, we then define a directed flag complex associated to it. Topology of this directed flag complex can be studied using homology groups  $H_{n}$ . In this work we calculate homology groups up to degree 3 ( $H_0 - H_3$ ) due to computational complexity and our neural network representation method's layer connectivity limit.

The dimensions of these homology groups are known as Betti numbers. The  $i$ -th Betti number is the number of  $i$ -dimensional voids in the simplicial complex ( $\beta_0$  gives the number of connected components of the simplicial complex,  $\beta_1$  gives the number of non-reducible loops and so on). For a deeper introduction to algebraic topology and computational topology, we refer to Edelsbrunner and Harer [15], Ghrist [17].

We work with a family of simplicial complexes,  $K_{\varepsilon}$ , for a range of values of  $\varepsilon \in \mathbb{R}$  so that the complex at step  $\varepsilon_t$  is embedded in the complex at  $\varepsilon_{t+1}$  for  $\varepsilon_t \leq \varepsilon_{t+1}$ , i.e.  $K_{\varepsilon} \subseteq K_{\varepsilon_{t+1}}$ . In our case,  $\varepsilon$  is the minimum weight of included edges of our graph representation of neural networks.

The nested family of simplicial complexes is called a filtration. We calculate a sequence of homology groups by varying the  $\varepsilon$  parameter, obtaining a persistence homology diagram. PH calculations are performed on  $\mathbb{Z}_2$ .

This filtration gives a collection of contained directed weighted graph or simplicial complex  $K_{\varepsilon_{min}} \subseteq \ldots \subseteq K_{\varepsilon_t} \subseteq K_{\varepsilon_{t+1}} \subseteq \ldots \subseteq K_{\varepsilon_{max}}$ , where  $t \in [0,1]$  and  $\varepsilon_{min} = 0$ ,  $\varepsilon_{max} = 1$  (recall that edge weights are normalized).

Given a filtration, one can look at the birth, when a homology class appears, and death, the time when the homology class disappears. The PH treats the birth and the death of these homological features in  $K_{\varepsilon}$  for different  $\varepsilon$  values. Lifespan of each homological feature can be represented as

an interval (birth, death), of the homological feature. Given a filtration, one can record all these intervals by a Persistence Barcode (PB) [8], or in a Persistence Diagram (PD), as a collection of multiset of intervals.

As mentioned previously, our interest in this work is to compare PDs from two different simplicial complexes. There are two distances traditionally used to compare PDs, Wasserstein distance and Bottleneck distance. Their stability with respect to perturbations on PDs has been object of different studies [9, 12].

In order to make computations feasible and to obviate noisy intervals, we filter the PDs by limiting the minimum PD interval size. We do so by setting a minimum threshold  $\eta = 0.01$ . Intervals with a lifespan under this value are not considered. Additionally, for computing distances, we need to remove infinity values. As we are only interested in the deaths until the maximum weight value, we replace all the infinity values by 1.0.

Wasserstein distance calculations are computationally hard for large PDs (each PD of our NN models has a million persistence intervals per diagram). Therefore we use a vectorized version of PDs instead, also called PD discretization. This vectorized version summaries have been proposed and used on recent literature [1, 6, 7, 24, 29].

For the persistence diagram distance calculation, we use the Giotto-TDA library [31] and compute the following supported vectorized persistence summaries: 1. Persistence landscape. 2. Weighted silhouette. 3. Heat vectorizations.

# 3.2 Experimental Framework

Datasets To determine the topological structural properties of trained NNs, we select different kinds of datasets. We opt for four well-known benchmarks in the machine learning community and one regarding language identification: (1) the MNIST $^5$  dataset for classifying handwritten digit images, (2) the Fashion MNIST [36] dataset for classifying clothing images into 10 categories, (3) the CIFAR-10 $^6$  (CIFAR) dataset for classifying 10 different objects, (4) the Reuters dataset for classifying news into 46 topics, and (5) the Language Identification Wikipedia dataset $^7$  for identifying 7 different languages.

We selected these datasets because, apart from being well-known benchmarks, the performances without transfer learning are good enough and they have different data types and sizes. For CIFAR-10 and Fashion MNIST datasets we train a Convolutional Neural Network (CNN) first, and the convolutional layers are shared between all the models of the same dataset as a feature extractor. Recall that in this work we are focusing on MLPs, so we do not consider that convolutional weights. For the MNIST, Reuters and Language Identification datasets, we use an MLP. For Reuters and Language identification datasets, we vectorize the sentences with character frequency.

Experiments Pipeline We study the following variables (hyperparameters): 1. Layer width, 2. Number of layers, 3. Input order $^8$ ), 4. Number of labels (number of considered classes).

We define the base architecture as the one with a layer width of 512, 2 layers, the original features order, and considering all the classes (10 in the case of MNIST, Fashion MNIST and CIFAR, 46 in the case of Reuters and 7 in the case of the language identification task). Then, doing one change at a time, keeping the rest of the base architecture hyperparameters, we experiment with architectures with the following configurations:

- Layer width: 128, 256, 512 (base) and 1024.  
- Number of layers: 2 (base), 4, 6, 8 and 10.  
- Input order: 5 different randomizations (with base structure), the control experiment.  
- Number of labels (MNIST, Fashion MNIST, CIFAR-10): 2, 4, 6, 8 and 10 (base).

![](images/6a79f3bd0d7cdb9d6c4aa131a870e23664a6ebf3df5d49dd381e7ee15d531b46.jpg)  
(a)Reuters

![](images/55e525d74b31a5b75887226980b7bcebf4a07e330b3f7282d2c4cc1d7e5a2ec3.jpg)  
(b) Language Identification

![](images/6b0b0a79a73910cc0369855caf57c830c3b024500dcb3ded92fe2248805862b7.jpg)  
Figure 1: Distance matrices using Silhouette discretization.

- Number of labels (Reuters): 2, 6, 12, 23 and 46 (base).  
- Number of labels (Language Identification): 2, 3, 4, 6 and 7 (base).

Note that this is not a grid search over all the combinations. We always modify one hyperparameter at a time, and keep the rest of them as in the base architecture. In other words, we experiment with all the combinations such that only one of the hyperparameters is set to a non-base value at a time.

For each dataset, we train 5 times (each with a different random weight initialization) each of these neural network configurations. Then, we compute the topological distances (persistence landscape, weighted silhouette, heat) among the different architectures. In total, we obtain  $5 \times 5 \times 3$  distance matrices (5 datasets, 5 random initializations, 3 distance measures). Finally, we average the 5 random initializations, such that we get  $5 \times 3$  matrices, one for each distance on each dataset. All the matrices have dimensions  $19 \times 19$ , since 19 is the number of experiments for each dataset (corresponding to the total the number of architectural configurations mentioned above). Note that the base architecture appears 8 times (1, on the number of neurons per layer, 1 on the number of layers, 1 on the number of labels and the 5 randomizations of weight initializations).

All experiments were executed in a machine with 2 NVIDIA V100 of 32GB, 2 Intel(R) Xeon(R) Platinum 8176 CPU @ 2.10GHz, and of 1.5TB RAM, for a total of around 3 days.

The code and results are fully open source $^{9}$  under MIT license.

# 4 Results & Discussion

Results from control experiments can be seen in the third group on Figures 1 and 4. In these figures, groups are separated visually using white dashed lines. Experiments groups are specified in Table 1. Control experiments in all the images appear very dimmed, which means that they are very similar, as expected. Recall that the control experiments consist of 5 (randomizations)  $\times$  5 (executions) and that 25 different neural networks have been trained; each one of the network has more than 690,000 parameters that have been randomly initialized. After the training, results show that these networks have very close topological distance, as

Table 1: Indices of the experiments of the distance matrices.  

<table><tr><td>Number</td><td>Experiment</td><td>Index</td></tr><tr><td>1</td><td>Layer size</td><td>1-4</td></tr><tr><td>2</td><td>Number of layers</td><td>5-9</td></tr><tr><td>3</td><td>Input order</td><td>10-14</td></tr><tr><td>4</td><td>Number of labels</td><td>15-19</td></tr></table>

![](images/e52d42078b6032cb2b4947306c8feec03ad4efd9566e9a354c33015674103ee5.jpg)  
(a) 1-norm

![](images/64d449819becfe6ae98c183deadbf347094dd17d18ccc3a0e881ea31cbe45b08.jpg)  
(b) Frobenius norm

![](images/b3bdf2c7b059b1276e3214d0609f476928001f85139d47f3807035a151a77d56.jpg)  
Figure 2: Control experiments using norms.

Table 2: Normalized difference comparison of self-norm against the maximum mean distance of the experiment.  

<table><tr><td>Norm</td><td>Minimum</td><td>Maximum</td><td>Mean</td><td>Standard deviation</td></tr><tr><td>1-Norm</td><td>0.6683</td><td>4.9159</td><td>1.9733</td><td>1.5693</td></tr><tr><td>Frobenius</td><td>0.0670</td><td>0.9886</td><td>0.4514</td><td>0.3074</td></tr></table>

For Figure 2 we computed both 1-norm and Frobenius norm (the baselines) for graphs' adjacency matrices of control experiments. Note that as we ran the experiment five times, we make the mean for each value of the matrix. In order to show whether the resulting values are positive or negative, we subtract to the maximum difference of each dataset the norm of each cell separately, we take the absolute value and we divide by the maximum difference of each dataset. Therefore, we obtain five values per dataset. Table 2 shows the statistics reflecting that the distance among the experiments are large and, thus, they are not characterizing any similarity but rather an important dissimilarity.

In contrast, Figure 3, with our method (Silhouette), shows perfect diagonal of similarity blocks. In the corresponding numeric results, we obtained show small distances, as shown in Table 3. We can appreciate that each dataset has its own hub. This confirms the validity of our proposed similarity measure.

The method we present also seems to capture some parts of hyperparameter setup. For instance, in Figure 4 we can observe gradual increase of distances in the first group regarding layer size meaning that, as layer size increases, the topological distance increases too. Similarly, for the number of layers (second group) and number of labels (fourth group) the same situation holds. Note that in Fashion MNIST and CIFAR-10, the distances are dimmer because we are not dealing with the weights of the CNNs. Recall that the CNN acts as a frozen extractor and are pretrained for all runs (with the same weights), such that the MLP layers themselves are the only potential source of dissimilarity between runs.

![](images/63d65ebd1154d136107ab52beec3943fdc1af2fd799e0afd2fc20770ccaed2b2.jpg)  
Figure 3: Control experiment comparison matrix using Silhouette discretization.

![](images/445de5fc9377b06914ef538e7d0024a77f6c4995a728bf79df9470dca46874ef.jpg)  
(a) MNIST

![](images/9ba7fee7bc4ea4bb5d68fa49398d2d5a3859a484b1ecd1921f6bf8ed24ea83e4.jpg)  
(b) Fashion MNIST

![](images/00d953b3327b5055c66645d6dd6aaeb1c4987c6a4a6bce1d44d8aa4dbd28019c.jpg)  
(c) CIFAR-10

![](images/bbfe3316f4cbb36da7bf4d5e91f851a1136424b3f71b7a66870d7b70b321c2ec.jpg)  
Figure 4: Distance matrices using Heat discretization.

Table 3: PH distances across input order (control) experiments, normalized by dataset.  

<table><tr><td></td><td colspan="2">Heat distance</td><td colspan="2">Silhouette distance</td></tr><tr><td>Dataset</td><td>Mean</td><td>Deviation</td><td>Mean</td><td>Deviation</td></tr><tr><td>MNIST</td><td>0.0291</td><td>0.0100</td><td>0.1115</td><td>0.0364</td></tr><tr><td>F. MNIST</td><td>0.0308</td><td>0.0132</td><td>0.0824</td><td>0.0353</td></tr><tr><td>CIFAR-10</td><td>0.0243</td><td>0.0068</td><td>0.0769</td><td>0.0204</td></tr><tr><td>Language I.</td><td>0.0159</td><td>0.0040</td><td>0.0699</td><td>0.0159</td></tr><tr><td>Reuters</td><td>0.0166</td><td>0.0051</td><td>0.0387</td><td>0.0112</td></tr></table>

Thus, our characterization is sensitive to the architecture (e.g., if we increase the capacity, distances vary), but at the same time, as we saw before, it is not dataset-agnostic, meaning that it also captures whether two neural networks are learning the same problem or not.

In Figure 4, Fashion MNIST (Figure 4b) and CIFAR (Figure 4c) dataset results are interestingly different from those of MNIST (Figure 4a) dataset. This is, presumably, because both Fashion MNIST and CIFAR use a pretrained CNN for the problem. Thus, we must analyze the results taking into account this perspective. The first fully connected layer size is important as it can avoid a bottleneck from the previous CNN output. Some works in the literature show that adding multiple fully connected layers does not necessarily enhance the prediction capability of CNNs [4], which is congruent with our results when adding fully connected layers (experiments 5 to 9) that result in dimmer matrices than the one from. Concerning the experiments on input order, there is slightly more homogeneity than in MNIST, again showing that the order of sample has negligible influence. Moreover, there could have been even more homogeneity taking into account that the fully connected network reduced its variance thanks to the frozen weights of the CNN. This also supports the fact that the CNN is the main feature extractor of the network. As in MNIST results, CIFAR results show that the topological properties are, indeed, a mapping of the practical properties of neural networks.

![](images/1e7c27d023b4d5803f396505f19b97977af6c6dff20098a72baf77fb0fbbb41b.jpg)  
Figure 5: Language Identification dataset PH Landscape distance matrix.

We refer to the Supplementary Material for all distance matrices for all datasets and all distances, as well as for the standard deviations matrices and experiment group statistics.

# 5 Conclusions & Future Work

Results from different experiments, in five different datasets from computer vision and natural language, lead to similar topological properties and are trivially interpretable, which yields to general applicability.

The bests discretizations chosen for this work are the Heat and Silhouette. They show better separation of experiment groups, and are effectively reflecting changes in a sensitive way. We also explored the Landscape discretization but it offers a very low interpretability and clearance.

In other words, it is not helpful for comparing PH diagrams associated to neural networks.

The most remarkable conclusion comes from the control experiments. The corresponding neural networks, with different input order but the same architecture, are very close to each other. The PH framework does, indeed, abstract away the specific weight values, and captures latent information from the networks, allowing comparisons to be based on the function they approximate. The selected neural network representation is reliable and complete, and yields coherent and meaningful results. Instead, the baseline measures, the 1-Norm and the Frobenius norm, implied an important dissimilarity between the experiments in the control experiments, meaning that they did not capture the fact that these neural networks were very similar in terms of the solved problem.

We conclude that our proposed characterization, does, indeed, capture meaningful information from neural network, and the computed distances can serve as an effective similarity measure between networks. To the best of our knowledge, this similarity measure between neural networks is the first of its kind.

As future work, we suggest adapting the method to different deep learning libraries and make it support popular neural architectures such as CNNs, Recurrent Neural Networks, and Transformers [33]. Finally, we suggest performing more analysis regarding the learning of a neural network, and trying to topologically answer the question of how a neural network learns.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]  
(b) Did you describe the limitations of your work? [Yes] See second paragraph of the introduction and last paragraph of the conclusions.  
(c) Did you discuss any potential negative societal impacts of your work? [N/A]  
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]  
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] Both code and outputs.  
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [Yes] Check the Experimental Framework Section and the code.  
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] We include means, standard deviations and raw outputs.  
(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] Check Experimental Framework Section.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes] In the case of the datasets. We do not use any other additional asset.  
(b) Did you mention the license of the assets? [No]  
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] Code, results and pictures we have made for explanations.  
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [N/A]  
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [N/A]

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]  
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]  
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# References

[1] H. Adams, T. Emerson, M. Kirby, R. Neville, C. Peterson, P. Shipman, S. Chepushtanova, E. Hanson, F. Motta, and L. Ziegelmeier. Persistence images: A stable vector representation of persistent homology. J. Mach. Learn. Res., 18:8:1-8:35, 2017.  
[2] M. Aktas, E. Akbas, and A. E. Fatmaoui. Persistence homology of networks: methods and applications. Applied Network Science, 4:1-28, 2019.  
[3] S. Ashmore and M. Gashler. A method for finding similarity between multi-layer perceptrons by forward bipartite alignment. In 2015 International Joint Conference on Neural Networks (IJCNN), pages 1-7, 2015. doi: 10.1109/IJCNN.2015.7280769.  
[4] S. H. S. Basha, S. R. Dubey, V. Pulabaigari, and S. Mukherjee. Impact of fully connected layers on performance of convolutional neural networks for image classification. CoRR, abs/1902.02771, 2019. URL http://arxiv.org/abs/1902.02771.  
[5] U. Bauer. Ripser: efficient computation of victoris-rips persistence barcodes, 2021.  
[6] E. Berry, Y.-C. Chen, J. Cisewski-Kehe, and B. T. Fasy. Functional summaries of persistence diagrams. Journal of Applied and Computational Topology, 4:211–262, 2020.  
[7] P. Bubenik. Statistical topological data analysis using persistence landscapes. J. Mach. Learn. Res., 16:77-102, 2015.  
[8] G. Carlsson. Topology and data. Bulletin of the American Mathematical Society, 46:255-308, 2009.  
[9] F. Chazal, V. D. Silva, and S. Oudot. Persistence stability for geometric complexes. Geometriae Dedicata, 173:193-214, 2012.  
[10] S. Chowdhury, T. Gebhart, S. Huntsman, and M. Yutin. Path homologies of deep feedforward networks. 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), pages 1077–1082, 2019.  
[11] J. Clough, I. Öksüz, N. Byrne, V. Zimmer, J. A. Schnabel, and A. P. King. A topological loss function for deep-learning based image segmentation using persistent homology. IEEE transactions on pattern analysis and machine intelligence, PP, 2020.  
[12] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer. Stability of persistence diagrams. Proceedings of the twenty-first annual symposium on Computational geometry, 2005.  
[13] C. Corneanu, M. Madadi, S. Escalera, and A. Martínez. Computing the testing error without a testing set. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2674-2682, 2020.  
[14] J. Donier. Capacity allocation analysis of neural networks: A tool for principled architecture design. ArXiv, abs/1902.04485, 2019.  
[15] H. Edelsbrunner and J. Harer. Computational Topology - an Introduction. American Mathematical Society, 2009.  
[16] T. Gebhart, P. Schrater, and A. Hylton. Characterizing the shape of activation space in deep neural networks. 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), pages 1537-1542, 2019.  
[17] R. Ghrist. Elementary Applied Topology. Self-published, 2014.  
[18] P. Giblin. Graphs, surfaces, and homology: an introduction to algebraic topology. Chapman and Hall, 1977.  
[19] W. H. Guss and R. Salakhutdinov. On characterizing the capacity of neural networks using algebraic topology. ArXiv, abs/1802.04443, 2018.  
[20] C. Hofer, F. Graf, M. Niethammer, and R. Kwitt. Topologically densified distributions. *ArXiv*, abs/2002.04805, 2020.  
[21] J. Jonsson. Simplicial complexes of graphs. PhD thesis, KTH Royal Institute of Technology, 2007.  
[22] E. Konuk and K. Smith. An empirical study of the relation between network architecture and complexity. 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 4597-4599, 2019.

[23] S. Kornblith, M. Norouzi, H. Lee, and G. E. Hinton. Similarity of neural network representations revisited. CoRR, abs/1905.00414, 2019. URL http://arxiv.org/abs/1905.00414.  
[24] P. Lawson, A. Sholl, J. Brown, B. T. Fasy, and C. Wenk. Persistent homology for the quantitative evaluation of architectural features in prostate cancer histology. *Scientific Reports*, 9, 2019.  
[25] B. Liu. Geometry and topology of deep neural networks' decision boundaries. ArXiv, abs/2003.03687, 2020.  
[26] D. Lütgehetmann, D. Govc, J. Smith, and R. Levi. Computing persistent homology of directed flag complexes. arXiv: Algebraic Topology, 2019.  
[27] G. Naitzat, A. Zhitnikov, and L. Lim. Topology of deep neural networks. J. Mach. Learn. Res., 21:184:1-184:40, 2020.  
[28] K. Ramamurthy, K. R. Varshney, and K. Mody. Topological data analysis of decision boundaries with application to model selection. ArXiv, abs/1805.09949, 2019.  
[29] B. A. Rieck, F. Sadlo, and H. Leitte. Topological machine learning with persistence indicator functions. ArXiv, abs/1907.13496, 2019.  
[30] B. A. Rieck, M. Togninalli, C. Bock, M. Moor, M. Horn, T. Gumbsch, and K. Borgwardt. Neural persistence: A complexity measure for deep neural networks using algebraic topology. ArXiv, abs/1812.09764, 2019.  
[31] G. Tauzin, U. Lupo, L. Tunstall, J. B. Pérez, M. Caorsi, A. Medina-Mardones, A. Dassatti, and K. Hess. giotto-tda: A topological data analysis toolkit for machine learning and data exploration, 2020.  
[32] C. Tralie, N. Saul, and R. Bar-On. Ripser.py: A lean persistent homology library for python. The Journal of Open Source Software, 3(29):925, Sep 2018. doi: 10.21105/joss.00925. URL https://doi.org/10.21105/joss.00925.  
[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. CoRR, abs/1706.03762, 2017. URL http://arxiv.org/abs/1706.03762.  
[34] J. T. Vogelstein, J. M. Conroy, V. Lyzinski, L. J. Podrazik, S. G. Kratzer, E. T. Harley, D. E. Fishkind, R. J. Vogelstein, and C. E. Priebe. Fast approximate quadratic programming for graph matching. PLOS ONE, 10(4):1-17, 04 2015. doi: 10.1371/journal.pone.0121002. URL https://doi.org/10.1371/journal.pone.0121002.  
[35] S. Watanabe and H. Yamana. Topological measurement of deep neural networks using persistent homology. In ISAIM, 2020.  
[36] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.  
[37] S. Zhang, M. Xiao, and H. Wang. GPU-accelerated computation of vietoris-rips persistence barcodes. In Symposium on Computational Geometry, 2020.