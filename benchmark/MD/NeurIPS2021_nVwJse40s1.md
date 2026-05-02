# Graph Adversarial Self-Supervised Learning

Anonymous Author(s)

Affiliation

Address

email

# Abstract

This paper studies a long-standing problem of learning the representations of a whole graph without human supervision. The recent self-supervised learning methods train models to be invariant to the transformations (views) of the inputs. However, designing these views requires the experience of human experts. Inspired by adversarial training, we propose an adversarial self-supervised learning (GASSL) framework for learning unsupervised representations of graph data without any handcrafted views. GASSL automatically generates challenging views by adding perturbations to the input and are adversarially trained with respect to the encoder. Our method optimizes the min-max problem and utilizes a gradient accumulation strategy to accelerate the training process. Experimental on ten graph classification datasets show that the proposed approach is superior to state-of-the-art self-supervised learning baselines, which are competitive with supervised models.

# 1 Introduction

Learning effective representations of graph-structured data plays an essential role in a variety of real-world applications, including social, biological, molecules, and financial networks [1]. Recently, graph neural networks (GNNs) have emerged as powerful architectures for learning and analyzing graph representations [2, 3, 4, 5, 6]. GNNs typically learn graph representations in a supervised or semi-supervised setting. In practice, obtaining a large number of labels is often difficult or even impossible, especially in specific areas that are very costly, such as in biochemistry. The labeled graphs may be limited, while unlabeled graphs are easy to collect. Self-supervised learning utilizing unlabeled data has made significant progress in computer vision [10, 11, 12, 13, 14, 15, 16] and shows great potential in exploring unlabeled data to enhance graph deep learning [17, 18, 19, 20].

Despite their success, existing self-supervised learning methods rely heavily on handcrafted view, where the view here refers to human-defined data transformations to preserve the invariance of their intrinsic properties. In recent years, researchers have designed views of graphs from various levels, including nodes dropping, edge perturbation, attribute masking, subgraph [18], and graph diffusion [17]. However, the handcrafted views require expert knowledge and trial and error but also do not yield consistent performance gains across multiple tasks [18]. Therefore, how to automatically search for augmentations for graph data remains an open problem.

GNNs are vulnerable to adversarial attacks, as are deep neural networks. Adversarial attacks usually exploit the gradient information to generate imperceptibly small perturbations that alter the model's output. Adding these adversarial samples to the training set, i.e., adversarial training, can improve the neural network to generalize to out-of-distribution samples [21, 22, 23]. Adversarial training usually leads to a trade-off between robustness and generalization. There has been much research on adversarial training for security purposes [24], in particular, it is still unclear how to combine adversarial training in self-supervised learning of GNNs to improve the classification accuracy.

In this paper, we are motivated to address the drawbacks mentioned above and propose a self-supervised learning framework to train a graph neural network without any class labels. We refer to this novel adversarial self-supervised learning approach as Graph Adversarial Self-Supervised Learning (GASSL). GASSL directly maximizes the similarity of a graph and its perturbed adversarial graph, relying on neither negative pairs nor handcrafted augmented views. In the training phase, we use the gradient accumulation strategy [25, 21] to accelerate the model training. We verify the effectiveness of GASSL on 10 datasets for the graph classification task including the TU datasets [40] and the large scale Open Graph Benchmark (OGB) [27]. We conduct extensive experiments across graph datasets by applying classical GNN models (GCN [4] and GIN[5]) as encoders. Our approach automatically generates challenging views to yield performance gains on multiple tasks compared to handcrafted views. The results show that our method outperforms state-of-the-art graph self-supervised learning and is close to the performance of the supervised GNNs.

Our contribution could be summarized as: (1) We propose a self-supervised learning method GASSL for graph representation learning without human supervision. (2) We use adversarial training to automatically generate challenging views for self-supervised learning in place of handcrafted views, which yield performance gains on multiple datasets. (3) We show that GASSL consistently outperforms state-of-the-art self-supervised models with a significant margin in graph classification tasks. When compared to supervised baselines, GASSL performs on par with or superior to the strong baselines.

# 2 Related work

Graph neural network (GNN) GNN is built on graph structures to learn representation vector  $\mathbf{H}_v$  for each node  $v\in \mathcal{V}$ , which are formalized as the following function:  $\mathbf{H}_v^{(k + 1)} = \mathrm{COMBINE}^{(k)}\left(\mathbf{H}_v^{(k)},\mathrm{AGGREGATE}^{(k)}\left(\left\{\mathbf{H}_u^{(k)},\forall u\in \mathcal{N}(v)\right\}\right)\right)$ , where  $\mathbf{H}_v^{(k)}$  is the embedding of node  $v$  at the  $k$ -th layer,  $\mathcal{N}(v)$  denotes a neighbor set of node  $v$ , and  $\mathbf{H}_v^{(0)} = \mathbf{X}_v$ . COMBINE and AGGREGATE are functions parameterized by neural networks. After  $K$  rounds of message passing, we obtain the final-layer node representations. To obtain the representation of the entire graph  $\mathbf{h}_{\mathcal{G}}\in \mathbb{R}^d$ , we need the permutation-invariant READOUT function as follows:  $\mathbf{h}_{\mathcal{G}} = \mathrm{READOUT}\left(\left\{\mathbf{H}_v\mid v\in \mathcal{V}\right\}\right)$ . Various GNNs have been proposed [4, 28, 5] with various pooling [7, 8, 9], achieving state-of-the-art performance in graph tasks.

Adversarial robustness For adversarial training, more attention is paid to improving model robustness and less on improving generalization performance. For image classification tasks, combining a contrast learning framework with perturbation of the input samples is effective. CLAE [51] employs the Fast Gradient Sign Method (FSGM), and RoCL [52] adopts the Projected Gradient Descent (PGD) method to improve the generalization performance of the model. Concurrent to our work, Tamkin et al. [30] proposed a model-agnostic network (ViewMaker) that perturbs the input by adding an  $\ell_p$  constraint to produce useful views and has successful applications on image, speech, and time-series data. Kong et al. [24] proposed to perturb the features of the input nodes of GNN for better generalization performance and utilized a gradient accumulation strategy to accelerate adversarial training in a supervised learning setting. Out of positive view, Hu et al. [53] proposed to directly learn a set of negative adversaries playing against the self-trained representation.

Graph self-supervised learning Self-supervised learning has recently made new advances in graph representation learning, in which contrast learning [44, 20, 18, 17, 45, 46, 47, 48, 49] has achieved the state-of-the-art performance. Infograph [20] maximizes the mutual information between the graph-level representation and the representations of substructures of different scales (e.g., nodes, edges, triangles). By doing so, the graph-level representations encode aspects of the data shared across different scales of substructures. GraphCL [18] designed four types of graph augmentations to incorporate various priors, including node dropping, edge perturbation, attribute masking, and subgraph. MVGRL [17] utilized graph diffusion for graph augmentation and found no performance gain for more than two views or multi-scales of encoding. For a thorough review, we refer the reader to the recent survey [50]. GCC [19] performs a random walk with a restart for each node to sample subgraph as augmentation. GRACE [48] adopt two augmentations, including removing edges and node feature masking. However, one limitation shared by all these successful approaches is the handcrafted view, which is the primary goal of our GASSL – how to learn a view automatically without resorting to handcrafting or expert domain knowledge.

# 3 Methodology

We now show how to learn the representation of a graph without handcrafted views with domain expert knowledge. Before that, we will briefly introduce adversarial training in supervised learning.

A graph can be represented as  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ , where  $\mathcal{V}$  is the set of  $|\mathcal{V}| = n$  nodes, the adjacency matrix  $\mathbf{A}\in \{0,1\}^{n\times n}$ , along with the  $c$  dimensional node attribute  $\mathbf{X}\in \mathbb{R}^{n\times c}$ . Our goal is to learn from multiple graphs in a dataset and predict the property of a single graph  $\mathcal{G}$ . The learned  $d$  dimensional distributed representation  $\mathbf{h}_{\mathcal{G}}\in \mathbb{R}^d$  is applied for downstream tasks (e.g. graph classification task).

Adversarial robustness We start with the definition of adversarial attacks under supervised settings. Given a dataset  $\mathcal{D} = (X,Y)$ , let  $x\in X$  and  $y\in Y$  denote a training sample and the corresponding label, respectively. Given a supervised learning model  $f_{\theta}:X\to Y$  with parameters  $\theta$ . Traditional adversarial attacks maximize the loss within a certain radius from the sample as follows:

$$
x ^ {i + 1} = \Pi_ {B (x, \epsilon)} \left(x ^ {i} + \alpha \operatorname {s i g n} \left(\nabla_ {x ^ {i}} \mathcal {L} _ {\mathrm {C E}} (\theta , x ^ {i}, y)\right)\right) \tag {1}
$$

where  $B(x,\epsilon)$  is the  $\ell_{\infty}$  norm-ball around  $x$  with radius  $\epsilon$ , and  $\Pi$  is the projection function for norm-ball,  $\alpha$  is the step size,  $i$  is the attack iterations,  $\mathrm{sign}(\cdot)$  returns the sign of the vector,  $\mathcal{L}_{\mathrm{CE}}$  is the cross-entropy loss for supervised training. The straightforward way to defend against adversarial attacks is to minimize the loss of adversarial samples. [26] proposed to seek to find optimal parameters  $\theta^{*}$  to minimize the maximum risk for any  $\delta$  within a norm ball as follows:

$$
\min  _ {\theta} \mathbb {E} _ {(x, y) \sim \mathcal {D}} \left[ \max  _ {\delta \in B (x, \epsilon)} \mathcal {L} _ {\mathrm {C E}} (\theta , x + \delta , y) \right] \tag {2}
$$

where  $\delta$  is the perturbation of the adversarial example. The conventional adversarial attacks [26, 29] require to have a class label  $y\in Y$ , which is not applicable to unlabeled data.

# 3.1 Self-supervised learning on graphs

Self-supervised learning typically design pretext tasks to bring different views of the same instance (positive view) closer and push views of different samples (negative view) farther apart. The simple and performant BYOL [12] does not need to maintain negative views explicitly and depends only on positive views. Inspired by BYOL, we propose GASSL framework (Figure 3.1) to learn graph representation. GASSL comprises the two networks: the teacher and student networks. The two networks shared the same architecture with different parameters. In detail, the teacher network is defined by a set of weights  $\theta$ , while the student network using a different set of weights  $\xi$ , which are an exponential moving average of parameters  $\theta$ . Given a target decay rate  $\beta \in [0,1]$ , after each training step, we perform the following update,  $\xi \gets \beta \xi + (1 - \beta)\theta$ . Note that the predictor  $q_{\theta}$  is only applied to the teacher network to avoid collapse, leading to an asymmetric architecture.

Encoders. In order to learn the graph representation  $z$ , we used GNN (defined in (2)) following with a two-layer multi-layer perceptron (MLP) as an encoder  $f(\cdot)$ . Our framework allows various choices of the network architecture without any constraints. We opt for simplicity and adopt the commonly used GCN [4] and GIN [5].

Similarity loss Our method relies on the positive view of the input graph, which commonly contains: node dropping, edge perturbation, subgraph, and these need to be designed manually. Here suppose we have obtained the positive view, denoted as  $\mathcal{G}'$ . We feed  $\mathcal{G}'$  and  $\mathcal{G}$  to the teacher network and the student network, and obtained the output representations  $q_{\theta}(z_{\theta}')$  and  $z_{\xi}$ , respectively. We then  $\ell_2$ -normalized both  $q_{\theta}(z_{\theta}')$  and  $z_{\xi}$  to  $\bar{q}_{\theta}(z_{\theta}') = q_{\theta}(z_{\theta}') / ||q_{\theta}(z_{\theta}')||_2$  and  $\bar{z}_{\xi} = z_{\xi} / ||z_{\xi}||_2$ . We define the mean squared error as follows,

$$
\mathcal {L} _ {\theta , \xi} = \| \bar {q} _ {\theta} \left(z _ {\theta} ^ {\prime}\right) - \bar {z} _ {\xi} \| _ {2} ^ {2} = 2 - 2 \cdot \frac {\langle \bar {q} _ {\theta} \left(z _ {\theta} ^ {\prime}\right) , \bar {z} _ {\xi} \rangle}{\| \bar {q} _ {\theta} \left(z _ {\theta} ^ {\prime}\right) \| _ {2} \cdot \| \bar {z} _ {\xi} \| _ {2}} \tag {3}
$$

We symmetrize the loss  $\mathcal{L}_{\theta, \xi}$  by separately feeding  $\mathcal{G}$  and  $\mathcal{G}'$  to student network and teacher network to compute  $\hat{\mathcal{L}}_{\theta, \xi}$ . At each training step, we optimize the loss as follows,

$$
\mathcal {L} _ {\theta , \xi} ^ {\text {G A S S L}} = \mathcal {L} _ {\theta , \xi} + \hat {\mathcal {L}} _ {\theta , \xi}. \tag {4}
$$

![](images/623d347340cb37340360e61cda2695bd386b31762d5aec5e7e5b2d2600ed31b2.jpg)  
Figure 1: The proposed GASSL consists of two networks, called teacher network (upper) and student network (lower). The encoder of the student network is the moving average of the teacher network. The teacher network has an additional MLP  $q_{\theta}$  to avoid collapse. A graph  $\mathcal{G}$  is processed by the encoder network  $(f_{\theta})$  and momentum encoder network  $(f_{\xi})$ , respectively. GASSL alternately performs the outer minimization training and the inner maximization adversarial training.

# 3.2 Graph adversarial self-supervised learning

To apply adversarial training to self-supervised learning, we optimize the self-supervised loss  $\mathcal{L}^{\mathrm{GASSL}}$  (Eq 4) to replace the supervised learning cross-entropy loss (Eq 2), allowing adversarial training to produce views without labels.

For self-supervised learning, the generated perturbations should be challenging and faithful [30]. The generated views should be complex and robust enough for the encoder to produce a useful representation. We generate perturbations by adversarial training to increase the loss between two networks. The perturbation should not make the encoder task impossible. We accomplish this by constraint the radius  $\epsilon$  of perturbation. We can add perturbations directly to the input node features or the output of the hidden layer of the GNN encoder.

Perturbation on initial node features It is common practice to add noise to the node features  $\mathbf{X}$ , which we denote the perturbed features as  $\mathbf{X}^{\mathrm{adv}} = \mathbf{X} + \boldsymbol{\delta}$ . The adversarial learning objective following the min-max formulation,

$$
\min  _ {\theta} \mathbb {E} _ {\mathcal {G} \sim \mathfrak {G}} \left[ \max  _ {\| \boldsymbol {\delta} \| _ {F} \leq \epsilon} \mathcal {L} ^ {\mathrm {G A S S L}} (\mathcal {G}; \mathbf {X} + \boldsymbol {\delta}) \right] \tag {5}
$$

Perturbation on hidden layers For GNNs, the features of nodes are aggregated by the neighborhoods. Perturbation on the hidden layer output to affect the nodes with their neighborhoods would produce a more challenging view. We denote the output of first hidden layer as  $\mathbf{H}^{(1)}$ , the perturbed output is  $\mathbf{H}^{(1)} + \delta$ . We optimize the following min-max formulation,

$$
\min  _ {\theta} \mathbb {E} _ {\mathcal {G} \sim \mathfrak {G}} \left[ \max  _ {\| \boldsymbol {\delta} \| _ {F} \leq \epsilon} \mathcal {L} ^ {\mathrm {G A S S L}} (\mathcal {G}; \mathbf {H} ^ {(1)} + \boldsymbol {\delta}) \right] \tag {6}
$$

The problem (5) and problem (6) are similar, we analyze problem 6 in the following. For problem (6), the outer 'min' of the is non-convex and the inner 'max' is non-concave. This saddle-point problem could be reliably solved with stochastic gradient descent (SGD) for outer minimization and projected

Algorithm 1 Graph Adversarial Self-Supervised Learning (GASSL)  
Input: Graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  ; input feature matrix  $X$  ; learning rate  $\tau$  ; ascent steps  $T$  ; ascent step size  $\alpha$  perturbation bound  $\epsilon$  , decay rate  $\beta$ $\mathcal{L}^{\mathrm{GASSL}}(\cdot)$  as objective function.   
Initialize  $\pmb{\theta}$    
for epoch  $= 1$  to  $\lceil N_{ep} / T\rceil$  do   
 $\delta_0\gets U(-\epsilon ,\epsilon)$ $g_0\gets 0$    
for  $t = 1$  to  $T$  do   
 $g_{t}\gets g_{t - 1} + \frac{1}{T}\nabla_{\theta}\mathcal{L}^{\mathrm{GASSL}}(\mathcal{G};\boldsymbol {H}^{(1)} + \delta_{t - 1})$  #Accumulate gradient of parameters  $\theta$ $g_{\delta}\gets \nabla_{\delta}\mathcal{L}^{\mathrm{GASSL}}(\mathcal{G};\boldsymbol {H}^{(1)} + \delta_{t - 1})$  #Update the perturbation  $\delta$  via gradient ascend   
 $\delta_t\gets \delta_{t - 1} + \alpha g_\delta /\| g_\delta \| _F$    
end for   
 $\theta \leftarrow \theta -\tau g_T$ $\xi \leftarrow \beta \xi +(1 - \beta)\theta$    
end for

gradient descent (PGD) for inner maximization [26]. In this work, we take unbounded adversarial attacks instead. The parameter  $\delta$  is updated after each step,

$$
\boldsymbol {\delta} _ {t + 1} = \boldsymbol {\delta} _ {t} + \alpha g (\boldsymbol {\delta} _ {t}) / \| g (\boldsymbol {\delta} _ {t}) \| _ {F}, \tag {7}
$$

where  $g(\delta_t) = \nabla_{\delta}\mathcal{L}^{\mathrm{GASSL}}(\mathcal{G};H^{(1)} + \delta_t)$  is the gradient of the loss with respect to  $\delta$

# 3.3 Acceleration training with gradient accumulation

The computation of  $\delta$  is inefficient since  $T$ -step updating takes  $T$  forward-backward passes, and the SGD takes only one pass through the neural network. We then leverage the 'free' strategy [25, 21] for efficient adversarial training. The core idea of 'free' strategy is to accumulate gradients of  $\nabla_{\theta}\mathcal{L}^{\mathrm{GASSL}}$  in each iteration of inner loop and update the model parameter  $\pmb{\theta}$  with the accumulated gradients. During the training procedure, suppose we run inner loop  $T$  times, each time computing gradient for  $\delta_t$  and  $\pmb{\theta}_{t-1}$ . By taking a decent step along the averaged gradients at  $\pmb{H}^{(1)} + \delta_0, \dots, \pmb{H}^{(1)} + \delta_{T-1}$ , we approximately optimize the following objective:

$$
\min  _ {\theta} \mathbb {E} _ {\mathcal {G} \sim \mathfrak {G}} \left[ \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \max  _ {\| \boldsymbol {\delta} _ {t} \| _ {F} \leq \epsilon} \mathcal {L} ^ {\text {G A S S L}} \left(\mathcal {G}; \boldsymbol {H} ^ {(1)} + \boldsymbol {\delta} _ {t}\right) \right] \tag {8}
$$

The overall procedure is shown in Algorithm 1.

Table 1: Statistics of graph classification benchmarks.  

<table><tr><td>Dataset</td><td>MUTAG</td><td>PTC-MR</td><td>IMDB-B</td><td>IMDB-M</td><td>COLLAB</td><td>NCI1</td><td>HIV</td><td>Tox21</td><td>ToxCast</td><td>BBBP</td></tr><tr><td>No. Graphs</td><td>188</td><td>344</td><td>1,000</td><td>1,500</td><td>5,000</td><td>4,110</td><td>41,127</td><td>7,831</td><td>8,576</td><td>2,039</td></tr><tr><td>No. Classes</td><td>2</td><td>2</td><td>2</td><td>3</td><td>3</td><td>2</td><td>2</td><td>12</td><td>617</td><td>2</td></tr><tr><td>No. Nodes</td><td>17.9</td><td>25.5</td><td>19.8</td><td>13.0</td><td>74.5</td><td>29.8</td><td>25.51</td><td>18.57</td><td>18.78</td><td>24.06</td></tr></table>

# 4 Experiment

# 4.1 Datasets

We selected 10 widely used graph classification datasets from TU datasets [40] and Open Graph Benchmark (OGB) [27]. For TU datasets, we select three bioinformatics datasets (MUTAG [41], PTC-MR [42, 41], NCI1 [43]) and three social network datasets (COLLAB [34], IMDB-BINARY [34], IMDB-MULTI [34]). Notably, since the nodes have no features for the social network datasets, we use the one-hot encodings of node degrees as features. We use classification accuracy as an evaluation metric. For OGB datasets, we selected four of the molecular datasets, including HIV, Tox21, ToxCast, and BBBP. We use the ROC-AUC for an evaluation metric. Statistics are reported in Table 1, and more details are described in the Appendix.

# 4.2Baselines

We select three families of baselines, including graph kernel methods, supervised GNN, unsupervised (self-supervised) methods. The graph kernel methods including shortest path kernel (SP) [31], Graphlet kernel (GK) [32], Weisfeiler-Lehman sub-tree kernel (WL) [33], deep graph kernels (DGK) [34], and multi-scale Laplacian kernel (MLG) [35] reported in [20]. The supervised GNN-based models including GraphSAGE [3], GCN[4], GAT [28], GIN-0 and  $\mathrm{GIN - }\epsilon$  reported in [5]. In addition, GNN incorporates newly developed pooling methods to further improve performance on graph classification tasks, and we have selected StructPool [7], MinCutPool [8], and Grpah Multiset Transformer (GMT) [9]. The unsupervised methods including random walk [36], node2vec [37], sub2vec [38], and graph2vec [39]. The state-of-the-art self-supervised graph representation learning including InfoGraph [20], MVGRL [17], GraphCL [18], and GCC [19].

# 4.3 Evaluation protocol

For all experiments on the TU dataset, we follow [20, 9] and report the mean 10-fold cross-validation accuracy with standard deviation after 5 runs followed by a linear SVM. The linear classifier is trained using cross-validation on training folds of data, and the best mean classification accuracy is reported. For OGB datasets, we evaluate the performance with their original feature extraction and following the original training/validation/test dataset splits [27]. We train a linear classifier on the top of a frozen encoder on existing self-supervised learning models [10].

We train the model using Adam optimizer with an initial learning rate of  $10^{-4}$ , and we choose the number of GCN and GIN layers  $\in \{2,3,4,5\}$ , number of epochs  $\in \{20,40,100,200\}$ , batch size  $\in \{32,64,128,256,512,1024\}$ , and the SVM parameter  $C \in \{10^{-3},10^{-2},\ldots,10^{2},10^{3}\}$ . The step size  $\alpha$  is set to  $8 \times 10^{-3}$ , the perturbation bound  $\epsilon$  is set to  $8 \times 10^{-3}$ , the embedding dimension is set to 128 (expect HIV set to 512). We also use early stopping with the patience of 20, where we stop training if there is no further improvement on the validation loss during 20 epochs. We conduct all the experiments on an Nvidia Titan Xp. Code and data are available in the supplemental material.

Table 2: Graph classification results (\%) on test sets. GraphCL-BYOL indicates that the backbone of GraphCL is replaced with BYOL. The best result is bolded, and the second is underlined.  

<table><tr><td></td><td>Backbone</td><td>View</td><td>MUTAG</td><td>PTC-MR</td><td>IMDB-B</td><td>IMDB-M</td><td>COLLAB</td><td>NCI1</td></tr><tr><td>GraphCL[18]</td><td>SimCLR</td><td>Best</td><td>86.8</td><td>-</td><td>71.1</td><td>-</td><td>71.3</td><td>77.8</td></tr><tr><td rowspan="5">GraphCL-BYOL</td><td>BYOL</td><td>NodeDrop</td><td>90.4</td><td>60.5</td><td>73.7</td><td>51.6</td><td>73.3</td><td>78.7</td></tr><tr><td>BYOL</td><td>EdgePert</td><td>90.5</td><td>59.9</td><td>73.9</td><td>50.8</td><td>71.2</td><td>78.5</td></tr><tr><td>BYOL</td><td>Subgraph</td><td>89.1</td><td>59.8</td><td>72.9</td><td>51.2</td><td>72.4</td><td>78.5</td></tr><tr><td>BYOL</td><td>AttrMask</td><td>89.9</td><td>59.6</td><td>73.6</td><td>50.4</td><td>71.4</td><td>79.1</td></tr><tr><td>BYOL</td><td>Best</td><td>90.5</td><td>60.5</td><td>73.9</td><td>51.6</td><td>73.3</td><td>79.1</td></tr><tr><td>GASSL (ours)</td><td>BYOL</td><td>Adversarial</td><td>90.9</td><td>64.6</td><td>74.2</td><td>51.7</td><td>78.0</td><td>80.2</td></tr><tr><td colspan="3">gain from backbone (SimCLR → BYOL)</td><td>3.7</td><td>-</td><td>2.8</td><td>-</td><td>2.0</td><td>1.3</td></tr><tr><td colspan="3">gain from view (GraphCL → Adversarial)</td><td>0.4</td><td>4.1</td><td>0.3</td><td>0.1</td><td>4.7</td><td>0.9</td></tr></table>

# 4.4 The role of adversarial views

Architecture vs. views To illustrate that views generated by adversarial training contribute to graph representation learning, we compare with GraphCL [18] that uses handcrafted views. Our proposed GASSL approach differs from the GraphCL in both the self-supervised learning approach and the view generation. Therefore, we construct a new baseline by combining the BYOL architecture with GraphCL's view, named GraphCL-BYOL. We follow the setting of GraphCL and use GIN as the encoder for all comparison methods. From the Table 2, we observe that replacing the backbone of GraphCL from SimCLR to BYOL yields a consistent performance improvement. We choose the best result as a baseline and compare it with our approach. Our GASSL obtained a boost ranging from  $0.1\% \sim 4.7\%$  using adversarial training. It is worth mentioning that compared to GraphCL, our approach improves  $6.7\%$  on COLLAB, which backbone contributes  $2.0\%$  and view contributes  $4.7\%$ . The improvement in classification performance indicates the effectiveness of adversarial training.

![](images/33fd110b4540df4beeff66a945bc8b26af91451cfa1e8abd71b82f1c8648c765.jpg)  
Figure 2: Effects of step size  $\alpha$  and perturbation bound  $\epsilon$  on BBBP dataset.

![](images/697c3995c4e6f01af5225fe1fd097076c17af47f47239885924d6a1ca56d826b.jpg)

Table 3: Graph classification results on test sets. The reported results are mean and standard deviation over 5 different runs. The compared numbers are from the corresponding papers under the same experiment settings.  

<table><tr><td></td><td>Dataset</td><td>MUTAG</td><td>PTC-MR</td><td>IMDB-B</td><td>IMDB-M</td><td>COLLAB</td><td>NCI1</td></tr><tr><td rowspan="5">Kernel</td><td>SP ([31])</td><td>85.2 ± 2.4</td><td>58.2 ± 2.4</td><td>55.6 ± 0.2</td><td>38.0 ± 0.3</td><td>-</td><td>-</td></tr><tr><td>GK ([32])</td><td>81.7 ± 2.1</td><td>57.3 ± 1.4</td><td>65.9 ± 1.0</td><td>43.9 ± 0.4</td><td>72.8 ± 0.3</td><td>62.3 ± 0.3</td></tr><tr><td>WL ([33])</td><td>80.7 ± 3.0</td><td>58.0 ± 0.5</td><td>72.3 ± 3.4</td><td>47.0 ± 0.5</td><td>-</td><td>80.0 ± 0.5</td></tr><tr><td>DGK ([34])</td><td>87.4 ± 2.7</td><td>60.1 ± 2.6</td><td>67.0 ± 0.6</td><td>44.6 ± 0.5</td><td>73.1 ± 0.3</td><td>62.5 ± 0.3</td></tr><tr><td>MLG ([35])</td><td>87.9 ± 1.6</td><td>63.3 ± 1.5</td><td>66.6 ± 0.3</td><td>41.2 ± 0.0</td><td>-</td><td>-</td></tr><tr><td rowspan="5">supervised</td><td>GraphSAGE([3])</td><td>85.1 ± 7.6</td><td>63.9 ± 7.7</td><td>72.3 ± 5.3</td><td>50.9 ± 2.2</td><td>-</td><td>77.7 ± 1.5</td></tr><tr><td>GCN ([4])</td><td>85.6 ± 5.8</td><td>64.2 ± 4.3</td><td>74.0 ± 3.4</td><td>51.9 ± 3.8</td><td>79.0 ± 1.8</td><td>80.2 ± 2.0</td></tr><tr><td>GIN-0 ([5])</td><td>89.4 ± 5.6</td><td>64.6 ± 7.0</td><td>75.1 ± 5.1</td><td>52.3 ± 2.8</td><td>80.2 ± 1.9</td><td>82.7 ± 1.7</td></tr><tr><td>GIN-ε ([5])</td><td>89.0 ± 6.0</td><td>63.7 ± 8.2</td><td>74.3 ± 5.1</td><td>52.1 ± 3.6</td><td>80.1 ± 1.9</td><td>82.7 ± 1.6</td></tr><tr><td>GAT ([28])</td><td>89.4 ± 6.1</td><td>66.7 ± 5.1</td><td>70.5 ± 2.3</td><td>47.8 ± 3.1</td><td>-</td><td>-</td></tr><tr><td rowspan="10">unsupervised</td><td>Random Walk ([36])</td><td>83.7 ± 1.5</td><td>57.9 ± 1.3</td><td>50.7 ± 0.3</td><td>34.7 ± 0.2</td><td>-</td><td>-</td></tr><tr><td>node2vec ([37])</td><td>72.6 ± 10.2</td><td>58.6 ± 8.0</td><td>-</td><td>-</td><td>-</td><td>54.9 ± 1.6</td></tr><tr><td>sub2vec ([38])</td><td>61.1 ± 15.8</td><td>60.0 ± 6.4</td><td>55.3 ± 1.5</td><td>36.7 ± 0.8</td><td>-</td><td>52.8 ± 1.5</td></tr><tr><td>graph2vec ([39])</td><td>83.2 ± 9.6</td><td>60.2 ± 6.9</td><td>71.1 ± 0.5</td><td>50.4 ± 0.9</td><td>-</td><td>73.2 ± 1.8</td></tr><tr><td>InfoGraph ([20])</td><td>89.0 ± 1.1</td><td>61.7 ± 1.4</td><td>73.0 ± 0.9</td><td>49.7 ± 0.5</td><td>70.6 ± 1.1</td><td>73.8 ± 0.7</td></tr><tr><td>MVGRL ([17])</td><td>89.7 ± 1.1</td><td>62.5 ± 1.7</td><td>74.2 ± 0.7</td><td>51.2 ± 0.5</td><td>71.3 ± 1.2</td><td>75.0 ± 0.7</td></tr><tr><td>GraphCL ([18])</td><td>86.8 ± 1.3</td><td>-</td><td>71.1 ± 0.4</td><td>-</td><td>71.3 ± 1.1</td><td>77.8 ± 0.4</td></tr><tr><td>GCC ([19])</td><td>86.4 ± 0.5</td><td>58.4 ± 1.2</td><td>71.9 ± 0.5</td><td>48.9 ± 0.8</td><td>75.2 ± 0.3</td><td>66.9 ± 0.2</td></tr><tr><td>GASSL-GCN (ours)</td><td>90.4 ± 7.9</td><td>62.2 ± 6.0</td><td>72.7 ± 0.7</td><td>49.6 ± 2.3</td><td>77.9 ± 2.0</td><td>77.0 ± 1.9</td></tr><tr><td>GASSL-GIN (ours)</td><td>90.9 ± 7.9</td><td>64.6 ± 6.1</td><td>74.2 ± 0.5</td><td>51.7 ± 2.5</td><td>78.0 ± 2.0</td><td>80.2 ± 1.9</td></tr></table>

Effect of step size  $\alpha$  and perturbation bound  $\epsilon$  Step size  $\alpha$  and perturbation bound  $\epsilon$  are critical factors of adversarial training. We evaluate the effect of  $\alpha$  and  $\epsilon$  on the classification accuracy of the BBBP dataset. We separately vary  $\alpha$  and  $\epsilon$  in the range  $\{0.00025, 0.0005, \dots, 64, 128\}$ , while fix the other as 0.008. From Figure 2, the algorithm achieves the best classification performance when  $\alpha \leq 0.008$ . As the step size increases, the classification performance gradually decreases. Observing the perturbation bound  $\epsilon$ , we find that the algorithm performance changes similarly. As mentioned in Section 3.2, the perturbations should be challenging and faithful. An overly large step size  $\alpha$  or perturbation bound  $\epsilon$  leads to perturbed samples that deviate too much from the input graph, and the encoder can hardly learn a useful representation. In the following experiments, we set  $\alpha = \epsilon = 0.008$ .

# 4.5 Comparison with state-of-the-art

# 4.5.1 Results on TU datasets

The results shown in Table 3 suggest that GASSL achieves state-of-the-art results with respect to unsupervised models. For example, on MUTAG it achieves  $90.9\%$  accuracy, a  $1.9\%$  relative improvement over the previous state-of-the-art. For kernel methods, our approach achieves better performance on all datasets. When compared to supervised baselines individually, our model outperforms GraphSAGE in all datasets and outperforms GCN in 4 out of 6 datasets, e.g., a  $4.3\%$  relative improvement on GCN for the MUTAG dataset.

Table 4: Graph classification results on test sets. The reported results are mean and standard deviation over five different runs. The compared numbers are from the corresponding papers under the same experiment settings. The encoder uses GCN combined with sum pooling, and GASSL-H and GASSL-X denote perturbation at the encoder's first hidden layer and input layer, respectively.  

<table><tr><td>Dataset</td><td>HIV</td><td>Tox21</td><td>ToxCast</td><td>BBBP</td></tr><tr><td>GCN[4]</td><td>76.81 ± 1.01</td><td>75.04 ± 0.80</td><td>60.63 ± 0.51</td><td>65.47 ± 1.73</td></tr><tr><td>GIN[5]</td><td>75.95 ± 1.35</td><td>73.27 ± 0.84</td><td>60.83 ± 0.46</td><td>67.65 ± 3.00</td></tr><tr><td>StructPool[7]</td><td>75.85 ± 1.81</td><td>75.43 ± 0.79</td><td>62.17 ± 1.61</td><td>67.01 ± 2.65</td></tr><tr><td>MinCutPool[8]</td><td>75.37 ± 2.05</td><td>75.11 ± 0.69</td><td>62.48 ± 1.33</td><td>65.97 ± 1.13</td></tr><tr><td>GMT[9]</td><td>77.56 ± 1.25</td><td>77.30 ± 0.59</td><td>65.44 ± 0.58</td><td>68.31 ± 1.62</td></tr><tr><td>GASSL-X (ours)</td><td>78.67 ± 1.23</td><td>74.60 ± 0.76</td><td>61.72 ± 0.34</td><td>70.46 ± 1.21</td></tr><tr><td>GASSL-H (ours)</td><td>78.68 ± 1.16</td><td>74.59 ± 0.81</td><td>61.96 ± 0.55</td><td>70.57 ± 1.25</td></tr></table>

Our approach outperforms the state-of-the-art contrastive learning approaches. For example, compared to MVGRL, GASSL has a relative improvement of  $2.55\%$  on average across all datasets. GASSL outperforms GraphCL and GCC with a relative improvement of  $4.08\%$  and  $5.31\%$ , respectively.

# 4.5.2 Results on OGB datasets

We evaluate our method GASSL on 4 larger scale OGB datasets. From Table 4, we observed that perturbing at the first hidden layer (GASSL-H) yields a slight performance gain compared to perturbing at the input node features (GASSL-X). Our method outperforms GCN and GIN on all datasets, demonstrating our method's potential to outperform supervised learning on larger datasets. Compared with the stronger baselines like structurePool and MinCutPool, which exploit the graph structure information. For StructPool, GASSL has a  $3\%$  and  $3.5\%$  gain for the HIV and BBBP datasets, respectively. GASSL outperforms MinCutPool by  $3.2\%$  and  $4.5\%$  for the HIV and BBBP datasets, respectively. The performance is similar on the Tox21 and ToxCast datasets. Our GASSL performs inferior to GMT on Tox21 and ToxCast and superior to GMT on HIV and BBBP. The above results show that our GASSL method can learn a good representation of the graph and outperforms even the state-of-the-art supervised learning methods.

# 4.6 Ablation studies

Table 5: Effect of batch size on the test ROC-AUC (%) on four OGB datasets, with GCN as the encoder and X and H denoting perturbation on the input layer and the first hidden layer, respectively.  

<table><tr><td>Dataset</td><td colspan="2">HIV</td><td colspan="2">Tox21</td><td colspan="2">ToxCast</td><td colspan="2">BBBP</td></tr><tr><td>Batch size</td><td>X</td><td>H</td><td>X</td><td>H</td><td>X</td><td>H</td><td>X</td><td>H</td></tr><tr><td>32</td><td>77.6</td><td>77.6</td><td>74.6</td><td>74.2</td><td>60.8</td><td>62.0</td><td>66.5</td><td>68.9</td></tr><tr><td>64</td><td>77.1</td><td>77.4</td><td>72.3</td><td>74.6</td><td>61.0</td><td>61.9</td><td>65.3</td><td>69.5</td></tr><tr><td>128</td><td>78.7</td><td>78.7</td><td>73.6</td><td>73.3</td><td>60.4</td><td>60.9</td><td>66.1</td><td>70.6</td></tr><tr><td>256</td><td>75.5</td><td>77.0</td><td>71.9</td><td>73.0</td><td>60.9</td><td>61.2</td><td>66.0</td><td>67.9</td></tr><tr><td>512</td><td>75.9</td><td>76.5</td><td>71.1</td><td>72.3</td><td>61.3</td><td>60.6</td><td>65.5</td><td>67.2</td></tr><tr><td>1024</td><td>75.2</td><td>75.1</td><td>70.9</td><td>71.5</td><td>61.2</td><td>59.6</td><td>61.2</td><td>65.4</td></tr><tr><td>Average</td><td>76.7</td><td>76.8(+0.1)</td><td>72.4</td><td>73.2(+0.8)</td><td>60.9</td><td>61.0(+0.1)</td><td>65.1</td><td>67.1(+2.0)</td></tr></table>

Effect of batch size We analyze the sensitivity of the algorithm to the batch size on four OGB datasets. We selected batch size from \{32, 64, 128, 256, 512, 1024\}. From the Table 5, we observe that GASSL performs stably under different batch sizes. The performance gradually decreases as the batch size increases. In particular, GASSL performs well for a batch size of 128. The benefit is that GASSL can be trained with fewer resources. Moreover, the perturbation on the first hidden layer output consistently leads to better test accuracy than perturbation on the input node features.

Table 6: Effect of the number of GNN layers and embedding dimension on the test ROC-AUC (%) on four OGB datasets, with GCN as encoder.  

<table><tr><td>Layers</td><td>HIV</td><td>Tox21</td><td>ToxCast</td><td>BBBP</td></tr><tr><td>2</td><td>78.7</td><td>73.3</td><td>60.9</td><td>70.6</td></tr><tr><td>3</td><td>73.8</td><td>72.2</td><td>60.3</td><td>69.6</td></tr><tr><td>4</td><td>72.1</td><td>72.5</td><td>59.2</td><td>63.0</td></tr><tr><td>5</td><td>71.8</td><td>72.3</td><td>60.5</td><td>63.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dimension</td><td>HIV</td><td>Tox21</td><td>ToxCast</td><td>BBBP</td></tr><tr><td>128</td><td>74.8</td><td>73.3</td><td>60.9</td><td>70.6</td></tr><tr><td>256</td><td>75.5</td><td>73.6</td><td>61.0</td><td>66.5</td></tr><tr><td>512</td><td>78.7</td><td>72.2</td><td>61.1</td><td>63.8</td></tr><tr><td>1024</td><td>77.2</td><td>72.1</td><td>60.0</td><td>66.9</td></tr></table>

Effect of the number of GNN layers We evaluate the effect of the number of layers on the classification accuracy using ROC-AUC performance on the OGB dataset, using GCN as the encoder, and selecting the number of layers from  $\{2,3,4,5\}$  respectively. From Table 6, we can observe that the performance of GASSL gradually decreases as the number of layers increases, while the best performance is obtained when using a 2-layer encoder.

Effect of embedding dimension We test the effect of encoding dimensions on classification accuracy on the OGB datasets. We choose the best encoding dimension among \{128, 256, 512, 1024\}. From Table 6 we observe that for HIV, the test accuracy increases as the encoding dimension increases. We set it to 512 for HIV and 128 for the rest, considering the computational efficiency.

Table 7: Effect of ascent steps  $T$  on the accuracy and training cost (in seconds) for 200 epochs on MUTAG and IMDB-MULTI datasets.  

<table><tr><td>Encoder</td><td>T</td><td>MUTAG</td><td>Cost(s)</td><td>Speed-up</td><td>IMDB-M</td><td>Cost(s)</td><td>Speed-up</td></tr><tr><td rowspan="3">GIN</td><td>1</td><td>89.8 ± 5.8</td><td>37</td><td>1x</td><td>51.5 ± 2.3</td><td>310</td><td>1x</td></tr><tr><td>2</td><td>89.3 ± 7.1</td><td>34</td><td>1.08x</td><td>51.2 ± 2.2</td><td>231</td><td>1.34x</td></tr><tr><td>3</td><td>90.9 ± 7.9</td><td>30</td><td>1.23x</td><td>51.7 ± 2.5</td><td>211</td><td>1.46x</td></tr></table>

Effect of ascent steps We explored the impact of ascent steps  $T$  on the performance of graph classification accuracy. We train the model in the same setting and vary  $T \in \{1,2,3\}$ . From Table 7, we observed that our method achieves a stable performance on test accuracy. When  $T = 3$ , for the IMDB-M dataset, there is an improvement in test accuracy along with a speedup of nearly 1.5 times. The results for other datasets are similar and are detailed in the Appendix.

# 5 Conclusion and Future work

In this paper, we explore a novel problem of how to learn graph representations without human supervision. We propose an adversarial self-supervised learning framework (GASSL) that automatically generates views using adversarial training. Our approach adversarially generates challenging views to train a self-supervised model. We obtain performance gain by generating views through adversarial training compared to handcrafted views. We use a gradient accumulation training method to improve the training efficiency. We conduct extensive experiments on ten datasets. The results show that our method outperforms state-of-the-art graph self-supervised learning and is close to the performance of the supervised GNNs.

For potential negative societal impact, the graph representations can be extended to many fields, such as financial networks, molecular biology. The use of transformations generated by adversarial perturbations does not certainly produce meaningful views. Expert knowledge is also required for domain-specific applications. The limitation of our approach falls in that it exploits the uniformly norm-bounded perturbation and ignores the distribution of the data. Besides, taking full advantage of the existing expert knowledge is the potential to improve performance.

In the future, we will explore the following directions: (1) Explore how to effectively combine adversarial training with existing handcrafted views to enhance performance further. (2) Theoretically analyze the use of adversarial training to improve the performance of downstream tasks. (3) Explore non-uniform norm-bounded perturbations on the graph to generate adversarial samples.

# References

[1] William L. Hamilton, Rex Ying, and Jure Leskovec. Representation learning on graphs: Methods and applications. IEEE Data(base) Engineering Bulletin, 40:52-74, 2017.  
[2] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. In International Conference on Learning Representations, 2016.  
[3] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, page 1025-1035, Red Hook, NY, USA, 2017. Curran Associates Inc.  
[4] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.  
[5] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks. In International Conference on Learning Representations, 2018.  
[6] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. In International Conference on Machine Learning, pages 5449-5458, 2018.  
[7] Hao Yuan and Shuiwang Ji. Structpool: Structured graph pooling via conditional random fields. In ICLR 2020: Eighth International Conference on Learning Representations, 2020.  
[8] Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral clustering with graph neural networks for graph pooling. In ICML 2020: 37th International Conference on Machine Learning, volume 1, pages 874-883, 2020.  
[9] Minki Kang Jinheon Baek and Sung Ju Hwang. Accurate learning of graph representations with graph multiset pooling. In ICLR 2021, 2021.  
[10] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML 2020: 37th International Conference on Machine Learning, volume 1, pages 1597-1607, 2020.  
[11] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. CoRR, abs/1911.05722, 2019.  
[12] Jean-Bastien Grill, Florian Strub, Florent Alché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In Advances in Neural Information Processing Systems, volume 33, 2020.  
[13] Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E. Hinton. Big self-supervised models are strong semi-supervised learners. In Advances in Neural Information Processing Systems, volume 33, pages 22243-22255, 2020.  
[14] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Thirty-fourth Conference on Neural Information Processing Systems (NeurIPS), volume 33, pages 9912-9924, 2020.  
[15] Xinlei Chen, Haoqi Fan, Ross B. Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.  
[16] Priya Goyal, Mathilde Caron, Benjamin Lefaudeaux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Ishan Misra, Armand Joulin, and Piotr Bojanowski. Self-supervised pretraining of visual features in the wild. arXiv preprint arXiv:2103.01988, 2021.  
[17] Kaveh Hassani and Amir Khasahmadi. Contrastive multi-view representation learning on graphs. In Proceedings of the 17th International Conference on Machine Learning (ICML 2020), 06 2020.

[18] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 5803-5815. Curran Associates, Inc., 2020.  
[19] Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, and Jie Tang. GCC: Graph contrastive coding for graph neural network pre-training. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD '20, page 1150-1160, New York, NY, USA, 2020. Association for Computing Machinery.  
[20] Fan-Yun Sun, Jordan Hoffman, Vikas Verma, and Jian Tang. Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization. In International Conference on Learning Representations, 2019.  
[21] Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu. Freelb: Enhanced adversarial training for natural language understanding. In Eighth International Conference on Learning Representations, 2020.  
[22] Cihang Xie, Mingxing Tan, Boqing Gong, Jiang Wang, Alan L. Yuille, and Quoc V. Le. Adversarial examples improve image recognition. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 819-828, 2020.  
[23] Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao. Smart: Robust and efficient fine-tuning for pre-trained natural language models through principled regularized optimization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2177-2190, 2020.  
[24] Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, and Tom Goldstein. Flag: Adversarial data augmentation for graph neural networks. In arxiv:cs.LG, 2021.  
[25] Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S. Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! In Advances in Neural Information Processing Systems, 2019.  
[26] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.  
[27] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In Advances in Neural Information Processing Systems, volume 33, pages 22118-22133, 2020.  
[28] Petar Velikovi, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Y. Bengio. Graph attention networks. In International Conference on Learning Representations, 2018.  
[29] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric Xing, Laurent El Ghaoui, and Michael Jordan. Theoretically principled trade-off between robustness and accuracy. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 7472-7482, 09-15 Jun 2019.  
[30] Alex Tamkin, Mike Wu, and Noah Goodman. Viewmaker networks: Learning views for unsupervised representation learning. In ICLR 2021: The Ninth International Conference on Learning Representations, 2021.  
[31] K. M. Borgwardt and H. P. Kriegel. Shortest-path kernels on graphs. In IEEE International Conference on Data Mining, 2006.  
[32] Nino Shervashidze, S. V. N. Vishwanathan, Tobias H. Petri, and Et Al Kurt Mehlhorn. Efficient graphlet kernels for large graph comparison. Aistats, 5:488-495, 2009.

[33] Nino Shervashidze, Pascal Schweitzer, Erik Jan, Van Leeuwen, and Karsten M. Borgwardt. Weisfeiler-lehman graph kernels. Journal of Machine Learning Research, 12(3):2539-2561, 2011.  
[34] Pinar Yanardag and S.V.N. Vishwanathan. Deep graph kernels. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '15, page 1365-1374, New York, NY, USA, 2015. Association for Computing Machinery.  
[35] Risi Kondor and Horace Pan. The multiscale laplacian graph kernel. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 29, pages 2990-2998. Curran Associates, Inc., 2016.  
[36] Thomas Gartner, Peter Flach, and Stefan Wrobel. On graph kernels: Hardness results and efficient alternatives. In Conference On Learning Theory, pages 129-143, 2003.  
[37] Aditya Grover and Jure Leskovec. Node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, page 855-864, New York, NY, USA, 2016. Association for Computing Machinery.  
[38] Bijaya Adhikari, Yao Zhang, Naren Ramakrishnan, and B. Prakash. Sub2Vec: Feature Learning for Subgraphs, pages 170–182. 06 2018.  
[39] Annamalai Narayanan, Chandramohan Mahinthan, Rajasekar Venkatesan, Lihui Chen, Yang Liu, and Shantanu Jaiswal. graph2vec: Learning distributed representations of graphs. In 13th International Workshop on Mining and Learning with Graphs, 07 2017.  
[40] Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, and Marion Neumann. Tudataset: A collection of benchmark datasets for learning with graphs, 2020.  
[41] Nils Krieger and Petra Mutzel. Subgraph matching kernels for attributed graphs. In Proceedings of the 17th International Conference on Machine Learning (ICML 2012). Morgan Kaufmann, 2012.  
[42] C. Helma, R. D. King, S. Kramer, and A. Srinivasan. The predictive toxicology challenge 2000-2001. Bioinformatics, 17(1):107-108, 2001.  
[43] Nikil Wale and George Karypis. Comparison of descriptor spaces for chemical compound retrieval and classification. In Sixth International Conference on Data Mining (ICDM'06), volume 14, pages 347-375, 2006.  
[44] Petar Velickovic, William Fedus, William L. Hamilton, Pietro Lio, Yoshua Bengio, and R. Devon Hjelm. Deep graph infomax. In International Conference on Learning Representations, 2018.  
[45] Jiaqi Zeng and Pengtao Xie. Contrastive self-supervised learning for graph classification. arXiv preprint arXiv:2009.05923, 2020.  
[46] Zhen Peng, Yixiang Dong, Minnan Luo, Xiao-Ming Wu, and Qinghua Zheng. Self-supervised graph representation learning via global context prediction. arXiv preprint arXiv:2003.01604, 2020.  
[47] Yizhu Jiao, Yun Xiong, Jiawei Zhang, Yao Zhang, Tianqi Zhang, and Yangyong Zhu. Sub-graph contrast for scalable self-supervised graph representation learning. In 2020 IEEE International Conference on Data Mining (ICDM), pages 222-231, 2020.  
[48] Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, and Liang Wang. Deep graph contrastive representation learning. arXiv preprint arXiv:2006.04131, 2020.  
[49] Yanqiao Zhu, Yichen Xu, Feng Yu, Qiang Liu, Shu Wu, and Liang Wang. Graph contrastive learning with adaptive augmentation. In WWW 2021: The Web Conference, 2021.  
[50] Yixin Liu, Shirui Pan, Ming Jin, Chuan Zhou, Feng Xia, and Philip S. Yu. Graph self-supervised learning: A survey, 2021.

[51] Chih-Hui Ho and Nuno Nvasconcelos. Contrastive learning with adversarial examples. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 17081-17093. Curran Associates, Inc., 2020.  
[52] Minseon Kim, Jihoon Tack, and Sung Ju Hwang. Adversarial self-supervised contrastive learning. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 2983-2994. Curran Associates, Inc., 2020.  
[53] Qianjiang Hu, Xiao Wang, Wei Hu, and Guo-Jun Qi. Adco: Adversarial contrast for efficient learning of unsupervised representations from self-trained negative adversaries, 2021.  
[54] Olivier Chapelle, Bernhard Schlkopf, and Alexander Zien. Semi-Supervised Learning. 2010.  
[55] Xiaojin Zhu, Andrew B. Goldberg, Ronald Brachman, and Thomas Dietterich. Introduction to Semi-Supervised Learning. 2009.  
[56] Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, and Tapani Raiko. Semi-supervised learning with ladder networks. In NIPS'15 Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2, volume 28, pages 3546-3554, 2015.  
[57] Samuli Matias Laine and Timo Oskari Aila. Temporal ensembling for semi-supervised learning, 2017.  
[58] Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in Neural Information Processing Systems, volume 30, pages 1195–1204, 2017.  
[59] Takeru Miyato, Andrew M. Dai, and Ian J. Goodfellow. Adversarial training methods for semi-supervised text classification. In ICLR (Poster), 2017.
