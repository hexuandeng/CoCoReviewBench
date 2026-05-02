# SPARK: CO-EXPLORING MODEL SPARSITY AND LOW-RANKNESS FOR COMPACT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Sparsification and low-rank decomposition are two important techniques for deep neural network (DNN) compression. To date, these two popular yet distinct approaches are typically used in a separate way; while their efficient integration for better compression performance is little explored. In this paper we perform systematic co-exploration on the model sparsity and low-rankness towards compact neural networks. We first investigate and analyze several important design factors for the joint pruning and low-rank factorization, including operational sequence, low-rank format, and optimization objective. Based on the observations and outcomes from our analysis, we then propose SPARK, a unified DNN compression framework that can simultaneously capture model SPArcity and low-RanKness in an efficient way. Empirical experiments demonstrate very promising performance of our proposed solution. Notably, on CIFAR-10 dataset, our approach can bring  $1.25\%$ ,  $1.02\%$  and  $0.16\%$  accuracy increase over the baseline ResNet-20, ResNet-56 and DenseNet-40 models, respectively, and meanwhile the storage and computational costs are reduced by  $70.4\%$  and  $71.1\%$  (for ResNet-20),  $37.5\%$  and  $39.3\%$  (for ResNet-56) and  $52.4\%$  and  $61.3\%$  (for DenseNet-40), respectively. On ImageNet dataset, our approach can enable  $0.52\%$  accuracy increase over baseline model with  $48.7\%$  fewer parameters.

# 1 INTRODUCTION

Deep neural network (DNN) has served as the backbone machine learning technique in many modern intelligent systems. To facilitate the low-cost deployment of DNN on the resource-constrained platforms, model compression, as a powerful strategy that can efficiently reduce DNN model size, has been extensively studied in recent years. To date, numerous compression approaches have been proposed to provide compact DNNs for many practical applications (Han et al. (2015b); Gong et al. (2019); Liao et al. (2021)).

Among various types of model compression techniques, sparsification (a.k.a., pruning) and low-rank decomposition are two representative and popular solutions (Wang et al. (2021); Gao et al. (2021); Li et al. (2021a); Xu et al. (2020)). As revealed by their names, the sparse and low-rank methods aim to explore and leverage the potential sparsity and low-rankness of the uncompressed DNNs, respectively. In practice, supported by the widely existed overparameterization phenomenon (Denil et al. (2013); Han et al. (2015b)), such hypothesized structure-level redundancy usually exists and thus it can be safely removed while still preserving high model performance.

Co-exploring Sparsity & Low-rankness: Motivation. Considering the current prosperity of these two methods and their very distinct structural assumptions, an interesting and promising research topic is to explore the efficient integration of sparse and low-rank approaches towards a better model compression solution. As indicated and observed by Yu et al. (2017), DNN models tend to exhibit both sparsity and low-rankness simultaneously. For instance, the smooth components in the weight filters can be represented in the low-rank space, and meanwhile some other important information is sparsely scattered. Evidently, fully leveraging such co-existence of these structure-level patterns, if being performed properly, can potentially bring a powerful compression solution with attractive performance.

Existing Works. Unlike the current extensive research activities on individual sparse and low-rank methods, the investigations on integrating these two approaches, in an efficient and non-trivial way,

are little explored. To date, only very few efforts study the joint exploration of sparsity and low-rankness for DNN model compression. As the pioneering work along this direction, Yu et al. (2017) develops a singular value decomposition (SVD)-free approach to closely approximate original DNN model via combining low-rank matrix factorization and sparse representation. Built on the interesting connection between filter decomposition and filter pruning, Li et al. (2020) interprets the decomposition and pruning of convolutional filter in a unified perspective. Most recently, Li et al. (2021b) proposes a collaborative compression scheme to integrate SVD into model sparsification. By adopting a multi-step heuristic removal strategy, this post-training approach achieves promising task and compression performance.

Unanswered Questions. Although these prior works have demonstrated the huge potentials and attractive benefits of jointly pruning and decomposing, the systematic investigation on their efficient integration is still missing. To be specific, several fundamental and critical questions, whose answers will directly impact the integration scheme and overall compression performance, have not been comprehensively explored yet. For instance, because pruning and decomposition can be jointly performed in several different ways, such as in parallel (Yu et al. (2017)) or in sequence (Li et al. (2021b)), which collaborative strategy is the best fit for the target DNN compression task? Also, considering low-rankness can be exploited from different perspectives, which type of low-rank approach should be adopted? The matrix factorization used in Li et al. (2020; 2021b)? Or even high-order tensor decomposition? In addition, to achieve promising compression performance, what is the most suitable optimization objective that the integration scheme should aim? The approximation error focused in Yu et al. (2017)? The sparsity/low-rankness regularized loss in Yang et al. (2020)? Or some other new alternatives?

Technical Preview and Contributions. To answer these questions and develop efficient integrated model compression solution, in this paper we perform systematic co-exploration on the model sparsity and low-rankness towards compact neural networks. To be specific, we first review and analyze several important design factors for the joint pruning and low-rank decomposition. Based on the observations and outcomes from our analysis, we then propose SPARK, a unified DNN compression framework that can simultaneously capture model SPArcity and low-RanKness in an efficient way. Overall, the contributions of this paper are summarized as follows:

- We systematically investigate and analyze the critical design knobs when co-exploring model sparsity and low-rankness, including operational sequence, low-rank format, and optimization objective. Based on our qualitative and quantitative analysis, we propose several recommended design options for efficient joint pruning and low-rank decomposition.  
- We develop a unified framework that formulates the integration of low-rank decomposition and pruning to an optimization problem with low-tensor-rank and sparse constraints. We then derive a training-aware approach to solve this challenging non-convex high-order tensor-format problem, and thereby leading to efficient exploration of rich low-rankness and sparsity in the model.  
- We empirically evaluate our proposed co-exploration solution for various DNN models on different datasets, and the experimental results demonstrate its very promising performance. Notably, on CIFAR-10 dataset, our solution can bring  $1.25\%$ ,  $1.02\%$  and  $0.16\%$  accuracy increase over the baseline ResNet-20, ResNet-56 and DenseNet-40 models, respectively, and meanwhile the storage and computational costs are reduced by  $70.4\%$  and  $71.1\%$  (for ResNet-20),  $37.5\%$  and  $39.3\%$  (for ResNet-56) and  $52.4\%$  and  $61.3\%$  (for DenseNet-40), respectively. On ImageNet dataset, our approach can enable  $0.52\%$  accuracy increase over baseline model with  $48.7\%$  fewer parameters.

# 2 RELATED WORK

Sparsification. Sparsification, also known as pruning, has been extensively studied for model compression (Han et al. (2015a); Wen et al. (2016); Gao et al. (2019); Guo et al. (2016); Rao et al. (2021)). In general, sparsifying a DNN can be realized via two ways. The first one is to use a certain criterion, e.g., weight magnitude (Han et al. (2015a)), to directly remove some part of the model, and then perform fine-tuning to recover the accuracy. The second one is to add the sparsity-induced regularization during the training, such as  $\ell_1$  or group lasso term (Wen et al. (2016)), to enforce the

sparsity on the model. In addition, in order to achieve good balance between accuracy and complexity reduction, Gao et al. (2019); Guo et al. (2016); Rao et al. (2021) also proposes dynamic pruning. In such scenario, the DNN sparsification is essentially performed in an input-aware way - which part of model should be pruned is dynamically determined by each input data.

Low-Rank Decomposition. Low-rank decomposition is another popular DNN compression approach. Based on different interpretation of the neural network models, the low-rank method can be categorized to matrix decomposition and tensor decomposition. Matrix decomposition views the 4-D weight tensor as the folded matrix, and hence it flattens the 4-D objective to 2-D format and decomposes the reshaped matrix to the product of two small matrices (Tai et al. (2016); Li & Shi (2018); Xu et al. (2020)). On the other aspect, tensor decomposition directly factorizes the 4-D weight tensor to multiple small tensor cores without flattening operations. Such explicit high-order processing, by its nature, can better preserve the important spatial information and correlation existed in the weight tensors. To date several tensor decomposition techniques, such as tensor train, Tucker and tensor ring etc., have been used for DNN model compression (Kim et al. (2016); Novikov et al. (2015); Wang et al. (2018)).

Joint Pruning and Decomposition. As observed by Yu et al. (2017), a well-trained DNN tends to exhibit both sparsity and low-rankness simultaneously. Motivated by this observation, some prior efforts propose to co-exlore these two complementary properties for model compression. As the pioneering work, Yu et al. (2017) decomposes the weight tensors of a pre-trained DNN model into independent low-rank and sparse parts and minimizes the reconstruction error. Different from this parallel scheme, Dubey et al. (2018); Li et al. (2021b) adopt a sequential compression strategy via performing matrix factorization on a pruned model. In addition, Li et al. (2020) proposes to use the sparse/low-rank regularization term, instead of reconstruction error, to enforce the desired structural patterns. Also, notice that all of the existing works focus on using either SVD-based or SVD-free matrix decomposition to exploit the low-rankness of DNN model.

# 3 CO-EXPLORING SPARSITY AND LOW-RANKNESS: ANALYSIS

As outlined in Section 2, the integration of pruning and low-rank decomposition can be specified by several important factors, including operational sequence, low-rankness format and the overall optimization objective. The existence of such large variety of different factors and their combinations, by its nature, calls for the systematic investigation on the best-suited co-exploration scheme for DNN compression. Such analysis framework, if being properly developed, can facilitate the optimal selection of various design factors already proposed in the existing literatures. More importantly, the outcome from this systematic study will further guide and provide the better integration choices that have not been discovered before.

Questions to be Answered. Next we analyze the critical design knobs and factors for efficient coexploration on model sparsity and low-rankness. To that end, three important questions need to be answered.

Question #1: What is the best suitable operational sequence when jointly pruning and low-rank decomposing DNN models?

Analysis. In general, the co-existence of model sparsity and low-rankness can be explored in different ways (see Figure 1). For instance, as adopted in Yu et al. (2017), a well-trained DNN can be closely approximated as the combination of a sparse component and a low-rank component. In other words, the two types of structure-level properties are imposed and leveraged in a spatially parallel way, and we denote this strategy as  $L + S$ , where  $S$  and  $L$  represent sparsification and low-rank decomposition, respectively. On the other hand, the joint use of pruning and factorization can also be performed in a temporally sequential way. As illustrated in Figure 1, the original model can be first imposed with low-rankness (or sparsity), and the size of the resulting partially compressed model can be further reduced by the second-stage pruning (or low-rank decomposition). Following the similar notation, such sequential operation can be denoted as  $S(L)$  and  $L(S)$ . In practice  $L(S)$  is a preferable choice that has been adopted in the prior works (Dubey et al. (2018); Li et al. (2021b)).

Our Proposal. Among the above described three general operational schemes, we believe  $L + S$  is the best choice when considering to integrate pruning and decomposition together for model compression. This is because unlike  $S(L)$  and  $L(S)$ , which ultimately still produce the compressed model

![](images/05605ec48978a288e3db05d7aaf78f84253abea57fd2b5b4195876f37dccb637.jpg)  
Figure 1: Different operational sequences when jointly performing pruning and low-rank decomposition. Here  $L$  and  $S$  represent low-rank decomposition and sparsification, respectively.

in a single representation (sparse or low-rank) space,  $L + S$  enables the simultaneous representation of rich information of DNN models across different subspace, and thereby better preserving the structural characteristics and reducing the potential information loss. To verify our hypothesis, we examine the approximation error incurred by three integration schemes. As shown in Figure 2, with the same compression ratio for the weight tensor of one layer of a pre-trained ResNet-20 on CIFAR-10 dataset,  $L + S$  shows much lower approximation error than its counterparts, especially in high compression ratio region. This experimental phenomenon demonstrates that  $L + S$  scheme indeed can capture both the sparse and low-rank characteristics of DNN model in an efficient way.

Question #2: What is the best suitable low-rank decomposition approach used when coexploring sparsity and low-rankness?

Analysis. From the perspective of linear algebra, the low-rankness of a DNN model can be exploited using different ways. As illustrated in Figure 3, for an example convolutional layer, imposing the low-rank structure can be realized by performing simple matrix factorization or high-order tensor decomposition. Specifically, Yu et al. (2017) chooses SVD-free method to factorize DNN model and obtain the low-rank component, and Li et al. (2021b) proposes to use SVD-based decomposition to serve as the second-stage compression approach in its adopted  $L(S)$  scheme. Notice that though the weights of convolutional layer essentially form a 4-D tensor format, the existing works exploit the low-rankness via using matrix decomposition – the 4-D tensor needs to be first flatten to a 2-D matrix and it is then factorized to two small matrix components.

![](images/f9823bcfeee99d364451554c149a48c181309c2c91e2bfa293dd2073ef82313d.jpg)  
Figure 2: The approximation error when compressing the weight tensor of one layer in ResNet-20 using different operational sequences  $(L + S, S(L)$  and  $L(S))$ . Here mean square error (MSE) is used to measure the difference between the original uncompressed weight tensor and the reconstruction. SVD is adopted as the low-rank decomposition method  $(L)$ . It is seen that  $L + S$  can bring much smaller approximation error than its counterparts with the same compression ratio. More detailed results are reported in Appendix B.1.

Our Proposal. We argue that the high-order tensor decomposition, the option that has not been explored in the integration scheme before, is the better choice than the low-order matrix decomposition adopted in the existing works. This is because as a reshaping-free technique that can directly factorize the tensor-format data to multiple tensor cores, tensor decomposition, such as Tensor Train (TT) and Tucker, can naturally capture and preserve the important spatial information and correlation of the original weight tensors in a more efficient way. Therefore, less information loss is expected after performing low-rank tensor-based DNN compression. To verify our hypothesis, we compare the feature maps of the compressed convolutional layer of ResNet-20 on CIFAR-10 dataset using dif

![](images/dad80cbef746b4db3706e1f926200253237ca94a500d02df1599b7a3bc79f7f2.jpg)  
Figure 3: Exploring low-rankness of convolutional layer via matrix decomposition (Top) and tensor decomposition (Bottom). Here tensor train (TT) decomposition is adopted for illustration.

![](images/426abecbec5ec42e993391aacc5133782eed35fbadc1385075c6106258a23de4.jpg)  
Figure 4: Output feature maps of one layer of ResNet-20 after non-compression (Left), tensor decomposition (Middle) and matrix decomposition (Right). The visualization shown here is based on the information of one channel. It is seen that high-order tensor decomposition makes the feature map of the compressed layer more similar to that of the original uncompressed layer. More detailed results are reported in Appendix B.2.

![](images/f9769f13e73fcbde882902c40442e762a4995fad0358a4d984ac7ec1f8132162.jpg)

![](images/829a9821cc2aec40f5188ab314b33a2575d68481b93b0912c7329107fcb8b46e.jpg)

ferent low-rank methods. As visualized in Figure 4, compared with the matrix decomposition-based approach with the same compression ratio, tensor decomposition can make the output feature map of the compressed layer much more similar to the feature map of the original uncompressed layer. In other words, low-rank tensor method can provide better preservation of the important feature information and thus it can bring potential higher model compression performance.

Question #3: What is the best suitable optimization objective that the integrated compression scheme should aim?

Analysis. To efficiently realize the joint exploration of model sparsity and low-rankness with promising compression performance, different optimization strategies have been proposed in the existing works. For instance, Yu et al. (2017); Ma et al. (2019) aim to minimize the difference between the original weight matrix/tensor and the approximated reconstruction. In addition, Ma et al. (2019) proposes to explicitly add the low-rank and sparse regularization terms to the overall objective function, which can guide the training-aware procedure to enforce the desired structural patterns.

Our Proposal. Different from the existing approximation error-centered or regularized loss-based solutions, we propose that the efficient co-exploration scheme should be interpreted as the optimization procedure with the low-rank and sparse constraints. Our rationale lies on two important observations of the inherent drawbacks of the prior efforts. First, the approximation strategy adopted in Yu et al. (2017); Ma et al. (2019) focuses on making the reconstructed model approach the original model as close as possible. However, since 1) the approximation error always exists; and 2) the original model is not the only choice to achieve the desired accuracy, such strategy inherently can only search the low-rank and sparse components in a limited exploration space, thereby affecting the overall compression performance. Second, though adding the regularization terms into loss function indeed facilitates the extraction of low-rank and sparse patterns, the effect of such simple regularizing method is still limited, especially considering the efforts of pushing for sparsity and for low-rankness may interfere with each other, thereby potentially causing unexpected conflicts. Instead, by explicitly imposing the sparse and low-rank constraints on the overall optimization problem, these two structural requirement can be simultaneously satisfied with the proper use of optimization technique (to be discussed in Section 4). As reported in Appendix A, our proposed constrained optimization strategy can successfully impose the desired low-rankness and sparsity onto the DNN models efficiently.

Summary of Our Analysis. 1 Performing joint pruning and low-rank decomposition in a spatially parallel way  $(L + S)$  is the preferable operational sequence. 2 High-order tensor decomposition is

the most suitable choice for the low-rank approach used in the integrated compression scheme. ③ Imposing low-rankness and sparsity as the direct hard constraints on the loss optimization should be adopted to better satisfy the desired structural requirement.

# 4 CO-EXPLORING SPARSITY AND LOW-RANKNESS: OUR METHOD

Based on the above three key takeaways obtained from Section 3, in this section we propose a unified framework to co-exlore sparsity and low-rankness in an efficient way. To be specific, we first formulate the integration of low-rank tensor decomposition and pruning to a unified optimization problem, and then develop an efficient algorithm to solve this non-convex high-order tensor-format problem.

# 4.1 PROBLEM FORMULATION

Recall that the analysis in Section 3 brings three important observations/proposals: using  $L + S$  operational sequence, choosing tensor decomposition, and directly imposing hard constraints. Built on such three fundamental principles, we are now ready to formulate the integration of pruning and tensor decomposition to a unified optimization problem. To be specific, given an uncompressed DNN model with weight tensor  $\mathcal{W} \in \mathbb{R}^{O \times I \times K \times K}$  of each layer, our goal is to find another compact model with weight tensors  $\mathcal{L} + \mathcal{S}$ , which consists of low-rank component  $\mathcal{L} \in \mathbb{R}^{O \times I \times K \times K}$  and sparse component  $\mathcal{S} \in \mathbb{R}^{O \times I \times K \times K}$  for each layer, to minimize the following loss function:

$$
\min _ {\mathcal {L}, \mathcal {S}} f (\mathcal {L}, \mathcal {S}),
$$

$$
\text {s . t .} \underbrace {\operatorname {r a n k} (\mathcal {L}) \leq \gamma_ {0} , \gamma_ {1} , \cdots , \gamma_ {d}} _ {\text {L o w - t e n s o r - r a n k c o n s t r a i n t}}, \underbrace {\operatorname {c a r d} (\mathcal {S}) \leq \kappa} _ {\text {S p a r s e c o n s t r a i n t}}, \tag {1}
$$

where  $f(\cdot)$  is the loss over the entire training dataset, and  $\gamma_0, \gamma_1, \dots, \gamma_d$  and  $\kappa$  are the desired tensor ranks and the number of non-zero entries for  $\mathcal{L}$  and  $\mathcal{S}$ , respectively. Notice that without loss of generality, we choose tensor train (TT) decomposition as the component high-order low-rank method in our framework. So here  $d$  is the number of decomposed tensor cores with TT decomposition.

# 4.2 OPTIMIZATION

Directly optimizing problem (1) is challenging because of the co-existence of the non-differentiable  $\mathrm{rank}(\cdot)$  and  $\mathrm{card}(\cdot)$  as well as its inherent high-order tensor format. To efficiently solve this problem, we propose to leverage alternating direction optimization method to split these two constraints. To be specific, after introducing two auxiliary variables  $\widehat{\mathcal{L}}$  and  $\widehat{\mathcal{S}}$  that represent the desired low-TT-rankness and sparsity in the optimization process, problem (1) can be then rewritten as:

$$
\min  _ {\boldsymbol {\mathcal {L}}, \boldsymbol {\mathcal {S}}, \hat {\boldsymbol {\mathcal {L}}} \in \mathcal {P}, \hat {\boldsymbol {\mathcal {S}}} \in \mathcal {Q}} f (\boldsymbol {\mathcal {L}}, \boldsymbol {\mathcal {S}}), \tag {2}
$$

$$
\begin{array}{l} \text {s . t .} \quad \mathcal {L} = \widehat {\mathcal {L}}, \mathcal {S} = \widehat {\mathcal {S}}, \end{array}
$$

where  $\mathcal{P} = \{\mathcal{L}|\mathrm{rank}(\mathcal{L})\leq \gamma_1,\dots ,\gamma_d\}$  is the set of all tensors that satisfy the low-tensor-rank constraint, and  $\mathcal{Q} = \{\mathcal{S}|\mathrm{card}(\mathcal{S})\leq \kappa \}$  is the set of all tensors that satisfy the sparse constraint. Then, we further relax the hard constraints to the corresponding augmented Lagrangian form and now we only need to optimize the following new constraint-free min-max problem:

$$
\min  _ {\mathcal {L}, \mathcal {S}, \widehat {\mathcal {L}} \in \mathcal {P}, \widehat {\mathcal {S}} \in \mathcal {Q}} \max  _ {\boldsymbol {u}, \boldsymbol {v}} f (\boldsymbol {\mathcal {L}}, \boldsymbol {s}) + \frac {\lambda}{2} \left(\| \boldsymbol {\mathcal {L}} - \widehat {\boldsymbol {\mathcal {L}}} + \boldsymbol {u} \| _ {F} ^ {2} + \| \boldsymbol {s} - \widehat {\boldsymbol {\mathcal {S}}} + \boldsymbol {v} \| _ {F} ^ {2} + \| \boldsymbol {u} \| _ {F} ^ {2} + \| \boldsymbol {v} \| _ {F} ^ {2}\right), \tag {3}
$$

where  $\mathcal{U}$  and  $\mathcal{V}$  are the dual multipliers associated to  $\mathcal{L}$  and  $\mathcal{S}$ , respectively, and  $\lambda$  is the penalty parameters. To solve this minmax problem, we can split it into three separated parts, and independently optimize them in an iterative way.

Update  $\mathcal{L}$  and  $S$  with SGD. The first independent optimization objective can be formulated as:

$$
\min  _ {\mathcal {L}, \mathcal {S}} f (\mathcal {L}, \mathcal {S}) + \frac {\lambda}{2} \left(\| \mathcal {L} - \widehat {\mathcal {L}} + \mathcal {U} \| _ {F} ^ {2} + \| \mathcal {S} - \widehat {\mathcal {S}} + \mathcal {V} \| _ {F} ^ {2}\right). \tag {4}
$$

Since there are no hard constraints on the target variables  $\mathcal{L}$  and  $\mathcal{S}$ , standard DNN optimizer (e.g., stochastic gradient descent (SGD)) can be directly applied with learning rate  $\alpha$  as:

$$
\mathcal {L} \leftarrow \mathcal {L} - \alpha \left[ \nabla_ {\mathcal {L}} f (\mathcal {L}, \mathcal {S}) + \lambda \left(\mathcal {L} - \widehat {\mathcal {L}} + \mathcal {U}\right) \right], \tag {5}
$$

$$
\boldsymbol {S} \leftarrow \boldsymbol {S} - \alpha \left[ \nabla_ {\boldsymbol {S}} f (\boldsymbol {\mathcal {L}}, \boldsymbol {\mathcal {S}}) + \lambda (\boldsymbol {S} - \widehat {\boldsymbol {\mathcal {S}}} + \boldsymbol {\mathcal {V}}) \right]. \tag {6}
$$

Update  $\widehat{\mathcal{L}}$  with TT Decomposition. To update the introduced  $\widehat{\mathcal{L}}$ , the optimization objective is:

$$
\min  _ {\widehat {\mathcal {L}} \in \mathcal {P}} \frac {\lambda}{2} \| \boldsymbol {\mathcal {L}} - \widehat {\boldsymbol {\mathcal {L}}} + \boldsymbol {\mathcal {U}} \| _ {F ^ {\prime}} ^ {2}. \tag {7}
$$

Because  $\widehat{\mathcal{L}}$  is strictly constrained to stay in the low-tensor-rank set  $\mathcal{P}$ , the desired update can be performed using an analytical solution via TT-rank truncation, i.e.

$$
\widehat {\mathcal {L}} \leftarrow \operatorname {t r u n c} _ {\mathcal {P}} (\mathcal {L} + \mathcal {U}). \tag {8}
$$

To realize such truncating operation, we first define a temporary tensor  $\mathcal{T} = \mathcal{L} + \mathcal{U}$  and reshape it as a new tensor  $\widetilde{\mathcal{T}}\in \mathbb{R}^{(K\times K)\times (O_1\times I_1)\times \dots \times (O_d\times I_d)}$  with  $O = \prod_{k = 1}^{d}I_{k},I = \prod_{k = 1}^{d}I_{k}$ . Then  $\widetilde{\mathcal{T}}$  can be decomposed to  $d + 1$  TT-cores as:

$$
\widetilde {\boldsymbol {\mathcal {T}}} \left(\left(k _ {1}, k _ {2}\right), \left(o _ {i}, i _ {1}\right), \dots , \left(o _ {d}, i _ {d}\right)\right)) = \boldsymbol {\mathcal {C}} _ {0} \left(k _ {1}, k _ {2}\right) \boldsymbol {\mathcal {C}} _ {1} (:, o _ {1}, i _ {1},:) \dots \boldsymbol {\mathcal {C}} _ {d} (:, o _ {d}, i _ {d},:), \tag {9}
$$

where  $\mathcal{C}_0\in \mathbb{R}^{K\times K},\mathcal{C}_j\in \mathbb{R}^{R_{j - 1}\times I_j\times I_j\times R_j},j = 1,\dots ,d.$  In this TT-format, the dimensions of TT-ranks in TT-cores are truncated to the desired target, i.e.,  $\mathcal{C}_j^{\prime} = \mathcal{C}(1:\gamma_{j - 1},:,;1:\gamma_j)$ . After that we use the truncated TT-cores to recover the original tensor via:

$$
\widetilde {\boldsymbol {\mathcal {T}}} ^ {\prime} \left(\left(k _ {1}, k _ {2}\right), \left(o _ {i}, i _ {1}\right), \dots , \left(o _ {d}, i _ {d}\right)\right) = \boldsymbol {\mathcal {C}} _ {0} \left(k _ {1}, k _ {2}\right) \boldsymbol {\mathcal {C}} _ {1} ^ {\prime} (:, o _ {1}, i _ {1},:) \dots \boldsymbol {\mathcal {C}} _ {d} ^ {\prime} (:, o _ {d}, i _ {d},:). \tag {10}
$$

And finally  $\widetilde{\pmb{\tau}}'$  is reshaped to the original shape of  $\widehat{\mathcal{L}}$  to serve as the updated  $\widehat{\mathcal{L}}$ .

Update  $\widehat{S}$  with Projection. For updating  $\widehat{S}$ , the third optimization objective is:

$$
\min  _ {\widehat {\boldsymbol {S}} \in \mathcal {Q}} \frac {\lambda}{2} \| \boldsymbol {S} - \widehat {\boldsymbol {S}} + \boldsymbol {\nu} \| _ {F} ^ {2}. \tag {11}
$$

Similar to the low-tensor-rank  $\widehat{\mathcal{L}}$ , the sparse-constrained  $\widehat{\mathcal{S}}$  can also be analytically updated as

$$
\widehat {\boldsymbol {S}} \leftarrow \operatorname {p r o j} _ {\mathcal {Q}} (\boldsymbol {S} + \boldsymbol {\mathcal {V}}), \tag {12}
$$

where  $\mathbf{proj}(\cdot)$  is the projection that removes the smallest values to ensure that the updated  $\widehat{\mathcal{S}}$  can satisfy the sparse constraint.

Update Multipliers  $\mathcal{U},\mathcal{V}$ . Upon the update of  $\widehat{\mathcal{L}}$  and  $\widehat{\mathcal{S}}$ , the dual multipliers  $\mathcal{U}$  and  $\mathcal{V}$  can be then directly updated as:

$$
\mathcal {U} \leftarrow \mathcal {U} + \mathcal {L} - \widehat {\mathcal {L}}, \mathcal {V} \leftarrow \mathcal {V} + \mathcal {S} - \widehat {\mathcal {S}}. \tag {13}
$$

Notice that after the iterative update finishes, the low-rank component  $\mathcal{L}$  is explicitly decomposed to TT-cores  $\{\mathcal{C}\}_{j=0}^{d}$ , and the entire compressed model consisting of TT-cores and sparse part  $\mathcal{S}$  is finally fine-tuned with standard SGD. The overall SPARK algorithm is summarized in Algorithm 1.

# 5 EXPERIMENTS

# 5.1 EXPERIMENTAL SETTING

Dataset and Baseline. We evaluate our proposed approach on two image classification datasets (CIFAR-10 and ImageNet). For experiments on CIFAR-10 dataset, three CNN models (ResNet-20, ResNet-56 and DenseNet-40) are compressed and tested. For experiments on ImageNet dataset, we evaluate our approach for ResNet-50 and compare its performance with state-of-the-art model compression methods.

Algorithm 1 The overall SPARK algorithm for co-exploring model sparsity and low-rankness

Input: Pre-trained weight tensor  $\mathcal{W}$ , target TT-ranks  $\{\gamma_j\}_{j=0}^d$ , sparse target  $\kappa$ , training epochs  $T$ .

Output: TT-cores  $\{\mathcal{C}\}_{j = 0}^{d}$ , sparse component  $\mathcal{S}$ .

1: Initialize  $\mathcal{L},\widehat{\mathcal{L}},\mathcal{S},\widehat{\mathcal{S}}$  with  $\mathcal{W}$  
2: Initialize  $\mathcal{U} \coloneqq 0, \mathcal{V} \coloneqq 0$  
3: for  $t = 1$  to  $T$  do  
4: Update  $\mathcal{U},\mathcal{V}$  using Eq. 13;  
5: Update  $\mathcal{L}$  and  $\mathcal{S}$  using Eq. 5 and Eq. 6;  
6: // Update  $\mathcal{L}$  using TT-truncation  
7:  $\widehat{\mathcal{L}}\gets \mathbf{trunc}_{\mathcal{P}}(\mathcal{L} + \mathcal{U})$  
8: // Update  $\mathcal{S}$  using projection  
9:  $\widehat{\mathbf{S}}\gets \mathbf{proj}_{\mathcal{O}}(\mathcal{S} + \mathcal{V})$  
10: end for  
11: Decompose  $\mathcal{L}$  to TT-cores  $\{\mathcal{C}_j^d\}_{j=0}$ ;  
12: Fine-tune model with  $\{\mathcal{C}\}_{j=0}^{d}$  and  $\mathcal{S}$ .

Hyperparameter. All the experiments are conducted using SGD optimizer with batch size, momentum and weight decay as 128, 0.9 and 0.0005, respectively. The learning rates adopted in the optimization and fine-tuning process are set as 0.1 and 0.005, respectively. Within the total 180 epochs, the learning rate is divided by 5 at epoch 54, 108, 144 and 171 gradually. The entire training procedure is performed on RTX 3090 GPUs with Pytorch 1.8.1.

# 5.2 EVALUATION

# AND COMPARISON ON CIFAR-10 DATASET

Table 1 shows the evaluation results of the compressed ResNet-20, ResNet-56 and DenseNet40 models on CIFAR-10 dataset. For each model, we compare the performance of our proposed SPARK with several types of compression methods, including decomposition-only  $(L)$ , pruning-only  $(S)$ , first-pruning-then

decomposition  $(L(S))$ , and layer-wise either-pruning-or-decomposition  $(S / L)$ .

ResNet-20. For ResNet-20 model, the proposed SPARK solution can bring  $1.25\%$  accuracy increase over baseline model with  $70.4\%$  and  $71.1\%$  model size and FLOPs reductions, respectively. With even more aggressive compression strategy aiming  $85.3\%$  smaller model size and  $86.1\%$  fewer FLOPs, our approach can still enable  $0.88\%$  higher accuracy than the original ResNet-20 model.

ResNet-56. For ResNet-56 model, our approach can bring  $1.02\%$  accuracy increase over baseline model with  $37.5\%$  and  $39.3\%$  model size and FLOPs reductions, respectively. When we perform more aggressive compression with  $65.6\%$  and  $66.0\%$  fewer parameters and computations, our compressed ResNet-56 can still enjoy  $0.64\%$  higher accuracy than the original uncompressed model, thereby exhibiting very superior performance than other state-of-the-art DNN compression methods.

DenseNet-40. For DenseNet-40 model, our proposed sparsity/low-rankness co-exploration can bring  $0.16\%$  accuracy increase over the baseline model with  $52.4\%$  and  $61.3\%$  model size and FLOPs reductions, respectively. Moreover, with further higher compression effect, our SPARK approach can still enable  $0.07\%$  higher accuracy than the original uncompressed model with  $65.3\%$  and  $74.5\%$  fewer parameters and computations, respectively; while the existing approaches suffer accuracy loss with even lower compression ratio.

# 5.3 EVALUATION AND COMPARISON ON IMAGENET DATASET

Table 2 summarizes the compression performance of our approach and other existing works for ResNet-50 on ImageNet dataset. It is seen that our SPARK solution can bring  $0.52\%$  accuracy increase over baseline model with  $48.7\%$  fewer parameters. When targeting for generating more compact model, our approach can still achieve high performance – it only has  $0.25\%$  accuracy drop with  $58.6\%$  model size reduction, which means it shows better test accuracy than its counterparts with even higher compression ratio.

# 5.4 DISCUSSION & ANALYSIS

To obtain deep understanding of our proposed approach, we also perform some empirical analysis and ablation studies on the co-exploration procedure. The details are referred to Appendix A.

# 6 CONCLUSION

In this paper we propose to systematically co-exlore DNN low-rankness and sparsity for efficient model compression. By performing comprehensive analysis on critical design factors, we propose

Table 1: Experimental results on CIFAR-10 dataset. Here “L” denotes low-rank decomposition and “S” denotes sparsification (pruning). Notice that no prior tensor decomposition work reports performance for compressing ResNet-56 and DenseNet-40.  

<table><tr><td rowspan="2">Compression Method</td><td rowspan="2">Type</td><td rowspan="2">Decomp. Format</td><td colspan="3">Top-1 Accuracy (%)</td><td rowspan="2">Params. ↓(%)</td><td rowspan="2">FLOPs ↓(%)</td></tr><tr><td>Baseline</td><td>Comp.</td><td>Δ</td></tr><tr><td colspan="8">ResNet-20</td></tr><tr><td>PSTRN (Li et al. (2021a))</td><td>L</td><td>Tensor</td><td>91.25</td><td>90.80</td><td>-0.45</td><td>55.6</td><td>N/A</td></tr><tr><td>PSTRN (Li et al. (2021a))</td><td>L</td><td>Tensor</td><td>91.25</td><td>89.30</td><td>-1.95</td><td>85.2</td><td>N/A</td></tr><tr><td>TRP (Xu et al. (2020))</td><td>L</td><td>Matrix</td><td>91.74</td><td>90.88</td><td>-0.86</td><td>48.1</td><td>51.0</td></tr><tr><td>SVDT (Yang et al. (2020))</td><td>L</td><td>Matrix</td><td>90.93</td><td>90.97</td><td>+0.04</td><td>N/A</td><td>54.5</td></tr><tr><td>Hinge (Li et al. (2020))</td><td>S/L</td><td>Matrix</td><td>92.54</td><td>91.84</td><td>-0.70</td><td>55.5</td><td>54.5</td></tr><tr><td>SCOP (Tang et al. (2020))</td><td>S</td><td>N/A</td><td>92.22</td><td>90.75</td><td>-1.47</td><td>56.3</td><td>55.7</td></tr><tr><td>FPGM (He et al. (2019))</td><td>S</td><td>N/A</td><td>92.20</td><td>90.44</td><td>-1.76</td><td>51.0</td><td>54.0</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>91.25</td><td>92.50</td><td>+1.25</td><td>70.4</td><td>71.1</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>91.25</td><td>92.13</td><td>+0.88</td><td>85.3</td><td>86.1</td></tr><tr><td colspan="8">ResNet-56</td></tr><tr><td>TRP (Xu et al. (2020))</td><td>L</td><td>Matrix</td><td>93.14</td><td>92.77</td><td>-0.37</td><td>N/A</td><td>56.7</td></tr><tr><td>HRank (Lin et al. (2020))</td><td>S</td><td>N/A</td><td>93.26</td><td>93.52</td><td>+0.26</td><td>16.8</td><td>29.3</td></tr><tr><td>HRank (Lin et al. (2020))</td><td>S</td><td>N/A</td><td>93.26</td><td>93.17</td><td>-0.09</td><td>42.4</td><td>50.0</td></tr><tr><td>SVDT (Yang et al. (2020))</td><td>L</td><td>Matrix</td><td>93.28</td><td>93.67</td><td>+0.39</td><td>N/A</td><td>63.0</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>93.33</td><td>93.87</td><td>+0.54</td><td>36.5</td><td>42.4</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>93.33</td><td>93.64</td><td>+0.31</td><td>48.2</td><td>52.0</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>93.27</td><td>94.29</td><td>+1.02</td><td>37.5</td><td>39.3</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>93.27</td><td>93.91</td><td>+0.64</td><td>65.6</td><td>66.0</td></tr><tr><td colspan="8">DenseNet-40</td></tr><tr><td>HRank (Lin et al. (2020))</td><td>S</td><td>N/A</td><td>94.81</td><td>94.24</td><td>-0.57</td><td>36.5</td><td>40.8</td></tr><tr><td>HRank (Lin et al. (2020))</td><td>S</td><td>N/A</td><td>94.81</td><td>93.68</td><td>-1.13</td><td>53.8</td><td>61.0</td></tr><tr><td>Hinge (Li et al. (2020))</td><td>S/L</td><td>Matrix</td><td>94.74</td><td>94.67</td><td>-0.07</td><td>27.5</td><td>44.4</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>94.81</td><td>94.67</td><td>-0.14</td><td>51.9</td><td>47.0</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>94.81</td><td>94.40</td><td>-0.41</td><td>64.4</td><td>60.4</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>94.81</td><td>94.97</td><td>+0.16</td><td>52.4</td><td>61.3</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>94.81</td><td>94.88</td><td>+0.07</td><td>65.3</td><td>74.5</td></tr></table>

Table 2: Experimental results on ImageNet dataset. Here “L” denotes low-rank decomposition and “S” denotes sparsification (pruning). Notice that no prior tensor decomposition work reports performance for compressing ResNet-50.  

<table><tr><td rowspan="2">Compression Method</td><td rowspan="2">Type</td><td rowspan="2">Decomp. Format</td><td colspan="3">Top-1 Accuracy (%)</td><td colspan="3">Top-5 Accuracy (%)</td><td rowspan="2">Params. ↓(%)</td></tr><tr><td>Base.</td><td>Comp.</td><td>Δ</td><td>Base.</td><td>Comp.</td><td>Δ</td></tr><tr><td colspan="10">ResNet-50</td></tr><tr><td>TRP (Xu et al. (2020))</td><td>L</td><td>Matrix</td><td>75.90</td><td>74.06</td><td>-1.84</td><td>92.70</td><td>92.07</td><td>-0.63</td><td>44.4</td></tr><tr><td>HRank(Lin et al. (2020))</td><td>S</td><td>N/A</td><td>76.15</td><td>74.98</td><td>-1.17</td><td>92.87</td><td>92.33</td><td>-0.54</td><td>36.7</td></tr><tr><td>SCOP (Tang et al. (2020))</td><td>S</td><td>N/A</td><td>76.15</td><td>75.95</td><td>-0.20</td><td>92.87</td><td>92.79</td><td>-0.08</td><td>42.8</td></tr><tr><td>SVDT (Yang et al. (2020))</td><td>L</td><td>Matrix</td><td>N/A</td><td>N/A</td><td>N/A</td><td>91.91</td><td>91.97</td><td>+0.06</td><td>30.6</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>76.15</td><td>75.59</td><td>-0.56</td><td>92.87</td><td>92.64</td><td>-0.23</td><td>48.4</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>76.13</td><td>76.65</td><td>+0.52</td><td>92.86</td><td>93.14</td><td>+0.28</td><td>48.7</td></tr><tr><td>TRP (Xu et al. (2020))</td><td>L</td><td>Matrix</td><td>75.90</td><td>72.69</td><td>-3.21</td><td>92.70</td><td>91.41</td><td>-1.29</td><td>56.5</td></tr><tr><td>HRank (Lin et al. (2020))</td><td>S</td><td>N/A</td><td>76.15</td><td>71.98</td><td>-4.17</td><td>92.87</td><td>91.01</td><td>-1.86</td><td>46.0</td></tr><tr><td>SCOP (Tang et al. (2020))</td><td>S</td><td>N/A</td><td>76.15</td><td>75.26</td><td>-0.89</td><td>92.87</td><td>92.53</td><td>-0.34</td><td>51.8</td></tr><tr><td>Hinge (Li et al. (2020))</td><td>S/L</td><td>Matrix</td><td>76.15</td><td>74.70</td><td>-1.45</td><td>N/A</td><td>N/A</td><td>N/A</td><td>53.5</td></tr><tr><td>CC (Li et al. (2021b))</td><td>L(S)</td><td>Matrix</td><td>76.15</td><td>74.54</td><td>-1.61</td><td>92.87</td><td>92.25</td><td>-0.62</td><td>58.6</td></tr><tr><td>SPARK (Ours)</td><td>L+S</td><td>Tensor</td><td>76.13</td><td>75.29</td><td>-0.84</td><td>92.86</td><td>92.61</td><td>-0.25</td><td>58.6</td></tr></table>

SPARK, a unified compression framework that can capture model sparsity and low-rankness simultaneously and efficiently. Evaluation results show that our proposed approach can bring significant model size and computational cost reductions while still preserving high model accuracy.

# REFERENCES

Misha Denil, Babak Shakibi, Laurent Dinh, Marc'Aurelio Ranzato, and Nando De Freitas. Predicting parameters in deep learning. arXiv preprint arXiv:1306.0543, 2013.  
Abhimanyu Dubey, Moitreya Chatterjee, and Narendra Ahuja. Coreset-based neural network compression. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 454-470, 2018.  
Shangqian Gao, Feihu Huang, Weidong Cai, and Heng Huang. Network pruning via performance maximization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9270-9280, 2021.  
Xitong Gao, Yiren Zhao, Łukasz Dudziak, Robert Mullins, and Cheng-zhong Xu. Dynamic channel pruning: Feature boosting and suppression. In International Conference on Learning Representations, 2019.  
Ruihao Gong, Xianglong Liu, Shenghu Jiang, Tianxiang Li, Peng Hu, Jiazhen Lin, Fengwei Yu, and Junjie Yan. Differentiable soft quantization: Bridging full-precision and low-bit neural networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4852–4861, 2019.  
Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. arXiv preprint arXiv:1608.04493, 2016.  
Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. arXiv preprint arXiv:1510.00149, 2015a.  
Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in Neural Information Processing Systems, 28:1135-1143, 2015b.  
Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4340-4349, 2019.  
Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin. Compression of deep convolutional neural networks for fast and low power mobile applications. In International Conference on Learning Representations, 2016.  
Chong Li and CJ Shi. Constrained optimization based low-rank approximation of deep neural networks. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 732-747, 2018.  
Nannan Li, Yu Pan, Yaran Chen, Zixiang Ding, Dongbin Zhao, and Zenglin Xu. Heuristic rank selection with progressively searching tensor ring network. Complex & Intelligent Systems, pp. 1-15, 2021a.  
Yawei Li, Shuhang Gu, Christoph Mayer, Luc Van Gool, and Radu Timofte. Group sparsity: The hinge between filter pruning and decomposition for network compression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8018-8027, 2020.  
Yuchao Li, Shaohui Lin, Jianzhuang Liu, Qixiang Ye, Mengdi Wang, Fei Chao, Fan Yang, Jincheng Ma, Qi Tian, and Rongrong Ji. Towards compact cnns via collaborative compression. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6438-6447, 2021b.  
Siyu Liao, Chunhua Deng, Miao Yin, and Bo Yuan. Doubly residual neural decoder: Towards low-complexity high-performance channel decoding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 8574-8582, 2021.  
Mingbao Lin, Rongrong Ji, Yan Wang, Yichen Zhang, Baochang Zhang, Yonghong Tian, and Ling Shao. Hrank: Filter pruning using high-rank feature map. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1529-1538, 2020.

Yuzhe Ma, Ran Chen, Wei Li, Fanhua Shang, Wenjian Yu, Minsik Cho, and Bei Yu. A unified approximation framework for compressing and accelerating deep neural networks. In 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), pp. 376-383. IEEE, 2019.  
Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. Advances in Neural Information Processing Systems, 28:442-450, 2015.  
Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. arXiv preprint arXiv:2106.02034, 2021.  
Cheng Tai, Tong Xiao, Yi Zhang, Xiaogang Wang, et al. Convolutional neural networks with low-rank regularization. In International Conference on Learning Representations, 2016.  
Yehui Tang, Yunhe Wang, Yixing Xu, Dacheng Tao, Chunjing Xu, Chao Xu, and Chang Xu. Scop: Scientific control for reliable neural network pruning. In Advances in Neural Information Processing Systems, volume 33, pp. 10936-10947, 2020.  
Wenqi Wang, Yifan Sun, Brian Eriksson, Wenlin Wang, and Vaneet Aggarwal. Wide compression: Tensor ring nets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9329-9338, 2018.  
Zi Wang, Chengcheng Li, and Xiangyang Wang. Convolutional neural network pruning with structural redundancy reduction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14913-14922, 2021.  
Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. Advances in neural information processing systems, 29:2074-2082, 2016.  
Yuhui Xu, Yuxi Li, Shuai Zhang, Wei Wen, Botao Wang, Yingyong Qi, Yiran Chen, Weiyao Lin, and Hongkai Xiong. Trp: Trained rank pruning for efficient deep neural networks. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pp. 977–983, 2020.  
Huanrui Yang, Minxue Tang, Wei Wen, Feng Yan, Daniel Hu, Ang Li, Hai Li, and Yiran Chen. Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 678-679, 2020.  
Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. On compressing deep models by low rank and sparse decomposition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7370-7379, 2017.
