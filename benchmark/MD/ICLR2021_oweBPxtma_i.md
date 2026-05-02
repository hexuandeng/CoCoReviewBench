# A SELF-EXPLANATORY METHOD FOR THE BLACK PROBLEM ON DISCRIMINATION PART OF CNN

Anonymous authors

Paper under double-blind review

# ABSTRACT

Convolution neural networks (CNNs) have surpassed human's abilities in some specific tasks. However, they are considered difficult to understand and explain. Recently, the black box problem of CNN, especially concerning the discrimination part, has been studied by different scientific communities. Many methods were proposed for extracting an interpretable model from the discrimination part, which can explain the prediction of the part. However, it is hard for the interpretable models to approximate the discrimination part because of the tradeoff problem between interpretability performance and generalization performance of the discrimination part. We suppose the tradeoff problem is mainly attributed to the fact that the sufficient and necessary condition for the consistent convergence of the both performances is hard to be guaranteed by tradition learning algorithm. So the tradeoff problem could be solved by shrinking the distance between the interpretable model and the discrimination part. This paper firstly introduces a Markov random field model (MRF), namely Deep Cognitive Learning Model(DCLM), which explains the causal relationship between the features(the weight matrixes in the first layer of the discrimination part of CNN can capture) and the output result of the discrimination part of CNN. A greedy algorithm is proposed for initiaitively extracting the DCLM from the discrimination part by solving a MAX-SAT problem. And then, a game process between two MAP inferences is implemented for shrinking an interpretation distance which can evaluate how close the discrimination part is to the DCLM. Finally, the proposed self-explanatory approach is evaluated by some contrastive experiments with certain baseline methods on some standard image processing benchmarks. These experiments indicate that the proposed method can improve the interpretability performance of the discrimination part without largely reducing its generalization performance, the generalization performance of the DCLM also can be improved and the discrimination part can be explained by the DCLM in real time during the training process after the interpretation distance converges.

# 1 INTRODUCTION

Convolution neural network(CNN) has surpassed human abilities in some specific tasks such as computer game and computer vision etc. However, they are considered difficult to understand and explain(Brandon, 2017), which leads to many problems in aspects of fairness, privacy leaking, reliability and robustness. Even wrong decision could possibly be made due to spurious correlations in training data, such as recognizing an object in an image by the properties of the background or lighting (Riccardo et al., 2018). Explanation technology is of immense help for companies to create safer, more trustable products, and to better manage any possible liability of them (Riccardo et al., 2018). Recently, the unexplainable problem of CNN, especially concerning the discrimination part, has been studied by different scientific communities. Many methods were proposed for extracting an interpretable model from the discrimination part, which can explain causal relationship between inputs and outputs of the discrimination part. However, because of data bias and noisy data in training data set the sufficient and necessary condition for the consistent convergence of an interpretability performance and the generalization performance of the discrimination part is difficult to be guaranteed in traditional machine learning methods. The interpretability performance is a distance between the discrimination part and its optimal interpretable model. So when a discrimination part has a well

generalization performance, it usually stay away from its optimal interpretable model, which leads to an incorrect interpretable model. We think that the tradeoff problem could be solved by the following procedure. Firstly, extracting an interpretable model which expresses causal relationships in the discrimination part in training process, and then, iteratively shrinking a distance between the model and the part, and amending them. It is because that the method not only can make the discrimination part approximate the interpretable model which can improve the interpretability performance of the part but also can make the interpretable model close to the part which makes it correctly express the causal relationship in the part.

Main contributions of this paper can be summarized as follows:

- An interpretable model, Deep Cognitive Learning Model(DCLM), is proposed to express the causal relationship in the discrimination part, and a greedy method is given for initia-tively extracting the DCLM from the discrimination part by solving its MAX-SAT problem.  
- A new game method is proposed to improve the interpretability performance of the discrimination part without largely reducing its generalization performance by iteratively shrinking a distance between the two models and amending the two models. At the same time, the generalization performance of the DCLM can also be improved.  
- Interpretable distance is proposed to evaluate a distance between the discrimination part and the interpretable model on the unexplainable problem.

# 2 RELATED WORK

There are usually two types of methods for the unexplainable problem of the discrimination part, such as post-hoc method and ante-hoc method (Holzinger et al., 2019). However, because ante-hoc method is a transparent modeling method (Arrietaa et al., 2020), it can not obtain an explanation about the discrimination part. So, the post-hoc method will be reviewed.

Early post-hoc method can obtain global explanations for a neural network by extracting an interpretable model. Some references (Craven & Shavlik, 1999; Krishnan et al., 1999; Boz, 2002; Johansson & Niklasson, 2009) proposed a few methods that can find a decision tree for explaining a neural network by maximizing the gain ratio and an estimation of the current model fidelity. Other references (Craven & Shavlik, 1994; Johansson & Niklasson, 2003; Augusta & Kathirvalavakumar, 2012; Sebastian et al., 2015; Zilke et al., 2016) proposed rule extraction methods for searching optimal interpretable rules from a neural network.

Recently, some feature relevance methods have become progressively more popular. Montavon et al.(Montavon et al., 2017) proposed a decomposition method from a network classification decision into contributions of its input elements based on deep Taylor decomposition. Shrikumar et al.(Shrikumar et al., 2016) proposed DeepLIFT which can compute importance scores in a multi-layer neural network by explaining the difference of the output from some reference output in terms of differences of the inputs from their reference inputs.

Some other works make complex black box model simpler. Che et al.(Che et al., 2017) proposed a simple distillation method called Interpretable Mimic Learning for extracting an interpretable simple model by gradient boosting trees. Thiagarajan et al.(Thiagarajan et al., 2016) build a Treeview representation of the complex model by hierarchical partitioning of the feature space. In addition, some references (Hinton et al., 2015; Bucila et al., 2006; Frosst & Hinton, 2017; Traore et al., 2019) proposed the distillation method of knowledge from an ensemble of models into a single model. Wu et al.(M. Wu, 2018) proposed a tree regularization method via knowledge distillation to represent the output feature space of a RNN based on a Multilayered perception. However, these methods can only solve the unexplainable problem of trained neural network or trained deep neural networks with explicit input characteristics. Wan et al.(Wan et al., 2020) constructed a decision tree using the last fully connection layer of the discrimination part of a CNN based on a prior structure.

In the paper, our goal is to improve the interpretability performance of multi full connection layers of CNN without hurting its generalization performance by initiatively extracting its logic relationships with no prior structure and finally obtain its explanation by these logic relationships.

# 3 DEEP COGNITIVE LEARNING MODEL

Convolution layers and pooling layers of a CNN constitute a feature extractor. The outputs from the feature extractor are the inputs to the fully connected layers of the CNN, namely feature maps,  $\tau_{1},\tau_{2},\dots,\tau_{k}$  where  $k$  is the number of feature maps. All these feature maps form a feature set  $\Gamma$ . The objective of the fully connected layers is to use the feature maps to classify the input image  $x$  of the CNN into a label. So, the fully connected layers compose a discrimination part of the CNN.

We suppose that the discrimination part should better be explained by the logic relationships between the features (the weight matrixes in the first layer of the part can capture) and its output vector. This is because that the relationships is the inherent relation of such CNN structure which is neither related to the input data nor related to the extent of the features' abstract. In order to express the relationships, deep cognitive learning model (DCLM) is proposed. The DCLM is shown in Fig.1.

![](images/ad6bfe30fab7ef2959768c0032daf8d7ead5b1763d51cf1d7a4974feb8febb63.jpg)  
Figure 1: Deep Cognitive Learning Model

![](images/444e26c890240f73577b5fed723b409cbe5ace653a4984ff658e0bdae4c22448.jpg)  
Figure 2: A disjunction relation in DCLM

The DCLM consists of three layers of nodes: feature predicate layer, disjunction layer and decision result predicate layer. The top layer is the feature predicate layer which consists of feature predicate nodes. Every feature predicate node has a feature predicate function  $Z(\Gamma)$  which expresses whether a neuron in the first fully connected layer of discrimination part of the CNN can capture a feature. The feature predicate function  $Z(\Gamma)$  is defined as follows.

$$
Z (\Gamma) = \left\{ \begin{array}{l l} 1, & \frac {\tau_ {i} * w _ {i}}{\| \tau_ {i} \| \| w _ {i} \|} > 0 \text {a n d} \tau_ {i} \in \Gamma , \\ \operatorname {n u l l}, & \frac {\tau_ {i} * w _ {i}}{\| \tau_ {i} \| \| w _ {i} \|} = 0 \text {a n d} \tau_ {i} \in \Gamma , \\ 0, & \text {o t h e r w i s e .} \end{array} \right. \tag {1''}
$$

where  $i \in \{1, 2, \dots, k\}$ ,  $\tau_i$  is the  $i$ th feature map in  $\Gamma$ ,  $w_i$  is its corresponding weight vector on the first fully connection layer, namely feature-capture map and  ${}^* *$  is a convolution.

The bottom layer is the decision result predicate layer which includes all decision result predicate nodes. Every decision result predicate node has a decision result predicate function which expresses whether the output  $y$  of an output neuron of the discrimination part is greater than 0. It is defined as following as

$$
D (y) = \left\{ \begin{array}{l l} 1, & y > 0, \\ 0, & \text {o t h e r w i s e .} \end{array} \right. \tag {2}
$$

All feature predicate nodes and every decision result predicate node are connected to one or more disjunction nodes in the middle layer, namely as disjunction layer, with true or false edges. Every disjunction node represents a disjunction relation between all feature predicate nodes and a decision result predicate node. A disjunction relation is expressed by a disjunctive normal form. If a predicate node is connected to the disjunction node by a false edge, its predicate function follows after a nonoperator in the disjunctive normal form. Potential function of a disjunctive normal form can be obtained by using the Lukasiewicz method(Giles, 1975).

$$
\phi_ {c} (y) = \min  (1, T (\Gamma , y)) \tag {3}
$$

where  $T(\Gamma, y) = \sum_{j=1}^{N} \{a_j[1 - Z_j(\Gamma)] + (1 - a_j)Z_j(\Gamma)\} + D(y)$  and  $N$  is the number of the feature predicate nodes. If  $a_j = 1$ , there is a false edge between predicate node and a disjunction node. If  $a_j = 0$ , there is a true edge.

In Fig.2 suppose a DCLM includes a disjunction relation  $\forall \Gamma \neg Z_1(\Gamma) \lor Z_2(\Gamma) \lor Z_3(\Gamma) \lor \neg Z_4(\Gamma) \lor D(y)$  which is equivalent to  $\forall \Gamma [Z_1(\Gamma) \land \neg Z_2(\Gamma) \land \neg Z_3(\Gamma) \land Z_4(\Gamma) \rightarrow D(y)]$ , where  $Z_j()$  is a feature predicate function and  $y$  is an output of an output neuron of CNN. Its potential function is  $\phi_c(y) = \min(1, \sum_{i=1}^{4} \{a_j[1 - Z_j(\Gamma)] + (1 - a_j)Z_j(\Gamma)\} + D(y))$  where  $a_1 = a_4 = 1$  and  $a_2 = a_3 = 0$ .

The conditional probability distribution that a ground DCLM is true is

$$
p \left(y _ {d c l m}, \Gamma\right) = \frac {1}{\Xi} \exp \left(\frac {\sum_ {i = 1} ^ {G} \lambda_ {i} \phi_ {c i} \left(y _ {d c l m , i}\right)}{\sum_ {i = 1} ^ {G} \lambda_ {i}}\right) \tag {4}
$$

where  $G$  is the number of all ground formulas,  $\Xi = \sum_{\Gamma \in \mathbb{F}}\exp \left(\frac{\sum_{i = 1}^{G}\lambda_{i}\phi_{ci}(y_{dclm,i})}{\sum_{i = 1}^{G}\lambda_{i}}\right)$  is a partition function,  $y_{dclm} = (y_{dclm,1},y_{dclm,2},\dots,y_{dclm,G})$ ,  $y_{dclm,i}$  is an output value of an output neuron of the CNN and  $\lambda_{i}$  is a weighted value of the ith ground formula.

By maximizing its likelihood function, the optimal  $y_{dclm}$ ,  $a_i$  and  $\lambda_i$  in the DCLM can be obtained.

$$
C (\Gamma) = \arg \max  _ {y _ {d c l m}, a _ {i}, \lambda_ {i}} [ \log p (y _ {d c l m}, \Gamma) ] = \arg \max  _ {y _ {d c l m}, a _ {i}, \lambda_ {i}} \left(\frac {\sum_ {i} \lambda_ {i} \phi_ {c i} \left(y _ {d c l m , i}\right)}{\sum_ {i} \lambda_ {i}}\right) \tag {5}
$$

For extracting a DCLM, a Maximum A Posterior(MAP) algorithm on the Maximum Satisfiability Problem (MAX-SAT) was designed, which is shown in the appendix.

# 4 EVALUATION OF INTERPRETABILITY PERFORMANCE

We consider that if the shape of function curve of the discrimination part of a CNN is similar as that of its optimal interpretable model, it can be very easily understood by humans and has well interpretability performance. Therefore, the interpretable performance of the discrimination part can be measured by the shape similarity between it and the optimal interpretable model. We posit that given the same input data set, the similarity may be measured by variance of differences between outputs of the both models. It can tentatively be named interpretation distance. It is easily proved that the smaller the interpretation distance is, the more similar the shape of the discrimination part is to the shape of the optimal interpretable model and the better the interpretability performance of the discrimination part would be.

Definition 1 If  $X$  is a compact metric space and  $\nu$  is a Borel measure in  $X$ , such as Lebesgue measure or marginal measures, in  $\mathcal{L}_{\nu}^{2}(X)$ , a square integrable function space on  $X$ , the interpretation distance,  $\phi_d(P^*, f)$ , between a discrimination part  $f(x)$  and its optimal interpretable model  $P^*(x)$  is

$$
\phi_ {d} \left(P ^ {*}, f\right) = \int_ {Z} (f (x) - P ^ {*} (x) - \mu^ {P ^ {*}} (f)) ^ {2} d \nu \tag {6}
$$

where

$$
\mu^ {P ^ {*}} (f) = \int_ {Z} (f (x) - P ^ {*} (x)) d \nu \tag {7}
$$

In the definition, the optimal interpretable model can have many forms, such as DCLM, other probabilistic graphical model, decision tree, decision rules and so on.

# 5 GAME BETWEEN A DCLM AND THE DISCRIMINATION PART OF A CNN

As discussed above, when the shapes of the discrimination part of a CNN and its optimal interpretable model are enough similar, the discrimination part has well interpretability performance. However, its generalization performance will tend to decrease. This is mainly attributed to the fact that the sufficient and necessary condition for the consistent convergence of the two performances,  $\phi_d(P^*,f^*) = 0(f^*$  is the optimal predication model), is difficult to be guaranteed. For proving the conclusion, we focus on a neuron of a CNN. From the foot we may judge of Hercules. Because every input channel  $f(x)$  of the neuron can be seen as a kernel function  $K(x,w)(w$  is a weight vector including a bias of the neuron), it may span a kernel Hilbert space  $\mathcal{H}_K = \{f(x)\in \mathcal{L}_\nu^2 (X)\mid f(x) = K(x,w) = \sum_{k = 1}\phi_k(w)\phi_k(x)\} .$ $\mathcal{H}_K$  is a linear function set on

Algorithm 1 Game between DCLM and the discrimination part of a CNN(It's time complexity(TC) is  $O(N + M)$  where  $N$  is TC of training CNN,  $M$  is TC of construction of Logic Net.)

Input: data  $X$ , target  $Y_{t}$   
initialize Logic Net  $LN$ , CNN  $CN$   
for  $i = 1$  to batch - size do  
[ FMs, f_{nn} = CN(X[i]). ]  
[ y_{dclm} = LN(FMs, Y_t[i]). ]  
[ CN = Updata_CNNGradient(y_{dclm}, f_{nn}, y_t[i]) ]  
[ FMs, FCMs, f_{nn} = CN(X[i]). ]  
[ LN = ConstructionLogicNet(FMs, FCMs, f_{nn}) ]  
end for

$\mathcal{L}_{\nu}^{2}(X)$ . However, because of data bias and noisy data in training data set, its optimal interpretable model which obey to human's cognition usually is a nonlinear function in  $\mathcal{L}_{\nu}^{2}(X)$  in the majority of engineering applications. Because continuous linear functional set is nowhere dense in  $\mathcal{L}_{\nu}^{2}(X)$ . So the sufficient and necessary condition is hard to be satisfied and the tradeoff problem always exists. This is also the case with the discrimination part of a CNN. However its optimal interpretable model  $P^{*}$  and its optimal discrimination part are unknown. Therefore, for the tradeoff problem, in training process, extracting an interpretable model from the discrimination part and then iteratively reducing the interpretation distance between the two models may be a feasible solution. A detailed discussion about the problem can be found in the appendix.

To avoid to reduce the generalization performance, the maximum probability  $p(w \mid X, y_t)$  should be guaranteed, where  $X$  is a training sample,  $w$  is parameter set of the CNN and  $y_t$  is target vector of  $X$ .

$$
p (w \mid X, y _ {t}) = \frac {p (w \mid X) p \left(y _ {t} \mid w , X\right)}{p \left(y _ {t} \mid X\right)} \propto p (w) p \left(y _ {t} \mid w, X\right) \tag {8}
$$

where  $p(y_{t} \mid w, X) = \int p(y_{t} \mid f, w, X) \int p(f \mid y_{dclm}, w, X) p(y_{dclm} \mid w, X) dy_{dclm} df$ .

When the DCLM is known,  $y_{dclm}^{*}$  is its optimal solution and  $p(y_{dclm}^{*} \mid w, X) = 1$ . Then

$$
\int p (f \mid y _ {d c l m}, w, X) p \left(y _ {d c l m} \mid w, X\right) d y _ {d c l m} = p \left(f \mid y _ {d c l m} ^ {*}, w, X\right) \tag {9}
$$

Similarly, known the input  $X$  and  $w$ ,  $f_{nn}$  is the optimal solution of the CNN.

$$
p \left(y _ {t} \mid w, X\right) = p \left(y _ {t} \mid f _ {n n}, w, X\right) p \left(f _ {n n} \mid y _ {d c l m} ^ {*}, w, X\right) \tag {10}
$$

If  $w$  and  $X$  are given and the loss function  $\phi_r(y_t, f_{nn}) = -\frac{1}{2} \sum_l |y_t - f_{nn}|^2$ , the conditional probability distribution function

$$
p \left(y _ {t} \mid f _ {n n}, w, X\right) = \frac {\exp \left(\phi_ {r} \left(y _ {t} , f _ {n n}\right)\right)}{\Xi_ {1}} \tag {11}
$$

Meanwhile,

$$
p \left(f _ {n n} \mid y _ {d c l m} ^ {*}, w, X\right) = \frac {\exp \left(- \phi_ {d} \left(y _ {d c l m} ^ {*} , f _ {n n}\right)\right)}{\Xi_ {2}} \tag {12}
$$

where  $\Xi_1$  and  $\Xi_2$  are partition functions. Then by maximizing a likelihood function of  $p(w\mid X,y_t)$  the optimal  $w$  can be obtained. In particular, assuming that  $w$  follows Gaussian distribution, we get:

$$
C _ {w} (X, y _ {t}) = \arg \max  _ {w} \left[ - \frac {\alpha}{2} \| w \| ^ {2} + \phi_ {r} \left(y _ {t}, f _ {n n}\right) - \phi_ {d} \left(y _ {d c l m} ^ {*}, f _ {n n}\right) \right] \tag {13}
$$

where  $\alpha$  is a meta-parameter determined by the variance of the selected Gaussian distributions. Turn it into a minimization problem:

$$
C _ {w} (X, y _ {t}) = \arg \min  _ {w} \left[ \frac {\alpha}{2} \| w \| ^ {2} - \phi_ {r} \left(y _ {t}, f _ {n n}\right) + \phi_ {d} \left(y _ {d c l m} ^ {*}, f _ {n n}\right) \right] \tag {14}
$$

The iterative optimization algorithm is shown as follows:

# 6 EXPERIMENTAL VERIFICATION

We designed some experiments to verify the effectiveness of the proposed method. The first experiment verified whether the self-explanatory method could improve the interpretability performance of the CNN without sacrificing its generalization performance. The second experiment verified whether the proposed method can tend towards stability and convergence in the game process.

In the experiments, the structure of the CNN3(includes 3 convolution layers, 3 MaxPooling layers, 3 fully connect layers(FCLs) and 1 output layer), CNN5(includes 5 convolution layers, 5 MaxPooling layers, 3 FCLs and 1 output layer) and CNN8(includes 8 convolution layers, 8 MaxPooling layers, 3 FCLs and 1 output layer) were used. The final structures obtained after the training on these CNNs by the proposed method are named as CNN3-DCLM, CNN5-DCLM and CNN8-DCLM respectively. All experiments used Mnist(Lecun et al., 1998), FashionMnist(Zalando, 2017) and Emmist(Cohen et al., 2017) benchmark data sets. All algorithms were implemented in Python using the Pytorch library(Paszke et al., 2019). All experiments ran on a server with Intel Xeon 4110(2.1GHz) Silver Processor, 20GB RAM and Nvidia Telsa T4.

Experiment 1: Performance verification of the proposed method on CNN. The experiment also compared soft decision tree(SDT) (Frosst & Hinton, 2017) based on trained CNN3,CNN5 and CNN8,namely "CNN3-SDT", "CNN5-SDT" and "CNN8-SDT" respectively. These baseline methods also include SDT (Frosst & Hinton, 2017) based on trained CNN3-DCLM,CNN5-DCLM and CNN8-DCLM, namely "CNN3\*-SDT", "CNN5\*-SDT" and "CNN8\*-SDT" respectively. These indicators of all algorithms on all testing data sets were shown in Table 1. Some values are  ${}^{*}$  ,which indicates that these results do not exist.

It is observed in Table 1 that the accuracy of all CNN-DCLMs are higher than the two interpretable models, such as SDT and DCLM, on all benchmark data sets and are around 1.4 percentage points lower than those of CNN. But it is worth noticing that on the most of data sets the interpretation distances of all CNN-DCLMs are around 5 percentage points lower than interpretation distances between the majority SDTs and their CNNs except CNN3-SDT on Emmist data set. These might prove that the self-explanatory method can improve the interpretability performance of the discrimination part of a CNN without largely reducing its generalization performance.

On Mnist data set the accuracies of the DCLMs are only 0.7 percentage points lower than those of the SDTs on CNN5-SDT and CNN8-SDT except CNN3-SDT. Though for Emmist data set and FashionMnist data set, we can find that the accuracies of the DCLMs are 2 percentage points lower than those of the SDTs on CNN3-SDT, we also can find that on the two data sets the accuracies of the DCLMs are around 4.3 percentage points higher than those of the SDTs on CNN5-SDT and CNN8-SDT. It is mainly because that the feature extract parts of CNN5 and CNN8 output more abstract feature maps than CNN3 for Emmist data set and FashionMnist data set. The abstract feature maps can be easily expressed without distortion by feature predicates of the DCLM, so it does not impede the generation of the DCLM.

Experiment 2: Convergence test of the proposed method We designed the experiments to demonstrate convergence of the proposed method. CNN3, CNN5 and CNN8 were used for comparing with CNN3-DCLM, CNN5-DCLM, and CNN8-DCLM respectively. Every training works out 25 epochs. All results were measured at every epoch and shown as the four figures, Fig.3, Fig.4, Fig.6 and Fig.5. Every figure includes nine subplots. The three subplots on the left column were shown for the experiments on Mnist data set. These subplots on the middle column were for FashionMnist data set and these subplots on the right column were for Emmist data set.

In Fig.3, the accuracies of the DCLMs and the CNN-DCLMs of every epoch in the game process were shown. From these figures, it is obvious that accuracies of the DCLMs and these CNN-DCLMs steadily increase in the early stage. In the next stage, their accuracies tend to be stable. This reflects that the game method did not affect the improvement of the generalization performances of these DCLMs and these CNNs.

Fig.4 shows the interpretation distances between the DCLMs and the CNNs and CNN-DCLMs. As seen from these subplots, the interpretation distances of CNNs not involved in the game are greater than those of CNN-DCLMs at the most of the epochs, especially by the end of the training. The results indicate that the game method can effectively improve the interpretability performance of CNN-DCLMs.

Table 1: Classification accuracies and interpretation distances for three game methods on DCLM-CNN.  

<table><tr><td colspan="4">DATA SET: MNIST</td></tr><tr><td>GAME METHOD</td><td>ACCURACY</td><td>ACCURACY OF SDT OR DCLM</td><td>INTERPRETATION DISTANCE</td></tr><tr><td>CNN3</td><td>0.988±0.012</td><td>—</td><td>—</td></tr><tr><td>CNN3-SDT</td><td>—</td><td>0.970±0.017</td><td>0.005±0.0039</td></tr><tr><td>CNN3*-SDT</td><td>—</td><td>0.953±0.014</td><td>0.008±0.0043</td></tr><tr><td>CNN3-DCLM</td><td>0.983±0.016</td><td>0.970±0.018</td><td>0.002±0.0023</td></tr><tr><td>CNN5</td><td>0.988±0.013</td><td>—</td><td>—</td></tr><tr><td>CNN5-SDT</td><td>—</td><td>0.968±0.017</td><td>0.005±0.0046</td></tr><tr><td>CNN5*-SDT</td><td>—</td><td>0.959±0.020</td><td>0.006±0.0032</td></tr><tr><td>CNN5-DCLM</td><td>0.983±0.015</td><td>0.962±0.022</td><td>0.002±0.0024</td></tr><tr><td>CNN8</td><td>0.991±0.012</td><td>—</td><td>—</td></tr><tr><td>CNN8-SDT</td><td>—</td><td>0.978±0.012</td><td>0.003±0.0019</td></tr><tr><td>CNN8*-SDT</td><td>—</td><td>0.974±0.014</td><td>0.006±0.0039</td></tr><tr><td>CNN8-DCLM</td><td>0.987±0.013</td><td>0.970±0.023</td><td>0.002±0.0022</td></tr></table>

<table><tr><td colspan="4">DATA SET: EMNIST</td></tr><tr><td>GAME METHOD</td><td>ACCURACY</td><td>ACCURACY OF SDT OR DCLM</td><td>INTERPRETATION DISTANCE</td></tr><tr><td>CNN3</td><td>0.982±0.014</td><td>—</td><td>—</td></tr><tr><td>CNN3-SDT</td><td>—</td><td>0.949±0.021</td><td>0.007±0.0035</td></tr><tr><td>CNN3*-SDT</td><td>—</td><td>0.863±0.032</td><td>0.022±0.0088</td></tr><tr><td>CNN3-DCLM</td><td>0.974±0.013</td><td>0.911±0.041</td><td>0.010±0.0043</td></tr><tr><td>CNN5</td><td>0.985±0.016</td><td>—</td><td>—</td></tr><tr><td>CNN5-SDT</td><td>—</td><td>0.870±0.034</td><td>0.022±0.0094</td></tr><tr><td>CNN5*-SDT</td><td>—</td><td>0.942±0.026</td><td>0.082±0.0224</td></tr><tr><td>CNN5-DCLM</td><td>0.973±0.012</td><td>0.942±0.027</td><td>0.006±0.0038</td></tr><tr><td>CNN8</td><td>0.989±0.012</td><td>—</td><td>—</td></tr><tr><td>CNN8-SDT</td><td>—</td><td>0.872±0.031</td><td>0.025±0.0092</td></tr><tr><td>CNN8*-SDT</td><td>—</td><td>0.855±0.039</td><td>0.037±0.0092</td></tr><tr><td>CNN8-DCLM</td><td>0.974±0.011</td><td>0.923±0.034</td><td>0.007±0.0042</td></tr></table>

<table><tr><td>GAME METHOD</td><td>ACCURACY</td><td>ACCURACY OF SDT OR DCLM</td><td>INTERPRETATION DISTANCE</td></tr><tr><td>CNN3</td><td>0.905±0.029</td><td>—</td><td>—</td></tr><tr><td>CNN3-SDT</td><td>—</td><td>0.842±0.041</td><td>0.083±0.0355</td></tr><tr><td>CNN3*-SDT</td><td>—</td><td>0.744±0.049</td><td>0.058±0.0150</td></tr><tr><td>CNN3-DCLM</td><td>0.880±0.034</td><td>0.825±0.032</td><td>0.016±0.0028</td></tr><tr><td>CNN5</td><td>0.900±0.027</td><td>—</td><td>—</td></tr><tr><td>CNN5-SDT</td><td>—</td><td>0.792±0.041</td><td>0.064±0.0119</td></tr><tr><td>CNN5*-SDT</td><td>—</td><td>0.715±0.053</td><td>0.078±0.0205</td></tr><tr><td>CNN5-DCLM</td><td>0.883±0.032</td><td>0.819±0.036</td><td>0.016±0.0028</td></tr><tr><td>CNN8</td><td>0.905±0.027</td><td>—</td><td>—</td></tr><tr><td>CNN8-SDT</td><td>—</td><td>0.785±0.043</td><td>0.122±0.0397</td></tr><tr><td>CNN8*-SDT</td><td>—</td><td>0.791±0.043</td><td>0.053±0.0129</td></tr><tr><td>CNN8-DCLM</td><td>0.876±0.033</td><td>0.811±0.032</td><td>0.017±0.0037</td></tr></table>

From Fig.5 it is evident that all DCLMs of CNN3, CNN5 and CNN8 have been found to have stable entropies at the end of the game. On Mnist data set, the entropies finally converge between 120 and 145. On FashionMnist data set, the entropies finally converge between 125 and 136. On Emmist data set, the entropies finally converge between 100 and 120. The results indicate that the game algorithm can ensure that the DCLM converges to a stable state.

In Fig.6, the accuracies of the CNNs and the CNN-DCLMs of every epoch in the game process were shown. From these subplots, it can be seen that the accuracies of the CNNs and those of the CNN-DCLMs steadily increase in the early stage and accuracies of CNN-DCLMs are lower than

![](images/09836bec6fec5d2f01070d06c610e3a95fa61fe12b6d15d41cbf3f11dc5faa3d.jpg)

![](images/c023b94e4fc437903883695771a39768ea65e68e7ff8c101bb9e307e7ee2242d.jpg)

![](images/634bd437684abc2a9bafdc219038ec7304a2a6c3eb566f858c54f56f31efc20d.jpg)

![](images/1f31b58f70393bd022a3ee383b7e6dcb898f31941bc0abef5a971d426d79b0ef.jpg)

![](images/654a28038b48475b746a7ae36adbca41742d8e03d2bd9460686e7006bf5491fc.jpg)

![](images/b12be615345ac0816ba7f330f9290e0880852ab4a5f9a715fdb00aaab38fc844.jpg)

![](images/3559d480273e53ab6c9768d4531b8342626c6aea9102489ab7d4c296631ec0cd.jpg)

![](images/a723b4aef541a9558ff0539bec3721f42a5447c532e794de976edbc135ab24f3.jpg)

![](images/45e336a9bbf1612e4fd8e59711c1f0508aa4b6fc79d91f364fe813c900ffb260.jpg)

![](images/1e0c44ee5bba3503ec9076773bf4bf22fa08dbd55327bac007dc9b08ab5db017.jpg)

![](images/2848b88307dc251211aec57d88e7596b36ccce099070844cc42b062b4e33f149.jpg)

![](images/84110f3bfc0105290f815d5575d55cdb86810513a6477141a0b735158e4811de.jpg)

![](images/8b96100deed2a01737316cc309e0a5dec3941c559be7836e7b9538ac348f338a.jpg)

![](images/021750aba1df76f9d3e19f123cda5fe969fa2b8f36e6d901b2e6f4a68a21e190.jpg)

![](images/336f7aeda78e09af80a23770e550abb68901f1c52c1cc4cc1b11c9ae13b6f9e6.jpg)

![](images/9e9757c9abb1187e80a26ce30cf136a3a3b62894ea132e5122e90209ba0d9a35.jpg)

![](images/8cf39a522e489d6fd0e862c1d4785a6113fae046d7a397746d58c90159e3e8b6.jpg)

![](images/08365392b2b36675caf90ecb1226a5678fa2e41ed6444be51a8852341fa774dd.jpg)

![](images/c7a00ae6a2d8ad42c2dc82a399cddaa6ad57d72384f4759cf745c514d82e8478.jpg)  
Figure 3: Acc on DCLM and CNN_DCLM

![](images/c7a86fe47ebafb8a31cee356cffc8135bceac915ac45a14c0f06c33b0732ba30.jpg)

![](images/33d87b27437b8445a9bccd73faa5204d680e32c5fc81d822dd83dc794e4628e7.jpg)

![](images/8bc5a8ffaa0fabb1cd203054f9f8c98fe6a0e1b9bda57be7b36629e0108efb13.jpg)

![](images/691265bb5674a1693d456f5e8f99e8a2367a3e48d38305696206fb0228d9d6c5.jpg)

![](images/db07be2ca17c024c68a6b2a5c85e70de56c9490b545332b4eb7e4953f6be3822.jpg)

![](images/30c8b5880980a940c47cf94859008f132704a56a3fff722a7fb214a91bfb70b1.jpg)  
Figure 5: Information Entrop of DCLM

![](images/9c47860618372bcc2b1b78a884884f1d3fa710bc8bf68ee0059eca3d16e5cce3.jpg)

![](images/8c7de08f8a75b6ca39be9883ba5b52247f961e4fa6f17e0bda65db10cca43c20.jpg)  
Figure 6: Acc on CNN and CNN_DCLM

![](images/f3051d7095d718de9b786f42a3e5a88a74be4289035f9ee61d717733e8e99645.jpg)  
Figure 4: Interpretation distance of DCLM

![](images/826907ce430337d2f0b762cb22ec49e3bee90307056284665c9165fbd5b11387.jpg)

![](images/c0a7e0aabc9c053ec8e4c0266898467ddd787e57da3420c5a0f91c1328012d60.jpg)

![](images/b91ac86014a3678a8792e6c813d2331fd959cb398fa77ae01bca45c9036c2093.jpg)

![](images/b5d8fbd591f5ce554e33116b398bb779a2ce307a606dcba1109c1b34cf39a0a2.jpg)

![](images/77340cbd83d343799462520ea446c59915b00ff9f48885986d510e2d3d563804.jpg)

![](images/94e6bc14697b09d21e411f9e8efd889fd384b49ef9eab2a190780eee00186a6a.jpg)

![](images/fef00fce39fc12fa0c5540610456b0dbd3d0178b6441c2c725da60dffa40878c.jpg)

![](images/622cd51e7cf0b38655f637cdfd10446fe45c4521e2a5b3f77e472a76531517ff.jpg)

those of CNNs. But in the last stage, their accuracies tend to be stable and consistent. The main reason is that in the early stage, a tradeoff problem between the generalization performance and interpretability performance of CNN-DCLMs must reduce its generalization performance of CNN-DCLMs to increase its interpretability performance. But the proposed game method can effectively reduce the gap between the two performances. This reflects that such method is effective for the tradeoff problem.

From these subplots in Fig.3 and Fig.4, we also can see that on every epoch a DCLM could be obtained. And after the 15th epoch, their interpretation distances tend to converge. The phenomenon indicates that the game process can decrease the distances and the training process of the discrimination part of the CNNs can be explained by their DCLMs in real time after the fifteenth epoch.

# 7 CONCLUSION

The performance of the proposed method was demonstrated by experiments on benchmark data sets. The proposed method showed prominent advantages over traditional learning algorithm on CNN for improving the generalization performance and the interpretability performance of the discrimination part of the CNN.

In practical engineering, the proposed method may provide a new learning paradigm. The method can not only predict an accurate result for new input data but also provide a reasonable causal interpretation for the prediction of the discrimination part of a CNN. We suppose that it can solve the black box problem in the discrimination part. We believe that the proposed method provides a way how human understands the discrimination part.

# REFERENCES

A.B. Arrietaa, N. Diaz-Rodriguezb, J.D. Sera, A. Bennetotb, S. Tabikg, A. Barbadoh, S. Garciaj, S. Gil-Lopezza, D. Molinag, R. Benjaminsh, R. Chatilaf, and F. Herrerag. Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Information Fusion, 58:82-115, 2020.  
M. Gethsiyal Augusta and T. Kathirvalavakumar. Reverse engineering the neural networks for rule extraction in classification problems. Neural Process. Lett, 35(2):131-150, 2012.  
O. Boz. Extracting decision trees from trained neural networks. pp. 456C461, the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2002.  
J. Brandon. An ai god will emerge by 2042 and write its own bible. will you worship it?, 2017. https://venturebeat.com.  
Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the Twelfth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Philadelphia, PA, USA, August 20-23, 2006, 2006.  
Z.P. Che, S. Purushotham, R. Khemani, and Y. Liu. Interpretable deep models for icu outcome prediction. Amia Annu Symp Proc, 2016:371-380, 2017.  
G. Cohen, S. Afshar, J. Tapson, and A.v. Schaik. EMNIST: an extension of MNIST to handwritten letters. CoRR, abs/1702.05373, 2017. URL http://arxiv.org/abs/1702.05373.  
Mark W. Craven and Jude W. Shavlik. Using sampling and queries to extract rules from trained neural networks. Machine Learning Proceedings, pp. 37-45, 1994.  
Mark W. Craven and Jude W. Shavlik. Extracting tree-structured representations of trained networks. Advances in Neural Information Processing Systems, 8:24-30, 1999.  
N. Frosst and G. Hinton. Distilling a neural network into a soft decision tree, 2017. arXiv preprint arXiv:1711.09784 (2017).  
R. Giles. Lukasiewicz logic and fuzzy set theory. International Journal of Man-Machine Studies, 8 (3):313-327, 1975.  
G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. Computer Science, 2015.  
A. Holzinger, G. Langs, H. Denk, K. Zatloukal, and H. Miller. Causability and explainability of artificial intelligence in medicine. *Causability and explainability of artificial intelligence in medicine*, 9(4):e1312, 2019.  
U. Johansson and L. Niklasson. Rule extraction from trained neural networks using genetic programming. Int.conf.neural Informprocessing, 2003.  
U. Johansson and L. Niklasson. Evolving decision trees using oracle guides. In Computational Intelligence and Data Mining, 2009. CIDM '09. IEEE Symposium on, 2009.  
R. Krishnan, G. Sivakumar, and P. Bhattacharya. Extracting decision trees from trained neural networks. Pattern Recognition, 32(12):1999-2009, 1999.  
Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
S. Parbhoo M. Zazzi V. Roth and F. Doshi-Velez M. Wu, M. C. Hughes. Beyond sparsity: Tree regularization of deep models for interpretability. AAAI, 2018.  
Grgoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek, and Klaus Robert Miller. Explaining nonlinear classification decisions with deep taylor decomposition. Pattern Recognition, 65:211-222, 2017.

Algorithm 2 Construction Logic Net(It's time complexity is  $O(scn)$  where  $s$  is number of iterations of MAX-SAT,  $c$  is number of sample categories and  $n$  is number of samples.)

Input: feature maps  $X$ , feature-capture maps  $F, y_{nn}$ , batch size  $m$ . Initialize  $L_a, y_{dclm} = y_{nn}, c = 0$ .  
repeat  
for  $i = 1$  to sizeof  $(X)$  do  
for  $j = \text{first}(i)$  to final  $(i)$  do  
 $IP[i][j] = \text{Cosine similarity}(X[i], F[j])$ .  
 $lamda[i][j] = |X[i]| | F[j]|$   
end for  
end for  
for  $i = 1$  to sizeof  $(y_m)$  do  
 $NewL_a[i] = \text{maxsat}(IP[[j], lamda][j], y_{dclm}[i])$ .  
 $L_a[i] = NewL_a[i] \cap L_a[i]$ .  
end for  
 $c++$ .  
until  $c == m$  is true

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pp. 8024-8035. Curran Associates, Inc., 2019.

G. Riccardo, M. Anna, R. Salvatore, T. Franco, G. Fosca, and P. Dino. A survey of methods for explaining black box models. ACM Computing Surveys, 51(5):1-42, 2018.

B. Sebastian, B. Alexander, M. Grgoire, K. Frederick, M. Klaus-Robert, S. Wojciech, and S.O. Deniz. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. Plos One, 10(7):e0130140-, 2015.

A. Shrikumar, P. Greenside, A. Shcherbina, and A. Kundaje. Not just a black box: Learning important features through propagating activation differences, 2016. arXiv preprint arXiv:1605.01713 (2016).

J.J. Thiagarajan, B. Kailkhura, P. Sattigeri, and K. N. Ramamurthy. Treeview: Peeking into deep neural networks via feature-space partitioning, 2016. arXiv preprint arXiv:1611.07429(2016).

R. Traore, H. Caselles-Dupre, T. Lesort, T. Sun, G. Cai, N. D. Rodriguez, and D. Filiat. Discorl: continual reinforcement learning via policy distillation, 2019. arXiv preprint arXiv:1907.05855(2019).

A. Wan, L. Dunlap, D. Ho, J. Yin, S. Lee, H. Jin, S. Petryk, S. A. Bargal, and J. E. Gonzalez. Nbdt: Neural-backed decision trees, 2020.

S.E. Zalando. Fashion mnist data set, 2017. https://github.com/zalandoresearch/fashion-mnist.

J.R. Zilke, E.L. Mencia, and F. Janssen. Deepred-rule extraction from deep neural networks. In International Conference on Discovery Science, Springer, pp. 457-473, 2016.

A APPENDIX: A MAXIMUM A POSTERIOR(MAP) ALGORITHM FOR EXTRACTING A DCLM

A Maximum A Posterior(MAP) algorithm on the Maximum Satisfiability Problem (MAX-SAT) for extracting an DCLM is as follows.
