# LEARNING NON-LINEAR TRANSFORM WITH DISCRIMINATIVE AND MINIMUM INFORMATION LOSS PRIORS

Anonymous authors

Paper under double-blind review

# ABSTRACT

This paper proposes learning a non-linear transform with two priors. The first is a discriminative prior defined using a measures on a support intersection and the second is a minimum information loss prior expressed as a constraint on the conditioning and the coherence. An approximation of the measures for the discriminative prior is addressed, connecting it to a similarity concentrations. Along quantifying the discriminative properties of the transform representation a sensitivity analysis of the similarity concentration w.r.t. the parameters of the nonlinear transform is given. Furthermore, a measure, related to the similarity concentration, reflecting the discriminative properties, named as discrimination power is introduced and its bounds are presented.

To support and validate the theoretical analysis a learning algorithm with the proposed prior is presented. The advantages and the potential of the proposed algorithm are evaluated by a computer simulation.

# 1 INTRODUCTION

Learning a transform that provides a sparse and discriminative representation is an active domain of research in many areas, including data processing, pattern recognition, image processing, language modeling, text analysis, gene separation, etc. A class of algorithms proposed by Kreutz-Delgado et al. (2003), Mairal et al. (2009), Bengio et al. (2012), Gangeh et al. (2015), Mairal et al. (2008), Jiang et al. (2011), Guo et al. (2012), Cai et al. (2014) and Liu et al. (2016) for learning discriminative sparse representation have been shown to perform well across various learning tasks. A subclass of them known as discriminative dictionary learning (DDL) Guo et al. (2012), Jiang et al. (2013), Cai et al. (2014), Shekhar et al. (2014), Xu et al. (2015), Liu et al. (2016), Bengio et al. (2012), Gangeh et al. (2015), con, Jiang et al. (2016), Vu & Monga (2016b) address the estimate of the dictionary in a supervised meaner such that the representation w.r.t. words (vectors) from the resulting dictionary (vector set) is discriminative. Most of the DDL methods synthesize the data sample  $\mathbf{x}_{c,k} \in \Re^{N}$  as an approximation by a linear combination  $\mathbf{y}_{c,k} \in \Re^{M}$  (refereed to as a sparse data representation) of a few words (vectors)  $\| \mathbf{y}_{c,k} \|_{0} \ll M$ , from a dictionary (vector set)  $\mathbf{D} \in \Re^{N \times M}$ , i.e.,  $\mathbf{x}_{c,k} = \mathbf{D}\mathbf{y}_{c,k} + \mathbf{v}_{c,k}, \mathbf{v}_{c,k} \in \Re^{N}$ , with  $\mathbf{v}_{c,k}$  denoting the approximation error. It is important to highlight that with the synthesis model approach the data reconstruction is addressed. The differences between the DDL methods Guo et al. (2012), Jiang et al. (2013), Cai et al. (2014), Shekhar et al. (2014), Gangeh et al. (2015), Xu et al. (2015), Liu et al. (2016), Bengio et al. (2012), Jiang et al. (2016), Vu & Monga (2016b) are determined by the prior defined on the sparse representation and the prior defined for the relations between the sparse representations for the data samples from same/different classes. The discrimination is enforced by approximating the prior with a structural constraints on the dictionary or imposing a discriminative term on the sparse representation. Additionally, in a number of works such as Mairal et al. (2008), Guo et al. (2012), Taalimi et al. (2015) is considered even a joint estimation/learning of dictionary, sparse representation and classifier by using iterative alternating minimization strategy. A comprehensive overview covering different approaches is given in Bengio et al. (2012), Cai et al. (2014) and Gangeh et al. (2015).

![](images/3db0349b29151ff9845347ca424990bfc6153f1248768a2cc0a3d3d953c1805a.jpg)  
Figure 1: a) Data samples  $\mathbf{X}_c, c \in \{1,2,3,4\}$  from four different classes, b) Given a  $k$ -th data sample  $\mathbf{x}_{c,k}$  from class  $c$ , the non-linear transform is represented as two step operation: linear mapping  $\mathbf{A}\mathbf{x}_{c,k}$  (step 1) followed by an element-wise thresholding function  $\mathbf{y}_{c,k} = \mathcal{H}_{\tau}(\mathbf{A}\mathbf{x}_{c,k})$  (step 2). c) The transform data samples  $\mathbf{Y}_c, c \in \{1,2,3,4\}$ .

# 1.1 OPEN ISSUES

The general open issue for the DDL methods is the computational complexity w.r.t. the optimal dictionary/transform learning and the discriminative encoding, since the sparse representation in the synthesis model is a solution to an inverse problem. An additional open issue with most of the proposed approaches Guo et al. (2012), Jiang et al. (2013), Cai et al. (2014), Gangeh et al. (2015), Liu et al. (2016), Bengio et al. (2012), Jiang et al. (2016) and Vu & Monga (2016b) is that there is no formal notion to measure the discriminative properties. Therefore, there are no means that provide a quantitative evaluation about the quality of the representation, other than the performance of a classifier used on top of the representation.

In terms of the specifics about the discriminative constraints, Yang et al. (2011b) proposed a synthesis model with a discriminative fidelity term and Fisher discriminant constraints, where the within-class scatter and the between-class scatter of the representation is minimized and maximized, respectively. Vu & Monga (2016c) proposed extension considering a low rank constraints on the dictionary. The authors in Guo et al. (2013) used a synthesis model with a pairwise constraints on the sparse representation. They have modeled the pair-wise constraints using a  $\ell_2$  distance metric. The methods of Yang et al. (2011b), Vu & Monga (2016c) and Guo et al. (2013) take into account assumption on the metric by defining the scatter and the pair-wise relation. Therefore, they constrain the space of the representation, that essentially is determined by the dictionary. However, these works do not consider whether the used metric is optimal for the sparse representation. The authors in Liu et al. (2016) have proposed a method that finds a dictionary under which the coefficients of a data sample from the same class  $c$  have a common sparse structure while the size of the overlapped signal support of different classes, denoted as  $c_1$  and  $c_2$  is minimized. Furthermore, assuming  $\mathbf{y}_{c1,k1} \in \Re^M$  and  $\mathbf{y}_{c2,k2} \in \Re^M$  are two sparse representations for data samples  $\mathbf{x}_{c1,k1} \in \Re^N$  and  $\mathbf{x}_{c2,k2} \in \Re^N$ , then the proposed similarity measure in Liu et al. (2016) is defined as expectation of  $\| \mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2} \|_0$ , where  $\odot$  represents the Hadamard product. Note that two transform data samples  $\mathbf{y}_{c1,k2}$  and  $\mathbf{y}_{c2,k2}$  that have small support overlap  $\| \mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2} \|_0 = s$ ,  $s << M$ , might not necessarily be similar or dissimilar, i.e.,  $\mathbf{y}_{c1,k1} = \mathbf{y}_{c2,k2}$  and  $\mathbf{y}_{c1,k1} = -\mathbf{y}_{c2,k2}$  with  $\| \mathbf{y}_{c1,k1} \|_0 = \| \mathbf{y}_{c2,k2} \|_0 = s$  and  $s$  small.

# 1.2 MOTIVATION AND APPROACH

This paper addresses a non-linear transform model, where the data reconstruction is not modeled, and at least it is not targeted explicitly. Moreover, the model does not restrict the transform representation to be in the column space of the dictionary, allowing a rich class of representations to be modeled. In this line, we have to mention that a special case of the non-linear transform model is the sparsifying transform model, first introduced in Ravishankar & Bresler (2012). This model assumes that the data sample  $\mathbf{x}$  is approximately sparsifiable under a linear transform  $\mathbf{A} \in \Re^{M \times N}$ , i.e.,  $\mathbf{A}\mathbf{x}_{c,k} = \mathbf{y}_{c,k} + \mathbf{z}_{c,k}, \mathbf{z}_{c,k} \in \Re^{M}$ , where  $\mathbf{y}_{c,k}$  is sparse  $\| \mathbf{y}_{c,k} \|_0 << M$ . Moreover, we note that

![](images/af27354bdc9669c32cb77e951026152d868c46217a783d9b09f9acbdf1ccb012.jpg)  
Figure 2: a) Two transform representations  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c1,k1}$ , b) the resulting Hadamard products  $\mathbf{y}_{c,k}^{+} \odot \mathbf{y}_{c2,k2}^{+}$  and  $\mathbf{y}_{c1,k1}^{-} \odot \mathbf{y}_{c2,k2}^{-}$  on the support intersection for the similarity contribution and c) the resulting Hadamard products  $\mathbf{y}_{c1,k1}^{+} \odot \mathbf{y}_{c2,k2}^{-}$  and  $\mathbf{y}_{c1,k1}^{-} \odot \mathbf{y}_{c2,k2}^{+}$  on the support intersection for the dissimilarity contribution between  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c2,k2}$ .

the sparsifying transform model represents a generalization of the analysis model Rubinstein et al. (2010), Rubinstein et al. (2013), Ravishankar & Bresler (2014) and Rubinstein & Elad (2014).

Together with the non-linear transform model, a novel prior is proposed, which is defined by a parametric measures on the support intersection. The first motivation behind using a measure defined on the support intersection is that it allows more freedom in imposing a regularization on the discriminative properties for the transform representation without taking into account any additional assumptions, contrary to Yang et al. (2011b), Vu & Monga (2016c), Guo et al. (2013) and Liu et al. (2016). Second, by a simple approximation on the parametric measure the focus of the regularization can be directly put on the contributing components for the similarity, i.e., consider the measure  $(\mathbf{y}_{c1,k1}^{+})^{T}\mathbf{y}_{c2,k2}^{+} + (\mathbf{y}_{c1,k1}^{-})^{T}\mathbf{y}_{c2,k2}^{-1}$  between two transform representations  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c2,k2}$ . This measure captures the main contributing components for similarity on the support intersection, whereas  $(\mathbf{y}_{c1,k1}^{+})^{T}\mathbf{y}_{c2,k2}^{-} + (\mathbf{y}_{c1,k1}^{-})^{T}\mathbf{y}_{c2,k2}^{+}$  captures the main contribution for dissimilarity between  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c2,k2}$ . Third, the expectation of this similarity measure across a data set captures the concentration of similarity for that set. Therefore, it provides the possibility to define a formal notion to quantify the discriminative properties. Forth, the measure  $(\mathbf{y}_{c1,k1}^{+})^{T}\mathbf{y}_{c2,k2}^{+} + (\mathbf{y}_{c1,k1}^{-})^{T}\mathbf{y}_{c2,k2}^{-}$ is not ambiguous w.r.t. a notion for similarity/dissimilarity between two sparse representations  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c2,k2}$ . This is because the support intersections for the positive and the negative components  $\| \mathbf{y}_{c1,k1}^{+} \odot \mathbf{y}_{c2,k2}^{+} \|_{1}$  and  $\| \mathbf{y}_{c1,k1}^{-} \odot \mathbf{y}_{c2,k2}^{-} \|_{1}$ , respectively, are considered separately. In addition, a measure for the strength on the support intersection, defined as  $\| \mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2} \|_{2}^{2}$  is also taken into account. Note that the expectation of the measure  $\| \mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2} \|_{2}^{2}$  across the data set captures the expected strength on the support intersection for that set.

A schematic diagram of the transform and the main idea behind the proposed concept are shown in Figure 1. In Figure 2 a), b), c) is given an illustrative example for the support intersection between the positive and negative component of two vectors  $\mathbf{y}_{c1,k1}$  and  $\mathbf{y}_{c2,k2}$  in the transform domain.

We propose a learning algorithm for the non-linear transform model with the discriminative prior. In a direction of quantifying the discriminative properties we give a sensitivity analysis w.r.t. the parameters of the transform. Along the same line, we present an information preservation rela

<table><tr><td>x(n), A(n, m)</td><td>scalar</td></tr><tr><td>xc,k, yc,k</td><td>vector</td></tr><tr><td>X, Xc, X\{k∈c}, Y, A</td><td>matrix</td></tr><tr><td>∅</td><td>Hadamard product</td></tr><tr><td>||.||p</td><td>ℓp-norm,</td></tr><tr><td>Pc={A ∈ RM×N,τc ∈ RM}</td><td>set of parameters</td></tr><tr><td>T^P(.)</td><td>non-linear parametric function with parameters P</td></tr><tr><td>D^P_{ℓ1}(x_{c1,k1}; x_{c2,k2})</td><td>similarity measure between T^P(x_{c1,k1}) and T^P(x_{c1,k2})</td></tr><tr><td>∂D^P_{ℓ1}(x_{c1,k1}; x_{c2,k2})/∂A</td><td>first order derivative of D^P_{ℓ1}(x_{c1,k1}; x_{c2,k2}) w.r.t. A</td></tr><tr><td>D^P_{ℓ1}(X)/∂A</td><td>concentration of similarity</td></tr><tr><td>∂D^P_{ℓ1}(X)/∂A</td><td>change of concentration w.r.t. change of A</td></tr><tr><td>∂D^B_N(X)/∂A_o|A_o=B</td><td>change of concentration w.r.t. change of A_o evaluated at B</td></tr><tr><td>I^o, I^l</td><td>discrimination power in original and transform domain</td></tr></table>

Table 1: a) Most important notations used thought the paper

tions for the change of a similarity concentrations w.r.t. change of the parameters of the transform, connecting all together the proposed prior, the model and the used non-linear transform. Furthermore, w.r.t. the similarity concentrations a notion defined as discrimination power, reflecting the discriminate properties of the representation for a data set is introduced. Based on the sensitivity analysis we present lower and upper bound on the discrimination power in the transform domain. On the practical side, the advantages and the potential of the proposed algorithm are demonstrated by a numerical experiments using the Extended YALE B Georghiades et al. (2001), AR Martínez & Benavente (1998), Norb LeCun et al. (2004), Coil-20 Nene et al. (1996), Clatech101 LeCun et al. (2008), UKB Nistér & Stewénius (2006) and MNIST Lecun & Cortes data sets.

# 1.3 NOTATIONS

A scalar variable is denoted using the usual symbols, i.e.,  $x$ , a vector is denoted by a bold, low caps symbols, i.e.,  $\mathbf{x}$ , a matrix by bold, upper cap symbol, i.e.,  $\mathbf{A}$ . A single element from a vector (or matrix) is denoted as  $x(n)$  (or  $A(m,n)$ ). A set is denoted by a calligraphic symbol, i.e.,  $\mathcal{S}$ . The  $\ell_p$ -norm is denoted as  $\|. \|_p$  and the nuclear norm as  $\|. \|_*$ . The  $\odot$  symbol represents the Hadamard product. Throughout the paper it is assumed that a set of data samples  $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_C] \in \Re^{N \times L}$ ,  $L = CK$  from  $C$  classes is given and that every class  $c \in \mathcal{C} = \{1, 2, 3, \dots, C\}$  has  $K$  samples,  $\mathbf{X}_c = [\mathbf{x}_{c,1}, \mathbf{x}_{c,2}, \dots, \mathbf{x}_{c,K}] \in \Re^{N \times K}$ ,  $\mathbf{x}_{c,k} \in \Re^N$ ,  $\forall c \in \mathcal{C}, \forall k \in \mathcal{K} = \{1, 2, \dots, K\}$ . We denote a transform data as  $\mathbf{Y} = [\mathbf{Y}_1, \mathbf{Y}_2, \dots, \mathbf{Y}_C] \in \Re^{M \times L}$ , where  $\mathbf{Y}_c = [\mathbf{y}_{c,1}, \mathbf{y}_{c,2}, \dots, \mathbf{y}_{c,K}] \in \Re^{M \times K}$  and  $\mathbf{y}_{c,k} \in \Re^M$ . We denote  $\mathbf{X}_{\backslash\{k \in c\}} = [\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_{c,\backslash k}, \dots, \mathbf{X}_C] \in \Re^{N \times (L-1)}$  as the matrix that has all the columns of  $\mathbf{X}$ , except the column  $\mathbf{x}_{c,k} \in \Re^N$ , where  $\mathbf{X}_{c,\backslash k} = [\mathbf{x}_{c,1}, \mathbf{x}_{c,2}, \dots, \mathbf{x}_{c,k-1}, \mathbf{x}_{c,k+1}, \dots, \mathbf{x}_{c,K}] \in \Re^{N \times (K-1)}$  is a matrix that has all the columns of block  $\mathbf{X}_c$ , except the column  $\mathbf{x}_{c,k}, \forall c \in C$  and  $\forall k \in K$ . We let  $\mathcal{N} = \{1, 2, \dots, N\}$  and  $\mathcal{M} = \{1, 2, \dots, M\}$ . We denote  $\mathcal{D}^M$  as the set of all  $M \times M$  diagonal matrices with non-negative diagonal elements.

# 2 LEARNING NON-LINEAR TRANSFORM WITH DISCRIMINATIVE AND MINIMUM INFORMATION LOSS PRIORS

Assume that training data  $\mathbf{X} = [\mathbf{X}_1, \mathbf{X}_2, \dots, \mathbf{X}_C] \in \Re^{N \times CK}$  are given, consisting of  $C$  classes. Per every class  $c \in \mathcal{C}$  there are  $K$  training samples, i.e.,  $\mathbf{X}_c = [\mathbf{x}_{c,1}, \mathbf{x}_{c,2}, \dots, \mathbf{x}_{c,K}]$ . Furthermore, we assume that per class  $c \in \mathcal{C}$  there exist an unknown non-linear functions defined by set of parameters:

$$
\mathcal {P} _ {c} = \left\{\mathbf {A} \in \Re^ {M \times N}, \boldsymbol {\tau} _ {c} \in \Re^ {M} \right\}, \forall c \in \mathcal {C}, \tag {1}
$$

that separates the data samples from different classes in the transform domain. The linear map  $\mathbf{A}$  is shared for all of them, but, the parameters  $\tau_{c}$  per different class  $c$  from the corresponding non-linear transforms  $\mathcal{P}_c$  are different. Additionally, given any  $\mathbf{x}_{c,k}$  from class  $c$  where  $k\in \mathcal{K}$  it is assumed that

the non-linear transform is expressible in two steps, consisting of a linear mapping (step 1) followed by an element-wise non-linearity (step 2), as follows:

$$
\mathbf {x} _ {c, k} \xrightarrow [ \text {s t e p 1} ]{\mathbf {A}} \mathbf {A x} _ {c, k} \xrightarrow [ \text {s t e p 2} ]{\mathcal {H} _ {\tau_ {c} , (.)}} \mathbf {y} _ {c, k}, \tag {2}
$$

where  $\mathcal{H}_{\tau_c}(.):\Re^M\to \Re^M$  is a non-linear thresholding function with parameters  $\tau_{c}$ . It is important to mention that the thresholding is done with a different thresholding parameter per different transform dimension, i.e.,  $\tau_{c}(m),\forall m\in \mathcal{M}$

The goal of learning a non-linear transform (2) is to estimate only one model defined by a set of parameters

$$
\mathcal {P} = \left\{\mathbf {A} \in \Re^ {M \times N}, \boldsymbol {\tau} = \tau \mathbf {1} \in \Re_ {+} ^ {M} \right\}, \tag {3}
$$

that is common for all data samples that come from all classes by taking into account a set of priors. That is, we try to find as accurate as possible approximation (3) to the models represented by a set of parameters  $\mathcal{P}_c = \{\mathbf{A} \in \Re^{M \times N}, \tau_c \in \Re^M\}$  with as small as possible loss in the discriminative properties of the representation in the transform domain.

# 2.1 THE NON-LINEAR TRANSFORM MODEL WITH A DISCRIMINATIVE PRIOR

The compact description of the non-linear transform (2) by a non-linear model is expressed as follows:

$$
\mathbf {A} \mathbf {x} _ {c, k} = \mathbf {y} _ {c, k} + \mathbf {z} _ {c, k} \text {w h e r e} \mathbf {y} _ {c, k} = \mathcal {T} ^ {\mathcal {P} _ {c}} \left(\mathbf {x} _ {c, k}\right), \tag {4}
$$

where the function  $\mathcal{T}^{\mathcal{P}_c}(.) : \Re^N \to \Re^M$  is a parametric non-linear function that gives  $\mathbf{y}_{c,k}$ , by using the set of parameters  $\mathcal{P}_c$ . The term  $\mathbf{z}_{c,k} = \mathbf{A}\mathbf{x}_{c,k} - \mathbf{y}_{c,k}$  is the non-linear transform error vector that represents the deviation of the transform data  $\mathbf{A}\mathbf{x}_{c,k}$  from the targeted transform representation  $\mathbf{y}_{c,k} = \mathcal{T}^{\mathcal{P}_c}(\mathbf{x}_{c,k})$  in the transform domain.

Assuming Gaussian distributed error vector, the prior on  $\mathbf{z}_{c,k}$  is modeled as  $p(\mathbf{x}_{c,k}|\mathbf{y}_{c,k})\propto$ $\exp (-\frac{\|\mathbf{A}\mathbf{x}_{c,k} - \mathbf{y}_{c,k}\|_2}{\beta_0})$ , additionally, assuming that the non-linear function  $\mathcal{T}^{\mathcal{P}_c}(\mathbf{x}_{c,k})$  gives sparse  $\mathbf{y}_{c,k}$ , then we have the improper prior on  $\mathbf{y}_{c,k}$ , defined as  $p(\mathbf{y}_{c,k})\propto \exp (-\frac{\|\mathbf{y}_{c,k}\|_1}{\beta_1})$ .

This paper models the discriminative prior as  $p(\pmb{\tau}_c, |\mathbf{y}_{c,k}) \propto \exp\left(-\frac{D(\mathbf{y}_{c,k}; \pmb{\tau}_c)}{\beta_2}\right)$  by using an parametric measure  $D(\mathbf{y}_{c,k}; \pmb{\tau}_c)$  with parameters  $\pmb{\tau}_c$ . Assuming that the measure  $D(\mathbf{y}_{c,k}; \pmb{\tau}_c)$  is defined on the support intersection between  $\mathbf{y}_{c,k}$  and  $\pmb{\tau}_c$  we propose the following formulation:

$$
D \left(\mathbf {y} _ {c, k}; \boldsymbol {\tau} _ {c}\right) = \left\| \mathbf {y} _ {c, k} ^ {+} \odot \boldsymbol {\tau} _ {c} ^ {+} \right\| _ {1} + \left\| \mathbf {y} _ {c, k} ^ {-} \odot \boldsymbol {\tau} _ {c} ^ {-} \right\| _ {1} + \left\| \mathbf {y} _ {c, k} \odot \boldsymbol {\tau} _ {c} \right\| _ {2} ^ {2} \tag {5}
$$

where  $\| \mathbf{y}_{c,k}^{+}\odot \pmb{\tau}_{c}^{+}\|_{1} + \| \mathbf{y}_{c,k}^{-}\odot \pmb{\tau}_{c}^{-}\|_{1}$  measures the similarity contribution on the support intersection using the positive and the negative components  $\mathbf{y}_{c,k}^{+},\pmb{\tau}_{c}^{+}$  and  $\mathbf{y}_{c,k}^{-},\pmb{\tau}_{c}^{-}$  of  $\mathbf{y}_{c,k}$  and  $\pmb{\tau}_{c}$ , respectively and  $\| \mathbf{y}_{c,k}\odot \pmb{\tau}_{c}\|_{2}^{2}$  measures the strength of the support intersection between  $\mathbf{y}_{c,k}$  and  $\pmb{\tau}_{c}$ .

We highlight that the true  $p(\tau_c)$ ,  $\tau_c$ , are not known and the only prior about  $D(\mathbf{y}_{c,k};\tau_c)$  and  $p(\tau_c)$  is that they should be informative enough w.r.t. the discriminative properties of the transform representation  $\mathbf{y}_{c,k}$ . Furthermore, instead of estimating them explicitly, an approximation to  $D(\mathbf{y}_{c,k};\tau_c)$  is considered, based only on the concentrations of the similarity on the support intersection and the expected strength of the support intersection for the transform data in the transform domain.

The approximation We propose an approximation that captures two expectations. The first one is the expected similarity on the support intersection for the positive and negative component between  $\mathbf{y}_{c,k}$  and the set of transform representations  $\mathbf{Y} \backslash c$  that come from all classes  $c1$  different from  $c$ , i.e.,  $c \neq c1$ . The second is the expected strength on the support intersection between  $\mathbf{y}_{c,k}$  and the set of transform representations  $\mathbf{Y} \backslash c$  that come from all classes  $c1$  different from  $c$ , i.e.,  $c \neq c1$ . The approximation is defined as:

$$
D (\mathbf {y} _ {c, k}; \boldsymbol {\tau} _ {c}) \sim D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X}) + S _ {\ell_ {2}} ^ {\mathcal {P}} (\mathbf {X}),
$$

$$
D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X}) = \sum_ {\substack {c 1 \\ c 1 \neq c}} \sum_ {k 1} \left(\| \mathbf {y} _ {c, k} ^ {+} \odot \mathbf {y} _ {c 1, k 1} ^ {+} \| _ {1} + \| \mathbf {y} _ {c, k} ^ {-} \odot \mathbf {y} _ {c 1, k 1} ^ {-} \| _ {1}\right), \tag{6}
$$

$$
S_{\ell_{2}}^{\mathcal{P}}(\mathbf{X}) = \sum_{\substack{c1\\ c1\neq c}}\sum_{k1}\| \mathbf{y}_{c,k}  \odot   \mathbf{y}_{c1,k1}\|_{2}^{2}
$$

where the  $m$ -th element  $y_{c,k}^{+}(m)$  of  $\mathbf{y}_{c,k}^{+}$  is defined as  $y_{c,k}^{+}(m) = \max(y_{c,k}(m), 0)$  and similarly,  $y_{c,k}^{-}(m) = \max(-y_{c,k}(m), 0)$ ,  $\forall m \in \mathcal{M}$ . We also define the expected similarity using the positive and negative components of  $\mathbf{y}_{c,k}$  for the set of transform representations  $\mathbf{Y}_c$  that come from the same classes  $c$  as  $D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X}) = \sum_{k1} \| \mathbf{y}_{c,k}^{+} \odot \mathbf{y}_{c,k1}^{+} \|_1 + \sum_{k1} \| \mathbf{y}_{c,k}^{-} \odot \mathbf{y}_{c,k1}^{-} \|_1$ . The terms  $\frac{2}{((C-1)K)(CK)} \sum_{\substack{c1 \\ c1 \neq c}} \sum_{k1} \mathbf{y}_{c1,k1}^{-}$  and  $\frac{2}{((C-1)K)(CK)} \sum_{\substack{c1 \\ c1 \neq c}} \sum_{k1} \mathbf{y}_{c1,k1}^{+}$  might also be seen finite sample estimates of the positive and negative component  $\tau_c^+$  and  $\tau_c^-$ , respectively, for the unknown variable  $\tau_c^2$ . The term  $\frac{2}{((C-1)K)(CK)} \sum_{\substack{c2 \\ c2 \neq c1}} \sum_{k2} \mathbf{y}_{c2,k2} \odot \mathbf{y}_{c2,k2}$  might also be seen as finite sample estimates of the Hadamard square  $\tau_c \odot \tau_c$  for the unknown variable  $\tau_c^3$ . If the measure  $D_{\ell_1}^{\mathcal{P}}(\mathbf{X})$  is not used then the approximation (6) is most similar to the one proposed in Liu et al. (2016).

Note that the Fisher discriminate constraint Yang et al. (2011b), the pairwise constraint Guo et al. (2013) and the support intersection constraint Liu et al. (2016) are all approximations of a discriminative prior. However, all of them are with specific assumptions on the distribution of the data representation in the transform domain. The advantage of using (6) is that the approximation is without any prior on the probability distributions  $p(\tau_c)$  and without any explicit assumption about the distance metric/measure, or the space/manifold in the transform domain.

# 2.2 THE LEARNING ALGORITHM

The summary of the used priors is as follows:

$$
p \left(\mathbf {x} _ {c, k} \mid \mathbf {y} _ {c, k}\right) \propto \exp \left(- \frac {\| \mathbf {A x} _ {c , k} - \mathbf {y} _ {c , k} \| _ {2}}{\beta_ {0}}\right)
$$

$$
p \left(\mathbf {y} _ {c, k}\right) \propto \exp \left(- \frac {\| \mathbf {y} _ {c , k} \| _ {1}}{\beta_ {1}}\right) \tag {7}
$$

$$
p (\boldsymbol {\tau} _ {c}, | \mathbf {y} _ {c, k}) \propto \exp \left(- \frac {D (\mathbf {y} _ {c , k} ; \boldsymbol {\tau} _ {c})}{\beta_ {2}}\right).
$$

Additionally, we have a prior on  $\mathbf{A}$  to regularize the information loss in order to avoid trivially unwanted matrices  $\mathbf{A}$ , i.e., matrices that have repeated or zero rows. The corresponding prior is defined as:

$$
p (\mathbf {A}) \propto \exp \left(- \left(\frac {1}{\beta_ {3}} \| \mathbf {A} \| _ {F} ^ {2} + \frac {1}{\beta_ {4}} \| \mathbf {A A} ^ {T} - \mathbf {I} \| _ {F} ^ {2} - \frac {1}{\beta_ {5}} \log | \det  \mathbf {A} ^ {T} \mathbf {A} |\right)\right), \tag {8}
$$

where the  $\| \mathbf{A}\| _F$  penalty helps regularize the scale ambiguity, the log  $|\operatorname *{det}(\mathbf{A}^T\mathbf{A})|$  and  $\| \mathbf{A}\| _F^2$  are functions of the singular values of  $\mathbf{A}$  and together help regularize the conditioning of  $\mathbf{A}$ . Assuming that the expected coherence  $\mu^2 (\mathbf{A})$  between the rows  $\mathbf{a}_m$  of  $\mathbf{A}$  (i.e.,  $\mathbf{A}^T = [\mathbf{a}_1,\mathbf{a}_2,\dots,\mathbf{a}_M]$ ) is defined as  $\mu^2 (\mathbf{A}) = \frac{2}{M(M - 1)}\sum_{m_1\neq m_2}|\mathbf{a}_{m_1}\mathbf{a}_{m_2}^T|^2,\forall m_1,m_2\in \{1,2,\ldots ,M\}$ . Then  $\| \mathbf{A}\mathbf{A}^T -\mathbf{I}\| _F^2$  measures the expected coherence  $\mu^2 (\mathbf{A})$  and the  $\ell_2$  norm for the rows of  $\mathbf{A}$ .

Note that the joint probability can be expressed as:

$$
\begin{array}{l} p \left(\mathbf {x} _ {c, k}, \mathbf {y} _ {c, k}, \boldsymbol {\tau} _ {c}, \mathbf {A}\right) = p \left(\mathbf {x} _ {c, k}, \mathbf {y} _ {c, k}, \boldsymbol {\tau} _ {c}, | \mathbf {A}\right) p (\mathbf {A}) = \\ p (\mathbf {A} | \mathbf {x} _ {c, k}, \mathbf {y} _ {c, k}, \boldsymbol {\tau} _ {c}) p (\mathbf {x} _ {c, k}, \mathbf {y} _ {c, k}, \boldsymbol {\tau} _ {c},) \tag {9} \\ \end{array}
$$

$$
p (\mathbf {x} _ {c, k}, \mathbf {y} _ {c, k}, \boldsymbol {\tau} _ {c}) = p (\mathbf {x} _ {c, k} | \mathbf {y} _ {c, k}) p (\boldsymbol {\tau} _ {c}, | \mathbf {y} _ {c, k}) p (\mathbf {y} _ {c, k}),
$$

since  $p(\mathbf{x}_{c,k}|\mathbf{y}_{c,k},\boldsymbol{\tau}_c,)=p(\mathbf{x}_{c,k}|\mathbf{y}_{c,k})$

$^{2}$ Note that since  $D_{\ell_1}^{\mathcal{P}}(\mathbf{X}) = \sum_{\substack{c1 \\ c1 \neq c}} \sum_{k1} \| \mathbf{y}_{c,k}^{+} \odot \mathbf{y}_{c1,k1}^{+} \|_1 + \sum_{\substack{c1 \\ c1 \neq c}} \sum_{k1} \| \mathbf{y}_{c,k}^{-} \odot$

$$
\mathbf{y}_{c1,k1}^{-}\|_{1} = \quad \| \left(\sum_{\substack{c1\\ c1\neq c}}\sum_{k1}\mathbf{y}_{c1,k1}^{+}\right)\odot \mathbf{y}_{c,k}^{+}\|_{1} + \quad \| \left(\sum_{\substack{c1\\ c1\neq c}}\sum_{k1}\mathbf{y}_{c1,k1}^{-}\right)\odot \mathbf{y}_{c,k}^{-}\|_{1},\quad \boldsymbol{\tau}_{c}^{-}\quad \sim
$$

$$
\frac{2}{((C - 1)K)(CK)}\sum_{\substack{c1\\ c1\neq c}}\sum_{k1}\mathbf{y}_{c1,k1}^{-}\text{and}\boldsymbol{\tau}_{c}^{+}\sim \frac{2}{((C - 1)K)(CK)}\sum_{\substack{c1\\ c1\neq c}}\sum_{k1}\mathbf{y}_{c1,k1}^{+}.
$$

$^3$ Note that since  $S_{\ell_2}^{\mathcal{P}}(\mathbf{X}) = \sum_{c2 \neq c1} \sum_{k2} \| \mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2} \|_2^2 = (\mathbf{y}_{c1,k1} \odot \mathbf{y}_{c2,k2})$

$$
\mathbf{y}_{c1,k1}\big)^{T}\left(\sum_{\substack{c2\\ c2\neq c1}}\sum_{k2}\mathbf{y}_{c2,k2} \odot \mathbf{y}_{c2,k2}\right),\\ \boldsymbol{\tau}_{c}\odot \boldsymbol{\tau}_{c}\sim \frac{2}{((C - 1)K)(CK)}\sum_{\substack{c2\\ c2\neq c1}}\sum_{k2}\mathbf{y}_{c2,k2}\odot \mathbf{y}_{c2,k2}.
$$

Algorithm 1 Non-linear transform learning algorithm  
Input  $\mathbf{X},\lambda_0,\lambda_1,\lambda_2,\lambda_3,\lambda_4$    
A  $\leftarrow$  initialize   
repeat   
DISCRIMINATIVE ENCODING closed form solution per data sample   
Y  $\leftarrow$  AX   
repeat for  $\forall c\in \mathcal{C}$  do  $\mathbf{d}_c^- \gets \sum_{c1\neq c}^{c1}\sum_{k1}\mathbf{y}_{c1,k1}^-$ $\mathbf{d}_c^+ \gets \sum_{c1\neq c}^{c1}\sum_{k1}\mathbf{y}_{c1,k1}^+$  and  $\mathbf{s}_c \gets \sum_{c1\neq c}^{c1}\sum_{k1}\mathbf{y}_{c1,k1} \odot \mathbf{y}_{c1,k1}$    
end for for  $\forall c\in \mathcal{C}$  and  $\forall k\in \mathcal{K}$  do g  $\leftarrow$  sign(max(Axc,k,0))  $\odot$  d  $^+$  +sign(max(-Axc,k,0))  $\odot$  dc y  $c,k$  1 sign(AX)  $\odot$  max (Axc,k|-λ0g-λ11,0) 0 (1+2λ0sc) y  $c,k$  1 y  $c,k / \| y_{c,k}\| _2$    
end for   
until convergence   
TRANSFORM UPDATE e-close closed form solution UxSigmaT X X T + λ2I and UUXXYSigmaXYVXTUXXY UXTXYT minσn(σn) σn4n + σn(2n-23) σn2n - σn(σn) σn4n - 2λ4 log σnA(n) where σn(n) nT(n,n), T n UUXXYSigmaXYUXTXY, ∀n ∈ N A VuxxyUTxxySigmaX-1UTX   
until convergence   
Output A, Y

Given the available training data set  $\mathbf{X} \in \Re^{N \times CK}$  maximizing  $p(\mathbf{x}_{c,k}, \mathbf{y}_{c,k}, \boldsymbol{\tau}_c, \mathbf{A})$  over  $\mathbf{Y}$  and  $\mathbf{A}$  is same as minimizing the following problem:

$$
\min  _ {\mathbf {Y}, \mathbf {A}} \frac {1}{2} \| \mathbf {A} \mathbf {X} - \mathbf {Y} \| _ {F} ^ {2} + \sum_ {c, k} \lambda_ {0} D \left(\mathbf {y} _ {c, k}; \boldsymbol {\tau} _ {c}\right) + \lambda_ {1} \| \mathbf {y} _ {c, k} \| _ {1} + \tag {10}
$$

$$
\frac {\lambda_ {2}}{2} \| \mathbf {A} \| _ {F} ^ {2} + \frac {\lambda_ {3}}{2} \| \mathbf {A A} ^ {T} - \mathbf {I} \| _ {F} ^ {2} - \lambda_ {4} \log | \det \mathbf {A} ^ {T} \mathbf {A} |,
$$

where  $\{\lambda_0,\lambda_1,\lambda_3,\lambda_4\}$  are inversely proportional to the scaling parameters  $\{\beta_{1},\beta_{2},\beta_{3},\beta_{4},\beta_{5}\}$ . Note that the solution to (10) is not equivalent to the maximum a priory (MAP) solution, which is difficult to compute, as it involves integrating over the vectors  $\mathbf{y}_{c,k}$ . In terms of optimization the problem is not convex in the variables  $(\mathbf{Y},\mathbf{A})$ . The solution is obtained by iteratively, marginally maximizing the probability  $p(\mathbf{x}_{c,k},\mathbf{y}_{c,k},\boldsymbol{\tau}_c,\mathbf{A})$  over  $\mathbf{Y}$  and  $\mathbf{A}$  which is equivalent to maximizing the conditional densities  $p(\mathbf{y}_{c,k}|\mathbf{x}_{c,k},\boldsymbol{\tau}_c,\mathbf{A})$  and  $p(\mathbf{A}|\mathbf{y}_{c,k},\mathbf{x}_{c,k})$ , respectively. Meaning that at one iterating step one of the variables  $\mathbf{Y}$  or  $\mathbf{A}$  is fixed and w.r.t. the other the problem (10) is minimized. The following text describes the iterating steps that consist of linear map estimation (maximizing  $p(\mathbf{A}|\mathbf{y}_{c,k},\mathbf{x}_{c,k}))$  and discriminative encoding (maximizing  $p(\mathbf{y}_{c,k}|\mathbf{x}_{c,k},\boldsymbol{\tau}_c,\mathbf{A}))$ ).

Linear map estimation: Given the available data samples  $\mathbf{X}$  and the corresponding transform representations  $\mathbf{Y}$  the linear map  $\mathbf{A}$  estimation problem reduces to:

$$
\min  _ {\mathbf {A}} \frac {1}{2} \| \mathbf {A} \mathbf {X} - \mathbf {Y} \| _ {F} ^ {2} + \frac {\lambda_ {2}}{2} \| \mathbf {A} \| _ {F} ^ {2} + \frac {\lambda_ {3}}{2} \| \mathbf {A} \mathbf {A} ^ {T} - \mathbf {I} \| _ {F} ^ {2} - \lambda_ {4} \log | \det  \mathbf {A} ^ {T} \mathbf {A} |, \tag {11}
$$

and we use the  $\epsilon$ -close closed form solution estimated as follows:

Proposition 1 ( $\epsilon$ -close solution): Given  $\mathbf{Y} \in \Re^{M \times L}$ ,  $\forall \mathbf{X} \in \Re^{N \times L}$  and  $M \geq N$ ,  $\forall \lambda_2 \geq 0, \lambda_3 \geq 0, \lambda_4 \geq 0$ , let the eigen value decomposition  $\mathbf{U}_X \boldsymbol{\Sigma}_X^2 \mathbf{U}_X^T$  of  $\mathbf{X} \mathbf{X}^T + \lambda_2 \mathbf{I}$  and the singular value decomposition  $\mathbf{U}_{U_X X Y} \boldsymbol{\Sigma}_{U_X X Y} \mathbf{V}_{U_X X Y}^T$  of  $\mathbf{U}_X^T \mathbf{X} \mathbf{Y}^T$  exist, then if and only if  $\sigma_X(n) > 0$ ,  $\forall n \in \mathcal{N} = \{1, 2, 3, \dots, N\}$ , (11) has  $\epsilon$ -close approximative solution as:

$$
\hat {\mathbf {A}} = \mathbf {V} _ {U _ {X} X Y} \mathbf {U} _ {U _ {X} X Y} ^ {T} \boldsymbol {\Sigma} _ {A} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}, \tag {12}
$$

where  $\Sigma_A$  is diagonal matrix,  $\Sigma_A(n,n) = \sigma_A(n)\geq 0$ , and  $\sigma_{A}(n)$  are solutions to

$$
\min  _ {\sigma_ {A} (n)} \frac {\lambda_ {3}}{\sigma_ {X} ^ {4}} \sigma_ {A} ^ {4} (n) + \left(\frac {\sigma_ {X} ^ {2} (n) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (n)}\right) \sigma_ {A} ^ {2} (n) - \frac {\sigma_ {\Gamma} (n)}{\sigma_ {X} (n)} \sigma_ {A} (n) - 2 \lambda_ {4} \log \frac {\sigma_ {A} (n)}{\sigma_ {X} (n)}, \tag {13}
$$

with  $\sigma_{\Gamma}(n) = T(n,n)$ ,  $\mathbf{T} = \mathbf{U}_{U_XXY}\pmb{\Sigma}_{U_XXY}\mathbf{U}_{U_XXY}^T$ ,  $\forall n\in \mathcal{N}$  (we give the proof in Appendix A).

Discriminative encoding: Given the available data samples  $\mathbf{X}$  and the current estimate of the transform  $\mathbf{A}$ , define for simplicity  $\mathbf{Q} = \mathbf{A}\mathbf{X} \in \Re^{M \times L}$ . The discriminative representation estimation problem is formulated as:

$$
\min  _ {\mathbf {Y}} \| \mathbf {Q} - \mathbf {Y} \| _ {F} ^ {2} + \sum_ {c, k} \lambda_ {0} D \left(\mathbf {y} _ {c, k}; \boldsymbol {\tau} _ {c}\right) + \lambda_ {1} \| \mathbf {y} _ {c, k} \|. \tag {14}
$$

Note that (10) and consequently (14) is a well defined problem only if the model variables  $\pmb{\tau}_{c}$ , or the probability distributions  $p(\pmb{\tau}_{c})$  are known in advance. Nevertheless, we show that the approximation (6) leads to an efficient solution of (14). Assuming that  $\mathbf{Y}_{\backslash c}$  is given, then for any sample  $k \in \mathcal{K}$  from any class  $c \in \mathcal{C}$  by using the approximation (6), problem (14) reduces to a constrained projection:

$$
\min  _ {\mathbf {y} _ {c, k}} \left\| \mathbf {A} \mathbf {x} _ {c, k} - \mathbf {y} _ {c, k} \right\| _ {2} ^ {2} + \lambda_ {0} D \left(\mathbf {y} _ {c, k}; \boldsymbol {\tau} _ {c}\right) + \lambda_ {1} \left\| \mathbf {y} _ {c, k} \right\| _ {1}, \tag {15}
$$

and has a closed form solution as:

$$
\mathbf {y} _ {c, k} = \left(\max  \left(\mathbf {A x} _ {c, k} - \lambda_ {0} \mathbf {g} + \lambda_ {1} \mathbf {1}, \mathbf {0}\right) - \max  \left(- \mathbf {A x} _ {c, k} - \lambda_ {0} \mathbf {g} + \lambda_ {1} \mathbf {1}, \mathbf {0}\right)\right) \otimes \left(\mathbf {1} + 2 \lambda_ {0} \mathbf {s} _ {c}\right), \tag {16}
$$

where  $\oslash$  denotes Hadamard (element-wise) division,  $\mathbf{g} = \mathrm{sign}(\max (\mathbf{A}\mathbf{x}_{c,k},\mathbf{0}))\odot \mathbf{d}_c^+$  +  $\mathrm{sign}(\max (-\mathbf{A}\mathbf{x}_{c,k},\mathbf{0}))\odot \mathbf{d}_c^{-}$ ,  $\mathbf{d}_c^- = \sum_{c1\neq c}\sum_{k1}\mathbf{y}_{c1,k1}^-$  and  $\mathbf{d}_c^+ = \sum_{c1\neq c}\sum_{k1}\mathbf{y}_{c1,k1}^+$ ,  $\mathbf{s}_c = \sum_{c1\neq c}\sum_{k1}\mathbf{y}_{c1,k1}\odot \mathbf{y}_{c1,k1}$  (the proof is given in Appendix B).

We note that at convergence (which we do not prove here) we can only claim that a joint local maximum in  $(\mathbf{Y},\mathbf{A})$  of  $p(\mathbf{x}_{c,k},\mathbf{y}_{c,k},\boldsymbol{\tau}_c,\mathbf{A})$  has been reached, even if, as in this case, each optimization step achieves the (marginal)  $\epsilon$ -close and global optimal solution, respectively.

The exact steps of the proposed non-linear transform learning are described by Algorithm 1.

# 3 SENSITIVITY ANALYSIS AND INTERPRETATIONS

The similarity concentration measure provides possibility to measure the discriminative properties, their deviation, increase (or decrease) and the corresponding relations between different non-linear transform models across one domain or different domains, thereby quantifying their quality w.r.t. the discriminative properties.

# 3.1 SENSITIVITY ANALYSIS W.R.T. THE SIMILARITY CONCENTRATION MEASURES

To measure the ability for an increase in discriminative properties by a non-linear transform<sup>4</sup> we first have to define a notion for the discriminative properties on a data set under different non-linear transform models. Therefore, first we introduce the "special" base models and then analyze the properties of the similarity concentration measures under the change in model parameter and the relation between the base model and the proposed non-linear transform model (3).

Any data set  $\mathbf{X}$  in the original domain might have a transform model with parameters  $\mathcal{B}^N = \{\mathbf{A}_o\in$ $\Re^{N\times N},\pmb {\tau} = \mathbf{0}\in \mathfrak{R}_{+}^{N}\}$ , if  $\mathbf{A}_o = \mathbf{I}\in \mathcal{D}_+^N$  we refer to it as a base original model. Similarly as in the original domain, any data set  $\mathbf{Y}$  in the transform domain might have a transform model with parameters  $\mathcal{B}^M = \{\mathbf{A}_t\in \mathfrak{R}^{M\times M},\pmb {\tau} = \mathbf{0}\in \mathfrak{R}_+^M\}$ , if  $\mathbf{A}_t = \mathbf{I}\in \mathcal{D}_+^M$  we refer to it as a base transform model. Any base model, defined ether in the original domain  $\mathcal{B}^N$  or in the transform domain  $\mathcal{B}^M$ , has domain equal to the co-domain, since  $\mathbf{x}_{c,k} = \mathcal{T}^{\mathcal{B}^N}(\mathbf{x}_{c,k})$  and  $\mathbf{y}_{c,k} = \mathcal{T}^{\mathcal{B}^M}(\mathbf{y}_{c,k})$  holds trivially, for the respective sets of parameters  $\mathcal{B}^N = \{\mathbf{A}_o = \mathbf{I}\in \mathcal{D}_+^N,\pmb {\tau} = \mathbf{0}\in \mathfrak{R}_+^N\}$  and  $\mathcal{B}^M = \{\mathbf{A}_t = \mathbf{I}\in \mathcal{D}_+^M,\pmb {\tau} = \mathbf{0}\in \mathfrak{R}_+^M\}$ . This is illustrated with a diagram shown in Figure 3.

![](images/d545e3eb0451035cc6547bb9d8290994b34b6b74ff48c749d6d7d772e1356d97.jpg)  
Figure 3: The original and the transform domains under a non-linear transforms with a set of parameters  $\mathcal{B}^N$ ,  $\mathcal{P}$  and  $\mathcal{B}^M$ , note that for  $\mathcal{T}^{\mathcal{B}^N}$  and  $\mathcal{T}^{\mathcal{B}^M}$  the original and the transform domains are the same.

A base original model provides a possibility to compare it with any other non-linear transform model with parameters  $\mathcal{P} = \{\mathbf{A}\in \Re^{M\times N},\pmb {\tau}\in \Re_{+}^{M}\}$ . Additionally, note that for  $\mathcal{B}^M = \{\mathbf{A}_t = \mathbf{I}\in$ $\mathcal{D}^M,\pmb {\tau} = \mathbf{0}\in \Re^M\}$  we have that  $D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y}) = D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})$  and that  $D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y}) = D_{\ell_1}^{\mathcal{P}}(\mathbf{X})$ . It implies that the similarity concentrations can be analyzed as a function in the original domain under model  $\mathcal{P}$  or in the transform domain under model  $\mathcal{B}^M$ . The main relations considering the preservation of change in the similarity concentration between two models, defined not necessary in the same domain are stated by Proposition 1.

Lemma 1: The non-linear transform model (4) totally preserves the information in the change for the similarity concentration for a data set  $\mathbf{X}$  w.r.t. a small change in the parameters of the models  $\mathcal{B}^N$ ,  $\mathcal{B}^M$  and  $\mathcal{P}$  if  $\| \pmb{\delta}_o\|_* = 0$  and  $\| \pmb{\delta}_t\|_* = 0$  as

$$
(R 1): \mathbf {A} \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}}\right) = \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}} + \boldsymbol {\delta} _ {o}
$$

$$
(R 2): \mathbf {A} \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}}\right) ^ {T} = \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\delta} _ {t} \tag {17}
$$

$$
(R 3): \mathbf {A} \boldsymbol {\delta} _ {o} ^ {T} = \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\delta} _ {t} ^ {T},
$$

$$
\text{where}\boldsymbol{\delta}_{o} = \sum_{c,c1}\sum_{\substack{k,k1\\ k\neq k1}}\mathbf{z}_{c1,k1}\mathbf{x}_{c,k}^{T} + \mathbf{z}_{c,k}\mathbf{x}_{c1,k1}^{T},\frac{\partial D_{\ell_{1},c}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}} +\frac{\partial D_{\ell_{1}}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}} = \sum_{c,c1}\sum_{\substack{k,k1\\ k\neq k1}}\mathbf{z}_{c1,k1}\mathbf{x}_{c,k}^{T}.
$$

$$
\mathbf {y} _ {c 1, k 1} \mathbf {x} _ {c, k} ^ {T} + \mathbf {y} _ {c, k} \mathbf {x} _ {c 1, k 1} ^ {T}, \delta_ {t} = \sum_ {c, c 1} \sum_ {\substack {k, k 1 \\ k \neq k 1}} \mathbf {z} _ {c 1, k 1} \mathbf {y} _ {c, k} ^ {T} + \mathbf {z} _ {c, k} \mathbf {y} _ {c 1, k 1} ^ {T}, \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} +
$$

$$
\frac {\partial D _ {\ell_ {1}} ^ {B ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} = \sum_ {c, c 1} \sum_ {\stackrel {k, k 1} {k \neq k 1}} \mathbf {y} _ {c 1, k 1} \mathbf {y} _ {c, k} ^ {T} + \mathbf {y} _ {c, k} \mathbf {y} _ {c 1, k 1} ^ {T}, \frac {\partial D _ {\ell_ {1} , c} ^ {B ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {B ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} =
$$

$$
\begin{array}{l} \sum_ {c, c 1} \sum_ {\stackrel {k, k 1} {k \neq k 1}} \mathbf {z} _ {c 1, k 1} \mathbf {z} _ {c, k} ^ {T} + \mathbf {z} _ {c, k} \mathbf {z} _ {c 1, k 1} ^ {T},   \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} = \sum_ {c} \sum_ {k, k 1} \mathbf {x} _ {c, k} \mathbf {x} _ {c, k 1} ^ {T} + \mathbf {x} _ {c, k 1} \mathbf {x} _ {c, k} ^ {T} a n d \\ \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} = \sum_ {c, c 1, c \neq c 1} \sum_ {k, k 1} \mathbf {x} _ {c, k} \mathbf {x} _ {c 1, k 1} ^ {T} + \mathbf {x} _ {c 1, k 1} \mathbf {x} _ {c, k} ^ {T}. T h e p r o o f i s g i v e n i n A p p e n d i x C. \end{array}
$$

The terms  $\mathbf{z}_{c,k}$  represent the non-linear transform error vectors that appear in the model  $\mathbf{A}\mathbf{x}_{c,k} = \mathbf{y}_{c,k} + \mathbf{z}_{c,k}$  as a result of applying an element-wise non-liner operation  $\mathcal{H}_{\tau}$  to  $\mathbf{A}\mathbf{x}_{c,k}$ , i.e.,  $\mathbf{y}_{c,k} = \mathcal{H}_{\tau}(\mathbf{A}\mathbf{x}_{c,k})$ . As an example in the sparsifying transform model  $\mathbf{z}_{c,k}$  is the "loss of information", that is the information about the values of the elements in  $\mathbf{A}\mathbf{x}_{c,k}$  that are discarded. The terms  $\delta_{o}$  and  $\delta_{t}$  correlate the errors  $\mathbf{z}_{c,k}$  with the original data  $\mathbf{x}_{c1,k1}$  and transform data  $\mathbf{y}_{c1,k1}$ , respectively. Note that if there is no loss of information (in the earlier example it means that there is no thresholding and just a simple linear transform model is used) then  $\delta_{o} = 0$  and  $\delta_{t} = 0$ . Moreover,  $\delta_{o}$  and  $\delta_{t}$  bear important information about the discriminative properties in the transform domain.

The terms  $\frac{\partial D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}}$  and  $\frac{\partial D_{\ell_1}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}}$  represent the change of the similarity concentrations under infinitesimally small change of the parameter  $\mathbf{A}$  from the model  $\mathcal{P}$ . The terms  $\frac{\partial D_{\ell_1,c}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}\big|_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial D_{\ell_1}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}\big|_{\mathbf{A}_o = \mathbf{I}}$  have dual interpretation. Assuming metric  $\mathbf{A}_o = \mathbf{I}$ , then the first one is considered as a change of the similarity concentrations under infinitesimally small change of the space metric,

or equivalently under small metric perturbation. Conversely, assuming the data samples are distributed under a Gaussian distribution with parameters identity covariance matrix and zero mean, i.e.  $\mathbf{x}_{c,k} \sim \mathcal{N}(\boldsymbol{\mu} = \mathbf{0}, \boldsymbol{\Sigma} = \mathbf{I})$ , then  $\frac{\partial D_{\ell_1,c}^{B^N}(\mathbf{X})}{\partial \mathbf{A}_o} |_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial D_{\ell_1}^{B^N}(\mathbf{X})}{\partial \mathbf{A}_o} |_{\mathbf{A}_o = \mathbf{I}}$  represent the change of the similarity concentrations under small change in the assumption away from a Gaussian distribution.

Equation (R1) relates the base transform for the original domain  $\mathcal{B}^N$  with any arbitrary transform defined in the original domain  $\mathcal{P}$ . The relation (R2) is a result about the preservation of change in the similarity concentration between two models  $\mathcal{B}^M$  and  $\mathcal{P}$  defined on two different domains. Whereas (R3) gives the preservation of change in the similarity concentration between the error in the transform domain.

The next result highlights the relation between: the linear projection (by the linear map  $\mathbf{A}$  that appears in the model  $\mathcal{P}$ ) of the change in the similarity concentration under the model  $\mathcal{B}^N$  in the original domain and the change of the similarity concentration under the model  $\mathcal{B}^M$  in the transform domain.

This relation exists independently for  $\frac{\partial D_{\ell_1,c}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial D_{\ell_1}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}$ , nevertheless, we will define the summarized and the independent versions. Therefore, first we define  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}} = \frac{\partial D_{\ell_1,c}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}} + \frac{\partial D_{\ell_1}^{\mathcal{B}^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} = \frac{\partial D_{\ell_1,c}^{\mathcal{B}^N}(\mathbf{Y})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} + \frac{\partial D_{\ell_1}^{\mathcal{B}^N}(\mathbf{Y})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}}$  and rewrite (R1) as  $\mathbf{A}\frac{\partial\mathcal{J}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}} - \boldsymbol {\delta}_o = \frac{\partial D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}} +\frac{\partial D_{\ell_1}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}}$  then replace  $\frac{\partial D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}} +\frac{\partial D_{\ell_1}^{\mathcal{P}}(\mathbf{X})}{\partial\mathbf{A}}$  in (R2) by the same term in (R1), use (R3), reorder and we have the following result.

Lemma 2: For fixed  $\tau$  any non-linear transform model (4) preserves the information in the change of similarity concentrations w.r.t. a small change in  $\mathbf{A}$  by

$$
(R 4): \mathbf {A} \frac {\partial \mathcal {J} _ {\ell_ {1}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} \mathbf {A} ^ {T} = \frac {\partial \mathcal {J} _ {\ell_ {1}} (\mathbf {V})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} = \frac {\partial \mathcal {J} _ {\ell_ {1}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\xi} _ {c} + \boldsymbol {\xi},
$$

$$
(R 5): \mathbf {A} \frac {\partial D _ {\ell_ {1} , c} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} \mathbf {A} ^ {T} = \frac {\partial D _ {\ell_ {1} , c} (\mathbf {V})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} = \frac {\partial D _ {\ell_ {1} , c} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\xi} _ {c}, \tag {18}
$$

$$
(R 6): \mathbf {A} \frac {\partial D _ {\ell_ {1}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} \mathbf {A} ^ {T} = \frac {\partial D _ {\ell_ {1}} (\mathbf {V})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} = \frac {\partial D _ {\ell_ {1}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\xi},
$$

where  $\mathbf{V} = \mathbf{A}\mathbf{X},\xi_{c} + \xi = \frac{\partial\mathcal{J}_{\ell_{1}}(\mathbf{Z})}{\partial\mathbf{A}_{t}} |_{\mathbf{A}_{t} = \mathbf{I}} + \frac{\partial\mathcal{J}_{\ell_{1}}(\mathbf{Y};\mathbf{Z})}{\partial\mathbf{A}_{t}} |_{\mathbf{A}_{t} = \mathbf{I}},\frac{\partial\mathcal{J}_{\ell_{1}}(\mathbf{Z})}{\partial\mathbf{A}_{t}} |_{\mathbf{A}_{t} = \mathbf{I}} = \frac{\partial D_{\ell_{1},c}^{B^{N}}(\mathbf{Z})}{\partial\mathbf{A}_{t}} |_{\mathbf{A}_{t} = \mathbf{I}} +$ $\frac{\partial D_{\ell_1}^{\mathcal{B}^N}(\mathbf{Z})}{\partial\mathbf{A}_t} |_{\mathbf{A}_t = \mathbf{I}}$  and  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y};\mathbf{Z})}{\partial\mathbf{A}_t} |_{\mathbf{A}_t = \mathbf{I}} = \pmb {\delta}_t + \pmb {\delta}_t^T$

Note that  $\frac{\partial^n\mathcal{J}_{\ell_1}(\mathbf{X})}{\partial^n\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}} = 4\frac{\partial^{n - 1}\mathcal{J}_{\ell_1}(\mathbf{X})}{\partial^{n - 1}\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial^n\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial^n\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} = 4\frac{\partial^{n - 1}\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial^{n - 1}\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}}$  therefore,  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}} = \frac{1}{4}\frac{\partial^2\mathcal{J}_{\ell_1}(\mathbf{X})}{\partial^2\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}$  and  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} = \frac{1}{4}\frac{\partial^2\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial^2\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}}$  might be interpreted as Fisher information matrices evaluated at  $\mathbf{A}_o = \mathbf{I}\in \mathcal{D}^N$  and  $\mathbf{A}_t = \mathbf{I}\in \mathcal{D}^M$ . The expressions by (R4) actually relate the metric in the original domain under the model  $\mathcal{B}^N$  to the induced metric in the transform domain for the model  $\mathcal{B}^M$ , with induction done by the model  $\mathcal{P}$  with parameter set  $\{\mathbf{A}\in \Re^{M\times N},\pmb {\tau}\in \Re_{+}^{M}\}$ . Moreover, the model  $\mathcal{P}$  might describe a transform domain with a non-smooth manifold. Since the manifolds of the original and the transform domain under the models  $\mathcal{B}^N$  and  $\mathcal{B}^M$  are smooth the analysis of their relations reveals insights about the relation between the manifolds under the models  $\mathcal{B}^N$  and  $\mathcal{P}$ . The terms  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Z})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}}$  and  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y};\mathbf{Z})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}}$  carry out the information about the breaks and the discontinuities of the regularity and smoothness in the manifold induced by the model  $\mathcal{P}$ . Furthermore, if  $\| \frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Z})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} \|_* = 0$  and  $\| \frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y};\mathbf{Z})}{\partial\mathbf{A}_t}|_{\mathbf{A}_t = \mathbf{I}} \|_* = 0$  then (R4) in information geometry Amari (2013) is seen as change of coordinates on a manifold, where the intrinsic properties of curvature remain unchanged under different parametrization.

# 3.2 A MEASURE FOR THE DISCRIMINATIVE PROPERTIES AND ITS BOUNDS

This paper proposes a notion for the discriminative properties of a data set under a non-linear transform named as discrimination power, based on a measure for the relations between the concentrations  $D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})$  and  $D_{\ell_1}^{\mathcal{P}}(\mathbf{X})$ .

![](images/ef02b24e702af795145406658c5b44fe6478b397fcfa975240bb10a83496f4bd.jpg)  
Figure 4: The relation for the definition of the discrimination power in the original and the transform domain under the base models  $\mathcal{B}^N$  and  $\mathcal{B}^M$ .

<table><tr><td></td><td>Cn(A)</td><td>μ(A)</td><td>te[min]</td></tr><tr><td>D1</td><td>2.21</td><td>0.03</td><td>5.10</td></tr><tr><td>D2</td><td>1.80</td><td>0.02</td><td>5.45</td></tr><tr><td>D3</td><td>2.12</td><td>0.02</td><td>6.55</td></tr><tr><td>D4</td><td>0.08</td><td>0.02</td><td>8.92</td></tr><tr><td>D5</td><td>6.01</td><td>0.01</td><td>12.8</td></tr><tr><td>D6</td><td>33.1</td><td>0.02</td><td>30.1</td></tr><tr><td>D7</td><td>1.60</td><td>0.02</td><td>5.00</td></tr></table>

<table><tr><td>I^O</td><td>I^RT</td><td>I^ST*</td><td>I^NT</td></tr><tr><td>0.03</td><td>0.18</td><td>0.68</td><td>1.98</td></tr><tr><td>0.02</td><td>0.10</td><td>1.30</td><td>1.79</td></tr><tr><td>0.00</td><td>0.01</td><td>0.71</td><td>1.61</td></tr><tr><td>0.08</td><td>0.61</td><td>0.89</td><td>1.89</td></tr><tr><td>0.01</td><td>0.16</td><td>1.02</td><td>2.12</td></tr><tr><td>0.06</td><td>0.53</td><td>1.36</td><td>3.36</td></tr><tr><td>0.13</td><td>0.63</td><td>1.06</td><td>1.96</td></tr></table>

Table 2: The conditioning number  $C_n(\mathbf{A})$  and the expected mutual coherence  $\mu(\mathbf{A})$  for the learned transform  $\mathbf{A}$ , The execution time  $t_e[min]$  in minutes of the proposed algorithm for 28 iterations at the transform domain dimensionality  $M = 19000$ .

Proposition 1: The discrimination power for any dataset  $\mathbf{X} \in \Re^{M \times CK}$  under any transform with parameter set  $\mathcal{P}$  is defined as:

$$
\mathcal {I} ^ {t} = \log \left(D _ {\ell_ {1}, c} ^ {\mathcal {P}} (\mathbf {X})\right) - \log \left(D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X}) + \epsilon\right) = \log \left(D _ {\ell_ {1}, c} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})\right) - \log \left(D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {Y}) + \epsilon\right). \tag {19}
$$

Remark 1: The advantage of this measure is that it logarithmically signifies the difference between  $D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X})$  and  $D_{\ell_1}^{\mathcal{P}}(\mathbf{X})^5$ .

The definition about the discrimination power of the data set  $\mathbf{X}$ , but, now under a model with a parameter set  $\mathcal{B}^N$  is equivalent to the one defined for  $\mathcal{I}^t$ , we denote it as  $\mathcal{I}^o$ . An illustration is given by a diagram shown in Figure 4.

The bound on the discrimination power is given by the following result.

Theorem 1: The discrimination power for any data set  $\mathbf{X} \in \Re^{N \times CK}$  under any transform with parameter set  $\mathcal{P}$  is bounded as:

$$
\log \left(\lambda_ {\min } \left(\mathbf {A} ^ {T} \mathbf {A}\right)\right) + \log \left(\frac {T r \left\{\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} \mid_ {\mathbf {A} _ {o} = \mathbf {I}} \right\}}{D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {A X}) + \epsilon}\right) \leq \mathcal {I} ^ {t} \leq \log \left(D _ {\ell_ {1}, c} ^ {\mathcal {B} ^ {M}} (\mathbf {A X})\right) - \log (\epsilon) \tag {20}
$$

The proof is given in Appendix  $D$

At first the resulting bounds might look counterintuitive since the loss of information seems to increase the discrimination power. This fact is true, however, up to a certain limit. Therefore, it is important to distinguish two main conclusions.

First, for any model with a set of parameters  $\mathcal{P}$  for which there is no loss of information  $\| \pmb{\xi}_c + \pmb{\xi}\|_* = 0$ , the only condition for the increase in the discrimination power is  $D_{\ell_1}^{\mathcal{B}^{\mathcal{N}}}(\mathbf{X}) \geq D_{\ell_1}^{\mathcal{B}^{\mathcal{M}}}(\mathbf{AX})$  and  $D_{\ell_1,c}^{\mathcal{B}^{\mathcal{N}}}(\mathbf{X}) \leq D_{\ell_1,c}^{\mathcal{B}^{\mathcal{M}}}(\mathbf{AX})$ .

![](images/ef1f851514dc679b35b5d0f05d8d6d29e2571ba4630ace76d0d03009f7029426.jpg)

![](images/8f6b1060dafbe523fd45087bfee64e581cafb4508cee82cf0aa99d66a9c163ba.jpg)

![](images/8ed8d946e135faeff02a626cc012c686d92c6484084fcb3740bbfdd62eec3aa1.jpg)  
Figure 5: The evolution of the similarity concentrations  $C_1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C_2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$ , their ratio  $C_1 / C_2$  and the discrimination power  $\log (C_1 / C_2) = \mathcal{I}^t$  during the learning of the non-linear transform with transform dimension  $M = 19000$ .

![](images/293c5311ca917c507d2d98d42e38a3b7e68ca4026798441f40fdd94d78bb1674.jpg)

![](images/12b3c47ff0f1fe77dc1156984eb6e065a52c7f89af81d92e973d02c9a69d6aa3.jpg)  
Figure 6: The ratio  $C1 / C2$  of the similarity concentrations  $C1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and the discrimination power  $\log (C1 / C2) = \mathcal{I}^t$  for randomly chosen subsets from all of the used databases under a non-linear transform with transform dimension  $M = 19000$  and varying thresholding parameter  $\tau = \lambda \mathbf{1}$ .

![](images/860ff256f21d7d155c8c425065272fa92e28e57db532665e57fe6b3a31ccb95d.jpg)

Second, in the rest of the cases for which  $D_{\ell_1}^{\mathcal{B}^{\mathcal{N}}}(\mathbf{X}) \geq D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and  $D_{\ell_1,c}^{\mathcal{B}^{\mathcal{N}}}(\mathbf{X}) \leq D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  holds true it will be possible to increase the discrimination power. Moreover, there is a trade-off between the increase in discrimination power as a result of loss of information  $\| \pmb {\xi}_c + \pmb {\xi}\|_* > 0$ .

# 4 NUMERICAL EXPERIMENTS

The numerical experiments are summarized in two different parts. In the first series of the experiments the properties of the learned map  $\mathbf{A}$  for the proposed algorithm are investigated. We evaluate the computational efficiency, as run time  $t_e[\min]$ , the conditioning number  $C_n(\mathbf{A})^6$ , the expected

![](images/6d7976faa7b52dc0909e5bf02182a304559f10e6c6ce17e7c4e497bcb7c9e206.jpg)  
Figure 7: The conditioning number  $C_n(\mathbf{A})$  and the expected mutual coherence  $\mu(\mathbf{A})$  for the learned linear transform  $\mathbf{A}$  at different dimensionality  $M \in \{100, 1150, 2200, 3250, 4300, 5350, 6400, 7450, 8500, 9550, 10600, 11650, 12700, 13750, 14800, 15850, 16900, 17950, 19000\}$ .

![](images/428acbb819b634495767aaf7cbdf1e9b4a0a94dd8c18d74e6d5b567e6841b36f.jpg)

![](images/8a0f14ef22106b3cb3126d2f59076e573400c529ddf3cea0901f6579d71a5f8f.jpg)

![](images/207dba02895afe7deb741a66bef9b3c8868e2c262bb9d9f84aa15fb41387b44d.jpg)

![](images/40b4d41d31c84a64af7ec3a06cd3ef856bfa6cba6eb28b30711f3d18d3617612.jpg)  
Figure 8: The similarity concentrations  $C1 = D_{\ell_1,c}^{B^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{B^M}(\mathbf{Y})$ , their ratio  $C1 / C2$  and the discrimination power  $\log(C1 / C2) = \mathcal{I}^t$  on a subset of the transform data using learned non-linear transform at different dimensionality  $M \in \{100, 1150, 2200, 3250, 4300, 5350, 6400, 7450, 8500, 9550, 10600, 11650, 12700, 13750, 14800, 15850, 16900, 17950, 19000\}$ .

![](images/263442b341a6dfa6ba13607533004fdfb29e4b2f8d630b7a2b07be58cb49f420.jpg)

mutual coherence  $\mu (\mathbf{A})$  and the discrimination power across several databases for a learned nonlinear transforms having different dimensionality. Additionally, a comparison between the resulting discrimination power in the original domain, after transform by a random matrix (having Gaussian random samples as entries and transform dimension of  $M = 19000$ ) and after a learned non-linear transform having transform dimension  $M = 19000$  without and with support dissimilarity prior, denoted as  $\mathcal{I}^0,\mathcal{I}^{RT},\mathcal{I}^{ST^*}$  and  $\mathcal{I}^{NT}$ , respectively, is estimated and presented.

The second part evaluates a comparison of the discrimination power between the proposed algorithm and different supervised dictionary learning methods (SDL) Ramirez et al. (2010), Yang et al. (2011a), Vu et al. (2015) and Vu & Monga (2016a). This comparison considers a setup where the used data sets are spited into a training and test set, moreover, the learning is performed on the

<table><tr><td></td><td>D1</td><td>D7</td><td></td><td>D1 Acc. [%]</td><td></td><td>D7 Acc. [%]</td></tr><tr><td>\( \mathcal{I}^{DLSI} \)</td><td>0.71</td><td>0.67</td><td>DLSI</td><td>96.5</td><td>DLSI</td><td>98.74</td></tr><tr><td>\( \mathcal{I}^{FDDL} \)</td><td>0.87</td><td>0.63</td><td>FDDL</td><td>97.5</td><td>FDDL</td><td>96.31</td></tr><tr><td>\( \mathcal{I}^{COPAR} \)</td><td>0.57</td><td>0.54</td><td>COPAR</td><td>98.3</td><td>COPAR</td><td>96.41</td></tr><tr><td>\( \mathcal{I}^{LRSDL} \)</td><td>0.42</td><td>0.40</td><td>LRSDL</td><td>98.7</td><td>LRSDL</td><td>-</td></tr><tr><td>\( \mathcal{I}^{NT} \)</td><td>0.98</td><td>0.81</td><td>NT</td><td>99.7</td><td>NT</td><td>99.02</td></tr><tr><td></td><td>a)</td><td></td><td></td><td>b)</td><td></td><td>c)</td></tr></table>

Table 3: a) The discrimination power for the methods  $DLSIR$  Ramirez et al. (2010),  $FDDL$  Yang et al. (2011a),  $COPAR$  Vu et al. (2015) and  $LRSDL$  Vu & Monga (2016a) and the proposed non-liner transform  $NT$ , b) and c) The recognition results on the Extended Yale B and MNIST database.

![](images/e565a7a647682b6cb249968912f64966233c5331f2ff946877061665fd7d00a4.jpg)

![](images/ae17b974bf910388cd68ad763c3275e0fbd681c07aa1ec890a7ddabe9d2a6fb6.jpg)

![](images/0abf6529425d8a5ce53eaa28af84ffdf5ac77ab943165efc4f41664670739585.jpg)  
Figure 9: a) and b) The recognition results and the discrimination power on the Extended Yale B and MNIST databases, respectively, using a non-linear transform with different dimensionality  $M$  and linear SVM classifier on top of the transform representation.  
Figure 10: a) and b) The expected loss per transform dimension  $\mathbb{E}[\frac{\|\mathbf{z}_{c,k}\|_2}{M}] = \mathbb{E}[\|\mathbf{A}\mathbf{x}_{c,k} - \mathbf{y}_{c,k}\|_2 / M]$  and the discrimination power on the Extended Yale B and MNIST databases, respectively, on the transform representation  $\mathbf{Y}$ , obtained by using a non-linear transform  $\mathcal{T}^{\mathcal{P}}$  at different dimensionality  $M$ .

![](images/b4456686b0f936b2918367fe90ba3c2ff00d8e7a0cd6f5a5953024d1f61bd41d.jpg)

training set and the evaluation is performed on the test set. In the same series of experiments the recognition accuracy for two data sets is also computed and compared.

![](images/c94dd5fa630db79faa189c74b3922b59c6f455b4c200b5821265de2b84333cd4.jpg)

![](images/838c608ad260c6d25d244eb862d228ee443f223bb6d04a8644e3ef780617088b.jpg)

![](images/73e11ee4beb777c70888297f04c0a742f95da35667cfea97daa5eaab84790c8c.jpg)  
Figure 11: The ratio of the similarity concentrations similarity concentrations  $C_1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C_2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and the discrimination power  $\log(C_1 / C_2) = \mathcal{I}^t$  for the Extended Yale B and MNIST databases under non-linear transforms having different transform dimension  $M$  and varying thresholding parameter  $\tau = \lambda \mathbf{1}$ .

![](images/5e09ba7660f67c9477713f88d987a6a6d20cdb2787de5ebc65276f782d64416b.jpg)

Data sets and algorithms set up The used data sets are Extended YALE B (D1)Georghiades et al. (2001), AR  $(D2)$  Martinez & Benavente (1998), Norb  $(D3)$  LeCun et al. (2004), Coil-20  $(D4)$  Nene et al. (1996), Clatech101  $(D5)$  LeCun et al. (2008), UKB  $(D6)$  Nister & Stewenius (2006) and MNIST  $(D7)$  Lecun & Cortes. All the images from the respective datasets were downscaled to resolutions  $21\times 21$ ,  $32\times 28$ ,  $24\times 24$ ,  $20\times 25$ ,  $21\times 21$ ,  $20\times 25$ ,  $28\times 28$ , respectively, and are normalized to unit variance. Considering the used implementation of the algorithm note that the singular value decomposition for large matrix has high computational complexity. However,  $\mathbf{A} - \hat{\mathbf{A}}$ , where  $\hat{\mathbf{A}}$  is estimated as a solution in the transform update step, can be considered as an proximal operator Parikh & Boyd (2014) for the gradient of the objective (11). Additionally, instead of using all of the available data samples  $\mathbf{X}$  a subset of them might be used. Therefore, one simple on-line variant for the update of  $\mathbf{A}$  w.r.t. a subset of the available training set has the form  $\mathbf{A}^{t + 1} = \mathbf{A}^t -\rho (\mathbf{A}^t -\hat{\mathbf{A}}^t)$  with  $\rho$  predefined step size. In the numerical experiments we use the on-line variant of the algorithm (the convergence analysis for this variant of the algorithm is left for future work). the parameters  $\lambda_0$  and  $\lambda_{1}$  are set such that the resulting non-linear transform representation has very small number of non-zeros w.r.t. the transform dimension, in the experiments here this number is set to be 15. The rest of the parameters are set as  $\{\lambda_2,\lambda_3,\lambda_4\} = \{1000000,1000000,1000000\}$ . The algorithm is initialized with a random matrix having i.i.d. Gaussian (zero mean, variance one) entries and is terminated after the 28th iteration. The results are obtained as average of 3 runs. An implementation presented in Vu & Monga (2016a) was used to learn the dictionaries and estimate the sparse codes for the respective supervised dictionary learning methods (SDL) Ramirez et al. (2010), Yang et al. (2011a), Vu et al. (2015) and Vu & Monga (2016a).

Linear map properties, the change in the similarity concentrations and the discrimination power The conditioning number and the expected coherence for the learned transforms are shown on Table 2. The learned transforms for all the databases have good conditioning numbers and low expected coherence. The running time  $t_e$ , measured in minutes and the number of used dimensions, denoted as  $M$  are also shown in Table 2. The learned transforms for all the data sets have relatively low execution time, regardless of a transform dimensions 19000. The discrimination power is sig-

nificantly increased in the transform domain  $\mathcal{I}^{NT}$  compared to the one in the original domain  $\mathcal{I}^O$  and is higher than  $\mathcal{I}^{ST^*}$  and  $\mathcal{I}^{RT}$ .

The evolution of the similarity concentrations  $C_1 = D_{\ell_1,c}^{B^M}(\mathbf{Y})$  and  $C_2 = D_{\ell_1}^{B^M}(\mathbf{Y})$ , their ratio  $C_1 / C_2$  and the discrimination power  $\log (C_1 / C_2) = \mathcal{I}^t$  for a subsets of the used databases after applying a non-linear transform with transform dimension  $M = 19000$  is shown on Figure 5. It is important to note that the similarity concentrations  $C_1 = D_{\ell_1,c}^{B^M}(\mathbf{Y})$  and  $C_2 = D_{\ell_1}^{B^M}(\mathbf{Y})$  are decreasing, meaning that there is a loss in information. However, how this loss affects the resulting similarity concentration is crucial for the discriminative properties. As shown on Figure 5 the slope of decrease for  $C_2$  is stronger then the slope of decrease for  $C_1$ , therefore, the discrimination power increases per iteration. For the Coil-20 ( $D4$ ) database there is a fluctuation. This is explained by the fact that during learning we used a small number of data samples from the same database and that in the data there is high variability.

The conditioning number and the expected coherence for the learned transforms for all the databases at different transform dimensions are shown on Figure 7. We see that the value of both the conditioning number and the coherence is reducing and converging to a common values, respectively, implying that the conditioning and the coherence constraints are effective.

The ratio  $C_1 / C_2$  between the similarity concentrations  $C_1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C_2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and the discrimination power  $\log (C_1 / C_2) = \mathcal{I}^t$  on a subsets of the used databases after applying a non-linear transform with transform dimension  $M = 19000$  and varying the thresholding parameter  $\pmb{\tau} = \lambda \mathbf{1}$  is shown on Figure 6. We used 70 different values for the parameter  $\lambda$ , sampled uniformly from the interval  $(0, (\max_{c,k} \max_m |\mathbf{a}_m^T \mathbf{x}_{c,k}|))$ . The result were obtained using a non-linear transforms learned with one value for the parameter  $\lambda$  for all the databases. Since all the databases have different variabilities and the amount of available data is different, this result suggests that per different database there should be different optimal value of the parameter  $\lambda$ .

The similarity concentrations  $C1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$ , their ratio  $C1 / C2$  and the discrimination power  $\log (C1 / C2) = \mathcal{I}^t$  for a subsets of the used databases after applying a non-linear transform having transform dimensions  $M \in \{100, 1150, 2200, 3250, 4300, 5350, 6400, 7450, 8500, 9550, 10600, 11650, 12700, 13750, 14800, 15850, 16900, 17950, 19000\}$  is shown on Figure 8. We can see a similar behavior as previous, that is, the similarity concentrations  $C1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  are decreasing, but, the slope of decrease for  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  is stronger. Therefore, the discrimination power increases as the transform dimension increases.

SNTL vs SDL discrimination power and recognition performance The proposed method is compared with DLSI Ramirez et al. (2010), FDDL Yang et al. (2011a), COPAR Vu et al. (2015) and LRSDL Vu & Monga (2016a). Half of the data samples from the data set Extended-YALE-B, sampled at random are used for learning and the remaining other half are used for evaluation. Considering the MNIST database the training set is used for learning and the test set is used for evaluation. We compute both the discrimination power and the recognition accuracy on the test sets.

The dictionary size (transform dimension  $M$ ) is set to be equal to 150, 75, 1515, 3825, 570, 150, 300 for the corresponding databases, respectively, in all of the comparing algorithms. The discrimination power of the comparing methods is denoted as  $\mathcal{I}^{DLSI}$ ,  $\mathcal{I}^{FDDL}$ ,  $\mathcal{I}^{COPAR}$  and  $\mathcal{I}^{LRSDL}$ , respectively. The recognition results for the methods  $DLSI$ ,  $FDDL$ ,  $COPAR$  and  $LRSDL$  on the data sets Extended YALE B and MNIST were not computed here, rather than that we use the best reported result from the respective papers Ramirez et al. (2010), Yang et al. (2011a), Vu et al. (2015) and Vu & Monga (2016a). Considering the proposed algorithm the transform was learned for the transform dimensions  $M$  equal to [100, 500, 1500, 4000] and [1000, 4000, 6000, 12000], respectively, for the used data sets. After the transform was learned the transform data samples were computed for the respective training and test sets. Then, the transform training data samples were used as features to learn a linear SVM classifier in one-against-all regime. Finally, the evaluation was performed on the respective test sets.

The results are shown on Table 3 a), b) and c). The discrimination power of the proposed nonlinear transform is higher that the discrimination power of the comparing methods. Whereas the

recognition accuracy is higher for high dimensionality of the proposed method and outperforms the SDL methods at dimensionality 4000 and 12000.

The results about the accuracy of recognition and the expected loss measured as  $\mathbb{E}\left[\frac{\|\mathbf{z}_{c,k}\|_2}{M}\right] = \mathbb{E}\left[\|\mathbf{A}\mathbf{x}_{c,k} - \mathbf{y}_{c,k}\|_2 / M\right]$  per the discrimination power at different transform dimension are shown on Figure 9 and Figure 10, respectively. It is interesting to highlight that as the discrimination power at different transform dimension increases it also increases the accuracy of recognition, moreover, the results on these two data bases show that this increase is approximately linear. On the other hand the expected loss per transform dimension decreases as the discrimination power at different transform dimension increases.

The results about the ratio  $C1 / C2$  between the similarity concentrations  $C1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and the discrimination power  $\log (C1 / C2) = \mathcal{I}^t$  on Extended-Yale-B and MNIST databases after applying a non-linear transform with transform dimensions  $M = \{100,500,1500,4000\}$  and  $M = \{1000,4000,6000,12000\}$ , respectively, and varying the thresholding parameter  $\pmb{\tau} = \lambda \mathbf{1}$  are shown on Figure 11. We again used 70 different values for the parameter  $\lambda$ , sampled uniformly from the interval  $(0, (\max_{c,k} \max_m |\mathbf{a}_m^T \mathbf{x}_{c,k}|))$ , that were different for the used databases. The result were obtained using a non-linear transforms learned with optimally choose values (by using cross-validation) of the parameter  $\lambda$  for the two different databases. As expected we can see the extreme points of the ratio between the similarity concentrations  $C1 = D_{\ell_1,c}^{\mathcal{B}^M}(\mathbf{Y})$  and  $C2 = D_{\ell_1}^{\mathcal{B}^M}(\mathbf{Y})$  and the discrimination power  $\log (C1 / C2) = \mathcal{I}^t$  is around the optimal values of the parameter  $\pmb{\tau} = \lambda \mathbf{1}$ .

# 5 CONCLUSION

This paper presented an analysis on the discriminative properties for non-linear transform models expressible as two step transform, linear mapping (step 1) followed by an element wise non-linearity (step 2). A novel discriminative prior was proposed and the properties around the model and the proposed prior were investigated. A low complexity learning algorithm was presented with the proposed priors.

The preliminary results w.r.t. the introduced measures and the recognition accuracy on the used databases showed promising performance. We showed that it is possible to increase the discrimination power with information loss. Moreover, we highlight that when expanding to high dimensional space with non-linear transform how the loss of information reflects the similarity concentrations is crucial for the discriminative properties.

A study on the recognition capabilities for other databases are our next future extensions. An analysis for the synthesis model and transform based auto-encoder towards generalization and fair comparison between different transforms and encoding methods is other future direction. In the line of the encoding one might consider to minimize directly the difference or the ratio between the introduced similarity concentrations or the discrimination power.

In a different direction the extensions covering the sufficient conditions for increase in discrimination power in the transform domain, together with an analysis for a deep architecture where per single layer we have one non-linear transform are left for our future work.

A.

# THE GLOBAL OPTIMAL SOLUTION

Given the current estimate of  $\mathbf{Y}$  the estimate of the transform  $\mathbf{A}$  is solution to the following problem:

$$
\left\{\hat {\mathbf {A}} \right\} = \underset {\mathbf {A}} {\operatorname {a r g m i n}} \| \mathbf {A} \mathbf {X} - \mathbf {Y} \| _ {F} ^ {2} + \lambda_ {2} \| \mathbf {A} ^ {T} \mathbf {A} \| _ {F} + \lambda_ {3} \| \mathbf {A} \mathbf {A} ^ {T} - \mathbf {I} \| _ {F} - \lambda_ {4} (\log | \det  \mathbf {A} ^ {T} \mathbf {A} |). \tag {21}
$$

Theorem 1 (global optimal solution): Given  $\mathbf{Y} \in \Re^{M \times L}$  and  $\mathbf{X} \in \Re^{N \times L}$ , if and only if the joint decomposition:

$$
\mathbf {X} \mathbf {X} ^ {T} = \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {2} \mathbf {U} _ {X} ^ {T} \tag {22}
$$

$$
\mathbf {X} \mathbf {Y} ^ {T} = \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X Y} \mathbf {V} _ {X Y} ^ {T},
$$

exists, where  $\mathbf{U}_X\in \Re^{N\times N}$  is orthonormal,  $\mathbf{V}_{XY}\in \Re^{M\times N}$  is per columns orthonormal and  $\pmb{\Sigma}_{X},\pmb{\Sigma}_{XY}\in \Re^{N\times N}$  are diagonal matrices with positive diagonal elements, then (21) has a global minimum as:

$$
\hat {\mathbf {A}} = \mathbf {V} _ {X Y} \boldsymbol {\Sigma} _ {A} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}, \tag {23}
$$

and  $\Sigma_A(n,n) = \sigma_A(n) > 0$ ,  $\sigma_A(n),\forall n$  are solutions to

$$
\frac {\lambda_ {3}}{\sigma_ {X} ^ {4} (i)} \sigma_ {A} ^ {4} (i) + \left(\frac {\sigma_ {X} ^ {2} (i) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (i)}\right) \sigma_ {A} ^ {2} (i) - \frac {\sigma_ {X Y} (i)}{\sigma_ {X} (i)} \sigma_ {A} (i) - 2 \lambda_ {4} \log \frac {\sigma_ {A} (i)}{\sigma_ {X} (i)} = 0. \tag {24}
$$

# Proof of Theorem 1:

Consider the equivalent trace form of (21) we have:

$$
\min  _ {\mathbf {A}} \operatorname {T r} \left\{\left(\mathbf {A} \mathbf {X} - \mathbf {Y}\right) ^ {T} \left(\mathbf {A} \mathbf {X} - \mathbf {Y}\right) \right\} + \lambda_ {2} \operatorname {T r} \left\{\mathbf {A} ^ {T} \mathbf {A} \right\} + \lambda_ {3} \operatorname {T r} \left\{\left(\mathbf {A} \mathbf {A} ^ {T} - \mathbf {I}\right) ^ {T} \left(\mathbf {A} \mathbf {A} ^ {T} - \mathbf {I}\right) \right\} - \tag {25}
$$

$$
\lambda_ {4} (\log | \det  \mathbf {A} ^ {T} \mathbf {A} |).
$$

Note that  $\forall \lambda_2 \geq 0$ ,  $\mathbf{X}\mathbf{X}^T + \lambda_2\mathbf{I}$  is symmetric positive definite matrix with all eigenvalues nonnegative, therefore it decomposes as:

$$
\mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} \mathbf {U} _ {X} ^ {T} \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} \mathbf {U} _ {X} ^ {T} = \mathbf {X X} ^ {T} + \lambda_ {2} \mathbf {I}. \tag {26}
$$

Let

$$
\mathbf {A} = \mathbf {B D}, \mathbf {D} = \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}. \tag {27}
$$

Define

$$
g _ {1} = \mathbf {B D X Y} ^ {T}, g _ {2} = \mathbf {B B} ^ {T}, g _ {3} = \left(\mathbf {B D D} ^ {T} \mathbf {B}\right) \left(\mathbf {B D D} ^ {T} \mathbf {B}\right) ^ {T}
$$

$$
g _ {4} = g _ {5} = (\mathbf {B D}) (\mathbf {B D}) ^ {T} \sum_ {m = 1} ^ {M} \mathbf {P} _ {m, M} (\mathbf {B D}) (\mathbf {B D}) ^ {T} \mathbf {P} _ {m, M} \tag {28}
$$

$$
g _ {6} = \sum_ {m = 1} ^ {M} \mathbf {P} _ {m, M} (\mathbf {B D}) (\mathbf {B D}) ^ {T} \mathbf {P} _ {m, M}, g _ {7} = \log | \det \mathbf {B D D} ^ {T} \mathbf {B} ^ {T} |.
$$

where  $\mathbf{P}_{m,M} \in \mathcal{D}^M$ ,  $P_{m,M}(m1,m1) = 1$  if  $m1 = m$  and  $P_{m,M}(m1,m1) = 0$  if  $m1 \neq m$ . Then (21) equivalently is:

$$
\min  _ {\mathbf {B}} - T r \left\{g _ {1} \right\} + T r \left\{g _ {2} \right\} + \lambda_ {3} \left(T r \left\{g _ {3} \right\} - T r \left\{g _ {4} \right\}\right) + \lambda_ {3} \left(T r \left\{g _ {5} \right\} - T r \left\{g _ {6} \right\}\right) - \lambda_ {4} g _ {7}. \tag {29}
$$

Assume that  $\mathbf{B}$  decomposes as:

$$
\mathbf {U} _ {B} \boldsymbol {\Sigma} _ {B} \mathbf {V} _ {B} ^ {T}, \tag {30}
$$

where  $\Sigma_B$  is diagonal with positive diagonal elements,  $\mathbf{U}_B$  is column orthogonal and  $\mathbf{V}_B$  is orthogonal square matrix. Moreover, let the following decomposition of  $\mathbf{X}\mathbf{Y}^T$  exists

$$
\mathbf {X} \mathbf {Y} ^ {T} = \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X Y} \mathbf {V} _ {X Y} ^ {T}, \tag {31}
$$

substitute as

$$
\mathbf {U} _ {B} = \mathbf {V} _ {X Y}, \mathbf {V} _ {B} = \mathbf {U} _ {X}, \tag {32}
$$

then

$$
T r \left\{g _ {1} \right\} = T r \left\{\mathbf {U} _ {B} \boldsymbol {\Sigma} _ {B} \mathbf {V} _ {B} \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T} \mathbf {X Y} ^ {T} \right\} = T r \left\{\boldsymbol {\Sigma} _ {B} \boldsymbol {\Sigma} _ {X} ^ {- 1} \boldsymbol {\Sigma} _ {X Y} \right\}. \tag {33}
$$

The term

$$
T r \left\{g _ {2} \right\} = T r \left\{\left(\mathbf {U} _ {B} \boldsymbol {\Sigma} _ {B} \mathbf {V} _ {B}\right) \left(\mathbf {U} _ {B} \boldsymbol {\Sigma} _ {B} \mathbf {V} _ {B}\right) ^ {T} \right\} = T r \left\{\boldsymbol {\Sigma} _ {B} ^ {2} \right\}. \tag {34}
$$

Define  $\Sigma = \Sigma_B\Sigma_X^{-2}\Sigma_B$  then

$$
T r \left\{g _ {3} \right\} = T r \left\{\left(\boldsymbol {\Sigma} \mathbf {U} _ {B} ^ {T} \mathbf {U} _ {B}\right) \left(\boldsymbol {\Sigma} \mathbf {U} _ {B} ^ {T} \mathbf {U} _ {B}\right) ^ {T} \right\} = T r \left\{\boldsymbol {\Sigma} \boldsymbol {\Sigma} \right\} = T r \left\{\boldsymbol {\Sigma} _ {B} ^ {4} \boldsymbol {\Sigma} _ {X} ^ {- 4} \right\}. \tag {35}
$$

Note that

$$
T r \left\{g _ {4} \right\} =
$$

$$
T r \left\{\mathbf {B} \mathbf {D} \mathbf {D} ^ {T} \mathbf {B} ^ {T} \sum_ {m = 1} ^ {M} \mathbf {P} _ {m, M} \mathbf {B} \mathbf {D} \mathbf {D} ^ {T} \mathbf {B} ^ {T} \mathbf {P} _ {m, M} \right\} = T r \left\{\left(\sum_ {m = 1} ^ {M} \mathbf {P} _ {m, M} \mathbf {U} _ {X} \boldsymbol {\Sigma} \mathbf {U} _ {X} ^ {T} \mathbf {P} _ {m, M}\right) ^ {2} \right\}. \tag {36}
$$

Therefore,  $\lambda_3Tr\{g_4\} = \lambda_3Tr\{g_5\}$ . The term  $Tr\{g_6\} = Tr\{\left(\pmb{\Sigma}_B\pmb{\Sigma}_X^{-1}\right)^2\}$ . The term  $g_7$  is

$$
\log \left| \det  \mathbf {A} ^ {T} \mathbf {A} \right| = \log \left| \det  \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {Y} ^ {- 1} \boldsymbol {\Sigma} _ {B} ^ {2} \boldsymbol {\Sigma} _ {Y} ^ {- 1} \mathbf {U} _ {X} \right| = \left| \det  \boldsymbol {\Sigma} _ {Y} ^ {- 1} \boldsymbol {\Sigma} _ {B} ^ {2} \boldsymbol {\Sigma} _ {Y} ^ {- 1} \right|. \tag {37}
$$

Finally (21) is reduced to:

$$
\min  _ {\boldsymbol {\Sigma} _ {B}} \sum_ {n = 1} ^ {N} \frac {\lambda_ {3}}{\sigma_ {X} ^ {4} (n)} \sigma_ {B} ^ {4} (n) + \left(\frac {\sigma_ {X} ^ {2} (n) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (i)}\right) \sigma_ {B} ^ {2} (n) - \frac {\sigma_ {X Y} (n)}{\sigma_ {X} (n)} \sigma_ {B} (n) - 2 \lambda_ {4} \log \frac {\sigma_ {B} (n)}{\sigma_ {X} (n)}. \tag {38}
$$

Equalling to zero the first order derivative w.r.t  $\sigma_B(n)$  of the objective in (38) and multiplying by  $\sigma_B(n)$  gives:

$$
4 \frac {\lambda_ {3}}{\sigma_ {X} ^ {4} (i)} \sigma_ {B} ^ {4} (i) + 2 \left(\frac {\sigma_ {X} ^ {2} (i) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (i)}\right) \sigma_ {B} ^ {2} (i) - \frac {\sigma_ {X Y} (i)}{\sigma_ {X} (i)} \sigma_ {B} (i) - 2 \lambda_ {4} = 0. \tag {39}
$$

A closed form solution to (39) exists and depends on the discriminant  $\Delta$  of the quartic polynomial. Moreover, since  $4\frac{\lambda_4}{\sigma_X^4}$  is positive a global minimum exists as

$$
\mathbf {A} = \mathbf {V} _ {X Y} \boldsymbol {\Sigma} _ {B} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}. \tag {40}
$$

![](images/15e2366a97c5b1818d6354397792dd954b48ccce9ffb6a2fb3a427b638a9ef76.jpg)

# THE  $\epsilon$ -CLOSE Closed FORM APPROXIMATION

# Proof of Proposition 1:

Consider the equivalent trace form of (21). Note that  $\forall \lambda_{2} \geq 0$ ,  $\mathbf{X}\mathbf{X}^{T} + \lambda_{2}\mathbf{I}$  is symmetric positive definite matrix with all eigenvalues non-negative, therefore it decomposes as

$$
\mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} \mathbf {U} _ {X} ^ {T} \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} \mathbf {U} _ {X} ^ {T} = \mathbf {X X} ^ {T} + \lambda_ {2} \mathbf {I}. \tag {41}
$$

Let

$$
\mathbf {D} = \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}, \text {a n d} \mathbf {d e f i n e} \mathbf {A} = \mathbf {B D}, \mathbf {U} _ {U _ {X} X Y} \boldsymbol {\Sigma} _ {U _ {X} X Y} \mathbf {V} _ {U _ {X} X Y} ^ {T} = \mathbf {U} _ {X} \mathbf {X} \mathbf {Y} ^ {T}. \tag {42}
$$

Assume that  $\mathbf{B}$  decomposes as  $\mathbf{U}_B\pmb {\Sigma}_B\mathbf{V}_B^T$  , where  $\pmb{\Sigma}_{B}$  is diagonal with positive diagonal elements,  $\mathbf{U}_B$  is column orthogonal and  $\mathbf{V}_B$  is orthogonal square matrix. Let

$$
\mathbf {U} _ {B} = \left(\mathbf {U} _ {U _ {X} X Y} \mathbf {V} _ {U _ {X} X Y} ^ {T}\right) ^ {T}, \mathbf {V} _ {B} = \mathbf {U} _ {X}, \tag {43}
$$

then

$$
T r \left\{\mathbf {A} \mathbf {X} \mathbf {Y} ^ {T} \right\} = T r \left\{\mathbf {B} \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T} \mathbf {X} \mathbf {Y} ^ {T} \right\} = T r \left\{\mathbf {V} _ {U _ {X} X Y} \mathbf {U} _ {U _ {X} X Y} ^ {T} \boldsymbol {\Sigma} _ {B} \right. \tag {44}
$$

$$
\left. \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {U _ {X} X Y} \boldsymbol {\Sigma} _ {U _ {X} X Y} \mathbf {V} _ {U _ {X} X Y} ^ {T} \right\}
$$

Moreover, since  $\mathbf{U}_B = (\mathbf{U}_{U_XX_Y}\mathbf{V}_{U_XX_Y}^T)^T$ ,  $\mathbf{V}_B = \mathbf{U}_X$ , using Mirsky (1959) and Neumann (1937), note that

$$
\min  _ {\boldsymbol {\Sigma} _ {B}} \max  _ {\mathbf {U} _ {B}, \mathbf {V} _ {B}} T r \left\{\mathbf {U} _ {B} \boldsymbol {\Sigma} _ {B} \mathbf {V} _ {B} ^ {T} \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} \mathbf {X Y} ^ {T} \right\} \leq \min  _ {\boldsymbol {\Sigma} _ {B}} T r \left\{\boldsymbol {\Sigma} _ {\Gamma} \boldsymbol {\Sigma} _ {B} \boldsymbol {\Sigma} _ {X} ^ {- 1} \right\} \tag {45}
$$

where  $\pmb{\Sigma}_{\Gamma}$  is diagonal matrix having diagonal elements  $\sigma_{\Gamma}(n) = T(n,n)$ ,  $\forall n \in \mathcal{N}$  and  $\mathbf{T} = \mathbf{U}_{UXXY}\pmb{\Sigma}_{UXXY}\mathbf{U}_{UXXY}^T$ . The term  $Tr\{\mathbf{BB}^T\} = Tr\{(\mathbf{U}_B\pmb{\Sigma}_B\mathbf{V}_B)(\mathbf{U}_B\pmb{\Sigma}_B\mathbf{V}_B)^T\} = Tr\{\pmb{\Sigma}_B^2\}$  and the term  $Tr\{(\mathbf{AA}^T)^2\} = Tr\{(\pmb{\Sigma}\mathbf{U}_B^T\mathbf{U}_B)(\pmb{\Sigma}\mathbf{U}_B^T\mathbf{U}_B)^T\} = Tr\{\pmb{\Sigma}_B^4\pmb{\Sigma}_X^{-4}\}$ . The term  $Tr\{\mathbf{AA}^T\} = Tr\{(\pmb{\Sigma}_B\pmb{\Sigma}_X^{-1})^2\}$ . The term  $\log |\det \mathbf{A}^T\mathbf{A}|$  is

$$
\log | \det \mathbf {A} ^ {T} \mathbf {A} | = \log | \det \mathbf {U} _ {X} \boldsymbol {\Sigma} _ {Y} ^ {- 1} \boldsymbol {\Sigma} _ {B} ^ {2} \boldsymbol {\Sigma} _ {Y} ^ {- 1} \mathbf {U} _ {X} | = | \det \boldsymbol {\Sigma} _ {Y} ^ {- 1} \boldsymbol {\Sigma} _ {B} ^ {2} \boldsymbol {\Sigma} _ {Y} ^ {- 1} |. \tag {46}
$$

Finally, using the bound (45), the approximation of (21) is reduced to

$$
\min _ {\boldsymbol {\Sigma} _ {B}} \sum_ {i = 1} ^ {N} \frac {\lambda_ {3}}{\sigma_ {X} ^ {4} (n)} \sigma_ {B} ^ {4} (n) + \left(\frac {\sigma_ {X} ^ {2} (n) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (n)}\right) \sigma_ {B} ^ {2} (n) - \frac {\sigma_ {\Gamma} (n)}{\sigma_ {X} (n)} \sigma_ {B} (n) - 2 \lambda_ {4} \log \frac {\sigma_ {B} (n)}{\sigma_ {X} (n)}. \tag {47}
$$

Equaling to zero the first order derivative w.r.t  $\sigma_B(n)$  of the objective for (47) and multiplying by  $\sigma_B(n)$  gives:

$$
4 \frac {\lambda_ {3}}{\sigma_ {X} ^ {4} (n)} \sigma_ {B} ^ {4} (n) + 2 \left(\frac {\sigma_ {X} ^ {2} (n) - 2 \lambda_ {3}}{\sigma_ {X} ^ {2} (n)}\right) \sigma_ {B} ^ {2} (n) - \frac {\sigma_ {\Gamma} (n)}{\sigma_ {X} (n)} \sigma_ {B} (n) - 2 \lambda_ {4} = 0. \tag {48}
$$

A closed form solution to (47) exists and it depends on the discriminant  $\Delta$  of the quartic polynomial. Moreover, since  $4\frac{\lambda_4}{\sigma_X^4}$  is positive a global minimum to (47) exists. Therefore, having  $\mathbf{U}_B = (\mathbf{U}_{U_XXY} \mathbf{V}_{U_XXY}^T)^T$  and  $\mathbf{V}_B = \mathbf{U}_X$  with the solution for  $\boldsymbol{\Sigma}_B$  by (47) gives the  $\epsilon$ -close closed form approximative solution to problem (21) as

$$
\mathbf {A} = \left(\mathbf {U} _ {U X X Y} \mathbf {V} _ {U X X Y} ^ {T}\right) ^ {T} \boldsymbol {\Sigma} _ {B} \boldsymbol {\Sigma} _ {X} ^ {- 1} \mathbf {U} _ {X} ^ {T}, \tag {49}
$$

where by (45) it implies that the  $\epsilon$ -close closed form solution is a lower bound on the solution to (21).

# APPENDIX B.

Let  $\mathbf{y}_{c,k} = \mathbf{y}_{c,k}^{+} + \mathbf{y}_{c,k}^{-}$ ,  $\mathbf{y}_{c,k}^{+} \in \Re_{+}^{M}$  and  $\mathbf{y}_{c,k}^{-} \in \Re_{-}^{M}$ . Note that the term  $D_{\ell_1}^{\mathcal{P}}(\mathbf{X})$  is defined as:

$$
D_{\ell_{1}}^{\mathcal{P}}(\mathbf{X}) = \sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}\| \mathbf{y}_{c1,k1}^{+}\odot \mathbf{y}_{c2,k2}^{+}\|_{1} + \sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}\| \mathbf{y}_{c1,k1}^{-}\odot \mathbf{y}_{c2,k2}^{-}\|_{1} =
$$

$$
\sum_ {\substack {c 1, c 2 \\ c 1 \neq c 2}} \sum_ {k 1, k 2} \left| \mathbf {y} _ {c 1, k 1} ^ {+} \right| ^ {T} \left| \mathbf {y} _ {c 2, k 2} ^ {+} \right| + \sum_ {\substack {c 1, c 2 \\ c 1 \neq c 2}} \sum_ {k 1, k 2} \left| \mathbf {y} _ {c 1, k 1} ^ {-} \right| ^ {T} \left| \mathbf {y} _ {c 2, k 2} ^ {-} \right| = \tag{50}
$$

$$
|\mathbf{y}_{c1,k1}^{+}|^{T}\sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}|\mathbf{y}_{c2,k2}^{+}| + |\mathbf{y}_{c1,k1}^{-}|^{T}\sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}|\mathbf{y}_{c2,k2}^{-}| =
$$

$$
\left| \mathbf {y} _ {c 1, k 1} ^ {+} \right| ^ {T} \mathbf {g} ^ {+} + \left| \mathbf {y} _ {c 1, k 1} ^ {-} \right| ^ {T} \mathbf {g} ^ {-},
$$

where  $\mathbf{g}^{+} = \sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}\left|\mathbf{y}_{c2,k2}^{+}\right|,\mathbf{g}^{-} = \sum_{\substack{c1,c2\\ c1\neq c2}}\sum_{k1,k2}\left|\mathbf{y}_{c2,k2}^{-}\right|$  and we abuse notation by denoting  $|\mathbf{y}_{c1,k1}|$  as the vector whose elements are the absolute values of the elements in  $\mathbf{y}_{c1,k1}$ .

Note that the term  $S_{\ell_2}^{\mathcal{P}}(\mathbf{X})$  is defined as:

$$
S _ {\ell_ {2}} ^ {\mathcal {P}} (\mathbf {X}) = \sum_ {\stackrel {c 2} {c 2 \neq c 1}} \sum_ {k 2} \| \mathbf {y} _ {c 1, k 1} \odot \mathbf {y} _ {c 2, k 2} \| _ {2} ^ {2} =
$$

$$
\sum_{\substack{c2\\ c2\neq c1}}\sum_{k2}(\mathbf{y}_{c1,k1}\odot \mathbf{y}_{c2,k2})^{T}(\mathbf{y}_{c1,k1}\odot \mathbf{y}_{c2,k2}) =
$$

$$
\sum_ {\substack {c 2 \\ c 2 \neq c 1}} \sum_ {k 2} (\mathbf {y} _ {c 1, k 1} \odot \mathbf {y} _ {c 1, k 1}) ^ {T} (\mathbf {y} _ {c 2, k 2} \odot \mathbf {y} _ {c 2, k 2}) = \tag{51}
$$

$$
\left(\mathbf{y}_{c1,k1}\odot \mathbf{y}_{c1,k1}\right)^{T}\left(\sum_{\substack{c2\\ c2\neq c1}}\sum_{k2}\mathbf{y}_{c2,k2}\odot \mathbf{y}_{c2,k2}\right) = \left(\mathbf{y}_{c1,k1}\odot \mathbf{y}_{c1,k1}\right)^{T}\mathbf{s}_{c}
$$

where  $\mathbf{s}_c = \left(\sum_{\substack{c2\\ c2\neq c1}}\sum_{k2}\mathbf{y}_{c2,k2}\odot \mathbf{y}_{c2,k2}\right)$

Consider the related problem

$$
\begin{array}{r l} & \min  _ {\mathbf {y} _ {c 1, k 1}} \| \mathbf {y} _ {c 1, k 1} - \mathbf {q} _ {c 1, k 1} \| _ {2} ^ {2} + \\ & \quad \lambda_ {0} \left(\left(\mathbf {y} _ {c 1, k 1} ^ {+}\right) ^ {T} \mathbf {g} ^ {+} + \left(\mathbf {y} _ {c 1, k 1} ^ {-}\right) ^ {T} \mathbf {g} ^ {-} + \left(\mathbf {y} _ {c 1, k 1} \odot \mathbf {y} _ {c 1, k 1}\right) ^ {T} \mathbf {s} _ {c}\right) + \lambda_ {1} \| \mathbf {y} _ {c 1, k 1} \| _ {1}, \end{array} \tag {52}
$$

by taking the first order derivative w.r.t.  $\mathbf{y}_{c1,k1}$  we have that

$$
\left(\mathbf {y} _ {c 1, k 1} - \mathbf {q} _ {c 1, k 1}\right) + \lambda_ {1} \operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1}\right) + \tag {53}
$$

$$
\lambda_ {0} \left(\operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1} ^ {+}\right) \odot \mathbf {g} ^ {+} + \operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1} ^ {-}\right) \odot \mathbf {g} ^ {-} + \mathbf {y} _ {c 1, k 1} \odot \mathbf {s} _ {c}\right) = \mathbf {0},
$$

take the sign magnitude decomposition of  $\mathbf{y}_{c1,k1} = \mathrm{sign}\bigl (\mathbf{y}_{c1,k1}\bigr)\odot \big|\mathbf{y}_{c1,k1}\big|$  then we have

$$
\operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1}\right) \odot \left| \mathbf {y} _ {c 1, k 1} \right| \odot \left(\mathbf {1} + 2 \lambda_ {0} \mathbf {s} _ {c}\right) - \operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1}\right) \odot \left| \mathbf {q} _ {c 1, k 1} \right| +
$$

$$
\lambda_ {1} \operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1}\right) + \lambda_ {0} \left(\operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1} ^ {+}\right) \odot \mathbf {g} ^ {+} + \operatorname {s i g n} \left(\mathbf {y} _ {c 1, k 1} ^ {-}\right) \odot \mathbf {g} ^ {-}\right) = \mathbf {0}.
$$

Let the sign of  $\mathbf{y}_{c1,k1}$ , i.e.  $\mathrm{sign}(\mathbf{y}_{c1,k1})$  be equal to the sign of  $\mathrm{sign}(\mathbf{q}_{c1,k1})$ , and Hadamard multiply from the left side by  $\mathrm{sign}(\mathbf{q}_{c1,k1})$  then we have

$$
\left(\mathbf {1} + 2 \lambda_ {0} \mathbf {s} _ {c}\right) \odot | \mathbf {y} _ {c 1, k 1} | - | \mathbf {q} _ {c 1, k 1} | + \lambda_ {1} \mathbf {1} + \lambda_ {0} (\operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1}\right) \odot \operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1} ^ {+}\right) \odot \mathbf {g} ^ {+} + \left. \right. \tag {55}
$$

$$
\operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1}\right) \odot \operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1} ^ {-}\right) \odot \mathbf {g} ^ {-}) = \mathbf {0},
$$

note that  $\mathrm{sign}(\mathbf{q}_{c1,k1})\odot \mathrm{sign}(\mathbf{q}_{c1,k1}^{+}) = \mathrm{sign}(\mathbf{q}_{c1,k1}^{+})$  and that  $\mathrm{sign}(\mathbf{q}_{c1,k1})\odot \mathrm{sign}(\mathbf{q}_{c1,k1}^{-}) = \mathrm{sign}(-\mathbf{q}_{c1,k1}^{-})$ , therefore we have

$$
\begin{array}{l} \left| \mathbf {y} _ {c 1, k 1} \right| = \\ \left(\left| \mathbf {q} _ {c 1, k 1} \right| - \lambda_ {1} \mathbf {1} - \lambda_ {0} \left(\operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1} ^ {+}\right) \odot \mathbf {g} ^ {+} + \operatorname {s i g n} \left(- \mathbf {q} _ {c 1, k 1} ^ {-}\right) \odot \mathbf {g} ^ {-}\right)\right) \oslash (\mathbf {1} + 2 \lambda_ {0} \mathbf {s} _ {c}). \end{array} \tag {56}
$$

Define  $\mathbf{G}^{c1,k1}$  as diagonal matrix with positive diagonal elements  $G^{c1,k1}(m,m) = z(m),\forall m\in \mathcal{M}$  where:

$$
\mathbf {z} = \operatorname {s i g n} \left(\max  \left(\mathbf {q} _ {c 1, k 1}, \mathbf {0}\right)\right) \odot \mathbf {g} ^ {+} + \operatorname {s i g n} \left(\max  \left(- \mathbf {q} _ {c 1, k 1}, \mathbf {0}\right)\right) \odot \mathbf {g} ^ {-}, \tag {57}
$$

and note that  $\mathbf{G}^{c1,k1}\mathbf{1} = \mathbf{z}$  then

$$
\left| \mathbf {y} _ {c 1, k 1} \right| = \left| \mathbf {q} _ {c 1, k 1} \right| - \lambda_ {1} \mathbf {1} - \lambda_ {0} \left(\mathbf {G} ^ {c 1, k 1} \mathbf {1}\right), \tag {58}
$$

since the magnitude might be only positive  $|\mathbf{y}_{c1,k1}| = \max (|\mathbf{q}_{c1,k1}| - \lambda_1\mathbf{1} - \lambda_0(\mathbf{G}^{c1,k1}\mathbf{1}),\mathbf{0})$ . Therefore, the closed form solution to (52) is:

$$
\mathbf {y} _ {c 1, k 1} = \operatorname {s i g n} \left(\mathbf {q} _ {c 1, k 1}\right) \odot \max  \left(\left| \mathbf {q} _ {c 1, k 1} \right| - \lambda_ {1} \mathbf {1} - \lambda_ {0} \mathbf {G} ^ {c 1, k 1} \mathbf {1}, \mathbf {0}\right) \otimes (\mathbf {1} + 2 \lambda_ {0} \mathbf {s} _ {c}). \tag {59}
$$

which completes the proof  $\square$

# APPENDIX C.

Note that for the model  $\mathcal{P}^t$  we have that

$$
\begin{array}{l} \mathbf {y} = \mathcal {T} (\mathbf {A x}) = \max  (\mathbf {A x} - \boldsymbol {\tau}, \mathbf {0}) - \max  (- \mathbf {A x} - \boldsymbol {\tau}, \mathbf {0}), \\ \quad \mathcal {T} (\mathbf {A x}) = \left( \begin{array}{l l l l} \mathbf {A x} & \mathbf {0} & \mathbf {0} & \mathbf {0} \end{array} \right) \end{array} \tag {60}
$$

$$
\mathbf {q} = \mathcal {T} (\mathbf {A x}) = \max  (\mathbf {A g} - \boldsymbol {\tau}, \mathbf {0}) - \max  (- \mathbf {A g} - \boldsymbol {\tau}, \mathbf {0}),
$$

since

$$
\operatorname {s i g n} (a) \max  (| a | - b, 0) = \max  (a - b, 0) - \max  (- a - b, 0), \tag {61}
$$

The first order derivative of the divergence  $D_{\ell_1}^{\mathcal{P}^t}(\mathbf{x};\mathbf{g})$  w.r.t. the parameter  $\mathbf{A}$  is:

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P} ^ {t}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A}} = \tag {62}
$$

$$
\frac {\partial (\max (\mathbf {A x} - \boldsymbol {\tau} , \mathbf {0}) ^ {T} \max (\mathbf {A g} - \boldsymbol {\tau} , \mathbf {0}))}{\partial \mathbf {A}} + \frac {\partial (\max (- \mathbf {A x} - \boldsymbol {\tau} , \mathbf {0}) ^ {T} \max (- \mathbf {A g} - \boldsymbol {\tau} , \mathbf {0}))}{\partial \mathbf {A}}
$$

we assume that the threshold parameter  $\tau$  is chosen such that the vector  $|\mathbf{A}\mathbf{x}| - \tau$  (or for any other  $\mathbf{q}$ , the vector  $|\mathbf{A}\mathbf{q}| - \tau$ ) has least one non-zero element, then

$$
\frac {\partial \left(\max \left(\mathbf {A} \mathbf {x} - \boldsymbol {\tau} , \mathbf {0}\right) ^ {T} \max \left(\mathbf {A} \mathbf {g} - \boldsymbol {\tau} , \mathbf {0}\right)\right)}{\partial \mathbf {A}} = \max  (\mathbf {A} \mathbf {g} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {x} ^ {T} + \max  (\mathbf {A} \mathbf {x} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {g} ^ {T}, \tag {63}
$$

and

$$
\frac {\partial \left(\max \left(- \mathbf {A} \mathbf {x} - \boldsymbol {\tau} , \mathbf {0}\right) ^ {T} \max \left(- \mathbf {A} \mathbf {g} - \boldsymbol {\tau} , \mathbf {0}\right)\right)}{\partial \mathbf {A}} = \tag {64}
$$

$$
- \max (- \mathbf {A} \mathbf {g} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {x} ^ {T} - \max (- \mathbf {A} \mathbf {x} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {g} ^ {T},
$$

combining (63) and (64) we have that

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P} ^ {t}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A}} =
$$

$$
\max  (\mathbf {A} \mathbf {g} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {x} ^ {T} + \max  (\mathbf {A} \mathbf {x} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {g} ^ {T} - \tag {65}
$$

$$
(\max (- \mathbf {A} \mathbf {g} + \boldsymbol {\tau}, \mathbf {0}) \mathbf {x} ^ {T} + \max (- \mathbf {A} \mathbf {x} - \boldsymbol {\tau}, \mathbf {0}) \mathbf {g} ^ {T}) =
$$

$$
\mathbf {q} \mathbf {x} ^ {T} + \mathbf {y} \mathbf {g} ^ {T}
$$

where

$$
y (m) = \operatorname {s i g n} \left(\mathbf {a} _ {m} ^ {T} \mathbf {x}\right) \max  \left(\left| \mathbf {a} _ {m} ^ {T} \mathbf {x} \right| - \tau (m), 0\right) \tag {66}
$$

$$
q (m) = \operatorname {s i g n} \left(\mathbf {a} _ {m} ^ {T} \mathbf {g}\right) \max  \left(\left| \mathbf {a} _ {m} ^ {T} \mathbf {g} \right| - \tau (m), 0\right),
$$

$\forall m\in \mathcal{M}$

Similarity, note that for the model  $\mathcal{P}_0^o = \{\mathbf{A}_o,\pmb {\tau} = \mathbf{0}\}$  we have that

$$
\mathbf {x} _ {o} = \mathcal {T} ^ {\mathcal {P} _ {0} ^ {o}} (\mathbf {A} _ {o} \mathbf {x}) =
$$

$$
\operatorname {s i g n} \left(\mathbf {A} _ {o} \mathbf {x}\right) \odot \max  \left(\left| \mathbf {A} _ {o} \mathbf {x} \right| - \mathbf {0}, \mathbf {0}\right) =
$$

$$
\max  \left(\mathbf {A} _ {o} \mathbf {x} - \mathbf {0}, \mathbf {0}\right) - \max  \left(- \mathbf {A} _ {o} \mathbf {x} - \mathbf {0}, \mathbf {0}\right), \tag {67}
$$

$$
\mathbf {g} _ {o} = \mathcal {T} ^ {\mathcal {P} _ {0} ^ {o}} (\mathbf {A} _ {o} \mathbf {g}) = \tag {67}
$$

$$
\operatorname {s i g n} \left(\mathbf {A} _ {o} \mathbf {g}\right) \odot \max  \left(\left| \mathbf {A} _ {o} \mathbf {g} \right| - \mathbf {g}, 0\right) =
$$

$$
\max  (\mathbf {A} _ {o} \mathbf {g} - \mathbf {0}, \mathbf {0}) - \max  (- \mathbf {A} _ {o} \mathbf {g} - \mathbf {0}, \mathbf {0}).
$$

The first order derivative of the divergence  $D_{\ell_1}^{\mathcal{P}_0^o}(\mathbf{x}; \mathbf{g})$  w.r.t  $\mathbf{A}_o$  is:

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P} _ {0} ^ {o}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A} _ {o}} = \mathbf {x} _ {o} \mathbf {g} ^ {T} + \mathbf {g} _ {o} \mathbf {x} ^ {T}, \tag {68}
$$

note that at  $\mathbf{A}_o = \mathbf{I}$ ,  $\mathcal{P}_0^t = \mathcal{B}^N$  and we have that

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} = \mathbf {x g} ^ {T} + \mathbf {g x} ^ {T} \tag {69}
$$

Also for the model  $\mathcal{P}_0^t = \{\mathbf{A}_t,\pmb {\tau} = \mathbf{0}\}$  we have that

$$
\mathbf {y} _ {t} = \mathcal {T} ^ {\mathcal {P} _ {0} ^ {t}} (\mathbf {A} _ {t} \mathbf {y}) =
$$

$$
\operatorname {s i g n} \left(\mathbf {A} _ {t} \mathbf {y}\right) \odot \max  \left(\left| \mathbf {A} _ {t} \mathbf {y} \right| - \mathbf {0}, \mathbf {0}\right) =
$$

$$
\max  _ {\mathcal {D} ^ {t}} \left(\mathbf {A} _ {t} \mathbf {y} - \mathbf {0}, \mathbf {0}\right) - \max  (- \mathbf {A} _ {t} \mathbf {y} - \mathbf {0}, \mathbf {0}), \tag {70}
$$

$$
\mathbf {q} _ {t} = \mathcal {T} ^ {\mathcal {P} _ {0} ^ {t}} (\mathbf {A} _ {t} \mathbf {q}) =
$$

$$
\operatorname {s i g n} \left(\mathbf {A} _ {t} \mathbf {q}\right) \odot \max  \left(\left| \mathbf {A} _ {t} \mathbf {q} \right| - \mathbf {q}, \mathbf {0}\right) =
$$

$$
\max  (\mathbf {A} _ {t} \mathbf {q} - \mathbf {0}, \mathbf {0}) - \max  (- \mathbf {A} _ {t} \mathbf {q} - \mathbf {0}, \mathbf {0}).
$$

The first order derivative of the divergence  $D_{\ell_1}^{\mathcal{P}_0^t}(\mathbf{y}; \mathbf{q})$  w.r.t  $\mathbf{A}_t$  is:

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P} _ {0} ^ {t}} (\mathbf {y} ; \mathbf {q})}{\partial \mathbf {A} _ {t}} = \mathbf {y} _ {t} \mathbf {q} ^ {T} + \mathbf {q} _ {t} \mathbf {y} ^ {T}, \tag {71}
$$

note that at  $\mathbf{A}_t = \mathbf{I}$ ,  $\mathcal{P}_0^t = \mathcal{B}^M$  and we have that

$$
\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {y} ; \mathbf {q})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {o} = \mathbf {I}} = \mathbf {y q} ^ {T} + \mathbf {q y} ^ {T} \tag {72}
$$

Consider the following

$$
\begin{array}{l}\mathbf {A g} = \mathbf {q} + \mathbf {z} _ {1} / \mathbf {x} ^ {T}\\\mathbf {A x} = \mathbf {y} + \mathbf {z} _ {2} / \mathbf {g} ^ {T}\end{array}\rightarrow \left\{\begin{array}{l}\mathbf {A x g} ^ {T} = \mathbf {y g} ^ {T} + \mathbf {z} _ {1} \mathbf {g} ^ {T}\\\mathbf {A g x} ^ {T} = \mathbf {q x} ^ {T} + \mathbf {z} _ {2} \mathbf {x} ^ {T}\end{array}\right. \tag {73}
$$

$$
\mathbf {A} (\mathbf {x g} ^ {T} + \mathbf {g x} ^ {T}) = \mathbf {y g} ^ {T} + \mathbf {q x} ^ {T} + \mathbf {z _ {1}} \mathbf {g} ^ {T} + \mathbf {z _ {2}} \mathbf {x} ^ {T}
$$

where

$$
\mathbf {z} _ {1} = \mathbf {A} \mathbf {x} - \operatorname {s i g n} (\mathbf {A} \mathbf {x}) \max  (| \mathbf {A} \mathbf {x} | - \tau \mathbf {1}, \mathbf {0}) \tag {74}
$$

$$
\mathbf {z} _ {2} = \mathbf {A} \mathbf {g} - \operatorname {s i g n} (\mathbf {A} \mathbf {g}) \max  (| \mathbf {A} \mathbf {g} | - \tau \mathbf {1}, \mathbf {0})
$$

A closer look at (73) reveals us that

$$
\left. \mathbf {A} \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A} _ {o}} \right| _ {\mathbf {A} _ {o} = \mathbf {I}} = \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A}} + \delta_ {o} ^ {z _ {1}, z _ {2}} \tag {75}
$$

where

$$
\boldsymbol {\delta} _ {o} ^ {z _ {1}, z _ {2}} = \mathbf {z} _ {1} \mathbf {g} ^ {T} + \mathbf {z} _ {2} \mathbf {x} ^ {T} \tag {76}
$$

By similar construction applied to the rest of the pairs of data samples we have the result:

$$
\mathbf {A} \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} \mid_ {\mathbf {A} _ {o} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} \mid_ {\mathbf {A} _ {o} = \mathbf {I}}\right) = \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X})}{\partial \mathbf {A}} + \boldsymbol {\delta} _ {o} \square \tag {77}
$$

Note that

$$
\begin{array}{l}\mathbf {A x} = \mathbf {y} + \mathbf {z} _ {1} / \mathbf {q} ^ {T}\\\mathbf {A g} = \mathbf {q} + \mathbf {z} _ {2} / \mathbf {y} ^ {T}\end{array}\rightarrow \left\{\begin{array}{l}\mathbf {A x q} ^ {T} = \mathbf {y q} ^ {T} + \mathbf {z} _ {1} \mathbf {q} ^ {T}\\\mathbf {A g y} ^ {T} = \mathbf {q y} ^ {T} + \mathbf {z} _ {2} \mathbf {y} ^ {T}\end{array}\right. \tag {78}
$$

$$
\mathbf {A} (\mathbf {x y} ^ {T} + \mathbf {q g} ^ {T}) = \mathbf {y q} ^ {T} + \mathbf {q y} ^ {T} + \mathbf {z _ {1}} \mathbf {q} ^ {T} + \mathbf {z _ {2}} \mathbf {y} ^ {T},
$$

where

$$
\mathbf {z} _ {1} = \mathbf {A} \mathbf {x} - \operatorname {s i g n} (\mathbf {A} \mathbf {x}) \max  (| \mathbf {A} \mathbf {x} | - \tau \mathbf {1}, \mathbf {0}) \tag {79}
$$

$$
\mathbf {z} _ {2} = \mathbf {A} \mathbf {g} - \operatorname {s i g n} (\mathbf {A} \mathbf {g}) \max  (| \mathbf {A} \mathbf {g} | - \tau \mathbf {1}, \mathbf {0})
$$

A closer look at (78) reveals us

$$
\mathbf {A} \left(\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {x} ; \mathbf {g})}{\partial \mathbf {A}}\right) ^ {T} = \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {y} ; \mathbf {q})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \boldsymbol {\delta} _ {t} ^ {z 1, z 2}. \tag {80}
$$

where

$$
\boldsymbol {\delta} _ {t} ^ {z _ {1}, z _ {2}} = \mathbf {z} _ {1} \mathbf {q} ^ {T} + \mathbf {z} _ {2} \mathbf {y} ^ {T} \tag {81}
$$

By similar construction applied to the rest of the pairs of data samples we have the result:

$$
\mathbf {A} \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {P} ^ {t}} (\mathbf {X})}{\partial \mathbf {A}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {P} ^ {t}} (\mathbf {X})}{\partial \mathbf {A}}\right) ^ {T} = \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} \mid_ {\mathbf {A} _ {t} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {Y})}{\partial \mathbf {A} _ {t}} \mid_ {\mathbf {A} _ {t} = \mathbf {I}}\right) + \delta_ {t} \square \tag {82}
$$

Note that

$$
\begin{array}{l}\mathbf {A x} = \mathbf {y} + \mathbf {z} _ {1} / \mathbf {z} _ {2} ^ {T}\\\mathbf {A g} = \mathbf {q} + \mathbf {z} _ {2} / \mathbf {z} _ {1} ^ {T}\end{array}\rightarrow \left\{\begin{array}{l}\mathbf {A x z} _ {2} ^ {T} = \mathbf {y z} _ {2} ^ {T} + \mathbf {z} _ {1} \mathbf {z} _ {2} ^ {T}\\\mathbf {A g z} _ {1} ^ {T} = \mathbf {q z} _ {1} ^ {T} + \mathbf {z} _ {2} \mathbf {z} _ {1} ^ {T}\end{array}\right. \tag {83}
$$

$$
\mathbf {A} (\mathbf {x z} _ {2} ^ {T} + \mathbf {q z} _ {1} ^ {T}) = \mathbf {y z} _ {2} ^ {T} + \mathbf {q z} _ {1} ^ {T} + \mathbf {z} _ {1} \mathbf {z} _ {2} ^ {T} + \mathbf {z} _ {2} \mathbf {z} _ {1} ^ {T},
$$

where

$$
\mathbf {z} _ {1} = \mathbf {A} \mathbf {x} - \operatorname {s i g n} (\mathbf {A} \mathbf {x}) \max  (| \mathbf {A} \mathbf {x} | - \tau \mathbf {1}, \mathbf {0}) \tag {84}
$$

$$
\mathbf {z} _ {2} = \mathbf {A} \mathbf {g} - \operatorname {s i g n} (\mathbf {A} \mathbf {g}) \max  (| \mathbf {A} \mathbf {g} | - \tau \mathbf {1}, \mathbf {0})
$$

A closer look at (83) reveals us

$$
\mathbf {A} \left(\delta_ {o} ^ {z _ {1}, z _ {2}}\right) ^ {T} = \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} \left(\mathbf {z} _ {1} ; \mathbf {z} _ {2}\right)}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \left(\delta_ {t} ^ {z _ {1}, z _ {2}}\right) ^ {T}. \tag {85}
$$

where

$$
\boldsymbol {\delta} _ {t} ^ {z _ {1}, z _ {2}} = \mathbf {z} _ {1} \mathbf {q} ^ {T} + \mathbf {z} _ {2} \mathbf {y} ^ {T} \tag {86}
$$

By similar construction applied to the rest of the pairs of data samples we have the result:

$$
\mathbf {A} \boldsymbol {\delta} _ {o} ^ {T} = \left(\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} + \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {Z})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}}\right) + \boldsymbol {\delta} _ {t} ^ {T} \square \tag {87}
$$

# APPENDIX D.

The result in (18) decomposes on the contributing components for similarity and the contributing components for dissimilarity, i.e.,  $\frac{\partial\mathcal{J}_{\ell_1}(\mathbf{AX})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_s - \frac{\partial\mathcal{J}_{\ell_1}(\mathbf{AX})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_d = \frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_s - \frac{\partial\mathcal{J}_{\ell_1}(\mathbf{Y})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_d + \pmb {\xi}_{c,d} - \pmb {\xi}_{c,d} + \pmb {\xi}_s - \pmb {\xi}_d$ . Moreover, w.r.t. the similarity concentrations we have the following splitting  $Tr\{\frac{\partial D_{\ell_1,c}^{B^M}(\mathbf{AX})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_s\} = D_{\ell_1,c}^{B^M}(\mathbf{Y}) + Tr\{\pmb {\xi}_{c,s}\} = D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X}) + Tr\{\pmb {\xi}_{c,s}\}$  and  $Tr\{\frac{\partial D_{\ell_1}^{B^M}(\mathbf{AX})}{\partial\mathbf{A}_t}|\mathbf{A}_t = \mathbf{I}|_s\} = D_{\ell_1}^{B^M}(\mathbf{Y}) + Tr\{\pmb {\xi}_s\} = D_{\ell_1,c}^{\mathcal{P}}(\mathbf{X}) + Tr\{\pmb {\xi}_s\}$ . Therefore, we have the following bounds

$$
a: T r \left\{\mathbf {A} \frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} \mid_ {\mathbf {A} _ {o} = \mathbf {I}} \mathbf {A} ^ {T} \right\} \leq D _ {\ell_ {1}, c} ^ {\mathcal {P}} (\mathbf {X}) \leq T r \left\{\frac {\partial D _ {\ell_ {1} , c} ^ {\mathcal {B} ^ {M}} (\mathbf {A X})}{\partial \mathbf {A} _ {t}} \mid_ {\mathbf {A} _ {t} = \mathbf {I}} \mid_ {s} \right\} = D _ {\ell_ {1}, c} ^ {\mathcal {B} ^ {M}} (\mathbf {A X}) \tag {88}
$$

$$
b: T r \{\mathbf {A} \frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {N}} (\mathbf {X})}{\partial \mathbf {A} _ {o}} | _ {\mathbf {A} _ {o} = \mathbf {I}} \mathbf {A} ^ {T} \} \leq D _ {\ell_ {1}} ^ {\mathcal {P}} (\mathbf {X}) \leq T r \{\frac {\partial D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {A X})}{\partial \mathbf {A} _ {t}} | _ {\mathbf {A} _ {t} = \mathbf {I}} | _ {s} \} = D _ {\ell_ {1}} ^ {\mathcal {B} ^ {M}} (\mathbf {A X})
$$

Note that  $c:\lambda_{min}(\mathbf{A}^T\mathbf{A})Tr\{\frac{\partial D_{\ell_1,c}^{B^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}\} \leq Tr\{\mathbf{A}\frac{\partial D_{\ell_1,c}^{B^N}(\mathbf{X})}{\partial\mathbf{A}_o}|_{\mathbf{A}_o = \mathbf{I}}\mathbf{A}^T\}$  where  $\lambda_{min}(\mathbf{A}^T\mathbf{A})$  is the minimum singular value to the matrix  $\mathbf{A}^T\mathbf{A}$ . Taking the logarithm of the ratio  $\frac{D_{\ell_1,c}^P(\mathbf{X})}{D_{\ell_1}^P(\mathbf{X}) + \epsilon}$  and using the bounds  $a,b$  and  $c$  we arrive at the desired result

# REFERENCES

Shun-ichi Amari. Information geometry and its applications: Survey. In GSI, volume 8085 of Lecture Notes in Computer Science, pp. 3. Springer, 2013.  
Yoshua Bengio, Aaron C. Courville, and Pascal Vincent. Unsupervised feature learning and deep learning: A review and new perspectives. CoRR, abs/1206.5538, 2012. URL http://arxiv.org/abs/1206.5538.  
Sijia Cai, Wangmeng Zuo, Lei Zhang, Xiangchu Feng, and Ping Wang. Support vector guided dictionary learning. In Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV, pp. 624-639, 2014.

Mehrdad J. Gangeh, Ahmed K. Farahat, Ali Ghodsi, and Mohamed S. Kamel. Supervised dictionary learning and sparse representation-a review. CoRR, abs/1502.05928, 2015. URL http://arxiv.org/abs/1502.05928.  
A. S. Georgiades, P. N. Belhumeur, and D. J. Kriegman. From few to many: Illumination cone models for face recognition under variable lighting and pose. IEEE Transactions on Pattern Analysis and Machine Intelligence, 23:643-660, 2001.  
Huimin Guo, Zhuolin Jiang, and Larry S. Davis. Discriminative dictionary learning with pairwise constraints. In Computer Vision - ACCV 2012 - 11th Asian Conference on Computer Vision, Daejeon, Korea, November 5-9, 2012, Revised Selected Papers, Part I, pp. 328-342, 2012.  
Huimin Guo, Zhuolin Jiang, and Larry S. Davis. Discriminative Dictionary Learning with Pairwise Constraints, pp. 328-342. Springer Berlin Heidelberg, Berlin, Heidelberg, 2013. ISBN 978-3-642-37331-2. doi: 10.1007/978-3-642-37331-2_25. URL https://doi.org/10.1007/978-3-642-37331-2_25.  
Rui Jiang, Hong Qiao, and Bo Zhang. Efficient fisher discrimination dictionary learning. Signal Process., 128(C):28-39, November 2016. ISSN 0165-1684. doi: 10.1016/j.sigpro.2016.03.013. URL http://dx.doi.org/10.1016/j.sigpro.2016.03.013.  
Zhuolin Jiang, Zhe Lin, and L. S. Davis. Learning a discriminative dictionary for sparse coding via label consistent k-svd. In Proceedings of the 2011 IEEE Conference on Computer Vision and Pattern Recognition, CVPR '11, pp. 1697-1704, Washington, DC, USA, 2011. IEEE Computer Society. ISBN 978-1-4577-0394-2. doi: 10.1109/CVPR.2011.5995354. URL http://dx.doi.org/10.1109/CVPR.2011.5995354.  
Zhuolin Jiang, Zhe Lin, and Larry S. Davis. Label consistent K-SVD: learning a discriminative dictionary for recognition. IEEE Trans. Pattern Anal. Mach. Intell., 35(11):2651-2664, 2013. doi: 10.1109/TPAMI.2013.88. URL http://dx.doi.org/10.1109/TPAMI.2013.88.  
Kenneth Kreutz-Delgado, Joseph F. Murray, Bhaskar D. Rao, Kjersti Engan, Te-Won Lee, and Terrence J. Sejnowski. Dictionary learning algorithms for sparse representation. *Neural Comput.*, 15(2):349-396, February 2003. ISSN 0899-7667. doi: 10.1162/089976603762552951. URL http://dx.doi.org/10.1162/089976603762552951.  
Yann Lecun and Corinna Cortes. The MNIST database of handwritten digits. URL http://yann.lecun.com/exdb/mnist/.  
Yann LeCun, Fu Jie Huang, and Léon Bottou. Learning methods for generic object recognition with invariance to pose and lighting. In Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, CVPR'04, pp. 97-104, Washington, DC, USA, 2004. IEEE Computer Society.  
Yann LeCun, David G. Lowe, Jitendra Malik, Jim Mutch, Pietro Perona, and Tomaso Poggio. Object Recognition, Computer Vision, and the Caltech 101: A Response to Pinto et al. Technical report, March 2008.  
Yang Liu, Wei Chen, Qingchao Chen, and Ian J. Wassell. Support discrimination dictionary learning for image classification. In Computer Vision - ECCV 2016 - 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II, pp. 375-390, 2016.  
Julien Mairal, Francis R. Bach, Jean Ponce, Guillermo Sapiro, and Andrew Zisserman. Supervised dictionary learning. In Advances in Neural Information Processing Systems 21, Proceedings of the Twenty-Second Annual Conference on Neural Information Processing Systems, Vancouver, British Columbia, Canada, December 8-11, 2008, pp. 1033-1040, 2008.  
Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online dictionary learning for sparse coding. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML '09, pp. 689-696, New York, NY, USA, 2009. ACM. ISBN 978-1-60558-516-1. doi: 10.1145/1553374.1553463. URL http://doi.acm.org/10.1145/1553374.1553463.

A. Martínez and R. Benavente. The ar face database. Technical Report 24, Computer Vision Center, Jun 1998. URL "http://www.cat.uab.cat/Public/Publications/1998/MaB1998".  
L. Mirsky. On the trace of matrix products. 20(3-6):171-174, 1959. ISSN 2167-3888.  
Sameer A. Nene, Shree K. Nayar, and Hiroshi Murase. Columbia object image library (coil-20. Technical report, 1996.  
J. Von Neumann. Some matrix-inequalities and metrization of matrix-space. Tomskii Univ. Rev., 1: 286-300, 1937.  
D. Nister and H. Stewenius. Scalable recognition with a vocabulary tree. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), volume 2, pp. 2161-2168, June 2006. oral presentation.  
Neal Parikh and Stephen Boyd. Proximal algorithms. Found. Trends Optim., 1(3):127-239, January 2014. ISSN 2167-3888.  
I. Ramirez, P. Sprechmann, and G. Sapiro. Classification and clustering via dictionary learning with structured incoherence and shared features. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pp. 3501-3508, June 2010. doi: 10.1109/CVPR.2010.5539964.  
Saiprasad Ravishankar and Yoram Bresler. Learning sparsifying transforms for image processing. In 19th IEEE International Conference on Image Processing, ICIP 2012, Lake Buena Vista, Orlando, FL, USA, September 30 - October 3, 2012, pp. 681-684. IEEE, 2012. ISBN 978-1-4673-2534-9. doi: 10.1109/ICIP.2012.6466951. URL http://dx.doi.org/10.1109/ICIP.2012.6466951.  
Saiprasad Ravishankar and Yoram Bresler. Doubly sparse transform learning with convergence guarantees. In IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2014, Florence, Italy, May 4-9, 2014, pp. 5262-5266. IEEE, 2014. doi: 10.1109/ICASSP.2014.6854607. URL http://dx.doi.org/10.1109/ICASSP.2014.6854607.  
Ron Rubinstein and Michael Elad. Dictionary learning for analysis-synthesis thresholding. IEEE Trans. Signal Processing, 62(22):5962-5972, 2014.  
Ron Rubinstein, Alfred M. Bruckstein, and Michael Elad. Dictionaries for sparse representation modeling. Proceedings of the IEEE, 98(6):1045-1057, 2010.  
Ron Rubinstein, Tomer Peleg, and Michael Elad. Analysis K-SVD: A dictionary-learning algorithm for the analysis sparse model. IEEE Trans. Signal Processing, 61(3):661-677, 2013.  
Sumit Shekhar, Vishal M. Patel, and Rama Chellappa. Analysis sparse coding models for image-based classification. In 2014 IEEE International Conference on Image Processing, ICIP 2014, Paris, France, October 27-30, 2014, pp. 5207-5211. IEEE, 2014. ISBN 978-1-4799-5751-4. doi: 10.1109/ICIP.2014.7026054. URL http://dx.doi.org/10.1109/ICIP.2014.7026054.  
Ali Taalimi, Shahab Ensafi, Hairong Qi, Shijian Lu, Ashraf A. Kassim, and Chew Lim Tan. Multimodal dictionary learning and joint sparse representation for hep-2 cell classification. pp. 308-315, 2015.  
T. H. Vu and V. Monga. Learning a low-rank shared dictionary for object classification. In 2016 IEEE International Conference on Image Processing (ICIP), pp. 4428-4432, Sept 2016a. doi: 10.1109/ICIP.2016.7533197.  
T. H. Vu, H. S. Mousavi, V. Monga, U. K. A. Rao, and G. Rao. Dfdl: Discriminative feature-oriented dictionary learning for histopathological image classification. In 2015 IEEE 12th International Symposium on Biomedical Imaging (ISBI), pp. 990-994, April 2015. doi: 10.1109/ISBI.2015.7164037.

Tiep Huu Vu and Vishal Monga. Fast low-rank shared dictionary learning for image classification. CoRR, abs/1610.08606, 2016b. URL http://arxiv.org/abs/1610.08606.  
Tiep Huu Vu and Vishal Monga. Fast low-rank shared dictionary learning for image classification. CoRR, abs/1610.08606, 2016c. URL http://arxiv.org/abs/1610.08606.  
Yong Xu, Yuping Sun, Yuhui Quan, and Bo Zheng. Discriminative structured dictionary learning with hierarchical group sparsity. Comput. Vis. Image Underst., 136(C):59-68, July 2015. ISSN 1077-3142. doi: 10.1016/j.cviu.2015.01.006. URL http://dx.doi.org/10.1016/j.cviu.2015.01.006.  
M. Yang, L. Zhang, X. Feng, and D. Zhang. Fisher discrimination dictionary learning for sparse representation. In 2011 International Conference on Computer Vision, pp. 543-550, Nov 2011a. doi: 10.1109/ICCV.2011.6126286.  
Meng Yang, Lei Zhang, Xiangchu Feng, and David Zhang. Fisher discrimination dictionary learning for sparse representation. In Proceedings of the 2011 International Conference on Computer Vision, ICCV '11, pp. 543-550, Washington, DC, USA, 2011b. IEEE Computer Society. ISBN 978-1-4577-1101-5. doi: 10.1109/ICCV.2011.6126286. URL http://dx.doi.org/10.1109/ICCV.2011.6126286.