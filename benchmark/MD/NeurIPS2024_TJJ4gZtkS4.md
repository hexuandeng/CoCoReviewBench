# Tropical Expressivity of Neural Networks

Anonymous Author(s)

Affiliation

Address

email

# Abstract

We propose an algebraic geometric framework to study the expressivity of linear activation neural networks. A particular quantity that has been actively studied in the field of deep learning is the number of linear regions, which gives an estimate of the information capacity of the architecture. To study and evaluate information capacity and expressivity, we work in the setting of tropical geometry—a combinatorial and polyhedral variant of algebraic geometry—where there are known connections between tropical rational maps and feedforward neural networks. Our work builds on and expands this connection to capitalize on the rich theory of tropical geometry to characterize and study various architectural aspects of neural networks. Our contributions are threefold: we provide a novel tropical geometric approach to selecting sampling domains among linear regions; an algebraic result allowing for a guided restriction of the sampling domain for network architectures with symmetries; and an open source library to analyze neural networks as tropical Puiseux rational maps. We provide a comprehensive set of proof-of-concept numerical experiments demonstrating the breadth of neural network architectures to which tropical geometric theory can be applied to reveal insights on expressivity characteristics of a network. Our work provides the foundations for the adaptation of both theory and existing software from computational tropical geometry and symbolic computation to deep learning.

# 1 Introduction

Deep learning has become the undisputed state-of-the-art for data analysis and has wide-reaching prominence in many fields of computer science, despite still being based on a limited theoretical foundation. Developing theoretical foundations to better understand the unparalleled success of deep neural networks is one of the most active areas of research in modern statistical learning theory. Expressivity is one of the most important approaches to quantifiably measuring the performance of a deep neural network—such as how they are able to represent highly complex information implicitly in their weights and to generalize from data—and therefore key to understanding the success of deep learning.

Tropical geometry is a reinterpretation of algebraic geometry that features piecewise linear and polyhedral constructions, where combinatorics naturally comes into play [e.g., 1, 2, 3]. These characteristics of tropical geometry make it a natural framework for studying the linear regions in a neural network—an important quantity in deep learning representing the network information capacity [4, 5, 6, 7, 8, 9, 10]. The intersection of deep learning theory and tropical geometry is a relatively new area of research with great potential towards the ultimate goal of understanding how and why deep neural networks perform so well. In this paper, we propose a new perspective for measuring and estimating the expressivity and information capacity of a neural networks by developing and expanding known connections between neural networks and tropical rational functions in both theory and practice.

Related Work. Tropical geometry has been used to characterize deep neural networks with piecewise linear activation functions, including two of the most popular and widely-used activation functions, namely, rectified linear units (ReLUs) and maxout units. The first explicit connection between tropical geometry and neural networks establishes that the decision boundary of a deep neural network with ReLU activation functions is a tropical rational function [11]. Concurrently, it was established that the maxout activation function fits input data by a tropical polynomial [12]. These works considered neural networks whose input domain is Euclidean, which was recently developed to incorporate tropically-motivated input domains, in particular, the tropical projective torus [13]. Most recently, tropical geometry has been used to construct convolutional neural networks that are robust to adversarial attacks via tropical decision boundaries [14].

Contributions. In this paper, we establish novel algebraic and geometric tools to quantify the expressivity of a neural network. Networks with a piecewise linear activation compute piecewise linear functions where the input space is divided into areas; the network computing a single linear function on each area. These areas are referred to as the linear regions of the network; the number of distinct linear regions is a quantifiable measure of expressivity of the network [e.g., 5]. In our work, we not only study the number of linear regions, we aim to understand their geometry. The main contributions of our work are the following.

- We provide a geometric characterization of the linear regions in a neural network via the input space: estimating the linear regions is typically carried out by random sampling from the input space, where randomness may cause some linear regions of a neural network to be missed and result in an inaccurate information capacity measure. We propose an effective sampling domain as a ball of radius  $R$ , which is a subset of the entire sampling space that hits all of the linear regions of a given neural network. We compute bounds for the radius  $R$  based on a combinatorial invariant known as the Hoffman constant, which effectively gives a geometric characterization and guarantee for the linear regions of a neural network.  
- We exploit geometric insight into the linear regions of a neural network to gain dramatic computational efficiency: when networks exhibit invariance under symmetry, we can restrict the sampling domain to a fundamental domain of the group action and thus reduce the number of samples required. We experimentally demonstrate that sampling from the fundamental domain provides an accurate estimate of the number of linear regions with a fraction of the compute requirements.  
- We provide an open source library integrated into the Open Source Computer Algebra Research (OSCAR) system [15] which converts both trained and untrained arbitrary neural networks into algebraic symbolic objects. This contribution then opens the door for the extensive theory and existing software on symbolic computation and computational tropical geometry to be used to study neural networks.

The remainder of this paper is organized as follows. We provide an overview of the technical background on tropical geometry and its connection to neural networks in Section 2. We then devote a section to each of the contributions listed above—Sections 3, 4, and 5, respectively—in which we present our theoretical contributions and numerical experiments. We close the paper with a discussion on limitations of our work and directions for future research in Section 6.

# 2 Technical Background

In this section, we give basic definitions from tropical geometry required to write tropical expressions for neural networks.

# 2.1 Tropical Polynomials

Algebraic geometry studies geometric properties of solution sets of polynomial systems that can be expressed algebraically, such as their degree, dimension, and irreducible components. Tropical geometry is a variant of algebraic geometry where the polynomials are defined in the tropical semiring,  $\mathbb{R} = (\mathbb{R}\cup \{\infty \} ,\oplus ,\odot)$  where the addition and multiplication operators are given by  $a\oplus b = \max (a,b)$  and  $a\odot b = a + b$  , respectively. Define  $a\oslash b\coloneqq a - b$

Using these operations, we can write polynomials as  $\bigoplus_{m} a_{m} T^{m}$ , where  $a_{i}$  are coefficients,  $T \in \bar{\mathbb{R}}$ , and where the sum is indexed by a finite subset of  $\mathbb{N}^n$ . In our work, we consider the following generalizations of tropical polynomials.  
Definition 2.1. A tropical Puiseux polynomial in the indeterminates  $T_{1},\ldots ,T_{n}$  is a formal expression of the form  $\bigoplus_{m}a_{m}T^{m}$  where the index  $n$  runs through a finite subset of  $\mathbb{Q}_{\geq 0}^{m}$  and  $T^m = T_1^{m_1}\odot \dots \odot T_n^{m_n}$ , and taking powers in the tropical sense.  
Definition 2.2. A tropical Puiseux rational map in  $T_1, \ldots, T_n$  is a tropical quotient of the form  $p \oslash q$  where  $p, q$  are tropical Puiseux polynomials.  
Tropical (Puiseux) polynomials and rational maps induce functions from  $\mathbb{R}^n\to \mathbb{R}$ , which take a point  $x\in \mathbb{R}^n$  to the number obtained by substituting  $T = x$  in the algebraic expression and performing the (tropical) operations. It is important to note that tropically, the formal algebraic expression contains strictly more information than the corresponding function, since different tropical expressions can induce the same function.

# 2.2 Tropical Expressions for Neural Networks

We now overview and recast the framework of [11], which establishes the first explicit connection between tropical geometry and neural networks, in a slightly different language for our results.

As in [11], the neural networks we will focus on are fully connected multilayer perceptrons with ReLU activation, i.e., functions  $\mathbb{R}^n\to \mathbb{R}^m$  of the form  $\sigma \circ L_d\circ \sigma \circ L_{i - 1}\circ \dots \circ L_1$  where  $L_{i}:\mathbb{R}^{n_{i - 1}}\rightarrow \mathbb{R}^{n_i}$  is an affine map and  $\sigma (t) = \max \{t,0\}$ . For the remainder of this paper, we use the term "neural network" to refer solely to these. We will always assume that the weights and biases of our neural networks are rational numbers. From a computational perspective, this is not a serious restriction since this is sufficient to describe any neural network with weights and biases given by floating point numbers. We refer to the tuple  $[n,n_1,\ldots ,n_{d - 1},m]$  as the architecture of the neural network.

One of the key observations intersecting tropical geometry and deep learning is that, up to rescaling of rational weights to obtain integers, neural networks can be written as tropical rational functions [11, Theorem 5.2]. From a more computational perspective, it is usually preferable to avoid such rescaling and simply work with the original weights. The proof of Theorem 5.2 in [11] can directly be adapted to show that any neural network can be written as the function associated to a tropical Puiseux rational map. In their language, this corresponds to saying that any neural network is a tropical rational signomial with nonnegative rational exponents.

# 3 Sampling Domain Selection Using a Hoffman Constant

Estimating the number of linear regions of a neural network typically proceeds by sampling points from the input domain and counting the memberships of these points. To guarantee that membership is exhaustive, we seek a sampling domain as a sufficiently large ball so that all linear regions are intersected. At the same time, we would like for the ball to be as small as possible to guarantee efficient sampling. We are thus searching for the smallest ball from which we can sample in such a way that all linear regions are intersected. Given the polyhedral geometry of tropical Puiseux rational maps, it turns out that the radius of this smallest ball that we seek is closely related to the Hoffman constant, which is a combinatorial invariant.

Our contribution in this section is a definition of a Hoffman constant of a neural network; we demonstrate its relationship to the smallest sampling ball and propose algorithms to compute its true value and lower and upper bounds.

# 3.1 Defining a Neural Network Hoffman Constant

In simpler terms, the Hoffman constant can be expressed for a matrix as follows. Let  $A$  be an  $m \times n$  matrix. For any  $b \in \mathbb{R}^m$ , let  $P(A, b) = \{x \in \mathbb{R}^n : Ax \leq b\}$  denote the polyhedron determined by  $A$  and  $b$ . For a nonempty polyhedron  $P(A, b)$ , let  $d(u, P(A, b)) = \min \{\| u - x \| : x \in P(A, b)\}$  denote the distance from a point  $u \in \mathbb{R}^n$  to the polyhedron, measured under an arbitrary norm  $\| \cdot \|$ .

on  $\mathbb{R}^n$ . Then there exists a constant  $H(A)$  only depending on  $A$  such that

$$
d (u, P _ {A, b}) \leq H (A) \| (A u - b) _ {+} \| \tag {1}
$$

where  $x_{+} = \max \{x,0\}$  is applied coordinate-wise [16]. The constant  $H(A)$  is called the Hoffman constant of  $A$ .

The Hoffman Constant for Tropical Polynomials and Rational Functions. Let  $f: \mathbb{R}^n \to \mathbb{R}$  be a tropical Puiseux polynomial and let  $\mathcal{U} = \{U_1, \ldots, U_m\}$  be the set of linear regions of  $f$ . Let  $f(x) = a_{i1}x_1 + \ldots + a_{in}x_n + b_i$  occur on the region  $U_i$ . Further, let  $A = [a_{ij}]_{m \times n}$  be the matrix of coefficients in the expression of  $f$  over  $\mathcal{U}$ . The linear region  $U_i$  is defined by the following inequalities

$$
a _ {i 1} x _ {1} + \dots + a _ {i n} x _ {n} + b _ {i} \geq a _ {j 1} x _ {1} + \dots + a _ {j n} x _ {n} + b _ {j}, \quad \forall j = 1, 2, \dots , m. \tag {2}
$$

In matrix form, (2) is equivalent to

$$
(A - \mathbf {1} a _ {i}) x \leq b _ {i} \mathbf {1} - b \tag {3}
$$

where  $\mathbf{1}$  is a column vector of all 1's;  $a_{i}$  is the ith row vector of  $A$ ; and  $b$  is a column vector of all  $b_{i}$ . Denote  $\widetilde{A}_{U_i} := A - \mathbf{1}a_i$  and  $\widetilde{b}_{U_i} := b_i\mathbf{1} - b$ . Then the linear region  $U_{i}$  is captured by the linear system of inequalities  $\widetilde{A}_{U_i}x \leq \widetilde{b}_{U_i}$ .

Definition 3.1. Let  $f: \mathbb{R}^n \to \mathbb{R}$  be a tropical Puiseux polynomial. The Hoffman constant of  $f$  is defined as

$$
H(f) = \max_{U_{i}\in \mathcal{U}}H(\widetilde{A}_{U_{i}}).
$$

Care needs to be taken in defining a Hoffman constant for a tropical Puiseux rational map: We want to avoid having all linear regions defined by systems of linear inequalities, since there exist linear regions which are not convex. To do so, we consider convex refinements of linear regions induced by intersections of linear regions of tropical polynomials.

Definition 3.2. Let  $p \oslash q$  be a difference of two tropical Puiseux polynomials. Let  $A_{p}$  (respectively  $A_{q}$ ) be the  $m_{p} \times n$  (respectively  $m_{q} \times n$ ) matrix of coefficients for  $p$  (respectively  $q$ ). The Hoffman constant of  $p \oslash q$  is

$$
H (p \oslash q) := \max  \left\{H \left(\left[ \begin{array}{l} A _ {p} \\ A _ {q} \end{array} \right] - \mathbf {1} \left[ \begin{array}{l} a _ {i _ {p}} \\ a _ {i _ {q}} \end{array} \right]\right): i _ {p} = 1, \dots , m _ {p}; i _ {q} = 1, \dots , m _ {q} \right\}. \tag {4}
$$

156 Let  $f$  be a tropical Puiseux rational map. Then the Hoffman constant of  $f$  is defined as the minimal 157 Hoffman constant of  $H(p\otimes q)$  over all possible expressions of  $f = p\otimes q$ .

Given the correspondence between neural networks and tropical Puiseux rational maps, the Hoffman constant is well-defined for any neural network and may be computed from the geometry and combinatorics of its linear regions.

# 3.2 The Minimal Effective Radius

For a neural network whose tropical Puiseux rational map is  $f:\mathbb{R}^n\to \mathbb{R}$  , let  $\mathcal{U} = \{U_1,\dots ,U_m\}$  be the collection of all linear regions. For any  $x\in \mathbb{R}^n$  , define the minimal effective radius of  $f$  at  $x$  as

$$
R _ {f} (x) := \min  \{r: B (x, r) \cap U _ {i} \neq \emptyset , U _ {i} \in \mathcal {U} \}
$$

where  $B(x,r)$  is the ball of radius  $r$  centered at  $x$ . That is,  $R_{f}(x)$  is the minimal radius such that the ball  $B(x,r)$  intersects all linear regions. It is the smallest required radius of sampling around  $x$  in order to express the full classifying capacity of the neural network  $f$ .

We start with the following lemma which relates the minimal effective radius to the Hoffman constant when  $f$  is a tropical Puiseux polynomial.

169 Lemma 3.3. Let  $f$  be a tropical Puiseux polynomial and  $x \in \mathbb{R}^n$  be any point, then

$$
R _ {f} (x) \leq H (f) \max  _ {U _ {i} \in \mathcal {U}} \| \left(\widetilde {A} _ {U _ {i}} x - \widetilde {b} _ {U _ {i}}\right) _ {+} \|. \tag {5}
$$

In particular, we are interested in studying when  $\mathbb{R}^m$  and  $\mathbb{R}^n$  are equipped with the  $\infty$ -norm. In this case, the minimal effective radius can be related to the Hoffman constant and function value of  $f = p \oslash q$ . For a tropical Puiseux polynomial  $p(x) = \max_{1 \leq i \leq m_p} \{a_i x + b_i\}$ , let  $\check{p}(x) = \min_{1 \leq j \leq m_q} \{a_j x + b_j\}$  be its min-conjugate.

Proposition 3.4. Let  $f = p \oslash q$  be a tropical Puiseux rational map. For any  $x \in \mathbb{R}^n$ , we have

$$
R _ {f} (x) \leq H (p \oslash q) \max  \{p (x) - \check {p} (x), q (x) - \check {q} (x) \}. \tag {6}
$$

# 3.3 Computing and Estimating Hoffman Constants

The PVZ Algorithm. In [17], the authors proposed a combinatorial algorithm to compute the precise value of the Hoffman constant for a matrix  $A \in \mathbb{R}^{m \times n}$ , which we refer to as the Peña-Vera-Zuluaga (PVZ) algorithm and sketch its main steps here.

Definition 3.5. A set-valued map  $\Phi : \mathbb{R}^n \to \mathbb{R}^m$  assigns a set  $\Phi(x) \subseteq \mathbb{R}^m$ . The map is surjective if  $\Phi(\mathbb{R}^n) = \cup_x \Phi(x) = \mathbb{R}^m$ . Let  $A \in \mathbb{R}^{m \times n}$ . For any  $J \subseteq \{1, 2, \ldots, m\}$ , let  $A_J$  be the submatrix of  $A$  consisting of rows with indices in  $J$ . The set  $J$  is called  $A$ -surjective if the set-valued map  $\Phi(x) = A_J x + \{y \in \mathbb{R}^J : y \geq 0\}$  is surjective.

Notice that  $A$ -surjectivity is a generalization of linear independence of row vectors. We illustrate this observation in the following two examples.

Example 3.6. If  $J$  is such that  $A_J$  is full-rank, then  $J$  is  $A$ -surjective, since for any  $y \in \mathbb{R}^J$ , there exists  $x \in \mathbb{R}^n$  such that  $y = A_Jx$ .

Example 3.7. Let  $A = \mathbf{1}_{m \times n}$  be the  $m \times n$  matrix whose entries are 1's. For any subset  $J$  of  $\{1, \ldots, m\}$  and for any  $y \in \mathbb{R}^J$ , let  $x \in \mathbb{R}^n$  such that  $\sum_{i} x_i \leq \min \{y_j, j \in J\}$ . Then  $y - A_J x \geq 0$ . Thus any  $J$  is  $A$ -surjective.

The PVZ algorithm is based on the following characterization of Hoffman constant.

Proposition 3.8. [17, Proposition 2] Let  $A \in \mathbb{R}^{m \times n}$ . Equip  $\mathbb{R}^m$  and  $\mathbb{R}^n$  with norm  $\| \cdot \|$  and denote its dual norm by  $\| \cdot \|^*$ . Let  $S(A)$  be the set of all  $A$ -surjective sets. Then

$$
H (A) = \max  _ {J \in \mathcal {S} (A)} H _ {J} (A) \tag {7}
$$

where

$$
H _ {J} (A) = \max  _ {y \in \mathbb {R} ^ {m} \| y \| \leq 1} \min  _ {\substack {x \in \mathbb {R} ^ {n} \\ A _ {J} x \leq y _ {J}}} \| x \| = \frac {1}{\min  _ {v \in \mathbb {R} _ {+} ^ {J} , \| v \| ^ {*} = 1} \| A _ {J} ^ {\top} v \| ^ {*}}. \tag{8}
$$

This characterization is particularly useful when  $\mathbb{R}^m$  and  $\mathbb{R}^n$  are equipped with the  $\infty$ -norm, since the computation of (8) reduces to a linear programming (LP) problem. The key problem is how to maximize over all  $A$ -surjective sets. To do this, the PVZ algorithm maintains three collections of sets  $\mathcal{F}, \mathcal{I}$ , and  $\mathcal{J}$  where during every iteration: (i)  $\mathcal{F}$  contains  $J$  such that  $J$  is  $A$ -surjective; (ii)  $\mathcal{I}$  contains  $J$  such that  $J$  is not  $A$ -surjective; and (iii)  $\mathcal{J}$  contains candidates  $J$  whose  $A$ -surjectivity will be tested.

To detect whether a candidate  $J \in \mathcal{J}$  is surjective, the PVZ algorithm requires solving

$$
\min  \left\| A _ {J} ^ {T} v \right\| _ {1}, \text {s . t .} v \in \mathbb {R} _ {+} ^ {J}, \| v \| _ {1} = 1. \tag {9}
$$

If the optimal value is positive, then  $J$  is  $A$ -surjective, and  $J$  is assigned to  $\mathcal{F}$  and all subsets of  $J$  are removed from  $\mathcal{J}$ . Otherwise, the optimal value is 0 and there is  $v \in \mathbb{R}_+^J$  such that  $A_J^\top v = 0$ . Let  $I(v) = \{i \in J : v_i > 0\}$  and assign  $I(v)$  to  $\mathcal{I}$ . Let  $\hat{J} \in \mathcal{J}$  be any set containing  $I(v)$ . Replace all such  $\hat{J}$  by sets  $\hat{J} \setminus \{i\}, i \in I(v)$  which are not contained in any sets in  $\mathcal{F}$ . The implementation used in our paper directly uses the MATLAB code provided by [17].

Lower and Upper Bounds. A limitation of the PVZ algorithm is that during each loop, every set in  $\mathcal{I}$  needs to be tested, and each test requires solving a LP problem. Although solving one LP problem in practice is fast, a complete while loop calls the LP solver many times.

Here, we propose an algorithm to estimate lower and upper bounds for Hoffman constants. An intuitive way to estimate the lower bound is to sample a number of random subsets from  $\{1,\dots ,m\}$  and test for  $A$  -surjectivity. This method bypasses optimizing combinatorially over  $S(A)$  of  $A$  surjective sets and gives a lower bound of Hoffman constant by Proposition 3.8.

To get an upper of Hoffman constant, we use the result from [18].

Theorem 3.9. [18, Theorem 4.2] Let  $A \in \mathbb{R}^{m \times n}$ . Let  $\mathcal{D}(A)$  be a set of subsets of  $J \subseteq \{1, \ldots, m\}$  such that  $A_J$  is full rank. Let  $\mathcal{D}^*(A)$  be the set of maximal elements in  $\mathcal{D}(A)$ . Then the Hoffman constant measured under 2-norm is bounded by

$$
H (A) \leq \max  _ {J \in \mathcal {D} ^ {*} (A)} \frac {1}{\hat {\rho} \left(A _ {J}\right)} \tag {10}
$$

where  $\hat{\rho}(A)$  is the smallest singular value of  $A$ .

Using the fact that  $\| \cdot \| _1\geq \| \cdot \| _2$  , and the characterization from (8), we see that the upper bound also holds when  $\mathbb{R}^m$  and  $\mathbb{R}^n$  are equipped with the  $\infty$  -norm. However, enumerating all maximal elements in  $\mathcal{D}(A)$  is not an improvement over enumerating  $A$  -surjective sets from a computational perspective. Instead, we will retain the strategy as in lower bound estimation to sample a number of sets from  $\{1,2,\dots,m\}$  and approximate the upper bound by (10). We verify this approach via synthetic data. The experiments are relegated to the Appendix.

# 4 Symmetry and the Fundamental Domain

In this section, we study a geometric characterization of the sampling domain for networks exhibiting symmetry. This corresponds to invariant neural networks.

# 4.1 Linear Regions of Invariant Neural Networks

The notion of invariance for a neural network describes when a manipulation of the input domain does not affect the output of the network. The manipulations we consider here are group actions.

Definition 4.1. Let  $\sigma : \mathbb{R}^n \to \mathbb{R}$  be a piecewise linear function, and let  $G$  be a group acting on the domain  $\mathbb{R}^n$ .  $\sigma$  is invariant under the group action of  $G$  if for any element  $g \in G$ ,  $\sigma \circ g = \sigma$ .

Given an invariant neural network, we can then define a sampling domain that takes into account the effect of the group action.

Definition 4.2. Let  $G$  be a group acting on  $\mathbb{R}^n$ . A subset  $\Delta \subseteq \mathbb{R}^n$  is a fundamental domain if it satisfies two following conditions: (i)  $\mathbb{R}^n = \bigcup_{g \in G} g \cdot \Delta$ ; and (ii)  $g \cdot \operatorname{int}(\Delta) \cap h \cdot \operatorname{int}(\Delta) = \emptyset$  for all  $g, h \in G, g \neq h$ .

The fundamental domain of a group  $G$  therefore provides a periodic tiling of  $\mathbb{R}^n$  by acting on  $\Delta$ . This is very useful in the context of numerical sampling for neural networks which are invariant under some symmetry, since it means we can sample from a smaller subset of the input domain with a guarantee to find all the linear regions in the limit. This allows us, in principle, to be able to use far fewer samples while maintaining the same density of points.

Theorem 4.3. Let  $f: \mathbb{R}^N \to \mathbb{R}$  be a tropical rational map invariant under group  $G$ . Let  $\Delta \subseteq \mathbb{R}^N$  be a fundamental domain of  $G$ . Suppose  $\mathcal{L}$  is the set of linear regions. Define the following two sets

$$
\mathcal {U} _ {c} := \{A \in \mathcal {U}: A \subseteq \Delta \}
$$

$$
\mathcal {U} _ {n} := \{A \in \mathcal {U}: A \cap \Delta \neq \emptyset \}.
$$

Then

$$
| G | | \mathcal {U} _ {c} | \leq | \mathcal {U} | \leq | G | | \mathcal {U} _ {c} | + \sum_ {A \in \mathcal {U} _ {n} \backslash \mathcal {U} _ {c}} \frac {| G |}{| G _ {A} |}.
$$

where  $|G_A|$  is the size of the stabilizer of  $A$ .

This gives us a method for estimating the total number of linear regions from sampling in the fundamental domain using multiplicity, which we discuss next.

# 4.2 Sampling from the Fundamental Domain

To demonstrate the potential performance improvements in numerical sampling exploiting symmetry in the network architecture, we consider permutation invariant neural networks inspired by deep sets [19]. Our numerical sampling approach is inspired by very recent work in this area [20].

Lemma 4.4 ([19]). An  $m \times m$  matrix  $W$  acting as a linear operator of the form  $W = \lambda I_{m \times m} + \gamma (\mathbf{1}^T\mathbf{1})$ , where  $\lambda, \gamma \in \mathbb{R}$  is permutation equivariant, meaning  $WPx = PWx$  for any  $x \in \mathbb{R}^m$ , so it commutes with any permutation matrix.

Using a weight matrix of this form, we can construct permutation invariant neural networks by setting the bias to 0, applying a ReLU activation after multiplication by  $W$ , and then summing. In this case, the network is invariant under the group action  $S_{n}$ , so the fundamental domain is the set of points with increasing coordinates, i.e.,  $\Delta = \{(x_{1},\ldots ,x_{n}):x_{1}\geq x_{2}\geq \ldots \geq x_{n}\}$ . This splits  $\mathbb{R}^n$  into  $n!$  tiles, so we have a clear and significant advantage in restricting sampling to the fundamental domain.

Note, however, that it is important to address the multiplicities of symmetric linear regions correctly: If a given Jacobian of shape  $n \times 1$  has no repeated elements, this means it is contained in the interior of some group action applied to the fundamental domain. This means there are  $n!$  total linear regions with this Jacobian. If, on the other hand, there are repeated coefficients in a given Jacobian  $J$ , we consider the set  $C(J)$  of counts of repeated elements. For example, for  $J = [1, 1, 0]$ ,  $C(J) = (2, 1)$ . Then the multiplicity of a given Jacobian is given by

$$
\operatorname {m u l t} (J) = \frac {n !}{\prod_ {c \in C (J)} c !}.
$$

Using this multiplicity calculation we can efficiently estimate the number of linear regions while reducing the number of point samples by a factor of  $n!$ . This provides a dramatic gain in sampling efficiency.

In Figure 1, we present the results when Algorithm 2 is run with  $R = 10$ ,  $N = 10$ ,  $M = 50$ . These results show that the fundamental domain estimate performs well for low dimensional inputs but appears to overcount linear regions as  $n$  increases. Despite divergence, there is still utility in this metric because we are often more concerned with obtaining an upper bound on the expressivity of a neural network than an exact figure and the fundamental domain estimate does not undercount the number of linear regions.

# 5 Symbolic Neural Networks

Here, we present the details on our practical contribution of a symbolic representation of neural networks as a new library integrated into OSCAR [15].

# 5.1 Computing Linear Regions of Tropical Puiseux Rational Maps

We present an algorithm that can compute the linear regions of any tropical Puiseux rational function. Intuitively, we do this by computing the linear regions of the numerator and denominator, and then considering intersections of such regions and how they fit together. Thus, a first step is to understand how the computation of linear regions works for tropical Puiseux polynomials. The key to our approach will be to exploit the polyhedral connection of tropical geometry and recast the problem in the language of polyhedral geometry. This, among other things, will allow us to make use of the extensive polyhedral geometry library in OSCAR [15] for implementation.

One important upshot from this study is that there is a strong connection between the number of linear regions of a tropical Puiseux rational function and the number of monomials that appear in its algebraic expression. Note, however, that the two are independent, in the sense that two Puiseux rational functions could have the same number of linear regions but different numbers of (nonzero) monomials, and conversely, the same number of monomials and a different number of linear regions. For instance, computing the number of linear regions requires some combinatorial data about the intersections of the polyhedra defined by monomials.

First, we need to know how to compute the linear regions of tropical polynomials. Let  $P = \bigoplus_{n} a_{n} \odot x^{n}$  where by  $x^{n}$  we mean  $x_{1}^{n_{1}} \odot \dots \odot x_{k}^{n_{k}}$  and powers are taken in the tropical sense. Then

as function  $\mathbb{R}^k\to \mathbb{R}$ ,  $P$  is given by  $\max_n\left\{a_n + n_1x_1\dots +n_kx_k\right\}$ . It follows that the linear regions of  $P$  are precisely the sets of the form

$$
S _ {n} = \left\{x \in \mathbb {R} ^ {n} \mid a _ {m} + m _ {1} x _ {1} \dots + m _ {k} x _ {k} \leq a _ {n} + n _ {1} x _ {1} \dots + n _ {k} x _ {k} \text {f o r a l l} m \neq n \right\}.
$$

For any set  $U$  on which  $P$  is linear, we write  $L(P,U)$  for the corresponding linear map. This gives us

$$
L \left(P, S _ {n}\right) (x) = a _ {n} + n _ {1} x _ {1} \dots + n _ {k} x _ {k}. \tag {11}
$$

We now rewrite (11) using polyhedral geometry. Recall that a polyhedron in  $\mathbb{R}^k$  is a set of the form  $P(A,b) = \{x\in \mathbb{R}^k\mid Ax\leq b\}$ . We claim that each linear region is a polyhedron: For a fixed index  $n$ , define the matrix  $A_{n}$  to be the  $(N - 1)\times k$  matrix whose rows are the vectors  $m - n$ , where  $m$  ranges over the support of the coefficients of  $P$  (ordered lexicographically) and  $b_{n}$  to be the vector with entries  $a_{n} - a_{m}$ . Then  $S_{n} = P(A_{n},b_{n})$ . This gives us a way to encode the computation of the linear regions of tropical Puiseux polynomials using polyhedral geometry. As a direct consequence, intersections of linear regions of tropical Puiseux polynomials are also polyhedra. In particular, there are algorithms from polyhedral geometry for determining whether such polyhedra are realizable. One of the key observations given by our algorithm is that the linear regions of tropical Puiseux rational maps are almost given by  $k$ -dimensional intersections of the linear regions of the numerator and the denominator. Indeed, note that if  $U$  is a linear region of  $p$  and  $V$  a linear region of  $q$ , then we have  $L(U\cap V,p\oslash q) = L(U,p) - L(V,q)$ . The only issue that arises is that there might be some repetition in the  $L(U\cap V,p\oslash q)$  as  $U$  ranges over the linear regions of  $p$  and  $V$  over the linear regions of  $q$ . In particular, linear regions of  $p\oslash q$  might end up corresponding to unions of such  $U\cap V$ .

# 5.2 Computing Linear Regions

Determining the linear regions of a neural network may be approached numerically or symbolically. The numerical approach exploits the fact that linear regions of a neural network correspond to regions where the gradient is constant. Thus, to estimate the number of linear regions, we can evaluate the gradient on a sample of points (e.g., a mesh) in some large box  $[-R,R]^n$ . For sufficiently large  $R$  and a sufficiently dense sample of points, we get an accurate estimate. The symbolic approach, on the other hand, exploits the connection between neural networks and tropical Puiseux rational maps. Indeed, we can symbolically compute a Puiseux rational map that represents the neural network and then compute the number of linear regions using the approach outlined in section 5.1.

To compare each method, we ran the computations on smaller networks with varying sizes to compare run times and precision. For the symbolic approach, we generate 20 neural networks with random weights for each architecture and then compute the tropical Puiseux rational function associated to each neural network and compute the linear regions using Algorithm 3.

For the numerical approach, we also work with synthetic data and generate 1000 neural networks with random weights for each architecture. We then estimate the number of linear regions in a box of size  $[-10, 10]^n$  and sample 1000 points from this domain.

In both cases, we use He initialization for the weights, i.e., we generate weights with distribution  $N(0,\frac{2}{\sqrt{d}})$  where  $d$  is the input dimension. The data we obtain in this manner is summarized in Tables 10 and 11. For the symbolic approach, we also track the number of nonzero monomials to compare this quantity with the number of linear regions. For networks with 3 layers, we find the numerical estimate to be quite close, but for 4 it seems to diverge. This could be because in the numerical approach, we are only counting the number of unique Jacobians that can be found in the domain. A situation could arise where the same linear function is disconnected and hence counted twice by the symbolic approach but only once for the numerical approach.

The main observations from our experimental study are as follows. The numerical approach is faster, but offers no guarantee of precision: When running the computation for a given  $R$  and mesh grid, there seems to be no easy way of determining whether we have indeed hit all the linear regions or whether we have obtained an accurate estimate of the arrangements of these regions. It is possible to either overestimate or underestimate the number of linear regions. In particular, there is a priori no obvious way to select the parameters. We found the symbolic approach to be more precise, but slower. In general, the number of monomials seems to be far larger than the number of linear regions, which contradicts the intuition of Figure 2.

Both algorithms suffer from the curse of dimensionality: in the case of the numerical approach, the number of samples in a meshgrid grows exponentially with respect to the dimension. In the case of

the symbolic approach, calculations with polytopes seem to scale poorly with dimension and with the complexity of the neural network.

# 6 Discussion: Limitations & Directions for Future Research

In this paper, we set up a framework to interpret and analyze the expressivity of neural networks using techniques from polyhedral and tropical geometry. We demonstrated several ways in which a symbolic interpretation can often enable computational optimizations for otherwise intractable tasks and provided new insights into the inner workings of these networks. To the best of our knowledge, ours is the first work to provide practical tropical geometric theory and algorithms to numerically compute and analyze the expressivity of a neural network both in terms of inherent neural network quantities as well as tropical geometric quantities.

Despite the theoretical and practical advancement of tropical deep learning that our work offers, it is nevertheless subject to limitations, which we now discuss and which inspire directions for future research.

Experimental Limitations. The curse of dimensionality is a common theme in deep learning, and our work is unfortunately no exception. The methods introduced in this paper are quite fast for small enough networks, but scale poorly with dimension and more complex architectures.

We note that the main computational bottlenecks of the Puiseux rational function associated with a neural network are the implementation of fast multivariate Puiseux series operations. Our current computations rely on a custom implementation of this type of operation, and one potential avenue for improvement would be using such methods once they have been implemented in OSCAR [15].

For the computation of linear regions, both the numerical and symbolic approaches suffer from the curse of dimensionality. For instance, the numerical approach requires sampling on a mesh grid in a box of the form  $[-R,R]^n$  where  $n$  is the input dimension. In particular, the number of points needed is proportional to the volume, which scales exponentially in  $n$ . Similarly, the symbolic approach relies on the computation of the Puiseux rational function associated with a neural network and polytope computations, both of which are challenging computational problems in higher dimensions.

Most of our computations rely on carrying out some elementary computations many times. Thus, another avenue of improvement for this would be to parallelize.

Structural Limitations. Much of what we are studying are basically framed as a combinatorial optimization problem, which are known to be difficult. In particular, computing the Hoffman constant is equivalent to the Stewart-Todd condition measure of a matrix and both quantities are NP-hard to compute in general cases [17, 21].

Further studying and understanding where and how symbolic computation algorithms can be made more efficient, e.g., by parallelization, would make our proposed approaches more applicable to larger neural networks. Our work effectively proposes a new intersection of symbolic computation and deep learning, so there remains infrastructure to set up to make methods from these two fields compatible.

# References

[1] Grigory Mikhalkin and Johannes Rau. Tropical geometry, volume 8. MPI for Mathematics, 2009.  
[2] David Speyer and Bernd Sturmfels. Tropical Mathematics. Mathematics Magazine, 82(3):163-173, 2009.  
[3] Diane Maclagan and Bernd Sturmfels. Introduction to tropical geometry, volume 161. American Mathematical Society, 2021.  
[4] Razvan Pascanu, Guido Montufar, and Yoshua Bengio. On the number of response regions of deep feed forward networks with piece-wise linear activations. arXiv preprint arXiv:1312.6098, 2013.

[5] Guido F Montúfar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear regions of deep neural networks. Advances in neural information processing systems, 27, 2014.  
[6] Raman Arora, Amitabh Basu, Poorya Mianjy, and Anirbit Mukherjee. Understanding deep neural networks with rectified linear units. arXiv preprint arXiv:1611.01491, 2016.  
[7] Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the expressive power of deep neural networks. In international conference on machine learning, pages 2847-2854. PMLR, 2017.  
[8] Boris Hanin and David Rolnick. Deep ReLU Networks Have Surprisingly Few Activation Patterns. Advances in neural information processing systems, 32, 2019.  
[9] Huan Xiong, Lei Huang, Mengyang Yu, Li Liu, Fan Zhu, and Ling Shao. On the number of linear regions of convolutional neural networks. In International Conference on Machine Learning, pages 10514–10523. PMLR, 2020.  
[10] Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear neural networks. Journal of Computational and Applied Mathematics, 441:115667, 2024.  
[11] Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical geometry of deep neural networks. In International Conference on Machine Learning, pages 5824-5832. PMLR, 2018.  
[12] Vasileios Charisopoulos and Petros Maragos. A tropical approach to neural networks with piecewise linear activations. arXiv preprint arXiv:1805.08749, 2018.  
[13] Ruriko Yoshida, Georgios Aliatimis, and Keiji Miura. Tropical neural networks and its applications to classifying phylogenetic trees. arXiv preprint arXiv:2309.13410, 2023.  
[14] Kurt Pasque, Christopher Teska, Ruriko Yoshida, Keiji Miura, and Jefferson Huang. Tropical decision boundaries for neural networks are robust against adversarial attacks. arXiv preprint arXiv:2402.00576, 2024.  
[15] Oscar - open source computer algebra research system, version 1.0.0, 2024.  
[16] Alan J Hoffman. On approximate solutions of systems of linear inequalities. In Selected Papers Of Alan J Hoffman: With Commentary, pages 174-176. World Scientific, 2003.  
[17] Javier Pena, Juan Vera, and Luis Zuluaga. An algorithm to compute the hoffman constant of a system of linear constraints. arXiv preprint arXiv:1804.08418, 2018.  
[18] Osman Güler, Alan J Hoffman, and Uriel G Rothblum. Approximations to solutions to systems of linear inequalities. SIAM Journal on Matrix Analysis and Applications, 16(2):688-696, 1995.  
[19] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Russ R Salakhutdinov, and Alexander J Smola. Deep sets. Advances in neural information processing systems, 30, 2017.  
[20] Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear neural networks. Journal of Computational and Applied Mathematics, 441:115667, 2024.  
[21] Javier F Pena, Juan C Vera, and Luis F Zuluaga. Equivalence and invariance of the chi and hoffman constants of a matrix. arXiv preprint arXiv:1905.06366, 2019.
