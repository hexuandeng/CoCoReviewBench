# NECESSARY AND SUFFICIENT CONDITIONS FOR COMPOSITIONAL REPRESENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Humans naturally use compositional representations for flexible recognition and expression, but current machine learning lacks such ability. Despite many efforts in specific cases, there is still absence of theories and tools to study it systematically. In this paper, we leverage group theory to mathematically prove necessary and sufficient conditions for two fundamental questions of compositional representations. (1) What are the properties for a set of components to be expressed compositionally. (2) What are the properties for mappings between compositional and original representations. We provide examples to better understand the borders of the conditions. We also use the theory to provide a new explanation of why attention mechanism helps compositionality. We hope this work will help to advance understanding of compositionality and improvement of artificial intelligence towards human level.

# 1 INTRODUCTION

Humans recognize the world and create imaginations in a supple way by leveraging systematic compositionality to achieve compositional generalization, the algebraic capacity to understand and produce large amount of novel combinations from known components (Chomsky, 1957; Montague, 1970). This is a key element of human intelligence (Minsky, 1986; Lake et al., 2017), and we hope to equip machines with such ability.

Conventional machine learning has been mainly developed with an assumption that training and test distributions are identical. Compositional generalization, however, is a type of out-of-distribution generalization (Bengio, 2017) which has different training and test distributions. In compositional generalization, a sample is a combination of several components. For example, an image object may have two factor components of color and rotation. In language, a sentence is a combination of grammar and lexicon. The generalization is enabled by recombining seen components for an unseen combination during inference.

The main approach for compositional generalization is to learn compositional representations<sup>1</sup> (Bengio, 2013), which contain several component representations. Each of them depends only on the corresponding underlying factor, and does not change when other factors change. Multiple methods have been proposed to learn compositional representations. However, little discussion has been made for some fundamental questions. What kind of factor combinations can be expressed in compositional representation? Though there are some common factor components such as colors and size, what property enable them? When a set of components satisfy the conditions, what kind of mappings are available between the original and compositional representations? Can we use the conditions to explain compositionality in conventional models such as attention?

In this paper, we mathematically prove two propositions (Proposition 1.1 and Proposition 1.2) for necessary and sufficient conditions regarding compositional representations. We construct groups for changes on representations, and relate compositional representation with group direct product, and compositional mapping with group action equivalence. Then, we use theorems and propositions in group theory to prove the conditions.

Proposition 1.1 (Compositional representation). A set of components can be expressed compositionally if and only if the subgroup product equals to the original group, each component subgroup is normal subgroup of the original group, and the group elements intersect only at identity element.

Proposition 1.2 (Compositional mapping). Given compositional representation, a mapping is compositional if and only if each component has equivalent action in compositional and original representations, and for each element of the original representation, the orbits intersect only at the element.

Please see Proposition 4.2 and Proposition 4.10 for symbolic statements. We provide examples to better understand the borders of the conditions. For the condition of compositional mapping, we use this result to provide a new explanation of why attention mechanism helps compositionality. Our contributions can be summarized as follows.

- We propose and prove necessary and sufficient conditions for compositional representation and compositional mapping.  
- We provide new explanation for existing compositional models, such as attention models.

# 2 RELATED WORK

Human-level compositional learning (Marcus, 2003; Lake & Baroni, 2018) has been an important open challenge (Yang et al., 2019; Keysers et al., 2020). There are recent progress on measuring compositionality (Andreas, 2019; Lake & Baroni, 2018; Keysers et al., 2020) and learning language compositionality for compositional generalization (Lake, 2019; Russian et al., 2019; Li et al., 2019; Gordon et al., 2020; Liu et al., 2020) and continual learning (Jin et al., 2020; Li et al., 2020).

Another line of related but different work is statistically and marginally independent disentangled representation learning (Burgess et al., 2018; Locatello et al., 2019). This setting assumes marginal independence between underlying factors hence does not have compositional generalization problem. On the other hand, compositional factors may not be marginally independent.

Understanding of compositionality has been discussed over time. Some discussions following Montague (1970) uses homomorphism to define composition operation between original and representations. Recently, Li et al. (2019) defines compositionality probabilistically without discussing conditions to achieve it. Gordon et al. (2020) finds compositionality in SCAN task can be expressed as permutation group action equivalence. This equivalent action is on a component subgroup, but it does not discuss equivalent action on the whole group and the relations between them. There are also other works related to group theory in machine learning (Kondor, 2008; Cohen & Welling, 2016; Ravanbakhsh et al., 2017; Higgins et al., 2018; Kondor & Trivedi, 2018). However, the previous works do not prove conditions for compositional representation or mapping.

In this paper, we provide and theoretically prove necessary and sufficient conditions for compositional representations and compositional mappings. We use definitions, propositions and theorems from group theory. Some of them are summarized in books, such as Dummit & Foote (2004) and Gallian (2012), and we refer to them in the later sections.

# 3 PRELIMINARIES FOR GROUP THEORY

In this section, we go through some preliminaries for group theory. We provide widely used definitions, propositions and theorems, with references for more details.

# 3.1 GROUPS

Definition 3.1 (Group). A group is an ordered pair  $(G, \circ)$  where  $G$  is a set and  $\circ$  is a binary operation on  $G$  satisfying associativity, identity and inverses. We say  $G$  is a group if the operation  $\circ$  is clear from the context, and we also omit  $\circ$ . Dummit & Foote (2004) p.16 (also Gallian (2012) p.43)

Definition 3.2 (Subgroup). If a subset  $H$  of a group  $G$  is itself a group under the operation of  $G$ , we say that  $H$  is a subgroup of  $G$ . Gallian (2012) p.61 (also Dummit & Foote (2004) p.46)

Definition 3.3 (Normal subgroup). A subgroup  $H$  of a group  $G$  is called a normal subgroup of  $G$  if  $aH = Ha, \forall a \in G$ . We denote this by  $H \triangleleft G$ . Dummit & Foote (2004) p.82, Gallian (2012) p.185

# 3.2 MAPPINGS

Definition 3.4 (Group homomorphism). A homomorphism  $\varphi$  from a group  $G$  to a group  $H$  is a mapping from  $G$  into  $H$  that preserves the group operation, i.e.,  $\forall a, b \in G, \varphi(ab) = \varphi(a)\varphi(b)$ . Gallian (2012) p.208, Dummit & Foote (2004) p.36.

Definition 3.5 (Group isomorphism). The map  $\varphi : G \to H$  is called an isomorphism and  $G$  and  $H$  are said to be isomorphic or of the same isomorphism type, written  $G \cong H$ , if  $\varphi$  is a homomorphism and a bijection. Dummit & Foote (2004) p.37, Gallian (2012) p.128.

# 3.3 PRODUCTS

Definition 3.6 (Product of subgroups). Let  $H_{1}, \ldots, H_{K}$  be subgroups of a group and define  $H_{1}H_{2}\ldots H_{K} = \{h_{1}h_{2}\ldots h_{K}|h_{i}\in H_{i},\forall i = 1,\dots ,K\}$ . Dummit & Foote (2004) p.93

Definition 3.7 (External direct product). Let  $G_{1},\ldots ,G_{K}$  be a finite collection of groups. The external direct product of  $G_{1},\ldots ,G_{K}$ , written as  $G_{1}\times \dots \times G_{K}$ , is the set of all  $K$ -tuples for which the  $i$ th component is an element of  $G_{i}$  and the operation is component wise. In symbols,

$$
G _ {1} \times \dots \times G _ {K} = \left\{\left(g _ {1}, \dots , g _ {K}\right) \mid g _ {i} \in G _ {i} \right\}, \quad \left(g _ {1}, \dots , g _ {K}\right) \left(g _ {1} ^ {\prime}, \dots , g _ {K} ^ {\prime}\right) = \left(g _ {1} g _ {1} ^ {\prime}, \dots , g _ {K} g _ {K} ^ {\prime}\right)
$$

Gallian (2012) p.162, Dummit & Foote (2004) p.152

Proposition 3.1 (External direct product is a group). If  $G_{1}, \ldots, G_{K}$  are groups, their external direct product is a group. Dummit & Foote (2004) p.153. Proposition 1.

Definition 3.8 (Internal direct product). Let  $H_1, \ldots, H_K$  be a finite collection of normal subgroups of  $G$ . We say that  $G$  is the internal direct product of  $H_1, \ldots, H_K$ , if

$$
G = H _ {1} H _ {2} \dots H _ {K} \quad \text {a n d} \quad H _ {1} H _ {2} \dots H _ {i} \cap H _ {i + 1} = \{e \}, \forall i = 1, \dots , n - 1
$$

Gallian (2012) p.197, Dummit & Foote (2004) p.172.

Theorem 3.2 (Recognition theorem). If a group  $G$  is the internal direct product of a finite number of subgroups  $H_{1},\ldots ,H_{K}$ , then  $G$  is isomorphic to the external direct product of  $H_{1},\ldots ,H_{K}$ . Gallian (2012) p.198, Dummit & Foote (2004) p.171.

# 3.4 GROUP ACTIONS

Definition 3.9 (Group action). A group action of a group  $G$  on a set  $X$  is a map from  $G \times X$  to  $X$  (written as  $g(x), \forall g \in G, x \in X$ ) satisfying the following properties:

$$
g _ {1} \left(g _ {2} (x)\right) = \left(g _ {1} g _ {2}\right) (x), \forall g _ {1}, g _ {2} \in G, x \in X \quad \text {a n d} \quad e (x) = x, \forall x \in X
$$

Dummit & Foote (2004) p.112

Definition 3.10 (Product action).  $G = G_{1} \times \dots \times G_{K}$  is a group, and  $X = X_{1} \times \dots \times X_{K}$  is a set.  $G$  acts on  $X$  by the rule  $(g_{1},\ldots ,g_{K})(x_{1},\ldots ,x_{K}) = (g_{1}x_{1},\ldots ,g_{K}x_{K})$ . Cameron et al. (2008), Praeger & Schneider (2018) p.71.

Definition 3.11 (Orbit). Let  $G$  be a group action on the nonempty set  $X$ . The equivalence class  $O_{x}^{G} = \{g(x)|g\in G\}$  is called the orbit of  $G$  containing  $x$ . Dummit & Foote (2004) p.115

Definition 3.12 (Transitive action). The action of  $G$  on  $A$  is called transitive if there is only one orbit, i.e., given any two elements  $a, b \in A$  there is some  $g \in G$  such that  $a = g(b)$ . Dummit & Foote (2004) p.115

Definition 3.13 (Equivalent action). Two actions of a group  $G$  on sets  $X$  and  $Y$  are called equivalent if there is a bijection  $\varphi : X \to Y$  such that  $\varphi(g(x)) = g(\varphi(x))$  for all  $g \in G$  and  $x \in X$ . We say  $G$  eq. acts on  $X$  and  $Y$ . Lovett (2015) p.385.

# 4 COMPOSITIONALITY AND GROUPS

In this section, we derive necessary and sufficient conditions for compositionality step by step. We first construct groups for representations. We then describe compositionality with group properties, and study the conditions for them. Based on that, we further study the conditions for mappings between two representations.

# 4.1 GROUPS ON REPRESENTATIONS

Compositionality arises when we compare different samples, where some components are the same but others are not. This means compositionality is related to the changes between samples. These changes can be regarded as mappings, and since the changes are invertible, the mappings are bijective. To study compositionality we consider a set of all bijections from a set of possible representation values to the set itself, and construct a group with the following Proposition 4.1.

Proposition 4.1. Let  $X$  be any nonempty set and  $S_{X}$  be the set of all bijections from  $X$  to itself. The set  $S_{X}$  is a group under function composition<sup>2</sup>. Dummit & Foote (2004) P.29

Since  $S_{X}$  contains all bijections, the group  $S_{X}$  acts on the set  $X$  (Definition 3.9), and the action is transitive (Definition 3.12). We consider two representations and corresponding sets.  $X$  is original entangled representation, and  $Y$  is compositional representation. We create group  $G$  on set  $X$ , and group  $H$  on set  $Y$ .

# 4.2 COMPOSITIONAL REPRESENTATION

When multiple hidden variables live in the same representation, and cannot be separated by simply splitting the representation, then these variables are entangled in the representation. For example, rotation and color are two hidden variables and they are both in a representation of image. We hope to extract the hidden variables by disentangling the representation. Suppose  $X$  is a set of original representations, and  $Y$  is a set of compositional representations.  $Y$  has Cartesian product of  $K$  small sets  $Y_{1},\ldots ,Y_{K}$ . We hope to find the conditions the changes on  $X$  can be expressed by the changes on the components in  $Y$ .

A component corresponds to a set. For example, color component can take blue, green, etc., from a set of colors. With Proposition 4.1, we can construct a group for each component. With Definition 3.2, each of these groups is a subgroup of the original group.

We consider  $K$  subgroups. We hope the changes on the original representation  $X$  are equally expressed by changes on the compositional representation  $Y$ . This means group  $G$  should be isomorphic with the external direct product (Proposition 3.1) of the subgroups  $H = N_{1} \times \dots \times N_{K}$ . The following Proposition 4.2 has the necessary and sufficient conditions.

Proposition 4.2.  $N_{1},\ldots ,N_{K}$  are subgroups of group  $G$  .  $G$  is isomorphic to the external direct product of the subgroups if and only if  $G$  is internal direct product of the subgroups. From Definition 3.8, we have the following.

$$
G \cong N _ {1} \times \dots \times N _ {K} \iff \left\{ \begin{array}{l l} G = N _ {1} N _ {2} \dots N _ {K} & \text {(A 1)} \\ N _ {i} \triangleleft G, \forall i = 1, \dots , K & \text {(A 2)} \\ \left(N _ {1} \dots N _ {i}\right) \cap N _ {i + 1} = \{e \}, \forall i = 1, \dots , K - 1 & \text {(A 3)} \end{array} \right.
$$

Proof. “ $\Longleftarrow$ ”: Theorem 3.2. “ $\Longrightarrow$ ”:  $G$  and  $N_{1} \times \dots \times N_{K}$  are isomorphism, and  $N_{1} \times \dots \times N_{K}$  satisfies the conditions by construction in definition.

(A1) means the subgroup product should cover the original group. (A2) means all the component subgroups are normal subgroups of the original group. (A3) means the intersection of a subgroup and the previous subgroups only contain the identity element. This corresponds to Proposition 1.1. We will provide examples and look into more details in discussion section.

# 4.3 COMPOSITIONAL MAPPING

We consider to create a mapping between the representations  $X$  and  $Y$ , which we can use to design models. We first consider what property the mapping should satisfy. We then explore conditions for the properties, based on the compositional representation conditions mentioned above. To make the ideas clear, we summarize the relations between sets and groups in Figure 1 (left). We have a

isomorphism  $\mu$  between group  $G$  and group  $H$ .  $G$  is constructed from set  $X$  and  $H$  is constructed from set  $Y$ .  $\varphi$  is a mapping between  $X$  and  $Y$ . We denote  $N_{i}^{\prime} = \{h\in H|h_{i}\in N_{i},h_{j} = e,\forall j\neq i\}$ . Then the relations between subsets and subgroups are in Figure 1 (right).

![](images/af19c458f4d383151d8ecc4a437986421902de3ac3a3bd8797d165a79b15f86e.jpg)  
Figure 1: Equivalent group action. We break down conditions for equivalence action on whole representations (left) to each component representation (right) and their relations.

![](images/ca8863ac6655cf0af99117b385c9c65f3f9abad8c747b13c86b00a7035829d4e.jpg)

We first consider what property we hope the mapping  $\varphi$  to have. We hope both representations always change together. However, the actions are defined for different groups  $G$  and  $H$ . With Proposition 4.5 below, we define action of the same group on both representations.

Proposition 4.3. For any group  $G$  and any nonempty set  $X$  there is a bijection between the actions of  $G$  on  $X$  and the homomorphisms of  $G$  into  $S_{X}$ . Dummit & Foote (2004) p.43, p.114

Lemma 4.4. The function composition of homomorphisms is a homomorphism. Clark (1984) p.45.

Proposition 4.5. For groups  $G$  and  $H$ ,  $G \cong H$  with bijection  $\mu, H$  acts on  $X$  with homomorphism  $\sigma: H \to S_X$ , then  $G$  acts on  $X$  with homomorphism  $\sigma \circ \mu: G \to S_X$ .

Proof. With Proposition 4.3, we only need to prove  $\sigma \circ \mu$  is homomorphism. This is true because  $\sigma$  and  $\mu$  are both homomorphisms (Lemma 4.4).

With such action, we look into more details of the requirements. When an action changes one representation, it should always changes the other representation uniquely, which means the mapping should be bijective. Also, the mapping  $\varphi$  should preserve the group action. This means the group actions on  $X$  and  $Y$  are equivalent. (Definition 3.13). Note that mapping direction can be either way because the mapping is bijective.

We then consider how to make the action equivalent, with the conditions for compositional representations. We observe that  $H$  is external group acting on Cartesian set  $Y$ , the action of  $H$  on  $Y$  is product action (Definition 3.10). We look at related properties as follows.

Proposition 4.6.  $N_{1} \times \dots \times N_{K}$  has production action on  $X = X_{1} \times \dots \times X_{K}$ , then  $\forall x \in X, \forall i = 1, \ldots, K - 1: O_{x}^{N_{1} \dots N_{i}} \cap O_{x}^{N_{i + 1}} = \{x\}$ .

Proof. A direct product is isomorphic to itself, so properties in Proposition 4.2 can be used.

$$
\begin{array}{l} \forall x \in X, \forall i = 1, \dots , K - 1, \forall n \in N _ {1} N _ {2} \dots N _ {i + 1}, n (x) \in O _ {x} ^ {N _ {1} \dots N _ {i}} \cap O _ {x} ^ {N _ {i + 1}} \\ \Longrightarrow n \in N _ {1} \dots N _ {i} \cap N _ {i + 1} = \left\{e \right\} \Longrightarrow n = e \Longrightarrow n (x) = x \\ \Longrightarrow O _ {x} ^ {N _ {1} \dots N _ {i}} \cap O _ {x} ^ {N _ {i + 1}} = \{x \} \\ \end{array}
$$

![](images/e88cc7ad7e50393c61561e1dafe423188a7d9df8af640e620ffd60816d03f7c9.jpg)

Proposition 4.7.  $N_{1},\ldots ,N_{K}$  are subgroups of a group acting on a set  $X$  . If  $\forall x\in X,\forall i = 1,\dots ,K - 1:O_x^{N_1\dots N_i}\cap O_x^{N_{i + 1}} = \{x\}$  , then  $\forall x\in X,\forall n_{1}\in N_{1},n_{2}\in N_{2},\ldots ,n_{K}\in N_{K},$

$$
n _ {1} n _ {2} \dots n _ {K} (x) = x \iff n _ {i} (x) = x, \forall i = 1, \dots , K
$$

Proof. “ $\Longleftarrow$ ”: Repeat for  $i = K, \ldots, 1$ :  $n_1 n_2 \ldots n_K(x) = n_1 n_2 \ldots n_{K-1}(x) = \cdots = x$

$$
“ \Longrightarrow ”: n _ {1} n _ {2} \dots n _ {K} (x) = x \Longrightarrow n _ {K} (x) = \left(n _ {1} n _ {2} \dots n _ {K - 1}\right) ^ {- 1} (x)
$$

$\Rightarrow n_K(x)\in O_x^{N_1\dots N_{K - 1}}\cap O_x^{N_K} = \{x\}\Rightarrow n_K(x) = x$  and  $n_1n_2\dots n_{K - 1}(x) = x$

Repeat for  $i = K - 1,\ldots ,2$  we have  $n_i(x) = x,\forall i = 1,\dots ,K$

Proposition 4.8.  $N_{1} \times \dots \times N_{K}$  has production action on  $X = X_{1} \times \dots \times X_{K}$ , then  $\forall x \in X, \forall n_{1} \in N_{1}, n_{2} \in N_{2}, \ldots, n_{K} \in N_{K}$ , we have  $n_{1}n_{2} \ldots n_{K}(x) = x \iff n_{i}(x) = x, \forall i = 1, \ldots, K$ .

Proof. Proposition 4.6 and Proposition 4.7.

![](images/ca829e6a6afc0c0aeba0b2ff8f99a7c38b11d5ee803a5126c1ceb06e633cb845.jpg)

Since  $H$  and  $Y$  are composed by multiple components, we hope to explore whether the equivalence action property can be broken down to conditions on each component, and the relation between components. A natural condition on each component is that the action is equivalent for the component. On  $Y$ , by its structure, the orbits of each component group action on a element  $y$  intersects only at  $y$ , so we hope this condition also applies to  $X$ . With Proposition 4.9, we prove in the following Proposition 4.10 that the two conditions together is actually the necessary and sufficient condition for the equivalent action.

Proposition 4.9. If  $\forall x\in X,\forall i = 1,\ldots ,K - 1:O_x^{N_1\dots N_i}\cap O_x^{N_{i + 1}} = \{x\} .\forall i,\forall x\in X:$ $N_{i}$  eq. acts on  $O_{x}^{N_{i}}$  and  $O_{\varphi (x)}^{N_i}$ , then  $\varphi :X\to Y$  is one-to-one.

Proof. Any component has bijective mapping, and preserves action. We prove by contradiction.

$$
\begin{array}{l} \forall x, x ^ {\prime} (x \neq x ^ {\prime}) \in X, \exists g = n _ {1} \dots n _ {K} \in N _ {1} N _ {2} \dots N _ {K}: x ^ {\prime} = g (x) (\text {t r a n s i t i v e}, \tag {Section 4.1} \\ \text {s u p p o s e} \varphi (x) = \varphi \left(x ^ {\prime}\right), \text {t h e n} \varphi (x) = \varphi (g (x)) = \varphi \left(n _ {1} \dots n _ {K} (x)\right) = n _ {1} \dots n _ {K} (\varphi (x)) \\ \stackrel {{\Longrightarrow}} {{\longrightarrow}} \varphi (x) = n _ {i} (\varphi (x)) \underset {\text {p r e s v .}} {=} \varphi (n _ {i} (x)), \forall i = 1, \dots , K \underset {\text {b i j .}} {\Longrightarrow} x = n _ {i} (x), \forall i = 1, \dots , K \\ \underset {\text {P r o p 4 . 7}} {\Longrightarrow} x = n _ {1} \dots n _ {K} (x) = g (x) = x ^ {\prime} (\text {c o n t r a d i c t i o n}) \Rightarrow \varphi (x) \neq \varphi \left(x ^ {\prime}\right) \\ \end{array}
$$

![](images/a6051400bb52997ac79da316a48670e67cfe81bc8d3af82d881d54150104b2e8.jpg)

Proposition 4.10.  $G \cong N_1 \times \dots \times N_K$ . With  $\varphi : X \to Y$

$$
G \text {e q . a c t s} X \text {a n d} Y \Longleftrightarrow \left\{ \begin{array}{l} \forall x \in X, \forall i = 1, \dots , K - 1: O _ {x} ^ {N _ {1} \dots N _ {i}} \cap O _ {x} ^ {N _ {i + 1}} = \{x \} \\ \forall i, \forall x \in X: N _ {i} \text {e q . a c t s} O _ {x} ^ {N _ {i}} \text {a n d} O _ {\varphi (x)} ^ {N _ {i}} \end{array} \right. \tag {B2}
$$

Proof. “ $\Leftarrow$ ”. From Definition 3.13, an equivalent action is bijective and preserves action. We first prove the mapping preserves action.

$$
\forall g = n _ {1} n _ {2} \dots n _ {K} \in G, \forall x \in X, \varphi (g (x)) = \varphi \left(n _ {1} n _ {2} \dots n _ {K} (x)\right) = n _ {1} n _ {2} \dots n _ {K} (\varphi (x)) = g (\varphi (x))
$$

We then prove that  $\varphi$  is bijection on  $X\to Y$ . We prove it is one-to-one, onto, well-defined. One-to-one: Proposition 4.9. Onto:  $\forall y\in Y,\forall x\in X\nexists n_1n_2\dots n_K$  (transitive, Section 4.1):  $y = n_{1}n_{2}\dots n_{K}(\varphi (x)) = \varphi (n_{1}n_{2}\dots n_{K}(x))$ , so  $\exists x^{\prime} = n_{1}n_{2}\dots n_{K}(x)\in X:y = \varphi (x^{\prime})$ . Well-defined:  $H$  has production action on  $Y$ , and  $\varphi$  is onto, and with Proposition 4.6 and Proposition 4.9,  $\varphi^{-1}:Y\to X$  is one-to-one, so  $\varphi$  is well-defined.

“ $\Longrightarrow$ ”. We first prove (B2). Since subgroup has the same operation with the original group, the equivalent action holds for each component. We then prove (B1).

$$
\begin{array}{l} \forall x \in X, \forall i = 1, \dots , K - 1, \forall x ^ {\prime} \in O _ {x} ^ {N _ {1} \dots N _ {i}} \cap O _ {x} ^ {N _ {i + 1}}, \exists n \in N _ {1} \dots N _ {i}, n ^ {\prime} \in N _ {i + 1}: \\ x ^ {\prime} = n (x) = n ^ {\prime} (x) \underset {\text {b i j .}} {\Longrightarrow} \varphi (n x) = \varphi \left(n ^ {\prime} x\right) \underset {\text {p r e s v .}} {\Longrightarrow} n \varphi (x) = n ^ {\prime} \varphi (x) \Longrightarrow n ^ {- 1} n ^ {\prime} \varphi (x) = \varphi (x) \\ \stackrel {{\Longrightarrow}} {{\Longrightarrow}} n ^ {\prime} \varphi (x) = \varphi (x) \underset {\text {p r e s v .}} {{\Longrightarrow}} \varphi (n ^ {\prime} x) = \varphi (x) \underset {\text {b i j .}} {{\Longrightarrow}} n ^ {\prime} (x) = x \underset {} {{\Longrightarrow}} x ^ {\prime} = x \\ \end{array}
$$

![](images/b44d20762fd40c06d58ab1dfc76583b82610ceca311defa2fe7103afb9f957b3.jpg)

To summarize, this proposition says that to find whether a mapping has equivalent action on both representations, we only need to examine whether, for each element, the action is equivalent for each subgroup, and the intersections of orbits only contains the element. This corresponds to Proposition 1.2. In cases both representations are entangled, and we hope to have a compositional representation to connect them, we can use the conditions twice for the mapping. We will discuss the relation to machine learning and compositional generalization, and provide examples in the discussion section.

# 5 DISCUSSIONS

In this section, we provide examples for the conditions of compositional representation and mapping, and look into more insights for better understanding.

# 5.1 COMPOSITIONAL REPRESENTATION

We provide examples for the boundaries of compositional representation conditions. Proposition 5.1 is used to test normal subgroups.

Proposition 5.1 (Normal Subgroup Test). A subgroup  $H$  of  $G$  is normal in  $G$  if and only if  $xHx^{-1} \subseteq H, \forall x \in G$ . Dummit & Foote (2004) p.82 Theorem 6(5), Gallian (2012) p.186 Theorem 9.1.

Object transformation We think about examples violating conditions in Proposition 4.2. For a two-dimensional geometric object (e.g. image of letter "P"), we consider rotation group  $N_{1}$ , and mirror reflection group  $N_{2}$ .

If  $G = N_{1}$ , and  $N_{2}$  contains non-identity element, then  $G \neq N_{1}N_{2}$ , because any combination of rotation does not generate a reflection. This violates (A1).

We set  $G = N_{1}N_{2}$ , and both rotation and reflection take all possible values. In this case, both  $N_{1}$  and  $N_{2}$  are normal subgroups of  $G$ ,  $N_{1}, N_{2} \triangleleft G$ . However the intersection of  $N_{1}$  and  $N_{2}$  does not only contain identity,  $N_{1} \cap N_{2} \neq \{e\}$ . For example, rotating by  $\pi$  is equivalent with vertical reflection then horizontal reflection. Therefore, this violates (A3).

If we constrain reflection action to horizontal reflection, and leave rotation to have all possible values, then rotation  $N_{1}$  and horizontal reflection  $N_{2}$  form a group  $G$ . In this case,  $N_{1}$  and  $N_{2}$  only intersects at identity,  $N_{1} \cap N_{2} = \{\mathbf{e}\}$ . However,  $N_{2}$  is not normal subgroup of  $G$ . If we set rotation action  $n_{1}$  to be rotate by  $\pi / 2$ ,  $n_{1}^{-1}$  is to rotate by  $-\pi / 2$ . Action of horizontal reflection has  $n_{2} = n_{2}^{-1}$ . We find  $n_{1}n_{2}n_{1}^{-1} \notin N_{2}$ , because rotate by  $\pi / 2$ , flip horizontally, and rotate  $-\pi / 2$ , then it does not recover the original by a horizontal reflection. With Proposition 5.1,  $N_{2}$  is not a normal subgroup of  $G$ , so this violates (A2).

We further think about an example, where we also constrain the rotation to be rotate opposite (by  $\pi$ ), and reflection remains only horizontal. They form group  $G$ . In this case, rotation and reflection are both normal subgroups of  $G$ ,  $N_{1}, N_{2} \triangleleft G$ . Also,  $N_{1}$  and  $N_{2}$  only have identity in their intersection  $N_{1} \cap N_{2} = \{e\}$ . Therefore,  $N_{1}$  and  $N_{2}$  can be expressed by compositional representation.

We have another example with color as  $N_{1}$  and combination of rotation and reflection as  $N_{2}$ . They form group  $G$ . In this case, regardless the elements in the sets,  $N_{1}$  and  $N_{2}$  are always normal subgroups of  $G$ , and their intersection always contains identity. So they can be expressed as compositional representation.

From these examples, we see that whether the components can be expressed with compositional representation does not depend only on each component itself, but also on their combination, and the possible values to take. For some combinations, they are not influenced by possible values.

Grammar tree node We also look at an example for language grammar tree node. In a grammar tree, each node  $G$  has multiple children. We regard each child as a component  $N_{i}, i = 1,\dots ,K$ . We then consider whether the components can be expressed by compositional representation.

For context free grammar,  $G = N_{1}N_{2}\ldots N_{K}$ . Each  $N_{i}$  is normal subgroup of  $G$ , because a change in one children does not affect others. Also, the intersection of them only contains identity, because each sub-tree is separated. Therefore, a tree node with context free grammar is possible to be expressed with compositional representation.

We also look at an example of root node in syntactic tree for fixed length sentences. When at least one subgroup actions change the phrase length, the product of  $N_{i}$  may be not a group, because it may change the sentence length ( $G \neq N_{1}N_{2}\dots N_{K}$ ). This means  $G$  does not fit the conditions. Note that such grammar is not a context free grammar.

# 5.2 COMPOSITIONAL MAPPINGS

Conditions for compositional mapping can be used to design models for the relation between two representations. We first describe the connection with machine learning, and then use the conditions to explain some existing neural network models and tasks.

Model training and architecture for compositional generalization We consider samples in training. For compositional generalization, some elements in the whole set  $Y$  (or  $X$ ) are missing, but for each subset  $Y_{i}$ , the elements are complete. For condition (B2) in Proposition 4.10, each subset has complete samples, so it is satisfied. For condition (B1), the set has missing elements, so we do not have information to tell whether it is satisfied. To address this problem for condition (B1), we may constraint the mapping  $\varphi$  that the images for each component intersect at only one element.

Attention mechanism Attention mechanism is used for compositional modelings (Goyal et al., 2019; Russin et al., 2019; Li et al., 2019). We consider a problem that there are two components for an object. One component is the position of the object, and the other is the local shape (or word for language processing) of the object. We look into an attention network that combines the two components to generate output.

We first check whether the two components can be expressed as compositional representation. Set  $N_{1}$  is group for position,  $N_{2}$  is group for shape, and  $G = N_{1}N_{2}$ . For an object, if we change shape, position, and shape back, the shape does not change. Similarly, change position, shape and position back, the position does not change. With Proposition 5.1,  $N_{1}$  and  $N_{2}$  are both normal subgroups of  $G$ . Also,  $N_{1} \cap N_{2} = \{e\}$  because changing position does not change shape, and changing shape does not change position. This means the components can be expressed as compositional representation.

We then check whether the model is compositional mapping. First, we look at each component. For both position and shape, the mapping is bijective and preserves action. Second, we look at the orbits of images. Since the shape only changes locally, it does not change position, and position does not change shape. Therefore, the model is compositional.

Note that we do not assume that the attention is sparse for each sample. This is different from some conventional explanations (Bengio, 2017; Goyal et al., 2019) of attention mechanism. The attention helps compositional generalization not because it is dynamically sparse, but it fits the conditions. For example, when there are multiple positions to attend in one attention map, it may still fit the conditions.

Spatial transformer In the example of attention mechanism, position is aligned with one build-in dimension of data structure. However, this is not necessary. Here, we provide another example with Spatial Transformer (Jaderberg et al., 2015) for such a case. We focus on the transformations for rotation and scaling. The data structure does not have such build-in dimensions. We see that rotation and scaling satisfy Proposition 4.2 to be expressed with compositional representation, and the mapping satisfies Proposition 4.10 to be compositional mapping.

However, if we consider rotation and shape, the network might not be compositional. For example, rotations by  $0, 2\pi/3, 4\pi/3$  map a triangle it to itself, but this does not apply to a square. This means if a set of rotation contains both  $0$  and  $2\pi/3$ , it is not bijective for triangle. If it contains  $0$  but not  $2\pi/3$ , it is not bijective for square. Therefore, it violates (B2).

Ambiguous context free grammar We discussed that a node for context free grammar is able to be expressed with compositional representation. However, when there is syntactic ambiguity, we are not able to get the compositional mapping. This violates the condition (B1), because the orbits have more than one elements in the intersection.

# 6 CONCLUSION

We use group theory to prove necessary and sufficient conditions for compositional representation and mapping. We discuss examples for the conditions, and understand the boarders of them. We also provide new explanation for existing methods. We hope this work will help to advance compositionality and artificial intelligence research.

# REFERENCES

Jacob Andreas. Measuring compositionality in representation learning. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=HJz05o0qK7.  
Yoshua Bengio. Deep learning of representations: Looking forward. In International Conference on Statistical Language and Speech Processing, pp. 1-37. Springer, 2013.  
Yoshua Bengio. The consciousness prior. arXiv preprint arXiv:1709.08568, 2017.  
Christopher P Burgess, Irina Higgins, Arka Pal, Loic Matthew, Nick Watters, Guillaume Desjardins, and Alexander Lerchner. Understanding disentangling in  $\beta$ -vae. arXiv preprint arXiv:1804.03599, 2018.  
Peter J Cameron, Daniele A Gewurz, and Francesca Merola. Product action. Discrete mathematics, 308(2-3):386-394, 2008.  
Noam Chomsky. Syntactic structures. Walter de Gruyter, 1957.  
Allan Clark. Elements of abstract algebra. Courier Corporation, 1984.  
Taco Cohen and Max Welling. Group equivariant convolutional networks. In International conference on machine learning, pp. 2990-2999, 2016.  
David Steven Dummit and Richard M Foote. Abstract algebra, volume 3. Wiley Hoboken, 2004.  
Joseph Gallian. Contemporary abstract algebra. Nelson Education, 2012.  
Jonathan Gordon, David Lopez-Paz, Marco Baroni, and Diane Bouchacourt. Permutation equivariant models for compositional generalization in language. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SylVNerFvr.  
Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio, and Bernhard Scholkopf. Recurrent independent mechanisms. arXiv preprint arXiv:1909.10893, 2019.  
Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a definition of disentangled representations. arXiv preprint arXiv:1812.02230, 2018.  
Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. In Advances in neural information processing systems, pp. 2017-2025, 2015.  
Xisen Jin, Junyi Du, and Xiang Ren. Visually grounded continual learning of compositional semantics. arXiv preprint arXiv:2005.00785, 2020.  
Daniel Keysers, Nathanael Scharli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, Dmitry Tsarkov, Xiao Wang, Marc van Zee, and Olivier Bousquet. Measuring compositional generalization: A comprehensive method on realistic data. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=SygcCnNKwr.  
Imre Risi Kondor. Group theoretical methods in machine learning, volume 2. Columbia University New York, 2008.  
Risi Kondor and Shubhendu Trivedi. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In International Conference on Machine Learning, pp. 2747-2755, 2018.  
Brenden Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International Conference on Machine Learning, pp. 2879-2888, 2018.

Brenden M Lake. Compositional generalization through meta sequence-to-sequence learning. In Advances in Neural Information Processing Systems, pp. 9791-9801, 2019.  
Brenden M Lake, Tomer D Ullman, Joshua B Tenenbaum, and Samuel J Gershman. Building machines that learn and think like people. Behavioral and Brain Sciences, 40, 2017.  
Yuanpeng Li, Liang Zhao, Jianyu Wang, and Joel Hestness. Compositional generalization for primitive substitutions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4284-4293, 2019.  
Yuanpeng Li, Liang Zhao, Kenneth Church, and Mohamed Elhoseiny. Compositional language continual learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rklnDgHtDS.  
Qian Liu, Shengnan An, Jian-Guang Lou, Bei Chen, Zeqi Lin, Yan Gao, Bin Zhou, Nanning Zheng, and Dongmei Zhang. Compositional generalization by learning analytical expressions. arXiv preprint arXiv:2006.10627, 2020.  
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In International Conference on Machine Learning, pp. 4114-4124, 2019.  
Stephen Lovett. Abstract Algebra: Structures and Applications. CRC Press, 2015.  
Gary F Marcus. The algebraic mind: Integrating connectionism and cognitive science. MIT press, 2003.  
Marvin Minsky. Society of mind. Simon and Schuster, 1986.  
Richard Montague. Universal grammar. Theoria, 36(3):373-398, 1970.  
Cheryl E Praeger and Csaba Schneider. Permutation groups and cartesian decompositions, volume 449. London Mathematical Society Lecture Note Series, 2018.  
Siamak Ravanbakhsh, Jeff Schneider, and Barnabas Poczos. Equivalence through parametersharing. arXiv preprint arXiv:1702.08389, 2017.  
Jake Russian, Jason Jo, and Randall C O'Reilly. Compositional generalization in a deep seq2seq model by separating syntax and semantics. arXiv preprint arXiv:1904.09708, 2019.  
Guangyu Robert Yang, Madhura R Joglekar, H Francis Song, William T Newsome, and Xiao-Jing Wang. Task representations in neural networks trained to perform many cognitive tasks. Nature neuroscience, pp. 1, 2019.