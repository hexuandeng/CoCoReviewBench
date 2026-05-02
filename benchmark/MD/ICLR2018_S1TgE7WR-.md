# COVARIANT COMPOSITIONAL NETWORKS FOR LEARNING GRAPHS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Most existing neural networks for learning graphs address permutation invariance by conceiving of the network as a message passing scheme, where each node sums the feature vectors coming from its neighbors. We argue that this imposes a limitation on their representation power, and instead propose a new general architecture for representing objects consisting of a hierarchy of parts, which we call covariant compositional networks (CCNs). Here, covariance means that the activation of each neuron must transform in a specific way under permutations, similarly to steerability in CNNs. We achieve covariance by making each activation transform according to a tensor representation of the permutation group, and derive the corresponding tensor aggregation rules that each neuron must implement. Experiments show that CCNs can outperform competing methods on standard graph learning benchmarks.

# 1. INTRODUCTION

Learning on graphs has a long history in the kernels literature, including approaches based on random walks (Gartner, 2002; Borgwardt & Kriegel, 2005; Feragen et al., 2013), counting subgraphs (Shervashidze et al., 2009), spectral ideas (Vishwanathan et al., 2010), label propagation schemes with hashing (Shervashidze et al., 2011; Neumann et al., 2016), and even algebraic ideas (Kondor & Borgwardt, 2008). Many of these papers address moderate size problems in chemo- and bioinformatics, and the way they represent graphs is essentially fixed.

Recently, with the advent of deep learning and much larger datasets, a sequence of neural network based approaches have appeared to address the same problem, starting with (Scarselli et al., 2009). In contrast to the kernels framework, neural networks effectively integrate the classification or regression problem at hand with learning the graph representation itself, in a single, end-to-end system. In the last few years, there has been a veritable explosion in research activity in this area. Some of the proposed graph learning architectures (Duvenaud et al., 2015; Kearns et al., 2016; Niepert et al., 2016) directly seek inspiration from the type of classical CNNs that are used for image recognition (LeCun et al., 1998; Krizhevsky et al., 2012). These methods involve first fixing a vertex ordering, then moving a filter across vertices while doing some computation as a function of the local neighborhood to generate a representation. This process is then repeated multiple times like in classical CNNs to build a deep graph representation. Other notable works on graph neural networks include (Li et al., 2015; Schütt et al., 2017; Battaglia et al., 2016; Kipf & Welling, 2017). Very recently, (Gilmer et al., 2017) showed that many of these approaches can be seen to be specific instances of a general message passing formalism, and coined the term message passing neural networks (MPNNs) to refer to them collectively.

While MPNNs have been very successful in applications and are an active field of research, they differ from classical CNNs in a fundamental way: the internal feature representations in CNNs are equivariant to such transformations of the inputs as translation and rotations (Cohen & Welling, 2016a;b), the internal representations in MPNNs are fully invariant. This is a direct result of the fact that MPNNs deal with the permutation invariance issue in graphs simply by summing the messages coming from each neighbor. In this paper we argue that this is a serious limitation that restricts the representation power of MPNNs.

MPNNs are ultimately compositional (part-based) models, that build up the representation of the graph from the representations of a hierarchy of subgraphs. To address the covariance issue, we

study the covariance behavior of such networks in general, introducing a new general class of neural network architectures, which we call compositional networks (comp-nets). One advantage of this generalization is that instead of focusing attention on the mechanics of how information propagates from node to node, it emphasizes the connection to convolutional networks, in particular, it shows that what is missing from MPNNs is essentially the analog of steerability.

Steerability implies that the activations (feature vectors) at a given neuron must transform according to a specific representation (in the algebraic sense) of the symmetry group of its receptive field, in our case, the group of permutations,  $\mathbb{S}_m$ . In this paper we only consider the defining representation and its tensor products, leading to first, second, third etc. order tensor activations. We derive the general form of covariant tensor propagation in comp-nets, and find that each "channel" in the network corresponds to a specific way of contracting a higher order tensor to a lower order one. Note that here by tensor activations we mean not just that each activation is expressed as a multidimensional array of numbers (as the word is usually used in the neural networks literature), but also that it transforms in a specific way under permutations, which is a more stringent criterion. The parameters of our covariant comp-nets are the entries of the mixing matrix that prescribe how these channels communicate with each other at each node. Our experiments show that this new architecture can beat scalar message passing neural networks on several standard datasets.

# 2. LEARNING GRAPHS

Graph learning encompasses a broad range of problems where the inputs are graphs and the outputs are class labels (classification), real valued quantities (regression) or more general, possibly combinatorial, objects. In the standard supervised learning setting this means that the training set consists of  $m$  input/output pairs  $\{(G_1, y_1), (G_2, y_2), \ldots, (G_m, y_m)\}$ , where each  $G_i$  is a graph and  $y_i$  is the corresponding label, and the goal is to learn a function  $h: G \to y$  that will successfully predict the labels of further graphs that were not in the training set.

By way of fixing our notation, in the following we assume the each graph  $G$  is a pair  $(V,E)$ , where  $V$  is the vertex set of  $G$  and  $E \subseteq V \times V$  is its edge set. For simplicity, we assume that  $V = \{1,2,\ldots ,n\}$ . We also assume that  $G$  has no self-loops  $((i,i)\notin E$  for any  $i\in V$ ) and that  $G$  is symmetric, i.e.,  $(i,j)\in E\Rightarrow (j,i)\in E^1$ . We will, however, allow each edge  $(i,j)$  to have a corresponding weight  $w_{i,j}$ , and each vertex  $i$  to have a corresponding feature vector (vertex label)  $l_{i}\in \mathbb{R}^{d}$ . The latter, in particular, is important in many scientific applications, where  $l_{i}$  might encode, for example, what type of atom occupies a particular site in a molecule, or the identity of a protein in a biochemical interaction network. All the topological information about  $G$  can be summarized in an adjacency matrix  $A\in \mathbb{R}^{n\times n}$ , where  $A_{i,j} = w_{i,j}$  if  $i$  and  $j$  are connected by an edge, and otherwise  $A_{i,j} = 0$ . When dealing with labeled graphs, we also have to provide  $(l_1,\dots ,l_n)$  to fully specify  $G$ .

One of the most fascinating aspects of graphs, but also what makes graph learning challenging, is that they involve structure at multiple different scales. In the case when  $G$  is the graph of a protein, for example, an ideal graph learning algorithm would represent  $G$  in a manner that simultaneously captures structure at the level of individual atoms, functional groups, interactions between functional groups, subunits of the protein, and the protein's overall shape.

The other major requirement for graph learning algorithms relates to the fact that the usual ways to store and present graphs to learning algorithms have a critical spurious symmetry: If we were to permute the vertices of  $G$  by any permutation  $\sigma \colon \{1,2,\ldots ,n\} \to \{1,2,\ldots ,n\}$  (in other words, rename vertex 1 as  $\sigma (1)$ , vertex 2 as  $\sigma (2)$ , etc.), then the adjacency matrix would change to

$$
A _ {i, j} ^ {\prime} = A _ {\sigma^ {- 1} (i), \sigma^ {- 1} (j)},
$$

and simultaneously the vertex labels would change to  $(l_1',\ldots ,l_n')$ , where  $l^{\prime}_i = l_{\sigma^{-1}(i)}$ . However,  $G^{\prime} = (A^{\prime},l_{1}^{\prime},\dots,l_{n}^{\prime})$  would still represent exactly the same graph as  $G = (A,l_1,\dots,l_n)$ . In particular, (a) in training, whether  $G$  or  $G^{\prime}$  is presented to the algorithm must not make a difference to the final hypothesis  $h$  that it returns, (b)  $h$  itself must satisfy  $h(G) = h(G^{\prime})$  for any labeled graph and its permuted variant.

![](images/8c26c11020651cd2e43f6f3f6a044abf36994523319cd1985313da5f9c460802.jpg)  
FIGURE 1. (a) A small graph  $G$  with 6 vertices and its adjacency matrix. (b) An alternative form  $G'$  of the same graph, derived from  $G$  by renumbering the vertices by a permutation  $\sigma \colon \{1,2,\ldots,6\} \mapsto \{1,2,\ldots,6\}$ . The adjacency matrices of  $G$  and  $G'$  are different, but topologically they represent the same graph. Therefore, we expect the feature map  $\phi$  to satisfy  $\phi(G) = \phi(G')$ .

![](images/0a32e4fcbd43749a89460301bd9e03117585cf2c9bb6111bd6f4b358c5d8b75e.jpg)

![](images/820791899bfbd5da68266b065ce2460d4574cd40aaf54d46a3a559e06c5fbc4c.jpg)

![](images/f4203772308d524eca90aeaf25b22a22c169a40c56260f1ff63af1490448ca31.jpg)

Most learning algorithms for combinatorial objects hinge on some sort of fixed or learned internal representation of data, called the feature map, which, in our case we denote  $\phi(G)$ . The set of all  $n!$  possible permutations of  $\{1, 2, \ldots, n\}$  forms a group called the symmetric group of order  $n$ , denoted  $\mathbb{S}_n$ . The permutation invariance criterion can then be formulated as follows (Figure 1).

Definition 1. Let  $\mathcal{A}$  be a graph learning algorithm that uses a feature map  $G\mapsto \phi (G)$ . We say that the feature map  $\phi$  (and consequently the algorithm  $\mathcal{A}$ ) is permutation invariant if, given any  $n\in \mathbb{N}$ , any  $n$  vertex labeled graph  $G = (A,l_1,\ldots ,l_n)$ , and any permutation  $\sigma \in \mathbb{S}_n$ , letting  $G^{\prime} = (A^{\prime},l_{1}^{\prime},\dots ,l_{n}^{\prime})$ , where  $A_{i,j}^{\prime} = A_{\sigma^{-1}(i),\sigma^{-1}(j)}$  and  $l_i^\prime = l_{\sigma^{-1}(i)}$ , we have that  $\phi (G) = \phi (G^{\prime})$ .

Capturing multiscale structure and respecting permutation invariance are the two the key constraints around which most of the graph learning literature revolves. In kernel based learning, for example, invariant kernels have been constructed by counting random walks (Gartner, 2002), matching eigenvalues of the graph Laplacian (Vishwanathan et al., 2010) and using algebraic ideas (Kondor & Borgwardt, 2008).

# 3. COMPOSITIONAL NETWORKS

Many recent graph learning papers, whether or not they make this explicit, employ a compositional approach to modeling graphs, building up the representation of  $G$  from representations of subgraphs. At a conceptual level, this is similar to part-based modeling, which has a long history in machine learning (Fischler & Elschlager, 1973; Ohta et al., 1978; Tu et al., 2005; Felzenszwalb & Huttenlocher, 2005; Zhu & Mumford, 2006; Felzenszwalb et al., 2010). In this section we introduce a general, abstract architecture called compositional networks (comp-nets) for representing complex objects as a combination of their parts, and show that several existing graph neural networks can be seen as special cases of this framework.

Definition 2. Let  $\mathcal{G}$  be a compound object with  $n$  elementary parts (atoms)  $\mathcal{E} = \{e_1,\dots ,e_n\}$ . A composition scheme for  $\mathcal{G}$  is a directed acyclic graph (DAG)  $\mathcal{M}$  in which each node  $\mathfrak{n}_i$  is associated with some subset  $\mathcal{P}_i$  of  $\mathcal{E}$  (these subsets are called the parts of  $\mathcal{G}$ ) in such a way that

1. If  $\mathfrak{n}_i$  is a leaf node, then  $\mathcal{P}_i$  contains a single atom  $e_{\xi (i)}^2$  
2.  $\mathcal{M}$  has a unique root node  $\mathfrak{n}_r$ , which corresponds to the entire set  $\{e_1, \ldots, e_n\}$ .  
3. For any two nodes  $\mathfrak{n}_i$  and  $\mathfrak{n}_j$ , if  $\mathfrak{n}_i$  is a descendant of  $\mathfrak{n}_j$ , then  $\mathcal{P}_i \subset \mathcal{P}_j$ .

We define a compositional network as a composition scheme in which each node  $\mathfrak{n}_i$  also carries a feature vector  $f_{i}$  that provides a representation of the corresponding part (Figure 2). When we want to emphasize the connection to more classical neural architectures, we will refer to  $\mathfrak{n}_i$  as the  $i$ 'th neuron,  $\mathcal{P}_i$  as its receptive field<sup>3</sup>, and  $f_{i}$  as its activation.

Definition 3. Let  $\mathcal{G}$  be a compound object in which each atom  $e_i$  carries a label  $l_i$ , and  $\mathcal{M}$  a composition scheme for  $\mathcal{G}$ . The corresponding compositional network  $\mathcal{N}$  is a DAG with the same structure as  $\mathcal{M}$  in which each node  $\mathfrak{n}_i$  also has an associated feature vector  $f_i$  such that

1. If  $\mathfrak{n}_i$  is a leaf node, then  $f_{i} = l_{\xi (i)}$

![](images/c07818220b93b9f3bc3153bbd65906e489d60ad7b0fcb4ca96e706679ad48823.jpg)  
FIGURE 2. (a) A composition scheme for an object  $\mathcal{G}$  is a DAG in which the leaves correspond to atoms, the internal nodes correspond to sets of atoms, and the root corresponds to the entire object. (b) A compositional network is a composition scheme in which each node  $\mathfrak{n}_i$  also carries a feature vector  $f_i$ . The feature vector at  $\mathfrak{n}_i$  is computed from the feature vectors of the children of  $\mathfrak{n}_i$ .

![](images/b177a962f6bec214be0bd0328418886147a18bfd14bf791fa9508d9fec4b050c.jpg)

![](images/e2cb978c8a8a270f76dd5f47200f49f4f7e10d38b4276a30fb7cdf775464f52f.jpg)  
FIGURE 3. A minimal requirement for composition schemes is that they be invariant to permutation, i.e. that if the numbering of the atoms is changed by a permutation  $\sigma$ , then we must get an isomorphic DAG. Any node in the new DAG that corresponds to  $\{e_{i_1}', \ldots, e_{i_k}'\}$  must have a corresponding node in the old DAG corresponding to  $\{e_{\sigma^{-1}(i_1)}, \ldots, e_{\sigma^{-1}(i_k)}\}$ .

2. If  $\mathfrak{n}_i$  is a non-leaf node, and its children are  $\mathfrak{n}_{c_1},\ldots ,\mathfrak{n}_{c_k}$ , then  $f_{i} = \Phi (f_{c_{1}},f_{c_{2}},\dots ,f_{c_{k}})$  for some aggregation function  $\Phi$ . (Note: in general,  $\Phi$  can also depend on the relationships between the subparts, but for now, to keep the discussion as simple as possible, we ignore this possibility.)

The representation  $\phi (\mathcal{G})$  afforded by the comp-net is given by the feature vector  $f_{r}$  of the root.

Note that while, for the sake of concreteness, we call the  $f_{i}$ 's "feature vectors", there is no reason a priori why they need to be vectors rather than some other type of mathematical object. In fact, in the second half of the paper we make a point of treating the  $f_{i}$ 's as tensors, because that is what will make it the easiest to describe the specific way that they transform with respect to permutations.

In compositional networks for graphs, the atoms will usually be the vertices, and the  $\mathcal{P}_i$  parts will correspond to clusters of nodes or neighborhoods of given radii. Comp-nets are particularly attractive in this domain because they can combine information from the graph at different scales. The comp-net formalism also suggests a natural way to satisfy the permutation invariance criterion of Definition 1.

Definition 4. Let  $\mathcal{M}$  be the composition scheme of an object  $\mathcal{G}$  with  $n$  atoms and  $\mathcal{M}'$  the composition scheme of another object that is equivalent in structure to  $\mathcal{G}$ , except that its atoms have been permuted by some permutation  $\sigma \in \mathbb{S}_n$  ( $e_i' = e_{\sigma^{-1}(i)}$  and  $\ell_i' = \ell_{\sigma^{-1}(i)}$ ). We say that  $\mathcal{M}$  (more precisely,

the algorithm generating  $\mathcal{M}$  is permutation invariant if there is a bijection  $\psi \colon \mathcal{M} \to \mathcal{M}'$  taking each  $\mathfrak{n}_a \in \mathcal{M}$  to some  $\mathfrak{n}_b' \in \mathcal{M}'$  such that if  $\mathcal{P}_a = \{e_{i_1}, \ldots, e_{i_k}\}$ , then  $\mathcal{P}_b' = \{e_{\sigma(i_1)}', \ldots, e_{\sigma(i_k')}'\}$ .

Proposition 1. Let  $\phi(\mathcal{G})$  be the output of a comp-net based on a composition scheme  $\mathcal{M}$ . Assume

1.  $\mathcal{M}$  is permutation invariant in the sense of Definition 4.  
2. The aggregation function  $\Phi(f_{c_1}, f_{c_2}, \ldots, f_{c_k})$  used to compute the feature vector of each node from the feature vectors of its children is invariant to the permutations of its arguments.

Then the overall representation  $\phi(\mathcal{G})$  is invariant to permutations of the atoms. In particular, if  $\mathcal{G}$  is a graph and the atoms are its vertices, then  $\phi$  is a permutation invariant graph representation.

# 3.1.MESSAGE PASSING NEURAL NETWORKS AS A SPECIAL CASE OF COMP-NETS

Graph learning is not the only domain where invariance and multiscale structure are important: the most commonly cited reasons for the success of convolutional neural networks (CNNs) in image tasks is their ability to address exactly these two criteria in the vision context. Furthermore, each neuron  $\mathfrak{n}_i$  in a CNN aggregates information from a small set of neurons from the previous layer, therefore its receptive field, corresponding to  $\mathcal{P}_i$ , is the union of the receptive fields of its "children", so we have a hierarchical structure very similar to that described in the previous section. In this sense, CNNs are a specific kind of compositional network, where the atoms are pixels. This connection has inspired several authors to frame graph learning as a generalization of convolutional nets to the graph domain (Bruna et al., 2014; Henaff et al., 2015; Duvenaud et al., 2015; Defferrard et al., 2016; Kipf & Welling, 2017). While in mathematics convolution has a fairly specific meaning that is side-stepped by this analogy, the CNN analogy does suggest that a natural way to define the  $\Phi$  aggregation functions is to let  $\Phi(f_{c_1}, f_{c_2}, \ldots, f_{c_k})$  be a linear function of  $f_{c_1}, f_{c_2}, \ldots, f_{c_k}$  followed by a pointwise nonlinearity, such as a ReLU operation.

To define a comp-net for graphs we also need to specify the composition scheme  $\mathcal{M}$ . Many algorithms define  $\mathcal{M}$  in layers, where each layer (except the last) has one node for each vertex of  $G$ :

$\mathcal{M}1$ . In layer  $\ell = 0$  each node  $\mathfrak{n}_i^0$  represents the single vertex  $\mathcal{P}_i^0 = \{i\}$ .

$\mathcal{M}2$ . In layers  $\ell = 1,2,\ldots ,L$ , node  $\mathfrak{n}_i^\ell$  is connected to all nodes from the previous level that are neighbors of  $i$  in  $G$ , i.e., the children of  $\mathfrak{n}_i^\ell$  are

$$
\operatorname {c h} \left(\mathfrak {n} _ {i} ^ {\ell}\right) = \left\{\mathfrak {n} _ {j} ^ {\ell - 1} \mid j \in \mathcal {N} (i) \right\},
$$

where  $\mathcal{N}(i)$  denotes the set of neighbors of  $i$  in  $G$ . Therefore,  $\mathcal{P}_i^\ell = \bigcup_{j\in \mathcal{N}(i)}\mathcal{P}_j^{\ell -1}$ .

$\mathcal{M}3$ . In layer  $L + 1$  we have a single node  $\mathfrak{n}_r$  that represents the entire graph and collects information from all nodes at level  $L$ .

Since this construction only depends on topological information about  $G$ , the resulting composition scheme is guaranteed to be permutation invariant in the sense of Definition 4.

A further important consequence of this way of defining  $\mathcal{M}$  is that the resulting comp-net can be equivalently interpreted as label propagation algorithm, where in each round  $\ell = 1,2,\dots ,L$ , each vertex aggregates information from its neighbors and then updates its own label.

Algorithm 1 The label propagation algorithm corresponding to  $\mathcal{M}1 - \mathcal{M}3$  
for each vertex  $i$ $f_{i}^{0} \gets l_{i}$   
for  $\ell = 1$  to  $L$   
for each vertex  $i$ $f_{i}^{\ell} \gets \Phi(f_{i_{1}}^{\ell - 1}, \ldots, f_{i_{k}}^{\ell - 1})$  where  $\mathcal{N}(i) = \{i_{1}, \ldots, i_{k}\}$ $\phi(G) \equiv f_{r} \gets \Phi(f_{1}^{L}, \ldots, f_{n}^{L})$

Many authors choose to describe graph neural networks exclusively in terms of label propagation, without mentioning the compositional aspect of the model. Gilmer et al. (2017) call this general approach message passing neural networks, and point out that a range of different graph learning architectures are special cases of it. More broadly, the classic Weisfeiler-Lehman test of isomorphism also follows the same logic (Weisfeiler & Lehman, 1968; Read & Corneil, 1977; Cai et al., 1992),

FIGURE 4. In convolutional neural networks if the input image is translated by some amount  $(t_1, t_2)$ , what used to fall in the receptive field of neuron  $\mathfrak{n}_{i,j}^{\ell}$  is moved to the receptive field of  $\mathfrak{n}_{i + t_1, j + t_2}^{\ell}$ . Therefore, the activations transform in the very simple way  $f_{i + t_1, j + t_2}^{\ell} = f_{i,j}^{\ell}$ . In contrast, rotations not only move the receptive fields around, but also permute the neurons in the receptive field internally, therefore, in general,  $f_{j,-i}^{\ell} \neq f_{i,j}^{\ell}$ . The right hand figure shows that if the CNN has a horizontal filter (blue) and a vertical one (red) then their activations are exchanged by a 90 degree rotation. In steerable CNNs, if  $(i, j) \mapsto (i', j')$ , then  $f_{i', j'}^{\ell} = R(f_{i,j}^{\ell})$  for some fixed linear function of the rotation.

and so does the related Weisfeiler-Lehman kernel, arguably the most successful kernel-based approach to graph learning (Shervashidze et al., 2011). Note also that in label propagation or message passing algorithms there is a clear notion of the source domain of vertex  $i$  at round  $\ell$ , as the set of vertices that can influence  $f_{i}^{\ell}$ , and this corresponds exactly to the receptive field  $\mathcal{P}_i^\ell$  of "neuron"  $\mathfrak{n}_i^\ell$  in the comp-net picture.

The following proposition is immediate from the form of Algorithm 1 and reassures us that message passing neural networks, as special cases of comp-nets, do indeed produce permutation invariant representations of graphs.

Proposition 2. Any label propagation scheme in which the aggregation function  $\Phi$  is invariant to the permutations of its arguments is invariant to permutations in the sense of Definition 1.

In the next section we argue that invariant message passing networks are limited in their representation power, however, and describe a generalization via comp-nets that overcomes some of these limitations.

# 4. COVARIANT COMPOSITIONAL NETWORKS

One of the messages of the present paper is that invariant message passing algorithms, of the form described in the previous section, are not the most general possible compositional models for producing permutation invariant representations of graphs (or of compound objects, in general).

Once again, an analogy with image recognition is helpful. Classical CNNs face two types of basic image transformations: translations and rotations. With respect to translations (barring pooling, edge effects and other complications), CNNs behave in a quasi-invariant way, in the sense that if the input image is translated by any integer amount  $(t_x,t_y)$ , the activations in each layer  $\ell = 1,2,\ldots L$  translate the same way: the activation of any neuron  $\mathfrak{n}_{i,j}^{\ell}$  is simply transferred to neuron  $\mathfrak{n}_{i + t_1,j + t_2}^{\ell}$ , i.e.,  $f_{i + t_1,j + t_2}^{\prime \ell} = f_{i,j}^{\ell}$ . This is the simplest manifestation of a well studied property of CNNs called equivariance (Cohen & Welling, 2016a; Worrall et al., 2017).

With respect to rotations, however, the situation is more complicated: if we rotate the input image by, e.g., 90 degrees, not only will the part of the image that fell in the receptive field of a particular neuron  $\mathfrak{n}_{i,j}^{\ell}$  move to the receptive field of a different neuron  $\mathfrak{n}_{j,-i}^{\ell}$ , but the orientation of the receptive field will also change (Figure 4). Consequently, features which were, for example, previously picked up by horizontal filters will now be picked up by vertical filters. Therefore, in general,  $f_{j,-i}^{\ell} \neq f_{i,j}^{\ell}$ . It can be shown that one cannot construct a CNN for images that behaves in a quasi-invariant way with respect to both translations and rotations unless every filter is directionless.

It is, however, possible to construct a CNN in which the activations transform in a predictable and reversible way, in particular,  $f_{j, -i}^{\ell} = R(f_{i,j}^{\ell})$  for some fixed invertible function  $R$ . This phenomenon is called steerability, and has a significant literature in both classical signal processing (Freeman & Adelson, 1991; Simoncelli et al., 1992; Perona, 1995; Teo & Hel-Or, 1998; Manduchi et al., 1998) and the neural networks field (Cohen & Welling, 2016b).

The situation in compositional networks is similar. The comp-net and message passing architectures that we have examined so far, by virtue of the aggregation function being symmetric in its arguments, are all quasi-invariant (with respect to permutations) in the following sense.

Definition 5. Let  $\mathcal{G}$  be a compound object of  $n$  parts and  $\mathcal{G}'$  an equivalent object in which the atoms have been permuted by some permutation  $\sigma$ . Let  $\mathcal{N}$  be a comp-net for  $\mathcal{G}$  based on an invariant

![](images/d67601d1d3987b6f5180f6d91c7ace424f52e3c49f45841dbdd80222121b5a24.jpg)

![](images/74277aa247149db319b057253b340475011f46beed39cdd016da9ac13306c09c.jpg)

![](images/544ef7adfa71a8eda8f36bdf1c3e866a30f6bab2a980fe77711bb644fb914334.jpg)  
FIGURE 5. Top left: At level  $\ell = 1$ $\mathfrak{n}_3$  aggregates information from  $\{\mathfrak{n}_4,\mathfrak{n}_5\}$  and  $\mathfrak{n}_2$  aggregates information  $\{\mathfrak{n}_5,\mathfrak{n}_6\}$ . At  $\ell = 2$ ,  $\mathfrak{n}_1$  collects this summary information from  $\mathfrak{n}_3$  and  $\mathfrak{n}_2$ . Bottom left: This graph is not isomorphic to the top one, but the activations of  $\mathfrak{n}_3$  and  $\mathfrak{n}_2$  at  $\ell = 1$  will be identical. Therefore, at  $\ell = 2$ ,  $\mathfrak{n}_1$  will get the same inputs from its neighbors, irrespective of whether or not  $\mathfrak{n}_5$  and  $\mathfrak{n}_7$  are the same node or not. Right: Aggregation at different levels. For keeping the figure legible only the neighborhood around one node in higher levels is marked.

composition scheme, and  $\mathcal{N}'$  be the corresponding network for  $\mathcal{G}'$ . We say that  $\mathcal{N}$  is quasi-invariant if for any  $\mathfrak{n}_i \in \mathcal{N}$ , letting  $\mathfrak{n}_j'$  be the corresponding node in  $\mathcal{N}'$ ,  $f_i = f_j'$  for any  $\sigma \in \mathbb{S}_n$

Quasi-invariance in comp-nets is equivalent to the assertion that the activation  $f_{i}$  at any given node must only depend on  $\mathcal{P}_i = \{e_{j_1},\dots ,e_{j_k}\}$  as a set, and not on the internal ordering of the atoms  $e_{j_1},\ldots ,e_{j_k}$  making up the receptive field. At first sight this seems desirable, since it is exactly what we expect from the overall representation  $\phi (G)$ . On closer examination, however, we realize that this property is potentially problematic, since it means that  $\mathfrak{n}_i$  has lost all information about which vertex in its receptive field has contributed what to the aggregate information  $f_{i}$ . In the CNN analogy, we can say that we have lost information about the orientation of the receptive field. In particular, if, further upstream,  $f_{i}$  is combined with some other feature vector  $f_{j}$  from a node with an overlapping receptive field, the aggregation process has no way of taking into account which parts of the information in  $f_{i}$  and  $f_{j}$  come from shared vertices and which parts do not (Figure 5).

The solution is to upgrade the  $\mathcal{P}_i$  receptive fields to be ordered sets, and explicitly establish how  $f_{i}$  co-varies with the internal ordering of the receptive fields. To emphasize that henceforth the  $\mathcal{P}_i$  sets are ordered, we will use parentheses rather than braces to denote their content.

Definition 6. Let  $\mathcal{G}$ ,  $\mathcal{G}'$ ,  $\mathcal{N}$  and  $\mathcal{N}'$  be as in Definition 5. Let  $\mathfrak{n}_i$  be any node of  $\mathcal{N}$  and  $\mathfrak{n}_j$  the corresponding node of  $\mathcal{N}'$ . Assume that  $\mathcal{P}_i = (e_{p_1},\ldots,e_{p_m})$  while  $\mathcal{P}_j' = (e_{q_1},\ldots,e_{q_m})$ , and let  $\pi \in \mathbb{S}_m$  be the permutation that aligns the orderings of the two receptive fields, i.e., for which  $e_{q_{\pi(a)}} = e_{p_a}$ . We say that  $\mathcal{N}$  is covariant to permutations if for any  $\pi$ , there is a corresponding function  $R_{\pi}$  such that  $f_j' = R_{\pi}(f_i)$ .

# 4.1. FIRST ORDER COVARIANT COMP-NETS

The form of covariance prescribed by Definition 6 is very general. To make it more specific, in line with the classical literature on steerable representations, we make the assumption that the  $\{f\mapsto R_{\pi}(f)\}_{\pi \in \mathbb{S}_m}$  maps are linear, and by abuse of notation, from now on simply treat them as matrices (with  $R_{\pi}(f) = R_{\pi}f$ ). The linearity assumption automatically implies that  $\{R_{\pi}\}_{\pi \in \mathbb{S}_m}$

is a representation of  $\mathbb{S}_m$  in the group theoretic sense of the word (for the definition of group representations, see the Appendix) $^4$ .

Proposition 3. If for any  $\pi \in \mathbb{S}_m$ , the  $f \mapsto R_{\pi}(f)$  map appearing in Definition 6 is linear, then the corresponding  $\{R_{\pi}\}_{\pi \in \mathbb{S}_m}$  matrices form a representation of  $\mathbb{S}_m$ .

The representation theory of symmetric groups is a rich subject that goes beyond the scope of the present paper (Sagan, 2001). However, there is one particular representation of  $\mathbb{S}_m$  that is likely familiar even to non-algebraists, the so-called defining representation, given by the  $P_{\pi} \in \mathbb{R}^{n \times n}$  permutation matrices

$$
[ P _ {\pi} ] _ {i, j} = \left\{ \begin{array}{l l} 1 & \text {i f} \pi (j) = i \\ 0 & \text {o t h e r w i s e .} \end{array} \right.
$$

It is easy to verify that  $P_{\pi_2\pi_1} = P_{\pi_2}P_{\pi_1}$  for any  $\pi_1,\pi_2\in \mathbb{S}_m$ , so  $\{P_{\pi}\}_{\pi \in \mathbb{S}_m}$  is indeed a representation of  $\mathbb{S}_m$ . If the transformation rules of the  $f_{i}$  activations in a given comp-net are dictated by this representation, then each  $f_{i}$  must necessarily be a  $|\mathcal{P}_i|$  dimensional vector, and intuitively each component of  $f_{i}$  carries information related to one specific atom in the receptive field, or the interaction of that specific atom with all the others. We call this case first order permutation covariance.

Definition 7. We say that  $\mathfrak{n}_i$  is a first order covariant node in a comp-net if under the permutation of its receptive field  $\mathcal{P}_i$  by any  $\pi \in \mathbb{S}_{|\mathcal{P}_i|}$ , its activation transforms as  $f_i \mapsto P_{\pi} f_i$ .

# 4.2. SECOND ORDER COVARIANT COMP-NETS

It is easy to verify that given any representation  $(R_g)_{g\in \mathfrak{G}}$  of a group  $\mathfrak{G}$ , the matrices  $(R_g\otimes R_g)_{g\in \mathfrak{G}}$  also furnish a representation of  $\mathfrak{G}$ . Thus, one step up in the hierarchy from  $P_{\pi}$ -covariant comp-nets are  $P_{\pi}\otimes P_{\pi}$ -covariant comp-nets, where the  $f_{i}$  feature vectors are now  $|\mathcal{P}_i|^2$  dimensional vectors that transform under permutations of the internal ordering by  $\pi$  as  $f_{i}\mapsto (P_{\pi}\otimes P_{\pi})f_{i}$ .

If we reshape  $f_{i}$  into a matrix  $F_{i}\in \mathbb{R}^{|\mathcal{P}_{i}|\times |\mathcal{P}_{i}|}$ , then the action

$$
F _ {i} \mapsto P _ {\pi} F _ {i} P _ {\pi} ^ {\top}
$$

is equivalent to  $P_{\pi} \otimes P_{\pi}$  acting on  $f_{i}$ . In the following, we will prefer this more intuitive matrix view, since it clearly expresses that feature vectors that transform this way express relationships between the different constituents of the receptive field. Note, in particular, that if we define  $A \downarrow_{\mathcal{P}_i}$  as the restriction of the adjacency matrix to  $\mathcal{P}_i$  (i.e., if  $\mathcal{P}_i = (e_{p_1}, \ldots, e_{p_m})$  then  $[A \downarrow_{\mathcal{P}_i}]_{a,b} = A_{p_a, p_b}$ ), then  $A \downarrow_{\mathcal{P}_i}$  transforms exactly as  $F_i$  does in the equation above.

Definition 8. We say that  $\mathfrak{n}_i$  is a second order covariant node in a comp-net if under the permutation of its receptive field  $\mathcal{P}_i$  by any  $\pi \in \mathbb{S}_{|\mathcal{P}_i|}$ , its activation transforms as  $F_{i} \mapsto P_{\pi} F_{i} P_{\pi}^{\top}$ .

# 4.3. THIRD AND HIGHER ORDER COVARIANT COMP-NETS

Taking the pattern further lets us consider third, fourth, and general,  $k$ 'th order nodes in our compnet, in which the activations are  $k$ 'th order tensors, transforming under permutations as

$$
F _ {i} \mapsto F _ {i} ^ {\prime} \qquad \text {w h e r e} \qquad [ F _ {i} ^ {\prime} ] _ {j _ {1}, \ldots , j _ {k}} = \sum_ {j _ {1} ^ {\prime}} \sum_ {j _ {2} ^ {\prime}} \ldots \sum_ {j _ {k} ^ {\prime}} [ P _ {\pi} ] _ {j _ {1}, j _ {1} ^ {\prime}} [ P _ {\pi} ] _ {j _ {2}, j _ {2} ^ {\prime}} \ldots [ P _ {\pi} ] _ {j _ {k}, j _ {k} ^ {\prime}} [ F _ {i} ] _ {j _ {1} ^ {\prime}, \ldots , j _ {k} ^ {\prime}},
$$

In the more compact, so called Einstein notation<sup>5</sup>,

$$
[ F _ {i} ^ {\prime} ] _ {j _ {1}, \dots , j _ {k}} = [ P _ {\pi} ] _ {j _ {1}} ^ {j _ {1} ^ {\prime}} [ P _ {\pi} ] _ {j _ {2}} ^ {j _ {2} ^ {\prime}} \dots [ P _ {\pi} ] _ {j _ {k}} ^ {j _ {k} ^ {\prime}} [ F _ {i} ] _ {j _ {1} ^ {\prime}, \dots , j _ {k} ^ {\prime}}. \tag {1}
$$

In general, we will call any quantity which transforms according to this equation a  $\mathbf{k}^{\prime}$ th order  $\mathbf{P}$  tensor. Note that this notion of tensors is distinct from the common usage of the term in neural

networks, and more similar to how the word is used in Physics, because it not only implies that  $F_{i}$  is a quantity representable by an  $m \times m \times \ldots \times m$  array of numbers, but also that  $F_{i}$  transforms in a specific way.

Since scalars, vectors and matrices can be considered as  $0^{\mathrm{th}}$ ,  $1^{\mathrm{st}}$  and  $2^{\mathrm{nd}}$  order tensors, respectively, the following definition covers Definitions 5, 7 and 8 as special cases (with quasi-invariance being equivalent to zeroth order equivariance). To unify notation and terminology, regardless of the dimensionality, in the following we will always talk about feature tensors rather than feature vectors, and denote the activations with  $F_{i}$  rather than  $f_{i}$ , as we did in the first half of the paper.

Definition 9. We say that  $\mathfrak{u}_i$  is a  $k$ 'th order covariant node in a comp-net if the corresponding activation  $F_i$  is a  $k$ 'th order  $P$ -tensor, i.e., it transforms under permutations of  $\mathcal{P}_i$  according to (1), or the activation is a sequence of  $c$  separate  $P$ -tensors  $F_i^{(1)}, \ldots, F_i^{(c)}$  corresponding to  $c$  distinct channels.

# 5. TENSOR AGGREGATION RULES

The previous sections prescribed how activations must transform in comp-nets of different orders, but did not explain how this can be assured, and what it entails for the  $\Phi$  aggregation functions. Fortunately, tensor arithmetic provides a compact framework for deriving the general form of these operations. Recall the four basic operations that can be applied to tensors<sup>6</sup>:

1. The tensor product of  $A \in \mathcal{T}^k$  with  $B \in \mathcal{T}^p$  yields a tensor  $C = A \otimes B \in \mathcal{T}^{p + k}$  where

$$
C _ {i _ {1}, i _ {2}, \dots , i _ {k + p}} = A _ {i _ {1}, i _ {2}, \dots , i _ {k}} B _ {i _ {k + 1}, i _ {k + 2}, \dots , i _ {k + p}}.
$$

2. The elementwise product of  $A \in \mathcal{T}^k$  with  $B \in \mathcal{T}^p$  along dimensions  $(a_1, a_2, \ldots, a_p)$  yields a tensor  $C = A \odot_{(a_1, \ldots, a_p)} B \in \mathcal{T}^k$  where

$$
C _ {i _ {1}, i _ {2}, \dots , i _ {k}} = A _ {i _ {1}, i _ {2}, \dots , i _ {k}} B _ {i _ {a _ {1}}, i _ {a _ {2}}, \dots , i _ {a _ {p}}}.
$$

3. The projection (summation) of  $A \in \mathcal{T}^k$  along dimensions  $\{a_1, a_2, \ldots, a_p\}$  yields a tensor  $C = A \downarrow_{a_1, \ldots, a_p} \in \mathcal{T}^{k-p}$  with

$$
C _ {i _ {1}, i _ {2}, \dots , i _ {k}} = \sum_ {i _ {a _ {1}}} \sum_ {i _ {a _ {2}}} \dots \sum_ {i _ {a _ {p}}} A _ {i _ {1}, i _ {2}, \dots , i _ {k}},
$$

where we assume that  $i_{a_1}, \ldots, i_{a_p}$  have been removed from amongst the indices of  $C$ .

4. The contraction of  $A \in \mathcal{T}^k$  along the pair of dimensions  $\{a, b\}$  (assuming  $a < b$ ) yields a  $k - 2$  order tensor

$$
C _ {i _ {1}, i _ {2}, \dots , i _ {k}} = \sum_ {j} A _ {i _ {1}, \dots , i _ {a - 1}, j, i _ {a + i}, \dots , i _ {b - 1}, j, i _ {b + 1}, \dots , k},
$$

where again we assume that  $i_{a}$  and  $i_{b}$  have been removed from amongst the indices of  $C$ . Using Einstein notation this can be written much more compactly as

$$
C _ {i _ {1}, i _ {2}, \dots , i _ {k}} = A _ {i _ {1}, i _ {2}, \dots , i _ {k}} \delta^ {i _ {a}, i _ {b}},
$$

where  $\delta^{i_a,i_b}$  is the diagonal tensor with  $\delta^{i,j} = 1$  if  $i = j$  and 0 otherwise. In a somewhat unorthodox fashion, we also generalize contractions to (combinations of) larger sets of indices  $\{\{a_1^1,\ldots ,a_{p_1}^1\} ,\{a_1^2,\ldots ,a_{p_2}^2\} ,\ldots ,\{a_1^q,\ldots ,a_{p_q}^q\} \}$  as the  $(k - \sum_{j}p_{j})$  order tensor

$$
C _ {\ldots} = A _ {i _ {1}, i _ {2}, \ldots , i _ {k}} \delta^ {a _ {1} ^ {1}, \ldots , a _ {p _ {1}} ^ {1}} \delta^ {a _ {1} ^ {2}, \ldots , a _ {p _ {2}} ^ {2}} \ldots \delta^ {a _ {1} ^ {q}, \ldots , a _ {p _ {q}} ^ {q}}.
$$

Note that this subsumes projections, since it allows us to write  $A \downarrow_{a_1, \ldots, a_p}$  in the slightly unusual looking form

$$
A \downarrow_ {a _ {1}, \dots , a _ {p}} = A _ {i _ {1}, i _ {2}, \dots , i _ {k}} \delta^ {i _ {a _ {1}}} \delta^ {i _ {a _ {2}}} \dots \delta^ {i _ {a _ {k}}}.
$$

The following proposition shows that, remarkably, all of the above operations (as well as taking linear combinations) preserve the way that  $P$ -tensors behave under permutations and thus they can be freely "mixed and matched" within  $\Phi$ .

Proposition 4. Assume that  $A$  and  $B$  are  $k$ 'th and  $p$ 'th order  $P$ -tensors, respectively. Then

1.  $A\otimes B$  is a  $k + p$  th order  $P$  -tensor.  
2.  $A\odot_{(a_1,\dots,a_p)}B$  is a k'th order  $P$  -tensor.  
3.  $A\downarrow_{a_1,\dots ,a_n}$  is a  $k - p$  th order  $P$  -tensor.  
4.  $A_{i_1,i_2,\dots ,i_k}\delta^{a_1^1,\dots ,a_{p_1}^1}\ldots \delta^{a_1^q,\dots ,a_{p_q}^q}$  is a  $k - \sum_{j}p_{j}$  th order  $P$  -tensor.

In addition, if  $A_1, \ldots, A_u$  are  $k$ 'th order  $P$ -tensors and  $\alpha_1, \ldots, \alpha_u$  are scalars, then  $\sum_{j} \alpha_j A_j$  is a  $k$ 'th order  $P$ -tensor.

The more challenging part of constructing the aggregation scheme for comp-nets is establishing how to relate  $P$ -tensors at different nodes. The following two propositions answer this question.

Proposition 5. Assume that node  $\mathfrak{n}_a$  is a descendant of node  $\mathfrak{n}_b$  in a comp-net  $\mathcal{N}$ ,  $\mathcal{P}_a = (e_{p_1},\ldots ,e_{p_m})$  and  $\mathcal{P}_b = (e_{q_1},\ldots ,e_{q_{m'}})$  are the corresponding ordered receptive fields (note that this implies that, as sets,  $\mathcal{P}_a\subseteq \mathcal{P}_b$ ), and  $\chi^{a\to b}\in \mathbb{R}^{m\times m'}$  is an indicator matrix defined

$$
\chi_ {i, j} ^ {a \to b} = \left\{ \begin{array}{l l} 1 & i f q _ {j} = p _ {i} \\ 0 & o t h e r w i s e. \end{array} \right.
$$

Assume that  $F$  is a  $k$ 'th order  $P$ -tensor with respect to permutations of  $(e_{p_1}, \ldots, e_{p_m})$ . Then, dropping the  $a \rightarrow b$  superscript for clarity,

$$
\widetilde {F} _ {i _ {1}, \dots , i _ {k}} = \chi_ {i _ {1}} ^ {j _ {1}} \chi_ {i _ {2}} ^ {j _ {2}} \dots \chi_ {i _ {k}} ^ {j _ {k}} F _ {j _ {1}, \dots , j _ {k}} \tag {2}
$$

is a  $k$ 'th order  $P$ -tensor with respect to permutations of  $(e_{q_1}, \ldots, e_{q_{m'}})$ .

Equation 2 tells us that when node  $\mathfrak{n}_b$  aggregates  $P$ -tensors from its children, it first has to "promote" them to being  $P$ -tensors with respect to the contents of its own receptive field by contracting along each of their dimensions with the appropriate  $\chi^{a\to b}$  matrix. This is a critical element in comp-nets to guarantee covariance.

Proposition 6. Let  $\mathfrak{n}_{c_1},\ldots ,\mathfrak{n}_{c_s}$  be the children of  $\mathfrak{n}_t$  in a message passing type comp-net with corresponding  $k$ th order tensor activations  $F_{c_1},\dots ,F_{c_s}$ . Let

$$
[ \widetilde {F} _ {c _ {u}} ] _ {i _ {1}, \dots , i _ {k}} = [ \chi^ {c _ {u} \rightarrow t} ] _ {i _ {1}} ^ {j _ {1}} [ \chi^ {c _ {u} \rightarrow t} ] _ {i _ {2}} ^ {j _ {2}} \dots [ \chi^ {c _ {u} \rightarrow t} ] _ {i _ {k}} ^ {j _ {k}} [ F _ {c _ {u}} ] _ {j _ {1}, \dots , j _ {k}}
$$

be the promotions of these activations to  $P$ -tensors of  $\mathfrak{n}_{t}$ . Assume that  $\mathcal{P}_t = (e_{p_1}, \ldots, e_{p_m})$ . Now let  $\overline{F}$  be a  $k + 1$ th order object in which the  $j$ th slice is  $\bar{F}_{p_j}$  if  $\mathfrak{n}_{p_j}$  is one of the children of  $\mathfrak{n}_t$ , i.e.,

$$
\overline {{F}} _ {i _ {1}, \dots , i _ {k}, j} = \left[ \widetilde {F} _ {p _ {j}} \right] _ {i _ {1}, \dots , i _ {k}},
$$

and zero otherwise. Then  $\overline{F}$  is a  $k + 1$ th order  $P$ -tensor of  $\mathfrak{n}_t$ .

Finally, as already mentioned, the restriction of the adjacency matrix to  $\mathcal{P}_i$  is a second order  $P$ -tensor, which gives an easy way of explicitly adding topological information to the activation.

Proposition 7. If  $F_{i}$  is a  $k$ 'th order  $P$ -tensor at node  $\mathfrak{n}_i$ , and  $A\downarrow_{\mathcal{P}_i}$  is the restriction of the adjacency matrix to  $\mathcal{P}_i$  as defined in Section 4.2, then  $F\otimes A\downarrow_{\mathcal{P}_i}$  is a  $k + 2$ 'th order  $P$ -tensor.

# 5.1. THE GENERAL AGGREGATION FUNCTION AND ITS SPECIAL CASES

Combining all the above results, assuming that node  $\mathfrak{n}_t$  has children  $\mathfrak{n}_{c_1},\ldots ,\mathfrak{n}_{c_s}$ , we arrive at the following general algorithm for the aggregation rule  $\Phi_t$ :

1. Collect all the  $k$ 'th order activations  $F_{c_1}, \ldots, F_{c_s}$  of the children.  
2. Promote each activation to  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  (Proposition 5).  
3. Stack  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  together into a  $k + 1$  order tensor  $T$  (Proposition 6).  
4. Optionally form the tensor product of  $T$  with  $A \downarrow_{\mathcal{P}_t}$  to get a  $k + 3$  order tensor  $H$  (otherwise just set  $H = T$ ) (Proposition 7).  
5. Contract  $H$  along some number of combinations of dimensions to get  $s$  separate lower order tensors  $Q_{1},\ldots ,Q_{s}$  (Proposition 4).  
6. Mix  $Q_{1}, \ldots, Q_{s}$  with a matrix  $W \in \mathbb{R}^{s' \times s}$  and apply a nonlinearity  $\Upsilon$  to get the final activation of the neuron, which consists of the  $s'$  output tensors

$$
F ^ {(i)} = \Upsilon \bigg [ \sum_ {j = 1} ^ {s} W _ {i, j} Q _ {j} + b _ {i} \mathbb {I} \bigg ] \qquad i = 1, 2, \ldots s ^ {\prime},
$$

where the  $b_{i}$  scalars are bias terms, and  $\mathbb{1}$  is the  $|\mathcal{P}_t| \times \ldots \times |\mathcal{P}_t|$  dimensional all ones tensor.

A few remarks are in order about this general scheme:

1. Since  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  are stacked into a larger tensor and then possibly also multiplied by  $A\downarrow_{\mathcal{P}_t}$ , the general tendency would be for the tensor order to increase at every node, and the corresponding storage requirements to increase exponentially. The purpose of the contractions in Step 5 is to counteract this tendency, and pull the order of the tensors back to some small number, typically 1, 2 or 3.  
2. However, since contractions can be done in many different ways, the number of channels will increase. When the number of input channels is small, this is reasonable, since otherwise the number of learnable weights in the algorithm would be too small. However, if unchecked, this can also become problematic. Fortunately, mixing the channels by  $W$  on Step 6 gives an opportunity to stabilize the number of channels at some value  $s'$ .  
3. In the pseudocode above, for simplicity, the number of input channels is one and the number of output channels is  $s'$ . More realistically, the inputs would also have multiple channels (say,  $s_0$ ) which would be propagated through the algorithm independently up to the mixing stage, making  $W$  an  $s' \times s \times s_0$  dimension tensor (not in the  $P$ -tensor sense!).  
4. The conventional part of the entire algorithm is Step 6, and the only learnable parameters are the entries of the  $W$  matrix (tensor) and the  $b_{i}$  bias terms. These parameters are shared by all nodes in the network and learned in the usual way, by stochastic gradient descent.  
5. Our scheme could be elaborated further while maintaining permutation covariance by, for example taking the tensor product of  $T$  with itself, or by introducing  $A\downarrow_{\mathcal{P}_t}$  in a different way. However, the way that  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  and  $A\downarrow_{\mathcal{P}_t}$  are combined by tensor products is already much more general and expressive than conventional message passing networks.  
6. Our framework admits many design choices, including the choice of the order odf the activations, the choice of contractions, and  $c'$ . However, the overall structure of Steps 1-5 is fully dictated by the covariance constraint on the network.  
7. The final output of the network  $\phi(G) = F_r$  must be permutation invariant. That means that the root node  $\mathfrak{n}_r$  must produce a tuple of zeroth order tensors (calars)  $(F_r^{(1)},\ldots,F_r^{(c)})$ . This is similar to how many other graph representation algorithms compute  $\phi(G)$  by summing the activations at level  $L$  or creating histogram features.

We consider a few special cases to explain how tensor aggregation relates to more conventional message passing rules.

# 5.1.1. ZEROTH ORDER TENSOR AGGREGATION

Constraining both the input tensors  $F_{c_1}, \ldots, F_{c_s}$  and the outputs to be zeroth order tensors, i.e., scalars, and foregoing multiplication by  $A \downarrow_{\mathcal{P}_t}$  greatly simplifies the form of  $\Phi$ . In this case there is no need for promotions, and  $T$  is just the vector  $(F_{c_1}^\ell, \ldots, F_{c_s}^\ell)$ . There is only one way to contract a vector into a scalar, and that is to sum its elements. Therefore, in this case, the entire aggregation

algorithm reduces to the simple formula

$$
F _ {i} = \Upsilon \left(w \sum_ {u = 1} ^ {c} F _ {c _ {u}} + b\right).
$$

For a neural network this is too simplistic. However, it's interesting to note that the Weisfeiler-Lehmann isomorphism test essentially builds on just this formula, with a specific choice of  $\Upsilon$  (Read & Corneil, 1977). If we allow more channels in the inputs and the outputs,  $W$  becomes a matrix, and we recover the simplest form of neural message passing algorithms (Duvenaud et al., 2015).

# 5.1.2. FIRST ORDER TENSOR AGGREGATION

In first order tensor aggregation, assuming that  $|\mathcal{P}_i| = m$ ,  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  are  $m$  dimensional column vectors, and  $T$  is an  $m\times m$  matrix consisting of  $\widetilde{F}_{c_1},\dots,\widetilde{F}_{c_s}$  stacked columnwise. There are two ways of contracting (in our generalized sense) a matrix into a vector: by summing over its rows, or summing over its columns. The second of these choices leads us back to summing over all contributions from the children, while the first is more interesting because it corresponds to summing  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  as vectors individually. In summary, we get an aggregation function that transforms a single input channel to two output channels of the form

$$
F _ {i} ^ {(1)} = \Upsilon \Big [ w _ {1, 1} (T ^ {\top} \mathbf {1}) + w _ {1, 2} (T \mathbf {1}) + b _ {1} \mathbf {1} \Big ], \qquad F _ {i} ^ {(2)} = \Upsilon \Big [ w _ {2, 1} (T ^ {\top} \mathbf {1}) + w _ {2, 2} (T \mathbf {1}) + b _ {2} \mathbf {1} \Big ],
$$

where  $\mathbf{1}$  denotes the  $m$  dimensional all ones vector. Thus, in this layer  $W\in \mathbb{R}^{2\times 2}$ . Unless constrained by  $c^{\prime}$ , in each subsequent layer the number of channels doubles further and these channels can all mix with each other, so  $W^{(2)}\in \mathbb{R}^{4\times 4}$ ,  $W^{(3)}\in \mathbb{R}^{8\times 8}$ , and so on.

# 5.1.3. SECOND ORDER TENSOR AGGREGATION WITHOUT THE ADJACENCY MATRIX

In second order tensor aggregation,  $T$  is a third order  $P$ -tensor, which can be contracted back to second order in three different ways, by projecting it along each of its dimensions. Therefore the outputs will be the three matrices

$$
F ^ {(i)} = \Upsilon \left(w _ {i, 1} T \downarrow_ {1} + w _ {i, 2} T \downarrow_ {2} + w _ {i, 3} T \downarrow_ {3} + b _ {i} \mathbf {1} _ {m \times m}\right) \quad i \in \{1, 2, 3 \},
$$

and the weight matrix is  $W\in \mathbb{R}^{3\times 3}$

# 5.1.4. SECOND ORDER TENSOR AGGREGATION WITH THE ADJACENCY MATRIX

The first nontrivial tensor contraction case occurs when  $\widetilde{F}_{c_1},\ldots ,\widetilde{F}_{c_s}$  are second order tensors, and we multiply with  $A_{\downarrow \mathcal{P}_t}$ , since in that case  $T$  is 5th order, and can be contracted down to second order in a total of 50 different ways:

1. The “ $1 + 1 + 1$ ” case contracts  $T$  in the form  $T_{i_1,i_2,i_3,i_4,i_5}\delta^{i_a_1}\delta^{i_a_2}\delta^{i_a_3}$ , i.e., it projects  $T$  down along 3 of its 5 dimensions. This alone can be done in  $\binom{5}{3}=10$  different ways $^7$  
2. The “ $1 + 2$ ” case contracts  $T$  in the form  $T_{i_1,i_2,i_3,i_4,i_5}\delta^{i_{a_1}}\delta^{i_{a_2},i_{a_3}}$ , i.e., it projects  $T$  along one dimension, and contracts it along two others. This can be done in  $3\binom{5}{3}=30$  ways.  
3. The "3" case is a single 3-fold contraction  $T_{i_1,i_2,i_3,i_4,i_5} \delta^{i_{a_1},i_{a_2},i_{a_3}}$ , which again can be done in  $\binom{5}{3}=10$  different ways.

Clearly, maintaining 50 channels in a message passing architecture is excessive, so in practice it is reasonable to set  $c' \approx 10$ , making  $W \in \mathbb{R}^{10 \times 50}$ .

# 6. EXPERIMENTS

We compared the second order variant of our CCN framework (Section 4.2) to several standard graph learning algorithms on three types of datasets that involve learning the properties of molecules from their structure:

![](images/fc81cc391a0540f1c8d465aaa10d14151789b424342ad55106ddd937a3efb62d.jpg)  
FIGURE 6. The activations of each neighbor are stacked into a tensor  $T$  which is tensor multiplied by the restriction of the adjacency matrix, and then reduced in different ways.

1. The Harvard Clean Energy Project (Hachmann et al., 2011), consisting of 2.3 million organic compounds that are candidates for use in solar cells. The regression target in this case is Power Conversion Efficiency (PCE). Due to time constraints, instead of using the entire dataset, the experiments were ran on a random subset of 50,000 molecules.  
2. QM9, which is a dataset of all 133k organic molecules with up to nine atoms (C,H,O,N and F) out of the GDB-17 universe of molecules. Each molecule has 13 target properties to predict. The dataset does contain spatial information relating to the atomic configurations, but we only used the chemical graph and atom node labels. For our experiments we normalized each target variable to have mean 0 and standard deviation 1.  
3. Graph kernels datasets, specifically (a) MUTAG, which is a dataset of 188 mutagenic aromatic and heteroaromatic compounds (Debnat et al., 1991); (b) PTC, which consists of 344 chemical compounds that have been tested for positive or negative toxicity in lab rats (Toivonen et al., 2003); (c) NCI1 and NCI109, which have 4110 and 4127 compounds respectively, each screened for activity against small cell lung cancer and ovarian cancer lines (Wale et al., 2008).

In the case of HCEP, we compared CCN to lasso, ridge regression, random forests, gradient boosted trees, optimal assignment Wesifeiler-Lehman graph kernel (Kriege et al., 2016) (WL), neural graph fingerprints (Duvenaud et al., 2015), and the "patchy-SAN" convolutional type algorithm from (Niepert et al., 2016) (referred to as PSCN). For the first four of these baseline methods, we created simple feature vectors from each molecule: the number of bonds of each type (i.e. number of H-H bonds, number of C-O bonds, etc) and the number of atoms of each type. Molecular graph fingerprints use atom labels of each vertex as base features. For ridge regression and lasso, we cross validated over  $\lambda$ . For random forests and gradient boosted trees, we used 400 trees, and cross validated over max depth, minimum samples for a leaf, minimum samples to split a node, and learning rate (for GBT). For neural graph fingerprints, we used 2 layers and a hidden layer size of 10. In PSCN, we used a patch size of 10 with two convolutional layers and a dense layer on top as described in their paper.

TABLE 1. HCEP regression results  

<table><tr><td></td><td>Test MAE</td><td>Test RMSE</td></tr><tr><td>Lasso</td><td>0.867</td><td>1.437</td></tr><tr><td>Ridge regression</td><td>0.854</td><td>1.376</td></tr><tr><td>Random forest</td><td>1.004</td><td>1.799</td></tr><tr><td>Gradient boosted trees</td><td>0.704</td><td>1.005</td></tr><tr><td>WL graph kernel</td><td>0.805</td><td>1.096</td></tr><tr><td>Neural graph fingerprints</td><td>0.851</td><td>1.177</td></tr><tr><td>PSCN (k=10)</td><td>0.718</td><td>0.973</td></tr><tr><td>Second order CCN (our method)</td><td>0.340</td><td>0.449</td></tr></table>

For the graph kernels datasets, we compare against graph kernel results as reported in (Kondor & Pan, 2016) (which computed kernel matrices using the Weisfeiler-Lehman, Weisfeiler-edge, shortest paths, graphlets and multiscale Laplacian graph kernels and used a C-SVM on top), Neural graph fingerprints (with 2 levels and a hidden size of 10) and PSCN. For QM9, we compared against the Weisfeiler-Lehman graph kernel (with C-SVM on top), neural graph fingerprints, and PSCN. The settings for NGF and PSCN are as described for HCEP.

For our own method, second order CCN, we initialized the base features of each vertex with computed histogram alignment kernel features (Kriege et al., 2016) of depth up to 10. Each vertex receives a base label  $l_{i} = \mathrm{concat}_{j=1}^{10} H_{j}(i)$  where  $H_{j}(i) \in \mathbb{R}^{d}$  (with  $d$  being the total number of distinct discrete node labels) is the vector of relative frequencies of each label for the set of vertices at distance equal to  $j$  from vertex  $i$ . We used two levels and doubled the intermediate channel size at each level. For computational efficiency, we only used 10 contractions as described in section 5.1.4 instead of the full 50 contractions.

In each experiment we used  $80\%$  of the dataset for training,  $10\%$  for validation, and evaluated on the remaining  $10\%$  test set. For the kernel datasets we performed the experiments on 10 separate training/validation/test stratified splits and averaged the resulting classification accuracies. We always used stochastic gradient descent with momentum 0.9. Our initial learning rate was set to 0.001 after experimenting on a held out set. The learning rate decayed linearly after each step towards a minimum of  $10^{-6}$ .

# 6.1.DISCUSSION

On the subsampled HCEP dataset, CCN outperforms all other methods by a very large margin. For the graph kernels datasets, SVM with the Weisfeiler-Lehman kernels achieve the highest accuracy on NCI1 and NCI109, while CCN wins on MUTAG and PTC. Perhaps this poor performance is to be expected, since the datasets are small and neural network approaches usually require tens of thousands of training examples at minimum to be effective. Indeed, neural graph fingerprints and PSCN also perform poorly compared to the Weisfeiler-Lehman kernels.

In the QM9 experiments, CCN beats the three other algorithms in both mean absolute error and root mean squared error. It should be noted that (Gilmer et al., 2017) obtained stronger results on QM9, but we cannot properly compare our results with theirs because our experiments only use the adjacency matrices and atom labels of each node, while theirs include comprehensive chemical features that better inform the target quantum properties.

# 7. CONCLUSIONS

We have presented a general framework called covariant compositional networks (CCNs) for constructing covariant graph neural networks, which encompasses other message passing approaches as special cases, but takes a more general and principled approach to ensuring covariance with respect to permutations. Experimental results on several benchmark datasets show that CCNs can outperform other state-of-the-art algorithms.

TABLE 2. Kernel Datasets Classification results (accuracy +/- standard deviation)  

<table><tr><td></td><td>MUTAG</td><td>PTC</td><td>NCI1</td><td>NCI109</td></tr><tr><td>WL</td><td>84.50 ± 2.16</td><td>59.97 ± 1.60</td><td>84.76 ± 0.32</td><td>85.12 ± 0.29</td></tr><tr><td>WL-edge</td><td>82.94 ± 2.33</td><td>60.18 ± 2.19</td><td>84.65 ± 0.25</td><td>85.32 ± 0.34</td></tr><tr><td>SP</td><td>85.50 ± 2.50</td><td>59.53 ± 1.71</td><td>73.61 ± 0.36</td><td>73.23 ± 0.26</td></tr><tr><td>Graphlet</td><td>82.44 ± 1.29</td><td>55.88 ± 0.31</td><td>62.40 ± 0.27</td><td>62.35 ± 0.28</td></tr><tr><td>p-RW</td><td>80.33 ± 1.35</td><td>59.85 ± 0.95</td><td>TIMED OUT</td><td>TIMED OUT</td></tr><tr><td>MLG</td><td>87.94 ± 1.61</td><td>63.26 ± 1.48</td><td>81.75 ± 0.24</td><td>81.31 ± 0.22</td></tr><tr><td>PSCN k = 10 (Niepert et al.)</td><td>88.95 ± 4.37</td><td>62.29 ± 5.68</td><td>76.34 ± 1.68</td><td>N/A</td></tr><tr><td>Neural graph fingerprints</td><td>89.00 ± 7.00</td><td>57.85 ± 3.36</td><td>62.21 ± 4.72</td><td>56.11 ± 4.31</td></tr><tr><td>Second order CCN (our method)</td><td>91.64 ± 7.24</td><td>70.62 ± 7.04</td><td>76.27 ± 4.13</td><td>75.54 ± 3.36</td></tr></table>

TABLE 3. QM9 regression results (MAE)  

<table><tr><td></td><td>WLGK</td><td>NGF</td><td>PSCN (k=10)</td><td>Second order CCN</td></tr><tr><td>alpha</td><td>0.46</td><td>0.43</td><td>0.20</td><td>0.16</td></tr><tr><td>Cv</td><td>0.59</td><td>0.47</td><td>0.27</td><td>0.23</td></tr><tr><td>G</td><td>0.51</td><td>0.46</td><td>0.33</td><td>0.29</td></tr><tr><td>gap</td><td>0.72</td><td>0.67</td><td>0.60</td><td>0.54</td></tr><tr><td>H</td><td>0.52</td><td>0.47</td><td>0.34</td><td>0.30</td></tr><tr><td>HOMO</td><td>0.64</td><td>0.58</td><td>0.51</td><td>0.39</td></tr><tr><td>LUMO</td><td>0.70</td><td>0.65</td><td>0.59</td><td>0.53</td></tr><tr><td>mu</td><td>0.69</td><td>0.63</td><td>0.54</td><td>0.48</td></tr><tr><td>omega1</td><td>0.72</td><td>0.63</td><td>0.57</td><td>0.45</td></tr><tr><td>R2</td><td>0.55</td><td>0.49</td><td>0.22</td><td>0.19</td></tr><tr><td>U</td><td>0.52</td><td>0.47</td><td>0.34</td><td>0.29</td></tr><tr><td>U0</td><td>0.52</td><td>0.47</td><td>0.34</td><td>0.29</td></tr><tr><td>ZPVE</td><td>0.57</td><td>0.51</td><td>0.43</td><td>0.39</td></tr></table>

TABLE 4. QM9 regression results (RMSE)  

<table><tr><td></td><td>WLGK</td><td>NGF</td><td>PSCN (k=10)</td><td>Second order CCN</td></tr><tr><td>alpha</td><td>0.68</td><td>0.65</td><td>0.31</td><td>0.26</td></tr><tr><td>Cv</td><td>0.78</td><td>0.65</td><td>0.34</td><td>0.30</td></tr><tr><td>G</td><td>0.67</td><td>0.62</td><td>0.43</td><td>0.38</td></tr><tr><td>gap</td><td>0.86</td><td>0.82</td><td>0.75</td><td>0.69</td></tr><tr><td>H</td><td>0.68</td><td>0.62</td><td>0.44</td><td>0.40</td></tr><tr><td>HOMO</td><td>0.91</td><td>0.81</td><td>0.70</td><td>0.55</td></tr><tr><td>LUMO</td><td>0.84</td><td>0.79</td><td>0.73</td><td>0.68</td></tr><tr><td>mu</td><td>0.92</td><td>0.87</td><td>0.75</td><td>0.67</td></tr><tr><td>omega1</td><td>0.84</td><td>0.77</td><td>0.73</td><td>0.65</td></tr><tr><td>R2</td><td>0.81</td><td>0.71</td><td>0.31</td><td>0.27</td></tr><tr><td>U</td><td>0.67</td><td>0.62</td><td>0.44</td><td>0.40</td></tr><tr><td>U0</td><td>0.67</td><td>0.62</td><td>0.44</td><td>0.39</td></tr><tr><td>ZPVE</td><td>0.72</td><td>0.66</td><td>0.55</td><td>0.51</td></tr></table>

# REFERENCES

P. Battaglia, R. Pascanu, M. Lai, D. J. Rezende, and K. Kavukcuoglu. Interaction networks for learning about objects, relations and physics. In Advances in neural information processing systems, pp. 4502-4510, 2016.  
K. M. Borgwardt and H. P. Kriegel. Shortest-path kernels on graphs. In Proceedings of the 5th IEEE International Conference on Data Mining(ICDM) 2005), 27-30 November 2005, Houston, Texas, USA, pp. 74-81, 2005.  
J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun. Spectral networks and locally connected networks on graphs. In Proceedings of International Conference on Learning Representations, 2014.

J. Y. Cai, M. Furer, and N. Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica, 12:389-410, December 1992.  
T. Cohen and M. Welling. Group equivariant convolutional networks. In Proceedings of the International Conference on Machine Learning, pp. 2990-2999, 2016a.  
T. Cohen and M. Welling. Steerable CNNs. arXiv preprint arXiv:1612.08498, 2016b.  
A. K. Debnat, R. L. Lopez de Compadre, G. Debnath, A. J. Shusterman, and C. Hansch. Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. Correlation with molecular orbital energies and hydrophobicity. J Med Chem, 34:786-97, 1991.  
M. Defferrard, X. Bresson, and P. Vandergheynst. Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in Neural Information Processing Systems, 2016.  
D. K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-Guzik, and R. P. Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems, pp. 2224-2232, 2015.  
P. F. Felzenszwalb and D. P. Huttenlocher. Pictorial structures for object recognition. International Journal of Computer Vision, 61:55-71, 2005.  
P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained part-based models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32:541-551, 2010.  
A. Feragen, N. Kasenburg, J. Peterson, M. de Bruijne, and K. M. Borgwardt. Scalable kernels for graphs with continuous attributes. In Advances in Neural Information Processing Systems, 2013.  
M. Fischler and R. Elschlager. The representation and matching of pictorial structures. IEEE Transactions on Computer, C-22:67-92, 1973.  
W. T. Freeman and E. H. Adelson. The design and use of steerable filters. IEEE Transactions on Pattern Analysis and Machine Intelligence, 13:891-906, September 1991.  
T. Gartner. Exponential and geometric kernels for graphs. In NIPS 2002 workshop on unreal data, volume Principles of modeling nonvectorial data, 2002.  
J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212, 2017.  
J. Hachmann, R. Olivares-Amaya, S. Atahan-Evrenk, C. Amador-Bedolla, R. S. Snchez-Carrera, A. Gold-Parker, L. Vogt, A. M. Brockway, and A. Aspuru-Guzik. The Harvard clean energy project: Large-scale computational screening and design of organic photovoltaics on the world community grid. The Journal of Physical Chemistry Letters, 2011.  
M. Henaff, J. Bruna, and Y. LeCun. Deep convolutional networks on graph-structured data. arXiv preprint arXiv:1506.05163, June 2015.  
S. Kearns, K. McCloskey, M. Brendl, V. Pande, and P. Riley. Molecular graph convolutions: moving beyond fingerprints. Journal of Computer-Aided Molecular Design, 30:595-608, 2016.  
T. N. Kipf and M. Welling. Semi-supervised classification with graph convolutional networks. In Proceedings of International Conference on Learning Representations, 2017.  
R. Kondor and K. M. Borgwardt. The skew spectrum of graphs. In Proceedings of the International Conference on Machine Learning, pp. 496-503, 2008.  
R. Kondor and H. Pan. The multiscale Laplacian graph kernel. In Neural Information Processing Systems, pp. 2982-2990, 2016.  
N. M. Kriege, P. Giscard, and R. Wilson. On valid optimal assignment kernels and applications to graph classification. Advances in Neural Information Processing Systems 29, 2016.  
A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pp. 1097-1105, 2012.  
Y. LeCun, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, pp. 2278-2324, 1998.  
Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493, 2015.  
R. Manduchi, P. Perona, and D. Shy. Efficient deformable filter banks. IEEE Transactions on Signal Processing, 46:1168-1173, April 1998.  
M. Neumann, R. Garnett, C. Baukhage, and K. Kersting. Propagation kernels: efficient graph kernels from propagated information. In Machine Learning, 2016.  
M. Niepert, M. Ahmed, and K. Kutzkov. Learning convolutional neural networks for graphs. In Proceedings of the International Conference on Machine Learning, 2016.  
Y. Ohta, T. Kanade, and T. Sakai. An analysis system for scenes containing objects with substructures. In Proceedings of 4th International Joint Conference on Pattern Recognition, pp. 752-754, 1978.

P. Perona. Deformable kernels for early vision. IEEE Transactions on Pattern Analysis and Machine Intelligence, 17:488-499, May 1995.  
R. C. Read and D. G. Corneil. The graph isomorphism disease. Journal of Graph Theory, 1:339-363, 1977.  
B. E. Sagan. The Symmetric Group. Graduate Texts in Mathematics. Springer, 2001.  
F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20:61-80, 2009.  
K. T. Schütt, Kristof T., F. Arbabzadah, S. Chmiela, K. R. Müller, and A. Tkatchenko. Quantum-chemical insights from deep tensor neural networks. Nature communications, 2017.  
N. Shervashidze, S. V. N. Vishwanathan, T. Petri, K. M., and K. M. Borgwardt. Efficient graphlet kernels for large graph comparison. In Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, AISTATS, pp. 488-495, 2009.  
N. Shervashidze, P. Schweitzer, E. J. van Leeuwan, K. Mehlhorn, and K. M. Borgwardt. Weisfeiler-Lehman graph kernels. Journal of Machine Learning Research, 12:2539-2561, 2011.  
E. P. Simoncelli, W. T. Freeman, E. H. Adelson, and D. J. Heeger. Shiftable multiscale transforms. IEEE Transactions on Information Theory, 38:587-607, March 1992.  
P. C. Teo and Y. Hel-Or. Lie generators for computing steerable functions. Pattern Recognition Letters, 16:7-17, October 1998.  
H. Toivonen, A. Srinivasan, R. D. King, S. Kramer, and C. Helma. Statistical evaluation of the predictive toxicology challenge. Bioinformatics, pp. 1183-1193, 2003.  
Z. W. Tu, X. R. Chen, A. L. Yuille, and S. C. Zhu. Image parsing: Unifying segmentation, detection, and recognition. International Journal of Computer Vision, 63:113-140, 2005.  
S. V. N. Vishwanathan, N. N. Schraudolf, R. Kondor, and K. M. Bogwardt. Graph kernels. Journal of Machine Learning Research, 11:1201-1242, 2010.  
N. Wale, I. A. Watson, and G. Karypis. Comparison of descriptor spaces for chemical compound retrieval and classification. Knowledge and Information Systems, pp. 347-375, 2008.  
B. Weisfeiler and A. A. Lehman. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Techniqueskaya Informatsia*, 9, 1968.  
D. E. Worrall, S. Garbin, D. Turmukhambetov, and G. J. Brostow. Harmonic networks: Deep translation and rotation equivariance. Technical report, 2017.  
S. Zhu and D. Mumford. A stochastic grammar of images. Foundations and Trends in Computer Graphics and Vision, 2:259-362, 2006.
