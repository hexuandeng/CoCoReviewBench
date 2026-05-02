# LOGIC AND THE 2-SIMPLICIAL TRANSFORMER

Anonymous authors

Paper under double-blind review

# ABSTRACT

We introduce the 2-simplicial Transformer, an extension of the Transformer which includes a form of higher-dimensional attention generalising the dot-product attention, and uses this attention to update entity representations with tensor products of value vectors. We show that this architecture is a useful inductive bias for logical reasoning in the context of deep reinforcement learning.

# 1 INTRODUCTION

Deep learning has grown to incorporate a range of differentiable algorithms for computing with learned representations. The most successful examples of such representations, those learned by convolutional neural networks, are structured by the scale and translational symmetries of the underlying space (e.g. a two-dimensional Euclidean space for images). It has been suggested that in humans the ability to make rich inferences based on abstract reasoning is rooted in the same neural mechanisms underlying relational reasoning in space (Constantinescu et al., 2016; Epstein et al., 2017; Behrens et al., 2018; Bellmund et al., 2018) and more specifically that abstract reasoning is facilitated by the learning of structural representations which serve to organise other learned representations in the same way that space organises the representations that enable spatial navigation (Whittington et al., 2018; Liu et al., 2019). This raises a natural question: are there any ideas from mathematics that might be useful in designing general inductive biases for learning such structural representations?

As a motivating example we take the recent progress on natural language tasks based on the Transformer architecture (Vaswani et al., 2017) which simultaneously learns to represent both entities (typically words) and relations between entities (for instance the relation between "cat" and "he" in the sentence "There was a cat and he liked to sleep"). These representations of relations take the form of query and key vectors governing the passing of messages between entities; messages update entity representations over several rounds of computation until the final representations reflect not just the meaning of words but also their context in a sentence. There is some evidence that the geometry of these final representations serve to organise word representations in a syntax tree, which could be seen as the appropriate analogue to two-dimensional space in the context of language (Hewitt & Manning, 2019).

The Transformer may therefore be viewed as an inductive bias for learning structural representations which are graphs, with entities as vertices and relations as edges. While a graph is a discrete mathematical object, there is a naturally associated topological space which is obtained by gluing 1-simplices (copies of the unit interval) indexed by edges along 0-simplices (points) indexed by vertices. There is a general mathematical notion of a simplicial set which is a discrete structure containing a set of  $n$ -simplices for all  $n \geq 0$  together with an encoding of the incidence relations between these simplices. Associated to each simplicial set is a topological space, obtained by gluing together vertices, edges, triangles (2-simplices), tetrahedrons (3-simplices), and so on, according to the instructions contained in the simplicial set. Following the aforementioned works in neuroscience (Constantinescu et al., 2016; Epstein et al., 2017; Behrens et al., 2018; Bellmund et al., 2018; Whittington et al., 2018; Liu et al., 2019) and their emphasis on spatial structure, it is natural to ask if a simplicial inductive bias for learning structural representations can facilitate abstract reasoning.

With this motivation, we begin in this paper an investigation of simplicial inductive biases for abstract reasoning in neural networks, by giving a simple method for incorporating 2-simplices (which relate three entities) into the existing Transformer architecture. We call this the 2-simplicial Transformer block. It has been established in recent work (Santoro et al., 2017; Zambaldi et al., 2019;

Vinyals et al., 2019) that relational inductive biases are useful for solving problems that draw on abstract reasoning in humans. In Section 5 we show that when embedded in a deep reinforcement learning agent our 2-simplicial Transformer block confers an advantage over the ordinary Transformer block in an environment with logical structure, and on this basis we argue that further investigation of simplicial inductive biases is warranted.

The environment in our reinforcement learning problem is a variant of the BoxWorld environment from (Zambaldi et al., 2019). The original BoxWorld is played on a rectangular grid populated by keys and locked boxes of varying colours, with the goal being to open the box containing the "Gem" by following the solution path in the presence of distractor branches. In our variant of the BoxWorld environment, bridge BoxWorld, each episode now requires two keys to obtain the Gem and beginning at each loose key is a solution path leading to one of the keys required to open the box containing the Gem. The eponymous bridges allow the player to cross between solution paths, thereby rendering the puzzle unsolvable.

The design of the BoxWorld environment was intended to stress the planning and reasoning components of an agent's policy (Zambaldi et al., 2019, p.2) and for this reason it is the underlying logical structure of the environment (rather than its representation in terms of coloured keys) that is of central importance. To explain this logical structure we introduce the following notation: given a colour  $c$ , we use  $C$  to stand for the proposition that a key of this colour is obtainable. Each episode expresses its own set of basic facts, or axioms, about obtainability. For instance, a loose key of colour  $c$  gives  $C$  as an axiom, and a locked box requiring a key of colour  $c$  in order to obtain a key of colour  $d$  gives an axiom that at first glance appears to be the implication  $C \longrightarrow D$  of classical logic. However, since a key may only be used once, this is actually incorrect; instead the logical structure of this situation is captured by the linear implication  $C \rightharpoonup D$  of linear logic (Girard, 1987). With this understood, each episode of the original BoxWorld provides in visual form a set of axioms  $\Gamma$  such that a strategy for obtaining the Gem is equivalent to a proof of  $\Gamma \vdash \mathbb{G}$  in intuitionistic linear logic, where  $\mathbb{G}$  stands for the proposition that the Gem is obtainable. There is a general correspondence in logic between strategies and proofs which we recall in Appendix I.

To describe the logical structure of bridge BoxWorld we need to encode the fact that two keys (say a green key and a blue key) are required to obtain the Gem. Once again, it is the linear conjunction  $\otimes$  of linear logic (also called the tensor product) rather than the conjunction of classical logic that properly captures the semantics. The axioms  $\Gamma$  encoded in an episode of bridge BoxWorld contain a single formula of the form  $X_{1} \otimes X_{2} \dashcirc \mathbb{G}$  where  $x_{1}, x_{2}$  are the colours of the keys on the Gem, and again a strategy is equivalent to a proof of  $\Gamma \vdash \mathbb{G}$ . In conclusion, the logical structure of the original BoxWorld consists of a fragment of linear logic containing only the connective  $\dashcirc$ , while bridge BoxWorld captures a slightly larger fragment containing  $\dashcirc$  and  $\otimes$ . The problem faced by the agent is to learn, purely through interaction, this underlying logical structure.

The architecture of our deep reinforcement learning agent largely follows (Zambaldi et al., 2019) and the details are given in Section 4. The key difference between our simplicial agent and the relational agent of (Zambaldi et al., 2019) is that in place of a standard Transformer block we use a 2-simplicial Transformer block. Our use of tensor products of value vectors is inspired by the semantics of linear logic in vector spaces (Girard, 1987; Mellies, 2009; Clift & Murfet, 2017; Wallbridge, 2018) in which an algorithm with multiple inputs computes on the tensor product of those inputs, but this is an old idea in natural language processing, used in models including the second-order RNN (Giles et al., 1989; Pollack, 1991; Goudreau et al., 1994; Giles et al., 1991), multiplicative RNN (Sutskever et al., 2011; Irsoy & Cardie, 2015), Neural Tensor Network (Socher et al., 2013) and the factored 3-way Restricted Boltzmann Machine (Ranzato et al., 2010), see Appendix A. More recently tensors have been used to model predicates in a number of neural network architectures aimed at logical reasoning (Serafini & Garcez, 2016; Dong et al., 2019). The main novelty in our model lies in the introduction of the 2-simplicial attention, which allows these ideas to be incorporated into the Transformer architecture.

# 2 2-SIMPLICIAL TRANSFORMER

In this section we first review the definition of the ordinary Transformer block and then explain the 2-simplicial Transformer block. We distinguish between the Transformer architecture which contains a word embedding layer, an encoder and a decoder (Vaswani et al., 2017), and the Transformer

block which is the sub-model of the encoder that is repeated. The fundamental idea, of propagating information between nodes using weights that depend on the dot product of vectors associated to those nodes, comes ultimately from statistical mechanics via the Hopfield network (Appendix B).

The ordinary and 2-simplicial Transformer blocks define operators on sequences  $e_1, \ldots, e_N$  of entity representations. Strictly speaking the entities are indices  $1 \leq i \leq N$  but we sometimes identify the entity  $i$  with its representation  $e_i$ . The space of entity representations is denoted  $V$ , while the space of query, key and value vectors is denoted  $H$ . We use only the vector space structure on  $V$ , but  $H = \mathbb{R}^d$  is an inner product space with the usual dot product pairing  $(h, h') \mapsto h \cdot h'$  and in defining the 2-simplicial Transformer block we will use additional algebraic structure on  $H$ , including the "multiplication" tensor  $B: H \otimes H \longrightarrow H$  of (10) (used to propagate tensor products of value vectors) and the Clifford algebra of  $H$  (used to define the 2-simplicial attention).

In the first step of the standard Transformer block we generate from each entity  $e_i$  a tuple of vectors via a learned linear transformation  $E: V \longrightarrow H^{\oplus 3}$ . These vectors are referred to respectively as query, key and value vectors and we write

$$
\left(q _ {i}, k _ {i}, v _ {i}\right) = E \left(e _ {i}\right). \tag {1}
$$

Stated differently,  $q_{i} = W^{Q}e_{i}, k_{i} = W^{K}e_{i}, v_{i} = W^{V}e_{i}$  for weight matrices  $W^{Q}, W^{K}, W^{V}$ . In the second step we compute a refined value vector for each entity

$$
v _ {i} ^ {\prime} = \sum_ {j = 1} ^ {N} \frac {e ^ {q _ {i} \cdot k _ {j}}}{\sum_ {s = 1} ^ {N} e ^ {q _ {i} \cdot k _ {s}}} v _ {j} = \sum_ {j = 1} ^ {N} \operatorname {s o f t m a x} \left(q _ {i} \cdot k _ {1}, \dots , q _ {i} \cdot k _ {N}\right) _ {j} v _ {j}. \tag {2}
$$

Finally, the new entity representation  $e_i'$  is computed by the application of a feedforward network  $g_{\theta}$ , layer normalisation and a skip connection

$$
e _ {i} ^ {\prime} = \text {L a y e r N o r m} \left(g _ {\theta} \left(v _ {i} ^ {\prime}\right) + e _ {i}\right). \tag {3}
$$

Remark 2.1. In the introduction we referred to the idea that a Transformer model learns representations of relations. To be more precise, these representations are heads, each of which determines an independent set of transformations  $W^{Q}, W^{K}, W^{V}$  which extract queries, keys and values from entities. Thus a head determines not only which entities are related (via  $W^{Q}, W^{K}$ ) but also what information to transmit between them (via  $W^{V}$ ). In multiple-head attention with  $K$  heads, there are  $K$  channels along which to propagate information between every pair of entities, each of dimension  $\dim(H)/K$ . More precisely, we choose a decomposition  $H = H_{1} \oplus \dots \oplus H_{K}$  so that

$$
E: V \longrightarrow \bigoplus_ {u = 1} ^ {K} \left(H _ {u} ^ {\oplus 3}\right)
$$

and write

$$
\left(q _ {i, (1)}, k _ {i, (1)}, v _ {i, (1)}, \dots , q _ {i, (K)}, k _ {i, (K)}, v _ {i, (K)}\right) = E \left(e _ {i}\right).
$$

To compute the output of the attention, we take a direct sum of the value vectors propagated along every one of these  $K$  channels, as in the formula

$$
e _ {i} ^ {\prime} = \operatorname {L a y e r N o r m} \left(g _ {\theta} \left[ \bigoplus_ {u = 1} ^ {K} \sum_ {j = 1} ^ {N} \operatorname {s o f t m a x} \left(q _ {i, (u)} \cdot k _ {1, (u)}, \dots , q _ {i, (u)} \cdot k _ {N, (u)}\right) _ {j} v _ {j, (u)} \right] + e _ {i}\right). \tag {4}
$$

In combinatorial topology the canonical one-dimensional object is the 1-simplex (or edge)  $j \longrightarrow i$ . Since the standard Transformer model learns representations of relations, we refer to this form of attention as 1-simplicial attention. The canonical two-dimensional object is the 2-simplex (or triangle) which we may represent diagrammatically in terms of indices  $i, j, k$  as

![](images/edec8c999f9944223e91a029c3590dcf2c9a79290fa6cd6f586b96f4f3ee65c0.jpg)

In the 2-simplicial Transformer block, in addition to the 1-simplicial contribution, each entity  $e_i$  is updated as a function of pairs of entities  $e_j, e_k$  using the tensor product of value vectors  $u_j \otimes u_k$

and a probability distribution derived from a scalar triple product  $\langle p_i, l_j^1, l_k^2 \rangle$  in place of the scalar product  $q_i \cdot k_j$ . This means that we associate to each entity  $e_i$  a four-tuple of vectors via a learned linear transformation  $E: V \longrightarrow H^{\oplus 4}$ , denoted

$$
\left(p _ {i}, l _ {i} ^ {1}, l _ {i} ^ {2}, u _ {i}\right) = E \left(e _ {i}\right). \tag {6}
$$

We still refer to  $p_i$  as the query,  $l_i^1, l_i^2$  as the keys and  $u_i$  as the value. Stated differently,  $p_i = W^P e_i, l_i^1 = W^{L_1} e_i, l_i^2 = W^{L_2} e_i$  and  $u_i = W^U e_i$  for weight matrices  $W^P, W^{L_1}, W^{L_2}, W^U$ .

Definition 2.2. The unsigned scalar triple product of  $a, b, c \in H$  is

$$
\langle a, b, c \rangle = \| (a \cdot b) c - (a \cdot c) b + (b \cdot c) a \| \tag {7}
$$

whose square is a polynomial in the pairwise dot products

$$
\langle a, b, c \rangle^ {2} = (a \cdot b) ^ {2} (c \cdot c) + (b \cdot c) ^ {2} (a \cdot a) + (a \cdot c) ^ {2} (b \cdot b) - 2 (a \cdot b) (a \cdot c) (b \cdot c). \tag {8}
$$

This scalar triple product has a simple geometric interpretation in terms of the volume of the tetrahedron with vertices  $0, a, b, c$ . To explain, recall that the triangle spanned by two unit vectors  $a, b$  in  $\mathbb{R}^2$  has an area  $A$  which can be written in terms of the dot product of  $a$  and  $b$ . In three dimensions, the analogous formula involves the volume  $V$  of the tetrahedron with vertices given by unit vectors  $a, b, c$ , and the scalar triple product as shown in Figure 1.

![](images/f012d7a7f3afb4cd2804e4f69d133527ba1579dab8458af6e3fb1198ab7205ce.jpg)  
$(a \cdot b)^2 = 1 - 4A^2$

![](images/2a7d0957b17c0173393632bd59fa22f5a4e31dc7b9f75143d5316c9a71785269.jpg)  
$\langle a,b,c\rangle^2 = 1 - 36V^2$  
Figure 1: The geometry of 1- and 2-simplicial attention. Left: the dot product in terms of the area  $A$  in  $\mathbb{R}^2$ . Right: the triple product in terms of the volume  $V$  in  $\mathbb{R}^3$ .

In general, given nonzero vectors  $a,b,c$  let  $\hat{a},\hat{b},\hat{c}$  denote unit vectors in the same directions. Then we can by Lemma C.10(v) factor out the length in the scalar triple product

$$
\langle a, b, c \rangle = \| a \| \| b \| \| c \| \langle \hat {a}, \hat {b}, \hat {c} \rangle \tag {9}
$$

so that a general scalar triple product can be understood in terms of the vector norms and configurations of three points on the 2-sphere. One standard approach to calculating volumes of such tetrahedrons is the cross product which is only defined in three dimensions. Since the space of representations  $H$  is high dimensional the natural framework for the triple scalar product  $\langle a, b, c \rangle$  is instead the Clifford algebra of  $H$  (see Appendix C).

For present purposes, we need to know that  $\langle a,b,c\rangle$  attains its minimum value (which is zero) when  $a,b,c$  are pairwise orthogonal, and attains its maximum value (which is  $\| a\| \| b\| \| c\|$ ) if and only if  $\{a,b,c\}$  is linearly dependent (Lemma C.10). Using the number  $\langle p_i,l_j^1,l_k^2\rangle$  as a measure of the degree to which entity  $i$  is attending to  $(j,k)$ , or put differently, the degree to which the network predicts the existence of a 2-simplex  $(i,j,k)$ , the update rule for the entities when using purely 2-simplicial attention is

$$
v _ {i} ^ {\prime} = \sum_ {j, k = 1} ^ {N} \frac {e ^ {\langle p _ {i} , l _ {j} ^ {1} , l _ {k} ^ {2} \rangle}}{\sum_ {s , t = 1} ^ {N} e ^ {\langle p _ {i} , l _ {s} ^ {1} , l _ {t} ^ {2} \rangle}} B \left(u _ {j} \otimes u _ {k}\right) \tag {10}
$$

where  $B:H\otimes H\longrightarrow H$  is a learned linear transformation. Although we do not impose any further constraints, the motivation here is to equip  $H$  with the structure of an algebra; in this respect we model conjunction by multiplication, an idea going back to Boole (Boole, 1847).

We compute multiple-head 2-simplicial attention in the same way as in the 1-simplicial case. To combine 1-simplicial heads (that is, ordinary Transformer heads) and 2-simplicial heads we use

separate inner product spaces  $H^1, H^2$  for each simplicial dimension, so that there are learned linear transformations  $E^1: V \longrightarrow (H^1)^{\oplus 3}, E^2: V \longrightarrow (H^2)^{\oplus 4}$  and the queries, keys and values are extracted from an entity  $e_i$  according to

$$
\left(q _ {i}, k _ {i}, v _ {i}\right) = E ^ {1} \left(e _ {i}\right),
$$

$$
(p _ {i}, l _ {i} ^ {1}, l _ {i} ^ {2}, u _ {i}) = E ^ {2} (e _ {i}).
$$

The update rule (for a single head in each simplicial dimension) is then:

$$
v _ {i} ^ {\prime} = \left\{\sum_ {j = 1} ^ {N} \frac {e ^ {q _ {i} \cdot k _ {j}}}{\sum_ {s = 1} ^ {N} e ^ {q _ {i} \cdot k _ {s}}} v _ {j} \right\} \oplus \text {L a y e r N o r m} \left\{\sum_ {j, k = 1} ^ {N} \frac {e ^ {\langle p _ {i} , l _ {j} ^ {1} , l _ {k} ^ {2} \rangle}}{\sum_ {s , t = 1} ^ {N} e ^ {\langle p _ {i} , l _ {s} ^ {1} , l _ {t} ^ {2} \rangle}} B (u _ {j} \otimes u _ {k}) \right\}, \tag {11}
$$

$$
e _ {i} ^ {\prime} = \text {L a y e r N o r m} \left(g _ {\theta} \left(v _ {i} ^ {\prime}\right) + e _ {i}\right). \tag {12}
$$

If there are  $K_{1}$  heads of 1-simplicial attention and  $K_{2}$  heads of 2-simplicial attention, then (11) is modified in the obvious way using  $H^{1} = \bigoplus_{u = 1}^{K_{1}}H_{u}^{1}$  and  $H^{2} = \bigoplus_{u = 1}^{K_{2}}H_{u}^{2}$ .

Remark 2.3. Without the additional layer normalisation on the output of the 2-simplicial attention we find that training is unstable. The natural explanation is that these outputs are constructed from polynomials of higher degree than the 1-simplicial attention, and thus computational paths that go through the 2-simplicial attention will be more vulnerable to exploding or vanishing gradients.

The time complexity of 1-simplicial attention as a function of the number of entities is  $O(N^2)$  while the time complexity of 2-simplicial attention is  $O(N^3)$  since we have to calculate the attention for every triple  $(i,j,k)$  of entities. For this reason we consider only triples  $(i,j,k)$  where the base of the 2-simplex  $(j,k)$  is taken from a set of pairs predicted by the ordinary attention, which we view as the primary locus of computation. More precisely, we introduce in addition to the  $N$  entities (now referred to as standard entities) a set of  $M$  virtual entities  $e_{N+1},\ldots,e_{N+M}$ . These virtual entities serve as a "scratch pad" onto which the iterated ordinary attention can write representations, and we restrict  $j,k$  to lie in the range  $N < j,k \leq N + M$  so that only value vectors obtained from virtual entities are propagated by the 2-simplicial attention.

With virtual entities the update rule is for  $1 \leq i \leq N$

$$
v _ {i} ^ {\prime} = \left\{\sum_ {j = 1} ^ {N} \frac {e ^ {q _ {i} \cdot k _ {j}}}{\sum_ {s = 1} ^ {N} e ^ {q _ {i} \cdot k _ {s}}} v _ {j} \right\} \oplus \text {L a y e r N o r m} \left\{\sum_ {j, k = N + 1} ^ {N + M} \frac {e ^ {\langle p _ {i} , l _ {j} ^ {1} , l _ {k} ^ {2} \rangle}}{\sum_ {s , t = 1} ^ {N + M} e ^ {\langle p _ {i} , l _ {l} ^ {1} , l _ {m} ^ {2} \rangle}} B (u _ {j} \otimes u _ {k}) \right\} \tag {13}
$$

and for  $N < i \leq N + M$

$$
v _ {i} ^ {\prime} = \left\{\sum_ {j = 1} ^ {N + M} \frac {e ^ {q _ {i} \cdot k _ {j}}}{\sum_ {s = 1} ^ {N + M} e ^ {q _ {i} \cdot k _ {s}}} v _ {j} \right\} \oplus \operatorname {L a y e r N o r m} \left(u _ {i}\right). \tag {14}
$$

The updated representation  $e_i'$  is computed from  $v_i'$ ,  $e_i$  using (12) as before. Observe that the virtual entities are not used to update the standard entities during 1-simplicial attention and the 2-simplicial attention is not used to update the virtual entities; instead the second summand in (14) involves the vector  $u_i = W^U e_i$ , which adds recurrence to the update of the virtual entities. After the attention phase the virtual entities are discarded.

The method for updating the virtual entities is similar to the role of the memory nodes in the relational recurrent architecture of (Santoro et al., 2018), the master node in (Gilmer et al., 2017, §5.2) and memory slots in the Neural Turing Machine (Graves et al., 2014). The update rule has complexity  $O(NM^2)$  and so if we take  $M$  to be of order  $\sqrt{N}$  we get the desired complexity  $O(N^2)$ .

# 3 RL ENVIRONMENT

The environment in our reinforcement learning problem is a variant of the BoxWorld environment from (Zambaldi et al., 2019). The standard BoxWorld environment is a rectangular grid in which are situated the player (a dark gray tile) and a number of locked boxes represented by a pair of horizontally adjacent tiles with a tile of colour  $x$ , the key colour, on the left and a tile of colour  $y$ , the lock colour, on the right. There is also one loose key in each episode, which is a coloured

tile not initially adjacent to any other coloured tile. All other tiles are blank (light gray) and are traversable by the player. The rightmost column of the screen is the inventory, which fills from the top and contains keys that have been collected by the player. The player can pick up any loose key by walking over it. In order to open a locked box, with key and lock colours  $x, y$ , the player must step on the lock while in possession of a copy of  $y$ , in which case one copy of this key is removed from the inventory and replaced by a key of colour  $x$ .

The goal is to attain a white key, referred to as the Gem (represented by a white square) as shown in the sample episode of Figure 2. In this episode, there is a loose pink key (marked 1) which can be used to open one of two locked boxes, obtaining in this way either key 5 or key  $2^{1}$ . The correct choice is 2, since this leads via the sequence of keys 3, 4 to the Gem.

![](images/e83c80899eb54d0ad2235d329a81ce38ce623aa96fd6a37456eaccb8b4e2ef91.jpg)  
Figure 2: Right: a sample episode of the BoxWorld environment. The rightmost column is the player inventory, currently empty. Left: graph representation of the puzzle, with key colours as vertices and an arrow  $C \longrightarrow D$  if key  $C$  can be used to obtain key  $D$ .

![](images/c1ba21690212f52e7702b0b1612d59f1d92c71ae3be118d63bfb7db657fb7792.jpg)

Some locked boxes, if opened, provide keys that are not useful for attaining the Gem. Since each key may only be used once, opening such boxes means the episode is rendered unsolvable. Such boxes are called distractors. An episode ends when the player either obtains the Gem (with a reward of  $+10$ ) or opens a distractor box (reward  $-1$ ). Opening any non-distractor box, or picking up a loose key, garners a reward of  $+1$ . The solution length is the number of locked boxes (including the one with the Gem) in the episode on the path from the loose key to the Gem. The episode in Figure 2 has solution length four.

![](images/f88e24022dc5bba319d25109d7dfca8bc008b9518ee42f184487520dec2f36d3.jpg)  
Figure 3: Right: a sample episode of the bridge BoxWorld environment, in which the Gem has two locks and there is a marked bridge. Left: graph representation of the puzzle, with upper and lower solutions paths and the bridge between them.

![](images/261c1f893451c2d168ade79e8f80f3c4a5e2178edefdc68fc82c6108099139d9.jpg)

Our variant of the BoxWorld environment, bridge BoxWorld, is shown in Figure 3. In each episode two keys are now required to obtain the Gem, and there are therefore two loose keys on the board. To obtain the Gem, the player must step on either of the lock tiles with both keys in the inventory, at which point the episode ends with the usual  $+10$  reward. Graphically, Gems with multiple locks are denoted with two vertical white tiles on the left, and the two lock tiles on the right. Two solution paths (of the same length) leading to each of the locks on the Gem are generated with no overlapping colours, beginning with two loose keys. In episodes with multiple locks we do not consider distractor boxes of the old kind; instead there is a new type of distractor that we call a bridge. This is a locked box whose lock colour is taken from one solution branch and whose key colour is taken from the other branch. Opening the bridge renders the puzzle unsolvable. An episode ends when the player

either obtains the Gem (reward +10) or opens the bridge (reward -1). Opening a box other than the bridge, or picking up a loose key, has a reward of +1 as before. In this paper we consider episodes with zero or one bridge (the player cannot fail to solve an episode with no bridge).

# 4 RL AGENT ARCHITECTURE

Our baseline relational agent is modeled closely on (Zambaldi et al., 2019) except that we found that a different arrangement of layer normalisations worked better in our experiments, see Remark 4.1. The code for our implementation of both agents is available online. In the following we describe the network architecture of both the relational and simplicial agent; we will note the differences between the two models as they arise.

The input to the agent's network is an RGB image, represented as a tensor of shape  $[R,C + 1,3]$  (i.e. an element of  $\mathbb{R}^R\otimes \mathbb{R}^{C + 1}\otimes \mathbb{R}^3)$  where  $R$  is the number of rows and  $C$  the number of columns (the  $C + 1$  is due to the inventory). This tensor is divided by 255 and then passed through a  $2\times 2$  convolutional layer with 12 features, and then a  $2\times 2$  convolutional layer with 24 features. Both activation functions are ReLU and the padding on our convolutional layers is "valid" so that the output has shape  $[R - 2,C - 1,24]$ . We then multiply by a weight matrix of shape  $24\times 62$  to obtain a tensor of shape  $[R - 2,C - 1,62]$ . Each feature vector has concatenated to it a two-dimensional positional encoding, and then the result is reshaped into a tensor of shape  $[N,64]$  where  $N = (R - 2)(C - 1)$  is the number of Transformer entities. This is the list  $(e_1,\dots ,e_N)$  of entity representations  $e_i\in V = \mathbb{R}^{64}$ .

In the case of the simplicial agent, a further two learned embedding vectors  $e_{N + 1}, e_{N + 2}$  are added to this list; these are the virtual entities. So with  $M = 0$  in the case of the relational agent and  $M = 2$  for the simplicial agent, the entity representations form a tensor of shape  $[N + M, 64]$ . This tensor is then passed through two iterations of the Transformer block (either purely 1-simplicial in the case of the relational agent, or including both 1 and 2-simplicial attention in the case of the simplicial agent). In the case of the simplicial agent the virtual entities are then discarded, so that in both cases we have a sequence of entities  $e_1^{\prime \prime}, \ldots, e_N^{\prime \prime}$ . Inside each block are two feedforward layers separated by a ReLU activation with 64 hidden nodes; the weights are shared between iterations of the Transformer block. In the 2-simplicial Transformer block the input tensor, after layer normalisation, is passed through the 2-simplicial attention and the result (after an additional layer normalisation) is concatenated to the output of the 1-simplicial attention heads before being passed through the feedforward layers. The pseudo-code for the ordinary and 2-simplicial Transformer blocks are:

def transformer_block(e): def simplicial_transformer_block(e):  
    x = LayerNorm(e)  
    a = 1 SimplicialAttention(x)  
    b = DenseLayer1(a)  
    c = DenseLayer2(b)  
    r = Add([e, c])  
    eprime = LayerNorm(r)  
    return eprime  
    r = Add([e, c])  
    eprime = LayerNorm(r)  
    return eprime

Our implementation of the standard Transformer block is based on an implementation in Keras from (Mavreshko, 2019). In both the relational and simplicial agent, the space  $V$  of entity representations has dimension 64 and we denote by  $H^1$ ,  $H^2$  the spaces of 1-simplicial and 2-simplicial queries, keys and values. In both the relational and simplicial agent there are two heads of 1-simplicial attention,  $H^1 = H_1^1 \oplus H_2^1$  with  $\dim(H_i^1) = 32$ . In the simplicial agent there is a single head of 2-simplicial attention with  $\dim(H^2) = 48$  and two virtual entities.

The output of our Transformer block is a tensor of shape  $[N + M, 64]$ . To this final entity tensor we apply max-pooling over the entity dimension, that is, we compute a vector  $v \in \mathbb{R}^{64}$  by the rule  $v_{i} = \max_{1 \leq j \leq N}(e_{j}^{\prime \prime})_{i}$  for  $1 \leq i \leq 64$ . This vector  $v$  is then passed through four fully-connected layers with 256 hidden nodes and ReLU activations. The output of the final fully-connected layer is

multiplied by one  $256 \times 4$  weight matrix to produce logits for the actions (left, up, right and down) and another  $256 \times 1$  weight matrix to produce the value function.

Remark 4.1. There is wide variation in the use of layer normalisation in the literature on Transformer models, compare (Vaswani et al., 2017; Child et al., 2019; Zambaldi et al., 2019). The architecture described in (Zambaldi et al., 2019) involves layer normalisation in two places: on the concatenation of the  $Q$ ,  $K$ ,  $V$  matrices, and on the output of the feedforward network  $g_{\theta}$ . We keep this second normalisation but move the first from after the linear transformation  $E$  of (1) to before this linear transformation, so that it is applied directly to the incoming entity representations. We found this works well in our experiments.

# 5 EXPERIMENTS AND RESULTS

The training of our agents uses the implementation in Ray RLlib (Liang et al., 2018) of the distributed off-policy actor-critic architecture IMPALA of (Espeholt et al., 2018) with optimisation algorithm RMSProp. The hyperparameters for IMPALA and RMSProp are given in Table 1 of Appendix E. Following (Zambaldi et al., 2019) and other recent work in deep reinforcement learning, we use RMSProp with a large value of the hyperparameter  $\varepsilon = 0.1$ . As we explain in Appendix G, this is effectively RMSProp with smoothed gradient clipping.

First we verified that our implementation of the relational agent can solve the standard BoxWorld environment (Zambaldi et al., 2019) with a solution length sampled from  $[1,5]$  and number of distractors sampled from  $[0,4]$  on a  $9\times 9$  grid. After training for  $2.35\times 10^{9}$  timesteps our implementation solved over  $93\%$  of puzzles (regarding the discrepancy with the reported sample complexity in (Zambaldi et al., 2019) see Appendix D).

Next we trained the relational and simplicial agent on bridge BoxWorld, under the following conditions: half of the episodes contain a bridge, the solution length is uniformly sampled from  $[1,3]$  (both solution paths are of the same length), colours are uniformly sampled from a set of 20 colours $^2$  and the boxes and loose keys are arranged randomly on a  $7\times 9$  grid, under the constraint that the box containing the Gem does not occur in the rightmost column or bottom row, and keys appear only in positions  $(y,x) = (2r,3c - 1)$  for  $1\leq r\leq 3,1\leq c\leq 3$ . The starting and ending point of the bridge are uniformly sampled with no restrictions (e.g. the bridge can involve the colours of the loose keys and locks on the Gem) but the lock colour is always on the top solution path. There is no curriculum and no cap on timesteps per episode.

We trained four independent trials of both agents to either  $5.5 \times 10^{9}$  timesteps or convergence, whichever came first. In Figure 4 we give the mean and standard deviation of these four trials of each agent, showing a clear advantage of the simplicial agent. We make some remarks about performance comparisons taking into account the fact that the relational agent is simpler (and hence faster to execute) than the simplicial agent in Appendix D. The training runs for the relational and simplicial agents are shown in Figure 6 and Figure 7 of Appendix F, together with analysis and visualization of the 1- and 2-simplicial attention in specific examples.

In the reported experiments we use only two Transformer blocks; we performed two trials of a relational agent using four Transformer blocks, but after  $5.5 \times 10^{9}$  timesteps neither trial exceeded the 0.85 plateau in terms of fraction solved. Our overall results therefore suggest that the 2-simplicial Transformer is more powerful than the standard Transformer, with its performance not matched by adding greater depth. This is further supported by the fact on a time-adjusted basis, the 2-simplicial model still converges faster than the ordinary model; see Figure 5 of Appendix D. The upshot is that parallel computation with simplicial representations is superior to deeper computation with relational representations, at least in this class of problems.

# 6 DISCUSSION

Motivated by the idea that abstract reasoning in humans is grounded in structural representations that are adapted from those evolved for spatial reasoning, we have presented a simplicial inductive bias and shown that in the context of a deep reinforcement learning environment with nontrivial

![](images/7d4bb0db512950d0d93c00511f9f2785991cd2c74b199c423d1bfaa960b0b29f.jpg)  
Figure 4: Training curve of mean relational and simplicial agents on bridge BoxWorld. Shown are the mean and standard deviation of four runs of each agent, including the best run of each.

logical structure, this bias is superior to a purely relational inductive bias. In this concluding section we briefly address some of the limitations of our work, and future directions.

Limitations. Our experiments involve only a small number of virtual entities, and a small number of iterations of the Transformer block: it is possible that for large numbers of virtual entities and iterations, our choices of layer normalisation are not optimal. Our aim was to test the viability of the simplicial Transformer starting with the minimal configuration, so we have also not tested multiple heads of 2-simplicial attention. Deep reinforcement learning is notorious for poor reproducibility (Henderson et al., 2017), and in an attempt to follow the emerging best practices we are releasing our agent and environment code, trained agent weights, and training notebooks.

Future directions. It is clear using the general formulas for the unsigned scalar product how to define an  $n$ -simplicial Transformer block, and this is arguably an idiomatic expression in the context of deep learning of the linear logic semantics of the  $\otimes$  connective. It would be interesting to extend this to include other connectives, in environments encoding a larger fragment of linear logic proofs. However, at present this seems out of reach because the  $O(N^2)$  complexity makes scaling to much larger environments impractical. We hope that some of the scaling work being done in the Transformer literature can be adapted to the simplicial Transformer; see (Child et al., 2019).

# REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In Proceedings of the International Conference on Learning Representations, 2016.  
Aristotle. Sophistical refutations. In J. Barnes (ed.), Complete works of Aristotle, Volume 1: The revised Oxford translation. Princeton University Press, 1984.  
David G. T. Barrett, Felix Hill, Adam Santoro, Ari S. Morcos, and Timothy P. Lillicrap. Measuring abstract reasoning in neural networks. In Proceedings of the 35th International Conference on Machine Learning, pp. 4477-4486, 2018.  
Timothy E.J. Behrens, Timothy H. Muller, James C.R. Whittington, Shirley Mark, Alon B. Baram, Kimberly L. Stachenfeld, and Zeb Kurth-Nelson. What is a cognitive map? Organizing knowledge for flexible behavior. Neuron, 100(2):490 - 509, 2018.  
Jacob Bellmund, Peter Gärdenfors, Edvard Moser, and Christian F. Doeller. Navigating cognition: Spatial codes for human thinking. Science, 362, 11 2018.  
George Boole. The mathematical analysis of logic: being an essay towards a calculus of deductive reasoning. Macmillan, Barclay & Macmillan, Cambridge, 1847.  
Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. preprint arXiv:1904.10509, 2019.  
James Clift and Daniel Murfet. Cofree coalgebras and differential linear logic. preprint arXiv:1701.01285, 2017.  
Alexandra O. Constantinescu, Jill X. O'Reilly, and Timothy E.J. Behrens. Organising conceptual knowledge in humans with a gridlike code. Science, 352:1464-1468, 2016.  
Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Universal transformers. In Proceedings of the International Conference on Learning Representations, 2019.  
Honghua Dong, Jiayuan Mao, Tian Lin, Chong Wang, Lihong Li, and Denny Zhou. Neural logic machines. In Proceedings of the International Conference on Learning Representations, 2019.  
Russell A. Epstein, Eva Zita Patai, Joshua B. Julian, and Hugo J. Spiers. The cognitive map in humans: spatial navigation and beyond. Nature Neuroscience, 20:1504-1513, 2017.  
Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. In Proceedings of the 35th International Conference on Machine Learning, pp. 1407-1416, 2018.  
Gottlob Frege. On sense and denotation (über sinn und bedeutung). Zeitschrift für Philosophie und philosophische Kritik, 100:25-50, 1892.  
C. Lee Giles, Guo-Zheng Sun, Hsing-Hen Chen, Yee-Chun Lee, and Dong Chen. Higher order recurrent networks and grammatical inference. In Advances in Neural Information Processing Systems 2, pp. 380-387. 1989.  
C. Lee Giles, Dong Chen, Clifford B. Miller, Hsing-Hen Chen, Guo-Zheng Sun, and Yee-Chun Lee. Second-order recurrent neural networks for grammatical inference. In IJCNN-91-Seattle International Joint Conference on Neural Networks, volume 2, pp. 273-281, 1991.  
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning, pp. 1263-1272, 2017.  
Jean-Yves Girard. Linear logic. Theor. Comput. Sci., 50(1):1-102, 1987.  
Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.

Mark W. Goudreau, C. Lee Giles, Srimat T. Chakradhar, and Dong Chen. First-order versus second-order single-layer recurrent neural networks. IEEE Transactions on Neural Networks, 5(3):511-513, 1994.  
Alex Graves, Greg Wayne, and Ivo Danihelka. Neural tuning machines. preprint arXiv:1410.5401, 2014.  
Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Gomez, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 538, 10 2016.  
Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. preprint arXiv:1709.06560, 2017.  
David Hestenes. New foundations for classical mechanics. Kluwer Academic Publishers, 2nd edition, 2002.  
John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019.  
John J. Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8):2554-2558, 1982.  
Martin Hyland. Game semantics, pp. 131-184. Publications of the Newton Institute. Cambridge University Press, 1997.  
Ozan Irsoy and Claire Cardie. Modeling compositionality with multiplicative recurrent neural networks. In Proceedings of the International Conference on Learning Representations, 2015.  
Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu. Population based training of neural networks. preprint arXiv:1711.09846, 2017.  
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, pp. 1885-1894, 2017.  
Eric Liang, Richard Liaw, Robert Nishihara, Philipp Moritz, Roy Fox, Ken Goldberg, Joseph Gonzalez, Michael I. Jordan, and Ion Stoica. Rllib: Abstractions for distributed reinforcement learning. In Proceedings of the 35th International Conference on Machine Learning, pp. 3059-3068, 2018.  
Yunzhe Liu, Raymond J. Dolan, Zeb Kurth-Nelson, and Timothy E.J. Behrens. Human replay spontaneously reorganizes experience. Cell, 178(3):640 - 652, 2019.  
Alan Macdonald. Sobczyk's simplicial calculus does not have a proper foundation. preprint arXiv:1710.08274, 2017.  
David J.C. MacKay. Information theory, inference and learning algorithms. Cambridge University Press, 2003.  
Nicholas John Mackintosh. Animal learning, 2019. https://www.britannica.com/science/animal-learning/Insight-and-reasoning.  
Chris Martens. *Programming interactive worlds with linear logic*. PhD thesis, Carnegie Mellon University, 2015.  
Kirill Mavreshko. keras-transformer. https://github.com/kpot/keras-transformer, 2019.

Paul-Andre Mellès. Categorical semantics of linear logic. In Interactive Models of Computation and Program Behaviour, Panoramas et Syntheses 27, Société Mathématique de France 1–196, 2009.  
Anh Mai Nguyen, Jason Yosinski, and Jeff Clune. Multifaceted feature visualization: uncovering the different types of features learned by each neuron in deep neural networks. In Proceedings of the 33rd International Conference on Machine Learning, 2016.  
Jordan B. Pollack. The induction of dynamical recognizers. Machine Learning, 7(2):227-252, 1991.  
Marc'Aurelio Ranzato, Alex Krizhevsky, and Geoffrey Hinton. Factored 3-way restricted Boltzmann machines for modeling natural images. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9, pp. 621-628, 2010.  
David Raposo. Personal communication, May 13, 2019.  
Adam Santoro, David Raposo, David G Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. In Advances in Neural Information Processing Systems 30, pp. 4967-4976. 2017.  
Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, and Timothy Lillicrap. Relational recurrent neural networks. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, pp. 7310-7321, 2018.  
Luciano Serafini and Artur S. d'Avila Garcez. Logic tensor networks: Deep learning and logical reasoning from data and knowledge. In Proceedings of the 11th International Workshop on Neural-Symbolic Learning and Reasoning (NeSy'16), 2016.  
Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 3145-3153, 2017.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In Proceedings of the International Conference on Learning Representations, 2013.  
Robin Smith. Aristotle's logic. In Edward N. Zalta (ed.), The Stanford Encyclopedia of Philosophy. Metaphysics Research Lab, Stanford University, summer 2019 edition, 2019.  
Garret Sobczyk. Simplicial calculus with geometric algebra. Fundamental Theories of Physics, vol. 47, 1992.  
Richard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng. Reasoning with neural tensor networks for knowledge base completion. In Advances in Neural Information Processing Systems 26, pp. 926-934. 2013.  
Paul Vincent Spade and Jaakko J. Hintikka. History of logic, 2019. URL https://www.britannica.com/topic/history-of-logic/Aristotle.  
Ilya Sutskever, James Martens, and Geoffrey Hinton. Generating text with recurrent neural networks. In Proceedings of the 28th International Conference on Machine Learning, pp. 1017-1024, 2011.  
Richard S. Sutton and Andrew G. Barto. Reinforcement learning: An introduction. Adaptive Computation and Machine Learning series. MIT Press, 2018.  
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, pp. 2818-2826, 2016.  
Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5 - rmsprop: Divide the gradient by a running average of its recent magnitude. [Coursera] Neural Networks for Machine Learning (University of Toronto), 2012.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30, pp. 5998-6008. 2017.  
Oriol Vinyals, Igor Babuschkin, Junyoung Chung, Michael Mathieu, Max Jaderberg, Wojtek Czarecki, Andrew Dudzik, Aja Huang, Petko Georgiev, Richard Powell, Timo Ewalds, Dan Horgan, Manuel Kroiss, Ivo Danihelka, John Agapiou, Junhyuk Oh, Valentin Dalibard, David Choi, Laurent Sifre, Yury Sulsky, Sasha Vezhnevets, James Molloy, Trevor Cai, David Budden, Tom Paine, Caglar Gulcehre, Ziyu Wang, Tobias Pfaff, Toby Pohlen, Yuhuai Wu, Dani Yogatama, Julia Cohen, Katrina McKinney, Oliver Smith, Tom Schaul, Timothy Lillicrap, Chris Apps, Koray Kavukcuoglu, Demis Hassabis, and David Silver. Alphastar: Mastering the real-time strategy game starcraft ii, 2019.  
James Wallbridge. Jets and differential linear logic. preprint arXiv:1811.06235, 2018.  
James C. R. Whittington, Timothy H. Muller, Shirely Mark, Caswell Barry, and Tim E. J. Behrens. Generalisation of structural knowledge in the hippocampal-entorhinal system. In Advances in Neural Information Processing Systems 31, pp. 8493–8504, 2018.  
Vinicius Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karl Tuyls, David Reichert, Timothy Lillicrap, Edward Lockhart, Murray Shanahan, Victoria Langston, Razvan Pascanu, Matthew Botvinick, Oriol Vinyals, and Peter Battaglia. Deep reinforcement learning with relational inductive biases. In Proceedings of the International Conference on Learning Representations, 2019.
