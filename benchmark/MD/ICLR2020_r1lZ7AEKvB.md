# THE LOGICAL EXPRESSIVENESS OF GRAPH NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

The ability of graph neural networks (GNNs) for distinguishing nodes in graphs has been recently characterized in terms of the Weisfeiler-Lehman (WL) test for checking graph isomorphism. This characterization, however, does not settle the issue of which Boolean node classifiers (i.e., functions classifying nodes in graphs as true or false) can be expressed by GNNs. We tackle this problem by focusing on Boolean classifiers expressible as formulas in the logic  $\mathrm{FOC}_2$ , a well-studied fragment of first order logic.  $\mathrm{FOC}_2$  is tightly related to the WL test, and hence to GNNs. We start by studying a popular class of GNNs, which we call AC-GNNs, in which the features of each node in the graph are updated, in successive layers, only in terms of the features of its neighbors. We show that this class of GNNs is too weak to capture all  $\mathrm{FOC}_2$  classifiers, and provide a syntactic characterization of the largest subclass of  $\mathrm{FOC}_2$  classifiers that can be captured by AC-GNNs. This subclass coincides with a logic heavily used by the knowledge representation community. We then look at what needs to be added to AC-GNNs for capturing all  $\mathrm{FOC}_2$  classifiers. We show that it suffices to add readout functions, which allow to update the features of a node not only in terms of its neighbors, but also in terms of a global attribute vector. We call GNNs of this kind ACR-GNNs. We experimentally validate our findings showing that, on synthetic data conforming to  $\mathrm{FOC}_2$  formulas, AC-GNNs struggle to fit the training data while ACR-GNNs can generalize even to graphs of sizes not seen during training.

# 1 INTRODUCTION

Graph neural networks (GNNs) (Merkwirth & Lengauer, 2005; Scarselli et al., 2009) are a class of neural network architectures that has recently become popular for a wide range of applications dealing with structured data, e.g., molecule classification, knowledge graph completion, and Web page ranking (Battaglia et al., 2018; Gilmer et al., 2017; Kipf & Welling, 2017; Schlichtkrull et al., 2018). The main idea behind GNNs is that the connections between neurons are not arbitrary but reflect the structure of the input data. This approach is motivated by convolutional and recurrent neural networks and generalize both of them (Battaglia et al., 2018). Despite the fact that GNNs have recently been proven very efficient in many applications, their theoretical properties are not yet well-understood. In this paper we make a step towards understanding their expressive power by establishing connections between GNNs and well-known logical formalisms. We believe these connections to be conceptually important, as they permit us to understand the inherently procedural behavior of some fragments of GNNs in terms of the more declarative flavor of logical languages.

Two recent papers (Morris et al., 2019; Xu et al., 2019) have started exploring the theoretical properties of GNNs by establishing a close connection between GNNs and the Weisfeiler-Lehman (WL) test for checking graph isomorphism. Specifically, consider the simple GNN architecture that updates the feature vector of each graph node by combining it with the aggregation of the feature vectors of its neighbors. We call such GNNs aggregate-combine GNNs, or AC-GNNs. The authors of these papers independently observe that the node labeling produced by the WL test always refines the labeling produced by any GNN. More precisely, if two nodes are labeled the same by the algorithm underlying the WL test, then the feature vectors of these nodes produced by any AC-GNN will always be the same. Moreover, there are AC-GNNs that can reproduce the WL labeling, and hence AC-GNNs can be as powerful as the WL test for distinguishing nodes. This does not imply, however, that AC-GNNs can capture every node classifier—that is, a function assigning true or false to

every node—that is refined by the WL test. In fact, it is not difficult to see that there are many such classifiers that cannot be captured by AC-GNNs; one simple example is a classifier assigning true to every node if and only if the graph has an isolated node. Our work aims to answer the question of what are the node classifiers that can be captured by GNN architectures such as AC-GNNs.

To start answering this question, we propose to focus on logical classifiers—that is, on unary formulas expressible in first order (FO) logic: such a formula classifies each node  $v$  according to whether the formula holds for  $v$  or not. This focus gives us an opportunity to link GNNs with declarative and well understood formalisms, and to establish conclusions about GNNs drawing upon the vast amount of work on logic. For example, if one proves that two GNN architectures are captured with two logics, then one can immediately transfer all the knowledge about the relationships between those logics, such as equivalence or incomparability of expressiveness, to the GNN setting.

For AC-GNNs, a meaningful starting point to measure their expressive power is the logic  $\mathrm{FOC}_2$ , the two variable fragment of first order logic extended with counting capabilities (Cai et al., 1992). Indeed, this choice of  $\mathrm{FOC}_2$  is justified by a classical result due to Cai et al. (1992) establishing a tight connection between  $\mathrm{FOC}_2$  and WL: two nodes in a graph are classified the same by the WL test if and only if they satisfy exactly the same unary  $\mathrm{FOC}_2$  formulas. Moreover, the counting capabilities of  $\mathrm{FOC}_2$  can be mimicked in FO (albeit with more than just two variables), hence  $\mathrm{FOC}_2$  classifiers are in fact logical classifiers according to our definition.

Given the connection between AC-GNNs and WL on the one hand, and that between WL and  $\mathrm{FOC}_2$  on the other hand, one may be tempted to think that the expressivity of AC-GNNs coincides with that of  $\mathrm{FOC}_2$ . However, the reality is not as simple, and there are many  $\mathrm{FOC}_2$  node classifiers (e.g., the trivial one above) that cannot be expressed by AC-GNNs. This leaves us with the following natural questions. First, what is the largest fragment of  $\mathrm{FOC}_2$  classifiers that can be captured by AC-GNNs? Second, is there an extension of AC-GNNs that allows to express all  $\mathrm{FOC}_2$  classifiers? In this paper we provide answers to these two questions. The following are our main contributions.

- We characterize exactly the fragment of  $\mathrm{FOC}_2$  formulas that can be expressed as AC-GNNs. This fragment corresponds to graded modal logic (de Rijke, 2000), or, equivalently, to the description logic  $\mathcal{A}\mathcal{L}\mathcal{Q}$ , which has received considerable attention in the knowledge representation community (Baader et al., 2003; Baader & Lutz, 2007).  
- Next we extend the AC-GNN architecture in a very simple way by allowing global read-outs, where in each layer we also compute a feature vector for the whole graph and combine it with local aggregations; we call these aggregate-combine-readout GNNs (ACR-GNNs). These networks are a special case of the ones proposed by Battaglia et al. (2018) for relational reasoning over graph representations. In this setting, we prove that each  $\mathrm{FOC}_2$  formula can be captured by an ACR-GNN.

We experimentally validate our findings showing that the theoretical expressiveness of ACR-GNNs, as well as the differences between AC-GNNs and ACR-GNNs, can be observed when we learn from examples. In particular, we show that on synthetic graph data conforming to  $\mathrm{FOC}_2$  formulas, AC-GNNs struggle to fit the training data while ACR-GNNs can generalize even to graphs of sizes not seen during training.

# 2 GRAPH NEURAL NETWORKS

In this section we describe the architecture of AC-GNNs and introduce other related notions. We concentrate on the problem of Boolean node classification: given a (simple, undirected) graph  $G = (V,E)$  in which each vertex  $v\in V$  has an associated feature vector  $\pmb{x}_v$ , we wish to classify each graph node as true or false; in this paper, we assume that these feature vectors are one-hot encodings of node colors in the graph, from a finite set of colors. The neighborhood  $\mathcal{N}_G(v)$  of a node  $v\in V$  is the set  $\{u\mid \{v,u\} \in E\}$ .

The basic architecture for GNNs, and the one studied in recent studies on GNN expressibility (Morris et al., 2019; Xu et al., 2019), consists of a sequence of layers that combine the feature vectors of every node with the multiset of feature vectors of its neighbors. Formally, let  $\{\mathrm{AGG}^{(i)}\}_{i=1}^{L}$  and  $\{\mathrm{COM}^{(i)}\}_{i=1}^{L}$  be two sets of aggregation and combination functions. An aggregate-combine GNN

(AC-GNN) computes vectors  $\pmb{x}_v^{(i)}$  for every node  $v$  of the graph  $G$ , via the recursive formula

$$
\boldsymbol {x} _ {v} ^ {(i)} = \operatorname {C O M} ^ {(i)} \left(\boldsymbol {x} _ {v} ^ {(i - 1)}, \operatorname {A G G} ^ {(i)} \left(\{\left\{\boldsymbol {x} _ {u} ^ {(i - 1)} \mid u \in \mathcal {N} _ {G} (v) \right\} \}\right)\right), \quad \text {f o r} i = 1, \dots , L \tag {1}
$$

where each  $\boldsymbol{x}_v^{(0)}$  is the initial feature vector  $\boldsymbol{x}_v$  of  $v$ . Finally, each node  $v$  of  $G$  is classified according to a Boolean classification function CLS applied to  $\boldsymbol{x}_v^{(L)}$ . Thus, an AC-GNN with  $L$  layers is defined as a tuple  $\mathcal{A} = \left(\{\mathrm{AGG}^{(i)}\}_{i=1}^{L}, \{\mathrm{COM}^{(i)}\}_{i=1}^{L}, \mathrm{CLS}\right)$ , and we denote by  $\mathcal{A}(G, v)$  the class (i.e., true or false) assigned by  $\mathcal{A}$  to each node  $v$  in  $G$ .<sup>1</sup>

There are many possible aggregation, combination, and classification functions, which produce different classes of GNNs (Hamilton et al., 2017; Kipf & Welling, 2017; Morris et al., 2019; Xu et al., 2019). A simple, yet common choice is to consider the sum of the feature vectors as the aggregation function, and a combination function as

$$
\operatorname {C O M} ^ {(i)} \left(\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2}\right) = f \left(\boldsymbol {x} _ {1} \boldsymbol {C} ^ {(i)} + \boldsymbol {x} _ {2} \boldsymbol {A} ^ {(i)} + \boldsymbol {b} ^ {(i)}\right), \tag {2}
$$

where  $C^{(i)}$  and  $A^{(i)}$  are matrices of parameters,  $b^{(i)}$  is a bias vector, and  $f$  is a non-linearity function, such as relu or sigmoid. We call simple an AC-GNN using these functions. Furthermore, we say that an AC-GNN is homogeneous if all  $\mathrm{AGG}^{(i)}$  are the same and all  $\mathrm{COM}^{(i)}$  are the same (share the same parameters across layers). In most of our positive results we construct simple and homogeneous GNNs, while our negative results hold in general (i.e., for GNNs with arbitrary aggregation, combining, and classification functions).

The Weisfeiler-Lehman (WL) test is a powerful heuristic used to solve the graph isomorphism problem (Weisfeiler & Leman, 1968), or, for our purposes, to determine whether the neighborhoods of two nodes in a graph are structurally close or not. Due to space limitations, we refer to (Cai et al., 1992) for a formal definition of the underlying algorithm, giving only its informal description: starting from a colored graph, the algorithm iteratively assigns, for a certain number of rounds, a new color to every node in the graph; this is done in such a way that the color of a node in each round has a one to one correspondence with its own color and the multiset of colors of its neighbors in the previous round. An important observation is that the rounds of the WL algorithm can be seen as the layers of an AC-GNN whose aggregation and combination functions are all injective (Morris et al., 2019; Xu et al., 2019). Furthermore, as the following proposition states, an AC-GNN classification can never contradict the WL test.

Proposition 2.1 (Morris et al., 2019; Xu et al., 2019). If the WL test assigns the same color to two nodes in a graph, then every AC-GNN classifies either both nodes as true or both nodes as false.

# 3 CONNECTION BETWEEN GNNS AND LOGIC

# 3.1 LOGICAL NODE CLASSIFIERS

Our study relates the power of GNNs to that of classifiers expressed in first order (FO) predicate logic over (undirected) graphs where each vertex has a unique color (recall that we call these classifiers logical classifiers). To illustrate the idea of logical node classifiers, consider the formula

$$
\alpha (x) := \operatorname {R e d} (x) \wedge \exists y (E (x, y) \wedge \operatorname {B l u e} (y)) \wedge \exists z (E (x, z) \wedge \operatorname {G r e e n} (z)). \tag {3}
$$

This formula has one free variable,  $x$ , which is not bounded by any quantifier of the form  $\exists$  or  $\forall$ , and two quantified variables  $y$  and  $z$ . In general, formulas with one free variable are evaluated over nodes of a given graph. For example, the above formula evaluates to true exactly in those nodes  $v$  whose color is Red and that have both a Blue and a Green neighbor. In this case, we say that node  $v$  of  $G$  satisfies  $\alpha$ , and denote this by  $(G, v) \models \alpha$ .

Formally, a logical (node) classifier is given by a formula  $\varphi(x)$  in FO logic with exactly one free variable. This formula classifies as true those nodes  $v$  in  $G$  such that  $(G, v) \models \varphi$ , while all other nodes (i.e., those with  $(G, v) \nmid \varphi$ ) are classified as false. We say that a GNN classifier captures a logical classifier when both classifiers coincide over every node in every possible input graph.

Definition 3.1. A GNN classifier  $\mathcal{A}$  captures a logical classifier  $\varphi(x)$  if for every graph  $G$  and node  $v$  in  $G$ , it holds that  $\mathcal{A}(G, v) = \text{true}$  if and only if  $(G, v) \models \varphi$ .

# 3.2 LOGIC FOC

Logical classifiers are useful as a declarative formalism, but as we will see, they are too powerful to compare them to AC-GNNs. Instead, for reasons we explain later we focus on classifiers given by formulas in  $\mathrm{FOC}_2$ , the fragment of FO logic that only allows formulas with two variables, but in turn permits to use counting quantifiers.

Let us briefly introduce  $\mathrm{FOC}_2$  and explain why it is a restriction of FO logic. The first remark is that reducing the number of variables used in formulas drastically reduces their expressive power. Consider for example the following FO formula expressing that  $x$  is a red node, and there is another node,  $y$ , that is not connected to  $x$  and that has at least two blue neighbors,  $z_1$  and  $z_2$ :

$$
\beta (x) := \operatorname {R e d} (x) \wedge \exists y (\neg E (x, y) \wedge \exists z _ {1} \exists z _ {2} [ E (y, z _ {1}) \wedge E (y, z _ {2}) \wedge z _ {1} \neq z _ {2} \wedge \operatorname {B l u e} (z _ {1}) \wedge \operatorname {B l u e} (z _ {2}) ]).
$$

The formula  $\beta(x)$  uses four variables, but it is possible to find an equivalent one with just three: the trick is to reuse variable  $x$  and replace every occurrence of  $z_2$  in  $\beta(x)$  by  $x$ . However, this is as far as we can go with this trick:  $\beta(x)$  does not have an equivalent formula with less than three variables. In the same way, the formula  $\alpha(x)$  given in Equation (3) can be expressed using only two variables,  $x$  and  $y$ , simply by reusing  $y$  in place of  $z$ .

That being said, it is possible to extend the logic so that some node properties, such as the one defined by  $\beta(x)$ , can be expressed with even less variables. To this end, consider the counting quantifier  $\exists^{\geq N}$  for every positive integer  $N$ . Analogously to how the quantifier  $\exists$  expresses the existence of a node satisfying a property, the quantifier  $\exists^{\geq N}$  expresses the existence of at least  $N$  different nodes satisfying a property. For example, with  $\exists^{\geq 2}$  we can express  $\beta(x)$  by using only two variables by means of the classifier

$$
\gamma (x) := \operatorname {R e d} (x) \wedge \exists y (\neg E (x, y) \wedge \exists^ {\geq 2} x [ E (y, x) \wedge \operatorname {B l u e} (x) ]). \tag {4}
$$

Based on this idea, the logic  $\mathrm{FOC}_2$  allows for formulas using all FO constructs and counting quantifiers, but restricted to only two variables. Note that, in terms of their logical expressiveness, we have that  $\mathrm{FOC}_2$  is strictly less expressive than FO (as counting quantifiers can always be mimicked in FO by using more variables and disequalities), but is strictly more expressive than  $\mathrm{FO}_2$ , the fragment of FO that allows formulas to use only two variables (as  $\beta(x)$  belongs to  $\mathrm{FOC}_2$  but not to  $\mathrm{FO}_2$ ).

The following result establishes a classical connection between  $\mathrm{FOC}_2$  and the WL test. Together with Proposition 2.1, this provides a justification for our choice of logic  $\mathrm{FOC}_2$  for measuring the expressiveness of AC-GNNs.

Proposition 3.2 (Cai et al., 1992). For any graph  $G$  and nodes  $u, v$  in  $G$ , the WL test colors  $v$  and  $u$  the same after any number of rounds iff  $u$  and  $v$  are classified the same by all  $FOC_2$  classifiers.

# 3.3 FOC $_2$  AND AC-GNN CLASSIFIERS

Having Propositions 2.1 and 3.2, one may be tempted to combine them and claim that every  $\mathrm{FOC}_2$  classifier can be captured by an AC-GNN. Yet, this is not the case as shown in Proposition 3.3 below. In fact, while it is true that two nodes are declared indistinguishable by the WL test if and only if they are indistinguishable by all  $\mathrm{FOC}_2$  classifiers (Proposition 3.2), and if the former holds then such nodes cannot be distinguished by AC-GNNs (Proposition 2.1), this by no means tells us that every  $\mathrm{FOC}_2$  classifier can be expressed as an AC-GNN.

Proposition 3.3. There is an  $FOC_{2}$  classifier that is not captured by any AC-GNN.

One such  $\mathrm{FOC}_2$  classifier is  $\gamma(x)$  in Equation (4), but there are infinitely many and even simpler  $\mathrm{FOC}_2$  formulas that cannot be captured by AC-GNNs. Intuitively, the main problem is that an AC-GNN has only a fixed number  $L$  of layers and hence the information of local aggregations cannot travel further than at distance  $L$  of every node along edges in the graph. For instance, the red node in  $\gamma(x)$  may be farther away than the node with the blue neighbours, which means that AC-GNNs would never be able to connect this information. Actually, both nodes may even be in different connected components of a graph, in which case no number of layers would suffice.

The negative result of Proposition 3.3 opens up the following important questions.

1. What kind of  $\mathrm{FOC}_2$  classifiers can be captured by AC-GNNs?  
2. Can we capture  $\mathrm{FOC}_2$  classifiers with GNNs using a simple extension of AC-GNNs?

We provide answers to these questions in the next two sections.

# 4 THE EXPRESSIVE POWER OF AC-GNNS

Towards answering our first question, we recall that the problem with AC-GNN classifiers is that they are local, in the sense that they cannot see across a distance greater than their number of layers. Thus, if we want to understand which logical classifiers this architecture is capable of expressing, we must consider logics built with similar limitations in mind. And indeed, in this section we show that AC-GNNs capture any  $\mathrm{FOC}_2$  classifier as long as we further restrict the formulas so that they satisfy such a locality property. This happens to be a well-known restriction of  $\mathrm{FOC}_2$ , and corresponds to graded modal logic (de Rijke, 2000) or, equivalently, to description logic  $\mathcal{A}\mathcal{L}\mathcal{Q}$  (Baader et al., 2003), which is fundamental for knowledge representation: for instance, the OWL 2 Web Ontology Language (Motik et al., 2012; W3C OWL Working Group, 2012) relies on  $\mathcal{A}\mathcal{L}\mathcal{Q}$ .

The idea of graded modal logic is to force all subformulas to be guarded by the edge predicate  $E$ . This means that one cannot express in graded modal logic arbitrary formulas of the form  $\exists y\varphi(y)$ , i.e., whether there is some node that satisfies property  $\varphi$ . Instead, one is allowed to check whether some neighbor  $y$  of the node  $x$  where the formula is being evaluated satisfies  $\varphi$ . That is, we are allowed to express the formula  $\exists y(E(x,y) \land \varphi(y))$  in the logic as in this case  $\varphi(y)$  is guarded by  $E(x,y)$ . We can define this fragment of FO logic using FO syntax as follows. A graded modal logic formula is either  $\operatorname{Col}(x)$ , for Col a node color, or one of the following, where  $\varphi$  and  $\psi$  are graded modal logic formulas and  $N$  is a positive integer:

$$
\neg \varphi (x), \quad \varphi (x) \wedge \psi (x), \quad \exists^ {\geq N} y \left(E (x, y) \wedge \varphi (y)\right).
$$

Notice then that the formula  $\delta(x) \coloneqq \operatorname{Red}(x) \wedge \exists y (E(x, y) \wedge \operatorname{Blue}(y))$  is in graded modal logic, but the logical classifier  $\gamma(x)$  in Equation (4) is not, because the use of  $\neg E(x, y)$  as a guard is disallowed. As required, we can now show that AC-GNNs can indeed capture all graded modal logic classifiers.

Proposition 4.1. Each graded modal logic classifier is captured by a simple homogeneous AC-GNN.

The key idea of the construction is that the vectors' dimensions used by the AC-GNN to label nodes, represent the sub-formulas of the captured classifier. Thus, if a feature in a node is 1 then the node satisfies the corresponding sub-formula, and the opposite holds after evaluating  $L$  layers, where  $L$  is the "quantifier depth" of the classifier (which does not depend on the graph). The construction uses simple, homogeneous AC-GNNs with the truncated relu non-linearity  $\max(0, \min(x, 1))$ . The formal proof of Proposition 4.1, as well as other formal statements, can be found in the Appendix.

The relationship between AC-GNNs and graded modal logic goes further: we can show that graded modal logic is the "largest" class of logical classifiers captured by AC-GNNs. This means that the only FO formulas that AC-GNNs are able to learn accurately are those in graded modal logic.

Theorem 4.2. A logical classifier is captured by AC-GNNs if and only if it can be expressed in graded modal logic.

The backward direction of this theorem is Proposition 4.1, while the proof of the forward direction is based on a recently communicated extension of deep results in finite model theory (Otto, 2019).

# 5 GNNS FOR CAPTURING FOC $_2$

# 5.1 GNNS WITH GLOBAL READOUTS

In this section we tackle our second question: which kind of GNN architecture we need to capture all  $\mathrm{FOC}_2$  classifiers? Recall that the main shortcoming of AC-GNNs for expressing such classifiers is their local behavior. A natural way to break such a behavior is to allow for a global feature computation on each layer of the GNN. This is called a global attribute computation in the framework

of Battaglia et al. (2018). Following the recent GNN literature (Gilmer et al., 2017; Morris et al., 2019; Xu et al., 2019), we refer to this global operation as a readout.

Formally, an aggregate-combine-readout GNN (ACR-GNN) extends AC-GNNs by specifying readout functions  $\{\mathrm{READ}^{(i)}\}_{i = 1}^{L}$ , which aggregate the current feature vectors of all the nodes in a graph. Then, the vector  $\pmb{x}_v^{(i)}$  of each node  $v$  in  $G$  on each layer  $i$ , is computed by the following formula, generalizing Equation (1):

$$
\boldsymbol {x} _ {v} ^ {(i)} = \operatorname {C O M} ^ {(i)} \left(\boldsymbol {x} _ {v} ^ {(i - 1)}, \operatorname {A G G} ^ {(i)} \left(\{\{\boldsymbol {x} _ {u} ^ {(i - 1)} \mid u \in \mathcal {N} _ {G} (v) \}\right), \operatorname {R E A D} ^ {(i)} \left(\{\{\boldsymbol {x} _ {u} ^ {(i - 1)} \mid u \in G \}\right)\right). \tag {5}
$$

Intuitively, every layer in an ACR-GNN first computes (i.e., "reads out") the aggregation over all the nodes in  $G$ ; then, for every node  $v$ , it computes the aggregation over the neighbors of  $v$ ; and finally it combines the features of  $v$  with the two aggregation vectors. All the notions about AC-GNNs extend to ACR-GNNs in a straightforward way; for example, a simple ACR-GNN uses the sum as the function  $\mathrm{READ}^{(i)}$  in each layer, and the combination function  $\mathrm{COM}^{(i)}(\pmb{x}_1, \pmb{x}_2, \pmb{x}_3) = f(\pmb{x}_1\pmb{C}^{(i)} + \pmb{x}_2\pmb{A}^{(i)} + \pmb{x}_3\pmb{R}^{(i)} + \pmb{b}^{(i)})$  with a matrix  $\pmb{R}^{(i)}$ , generalizing Equation (2).

# 5.2 ACR-GNNs AND FOC $_2$

To see how a readout function could help in capturing non-local properties, consider again the logical classifier  $\gamma(x)$  in Equation (4), that assigns true to every red node  $v$  as long as there is another node not connected with  $v$  having two blue neighbors. We have seen that AC-GNNs cannot capture this classifier. However, using a single readout plus local aggregations one can implement this classifier as follows. First, define by  $B$  the property "having at least 2 blue neighbors". Then an ACR-GNN that implements  $\gamma(x)$  can (1) use one aggregation to store in the local feature of every node if the node satisfies  $B$ , then (2) use a readout function to count how many nodes satisfying  $B$  exist in the whole graph, and (3) use another local aggregation to count how many neighbors of every node satisfy  $B$ . Then  $\gamma$  is obtained by classifying as true every red node having less neighbors satisfying  $B$  than the total number of nodes satisfying  $B$  in the whole graph. It turns out that the usage of readout functions is enough to capture all non-local properties of  $\mathrm{FOC}_2$  classifiers.

Theorem 5.1. Each  $FOC_{2}$  classifier can be captured by a simple homogeneous ACR-GNN.

The construction is similar to that of Proposition 4.1 and uses simple, homogeneous ACR-GNNs—that is, the readout function is just the sum of all the local node feature vectors. Moreover, the readout functions are only used to deal with subformulas asserting the existence of a node that is not connected to the current node in the graph, just as we have done for classifier  $\gamma(x)$ . As an intermediate step in the proof, we use a characterization of  $\mathrm{FOC}_2$  using an extended version of graded modal logic, which was obtained by Lutz et al. (2001). We leave as a challenging open problem whether  $\mathrm{FOC}_2$  classifiers are exactly the logical classifiers captured by ACR-GNNs.

# 5.3 COMPARING THE NUMBER OF READOUT LAYERS

The proof of Theorem 5.1 constructs GNNs whose number of layers depends on the formula being captured—that is, readout functions are used unboundedly many times in ACR-GNNs for capturing different  $\mathrm{FOC}_2$  classifiers. Given that a global computation can be costly, one might wonder whether this is really needed, or if it is possible to cope with all the complexity of such classifiers by performing only few readouts. We next show that actually just one readout is enough. However, this reduction in the number of readouts comes at the cost of severely complicating the resulting GNN.

Formally, an aggregate-combine GNN with final readout (AC-FR-GNN) results out of using any number of layers as in the AC-GNN definition, together with a final layer that uses a readout function, according to Equation (5).

Theorem 5.2. Each  $FOC_{2}$  classifier is captured by an AC-FR-GNN.

The AC-FR-GNN in the proof of this theorem is not based on the idea of evaluating the formula incrementally along layers, as in the proofs of Proposition 4.1 and Theorem 5.1, and it is not simple (note that AC-FR-GNNs are never homogeneous). Instead, it is based on a refinement of the GIN architecture proposed by Xu et al. (2019) to obtain as much information as possible about the local

<table><tr><td></td><td>Line Train</td><td colspan="2">Line Test</td><td>E-R Train</td><td colspan="2">E-R Test</td></tr><tr><td></td><td></td><td>same-size</td><td>bigger</td><td></td><td>same-size</td><td>bigger</td></tr><tr><td>AC-5</td><td>0.887</td><td>0.886</td><td>0.892</td><td>0.951</td><td>0.949</td><td>0.929</td></tr><tr><td>AC-7</td><td>0.892</td><td>0.892</td><td>0.897</td><td>0.967</td><td>0.965</td><td>0.958</td></tr><tr><td>GIN-5</td><td>0.861</td><td>0.861</td><td>0.867</td><td>0.830</td><td>0.831</td><td>0.817</td></tr><tr><td>GIN-7</td><td>0.863</td><td>0.864</td><td>0.870</td><td>0.818</td><td>0.819</td><td>0.813</td></tr><tr><td>ACR-1</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr></table>

Table 1: Results on synthetic data for nodes labeled by classifier  $\alpha \left( x\right)  \mathrel{\text{:=}} \operatorname{Red}\left( x\right)  \land  \exists y\operatorname{Blue}\left( y\right)$

neighborhood in graphs, followed by a readout and combine functions that use this information to deal with non-local constructs in formulas. The first component we build is an AC-GNN that computes an invertible function mapping each node to a number representing its neighborhood (how big is this neighborhood depends on the classifier to be captured). This information is aggregated so that we know for each different type of a neighborhood how many times it appears in the graph. We then use the combine function to evaluate  $\mathrm{FOC}_2$  formulas by decoding back the neighborhoods.

# 6 EXPERIMENTAL RESULTS

We perform experiments with synthetic data to empirically validate our results. The motivation of this section is to show that the theoretical expressiveness of ACR-GNNs, as well as the differences between AC- and ACR-GNNs, can actually be observed when we learn from examples. We perform two sets of experiments: experiments to show that ACR-GNNs can learn a very simple  $\mathrm{FOC}_2$  node classifier that AC-GNNs cannot learn, and experiments involving complex  $\mathrm{FOC}_2$  classifiers that need more intermediate readouts to be learned. We implemented our experiments in the PyTorch Geometric library (Fey & Lenssen, 2019). Besides testing simple AC-GNNs, we also tested the GIN network proposed by Xu et al. (2019) (we consider the implementation by Fey & Lenssen (2019) and adapted it to classify nodes). Our experiments use synthetic graphs, with five initial colors encoded as one-hot features, divided in three sets: train set with  $5\mathrm{k}$  graphs of size up to 50-100 nodes, test set with 500 graphs of size similar to the train set, and another test set with 500 graphs of size bigger than the train set. We tried several configurations for the aggregation, combination and readout functions, and report the accuracy on the best configuration. Accuracy in our experiments is computed as the total number of nodes correctly classified among all nodes in all the graphs in the dataset. In every case we run up to 20 epochs with the Adam optimizer. More details on the experimental setting, data, and code can be found in the Appendix. We finally report results on a real benchmark (PPI) where we did not observe an improvement of ACR-GNNs over AC-GNNs.

Separating AC-GNNs and ACR-GNNs We consider a very simple  $\mathrm{FOC}_2$  formula defined by  $\alpha(x) := \operatorname{Red}(x) \wedge \exists y \operatorname{Blue}(y)$ , which is satisfied by every red node in a graph provided that the graph contains at least one blue node. We tested with line-shaped graphs and Erdős-Renyi (E-R) random graphs with different connectivities. In every set (train and test) we consider  $50\%$  of graphs not containing any blue node, and  $50\%$  containing at least one blue node (around  $20\%$  of nodes are in the true class in every set). For both types of graphs, already single-layer ACR-GNNs showed perfect performance (ACR-1 in Table 1). This was what we expected given the simplicity of the property being checked. In contrast, AC-GNNs and GINs (shown in Table 1 as AC- $L$  and GIN- $L$ , representing AC-GNNs and GINs with  $L$  layers) struggle to fit the data. For the case of the line-shaped graph, they were not able to fit the train data even by allowing 7 layers. For the case of random graphs, the performance with 7 layers was considerably better. In a closer look at the performance for different connectivities of E-R graphs, we found an improvement for AC-GNNs when we train them with more dense graphs (details in the Appendix). This is consistent with the fact that AC-GNNs are able to move information of local aggregations to distances up to their number of layers. This combined with the fact that random graphs that are more dense make the maximum distances between nodes shorter, may explain the boost in performance for AC-GNNs.

<table><tr><td></td><td>α1 Train</td><td colspan="2">α1 Test</td><td>α2 Train</td><td colspan="2">α2 Test</td><td>α3 Train</td><td colspan="2">α3 Test</td></tr><tr><td></td><td></td><td>same-size</td><td>bigger</td><td></td><td>same-size</td><td>bigger</td><td></td><td>same-size</td><td>bigger</td></tr><tr><td>AC</td><td>0.839</td><td>0.826</td><td>0.671</td><td>0.694</td><td>0.695</td><td>0.667</td><td>0.657</td><td>0.636</td><td>0.632</td></tr><tr><td>GIN</td><td>0.567</td><td>0.566</td><td>0.536</td><td>0.689</td><td>0.693</td><td>0.672</td><td>0.656</td><td>0.643</td><td>0.580</td></tr><tr><td>AC-FR-2</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.863</td><td>0.860</td><td>0.694</td><td>0.788</td><td>0.775</td><td>0.770</td></tr><tr><td>AC-FR-3</td><td>1.000</td><td>1.000</td><td>0.825</td><td>0.840</td><td>0.823</td><td>0.604</td><td>0.787</td><td>0.767</td><td>0.771</td></tr><tr><td>ACR-1</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.827</td><td>0.834</td><td>0.726</td><td>0.760</td><td>0.762</td><td>0.773</td></tr><tr><td>ACR-2</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.895</td><td>0.897</td><td>0.770</td><td>0.800</td><td>0.799</td><td>0.771</td></tr><tr><td>ACR-3</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.903</td><td>0.902</td><td>0.836</td><td>0.817</td><td>0.802</td><td>0.748</td></tr></table>

Table 2: Results on E-R synthetic data for nodes labeled by classifiers  ${\alpha }_{i}\left( x\right)$  in Equation (6)

Complex  $\mathbf{FOC}_2$  properties In the second experiment we consider classifiers  $\alpha_{i}(x)$  constructed as

$$
\alpha_ {0} (x) := \operatorname {B l u e} (x), \quad \alpha_ {i + 1} (x) := \exists^ {[ N, M ]} y (\alpha_ {i} (y) \wedge \neg E (x, y)), \tag {6}
$$

where  $\exists^{[N,M]}$  stands for "there exist between  $N$  and  $M$  nodes" satisfying a given property (each  $\alpha_{i}(x)$  can be expressed in  $\mathrm{FOC}_2$  by combining  $\exists^{\geq N}$  and  $\neg \exists^{\geq M + 1}$ ). We created datasets with E-R dense graphs and labeled them according to  $\alpha_{1}(x),\alpha_{2}(x)$ , and  $\alpha_{3}(x)$ , ensuring in each case that approximately half of all nodes in our dataset satisfy every property. Our experiments show that when increasing the depth of the formula (existential quantifiers with negations inside other existential quantifiers) more layers are needed to increase train and test accuracy (see Table 2). We report ACR-GNNs performance up to 3 layers (ACR-  $L$  in Table 2) as beyond that we did not see any significant improvement. We also note that for the bigger test set, AC-GNNs and GINs are unable to substantially depart from a trivial baseline of  $50\%$ . We tested these networks with up to 10 layers but only report the best results on the bigger test set. We also test AC-FR-GNNs with two and three layers (AC-FR-  $L$  in Table 2). As we expected, although theoretically using a single readout gives the same expressive power as using several of them (Theorem 5.2), in practice more than a single readout can actually help the learning process of complex properties.

PPI We also tested AC- and ACR-GNNs on the Protein-Protein Interaction (PPI) benchmark (Zitnik & Leskovec, 2017). We chose PPI since it is a node classification benchmark with different graphs in the train set (as opposed to other popular benchmarks for node classification such as Core or Citeseer that have a single graph). Although the best results for both classes of GNNs on PPI were quite high (AC: 97.5 F1, ACR: 95.4 F1 in the test set), we did not observe an improvement when using ACR-GNNs. Chen et al. (2019) recently observed that commonly used benchmarks are inadequate for testing advanced GNN variants, and ACR-GNNs might be suffering from this fact.

# 7 FINAL REMARKS

Our results show the theoretical advantages of mixing local and global information when classifying nodes in a graph. Recent works have also observed these advantages in practice, e.g., Deng et al. (2018) use global-context aware local descriptors to classify objects in 3D point clouds, You et al. (2019) construct node features by computing shortest-path distances to a set of distant anchor nodes, and Haonan et al. (2019) introduced the idea of a "star node" that stores global information of the graph. As mentioned before, our work is close in spirit to that of Xu et al. (2019) and Morris et al. (2019) establishing the correspondence between the WL test and GNNs. In contrast to our work, they focus on graph classification and do not consider the relationship with logical classifiers.

Morris et al. (2019) also studied  $k$ -GNNs, which are inspired by the  $k$ -dimensional WL test. In  $k$ -GNNs, graphs are considered as structures connecting  $k$ -tuples of nodes instead of just pairs of them. We plan to study how our results on logical classifiers relate to  $k$ -GNNs, in particular, with respect to the logic  $\mathrm{FOC}_k$  that extends  $\mathrm{FOC}_2$  by allowing formulas with  $k$  variables, for each fixed  $k > 1$ . Recent work has also explored the extraction of finite state representations from recurrent neural networks as a way of explaining them (Weiss et al., 2018; Koul et al., 2019; Oliva & Lago-Fernandez, 2019). We would like to study how our results can be applied for extracting logical formulas from GNNs as possible explanations for their computations.

# REFERENCES

Franz Baader and Carsten Lutz. Description logic. In Handbook of Modal Logic, pp. 757-819. North-Holland, 2007.  
Franz Baader, Diego Calvanese, Deborah L. McGuinness, Daniele Nardi, and Peter F. Patel-Schneider (eds.). The Description Logic Handbook: Theory, Implementation, and Applications. Cambridge University Press, 2003.  
Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Flores Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulçehre, H. Francis Song, Andrew J. Ballard, Justin Gilmer, George E. Dahl, Ashish Vaswani, Kelsey R. Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matthew Botvinick, Oriol Vinyals, Yujia Li, and Razvan Pascanu. Relational inductive biases, deep learning, and graph networks. CoRR, abs/1806.01261, 2018. URL http://arxiv.org/abs/1806.01261.  
Jin-Yi Cai, Martin Fürer, and Neil Immerman. An optimal lower bound on the number of variables for graph identification. Combinatorica, 12(4):389-410, 1992.  
Ting Chen, Song Bian, and Yizhou Sun. Are powerful graph neural nets necessary? a dissection on graph classification, 2019.  
Maarten de Rijke. A Note on Graded Modal Logic. Studia Logica, 64(2):271-283, 2000.  
Haowen Deng, Tolga Birdal, and Slobodan Ilic. PPFnet: Global context aware local features for robust 3d point matching. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 195-205, 2018.  
Matthias Fey and Jan Eric Lenssen. Fast Graph Representation Learning with PyTorch Geometric. CoRR, abs/1903.02428, 2019.  
Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural Message Passing for Quantum Chemistry. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, pp. 1263-1272, 2017.  
William L. Hamilton, Zhitao Ying, and Jure Leskovec. Inductive Representation Learning on Large Graphs. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, 4-9 December 2017, Long Beach, CA, USA, pp. 1024-1034, 2017.  
Lu Haonan, Seth H Huang, Tian Ye, and Guo Xiuyan. Graph star net for generalized multi-task learning. arXiv preprint arXiv:1906.12330, 2019.  
Thomas N. Kipf and Max Welling. Semi-Supervised Classification with Graph Convolutional Networks. In Proceedings of the 5th International Conference on Learning Representations, ICLR 2017, Toulouse, France, April 24-26, 2017.  
Anurag Koul, Sam Greydanus, and Alan Fern. Learning finite state representations of recurrent policy networks. In ICLR, 2019.  
Carsten Lutz, Ulrike Sattler, and Frank Wolter. Modal logic and the two-variable fragment. In Proceedings of the International Workshop on Computer Science Logic, pp. 247-261. Springer, 2001.  
Christian Merkwirth and Thomas Lengauer. Automatic Generation of Complementary Descriptors with Molecular Graph Networks. J. of Chemical Information and Modeling, 45(5):1159-1168, 2005.  
Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence, AAAI 2019, Honolulu, Hawaii, USA, January 27 – February 1, pp. 4602–4609, 2019.

Boris Motik, Bernardo Cuenca Grau, Ian Horrocks, Zhe Wu, Achille Fokoue, and Carsten Lutz. OWL 2 Web ontology language profiles (second edition). W3C recommendation, W3C, 2012. URL http://www.w3.org/TR/owl2-profiles/.  
Christian Oliva and Luis F Lago-Fernández. On the Interpretation of Recurrent Neural Networks as Finite State Machines. In International Conference on Artificial Neural Networks, pp. 312-323. Springer, 2019.  
Martin Otto. Graded modal logic and counting bisimulation. https://www2.mathematik.tu-darmstadt.de/~otto/papers/cml19.pdf, 2019.  
Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The Graph Neural Network Model. IEEE Trans. Neural Networks, 20(1):61-80, 2009.  
Michael Sejr Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling. Modeling Relational Data with Graph Convolutional Networks. In *The Semantic Web - 15th International Conference*, ESWC 2018, Heraklion, Crete, Greece, June 3-7, 2018, Proceedings, pp. 593-607, 2018.  
W3C OWL Working Group. OWL 2 Web ontology language document overview (second edition). W3C recommendation, W3C, 2012. URL https://www.w3.org/TR/owl2-overview/.  
Boris Yu. Weisfeiler and Andrei A. Leman. A Reduction of a Graph to a Canonical Form and an Algebra Arising during this Reduction. Nauchno-Technicheskaya Informatsia, 2(9):12-16, 1968. Translated from Russian.  
Gail Weiss, Yoav Goldberg, and Eran Yahav. Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples. In International Conference on Machine Learning, pp. 5244-5253, 2018.  
Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How Powerful are Graph Neural Networks? In Proceedings of the 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019.  
Jiaxuan You, Rex Ying, and Jure Leskovec. Position-aware Graph Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, pp. 7134-7143, 2019.  
Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue networks. CoRR, abs/1707.04638, 2017. URL http://arxiv.org/abs/1707.04638.
