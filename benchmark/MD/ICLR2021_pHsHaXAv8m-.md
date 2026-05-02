# TOWARDS PRINCIPLED REPRESENTATION LEARNING FOR ENTITY ALIGNMENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Knowledge graph (KG) representation learning for entity alignment has recently received great attention. Compared with conventional methods, these embedding-based ones are considered to be robuster for highly-heterogeneous and cross-lingual entity alignment scenarios as they do not rely on the quality of machine translation or feature extraction. Despite the significant improvement that has been made, there is little understanding of how the embedding-based entity alignment methods actually work. Most existing methods rest on the foundation that a small number of pre-aligned entities can serve as anchors to connect the embedding spaces of two KGs. But no one investigates the rationality of such foundation. In this paper, we define a typical paradigm abstracted from the existing methods, and analyze how the representation discrepancy between two potentially-aligned entities is implicitly bounded by a predefined margin in the scoring function for embedding learning. However, such a margin cannot guarantee to be tight enough for alignment learning. We mitigate this problem by proposing a new approach that explicitly learns KG-invariant and principled entity representations, meanwhile preserves the original infrastructure of existing methods. In this sense, the model not only pursues the closeness of aligned entities on geometric distance, but also aligns the neural ontologies of two KGs to eliminate the discrepancy in feature distribution and underlying ontology knowledge. Our experiments demonstrate consistent and significant improvement in performance against the existing embedding-based entity alignment methods, including several state-of-the-art ones.

# 1 INTRODUCTION

Knowledge Graphs (KGs), such as DBpedia (Auer et al., 2007) and Wikidata (Vrandecic & Krötzsch, 2014), have become crucial data resources for many AI applications. Although a large-scale KG offers structured knowledge derived from millions of facts in the real world, it is still incomplete by nature, and the downstream applications are always demanding for more knowledge. To resolve this issue, the task of entity alignment (EA) is proposed, which exploits the potentially-aligned entities among different KGs to facilitate knowledge fusion and exchange.

Recently, embedding-based entity alignment (EEA) methods (Chen et al., 2017; Zhu et al., 2017; Wang et al., 2018; Guo et al., 2019; Ye et al., 2019; Wu et al., 2019; Sun et al., 2020a; Fey et al., 2020) have been prevailing in this area. Their common idea is to encode semantics into embeddings and estimate the similarities by embedding distance. During this process, a small number of aligned entity pairs (a.k.a., seed alignment) are required as supervision data to align (or merge) the embedding spaces of KGs. These methods either learn an alignment function  $f_{a}$  to minimize the difference between two entity embeddings in each seed (Wang et al., 2018), or directly map aligned entities to one embedding vector (Sun et al., 2017). Meanwhile, they also leverage a shared scoring function  $f_{s}$  to encode semantics into representations, such that two underlying aligned entities that connect to respective sides of a seed shall have similar characteristics in their feature expression.

Although the effectiveness of current EEA methods are empirically demonstrated (Sun et al., 2020b), little efforts have been made on the theoretical analysis. In this paper, we fill this gap by formally defining a paradigm leveraged by the current methods. We show that the representation discrepancy of an underlying aligned entity pair is bounded in an indirect way by a margin  $\lambda$  in the scoring

function  $f_{s}$ . Unfortunately, we further find that this margin-based bound cannot be set as tight as expected, causing that little constrain can be put on the entities with few neighbors.

To mitigate the above problem, we propose neural ontology driven entity alignment (abbr., NeoEA), in which the entity representations are optimized jointly with a neural ontology. An ontology (Baader et al., 2005) is usually comprised of axioms that define the legitimate relationships among entities and relations. Those axioms make a KG principled (i.e., constrained by rules). For example, an "Object Property Domain" axiom in OWL2 (Baader et al., 2005) claims the valid head entities for a specific relation (e.g., the head entities of relation "birthPlace" should be in class "Person"), and it thus determines the head entity distributions of this relation. The neural ontology in this paper, however, is reversely deduced from the entity distributions. We expect to align the high-level neural ontology to diminish the discrepancy of feature distributions, as well as ontology knowledge, between two KGs.

The main contributions of this paper are threefold:

- We define the paradigm of the current EEA methods, and demonstrate that the embedding discrepancy in each potential alignment pair is implicitly bounded by the margin in the scoring function. We show that this bound cannot be as tight as we expect.  
- We propose NeoEA to learn  $KG$ -invariant as well as principled representations by aligning the neural axioms of two KGs. We prove that minimizing the difference can substantially align their corresponding ontology-level knowledge, without the assumption about the existence of real ontology data.  
- We conducted experiments to verify the effectiveness of NeoEA with several state-of-the-art methods as baselines. The results show that NeoEA can consistently and significantly improve the performance of the EEA methods.

# 2 EMBEDDING-BASED ENTITYALIGNMENT

# 2.1 METHODOLOGY

We first summarize the common paradigm employed by most existing EEA methods (Chen et al., 2017; Sun et al., 2017; Zhu et al., 2017; Sun et al., 2018; Wang et al., 2018; Pei et al., 2019; Guo et al., 2019; Wu et al., 2019; Ye et al., 2019; Sun et al., 2020a):

Definition 1 (Embedding-based Entity Alignment). The input of EEA is two KGs  $\mathcal{G}_1 = (\mathcal{E}_1, \mathcal{R}_1, \mathcal{T}_1)$ ,  $\mathcal{G}_2 = (\mathcal{E}_2, \mathcal{R}_2, \mathcal{T}_2)$ , and a small subset of aligned entity pairs  $\mathcal{S} \subset \mathcal{E}_1 \times \mathcal{E}_2$  as seeds to connect  $\mathcal{G}_1$  with  $\mathcal{G}_2$ . An EEA model consists of two neural functions: an alignment function  $f_a$ , which is used to regularize the embeddings of pairwise entities in  $\mathcal{S}$ ; and a scoring function  $f_s$ , which scores the representations based on the joint triple set  $\mathcal{T}_1 \cup \mathcal{T}_2$ . EEA estimates the alignment score of an arbitrary entity pair  $(e_i^1, e_j^2)$  by their geometric distance  $d(\mathbf{e}_i^1, \mathbf{e}_j^2)$ , where  $\mathbf{e}_i^1$ ,  $\mathbf{e}_j^2$  denote the embeddings of  $e_i^1$ ,  $e_j^2$  respectively.

It is worth noting that, the existing EEA methods have different settings in relation seed alignment. Some works (Chen et al., 2017; Zhu et al., 2017) assume that all aligned relation pairs are known in advance. Others (Sun et al., 2017; 2018) suppose that the number of relations is much smaller than that of entities, i.e.,  $|\mathcal{R}| \ll |\mathcal{E}|$ , which means that the training data for aligning relations is sufficient. In this paper, we do not explore the details of relation seed setting. We assume that the relation representations for a well-trained EEA model are aligned.

The existing works have explored the diversity of  $f_{a}$ . For example, the pioneering work MTransE (Chen et al., 2017) proposed to learn a mapping matrix to cast an entity representation  $\mathbf{e}_i^1$  to the feature space of  $\mathcal{G}_2$ . However, this approach was soon replaced by a simpler yet more efficient choice that directly maps  $(e_i^1, e_i^2) \in S$  to one embedding vector  $\mathbf{e}_i$  (Sun et al., 2017; Zhu et al., 2017; Trsedya et al., 2019; Guo et al., 2019). Recently, researchers (Wang et al., 2018; Pei et al., 2019; Wu et al., 2019) start to leverage a softer way to incorporate seed information, in which the distance between entities in a positive pair (i.e., supervised data in  $S$ ) is minimized, while that referred to the negative one will be enlarged. As the most common choice, we consider  $f_{a}$  as Euclidean distance

between two embeddings, such that the corresponding alignment loss can be written as follows:

$$
\mathcal {L} _ {a} = \sum_ {\left(e _ {i} ^ {1}, e _ {i} ^ {2}\right) \in \mathcal {S}} \left\| \mathbf {e} _ {i} ^ {1} - \mathbf {e} _ {i} ^ {2} \right\| + \sum_ {\left(e _ {i ^ {\prime}} ^ {1}, e _ {j ^ {\prime}} ^ {2}\right) \in \mathcal {S} ^ {-}} R e L U \left(\alpha - \left\| \mathbf {e} _ {i ^ {\prime}} ^ {1} - \mathbf {e} _ {j ^ {\prime}} ^ {2} \right\|\right), \tag {1}
$$

where  $S^{-}$  denotes the sampled set of negative pairs.  $\alpha$  is the minimal margin allowed between entities in each negative entity pair.

On the other hand, the scoring function  $f_{s}$  can be also designed diversely. Most methods (Chen et al., 2017; Sun et al., 2017; Pei et al., 2019) choose TransE as their scoring function, i.e.,  $f_{s}(e_{i},r,e_{j}) = ||\mathbf{e}_{i} + \mathbf{r} - \mathbf{e}_{j}||$ ,  $(e_i,r,e_j)\in \mathcal{T}_1\cup \mathcal{T}_2$ . The corresponding loss is:

$$
\mathcal {L} _ {s} = \sum_ {\tau \in \mathcal {T} _ {1} \cup \mathcal {T} _ {2}} R e L U \left(f _ {s} (\tau) - \lambda\right) + \sum_ {\tau^ {\prime} \in \mathcal {T} _ {1} ^ {-} \cup \mathcal {T} _ {2} ^ {-}} R e L U \left(\lambda - f _ {s} \left(\tau^ {\prime}\right)\right), \tag {2}
$$

where  $\mathcal{T}_1^-$  and  $\mathcal{T}_2^-$  are negative triple sets.  $\mathcal{L}_s$  is a margin-based loss in which the distance  $d(\mathbf{e}_i + \mathbf{r},\mathbf{e}_j)$  in a positive triple should at least be smaller than  $\lambda \geq 0$ , while larger than  $\lambda$  for negative ones. Note that, the negative triples are usually generated by randomly replacing the head or tail entity of a positive triple. If we only look at the replaced entity, minimizing the above loss can be also understood as randomly pushing entities away from this entity. This phenomena has also been studied in Wang & Isola (2020).

Additionally, some graph neural network (GNN) based methods (Wang et al., 2018; Sun et al., 2020a; Wu et al., 2019) do not directly optimize  $f_{s}$ . They encode the semantic information inside graph convolution. Therefore, the output vectors of GNN will be regarded as entity embeddings to feed into  $f_{a}$ . For example, Ye et al. (2019); Wu et al. (2019); Sun et al. (2020a) leverage TransE as scoring function in the aggregation of relational neighbors.

# 2.2 UNDERSTANDING EEA

We illustrate how an EEA model works by an example. Let  $(e_x^1, e_y^2) \in \mathcal{G}_1 \times \mathcal{G}_2$  be a potentially-aligned entity pair. Each entity in this pair has only one neighbor, connected by the same relation  $r^1 = r^2$ . We assume that their neighbors are actually a pair of entities  $(e_i^1, e_i^2) \in S$ . Therefore, if an EEA model is well-trained and almost optimal, we should have  $\mathbf{e}_i^1 = \mathbf{e}_i^2$  (as  $\mathcal{L}_a$  is minimized) and  $\mathbf{r}^1 = \mathbf{r}^2$  (denoted by  $\mathbf{r}$  for simplicity). According to Equation 2, we have:

$$
\left| \left| f _ {s} \left(\mathbf {e} _ {x} ^ {1}, \mathbf {r}, \mathbf {e} _ {i} ^ {1}\right) \right| \right| \approx \left| \left| f _ {s} \left(\mathbf {e} _ {y} ^ {2}, \mathbf {r}, \mathbf {e} _ {i} ^ {2}\right) \right| \right| \leq \lambda . \tag {3}
$$

Take the scoring function of TransE as  $f_{s}$ , we then derive:

$$
\left| \left| \mathbf {e} _ {x} ^ {1} + \mathbf {r} - \mathbf {e} _ {i} ^ {1} \right| \right| \leq \lambda , \quad \left| \left| \mathbf {e} _ {y} ^ {2} + \mathbf {r} - \mathbf {e} _ {i} ^ {2} \right| \right| \leq \lambda . \tag {4}
$$

As  $\mathbf{e}_i^1 = \mathbf{e}_i^2$ , we can conclude that:

Proposition 1 (Discrepancy Bound). The representation difference of two potentially-aligned entities is bound by  $\epsilon$ , which is proportional to the hyper-parameter  $\lambda$ :

$$
\left| \left| \mathbf {e} _ {x} ^ {1} - \mathbf {e} _ {y} ^ {2} \right| \right| \leq \epsilon \propto \lambda . \tag {5}
$$

The above proposition suggests that decreasing the value of margin  $\lambda$  will tighten the feature discrepancy of entities in the underlying aligned entity pairs. However, we soon find that  $\lambda$  cannot be set as small as we want.

We consider a more complicated yet realistic example, where each entity in  $(e_x^1,e_y^2)$  has a considerable number of neighbors. We denote the corresponding triple sets of  $e_x^1,e_x^2$  as  $\mathcal{T}_{e_x}^1,\mathcal{T}_{e_y}^2$ , respectively. In this setting, a well-trained EEA model should satisfy that:

$$
\forall \tau \in \mathcal {T} _ {e _ {x}} ^ {1} \cup \mathcal {T} _ {e _ {y}} ^ {2}, \| f _ {s} (\tau) \| \leq \lambda . \tag {6}
$$

Evidently, TransE with a small margin is not sufficient to fully express the semantics contained in  $\mathcal{T}_{e_x}^1 \cup \mathcal{T}_{e_y}^2$ , which has already been explored by previous works (Trouillon et al., 2016; Kazemi & Poole, 2018; Sun et al., 2019). Some empirical statistics (Sun et al., 2018) also illustrate such results. However, enlarging the margin  $\lambda$  will bring significant variance between  $\mathbf{e}_x^1$  and  $\mathbf{e}_y^2$ .

![](images/80fa46f67521fc4df81af20cc2d2bf4b5bb181b02dff08125c2dc93fba9e1ef6.jpg)  
(a)

![](images/1d7460f2c4de8383ffdc5723c578031493b0bc9a697836b0b21d310075814fc1.jpg)  
(b)  
Figure 1: Example of different feature distributions. (a) Overall entity feature distributions of two KGs, i.e.,  $\mathbb{A}_E$ . The two distributions are nearly uniformly distributed and almost aligned (based on the EEA model RDGCN (Wu et al., 2019)). (b) The head entity feature distributions of relation "genre". The two distributions are only aligned partially. (c) Head entity feature distributions conditioned on "genre", i.e.,  $\mathbb{A}_{E_h|r_i}$  (based on NeoEA with RDGCN as EEA model, the same below). Two conditioned distributions are aligned as expected. (d) The head entity distributions conditioned on three different relations: "genre" (colors: <blue,orange>), "writer" (colors: <purple,pink>), "brithPlace" (colors: <green,red>). The distributions corresponding to the first two relations are overlapped, while a clear decision boundary between them and the last one is observed. (e) Triple feature distributions conditioned on relations "artist" (colors: <blue,orange>) and "musicalArtist" (colors: <purple,pink>), respectively. The distributions referred to sub-relation "musicalArtist" are covered by those corresponded to "artist".

![](images/e00ca50bec6881e5a650c12a3d0e95c33d0b9136ded607d21806c4ed7d63f5a8.jpg)  
(c)

![](images/9b82b604a7a6bda2f3469112fabb480e84b1ee145aef8d92f79d3c0ceb39dd56.jpg)  
(d)

![](images/c58be7ed4f322d110ad5c292e7a6abc2620958a7dfe444cb9a390d6e2cb56094.jpg)  
(e)

On the other hand, if the scoring function does not belong to the TransE family, e.g., it is neural-based like ConvE (Dettmers et al., 2018) or composition-based like ComplEx (Trouillon et al., 2016), both of which are fully expressive (Kazemi & Poole, 2018). In this case, entities with a large number of neighbors can be correctly modeled, while those with only few neighbors are less constrained. Therefore, those models allow even more diversity between  $\mathbf{e}_x^1$  and  $\mathbf{e}_y^2$ . We believe this is why they performed badly in EA task (Guo et al., 2019; Sun et al., 2020b).

In short, most existing works adopt the above implicit strategy to learn cross-KG representations for EA, which makes them struggled in balancing between the bound and the expressiveness. In this paper, we explore a new direction to explicitly align the feature distributions of two KGs, which ensures the embeddings tight and expressive.

# 3 NEURAL ONTOLOGY

# 3.1 BASIC NEURAL AXIOM AND NEURAL AXIOM ALIGNMENT

We start by defining the basic neural axiom:

Definition 2 (Basic Neural Axiom).

$$
\mathbb {A} _ {E} = \{\mathbf {e} \mid e \sim \mathcal {E} \}. \tag {7}
$$

Aligning the basic neural axioms  $\mathbb{A}_E^1$  and  $\mathbb{A}_E^2$  of two KGs is trivial, as we can take the advantages of existing domain adaptation methods (Ben-David et al., 2010; Ganin & Lempitsky, 2015a;b; Courty et al., 2017; Shen et al., 2018), which aims to learn domain-invariant representations for various tasks. We consider the adversarial learning based ones (Ganin & Lempitsky, 2015a; Shen et al., 2018). In this way, a KG discriminator is leveraged to distinguish entity representations of  $\mathcal{G}_1$  from those of  $\mathcal{G}_2$  (or vice versa), while the embeddings will try to confuse the discriminator. Therefore, the same semantics in two KGs shall be encoded in the same way into the embeddings to fool the discriminator.

Specifically, if we regard two KGs  $\mathcal{G}_1$ ,  $\mathcal{G}_2$  as two different domains, and their embedding vectors as "learnable features", we can align the above axioms by an empirical Wasserstein distance based loss (Arjovsky et al., 2017; Shen et al., 2018):

$$
\mathcal {L} _ {\mathbb {A} _ {E}} = \mathbb {E} _ {\mathbb {A} _ {E ^ {1}}} [ f _ {w} (\mathbf {e}) ] - \mathbb {E} _ {\mathbb {A} _ {E ^ {2}}} [ f _ {w} (\mathbf {e}) ], \tag {8}
$$

where  $f_{w}$  is the learnable domain critic that maps the embedding vector to a scalar value. As suggested in (Arjovsky et al., 2017), the empirical Wasserstein distance can be approximated by maximizing  $\mathcal{L}_{\mathbb{A}_E}$ , if the parameterized family of  $f_{w}$  are all 1-Lipschitz.

However, from Figure 1a, we observe that the trained embeddings are nearly uniformly distributed in the feature space, which we can also derive from Equation 1 and Equation 2. Recall that the alignment loss  $\mathcal{L}_a$  consists of two terms. The first is

$$
\sum_ {\left(e _ {i} ^ {1}, e _ {i} ^ {2}\right) \in \mathcal {S}} \left| \left| \mathbf {e} _ {i} ^ {1} - \mathbf {e} _ {i} ^ {2} \right| \right|, \tag {9}
$$

which aims to minimize the difference of embeddings for each positive pair. The cardinality of  $\mathcal{S}$  is usually small. But this contrastive requires a large size of negative samples, which means that  $||\mathcal{S}|| \ll ||\mathcal{S}^{-}||$ . Therefore, the model more focuses on the second term

$$
\sum_ {\left(e _ {i ^ {\prime}} ^ {1}, e _ {j ^ {\prime}} ^ {2}\right) \in \mathcal {S} ^ {-}} R e L U \left(\alpha - \left| \left| \mathbf {e} _ {i ^ {\prime}} ^ {1} - \mathbf {e} _ {j ^ {\prime}} ^ {2} \right| \right|\right), \tag {10}
$$

of which the main target is to randomly push the embeddings of different entities away from each other. Furthermore,  $\mathcal{L}_s$  is also a contrastive loss, and has a similar effect on maximizing the pairwise distance between each positive entity and its corresponding sampled negative ones. Therefore, we conclude that:

Proposition 2 (Uniformity). The entity embeddings tend to be uniformly distributed in feature space as an EEA model is optimized.

The above proposition suggests that only aligning the basic axioms may be insufficient to facilitate EEA. Hence, we leverage conditional neural axioms which are more specific and expressive.

# 3.2 CONDITIONAL NEURAL AXIOM

Conditional neural axioms describe the entity (or triple) feature distributions under specific semantic conditions.

Definition 3 (Conditional Neural Axioms).

$$
\mathbb {A} _ {E _ {h} | r _ {i}} = \left\{\mathbf {e} \mid \mathbf {r} _ {i}, e \sim \{e \mid \forall e ^ {\prime}, (e, r _ {i}, e ^ {\prime}) \in \mathcal {T} \} \right\}
$$

$$
\mathbb {A} _ {E _ {h, t} \mid r _ {i}} = \left\{\left(\mathbf {e} _ {h}, \mathbf {e} _ {t}\right) \mid \mathbf {r} _ {i}, \left(e _ {h}, e _ {t}\right) \sim \left\{\left(e _ {h}, e _ {t}\right) \mid \left(e _ {h}, r _ {i}, e _ {t}\right) \in \mathcal {T} \right\} \right\} \tag {11}
$$

where  $\mathbb{A}_{E_h|r_i}$  denotes the head entities feature distribution conditioned on the relation embedding  $\mathbf{r}_i$ , the similar to  $\mathbb{A}_{E_{h,t}|r_i}$  (we reduce  $(\mathbf{e}_h,\mathbf{r}_i,\mathbf{e}_t)|\mathbf{r}_i$  to  $(\mathbf{e}_h,\mathbf{e}_t)|\mathbf{r}_i$  for simplicity).

Numerous methods are proposed to process the neural conditioning operation, ranging from addition and concatenation (Mirza & Osindero, 2014; Wang et al., 2014; Yang et al., 2015), to matrix multiplication (Lin et al., 2015a; Ji et al., 2015; Nguyen et al., 2016). Comparing with elaborating this operation, we value more on its common merit, which can be understood as projecting the entities to a relation-specific subspace (Wang et al., 2014; Lin et al., 2015a; Nguyen et al., 2016). Hence, the corresponding feature distributions conditioned on different relation embeddings become discriminative, rather than almost uniformly distributed in original feature space (Lin et al., 2015a).

Furthermore, conditional neural axioms capture high-level ontology knowledge.

Theorem 1 (Expressiveness). Aligning the conditional neural axioms minimizes the discrepancy of two KGs at ontology level.

Proof. See Appendix A for details. We take  $\mathbb{A}_{E_h|r_i}$  for example, which can summarize the empirical "Object Property Domain" axiom of  $r_i$  in OWL2 (Baader et al., 2005). Supposed there exists such an axiom that states the head entities of  $r_i$  should belong to some specific class  $c$  (e.g., only head entities under class "Person" have the relation "birthPlace"). We further suppose that there exists a classifier  $f_c(\mathbf{e})\in [0,1]$ , such that  $f_{c}(\mathbf{e}_{j}) = 1$  if head entity  $e_j$  belongs to class  $c$ , and 0 otherwise. Then, with the knowledge of the given axiom, one may derive the following rule:

$$
\forall e \in \{e | \forall e ^ {\prime}, (e, r _ {i}, e ^ {\prime}) \in \mathcal {T} _ {1} \cup \mathcal {T} _ {2} \}, f _ {c} (\mathbf {e}) = 1, \tag {12}
$$

which is equivalent to:

$$
\mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {1}} \left[ f _ {c} (\mathbf {e}) \right] = \mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {2}} \left[ f _ {c} (\mathbf {e}) \right] = 1, \tag {13}
$$

both of which means that all head entities of  $r_i$  in either KG should be correctly classified to  $c$ . Then, we have:

$$
\mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {1}} \left[ f _ {c} (\mathbf {e}) \right] - \mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {2}} \left[ f _ {c} (\mathbf {e}) \right] = 0. \tag {14}
$$

In fact, we do not have such knowledge about  $r_i$  and class  $c$ , instead we can leverage a neural function  $f_{c'}(\mathbf{e}|\mathbf{r}_i)$  to empirically estimate  $f_c$ . In this way,  $\mathbb{A}_{E_h|r_i}^s$  and  $\mathbb{A}_{E_h|r_i}^t$  are supposed to be aligned to minimize the loss corresponding to the above rule. Therefore, we deduce this problem back to a similar form to Equation 8, i.e.,

$$
\mathcal {L} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}}} = \mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {1}} \left[ f _ {c ^ {\prime}} (\mathbf {e} \mid \mathbf {r} _ {i}) \right] - \mathbb {E} _ {\mathbb {A} _ {E _ {h} \mid r _ {i}} ^ {2}} \left[ f _ {c ^ {\prime}} (\mathbf {e} \mid \mathbf {r} _ {i}) \right], \tag {15}
$$

which suggests that aligning the above conditional neural axioms can minimize the discrepancy of potential "Object Property Domain" axioms between two KGs.

![](images/3059419fd14857729f054847197c2da65f29cfb4b3e78bdec5fa9bdd44c362c3.jpg)

Example 1 (OWL2 axiom: ObjectPropertyDomain). As shown in Figure 1b and Figure 1c, we assume that the head entity of relation "genre" are under class "Work of Art" (although it does not exist in the dataset). It is clear that the head entity feature distributions are only partially aligned in Figure 1b, while those in Figure 1c are matched well.

In Figure 1d we illustrate a more complicated example. The head entities of relations "genre" and "writer" mainly belong to "Work of Art", which show overlapped distributions (blue-orange, pink-purple) in the figure. By contrast, there exists a clear decision boundary between them and the distributions conditioned on relation "birthPlace" (red-green), as the head entities of relation "birthPlace" are under class "Person".

Example 2 (OWL2 axiom: SubObjectPropertyOf). We consider two relations "musicalArtist" and "artist" as example, where the former one is the sub-relation of the later one. In Figure 1e, the triple distributions conditioned on "musicalArtist" (pink-purple) are covered by those conditioned on "artist" (orange-blue).

# 4 EXPERIMENTS

In this section, we empirically verify the effectiveness of NeoEA by a series of experiments, with several state-of-the-art methods as baselines.

# 4.1 IMPLEMENTATION

We illustrate the implementation of NeoEA in Algorithm 1. The whole framework is based on the OpenEA project (Sun et al., 2020b), which includes the implementations of latest EEA methods. Specifically, we implemented neural ontology as an external module, based on which we modified only the initialization of the original project. In this sense,, the EEA methods were unaware of the existence of neural ontologies. Furthermore, we kept the optimal hyper-parameter settings in OpenEA to ensure fair comparison. Please see Appendix B for the details.

We selected several best-performing and representative methods as our baselines:

- BootEA (Sun et al., 2018), a TransE-based EEA model with only structure data.  
- SEA (Pei et al., 2019), a TransE-based model with both structure and attribute data.  
- RSN (Guo et al., 2019), an RNN-based EEA model with only structure data.  
- RDGCN (Wu et al., 2019), a GCN-based model with both structure and attribute data.

The data distributions of some previous benchmarks such as JAPE (Sun et al., 2017) and BootEA (Sun et al., 2018) are clearly different from those of real-world KGs, which means that conducting experiments on those benchmarks cannot reflect the realistic performance of an EEA model (Guo et al., 2019; Sun et al., 2020b). Therefore, we consider the latest benchmark (Sun et al., 2020b), which consists of four sub-datasets, with two different density settings. Specifically, "D-W", "D-Y" denote "DBpedia (Auer et al., 2007)-WikiData (Vrandecic & Krötzsch, 2014)", "DBpedia-YAGO (Fabian

et al., 2007), respectively. "EN-DE" and "EN-FR" denote two cross-lingual datasets, both of which are sampled from DBpedia. "V1" denotes the sampled KGs having the similar distributions as the original KGs, while "V2" denotes the sampled KGs with doubled density. For detail statistics, please refer to Sun et al. (2020b).

# Algorithm 1 NeoEA

1: Input: two KGs  $\mathcal{G}_1, \mathcal{G}_2$ , the alignment seed set  $\mathcal{S}$ , the EEA model  $\mathcal{M}(f_s, f_a)$ , number of steps for NeoEA  $n$ ;  
2: Initialize all variables;  
3: repeat  
4: for  $i := 1$  to  $n$  do  
5: Sample sub-KGs from respective KGs  $\mathcal{G}_1, \mathcal{G}_2$ ;  
6: Compute the Wasserstein distance based loss  $\mathcal{L}_w$  for each pair of neural axioms;  
7: Optimize the Wasserstein distance critic  $f_{w}$  by maximizing  $\mathcal{L}_w$ .  
8: end for  
9: Sample sub-KGs from respective KGs  $\mathcal{G}_1, \mathcal{G}_2$ ;  
10: Compute Wasserstein distance based loss  $\mathcal{L}_w$  for each pair of neural axioms;  
11: Compute the losses  $\mathcal{L}_r, \mathcal{L}_s$  of the EEA model  $\mathcal{M}$ ;  
12: Optimize the EEA model and embeddings by minimizing  $\mathcal{L}_r, \mathcal{L}_s, \mathcal{L}_w$ ;  
13: until the alignment loss on validation set converged.

# 4.2 EMPIRICAL COMPARISONS

The main results are shown in Table 1, from which we find that: (1) The performance of four baseline models varied from different datasets, but all of them gained improvement with NeoEA. (2) The performance improvement on SEA and RDGCN was more significant than that on BootEA and RSN, as both BootEA and RSN are not typical EEA models. BootEA has a sophisticated bootstrapping procedure, which may be difficult to be injected with NeoEA. RSN tries to capture long-term dependencies among entities and relations. The complicated objective may be conflict with NeoEA more or less. However, on some datasets (e.g., EN-DE, V1), we still observe relatively significant improvement. Therefore, we believe the performance of these two models can be further refined through a joint hyper-parameter turning with NeoEA, which we leave to future work.

Table 1: Entity alignment results (5-fold cross-validation).  

<table><tr><td rowspan="2"></td><td rowspan="2">Models</td><td colspan="3">V1-Original</td><td colspan="3">V1-NeoEA</td><td colspan="3">V2-Original</td><td colspan="3">V2-NeoEA</td></tr><tr><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td></tr><tr><td rowspan="4">EN-FR</td><td>BootEA</td><td>.507</td><td>.718</td><td>.603</td><td>.521</td><td>.733</td><td>.617</td><td>.660</td><td>.850</td><td>.745</td><td>.665</td><td>.853</td><td>.749</td></tr><tr><td>SEA</td><td>.280</td><td>.530</td><td>.397</td><td>.320</td><td>.584</td><td>.443</td><td>.360</td><td>.651</td><td>.494</td><td>.375</td><td>.666</td><td>.508</td></tr><tr><td>RSN</td><td>.393</td><td>.595</td><td>.487</td><td>.399</td><td>.597</td><td>.490</td><td>.579</td><td>.759</td><td>.662</td><td>.583</td><td>.760</td><td>.666</td></tr><tr><td>RDGCN</td><td>.755</td><td>.854</td><td>.800</td><td>.775</td><td>.868</td><td>.817</td><td>.847</td><td>.919</td><td>.880</td><td>.864</td><td>.933</td><td>.896</td></tr><tr><td rowspan="4">EN-DE</td><td>BootEA</td><td>.675</td><td>.820</td><td>.740</td><td>.676</td><td>.820</td><td>.740</td><td>.833</td><td>.912</td><td>.869</td><td>.834</td><td>.916</td><td>.870</td></tr><tr><td>SEA</td><td>.530</td><td>.718</td><td>.617</td><td>.586</td><td>.766</td><td>.668</td><td>.606</td><td>.779</td><td>.687</td><td>.637</td><td>.800</td><td>.712</td></tr><tr><td>RSN</td><td>.587</td><td>.752</td><td>.662</td><td>.600</td><td>.759</td><td>.673</td><td>.791</td><td>.890</td><td>.837</td><td>.794</td><td>.892</td><td>.839</td></tr><tr><td>RDGCN</td><td>.830</td><td>.895</td><td>.859</td><td>.846</td><td>.908</td><td>.874</td><td>.833</td><td>.891</td><td>.860</td><td>.849</td><td>.902</td><td>.874</td></tr><tr><td rowspan="4">D-W</td><td>BootEA</td><td>.572</td><td>.744</td><td>.649</td><td>.579</td><td>.753</td><td>.658</td><td>.821</td><td>.926</td><td>.867</td><td>.822</td><td>.926</td><td>.869</td></tr><tr><td>SEA</td><td>.360</td><td>.572</td><td>.458</td><td>.389</td><td>.608</td><td>.490</td><td>.567</td><td>.770</td><td>.660</td><td>.588</td><td>.784</td><td>.677</td></tr><tr><td>RSN</td><td>.441</td><td>.615</td><td>.521</td><td>.450</td><td>.624</td><td>.530</td><td>.723</td><td>.854</td><td>.782</td><td>.729</td><td>.858</td><td>.787</td></tr><tr><td>RDGCN</td><td>.515</td><td>.669</td><td>.584</td><td>.527</td><td>.671</td><td>.592</td><td>.623</td><td>.757</td><td>.684</td><td>.632</td><td>.760</td><td>.690</td></tr><tr><td rowspan="4">D-Y</td><td>BootEA</td><td>.739</td><td>.849</td><td>.788</td><td>.756</td><td>.859</td><td>.797</td><td>.958</td><td>.984</td><td>.969</td><td>.958</td><td>.984</td><td>.969</td></tr><tr><td>SEA</td><td>.500</td><td>.706</td><td>.591</td><td>.549</td><td>.752</td><td>.638</td><td>.899</td><td>.950</td><td>.923</td><td>.917</td><td>.959</td><td>.936</td></tr><tr><td>RSN</td><td>.514</td><td>.655</td><td>.580</td><td>.522</td><td>.663</td><td>.588</td><td>.933</td><td>.974</td><td>.951</td><td>.935</td><td>.976</td><td>.953</td></tr><tr><td>RDGCN</td><td>.931</td><td>.969</td><td>.949</td><td>.941</td><td>.972</td><td>.955</td><td>.936</td><td>.966</td><td>.950</td><td>.940</td><td>.970</td><td>.953</td></tr></table>

The results improved most are boldfaced.

# 4.3 ABLATION STUDY

We designed an ablation study which is expected to empirically prove some claims in Section 3. We choose the current state-of-the-art model RDGCN as our baseline. As shown in Table 2, "Full"

Table 2: Results of ablation study based on the best-performing model RDGCN, on V1 datasets.  

<table><tr><td rowspan="2">Models</td><td colspan="3">EN-FR</td><td colspan="3">EN-DE</td><td colspan="3">D-W</td><td colspan="3">D-Y</td></tr><tr><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td><td>H@1</td><td>H@5</td><td>MRR</td></tr><tr><td>Full</td><td>.775</td><td>.868</td><td>.817</td><td>.846</td><td>.908</td><td>.874</td><td>.527</td><td>.671</td><td>.592</td><td>.941</td><td>.972</td><td>.955</td></tr><tr><td>Partial</td><td>.771</td><td>.863</td><td>.813</td><td>.840</td><td>.900</td><td>.871</td><td>.523</td><td>.669</td><td>.590</td><td>.936</td><td>.971</td><td>.952</td></tr><tr><td>Basic</td><td>.755</td><td>.853</td><td>.799</td><td>.827</td><td>.895</td><td>.858</td><td>.512</td><td>.656</td><td>.578</td><td>.931</td><td>.969</td><td>.948</td></tr><tr><td>Original</td><td>.755</td><td>.854</td><td>.800</td><td>.830</td><td>.895</td><td>.859</td><td>.515</td><td>.669</td><td>.584</td><td>.931</td><td>.969</td><td>.949</td></tr></table>

![](images/f47f48aba865938819cb2a9be1b2a83838e407b3ea31c8b84717b46369a64366.jpg)  
Figure 2: Normalized histograms of alignment rankings on EN-FR, V1 (left, long-tail entities; right, popular entities).

Table 3: Average ranking improvement.  

<table><tr><td>Datasets</td><td>Overall</td><td>Popular</td><td>Long-tail</td></tr><tr><td>EN-FR</td><td>63.5</td><td>36.9</td><td>116.7</td></tr><tr><td>EN-DE</td><td>13.0</td><td>8.1</td><td>23.4</td></tr><tr><td>D-W</td><td>43.5</td><td>34.5</td><td>61.3</td></tr><tr><td>D-Y</td><td>119.3</td><td>59.2</td><td>214.2</td></tr></table>

denote NeoEA with full set of neural axioms. "Partial" denotes NeoEA that removed the conditional triple axioms. We further removed the conditional entity axioms from "Partial" to construct "Basic", and the last one "Original" denotes the original EEA model. From the results we observe that: (1) Aligning basic axioms was less effective or even harmful to the model, which verifies Proposition 2 (Uniformity). (2) Aligning only a part of conditional axioms  $\mathbb{A}_{E_h|r_i}$ ,  $\mathbb{A}_{E_t|r_i}$  that describe entity feature distributions conditioned on relation representations was significantly helpful for the model. (3) Additional improvement was observed on the model with the full conditional axioms. Note that, the improvement from "Partial" to "Full" was not as significant as that from "Basic" to "Partial". This is because that the conditional triple axioms mainly describe the axioms between relations (see Appendix A). Due to the sampling strategy of the existing datasets, the number of relations is relatively small. Few correlated relation pairs exist in the datasets, resulting in limited improvement with conditional triple neural axiom alignment.

# 4.4 FURTHER ANALYSIS ON THE BOUND

We have shown that the discrepancy between each underlying aligned pair is bounded by  $\epsilon$  associated with  $\lambda$ , in Section 2. But we still expect to obverse empirical statistics to verify this point. To this end, we manually split the entities into two groups: (1) long-tail entities, which are disconnected to seeds and have at most two neighbors; (2) popular entities, the remaining. We draw the histograms of alignment rankings w.r.t. respective groups based on the EEA model SEA. From Figure 2, we can find that the proportion of the inexact alignments (i.e., ranking  $>5$ ) for long-tail entities is evidently larger than that of popular entities, especially for the bins [50, 100]. This verified that the long-tail entities are less constrained compared to those popular entities. Furthermore, with NeoEA, the rankings of those long-tail entities were also improved more significantly compared with those of popular entities, which empirically proved that NeoEA tightened the representation discrepancy of those entities that were less restrained. We report the average ranking improvement on four datasets (V1) in Table 3, which shows consistent observations.

# 5 CONCLUSION

In this paper, we proposed a new approach to learn representations for entity alignment. We proved its expressiveness theoretically and demonstrated its efficiency by conducting experiments on the latest benchmarks. We observed that four state-of-the-art EEA methods gained evident improvements with NeoEA. Finally, we showed that the proposed conditional neural axioms are the key to improve the performance of current EEA methods.

# REFERENCES

Martín Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein GAN. CoRR, 2017.  
Soren Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary G. Ives. Dbpedia: A nucleus for a web of open data. In ISWC, 2007.  
Franz Baader, Sebastian Brandt, and Carsten Lutz. Pushing the EL envelope. In *IJCAI*, 2005.  
Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Mach. Learn., 2010.  
Muhao Chen, Yingtao Tian, Mohan Yang, and Carlo Zaniolo. Multilingual knowledge graph embeddings for cross-lingual knowledge alignment. In *IJCAI*, 2017.  
Nicolas Courty, Rémi Flamary, Amaury Habrard, and Alain Rakotomamonjy. Joint distribution optimal transportation for domain adaptation. In NeurIPS, 2017.  
Tim Dettmers, Pasquale Minervini, Pontus Stenetorp, and Sebastian Riedel. Convolutional 2D knowledge graph embeddings. In AAAI, 2018.  
MS Fabian, Kasneci Gjergji, WEIKUM Gerhard, et al. Yago: A core of semantic knowledge unifying wordnet and wikipedia. In WWW, 2007.  
Matthias Fey, Jan Eric Lenssen, Christopher Morris, Jonathan Masci, and Nils M. Kriege. Deep graph matching consensus. In ICLR, 2020.  
Yaroslav Ganin and Victor S. Lempitsky. Unsupervised domain adaptation by backpropagation. In ICML, 2015a.  
Yaroslav Ganin and Victor S. Lempitsky. Unsupervised domain adaptation by backpropagation. In Francis R. Bach and David M. Blei (eds.), ICML, 2015b.  
Lingbing Guo, Zequn Sun, and Wei Hu. Learning to exploit long-term relational dependencies in knowledge graphs. In ICML, 2019.  
Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. Knowledge graph embedding via dynamic mapping matrix. In ACL, 2015.  
Seyed Mehran Kazemi and David Poole. Simple embedding for link prediction in knowledge graphs. In NeurlIPS, Montreal, Canada, 2018.  
Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. Learning entity and relation embeddings for knowledge graph completion. In AAAI, 2015a.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, 2014.  
Dat Quoc Nguyen, Kairit Sirts, Lizhen Qu, and Mark Johnson. STransE: A novel embedding model of entities and relationships in knowledge bases. In NAACL, San Diego, USA, 2016.  
Shichao Pei, Lu Yu, Robert Hoehndorf, and Xiangliang Zhang. Semi-supervised entity alignment via knowledge graph embedding with awareness of degree difference. In WWW, 2019.  
Jian Shen, Yanru Qu, Weinan Zhang, and Yong Yu. Wasserstein distance guided representation learning for domain adaptation. In AAAI, 2018.  
Zequn Sun, Wei Hu, and Chengkai Li. Cross-lingual entity alignment via joint attribute-preserving embedding. In ISWC, 2017.  
Zequn Sun, Wei Hu, Qingheng Zhang, and Yuzhong Qu. Bootstrapping entity alignment with knowledge graph embedding. In *IJCAI*, 2018.  
Zequn Sun, Chengming Wang, Wei Hu, Muhao Chen, Jian Dai, Wei Zhang, and Yuzhong Qu. Knowledge graph alignment network with gated multi-hop neighborhood aggregation. In AAAI, 2020a.

Zequn Sun, Qingheng Zhang, Wei Hu, Chengming Wang, Muhao Chen, Farahnaz Akrami, and Chengkai Li. A benchmarking study of embedding-based entity alignment for knowledge graphs. PVLDB, 2020b.  
Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. Rotate: Knowledge graph embedding by relational rotation in complex space. In ICLR, 2019.  
Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex embeddings for simple link prediction. In ICML, 2016.  
Bayu Distiawan Trsedya, Jianzhong Qi, and Rui Zhang. Entity alignment between knowledge graphs using attribute embeddings. In AAAI, 2019.  
Denny Vrandecic and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Communications of the ACM, 57, 2014.  
Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020.  
Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. Knowledge graph embedding by translating on hyperplanes. In AAAI, 2014.  
Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang. Cross-lingual knowledge graph alignment via graph convolutional networks. In EMNLP, 2018.  
Yuting Wu, Xiao Liu, Yansong Feng, Zheng Wang, Rui Yan, and Dongyan Zhao. Relation-aware entity alignment for heterogeneous knowledge graphs. In *IJCAI*, 2019.  
Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations for learning and inference in knowledge bases. In ICLR, 2015.  
Rui Ye, Xin Li, Yujie Fang, Hongyu Zang, and Mingzhong Wang. A vectorized relational graph convolutional network for multi-relational network alignment. In *IJCAI*, 2019.  
Hao Zhu, Ruobing Xie, Zhiyuan Liu, and Maosong Sun. Iterative entity alignment via joint knowledge embeddings. In *IJCAI*, 2017.
