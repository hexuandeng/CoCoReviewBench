# LEARNING FROM PROTEIN STRUCTURE WITH GEOMETRIC VECTOR PERCEPTRONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Learning on 3D structures of large biomolecules is emerging as a distinct area in machine learning, but there has yet to emerge a unifying network architecture that simultaneously leverages the geometric and relational aspects of the problem domain. To address this gap, we introduce geometric vector perceptrons, which extend standard dense layers to operate on collections of Euclidean vectors. Graph neural networks equipped with such layers are able to perform both geometric and relational reasoning on efficient representations of macromolecules. We demonstrate our approach on two important problems in learning from protein structure: model quality assessment and computational protein design. Our approach improves over existing classes of architectures on both problems, including state-of-the-art convolutional neural networks and graph neural networks.

# 1 INTRODUCTION

Many efforts in structural biology aim to predict, or derive insights from, the structure of a macromolecule (such as a protein, RNA, or DNA), represented as a set of positions associated with atoms or groups of atoms in 3D Euclidean space. These problems can often be framed as functions mapping the input domain of structures to some property of interest—for example, predicting the quality of a structural model or determining whether two molecules will bind. Thanks to their importance and difficulty, such problems, which we broadly refer to as learning from structure, have recently developed into an exciting and promising application area for deep learning (Graves et al., 2020; Ingraham et al., 2019; Pereira et al., 2016; Townshend et al., 2019; Won et al., 2019).

Successful applications of deep learning are often driven by techniques that leverage the problem structure of the domain—for example, convolutions in computer vision (Cohen & Shashua, 2016) and attention in natural language processing (Vaswani et al., 2017). What are the relevant considerations in the domain of learning from structure? Using proteins as the most common example, we have on the one hand the arrangement and orientation of the amino acids in space, which govern the dynamics and function of the molecule (Berg et al., 2002). On the other hand, proteins also possess relational structure in terms of their amino-acid sequence and the residue-residue interactions that mediate the aforementioned protein properties (Hammes-Schiffer & Benkovic, 2006). We refer to these as the geometric and relational aspects of the problem domain, respectively.

Recent state-of-the-art methods for learning from structure are successful by leveraging one of these two aspects. Commonly, such methods either employ graph neural networks (GNNs), which are expressive in terms of relational reasoning (Battaglia et al., 2018), or convolutional neural networks (CNNs), which operate directly on the geometry of the structure. Here, we present a unifying architecture that bridges these two families of methods to leverage both aspects of the problem domain.

We do so by introducing geometric vector perceptrons (GVPs), a drop-in replacement for standard multi-layer perceptrons (MLPs) in aggregation and feed-forward layers of GNNs. GVPs operate directly on both scalar and geometric features—features that transform as a vector under a rotation of spatial coordinates. GVPs therefore allow for the embedding of geometric information at nodes and edges without reducing such information to scalars that may not fully capture complex geometry. We postulate that our approach makes it easier for a GNN to learn functions whose significant features are both geometric and relational.

Our method (GVP-GNN) can be applied to any problem where the input domain is a structure of a single macromolecule or of molecules bound to one another. In this work, we specifically demonstrate our approach on two problems connected to protein structure: model quality assessment and computational protein design. Model quality assessment (MQA) aims to select the best structural model of a protein from a large pool of candidate structures and is a crucial step in protein structure prediction (Cheng et al., 2019). Computational protein design (CPD) is the conceptual inverse of structure prediction, aiming to infer an amino acid sequence that will fold into a given structure. Our method outperforms existing methods on both tasks.

# 2 RELATED WORK

ML methods for learning from protein structure largely fall into one of three types, operating on sequential, voxelized, or graph-structured representations of proteins. We briefly discuss each type and introduce state-of-the-art examples for MQA and CPD to set the stage for our experiments later.

Sequential representations In traditional models of learning from protein structure, each amino acid is represented as a feature vector using hand-crafted representations of the 3D structural environment, such as contact-based features (Olechnovič & Venclovas, 2017), orientations or positions collectively projected to local coordinates (Karasikov et al., 2019; Wang et al., 2018), and physics-inspired energy terms (O'Connell et al., 2018; Uziela et al., 2017). The structure is then viewed as a sequence of such feature vectors which can be fed into a 1D convolutional network, RNN, or dense feedforward network. Although these methods only indirectly represent the full 3D structure of the protein, a number of them, such as ProQ3D (Uziela et al., 2017), VoroMQA (Olechnovič & Venclovas, 2017), and SBROD (Karasikov et al., 2019), are competitive in assessments of MQA.

Voxelized representations In lieu of hand-crafted representations of structure, 3D convolutional neural networks (CNNs) can operate directly on the positions of atoms in space, encoded as occupancy maps in a voxelized 3D volume. The hierarchical convolutions of such networks are easily compatible with the detection of structural motifs, binding pockets, and the specific shapes of other important structural features, leveraging what we call the geometric aspect of the domain. The MQA methods 3DCNN (Derevyanko et al., 2018) and Ornate (Pagès et al., 2019) and a number of CPD methods (Anand et al., 2020; Zhang et al., 2019) exemplify the power of this approach.

Graph-structured representations A 3D protein structure can also be represented as a proximity graph over amino acid nodes, reducing the challenging task of representing a collective structural neighborhood in a single feature vector to that of representing individual edges. Graph neural networks (GNNs) can then perform complex relational reasoning over structures (Battaglia et al., 2018)—perhaps identifying key relationships among a set of amino acids, or flexible structural motifs best described as a connectivity pattern rather than a rigid shape. While some GNN methods, such as GraphQA (MQA) (Baldassarre et al., 2020), simply represent edges by their length, others, such as Structured Transformer (CPD) (Ingraham et al., 2019), indirectly encode the 3D geometry of the proximity graph in terms of relative orientations and other scalar features.

# 3 METHODS

The GNNs described in the previous section encode the 3D geometry of the protein by encoding vector features (such as node orientations and edge directions) in terms of rotation-invariant scalars, often by defining a local coordinate system at each node. We instead propose that these features be directly represented as geometric vectors—features in  $\mathbb{R}^3$  which transform appropriately under a change of spatial coordinates—at all steps of graph propagation. This conceptual shift has two important ramifications. First, the input representation is more efficient: instead of encoding the orientation of a node by its relative orientation with all of its neighbors, we only have to represent one absolute orientation per node. Second, it standardizes a global coordinate system across the entire structure, which allows geometric features to be directly propagated without transforming between local coordinates. We postulate this allows the GNN to more easily access global geometric properties of the structure. The key challenge with such a representation, however, is to perform graph propagation in a way that simultaneously preserves the full expressive power of the original

![](images/510a91bc7a677df3e7b814f0c065a5d48bad82851a35748df4a9a0770e37196f.jpg)  
Figure 1: (A) Schematic of the geometric vector perceptron illustrating Algorithm 1. Given a tuple of scalar and vector input features  $(\mathbf{s},\mathbf{V})$ , the perceptron computes an updated tuple  $(\mathbf{s}',\mathbf{V}')$ .  $\mathbf{s}'$  is a function of both  $\mathbf{s}$  and  $\mathbf{V}$ . (B) Illustration of the structure-based prediction tasks. In model quality assessment (top), the goal is to predict a quality score given the 3D structure of a candidate model. Individual atoms are represented as colored spheres. The quality score measures the accuracy of a candidate structure with respect to an experimentally determined structure (shown in gray). In computational protein design (bottom), the goal is to predict an amino acid sequence that would fold into a given protein backbone structure.

![](images/20b5fc37c60b2fa8d2d93720feb8a8dd62be0dbfc666ae3baba60ffb7312a7eb.jpg)

GNN while maintaining the rotation invariance provided by the scalar representations. We do so by introducing a new module, the geometric vector perceptron, to replace dense layers in a GNN.

# 3.1 GEOMETRIC VECTOR PERCEPTRONS

The geometric vector perceptron is a simple module for learning vector-valued and scalar-valued functions over geometric vectors and scalars. That is, given a tuple  $(\mathbf{s},\mathbf{V})$  of scalar features  $\mathbf{s}\in \mathbb{R}^n$  and vector features  $\mathbf{V}\in \mathbb{R}^{\nu \times 3}$ , we compute new features  $(\mathbf{s}',\mathbf{V}')\in \mathbb{R}^{m}\times \mathbb{R}^{\mu \times 3}$ . The computation is formally described in Algorithm 1 and illustrated in Figure 1A.

At its core, the GVP consists of two separate linear transformations  $\mathbf{W}_m$ ,  $\mathbf{W}_h$  for the scalar and vector features, followed by nonlinearities  $\sigma, \sigma^+$ . However, before the scalar features are transformed, we concatenate the  $L_2$  norm of the transformed vector features  $\mathbf{V}_h$ ; this allows us to extract rotation-invariant information from the input vectors  $\mathbf{V}$ . An additional linear transformation  $\mathbf{W}_{\mu}$  is inserted just before the vector nonlinearity to control the output dimensionality independently of the number of norms extracted.

Algorithm 1 Geometric vector perceptron  
Input: Scalar and vector features  $(\mathbf{s},\mathbf{V})\in \mathbb{R}^n\times \mathbb{R}^{\nu \times 3}$    
Output: Scalar and vector features  $(\mathbf{s}',\mathbf{V}')\in \mathbb{R}^{m}\times \mathbb{R}^{\mu \times 3}$ $h\gets \max (\nu ,\mu)$    
GVP:   
 $\begin{array}{rl} & {\mathbf{V}_h\leftarrow \mathbf{W}_h\mathbf{V}\quad \in \mathbb{R}^{h\times 3}}\\ & {\mathbf{V}_\mu \leftarrow \mathbf{W}_\mu \mathbf{V}_h\quad \in \mathbb{R}^{\mu \times 3}}\\ & {\mathrm{s}_h\leftarrow \| \mathbf{V}_h\| _2\mathrm{(row - wise)}\quad \in \mathbb{R}^h}\\ & {\mathrm{v}_\mu \leftarrow \| \mathbf{V}_\mu \| _2\mathrm{(row - wise)}\quad \in \mathbb{R}^\mu} \end{array}$ $\begin{array}{rl} & {\mathrm{s}_{h + n}\leftarrow \mathrm{concat}\left(\mathrm{s}_h,\mathbf{s}\right)\quad \in \mathbb{R}^{h + n}}\\ & {\mathrm{s}_m\leftarrow \mathbf{W}_m\mathrm{s}_{h + n} + \mathbf{b}\quad \in \mathbb{R}^m}\\ & {\mathbf{s}'\leftarrow \sigma \left(\mathrm{s}_m\right)\quad \in \mathbb{R}^m}\\ & {\mathbf{V}'\leftarrow \sigma^+ (\mathrm{v}_\mu)\odot \mathrm{V}_\mu ,(\mathrm{row - wise~multiplication})\quad \in \mathbb{R}^{\mu \times 3}} \end{array}$    
return (s',V')

The GVP is conceptually simple, yet provably possesses the desired properties of invariance/equivalence and expressiveness. First, the vector and scalar outputs of the GVP are equivariant and invariant, respectively, with respect to an arbitrary composition of rotations and reflections in

3D Euclidean space described by  $R$  i.e.,

$$
\operatorname {G V P} \left(\left(\mathbf {s}, R (\mathbf {V})\right)\right) = \left(\mathbf {s} ^ {\prime}, R \left(\mathbf {V} ^ {\prime}\right)\right) \tag {1}
$$

This is due to the fact that the only operations on vector-valued inputs are scalar multiplication, linear combination, and the  $L_{2}$  norm. We include a formal proof in Appendix A.

In addition, a GVP can approximate any continuous rotation- and reflection-invariant scalar-valued function of  $\mathbf{V}$ . More precisely, let  $G_{s}$  be a GVP defined with  $n,\mu = 0$  that is, the part of a GVP that transforms vector features to scalar features. Then  $G_{s}$  is able to  $\epsilon$ -approximate a function  $F:\mathbb{R}^{\nu \times 3}\to \mathbb{R}$  that is invariant with respect to rotations and reflections in 3D under mild assumptions.

Theorem. Let  $R$  describe an arbitrary rotation or reflection in  $\mathbb{R}^3$ . For  $\nu \geq 3$  let  $\Omega^\nu \subset \mathbb{R}^{\nu \times 3}$  be the set of all  $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_\nu]^T \in \mathbb{R}^{\nu \times 3}$  such that  $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$  are linearly independent and  $0 < ||\mathbf{v}_i||_2 \leq b$  for all  $i$  and some finite  $b > 0$ . Then for any continuous  $F: \Omega^\nu \to \mathbb{R}$  such that  $F(R(\mathbf{V})) = F(\mathbf{V})$  and for any  $\epsilon > 0$ , there exists a form  $f(\mathbf{V}) = \mathbf{w}^T G_s(\mathbf{V})$  such that  $|F(\mathbf{V}) - f(\mathbf{V})| < \epsilon$  for all  $\mathbf{V} \in \Omega^\nu$ .

We include a formal proof in Appendix A. As a corollary, a GVP with nonzero  $n,\mu$  is also able to approximate similarly-defined functions over the full input domain  $\mathbb{R}^n\times \mathbb{R}^{\nu \times 3}$

In addition to the GVP layer itself, we use a version of dropout that drops entire vector channels at random (as opposed to coordinates within vector channels). We also introduce layer normalization for the vector features as

$$
\mathbf {v} _ {i} \leftarrow \mathbf {v} _ {i} / \sqrt {\frac {1}{| \{j \} |} \sum_ {j} \| \mathbf {v} _ {j} \| _ {2} ^ {2}} \quad \forall i, \tag {2}
$$

where  $\mathbf{v}_i\in \mathbb{R}^3$  are the individual row vectors of the vector feature matrix  $\mathbb{R}^{\nu \times 3}$ . This vector layer norm has no trainable parameters, but we continue to use normal layer normalization on scalar channels with trainable parameters  $\gamma ,\beta$ .

We study our hypothesis that GVPs augment the geometric reasoning ability of GNNs on a synthetic dataset (Appendix B). The synthetic dataset allows us to control the function underlying the ground-truth label in order to explicitly separate geometric and relational aspects in different tasks. The GVP-GNN matches a CNN on a geometric task and a GNN on a relational task. However, when we combine the two tasks in one objective, the GVP-GNN does significantly better than either a GNN or a CNN.

# 3.2 REPRESENTATIONS OF PROTEINS

The main empirical validation of our architecture is its performance on two real-world tasks: model quality assessment (MQA) and computational protein design (CPD). These tasks, as illustrated in Figure 1B and described in detail in Section 4, are complementary in that one (MQA) predicts a global property and the other (CPD) predicts a property for each amino acid.

We represent a protein structure input as a proximity graph with a minimal number of scalar and vector features to fully specify the 3D structure of the molecule. A protein structure is a sequence of amino acids, where each amino acid consists of four backbone atoms $^2$  and a set of sidechain atoms located in 3D Euclidean space. Here we represent only the backbone because our MQA benchmark corresponds to the assessment of backbone structure. In CPD, the sidechains are by definition unknown.

Let  $X_{i}$  be the position of atom  $X$  in the  $i$ th amino acid. We represent backbone structure as a graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  where each node  $\mathfrak{v}_i\in \mathcal{V}$  corresponds to an amino acid and has embedding  $\mathbf{h}_{\mathfrak{v}}^{(i)}$  with the following features:

- Scalar features  $\{\sin, \cos\} \circ \{\phi, \psi, \omega\}$ , where  $\phi, \psi, \omega$  are the dihedral angles computed from  $\mathbf{C}_{i-1}$ ,  $\mathbf{N}_i$ ,  $\mathbf{C}\alpha_i$ ,  $\mathbf{C}_i$ , and  $\mathbf{N}_{i+1}$ .  
- The forward and reverse unit vectors in the directions of  $\mathbf{C}\alpha_{i + 1} - \mathbf{C}\alpha_{i}$  and  $\mathbf{C}\alpha_{i - 1} - \mathbf{C}\alpha_{i}$ , respectively.

- The unit vector in the imputed direction of  $\mathbf{C}\beta_{i} - \mathbf{C}\alpha_{i}$ . This is computed by assuming tetrahedral geometry and normalizing

$$
\sqrt {\frac {1}{3}} (\mathbf {n} \times \mathbf {c}) / | | \mathbf {n} \times \mathbf {c} | | _ {2} - \sqrt {\frac {2}{3}} (\mathbf {n} + \mathbf {c}) / | | \mathbf {n} + \mathbf {c} | | _ {2}
$$

where  $\mathbf{n} = \mathbf{N}_i - \mathbf{C}\alpha_i$  and  $\mathbf{c} = \mathbf{C}_i - \mathbf{C}\alpha_i$ . This vector, along with the forward and reverse unit vectors, unambiguously define the orientation of each amino acid residue.

- A one-hot representation of amino acid identity, when available.

The set of edges is  $\mathcal{E} = \{\mathbf{e}_{i\rightarrow j}\}_{i\neq j}$  for all  $i,j$  where  $\mathfrak{v}_i$  is among the  $k = 30$  nearest neighbors of  $\mathfrak{v}_j$  as measured by the distance between their  $\mathbf{C}\alpha$  atoms. Each edge has an embedding  $\mathbf{h}_e^{(i\rightarrow j)}$  with the following features:

- The unit vector in the direction of  $\mathrm{C}\alpha_{i} - \mathrm{C}\alpha_{j}$ .  
- The radial basis encoding of the distance  $\mathrm{RBF}(|\mathbf{C}\alpha_{i} - \mathbf{C}\alpha_{j}|_{2})$  
- A sinusoidal encoding of  $i - j$  as described in Vaswani et al. (2017), representing distance along the backbone.

In our notation, each feature vector  $\mathbf{h}$  is a concatenation of scalar and vector features as described above. Collectively, these features are sufficient for a complete description of the protein backbone.

# 3.3 NETWORK ARCHITECTURE

Our architecture is a GVP-augmented graph neural network (or GVP-GNN) that takes as input the protein graph defined above and performs graph propagation steps to update the node embeddings:

$$
\mathbf {h} _ {m} ^ {(i \rightarrow j)} := g \left(\operatorname {c o n c a t} \left(\mathbf {h} _ {\mathfrak {v}} ^ {(i)}, \mathbf {h} _ {e} ^ {(i \rightarrow j)}\right)\right) \tag {3}
$$

$$
\mathbf {h} _ {\mathfrak {v}} ^ {(j)} \leftarrow \text {L a y e r N o r m} \left(\mathbf {h} _ {\mathfrak {v}} ^ {(j)} + \frac {1}{\| \{i : \mathbf {e} _ {i \rightarrow j} \in E \} \|} \operatorname {D r o p o u t} \left(\sum_ {i: \mathbf {e} _ {i \rightarrow j} \in E} \mathbf {h} _ {m} ^ {(i \rightarrow j)}\right)\right) \tag {4}
$$

Here,  $g$  is a sequence of three GVPs. We do not update edge embeddings and do not use a global graph embedding. Between graph propagation steps, we also use a feed-forward point-wise update layer:

$$
\mathbf {h} _ {\mathfrak {v}} ^ {(i)} \leftarrow \text {L a y e r N o r m} \left(\mathbf {h} _ {\mathfrak {v}} ^ {(i)} + \text {D r o p o u t} \left(g \left(\mathbf {h} _ {\mathfrak {v}} ^ {(i)}\right)\right)\right) \tag {5}
$$

where  $g$  is a sequence of two GVPs. These graph propagation and feed-forward steps update the vector features at each node in addition to its scalar features.

In model quality assessment, we use three graph propagation steps and perform regression against the true quality score of a candidate structure, a global scalar property. To obtain a single global representation, we apply a node-wise GVP to reduce all node embeddings to scalars. We then average the representations across all nodes<sup>4</sup> and apply a final dense feed-forward network to output the network's prediction.

In computational protein design, the network learns a generative model over the space of protein sequences conditioned on the given backbone structure. Following Ingraham et al. (2019), we frame this as an autoregressive task and use a masked encoder-decoder architecture to capture the joint distribution over all positions: for each  $i$ , the network models the distribution at  $i$  based on the complete structure graph, as well as the sequence information at positions  $j < i$ . The encoder first performs three graph propagation steps on the structural information only. Then, sequence information is added to the graph, and the decoder performs three further graph propagation steps where incoming messages  $\mathbf{h}_m^{(i\rightarrow j)}$  for  $i\geq j$  are computed only with the encoder embeddings. Finally, we use one last GVP with 20-way scalar softmax output to predict the probability of the amino acids.

Further details regarding training and hyperparameters can be found in Appendix D.

# 4 DATASET AND EVALUATION

In this section, we set up the background and methodology for benchmarking the architecture on model quality assessment and protein design.

Model quality assessment Model quality assessment aims to select the best structural model of a protein from a large pool of candidate structures. The performance of different MQA methods is evaluated every two years in the community-wide Critical Assessment of Structure Prediction (CASP) (Cheng et al., 2019). For a number of recently solved but unreleased structures, called targets, structure generation programs produce a large number of candidate structures. MQA methods are evaluated by how well they predict the GDT-TS score of a candidate structure compared to the experimentally solved structure for that target. GDT-TS is a scalar measure of how similar two protein backbones are after global alignment (Zemla et al., 2001).

In addition to accurately predicting the absolute quality of a candidate structure, a good MQA method should also be able to accurately assess the relative model qualities among a pool of candidates for a given target so that the best ones can be selected, perhaps for further refinement. Therefore, MQA methods are commonly evaluated on two metrics: a global correlation between the predicted and ground truth scores, pooled across all targets, and the average per-target correlation among only the candidate structures for a specific target (Cao & Cheng, 2016; Derevyanko et al., 2018; Pagès et al., 2019). We follow this convention in our experiments.

We train and validate on the set of candidate structures generated in the CASP 5-10 assessments, which collectively contain 528 targets and 79200 candidate structures. For testing, we predict model quality for a total of 20880 stage 1 and stage 2 candidate structures from CASP 11 (84 targets) and 12 (40 targets). Further details on the dataset can be found in Appendix C.

Protein design Computational protein design is the conceptual inverse of structure prediction, aiming to infer an amino acid sequence that will fold into a given structure. In comparison to model quality assessment, computational protein design is more difficult to unambiguously benchmark, as some structures may correspond to a large space of sequences and others may correspond to none at all. Therefore, the proxy metric of native sequence recovery—splitting the set of all known native structures in the PDB and attempting to design the sequences corresponding to held-out structures—is typically used to benchmark CPD models (Li et al., 2014; O'Connell et al., 2018; Wang et al., 2018). Drawing an analogy between sequence design and language modelling, Ingraham et al. (2019) also evaluate the model perplexity on held-out native sequences. Both metrics rest on the implicit assumption that native sequences are optimized for their structures (Kuhlman & Baker, 2000) and should be assigned high probabilities.

To best approximate real-world applications that may require design of novel structures, the held-out evaluation set should bear minimal similarity to the training structures. We use the CATH 4.2 dataset curated by Ingraham et al. (2019) in which all available structures with  $40\%$  nonredundancy are partitioned by their CATH (class, architecture, topology/fold, homologous superfamily) classification. The canonical training, validation, and test splits consist of 18204, 608, and 1120 structures, respectively.

We also report results on TS50, an older test set of 50 native structures first introduced by Li et al. (2014). The smaller size of this benchmark allows a comparison to the computationally expensive physics-based calculations of the fixbb protocol in Rosetta, a software suite well-established in the structural biology community (Das & Baker, 2008). No canonical training and validation sets exist for TS50. In order to evaluate on TS50, we remove sequences with more than  $30\%$  similarity with any structure in TS50 from our training and validation sets and retrain our model.

# 5 EXPERIMENTS

Model quality assessment We compare our method against state-of-the-art methods on the CASP 11-12 test set in Table 1. These include the CNN methods 3DCNN (Derevyanko et al., 2018)

Table 1: Comparison with state-of-the-art methods on CASP 11 and 12 in terms of global (Glob) and mean per-target (Per) Pearson correlation coefficients (higher is better). Each method is classified as one of the three types discussed in Section 2. ProQ3D is set aside as the only method which additionally uses non-structure information. The top performing structure-only method for each metric is in bold, as is the top performing-method overall (if different). Our method generally improves over all other methods.  

<table><tr><td rowspan="3">Method</td><td rowspan="3">Type</td><td colspan="4">CASP 11</td><td colspan="4">CASP 12</td></tr><tr><td colspan="2">Stage 1</td><td colspan="2">Stage 2</td><td colspan="2">Stage 1</td><td colspan="2">Stage 2</td></tr><tr><td>Glob</td><td>Per</td><td>Glob</td><td>Per</td><td>Glob</td><td>Per</td><td>Glob</td><td>Per</td></tr><tr><td>Ours</td><td>GNN</td><td>0.84</td><td>0.66</td><td>0.87</td><td>0.45</td><td>0.79</td><td>0.73</td><td>0.82</td><td>0.62</td></tr><tr><td>3DCNN</td><td>CNN</td><td>0.59</td><td>0.52</td><td>0.64</td><td>0.40</td><td>0.49</td><td>0.44</td><td>0.61</td><td>0.51</td></tr><tr><td>Ornate</td><td>CNN</td><td>0.64</td><td>0.47</td><td>0.63</td><td>0.39</td><td>0.55</td><td>0.57</td><td>0.67</td><td>0.49</td></tr><tr><td>GraphQA</td><td>GNN</td><td>0.83</td><td>0.63</td><td>0.82</td><td>0.38</td><td>0.72</td><td>0.68</td><td>0.81</td><td>0.61</td></tr><tr><td>VoroMQA</td><td>Seq</td><td>0.69</td><td>0.62</td><td>0.65</td><td>0.42</td><td>0.46</td><td>0.61</td><td>0.61</td><td>0.56</td></tr><tr><td>SBROD</td><td>Seq</td><td>0.58</td><td>0.65</td><td>0.55</td><td>0.43</td><td>0.37</td><td>0.64</td><td>0.47</td><td>0.61</td></tr><tr><td>ProQ3D</td><td>Seq</td><td>0.80</td><td>0.69</td><td>0.77</td><td>0.44</td><td>0.67</td><td>0.71</td><td>0.81</td><td>0.60</td></tr></table>

and Ornate (Pagès et al., 2019), the GNN method GraphQA (Baldassarre et al., 2020), and three methods that use sequential representations—VoroMQA (Olechnović & Venclovas, 2017), SBROD (Karasikov et al., 2019), and ProQ3D (Uziela et al., 2017). All of these methods learn solely from protein structure, $^{6}$  with the exception of ProQ3D, which in addition uses sequence information on related proteins, which is not always available. We include ProQ3D because it is an improved version of the best single-model method in CASP 11 and CASP 12 (Uziela et al., 2017). Our method outperforms all other structural methods in both global and per-target correlation, and even performs better than ProQ3D on all but one benchmark.

We also evaluate DimeNet, a recent 3D-aware GNN architecture which achieves state-of-the-art on many small-molecule tasks (Klicpera et al., 2020), on CASP 11-12. DimeNet differs from our method in that it uses relative edge orientations to indirectly encode geometry into its message-passing operations. While this paradigm appears well-suited for the domain of learning from small molecules, DimeNet does not outperform any of the models in Table 1 on model quality assessment. See Appendix F for detailed results and discussion.

Recent MQA methods have most extensively been benchmarked on the CASP 11-12 datasets. However, for completeness, we also evaluate our method on CASP 13 (Table 2). Because of its recency, many target structures remain publicly unavailable. We use the stage 2 candidate structures of a subset of 20 targets previously used for benchmarking (Baldassarre et al., 2020). Our method achieves improved results over all other methods, including ProteinGCN (Sanyal et al., 2020), a more recent GNN method which was evaluated only on CASP 13. Because of the small sample size, we emphasize that these results, although promising, should be considered preliminary until further structures for CASP 13 are available.

Finally, because our architecture updates vector features along with scalar features at each node embedding, it is possible to visualize learned vector features in the intermediate layers of the trained MQA network. We show and discuss the interpretability of such features in Appendix E.

**Protein design** Our method achieves state-of-the-art performance on CATH 4.2, representing a substantial improvement both in terms of perplexity and sequence recovery over Structured Transformer (Ingraham et al., 2019), a GNN method which was trained using the same training and validation sets (Table 4). Following Ingraham et al. (2019), we report evaluation on short (100 or fewer amino acid residues) and single-chain subsets of the CATH 4.2 test set, containing 94 and 103 proteins, respectively, in addition to the full test set. Although Structured Transformer leverages an attention mechanism on top of a graph-structured representation of proteins, the authors note in ablation studies that removing attention appeared to increase performance. We therefore retrain and

Table 2: Performance on the 20 publicly available CASP 13 targets, stage 2 in terms of global and mean per-target Pearson correlation. As before, the top structure-only method is in bold, as is the top method overall (if different). Our method outperforms all other methods.  

<table><tr><td>Method</td><td>Type</td><td>Global</td><td>Per-target</td></tr><tr><td>Ours</td><td>GNN</td><td>0.881</td><td>0.685</td></tr><tr><td>ProteinGCN</td><td>GNN</td><td>0.723</td><td>0.603</td></tr><tr><td>VoroMQA</td><td>Seq</td><td>0.769</td><td>0.665</td></tr><tr><td>ProQ3D</td><td>Seq</td><td>0.849</td><td>0.671</td></tr></table>

Table 3: Relative performance of ablated methods and state-of-the-art GNNs compared with the GVP on sequence recovery (CPD) and mean correlation from CASP 11-12, stage 2 (MQA).  

<table><tr><td>Modification</td><td>MQA</td><td>CPD</td></tr><tr><td>Baseline GVP</td><td>100.0%</td><td>100.0%</td></tr><tr><td>MLP</td><td>92.0%</td><td>76.1%</td></tr><tr><td>Scalars only</td><td>94.3%</td><td>80.6%</td></tr><tr><td>No Wh</td><td>96.4%</td><td>92.3%</td></tr><tr><td>GraphQA</td><td>94.0%</td><td>-</td></tr><tr><td>Structured GNN</td><td>-</td><td>92.8%</td></tr></table>

Table 4: Performance on the CATH 4.2 test set and its short and single-chain subsets in terms of per-residue perplexity (lower is better) and recovery (higher is better). Recovery is reported as the median over all structures of the mean recovery of 100 sequences per structure. Our method performs better than Structured Transformer and a variant of it, Structured GNN, in which we replaced the attention mechanisms with standard graph propagation operations (see main text).  

<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="3">Perplexity</td><td colspan="3">Recovery %</td></tr><tr><td>Short</td><td>Single-chain</td><td>All</td><td>Short</td><td>Single-chain</td><td>All</td></tr><tr><td>Ours</td><td>GNN</td><td>7.10</td><td>7.44</td><td>5.29</td><td>32.1</td><td>32.0</td><td>40.2</td></tr><tr><td>Structured GNN</td><td>GNN</td><td>8.31</td><td>8.88</td><td>6.55</td><td>28.4</td><td>28.1</td><td>37.3</td></tr><tr><td>Structured Transformer</td><td>GNN</td><td>8.54</td><td>9.03</td><td>6.85</td><td>28.3</td><td>27.6</td><td>36.4</td></tr></table>

compare against a version of Structured Transformer with the attention layers replaced with standard graph propagation operations (Structured GNN). Our method also improves upon this model.

On the smaller test set TS50, we achieve  $44.9\%$  recovery compared to Rosetta's  $30\%$  and outperform methods based on each of the three classes of structural representations. Overall, we place 2nd out of 8 methods in terms of recovery (see Appendix F). However, the results for this test set should be taken with a grain of salt, given that the different methods did not use canonical training datasets.

Ablation studies The MQA and CPD methods we have compared against include a number of GNNs (GraphQA, ProteinGCN, Structured Transformer/GNN). We perform ablation studies to identify the aspects of the GVP which most contribute to our improvement over these state-of-the-art models (Table 3). Replacing the GVP with a vanilla MLP layer eliminates the view of geometric information as geometric vectors, and propagating only scalar features forces the model to learn scalar-valued, indirect representations of geometry. Both modifications result in considerable and similar decreases in performance, underscoring the importance of direct access to geometric information. Eliminating the intermediate transformation  $\mathbf{W}_h$  results in a slight but non-negligible decrease in performance. All of these elements contributed to our improvement over Structured GNN (on CPD) and GraphQA (on MQA). See Appendix F for a more detailed discussion.

# 6 CONCLUSION

In this work, we developed the first architecture designed specifically for learning on dual relational and geometric representations of 3D macromolecular structure. At its core, our method, GVP-GNN, augments graph neural networks with computationally simple layers that perform expressive geometric reasoning over Euclidean vector features. Our method possesses desirable theoretical properties and empirically outperforms existing architectures on learning quality scores and sequence designs, respectively, from protein structure.

In further work, we hope to apply our architecture to other important structural biology problem areas, including protein complexes, RNA structure, and protein-ligand interactions.

# REFERENCES

Namrata Anand, Raphael Ryuichi Eguchi, Alexander Derry, Russ B Altman, and Possu Huang. Protein sequence design with a learned potential. bioRxiv, 2020.  
Federico Baldassarre, David Menéndez Hurtado, Arne Elofsson, and Hossein Azizpour. GraphQA: Protein model quality assessment using graph convolutional network, 2020. URL https://openreview.net/forum?id=HyxgBerKwB.  
Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.  
Jeremy M Berg, John L Tymoczko, and Lubert Stryer. Biochemistry, W.H. Freeman and Company, 2002.  
Renzhi Cao and Jianlin Cheng. Protein single-model quality assessment by feature-based probability density functions. Scientific reports, 6:23990, 2016.  
Sheng Chen, Zhe Sun, Lihua Lin, Zifeng Liu, Xun Liu, Yutian Chong, Yutong Lu, Huiying Zhao, and Yuedong Yang. To improve protein sequence profile prediction through image captioning on pairwise residue distance map. Journal of Chemical Information and Modeling, 2019.  
Jianlin Cheng, Myong-Ho Choe, Arne Elofsson, Kun-Sop Han, Jie Hou, Ali HA Maghrabi, Liam J McGuffin, David Menéndez-Hurtado, Kliment Olechnovič, Torsten Schwede, et al. Estimation of model accuracy in casp13. Proteins: Structure, Function, and Bioinformatics, 87(12):1361-1377, 2019.  
Nadav Cohen and Amnon Shashua. Inductive bias of deep convolutional networks through pooling geometry. arXiv preprint arXiv:1605.06743, 2016.  
George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.  
Rhiju Das and David Baker. Macromolecular modeling with rosetta. Annu. Rev. Biochem., 77: 363-382, 2008.  
Georgy Derevyanko, Sergei Grudinin, Yoshua Bengio, and Guillaume Lamoureux. Deep convolutional networks for quality assessment of protein folds. Bioinformatics, 34(23):4046-4053, 2018.  
Jordan Graves, Jacob Byerly, Eduardo Priego, Naren Makkapati, S Vince Parish, Brenda Medellin, and Monica Berrondo. A review of deep learning methods for antibodies. Antibodies, 9(2):12, 2020.  
Sharon Hammes-Schiffer and Stephen J Benkovic. Relating protein motion to catalysis. Annu. Rev. Biochem., 75:519-541, 2006.  
John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for graph-based protein design. In Advances in Neural Information Processing Systems, pp. 15794-15805, 2019.  
Mikhail Karasikov, Guillaume Pagès, and Sergei Grudinin. Smooth orientation-dependent scoring function for coarse-grained protein quality assessment. Bioinformatics, 35(16):2801-2808, 2019.  
Johannes Klicpera, Janek Gro, and Stephan Gnnemann. Directional message passing for molecular graphs, 2020.  
Georgii G Krivov, Maxim V Shapovalov, and Roland L Dunbrack Jr. Improved prediction of protein side-chain conformations with scwrl4. Proteins: Structure, Function, and Bioinformatics, 77(4): 778-795, 2009.  
Brian Kuhlman and David Baker. Native protein sequences are close to optimal for their structures. Proceedings of the National Academy of Sciences, 97(19):10383-10388, 2000.

Zhixiu Li, Yuedong Yang, Eshel Faraggi, Jian Zhan, and Yaoqi Zhou. Direct prediction of profiles of sequences compatible with a protein structure by neural networks with fragment-based local and energy-based nonlocal profiles. Proteins: Structure, Function, and Bioinformatics, 82(10): 2565-2573, 2014.  
James O'Connell, Zhixiu Li, Jack Hanson, Rhys Heffernan, James Lyons, Kuldip Paliwal, Abdollah Dehzangi, Yuedong Yang, and Yaoqi Zhou. Spin2: Predicting sequence profiles from protein structures using deep neural networks. Proteins: Structure, Function, and Bioinformatics, 86(6): 629-633, 2018.  
Kliment Olechnovič and Česlovas Venclovas. Voromqa: Assessment of protein structure quality using interatomic contact areas. Proteins: Structure, Function, and Bioinformatics, 85(6):1131-1145, 2017.  
Guillaume Pagès, Benoit Charmettant, and Sergei Grudinin. Protein model quality assessment using 3d oriented convolutional neural networks. Bioinformatics, 35(18):3313-3319, 2019.  
Janaina Cruz Pereira, Ernesto Raul Caffarena, and Cicero Nogueira dos Santos. Boosting docking-based virtual screening with deep learning. Journal of chemical information and modeling, 56 (12):2495-2506, 2016.  
Yifei Qi and John ZH Zhang. Densecpd: Improving the accuracy of neural-network-based computational protein sequence design with densenet. Journal of Chemical Information and Modeling, 60(3):1245-1252, 2020.  
Soumya Sanyal, Ivan Anishchenko, Anirudh Dagar, David Baker, and Partha Talukdar. Proteingcn: Protein model quality assessment using graph convolutional networks. BioRxiv, 2020.  
Raphael Townshend, Rishi Bedi, Patricia Suriana, and Ron Dror. End-to-end learning on 3d protein structure for interface prediction. In Advances in Neural Information Processing Systems, pp. 15616-15625, 2019.  
Karolis Uziela, David Menéndez Hurtado, Nanjiang Shu, Björn Wallner, and Arne Elofsson. Proq3d: improved model quality assessments using deep learning. Bioinformatics, 33(10):1578-1580, 2017.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.  
Jingxue Wang, Huali Cao, John ZH Zhang, and Yifei Qi. Computational protein design with deep learning neural networks. *Scientific reports*, 8(1):1-9, 2018.  
Jonghun Won, Minkyung Baek, Bohdan Monastyrskyy, Andriy Kryshtafovych, and Chaok Seok. Assessment of protein model structure accuracy estimation in casp13: Challenges in the era of deep learning. Proteins: Structure, Function, and Bioinformatics, 87(12):1351-1360, 2019.  
Adam Zemla, Ceslovas Venclovas, John Moult, and Krzysztof Fidelis. Processing and evaluation of predictions in casp4. Proteins: Structure, Function, and Bioinformatics, 45(S5):13-21, 2001.  
Yuan Zhang, Yang Chen, Chenran Wang, Chun-Chao Lo, Xiuwen Liu, Wei Wu, and Jinfeng Zhang. Prodconn: Protein design using a convolutional neural network. Proteins: Structure, Function, and Bioinformatics, 2019.
