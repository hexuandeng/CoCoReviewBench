# INTRINSIC-EXTRINSIC CONVOLUTION AND POOLING FOR LEARNING ON 3D PROTEIN STRUCTURES

Anonymous authors

Paper under double-blind review

# ABSTRACT

Proteins perform a large variety of functions in living organisms and thus play a key role in biology. However, commonly used algorithms in protein representation learning were not specifically designed for protein data, and are therefore not able to capture all relevant structural levels of a protein during learning. To fill this gap, we propose two new learning operators, specifically designed to process protein structures. First, we introduce a novel convolution operator that considers the primary, secondary, and tertiary structure of a protein by using  $n$ -D convolutions defined on both the Euclidean distance, as well as multiple geodesic distances between the atoms in a multi-graph. Second, we introduce a set of hierarchical pooling operators that enable multi-scale protein analysis. We further evaluate the accuracy of our algorithms on common downstream tasks, where we outperform state-of-the-art protein learning algorithms.

# 1 INTRODUCTION

Proteins perform specific biological functions essential for all living organisms and hence play a key role when investigating the most fundamental questions in the life sciences. These biomolecules are composed of one or several chains of amino acids, which fold into specific conformations to enable various biological functionalities. Proteins can be defined using a

![](images/042e25728d3138f3a8511313a5195dfaadd5796535c11b5ad5acdf7657417fb7.jpg)  
Figure 1: Invariances present in protein structures.

multi-level structure:: The primary structure is given by the sequence of amino acids that are connected through covalent bonds and form the protein backbone. Hydrogen bonds between distant amino acids in the chain form the secondary structure, which defines substructures such as  $\alpha$ -helices and  $\beta$ -sheets. The tertiary structure results from protein folding and expresses the 3D spatial arrangement of the secondary structures. Lastly, the quaternary structure is given by the interaction of multiple amino acid chains.

Considering only one subset of these levels can lead to misinterpretations due to ambiguities. As shown by Alexander et al. (2009), proteins with almost identical primary structure, i.e., only containing a few different amino acids, can fold into entirely different conformations. Conversely, proteins from SH3 and OB folds have similar tertiary structures, but their primary and secondary structures differ significantly (Agrawal & Kishan, 2001) (Fig. 1). To avoid misinterpretations arising from these observations, capturing the invariances with respect to primary, secondary, and tertiary structures is of key importance when studying proteins and their functions.

Previously, the SOTA was dominated by methods based on hand-crafted features, usually extracted from multi-sequence alignment tools (Altschul et al., 1990) or annotated databases (El-Gebali et al., 2019). In recent years, these have been outperformed by protein representation learning algorithms in different protein modeling tasks such as protein fold classification (Hou et al., 2018; Rao et al., 2019; Bepler & Berger, 2019; Alley et al., 2019; Min et al., 2020) or protein function prediction (Strodthoff et al., 2020; Gligorijevic et al., 2019; Kulmanov et al., 2017; Kulmanov & Hoehndorf, 2019; Amidi et al., 2017). This can be attributed to the ability of machine learning algorithms to learn meaningful representations of proteins directly from the raw data. However, most of these

techniques only consider a subset of the relevant structural levels of proteins and thus can only create a representation from partial information. For instance, due to the high amount of available protein sequence data, most techniques solely rely on protein sequence data as input, and apply learning algorithms from the field of natural language processing (Rao et al., 2019; Alley et al., 2019; Min et al., 2020; Strodthoff et al., 2020), 1D convolutional neural networks (Kulmanov et al., 2017; Kulmanov & Hoehndorf, 2019), or use structural information during training (Bepler & Berger, 2019). Other methods have solely used 3D atomic coordinates as an input, and applied 3D convolutional neural networks (3DCNN) (Amidi et al., 2017; Derevyanko et al., 2018) or graph convolutional neural networks (GCNN) (Kipf & Welling, 2017). While few attempts have been made to consider more than one structural level of proteins in the network architecture (Gligorijevic et al., 2019), none of these hybrid methods incorporate all structural levels of proteins simultaneously. In contrast, a common approach is to process one structural level with the network architecture and the others indirectly as input features(Baldassarre et al. (2020) or Hou et al. (2018)).

In this paper, we introduce a novel end-to-end protein learning algorithm, that is able to explicitly incorporate the multi-level structure of proteins and captures the resulting different invariances. We show how a multi-graph data structure can represent the primary and secondary structures effectively by considering covalent and hydrogen bonds, while the tertiary structure can be represented by the spatial 3D coordinates of the atoms (Sec. 3). As chain bindings affect the tertiary structure, the quaternary structure is captured implicitly. Furthermore, by borrowing terminology from differential geometry of surfaces, we define a new convolution operator that uses both intrinsic (primary and secondary structures) and extrinsic (tertiary and quaternary structures) distances (Sec. 4). Moreover, since protein sizes range from less than one hundred to tens of thousands of amino acids (Brochieri & Karlin, 2005), we propose protein-specific pooling operations that allow hierarchical grouping of such a wide range of sizes, enabling the detection of features at different scales (Sec. 5). Lastly, we demonstrate, that by considering all mentioned protein structure levels, we can significantly outperform recent SOTA methods on protein tasks, such as protein fold and enzyme classification.

# 2 RELATED WORK

Early works on learning protein representations (Asgari & Mofrad, 2015; Yang et al., 2018) used word embedding algorithms (Mikolov et al., 2013), as employed in Natural Language Processing (NLP). Other approaches have used 1D convolutional neural networks (CNN) to learn protein representations directly from an amino acid sequence, for tasks such as protein function prediction (Kulmanov et al., 2017; Kulmanov & Hoehndorf, 2019), protein-compound interaction (Tsubaki et al., 2018), or protein fold classification (Hou et al., 2018). Recently, researchers have applied complex NLP models trained unsupervised on millions of unlabeled protein sequences and fine-tune them for different downstream tasks (Rao et al., 2019; Alley et al., 2019; Min et al., 2020; Strodthoff et al., 2020; Bepler & Berger, 2019). While representing proteins as amino acid sequences during learning, is helpful when only sequence data is available, it does not leverage the full potential of spatial protein representations that become more and more available with modern imaging and reconstruction techniques.

To learn beyond sequences, approaches have been developed, that consider the 3D structure of proteins. A range of methods has sampled protein structures to regular volumetric 3D representations and assessed the quality of the structure (Derevyanko et al., 2018), classified proteins in enzymes classes (Amidi et al., 2017), predicted the protein-ligand binding affinity (Ragoza et al., 2017) and the binding site (Jiménez et al., 2017), as well as the contact region between two proteins (Townshend et al., 2019). While this is attractive, as 3D grids allow for unleashing the benefits of all approaches developed for 2D images, such as pooling and multi-resolution techniques, unfortunately, grids do not scale well to fine structures or many atoms, and even more importantly, they do not consider the primary and secondary structure of proteins.

Another approach that makes use of a protein's 3D structure, is representing proteins as graphs and applying GCNNs (Kipf & Welling, 2017; Hamilton et al., 2017). Works based on this technique represent each amino acid as a node in the graph, while edges between them are created if they are at a certain Euclidean distance. This approach has been successfully applied to different problems. Classification of protein graphs into enzymes, for example, have become part of the standard data sets used to compare GCNN architectures (Gao & Ji, 2019; Ying et al., 2018). Moreover, other works with similar architectures have predicted protein interfaces (Fout et al., 2017), or protein structure

quality (Baldassarre et al., 2020). However, GCNN approaches suffer from over-smoothing, i.e., indistinguishable node representations after stacking several layers, which limits the maximum depth usable for such architectures (Cai & Wang, 2020).

It is also worth noticing that some of the aforementioned works on GCNN have considered different levels of protein structures indirectly by providing secondary structure type or distance along the sequence as initial node or edge features. However, these are not part of the network architecture and can be blended out due to the over-smoothing problem of GCNN. On the other hand, a recent protein function prediction method proposed by Gligorijevic et al. (2019) uses Long-Short Term Memory cells (LSTM) to encode the primary structure and then apply GCNNs to capture the tertiary structure. Also, the recent work from Ingraham et al. (2019) proposes an amino acid encoder that can capture primary and tertiary structures in the context of protein generative models. Unfortunately, none of these previous methods can incorporate all structural protein levels within the network architecture.

# 3 MULTI-GRAPH PROTEIN REPRESENTATION

To simultaneously take into account the primary, secondary, and tertiary protein structure during learning, we propose to represent proteins as a multi-graph  $\mathcal{G} = (\mathcal{N}, F, \mathcal{A}, \mathcal{B})$ . In this graph, atoms are represented as nodes associated with their 3D coordinates,  $\mathcal{N} \in \mathbb{R}^{n \times 3}$ , and associated features,  $\mathcal{F} \in \mathbb{R}^{n \times t}$ ,  $n$  being the number of atoms, and  $t$  the number of features. Moreover,  $\mathcal{A} \in \mathbb{R}^{n \times n}$  and  $\mathcal{B} \in \mathbb{R}^{n \times n}$  are two different adjacency matrices representing the connectivity of the graph. Elements of matrix  $\mathcal{A}$  are defined as  $\mathcal{A}_{ij} = 1$  if there is a covalent bond between atom  $i$  and atom  $j$ , and  $\mathcal{A}_{ij} = 0$  otherwise. Similarly, the elements of matrix  $\mathcal{B}$  are defined as  $\mathcal{B}_{ij} = 1$  if there is a covalent or hydrogen bond between atom  $i$  and atom  $j$ , and  $\mathcal{B}_{ij} = 0$  otherwise.

# 3.1 INTRINSIC-EXTRINSIC DISTANCES

Differential geometry of surfaces (Pogorelov, 1973) defines intrinsic geometric properties as those, that are invariant under isometric mappings, i.e., under deformations preserving the length of curves on a surface. On the other hand, extrinsic geometric properties are dependent on the embedding of the surfaces into the Euclidean space. Analogously, in our protein multi-graph, we define intrinsic geometric properties as those that are invariant under deformations preserving the length of paths along the graph, i.e., deformations that preserve the connectivity of the protein. Additionally, we define extrinsic geometric properties as those, that depend on the embedding of the protein into the Euclidean space, i.e., on the 3D protein conformation. Using this terminology, we define three distances in

our multi-graph, one extrinsic and two intrinsic (see Fig. 2). The extrinsic distance  $\tau_{\mathrm{e}}$  is defined by the protein conformation in Euclidean space, therefore we use the Euclidean distance between atoms, which enables us to capture the tertiary and quaternary structures of the protein. The intrinsic distances are inherent of the protein and independent of the actual 3D conformation. For the first intrinsic distance  $\tau_{\mathrm{i1}}$  we use the shortest path between two atoms along the adjacency matrix  $\mathcal{A}$  of the graph, capturing the primary structure. The second intrinsic distance  $\tau_{\mathrm{i2}}$  is defined as the shortest path between two atoms along the adjacency matrix  $\mathcal{B}$ , capturing thus the secondary structure.

![](images/ea9475c8e252057501a8f7495b3f5bdb7bdf97c0b5989555c8811012df995cf5.jpg)  
Figure 2: Distances between atoms in our protein graph.

# 4 INTRINSIC-EXTRINSIC PROTEIN CONVOLUTION

The key idea of our work is to take into account the multiple invariances described in Sec. 1 during learning. Therefore, based on the success of convolutional neural networks for images (Krizhevsky et al., 2012) and point clouds (Qi et al., 2017a), in this paper we define a convolution operator for proteins which is able to capture these invariances effectively. To this end, we propose a convolution on 3D protein structures, which is inspired by conventional convolutions as used to learn on structured images. First, we define the neighborhood of an atom as all atoms at a Euclidean distance smaller than  $m_{\mathrm{e}}$ . Moreover, we define our convolution kernels as a single Multi Layer Perceptron (MLP) which takes as input the three distances defined in Sec. 3.1, one extrinsic and two intrinsic, and outputs the

![](images/c02495de9c1d786c18439e8f079029d9e1c58e5cc1df00da4c9c87c1b1b3a11a.jpg)  
Figure 3: Intrinsic-extrinsic convolution on our multi-graph for atom A. First, we detect the neighboring atoms involved in the convolution using a ball query. For each atom, the extrinsic (pink) and two intrinsic (blue and green) distances are input into the kernel and the result is multiplied by the atom's features. Lastly, all contributions from neighboring atoms are summed up.

values of all kernels. This enables the convolution to learn kernels based on one or multiple structural levels of the protein. Thus, the proposed convolution operator is defined as:

$$
(\kappa * F) (\mathbf {x}) = \sum_ {i \in \mathcal {N} (\mathbf {x})} \sum_ {j = 0} ^ {t} F _ {i, j} \cdot \kappa_ {j} \left(\frac {\tau_ {\mathrm {e}} (\mathbf {x} , \mathbf {x} _ {i})}{m _ {\mathrm {e}}}, \frac {\tau_ {\mathrm {i} 1} (\mathbf {x} , \mathbf {x} _ {i})}{m _ {1}}, \frac {\tau_ {\mathrm {i} 2} (\mathbf {x} , \mathbf {x} _ {i})}{m _ {2}}\right) \tag {1}
$$

where  $\mathcal{N}(\mathbf{x})$  are the atoms at Euclidean distance  $d < m_{\mathrm{e}}$  from  $\mathbf{x}$ ,  $F_{i,j}$  is the input feature  $j$  of atom  $\mathbf{x}_i$ ,  $\tau_{\mathrm{e}}(\mathbf{x},\mathbf{x}_i)$  is the Euclidean distance between atom  $\mathbf{x}$  and atom  $\mathbf{x}_i$ ,  $\tau_{\mathrm{i1}}$  and  $\tau_{\mathrm{i2}}$  are the two intrinsic distances, and  $m_{\mathrm{e}}$ ,  $m_1$ , and  $m_2$  are the maximum distances allowed ( $m_1 = m_2 = 6$  hops in all experiments while  $m_{\mathrm{e}}$  is layer-dependent). All normalized distances are clamped to  $[0,1]$ . The convolution, performed independently for every atom, is illustrated on a model protein in Fig. 3, where the distances between neighboring atoms are shown. This operation has all properties of the standard convolution, locality and translational invariance, and at the same time rotational invariant.

Although this operation could appear similar to message passing algorithms, they differ significantly. Whilst message passing algorithms compute the new features of nodes/edges by transforming the previous features independently and aggregating them over the edges, which leads to over-smoothing, our convolution operator, on the other hand, decides which information to aggregate in each layer based on the distances along the different structural levels.

# 5 HIERARCHICAL PROTEIN POOLING

We consider proteins at atomic resolutions. This allows us to identify the spatial configuration of the amino acid side chains, something of key importance for active site detection. However, proteins can have a large number of atoms, preventing the usage of large receptive fields in our convolutions (computational restriction) and limiting the number of learned features per atom (memory restriction).

Pooling is a technique commonly used in convolutional neural networks, as it hierarchically reduces the dimensionality of the data by aggregating local information (Krizhevsky et al., 2012). While it is able to overcome computation and memory restrictions when learning on images, unfortunately, it cannot be directly applied to proteins, as conventional pooling methods rely on discrete sampling. While, on the other hand, some of the techniques developed for unstructured point cloud data (Qi et al., 2017b; Hermosilla et al., 2018) could be applied to learn on the 3D coordinates of atoms, it is unclear what the edges of that new graph representation would be.

Therefore, we define a set of operations that successively reduce the number of nodes in our protein graph. First, we iteratively reduce the number of atoms per amino acid to its alpha carbon. Afterward, we reduce the number of nodes along the backbone. Fig. 4 illustrates this process, which we describe in detail in the following paragraphs.

Amino acid pooling Simplified representations of amino acids have been previously used in Molecular Dynamics (MD) simulations, in order to get the number of computations down to a manageable level. Nevertheless, in contrast to our approach, these techniques do not use a uniform pooling pattern. Rather, while the atoms belonging to the backbone are not simplified, most of these approaches (Si-mons et al., 1997) simplify all atoms of the side chains to a single node. Other more conservative approaches, such as PRIMO (Kar et al., 2013) or Klein's models (DeVane et al., 2009), represent

![](images/77fbc3724178517a980e76230a596aef7c27a19d43ae0fefb0ba4fd7368a04c5.jpg)  
Figure 4: Hierarchical protein pooling: We segment the protein into amino acids (blue, orange, green). First, (a), we apply spectral clustering on each independent amino acid graph. Then, (b), each resulting amino acid is pooled to its  $\alpha$ -carbon. After that, we apply two backbone pooling operations, (c) and (d). Lastly, we apply the symmetric operation average, (e), to obtain the final feature vector.

side chains with a variable number of nodes using a lookup table. However, they do not perform a uniform simplification, resulting in a high variance in the number of atoms per cluster. Moreover, these methods need to manually update the lookup table to incorporate rare amino acids.

In this work, we decided instead to follow the common practice in CNN for images where a uniform pooling is used over the whole image. We propose a method that is able to reduce the number of nodes in the protein graph by half. To this end, we generate an independent graph for each amino acid using the covalent bonds as edges and apply spectral clustering (von Luxburg, 2007) to reduce by half the number of nodes. Since the number of amino acids is finite (20 appearing in the genetic code), these pooling matrices are reused among all proteins (see Fig. 4 (a)). However, since the method only requires a graph as input, it can be directly applied to synthetic or rare amino acids.

From the amino acid pooling matrices, we create a protein pooling matrix  $\mathcal{P} \in \mathbb{R}^{n \times m}$ , where  $n$  is the number of input atoms in the protein and  $m$  is the number of resulting clusters in the simplified protein. This matrix is defined as  $\mathcal{P}_{ij} = 1$  if the atom  $i$  collapses into cluster  $j$ , and  $\mathcal{P}_{ij} = 0$  otherwise. Using  $\mathcal{P}$  we can create a simplified protein graph,  $\mathcal{G}' = (\mathcal{N}', F', \mathcal{A}', \mathcal{B}')$ , with the following equations:

$$
\mathcal {N} ^ {\prime} = \mathcal {D} ^ {- 1} \mathcal {P} \mathcal {N} \quad F ^ {\prime} = \mathcal {D} ^ {- 1} \mathcal {P} F \quad \mathcal {A} ^ {\prime} = \mathcal {P} \mathcal {A P} ^ {T} \quad \mathcal {B} ^ {\prime} = \mathcal {P B P} ^ {T} \tag {2}
$$

where  $\mathcal{D} \in \mathbb{R}^{m \times m}$  is a diagonal matrix with  $\mathcal{D}_{jj} = \sum_{i=0}^{n} \mathcal{P}_{ij}$ . Note, that the resulting matrices  $\mathcal{A}'$  and  $\mathcal{B}'$  might not be binary adjacency matrices, i.e., non-zero diagonal values or values greater than one. Therefore, we assign zeros to the diagonals and clamp edge values to one. These matrices are computed using sparse representations in order to reduce the memory footprint.

Alpha carbon pooling In a second pooling step, we simplify the protein graph to a backbone representation, whereby we cluster all nodes from the same amino acid to a single node. We define  $\mathcal{P}$  accordingly, and the number of clusters  $m$  is equal to the number of amino acids, while  $\mathcal{P}_{ij} = 1$  if node  $i$  belongs to amino acid  $j$ , and  $\mathcal{P}_{ij} = 0$  otherwise. We use Eq. 2 to compute  $F'$ ,  $\mathcal{A}'$ , and  $\mathcal{B}'$ . However,  $\mathcal{N}'$  is defined as the alpha carbon positions of each amino acid since they better represent the backbone and the secondary structures.

Backbone pooling The last pooling steps are simplifications of the backbone chain. In each pooling step, every two consecutive amino acids in the chain are clustered together, effectively reducing by half the number of amino acids. Therefore, to compute  $\mathcal{N}^{\prime}, F^{\prime}, \mathcal{A}^{\prime}$ , and  $\mathcal{B}^{\prime}$ , we define  $\mathcal{P}$  accordingly and use Equation 2. For single-chain proteins, for example,  $\mathcal{P}_{ij} = 1$  if  $\lfloor i / 2\rfloor = j$ , or 0 otherwise.

# 6 EVALUATION

To evaluate our proposed protein learning approach, we specify a deep architecture (Sec. 6.1) which we compare to SOTA methods (Sec. 6.2) as well as different ablations of our approach (Sec. 6.3). During the entire evaluation, we focus on two downstream tasks in Sec. 6.4 and Sec. 6.5.

![](images/84c59346928dbaba75e8d144852fd7827925d33fab1712a44b060f43ba282edd.jpg)  
Figure 5: The architecture of our model. The input is composed of atom features and an atom embedding learned together with the network. Each layer is composed of two ResNet (He et al., 2016) bottleneck blocks, for which the radius in angstrom and the number of features are indicated in parentheses, followed by a pooling operation. An illustration of a single ResNet bottleneck block is presented in the right for  $D$  input features (before each convolution we use batch normalization and a Leaky ReLU). The global protein features are processed by an MLP which computes the final probabilities. Protein graphs used in each level are indicated at the bottom.

# 6.1 OUR ARCHITECTURE

By facilitating protein convolution and pooling, we are able to realize a deep architecture that enables learning on complex structures. In particular, the proposed architecture encodes an input protein into a latent representation which is later used in different downstream tasks.

The input of our network is a protein graph at atom level with a 6D input feature vector per each atom. The atom features comprise 1) covalent radius, 2) van der Waals radius, 3) atom mass, and 4-6) are the features of an atom type embedding learned together with the network. The input is then processed by 5 layers which iteratively increases the number of features, each composed of two ResNet (He et al., 2016) bottleneck blocks followed by a pooling operation (see Fig. 5). The respective receptive fields are [3, 6, 8, 12, 16] angstroms  $(\mathring{\mathrm{A}})$  using [64, 128, 256, 512, 1024] features. The size of the receptive fields has been chosen to include a reduced number of nodes in it.

As the architecture is fully-convolutional, it can process proteins of arbitrary size, but after finitely many steps, it obtains an intermediate result of varying size. Hence, to reduce this to one final result vector, a symmetric aggregation operation, average, is used. Lastly, a single layer MLP with 1024 hidden neurons is used to predict the final probabilities.

# 6.2 OTHER ARCHITECTURES

We compare our architecture, as described above, to SOTA learners designed for similar downstream tasks. First, we compare to the latest sequence-based methods pre-trained unsupervised on millions of sequences: Rao et al. (2019); Bepler & Berger (2019); Alley et al. (2019); Strodthoff et al. (2020). Second, we compare to different methods that take only the 3D protein structure into account: Kipf & Welling (2017); Diehl (2019); Derevyanko et al. (2018). Lastly, we also compare to two recent hybrid methods: Gligorijevic et al. (2019), who process primary structure with several LSTM cells first and then apply GCNN, and Baldassarre et al. (2020), who indirectly process primary and secondary structures by using as input distances along the backbone as edge features and secondary structure type as node features in a GCNN setup. When available, we provide the results reported in the original papers, while more details of the training procedures are provided in Appendix E.

# 6.3 ABLATIONS OF OUR METHOD

We study four axes of ablation: convolution, neighborhood, pooling, and representation. When moving along one axis, all other axes are fixed to Ours  $(\bullet)$ .

Convolution ablation. We consider four different ablations of convolutions: GCNN (·) (Kipf & Welling, 2017); ExConv (·), kernels defined in 3D space only (Hermosilla et al., 2018); InConvC (·) uses only intrinsic distance  $\tau_{i1}$ ; InConvH (·) uses only intrinsic distance  $\tau_{i2}$ ; InConvCH (·) makes use of both intrinsic distances at the same time; Ours3DCH (·) uses both intrinsic distances plus distances along the three spatial dimensions; and lastly, Ours (·), that refers to our proposed convolution which uses both geodesics plus Euclidean distance.

Table 1: Comparison of our network to other methods on the two tasks (protein fold and enzyme catalytic reaction classification) measured as mean accuracy, where we outperform all methods.  

<table><tr><td rowspan="2"></td><td rowspan="2">Architecture</td><td rowspan="2"># params</td><td colspan="3">FOLD</td><td rowspan="2">REACTION</td></tr><tr><td>Fold</td><td>Super.</td><td>Fam.</td></tr><tr><td rowspan="3">Rao et al. (2019)*</td><td>1D ResNet</td><td>41.7 M</td><td>17.0 %</td><td>31.0 %</td><td>77.0 %</td><td>70.9 %</td></tr><tr><td>LSTM</td><td>43.0 M</td><td>26.0 %</td><td>43.0 %</td><td>92.0 %</td><td>79.9 %</td></tr><tr><td>Transformer</td><td>38.4 M</td><td>21.0 %</td><td>34.0 %</td><td>88.0 %</td><td>69.8 %</td></tr><tr><td>Bepler &amp; Berger (2019)*</td><td>LSTM</td><td>31.7 M</td><td>17.0 %</td><td>20.0 %</td><td>79.0 %</td><td>74.3 %</td></tr><tr><td>Alley et al. (2019)*</td><td>mLSTM</td><td>18.2 M</td><td>23.0 %</td><td>38.0 %</td><td>87.0 %</td><td>72.9 %</td></tr><tr><td>Strothoff et al. (2020)*</td><td>LSTM</td><td>22.7 M</td><td>14.9 %</td><td>21.5 %</td><td>83.6 %</td><td>73.9 %</td></tr><tr><td>Kipf &amp; Welling (2017)</td><td>GCNN</td><td>1.0 M</td><td>16.8 %</td><td>21.3 %</td><td>82.8 %</td><td>67.3 %</td></tr><tr><td>Diehl (2019)</td><td>GCNN</td><td>1.0 M</td><td>12.9 %</td><td>16.3 %</td><td>72.5 %</td><td>57.9 %</td></tr><tr><td>Derevyanko et al. (2018)</td><td>3D CNN</td><td>6.0 M</td><td>31.6 %</td><td>45.4 %</td><td>92.5 %</td><td>78.8 %</td></tr><tr><td>Gligorijevic et al. (2019)*</td><td>LSTM+GCNN</td><td>6.2 M</td><td>15.3 %</td><td>20.6 %</td><td>73.2 %</td><td>63.3 %</td></tr><tr><td>Baldassarre et al. (2020)</td><td>GCNN</td><td>1.3 M</td><td>23.7 %</td><td>32.5 %</td><td>84.4 %</td><td>60.8 %</td></tr><tr><td>Ours</td><td></td><td>9.8 M</td><td>45.0 %</td><td>69.7 %</td><td>98.9 %</td><td>87.2 %</td></tr></table>

*Methods pre-trained unsupervised on 10-31 million protein sequences.

Neighborhood ablation. We compare several methods to define our receptive field: CovNeigh  $(\bullet)$ , which uses the intrinsic distance  $\tau_{i1}$  on the graph; HyNeigh  $(\bullet)$ ; which uses the intrinsic distance  $\tau_{i2}$ , as well as Ours  $(\bullet)$  which uses the Euclidean distance.

Pooling ablation. We consider six options: NoPool (·), which does not use any pooling operation; GridPool (·) overlays the protein with increasingly coarse grids and pools all atoms into one cell (Thomas et al., 2019); TopKPool (·) learns a per-node importance score which is used to drop nodes (Gao & Ji, 2019); EdgePool (·), which learns a per-edge importance score to collapse edges (Diehl, 2019); RosettaCEN (·), which uses the centroid simplification method used by the Rosetta software in Molecular Dynamics simulation (Simons et al., 1997); and our pooling, Ours (·).

Representation ablation. Lastly, we evaluate the granularity of the input, considering the protein at the amino acid level, AminoGraph  $(\bullet)$ , or at atomic level, Ours  $(\bullet)$ .

# 6.4 TASK 1: FOLD CLASSIFICATION (FOLD)

Task. Protein fold classification is of key importance in the study of the relationship between protein structure and function, and protein evolution. The different fold classes group proteins with similar secondary structure compositions, orientations, and connection orders. In this task, given a protein, we predict the fold class, whereby the performance is measured as mean accuracy.

Data set. We use the training/validation/test splits of the SCOPe 1.75 data set of Hou et al. (2018). This data set consolidated 16, 712 proteins from 1, 195 folds. We obtained the 3D structures of the proteins from the SCOPe 1.75 database (Murzin et al., 1955). The data set provides three different test sets: Fold, in which proteins from the same superfamily are not present during training; Superfamily, in which proteins from the same family are not seen during training; and Family, in which proteins of the same family are present during training.

Results. Results, as compared to other methods, are reported in Tbl. 1. We find our method to perform better by a large margin without using additional features as input or pre-trained on additional data. Tbl. 2 shows how the tested ablations perform on this task. Overall we see that Ours  $(\bullet)$  performs better than any ablation, indicating that our protein-specific representation, convolution, and pooling are all important ingredients. Lastly, we compare in Tbl. 4 to the state of the art results reported in the paper (Hou et al., 2018) as to the results obtained using sequence alignment, where we outperform both.

Table 3: Comparison to the reported results for the SCOPe 1.75 data set in the paper Hou et al. (2018).  

<table><tr><td></td><td>Fold</td><td>Super.</td><td>Family</td></tr><tr><td>DeepSF</td><td>40.9 %</td><td>50.7 %</td><td>76.2 %</td></tr><tr><td>BLAST</td><td>5.6 %</td><td>42.2 %</td><td>96.8 %</td></tr><tr><td>Ours</td><td>45.0 %</td><td>69.7 %</td><td>98.9 %</td></tr></table>

Table 2: Study of ablations (rows) for the FOLD and REACTION tasks (columns).  

<table><tr><td rowspan="2" colspan="2"></td><td colspan="3">FOLD</td><td rowspan="2">REACTION</td></tr><tr><td>Fold</td><td>Super.</td><td>Fam.</td></tr><tr><td rowspan="7">Conv.</td><td>GCNN</td><td>25.7%</td><td>46.5%</td><td>95.9%</td><td>84.9%</td></tr><tr><td>ExConv</td><td>30.1%</td><td>46.3%</td><td>92.0%</td><td>85.0%</td></tr><tr><td>InConvC</td><td>37.6%</td><td>65.1%</td><td>98.7%</td><td>85.4%</td></tr><tr><td>InConvH</td><td>40.8%</td><td>62.0%</td><td>98.4%</td><td>85.5%</td></tr><tr><td>InConvCH</td><td>43.5%</td><td>66.7%</td><td>98.7%</td><td>85.2%</td></tr><tr><td>Ours3DCH</td><td>40.7%</td><td>62.2%</td><td>98.1%</td><td>85.8%</td></tr><tr><td>Ours</td><td>45.0%</td><td>69.7%</td><td>98.9%</td><td>87.2%</td></tr><tr><td rowspan="3">Neighbors</td><td>CovNeigh</td><td>27.2%</td><td>41.5%</td><td>92.3%</td><td>41.6%</td></tr><tr><td>HyNeigh</td><td>33.3%</td><td>50.6%</td><td>96.9%</td><td>56.9%</td></tr><tr><td>Ours</td><td>45.0%</td><td>69.7%</td><td>98.9%</td><td>87.2%</td></tr><tr><td rowspan="6">Pool</td><td>NoPool</td><td>37.1%</td><td>59.8%</td><td>97.6%</td><td>84.7%</td></tr><tr><td>GridPool</td><td>28.6%</td><td>41.8%</td><td>91.8%</td><td>86.1%</td></tr><tr><td>TopKPool</td><td>40.7%</td><td>65.4%</td><td>98.4%</td><td>84.5%</td></tr><tr><td>EdgePool</td><td>44.4%</td><td>69.6%</td><td>99.0%</td><td>86.9%</td></tr><tr><td>RosettaCEN</td><td>41.7%</td><td>66.5%</td><td>98.9%</td><td>86.5%</td></tr><tr><td>Ours</td><td>45.0%</td><td>69.7%</td><td>98.9%</td><td>87.2%</td></tr><tr><td rowspan="2">Repr.</td><td>AminoGraph</td><td>39.6%</td><td>64.7%</td><td>99.1%</td><td>85.3%</td></tr><tr><td>Ours</td><td>45.0%</td><td>69.7%</td><td>98.9%</td><td>87.2%</td></tr></table>

# 6.5 TASK 2: ENZYME-CATALYZED REACTION CLASSIFICATION (REACTION)

Task. For this task, we classify proteins based on the enzyme-catalyzed reaction according to all four levels of the Enzyme Commission (EC) number (Webb, 1992). The performance is again evaluated as mean accuracy.

Data set. We collected a total of 37,428 proteins from 384 EC numbers. The data was then split into 29,215 instances for training, 2,562 instances for validation, and 5,651 for testing. Note that all proteins have less than  $50\%$  sequence-similarity in-between splits. A full description of the data set is provided in Appendix D.

Results. Again, our method outperforms previous works on this task (Tbl. 1). Ablations of our method (Tbl. 2) give further indication that all our components are relevant also for other tasks.

# 7 CONCLUSIONS

Based on the multi-level structure of proteins, we have proposed a neural network architecture to process protein structures. The presented architecture takes advantage of primary, secondary, and tertiary protein structures to learn a convolution operator that works with intrinsic and extrinsic distances as input. Moreover, we have presented a set of pooling operations that enable the dimensionality reduction of the input mimicking the designs used by convolutional neural networks on images. Lastly, our evaluation has shown that by incorporating all the structural levels of a protein in our designs we are able to outperform existing SOTA protein learning algorithms on two downstream tasks, protein fold and enzyme classification.

Despite the reported success achieved in protein learning, some limitations apply to our idea. The main one being, that it requires the 3D structure of each protein, whilst other methods only make use of the protein sequence. This, however, may be alleviated by the advances in protein structure determination (Callaway, 2020) and prediction (Senior et al., 2020). Moreover, like other commonly used approaches such as GCNN, our convolution operator is invariant to rotations but also to chirality changes. This could be solved by incorporating directional information into the kernel's input. We leave this improvement for future work.

# REFERENCES

Vishal Agrawal and Radha KV Kishan. Functional evolution of two subtly different (similar) folds. BMC Structural Biology, 2001.  
Patrick A. Alexander, Yanan He, Yihong Chen, John Orban, and Philip N. Bryan. A minimal sequence code for switching protein structure and function. Proceedings of the National Academy of Sciences, 2009.  
Ethan C. Alley, Grigory Khimulya, Surojit Biswas, Mohammed AlQuraishi, and George M. Church. Unified rational protein engineering with sequence-based deep representation learning. Nature Methods, 2019.  
S.F. Altschul, W. Gish, W. Miller, E.W. Myers, and D.J Lipman. Basic local alignment search tool. J Molecular Biology, 215(3):403-10, 1990.  
A. Amidi, S. Amidi, D. Vlachakis, V. Megalooikonomou, N. Paragios, and E. Zacharaki. EnzyNet: enzyme classification using 3D convolutional neural networks on spatial representation. arXiv:1707.06017, 2017.  
E. Asgari and M.R.K. Mofrad. Continuous distributed representation of biological sequences for deep proteomics and genomics. PLoS ONE, 2015.  
F. Baldassarre, D. M. Hurtado, A. Elofsson, and H. Azizpour. GraphQA: Protein Model Quality Assessment using Graph Convolutional Networks. Bioinformatics, 2020.  
Tristan Bepler and Bonnie Berger. Learning protein sequence embeddings using information from structure. International conference on learning representations, 2019.  
H. M. Berman, J. Westbrook, Z. Feng, G. Gilliland, T. N. Bhat, H. Weissig, I. N. Shindyalov, and P. E. Bourne. The Protein Data Bank. *Nucleic Acids Res*, 2000.  
Luciano Brochieri and Samuel Karlin. Protein length in eukaryotic and prokaryotic proteomes. *Nucleic Acids Res*, 33(10):3390-400, 2005.  
Chen Cai and Yusu Wang. A note on over-smoothing for graph neural networks. ICML 2020 Graph Representation Learning workshop, 2020.  
E. Callaway. Revolutionary cryo-em is taking over structural biology. Nature News, 2020.  
Papers With Code. URL paperswithcode.com/sota/ graph-classification-on-dd. Accessed on 1.10.2020.  
J. M Dana, A. Gutmanas, N. Tyagi, G. Qi, C. O'Donovan, M. Martin, and S. Velankar. SIFTS: updated Structure Integration with Function, Taxonomy and Sequences resource allows 40-fold increase in coverage of structure-based annotations for proteins. *Nucleic Acids Research*, 2018.  
Georgy Derevyanko, Sergei Grudinin, Yoshua Bengio, and Guillaume Lamoureux. Deep convolutional networks for quality assessment of protein folds. Bioinformatics, 34(23):4046-53, 2018.  
R. DeVane, W. Shinoda, P. B. Moore, and M. L. Klein. Transferable coarse grain nonbonded interaction model for amino acids. Journal of Chemical Theory and Computation, 2009.  
Frederik Diehl. Edge contraction pooling for graph neural networks. arxiv:1905.10990, 2019.  
Paul D Dobson and Andrew J Doig. Distinguishing enzyme structures from non-enzymes without alignments. *J Molecular Biology*, 330(4):771–783, 2003.  
Sara El-Gebali, Jaina Mistry, Alex Bateman, Sean R Eddy, Aurélien Luciani, Simon C Potter, Matloob Qureshi, Lorna J Richardson, Gustavo A Salazar, Alfredo Smart, Erik L L Sonnhammer, Layla Hirsh, Lisanna Paladin, Damiano Piovesan, Silvio C E Tosatto, and Robert D Finn. The Pfam protein families database in 2019. Nucleic Acids Research, 2019.  
A. Fout, J. Byrd, B. Shariat, and A. Ben-Hur. Protein interface prediction using graph convolutional networks. In Advances in Neural Information Processing Systems 30. 2017.

Hongyang Gao and Shuiwang Ji. Graph U-nets. In ICML, pp. 2083-2092, 2019.  
V. Gligorijevic, P. D. Renfrew, T. Kosciolek, J. K. Leman, K. Cho, T. Vatanen, D. Berenberg, B. Taylor, I. M. Fisk, R. J. Xavier, R. Knight, and R. Bonneau. Structure-based function prediction using graph convolutional networks. bioRxiv, 2019.  
Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In NIPS, pp. 1024-34, 2017.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.  
P. Hermosilla, T. Ritschel, P-P Vazquez, A. Vinacua, and T. Ropinski. Monte carlo convolution for learning on non-uniformly sampled point clouds. ACM Trans. Graph. (Proc. SIGGRAPH Asia), 37 (6), 2018.  
J. Hou, B. Adhikari, and J. Cheng. Deepsf: Deep convolutional neural network for mapping protein sequences to folds. In Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics, 2018.  
John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for graph-based protein design. In NeurIPS, pp. 15820-15831, 2019.  
J Jiménez, S Doerr, G Martínez-Rosell, and A S Rose. DeepSite: protein-binding site predictor using 3D-convolutional neural networks. Bioinformatics, 33(19):3036-3042, 2017.  
P. Kar, S. M. Gopal, Y.-M. Cheng, A. Predeus, and M. Feig. Primo: A transferable coarse-grained force field for proteins. Journal of Chemical Theory and Computation, 2013.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. ICML, 2017.  
A. Krizhevsky, I. Sutskever, and G. E Hinton. Imagenet classification with deep convolutional neural networks. In NIP, 2012.  
Maxat Kulmanov and Robert Hoehndorf. DeepGOPlus: improved protein function prediction from sequence. Bioinformatics, 36(2):422-429, 2019.  
Maxat Kulmanov, Mohammed Asif Khan, and Robert Hoehndorf. DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier. Bioinformatics, 34(4):660-668, 2017.  
T. Mikolov, K. Chen, G. Corrado, and J. Dean. Efficient estimation of word representations in vector space. arXiv, 2013.  
Seonwoo Min, Seunghyun Park, Siwon Kim, Hyun-Soo Choi, and Sungroh Yoon. Pre-training of deep bidirectional protein sequence representations with structural information. Bioinformatics, 2020.  
A.G. Murzin, S.E. Brenner, T. Hubbard, and C. Chothia. SCOP: a structural classification of proteins database for the investigation of sequences and structures. Journal of Molecular Biology, 1955.  
D. Q. Nguyen, T. D. Nguyen, and D. Phung. Universal self-attention network for graph classification. arXiv, 2020.  
A. V. Pogorelov. Extrinsic geometry of convex surfaces, volume 35. American Mathematical Soc., 1973.  
Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. CVPR, 2017a.  
Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. NIPS, 2017b.

Matthew Ragoza, Joshua Hochuli, Elisa Idrobo, Jocelyn Sunseri, and David Ryan Koes. Protein-ligand scoring with convolutional neural networks. *J Chemical Information and Modeling*, 57 (4):942-957, 2017.  
Roshan Rao, Nicholas Bhattacharya, Neil Thomas, Yan Duan, Xi Chen, John Canny, Pieter Abbeel, and Yun S. Song. Evaluating protein transfer learning with tape. Advances in Neural Information Processing Systems, 2019.  
A. W. Senior, R. Evans, J. Jumper, J. Kirkpatrick, L. Sifre, T. Green, C. Qin, A. Židek, A. W. R. Nelson, A. Bridgland, H. Penedones, S. Petersen, K. Simonyan, S. Crossan, P. Kohli, D. T. Jones, D. Silver, K. Kavukcuoglu, and D. Hassabis. Improved protein structure prediction using potentials from deep learning. Nature, 2020.  
K. T. Simons, C. Kooperberg, E. Huang, and D. Baker. Assembly of protein tertiary structures from fragments with similar local sequences using simulated annealing and bayesian scoring functions. Journal of Molecular Biology, 1997.  
Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek. Udsmprot: universal deep sequence models for protein classification. Bioinformatics, 2020.  
Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatrix Marcotegui, François Goulette, and Leonidas J. Guibas. KPConv: Flexible and deformable convolution for point clouds. ICCV, 2019.  
Matteo Togninalli, Elisabetta Ghisu, Felipe Llinares-López, Bastian Rieck, and Karsten Borgwardt. Wasserstein weisfeiler-lehman graph kernels. In Advances in Neural Information Processing Systems 32. 2019.  
Raphael Townshend, Rishi Bedi, Patricia Suriana, and Ron Dror. End-to-end learning on 3D protein structure for interface prediction. In NeurIPS, pp. 15642-15651, 2019.  
Masashi Tsubaki, Kentaro Tomii, and Jun Sese. Compound-protein interaction prediction with end-to-end learning of neural networks for graphs and sequences. Bioinformatics, 35(2):309-318, 2018.  
L. van der Maaten and G. E. Hinton. Visualizing high-dimensional data using t-sne. Journal of Machine Learning Research, 2008.  
Ulrike von Luxburg. A tutorial on spectral clustering. Statistics and computing, 17(4):395-416, 2007.  
Edwin C. Webb. Enzyme Nomenclature 1992. Academic Press, 1992.  
K. K Yang, Z. Wu, C. N Bedbrook, and F. H Arnold. Learned protein embeddings for machine learning. Bioinformatics (Oxford, England), 2018.  
Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In NIPS, pp. 4800-4810, 2018.  
Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. An end-to-end deep learning architecture for graph classification. In Proc. AAAI Conference on Artificial Intelligence, pp. 4438-4445, 2018.  
Zhen Zhang, Jiajun Bu, Martin Ester, Jianfeng Zhang, Chengwei Yao, Zhi Yu, and Can Wang. Hierarchical graph pooling with structure learning. arXiv, 2019.  
Qi Zhao and Yusu Wang. Learning metrics for persistence-based summaries and applications for graph classification. In Advances in Neural Information Processing Systems 32. 2019.
