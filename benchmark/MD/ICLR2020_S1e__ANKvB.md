# MOLECULAR GRAPH ENHANCED TRANSFORMER FOR RETROSYNTHESIS PREDICTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

With massive possible synthetic routes in chemistry, retrosynthesis prediction is still a challenge for researchers. Recently, retrosynthesis prediction is formulated as a Machine Translation (MT) task. Namely, since each molecule can be represented as a Simplified Molecular-Input Line-Entry System (SMILES) string, the process of synthesis is analogized to a process of language translation from reactants to products. However, the MT models that applied on SMILES data usually ignore the information of natural atomic connections and the topology of molecules. In this paper, we propose a Graph Enhanced Transformer (GET) framework, which adopts both the sequential and graphical information of molecules. Four different GET designs are proposed, which fuse the SMILES representations with atom embedding learned from our improved Graph Neural Network (GNN). Empirical results show that our model significantly outperforms the Transformer model in test accuracy.

# 1 INTRODUCTION

Retrosynthesis prediction aims to predict a set of suitable reactants that can synthesize the desired molecule via a series of reactions. It pushes forward an immense influence in agriculture, medical treatment, drug discovery and so on. However, the retrosynthesis prediction is challenging since there are massive possible synthetic routes available and it is often difficult to navigate the direction of retrosynthesis process. Indeed, each bond in the target molecule may represent a possible retrosynthetic disconnection, leading to a vast space of possible starting materials. Besides, the difference between two synthetic routes may be subtle, which usually depends on the global structures. Actually, planning a proper retrosynthetic route for a complex molecule is also a tough work even for the professional chemists.

One of the prevailing methods is to deem the retrosynthesis prediction as a machine translation task. This analogy is comprehensible since every molecule has a unique text representation named SMILES (Weininger, 1988). In this case, given a target molecule written in SMILES notation, the retrosynthesis prediction is just to predict a string of SMILES which represents the reactants. Based on this idea, Liu et al. (2017) first applied the LSTM with attention mechanism in retrosynthesis prediction and achieved comparable performance compared with previous traditional methods. Whereafter, many works (Karpov et al., 2019; Zheng et al., 2019; Lin et al., 2019; Lee et al., 2019) tried to employ Transformer (Vaswani et al., 2017), which is a more powerful Sequence-to-Sequence(Seq2Seq) model, to improve prediction accuracy in retrosynthesis. However, these methods just utilize the sequential representations of the molecule, while ignoring the natural topological connections between atoms within the molecule. These atomic connections can provide more flexible and accurate chemical information, which is critical in many related chemical tasks like molecular representation (Duvenaud et al., 2015; Gilmer et al., 2017) and chemical reaction prediction (Jin et al., 2017; Do et al., 2019). We believe that the absence of this molecular graph information hinders the further improvement of the present methods for retrosynthesis. How to effectively make use of this natural graphical information of the molecular structure, therefore, becomes a vital problem.

To tackle this problem, we propose Graph Enhanced Transformer(GET) framework that can enjoy the advantage of both graph-level representations and sequence-level representations. Specifically, to solve the retrosynthesis problem, we design an improved Graph Neural Network(GNN) called Graph Attention with Edge and Skip-connection (GAES) to learn each atom's representation, and

try four strategies to incorporate it with the original SMILES representation in the encoder. The main contributions of this paper are as follows:

- We propose a new framework called GET that fuses graphical representations with sequential representations of the target molecule to solve retrosynthesis prediction task.  
- We design a powerful GNN called GAES that learns high-quality representations of atom nodes in a self-attention manner with bond features, and it is less affected by the side-effect of stacking more layers.

GET is evaluated on USPTO-50K, a common benchmark dataset for retrosynthesis. Experimental results show that our model achieves new records for top-1 prediction accuracy in the state-of-the-art methods and outperforms Seq2Seq-based methods in all tested top- $n$  accuracy, demonstrating the effectiveness of fusing the molecular graph information with the SMILES sequence information.

# 2 RELATED WORK

Prior work on retrosynthesis can be mainly summarized into two categories: template-based methods and template-free methods.

Template-based Methods The majority of computer-aided retrosynthetic methods in the early period were relied on encoding reaction templates or generalized subgraph matching rules. LHASA (Corey & Wipke, 1969) was the first software for retrosynthetic analysis. Recently, one of the most well-known retrosynthesis analysis tool is Synthia (Szymkuć et al., 2016) that integrated about 70,000 hand-encoded reaction rules collected by manual. Based on the 60,000 reaction templates derived from 12 million single-step reaction examples, Schreck et al. (2019) introduced Reinforcement Learning (RL) into this area by treating retrosynthesis as a game whose goal is to identify policies that make (near) optimal reaction choices during each step of retrosynthetic planning. Segler et al. (2018) extracted two sets of transformation rules and combined Monte Carlo tree search with symbolic AI to discover possible retrosynthetic routes. Besides manual extracted rules, some works (Law et al., 2009; Segler & Waller, 2017; Coley et al., 2017) tried to collect reaction templates automatically and perform retrosynthesis based on these automated templates. Although template-based methods work well in many cases, they still face a serious drawback that they generally cannot achieve accurate prediction accuracy outside of their known templates.

Template-free Methods Our model just belongs to this category. Emerging template-free methods are to treat retrosynthesis as a machine translation task as introduced in section 1. Since these methods do not need any reaction templates and prior chemistry knowledge, they are attracting more and more attention from academia. Moreover, without the constraint of fixed templates, they have the potential of discovering novel synthetic routes. The most related work to ours are Zheng et al. (2019) Lin et al. (2019); Karpov et al. (2019); Lee et al. (2019) that apply Transformer in retrosynthesis prediction.

# 3 BACKGROUND

In this section, we first introduce how GNN is used in learning the molecule (or atoms) representation, and then describe how the Transformer model is previously applied in retrosynthesis prediction.

# 3.1 GNN FOR MOLECULE REPRESENTATION LEARNING

GNN has been widely used in learning the representation of the molecule and its atoms. Naturally, molecules can be represented as graph structure with atoms as nodes and bonds as edges. Suppose that a molecular graph  $G$  has initial node representations  $h_v$  and edge representations  $e_{vw}$ , a typical one-layer GNN can learn new and more powerful node representations from  $G$  by the following message passing process described in Gilmer et al. (2017):

$$
\boldsymbol {m} _ {v} = \sum_ {w \in N (v)} M \left(\boldsymbol {h} _ {v}, \boldsymbol {h} _ {w}, \boldsymbol {e} _ {v w}\right), \tag {1}
$$

$$
\boldsymbol {h} _ {v} ^ {\text {n e w}} = U \left(\boldsymbol {h} _ {v}, \boldsymbol {m} _ {v}\right), \tag {2}
$$

where  $N(v)$  denotes the neighbors of node  $v$  in graph  $G$ ,  $M$  is the message function that is responsible for collecting information from neighbors, and  $U$  is the update function for fusing collected

information  $m_v$  with old node representation  $h_v$  to obtain the new node representation  $h_v^{new}$ . Further, we can stack several these GNN layers to capture higher-order neighbors' information.

Then a readout function  $R$  can be used to integrate all node representations into a whole graph representation  $g$ :

$$
\boldsymbol {g} = R \left(\left\{\boldsymbol {h} _ {v} \mid v \in G \right\}\right). \tag {3}
$$

The  $h_v$  and  $g$ , which represent the atoms and the whole molecule, are often trained in an end-to-end way for a specific chemical task, such as chemical properties prediction, reaction prediction and molecule optimization.

# 3.2 TRANSFORMER FOR RETROSYNTHESIS PREDICTION

Transformer (Vaswani et al., 2017) is a Seq2Seq model that has shown excellent performance in machine translation task. Also, it has been applied in chemical reaction prediction and retrosynthesis prediction before. Given an input SMILES that represents the target molecule and a specified reaction type (optional), retrosynthesis prediction is to predict the output SMILES which represents the possible reactants that can synthesis the target molecule in the specified reaction type. Thus, retrosynthesis prediction can be deemed as a machine translation task whose source language is target molecule SMILES and the target language is reactants SMILES.

In this view, Transformer can be applied to retrosynthesis prediction as the same as to machine translation. Since Transformer is a mature model that has been widely used in natural language processing (NLP), we just give a simple introduction here. Specifically, Transformer follows an encoder-decoder structure and is composed of several combinations of multi-head attention layers and position-wise feed forward layers. The encoder consists of a stack of  $N = 6$  identical layers. Each layer includes two main components: (multi-head) self-attention layer and feed-forward network. Given an input vectors  $(\pmb{p}_1, \dots, \pmb{p}_n)$ ,  $\pmb{p} \in \mathbb{R}^d$ , the  $t$ -th output  $s_t$  of the self-attention layer is calculated by:

$$
\boldsymbol {q} _ {t} = \boldsymbol {W} _ {q} \boldsymbol {p} _ {t}, \quad \boldsymbol {k} _ {m} = \boldsymbol {W} _ {m} \boldsymbol {p} _ {m}, \quad \boldsymbol {v} _ {t} = \boldsymbol {W} _ {v} \boldsymbol {p} _ {t}, \quad \boldsymbol {s} _ {t} = \sum_ {m = 1} ^ {n} \operatorname {s o f t m a x} \left(\frac {<   \boldsymbol {q} _ {t} , \boldsymbol {k} _ {m} >}{\sqrt {d _ {k}}}\right) \boldsymbol {v} _ {t}, \tag {4}
$$

where  $d_k$  is the dimension of  $\mathbf{q}$  and  $\pmb{k}$ ,  $\pmb{W}_q, \pmb{W}_m, \pmb{W}_v$  are weight matrices. One such operation is called one head, and we can concatenate several heads to change to multi-head self attention.

The feed-forward network is composed of two linear transformations with a ReLU activation:

$$
\operatorname {F F N} (\boldsymbol {x}) = \max  \left(0, \boldsymbol {x} \boldsymbol {W} _ {1} + \boldsymbol {b} _ {1}\right) \boldsymbol {W} _ {2} + \boldsymbol {b} _ {2}, \tag {5}
$$

where  $\pmb{W}_1, \pmb{W}_2$  are weight matrices and  $\pmb{b}_1, \pmb{b}_2$  are biases.

Similarly, the decoder is also mainly composed of multi-attention layers and position-wise feed forward layers. It will generate the output SMILES step by step. At step  $t$ , it utilizes the encoder's output  $(p_{1}', \dots, p_{n}')$  and all previous steps' output  $(x_{1}', \dots, x_{t - 1}')$  to generate the next SMILES character  $x_{t}'$ . This process repeated until generating a specific termination character, i.e.,  $x_{t}' = < EOS >$ .

# 4 GRAPH ENHANCED TRANSFORMER

In this section, we provide the details about our Graph Enhanced Transformer (GET) framework for retrosynthesis prediction. Figure 1 shows an overview of GET. On the whole, GET is accord with typical encoder-decoder structure, of which the integral encoder is composed of the graph encoder and transformer encoder for learning the representation in graph-level and sequence-level respectively. Given a target molecule's SMILES, it will first pass through the two encoders somehow to get the hidden representation of each character, and then the decoder will utilize these hidden representations to generate the output SMILES.

# 4.1 GRAPH ENCODER

We design a new powerful GNN called Graph Attention with Edge and Skip-connection (GAES) as the graph encoder, which can learn the representation of each atom in a molecule. This graphical

![](images/69cb4d4476c9ec45f5712403a1489585de8cef252cd07e92aa945041d3683ba1.jpg)  
Figure 1: Overview of GET. The input SMILES will be processed by the two-sub encoders (graph encoder and transformer encoder) somehow to be transformed to its hidden representation. Then, at each step, the decoder will utilize the hidden representation and all outputs of the previous steps to generate the present step's SMILES character.

representation reflects the connection of atoms within a molecule and may play a significant role in further alleviating the long-term dependency problem to avoid generating chemically invalid output. We use RDKit(Landrum, 2016) to transform a SMILES into the molecular graph, whose nodes are atoms and edges are chemical bonds. The input representation of the atom node is a 21-dimensional vector that contains some chemical information about the atom, and the detail can be found in Table 1 (nearly consistent with Gilmer et al. (2017)). The input representation of the edge is a 4-dimensional one-hot vector that encodes the bond types including single, double, triple and aromatic.

Since the SMILES sequence  $(x_{1},\dots,x_{n})$  has been transformed into a graph  $G$  with the input representations  $(h_1,\ldots ,h_N)$  for nodes and  $\{e_{ij}\}$  for edges that exist between node  $i$  and node  $j$ , our GNN will produce new representation  $h_i^\prime$  for each node  $i$  by the following message passing operations:

$$
\alpha_ {i j} = \frac {\exp (\text {L e a k y R e L U} \left(\mathbf {a} ^ {T} \left[ \boldsymbol {W h} _ {i} \| \boldsymbol {W h} _ {j} \| \boldsymbol {e} _ {i j} \right]\right))}{\sum_ {k \in \mathcal {N} _ {i}} \exp (\text {L e a k y R e L U} \left(\mathbf {a} ^ {T} \left[ \boldsymbol {W h} _ {i} \| \boldsymbol {W h} _ {k} \| \boldsymbol {e} _ {i k} \right]\right))}, \tag {6}
$$

$$
\hat {\boldsymbol {h}} _ {i} = \sigma \left(\sum_ {j \in \mathcal {N} _ {i}} \alpha_ {i j} \boldsymbol {W h} _ {j}\right), \tag {7}
$$

where  $\mathbf{a} \in \mathbb{R}^{(2F' + E)}$  is a weight vector for attention mechanism and  $\mathbf{W} \in \mathbb{R}^{F' \times F}$  is a weight matrix for transforming the node features, so  $F$  is the input dimension of nodes,  $F'$  is the output dimension of nodes and  $E$  is the dimension of edges.  $\mathcal{N}_i$  is the set of first-order neighbors of node  $i$  (including itself).  $\sigma$  is an activation function, e.g., ReLU.

In practice, we perform  $K$  multi-head attention (Veličković et al., 2017) to enrich the model capacity and to stabilize the learning process. Each attention head has its own parameters and we average their outputs to get better representation:

$$
\hat {\boldsymbol {h}} _ {i} = \sigma \left(\frac {1}{K} \sum_ {k = 1} ^ {K} \sum_ {j \in \mathcal {N} _ {i}} \alpha_ {i j} ^ {k} \boldsymbol {W} ^ {k} \boldsymbol {h} _ {j}\right). \tag {8}
$$

The above operations can be seen as GAT (Veličković et al., 2017) extended to include edge features. Then, to mitigate the accuracy reduction issue (Kipf & Welling, 2016) caused by stacked graph convolution layers, we adopt the gated skip-connection mechanism (Ryu et al., 2018) to get the final representation:

$$
\boldsymbol {z} _ {i} = \operatorname {s i g m o d} \left(\boldsymbol {U} _ {1} \hat {\boldsymbol {h}} _ {i} + \boldsymbol {U} _ {2} \boldsymbol {h} _ {i} + \boldsymbol {b}\right), \tag {9}
$$

$$
\boldsymbol {h} _ {i} ^ {\prime} = \boldsymbol {z} _ {i} \odot \hat {\boldsymbol {h}} _ {i} + (1 - \boldsymbol {z} _ {i}) \odot \boldsymbol {h} _ {i}, \tag {10}
$$

Table 1: Input representation of atom nodes  

<table><tr><td>Atom Feature</td><td>Description</td></tr><tr><td>Atom type</td><td>C, N, O, S, P, B, F, I, Sn, Cl, Br, Se, Si (one-hot)</td></tr><tr><td>Atom number</td><td>Numbers of protons (integer)</td></tr><tr><td>Acceptor</td><td>Accepts electrons (binary)</td></tr><tr><td>Donor</td><td>Donates electrons (binary)</td></tr><tr><td>Aromatic</td><td>In an aromatic system (binary)</td></tr><tr><td>Hybridization</td><td>sp, sp2, sp3 (one-hot or null)</td></tr><tr><td>Number of Hydrogens</td><td>(integer)</td></tr></table>

where  $U_{1}, U_{2}$  and  $\pmb{b}$  are trainable parameters.

Note that the above operations are just in one layer of our GAES, and we can stack several layers to capture the information about higher-order atom neighbors so that to obtain more comprehensive representations.

# 4.2 TRANSFORMER ENCODER

The transformer encoder is the same as described in section 3, which can capture the sequential representations of molecules (or atoms) represented by SMILES. The original SMILES  $(x_{1},\dots,x_{n})$  is changed to a sequence of vectors  $(p_1,\dots,p_n),p\in \mathbb{R}^d$  after passing the embedding layer. And it will be further updated to vectors  $(p_1',\dots,p_n')$  by the transformer encoder.

# 4.3 REPRESENTATION FUSION

Intuitively, graphical representations reflect the intrinsic structural features of molecules and should be beneficial to generate chemical-valid and more accurate SMILES output. To this end, we propose four fusion strategies to fuse these graphical and sequential embeddings.

![](images/0981f2221438897b629fa06906642800dec82f5866ae27a20ebb3ff2a0808875.jpg)  
Figure 2: Illustration of four fusion strategies in the encoder of GET. The integral encoder is composed of two sub-encoders: graph encoder and transformer encoder. The embedding layer is as described in 4.2. The hidden representation of the target molecule can be obtained in four ways (GET-LT1, GET-LT2, GET-CT and GET-LG).

# 4.3.1 GRAPH LINK TRANSFORMER

As shown in Figure 2 (GET-LT1), we concatenate the atom representations with embeddings of SMILES and perform a linear transformation by weight matrix  $M$ . Then the new representations are sent to the transformer encoder to produce the output of the integral encoder. For non-atomic characters in SMILES, the corresponding atom representations are set to zero vector. Formally,

$$
\hat {\boldsymbol {p}} _ {i} = \left[ \boldsymbol {p} _ {i} \| \boldsymbol {h} _ {i} ^ {\prime} \right], \tag {11}
$$

$$
\left(\boldsymbol {v} _ {1}, \dots , \boldsymbol {v} _ {n}\right) = \left(\boldsymbol {p} _ {1} ^ {\prime}, \dots , \boldsymbol {p} _ {n} ^ {\prime}\right) = \text {T r a n s f o r m e r E n c o d e r} \left(\boldsymbol {M} \hat {\boldsymbol {p}} _ {1}, \dots , \boldsymbol {M} \hat {\boldsymbol {p}} _ {n}\right), \tag {12}
$$

where  $\pmb{h}_i' = \pmb{0}$  if  $x_i$  is non-atomic character,  $\parallel$  is the concatenation operation.  $(\pmb{v}_1, \dots, \pmb{v}_n)$  is integral encoder's output.

Considering that there may exist inconsistency between encoder and decoder since the initial atom features inputted to the graph encoder cannot be utilized by the decoder when making inference, we try another way by replacing the natural atom features  $(h_1,\dots,h_n)$  with  $(p_1,\dots,p_n)$  as the input representations of the graph encoder:

$$
\left(\boldsymbol {h} _ {1} ^ {\prime}, \dots , \boldsymbol {h} _ {n} ^ {\prime}\right) = \operatorname {G r a p h E n c o d e r} \left(\boldsymbol {p} _ {1}, \dots , \boldsymbol {p} _ {n}\right). \tag {13}
$$

In this way, the graph encoder can enhance the original sequential representations  $(\pmb{p}_1,\dots,\pmb{p}_n)$  with molecule structure information directly, but the natural atom features have to be "sacrificed". Then we send the output of the graph encoder to the transformer encoder to get the final output:

$$
\left(\boldsymbol {v} _ {1}, \dots , \boldsymbol {v} _ {n}\right) = \left(\boldsymbol {p} _ {1} ^ {\prime}, \dots , \boldsymbol {p} _ {n} ^ {\prime}\right) = \text {T r a n s f o r m e r E n c o d e r} \left(\boldsymbol {h} _ {1} ^ {\prime}, \dots , \boldsymbol {h} _ {n} ^ {\prime}\right). \tag {14}
$$

We name the first scheme GET-LT1 and the second scheme GET-LT2.

# 4.3.2 GRAPHCONCATENATEWITHTRANSFORMER

As shown in Figure 2 (GET-CT), we concatenate the outputs of the graph encoder and transform encoder, and also perform linear transformation to get the output of the integral encoder:

$$
\hat {\boldsymbol {p}} _ {i} = \left[ \boldsymbol {p} _ {i} ^ {\prime} \| \boldsymbol {h} _ {i} ^ {\prime} \right], \tag {15}
$$

$$
\left(\boldsymbol {v} _ {1}, \dots , \boldsymbol {v} _ {n}\right) = \left(M \hat {\boldsymbol {p}} _ {1} ^ {\prime}, \dots , M \hat {\boldsymbol {p}} _ {n} ^ {\prime}\right). \tag {16}
$$

The notations are consistent with 4.3.1.

# 4.3.3 TRANSFORMER LINK GRAPH

As shown in Figure 2 (GET-LG), the SMILES sequence first pass through the transformer encoder, then it is concatenated with natural atom features  $(h_1, \dots, h_n)$  to be the input representation of the graph encoder. Those non-atomic characters are added into the molecular graph as standalone nodes which do not connect with any other node, and their "atom feature vectors" are just zero vectors. Finally, the graph encoder's output will be the integral encoder's output:

$$
\left(\boldsymbol {v} _ {1}, \dots , \boldsymbol {v} _ {n}\right) = \left(\boldsymbol {h} _ {1} ^ {\prime}, \dots , \boldsymbol {h} _ {n} ^ {\prime}\right) = \operatorname {G r a p h E n c o d e r} \left(\left[ \boldsymbol {p} _ {1} ^ {\prime} \| \boldsymbol {h} _ {1} \right], \dots , \left[ \boldsymbol {p} _ {n} ^ {\prime} \| \boldsymbol {h} _ {n} \right]\right). \tag {17}
$$

# 4.4 DECODER

The decoder is the same as vanilla Transformer's (Vaswani et al., 2017) decoder which has been introduced in section 3. At step  $t$ , the encoder's output  $(v_{1},\dots,v_{n})$  and all previous steps' output  $(x_1',\ldots ,x_{t - 1}')$  are used by the decoder to generate the next SMILES character  $x_{t}^{\prime}$  until  $x_{t}^{\prime} = < EOS >$ .

# 5 EXPERIMENTS

In this section, we evaluate our model for retrosynthesis prediction on a common benchmark dataset USPTO-50K which is derived from USPTO granted patents that includes 50,033 reactions classified into 10 reaction types. A reaction is described as a pair of sequences which consist of SMILES notations for target molecule (with reaction type) and reactants. For example, an heterocycle formation reaction is described as:  $(^{*} <   \mathrm{RX\_4 > c1ccc(-c2nnn[nH]2)cc1},$  "N#Cc1cccccc1.[N-]=[N+]=[N-]) where  $^ { \text{一} } <   \mathrm { R X } _ { - } 4 > "$  represents heterocycle formation reaction, "c1ccc(-c2nnn[nH]2)cc1" is SMILES of the target molecule, "N#c1cccccc1" and "[N-]=[N+]=[N-]" are SMILES of two reactants separated by ".

# 5.1 SETTINGS

- Dataset Split Many previous works (Liu et al., 2017; Coley et al., 2017; Zheng et al., 2019; Karpov et al., 2019) follow a specific split strategy with 40,029, 5,004 and 5,004 reactions for training, validation and testing, and we keep the same.  
- Implementation For the graph encoder, it is implemented based on DGL (Wang et al., 2018). We stack 3 identical layers in our GNN. The input and output dimension of nodes are set to 21 and 256 respectively. The number of multi-head is set to 2; For the transformer encoder and the decoder, we implement them using OpenNMT (Klein et al., 2017), and the parameter settings are presented in the code; Besides, the final dimension of the integral encoder's output  $v$  is set to 256.

# 5.2 RESULT

We compare our model with the vanilla Transformer (Vaswani et al., 2017), Rule-based Expert System mentioned in Liu et al. (2017), Similarity (Coley et al., 2017) and LSTM+Attention (Liu et al., 2017). Note that the results of vanilla Transformer are based on our own experiments since the results reported by previous works (Zheng et al., 2019; Lin et al., 2019; Karpov et al., 2019; Lee et al., 2019) are different from each other. The retrosynthesis prediction accuracy across all classes is provided in Table 2. Moreover, we also test the performance of GET-LT1 when removing the reaction type from the original dataset, and the result is shown in Table 3.

Table 2: Comparison of top-  $n$  accuracies across all classes  

<table><tr><td rowspan="2">Model</td><td colspan="4">top-n accuracy (%)</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>Rule-based Expert System</td><td>35.4</td><td>52.3</td><td>59.1</td><td>65.1</td></tr><tr><td>LSTM+Attention</td><td>37.4</td><td>52.4</td><td>57.0</td><td>61.7</td></tr><tr><td>Similarity</td><td>52.9</td><td>73.8</td><td>81.2</td><td>88.1</td></tr><tr><td>Transformer (baseline)</td><td>54.3</td><td>68.4</td><td>72.0</td><td>74.4</td></tr><tr><td>GET-CT (our)</td><td>55.9</td><td>70.1</td><td>73.2</td><td>76.3</td></tr><tr><td>GET-LG (our)</td><td>54.9</td><td>69.7</td><td>72.2</td><td>74.6</td></tr><tr><td>GET-LT2 (our)</td><td>56.2</td><td>69.4</td><td>72.5</td><td>74.7</td></tr><tr><td>GET-LT1 (our)</td><td>57.4</td><td>71.3*</td><td>74.8*</td><td>77.4*</td></tr></table>

Table 3: Comparison of top-  $n$  accuracies across all classes without reaction type  

<table><tr><td rowspan="2">Model</td><td colspan="4">top-n accuracy (%)</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>Similarity</td><td>37.3</td><td>54.7</td><td>63.3</td><td>74.1</td></tr><tr><td>Transformer (baseline)</td><td>42.3</td><td>57.5</td><td>61.0</td><td>65.7</td></tr><tr><td>GET-LT1 (our)</td><td>44.9</td><td>58.8</td><td>62.4</td><td>65.9</td></tr></table>

Results show that our models outperform all of previous methods in top-1 accuracy, and our best model GET-LT1 achieves the new state-of-the-art among all Seq2Seq-based methods, i.e,

LSTM+Attention, vanilla Transformer and models of GET. Compared with vanilla Transformer, GET-LT1 can improve the prediction accuracy by  $3.1\%$ ,  $2.9\%$ ,  $2.8\%$  and  $3.0\%$  in top-1, top-3, top-5 and top-10 accuracy. Other variants also have varying degrees of performance improvement over vanilla Transformer. And our model can retain this comprehensive superiority after removing the reaction type, demonstrating that molecule structure information can help Transformer to predict more accurate reactants. Besides, note that Similarity (Coley et al., 2017) is a template-based model which predicts 100 candidates and just chooses the top-10 as the final result, while other template-free models, i.e., LSTM+Attention, Transformer and GET, are trained to accurately predict the top-1 output and only generate 10 candidates using beam search. Therefore, it is not surprising that Similarity achieves very high accuracy. Nevertheless, even in this unfair situation, our models can surpass Similarity up to  $4.5\%$  in top-1 accuracy.

In addition, we present the detailed top-10 accuracy of three Seq2Seq-based models (LSTM+Attention, vanilla Transformer and GET-LT1) for each reaction class in Table 4. Results show that our approach can improve vanilla Transformer on 9 of 10 reaction classes by a margin of  $1.3\%$  to  $17.4\%$ , indicating the better generalization ability and comprehensiveness of our model.

Table 4: Comparison of the top-10 accuracy for each reaction class  

<table><tr><td rowspan="2">Model</td><td colspan="10">top-10 accuracy (%)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>LSTM+Attention</td><td>57.5</td><td>74.6</td><td>46.1</td><td>27.8</td><td>80.0</td><td>62.8</td><td>67.8</td><td>69.1</td><td>47.3</td><td>56.5</td></tr><tr><td>Transformer (baseline)</td><td>73.5</td><td>81.9</td><td>62.7</td><td>52.2</td><td>86.1</td><td>71.5</td><td>80.0</td><td>83.9</td><td>65.2</td><td>73.9</td></tr><tr><td>GET-LT1 (our)</td><td>76.6</td><td>84.2</td><td>66.1</td><td>65.6</td><td>89.2</td><td>75.7</td><td>81.3</td><td>81.5</td><td>71.7</td><td>91.3</td></tr></table>

Furthermore, the rate of producing grammatically invalid SMILES for different beam sizes are shown in Table 5 (with reaction type). As can be seen, after fusing molecule structure information, the model is more inclined to generate chemical-valid SMILES compared with vanilla Transformer, since the graphical representations, which directly capture the topological connection of atoms, are able to break the limitation of the SMILES sequence and can give the model additional guidance to produce chemical-valid compound.

Table 5: The rate of producing grammatically invalid SMILES for different beam sizes  

<table><tr><td rowspan="2">Model</td><td colspan="4">invalid SMILES&#x27; rate (%) when beam size k =</td></tr><tr><td>1</td><td>3</td><td>5</td><td>10</td></tr><tr><td>LSTM+Attention</td><td>12.2</td><td>15.3</td><td>18.4</td><td>22.0</td></tr><tr><td>Transformer (baseline)</td><td>3.5</td><td>14.3</td><td>20.3</td><td>30.2</td></tr><tr><td>GET-LT1 (our)</td><td>2.2</td><td>13.4</td><td>19.5</td><td>29.3</td></tr></table>

# 6 CONCLUSION AND FUTURE WORK

We propose Graph Enhanced Transformer(GET), an effective framework that successfully combines the graphical and sequential representations of the molecule to improve the retrosynthesis prediction performance. Experiments indicate that our model outperforms state-of-the-art Seq2Seq-based methods on USPTO-50K dataset, and shows promising ability in reducing invalid SMILES rate.

In the future, we plan to 1) explore how to utilize the molecular graph information in the decoder. 2) research how to let the decoder generate reactants in the form of graph directly.

# REFERENCES

Connor W Coley, Luke Rogers, William H Green, and Klavs F Jensen. Computer-assisted retrosynthesis based on molecular similarity. ACS central science, 3(12):1237-1245, 2017.  
EJ Corey and W Todd Wipke. Computer-assisted design of complex organic syntheses. Science, 166(3902):178-192, 1969.

Kien Do, Truyen Tran, and Svetha Venkatesh. Graph transformation policy network for chemical reaction prediction. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 750-760. ACM, 2019.  
David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. Convolutional networks on graphs for learning molecular fingerprints. In Advances in neural information processing systems, pp. 2224-2232, 2015.  
Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1263-1272. JMLR.org, 2017.  
Wengong Jin, Connor Coley, Regina Barzilay, and Tommi Jaakkola. Predicting organic reaction outcomes with weisfeiler-lehman network. In Advances in Neural Information Processing Systems, pp. 2607-2616, 2017.  
Pavel Karpov, Guillaume Godin, and Igor V Tetko. A transformer model for retrosynthesis. In International Conference on Artificial Neural Networks, pp. 817-830. Springer, 2019.  
Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.  
Guillaume Klein, Yoon Kim, Yuntian Deng, Jean Senellart, and Alexander M Rush. Opennmt: Open-source toolkit for neural machine translation. arXiv preprint arXiv:1701.02810, 2017.  
G Landrum. Rdkit: open-source cheminformatics software, 2016.  
James Law, Zsolt Zsoldos, Aniko Simon, Darryl Reid, Yang Liu, Sing Yoong Khew, A Peter Johnson, Sarah Major, Robert A Wade, and Howard Y Ando. Route designer: a retrosynthetic analysis tool utilizing automated retrosynthetic rule generation. Journal of chemical information and modeling, 49(3):593-602, 2009.  
AA Lee, Q Yang, V Sresht, P Bolgar, X Hou, JL Klug-McLeod, and CR Butler. Molecular transformer unifies reaction prediction and retrosynthesis across pharma chemical space. Chemical communications (Cambridge, England), 2019.  
Kangjie Lin, Youjun Xu, Jianfeng Pei, and Luhua Lai. Automatic retrosynthetic pathway planning using template-free models. arXiv preprint arXiv:1906.02308, 2019.  
Bowen Liu, Bharath Ramsundar, Prasad Kawthekar, Jade Shi, Joseph Gomes, Quang Luu Nguyen, Stephen Ho, Jack Sloane, Paul Wender, and Vijay Pande. Retrosynthetic reaction prediction using neural sequence-to-sequence models. ACS central science, 3(10):1103-1113, 2017.  
Seongok Ryu, Jaechang Lim, Seung Hwan Hong, and Woo Youn Kim. Deeply learning molecular structure-property relationships using attention-and gate-augmented graph convolutional network. arXiv preprint arXiv:1805.10988, 2018.  
John S Schreck, Connor W Coley, and Kyle JM Bishop. Learning retrosynthetic planning through self-play. arXiv preprint arXiv:1901.06569, 2019.  
Marwin HS Segler and Mark P Waller. Neural-symbolic machine learning for retrosynthesis and reaction prediction. Chemistry-A European Journal, 23(25):5966-5971, 2017.  
Marwin HS Segler, Mike Preuss, and Mark P Waller. Planning chemical syntheses with deep neural networks and symbolic ai. Nature, 555(7698):604, 2018.  
Sara Szymkuc, Ewa P Gajewska, Tomasz Klucznik, Karol Molga, Piotr Dittwald, Michal Startek, Michal Bajczyk, and Bartosz A Grzybowski. Computer-assisted synthetic planning: The end of the beginning. Angewandte Chemie International Edition, 55(20):5904-5937, 2016.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.

Petar Velicković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.  
Minjie Wang, Lingfan Yu, Quan Gan, Da Zheng, Yu Gai, Zihao Ye, Mufei Li, Jinjing Zhou, Qi Huang, Junbo Zhao, Haibin Lin, Chao Ma, Damon Deng, Qipeng Guo, Hao Zhang, Jinyang Li, Alexander J Smola, and Zheng Zhang. Deep graph library, 2018. URL http://dgl.ai.  
David Weininger. Smiles, a chemical language and information system. 1. introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1):31-36, 1988.  
Shuangjia Zheng, Jiahua Rao, Zhongyue Zhang, Jun Xu, and Yuedong Yang. Predicting retrosynthetic reaction using self-corrected transformer neural networks. arXiv preprint arXiv:1907.01356, 2019.