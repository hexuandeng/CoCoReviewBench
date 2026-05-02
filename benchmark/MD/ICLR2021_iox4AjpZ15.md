# INVERTIBLE MANIFOLD LEARNING FOR DIMENSION REDUCTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

It is widely believed that a nonlinear dimension reduction (NLDR) process drops information inevitably in most practical scenarios, and even with the manifold assumption, most existing methods are unable to preserve structure of data after DR due to the loss of information, especially in high-dimensional cases. In the context of manifold learning, we think a good low-dimensional representation should preserve topological and geometric properties of data manifold. To achieve this, the invariability of a NLDR transformation is required such that the learned representation is reconstructible via its inverse transformation. In this paper, we propose a novel method, called invertible manifold learning (inv-ML), to tackle this problem. A locally isometric smoothness (LIS) constraint for preserving local geometry is applied to a two-stage inv-ML algorithm. Firstly, a homeomorphic sparse coordinate transformation is learned to find the low-dimensional representation without loss of topological information. Secondly, a linear compression is performed on the learned sparse coding, with the trade-off between the target dimension and the incurred information loss. Experiments are conducted on seven datasets, whose results demonstrate that the proposed inv-ML not only achieves better invertible NLDR in comparison with typical existing methods but also reveals the characteristics of the learned manifolds through linear interpolation in latent space. Moreover, we find that the reliability of tangent space approximated by its local neighborhood on real-world datasets is a key to the success of manifold based DR algorithms. The code will be made available soon.

# 1 INTRODUCTION

In real-world scenario, it is widely believed that the loss of data information is inevitable after dimension reduction (DR), though the goal of DR is to preserve as much information as possible in the low-dimensional space. In the case of linear DR, compressed sensing (Donoho, 2006) breaks this common sense with practical sparse conditions of the given data. In the case of nonlinear dimension reduction (NLDR), however, it has not been clearly discussed, e.g. what is the structure within data and how to maintain these structure after DR? From the perspective of manifold learning, the manifold Assumption is widely adopted, but classical manifold based DR algorithms are usually performed not well in the complex practical case. Therefore, what is the gap between theoretical and real-world applications of manifold learning? Here, we give the first detailed discussion of this two problem in the context of manifold learning. We think that a good low-dimensional representation should preserve the topology and geometry of input data, and thus introduce a homeomorphism based invertible NLDR process, combining sparse coordinate transformation and local isometry smoothness (LIS) constraints which preserve the property of topology and geometry, to explain the information-lossless NLDR in manifold learning theoretically. We instantiate the proposed inv-ML as a neural network called  $i$ -ML-Enc via a cascade of equi-dimensional layers and a linear compression layer, and conduct sufficient experiments to validate the invertible DR ability of  $i$ -ML-Enc and analyze inherent difficulties of classical manifold learning.

Topology preserving dimension reduction. To start, we first make out the definition of DR on a manifold.  $f:\mathcal{M}_0^d\to \mathbb{R}^m$  is a smooth mapping of a differential manifold into another, and if  $f$  is a homeomorphism of  $\mathcal{M}_0^d$  into  $\mathcal{M}_1^d = f(\mathcal{M}_0^d)\subset \mathbb{R}^m$ , we call  $f$  is an embedding of  $\mathcal{M}_0^d$  into  $\mathbb{R}^m$ . Assume that the data set  $\mathcal{X} = \{\pmb {x}_j|1\leq j\leq n\}$  sampled from the compact manifold  $\mathcal{M}_1^d\subset \mathbb{R}^m$

which we call the data manifold and is homeomorphic to  $\mathcal{M}_0^d$ . For the sample points we get are represented in the coordinate after inclusion mapping  $i_1$ , we can only regard them as points from Euclidean space  $\mathbb{R}^m$  without any prior knowledge. According to the Whitney Embedding Theorem (Seshadri & Verma, 2016),  $\mathcal{M}_0^d$  is can be embedded smoothly into  $\mathbb{R}^{2d}$  by a homeomorphism  $g$ . Rather than to find the  $f^{-1}: \mathcal{M}_1^d \to \mathcal{M}_0^d$ , our goal is to seek a smooth map  $h: \mathcal{M}_1^d \to \mathbb{R}^s \subset \mathbb{R}^{2d}$ , where  $h = g \circ f^{-1}$  is a homeomorphism of  $\mathcal{M}_1^d$  into  $\mathcal{M}_2^d = h(\mathcal{M}_1^d)$  and  $d \leq s \leq 2d \ll m$ , and thus the  $\dim(h(\mathcal{X})) = s$ , which achieves the NLDR while preserving the topology. Owing to the homeomorphism  $h$  we seek as a NLDR mapping, the data manifold  $\mathcal{M}_1^d$  is reconstructible via  $\mathcal{M}_1^d = h^{-1} \circ h(\mathcal{M}_1^d)$ , by which we mean  $h$  a topology preserving NLDR as well as information-lossless NLDR.

![](images/c83c9357209e9f6f64a363fcc2c5df697ac129d9280a536d335cfd641b91c0c0.jpg)  
Figure 1: Illustration of the process of NLDR.  $\pmb{x}$  is sampled from  $\mathcal{M}_1^d$  and represented in the Euclidean space  $\mathbb{R}^m$  after an inclusion mapping  $i_i$ . For the topology preserving dimension reduction methods, it aims to find a homeomorphism  $g \circ f^{-1}$  to map  $\pmb{x}$  into  $\pmb{z}$  which is embedded in  $\mathbb{R}^s$ .

Geometry preserving dimension reduction. While the topology of the data manifold  $\mathcal{M}_1^d$  can be preserved by the homeomorphism  $h$  discussed above, it may distort the geometry. To preserve the local geometry of the data manifold, the map should be isometric on the tangent space  $\mathcal{T}_p\mathcal{M}_1^d$  for every  $p\in \mathcal{M}_1^d$ , indicating that  $d_{\mathcal{M}_1^d}(u,v) = d_{\mathcal{M}_2^d}(h(u),h(v)),\forall u,v\in \mathcal{T}_p\mathcal{M}_1^d$ . By Nash's Embedding Theorem (Nash, 1956), any smooth manifold of class  $C^k$  with  $k\geq 3$  and dimension  $d$  can be embedded isometrically in the Euclidean space  $\mathbb{R}^s$  with  $s$  polynomial in  $d$ .

# 2 RELATED WORK

Manifold learning. Most classical linear or nonlinear DR methods aim to preserve geometric properties of the manifold. The Isomap (Tenenbaum et al., 2000) based methods aim to preserve the global metric between every pairs of sample points. For example, McQueen et al. (2016) can be regarded as such methods based on the push-forward Riemannian metric. For the other aspect, LLE (Roweis & Saul, 2000) based methods try to preserve local geometry after DR, whose derivatives like LTSA (Zhang & Zha, 2004), MLLE (Zhang & Wang, 2007), etc. have been widely used but usually fail in the high-dimensional case. Recently, based on the local properties of mainifold, MLDL (Li et al., 2020) has been proposed as a robust locally geometry preserving method implemented by neural network, abandoning the retention of topology. In contrast, our method takes preservation of geometry as well as topology into consideration, trying to maintain these properties of manifolds even in excessive dimension reduction, where the target dimension  $s'$  is smaller than  $s$ .

Invertible model. From AutoEncoder (AE) (Hinton & Salakhutdinov, 2006), the fundamental neural network based model, having achieved DR and cut information loss by minimizing the reconstruction loss, some AE based generative models like VAE (Kingma & Welling, 2014) and manifold-based NLDR models like TopoAE (Moor et al., 2020) has emerged. These methods cannot avoid information loss after NLDR, and thus, some invertible models consist of a series of equi-dimensional layers have been proposed, some of which aim to generate samples by density estimation through layers (Dinh et al., 2015) (Dinh et al., 2017) (Behrmann et al., 2019), and the other of which are established for other targets, e.g. validating the mutual information bottleneck (Jacobsen et al., 2018). Different from methods mentioned above, our proposed  $i$ -ML-Enc is a neural network based encoder, with DR as well as maintaining structure of raw data points based on manifold assumption via a series of equi-dimensional layers.

Compressed sensing. The Johnson-Lindenstrauss Theorem (Johnson & Lindenstrauss, 1984) provides the lower bound of target dimension for linear DR with the pairwise distance loss. Given a small constant  $\epsilon \in (0,1)$  and  $n$  samples  $\{\pmb{x}_i\}_{i=1}^n$  in  $\mathbb{R}^m$ , a linear projection  $W: \mathbb{R}^m \to \mathbb{R}^s$ ,  $s > O\left(\frac{\log m}{\epsilon^2}\right)$  can be found, which embeds samples into a  $s$ -dimensional space with  $(1 + \epsilon)$  distortion of any sample

pairs  $(\pmb{x}_i,\pmb{x}_j)$ . It adopts a prior assumption that the given samples in high-dimensional space have a relevant low-dimensional structure constraint which can be maintained by keeping the pairwise distance. Further, compressed sensing (CS) provides strict sparse conditions of linear DR with great probability to recover the compressed signal, which usually cooperates with sparse dictionary learning (Hawe et al., 2013). The core of CS is Restricted Isometry Property (RIP) condition, which reads

$$
(1 - \epsilon) \| \boldsymbol {x} _ {1} - \boldsymbol {x} _ {2} \| ^ {2} \leq \| W (\boldsymbol {x} _ {1} - \boldsymbol {x} _ {2}) \| ^ {2} \leq (1 + \epsilon) \| \boldsymbol {x} _ {1} - \boldsymbol {x} _ {2} \| ^ {2}, \tag {1}
$$

where  $\epsilon \in (0,1)$  is a rather small constant and  $W$  is a linear measurement of signal  $\pmb{x}_1$  and  $\pmb{x}_2$ . Given a signal  $\pmb{x} \in \mathbb{R}^m$  with  $s$ -sparse representation  $\alpha = \Phi x$  on an  $m$ -dimensional orthogonal basis  $\Phi$ ,  $\alpha$  can be recovered from the linear measurement  $\pmb{y} = W\alpha$  with great probability by the sparse optimization if  $W_{m \times s}$  satisfies the RIP condition:  $\arg \min_{\tilde{\alpha}} ||\tilde{\alpha}||_0$ , s.t.  $y = W\tilde{\alpha}$ . The linear measurement is rewritten as  $\pmb{y} = \Psi \Phi \alpha = \Psi \pmb{x}$  where  $\Psi$  is a low-dimensional orthogonal basis and  $\Phi$  can be found by the nonlinear dictionary learning. There are some reconstructible CS-based NLDR methods (Wei et al., 2015) (Wei et al., 2019), but their performance is not impressive.

# 3 PROPOSED METHOD

We will specifically discuss the proposed two stage NLDR as the first stage in Sec 3.1, in which a  $s$ -dimensional representation is learned by a homeomorphism transformation while keeping all topological and geometric structure of the data manifold; then give applicable conditions in real-world scenarios as the second stage in Sec 3.2, in which the dimension is further compressed to  $s'$ . We instantiate the proposed DR process as a neural network based on ML-Enc (Li et al., 2020) in Sec 3.3.

![](images/3fdf4e160e6e7df226f945b0fbc401ca9b43b6141ee6af4a741ccde0092ff921.jpg)  
Figure 2: The network structure for the proposed methods. The first  $L - 1$  layers equi-dimensional mapping in the green box are the first stage which achieves  $s$ -sparse and has inverse process in the purple box. (a) is a layer of nonlinear homeomorphism transformation. (b) linearly transforms  $s$ -sparse representation in  $\mathbb{R}^m$  into  $\mathbb{R}^{s'}$  in the second stage. (c) are the extra heads. (d) indicates the padding loss of the  $l$ -th layer to force  $d^{(l)}$ -sparse.

# 3.1 TOPOLOGY AND GEOMETRY PRESERVATION

Canonical embedding for homeomorphism. To seek the smooth homeomorphism  $h$ , we turn to the theorem of local canonical form of immersion (Mei, 2013). Let  $f: \mathcal{M} \to \mathcal{N}$  an immersion, and for any  $p \in \mathcal{M}$ , there exist local coordinate systems  $(U, \phi)$  around  $p$  and  $(V, \psi)$  around  $f(p)$  such that  $\psi \circ f \circ \phi^{-1}: \phi(U) \to \psi(V)$  is a canonical embedding, which reads

$$
\psi \circ f \circ \phi^ {- 1} (x ^ {1}, x ^ {2}, \dots , x ^ {d}) = (x ^ {1}, x ^ {2}, \dots , x ^ {d}, 0, 0, \dots , 0).
$$

In our case, let  $\mathcal{M} = \mathcal{M}_2^d$ , and  $\mathcal{N} = \mathcal{M}_1^d$ , any point  $z = (z^1, z^2, \dots, z^s) \in \mathcal{M}_1^d \subset \mathbb{R}^s$  can be mapped to a point in  $\mathbb{R}^m$  by the canonical embedding

$$
\psi \circ h ^ {- 1} \left(z ^ {1}, z ^ {2}, \dots , z ^ {s}\right) = \left(z ^ {1}, z ^ {2}, \dots , z ^ {s}, 0, 0, \dots , 0\right). \tag {2}
$$

For the point  $z$  is regarded as a point in  $\mathbb{R}^s$ ,  $\phi = \mathbb{I}$  is an identity mapping, and for  $h = g \circ f^{-1}$  is a homeomorphism,  $h^{-1}$  is continuous. The Eq. 2 can be written as

$$
\begin{array}{l} \left(z ^ {1}, z ^ {2}, \dots , z ^ {s}\right) = h \circ \psi^ {- 1} \left(z ^ {1}, z ^ {2}, \dots , z ^ {s}, 0, 0, \dots , 0\right) \\ = h (x ^ {1}, x ^ {2}, \dots , x ^ {m}) \\ \end{array}
$$

Therefore, to reduce  $dim(\mathcal{X}) = m$  to  $s$ , we can decompose  $h$  into  $\psi$  and  $h \circ \psi^{-1}$ , by firstly finding a homeomorphic coordinate transformation  $\psi$  to map  $\pmb{x} = (x^{1}, x^{2}, \dots, x^{m})$  into  $\psi(\pmb{x}) = (z^{1}, z^{2}, \dots, z^{s}, 0, 0, \dots, 0)$ , which is called a sparse coordinate transformation, and  $h \circ \psi^{-1}$  can be easily obtained by Eq. 2. We denote  $h \circ \psi^{-1}$  by  $h_{0}$  and call it a sparse compression. The theorem holds for any manifold, while in our case, we aims to find the mapping of  $\mathcal{X} \subset \mathbb{R}^{m}$  into  $\mathbb{R}^{s}$ , so the local coordinate systems can be extended to the whole space of  $\mathbb{R}^{m}$ .

Local isometry constraint. The local isometry proposed by MLDL (Li et al., 2020) is employed, imposing the prior LIS constraint, which aims to preserve distances (or some other metrics) locally so that  $d_{\mathcal{M}_1^d}(u,v) = d_{\mathcal{M}_2^d}(h(u),h(v)), \forall u,v \in \mathcal{T}_p\mathcal{M}_1^d$ .

# 3.2 LINEAR COMPRESSION

With the former discussed method, manifold-based NLDR can be achieved with topology and geometry preserved, i.e.  $s$ -sparse representation in  $\mathbb{R}^m$ . However, the target dimension  $s'$  may be even less than  $s$ , further compression can be performed through the linear compression  $h_0': \mathbb{R}^m \to \mathbb{R}^{s'}$  instead of sparse compression, where  $h_0'(z) = W_{m \times s'} z$ , with minor information loss. In general, the sparse compression is a particular case of linear compression with  $h_0(z) = h_0'(z) = \Lambda z$ , where  $\Lambda = (\delta_{i,j})_{m \times s}$  and  $\delta_{i,j}$  is the Kronecker delta. We discuss the information loss caused by a linear compression under different target dimensions  $s'$  as following cases.

Theoretical case. In the case of  $d \leq s \leq s'$ , based on compressed sensing, we can reconstruct the raw input data after NLDR process without loss of any information by solving the sparse optimization problem mensioned in Sec. 2 when the transformation matrix  $W_{m \times s'}$  has full rank of column. In the case of  $d \leq s' < s$ , it is inevitable to drop the topological properties because the two spaces before and after NLDR are not homeomorphic, and it is reduced to local geometry preservation by LIS constraint. However, in the case of  $s' \leq d < s$ , both topological and geometric information is lost to varying degrees. Therefore, we can only try to retain as much geometric structure as possible.

Practical case. In real-world scenarios, the target dimension  $s'$  is usually lower than  $s$ , even lower than  $d$ . Meanwhile, the data sampling rate is quite low, and clustering effect is extremely significant, indicating it is possible to approximate  $\mathcal{M}_1$  by low-dimensional hyper-plane in the Euclidean space. In the case of  $s' < s$ , we can retain as the prior Euclidean topological structure as additional topological information of the raw data points.

![](images/cf37ffac3cbe132201d44e964e92db192f47cd2a2ae708e201e51e4375bff12b.jpg)  
Figure 3: Sparsity and clustering effect.

# 3.3 NETWORK FOR IMPLEMENTATION

Based on Sec 3.1 and Sec 3.2, we propose a neural network  $i$ -ML-Enc which achieves two stages NLDR preserving both topology and geometry, as shown in Fig. 2. In this section, we will introduce the function of network structures and loss functions respectively, including the orthogonal loss, padding loss and extra heads for the first stage, and the LIS loss, push-away loss for the second stage.

Cascade of homeomorphisms. Since the sparse coordinate transformation  $\psi$  (and its inverse) can be highly nonlinear and complex, we decompose it into a cascade of  $L - 1$  isometric homeomorphisms  $\psi = \psi^{(L - 1)}\circ \dots \circ \psi^{(2)}\circ \psi^{(1)}$ , which can be achieved by  $L - 1$  equi-dimensional network layers. For each  $\psi^{(l)}$ , it is a sparse coordinate transformation, where  $\psi^l (z^{1,(l)},z^{2,(l)},\dots ,z^{s_l,(l)},0,\dots ,0) = (z^{1,(l + 1)},z^{2,(l + 1)},\dots ,z^{s_{l + 1},(l + 1)},0,\dots ,0)$  with  $s_{l + 1} < s_l$  and  $s_{L - 1} = s$ . The layer-wise transformation  $Z^{(l + 1)} = \psi^{(l)}(Z^{(l)})$  and its inverse is written as

$$
Z ^ {(l + 1)} = \sigma \left(W _ {l} X ^ {(l)}\right), Z ^ {(l) ^ {\prime}} = W _ {l} ^ {- 1} \left(\sigma^ {- 1} \left(Z ^ {(l + 1) ^ {\prime}}\right)\right), \tag {3}
$$

in which  $W_{l}$  is the  $l$ -th weight matrix of the neural network to be learned, and  $\sigma(..)$  is a nonlinear activation. The bias term is removed here to facilitate its simple inverse structure.

Orthogonal loss. Each layer-wise transformation is thought to be a homeomorphism between  $Z^{(l)}$  and  $Z^{(l + 1)}$ , and we want it to be a nearly isometric. We force each  $W_{l}$  to be an orthogonal matrix, which allows simple calculation of the inverse of  $W_{l}$ . Based on RIP condition, the orthogonal constraint of the weight matrix in the first  $L - 1$  layers can be obtained as

$$
L _ {o r t h} = \sum_ {l = 1} ^ {L - 1} \alpha^ {(l)} \rho (W _ {l} ^ {T} W _ {l} - I),
$$

where  $\{\alpha^{(l)}\}$  are the loss weights. Notice that  $\rho(W) = \sup_{z \in \mathbb{R}^m, z \neq \mathbf{0}} \frac{|Wz|}{|z|}$  is the spectral norm of  $W$ , and the loss term can be written as  $\rho(W_l^T W_l - I) = \sup_{z \in \mathbb{R}^m, z \neq \mathbf{0}} \frac{|Wz|}{|z|}$  which is equivalent to RIP condition.

Padding loss. To force sparsity from the second to  $L - 1$ -th layers, we add a zero padding loss to each of these layers. For the  $l$ -th layer whose target dimension is  $s^{(l)}$ , pad the last  $m - s^l$  elements of  $z^{(l + 1)}$  with zeros and panish these elements with  $L_{1}$  norm loss:

$$
L _ {p a d} = \sum_ {l = 2} ^ {L - 1} \beta^ {(l)} \sum_ {i = s ^ {(l)}} ^ {m} | \boldsymbol {z} _ {i} ^ {(l + 1)} |,
$$

where  $\{\beta^{(l)}\}$  are loss weights. The target dimension  $s^{(l)}$  can be set heuristically.

Linear transformation head. We use the linear transformation head to achieve the linear compression step in our NLDR process, which is a transformation between the orthogonal basis of high dimension and lower dimension. Thus, we apply the row orthogonal constraint to  $W_{L}$ .

LIS loss. Since the linear DR is applied at the end of the NLDR process, we apply LIS constraint to preserve the local geometric properties. Take the LIS loss in the  $l$ -th layer as an example:

$$
L _ {L I S} (\psi) = \sum_ {i = 1} ^ {n} \sum_ {j \in \mathcal {N} _ {i} ^ {k}} \| d _ {X} (\boldsymbol {x} _ {i}, \boldsymbol {x} _ {j}) - d _ {Z} (\psi (\boldsymbol {x} _ {i}), \psi (\boldsymbol {x} _ {j})) \|,
$$

where  $\mathcal{N}_i^k$  is the set of  $x_{i}$ 's  $k$ -nearest neighborhood.

Push-away loss. In the real case discussed in Sec 3.2, the latent space of the  $L - 1$ -th layer can approximately be a hyper-plane in high-dimensional, so that we introduce push-away loss to repel the non-adjacent sample points of each  $x_{i}$  in its  $B$ -radius neighbourhood in the latent space. It deflates the manifold locally when acting together with  $L_{LIS}$ . Similarly,  $L_{push}$  is applied after the linear transformation in the  $L$ -th layer:

$$
L _ {p u s h} = - \sum_ {i = 1} ^ {n} \sum_ {j \in \mathcal {N} _ {i} ^ {k}} \mathbf {1} _ {d _ {Z} (\boldsymbol {x} _ {i} ^ {(l)}, \boldsymbol {x} _ {j} ^ {(l)}) <   B} \log \left(1 + d _ {Z} (\boldsymbol {x} _ {i} ^ {(l)}, \boldsymbol {x} _ {j} ^ {(l)})\right).
$$

Extra heads. In order to force the first  $L - 1$  layers of the network to achieve NLDR gradually, we introduce auxiliary DR branches, called extra head, at layers from the second to the  $L - 1$ -th. The structure of each extra head is same as the linear transformation head and will be discarded after training.  $L_{extra}$  is written as

$$
L _ {e x t r a} = \sum_ {l = 1} ^ {L - 1} \gamma^ {(l)} \left(L _ {L I S} + \mu^ {(l)} L _ {p u s h}\right),
$$

where  $\{\gamma^{(l)}\}$  and  $\{\mu^{(l)}\}$  are loss weights which can be set based on  $\{s^{(l)}\}$ .

Inverse process. The inverse process is the decoder directly obtained by the first  $L - 1$  layers of the encoder given by Eq. 3, which does not involve in the training process. When the target dimension  $s'$  is equal to  $s$ , the inverse of the layer-  $L$  can be solved by compressed sensing rather than pseudo-inverse.

# 4 EXPERIMENT

To evaluate the invertibility of  $i$ -ML-Enc and analyze the property of data manifolds, we carry out experiments on seven datasets: (i) Swiss roll (Pedregosa et al., 2011), (ii) Spheres (Moor et al., 2020) and Half Spheres, (iii) USPS (Hull, 1994), (iv) MNIST (LeCun et al., 1998), (v) KMNIST (Clanuwat et al., 2018), (vi) FMNIST (Xiao et al., 2017), (vii) COIL-20 (Nene et al., 1996b). The implementation is based on the PyTorch 1.3.0 library running on NVIDIA v100 GPU.

# 4.1 METHODS COMPARISON

We compare the proposed  $i$ -ML-Enc with several typical methods in NLDR and inverse scenarios on Swiss roll, Spheres and Half Spheres, USPS, MNIST, FMNIST and COIL-20 datasets. Six methods for manifold learning: MLLE (Zhang & Wang, 2007), t-SNE (Maaten & Hinton, 2008) and ML-Enc (Li et al., 2020) are compared for NLDR; three AE-based methods VAE (Kingma & Welling, 2014), TopoAE (Moor et al., 2020) and ML-AE (Li et al., 2020) are compared for reconstructible manifold learning. Three methods for inverse models: INN (Nguyen et al., 2019), i-RevNet (Jacobsen et al., 2018), and i-ResNet (Behrmann et al., 2019) are compared for bijective property. Among them, i-RevNet and i-ResNet are supervised algorithms while the rest are unsupervised. Hyperparameter values of  $i$ -ML-Enc and configurations of these datasets such as the input and target dimension are provided in Appendix A.2.

Table 1: Comparison in representation and invertible quality on MNIST datasets  

<table><tr><td>Dataset</td><td>Algorithm</td><td>RMSE</td><td>MNE</td><td>Trust</td><td>Cont</td><td>Kmin</td><td>Kmax</td><td>l-MSE</td><td>Acc</td></tr><tr><td rowspan="11">MNIST</td><td>MLLE</td><td>-</td><td>-</td><td>0.6709</td><td>0.6573</td><td>1.873</td><td>6.7e+9</td><td>36.80</td><td>0.8341</td></tr><tr><td>t-SNE</td><td>-</td><td>-</td><td>0.9896</td><td>0.9886</td><td>5.156</td><td>324.9</td><td>48.07</td><td>0.9246</td></tr><tr><td>ML-Enc</td><td>-</td><td>-</td><td>0.9862</td><td>0.9927</td><td>1.761</td><td>58.91</td><td>18.98</td><td>0.9326</td></tr><tr><td>VAE</td><td>0.5263</td><td>33.17</td><td>0.9712</td><td>0.9703</td><td>5.837</td><td>130.5</td><td>22.79</td><td>0.8652</td></tr><tr><td>TopoAE</td><td>0.5178</td><td>31.45</td><td>0.9915</td><td>0.9878</td><td>4.943</td><td>265.3</td><td>24.98</td><td>0.8993</td></tr><tr><td>ML-AE</td><td>0.4012</td><td>16.84</td><td>0.9893</td><td>0.9926</td><td>1.704</td><td>57.48</td><td>19.05</td><td>0.9340</td></tr><tr><td>i-ML-Enc (L)</td><td>0.0457</td><td>0.5085</td><td>0.9906</td><td>0.9912</td><td>2.033</td><td>60.14</td><td>18.16</td><td>0.9316</td></tr><tr><td>INN</td><td>0.0615</td><td>0.5384</td><td>0.9851</td><td>0.9823</td><td>1.875</td><td>22.38</td><td>7.494</td><td>0.9176</td></tr><tr><td>i-RevNet</td><td>0.0443</td><td>0.4679</td><td>0.9118</td><td>0.8785</td><td>13.41</td><td>142.5</td><td>6.958</td><td>0.9901</td></tr><tr><td>i-ResNet</td><td>0.0502</td><td>0.6422</td><td>0.9149</td><td>0.8922</td><td>1.876</td><td>19.28</td><td>10.78</td><td>0.9925</td></tr><tr><td>i-ML-Enc(L-1)</td><td>0.0407</td><td>0.5085</td><td>0.9986</td><td>0.9973</td><td>1.256</td><td>5.201</td><td>5.895</td><td>0.9580</td></tr></table>

![](images/7db780bac8da4580462442e4f7320348e6aba6729e5bb8a6e537c466662943f6.jpg)  
(a) Swiss roll

![](images/c0af2dccfc9292552808047f8990fbf6161930e1aea5008dac7e2a0638e269c0.jpg)  
i-ML-Enc,  $L_{3}$

![](images/d6a76476c781fba14b56a87d1b52cc55624d2579cb0aa770049533d921bc2174.jpg)  
i-ML-Enc,  $L_{6}$

![](images/e8b1fd8b16e3a01680d5f3a78df2cd46b46c3578efdd71884e977a68d1cd8436.jpg)  
i-ML-Enc,  $L_{8}$

![](images/90c22ac1c92a563d46955e7a1faa70801ae1ba475a38d7cd2b68885a00c32199.jpg)

![](images/48d8a0e39074e55bbc908711689c9af1df80387f3ae54c407dab04bb4dd0773c.jpg)

![](images/1757502471f8534e698e416100079d711d85f0bf13676c52aed91be7da025c0a.jpg)

![](images/50f359a2262f3df86baaa39f13b5805cb9723d059208cbd268afd8c2de35642e.jpg)

![](images/40ea02b33bee1a13431f761da6ac5215042b0aca2f8ad714052f100848572e22.jpg)  
(b) Spere (PCA)  
(d) MNIST (PCA)  
Figure 4: Visualization of invertible NLDR results of  $i$ -ML-Enc compared to ML-Enc and t-SNE. (a) shows the NLDR and its inverse process of on test set of Swiss roll with the target dimension  $s' = 2$ ; (b) shows the failure case of reducing 101-D Spheres  $S^{100}$  into 10-D, while (c) shows DR of 101-D half spheres  $S^{10}$  with  $s' = 10$ ; (d) and (e) show results of the relatively dense case on MNIST and sparse case on COIL-20 with  $s' = 2$ . The high-dimensional results are visualized by PCA.

![](images/6e8080444dc14f052ff8766770736753169e3ee5e476242c23dca8e82c1744dc.jpg)  
i-ML-Enc,  $L_{7}$  
i-ML-Enc,  $L_{8}$

![](images/ab9d9c0baf8b3bd3b3812669b430b8012779c2579dae35ee3d162d2548a354ad.jpg)  
i-ML-Enc,  $L_{8}$  
ML-Enc

![](images/bcf9e043ea393e8b3d898ec1b0c68d1c7d51193998248003c55430bedd203847.jpg)  
ML-Enc  
t-SNE

![](images/ae0a46dcb8fa1507fa23c5b6dc358d14aae5e7707674983725bfcc3db5f59a8d.jpg)  
i-ML-Dnc,  $L_6^{\prime}$

![](images/f62f86edc1c355e8e168286f90a8e06374f8fd90bb354b6e9d0da452ae85dc1b.jpg)  
i-ML-Dnc,  $L_3^{\prime}$

![](images/168c9f1f49c2398d44e18f09e70926da54a31cf3a6dda7734e078162a3511019.jpg)  
reconstruction

![](images/482220f995a4a8f37c89efc03592c2efe397b8d85b8d59f1debebdb075b5425f.jpg)  
ML-Enc

![](images/7c7ea9c6daf61373ad7c6e369c3254a16f0dca590020d8e7472292c8b02b099f.jpg)

![](images/d28efacc281e732af2a3cd403e3cca7a0a64455496095dd68d205a14784b8ae9.jpg)

![](images/93b1725d9a759713c58200b43a99d6f76b058cbc0080503fffbb23c621d4e851.jpg)

![](images/fedc3cd631c92952e4e16abbaade28f995b98af433005ec97c7f782a34c59b5c.jpg)

![](images/120d2cd99adff0d1156d31b632fb24ab8230e869a2fbe302c312e69bdb5d8ad9.jpg)  
(c) Half Spere (PCA)  
(e) COIL-20 (PCA)

![](images/49053d060aef621e2276da2f35c7dfdeeeda4e1db6769afe9fd9f0643ee36ffe.jpg)  
i-ML-Enc,  $L_{7}$  
i-ML-Enc,  $L_{6}$

![](images/e7cba7a52c60e20a4e0b4ab1e4b8799e7c36b866d0e43e4884ab76ba6c016031.jpg)  
i-ML-Enc,  $L_{8}$  
ML-Enc

![](images/83fe4b9149f666c6fa88ef1e111b5aa03496a5a18ee590a8f4fd8917e838f303.jpg)  
ML-Enc  
t-SNE

Evaluation metrics. We evaluate the proposed invertible NLDR algorithm from three aspects: (1) Invertible property. Reconstruction MSE (RMSE) and maximum norm error (MNE) measure the difference between the input and reconstruction results by norm-based errors. (2) NLDR quality. Trustworthiness (Trust) and Continuity (Cont) (Moor et al., 2020), latent MSE (l-MSE), Minimum (Kmin) and Maximum (Kmax) local Lipschitz constant (Li et al., 2020) are used to evaluate the quality of the low-dimensional representation. (3) Generalization ability of the representation. Mean accuracy (Acc) of linear classification on the representation measures models' generalization ability to downstream tasks. Their exact definitions and purpose are given in Appendix A.1.

Conclusion. Table 1 compares the  $i$ -ML-Enc with the related methods on MNIST, more detailed analysis and results on Swiss roll, Half Spheres, USPS, FMNIST and COIL-20 are given in Appendix A.2. Compared with NLDR algorithms, the  $L$ -th layer of  $i$ -ML-Enc achieves second best NLDR results while preserving the all information of the dataset manifold in the  $L - 1$ -th layer. Compared with inverse models,  $i$ -ML-Enc nearly outperforms the other methods in both invertible and NLDR metrics, which indicates that good low-dimensional representation of manifolds can be learned by the equi-dimensional layers. The NLDR and its inverse process of  $i$ -ML-Enc are visualized in Fig. 4.

# 4.2 LATENT SPACE INTERPOLATION

Since the first  $L - 1$  layers of  $i$ -ML-Enc are nearly homeomorphism, we carry out linear interpolation experiments on the discrete data points in both the input space and the  $L - 1$ -th layer latent space to approximate the intrinsic continuous manifold, and verify the latent results by its inverse process. A good low-dimensional representation of the manifold shall not only preserve the local properties, but also is flatter and denser with lower curvature. Thus, we expect that the local linear interpolation results in the latent space should be more reliable than in the input space. The complexity of data manifolds increases from USPS(256), MNIST(256), MNIST(784), KMNIST(784) to FMNIST(784), which is analyzed in Appendix A.3.1.

![](images/0481fd5de963d2c3c5803d3f9338d0a1268bfd781208c387cab71854657df5e9.jpg)  
(a)

![](images/53239d05fb20e16c6938a1859e1bfd677c84985752c7e8cd8c6a38e944815ffe.jpg)  
Figure 5: (a) shows the proposed geodesics interpolation on a manifold; (b) reports the MSE loss of 1 to 10 nearest neighbors interpolation results on interpolation datasets.  
(b)

K-nearest neighbor interpolation. We first verify the reliability of the low-dimensional representation in a small local system by kNN interpolation. Given a sample  $\boldsymbol{x}_i$ , randomly select  $\boldsymbol{x}_j$  in  $\boldsymbol{x}_i$ 's k-nearest neighborhood in the latent space to form a sample pair  $(\boldsymbol{x}_i, \boldsymbol{x}_j)$ . Perform linear interpolation of the latent representation of the pair and get reconstruction results for evaluation as:  $\hat{\boldsymbol{x}}_{i,j}^{t} = \psi^{-1}(t\psi(\boldsymbol{x}_i) + (1 - t)\psi(\boldsymbol{x}_j))$ ,  $t \in [0,1]$ . The experiment is performed on i-ML-Enc with  $L = 6$  and  $K = 15$ , training with 8000 samples for USPS and MNIST(256), 20000 sapmles for MNIST(784), KMNIST, FMNIST.

Evaluation. (1) Calculate the MSE loss between reconstruction results of the latent interpolation  $\hat{\pmb{x}}_{i,j}^{t}$  and the input space result  $x_{i,j}^{t}$  which is the corresponding interpolation results in the local neighborhood of the input space with  $\pmb{x}_{i,j}^{t} = t\pmb{x}_{i} + (1 - t)\pmb{x}_{j}$ . Fig. 5 shows the results of  $k = 1,2,\ldots,10$ . (2) Visualize the typical results of the input space and the latent space for comparison, as shown in Fig. 6. More results and detailed analysis are given in Appendix A.3.2.

Geodesic interpolation. Based on 4.2.1, we further employ a more reasonable method to generate the sampling points between two distant samples pairs in the latent space. Given a sample pair  $(x_{i}, x_{j})$

![](images/f21c138bf150d204bb9c76691884e325e64076d5b4de2f85747244ccb895f5e9.jpg)  
(a) USPS (256)  
K≤5

![](images/0e9230742cc154f2da5fc9aa4dedf15eb097485b5d5d74d9aa2e0a89efd50c20.jpg)  
(b) MNIST(784)

![](images/e110a5b779c1ef6d3a3ba544740976228be638f29c49fa56f5708ced89d41d79.jpg)  
(c) KMNIST  
K≤5

![](images/360805fd6a7aaab8b2328ab4f0dc0afeefb1f6cc42e5dbb2b5e206a3b51893dc.jpg)  
Figure 6: The interpolation results of  $k$  nearest neighbors interpolation in the latent space. The latent results show more noise but less overlapping and pseudo-contour than the input results.  
(d) FMNIST

with  $k \geq 45$  from different clusters, we select the three intermediate sample pairs  $(x_{i}, x_{i_{1}}), (x_{i_{1}}, x_{i_{2}}), (x_{i_{2}}, x_{j})$  with  $k \leq 20$  along the geodesic path in latent space for piece-wise linear interpolation in both space. Visualization results are given in Appendix A.3.2.

Conclusion. Compared with results of the kNN and geodesic interpolation, we can conclude: (1) Because of the sparsity of the high-dimensional latent space, noises are inevitable on the latent results indicating the limitation of linear approximation. Empirically, the reliability of the latent interpolation decreases with the expansion of the local neighborhood on the same dataset. (2) We will get worse latent results in the following cases: on the similar manifolds, the sampling rate is lower or the input dimension is higher indicated by USPS(256), MNIST(256) and MNIST(784); with the same sampling rate and input dimension, the manifold is more complex indicated by MNIST(784), KMNIST to FMNIST. They indicate that the confidence of the tangent space estimated by local neighborhood decreases on more complex manifolds with sparse sampling. (3) The interpolation between two samples in latent space is more smooth than that the input space, validating the flatness and density of the lower-dimensional representation learned by  $i$ -ML-Enc. Overall, we infer that the unreliable approximation of the local tangent space by the local neighborhood is the basic reason of the manifold learning fails in the real-world case, because the geometry should be preserved in the first place. To come up with this common situation, it is necessary to import other prior assumption or knowledge when the sampling rate of the data manifold is quite low, e.g. the Euclidean space assumption, semantic information of down-steam tasks.

# 4.3 ABLATION STUDY

We perform ablation study on MNIST, USPS, KMNIST, FMNIST and COIL-20 to evaluate effects of the network structure and the loss terms in  $i$ -ML-Enc for manifold learning and invertible property. Based on ML-Enc, three proposed parts are added: the extra head (Ex), the orthogonal loss  $\mathcal{L}_{orth}$  (Orth), the zero padding loss  $\mathcal{L}_{pad}$  (Pad). Besides the previous 8 indicators, we introduce the rank of the output matrix of the layer  $L - 1$  as  $r(Z^{L - 1})$ , to measure the sparsity of the high-dimensional representation. We conclude that the Ex+Orth+Pad is the best combination to achieve invertible NLDR with  $s$ -sparse. The detailed analysis of experiment results and further discussion of the  $s$ -sparse are given in Appendix A.4.

# 5 CONCLUSION

A novel invertible DR process  $inv-ML$  and a neural network implementation  $inv-ML-Enc$  are proposed to tackle two problems of manifold-based DR in practical scenarios, i.e., the condition for information-lossless NLDR and the key issue of manifold learning. Firstly, the sparse coordinate transformation is learned to find a flatter and denser low-dimensional representation with preservation of geometry and topology of data manifolds. Secondly, we discuss the information loss with different target dimensions in linear compression. Experiment results of  $i-ML-Enc$  on seven datasets validate its invertibility. Further, the interpolation experiments reveal that finding a reliable tangent space by the local neighborhood on real-world datasets is the inherent defect of manifold based DR methods.

# REFERENCES

Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, and Jörn-Henrik Jacobsen. Invertible residual networks. In Proceedings of the 36th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, pp. 573-582, 2019.  
Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. arXiv preprint arXiv:1812.01718, 2018. URL http://arxiv.org/abs/1812.01718.  
Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: non-linear independent components estimation. In 3rd International Conference on Learning Representations (ICLR), 2015. URL http://arxiv.org/abs/1410.8516.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In 5th International Conference on Learning Representations (ICLR). OpenReview.net, 2017. URL https://openreview.net/forum?id=HkpbnH9lx.  
David L. Donoho. Compressed sensing. IEEE Trans. Inf. Theory, 52:1289-1306, 2006.  
Simon Hawe, Matthias Seibert, and Martin Kleinsteuber. Separable dictionary learning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 438-445. IEEE Computer Society, 2013.  
Matthias Hein and Jean-Yves Audibert. Intrinsic dimensionality estimation of submanifolds in r. pp. 289-296, 2005.  
Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504-507, 2006.  
Jonathan Hull. Database for handwritten text recognition research. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16:550 - 554, 05 1994. doi: 10.1109/34.291440.  
Jörn-Henrik Jacobsen, Arnold W. M. Smeulders, and Edouard Oyallon. i-revnet: Deep invertible networks. In Proceedings of 6th International Conference on Learning Representations (ICLR). OpenReview.net, 2018.  
William B. Johnson and JohnsonJoram Lindenstrauss. Extensions of lipschitz maps into a hilbert space. Contemporary Mathematics, 26:189-206, 01 1984.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of 3rd International Conference on Learning Representations (ICLR), 2015. URL http://arxiv.org/abs/1412.6980.  
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations (ICLR), 2014.  
Yann LeCun, Léon Bottou, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
Stan Z. Li, Zelin Zhang, and Lirong Wu. Markov-lipschitz deep learning. arXiv preprint arXiv:2006.08256, abs/2006.08256, 2020. URL https://arxiv.org/abs/2006.08256.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579-2605, 2008.  
James McQueen, Marina Meila, and Dominique Joncas. Nearly isometric embedding by relaxation. In Proceedings of the 29th Neural Information Processing Systems (NIPS), pp. 2631-2639, 2016.  
Jiaqiang Mei. Introduction to Manifold and Geometry. Beijing Science Press, 2013.  
Michael Moor, Max Horn, Bastian Rieck, and Karsten Borgwardt. Topological autoencoders. In Proceedings of the 37th International Conference on Machine Learning (ICML), Proceedings of Machine Learning Research, 2020.

John Nash. The imbedding problem for riemannian manifolds. Annals of Mathematics, 63:20-63, 1956.  
Sameer Nene, Shree Nayar, and H. Murase. Columbia object image library (coil-100). Technical report, 03 1996a. URL https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php.  
Sameer A. Nene, Shree K. Nayar, and Hiroshi Murase. Columbia object image library (coil-20). Technical report, Columbia University, 1996b. URL https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php.  
The-Gia Leo Nguyen, Lynton Ardizzone, and Ullrich Köthe. Training invertible neural networks as autoencoders. In Proceedings of 41st German Conference of Pattern Recognition (GCPR), volume 11824, pp. 442-455. Springer, 2019.  
Fabian Pedregosa, Gáel Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournaepau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12(85):2825-2830, 2011. URL http://jmlr.org/papers/v12/pedregosa11a.html.  
Sam T Roweis and Lawrence K Saul. Nonlinear dimensionality reduction by locally linear embedding. science, 290:2323-2326, 2000.  
Harish Seshadri and Kaushal Verma. The embedding theorems of whitney and nash. Resonance, pp. 815-826, 2016.  
Joshua B Tenenbaum, Vin De Silva, and John C Langford. A global geometric framework for nonlinear dimensionality reduction. science, 290(5500):2319-2323, 2000.  
Xian Wei, Martin Kleinsteuber, and Hao Shen. Invertible nonlinear dimensionality reduction via joint dictionary learning. In 12th Latent Variable Analysis and Signal Separation (LVA/ICA), volume 9237 of Lecture Notes in Computer Science, pp. 279-286. Springer, 2015.  
Xian Wei, Hao Shen, Yuanxiang Li, Xuan Tang, Fengxiang Wang, Martin Kleinsteuber, and Yi Lu Murphey. Reconstructible nonlinear dimensionality reduction via joint dictionary learning. IEEE Trans. Neural Networks Learn. Syst., 30(1):175-189, 2019.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. URL http://arxiv.org/abs/1708.07747.  
Zhenyue Zhang and Jing Wang. Mlle: Modified locally linear embedding using multiple weights. In Advances in Neural Information Processing systems, pp. 1593-1600, 2007.  
Zhenyue Zhang and Hongyuan Zha. Principal manifolds and nonlinear dimensionality reduction via tangent space alignment. SIAM journal on scientific computing, 26(1):313-338, 2004.
