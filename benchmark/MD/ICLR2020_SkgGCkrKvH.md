# DECENTRALIZED DEEP LEARNING WITH ARBITRARY COMMUNICATION COMPRESSION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Decentralized training of deep learning models is a key element for enabling data privacy and on-device learning over networks, as well as for efficient scaling to large compute clusters. As current approaches are limited by network bandwidth, we propose the use of communication compression in the decentralized training context. We show that CHOCO-SGD achieves linear speedup in the number of workers for arbitrary high compression ratios on general non-convex functions, and non-IID training data. We demonstrate the practical performance of the algorithm in two key scenarios: the training of deep learning models (i) over decentralized user devices, connected by a peer-to-peer network and (ii) in a datacenter.

# 1 INTRODUCTION

Distributed machine learning—i.e. the training of machine learning models using distributed optimization algorithms—has enabled many recent successful applications in research and industry. Such methods offer two of the key success factors: 1) computational scalability by leveraging the simultaneous computational power of many devices, and 2) data-locality, the ability to perform joint training while keeping each part of the training data local to each participating device. Recent theoretical results indicate that decentralized schemes can be as efficient as the centralized approaches, at least when considering convergence of training loss vs. iterations (Scaman et al., 2017; 2018; Lian et al., 2017; Tang et al., 2018; Koloskova et al., 2019; Assran et al., 2019).

Gradient compression techniques have been proposed for the standard distributed training case (Alistarh et al., 2017; Wen et al., 2017; Lin et al., 2018b; Wangni et al., 2018; Stich et al., 2018), to reduce the amount of data that has to be sent over each communication link in the network. For decentralized training of deep neural networks, Tang et al. (2018) introduce two algorithms (DCD, ECD) which allow for communication compression. However, both these algorithms are restrictive with respect to the used compression operators, only allowing for unbiased compressors and—more significantly—so far not supporting arbitrarily high compression ratios. We here study CHOCO-SGD—recently introduced for convex problems only (Koloskova et al., 2019)—which overcomes these constraints.

For the evaluation of our algorithm we in particular focus on the generalization performance (on the test-set) on standard machine learning benchmarks, hereby departing from previous work such as e.g. (Tang et al., 2018; Wang et al., 2019; Tang et al., 2019; Reisizadeh et al., 2019) that mostly considered training performance (on the train-set). We study two different scenarios: firstly, (i) training in a datacenter setting, where decentralized communication patterns allow better scalability than centralized approaches. For this setting we show that communication efficient CHOCO-SGD can improve time-to-accuracy on large tasks, such as e.g. ImageNet training. Secondly, (ii) training on a more challenging peer-to-peer setting, where the training data is distributed over the training devices (and not allowed to move), similar to the federated learning setting (McMahan et al., 2017). We are again able to show speed-ups for CHOCO-SGD over the decentralized baseline (Lian et al., 2017) with much less communication overhead. However, when investigating the scaling of these algorithms to larger number of nodes we observe that (all) decentralized schemes encounter difficulties and often do not reach the same (test and train) performance as centralized schemes. As these findings do point out some deficiencies of current decentralized training schemes (and are not particular to our scheme) we think that reporting these results is a helpful contribution to the community to spur further research on decentralized training schemes that scale to large number of peers.

Contributions. Our contributions can be summarized as:

- On the theory side, we are the first to show that CHOCO-SGD converges at rate  $\mathcal{O}\left(1 / \sqrt{nT} + n / (\rho^4\delta^2 T)\right)$  on non-convex smooth functions, where  $n$  denotes the number of nodes,  $T$  the number of iterations,  $\rho$  the spectral gap of the mixing matrix and  $\delta$  the compression ratio. The main term,  $\mathcal{O}\left(1 / \sqrt{nT}\right)$ , matches with the centralized baselines with exact communication and shows a linear speedup in the number of workers  $n$ . Both  $\rho$  and  $\delta$  only affect the asymptotically smaller second term.  
- On the practical side, we present a version of CHOCO-SGD with momentum and analyze its practical performance on two relevant scenarios:

$\circ$  in a datacenter setting for computational scalability of training deep learning models for resource efficiency and improved time-to-accuracy  
- for on-device training over a realistic peer-to-peer social network, where lowering the bandwidth requirements of joint training is especially impactful

- Lastly, systematically investigate performance of the decentralized schemes when scaling to larger number of nodes and we point out some (shared) difficulties encountered by current decentralized learning approaches.

# 2 RELATED WORK

For the training in communication restricted settings a variety of methods have been proposed. For instance, decentralized schemes (Lian et al., 2017; Nedic et al., 2018; Koloskova et al., 2019), gradient compression (Seide et al., 2014; Strom, 2015; Alistarh et al., 2017; Wen et al., 2017; Lin et al., 2018b; Wangni et al., 2018; Bernstein et al., 2018; Lin et al., 2018b; Alistarh et al., 2018; Stich et al., 2018; Karimireddy et al., 2019), asynchronous methods (Recht et al., 2011; Assran et al., 2019) or performing multiple local SGD steps before averaging (Zhang et al., 2016; McMahan et al., 2017; Lin et al., 2018a). This especially covers learning over decentralized data, as extensively studied in the federated Learning literature for the centralized algorithms (McMahan et al., 2016). In this paper we advocate for combining decentralized SGD schemes with gradient compression.

Decentralized SGD. We in particular focus on approaches based on gossip averaging (Kempe et al., 2003; Xiao & Boyd, 2004; Boyd et al., 2006) whose convergence rate typically depends on the spectral gap  $\rho \geq 0$  of the mixing matrix (Xiao & Boyd, 2004). Lian et al. (2017) combine SGD with gossip averaging and show convergence at the rate  $\mathcal{O}\left(1 / \sqrt{nT} + n / (\rho^2 T)\right)$ . The leading term in the rate,  $\mathcal{O}\left(1 / \sqrt{nT}\right)$ , is consistent with the convergence of the centralized mini-batch SGD (Dekel et al., 2012) and the spectral gap only affects the asymptotically smaller terms. Similar results have been observed very recently for related schemes (Scaman et al., 2017; 2018; Koloskova et al., 2019; Yu et al., 2019).

Quantization. Communication compression with quantization has been popularized in the deep learning community by the reported successes in (Seide et al., 2014; Strom, 2015). Theoretical guarantees were first established for schemes with unbiased compression (Alistarh et al., 2017; Wen et al., 2017; Wangni et al., 2018) but soon extended to biased compression (Bernstein et al., 2018) as well. Schemes with error correction work often best in practice and give the best theoretical guarantees (Lin et al., 2018b; Alistarh et al., 2018; Stich et al., 2018; Karimireddy et al., 2019). Recently, also proximal updates and variance reduction have been studied in combination with quantized updates (Mishchenko et al., 2019; Horváth et al., 2019).

Decentralized Optimization with Quantization. It has been observed that gossip averaging can diverge (or not converge to the correct solution) in the presence of quantization noise (Xiao et al., 2005; Carli et al., 2007; Nedic et al., 2008; Dimakis et al., 2010; Carli et al., 2010b; Yuan et al., 2012). Reisizadeh et al. (2018a) propose an algorithm that can still converge, though at a slower rate than the exact scheme. Another line of work proposed adaptive schemes (with increasing compression accuracy) that converge at the expense of higher communication cost (Carli et al., 2010a; Doan et al., 2018; Berahas et al., 2019). For deep learning applications, Tang et al. (2018) proposed the DCD and ECD algorithms that converge at the same rate as the centralized baseline though only for constant compression ratio. The CHOCO-SGD algorithm that we consider in this work can deal with arbitrary high compression, and has been introduced in (Koloskova et al., 2019) but only been analyzed for convex functions. For non-convex functions we show a rate of  $\mathcal{O}\left(1 / \sqrt{nT} + n / (\rho^4\delta^2 T)\right)$ . Where here  $\delta > 0$  measures the compression quality.

Algorithm 1 CHOCO-SGD (Koloskova et al., 2019)  
input: Initial values  $\mathbf{x}_i^{(0)}\in \mathbb{R}^d$  on each node  $i\in [n]$  , consensus stepsize  $\gamma$  , SGD stepsize  $\eta$  communication graph  $G = ([n],E)$  and mixing matrix  $W$  , initialize  $\hat{\mathbf{x}}_i^{(0)}\coloneqq \mathbf{0}\forall i\in [n]$  1: for t in 0...T-1 do {in parallel for all workers  $i\in [n]$  } 2:  $\mathbf{x}_i^{(t)}\coloneqq \mathbf{x}_i^{(t - \frac{1}{2})} + \gamma \sum_{j:\{i,j\} \in E}w_{ij}\bigl (\hat{\mathbf{x}}_j^{(t)} - \hat{\mathbf{x}}_i^{(t)}\bigr)$  modified gossip averaging 3:  $\mathbf{q}_i^{(t)}\coloneqq Q(\mathbf{x}_i^{(t)} - \hat{\mathbf{x}}_i^{(t)})$  compression 4: for neighbors  $j\colon \{i,j\} \in E$  (including  $\{i\} \in E$  ) do 5: Send  $\mathbf{q}_i^{(t)}$  and receive  $\mathbf{q}_j^{(t)}$  communication 6:  $\hat{\mathbf{x}}_j^{(t + 1)}\coloneqq \mathbf{q}_j^{(t)} + \hat{\mathbf{x}}_j^{(t)}$  local update 7: end for 8: Sample  $\xi_i^{(t)}$  , compute gradient  $\mathbf{g}_i^{(t)}\coloneqq \nabla F_i(\mathbf{x}_i^{(t)},\xi_i^{(t)})$  9:  $\mathbf{x}_i^{(t + \frac{1}{2})}\coloneqq \mathbf{x}_i^{(t)} - \eta \mathbf{g}_i^{(t)}$  stochastic gradient update 10: end for

# 3 CHOCO-SGD

In this section we formally introduce the decentralized optimization problem, compression operators, and the gossip-based stochastic optimization algorithm CHOCO-SGD from (Koloskova et al., 2019).

Distributed Setup. We consider optimization problems distributed across  $n$  nodes of the form

$$
f ^ {\star} := \min  _ {\mathbf {x} \in \mathbb {R} ^ {d}} \left[ f (\mathbf {x}) := \frac {1}{n} \sum_ {i = 1} ^ {n} f _ {i} (\mathbf {x}) \right], \quad f _ {i} := \mathbb {E} _ {\xi_ {i} \sim D _ {i}} F _ {i} (\mathbf {x}, \xi_ {i}), \quad \forall i \in [ n ], \tag {1}
$$

where  $D_{1},\ldots D_{n}$  are local distributions for sampling data which can be different on every node,  $F_{i}\colon \mathbb{R}^{d}\times \Omega \to \mathbb{R}$  are possibly non-convex (and non-identical) loss functions. This setting covers the important case of empirical risk minimization in distributed machine learning and deep learning applications.

Communication. Every device is only allowed to communicate with its local neighbours defined by the network topology, given as a weighted graph  $G = ([n], E)$ , with edges  $E$  representing the communication links along which messages (e.g. model updates) can be exchanged. We assign a positive weight  $w_{ij}$  to every edge ( $w_{ij} = 0$  for disconnected nodes  $\{i, j\} \notin E$ ).

Assumption 1 (Mixing matrix). We assume that  $W \in [0,1]^{n \times n}$ ,  $(W)_{ij} = w_{ij}$  is a symmetric  $(W = W^{\top})$  doubly stochastic  $(W\mathbf{1} = \mathbf{1},\mathbf{1}^{\top}W = \mathbf{1}^{\top})$  matrix with eigenvalues  $1 = |\lambda_1(W)| > |\lambda_2(W)| \geq \dots \geq |\lambda_n(W)|$  and spectral gap  $\rho := 1 - |\lambda_2(W)| \in (0,1]$ .

In our experiments we set the weights based on the local node degrees:  $w_{ij} = \max \{\deg(i), \deg(j)\}^{-1}$  for  $\{i,j\} \in E$ . This will not only guarantee  $\rho > 0$  but these weights can easily be computed in a local fashion on each node (Xiao & Boyd, 2004).

Compression. We aim to only transmit compressed (e.g. quantized or sparsified) messages. We formalized this through the notion of compression operators that was e.g. also used in (Tang et al., 2018; Stich et al., 2018).

Definition 3.1 (Compression operator).  $Q\colon \mathbb{R}^d\to \mathbb{R}^d$  is a compression operator if it satisfies

$$
\mathbb {E} _ {Q} \left\| Q (\mathbf {x}) - \mathbf {x} \right\| ^ {2} \leq (1 - \delta) \| \mathbf {x} \| ^ {2}, \quad \forall \mathbf {x} \in \mathbb {R} ^ {d}, \tag {2}
$$

for a parameter  $\delta > 0$ . Here  $\mathbb{E}_Q$  denotes the expectation over the internal randomness of operator  $Q$ .

In contrast to the quantization operators used in e.g. (Alistarh et al., 2017; Horváth et al., 2019), compression operators defined as in (2) are not required to be unbiased and therefore supports a larger class of compression operators. Some examples can be found in (Koloskova et al., 2019) and we further discuss specific compression schemes in Section 5.

Algorithm. CHOCO-SGD is summarized in Algorithm 1. Every worker  $i$  stores its own private variable  $\mathbf{x}_i\in \mathbb{R}^d$  that is updated by a stochastic gradient step in part (2) and a modified gossip averaging step on line 2. This step is a key element of the algorithm as it preserves the averages of

the iterates even in presence of quantization noise (the compression errors are not discarded, but aggregated in the local variables  $\mathbf{x}_i$ , see also (Koloskova et al., 2019)). The nodes communicate with their neighbors in part (1) and update the variables  $\hat{\mathbf{x}}_j \in \mathbb{R}^d$  for all their neighbors  $\{i,j\} \in E$  only using compressed updates. These  $\hat{\mathbf{x}}_i$  are available to all the neighbours of the node  $i$  and represent the 'publicly available' copies of the private  $\mathbf{x}_i$ , in general  $\mathbf{x}_i \neq \hat{\mathbf{x}}_i$ , due to the communication restrictions.

From an implementation aspect, it is worth highlighting that the communication part ① and the gradient computation part ② can both be executed in parallel because they are independent. Moreover, each node only needs to store 3 vectors at most, independent of the number of neighbors (this might not be obvious from the notation used here for additional clarity, for further details c.f. (Koloskova et al., 2019)). We further propose a momentum-version of CHOCO-SGD in Section D.

# 4 CONVERGENCE OF CHOCO-SGD ON SMOOTH NON-CONVEX PROBLEMS

As the first main contribution, we here extend the analysis of CHOCO-SGD to non-convex problems. For this we make the following technical assumptions:

Assumption 2. Each function  $f_{i} \colon \mathbb{R}^{d} \to \mathbb{R}$  for  $i \in [n]$  is  $L$ -smooth, that is

$$
\left\| \nabla f _ {i} (\mathbf {y}) - \nabla f _ {i} (\mathbf {x}) \right\| \leq L \left\| \mathbf {y} - \mathbf {x} \right\|, \quad \forall \mathbf {x}, \mathbf {y} \in \mathbb {R} ^ {d}, i \in [ n ],
$$

and the variance of the stochastic gradients is bounded on each worker:

$$
\mathbb {E} _ {\xi_ {i}} \left\| \nabla F _ {i} (\mathbf {x}, \xi_ {i}) - \nabla f _ {i} (\mathbf {x}) \right\| ^ {2} \leq \sigma_ {i} ^ {2}, \quad \mathbb {E} _ {\xi_ {i}} \left\| \nabla F _ {i} (\mathbf {x}, \xi_ {i}) \right\| ^ {2} \leq G ^ {2}, \quad \forall \mathbf {x} \in \mathbb {R} ^ {d}, i \in [ n ], \tag {3}
$$

where  $\mathbb{E}_{\xi_i}[\cdot ]$  denotes the expectation over  $\xi_{i}\sim \mathcal{D}_{i}$  . We also denote  $\overline{\sigma}^2\coloneqq \frac{1}{n}\sum_{i = 1}^{n}\sigma_i^2$  for convenience.

Theorem 4.1. Under Assumptions 1-2, with constant stepsize  $\eta = \sqrt{n} / \sqrt{T + 1}$  and the consensus stepsize from (Koloskova et al., 2019),  $\gamma := \frac{\rho^2\delta}{16\rho + \rho^2 + 4\beta^2 + 2\rho\beta^2 - 8\rho\delta}$  with  $\beta = \| I - W\|_2 \in [0,2]$ , and  $T \geq 64nL^2$ , the averaged iterates  $\overline{\mathbf{x}}^{(t)} := \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i^{(t)}$  of Algorithm 2 satisfy:

$$
\frac {1}{T + 1} \sum_ {t = 0} ^ {T} \left\| \nabla f (\overline {{\mathbf {x}}} ^ {(t)}) \right\| _ {2} ^ {2} \leq \frac {4 (f (\overline {{\mathbf {x}}} ^ {(0)}) - f ^ {\star}) + 4 \bar {\sigma} ^ {2} L}{\sqrt {n (T + 1)}} + \frac {2 4 G ^ {2} n L}{c ^ {2} (T + 1)},
$$

where  $c \coloneqq \frac{\rho^2\delta}{82}$  denotes the convergence rate of the underlying consensus averaging scheme of (Koloskova et al., 2019).

This result shows that CHOCO-SGD converges asymptotically as  $\mathcal{O}\bigl (1 / \sqrt{nT} +n / (\rho^4\delta^2 T)\bigr)$ . The first term shows a linear speed-up compared to SGD on a single node, while compression and graph topology affect only the higher order second term. For slightly more general statements than Theorem 4.1 (with improved constants) as well as for the proofs and convergence of the individual iterates  $\mathbf{x}_i$  we refer to Appendix A.

# 5 COMPARISON TO BASELINES FOR VARIOUS COMPRESSION SCHEMES

In this section we experimentally compare CHOCO-SGD to the relevant baselines for a selection of commonly used compression operators. For the experiments we further leverage momentum in all implemented algorithms. The newly developed momentum version of CHOCO-SGD is given as Algorithm 3 in Appendix D.

Setup. In order to match the setting in (Tang et al., 2018) for our first set of experiments, we use a ring topology with  $n = 8$  nodes and train the ResNet20 architecture (He et al., 2016) on the Cifar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012). We randomly split the training data between workers and shuffle it after every epoch, following standard procedure as e.g. in (Goyal et al., 2017). We implement DCD and ECD with momentum (Tang et al., 2018), CHOCOSGD with momentum (Algorithm 3) and standard (all-reduce) mini-batch SGD with momentum and without compression (Dekel et al., 2012). The momentum factor is set to 0.9 without dampening. For

Table 1: Top-1 test accuracy for decentralized DCD, ECD and CHOCO-SGD with different compression schemes. Reported top-1 test accuracies are averaged over three runs with fine-tuned hyper-parameters (learning rate, weight decay, consensus stepsize). The fine-tuned all-reduce baseline reaches accuracy 92.64, with 1.04 MB gradient transmission per iteration. ( $\star$  indicates that 2 out of 3 runs diverged).  

<table><tr><td>Algorithm</td><td colspan="4">Quantization (QSGD)</td><td colspan="3">Sparsification (random-%)</td></tr><tr><td>quantization level</td><td>16 bits</td><td>8 bits</td><td>4 bits</td><td>2 bits</td><td>50%</td><td>10%</td><td>1%</td></tr><tr><td>transmitted data/iteration</td><td>0.52 MB</td><td>0.26 MB</td><td>0.13 MB</td><td>0.065 MB</td><td>1.04 MB</td><td>0.21 MB</td><td>0.031 MB</td></tr><tr><td>DCD-PSGD</td><td>92.51 ± 0.05</td><td>92.36 ± 0.28</td><td>23.56 ± 2.97</td><td>diverges</td><td>92.05 ± 0.25</td><td>diverges</td><td>diverges</td></tr><tr><td>ECD-PSGD</td><td>92.02 ± 0.14</td><td>59.11 ± 1.57</td><td>diverges</td><td>diverges</td><td>diverges</td><td>diverges</td><td>diverges</td></tr><tr><td>CHOCO-SGD</td><td>92.34 ± 0.19</td><td>92.30 ± 0.08</td><td>91.92 ± 0.27</td><td>91.41 ± 0.11</td><td>92.54 ± 0.26</td><td>91.87 ± 0.21</td><td>91.32 ± 0.17</td></tr><tr><td>Algorithm</td><td colspan="3">Sparsification (top-%)</td><td>Sign+Norm</td><td></td><td></td><td></td></tr><tr><td>quantization level</td><td>50%</td><td>10%</td><td>1%</td><td>-</td><td></td><td></td><td></td></tr><tr><td>transmitted data/iteration</td><td>1.04 MB</td><td>0.21 MB</td><td>0.031 MB</td><td>0.032 MB</td><td></td><td></td><td></td></tr><tr><td>DCD-PSGD</td><td>92.40 ± 0.11</td><td>91.97 ± 0.14</td><td>89.79 ± 0.40</td><td>92.40 ± 0.14</td><td></td><td></td><td></td></tr><tr><td>ECD-PSGD</td><td>17.03 *</td><td>16.78 *</td><td>18.03 *</td><td>diverges</td><td></td><td></td><td></td></tr><tr><td>CHOCO-SGD</td><td>92.54 ± 0.26</td><td>92.29 ± 0.05</td><td>91.73 ± 0.11</td><td>92.46 ± 0.10</td><td></td><td></td><td></td></tr></table>

all algorithms we fine-tune the initial learning rate and gradually warm it up from a relative small value (0.1) (Goyal et al., 2017) for the first 5 epochs. The learning rate is decayed by 10 twice, at 150 and 225 epochs, and stop training at 300 epochs. For CHOCO-SGD the consensus learning rate  $\gamma$  is also tuned. The detailed hyper-parameter tuning procedure refers to Appendix F. Every compression scheme is applied to every layer of ResNet20 separately. We evaluate the top-1 test accuracy on every node separately over the whole dataset and report the average performance over all nodes.

Compression Schemes. We implement two unbiased compression schemes: (i)  $\mathrm{gsgd}_b$  quantization that randomly rounds the weights to  $b$ -bit representations (Alistarh et al., 2017), and (ii)  $\mathrm{random}_a$  sparsification, which preserves a randomly chosen  $a$  fraction of the weights and sets the other ones to zero (Wangni et al., 2018). Further two biased compression schemes: (iii)  $\mathrm{top}_a$ , which selects the  $a$  fraction of weights with the largest magnitude and sets the other ones to zero (Alistarh et al., 2018; Stich et al., 2018), and (iv) sign compression, which compresses each weight to its sign scaled by the norm of the full vector (Bernstein et al., 2018; Karimireddy et al., 2019). We refer to Appendix C for exact definitions of the schemes.

DCD and ECD have been analyzed only for unbiased quantization schemes, thus the combination with the two biased schemes is not supported by theory. In converse, CHOCO-SGD has been studied only for biased schemes according to Definition 2. However, both unbiased compression schemes can be scaled down in order to meet the specification (cf. discussions in (Stich et al., 2018; Koloskova et al., 2019)) and we adopt this for the experiments.

Results. The results are summarized in Table 1. For unbiased compression schemes, ECD and DCD only achieve good performance when the compression ratio is small, and sometimes even diverge when the compression ratio is high. This is consistent<sup>1</sup> with the theoretical and experimental results in (Tang et al., 2018). We further observe that the performance of DCD with the biased  $\mathrm{top}_a$  sparsification is much better than with the unbiased  $\mathrm{random}_a$  counterpart, though this operator is not yet supported by theory.

CHOCO-SGD can generalize reasonably well in all scenarios (at most  $1.65\%$  accuracy drop) for fixed training budget. The sign compression achieves state-of-the-art accuracy and requires approximately  $32 \times$  less bits per weight than the full precision baseline.

# 6 USE CASE I: ON-DEVICE PEER-TO-PEER LEARNING

We now shift our focus to challenging real-world scenarios which are intrinsically decentralized, i.e. each part of the training data remains local to each device, and thus centralized methods either fail or are inefficient to implement. Typical scenarios comprise e.g. sensor networks, or mobile devices or hospitals which jointly train a machine learning model. Common to these applications is that i) each

![](images/d4e800bcc07729a31bc4f706df460d5c48eba6ea6c8c65a9fd9f4d6c6e62016c.jpg)  
Fix budget of 300 epochs

![](images/8a18b4db5463d01c562c09305dd68847c850c56500c02522af0c91ac5563b4f1.jpg)  
Fixed budget of communication size (1000 MB)  
Figure 1: Scaling of CHOCO-SGD with sign compression to large number of devices on Cifar10 dataset. Left: best testing accuracy of the algorithms reached after 300 epochs. Right: best testing accuracy reached after communicating 1000 MB.

device has only access to locally stored or acquired data, ii) communication bandwidth is limited (either physically, or artificially for e.g. metered connections), iii) the global network topology is typically unknown to a single device, and iv) the number of connected devices is typically large. Additionally, this fully decentralized setting is also strongly motivated by privacy aspects, enabling to keep the training data private on each device at all times.

Modeling. To simulate this scenario, we permanently split the training data between the nodes, i.e. the data is never shuffled between workers during training, and every node has distinct part of the dataset. To the best of our knowledge, no prior works studied this scenario for decentralized deep learning. For the centralized approach, gathering methods such as all-reduce are not efficiently implementable in this setting, hence we compare to the centralized baseline where all nodes route their updates to a central coordinator for aggregation. For the comparison we consider CHOCOSGD with sign compression (this combination achieved the compromise between accuracy and compression level in Table 1)), decentralized SGD without compression, and centralized SGD without compression.

Scaling to Large Number of Nodes. To study the scaling properties of CHOCOSGD, we train on 4,16,36 and 64 number of nodes. We compare decentralized algorithms on two different topologies: ring as the worst possible topology, and on the torus with much larger spectral gap. Their parameters are listed in the table 2.

We train ResNet8 (He et al., 2016) (78K parameters), on CIFar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012). For the simplicity, we keep the learning rate constant and separately tune it for all methods. We tune consensus learning rate for CHOCO-SGD.

Table 2: Summary of communication topologies.  

<table><tr><td rowspan="2">Topology</td><td colspan="4">spectral gap ρ</td></tr><tr><td>max. node degree</td><td>n = 4</td><td>n = 16</td><td>n = 36</td></tr><tr><td>ring</td><td>2</td><td>0.67</td><td>0.05</td><td>0.01</td></tr><tr><td>torus</td><td>4</td><td>0.67</td><td>0.4</td><td>0.2</td></tr><tr><td>fully-connected</td><td>d</td><td>1</td><td>1</td><td>1</td></tr></table>

The results are summarized in Figure 1. First we compare the testing accuracy reached after 300 epochs (Fig. 1, Left). CentralizedSGD has a good performance for all the considered number of nodes. CHOCO-SGD slows down due to the influence of graph topology (Decentralized curve), which is consistent with the spectral gaps order (see Tab. 2), and also influenced by the communication compression (CHOCO curve), which slows down training uniformly for both topologies. We observed that the train performance is similar to the test on Fig. 1, therefore the performance degradation is explained by the slower convergence (Theorem 4.1) and is not a generalization issue. Increasing the number of epochs improves the performance of the decentralized schemes. However, even using 10 times more epochs, we were not able to perfectly close the gap between centralized and decentralized algorithms for both train and test performance.

In the real decentralized scenario, the interest is not to minimize the epochs number, but the amount of communication to reduce the cost of the user's mobile data. We therefore fix the number of transmitted bits to  $1000\mathrm{MB}$  and compare the best testing accuracy reached (Fig. 1, Right). CHOCOSGD performs the best while having slight degradation due to increasing number of nodes. It is beneficial to use torus topology when the number of nodes is large because it has good mixing properties, for small networks there is not much difference between these two topologies—the benefit of large spectral gap is canceled by the increased communication due larger node degree for torus

![](images/ce17e23ca3d78e30d287ab0149c5b712328d60407c4280a8d97991abaac6b059.jpg)  
Figure 2: Image classification: ResNet-20 on CIFAR-10 on social network topology.

![](images/022e1b4f6f89959717dcf0418fc92802d3a050219faeff9716f7099b8e4a4b08.jpg)

![](images/3d944424f9221ac6462a7a07cd9f1e0896bb319cef4446a1cf3f9516e457cfe0.jpg)

![](images/f480c5596b360cb1948ca35a2942f9dfe84dabd0f7439bbd876c95129d74c0b8.jpg)  
Figure 3: Language modeling: LSTM on WikiText-2 on social network topology.

![](images/24f834dcab305a4d1250165cdab60bc7e2d2a1057d830c060bc63993cf9eb137.jpg)

![](images/21e6e37343dfeb79157dadddfadf6641ac3d126c73821e46dcb7996aa075a6e6.jpg)

topology. Both Decentralized and Centralized SGD requires significantly larger number of bits to reach reasonable accuracy.

Experiments on a Real Social Network Graph. We simulate training models on user devices (e.g. mobile phones), connected by a real social network. We chosen Davis Southern women social network (Davis et al., 1941) with 32 nodes. We train ResNet20 (0.27 million parameters) model on the Cifar10 dataset (50K/10K training/test samples) (Krizhevsky, 2012) for image classification and a three-layer LSTM architecture (Hochreiter & Schmidhuber, 1997) (28.95 million parameters) for a language modeling task on WikiText-2 (600 training and 60 validation articles with a total of  $2'088'628$  and  $217'646$  tokens respectively) (Merit et al., 2016). We use exponentially decaying learning rate schedule. For more detailed experimental setup we refer to Appendix F.

The results are summarized in Figures 2-3 and in Table 3. For the image classification task, when comparing the training accuracy reached after the same number of epochs, we observe that the decentralized algorithm performs best, follows by the centralized and lastly the quantized decentralized. However, the test accuracy is highest for the centralized scheme. When comparing the test accuracy reached for the same transmitted data $^2$ , CHOCO-SGD significantly outperforms the exact decentralized scheme, with the centralized performing worst. We note a slight accuracy drop, i.e. after the same number of epochs (but much less transmitted data), CHOCO-SGD does not reach the same level of test accuracy than the baselines.

For the language modeling task, both decentralized schemes suffer a drop in the training loss when the evaluation reaching the epoch budget; while our CHOCO-SGD outperforms the centralized SGD in test perplexity. When considering perplexity for a fixed data volume (middle and right subfigure of Figure 3), CHOCO-SGD performs best, followed by the exact decentralized and centralized algorithms.

Table 3: Summary of performance when training with the same epoch budget (as centralized SGD).  

<table><tr><td rowspan="2">Algorithm</td><td colspan="3">ResNet-20 (Figure. 2)</td><td colspan="2">LSTM (Figure. 3)</td></tr><tr><td>max. connections/node</td><td>data/gradients</td><td>top-1 test acc.</td><td>data/gradients</td><td>test perplexity</td></tr><tr><td>Centralized SGD</td><td>32</td><td>1.04 MB</td><td>93.00</td><td>110.43 MB</td><td>89.39</td></tr><tr><td>Exact Decentralized SGD</td><td>14</td><td>1.04 MB</td><td>92.29</td><td>110.43 MB</td><td>96.33</td></tr><tr><td>sign-CHOCO-SGD</td><td>14</td><td>0.032 MB</td><td>91.90</td><td>3.45 MB</td><td>89.09</td></tr></table>

![](images/028f399faeb60ab2bd33d102339acd8d7dce85a693ffbff8ee385990a56a18ed.jpg)  
Figure 4: Large-scale training: Resnet-50 on ImageNet-1k in the datacenter setting.

![](images/2c38a5afb1cb686675fd0c58e772178a8a3a1922429e294ce007f18dc03b6608.jpg)

![](images/dbb67ea88b42f339cedfeed07868db321e5dcd93d169a2d30511e2cb03b63eb2.jpg)

# 7 USE CASE II: EFFICIENT LARGE-SCALE TRAINING IN A DATACENTER

Decentralized optimization methods offer a way to address scaling issues even for well connected devices, such as e.g. in datacenter with fast InfiniBand (100Gbps) or Ethernet (10Gbps) connections. Lian et al. (2017) describe scenarios when decentralized schemes can outperform centralized ones, and recently, Assran et al. (2019) presented impressive speedups for training on 256 GPUs, for the setting when all nodes can access all training data. The main differences of their algorithm to CHOCO-SGD are the asynchronous gossip updates, time-varying communication topology and most importantly exact communication, making their setup not directly comparable to ours. We note that these properties of asynchronous communication and changing topology for faster mixing are orthogonal to our contribution, and offer promise to be combined.

Setup. We train ImageNet-1k (1.28M/50K training/validation) (Deng et al., 2009) with Resnet-50 (He et al., 2016). We perform our experiments on 8 machines (n1-standard-32 from Google Cloud), where each of machines has 4 Tesla P100 GPUs. Within one machine communication is fast and we perform all-reduce with the full model. Between different machines we use decentralized communication with compressed communication (sign-CHOCO-SGD) in a ring topology. The mini-batch size on each GPU is 128, and we follow the general SGD training scheme in (Goyal et al., 2017) and directly use all their hyperparameters for CHOCO-SGD. Due to the limitation of the computational resource, we did not heavily tune the consensus stepsize for CHOCO-SGD<sup>3</sup>.

Results. We depict the training loss and top-1 test accuracy in terms of epochs and time in Figure 4. CHOCO-SGD benefits from its decentralized and parallel structure and takes less time than all-reduce to perform the same number of epochs, while having only a slight  $1.5\%$  accuracy loss. (All-reduce with full precision gradients achieved test accuracy of  $76.37\%$ , vs.  $75.15\%$  for CHOCO-SGD). In terms of time per epoch, our speedup does not match that of (Assran et al., 2019), as the used hardware is very different. Their scheme is orthogonal to our approach and could be integrated for better training efficiency. Nevertheless, we still demonstrate a time-wise  $20\%$  gain over the common all-reduce baseline, on our used commodity hardware cluster.

# 8 CONCLUSION

We propose the use of CHOCO-SGD (and its momentum version) for enabling decentralized deep learning training in bandwidth-constrained environments. We provide theoretical convergence guarantees for the non-convex setting and show that the algorithm enjoys the a linear speedup in the number of nodes. We empirically study the performance of the algorithm in a variety of settings on image classification (ImageNet-1k, Cifar10) and on a language modeling task (WikiText-2). Whilst previous work successfully demonstrated that decentralized methods can be a competitive alternative to centralized training schemes when no communication constraints are present (Lian et al., 2017; Assran et al., 2019), our main contribution is to enable training in strongly communication-restricted environments, and while respecting the challenging constraint of locality of the training data. We theoretically and practically demonstrate the performance of decentralized schemes for arbitrary high communication compression, and under data-locality, and thus significantly expand the reach of potential applications of fully decentralized deep learning.

# REFERENCES

David Aldous and James Allen Fill. Reversible markov chains and random walks on graphs, 2002. Unfinished monograph, recompiled 2014, available at http://www.stat.berkeley.edu/\sim sim\aldous/RWG/book.html.  
Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient SGD via gradient quantization and encoding. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), NIPS - Advances in Neural Information Processing Systems 30, pp. 1709-1720. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/6768-qsgd-communication-efficient-sgd-via-gradient-quantization-and-encoding.pdf.  
Dan Alistarh, Torsten Hoefer, Mikael Johansson, Nikola Konstantinov, Sarit Khirirat, and Cedric Renggli. The convergence of sparsified gradient methods. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), NeurIPS - Advances in Neural Information Processing Systems 31, pp. 5977-5987. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7837-the-convergence-of-sparsified-gradient-methods.pdf.  
Mahmoud Assran, Nicolas Loizou, Nicolas Ballas, and Michael Rabbat. Stochastic Gradient Push for Distributed Deep Learning. ICML 2019, 2019.  
Albert S. Berahas, Charikleia Iakovidou, and Ermin Wei. Nested distributed gradient methods with adaptive quantized communication. arXiv e-prints, pp. arXiv:1903.08149, 2019.  
Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signSGD: Compressed optimisation for non-convex problems. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 560–569, Stockholm mssan, Stockholm Sweden, 10–15 Jul 2018. PMLR. URL http://proceedings.mlr.press/v80/bernstein18a.html.  
Léon Bottou. Large-scale machine learning with stochastic gradient descent. In Yves Lechevallier and Gilbert Saporta (eds.), Proceedings of COMPSTAT'2010, pp. 177-186, Heidelberg, 2010. Physica-Verlag HD. ISBN 978-3-7908-2604-3.  
Stephen Boyd, Arpita Ghosh, Balaji Prabhakar, and Devavrat Shah. Randomized gossip algorithms. IEEE/ACM Trans. Netw., 14(SI):2508-2530, June 2006. ISSN 1063-6692. doi: 10.1109/TIT.2006.874516. URL https://doi.org/10.1109/TIT.2006.874516.  
R. Carli, F. Fagnani, P. Frasca, T. Taylor, and S. Zampieri. Average consensus on networks with transmission noise or quantization. In 2007 European Control Conference (ECC), pp. 1852-1857, July 2007. doi: 10.23919/ECC.2007.7068829.  
R. Carli, F. Bullo, and S. Zampieri. Quantized average consensus via dynamic coding/decoding schemes. International Journal of Robust and Nonlinear Control, 20:156-175, 2010a. ISSN 1049-8923.  
R. Carli, P. Frasca, F. Fagnani, and S. Zampieri. Gossip consensus algorithms via quantized communication. Automatica, 46:70-80, 2010b. ISSN 0005-1098.  
A. Davis, B. B. Gardner, and M. R. Gardner. Deep South. University of Chicago Press, Chicago, IL., May 1941.  
Ofer Dekel, Ran Gilad-Bachrach, Ohad Shamir, and Lin Xiao. Optimal distributed online prediction using mini-batches. J. Mach. Learn. Res., 13(1):165-202, January 2012. ISSN 1532-4435. URL http://dl.acm.org/citation.cfm?id=2503308.2188391.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.  
A. G. Dimakis, S. Kar, J. M. F. Moura, M. G. Rabbat, and A. Scaglione. Gossip algorithms for distributed signal processing. Proceedings of the IEEE, 98(11):1847-1864, Nov 2010. ISSN 0018-9219. doi: 10.1109/JPROC.2010.2052531.  
Thinh T. Doan, Siva Theja Maguluri, and Justin Romberg. Accelerating the Convergence Rates of Distributed Subgradient Methods with Adaptive Quantization. arXiv e-prints, art. arXiv:1810.13245, Oct 2018.  
Priya Goyal, Piotr Dollar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch SGD: Training ImageNet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9:1735-80, 12 1997. doi: 10.1162/neco.1997.9.8.1735.  
Samuel Horváth, Dmitry Kovalev, Konstantin Mishchenko, Peter Richtárik, and Sebastian Urban Stich. Stochastic distributed learning with gradient quantization and variance reduction. Technical Report, pp. arXiv:1904.05115, 2019. URL https://arxiv.org/abs/1904.05115.  
Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Urban Stich, and Martin Jaggi. Error feedback fixes SignSGD and other gradient compression schemes. Technical Report, pp. arXiv:1901.09847, 2019. URL https://arxiv.org/abs/1901.09847.  
David Kempe, Alin Dobra, and Johannes Gehrke. Gossip-based computation of aggregate information. In Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science, FOCS '03, pp. 482-, Washington, DC, USA, 2003. IEEE Computer Society. ISBN 0-7695-2040-5. URL http://dl.acm.org/citation.cfm?id=946243.946317.  
Anastasia Koloskova, Sebastian Urban Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. arXiv e-prints, art. arXiv:1902.00340, 2019. URL https://arxiv.org/abs/1902.00340.  
Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 05 2012.  
Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), NIPS - Advances in Neural Information Processing Systems 30, pp. 5330-5340. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/7117-can-decentralized-algorithms-outperform-centralized-algorithms-a-case-study-for-decent.pdf.  
Tao Lin, Sebastian Urban Stich, Kumar Kshitij Patel, and Martin Jaggi. Don't use large mini-batches, use local SGD. Technical Report, pp. arXiv:1808.07217, 2018a. URL https://arxiv.org/abs/1808.07217.  
Yujun Lin, Song Han, Huizi Mao, Yu Wang, and Bill Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In ICLR 2018 - International Conference on Learning Representations, 2018b. URL https://openreview.net/forum?id=SkhQHMW0W.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Areas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In AISTATS 2017 - Proceedings of the 20th International Conference on Artificial Intelligence and Statistics, pp. 1273-1282, 2017.  
H. Brendan McMahan, Eider Moore, Daniel Ramage, and Blaise Agüera y Arcas. Federated learning of deep networks using model averaging. CoRR, abs/1602.05629, 2016. URL http://arxiv.org/abs/1602.05629.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.  
Stephen Merity, Nitish Shirish Keskar, and Richard Socher. Regularizing and optimizing LSTM language models. arXiv preprint, pp. arXiv:1708.02182, 2017.  
Konstantin Mishchenko, Eduard Gorbunov, Martin Takáč, and Peter Richtárik. Distributed learning with compressed gradient differences. arXiv e-prints, pp. arXiv:1901.09269, 2019.  
A. Nedic, A. Olshevsky, and M. G. Rabbat. Network topology and communication-computation tradeoffs in decentralized optimization. Proceedings of the IEEE, 106(5):953-976, May 2018. ISSN 0018-9219. doi: 10.1109/JPROC.2018.2817461.  
Angelia Nedic, Alex Olshevsky, Asuman Ozdaglar, and John N. Tsitsiklis. Distributed subgradient methods and quantization effects. In Proceedings of the 47th IEEE Conference on Decision and Control, CDC 2008, pp. 4177-4184, 2008. ISBN 9781424431243. doi: 10.1109/CDC.2008.4738860.

Benjamin Recht, Christopher Re, Stephen Wright, and Feng Niu. Hogwild: A lock-free approach to parallelizing stochastic gradient descent. In J. Shawe-Taylor, R. S. Zemel, P. L. Bartlett, F. Pereira, and K. Q. Weinberger (eds.), NIPS - Advances in Neural Information Processing Systems 24, pp. 693-701. Curran Associates, Inc., 2011. URL http://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent.pdf.  
Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, and Ramtin Pedarsani. An exact quantized decentralized gradient descent algorithm. arXiv e-prints, pp. arXiv:1806.11536, 2018a.  
Amirhossein Reisizadeh, Aryan Mokhtari, S. Hamed Hassani, and Ramtin Pedarsani. Quantized decentralized consensus optimization. CoRR, abs/1806.11536, 2018b. URL http://arxiv.org/abs/1806.11536.  
Amirhossein Reisizadeh, Hossein Taheri, Aryan Mokhtari, Hamed Hassani, and Ramtin Pedarsani. Robust and communication-efficient collaborative learning. arXiv e-prints, pp. arXiv:1907.10595, 2019. URL https://arxiv.org/abs/1907.10595.  
Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. The Annals of Mathematical Statistics, 22(3):400-407, September 1951.  
Kevin Scaman, Francis Bach, Sébastien Bubeck, Yin Tat Lee, and Laurent Massoulie. Optimal algorithms for smooth and strongly convex distributed optimization in networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 3027-3036, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR. URL http://proceedings.mlr.press/v70/scaman17a.html.  
Kevin Scaman, Francis Bach, Sebastien Bubeck, Laurent Massoulie, and Yin Tat Lee. Optimal algorithms for non-smooth distributed optimization in networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), NeurIPS - Advances in Neural Information Processing Systems 31, pp. 2745-2754. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7539-optimal-algorithms-for-non-smooth-distributed-optimization-in-networks.pdf.  
Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu. 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs. In Haizhou Li, Helen M. Meng, Bin Ma, Engsiong Chng, and Lei Xie (eds.), INTERSPEECH, pp. 1058-1062. ISCA, 2014. URL http://dblp.uni-trier.de/db/conf/interspeech/interspeech2014.html#SeideFDLY14.  
Sebastian U Stich, Jean-Baptiste Cordonnier, and Martin Jaggi. Sparsified SGD with memory. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), NeurIPS - Advances in Neural Information Processing Systems 31, pp. 4452-4463. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7697-sparsified-sgd-with-memory.pdf.  
Nikko Strom. Scalable distributed dnn training using commoditygpu cloud computing. In INTERSPEECH, pp. 1488-1492. ISCA, 2015.  
Hanlin Tang, Shaoduo Gan, Ce Zhang, Tong Zhang, and Ji Liu. Communication compression for decentralized training. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), NeurIPS - Advances in Neural Information Processing Systems 31, pp. 7663-7673. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7992-communication-compression-for-decentralized-training.pdf.  
Hanlin Tang, Xiangru Lian, Shuang Qiu, Lei Yuan, Ce Zhang, Tong Zhang, and Ji Liu. Deepsqueeze: Decentralization meets error-compensated compression. arXiv e-prints, pp. arXiv:1907.07346, 2019. URL https://arxiv.org/abs/1907.07346.  
Jianyu Wang, Anit Kumar Sahu, Zhouyi Yang, Gauri Joshi, and Soummya Kar. An exact quantized decentralized gradient descent algorithm. arXiv e-prints, pp. arXiv:1905.09435, 2019. URL https://arxiv.org/abs/1905.09435.  
Jianqiao Wangni, Jialei Wang, Ji Liu, and Tong Zhang. Gradient sparsification for communication-efficient distributed optimization. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), NeurIPS - Advances in Neural Information Processing Systems 31, pp. 1306-1316. Curran Associates, Inc., 2018. URL http://papers.nips.cc/paper/7405-gradient-sparsification-for-communication-efficient-distributed-optimization.pdf.

Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), NIPS - Advances in Neural Information Processing Systems 30, pp. 1509-1519. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning.pdf.  
L. Xiao, S. Boyd, and S. Lall. A scheme for robust distributed sensor fusion based on average consensus. In IPSN 2005. Fourth International Symposium on Information Processing in Sensor Networks, 2005., pp. 63-70, April 2005. doi: 10.1109/IPSN.2005.1440896.  
Lin Xiao and Stephen Boyd. Fast linear iterations for distributed averaging. Systems & Control Letters, 53(1):65-78, 2004. ISSN 0167-6911. doi: https://doi.org/10.1016/j.sysconle.2004.02.022. URL http://www.sciencedirect.com/science/article/pii/S0167691104000398.  
Hao Yu, Rong Jin, and Sen Yang. On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization. ICML, May 2019.  
Deming Yuan, Shengyuan Xu, Huanyu Zhao, and Lina Rong. Distributed dual averaging method for multi-agent optimization with quantized communication. Systems & Control Letters, 61(11):1053 - 1061, 2012. ISSN 0167-6911. doi: https://doi.org/10.1016/j.sysconle.2012.06.004. URL http://www.sciencedirect.com/science/article/pii/S0167691112001193.  
Jian Zhang, Christopher De Sa, Ioannis Mitlagkas, and Christopher Ré. Parallel SGD: When does averaging help? arXiv, 2016.
