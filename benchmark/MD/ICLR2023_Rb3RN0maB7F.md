# QUAFL: FERERATED AVERAGING MADE ASYNCHRONOUS AND COMMUNICATION-EFFICIENT

Anonymous authors

Paper under double-blind review

# ABSTRACT

Federated Learning (FL) is an emerging paradigm to enable the large-scale distributed training of machine learning models, while still allowing individual nodes to maintain local data. In this work, we take steps towards addressing two of the main practical challenges when scaling federated optimization to large node counts: the need for tight synchronization between the central authority and individual computing nodes, and the large communication cost of transmissions between the central server and clients. Specifically, we present a new variant of the classic federated averaging (FedAvg) algorithm, which supports both asynchronous communication and communication compression. We provide a new analysis technique showing that, in spite of these system relaxations, our algorithm can provide similar convergence to FedAvg in some parameter regimes. On the experimental side, we show that our algorithm ensures fast convergence for standard federated tasks.

# 1 INTRODUCTION

Federated learning (FL) (Konečný et al., 2016; McMahan et al., 2017) is a paradigm for large-scale distributed learning, in which multiple clients, orchestrated by a central authority, cooperate to jointly optimize a machine learning model given their local data. The key promise is to enable joint training over distributed client data, often located on end devices which are computationally- and communication-limited, without the data leaving the client device.

The basic optimization algorithm underlying the learning process is known as federated averaging (FedAvg) (McMahan et al., 2017), and works roughly by having a central authority periodically communicate a shared model to all clients; then, the clients optimize this model locally based on their data, and communicate the resulting models to a central authority, which incorporates these models, often via some form of averaging, after which it initiates the next iteration. This algorithmic blueprint has been shown to be effective in practice (Li et al., 2020), and has also motivated a rich line of research analyzing its convergence properties (Stich, 2018; Haddadpour & Mahdavi, 2019), as well as proposing improved variants (Reddi et al., 2020; Karimireddy et al., 2020; Li & Richtárik, 2021).

Despite its popularity, scaling federated learning runs into a number of practical challenges (Kairouz et al., 2021). One natural bottleneck is synchronization between the server and the clients: as practical deployments may contain thousands of nodes, it is infeasible for the central server to orchestrate synchronous rounds among all participants. A simple mitigating approach is node sampling, e.g. (Smith et al., 2017; Bonawitz et al., 2019; He et al., 2018), by which the server only communicates with a subset of the nodes in a round; another, more general, approach is asynchronous communication, e.g. (Wu et al., 2020; Nguyen et al., 2022), by which the server and the nodes may work with inconsistent versions of the shared model, avoiding the need for lock-step synchronization.

An orthogonal scalability barrier for federated learning is the high communication cost of transmitting updates between the server and clients in every iteration (Kairouz et al., 2021): the burden of repeatedly transmitting parameter-heavy updates may overwhelm the bandwidth of communication-limited clients. Several approaches have been proposed to address this bottleneck, by allowing the server and clients to apply communication-compression to the transmitted updates, e.g. (Jin et al., 2020; Jhunjhunwala et al., 2021; Li & Richtárik, 2021; Wang et al., 2022).

It is reasonable to assume that both these bottlenecks would need to be mitigated in practice: for instance, communication-reduction may not be as effective if the server has to wait for each of the clients to complete their local steps on a version of the model; yet, synchrony is assumed by most references with compressed communication. Yet, removing synchrony completely may lead to divergence, especially given that local data is commonly assumed to be heterogenous. Thus, it is interesting to ask under which conditions asynchrony and communication compression, and heterogenous local data, can be jointly supported in federated learning.

Contributions. In this paper, we take a step towards answering this question, and propose an algorithm for Quantized Asynchronous Federated Learning called QuAFL, which is an extension of FedAvg, specifically adapted to support both asynchronous communication and communication compression. We provide a rigorous theoretical analysis showing that the algorithm still converges, and, in some parameter regimes, can asymptotically match the convergence of FedAvg despite compressed and asynchronous communication. Experimental results in a simulated environment with up to 300 nodes show that it can also lead to practical performance gains.

The general idea behind QuAFL is that we allow clients to perform their local steps independently of the round structure implemented by the server, and on a local, inconsistent version of the parameters, assuming a probabilistic scheduling model. Specifically, all clients receive a copy of the model when joining the computation, and start performing at most  $K \geq 1$  optimization steps on it based on their local data. Independently, in each "logical round," the server samples a set of  $s$  clients uniformly at random, and sends them a compressed copy of its current model. Whenever receiving the server's message, clients immediately respond with a compressed version of their current model, which may still be in the middle of the local optimization process, and therefore may not include recent server updates, nor the totality of the  $K$  local optimization steps. In fact, we even allow that, with some probability, some clients do not take any local steps at all. Clients carefully integrate the received server model into their next local iteration, while the server does the same with the models it receives.

The key missing piece regards quantization. Applying a standard quantizers (Alistarh et al., 2017; Karimireddy et al., 2019), runs into the issue that the quantization error may be too large, as it is proportional to the norm of the (updated) model at the client. Resolving this analytically would require either an unrealistic second-moment bound on the maximum gradient update, e.g. (Chen et al., 2021), or applying variance-reduction techniques (Gorbunov et al., 2021), which may be complex in practice. We circumvent this issue differently, by leveraging a recent lattice-based quantizer (Davies et al., 2021), which has the property that the quantization error only depends on the difference between the quantized model and a carefully-chosen "reference point." We instantiate this technique for the first time in the federated setting. To employ it successfully, we overcome non-trivial challenges in defining the right reference points for encoding and decoding, as well as the fact that this procedure has a non-zero probability of error.

Our analysis technique relies on a new potential argument, which shows that the discrepancy between the client and server models is always bounded. This bound has dual usage: both to control the "noise" at different steps due to model inconsistency, but also to ensure that the local models are consistent enough to allow correct encoding and decoding via lattice quantization. The technique is complex yet modular, and should allow further analysis of more complex algorithmic variants (Karimireddy et al., 2020; Reddi et al., 2020). Our algorithm has strong convergence bounds, given the degree of relaxation in the system, and the fact that we consider local data to be heterogenous. Specifically, in some parameter regimes, we can asymptotically match the best known bounds for the synchronous and full-communication FedAvg algorithm (Karimireddy et al., 2020), showing that asynchrony and quantization can be supported without significant impact of convergence.

We validate our algorithm experimentally in the rigorous LEAF (Caldas et al., 2018) environment, on a series of standard tasks. Specifically, in practice, QuAFL can compress updates by more than  $3 \times$  without significant loss of convergence, and can withstand a large constant fraction of "slow" clients submitting infrequent updates. Moreover, in a setting where client computation setting are heterogenous, QuAFL provides end-to-end speedup, since the server can progress without waiting for all clients to complete their local computation.

# 2 RELATED WORK

The federated averaging (FedAvg) algorithm was introduced by McMahan et al. (2017), and Stich (2018) was among the first to consider its convergence rate in the homogeneous data setting. Here, we investigate whether one can jointly eliminate two of the main scalability bottlenecks of this algorithm, the synchrony between the server and client iterations, as well as the necessity of full-precision communication, with heterogeneous data distributions. Due to space constraints, we focus on prior work which seeks to mitigate these two constraints in the context of FL.

There is significant research into communication-compression federated averaging (Philippenko & Dieuleveut, 2020; Reisizadeh et al., 2020; Jin et al., 2020; Haddadpour et al., 2021). However, virtually all of this work considers synchronous iterations. Reisizadeh et al. (2020) introduced FedPAQ, a variant of FedAvg which supports quantized communication via standard compressors, and provides strong convergence bounds, under the strong assumption of i.i.d. client data. Jin et al.

(2020) examines the viability of a variant of the signSGD quantizer (Seide et al., 2014; Karimireddy et al., 2019) in the context of FedAvg, providing convergence guarantees; however, the rate guarantees have a polynomial dependence in the model dimension  $d$ , rendering them less practically meaningful. Haddadpour et al. (2021) proposed FedCOM, a family of federated optimization algorithms with communication-compression and convergence rates; yet, we note that, in order to prove convergence in the challenging heterogeneous-data setting, this reference requires non-trivial technical assumptions on the quantized gradients (Haddadpour et al., 2021, Assumption 5). Chen et al. (2021) also considered update compression, but under convex losses, coupled with a rather strong second-moment bound assumption on the gradients. Finally, Jhunjunwala et al. (2021) examine adapting the degree of compression during the execution, proving convergence bounds for their scheme, under the non-standard i.i.d. data sampling assumption. We observe that each of these references requires at least one non-standard assumption for convergence guarantees for FedAvg under communication compression. By contrast, our analysis works for general (non-convex) losses, under a standard non-i.i.d. data distribution, without relying on second-moment bounds on the gradients.

A complementary approach to reducing communication cost in the federated setting has been to investigate optimizers with faster convergence, e.g. (Mishchenko et al., 2019; Karimireddy et al., 2020), or adaptive optimizers (Reddi et al., 2020; Tong et al., 2020). Moreover, recent work has shown that these approaches can be compatible with communication-compression (Gorbunov et al., 2021; Li & Richtárik, 2021; Wang et al., 2022). Specifically, for non-convex losses, MARINA Gorbunov et al. (2021) offers theoretical guarantees both in terms of convergence and bits transmitted. However, MARINA is structured in synchronous rounds; moreover, it periodically (with some probability) has clients compute full gradients and transmit uncompressed model updates, and requires fairly complex synchronization and variance-reduction to compensate for the extra noise due to quantization. Recent work by Tyurin & Richtárik (2022) proposed a family of theoretical methods called DASHA, which combines the general structure of MARINA with Momentum Variance Reduction (MVR) methods (Cutkosky & Orabona, 2019), while relaxing the coupling between the server and the nodes and allowing fully-compressed updates.

In contrast to these works, we focus on obtaining a practical algorithm with good convergence bounds: we always transmit compressed, low-precision messages, and consider a notion of asynchronous communication which allows the server and nodes to make progress independently, in non-blocking fashion. We focus on the classic, practical FedAvg algorithm, although our general algorithmic and analytic approach should generalize to more complex notions of local optimization.

Our approach extends ideas from the analysis of decentralized variants of SGD (Lian et al., 2017; Tang et al., 2018; Nadiradze et al., 2021; Koloskova et al., 2019; Lu & De Sa, 2020), bringing them into the context of federated optimization. Significant differences exist: notably, we introduce a novel potential argument, adapted to FL, and cannot rely on stronger assumptions available in the decentralized setting, e.g. a gradient second-moment bound (Lu & De Sa, 2020).

# 3 THE ALGORITHM

# 3.1 SYSTEM OVERVIEW

System Model. We assume a distributed system with one coordinator and  $n$  client nodes, jointly minimizing a  $d$ -dimensional, differentiable function  $f: \mathbb{R}^d \to \mathbb{R}$ . We consider the empirical risk minimization (ERM) setting, in which data samples are located at the  $n$  nodes. Each agent  $i$  has a local function  $f_i$  associated to its own local fraction of the data, i.e.  $\forall x \in \mathbb{R}^d: f(x) = \sum_{i=1}^{n} f_i(x) / n$ . The goal is to converge on a model  $x^*$  which minimizes the empirical loss. Clients run a distributed variant of SGD, coordinated by the central node. We will assume that each client  $i$  is able to obtain unbiased stochastic gradients  $\widetilde{g}_i$  of its own local function  $f_i$ , i.e.  $\mathbb{E}[\widetilde{g}_i(x)] = \nabla f_i(x)$ . These stochastic gradients can be computed by each agent by sampling i.i.d. from its own local distribution. Our analysis will consider the case where each client distribution is distinct, but there is a bound on the maximum gradient discrepancy.

We model client asynchrony as follows: between two consecutive interactions with the server, each client should perform a number of gradient steps on its local model. We treat the number of local steps at client  $i$  as a random variable  $\mathcal{H}_i$ , taking values in  $\{0, 1, 2, \dots, K\}$ , where  $K$  is a bound on how many steps a client can take in isolation. We emphasize the fact that  $\mathcal{H}_i$  can take the value 0, meaning that the client may take no steps since last contacted. Our only assumption regarding asynchrony is that the expected value of  $\mathcal{H}$ , denoted by  $H$ , exists and is  $> 0$ . That is, we assume that, on average, each client makes non-trivial progress, and clients progress at similar rates. We note that step distributions  $\mathcal{H}_i$  at clients can be completely different, as long as their expectations match.

# 3.2 ALGORITHM DESCRIPTION

Overview. Our algorithm starts from the standard pattern used by federated averaging (FedAvg): computation and communication are organized in logical "rounds," where in each round the server transmits its current version of the model to either all, or a subset of clients. The clients should then take some number of local optimization steps on the received model, which is at most  $K \geq 1$ , and transmit the result to the server, which integrates these updates. Our algorithm will relax this pattern in two orthogonal ways, allowing for both quantized and asynchronous communication.

Quantized Communication. The first relaxation is to only allow for compressed communication of the server model and of the client updates, via quantization. For this, we employ a carefully-parametrized version of the lattice-based quantization technique of Davies et al. (2021), whose analytical properties we describe in the analysis section. For practical purposes, this quantization technique presents an encoding function  $Enc(A)$ , which encodes an arbitrary input  $A$  to its quantized representation. (We always communicate vectors via their quantized representations.) To "read" an encoded message  $Enc(A)$ , a node must call the symmetric  $Dec(B, Enc(A))$  function, which allows for the "decoding" of the input  $Enc(A)$  with respect to a reference point  $B$ , returning a quantized output  $Q(A)$ . We formally specify the properties of the compression process in Section 4.

Algorithm 1 Pseudocode for QuAFL Algorithm.  
$\%$  Initial models  $X_0 = X^1 = X^2 = \ldots = X^n = 0^d$  , number of local steps  $K$ $\%$  Encoding (Enc(A)) and decoding (Dec(B, Enc(A))) functions, with common parametrization.   
 $\%$  At the Server:   
1: for  $t = 0$  to  $T - 1$  do   
2: Server chooses  $s$  clients uniformly at random, let  $S$  be the resulting set.   
3: for all clients  $i\in S$  do   
4: Server sends Enc  $(X_{t})$  to the client i.   
5: Server receives Enc  $(Y^{i})$  from client i.   
6:  $Q(Y^{i})\gets Dec(X_{t},Enc(Y^{i}))$  % Decodes client messages relative to  $X_{t}$    
7: end for   
8:  $X_{t + 1} = \frac{1}{s + 1} X_t + \frac{1}{s + 1}\sum_{i\in S}Q(Y^i)$    
9: end for   
 $\%$  At Client i:   
 $\%$  Upon (asynchronous) contact from the server run INTERACTWITHSERVER   
 $\%$  Local variables:   
 $\% X^i$  stores the base client model, following the last server interaction. Initially  $0^{d}$ $\% \widetilde{h}_i$  accumulates local gradient steps since last server interaction, initially  $0^{d}$    
1: function INTERACTWITHSERVER   
2:  $MSG_{i}\gets Enc(X^{i} - \eta h_{i})$  % Client i compresses its local progress since last contacted.   
3: Client sends  $MSG_{i}$  to the server.   
4: Client receives Enc  $(X_{t})$  from the server, where t is the current server time.   
5:  $Q(X_{t})\gets Dec(X^{i},Enc(X_{t}))$  % Client decodes the message using its old model as reference point.   
6:  $\%$  The client then updates its local model   
7:  $X^{i} = \frac{1}{s + 1} Q(X_{t}) + \frac{s}{s + 1}(X^{i} - \eta \widetilde{h}_{i})$    
8:  $\%$  Finally, it performs  $K$  new local steps on the updated  $X^i$  , unless interrupted again.   
9: LOCALUPDATES  $(X^i,K)$    
10: WAIT()   
11: end function   
1: function LOCALUPDATES  $(X^i,K)$    
2:  $\widetilde{h}_i = 0\%$  local gradient accumulator   
3: for  $q = 0$  to  $K - 1$  do   
4:  $\widetilde{h}_i^q = \widetilde{g}_i(X^i -\eta \sum_{\ell = 0}^{q - 1}\widetilde{h}_i^\ell)\%$  compute the qth local gradient   
5:  $\widetilde{h}_i = \widetilde{h}_i + \widetilde{h}_i^q\%$  add it to the accumulator   
6: end for   
7: end function

Asynchronous Communication. A key practical limitation of the FedAvg pattern is the fact that the server and its clients have to communicate in synchronous, lock-step fashion: thus, the server must wait for the results of computation at a round before it can move to the next round. In particular, this means that the server has to wait for the slowest client to complete its local steps before it can proceed. QuAFL relaxes this requirement by essentially allowing any contacted node  $i$  to immediately return

(a quantized version of) its current version of the model to the server upon being contacted, even though the client might still not have completed all its  $K$  local optimization steps for the round. More precisely, the client always records its "base" model at the end of the last interaction with the server into parameter  $X^i$ , and sums up its gradient updates since the last interaction into the buffer  $\widetilde{h}_i$ . Upon being contacted, the client simply sends its current progress  $Y^i = X^i - \eta \widetilde{h}_i$  to the server (excluding the local step for which computation was not finished due to interruption from the server), where  $\eta$  is the learning rate, in quantized form. It is possible that this progress is zero. The client then decodes the quantized server model  $X_t$ , using its old local model  $X^i$  as the decoding key. Finally, the client updates  $X^i$  to include the server's new information via averaging. It is then ready to restart its local update loop, upon this new model.

It is important to notice that the server interaction occurs asynchronously, and that it might occur either while the client is still performing local steps, or after the client has completed its  $K$  local steps, and is idle, waiting for server contact. In the former case, upon being contacted, immediately calls the server interaction function, without performing additional steps. (In particular, we allow the number of completed local steps to be 0.) Globally, the server contacts  $s$  random agents in each logical round, sends them a quantized version of the global model  $X_{t}$ , then receives quantized versions of their progress, and then incorporates this into the global model which will be sent at the next round.

Discussion. As we will see in the experiments, the practical advantage of QuAFL is that the server does not have to wait for each of the contacted clients to complete their local optimization on the global model  $X_{t}$ . In addition, an important departure from FedAvg is the averaging between the server and client models. Our formulation is important for fast convergence: as we show in Figure 3, other forms, such as just adopting the client average, lead to worse convergence. Our assumption on the expected local step count  $H > 0$  can be restated as saying that clients do make some local progress on average between server interactions, although that is not always required. However, we will prove that, when  $H = \Theta (K)$ , asymptotic convergence is the same as FedAvg.

# 4 CONVERGENCE ANALYSIS

# 4.1 ANALYTICAL ASSUMPTIONS

We begin by stating the assumptions we make in the theoretical analysis of our algorithm. Specifically, we assume the following for the global loss function  $f$ , the individual client losses  $f_{i}$ , and the stochastic gradients  $\widetilde{g}_{i}$ :

1. Uniform Lower Bound: There exists  $f_{*} \in \mathbb{R}$  such that  $f(x) \geq f_{*}$  for all  $x \in \mathbb{R}^d$ .  
2. Smooth Gradients: For any client  $i$ , the gradient  $\nabla f_{i}(x)$  is  $L$ -Lipschitz continuous for some  $L > 0$ , i.e. for all  $x, y \in \mathbb{R}^{d}$ :

$$
\| \nabla f _ {i} (x) - \nabla f _ {i} (y) \| \leq L \| x - y \|. \tag {1}
$$

3. Bounded Variance: For any client  $i$ , the variance of the stochastic gradients is bounded by some  $\sigma^2 > 0$ , i.e. for all  $x \in \mathbb{R}^d$ :

$$
\mathbb {E} \left\| \widetilde {g} _ {i} (x) - \nabla f _ {i} (x) \right\| ^ {2} \leq \sigma^ {2}. \tag {2}
$$

4. Bounded Gradient Dissimilarity: There exist constants  $G^2 \geq 0$  and  $B^2 \geq 1$ , such that for all  $x \in \mathbb{R}^d$ :

$$
\sum_ {i = 1} ^ {n} \frac {\left\| \nabla f _ {i} (x) \right\| ^ {2}}{n} \leq G ^ {2} + B ^ {2} \left\| \nabla f (x) \right\| ^ {2}. \tag {3}
$$

All these assumptions are standard in the context of non-convex federated learning, e.g. (Karimireddy et al., 2020; Jin et al., 2020; Gorbunov et al., 2021). Specifically, the first three assumptions are fairly universal in distributed stochastic optimization over non-convex losses, whereas the fourth is a standard way of encoding the fact that there must be a bound on the amount of divergence between the local distributions at the nodes in order to allow for joint optimization.

Quantization Procedure. For completeness, we restate the formal guarantees of the quantization function (Davies et al., 2021) (Lemma 23):

Lemma 4.1. (Lattice Quantization) Fix parameters  $R$  and  $\gamma >0$ . There exists a quantization procedure defined by an encoding function  $Enc_{R,\gamma}:\mathbb{R}^d\to \{0,1\}^*$  and a decoding function  $Dec_{R,\gamma} = \mathbb{R}^d\times \{0,1\}^*\to \mathbb{R}^d$  such that, for any vector  $x\in \mathbb{R}^d$  which we are trying to quantize, and

any vector  $y$  which is used by decoding, which we call the decoding key, if  $\| x - y\| \leq R^{R^d}\gamma$  then with probability at least  $1 - \log \log (\frac{\|x - y\|}{\gamma})O(R^{-d})$ , the function  $Q_{R,\gamma}(x) = Dec_{R,\gamma}(y,Enc_{R,\gamma}(x))$  has the following properties:

1. (Unbiased decoding)  $\mathbb{E}[Q_{R,\gamma}(x)] = \mathbb{E}[Dec_{R,\gamma}(y, Enc_{R,\gamma}(x))] = x$ ;  
2. (Error bound)  $\| Q_{R,\gamma}(x) - x\| \leq (R^2 +7)\gamma$  
3. (Communication bound)  $O\left(d\log \left(\frac{R}{\gamma}\| x - y\|\right)\right)$  bits are needed to send  $Enc_{R,\gamma}(x)$ .

Thus, our aim is to show that in our algorithm, local models of the clients stay close to the local model of the server, so that we can successfully apply the above lemma to our setting.

# 4.2 MAIN RESULTS

Let  $\mu_t = (X_t + \sum_{i=1}^n X^i) / (n+1)$  be the mean over all the node models in the system at a given  $t$ . Our main result shows the following:

Theorem 4.2. Assume the total number of steps  $T \geq \Omega(n^3)$ , the learning rate  $\eta = \frac{n + 1}{sH\sqrt{T}}$ , and quantization parameters  $R = 2 + T^{\frac{3}{d}}$  and  $\gamma^2 = \frac{\eta^2}{(R^2 + 7)^2}\left(\sigma^2 + 2KG^2 + \frac{f(\mu_0) - f_*}{L}\right)$ . Let  $H > 0$  be the expected number of local steps already performed by a client when interacting with the server. Then, with probability at least  $1 - O\left(\frac{1}{T}\right)$  we have that Algorithm 1 converges at the following rate

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2} \leq \frac {5 (f (\mu_ {0}) - f _ {*})}{\sqrt {T}} + \frac {8 K L (\sigma^ {2} + 2 K G ^ {2})}{H ^ {2} \sqrt {T}} + O \left(\frac {n ^ {3} K L ^ {2} (\sigma^ {2} + 2 K G ^ {2})}{s H ^ {3} T}\right)
$$

and uses  $O\left(sT(d\log n + \log T)\right)$  expected communication bits in total.

The result provides a trade-off between the convergence speed of the algorithm, the variance of the local distributions (given by  $\sigma$  and  $G$ ), the sampling set size  $s$ , and the average number of local steps  $H$  performed by a node when contacted by the server. Intuitively, the first additive term appears to be asymptotically-optimal, as the server relies on progress from  $s$  clients, each having taken  $H$  local gradient steps on average, and the objective is general non-convex. The second term essentially contains a "total upper bound" on variance in its numerator, divided by the total number of expected gradient steps in the denominator. The third term contains similar "nuisance terms" to the second, with the addition of the  $n^3$  factor, and also bounds the extra variance. Crucially, this larger term is divided by  $T$ , as opposed to  $\sqrt{T}$ ; since  $T$  is our asymptotic parameter, it is common to assume that this extra term becomes negligible as  $T$  is large, e.g. (Lu & De Sa, 2020).

Under this convention, the first two terms essentially recover the convergence bound provided by Karimireddy et al. (2020) for FedAvg if we assume that  $H = \Theta(K)$  and that  $s = \Theta(n)$ ; this reference also provided arguments showing that the dependency in  $\sigma$  and  $G$  should be inherent in the heterogenous data setting that we consider (Karimireddy et al., 2020, Section 3.2).

In practice, it should be reasonable to assume that  $H = \Theta(K)$ , that is, that on average each client  $i$  will have completed its local steps on the old version of the model  $X^i$  when being contacted: otherwise, the sampling frequency of the server is too high, and prevents clients from making progress on their local optimization, and the server should simply decrease it.

Convergence at the Server. Finally, we show that not only convergence at the server, as opposed to the convergence of the mean of the local models as in Theorem 4.2. We get that:

Corollary 4.3. Assume the total number of steps  $T \geq \Omega(n^4)$ , the learning rate  $\eta = \frac{n + 1}{sH\sqrt{T}}$ , and quantization parameters  $R = 2 + T^{\frac{3}{d}}$  and  $\gamma^2 = \frac{\eta^2}{(R^2 + 7)^2}\left(\sigma^2 + 2KG^2 + \frac{f(\mu_0) - f_*}{L}\right)$ . Let  $H > 0$  be the expected number of local steps already performed by a client when interacting with the server. Then, with probability at least  $1 - O\left(\frac{1}{T}\right)$  we have that Algorithm 1 converges at the following rate

$$
\frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (X _ {t}) \| ^ {2} \leq \frac {5 (f (\mu_ {0}) - f _ {*})}{\sqrt {T}} + \frac {8 K L (\sigma^ {2} + 2 K G ^ {2})}{H ^ {2} \sqrt {T}} + O \left(\frac {n ^ {4} K L ^ {2} (\sigma^ {2} + 2 K G ^ {2})}{s H ^ {2} T}\right).
$$

This corollary yields a very similar bound to our main result, except for the larger dependency between  $T$  and  $n$ , which is intuitively required due to the additional time required for the server to converge to a similar bound to the mean  $\mu_t$ . The third term may be significant for large numbers of nodes  $n$ ;

however, since it is divided by  $T$  (as opposed to  $\sqrt{T}$ ) it can be seen as negligible for moderate  $n$  and large  $T$ . The fact that QuAFL can match some of the best known rates for FedAvg under some parameter settings may seem surprising, since our algorithm is asynchronous (in particular, nodes take steps on local, delayed versions of the server model) and also supports communication-compression.

# 4.3 OVERVIEW OF THE ANALYSIS

The complete analysis is fairly complex, and is provided in full in the Appendix. Due to space constraints, we only provide an overview of the proofs, outlining the main intermediate results. The first step in the proof is bounding the deviation between the local models and their mean. For this, we introduce the potential function  $\Phi_t = \| X_t - \mu_t\|^2 + \sum_{i=1}^n \| X^i - \mu_t\|^2$ , and we use a load-balancing approach to show that this potential has the following supermartingale-type property:

Lemma 4.4. For any time step  $t$  we have:

$$
\mathbb {E} \left[ \Phi_ {t + 1} \right] \leq \left(1 - \frac {1}{4 n}\right) \mathbb {E} \left[ \Phi_ {t} \right] + 8 s \eta^ {2} \sum_ {i = 1} ^ {n} \mathbb {E} \| \widetilde {h} _ {i} \| ^ {2} + 1 6 n (R ^ {2} + 7) ^ {2} \gamma^ {2}.
$$

The intuition behind this result is that potential  $\Phi_t$  will stay well-concentrated around its mean, except for influences from the variance due to local steps (second term) or quantization (third term). With this in place, the next lemma allows us to track the evolution of the average of the local models, with respect to local step and quantization variance:

Lemma 4.5. For any step  $t$

$$
\mathbb {E} \| \mu_ {t + 1} - \mu_ {t} \| ^ {2} \leq \frac {2 s ^ {2} \eta^ {2}}{n (n + 1) ^ {2}} \sum_ {i} \mathbb {E} \left\| \widetilde {h} _ {i} \right\| ^ {2} + \frac {2}{(n + 1) ^ {2}} (R ^ {2} + 7) ^ {2} \gamma^ {2}.
$$

In both cases, the upper bound depends on the second moment of the nodes' local progress  $\sum_{i}\mathbb{E}\left\| \widetilde{h}_i\right\|^2$  (This is due to the fact that the server contacts  $s$  clients, which are chosen uniformly at random.) Then, our main technical lemma uses properties (1), (2) and (3), to concentrate  $\sum_{i}\mathbb{E}\left\| \widetilde{h}_i(X_t^i)\right\|^2$  around the true gradient  $\mathbb{E}\| \nabla f(\mu_t)\|^2$ , where the expectation is taken over the algorithm's randomness.

Lemma 4.6. For any step  $t$ , we have that

$$
\sum_ {i = 1} ^ {n} \mathbb {E} \| \widetilde {h} _ {i} \| ^ {2} \leq 2 n K (\sigma^ {2} + 2 K G ^ {2}) + 8 L ^ {2} K ^ {2} \mathbb {E} [ \Phi_ {t} ] + 4 n K ^ {2} B ^ {2} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}.
$$

We can then combine Lemmas 4.4 and 4.6 to get an upper bound on the potential with respect to  $\mathbb{E}\| \nabla f(\mu_t)\|^2$ . Summing over steps, we obtain the following:

Lemma 4.7.

$$
\sum_ {t = 0} ^ {T} \mathbb {E} \left[ \Phi_ {t} \right] \leq 8 0 T n ^ {2} (R ^ {2} + 7) ^ {2} \gamma^ {2} + 8 0 T n ^ {2} s K \eta^ {2} (\sigma^ {2} + 2 K G ^ {2}) + 1 6 0 B ^ {2} n ^ {2} s K ^ {2} \eta^ {2} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}.
$$

Next, using the  $L$ -smoothness of the function  $f$ , implied by (1), we can show that

$$
\mathbb {E} [ f (\mu_ {t + 1}) ] \leq \mathbb {E} [ f (\mu_ {t}) ] + \mathbb {E} \langle \nabla f (\mu_ {t}), \mu_ {t + 1} - \mu_ {t} \rangle + \frac {L}{2} \mathbb {E} \| \mu_ {t + 1} - \mu_ {t} \| ^ {2}. \tag {4}
$$

Final argument. Using the above inequality, and given that  $\mathbb{E}[\mu_{t + 1} - \mu_t] = -\frac{\eta}{n + 1}\sum_{i\in S}\widetilde{h}_i(X_t^i)$ , we observe that the sum  $\sum_{i = 1}^{n}\mathbb{E}\langle \nabla f(\mu_t),\mu_{t + 1} - \mu_t\rangle$  can be concentrated around  $\mathbb{E}\| \nabla f(\mu_t)\|^2$ , in similar fashion as in Lemma 4.6. Together with Lemma 4.5, this results in the following bound:

$$
\begin{array}{l} \mathbb {E} [ f (\mu_ {t + 1}) ] - \mathbb {E} [ f (\mu_ {t}) ] \leq \frac {5 \eta s K L ^ {2} \mathbb {E} [ \Phi_ {t} ]}{n (n + 1)} + \left(\frac {4 s L ^ {2} \eta^ {3} K ^ {3}}{n + 1} + \frac {2 s ^ {2} K \eta^ {2} L}{(n + 1) ^ {2}}\right) (\sigma^ {2} + 2 K G ^ {2}) \\ + \frac {(R ^ {2} + 7) ^ {2} \gamma^ {2} L}{(n + 1) ^ {2}} + \big (\frac {- 3 \eta s H}{4 (n + 1)} + \frac {8 B ^ {2} L ^ {2} \eta^ {3} s K ^ {3}}{n + 1} + \frac {4 B ^ {2} s ^ {2} K ^ {2} L \eta^ {2}}{(n + 1) ^ {2}} \big) \mathbb {E} \| \nabla f (\mu_ {t}) \| ^ {2}. \\ \end{array}
$$

For  $\eta = (n + 1) / \sqrt{T}$ , as stated in the Theorem, we can use Lemma 4.7 to cancel out the terms containing the potential  $\Phi_t$  (after summing up the inequality over  $T$  steps). Replacing these terms, and modulo some additional term wrangling, we obtain the claimed convergence bound.

![](images/bc4f406e2f0f85271948f7dbf5f8e41f5ac90389437bfb5cc35dd77da1ed83d0.jpg)  
Figure 1: Impact of peers  $s \in \{10,20,30,40\}$  on convergence, for  $n = 100$  clients, 14-bit quantization, on CelebA.

![](images/d6f230fb705266499c576056afb103dedf2d9aa9c0be5f26911fcc45fbd14dc8.jpg)  
Figure 2: Impact of the number of bits  $b \in \{8,10,12,32\}$  on convergence, for  $n = 40$  clients,  $s = 5$  peers on MNIST.

![](images/97e5660dd20eadef39c487485ad3578ae76477b5c272e8d07824d3b4311b3d7f.jpg)  
Figure 3: Impact of different averaging variants on the validation accuracy of the algorithm on CelebA, vs. rounds.

![](images/128f1f76ed387cbdefaa7f537ebcb182f69ffa219c7149c87c5798fa037b01d1.jpg)  
Figure 4: Convergence comparison relative to simulated time between QuAFL and FedAvg, for ResNet20/CIFAR10.

Quantization Impact. Finally, we address the correctness of the quantization technique. We show that the quantization fails with negligible probability:

Lemma 4.8. Let  $T \geq \Omega(n^3)$ , then for quantization parameters  $R = 2 + T^{\frac{3}{d}}$  and  $\gamma^2 = \frac{\eta^2}{(R^2 + 7)^2} (\sigma^2 + 2KG^2 + \frac{f(\mu_0) - f_*}{L})$  we have that the probability of quantization never failing during the entire run of the Algorithm 1 is at least  $1 - O\left(\frac{1}{T}\right)$ .

Per Lemma 4.1, in order for the communication to fail with negligible probability, we need to show that whenever the server communicates with a client, the two norm of their local models is at most  $R^{R^d} \gamma$ . Hence, we need to use bound  $\mathbb{E}[\Phi(t)]$ . The only similar use of this technique was in Nadiradze et al. (2021); however, the authors of this reference could benefit from assuming that the second-moment of the gradients was bounded. Since we make no such assumption here, we need to find a way to bound  $\sum_{t=0}^{T-1} \mathbb{E} \| \nabla f(\mu_t) \|^2$ . Fortunately, our main result shows that the gradients are vanishing, so we can take the advantage of the convergence rate and plug it back into Lemma 4.7.

Similarly, due to the Property 3, Lemma 4.1, the number of the bits used by our algorithm, in one communication between the server and a client, depends on the two norm of the distance between their local models. Thus, we can use the bound on  $\sum_{t=0}^{T-1} \mathbb{E}\|\nabla f(\mu_t)\|^2$  to show the following.

Lemma 4.9. Let  $T \geq \Omega(n^3)$ , then for quantization parameters  $R = 2 + T^{\frac{3}{d}}$  and  $\gamma^2 = \frac{\eta^2}{(R^2 + 7)^2} (\sigma^2 + 2KG^2 + \frac{f(\mu_0) - f_*}{L})$  we have that the expected number of bits used by Algorithm 1 in total is  $O(sT(d\log(n) + \log(T))$ .

We note that the communication cost per step is also asymptotically optimal, modulo the multiplicative  $\log n$  and additive  $\log T$  terms, required to ensure error probability  $1 - O(1 / T)$ .

# 5 EXPERIMENTAL RESULTS

Experimental Setup and Goals. We implemented our algorithm in Pytorch in order to train neural networks for image classification tasks, specifically residual CNNs (He et al., 2016) on the MNIST (LeCun & Cortes, 2010), Fashion MNIST (Xiao et al., 2017), CIFAR-10 (Krizhevsky & Hinton, 2009) and CelebA (Liu et al., 2015) datasets, in the rigorous FL setup of LEAF (Caldas et al., 2018). The model and dataset details are presented in the Appendix. We aim to validate our analysis relative to the impact of various parameters. We omit error bars for readability, as we observed that the variance is very low.

Specifically, some of the basic parameter we examine are  $n$ ,  $s$ , and  $K$ , which have the same meaning as in our theoretical analysis. Our experiments are described by  $(n, s, K, b)$ , where  $b$  is the number of bits for quantization. In addition, we define  $swt$  as the server waiting time between two consecutive calls, and the server interaction time,  $sit$ , as the amount of time that server needs to send and receive necessary data. We assume a server and  $n$  clients. The training dataset is distributed among clients so that each has access to a fixed  $1/n$  partition of the training data. We track the accuracy of the server's model on an unseen validation dataset. We measure loss and accuracy of the model with respect to simulation time and total gradient steps performed by clients. In each round, the server chooses  $s$  clients uniformly at random. It then sends its model to those clients and asks for their current local models. Each client will have taken a maximum of  $K$  local steps by the time it is contacted by the server. We update the both client and server models following QuAFL, and then increase the server time by  $sit$ . The server then waits for another interval of server waiting time  $swt$  to make its next call. All communication is compressed, as models get encoded in their source and decoded in their destination.

We differentiate between two types of timing experiments: uniform timing experiments assume all clients take the same amount of time for a gradient step, while in non-uniform timing experiments we differentiate clients to be either fast or slow. Specifically, the length of each client step is taken to be a random variable  $X \sim \text{exponential}(\lambda)$ , where  $\lambda$  is  $1/2$  for fast clients and  $1/8$  for slow clients; the expected runtime  $\mathbb{E}(X)$  would be 2 and 8, respectively. In each timing experiment, we assumed  $30\%$  of clients to be slow.

In our first experiment, presented in Figure 1, we examine the impact of the number of peers  $s$  sampled by the algorithm, when training ResNet18 on the CelebA dataset, where we assumed  $30\%$  of clients to be slow. We first observe that convergence speed clearly follows the ordering of the number of peers  $s$ , confirming our analysis. Interestingly, the interaction and step timings in this experiment are set up so that there is a  $27\%$  probability that a slow client will not have taken any steps when interacting with the server. (We validated this proportion experimentally; this probability decreases as  $s$  increases.) Thus, this experiment also shows that QuAFL is indeed robust to such slow clients, although their proportion can impact convergence. In our second experiment (Figure 2), we examine the impact of the number of quantization bits  $b$  on the convergence of the algorithm. Again, we observe that increasing the number of bits from 8 to 10 improves convergence; however, there is a clear saturation occurring after 10 bits, after which the algorithm converges at the same rate as the full-precision baseline.

In our third experiment, provided in Figure 3, we examine the impact of different types of averaging on the convergence of the basic QuAFL pattern, on the CelebA dataset, with  $n = 100$  clients. All variants execute in the same setup, with individually-tuned hyper-parameters. We clearly observe that the variant where averaging is applied both at the server and at the client performs the best, which indirectly validates our algorithmic choices.

Finally, in Figure 4 we examine the validation accuracy ensured by FedAvg and QuAFL versus the simulated execution time, in a system with 20 clients, out of which  $25\%$  are slow. (The Baseline is a single slow node that performs an optimization step per round.) Here, it becomes evident that the asynchronous nature of QuAFL communication allows it to provide a faster convergence speed in terms of wall-clock time, than its synchronous counterpart. We present additional experimental results in the Appendix, specifically on higher node counts (up to 300), as well as other tasks.

# 6 CONCLUSIONS AND LIMITATIONS

We have provided the first variant of FedAvg which incorporates both asynchronous and compressed communication, and have shown that this algorithm can still provide good convergence guarantees. Our analysis should be extensible to more complex federated optimizers, such as gradient tracking, e.g. (Haddadpour et al., 2021), controlled averaging (Karimireddy et al., 2020), or variance-reduced variants (Gorbunov et al., 2021).

Our work has the following limitations. First, this version of the analysis requires the expected number of local steps  $H$  to be the same across all devices. We believe that this can be addressed either by modifying the objective, or by de-biasing via sampling. Second, our algorithm has an optimal convergence rate when  $H = \Theta(K)$ , which we believe is natural due to asynchrony. Third, Assumption (3) does impose restrictions on the way data is distributed among the clients. We believe that it is necessary for the convergence of our algorithm since we do not employ variance-reduction or weighted sampling techniques.

# 7 REPRODUCIBILITY STATEMENT

All the code required to reproduce our experimental setup and our experiments is available at https://anonymous.4open.science/r/QuAFL-Anonymous.

# REFERENCES

Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. QSGD: Communication-efficient sgd via gradient quantization and encoding. In NIPS, pp. 1709-1720, 2017.  
Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi, Brendan McMahan, et al. Towards federated learning at scale: System design. Proceedings of Machine Learning and Systems, 1: 374-388, 2019.  
Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H Brendan McMahan, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. arXiv preprint arXiv:1812.01097, 2018.  
Mingzhe Chen, Nir Shlezinger, H Vincent Poor, Yonina C Eldar, and Shuguang Cui. Communication-efficient federated learning. Proceedings of the National Academy of Sciences, 118(17), 2021.  
Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd. Advances in neural information processing systems, 32, 2019.  
Peter Davies, Vijaykrishna Gurunanthan, Niusha Moshrefi, Saleh Ashkboos, and Dan Alistarh. New bounds for distributed mean estimation and variance reduction. In International Conference on Learning Representations, 2021.  
Eduard Gorbunov, Konstantin P Burlachenko, Zhize Li, and Peter Richtárik. MARINA: Faster nonconvex distributed learning with compression. In International Conference on Machine Learning, pp. 3788-3798. PMLR, 2021.  
Farzin Haddadpour and Mehrdad Mahdavi. On the convergence of local descent methods in federated learning. arXiv preprint arXiv:1910.14425, 2019.  
Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, and Mehrdad Mahdavi. Federated learning with compression: Unified analysis and sharp guarantees. In International Conference on Artificial Intelligence and Statistics, pp. 2350-2358. PMLR, 2021.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Lie He, An Bian, and Martin Jaggi. COLA: Decentralized linear learning. Advances in Neural Information Processing Systems, 31, 2018.  
Divyansh Jhunjhunwala, Advait Gadhikar, Gauri Joshi, and Yonina C Eldar. Adaptive quantization of model updates for communication-efficient federated learning. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3110-3114. IEEE, 2021.  
Richeng Jin, Yufan Huang, Xiaofan He, Huaiyu Dai, and Tianfu Wu. Stochastic-sign SGD for federated learning with theoretical guarantees. arXiv preprint arXiv:2002.10940, 2020.  
Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and Trends® in Machine Learning, 14(1-2):1-210, 2021.  
Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes signSGD and other gradient compression schemes. In International Conference on Machine Learning, pp. 3252-3261. PMLR, 2019.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International Conference on Machine Learning, pp. 5132-5143. PMLR, 2020.  
Anastasia Koloskova, Sebastian Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. In International Conference on Machine Learning, pp. 3478-3487. PMLR, 2019.  
Jakub Konečný, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.  
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. 2009.  
Yann LeCun and Corinna Cortes. MNIST handwritten digit database. 2010. URL http://yann.lecun.com/exdb/mnist/.  
Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. IEEE Signal Processing Magazine, 37(3):50-60, 2020.  
Zhize Li and Peter Richtárik. CANITA: Faster rates for distributed convex optimization with communication compression. Advances in Neural Information Processing Systems, 34, 2021.  
Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jio Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. arXiv preprint arXiv:1705.09056, 2017.  
Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.  
Yucheng Lu and Christopher De Sa. Moniqua: Modulo quantized communication in decentralized sgd. In International Conference on Machine Learning, pp. 6415-6425. PMLR, 2020.  
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273-1282. PMLR, 2017.  
Konstantin Mishchenko, Eduard Gorbunov, Martin Takáč, and Peter Richtárik. Distributed learning with compressed gradient differences. arXiv preprint arXiv:1901.09269, 2019.  
Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, and Dan Alistarh. Asynchronous decentralized sgd with quantized and local updates. Advances in Neural Information Processing Systems, 34, 2021.  
John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek, and Dzmitry Huba. Federated learning with buffered asynchronous aggregation. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera (eds.), Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pp. 3581-3607. PMLR, 28-30 Mar 2022. URL https://proceedings.mlr.press/v151/nguyen22b.html.  
Constantin Philippenko and Aymeric Dieuleveut. Bidirectional compression in heterogeneous settings for distributed or federated learning with partial participation: tight convergence guarantees. arXiv preprint arXiv:2006.14591, 2020.  
Sashank Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Konečný, Sanjiv Kumar, and H Brendan McMahan. Adaptive federated optimization. arXiv preprint arXiv:2003.00295, 2020.  
Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jabbabaie, and Ramtin Pedarsani. Fedpaq: A communication-efficient federated learning method with periodic averaging and quantization. In International Conference on Artificial Intelligence and Statistics, pp. 2021-2031. PMLR, 2020.

F. Seide, H. Fu, L. G. Jasha, and D. Yu. 1-bit stochastic gradient descent and application to data-parallel distributed training of speech dnns. Interspeech, 2014.  
Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. Advances in neural information processing systems, 30, 2017.  
Sebastian U Stich. Local SGD converges fast and communicates little. arXiv preprint arXiv:1805.09767, 2018.  
Hanlin Tang, Ce Zhang, Shaoduo Gan, Tong Zhang, and Ji Liu. Decentralization meets quantization. CoRR, abs/1803.06443, 2018.  
Qianqian Tong, Guannan Liang, and Jinbo Bi. Effective federated adaptive gradient methods with non-iid decentralized data. arXiv preprint arXiv:2009.06557, 2020.  
Alexander Tyurin and Peter Richtárik. DASHA: Distributed nonconvex optimization with communication compression, optimal oracle complexity, and no client synchronization. arXiv preprint arXiv:2202.01268, 2022.  
Yujia Wang, Lu Lin, and Jinghui Chen. Communication-efficient adaptive federated learning. arXiv preprint arXiv:2205.02719, 2022.  
Wentai Wu, Ligang He, Weiwei Lin, Rui Mao, Carsten Maple, and Stephen Jarvis. Safa: a semisynchronous protocol for fast federated learning with low overhead. IEEE Transactions on Computers, 70(5):655-668, 2020.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.
