# SLALOM: FAST, VERIFIABLE AND PRIVATE EXECUTION OF NEURAL NETWORKS IN TRUSTED HARDWARE

Anonymous authors

Paper under double-blind review

# ABSTRACT

As Machine Learning (ML) gets applied to security-critical or sensitive domains, there is a growing need for integrity and privacy for outsourced ML computations. A pragmatic solution comes from Trusted Execution Environments (TEEs), which use hardware and software protections to isolate sensitive computations from the untrusted software stack. However, these isolation guarantees come at a price in performance, compared to untrusted alternatives. This paper initiates the study of high performance execution of Deep Neural Networks (DNNs) in TEEs by efficiently partitioning DNN computations between trusted and untrusted devices. Building upon an efficient outsourcing scheme for matrix multiplication, we propose Slalom, a framework that securely delegates execution of all linear layers in a DNN from a TEE (e.g., Intel SGX or Sanctum) to a faster, yet untrusted, co-located processor. We evaluate Slalom by executing DNNs in an Intel SGX enclave, which selectively delegates work to an untrusted GPU. For two canonical DNNs, VGG16 and MobileNet, we obtain  $20 \times$  and  $6 \times$  increases in throughput for verifiable inference, and  $11 \times$  and  $4 \times$  for verifiable and private inference.

# 1 INTRODUCTION

Machine learning is increasingly used in sensitive decision making and security-critical settings. At the same time, the growth in both cloud offerings and software stack complexity widens the attack surface for ML applications. This raises the question of integrity and privacy guarantees for ML computations in untrusted environments, in particular for ML tasks outsourced by a client to a remote server. Prominent examples include cloud-based ML APIs (e.g., a speech-to-text application that consumes user-provided data) or general ML-as-a-Service platforms.

Trusted Execution Environments (TEEs), e.g., Intel SGX (McKeen et al., 2013), ARM TrustZone (Alves & Felton, 2004) or Sanctum (Costan et al., 2016) offer a pragmatic solution to this problem. TEEs use hardware and software protections to isolate sensitive code from other applications, while attesting to its correct execution. Running outsourced ML computations in TEEs provides remote clients with strong privacy and integrity guarantees.

For outsourced ML computations, TEEs outperform pure cryptographic approaches (e.g., (Gilad-Bachrach et al., 2016; Mohassel & Zhang, 2017; Ghodsi et al., 2017; Juvekar et al., 2018)) by multiple orders of magnitude. At the same time, the isolation guarantees of TEEs still come at a steep price in performance, compared to untrusted alternatives (i.e., running ML models on contemporary hardware with no security guarantees). For instance, Intel SGX (Intel Corp., 2015) incurs significant overhead for memory intensive tasks (Orenbach et al., 2017; Harnik & Tsfadia, 2017), has difficulties exploiting multi-threading, and is currently limited to desktop CPUs that are unmatched by untrusted alternatives (e.g., GPUs or server CPUs). Thus, our thesis is that for modern ML workloads, TEEs will be at least an order of magnitude less efficient than the best available untrusted hardware.

This leads us to the main question of this paper: How can we most efficiently leverage TEEs for secure machine learning? This was posed by Stoica et al. (2017) as one of nine open research problems for system challenges

in AI. A specific challenge they raised is that of appropriately splitting ML computations between trusted and untrusted components, to increase efficiency as well as security by minimizing the Trusted Computing Base.

This paper explores a novel approach to this challenge, wherein a Deep Neural Network (DNN) execution is partially outsourced from a TEE to a co-located, untrusted but faster device. Our approach, inspired by the verifiable ASICs of Wahby et al. (2016), differs from cryptographic ML outsourcing. In our case, work is delegated between two co-located parties, thus allowing for highly interactive—yet conceptually simpler—outsourcing protocols with orders-of-magnitude better efficiency. Our work also deviates from prior systems which execute DNNs fully in a TEE (Ohrimenko et al., 2016; Hunt et al., 2018; Cheng et al., 2018).

The main observation that guides our approach is that matrix multiplication—the main bottleneck in DNNs—admits a concretely efficient verifiable outsourcing scheme known as Freivalds' algorithm (Freivalds, 1977), which can also be turned private in our setting. Our TEE selectively outsources these CPU intensive steps to a fast untrusted co-processor (and runs the remaining steps itself) therefore achieving much better performance than running the entire computation in the enclave, without compromising security.

Contributions. We propose Slalom, a framework for efficient DNN inference in any trusted execution environment (e.g., SGX or Sanctum). To evaluate Slalom, we build a lightweight DNN library for Intel SGX, which may be of independent interest. Our library allows for outsourcing all linear layers to an untrusted GPU without compromising integrity or privacy. Our code is available at https://redacted-for-submission.

We formally prove Slalom's security, and evaluate it on two DNNs—VGG16 (Simonyan & Zisserman, 2014) and MobileNet (Howard et al., 2017)—that lie at two extremes of the computational efficiency spectrum. Compared to running all computations in SGX, outsourcing linear layers to an untrusted GPU increases throughput (and energy efficiency to a similar extent) by  $6 - 20 \times$  for verifiable inference, and by  $4 - 11 \times$  for verifiable and private inference. Finally, we discuss open challenges towards efficient verifiable training of DNNs in TEEs.

# 2 BACKGROUND

# 2.1 PROBLEM SETTING

We consider an outsourcing scheme between a client  $\mathcal{C}$  and a server  $S$ , where  $S$  executes a DNN  $F(x): \mathcal{X} \to \mathcal{Y}$  on data provided by  $\mathcal{C}$ . The DNN can either belong to the user (e.g., as in some ML-as-a-service platforms), or to the server (e.g., as in a cloud-based ML API). Depending on the application, this scheme should satisfy one or more of the following security properties (see Appendix B for formal definitions):

- t-Integrity: For any  $S$  and input  $x$ , the probability that a user interacting with  $S$  does not abort (i.e., output  $\bot$ ) and outputs an incorrect value  $\tilde{y} \neq F(x)$  is less than  $t$ .  
- Privacy: The server  $S$  learns no information about the user's input  $x$ .  
- Model privacy: If the model  $F$  is provided by the user,  $S$  learns no information about  $F$  (beyond e.g., its approximate size). If  $F$  belongs to the server,  $\mathcal{C}$  learns no more about  $F$  than what is revealed by  $y = F(x)$ .<sup>1</sup>

# 2.2 TRUSTED EXECUTION ENVIRONMENTS (TEES), INTEL SGX, AND A STRONG BASELINE

Trusted Execution Environments (TEE) such as Intel SGX, ARM TrustZone or Sanctum (Costan et al., 2016) enable execution of programs in secure enclaves. Hardware protections isolate computations in enclaves from all programs on the same host, including the operating system. Enclaves can produce remote attestations—digital

Table 1: Security guarantees and performance (relative to baseline) of different ML outsourcing schemes.  

<table><tr><td rowspan="2">Approach</td><td rowspan="2">TEE</td><td rowspan="2">Integrity</td><td rowspan="2">Privacy</td><td colspan="2">Model Privacy</td><td rowspan="2">Throughput (relative)</td></tr><tr><td>w.r.t. Server</td><td>w.r.t. Client</td></tr><tr><td>SafetyNets (Ghodsi et al., 2017)</td><td>-</td><td>●</td><td>○</td><td>○</td><td>○</td><td>≤1/200×</td></tr><tr><td>Gazelle (Juvekar et al., 2018)</td><td>-</td><td>○</td><td>●*</td><td>○</td><td>4</td><td>≤1/1000×</td></tr><tr><td>Secure baseline (run DNN in TEE)</td><td>✓</td><td>●</td><td>●</td><td>●</td><td>●</td><td>1×</td></tr><tr><td>Insecure baseline (run DNN on GPU)</td><td>-</td><td>○</td><td>○</td><td>○</td><td>4</td><td>≥50×</td></tr><tr><td>Slalom (Ours)</td><td>✓</td><td>●</td><td>●*</td><td>○</td><td>●</td><td>4× - 20×</td></tr></table>

* With an offline preprocessing phase.

signatures over an enclave's code—that a remote party can verify using the manufacturer's public key. Our experiments with Slalom use hardware enclaves provided by Intel SGX (see Appendix A for details).<sup>2</sup>

TEEs offer an efficient solution for ML outsourcing: The server runs an isolated enclave that establishes a secure communication channel with  $\mathcal{C}$  and evaluates a model  $F$  on  $\mathcal{C}$ 's input data. This simple scheme (which we implemented in SGX, see Section 4) outperforms cryptographic ML outsourcing protocols by 2-3 orders of magnitude. See Table 1 for a comparison to two representative works, and Appendix C for more details.

Yet, SGX's security comes at a performance cost, and there remains a large gap between TEEs and untrusted devices. For example, current SGX CPUs are limited to 128 MB of Processor Reserved Memory (PRM) (Costan & Devadas, 2016) and incur severe paging overheads when exceeding this allowance (Orenbach et al., 2017). We also failed to achieve noticeable speed ups for multi-threaded DNN evaluations in SGX enclaves (see Appendix I). For DNN computations, current SGX enclaves thus cannot compete—in terms of performance or energy efficiency (see Appendix C)—with contemporary untrusted hardware, such as a GPU or server CPU.

In this work, we treat the above simple (yet powerful) TEE scheme as a baseline, and identify settings where we can still improve upon it. We will show that our system, Slalom, substantially outperforms this baseline when the server has access to the model  $F$  (e.g.,  $F$  belongs to  $S$  as in cloud ML APIs, or  $F$  is public). Slalom performs best for verifiable inference (the setting considered in SafetyNets (Ghodsi et al., 2017)). If the TEE can run some offline data-independent preprocessing (e.g., as in Gazelle (Juvekar et al., 2018)), Slalom also outperforms the baseline for private (and verifiable) outsourced computations in a later online phase. Such a two-stage approach is viable if user data is sent at irregular intervals yet has to be processed with high throughput when available.

# 2.3 OUTSOURCING OUTSOURCED DNNS AND FREIVALDS' ALGORITHM

Our idea for speeding up DNN inference in TEEs is to further outsource work from the TEE to a co-located faster untrusted processor. Improving upon the above baseline thus requires that the combined cost of doing work on the untrusted device and verifying it in the TEE be cheaper than evaluating the full DNN in the TEE.

Wahby et al. (2016; 2017) aim at this goal for arbitrary computations outsourced between co-located ASICs. The generic non-interactive proofs they use for integrity are similar to those used in SafetyNets (Ghodsi et al., 2017), which incur overheads that are too large to warrant outsourcing in our setting (e.g., Wahby et al. (2016) find that the technology gap between trusted and untrusted devices needs to be of over two decades for their scheme to break even). Similarly for privacy, standard cryptographic outsourcing protocols (e.g., (Juvekar et al., 2018)) are unusable in our setting as simply running the computation in the TEE is much more efficient (see Table 1).

To overcome this barrier, we design outsourcing protocols tailored to DNNs, leveraging two insights:

1. In our setting, the TEE is co-located with the server's faster untrusted processors, thus widening the design space to interactive outsourcing protocols with high communication but better efficiency.  
2. The TEE always has knowledge of the model and can selectively outsource part of the DNN evaluation and compute others—for which outsourcing is harder—itself.

DNNs are a class of functions that are particularly well suited for selective outsourcing. Indeed, non-linearities—which are hard to securely outsource (with integrity or privacy)—represent a small fraction of the computation in a DNN so we can evaluate these in the TEE (e.g., for VGG16 inference on a single CPU thread, about  $1.5\%$  of the computation is spent on non-linearities). In contrast, linear operators—the main computational bottleneck in DNNs—admit for a conceptually simple yet concretely efficient secure delegation scheme, described below.

$\textbf{Integrity.}$  We verify integrity of outsourced linear layers using variants of an algorithm by Freivalds (1977).

Lemma 2.1 (Freivalds). Let  $A, B$  and  $C$  be  $n \times n$  matrices over a field  $\mathbb{F}$  and let  $s$  be a uniformly random vector in  $\mathbb{S}^n$ , for  $\mathbb{S} \subseteq \mathbb{F}$ . Then,  $\operatorname*{Pr}_{s \leftarrow^{\mathbb{R}}\mathbb{S}^n}[Cs = A(Bs) \mid C \neq AB] = \operatorname*{Pr}_{s \leftarrow^{\mathbb{R}}\mathbb{S}^n}[(C - AB)s = \mathbf{0} \mid (C - AB) \neq \mathbf{0}] \leq 1 / |\mathbb{S}|$ .

The randomized check requires  $3n^2$  multiplications, a significant reduction (both in concrete terms and asymptotically) over evaluating the product directly. The algorithm has no false negatives and trivially extends to rectangular matrices. Independently repeating the check  $k$  times yields soundness error  $1 / |\mathbb{S}|^k$ .

Privacy. Input privacy for outsourced linear operators could be achieved with linearly homomorphic encryption, but the overhead (see the micro-benchmarks in (Juvekar et al., 2018)) is too high to compete with our baseline (i.e., computing the function directly in the TEE would be faster than outsourcing it over encrypted data).

We instead propose a very efficient two-stage approach based on symmetric cryptography, i.e., an additive stream cipher. Let  $f: \mathbb{F}^m \to \mathbb{F}^n$  be a linear function over a field  $\mathbb{F}$ . In an offline phase, the TEE generates a stream of one-time-use pseudorandom elements  $r \in \mathbb{F}^m$ , and pre-computes  $u = f(r)$ . Then, in the online phase when the remote client sends an input  $x$ , the TEE computes  $\mathsf{Enc}(x) = x + r$  over  $\mathbb{F}^m$  (i.e., a secure encryption of  $x$  with a stream cipher), and outsources the computation of  $f(\mathsf{Enc}(x))$  to the faster processor. Given the result  $f(\mathsf{Enc}(x)) = f(x + r) = f(x) + f(r) = f(x) + u$ , the TEE recovers  $f(x)$  using the pre-computed  $u$ .

Communication. Using Freivalsd' algorithm and symmetric encryption for each linear layer in a DNN incurs high interaction and communication between the TEE and untrusted co-processor (e.g., over 50MB per inference for VGG16, see Table 3). This would be prohibitive if they were not co-located. There are protocols with lower communication than repeatedly using Freivalsd' ((Fiore & Gennaro, 2012; Thaler, 2013; Ghodsi et al., 2017)). Yet, these incur a high overhead on the prover in practice and are thus not suitable in our setting.

# 3 SLALOM

We introduce Slalom, a three-step approach for outsourcing DNNs from a TEE to an untrusted but faster device: (1) Inputs and weights are quantized and embedded in a field  $\mathbb{F}$ ; (2) Linear layers are outsourced and verified using Freivals' algorithm; (3) Inputs of linear layers are encrypted with a pre-computed pseudorandom stream to guarantee privacy. Figure 1 shows two Slalom variants, one to achieve integrity, and one to also achieve privacy.

We focus on feed-forward networks with fully connected layers, convolutions, separable convolutions, pooling layers and activations, although Slalom could be extended to other architecture (e.g., residual networks).

# 3.1 QUANTIZATION

The techniques we use for integrity and privacy (Freivalds' algorithm and stream ciphers) work over a field  $\mathbb{F}$ . We thus quantize all inputs and weights of a DNN to integers, and embed these integers in the field  $\mathbb{Z}_p$  of integers modulo a prime  $p$  (where  $p$  is larger than all values computed in a DNN evaluation, so as to avoid wrap-around).

![](images/40358e7ee5a97f864c07af8fad4dd67dc294ae99a6f3889fb884d92dda6bb6eb.jpg)  
Figure 1: The Slalom algorithms for verifiable and private DNN inference. The TEE outsources computation of  $n$  linear layers of a model  $F$  to the untrusted host server  $S$ . Each linear layer is defined by a matrix  $W_{i}$  of size  $m_{i} \times n_{i}$  and followed by an activation  $\sigma$ . All operations are over a field  $\mathbb{F}$ . The Freivalds  $(y_{i}, x_{i}, w_{i})$  subroutine performs  $k$  repetitions of Freivalds' check (possibly using precomputed values as in Section 3.2). The pseudorandom elements  $r_{i}$  (we omit the PRNG for simplicity) and precomputed values  $u_{i}$  are used only once.

![](images/4460eb55df5a869a44f7a06c3c7df6ee039a4d1560425d3055c51dcc8f997dac.jpg)

As in (Gupta et al., 2015), we convert floating point numbers  $x$  to a fixed-point representation as  $\tilde{x} = \mathbb{F}\mathbb{P}(x;l) \coloneqq \mathrm{round}(2^l \cdot x)$ . For a linear layer with kernel  $W$  and bias  $b$ , we define integer parameters  $\tilde{W} = \mathbb{F}\mathbb{P}(W,l), \tilde{b} = \mathbb{F}\mathbb{P}(b,2l)$ . After applying the layer to a quantized input  $\tilde{x}$ , we scale the output by  $2^{-l}$  and re-round to an integer.

For efficiency reasons, we perform integer arithmetic using floats (so-called fake quantization), and choose  $p < 2^{24}$  to avoid loss of precision (we use  $p = 2^{24} - 3$ ). For the models we evaluate, setting  $l = 8$  for all weights and inputs ensures that all DNN values are bounded by  $2^{24}$ , with less than a  $0.5\%$  drop in accuracy (see Table 3). When performing arithmetic modulo  $p$  (e.g., for Freivalds' algorithm or when computing on encrypted data), we use double-precision floats, to reduce the number of modular reductions required (details are in Appendix F).

# 3.2 VERIFYING COMMON LINEAR OPERATORS

We now describe Slalom's approach to verifying the integrity of outsourced linear layers. We describe these layers in detail in Appendix D and summarize this section's results in Table 2.

Freivalds' Algorithm for Batches. The most direct way of applying Freivalds' algorithm to arbitrary linear layers of a DNN is by exploiting batching. Any linear layer  $f(x)$  from inputs of size  $m$  to outputs of size  $n$  can be represented (with appropriate reshaping) as  $f(x) = x^{\top}W$  for a (often sparse and implicit)  $m \times n$  matrix  $W$ .

For a batch  $X$  of size  $B$ , we can outsource  $f(X)$  and check that the output  $Y$  satisfies  $f(s^{\top}X) = s^{\top}Y$ , for a random vector  $s$  (we are implicitly applying Freivalds to the matrix product  $XW = Y$ ). As the batch size  $B$  grows, the cost of evaluating  $f$  is amortized and the total verification cost is  $|X| + |Y| + \mathrm{cost}_f$  multiplications (i.e., we approach one operation per input and output). Yet, as we show in Appendix G, while batched verification is worthwhile for processors with larger memory, it is prohibitive in SGX enclaves due to the limited PRM.

For full convolutions (and pointwise convolutions), a direct application of Freivals' check is worthwhile even for single-element batches. For  $f(x) = \mathrm{Conv}(x,W)$  and purported output  $y$ , we can sample a random vector  $s$  of dimension  $c_{\mathrm{out}}$  (the number of output channels), and check that  $\mathrm{Conv}(x,Ws) = ys$  (with appropriate reshaping). For a batch of inputs  $X$ , we can also apply Freivals' algorithm twice to reduce both  $W$  and  $X$ .

Preprocessing. We now show how to obtain an outsourcing scheme for linear layers that has optimal verification complexity (i.e.,  $|x| + |y|$  operations) for single-element batches and arbitrary linear operators, while at the same time compressing the DNN's weights (a welcome property in our memory-limited TEE model).

Table 2: Complexity (number of multiplications) for evaluating and verifying linear functions. The layers are "Fully Connected", "Convolution", "Depthwise Convolution" and "Pointwise Convolution", defined in Appendix D. Each layer  $f$  has an input  $x$  ,output  $y$  and kernel  $W$  . We assume a batch size of  $B \geq  1$  .  

<table><tr><td>Layer</td><td>|x|, |y|</td><td>|W|</td><td>costf(B = 1)</td><td>Batched verification</td><td>With preproc.</td></tr><tr><td>FC</td><td>h_in, h_out</td><td>h_in · h_out</td><td>|x| · |y|</td><td>B · (|x| + |y|) + costf</td><td>B · (|x| + |y|)</td></tr><tr><td>Conv</td><td>h · w · cin, h · w · cout</td><td>k2 · cin · cout</td><td>|x| · k2 · cout</td><td>B · (|x| + |y|) + cin · cout + |x| · k2</td><td>B · (|x| + |y|)</td></tr><tr><td>Depth. Conv</td><td>h · w · cin, h · w · cin</td><td>k2 · cin</td><td>|x| · k2</td><td>B · (|x| + |y|) + costf</td><td>B · (|x| + |y|)</td></tr><tr><td>Point. Conv</td><td>h · w · cin, h · w · cout</td><td>cin · cout</td><td>|x| · cout</td><td>B · (|x| + |y|) + cin · cout</td><td>B · (|x| + |y|)</td></tr></table>

We leverage two facts: (1) DNN weights are fixed at inference time, so part of Freivals' check can be precomputed; (2) the TEE can keep secrets from the host  $S$ , so the random values  $s$  can be re-used across layers or inputs (if we run Freivals' check  $n$  times with the same secret randomness, the soundness errors grows at most by a factor  $n$ ). Our verification scheme with preprocessing follows from a reformulation of Lemma (2.1):

Lemma 3.1. Let  $f: \mathbb{F}^m \to \mathbb{F}^n$  be a linear operator,  $f(x) \coloneqq x^\top W$ . Let  $s$  be uniformly random in  $\mathbb{S}^n$ , for  $\mathbb{S} \subseteq \mathbb{F}$ , and let  $\tilde{s} \coloneqq \nabla F_x(s) = Ws$ . For any  $x \in \mathbb{F}^m$ ,  $y \in \mathbb{F}^n$ , we have  $\operatorname*{Pr}_{s \stackrel{\mathrm{R}}{\leftarrow} \mathbb{S}^n} [y^\top s = x^\top \tilde{s} \mid y \neq f(x)] \leq 1 / |\mathbb{S}|$ .

The cost of the check is  $|x| + |y|$  multiplications, at the expense of storing  $s$  and  $\tilde{s} \coloneqq Ws$  (of size  $|x|$  and  $|y|$  respectively). To save space, we can reuse the same  $s$  (or slices of it) for every layer. The memory footprint of a model is then equal to the size of the inputs of all its linear layers. For VGG16 for instance, the memory requirements for the TEE are reduced from 550MB to about 36MB (see Table 3).

# 3.3 INPUT PRIVACY

To guarantee privacy of the client's inputs, we use precomputed blinding factors for each outsourced computation, as described in Section 2.3. The TEE uses a cryptographic Pseudo Random Number Generator (PRNG) to generate blinding factors. The precomputed "unblinding factors" are encrypted and stored in untrusted memory or disk. In the online phase, the TEE regenerates the blinding factors using the same PRNG seed, and uses the precomputed unblinding factors to decrypt the output of the outsourced linear layer.

This blinding process incurs several overheads: (1) the computations on the untrusted device have to be performed over  $\mathbb{Z}_p$  so we use double-precision arithmetic. (2) The trusted and untrusted processors exchange data in-between each layer, rather than at the end of a full inference pass. (3) The TEE has to efficiently load precomputed unblinding factors, which requires either a large amount of RAM, or a fast access to disk (e.g., a PCIe SSD).

Slalom's security is summarized by the following results. Formal definitions and proofs are in Appendix B.

Theorem 3.2. Let Slalom be the protocol from Figure 1 (right), where  $F$  is an  $n$ -layer DNN, and Freivalds' algorithm is repeated  $k$  times per layer with random vectors drawn from  $\mathbb{S} \subseteq \mathbb{F}$ . Assume all random values are generated using a secure PRNG with security parameter  $\lambda$ . Then, Slalom is a secure outsourcing scheme for  $F$  between a TEE and an untrusted co-processor  $S$  with privacy and  $t$ -integrity for  $t = \sqrt[n]{|\mathbb{S}|^k} - \mathrm{negl}(\lambda)$ .

Corollary 3.3. Assuming the TEE is secure (i.e., it acts as a trusted third party hosted by  $S$ ), Slalom is a secure outsourcing scheme between a remote client  $C$  and server  $S$  with privacy and  $t$ -integrity for  $t = n / |\mathbb{S}|^k - \mathrm{negl}(\lambda)$ . If the model  $F$  is the property of  $S$ , the scheme further satisfies model privacy.

# 4 EMPIRICAL EVALUATION

We evaluate Slalom on real Intel SGX hardware, on synthetic micro-benchmarks and two sample applications (ImageNet inference with VGG16 and MobileNet). Our aim is to show that, compared to a baseline that runs inference fully in the TEE, outsourcing linear layers increases performance without sacrificing security.

# 4.1 IMPLEMENTATION

As enclaves cannot access most OS features (e.g., multi-threading, disk and driver IO), porting a large framework such as TensorFlow or Intel's MKL-DNN to SGX is hard. Instead, we designed a lightweight  $\mathrm{C + + }$  library for feed-forward networks based on Eigen, a linear-algebra library which TensorFlow uses as a CPU backend. Our library implements the forward pass of DNNs, with support for dense layers, standard and separable convolutions, pooling, and activations. When run on a native CPU (without SGX), its performance is comparable to TensorFlow on CPU (compiled with AVX). Our code is available at https://redacted-for-submission.

Slalom performs arithmetic over  $\mathbb{Z}_p$ , for  $p = 2^{24} - 3$ . For integrity, we apply Freivalds' check twice to each layer ( $k = 2$ ), with random values from  $\mathbb{S} = [-2^{19}, 2^{19}]$ , to achieve 40 bits of statistical soundness per layer (see Appendix F for details on the selection of these parameters). For a 50-layer DNN,  $S$  has a chance of less than 1 in 22 billion of fooling the TEE on any incorrect DNN evaluation (a slightly better guarantee than in SafetyNets). For privacy, we use AES-CTR and AES-GCM to generate, encrypt and authenticate blinding factors.

# 4.2 SETUP

We use an Intel Core i7-6700 Skylake 3.40GHz processor with 8GB of RAM, a desktop processor with SGX support. The outsourced computations are performed on a co-located Nvidia TITAN XP GPU. Due to a lack of native internal multi-threading in SGX, we run our TEE in a single CPU thread. We discuss challenges for efficient parallelization in Appendix I. We evaluate Slalom on the following workloads:

- Synthetic benchmarks for matrix products, convolutions and separable convolutions, where we compare the enclave's running time for computing a linear operation to that of solely verifying the result (see Appendix G)  
- ImageNet (Deng et al., 2009) classification with VGG16 (Simonyan & Zisserman, 2014) and MobileNet (Howard et al., 2017) models (with fused Batch Normalization layers).

MobileNet, a model tailored for low compute devices, serves as a worst-case benchmark for Slalom, as the model's design aggressively minimizes the amount of computation performed per layer. We also consider a "fused" variant of MobileNet with no activation between depthwise and pointwise convolutions. Removing these activations improves convergence and accuracy (Chollet, 2017; Sheng et al., 2018), while also making the network more outsourcing-friendly (i.e., it is possible to verify a separable convolution in a single step).

Our evaluation focuses on throughput (number of forward passes per second). We also discuss energy efficiency in Appendix C to account for hardware differences between our baseline (TEE only) and Slalom (TEE + GPU).

# 4.3 RESULTS

Micro-Benchmarks. Results of our benchmarks for verifying linear operators are in Appendix G and we summarize our main findings here: (1) Verifying outsourced matrix products and convolutions is much faster  $(3\times -63\times)$  than computing them in the TEE, especially with preprocessing as described in Section 3.2; (2) It is hard to evaluate large matrix products or to leverage batched verification in Intel SGX, due to the limited PRM.

Verifiable Inference. Figure 2 shows the throughout of end-to-end forward passes in two neural networks, VGG16 and MobileNet. For integrity, we compare the secure baseline (executing the DNN fully in the enclave) to two variants of the Slalom algorithm in Figure 1. The first (in red) applies Freivals' algorithm "on-the-fly", while the second more efficient variant (in orange) pre-computes part of Freivals' check as described in Section 3.2.

The VGG16 network is much larger (500MB) than SGX's PRM. As a result, there is a large overhead on the forward pass and verification without preprocessing. If the enclave securely stores preprocessed products  $Wr$  for all network weights, we drastically reduce the memory footprint and achieve up to a  $20.3 \times$  increase in throughput. We also ran the lower-half of the VGG16 network (without the fully connected layers), a common

![](images/865865089fd615d55b5b8918db9fdf241ec998496039b7e5e41956ecba2ff744.jpg)  
Figure 2: Verifiable and private inference with Intel SGX. We show results for VGG16, VGG16 without the fully connected layers, MobileNet, and a fused MobileNet variant with no intermediate activation for separable convolutions. We compare the baseline of fully executing the DNN in the enclave (blue) to different secure outsourcing schemes: integrity with Freivals (red); integrity with Freivals and precomputed secrets (yellow); privacy only (black); privacy and integrity (purple).

approach for extracting features for transfer learning or object recognition (Liu et al., 2016). This part fits in the PRM, and we thus achieve higher throughput for in-encclave forward passes and on-the-fly verification.

For MobileNet, we achieve between  $3.6 \times$  and  $6.4 \times$  speedups when using Slalom for verifiable inference (for the standard or "fused" model, respectively). The speedups are smaller than for VGG16, as MobileNet performs much fewer operations per layer (verifying a linear layer requires computing at least two multiplications for each input and output. The closer the forward pass gets to that lower-bound, the less we can save by outsourcing).

Private Inference. We further benchmark the cost of private DNN inference, where inputs of outsourced linear layers are additionally blinded. Blinding and unblinding each layer's inputs and outputs is costly, especially in SGX due to the extra in-enclave memory reads and writes. Nevertheless, for VGG16 and the fused MobileNet variant without intermediate activations, we achieve respective speedups of  $13.0 \times$  and  $5.0 \times$  for private outsourcing (in black in Figure 2), and speedups of  $10.7 \times$  and  $4.1 \times$  when also ensuring integrity (in purple). For this benchmark, the precomputed unblinding factor are stored in untrusted memory.

We performed the same experiments on a standard CPU (i.e., without SGX) and find that Slalom's improvements are even higher in non-resource-constrained or multi-threaded environments (see Appendix H-I). Slalom's improvements over the baseline also hold when accounting for energy efficiency (see Section C).

# 5 CONCLUSION

This paper has studied the efficiency of evaluating a DNN in a Trusted Execution Environment (TEE) to provide strong integrity and privacy guarantees. We explored new approaches for segmenting a DNN evaluation to securely outsource work from a trusted environment to a faster co-located but untrusted processor.

We designed Slalom, a framework for efficient DNN evaluation that outsources all linear layers from a TEE to a GPU. Slalom leverage Freivals' algorithm for verifying correctness of linear operators, and additionally encrypts inputs with precomputed blinding factors to preserve privacy. Slalom can work with any TEE and we evaluated its performance using Intel SGX on various workloads. For two canonical DNNs (VGG16 and MobileNet), we have shown that Slalom boosts inference throughput without compromising security.

Securely outsourcing matrix products from a TEE has applications in ML beyond DNNs (e.g., non negative matrix factorization, dimensionality reduction, etc.) In Appendix J, we also explore avenues and challenges towards applying similar techniques to DNN training, an interesting direction for future work. Finally, our general approach of outsourcing work from a TEE to a faster co-processor could be applied to other problems which have fast verification algorithms, e.g., those considered in (McConnell et al., 2011; Zhang et al., 2014).

# REFERENCES

Tiago Alves and Don Felton. Trustzone: Integrated hardware and software security-enabling trusted computing in embedded systems. Technical report, ARM, 2004.  
Ferdinand Brasser, Urs Müller, Alexandra Dmitrienko, Kari Kostiainen, Srdjan Capkun, and Ahmad-Reza Sadeghi. Software grand exposure: SGX cache attacks are practical. In USENIX Workshop on Offensive Technologies, 2017.  
Ran Canetti, Yehuda Lindell, Rafail Ostrovsky, and Amit Sahai. Universally composable two-party and multi-party secure computation. In Proceedings of the thirty-fourth annual ACM symposium on Theory of computing, pp. 494-503. ACM, 2002.  
Guoxing Chen, Sanchuan Chen, Yuan Xiao, Yiqian Zhang, Zhiqiang Lin, and Ten H Lai. SGXPECTRE attacks: Leaking enclave secrets via speculative execution. arXiv preprint arXiv:1802.09085, 2018.  
Sanchuan Chen, Xiaokuan Zhang, Michael K Reiter, and Yinqian Zhang. Detecting privileged side-channel attacks in shielded execution with déjà vu. In ACM Asia Conference on Computer and Communications Security (ASIACCS), pp. 7-18. ACM, 2017.  
Raymond Cheng, Fan Zhang, Jernej Kos, Warren He, Nicholas Hynes, Noah Johnson, Ari Juels, Andrew Miller, and Dawn Song. Ekiden: A platform for confidentiality-preserving, trustworthy, and performant smart contract execution. arXiv preprint arXiv:1804.05141, 2018.  
François Chollet. Xception: Deep learning with depthwise separable convolutions. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.  
Victor Costan and Srinivas Devadas. Intel SGX explained. https://eprint.iacr.org/2016/086, 2016.  
Victor Costan, Ilia Lebedev, and Srinivas Devadas. Sanctum: Minimal hardware extensions for strong software isolation. In USENIX Security Symposium, 2016.  
Fergus Dall, Gabrielle De Micheli, Thomas Eisenbarth, Daniel Genkin, Nadia Heninger, Ahmad Moghimi, and Yuval Yarom. Cachequote: Efficiently recovering long-term secrets of sgx epid via cache attacks. IACR Transactions on Cryptographic Hardware and Embedded Systems, 2018(2):171-191, 2018.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248-255. IEEE, 2009.  
Dario Fiore and Rosario Gennaro. Publicly verifiable delegation of large polynomials and matrix computations, with applications. In Proceedings of the 2012 ACM conference on Computer and communications security, pp. 501-512. ACM, 2012.  
Ben Fisch, Dhinakaran Vinayagamurthy, Dan Boneh, and Sergey Gorbunov. Iron: functional encryption using intel sgx. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 765-782. ACM, 2017.  
Rusins Freivalds. Probabilistic machines can use less running time. In IFIP congress, volume 839, pp. 842, 1977.  
Zahra Ghodsi, Tianyu Gu, and Siddharth Garg. Safetynets: Verifiable execution of deep neural networks on an untrusted cloud. In Advances In Neural Information Processing Systems (NIPS), pp. 4675-4684, 2017.  
Ran Gilad-Bachrach, Nathan Dowlin, Kim Laine, Kristin Lauter, Michael Naehrig, and John Wernsing. Cryptonets: Applying neural networks to encrypted data with high throughput and accuracy. In International Conference on Machine Learning (ICML), pp. 201-210, 2016.  
Johannes Görtzfried, Moritz Eckert, Sebastian Schinzel, and Tilo Müller. Cache attacks on Intel SGX. In European Workshop on Systems Security, pp. 2. ACM, 2017.

Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In International Conference on Machine Learning (ICML), pp. 1737-1746, 2015.  
Danny Harnik and Eliad Tsfadia. Impressions of Intel SGX performance. https://medium.com/@danny_harnik/ impressions-of-intel-sgx-performance-22442093595a, 2017. Accessed on May 17, 2018.  
Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.  
Tyler Hunt, Congzheng Song, Reza Shokri, Vitaly Shmatikov, and Emmett Witchel. Chiron: Privacy-preserving machine learning as a service. arXiv preprint arXiv:1803.05961, 2018.  
Intel Corp. Intel Software Guard Extensions Evaluation SDK. https://software.intel.com/en-us/sgx-sdk, 2015.  
Intel Corp. Intel software guard extensions (sgx) SW development guidance for potential bounds check bypass (CVE-2017-5753) side channel exploits. https://software.intel.com/sites/default/files/180204_SGX SDK Developer_Guidance_v1.0.pdf, 2018.  
Chiraag Juvekar, Vinod Vaikuntanathan, and Anantha Chandrakasan. Gazelle: A low latency framework for secure neural network inference. arXiv preprint arXiv:1801.05507, 2018.  
Paul Kocher, Daniel Genkin, Daniel Gruss, Werner Haas, Mike Hamburg, Moritz Lipp, Stefan Mangard, Thomas Prescher, Michael Schwarz, and Yuval Yarom. Spectre attacks: Exploiting speculative execution. arXiv preprint arXiv:1801.01203, 2018.  
Sangho Lee, Ming-Wei Shih, Prasun Gera, Taesoo Kim, Hyesoon Kim, and Marcus Peinado. Inferring fine-grained control flow inside SGX enclaves with branch shadowing. In USENIX Security Symposium, pp. 16-18, 2017.  
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C Berg. SSD: Single shot multibox detector. In European Conference on Computer Vision (ECCV), pp. 21-37. Springer, 2016.  
Ross M McConnell, Kurt Mehlhorn, Stefan Näher, and Pascal Schweitzer. Certifying algorithms. Computer Science Review, 5(2):119-161, 2011.  
Frank McKeen, Ilya Alex, Alex Berenzon, Carlos Rozas, Hisham Shafi, Vedvyas Shanbhogue, and Uday Savagaonkar. Innovative instructions and software model for isolated execution. In International Workshop on Hardware and Architectural Support for Security and Privacy (HASP), 2013.  
Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaev, Ganesh Venkatesh, et al. Mixed precision training. In International Conference on Learning Representations (ICLR), 2018.  
Ahmad Moghimi, Gorka Irazoqui, and Thomas Eisenbarth. Cachezoom: How SGX amplifies the power of cache attacks. In International Conference on Cryptographic Hardware and Embedded Systems, pp. 69-90. Springer, 2017.  
Payman Mohassel and Yupeng Zhang. SecureML: A system for scalable privacy-preserving machine learning. In IEEE Symposium on Security and Privacy, pp. 19-38. IEEE, 2017.  
Olga Ohrimenko, Felix Schuster, Cdric Fournet, Aastha Mehta, Sebastian Nowozin, Kapil Vaswani, and Manuel Costa. Oblivious multi-party machine learning on trusted processors. In USENIX Security Symposium, 2016.  
Meni Orenbach, Pavel Lifshits, Marina Minkin, and Mark Silberstein. Eleos: Exitless os services for sx enclaves. In Proceedings of the Twelfth European Conference on Computer Systems, pp. 238-253. ACM, 2017.  
Rafael Pass, Elaine Shi, and Florian Tramèr. Formal abstractions for attested execution secure processors. In EUROCRYPT'17, 2017.

Tao Sheng, Chen Feng, Shaojie Zhuo, Xiaopeng Zhang, Liang Shen, and Mickey Aleksic. A quantization-friendly separable convolution for mobilenets. arXiv preprint arXiv:1803.08607, 2018.  
Ming-Wei Shih, Sangho Lee, Taesoo Kim, and Marcus Peinado. T-SGX: Eradicating controlled-channel attacks against enclave programs. In Network and Distributed System Security Symposium (NDSS), 2017.  
Shweta Shinde, Zheng Leong Chua, Viswesh Narayanan, and Prateek Saxena. Preventing page faults from telling your secrets. In ACM Asia Conference on Computer and Communications Security (ASIACCS), pp. 317-328. ACM, 2016.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Ion Stoica, Dawn Song, Raluca Ada Popa, David Patterson, Michael W Mahoney, Randy Katz, Anthony D Joseph, Michael Jordan, Joseph M Hellerstein, Joseph E Gonzalez, et al. A Berkeley view of systems challenges for AI. arXiv preprint arXiv:1712.05855, 2017.  
Pramod Subramanyan, Rohit Sinha, Ilia Lebedev, Srinivas Devadas, and Sanjit A Seshia. A formal foundation for secure remote execution of enclaves. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 2435-2450. ACM, 2017.  
Justin Thaler. Time-optimal interactive proofs for circuit evaluation. In Advances in Cryptology-CRYPTO 2013, pp. 71-89. Springer, 2013.  
Florian Tramér, Fan Zhang, Huang Lin, Jean-Pierre Hubaux, Ari Juels, and Elaine Shi. Sealed-Glass Proofs: Using transparent enclaves to prove and sell knowledge. In IEEE European Symposium on Security and Privacy, 2017.  
Jo Van Bulck, Nico Weichbrodt, Rüdiger Kapitz, Frank Piessens, and Raoul Strackx. Telling your secrets without page faults: Stealhy page table-based attacks on enclave execution. In USENIX Security Symposium, 2017.  
Jo Van Bulck, Marina Minkin, Ofir Weisse, Daniel Genkin, Baris Kasikci, Frank Piessens, Mark Silberstein, Thomas F. Wenisch, Yuval Yarom, and Raoul Strackx. Foreshadow: Extracting the keys to the Intel SGX kingdom with transient out-of-order execution. In Proceedings of the 27th USENIX Security Symposium, 2018.  
Riad S Wahby, Max Howald, Siddharth Garg, Abhi Shelat, and Michael Walfish. Verifiable ASICs. In IEEE Symposium on Security and Privacy, pp. 759-778. IEEE, 2016.  
Riad S Wahby, Ye Ji, Andrew J Blumberg, Abhi Shelat, Justin Thaler, Michael Walfish, and Thomas Wies. Full accounting for verifiable outsourcing. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 2071-2086. ACM, 2017.  
Yuanzhong Xu, Weidong Cui, and Marcus Peinado. Controlled-channel attacks: Deterministic side channels for untrusted operating systems. In S&P'15, pp. 640-656. IEEE, 2015.  
Yupeng Zhang, Charalampos Papamanthou, and Jonathan Katz. Alitheia: Towards practical verifiable graph processing. In Proceedings of the 2014 ACM SIGSAC Conference on Computer and Communications Security, pp. 856-867. ACM, 2014.
