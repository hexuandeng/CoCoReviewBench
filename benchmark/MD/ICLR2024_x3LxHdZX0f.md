# PUMA: SECURE INFERENCE OF LLAMA-7B IN FIVE MINUTES

Anonymous authors

Paper under double-blind review

# ABSTRACT

With ChatGPT as a representative, tons of companies have begun to provide services based on large Transformers models. However, using such a service inevitably leak users' prompts to the model provider. Previous studies have studied secure inference for Transformer models using secure multiparty computation (MPC), where model parameters and clients' prompts are kept secret. Despite this, these frameworks are still limited in terms of model performance, efficiency, and deployment. To address these limitations, we propose framework PUMA to enable fast and secure Transformer model inference. Our framework designs high quality approximations for expensive functions such as GeLU and softmax, and significantly reduce the cost of secure inference while preserving the model performance. Additionally, we design secure Embedding and LayerNorm procedures that faithfully implement the desired functionality without undermining the Transformer architecture. PUMA is about  $2 \times$  faster than the state-of-the-art framework MPCFORMER(ICLR 2023) and has similar accuracy as plaintext models without fine-tuning (which the previous works failed to achieve). PUMA can even evaluate LLaMA-7B in around 5 minutes to generate 1 token. To our best knowledge, this is the first time that a model with such a parameter size is able to be evaluated under MPC.

# 1 INTRODUCTION

Pre-trained Transformer models (Vaswani et al., 2017) have attracted much attention for their high performance in practical tasks (Radford & Narasimhan, 2018; Zhuge et al., 2021) and been widely in Deep Learning as a Service (DLaaS) paradigm (Soifer et al., 2019). However, these services can raise privacy concerns, such as in the case of ChatGPT (Brown et al., 2020), which requires either users to reveal their private prompts to the service provider or the service provider to release their proprietary trained weights to users.

One solution to address the privacy concerns of Transformer models service is Secure Multi-Party Computation (MPC) (Shamir, 1979; Yao, 1986; Goldreich et al., 1987), which can keep data and model weights private during inference. (Hao et al., 2022; Li et al., 2023; Akimoto et al., 2023; Liang et al., 2023; Liu & Liu, 2023) have proposed various ways to support secure Transformer models inference, but these approaches still have one or several of the following drawbacks:

High inference cost. Non-linear functions like GeLU and softmax are challenge to design in MPC. (Hao et al., 2022) computes these non-linear functions in a faithful way. e.g., they design GeLU using tanh based on general MPC exponentiation method proposed by (Rathee et al., 2021). But these general methods are quite expensive in terms of computation and communication, and only tested under small bitwidth (e.g. below 32).

Retraining required. To reduce the cost of non-linear functions, several works (Li et al., 2023; Akimoto et al., 2023; Liu & Liu, 2023) suggested to approximate GeLU and softmax using simpler functions like ReLU and quadratics. These functions are up to an order of magnitude cheaper in MPC, but would introduce utility loss to the Transformer model. As a result, they require an extra step of model retraining (fine-tuning). However, retraining is unfriendly for data-limited participants, and might not achieve satisfactory performance (Kumar et al., 2022).

Incompatible architectures. (Li et al., 2023; Liang et al., 2023) proposed to modify the architecture of Transformer models to further accelerate secure inference, e.g., decompose the embedding proce

dure or reorganize the linear layers. Worsely, (Li et al., 2023) does not support secure LayerNorm and simulated the costs using BatchNorm, resulting in incorrect secure inference results. These modifications are in conflicts with existing plaintext Transformer systems, and would lead to deployment obstacles.

To summarize, in the field of MPC Transformer inference, achieving both model performance and efficiency is challenging, and people may ask the following question:

Could pre-trained large transformer models be securely and efficiently evaluated with similar accuracy as in plaintext, without further retraining?

To address this challenge, we propose the PUMA framework, which is a fast and accurate end-to-end secure Transformer inference framework. Our contributions can be summarized as follows:

- New Approximations for Non-linear Functions. We propose more accurate and faster approximations for the expensive non-linear functions (e.g., GeLU and softmax) in Transformer models. Different from existing works, we design the approximations based on the specialized properties of these non-linear functions to achieve both accuracy and efficiency.  
- Faster and More Accurate Secure Inference. We make extensive experiments on 6 transformer models and 4 datasets, the results show that PUMA's precision is similar to plaintext ones' and is about  $2 \times$  faster than MPCFORMER (note that MPCFORMER does not achieve similar precision as PUMA). PUMA can even evaluate LLaMA-7B in around 5 minutes to generate one word. To our best knowledge, this is the first time that such a large language model is able to be evaluated under MPC.  
- End-to-End Framework compatible with plaintext. We design and implement all the layers required by Transformer (including the Embedding and LayerNorm layers that are missing in other works) in MPC. This allows us to load and securely evaluate the pre-trained plaintext Transformer models (e.g. downloaded from Hugging face) easily. To our best knowledge, PUMA is the first open-sourced MPC solution<sup>1</sup> that supports accurate inference of pre-trained Transformer models without further modifications such as re-training.

Organization. We summarize the related work in  $\S 2$  and present the background in  $\S 3$ . We give PUMA's high-level view and concrete design in  $\S 4$ . We analyze the experimental results in  $\S 5$  and conclude this work in  $\S 6$ .

# 2 RELATED WORK

Secure Multiparty Computation (MPC) (Yao, 1986; Goldreich et al., 1987) enables distrusted parties to jointly compute a function while keeping their inputs private, and secure deep learning inference using MPC has gained much attention due its high privacy protection. These works operate in a variety of models and architectures, including two-party setting (Mohassel & Zhang, 2017; Liu et al., 2017; Mishra et al., 2020; Huang et al., 2022; Patra et al., 2021; Rathee et al., 2020), three-party setting (Wagh et al., 2019; Mohassel & Rindal, 2018; Wagh et al., 2020; Kumar et al., 2019; Patra & Suresh, 2020; Tan et al., 2021; Dong et al., 2023), or four-party setting (Byali et al., 2020; Dalskov et al., 2021). However, most of these approaches only consider secure inference of convolutional/deep neural networks, and cannot be directly extended to support Transformer models. Recently several research works (Hao et al., 2022; Li et al., 2023; Akimoto et al., 2023; Liang et al., 2023; Liu & Liu, 2023) have proposed MPC-based secure inference solutions for Transformer models, but these approaches still have limitations in terms of model performance, efficiency, and deployment. Among these works, MPCFORMER (Li et al., 2023) is the only one that have been open-sourced, it is based on CrypTen (Knott et al., 2021) which is a three-party framework that uses a non-colluding third party to produce correlated randomness for the client and server. Also their three-party model with non-colluding assumption has the highest concrete efficiency among different MPC settings. So we mainly compare our proposed framework PUMA with MPCFORMER under the same three-party setting.

# 3 BACKGROUND

# 3.1 NOTATIONS

The main used notations are as follows:  $P_{i}$  represents the  $i$ -th computing party,  $i \in \{0,1,2\}$ . The uppercase bold letter  $\mathbf{X}$  is used for matrices, and the lowercase bold letter  $\mathbf{x}$  denotes vectors.  $\mathbf{x}[i]$  denotes the  $i$ -th element of vector  $\mathbf{x}$ , while lowercase letter  $x$  is used for scalar values.  $\mathbb{Z}_{2^{\ell}}$  denotes the discrete ring modulo  $2^{\ell}$ ,  $\mathbb{R}$  denotes real numbers.  $[[\cdot]]$  is used for 2-out-of-3 replicated secret sharing (Araki et al., 2016; Mohassel & Rindal, 2018).

# 3.2 TRANSFORMER MODEL

Transformer models have achieved remarkable success in language understanding (Radford & Narasimhan, 2018; Devlin et al., 2019; Yang et al., 2019; Touvron et al., 2023), vision understanding (Zhuge et al., 2021; Dong et al., 2022; Chen et al., 2021), and etc. Two popular variants are Bert (Bidirectional Encoder Representations from Transformers) (Devlin et al., 2019) and GPT (Generative Pre-Trained models) (Radford & Narasimhan, 2018). A Transformer model (Vaswani et al., 2017) mainly consists of Embedding, Attention, Feed-Forward Network, and LayerNorm sub-layers:

Attention. Given inputs  $(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ , the Attention function is computed as Attention( $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ ) =  $\text{softmax}(\mathbf{Q} \cdot \mathbf{K}^{\top} + \mathbf{M}) \cdot \mathbf{V}$ , where  $\mathbf{M}$  can be viewed as a bias matrix. Besides, (Vaswani et al., 2017) proposed Multi-Head Attention to jointly attend to information from different representation subspaces at different positions.

Feed-Forward Network (FFN). FFN is applied to each position separately and identically. This consists of two linear transformations with an activation in between, and the most commonly used activation function is GeLU. Given input  $\mathbf{x}$  and parameters  $\{\mathbf{W}_1,\mathbf{b}_1,\mathbf{W}_2,\mathbf{b}_2\}$ , FFN can be formalized as  $\mathrm{FFN}(\mathbf{x}) = \mathbf{W}_2\mathrm{GeLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ . Note that the parameters of linear transformations are different from layer to layer.

LayerNorm. Given vector \(\mathbf{x} \in \mathbb{R}^n\), LayerNorm is defined as: LayerNorm(\(\mathbf{x}\))[i] = \gamma \cdot \frac{\mathbf{x}[i] - \mu}{\sqrt{\sigma}} + \beta\), where \((\gamma, \beta)\) are trained parameters, \(\mu = \frac{\sum_{i=1}^{n} \mathbf{x}[i]}{n}\), and \(\sigma = \sum_{i=1}^{n} (\mathbf{x}[i] - \mu)^2\).

# 3.3 2-OUT-OF-3 REPLICATED SECRET SHARING

A secret value  $x \in \mathbb{Z}_{2^{\ell}}$  is shared by three random values  $x_0, x_1, x_2 \in \mathbb{Z}_{2^{\ell}}$  with  $x = x_0 + x_1 + x_2$  (mod  $2^{\ell}$ ). In 2-out-of-3 replicated secret sharing (denoted as  $[[\cdot]]$ -sharing), party  $P_i$  gets  $[[x]]_i = (x_i, x_{i+1})$ . Without special declaration, we compute in  $\mathbb{Z}_{2^{\ell}}$  and omit (mod  $2^{\ell}$ ) for brevity. In the case of  $\ell > 1$  (e.g.,  $\ell = 64$ ) which support arithmetic operations (e.g., +, -, and ·), we refer to this type as Arithmetic Sharing and use notation  $[[\cdot]]$ . Boolean Sharing  $([\cdot]]^{\mathsf{B}}$  refers to  $\ell = 1$  where (+, -) and · are respectively replaced by bit-wise  $\oplus$  and  $\wedge$ .

Addition. Let  $(c_{1}, c_{2}, c_{3})$  be public constants, and  $([[x]], [[y]])$  be two secret-shared values. Then,  $\llbracket c_{1}x + c_{2}y + c_{3}\rrbracket$  can be computed as  $(c_{1}x_{0} + c_{2}y_{0} + c_{3}, c_{1}x_{1} + c_{2}y_{1}, c_{1}x_{2} + c_{2}y_{2})$  where  $P_{i}$  can compute its share locally. When  $(c_{1} = 1, c_{2} = 1, c_{3} = 0)$ , we get  $\llbracket x + y\rrbracket$ .

Multiplication. In secure multiplication protocol  $\Pi_{\mathsf{Mul}}$ , given two shared values  $[x]$  and  $[y]$ , parties follow steps: i) First,  $P_{i}$  computes  $z_{i} = x_{i}y_{i} + x_{i + 1}y_{i} + x_{i}y_{i + 1}$  locally, ii) Parties then perform re-sharing by letting  $P_{i}$  sends  $z_{i}' = \alpha_{i} + z_{i}$  to  $P_{i - 1}$ , where  $\alpha_0 + \alpha_1 + \alpha_2 = 0$  ( $P_{i}$  can generate  $\alpha_{i}$  in the setup phase as Mohassel & Rindal (2018)). iii) Finally,  $\{(z_0', z_1'), (z_1', z_2'), (z_2', z_0')\}$  form  $[x \cdot y]$ .

Underlying Protocols. In addition to addition and multiplication, PUMA relies on several other underlying protocols: boolean-arithmetic multiplication  $(\Pi_{Mul_{BA}})$ , square  $\Pi_{Square}$ , equality test  $(\Pi_{Eq})$ , less than  $(\Pi_{LT})$ , reciprocal  $(\Pi_{Recip})$ , maximum  $(\Pi_{Max})$ , and reciprocal of square root  $(\Pi_{rSqrt})$ , from the state-of-the-art works. We employ them in a black-box manner, and only enumerate the inputs and outputs of these protocols as follows:

-  $\llbracket z\rrbracket = \Pi_{\mathrm{Mul}_{\mathrm{BA}}}([\llbracket b\rrbracket^{\mathsf{B}},[\llbracket x\rrbracket])$  , s.t.  $z = b\cdot x$

-  $\llbracket z\rrbracket^{\mathsf{B}} = \Pi_{\mathrm{LT}}(\llbracket x\rrbracket ,\llbracket y\rrbracket)$ , s.t.  $z = 1\{x < y\}$

-  $\llbracket z\rrbracket = \Pi_{\mathrm{Square}}(\llbracket x\rrbracket)$ , s.t.  $z = x^{2}$

-  $\llbracket z\rrbracket = \Pi_{\mathrm{Recip}}(\llbracket x\rrbracket)$ , s.t.  $z = 1 / x$

-  $\llbracket z\rrbracket^{\mathsf{B}} = \Pi_{\mathbf{Eq}}(\llbracket x\rrbracket ,\llbracket y\rrbracket)$ , s.t.  $z = 1\{x = y\}$

-  $\llbracket z\rrbracket = \Pi_{\mathrm{rSqrt}}([\llbracket x\rrbracket ])$  ,s.t.  $z = 1 / \sqrt{x}$

$$
\llbracket z \rrbracket = \Pi_ {\operatorname {M a x}} (\llbracket \mathbf {x} \rrbracket), \text {s . t .} z = \operatorname {m a x i m u m} (\mathbf {x})
$$

$1\{e\}$  returns 1 that when condition  $e$  is true, and 0 otherwise. For detailed protocol constructions, please refer to (Mohassel & Rindal, 2018; Lu et al., 2020; Keller, 2020).

Fixed-Point Representation & Truncation. Real numbers have to be encoded into fixed-point numbers before represented in finite rings/fields. To avoid overflow,  $\Pi_{\mathrm{Trunc}}^f$  has to be used after each fixed-point multiplication to truncate the least  $f$  bits securely. For simpler description, we include  $\Pi_{\mathrm{Trunc}}^f$  in  $\Pi_{\mathrm{Mul}}$  and  $\Pi_{\mathrm{Square}}$  by default and and do not explicitly mention it in our protocol designs.

The above operations can be easily extended to vectors and matrices, and we use the same notation for vector and matrix operations for simplicity. For more details, please refer to (Mohassel & Rindal, 2018; Wagh et al., 2020).

Threat Model. Following previous works (Mohassel & Rindal, 2018; Li et al., 2023), PUMA is secure against a semi-honest adversary that corrupts no more than one of the three computing parties. Semi-honest means such an adversary will follow the protocol specifications, but may try to learn other's private information during the protocol. Please note that PUMA cannot defend against attacks based on inference results, and the mitigation of such attacks (e.g., differential privacy (Abadi et al., 2016)) falls outside the scope of this study.

# 4 SECURE DESIGN OF PUMA

In this section, we first present an overview of PUMA, and present the protocols for secure GeLU, softmax, embedding, and LayerNorm used by PUMA. Note that the linear layers such as matrix multiplication are straightforward in replicated secret sharing, so we mainly describe our protocols for non-linear layers in this manuscript.

# 4.1 OVERVIEW OF PUMA

To achieve secure inference of Transformer models, PUMA defines three kinds of roles: one model owner, one client, and three computing parties. The model owner and the client provide their models or inputs to the computing parties (i.e.,  $P_0$ ,  $P_1$ , and  $P_2$ ) in a secret-shared form, then the computing parties execute the MPC protocols and send the results back to the client. Note that the model owner and client can also act as one of the computing party, we describe them separately for generality. e.g., when the model owner acts as  $P_0$ , the client acts as  $P_1$ , a third-party dealer acts as  $P_2$ , the system model becomes the same with MPCFORMER (Li et al., 2023).

During the secure inference process, a key invariant is maintained: For any layer, the computing parties always start with 2-out-of-3 replicated secret shares of the previous layer's output and the model weights, and end with 2-out-of-3 replicated secret shares of this layer's output. As the shares do not leak any information to each party, this ensures that the layers can be sequentially combined for arbitrary depths to obtain a secure computation scheme for any Transformer-based model.

# 4.2 PROTOCOL FOR SECURE GELU

Most of the current approaches view the GeLU function as a composition of smaller functions and try to optimize each piece of them, making them to miss the chance of optimizing the private GeLU as a whole. Given the GeLU function:

$$
\begin{array}{l} \operatorname {G e L U} (x) = \frac {x}{2} \cdot \left(1 + \tanh  \left(\sqrt {\frac {2}{\pi}} \cdot \left(x + 0. 0 4 4 7 1 5 \cdot x ^ {3}\right)\right)\right), \tag {1} \\ \approx x \cdot \operatorname {s i g m o i d} \left(0. 0 7 1 3 5 5 \cdot x ^ {3} + 1. 5 9 5 7 6 9 \cdot x\right) \\ \end{array}
$$

these approaches (Hao et al., 2022; Wang et al., 2022) focus either on designing approximate protocols for function tanh or using existing general MPC protocols of exponentiation and reciprocal for sigmoid.

However, none of current approaches have utilized the fact that  $\mathrm{GeLU}$  function is almost linear on the two sides (i.e.,  $\mathrm{GeLU}(x) \approx 0$  for  $x < -4$  and  $\mathrm{GeLU}(x) \approx x$  for  $x > 3$ ). Within the short

# Algorithm 1 Secure GeLU Protocol  $\Pi_{\mathrm{GeLU}}$

Input:  $P_{i}$  holds the 2-out-of-3 replicate secret share  $\llbracket x\rrbracket_{i}$  for  $i\in \{0,1,2\}$

Output:  $P_{i}$  gets the 2-out-of-3 replicate secret share  $\llbracket y\rrbracket_{i}$  for  $i\in \{0,1,2\}$ , where  $y = \operatorname {GeL}U(x)$ .

1:  $P_0, P_1$ , and  $P_2$  jointly compute

$$
\llbracket b _ {0} \rrbracket^ {\mathsf {B}} = \Pi_ {\mathsf {L T}} (\llbracket x \rrbracket , - 4), \quad \triangleright b _ {0} = 1 \{x <   - 4 \}
$$

$$
\llbracket b _ {1} \rrbracket^ {\mathbf {B}} = \Pi_ {\mathrm {L T}} (\llbracket x \rrbracket , - 1. 9 5), \quad \triangleright b _ {1} = 1 \{x <   - 1. 9 5 \}
$$

$$
\llbracket b _ {2} \rrbracket^ {\mathsf {B}} = \Pi_ {\mathsf {L T}} (3, \llbracket x \rrbracket), \quad \triangleright b _ {2} = 1 \{3 <   x \}
$$

and compute  $\llbracket z_0\rrbracket^{\mathsf{B}} = \llbracket b_0\rrbracket^{\mathsf{B}}\oplus \llbracket b_1\rrbracket^{\mathsf{B}},\llbracket z_1\rrbracket^{\mathsf{B}} = \llbracket b_1\rrbracket^{\mathsf{B}}\oplus \llbracket b_2\rrbracket^{\mathsf{B}}\oplus 1,$  and  $\llbracket z_2\rrbracket^{\mathsf{B}} = \llbracket b_2\rrbracket^{\mathsf{B}}$  . Note that  $z_{0} = 1\{-4\leq x <   - 1.95\}$ $z_{1} = 1\{-1.95\leq x\leq 3\}$  , and  $z_{2} = 1\{x > 3\}$

2: Jointly compute  $\llbracket x^2\rrbracket = \Pi_{\mathrm{Square}}(\llbracket x\rrbracket),\llbracket x^3\rrbracket = \Pi_{\mathrm{Mul}}(\llbracket x\rrbracket,\llbracket x^2\rrbracket),\llbracket x^4\rrbracket = \Pi_{\mathrm{Square}}(\llbracket x^2\rrbracket)$ , and  $\llbracket x^6\rrbracket = \Pi_{\mathrm{Square}}(\llbracket x^3\rrbracket)$ .  
3: Computing polynomials  $\llbracket F_0(x)\rrbracket$  and  $\llbracket F_1(x)\rrbracket$  based on  $\{\llbracket x\rrbracket ,\llbracket x^2\rrbracket ,\llbracket x^3\rrbracket ,\llbracket x^4\rrbracket ,\llbracket x^6\rrbracket \}$  as equation (2) securely.  
4: return  $\llbracket y\rrbracket = \Pi_{\mathrm{Mul}_{\mathsf{BA}}}(\llbracket z_0\rrbracket^{\mathsf{B}},\llbracket F_0(x)\rrbracket) + \Pi_{\mathrm{Mul}_{\mathsf{BA}}}(\llbracket z_1\rrbracket^{\mathsf{B}},\llbracket F_1(x)\rrbracket) + \Pi_{\mathrm{Mul}_{\mathsf{BA}}}(\llbracket z_2\rrbracket^{\mathsf{B}},\llbracket x\rrbrack)$ .

interval  $[-4,3]$  of GeLU, we suggest a piece-wise approximation of low-degree polynomials is a more efficient and easy-to-implement choice for its secure protocol. Concretely, our piece-wise low-degree polynomials are shown as equation (2):

$$
\operatorname {G e L U} (x) = \left\{ \begin{array}{l l} 0, & x <   - 4 \\ F _ {0} (x), & - 4 \leq x <   - 1. 9 5 \\ F _ {1} (x), & - 1. 9 5 \leq x \leq 3 \\ x, & x > 3 \end{array} , \right. \tag {2}
$$

where polynomials  $F_0()$  and  $F_1()$  are computed by library numpy.  $ployfit^2$  as equation (3). Surprisingly, the above simple poly fit works very well and our max error  $< 0.01403$ , median error  $< 4.41e - 05$ , and mean error  $< 0.00168$ .

$$
\left\{ \begin{array}{l l} F _ {0} (x) & = - 0. 0 1 1 0 3 4 1 3 4 0 3 0 6 1 5 7 2 8 x ^ {3} - 0. 1 1 8 0 7 6 1 2 9 5 1 1 8 1 9 5 3 x ^ {2} \\ & - 0. 4 2 2 2 6 5 8 1 1 5 1 9 8 3 8 6 6 x - 0. 5 0 5 4 0 3 1 1 9 9 7 0 8 1 7 4 \\ F _ {1} (x) & = 0. 0 0 1 8 0 6 7 4 6 2 6 0 6 1 4 1 1 8 7 x ^ {6} - 0. 0 3 7 6 8 8 2 0 0 3 6 5 9 0 4 2 3 6 x ^ {4} \\ & + 0. 3 6 0 3 2 9 2 6 9 2 7 8 9 6 2 9 x ^ {2} + 0. 5 x + 0. 0 0 8 5 2 6 3 2 1 5 4 1 0 3 8 0 8 4 \end{array} \right. \tag {3}
$$

Formally, given secret input  $\llbracket x\rrbracket$  , our secure GeLU protocol  $\Pi_{\mathrm{GeLU}}$  is constructed as algorithm 1.

# 4.3 PROTOCOL FOR SECURE SOFTMAX

In the function Attention( $\mathbf{Q}$ ,  $\mathbf{K}$ ,  $\mathbf{V}$ ) =  $\mathrm{softmax}(\mathbf{Q} \cdot \mathbf{K}^{\top} + \mathbf{M}) \cdot \mathbf{V}$ , the key challenge is computing function softmax. For the sake of numerical stability, the softmax function is computed as

$$
\operatorname {s o f t m a x} (\mathbf {x}) [ i ] = \frac {\exp (\mathbf {x} [ i ] - \bar {x} - \epsilon)}{\sum_ {i} \exp (\mathbf {x} [ i ] - \bar {x} - \epsilon)}, \tag {4}
$$

where  $\bar{x}$  is the maximum element of the input vector  $\mathbf{x}$ . For the normal plaintext softmax,  $\epsilon = 0$ . For a two-dimension matrix, we apply equation (4) to each of its row vector.

Formally, our detailed secure protocol  $\Pi_{\mathrm{softmax}}$  is illustrated in algorithm 2, where we propose two optimizations:

- For the first optimization, we set  $\epsilon$  in equation 4 to a tiny and positive value, e.g.,  $\epsilon = 10^{-6}$ , so that the inputs to exponentiation in equation 4 are all negative. We exploit the negative operands for acceleration. Particularly, we compute the exponentiation using the Taylor series (Tan et al., 2021) with a simple clipping

$$
\operatorname {n e g E x p} (x) = \left\{ \begin{array}{l l} 0, & x <   T _ {\exp} \\ \left(1 + \frac {x}{2 ^ {t}}\right) ^ {2 ^ {t}}, & x \in [ T _ {\exp}, 0 ]. \end{array} \right. \tag {5}
$$

Algorithm 2 Secure softmax Protocol  $\Pi_{\mathrm{softmax}}$  
Input:  $P_{i}$  holds the replicate secret share  $\mathbb{[x]_i}$  for  $i\in \{0,1,2\}$ , and  $\mathbf{x}$  is a vector of size  $n$ .  
Output:  $P_{i}$  gets the replicate secret share  $\mathbb{[y]_i}$  for  $i\in \{0,1,2\}$ , where  $\mathbf{y} = \mathrm{softmax}(\mathbf{x})$ .  
1:  $P_{0}, P_{1}$ , and  $P_{2}$  jointly compute  $\mathbb{[b]}^{\mathsf{B}} = \Pi_{\mathrm{LT}}(T_{\mathrm{exp}}, \mathbb{[x]})$  and the maximum  $\mathbb{[x]} = \Pi_{\mathrm{Max}}(\mathbb{[x]})$ .  
2: Parties locally compute  $\mathbb{[x]} = \mathbb{[x]} - \mathbb{[x]} - \epsilon$ , and jointly compute  $\mathbb{[z_0]} = 1 + \Pi_{\mathrm{Trunc}}^t (\mathbb{[x]})$ .  
3: for  $j = 1,2,\ldots,t$  do  
4:  $\mathbb{[z_j]} = \Pi_{\mathrm{Square}}(\mathbb{[z_{j-1}]})$ .  
5: end for  
6: Parties locally compute  $\mathbb{[z]} = \sum_{i=1}^{n} \mathbb{[z[i]]}$  and jointly compute  $\mathbb{[1/z]} = \Pi_{\mathrm{Recip}}(\mathbb{[z]})$ .  
7: Parties jointly compute  $\mathbb{[z/z]} = \Pi_{\mathrm{Mul}}(\mathbb{[z]}, [1/z])$   
8: return  $\mathbb{[y]} = \Pi_{\mathrm{Mul}_{\mathrm{BA}}}(\mathbb{[b]}^{\mathsf{B}}, [\mathbf{z}/z])$ .

Indeed, we apply the less-than for the branch  $x < T_{\mathrm{exp}}$ . The division by  $2^t$  can be achieved using  $\Pi_{\mathrm{Trunc}}^t$  since the input is already negative. Also, we can compute the power-of  $2^t$  using  $t$ -step sequences of square function  $\Pi_{\mathrm{square}}$  and  $\Pi_{\mathrm{Trunc}}^f$ . Suppose our MPC program uses 18-bit fixed-point precision. Then we set  $T_{\mathrm{exp}} = -14$  given  $\exp(-14) < 2^{-18}$ , and empirically set  $t = 5$ .

- Our second optimization is to reduce the number of divisions, which ultimately saves computation and communication costs. To achieve this, for a vector  $\mathbf{x}$  of size  $n$ , we have replaced the operation  $\operatorname{Div}(\mathbf{x}, \operatorname{Broadcast}(y))$  with  $\mathbf{x} \cdot \operatorname{Broadcast}\left(\frac{1}{y}\right)$ , where  $y = \sum_{i=1}^{n} \mathbf{x}[i]$ . By making this replacement, we effectively reduce  $n$  divisions to just one reciprocal operation and  $n$  multiplications. This optimization is particularly beneficial in the case of the softmax operation. The  $\frac{1}{y}$  in the softmax operation is still large enough to maintain sufficient accuracy under fixed-point values. As a result, this optimization can significantly reduce the computational and communication costs while still providing accurate results.

# 4.4 PROTOCOL FOR SECURE EMBEDDING

The current secure embedding procedure described in (Li et al., 2023) necessitates the client to generate a one-hot vector using the token id locally. This deviates from a plaintext Transformer workflow where the one-hot vector is generated inside the model. As a result, they have to carefully strip off the one-hot step from the pre-trained models, and add the step to the client side, which could be an obstacle for deployment.

To address this issue, we propose a secure embedding design as follows. Assuming that the token  $\mathrm{id} \in [n]$  and all embedding vectors are denoted by  $\mathbb{E} = (\mathbf{e}_1^T, \mathbf{e}_2^T, \ldots, \mathbf{e}_n^T)$ , the embedding can be formulated as  $\mathbf{e}_{\mathrm{id}} = \mathbf{E}[\mathrm{id}]$ . Given  $(\mathrm{id}, \mathbb{E})$  are in secret-shared fashion, our secure embedding protocol  $\Pi_{\mathrm{Embed}}$  works as follows:

- The computing parties securely compute the one-hot vector  $\mathbb{[}\mathbf{o}\mathbb{]^{B}}$  after receiving  $\mathbb{[}\mathrm{id}\mathbb{]}$  from the client. Specifically,  $\mathbb{[}\mathbf{o}[i]\mathbb{]}^{\mathrm{B}} = \Pi_{\mathrm{Eq}}(i,\mathbb{[}\mathrm{id}])$  for  $i\in [n]$ .  
- The parties can compute the embedded vector via  $\llbracket \mathbf{e}_{\mathrm{id}}\rrbracket = \Pi_{\mathsf{Mul}_{\mathsf{BA}}}(\llbracket \mathbb{E}\rrbracket ,\llbracket \mathbf{o}\rrbracket^{\mathsf{B}})$ , where does not require secure truncation.

In this way, our  $\Pi_{\text{Embed}}$  does not require explicit modification of the workflow of plaintext Transformer models, at the cost of more  $\Pi_{\text{Eq}}$  and  $\Pi_{\text{Mul}_{\text{BA}}}^{\text{operations}}$ .

# 4.5 PROTOCOL FOR SECURE LAYERNORM

Recall that given a vector  $\mathbf{x}$  of size  $n$ , LayerNorm  $(\mathbf{x})[i] = \gamma \cdot \frac{\mathbf{x}[i] - \mu}{\sqrt{\sigma}} + \beta$ , where  $(\gamma, \beta)$  are trained parameters,  $\mu = \frac{\sum_{i=1}^{n} \mathbf{x}[i]}{n}$ , and  $\sigma = \sum_{i=1}^{n} (\mathbf{x}[i] - \mu)^2$ . In MPC, the key challenge is the evaluation of the divide-square-root  $\frac{\mathbf{x}[i] - \mu}{\sqrt{\sigma}}$  formula. To securely evaluate this formula, CropyTen sequentially executes the MPC protocols of square-root, reciprocal, and multiplication. However, we observe that  $\frac{\mathbf{x}[i] - \mu}{\sqrt{\sigma}}$  is equal to  $(\mathbf{x}[i] - \mu) \cdot \sigma^{-1/2}$ . And in the MPC side, the costs of computing the inverse-square-root  $\sigma^{-1/2}$  is similar to that of the square-root operation (Lu et al., 2020). Besides, inspired by the

Algorithm 3 Secure LayerNorm Protocol  $\Pi_{\text{LayerNorm}}$  
Input:  $P_{i}$  holds the replicate secret share  $\mathbb{[x]_i}$  for  $i\in \{0,1,2\}$ , and  $\mathbf{x}$  is a vector of size  $n$ .  
Output:  $P_{i}$  gets the replicate secret share  $\mathbb{[y]_i}$  for  $i\in \{0,1,2\}$ , where  $\mathbf{y} = \text{LayerNorm}(\mathbf{x})$ .  
1:  $P_{0}, P_{1}$ , and  $P_{2}$  compute  $\mathbb{[}\mu \mathbb{]} = \frac{1}{n} \cdot \sum_{i=1}^{n} \mathbb{[}x[i]\mathbb{]}$  and  $\mathbb{[}\sigma \mathbb{]} = \sum_{i=1}^{n} \Pi_{\text{Square}}(\mathbb{[}x] - \mathbb{[}\mu \mathbb{]})[i]$ .  
2: Parties jointly compute  $\mathbb{[}\sigma^{-1/2}\mathbb{]} = \Pi_{r\text{Sqrt}}(\mathbb{[}\sigma\mathbb{])$ .  
3: Parties jointly compute  $\mathbb{[}c\mathbb{]} = \Pi_{Mul}((\mathbb{[}x] - \mathbb{[}\mu\mathbb{]}, \mathbb{[}\sigma^{-1/2}\mathbb{]})$   
4: return  $\mathbb{[}y\mathbb{]} = \Pi_{Mul}(\mathbb{[}\gamma\mathbb{]}, \mathbb{[}c\mathbb{]}) + \mathbb{[}\beta\mathbb{]}$ .

second optimization of  $\S 4.3$ , we can first compute  $\sigma^{-1/2}$  and then  $\mathrm{Broadcast}(\sigma^{-1/2})$  to support fast and secure LayerNorm(x). And our formal protocol  $\Pi_{LayerNorm}$  is shown in algorithm 3.

# 5 EXPERIMENTAL EVALUATIONS

Implementation. We implement PUMA on top of SecretFlow-SPU (Ma et al., 2023) in  $\mathrm{C + + }$  and Python. We encode the data in a fixed-point form under ring  $\mathbb{Z}_{2^{64}}$  with 18-bit fractional part. Our experiments are ran on 3 Alibaba Cloudecs.g7.8xlarge servers with 32 vCPUs and 128GB RAM each. The CPU model is Intel Xeon(Ice Lake) Platinum 8369B CPU @  $2.70\mathrm{GHz}$ . We evaluate PUMA on Ubuntu 20.04.6 LTS with Linux kernel 5.4.0-144-generic. Our bandwidth is about 5Gbps and round trip time is about 1ms.

Models & Datasets. We evaluate PUMA on seven NLP models: Bert-Base, Roberta-Base, and Bert-Large (Devlin et al., 2019); GPT2-Base, GPT2-Medium, and GPT2-Large (Radford & Narasimhan, 2018); and LLaMA-7B (Touvron et al., 2023). We measure the Bert performance for three NLP tasks over the datasets of Corpus of Linguistic Acceptability (CoLA), Recognizing Textual Entailment (RTE), Stanford Question Answering Dataset (QNLI) from GLUE benchmarks (Wang et al., 2019), and GPT2 performance on Wikitext-103 V1 (Merit et al., 2016).

Baseline. We compare PUMA to the most similar prior work MPCFORMER (Li et al., 2023). But for fair comparison, we have the following considerations: i) As MPCFORMER neither supports loading pretrained transformer models nor implements LayerNorm faithfully<sup>3</sup>, we cannot achieve meaningful secure inference results using their framework. Therefore, we compare our performance to that of plaintext (floating-point) to show our precision guarantee. ii) MPCFORMER with Quad approximations requires retraining the modified models. As PUMA does not require retraining, we compare our cost to that of MPCFORMER without Quad approximations. Also, we re-run MPCFORMER in our environment.

# 5.1 PRECISION

We randomly sample 500 instances from the validation split of each dataset within the GLUE benchmark to assess Bert inference performance, and for evaluating GPT2 inference performance, we select 100 samples from the validation split of Wikitext-103 V1. We compare our secure model inference performance to that of plaintext (floating-point) in Table 1 and 2 to show our precision guarantee.

In Table 1, we show the Matthews correlation/accuracy of plaintext and PUMA on the Bert models. We observe that the accuracy achieved by PUMA matches the accuracy of the plaintext Flax code. Specifically, the accuracy difference does not exceed 0.011 over all datasets. Moreover, in Table 2, we also compare our perplexity on dataset Wikitext-103 V1 with the plaintext baseline on GPT2 models. The results are similar and the perplexity differences do not exceed 0.02 over all models.

The above accuracy and perplexity advantages experimentally validate that our protocols are numerically precise.

Table 1: Performance on GLUE benchmark of Bert-Base, Roberta-Base, and Bert-Large on CoLA, RTE, and QNLI. Matthews correlation is reported for CoLA. Accuracy is reported for other datasets.  

<table><tr><td>Model</td><td colspan="3">Bert-Base</td><td colspan="3">Roberta-Base</td><td colspan="3">Bert-Large</td></tr><tr><td>TASK</td><td>CoLA</td><td>RTE</td><td>QNLI</td><td>CoLA</td><td>RTE</td><td>QNLI</td><td>CoLA</td><td>RTE</td><td>QNLI</td></tr><tr><td>Plaintext</td><td>0.616</td><td>0.700</td><td>0.916</td><td>0.629</td><td>0.805</td><td>0.920</td><td>0.686</td><td>0.755</td><td>0.922</td></tr><tr><td>PUMA</td><td>0.613</td><td>0.700</td><td>0.916</td><td>0.618</td><td>0.805</td><td>0.918</td><td>0.690</td><td>0.747</td><td>0.918</td></tr></table>

Table 2: Perplexity of GPT2-Base, GPT2-Medium, and GPT2-Large on Wikitext-103 V1.  

<table><tr><td>Model</td><td>GPT2-Base</td><td>GPT2-Medium</td><td>GPT2-Large</td></tr><tr><td>Plaintext</td><td>16.284</td><td>12.536</td><td>10.142</td></tr><tr><td>PUMA</td><td>16.284</td><td>12.540</td><td>10.161</td></tr></table>

Table 3: Costs of Bert-Base, Roberta-Base, and Bert-Large for one sentence of length 128. Time is in seconds and Communication (Comm. for short) is in GB, which is the same for the following tables.  

<table><tr><td>Model</td><td colspan="2">Bert-Base</td><td colspan="2">Roberta-Base</td><td colspan="2">Bert-Large</td></tr><tr><td>Costs</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td></tr><tr><td>MPCFORMER</td><td>55.320</td><td>12.089</td><td>57.256</td><td>12.373</td><td>141.222</td><td>32.577</td></tr><tr><td>PUMA</td><td>33.913</td><td>10.773</td><td>41.641</td><td>11.463</td><td>73.720</td><td>27.246</td></tr><tr><td>Improv.</td><td>1.631×</td><td>1.122×</td><td>1.375×</td><td>1.079×</td><td>1.916×</td><td>1.195×</td></tr></table>

Table 4: Costs of GPT2-Base, GPT2-Medium, and GPT2-Large. The input sentence is of length 32, all of the costs are for generating 1 token.  

<table><tr><td>Model</td><td colspan="2">GPT2-Base</td><td colspan="2">GPT2-Medium</td><td colspan="2">GPT2-Large</td></tr><tr><td>Costs</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td></tr><tr><td>MPCFORMER</td><td>34.889</td><td>4.999</td><td>73.078</td><td>11.766</td><td>129.095</td><td>22.522</td></tr><tr><td>PUMA</td><td>15.506</td><td>3.774</td><td>30.272</td><td>7.059</td><td>54.154</td><td>11.952</td></tr><tr><td>Improv.</td><td>2.250×</td><td>1.325×</td><td>2.414×</td><td>1.667×</td><td>2.383×</td><td>1.884×</td></tr></table>

Table 5: Costs of Bert-Base and GPT2-Base for different input length (denoted as #Input). The input lengths for Bert-Base and GPT2-Base are respectively {64, 128, 256} and {16, 32, 64}. GPT2-Base generates 1 token.  

<table><tr><td></td><td>#Input</td><td colspan="2">64/16</td><td colspan="2">128/32</td><td colspan="2">256/64</td></tr><tr><td></td><td>Costs</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td></tr><tr><td rowspan="3">Bert</td><td>MPCFORMER</td><td>36.354</td><td>5.707</td><td>55.320</td><td>12.089</td><td>112.453</td><td>29.927</td></tr><tr><td>PUMA</td><td>21.141</td><td>4.881</td><td>33.913</td><td>10.773</td><td>61.210</td><td>26.004</td></tr><tr><td>Improv.</td><td>1.720×</td><td>1.169×</td><td>1.631×</td><td>1.122×</td><td>1.837×</td><td>1.151×</td></tr><tr><td rowspan="3">GPT2</td><td>MPCFORMER</td><td>29.695</td><td>4.011</td><td>34.889</td><td>4.999</td><td>43.344</td><td>7.318</td></tr><tr><td>PUMA</td><td>11.056</td><td>1.875</td><td>15.506</td><td>3.777</td><td>24.860</td><td>7.821</td></tr><tr><td>Improv.</td><td>2.686×</td><td>2.139×</td><td>2.250×</td><td>1.324×</td><td>1.744×</td><td>0.936×</td></tr></table>

# 5.2 INFERENCE COSTS

We compare PUMA's inference cost to that of MPCFORMER in Table 3 and Table 4. The costs are for processing one input sentence: i) For Bert models the input sentence is of length 128. ii) For GPT2 models the input length is 32 and generate 1 new word.

On the 3 Bert models in Table 3, PUMA is  $1.375 \sim 1.916 \times$  faster than MPCFORMER, and is  $1.079 \sim 1.195 \times$  more communication-efficient. For the GPT2 models in Table 4, PUMA is  $2.250 \sim 2.414 \times$  faster than MPCFORMER, and is  $1.325 \sim 1.884 \times$  more communication-efficient.

We observe that PUMA's improvements increase as the model size grows, particularly for the GPT2 models. This trend is because our specialized optimizations are more effective when processing large-scale evaluations.

# 5.3 SCALABILITY

In this subsection, we measure the costs of evaluating PUMA on Bert-Base and GPT2-Base models for batched inputs, varying-length inputs, and varying-length outputs (only for GPT2-Base). We also compare our costs to those of MPCFORMER to demonstrate our improvements.

Input Length Evaluation. Table 5 shows our costs on varying-length inputs, we evaluate Bert-Base on inputs of length \{64, 128, 256\}, and GPT2-Base on inputs of length \{16, 32, 64\}. For Bert-Base, PUMA is  $1.631 \sim 1.837 \times$  faster, and for GPT2-Base, PUMA is  $1.744 \sim 2.686 \times$  faster.

Output Length Evaluation. Fig 1 presents our costs on varying-length outputs for GPT2-Base. Our improvements against MPCFORMER range from  $1.279 \sim 2.700 \times$ .

We observe in Table 5 and Fig 1 that for GPT-2, our efficiency gains decrease with more input/output tokens. This is because PUMA introduces extra one-hot embedding costs (as described in 4.4). We should

![](images/ecdd84bd778fd0eebb3e82fc6553de6ead1cc62f3051381965794c6fb4e8dfa7.jpg)  
Figure 1: Runtime of GPT2-Base for generating different output tokens, the input length is of length 32.

emphasize again that PUMA is compatible with plaintext models, and could achieve a similar accuracy as plaintext models while MPCFORMER could not.

# 5.4 EVALUATING LLAMA-7B IN FIVE MINUTES.

Our protocols are already complete for evaluating any Transformer-based models including LLaMA-7B. Unfortunately, existing serialization libraries such as Protobuf (Varda, 2008) and FlatBuffers (van Oortmerssen, 2014) only support data trunks with size up to 2GB, which is not sufficient for large MPC tasks. To address this problem, we propose an optimization to SecretFlow-SPU. Concretely, the system could automatically divide and serialize overly large secret-shared structures into smaller chunks when communicating or performing I/O operations.

We evaluated the large language model LLaMA-7B using PUMA under 3 Alibaba Cloudecs.r7.32xlarge servers, each has 128 threads and 1TB RAM, with 20GB bandwidth, 0.1ms round-trip-time. As shown in Table 6, PUMA can support secure inference of LLaMA-7B with reasonable costs. For example, given an input sentence of 8 tokens, PUMA can output one token in around 200 seconds with communication costs of 1.794 GB. To our knowledge, this is the first time that LLaMA-7B has been evaluated using MPC. Moreover, PUMA can generate the same tokens exactly as plaintext LLaMA-7B, see Appendix for an example.

Table 6: Costs of the secure inference of LLaMA-7B, #Input denotes the length of input sentence and #Output denotes the number of generated tokens.  

<table><tr><td>(#Input, #Output)</td><td colspan="2">(4,1)</td><td colspan="2">(8,1)</td><td colspan="2">(8,2)</td></tr><tr><td>Costs</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td><td>Time</td><td>Comm.</td></tr><tr><td>PUMA</td><td>122.004</td><td>0.907</td><td>200.473</td><td>1.794</td><td>364.527</td><td>3.857</td></tr></table>

# 6 CONCLUSION

We propose an efficient MPC framework PUMA for secure inference on Transformer models based on replicated secret sharing. To reduce the costs of secure inference, we approximate expensive functions with accurate polynomials and propose secure Embedding and LayerNorm protocols to support end-to-end secure inference. Although the inference cost is still quite high, we successfully make it one step closer to solving users' privacy concerns in Transformer-based DLaaS. We believe that by combining PUMA with quantization methods and hardware accelerations in the future, secure inference of large Transformer models in seconds is no longer impossible.

# REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pp. 308-318, 2016.  
Y. Akimoto, K. Fukuchi, Y. Akimoto, and J. Sakuma. Privformer: Privacy-preserving transformer with mpgc. In 2023 IEEE 8th European Symposium on Security and Privacy (EuroSP), pp. 392-410, Los Alamitos, CA, USA, 2023. IEEE Computer Society. doi:10.1109/EuroSP57164.2023.00031. URL https://doi.ieeecomputersociety.org/10.1109/EuroSP57164.2023.00031.  
Toshinori Araki, Jun Furukawa, Yehuda Lindell, Ariel Nof, and Kazuma Ohara. High-throughput semi-honest secure three-party computation with an honest majority. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, pp. 805-817, 2016.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.  
Megha Byali, Harsh Chaudhari, Arpita Patra, and Ajith Suresh. Flash: Fast and robust framework for privacy-preserving machine learning. Proc. Priv. Enhancing Technol., 2020(2):459-480, 2020.  
Hanting Chen, Yunhe Wang, Tianyu Guo, Chang Xu, Yiping Deng, Zhenhua Liu, Siwei Ma, Chunjing Xu, Chao Xu, and Wen Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12299-12310, 2021.  
Anders Dalskov, Daniel Escudero, and Marcel Keller. Fantastic four: Honest-majority four-party secure computation with malicious security. In 30th {USENIX} Security Symposium ( {USENIX} Security 21), 2021.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. ArXiv, abs/1810.04805, 2019.  
Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, and Nenghai Yu. Bootstrapped masked autoencoders for vision bert pretraining. In European Conference on Computer Vision, pp. 247-264. Springer, 2022.  
Ye Dong, Chen Xiaojun, Weizhan Jing, Li Kaiyun, and Weiping Wang. Meteor: Improved secure 3-party neural network inference with reducing online communication costs. In Proceedings of the ACM Web Conference 2023, WWW '23, pp. 2087-2098, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394161.  
O. Goldreich, S. Micali, and A. Wigderson. How to play any mental game. In Proceedings of the Nineteenth Annual ACM Symposium on Theory of Computing, STOC '87, pp. 218-229, New York, NY, USA, 1987. Association for Computing Machinery. ISBN 0897912217.  
Meng Hao, Hongwei Li, Hanxiao Chen, Pengzhi Xing, Guowen Xu, and Tianwei Zhang. Iron: Private inference on transformers. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=deyqjpcTfsG.  
Zhicong Huang, Wen jie Lu, Cheng Hong, and Jiansheng Ding. Cheetah: Lean and fast secure Two-Party deep neural network inference. In 31st USENIX Security Symposium (USENIX Security 22), pp. 809–826, Boston, MA, August 2022. USENIX Association. ISBN 978-1-939133-31-1.  
Marcel Keller. Mp-spdz: A versatile framework for multi-party computation. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 1575-1590, 2020.  
B. Knott, S. Venkataraman, A.Y. Hannun, S. Sengupta, M. Ibrahim, and L.J.P. van der Maaten. Crypten: Secure multi-party computation meets machine learning. In arXiv 2109.00984, 2021.

Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. arXiv preprint arXiv:2202.10054, 2022.  
Nishant Kumar, Mayank Rathee, Nishanth Chandran, Divya Gupta, Aseem Rastogi, and Rahul Sharma. Cryptflow: Secure tensorflow inference. arXiv preprint arXiv:1909.07814, 2019.  
Dacheng Li, Hongyi Wang, Rulin Shao, Han Guo, Eric Xing, and Hao Zhang. MPCFORMER: FAST, PERFORMANT AND PRIVATE TRANSFORMER INFERENCE WITH MPC. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=CWmvjOEtH-.  
Zi Liang, Pinghui Wang, Ruofei Zhang, Lifeng Xing, Nuo Xu, and Shuo Zhang. Merge: Fast private text generation, 2023.  
Jian Liu, Mika Juuti, Yao Lu, and Nadarajah Asokan. Oblivious neural network predictions via minionn transformations. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 619-631, 2017.  
Xuanqi Liu and Zhuotao Liu. Llms can understand encrypted prompt: Towards privacy-computing friendly transformers, 2023.  
Wen-jie Lu, Yixuan Fang, Zhicong Huang, Cheng Hong, Chaochao Chen, Hunter Qu, Yajin Zhou, and Kui Ren. Faster secure multiparty computation of adaptive gradient descent. In Proceedings of the 2020 Workshop on Privacy-Preserving Machine Learning in Practice, PPMLP'20, pp. 47-49, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450380881.  
Junming Ma, Yancheng Zheng, Jun Feng, Derun Zhao, Haoqi Wu, Wenjing Fang, Jin Tan, Chaofan Yu, Benyu Zhang, and Lei Wang. SecretFlow-SPU: A performant and User-Friendly framework for Privacy-Preserving machine learning. In 2023 USENIX Annual Technical Conference (USENIX ATC 23), pp. 17-33, Boston, MA, July 2023. USENIX Association. ISBN 978-1-939133-35-9. URL https://www.usenix.org/conference/atc23/presentation/ma.  
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.  
Pratyush Mishra, Ryan Lehmkuhl, Akshayaram Srinivasan, Wenting Zheng, and Raluca Ada Popa. Delphi: A cryptographic inference service for neural networks. In 29th {USENIX} Security Symposium (  $\{\mathsf{USENIX}\}$  Security 20), pp. 2505-2522, 2020.  
Payman Mohassel and Peter Rindal. Aby3: A mixed protocol framework for machine learning. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, pp. 35-52, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450356930. doi:10.1145/3243734.3243760. URL https://doi.org/10.1145/3243734.3243760.  
Payman Mohassel and Yupeng Zhang. Securel: A system for scalable privacy-preserving machine learning. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 19-38. IEEE, 2017.  
Arpita Patra and Ajith Suresh. Blaze: blazing fast privacy-preserving machine learning. arXiv preprint arXiv:2005.09042, 2020.  
Arpita Patra, Thomas Schneider, Ajith Suresh, and Hossein Yalame. {ABY2.0}: Improved {Mixed-Protocol} secure {Two-Party} computation. In 30th USENIX Security Symposium (USENIX Security 21), pp. 2165-2182, 2021.  
Alec Radford and Karthik Narasimhan. Improving language understanding by generative pre-training. 2018.  
Deevashwer Rathee, Mayank Rathee, Nishant Kumar, Nishanth Chandran, Divya Gupta, Aseem Rastogi, and Rahul Sharma. Cryptflow2: Practical 2-party secure inference. New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450370899. URL https://doi.org/10.1145/3372297.3417274.

Deevashwer Rathee, Mayank Rathee, Rahul Kranti Kiran Goli, Divya Gupta, Rahul Sharma, Nishanth Chandran, and Aseem Rastogi. Sirnn: A math library for secure rnns inference. arXiv preprint arXiv:2105.04236, 2021.  
Adi Shamir. How to share a secret. Communications of the ACM, 22(11):612-613, 1979.  
Jonathan Soifer, Jason Li, Mingqin Li, Jeffrey Zhu, Yingnan Li, Yuxiong He, Elton Zheng, Adi Oltean, Maya Mosyak, Chris Barnes, et al. Deep learning inference service at microsoft. In 2019 USENIX Conference on Operational Machine Learning (OpML 19), pp. 15-17, 2019.  
Sijun Tan, Brian Knott, Yuan Tian, and David J Wu. Cryptgpu: Fast privacy-preserving machine learning on thegpu. arXiv preprint arXiv:2104.10949, 2021.  
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  
Wouter van Oortmerssen. Flat buffers: a memory efficient serialization library. Web Page. android developers. googleblog. com/2014/06/flatuffers-memory-efficient. html, 2014.  
Kenton Varda. Protocol buffers: Google's data interchange format. Google Open Source Blog, Available at least as early as Jul, 72:23, 2008.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.  
Sameer Wagh, Divya Gupta, and Nishanth Chandran. Securenn: 3-party secure computation for neural network training. Proceedings on Privacy Enhancing Technologies, 2019(3):26-49, 2019.  
Sameer Wagh, Shruti Tople, Fabrice Benhamouda, Eyal Kushilevitz, Prateek Mittal, and Tal Rabin. Falcon: Honest-majority maliciously secure framework for private deep learning. arXiv preprint arXiv:2004.02229, 2020.  
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJ4km2R5t7.  
Yongqin Wang, G. Edward Suh, Wenjie Xiong, Benjamin Lefaudeaux, Brian Knott, Murali Annavaram, and Hsien-Hsin S. Lee. Characterization of mpc-based private inference for transformer-based models. In 2022 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), pp. 187-197, 2022. doi:10.1109/ISPASS55109.2022.00025.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38-45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.  
Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V. Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2019. Curran Associates Inc.  
Andrew Chi-Chih Yao. How to generate and exchange secrets. In 27th Annual Symposium on Foundations of Computer Science (sfcs 1986), pp. 162-167. IEEE, 1986.  
Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, and Ling Shao. Kaleido-bert: Vision-language pre-training on fashion domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12647-12657, 2021.
