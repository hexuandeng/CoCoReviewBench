# QUATERNION RECURRENT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recurrent neural networks (RNNs) are powerful architectures to model sequential data, due to their capability to learn short and long-term dependencies between the basic elements of a sequence. Nonetheless, popular tasks such as speech or images recognition, involve multi-dimensional input features that are characterized by strong internal dependencies between the dimensions of the input vector. We propose a novel quaternion recurrent neural network (QRNN), alongside with a quaternion long-short term memory neural network (QLSTM), that take into account both the external relations and these internal structural dependencies with the quaternion algebra. Similarly to capsules, quaternions allow the QRNN to code internal dependencies by composing and processing multidimensional features as single entities, while the recurrent operation reveals correlations between the elements composing the sequence. We show that both QRNN and QLSTM achieve better performances than RNN and LSTM in a realistic application of automatic speech recognition. Finally, we show that QRNN and QLSTM reduce by a maximum factor of  $3.3\mathrm{x}$  the number of free parameters needed, compared to real-valued RNNs and LSTMs to reach better results, leading to a more compact representation of the relevant information.

# 1 INTRODUCTION

In the last few years, deep neural networks (DNN) have encountered a wide success in different domains due to their capability to learn highly complex input to output mapping. Among the different DNN-based models, the recurrent neural network (RNN) is well adapted to represent sequential data. Indeed, RNNs build a vector of activations at each timestep to code latent relations between input vectors. Deep RNNs have been recently used to obtain hidden representations of speech unit sequences (30) or text word sequences (6). However, many recent tasks based on multi-dimensional input features, such as pixels of an image, acoustic features, or orientations of 3D models, require to represent both external dependencies between each entity, and internal relations between the features that compose this entity. Moreover, RNN-based algorithms commonly require a huge number of parameters to represent sequential data in the hidden space.

Quaternions are hypercomplex numbers that contain a real and three separate imaginary components, fitting perfectly to 3 and 4 dimensional feature vectors, such as for image processing and robot kinematics (32; 28; 3). The idea of bundling groups of numbers into separate entities is also exploited by the recent capsule network (31). Contrary to traditional homogeneous representations, capsule and quaternion networks bundle sets of features together. Thereby, quaternion numbers allow neural network based models to code latent inter-dependencies between groups of input features during the learning process with less parameters than RNNs, by taking advantage of the Hamilton product as the equivalent of the ordinary product, but between quaternions. Early applications of quaternion-valued backpropagation algorithms (2; 1) have efficiently solved quaternion functions approximation tasks. More recently, neural networks of complex and hypercomplex numbers have received an increasing attention (13; 38; 7; 39), and some efforts have shown promising results in different applications. In particular, a deep quaternion network (25; 26; 36), a deep quaternion convolutional network (4; 27), or a deep complex convolutional network (37) have been successfully employed for challenging tasks such as images and language processing. However, none of these applications merged recurrent neural networks and the quaternion algebra.

This paper proposes to integrate local spectral features in a novel model called quaternion

recurrent neural network $^{1}$  (QRNN), and its gated extension called quaternion long-short term memory neural network (QLSTM). The model is proposed along with a well-adapted parameters initialization and turned out to learn both inter- and intra-dependencies between multidimensional input features and the basic elements of a sequence with drastically less parameters (Section 3), making the approach more suitable for low-ressource applications. The effectiveness of the proposed QRNN and QLSTM is evaluated on the realistic TIMIT phoneme recognition task (Section 4.2) that shows that both QRNN and QLSTM obtain better performances than RNNs and LSTMs with a best observed phoneme error rate (PER) of  $18.5\%$  and  $15.1\%$  for QRNN and QLSTM, compared to  $19.0\%$  and  $15.3\%$  for RNN and LSTM. Moreover, these results are obtained alongside with a reduction of 3.3 times of the number of free parameters.

# 2 MOTIVATIONS

A major challenge of current machine learning models is to well-represent in the latent space the astonishing amount of data available for recent tasks. For this purpose, a good model has to efficiently encode local relations within the input features, such as between the Red, Green, and Blue (R,G,B) channels of a single pixel, as well as structural relations, such as those describing edges or shapes composed by groups of pixels. Moreover, in order to learn an adequate representation with the available set of training data and to avoid overfitting, it is convenient to conceive a neural architecture with the smallest number of parameters to be estimated. In the following, we detail the motivations to employ a quaternion-valued RNN instead of a real-valued one to code inter and intra features dependencies with less parameters.

As a first step, a better representation of multidimensional data has to be explored to naturally capture internal relations within the input features. For example, an efficient way to represent the information composing an image is to consider each pixel as being a whole entity of three strongly related elements, instead of a group of unidimensional elements that could be related to each others, as in traditional real-valued neural networks. Indeed, with a real-valued RNN, the latent relations between the RGB components of a given pixel are hardly coded in the latent space since the weight has to find out these relations among all the pixels composing the image. This problem is effectively solved by replacing real numbers by quaternions. Indeed, quaternions are fourth dimensional and allow one to build and process entities made of up to 4 elements. The quaternion algebra and more precisely the Hamilton product allows quaternion neural network to capture these internal latent relations within the features of a quaternion. It has been shown that QNN are able to restore the spatial relations within 3D coordinates (20), and within color pixels (16), while real-valued NN failed. This is easily explained by the fact that the quaternion-weight components are shared through multiple quaternion-input parts during the Hamilton product, creating relations within the elements, as depicted in Figure 1. Indeed, Figure 1 shows that the multiple weights required to code latent relations within a feature are considered at the same level as for learning global relations between different features, while the quaternion weight  $w$  codes these internal relations within a unique quaternion  $Q_{out}$  during the Hamilton product (right).

Then, while bigger neural networks allow better performances, quaternion neural networks make it possible to deal with the same signal dimension but with four times less neural parameters. Indeed, a 4-number quaternion weight linking two 4-number quaternion units only has 4 degrees of freedom, whereas a standard neural net parametrization have  $4 \times 4 = 16$ , i.e., a 4-fold saving in memory. Therefore, the natural multidimensional representation of quaternions alongside with their ability to drastically reduce the number of parameters indicate that hyper-complex numbers are a better fit than real numbers to create more efficient models in multidimensional spaces. Based on the success of previous deep quaternion convolutional neural networks and smaller quaternion feed-forward architectures (19; 17; 36), this work proposes to merge the efficient representation of hyper-complex numbers to the capability of recurrent neural networks into a natural and efficient framework to multidimensional sequential tasks such as speech recognition.

Indeed, modern automatic speech recognition systems usually employ input sequences composed of multidimensional acoustic features, such as log Mel features, that are often enriched with their first, second and third time derivatives (8; 9), to integrate contextual information. In standard RNNs, static features are simply concatenated with their derivatives to form a large input vector, without

![](images/0f17dc155d579efd9ddf9f0deeb7fa9a6aa4c82b7a0a59801a041934d20c26a9.jpg)  
Figure 1: Illustration of the input features  $(Q_{in})$  latent relations learning ability of a quaternion-valued layer (right) due to the quaternion weight sharing of the Hamilton product (Eq. 5), compared to a standard real-valued layer (left).

effectively considering that signal derivatives represent different views of the same input. Nonetheless, it is crucial to consider that time derivatives of the spectral energy in a given frequency band at a specific time frame represent a special state of a time-frame, and are thus correlated. Based on the above motivations and the results observed on previous works about quaternion neural networks, we hypothesize that quaternion RNNs naturally provide a more suitable representation of the input sequence, since these multiple views can be directly embedded in the multiple dimensions space of the quaternion, leading to better generalization.

# 3 QUATERNION RECURRENT NEURAL NETWORKS

This Section describes the quaternion algebra (Section 3.1), the internal quaternion representation (Section 3.2), the backpropagation through time (BPTT) for quaternions (Section 3.3.2), and proposes an adapted weight initialization to quaternion-valued neurons (Section 3.4).

# 3.1 QUATERNION ALGEBRA

The quaternion algebra  $\mathbb{H}$  defines operations between quaternion numbers. A quaternion  $\mathbf{Q}$  is an extension of a complex number defined in a four dimensional space as:

$$
Q = r 1 + x \mathbf {i} + y \mathbf {j} + z \mathbf {k}, \tag {1}
$$

where  $r, x, y,$  and  $z$  are real numbers, and 1, i, j, and k are the quaternion unit basis. In a quaternion,  $r$  is the real part, while  $x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$  with  $\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1$  is the imaginary part, or the vector part. Such a definition can be used to describe spatial rotations. The information embedded in the quaternion  $Q$  can be summarized into the following matrix of real numbers, that turns out to be more suitable for computations:

$$
Q _ {m a t} = \left[ \begin{array}{c c c c} r & - x & - y & - z \\ x & r & - z & y \\ y & z & r & - x \\ z & - y & x & r \end{array} \right]. \tag {2}
$$

The conjugate  $Q^{*}$  of  $Q$  is defined as:

$$
Q ^ {*} = r 1 - x \mathbf {i} - y \mathbf {j} - z \mathbf {k}. \tag {3}
$$

Then, a normalized or unit quaternion  $Q^{\triangleleft}$  is expressed as:

$$
Q ^ {\triangleleft} = \frac {Q}{\sqrt {r ^ {2} + x ^ {2} + y ^ {2} + z ^ {2}}}. \tag {4}
$$

Finally, the Hamilton product  $\otimes$  between two quaternions  $Q_{1}$  and  $Q_{2}$  is computed as follows:

$$
\begin{array}{l} Q _ {1} \otimes Q _ {2} = \left(r _ {1} r _ {2} - x _ {1} x _ {2} - y _ {1} y _ {2} - z _ {1} z _ {2}\right) + \\ \left(r _ {1} x _ {2} + x _ {1} r _ {2} + y _ {1} z _ {2} - z _ {1} y _ {2}\right) \mathbf {i} + \\ \left(r _ {1} y _ {2} - x _ {1} z _ {2} + y _ {1} r _ {2} + z _ {1} x _ {2}\right) \boldsymbol {j} + \\ \left(r _ {1} z _ {2} + x _ {1} y _ {2} - y _ {1} x _ {2} + z _ {1} r _ {2}\right) \boldsymbol {k}. \tag {5} \\ \end{array}
$$

The Hamilton product (a graphical view is depicted in Figure 1) is used in QRNNs to perform transformations of vectors representing quaternions, as well as scaling and interpolation between two rotations following a geodesic over a sphere in the  $\mathbb{R}^3$  space as shown in (22).

# 3.2 QUATERNION REPRESENTATION

The QRNN is an extension of the real-valued (21) and complex-valued (15; 33) recurrent neural networks to hypercomplex numbers. In a quaternion dense layer, all parameters are quaternions, including inputs, outputs, weights, and biases. The quaternion algebra is ensured by manipulating matrices of real numbers (4). Consequently, for each input vector of size  $N$ , output vector of size  $M$ , dimensions are split into four parts: the first one equals to  $r$ , the second is  $x\mathbf{i}$ , the third one equals to  $y\mathbf{j}$ , and the last one to  $z\mathbf{k}$  to compose a quaternion  $Q = r1 + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$ . The inference process of a fully-connected layer is defined in the real-valued space by the dot product between an input vector and a real-valued  $M \times N$  weight matrix. In a QRNN, this operation is replaced with the Hamilton product (eq. 5) with quaternion-valued matrices (i.e. each entry in the weight matrix is a quaternion).

# 3.3 LEARNING ALGORITHM

The QRNN differs from the real-valued RNN in each learning sub-processes. Therefore, let  $x_{t}$  be the input vector at timestep  $t$ ,  $h_{t}$  the hidden state,  $W_{hx}$ ,  $W_{hy}$  and  $W_{hh}$  the input, output and hidden states weight matrices respectively. The vector  $b_{h}$  is the bias of the hidden state and  $p_{t}, y_{t}$  are the output and the expected target vectors. More details of the learning process and the parametrization are available on Appendix 6.1.

# 3.3.1 FORWARD PHASE

Based on the forward propagation of the real-valued RNN (21), the QRNN forward equations are extended as follows:

$$
h _ {t} = \alpha \left(W _ {h h} \otimes h _ {t - 1} + W _ {h x} \otimes x _ {t} + b _ {h}\right), \tag {6}
$$

where  $\alpha$  is a quaternion split activation function (40) defined as:

$$
\alpha (Q) = f (r) + f (x) \mathbf {i} + f (y) \mathbf {j} + f (z) \mathbf {k}, \tag {7}
$$

with  $f$  corresponding to any standard activation function. The output vector  $p_t$  is computed as:

$$
p _ {t} = \beta \left(W _ {h y} \otimes h _ {t}\right), \tag {8}
$$

where  $\beta$  is any split activation function. Finally, the objective function is a classical real-valued loss applied component-wise(e.g., mean squared error, negative log-likelihood).

# 3.3.2 QUATERNION BACK PROPAGATION THROUGH TIME

The backpropagation through time (BPTT) for quaternion numbers is an extension of the standard quaternion backpropagation (24), and its full derivation is available in Appendix 6.2. The gradient with respect to the loss  $E_{t}$  is expressed for each weight matrix as  $\Delta_{hy} = \frac{\partial E_t}{\partial W_{hy}}$ ,  $\Delta_{hh} = \frac{\partial E_t}{\partial W_{hh}}$ , and  $\Delta_{hx} = \frac{\partial E_t}{\partial W_{hx}}$ , and can be generalized to  $\Delta = \frac{\partial E_t}{\partial W}$  with:

$$
\frac {\partial E _ {t}}{\partial W} = \frac {\partial E _ {t}}{\partial W ^ {r}} + \mathbf {i} \frac {\partial E _ {t}}{\partial W ^ {i}} + \mathbf {j} \frac {\partial E _ {t}}{\partial W ^ {j}} + \mathbf {k} \frac {\partial E _ {t}}{\partial W ^ {k}}. \tag {9}
$$

Each term of the above relation is then computed by applying the chain rule. As a use-case for the equations, the mean squared error at a timestep  $t$  and named  $E_{t}$  is used as the loss function. Moreover,

let  $\lambda$  be a fixed learning rate. First, the weight matrix  $W_{hy}$  is only seen in the equations of  $p_t$ . It is therefore straightforward to update each weight of  $W_{hy}$  at timestep  $t$  following:

$$
W _ {h y} = W _ {h y} + \lambda \Delta_ {h y} ^ {t} \otimes h _ {t} ^ {*}, \text {w i t h} \Delta_ {h y} ^ {t} = \frac {\partial E _ {t}}{\partial W _ {h y}} = \left(p _ {t} - y _ {t}\right). \tag {10}
$$

Then, the weight matrices  $W_{hh}, W_{hx}$  and biases  $b_{h}$  are arguments of  $h_t$  with  $h_{t-1}$  involved. Therefore, the update equations are derived as:

$$
W _ {h h} = W _ {h h} + \lambda \Delta_ {h h} ^ {t}, \quad W _ {h x} = W _ {h x} + \lambda \Delta_ {h x} ^ {t}, \quad b _ {h} = b _ {h} + \lambda \Delta_ {h h} ^ {t}, \tag {11}
$$

with,

$$
\Delta_ {h h} ^ {t} = \frac {\partial E _ {t}}{\partial W _ {h h}} = \sum_ {m = 0} ^ {t} \left(\prod_ {n = m} ^ {t} \delta_ {n}\right) \otimes h _ {m - 1} ^ {*}, \quad \Delta_ {h x} ^ {t} = \frac {\partial E _ {t}}{\partial W _ {h x}} = \sum_ {m = 0} ^ {t} \left(\prod_ {n = m} ^ {t} \delta_ {n}\right) \otimes x _ {m} ^ {*}, \tag {12}
$$

and,

$$
\delta_ {n} = \left\{ \begin{array}{l l} W _ {h h} ^ {*} \otimes \delta_ {n + 1} \times \alpha^ {\prime} \left(h _ {n} ^ {\text {p r e a c t}}\right) & \text {i f} n \neq t \\ W _ {h y} ^ {*} \otimes \left(p _ {n} - y _ {n}\right) \times \beta^ {\prime} \left(p _ {n} ^ {\text {p r e a c t}}\right) & \text {e l s e}, \end{array} \right. \tag {13}
$$

with  $h_n^{preact}$  and  $p_n^{preact}$  the pre-activation values of  $h_n$  and  $p_n$  respectively.

# 3.4 PARAMETER INITIALIZATION

A well-designed parameter initialization scheme strongly impacts on the efficiency of a DNN. An appropriate initialization, in fact, improves DNN convergence, reduces the risk of exploding or vanishing gradient, and often leads to a substantial performance improvement (11). It has been shown that the backpropagation through time algorithm of RNNs is degraded by a inappropriate parameter initialization (35). Moreover, an hyper-complex parameter cannot be simply initialized randomly and component-wise, due to the interactions between components. Therefore, this section proposes a procedure reported in Algorithm 1 to initialize a matrix  $W$  of quaternion-valued weights. The proposed initialization equations are derived from the polar form of a weight  $w$  of  $W$ :

$$
w = | w | e ^ {q _ {i m a g} ^ {\triangleleft} \theta} = | w | (\cos (\theta) + q _ {i m a g} ^ {\triangleleft} \sin (\theta)), \tag {14}
$$

and,

$$
w _ {\mathbf {r}} = \varphi \cos (\theta),
$$

$$
w _ {\mathbf {i}} = \varphi q _ {i m a g \mathbf {i}} ^ {\triangleleft} \sin (\theta), \quad w _ {\mathbf {j}} = \varphi q _ {i m a g \mathbf {j}} ^ {\triangleleft} \sin (\theta), \quad w _ {\mathbf {k}} = \varphi q _ {i m a g \mathbf {k}} ^ {\triangleleft} \sin (\theta). \tag {15}
$$

The angle  $\theta$  is randomly generated in the interval  $[- \pi, \pi]$ . The quaternion  $q_{imag}^{\triangleleft}$  is defined as purely normalized imaginary, and is expressed as  $q_{imag}^{\triangleleft} = 0 + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$ . The imaginary components  $\mathbf{x}\mathbf{i}$ ,  $\mathbf{y}\mathbf{j}$ , and  $\mathbf{z}\mathbf{k}$  are sampled from an uniform distribution in  $[0,1]$  to obtain  $q_{imag}$ , which is then normalized (following eq. 4) to obtain  $q_{imag}^{\triangleleft}$ . The parameter  $\varphi$  is a random number generated with respect to well-known initialization criterions (such as Glorot or He algorithms) (11; 12).

However, the equations derived in (11; 12) are defined for real-valued weight matrices. Therefore, the variance of  $\bar{W}$  has to be investigated in the quaternion space to obtain  $\varphi$  (the full demonstration is provided in Appendix 5.1). The variance of  $W$  is defined as:

$$
V a r (W) = \mathbb {E} \left(\left| W \right| ^ {2}\right) - \left[ \mathbb {E} \left(\left| W \right|\right) \right] ^ {2}, \text {w i t h} \left[ \mathbb {E} \left(\left| W \right|\right) \right] ^ {2} = 0. \tag {16}
$$

Indeed, the weight distribution is normalized. The value of  $Var(W) = \mathbb{E}(|W|^2)$ , instead, is not trivial in the case of quaternion-valued matrices. Indeed,  $W$  follows a Chi-distribution with four degrees of freedom (DOFs). Consequently,  $Var(W)$  is expressed and computed as follows:

$$
V a r (W) = \mathbb {E} (| W | ^ {2}) = \int_ {0} ^ {\infty} x ^ {2} f (x) \mathrm {d} x = 4 \sigma^ {2}. \tag {17}
$$

The Glorot (11) and He (12) criterions are thus extended to quaternion as following:

$$
\sigma = \frac {1}{\sqrt {2 \left(n _ {i n} + n _ {o u t}\right)}}, \text {a n d} \sigma = \frac {1}{\sqrt {2 n _ {i n}}}, \tag {18}
$$

with  $n_{in}$  and  $n_{out}$  the number of neurons of the input and output layers respectively. Finally,  $\varphi$  can be sampled from  $[-\sigma, \sigma]$  to complete the weight initialization of eq. 15.

Algorithm 1 Quaternion-valued weight initialization  
1: procedure QINIT(W,  $n_{in}$  nout)   
2:  $\sigma \leftarrow \frac{1}{\sqrt{2(n_{in} + n_{out})}}$  ▷w.r.t to Glorot criterion and eq. 18   
4:   
5: for w in W do   
6:  $\theta \gets rand(-\pi ,\pi)$    
7:  $\varphi \gets rand(-\sigma ,\sigma)$    
8:  $x,y,z\gets rand(0,1)$    
9:  $q_{imag}\gets$  Quaternion(0,x,y,z)   
10:  $q_{imag}^{\triangleleft}\gets \frac{q_{imag}}{\sqrt{x^{2} + y^{2} + z^{2}}}$    
11:  $w_{r}\gets \varphi \times cos(\theta)$  ▷ See eq. 15   
12:  $w_{i}\gets \varphi \times q_{imag_{i}}^{\triangleleft}\times sin(\theta)$    
13:  $w_{j}\gets \varphi \times q_{imag_{j}}^{\triangleleft}\times sin(\theta)$    
14:  $w_{k}\gets \varphi \times q_{imag_{k}}^{\triangleleft}\times sin(\theta)$    
15:  $w\gets$ Quaternion(wr,wi,wj,wk)

# 4 EXPERIMENTS

This Section details the acoustic features extraction (Section 4.1), the experimental setups and the results obtained with QRNNs, QLSTMs, RNNs and LSTMs on the TIMIT speech recognition tasks (Section 4.2). The results reported in bold on tables are obtained with the best configurations of the neural networks observed with the validation set.

# 4.1 QUATERNION ACOUSTIC FEATURES

The raw audio is first splitted every 10ms with a window of  $25\mathrm{ms}$ . Then 40-dimensional log Mel-filter-bank coefficients with first, second, and third order derivatives are extracted using the pytorch-kaldi<sup>2</sup> toolkit and the Kaldi s5 recipes (29). An acoustic quaternion  $Q(f,t)$  associated with a frequency  $f$  and a time-frame  $t$  is formed as follows:

$$
Q (f, t) = e (f, t) + \frac {\partial e (f , t)}{\partial t} \mathbf {i} + \frac {\partial^ {2} e (f , t)}{\partial^ {2} t} \mathbf {j} + \frac {\partial^ {3} e (f , t)}{\partial^ {3} t} \mathbf {k}. \tag {19}
$$

$Q(f,t)$  represents multiple views of a frequency  $f$  at time frame  $t$ , consisting of the energy  $e(f,t)$  in the filter band at frequency  $f$ , its first time derivative describing a slope view, its second time derivative describing a concavity view, and the third derivative describing the rate of change of the second derivative. Quaternions are used to learn the spatial relations that exist between the 3 described different views that characterize a same frequency. Thus, the quaternion input vector length is  $160/4 = 40$ . Decoding is based on Kaldi (29) and weighted finite state transducers (WFST) (23) that integrate acoustic, lexicon and language model probabilities into a single HMM-based search graph.

# 4.2 THE TIMIT CORPUS

The training process is based on the standard 3,696 sentences uttered by 462 speakers, while testing is conducted on 192 sentences uttered by 24 speakers of the TIMIT (10) dataset. A validation set composed of 400 sentences uttered by 50 speakers is used for hyper-parameter tuning. The models are compared on a fixed number of layers  $M = 4$  and by varying the number of neurons  $N$  from 256 to 2,048, and 64 to 512 for the RNN and QRNN respectively. Indeed, it is worth underlying that the number of hidden neurons in the quaternion and real spaces do not handle the same amount of real-number values. Indeed, 256 quaternion neurons output are  $256 \times 4 = 1024$  real values. Tanh activations are used across all the layers except for the output layer that is based on a softmax function. Models are optimized with RMSPROP (18) with vanilla hyper-parameters and an initial learning rate of  $8 \cdot 10^{-4}$ . The learning rate is progressively annealed using an halving factor of 0.5 that is

applied when no performance improvement on the validation set is observed. The models are trained during 25 epochs. All the models converged to a minimum loss, due to the annealed learning rate. A dropout rate of 0.2 is applied over all the hidden layers (34) except the output one. The negative log-likelihood loss function is used as an objective function. All the experiments are repeated 5 times (5-folds) with different seeds and are averaged to limit any variation due to the random initialization.

Table 1: Phoneme error rate (PER%) of QRNN and RNN models on the development and test sets of the TIMIT dataset. "Params" stands for the total number of trainable parameters.  

<table><tr><td>Models</td><td>Neurons</td><td>Dev.</td><td>Test</td><td>Params</td></tr><tr><td rowspan="4">RNN</td><td>256</td><td>22.4</td><td>23.4</td><td>1M</td></tr><tr><td>512</td><td>19.6</td><td>20.4</td><td>2.8M</td></tr><tr><td>1,024</td><td>17.9</td><td>19.0</td><td>9.4M</td></tr><tr><td>2,048</td><td>20.0</td><td>20.7</td><td>33.4M</td></tr><tr><td rowspan="4">QRNN</td><td>64</td><td>23.6</td><td>23.9</td><td>0.6M</td></tr><tr><td>128</td><td>19.2</td><td>20.1</td><td>1.4M</td></tr><tr><td>256</td><td>17.4</td><td>18.5</td><td>3.8M</td></tr><tr><td>512</td><td>17.5</td><td>18.7</td><td>11.2M</td></tr></table>

The results on the TIMIT task are reported in Table 1. The best PER in realistic conditions (w.r.t to the best validation PER) is  $18.2\%$  and  $19.0\%$  on the test set for QRNN and RNN models respectively, highlighting an absolute improvement of  $0.5\%$  obtained with QRNN. These results compare favorably with the best results obtained so far with architectures that do not integrate access control in multiple memory layers (30). Moreover, a remarkable advantage of QRNNs is a drastic reduction (with a factor of  $2.5\times$ ) of the parameters needed to achieve these results. Indeed, such PERs are obtained with models that employ the same internal dimensionality corresponding to 1,024 real-valued neurons and 256 quaternion-valued ones, resulting in a number of parameters of 3.8M for QRNN against the 9.4M used in the real-valued RNN. It is also worth noting that QRNNs consistently need less parameters than equivalently sized RNNs, with an average reduction factor of 2.26 times. This is easily explained by considering the content of the quaternion algebra. Indeed, for a fully-connected layer with 2,048 input values and 2,048 hidden units, a real-valued RNN has  $2,048^2 \approx 4.2\mathrm{M}$  parameters, while to maintain equal input and output dimensions the quaternion equivalent has 512 quaternions inputs and 512 quaternion hidden units. Therefore, the number of parameters for the quaternion-valued model is  $512^2 \times 4 \approx 1\mathrm{M}$ . Such a complexity reduction turns out to produce better results and have other advantages such as a smaller memory footprint while saving models on budget memory systems. This characteristic makes our QRNN model particularly suitable for speech recognition conducted on low computational power devices like smartphones and tablets (5). QRNNs and RNNs accuracies vary accordingly to the architecture with better PER on bigger and wider topologies. Therefore, while good PER are observed with higher number of parameters, smaller architectures performed at  $23.9\%$  and  $23.4\%$ , with 1M and 0.6M parameters for the RNN and the QRNN respectively. Such low PER are due to a too small number of parameters to solve the TIMIT task.

# 4.2.1 QUATERNION LONG-SHORT TERM MEMORY NEURAL NETWORKS

We propose to extend the QRNN to state-of-the-art models such as long-short term memory neural networks (LSTM), to support and improve the results already observed with the QRNN compared to the RNN in more realistic conditions. LSTM (14) neural networks were introduced to solve the problems of long-term dependencies learning and vanishing or exploding gradient observed with long sequences. Based on the equations of the forward propagation and back propagation through time of QRNN described in Section 3.3.1, and Section 3.3.2, one can easily derive the equations of a quaternion-valued LSTM. Gates are defined with quaternion numbers following the proposal of (7). Therefore, the gate action is characterized by an independent modification of each component of the quaternion-valued signal following a component-wise product with the quaternion-valued gate potential. Let  $f_{t}, i_{t}, o_{t}, c_{t}$ , and  $h_{t}$  be the forget, input, output gates, cell states and the hidden state of a LSTM cell at time-step  $t$ :

$$
f _ {t} = \alpha \left(W _ {f} \otimes x _ {t} + R _ {f} \otimes h _ {t - 1} + b _ {f}\right), \tag {20}
$$

$$
i _ {t} = \alpha \left(W _ {i} \otimes x _ {t} + R _ {i} \otimes h _ {t - 1} + b _ {i}\right), \tag {21}
$$

$$
c _ {t} = f _ {t} \times c _ {t - 1} + i _ {t} \times \tanh  \left(W _ {c} x _ {t} + R _ {c} h _ {t - 1} + b _ {c}\right), \tag {22}
$$

$$
o _ {t} = \alpha \left(W _ {o} \otimes x _ {t} + R _ {o} \otimes h _ {t - 1} + b _ {o}\right), \tag {23}
$$

$$
h _ {t} = o _ {t} \times \tanh  \left(c _ {t}\right), \tag {24}
$$

where  $W$  are rectangular input weight matrices,  $R$  are square recurrent weight matrices, and  $b$  are bias vectors.  $\alpha$  is the split activation function and  $\times$  denotes a component-wise product between two quaternions. Both QLSTM and LSTM are bidirectionals and trained on the same conditions than for the QRNN and RNN experiments.

Table 2: Phoneme error rate (PER%) of QLSTM and LSTM models on the development and test sets of the TIMIT dataset. "Params" stands for the total number of trainable parameters.  

<table><tr><td>Models</td><td>Neurons</td><td>Dev.</td><td>Test</td><td>Params</td></tr><tr><td rowspan="4">LSTM</td><td>256</td><td>14.9</td><td>16.5</td><td>3.6M</td></tr><tr><td>512</td><td>14.2</td><td>16.1</td><td>12.6M</td></tr><tr><td>1,024</td><td>14.4</td><td>15.3</td><td>46.2M</td></tr><tr><td>2,048</td><td>14.0</td><td>15.9</td><td>176.3M</td></tr><tr><td rowspan="4">QLSTM</td><td>64</td><td>15.5</td><td>17.0</td><td>1.6M</td></tr><tr><td>128</td><td>14.1</td><td>16.0</td><td>4.6M</td></tr><tr><td>256</td><td>14.0</td><td>15.1</td><td>14.4M</td></tr><tr><td>512</td><td>14.2</td><td>15.1</td><td>49.9M</td></tr></table>

The results reported on Table 2 support the initial intuitions and the previously established trends. We first point out that the best PER observed is  $15.1\%$  and  $15.3\%$  on the test set for QLSTMs and LSTM models respectively with an absolute improvement of  $0.2\%$  obtained with QLSTM using 3.3 times less parameters compared to LSTM. These results are among top of the line results and prove that the proposed quaternion approach can be used in state-of-the-art models.

# 5 CONCLUSION

Summary. This paper proposes to process sequences of multidimensional features (such as acoustic data) with a novel quaternion recurrent neural network (QRNN) and quaternion long-short term memory neural network (QLSTM). The experiments conducted on the TIMIT phoneme recognition task show that QRNNs and QLSTMs are more effective to learn a compact representation of multidimensional informations by outperforming RNNs and LSTMs with 2 to 3 times less free parameters. Therefore, our initial intuition that the quaternion algebra offers a better and more compact representation for multidimensional features, alongside with a better learning capability of feature internal dependencies through the Hamilton product, have been demonstrated.

Future Work. Future investigations will develop other multi-view features that contribute to decrease ambiguities in representing phonemes in the quaternion space. In this extend, a recent approach based on a quaternion Fourier transform to create quaternion-valued signal has to be investigated.

# REFERENCES

[1] Paolo Arena, Luigi Fortuna, Giovanni Muscato, and Maria Gabriella Xibilia. Multilayer perceptrons to approximate quaternion valued functions. Neural Networks, 10(2):335-342, 1997.  
[2] Paolo Arena, Luigi Fortuna, Luigi Occhipinti, and Maria Gabriella Xibilia. Neural networks for quaternion-valued function approximation. In Circuits and Systems, ISCAS'94., IEEE International Symposium on, volume 6, pages 307-310. IEEE, 1994.

[3] Nicholas A Asragathos and John K Dimitros. A comparative study of three methods for robot kinematics. Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on, 28(2):135-145, 1998.  
[4] Anthony Maida Chase Gaudet. Deep quaternion networks. arXiv preprint arXiv:1712.04604v2, 2017.  
[5] G. Chen, C. Parada, and G. Heigold. Small-footprint keyword spotting using deep neural networks. In 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4087-4091, May 2014.  
[6] Alexis Conneau, German Kruszewski, Guillaume Lample, Loic Barrault, and Marco Baroni. What you can cram into a single vector: Probing sentence embeddings for linguistic properties, 2018.  
[7] Ivo Danihelka, Greg Wayne, Benigno Uria, Nal Kalchbrenner, and Alex Graves. Associative long short-term memory. arXiv preprint arXiv:1602.03032, 2016.  
[8] Steven B Davis and Paul Mermelstein. Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. In Readings in speech recognition, pages 65-74. Elsevier, 1990.  
[9] Sadaoki Furui. Speaker-independent isolated word recognition based on emphasized spectral dynamics. In Acoustics, Speech, and Signal Processing, IEEE International Conference on ICASSP'86., volume 11, pages 1991-1994. IEEE, 1986.  
[10] John S Garofolo, Lori F Lamel, William M Fisher, Jonathan G Fiscus, and David S Pallett. Darpa limit acoustic-phonetic continuous speech corpus cd-rom. nist speech disc 1-1.1. NASA STI/Recon technical report n, 93, 1993.  
[11] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In International conference on artificial intelligence and statistics, pages 249-256, 2010.  
[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026-1034, 2015.  
[13] Akira Hirose and Shotaro Yoshida. Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence. IEEE Transactions on Neural Networks and learning systems, 23(4):541-551, 2012.  
[14] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.  
[15] Jin Hu and Jun Wang. Global stability of complex-valued recurrent neural networks with time-delays. IEEE Transactions on Neural Networks and Learning Systems, 23(6):853-865, 2012.  
[16] Teijiro Isokawa, Tomoaki Kusakabe, Nobuyuki Matsui, and Ferdinand Peper. Quaternion neural network and its application. In International Conference on Knowledge-Based and Intelligent Information and Engineering Systems, pages 318-324. Springer, 2003.  
[17] Teijiro Isokawa, Nobuyuki Matsui, and Haruhiko Nishimura. *Quaternionic neural networks: Fundamental properties and applications*. Complex-Valued Neural Networks: Utilizing High-Dimensional Parameters, pages 411–439, 2009.  
[18] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[19] Hiromi Kusamichi, Teijiro Isokawa, Nobuyuki Matsui, Yuzo Ogawa, and Kazuaki Maeda. A new scheme for color night vision by quaternion neural network. In Proceedings of the 2nd International Conference on Autonomous Robots and Agents, volume 1315, 2004.

[20] Nobuyuki Matsui, Teijiro Isokawa, Hiromi Kusamichi, Ferdinand Peper, and Haruhiko Nishimura. Quaternion neural network with geometrical operators. Journal of Intelligent & Fuzzy Systems, 15(3, 4):149-164, 2004.  
[21] Larry R. Medsker and Lakhmi J. Jain. Recurrent neural networks. Design and Applications, 5, 2001.  
[22] Toshifumi Minemoto, Teijiro Isokawa, Haruhiko Nishimura, and Nobuyuki Matsui. Feed forward neural network with random quaternionic neurons. Signal Processing, 136:59-68, 2017.  
[23] Mehryar Mohri, Fernando Pereira, and Michael Riley. Weighted finite-state transducers in speech recognition. Computer Speech and Language, 16(1):69 - 88, 2002.  
[24] Tohru Nitta. A quaternary version of the back-propagation algorithm. In Neural Networks, 1995. Proceedings., IEEE International Conference on, volume 5, pages 2753-2756. IEEE, 1995.  
[25] Titouan Parcollet, Mohamed Morchid, Pierre-Michel Bousquet, Richard Dufour, Georges Linares, and Renato De Mori. Quaternion neural networks for spoken language understanding. In Spoken Language Technology Workshop (SLT), 2016 IEEE, pages 362-368. IEEE, 2016.  
[26] Titouan Parcollet, Mohamed Morchid, and Georges Linares. Deep quaternion neural networks for spoken language understanding. In Automatic Speech Recognition and Understanding Workshop (ASRU), 2017 IEEE, pages 504-511. IEEE, 2017.  
[27] Titouan Parcollet, Ying Zhang, Mohamed Morchid, Chiheb Trabelsi, Georges Linares, Renato De Mori, and Yoshua Bengio. Quaternion convolutional neural networks for end-to-end automatic speech recognition. arXiv preprint arXiv:1806.07789, 2018.  
[28] Soo-Chang Pei and Ching-Min Cheng. Color image processing by using binary quaternion-moment-preserving thresholding technique. IEEE Transactions on Image Processing, 8(5):614-628, 1999.  
[29] Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukas Burget, Ondrej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlicek, Yanmin Qian, Petr Schwarz, Jan Silovsky, Georg Stemmer, and Karel Vesely. The kaldi speech recognition toolkit. In IEEE 2011 Workshop on Automatic Speech Recognition and Understanding. IEEE Signal Processing Society, December 2011. IEEE Catalog No.: CFP11SRW-USB.  
[30] Mirco Ravanelli, Philemon Brakel, Maurizio Omologo, and Yoshua Bengio. Light gated recurrent units for speech recognition. IEEE Transactions on Emerging Topics in Computational Intelligence, 2(2):92-102, 2018.  
[31] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. arXiv preprint arXiv:1710.09829v2, 2017.  
[32] Stephen John Sangwine. Fourier transforms of colour images using quaternion or hypercomplex, numbers. Electronics letters, 32(21):1979-1980, 1996.  
[33] Jingyan Song and Yeung Yam. Complex recurrent neural network for computing the inverse and pseudo-inverse of the complex matrix. Applied mathematics and computation, 93(2-3):195-205, 1998.  
[34] Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: A simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
[35] Ilya Sutskever, James Martens, George Dahl, and Geoffrey Hinton. On the importance of initialization and momentum in deep learning. In International conference on machine learning, pages 1139-1147, 2013.  
[36] Parcollet Titouan, Mohamed Morchid, and Georges Linares. Quaternion denoising encoder-decoder for theme identification of telephone conversations. Proc. Interspeech 2017, pages 3325-3328, 2017.

[37] Chiheb Trabelsi, Olexa Bilaniuk, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, and Christopher J Pal. Deep complex networks. arXiv preprint arXiv:1705.09792, 2017.  
[38] Mark Tygert, Joan Bruna, Soumith Chintala, Yann LeCun, Serkan Piantino, and Arthur Szlam. A mathematical motivation for complex-valued convolutional networks. Neural computation, 28(5):815-825, 2016.  
[39] Scott Wisdom, Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity unitary recurrent neural networks. In Advances in Neural Information Processing Systems, pages 4880-4888, 2016.  
[40] D Xu, L Zhang, and H Zhang. Learning algrithms in quaternion neural networks using ghr calculus. Neural Network World, 27(3):271, 2017.
