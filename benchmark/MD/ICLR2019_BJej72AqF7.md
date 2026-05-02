# A MAX-AFFINE SPLINE PERSPECTIVE OF RECURRENT NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We develop a framework for understanding and improving recurrent neural networks (RNNs) using max-affine spline operators (MASO). We prove that RNNs using piecewise affine and convex nonlinearities can be written as a simple piecewise affine spline operator. The resulting representation provides several new perspectives for analyzing RNNs, three of which we study in this paper. First, we show that an RNN internally partitions the input space during training using vector quantization and that it builds up the partition through time. Second, we show that the affine parameter of an RNN corresponds to an input-specific template, from which we can interpret an RNN as performing a simple template matching (matched filtering) given the input. Third, by closely examining the MASO RNN formula, we prove that injecting Gaussian noise in the initial hidden state in RNNs corresponds to an explicit  $\ell_2$  regularization on the affine parameters, which links to exploding gradient issues and improves generalization. Extensive experiments on several datasets of various modalities demonstrates and validates each of the above analyses. In particular, using initial hidden states elevates simple RNNs to state-of-the-art performance on these datasets.

# 1 INTRODUCTION

Recurrent neural networks (RNNs) are a powerful class of models for processing sequential inputs and have been the basic building block for more advanced models that have found success in challenging problems, including classification (e.g., sentiment analysis (Socher et al., 2013; Li et al., 2016; Teng et al., 2016)), sequence generation (e.g., machine translation (Bahdanau et al., 2014)), speech recognition and image captioning. Despite the successes, our understanding of how RNNs work still remains limited. An attractive theoretical result on RNNs are their universal approximation property which states that RNNs can approximate an arbitrary function (Schafer & Zimmermann, 2006; Siegelmann & Sontag, 1995; Hammer, 2000). These classical theoretical results are mostly obtained from a dynamic system (Siegelmann & Sontag, 1995; Schafer & Zimmermann, 2006) and measure theory (Hammer, 2000) perspective. These theories provide bounds on approximation error but unfortunately provide limited guidance on using RNNs and understanding their performance in practice.

In this paper, we study a specific class of RNNs with piecewise affine and convex nonlinearities. We provide a new angle of understanding RNNs with max-affine spline operators (MASOs) (Magnani & Boyd, 2009; Hannah & Dunson, 2013) from approximation theory. MASOs are piecewise affine approximation to arbitrary functions that provide a useful framework to examine neural networks. For example, Balestriero & Baraniuk (2018); Balestriero & Baraniuk (2018) have provided detailed analysis in the context of feedforward networks that highlights the practical value that spline operators bring to interpreting deep networks. Here, we go one step further and demonstrate the new insights and interpretations from the MASO perspective for RNNs. Below is a summary of our key contributions:

Contribution 1. We prove that RNNs with piecewise affine and convex nonlinearities and can be rewritten as composition of MASOs, making an RNN a piecewise affine spline operator that has an elegant analytical form (Section 2).

Contribution 2. We leverage the vector quantization (VQ) of piecewise affine spline operators to analyze the input space partitioning that RNN implicitly performs. We show that RNN calculates

![](images/db17b817bc83928ea50c6a5d17b417ee7e6ef33b9de367abe2df2d119582b19f.jpg)  
Figure 1: Visualization of an RNN. A cell and a layer are highlighted.

a new, high dimensional embedding of the input sequence that captures informative underlying characteristics of the input. We also provide a new perspective for RNN dynamics by visualizing the evolution of RNN VQ partitioning through time. (Section 3).

Contribution 3. We show the piecewise affine mapping in an RNN associated with a given input sequence is an input-dependent template, from which we can interpret the RNN as performing greedy matched filtering at every RNN cell. (Section 4).

Contribution 4. We rigorously prove that adding noise to the initial hidden state of an RNN corresponds to an explicit regularizer that links to exploding gradient, and empirically show that such regularization improves RNN performance on four datasets of different modalities (Section 5).

# 1.1 BACKGROUND: RECURRENT NEURAL NETWORKS (RNNS)

For concreteness, we study a specific class of simple RNNs (Elman, 1990). The RNN unit per time step  $t$  and layer  $\ell$ , referred to as a "cell", performs the following recursive computation:

$$
\boldsymbol {h} ^ {(\ell , t)} = \sigma \left(\boldsymbol {W} ^ {(\ell)} \boldsymbol {h} ^ {(\ell - 1, t)} + \boldsymbol {W} _ {r} ^ {(\ell)} \boldsymbol {h} ^ {(\ell , t - 1)} + \boldsymbol {b} ^ {(\ell)}\right), \tag {1}
$$

where  $\pmb{h}^{(\ell, t)} \in \mathbb{R}^{D^{(\ell)}}$  is the hidden state at timestep  $t$  and layer  $\ell$ ,  $\pmb{h}^{(0, t)} \coloneqq \pmb{x}^{(t)}$ ,  $\sigma$  is an activation function and  $\pmb{W}^{(\ell)}$ ,  $\pmb{W}_r^{(\ell)}$  and  $\pmb{b}^{(\ell)}$  are time-invariant model parameters. Unrolling the RNN through time gives an intuitive view of the RNN dynamics, which we visualize in Figure 1. The output of the overall RNN is usually an affine transformation of hidden state at every time step  $t$  of the last layer  $L$ :

$$
\boldsymbol {z} ^ {(t)} = \boldsymbol {W h} ^ {(L, t)} + \boldsymbol {b}. \tag {2}
$$

In the special case when the RNN has one output at the end of processing the entire input sequence  $\pmb{x}^{(1:T)}$  of length  $T$ , the RNN output is an affine transformation of the hidden state at the last time step  $T$ , i.e.,  $\pmb{z} = \pmb{z}^{(T)} = \pmb{W}\pmb{h}^{(L,T)} + \pmb{b}$ .

# 1.2 BACKGROUND: MAX-AFFINE SPLINE OPERATORS (MASOS)

A max-affine spline operator (MASO) is piecewise affine and convex with respect to each output dimension  $k = 1,\dots ,K$ . It is defined as  $S[A,B]:\mathbb{R}^D\to \mathbb{R}^K$  with parameters  $A\in \mathbb{R}^{K\times R\times D}$  and  $B\in \mathbb{R}^{K\times R}$ . This operator leverages  $K$  independent max-affine splines Magnani & Boyd (2009), each with  $R$  regions. Its output is produced via

$$
\left[ \boldsymbol {z} ^ {(\ell)} \right] _ {k} = \left[ S \left[ A ^ {(\ell)}, B ^ {(\ell)} \right] \left(\boldsymbol {z} ^ {(\ell - 1)}\right) \right] _ {k} = \max  _ {r = 1, \dots , R ^ {(\ell)}} \left(\left\langle \left[ A ^ {(\ell)} \right] _ {k, r, \cdot}, \boldsymbol {z} ^ {(\ell - 1)} \right\rangle + \left[ B ^ {(\ell)} \right] _ {k, r}\right), \tag {3}
$$

where  $[z^{(\ell)}]_k$  denotes the  $k^{\mathrm{th}}$  dimension of the vector  $z^{(\ell)}$ .

We highlight two key MASO properties relevant to the discussions in Section 3 and 4. First, MASOs perform vector quantization (VQ) to build their input partition, which is made explicit by rewriting Eq. 3 as

$$
[ \boldsymbol {z} ^ {(\ell)} ] _ {k} = \sum_ {r = 1} ^ {R ^ {(\ell)}} [ Q ^ {(\ell)} ] _ {k, r} \left(\left\langle \left[ A ^ {(\ell)} \right] _ {k, r,..}, \boldsymbol {z} ^ {(\ell - 1)} \right\rangle + \left[ B ^ {(\ell)} \right] _ {k, r}\right), \tag {4}
$$

where  $Q^{(\ell)} \in \mathbb{R}^{D^{(\ell)} \times R^{(\ell)}}$  is the VQ matrix<sup>1</sup>. The VQ matrix contains  $D^{(\ell)}$  stacked one-hot row vectors, each with the one-hot position at index  $[q^{(\ell)}]_k \in \{1, \dots, R\}$  corresponding to the arg max over  $r = 1, \dots, R^{(\ell)}$  of Eq. 3. Second, given the partition that an input belongs to, the output of MASO of each dimension  $k$  reduces to a simple linear transformation of the input with parameters corresponding to the input's partition region. This is a direct consequence of Eq. 3.

# 2 RNNS AS PIECEWISE AFFINE SPLINE OPERATORS

We now leverage the MASO framework to rewrite, interpret and analyse RNNs. We focus on RNNs with piecewise affine and convex nonlinearities to derive analytical results. Our goal of this section is to show that an RNN becomes an simple, elegant affine mapping using MASOs. The analysis of RNNs with convex, non-piecewise affine and non-convex nonlinearities are left for future work.

We start by deriving the MASO formula for an RNN cell (Eq. 1), then extend to one layer of time-unrolled RNNs and finally to multi-layer, time-unrolled RNNs. Let  $\boldsymbol{z}^{(\ell,t)} = \left[ \boldsymbol{h}^{(\ell-1,t)^{\top}}, \boldsymbol{h}^{(\ell,t-1)^{\top}} \right]^{\top}$  denote the input to an RNN cell which is the concatenation of the current input  $\boldsymbol{h}^{(\ell-1,t)}$  and the previous hidden state  $\boldsymbol{h}^{(\ell,t-1)}$ . Then we have the following result, which is a straightforward extension of Prop. 4 in Balestriero & Baraniuk (2018).

Proposition 1. An RNN cell of the form in Eq. 1 can be written as a MASO as follows:

$$
\boldsymbol {h} ^ {(\ell , t)} = A ^ {(\ell , t)} \left[ \boldsymbol {z} ^ {(\ell , t)} \right] \boldsymbol {z} ^ {(\ell , t)} + B ^ {(\ell , t)} \left[ \boldsymbol {z} ^ {(\ell , t)} \right], \tag {5}
$$

where  $A^{(\ell, t)}[z^{(\ell, t)}] = A_{\sigma}^{(\ell, t)}[z^{(\ell, t)}][W^{(\ell)}, W_r^{(\ell)}]$ ,  $B^{(t)}[z^{(\ell, t)}] = A_{\sigma}^{(\ell, t)}[z^{(\ell, t)}]b^{(\ell)}$  and  $A_{\sigma}^{(\ell, t)}[z^{(\ell, t)}]$  are the affine parameters.

To simplify notation, we drop the dependencies of the affine parameters except for the final affine parameter. We then proceed to derive the explicit affine mapping of a time-unrolled RNN at layer  $\ell$ . Let  $h^{(\ell -1,1:T)} = \left[h^{(\ell -1,1)^{\top}},\dots ,x^{(\ell -1,T)^{\top}}\right]^{\top}$  be the entire input sequence to the RNN at layer  $\ell$  and  $h^{(\ell ,1:T)} = \left[h^{(\ell ,1)^{\top}},\dots ,x^{(\ell ,T)^{\top}}\right]^{\top}$  be the hidden states of all time steps which are the outputs of the RNN at layer  $\ell$ . With some algebra and simplification, we arrive at the following result.

Theorem 1. The  $\ell^{\mathrm{th}}$  layer of an RNN is a piecewise affine mapping defined as follows:

$$
\begin{array}{l} \boldsymbol {h} ^ {(\ell , 1: \underline {{T}})} = \left( \begin{array}{c c c} \mathcal {A} _ {T: T} ^ {(\ell)} & \dots & \mathcal {A} _ {1: T} ^ {(\ell)} \\ \vdots & \ddots & \vdots \\ \boldsymbol {0} & \dots & \mathcal {A} _ {1: 1} ^ {(\ell)} \end{array} \right) \left( \begin{array}{c c c} A _ {\sigma} ^ {(\ell , T)} \boldsymbol {W} ^ {(\ell)} & \dots & \boldsymbol {0} \\ \vdots & \ddots & \vdots \\ \boldsymbol {0} & \dots & A _ {\sigma} ^ {(\ell , 1)} \boldsymbol {W} ^ {(\ell)} \end{array} \right) \left( \begin{array}{c} \boldsymbol {h} ^ {(\ell - 1, T)} \\ \vdots \\ \boldsymbol {h} ^ {(\ell - 1, 1)} \end{array} \right) + \left( \begin{array}{c} \sum_ {t = T} ^ {1} \mathcal {A} _ {t: T} ^ {(\ell)} B ^ {(\ell , t)} + \mathcal {A} _ {0: T} ^ {(\ell)} \boldsymbol {h} ^ {(\ell , 0)} \\ \vdots \\ \mathcal {A} _ {1: 1} ^ {(\ell)} B ^ {(\ell , t)} + \mathcal {A} _ {0: 1} ^ {(\ell)} \boldsymbol {h} ^ {(\ell , 0)} \end{array} \right) \\ = A _ {\mathrm {R N N}} ^ {(\ell)} \left[ \boldsymbol {h} ^ {(\ell - 1, 1: T)}, \boldsymbol {h} ^ {(\ell , 0)} \right] \boldsymbol {h} ^ {(\ell - 1, 1: T)} + B _ {\mathrm {R N N}} ^ {(\ell)} \left[ \boldsymbol {h} ^ {(\ell - 1, 1: T)}, \boldsymbol {h} ^ {(\ell , 0)} \right], \tag {6} \\ \end{array}
$$

where  $\mathcal{A}_{t:T'}^{(\ell)} = \left(\prod_{s=T'}^{t+1} A_{\sigma}^{(\ell,s)} \mathbf{W}_r^{(\ell)}\right)$  for  $t < T'$  and identity otherwise,  $\pmb{h}^{(\ell,0)}$  is the initial hidden state of the RNN at layer  $\ell$  and  $A_{\mathrm{RNN}}^{(\ell)}$ $[\pmb{h}^{(\ell-1,1:T)}, \pmb{h}^{(\ell,0)}]$  and  $B_{\mathrm{RNN}}^{(\ell)}$ $[\pmb{h}^{(\ell-1,1:T)}, \pmb{h}^{(\ell,0)}]$  are affine parameters that depend on  $\pmb{h}^{(\ell-1,1:T)}$  and  $\pmb{h}^{(\ell,0)}$ .

We present the proof for Thm. 1 in Appendix F. The key point here is that, by leveraging MASOs, we can represent the time-unrolled RNN as a simple affine mapping from the entire input sequence into the  $\ell^{\mathrm{th}}$  layer hidden states of all time steps (Eq. 6). Note also that the initial hidden state affects the layer output by influencing the affine parameters and contributing a bias term  $A_{0:t}^{(\ell)}\pmb{h}^{(\ell,0)}$  to the bias term  $B_{\mathrm{RNN}}^{(\ell)}\left[\pmb{h}^{(\ell-1,1:T)},\pmb{h}^{(\ell,0)}\right]$  We study the impact of  $\pmb{h}^{(0)}$  in more detail in Section 5.

We are now ready to generalize the above result to multi-layer RNNs. Let  $\pmb{h}^{(0,1:T)} = \pmb{x}^{(1:T)} = \left[\pmb{x}^{(1)^{\top}},\dots ,\pmb{x}^{(T)^{\top}}\right]^{\top}$  denote the input sequence to the multi-layer RNN and and  $\pmb{z}^{(1:T)} =$

$\left[\boldsymbol{z}^{(1)^{\top}}, \dots, \boldsymbol{z}^{(T)^{\top}}\right]^{\top}$  the output sequence. We state the following result for the overall mapping of a multi-layer RNN.

Theorem 2. An RNN of  $L$  layers is a piecewise affine spline operator defined as follows:

$$
\begin{array}{l} \boldsymbol {z} ^ {(1: T)} = f \left(\boldsymbol {x} ^ {(1: T)}, \left\{\boldsymbol {h} ^ {(\ell , 0)} \right\} _ {\ell = 1} ^ {L}\right) \\ = \mathcal {W} \underbrace {\left(A _ {\mathrm {R N N}} \left[ \boldsymbol {x} ^ {(1 : T)} , \left\{\boldsymbol {h} ^ {(\ell , 0)} \right\} _ {\ell = 1} ^ {L} \right] \boldsymbol {x} ^ {(1 : T)} + B _ {\mathrm {R N N}} \left[ \boldsymbol {x} ^ {(1 : T)} , \left\{\boldsymbol {h} ^ {(\ell , 0)} \right\} _ {\ell = 1} ^ {L} \right]\right)} _ {\text {R N N f o r m u l a}} + \boldsymbol {b}, \tag {7} \\ \end{array}
$$

where  $A_{\mathrm{RNN}}\left[\pmb{x}^{(1:T)}, \{\pmb{h}^{(\ell,0)}\}_{\ell=1}^L\right] = \prod_{\ell=L}^{1} A_{\mathrm{RNN}}^{(\ell)}\left[\pmb{x}^{(1:T)}, \pmb{h}^{(\ell,0)}\right]$  and

$B_{\mathrm{RNN}}\left[\pmb{x}^{(1:T)}, \{\pmb{h}^{(\ell,0)}\}_{\ell=1}^{L}\right] = \sum_{\ell=1}^{L}\left(\prod_{\ell'= \ell}^{L-1} A_{\mathrm{RNN}}^{(\ell)}\left[\pmb{x}^{(1:T)}, \pmb{h}^{(\ell,0)}\right]\right) B_{\mathrm{RNN}}^{(\ell)}\left[\pmb{x}^{(1:T)}, \pmb{h}^{(\ell,0)}\right]$

are the affine parameters of the entire RNN.  $\mathcal{W}$  and  $\pmb{b}$  are parameters of the output fully connected layer where  $\mathcal{W} = [W, W, \dots, W]$  when the RNN outputs at every time step and  $\mathcal{W} = [W, 0, \dots, 0]$  when the RNN outputs only at the last time step.

Thm. 2 shows that, using MASOs, we have a simple, elegant affine mapping between the input and output sequence of a multi-layer RNN, and that the output is computed via locally very simple functions. We now leverage the affine formula of RNN and apply it to visualize and understand RNNs. First, we analyze and visualize the VQ partitioning that the RNN forms over time. Second, we analyze the form of the affine parameters and link them to matched filtering. Third, we study the impact of initial hidden state and justify the use of noise in initial hidden state.

# 3 APPLICATION: INTERNAL INPUT SPACE PARTIONING IN RNNS

In this section, we provide a new perspective of the dynamics of an RNN using its affine formula. For simplicity, we assume here that initial hidden states are set to 0. In the previous section, we denote the input region dependency in the affine parameters as  $A^{(\ell)} \left[ \boldsymbol{x}^{(1:T)} \right]_{\mathrm{RNN}}$  and  $B^{(\ell)} \left[ \boldsymbol{x}^{(1:T)} \right]_{\mathrm{RNN}}$ . We now make region dependency explicit by introducing the VQ tensor (from Eq. 4) to the RNN case:

$$
Q \left(\boldsymbol {x} ^ {(1: T)}\right) = \left( \begin{array}{c c c} Q ^ {(L, T)} \left(\boldsymbol {x} ^ {(1: T)}\right) & \dots & Q ^ {(L, 1)} \left(\boldsymbol {x} ^ {(1: 1)}\right) \\ \vdots & \ddots & \vdots \\ Q ^ {(1, T)} \left(\boldsymbol {x} ^ {(1: T)}\right) & \dots & Q ^ {(1, 1)} \left(\boldsymbol {x} ^ {(1: 1)}\right) \end{array} \right), \tag {8}
$$

where each  $Q^{(\ell, t)} \left( \boldsymbol{x}^{(1:t)} \right)$  is the VQ matrix as presented in Eq. 4 and applied to the MASO formula of the RNN cell in Prop. 1. Thus, each  $Q^{(\ell, t)} \left( \boldsymbol{x}^{(1:t)} \right)$  is the partition of the truncated input timeseries  $\boldsymbol{x}^{(1:t)}$  from the beginning to the current time step  $t$ . We can then view the RNN as developing a partition for an input sequence through time, receiving new input at each time step and refining its partitioning. We can see this from the form of the VQ tensor in Eq. 8, which accumulates local VQs from right to left through time.

We demonstrate the development of RNN input space partitioning using a one layer ReLU RNN trained on MNIST dataset as an example. We flatten each image into a 1-dimensional sequence of length  $T = 784$ . Details of model and experiments are in Appendix B. For ReLU, the VQ for each input time-serie is simply the concatenation of binarized hidden states since ReLU induces 2 partition regions. Figure 2 visualizes the input space partitioning of the MNIST test set using tSNE<sup>3</sup> for various time steps. The figure clearly shows the evolution of the RNN input space partitioning, from hardly any separation to forming clear clusters through time. Many other visualizations on the RNN input space partitioning are available in Section C.

# 4 APPLICATION: RNNS AS A MATCHED FILTER BANK

In this section, we provide another insight into the RNN computation from the matched filtering perspective. We again assume zero initial hidden state for simplicity. Combining Eqs. 3 and 5, we

![](images/0b622baf5e1a9b2c091233245f097de1ace88d6f889fcdf34e394d2ecfe012d9.jpg)

![](images/2154fcfe864d40e4e15033fe71ea900fae3faef1a735951a7929f572d80dcfd7.jpg)

![](images/7c801237d06cce15df5b0c108f5a16fea60cf5040366afa18cb1b15b4e152c55.jpg)  
Figure 2: tSNE visualization of the evolution of RNN VQ partitioning on MNIST test set. This visualization is enabled by the new parametrization that the RNN computes using VQ. Each color represents one class. We can see that RNN gradually refines the VQ partitioning through time. As the VQ is built through the time steps, as the classes become

![](images/17fbdc1f1f936fbad1a26304666cf814a53c18b779750f5b53c3e8d010999b58.jpg)

see that the computation of an RNN cell is equivalently the maximization of the affine transformation of the input at the current time step. Thus, we can obtain the locally optimal template, i.e., the linear map  $A^{(\ell ,t)}\left[z^{(1:T)}\right]_{\mathrm{RNN}}$  in the affine parameters, by finding the one that produces the largest inner product with the input. For RNNs that produce a single output at the last time step, the overall matched filter bank  $A_{\mathrm{RNN}}\left[\boldsymbol{x}^{(1:T)}\right]$  then corresponds to the composition of optimal matched obtained at each RNN cell and can be computed simply via  $dz / dx^{(1:T)}$ . Thus, we can view the RNN as a matched filter bank that computes the output by finding the maximum inner product between the overall template and the input.

The overall template is also known in the machine learning community as salience map, the visual examination of which is helpful for qualitatively diagnosing a model. For RNNs, salience map visualization has been explored previously by Li et al. (2016). Our insight and emphasis here is that a good template produces a larger inner product with the input regardless of the visual quality of the template. A template matching view of RNNs thus provide a quantitative model diagnosis by examining the inner products between inputs and templates.

As an example, we train a one layer ReLU RNN on the polarized Stanford Sentiment Treebank (SST-2) (Socher et al., 2013), which is a binary classification problem, and show in Figure 3 both correct and incorrect class templates of an input of negative sentiment. We see that the input has a much bigger inner product with the correct class template (left plot) than the incorrect class template (right plot), which demonstrates the template matching that the RNN performs. Additional experimental results are in Appendix D.

# 5 APPLICATION: IMPROVING RNN PERFORMANCE VIA NOISY INITIAL HIDDEN STATE

In this section, we provide theoretical insights on the use of noise in initial hidden state. Little is understood about setting the value for the initial hidden state except for Zimmermann et al. (2012)'s argument of using noisy initial hidden state from a dynamical system's perspective. Therefore the initial hidden state is typically set to zero without much justification. Leveraging the MASO formula of RNNs, we prove that noisy initial hidden state is equivalent to explicit regularization of the affine parameter associated with the initial hidden state.

# 5.1 NOISY INITIAL HIDDEN STATE AS EXPLICIT REGULARIZER

Let  $\widetilde{\mathcal{L}}$  denote the loss with noise injected to the initial hidden state. We first state the theoretical result for one layer RNN with one output  $z$  at the last hidden state, and thus remove the layer index  $\ell$  to simplify notation. We then link our results to the exploding gradient problem.

![](images/1e54da772a9edef4d6cf17365180244a9f0d2a53c4e0b85dcf5822c2f6785861.jpg)  
Negative class, inner product  $= 2.409$

![](images/e8e853bd4ddbf94dde4a8d814ec1d9d0cfa6fbf700b5e512a58dfecbfc550407.jpg)  
Positive (correct) class, inner product  $= -0.390$  
Figure 3: Templates of an example from the SST-2 dataset. Quantitatively, we can see that the inner product between input and the correct class template (left) produces a bigger value than that between input and the incorrect class template (right), which demonstrate the template matching that an RNN performs. The visual quality of the correct class template is also telling since semantically meaningful words have noticeable values whereas the incorrect class template does not show any noticeable value.

Theorem 3. Let  $\epsilon \sim \mathcal{N}(0, \sigma_{\epsilon}^{2}\pmb{I})$  be a Gaussian noise where  $\sigma_{\epsilon}^{2}$  is small. We have that

$$
\mathbb {E} _ {\epsilon} \left[ \widetilde {\mathcal {L}} \right] = \mathcal {L} + \mathcal {R}, \tag {9}
$$

where  $\mathcal{R} = \frac{\sigma_{\epsilon}^{2}}{2N}\sum_{n = 1}^{N}\left\| \mathrm{diag}\left(\left[\frac{d\widehat{y}_{ni}}{\partial z_{nj}}\right]_{i = j}\right)\mathcal{A}_{h}\left[\pmb{x}_{n}^{(1:T)}\right]\right\|^{2}$  for cross entropy loss with softmax output and  $\mathcal{R} = \frac{\sigma_{\epsilon}^{2}}{2N}\sum_{n = 1}^{N}\left\| \mathcal{A}_{h}\left[\pmb{x}_{n}^{(1:T)}\right]\right\|^{2}$  for mean squared error loss.  $\mathcal{A}_h\left[\pmb{x}_n^{(1:T)}\right] = \mathcal{A}_{1:T}\left[\pmb{x}_n^{(1:T)}\right] =$ $\prod_{s = T}^{1}A_{\sigma}^{(\ell ,s)}\left[\pmb{x}_n^{(1:T)}\right]\pmb{W}_r^{(\ell)}$  and  $i,j\in \{1,\dots ,C\}$  are the class index where  $C$  is the total number of classes.

We prove for the cross entropy case in Appendix F.2. We can see that the noise standard deviation  $\sigma_{\epsilon}$  controls the importance of the regularization term and recovers the standard case for  $\sigma_{\epsilon} = 0$ . Additionally, the regularizer does not depend on the accuracy of the classifier, which is a regularization scheme on the model itself (Wager et al., 2013).

# 5.2 INTERPRETATION AND CONNECTIONS TO PRIOR WORK

We now propose to understand the implication of this finding by demonstrating below how this relates to the problem of exploding gradient in RNN. Closely inspecting the form of  $\mathcal{A}_h$ , we see that this term is the gradient of the RNN output with respect to the initial hidden state. In fact, the gradient of the RNN output with respect to hidden state at any time step takes this form:

$$
\frac {\partial \mathcal {L}}{\partial \boldsymbol {h} ^ {(t)}} = \frac {\partial \mathcal {L}}{\partial \boldsymbol {z}} \frac {\partial \boldsymbol {z}}{\partial \boldsymbol {h} ^ {(T)}} \left(\prod_ {s = t} ^ {T} \frac {\partial \boldsymbol {h} ^ {(T)}}{\partial \boldsymbol {h} ^ {(s)}}\right) = \frac {\partial \mathcal {L}}{\partial \boldsymbol {z}} \boldsymbol {W} \left(\prod_ {s = t} ^ {T} \boldsymbol {D} _ {\sigma} ^ {(s)} \boldsymbol {W} _ {r}\right), \tag {10}
$$

which is the basis of studying gradient exploding problems in RNNs, since it is clear that constraining this term will help limit the gradient in the backward pass. A more detailed review of exploding gradient problem is provided in Appendix G; We note here that studying this term has inspired a number of works on the unitary or orthogonal parametrization of the recurrent weight in order to maintain unitarity of the matrix multiplication in Eq. 10 (Arjovsky et al., 2016; Wisdom et al., 2016; Helfrich et al., 2018; Jing et al., 2017; Hyland & Ratsch, 2017). By introducing noise to the initial hidden state, we regularize the term with the most number of matrix multiplications in the gradient calculation of the recurrent weight. While whether regularizing this term amounts to regularizing every other term in the recurrent weight gradient calculation remains to be investigated, we provide empirical evidence that the regularization introduced by noisy initial hidden state already reduces the magnitude of the recurrent weight gradient and improves model performance.

![](images/4365b8fd728b586d3230ca626336eeed9fe877f2cc64ac4033daf8e4bf4868a8.jpg)  
Figure 4: Visualization of the regularization effect on the adding task ( $T = 100$ ). Top: norm of  $\mathbf{A_h}$  at every 100 iterations; Middle: norm of gradient of recurrent weight at every 100 iterations; Bottom: validation loss at every epoch. Each epoch contains 1000 iterations.

# 5.3 EXPERIMENTS

We provide extensive empirical evidence to advocate for the use of noisy initial hidden state and demonstrate that a properly chosen noise standard deviation  $\sigma_{\epsilon}$  improves performance for all the datasets that we experiment on. Unless otherwise mentioned, we use identity-initialized ReLU RNNs of 128-dimensional hidden state (denoted as RNN) in all our experiments. We summarize in this section the gist of our experimental results; Details on models, datasets, experiment setup and additional results are available in Appendix B and E.

Visualizing the Regularization Effect of the Noisy Initial Hidden State. We first visualize the regularization effect when injecting noise into the initial hidden state on a simulation task of adding 2 sequences of length 100. This is a ternary classification problem with input  $\mathbf{X} \in \mathbb{R}^{2 \times T}$  and target  $y \in \{0,1,2\}$ ,  $y = \sum_{i} \mathbb{1}_{\mathbf{X}_{2i} = 1} \cdot \mathbf{X}_{1i}$ . The first row of  $\mathbf{X}$  contains randomly chosen 0's and 1's, and the second row of  $\mathbf{X}$  contains 1's at 2 randomly chosen indices and 0's everywhere else. Note that prior work treat this task as a regression task instead (Arjovsky et al., 2016). Results for treating this task as a regression problem is provided in Appendix E.1.

In Figure 4, we visualize the norm of  $\mathcal{A}_h$ , the norm of recurrent weight gradient and validation loss against training iterations/epochs for various noise standard deviations. The top two plots clearly demonstrate the regularization effect of noisy initial hidden state in regularize both  $\mathcal{A}_h$  and norm of the recurrent weight gradient, since larger  $\sigma_{\epsilon}$  reduces the magnitudes of both  $A_h$  and  $\frac{d\mathcal{L}}{dW_r}$ . Notably, this regularization effect happens even when the regularizer does not directly regularize these two terms, which validates our intuition. The plot in the last row shows that setting a big  $\sigma_{\epsilon}$  can negatively impact learning. This can be explained by having too much regularization effect with a large  $\sigma_{\epsilon}$ . This brings the question of choosing  $\sigma_{\epsilon}$  in practice, which we investigate below.

Choosing the Noise Parameter. We examine the performance of different noise standard deviations using RMSprop and SGD with varying learning rates to provide insights on choosing the noise parameter  $\sigma_{\epsilon}$ . We perform experiments on MNIST dataset, each image flattened to a length 784 sequence (recall Section 3). Experimental results are included in Appendix E.2. Here, we report two interesting findings. First, for both optimizers, using noisy initial hidden state permits the use of higher learning rates that lead to exploding gradient when training model without noisy initial hidden state. Second, RMSprop is more tolerable to the choice of  $\sigma_{\epsilon}$  than SGD and achieves favorable accuracies even when  $\sigma_{\epsilon}$  is very large (e.g.,  $\sigma_{\epsilon} = 5$ ). This might be due to the gradient smoothing that RMSprop performs during optimization. We therefore recommend the use of RMSprop with noisy initial hidden state to improve model performance.

We then use RMSprop to train ReLU RNNs of one or two layers and with or without noisy initial hidden state on MNIST, permuted MNIST $^4$  and SST-2 datasets. Table 1 shows the classification accuracies of these models as well as a few state-of-the-art results. These results further suggest

Table 1: Test set classification accuracies on MNIST and permuted MNIST datasets for various models. We see that using noise in the simplest RNNs elevates these RNNs to strong competitors of many complex, state-of-the-art models.  

<table><tr><td rowspan="2">Model</td><td colspan="3">Dataset</td></tr><tr><td>MNIST</td><td>permuted MNIST</td><td>SST-2</td></tr><tr><td>RNN, 1 layer</td><td>0.970</td><td>0.891</td><td>0.871</td></tr><tr><td>RNN, 1 layer, noise</td><td>0.981(σε=0.1)</td><td>0.922(σε=0.01)</td><td>0.873(σε=0.1)</td></tr><tr><td>RNN, 2 layer</td><td>0.969</td><td>0.873</td><td>0.884</td></tr><tr><td>RNN, 2 layer, noise</td><td>0.987(σε=0.5)</td><td>0.927(σε=0.005)</td><td>0.888(σε=0.005)</td></tr><tr><td>LSTM (Wisdom et al., 2016)</td><td>0.978</td><td>0.913</td><td>0.849</td></tr><tr><td>uRNN (Arjovsky et al., 2016)</td><td>0.951</td><td>0.914</td><td>-</td></tr><tr><td>scoRNN (Helfrich et al., 2018)</td><td>0.985</td><td>0.966</td><td>-</td></tr><tr><td>C-LSTM (Zhou et al., 2015)</td><td>-</td><td>-</td><td>0.878</td></tr><tr><td>Tree-LSTM (Tai et al., 2015)</td><td>-</td><td>-</td><td>0.88</td></tr><tr><td>Bi-LSTM+SWN-Lex (Teng et al., 2016)</td><td>-</td><td>-</td><td>0.892</td></tr></table>

the use of noisy initial hidden state in improving performance of ReLU RNN, elevating such simple model to a strong competitor of a number of more complicated, state-of-the-art models.

Noisy Initial Hidden State in Complex Models. We extend to RNNs with non-piecewise affine nonlinearities and provide promising preliminary results on the use of initial hidden state in complex models. We perform an experiment on the bird audio dataset; details of the dataset is available in Appendix B. This dataset involves a binary classification problem to detect whether or not an audio recording contain bird songs. We use the area under the curve (AUC) as the evaluation metric, since the dataset is highly imbalanced.

We implement and train a convolutional-recurrent model composed of 4 convolution layers followed by 2 gated recurrent unites  $(\mathrm{GRU})^5$ , with and without noise added to the initial hidden state in both layers of the GRU. Experimental results show that, without added noise, we achieve  $91\%$  test AUC; With added noise, we achieve  $93\%$  test AUC, which is the new state-of-the-art on this task. This results implies that noisy hidden state has the potential to improve performance even for complex models with non-piecewise affine nonlinearities such as GRUs.

# 6 CONCLUSIONS AND FUTURE WORK

We have provided a novel perspective of understanding RNNs from the use of max-affine spline operators (MASOs). With MASOs, RNNs with piecewise affine and convex nonlinearities become a simple, elegant affine spline operator. Leveraging this formulation, we provide new insights and visualizations from vector quantization and matched filtering perspectives. Furthermore, we show that injecting noise in the initial hidden state of an RNN corresponds to an explicit regularization, which improves generalization by alleviating the exploding gradient problem. Extensive empirical studies of the use of noisy initial hidden state in vanilla ReLU RNNs and more complex models on various datasets have demonstrated the promise of using noisy initial state in improving RNN performance.

The MASO framework opens a new door for understanding RNNs. In this paper, we merely scratched the surface of RNNs from a MASO perspective. For example, the analyses presented in this paper are only applicable to piecewise affine and convex nonlinearities. We will extend our analyses to RNNs with noncontext and non-piecewise nonlinearities such LSTMs and GRUs that are more widely applicable in practice. We will also study RNNs under different problem settings such as sequence generation which outputs at every time step. These future analyses that leverage MASOs have the potential to provide new and practical insights into the inner workings of other RNN categories.

# REFERENCES

M. Arjovsky, A. Shah, and Y. Bengio. Unitary evolution recurrent neural networks. In Proc. Int. Conf. Mach. Learn., volume 48, pp. 1120-1128, Jun. 2016.  
D. Bahdanau, K. Cho, and Y. Bengio. Neural Machine Translation by Jointly Learning to Align and Translate. *ArXiv e-prints*, September 2014.  
R. Balestriero and R. G. Baraniuk. Mad max: Affine spline insights into deep learning. *ArXiv e-prints*, May 2018.  
R. Balestriero and R. G. Baraniuk. A spline theory of deep networks. In Proc. Int. Conf. Mach. Learn., volume 80, pp. 374-383, Jul 2018.  
E. Cakir, S. Adavanne, G. Parascandolo, K. Drossos, and T. Virtanen. Convolutional recurrent neural networks for bird audio detection. In European Signal Processing Conference (EUSIPCO), pp. 1744-1748, Aug 2017.  
T. Cooijmans, N. Ballas, C. Laurent, C. Gulcehre, and A. C. Courville. Recurrent batch normalization. In Proc. International Conference on Learning Representations, Apr. 2017.  
A. Dieng, R. Ranganath, J. Altosaar, and D. Blei. Noisin: Unbiased regularization for recurrent neural networks. In Proc. International Conference on Machine Learning, volume 80, pp. 1252-1261, Jul. 2018.  
J. L. Elman. Finding structure in time. Cogn. Sci., 14:179-211, 1990.  
B. Hammer. On the approximation capability of recurrent neural networks. Neurocomputing, 31(1): 107-123, Mar. 2000.  
L. A. Hannah and D. B. Dunson. Multivariate convex regression with adaptive partitioning. J. Mach. Learn. Res., 14:3261-3294, 2013.  
K. Helfrich, D. Willmott, and Q. Ye. Orthogonal recurrent neural networks with scaled Cayley transform. In Proc. Int. Conf. Mach. Learn., volume 80, pp. 1969-1978, Jul. 2018.  
M. Henaff, A. Szlam, and Y. LeCun. Recurrent orthogonal networks and long-memory tasks. In Proc. International Conference on Machine Learning, volume 48, pp. 2034-2042, Jun. 2016.  
S. L. Hyland and G. Ratsch. Learning unitary operators with help from u (n). In Proc. AAAI conference on Artificial Intelligence, pp. 2050-2058, Feb. 2017.  
L. Jing, Y. Shen, T. Dubcek, J. Peurifoy, S. Skirlo, Y. LeCun, M. Tegmark, and M. Soljacic. Tunable efficient unitary neural networks (EUNN) and their application to RNNs. In Proc. International Conference on Machine Learning, volume 70, pp. 1733-1741, Aug. 2017.  
C. Jose, M. Cisse, and F. Fleuret. Kronecker recurrent units. In Proc. International Conference on Learning Representations, Apr. 2018.  
D Krueger, T. Maharaj, J. Kramar, M. Pezeshki, N. Ballas, N. R. Ke, A. Goyal, Y. Bengio, H. Larochelle, A. C. Courville, and C. Pal. Zoneout: Regularizing rnns by randomly preserving hidden activations. In Proc. International Conference on Learning Representations, Apr. 2017.  
Q. V. Le, N. Jaitly, and G. E. Hinton. A Simple Way to Initialize Recurrent Networks of Rectified Linear Units. ArXiv e-prints, Apr. 2015.  
J. Li, X. Chen, E. Hovy, and D. Jurafsky. Visualizing and understanding neural models in NLP. In Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Language Technol., pp. 681-691, Jun. 2016.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(Nov):2579-2605, 2008.  
A. Magnani and S. P. Boyd. Convex piecewise-linear fitting. Optimization Eng., 10(1):1-17, Mar. 2009.

Z. Mhammedi, A. Hellicar, A. Rahman, and J. Bailey. Efficient orthogonal parametrisation of recurrent neural networks using householder reflections. In Proc. International Conference on Machine Learning, volume 70, pp. 2401-2409, Aug. 2017.  
R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. In Proc. Int. Conf. Mach. Learn., pp. 1310-1318, Jun. 2013.  
J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In Proc. Conf. Empirical Methods Natural Language Proc. (EMNLP), pp. 1532-1543, Oct. 2014.  
V. Pham, T. Bluche, C. Kermorvant, and J. Louradour. Dropout improves recurrent neural networks for handwriting recognition. In Proc. International Conference on Frontiers in Handwriting Recognition, pp. 285-290, Sept. 2014.  
A. M. Schäfer and H. G. Zimmermann. Recurrent neural networks are universal approximators. In Proc. Int. Conf. Artificial Neural Netw., pp. 632-640, Sept. 2006.  
H. T. Siegelmann and E. D. Sontag. On the computational power of neural nets. J. Comput. Syst. Sci., 50(1):132-150, Feb. 1995.  
R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proc. Conf. Empirical Methods Natural Language Proc. (EMNLP), pp. 1631-1642, Oct. 2013.  
D. Stowell and M. D. Plumbey. An open dataset for research on audio field recording archives: Freefield1010. In Proc. Audio Eng. Soc. 53rd Conf. Semantic Audio (AES53), pp. 1-7, 2014.  
Kai Sheng Tai, Richard Socher, and Christopher D. Manning. Improved semantic representations from tree-structured long short-term memory networks. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1556–1566, Jul. 2015.  
S. S. Talathi and A. Vartak. Improving performance of recurrent neural network with relu nonlinearity. In Proc. International Conference on Learning Representations, May 2016.  
Z. Teng, D. T. Vo, and Y. Zhang. Context-sensitive lexicon features for neural sentiment analysis. In Proc. of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1629-1638, Nov. 2016.  
S. Wager, S. Wang, and P. Liang. Dropout training as adaptive regularization. In Proc. Advances Neural Inform. Process. Syst., volume 1, pp. 351-359, Dec. 2013.  
S. Wisdom, T. Powers, J. R. Hershey, J. Le Roux, and L. Atlas. Full-capacity unitary recurrent neural networks. In Proc. Advances Neural Inform. Process. Syst., Dec. 2016.  
W. Zaremba, I. Sutskever, and O. Vinyals. Recurrent Neural Network Regularization. ArXiv e-prints, 2014.  
C. Zhou, C. Sun, Z. Liu, and F. C. M. Lau. A C-LSTM Neural Network for Text Classification. ArXiv e-prints, November 2015.  
H. Zimmermann, C. Tietz, and R. Grothmann. Forecasting with recurrent neural networks: 12 tricks. Neural Networks: Tricks of the Trade: Second Edition, pp. 687-707, 2012.
