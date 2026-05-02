# HYPERNETWORKS

David Ha\* Andrew Dai, Quoc V. Le

Google Brain

{hadavid, adai, qvl}@google.com

# ABSTRACT

This work explores hypernetworks: an approach of using a one network, also known as a hypernetwork, to generate the weights for another network. Hypernetworks provide an abstraction that is similar to what is found in nature: the relationship between a genotype – the hypernetwork – and a phenotype – the main network. Though they are also reminiscent of HyperNEAT in evolution, our hypernetworks are trained end-to-end with backpropagation and thus are usually faster. The focus of this work is to make hypernetworks useful for deep convolutional networks and long recurrent networks, where hypernetworks can be viewed as relaxed form of weight-sharing across layers. Our main result is that hypernetworks can generate non-shared weights for LSTM and achieve near state-of-the-art results on a variety of sequence modelling tasks including character-level language modelling, handwriting generation and neural machine translation, challenging the weight-sharing paradigm for recurrent networks. Our results also show that hypernetworks applied to convolutional networks still achieve respectable results for image recognition tasks compared to state-of-the-art baseline models while requiring fewer learnable parameters.

# 1 INTRODUCTION

In this work, we consider an approach of using a small network (called a "hypernetwork") to generate the weights for a larger network (called a main network). The behavior of the main network is the same with any usual neural network: it learns to map some raw inputs to their desired targets; whereas the hypernetwork takes a set of inputs that contain information about the structure of the weights and generates the weight for that layer (see Figure 1).

![](images/40c5ad1165546489c6aa0791c0224c266018498c5f39a1c7945e30a7ce6ccd9e.jpg)  
Figure 1: A hypernetwork generates the weights for a feedforward network. Black connections and parameters are associated with the main network whereas orange connections and parameters are associated with the hypernetwork.

HyperNEAT (Stanley et al., 2009) is an example of hypernetworks where the inputs are a set of virtual coordinates for each weight in the main network. In this work, we will focus on a more powerful approach where the input is an embedding vector that describes the entire weights of a given layer. Our embedding vectors can be fixed parameters that are also learned during end-to-end training, allowing approximate weight-sharing within a layer and across layers of the main network. In

addition, our embedding vectors can also be generated dynamically by our hypernetwork, allowing the weights of a recurrent network to change over timesteps and also adapt to the input sequence.

We perform experiments to investigate the behaviors of hypernetworks in a range of contexts and find that hypernetworks mix well with other techniques such as batch normalization and layer normalization. Our main result is that hypernetworks can generate non-shared weights for LSTM that work better than the standard version of LSTM (Hochreiter & Schmidhuber, 1997). On language modelling tasks with Character Penn Treebank, Hutter Prize Wikipedia datasets, hypernetworks for LSTM achieve near state-of-the-art results. On a handwriting generation task with IAM handwriting dataset, Hypernetworks for LSTM achieves high quantitative and qualitative results. On image classification with CIFAR-10, hypernetworks, when being used to generate weights for a deep convnet (LeCun et al., 1990), obtain respectable results compared to state-of-the-art models while having fewer learnable parameters. In addition to simple tasks, we show that Hypernetworks for LSTM offers an increase in performance for large, production-level neural machine translation models.

# 2 MOTIVATION AND RELATED WORK

Our approach is inspired by methods in evolutionary computing, where it is difficult to directly operate in large search spaces consisting of millions of weight parameters. A more efficient method is to evolve a smaller network to generate the structure of weights for a larger network, so that the search is constrained within the much smaller weight space. An instance of this approach is the work on the HyperNEAT framework (Stanley et al., 2009). In the HyperNEAT framework, Compositional Pattern-Producing Networks (CPPNs) are evolved to define the weight structure of much larger main network. Closely related to our approach is a simplified variation of HyperNEAT, where the structure is fixed and the weights are evolved through Discrete Cosine Transform (DCT) is called Compressed Weight Search (Koutnik et al., 2010). Even more closely related to our approach are Differentiable Pattern Producing Networks (DPPNs), where the structure is evolved but the weights are learned (Fernando et al., 2016), and ACDC-Networks (Moczulski et al., 2015), where linear layers are compressed with DCT and the parameters are learned.

Most reported results using these methods, however, are in small scales, perhaps because they are both slow to train and require heuristics to be efficient. The main difference between our approach and HyperNEAT is that hypernetworks in our approach are trained end-to-end with gradient descent together with the main network, and therefore are more efficient.

In addition to end-to-end learning with gradient descent, our approach strikes a good balance between Compressed Weight Search and HyperNEAT in terms of model flexibility and training simplicity. First, it can be argued that Discrete Cosine Transform used in Compressed Weight Search may be too simple and using the DCT prior may not be suitable for many problems. Second, even though HyperNEAT is more flexible, evolving both the architecture and the weights in HyperNEAT is often an overkill for most practical problems.

Even before the work on HyperNEAT and DCT, Schmidhuber (1992; 1993) has suggested the concept of fast weights in which one network can produce context-dependent weight changes for a second network. Small scale experiments were conducted to demonstrate fast weights for feed forward networks at the time, but perhaps due to the lack of modern computational tools, the recurrent network version was mentioned mainly as a thought experiment (Schmidhuber, 1993). A subsequent work demonstrated practical applications of fast weights (Gomez & Schmidhuber, 2005), where a generator network is learnt through evolution to solve an artificial control problem. The concept of a network interacting with another network is central to the work of (Jaderberg et al., 2016; Andrychowicz et al., 2016), and especially (Denil et al., 2013; Yang et al., 2015; Bertinetto et al., 2016; De Brabandere et al., 2016), where certain parameters in a convolutional network are predicted by another network. These studies however did not explore the use of this approach to recurrent networks, which is a main contribution of our work.

The focus of this work is to generate weights for practical architectures, such as convolutional networks and recurrent networks by taking layer embedding vectors as inputs. However, our hypernetworks can also be utilized to generate weights for a fully connected network by taking coordinate information as inputs similar to DPPNs. Using this setting, hypernetworks can approximately re

cover the convolutional architecture without explicitly being told to do so, a similar result obtained by "Convolution by Evolution" (Fernando et al., 2016). This result is described in Appendix A.1.

# 3 METHODS

In this paper, we view convolutional networks and recurrent networks as two ends of a spectrum. On one end, recurrent networks can be seen as imposing weight-sharing across layers, which makes them inflexible and difficult to learn due to vanishing gradient. On the other end, convolutional networks enjoy the flexibility of not having weight-sharing, at the expense of having redundant parameters when the networks are deep. Hypernetworks can be seen as a form of relaxed weight-sharing, and therefore strikes a balance between the two ends. See Appendix A.2 for conceptual diagrams of Static and Dynamic Hypernetworks.

# 3.1 STATIC HYPERNETWORK: A WEIGHT FACTORIZATION APPROACH FOR DEEP CONVOLUTIONAL NETWORKS

First we will describe how we construct a hypernetwork for the purpose of generating the weights of a feedforward convolutional network. In a typical deep convolutional network, the majority of model parameters are in the kernels of convolutional layers. Each kernel contains  $N_{in} \times N_{out}$  filters and each filter has dimensions  $f_{size} \times f_{size}$ . Let's suppose that these parameters are stored in a matrix  $K^{j} \in \mathbb{R}^{N_{in}f_{size} \times N_{out}f_{size}}$  for each layer  $j = 1,..,D$ , where  $D$  is the depth of the main convolutional network. For each layer  $j$ , the hypernetwork receives a layer embedding  $z^{j} \in \mathbb{R}^{N_{z}}$  as input and predicts  $K^{j}$ , which can be generally written as follows:

$$
K ^ {j} = g \left(z ^ {j}\right), \quad \forall j = 1, \dots , D \tag {1}
$$

We note that this matrix  $K^j$  can be broken down as  $N_{in}$  slices of a smaller matrix with dimensions  $f_{size} \times N_{out}f_{size}$ , each slice of the kernel is denoted as  $K_i^j \in \mathbb{R}^{f_{size} \times N_{out}f_{size}}$ . Therefore, in our approach, the hypernetwork is a two-layer linear network. The first layer of the hypernetwork takes the input vector  $z^j$  and linearly projects it into the  $N_{in}$  inputs, with  $N_{in}$  different matrices  $W_i \in \mathbb{R}^{d \times N_z}$  and bias vectors  $B_i \in \mathbb{R}^d$ , where  $d$  is the size of the hidden layer in the hypernetwork. For our purpose, we fix  $d$  to be equal to  $N_z$  although they can be different. The final layer of the hypernetwork is a linear operation which takes an input vector  $a_i$  of size  $d$  and linearly projects that into  $K_i$  using a common tensor  $W_{out} \in \mathbb{R}^{f_{size} \times N_{out}f_{size} \times d}$  and bias matrix  $B_{out} \in \mathbb{R}^{f_{size} \times N_{out}f_{size}}$ . The final kernel  $K^j$  will be a concatenation of every  $K_i^j$ . Thus  $g(z^j)$  can be written as follows:

$$
a _ {i} ^ {j} = W _ {i} z ^ {j} + B _ {i}, \quad \forall i = 1,.., N _ {i n}, \forall j = 1,.., D
$$

$$
K _ {i} ^ {j} = \left\langle W _ {\text {o u t}}, a _ {i} ^ {j} \right\rangle^ {1} + B _ {\text {o u t}}, \quad \forall i = 1,.., N _ {\text {i n}}, \forall j = 1,.., D \tag {2}
$$

$$
K ^ {j} = \left( \begin{array}{c c c c c c} K _ {1} ^ {j} & K _ {2} ^ {j} & \ldots & K _ {i} ^ {j} & \ldots & K _ {N _ {i n}} ^ {j} \end{array} \right), \qquad \qquad \forall j = 1, \ldots , D
$$

In our formulation, the learnable parameters are  $W_{i}$ ,  $B_{i}$ ,  $W_{out}$ ,  $B_{out}$  together with all  $z^{j}$ 's. During inference, the model simply takes the layer embeddings  $z^{j}$  learned during training to reproduce the kernel weights for layer  $j$  in the main convolutional network. As a side effect, the number of learnable parameters in hypernetwork will be much lower than the main convolutional network. In fact, the total number of learnable parameters in hypernetwork is  $N_{z} \times D + d \times (N_{z} + 1) \times N_{i} + f_{size} \times N_{out} \times f_{size} \times (d + 1)$  compared to the  $D \times N_{in} \times f_{size} \times N_{out} \times f_{size}$  parameters for the kernels of the main convolutional network.

Our approach of constructing  $g(.)$  is similar to the hierarchically semiseparable matrix approach proposed by Xia et al. (2010). Note that even though it seems redundant to have a two-layered linear hypernetwork as that is equivalent to a one-layered hypernetwork, the fact that  $W_{out}$  and  $B_{out}$  are shared makes our two-layered hypernetwork more compact than a one-layered hypernetwork. More concretely, a one-layered hypernetwork would have  $N_z \times N_{in} \times f_{size} \times N_{out} \times f_{size}$  learnable parameters which is usually much bigger than a two-layered hypernetwork does.

The above formulation assumes that the network architecture consists of kernels with same dimensions. In practice, deep convolutional network architectures consist of kernels of varying dimensions. Typically, in many designs, the kernel dimensions are integer multiples of a basic size. This is indeed the case in the residual network family of architectures (He et al., 2016a) that we will be experimenting with later is an example of such a design. In our experiments, although the kernels of a residual network do not share the same dimensions, the  $N_{i}$  and  $N_{out}$  dimensions for each kernel are integer multiples of 16. To modify our approach to work with this architecture, we have our hypernetwork generate kernels for this basic size of 16, and if we require a larger kernel for a certain layer, we will concatenate multiple basic kernels together to form the larger kernel.

$$
K _ {3 2 \times 6 4} = \left( \begin{array}{l l l l} K _ {1} & K _ {2} & K _ {3} & K _ {4} \\ K _ {5} & K _ {6} & K _ {7} & K _ {8} \end{array} \right) \tag {3}
$$

For example, if we need to generate a kernel with  $N_{i} = 32$  and  $N_{out} = 64$ , we will tile eight basic kernels together. Each basic kernel is generated by a unique  $z$  embedding, hence the larger kernel will be expressed with eight embeddings. Therefore, kernels that are larger in size will require a proportionally larger number of embedding vectors. For visualizations of concatenated kernels, please see Appendix A.2.1. Figure 2 shows the similarity between kernels learned by a ConvNet to classify MNIST digits and those learned by a hypernetwork generating weights for a ConvNet.

![](images/a919e81ea2c2868dd7f53ca2670b09483a7e082c52ac28879b6117c4843dad01.jpg)  
Figure 2: Kernels learned by a ConvNet to classify MNIST digits (left). Kernels learned by a hypernetwork generating weights for the ConvNet (right).

![](images/c69c8a210160bc688b8faf6706cef04359cd12c0f36d3211de666db59edfedf7.jpg)

# 3.2 DYNAMIC HYPERNETWORK: ADAPTIVE WEIGHT GENERATION FOR RECURRENT NETWORKS

In the previous section, we outlined a procedure for using a hypernetwork to generate the weights for a deep convolutional network. In this section, we will use a recurrent network to dynamically generate weights for another recurrent network, such that the weights can vary across many timesteps. In this context, hypernetworks are called dynamic hypernetworks, and can be seen as a form of relaxed weight-sharing, a compromise between hard weight-sharing of traditional recurrent networks, and no weight-sharing of convolutional networks. This relaxed weight-sharing approach allows us to control the trade off between the number of model parameters and model expressiveness.

Our dynamic hypernetworks can be used to generate weights for RNN and LSTM. When a hypernetwork is used to generate the weights for an RNN, it is called HyperRNN. At every time step  $t$ , a HyperRNN takes as input the concatenated vector of input  $x_{t}$  and the hidden states of the main RNN  $h_{t-1}$ , it then generates as output the vector  $\hat{h}_{t}$ . This vector is then used to generate the weights for the main RNN at the same timestep. Both the HyperRNN and the main RNN are trained jointly with backpropagation and gradient descent. In the following, we will give a more formal description of the model.

The standard formulation of a Basic RNN is given by:

$$
h _ {t} = \phi \left(W _ {h} h _ {t - 1} + W _ {x} x _ {t} + b\right) \tag {4}
$$

where  $h_t$  is the hidden state,  $\phi$  is a non-linear operation such as  $\tanh$  or  $\text{relu}$ , and the weight matrices and bias  $W_h \in \mathbb{R}^{N_h \times N_h}$ ,  $W_x \in \mathbb{R}^{N_h \times N_x}$ ,  $b \in \mathbb{R}^{N_h}$  is fixed each timestep for an input sequence  $X = (x_1, x_2, \ldots, x_T)$ .

![](images/6807a50d599198ddbd5d41ae3283c4b4dbbc2760932fe9d07d61ab1f15ebb0c5.jpg)  
Figure 3: An overview of HyperRNNs. Black connections and parameters are associated basic RNNs. Orange connections and parameters are introduced in this work and associated with HyperRNNs. Dotted arrows are for parameter generation.

In HyperRNN, we allow  $W_{h}$  and  $W_{x}$  to float over time by using a smaller hypernetwork to generate these parameters of the main RNN at each step (see Figure 3). More concretely, the parameters  $W_{h}, W_{x}, b$  of the main RNN are different at different time steps, so that  $h_{t}$  can now be computed as:

$$
h _ {t} = \phi \left(W _ {h} \left(z _ {h}\right) h _ {t - 1} + W _ {x} \left(z _ {x}\right) + b \left(z _ {b}\right)\right), \text {w h e r e}
$$

$$
\begin{array}{l} W _ {h} \left(z _ {h}\right) = \left\langle W _ {h z}, z _ {h} \right\rangle \\ W _ {x} \left(z _ {x}\right) = \left\langle W _ {x z}, z _ {x} \right\rangle \end{array} \tag {5}
$$

$$
b (z _ {b}) = W _ {b z} z _ {b} + b _ {0}
$$

Where  $W_{hz}\in \mathbb{R}^{N_h\times N_h\times N_z},W_{xz}\in \mathbb{R}^{N_h\times N_x\times N_z},W_{bz}\in \mathbb{R}^{N_h\times N_z},b_0\in \mathbb{R}^{N_h}$  and  $z_{h},z_{x},z_{z}\in$ $\mathbb{R}^{N_z}$ . We use a recurrent hypernetwork to compute  $z_{h},z_{x}$  and  $z_{b}$  as a function of  $x_{t}$  and  $h_{t - 1}$ :

$$
\hat {x} _ {t} = \left( \begin{array}{c} h _ {t - 1} \\ x _ {t} \end{array} \right)
$$

$$
\hat {h} _ {t} = \phi \left(W _ {\hat {h}} \hat {h} _ {t - 1} + W _ {\hat {x}} \hat {x} _ {t} + \hat {b}\right)
$$

$$
z _ {h} = W _ {\hat {h} h} \hat {h} _ {t - 1} + b _ {\hat {h} h} \tag {6}
$$

$$
z _ {x} = W _ {\hat {h} x} \hat {h} _ {t - 1} + b _ {\hat {h} x}
$$

$$
z _ {b} = W _ {\hat {h} b} \hat {h} _ {t - 1}
$$

Where  $W_{\hat{h}} \in \mathbb{R}^{N_{\hat{h}} \times N_{\hat{h}}}, W_{\hat{x}} \in \mathbb{R}^{N_{\hat{h}} \times (N_{h} + N_{z})}, b \in \mathbb{R}^{N_{\hat{h}}}$ , and  $W_{\hat{h} h}, W_{\hat{h} x}, W_{\hat{h} b} \in \mathbb{R}^{N_{z} \times N_{\hat{h}}}$  and  $b_{\hat{h} h}, b_{\hat{h} x} \in \mathbb{R}^{N_{z}}$ . This HyperRNN Cell has  $N_{\hat{h}}$  hidden units. Typically  $N_{\hat{h}}$  is much smaller than  $N_{h}$ .

As the embeddings  $z_{h}, z_{x}$  and  $z_{b}$  are of dimensions  $N_{z}$ , which is typically smaller than the hidden state size  $N_{\hat{h}}$  of the HyperRNN cell, a linear network is used to project the output of the HyperRNN cell into the embeddings in Equation 6. After the embeddings are computed, they will be used to generate the full weight matrix of the main RNN.

The above is a general formulation of a linear dynamic hypernetwork applied to RNNs. However, we found that in practice, Equation 5 is often not practical because the memory usage becomes too large for real problems. The amount of memory required in the system described in Equation 5 will be  $N_{z}$  times the memory of a Basic RNN, which limits the number of hidden units we can use in many practical applications.

We can modify the dynamic hypernetwork system described in Equation 5 so that it can be much more scalable and memory efficient. Our approach borrows from the static hypernetwork section and we will use an intermediate hidden vector  $d(z) \in \mathbb{R}^{N_h}$  to parametrize a weight matrix, where  $d(z)$  will be a linear projection of  $z$ . To dynamically modify a weight matrix  $W$ , we will allow each

row of this weight matrix to be scaled linearly by an element in vector  $d$ . We refer  $d$  as a weight scaling vector. Below is the modification to  $W(z)$ :

$$
W (z) = W \left(d (z)\right) = \left( \begin{array}{c} d _ {0} (z) W _ {0} \\ d _ {1} (z) W _ {1} \\ \dots \\ d _ {N _ {h}} (z) W _ {N _ {h}} \end{array} \right) \tag {7}
$$

While we sacrifice the ability to construct an entire weight matrix from a linear combination of  $N_{z}$  matrices of the same size, we are able to linearly scale the rows of a single matrix with  $N_{z}$  degrees of freedom. We find this to be a good trade off, as this formulation of converting  $W(z)$  into  $W(d(z))$  decreases the amount of memory required by the dynamic hypernetwork. Rather than requiring  $N_{z}$  times the memory of a Basic RNN, we will only be using memory in the order  $N_{z}$  times the number of hidden units, which is an acceptable amount of extra memory usage that is often available in many applications. In addition, the row-level operation in Equation 7 can be shown to be equivalent to an element-wise multiplication operator and hence computationally much more efficient in practice. Below is the more memory efficient version of the setup of Equation 5:

$$
h _ {t} = \phi \left(d _ {h} \left(z _ {h}\right) \odot W _ {h} h _ {t - 1} + d _ {x} \left(z _ {x}\right) \odot W _ {x} x _ {t} + b \left(z _ {b}\right)\right), \text {w h e r e}
$$

$$
d _ {h} \left(z _ {h}\right) = W _ {h z} z _ {h} \tag {8}
$$

$$
d _ {x} \left(z _ {x}\right) = W _ {x z} z _ {x}
$$

$$
b \left(z _ {b}\right) = W _ {b z} z _ {b} + b _ {0}
$$

This formulation of the HyperRNN has some similarities to Recurrent Batch Normalization (Cooijmans et al., 2016) and Layer Normalization (Ba et al., 2016). The central idea for the normalization techniques is to calculate the first two statistical moments of the inputs to the activation function, and to linearly scale the inputs to have zero mean and unit variance. An additional set of fixed parameters are learned to unscale the activations if required. This element-wise operation also has similarities to the Multiplicative RNN (Sutskever et al., 2011) and Multiplicative Integration RNN (Wu et al., 2016) where it was demonstrated that the multiplication-operation encouraged better gradient flow.

Since the HyperRNN cell can indirectly modify the rows of each weight matrix and also the bias of the main RNN, it is implicitly also performing a linear scaling to the inputs of the activation function. The difference here is that the linear scaling parameters can be different for each timestep and also for each input sample. It will be interesting to compare the scaling policy that the HyperRNN cell comes up with, to the hand engineered statistical-moments based scaling approaches. In addition, we note that the existing normalization approaches can work together with the HyperRNN approach, where the HyperRNN cell will be tasked with discovering a better dynamical scaling policy to complement normalization. We will also explore this combination in our experiments.

The Long Short-Term Memory (LSTM) architecture (Hochreiter & Schmidhuber, 1997) is usually better than the Basic RNN at storing and retrieving information over longer time steps. In our experiments, we will focus on this LSTM version of the HyperRNN, called the HyperLSTM. The details of the HyperLSTM architecture is described in Appendix A.2.2, along with specific implementation details in Appendix A.2.3. We want to know whether the HyperLSTM cell can learn a weight adjustment policy that can rival statistical moments-based normalization methods, hence Layer Normalization will be one of our baseline methods. We will therefore conduct experiments on two versions of HyperLSTM, one with and one without the application of Layer Normalization.

# 4 EXPERIMENTS

In the following experiments, we will benchmark the performance of static hypernetworks on image recognition with MNIST and CIFAR-10, and the performance of dynamic hypernetworks on language modelling with Penn Treebank and Hutter Prize Wikipedia (enwik8) datasets and handwriting generation.

# 4.1 USING STATIC HYPERNETWORKS TO GENERATE FILTERS FOR CONVOLUTIONAL NETWORKS AND MNIST

We start by applying a hypernetwork to generate the filters for a convolutional network on MNIST. Our main convolutional network is a small two layer network and the hypernetwork is used to generate the kernel for the second layer (7x7x16x16), which contains the bulk of the trainable parameters in the system. Our weight matrix will be summarized by an embedding of size  $N_z = 4$ . See Appendix A.3.1 for further experimental setup details.

For this task, the hypernetwork achieved a test accuracy of  $99.24\%$ , comparable to the  $99.28\%$  for the conventional method. In this example, a kernel consisting of 12,544 weights is represented by an embedding vector of only 4 parameters, generated by a hypernetwork that has 4240 parameters. We can see the weight matrix this network produced by the hypernetwork in Figure 2. Now the question is whether we can also train a deep convolutional network, using a single hypernetwork generating a set of weights for each layer, on a dataset more challenging than MNIST.

# 4.2 STATIC HYPERNETWORKS FOR RESIDUAL NETWORK ARCHITECTURE AND CIFAR-10

The residual network architectures (He et al., 2016a; Zagoruyko & Komodakis, 2016) are popular for image recognition tasks, as they can accommodate very deep networks while maintaining effective gradient flow across layers using skip connections. The original resnet and subsequent derivatives (Zhang et al., 2016; Huang et al., 2016a) achieved state-of-the-art image recognition performance on a variety of public datasets. While residual networks can be very deep, and in some experiments as deep as 1001 layers ((He et al., 2016b), it is important to understand whether some these layers share common properties and can be reduced effectively by introducing weight sharing. If we enforce weight-sharing across many layers of a deep feed forward network, the network may share many properties to that of a recurrent network. In this experiment, we want to explore this idea of enforcing relaxed weight sharing across all of the layers of a deep residual network. We will take a simple version of residual network, use a single hypernetwork to generate the weights of all of its layers for image classification task on the CIFAR-10 dataset.

<table><tr><td>group name</td><td>output size</td><td>block type</td></tr><tr><td>conv1</td><td>32 × 32</td><td>[3×3, 16]</td></tr><tr><td>conv2</td><td>32×32</td><td>[3×3, 16×k3×3, 16×k]×N</td></tr><tr><td>conv3</td><td>16×16</td><td>[3×3, 32×k3×3, 32×k]×N</td></tr><tr><td>conv4</td><td>8×8</td><td>[3×3, 64×k3×3, 64×k]×N</td></tr><tr><td>avg-pool</td><td>1 × 1</td><td>[8 × 8]</td></tr></table>

Table 1: Structure of Wide Residual Networks in Zagoruyko & Komodakis (2016).  $N$  determines the number of residual blocks in each group. Network width is determined by factor  $k$ .

Our experiment will use a version of the wide residual network (Zagoruyko & Komodakis, 2016), described in Table 1, a popular and simple variant of the family of residual network architectures, and we will focus configurations  $(N = 6, K = 1)$  and  $(N = 6, K = 2)$ , referred to as WRN 40-1 and WRN 40-2 respectively. In this setup, we will use a hypernetwork to generate all of the kernels in conv2, conv3, and conv4, so we will generate 36 layers of kernels in total. The WRN architecture uses a filter size of 3 for every kernel. We use the method outlined in the Methods section to deal with kernels of varying sizes, and use the an embedding size of  $N_z = 64$  in our experiments. See Appendix A.3.2 for further experimental setup details.

We obtained similar classification accuracy numbers as reported in (Zagoruyko & Komodakis, 2016) with our own implementation. We also note that the weights generated by the hypernetwork are used in a batch normalization setting without modification to the original model. In principle, hypernetworks can also be applied to the newer variants of residual networks with more skip connections, such as DenseNets and ResNets of Resnets.

From the results, we see that enforcing a relaxed weight sharing constraint to the deep residual network cost us  $\sim 1.25 - 1.5\%$  in classification accuracy, while drastically reducing the number of

<table><tr><td>Model</td><td>Test Error</td><td>Param Count</td></tr><tr><td>Network in Network (Lin et al., 2014)</td><td>8.81%</td><td></td></tr><tr><td>FitNet (Romero et al., 2014)</td><td>8.39%</td><td></td></tr><tr><td>Deeply Supervised Nets (Lee et al., 2015)</td><td>8.22%</td><td></td></tr><tr><td>Highway Networks (Srivastava et al., 2015)</td><td>7.72%</td><td></td></tr><tr><td>ELU (Clevert et al., 2015)</td><td>6.55%</td><td></td></tr><tr><td>Original Resnet-110 (He et al., 2016a)</td><td>6.43%</td><td>1.7 M</td></tr><tr><td>Stochastic Depth Resnet-110 (Huang et al., 2016b)</td><td>5.23%</td><td>1.7 M</td></tr><tr><td>Wide Residual Network 40-1 (Zagoruyko &amp; Komodakis, 2016)</td><td>6.85%</td><td>0.6 M</td></tr><tr><td>Wide Residual Network 40-2 (Zagoruyko &amp; Komodakis, 2016)</td><td>5.33%</td><td>2.2 M</td></tr><tr><td>Wide Residual Network 28-10 (Zagoruyko &amp; Komodakis, 2016)</td><td>4.17%</td><td>36.5 M</td></tr><tr><td>ResNet of ResNet 58-4 (Zhang et al., 2016)</td><td>3.77%</td><td>13.3 M</td></tr><tr><td>DenseNet (Huang et al., 2016a)</td><td>3.74%</td><td>27.2 M</td></tr><tr><td>Wide Residual Network 40-12</td><td>6.73%</td><td>0.563 M</td></tr><tr><td>Hyper Residual Network 40-1 (ours)</td><td>8.02%</td><td>0.097 M</td></tr><tr><td>Wide Residual Network 40-22</td><td>5.66%</td><td>2.236 M</td></tr><tr><td>Hyper Residual Network 40-2 (ours)</td><td>7.23%</td><td>0.148 M</td></tr></table>

Table 2: CIFAR-10 Classification with hypernetwork generated weights.

parameters in the model as a trade off. One reason for this reduction in accuracy is because different layers of a deep network is trained to extract different levels of features, and require different kinds of filters to perform optimally. The hypernetwork enforces some commonality between every layer, but offers each layer 64 degrees of freedom to distinguish itself from the other layers. While the network is no longer able to learn the optimal set of filters for each layer, it will learn the best set of filters given the constraints, and the resulting number of model parameters is drastically reduced.

# 4.3 HYPERLSTM FOR CHARACTER-LEVEL PENN TREEBANK LANGUAGE MODELLING

The HyperLSTM model is evaluated on character level prediction task on the Penn Treebank corpus (Marcus et al., 1993) using the train/validation/test split outlined in (Mikolov et al., 2012). As the dataset is quite small, it is prone to over fitting, we apply dropout on both input and output layers with a keep probability of 0.90. Unlike previous approaches (Graves, 2013; Ognawala & Bayer, 2014), applying weight noise during training, we instead also apply dropout to the recurrent layer (Henaff et al., 2016) with the same dropout probability.

We compare our model to the basic LSTM cell, stacked LSTM cells (Graves, 2013), and LSTM with layer normalization applied. In addition, we also experimented with applying layer normalization to HyperLSTM. Using the setup in (Graves, 2013), we use networks with 1000 units and train the network to predict the next character. In this task, the HyperLSTM cell has 128 units and a signal size of 4. As the HyperLSTM cell has more trainable parameters compared to the basic LSTM Cell, we also experimented with an LSTM Cell with 1250 units as well. For more details regarding experimental setup, please refer to Appendix A.3.3

It is interesting to note that combining Recurrent Dropout with a basic LSTM cell achieves quite formidable performance. Our implementation of Recurrent Dropout Basic LSTM cell reproduced similar results as (Semeniuta et al., 2016), where they have also experimented with different dropout settings. We also found that Layer Norm LSTM performed quite well when combined with recurrent dropout, making it both a formidable baseline and also an extension for HyperLSTM.

In addition to outperforming both the larger or deeper version of the LSTM network, HyperLSTM also achieved similar performance of Layer Norm LSTM. This suggests by dynamically adjusting the weight scaling vectors, the HyperLSTM cell has learned a policy of scaling inputs to the activation functions that is as efficient as the statistical moments-based strategy employed by Layer Norm, and that the required extra computation required is embedded inside the extra 128 units inside the HyperLSTM cell. When we combine HyperLSTM with Layer Norm, we see an additional performance gain, implying that the HyperLSTM cell learned an adjustment policy that goes beyond moments-based regularization. We also demonstrate that increasing the size of the embedding vector or stacking HyperLSTM layers together can also increase its performance.

<table><tr><td>Model1</td><td>Test</td><td>Validation</td><td>Param Count</td></tr><tr><td>ME n-gram (Mikolov et al., 2012)</td><td>1.37</td><td></td><td></td></tr><tr><td>Batch Norm LSTM (Cooijmans et al., 2016)</td><td>1.32</td><td></td><td></td></tr><tr><td>Recurrent Dropout LSTM (Semeniuta et al., 2016)</td><td>1.301</td><td>1.338</td><td></td></tr><tr><td>Zoneout RNN (Krueger et al., 2016)</td><td>1.27</td><td></td><td></td></tr><tr><td>HM-LSTM3 (Chung et al., 2016)</td><td>1.27</td><td></td><td></td></tr><tr><td>LSTM, 1000 units2</td><td>1.312</td><td>1.347</td><td>4.25 M</td></tr><tr><td>LSTM, 1250 units2</td><td>1.306</td><td>1.340</td><td>6.57 M</td></tr><tr><td>2-Layer LSTM, 1000 units2</td><td>1.281</td><td>1.312</td><td>12.26 M</td></tr><tr><td>Layer Norm LSTM, 1000 units2</td><td>1.267</td><td>1.300</td><td>4.26 M</td></tr><tr><td>HyperLSTM (ours), 1000 units</td><td>1.265</td><td>1.296</td><td>4.91 M</td></tr><tr><td>Layer Norm HyperLSTM, 1000 units (ours)</td><td>1.250</td><td>1.281</td><td>4.92 M</td></tr><tr><td>Layer Norm HyperLSTM, 1000 units, Large Embedding (ours)</td><td>1.233</td><td>1.263</td><td>5.06 M</td></tr><tr><td>2-Layer Norm HyperLSTM, 1000 units</td><td>1.219</td><td>1.245</td><td>14.41 M</td></tr></table>

# 4.4 HYPERLSTM FOR HUTTER PRIZE WIKIPEDIA LANGUAGE MODELLING

We train our model on the larger and more challenging Hutter Prize Wikipedia dataset, also known as enwik8 (Hutter, 2012) consisting of a sequence of 100M characters composed of 205 unique characters. Unlike Penn Treebank, enwik8 contains some foreign words (Latin, Arabic, Chinese), indented XML, metadata, and internet addresses, making it a more realistic and practical dataset to test character language models. For more details regarding experimental setup, please refer to Appendix A.3.4. Examples of these mixed variety of text samples that our HyperLSTM model can generate is in Appendix A.4.

Table 3: Bits-per-character on the Penn Treebank test set.  

<table><tr><td>Model1</td><td>enwik8</td><td>Param Count</td></tr><tr><td>Stacked LSTM (Graves, 2013)</td><td>1.67</td><td>27.0 M</td></tr><tr><td>MRNN (Sutskever et al., 2011)</td><td>1.60</td><td></td></tr><tr><td>GF-RNN (Chung et al., 2015)</td><td>1.58</td><td>20.0 M</td></tr><tr><td>Grid-LSTM (Kalchbrenner et al., 2016)</td><td>1.47</td><td>16.8 M</td></tr><tr><td>LSTM (Rocki, 2016b)</td><td>1.45</td><td></td></tr><tr><td>MI-LSTM (Wu et al., 2016)</td><td>1.44</td><td></td></tr><tr><td>Recurrent Highway Networks (Zilly et al., 2016)</td><td>1.42</td><td>8.0 M</td></tr><tr><td>Recurrent Memory Array Structures (Rocki, 2016a)</td><td>1.40</td><td></td></tr><tr><td>HM-LSTM3 (Chung et al., 2016)</td><td>1.40</td><td></td></tr><tr><td>Surprisal Feedback LSTM4 (Rocki, 2016b)</td><td>1.37</td><td></td></tr><tr><td>LSTM, 1800 units, no recurrent dropout2</td><td>1.470</td><td>14.81 M</td></tr><tr><td>LSTM, 2000 units, no recurrent dropout2</td><td>1.461</td><td>18.06 M</td></tr><tr><td>Layer Norm LSTM, 1800 units2</td><td>1.402</td><td>14.82 M</td></tr><tr><td>HyperLSTM (ours), 1800 units</td><td>1.391</td><td>18.71 M</td></tr><tr><td>Layer Norm HyperLSTM, 1800 units (ours)</td><td>1.353</td><td>18.78 M</td></tr><tr><td>Layer Norm HyperLSTM, 2048 units (ours)</td><td>1.340</td><td>26.54 M</td></tr></table>

Table 4: Bits-per-character on the enwik8 test set.

We see that HyperLSTM is once again competitive to Layer Norm LSTM, and if we combine both techniques, the Layer Norm HyperLSTM achieves respectable results. The version of HyperLSTM that uses 2048 hidden units achieve near state-of-the-art performance for this task. In addition, HyperLSTM converges quicker per training step compared to LSTM and Layer Norm LSTM. Please refer to Figure 6 for the loss graphs.

![](images/43e58ed42db7c831da921822b552d1eb3f680f5b73af42647e2bd74545169c5a.jpg)  
Figure 4: Example text generated from HyperLSTM model. We visualize how four of the main RNN's weight matrices  $(W_h^i, W_h^g, W_h^f, W_h^o)$  effectively change over time by plotting the norm of the changes below each generated character. High intensity represents large changes being made to weights of main RNN.

When we use this prediction model as a generative model to sample a text passage, we use main RNN to model a probability distribution over possible characters conditioned over the preceding characters. In the case of the HyperRNN, we allow the model parameters of this generative model to vary over time, so in effect the HyperRNN cell is choosing the best model at any given time to generate a probability distribution to sample from. We can demonstrate this by visualizing how the weight scaling vectors of the main RNN change during the character sampling process. In Figure 4, we examine a sample text passage generated by HyperLSTM after training on enwik8 along with the weight differences below the text. We see that in regions of low intensity, where the weights of the main RNN are relatively static, the types of phrases generated seem more deterministic. For example, the weights do not change much during the words Europeans, possessions and reservation. The regions of high intensity is when the HyperRNN cell is making relatively large changes to the weights of the main RNN. These tend to happen in the areas between words, or sometimes during brackets.

One might also wonder whether the HyperLSTM cell (without Layer Norm), via dynamically tuning the weight scaling vectors, has developed a policy that is similar to the statistics-based approach used by Layer Norm, given that both methods have similar performance. One way to see this effect is to look at the histogram of the hidden states in the network. In Figure 5, we examine the histograms of  $\phi(c_{t})$ , the hidden state of the LSTM before applying the output gate.

![](images/459e8d0e8345772b0cfb29b546412c007cf6f050d2fa961f54d54f64f3ced296.jpg)  
Figure 5: Normalized Histogram plots of  $\phi (c_t)$  for different models during sampling.

![](images/b560a92374a941fcaedbd3aaa49d59cc90e186caa57c24a01a0c3946d2001660.jpg)

![](images/ec97b2f9df663a4f6e01f24766527675b839160b5ce8a88c051800a994145c84.jpg)

![](images/ea3eda9cc54ab4c6d57d10becd543a636727cde4ecc69a6a8fa675f69e17e888.jpg)

We see that the normalization process employed by Layer Norm reduces the saturation effects compared to the vanilla LSTM. However, for the case of the HyperLSTM, we notice that most of the time the cell is saturated. The HyperLSTM cell's dynamic weight adjustment policy appears to be doing something very different compared to statistical normalization, although the policy it came up with ended up providing similar performance as Layer Norm. It is interesting to see that when we combine both methods, the HyperLSTM cell will need to determine an adjustment policy in spite of the normalization forced upon it by Layer Norm. An interesting question is whether there are problems where statistical normalization may actually be a setback to the policy developed by the HyperLSTM, and the best strategy is to ignore it.

![](images/e9a0578ffb95e7ce0d07c289ac95fd0498e8f56eede3360c122af4c01e0f96b4.jpg)  
Figure 6: Loss Graph for enwik8 (left). Loss Graph for Handwriting Generation (right)

![](images/f09ac5d6d6ba2ecb16f8d398bc82f05a2ec8713413ceb3e3432fd426792a9d6b.jpg)

# 4.5 HYPERLSTM FOR HANDWRITING SEQUENCE GENERATION

In addition to modelling discrete sequential data, we want to see how the model performs when modelling sequences of real valued data. We will train our model on the IAM online handwriting database (Liwicki & Bunke, 2005) and have our model predict pen strokes as per Section 4.2 of (Graves, 2013). The dataset has contains 12179 handwritten lines from 221 writers, digitally recorded from a tablet. We will model the  $(\mathrm{x},\mathrm{y})$  coordinate of the pen location at each recorded time step, along with a binary indicator of pen-up/pen-down. The average sequence length is around 700 steps and the longest around 1900 steps, making the training task particularly challenging as the network needs to retain information about both the stroke history and also the handwriting style in order to predict plausible future handwriting strokes. For experimental setup details, please refer to Appendix A.3.5.

<table><tr><td>Model</td><td>Log-Loss</td><td>Param Count</td></tr><tr><td>LSTM, 900 units (Graves, 2013)</td><td>-1,026</td><td></td></tr><tr><td>3-Layer LSTM, 400 units (Graves, 2013)</td><td>-1,041</td><td></td></tr><tr><td>3-Layer LSTM, 400 units, adaptive weight noise (Graves, 2013)</td><td>-1,058</td><td></td></tr><tr><td>LSTM, 900 units, no dropout, no data augmentation.1</td><td>-1,026</td><td>3.36 M</td></tr><tr><td>3-Layer LSTM, 400 units, no dropout, no data augmentation.1</td><td>-1,039</td><td>3.26 M</td></tr><tr><td>LSTM, 900 units2</td><td>-1,055</td><td>3.36 M</td></tr><tr><td>LSTM, 1000 units2</td><td>-1,048</td><td>4.14 M</td></tr><tr><td>3-Layer LSTM, 400 units2</td><td>-1,068</td><td>3.26 M</td></tr><tr><td>2-Layer LSTM, 650 units2</td><td>-1,135</td><td>5.16 M</td></tr><tr><td>Layer Norm LSTM, 900 units2</td><td>-1,096</td><td>3.37 M</td></tr><tr><td>Layer Norm LSTM, 1000 units2</td><td>-1,106</td><td>4.14 M</td></tr><tr><td>Layer Norm HyperLSTM, 900 units (ours)</td><td>-1,067</td><td>3.95 M</td></tr><tr><td>HyperLSTM (ours), 900 units</td><td>-1,162</td><td>3.94 M</td></tr></table>

Table 5: Log-Loss of IAM Online DB validation set.

In this task, we note that data augmentation and applying recurrent dropout improved the performance of all models, compared to the original setup by (Graves, 2013). In addition, for the LSTM model, increasing unit count per layer may not help the performance compared to increasing the layer depth. We notice that a 3-layer 400 unit LSTM outperforms a 1-layer 900 unit one, and we found that a 2-layer 650 unit LSTM outperforming most configurations. While layer norm helps with the performance, we found that in this task, layer norm does not combine well with HyperLSTM, and in this task the 900 unit HyperLSTM without layer norm achieved the best performance.

Unlike the language modelling task, perhaps statistical normalization is far from the optimal approach for a weight adjustment policy. The policy learned by the HyperLSTM cell not only per

formed well against the baseline, its convergence rate is also as fast as the 2-layer LSTM model. Please refer to Figure 6 for the loss graphs.

In Appendix A.5, we display three sets of handwriting samples generated from LSTM, Layer Norm LSTM, and HyperLSTM, corresponding to log-loss scores of -1055, -1096, and -1162 nats respectively in Table 5. Qualitative assessments of handwriting quality is always subjective, and depends an individual's taste in calligraphy. From looking at the examples produced by the three models, our opinion is that the samples produced by LSTM is noisier than the other two models. We also find HyperLSTM's samples to be a bit more coherent than the samples produced by Layer Norm LSTM. We leave to the reader to judge which model produces handwriting samples of higher quality.

![](images/3cbf338587439c7f51d8478aaf7f3956e342443d5ab27b561588115dd5de6d6e.jpg)  
Figure 7: Handwriting sample generated from HyperLSTM model. We visualize how four of the main RNN's weight matrices  $(W_h^i W_h^g, W_h^f, W_h^o)$  effectively change over time, by plotting norm of changes made to them over time.

Similar to the earlier character generation experiment, we show a generated handwriting sample from the HyperLSTM model in Figure 7, along with a plot of how the weight scaling vectors of the main RNN is changing over time below the sample. For a more detailed interactive demonstration of handwriting generation using HyperLSTM, visit http://blog.otoro.net/2016/09/28/hyper-networks/.

We see that the regions of high intensity seem to be concentrated at many discrete instances, rather than slowly varying over time. This implies that the weights experience regime changes rather than gradual slow adjustments. We can see that many of these weight changes occur between the written words, and sometimes between written characters. While the LSTM model alone already does a formidable job of generating time-varying parameters of a Mixture Gaussian distribution used to generate realistic handwriting samples, the ability to go one level deeper, and to dynamically generate the generative model is one of the key advantages of HyperRNN over a normal RNN.

# 4.6 HYPERLSTM FOR NEURAL MACHINE TRANSLATION

We experiment with the Neural Machine Translation task using the same experimental setup outlined in (Wu et al., 2016). Our model is the same wordpiece model architecture with a vocabulary size of  $32\mathrm{k}$ , but we replace the LSTM cells with HyperLSTM cells. We benchmark the modified model on WMT'14  $\mathrm{En} \rightarrow \mathrm{Fr}$  using the same test/validation set split described in the GNMT paper (Wu et al., 2016). Please refer to Appendix A.3.6 for experimental setup details.

<table><tr><td>Model</td><td>Test BLEU</td><td>Log Perplexity</td></tr><tr><td>Deep-Att + PosUnk (Zhou et al., 2016)</td><td>39.2</td><td></td></tr><tr><td>GNMT WPM-32K, LSTM (Wu et al., 2016)</td><td>38.95</td><td>1.027</td></tr><tr><td>GNMT WPM-32K, ensemble of 8 LSTMs (Wu et al., 2016)</td><td>40.35</td><td></td></tr><tr><td>GNMT WPM-32K, HyperLSTM (ours)</td><td>40.03</td><td>0.993</td></tr></table>

Table 6: Single model results on WMT En→Fr (newstest2014)

The HyperLSTM cell improves the performance of the existing GNMT model, achieving state-of-the-art single model results for this dataset. In addition, we demonstrate the applicability of hypernetworks to large-scale models used in production systems. Please see Appendix A.6 for actual translation samples generated from both models for a qualitative comparison.

# 5 CONCLUSION

In this paper, we presented a method to use a hypernetwork to generate weights for another neural network. Our hypernetworks are trained end-to-end with backpropagation and therefore are efficient and scalable. We focused on two use cases of hypernetworks: static hypernetworks to generate weights for a convolutional network, dynamic hypernetworks to generate weights for recurrent networks. We found that the method works well while using fewer parameters. On image recognition, language modelling and handwriting generation, hypernetworks are competitive to or sometimes better than state-of-the-art models.

# ACKNOWLEDGMENTS

We thank Jeff Dean, Geoffrey Hinton, Mike Schuster and the Google Brain team for their help with the project.

# REFERENCES

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Gregory S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian J. Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Józefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Rajat Monga, Sherry Moore, Derek Gordon Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul A. Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda B. Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. CoRR, abs/1603.04467, 2016. URL http://arxiv.org/abs/1603.04467.  
M. Andrychowicz, M. Denil, S. Gomez, M. W. Hoffman, D. Pfau, T. Schaul, and N. de Freitas. Learning to learn by gradient descent by gradient descent. arXiv preprint arXiv:1606.04474, 2016.  
Jimmy L. Ba, Jamie R. Kiros, and Geoffrey E. Hinton. Layer normalization. NIPS, 2016.  
Luca Bertinetto, João F. Henriques, Jack Valmadre, Philip H. S. Torr, and Andrea Vedaldi. Learning feed-forward one-shot learners. In NIPS, 2016.  
Christopher M. Bishop. Mixture density networks. Technical report, 1994.  
Junyoung Chung, Caglar Gülcehre, Kyunghyun Cho, and Yoshua Bengio. Gated feedback recurrent neural networks. arXiv preprint arXiv:1502.02367, 2015.  
Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.  
Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). arXiv preprint arXiv:1511.07289, 2015.  
Tim Coolijmans, Nicolas Ballas, Cesar Laurent, and Caglar Gulcehre. Recurrent Batch Normalization. arXiv:1603.09025, 2016.  
Bert De Brabandere, Xu Jia, Tinne Tuytelaars, and Luc Van Gool. Dynamic filter networks. In NIPS, 2016.  
Misha Denil, Babak Shakibi, Laurent Dinh, Marc'Aurelio Ranzato, and Nando de Freitas. Predicting Parameters in Deep Learning. In NIPS, 2013.  
Chrisantha Fernando, Dylan Banarse, Malcolm Reynolds, Frederic Besse, David Pfau, Max Jaderberg, Marc Lanctot, and Daan Wierstra. Convolution by evolution: Differentiable pattern producing networks. In GECCO, 2016.  
Faustino Gomez and Jürgen Schmidhuber. Evolving modular fast-weight networks for control. In ICANN, 2005.

Alex Graves. Generating sequences with recurrent neural networks. arXiv:1308.0850, 2013.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. arXiv preprint arXiv:1603.05027, 2016b.  
Mikael Henaff, Arthur Szlam, and Yann LeCun. Orthogonal RNNs and long-memory tasks. In ICML, 2016.  
Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov. Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580, 2012.  
Sepp Hochreiter and Juergen Schmidhuber. Long short-term memory. Neural Computation, 1997.  
Gao Huang, Zhuang Liu, and Kilian Q. Weinberger. Densely connected convolutional networks. arXiv preprint arXiv:1608.06993, 2016a.  
Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Weinberger. Deep networks with stochastic depth. arXiv preprint arXiv:1603.09382, 2016b.  
Marcus Hutter. The human knowledge compression contest. 2012. URL http://prize.hutter1.net/.  
Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Decoupled Neural Interfaces using Synthetic Gradients. arXiv preprint arXiv:1608.05343, 2016.  
Nal Kalchbrenner, Ivo Danihelka, and Alex Graves. Grid long short-term memory. In  $ICLR$ , 2016.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Jan Koutnik, Faustino Gomez, and Jurgen Schmidhuber. Evolving neural networks in compressed weight space. In GECCO, 2010.  
David Krueger, Tegan Maharaj, János Kramár, Mohammad Pezeshki, Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, Hugo Larochelle, Aaron Courville, et al. Zoneout: Regularizing RNNs by randomly preserving hidden activations. arXiv preprint arXiv:1606.01305, 2016.  
Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Handwritten digit recognition with a back-propagation network. In NIPS, 1990.  
Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, and Zhuowen Tu. Deeply-supervised nets. In AISTATS, volume 2, pp. 6, 2015.  
Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. In ICLR, 2014.  
Marcus Liwicki and Horst Bunke. IAM-OnDB - an on-line English sentence database acquired from handwritten text on a whiteboard. In ICDAR, 2005.  
Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313-330, 1993.  
Tomáš Mikolov, Ilya Sutskever, Anoop Deoras, Hai-Son Le, Stefan Kombrink, and Jan Cernocky. Subword language modeling with neural networks. preprint, 2012.  
Marcin Moczulski, Misha Denil, Jeremy Appleyard, and Nando de Freitas. ACDC: A Structured Efficient Linear Layer. arXiv preprint arXiv:1511.05946, 2015.  
Saahil Ognawala and Justin Bayer. Regularizing recurrent networks-on injected noise and norm-based methods. arXiv preprint arXiv:1410.5684, 2014.  
Kamil Rocki. Recurrent memory array structures. arXiv preprint arXiv:1607.03085, 2016a.

Kamil Rocki. Surprisal-driven feedback in recurrent networks. arXiv preprint arXiv:1608.06027, 2016b.  
Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.  
Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131-139, 1992.  
Jürgen Schmidhuber. A 'self-referential' weight matrix. In ICANN, 1993.  
Stanislaw Semeniuta, Aliases Severyn, and Erhardt Barth. Recurrent dropout without memory loss. arXiv:1603.05118, 2016.  
Rupesh Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. In NIPS, 2015.  
Kenneth O. Stanley, David B. D'Ambrosio, and Jason Gauci. A hypercube-based encoding for evolving large-scale neural networks. Artificial Life, 15(2):185-212, 2009.  
Ilya Sutskever, James Martens, and Geoffrey E. Hinton. Generating text with recurrent neural networks. In ICML, 2011.  
Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, J. Klingner, A. Shah, M. Johnson, X. Liu, L. Kaiser, S. Gouws, Y. Kato, T. Kudo, H. Kazawa, K. Stevens, G. Kurian, N. Patil, W. Wang, C. Young, J. Smith, J. Riesa, A. Rudnick, O. Vinyals, G. Corrado, M. Hughes, and J. Dean. Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. *ArXiv e-prints*, 2016.  
Yuhuai Wu, Saizheng Zhang, Ying Zhang, Yoshua Bengio, and Ruslan Salakhutdinov. On multiplicative integration with recurrent neural networks. NIPS, 2016.  
Jianlin Xia, Shivkumar Chandrasekaran, Ming Gu, and Xiaoye S. Li. Fast algorithms for hierarchically semiseparable matrices. Numerical Linear Algebra with Applications, 2010.  
Z. Yang, M. Moczulski, M. Denil, N. de Freitas, A. Smola, L. Song, and Z. Wang. Deep Fried Convnets. In ICCV, 2015.  
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In BMVC, 2016.  
Ke Zhang, Miao Sun, Tony X. Han, Xingfang Yuan, Liru Guo, and Tao Liu. Residual networks of residual networks: Multilevel residual networks. arXiv preprint arXiv:1608.02908, 2016.  
Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016. URL http://arxiv.org/abs/1606.04199.  
Julian Zilly, Rupesh Srivastava, Jan Koutnik, and Jürgen Schmidhuber. Recurrent highway networks. arXiv preprint arXiv:1607.03474, 2016.
