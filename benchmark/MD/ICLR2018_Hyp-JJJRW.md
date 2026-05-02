# STYLE MEMORY: MAKING A CLASSIFIER NETWORK GENERATIVE

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep networks have shown great performance in classification tasks. However, the parameters learned by the classifier networks usually discard stylistic information of the input, in favour of information strictly relevant to classification. We introduce a network that has the capacity to do both classification and reconstruction by adding a "style memory" to the output layer of the network. We also show how to train such a neural network as a deep multi-layer autoencoder, jointly minimizing both classification and reconstruction losses. The generative capacity of our network demonstrates that the combination of style-memory neurons with the classifier neurons yield good reconstructions of the inputs when the classification is correct. We further investigate the nature of the style memory, and how it relates to composing digits and letters.

# 1 INTRODUCTION

Deep neural networks now rival human performance in many complex classification tasks, such as image recognition. However, these classification networks are different from human brains in some basic ways. First of all, the mammalian cortex has many feed-back connections that project in the direction opposite the sensory stream (Bullier et al., 1988). Moreover, these feed-back connections are implicated in the processing of sensory input, and seem to enable improved object/background contrast (Poort et al., 2012), and imagination (Reddy et al., 2011). Feed-back connections are also hypothesized to be involved in generating predictions in the service of perceptual decision making (Summerfield & De Lange, 2014).

Humans (and presumably other mammals) are also less susceptible to being fooled by ambiguous or adversarial inputs. Deep neural networks have been shown to be vulnerable to adversarial examples (Szegedy et al., 2014; Goodfellow et al., 2015). Slight modifications to an input can cause the neural network to misclassify it, sometimes with great confidence! Humans do not get fooled as easily, leading us to wonder if the feed-back, generative nature of real mammalian brains contributes to accurate classification.

In pursuit of that research, we wish to augment classification networks so that they are capable of both recognition (in the feed-forward direction) and reconstruction (in the feed-back direction). We want to build networks that are both classifiers and generative.

The nature of a classifier network is that it throws away most of the information, keeping only what is necessary to make accurate classifications. Simply adding feed-back connections to the network will not be enough to generate specific examples of the input - only a generic class archetype. But what if we combine the features of a classifier network and an autoencoder network by adding a "style memory" to the top layer of the network? The top layer would then consist of a classification component as well as a collection of neurons that are not constrained by any target classes.

We hypothesized that adding a style memory to the top layer of a deep autoencoder would give us the best of both worlds, allowing the classification neurons to contribute the class of the input, while the style memory would record additional information about the encoded input – presumably information not encoded by the classification neurons. The objective of our network is to minimize both classification and reconstruction losses so that the network can perform both classification and reconstruction effectively. As a proof of concept, we report on a number of experiments with MNIST and EMNIST that investigate the properties of this style memory.

# 2 RELATED WORK

Others have developed neural architectures that encode both the class and style of digits to enable reconstruction. Luo et al. (2017) recently introduced a method called bidirectional backpropagation. Their network is generative because it has feed-back connections that project down from the top (soft-max) layer. A digit class can be chosen at the top layer, and the feed-back connections render a digit of the desired class in the bottom layer (as an image). However, the network always renders the same, generic sample of the class, and does not reconstruct specific samples from the data.

Networks that have the capacity to generate images have been shown to learn meaningful features. Previous work (Hinton, 2007) showed that in order to recognize images, the network needs to first learn to generate images. Salakhutdinov & Hinton (2009) showed that a network consisting of stacked Restricted Boltzmann Machines (RBMs) learns good generative models, effective for pretraining a classifier network. RBMs are stochastic in nature, so while they can generate different inputs, they are not used to generate a specific sample of input. Bengio et al. (2006) also demonstrated that autoencoders pre-trained in a greedy manner also lead to better classifier networks. Both (Hinton et al., 2006) and (Bengio et al., 2006) use tied weights, where the feed-back weight matrices are simply the transpose of the feed-forward weights; this solution is not biologically feasible. These findings have inspired other successful models such as stacked denoising autoencoders (Vincent et al., 2010), which learn to reconstruct the original input image given a noise-corrupted input image.

Lastly, Salakhutdinov & Hinton (2007) also showed another method to map an input to a lower dimensional space that minimizes within-class distance of the input. They first pre-trained a network as RBMs, and then "unrolled" the network to form a deep autoencoder. The network was then fine-tuned by performing nonlinear neighbourhood component analysis (NCA) between the low-dimensional representations of inputs that have the same class. They were able to separate the class-relevant and class-irrelevant parts by using only  $60\%$  of the lower-dimensional code units when performing nonlinear NCA, but all the codes were used to perform reconstruction. As a result, their network was able to minimize within-class distance in the lower-dimensional space while maintaining good reconstruction. Inference was then performed by using K-nearest neighbour in that lower-dimensional space. Our method is similar, but our top layer includes an explicit classification vector alongside the class-agnostic style memory.

# 3 METHOD

# 3.1 MODEL DESCRIPTION

Our bidirectional network consists of an input layer, convolutional layers, fully connected layers, and an output layer. However, the output layer is augmented; in addition to classifier neurons (denoted by  $y$  in Fig. 1), it also includes style-memory neurons (denoted  $m$  in Fig. 1). A standard classifier network maps  $x \in X \to y \in Y$ , where the dimension of  $Y$  is usually much smaller than the dimension of  $X$ . The feed-forward connections of our augmented network map  $x \in X \to (y, m) \in Y \times M$ . The output  $y$  is the classification vector (softmax). The output  $m$  is the style memory, meant to encode information about the particular form of an input. For the example of MNIST, the classification vector might represent that the digit is a '2', while the style memory records that the '2' was written on a slant, and with a loop in the bottom-left corner.

A classifier network can be trained as a deep autoencoder network. However, the decoder will only be able to generate a single, generic element of a given class. By adding a style memory in the output layer, the network will be able to learn to generate a variety of different renderings of a particular class.

# 3.2 TRAINING

We trained the network following a standard training procedure for deep autoencoders, depicted in Fig. 2. For the input layer, we follow the work from Vincent et al. (2010) by injecting small additive Gaussian noise to the input.

![](images/4ebd7685104be9d2d06836f69092c14172d8a412a9a4b2d3cb893ced8d5a4ee1.jpg)  
Figure 1: Our bidirectional network with a style memory in the output layer. Here,  $x$  denotes the input  $(x \in X)$ , while  $Conv_{i}$  and  $FC_{i}$  denote convolutional layer and fully connected layer  $i$ , respectively. Lastly,  $y$  denotes output label  $(y \in Y)$ , and  $m$  denotes the style memory  $(m \in M)$ .

![](images/0a75ec96600e60d304b0e6228ddcde10423a5f0cd882fe8c1438a436a88780e2.jpg)  
Figure 2: The "unrolled" network. Learning consists of training the network as a deep autoencoder, where  $h_i$  denotes the hidden layer representation of layer  $i$ .

The objective for our network's top layer is to jointly minimize two loss functions. The first loss function is the classifier loss  $L_{y}$ , which is a categorical cross-entropy loss function,

$$
L _ {y} \left(y _ {t}, y\right) = - \sum_ {x} y _ {t} \log (y), \tag {1}
$$

where  $y_{t}$  is the target label, and  $y$  is the predicted label. The second loss function is the reconstruction loss between the input and its reconstruction. This reconstruction loss, denoted  $L_{r}$ , is the Euclidean distance between the input to the top layer, and the reconstruction of that input,

$$
L _ {r} (\hat {x}, x) = \| \hat {x} - x \| _ {2}, \tag {2}
$$

where  $\hat{x}$  is the reconstruction of the input  $x$ , as shown in Fig. 2.

Our goal is to find connection weights,  $W^{*}$ , that minimize the combination of both loss functions in the last layer,

$$
W ^ {*} = \arg \min  _ {W} \sum_ {x \in X} L _ {y} \left(y _ {t}, y\right) + \alpha \left(L _ {r} (\hat {x}, x)\right), \tag {3}
$$

where  $W$  represents the parameters of the network, and  $\alpha$  adjusts the weight of  $L_{r}$ .

# 4 EXPERIMENTS

We performed all experiments in this paper using digits from MNIST and letters from Extended MNIST (EMNIST) (Cohen et al., 2017) datasets, with an input dimensionality of  $28 \times 28$  pixels. The networks used for the experiments have two convolutional layers and two fully connected layers. The first and second convolutional layers are made of 32 and 64 filters, respectively. The receptive fields of both convolutional layers are  $5 \times 5$  with a stride of 2, using ReLU activation functions. The fully connected layers  $FC_{1}$  and  $FC_{2}$  have 256 and 128 ReLU neurons, respectively.

The style memory consists of 16 logistic neurons, and the classifier vector contains either 10 or 26 softmax neurons, for MNIST or EMNIST, respectively. The reconstruction loss weight  $(\alpha)$  was set to 0.05, and the optimization method used to train the network was Adam (Kingma & Ba, 2014) with a learning rate  $\eta$  of 0.00001 for 250 epochs. The network achieved  $98.48\%$  and  $91.27\%$  classification accuracy on the MNIST and EMNIST test sets, respectively.

# 4.1 RECONSTRUCTION USING STYLE MEMORY

The reconstructions produced by our network show that the network has the capacity to reconstruct a specific sample, rather than just a generic example from a specific class. Figures 3 and 4 show

![](images/ff2c5d5a214c3483f26e86d84a26cf90dcf7c03b53d35862aade3f3786d2868d.jpg)  
Figure 3: Reconstruction of MNIST digits using the network's predictions and style memories. The top row shows the original images from the MNIST test set, and the bottom row shows the corresponding reconstructions produced by the network.

![](images/178a0d2b278f2264144a82c50a640881107ecff23623958fae4b6b24371506dc.jpg)  
Figure 4: Reconstruction of EMNIST letters using the network's predictions and style memories.

examples of digit and letter reconstructions. Notice how the network has the ability to reconstruct different styles of a class, like the two different '4's, two different '9's, and two different 'A's. For each sample, the reconstruction mimics the style of the original character. Note that the digits and letters in both figures were correctly classified by the network.

# 4.2 RECONSTRUCTION OF MISCLASSIFIED SAMPLES

How do the softmax classification nodes and the style memory interact when a digit or letter is misclassified? The first column in Fig. 5 shows an example where the digit '3' was misclassified as a '5' with  $71\%$  confidence. The resulting reconstruction in the middle row looks more like a '5' (although there is a hint of a '3'). However, correcting the softmax neurons to the one-hot ground truth label for '3' changed the reconstruction to look more like a '3', as shown in the bottom row of Fig. 5. Similar results were observed when we used letters from the EMNIST dataset, as shown in Fig. 6.

We believe that the generative abilities of these classifier networks enable it to identify misclassified inputs. If the reconstruction does not closely match the input, then it is likely that the input was misclassified. This idea forms the crux of how these networks might defend against being fooled by adversarial or ambiguous inputs.

![](images/ffdfec92ad64f3d80ad7c15a3bf198cc8e6aad7a8312c4881c92cc149041515a.jpg)

![](images/f4b2b62d5e4d2b108718570a83cc08a95942057db9ee15b5ba47bc8b09abbf4a.jpg)  
Prediction: 5 (0.71)

![](images/d38b5e471ce3b99d381621d2e666f0ca4b39e3b2bfdbef344f4b4880a53669ec.jpg)  
Ground Truth: 3

![](images/ab10e47f0817b900939260e8953c87eec2a96200bfb65c2dfb1cb2b247cfc0a6.jpg)

![](images/9bdd100e7aafbd0ad097e3aa691db15ca777a239722fdfbf9affc109c875d664.jpg)  
Prediction: 3 (0.97)

![](images/e7aeb6c1521b378f78dd60a5553d83dba52745a4c129882e02eb1f7be2792236.jpg)  
Ground Truth: 7  
Figure 5: Comparison of MNIST digit reconstruction using the prediction from the network versus ground truth label. The top row shows the original images from the MNIST test set that the network misclassified. The middle row shows the reconstruction of the images, along with the incorrect class and confidence score. The bottom row shows the reconstructions using the corrected one-hot labels.

![](images/14006cccc29c9c690c83f6259abb53e1fb4b48695448e7ae0d618e8d18550ac3.jpg)

![](images/073e3d82475024168e038cbe96681ffc4aa1161ab8cc87f29c4236ebbfc67113.jpg)  
Prediction: 8 (0.84)

![](images/fee4fab7a029dd6ff662e7336c3eadefb0a28170ba9e1784591f718e50623d3a.jpg)  
Ground Truth: 5

![](images/d1eb83881fa90f33c9ed8b86a726c9c5b81da528bfcae29d586f2dde96067c0d.jpg)

![](images/1819b5438a7be2202181de8ff52f2e94c0ff70ab6244c9dbc6ad95968e9bda3c.jpg)  
Prediction: 6 (0.99)

![](images/93d22911ba5b85bdd88534502a6fafd823be7cc450263c91e639852c57601b66.jpg)  
Ground Truth: 4

![](images/5193547cd4b60b2dffd53600e869c5a52f393117196753a72404eadaf67afbba.jpg)

![](images/64c76966aec2472b1a18ae870f248ede99ee6db37645a2c7e59051466e4a842c.jpg)  
Prediction: 5 (0.77)

![](images/9bfc49a30623e63da8ad9f926dd571891d4a79208606af4e7d4722f2cc8d9173.jpg)  
Ground Truth: 6

![](images/c73811e886f73bc55a812977c572b993ec74caec22e4019f040b6257bb5c8c3f.jpg)

![](images/f43cd60821b8b1e70b71c87df3e2435691ab135ade6be4c757cfb685a18eb6fc.jpg)  
Prediction: 3 (0.99)

![](images/dcdc78bf1161b5ab673ab5ae5a4ec04f7bcbf3a55d53a3f8233359e17c55ce76.jpg)  
Ground Truth: 2

![](images/19ad89ec5f9aceacba6c98dc54391767482220b6501b420d816297d02249eff6.jpg)

![](images/c44ac7927fbe59f0942986959519000d2cb48168a7ede09c26148d3c69fcc477.jpg)

![](images/ebc569432ebdb6f47c9ceaad9dadf2c612cf737fbddb661c40978fcd7935e237.jpg)  
Figure 6: Comparison of EMNIST letter reconstruction using the prediction from the network versus ground truth label.

![](images/9494df986282fad2d32a1b79b7c789079e4005a64ce6542d8782a22de7311365.jpg)

![](images/a78c20ff3dea2dc7ec60d8a1487d12b1c4801f4f6d71f49e9929dd2ea1443176.jpg)  
Prediction: E (0.99)

![](images/2b5918102366de0b2398934d7496f1760702684857ae8f7ed653becbfe325840.jpg)

![](images/e8c49cc24f3e3b803b5689963f7846dca55bfc0aabcef40eebd0235c09950d0a.jpg)

![](images/916775bb922121b029b8f867847acf5a0841d576219eeeda11fff40eefb7962a.jpg)  
Prediction: U (0.92)

![](images/d06c7eb854ee2385c874d97dc9c0aca2faabf0513908334611e85e489f3e3173.jpg)  
Ground Truth: A

![](images/0c2ec8e2ccef21a493cd4ff9cf1c44787fba249e27437a8c9299b6774bc60e60.jpg)

![](images/48eb6b3303a6e5ab043ed075d44cca6ab06c199b934dd9209108e327eef628b8.jpg)  
Prediction: | (0.98)

![](images/d2368083cc4abb459589c85b5e93a3ac177669d8d7cb85b128f0e68de659a787.jpg)  
Ground Truth: S

![](images/63cf535a9deeff8c982522eef28cb1743b4091b7519b9f79bd874c3bd3819f76.jpg)

![](images/3e5fc698eaa03633aa5a0ae90ee21ed376d8176a23dc4a41ef77bdfed84d0e20.jpg)  
Prediction: K (0.98)

![](images/711968d49ba5025fe8e08c5cb58664ee57bf955c450f8c7546130abffcb50e23.jpg)  
Ground Truth: X

![](images/e2fdc6705eb359861b78bc457cfcf5d481de1572ae62535c9a823b5ad69c31af.jpg)

![](images/fd3f9d09051ac555c8f2dd5f3de9254fbf8438e37313456856acae30ac428859.jpg)  
Prediction: C (0.99)

![](images/f58606b5f049bbf5ce156af02d409dace3a0ca42b5028600b7c1d2c02bc7643f.jpg)  
Ground Truth E

![](images/d74842e8318d9e8178bcee88f3c94325fa43d45a03a14372abbf355a253c1f17.jpg)  
(a) Image Dist=8.6, Style Dist=1.2

![](images/422224fc8fced8540149803e8270312e24ad3e608a838fdeb0891cfa60df2a15.jpg)  
(c) Image Dist=8.5, Style Dist=1.2  
Figure 7: Nearest neighbours in image space and style-memory space. (a) and (c) show the 97 digit images closest to the image in the top-left, as well as their corresponding style-memories. (b) and (d) show the 97 style memories closest to the style memory in the top-left, as well as their corresponding digit images. The order of elements (across rows, then down) indicate increasing Euclidean distance. The subfigure captions give the average distance from the top-left element, both in image space, and style-memory space.

![](images/450f7ab5ed6a25120bb72f8abeabb729b5c7d02f2abcdaced36882216637acc1.jpg)  
(b) Image Dist=9.3, Style Dist=0.98

![](images/242f4baa563f119188d918f2b25c6842d20bd50a4dfcbc8d253cb4a266cbf0ec.jpg)  
(d) Image Dist=9.5, Style Dist=1.0

# 4.3 STYLE MEMORY REPRESENTATION

To better understand what was being encoded in the style memory, we generated digits that were close together in the style memory space (16-dimensional) and compared them with digits that are close together in the image space (784-dimensional). The distance, in either space, was calculated using the Euclidean norm.

![](images/299ea1ca9f2a69d50bec341811e8ced7bca2fc04b3bc180560fd27ff0df806a9.jpg)  
(a) Image Dist=9.1, Style Dist=1.3

![](images/b48aeccbf538bf56f65da74fdccc9191df220d9f6a79e4ccbe90f87b3a0c6670.jpg)  
(b) Image Dist=10.5, Style Dist=0.91

![](images/4ff0767bacc6bb1defb9f9aef2f8fc5d85f18fbc4a869ad161ed0cc0f51a5107.jpg)  
(c) Image Dist=8.5, Style Dist=1.3

![](images/6ad17d9f100033395e5c7fa4f8b6e593fdd36782d5984e6a54b8355f14741180.jpg)  
(d) Image Dist=9.8, Style Dist=0.99  
Figure 8: Nearest neighbours in image space and style-memory space of EMNIST dataset.

From Fig. 7 and Fig. 8, we can observe that proximity in the style-memory space has different semantic meaning than proximity in the image space. Figure 7a, showing the 97 images that are closest to the '5' image in the top-left corner, displays many digits that share common pixels. However, Fig. 7b, which shows the 97 digits with the closest style memories, displays digits that come from various different classes. Similarly, Fig. 7c shows many digits of class '3', while Fig. 7d is less dominated by digit '3'.

There are 18 digits of '5' in Fig. 7a, while there are only 13 digits of '5' in Fig. 7b. However, Fig. 7a is actually dominated by '0', even though the base digit is a '5'. There are 54 digits of '0' in Fig. 7a, while there are only 25 digits of '0' in Fig. 7b. Similarly, there are 76 digits of '3' in Fig. 7c, while there are only 46 digits of '3' in Fig. 7d. We also observed that the image distance between Fig. 7a and Fig. 7b increased from 8.6 to 9.3, while the style distance decreased from 1.2 to 0.98. The image distance between Fig. 7c and Fig. 7d also increased from 8.5 to 9.5, while the style distance decreased from 1.2 to 1.0.

Similarly, there are 52 letters of 'S' in Fig. 8a, while there are only 6 letters of 'S' in Fig. 8b. Furthermore, there are 47 letters of 'P' in Fig. 8c, while there are only 17 letters of 'P' in Fig. 8d. The image distance between Fig. 8a and Fig. 8b increased from 9.1 to 10.5, while the style distance decreased from 1.3 to 0.91. Lastly, The image distance between Fig. 8c and Fig. 8d also increased from 8.5 to 9.8, while the style distance decreased from 1.3 to 0.99.

These results show that style memory successfully separates some of the class information from the data, while not being fully class-agnostic.

# 4.4 STYLE MEMORY INTERPOLATION

In this experiment, we attempted to reconstruct a continuum of images that illustrate a gradual transformation between two different styles of the same character class. For example, we encoded two different digits for each MNIST class, as shown in Fig. 9. We then generated a sequence of

![](images/576ded7d031c1a51cd0b2eabf33a64ecbb2e208884dc6bf279f053cb4b7c2cc6.jpg)  
Figure 9: Two different styles of digits form the endpoints for the style interpolation experiment.

![](images/68f8fc5cc4b05c5835fc48f15827c94b93af0d77cdb0594579a9ea2bf4ca9ec3.jpg)  
Figure 10: Two different styles of letters form the endpoints for the style interpolation experiment.

images that slowly evolve from one style to the other. We performed the interpolation by simply taking convex combinations of the two style memories, using

$$
\hat {m} (\lambda) = \lambda m _ {1} + (1 - \lambda) m _ {2}, \tag {4}
$$

where  $m_{1}$  and  $m_{2}$  denote the style memories. The interpolated style memory is denoted by  $\hat{m} (\lambda)$  where  $\lambda \in [0,1]$  denotes the interpolation coefficient.

Figure 11 shows the interpolated digits and letters, illustrating that the generated images transform smoothly when the style memory is interpolated. The results of within-class interpolation suggest that style memory captures style information about how a digit was drawn. The figure also shows examples of attempted interpolations between incongruous letter forms (eg. 'A' to 'a', and 'r' to 'R'). Not surprisingly, the interpolated characters are nonsensical in those cases.

An obvious experiment is to try transferring the style memory of one digit onto another digit class. Although not shown here, we observed that the style memory of a digit can, in some cases, be transferred to some other classes. However, in general, the reconstructions did not look like characters.

# 5 CONCLUSIONS AND FUTURE WORK

Classification networks do not typically maintain enough information to reconstruct the input; they do not have to. Their goal is to map high-dimensional inputs to a small number of classes, typically using a lower-dimensional vector representation. In order for a classification network to be capable of generating samples, additional information needs to be maintained. In this paper, we proposed the addition of "style memory" to the top layer of a classification network. The top layer is trained using a multi-objective optimization, trying to simultaneously minimize classification error and reconstruction loss.

![](images/042ae65e9e2f62ac4540203c674902d0455b15d02c90f3ec3de7e65c99b75463.jpg)  
(a)

![](images/efdbbe537c74ab5ee85bdd9da6ac803a44538a1417034a6df9ab0ec0fff303b6.jpg)  
(b)  
Figure 11: Image reconstruction with style memory interpolation between digits and letters shown in Fig. 9 and Fig. 10, where  $\lambda$  was increasing from 0.1 to 1.0 with a step of 0.1 from top to bottom.

Our experiments suggest that the style memory encodes information that is largely disjoint from the classification vector. For example, proximity in image space yields digits that employ an overlapping set of pixels. However, proximity in style-memory space yielded a different set of digits.

For the style interpolation experiment, we generated images from a straight line in style-memory space. However, each position on this line generates a sample in image space – an image; it would be interesting to see what shape that 1-dimensional manifold takes in image space, and how it differs from straight-line interpolation in image space. However, the fact that we were able to interpolate digits and letters within the same class using novel style-memory activation patterns suggests that the style memory successfully encodes additional, abstract information about the encoded input.

To our knowledge, existing defence mechanisms to combat adversarial inputs do not involve the generative capacity of a network. Motivated by the results in Sec. 4.1, preliminary experiments that we have done suggest that treating perception as a two-way process, including both classification and reconstruction, is effective for guarding against being fooled by adversarial or ambiguous inputs. Continuing in this vein is left for future work.

Finally, we saw that the network has a property where the reconstruction generated was affected both by the classification neurons and style memory. Inspired by how human perception is influenced by expectation (Summerfield & De Lange, 2014), we believe that this work opens up opportunities to create a classifier network that takes advantage of its generative capability to detect misclassifications. Moreover, predictive estimator networks might be a natural implementation for such feed-back networks (Xu et al., 2017; Summerfield & De Lange, 2014; Orchard & Castricato, 2017). Perception and inference could be the result of running the network in feed-forward and feed-back directions simultaneously, like in the wake-sleep approach (Hinton et al., 1995). These experiments are ongoing.

# REFERENCES

Yoshua Bengio, Pascal Lamblin, Dan Popovici, and Hugo Larochelle. Greedy layer-wise training of deep networks. In Proceedings of the 19th International Conference on Neural Information Processing Systems, NIPS'06, pp. 153-160, Cambridge, MA, USA, 2006. MIT Press.  
J Bullier, ME McCourt, and GH Henry. Physiological studies on the feedback connection to the striate cortex from cortical areas 18 and 19 of the cat. Experimental Brain Research, 70(1):90-98, 1988.  
Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik. EMNIST: an extension of MNIST to handwritten letters. CoRR, abs/1702.05373, 2017.  
Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.  
Geoffrey E. Hinton. To recognize shapes, first learn to generate images. In Paul Cisek, Trevor Drew, and John F. Kalaska (eds.), Computational Neuroscience: Theoretical Insights into Brain Function, volume 165 of Progress in Brain Research, pp. 535 - 547. Elsevier, 2007.  
Geoffrey E Hinton, Peter Dayan, Brendan J Frey, and Radford M Neal. The "Wake-Sleep" Algorithm for Unsupervised Neural Networks. Science, 268(5214):1158-1161, 1995.  
Geoffrey E. Hinton, Simon Osindero, and Yee Whye Teh. A fast learning algorithm for deep belief nets. Neural Computation, 18:1527-1554, 2006.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
Hongyin Luo, Jie Fu, and James R. Glass. Bidirectional backpropagation: Towards biologically plausible error signal transmission in neural networks. CoRR, abs/1702.07097, 2017.  
Jeff Orchard and Louis Castricato. Combating Adversarial Inputs Using a Predictive-Estimator Network. In Proc. of the International Conference on Neural Information Processing, volume LNCS 10638, pp. 118-125, 2017.  
Jasper Poort, Florian Raudies, Aurel Wannig, Victor A F Lamme, Heiko Neumann, and Pieter R. Roelfsema. The role of attention in figure-ground segregation in areas V1 and V4 of the visual cortex. *Neuron*, 75(1):143-156, 2012.  
Leila Reddy, Naotsugu Tsuchiya, and Thomas Serre. Reading the mind's eye: decoding category information during mental imagery. NeuroImage, 50(2):818-825, 2011.  
Ruslan Salakhutdinov and Geoffrey Hinton. Deep Boltzmann machines. In David van Dyk and Max Welling (eds.), Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, volume 5 of Proceedings of Machine Learning Research, pp. 448-455, Hilton Clearwater Beach Resort, Clearwater Beach, Florida USA, 16-18 Apr 2009. PMLR.  
Ruslan Salakhutdinov and Geoffrey E Hinton. Learning a nonlinear embedding by preserving class neighbourhood structure. In International Conference on Artificial Intelligence and Statistics, pp. 412-419, 2007.  
Christopher Summerfield and Floris P. De Lange. Expectation in perceptual decision making: Neural and computational mechanisms. Nature Reviews Neuroscience, 15(11):745-756, oct 2014. ISSN 14710048. doi: 10.1038/nrn3838. URL http://dx.doi.org/10.1038/nrn3838.  
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. URL http://arxiv.org/abs/1312.6199.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res., 11:3371-3408, December 2010.  
D. Xu, A. Clappison, C. Seth, and J. Orchard. Symmetric predictive estimator for biologically plausible neural learning. IEEE Transactions on Neural Networks and Learning Systems, PP(99): 1-12, 2017. ISSN 2162-237X. doi: 10.1109/TNNLS.2017.2756859.