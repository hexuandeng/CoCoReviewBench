# INTROSPECTION: ACCELERATING NEURAL NETWORK TRAINING BY LEARNING WEIGHT EVOLUTION

# Abhishek Sinha*

Department of Electronics and Electrical Comm. Engg.

IITKharagpur

West Bengal, India

abhishek.sinha94 at gmail dot com

# Aahitagni Mukherjee*

Department of Computer Science

IIT Kanpur

Uttar Pradesh, India

ahitagnimukherjeeam at gmail dot com

# Mausoom Sarkar

Adobe Systems Inc, Noida

Uttar Pradesh,India

msarkar at adobe dot com

# Balaji Krishnamurthy

Adobe Systems Inc, Noida

Uttar Pradesh,India

kbalaji at adobe dot com

# ABSTRACT

Neural Networks are function approximators that have achieved state-of-the-art accuracy in numerous machine learning tasks. In spite of their great success in terms of accuracy, their large training time makes it difficult to use them for various tasks. In this paper, we explore the idea of learning weight evolution pattern from a simple network for accelerating training of novel neural networks. This method can also be used with other optimizers to give faster convergence.

We use a neural network to learn the training pattern from MNIST classification and utilize it to accelerate training of neural networks used for CIFAR-10 and ImageNet classification. Our method has a low memory footprint and is computationally efficient. The results indicate a general trend in the weight evolution during training of any neural network.

# 1 INTRODUCTION

Deep neural networks have been very successful in modeling high-level abstractions in data. However, training a deep neural network for any AI task is a time-consuming process. This is because a large number of parameters need to be learnt using training examples. Most of the deeper networks can take days to get trained even on GPU thus making it a major bottleneck in the large-scale application of deep networks. Reduction of training time through an efficient optimizer is essential for fast design and testing of deep neural nets.

In the context of neural networks, an optimization algorithm iteratively updates the parameters (weights) of a network based on a batch of training examples, to minimize an objective function. The most widely used optimization algorithm is Stochastic Gradient Descent. Even with the advent of newer and faster optimization algorithms like Adagrad, Adadelta, RMSProp, there is still a need for achieving faster convergence.

In this work we apply neural network to predict weights of other in-training neural networks to accelerate their convergence. Our method has a very low memory footprint and is computationally efficient. Another aspect of this method is that we can update the weights of all the layers in parallel.

# 2 RELATED WORK

Several extensions of Stochastic Gradient Descent have been proposed for faster training of neural networks. Some of them are Momentum(Rumelhart et al., 1986), AdaGrad(Duchy et al., 2011), AdaDelta(Zeiler, 2012), Adam(Kingma & Ba, 2014), RMSProp(Hinton et al., 2012). All of them

reduce the convergence time by suitably altering the learning rate during training. Our method can be used along with any of the above-mentioned methods to further improve convergence time.

In the above approaches, the weight update is always a product of the gradient and the modified/unmodified learning rate. More recent approaches(Andrychowicz et al., 2016) have tried to learn the function that takes as input the gradient and outputs the appropriate weight update. This exhibited a faster convergence compared to a simpler multiplication operation between the learning rate and gradient. Our approach is different from this, because Introspection Network does not use the current gradient for weight update, but rather uses the weight history to predict its future value many time steps ahead where network would exhibit better convergence. Our approach generalizes better between different architectures and datasets without additional retraining. Further our approach has far lesser memory footprint as compared to (Andrychowicz et al., 2016). Also our approach need not be involved at every weight update and hence can be invoked asynchronously which makes it computationally efficient.

Another recent approach, called Q-gradient descent(Fu et al., 2016), uses a reinforcement learning framework to tune the hyperparameters of the optimization algorithm as the training progresses. The Deep-Q Network used for tuning the hyperparameters itself needs to be trained with data from any specific network  $N$  to be able to optimize the training of  $N$ . Our approach is different because we use a pre-trained Introspection Network that can optimise any network  $N$  without training itself by data from  $N$ .

Finally the recent approach by (Jaderberg et al., 2016) to predict synthetic gradients is similar to our work, in the sense that the weights are updates independently, but it still relies on an estimation of the gradient, while our update method does not.

Our method is distinct from all the above approaches because it uses information obtained from the training process of existing neural nets to accelerate the training of novel neural nets.

# 3 PATTERNS IN WEIGHT EVOLUTION

Experiments were performed on different classification tasks such as on MNIST, CIFAR-10 and ImageNet dataset to observe the evolution of the weights of different layers, both conv and fully connected, with the progress of training. The experiments were varied over different network architectures as well as different optimization rules. It was observed that the evolution followed a general trend independent of what task the model was performing or to which layer the parameters belonged to. While a major proportion of the weights did not undergo any significant change, the few that did change followed an observable trend. These weight values would keep on increasing or decreasing with the progress of training in a predictable fashion.

# 3.1 WEIGHT PREDICTION

We collect the weight evolution trends of a network that are being trained and use the collected data to train a neural network  $I$  to forecast the future values of each weight based on its values in the previous time steps. The trained network  $I$  is then used to predict the weight values of an unseen network  $N$  during its training which move  $N$  to a state that enables a faster convergence. The time taken for the forecast is significantly smaller compared to the time a standard optimizer (e.g. SGD) would have taken to achieve the same accuracy. This leads to a reduction in the total training time. The predictor  $I$  that is used for forecasting weights is a comparatively smaller neural network, whose inference time is negligible compared to the training time of the network that needs to be trained(N).

The forecasting network I is a simple 1-layered feedforward neuralnet. The input layer consists of four neurons that take four samples from the training history of a weight. The hidden layer consists of 40 neurons, fully connected to the input layer, with ReLU activation. The output layer is a single neuron that outputs the predicted future value of the weight.

The figure 1 below shows a comparison of the weight evolution for a single scalar weight value with and without using the introspection network  $I$ . The vertical green bars indicate the points at which the introspection network was used to predict the future values. Post prediction, the network

continues to get trained normally by SGD, until the introspection network  $I$  is used once again to jump to a new weight value.

![](images/a77b812e3ea54862206dfc88e43d5c4c9c1c15bc14dcd4ce8ec7c46d3f5140d9.jpg)  
Figure 1: Example of weight update using Introspection Network.

# 4 EXPERIMENTS

# 4.1 TRAINING OF INTROSPECTION NETWORK

The introspection network  $I$  is trained on the training history of the weights of a network  $N_0$  which was trained on MNIST dataset. The  $N_0$  consisted of 3 convolutional layers and two fully connected layers. The training set of  $I$  is prepared as follows. A random training step  $t$  is selected for each weight of  $N_0$  and the following 4 values are given as inputs for training  $I$ :

1. value of the weight at step  $t$  
2. at step 0 (i.e. the initialized value)  
3. value of the weight at step  $4t / 10$  
4. value of the weight at step  $7t / 10$

The expected output of  $I$ , which is used for training  $I$  using backpropagation, is a single scalar the value of the same weight at step  $2t$ . All of these time steps are empirical choices. For example, any step  $kt$  with  $k > 1$  can be chosen instead of  $2t$ . Approximately 0.8 million examples of weight history are used to train  $I$ .

# 4.2 USING PRE-TRAINED INTROSPECTION NETWORK TO TRAIN UNSEEN NETWORKS

The introspection network once trained can be then used to guide the training of other networks. We illustrate our method by using it to accelerate the training of several deep neural nets with varying

architectures on 3 different datasets, namely MNIST, CIFAR10 andImagenet. We note that the same introspection network  $I$ , trained on the weight evolutions of the MNIST network  $N_0$  was used in all these different cases.

All the networks have been trained using Stochastic Gradient Descent, and the network  $I$  is used at a few intermediate steps to propel the network to a state with higher accuracy. We refer to the time step at which the introspection network  $I$  is applied to update all the weights as a "jump point".

The selection of the steps at which  $I$  is to be used is dependent on the distribution of the training step  $t$  used for training  $I$ . We show the effect of varying the timing of the initial jump and the time interval between jump points in section 4.2.2. It has been observed that  $I$  gives a better increase in accuracy when it is used in later training steps rather than in the earlier ones.

All the networks trained using  $I$  required comparatively less time to reach the same accuracy as normal SGD training. Also, when the same network was trained for the same time with and without updates by  $I$ , the former is observed to have better accuracy. These results show that there is a remarkable similarity in the weight evolution trajectories across network architectures, tasks and datasets.

# 4.2.1 MNIST

Two different neural networks were trained using  $I$  on MNIST dataset:

1. A convolutional neural network  $MNIST_{1}$  with 3 convolutional layers and 3 fully connected layers for classification task on MNIST image dataset. It takes approximately 30,000 steps for convergence. For  $MNIST_{1}$ ,  $I$  was used to update all weights at training step 7000, 8000, and 10000.  
2. A convolutional network  $MNIST_{2}$  with 2 convolutional layers and 2 fully connected layers. It takes approximately 10,000 steps for convergence. The network  $I$  was used to update weights at training step 2500 and 3000.

A comparison of the validation accuracy with and without updates by  $I$  is shown in figures 2 and 3. The green lines indicate the steps at which the introspection network  $I$  is used. For the MNIST1 network with the application of the introspection network  $I$  at three points, we found that it took 251 seconds and 20000 SGD steps to reach a validation accuracy of  $98.22\%$ . In the same number of SGD steps, normal training was able to reach a validation accuracy of only  $97.22\%$ . In the same amount of time (251 seconds), normal training only reached  $97.92\%$ . Hence the gain in accuracy with the application of introspection network translates to real gains in training times.

For the MNIST2 network, the figure 3 shows that to reach an accuracy of  $99.11\%$ , the number of iterations required by normal SGD was 6000, whereas with the application of the introspection network  $I$ , the number of iterations needed was only 3500, which represents a significant savings in time and computational effort.

The initial drop in accuracy seen after a jump in MNIST2 figure 3 can be attributed to the fact that each weight scalar is predicted independently, and the interrelationship between the weight scalars in a layer or across different layers is not taken into consideration. This interrelationship is soon reestablished after few SGD steps. This phenomenon is noticed in the CIFAR andImagenet cases too.

# 4.2.2 CIFAR-10

We applied our introspection network  $I$  on a CNN  $CIFAR_{1}$  for classifying images in the CIFAR10(Krizhevsky, 2009) dataset. It has 2 convolutional layers, 2 fully connected layer and a final softmax layer. Max pooling and batch normalization has been applied after each convolutional layer. It takes approximately 50,000 steps for convergence. The experiments on  $CIFAR_{1}$  were done to investigate two issues. The first was to investigate if the introspection network trained on MNIST weight evolutions is able to generalize to a different network and different dataset. The second was to investigate the effect of varying the timing of the initial jump, the interval between successive jumps and the number of jumps. To investigate these issues, four separate training instances were performed with 4 different set of jump points:

![](images/a8adc5442b32caafa39601fecd01832d2f6c7dd96351003466910b2b0c7c2f1a.jpg)  
Figure 2: Validation accuracy plot for  $MNIST_{1}$

![](images/db6ba30707d90a6ff529d5d231f95a5b00c57fa13e4e3edf0895f40856b24f9b.jpg)  
Figure 3: Validation accuracy plot for  $MNIST_{2}$

1.  $Set_{1}$ : Weight updates were carried out at training steps 12000 and 17000.  
2.  $Set_{2}$  : Weight updates at steps 15000 and 18000.  
3.  $Set_{3}$  : Weight updates at steps 12000, 15000 and 19000.  
4.  $Set_4$  : Weight updates at steps 14000, 17000 and 20000.

We observed that for the  $CIFAR_{1}$  network that in order to reach a validation accuracy of  $85.7\%$ , we need 40,000 iterations with normal SGD without any intervention with the introspection network  $I$ . In all the four sets where the introspection network was used, the target accuracy of  $85.7\%$  was reached in approximately 28,000 steps. This shows that the introspection network is able to successfully generalize to a new dataset and new architecture and show significant gains in training time.

On  $CIFAR_{1}$ , the time taken by  $I$  for prediction is negligible compared to the time required for SGD. So the training times in the above cases on  $CIFAR_{1}$  can be assumed to be proportional to the number of SGD steps required.

A comparison of the validation accuracy with and without updates by  $I$  at the four different sets of jump points are shown in figures 4, 5, 6 and 7. The results show that the while choice of jump points have some effect on the final result, the effects are not very huge. In general, we notice that better accuracy is reached when the jumps take place in later training steps.

![](images/8d8096d96d0aa1ce20a97f64be1c89325fd7dac33ee80cd6209dd30ff119a62e.jpg)  
Figure 4: Validation accuracy plot for  $CIFAR_{1}$  with jumps at  $Set_{1}$

![](images/f8c7c69f14367882eb6ce11de3d082239664b9aeba2e44011f860d652cddeb65.jpg)  
Figure 5: Validation accuracy plot for  $CIFAR_{1}$  with jumps at  $Set_{2}$

![](images/edf0014ce34f1ccded45d5a6b8931c90ea32057b2602397111422fd0a3606e25.jpg)  
Figure 6: Validation accuracy plot for  $CIFAR_{1}$  with jumps at Set3

![](images/26f1e0548c84d7f5b3424c6d7fc4796347085996a64cb9f71cedc8a9d022b159.jpg)  
Figure 7: Validation accuracy plot for  $CIFAR_{1}$  with jumps at Set4

# 4.2.3 IMAGENET

To investigate the practical feasibility and generalization ability of our introspection network, we applied it in training alexnet  $(Alexnet_{1})$  on the ImageNet(Russakovsky et al., 2015) dataset. It has 5 conv layers and 3 fully connected layers. Max pooling and local response normalization have been used after the two starting conv layers and the pooling layer is there after the fifth conv layer as well. It takes approximately 300,000 steps for convergence. The weight updates were carried out at training steps 120,000, 130,000, 144,000 and 160,000.

We find that in order to achieve a top-5 accuracy of  $72\%$ , the number of iterations required in the normal case was 196,000. When the introspection network was used, number of iterations required to reach the same accuracy was 179,000. Again the time taken by  $I$  for prediction is negligible compared to the time required for SGD. A comparison of the validation accuracy with and without updates by  $I$  is shown below. The green lines indicate the steps at which the introspection network  $I$  is used.

![](images/40077fabc08fd121d8c7a2d544e6527db6870a060408cc72348384f428877817.jpg)  
Figure 8: Validation accuracy plot for Alexnet on ImageNet

The results on  $Alexnet_{1}$  show that our approach has a small memory footprint and computationally efficient to be able to scale to training practical large scale networks.

# 5 LIMITATIONS AND OPEN QUESTIONS

Some of the open questions to be investigated relate to determination of the optimal jump points and investigations regarding the generalization capacity of the introspection network to speed up training in RNN and non-image training. Also, we noticed that applying the jumps in very early training steps while training  $Alexnet_{1}$  tended to degrade the final outcomes. This may be due to the fact that our introspection network is extremely simple and has been trained only on weight evolution data fro MNIST. A combination of a more powerful network and training data derived from a diverse set may ameliorate this problem.

# 6 CONCLUSION

We introduced a method to accelerate neural network training. For this purpose, we used a neural network  $I$  that learns a general trend in weight evolution of all neural networks. After learning the trend from one neural network training,  $I$  is used to update weights of 3 deep neural nets on 3 different tasks - MNIST, CIFAR-10, and ImageNet, which led to faster convergence compared to existing methods in all the 3 cases. Our method has a small memory footprint, is computationally efficient and is usable in practical settings. Our method is different from other existing methods in the aspect that it utilizes the knowledge obtained from weights of one neural network training to accelerate the training of several unseen networks on new tasks. The results reported here indicates the existence of a general underlying pattern in the weight evolution of any neural network.

# REFERENCES

Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, and Nando de Frietas. Learning to learn by gradient descent by gradient descent. 2016. URL https://arxiv.org/pdf/1606.04474v1.pdf.  
John Duchy, Elad Hazan, and Yoram Singer. Adaptive Subgradients Method For Online Learning and Stochastic Optimization. 2011. URL http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.  
Zie Fu, Zichuan Lin, Danlu Chen, Miau Liu, Nicholas Leonard, Jiashi Feng, and Tat-Seng Chua. Deep Q-Networks for Accelerating the Training of Deep Neural Networks. 2016. URL https://arxiv.org/pdf/1606.01467v3.pdf.  
Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Lecture 6a: Overview of mini-batch gradient descent. 2012. URL https://class.coursera.org/neuralnets-2012-001/lecture.  
Max Jaderberg, Wojciech M. Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Decoupled Neural Interfaces using Synthetic Gradients. 2016. URL https://arxiv.org/pdf/1608.05343.pdf.  
Diedirik P. Kingma and Jimmy Lei Ba. Adam: A Method For Stochastic Optimization. 2014. URL https://arxiv.org/pdf/1412.6980v8.pdf.  
Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.  
David Rumelhart, Geoffrey Hinton, and Ronald Williams. Learning representations by backpropagating errors. 1986. URL http://www.nature.com/nature/journal/v323/n6088/abs/323533a0.html.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. 2015.  
Matthew D. Zeiler. Adadelta: An adaptive learning method. 2012. URL https://arxiv.org/pdf/1212.5701v1.pdf.