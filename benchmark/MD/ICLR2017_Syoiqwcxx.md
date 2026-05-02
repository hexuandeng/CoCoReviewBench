# LOCAL MINIMA IN TRAINING OF DEEP NETWORKS

Grzegorz Świrszcz, Wojciech Marian Czarnecki & Razvan Pascanu  
DeepMind  
London, UK  
{swirszcz, lejlot, razp} @ google.com

# ABSTRACT

There has been a lot of recent interest in trying to characterize the error surface of deep models. This stems from a long standing question. Given that deep networks are highly nonlinear systems optimized by local gradient methods, why do they not seem to be affected by bad local minima? It is widely believed that training of deep models using gradient methods works so well because the error surface either has no local minima, or if they exist they need to be close in value to the global minimum. It is known that such results hold under very strong assumptions which are not satisfied by real models. In this paper we present examples showing that for such theorem to be true additional assumptions on the data, initialization schemes and/or the model classes have to be made. We look at the particular case of finite size datasets. We demonstrate that in this scenario one can construct counter-examples (datasets or initialization schemes) when the network does become susceptible to bad local minima over the weight space.

# 1 INTRODUCTION

Deep Learning (LeCun et al., 2015; Schmidhuber, 2015) is a fast growing subfield of machine learning, with many impressive results. One particular criticism often brought up against this family of models is the fact that it relies on non-convex functions which are optimized using local gradient descent methods. This means one has no guarantee that the optimization algorithm will converge to a meaningful minimum or even that it will converge at all. However, this theoretical concern seems to have little bearing in practice.

In Dauphin et al. (2013) a conjecture had been put forward for this based on insights from statistical physics which point to the scale of neural networks as a possible answer. The claim is that the error structure of neural networks might follow the same structure as that of random Gaussian fields which have been recently understood and studied in Fyodorov & Williams (2007); Bray & Dean (2007). The critical points of these functions, as the dimensionality of the problem increases, seem to have a particularly friendly behaviour where local minima align nicely close to the global minimum of the function. Choromanska et al. (2015) provides a study of the conjecture by mapping deep neural models onto spin glass ones for whom the above structure holds. These work has been extended further (see Section 2 for a review of the topic).

We believe many of these results do not trivially extend to the case of finite size datasets/finite size models. The learning dynamics of the neural network in this particular case can be arbitrarily bad. Our assertions are based on constructions of counterexamples that exploit particular architectures, the full domain of the parameters and particular datasets.

# 2 LITERATURE REVIEW

One view, that can be dated back to Baldi & Hornik (1989), about why the error surface of neural networks seems well behaved is the one stated in Dauphin et al. (2013). We would refer to this hypothesis as the "no bad local minima" hypothesis. In Baldi & Hornik (1989) it is shown that an MLP with a single linear intermediate layer has no local minima, only saddle points and a global minimum. This intuition is carried further by Saxe et al. (2014; 2013), where deep linear models are studied. While, from a representational perspective, deep linear models are not useful, the hope is

that the learning dynamics of such models can be mathematically understood while still being rich enough to mirror the dynamics of nonlinear networks. The findings of these works are aligned with Baldi & Hornik (1989) and suggest that one has only to go through several saddles to reach a global minimum.

These intuitions are expressed clearly for generic deep networks in Dauphin et al. (2013). The key observation of this work is that intuitions from low dimensional spaces are usually misleading when moving to high dimensional spaces. The work makes a connection with deep results obtained in statistical physics. In particular Fyodorov & Williams (2007); Bray & Dean (2007) showed, using the Replica Theory (Parisi, 2007), that random Gaussian error functions have a particular friendly structure. Namely, if one looks at all the critical points of the function and plots error versus the (Morse) index of the critical point (the number of negative eigenvalues of the Hessian) these points align nicely on a monotonically increasing curve. That is, all points with a low index (note that every minimum has to have this index equal to 0) have roughly the same performance, while critical points of high error implicitly have a large number of negative eigenvalue which means they are saddle points.

These observations also align with the theory of random matrices (Wigner, 1958) which predicts the same behaviour for the eigenvalues of a random matrix as the size of the matrix grows. The claim of Dauphin et al. (2013) is that same structure holds for neural network as well when they become large enough. Similar claim is put forward in Sagun et al. (2014). The conjecture is very appealing as it provides a strong argument why deep networks end up performing not only well, but also reliably so. Choromanska et al. (2015) provides a study of the conjecture that rests on recasting a neural network as a spin-glass model for which the Replica Theory can be applied directly to re-derive the above mentioned results. To obtain this result several assumptions need to be made, which the authors of the work, at that time, acknowledged that were not realistic in practice. The same line of attack is taken by Kawaguchi (2016).

Goodfellow et al. (2016) argues and provides empirical evidence that while moving from the original initialization of the model along a straight line to the solution (found via gradient descent) the loss seems to be only monotonically decreasing, which speaks towards the apparent convexity of the problem. Soudry & Carmon (2016); Safran & Shamir (2015) also look at the error surface of the neural network, providing theoretical arguments for the error surface becoming well-behaved in the case of over-parametrized models.

A different view, presented in Lin & Tegmark (2016); Shamir (2016), aligned with this work, is that the underlying easiness of optimizing deep networks does not simply rest just in the emerging structures due to high dimensional spaces, but is rather tightly connected to the intrinsic characteristics of the data these models are run on.

# 3 FINITE DATASETS FOR RECTIFIED MLPS

We propose to analyze the error surface of rectified MLPs on finite datasets. The approach we take is a construction one where we build examples of datasets and model initializations that result in bad learning dynamics.

# 3.1 EXAMPLES OF BAD LOCAL MINIMA

We start our examples with experiments showing that bad initialization can lead to training getting stuck in a local minimum on MNIST dataset.

# 3.1.1 BAD INITIALIZATION ON MNIST

Figure 1 shows the training error of rectified MLP on the MNIST dataset for different seeds and different model sizes. The learning algorithms used is Adam (Kingma & Ba, 2014) and everything except initialization, when specifically stated, follows an accepted protocol (see Appendix A). The results show that models that are not initialized in a good interval do not seem to converge to a good solution of the problem even after 1,000,000 updates. Depth does not seem to be able to resolve the bad initialization of the model. The bottom row experiments are similar to those presented in Zhang et al. (2017), though more limited in their scope. They explore the correlation between

the structure in the data and learning, and, at least in appearance, they do not seem to support our working hypothesis that the structure is essential. It is worth noticing though that the initialization is even more important in that setting; destroying the structure makes the model significantly more susceptible to bad initializations than when trained on the data with unpermuted labels (second column of Figure 1, the network requires at least 400 units to be able to achieve 0 training error).

![](images/7095a728fbed2a9e7a495c34172e14fa1f4d37823bc933325dd95b4ec9c0f6ca.jpg)  
Figure 1: Plots of final training accuracy on MNIST dataset after 1,000,000 updates. Each point is a single neural net (blue triangles - 5 layer models with same number of hidden units in each layer, red triangles - 2 layer models with same number of hidden units in each layer). The title of each column shows the distribution used to initialize weights  $(w)$  and biases  $(b)$ . Top row shows results on MNIST, bottom row shows results when the labels of MNIST had been randomly permuted. The number of hidden units per layer is indicated on x-axis.

The bad initializations used in these experiments are meant to target the blind spots of the rectifier model. The main idea is that by changing the initialization of the model (the mean of the normal distribution used to sample weights) one can force all hidden units to be deactivated for the most or for all examples in the training set. This prevents said examples from being learned, even though the task might be linearly separable. The construction may seem contrived, but it has important theoretical consequences. It shows that one can not prove well behaved learning for finite sized neural networks, when applied to finite sized data, without taking into account the initialization or data. The results can be generalized to other loss functions or distributions used to initialize the weights. We formalize this idea in the Proposition 4, making the observation that the effect can be achieved by either changing the initialization of the model, or the data. In particular, by introducing particular outliers, one can force most of the data examples in the dataset to be in the blind spot of the neural network.

![](images/56a13882846b5c5840340d7de5dadcff5ae7e3690bcebf457cc1b3271f615cd9.jpg)  
Figure 2: Plots of the final train accuracy on scaled MNIST dataset after 1,200,000 updates of a single hidden layer neural net. The title of each column shows the scaling factor applied to the data.

Figure 2 provides an empirical evidence towards a different mechanism of constructing a bad learning behaviour. Details of the experimental setup are given in Appendix A. To answer the results of

<table><tr><td>h</td><td></td><td>XOR ReLU</td><td>XOR Sigmoid</td><td>Jellyfish ReLU</td><td>Jellyfish Sigmoid</td><td></td><td>XOR ReLU</td><td>XOR Sigmoid</td><td>Jellyfish ReLU</td><td>Jellyfish Sigmoid</td></tr><tr><td>2</td><td>Adam</td><td>28%</td><td>79%</td><td>7%</td><td>0%</td><td>GD</td><td>23%</td><td>90%</td><td>16%</td><td>62%</td></tr><tr><td>3</td><td>Adam</td><td>52%</td><td>98%</td><td>34%</td><td>0%</td><td>GD</td><td>47%</td><td>100%</td><td>33%</td><td>100%</td></tr><tr><td>4</td><td>Adam</td><td>68%</td><td>100%</td><td>50%</td><td>2%</td><td>GD</td><td>70%</td><td>100%</td><td>66%</td><td>100%</td></tr><tr><td>5</td><td>Adam</td><td>81%</td><td>100%</td><td>51%</td><td>27%</td><td>GD</td><td>80%</td><td>100%</td><td>68%</td><td>100%</td></tr><tr><td>6</td><td>Adam</td><td>91%</td><td>100%</td><td>61%</td><td>17%</td><td>GD</td><td>89%</td><td>100%</td><td>69%</td><td>100%</td></tr><tr><td>7</td><td>Adam</td><td>97%</td><td>100%</td><td>69%</td><td>58%</td><td>GD</td><td>89%</td><td>100%</td><td>86%</td><td>100%</td></tr></table>

Table 1: "Convergence" rate for  $2-h-1$  network with random initializations on simple 2 dimensional datasets using either Adam or Gradient Descent (GD) as an optimizer.

Figure 1 (bottom row), we speculate that perhaps (from an optimization perspective) the important relationship is not only the one between the inputs and targets, but also between the inputs and the way the model partitions the input space (in here we focus on rectifier models which are, from a mathematical perspective, piece-wise linear functions). To empirically test if this is a viable hypothesis we consider the MNIST dataset, where we scale the inputs by a factor  $\tau$ . The intuition is not to force the datasets into the blind spot of the model, but rather to concentrate most of the datapoints in very few linear regions (given by the initialization of the MLP). While these results do not necessarily point towards the model being locked in a bad minimum, they suggest that learning becomes less well behaved.

# 3.1.2 THE JELLYFISH - LOCAL MINIMA FOR CLASSIFICATION USING SIGMOIDS

To improve our understanding of learning dynamics beyond exploiting blind spots, we look at one of the most theoretically well-studied datasets, the XOR problem. We analyze the dataset using a single hidden layer network (with either ReLU units or sigmoid units).

A first observation is that while SGD can solve the task with only 2 hidden units, full batch methods do not always succeed. Switching from gradient descent to more aggressive optimizers like Adam does not seem to help, but rather tend to make it more likely to get stuck in suboptimal solutions (Table 3.1.2).

![](images/c4ee209765ad1864292dc08fb8377bc3c857f3a75594133694198f11f91187bf.jpg)  
(a) Optimally converged net for Jellyfish.

![](images/fc41aba4ae6400ffe3e057d4c9e99f8684478793b7a347e26abca5b298fd26bb.jpg)  
Figure 3: Examples of different outcomes of learning on the Jellyfish dataset.

![](images/1cebcbdc789afeaf74ef20dad7c2be8e6d58f2d41c78ee6c35208388746fd228.jpg)

![](images/d975fff548dc72aa4bf9549ef5eb358d46ffbfe3172fd2c5cfe51b36db4da002.jpg)

![](images/52a3ae57faf594ea4e6a11c39f386a9627504897c1c4572eb4deed3392b4064e.jpg)  
(b) Stuck net for Jellyfish.

![](images/5891437d463d0603f4edaf8d00e71009af27c3d2c93c2db56de60c0ae1793eaf.jpg)

![](images/e421eae69a7c1cd951508c0d93391de00d9dc354d40eb8d51f16e9e76e0fd14d.jpg)

![](images/2cf63572bc48cd69e82c469cfa10ec774e792e1cd9b726b39a557af73fa27149.jpg)

By exploiting observations made in the failure modes observed for the XOR problem, we were able to construct a similar dataset, the jellyfish, that results in suboptimal learning dynamics. The dataset is formed of four datapoints, where the positive class is given by  $[1.0, 0.0]$ ,  $[0.2, 0.6]$  and the negative one by  $[0.0, 1.0]$ ,  $[0.6, 0.2]$ . The datapoints can be seen in the Figure 3.

Compared to the XOR problem it seems the jellyfish problem poses even more issues, especially for ReLU units, where with 4 hidden units one still only gets 2 out of 3 runs to end with 0 training error (when using GD). One particular observation (see Figure 3) is that in contrast with good solutions, when the model fails on this dataset, its behaviour close to the datapoints is almost linear. We argue hence, that the failure mode might come from having most datapoints concentrated in the same linear region of the model (in the case of the rectifiers), hence forcing the model to suboptimally fit these points.

# 3.1.3 LOCAL MINIMA IN A RECTIFIER-BASED REGRESSION

Rectifier networks are the most commonly used architecture for both classification and regression tasks (e.g. in deep reinforcement learning (Mnih et al., 2015; 2016)). We focus our attention on regression tasks. We start with some empirical results in Figure 5, on simple Zig-Zag regression task (see the right panel for a description of the dataset). As for the MNIST case, the experiments suggest that as data becomes more concentrated in the same linear regions (of the freshly initialized model) the learning becomes really hard, even if the model has close to 3000 units.

We follow these intuitions, in Figure 4, with 3 examples of local minima for regression using a single layer with 1, 2 and 3 hidden rectifier units on 1-dimensional data. For the sake of simplicity of our presentation we will describe in detail the case with 1 hidden neuron, the other two cases can be treated similarly. In case of one hidden neuron the regression problem becomes

$$
\underset {w, b, v, c} {\arg \min } \mathcal {L} (w, b, v, c) = \sum_ {i = 1} ^ {n} \left(v \cdot \operatorname {R e L U} \left(w x _ {i} + b\right) + c - y _ {i}\right) ^ {2}. \tag {1}
$$

Consider a dataset  $\mathcal{D}_1$  (see Figure 4 (a)):

$$
(x _ {1}, y _ {1}) = (5, 2), (x _ {2}, y _ {2}) = (4, 1), (x _ {3}, y _ {3}) = (3, 0), (x _ {4}, y _ {4}) = (1, - 3), (x _ {5}, y _ {5}) = (- 1, 3).
$$

Proposition 1. For the dataset  $\mathcal{D}_1$  and  $\mathcal{L}$  defined in Equation (1) the point  $v = 1, b = -3, w = 1, c = 0$  is a local minimum of  $\mathcal{L}$ , which is not a global minimum.

Proof. See Appendix B.4.

![](images/114b61088cbc46b04eb40f6962a43aca8fa42a2fae7f356adfe39170bdb0a978.jpg)

Remark 1. The point  $(1, -3, 1, 0)$  is a minimum, but it is not a "strict" minumum - it is not isolated, but lies on a 1-dimensional manifold at which  $\mathcal{L} \equiv 18$  instead.

One could ask whether blind spots are the only reasons for bad behaviour of rectifier nets. The answer is actually negative, and as following examples show - they can be completely absent in local optima, at the same time existing in a global solution!

![](images/2db17fccaf1b8815c10f144257e68ea8f56ff91b9a8c366ad5675577e0327975.jpg)  
(a) Two local minima for 1 hidden neuron.

![](images/8fcc7f01ec163e67e2344fb1bd58c7091d1efe8b049640856c1436621023a189.jpg)  
(b) Two local minima for 2 hidden neurons.

![](images/74f34c7921c46e0a86a0f9f5450ea9d5468cb2767e0346c962141c82121ea936.jpg)  
(c) Two local minima for 3 hidden neurons.  
Figure 4: Local minima for ReLU-based regression. Both lines represent local optima, where the blue one is better than the red one.

Proposition 2. Let us consider a dataset  $\mathcal{D}_2$  with  $d = 1$ , given by points  $(x_1, y_1) = (-1, 5)$ ,  $(x_2, y_2) = (0, 0)$ ,  $(x_3, y_3) = (1, -1)$ ,  $(x_4, y_4) = (10, -3)$ ,  $(x_5, y_5) = (11, -4)$ ,  $(x_6, y_6) = (12, -5)$  (Figure 4(b)). Then, for a rectifier network with  $m = 2$  hidden units and a squared error loss the set of weights  $\mathbf{w} = (-5, -1)$ ,  $\mathbf{b} = (1, -8)$ ,  $\mathbf{v} = (1, -1)$ ,  $c = -1$  is a global minimum (with perfect fit) and the set of weights  $\mathbf{w} = (-3, -1)$ ,  $\mathbf{b} = (4 + \frac{1}{3}, -10)$ ,  $\mathbf{v} = (1, -1)$ ,  $c = -3$  is a suboptimal local minimum.

Proof. Analogous to the previous one.

![](images/532c4882d0d2515874df81af7335f445d89acc3d149c77dee189700f83c0587a.jpg)

Maybe quite surprisingly, the global solution now has a blind spot since all neurons deactivate in  $x_{3}$ , nevertheless the network still attains 0 training error. This shows that even though blind spots were used previously to construct very bad examples for neural nets, sometimes they are actually needed to fit the dataset.

![](images/bd3aa78f29b3965897b00bf5c9e314ecca67c8cc844497e31c47355da91883c5.jpg)  
Figure 5: Plots of training MSE error on the Zig-Zag regression task after 2,000,000 updates. See caption of Figure 2 for more details. Right figure depicts the Zig-Zag regression task with three found solutions for  $\tau = 0.01$ . The actual datapoints are shown by the diamond shaped dots.

![](images/01ff73e03e99f9e42437e6175d8d93a22d07d9dae7af9fe67445e1ef4df5db9b.jpg)

Proposition 3. Let us consider a dataset  $\mathcal{D}_3$  with  $d = 1$ , given by points  $(x_1, y_1) = (-1, 3)$ ,  $(x_2, y_2) = (0, 0)$ ,  $(x_3, y_3) = (1, -1)$ ,  $(x_4, y_4) = (10, -3)$ ,  $(x_5, y_5) = (11, -4)$ ,  $(x_6, y_6) = (12, -6)$  (Figure 4 (c)). Then, for a rectifier network with  $m = 3$  hidden units and a squared error loss the set of weights  $\mathbf{w} = (-1.5, -1.5, 1.5)$ ,  $\mathbf{b} = (1, 0, -13 - \frac{1}{6})$ ,  $\mathbf{v} = (1, 1, -1)$ ,  $c = -1$  is a better local minimum than the local minimum obtained for  $\mathbf{w} = (-2, 1, 1)$ ,  $\mathbf{b} = (3 + \frac{2}{3}, -10, -11)$ ,  $\mathbf{v} = (1, -1, -1)$ ,  $c = -3$ .

Proof. Completely analogous, using the fact that in each part of the space linear models are either optimal linear regression fits (if there is just one neuron active) or perfect (0 error) fit when two neurons are active and combined.  $\square$

Note that again that the above construction is not relying on the blind spot phenomenon. The idea behind this example is that if, due to initial conditions, the model partitions the input space in a suboptimal way, it might become impossible to find the optimal partitioning using gradient descent. Let us call  $(- \infty, 6)$  the region I, and  $[6, \infty)$  the region II. Both solutions in Proposition 3 are constructed in such way that each one has the best fit for the points assigned to any given region, the only difference being the number of hidden units used to describe each of them. In the local optimum two neurons are used to describe region II, while only one describes region I. Symmetrically, the better solution assigns two neurons to region I (which is more complex) and only one to region II.

We believe that the core idea behind this construction can be generalized (in a non-trivial way) to high dimensional problems. We plan to extend the construction as future work.

# 3.2 THEORETICAL RESULTS

In this subsection we prove some general results regarding bad initialization phenomenon.

Proposition 4. There exists a normalized (whitened) dataset, such that for any feed forward rectifier network with weights initialized from a normal distribution and biases initialized to 0 and an arbitrary  $\epsilon \in [0,1)$ , with probability at least  $1 - \epsilon$ , gradient based techniques using log loss never achieve 0 training error nor they ever converge. Furthermore, this dataset can have a full rank covariance matrix and be linearly separable.

Proof. See Appendix B.1.

![](images/b8129096be381b02a30568513e2323078195ea3bd9375b4f3359a21dac565f13.jpg)

Even though the above construction requires control over the means of the normal distributions the weights are drawn from, as one can see in Figure 6, they do not have to be very large in practice. In particular, if one uses an initialization with  $\sigma$  as prescribed by LeCun et al. (1998) or Glorot & Bengio (2010) then the value of  $\mu = 0.24$  is sufficient to break the learning, even if we have 10,000 hidden units in each of 100 hidden layers. Using fixed  $\sigma = 0.01$  instead fails even with  $\mu = 0.07$ .

It is worth noting that even though this observations is about the existence of such dataset, our proof is actually done by construction, meaning that we show a way to build infinite amount of such datasets (as opposed to purely existential proofs). We would like to remark that it was well known that the initialization is important for the behaviour of learning (Glorot & Bengio, 2010; LeCun

![](images/b836d0cebd269d237e8da3fa4a6303ac2f89f53a1b70a5c3dcaa5be835a03704.jpg)  
Figure 6: On the left: exemplary dataset constructed in Proposition 4, color denotes label. Two middle ones: how big the mean of the normal distribution  $\mathcal{N}(\mu, \sigma^2)$  has to be in order to have at least  $99\%$  probability of the effect (very bad local minima) described in the Proposition 4, as a function of number of hidden units in  $2 - h - \ldots - h - 1$  classification network. By LeCun'98 initialization we mean taking weights from  $\mathcal{N}(\mu, \frac{1}{h})$  and by Xavier'10 from  $\mathcal{N}(\mu, \frac{2}{h_{\mathrm{in}} + h_{\mathrm{out}}})$ . In both cases the original papers used  $\mu = 0$ . Rightmost one: Proposition 5, probability of learning failing with increasing number of layers when the initialization is fully correct.

![](images/a5652edce30c0fd5ff61e209f6c16515618d6994035e656b02c0fd05bf079950.jpg)

![](images/a6c500d04e4190687db9cee94a81044af5e0e317bd6f34a04ecc542444a66b61.jpg)

![](images/fcb7100e2172ec387757340a37e0b49febc5eaa7cbdfa7149ff52cd2b51aca65.jpg)

et al., 1998; Sutskever et al., 2013; Pascanu et al., 2013). Here we are merely exploiting these ideas in order to better understand the error surface of the model.

If we do not care about the lack of convergence, and we are simply interested in learning failure, we can prove an even stronger proposition, which works for every single dataset:

Proposition 5. For every dataset, every feed forward rectifier network built for it, and every distribution used to initialize both weights and biases such that  $\mathbb{E}[w] = 0$ ,  $\mathbb{E}[b] = 0$ ,  $\mathbf{Var}[w] > 0$ ,  $\mathbf{Var}[b] \geq 0$ , the probability that the gradient based training of any loss function will lead to a trivial model (predicting the same label for all datapoints) goes to 1 as the number of hidden layers goes to infinity.

Proof. See Appendix B.2.

![](images/119fa1e73c4b98b96ca5c74e2020639ce2f36c656f94c632abf209f666783f77.jpg)

We can extend the previous proposition to show that for any regression dataset a rectifier model has at least one local minimum with a large basin of attraction (over the parameter space). Again, we rely on the blind spots of the rectified models. We show that there exists such blind spot that corresponds to a region in parameter space of same dimensionality (codimension 0). The construction relies on the fact that the dataset is finite. As such, it is bounded, and one can compute conditions for the weights of any given layer of the model such that for any datapoint all the units of that layer are deactivated. Furthermore, we show that one can obtain a better solution than the one reached from such a state. The formalization of this result is as follows.

We consider a  $k$ -layer deep regression model using  $m$  ReLU activation functions  $\mathrm{ReLU}(x) = \max(0, x)$ . Our dataset is a collection  $(\mathbf{x}_i, y_i) \in \mathbb{R}^d \times \mathbb{R}, i = 1, \dots, N$ . We denote  $\mathbf{h}_n(\mathbf{x}_i) = \mathrm{ReLU}(\mathbf{W}_n \mathbf{h}_{n-1}(\mathbf{x}_i) + \mathbf{b}_n)$  where the ReLU functions are applied component-wise to the vector  $\mathbf{W}_n \mathbf{h}_{n-1}(\mathbf{x}_i)$  and  $\mathbf{h}_0(\mathbf{x}_i) = \mathbf{x}_i$ . We also denote the final output of the model by  $\mathcal{M}(\mathbf{x}_i) = \mathbf{W}_k \mathbf{h}_{k-1} + \mathbf{b}_k$ . Solving the regression problem means finding

$$
\underset {(\mathbf {W} _ {n}) _ {n = 1} ^ {k}, (b _ {n}) _ {n = 1} ^ {k}} {\arg \min } \mathcal {L} \left(\left(\mathbf {W} _ {n}\right) _ {n = 1} ^ {k}, \left(\mathbf {b} _ {n}\right) _ {n = 1} ^ {k}\right) = \sum_ {i = 1} ^ {N} \left[ \mathcal {M} \left(\mathbf {x} _ {i}\right) - y _ {i} \right] ^ {2}. \tag {2}
$$

Let us state two simple yet in our opinion useful Lemmata.

Lemma 1 (Constant input). If  $\mathbf{x}_1 = \ldots = \mathbf{x}_N$ , then the solution to regression (2) has a constant output  $\mathcal{M} \equiv \frac{y_1 + \ldots + y_N}{N}$  (the mean of the values in data).

Proof. Obvious from the definitions and the fact, that  $\frac{y_1 + \ldots + y_N}{N} = \arg \min_{c}\sum_{i = 1}^{N}(c - y_i)^2$ .

![](images/934b707f81d4f5e97a2398ddbe62d1fd4b1a7647d24f59b2bbd09734de4e2b2b.jpg)

Lemma 2. If there holds  $\mathbf{W}_1\mathbf{x}_i < -\mathbf{b}_1$  for all  $i$ -s, then the model  $\mathcal{M}$  has a constant output. Moreover, applying local optimization does not change the values of  $\mathbf{W}_1$ ,  $\mathbf{b}_1$ .

Proof. Straightforward from the definitions.

![](images/d5293bfb4825dcba1bf5de360d227e9bd23762bb5914c4e1558fe5b729496a10.jpg)

Combining those two lemmata yields:

Corollary 1. If for any  $1 \leq j \leq k$  there holds  $\mathbf{W}_n\mathbf{h}_{n-1} < -b_n$  for all  $i$ -s then, after the training, the model  $\mathcal{M}$  will output  $\frac{y_1 + \ldots + y_N}{N}$ .

We will denote  $M(\{a_1, \ldots, a_L\}) = \frac{a_1 + \ldots + a_L}{L}$  the mean of the numbers  $a_1, \ldots, a_L$ .

Definition 1. We say that the dataset  $(\mathbf{x}_i, y_i)$  is **decent** if there exists  $r$  such that  $M(\{y_p : \mathbf{x}_p = \mathbf{x}_r\} \neq M(\{y_p : p = 1, \dots, N\})$ .

Theorem 1. Let  $\pmb{\theta} = ((\mathbf{W}_n)_{n=1}^k, (\mathbf{b}_n)_{n=1}^k)$  be any point in the parameter space satisfying  $\mathbf{W}_n \mathbf{h}_n(\mathbf{x}_i) < -\mathbf{b}_n$  (coordinate-wise) for all  $i$ -s. Then

i)  $\pmb{\theta}$  is a local minimum of the error surface,  
ii) if the first layer contains at least 3 neurons and if the dataset  $(\mathbf{x}_i, y_i)$  is decent, then  $\pmb{\theta}$  is not a global minimum.

Proof. See Appendix B.3.

![](images/e4deb60b6e7f41d67ab10decb7f499d162e3a63e871360e32a6358d0efc9967e.jpg)

# 4 DISCUSSION

Previous results (Dauphin et al., 2013; Saxe et al., 2014; Choromanska et al., 2015) provide insightful description of the error surface of deep models divorced from the dataset or initialization, as an expectation over all these aspects of the problem. While such analysis is very valuable not only for building up the intuition but also for the development of the tools for studying neural networks, it only provides one facade of the problem. In this work we move from the generic to the specific. We show that for finite sized models/finite sized datasets one does not have a globally good behaviour of learning regardless of the model size (and even of the ratio of model size to the dataset size).

The overwhelming amount of empirical evidence points towards learning being well behaved in practice. We argue that the way to reconcile these observations is to show that the well-behaved learning dynamics are local and conditioned on the data structure, initialization and perhaps on other architectural choices. One can imagine a continuum ranging from the very specific, where every detail of the setup is important to attain good learning dynamics, to the generic, where learning is globally well behaved regardless of dataset or initialization. We believe that an important step forward in the theoretical study of the neural networks can be made by identifying where exactly this class of models falls on that continuum. In particular, what are the most generic sets of constraints that need to be respected in order to attain the good behaviour. Our results focus on constructing counterexamples which result in a bad learning dynamics. While this does not lead directly to sufficient conditions for well-behaved systems, we hope that by carving out the space of possible conditions we are moving forward towards that goal.

Similar to Lin & Tegmark (2016) we put forward a hypothesis that the learning is only well behaved conditioned on the structure of the data. We point out, that for the purpose of learning, this structure can not be divorced from the particular initialization of the model. We postulate that learning becomes difficult if the data is structured such that there exist regions with a high density of datapoints (that belong to different classes) and the initialization results in models that assign these points to very few linear regions. While constraining the density per region alone might not be sufficient, it can provide a good starting point to understand learning for rectifier models. Another interesting question arising in that regard is what are the consequences on overfitting for enforcing a relatively low density of points per linear regions? Understanding of the structure of the error surface is an extremely challenging problem. We believe that as such, in agreement with a scientific tradition, it should be approached by gradually building up a related knowledge base, both by trying to obtain positive results (possibly under weakened assumptions, as it was done so far) and by studying the obstacles and limitations arising in concrete examples.

# ACKNOWLEDGMENTS

We would want to thank Neil Rabinowitz for insightful discussions.

# REFERENCES

Baldi, P. and Hornik, K. Neural networks and principal component analysis: Learning from examples without local minima. *Neural Networks*, 2(1):53-58, 1989.  
Bray, Alan J. and Dean, David S. Statistics of critical points of gaussian fields on large-dimensional spaces. Physics Review Letter, 98:150201, Apr 2007.  
Choromanska, Anna, Henaff, Mikael, Mathieu, Michael, Arous, Gerard Ben, and LeCun, Yann. The loss surfaces of multilayer networks. In AISTATS, 2015.  
Dauphin, Yann, Pascanu, Razvan, Gulcehre, Caglar, Cho, Kyunhyun, Ganguli, Surya, and Bengio, Yoshua. Identifying and attacking the saddle point problem in high dimensional non-convex optimization. NIPS, 2013.  
Fyodorov, Yan V. and Williams, Ian. Replica symmetry breaking condition exposed by random matrix calculation of landscape complexity. Journal of Statistical Physics, 129(5-6):1081-1116, 2007.  
Glorot, Xavier and Bengio, Yoshua. Understanding the difficulty of training deep feedforward neural networks. In JMLR W&CP: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2010), volume 9, pp. 249-256, May 2010.  
Goodfellow, Ian J., Vinyals, Oriol, and Saxe, Andrew M. Qualitatively characterizing neural network optimization problems. Int'l Conference on Learning Representations, ICLR, 2016.  
Kawaguchi, Kenji. Deep learning without poor local minima. CoRR, abs/1605.07110, 2016.  
Kingma, Diederik P. and Ba, Jimmy. Adam: A method for stochastic optimization. CoRR, abs/1412.6980, 2014. URL http://arxiv.org/abs/1412.6980.  
LeCun, Yann, Bottou, Léon, Orr, Genevieve B., and Müller, Klaus-Robert. Efficient backprop. In Neural Networks: Tricks of the Trade. 1998.  
LeCun, Yann, Bengio, Yoshua, and Hinton, Geoffrey. Deep learning. Nature, 521(7553):436-444, 5 2015. ISSN 0028-0836. doi: 10.1038/nature14539.  
Lin, Henry W. and Tegmark, Max. Why does deep and cheap learning work so well?, 2016. URL http://arxiv.org/abs/1608.08225.  
Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A, Veness, Joel, Bellemare, Marc G, Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K, Ostrovski, Georg, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529-533, 2015.  
Mnih, Volodymyr, Badia, Adria Puigdomenech, Mirza, Mehdi, Graves, Alex, Lillicrap, Timothy P, Harley, Tim, Silver, David, and Kavukcuoglu, Koray. Asynchronous methods for deep reinforcement learning. arXiv preprint arXiv:1602.01783, 2016.  
Parisi, Giorgio. Mean field theory of spin glasses: statistics and dynamics. Technical Report Arxiv 0706.0094, 2007.  
Pascanu, Razvan, Mikolov, Tomas, and Bengio, Yoshua. On the difficulty of training recurrent neural networks. In ICML'2013, 2013.  
Safran, Itay and Shamir, Ohad. On the quality of the initial basin in overspecified neural networks. CoRR, abs/1511.04210, 2015.  
Sagun, Levent, Guney, Ugur, Arous, Gerard Ben, and LeCun, Yann. Explorations on high dimensional landscapes. CoRR, abs/1412.6615, 2014.  
Saxe, Andrew, McClelland, James, and Ganguli, Surya. Learning hierarchical category structure in deep neural networks. Proceedings of the 35th annual meeting of the Cognitive Science Society, pp. 1271-1276, 2013.

Saxe, Andrew, McClelland, James, and Ganguli, Surya. Exact solutions to the nonlinear dynamics of learning in deep linear neural network. In International Conference on Learning Representations, 2014.  
Schmidhuber, J. Deep learning in neural networks: An overview. Neural Networks, 61:85-117, 2015. doi: 10.1016/j.neunet.2014.09.003. Published online 2014; based on TR arXiv:1404.7828 [cs.NE].  
Shamir, Ohad. Distribution-specific hardness of learning neural networks. CoRR, abs/1609.01037, 2016.  
Soudry, Daniel and Carmon, Yair. No bad local minima: Data independent training error guarantees for multilayer neural networks. CoRR, abs/1605.08361, 2016.  
Sutskever, Ilya, Martens, James, Dahl, George E., and Hinton, Geoffrey E. On the importance of initialization and momentum in deep learning. In Dasgupta, Sanjoy and Mcallester, David (eds.), Proceedings of the 30th International Conference on Machine Learning (ICML-13), volume 28, pp. 1139-1147. JMLR Workshop and Conference Proceedings, May 2013. URL http://jmlr.org/proceedings/papers/v28/sutskever13.pdf.  
Wigner, Eugene P. On the distribution of the roots of certain symmetric matrices. The Annals of Mathematics, 67(2):325-327, 1958.  
Zhang, Chiyuan, Bengio, Samy, Hardt, Moritz, Recht, Benjamin, and Vynalis, Oriol. Understanding deep learning requires rethinking generalization. In Submitted to Int'l Conference on Learning Representations, ICLR, 2017.
