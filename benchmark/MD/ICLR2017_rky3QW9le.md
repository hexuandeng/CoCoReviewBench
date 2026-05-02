# TRANSFORMATIONAL SPARSE CODING

Dimitrios C. Gklezakos & Rajesh P. N. Rao

Department of Computer Science

and Center for Sensorimotor Neural Engineering

University of Washington

Seattle, WA 98105, USA

{gklezd,rao}@cs.washington.edu

# ABSTRACT

A fundamental problem faced by object recognition systems is that objects and their features can appear in different locations, scales and orientations. Current deep learning methods attempt to achieve invariance to local translations via pooling, discarding the locations of features in the process. Other approaches explicitly learn transformed versions of the same feature, leading to representations that quickly explode in size. Instead of discarding the rich and useful information about feature transformations to achieve invariance, we argue that models should learn object features conjointly with their transformations to achieve equivariance. We propose a new model of unsupervised learning based on sparse coding that can learn object features jointly with their affine transformations directly from images. Results based on learning from natural images indicate that our approach matches the reconstruction quality of traditional sparse coding but with significantly fewer degrees of freedom while simultaneously learning transformations from data. These results open the door to scaling up unsupervised learning to allow deep feature+transformation learning in a manner consistent with the ventral+dorsal stream architecture of the primate visual cortex.

# 1 INTRODUCTION

A challenging problem in computer vision is the reliable recognition of objects under a wide range of transformations. Approaches such as deep learning that have achieved success in recent years usually require large amounts of labeled data, whereas the human brain has evolved to solve the problem using an almost unsupervised approach to learning object representations. During early development, the brain builds an internal representation of objects from unlabeled images that can be used in a wide range of tasks.

Much of the complexity in learning efficient and general-purpose representations comes from the fact that objects can appear in different poses, at different scales, locations, orientations and lighting conditions. Models have to account for these transformed versions of objects and their features. Current successful approaches to recognition use pooling to allow limited invariance to two-dimensional translations Ranzato et al. (2007). At the same time pooling discards information about the location of the detected features. This can be problematic because scaling to large numbers of objects requires modeling objects in terms of parts and their relative pose, requiring the pose information to be retained.

Previous unsupervised learning techniques such as sparse coding Olshausen & Field (1997) can learn features similar to the ones in the visual cortex but these models have to explicitly learn large numbers of transformed versions of the same feature and as such, quickly succumb to combinatorial explosion, preventing hierarchical learning. Other approaches focus on computing invariant object signatures Anselmi et al. (2013; 2016), but are completely oblivious to pose information.

Ideally, we want a model that allows object features and their relative transformations to be simultaneously learned, endowing itself with a combinatorial explanatory capacity by being able to apply learned object features with object-specific transformations across large numbers of objects. The goal of modeling transformations in images is two-fold: (a) to facilitate the learning of

pose-invariant sparse feature representations, and (b) to allow the use of pose information of object features in object representation and recognition.

We propose a new model of sparse coding called transformational sparse coding that exploits a tree structure to account for large affine transformations. We apply our model to natural images. We show that our model can extract pose information from the data while matching the reconstruction quality of traditional sparse coding with significantly fewer degrees of freedom. Our approach to unsupervised learning is consistent with the concept of "capsules" first introduced by Hinton and colleagues Hinton et al. (2011), and more generally, with the dorsal-ventral (features+transformations) architecture observed in the primate visual cortex.

# 2 TRANSFORMATIONAL SPARSE CODING

# 2.1 TRANSFORMATION MODEL

Sparse coding (Olshausen & Field (1997)) models each image  $I$  as a sparse combination of features:

$$
I \simeq F w \quad \text {s . t .} \mathrm {w i s s p a r s e}
$$

Sparsity is usually enforced by the appropriate penalty. Typical choices include:  $S_{1}(w) = \| w\|_{1}$  or the smoothed squared root version:

$$
S _ {\epsilon} (w) = \sum_ {1} ^ {k} \sqrt {w _ {k} ^ {2} + \epsilon}
$$

We can enhance sparse coding with affine transformations by transforming features before combining them. The vectorized input image  $I$  is then modeled as:

$$
I = \sum_ {k = 1} ^ {K} w _ {k} T (x _ {k}) F _ {k}
$$

where  $w_{k}, F_{k}$  denote the  $k$ -th weight specific to the image and the  $k$ -th feature respectively and  $T(x_{k})$  is a feature and image specific transformation.

In modeling image transformations we follow the approach of Rao & Ruderman (1999) and Rao & Miao (2007). We consider the 2D general affine transformations. These include rigid motions such as vertical and horizontal translations and rotations, as well as scaling and two types of hyperbolic deformations.

Any subset of these transformations forms a Lie group with the corresponding number of dimensions (6 for the full set). Any transformation in this group can be expressed as the matrix exponential of a weighted combination of matrices (the group generators) that describe the behaviour of infinitesimal transformations around the identity:

$$
T (x) = e ^ {\sum_ {j} x _ {j} G _ {j}}
$$

For images of  $M$  pixels,  $T(x)$  is a matrix of size  $M \times M$ . Note that the generator matrices and the features used are common across all images. The feature weights and transformation parameters can be inferred (and the features learned) by gradient descent on the regularized MSE objective:

$$
L (w, x, F) = \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| I _ {i} - \sum_ {k = 1} ^ {K} w _ {i k} T (x _ {i k}) F _ {k} \right\| _ {2} ^ {2} + \lambda_ {S} S _ {\epsilon} (w) + \lambda_ {F} \| F \| _ {2} ^ {2}
$$

Although this model ties sparse coding with transformations elegantly, learning large transformations with it is intractable. The error surface of the loss function is highly non-convex with many shallow local minima. Figures 1(a), 1(b), 1(c) show the surface of  $L$  as a function of horizontal and vertical translation, horizontal translation and rotation and vertical translation and rotation parameters. The model tends to settle for small transformations around the identity.

Although the gradients  $\frac{\partial L}{\partial F}, \frac{\partial L}{\partial w}$  can be computed analytically, this is not the case for  $\frac{\partial L}{\partial x}$ . When the generators  $G_{j}$  do not commute the gradient has to be derived numerically. This, combined with the fact that we need to maintain a set of transformation parameters for each data point makes a random restart approach infeasible.

![](images/1295777349ad74ae374f59f2144415260644cc6b6e9effcd812562dc2ae9a7b8.jpg)  
(a)

![](images/38b2a809f88fed2477942727adb75de51df4b9731f014d722244ae6fa4c7375e.jpg)  
(b)

![](images/89bff26c8b44ec6be84bed03779fa3e2af30f30605450be2f68b8099c8689f81.jpg)  
(c)

![](images/f58115488357ce2d822f9ab196572b8704320118a50a347eb5990af2a8e82bba.jpg)  
(d)

![](images/ab0a97fa04eeff374c262e8f4b4dfbd6e84544d8581937f600aa28255627f64d.jpg)  
(e)

![](images/00dea86468fd52e34ef9640526ea991417007868e74430b073771426bad41f46.jpg)  
(f)  
Figure 1: Normalized reconstruction error for individual vs. batch  $8 \times 8$  natural image patches. (a),(b),(c) show the surface of the reconstruction error for horizontal and vertical translations, horizontal translations and rotation, vertical translations and rotations for an individual data point and feature. (d),(e),(f) show the same, averaged over a batch of 2000 data points. The error is normalized between 0 and 1 for comparison. The global minimum in the range is marked in red. In the batch case, averaging makes the error surface smoother and learning easier.

# 2.2 TRANSFORMATION TREES

We introduce Transformational Sparse Coding Trees to circumvent this problem using hierarchies of transformed features. The main idea is to gradually marginalize over an increasing range of transformations. Each node in the tree represents a feature derived as a transformed version of its parent, with the root being the template of the feature. The leaves are equivalent to a set of sparse basis features and are combined to reconstruct the input as described above. A version of the model using a forest of trees of depth one, is given by:

$$
I \simeq \sum_ {v = 1} ^ {V} \sum_ {b \sim c h (v)} w _ {b} U _ {b}
$$

where  $U_{b} = T(x_{v\rightarrow b})F_{v}$  and  $ch(v)$  the children of root  $v$ . The feature  $U_{b}$  is a leaf, derived from the root feature  $F_{v}$  via the fixed (across all data-points) transformation  $T(x_{v\rightarrow b})$ . Deeper trees can be built accordingly. A small example of a tree learned from natural image patches is shown in Figure 2.

There are multiple advantages to such a hierarchical organization of sparse features. Some transformations are more common in data than others. Each path in the tree corresponds to a chain of transformations that is common across images. Such a path can be viewed as a "transformation feature" learned from the data. Each additional node in the tree "costs" a fixed set of new parameters equal in size to the dimensions of the underlying Lie group (six in our case). At the same time each additional node contributes a whole new feature to the sparse code. Averaging over many data points, smoothens the surface of the error function and makes larger transformations more accessible to optimization. Figures 1(d), 1(e), 1(f) show the error surface averaged over a batch of 2000 patches.

![](images/3b7e0dac303b46b26abf1c4172b2d3720d4eaae74878d76ecce5576af054ecc4.jpg)  
(a)  
Figure 2: Example of a tree learned from natural image patches. The leaves correspond to rigid transformations of the root.

For every leaf that is activated, the root template represents the identity of the feature and the transformation associated with the path to the root, the pose. In other words the tree is an equivariant representation of the feature over the parameter region defined by the set of paths to the leaves, very similar to the concept of a capsule introduced by Hinton et al. (2011). In fact, every increasing subtree corresponds to a capsule of increasing size.

# 2.3 REGULARIZATION

Since features in the same tree are no longer completely independent, it can be the case that a tree has a few heavily used features that prevent useful learning transitions at the root. Then, the model gets stuck in configurations where a large fraction of the features is unused. Ideally we would want a sparse solution that utilizes each feature with similar frequency.

To address that, we replace  $S_{\epsilon}$  with a combination of intra- and inter-tree sparsity penalties. The intra-tree penalty favors solutions that: (a) are sparse within a tree and (b) do not overuse any tree. We use a penalty that penalizes the  $\ell_1$  norm within each tree, together with the combined magnitude of the tree weights:

$$
S _ {i n} (w) = \sum_ {v = 1} ^ {V} \left(\sum_ {b \sim c h (v)} \sqrt {w _ {b} ^ {2} + \epsilon}\right) ^ {2}
$$

The inter-tree penalty is similar to that used in topographic sparse coding (Kavukcuoglu et al. (2009)), if we treat each tree as a group:

$$
S _ {o u t} (w) = \sum_ {v = 1} ^ {V} \sqrt {\sum_ {b \sim c h (v)} w _ {b} ^ {2} + \epsilon}
$$

Since one can always decrease the weights and increase the magnitudes of the features, reducing the loss, the problem is under-constrained. Following traditional sparse coding we also penalize the  $\ell_2$  norm of the leaves:

$$
P (x, F) = \sum_ {v = 1} ^ {V} \sum_ {b \sim c h (v)} \| T (x _ {v \rightarrow b}) F _ {v} \| _ {2} ^ {2}
$$

In our model, the magnitudes of the features are functions of the transformation parameters and the root features. Rigid motions leave the magnitude the same, whereas scaling clearly does not. To avoid extreme scaling of the features, we directly penalize the scaling and hyperbolic deformation parameters of the transformations.

# 2.4 LEARNING

The reconstruction mean squared-error (MSE) for a forest of trees of depth one is given by:

$$
L _ {M S E} (w, x, F) = \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| I _ {i} - \sum_ {v = 1} ^ {V} \sum_ {b \sim c h (v)} w _ {i b} T (x _ {v \rightarrow b}) F _ {v} \right\| _ {2} ^ {2}
$$

The loss function of the model is:

$$
L (w, x, F) = L _ {M S E} (w, x, F) + \lambda_ {i n} S _ {i n} (w) + \lambda_ {o u t} S _ {o u t} (w) + \gamma P (x, F) + \lambda_ {s} \| X _ {[ s, h _ {1}, h _ {2} ]} \| _ {2} ^ {2}
$$

where  $X_{[s,h_1,h_2]}$  is the vector of the collective scaling and hyperbolic deformation parameters.

Following the alternating minimization approach for sparse coding, we first perform inference using gradient descent on  $L(w,x,F)$  with respect to the weights. Then we adjust the leaves by optimizing  $x$  and finally adjust the roots by optimizing  $F$ .

# 3 EXPERIMENTS

# 3.1 LEARNING REPRESENTATIONS

We apply transformational sparse coding (TSC) with forests of trees of depth one to  $8 \times 8$  natural image patches. Our approach allows us to learn features resembling those of traditional sparse coding. Apart from reconstructing the input, the model also extracts transformation parameters from the data. Figure 3 shows a reconstruction example. Figure 4 shows the root features learned from  $8 \times 8$  natural image patches using a forest of size 8 with branching factor 8, equipped with rigid motions. The forest has a total of 64 features. Figure 4(a) shows the features corresponding to the roots. Figure 4(b) shows the corresponding leaves. Each row contains features derived from the same root. More examples of learned features are shown in Figures 6 and 7 in the Appendix.

![](images/d69828a0e4f072ac5f0ab8208ca5e4da4a24149cb4911af3017032e111f53703.jpg)  
(a)  
Figure 3: Reconstruction example. The root features are transformed and combined with different weights to reconstruct (bottom right) the  $8 \times 8$  natural image patch in the top right corner.

To determine whether the model learns interesting, large transformations we control for the magnitudes of the group parameters and their distance from their initialization values. Figure 5 shows these quantities for the  $8 \times 8$  forest layout.

# 3.2 COMPARISON WITH SPARSE CODING

Even though derivative features have to be exactly constructed for inference, the degrees of freedom of our model are significantly lower than that of traditional sparse coding. Specifically:

$$
d f _ {T S C} = (\# \text {o f r o o t s}) \times (\# \text {o f p i e x e l s + b r a n c h i n g f a c t o r} \times \text {g r o u p d i m e n s i o n})
$$

![](images/9e73fd3f93ce49b09b4c0a53218459f39de38bde26899ad8af38d8af7ff9be9f.jpg)  
(a)

![](images/c7c26339d0cf57577a7d03158aaae9237332edfcc1f0ffb597b324f160579d0f.jpg)  
(b)

![](images/9cefa81d1be296320decdf2822e72335d3d83f0be97b2f5e4efb462ff0fd3629.jpg)  
Figure 4: Learned features for 8 trees with a branching factor of 8. (a) Features corresponding to the roots. (b) Features/Leaves: Each row corresponds to leaves/transformations of the same root.  
(a)

![](images/d893bff53a6013f9bb9bcaa44c3312bcc758a1ec1aa7727213ad6439d7d36009.jpg)  
(b)  
Figure 5:  $\ell_1$  distance from initialized values and  $\ell_1$  magnitude of the transformation parameters per iteration for the  $8\times 8$  forest layout.

whereas:

$$
d f _ {S C} = \# \text {o f f e a t u r e s} \times \# \text {o f p i x e l s}
$$

Note that the group dimension is equal to 3 for rigid motions and 6 for general 2D affine transformations.

We compare the reconstruction error of forests of different sizes equipped with all six general affine transformations with that of traditional sparse coding with the same number of features and mean sparsity. For the comparison we used the algorithm introduced in Lee et al. (2007). The results are shown in Table 1. The performance of our approach is close to that of sparse coding, despite the fact that the degrees of freedom are significantly lower. At the same time our model can extract pose information in the form of group parameters.

Table 1: Comparison of transformational sparse coding (TSC) with sparse coding (SC). We compare the error (MSE) and the degrees of freedom  $(df)$  for batches of 2000 data points.  

<table><tr><td colspan="4">TSC</td><td colspan="5">SC</td></tr><tr><td>Forest Layout</td><td>MSE</td><td>Sparsity</td><td>dfTSC</td><td>MSE</td><td>Sparsity</td><td>dfSC</td><td># of features</td><td>dfSC/dfTSC</td></tr><tr><td>8 × 8</td><td>1.60</td><td>5.4</td><td>896</td><td>1.42</td><td>5.1</td><td>4096</td><td>64</td><td>4.57</td></tr><tr><td>8 × 8</td><td>0.45</td><td>15.8</td><td>896</td><td>0.28</td><td>15.5</td><td>4096</td><td>64</td><td>4.57</td></tr><tr><td>8 × 32</td><td>0.55</td><td>12.7</td><td>2048</td><td>0.28</td><td>13.2</td><td>16384</td><td>256</td><td>8</td></tr><tr><td>16 × 16</td><td>0.36</td><td>17.5</td><td>2560</td><td>0.11</td><td>17.2</td><td>16384</td><td>256</td><td>6.4</td></tr></table>

# 4 RELATED WORK

Sohl-Dickstein et al. (2010) present a model for fitting Lie groups to video data. Their approach only works for estimating a global transformation between consecutive video frames. They only support transformations of a single kind (ie only rotations). Different such single-parameter transformations have to be chained together to produce the global one. The corresponding transformation parameters also have to be inferred and stored in memory and cannot be directly converted to parameters of a single transformation. Kokiopoulou & Frossard (2009) present an approach to optimally estimating transformations between pairs of images. They support rigid motions and isotropic scaling. Our model supports all six transformations and learns object parts and their individual transformations. In contrast with those approaches, our model learns object parts jointly with their transformations within the same image. Our model utilizes the full, six-dimensional, general affine Lie group and captures the pose of each object part in the form of a single set of six transformation parameters.

The work closest to ours is that of Hinton et al. (2011) on capsules. A capsule learns to recognize its template (feature) over a wide range of poses. The pose is computed by a neural network (encoder). The decoder, resembling a computer graphics engine combines the capsule templates in different poses to reconstruct the image. Each transformational sparse coding tree can be thought of as a capsule. The template corresponds to the root. The tree learns to "recognize" transformed versions of that template. Our work arrives at the concept of a capsule from a sparse coding perspective. A major difference is that our approach allows us to reuse each feature multiple times in different, transformed versions for each data point.

# 5 CONCLUSION

In this paper, we proposed a sparse coding based model that learns object features jointly with their transformations, from data. Naively extending sparse coding for data-point specific transformations makes inference intractable. We solve this problem by using a tree structure that represents common transformations in data. We show that our approach can learn interesting features from natural image patches with performance comparable to that of traditional sparse coding.

Learning deeper trees to allow for larger transformations and stacking trees to account for more complex features and larger receptive fields are subjects of ongoing research.

# REFERENCES

Fabio Anselmi, Joel Z. Leibo, Lorenzo Rosasco, Jim Mutch, Andrea Tacchetti, and Tomaso A. Poggio. Unsupervised learning of invariant representations in hierarchical architectures. CoRR, abs/1311.4158, 2013. URL http://arxiv.org/abs/1311.4158.  
Fabio Anselmi, Joel Z. Leibo, Lorenzo Rosasco, Jim Mutch, Andrea Tacchetti, and Tomaso Poggio. Unsupervised learning of invariant representations. Theor. Comput. Sci., 633(C):112-121, June 2016. ISSN 0304-3975. doi: 10.1016/j.tcs.2015.06.048. URL http://dx.doi.org/10.1016/j.tcs.2015.06.048.  
Geoffrey E. Hinton, Alex Krizhevsky, and Sida D. Wang. Transforming auto-encoders. In Proceedings of the 21th International Conference on Artificial Neural Networks - Volume Part I,

ICANN'11, pp. 44-51, Berlin, Heidelberg, 2011. Springer-Verlag. ISBN 978-3-642-21734-0. URL http://dl.acm.org/citation.cfm?id=2029556.2029562.  
K. Kavukcuoglu, M. A. Ranzato, R. Fergus, and Yann Le-Cun. Learning invariant features through topographic filter maps. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 1605-1612. IEEE, June 2009. ISBN 978-1-4244-3992-8. doi: 10.1109/cvpr.2009.5206545. URL http://dx.doi.org/10.1109/cvpr.2009.5206545.  
E. Kokiopoulou and P. Frossard. Minimum distance between pattern transformation manifolds: Algorithm and applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(7):1225-1238, July 2009. ISSN 0162-8828. doi: 10.1109/TPAMI.2008.156.  
Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng. Efficient sparse coding algorithms. In B. Schölkopf, J. C. Platt, and T. Hoffman (eds.), Advances in Neural Information Processing Systems 19, pp. 801-808. MIT Press, 2007. URL http://papers.nips.cc/paper/2979-efficient-sparse-coding-algorithms.pdf.  
Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision research, 37(23):3311-3325, 1997.  
Marc'Aurelio Ranzato, Fu-Jie Huang, Y-Lan Boureau, and Yann LeCun. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In Proc. Computer Vision and Pattern Recognition Conference (CVPR'07). IEEE Press, 2007.  
Rajesh P. N. Rao and Xu Miao. Learning the Lie Groups of Visual Invariance. 2007.  
Rajesh P. N. Rao and Daniel L. Ruderman. Learning lie groups for invariant visual perception. In In NIPS, 1999.  
Jascha Sohl-Dickstein, Jimmy C. Wang, and Bruno A. Olshausen. An unsupervised algorithm for learning lie group transformations. CoRR, abs/1001.1027, 2010. URL http://arxiv.org/abs/1001.1027.

APPENDIX A: LEARNED FEATURES

![](images/1ee19b1b5e4d9c740b3352cde1224a3fcfbeeed2ef806007dfb71846cab61c73.jpg)  
(a)

![](images/e38a720c4c1716034816829959099720a9963d5e23af9e02a3246e6ff49a1a38.jpg)  
(b)  
Figure 6: Learned features for 16 trees with branching factor 16. Each row corresponds to leaves/transformations of the same root.

![](images/9f1e72e7f914a7792a4ad138c1cafba73aea232258fae0c449aa0451733457eb.jpg)  
(a)

![](images/62f3328776dc2668e4b9fdaa56dd30d68ca72af34ac710a6cce1b67021404b35.jpg)  
(b)  
Figure 7: Learned features for 8 trees with branching factor 32. Each row corresponds to leaves/transformations of the same root.