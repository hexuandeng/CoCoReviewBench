# CGANS WITH PROJECTION DISCRIMINATOR

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose a novel, projection based way to incorporate the conditional information into the discriminator of GANs that respects the role of the conditional information in the underlining probabilistic model. This approach is in contrast with most frameworks of conditional GANs used in application today, which use the conditional information by concatenating the (embedded) conditional vector to the feature vectors. With this modification, we were able to significantly improve the quality of the class conditional image generation on ILSVRC2012 (ImageNet) dataset from the current state-of-the-art result, and we achieved this with a single pair of a discriminator and a generator. We were also able to extend the application to super-resolution and succeeded in producing highly discriminative super-resolution images. This new structure also enabled high quality category transformation based on parametric functional transformation of conditional batch normalization layers in the generator.

# 1 INTRODUCTION

Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) are a framework to construct a generative model that can mimic the target distribution, and in recent years it has given birth to arrays of state-of-the-art algorithms of generative models on image domain (Radford et al., 2016; Salimans et al., 2016; Ledig et al., 2016; Zhang et al., 2017; Reed et al., 2016). The most distinctive feature of GANs is the discriminator  $D(\pmb{x})$  that evaluates the divergence between the current generative distribution  $p_{G}(\pmb{x})$  and the target distribution  $q(\pmb{x})$  (Goodfellow et al., 2014; Nowozin et al., 2016; Arjovsky et al., 2017). The algorithm of GANs trains the generator model by iteratively training the discriminator and generator in turn, with the discriminator acting as an increasingly meticulous critic of the current generator.

Conditional GANs (cGANs) are a type of GANs that use conditional information (Mirza & Osindero, 2014) for the discriminator and generator, and they have been drawing attention as a promising tool for class conditional image generation (Odena et al., 2017), the generation of the images from text (Reed et al., 2016; Zhang et al., 2017), and image to image translation (Zhu et al., 2017). Unlike in standard GANs, the discriminator of cGANs discriminates between the generator distribution and the target distribution on the set of the pairs of generated samples  $x$  and its intended conditional variable  $y$ . To the authors' knowledge, most frameworks of discriminators in cGANs at the time of writing feeds the pair the conditional information  $y$  into the discriminator by naively concatenating (embedded)  $y$  to the input or to the feature vector at some middle layer (Mirza & Osindero, 2014; Reed et al., 2016; Zhang et al., 2017; Perarnau et al., 2016; Saito et al., 2017; Dumoulin et al., 2017; Sricharan et al., 2017). We would like to however, take into account the structure of the assumed conditional probabilistic models underlined by the structure of the discriminator, which is a function that measures the information theoretic distance between the generative distribution and the target distribution.

By construction, any assumption about the form of the distribution would act as a regularization on the choice of the discriminator. In this paper, we propose a specific form of the discriminator, a form motivated by a probabilistic model in which the distribution of the conditional variable  $y$  given  $x$  is discrete or uni-modal continuous distributions. This model assumption is in fact common in many real world applications, including class-conditional image generation and super-resolution.

As we will explain in the next section, adhering to this assumption will give rise to a structure of the discriminator that requires us to take an inner product between the embedded condition vector  $\mathbf{y}$  and

![](images/795e4fc7af787d09e90e16fa8e9dd6c7fd5c97821514bfabee3ef55d0cfa737c.jpg)  
Figure 1: Discriminator models when  $y$  is a categorical variable

![](images/222cb5b5a64c325cc2a83ffbb990a24ad6c4b91105ec9bc289519f115038da90.jpg)

![](images/aace9381b40c3c8463b3a6500f3f273daf55be255691e5824e8d507a39a66210.jpg)

![](images/b9c3d8767f59c3ab52db43686c8a48194d67a5e18e914f2cb681bb9665ad687e.jpg)

![](images/f2903f1d707d74ce0adb0d2a5d62865a804cdaf51f16f25e6283d6c5f92a376b.jpg)

![](images/8c0205db22cbbafdab8a23ca2cf21e839e242fa0e68a6499265c97f01caafc40.jpg)

![](images/f0094c8d52f94ef508a489f9fba4777cfe2c7b06a0d17aef5f2175fce9455259.jpg)  
(a) Images generated with the projection model. (left) Tibetan terrier and (right) mushroom.  
(b) (left) Consecutive category morphing with fixed  $z$ . geyser  $\rightarrow$  Tibetan terrier  $\rightarrow$  mushroom  $\rightarrow$  robin. (right) category morphing from Tibetan terrier to mushroom with different value of fixed  $z$  
Figure 2: The generator trained with the projection model can generate diverse set of images. For more results, in the experiment section and the appendix section.

![](images/4a1fe81bf424dba62727f9c56aee9180de0331497bcd32bc953eeff7fe827e40.jpg)

the feature vector (Figure 1d). With this modification, we were able to significantly improve the quality of the class conditional image generation on 1000-class ILSVRC2012 dataset (Russakovsky et al., 2015) with a single pair of a discriminator and generator (see the generated examples in Figure 2). Also, when we applied our model of cGANs to a super-resolution task, we were able to produce high quality super-resolution images that are more discriminative in terms of the accuracy of the label classifier than the cGANs based on concatenation, as well as the bilinear and the bicubic method.

# 2 THE ARCHITECTURE OF THE CGAN DISCRIMINATOR WITH A PROBABILISTIC MODEL ASSUMPTIONS

Let us denote the input vector by  $\pmb{x}$  and the conditional information by  $\pmb{y}^1$ . We also denote the cGAN discriminator by  $D(\pmb{x},\pmb{y};\theta) \coloneqq \mathcal{A}(f(\pmb{x},\pmb{y};\theta))$ , where  $f$  is a function of  $\pmb{x}$  and  $\pmb{y}$ ,  $\theta$  is the parameters of  $f$ , and  $\mathcal{A}$  is an activation function of the users' choice. The standard adversarial loss for the discriminator is:

$$
\mathcal {L} (D) = - E _ {q (\boldsymbol {y})} \left[ E _ {q (\boldsymbol {x} | \boldsymbol {y})} \left[ \log \left(D (\boldsymbol {x}, \boldsymbol {y})\right) \right] \right] - E _ {p (\boldsymbol {y})} \left[ E _ {p (\boldsymbol {x} | \boldsymbol {y})} \left[ \log \left(1 - D (\boldsymbol {x}, \boldsymbol {y})\right) \right] \right], \tag {1}
$$

where  $p(\pmb{x}|\pmb{y})$  and  $p(\pmb{y})$  are given the generator models, and  $\mathcal{A}$  of  $D$  is the sigmoid function. By construction, the nature of the 'critic'  $D$  significantly affects the performance of  $G$ . A conventional way of feeding  $y$  to  $D$  until now has been to concatenate the vector  $y$  to the feature vector  $x$ , either at the input layer (Mirza & Osindero, 2014; Saito et al., 2017), or at some hidden layer (Reed et al., 2016; Zhang et al., 2017; Perarnau et al., 2016; Dumoulin et al., 2017; Sricharan et al., 2017) (see Figure 1a and Figure 1b). We would like to propose an alternative to this approach by observing the

form of the optimal solution for the loss function (1), which is given by (Goodfellow et al., 2014):

$$
f ^ {*} (\boldsymbol {x}, \boldsymbol {y}) = r (\boldsymbol {x}, \boldsymbol {y}) := \log \frac {q (\boldsymbol {x} | \boldsymbol {y}) q (\boldsymbol {y})}{p (\boldsymbol {x} | \boldsymbol {y}) p (\boldsymbol {y})} = \log \frac {q (\boldsymbol {y} | \boldsymbol {x})}{p (\boldsymbol {y} | \boldsymbol {x})} + \log \frac {q (\boldsymbol {x})}{p (\boldsymbol {x})} = r (\boldsymbol {y} | \boldsymbol {x}) + r (\boldsymbol {x}) \tag {2}
$$

and we can model  $r(\pmb{y}|\pmb{x})$  and  $r(\pmb{x})$  by some parametric functions  $f_{1}$  and  $f_{2}$  respectively. If we make a standing assumption that  $p(\pmb{y}|\pmb{x})$  and  $q(\pmb{y}|\pmb{x})$  are simple distributions like those that are Gaussian or discrete log linear on the feature space, then, as we will show, the parametrization of the following form becomes natural:

$$
f (\boldsymbol {x}, \boldsymbol {y}; \theta) := f _ {1} (\boldsymbol {x}, \boldsymbol {y}; \theta) + f _ {2} (\boldsymbol {x}; \theta) = \boldsymbol {y} ^ {\mathrm {T}} V \phi (\boldsymbol {x}; \theta_ {\Phi}) + \psi \left(\phi \left(\boldsymbol {x}; \theta_ {\Phi}\right); \theta_ {\Psi}\right), \tag {3}
$$

where  $V$  is the embedding matrix of  $\pmb{y}$ ,  $\phi(\cdot, \theta_{\Phi})$  is a vector output function of  $\pmb{x}$ , and  $\psi(\cdot, \theta_{\Psi})$  is a scalar function of the same  $\phi(\pmb{x}; \theta_{\Phi})$  that appears in  $f_1$  (see Figure 1d). The learned parameters  $\theta = \{V, \theta_{\Phi}, \theta_{\Psi}\}$  are to be trained to optimize the adversarial loss. From this point on, we will refer to this model of the discriminator as projection for short. In the next section, we would like to elaborate on how we can arrive at this form.

# 3 MOTIVATION BEHIND THE projection DISCRIMINATOR

In this section, we will begin from specific, often recurring, models and show that, with certain regularity assumption, we can write the optimal solution of the discriminator objective function in the form of (3). Let us first consider the case of categorical variable. Assume that  $y$  is a categorical variable taking a value in  $\{1,\dots ,C\}$ , which is often common for a class conditional image generation task. The most popular model for  $p(y|\boldsymbol {x})$  is the following log linear model:

$$
\log p (y = c | \boldsymbol {x}) := \boldsymbol {v} _ {c} ^ {p \mathrm {T}} \phi (\boldsymbol {x}) - \log \left(\sum_ {j = 1} ^ {C} \exp \left(\boldsymbol {v} _ {j} ^ {p \mathrm {T}} \phi (\boldsymbol {x})\right)\right) = \boldsymbol {v} _ {c} ^ {p \mathrm {T}} \phi (\boldsymbol {x}) - \log Z (\phi (\boldsymbol {x})), \tag {4}
$$

where  $Z(\phi(\boldsymbol{x})) \coloneqq \left( \sum_{j=1}^{C} \exp \left( \boldsymbol{v}_j^{p^{\mathrm{T}}} \phi(\boldsymbol{x}) \right) \right)$  is the partition function, and  $\phi : \boldsymbol{x} \mapsto \mathbb{R}^{d^L}$  is the input to the final layer of the network model. Here, let us further assume that the network model can be shared within  $p(y|\boldsymbol{x})$  and  $q(y|\boldsymbol{x})$ . This way, the log density ratio of  $q(y = c|\boldsymbol{x})$  and  $p(y = c|\boldsymbol{x})$  becomes

$$
r (y | \boldsymbol {x}) = \log \frac {q (y = c | \boldsymbol {x})}{p (y = c | \boldsymbol {x})} = \left(\boldsymbol {v} _ {c} ^ {q} - \boldsymbol {v} _ {c} ^ {p}\right) ^ {\mathrm {T}} \phi (\boldsymbol {x}) - \left(\log Z ^ {q} (\phi (\boldsymbol {x})) - \log Z ^ {p} (\phi (\boldsymbol {x}))\right). \tag {5}
$$

If we make the values of  $(\pmb{v}_c^q, \pmb{v}_c^p)$  implicit and put  $\pmb{v}_c \coloneqq (\pmb{v}_c^q - \pmb{v}_c^p)$ , we can write  $f_1(\pmb{x}, y = c) = \pmb{v}_c^{\mathrm{T}} \pmb{\phi}(\pmb{x})$ . Now, if we can put together the normalization constant  $-(\log Z^q (\phi (\pmb{x})) - \log Z^p (\phi (\pmb{x})))$  and  $r(\pmb{x})$  into one expression  $\psi (\phi (\pmb{x}))$ , we can rewrite the equation above as

$$
f (\boldsymbol {x}, \boldsymbol {y}) := \boldsymbol {y} ^ {\mathrm {T}} V \phi (\boldsymbol {x}) + \psi (\phi (\boldsymbol {x})). \tag {6}
$$

by using  $\mathbf{y}$  to denote a one-hot vector of the label  $y$  and using  $V$  to denote the matrix consisting of the row vectors  $\mathbf{v}_c$ . Most notably, this formulation introduces the label information via an inner product, as opposed to concatenation. The form (6) is indeed the form we proposed in (3).

We can also arrive at the form (3) for unimodal continuous distributions  $p(\pmb{y}|\pmb{x})$  as well. Let  $\pmb{y} \in \mathbb{R}^d$  be a  $d$ -dimensional continuous variable, and let us assume that conditional  $q(\pmb{y}|\pmb{x})$  and  $p(\pmb{y}|\pmb{x})$  are both given by Gaussian distributions, so that  $q(\pmb{y}|\pmb{x}) = \mathcal{N}(\pmb{y}|\pmb{\mu}_q(\pmb{x}),\pmb{\Lambda}_q^{-1})$  and  $p(\pmb{y}|\pmb{x}) = \mathcal{N}(\pmb{y}|\pmb{\mu}_p(\pmb{x}),\pmb{\Lambda}_p^{-1})$  where  $\pmb{\mu}_q(\pmb{x}) \coloneqq W^q\phi (\pmb{x})$  and  $\pmb{\mu}_p(\pmb{x}) \coloneqq W^p\phi (\pmb{x})$ . Then the log density ratio  $r(\pmb{y}|\pmb{x}) = \log (q(\pmb{y}|\pmb{x}) / p(\pmb{y}|\pmb{x}))$  is given by:

$$
r (\boldsymbol {y} | \boldsymbol {x}) = - \frac {1}{2} \boldsymbol {y} ^ {\mathrm {T}} \left(\boldsymbol {\Lambda} _ {q} - \boldsymbol {\Lambda} _ {p}\right) \boldsymbol {y} + \boldsymbol {y} ^ {\mathrm {T}} \left(\boldsymbol {\Lambda} _ {q} W ^ {q} - \boldsymbol {\Lambda} _ {p} W ^ {p}\right) \phi (\boldsymbol {x}) + \psi (\phi (\boldsymbol {x})), \tag {7}
$$

where  $\psi(\phi(\boldsymbol{x}))$  represents the terms independent of  $\boldsymbol{y}$ . Now, if we assume that  $\Lambda_q = \Lambda_p \coloneqq \Lambda$ , we can ignore the quadratic term. If we further express  $\Lambda_q W^q - \Lambda_p W^p$  in the form  $V$ , we can arrive at the form (3) again. Indeed, however, the way that this regularization affects the training of the generator  $G$  is a little unclear in its formulation. As we have repeatedly explained, our discriminator measures the divergence between the generator distribution  $p$  and the target distribution

$q$  on the assumption that  $p(\boldsymbol{y}|\boldsymbol{x})$  and  $q(\boldsymbol{y}|\boldsymbol{x})$  are relatively simple, and it is highly possible that we are gaining stability in the training process by imposing a regularity condition on the divergence measure. Meanwhile, however, the actual  $p(\boldsymbol{y}|\boldsymbol{x})$  can only be implicitly derived from  $p(\boldsymbol{x},\boldsymbol{y})$  in computation, and can possibly take numerous forms other than the ones we have considered here. We must admit that there is a room here for an important theoretical work to be done in order to assess the relationship between the choice of the function space for the discriminator and training process of the generator.

# 4 COMPARISON WITH OTHER METHODS

As described above, (3) is a form that is true for frequently occurring situations. In contrast, incorporation of the conditional information by concatenation is rather arbitrary and can possibly include into the pool of candidate functions some sets of functions for which it is difficult to find a logical basis. Indeed, if the situation calls for multimodal  $p(\boldsymbol{y}|\boldsymbol{x})$ , it might be smart not to use the model that we suggest here. Otherwise, however, we expect our model to perform better; in general, it is preferable to use a discriminator that respects the presumed form of the probabilistic model.

Still another way to incorporate the conditional information into the training procedure is to directly manipulate the loss function. The algorithm of AC-GANs (Odena et al., 2017) use a discriminator  $(D_{1})$  that shares a part of its structure with the classifier  $(D_{2})$ , and incorporates the label information into the objective function by augmenting the original discriminator objective with the likelihood score of the classifier on both the generated and training dataset (see Figure 1c). Plug and Play Generative models (PPG) (Nguyen et al., 2017) is another approach for the generative model that uses an auxiliary classifier function. It is a method that endeavors to make samples from  $p(\boldsymbol{x}|\boldsymbol{y})$  using an MCMC sampler based on the Langevin equation with drift terms consisting of the gradient of an autoencoder prior  $p(\boldsymbol{x})$  and a pretrained auxiliary classifier  $p(y|\boldsymbol{x})$ . With these methods, one can generate a high quality image. However, these ways of using auxiliary classifier may unwittingly encourage the generator to produce images that are particularly easy for the auxiliary classifier to classify, and deviate the final  $p(\boldsymbol{x}|\boldsymbol{y})$  from the true  $q(\boldsymbol{x}|\boldsymbol{y})$ . In fact, Odena et al. (2017) reports that this problem has a tendency to exacerbate with increasing number of labels. We were able to reproduce this phenomena in our experiments; when we implemented their algorithm on a dataset with 1000 class categories, the final trained model was able to generate only one image for most classes. Nguyen et al.'s PPG is also likely to suffer from the same problem because they are using an order of magnitude greater coefficient for the term corresponding to  $p(y|\boldsymbol{x})$  than for the other terms in the Langevin equation.

# 5 EXPERIMENTS

In order to evaluate the effectiveness of our newly proposed architecture for the discriminator, we conducted two sets of experiments: class conditional image generation and super-resolution on ILSVRC2012 (ImageNet) dataset (Russakovsky et al., 2015). For both tasks, we used the ResNet (He et al., 2016b) based discriminator and the generator used in Gulrajani et al. (2017), and applied spectral normalization (Miyato et al., 2017) to the all of the weights of the discriminator to regularize the Lipschitz constant. For the objective function, we used the following hinge version of the standard adversarial loss (1) (Lim & Ye, 2017; Tran et al., 2017)

$$
L (\hat {G}, D) = E _ {q (y)} \left[ \right. E _ {q (\boldsymbol {x} | y)} \left[ \max (0, 1 - D (\boldsymbol {x}, y) ] \right] + E _ {q (y)} \left[ E _ {p (\boldsymbol {z})} \left[ \max (0, 1 + D (\hat {G} (\boldsymbol {z}, y), y)) \right]\right],
$$

$$
L (G, \hat {D}) = - E _ {q (y)} \left[ E _ {p (\boldsymbol {z})} \left[ \hat {D} (G (\boldsymbol {z}, y), y)) \right] \right], \tag {8}
$$

where the last activation function  $\mathcal{A}$  of  $D$  is identity function.  $p(z)$  is standard Gaussian distribution and  $G(z,y)$  is the generator network. For all experiments, we used Adam optimizer (Kingma & Ba, 2015) with hyperparameters set to  $\alpha = 0.0002$ ,  $\beta_{1} = 0$ ,  $\beta_{2} = 0.9$ . We updated the discriminator five times per each update of the generator. We will use concat to designate the models (Figure 1b)<sup>2</sup>, and use projection to designate the proposed model (Figure 1d).

![](images/6fa25b4ba4d10c78c9afc5a509a22c7bcc9dd0aacfce4ed1ffd029577230b323.jpg)  
Figure 3: Learning curves of cGANs with concat and projection on ImageNet. Figure 4: Comparison of intra FID scores for projection concat, and AC-GANs on ImageNet. Each dot corresponds to a class.

![](images/8de2bcd8a6018700763921d523b9d0857a6b18a4e73b51e9ccd129cb3fffa6cd.jpg)  
(a) concat vs projection

![](images/24035717860d5670474dd95520e9eb17c7e343acfab1f8447cd4ee908c3fde3a.jpg)  
(b) AC-GANs vs projection

# 5.1 CLASS-CONDITIONAL IMAGE GENERATION

The ImageNet dataset used in the experiment of class conditional image generation consisted of 1,000 image classes of approximately 1,300 pictures each. Unlike for AC-GANs<sup>3</sup> we used a single pair of a ResNet-based generator and a discriminator. Also, we used conditional batch normalization (Dumoulin et al., 2017) for the generator. As for the architecture of the generator network used in the experiment, please see Figure 12 for more detail. Our proposed projection model discriminator is equipped with a 'projection layer' that takes inner product between the embedded one-hot vector  $\mathbf{y}$  and the intermediate output (Figure 12a). As for the structure of the the concat model discriminator to be compared against, we used the identical bulk architecture as the projection model discriminator, except that we removed the projection layer from the structure and concatenated the spatially replicated embedded conditional vector  $\mathbf{y}$  to the output of fourth ResBlock. We also experimented with AC-GANs as the current state of the art model. For AC-GANs, we placed the softmax layer classifier to the same structure shared by concat and projection. For each method, we updated the generator 450K times, and applied linear decay for the learning rate after 400K iterations so that the rate would be 0 at the end.

We used inception score (Salimans et al., 2016) for the evaluation of the visual appearance of the generated images. It is in general difficult to evaluate how 'good' the generative model is. Indeed, however, either subjective or objective, some definite measures of 'goodness' exist, and essential two of them are 'diversity' and the sheer visual quality of the images. One possible candidate for quantitative measure of diversity and visual appearance is FID (Heusel et al., 2017). We computed FID between the generated images and dataset images within each class, and designated the values as intra FIDs. More precisely, FID (Heusel et al., 2017) measures the 2-Wasserstein distance between the two distributions  $q_{y}$  and  $p_{y}$ , and is given by  $F(q_{y},p_{y}) = \| \pmb{\mu}_{q_{y}} - \pmb{\mu}_{p_{y}}\|_{2}^{2} + \mathrm{trace}\left(C_{q_{y}} + C_{p_{y}} - 2(C_{q_{y}}C_{p_{y}})^{1 / 2}\right)$ , where  $\{\pmb{\mu}_{q_y},C_{q_y}\} ,\{\pmb{\mu}_{p_y},C_{p_y}\}$  are respectively the mean and the covariance of the final feature vectors produced by the inception model (Szegedy et al., 2015) from the true samples and generated samples of class  $y$ . When the set of generated examples have collapsed modes, the trace of  $C_{p_y}$  becomes small and the trace term itself becomes large. In order to compute  $C_{q_y}$  we used all samples in the training data belonging to the class of concern, and used 5000 generated samples for the computation of  $C_{p_y}$ . We empirically observed in our experiments that intra FID is, to a certain extent, serving its purpose well in measuring the diversity and the visual quality.

To highlight the effectiveness of our inner-product based approach (projection) of introducing the conditional information into the model, we compared our method against the state of the art AC-GANs as well as the conventional incorporation of the conditional information via concatenation at hidden layer (concat). As we can see in the training curves Figure 3, projection outperforms inception score than concat throughout the training. Table 1 compares the intra class FIDs of the images generated by each method. The result shown here for the AC-GANs is that of the model at its prime in terms of the inception score, because the training collapsed at the end. We see that the images generated by projection have lower intra FID scores than both adversaries, indicating that the Wasserstein distance between the generative distribution by projection to the target distribution

Table 1: Inception score and intra FIDs on ImageNet.  

<table><tr><td>Method</td><td>Inception Score</td><td>Intra FID</td></tr><tr><td>AC-GANs</td><td>28.7</td><td>252.0</td></tr><tr><td>concat</td><td>21.9</td><td>142.1</td></tr><tr><td>projection</td><td>30.3</td><td>103.3</td></tr></table>

![](images/83f2c46f101ee99436a7cc8e06825f8e58923aa1fed06baa86ba41332da23e01.jpg)  
Figure 5: comparison of the images generated by (a) ACGANs and (b) projection.

![](images/7e8ad6d82a8c1e8d2cf5b59d67f816c649e53cb035a827bbcc36ee878d1976c8.jpg)  
Figure 6: Collapsed images on the concat model.

![](images/3aeb4c9cecb1fde5ca2f1bd1fce913b270efcd17b5129af4fdf21db6670187e9.jpg)

is smaller. Figure 10a and 10b shows the set of classes for which (a) projection yielded results with better intra FIDs than the concat and (b) the reverse. From the top, the figures are listed in descending order of the ratio between the intra FID score between the two methods. Note that when the concat outperforms projection it only wins by a slight margin, whereas the projection outperforms concat by large margin in the opposite case. A quick glance on the cases in which the concat outperforms the projection suggests that the FID is in fact measuring the visual quality, because both sets look similar to the human eyes in terms of appearance. Figure 5 shows an arbitrarily selected set of results yielded by AC-GANs from variety of  $\mathbf{z} \mathbf{s}$ . We can clearly observe the mode-collapse on this batch. This is indeed a tendency reported by the inventors themselves Odena et al. (2017). AC-GANs can generate easily recognizable (i.e classifiable) images, but at the cost of losing diversity and hence at the cost of constructing a generative distribution that is significantly different from the target distribution as a whole. We can also assess the low FID score of projection from different perspective. By construction, the trace term of intra FID measures the degree of diversity within the class. Thus, our result on the intra FID scores also indicates that our projection is doing better in reproducing the diversity of the original. The concat method also suffered from mode-collapse for some classes (see Figure 6). For the set of images generated by projection, we were not able to detect any notable mode-collapse.

Figure 7a shows the samples generated with the projection model for the classes on which the cGAN achieved lowest intra FID scores (that is the classes on which the generative distribution were particularly close to target conditional distribution), and Figure 7b the reverse. While most of the images listed in Figure 7a are of relatively high quality, we still observe some degree of mode-collapse. Note that the images in the classes with high FID are featuring complex objects like human; that is, one can expect the diversity within the class to be wide. However, we note that we did not use the most complicated neural network available for the experiments presented on this paper, because we prioritized the completion of the training within a reasonable time frame. It is very possible that, by increasing the complexity of the model, we will be able to further improve the visual quality of the images and the diversity of the distribution. In Appendix C, we list images of numerous classes generated by cGANs trained with our projection model.

Category Morphing With our new architecture, we were also able to successfully perform category morphism. When there are classes  $y_{1}$  and  $y_{2}$ , we can create an interpolated generator by simply mixing the parameters of conditional batch normalization layers of the conditional generator corresponding to these two classes. Figure 8 shows the output of the interpolated generator with the same  $z$ . Interestingly, the combination is also yielding meaningful images when  $y_{1}$  and  $y_{2}$  are significantly different.

Fine-tuning with the pretrained model on the ILSVRC2012 classification task. As we mentioned in Section 4, the authors of Plug and Play Generative model (PPG) (Nguyen et al., 2017) were able to improve the visual appearance of the model by augmenting the cost function with that of the

![](images/bf9d3906eaf99dccfd302ce3e9b81950532ae640556234db23ca8259eab5cf58.jpg)  
(b) generated images on the class with 'high' FID scores.

![](images/b674851a5bb4a3f398fe252fd572d8dd934d1a306d80f16cd5e31284cdf7c6c1.jpg)  
Figure 7:  $128 \times 128$  pixel images generated by the projection method for the classes with (a) bottom five FID scores and (b) top five FID scores. The string and the value above each panel are respectively the name of the corresponding class and the FID score. The second row in each panel corresponds to the original dataset.  
Figure 8: Category morphing. More results are in the appendix section.

label classifier. We also followed their footsteps and augmented the original generator loss with an additional auxiliary classifier loss. As warned earlier regarding this type of approach, however, this type of modification tends to only improve the visual performance of the images that are easy for the pretrained model to classify. In fact, as we can see in Appendix B, we were able to improve the visual appearance the images with the augmentation, but at the cost of diversity.

# 5.2 SUPER-RESOLUTION

We also evaluated the effectiveness of (3) in its application to the super-resolution task. Put formally, the super-resolution task is to infer the high resolution RGB image of dimension  $\pmb{x} \in \mathbb{R}^{R_H \times R_H \times 3}$  from the low resolution RGB image of dimension  $\pmb{y} \in \mathbb{R}^{R_L \times R_L \times 3}$ ;  $R_H > R_L$ . This task is very much the case that we presumed in our model construction, because  $p(\pmb{y}|\pmb{x})$  is most likely unimodal even if  $p(\pmb{x}|\pmb{y})$  is multimodal. For the super-resolution task, we used the following formulation for discriminator function:

$$
f (\boldsymbol {x}, \boldsymbol {y}; \theta) = \sum_ {i, j, k} \left(y _ {i j k} F _ {i j k} \left(\phi \left(\boldsymbol {x}; \theta_ {\Phi}\right)\right)\right) + \psi \left(\phi \left(\boldsymbol {x}; \theta_ {\Phi}\right); \theta_ {\Psi}\right), \tag {9}
$$

where  $F(\phi(\boldsymbol{x}; \theta_{\Phi})) = V * \phi(\boldsymbol{x}; \theta_{\Phi})$  where  $V$  is a convolutional kernel and * stands for convolution operator. Please see Figure 13 in the appendix section for the actual network architectures we used for this task. For this set of experiments, we constructed the concat model by removing the module in the projection model containing the inner product layer and the accompanying convolution layer altogether, and simply concatenated  $\boldsymbol{y}$  to the output of the ResBlock preceding the inner product

![](images/82a7c41b83cf93089ef1a34b64d8aeeaf843be84da4d654b87d8dfd7f94d2b0e.jpg)  
Figure 9:  $32 \times 32$  to  $128 \times 128$  super-resolution by different methods

Table 2: Inception accuracy and MS-SSIM on different super-resolution methods. We picked up dataset images from the validation set.  

<table><tr><td>Method</td><td>bilinear</td><td>bicubic</td><td>concat</td><td>projection</td><td>projection (10 MC)</td></tr><tr><td>Inception Acc.(%)</td><td>24</td><td>32.1</td><td>10.1</td><td>35.2</td><td>36.7</td></tr><tr><td>MS-SSIM</td><td>0.835</td><td>0.859</td><td>0.829</td><td>0.862</td><td>-</td></tr></table>

module in the original. As for the resolutions of the image datasets, we chose  $R_{H} = 128$  and  $R_{L} = 32$ , and created the low resolution images by applying bilinear downsampling on high resolution images. We updated the generators 100K times for all methods, and applied linear decay for the learning rate after 50K iterations so that the final learning rate was 0 at 100K-th iteration.

Figure 9 shows the result of our super-resolution. The bicubic super-resolution is very blurry, and concat result is suffering from excessively sharp and rough edges. On the other hand, the edges of the images generated by our projection method are much clearer and smoother, and the image itself is much more faithful to the original high resolution images. In order to qualitatively compare the performances of the models, we checked MS-SSIM (Wang et al., 2003) and the classification accuracy of the inception model on the generated images using the validation set of the ILSVRC2012 dataset. As we can see in Table 2, our projection model was able to achieve high inception accuracy and high MS-SSIM when compared to bicubic and concat. Note that the performance of superresolution with concat model even falls behind those of the bilinear and bicubic super-resolutions in terms of the inception accuracy. Also, we used projection model to generate multiple batches of images with different random values of  $z$  to be fed to the generator and computed the average of the logits of the inception model on these batches (MC samples). We then used the so-computed average logits to make prediction of the labels. With an ensemble over 10 seeds (10 MC in Table 2), we were able to improve the inception accuracy even further. This result indicates that our GANs are learning the super-resolution as an distribution, as opposed to deterministic function. Also, the success with the ensemble also suggests a room for a new way to improve the accuracy of classification task on low resolution images.

# 6 CONCLUSION

Any specification on the form of the discriminator imposes a regularity condition for the choice for the generator distribution and the target distribution. In this research, we proposed a model for the discriminator of cGANs that is motivated by a commonly occurring family of probabilistic models. This simple modification was able to significantly improve the performance of the trained generator on conditional image generation task and super-resolution task. The result presented in this paper is strongly suggestive of the importance of the choice of the form of the discriminator and the design of the distributional metric. We plan to extend this approach to other applications of cGANs, such as semantic segmentation tasks and image to image translation tasks.

![](images/87c715ce4e9e61f8b52df7a88441186662e50a479ec749a5934f40c5674a1879.jpg)  
Figure 10: Comparison of concat vs. projection. The value attached above each panel represents the achieved FID score.

# REFERENCES

Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In ICML, pp. 214-223, 2017.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. In ICLR, 2017.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pp. 2672-2680, 2014.  
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein GANs. arXiv preprint arXiv:1704.00028, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016a.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Identity mappings in deep residual networks. In European Conference on Computer Vision, pp. 630-645. Springer, 2016b.  
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Günter Klambauer, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a nash equilibrium. arXiv preprint arXiv:1706.08500, 2017.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Christian Ledig, Lucas Theis, Ferenc Huszár, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, et al. Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint arXiv:1609.04802, 2016.

Jae Hyun Lim and Jong Chul Ye. Geometric gan. arXiv preprint arXiv:1705.02894, 2017.  
Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.  
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. ICML Implicit Models Workshop, 2017.  
Anh Nguyen, Jeff Clune, Yoshua Bengio, Alexey Dosovitskiy, and Jason Yosinski. Plug & play generative networks: Conditional iterative generation of images in latent space. In CVPR, 2017.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-GAN: Training generative neural samplers using variational divergence minimization. In NIPS, pp. 271-279, 2016.  
Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier GANs. In ICML, pp. 2642-2651, 2017.  
Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, and Jose M Álvarez. Invertible conditional gans for image editing. In NIPS Workshop on Adversarial Training, 2016.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. In ICLR, 2016.  
Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. arXiv preprint arXiv:1605.05396, 2016.  
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211-252, 2015. doi: 10.1007/s11263-015-0816-y.  
Masaki Saito, Eiichi Matsumoto, and Shunta Saito. Temporal generative adversarial nets with singular value clipping. In ICCV, 2017.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In NIPS, pp. 2226-2234, 2016.  
Kumar Sricharan, Raja Bala, Matthew Shreve, Hui Ding, Kumar Saketh, and Jin Sun. Semi-supervised conditional GANs. arXiv preprint arXiv:1708.05789, 2017.  
Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In CVPR, pp. 1-9, 2015.  
Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for nonparametric object and scene recognition. IEEE transactions on pattern analysis and machine intelligence, 30 (11):1958-1970, 2008.  
Dustin Tran, Rajesh Ranganath, and David M Blei. Deep and hierarchical implicit models. arXiv preprint arXiv:1702.08896, 2017.  
Zhou Wang, Eero P Simoncelli, and Alan C Bovik. Multiscale structural similarity for image quality assessment. In Asilomar Conference on Signals, Systems and Computers, pp. 1398-1402, 2003.  
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaolei Huang, Xiaogang Wang, and Dimitris Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In ICCV, 2017.  
Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2017.
