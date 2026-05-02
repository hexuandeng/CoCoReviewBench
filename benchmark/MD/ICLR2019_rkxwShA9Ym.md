# LABEL SUPER-RESOLUTION NETWORKS

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a deep learning-based method for super-resolving coarse (low-resolution) labels assigned to groups of image pixels into pixel-level (high-resolution) labels, given the joint distribution between those low- and high-resolution labels. This method involves a novel loss function that minimizes the distance between a distribution determined by a set of model outputs and the corresponding distribution given by low-resolution labels over the same set of outputs. This setup does not require that the high-resolution classes match the low-resolution classes and can be used in high-resolution semantic segmentation tasks where high-resolution labeled data is not available. Furthermore, our proposed method is able to utilize both data with low-resolution labels and any available high-resolution labels, which we show improves performance compared to a network trained only with the same amount of high-resolution data. We test our proposed algorithm in a challenging land cover mapping task to super-resolve labels at a  $30\mathrm{m}$  resolution to a separate set of labels at a 1m resolution. We compare our algorithm with models that are trained on high-resolution data and show that 1) we can achieve similar performance using only low-resolution data; and 2) we can achieve better performance when we incorporate a small amount of high-resolution data in our training. We also test our approach on a medical imaging problem, resolving low-resolution probability maps into high-resolution segmentation of lymphocytes with accuracy equal to that of fully supervised models.

# 1 INTRODUCTION

Semantic image segmentation is the task of labeling each pixel in an input image  $X = \{x_{ij}\}$  as belonging to one of  $L$  fine-scale application classes,  $Y = \{y_{ij}\}$ ,  $y \in \{1, \dots, L\}$ . In weakly supervised segmentation, instances in the training set only contain partial observations of the target ground truth labels, e.g., summary of class labels instead of pixel-level labels. We aim to solve a variant of this problem where coarse-scale, low-resolution accessory classes,  $Z = \{z_k\}$ ;  $z \in \{1, \dots, N\}$ , are defined for sets of pixels in the input images, where we are given the joint distribution  $P(Y,Z)$  between the accessory class labels and the application labels. Specifically, a training image  $X$  is divided into  $K$  sets  $B_k$ , each with an accessory class label  $z_k$ , and our models are trained to produce the high-resolution application labels  $y_{ij}$ . For example, in Figure 1, a high-resolution aerial image is shown alongside the low-resolution ground truth land cover map (defined over accessory classes) and the target high-resolution version (defined over application classes). We aim to derive the high-resolution land cover map based on the aerial image and low-resolution ground truth.

Compared to other weakly supervised image segmentation techniques, the formulation of the problem we aim to solve is more general: it applies both to existing weakly supervised image segmentation problems, as well as to other problems with different characteristics of weak labels. The more general formulation is necessary for tasks such as land cover mapping from aerial imagery and lymphocyte segmentation from pathology imagery. In these applications, coarse labels do not necessarily match the fine-scale labels, as shown in Figure 1. The distinction between the fine-scale application and coarse-scale accessory classes is necessary for situations in which the ground-truth information that is known about an image does not match with the application classes that we aim to label the image with, but instead suggests a distribution over the application labels. State-of-the-art methods for weakly supervised semantic segmentation exploit the structure of weak labels in ways that are not applicable in our examples: we cannot create bounding boxes around land cover object instances (Dai et al. (2015); Papandreou et al. (2015)) – we consider data that is generally given

![](images/596860d6e4edad54c4e1aa6f96ab7bd42a6f86d7e1e64df72c54529874ded27d.jpg)  
Figure 1: An Illustration of land cover data and label super-resolution. Our method takes an input image  $(x)$  with low-resolution labels  $(z)$  and outputs a set of super-resolved label predictions  $(y)$ , utilizing the statistical descriptions between low-resolution and high-resolution labels (Appendix B) e.g., one low-resolution class designates areas of low-intensity development, with  $20\%$  to  $49\%$  of impervious surfaces (such as houses or roads).

at scales much larger than the objects being segmented and does not carry foreground-background morphology – nor use coarse approximations of ground-truth segmentation (Krähenbühl & Koltun (2011); Hong et al. (2015)). Other work attempts to match a class “density function” to weak labels (Lempitsky & Zisserman (2010)), but it mainly targets localization and enumeration of small foreground objects with known sizes. Existing Weak supervision approaches also often involve expensive steps in inference, such as CRFs or iterative evaluation, which are impractical on large datasets. At the same time, thorough analyses of training algorithms only exist for models that are not sufficiently expressive for the applications we consider (Yu et al. (2013)). While our formulation of the problem allows us to specifically address the previously mentioned land cover mapping and lymphocyte segmentation, it can also be applied to more traditional segmentation tasks such as foreground/background segmentation as we explore in Appendix. D.

Our proposed method is illustrated in Figure 2. Briefly, a standard segmentation network will output probabilistic estimates of the application labels. Our methodology summarizes these estimates over the sets  $B_{k}$ , which results in an estimated distribution of application labels for each set. These distributions can then be compared to the expected distribution from the accessory (low-resolution) labels using standard distribution distance metrics. This extension is fully differentiable and can thus be used to train image segmentation neural networks end-to-end from pairs of images and coarse labels.

Land cover mapping from aerial imagery is an important application in need of such methodology. Land cover maps are essential in many sustainability-related applications such as conservation planning, monitoring habitat loss, and informing land management. In Section 3.1 we describe land cover mapping in detail and show how our method creates high-resolution land cover maps solely from high-resolution imagery low-resolution labels, at an accuracy similar to that of models trained on high-resolution labels. We further show how to train models with a combination of low- and high-resolution labels that outperform the high-res models in transfer learning tasks. As low-resolution labels are much easier to collect, and indeed exist over a much wider geographic area in our land cover mapping application, the ability to combine low- and high-resolution labels is an important feature of our proposed methods.

In a second example (Section 3.2), we segment tumor infiltrating lymphocytes from high-resolution (gigapixel) pathology images. Understanding the spatial distribution of immune cells, such as lymphocytes in pathology images, is fundamental for immunology and the treatment of cancer (Finn (2008); Thorsson et al. (2018)). Here, coarse labels are probabilities of lymphocyte infiltration (having two or more lymphocytes) on  $100 \times 100$  pixel regions, given by an automatic classifier (Saltz et al. (2018)). Our super-resolution model trained on coarse labels performs the same as a lymphocyte classifier trained with high-resolution (cell-level) supervision (Hou et al. (2018)).

To summarize, as our first contribution, we propose a label super-resolution network which utilizes the distribution of high-resolution labels suggested by given low-resolution labels, based on visual

![](images/dd7f6eca63e0704d28076c2c2b8233227263d027a205a55ca57cef21c5847a93.jpg)  
Figure 2: Proposed statistical matching loss function for label super-resolution shown with example images from our land cover labeling application. The model's high-resolution predictions in each low-resolution block are summarized by a label counting layer and matched with the distributions dictated by the low-resolution labels.

cues in the input images, to derive high-resolution label predictions consistent to the input image. Our second contribution is that we evaluate our method extensively on the application of land cover segmentation and conclude that when there are not enough representative high-resolution training data, our method is much more robust than a model trained on high-resolution training data only, since our method utilizes more training data with weak labels. We show the generality of our method on the lymphocyte segmentation task and the task of segmenting foreground given object bounding boxes (in Appendix D).

# 2 CONVERTING A SEMANTIC SEGMENTATION NETWORK INTO A LABEL SUPER-RESOLUTION NETWORK

A semantic segmentation network takes pixels  $X = \{x_{ij}\}$  as input and produces a distribution over labels  $Y = \{y_{ij}\}$  as output. If  $\phi$  are learned network parameters, this distribution is factorized as:

$$
p (Y | X; \phi) = \prod_ {i, j} p \left(y _ {i j} | X; \phi\right), \tag {1}
$$

Each  $p(y_{ij}|X;\phi)$  is a distribution over the possible labels,  $y \in \{1,\dots ,L\}$ . Typically, a network would be trained on pairs of observed training images and label images,  $(X^{t},Y^{t})$ , to maximize:

$$
\hat {\phi} = \underset {\phi} {\arg \max } \log \prod_ {t} p \left(Y ^ {t} \mid X ^ {t}; \phi\right) = \underset {\phi} {\arg \max } \sum_ {t} \sum_ {i, j} \log p \left(y _ {i j} ^ {t} \mid X ^ {t}; \phi\right). \tag {2}
$$

In this paper, we assume that we do not have pixel-level supervision,  $Y^{t}$ , but only coarse accessory (low-resolution) labels  $z_{k}\in \{1,\ldots ,N\}$  given on sets (blocks) of input pixels,  $B_{k}$ . We also assume a statistical joint distribution over the number of pixels  $c_{\ell}$  of each application label  $\ell \in \{1,\dots ,L\}$  occurring in a block labeled with an accessory (low-resolution) label  $z$ ,  $p_{\mathrm{coarse}}(c_1,c_2,\dots ,c_L|z)$ . Our extension of semantic segmentation networks is described in the following three sections.

Using coarse labels as statistical descriptors. In computer vision applications, pixel-level labeled data is typically expensive to produce, as is the case of high resolution land cover mapping where high-resolution labels only exist for limited geographic regions. On the other hand, coarse low-resolution labels are often easy to acquire and are readily available for larger quantities of data.

Coarse labels can provide weak supervision by dividing blocks of pixels into categories that are statistically different from each other. To exploit this we must formally represent the distribution of high-resolution pixel counts in these blocks,  $p_{\mathrm{coarse}}(c|z)$ .

For example, in the case of land cover mapping with four types of high-resolution land cover classes<sup>1</sup>, the descriptions of labels from the National Land Cover Database (NLCD) – at 30 times lower resolution than available aerial imagery (Homer et al. (2015)) – suggest distributions over the high-resolution labels. For instance, the “Developed, Medium Intensity” class – see Table 3 in the appendix – is described as “Areas with a mixture of constructed materials and vegetation. Impervious surfaces account for  $50\%$  to  $79\%$  of the total cover”. While such a designation does not tell us the precise composition or arrangement of high-resolution labels within a particular “Developed, Medium Intensity” label, it does describe a distribution. One mathematical interpretation of this particular example is

$$
c _ {\text {i m p e r v}} \sim \operatorname {u n i f} (0. 5, 0. 8), \quad c _ {\text {f o r e s t}} + c _ {\text {f i e l d}} = 1 - c _ {\text {i m p e r v}}, \quad c _ {\text {w a t e r}} \approx 0.
$$

In practice these descriptions should be interpreted in a softer manner (e.g. with Gaussian distributions) that can account for variance in real-world instances of the coarse classes<sup>2</sup>.

Label counting. Assume  $p_{\mathrm{coarse}}(c|z)$ , a connection between the coarse and fine labels, has been represented. Suppose we have a model that outputs distributions over high-resolution labels,  $p(Y|X)$  given inputs  $X$ . We must summarize the model's output over the low-resolution blocks  $B_k$ . Namely, a label counting layer computes a statistical representation  $\theta_k$  of the label counts in each block  $B_k$ .

If we sampled the model's predictions  $y_{ij}$  at each pixel, the count of predicted labels of class  $\ell$  in block  $B_k$  would be

$$
c _ {\ell} = \frac {1}{\left| B _ {k} \right|} \sum_ {(i, j) \in B _ {k}} \delta \left(y _ {i j} = \ell\right). \tag {3}
$$

By averaging many random variables, these counts  $c_{\ell}$  will follow an approximately Gaussian distribution,

$$
p _ {\text {n e t}} \left(c _ {\ell , k} = c | X\right) = \mathcal {N} \left(c; \mu_ {\ell , k}, \sigma_ {\ell , k} ^ {2}\right),
$$

where

$$
\mu_ {\ell , k} = \frac {1}{| B _ {k} |} \sum_ {(i, j) \in B _ {j}} p \left(y _ {i j} = \ell \mid X, \phi\right), \quad \sigma_ {\ell , k} ^ {2} = \frac {1}{| B _ {k} | ^ {2}} \sum_ {(i, j) \in B _ {k}} p \left(y _ {i j} = \ell \mid X, \phi\right) \left(1 - p \left(y _ {i j} = \ell \mid X, \phi\right)\right). \tag {4}
$$

These two parameters for each label  $\ell$  constitute the output of each block's label counting layer  $\theta_{k} = \{\mu_{\ell ,k},\sigma_{\ell ,k}^{2}\}_{\ell = 1}^{L}$ . Note that treating each count  $c_{\ell}$  as an independent Gaussian variable (given the input  $X$ ) ignores the constraint  $\sum_{\ell}c_{\ell} = 1$ , and more exact choices exist for modeling joint distributions  $p_{net}(\{c_{\ell}\}|X)$ ; however, we do have  $\sum \mu_{\ell} = 1$  and thus  $\mathbb{E}[\sum_{\ell}c_{\ell}] = 1$ . In practice, this approximation works well.

Statistics matching loss. The coarse labels  $z$  provide statistical descriptions for each block  $p_{\mathrm{coarse}}(\{c_{\ell}\} |z)$ , while the label counting modules produce distributions over what the segmentation network sees in the block given the high-res input image  $X$ ,  $p_{\mathrm{net}}(\{c_{\ell}\} |X)$ . The statistics matching module computes the amount of mismatch between these two distributions,  $D(p_{\mathrm{net}},p_{\mathrm{coarse}})$ , which we then use as an optimization criterion for the core segmentation model. Namely, we set

$$
C _ {\ell} = \underset {c _ {\ell}} {\arg \max } \left[ \sum \log p _ {\text {n e t}} \left(c _ {\ell} | X\right) \log p _ {\text {c o a r s e}} \left(c _ {\ell} | z\right) \right]
$$

and seek to maximize  $D(p_{\mathrm{net}}, p_{\mathrm{coarse}}) = \sum_l \log p_{\mathrm{net}}(C_\ell | X)$ , that is, the likelihood of the distribution of labels  $C$  that represents the optimal compromise between what the segmentation network expects (given the image  $X$ ) and what the joint distribution dictates (given the coarse label  $z$ ). In particular, if the distributions  $p_{\mathrm{coarse}}(c_\ell | z)$  are also represented as products of Gaussians, i.e.,

$$
p _ {\text {c o a r s e}} \left(\left\{c _ {\ell} \right\} \mid z\right) = \prod_ {\ell} \mathcal {N} \left(c _ {\ell}; \eta_ {\ell , z}, \rho_ {\ell , z} ^ {2}\right), \tag {5}
$$

![](images/4d549c54fa8172a5b4b8c5d24e5fe0fea693a8aba8c80dbcf7a131e7cd753873.jpg)  
Figure 3: Our model is useful detecting land cover change over years, at the same geographical location, which cannot be achieved effectively by directly comparing satellite images. For a detailed description of how we detect land cover change, see Appendix E

then

$$
C _ {\ell} = \frac {\rho_ {\ell , z} ^ {2} \mu_ {\ell} + \sigma_ {\ell} ^ {2} \eta_ {\ell , z}}{\sigma_ {\ell} ^ {2} + \rho_ {\ell , z} ^ {2}}, \tag {6}
$$

$$
D \left(p _ {\text {n e t}}, p _ {\text {c o a r s e}}\right) = \log p _ {\text {n e t}} \left(C _ {\ell} | X\right) \sim \operatorname {c o n s t} - \frac {1}{2} \frac {\left(\mu_ {\ell} - \eta_ {\ell , z}\right) ^ {2}}{\sigma_ {\ell} ^ {2} + \rho_ {\ell , z} ^ {2}} - \frac {1}{2} \log 2 \pi \sigma_ {\ell} ^ {2}, \tag {7}
$$

a function that is differentiable in the output of the label counting layer  $\theta = \{\mu_{\ell},\sigma_{\ell}^{2}\}$ . In turn, these are differentiable functions of the input image  $X$ . Thus, the network can be trained to minimize the sum of the expressions (7) over all blocks  $k$  in the input image<sup>3</sup>.

# 3 APPLICATIONS AND EXPERIMENTS

# 3.1 LAND COVER SUPER-RESOLUTION

We use our proposed methods in the land cover classification task. Land cover mapping is typically a part automatic, part manual process through which governmental agencies and private companies segment aerial or satellite imagery into different land cover classes Demir et al. (2018); Kuo et al. (2018); Davydow et al. (2018); Tian et al. (2018). Land cover data is useful in many settings: government agencies - local, state and federal - use this data to inform programs ranging from land stewardship and environment protection to city planning and disaster response, however this data is difficult and expensive to acquire at the high-resolutions where it is most useful. The Chesapeake Conservancy, for example, spent 10 months and $1.3 million to generate the first large high-resolution (1m) land cover map for over 160,000 km² of land in the Chesapeake Bay area (Chesapeake Bay Conservancy (2016; 2017)). Deep learning models that can automate the creation of land cover maps have a large practical value. As an example application, we create a method for automated land cover change detection using our models, described in Appendix E (with results in Figure 3). Furthermore, we create an interactive web application that lets users query our best performing models and "paint" land cover maps throughout the US, described in Appendix F.

![](images/5e9e3c682582fdb7c42f4d8c956ddfc199d0f32262cc008e5206b48664efbb0a.jpg)

![](images/dc7e053d4b1bb79970eddb5978a88f5a7d051bd15ab86c95717e412070959fbb.jpg)

![](images/d42ae16884d3c95040f7e81ac6d8a9ef08c6601e0875dbdded0945407ef8c370.jpg)  
Figure 4: Land cover segmentation examples. The SR model, while never shown pixel-level data in training, finds sharper edges of buildings than the high-res model and even identifies some features along the shoreline that the high-res model misses. For more qualitative examples, see Appendix F.

![](images/bb8e1964d975b294f9a1a3d09fd1aa608b5b01cd3d843442184210cd60996578.jpg)

Datasets and training. To demonstrate the effectiveness of our label super-resolution method we have three goals: (1) show how models trained solely with low-resolution data and label super-resolution compare to segmentation models that have access to enough representative high-resolution training data; (2) show how models trained using label super-resolution are able to identify details in heterogeneous land cover settings (i.e. in urban areas) more effectively than baseline weakly-supervised models; and (3) show how models trained using a combination of low- and high-resolution data, using our method, are able to generalize more effectively than models which rely on low- or high-resolution labels alone.

We use three datasets: 4-channel high-resolution (1m) aerial imagery from the US Department of Agriculture, expensive high-resolution (1m) land cover data covering the Chesapeake Bay watershed in the north eastern United States (Chesapeake Bay Conservancy (2016; 2017)), and much more widely available low-resolution (30m) NLCD land cover data (see Fig. 1 for examples of the data, and Appendix B). We divide these datasets into four geographical regions: Maryland 2013 training region with high-resolution training labels, Maryland 2013 test region, Chesapeake 2013 test region, and Chesapeake 2014 test region. We make the distinction between years the data was collected as weather, time of day, time of year, and photography conditions greatly change the quality of the imagery from year to year – see Fig. 3.

With these datasets we train and test four groups of models: HR models which will only have access to high-resolution data in the Maryland 2013 training region, SR models, trained with our label-super resolution technique, that only have access to low-resolution labels from the region in which they are tested, baseline weakly-supervised models, described in the next section, which will also only have access to low-resolution labels from region in which they are tested, and HR + SR models which will have access to the high-resolution labels from Maryland 2013, and low-resolution labels from the region in which they are tested. Given this setup, our experiments will vary two factors:

- The dataset on which low-resolution data is used, and on which the model is tested. As low-resolution labeled data is commonly available, we can train models with high-resolution data from the region in which we have it, as well as with low-resolution data from the area that we want our model to generalize to but where high-resolution data isn't available. We simulate this scenario with high-res data from Maryland and low-res data from the entire Chesapeake, even though the high-res labels are available – and used for our accuracy evaluation – in the rest of Chesapeake as well. This addresses our first two goals.  
- The amount of high-resolution data seen in training. In practical settings it is often the case that a small amount of high-resolution labels exist, as is the case with the land cover data from the Chesapeake Conservancy. To test how well our models will perform under this relaxation, we vary the amount of high-resolution data available from the Maryland 2013 training set from none (i.e.

only trained with low-resolution data using our SR models) to all data in the training set. Here, if both low-res and high-res data are used, we jointly optimize the core model on high-res data (using pixelwise cross-entropy loss) and on the low-res data (using our super-res criterion), using a weighted linear combination of the two losses. This addresses our third goal.

We use a U-Net architecture as our core segmentation model, and derive the parameters of the joint distributions between accessory (low-resolution) and application labels  $(\eta_{\ell,z}, \rho_{\ell,z}^{2})$  used in super-res model training from (low-res, high-res) pairs from the Maryland 2013 training set as the true means and variances of the frequencies of high-res label  $\ell$  in blocks of low-res class  $z$ . See Appendices A and B for details on the model architecture/training and joint distribution parameters between low-resolution and high-resolution classes.

Baseline models. Our main high resolution baseline model is the U-Net core trained to minimize pixelwise cross-entropy loss using the respective high-resolution labels. U-Net was chosen after experimentation with other standard neural segmentation models: SegNet (Badrinarayanan et al. (2017)), ResNet, and full-resolution ResNet (Pohlen et al. (2017)), all of which achieved overall accuracies from 80 to  $83\%$ .

In addition to the high-resolution model, we consider three baseline approaches to weakly supervised segmentation, which we compare to our SR models:

- "Soft naive": naively assigning the NLCD mean frequencies  $\eta_{\ell, c}$  as target labels for every pixel in a low-res block and training the core using cross-entropy loss as above.  
- "Hard naïve": Doing the same, but using a one-hot vector corresponding to the most frequent label in a given NLCD class ( $\arg \max_{\ell} \eta_{\ell, z}$ ) as the target.  
- An EM approach as in Papandreou et al. (2015): (1) M-step: train the super-res model only; (2) E-step: perform inference of high-res labels on the training set, followed by superpixel denoising (average predictions in each block); finally, assign labels in each block according to this smoothed prediction; (3) Repeat the EM iteration. Note that we use superpixel denoising instead of dense-CRF proposed by Papandreou et al. (2015), due to large computational overhead on the land cover dataset of over 10 billion pixels.

We also attempted directly comparing output label frequencies  $(\mu_{\ell})$  to the NLCD class means  $\eta_{\ell, z}$  using  $L^2$  loss, as well as using an  $L^1$  criterion (Lempitsky & Zisserman (2010)). In each case, the model converged to one that either predicted near-uniform class distributions at every pixel or always predicted the same class, giving accuracies below  $30\%$ . Interestingly, this occurred even when training was initialized with a well-performing trained model. These results indicate that the log-variance term in our criterion (7) is essential. (In these experiments, we used the same sample reweighting as in our super-res training and did a search through learning rates within a factor of 1000 of our baseline model learning rate.) Other approaches are discussed in Appendix C.

Results. The results for the baseline weakly supervised models and our SR models are shown in the first half of Table 1. We separately report overall results and results in NLCD blocks labeled with "Developed" (urban) low-resolution classes, which are the main source of errors for all models. Second, the Jaccard score is a more telling measure of classification quality than overall accuracy, which is dominated by large, easy-to-classify homogeneous areas (e.g. forests) and gives little weight to accuracy on impervious surfaces. Thus the most important single metric is Jaccard score in developed classes (in italics in Table 1). In these areas, our SR-only model tends to outperform the baselines (see second goal below).

First goal: In the second half of the table, the HR only model serves as an upper bound for what is achievable by models that use only low-res data. Unsurprisingly, models trained only on low-res data only are less accurate than those trained on high-res on the same region (Maryland 2013). Using low-res data together with high-res data adds uncertainty in training and slightly worsens results in Maryland, where high-res training data was used. However, adding low-res data from the test area allows our model to adapt to new geographies, with performance in developed areas in the two Chesapeake sets comparable to that in the original (Maryland) set. Furthermore, the SR-only model, not given high-res guidance, often produces segmentation that better match the true color segments and fine features of the images - see Fig. 4, an example from Maryland 2013).

<table><tr><td></td><td colspan="4">Maryland 2013 test region</td><td colspan="4">Chesapeake 2013 test region</td><td colspan="4">Chesapeake 2014 test region</td></tr><tr><td></td><td>all acc%</td><td>iou%</td><td>developed acc%</td><td>iou%</td><td>all acc%</td><td>iou%</td><td>developed acc%</td><td>iou%</td><td>all acc%</td><td>iou%</td><td>developed acc%</td><td>iou%</td></tr><tr><td colspan="13">Models trained on test geographical regions, without using high-resolution labels</td></tr><tr><td>Hard naïve</td><td>83.5</td><td>70.1</td><td>58.2</td><td>38.5</td><td>87.7</td><td>68.0</td><td>63.4</td><td>40.2</td><td>88.1</td><td>63.6</td><td>68.2</td><td>46.4</td></tr><tr><td>Soft naïve</td><td>85.5</td><td>71.4</td><td>65.1</td><td>45.6</td><td>87.9</td><td>66.7</td><td>65.6</td><td>42.7</td><td>88.6</td><td>62.9</td><td>70.2</td><td>48.3</td></tr><tr><td>EM</td><td>73.9</td><td>40.0</td><td>59.9</td><td>32.2</td><td>82.3</td><td>42.1</td><td>60.0</td><td>32.0</td><td>81.7</td><td>40.4</td><td>61.9</td><td>33.0</td></tr><tr><td>SR</td><td>82.6</td><td>71.7</td><td>74.3</td><td>49.7</td><td>87.0</td><td>68.2</td><td>73.4</td><td>47.4</td><td>82.0</td><td>57.4</td><td>73.4</td><td>48.2</td></tr><tr><td colspan="13">Models using high-resolution labels in Maryland 2013 training set (more than 1010 labeled pixels)</td></tr><tr><td>HR only</td><td>91.1</td><td>82.4</td><td>80.7</td><td>64.9</td><td>87.9</td><td>71.8</td><td>71.8</td><td>54.4</td><td>67.2</td><td>40.4</td><td>71.7</td><td>48.7</td></tr><tr><td>HR + SR</td><td>90.8</td><td>81.9</td><td>79.9</td><td>63.4</td><td>89.0</td><td>73.3</td><td>78.2</td><td>55.5</td><td>82.4</td><td>56.7</td><td>77.2</td><td>57.5</td></tr></table>

Table 1: Accuracies and Jaccard, or average intersection over union (IOU), scores on several data sets and models. Note that we train models with high-resolution data from only the Maryland 2013 training region and low-resolution data from the region on which they are tested. We give both overall metrics and those on areas labeled with NLCD "Developed, {Open, Low, Medium, High} Intensity" classes.  

<table><tr><td colspan="2">Maryland 2013</td><td colspan="2">Chesapeake 2013</td><td colspan="2">Chesapeake 2014</td></tr><tr><td rowspan="6">Accuracy</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td><td>0.9</td></tr><tr><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>0.7</td><td>0.7</td><td>0.7</td><td>0.7</td><td>0.7</td></tr><tr><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td></tr><tr><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td rowspan="8">Average laccard</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td><td>0.8</td></tr><tr><td>0.7</td><td>0.7</td><td>0.7</td><td>0.7</td><td>0.7</td></tr><tr><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td></tr><tr><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>0.4</td><td>0.5</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>0.3</td><td>0.3</td><td>0.3</td><td>0.3</td><td>0.3</td></tr><tr><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Number of labeled pixels</td><td>Number of labeled pixels</td><td>Number of labeled pixels</td><td>Number of labeled pixels</td><td>Number of labeled pixels</td></tr></table>

Figure 5: The effect of adding high-resolution data in training of super-resolution models. We show the baseline (HR only and SR only) and the results of  $\mathrm{HR + SR}$  models with varying number of high-res label data seen in training, both overall and in developed classes (as in Table 1). All results presented are average of 5 experiments with different random samples of high-res data.

Second goal: Overall, our super-res model performs better than the weakly supervised baselines. The "naive" training criteria and the EM method perform especially poorly in developed classes. Indeed, while in classes such as "Open Water" or "Deciduous Forest", most pixels are labeled with the majority class label, in developed areas, the mean distributions are rather flat - "forest", "field", and "impervious" occur with nearly equal frequency in the "Developed, Low Intensity" class (see Table 4 in the appendix). Thus a model would prefer to make a highly uncertain prediction at every pixel in such a patch, rather than classify each pixel with confidence.

Third goal: Figure 5 shows how results of a model trained with our super-resolution technique improve when high-resolution labels are added in training. Performance gradually increases over the super-res baseline as more high-resolution data is seen. Models trained on both high-resolution and low-resolution data outperform HR-only models in all test sets even when a small number  $(10^{6}$  pixels  $= 1\mathrm{km}^2)$  of high-resolution pixels is seen in training. In the Chesapeake 2014 dataset, HR+SR continues to far outperform HR-only even when all high-res data is used, and the metrics of HR+SR in developed classes even exceed those of HR-only overall. This demonstrates that super

![](images/bd3473b5159c382cd9e9880089f2aa5c7044ceb6d5d51a3dfd4fcd078f7c1e2e.jpg)  
Figure 6: Our method is able to super-resolve the low-resolution probabilities of lymphocyte infiltration into pixel-level lymphocyte segmentation. Lymphocytes are dark, rounded small cells. Our method gives reasonable lymphocyte segmentation results (in green contours).

resolution models can readily be used to create fine-scale labels in new geographies with only a small amount of strong supervision.

The full  $\mathrm{HR + SR}$  accuracy of  $89\%$  on the Chesapeake 2013 dataset is in fact close to the estimated accuracy  $(90\%)$  of the "ground truth" labels over the entire Chesapeake region (Chesapeake Bay Conservancy (2017)) based on the same aerial imagery, which were themselves produced by a much more labor-intensive semiautomatic process (Chesapeake Bay Conservancy (2016)).

# 3.2 LYMPHOCYTE SEGMENTATION

We apply our method for lymphocyte segmentation in pathology images. Lymphocytes are a type of white blood cell that play an important role in human immune systems. Quantitative characterization of tumor infiltrating lymphocytes (TILs) is of rapidly increasing importance in precision medicine (Barnes et al. (2018); Finn (2008); Thorsson et al. (2018)). With the growth of cancer immunotherapy, these characterizations are likely to be of increasing clinical significance, as understanding each patient's immune response becomes more important. However, due to the heterogeneity of pathology images, the existing state-of-the-art approach only classifies relatively large tumor regions as lymphocyte-infiltrated or not. We show that our method is able to super-resolve the low-resolution probabilities of lymphocyte infiltration, given by the existing method (Saltz et al. (2018)), into pixel-level lymphocyte segmentation results. We illustrate this application in Figure 6.

Datasets and training. A typical resolution of pathology whole slide images is  $50\mathrm{k}\times 50\mathrm{k}$  pixels with 0.5 microns per pixel. An existing method (Saltz et al. (2018)) generated a probability heatmap for each of the 5000 studied whole slide images: every  $100\times 100$  pixel region was assigned a probability of being lymphocyte infiltrated. We use these probability heatmaps as low-resolution ground truth labels and super-resolve them into high-resolution (pixel-level) lymphocyte segmentation. To evaluate the segmentation performance, we use the lymphocyte classification dataset introduced in Hou et al. (2018). This dataset contains 1786 image patches. Each patch has a label indicating if the cell in the center of the image is a lymphocyte or not.

Baseline models. In addition to the Hard naïve and Soft naïve methods, we compare with the published models (Hou et al. (2018)) which are trained for lymphocyte classification in a supervised fashion. In particular:

- HR SVM: The authors first segment the object in the center of pathology patch with a level-set based method (Zhou et al. (2017)). Then they extract hand-crafted features such as the area, average color, roundness of the object (Zhou et al. (2017)). Finally they train an SVM (Chang & Lin (2011)) using these features.  
- HR: Hou et al. (2018) directly train a CNN to classify each object in the center of image patches. This can be viewed as a CNN trained using high-resolution labels, although only the label of the center pixel is given.

- HR semi-supervised: Hou et al. (2018) initialize a HR CNN using a trained sparse convolutional autoencoder. Then the authors train the CNN to classify each object in the center of image patches.

Because all baseline CNNs require supervised data, they are all evaluated using four-fold cross-validation on the aforementioned dataset of 1786 image patches.

Label super-resolution. To use the low-resolution probability map as labels, we quantize the probability values into 10 classes with ranges [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]. In each low-resolution class, we sampled 5 regions labeled with this class and visually assessed the average number of lymphocytes, based on which we set the expected ratio of lymphocyte pixels in a given region ranging from  $0\%$  to  $40\%$ . With this joint distribution between low-resolution and high-resolution labels, we train our super-resolution network on 150 slides with low-resolution labels randomly selected from the 5000 slides. To guide our algorithm to focus on lymphocyte and non-lymphocyte cells instead of tissue background, we assign labels to  $20\%$  of the pixels in each input patch as non-lymphocyte, based on color only - pathology images are typically stained with Hematoxylin and Eosin, which act differently on nuclei of cells and cytoplasm (background), resulting in different colors. In terms of the network architecture, we apply the same U-Net as in the land cover experiment.

Results. We present quantitative results, obtained using the lymphocyte classification dataset, in Table 2. A testing image patch is classified as lymphocyte/non-lymphocyte by our method if its center pixel is segmented as lymphocyte/non-lymphocyte respectively. Our method performs as well as the best-performing baseline method with cell-level supervision.

<table><tr><td></td><td>HR SVM</td><td>HR</td><td>HR semi-supervised</td><td>Hard naïve</td><td>Soft naïve</td><td>SR</td></tr><tr><td>AUC</td><td>0.7132</td><td>0.4936</td><td>0.7856</td><td>0.5000†</td><td>0.6254</td><td>0.7833</td></tr></table>

Table 2: Area Under receiver operating characteristic Curve (AUC) results of super-resolving low-resolution lymphocyte infiltration probability maps to individual lymphocyte segmentation, on the lymphocyte classification dataset from Hou et al. (2018). A testing image patch is classified as lymphocyte/non-lymphocyte by our method if its center pixel is segmented as lymphocyte/non-lymphocyte respectively. All HR baseline methods are directly evaluated on the classification dataset by four-fold cross-validation and reported by Hou et al. (2018). Our weakly supervised method performs effectively as well as the best-performing baseline method with cell-level supervision. †: Hard naïve achieves 0.50 AUC because there is no positive HR label, due to hard label assignment.

# 4 CONCLUSIONS

We proposed a label super-resolution network which is capable of deriving high-resolution labels, given low-resolution labels that do not necessarily match the targeting high-resolution labels in a one-to-one manner – we only assume that the joint distribution between the low-resolution and high-resolution classes is known. In particular, we train a network to predict high-resolution labels, minimizing the distance/divergence between two distributions: distribution of predicted high-resolution labels and expected distribution suggested by the low-resolution labels. We applied our method in two real-world applications where high res labels are very expensive to obtain compared to low res labels, and achieved similar or better results compared to the conventional fully supervised methods trained on high-resolution labels. We also show how combining low and high res labels leads to better generalization to out-of-sample test sets.

# REFERENCES

Vijay Badrinarayanan, Alex Kendall, and Roberto Cipolla. Segnet: A deep convolutional encoder-decoder architecture for image segmentation. IEEE Transactions on Pattern Analysis & Machine Intelligence, (12):2481-2495, 2017.  
Michael Barnes, Anindya Sarkar, Rachel Redman, Charles Bechert, and Chukka Srinivas. Development of a histology-based digital pathology image analysis algorithm for assessment of

tumor infiltrating lymphocytes in her2+ breast cancer. Cancer Research, 2018. URL http://cancerres.aacrjournals.org/content/78/4_Supplement/P5-03-08.  
Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST), 2(3):27, 2011.  
Chesapeake Bay Conservancy. High resolution lulc classification accuracy assessment methodology, 2016. URL https://www.chesapeakebay.net/channel_files/24793/lulcaccuracyassessmentDetailed_methodology.pdf. [Online].  
Chesapeake Bay Conservancy. Land cover data project, January 2017. URL https://chesapeakeconservancy.org/wp-content/uploads/2017/01/LandCover101Guide.pdf. [Online].  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3213-3223, 2016.  
Jifeng Dai, Kaiming He, and Jian Sun. Boxsup: Exploiting bounding boxes to supervise convolutional networks for semantic segmentation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 1635-1643, 2015.  
Alex Davydow, OU Neuromation, and Sergey Nikolenko. Land cover classification with superpixels and jaccard index post-optimization. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018.  
Ilke Demir, Krzysztof Koperski, David Lindenbaum, Guan Pang, Jing Huang, Saikat Basu, Forest Hughes, Devis Tuia, and Ramesh Raskar. Deep globe 2018: A challenge to parse the earth through satellite images. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018.  
Olivera J Finn. Cancer immunology. New England Journal of Medicine, 358(25):2704-2715, 2008.  
Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning lecture 6a overview of mini-batch gradient descent.  
Collin Homer, Jon Dewitz, Limin Yang, Suming Jin, Patrick Danielson, George Xian, John Coulston, Nathaniel Herold, James Wickham, and Kevin Megown. Completion of the 2011 national land cover database for the conterminous united states—representing a decade of land cover change information. Photogrammetric Engineering & Remote Sensing, 81(5):345-354, 2015.  
Seunghoon Hong, Hyeonwoo Noh, and Bohyung Han. Decoupled deep neural network for semi-supervised semantic segmentation. In Advances in neural information processing systems, pp. 1495-1503, 2015.  
Le Hou, Vu Nguyen, Ariel B Kanevsky, Dimitris Samaras, Tahsin M Kurc, Tianhao Zhao, Rajarsi R Gupta, Yi Gao, Wenjin Chen, David Foran, et al. Sparse autoencoder for unsupervised nucleus detection and representation in histopathology images. Pattern Recognition, 2018.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pp. 448-456, 2015.  
Philipp Krahenbuhl and Vladlen Koltun. Efficient inference in fully connected crfs with gaussian edge potentials. In Advances in neural information processing systems, pp. 109-117, 2011.  
Tzu-Sheng Kuo, Keng-Sen Tseng, Jia-Wei Yan, Yen-Cheng Liu, and Yu-Chiang Frank Wang. Deep aggregation net for land cover classification. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018.  
Victor Lempitsky and Andrew Zisserman. Learning to count objects in images. In Advances in neural information processing systems, pp. 1324-1332, 2010.

George Papandreou, Liang-Chieh Chen, Kevin P Murphy, and Alan L Yuille. Weakly-and semi-supervised learning of a deep convolutional network for semantic image segmentation. In Proceedings of the IEEE international conference on computer vision, pp. 1742-1750, 2015.  
Deepak Pathak, Philipp Krahenbuhl, and Trevor Darrell. Constrained convolutional neural networks for weakly supervised segmentation. In Proceedings of the IEEE international conference on computer vision, pp. 1796-1804, 2015.  
Tobias Pohlen, Alexander Hermans, Markus Mathias, and Bastian Leibe. Full-resolution residual networks for semantic segmentation in street scenes. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3309-3318. IEEE, 2017.  
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pp. 234-241. Springer, 2015.  
Joel Saltz, Rajarsi Gupta, Le Hou, Tahsin Kurc, Pankaj Singh, Vu Nguyen, Dimitris Samaras, Kenneth R Shroyer, Tianhao Zhao, Rebecca Batiste, et al. Spatial organization and molecular correlation of tumor-infiltrating lymphocytes using deep learning on pathology images. Cell reports, 23(1):181, 2018.  
Frank Seide and Amit Agarwal. Cntk: Microsoft's open-source deep-learning toolkit. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 2135-2135. ACM, 2016.  
Vésteinn Thorsson, David L Gibbs, Scott D Brown, Denise Wolf, Dante S Bortone, Tai-Hsien Ou Yang, Eduard Porta-Pardo, Galen F Gao, Christopher L Plaisier, James A Eddy, et al. The immune landscape of cancer. Immunity, 48(4):812-830, 2018.  
Chao Tian, Cong Li, and Jianping Shi. Dense fusion classmate network for land cover classification. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2018.  
Felix Yu, Dong Liu, Sanjiv Kumar, Jebara Tony, and Shih-Fu Chang.  $\propto$  SVM for Learning with Label Proportions. In International Conference on Machine Learning, pp. 504-512, 2013.  
Shanshan Zhang, Rodrigo Benenson, and Bernt Schiele. Citypersons: A diverse dataset for pedestrian detection. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), volume 1, pp. 3, 2017.  
Naiyun Zhou, Xiaxia Yu, Tianhao Zhao, Si Wen, Fusheng Wang, Wei Zhu, Tahsin Kurc, Allen Tannenbaum, Joel Saltz, and Yi Gao. Evaluation of nucleus segmentation in digital pathology images through large scale image synthesis. In Medical Imaging 2017: Digital Pathology, volume 10140, pp. 101400K. International Society for Optics and Photonics, 2017.
