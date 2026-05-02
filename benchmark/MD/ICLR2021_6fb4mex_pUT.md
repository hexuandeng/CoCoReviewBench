# AN ALGORITHM FOR OUT-OF-DISTRIBUTION ATTACK TO NEURAL NETWORK ENCODER

Anonymous authors

Paper under double-blind review

# ABSTRACT

Deep neural network (DNN), especially convolutional neural network, has achieved superior performance on image classification tasks. However, such performance is only guaranteed if the input to a trained model is similar to the training samples, i.e., the input follows the probability distribution of the training set. Out-Of-Distribution (OOD) samples do not follow the distribution of training set, and therefore the predicted class labels on OOD samples become meaningless. Classification-based methods have been proposed for OOD detection; however, in this study we show that this type of method has no theoretical guarantee and is practically breakable because of dimensionality reduction in the DNN model. We also show that Glow likelihood-based OOD detection is ineffective as well. Our analysis is demonstrated on five open datasets, including a COVID-19 CT dataset. At last, we present a simple theoretical solution with guaranteed performance for OOD detection.

# 1 INTRODUCTION

Deep neural network (DNN), especially convolutional neural network (CNN), has become the dominant technique for image classification. Under the i.i.d. (independent and identically distributed) assumption, a high-performance DNN model can correctly-classify an input sample as long as the sample is "generated" from the distribution of training data. If an input sample is not from this distribution, which is called Out-Of-Distribution (OOD), then the predicted class label from the model is meaningless. It would be great if the model has the ability to distinguish OOD samples from in-distribution samples. OOD detection is needed especially when applying DNN models in life-critical applications, e.g., vision-based self-driving or image-based medical diagnosis.

It was shown by Nguyen et al. in 2015 (Nguyen et al., 2015) that DNN classifiers can be easily fooled by OOD data, and evolutionary algorithm was used to generate OOD samples such that DNN classifiers had high output confidence on these samples. Since then, many methods are proposed for OOD detection using classifiers or encoders (Hendrycks & Gimpel, 2016)(Hendrycks et al., 2018)(Liang et al., 2017) (Lee et al., 2018) (Lee et al., 2017) (Alemi et al., 2018). For instance, Hendrycks et al. (Hendrycks & Gimpel, 2016) shows that a classifier's prediction probability of OOD examples tends to be lower than the prediction probability of in-distribution samples, and therefore the maximum predicted class probability from the softmax layer was used for OOD detection. Regardless of the details of these methods, every method needs a classifier or an encoder, which takes an image  $x$  as input and compresses it into a vector  $z$  in the latent space; after some further transform,  $z$  is converted to an OOD detection score  $\tau$ . This computing process can be expressed as:  $z = f(x)$  and  $\tau = d(z)$ . To perform OOD detection, a detection threshold needs to be specified, and then  $x$  is OOD if  $\tau$  is smaller/larger than the threshold. For OOD detection method evaluation (Hendrycks & Gimpel, 2016), usually, an OOD detector is trained on a dataset (e.g. Fashion-MNIST as in-distribution) and then it is tested on another dataset (e.g. MNIST as OOD).

As will be shown in this study, the above mentioned classification-based OOD detection is theoretically almost ineffective and practically breakable. As an example (more details in section 3), we used the Resnet-18 model (He et al., 2016) pre-trained on ImageNet dataset. Let  $x_{in}$  denote a  $224 \times 224 \times 3$  image (in-distribution sample) in ImageNet and  $x_{out}$  denote an OOD sample which could be any kind of images (even random noises) not belonging to any category in ImageNet. Let  $z$  denote the 512-dimensional feature vector in Resnet-18, which is the input to the last fully-connected linear

layer before softmax operation. Thus, we have  $z_{in} = f(x_{in})$  and  $z_{out} = f(x_{out})$ . As shown in Fig. 1,  $x_{in}$  is the image of Santa Claus, and  $x_{out}$  could be a chest x-ray image or a random-noise image, and "surprisingly",  $z_{out} \cong z_{in}$  which renders OOD detection score to be useless:  $d(z_{out}) \cong d(z_{in})$ .

In section 2, we will introduce an algorithm to generate OOD samples such that  $z_{out} \cong z_{in}$ . In section 3, we will show the evaluation results on publicly available datasets, including ImageNet subset, GTSRB, OCT, and COVID-19 CT. Since some generative models (e.g. Glow (Kingma &

![](images/d23094a504fdf00924cd6f21da337e104b0db323385b7546951b3e51eda5c4ec.jpg)

![](images/05a8cc76a65fa6b41f9a11754981e69312bdca283af8ea729bb144b3da26685d.jpg)  
(a)

![](images/7dc2db5d0660ebfa641cd355010065048a95f0fcf9ae68f37c272ac468523c43.jpg)

![](images/dcbcffd86adbbaafc474af4723f028f732c69548e812289714d45243de08ff29.jpg)  
Figure 1: The 1st column shows the image of Santa Claus  $x_{in}$  and the scatter plot of  $z_{in}$  using blue dots. The 2nd column shows a chest x-ray image  $x_{out}$  and the scatter plot of  $z_{out}$  (red circles) and  $z_{in}$  (blue). The 3rd column shows a random image  $x_{out}$ , and the scatter plot of  $z_{out}$  (red) and  $z_{in}$  (blue).  
(b)

![](images/9c560aa5b79bb23641d6310dbe61bd880ca85a228284b5bc791841f6e4babb51.jpg)

![](images/765e4087241bdbe17972ff96aa8000f1a9e4bc3090a525808ed24f55f4d561d3.jpg)  
(c)

Dhariwal, 2018)) can approximate the distribution of training samples (i.e.  $p(x_{in})$ ), likelihood-based generative models were utilized for OOD detection (Nalisnick et al., 2018). It has been shown that likelihoods derived from generative models may not distinguish between OOD and training samples (Nalisnick et al., 2018) (Ren et al., 2019) (Choi et al., 2018), and the fix to the problem could be using likelihood ratio instead of raw likelihood score (Serrà et al., 2019). Although not the main focus of this study, we will show that OOD sample's likelihood score from the Glow model (Kingma & Dhariwal, 2018) (Serrà et al., 2019) can be arbitrarily manipulated by our algorithm (section 2.1) such that the output probability  $p(x_{in}) \cong p(x_{out})$ , which further diminishes the effectiveness of any Glow likelihood-based detection methods.

# 2 METHODOLOGY

# 2.1 OOD ATTACK ON DNN ENCODER

In this section, we will introduce an algorithm to perform OOD attack on DNN encoder  $z = f(x)$  which takes an image  $x$  as input and transforms it into a feature vector  $z$  in a latent space. Preprocessing on  $x$  can be considered as the very first layer inside of the model  $f(x)$ . The algorithm needs a weak assumption that  $f(x)$  is differentiable. A CNN classifier can be considered a composition of a feature encoder  $z = f(x)$  and a feature classifier  $p = g(z)$  where  $p$  is the softmax probability distribution over multiple classes.

Let's consider an in-distribution sample  $x_{in}$  and an OOD sample  $x_{out}^{\prime}$ , and apply the model:  $z_{in} = f(x_{in})$  and  $z_{out}^{\prime} = f(x_{out}^{\prime})$ . Usually,  $z_{out}^{\prime} \neq z_{in}$ . However, if we add a relatively small amount of noise  $\delta$  to  $x_{out}^{\prime}$ , then it could be possible that  $f(x_{out}^{\prime} + \delta) = z_{in}$  and  $x_{out}^{\prime} + \delta$  is still OOD. This idea is realized in Algorithm 1, OOD attack on DNN Encoder.

The clip operation in the above algorithm is very important: it can ensure that  $x_{out}$  is OOD after a small modification to  $x_{out}'$ . The algorithm is inspired by the method called projected gradient descent (PGD) (Kurakin et al., 2016) (Madry et al., 2017) which is used for adversarial attack. We note that the term "adversarial attack" and "adversarial noise" usually refer to adding a small perturbation to a clean sample  $x$  in a dataset such that a classifier will incorrectly-classify the noisy

# Algorithm 1 OOD attack on DNN Encoder

Input: An in-distribution sample  $x_{in}$  in a dataset. An OOD sample  $x_{out}'$  not similar to anyone in the dataset.  $f$ , the neural network feature encoder.  $\epsilon$ , the maximum perturbation measured by Lp norm.  $N$ , the total number of iterations.  $\alpha$  the learning rate of optimizer.

Output: an OOD sample  $x_{out}$  s.t.  $f(x_{out}) \cong f(x_{in})$

# Process:

1: Generate a random noise  $\xi$  with  $||\xi ||\leq \epsilon$  
2: Initialize  $x_{out} = x_{out}' + \xi$  
3: Setup loss  $J(x_{out}) = ||f(x_{out}) - f(x_{in})||^{2}$  (L2 norm)  
4: for  $n$  from 1 to  $N$  do  
5:  $x_{out} \gets \text{clip}(x_{out} - \alpha \cdot h(J'(x_{out})))$  
6: end for

Note: The clip operation ensures that  $||x_{out} - x_{out}'||_p \leq \epsilon$ . The clip operation also ensures that pixel values stay within the feasible range (e.g. 0 to 1). If L-inf norm is used,  $h(J')$  is the sign function; and if L2 norm is used,  $h(J')$  normalizes  $J'$  by its L2 norm. Adamax optimizer is used in the implementation

sample while being able to correctly-classify the original clean sample  $x$ . Thus, OOD attack and adversarial attack are completely different things.

In practice, the Algorithm 1 can repeat many times to find the best solution. Random initialization is performed in line-1 and line-2 of the algorithm process. By adding initial random noise  $\xi$  to  $x_{out}^{\prime}$ , the algorithm will have a better chance to avoid local minimum caused by a bad initialization.

# 2.2 DIMENSIONALITY REDUCTION AND OOD ATTACK

Recall that in a classification-based OOD Detection approach, a DNN encoder transforms the input to a feature vector, i.e.,  $z = f(x)$ , and an OOD detection score is computed by another transform on  $z$ , i.e., and  $\tau = d(z)$ . If  $z_{out} \cong z_{in}$ , then  $d(z_{out}) \cong d(z_{in})$  which breaks the OOD detector regardless of the transform  $d$ . Usually, a DNN feature encoder makes dimensionality reduction: the dimension of  $z$  is significantly smaller than the dimension of  $x$ . In the example shown in Fig. 1,  $z$  is a 512-dimensional feature vector ( $dim(z) = 512$ ) in Resnet-18, and the dimension of  $x$  is 150528 ( $224 \times 224 \times 3$ ).

Dimensionality reduction in an encoder provides the opportunity for the existence of the mapping of OOD and in-distribution samples to the same locations in the latent space. This simply because the vectors in a lower-dimensional space cannot represent all of the vectors/objects in a higher-dimensional space, which is the Pigeonhole Principle. Let's do an analysis on the Resnet-18 example in Fig. 1. A pixel of the color image  $x$  has 8-bits. In the 150528-dimension discrete input space, there are  $8^{224 \times 224 \times 3}$  different images/vectors, which defines the size of the input space. float32 data type is usually used in computation, a float32 variable can roughly represent  $2^{32}$  unique real numbers. Thus, in the 512-dimensional latent space, there are  $2^{32 \times 512}$  unique vectors/objects, which defines the size of the latent space. The ratio is  $\left( \frac{2^{32 \times 512}}{8^{224 \times 224 \times 3}} \right) \ll 1$ , and it shows that the latent space is significantly smaller than the input space. Thus, for some sample  $x$  in the dataset, we can find another sample  $x'$  such that  $f(x') = f(x)$  as long as  $\dim(z) < \dim(x)$ . A question arises: will the  $x'$  be in-distribution or OOD? To answer this question, let's partition the input discrete space  $\Omega$  into two disjoint regions  $(\Omega = \Omega_{in} \cup \Omega_{out})$ ,  $\Omega_{in}$  of in-distribution samples and  $\Omega_{out}$  of OOD samples.  $|\Omega|$  denotes the size of  $\Omega$ . Usually, training set is only a subset of  $\Omega_{in}$ , and the size of  $\Omega_{out}$  is significantly larger than the size of  $\Omega_{in}$ . For example, if  $\Omega_{in}$  is ImageNet, then  $\Omega_{out}$  contains medical images, noise images, and other weird images. If  $\Omega_{in}$  contains human face images, then  $\Omega_{out}$  contains non-face images and then  $|\Omega_{in}| \ll |\Omega_{out}|$ . The latent space (z-space) is denoted by  $\mathcal{F}$  and partitioned into two subspaces:  $\mathcal{F} = \mathcal{F}_{in} \cup \mathcal{F}_{out}$ . An encoder is applied such that  $\Omega_{in} \to \mathcal{F}_{in}$  and  $\Omega_{out} \to \mathcal{F}_{out}$ . If there is overlap  $\mathcal{F}_{in} \cap \mathcal{F}_{out} \neq \emptyset$ , then the encoder is vulnerable to OOD attack. Usually, the encoder is a part of a classifier trained to classify in-distribution samples into different classes, and therefore the encoder cannot guarantee that there is no overlap between  $\mathcal{F}_{in}$  and  $\mathcal{F}_{out}$ . What is the size of  $\mathcal{F}_{in} \cap \mathcal{F}_{out}$  or what is the probability  $P(|\mathcal{F}_{in} \cap \mathcal{F}_{out}| \geq a)$ ? While it is hard to calculate it for an arbitrary encoder and dataset, we can do a worst-case-scenario analysis.

The intuition is that if  $|\Omega_{out}| \gg |\mathcal{F}|$ , then it is highly possible that the entire latent space is crawling with the shadows of OOD samples. Assuming that every OOD sample is i.i.d. mapped to the latent space with a uniform distribution over a number of  $|\mathcal{F}|$  spots, then the probability of OOD samples covering the entire latent space is  $P(\mathcal{F}_{out} = \mathcal{F}) = |\mathcal{F}|! \times \text{Stirling}(|\Omega_{out}|, |\mathcal{F}|) / |\mathcal{F}|^{\left|\Omega_{out}\right|} \to 1$  as  $|\mathcal{F}| / |\Omega_{out}| \to 0$ , where Stirling is the Stirling number of the second kind. Noting that  $|\mathcal{F}| / |\Omega_{out}| = \frac{2^{32 \times 512}}{8^{224 \times 224 \times 3} - 1.4 \times 10^7} \approx 0$  and  $1.4 \times 10^7$  being the number of samples in ImageNet, then it could be true that almost (with probability close to 1) the entire latent space of Resnet-18 is covered by the  $z$  vectors of OOD samples.

Next, we discuss how to construct OOD samples to fool neural networks. First, let's take a look at one-layer linear network:  $z = Wx$ , and make notations: an in-distribution input  $x \in \mathcal{R}^M$ , late code  $z \in \mathcal{R}^K$  and  $K \ll M$ .  $W$  is a  $K \times M$  matrix, and  $\text{rank}(W) \leq K$ . The null space of  $W$  is  $\Omega_{null} = \{\eta; W\eta = 0\}$ . Now, let's take out the basis vectors of this space,  $\eta_1, \eta_2, \dots, \eta_{M-K}$ , and compute  $x' = \sum_i \lambda_i \eta_i + x$  where  $\lambda_i$  is a non-zero scalar. Obviously,  $z' = Wx' = z$ . We can set the magnitude of the "noise"  $\sum_i \lambda_i \eta_i$  to be arbitrarily large such that  $x'$  will look like garbage and become OOD, which is another explanation of the existence of OOD samples. Then, we can try to apply this attack method to multi-layer neural network. If the neural network only uses ReLU activation, then the input-output relationship can be exactly expressed as a linear mapping (Ding et al., 2018), a similar approach can be applied layer by layer. If ReLU is not used, a new method is needed. We note that the filter bank of a convolution layer can be converted to a weight matrix. We have examined the state-of-art CNN models that are pre-trained on ImageNet and available in Pytorch, and dimensionality reduction is performed in most of the layers (except 1 or 2 layers near the input), i.e.  $|\mathcal{F}| \leq |\Omega_{in}| \ll |\Omega_{out}|$ . Instead of constructing an OOD sample by adding perturbations to an in-distribution sample, in Algorithm-1, we construct OOD samples paired with in-distribution samples by starting from a single initial sample that is OOD.

Could an encoder be made robust to the OOD attack by including OOD samples in training set for supervised binary classification: in vs out? Usually  $|\Omega_{in}| \ll |\Omega_{out}|$  and we will have to collect and label "enough" samples in  $\Omega_{out}$ , which is infeasible considering the large size of  $\Omega_{out} \approx \Omega$ . As a comparison, to enhance DNN classifier robustness against adversarial noises, it is very effective to include noisy samples in the training set, i.e.  $\Omega_{in} = \Omega_{in\_clean} \cup \Omega_{in\_noisy}$ . It is known as adversarial training (Goodfellow et al., 2018) and computationally feasible as  $|\Omega_{in\_noisy}| \ll |\Omega_{out}|$ .

# 2.3 PROBLEM OF GLOW LIKELIHOOD-BASED OOD DETECTION

Generative models have been developed to approximate the training data distribution. Glow (Kingma & Dhariwal, 2018) is one of these models, and it has a very special property: it is bijective and the latent space dimension is the same as the input space dimension, i.e., no dimensionality reduction, which is the reason that we studied this model.

Several studies have found the problem of Glow-based OOD detection: likelihoods derived from Glow may not distinguish between OOD and training samples (Ren et al., 2019) (Choi et al., 2018), and a possible fix to the issue could be using likelihood ratio (Serrà et al., 2019). In this study, we further show that OOD sample's negative log-likelihood (NLL) from the Glow model can be arbitrarily manipulated by our algorithm in which  $f(x)$  denotes NLL. The results on CelebA face image dataset are in Section 3. We think the major reason causing Glow's vulnerability to OOD attack is that we do not have enough training data in high dimensional space. Glow is a mapping:  $x_{in} \rightarrow z_{in} \rightarrow p(z_{in}) \rightarrow p(x_{in})$ , the probability of  $x_{in}$ . For an OOD sample  $x_{out}$ , the mapping is  $x_{out} \rightarrow z_{out} \rightarrow p(z_{out}) \rightarrow p(x_{out})$ . Since the number of training samples is significantly smaller than the size of the space, there are a huge number of "holes" in the space, and it is easy to put  $z_{out}$  in one of these "holes" close to  $z_{in}$  such that  $p(z_{out}) \cong p(z_{in})$ .

# 2.4 RECONSTRUCTION-BASED OOD DETECTION

Auto-encoder style OOD detection has been developed for anomaly detection (Chalapathy & Chawla, 2019)(Cohen et al., 2019) based on reconstruction error. The data flow of an auto-encoder is  $x \to z \to \hat{x}$  where  $\hat{x}$  is the reconstruction of  $x$ . The OOD detection score can be the difference between  $x$  and  $\hat{x}$ , e.g., the Lp distance  $\| x - \hat{x} \|_p$  or Mahalanobis Distance. This type of method has two known issues. The first issue is that auto-encoder may well reconstruct OOD samples, i.e.,

$x_{out} \approx \hat{x}_{out}$ . Thus, one needs to make sure it has large reconstruction errors on OOD samples, which can be done by limiting the capacity of auto-encoder or saturating it with in-distribution samples. The second issue is that pixel-to-pixel distance is not a good measurement of image dissimilarity, especially for medical images. For example,  $x$  could be a CT image of a heart and  $\hat{x}$  could be the image of the same heart that deforms a little bit, but the pixel-to-pixel distance between  $x$  and  $\hat{x}$  can be very large. Thus, a robust image similarity measurement is needed.

Interestingly, the proposed OOD attack algorithm has no effect on this type of method. Let's consider the data flow:  $x_{in} \rightarrow z_{in} \rightarrow \hat{x}_{in}$  and  $x_{out} \rightarrow z_{out} \rightarrow \hat{x}_{out}$ . If  $z_{out} = z_{in}$ , then  $\hat{x}_{out} = \hat{x}_{in}$ . Then, it is easy to find out that  $x_{out}$  is OOD because  $||x_{out} - \hat{x}_{out}||_p = ||x_{out} - \hat{x}_{in}||_p$  which is very large. Ironically, in this case, the attack algorithm helps to identify the OOD sample. In future work, we will evaluate the effectiveness of combining the proposed algorithm and auto-encoder for OOD detection.

# 3 EXPERIMENT

We applied the proposed algorithm to attack state-of-art DNN models on open image datasets. For each in-distribution sample  $x_{in}$  in our evaluation, an OOD sample  $x_{out}$  is generated by the algorithm. To measure attack strength, mean absolute percentage error is calculated by  $MAPE(z_{out}) = \max(|z_{out} - z_{in}|)/\max(|z_{in}|)$ . Here,  $z_{out} = f(x_{out})$  and  $z_{in} = f(x_{in})$ .  $|z_{out} - z_{in}|$  is an error vector, and  $\text{mean}(|z_{out} - z_{in}|)$  is the average error.  $\text{max}(|z_{in}|)$  is the maximum absolute value in the vector  $z_{in}$ . We also applied the algorithm to attack the Glow model on CelebA dataset. In all of the evaluations, L2 norm was used in the proposed algorithm. Pytorch was used to implement the algorithm. Nvidia Titan V GPU was used for model training and testing.

# 3.1 EVALUATION ON A SUBSET OF IMAGENET

ILSVRC2012 ImageNet has over 1 million images in 1000 classes. Given the limited computing power, it is impractical to test the algorithm on the whole dataset. Instead, we used a subset of 1000 images in 200 categories. The size of each image is  $224 \times 224 \times 3$ . Two CNN models pretrained on the ImageNet were evaluated, which are Resnet-18 and Densenet-121 available in Pytorch.

Resnet-18 latent space has 512 dimensions. Since ImageNet covers a variety of natural and artificial objects, we choose medical images and random-noise images to make sure that  $x_{out}^{\prime}$  is indeed OOD. Using each of the three initial OOD samples (chest x-ray, lung-CT, and random noise to be  $x_{out}^{\prime}$ ), we generated 1000 OOD samples paired with the 1000 (in-distribution) samples in the dataset and calculated MAPE values. The three MAPE histograms are shown in Fig. 3. Most of the MAPE values are less than  $0.1\%$ .

We also evaluated another CNN, named Densenet-121, and obtained similar results. The latent space has 1024 dimensions. Again, using each of the three initial OOD samples, 1000 OOD samples are generated for the samples in the dataset, and then MAPE values are calculated. The three MAPE histograms are shown in Fig. 4. Most of the MAPE values are less than  $0.1\%$ , indicating strong OOD attack.

From the results in Fig.2 to Fig. 4, it can be seen that each of the two CNN models mapped significantly different OOD samples and in-distribution samples to the same locations in the latent space. Dimensionality reduction leads to the existence of such mapping, and our algorithm can find such OOD samples out. In other words, the mapping from input space to the latent space is many-to-one, not bijective. And therefore, it is almost guaranteed that such OOD samples exist and they can break any OOD detector  $d$  that computes a detection score  $d(z)$  only from the latent space (z-space). Given the results of  $z_{out} \cong z_{in}$  ensuring  $d(z_{out}) \cong d(z_{in})$ , it is unnecessary to re-implement and test OOD detectors based on  $d(z)$ . For illustration purpose, we tested a classical OOD detection method using the maximum of softmax output as detection score (Hendrycks & Gimpel, 2016). The results are shown in Table-1, and the AUROC scores are close to 0.5, showing that the method is unable to tell the difference between the 1000 OOD samples and 1000 in-distribution samples.

![](images/d979d36a9c8b2d4e1d24f47492443e2e11e7360dab6c4c2ddcfb38e4cbab1ea4.jpg)

![](images/2ab75901bcfec07e6b76c350cb62e3b98669038b8174d0a9a8551727537b8d95.jpg)

![](images/b221a69e52434290b73f660e80b8c6f1c087acad34f8d38e8a3d849f7b39f435.jpg)

![](images/1cb7df151c315d212dcce2f642ed31f2bcbe5692e124b0b19712c760787a8d0e.jpg)

![](images/666a36c6942550e8595baefae0dbca29927044603208fcc51625d2e635a6dc4b.jpg)

![](images/3ef03bddc1ac0c60ed86c048805b50387c46e7888f4f90078c12da2001a92175.jpg)  
(a)

![](images/ca5ff2938c3c7e0e36b991ece9cfd89d29912250997a27a7c33a77a535a8fce9.jpg)

![](images/d537c1d774ecb77eab7780e1f3aab1aaf6c39026338ceda7aeaac9a4ba5e323d.jpg)

![](images/39d1128f421feae5dc56197efd12c5272498f8267faa42887a40bddfdaeb9c77.jpg)

![](images/298602b747fd3961434be3dd7dac4863b54f32cff61cb5324afe10cba325b32d.jpg)

![](images/1efd865291f0a5e1399847ecd45720800dd69f62e3c88bfb7fada210f53437c1.jpg)

![](images/3f2996d5d158106f7e10acaac079a7bab6fe165cc2a23402948e8df0bab8b881.jpg)  
(b)

![](images/bd73342427b76972618a44e1f38aa1d647b12bb501db1b5f658eb3fae422035e.jpg)

![](images/85c3340187f06d14a98f4f23da56513608e3de931998476067d15920f3ebfa74.jpg)

![](images/3ac8868bcec75514406c09e113999a82b703beb08b769d5760c75d9a3e0b86ff.jpg)

![](images/43d6845b0c0faf9663a114e53c970287bb60dcf8a90309680ad6147f4dcb7458.jpg)

![](images/f075ae3fbaee775fa7ab2f6437f492c132b7d495898bbf470c771cbc4ad1ddad.jpg)

![](images/ad1d6f86b30e26a9a3b2c1b3b910cef7a6b84fd5450757d59498f13e4b84c4a2.jpg)  
(c)

![](images/bc7437f6401c2a4e71feabc4dea55902c91ca2bff86fcff9c107293cfbb83448.jpg)

![](images/2ed20259e6c27b1b37bb907c6301db2f251e9375d8092395a9a1a11e6d41b2f8.jpg)

![](images/ef1e67704cf34a969a0bf1ed46543058b79b48b8926344271a7b31a95c8d3238.jpg)

![](images/baa320ca40eaaa26a36bc9d8dc59230648d98ecd4042b18bb265839a4f139c52.jpg)

![](images/2a465d3520c7b2b7c050d1df1a7d71ebb3be6bfb6bac015224eab408a0a9ec45.jpg)

![](images/446ea091039d8503645391861cb5171524d69272e0a05b81a658b335d3d2aff9.jpg)  
(d)

![](images/500391fd3e9c9540eae11d4b9a341669eb0a780ec9b10c0e957be9643508983b.jpg)  
Figure 2: The 1st column shows three in-distribution samples (e.g. Santa Claus)  $x_{in}$ , and the corresponding scatter plots of  $z_{in}$  (blue dots). The 2nd column shows OOD samples  $x_{out}$  generated from a CT image  $x_{out}'$ , and the corresponding scatter plots of  $z_{out}$  (red) and  $z_{in}$  (blue). The 3rd column shows OOD samples  $x_{out}$  generated from a random image  $x_{out}'$ , and the corresponding scatter plots of  $z_{out}$  (red) and  $z_{in}$  (blue). The 4th column shows OOD samples  $x_{out}$  generated from a x-ray image  $x_{out}'$ , and the corresponding scatter plots of  $z_{out}$  (red) and  $z_{in}$  (blue). MAPE values are embedded in these scatter-plots. Please zoom-in for better visualization.

![](images/4d6d43d2b3360b99cc611ef3e7ea5aa8c697c16fecb4c50ebee7226049b4074e.jpg)  
Figure 3: Left: the MAPE histogram using a chest x-ray as the initial OOD sample. Middle: the MAPE histogram using a lung CT image as the initial OOD sample. Right: the MAPE histogram using a random-noise image as the initial OOD sample. Please zoom-in for better visualization.

![](images/d7df9ce8d6933c2c56c7ce7732d2e9cdc97ec9c35813c63438d0fe2b43345fff.jpg)

![](images/78f6108897e32a2184b78e031eaf60ca5ba7c287d2e134d087af3d9540d8f7ba.jpg)  
Figure 4: Left: MAPE histogram using a chest x-ray as the initial OOD sample. Middle: MAPE histogram using a lung CT image as the initial OOD sample. Right: MAPE histogram using a random-noise image as the initial OOD sample. Please zoom-in for better visualization.

![](images/c09ede4e5b78a57e13cbdf482dd0b8df2a5928c64d3e2b14948acc0b353a0495.jpg)

![](images/c1bebce48289552fe865b31827504534a6f7f4609ec491679880de68ddb5ebd1.jpg)

Table 1: AUROC of two networks under OOD attack with each of x-ray, CT and noise as the initial OOD sample  

<table><tr><td></td><td>x-ray</td><td>CT</td><td>random-noise</td></tr><tr><td>Resnet-18</td><td>0.643</td><td>0.633</td><td>0.500</td></tr><tr><td>Densenet-121</td><td>0.638</td><td>0.651</td><td>0.500</td></tr></table>

# 3.2 EVALUATION ON CELEBA DATASET

We tested the algorithm and the Glow model (Kingma & Dhariwal, 2018) on the CelebA dataset (human face images). The size of each image is  $64 \times 64 \times 3$ . After training, the model was able to generate realistic face images. The model also outputs the negative log-likelihood (NLL) of the input sample, i.e.,  $NNL(x) = -\log(p(x))$ . By setting  $f(x) = NNL(x)$ , our algorithm can make  $f(x_{out})$  to be close to 0 or very large to match any  $f(x_{in})$ , which renders NLL score useless for OOD detection. To demonstrate the effectiveness of our algorithm, we randomly selected 160 (in-distribution) samples in the dataset. We used a color spiral image as the initial OOD sample  $x_{out}^{\prime}$ , and  $NNL(x_{out}^{\prime}) = 3.5268$ . The distributions of  $NLL(x_{in})$  from 160 in-distribution samples and  $NLL(x_{out})$  from 160 corresponding OOD samples, as well as OOD sample images are shown in Fig. 5. The two distributions are almost identical. More examples of OOD samples are shown in Fig. 6. In each row of Fig. 6, although the images have different NLL scores, they look like each other.

![](images/5911e64cd3468424a9436fcd86d3c7ebeafcdd307173dcd86f226748fea539a0.jpg)

![](images/6d1af6b0d37830b5e9387d732364e51c278622ff30b3db8ba5890989e19f2596.jpg)

![](images/176ae17adf972499c72bbdcf999e0c2ec0e160235c11c5f500607330a4e192a9.jpg)

![](images/5ef069fa0913c54a5cdd4f5b55c46daf0d7d78df9dea447da7374022a9ea1390.jpg)

![](images/482ae43630a7d39670d2717c5e696ab107c41f93baf8747a0bc91dd733afc82c.jpg)

![](images/9c7bd13331e3976c35f40c32e7194e1a7237e8c242b2a7e0ea40aa60c73e87b7.jpg)

![](images/dba759dd3b8ef3dd80a690917f32f9ae3555f6b3bf8295ec7053a3076aed21f1.jpg)

![](images/ec17faaef784afff1abf70cab5d8a8d839b1df6f0025185f68398a1ef14e0317.jpg)

![](images/de83917bd3ee38e51593fb2838afafcceafcd6f1ab1a908df7af4424f27299c2.jpg)

![](images/01fd76c4579cb01324bd993c47fb62170a1a9d5e0e73b7f4112c0639be23e4cb.jpg)  
Figure 5: Top: NLL histogram (blue bars) of Figure 6: Top: OOD samples generated by us the in-distribution; samples Middle: NLL hist- ing one  $2 \times 2$  checkerbox image for initializatogram (red bars) of OOD samples; Bottom: tion; Bottom: OOD samples generated by using some OOD samples with NLL from 0 to 1. The one  $8 \times 8$  checkerbox image for initialization. initial OOD sample is a spiral image.  
NLL=0.00

![](images/6487cea5640cf8cc62bc2a1cd1680519134bc56ff61d41a400e77d74c71c2eb5.jpg)  
NLL=0.20

![](images/166448ec9bd8fa9d1e938944badaf4856b5cf8876e790179444235275452aa59.jpg)  
NLL=0.40

![](images/aad6e7ba0746bbd21972945c4e8302ed63dab8bc7affe93f8f6bc603b1e007c0.jpg)  
NLL=0.60  
NLL=0.80

![](images/a29dac4244dee0e82aa1f41c2f4800230c747158f68ccafecfb5bb05c4d7ec0c.jpg)

![](images/ca39b17b4253f68d471b043779392e0822b90455dbdf4e61c05e941095904a0b.jpg)  
NLL=1.00

# Other three evaluations are shown in Appendix.

# 4 DICUSSION

For a DNN model that makes dimensionality reduction and therefore is not bijective, our algorithm can find OOD samples that will be mapped by the model to the locations of the in-distribution samples in the latent space, which invalidates any OOD detector that computes a detection score only from the latent space. We performed extensive tests to demonstrate the analysis and our algorithm on different datasets and applications.

The only assumption of the algorithm is that the model is differentiable. Non-differentiable preprocessing on the input cannot be incorporated into the model as the very first layer, and therefore it is ignored in this study. Such preprocessing has little effect on the OOD sample space and only makes it harder to find OOD samples. To handle such cases, our algorithm can be enhanced by using a sampling-based gradient estimator, as done in SPSA (Maryak & Chin, 2001)(Uesato et al., 2018).

We also show that the output NLL score of the generative model Glow is easily manipulated by the algorithm, and therefore Glow NLL score is not suitable for OOD detection. We note that there are other generative methods for OOD detection, which have shown promising results. Evaluation of these methods is out of the scope of this study.

We would like to point out that it is difficult to evaluate an OOD detector to "prove" that it can detect, say  $90\%$  of the OOD samples by experimentally testing it on  $\Omega_{out}$  because  $\Omega_{out}$  is too large to be tested on:  $|\Omega_{in}| \ll |\Omega_{out}| \approx |\Omega|$ . For example, if Fashion-MNIST is used as in-distribution, then MINST and Omniglot are usually as OOD, which is the "standard" approach in the literature. Clearly, MINST and Omniglot cannot cover  $\Omega_{out}$  the space of OOD samples. If the image size is larger, then  $|\Omega_{out}|$  becomes much larger. Could we design an evaluation method (experimental or analytical) that does not rely on OOD samples?

At last, we present a simple method for OOD detection with theoretical guarantee: subspace saturation training of bijective DNN for OOD detection. The method does not need any OOD samples. In a bijective mapping,  $x \to f(x) \to z \to f^{-1}(x) \to x$ , the dimension of output  $z$  is equal to the dimension of input  $x$ , i.e., no dimensionality reduction. We "manually" partition the latent space (z-space) into two disjoint subspaces  $\mathcal{F} = A \cup B$ ,  $A \cap B = empty$ , and then the network will be trained to map a huge number of in-distribution samples into the subspace A such that the subspace A is saturated by in-distribution samples (i.e. no 'holes'). Thus, for any OOD sample  $x_{out}$ , it is guaranteed that  $z_{out} \in B$  because there is no empty spot in the subspace  $A$  and the mapping is bijective. As long as the subspace A is saturated by in-distribution samples, then the network can detect all of the OOD samples, which means to evaluate the performance of the network, we only need to check the saturation rate and do not need OOD samples. We provide two implementations of this method in the Appendix.

Before the OOD issue is fully resolved, for life-critical applications, any machine learning system that uses DNN classifiers should not make decisions independently and can only serve as assistants to humans.

We will release the code on GitHub when the paper is accepted. All figures are in high-resolution, please zoom in.

# REFERENCES

Alexander A Alemi, Ian Fischer, and Joshua V Dillon. Uncertainty in the variational information bottleneck. arXiv preprint arXiv:1807.00906, 2018.  
Alvaro Arcos-Garcia, Juan A Alvarez-Garcia, and Luis M Soria-Morillo. Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods. Neural Networks, 99:158-165, 2018.  
Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jorn-Henrik Jacobsen. Invertible residual networks. In International Conference on Machine Learning, pp. 573-582, 2019.

Raghavendra Chalopathy and Sanjay Chawla. Deep learning for anomaly detection: A survey. arXiv preprint arXiv:1901.03407, 2019.  
Hyunsun Choi, Eric Jang, and Alexander A Alemi. Waic, but why? generative ensembles for robust anomaly detection. arXiv preprint arXiv:1810.01392, 2018.  
Joseph Paul Cohen, Paul Bertin, and Vincent Frappier. Chester: A web delivered locally computed chest x-ray disease prediction system. arXiv preprint arXiv:1901.11210, 2019.  
Gavin Weiguang Ding, Yash Sharma, Kry Yik Chau Lui, and Ruitong Huang. Max-margin adversarial (mma) training: Direct input space margin maximization through adversarial training. arXiv preprint arXiv:1812.02637, 2018.  
Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1625-1634, 2018.  
Ian Goodfellow, Patrick McDaniel, and Nicolas Papernot. Making machine learning robust against adversarial inputs. Communications of the ACM, 61(7):56-66, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016.  
Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. arXiv preprint arXiv:1812.04606, 2018.  
Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Carolina CS Valentim, Huiying Liang, Sally L Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al. Identifying medical diagnoses and treatable diseases by image-based deep learning. Cell, 172(5):1122-1131, 2018.  
Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In Advances in neural information processing systems, pp. 10215-10224, 2018.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016.  
Kimin Lee, Honglak Lee, Kibok Lee, and Jinwoo Shin. Training confidence-calibrated classifiers for detecting out-of-distribution samples. arXiv preprint arXiv:1711.09325, 2017.  
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. In Advances in Neural Information Processing Systems, pp. 7167-7177, 2018.  
Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. arXiv preprint arXiv:1706.02690, 2017.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
John L Maryak and Daniel C Chin. Global random optimization by simultaneous perturbation stochastic approximation. In Proceedings of the 2001 American Control Conference (Cat. No. 01CH37148), volume 2, pp. 756-762. IEEE, 2001.  
Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don't know? arXiv preprint arXiv:1810.09136, 2018.  
Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 427-436, 2015.

Jie Ren, Peter J Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan. Likelihood ratios for out-of-distribution detection. In Advances in Neural Information Processing Systems, pp. 14707-14718, 2019.  
Joan Serrà, David Álvarez, Vicenç Gómez, Olga Slizovskaia, José F Núñez, and Jordi Luque. Input complexity and out-of-distribution detection with likelihood-based generative models. arXiv preprint arXiv:1909.11480, 2019.  
Feng Shi, Jun Wang, Jun Shi, Ziyan Wu, Qian Wang, Zhenyu Tang, Kelei He, Yinghuan Shi, and Dinggang Shen. Review of artificial intelligence techniques in imaging data acquisition, segmentation and diagnosis for Covid-19. IEEE reviews in biomedical engineering, 2020.  
Eduardo Soares, Plamen Angelov, Sarah Biaso, Michele Higa Froes, and Daniel Kanda Abe. Sarscov-2 ct-scan dataset: A large dataset of real patients ct scans for sars-cov-2 identification. medRxiv, 2020.  
Jonathan Uesato, Brendan O'Donoghue, Aaron van den Oord, and Pushmeet Kohli. Adversarial risk and the dangers of evaluating against weak attacks. arXiv preprint arXiv:1802.05666, 2018.  
Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European conference on computer vision (ECCV), pp. 3-19, 2018.
