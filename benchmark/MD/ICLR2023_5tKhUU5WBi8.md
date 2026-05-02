# OUT-OF-DISTRIBUTION DETECTION WITH DIFFUSION-BASED NEIGHBORHOOD

Anonymous authors

Paper under double-blind review

# ABSTRACT

Out-of-distribution (OOD) detection is an important task to ensure the reliability and safety of deep learning and the discriminator models outperform others for now. However, the feature extraction of such models must compress the data and lose certain information, leaving room for bad cases and malicious attacks. However, despite effectively fitting the data distribution and producing high-quality samples, generative models lack suitable indicator scores to match with discriminator models in the OOD detection tasks. In this paper, we find that these two kinds of models can be combined to solve each other's problems. We introduce diffusion models (DMs), a kind of powerful generative model, into OOD detection and find that the denoising process of DMs also functions as a novel form of asymmetric interpolation. This property establishes a diffusion-based neighborhood for each input data. Then, we perform discriminator-based OOD detection based on the diffusion-based neighborhood instead of isolated data. In this combination, the discriminator models provide detection metrics for generation models and the diffusion-based neighborhood reduces the information loss of feature extraction. According to our experiments on CIFAR10 and CIFAR100, our new methods successfully outperform state-of-the-art methods. Our implementation is put in the supplementary materials.

# 1 INTRODUCTION

Out-of-distribution (OOD) detection is an important task for deep models that helps the models determine their capability boundary and keep them from being fooled by OOD data. It has strong connection with many real-world machine learning applications, such as cybersecurity (Xin et al., 2018), medical diagnosis (Latif et al., 2018; Guo et al., 2020) and autopilot (Geiger et al., 2012). The existing methods for OOD detection can be generally categorized into discriminator-based and generation-based methods. The discriminator-based methods (Wang et al., 2022) use the logit or the feature space to do that. The generation-based methods (An & Cho, 2015; Nalisnick et al., 2019) use the reconstruction difference in data space or density estimation in latent space to do that.

The discriminator-based methods can extract useful features and make the detection faster and better in most cases. However, such extraction and compression lose some information and leave room for bad cases and malicious attacks (Goodfellow et al., 2014; Amodei et al., 2016). The generation-based methods can capture the whole data distribution but lack effective indicator scores to compete with the SOTA discriminator-based methods, partly because of the curse of dimensionality. Previous works mostly concentrate on solving these challenges using only one kind of model. For discriminator-based methods, Wang et al. (2022) combine the information from both features and logits. Sehwag et al. (2020) use self-supervised learning to improve the feature extraction. For generation-based methods, Nalisnick et al. (2019) use the typicality set to design better indicator scores. Jiang et al. (2022) use statistical methods in the latent space, such as the Kolmogorov-Smirnov test.

In addition to overcoming the problems of each kind of model by itself, we find that generative and discriminative models can be combined and solve each other's problems. Our solution is that we don't expect discriminator models to handle the target feature space perfectly, but by introducing additional generative models to adjust the features. Such adjustment establishes a dynamic adjustment between the discriminator-sensitive features and the remaining. If the discriminator result is relatively consistent under such adjustment. This means the features of the input have a similar density in

the discriminator-sensitive area and the remaining, and the discriminator result is not overconfident. When the discriminator result has high confidence and decreases sharply, the features of input only concentrate on the discriminator-sensitive area, which means that the discriminator result is overconfident and the input is most likely an OOD sample.

To design suitable generation strategies that can enhance discriminator models, we introduce diffusion models (DMs), which play an important role in generation models, into OOD detection. DMs have created many state-of-the-art generation results, including (Vahdat et al., 2021; Ho et al., 2022). We think their powerful data fitting ability can be beneficial to OOD detection, too. We dive into the structure of DMs and find that the diffusion denoising process (DDP) of DMs can be an ideal choice for the dynamic adjustment we mentioned above. Because it can adjust any level of feature space and provides tools to keep the adjustment under control by using the denoising and interpolation properties. Such dynamic adjustment needs to be resampled several times to make the result convergent, which builds a neighborhood of input data, called the diffusion-based neighborhood (DiffNB). According to the property of dynamic adjustment, the high-level feature of the elements in the whole DiffNB should keep a dynamic balance and there is only a small change for in-distribution (InD) data. On the other hand, the feature can change sharply for OOD data. Therefore, we can determine OOD samples by detecting such changes using Euclidean distance. Our pipeline is in Figure 1.

![](images/d67649cbd650f56311349317c486c57ef6050f5387078bb718a17249d108c76f.jpg)  
Figure 1: The pipeline of our detection method.

We choose ten representative methods to compare with our methods on several datasets: CIFAR10, CIFAR100 and so on. According to our experiments, our new methods outperform existing models and methods in most cases. Our work has the following contributions:

- We find that the diffusion denoising process of invertible diffusion models is a novel kind of asymmetric interpolation, which can keep the InD data relatively unchanged and provide tools to control the direction of the denoising process.  
- We design a general strategy to combine the benefit of discriminator and generation models. Under the guidance of this strategy, we use a ResNet to extract features and use the diffusion denoising process of a diffusion model to reduce information loss.  
- Our method is explainable, which has a clear motivation and solid theoretical background, in the meanwhile, it gets competitive OOD detection results with SOTA methods.

# 2 BACKGROUND

In this section, we first introduce existing methods for OOD detection. Then, we show the development of diffusion models related to our paper. Because of the limited space in the main paper, more related works about diffusion models can be found in Appendix A.1.1.

# 2.1 OUT-OF-DISTRIBUTION DETECTION

OOD detection is an important task that can help neural networks to determine their capability boundary. More specifically, let  $X = \{x_{1},\ldots ,x_{n}\} \sim p$  be a group of images from the in-distribution (InD)  $p$ . When we get another group of data  $Y = \{y_{1},\dots ,y_{n}\}$ , we need to decide whether this group is from (InD)  $p$  or an unknown distribution  $q$ . If  $n = 1$ , this is pointwise OOD detection, and if  $n\geq 2$ , this is group OOD detection. In general, the existing OOD detection methods can be categorized into discriminator-based and generation-based methods.

Discriminator-based methods design indicator scores based on the output of discriminator models. Some methods can be used without modifying the model. ODIN (Liang et al., 2018) uses temperature scaling and the softmax results to detect OOD samples. ViM (Wang et al., 2022) combines the information of features and logits. KNN (Sun et al., 2022) includes the kth nearest neighbor of the input in feature space into the detection process. Some methods try to improve the detection ability in

the training process. G-ODIN (Hsu et al., 2020) designs a new loss function. ConfGAN (Sricharan & Srivastava, 2018) generates OOD data using GANs to help the discriminator models to determine the boundary. PixMix (Hendrycks et al., 2022) uses data augmentation to improve the detection results. SSD (Sehwag et al., 2020) uses self-supervised learning to improve feature extraction.

Generation-based methods use the reconstruction difference in the input space and the density estimation in the latent space to do OOD detection. An & Cho (2015) use the reconstruction ability of VAEs. Some methods assume that the generation models can reconstruct the in-distribution data better. Some methods use the Distribution transformation capability of generation models and transfer the input distribution into simple Gaussian distribution. The likelihood of the input becomes a direct choice, but Nalisnick et al. (2018) finds that OOD data can also locate in the high-likelihood area. Nalisnick et al. (2019) find the InD data is concentrated in the typical set instead of the high likelihood area and design new methods using the typical set. Serrà et al. (2019) find that we can use input complexity to correct the bias of likelihood. In addition to the likelihood, many existing statistical methods can detect whether a distribution obeys standard Gaussian distribution. Zhang et al. (2020) uses KL-divergence to detect OOD data. Jiang et al. (2022) use a nonparametric statistics method called the Kolmogorov-Smirnov test.

# 2.2 DIFFUSION MODEL

Classical diffusion model DMs build a transformation from Gaussian distribution to image distribution through a multistep denoising process. Given a data distribution  $x_0 \sim q(x_0)$ , the diffusion process satisfies a Markov process as following Ho et al. (2020):

$$
q \left(x _ {1: T} \mid x _ {0}\right) = \prod_ {t = 1} ^ {T} \mathcal {N} \left(\sqrt {1 - \beta_ {t}} x _ {t - 1}, \beta_ {t} I\right) \tag {1}
$$

$$
q \left(x _ {t} \mid x _ {0}\right) = \mathcal {N} \left(\sqrt {\bar {\alpha} _ {t}} x _ {0}, (1 - \bar {\alpha} _ {t}) I\right).
$$

Here,  $T = 1000$ , which is the max iteration step.  $\beta_{t} \in (0,1)$ , which controls the speed of adding noise. Additionally,  $\alpha_{t} = 1 - \beta_{t}$ ,  $\bar{\alpha}_{t} = \prod_{i=1}^{t} \alpha_{i}$ ,  $\bar{\mu}_{t} = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_{t}}{1 - \bar{\alpha}_{t}} x_{0} + \frac{\sqrt{\bar{\alpha}_{t}} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}} x_{t}$  and  $\bar{\beta}_{t} = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \beta_{t}$ . The objective function is defined by:

$$
L _ {t - 1} = \mathbb {E} _ {x _ {0}, \epsilon} \left[ \frac {\beta_ {t} ^ {2}}{\alpha_ {t} (1 - \bar {\alpha} _ {t})} | | \epsilon - \epsilon_ {\theta} (\sqrt {\bar {\alpha} _ {t}} x _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \epsilon , t) | | ^ {2} \right]. \tag {2}
$$

Here,  $\epsilon_{\theta}$  is an estimate of the noise  $\epsilon$ . After we get well-trained  $\epsilon_{\theta}$ , according to Song et al. (2021a), the denoising process of Denoising Diffusion Probabilistic Models (DDPMs) and Denoising Diffusion Implicit Models (DDIMs) satisfies:

$$
x _ {t - \delta} = \sqrt {\bar {\alpha} _ {t - \delta}} \left(\frac {x _ {t} - \sqrt {1 - \bar {\alpha} _ {t}} \epsilon_ {\theta} (x _ {t} , t)}{\sqrt {\bar {\alpha} _ {t}}}\right) + \sqrt {1 - \bar {\alpha} _ {t - \delta} - \sigma_ {t} ^ {2}} \epsilon_ {\theta} (x _ {t}, t) + \sigma_ {t} \epsilon_ {t}. \qquad (3)
$$

Here,  $\delta$  is the iteration step size. If  $\sigma_{t}$  equals one, Equation (3) represents the denoising process of DDPMs; if  $\sigma_{t}$  equals zero, this equation represents the denoising process of DDIMs.

Score-based generation model Song et al. (2021b) show that the diffusion-denoising process can also be treated as two differential equations:

$$
d x = (\sqrt {1 - \beta (t) - 1}) x (t) d t + \sqrt {\beta (t)} d w
$$

$$
d x = \left(\left(\sqrt {1 - \beta (t) - 1}\right) x (t) - \frac {1}{2} \beta (t) s _ {\theta} (x (t), t)\right) d t. \tag {4}
$$

This is called probability flows (PFs). The noise  $\epsilon_{\theta}$  of DMs and the gradient of logic likelihood  $s_{\theta}$  are equivalent Bao et al. (2022). More specifically, we have that  $s_{\theta}(x,t) = -\frac{1}{1 - \bar{\alpha}_t}\epsilon_{\theta}(x,t)$ .

Pseudo numerical method Liu et al. (2022) provide pseudo numerical methods for diffusion models (PNDMs) to accelerate DDIMs. PNDMs define Equation (3) with  $\sigma_t = 0$  as transfer function:

$$
\phi (x _ {t}, \epsilon_ {t}, t, t - \delta) = \frac {\sqrt {\bar {\alpha} _ {t - \delta}}}{\sqrt {\bar {\alpha} _ {t}}} x _ {t} - \frac {(\bar {\alpha} _ {t - \delta} - \bar {\alpha} _ {t})}{\sqrt {\bar {\alpha} _ {t}} (\sqrt {(1 - \bar {\alpha} _ {t - \delta}) \bar {\alpha} _ {t}} + \sqrt {(1 - \bar {\alpha} _ {t}) \bar {\alpha} _ {t - \delta}})} \epsilon_ {t}. \tag {5}
$$

![](images/0106f2881773fcb87e7cab0e8a90faea1708fa1f516ba17c5a5690762d8866c2.jpg)  
Figure 2: The left side shows the detection process of our toy example and a bad case under this setting. The right side provides three different operators to correct the result of this bad case.

PNDMs combine this transfer function with the noise estimated by classical numerical methods, like the linear multistep method, to get the new denoising equations:

$$
\left\{ \begin{array}{l} \epsilon_ {t} ^ {\prime} = \frac {1}{2 4} \left(5 5 \epsilon_ {t} - 5 9 \epsilon_ {t + \delta} + 3 7 \epsilon_ {t + 2 \delta} - 9 \epsilon_ {t + 3 \delta}\right) \\ x _ {t - \delta} = \phi \left(x _ {t}, \epsilon_ {t} ^ {\prime}, t, t - \delta\right). \end{array} \right. \tag {6}
$$

Here,  $\epsilon_{t} = \epsilon_{\theta}(x_{t},t)$ . Both PFs and PNDMs accelerate the denoising process without loss of quality.

Classifier-free guidance Ho & Salimans (2021) show a simple and effective way to generate conditional samples called classifier-free guidance. It adds a condition embedding  $c$  into  $\epsilon_{\theta}$  in the training process and changes the final estimation of noise as:

$$
\bar {\epsilon} _ {\theta} \left(x _ {t}, c\right) = (1 + \omega) \epsilon_ {\theta} \left(x _ {t}, t\right) - \omega \epsilon_ {\theta} \left(x _ {t}\right). \tag {7}
$$

Here,  $\omega$  is the guidance weight, which controls the balance between realness and diversity.

# 3 DIFFUSION-BASED NEIGHBORHOOD FOR OOD DETECTION

In this section, we give more detailed explanations of the toy example mentioned in Section 1. It inspires us to combine the interpolation of diffusion models with the feature extraction of discriminator models to do OOD detection. Then, we dive into the structure of diffusion models to show that they can provide us with a special kind of interpolation using the diffusion denoising process, which is especially suitable for OOD detection. After that, we provide the whole pipeline of our methods. We use a kth neighbor search and a diffusion-based interpolation to build the corresponding diffusion-based neighbor of input data in image space and then use a discriminator model to detect OOD samples based on the whole diffusion-based neighbor.

# 3.1 TOY EXAMPLE

Here, we first restrict the general OOD detection to a simple case and show how to solve it perfectly. Our input is from image space and our training set contains only one element zero (the empty image). Our InD is just a uniform distribution in a spherical neighborhood of zero and our target is to detect whether a new input is InD or OOD, namely, detect whether it falls in the neighborhood of zero. In addition, we need to do this task using a discriminator similar to the real cases. Here, we choose a simple mask operator to represent the information loss. We show this toy example on the left side of Figure 2 and the third row shows a bad case under this setting, which looks like zero after the mask operator. Then on the right side of Figure 2, we use three kinds of additional operators moving, mixing and reconstruction to solve this problem.

We find that the first two kinds of solutions can perfectly solve the problem of bad cases. To make this claim strict, we need to make some definitions. Each image can be represented as a function on  $[0,1]^2$  and the value of  $f(x,y)$  is the RGB value at position  $(x,y)$ . And the images are continuous in most positions. Therefore, we simplify the input space to  $\mathcal{C}([0,1]^2)$  the continuous function on  $[0,1]^2$  and then to one-dimensional  $\mathcal{C}([0,1])$  for simplicity. The mask operator is a restriction

operator $^1$ $\phi_S(f) = f|_{S=[0,0.25]\cup[0.5,0.75]}$  here and we call  $S$  the support area. The InD is just  $\{f\in \mathcal{C}([0,1])\mid |f|\leq \delta \}$  and  $|f|$  is the max absolute value of  $f$  on  $[0,1]$ . The bad cases form a set, called the annihilator set  $\mathcal{A}_{\delta}(\phi_S)$ . This set satisfies  $\mathcal{A}_{\delta}(\phi_S) = \{u|\phi_S(u) = \phi_S(0) = 0,|u| > \delta \}$ . And the moving and mixing operators are some additional operators  $\{g_i\}$  that can be compounded with the  $\phi_S$ . Now, we can transfer a toy example into:

Given the input space  $\mathcal{C}([0,1])$ , the restriction operator  $\phi_S$  and a fixed  $\delta$ , how can we design additional operator set  $\{g_i\}$  to minimize the annihilator set  $\mathcal{A}_{\delta}(\{g_i|_S\})$ ?

A straightforward solution is that for each  $f \in \mathcal{C}(\mathbb{R})$ , let  $\{g(x)_i = f(x + i)|i \in \{0, \pm 0.25\}\}$ , which represents the moving operator. The proof is that we can use  $g_0|_S$  to get the information about  $f$  on  $S$  and use  $g_{\pm 0.25}|_S$  to get the remaining on  $[0,1] / S$ . Then  $f$  must satisfy  $|f(x)| < \delta, \forall x \in [0,1]$ . There also exist other kinds of solutions. For example, let  $\{g(x)_{a,b} \equiv \frac{1}{b - a} \int_a^b f(x)dx|a,b \in [0,1]\}$ , which represents the mixing operator. The full proof can be found in Appendix A.2.

Existing methods, whether using discriminators or generators, all try to design better discriminator operators  $\phi$  directly. However, we want to show that even if we use a relatively bad downsampling operator, we can eliminate the information loss by adding an additional operator set. The key point of these solutions is that they move or mix information from one place to another. Then the restriction operator can get the whole information from a small support area  $S$ . Such ideas can be generalized to more general cases and the third reconstruction operator in Figure 2 plays a central role here. We find that the destruction and reconstruction strategy successfully contains the moving operator (the face of the cat moves to the left and down), the mix operator (the boundary of each small box becomes unclear) and the semantic level moving operator (the color of the cat is lighter). We can move or mix the information at both the semantic and pixel level instead of only the pixel level now. Therefore, it provides the possibility to extend our solution to more general cases.

Such destruction and reconstruction strategy can be finished with any kind of generation model. However, this strategy also has its own weakness. In Figure 2, we can find the OOD examples more easily using a reconstruction operator. However, we also need to avoid determining the InD data as OOD data, namely, to keep the pure white picture from being dirty. This challenge tells us that we need to add more control to such destruction and reconstruction strategy. In the following, we use the diffusion-denoising process to solve this challenge. It combines the benefit of denoising and interpolation and provide many powerful tools to control the destruction and reconstruction process.

# 3.2 DIFFUSION DENOISING PROCESS

To design a suitable destruction and reconstruction strategy satisfying the above requirements. We introduce diffusion models into OOD detection. We dive into the structure of diffusion models and show that the diffusion denoising process (DDP) is a kind of interpolation under the invertible condition. Both denoising and interpolation can provide additional control for the destruction and reconstruction strategy. Song et al. (2021a;b) show the possibility that DMs can be invertible theoretically. However, all the methods in the above papers are totally invertible only when the generation step goes to infinity. The invertibility in practice, such as only 50 steps of iteration, is not been well analyzed before. Therefore, we use these existing DMs as examples to analyze how to make DMs more invertible under actual situations. Here, we use the reconstruction error to represent

the invertibility of DMs.

Invertible diffusion model We show the test results in Figure 3. For DDIMs, the error occurs at the beginning, and the error accumulates with the increase of the total generation step. For PFs, the initial error is not huge, but the cumulative error occurs when the number of the total generation steps is bigger than 500. DDIMs are first-order methods, and other methods are high-order methods. We can say higher convergent order can increase the invertibility. PFs use numerical methods of adaptive step size, and PNDMs use methods of fixed step size. Therefore, we think that fixed step size can also benefit invertibility. To verify this,

![](images/4c22c4b5a7aa77f765bf7b4b9ca1e5eca3716acaeb6c5b5b57d9eb6a6f5dcad9.jpg)  
Figure 3: The reconstruction error under different iteration interval  $[0, t]$  and fixed step size 20.

we replace the methods of adaptive step size used by PFs with the

methods of fixed step size and call it probability flows plus  $(\mathrm{PFs}+)$ . We find that the errors decrease immediately. The reason is that fixed step size maintains consistency between the sampling locations of the forward and reverse processes, which benefits the invertibility. Combining the above analysis, we have the following property:

# High convergent order and fixed iteration step size can improve the invertibility of DMs under fixed total iteration steps.

Interpolation After we have available invertible DMs, we get an interesting application: a new kind of interpolation. Let us assume that  $x_0$  is an image and  $\epsilon$  is Gaussian noise, which is the reverse of an image  $x_1$ , and we use  $x_0$  and  $\epsilon$  to get  $x_t$ . The diffusion-denoising process is showed in Algorithm 1 and we define DDP as  $\Phi(x_t, t, 0)$ . When let  $t$  equals zeros, we do not add any noise to the images, we can get the original image  $x_0$ , and when  $t$  equals  $T$ , we remove the total image and only leave the noise  $\epsilon$ , we will do a full denoising process to this noise. Because this denoising process is invertible, we can get the original image  $x_1$ . Therefore, the outputs of DDP gradually change from  $x_0$  to  $x_1$ .

DDP is a kind of destruction and reconstruction strategy, in the meanwhile, it is also a denoising operator and an interpolation operator. That DDP is a denoising operator means that it can keep the InD data relatively unchanged and pull the OOD data to the high-density area of InD, which is a distribution-sensitive property. That DDP is an interpolation operator means that even if we cannot reconstruct the input perfectly, we can at least control the direction of change. Both these properties give us more possibilities to control DDP and solve the challenge mentioned above. Finally, we get the following property:

The invertible diffusion-denoising process is also a type of distribution-sensitive interpolation.

# 3.3 DIFFUSION-BASED NEIGHBORHOOD

After we prepare enough properties of diffusion models, we can use a new example to show how DDP works in the feature space now. In Figure 4, we use the light blue area to represent the ideal feature set and use the dark blue area to show the features used by the discriminator. The point in each image represents several features of a single input and the corresponding arrow shows the movement under DDP. And we use the number of points  $N$  that fall in the dark blue area to represent the value of confidence.

For a normal InD input (the first row), its features are uniformly distributed in the feature area, which ensures that  $N$  maintains a dynamic balance under the perturbation of DDP in the feature space. For a normal OOD input (the second row), DDP pulls them to the high-density area and  $N$  increases at the same time. For more challenging OOD data (the third row), it is not InD and its features are relatively sparse in the ideal feature set. However, all these features fall in the dark blue area at the same time, which causes the over-confidence problem. Such imbalance breaks the dynamic balance between the dark blue area and the remaining and causes a rapid decline of  $N$  under the perturbation of DDP. Therefore, all two kinds of OOD inp change of confidence.

![](images/f75f47594dea00caa3eba4af4096503362c019f816eba6c174f9e017baef2035.jpg)  
Figure 4: The change in different levels of feature space under the disturbance of DDP.

A following problem is that if an OOD sample cannot be detected, what should it be like? Firstly, it must have some feature points falling in the dark blue area, or  $N$  will increase obviously. Secondly, it should have some additional points falling in the remaining light blue area to keep the dynamic balance. However, the total number of features is finite. Then almost every feature points fall in the ideal feature set and it is just an InD input. In addition, the object in this example can be extended to two different feature spaces. The difference is that the high-level feature space is more than one dimension, but the trend is consistent.

# Algorithm 1 Diffusion-denoising process

Input: Images  $x_0$ , generative interval  $[0, T]$ , generative gap  $\delta$

1: for  $t = T, \dots, \delta$  do  
2:  $x_{t}^{0} = \frac{1}{\sqrt{\bar{\alpha}_{t}}}\big(x_{t} - \sqrt{1 - \bar{\alpha}_{t}}\epsilon_{\theta}(x_{t},t)\big)$  
3:  $\epsilon = \epsilon_{\theta}(x_t,t)$  
4:  $x_{t - \delta} = \sqrt{\bar{\alpha}_{t - \delta}} x_t^0 +\sqrt{1 - \bar{\alpha}_{t - \delta}\epsilon}$  
5: end for  
6: Return  $x_0$

# Algorithm 2 Unconditional neighborhood

Input: Images  $x$

1:  $x^{i} = \mathrm{KNN}(x, \{\mathrm{training data}\})$  
2:  $\epsilon = \Phi (x^i,0,T)$  
3:  $x_{noise} = \sqrt{\bar{\alpha}_t} x + \sqrt{1 - \bar{\alpha}_t\epsilon}$  
4:  $x_{\text{neighbor}} = \Phi(x^i, t, 0)$  
5: Return  $x_{\text{neighbor}}$

# Algorithm 3 Conditional neighborhood

Input: Images  $x$

1:  $y^{i} = \mathrm{FC}(\mathrm{ResNet}(x^{i}))$  
2:  $\epsilon \sim \mathcal{N}(0,1)$  
3:  $x_{noise} = \sqrt{\bar{\alpha}_t} x + \sqrt{1 - \bar{\alpha}_t}\epsilon$  
4:  $x_{\text{neighbor}} = \Phi(x^i, y^i, t, 0)$  
5: Return  $x_{\text{neighbor}}$

# Algorithm 4 OOD detection

Input: Images  $x$ , diffusion-based neighborhood

$x_{\text{neighbor}}$

1: fea = ResNet(x)  
2: fea  $_ { n e i g h b o r } = \operatorname { R e s N e t } ( x _ { n e i g h b o r } )$  
3: indicator = ∑ |fea - fea<sub>neighbor</sub>  
4: if indicator  $> \delta$  then  
5: Return OOD  
6: end if

This example has show the power of DDP, but we still need to take multiple categories into account and avoid obviously category migration for InD input. Here, we need the help of the interpolation property and there exist two choices. First, we can search the kth nearest neighbor of the input in the input space and generate the corresponding noises of them. We interpolate these noises with the original input using DDP. Another more interesting choice is that we can train a conditional diffusion model, and fix the class condition to the class of input<sup>2</sup>. Then all noise are corresponding to the images in the same class. We can interpolate the input with any noise, instead of searching for it first. What's more, we can choose several noises for each input and all the results of DDP become a neighborhood of the input. We call this neighborhood the diffusion-based neighborhood. Then we use a discriminator model to detect the difference in this neighborhood and determine the OOD samples based on that. We put our algorithm in Algorithm 4.

# 3.4 SIMILARITY AND DIFFERENCE

Several baselines are similar to ours in some ways. The first one is KNN (Sun et al., 2022), which does a KNN search in the feature space. However, KNN ignores the possibility that an OOD input may have a similar feature as an InD data, and all methods that only use the final feature have this problem, too. In addition, using KNN in the input space directly is also invalid, because of data sparsity and irrelevant information interference. All generation models can be used to do interpolation. However, the problem with them is that they cannot keep the InD data relatively unchanged, which provides more interference for OOD detection.

Another similar approach is data generation and augmentation. These methods retrain the discriminator, but our generation and discriminator models are trained separately. What's more, existing methods use generation models to generate OOD data (Marek et al., 2021) to help the discriminator models to know the capability boundary. Our methods use the generation models to do interpolation between the new input data and the training data, which do not need to carefully classify the training data and design new loss functions. Some methods use data augmentation to help the training process of discriminator models. Although classical data augmentation can enhance the richness of data and keep the data realistic, some new methods start to add complex and unreal augmentation (Hendrycks et al., 2022), which increases the burden on the models and lacks clear motivation.

# 4 EXPERIMENT

In this section, we first show the detailed setting of our experiments. Then we offer our OOD detection results, including our method and existing representative methods. After that, we provide ablation study results to show the contributions of each item and hyperparameter in our new scores.

Table 1: The AUROC results of different methods. We train the models on the training data for 160k epochs and test the results on the test data. We use the conditional version method and the guidance weight is 2. We set the disturbance degree  $t = 300$ , the repeat size  $r = 4$  and use the logic space as our detection space. The Higher results are better and the bold results are the best in each case.  

<table><tr><td rowspan="2">InD OOD</td><td rowspan="2">cifar100</td><td rowspan="2">tin</td><td rowspan="2">cifar10 svhn</td><td rowspan="2">texture</td><td rowspan="2">place</td><td colspan="2">cifar10</td><td rowspan="2" colspan="2">cifar100</td><td rowspan="2">texture</td><td rowspan="2">place</td><td rowspan="2">avg</td></tr><tr><td>tin</td><td>svhn</td></tr><tr><td>ODIN</td><td>77.76</td><td>79.65</td><td>73.41</td><td>80.76</td><td>82.61</td><td>78.1</td><td>81.33</td><td>70.97</td><td>79.31</td><td>79.76</td><td>78.37</td><td></td></tr><tr><td>EBO</td><td>86.19</td><td>88.61</td><td>88.42</td><td>86.88</td><td>89.62</td><td>79.07</td><td>82.46</td><td>77.81</td><td>77.84</td><td>80.16</td><td>83.71</td><td></td></tr><tr><td>ReAct</td><td>86.37</td><td>88.91</td><td>89.52</td><td>88.19</td><td>90.1</td><td>73.48</td><td>79.63</td><td>84.45</td><td>83.58</td><td>76.94</td><td>84.12</td><td></td></tr><tr><td>MLS</td><td>86.14</td><td>88.53</td><td>88.47</td><td>86.89</td><td>89.5</td><td>79.18</td><td>82.59</td><td>77.68</td><td>77.94</td><td>80.29</td><td>83.72</td><td></td></tr><tr><td>VIM</td><td>87.19</td><td>88.86</td><td>97.28</td><td>96.03</td><td>90.03</td><td>71.54</td><td>78.34</td><td>81.15</td><td>87.41</td><td>75.77</td><td>85.36</td><td></td></tr><tr><td>KNN</td><td>89.62</td><td>91.48</td><td>95.07</td><td>92.84</td><td>91.86</td><td>76.48</td><td>83.33</td><td>82.09</td><td>83.69</td><td>79.03</td><td>86.55</td><td></td></tr><tr><td>G-ODIN</td><td>88.75</td><td>90.7</td><td>98.05</td><td>95.45</td><td>91.86</td><td>72.79</td><td>81.38</td><td>89.85</td><td>89.41</td><td>77.44</td><td>87.57</td><td></td></tr><tr><td>CSI</td><td>87.36</td><td>89.64</td><td>94.52</td><td>89.82</td><td>88.44</td><td>69.43</td><td>72.83</td><td>77.14</td><td>59.38</td><td>69.1</td><td>79.77</td><td></td></tr><tr><td>CutMix</td><td>85.72</td><td>87.99</td><td>90.14</td><td>86.51</td><td>90.28</td><td>78.6</td><td>82.43</td><td>84.05</td><td>77.26</td><td>78.53</td><td>84.15</td><td></td></tr><tr><td>PixMix</td><td>90.62</td><td>92.6</td><td>97.33</td><td>95.8</td><td>92.23</td><td>75.77</td><td>81.86</td><td>93.79</td><td>84.36</td><td>78.88</td><td>88.32</td><td></td></tr><tr><td>Diff (ours)</td><td>90.53</td><td>92.85</td><td>95.09</td><td>93.66</td><td>92.65</td><td>76.43</td><td>84.23</td><td>84.96</td><td>80.64</td><td>78.7</td><td>86.97</td><td></td></tr></table>

# 4.1 SETUP

We evaluate our methods on the most recent OOD detection benchmarks, OpenOOD benchmarks (Yang et al., 2022). We use images from six different datasets, which are filtered to ensure that the in-distribution and the OOD do not have overlapping data. We use the Cifar10 (Krizhevsky et al., 2009) and Cifar100 as InD samples. For the Cifar10 dataset, we use Cifar100, TinyImagenet (Krizhevsky et al., 2017), SVHN (Netzer et al., 2011), Texture and Places365 (Zhou et al., 2017) as OOD data. For the Cifar100 dataset, the OOD datasets are the same, except for swapping Cifar100 for Cifar10 as the OOD dataset. For a fair comparison, we first train discriminator and generation models using the training set. We evaluate the results by calculating the area under the receiver operating characteristic curve (AUROC) Fawcett (2006) between the test set of the InD dataset and the test set of others, to avoid the influence of model overfitting. All images from different datasets are resized into  $32 \times 32$ . The discriminator models are pre-trained ResNet18 from OpenOOD. The diffusion model used in this paper is just the classical model from DDPMs. We use pre-trained unconditional models and train the conditional version by ourselves.

# 4.2 OUT-OF-DISTRIBUTION DETECTION

We put the results in Table 1. We choose eleven representative baselines. The first seven methods do not adjust the discriminator model similar to our method. ODIN (Liang et al., 2018) uses temperature scaling and gradient-based input perturbation. EBO (Liu et al., 2020) uses an energy-based function. ReAct (Sun et al., 2021) uses rectified activation. MLS (Hendrycks et al., 2019) uses maximum logit scores. VIM (Wang et al., 2022) combines the information of feature space and logic space. KNN (Sun et al., 2022) uses the nearest neighbor in the feature space. All these methods are post-hoc methods and we outperform them in all cases of Cifar10 and two cases of Cifar100. We also compare our methods with four methods with additional training on the discriminator model. G-ODIN (Hsu et al., 2020) decomposes the posterior to model the probability of InD.

CSI (Tack et al., 2020) explores the effectiveness of contractive learning objectives. CutMix (Yun et al., 2019) and PixMix (Hendrycks et al., 2022) are two new kinds of data augmentation to improve the capability of models. Our method outperforms them in three cases and gets competitive results in the others.

Our method performs worse than the SOTA methods when the test dataset is SVHN. A performance bottleneck is that, in addition to density estimation, DDP also has a lazy strategy in the denoising process. It tends to keep the smooth area unchanged. In Figure 5, we show this phenomenon using a simple cases. We resize an InD image to  $r \times r$  and then resize it back, which pulls the InD data away. However, the reconstruction error

![](images/727a335c8061c15ec1a2c4512cb0198b2362ccfa413ec6fa8f7194fb5f05b78c.jpg)  
Figure 5: The bad denoising cases for DDP.

![](images/072ef2073335bdb30754e55e4c29ff7d90cea3f312dad1d31579a84376055eb9.jpg)  
(a)

![](images/63b5e4424009b591cd6c211ab06a66d91874d8b1f73c339454128d0864d93859.jpg)  
Figure 7: The AUROC results under different guidance weights, training steps, disturbance degrees and resampling sizes.  
(b)

![](images/921fc281b37d912752850485c872e8018b7f8896a5c56a9445fe9107ba5a130f.jpg)  
(c)

![](images/47efa86a9e607b7cf82182ea2b5f8b932ddee287bcdc49d6d76b1631d130249a.jpg)  
(d)

decrease instead of increase under this operator. This phenomenon also occurs when the input is the relatively simple SVHN dataset.

# 4.3 ABLATION STUDY

Here, we show the influence of each item and hyperparameter on our scores. And all the setting is the same as the main experiments in Table 1 except for the ablation object.

Detection space In Figure 6, we compare the results when we use different detection spaces, including the input image space, the different level feature spaces, and the logit space. We get the best results when we use the high-level feature or the logit. Here, the logit has 10 dimensions and is much smaller than the high-level feature space (512 dimensions). This shows that DDP can reduce information loss successfully. The low-level features and the image space get relatively bad results, the main reason is the information is still redundant at these levels.

![](images/0437343ad79cb373f458acaf910d7f5c53dc741e92b151b993f8716ee3aec9dc.jpg)  
Figure 6: The results using different detection spaces.

Condition In Figure 7a, we compare our the unconditional and conditional methods. The main problem is that Cifar100 has much more classes, which makes separating the feature space become more difficult and unconditional diffusion models cannot keep the interpolation in a single class. We also test different class weights  $\omega$ , we find that a higher class weight can get relatively better results. This shows that realness is more important than diversity in the OOD detection task. In addition, we also find that the conditional version improves the detection results on Cifar100 more obviously, which means the conditional control is important especially when the number of classes is big.

Training In Figure 7b, we find that although the training process of diffusion models is relatively computation-cost to achieve the best FID results, the OOD detection does not need the models to be 100 percent well-trained (200k epoch). After 40k epoch training, we can get relatively good results and the improvement of FID does not help the OOD detection after that.

Timestep In Figure 7c, we determine how to choose the best  $t$  in DDP. We find that the best choice is  $t = 300$  and this is consistent with our examples in Figure 4. When  $t \leq 300$ , the difference caused by DDP is still not obvious enough. When  $t \geq 300$ , the information starts to lose because the noise item accounts for a larger and larger proportion, which limits the OOD detection results.

Resampling In Figure 7d, we determine the influence of the repeated sampling size. According to our analysis, the consistent detection results are maintained by dynamic balance, therefore, we need to resample several times to remove the random error in DDP. We find that more is better and 4 times resampling is good enough.

# 5 DISCUSSION

In this paper, we start with a toy example to show how to combine discriminator and generation models to solve the OOD detection task. Under the setting of the first toy example, such a strategy can perfectly solve the OOD detection problem. Although we cannot say the general cases can also be perfectly solved, we show how to use this idea in the abstract feature space and get competitive results on Cifar10 and Cifar100 using the combination of a ResNet and a diffusion model. Our approach has good interpretability and a solid theoretical background. We believe that this strategy opens a new door to developing more powerful OOD detection methods and has the potential to be applied to OOD generalization and other related tasks.

# REFERENCES

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in ai safety. arXiv preprint arXiv:1606.06565, 2016.  
Jinwon An and Sungzoon Cho. Variational autoencoder based anomaly detection using reconstruction probability. *Special Lecture on IE*, 2(1):1-18, 2015.  
Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981-17993, 2021.  
Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-dpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations, 2022.  
Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.  
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34, 2021.  
Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. Advances in Neural Information Processing Systems, 32, 2019.  
Tom Fawcett. An introduction to roc analysis. Pattern recognition letters, 27(8):861-874, 2006.  
Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pp. 3354-3361. IEEE, 2012.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Peng Guo, Zhiyun Xue, Zac Mtema, Karen Yeates, Ophira Ginsburg, Maria Demarco, L Rodney Long, Mark Schiffman, and Sameer Antani. Ensemble deep learning for cervix image selection toward improving reliability in automated cervical precancer screening. Diagnostics, 10(7):451, 2020.  
Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. arXiv preprint arXiv:1911.11132, 2019.  
Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, and Jacob Steinhardt. Pixmix: Dreamlike pictures comprehensively improve safety measures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16783-16792, 2022.  
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840-6851, 2020.  
Jonathan Ho, Tim Salimans, Alexey A Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In ICLR Workshop on Deep Generative Models for Highly Structured Data, 2022.  
Yen-Chang Hsu, Yilin Shen, Hongxia Jin, and Zsolt Kira. Generalized odin: Detecting out-of-distribution image without learning from out-of-distribution data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10951-10960, 2020.  
Dihong Jiang, Sun Sun, and Yaoliang Yu. Revisiting flow generative models for out-of-distribution detection. In International Conference on Learning Representations, 2022.

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations, 2020.  
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.  
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Communications of the ACM, 60(6):84-90, 2017.  
Max WY Lam, Jun Wang, Dan Su, and Dong Yu. Bddm: Bilateral denoising diffusion models for fast and high-quality speech synthesis. In International Conference on Learning Representations, 2021.  
Siddique Latif, Muhammad Usman, Rajib Rana, and Junaid Qadir. Phonocardiographic sensing using deep learning for abnormal heartbeat detection. IEEE Sensors Journal, 18(22):9393-9400, 2018.  
Shiyu Liang, Yixuan Li, and R Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. In International Conference on Learning Representations, 2018.  
Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds. In International Conference on Learning Representations, 2022.  
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in Neural Information Processing Systems, 33:21464-21475, 2020.  
Petr Marek, Vishal Ishwar Naik, Vincent Auvray, and Anuj Goyal. Oodgan: Generative adversarial network for out-of-domain data generation. arXiv preprint arXiv:2104.02484, 2021.  
Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don't know? arXiv preprint arXiv:1810.09136, 2018.  
Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, and Balaji Lakshminarayanan. Detecting out-of-distribution inputs to deep generative models using typicality. arXiv preprint arXiv:1906.02994, 2019.  
Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. 2011.  
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021.  
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pp. 8821-8831. PMLR, 2021.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.  
Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In International Conference on Learning Representations, 2022.  
Vikash Sehwag, Mung Chiang, and Prateek Mittal. Ssd: A unified framework for self-supervised outlier detection. In International Conference on Learning Representations, 2020.  
Joan Serrà, David Álvarez, Vicenç Gómez, Olga Slizovskaia, José F Núñez, and Jordi Luque. Input complexity and out-of-distribution detection with likelihood-based generative models. In International Conference on Learning Representations, 2019.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a.  
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b.

Kumar Sricharan and Ashok Srivastava. Building robust classifiers through generation of confident out of distribution examples. arXiv preprint arXiv:1812.00239, 2018.  
Yiyou Sun, Chuan Guo, and Yixuan Li. React: Out-of-distribution detection with rectified activations. Advances in Neural Information Processing Systems, 34:144-157, 2021.  
Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. arXiv preprint arXiv:2204.06507, 2022.  
Jihoon Tack, Sangwoo Mo, Jongheon Jeong, and Jinwoo Shin. Csi: Novelty detection via contrastive learning on distributionally shifted instances. Advances in neural information processing systems, 33:11839-11852, 2020.  
Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. Advances in Neural Information Processing Systems, 34, 2021.  
Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. Vim: Out-of-distribution with virtual-logit matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4921-4930, 2022.  
Yang Xin, Lingshuang Kong, Zhi Liu, Yuling Chen, Yanmiao Li, Hongliang Zhu, Mingcheng Gao, Haixia Hou, and Chunhua Wang. Machine learning and deep learning methods for cybersecurity. IEEE access, 6:35365-35381, 2018.  
Jingkang Yang, Pengyun Wang, Dejian Zou, Zitang Zhou, Kunyuan Ding, Wenxuan Peng, Haoqi Wang, Guangyao Chen, Bo Li, Yiyou Sun, et al. Openood: Benchmarking generalized out-of-distribution detection. arXiv preprint, 2022.  
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6023-6032, 2019.  
Yufeng Zhang, Wanwei Liu, Zhenbang Chen, Ji Wang, Zhiming Liu, Kenli Li, and Hongmei Wei. Out-of-distribution detection with distance guarantee in deep generative models. arXiv preprint arXiv:2002.03328, 2020.  
Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6):1452-1464, 2017.
