# UNROLLED GENERATIVE ADVERSARIAL NETWORKS

Luke Metz*

Google Brain

lmetz@google.com

Ben Poole†

Stanford University

poole@cs.stanford.edu

David Pfau

Google DeepMind

pfau@google.com

Jascha Sohl-Dickstein

Google Brain

jaschasd@google.com

# ABSTRACT

We introduce a method to stabilize Generative Adversarial Networks (GANs) by defining the generator objective with respect to an unrolled optimization of the discriminator. This allows training to be adjusted between using the optimal discriminator in the generator's objective, which is ideal but infeasible in practice, and using the current value of the discriminator, which is often unstable and leads to poor solutions. We show how this technique solves the common problem of mode collapse, stabilizes training of GANs with complex recurrent generators, and increases diversity and coverage of the data distribution by the generator.

# 1 INTRODUCTION

The use of deep neural networks as generative models for complex data has made great advances in recent years. This success has been achieved through a surprising diversity of training losses and model architectures, including denoising autoencoders (Vincent et al., 2010), variational autoencoders (Kingma & Welling, 2013; Rezende et al., 2014; Gregor et al., 2015; Kulkarni et al., 2015; Burda et al., 2015; Kingma et al., 2016), generative stochastic networks (Alain et al., 2015), diffusion probabilistic models (Sohl-Dickstein et al., 2015), autoregressive models (Theis & Bethge, 2015; van den Oord et al., 2016a;b), real non-volume preserving transformations (Dinh et al., 2014; 2016), Helmholtz machines (Dayan et al., 1995; Bornschein et al., 2015), and Generative Adversarial Networks (GANs) (Goodfellow et al., 2014).

# 1.1 GENERATIVE ADVERSARIAL NETWORKS

While most deep generative models are trained by maximizing log likelihood or a lower bound on log likelihood, GANs take a radically different approach that does not require inference or explicit calculation of the data likelihood. Instead, two models are used to solve a minimax game: a generator which samples data, and a discriminator which classifies the data as real or generated. In theory these models are capable of modeling an arbitrarily complex probability distribution. When using the optimal discriminator for a given class of generators, the original GAN proposed by Goodfellow et al. minimizes the Jensen-Shannon divergence between the data distribution and the generator, and extensions generalize this to a wider class of divergences (Nowozin et al., 2016; Sonderby et al., 2016).

The ability to train extremely flexible generating functions, without explicitly computing likelihoods or performing inference, and while targeting more mode-seeking divergences has made GANs extremely successful in image generation (Odena et al., 2016; Salimans et al., 2016; Radford et al., 2015), and image super resolution (Ledig et al., 2016). The flexibility of the GAN framework has also enabled a number of successful extensions of the technique, for instance for structured prediction (Reed et al., 2016a;b; Odena et al., 2016), training energy based models (Zhao et al., 2016), and combining the GAN loss with a mutual information loss (Chen et al., 2016).

In practice, however, GANs suffer from many issues, particularly during training. One common failure mode involves the generator collapsing to produce only a single sample or a small family of very similar samples. Another involves the generator and discriminator oscillating during training, rather than converging to a fixed point. In addition, if one agent becomes much more powerful than the other, the learning signal to the other agent becomes useless, and the system does not learn. To train GANs many tricks must be employed, such as careful selection of architectures (Radford et al., 2015), minibatch discrimination (Salimans et al., 2016), and noise injection (Salimans et al., 2016; Sonderby et al., 2016). Even with these tricks the set of hyperparameters for which training is successful is generally very small in practice.

Once converged, the generative models produced by the GAN training procedure normally do not cover the whole distribution (Dumoulin et al., 2016), even when targeting a mode-covering divergence such as KL. Additionally, because it is intractable to compute the GAN training loss, and because approximate measures of performance such as Parzen window estimates suffer from major flaws (Theis et al., 2016), evaluation of GAN performance is challenging. Currently, human judgement of sample quality is one of the leading metrics for evaluating GANs. In practice this metric does not take into account mode dropping if the number of modes is greater than the number of samples one is visualizing. In fact, the mode dropping problem generally helps visual sample quality as the model can choose to focus on only the most common modes. These common modes correspond, by definition, to more typical samples. Additionally, the generative model is able to allocate more expressive power to the modes it does cover than it would if it attempted to cover all modes.

# 1.2 DIFFERENTIATING THROUGH OPTIMIZATION

Many optimization schemes, including SGD, RMSProp (Tieleman & Hinton, 2012), and Adam (Kingma & Ba, 2014), consist of a sequence of differentiable updates to parameters. Gradients can be backpropagated through unrolled optimization updates in a similar fashion to backpropagation through a recurrent neural network. The parameters output by the optimizer can thus be included, in a differentiable way, in another objective (Maclaurin et al., 2015). This idea was first suggested for minimax problems in (Pearlmutter & Siskind, 2008), while (Zhang & Lesser, 2010) provided a theoretical analysis and experimental results on differentiating through a single step of gradient ascent for simple matrix games. Differentiating through unrolled optimization was first scaled to deep networks in (Maclaurin et al., 2015), where it was used for hyperparameter optimization. More recently, (Belanger & McCallum, 2015; Han et al., 2016; Andrychowicz et al., 2016) backpropagate through optimization procedures in contexts unrelated to GANs or minimax games.

In this work we address the challenges of unstable optimization and mode collapse in GANs by unrolling optimization of the discriminator objective during training.

# 2 METHOD

# 2.1 GENERATIVE ADVERSARIAL NETWORKS

The GAN learning problem is to find the optimal parameters  $\theta_G^*$  for a generator function  $G(z;\theta_G)$  in a minimax objective,

$$
\theta_ {G} ^ {*} = \underset {\theta_ {G}} {\operatorname {a r g m i n}} \underset {\theta_ {D}} {\max } f \left(\theta_ {G}, \theta_ {D}\right) \tag {1}
$$

$$
= \underset {\theta_ {G}} {\operatorname {a r g m i n}} f \left(\theta_ {G}, \theta_ {D} ^ {*} \left(\theta_ {G}\right)\right) \tag {2}
$$

$$
\theta_ {D} ^ {*} \left(\theta_ {G}\right) = \underset {\theta_ {D}} {\operatorname {a r g m a x}} f \left(\theta_ {G}, \theta_ {D}\right), \tag {3}
$$

where  $f$  is commonly chosen to be

$$
f \left(\theta_ {G}, \theta_ {D}\right) = \mathbb {E} _ {x \sim p _ {d a t a}} \left[ \log \left(D \left(x; \theta_ {D}\right)\right) \right] + \mathbb {E} _ {z \sim \mathcal {N} (0, I)} \left[ \log \left(1 - D \left(G \left(z; \theta_ {G}\right); \theta_ {D}\right)\right) \right]. \tag {4}
$$

Here  $x \in \mathcal{X}$  is the data variable,  $z \in \mathcal{Z}$  is the latent variable,  $p_{data}$  is the data distribution, the discriminator  $D(\cdot; \theta_D): \mathcal{X} \to [0,1]$  outputs the estimated probability that a sample  $x$  comes from the data distribution,  $\theta_D$  are the discriminator's parameters, and the generator function  $G(\cdot; \theta_G): \mathcal{Z} \to \mathcal{X}$  transforms a sample in the latent space into a sample in the data space.

For the minimax loss in Eq. 4, the optimal discriminator  $D^{*}(x)$  is a known smooth function of the generator probability  $p_{G}(x)$  (Goodfellow et al., 2014),

$$
D ^ {*} (x) = \frac {p _ {\text {d a t a}} (x)}{p _ {\text {d a t a}} (x) + p _ {G} (x)}. \tag {5}
$$

When the generator loss in Eq. 2 is rewritten directly in terms of  $p_G(x)$  and Eq. 5 rather than  $\theta_G$  and  $\theta_D^* (\theta_G)$ , then it is similarly a smooth function of  $p_G(x)$ . These smoothness guarantees are typically lost when  $D(x; \theta_D)$  and  $G(z; \theta_G)$  are drawn from parametric families. They nonetheless suggest that the true generator objective in Eq. 2 will often be well behaved, and is a desirable target for direct optimization.

Explicitly solving for the optimal discriminator parameters  $\theta_D^* (\theta_G)$  for every update step of the generator  $G$  is computationally infeasible for discriminators based on neural networks. Therefore this minimax optimization problem is typically solved by alternating gradient descent on  $\theta_{G}$  and ascent on  $\theta_{D}$ .

The optimal solution  $\theta^{*} = \{\theta_{G}^{*},\theta_{D}^{*}\}$  is a fixed point of these iterative learning dynamics. Additionally, if  $f(\theta_G,\theta_D)$  is convex in  $\theta_{G}$  and concave in  $\theta_{D}$ , then alternating gradient descent (ascent) trust region updates are guaranteed to converge to the fixed point, under certain additional weak assumptions (Juditsky et al., 2011). However in practice  $f(\theta_G,\theta_D)$  is typically very far from convex in  $\theta_{G}$  and  $\theta_{D}$ , and updates are not constrained in an appropriate way. As a result GAN training suffers from mode collapse, undamped oscillations, and other problems detailed in Section 1.1. In order to address these difficulties, we will introduce a surrogate objective function  $f_{K}(\theta_{G},\theta_{D})$  for training the generator which more closely resembles the true generator objective  $f(\theta_G,\theta_D^* (\theta_G))$ .

# 2.2 UNROLLING GANS

A local optimum of the discriminator parameters  $\theta_D^*$  can be expressed as the fixed point of an iterative optimization procedure,

$$
\theta_ {D} ^ {0} = \theta_ {D} \tag {6}
$$

$$
\theta_ {D} ^ {k + 1} = \theta_ {D} ^ {k} + \eta^ {k} \frac {\mathrm {d} f \left(\theta_ {G} , \theta_ {D} ^ {k}\right)}{\mathrm {d} \theta_ {D} ^ {k}} \tag {7}
$$

$$
\theta_ {D} ^ {*} = \lim  _ {k \rightarrow \infty} \theta_ {D} ^ {k}, \tag {8}
$$

where  $\eta^k$  is the learning rate schedule. For simplicity, we have expressed Eq. 7 as a full batch steepest gradient ascent equation. More sophisticated optimizers can be similarly unrolled. In our experiments we unroll Adam (Kingma & Ba, 2014).

By unrolling for  $K$  steps, we create a surrogate objective for the update of the generator,

$$
f _ {K} \left(\theta_ {G}, \theta_ {D}\right) = f \left(\theta_ {G}, \theta_ {D} ^ {K} \left(\theta_ {G}, \theta_ {D}\right)\right). \tag {9}
$$

When  $K = 0$  this objective corresponds exactly to the standard GAN objective, while as  $K \to \infty$  it corresponds to the true generator objective function  $f(\theta_G, \theta_D^*(G))$ . By adjusting the number of unrolling steps  $K$ , we are thus able to interpolate between standard GAN training dynamics with their associated pathologies, and more costly gradient descent on the true generator loss.

# 2.3 PARAMETER UPDATES

The generator and discriminator parameter updates using this surrogate loss are

$$
\theta_ {G} \leftarrow \theta_ {G} - \eta \frac {\mathrm {d} f _ {K} \left(\theta_ {G} , \theta_ {D}\right)}{\mathrm {d} \theta_ {G}} \tag {10}
$$

$$
\theta_ {D} \leftarrow \theta_ {D} + \eta \frac {\mathrm {d} f \left(\theta_ {G} , \theta_ {D}\right)}{\mathrm {d} \theta_ {D}}. \tag {11}
$$

For clarity we describe full batch steepest gradient descent (ascent) with stepsize  $\eta$  above, while in experiments we instead use minibatch Adam for both updates. The gradient in Eq. 10 requires backpropagating through the optimization process in Eq. 7. A clear description of differentiation

through gradient descent is given as Algorithm 2 in (Maclaurin et al., 2015), though in practice the use of an automatic differentiation package means this step does not need to be programmed explicitly.

It is important to distinguish this from an approach suggested in (Goodfellow et al., 2014), that several update steps of the discriminator parameters should be run before each single update step for the generator. In that approach, the update steps for both models are still gradient descent (ascent) with respect to fixed values of the other model parameters, rather than the surrogate loss we describe in Eq. 9.

Performing  $K$  steps of discriminator update between each single step of generator update corresponds to updating the generator parameters  $\theta_{G}$  using only the first term in Eq. 12 below. In principle, the two approaches could be used in combination, updating the generator with the surrogate loss, the running several update steps for the discriminator.

# 2.4 THE MISSING GRADIENT TERM

To better understand the behavior of the surrogate loss  $f_{K}(\theta_{G},\theta_{D})$ , we examine its gradient with respect to the generator parameters  $\theta_{G}$

$$
\frac {\mathrm {d} f _ {K} \left(\theta_ {G} , \theta_ {D}\right)}{\mathrm {d} \theta_ {G}} = \frac {\partial f \left(\theta_ {G} , \theta_ {D} ^ {K} \left(\theta_ {G} , \theta_ {D}\right)\right)}{\partial \theta_ {G}} + \frac {\partial f \left(\theta_ {G} , \theta_ {D} ^ {K} \left(\theta_ {G} , \theta_ {D}\right)\right)}{\partial \theta_ {D} ^ {K} \left(\theta_ {G} , \theta_ {D}\right)} \frac {\mathrm {d} \theta_ {D} ^ {K} \left(\theta_ {G} , \theta_ {D}\right)}{\mathrm {d} \theta_ {G}}. \tag {12}
$$

Standard GAN training corresponds exactly to updating the generator parameters using only the first term in this gradient, with  $\theta_D^K (\theta_G,\theta_D)$  being the parameters resulting from the discriminator update step. An optimal generator for any fixed discriminator is a delta function at the  $x$  to which the discriminator assigns highest data probability. Therefore, in standard GAN training, each generator update step is a partial collapse towards a delta function.

The second term captures how the discriminator would react to a change in the generator. It reduces the tendency of the generator to engage in mode collapse. For instance, the second term reflects that as the generator collapses towards a delta function, the discriminator reacts and assigns lower probability to that state, increasing the generator loss. It therefore discourages the generator from collapsing, and may improve stability.

As  $K \to \infty$ ,  $\theta_D^K$  goes to a local optimum of  $f$ , where  $\frac{\partial f}{\partial \theta_D^K} = 0$ , and therefore the second term in Eq. 12 goes to 0 (Danskin, 2012). The gradient of the unrolled surrogate loss  $f_K(\theta_G, \theta_D)$  with respect to  $\theta_G$  is thus identical to the gradient of the standard GAN loss  $f(\theta_G, \theta_D)$  both when  $K = 0$  and when  $K \to \infty$ , where we take  $K \to \infty$  to imply that in the standard GAN the discriminator is also fully optimized between each generator update. Between these two extremes,  $f_K(\theta_G, \theta_D)$  captures additional information about the response of the discriminator to changes in the generator.

# 2.5 CONSEQUENCES OF THE SURROGATE LOSS

GANs can be thought of as a game between the discriminator  $(D)$  and the generator  $(G)$ . The agents take turns taking actions and updating their parameters until a Nash equilibrium is reached. The optimal action for  $D$  is to evaluate the probability ratio  $\frac{p_{data}(x)}{p_G(x) + p_{data}(x)}$  for the generator's move  $x$  (Eq. 5). The optimal generator action is to move its mass to maximize this ratio.

The initial move for  $G$  will be to move as much mass as its parametric family and update step permits to the single point that maximizes the ratio of probability densities. The action  $D$  will then take is quite simple. It will track that point, and to the extent allowed by its own parametric family and update step assign low data probability to it, and uniform probability everywhere else. This cycle of  $G$  moving and  $D$  following will repeat forever or converge depending on the rate of change of the two agents. This is similar to the situation in simple matrix games like rock-paper-scissors and matching pennies, where alternating gradient descent (ascent) with a fixed learning rate is known not to converge (Singh et al., 2000; Bowling & Veloso, 2002).

In the unrolled case, however, this undesirable behavior no longer occurs. Now  $G$ 's actions take into account how  $D$  will respond. In particular,  $G$  will try to make steps that  $D$  will have a hard time

![](images/67d0eeeac9a490d17ed03c74110a32ed3e4ed5bb32a69d6f48bf9aba9e4a841e.jpg)  
Figure 1: Unrolling the discriminator stabilizes GAN training on a toy 2D mixture of Gaussians dataset. Columns show a heatmap of the generator distribution after increasing numbers of training steps. The final column shows the data distribution. The top row shows training for a GAN with 10 unrolling steps. Its generator quickly spreads out and converges to the target distribution. The bottom row shows standard GAN training. The generator rotates through the modes of the data distribution. It never converges to a fixed distribution, and only ever assigns significant mass to a single data mode at once.

responding to. This extra information helps the generator spread its mass to make the next  $D$  step less effective instead of collapsing to a point.

In principle, a surrogate loss function could be used for both  $D$  and  $G$ . In the case of 1-step unrolled optimization this is known to lead to convergence for games in which gradient descent (ascent) fails (Zhang & Lesser, 2010). However, the motivation for using the surrogate generator loss in Section 2.2, of unrolling the inner of two nested min and max functions, does not apply to using a surrogate discriminator loss. Additionally, it is more common for the discriminator to overpower the generator than vice-versa when training a GAN. Giving more information to  $G$  by allowing it to 'see into the future' may thus help the two models be more balanced.

# 3 EXPERIMENTS

In this section we demonstrate improved mode coverage and stability by applying this technique to three datasets of increasing complexity. Evaluation of generative models is a notoriously hard problem (Theis et al., 2016). As such the de facto standard in GAN literature has become sample quality as evaluated by a human and/or evaluated by a heuristic (Inception score for example, (Salimans et al., 2016)). While these evaluation metrics do a reasonable job capturing sample quality, they fail to capture sample diversity. In our first 2 experiments diversity is easily evaluated via visual inspection. In our last experiment this is not the case, and we will introduce new methods to quantify coverage of samples.

When doing stochastic optimization, we must choose which minibatches to use in the unrolling updates in Eq. 7. We experimented with both a fixed minibatch and re-sampled minibatches for each unrolling step, and found it did not significantly impact the result. We use fixed minibatches for all experiments in this section.

# 3.1 MIXTURE OF GAUSSIANS DATASET

To illustrate the impact of discriminator unrolling, we train a simple GAN architecture on a 2D mixture of 8 Gaussians arranged in a circle. For a detailed list of architecture and hyperparameters see Appendix A. Figure 1 shows the dynamics of this model through time. Without unrolling the generator rotates around the valid modes of the data distribution but is never able to spread out mass. When adding in unrolling steps G quickly learns to spread probability mass and the system converges to the data distribution.

# 3.2 PATHOLOGICAL MODELS

To evaluate the ability of this approach to improve trainability, we look to a traditionally challenging family of models to train – recurrent neural networks (RNN). In this experiment we try to generate MNIST samples using an LSTM (Hochreiter & Schmidhuber, 1997). MNIST digits are  $28 \times 28$  pixel

![](images/3b8ad4ac671b6d0d03b0ba6b273593354703b90414a1ae0297b9fd8576b7b588.jpg)  
Figure 2: Unrolled GAN training increases stability for an RNN generator and convolutional discriminator trained on MNIST. The top row was run with 20 unrolling steps. The bottom row is a standard GAN, with 0 unrolling steps. Images are samples from the generator after the indicated number of training steps.

images. At each timestep of the generator LSTM, it outputs one column of this image, so that after 28 timesteps it has output the entire sample. We use a convolutional neural network as the discriminator. See Appendix B for the full model and training details. Unlike in all previously successful GAN models, there is no symmetry between the generator and the discriminator in this task, resulting in a more complex power balance. Results can be seen in Figure 2. Once again, without unrolling the model quickly collapses to a single mode and rotates around the data distribution. Instead of rotating spatially, it cycles through proto-digit like blobs. When running with unrolling steps the generator disperses and appears to cover the whole data distribution, as in the 2D example.

# 3.3 IMAGE MODELING

Finally we test our technique on a more traditional convolutional GAN architecture and task, similar to those used in (Radford et al., 2015; Salimans et al., 2016). In the previous experiments we tested models where the standard GAN training algorithm would not converge. In this section we improve a standard model by reducing its tendency to engage in mode collapse. We ran 4 configurations of this model, varying the number of unrolling steps to be 0, 1, 5, or 10. Each configuration was run 5 times with different random seeds. For full training details see Appendix C. Samples from each of the 4 configurations can be found in Figure 3. There is no obvious difference in visual quality across these model configurations. Visual inspection however provides only a poor measure of sample diversity.

By training with an unrolled discriminator, we expect to generate more diverse samples which more closely resemble the underlying data distribution. We introduce two techniques to examine sample diversity: inference via optimization, and pairwise distance distributions.

# 3.3.1 INFERENCE VIA OPTIMIZATION

Since likelihood cannot be tractably computed, over-fitting of GANs is typically tested by taking samples and computing the nearest-neighbor images in pixel space from the training data (Goodfellow et al., 2014). We will do the reverse, and measure the ability of the generative model to generate images that look like specific samples from the training data. If we did this by generating random samples from the model, we would need an exponentially large number of samples. We instead treat

![](images/2d80f3419ca7fa1951b47ddee8c38dfa95b007bbe7ec56889f02696b65b5cfd2.jpg)  
Figure 3: Visual perception of sample quality and diversity is very similar for models trained with different numbers of unrolling steps. Actual sample diversity is higher with more unrolling steps. Each pane shows samples generated after training a model on CIFAR10 with 0, 1, 5, and 10 steps of unrolling.

finding the nearest neighbor  $x_{\mathrm{nearest}}$  to a target image  $x_{\mathrm{target}}$  as an optimization task,

$$
z _ {\text {n e a r e s t}} = \underset {z} {\operatorname {a r g m i n}} \| G (z; \theta_ {G}) - x _ {\text {t a r g e t}} \| _ {2} ^ {2} \tag {13}
$$

$$
x _ {\text {n e a r e s t}} = G \left(z _ {\text {n e a r e s t}}; \theta_ {G}\right). \tag {14}
$$

This concept of backpropagating to generate images has been widely used in visualizing features from discriminative networks (Simonyan et al., 2013; Yosinski et al., 2015; Nguyen et al., 2016) and has been applied to explore the visual manifold of GANs in (Zhu et al., 2016).

We apply this technique to each of the models trained. We optimize with 3 random starts using LBFGS, which is the optimizer typically used in similar settings such as style transfer (Johnson et al., 2016; Champandard, 2016). Results comparing average mean squared errors between  $x_{\mathrm{nearest}}$  and  $x_{\mathrm{target}}$  in pixel space can be found in Table 1. In addition we compute the percent of images for which a certain configuration achieves the lowest loss when compared to the other configurations. In the zero step case, there is poor reconstruction and less than  $1\%$  of the time does it obtain the lowest error of the 4 configurations. This shows that for almost all images in the training set unrolling improves the ability of the GAN to generate similar samples, as judged by an MSE metric. Taking 1 unrolling step results in a significant improvement in MSE. Taking 10 unrolling steps results in more modest improvement, but continues to reduce the reconstruction MSE.

To visually see this, we compare the result of optimization process for 0, 1, 5, and 10 step configurations in Figure 4. To select for images where differences in behavior is most apparent, we sort the data by the absolute value of a fractional difference in MSE between the 0 and 10 step models,  $\left|\frac{l_{0\text{step}} - l_{10\text{step}}}{\frac{1}{2}(l_{0\text{step}} + l_{10\text{step}})}\right|$ . This highlights examples where either the 0 or 10 step model cannot accurately fit the data example but the other can. Images in all cases are generated by the first model in each duplicated configuration. See Appendix D for comparisons of the remaining models. Many of the zero step images are fuzzy and ill-defined suggesting that these images cannot be generated by the standard GAN generative model, and come from a dropped mode. As more unrolling steps are added, the outlines become more clearer and well defined – the model covers more of the distribution and thus can recreate these samples.

# 3.3.2 PAIRWISE DISTANCES

A second complementary approach is to compare statistics of data samples to the corresponding statistics for samples generated by the various models. One particularly simple and relevant statistic is the distribution over pairwise distances between random pairs of samples. In the case of mode collapse, greater probability mass will be concentrated in smaller volumes, and the distribution

<table><tr><td>Unrolling Steps</td><td>0 steps</td><td>1 step</td><td>5 steps</td><td>10 steps</td></tr><tr><td>Average MSE</td><td>0.0231 ± 0.0024</td><td>0.0195 ± 0.0021</td><td>0.0200 ± 0.0023</td><td>0.0181 ± 0.0018</td></tr><tr><td>Percent Best Rank</td><td>0.63%</td><td>22.97%</td><td>15.31%</td><td>61.09%</td></tr></table>

Table 1: GANs trained with unrolling are better able to match images in the training set than standard GANs, likely due to mode dropping by the standard GAN. Results show the MSE between training images and the best reconstruction for a model with the given number of unrolling steps. The fraction of training images best reconstructed by a given model is given in the final column. The best reconstructions is found by optimizing the latent representation  $z$  to produce the closest matching pixel output  $G(z; \theta_G)$ . Results are averaged over all 5 runs of each model with different random seeds.

![](images/2ff8b1ec4baa76b7c1b64b69ccaa018e490c71d023c021d055f25dc88cbb7767.jpg)  
Figure 4: Training set images are more accurately reconstructed using GANs trained with unrolling than by a standard (0 step) GAN, likely due to mode dropping by the standard GAN. Raw data is on the left, and the optimized images to reach this target follow for 0, 1, 5, and 10 unrolling steps. The reconstruction MSE is listed below each sample. A random 1280 images where selected from the training set, and corresponding best reconstructions for each model were found via optimization. Shown here are the ten images with the largest absolute fractional difference between GANs trained with 0 and 10 unrolling steps.

![](images/bda11f715b0e0fcbcd7331c173e1e668e2fe203aa0f9f025cb98d48ba7b0e596.jpg)

over inter-sample distances should be skewed towards smaller distances. We sample random pairs of images from each model, as well as from the training data, and compute histograms of the  $\ell_2$  distances between those sample pairs. As illustrated in Figure 5, the standard GAN, with zero unrolling steps, has its probability mass skewed towards smaller  $\ell_2$  intersample distances, compared to real data. As the number of unrolling steps is increased, the histograms over intersample distances increasingly come to resemble that for the data distribution. This is further evidence in support of unrolling decreasing the mode collapse behavior of GANs.

# 4 DISCUSSION

In this work we developed a method to stabilize GAN training and reduce mode collapse by defining the generator objective with respect to unrolled optimization of the discriminator. We then demonstrated the application of this method to several tasks, where it either rescued unstable training, or reduced the tendency of the model to drop regions of the data distribution.

The main drawback to this method is computational cost of each training step, which increases linearly with the number of unrolling steps. There is a tradeoff between better approximating the true generator loss and the computation required to make this estimate. Depending on the architecture, one unrolling step can be enough. In other more unstable models, such as the RNN case, more are

![](images/b01973d291213c22d5ec4bbfd4a41ddb38ade2bd3fb116ab284f0d810a8d7df0.jpg)  
Figure 5: As the number of unrolling steps in GAN training is increased, the distribution of pairwise distances between model samples more closely resembles the same distribution for the data. Here we plot histograms of pairwise distances between randomly selected samples. The red line gives pairwise distances in the data, while each of the five blue lines in each plot represents a model trained with a different random seed. The vertical lines are the medians of each distribution.

needed to stabilize training. We have some initial positive results suggesting it may be sufficient to further perturb the training gradient in the same direction that a single unrolling step perturbs it. While this is more computationally efficient, further investigation is required.

The method presented here bridges some of the gap between theoretical and practical results for training of GANs. We believe developing better update rules for the generator and discriminator is an important line of work for GAN training. In this work we have only considered a small fraction of the design space. For instance, the approach could be extended to unroll  $G$  when updating  $D$  as well - letting the discriminator react to how the generator would move. It is also possible to unroll sequences of  $G$  and  $D$  updates. This would make updates that are recursive:  $G$  could react to maximize performance as if  $G$  and  $D$  had already updated.

# ACKNOWLEDGMENTS

We would like to thank Laurent Dinh, David Dohan, Vincent Dumoulin, Liam Fedus, Ishaan Gulrajani, Julian Ibarz, Eric Jang, Matthew Johnson, Marc Lanctot, Augustus Odena, Gabriel Pereyra, Colin Raffel, Sam Schoenholz, Jon Shlens, and Dale Schuurmans for insightful conversation, as well as the rest of the Google Brain Team.

# REFERENCES

Guillaume Alain, Yoshua Bengio, Li Yao, Jason Yosinski, Eric Thibodeau-Laufer, Saizheng Zhang, and Pascal Vincent. Gsns: Generative stochastic networks. arXiv preprint arXiv:1503.05571, 2015.  
Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, and Nando de Freitas. Learning to learn by gradient descent by gradient descent. arXiv preprint arXiv:1606.04474, 2016.  
David Belanger and Andrew McCallum. Structured prediction energy networks. arXiv preprint arXiv:1511.06350, 2015.  
Jorg Bornschein, Samira Shabanian, Asja Fischer, and Yoshua Bengio. Bidirectional helmholtz machines. arXiv preprint arXiv:1506.03877, 2015.  
Michael Bowling and Manuela Veloso. Multiagent learning using a variable learning rate. Artificial Intelligence, 136(2):215-250, 2002.  
Yuri Burda, Roger B. Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.  
Alex J. Champandard. Semantic style transfer and turning two-bit doodles into fine artworks. arXiv preprint arXiv:1603.01768, 2016.  
Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Info-gan: Interpretable representation learning by information maximizing generative adversarial nets. arXiv preprint arXiv:1606.03657, 2016.  
John M Danskin. The theory of max-min and its application to weapons allocation problems, volume 5. Springer Science & Business Media, 2012.  
Peter Dayan, Geoffrey E Hinton, Radford M Neal, and Richard S Zemel. The helmholtz machine. Neural computation, 7(5):889-904, 1995.  
Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.  
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. arXiv preprint arXiv:1605.08803, 2016.  
Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Alex Lamb, Martin Arjovsky, Olivier Mastropietro, and Aaron Courville. Adversarily learned inference. arXiv preprint arXiv:1606.00704, 2016.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In JMLR W&CP: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2010), volume 9, pp. 249-256, May 2010.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger (eds.), Advances in Neural Information Processing Systems 27, pp. 2672-2680. Curran Associates, Inc., 2014. URL http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf.  
Karol Gregor, Ivo Danihelka, Alex Graves, and Daan Wierstra. DRAW: A recurrent neural network for image generation. In Proceedings of The 32nd International Conference on Machine Learning, pp. 1462-1471, 2015. URL http://www.jmlr.org/proceedings/papers/v37/gregor15.html.  
Tian Han, Yang Lu, Song-Chun Zhu, and Ying Nian Wu. Alternating back-propagation for generator network, 2016. URL https://arxiv.org/abs/1606.08571.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural Comput., 9(8):1735-1780, November 1997. ISSN 0899-7667. doi: 10.1162/neco.1997.9.8.1735. URL http://dx.doi.org/10.1162/neco.1997.9.8.1735.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, pp. 448-456, 2015. URL http://jmlr.org/proceedings/papers/v37/ioffe15.html.  
Justin Johnson, Alexandre Alahi, and Fei-Fei Li. Perceptual losses for real-time style transfer and super-resolution. arXiv preprint arXiv:1603.08155, 2016.  
Anatoli Juditsky, Arkadi Nemirovski, et al. First order methods for nonsmooth convex large-scale optimization, i: general purpose methods. Optimization for Machine Learning, pp. 121-148, 2011.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2013. URL https://arxiv.org/abs/1312.6114.  
Diederik P. Kingma, Tim Salimans, and Max Welling. Improving variational inference with inverse autoregressive flow. 2016.  
Tejas D. Kulkarni, Will Whitney, Pushmeet Kohli, and Joshua B. Tenenbaum. Deep convolutional inverse graphics network. arXiv preprint arXiv:1503.03167, 2015.  
Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi. Photo-realistic single image super-resolution using a generative adversarial network, 2016. URL https://arxiv.org/abs/1609.04802.  
Dougal Maclaurin, David Duvenaud, and Ryan P. Adams. Gradient-based hyperparameter optimization through reversible learning, 2015.  
Anh Nguyen, Alexey Dosovitskiy, Jason Yosinski, Thomas Brox, and Jeff Clune. Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. arXiv preprint arXiv:1605.09304, 2016.  
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization. arXiv preprint arXiv:1606.00709, 2016.  
Augustus Odena, Christopher Olah, and Jonathon Shlens. Conditional image synthesis with auxiliary classifier gans. arXiv preprint arXiv:1610.09585, 2016.  
Barak A. Pearlmutter and Jeffrey Mark Siskind. Reverse-mode ad in a functional framework: Lambda the ultimate backpropagator. ACM Trans. Program. Lang. Syst., 30(2):7:1-7:36, March 2008. ISSN 0164-0925. doi: 10.1145/1330017.1330018. URL http://doi.acm.org/10.1145/1330017.1330018.  
Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015.  
Scott Reed, Zeynep Akata, Santosh Mohan, Samuel Tenka, Bernt Schiele, and Honglak Lee. Learning what and where to draw. In NIPS, 2016a.  
Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text-to-image synthesis. In Proceedings of The 33rd International Conference on Machine Learning, 2016b.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and variational inference in deep latent gaussian models. In International Conference on Machine Learning. Citeseer, 2014.

Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. arXiv preprint arXiv:1606.03498, 2016.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.  
Satinder Singh, Michael Kearns, and Yishay Mansour. Nash convergence of gradient dynamics in general-sum games. In Proceedings of the Sixteenth conference on Uncertainty in artificial intelligence, pp. 541-548. Morgan Kaufmann Publishers Inc., 2000.  
Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of The 32nd International Conference on Machine Learning, pp. 2256-2265, 2015. URL http://arxiv.org/abs/1503.03585.  
Casper Kaeae Sonderby, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Huszar. Amortised map inference for image super-resolution, 2016. URL https://arxiv.org/abs/1610.04490v1.  
L. Theis and M. Bethge. Generative image modeling using spatial lstms. In Advances in Neural Information Processing Systems 28, Dec 2015. URL http://arxiv.org/abs/1506.03478/.  
L. Theis, A. van den Oord, and M. Bethge. A note on the evaluation of generative models. In International Conference on Learning Representations, Apr 2016. URL http://arxiv.org/abs/1511.01844.  
T. Tieleman and G. Hinton. Lecture 6.5—RmsProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012.  
Aäron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, abs/1601.06759, 2016a. URL http://arxiv.org/abs/1601.06759.  
Aäron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelCNN decoders. arXiv preprint arXiv:1606.05328, 2016b.  
Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and Pierre-Antoine Manzagol. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res., 11:3371-3408, December 2010. ISSN 1532-4435. URL http://dl.acm.org/citation.cfm?id=1756006.1953039.  
Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization. arXiv preprint arXiv:1506.06579, 2015.  
Chongjie Zhang and Victor R Lesser. Multi-agent learning with policy prediction. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence, 2010.  
Junbo Zhao, Michael Mathieu, and Yann LeCun. Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126, 2016.  
Jun-Yan Zhu, Philipp Krahenbuhl, Eli Shechtman, and Alexei A. Efros. Generative visual manipulation on the natural image manifold. In Proceedings of European Conference on Computer Vision (ECCV), 2016.
