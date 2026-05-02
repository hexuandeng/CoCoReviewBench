# THE CONDITIONAL ENTROPY BOTTLENECK

Anonymous authors

Paper under double-blind review

# ABSTRACT

We present a new family of objective functions, which we term the Conditional Entropy Bottleneck (CEB). We demonstrate the application of CEB to classification tasks. In our experiments, CEB gives: well-calibrated predictions; essentially perfect detection of challenging out-of-distribution examples and powerful whitebox adversarial examples; and robustness to the same. Finally, we report that CEB fails to learn a dataset with fixed random labels, providing a possible resolution to the problem of generalization observed in Zhang et al. (2016).

# 1 INTRODUCTION

The field of Machine Learning has suffered from the following well-known problems in recent years<sup>1</sup>:

- Vulnerability to adversarial examples. Essentially all machine-learned systems are currently believed by default to be highly vulnerable to adversarial examples. Many defenses have been proposed, but very few have demonstrated robustness against a powerful, general-purpose adversary. Lacking a clear theoretical framework for adversarial attacks, most proposed defenses are ad-hoc and fail in the presence of a concerted attacker (Carlini & Wagner, 2017a; Athalye et al., 2018).  
- Poor out-of-distribution detection. Classifiers do a poor job of signaling that they have received data that is substantially different from the data they were trained on. Ideally, a trained classifier would give less confident predictions for data that was far from the training distribution (as well as for adversarial examples). Barring that, there would be a clear, principled statistic that could be extracted from the model to tell whether the model should have made a low-confidence prediction. Many different approaches to providing such a statistic have been proposed (Guo et al., 2017; Lakshminarayanan et al., 2016; Hendrycks & Gimpel, 2016; Liang et al., 2017; Lee et al., 2017; DeVries & Taylor, 2018), but most seem to do poorly on what humans intuitively view as obviously different data.  
- Miscalibrated predictions. Related to the issues above, classifiers tend to be very overconfident in their predictions (Guo et al., 2017). This may be a symptom, rather than a cause, but miscalibration does not give practitioners confidence in their models.  
- Overfitting to the training data. Zhang et al. (2016) demonstrated that classifiers can memorize fixed random labelings of training data, which means that it is possible (or even easy) to learn a classifier with perfect inability to generalize. This critical observation makes it clear that a fundamental test of generalization is that the model should fail to learn when given fixed random labels.

This paper does not set out to solve any of these problems. Instead, our sole interest is the learning of optimal representations. In pursuit of that goal, we attempt to be as general as possible, considering only how to define optimal representations, what objective function might be capable of learning them, and what requirements such an objective function places on the form of the model.

Given an optimal objective function, however, it is natural to explore the problems listed above, to see if such an objective function can ameliorate some of the core issues in the field of machine learning. We make those explorations in this paper, and find that our objective function, the Conditional Entropy Bottleneck (CEB) appears to impact all of the issues listed above.

![](images/1a6a3c7ac3f26e8b1538e1ccb413f411b8c18ba299c9c767c6643be587ee6369.jpg)  
Figure 1: (Left): Information Venn diagram showing the joint distribution over  $X, Y$ . (Right): The joint distribution  $Z_{X} \gets X \leftrightarrow Y$ .  $Z_{X}$  is carefully positioned to indicate its conditional independence from  $Y$  given  $X$ .

![](images/83296bc56fa9dd3db0840944b0c95bf3125b3721364aa02f267c93dd4c056e10.jpg)

# 2 OPTIMAL REPRESENTATIONS

In the following discussion, we will take some liberty and conflate the scalar values of the information theoretic functionals, the entropy and the mutual information, with the underlying set-theoretic elements that those functionals measure. See Reza (1994) for a discussion of the set-theoretic analogies to information theory. When we talk about "covering" information quantities, we have these set-theoretic elements in mind.

Consider a joint distribution,  $p(x,y)$ , represented by the graphical model:

$$
X \leftrightarrow Y
$$

This joint distribution is our data, and may take any form. We don't presume to know how the data factors. It may factor as  $p(x,y) = p(x)p(y|x)$ ,  $p(x,y) = p(y)p(x|y)$ , or even  $p(x,y) = p(x)p(y)$ .

The first two factorings are depicted in Figure 1 in a standard information diagram showing the various entropies and the mutual information. We can ask: given this generic setting, what is the optimal representation? It seems there are only two options worth considering:  $H(X,Y)$  and  $I(X;Y)$ .

The field of lossless compression is concerned with representations that perfectly maintain  $H(X, Y)$ , as are the closely related studies of Kolmogorov Complexity (Kolmogorov, 1965) and Minimum Description Length (MDL) (Grünwald, 2007), all three of which are concerned with perfect reconstruction of inputs or messages.

In contrast, we think that the field of machine learning is primarily concerned with optimal generalization to unseen data. All of these fields recognize the importance of minimality, but the requirements of perfect reconstruction necessarily result in the retention of much more information in the model than may be needed for prediction or stochastic generation tasks. For most such machine learning tasks, this leaves only the representation corresponding to mutual information between  $X$  and  $Y$ .

The mutual information is defined in a variety of ways; we will use two (Cover & Thomas, 2006):

$$
I (X; Y) = H (X) - H (X | Y) = H (Y) - H (Y | X) \tag {1}
$$

$I(X;Y)$  is the unique, minimal representation defined by a dataset. The selection of a particular dataset entails this representation - all information not covered by  $I(X;Y)$  is superfluous. For example, consider a labeled dataset, where  $X$  is high-dimensional and information-rich, and  $Y$  is a single integer. All of the information in  $X$  that is not needed to correctly predict the single value  $Y = y$  is useless for the task defined by the dataset, and may actually be harmful to the performance of a machine learning system if retained in the learned representation, as we will show below.

In Appendix A, we define the Minimal Necessary Information (MNI) criterion for determining the optimality of a representation. In the two-variable case  $(X,Y)$ , this corresponds exactly to the mutual information,  $I(X;Y)$ . Thus, we take the position that an optimal representation  $Z$  for two observed variables,  $X,Y$ , must "cover"  $I(X;Y)$  in some sense. We show how to do this next.

# 3 THE CONDITIONAL ENTROPY BOTTLENECK

Given the Minimum Necessary Information criterion, we would like to find a way to learn  $I(X,Y)$  for an arbitrary dataset  $P(X,Y)$ . We can view much of machine learning as taking a data distribution outside of our control ( $P(X,Y)$  in this case), and adding a new random variable,  $Z_{X}$ , that is under our control.2 Doing this in the natural way gives the Markov chain  $Z_{X} \gets X \leftrightarrow Y$ , shown as an information diagram in Figure 1 (Right). The placement of  $H(Z_{X})$  in that diagram carefully maintains the conditional independence between  $Y$  and  $Z_{X}$  given  $X$ , but is otherwise fully general. Some of the entropy of  $Z_{X}$  is unassociated with any other variable; some is only associated with  $X$ , and some is associated with  $X$  and  $Y$  together. Figure 1 (Right), then, shows diagrammatically the state of the learned representation early in training. We would like instead for  $H(Z_{X})$  to exactly cover  $I(X;Y)$ , as in the gray area in Figure 1. We can find an objective function that achieves this goal optimally by considering the equalities that must hold between the various entropies and mutual informations at the moment that  $H(Z_{X})$  covers  $I(X;Y)$ .

Broadly, then, our goal is to take the information diagram in Figure 1 (Right), and transform it into Figure 1 (Left), where the information that  $Z_{X}$  represents coincides exactly with the information shared between  $X$  and  $Y$ .

First, note the following equalities in that scenario:

$$
I (X; Y; Z _ {X}) = I (X; Y) = I (X; Z _ {X}) = I (Y; Z _ {X})
$$

These equalities can be read directly from the information diagram in Figure 1 (Left). Because of our Markov chain, we can say definitively that maximizing  $I(X;Z_X)$  can never lead to a minimal representation. While doing so will cover  $I(X;Y)$ , there will be nothing in the objective function to prevent it from also covering all of  $H(X)$ . However, maximizing  $I(Y;Z_X)$  is consistent with learning the Minimal Information and necessary for learning the Necessary Information. It is not sufficient, though, as  $H(Z)$  can still cover  $H(X)$  completely when  $I(Y;Z_X)$  is maximal.

Now, consider the following equalities, also visible in Figure 1 (Left):

$$
I (X; Y | Z _ {X}) = I (X; Z _ {X} | Y) = I (Y; Z _ {X} | X) = 0
$$

With our Markov chain, we have the following well-known equality (Cover & Thomas, 2006):

$$
I (X; Z _ {X} | Y) = I (X; Z _ {X}) - I (Y; Z _ {X}) \tag {2}
$$

This is guaranteed to be non-negative, even for continuous random variables, as both terms are mutual informations, which are non-negative, and the Markov chain guarantees that  $I(Y;Z_X)$  is no larger than  $I(X;Z_X)$ , by the data processing inequality. From an optimization perspective, this is ideal - we have a term that we can minimize, and we can directly know how far we are from the optimal value of 0 (measured in nats, so it is interpretable), when we are done (when it's close enough to 0 that we are satisfied), and when our model is insufficient for the task (i.e., when this term isn't close enough to 0).

The above derivation gives the fully general Conditional Entropy Bottleneck objective:

$$
\mathrm {C E B} \equiv \min  I (X; Z _ {X} | Y) - I (Y; Z _ {X}) \tag {3}
$$

It is straightforward to turn this into a variational objective function, similar to the Variational Information Bottleneck (VIB) (Alemi et al., 2017). Taking the two terms in turn, we have:3

$$
\begin{array}{l} \min  I (X; Z _ {X} | Y) = I (X; Z _ {X}) - I (Y; Z _ {X}) (4) \\ = H \left(Z _ {X}\right) - H \left(Z _ {X} \mid X\right) - H \left(Z _ {X}\right) + H \left(Z _ {X} \mid Y\right) (5) \\ = - H \left(Z _ {X} | X\right) + H \left(Z _ {X} | Y\right) (6) \\ = \left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log p \left(z _ {X} | y\right) \right\rangle (7) \\ \leq \left\langle \log e \left(z _ {X} | x \right\rangle \right\rangle - \left\langle \log p \left(z _ {X} | y\right) \right\rangle + \mathrm {K L} [ p \left(z _ {X} | y\right) \| b \left(z _ {X} | y\right) ] (8) \\ = \left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log b \left(z _ {X} | y\right) \right\rangle (9) \\ \end{array}
$$

$e(z_{X}|x)$  is our encoder. It is not a variational approximation, even though it has learned parameters.  $b(z_{X}|y)$  is the variational approximation to what we think of as the backward encoder.

The second term:

$$
\begin{array}{l} \max  I (Y; Z _ {X}) = H (Y) - H (Y \mid Z _ {X}) (10) \\ \Rightarrow \max  - H (Y | Z _ {X}) (11) \\ = \left\langle \log p (y \mid z _ {X}) \right\rangle (12) \\ \geq \left\langle \log p (y | z _ {X}) \right\rangle - \mathrm {K L} [ p (y | z _ {X}) \| c (y | z _ {X}) ] (13) \\ = \left\langle \log p (y | z _ {X}) \right\rangle - \left\langle \log p (y | z _ {X}) \right\rangle + \left\langle \log c (y | z _ {X}) \right\rangle (14) \\ = \left\langle \log c (y \mid z _ {X}) \right\rangle (15) \\ \end{array}
$$

$c(y|z_x)$  is the variational approximation to the classifier (although that name is arbitrary, given that  $Y$  may not be labels).

The variational bounds derived above give us a fully tractable objective function that works on large-scale problems, Variational Conditional Entropy Bottleneck (VCEB):

$$
\mathrm {C E B} \equiv \min  I (X; Z _ {X} | Y) - I (Y; Z _ {X}) \Rightarrow \min  \left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log b \left(z _ {X} | y\right) \right\rangle - \left\langle \log c \left(y | z _ {X}\right) \right\rangle \equiv \mathrm {V C E B} \tag {16}
$$

The distributions with letters other than  $p$  are assumed to have learned parameters, which we otherwise omit in the notation. In other words, all three of  $e(\cdot), b(\cdot)$ , and  $c(\cdot)$  have learned parameters, just as in the encoder and decoder of a normal VAE (Kingma & Welling, 2014), or the encoder, classifier, and marginal in a VIB model (Alemi et al., 2017). Indeed, it is possible to switch between a CEB model and a VIB model simply by replacing the marginal  $m(z_{X})$  with the backward encoder,  $b(z_{X}|y)$  and updating the loss function provided to the optimizer.

We will name the  $I(X;Z_X|Y)$  term the Residual Information - this is the excess information in our representation beyond the information shared between  $X$  and  $Y$ :

$$
R e _ {X / Y} \equiv \left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log b \left(z _ {X} | y\right) \right\rangle \geq - H \left(Z _ {X} | X\right) + H \left(Z _ {X} | Y\right) = I (X; Z _ {X} | Y) \tag {17}
$$

There are a number of natural variations on this objective. We describe a few of them in Appendix C.

# 4 VARIATIONAL INFORMATION BOTTLENECK

The Information Bottleneck (IB) (Tishby et al., 2000) attempts to learn a representation of  $X$  and  $Y$  subject to an information constraint:

$$
\max  I (Z; Y) \text {s u b j e c t} I (Z; X) \leq R \tag {18}
$$

where  $R$  is a constant bottleneck. This can be rewritten as an unconstrained Lagrangian optimization:

$$
\max  I (Z; Y) - \beta I (Z; X) \tag {19}
$$

where  $\beta$  controls the size of the bottleneck.

A variational version of this objective is presented in Alemi et al. (2017). That objective is the Variational Information Bottleneck (VIB):

$$
V I B \equiv \min  \beta \left(\left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log m \left(z _ {X}\right) \right\rangle\right) - \left\langle \log c (y \mid z _ {X}) \right\rangle \tag {20}
$$

This is very similar to CEB, but instead of the backward encoder, VIB has a marginal posterior,  $m(z_{X})$ , which is a variational approximation to  $e(z_{X}) = \int dx p(x)e(z_{X}|x)$ . Additionally, it has a hyperparameter,  $\beta$ . We show in Appendix B that the optimal value for  $\beta = \frac{1}{2}$  when attempting to adhere to the MNI criterion.

Following Alemi et al. (2018), we define the Rate  $(R)$ :

$$
R \equiv \left\langle \log e \left(z _ {X} | x\right) \right\rangle - \left\langle \log m \left(z _ {X}\right) \right\rangle \geq I (X; Z _ {X}) \tag {21}
$$

# 5 TRAINING

Because of the properties of  $Re_{X/Y}$ , we can consider different training algorithms. In particular, we can avoid needing to look at validation set performance in order to decide when to lower the learning

Table 1: Accuracy and rates  $(R)$  for each model. Bold indicates the best score in that column. Determ doesn't have a rate, since it doesn't have an explicit encoder distribution. The final rate for the other four models is reported, as well as the peak rate achieved during training. The true mutual information for Fashion MNIST is  $I(X;Y) = 2.3$  nats, so achieving  $R = 2.3$  is optimal according to MNI.  

<table><tr><td>Model</td><td>Accuracy</td><td>Train R 
final (peak)</td></tr><tr><td>Determ</td><td>92.7</td><td>n/a</td></tr><tr><td>VIB0.01</td><td>93.0</td><td>2.6 (11.6)</td></tr><tr><td>VIB0.1</td><td>92.7</td><td>2.3 (3.2)</td></tr><tr><td>VIB0.5</td><td>90.0</td><td>2.3 (2.4)</td></tr><tr><td>CEB</td><td>92.9</td><td>2.3 (2.3)</td></tr></table>

rate. The closer we can get  $Re_{X/Y}$  to 0 on the training set, the better we will generalize to data drawn from the same distribution. Consequently, one simple approach to training is to set a high initial learning rate (possibly with reverse annealing of the learning rate (Goyal et al., 2017)), and then lower the learning rate after any epoch of training that doesn't result in a new lowest mean residual information on the training data. This is equivalent to the dev-decay training algorithm of Wilson et al. (2017), but does not require the use of a validation set. Additionally, since the training set is typically much larger than a validation set would be, the average loss over the epoch is much more stable, so the learning rate is less likely to be lowered spuriously.

Of course, it would be inefficient to compute the mean residuals for the entire training set at the end of each epoch of training, but the same general argument holds for maintaining a running mean during the epoch. When training is making progress, this estimate is an upper bound on the training set residuals at the end of the epoch. However, it is possible for training to stall or even diverge during an epoch, and this estimate may not pick up immediately on those cases, delaying lowering the learning rate until more than an epoch of stalled or divergent training has happened. In our experience, this is not practically a problem – CEB models are very stable throughout training and can tolerate the occasional epoch where the learning rate is still too high. Indeed, we find it easiest to prevent any lowering of the learning rate for a large number of epochs (e.g., 40), and thereafter following this algorithm. We do not attempt to prove that this algorithm is optimal.

Remark.  $Re_{X/Y}$  directly measures how far from optimal our learned representation is. If our optimization procedure is sufficiently effective, the residual indicates that we could improve performance by increasing the capacity of our architecture or considering ways in which our model may be misspecified. Thus, CEB directly informs us of the possibility to improve our model.

# 6 CLASSIFICATION EXPERIMENTS

Our primary experiments are focused on comparing the performance of otherwise identical models when we merely change the objective function. Consequently, we aren't interested in demonstrating state-of-the-art results in all things. Instead, we are interested in relative differences in performance that can be directly attributed to the difference in objective.

With that in mind, we present results for classification of Fashion MNIST (Xiao et al., 2017) for five different models. The five models are: a deterministic model (Determ); three VIB models, with  $\beta \in \{\frac{1}{2}, 10^{-1}, 10^{-2}\}$  ( $\mathrm{VIB}_{0.5}$ ,  $\mathrm{VIB}_{0.1}$ ,  $\mathrm{VIB}_{0.01}$ ); and a CEB model.

All five models share the same core architecture mapping  $X$  to  $Y$ : a  $7 \times 2$  Wide Resnet (Zagoruyko & Komodakis, 2016) for the encoder, with a final layer of  $D = 4$  dimensions for the latent representation, followed by a two layer MLP classifier using ELU (Clevert et al., 2015) activations with a final categorical distribution over the 10 classes. The stochastic models parameterize the mean and variance of a  $D = 4$  fully covariate multivariate Normal distribution with the output of the encoder. Samples from that distribution are passed into the classifier MLP. Apart from that difference, the stochastic models don't differ from Determ during evaluation. None of the five models uses any form of regularization (e.g.,  $L_1$ ,  $L_2$ , DropOut (Srivastava et al., 2014), BatchNorm (Ioffe & Szegedy, 2015)).

![](images/7aa6dad08ecc0950c4b26c6cb342bfecbaca458fbff36cdb6a98d61b21adc937.jpg)  
Figure 2: Calibration plots with  $90\%$  confidence intervals for four of the models after 2,000 steps, 20,000 steps, and 40,000 steps (left, center, and right of each trio, respectively): a is CEB, b is  $\mathrm{VIB}_{0.5}$ , c is  $\mathrm{VIB}_{0.1}$ , d is Determ. Perfect calibration corresponds to the dashed diagonal lines. Underconfidence occurs when the points are above the diagonal. Overconfidence occurs when the points are below the diagonal.

The VIB models have an additional learned marginal,  $m(z_{X})$ , which is a mixture of  $240D = 4$  fully covariate multivariate Normal distributions. The CEB model instead has the backward encoder,  $b(z_{X}|y)$  which is a  $D = 4$  fully covariate multivariate Normal distribution parameterized by a 1 layer MLP mapping the label,  $Y = y$ , to the mean and variance. In order to simplify comparisons, for CEB we additionally train a marginal posterior identical in form to that used by the VIB models. However, for CEB,  $m(z_{X})$  is trained using a separate optimizer so that it doesn't impact training of the CEB objective in any way. Having  $m(z_{X})$  for both CEB and VIB allows us to compare the rate,  $R$ , of each model except Determ.

Since Fashion MNIST doesn't have a prespecified validation set, it offers an opportunity to test training algorithms that only look at training results, rather than relying on cross validation. To that end, the five models presented here are the first models with these hyperparameters that we trained on Fashion MNIST. The learning rate for the CEB model was lowered according to the training algorithm described in Section 5. The other four models followed the same algorithm, but instead of tracking  $Re_{X/Y}$ , they simply tracked their training loss. All five models were required to retain the initial learning rate of 0.001 for 40 epochs before they could begin lowering the learning rate. At no point during training did any of the models exhibit non-monotonic test accuracy, so we do not believe that this approach harmed any performance – all five models converged essentially smoothly to their final, reported performance.

In the case of a simple classification problem with a uniform distribution over classes in the training set, we can directly compute  $I(X;Y)$  as  $\log C$ , where  $C$  is the number of classes. See Table 1 for a comparison of the rates between the four variational models, as well as their accuracies. All but  $\mathrm{VIB}_{0.5}$  achieve the same accuracy. All four stochastic models get close to the ideal rate of 2.3 nats, but they get there by different paths. For the VIB models, the lower  $\beta$  is, the higher the rate goes early in training, before converging down to (close to) 2.3 nats. CEB never goes above 2.3 nats.

# 7 CALIBRATION

In Figure 2, we show calibration plots at various points during training for the four models. Calibration curves help analyze whether models are underconfident or overconfident. Each point in the plots corresponds to a  $5\%$  confidence range. Accuracy is measured in each bin. A well-calibrated model is correct half of the time it gives a confidence of  $50\%$  for its prediction.

All of the networks move from under- to overconfidence during training. However, CEB and  $\mathrm{VIB}_{0.5}$  are only barely overconfident, while reducing  $\beta$  to 0.1 is sufficient to make it nearly as overconfident

Table 2: Results for out-of-distribution detection (OoD). Thrsh. is the threshold score used:  $H$  is the entropy of the classifier;  $R$  and  $R{e}_{X/\widehat{Y}}$  are defined in Section 8. Arrows denote whether higher or lower scores are better. Bold indicates the best score in that column for a particular OoD dataset.  

<table><tr><td>OoD</td><td>Method</td><td>Thrsh.</td><td>FPR @ 95% TPR ↓</td><td>AUROC ↑</td><td>AUPR In ↑</td></tr><tr><td rowspan="10">U(0,1)</td><td>Determ</td><td>H</td><td>35.8</td><td>93.5</td><td>97.1</td></tr><tr><td rowspan="2">VIB0.01</td><td>H</td><td>41.1</td><td>92.5</td><td>96.0</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.1</td><td>H</td><td>43.5</td><td>94.5</td><td>96.2</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.5</td><td>H</td><td>73.2</td><td>87.0</td><td>90.5</td></tr><tr><td>R</td><td>80.6</td><td>57.1</td><td>51.4</td></tr><tr><td rowspan="3">CEB</td><td>H</td><td>63.4</td><td>92.8</td><td>95.1</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td>ReX/Ŷ</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="10">MNIST</td><td>Determ</td><td>H</td><td>59.0</td><td>88.4</td><td>90.0</td></tr><tr><td rowspan="2">VIB0.01</td><td>H</td><td>42.3</td><td>91.6</td><td>95.9</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.1</td><td>H</td><td>60.3</td><td>84.7</td><td>89.7</td></tr><tr><td>R</td><td>0.5</td><td>86.8</td><td>99.8</td></tr><tr><td rowspan="2">VIB0.5</td><td>H</td><td>70.2</td><td>79.6</td><td>86.8</td></tr><tr><td>R</td><td>12.3</td><td>66.7</td><td>91.1</td></tr><tr><td rowspan="3">CEB</td><td>H</td><td>70.6</td><td>77.8</td><td>73.0</td></tr><tr><td>R</td><td>0.1</td><td>94.4</td><td>99.9</td></tr><tr><td>ReX/Ŷ</td><td>0.2</td><td>92.0</td><td>99.9</td></tr><tr><td rowspan="10">Vertical Flip</td><td>Determ</td><td>H</td><td>66.8</td><td>88.6</td><td>90.2</td></tr><tr><td rowspan="2">VIB0.01</td><td>H</td><td>57.6</td><td>82.6</td><td>80.3</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.1</td><td>H</td><td>65.3</td><td>84.5</td><td>85.2</td></tr><tr><td>R</td><td>0.0</td><td>99.2</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.5</td><td>H</td><td>79.7</td><td>79.8</td><td>81.4</td></tr><tr><td>R</td><td>17.3</td><td>52.7</td><td>91.3</td></tr><tr><td rowspan="3">CEB</td><td>H</td><td>68.0</td><td>84.9</td><td>85.5</td></tr><tr><td>R</td><td>0.0</td><td>90.7</td><td>100.0</td></tr><tr><td>ReX/Ŷ</td><td>0.0</td><td>92.6</td><td>100.0</td></tr></table>

as the deterministic model. This overconfidence is one of the issues that is correlated with exceeding the MNI during training (see Table 1). See Appendix B for a more in-depth explanation for how this can occur.

# 8 OUT-OF-DISTRIBUTION DETECTION

We test the ability of the five models to detect three different out-of-distribution (OoD) detection settings.  $U(0,1)$  is uniform noise in the image domain. MNIST uses the MNIST test set. Vertical Flip is the most challenging, using vertically flipped Fashion MNIST test images, as originally proposed in Alemi et al. (2018).

We use three different metrics for thresholding. The first two,  $H$  and  $R$ , were proposed in Alemi et al. (2018).  $H$  is the classifier entropy.  $R$  is the rate, defined in Section 4. The third metric is specific to CEB:  $Re_{X / \hat{Y}}$ . This is the predicted residual information - since we don't have access to the true value of  $Y$  at test time, we use  $\hat{y} \sim c(y|z_X)$  to calculate  $H(Z_X|\hat{Y})$ . This is no longer a valid bound on  $Re_{X / Y}$  as  $\hat{y}$  may not be from the true distribution  $p(x,y,z_X)$ . However, the better the classifier, the closer the estimate should be.

These three threshold scores are used with the standard suite of proper scoring rules: False Positive Rate at  $95\%$  True Positive Rate (FPR  $95\%$  TPR), Area Under the ROC Curve (AUROC), and Area Under the Precision-Recall Curve (AUPR). See Lee et al. (2018) for definitions.

The core result is that  $\mathrm{VIB}_{0.5}$  performs much less well at the OoD tasks than the other two VIB models and CEB. We believe that this is another result of  $\mathrm{VIB}_{0.5}$  learning the right amount of information, but

not learning all of the right information, thereby demonstrating that it is not a valid MNI objective, as explored in Appendix B. On the other hand, the other two VIB objectives seem to perform extremely well, which is the benefit they get from capturing a bit more information about the training set. We will see below that there is a price for that information, however.

# 9 ADVERSARIAL EXAMPLE ROBUSTNESS AND DETECTION

Table 3: Results for adversarial example detection (Attack). All attacks are targeting the "trousers" class in Fashion MNIST. CW is Carlini & Wagner (2017b).  $CW$ ,  $(C = 1)$  is CW with an additional confidence penalty set to 1.  $CW$ ,  $(C = 1)$  Det. is a custom CW attack targeting CEB's detection mechanism,  $Re_{X / \hat{Y}}$ .  $L_0, L_1, L_2, L_\infty$  report the corresponding norm (mean  $\pm 1$  std.) of successful adversarial perturbations. Higher norms on CW indicate that the attack had a harder time finding adversarial perturbations, since it starts by looking for the smallest possible perturbation. The remaining columns are as in Table 2. Arrows denote whether higher or lower scores are better. **Bold** indicates the best score in that column for a particular adversarial attack. See Section 6 for details of the models and Section 9 for details of the attacks.

<table><tr><td>Attack</td><td>Model</td><td>Attack Success ↓</td><td>L0↑</td><td>L1↑</td><td>L2↑</td><td>L∞↑</td><td>Thrsh.</td><td>FPR @ 95% TPR ↓</td><td>AUROC ↑</td><td>AUPR In ↑</td></tr><tr><td rowspan="10">CW</td><td>Determ</td><td>100.0%</td><td>377.1 ±100.3</td><td>16.2 ±10.2</td><td>1.4 ±1.7</td><td>0.2 ±0.1</td><td>H</td><td>15.4</td><td>90.7</td><td>86.0</td></tr><tr><td rowspan="2">VIB0.01</td><td rowspan="2">55.2%</td><td rowspan="2">389.6 ±100.9</td><td rowspan="2">17.1 ±10.3</td><td rowspan="2">1.5 ±1.8</td><td rowspan="2">0.2 ±0.1</td><td>H</td><td>11.2</td><td>59.9</td><td>90.0</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.1</td><td rowspan="2">68.8%</td><td rowspan="2">392.1 ±101.6</td><td rowspan="2">29.2 ±18.1</td><td rowspan="2">5.1 ±7.5</td><td rowspan="2">0.4 ±0.2</td><td>H</td><td>16.5</td><td>77.4</td><td>80.0</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.5</td><td rowspan="2">35.8%</td><td rowspan="2">432.0 ±99.6</td><td rowspan="2">40.1 ±32.1</td><td rowspan="2">9.4 ±14.4</td><td rowspan="2">0.5 ±0.3</td><td>H</td><td>64.2</td><td>62.5</td><td>55.3</td></tr><tr><td>R</td><td>0.0</td><td>98.7</td><td>100.0</td></tr><tr><td rowspan="3">CEB</td><td rowspan="3">35.8%</td><td rowspan="3">416.4 ±97.7</td><td rowspan="3">33.6 ±30.3</td><td rowspan="3">7.4 ±15.0</td><td rowspan="3">0.3 ±0.2</td><td>H</td><td>62.2</td><td>65.2</td><td>57.1</td></tr><tr><td>R</td><td>0.0</td><td>99.7</td><td>100.0</td></tr><tr><td>ReX/Ŷ</td><td>0.0</td><td>99.5</td><td>100.0</td></tr><tr><td rowspan="10">CW (C=1)</td><td>Determ</td><td>100.0%</td><td>378.7 ±100.3</td><td>16.6 ±10.4</td><td>1.4 ±1.9</td><td>0.2 ±0.1</td><td>H</td><td>17.9</td><td>90.9</td><td>85.7</td></tr><tr><td rowspan="2">VIB0.01</td><td rowspan="2">96.7%</td><td rowspan="2">381.3 ±101.5</td><td rowspan="2">17.4 ±10.5</td><td rowspan="2">1.6 ±1.9</td><td rowspan="2">0.2 ±0.1</td><td>H</td><td>19.6</td><td>72.1</td><td>89.6</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.1</td><td rowspan="2">97.3%</td><td rowspan="2">382.8 ±100.4</td><td rowspan="2">28.2 ±17.2</td><td rowspan="2">4.8 ±7.4</td><td rowspan="2">0.4 ±0.2</td><td>H</td><td>28.7</td><td>86.0</td><td>79.1</td></tr><tr><td>R</td><td>0.0</td><td>100.0</td><td>100.0</td></tr><tr><td rowspan="2">VIB0.5</td><td rowspan="2">50.4%</td><td rowspan="2">422.0 ±101.3</td><td rowspan="2">36.4 ±28.6</td><td rowspan="2">7.8 ±12.3</td><td rowspan="2">0.4 ±0.2</td><td>H</td><td>86.5</td><td>59.8</td><td>54.1</td></tr><tr><td>R</td><td>0.1</td><td>96.2</td><td>100.0</td></tr><tr><td rowspan="3">CEB</td><td rowspan="3">48.0%</td><td rowspan="3">417.6 ±95.5</td><td rowspan="3">33.3 ±29.8</td><td rowspan="3">7.3 ±15.4</td><td rowspan="3">0.4 ±0.2</td><td>H</td><td>77.4</td><td>63.5</td><td>56.4</td></tr><tr><td>R</td><td>0.0</td><td>99.3</td><td>100.0</td></tr><tr><td>ReX/Ŷ</td><td>0.0</td><td>98.7</td><td>100.0</td></tr><tr><td rowspan="3">CW (C=1) Det.</td><td rowspan="3">CEB</td><td rowspan="3">25.1%</td><td rowspan="3">416.4 ±92.2</td><td rowspan="3">84.1 ±44.0</td><td rowspan="3">34.4 ±22.8</td><td rowspan="3">0.9 ±0.1</td><td>H</td><td>95.1</td><td>56.4</td><td>45.0</td></tr><tr><td>R</td><td>66.5</td><td>69.3</td><td>88.5</td></tr><tr><td>ReX/Ŷ</td><td>72.9</td><td>69.9</td><td>87.6</td></tr></table>

Adversarial examples were first noted in Szegedy et al. (2013). The first practical attack, Fast Gradient Method (FGM) was introduced shortly after (Goodfellow et al., 2015). Since then, many new attacks have been proposed. Most relevant to us is the Carlini-Wagner (CW) attack (Carlini & Wagner, 2017b), which was the first practical attack to directly use a blackbox optimizer to find minimal perturbations. Many defenses have also been proposed, but almost all of them are broken (Carlini & Wagner, 2017a; Athalye et al., 2018). This work may be seen as a natural continuation of the adversarial analysis of Alemi et al. (2017), which showed that VIB naturally had robustness to whitebox adversaries, including CW. In that work, the authors did not train any VIB models with a learned  $m(z_{X})$ , which results in much weaker models, as shown in Alemi et al. (2018). We believe this is the first work exploring learning a marginal and using that marginal in an adversarial setting.

We consider CW in the whitebox setting to be the current gold standard attack, even though it is much more expensive than FGM or the various iterative attacks like DeepFool (Moosavi-Dezfooli et al., 2016) or iterative variants of FGM (Kurakin et al., 2016). Running an optimizer directly on the model to find the perturbation that can fool the model tells us much more about the robustness of the model than approaches that only take a small, fixed number of gradient steps. Additionally searching over the space of perturbation magnitudes makes the attack very hard to defend against, and consequently the current best option for testing robustness.

Here, we explore three variants of the CW  $L_{2}$  targeted attack. The implementation the first two CW attacks are from Papernot et al. (2018). CW and CW ( $C = 1$ ) are the baseline CW attack, and CW with a confidence adjustment of 1. Note that in order for these attacks to succeed at all on CEB, we had to increase the default CW learning rate to  $5 \times 10^{-1}$ . Without that increase, CW found almost no adversaries in our early experiments. All other parameters are left at their defaults for CW, apart from setting the clip ranges to [0, 1]. The final attack, CW ( $C = 1$ ) Det. is a modified version of CW ( $C = 1$ ) that additionally incorporates a detection tensor into the loss. For CEB, we had it target minimizing  $Re_{X / \hat{Y}}$  in order to try to break the network's ability to detect the attack.

All of the attacks are targeting the trouser class of Fashion MNIST, as that is the most distinctive class. Targeting a less distinctive class, such as one of the shirt classes, would confuse the difficulty of classifying the different shirts and the difficulty of the adversary. We run each of the first three attacks on the entire Fashion MNIST test set (all 10,000 images). For the stochastic networks, we permit 32 encoder samples and take the mean classification result (these samples are also used for gradient generation in the attacks to be fair to the attacker). CW is expensive, but we are able to run these on a single GPU in about 30 minutes. However, CW ( $C = 1$ ) Det. ends up being about 200 times more expensive – we were only able to run 1000 images, permitting only 8 samples from the encoder, and it took  $2\frac{1}{2}$  hours. Consequently, we only run CW ( $C = 1$ ) Det. on the CEB model, and the results are less significant.

Our metric for robustness is the following: we count the number of adversarial examples that change a correct prediction to an incorrect prediction of the target class, and divide by the number of correct predictions the model makes on the non-adversarial inputs. We additionally measure the size of the resulting perturbations. For CW, a larger perturbation generally indicates that the attack had to work harder to find an adversarial example, making this a secondary indication of robustness. Finally, we measure detection using the same thresholding techniques from Table 2.

The results of these experiments are in Table 3. We show all 20,000 images for four of the models in Figure 7. The most striking pattern in the models is how well  $\mathrm{VIB}_{0.01}$  and  $\mathrm{VIB}_{0.1}$  do at detection, while  $\mathrm{VIB}_{0.5}$  is dramatically more robust. We think that this is the most compelling indication of the importance of not overshooting  $I(X;Y)$  - even minor amounts of overshooting appear to destroy the robustness of the model. On the other hand,  $\mathrm{VIB}_{0.5}$  has a hard time with detection, which indicates that, while it has learned a highly compressed representation, it has not learned the optimal set of bits. Thus, as we discuss in Appendix B, VIB trades off between learning the necessary information, which allows it to detect attacks perfectly, and learning the minimum information, which allows it to be robust to attacks.

In the end, however, CEB permits both – it maintains the necessary information for detecting powerful whitebox attacks, but also retains the minimum information, providing robustness. This is again visible in the CW ( $C = 1$ ) Det. attack, which directly targets CEB's detection mechanism. Even though it no longer does well detecting the attack, the model becomes more robust to the attack, as indicated both by the much lower attack success rate and the much larger perturbation magnitudes.

# 10 GENERALIZATION EXPERIMENTS

We replicate the basic experiment from Zhang et al. (2016): we use the images from Fashion MNIST, but replace the training labels with fixed random labels (i.e., the same random value every epoch of training). We use that dataset to train multiple deterministic models, CEB models, and a range of VIB models. We find that the CEB model never learns (even after 100 epochs of training), the deterministic model always learns (after about 40 epochs of training it begins to memorize the random labels), and the VIB models only learn with  $\beta \leq 0.001$ .

The fact that CEB and VIB with  $\beta$  near  $\frac{1}{2}$  manage to resist memorizing random labels is our final empirical demonstration that MNI is a powerful criterion for objective functions.

# 11 CONCLUSION

We have presented the basic form of the Conditional Entropy Bottleneck (CEB), motivated by the Minimum Necessary Information (MNI) criterion for optimal representations. We have shown through careful experimentation that simply by switching to CEB, you can expect substantial improvements in OoD detection, adversarial example detection and robustness, calibration, and generalization. Additionally, we have shown that it is possible to get all of these advantages without using any additional form of regularization, and without any new hyperparameters. We have argued empirically that objective hyperparameters can lead to hard-to-predict suboptimal behavior, such as memorizing random labels, or reducing robustness to adversarial examples. In Appendix C and in future work, we will show how to generalize CEB beyond the simple case of two observed variables.

It is our perspective that all of the issues explored here – miscalibration, failure at OoD tasks, vulnerability to adversarial examples, and dataset memorization – stem from the same underlying issue, which is retaining too much information in the representation learned, either implicitly or explicitly, for the training inputs. We believe that the MNI criterion and CEB show a path forward for many tasks in machine learning, permitting fast, amortized inference while ameliorating major problems.

ACKNOWLEDGMENTS

REDACTED

# REFERENCES

A. A. Alemi, B. Poole, I. Fischer, J. V. Dillon, R. A. Saurous, and K. Murphy. Fixing a Broken ELBO. ICML2018, 2018. URL http://arxiv.org/abs/1711.00464.  
Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. Deep Variational Information Bottleneck. In International Conference on Learning Representations, 2017. URL http://arxiv.org/abs/1612.00410.  
Alexander A Alemi, Ian Fischer, and Joshua V Dillon. Uncertainty in the variational information bottleneck. arXiv preprint arXiv:1807.00906, 2018.  
Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.  
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798-1828, 2013.  
William Bialek and Naftali Tishby. Predictive information. arXiv preprint cond-mat/9902341, 1999.  
Nicholas Carlini and David Wagner. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security, pp. 3-14. ACM, 2017a.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy (SP), pp. 39-57. IEEE, 2017b.  
Djork-Arné Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289, 2015.  
Thomas M Cover and Joy A Thomas. Elements of information theory 2nd edition. John Wiley & Sons, 2006.  
T. DeVries and G. W. Taylor. Learning Confidence for Out-of-Distribution Detection in Neural Networks. arXiv: 1802.04865, 2018. URL https://arxiv.org/abs/1802.04865.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In CoRR, 2015.  
Priya Goyal, Piotr Dólar, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
Peter D Grünwald. The Minimum Description Length Principle. MIT press, 2007.  
C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On Calibration of Modern Neural Networks. arXiv: 1706.04599, 2017. URL https://arxiv.org/abs/1706.04599.  
D. Hendrycks and K. Gimpel. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. arXiv: 1610.02136, 2016. URL https://arxiv.org/abs/1610.02136.  
Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412.6980.  
Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations, 2014.  
Andrei N Kolmogorov. Three approaches to the quantitative definition of information'. Problems of information transmission, 1(1):1-7, 1965.  
Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533, 2016.  
B. Lakshminarayanan, A. Pritzel, and C. Blundell. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. arXiv: 1612.01474, 2016. URL https://arxiv.org/abs/1612.01474.  
Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278-2324, 1998.  
K. Lee, H. Lee, K. Lee, and J. Shin. Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples. arXiv: 1711.09325, 2017. URL https://arxiv.org/abs/1711.09325.  
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. arXiv preprint arXiv:1807.03888, 2018.  
S. Liang, Y. Li, and R. Srikant. Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks. arXiv: 1706.02690, 2017. URL https://arxiv.org/abs/1706.02690.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2574-2582, 2016.  
Nicolas Papernot, Fartash Faghri, Nicholas Carlini, Ian Goodfellow, Reuben Feinman, Alexey Kurakin, Cihang Xie, Yash Sharma, Tom Brown, Aurko Roy, Alexander Matyasko, Vahid Behzadan, Karen Hambardzumyan, Zhishuai Zhang, Yi-Lin Juang, Zhi Li, Ryan Sheatsley, Abhibhav Garg, Jonathan Uesato, Willi Gierke, Yinpeng Dong, David Berthelot, Paul Hendricks, Jonas Rauber, and Rujun Long. Technical report on the cleverhans v2.1.0 adversarial examples library. arXiv preprint arXiv:1610.00768, 2018.  
Fazlollah M Reza. An introduction to information theory. Courier Corporation, 1994.  
Claude Elwood Shannon. A Mathematical Theory of Communication. The Bell System Technical Journal, 27:379-423, 1948.  
Ravid Shwartz-Ziv and Naftali Tishby. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810, 2017.

Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan, I. Goodfellow, and R. Fergus. Intriguing properties of neural networks. arXiv: 1312.6199, 2013. URL https://arxiv.org/abs/1312.6199.  
N. Tishby and N. Zaslavsky. Deep Learning and the Information Bottleneck Principle. arXiv: 1503.02406, 2015. URL https://arxiv.org/abs/1503.02406.  
Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000.  
Aäron Van Den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew W Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. In SSW, pp. 125, 2016.  
Ramakrishna Vedantam, Ian Fischer, Jonathan Huang, and Kevin Murphy. Generative models of visually grounded imagination. International Conference on Learning Representations, 2018. URL https://arxiv.org/abs/1705.10762.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pp. 1096-1103. ACM, 2008.  
A. C. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht. The Marginal Value of Adaptive Gradient Methods in Machine Learning. arXiv: 1705.08292, 2017. URL https://arxiv.org/abs/1705.08292.  
Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.  
S. Zagoruyko and N. Komodakis. Wide Residual Networks. arXiv: 1605.07146, 2016. URL https://arxiv.org/abs/1605.07146.  
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.

Here we collect a number of results that are not critical to the core of the paper, but may be of interest to particular audiences.
