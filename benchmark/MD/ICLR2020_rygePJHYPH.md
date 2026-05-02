# TOWARDS TRUSTWORTHY PREDICTIONS FROM DEEP NEURAL NETWORKS WITH FAST ADVERSARIAL CALIBRATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

To facilitate a wide-spread acceptance of AI systems guiding decision making in real-world applications, trustworthiness of deployed models is key. That is, it is crucial for predictive models to be uncertainty-aware and yield well-calibrated (and thus trustworthy) predictions for both in-domain samples as well as under domain shift. Recent efforts to account for predictive uncertainty include post-processing steps for trained neural networks, Bayesian neural networks as well as alternative non-Bayesian approaches such as ensemble approaches and evidential deep learning. Here, we propose an efficient yet general modelling approach for obtaining well-calibrated, trustworthy probabilities for samples obtained after a domain shift. We introduce a new training strategy combining an entropy-encouraging loss term with an adversarial calibration loss term and demonstrate that this results in well-calibrated and technically trustworthy predictions for a wide range of perturbations. We comprehensively evaluate previously proposed approaches on different data modalities, a large range of data sets including sequence data, network architectures and perturbation strategies and observe that our modelling approach substantially outperforms existing state-of-the-art approaches, yielding well-calibrated predictions for both in-domain and out-of-domain samples.

# 1 INTRODUCTION

To facilitate a wide-spread acceptance of AI systems guiding decision making in real-world applications, trustworthiness of deployed models is key. Not only in safety-critical applications such as autonomous driving or medicine (Helldin et al., 2013; Caruana et al., 2015; Leibig et al., 2017), but also in dynamic open world systems in industry it is crucial for predictive models to be uncertainty-aware and yield well-calibrated (and thus trustworthy) predictions in the case of any gradual domain shift, covering the entire spectrum from in-domain ("known unknowns") to truly out-of-domain samples ("unknown unknowns"). In particular in industrial and IoT settings, deployed models may encounter erroneous and inconsistent inputs far away from the input domain throughout the life-cycle; in addition, the distribution of the input data may gradually move away from the distribution of the training data (e.g. due to wear and tear of the assets, maintenance procedures or change in usage patterns). The importance of technical robustness and safety in such settings is also highlighted by the recently published ethics guidelines by the European Commission, requiring for a trustworthy AI to be lawful, ethical and robust (technically and taking into account its social environment) $^{1}$ .

Recent efforts to account for predictive uncertainty include post-processing steps for trained neural networks, where for example a validation set, drawn from the same distribution as the training data, is used to rescale the logit vectors returned by a trained neural network such that in-domain predictions are well calibrated (Platt, 1999; Guo et al., 2017). Orthogonal approaches have been proposed where trust scores and other measures for out-of-distribution (OOD) detection are derived, typically also based on trained networks (Liang et al., 2018; Jiang et al., 2018; Papernot & McDaniel, 2018); however these latter approaches are designed to detect only truly OOD samples and do not consider the continuum of domain shifts from in-domain to truly OOD. Alternative avenues towards intrinsically uncertainty-aware networks have been followed by training probabilistic models. In particular, a lot of research effort has been put into training Bayesian neural networks, where typically a prior

distribution over the weights is specified and, given the training data, a posterior distribution over the weights is inferred. This distribution can then be used to quantify predictive uncertainty. Since exact inference is untractable, a range of approaches for approximate inference has been proposed. In particular deterministic approaches based on variational approximations have recently received a lot of attention and range from estimators of the fully factorized posterior (Blundell et al., 2015), to the interpretation of Gaussian dropout as performing approximate inference with log-uniform priors and multiplicative Gaussian posteriors (Gal & Ghahramani, 2016) and facilitating a complex posterior using normalising flows (Louizos & Welling, 2017). Since such Bayesian approaches often come at a high computational cost, alternative non-Bayesian approaches have been proposed, that can also account for predictive uncertainty. These include ensemble approaches, where smooth predictive estimates can be obtained by training ensembles of neural networks using adversarial examples (Lakshminarayanan et al., 2017), and evidential deep learning, where predictions of a neural net are modelled as subjective opinions by placing a Dirichlet distribution on the class probabilities (Sensoy et al., 2018). Both for Bayesian and non-Bayesian approaches, uncertainty-awareness and the quality of predictive uncertainty are typically evaluated by analysing the behaviour of the predictive entropy for out-of-domain predictions in form of gradual perturbations (e.g. rotation of an image), adversarial examples or held-out classes. However, while an increasing predictive entropy for increasingly strong perturbations can be an indicator for uncertainty-awareness, simply high predictive entropy is not sufficient for trustworthy predictions, since this requires well-calibrated uncertainties, with the entropy matching the actual predictive power of the model. For example, if the entropy is too high, the model will yield under-confident predictions and similarly, if the entropy is too low, predictions will be over-confident. Notably, the focus of related work introduced above has been on image data and it remains unclear how these approaches perform for other data modalities, in particular when modelling sequences with long-range dependencies using complex architectures such as LSTMs (Hochreiter & Schmidhuber, 1997) or GRUs (Cho et al., 2014). Here, we propose an efficient yet general modelling approach for obtaining well-calibrated, trustworthy probabilities for both in-domain samples as well as under domain shift that can readily be applied to a wide range of data modalities and model architectures. More specifically, we first introduce a simple loss function to encourage high entropy on wrong predictions and combine this with an adversarial calibration loss term. We demonstrate on an array of perturbations that combining these two steps can allow us to train complex neural networks that make trustworthy predictions when faced with diverse types of domain shift. Our approach is simple and general, requiring only a small modification of existing training procedures. Thus, our contribution in this paper is three-fold. (i) we illustrate the limitations of entropy as measure for trustworthy predictions and introduce a new metric to quantify technical trustworthiness based on the concept of calibration (Dawid, 1982; DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005; Naeini et al., 2015; Guo et al., 2017). (ii) we introduce a new training strategy combining an entropy-encouraging loss with an adversarial calibration loss term and demonstrate that this results in better calibration and technical trustworthiness of predictions for diverse types of out-of-domain samples and perturbations, compared to the state-of-the-art. (iii) We apply the concept of uncertainty-awareness and trustworthiness to sequence models, systematically evaluate the predictive uncertainty of recurrent neural networks on a wide range of perturbations and demonstrate that our approach substantially improves predictive uncertainty over existing approaches when classifying long sequences. While previous studies only compared predictive entropy for one simple architecture (LeNet) and typically one type of domain shift (Sensoy et al., 2018; Louizos & Welling, 2017), we here present an extensive comparison of 4 different architectures across 10 different perturbation strategies.

# 2 TOWARDS TECHNICALLY TRUSTWORTHY PREDICTIONS

# 2.1 LIMITATIONS OF ENTROPY AS MEASURE FOR UNCERTAINTY-AWARENESS

Recent efforts in terms of evaluating predictive uncertainty have focused on entropy as measure for uncertainty-awareness for predictions under domain shift. While entropy quantifies the uncertainty encoded in the model output, it is not clear what absolute entropy is required for a model to be reliable, given a set of samples from an out-of-domain distribution. For example, a popular evaluation strategy consists of computing the absolute entropy for out-of-domain samples generated using perturbation strategies based on the images in the test set (e.g. gradual rotation of images) (Sensoy et al., 2018; Louizos & Welling, 2017). In this case, the entropy should increase with rotation angle,

as the accuracy decreases in a coordinated fashion (since the model was not trained with rotated images) (Fig. 1). However, such evaluations alone are not sufficient to determine whether model predictions are technically reliable (or trustworthy), since it is not clear whether accuracy and model confidence/uncertainty are coupled in a meaningful way. Building on prior work utilising the concept of calibration for in-domain predictions, this coupling can be quantified using reliability diagrams (Guo et al., 2017), where the model confidence (i.e. the probability associated with the predicted class label) is linked to accuracy in a stratified manner. For example, if a model makes a prediction on images rotated by 20 degrees, the accuracy as well as the confidence of the predictions should drop in a meaningful way: if a model is well calibrated, confidence and accuracy should match for all confidence levels between  $1 / n_{\mathrm{classes}}$  and 1.0. That is, for the subset of samples with confidence between e.g.  $60\%$  and  $70\%$  the average accuracy should lie in that same range; this relationship should hold for all intervals. Figure 1 illustrates that the accuracy decreases, while the entropy increases if perturbed images are fed to a trained neural network (top right); however, additional information directly linking the uncertainty or confidence of a model to its accuracy is required to establish whether predictions are calibrated. This is illustrated by reliability diagrams in figure 1 (bottom row), showing accuracy as function of binned confidence and the expected calibration error (ECE) curve, summarizing the calibration gap perturbations covering the entire spectrum of domain shifts. (DeGroot & Fienberg, 1983; Niculescu-Mizil & Caruana, 2005).

![](images/4d13caac5d1f4da66ca3338f16b827be0d1908c3e87270b723b4ebee574300b7.jpg)

![](images/d774cdc250d5525ba08425d2790c1ad8875b5bb1acf22855510da5dfda65b7d5.jpg)

![](images/1786b648478e387cff17636c39addf6945ec7c1fa8a32bff6a7575c7ff7ec330.jpg)  
Figure 1: Calibration of the predictive uncertainty under domain shift. Here, a LeNet model is trained on MNIST data and calibration of the predictive uncertainty is evaluated on images perturbed with increasing y-zoom. Epsilon denotes the relative perturbation strength. Top: For in-domain samples the model has a high accuracy and low entropy, for higher domain shifts wrong predictions are often made with high confidence (left). While increasing domain shift results in a decreased accuracy and increased entropy, it is not clear whether this increased entropy reflects a well calibrated model confidence (right). Bottom: Only reliability diagrams and the expected calibration error (ECE) reveal that the decline in accuracy does not match the confidence of the model. Left: Confidence matches accuracy for most bins. Middle: Model makes overconfident predictions (red bars illustrate calibration gap). Right: ECE curve quantifies how miss-calibration changes with increasing perturbation strength.

![](images/13a76c5cd0631d27a5574343e7dc6bdfdbf13c5fa551bba06e49876cecd18478.jpg)

![](images/0b98ddc1922cd6ef624f9082c9482fcde9b2e88eaf59f00fa3f73fa7cd22f31f.jpg)

# 2.1.1 QUANTIFYING CALIBRATION UNDER DOMAIN SHIFT USING THE EXPECTED-CALIBRATION-ERROR CURVE

We follow Guo et al. (2017) and define perfect calibration such that confidence and accuracy match for all confidence levels:

$$
\mathbb {P} (\hat {Y} = Y | \hat {P} = p) = p, \forall p \in [ 0, 1 ] \tag {1}
$$

with  $\hat{Y}$  being a class prediction of a label  $Y$  and  $\hat{P}$  its associated confidence. This directly leads to a definition of miss-calibration as the difference in expectation between confidence and accuracy:

$$
\underset {\hat {P}} {\mathbb {E}} \left[ | \mathbb {P} (\hat {Y} = Y | \hat {P} = p) - p | \right] \tag {2}
$$

A scalar summary measure, summarizing reliability diagrams in form of the calibration gap (red bars in figure 1, bottom row left and middle) and also approximating eq. 2 is the expected calibration error (ECE) (Naeini et al., 2015). The ECE takes a weighted average over the  $M$  equally spaced bins of the reliability diagram:

$$
\mathrm {E C E} = \sum_ {m = 1} ^ {M} \frac {\left| B _ {m} \right|}{n} \left| \operatorname {a c c} \left(B _ {m}\right) - \operatorname {c o n f} \left(B _ {m}\right) \right| \tag {3}
$$

with  $B_{m}$  being the set of indices of samples whose prediction confidence falls into its associated interval  $I_{m}$ .  $\mathrm{conf}(B_m)$  and  $\mathrm{acc}(B_m)$  are the average confidence and accuracy associated to  $B_{m}$  respectively and  $n$  the number of samples in the dataset.

It can be shown that ECE is directly connected to miss-calibration as ECE using  $M$  bins converges to the M-term Riemann-Stieltjes sum of eq. 2 (Guo et al., 2017).

To evaluate the robustness of a predictive model under domain shifts covering the entire spectrum from in-domain to truly OOD samples, we define 10 distinct perturbation types (not seen during training). Each perturbation strategy mimics a scenario where the data a deployed model encounters stems from a distribution that gradually shifts away from the training distribution in a different manner. For each perturbation type we compute the ECE for a range of perturbation strengths. We then generate a ECE-perturbation curve and introduce a measure summarizing overall calibration by computing a micro-averaged ECE across all perturbation strengths.

We assess 9 distinct image-based perturbation types including left rotation, right rotation, shift in x direction, shift in y direction, xy shift, shear, zoom in x direction, zoom in y direction and xy zoom for image data. In addition, we investigate robustness to random word swaps for text data. More specifically, a perturbation is generated by first drawing a random set of words in a corpus. Next each of these words is replaced by a word drawn at random from the vocabulary.

For all perturbation strategies, perturbed samples were generated at 10 different levels, starting at no perturbation, until accuracy reached random levels; relative perturbation strength is denoted by epsilon. The micro-averaged ECE for a specific perturbation strategy was computed by first perturbing each sample in the test set at 10 different levels and then calculating the overall ECE across all samples. By computing this micro-averaged ECE for 10 distinct perturbation types, we quantify the ability of neural networks to yield well-calibrated, technically robust predictions in diverse circumstances.

# 2.2 A SIMPLE APPROACH FOR CALIBRATED PREDICTIVE UNCERTAINTY ESTIMATION

# 2.2.1 PREDICTIVE ENTROPY

To mitigate overconfident predictions displayed by conventional deep neural networks, we first introduce a loss term encouraging a uniform distribution of the scores in case the model "does not know". That is, after removing non-misleading evidence, we distribute the remaining probability mass uniformly over  $C$  classes:  $L_{S} = \sum_{i=1}^{n} \sum_{j=1}^{C} -\frac{1}{C} \log(p_{ij}(1 - y_{ij}) + y_{ij})$ , with  $p_{ij}$  being the confidence associated to the  $j$ th class of sample  $i$ ,  $y_{ij}$  its one-hot encoded label.

This simple loss term increases uncertainty-awareness by encouraging an increased entropy  $(S)$  in the presence of high predictive uncertainty, while the loss surface remains largely unchanged. This has the advantage that our approach - in contrast to Bayesian neural networks or evidential deep learning - can be readily applied to complex architectures based on LSTMs or GRUs. In addition, the loss term is parameter free and thus does not require hyperparameter tuning, again facilitating easy usage.

# 2.2.2 ADVERSARIAL CALIBRATION

While the entropy-based loss term does encourage uncertainty-awareness, we found that it is beneficial to introduce an additional loss term addressing model calibration directly. Explicitly encouraging calibration for out-of-domain samples, however - e.g. via an ECE-based measure - requires knowledge on the type of perturbed, erroneous or even adversarial samples the model is expected to encounter. In many real-world applications it is not clear from which distribution these samples will be drawn and, more importantly, for model predictions to be truly trustworthy requires robustness against all such potential out-of-domain samples. That is, we would like our model to be technically robust for inputs around an  $\epsilon$ -neighbourhood of the in-domain training samples, for a wide range of  $\epsilon$  and for all

$2^{D}$  directions in  $\{-1,1\}^{D}$ . While inputs from a random direction are unlikely to be representative examples for generic out-of-domain samples, by definition adversarial examples are generated along a dimension where the loss is high. Lakshminarayanan et al. (2017) show that adversarial training can improve the smoothness of predictions, in particular when training an ensemble of 5 neural networks in an adversarial fashion. Here, we demonstrate that using adversarial samples to directly optimise model calibration (rather than the squared error of one-hot encoded labels (Lakshminarayanan et al., 2017)) results in substantially more trustworthy predictions for out-of-domain samples from a large number of unrelated directions.

We implement the calibration loss by minimizing the ECE for samples generated using the fast gradient sign method (FGSM) (Goodfellow et al., 2014), with  $\epsilon$  ranging from 0 to 0.5 (sampled at 10 equally spaced bins at random). Note that we do not use the FGSM samples for adversarial training in the sense that we do not try to minimize the reconstruction error (cross entropy) for those samples.

$$
\begin{array}{l} L _ {\mathrm {a d v}} = \left\| \left(\sum_ {m = 1} ^ {M} \frac {| B _ {m} |}{n} \mid \operatorname {a c c} (B _ {m}) - \operatorname {c o n f} (B _ {m}) |\right) \right\| _ {2} \\ = \| \mathrm {E C E} \| _ {2} \\ \end{array}
$$

The final loss balancing a standard reconstruction loss (categorical cross entropy (CCE)) against the entropy and adversarial calibration loss can then be written as  $L = L_{\mathrm{CCE}} + \lambda_{\mathrm{adv}}L_{\mathrm{adv}} + \lambda_{S}L_{S}$

The choice of hyperparameters  $\lambda_{\mathrm{adv}}$  and  $\lambda_S$  is described in the appendix along with a summary of the algorithm.

# 3 EXPERIMENTAL RESULTS

We compare our approach for fast adversarial calibration to both Bayesian and non-Bayesian work and perform an extensive set of experiments. We evaluate model trustworthiness by quantifying model calibration for 10 distinct strategies to generate out-of-domain samples. We show that our approach is able to yield technically trustworthy predictions across 4 datasets, 4 model architectures and three data modalities. We first show that our modelling approach substantially outperforms existing approaches for sequence models (sequences of pixels and sequences of words) and then illustrate improved performance for image data.

To evaluate our modelling approach for sequence data, we fit models on the following datasets and quantified technical robustness by computing the micro-averaged ECE:

1. Sequential MNIST. 10 classes of handwritten digits. Images are converted to pixel-wise sequences of length  $28 \times 28$ .  
2. 20 Newsgroups. News articles partitioned into 20 classes. News classes are modelled as sequences of words using word embeddings. We used the 20,000 most common words as vocabulary and a maximum word length of 2500.

We fitted LSTM and GRU models with one hidden layer for all sequence modelling tasks.

For the image classification tasks, we fitted a LeNet model to MNIST data in order to establish a fair comparison to the state-of-the-art (Guo et al., 2017; Sensoy et al., 2018). To evaluate the performance for more complex architectures, we further fitted a deep neural net with VGG19 architecture on the CIFAR10 dataset. We used standard splits into training and test set for all datasets.

We compared the following modelling approaches: (i) L2-Dropout, referring to a standard neural net with L2 regularisation as baseline, (ii) MC-Dropout corresponding to the modelling approach presented by Gal & Ghahramani (2016), (iii) Deep Ensembles referring to an approach based on an ensemble of neural nets trained using adversarial examples (Lakshminarayanan et al., 2017), (iv) EDL referring to Evidential Deep Learning (Sensoy et al., 2018), (v) MNF referring to a Bayesian neural network trained using multiplicative normalising flows Louizos & Welling (2017) and (vi) FALCON, which is our method based on Fast AdversariaL CalibratiON.

# 3.1 PREDICTIVE UNCERTAINTY FOR SEQUENCE MODELING

We trained LSTM models with one hidden layer of 130 hidden units using the RMSPROP optimizer. GRU models were trained with one hidden layer of 250 hidden units to reflect the reduced complexity

![](images/12f4a57832fbae7ed282faaf40b93e6dc5f2260c24edb0141bf41db4489c136d.jpg)  
(a) LSTM models

![](images/db6e50bc1933cd1a4c8c6b7ed906a7b374ff707874f238b7a772abec6801eb51.jpg)  
(b) GRU models  
Figure 2: Technical robustness of sequence models for classifying sequential MNIST data, quantified by computing the micro-averaged expected calibration error (lower is better). FALCON results in consistently well calibrated and robust predictions across 9 different perturbation strategies with substantially lower micro-averaged ECEs compared to existing methods, both for LSTM and GRU models. For fair comparison, we only show micro-averaged ECE for models with competitive accuracy, omitting EDL (see also Table S1)

Table 1: Test accuracy and average ECE (lower is better) across all perturbation strategies for LSTM and GRU models.  

<table><tr><td></td><td colspan="2">LSTM</td><td colspan="2">GRU</td></tr><tr><td></td><td>Test acc.</td><td>Mean ECE</td><td>Test acc.</td><td>Mean ECE</td></tr><tr><td>L2-Dropout</td><td>0.986</td><td>0.327</td><td>0.991</td><td>0.334</td></tr><tr><td>MC-Dropout</td><td>0.986</td><td>0.334</td><td>0.98</td><td>0.296</td></tr><tr><td>Deep-Ensemble</td><td>0.99</td><td>0.222</td><td>0.99</td><td>0.168</td></tr><tr><td>FALCON</td><td>0.978</td><td>0.118</td><td>0.988</td><td>0.108</td></tr></table>

of GRU cells compared to LSTM cells. The Bayesian neural network based on multiplicative normalizing flows (MNF) was developed for convolutional neural networks; since the transfer of such a complex modelling approach from convolutional neural networks to recurrent neural networks is out of the scope of this work, we omitted MNF in our comparison of sequence models.

Sequential MNIST For deep ensembles of LSTMs trained on sequential MNIST we found that models did not converge when training the networks with adversarial examples; we therefore also trained ensembles with a reduced  $\epsilon$  of 0.005 and report performance for this modified Deep Ensemble approach. For the deep ensemble of GRUs on sequential MNIST and the deep ensemble of LSTMs on the 20 Newsgroups data, we report performance with standard adversarial training  $(\epsilon = 0.01)$ .

Fitting LSTM models on sequential MNIST is a challenging task (Bai et al., 2018), and it was only possible to achieve state-of-the-art predictive power with EDL for shorter sequences (downsampling of images before conversion to sequence). While performance of GRUs was better for all modelling approaches, EDL also did not achieve a competitive accuracy (Table S1). We found that our approach achieved competitive predictive power for LSTM and GRU models and substantially improved calibration of the predictive uncertainty for both models (Figure 2, Table 1). This illustrates that in contrast to existing approaches FALCON is able to yield well-calibrated and trustworthy predictions without compromising on accuracy, even for challenging tasks such as classifying long sequences with LSTMs.

20 Newsgroups To further evaluate the ability of FALCON to model sequence data, we compared the performance of FALCON to existing approaches for an NLP task. To this end, we trained LSTMs to classify news articles into one of 20 classes. We generated vector representations of words using the pre-trained GLOVE embedding (length 100) and used the first 2500 words of an article as input for an LSTM. We trained LSTMs with one hidden layer of 130 hidden units and evaluated it on a

![](images/02ada887a532a03dd005d39c5b5117c79788f26f264cf364e460e7134771ae5e.jpg)  
Figure 4: Calibration of the predictive uncertainty under domain shift generated by increasing the y-zoom of each image in the test set in 10 steps (MNIST data). Left With increasing domain shift the confidence of predictions with FALCON decreases such that they match accuracy (c.f. overconfident predictions of same samples with L2 in Fig. 1). Middle: expected calibration error at 10 increasingly large levels of y-zoom. Only EDL and FALCON maintain a low ECE across all levels of y-zoom. Right: Entropy increases with larger y-zoom for all methods. While EDL starts at the highest entropy, this reflects under-confident predictions for low levels of perturbation (c.f. high ECE in middle panel, figure S3 (appendix)). Accuracy decreases with larger zoom to almost random levels.

![](images/c507894c338de54285bf3f68155066791b818a9ecdeccd5ce23bcbd1bec4172a.jpg)

perturbation strategy based on random word swaps. To establish a perturbation strategy with gradually increasing perturbations, we varied the fraction of words drawn from each sample between  $0\%$  and  $45\%$  in  $5\%$  steps (gradually decreasing accuracy to random levels).

Similar to the LSTM model trained on sequential MNIST, we found that EDL did not achieve competitive predictive power, with an accuracy of  $49.3\%$  only. In contrast, FALCON resulted in well-calibrated predictions while maintaining a competitive accuracy of  $75.7\%$ , compared to  $75.9\%$ ,  $72.8\%$  and  $77.3\%$  for L2-Dropout, MC-Dropout and Deep Ensemble respectively. As before, the model confidence of FALCON was substantially better calibrated than existing methods (Figure 3).

# 3.2 PREDICTIVE UNCERTAINTY FOR IMAGE CLASSIFICATION

We next evaluated the trustworthiness of predictions for image classification tasks. To establish a fair comparison with state-of-the-art models, including Bayesian neural networks, we first trained the 5 existing approaches and evaluated them on 9 different perturbation strategies (not used during training). While with increasingly strong perturbations the predictive entropy increased for all models, this was not necessarily matched by a good calibration across the range of the perturbation. At the typical example of the perturbation y-zoom, it becomes clear that for most methods entropy did not increase sufficiently fast to match the decrease in accuracy, resulting in increasingly overconfident predictions and an increasing ECE for stronger perturbations (Fig. 4). While FALCON and EDL yielded well-calibrated predictions that were robust across all perturbation levels, it is worth noting that EDL has a substantially higher ECE for in-domain predictions, reflecting under-confident predictions on the test set (see also Suppl. Fig. S3). We observed this tendency of EDL towards under-confidence when faced with new samples drawn from the same distribution as the training data (known unknowns) also for a different dataset and architecture (VGG19 on CIFAR10;  $\mathrm{ECE}_{\mathrm{FALCON}} = 0.107$ ,

$\mathrm{ECE}_{\mathrm{EDL}} = 0.125$  on the test set). We observed a similar behaviour across all other 8 perturbation strategies, which was reflected in the lowest micro-averaged ECE for FALCON, followed by EDL (Figure 5; Table 2).

To evaluate the technical robustness and calibration of FALCON on a more complex architecture for image classification, we trained a VGG19 model on the CIFAR10 dataset. We again observed a similar trend as for the MNIST data, with FALCON yielding well calibrated predictions across all perturbation strategies (Figure 5). Note that we omitted MNF due to the large memory requirements stemming from the use of multiplicative normalising flows.

![](images/b9254998f7443f6e9ff6cc500caa3811165969527e1edcc59d1f8ae15c282a68.jpg)  
Figure 3: Expected calibration error for 20 Newsgroups data.

![](images/0ff76d59ff819ecef8852743ba4a55a597eadfee143f882bf65ef4f2d5208fec.jpg)  
(a) LeNet model for MNIST data

![](images/2c4fc79c74dfaea65755b0273a06022570ddd499663f84e56cf485afe4888b6a.jpg)  
(b) LeNet model for CIFAR10 data  
Figure 5: Technical robustness of image classification models, quantified by computing the microaveraged expected calibration error (lower is better). FALCON results in consistently well calibrated and robust predictions across 9 different perturbation strategies.

Table 2: Test accuracy and mean ECE across all 9 perturbation strategies for the LeNet model trained on MNIST and the VGG19 model trained on CIFAR10  

<table><tr><td></td><td colspan="2">LeNet-MNIST</td><td colspan="2">VGG19-CIFAR10</td></tr><tr><td></td><td>Test acc.</td><td>Mean ECE</td><td>Test acc.</td><td>Mean ECE</td></tr><tr><td>L2-Dropout</td><td>0.99</td><td>0.243</td><td>0.88</td><td>0.57</td></tr><tr><td>MC-Dropout</td><td>0.992</td><td>0.179</td><td>0.839</td><td>0.377</td></tr><tr><td>MNF</td><td>0.993</td><td>0.197</td><td>NA</td><td>NA</td></tr><tr><td>Deep-Ensembles</td><td>0.98</td><td>0.242</td><td>0.847</td><td>0.334</td></tr><tr><td>EDL</td><td>0.989</td><td>0.102</td><td>0.876</td><td>0.197</td></tr><tr><td>FALCON</td><td>0.991</td><td>0.082</td><td>0.871</td><td>0.146</td></tr></table>

# 4 DISCUSSION AND CONCLUSION

We presented a fast, simple and generalizable approach for encouraging well-calibrated uncertainty-awareness of deep neural networks. To this end, we combine an entropy encouraging loss-term with an adversarial calibration loss and show on diverse data modalities and model architectures that our approach yields well-calibrated predictions for both in-domain and out-of-domain samples generated based on 10 distinct perturbations. We present the first detailed analysis of predictive uncertainty for out-of-domain predictions of recurrent neural networks and identify major drawbacks of existing methods that were developed for (and evaluated on) image classification tasks. Thus, Deep Ensembles of LSTMs did not converge when performing adversarial training the MNIST dataset; while it was possible to obtain meaningful predictions with very limited adversarial training, this means that higher entropy is mostly achieved by the ensemble effect rather than benefits from adversarial training itself. In addition, training an ensemble of neural networks increases training time linearly with the number of networks in the ensemble, which can be substantial for applications where training of a deep network on a large dataset can take several weeks. Similarly, EDL was only able to result in networks with a high accuracy when trained on short sequences; both for the sequential MNIST and and 20 Newsgroups data, the EDL approach resulted in a substantially lower accuracy compared to baseline LSTM and GRU models. This may be due to the joint goals of minimizing the prediction error and the variance of the Dirichlet experiment generated by the neural net changing the loss surface such that is more difficult to navigate, which can be problematic for complex models based on LSTM cells or GRU cells. While MC dropout is easy to fit and fast, it results only in small improvements over the L2-Dropout baseline, especially for sequence data. In contrast, our modeling approach is fast and robust, with well-calibrated predictive uncertainty across 10 perturbations, 4 datasets, 4 model architectures and three data modalities.

# REFERENCES

Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271, 2018.  
Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural networks. arXiv preprint arXiv:1505.05424, 2015.  
Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1721-1730. ACM, 2015.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnnc encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.  
A Philip Dawid. The well-calibrated bayesian. Journal of the American Statistical Association, 77 (379):605-610, 1982.  
Morris H DeGroot and Stephen E Fienberg. The comparison and evaluation of forecasters. Journal of the Royal Statistical Society: Series D (The Statistician), 32(1-2):12-22, 1983.  
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050-1059, 2016.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1321-1330. JMLR.org, 2017.  
Tove Helldin, Goran Falkman, Maria Riveiro, and Staffan Davidsson. Presenting system uncertainty in automotive uis for supporting trust calibration in autonomous driving. In Proceedings of the 5th international conference on automotive user interfaces and interactive vehicular applications, pp. 210-217. ACM, 2013.  
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735-1780, 1997.  
Heinrich Jiang, Been Kim, Melody Guan, and Maya Gupta. To trust or not to trust a classifier. In Advances in Neural Information Processing Systems, pp. 5541-5552, 2018.  
Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, pp. 6402-6413, 2017.  
Christian Leibig, Vaneeda Allken, Murat Seçkin Ayhan, Philipp Berens, and Siegfried Wahl. Leveraging uncertainty information from deep neural networks for disease detection. Scientific reports, 7(1):17816, 2017.  
Shiyu Liang, Yixuan Li, and R. Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. 2018. URL https://openreview.net/forum?id=H1VGkIxRZ.  
Christos Louizos and Max Welling. Multiplicative normalizing flows for variational bayesian neural networks. In International Conference on Machine Learning, pp. 2218-2227, 2017. URL http://proceedings.mlr.press/v70/luozos17a.html.  
Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In Twenty-Ninth AAAI Conference on Artificial Intelligence, 2015.

Alexandru Niculescu-Mizil and Rich Caruana. Predicting good probabilities with supervised learning. In Proceedings of the 22nd international conference on Machine learning, pp. 625-632. ACM, 2005.  
Nicolas Papernot and Patrick McDaniel. Deep k-nearest neighbors: Towards confident, interpretable and robust deep learning. arXiv preprint arXiv:1803.04765, 2018.  
John C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In ADVANCES IN LARGE MARGIN CLASSIFIERS, pp. 61-74. MIT Press, 1999.  
Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In Advances in Neural Information Processing Systems, pp. 3179-3189, 2018.
