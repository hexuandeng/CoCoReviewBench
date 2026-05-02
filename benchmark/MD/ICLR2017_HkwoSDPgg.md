# SEMI-SUPERVISED KNOWLEDGE TRANSFER FOR DEEP LEARNING FROM PRIVATE TRAINING DATA

Nicolas Papernot*

Pennsylvania State University ngp5056@cse.psu.edu

Martín Abadi

Google Brain abadi@google.com

Ulfar Erlingsson

Google ulfar@google.com

Ian Goodfellow

OpenAI  
ian@openai.com

Kunal Talwar

Google Brain kunal@google.com

# ABSTRACT

Some machine learning applications involve training data that is sensitive, such as the medical histories of patients in a clinical trial. A model may inadvertently and implicitly store some of its training data; careful analysis of the model may therefore reveal sensitive information.

To address this problem, we demonstrate a generally applicable approach to providing strong privacy guarantees for training data. The approach combines, in a black-box fashion, multiple models trained with disjoint datasets, such as records from different subsets of users. Because they rely directly on sensitive data, these models are not published, but instead used as "teachers" for a "student" model. The student learns to predict an output chosen by noisy voting among all of the teachers, and cannot directly access an individual teacher or the underlying data or parameters. The student's privacy properties can be understood both intuitively (since no single teacher and thus no single dataset dictates the student's training) and formally, in terms of differential privacy. These properties hold even if an adversary can not only query the student but also inspect its internal workings.

Compared with previous work, the approach imposes only weak assumptions on how teachers are trained: it applies to any model, including non-convex models like DNNs. We achieve state-of-the-art privacy/utility trade-offs on MNIST and SVHN thanks to an improved privacy analysis and semi-supervised learning.

# 1 INTRODUCTION

Some machine learning applications with great benefits are enabled only through the analysis of sensitive data, such as users' personal contacts, private photographs or correspondence, or even medical records or genetic sequences (Alipanahi et al., 2015; Kannan et al., 2016; Kononenko, 2001; Sweeney, 1997). Ideally, in those cases, the learning algorithms would protect the privacy of users' training data, e.g., by guaranteeing that the output model generalizes away from the specifics of any individual user. Unfortunately, established machine learning algorithms make no such guarantee; indeed, though state-of-the-art algorithms generalize well to the test set, they continue to overfit on specific training examples in the sense that some of these examples are implicitly memorized.

Recent attacks exploiting this implicit memorization in machine learning have demonstrated that private, sensitive training data can be recovered from models. Such attacks can proceed directly, by analyzing internal model parameters, but also indirectly, by repeatedly querying opaque models to gather data for the attack's analysis. For example, Fredrikson et al. (2015) used hill-climbing on the output probabilities of a computer-vision classifier to reveal individual faces from the training data. Because of those demonstrations—and because privacy guarantees must apply to worst-case outliers, not only the average—any strategy for protecting the privacy of training data should prudently assume that attackers have unfettered access to internal model parameters.

To protect the privacy of training data, this paper improves upon a specific, structured application of the techniques of knowledge aggregation and transfer (Breiman, 1994), previously explored by Nissim et al. (2007b), Pathak et al. (2010), and particularly Hamm et al. (2016). In this strategy, first, an ensemble (Dietterich, 2000) of teacher models is trained in such a way that each model is based on a disjoint subset of the sensitive data. Then, using auxiliary, unlabeled non-sensitive data, a student model is trained on the aggregate output of the ensemble, such that the student learns to accurately mimic the ensemble. Intuitively, this strategy ensures that the student does not depend on the details of any single sensitive training data point (e.g., of any single user), and, thereby, the privacy of the training data is protected even if attackers can observe the student's internal model parameters.

This paper shows how this strategy's privacy guarantees can be strengthened by restricting student training to a limited number of teacher votes, and by revealing only the topmost vote after carefully adding random noise. Furthermore, we introduce an improved privacy analysis that makes the strategy generally applicable to machine learning algorithms with high utility and meaningful privacy guarantees—in particular, when combined with semi-supervised learning.

To establish strong privacy guarantees, it is important to limit the student's access to its teachers, so that the student's exposure to teachers' knowledge can be meaningfully quantified and bounded. Fortunately, there are many techniques for speeding up knowledge transfer that can reduce the rate of student/teacher consultation during learning. We describe four such techniques in this paper, the most effective of which makes use of generative adversarial networks (Goodfellow et al., 2014) applied to semi-supervised learning. We use the implementation proposed by Salimans et al. (2016). Like all semi-supervised learning methods, it assumes the student has access to additional, unlabeled data, which, in this context, must be public or non-sensitive. This assumption should not greatly restrict our method's applicability: even when learning on sensitive data, a non-overlapping, unlabeled set of data typically exists, from which semi-supervised methods can extract distribution priors. For instance, large public datasets exist for text and images, and similarly even for medical data.

It seems intuitive, or even obvious, that a student machine learning model will provide good privacy when trained without access to sensitive training data, apart from a few, noisy votes from a teacher quorum. However, intuition is not sufficient because privacy properties can be surprisingly hard to reason about; for example, even a single data item can greatly impact machine learning models trained on a large corpus (Chaudhuri et al., 2011). Therefore, to limit the effect of any single sensitive data item on the student's learning, precisely and formally, we apply the well-established, rigorous standard of differential privacy (Dwork & Roth, 2014). Like all differentially private algorithms, our learning strategy carefully adds noise, so that the privacy impact of each data item can be analyzed and bounded. In particular, we dynamically analyze the sensitivity of the teachers' noisy votes; for this purpose, we use the state-of-the-art moments accountant technique from Abadi et al. (2016), which tightens the privacy bound when the topmost vote has a large quorum. As a result, for MNIST and similar benchmark learning tasks, our methods allow students to provide excellent utility, while our analysis provides meaningful worst-case guarantees. In particular, we can bound the metric for privacy loss (the differential-privacy  $\varepsilon$ ) to a range similar to that of existing, real-world privacy-protection mechanisms, such as Google's RAPPOR (Erlingsson et al., 2014).

Finally, it is an important advantage that our learning strategy and our privacy analysis do not depend on the details of the machine learning techniques used to train either the teachers or their student. Therefore, the techniques in this paper apply equally well for deep learning methods, or any such learning methods with large numbers of parameters, as they do for shallow, simple techniques. In comparison, Hamm et al. (2016) guarantee privacy only conditionally, for a restricted class of student classifiers—in effect, limiting applicability to logistic regression with convex loss. Also, unlike the methods of Abadi et al. (2016), which represent the state-of-the-art in differentially-private deep learning, our techniques make no assumptions about details such as batch selection, the loss function, or whether optimization is by gradient descent or otherwise. Even so, as we show in experiments on MNIST and SVHN, our techniques provide a privacy/utility tradeoff that equals or improves upon bespoke learning methods such as those of Abadi et al. (2016).

Appendix A further discusses the related work—on top of which our contributions are as follows:

- We demonstrate a general machine learning strategy that provides differential privacy for training data in a "black-box" manner, i.e., independent of the learning algorithm.

![](images/914661f020ef9e87671de205eca28fbf9803a3484e093349f020e7b58e6542aa.jpg)  
Figure 1: Overview of the approach: (1) an ensemble of teachers is trained on disjoint subsets of the sensitive data, (2) a student model is trained on public data labeled using the ensemble.

- We improve upon the strategy outlined in Hamm et al. (2016) for learning machine models that protect training data privacy. In particular, our student only accesses the teachers' top vote and the model does not need to be trained with a restricted class of convex losses.  
- We explore four different approaches for reducing the student's dependence on its teachers, and show how the application of GANs to semi-supervised learning of Salimans et al. (2016) can greatly reduce the privacy loss by radically reducing the need for supervision.  
- We present a new application of the moments accountant technique from Abadi et al. (2016) for improving the differential-privacy analysis of knowledge transfer, which allows the training of students with meaningful privacy bounds.  
- We evaluate our framework on MNIST and SVHN, allowing for a comparison of our results with previous differentially private machine learning methods. Our classifiers achieve an  $(\varepsilon, \delta)$  differential-privacy bound of  $(2.04, 10^{-5})$  for MNIST and  $(8.19, 10^{-6})$  for SVHN, respectively with accuracy of  $98.00\%$  and  $90.66\%$ . In comparison, for MNIST, Abadi et al. (2016) obtain a looser  $(8, 10^{-5})$  privacy bound and  $97\%$  accuracy. For SVHN, Shokri & Shmatikov (2015) report approx.  $92\%$  accuracy with  $\varepsilon > 2$  per each of 300,000 model parameters, naively making the total  $\varepsilon > 600,000$ , which guarantees no meaningful privacy.

Our results are encouraging, and highlight the benefits of combining a learning strategy based on semi-supervised knowledge transfer with a precise, data-dependent privacy analysis. However, the most appealing aspect of this work is probably that its guarantees can be compelling to both an expert and a non-expert audience. In combination, our techniques simultaneously provide both an intuitive and a rigorous guarantee of training data privacy, without sacrificing the utility of the targeted model. This gives hope that users will increasingly be able to confidently and safely benefit from machine learning models built from their sensitive data.

# 2 PRIVATE LEARNING WITH ENSEMBLES OF TEACHERS

In this section, we introduce the specifics of our approach, of which we give an overview in Figure 1. We describe how the data is partitioned to train an ensemble of teachers, and how the predictions made by this ensemble are noisily aggregated. In addition, we provide elements justifying the superiority of using GANs to learn the student model with few queries made to the ensemble.

# 2.1 TRAINING THE ENSEMBLE OF TEACHERS

Data partitioning and teachers: Instead of training a single model to solve the task associated with dataset  $(X,Y)$ , where  $X$  denotes the set of inputs, and  $Y$  the set of labels, we partition the data in  $n$  disjoint sets  $(X_{n},Y_{n})$  and train a model separately on each set. As evaluated in Section 4.1, assuming that  $n$  is not too large with respect to the dataset size and task complexity, we obtain  $n$  classifiers  $f_{i}$  called teachers. We then deploy them as an ensemble making predictions on unseen inputs  $x$  by querying each teacher for a prediction  $f_{i}(x)$  and aggregating these into a single prediction.

Aggregation: The privacy guarantees of this teacher ensemble stems from its aggregation. Let  $m$  be the number of classes in our task. The label count for a given class  $j \in 1..m$  and an input  $\vec{x}$  is

the number of teachers that assigned class  $j$  to input  $\vec{x}$ :  $n_j(\vec{x}) = |\{i : i \in 1..n, f_i(\vec{x}) = j\}|$ . If we simply apply plurality—use the label with the largest count—the ensemble's decision may depend on a single teacher's vote. Indeed, when two labels have a vote count differing by at most one, there is a tie: the aggregated output changes if one teacher makes a different prediction. Therefore, we add random noise to the vote counts  $n_j$  to introduce ambiguity:

$$
f (x) = \arg \max  _ {j} \left\{n _ {j} (\vec {x}) + L a p \left(\frac {1}{\varepsilon}\right) \right\} \tag {1}
$$

In this equation,  $\varepsilon$  is a privacy parameter and  $Lap(b)$  the Laplacian distribution with location 0 and scale  $b$ . Smaller  $\varepsilon$  values provide stronger privacy guarantees but degrade the accuracy of the aggregation: the noise added can potentially change the respective order of label counts (see Section 4.1).

To continue providing strong privacy guarantees as successive queries are made, larger noise needs to be added. Hence, the model utility is subject to degradation. Furthermore, privacy guarantees do not hold when an adversary has access to the model parameters, or recovers them by making a large number of queries to the model. Indeed, as each teacher  $f_{i}$  was trained without taking into account privacy, it is conceivable that they have sufficient capacity to retain details of the training data. To address these limitations, we train another model, the student, using a fixed number of labels predicted by the teacher ensemble.

# 2.2 SEMI-SUPERVISED TRANSFER OF THE KNOWLEDGE FROM AN ENSEMBLE TO A STUDENT

We train a student on nonsensitive and unlabeled data, some of which we label using the aggregation mechanism. This student model is the one deployed, in lieu of the teacher ensemble, so as to fix the privacy loss to a value that is constant with respect to the number of user queries made to the student model. Indeed, the privacy loss is now determined by the number of queries made to the teacher ensemble during student training and does not increase as end-users query the deployed student model. Thus, the privacy of users who contributed to the original training dataset is preserved even if the student's architecture and parameters are public or reverse-engineered by an adversary.

We considered three classes of techniques to maximize the student's learning while decreasing the number of labels it needs to access: distillation, active learning, semi-supervised learning (see Appendix C). Here, we only describe the most successful: semi-supervised learning with GANs.

Training the student with GANs: The GAN framework involves two machine learning models, a generator and a discriminator. They are trained in a competing fashion, in what can be viewed as a two-player game (Goodfellow et al., 2014). The generator produces samples from the data distribution by transforming vectors sampled from a Gaussian distribution. The discriminator is trained to distinguish samples artificially produced by the generator from samples part of the real data distribution. Models are trained via simultaneous gradient descent steps on both players' costs with the goal of converging to the Nash equilibrium. In practice, such convergence is complex to achieve when the parameterization of each player's strategy is a non-convex function of the parameters, like a DNN. In their application of GANs to semi-supervised learning, Salimans et al. (2016) made the following modifications. The discriminator is extended from a binary classifier (data vs. generator sample) to a multi-class classifier (one of  $k$  classes of data samples, plus a class for generated samples). This classifier is then trained to classify real samples in one of the  $k$  classes, and the generated samples in the additional class. Although no formal results currently explain why yet, the technique was empirically demonstrated to greatly improve semi-supervised learning of classifiers on several datasets, especially when the classifier is trained with feature matching loss (Salimans et al., 2016).

Training the student in a semi-supervised fashion makes better use of the entire data available to the student, while still only labeling a subset of it. Unlabeled inputs are used in unsupervised learning to estimate a good prior for the distribution. Labeled inputs are then used for supervised learning.

# 3 PRIVACY ANALYSIS OF THE APPROACH

We now analyze the differential privacy guarantees of our approach. Specifically, we keep track of the privacy budget throughout the student's training using the moments accountant (Abadi et al., 2016). When teachers reach a strong quorum, this allows us to bound privacy costs more strictly.

# 3.1 DEFINING DIFFERENTIAL PRIVACY FOR AN INITIAL, NAIVE ANALYSIS

Differential privacy (Dwork et al., 2006b; Dwork, 2011) has established itself as a strong standard. It provides privacy guarantees for algorithms analyzing databases, which in our case is a machine learning training algorithm processing a training dataset. Differential privacy is defined using pairs of adjacent databases: in the present work, these are datasets that only differ by one training example. Recall the following variant of differential privacy introduced in Dwork et al. (2006a).

Definition 1. A randomized mechanism  $\mathcal{M}$  with domain  $\mathcal{D}$  and range  $\mathcal{R}$  satisfies  $(\varepsilon, \delta)$ -differential privacy if for any two adjacent inputs  $d, d' \in \mathcal{D}$  and for any subset of outputs  $S \subseteq \mathcal{R}$  it holds that:

$$
\operatorname * {P r} [ \mathcal {M} (d) \in S ] \leq e ^ {\varepsilon} \Pr [ \mathcal {M} (d ^ {\prime}) \in S ] + \delta . \tag {2}
$$

It will be useful to define the privacy loss and the privacy loss random variable. They capture the differences in the probability distribution resulting from running  $\mathcal{M}$  on  $d$  and  $d'$ .

Definition 2. Let  $\mathcal{M}\colon \mathcal{D}\to \mathcal{R}$  be a randomized mechanism and  $d,d^{\prime}$  a pair of adjacent databases. Let aux denote an auxiliary input. For an outcome  $o\in \mathcal{R}$ , the privacy loss at  $o$  is defined as:

$$
c \left(o; \mathcal {M}, \boldsymbol {a u x}, d, d ^ {\prime}\right) \triangleq \log \frac {\Pr \left[ \mathcal {M} \left(\boldsymbol {a u x} , d\right) = o \right]}{\Pr \left[ \mathcal {M} \left(\boldsymbol {a u x} , d ^ {\prime}\right) = o \right]}. \tag {3}
$$

The privacy loss random variable  $C(\mathcal{M}, \mathsf{aux}, d, d')$  is defined as  $c(\mathcal{M}(d); \mathcal{M}, \mathsf{aux}, d, d')$ .

A natural mean to bounding our approach's privacy loss is to first bound the privacy cost of each label queried by the student, and then use the strong composition theorem (Dwork et al., 2010) to derive the total cost of training the student. For neighboring databases  $d, d'$ , all teachers except one get the same training data so that the label counts  $n_j(\vec{x})$  for any example  $\vec{x}$ , on  $d$  and  $d'$  differ by at most 1 in at most two locations. In Appendix B, we show how this yields loose guarantees: even if the aggregation mechanism ensures that each label query is answered with  $(0.05, 10^{-6})$  differential privacy and the student is trained with only 1,000 labels, the total cost of training is still  $\varepsilon \approx 26$ . We improve on this bound using the moments accountant and a data-dependent privacy analysis.

# 3.2 THE MOMENTS ACCOUNTANT: A BUILDING BLOCK FOR BETTER ANALYSIS

To keep track of the privacy cost, we use recent advances in privacy cost accounting. The moments accountant was introduced by Abadi et al. (2016), building on previous work (Bun & Steinke, 2016; Dwork & Rothblum, 2016; Mironov, 2016). Our data-dependent privacy analysis takes advantage of the fact that when the quorum among the teachers is very strong, the majority outcome has overwhelming likelihood, in which case the privacy cost is small whenever this outcome occurs. The moments accountant allows us analyze the composition of such mechanisms in a unified framework.

Definition 3. Let  $\mathcal{M}\colon \mathcal{D}\to \mathcal{R}$  be a randomized mechanism and  $d,d^{\prime}$  a pair of adjacent databases. Let aux denote an auxiliary input. The moments accountant is defined as:

$$
\alpha_ {\mathcal {M}} (\lambda) \triangleq \max  _ {\boldsymbol {a u x}, d, d ^ {\prime}} \alpha_ {\mathcal {M}} (\lambda ; \boldsymbol {a u x}, d, d ^ {\prime}) \tag {4}
$$

where  $\alpha_{\mathcal{M}}(\lambda ;\mathsf{aux},d,d^{\prime})\triangleq \log \mathbb{E}[\exp (\lambda C(\mathcal{M},\mathsf{aux},d,d^{\prime}))]$  is the moment generating function.

# 3.3 OUR PRECISE, DATA-DEPENDENT PRIVACY ANALYSIS

We observe that the actual privacy cost of the aggregation mechanism is very small when the margin between the best and the second-best label is large: we refer to it as the gap. So our analysis is data-dependent and keeps track of the privacy cost with the moments accountant. We prove the following data-dependent bound on the moment generating function for the mechanism in Appendix B.

Theorem 1. Let  $\mathcal{M}$  be  $(2\gamma, 0)$ -differentially private and  $q \geq \operatorname*{Pr}[\mathcal{M}(d) \neq o^*]$  for some outcome  $o^*$ . Then for any  $\mathsf{aux}$  and any neighbor  $d'$  of  $d$ ,  $\mathcal{M}$  satisfies

$$
\alpha_ {\mathcal {M}} (l; \boldsymbol {a u x}, d, d ^ {\prime}) \leq \log ((1 - q) (\frac {1 - q}{1 - e ^ {2 \gamma} q}) ^ {l} + q \exp (2 \gamma l)).
$$

This bounds the  $\alpha$  for each step, and properties of the moments accountant allow us to add up such bounds and derive an  $(\varepsilon, \delta)$  guarantee from the final  $\alpha$  (see Theorem 2 in the appendix).

Since the privacy moments are themselves now data dependent, the final  $\varepsilon$  is itself data-dependent and should not be revealed. To get around this, one can bound the smooth sensitivity (Nissim et al., 2007a) of the moments and add noise proportional to it to the moments themselves. This gives us a differentially private estimate of the privacy cost itself. However, our evaluation ignores this overhead and reports the un-noised values of  $\varepsilon$ .

# 4 EVALUATION

We first train a teacher ensemble for each dataset. The trade-off between the accuracy and privacy of labels predicted by the ensemble is greatly dependent on the number of teachers it is made up of: being able to train a large set of teachers is essential to support the injection of noise yielding strong privacy guarantees while having a limited impact on accuracy. Second, we minimize the privacy budget spent on learning the student by training it with as few queries to the ensemble as possible.

Our experiments use MNIST and the extended SVHN. Our MNIST model stacks two convolutional layers with max-pooling and one ReLU layer. When trained on the entire dataset, the non-private model has a  $99.18\%$  test accuracy. For SVHN, we add two hidden layers. The non-private model achieves a  $92.8\%$  test accuracy, which is shy of the state-of-the-art. However, we are primarily interested in comparing the private student's accuracy with the one of a non-private model trained on the entire dataset, for different privacy guarantees. Results presented in this section are reproducible.

# 4.1 TRAINING AN ENSEMBLE OF TEACHERS PRODUCING PRIVATE LABELS

As mentioned above, compensating the noise introduced by the Laplacian mechanism presented in Equation 1 requires large ensembles. We evaluate the extent to which the two datasets considered can be partitioned with a reasonable impact on the performance of individual teachers. Specifically, we show that for MNIST and SVHN, we are able to train ensembles of 250 teachers. Their aggregated predictions are accurate despite the injection of large amounts of random noise to ensure privacy. The aggregation mechanism output has an accuracy of  $93.18\%$  for MNIST and  $87.79\%$  for SVHN, when evaluated on their respective test sets, while each query has a low privacy budget of  $\varepsilon = 0.05$ .

Prediction accuracy: The number  $n$  of teachers is limited by a trade-off between the classification task's complexity and the available data. We train  $n$  teachers by partitioning the training data  $n$ -way. Since datasets considered do not have a specific ordering, this is equivalent to random partitioning. Larger values of  $n$  allow for more random noise to be injected during aggregation, thus providing stronger privacy. We empirically find the maximum values of  $n$  for the MNIST and SVHN datasets by measuring the test set accuracy of each teacher trained on one of the  $n$  partitions of the training data. We find that even for  $n = 250$ , the average test accuracy of individual teachers is  $83.86\%$  for MNIST and  $83.18\%$  for SVHN. The larger size of SVHN compensates its increased task complexity.

Prediction confidence: As outlined in Section 3, the privacy of predictions made by an ensemble of teachers intuitively requires that a quorum of teachers generalizing well agree on identical labels. This observation is reflected by our data-dependent privacy analysis, which provides stricter privacy bounds when the quorum is strong. We study the disparity of labels assigned by teachers. In other words, we count the number of teacher votes for each possible label and sort labels by decreasing number of votes. We then measure how many additional votes were received by the top label compared to the second label, i.e. the gap. If the gap is small, introducing noise during aggregation might change the label assigned from the first to the second. Figure 3 shows the gap normalized by the total number of teachers  $n$ . As  $n$  increases, the gap remains limited to less than  $20\%$  of the teachers, allowing for aggregation mechanisms to output the correct label in the presence of noise.

Noisy aggregation: For MNIST and SVHN, we consider three ensembles of teachers with varying number of teachers  $n \in \{10, 100, 250\}$ . For each of them, we perturb the vote counts with Laplacian

![](images/c3c06f115bcee1c4578cd38b408e6b5806b6abb77854f35d18f4b60921b830be.jpg)  
Figure 2: How much noise can be injected to a query? Accuracy of the noisy aggregation for three MNIST and SVHN teacher ensembles and varying  $\varepsilon$  value per query. The noise introduced to achieve a given  $\varepsilon$  scales inversely proportionally to the value of  $\varepsilon$ : small values of  $\varepsilon$  on the left of the axis correspond to large noise amplitudes and large  $\varepsilon$  values on the right to small noise.

![](images/b2d548b913f9eeba813bcd482b49df722d7f09b0cd2b781a58bb84b73022ae9e.jpg)  
Figure 3: How certain is the aggregation of teacher predictions? Gap between the number of votes assigned to the most and second most frequent labels normalized by the number of teachers in an ensemble. Larger gaps indicate that the ensemble is confident in assigning the labels, and will be robust to more noise injection. Gaps were computed by averaging over the test data.

noise of inversed scale  $\varepsilon$  ranging between 0.01 and 1. This choice is justified below in Section 4.2. We report in Figure 2 the accuracy of test set labels inferred by the noisy aggregation mechanism for these values of  $\varepsilon$ . Notice that the number of teachers needs to be large to compensate the impact of noise injection on the accuracy. This is explained by the teacher sensitivity, assumed to be equal to 1. In other words it accounts for the case where changing one point in the training set of a single teacher suffices to change the teacher's predictions. This worst-case analysis allows our approach to be independent of the learning algorithm: the model need not be trained in any specific way.

# 4.2 SEMI-SUPERVISED TRAINING OF THE STUDENT WITH PRIVACY

The noisy aggregation mechanism labels the student's unlabeled training set in a privacy-preserving fashion. To reduce the privacy budget spent on student training, we are interested in making as few label queries to the teachers as possible. We therefore use the semi-supervised training approach described previously. Our MNIST and SVHN students with  $(\varepsilon, \delta)$  differential privacy of  $(2.04, 10^{-5})$  and  $(8.19, 10^{-6})$  achieve accuracies of  $98.00\%$  and  $90.66\%$ . These results improve the differential privacy state-of-the-art for these datasets. Abadi et al. (2016) previously obtained  $97\%$  accuracy with a  $(8, 10^{-5})$  bound on MNIST. Shokri & Shmatikov (2015) reported about  $92\%$  accuracy on SVHN with  $\varepsilon > 2$  per model parameter and a model with over 300,000 parameters. Naively, this corresponds to a total  $\varepsilon > 600,000$ .

We apply semi-supervised learning with GANs to our problem using the following setup for each dataset. In the case of MNIST, the student has access to 9,000 samples, among which a subset of either 100, 500, or 1,000 samples are labeled using the noisy aggregation mechanism discussed in Section 2.1. Its performance is evaluated on the 1,000 remaining samples of the test set. Note that for the MNIST dataset, we randomly shuffle the test set to ensure that the different classes are balanced when selecting the (small) subset labeled to train the student. For SVHN, the student has access to 10,000 training inputs, among which it labels 500 or 1,000 samples using the noisy aggregation mechanism. Its performance is evaluated on the remaining 16,032 samples. For both datasets, the ensemble is made up of 250 teachers. We use Laplacian scale of 20 to guarantee an individual query privacy bound of  $\varepsilon = 0.05$ . These parameter choices are motivated by the results from Section 4.1. In Figure 4, we report the values of the  $(\varepsilon, \delta)$  differential privacy guarantees provided and the corresponding student accuracy, as well as the number of queries made by each student. The MNIST student is able to learn a  $98\%$  accurate model, which is shy of  $1\%$  when compared to the accuracy of a model learned with the entire training set, with only 100 label queries. This results in a strict differentially private bound of  $\varepsilon = 2.04$  for a failure probability fixed at  $10^{-5}$ . The

<table><tr><td>Dataset</td><td>ε</td><td>δ</td><td>Queries</td><td>Non-Private Baseline</td><td>Student Accuracy</td></tr><tr><td>MNIST</td><td>2.04</td><td>10-5</td><td>100</td><td>99.18%</td><td>98.00%</td></tr><tr><td>MNIST</td><td>8.03</td><td>10-5</td><td>1000</td><td>99.18%</td><td>98.10%</td></tr><tr><td>SVHN</td><td>5.04</td><td>10-6</td><td>500</td><td>92.80%</td><td>82.72%</td></tr><tr><td>SVHN</td><td>8.19</td><td>10-6</td><td>1000</td><td>92.80%</td><td>90.66%</td></tr></table>

Figure 4: Utility and privacy of the semi-supervised students: each row is a variant of the student model trained with generative adversarial networks in a semi-supervised way, with a different number of label queries made to the teachers through the noisy aggregation mechanism. The last column reports the accuracy of the student and the second and third column the bound  $\varepsilon$  and failure probability  $\delta$  of the  $(\varepsilon, \delta)$  differential privacy guarantee.

SVHN student achieves  $90.66\%$  accuracy, which is also comparable to the  $92.80\%$  accuracy of one teacher learned with the entire training set. The corresponding privacy bound is  $\varepsilon = 8.19$ , which is higher than for the MNIST dataset, most likely because of the larger number of queries made to the aggregation mechanism.

Note that these bounds are themselves sensitive and should not be released publicly directly as is. In Appendix B, we discuss how to protect bound values in the case of a real deployment. In addition, we observe that our private student outperforms the aggregation's output in terms of accuracy, with or without the injection of Laplacian noise. While this shows the power of semi-supervised learning, it may not perform as well on different kinds of data, where categories are not explicitly designed by humans to be salient in the input space: e.g., medical data.

# 5 CONCLUSIONS

To protect the privacy of sensitive training data, this paper has advanced a learning strategy and a corresponding privacy analysis based on knowledge aggregation and transfer from "teacher" models, trained on disjoint data, to a "student" model whose attributes may be made public. In combination, the paper's techniques demonstrably achieve excellent utility on the MNIST and SVHN benchmark tasks, while simultaneously providing a formal, state-of-the-art bound on users' privacy loss. While our results are not without limits—e.g., they require disjoint training data for a large number of teachers—they are encouraging, and highlight the advantages of combining semi-supervised learning with precise, data-dependent privacy analysis, which will hopefully trigger further work. In particular, such future work may investigate whether or not our semi-supervised approach will also reduce teacher queries for tasks other than MNIST and SVHN, for example when the discrete output categories are not as distinctly defined by the salient input space features.

A key advantage is that this paper's techniques establish a precise guarantee of training data privacy in a manner that is both intuitive and rigorous. Therefore, they can be appealing, and easily explained, to both an expert and non-expert audience. However, perhaps equally compelling are the techniques' generality and wide applicability. Both our learning approach and our analysis methods are "black-box," i.e., independent of the learning algorithm for either teachers or students, and therefore apply, in general, to non-convex, deep learning, and other learning methods. Also, because our techniques do not constrain the selection or partitioning of training data, they apply when training data is naturally and non-randomly partitioned—e.g., because of privacy, regulatory, or competitive concerns—or when each teacher is trained in isolation, with a different method. We look forward to further such applications, for example on RNNs and other sequence-based models.

# ACKNOWLEDGMENTS

Nicolas Papernot is supported by a Google PhD Fellowship in Security. The authors would like to thank Ilya Mironov and Li Zhang for insightful discussions about early drafts of this document.

# REFERENCES

Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In CCS, 2016. arXiv preprint

arXiv:1607.00133.  
Charu C Aggarwal. On k-anonymity and the curse of dimensionality. In Proceedings of the 31st international conference on Very large data bases, pp. 901-909. VLDB Endowment, 2005.  
Babak Alipanahi, Andrew Delong, Matthew T Weirauch, and Brendan J Frey. Predicting the sequence specificities of dna-and rna-binding proteins by deep learning. Nature biotechnology, 2015.  
Dana Angluin. Queries and concept learning. Machine learning, 2(4):319-342, 1988.  
Raef Bassily, Adam Smith, and Abhradeep Thakurta. Differentially private empirical risk minimization: Efficient algorithms and tight error bounds. arXiv preprint arXiv:1405.7085, 2014.  
Eric B Baum. Neural net algorithms that learn in polynomial time from examples and queries. IEEE Transactions on Neural Networks, 2(1):5-19, 1991.  
Leo Breiman. Bagging predictors. Machine Learning, 24(2):123-140, 1994.  
Jane Bromley, James W Bentz, Léon Bottou, Isabelle Guyon, Yann LeCun, Cliff Moore, Eduard Säckinger, and Roopak Shah. Signature verification using a "siamese" time delay neural network. International Journal of Pattern Recognition and Artificial Intelligence, 7(04):669-688, 1993.  
Cristian Bucilua, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 535-541. ACM, 2006.  
Mark Bun and Thomas Steinke. Concentrated differential privacy: Simplifications, extensions, and lower bounds. In Proceedings of TCC, 2016.  
Kamalika Chaudhuri and Claire Monteleoni. Privacy-preserving logistic regression. In Advances in Neural Information Processing Systems, pp. 289-296, 2009.  
Kamalika Chaudhuri, Claire Monteleoni, and Anand D Sarwate. Differentially private empirical risk minimization. Journal of Machine Learning Research, 12(Mar):1069-1109, 2011.  
Thomas G Dietterich. Ensemble methods in machine learning. In International workshop on multiple classifier systems, pp. 1-15. Springer, 2000.  
Cynthia Dwork. A firm foundation for private data analysis. Communications of the ACM, 54(1): 86-95, 2011.  
Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Foundations and Trends in Theoretical Computer Science, 9(3-4):211-407, 2014.  
Cynthia Dwork and Guy Rothblum. Concentrated differential privacy. CoRR, abs/1603.01887, 2016. URL http://arxiv.org/abs/1603.01887.  
Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Advances in Cryptology-EUROCRYPT 2006, pp. 486-503. Springer, 2006a.  
Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography, pp. 265-284. Springer, 2006b.  
Cynthia Dwork, Guy N. Rothblum, and Salil Vadhan. Boosting and differential privacy. In 2010 IEEE 51st Annual Symposium on Foundations of Computer Science, pp. 51-60. IEEE, oct 2010. ISBN 978-1-4244-8525-3. doi: 10.1109/FOCS.2010.12. URL http://dl.acm.org/citation.cfm?id=1917827.1918366.  
Ulfrar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. Rappor: Randomized aggregatable privacy-preserving ordinal response. In Proceedings of the 2014 ACM SIGSAC conference on computer and communications security, pp. 1054-1067. ACM, 2014.

Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pp. 1322-1333. ACM, 2015.  
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pp. 2672-2680, 2014.  
Jihun Hamm, Paul Cao, and Mikhail Belkin. Learning privately from multiparty data. arXiv preprint arXiv:1602.03552, 2016.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Anjuli Kannan, Karol Kurach, Sujith Ravi, Tobias Kaufmann, Andrew Tomkins, Balint Miklos, Greg Corrado, et al. Smart reply: Automated response suggestion for email. In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), volume 36, pp. 495-503, 2016.  
Gregory Koch. Siamese neural networks for one-shot image recognition. PhD thesis, University of Toronto, 2015.  
Igor Kononenko. Machine learning for medical diagnosis: history, state of the art and perspective. Artificial Intelligence in medicine, 23(1):89-109, 2001.  
L. Sweeney. k-anonymity: A model for protecting privacy. volume 10, pp. 557-570. World Scientific, 2002.  
Ilya Mironov. Renyi differential privacy. manuscript, 2016.  
Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. Smooth sensitivity and sampling in private data analysis. In Proceedings of the Thirty-ninth Annual ACM Symposium on Theory of Computing, STOC '07, pp. 75-84, New York, NY, USA, 2007a. ACM. ISBN 978-1-59593-631-8. doi: 10.1145/1250790.1250803. URL http://doi.acm.org/10.1145/1250790.1250803.  
Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. Smooth sensitivity and sampling in private data analysis. In Proceedings of the thirty-ninth annual ACM symposium on Theory of computing, pp. 75-84. ACM, 2007b.  
Manas Pathak, Shantanu Rane, and Bhiksha Raj. Multiparty differential privacy via aggregation of locally trained classifiers. In Advances in Neural Information Processing Systems, pp. 1876-1884, 2010.  
Manas Pathak, Shantanu Rane, Wei Sun, and Bhiksha Raj. Privacy preserving probabilistic inference with hidden markov models. In 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5868-5871. IEEE, 2011.  
Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. arXiv preprint arXiv:1606.03498, 2016.  
Reza Shokri and Vitaly Shmatikov. Privacy-preserving deep learning. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pp. 1310-1321. ACM, 2015.  
Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE, pp. 245-248. IEEE, 2013.  
Latanya Sweeney. Weaving technology and policy together to maintain confidentiality. The Journal of Law, Medicine & Ethics, 25(2-3):98-110, 1997.  
Martin J Wainwright, Michael I Jordan, and John C Duchi. Privacy aware learning. In Advances in Neural Information Processing Systems, pp. 1430-1438, 2012.  
Stanley L Warner. Randomized response: A survey technique for eliminating evasive answer bias. Journal of the American Statistical Association, 60(309):63-69, 1965.
