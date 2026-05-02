# LEARNING GLOBAL ADDITIVE EXPLANATIONS FOR NEURAL NETS USING MODEL DISTILLATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Interpretability has largely focused on local explanations, i.e. explaining why a model made a particular prediction for a sample. These explanations are appealing due to their simplicity and local fidelity. However, they do not provide information about the general behavior of the model. We propose to leverage model distillation to learn global additive explanations that describe the relationship between input features and model predictions. These global explanations take the form of feature shapes, which are more expressive than feature attributions. Through careful experimentation, we show qualitatively and quantitatively that global additive explanations are able to describe model behavior and yield insights about models such as neural nets. A visualization of our approach applied to a neural net as it is trained is available at https://youtu.be/ErQYwNqzEdc.

# 1 INTRODUCTION

Recent research in interpretability has focused on developing local explanations: given an existing model and a sample, explain why the model made a particular prediction for that sample (Ribeiro et al., 2016). The accuracy and quality of these explanations have rapidly improved, and they are becoming important tools to understand model decisions for individual samples. However, the human cost of examining multiple local explanations can be prohibitive with today's large data sets, and it is unclear whether multiple local explanations can be aggregated without contradicting each other (Ribeiro et al., 2018; Alvarez-Melis & Jaakkola, 2018).

In this paper, we are interested in global explanations that describe the overall behavior of a model. While usually not as accurate as local explanations on individual samples, global explanations provide a different, complementary view of the model. They allow us to clearly visualize trends in feature space, which is useful for key tasks such as understanding which features are important, detecting unexpected patterns in the training data and debugging errors learned by the model.

We propose to use model distillation techniques (Bucilua et al., 2006; Hinton et al., 2015) to learn global additive explanations of the form

$$
\hat {F} (\mathbf {x}) = h _ {0} + \sum_ {i} h _ {i} \left(x _ {i}\right) + \sum_ {i \neq j} h _ {i j} \left(x _ {i}, x _ {j}\right) + \sum_ {i \neq j} \sum_ {j \neq k} h _ {i j k} \left(x _ {i}, x _ {j}, x _ {k}\right) + \dots \tag {1}
$$

to approximate the prediction function of the model,  $F(\mathbf{x})$ . Figure 1 illustrates our approach. The output of our approach is a set of  $p$  feature shapes  $\{h_i\}_1^p$  that can be visually inspected, used for feature attribution, and composed to form an explanation model that can be quantitatively evaluated. Through controlled experiments, we empirically validate that these feature shapes provide accurate and interesting insights into the behavior of complex models. In this paper, we focus on interpreting  $F$  from fully-connected neural nets trained on tabular data.

Our goal is not to replace local explanations, nor to explain how the model functions internally. What we claim is that we can complement local explanations with global additive explanations that visualize the input-output relationship between features and predictions. Our contributions are:

- We propose to learn global additive explanations for complex, non-linear models such as neural nets. These explanations do not aim at competing with local explanations, and instead complement them to shed a different light into the models.

![](images/3af74adea5f7de446140ebffe1fd10adff99a37c6c7edc012a617b78a10ec25a.jpg)  
Figure 1: Given a black box model and unlabeled samples (new unlabeled data or training data with labels discarded), our approach leverages model distillation to learn feature shapes that describe the relationship between input features and model predictions.

- We leverage powerful tree- and spline-based additive models in a model distillation setting to learn global feature shapes that are more expressive than feature attributions.  
- We quantitatively evaluate different global explanation methods in terms of fidelity to the model being explained and accuracy on independent test data.  
- Through controlled experiments, we show that these global explanations can provide accurate and interesting insights into the behavior of complex models.

# 2 LEARNING GLOBAL ADDITIVE EXPLANATIONS

Although our approach of using model distillation with powerful additive models of the form in equation 1 is new, our work is based on two previous research threads: (1) decomposing  $F$  into additive  $\hat{F}$  to understand how  $F$  is affected by its inputs (e.g. Hooker (2007)), and (2) learning an interpretable model (often some form of decision tree) to mimic  $F$  (e.g. Craven & Shavlik (1995)).

# 2.1 ADDITIVE  $\hat{F}$

Global additive explanations have been used to analyze inputs to complex, nonlinear mathematical models and computer simulations (Sobol, 2001), analyze how hyperparameters affect the performance of machine learning algorithms (Hutter et al., 2014), and decompose prediction functions into lower-dimensional components (Hooker, 2004). They are determined by the choice of metric  $L$  between  $F$  and its approximation  $\hat{F}$ , degree  $d$  of highest order components ( $d = 3$  in equation 1, and type of base learner  $h$ ). One common theme shared by these methods is that they decompose  $F$  into  $\hat{F}$  using numerical or computational methods (e.g. matrix inversion, quasi Monte Carlo).

Rather than approximately decomposing  $\hat{F}$  (which can be prohibitively expensive with large  $n$  or  $p$ ), we propose to learn  $\hat{F}$  using model distillation. This is equivalent to choosing  $L$  that minimizes the empirical risk between the prediction function  $F$  and our global additive explanation  $\hat{F}$  on the training data. By limiting  $d = 1$ , we investigate how much of  $F$  can be approximated by feature shapes of main components  $h_i$  alone, and study the impact of each input feature on  $F$ . To minimize approximation error  $\| F - \hat{F} \|_L$ , we select two flexible, nonparametric base learners for  $h$ : splines (Wood, 2006) and bagged trees. This gives us two global additive explanation models: Student Bagged Additive Boosted Trees (SAT) and Student Additive Splines (SAS). Other choices of  $h$  are possible. We describe our distillation setup to learn these models in Section 2.3.

# 2.2 INTERPRETABLE BUILDING BLOCKS OF  $\hat{F}$ : FEATURE SHAPES

Our global additive explanation models, SAT and SAS, can be visualized as feature shapes (Figure 1). These are plots with x-axis being the domain of input feature  $x_{i}$  and y-axis being the feature's contribution to the prediction  $h_i(x_i)$ . This class of model is also used in other work in interpretability that learn models from the original data (i.e. without distillation) with feature shapes that fulfill monotonicity (Gupta et al., 2016) or concavity/convexity (Pya & Wood, 2015) constraints.

How are feature shapes different from feature attribution? A classic way to interpret black-box models is feature attribution/importance measures. Examples include permutation-based measures

(Breiman, 2001), gradients/saliency (see Montavon et al. (2017) or Ancona et al. (2018) for a review), and measures based on variance decomposition (Iooss & Lemaitre, 2015), game theory (Datta et al., 2016; Lundberg & Lee, 2017), etc. We highlight that feature shapes are different from and more expressive than feature attributions. Feature attribution is a single number describing the feature's contribution to either the prediction of one sample (local) or the model (global), whereas our feature shapes describe the contribution of a feature, across the entire domain of the feature, to the model. Nonetheless, feature attribution, both global and local, can be automatically derived from feature shapes: global feature attribution by averaging feature shape values at each unique feature value; local feature attribution by simply taking one point on the feature shape.

# 2.3 LEARNING  $\hat{F}$  USING MODEL DISTILLATION

Model distillation was originally proposed to transfer knowledge from a large, complex model (teacher) to a faster, simpler model (student) without significant loss in prediction accuracy (Bucilua et al., 2006; Ba & Caruana, 2014; Hinton et al., 2015). We use model distillation for a different purpose: to learn global explanations for the teacher model. Neural nets and other black-box teachers have been distilled into interpretable models such as trees (Craven & Shavlik, 1995; Che et al., 2016; Frosst & Hinton, 2017; Bastani et al., 2017), rules (Ribeiro et al., 2018) and sets (Lakkaraju et al., 2017). An advantage of using additive student models over these models is that our feature shapes have automatic feature attribution, unlike e.g. decision trees (White & Liu, 1994).

Training teacher neural nets. Our teacher models are fully-connected nets with ReLU nonlinearities. We use the Adam optimizer (Kingma & Ba, 2015) with Xavier initialization (Glorot & Bengio, 2010) and early stopping based on validation loss. At each depth, we search for optimal hyperparameters (number of hidden units, learning rate, weight decay, dropout probability, batch size, enabling batch norm) based on average validation performance on multiple train-validation splits and random initializations. The most accurate nets we trained are fully-connected models with 2-hidden layers and 512 hidden units per layer (2H-512,512); nets with three or more hidden layers had lower training loss, but did not generalize as well and had worse validation loss. In some experiments we also use a restricted-capacity model with 1 hidden layer of 8 units (1H-8) to compare explanations.

Training student additive explanation models. To train SAT and SAS, we find optimal feature shapes  $\{h_i\}_{1}^{p}$  that minimize the mean square error between the teacher  $F$  and the student  $\hat{F}$ , i.e.

$$
L \left(h _ {0}, h _ {1}, \dots , h _ {p}\right) = \frac {1}{T} \sum_ {t = 1} ^ {T} \| F \left(x ^ {t}\right) - \hat {F} \left(x ^ {t}\right) \| _ {2} ^ {2} = \frac {1}{T} \sum_ {t = 1} ^ {T} \| F \left(x ^ {t}\right) - \left(h _ {0} + \sum_ {i = 1} ^ {p} h _ {i} \left(x _ {i} ^ {t}\right)\right) \| _ {2} ^ {2}, \tag {2}
$$

where  $F(x)$  is the output of the teacher model (scores for regression tasks and logits for classification tasks),  $T$  is the number of training samples,  $x^t$  is the t-th training sample, and  $x_i^t$  is its i-th feature. The exact optimization details depend on the choice of  $h$ . For trees we use cyclic gradient boosting (Buhlmann & Yu, 2003; Lou et al., 2012) which learns the feature shapes in a cyclic manner. As trees are high-variance, low-bias learners (Hastie et al., 2001), when used as base learners in additive models, it is standard to bag multiple trees (Lou et al., 2012; 2013; Caruana et al., 2015). We follow that approach here. For splines, we use cubic regression splines trained using penalized maximum likelihood in R's mgcv library (Wood, 2011) and cross-validate the splines' smoothing parameters.

In most of this paper, our learned explanations  $\hat{F}$  are composed of main components  $h_i$ . Higher order components  $h_{ij}$ ,  $h_{ijk}$  can increase the accuracy of  $\hat{F}$ , but make interpretation more difficult because we no longer get one shape per input feature and some shapes now have three or more dimensions. When  $\hat{F}$  consists of only main components  $h_i$ , any pairwise or higher order interactions in  $F$  are expressed as a best-fit additive approximation added to main components  $h_i$ , plus a pure-interaction residual. We show examples of this expression in Section 4.1, and in Section 4.5 show an example of an explanation  $\hat{F}$  that includes higher-order interaction components  $h_{ij}$  and  $h_{ijk}$ .

# 3 EVALUATING GLOBAL EXPLANATIONS

Lundberg & Lee (2017) suggested the perspective of viewing an explanation of a model's prediction as a model itself. With this perspective, we propose to quantitatively evaluate explanation models as if they were models. Specifically, we evaluate not just fidelity (how well the explanation matches

the teacher's predictions) but also accuracy (how well the explanation predicts the original label). Note that Lundberg & Lee (2017) and Ribeiro et al. (2016) evaluated local fidelity (called local accuracy by Lundberg & Lee (2017)), but not accuracy. A similar evaluation of global accuracy was performed by Kim et al. (2016) who used their explanations (prototypes) to classify test data. In our case, we use the feature shapes generated by our approach to predict on independent test data.

Baselines. We compare to three other explanation methods commonly used for tabular data: partial dependence (Friedman, 2001) as well as two local methods that we first adapt to the global setting: Shapley additive explanations (Lundberg & Lee, 2017) and linearization through gradients.

Partial dependence (PD) is a classic global explanation method that estimates how predictions change as feature  $x_{j}$  varies over its domain:  $PD(x_{j} = z) = \frac{1}{T}\sum_{t = 1}^{T}F((x_{1}^{t},\ldots ,x_{j}^{t} = z,\ldots ,x_{p}^{t})$  where the neural net is queried with new data samples generated by setting the value of their  $x_{j}$  feature to  $z$ , a value in the domain of  $x_{j}$ . Plotting  $PD(x_{j} = z)$  by  $z$  returns a feature shape.

Linearization through gradient approximation (GRAD). We construct the additive function  $G$  through the Taylor decomposition of  $F$ , defining  $G(x) = F(0) + \sum_{i=1}^{p} \frac{\partial F(x)}{\partial x_i} x_i$ , and defining the attribution of feature  $i$  of value  $x_i$  as  $\frac{\partial F(x)}{\partial x_i} x_i$ . This formulation is related to the "gradient*input" method (e.g. Shrikumar et al. (2017)) used to generate saliency maps for images.

Shapley additive explanations (SHAP). SHAP is a state-of-the-art local explanation method that satisfies several desirable local explanation properties (Lundberg & Lee, 2017). Given a sample and its prediction, SHAP decomposes the prediction additively between features using a game-theoretic approach. We use the python package by the authors of SHAP.

Both GRAD and SHAP provide local explanations that we adapt to a global setting by averaging the generated local attributions at each unique feature value. For example, the global attribution for feature "Temperature" at value 10 is the average of local attribution "Temperature" for all training samples with "Temperature=10". This is the red line passing through the points in Figure 2. Apply

ing this procedure to GRAD and SHAP's local attributions, we obtain global attributions gGRAD and gSHAP that we can now plot as feature shapes and evaluate quantitatively.

![](images/a6a34e890a010275a1aa1aff5073ddb97e212198e7fc2ce198618d7f330ff797.jpg)  
Figure 2: From SHAP to gSHAP. Blue points are individual SHAP values; red line is gSHAP feature shape.

# 4 EXPERIMENTAL RESULTS

We divide our experimental evaluation into four parts. First, we generate global additive explanations of synthetic functions with known ground-truth feature shapes (Section 4.1). In the second part, we quantitatively evaluate our global additive explanations against other explanations (Section 4.2). Next, we further validate our explanations with controlled experiments on real data (Section 4.3). Finally, we discuss insights obtained from our explanations (Section 4.4).

# 4.1 VALIDATION USING SYNTHETIC DATA WITH KNOWN GROUND-TRUTH

For this experiment, we simulate data from synthetic functions with known ground-truth feature shapes, which allows us to test our predicted shapes. We are particularly interested in observing how predicted feature shapes differ for neural nets of different capacity trained on the same data. Our expectation is that for neural nets that are accurate, our predicted shapes would match the ground-truth feature shapes, independent of how the features are used internally by the net. On the other hand, predicted shapes of less accurate models should less accurately match ground-truth shapes.

We design an additive, highly nonlinear function combining components from synthetic functions proposed by Hooker (2004), Friedman & Popescu (2008) and Tsang et al. (2018):  $F_{1}(\mathbf{x}) = 3x_{1} + x_{2}^{3} - \pi^{x_{3}} + \exp (-2x_{4}^{2}) + \frac{1}{2 + |x_{5}|} + x_{6}\log (|x_{6}|) + \sqrt{2|x_{7}|} + \max (0,x_{7}) + x_{8}^{4} + 2\cos (\pi x_{8})$ . Like Tsang et al. (2018), we set the domain of all features to be  $\mathcal{U}(-1,1)$ . Like Friedman & Popescu

![](images/90ac6aaa8165fdc9bedfe3011ad63ddfc67ce16dcdb116522df4f71e21cc63e1.jpg)

![](images/49aae9e10f15b1b45a37df91161528804bd28a643c5278fe3d9602bfc376b00b.jpg)

![](images/2803a4ad7c7c8025a9ca0b4a8661f6908b768153525424cdd5e7f71ca28245e7.jpg)

![](images/9560513e175e37c7c1220bc93fda17764155df4878c69a597c348c049d21bc8b.jpg)

![](images/bfba6a3fc01c34cf8cdf99eb6c2e403c2b65184c8f7726801357b5d967b4bfc5.jpg)  
Figure 3: Four of ten feature shapes learned for  $F_{1}$  (top row) and  $F_{2}$  (bottom row) by small (1H-8) and large (2H-512,512) nets. To view shapes learned for all features, see the Appendix.

![](images/cc6a49cbf859cefc1e1554939cdd27352fc353da6cc90fdcbf394bc87cc8af95.jpg)

![](images/8e591fac118a5472642d2061a32f7a859608141f391999376209eb9f4cb58a60.jpg)

![](images/41f4dcc9b02242e13b88870a20bbc7a88b92761b0def666af84cc51fa3c48339.jpg)

(2008), we add noise features to our samples that have no effect on  $F_{1}(x)$  via two noise features  $x_{9}$  and  $x_{10}$ . Over 50,000 samples, the mean of  $F_{1}(x)$  is 1.15, maximum is 8.65 and minimum is -6.62.

We started by training two teacher neural nets, 2H-512,512 and 1H-8 as described in Section 2.3. The high-capacity 2H neural net obtained test RMSE of 0.14, while the low-capacity neural net obtained test RMSE of 0.48, more than  $3\mathrm{x}$  larger. For each neural net, we used our approach to generate two global additive explanation models, SAT and SAS. These explanation models are faithful: the reconstruction RMSE of SAT is 0.14 for the 1H model and 0.08 for the 2H model, while the reconstruction RMSE of SAS is 0.14 for the 1H model and 0.07 for the 2H model. This suggests that both student methods should accurately represent the teacher, and that they probably will be very similar to each other.

Do SAT and SAS explain the teacher model, or just the original data? The top row of Figure 3 compares the feature shapes of our global explanation models SAT and SAS to function  $F_{1}$ 's analytic ground-truth feature shapes. SAT and SAS' feature shapes are almost identical. More importantly, it is clear that the feature shapes for the 2H model are different from shapes for the 1H model, and that the shapes for the 2H model

<table><tr><td>Model</td><td>Easy</td><td>All</td><td>Hard</td></tr><tr><td>1H-8</td><td>0.42</td><td>0.48</td><td>-</td></tr><tr><td>2H-512,512</td><td>-</td><td>0.14</td><td>0.17</td></tr></table>

Table 1: RMSE error of the teacher models on "easy" and "hard" samples chosen through the predicted attribution.

better match ground-truth shapes. In general, the shapes of the 2H model are very faithful to the ground-truth shapes, but sometimes fall short when there are sharp changes in the ground-truth, highlighting the limitations of a 2-hidden-layer neural net (which achieves 0.14 test RMSE, as noted before). On the other hand, both SAT and SAS' feature shapes for the 1H neural net show a less accurate teacher model that captures the gist of the ground-truth function but not its details, which is consistent with the original teacher RMSE of 0.48. This showcases that our methods fit what the teacher model has learned, and not the original data, and that when the teacher model is accurate the learned shapes match the ground-truth shapes.

Do SAT and SAS' feature shapes match the real behavior of the model? To further validate this we use the feature shapes to predict which samples will be inaccurately predicted by the teacher model. Specifically, we sample testing points with feature values where the feature shape of the 2H model is less accurate according to the feature shape ground-truth (for example, with  $x_{4}, x_{5}, x_{7} = 0$  and  $x_{6} = 0.3$ ) and evaluate them using the teacher model. If the learned feature shapes correctly represent the teacher model, the teacher should also be less accurate on those points than on other points where the learned and ground-truth feature shapes match. Similarly, by sampling points where the feature shapes of the 1H model and the ground-truth overlap, we would expect the error of the 1H teacher to be low. Indeed, as shown in Table 1, points sampled to be easy or hard guided by the feature shapes lead to lower and higher RMSE error, respectively, providing more evidence that our learned feature shapes are faithful.

How do interactions between features affect feature shapes? We design an augmented version of  $F_{1}$  to investigate how interactions in the teacher's predictions are expressed by feature shapes:  $F_{2}(\mathbf{x}) = F_{1}(\mathbf{x}) + x_{1}x_{2} + |x_{3}|^{2|x_{4}|} + \sec (x_{3}x_{5}x_{6})$ . We again simulate 50,000 samples. The mean of  $F_{2}(x)$  is 2.74, maximum is 11.48 and minimum is -4.46. Note that this function is much harder

![](images/86c97074d5118924acdd72379026cb3333406274001becaed1a5f5ec19d87513.jpg)  
Figure 4: Example feature shapes from Pneumonia (left), Magic (center), and Loan (right). SAT and SAS tend to agree. gSHAP, PD, and gGRAD capture the trend of the shape but not the details. Best seen on a screen.

![](images/84e584065112ec1996cf18fd994e266701e9b6e96ad574a5b845a5438cb48d8a.jpg)

![](images/776b9ece13e212161058f05e7dee076480cf544106fa94ff250fe5121419f31f.jpg)

to learn (the 2H model obtained an RMSE of 0.21) and also harder for students that do not model interactions to mimic (SAT and SAS obtain fidelity RMSEs of 0.35). The bottom row of Figure 3 displays features with interactions  $(x_4, x_6, x_2)$  and a feature without interactions  $(x_8)$ , and compares them with the shapes from  $F_1$ . For features  $x_4$  and  $x_6$ , the part of the interactions that can be approximated additively by  $h_i$ 's has been absorbed into the  $h_i$  feature shapes, changing their shapes as expected. On the other hand, we were still able to recover perfectly the feature shapes of features without interactions (e.g.  $x_8$ ). An interesting case study is  $x_2$ , where, despite the interaction, its feature shape has not changed. This is less surprising if we understand the feature shapes as the expected importance of the feature, learned in a data-driven fashion. The interaction term is  $x_1x_2$ , which, for  $x_1 \sim \mathcal{U}(-1,1)$ , has an expected value of zero, and therefore does not affect the feature shape. Similarly, the expected value of  $|x_3|^{2|x_4|}$  when  $x_3 \sim \mathcal{U}(-1,1)$  is  $1/(2|x_4| + 1)$ , an upward pointing cusp, which modifies the feature shape as shown in Figure 3 (bottom left figure).

# 4.2 COMPARING EXPLANATION METHODS ON REAL DATA

We selected five data sets to evaluate our approach: two UCI data sets (Bikeshare and Magic), a Loan risk scoring data set from an online lending company (LendingClub, 2011), the 2018 FICO Explainable ML Challenge's credit data set (FICO, 2018), and the pneumonia data set analyzed by Caruana et al. (2015). Table 2 provides details about the datasets and performance of the 1H and 2H teacher models.

<table><tr><td rowspan="2">Data</td><td rowspan="2">n</td><td rowspan="2">p</td><td rowspan="2">Type</td><td colspan="3">Performance</td></tr><tr><td>1H</td><td>2H</td><td></td></tr><tr><td>Bikeshare</td><td>17,000</td><td>12</td><td>Reg</td><td>RMSE</td><td>0.60</td><td>0.38</td></tr><tr><td>Loan</td><td>42,506</td><td>22</td><td>Reg</td><td>RMSE</td><td>2.71</td><td>1.91</td></tr><tr><td>Magic</td><td>19,000</td><td>10</td><td>Class</td><td>AUC</td><td>92.52</td><td>94.06</td></tr><tr><td>Pneumonia</td><td>14,199</td><td>46</td><td>Class</td><td>AUC</td><td>81.81</td><td>82.18</td></tr><tr><td>FICO</td><td>9,861</td><td>24</td><td>Class</td><td>AUC</td><td>79.08</td><td>79.37</td></tr></table>

Table 2: Performance of neural net teachers

# 4.2.1 QUANTITATIVE EVALUATION OF EXPLANATION METHODS

Table 3 presents the fidelity (how well does the student reproduce the teacher scores) and accuracy (how well does the student perform on the original task on independent test data) results for different global explanations of the 2H neural nets. We discuss results for the 1H neural nets in the Appendix. Accuracy is measured in terms of RMSE for regression tasks and AUROC for classification tasks, while fidelity is always measured as the RMSE between the student's predictions and the teacher's scores or logits (equation 2).

We draw several conclusions. First, SAT and SAS yield similar results in all cases, both in terms of accuracy and fidelity. In some cases, such as Magic, SAT (which uses tree base learners) can be more accurate, while in some others such as FICO, SAS (which uses spline base learners) may have the edge. Our interpretation is that trees are able to adapt better to sudden changes in shape than splines, but that also gives them more capacity to slightly overfit. We also see this in the feature shapes, where trees may be slightly more jagged than the splines, particularly in regions with fewer points. Figure 4 displays a few feature shapes for Pneumonia, Magic, and Loan. The feature shapes produced by PD tend to be much too smooth, which hurts its fidelity and accuracy. Second, in all cases, trees and splines have similar feature shapes and obtain equal or better accuracy and fidelity than the other methods. This is not surprising as the other methods are either local methods adapted to the global setting (gSHAP, gGRAD), or are global explanations that are not optimized to learn the teacher's predictions (PD). For reference, gSHAP when used as a local method (i.e. individual SHAP values, not global feature shapes) achieved a lower RMSE of 0.37 compared to 1.02 on

<table><tr><td>Accuracy
Global Explanation</td><td>Bikeshare
RMSE</td><td>Loan score
RMSE</td><td>Magic
AUC</td><td>Pneumonia
AUC</td><td>FICO
AUC</td></tr><tr><td>SAT</td><td>0.98 ± 0.00</td><td>2.35 ± 0.01</td><td>90.75 ± 0.06</td><td>82.24 ± 0.05</td><td>79.42 ± 0.04</td></tr><tr><td>SAS</td><td>0.98 ± 0.00</td><td>2.34 ± 0.00</td><td>90.58 ± 0.02</td><td>82.12 ± 0.04</td><td>79.51 ± 0.02</td></tr><tr><td>gGRAD</td><td>1.25 ± 0.00</td><td>6.04 ± 0.01</td><td>80.95 ± 0.13</td><td>81.88 ± 0.05</td><td>79.28 ± 0.02</td></tr><tr><td>gSHAP</td><td>1.02 ± 0.00</td><td>5.10 ± 0.01</td><td>88.98 ± 0.05</td><td>82.31 ± 0.03</td><td>79.36 ± 0.01</td></tr><tr><td>PD</td><td>1.00 ± 0.00</td><td>4.31 ± 0.00</td><td>82.78 ± 0.00</td><td>82.15 ± 0.00</td><td>79.47 ± 0.00</td></tr><tr><td>Fidelity
Global Explanation</td><td>Bikeshare
RMSE</td><td>Loan score
RMSE</td><td>Magic
RMSE</td><td>Pneumonia
RMSE</td><td>FICO
RMSE</td></tr><tr><td>SAT</td><td>0.92 ± 0.00</td><td>1.74 ± 0.01</td><td>1.78 ± 0.00</td><td>0.35 ± 0.00</td><td>0.15 ± 0.00</td></tr><tr><td>SAS</td><td>0.92 ± 0.00</td><td>1.71 ± 0.00</td><td>1.75 ± 0.00</td><td>0.35 ± 0.00</td><td>0.14 ± 0.00</td></tr><tr><td>gGRAD</td><td>1.20 ± 0.00</td><td>5.93 ± 0.01</td><td>2.93 ± 0.01</td><td>0.43 ± 0.00</td><td>0.16 ± 0.00</td></tr><tr><td>gSHAP</td><td>0.96 ± 0.00</td><td>4.83 ± 0.00</td><td>2.15 ± 0.00</td><td>0.46 ± 0.00</td><td>0.16 ± 0.00</td></tr><tr><td>PD</td><td>0.94 ± 0.00</td><td>3.85 ± 0.00</td><td>3.17 ± 0.00</td><td>0.47 ± 0.00</td><td>0.16 ± 0.00</td></tr></table>

Table 3: Accuracy and fidelity of global explanation models for 2H-512,512 neural nets on different datasets. For RMSE, lower is better. For AUC, higher is better. Results for 1H-8 nets in Appendix.

Bikeshare, and a lower RMSE of 1.99 compared to 5.10 on Loan, which is comparable to its 2H teacher's RMSE on test data (Table 2). The conclusion is that methods such as gSHAP excel at local explanations and should be used for those, but, to produce global explanations, global model distillation methods optimized to learn the teacher's predictions should be used instead.

# 4.3 VALIDATION USING CONTROLLED EXPERIMENTS ON REAL DATA

In this section we demonstrate the utility of global additive explanations on real data. Although here we do not have an analytic solution for the ground-truth feature shapes, we can still design experiments where we modify data in ways that will lead to expected known changes to the ground-truth feature shapes and then verify that these changes are captured in the learned feature shapes.

Label modification. In the bikeshare data, we added 1.0 to the label (the number of rented bikes) for samples where one of the features (humidity) is between 55 and 65. We then retrained a 2H neural net on the modified data, and applied our approach to learn feature shapes from the 2H net. Ideally, the feature shapes of that new neural net should be almost identical to those of the original net except in that particular range of the humidity feature, where we should see an abrupt "bump" that increases its feature shape value by one. Figure 5 (left) displays the feature shapes. Our method was able to recover the change to the label for the neural net in the new feature shape.

Data modification: expert discretization. Sometimes features are transformed before training. For example, in medical data, continuous variables such as body temperature may be discretized by domain experts into bins such as normal, mild fever, moderate fever, high fever, etc. In this experiment we test if our additive explanation models can recover these discretizations from the neural net without access to the discretized features. We train our student additive models using as input features the original un-discretized features, but using as labels the outputs of a neural net that was trained on discretized features. Our expectation is that if the student models are an accurate representation of what the neural net learned from the discretized features, they will detect the discretizations, even if they never have access to the discretized features or to the internal structure of the neural-net teacher. We study the feature shapes of two features in the Pneumonia data (Blood  $\mathrm{pO}_2$  and Respiration Rate) in Figure 5, where we compare the feature shapes learned from teachers trained on the original continuous data (dotted lines) with those from teachers trained on discretized features (solid lines). Recall that in both cases the student models only saw non-discretized features to generate feature shapes. Our approach captures the expected discretization intervals (in yellow) as described in Cooper et al. (1997).

# 4.4 INSIGHTS FROM GLOBAL ADDITIVE EXPLANATIONS

Checking for monotonicity. Domains such as credit scoring have regulatory requirements that prescribe monotonic relationships between predictions and some features (Federal Reserve Governors, 2007). For example, the 2018 FICO Explainable ML Challenge encouraged participants to impose monotonicity on 16 features (FICO, 2018). We use feature shapes to see if the function learned by the

![](images/eac75d15e42068559aa4cebffc41b4220511e575ecded6ba964fe78ecd38d115.jpg)  
Figure 5: Feature shapes from controlled experiments on real data. Left: Label modification experiment. Center and right: Data modification experiment. See details in Section 4.3.

![](images/d88c11d1bb3035a499dbcd5dd191f7865a7beb336869ff6718f158c702a934e0.jpg)

![](images/4fc933d79440c84cbd7a44cfc78d709890a8fffaa78b8b749555ddde26dce671.jpg)

neural net is monotone for these features. 15 of 16 features are monotonically increasing/decreasing as required. One feature, however, "Months Since Most Recent Trade Open" was expected to decrease monotonically, but actually increased monotonically. This is true not just in our explanations, but also in PD, gGRAD, and gSHAP (Figure A3). Note that testing for monotonicity requires global explanations or checking and aggregating many local explanations. See Appendix for details.

Visualizing neural net training: from underfit to overfit. Using additive models to peek inside a neural net creates many opportunities. For example, we can see what happens in the neural net when it is underfit or overfit; when it is trained with different losses such as squared, log, or rank loss or with different activation functions such as sigmoid or ReLUs; when regularization is performed with dropout or weight decay; when features are coded in different ways; etc. The video at https://youtu.be/ErQYwNqzEdc shows what is learned by a neural net as it trains on a medical dataset. The movie shows feature shapes for five features before, at, and after the early-stopping point as the neural net progresses from underfit to optimally fit to overfit. We had expected that the main cause of overfitting would be increased non-linearity (bumpiness) in the fitting function, but a significant factor in overfitting appears to be unwarranted growth in the confidence of the model as the logits grow more positive or negative than the early-stopping shape suggests is optimal.

# 4.5 EXTENDING  $\hat{F}$  TO INCLUDE INTERACTIONS

Functions learned by neural nets cannot always be represented with adequate fidelity by the additive function  $\hat{F}$  in equation 1. We can improve  $\hat{F}$ 's expressive power by adding pairwise and higher-order components  $h_{ij}, h_{ijk}$ , and so on to account for interactions between two or more input features. In Bikeshare, RMSE decreases from 0.98 to 0.60 when we add pairwise interactions to the student model. Figure 6 shows an interesting interaction between two features: "Time of Day", and "Working Day". On working days, the highest bike rental demand occurs at 7-9am and 5-7pm, but on weekends there is very low demand at 7-9am (presumably because

people are still sleeping) and at 5-7pm, and demand peaks during midday from 10am-4pm. These two features also form a three-way interaction with temperature. Because the teacher neural net learned these (and other) interactions, a global explanation method must also incorporate interactions if it is to provide high-fidelity explanations of the teacher model.

![](images/1cf642d00a95cb0ef8cb4d9d1bbaf7d2c806958a76bf32bcff71b16dc7643169.jpg)  
Figure 6: An important pairwise interaction in Bikeshare.

# 5 CONCLUSIONS

We present a method for "opening up" complex models such as neural nets trained on tabular data. The method is based on distillation with high-accuracy additive models to provide a global explanation of what a neural net has learned. Although in this paper we focus on explaining fully-connected neural nets, the method will work with any classification or regression model including CNNs and RNNs, but is not designed to work with raw inputs such as pixels where providing a global explanation in terms of the input pixels is not well-defined. We perform a battery of experiments to show that explanations generated by the method are faithful representations of the complex teacher model, and compared the method to other global explanation methods such as partial dependence, Shapley adapted to a global setting, and gradient methods. Our method is computationally efficient and requires only that the teacher neural net label a training set; it does not require repeated probing or access to the teacher model's internal structure or derivatives. We also show how our global explanations can be improved by adding pairwise and higher-order interactions to the explanations.

# REFERENCES

David Alvarez-Melis and Tommi S Jaakkola. Towards robust interpretability with self-explaining neural networks. In NIPS, 2018.  
Marco Ancona, Enea Ceolini, Cengiz ztiireli, and Markus Gross. Towards better understanding of gradient-based attribution methods for deep neural networks. In ICLR, 2018.  
Jimmy Ba and Rich Caruana. Do deep nets really need to be deep? In NIPS, 2014.  
Osbert Bastani, Carolyn Kim, and Hamsa Bastani. Interpreting blackbox models via model extraction. In FAT/ML Workshop, 2017.  
Leo Breiman. Random forests. Machine Learning, 2001.  
Cristian Bucilua, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In KDD, 2006.  
Peter Buhlmann and Bin Yu. Boosting with the 12 loss: regression and classification. Journal of the American Statistical Association, 2003.  
Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, and Noemie Elhadad. Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In KDD, 2015.  
Zhengping Che, Sanjay Purushotham, Robinder G. Khemani, and Yan Liu. Interpretable deep models for ICU outcome prediction. In AMIA Annual Symposium, 2016.  
Gregory F. Cooper, Constantin F. Aliferis, Richard Ambrosino, John M. Aronis, Bruce G. Buchanan, Rich Caruana, Michael J. Fine, Clark Glymour, Geoffrey J. Gordon, Barbara H. Hanusa, Janine E. Janosky, Christopher Meek, Tom M. Mitchell, Thomas S. Richardson, and Peter Spirtes. An evaluation of machine-learning methods for predicting pneumonia mortality. Artificial Intelligence in Medicine, 1997.  
Mark W. Craven and Jude W. Shavlik. Extracting tree-structured representations of trained networks. In NIPS, 1995.  
Anupam Datta, Shayak Sen, and Yair Zick. Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems. In IEEE Symposium on Security and Privacy, 2016.  
Federal Reserve Governors. Report to the congress on credit scoring and its effects on the availability and affordability of credit. 2007. URL https://www.federalreserve.gov/boarddocs/rptcongress/creditscore/creditscore.pdf.  
FICO. Explainable machine learning challenge, 2018. URL https://community.fico.com/s/explainable-machine-learning-challenge.  
Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 2001.  
Jerome H Friedman and Bogdan E Popescu. Predictive learning via rule ensembles. The Annals of Applied Statistics, 2008.  
Nicholas Frosst and Geoffrey Hinton. Distilling a neural network into a soft decision tree. arXiv preprint arXiv:1711.09784, 2017.  
Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In AISTATS, 2010.  
Maya Gupta, Andrew Cotter, Jan Pfeifer, Konstantin Voevodski, Kevin Canini, Alexander Mangylov, Wojciech Moczydlowski, and Alexander Van Esbroeck. Monotonic calibrated interpolated look-up tables. JMLR, 2016.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning. Springer, 2001.  
Geoff Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. NIPS Deep Learning Workshop, 2015.  
Giles Hooker. Discovering additive structure in black box functions. In KDD, 2004.  
Giles Hooker. Generalized functional ANOVA diagnostics for high dimensional functions of dependent variables. Journal of Computational and Graphical Statistics, 2007.  
Frank Hutter, Holger Hoos, and Kevin Leyton-Brown. An efficient approach for assessing hyperparameter importance. In ICML, 2014.  
Bertrand Iooss and Paul Lemaitre. A review on global sensitivity analysis methods. In Uncertainty Management in Simulation-Optimization of Complex Systems. 2015.  
Been Kim, Rajiv Khanna, and Oluwasanmi O Koyejo. Examples are not enough, learn to criticize! criticism for interpretability. In NIPS, 2016.  
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.  
Himabindu Lakkaraju, Ece Kamar, Rich Caruana, and Jure Leskovec. Interpretable and explorable approximations of black box models. In FAT/ML Workshop, 2017.  
LendingClub. Lending club loan data, 2011. URL https://www.lendingclub.com/info/download-data.action.  
Yin Lou, Rich Caruana, and Johannes Gehrke. Intelligible models for classification and regression. In KDD, 2012.  
Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker. Accurate intelligible models with pairwise interactions. In KDD, 2013.  
Scott M Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In NIPS, 2017.  
Grégoire Montavon, Wojciech Samek, and Klaus-Robert Müller. Methods for interpreting and understanding deep neural networks. Digital Signal Processing, 2017.  
Natalya Pya and Simon N Wood. Shape constrained additive models. Statistics and Computing, 2015.  
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "Why Should I Trust You?: Explaining the predictions of any classifier. In KDD, 2016.  
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. Anchors: High-precision model-agnostic explanations. In AAAI, 2018.  
Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning important features through propagating activation differences. In ICML, 2017.  
Ilya M Sobol. Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates. Mathematics and Computers in Simulation, 2001.  
Michael Tsang, Dehua Cheng, and Yan Liu. Detecting statistical interactions from neural network weights. In ICLR, 2018.  
Allan P White and Wei Zhong Liu. Bias in information-based measures in decision tree induction. Machine Learning, 1994.  
Simon N Wood. Generalized Additive Models: An Introduction with R. Chapman and Hall/CRC, 2006.  
Simon N Wood. Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. Journal of the Royal Statistical Society (B), 2011.
