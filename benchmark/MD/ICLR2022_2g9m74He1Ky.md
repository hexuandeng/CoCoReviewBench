# SPATIO-TEMPORAL DISENTANGLED REPRESENTATION LEARNING FOR MOBILITY FORECASTING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Spatio-temporal (ST) prediction task like mobility forecasting is of great significance to traffic management and public safety. There is an increasing number of works proposed for mobility forecasting problems recently, and they typically focus on better extraction of the features from the spatial and temporal domains. Although prior works show promising results on more accurate predictions, they still suffer in characterising and separating the dynamic and static components, making it difficult to make further improvements. Disentangled representation learning separates the learnt latent representation into independent variables associated with semantic factors (Duan et al., 2019). It offers a better separation of the spatial and temporal features, which could improve the performance of mobility forecasting models. In this work, we propose a VAE-based architecture for learning the disentangled representation from real spatio-temporal data for mobility forecasting. Our deep generative model learns a latent representation that (i) separates the temporal dynamics of the data from the spatially varying component and generates effective reconstructions; (ii) is able to achieve state-of-the-art performance across multiple spatio-temporal datasets. Moreover, we investigate the effectiveness of our method by eliminating the non-informative features from the learnt representations, and the results show that models can benefit from this operation.

# 1 INTRODUCTION

Spatio-temporal prediction tasks like mobility forecasting are critically important for smart city applications (Zheng et al., 2014). With the help of the rapid deployment of IoT (Internet of Things) devices and sensors, massive amounts of saturated ST datasets are available, and many researchers have put their efforts to improve the performance of ST prediction (Zhang et al., 2016; Wang et al., 2018; Zonoozi et al., 2018; Jin et al., 2018; Yao et al., 2019b). A central problem in deep learning for crowd flow prediction tasks is the extraction of features from both spatial and temporal domains. Researchers first tried to extract the entangled spatio-temporal features directly from the data (Zhang et al., 2016). Then, they tried to extract features from spatial and temporal domains separately. Wang et al. (2018) and Zonoozi et al. (2018) proposed models that extract periodic representations or short-term temporal features directly from the ST data using recurrent-based convolution operations and find their effectiveness in producing more accurate predictions. Based on that, in STDN (Yao et al., 2019b), the explicit extraction of the long-term periodic information was shown to also improve the prediction.

Although the existing approaches appear to be powerful in terms of results and predictions, two major challenges hinder these models: 1) Difficulty in characterising dynamic and spatial components. Spatio-temporal correlation is more complex since it comprises dependencies from both the spatial and temporal regions. Models like LDRSN (Tian et al., 2020) and RegionTrans (Wang et al., 2018) are capturing temporal dependencies explicitly. However, their complicated structure makes it hard to characterise and validate the effectiveness of their dynamic and spatial components. 2) Difficulty in separating the extraction of spatial and temporal features. Except for the spatial and temporal features from the most recent data, many approaches try to extract long-term/periodical features to improve the performance. However, without an explicit separation mechanism, the extracted long-term temporal dependencies introduce irrelevant noises to the predictor (factors do not vary with time and are only relevant to the long-term sequence). The extent to which generative models can model, extract and disentangle the spatial and temporal features in ST-raster data is an open problem.

To address the above challenges, we introduce the disentangled representation learning to the mobility forecasting task. Disentangled representation learning, which separates the learnt representation into independent variables such that each variable relates to one semantic factor of sensory data (Bengio et al., 2013), offers a solution for the problem mentioned above. For a spatio-temporal task like mobility forecasting, an ideal disentangled representation should have the ability to separate time-relevant components from the factors that don't vary with time, which can help improve the predictor's performance.

Many prior works have explored disentangled representation learning for spatio-temporal data (Hsu et al., 2017; Li & Mandt, 2018; Denton & Birodkar, 2017; Zhu et al., 2020). Most of them assume that the features can be disentangled into two sets: a set of dynamic features that extract the temporal correlations and a set of static features that describe factors that are constant through the input sequence. However, since most of them focus on the movement of some predefined objects, their assumption is inaccurate when applying to datasets for mobility forecasting. For example, a sudden car accident might show no relationship to the temporal influence from the previous timestep and the static traffic network structure. Therefore, there are still gaps in how to model the spatio-temporal data using the disentangled representation learning method. To address this problem, we assume that each timestep has its own temporal features and features that do not vary with time. By doing so, our model can better formulate the spatio-temporal data, and we can achieve controlled data generation frame by frame.

In this work, we proposed a VAE-based model to learn disentangled spatio-temporal representation. Compared to the conventional methods for the ST prediction problem, our approach extracts the entangled features first and then explicitly separates them into temporal variables and spatial variables using disentangled representation learning method. This will force the model to keep the learnt spatial/temporal features as mutually exclusive as possible. For applying disentangled representation learning method to real ST data, we assume that each frame has its own spatial and temporal variables and separate these two groups with auxiliary regularisation. It helps the model to formulate the complicated spatio-temporal sequence. Our experimental results (see Section 4) show that the learnt representations have a similar level of performance with the current state-of-the-art methods. Our key contributions can be summarised as follows:

1. We propose a novel approach to learn disentangled spatio-temporal representations for mobility forecasting tasks. The learnt representation is separated into two independent groups: spatial and temporal factors.  
2. We conducted several experiments on multiple spatio-temporal datasets and used the learnt representation for mobility forecasting. Results show that our methods achieve state-of-the-art performance compared to other baseline mobility forecasting methods.  
3. We investigate the effectiveness of our methods under the "Closeness, Period, Trend" scheme and how to further improve the model's performance by selecting the informative features from the learnt representations.

# 2 RELATED WORK

Deep spatial-temporal networks for mobility forecasting: In order to make accurate traffic prediction, many researchers have paid attention to capturing the spatio-temporal dependencies hidden behind the traffic data. Besides the conventional methods like Seasonal ARIMA (Moreira-Matias et al., 2013), deep learning methods are increasingly used in more and more works for mobility forecasting.

ST-ResNet (Zhang et al., 2016) was proposed to capture the spatial dependencies through a stack of residual convolution layers. It also stacks the frame from a near and a distant time period to capture temporal dynamics. The goal of using the residual units is to overcome the gradient vanishing problem, and the results show that capturing distant spatial features can improve the performance of mobility forecasting. DeepSTN+ (Lin et al., 2019) proposed a ResPlus unit that is capable of capturing long-range spatial correlations. Although they succeed in extracting distant spatial features, they lack attention on capturing the temporal dependencies. To further explore the effectiveness of temporal features, recurrent-based approaches were introduced to capture the temporal correlations. PCRN (Zonoozi et al., 2018) was proposed, which first extract entangled spatio-temporal representation

using a convolutional recurrent network (CRN) and then updating the periodic representations by CRN's hidden state. Attention-based LSTM methods are adopted by STDN (Yao et al., 2019b) and ST-DCCNAL (Li et al., 2019), which try to capture long-term temporal dependencies. In summary, our proposed method differs from other methods in the explicit separation of spatial and temporal features using a disentangled representation mechanism.

Disentangled Representation Learning for Spatio-temporal data: Most prior works on the disentangled representation learning problem are developed based on Variational Autoencoders (VAE) (Kingma & Welling, 2013), which is an unsupervised generative learning method.  $\beta$ -VAE, proposed by Higgins et al. (2016), forces the inference model to disentangle the latent representation by adding a new hyperparameter  $\beta$  to create an information bottleneck on the prior. FactorVAE (Kim & Mnih, 2018) further breaking down the objective function and try to enhance disentanglement by penalising the total correlation of the learnt representation. As for sequence modelling, a number of prior publications have extended VAE to video and speech data (Fabius & Van Amersfoort, 2014; Chung et al., 2015; Bayer & Osendorfer, 2014). These models, although being able to generate realistic sequences, do not explicitly disentangle the representation of time-invariant and time-dependent information. Thus it is inconvenient for these models to perform tasks such as controlled generation. S3VAE (Zhu et al., 2020) is proposed to separate static and dynamic factors of sequential data. Another approach proposed by (Li & Mandt, 2018) is also focusing on separating the dynamic factors from static factors. Although they share a similar idea which uses an RNN-based architecture to extract dynamic factors for each timestep, they use different prior setups. Each frame in (Li & Mandt, 2018) has its own content features while the time-invariant variables are shared by the whole sequence in S3VAE. As for the spatial-temporal type of data like video, SV2P (Babaeizadeh et al., 2017) uses the variational model to extract the time-invariant latent and make predictions for multiple frames. Models like (Denton & Birodkar, 2017; Hsieh et al., 2018) try to factorise each frame into a stationary part and a temporally dynamic component.

# 3 SPATIO-TEMPORAL VAE MODEL

# 3.1 SPATIO-TEMPORAL DATA

In this work, we will use  $\mathcal{D} = \{X^i\}^{i = 1:N}$  to denote a spatio-temporal raster dataset that comprises  $N$  i.i.d. sequences. Each  $X\equiv x_{1:T} = \{x_1,x_2,\dots,x_T\}$  in that dataset denotes a sequence of raster data with  $T$  frames. Since we focus on mobility forecasting tasks using grid-based spatial representations, which all have the similar dimensions  $H\times W$ . Hence each frame  $x_{t}\in \mathbb{R}^{H\times W}$  represents the mobility flow of a certain area at a given time interval  $t$ .

# 3.2 SEPARATING SPATIAL FEATURES FROM TEMPORAL FEATURES

Many prior works have explored disentangled representation learning for spatio-temporal data. Models like FHVAE (Hsu et al., 2017) tried to separate global variables from segment (dynamic) variables for speech data. DSVAE (Li & Mandt, 2018), DRNET (Denton & Birodkar, 2017) and S3VAE (Zhu et al., 2020) aim at the disentangled representation for video, which also factorised latent variables into static and dynamic parts. However, current approaches for learning the disentangled representation on sequence or spatio-temporal data like video assume that there is a fixed content or object shared by all frames in the sequence since they are using video datasets like the Stochastic Moving MNIST (Denton & Fergus, 2018) and Sprite (Li & Mandt, 2018). Sequences in these datasets often comprise images describing the movement of a set of the same numbers or virtual avatars. Under this context, static features describing this object can be extracted, and its performance can be evaluated by the accuracy of a classification task. However, for most of the real spatio-temporal datasets, the static features are not enough. For example, in a traffic flow dataset, in addition to the fixed structure of the traffic network and the temporal influences, there might be some hotspot suddenly emerges. At that specific time, the rise of those events shows no correlation to the temporal features and the fixed content. Therefore, we assume that for each frame, data is generated based on two sets of features: a set of temporal features which comprise the influence from the sequence before it and a set of spatial features that describe the structure of the network and local events. In this work, we propose a novel architecture that extracts mixed (entangled) feature maps for each timestep in the input sequence and then separates them into temporal and spatial variables. It allows us to analyse and controlled generate each frame separately.

![](images/416e83fc38226293453ef6b8ac521c384d1e6d20ba20561e60508b3512c06b5a.jpg)  
(a) The framework of our proposed model

![](images/84e19f2a0c74774e838c080cc7cb880a1e8f5489326c403aae64dc2c8979b799.jpg)  
Figure 1: The framework of our proposed model. Each frame of a spatio-temporal sequence  $x_{1:T}$  is fed into the Image Encoder first to extract entangled high-level feature maps, which are then passed through a temporal gated convolution layer to capture the time-varying variables (temporal features)  $z_{1:T}^{T_e}$ . Then the time-irrelevant variables (spatial features)  $z_t^{Sp}$  for each frame  $x_t$  is captured through several convolutional blocks. In addition to the objective of the VAE, a Total Correlation regularizer is used to encourage the disentanglement of the learnt representation. Reconstruction input is generated for each frame based on the sampled latent from both domains.  
(b) A detailed structure of the Feature Separation module

The architecture of our method for spatio-temporal disentangled representation learning is illustrated in Fig.1a. It consists of three major modules: (i) Image Encoder to yield initial feature maps from the input spatio-temporal sequence. (ii) A Feature Separation module to separate the spatial and temporal variables from the initially mixed feature maps. (iii) A Decoder to reconstruct the sequence frame by frame based on their corresponding spatial and temporal variables.

The Image Encoder comprises several layers of CNN, and the feature maps are extracted from each frame separately. The goal here is to yield some higher-level features for the next step. An Instance Normalisation layer is used after each convolutional layer to improve the reconstruction results since using batch normalisation might remove the instance-specific contrast information from the data, which is useful in the later step (Ulyanov et al., 2016).

Intuitively, without any disentanglement constraints, the feature maps that come out of the Image Encoder should be the entanglement of features from both spatial and temporal domain. Therefore, the goal for the Feature Separation module is to separate them explicitly. For temporal features, inspired by (Yu et al., 2017; Gehring et al., 2017), we use a Gated Convolutional structure to capture temporal dynamics from the mixed feature maps. This temporal gated convolution layer contains two 1-D causal convolution layers. As Fig.1b shows, those 1-D convolution operations will be applied on the time axis for each pixel of the feature maps, which means the capture of the temporal dynamics from each mixed high-level feature. It is much easier to understand the usage of the Image Encoder here since the 1-D convolutional operation on the high-level features map is definitely better for separating the temporal dynamics than applying them on the individual pixel of the original input.

The Decoder module is used to reconstruct each frame  $x_{t}$  of the by using their corresponding spatial variables  $z_{t}^{Sp}$  and temporal variables  $z_{1:t}^{Te}$  as input.

# 3.3 SPATIO-TEMPORAL VAE MODEL

**Priors:** In this work, we proposed a spatio-temporal variational autoencoder architecture to learn disentangled representations from ST raster datasets. Our assumption for variational autoencoder is that each input  $x_{t}$  is generated from a corresponding latent representation  $z_{t}$  which can be separated into two disentangled subgroup: variables  $z_{t}^{Sp}$  which contains the spatial (time-irrelevant) features and the temporal (time-varying) features  $z_{1:t}^{Te}$ .

On the one hand, since the spatial variables  $z_{t}^{Sp}$  is considered time-irrelevant, its prior is defined as a standard Gaussian distribution  $z_{t}^{Sp} \sim \mathcal{N}(0,1)$ . On the other hand, the prior of the temporal dynamic variables follow a sequential prior  $z_{1:t}^{Te} = z_{t}^{Te}|z_{<t}^{Te}$  and the latent prior  $z_{t}$  is formed as:

$$
p \left(z _ {t}\right) = p \left(z _ {t} ^ {S p}\right) p \left(z _ {1: t} ^ {T e}\right) = p \left(z _ {t} ^ {S p}\right) \prod_ {t = 1} ^ {T} p \left(z _ {t} ^ {T e} \mid z _ {<   t} ^ {T e}\right) \tag {1}
$$

Generative model: For the generative model, we assume that the generation of each frame  $x_{t}$  at a given time  $t$  depends on the combination of its corresponding spatial variables  $z_{t}^{Sp}$  and temporal variables  $z_{t}^{Te}$ . Therefore, the generation process for the whole sequence  $x_{1:T}$  can be formed as:

$$
p _ {\theta} \left(x _ {1: T}, z _ {T} ^ {S p}, z _ {1: T} ^ {T e}\right) = \prod_ {t = 1} ^ {T} p _ {\theta} \left(x _ {t} \mid z _ {t} ^ {S p}, z _ {t} ^ {T e}\right) p _ {\theta} \left(z _ {t} ^ {S p}\right) p _ {\theta} \left(z _ {t} ^ {T e} \mid z _ {<   t} ^ {T e}\right) \tag {2}
$$

where  $\theta$  are the parameters for the decoder.

Inference models: We use a deep structured model as an encoder to approximate the posterior distribution, which can factorise the latent  $z$  into disentangled spatial and temporal components. The amortised variational distribution is formed as:

$$
q _ {\phi} \left(z _ {1: T} ^ {T e}, z _ {T} ^ {S p} \mid x _ {1: T}\right) = \prod_ {t = 1} ^ {T} q _ {\phi} \left(z _ {t} ^ {S p} \mid x _ {t}\right) \prod_ {t = 1} ^ {T} q _ {\phi} \left(z _ {t} ^ {T e} \mid x _ {<   t}\right) \tag {3}
$$

# 3.4 Loss

In this work, the objective of our proposed method is defined as the combination of the VAE loss and total correlation regularisation. we formulate the :

$$
\mathcal {L} = \mathcal {L} _ {V A E} + \beta \mathcal {L} _ {T C} \tag {4}
$$

where  $\beta$  is the hyperparameter for the TC regularisation. Theoretically, applying higher value on  $\beta$  will emphasise the disentanglement of the learnt representation and lead to better separation of the spatial and temporal variables. We estimate the objective based on FactorVAE's approach (Kim & Mnih, 2018).

VAE Objective Function: The objective function of the our method is derived from the variational lower bound (Evidence Lower Bound, ELBO) of the vanilla VAE (Kingma & Welling, 2013) and is formed as follow:

$$
\begin{array}{l} \mathcal {L} _ {V A E} (\theta , \phi ; x _ {1: T}) = E _ {q _ {\phi} (z _ {1: T} ^ {S _ {p}}, z _ {1: T} ^ {T e} | x _ {1: T})} [ \sum_ {t = 1} ^ {T} \log p _ {\theta} (x _ {t} | z _ {t} ^ {S _ {p}}, z _ {t} ^ {T e}) ] - \sum_ {t = 1} ^ {T} D _ {K L} (q _ {\phi} (z _ {t} ^ {S _ {p}} | x _ {t}) | | p _ {\theta} (z _ {t} ^ {S _ {p}})) \\ - \sum_ {t = 1} ^ {T} D _ {K L} \left(q _ {\phi} \left(z _ {t} ^ {T e} \mid x _ {\leq t}\right) \mid \mid p _ {\theta} \left(z _ {t} ^ {T e} \mid z _ {<   t} ^ {T e}\right)\right) \tag {5} \\ \end{array}
$$

Note that S3VAE (Zhu et al., 2020) already propose a sequential VAE that considers the continuity of dynamic variables, but it only assumes that there are a whole set of static features share by the whole sequence. In contrast, we model the temporal and spatial features for each time step independently, resulting in detailed information for emergencies in mobility forecasting data.

Total Correlation Regularization: To encourage the overall disentanglement of the learnt representation, we introduced the Total Correlation (TC) among variables as a regularization term. It quantifies the dependency among a set variables (Alfonso et al., 2010). Experimental results from  $\beta$ -TCVAE (Chen et al., 2018) and FactorVAE (Kim & Mnih, 2018) show that, by amplifying the penalty on this term, the dependence between the variables is reduced hence emphasising the disentanglement. In this work, we estimate the total correlation using the same approach like FactorVAE (Kim & Mnih, 2018), i.e. by introducing a discriminator and using the independence testing trick and the density-ratio trick to approximate the KL term in the above equation.

# 4 EXPERIMENTS

# 4.1 DATASET AND METRICS

In this work, we focus on learning disentangled representation of spatio-temporal raster data. Therefore, we choose to conduct the experiments on the following three real-world urban flow datasets:

BikeNYC (Lin et al., 2019) is a bike usage data collected from New York City's Citi Bike bicycle sharing service, which records the trajectory of all shared bikes in the system. This work covers the time period from 2014-04-01 to 2014-09-30.

TaxiNYC (Yao et al., 2019a) is a dataset that contains taxi in-out flow data taxi New York City, created from the NYC-Taxi GPS data, which covers the period from 2015-01-01 to 2015-03-01.

TaxiBJ (Zhang et al., 2016) comprises the taxi in-out flow data that aggregate the taxi GPS position in Beijing from 2013 to the year 2016. Although it spans across 4 consecutive years, the data is not continuous (covers the period: 2013-07-01 to 2013-10-30, 2014-05-01 to 2014-06-30, 2015-03-01 to 2015-06-30, and 2015-11-01 to 2016-04-10).

For disentangled representation learning, we use all of the data to train the feature separation module, and for the mobility forecasting tasks, we use the same setup described in (Xue & Salim, 2021), i.e., the first  $80\%$  data is used for training the prediction model, and the rest  $20\%$  data is used for testing. Besides, Min-Max normalization is adopted to transform the urban flow values into the range [0, 1] for better training purpose. We evaluate the effectiveness of our models with two commonly used metrics: the Mean Absolute Error (MAE) and the Root Mean Square Error (RMSE).

# 4.2 IMPLEMENTATION DETAILS

To compare the performance of our model in mobility forecasting with the current state-of-the-art methods, we use the "Closeness, Period, Trend" scheme to form the input of our prediction model. It is widely used in the urban flow prediction area, where all three of them comprise a sequence of raster data. It designs a set of unique input sequences, namely Closeness, Period, and Trend, which correspond to the recent time intervals, daily periodicity, and weekly trend, respectively (Zhang et al., 2016). Those three sequences are then fed as the input of their models.

As shown in Figure 2, in this work, we use the same setup used by VLUC-Net (Jiang et al., 2019), where the Closeness sequence contains the previous six steps before the prediction target; the Period is the sequence of the previous day, and the Trend comprises data from the

previous week. We first train our model to learn disentangled representation on the closeness, period and trend sequences separately. Then the learnt representations are fused to train a Multi-Layer perceptron (MLP) regressor to predict the next frame of the sequence. It should be noted that the input of the MLP is not the actual features  $z$ , but the learnt  $\mu$  and  $\sigma$  of the distribution. The logic of doing this is because the sampling part will introduce uncertainty which has a huge impact on the training.

![](images/1393f7274098454c9adf66708f0b7dd5ee9855118d89223dc44cbfa915a0d608.jpg)  
Figure 2: Illustration of the "Closeness, Period, Trend" components and the actual architecture in the mobility forecasting experiment.

# 4.3 MOBILITY FORECASTING: COMPARISON AGAINST OTHER METHODS

In our experiments, we compare our method against the following mobility flow prediction methods: Historical Average (HA); Convolutional Neural Network(CNN); Convolutional LSTM (ConvLSTM); ST-ResNet(Zhang et al., 2016); DMVST(Yao et al., 2018); DeepSTN+(Lin et al., 2019); STDN(Yao et al., 2019a); VLUC-Net(Jiang et al., 2019).

Table 1: Effectiveness Evaluation of Traffic In-Out Flow Prediction. Lower is better.  

<table><tr><td rowspan="2"></td><td colspan="2">BikeNYC</td><td colspan="2">TaxiNYC</td><td colspan="2">TaxiBJ</td></tr><tr><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td></tr><tr><td>HA</td><td>4.874</td><td>1.500</td><td>21.535</td><td>7.121</td><td>45.004</td><td>24.475</td></tr><tr><td>CNN</td><td>4.511</td><td>1.574</td><td>16.741</td><td>6.884</td><td>23.550</td><td>13.797</td></tr><tr><td>ConvLSTM</td><td>3.174</td><td>1.133</td><td>12.143</td><td>4.811</td><td>19.247</td><td>10.816</td></tr><tr><td>ST-ResNet(Zhang et al., 2016)</td><td>3.191</td><td>1.169</td><td>11.553</td><td>4.535</td><td>18.702</td><td>10.493</td></tr><tr><td>DMVST-Net(Yao et al., 2018)</td><td>3.521</td><td>1.287</td><td>13.605</td><td>4.928</td><td>20.389</td><td>11.832</td></tr><tr><td>DeepSTM+(Lin et al., 2019)</td><td>3.205</td><td>1.245</td><td>11.420</td><td>4.441</td><td>18.141</td><td>10.126</td></tr><tr><td>STDN(Yao et al., 2019a)</td><td>3.004</td><td>1.167</td><td>11.252</td><td>4.474</td><td>17.826</td><td>9.901</td></tr><tr><td>VLUC-Net(Jiang et al., 2019)</td><td>3.119</td><td>1.124</td><td>10.654</td><td>4.157</td><td>18.378</td><td>10.325</td></tr><tr><td>Spatial Only</td><td>3.584</td><td>1.285</td><td>14.640</td><td>4.812</td><td>21.111</td><td>12.435</td></tr><tr><td>Temporal Only</td><td>3.107</td><td>1.167</td><td>14.175</td><td>4.669</td><td>19.825</td><td>11.638</td></tr><tr><td>Ours</td><td>2.903</td><td>1.119</td><td>12.022</td><td>4.055</td><td>19.185</td><td>10.741</td></tr></table>

The overall evaluation results on effectiveness are summarised in Table 1 for TaxiBJ, TaxiNYC and BikeNYC. The upper half of Table 1 shows the results of baseline methods on those datasets, and the lower part shows the results of our proposed approach. The best result for each column is given in bold, and methods except HA, CNN, and ConvLSTM are all the current state-of-the-art methods in crowd flow prediction. In general, we can find that our method shows the best results on the BikeNYC and TaxiNYC datasets and compatible performance on the TaxiBJ dataset.

Besides the crowd flow prediction with a full feature set, we also trained models with only spatial/temporal features to see which set contributes more to the mobility forecasting. We find that the prediction performance of the models trained on the temporal feature set leads those trained on the spatial features by a relatively large margin. This shows that for the crowd flow prediction task, the extraction of the temporal dependencies is important than the spatial dependencies, which also coincides with the research direction in this area. It should also be noted that the models trained on the temporal feature already show compatible results with the current state-of-the-art methods. The is still room for improvement since the input of our method is the raw data alone with no context information.

# 4.4 ABLATION STUDY

To explore the effectiveness of each module in the "Closeness, Period, Trend" scheme, we perform an ablation study considering different configurations of the input sequences. The detailed configuration and their corresponding results are summarised in Table 2. The left half of the table shows the sequences used for each setup. By discarding extracted features from some sequences in the scheme, we are able to evaluate their effectiveness in regrading the mobility forecasting task. Since the quality of the learnt disentangled representation is sensitive to the choice of hyperparameters (Duan et al., 2019), we trained multiple feature separation modules with different hyperparameters. The mean and standard deviation of each configuration on all three datasets are presented in the right half of Table 2.

We can first find that the performance of models using period or trend sequence alone is poor, and sometimes adding those extra information does not help the mobility forecasting task. On average, the configuration that achieves the best results for all three datasets is not configuration C6. Although the features from the trend sequence improve the results for the BikeNYC dataset by a tiny margin, introducing the middle/long-term data worsens the performance for both the TaxiNYC and TaxiBJ datasets. One of the reasons might relate to the size of the representation. Although there might be some information that can contribute to the prediction, using all three sequences tripling the dimension of the input and let the noise offset the potential benefit. Besides, we can find out that the variance results are high for all three datasets, which agree with the assumption that the quality of the learnt representation regarding the mobility forecasting task is sensitive to the choice of hyperparameters.

Table 2: Seven different configurations of the input sequences and their corresponding RMSE and MAE results. It should be noted that C6 is the widely-used setup of the current state-of-the-art mobility forecasting models  

<table><tr><td rowspan="2"></td><td colspan="3">Configuration</td><td colspan="2">BikeNYC</td><td colspan="2">TaxiNYC</td><td colspan="2">TaxiBJ</td></tr><tr><td>Closeness</td><td>Period</td><td>Trend</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td><td>RMSE</td><td>MAE</td></tr><tr><td>C0</td><td>✓</td><td>✘</td><td>✘</td><td>3.660 ±1.88</td><td>1.209 ±0.45</td><td>14.323 ±1.31</td><td>4.670 ±0.35</td><td>22.035 ±6.87</td><td>14.078 ±3.57</td></tr><tr><td>C1</td><td>✘</td><td>✓</td><td>✘</td><td>5.924 ±2.12</td><td>1.658 ±0.61</td><td>25.364 ±2.06</td><td>7.268 ±0.45</td><td>35.222 ±3.72</td><td>21.785 ±2.24</td></tr><tr><td>C2</td><td>✘</td><td>✘</td><td>✓</td><td>4.217 ±2.17</td><td>1.361 ±0.53</td><td>22.387 ±2.58</td><td>6.841 ±0.70</td><td>42.361 ±2.93</td><td>25.967 ±1.85</td></tr><tr><td>C3</td><td>✓</td><td>✓</td><td>✘</td><td>3.707 ±2.43</td><td>1.222 ±0.54</td><td>14.431 ±1.60</td><td>4.704 ±0.44</td><td>22.140 ±6.76</td><td>14.182 ±3.52</td></tr><tr><td>C4</td><td>✓</td><td>✘</td><td>✓</td><td>3.558 ±1.77</td><td>1.198 ±0.44</td><td>17.020 ±2.38</td><td>5.281 ±0.56</td><td>22.585 ±7.21</td><td>14.434 ±3.66</td></tr><tr><td>C5</td><td>✘</td><td>✓</td><td>✓</td><td>4.406 ±1.92</td><td>1.388 ±0.49</td><td>22.634 ±2.34</td><td>6.692 ±0.54</td><td>36.454 ±3.24</td><td>22.492 ±1.92</td></tr><tr><td>C6</td><td>✓</td><td>✓</td><td>✓</td><td>3.601 ±2.10</td><td>1.210 ±0.50</td><td>16.677 ±2.29</td><td>5.173 ±0.52</td><td>22.504 ±6.97</td><td>14.388 ±3.52</td></tr></table>

# 4.5 INFORMATIVE FEATURES

Given the thought that the dimension of MLP's input might be too large and the learnt representations from the previous section, we want to investigate whether we can find the "informative" features from the learnt representations. And for those "normal" representations (contains valuable information but does not achieve the best results), will the models benefit when we exclude those "non-informative" features from their input. In our work, we define the "informative" features using the definition from Unsupervised Disentanglement Ranking (UDR)(Duan et al., 2019): A latent dimension is treated as an "informative" feature if it learns a latent posterior that diverges from the prior.

$$
I _ {K L} (a) = \left\{ \begin{array}{l l} 1 & K L \left(q _ {\phi} \left(z _ {a} \mid x\right) \mid p \left(z _ {a}\right)\right) > 0. 0 0 1 \\ 0 & o t h e r w i s e \end{array} \right. \tag {6}
$$

where  $a$  is the index of the variable. We use a smaller threshold here since the number of features is larger than the UDR (Duan et al., 2019). To avoid the influence from random noise, we test the performance of mobility forecasting on all representations that we get from the previous section, and the results are summarised in Figure 3a.

First, it should be noted that there exist representations that do not contain any informative features. As shown in the figure, those "poorly-learnt" representations perform a lot worse compared to the ones with informative features. That proves that a "well-disentangled, well-learnt" representation can contribute to the downstream task, at least for the mobility forecasting task. To prove the necessity of the filtering operation (remove the "non-informative" features), we first decrease the representation size and train the feature separation module to extract representations. The results show that when the size is below a certain threshold, all representations with smaller sizes will not contain any informative features. And the ones containing informative features will always have some space. That indicates the necessity of the filtering operation since we still haven't found a solution to learn disentangled representation that only contains informative features.

Another thing that we want to investigate is whether the models can benefit when removing those non-informative features from their input. Therefore, we train those MLPs with only the informative features and calculate the difference in performance and summarised the results in Figure 3b. It is worth noted that the MLPs have the same structure except the first layer due to the reducing amount of input features.

In summary, for the BikeNYC and TaxiNYC datasets, we can find out that, although the MLP that trained with the best representation does not get performance improvement, the majority of the MLPs that trained with "normal" representations show better results after removing the "non-informative"

![](images/4929e69e9ebc2a36861158b4a913fd8e9723e25ddf9896b791c7fe36957a9887.jpg)  
(a) Results of different configurations for representation contain/not contain informative features.

![](images/88a79b6e6cf2c52db52057546f41f3efa63ef810d92d5d16b7cd702761b3e80a.jpg)

![](images/3d3e443a3776a4ebf9ab8b1ec1c1cb45e0ca638d9a6b927fcde63afe503ded5c.jpg)

![](images/1d4a507903d67f99a7f966f8f2002964fbc091aaa93627630630b8b3dc3e58f2.jpg)  
(b) Improvements of different configurations after only using informative features as input.

![](images/08625535b124116e108caeb41fea5caff77116cb0705c81dffd2b7404311c1ee.jpg)  
Figure 3: The boxplot of RMSE and MAE results for different representations (learnt with different hyperparameters).

![](images/2b117543f0a79aa4f833bf93e8091207d88c9e8b503a2964e4b0906524d5ff57.jpg)

features. As for the TaxiBJ dataset, there is a slight drop in performance but only with a tiny margin. The reason for that might be due to the different sizes and data distribution of TaxiBJ. The overall grid size for TaxiBJ is larger than the other two datasets, and the Cumulative Distribution Function (CDF) shows that it contains more large values in the dataset, which might lead to different behaviour when we are removing the "non-informative" features. More details can be found in the Appendix.

# 5 CONCLUSION

Spatio-temporal (ST) prediction tasks like mobility forecasting have attracted significant attention since they greatly influence traffic management and public safety. We introduce the disentangled representation learning method and modify it to fit spatio-temporal data. The experimental evaluation results show that our method can achieve state-of-the-art performance and is able to extract desirable spatial/temporal features. Moreover, we investigate the effectiveness of recent/middle/long-term temporal features and find that sometimes our method can achieve state-of-the-art results without long-term temporal features. Finally, we also demonstrated that a well-learnt representation shows better results in mobility forecasting tasks and removing the non-informative features from the input of downstream models sometimes can also boost performance. Hence, we hope that our method can contribute to the mobility forecasting task by introducing the disentangled representation learning mechanism. One future direction of this work is forcing the model to learn a more compact representation that only contains "informative" features. A better mechanism to separate the spatial/temporal features and link them to real semantics is also needed to be investigated.

# REFERENCES

Leonardo Alfonso, Arnold Lobbrecht, and Roland Price. Optimization of water level monitoring network in polder systems using information theory. Water Resources Research, 46(12), 2010.  
Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H Campbell, and Sergey Levine. Stochastic variational video prediction. arXiv preprint arXiv:1710.11252, 2017.  
Justin Bayer and Christian Osendorfer. Learning stochastic recurrent networks. arXiv preprint arXiv:1411.7610, 2014.  
Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798-1828, 2013.  
Ricky TQ Chen, Xuechen Li, Roger B Grosse, and David K Duvenaud. Isolating sources of disentanglement in variational autoencoders. In Advances in Neural Information Processing Systems, pp. 2610-2620, 2018.  
Junyoung Chung, Kyle Kastner, Laurent Dinh, Kratarth Goel, Aaron Courville, and Yoshua Bengio. A recurrent latent variable model for sequential data. arXiv preprint arXiv:1506.02216, 2015.  
Emily Denton and Vighnesh Birodkar. Unsupervised learning of disentangled representations from video. arXiv preprint arXiv:1705.10915, 2017.  
Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In International Conference on Machine Learning, pp. 1174-1183. PMLR, 2018.  
Sunny Duan, Loic Matthew, Andre Saraiva, Nicholas Watters, Christopher P Burgess, Alexander Lerchner, and Irina Higgins. Unsupervised model selection for variational disentangled representation learning. arXiv preprint arXiv:1905.12614, 2019.  
Otto Fabius and Joost R Van Amersfoort. Variational recurrent auto-encoders. arXiv preprint arXiv:1412.6581, 2014.  
Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. Convolutional sequence to sequence learning. In International Conference on Machine Learning, pp. 1243-1252. PMLR, 2017.  
Irina Higgins, Loic Matthew, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. 2016.  
Jun-Ting Hsieh, Bingbin Liu, De-An Huang, Li Fei-Fei, and Juan Carlos Niebles. Learning to decompose and disentangle representations for video prediction. arXiv preprint arXiv:1806.04166, 2018.  
Wei-Ning Hsu, Yu Zhang, and James Glass. Unsupervised learning of disentangled and interpretable representations from sequential data. arXiv preprint arXiv:1709.07902, 2017.  
Renhe Jiang, Zekun Cai, Zhaonan Wang, Chuang Yang, Zipei Fan, Xuan Song, Kota Tsubouchi, and Ryosuke Shibasaki. Vluc: An empirical benchmark for video-like urban computing on citywide crowd and traffic prediction. arXiv preprint arXiv:1911.06982, 2019.  
Wenwei Jin, Youfang Lin, Zhihao Wu, and Huaiyu Wan. Spatio-temporal recurrent convolutional networks for citywide short-term crowd flows prediction. In Proceedings of the 2nd International Conference on Compute and Data Analysis, pp. 28-35, 2018.  
Hyunjik Kim and Andriy Mnih. Disentangling by factorising. arXiv preprint arXiv:1802.05983, 2018.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Wei Li, Wei Tao, Junyang Qiu, Xin Liu, Xingyu Zhou, and Zhisong Pan. Densely connected convolutional networks with attention LSTM for crowd flows prediction. IEEE Access, 7:140488-140498, 2019.  
Yingzhen Li and Stephan Mandt. Disentangled sequential autoencoder. arXiv preprint arXiv:1803.02991, 2018.  
Ziqian Lin, Jie Feng, Ziyang Lu, Yong Li, and Depeng Jin. Deepstn+: Context-aware spatial-temporal neural network for crowd flow prediction in metropolis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1020-1027, 2019.  
Luis Moreira-Matias, Joao Gama, Michel Ferreira, Joao Mendes-Moreira, and Luis Damas. Predicting taxi-passenger demand using streaming data. IEEE Transactions on Intelligent Transportation Systems, 14(3):1393-1402, 2013.  
Chujie Tian, Xinning Zhu, Zheng Hu, and Jian Ma. Deep spatial-temporal networks for crowd flows prediction by dilated convolutions and region-shifting attention mechanism. Applied Intelligence, 50(10):3057-3070, 2020.  
Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022, 2016.  
Leye Wang, Xu Geng, Xiaojuan Ma, Feng Liu, and Qiang Yang. Cross-city transfer learning for deep spatio-temporal prediction. arXiv preprint arXiv:1802.00386, 2018.  
Hao Xue and Flora D Salim. Termcast: Temporal relation modeling for effective urban flow forecasting. In PAKDD (1), pp. 741-753, 2021.  
Huaxiu Yao, Fei Wu, Jintao Ke, Xianfeng Tang, Yitian Jia, Siyu Lu, Pinghua Gong, Jieping Ye, and Zhenhui Li. Deep multi-view spatial-temporal network for taxi demand prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.  
Huaxiu Yao, Xianfeng Tang, Hua Wei, Guanjie Zheng, and Zhenhui Li. Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 5668-5675, 2019a.  
Huaxiu Yao, Xianfeng Tang, Hua Wei, Guanjie Zheng, and Zhenhui Li. Revisiting spatial-temporal similarity: A deep learning framework for traffic prediction. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 5668-5675, 2019b.  
Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. arXiv preprint arXiv:1709.04875, 2017.  
Junbo Zhang, Yu Zheng, and Dekang Qi. Deep spatio-temporal residual networks for citywide crowd flows prediction. arXiv preprint arXiv:1610.00081, 2016.  
Yu Zheng, Licia Capra, Ouri Wolfson, and Hai Yang. Urban computing: concepts, methodologies, and applications. ACM Transactions on Intelligent Systems and Technology (TIST), 5(3):1-55, 2014.  
Yizhe Zhu, Martin Renqiang Min, Asim Kadav, and Hans Peter Graf. S3vae: Self-supervised sequential vae for representation disentanglement and data generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6538-6547, 2020.  
Ali Zonoozi, Jung-jae Kim, Xiao-Li Li, and Gao Cong. Periodic-crn: A convolutional recurrent model for crowd density prediction with recurring periodic patterns. In *IJCAI*, pp. 3732–3738, 2018.
