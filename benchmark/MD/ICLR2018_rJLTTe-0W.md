# BAYESIAN TIME SERIES FORECASTING WITH CHANGE POINT AND ANOMALY DETECTION

Anonymous authors

Paper under double-blind review

# ABSTRACT

Time series forecasting plays a crucial role in marketing, finance and many other quantitative fields. A large amount of methodologies has been developed on this topic, including ARIMA, Holt-Winters, etc. However, their performance is easily undermined by the existence of change points and anomaly points, two structures commonly observed in real data, but rarely considered in the aforementioned methods. In this paper, we propose a novel state space time series model, with the capability to capture the structure of change points and anomaly points, as well as trend and seasonality. To infer all the hidden variables, we develop a Bayesian framework, which is able to obtain distributions and forecasting intervals for time series forecasting, with provable theoretical properties. For implementation, an iterative algorithm with Markov chain Monte Carlo (MCMC), Kalman filter and Kalman smoothing is proposed. In both synthetic data and real data applications, our methodology yields a better performance in time series forecasting compared with existing methods, along with more accurate change point detection and anomaly detection.

# 1 INTRODUCTION

Time series forecasting has a rich and luminous history, and is essentially important in most of business operations nowadays. The main aim of time series forecasting is to carefully collect and rigorously study the past observations of a time series to develop an appropriate model which could describe the inherent structure of the series, in order to generate future values. For instance, some internet companies may be interested in the number of daily active users (DAU), say, what is DAU after certain period of time, or when will reach their target DAU goal.

Time series forecasting is a fruitful research area with many existing methodologies. The most popular and frequently used time series model might be the Autoregressive Integrated Moving Average (ARIMA) (Box et al., 2015; Zhang, 2003; Cochrane, 2005; Hipel & McLeod, 1994). Taking seasonality into consideration, Box et al. (2015) proposed the Seasonal ARIMA. The Holt-Winters method (Winters, 1960) is also very popular by using exponential smoothing. State space model (Durbin & Koopman, 2012; Scott & Varian, 2014; Brodersen et al., 2015) also attracts much attention, which is a linear function of an underlying Markov process plus additive noise. Exponential Smoothing State Space Model (ETS) (Hyndman et al., 2008) decomposes times series into error, trend, seasonal that change over time. In Internet industry, Google develops the Bayesian structure time series (BSTS) model (Brodersen et al., 2015; Scott & Varian, 2014) to capture the trend, seasonality, and similar components of the target series. Recently, Facebook proposes the Prophet approach (Taylor & Letham, 2017) based on a decomposable model with interpretable parameters that can be intuitively adjusted by analyst.

However, as in the DAU example, some special events like Christmas Holiday or President Election, newly launched apps or features, may cause short period or long-term change of DAU, leading to weird forecasting of those traditional models. The aforementioned special cases are well known as

- Anomaly points. The items, events or observations that don't conform to an expected pattern or other items in the dataset, leading to a sudden spike or decrease in the series.  
- Change points. A market intervention, such as a new product launch or the onset of an ads campaign, may lead to the level change of the original series.

Time series forecasting without change/anomaly point detection and adjustment may also lead to bizarre forecasting since these models might learn the abrupt changes in the past. There are literatures on detecting anomaly or change points individually, examples can be found in Twitter (2017); Netflix (2017); Barry & Hartigan (1993); Killick & Eckley (2014); twitter (2017). However, the aforementioned change point detection models could not support detection in the presence of seasonality, while the presence of trend/change point is not handled by the anomaly detection models. Most importantly, there is a discrepancy between anomaly/change points detection and adjustment, and commonly used manually adjustment might be a bit arbitrary.

In this paper, we develop a state space time series forecasting model in the Bayesian framework, jointly detect anomaly and change points. For implementation, an iterative algorithm with Markov chain Monte Carlo (MCMC), Kalman filter and Kalman smoothing is proposed. The novel model could capture the structure of change points, anomaly points, trend and seasonality, as also provide the distributions and forecasting intervals for time series forecasting. Both synthetic and real data sets show the better performance of proposed model, in comparison with existing baseline. Moreover, our proposed model outperforms state-of-the-art models in identifying anomaly and change points. To summarize, our work has the following contributions.

- Our proposed method outperforms the state-of-the-art methods in time series forecasting, especially when there exist change points and anomalies. By our method, we are able to obtain distributions and intervals for forecasting.  
- Along with time series forecasting, our proposed method automatically detects change points and anomalies, and it achieves high accuracy and low false discovery rate on both tasks, outperforming some popular change point and anomaly detection methods.  
- Our method is flexible to capture the structure of time series under various scenarios. By default, it takes the trend, seasonality, change points and anomalies into consideration, but it can be easily modified to study time series without some components, for example, time series without seasonality. Thus, our method can be applied to many settings in practice.

# 2 MODEL

State space time series model has been one of the most popular models in time series analysis. It is capable of fitting complicated time series structure including linear trend and seasonality. However, times series observed in real life are almost all prevailed with outliers. Change points, less in frequency but are still widely observed in real time series analysis. Unfortunately, both structures are ignored in the classic state space time series model. In the section, we aim to address this issue by introducing a novel state space time series model.

![](images/5e9a632ccaca47183341a48d0bd5dbb3cb2f32f9b9c0231e1a0d76205310e638.jpg)  
Figure 1: Demonstration of Decompositions.

Let  $\pmb{y} = (y_{1}, y_{2}, \dots, y_{n})$  be a sequence of time series observations with length  $n$ . The ultimate goal is to forecast  $(y_{n+1}, y_{n+2}, \dots)$ . The accuracy in forecasting lies in a successful decomposition of  $\pmb{y}$  into existing components. Apart from the residuals, we assume the time series is composed by trend, seasonality, change points and anomaly points. In a nutshell, we have an additive model with

time series  $=$  trend + seasonality + change point + anomaly point + residual.

Figure 1 provides a demonstration of desired decomposition of time series. In Figure 1, the top left panel shows the observed time series. And it can be decomposed into the remaining five panels. The shift in the change point panel shows where the change point lies. And the spikes in the last panel reveals the anomaly points.

As the classical state space model, we have observation equation and transition equations to model  $\pmb{y}$  and hidden variables. We use  $\pmb{\mu} = (\mu_1,\mu_2,\dots ,\mu_n)$  to model trend, and use  $\gamma = (\gamma_{1},\gamma_{2},\ldots ,\gamma_{n})$  to model seasonality. We use a binary vector  $z^a = (z_1^a,z_2^a,\dots ,z_n^a)$  to indicate anomaly points. Then we have

$$
\text {O b s e r v a t i o n} y _ {t} = \mu_ {t} + \gamma_ {t} + \left\{ \begin{array}{l} \epsilon_ {t}, \text {i f} z _ {t} ^ {a} = 0 \\ o _ {t}, \text {i f} z _ {t} ^ {a} = 1 \end{array} \right. \tag {1}
$$

The deviation between the observation  $y_{t}$  and its "mean"  $\mu_t + \gamma_t$  is modeled by  $\epsilon_t$  and  $o_t$ , depending on the value of  $z_t^a$ . If  $z_t^a = 1$ , then  $y_{t}$  is an anomaly point; otherwise it is not. Distinguished from the residues  $\epsilon = (\epsilon_1, \epsilon_2, \ldots, \epsilon_n)$ , the anomaly is captured by  $\pmb{o} = (o_1, o_2, \ldots, o_n)$  which has relative large magnitude.

The hidden state variable  $\mu$  and  $\gamma$  have intrinsic structures. There are two transition equations, for trend and seasonality separately

$$
\text {T r e n d :} \quad \mu_ {t} = \mu_ {t - 1} + \delta_ {t - 1} + \left\{ \begin{array}{l} u _ {t}, \text {i f} z _ {t} ^ {c} = 0 \\ r _ {t}, \text {i f} z _ {t} ^ {c} = 1 \end{array} , \right. \tag {2}
$$

$$
\delta_ {t} = \delta_ {t - 1} + v _ {t},
$$

$$
\text {S e a s o n a l i t y :} \gamma_ {t} = - \sum_ {s = 1} ^ {S - 1} \gamma_ {t - s} + w _ {t}. \tag {3}
$$

In Equation (2),  $\delta = (\delta_{1},\delta_{2},\ldots ,\delta_{n})$  can be viewed as the "slope" of the trend, measuring how fast the trend changes over time. The change point component is also incorporated in Equation (2) by a binary vector  $z^{c} = (z_{1}^{c},z_{2}^{c},\dots,z_{n}^{c})$ . If  $z_{t}^{c} = 1$ , it means the  $t$ -th point is a change point, with  $\mu_t$  differs from  $\mu_{t - 1} + \delta_{t - 1}$  (which can be interpreted as the "momentum" from the previous status) by  $r_t$ ; otherwise it is not a change point and they differ by  $u_{t}$ . We model the change points in a way such that  $\boldsymbol {r} = (r_1,r_2,\dots,r_n)$  have larger magnitude compared  $\boldsymbol {u} = (u_{1},u_{2},\dots,u_{n})$ . The "slope" part  $\delta$  also has its own noise  $\pmb {v} = (v_{1},v_{2},\dots,v_{n})$ .

A first look on Equation (2) may bring up with the question that it is not presented in an exactly the same way as shown in Figure 1. In Figure 1, the change points component is a step function, and it is one of the five additive components along with trend, seasonality, anomaly points and residuals. Here we model the change point directly into the trend component. Though differing in formulation, they are equivalent to each other. We choose to model in as in Equation (2) due to simplicity, and its similarity with the definition of anomaly points in Equation (1).

The seasonality component is presented in Equation (3). Here  $S$  is the length of one season and  $\boldsymbol{w} = (w_{1}, w_{2}, \dots, w_{n})$  is the noise for seasonality. The seasonality component is assumed to have almost zero average in each season.

The observation equation and transition equations together (i.e., Equation (1,2,3)) together define how  $\pmb{y}$  is generated from all the hidden variables including change points and anomaly points. We continue to explore this new model, under a Bayesian framework.

# 3 BAYESIAN FRAMEWORK

Bayesian methods are widely used in many data analysis fields. It is easy to implement and interpret, and it also has the ability to produce posterior distribution. The Bayesian method on state space time series model has been investigated in Scott & Varian (2014); Brodersen et al. (2015). In this section, we also consider Bayesian framework for our novel state space time series model. We assume all the noises are normally distributed

$$
\{\epsilon_ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {\epsilon} ^ {2}), \quad \{o _ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {o} ^ {2}), \quad \{u _ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {u} ^ {2}),
$$

$$
\{r _ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {r} ^ {2}), \quad \{v _ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {v} ^ {2}), \quad \{w _ {t} \} _ {t = 1} ^ {n} \stackrel {i i d} {\sim} \mathcal {N} (0, \sigma_ {w} ^ {2}),
$$

where  $\sigma_{\epsilon}, \sigma_{o}, \sigma_{u}, \sigma_{r}, \sigma_{v}, \sigma_{w}$  are parameters for standard deviation. As binary vectors, a natural choice is to model anomaly point indicator  $z^{\mathbf{a}}$  and change point indicator  $z^{\mathbf{c}}$  to the model them as Bernoulli random variables

$$
\{z _ {t} ^ {a} \} _ {t = 1} ^ {n} \stackrel {{i i d}} {{\sim}} \operatorname {B e r} (p _ {a}), \quad \{z _ {t} ^ {c} \} _ {t = 1} ^ {n} \stackrel {{i i d}} {{\sim}} \operatorname {B e r} (p _ {c}),
$$

![](images/1db321b341dfbbc9c8fc2b0824fcfe078b494da825e9826ecc76c8408dfdba80.jpg)  
Figure 2: Graphical presentation of our model. Note that  $\pmb{y}$  is observed, highlighted by gray background, distinguished from all the remaining ones that are hidden. Among the hidden ones, squares indicate fixed parameters, and circles indicate random variables.

where  $p_a, p_c$  are probabilities for each point to be an anomaly or change point.

For simplicity, we denote  $\alpha_{t} = (\mu_{t},\delta_{t},\gamma_{t},\gamma_{t - 1},\dots ,\gamma_{t - (S - 2)})$  to include the main hidden variables (except  $z_{t}^{a}$  and  $z_{t}^{c}$ ) in the transition equations. All the  $\alpha_{t}$  are well defined and can be generated from the previous status, except  $\alpha_{1}$ . We denote  $\pmb{a}_{1}$  to be the parameter for  $\alpha_{1}$ , which can be interpreted as the "mean" for  $\alpha_{1}$ .

With Bayesian framework, we are able to represent our model graphically as in Figure 2. As shown in Figure 2, the only observations are  $\mathbf{y}$  and all the others are hidden. In this paper, we assume there is no additional information on all the hidden states. If we have some prior information, for example, some points are more likely to be change points, then our model can be easily modified to incorporate such information, by using proper prior.

In Figure 2, we use squares and circles to classify unknown variables. Despite all being unknown, they actually behave differently according to their own functionality. For those in squares, they behave like turning parameters. Once they are initialized or given, those in circles behaves like latent variables. We call the former "parameters" and the latter "latent variable", as listed in Table 1.

Table 1: Two Categories for Hidden Variables  

<table><tr><td>Category</td><td>Hidden Variable</td><td>Definition</td></tr><tr><td rowspan="2">Latent Variable</td><td>α = (α1, α2, ..., αn)</td><td>Trend and seasonality</td></tr><tr><td>z = (za,zc)</td><td>Anomaly and change points</td></tr><tr><td rowspan="3">Parameter</td><td>a1</td><td>The “mean” for the initial trend and seasonality</td></tr><tr><td>p = (pa, pc)</td><td>Probabilities for each point to be anomaly or change point</td></tr><tr><td>σ = (σε, σo, σu, σr, σv, σw)</td><td>Standard deviation</td></tr></table>

The discrepancy between these two categories is clearly captured by the joint likelihood function. From Figure 2, the joint distribution (i.e., the likelihood function can be written down explicitly as

$$
\begin{array}{l} L _ {a _ {1}, p, \sigma} (\boldsymbol {y}, \boldsymbol {\alpha}, z) \tag {4} \\ = \prod_ {\{t: z _ {t} ^ {a} = 0 \}} g \left(y _ {t} - \mu_ {t} - \gamma_ {t}, \sigma_ {\epsilon}\right) \times \prod_ {\{t: z _ {t} ^ {a} = 1 \}} g \left(y _ {t} - \mu_ {t} - \gamma_ {t}, \sigma_ {o}\right) \times \prod_ {\{t: z _ {t} ^ {c} = 0 \}} g \left(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}, \sigma_ {u}\right) \\ \times \prod_ {\{t: z _ {t} ^ {c} = 1 \}} g (\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}, \sigma_ {r}) \times \prod_ {t = 1} ^ {n} g (\delta_ {t} - \delta_ {t - 1}, \sigma_ {v}) \times \prod_ {t = 1} ^ {n} g (- \sum_ {s = 1} ^ {S - 1} \gamma_ {t - s}, \sigma_ {v}) \times \prod_ {i = 1} ^ {n} (p _ {a}) ^ {z _ {t} ^ {a}} (1 - p _ {a}) ^ {1 - z _ {t} ^ {a}} (p _ {c}) ^ {z _ {t} ^ {c}} (1 - p _ {c}) ^ {1 - z _ {t} ^ {c}}, \\ \end{array}
$$

where  $g(x_{1},x_{2}) = \frac{1}{\sqrt{2\pi}x_{2}}\exp \left(-x_{1}^{2} / (2x_{2}^{2})\right)$  is the density function for normal distribution with mean  $x_{1}$  and standard deviation  $x_{2}$ . Here we slightly abuse the notation by using  $\mu_0,\delta_0,\gamma_0,\gamma_{-1},\ldots ,\gamma_{2 - S}$ , which are actually the corresponding coordinates of  $a_1$ .

As long with other probabilistic graphical models, our model can also be viewed as a generative model. Given the parameters  $\pmb{a}_1, \pmb{p}, \pmb{\sigma}$ , we are able to generate time series. We present the generative procedure as follows.

# Algorithm 1: Generative Procedure

Input: Parameters  $\pmb{a}_{1}, \pmb{\sigma} = (\sigma_{\epsilon}, \sigma_{o}, \sigma_{u}, \sigma_{r}, \sigma_{v}, \sigma_{w})$  and  $p_{a}, p_{c}$ , length of time series to generate  $m$   
Output: Time series  $y = (y_{1}, y_{2}, \dots, y_{m})$

1 Generate the indexes where anomalies or change points occur

$$
\{z _ {t} ^ {a} \} _ {t = 1} ^ {n} \stackrel {{i i d}} {{\sim}} \operatorname {B e r} (p _ {a}), \quad \{z _ {t} ^ {c} \} _ {t = 1} ^ {n} \stackrel {{i i d}} {{\sim}} \operatorname {B e r} (p _ {c});
$$

2 Generate all the noises  $\pmb{\epsilon},\pmb{o},\pmb{u},\pmb{r},\pmb{v},\pmb{w}$  as independent normal random variables with mean zero and standard deviation  $\sigma_{\epsilon},\sigma_{o},\sigma_{u},\sigma_{r},\sigma_{v},\sigma_{w}$  respectively;  
3 Generate  $\{\pmb {\alpha}_t\}_{t = 1}^m$  sequentially by the transition functions in Equation (2) and (3);  
4 Generate time series  $\{y_t\}_{t=1}^m$  by the observation function in Equation (1).

# 4 INFERENCE

This section is about inferring unknown variables from  $\mathbf{y}$ , given the Bayesian setting described in the previous section. The main framework here is to sequentially update each hidden variable by fixing the remaining ones. As stated in the previous section, there are two different categories of unknown variables. Different update schemes need to be used due to the difference in their functionality. For the latent variables, we implement Markov chain Monte Carlo (MCMC) for inference. Particular, we use Gibbs sampler. We will elaborate the details of updates in the following sections.

# 4.1 UPDATES ON TREND AND SEASONALITY

In this section, we focus on updating  $\alpha$  assuming all the other hidden variables are given and fixed. The essence of Gibbs sampler is to obtain posterior distribution  $p_{a_1,p,\sigma}(\alpha |y,z)$ . This can be achieved by a combination of Kalman filter, Kalman smoothing and the so-called "fake-path" trick. We provide some intuitive explanation here and refer the readers to Durbin & Koopman (2012) for detailed implementation.

Kalman filter and Kalman smoothing are classic algorithms in signal processing and pattern recolonization for Bayesian inference. It is well related to other algorithms especially message passing algorithm. Kalman filter collects information forwards to obtain  $\mathbb{E}(\pmb{\alpha}_t | y_1, y_2, \dots, y_t)$ ; while Kalman smoothing distribute information backwards to achieve  $\mathbb{E}(\pmb{\alpha}_t | \pmb{y})$ .

However, the combination of Kalman filter and Kalman smoothing is not enough, as it only gives the expectations of marginal distributions  $\{\mathbb{E}(\pmb{\alpha}_t|\pmb{y})\}_{t=1}^n$ , instead of the joint distribution required for Gibbs sampler. To address this issue, we can use the "fake-path" trick described in Brodersen et al. (2015); Durbin & Koopman (2012). The main idea underlying this trick lies on the fact that the covariance structure of  $p(\pmb{\alpha}_t|\pmb{y})$  is not dependent on the means. If we are able to obtain the covariance by some other way, then we can add it up with  $\{\mathbb{E}(\pmb{\alpha}_t|\pmb{y})\}_{t=1}^n$  to obtain a sample from  $p(\pmb{\alpha}|\pmb{y})$ . This trick involves three steps. Note that all the other hidden variables  $\pmb{z}, \pmb{p}, \pmb{\sigma}$  are given.

1. Pick some vector  $\tilde{\pmb{a}}_1$ , and generate a sequence of time series  $\tilde{\pmb{y}}$  from it by Algorithm 1. In this way, we also observe  $\tilde{\alpha}$ .  
2. Obtain  $\{\mathbb{E}(\tilde{\alpha}_t|\tilde{y})\}_{t = 1}^n$  from  $\tilde{y}$  by Kalman filter and Kalman smoothing.  
3. We use  $\{\tilde{\alpha}_t - \mathbb{E}(\tilde{\alpha}_t|\tilde{y}) + \mathbb{E}(\alpha_t|y)\}_{t = 1}^n$  as our sampling from the conditional distribution.

# 4.2 CHANGE POINT AND ANOMALY DETECTION

In this section, we update  $z$  by Gibbs sampler, assuming  $\alpha$ ,  $a_1$ ,  $p$ ,  $\sigma$  are all given and fixed. We need to obtain the conditional distribution  $p_{a_1,p,\sigma}(z|\mathbf{y},\alpha)$ . Note that in the graphical model described in Section 2,  $\{z_t^a\}_{t=1}^n$  and  $\{z_t^c\}_{i=1}^n$  are all Bernoulli random variables and independent of each other. Then the conditional distribution  $p_{a_1,p,\sigma}(z|\mathbf{y},\alpha)$  can also be decomposed into product of Bernoulli

density functions. In other words, conditioned on  $\pmb{y},\pmb{\alpha}$ ,  $\{z_t^a\}_{t = 1}^n$  and  $\{z_t^c\}_{i = 1}^n$  are still independent Bernoulli random variables, but possibly with different success probabilities. Thus, we can take the calculation point by point. For example, for the anomaly detection for the  $t$ -th point, we have

$$
z _ {t} ^ {a} = 0: y _ {t} - \mu_ {t} - \gamma_ {t} \sim \mathcal {N} \left(0, \sigma_ {\epsilon} ^ {2}\right)
$$

$$
z _ {t} ^ {a} = 1: y _ {t} - \mu_ {t} - \gamma_ {t} \sim \mathcal {N} (0, \sigma_ {o} ^ {2}).
$$

And the prior on  $z_{t}^{a}$  is  $\mathbb{P}(z_t^a = 1) = p_a$  and  $\mathbb{P}(z_t^a = 0) = p_1$ . Let  $p_t^a = \mathbb{P}(z_t^a = 1|\pmb {y},\pmb {\alpha})$ . Directly calculation leads to

$$
p _ {t} ^ {a} = \frac {\frac {p _ {a}}{\sigma_ {o}} \exp \left[ - \frac {\left(y _ {t} - \mu_ {t} - \gamma_ {t}\right) ^ {2}}{2 \sigma_ {o} ^ {2}} \right]}{\frac {1 - p _ {a}}{\sigma_ {\epsilon}} \exp \left[ - \frac {\left(y _ {t} - \mu_ {t} - \gamma_ {t}\right) ^ {2}}{2 \sigma_ {\epsilon} ^ {2}} \right] + \frac {p _ {a}}{\sigma_ {o}} \exp \left[ - \frac {\left(y _ {t} - \mu_ {t} - \gamma_ {t}\right) ^ {2}}{2 \sigma_ {o} ^ {2}} \right]}. \tag {5}
$$

This equality holds for all  $t = 1,2,\dots ,n$ . Similarly for change point detection, let  $p_t^c = \mathbb{P}(z_t^c = 1|\pmb {y},\pmb {\alpha})$ , and we have

$$
p _ {t} ^ {c} = \frac {\frac {p _ {c}}{\sigma_ {r}} \exp \left[ - \frac {\left(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}\right) ^ {2}}{2 \sigma_ {r} ^ {2}} \right]}{\frac {1 - p _ {c}}{\sigma_ {u}} \exp \left[ - \frac {\left(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}\right) ^ {2}}{2 \sigma_ {u} ^ {2}} \right] + \frac {p _ {c}}{\sigma_ {r}} \exp \left[ - \frac {\left(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}\right) ^ {2}}{2 \sigma_ {r} ^ {2}} \right]}. \tag {6}
$$

As mentioned above, all the coordinates in  $z$  are still independent Bernoulli random variables conditioned on  $y, \alpha$ . Thus, for Gibbs sampler, we can generate  $z$  by sampling independently with

$$
\left\{z _ {t} ^ {a} \right\} _ {t = 1} ^ {n} \sim \operatorname {B e r} \left(p _ {t} ^ {a}\right), \quad \left\{z _ {t} ^ {c} \right\} _ {t = 1} ^ {n} \sim \operatorname {B e r} \left(p _ {t} ^ {c}\right).
$$

For change point detection here, we have an additional segment control step. After obtaining  $\{z_t^c\}_{t=1}^n$  as mentioned above, we need to make sure that the change points detected satisfy some additional requirement on the length of segment among two consecutive change points. This issue arises from the ambiguity between the definitions of change point and anomaly points. For example, consider a time series with value  $(0,0,0,0,1,1,1,0,0,0)$ . We can view it with two change points, one increases the trend by 1 and the other decreases it by 1. Alternatively, we can also argue the three 1s in this time series are anomalies, though next to each other. One way to address this ambiguity is by defining the minimum length of segment (denoted as  $l$ ). In this toy example, if we set the minimum length to be 4, then they are anomaly points; if we set it to be 3, then we regard them to be change points. But a more complicated criterion is needed than using minimum length as the time series usually own much more complex structure than this toy example. Consider time series  $(0,0,0,0,-1,-1,1,1,1,1)$  and the minimum time series parameter  $l = 3$ . It is reasonable to view it with one change point with increment 1, and the two -1s should be regarded as anomalies. As a combination of all these factors, we propose the following segment control method. A default value for the parameter  $l$  is the length of seasonality, i.e.,  $l = S$ .

# Algorithm 2: Segment control on change points

Input: change point binary vector  $z^c$ , trend  $\mu$ , standard deviation for outliers  $\sigma_r$ , change point minimum segment  $l$

Output: change point binary vector  $z^{c}$

Denote  $t_1 < t_2 < \ldots$  to be all the indexes such that  $z_{t_i}^c = 1$ ;

while there exists  $i$  such that  $|t_{i + 1} - t_i| < l$  do

2 Check if  $|\mu_{t_i - 1} - \mu_{t_{i + 1} + 1}|\leq \sigma_r / 2$  . If so, exclude both them from change points by setting

$z_{t_i}^c = z_{t_{i + 1}}^c = 0$ . Otherwise, randomly exclude one of them by setting the corresponding coordinate in  $z^c$  to be 0;

3 Update all the indexes of change points in  $z^c$

end

# 4.3 INITIALIZATION AND UPDATES ON PARAMETERS

The parameters  $\sigma, a_1$  and  $p$  need both initialization and update. We have different initializations and update schemes for each of them.

For all the standard deviations, once we obtain  $\alpha$  and  $z$ , we update them by taking the empirical standard deviation correspondingly. For  $\sigma_{\delta}$  and  $\sigma_{\gamma}$ , the calculation is straightforward as they only involve  $\delta$  and  $\gamma$  respectively. For  $\sigma_{\epsilon}, \sigma_{o}, \sigma_{u}$  and  $\sigma_{r}$ , it is a bit more involved due to  $z$ . Nevertheless, we can obtain the following update equations for all of them:

$$
\sigma_ {\epsilon} = \sqrt {\sum_ {\{t : z _ {t} ^ {a} = 0 \}} \frac {(y _ {t} - \mu_ {t} - \gamma_ {t}) ^ {2}}{| \{t : z _ {t} ^ {a} = 0 \} |}}, \sigma_ {o} = \sqrt {\sum_ {\{t : z _ {t} ^ {a} = 1 \}} \frac {(y _ {t} - \mu_ {t} - \gamma_ {t}) ^ {2}}{| \{t : z _ {t} ^ {a} = 1 \} |}}, \sigma_ {u} = \sqrt {\sum_ {\{t : z _ {t} ^ {c} = 0 \}} \frac {(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}) ^ {2}}{| \{t : z _ {t} ^ {c} = 0 \} |}}, \tag {7}
$$

$$
\sigma_ {r} = \sqrt {\sum_ {\{t : z _ {t} ^ {c} = 1 \}} \frac {\left(\mu_ {t} - \mu_ {t - 1} - \delta_ {t - 1}\right) ^ {2}}{\mid \{t : z _ {t} ^ {c} = 0 \} \mid}}, \sigma_ {\delta} = \sqrt {\frac {1}{n} \sum_ {t = 1} ^ {n} (\delta_ {t} - \delta_ {t - 1}) ^ {2}}, \sigma_ {\gamma} = \sqrt {\frac {1}{n} \sum_ {t = 1} ^ {n} \left(\sum_ {s = 0} ^ {S - 1} \gamma_ {t - s}\right) ^ {2}}. \tag {8}
$$

Note that in some iterations, when there is no change point or anomaly detected in  $z$ , then the updates above for  $\sigma_o, \sigma_r$  are not well-defined. In those cases, we simply let them remain the same. To initialize  $\pmb{\sigma}$ , we let them all equal to the standard deviation of  $y$ .

For  $a_1$ , we initialize it by letting its first coordinate to be equal to the average of  $y_1, y_2, \ldots, y_S$ , and all the remaining coordinates to be equal to 0. Since  $a_1$  can be interpreted as the mean vector of  $\alpha_1$ , in this way the trend is initialized to be matched up with average of the first season, and the slope and seasonality are initialized to be equal to 0. We update  $a_1$  by using information of  $\alpha$ . We let the first two coordinates (trend and slope) of  $a_1$  to be equal to those of  $\alpha_1$ , and we let the remaining coordinates (seasonality) of  $a_1$  to be equal to those of  $\alpha_{S+1}$ . The reason why we do not let  $a_1$  to be equal to  $\alpha_1$  entirely is due to the consideration on convergence and robustness. Since we initialize the seasonality part in  $a_1$  as 0, it will remain 0 if we let  $a_1$  equals  $\alpha_1$  entirely (due to the mechanism how we update  $\alpha_1$  as described in Section 4.1. We can avoid such trouble via using  $\alpha_{S+1}$ .

For  $\pmb{p}$ , we initialize them to be equal to  $1 / n$ . If we have additional information on the number of change points or anomaly points, we can initiate them with different values, for example,  $0.1 / n$ , or  $10 / n$ . We can update  $\pmb{p}$  after obtaining  $z$ , but we choose not to, also for the sake of robustness. In the early iterations when the algorithm is far from convergence, it is highly possible that  $z^a$  or  $z^c$  may turn out to be all 0. If we update  $\pmb{p}$ , say, by taking the proportion of change point or anomaly points in  $z$ . Then  $p_a$  or  $p_c$  might be 0, and it may get stuck in 0 in the remaining iterations.

# 5 FORECASTING

Once we infer all the latent variables  $\alpha, z$  and tune all the parameters  $p, a_1, \sigma$ , we are able to forecast the future time series  $\mathbf{y}_{\mathrm{future}}$ . From the graphical model described in Section 3, the future forecasting only involves  $\alpha_n$  instead of the whole  $\alpha$ . Note that we assume that there exists no change point and anomaly point in the future. This is reasonable as in most cases we have no additional information on the future time series. Given  $\alpha_n$  and  $\sigma$  we can use our predictive procedures (i.e., Algorithm 1) to generate future time series  $\mathbf{y}_{\mathrm{future}}$ . We can further integral out  $\alpha_n$  to have the posterior predictive distribution as  $p_{\sigma}(\mathbf{y}_{\mathrm{future}}|\mathbf{y})$ .

The forecasting on future time series is not deterministic. There are two sources for the randomness in  $\mathbf{y}_{\mathrm{future}}$ . One comes from the inference of  $\alpha_{n}$  (and also  $\sigma$ ) from  $\mathbf{y}$ . Under the Bayesian framework in Section 3, we have a posterior distribution over  $\alpha_{n}$  rather than a single point estimation. The second one comes from the forecasting function itself. The forecasting involves intrinsic noise like  $\epsilon_{t}, u_{t}, v_{t}$  and  $w_{t}$ . Thus, the predictive density function  $p_{\sigma}(y_{\mathrm{future}} | \mathbf{y}, \alpha_{n})$  will lead to different paths even with fixed  $\sigma$  and  $\alpha_{n}$ . In this way we are able to obtain distribution and predictive interval for forecasting. We also suggest to take the average of multiple forecasting paths, as the posterior mean for the forecasting.

The average of multiple forecasting paths (denoted as  $\bar{y}_{\mathrm{future}}$ ), if the number of paths is large enough, always takes the form as a combination of linear trend and seasonality. This can be observed in both our synthesis data (Section 7) and real data analysis (Section 8). This seems to be surprising at the first glance, but makes some sense intuitively. Under our assumption, we have no information on the future, and thus a reliable way to forecast the future is to use the information collected at the end of observed time series, i.e., trend  $\mu_{n}$ , slope  $\delta_{n}$  and seasonality structure. Theorem 1 gives mathematical explanation of the linearity of  $\bar{y}_{\mathrm{future}}$ , in both mean and standard deviation.

Theorem 1. Let  $N$  be the number of future time series paths we generate from Algorithm 1). Let  $m$  be the number of points we are going to forecast. Denote  $\{y_{n+j}^{(1)}\}_{j=1}^{m}, \{y_{n+j}^{(2)}\}_{j=1}^{m}, \ldots, \{y_{n+j}^{(N)}\}_{j=1}^{m}$

to be the future paths. Define  $\bar{y}_{\text{future}} = (\bar{y}_{n+1}, \bar{y}_{n+2}, \dots, \bar{y}_{n+m})$  to be the average such that

$$
\bar {y} _ {n + j} = \frac {1}{N} \sum_ {i = 1} ^ {N} y _ {n + j} ^ {(i)}.
$$

Then for all  $j = 1,2,\ldots ,N$ , we have  $\bar{y}_{n + j}$  as a normal distribution with mean and variance as

$$
\mathbb {E} [ \bar {y} _ {n + j} ] = \mu_ {n} + j \delta_ {n} + \gamma_ {n - S + (j \bmod S)}
$$

$$
V a r \left[ \bar {y} _ {n + j} \right] = \frac {1}{N} \left(j (j + 1) \sigma_ {v} ^ {2} / 2 + j \left(\sigma_ {u} ^ {2} + \sigma_ {w} ^ {2}\right) + \sigma_ {\epsilon} ^ {2}\right).
$$

Consequently, for all  $j = 1,2,\dots,m$ ,  $\mathbb{E}[\bar{y}_{n + j}]$  is in a linear form with respect to  $j$ , and the standard deviation of  $\bar{y}_{n + j}$  also takes a approximately linear form with respect to  $j$ .

Proof. Recall that  $\alpha_{n},\sigma$  are given and fixed, and we assume there is no change point or anomaly in the future time series. The Equation (2) leads to  $\delta_{n + j} = \delta_n + \sum_{l = 1}^j v_{n + l}$ , which implies that

$$
\mu_ {n + j} = \mu_ {n} + j \delta_ {n} + \sum_ {l = 1} ^ {j} (j + 1 - l) v _ {n + l} + \sum_ {l = 1} ^ {j} u _ {m + l}.
$$

For the seasonality part, simple linear algebra together with Equation 3 leads to  $\gamma_{n+j} = \gamma_{n-S+(j \bmod S)} + \sum_{l=1}^{j} w_{n+l}$ . Thus,

$$
\bar {y} _ {n + j} = \frac {1}{N} \sum_ {i = 1} ^ {N} \left[ \mu_ {n} + j \delta_ {n} + \gamma_ {n - S + (j \bmod S)} + \sum_ {l = 1} ^ {j} (j + 1 - l) v _ {n + l} ^ {(i)} + \sum_ {l = 1} ^ {j} u _ {m + l} ^ {(i)} + \sum_ {l = 1} ^ {j} w _ {n + l} ^ {(i)} + \epsilon_ {n + j} ^ {(i)} \right].
$$

Due to the independence and Gaussian distribution of all the noises,  $\bar{y}_{n + j}$  is also normally distributed and its means and variance can be calculated accordingly.

# 6 ALGORITHM

Our proposed method can be divided into three parts: initialization, inference, and forecasting. Section 4 and Section 5 provide detailed explanation and reasoning for each of them. We present a whole picture of our proposed methodology in Algorithm 3.

It is worth mentioning that our proposed methodology is downward compatible with many simpler state space time series models. By letting  $p_c = 0$ , we assume there is no change point in the time series. By letting  $p_a = 0$ , we assume there is no anomaly point in the time series. If both  $p_c$  and  $p_a$  are set to be 0, then our model is reduced to the classic state space time series model. Also, the seasonality and slope can be removed from our model, if we know there exists no such structure in the data.

# 7 SIMULATION

In this section, we study the synthetic data generated from our model. We let  $S = 7$  and provide values for  $\sigma$  and  $a_1$ . The change points and anomaly points are randomly generated. We use our generative procedure (Algorithm 1) to generate time series with total length 500 by fixed parameters. The first 350 points will be used as training set and the remaining 150 points will be used to evaluate the performance of forecasting.

When generating, we let the time series have weekly seasonality with  $S = 7$ . For  $\sigma$  we have  $\sigma_{\epsilon} = 0.1$ ,  $\sigma_{u} = 0.1$ ,  $\sigma_{v} = 0.0004$ ,  $\sigma_{w} = 0.01$ ,  $\sigma_{r} = 1$ ,  $\sigma_{o} = 4$ . For  $\alpha_{1}$  we have value for  $\mu$  as 20, value for  $\delta$  as 0, and value for seasonality as  $(1,2,4,-1,-3,-2)/10$ . For  $p$  we have  $p_{c} = 4/350$  and  $p_{a} = 10/350$ . Despite that, to make sure that at least one change point is in existence, we force  $z_{330}^{c} = 1$  and  $r_{330} = 2$ . That is, for each time series we generate, its 330th point is a change point with the mean shifted up by 3. Also to be consistent with our assumption, we force  $z_{i}^{c} = z_{i}^{a} = 0, \forall 351 \leq i \leq 500$  so there exists no change point or anomaly point in the testing part.

Algorithm 3: Proposed Algorithm

Input: Observed time series  $\mathbf{y} = (y_{1}, y_{2}, \dots, y_{n})$ , seasonality length  $S$ , length of time series for forecasting  $m$ , number of predictive paths  $N$ , change point minimum segment  $l$

Output: Change point detection  $z^c$ , anomaly points  $z^a$ , forecasting result  $y_{\mathrm{future}} = (y_{n + 1},y_{n + 1},\dots ,y_{n + m})$  and its distribution or predictive intervals

# Part I: Initialization;

1 Initialize  $\sigma_{\epsilon},\sigma_{o},\sigma_{u},\sigma_{r},\sigma_{v},\sigma_{w}$  all with the empirical standard deviation of  $\pmb{y}$  
2 Initialize  $\pmb{a}_{1}$  such that its first coordinate equals to the average of  $(y_{1}, y_{2}, \dots, y_{S})$  and all the remaining  $S$  coordinates with 0;  
3 Initialize  $p_a$  and  $p_c$  by  $1/n$ . Then generate  $z^a$  and  $z^c$  as independent Bernoulli random variables with success probability  $p_a$  and  $p_c$  respectively;

# Part II: Inference;

while the likelihood function  $L_{a_1,p,\sigma}(\pmb {y},\pmb {\alpha},\pmb {z})$  not converges do

4 Infer  $\alpha$  by Kalman filter, Kalman smoothing and "fake-path" trick described in Section 4.1;  
5 Update  $z^a$  and  $z^c$  by sampling from

$$
\left\{z _ {t} ^ {a} \right\} _ {t = 1} ^ {n} \sim \operatorname {B e r} \left(p _ {t} ^ {a}\right), \quad \left\{z _ {t} ^ {c} \right\} _ {t = 1} ^ {n} \sim \operatorname {B e r} \left(p _ {t} ^ {c}\right),
$$

where the success probability  $\{p_t^a\}_{t = 1}^n$  and  $\{p_t^c\}_{t = 1}^n$  are defined in Equation (5) and (6);  
6 Segment control on  $z^c$  by Algorithm 2;  
7 Update  $\sigma$  by Equation (7) to (8);  
8 Update  $\pmb{a}_{1}$  such that its first two coordinates equal to the those of  $\alpha_{1}$  and the remaining  $(S - 1)$  coordinates equals to those of  $\alpha_{S + 1}$ ;  
9 | Calculate the likelihood function  $L_{a_1,p,\sigma}(\pmb{y},\pmb{\alpha},z)$  given in Equation (4);

# end

# Part III: Forecasting;

With  $a_{n}$  and  $\sigma$ , use the generate procedure in Algorithm 1 to generate future time series  $\pmb{y}_{\mathrm{future}}$  with length  $m$ . Repeat the generative procedure to obtain multiple future paths

$$
\mathbf {y} _ {\text {f u t u r e}} ^ {(1)}, \mathbf {y} _ {\text {f u t u r e}} ^ {(2)}, \dots , \mathbf {y} _ {\text {f u t u r e}} ^ {(N)};
$$

11 Combine all the predictive paths give the distribution for the future time series forecasting. If needed, calculate the point-wise quantile to obtain predictive intervals. Use the point-wise average as our final forecasting result.

The top panel of Figure 3 shows one example of synthesis data. The blue line marks the separation between training and testing set. The blue dashed line indicates the locations for the change point, while the yellow dots indicate the positions of anomaly points. Also see Figure 3 for illustration on the results returned by implementing our proposed algorithm on the same dataset. The red line gives the fitting results in the first 350 points and forecasting results in the last 150 points. The change points detected are marked with vertical red dotted line, and the anomaly detected are flagged with purple squares. Figure 3 shows that on this dataset, our proposed algorithm yields perfect detection on both change points and anomaly points. In Figure 3, the gray part indicates the  $90\%$  predictive interval for forecasting.

![](images/214da7b3b7ec9b2486b0ca449ac1ad0f4570016b9275864988c6b4f4654a0e26.jpg)  
Figure 3: An example of synthesis data (left), and the result after applying our algorithm (right).

![](images/947c940a9a242f187448473c0bb1de060b56da801217ed1edb9d342e2528c43f.jpg)

We run our generative model 100 times to produce 100 different time series, and implement multiply methods on each of them, and aggregate the results together for comparison. We include the following methodologies. For time series forecasting, we compare our method against Bayesian Structural Time Series (BSTS) (Scott & Varian, 2014; Brodersen et al., 2015)), Seasonal Decomposition of Time Series by Loess (STL) (Cleveland et al., 1990)), Seasonal ARIMA (Box et al., 2015), Holt-Winters (Holt, 2004), Exponential Smoothing State Space Model (ETS) (Hyndman et al., 2008)), and the Prophet R package by Taylor & Letham (2017). We evaluate the performances by mean absolute percentage error (MAPE), mean square error (MSE) and mean absolute error (MAE) on forecasting set. The mathematical definition of these three criterion is given as follows. Let  $x_{1}, x_{2}, \ldots, x_{n}$  be the true value and  $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_n$  be the estimation or predictive values. Then we have

$$
\mathrm {M A P E} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\left| x _ {i} - \hat {x} _ {i} \right|}{x _ {i}}, \mathrm {M S E} = \sqrt {\frac {1}{n} \sum_ {i = 1} ^ {n} (x _ {i} - \hat {x} _ {i}) ^ {2}}, \mathrm {M A E} = \frac {1}{n} \sum_ {i = 1} ^ {n} | x _ {i} - \hat {x} _ {i} |.
$$

The comparison of our proposed algorithm and the aforementioned algorithms are included below in Table 2. As we mentioned in Section 6, our algorithm is downward compatible with the cases ignoring the existence of change point or anomaly, by setting  $p_c = 0$  or  $p_a = 0$ . We also run proposed algorithm on the synthetic data with  $p_c = 0$  (no change point), or  $p_a = 0$  (no anomaly point), or  $p_c = p_a = 0$  (no change and anomaly point), for the purpose of numeric comparison.

Table 2: Comparison of methodologies on Forecasting  

<table><tr><td>Methods</td><td>MAPE</td><td>MSE</td><td>MAE</td></tr><tr><td>Proposed</td><td>0.041 ± 0.027</td><td>1.03 ± 0.59</td><td>0.89 ± 0.53</td></tr><tr><td>Proposed (pa=0)</td><td>0.069 ± 0.068</td><td>1.71 ± 1.61</td><td>1.49 ± 1.44</td></tr><tr><td>Proposed (pc=0)</td><td>0.065 ± 0.058</td><td>1.67 ± 1.53</td><td>1.43 ± 1.35</td></tr><tr><td>Proposed (pa=0,pc=0)</td><td>0.084 ± 0.079</td><td>2.15 ± 2.00</td><td>1.87 ± 1.77</td></tr><tr><td>BSTS</td><td>0.162 ± 0.110</td><td>4.10 ± 2.81</td><td>3.59 ± 2.48</td></tr><tr><td>STL</td><td>0.047 ± 0.039</td><td>1.18 ± 1.06</td><td>1.03 ± 0.95</td></tr><tr><td>ARIMA</td><td>0.076 ± 0.050</td><td>1.88 ± 1.38</td><td>1.71 ± 1.24</td></tr><tr><td>Holt-Winters</td><td>0.093 ± 0.082</td><td>2.35 ± 2.06</td><td>2.05 ± 1.84</td></tr><tr><td>ETS</td><td>0.054 ± 0.042</td><td>1.37 ± 1.05</td><td>1.19 ± 0.94</td></tr><tr><td>Prophet</td><td>0.082 ± 0.055</td><td>2.06 ± 1.33</td><td>1.78 ± 1.16</td></tr></table>

From Table 2 it turns out that our proposed algorithm achieves the best performance compared to other existing methods. Our proposed algorithm also performs better compared with the cases ignoring change point or anomaly point. This is a convincing evidence on the importance of incorporating both change point structure and anomaly point structure when modeling, for time series forecasting.

We also compare our proposed method with other existing change point detection methods and anomaly detection algorithm with respect to the performance of detections. We evaluate the performance by two criterions: True Positive Rate (TPR) and False Positive (FP). TPR measures the percentage of change points or anomalies to be correctly detected. FP count the number of points wrongly detected as change points or anomaly points. The mathematical definitions of TPR and FP are as follows. Let  $(z_{1},z_{2},\ldots ,z_{n})$  be the true binary vector for change points or anomalies, and  $(\hat{z}_1,\hat{z}_2,\dots ,\hat{z}_n)$  are the estimated ones. Then

$$
\mathrm {T P R} = \frac {\left| \left\{i : z _ {i} = 1 , \hat {z} _ {i} = 1 \right\} \right|}{\left| \left\{i : z _ {i} = 1 \right\} \right|}, \quad \mathrm {F P} = \left| \left\{i : z _ {i} = 0, \hat {z} _ {i} = 1 \right\} \right|.
$$

From the definition, we can see high TPR and low FP means the algorithm has better performance in detection.

The comparison on change point detection is shown in Table 3. We compare our results against three popular change point detection methods: Bayesian Change Point (BCP) (Barry & Hartigan, 1993), Change-Point (CP) (Killick & Eckley, 2014) and Breakout (twitter, 2017). From Table 3 our proposed method outperforms the most of the others by both TPR and FP. We have smaller TPR compared to CP, but we are better in FP.

In Table 4, we also compare the performance of our algorithm on anomaly detection with three existing common anomaly detection methods: the AnomalyDetection package by Twitter (2017),

Table 3: Comparison of Change Point Detection  
Table 4: Comparison of Anomaly Detection  

<table><tr><td>Mehtods</td><td>TPR</td><td>FP</td><td>Mehtods</td><td>TPR</td><td>FP</td></tr><tr><td>Proposed</td><td>0.41 ± 0.26</td><td>0.34 ± 0.57</td><td>Proposed</td><td>0.88 ± 0.12</td><td>0.58 ± 0.96</td></tr><tr><td>Proposed (pa=0)</td><td>0.14 ± 0.21</td><td>0.26 ± 0.60</td><td>Proposed (pc=0)</td><td>0.87 ± 0.12</td><td>2.56 ± 1.49</td></tr><tr><td>BCP</td><td>0.58 ± 0.22</td><td>29.84 ± 8.13</td><td>AnomalyDetection</td><td>0.32 ± 0.19</td><td>1.03 ± 1.94</td></tr><tr><td>CP</td><td>0.29 ± 0.22</td><td>1.71 ± 1.15</td><td>RAD</td><td>0.88 ± 0.11</td><td>19.33 ± 3.58</td></tr><tr><td>Breakout</td><td>0.01 ± 0.04</td><td>0.53 ± 0.86</td><td>tsoutlier</td><td>0.81 ± 0.14</td><td>4.76 ± 4.29</td></tr></table>

RAD by Netflix (2017) and Tsoutlier by Chen & Liu (1993). The comparison is listed in Table 4. We can see our method also outperforms most of the others with respect to anomaly detection, by both TPR and FP. RAD has slightly better TPR but its FP is much worse compared with ours.

# 8 REAL DATA ANALYSIS

In this section, we implement our proposed method on real-world datasets. We also compare its performance against other existing time series forecasting methodologies. We consider two datasets, one is a public data called Well-log dataset, and the other is an unpublished internet traffic dataset. The bottom panels of Figure 4 and Figure 5 give the result of our proposed algorithms. The blue line separates the training set and testing set. We use red line to show our fitting and forecasting result, vertical red dashed line to indicate change points and purple dots to indicate anomaly points. The gray part shows  $90\%$  predication interval.

# 8.1 WELL-LOG DATA

This dataset (Fearnhead & Clifford, 2003; JK & WJ, 1996) was collected when drilling a well. It measures the nuclear magnetic response, which provides geophysical information to analyze the structure of rock surrounding the well. This dataset is public and available online<sup>1</sup>. It has 4050 points in total. We split it such that the first 3000 points are used as training set and last 1000 points are used to evaluate the forecasting performance.

![](images/57a3ba51d7da843ed8ad9f8135f84ea37306f7e122fac61e26b6898acf4835c2.jpg)  
Figure 4: Well-log Data (left). The result of implementing our proposed algorithm (right).

![](images/5c98821b59c6cbd00dfdf8572bbb8e725c6bd01029be291ef36656204877ae09.jpg)

From Figure 4, it is obvious that there exists no seasonality or slope structure in the dataset. This motivates us not to include these two components in our model. We implement our proposed algorithm without seasonality and slope, and compare the forecasting performance with other methods in Table 5. Our method outperforms BSTS, ARIMA, ETS and Prophet. However in Table 5 the performance can be slightly improved if we ignore the existence of anomaly points by letting  $p_a = 0$ . This may be caused by model mis-specification as the data may not generate in a way not entirely captured by our model. Nevertheless, the performances of our method considering anomaly points or not, are comparable to each other.

In this dataset there is no ground-truth of change point and anomaly point on their locations or even existence. However, from bottom panel of Figure 4, there are some obvious changes in the sequence and they all successfully captured by our algorithm.

Table 5: Comparison of Forecasting in Well-log Data  

<table><tr><td>Methods</td><td>MAPE</td><td>MSE</td><td>MAE</td></tr><tr><td>Proposed</td><td>0.031</td><td>5296</td><td>3120</td></tr><tr><td>Proposed (pa=0)</td><td>0.029</td><td>5252</td><td>2957</td></tr><tr><td>Proposed (pc=0)</td><td>0.033</td><td>5434</td><td>3409</td></tr><tr><td>Proposed (pa=0,pc=0)</td><td>0.038</td><td>5703</td><td>3908</td></tr><tr><td>BSTS</td><td>0.250</td><td>32030</td><td>27210</td></tr><tr><td>ARIMA</td><td>0.084</td><td>10480</td><td>8738</td></tr><tr><td>ETS</td><td>0.037</td><td>6071</td><td>3860</td></tr><tr><td>Prophet</td><td>0.159</td><td>19530</td><td>17480</td></tr></table>

# 8.2 INTERNET TRAFFIC DATA

Our second real data is an Internet traffic data acquired from a major Tech company (see Figure 5). It is a daily traffic data, with seasonality  $S = 7$ . We use the first 800 observations as training set and evaluate the performance of forecasting on the remaining 265 points. The bottom panel of Figure 5 show the result from implementing our algorithm.

![](images/3d1de96c3120c246fbbf1665e88ef86ae6e1c77c70eb6e180b7415ccfee226d2.jpg)  
Figure 5: Internet Traffic Data (top); The result of implementing our proposed algorithm (bottom).

![](images/cf10467287967e9b7e48406da7323406f422f011bc7350cf23aa50fdd86e3f52.jpg)

We also do the comparison of forecasting performance of our proposed algorithm together with other existing methods, shown in Table 6. We can also see that our algorithm outperforms all the other algorithms with respect to MAPE, MSE and MAE.

Table 6: Comparison of Forecasting in Internet traffic data  

<table><tr><td>Methods</td><td>MAPE</td><td>MSE</td><td>MAE</td></tr><tr><td>Proposed</td><td>0.0837</td><td>0.1216</td><td>0.08414</td></tr><tr><td>Proposed (pa=0)</td><td>0.0838</td><td>0.1215</td><td>0.08320</td></tr><tr><td>Proposed (pc=0)</td><td>0.0934</td><td>0.1332</td><td>0.09296</td></tr><tr><td>Proposed (pa=0,pc=0)</td><td>0.0934</td><td>0.1366</td><td>0.09223</td></tr><tr><td>BSTS</td><td>0.2756</td><td>0.3087</td><td>0.27960</td></tr><tr><td>STL</td><td>0.1014</td><td>0.1258</td><td>0.09910</td></tr><tr><td>ARIMA</td><td>0.1409</td><td>0.1653</td><td>0.12580</td></tr><tr><td>Holt-Winters</td><td>0.2495</td><td>0.2739</td><td>0.25270</td></tr><tr><td>ETS</td><td>0.0893</td><td>0.1199</td><td>0.09362</td></tr><tr><td>Prophet</td><td>0.1015</td><td>0.1405</td><td>0.11450</td></tr></table>

From Figure 5 our proposed algorithm identifies one change point (the 576th point, indicated by the vertical red dashed line), which can be confirmed that this is exactly the only one change point existing in this time series caused by the change of counting methods, by some external information. Thus, we give the perfect change point detection in this Internet traffic data.

For this Internet traffic dataset, since we have ground-truth for change point, we can compare the performance of change point detection of different methodologies. BCP returns posterior distribution, which peaks in the the 576th point with posterior probability value 0.5. And it also returns with many other points with posterior probability value around 0.1. CP returns 4 change points, where the 576th point (the only true one) is one of them. Breakout returns 8 change points without including the 576th point. To sum up, our proposed method achieves the best change point detection in this real dataset.

# 9 CONCLUSION

We incorporate the change point structure and anomaly point structure into the classic space state time series model. We provide a Bayesian scheme for inference and time series forecasting. We compare the performance of our methodology and state-of-the-art methods on both synthetic data and real datasets. Our method performs the best with respect to forecasting, change point detection, and anomaly detection as well.

# REFERENCES

Daniel Barry and John A Hartigan. A bayesian analysis for change point problems. Journal of the American Statistical Association, 88(421):309-319, 1993.  
George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015.  
Kay H Brodersen, Fabian Gallusser, Jim Koehler, Nicolas Remy, Steven L Scott, et al. Inferring causal impact using bayesian structural time-series models. The Annals of Applied Statistics, 9 (1):247-274, 2015.  
Chung Chen and Lon-Mu Liu. Joint estimation of model parameters and outlier effects in time series. Journal of the American Statistical Association, 88(421):284-297, 1993.  
Robert B Cleveland, William S Cleveland, and Irma Terpenning. Stl: A seasonal-trend decomposition procedure based on loess. Journal of Official Statistics, 6(1):3, 1990.  
John H Cochrane. Time series for macroeconomics and finance. 2005.  
James Durbin and Siem Jan Koopman. Time series analysis by state space methods, volume 38. OUP Oxford, 2012.  
Paul Fearnhead and Peter Clifford. On-line inference for hidden markov models via particle filters. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 65(4):887-899, 2003.  
Keith W Hipel and A Ian McLeod. Time series modelling of water resources and environmental systems, volume 45. Elsevier, 1994.  
Charles C Holt. Forecasting seasonals and trends by exponentially weighted moving averages. International journal of forecasting, 20(1):5-10, 2004.  
Rob Hyndman, Anne B Koehler, J Keith Ord, and Ralph D Snyder. Forecasting with exponential smoothing: the state space approach. Springer Science & Business Media, 2008.  
OR JK and F WJ. Numerical bayesian methods applied to signal processing, 1996.  
Rebecca Killick and Idris Eckley. changepoint: An r package for changepoint analysis. Journal of Statistical Software, 58(3):1-19, 2014.  
Netflix. Rad: Time series anomaly detection. 2017.  
Steven L Scott and Hal R Varian. Predicting the present with bayesian structural time series. International Journal of Mathematical Modelling and Numerical Optimisation, 5(1-2):4-23, 2014.  
S. J. Taylor and Letham. Prophet: forecasting at scale. 2017.  
Twitter. Anomalydetection: Anomaly detection with r. 2017.  
twitter. Breakout detection via robust e-statistics. 2017.  
Peter R Winters. Forecasting sales by exponentially weighted moving averages. Management science, 6(3):324-342, 1960.  
G Peter Zhang. Time series forecasting using a hybrid arima and neural network model. Neurocomputing, 50:159-175, 2003.