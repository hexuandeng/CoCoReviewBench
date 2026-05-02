# TOTALRECALL: A BIDIRECTIONAL CANDIDATES GENERATION FRAMEWORK FOR LARGE SCALE RECOMMender & ADVERTISING SYSTEMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recommender (RS) and Advertising/Marketing Systems (AS) play the key roles in E-commerce companies like Amazon and Alibaba. RS needs to generate thousands of item candidates for each user ( $u2i$ ), while AS needs to identify thousands or even millions of high-potential users for given items so that the merchant can advertise these items efficiently with limited budget ( $i2u$ ). This paper proposes an elegant bidirectional candidates generation framework that can serve both purposes all together. Besides, our framework is also superior in these aspects:  $i$ ). Our framework can easily incorporate many DNN-architectures of RS ( $u2i$ ), and increase the HitRate and Recall by a large margin.  $ii$ ). We archive much better results in  $i2u$  candidates generation compare to strong baselines.  $iii$ ). We empirically show that our framework can diversify the generated candidates, and ensure fast convergence to better results.

# 1 INTRODUCTION

In the Internet era, E-commerce companies usually act as a platform to connect both users and merchants. They can collect huge amount of data of users' behavior on merchants' items, and utilize the data to serve them better, for example, to build better search/recommender/advertising systems.

When users browse the E-commerce websites/apps, they usually want to buy some desired items, or find something interesting, and that's what the search engine and recommender system (RS) are built for. Besides, the merchant usually wants to mine the potential users of its items, and then starts a campaign to prompt those users to buy the items. The advertising system has the functionality to help the merchant, and we may call it the potential users mining system (PUMS).

In an industrial large-scale RS, there are usually two stages, the candidates generation (CG) stage and the ranking stage (Covington et al., 2016). CG plays a very import role as it sets the upper bound of the recommender. A single CG method usually lacks diversity, so the industrial CG system usually consists of different subsystems to generate candidate items, for example,  $i$ ). hot items,  $ii$ ). items similar to users (Covington et al., 2016; Li et al., 2019),  $iii$ ). items similar to clicked items (Linden et al., 2003),  $iv$ . items clicked by similar users (Wang et al., 2006).

The PUMS has been studied and applied in marketing community for many years (Bult & Wansbeek, 1995; Kim et al., 2005), and recently it is also introduced to social network to attract potential customers of companies (Pennacchiotti & Popescu, 2011; Pang et al., 2013). Similarly in E-commerce companies, it can be used to discover potential customers for merchants, brands and etc. Although it is widely used, people usually treat it as a simple binary classification problem, i.e., to predict whether a given user is the potential user or not. Models like LR, SVM, XGBoost and etc. (Hosmer Jr et al., 2013; Chang & Lin, 2011; Chen & Guestrin, 2016) are usually employed.

In this paper, we unify the two systems in our TotalRecall (TR) framework, and demonstrate the superiority using different metrics. Our main contributions are listed below:

- Bidirectional Candidates Generation: TR can be trained once and then used to generate candidate items  $(u2i)$  and users  $(i2u)$  all together without any additional operations. The

oretically we show that the superiority is due to the modeling of joint probability of  $u$  and  $i$ .

- Improving the HitRate and Recall of RS: We compare to two different recommender algorithms, i.e., Matrix Factorization (MF) with Bernoulli distribution and sequential modeling with Multinormal distribution, and archive comparable or better results than MF's SOTA, and improve the HitRate and Recall by a large margin in sequential modeling.  
- Mining High-Quality Potential Users in PUMS: TR mines potential users with higher accuracy compare to other methods. It can be applied in E-commerce and easily customized for various needs, for example, to mine users of a single item or a group of items, and the item could be the merchant, brand and etc.  
- Fast Convergence, and Diversified Candidates: TR converges much faster compare to MF models, up to  $16 \times$  times. TR also increases the diversity of both generated users and items.

# 2 RELATED WORK

# 2.1 RECOMMENDER SYSTEM (RS)

RS has been studied for many years, and in the early years, Collaborative Filtering (CF) and its variants (Su & Khoshgoftaar, 2009) are widely adopted. Later its descendant MF (Funk) is proposed to solve the problem more elegantly with higher accuracy, and Probabilistic Matrix Factorization (PMF) (Mnih & Salakhutdinov, 2008) builds a solid theoretic foundation for MF based on probability theory.

Although MF performs very well in many datasets (Rendle et al., 2020), it is usually not adopted in industry. The modern industrial RS can involve up to billions of users and hundreds of millions of items, and consist of multiple stages, e.g., CG and ranking stages (Covington et al., 2016).

Various deep neural networks are adopted to tackle the complicated system, for example, in the CG stage, the problem is formulated as the multi-class classification, and users' and items' representations are detached with the two-tower architecture to balance the efficiency and accuracy in online serving (Covington et al., 2016; Li et al., 2019; Cen et al., 2020).

In the ranking stage, the problem is formulated as the binary classification, and various models has been developed: wide & deep model (Cheng et al., 2016) captures explicit and implicit feature crossing, Deep Interest Network (Zhou et al., 2018) and Deep User Perception Network (Ni et al., 2018) utilize attention mechanism (Bahdanau et al. (2014)) to capture users' short-term interests.

# 2.2 POTENTIAL USERS MINING SYSTEM (PUMS)

PUMS mines the potential users for given items. The 'item' could be anything that users can interact with, for example, an insurance product (Kim & Street, 2004), a company/business (Pennacchiotti & Popescu, 2011; Pang et al., 2013; Lo et al., 2016), a specific information (e.g., tweet) in social medias (Tang et al., 2015; Gui et al., 2019) and even another user (Guy, 2018). These problems are usually treated as binary classification and solved with tools like SVM, LR and etc. for each individual item independently (Hosmer Jr et al., 2013; Chang & Lin, 2011; Chen & Guestrin, 2016).

In the E-commerce company like Amazon, the 'item' could be a product, a brand, a product category, a merchant and etc, and the number of the 'items' range from thousands to hundreds of millions. It will be impractical to model each item individually, so we usually model the items all together using binary classification similar to Rendle et al. (2020), which will be stated in detail later in Sec. 5.

# 2.3 SAMPLED- SOFTMAX AND INFONCE LOSS

In the CG stage of RS, sampled-softmax (SSM) loss (Jean et al., 2015) is widely used with different background distributions, such as log-uniform distribution (Powers, 1998), empirical marginal distribution (Yi et al., 2019) and mixed distribution (Yang et al., 2020).

In contrastive learning, InfoNCE loss is proposed in Oord et al. (2018) to learn representations through self-supervision. Later it is widely adopted in CV (Chen et al. (2020)) and NLP (Gao et al. (2021)) and performs very well.

InfoNCE and SSM losses are both trying to solve multi-class classification of very large vocabulary. They both use negative sampling and can be written in the same analytical form:

$$
l = \frac {1}{| \mathcal {D} |} \sum_ {(x, y) \in \mathcal {D}} - \log \frac {\exp \left(f _ {\theta} (x , y) - \log p _ {n} (y \mid x)\right)}{\exp \left(f _ {\theta} (x , y) - \log p _ {n} (y \mid x)\right) + \sum_ {y ^ {\prime} \in \mathbb {Y} _ {x}} \exp \left(f _ {\theta} (x , y ^ {\prime}) - \log p _ {n} \left(y ^ {\prime} \mid x\right)\right)}, \tag {1}
$$

where  $(x,y)\in \mathcal{D}$  is the training data,  $f_{\theta}(x,y)$  is the output of the neural network parametrized by  $\theta$  and  $\mathbb{Y}_x$  contains hundreds or thousands of negative samples from background distribution  $p_n(y|x)$ . The main differences are that the  $p_n(y|x)$  in SSM can be arbitrarily chosen, while in InfoNCE it is the empirical marginal distribution  $\hat{p}_{\mathrm{data}}(y)$ .

Recently, InfoNCE loss is exploited in RS to optimize  $\frac{p(y|x)}{p(y)}$  instead of  $p(y|x)$ , since it can be easily implemented with in-batch negative sampling and features of  $y$  can be fully utilized (Zhou et al., 2021).

# 3 METHOD

# 3.1 PROBLEM DEFINITION

In E-commerce companies, when a user  $u$  clicked an item  $i$  at time  $t$ , a record  $(u,i,t)$  was logged as the implicit feedback.

Given the raw logs  $\{(u,i,t)\}$ , there are usually two ways to process the data: one is to directly create a user-item interaction matrix  $S_{ui}$  with the value  $s_{u,i} = 1$  if the  $(u,i)$  pair appears in the log (Su & Khoshgoftaar (2009)); the other is to formulate the problem as next-item prediction, i.e., from the raw log we create a dataset  $\mathcal{D} = \{(x_{u,t},y_{u,t}):t\in \{1,2,\dots,T_u\} |u\in \{1,2,\dots,N\} \}$ , where  $x_{u,t} = \{y_{u,1:(t - 1)}\}$  represents  $u$ 's clicks prior to  $t$ -th click  $y_{u,t}$  and  $T_{u}$  is the number of clicks of  $u$  (Covington et al. (2016)). We define the click sequence  $x_{u,t}$  as the pseudo-user, so we can create a pseudo-user-item interaction matrix from  $\mathcal{D}$ , and all possible click sequences form the pseudo-user set.

Therefore, they both deal with the 2D interaction matrix  $S_{ui}$ , irrespective of whether the  $u$  is a real user or pseudo-user or not. In the remaining of the paper, we will not distinguish between user and pseudo-user, and only focus on solving the  $S_{ui}$ .

$S_{ui}$  consists of all users as the row and all items as the column, and its entries are either 1 or unknown. Let's assume all users form the set  $\mathbb{U} = \{u_1, u_2, \dots, u_M\}$ , and all items form the set  $\mathbb{I} = \{i_1, i_2, \dots, i_K\}$ . Given a user  $u \in \mathbb{U}$ , we will generate candidate items from the item pool  $\mathbb{I}$  (RS,  $u2i$ ), while given an item  $i \in \mathbb{I}$ , we shall find out the potential users out of all users  $\mathbb{U}$  (PUMS,  $i2u$ ).

The candidates generation of RS and PUMS are of the opposite direction, and traditionally they are solved separately. But essentially they are both trying to estimate some kind of probabilities of the unknown entries in  $S_{ui}$ , so we propose to model them jointly.

# 3.2 RECOMMENDER SYSTEM

In RS, there are usually two types of modeling objectives when dealing with  $S_{ui}$ , and we briefly introduce them in the following sections (See App. A.1 for more details).

# 3.2.1 GAUSSIAN AND BERNOULLI DISTRIBUTION

The  $u$ -i interaction score  $s_{u,i}$  of  $S_{ui}$  is treated as the random variable, and assumed to be i.i.d. following a certain distribution, for example, Gaussian (Mnih & Salakhutdinov (2008)) or Bernoulli (Johnson (2014); He et al. (2017); Rendle et al. (2020)) distributions.

# 3.2.2 MULTINORMIAL DISTRIBUTION

Here the modeling objective is the random variable  $s_u$ , defined as the interacting item of  $u$ . For example, given the total number of interactions  $N_u = \sum_{i \in \mathbb{I}} s_{u,i}$  of  $u$ ,  $s_u$  is assumed to be sampled from a multinomial distribution  $\mathrm{Mult}(N_u, p_u)$ , where  $p_u$  is a  $K$ -dimensional vector whose elements are the probability and sum to 1 (Liang et al., 2018). Although not explicitly stated in many papers (Covington et al., 2016; Yi et al., 2019; Li et al., 2019; Cen et al., 2020), the multinormial distribution is actually assumed.

The modeling objectives in Sec. 3.2.1 and 3.2.2 are quite different, as the former is  $s_{u,i}$  and the latter is  $s_u$ . In MF, we usually model  $s_{u,i}$ , while in the sequential modeling of the CG stage in industry, we model  $s_u$ . Later in Sec. 4.1, we will show that modeling  $s_u$  in MF can archive comparable or better results than modeling  $s_{u,i}$ , and boost the converging speed a lot.

# 3.3 POTENTIAL USERS MINING SYSTEM

The PUMS seems to be the twin system of RS, and similarly we could also define two modeling objectives  $s_{u,i}$  and  $s_i$ , where  $s_i$  denotes the user who interacts with the item  $i$ .  $s_{u,i}$  and  $s_i$  are assumed to follow Bernoulli and Multinomial distributions respectively. In Sec. 5 we will compare the two methods.

# 3.4 TOTALRECALL: UNIFICATION OF RS AND PUMS

Motivated by Sec. 3.2 and 3.3, we propose to model the two systems jointly. Specifically, the optimization objectives  $s_u$  and  $s_i$  in Sec. 3.2.2 and 3.3 are formulated in one framework and optimized together. Later we show that by properly choosing the model architecture and loss function, we are treating  $u$  and  $i$  equally and modeling the joint probability.

![](images/33a8e8361b3d0bcbe17b88e1a342eaf1781381ffbdf41b6a5452f6eb2ee57f0f.jpg)  
Figure 1: Model architecture of TotalRecall. Our framework will make to dot product  $\phi_{\theta}(u,i)$  converge to the logarithm of the joint probability  $p(u,i)$ , and thus can be directly applied in RS to emulate the conditional probability  $p(i|u)$ , and also in PUMS to replace  $p(u|i)$ .

# 3.4.1 MODEL ARCHITECTURE

We choose the classical two-tower architecture (Huang et al., 2013; Covington et al., 2016; Rendle et al., 2020) for two reasons: one is that users and items can be treated equivalently, and the other is that no feature crossing occurs before the final logits (See Fig. 1). So users' and items' embeddings can be inferred separately, and then approximate nearest neighbor (ANN) (Liu et al., 2004) search algorithm can be applied in online serving.

The output of two towers are  $d$ -dimensional vectors  $\pmb{u} = f_{\theta}(u) \in \mathbb{R}^d$  and  $\pmb{i} = g_{\theta}(i) \in \mathbb{R}^d$ , where  $\theta$  is the model parameter. The dot product  $\phi_{\theta}(\pmb{u}, \pmb{i}) = \langle \pmb{u} | \pmb{i} \rangle$ , or the function of it is used as the sufficient statistics of the probability distributions defined in Sec. 3.2 and 3.3 (Check App. A.1 for

details). We find that 12-normalize  $\pmb{u}$  and  $\pmb{i}$  and then rescale the dot product by the temperature  $\tau$  will lead to better and robust results:

$$
\phi_ {\theta} (\boldsymbol {u}, \boldsymbol {i}) = \frac {1}{\tau} \frac {\langle \boldsymbol {u} | \boldsymbol {i} \rangle}{\| \boldsymbol {u} \| _ {2} \| \boldsymbol {i} \| _ {2}}. \tag {2}
$$

# 3.4.2 LOSS FUNCTION

We propose the Bi-InfoNCE loss:

$$
\begin{array}{l} l = \frac {1}{| \boldsymbol {S} _ {u , i} |} \sum_ {u \in \mathbb {U}, i \in \mathbb {I}, s _ {u, i} = 1} - \log \frac {\exp (\phi_ {\theta} (u , i) - \log q (i))}{\exp (\phi_ {\theta} (u , i) - \log q (i)) + \sum_ {i ^ {\prime} \in \mathbb {I} _ {u}} \exp (\phi_ {\theta} (u , i ^ {\prime}) - \log q (i ^ {\prime}))} \\ - \log \frac {\exp \left(\phi_ {\theta} (u , i) - \log \hat {q} (u)\right)}{\exp \left(\phi_ {\theta} (u , i) - \log \hat {q} (u)\right) + \sum_ {u ^ {\prime} \in \mathbb {U} _ {i}} \exp \left(\phi_ {\theta} \left(u ^ {\prime} , i\right) - \log \hat {q} \left(u ^ {\prime}\right)\right)} \tag {3} \\ \end{array}
$$

where  $\mathbb{I}_u\subset \mathbb{I}$  and  $\mathbb{U}_i\subset \mathbb{U}$  contain hundreds or thousands of in-batch negative samples, and  $q(i)$  and  $\hat{q} (u)$  are empirical marginal distribution calculated using the training data  $S_{ui}$ . It can be shown that optimizing this loss will make  $\phi_{\theta}(u,i)$  converge to  $\log \hat{p}_{\mathrm{data}}(u,i)$  (See App. A.3 for details.).

If we only optimize the first part of Eq. 3,  $\phi_{\theta}(u,i)$  will converge to  $\log \hat{p}_{\mathrm{data}}(i|u)$ , and optimizing only the second part will let  $\phi_{\theta}(u,i)$  converge to  $\log \hat{p}_{\mathrm{data}}(u|i)$ . We refer to the two cases as Uni-InfoNCE losses.

# 4 EXPERIMENTS: RS

As stated in Sec. 3.2, two types of modeling objective are well studied in literature, and usually they use different datasets and baselines. So we compare our results with them separately using different datasets. The details of the datasets and processing methods can be found in App. A.4.

# 4.1 MF WITH BERNOULLI DISTRIBUTION

We compare TR to MF with Bernoulli distribution  $(\mathrm{MF}_b)$  in Rendle et al. (2020), as described in Sec. 3.2.1. Our implementation of  $\mathrm{MF}_b$  is a bit different from Rendle et al. (2020): we implement it using tensorflow (Abadi et al., 2016) with Adam optimizer (Kingma & Ba, 2014).

We tried L2 regularization but failed to obtain the results reported by Rendle et al. (2020). We also tried dropout (Srivastava et al., 2014), but it was unstable and need to be re-tuned for different dimensions  $d$ . Finally we find that the  $\phi_{\theta}(\pmb{u},\pmb{i})$  defined in Eq. 2 can actually act as a kind of regularization, so we use it rather than the one defined by Eq. (2) in Rendle et al. (2020).

We use Movielens-1m and Pinterest datasets, and the evaluation metrics are HiteRate@10 and NDCG@10.  $\mathrm{MF}_b$  and TR have the same two-tower architecture and same number of parameters, and the key differences are the training data and the optimizing objectives:  $\mathrm{MF}_b$  uses explicit negative sampling and optimize the binary cross-entropy loss, and TR uses implicit in-batch negative sampling and Bi-InfoNCE loss.

The hyper-parameters  $\tau$  and batch-size are chosen using the validation data. For Movielens-1m, we use  $\tau = 0.125$ , batch-size  $= 4096$  for  $\mathrm{MF}_b$ , and  $\tau = 0.111$ , batch-size  $= 512$  for TR. For Pinterest, we use  $\tau = 0.143$ , batch-size  $= 8192$  for  $\mathrm{MF}_b$ , and  $\tau = 0.167$ , batch-size  $= 8192$  for TR. We have  $d \in \{16, 32, 64, 96, 128, 196\}$  for a comprehensive comparison. The results are shown in Fig. 2 and Tab. 1.

In general,  $\mathrm{MF}_b$  and TR are comparable, and TR is slightly better in ranking as measured by NDCG@10. When measuring the training efficiency, TR converges to the optima up  $16\times$  times faster than  $\mathrm{MF}_b$  (See Fig. 3 in Appendix), so it is more practical to use TR in industrial applications with huge amount of data. Besides, the diversity of TR is also larger than  $\mathrm{MF}_b$  as shown in Fig. 3.

![](images/441d133d871d4c8b7be70ec35d3f97187d96f43e512aebfd9163411015d80130.jpg)

![](images/39069ab41bd55bd1fe45cdabd3fef46d1a0cc34fd4886c24b8bdc33fc7e4c8b3.jpg)

![](images/30e81006788dec4008b0b01a7a137afd3ed5f226eae60596c2e48d2af73dc076.jpg)  
Figure 2: Comparison of results of MF given by Bernoulli distribution and Multinormal distribution. 'MF by Rendle' is from Rendle et al. (2020), 'MF by us' is our implementation. In general these results are quite close and outperform NCF by a large margin (Rendle et al. (2020)). At the same time, the ranking performance of TR is better than MF as measured by NDCG.

![](images/90f33e32dd7832d7e9a71de5177d74775f132e7fe670399d53eeacaf4442917d.jpg)

Table 1: Baselines from Dacrema et al. (2021) and Rendle et al. (2020) and our results. The best results are highlighted in bold, the second best result is underlined.  

<table><tr><td rowspan="2">Method</td><td colspan="2">Movielens</td><td colspan="2">Pinterest</td><td rowspan="2">Result from</td></tr><tr><td>HR@10</td><td>NDCG@10</td><td>HR@10</td><td>NDCG@10</td></tr><tr><td>Popularity</td><td>0.4533</td><td>0.2542</td><td>0.2740</td><td>0.1409</td><td>Dacrema et al. (2021)</td></tr><tr><td>SLIM</td><td>0.7162</td><td>0.4468</td><td>0.8679</td><td>0.5601</td><td>Dacrema et al. (2021)</td></tr><tr><td>iALS</td><td>0.7109</td><td>0.4382</td><td>0.8761</td><td>0.5590</td><td>Dacrema et al. (2021)</td></tr><tr><td>MLP+GMF</td><td>0.7093</td><td>0.4349</td><td>0.8777</td><td>0.5576</td><td>Dacrema et al. (2021)</td></tr><tr><td>MFb by Rendle</td><td>0.7294</td><td>0.4523</td><td>0.8895</td><td>0.5794</td><td>Rendle et al. (2020)</td></tr><tr><td>MFb (ours)</td><td>0.726</td><td>0.4505</td><td>0.8925</td><td>0.5838</td><td>Fig. 2</td></tr><tr><td>TR (ours)</td><td>0.7281</td><td>0.4541</td><td>0.8936</td><td>0.5866</td><td>Fig. 2</td></tr></table>

# 4.2 SEQUENTIAL MODELING WITH MULTINORMIAL DISTRIBUTION

We compare TR to the sequential modeling with Multinormial distribution, as described in Sec. 3.2.2. This modeling method is usually used in the CG stage of industrial large-scale RS, because it can deal with huge amount of data very efficiently, and yield good results.

For simplicity, we use the popular Youtube DNN (Covington et al., 2016) implemented in Cen et al. (2020) as the backbone architecture of TR. Other complicated architectures can be easily incorporated into TR framework.

We use the same datasets as Cen et al. (2020) for a fair comparison, i.e., Amazon books data<sup>1</sup> and Taobao click data in Tianchi competition<sup>2</sup>. The data is split into train, validation and test data. The training data is usually processed as next-item prediction as in Sec. 3.1, and we extend it into

Table 2: For Amazon books data, we use  $\tau = 0.1$ , batch-size  $= 128$ . For Taobao data, we use  $\tau = 0.067$ , batch-size  $= 2048$ . ComiRec-SA and ComiRec-DR are two multi-interest models proposed in Cen et al. (2020).  

<table><tr><td></td><td colspan="2">Amazon Books Metrics@50</td><td colspan="2">Taobao Metrics@50</td></tr><tr><td></td><td>Recall</td><td>Hit Rate</td><td>Recall</td><td>Hit Rate</td></tr><tr><td>Popular</td><td>2.400</td><td>5.226</td><td>0.735</td><td>9.309</td></tr><tr><td>YouTube DNN</td><td>7.312</td><td>15.894</td><td>6.172</td><td>39.108</td></tr><tr><td>ComiRec-SA</td><td>8.467</td><td>17.202</td><td>9.462</td><td>51.064</td></tr><tr><td>ComiRec-DR</td><td>8.106</td><td>17.583</td><td>9.818</td><td>52.418</td></tr><tr><td>Youtube [SSM with l2-norm]</td><td>9.37 (+28%)</td><td>19.25 (+21%)</td><td>6.9 (+12%)</td><td>42.87 (+10%)</td></tr><tr><td>TR-Youtu [Uni-InfoNCE u2i]</td><td>10.34 (+41%)</td><td>21.15 (+33%)</td><td>7.15 (+16%)</td><td>43.79 (+12%)</td></tr><tr><td>TR-Youtu [Bi-InfoNCE]</td><td>10.12 (+38%)</td><td>20.554 (+29%)</td><td>7.14 (+16%)</td><td>43.57 (+11%)</td></tr><tr><td>TR-Youtu [next-7-prediction]</td><td>11.571 (+58%)</td><td>23.221 (+46%)</td><td>-</td><td>-</td></tr></table>

next- $n$ -item prediction, i.e., instead of predicting only the next item, we predict all the next  $n$  items. The results are shown in Tab. 2. More results are shown in Fig. 4 and 5 in Appendix.

Our experiments show that TR surpasses baselines in the three aspects.  $i$ ). Better results by a large margin: TR achieves much better results in terms of Recall and HitRate as shown in Tab. 2. The improvement is mainly attributed to l2-normalization and better training data, i.e., next- $n$ -prediction. Specifically, l2-normalization acts like a regularizer, as it confines  $\phi_{\theta}(u,i)$  inside  $[-1 / \tau ,1 / \tau ]$ . Next- $n$ -prediction alters the distribution of the training data so that it gets close to the test data (See Fig. 6 in Appendix).  $ii$ ). Improved diversity: We tried SSM with different background distributions, and compare with Uni-InfoNCE and Bi-InfoNCE, and found that Bi-InfoNCE gives better overall diversity as measured by the number of distinct items and categories of items (See Fig. 4 in Appendix). This could be due to that Bi-InfoNCE can build a more coherent vector space for  $u$  and  $i$ .  $iii$ ). Faster convergence: Compare to SSM or Uni-InfoNCE without l2-normalization, adding l2-normalization or using Bi-InfoNCE will make the models learn faster (See Fig. 4 in Appendix). This is because l2-normalization will restrict the vector space that  $u$  and  $i$  could explore, and Bi-InfoNCE can utilize the data more efficiently.

# 5 EXPERIMENTS: PUMS

In E-commerce, there are usually two practical scenarios of  $i2u$  candidates generation: one is to mine the potential users for a single item, and the other is to find users for a group of similar items. Previously, the problem is usually formulated as the binary classification for each individual case, but it is not generally applicable because the number of items or groups are quite large, and it is impractical to build so many models.

One solution is to merge all the binary classification models into one, just like  $\mathrm{MF}_b$  as in Sec. 3.2.1 and 3.3. The other is our TR framework.

We use 4 variants of  $\mathrm{MF}_b$  as the baselines. These variants are due to different negative sampling methods, i.e., for a given positive  $(u,i)$  pair, we sample the negative pairs in these ways:  $i$ ).  $\mathrm{MF}_b$  [0:8] $^3$ , we randomly sample 8 is from the item set  $\mathbb{I}$ , and construct 8 negative pairs with the given  $u;2$ ).  $\mathrm{MF}_b$  [8:0], we sample 8 us from the user set  $\mathbb{U}$ , and construct 8 negative pairs with the given  $i;3$ ).  $\mathrm{MF}_b$  [4:4], we sample 4 us and 4 is, and construct 8 negative pairs with the given  $i$  and  $u$  respectively; 4). similar to 3), but the negative samples are doubled.

# 5.1 DATASET AND EVALUATION METRICS

We use Movielens-1m because it is also used in  $\mathrm{MF}_b$  of RS, and it can be used to mine the potential users of both the single item and the group of similar items.

Table 3: Overall performance of mining the potential users of a single item. The % is omitted for precision and recall. The hyper-parameters of  $\mathrm{MF}_b$  are batch-size  $= 4096$  and  $\tau = 0.125$ , and for TR they are batch-size  $= 512$  and  $\tau = 0.125$ . We use  $d = 96$  for all models for a fair comparison. The diversity values of models with very low accuracy are omitted.  

<table><tr><td>model</td><td>precision@10</td><td>recall@10</td><td>distinct users</td></tr><tr><td>\(\mathbf{MF}_b\) [0:8]</td><td>1.47</td><td>5.61</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:0]</td><td>4.44</td><td>12.82</td><td>2046</td></tr><tr><td>\(\mathbf{MF}_b\) [4:4]</td><td>3.12</td><td>9.28</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:8]</td><td>3.02</td><td>8.64</td><td>-</td></tr><tr><td>TR [Uni-InfoNCE u2i]</td><td>1.19</td><td>4.51</td><td>-</td></tr><tr><td>TR [Uni-InfoNCE i2u]</td><td>4.35</td><td>12.78</td><td>2064</td></tr><tr><td>TR [Bi-InfoNCE]</td><td>4.58</td><td>13.82</td><td>2202</td></tr></table>

Movielens-1m records 6040 users and 3706 movies. The data is split into training, validation and test data, and we use the validation data to select the proper model configurations. The final results are reported on the test data.

We use precision@10 and recall@10 to evaluate the accuracy of different algorithms, i.e., we mine the top 10 users for a single item or a group of items, and then evaluate the precision and recall. We count the distinct users of the overall results to measure the diversity. (See App. A.5 for details about the dataset and metrics.)

# 5.2 POTENTIAL USERS OF A SINGLE ITEM

We first select movies which have been watched by at least 50 users, and it results in 2514 movies. For each of these movies, we randomly pick  $1\%$  of its users to form the valid and test datasets, with half of the movies as the validation data and the other half as the test data (See App. A.5.). The remaining data is used to train the model, and the validation data is for selecting the best hyperparameters, and the results are reported on the test data.

The results are shown in Tab. 3. 'TR [Bi-InfoNCE]' archives the best results in terms of the accuracy and diversity. High diversity implies that the advertisements can reach more users, even the non-active users, thus alleviate the advertising fatigue.

Besides, TR uses the same training data and model configurations for both RS and PUMS, and archives the best results in both cases. The learned  $u / i$  representations can be applied to the two systems without any additional modifications. In contrast,  $\mathrm{MF}_b$ 's modeling objective (Eq. 4) seems to treat  $u$  and  $i$  equally, but the outcomes depend highly on the negative sampling methods (See Tab. 3). We tried different sampling methods (e.g.,  $\mathrm{MF}_b$  [4:4] and  $\mathrm{MF}_b$  [8:8]) in the hope that  $\mathrm{MF}_b$  could do well in both  $u2i$  and  $i2u$ , but it fails to give comparable results.

Last but not least, TR learns  $16 \times$  times faster than  $\mathrm{MF}_b$ . (See Fig. 7 in Appendix).

# 5.3 POTENTIAL USERS OF A GROUP OF ITEMS

In practice, movies of different genres and different popularity will behave very differently, e.g., precision and recall of popular Action movies and unpopular Western movies will differ a lot. Movie-lens have 18 genres, and we divide the movies of the same genre into 3 categories based on their popularity, so we can split the chosen movies into 54 groups (See App. A.5 for the exact definition of groups).

Given a group and its affiliated items, we define its potential users this way: if a user interacts with one or more items in the group (excluding the items he has interacted previously), then he will be the potential users of the group. For example, given the group  $\mathbb{G} = \{i_1,i_2,i_3\}$  and the user  $u_{1}$ , if  $u_{1}$  has interacted with  $i_{1}$  in the historical data, then we cannot pick him as the potential user; if in the future data,  $u_{1}$  interact with  $i_{2}$  or  $i_{3}$ , then he will be regarded as the potential user of  $\mathbb{G}$ .

Table 4: Overall performance of mining the potential users of a group of items. The  $\%$  is omitted for precision and recall. The hyper-parameters of  $\mathrm{MF}_b$  are batch-size  $= 4096$  and  $\tau = 0.125$ , and for TR batch-size  $= 512$  and  $\tau = 0.111$ . We use  $d = 96$  for all models.  

<table><tr><td>model</td><td>precision@10</td><td>recall@10</td><td>distinct users</td></tr><tr><td>\(\mathbf{MF}_b\) [0:8] mean</td><td>5.93</td><td>1.52</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:0] sum</td><td>5.93</td><td>2.58</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:0] max</td><td>14.63</td><td>4.23</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:0] mean</td><td>17.04</td><td>4.57</td><td>246</td></tr><tr><td>\(\mathbf{MF}_b\) [4:4] mean</td><td>11.11</td><td>2.63</td><td>-</td></tr><tr><td>\(\mathbf{MF}_b\) [8:8] mean</td><td>12.78</td><td>3.05</td><td>-</td></tr><tr><td>TR [Uni-InfoNCE u2i] mean</td><td>4.63</td><td>1.38</td><td>-</td></tr><tr><td>TR [Uni-InfoNCE i2u] mean</td><td>16.85</td><td>7.25</td><td>262</td></tr><tr><td>TR [Bi-InfoNCE] sum</td><td>12.41</td><td>6.79</td><td>-</td></tr><tr><td>TR [Bi-InfoNCE] max</td><td>16.48</td><td>7.01</td><td>-</td></tr><tr><td>TR [Bi-InfoNCE] mean</td><td>20.56</td><td>7.58</td><td>258</td></tr></table>

Different groups have different number of movies, e.g., the group 'Drama-top  $10\%$  contains 93 movies, while 'Documentary top  $10\%$  has only 3 movies, and their precision and recall will also differ a lot. We report the averaged precision and recall over all groups in Tab. 4.

For small dataset like movie-lens-1m, the model generates the potential users of a group this way: for each item in the group, we first calculate  $\phi_{\theta}(\pmb{u},\pmb{i})$  of Eq. 2, and then the score  $\hat{p}_{\mathrm{TR}}(\pmb{u},\pmb{i}) = \exp (\phi_{\theta}(\pmb{u},\pmb{i}))$  for TR and  $\hat{p}_{\mathrm{MF}}(\pmb{u},\pmb{i}) = \sigma (\phi_{\theta}(\pmb{u},\pmb{i}))^4$  for  $\mathrm{MF}_b$ , with respect to the candidate users. Thus each candidate user will have several scores, and then we aggregate the scores to obtain a match score between the user and the group, and finally pick the top- $k$  users. The aggregation methods could be max, mean or sum.

In Tab. 4,  $\mathrm{MF}_b$  [0:8] and 'TR [Uni-InfoNCE  $u2i$ ] are specifically for  $u2i$  CG of RS, so they give the worst results here. Models tailored for  $i2u$  ( $\mathrm{MF}_b$  [8:0] and 'TR [Uni-InfoNCE  $i2u$ ]) and the general model for bidirectional CG ('TR [Bi-InfoNCE]) perform much better.

The aggregation methods also affect the accuracy, and mean  $> \max > \sum$ . If a user has scores with many items of the group, the sum could be high even if individual scores are low, and this also implies that the user has few interactions with the items of the group in the historical data. So it would not be suitable that sum ranks this user high. In contrast, mean and max will not have this problem, and mean considers more scores than just the max one and gives best results.

# 6 CONCLUSION

In this paper, CF and the sequential modeling (i.e., next-item prediction) of RS, and PUMS are abstracted into a unified  $u - i$  interaction matrix  $S_{ui}$ , and thus are treated equivalently as deriving some probabilities for the unknown entries from the known entries of  $S_{ui}$ . We also show their connections and differences based on probability theory. Built upon these observations, we propose the TR framework to unify them and solve  $S_{u,i}$  all together.

In TR,  $u$  and  $i$  are treated equally, and the joint probability  $p(u,i)$  is actually derived, instead of the conditional probability  $p(i|u)$  or  $p(u|i)$ . Therefore, the learned representations of  $u$  and  $i$  in TR can be used in both  $u2i$  and  $i2u$  CG.

Our extensive experiments show that TR can archive comparable and better results in each domain, and learns  $16 \times$  times faster than  $\mathrm{MF}_b$ . Besides, TR generates diversified candidates.

In the future, we plan to study the vector space of TR in depth, and try to unify different objects and users into one space, for example, products, merchants, brands, categories of products. In such a space, everything is linked, and candidates generation can be applied on any pair of objects.

# REFERENCES

Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: a system for large-scale machine learning. In OSDI, volume 16, pp. 265-283, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Jan Roelf Bult and Tom Wansbeek. Optimal selection for direct mail. Marketing Science, 14(4): 378-394, 1995.  
Yukuo Cen, Jianwei Zhang, Xu Zou, Chang Zhou, Hongxia Yang, and Jie Tang. Controllable multi-interest framework for recommendation. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2942-2951, 2020.  
Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector machines. ACM transactions on intelligent systems and technology (TIST), 2(3):1-27, 2011.  
Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm SIGkdd international conference on knowledge discovery and data mining, pp. 785-794. ACM, 2016.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597-1607. PMLR, 2020.  
Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems, pp. 7-10. ACM, 2016.  
Paul Covington, Jay Adams, and Emre Sargin. Deep neural networks for youtube recommendations. In Proceedings of the 10th ACM conference on recommender systems, pp. 191-198, 2016.  
Maurizio Ferrari Dacrema, Simone Boglio, Paolo Cremonesi, and Dietmar Jannach. A troubling analysis of reproducibility and progress in recommender systems research. ACM Transactions on Information Systems (TOIS), 39(2):1-49, 2021.  
Simon Funk. Netflix update: Try this at home. URL http://sifter.org/~simon/journal/20061211.html.  
Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. arXiv preprint arXiv:2104.08821, 2021.  
Xue Geng, Hanwang Zhang, Jingwen Bian, and Tat-Seng Chua. Learning image and user features for recommendation in social networks. In Proceedings of the IEEE International Conference on Computer Vision, pp. 4274-4282, 2015.  
Tao Gui, Peng Liu, Qi Zhang, Liang Zhu, Minlong Peng, Yunhua Zhou, and Xuanjing Huang. Mention recommendation in twitter with cooperative multi-agent reinforcement learning. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 535-544, 2019.  
Ido Guy. People recommendation on social media. In Social information access, pp. 570-623. Springer, 2018.  
F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis), 5(4):1-19, 2015.  
Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web, pp. 173-182, 2017.

David W Hosmer Jr, Stanley Lemeshow, and Rodney X Sturdivant. Applied logistic regression, volume 398. John Wiley & Sons, 2013.  
Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pp. 2333-2338, 2013.  
Sebastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. On using very large target vocabulary for neural machine translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1-10, 2015.  
Christopher C Johnson. Logistic matrix factorization for implicit feedback data. Advances in Neural Information Processing Systems, 27(78):1-9, 2014.  
YongSeog Kim and W Nick Street. An intelligent system for customer targeting: a data mining approach. Decision Support Systems, 37(2):215-228, 2004.  
YongSeog Kim, W Nick Street, Gary J Russell, and Filippo Menczer. Customer targeting: A neural network approach guided by genetic algorithms. Management Science, 51(2):264-276, 2005.  
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. Multi-interest network with dynamic routing for recommendation atmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 2615-2623, 2019.  
Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. Variational autoencoders for collaborative filtering. In Proceedings of the 2018 world wide web conference, pp. 689-698, 2018.  
Greg Linden, Brent Smith, and Jeremy York. Amazon.com recommendations: Item-to-item collaborative filtering. IEEE Internet computing, 7(1):76-80, 2003.  
Ting Liu, Andrew W Moore, Alexander G Gray, and Ke Yang. An investigation of practical approximate nearest neighbor algorithms. In NIPS, volume 12, pp. 2004. Citeseer, 2004.  
Siaw Ling Lo, Raymond Chiong, and David Cornforth. Ranking of high-value social audiences on twitter. Decision Support Systems, 85:34-48, 2016.  
Andriy Mnih and Russ R Salakhutdinov. Probabilistic matrix factorization. In Advances in neural information processing systems, pp. 1257-1264, 2008.  
Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 188-197, 2019.  
Yabo Ni, Dan Ou, Shichen Liu, Xiang Li, Wenwu Ou, Anxiang Zeng, and Luo Si. Perceive your users in depth: Learning universal user representations from multiple e-commerce tasks. arXiv preprint arXiv:1805.10727, 2018.  
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
Guansong Pang, Shengyi Jiang, and Dongyi Chen. A simple integration of social relationship and text data for identifying potential customers in microblogging. In International Conference on Advanced Data Mining and Applications, pp. 397-409. Springer, 2013.  
Marco Pennacchiotti and Ana-Maria Popescu. Democrats, republicans and starbucks afflictionados: user classification in twitter. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 430-438, 2011.

David MW Powers. Applications and explanations of zipf's law. In New methods in language processing and computational natural language learning, 1998.  
Steffen Rendle, Walid Krichene, Li Zhang, and John Anderson. Neural collaborative filtering vs. matrix factorization revisited. In Fourteenth ACM Conference on Recommender Systems, pp. 240-248, 2020.  
Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 3859-3869, 2017.  
Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1):1929-1958, 2014.  
Xiaoyuan Su and Taghi M Khoshgoftaar. A survey of collaborative filtering techniques. Advances in artificial intelligence, 2009, 2009.  
Liyang Tang, Zhiwei Ni, Hui Xiong, and Hengshu Zhu. Locating targets through mention in twitter. World Wide Web, 18(4):1019-1049, 2015.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pp. 5998-6008, 2017.  
Jun Wang, Arjen P De Vries, and Marcel JT Reinders. Unifying user-based and item-based collaborative filtering approaches by similarity fusion. In Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval, pp. 501-508, 2006.  
Ji Yang, Xinyang Yi, Derek Zhiyuan Cheng, Lichan Hong, Yang Li, Simon Xiaoming Wang, Taibai Xu, and Ed H Chi. Mixed negative sampling for learning two-tower neural networks in recommendations. In Companion Proceedings of the Web Conference 2020, pp. 441-447, 2020.  
Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, and Ed Chi. Sampling-bias-corrected neural modeling for large corpus item recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems, pp. 269-277, 2019.  
Chang Zhou, Jianxin Ma, Jianwei Zhang, Jingren Zhou, and Hongxia Yang. Contrastive learning for debiased candidate generation in large-scale recommender systems. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 3985-3995, 2021.  
Guorui Zhou, Xiaogiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1059-1068. ACM, 2018.  
Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 1079-1088, 2018.
