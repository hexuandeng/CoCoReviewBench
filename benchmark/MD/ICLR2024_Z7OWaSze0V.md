# UNIFYING USER PREFERENCES AND CRITIC OPINIONS: A MULTI-VIEW CROSS-DOMAIN ITEM-SHARING RECOMMender SYSTEM

Anonymous authors

Paper under double-blind review

# ABSTRACT

Traditional cross-domain recommender systems often assume user overlap and similar user behavior across domains. However, these presumptions may not always hold true in real-world situations. In this paper, we explore an less explored but practical scenario: cross-domain recommendation with distinct user groups, sharing only item-specific data. Specifically, we consider user and critic review scenarios. Critic reviews, typically from professional media outlets, provide expert and objective perspectives, while user reviews offer personalized insights based on individual experiences. The challenge lies in leveraging critic expertise to enhance personalized user recommendations without sharing user data. To tackle this, we propose a Multi-View Cross-domain Item-sharing Recommendation (MCIR) framework that synergizes user preferences with critic opinions. We develop separate embedding networks for users and critics. The user-rating network leverage a variational autoencoder to capture user scoring embeddings, while the user-comment network use pretrained text embeddings to obtain user commentary embeddings. In contrast, critic network utilize multi-task learning to derive insights from critic ratings and reviews. Further, we use Graph Convolutional Network layers to gather neighborhood information from the user-item-critic graph, and implement an attentive integration mechanism and cross-view contrastive learning mechanism to align embeddings across different views. Real-world dataset experiments validate the effectiveness of the proposed MCIR framework, demonstrating its superiority over many state-of-the-art methods.

# 1 INTRODUCTION

Cross-domain recommendation systems have attracted considerable attention recently due to their ability to mitigate issues such as data sparsity in recommender systems by leveraging auxiliary information from associated domains (Zhu et al., 2020; Hu et al., 2018; Kang et al., 2019; Chen et al., 2022). Traditional cross-domain recommenders often presume an overlap of users and similar user types across domains (Singh & Gordon, 2008; Hu et al., 2018; Yan et al., 2019), and they typically share user-related information across domains to enhance recommendation performance. However, real-world scenarios may present entirely different user types across auxiliary and target domains, and sharing user-centric information might not be practical or permissible due to privacy issues or operational constraints. Addressing this, our study delves into a less charted yet practical area within cross-domain recommendation systems where solely item-related information is shared between disparate user types in the auxiliary and target domains. Specifically, we concentrate on the user and critic review scenarios. Noticeably, without loss of generality, our framework can be easily applied in analogous item-sharing situations.

The fast growth of review websites like Metacritic (met) and ROTTEN TOMATOES (rot) offers a space where users can not only share their perspectives but also gain from the expert evaluations of critics. Critic reviews, predominantly sourced from authoritative media institutions, can give professional and objective insights, serving as an invaluable resource for decision-making users. In contrast, user reviews often echo personalized, experience-centric perspectives. The substantial impact of critic reviews on user preferences is well documented in existing marketing research (Basuroy et al., 2003; Tsao, 2014). Researchers also underscore the divergent nature of critiques offered by critics and users (Santos et al., 2019; Dillio, 2013; Parikh et al., 2017), such as "Experts Write More Complex and Detached Reviews, While Amateurs More Accessible and Emotional Ones". This

![](images/8604fbf72205e7e74882042f4088a1acb8e15ea78c15801597ddf8f5c471b3d2.jpg)  
Guardians of the Galaxy

Summary: Our beloved band of misfits are looking a bit different these days. Peter Quill, still reeling from the loss of Gamora, must rally his team around him to defend the universe along with protecting one of their own...

# Critic Reviews

![](images/70d21fc6a90c7a4529ec14877808aa04586260571e01f00abd1a833816d13260.jpg)

James Gunn bets big that you love each and every one of these Guardians. It's a movie about friendship and the love these characters have for each other and risking...

![](images/a383ca874e4cdec953988fff1a7869f8c72f400ab9e08f0fbaef587bb1dda461.jpg)

For all his puerile instincts, Gunn is able to create stakes in this film that feel real and meaningful - perhaps because of the care that has gone into

# User Reviews

![](images/aa9aca9469407c439aa9147475cbb20be8d93c407beb60a7a6b31f770f799da4.jpg)

I don't know what some of the critics are smoking. This is a great movie. This is the product of taking time to do it right. Probably not better than the first...

![](images/a37d1dad034beba278edffd2057cfd2475031d39686df99b17bd4a7c9031a7bd.jpg)

Really fun movie. Like all MCU movies, you can expect it to be fun. Great soundtrack, awesome theatrics andcinematics, comedic relief

![](images/f485f9fc83ec997d4e8daed43f911ed6863ec01246f93dcef864051dc093ac6c.jpg)  
Figure 1: An example of the user and critic review scenario (left) and the illustration of cross-domain recommendation without sharing users (right).

![](images/197964c5de77c4cfe1dfe4329d5dc46c8fb6c4a2538a017d92cfcf3dfec910ee.jpg)

distinction accentuates the importance of integrating expert knowledge from the critic domain to enhance the quality and reliability of recommendations in the user domain.

However, transferring critic information into the user domain presents several challenges. First, user comments can be diverse and may lack correlation with item properties (Parikh et al., 2017), making it crucial to learn comprehensive user preferences from both rating and comment views. Second, as illustrated in Figure 1, critic reviews can also greatly differ from each other, making it necessary to identify the most influential critiques for users. Third, there are no direct links between users and critics, making it necessary to use items as a bridge to capture consistency. Hence, leveraging critic domain information to enhance personalized user recommendations becomes a complex yet rewarding task (Gao et al., 2019; 2021).

To address these challenges, we propose a novel Multi-view Cross-domain Item-sharing Recommendation (MCIR) framework that effectively synthesizes user preferences and critic opinions. We initially design unique embedding networks tailored to users and critics for learning multi-view information. The user-rating network employs a variational autoencoder to capture user scoring embeddings, while the user-comment network utilizes pretrained text embeddings to obtain user commentary embeddings. Conversely, the critic network employs multi-task learning to derive insights from critic ratings and comments synchronously. Based on the multi-view representations, we devise an attentive integration mechanism to obtain comprehensive item representations. Further, we extract detailed neighborhood information from the user-item-critic graph and propose a cross-view contrastive learning method to harmonize embeddings across different views. Extensive experiments on real-world datasets demonstrate that our proposed MCIR framework outperforms state-of-the-art methods, effectively addressing the challenges of cross-domain item-sharing recommendations.

# 2 RELATED WORKS

Cross-domain recommendation (CDR) is a widely-used technique to counter challenges like data sparsity by incorporating data from auxiliary domains. CDR mainly encompasses collaborative and content-based approaches (Mirbakhsh & Ling, 2015; Gao et al., 2021).

On one hand, collaborative CDR methods draw on interaction data across domains. For example, Collective Matrix Factorization (CMF) (Singh & Gordon, 2008) is a classic CDR approach, assuming a global user factor matrix across all domains while factorizing multiple domain matrices. Differently, Man et al. (2017) used a multi-layer perceptron to capture nonlinear mapping across domains. Similarly, DCDCSR (Zhu et al., 2020) combined user latent vectors and learns a mapping function between target domains. CoNet (Hu et al., 2018) used cross-connections between neural networks to transfer and consolidate knowledge. SSCDR (Kang et al., 2019) merged collaborative filtering with sparse subspace clustering to enhance recommendation systems by aligning latent subspaces. DeepAPF (Yan et al., 2019) modeled user-video interactions by capturing cross-site and site-specific interests with an attentional network. BiTGCF (Liu et al., 2020) used a feature propagation layer for high-order connectivity within a domain's user-item graph, enabling knowledge transfer. CATART (Li et al., 2023) and COAST (Zhao et al., 2023) improved performance across domains through representation learning, embedding transfer, and aligning user interests. Besides, for addressing privacy, PriCDR (Chen et al., 2022) employed a privacy-preserving CDR model.

![](images/24b6420da4ab3feddcb2bf1d5bb0ddb83d038a4b1f1340ed34bcf974765225e1.jpg)  
Figure 2: The network architectures of the MCIR Framework.

On the other hand, content-based CDR methods utilize user or item attributes from auxiliary domains. For instance, LFM (Agarwal et al., 2011) used multi-modal user profiles, and CKE (Zhang et al., 2016) enhanced item embeddings with textual, structural, and visual knowledge. CATN (Zhao et al., 2020) modeled user preference transfer from reviews, while CCDR (Xie et al., 2022) addressed popularity bias through a diversified preference contrastive learning.

Although Cross-Domain Recommendation (CDR) methods have achieved notable successes in academic literature, particularly for user-sharing scenarios, there remains a gap in addressing contexts where only items are shared across varied user types. Recently, Gao et al. (2019; 2021) performed cross-domain recommendations without sharing user-sided data, but these efforts persisted in assuming uniform user types across domains. Contrary to preceding studies, our work uniquely concentrates on scenarios where the auxiliary and target domains are characterized by distinct user types, with only items being a consistent shared entity across the domains.

# 3 THE MCIR FRAMEWORK

In this section, we delve into the technical specifics of our proposed Multi-View Cross-domain Item-sharing Recommendation (MCIR) framework. Initially, we present the relevant notations and outline our framework. Subsequently, we provide a detailed explanation of the multi-view learning process in the MCIR framework. Finally, we combine the different views to generate recommendations.

# 3.1 PROBLEM DEFINITION

In this study, we deal with two distinct domains of data. Let's assume there are  $N_{u}$  users,  $N_{v}$  items, and  $N_{w}$  critics. In the user domain, we define the rating matrix  $R \in \mathbb{R}^{N_u \times N_v}$  as the composition of historical ratings represented by real numbers, with missing entries denoted by 0. Let  $R_{i}$  represent the rating records of the  $i$ -th user across all items. Besides, we use  $Y_{ij}$  to represent the comment text by user  $i$  on item  $j$  and  $Y_{i} = \{Y_{ij} | I(Y_{ij}) = 1\}$  to denote the set of user  $i$ 's comments. In the critic domain, we similarly use  $R_{lj}^{c}$  and  $Y_{lj}^{c}$  to represent the rating and comment of critic  $l$  on item  $j$ , respectively. Note that in the following text, variables marked with a superscript c (e.g.,  $Y_{lj}^{c}$  and  $R_{lj}^{c}$ ) all denote variables within the critic domain. Then the problem can then be defined as:

Definition 1. (Cross-domain Item-sharing Problem.) Consider the auxiliary domain data (comprising rating matrix  $R^c$  and comment records  $Y^c$ ) and the target domain data ( $R$  and  $Y$ ). The objective is to make item recommendations to target users by harnessing information from both the target and auxiliary domains. This scenario is characterized by the exclusive sharing of item-relevant data, with no user overlap between the target and auxiliary domains.

# 3.2 FRAMEWORK OVERVIEW

As depicted in Figure 2, our MCIR framework adeptly performs multi-view representation learning to precisely model cross-domain interactions. Initially, the input  $R_{i}$  is encoded from a user-rating perspective, producing latent rating vectors  $u_{i}$  that embody user preferences from their rating histories. Following this, comment records  $Y_{i}$  are used to ascertain latent user-comment vectors  $w_{i}$ . These two views are subsequently merged to formulate a unified latent user vector. In the critic domain, a multi-task network is utilized to simultaneously discern both the critic-rating vector  $u_{i}^{c}$  and the critic/comment vector  $w_{l}^{c}$ . Concerning items, latent item-text vectors are derived from item summary texts, and latent item-rating vectors are gleaned from both user and critic domains. Our framework further leverages the user-item-critic graph to unearth latent neighborhood vectors, enhancing the capture of user preferences and item attributes from a neighborhood perspective. To ensure alignment across all views, a cross-view contrastive learning mechanism is proposed. Concluding the framework's operation, the user decoder network seamlessly performs the rating prediction, facilitating robust and informed item recommendations.

# 3.3 METHODOLOGY

To tackle the challenge of cross-domain recommendation in the absence of shared users, we devise a novel solution framework, termed MCIR, which harnesses multi-view knowledge to adeptly discern user preferences and item properties across various domains.

User-Rating Embedding Network. Given the diverse nature of user comments and their potential limited correlation with item property evaluations (Santos et al., 2019; Dillio, 2013; Parikh et al., 2017), we opt for independent learning of two distinct types of latent embeddings from both user-rating and user-comment perspectives, respectively. Initially, from the user-rating view and inspired by the famous matrix factorization models for collaborative filtering (Mnih & Salakhutdinov, 2008), our goal is to decompose the rating matrix into two latent representations  $U \in \mathbb{R}^{d \times N_u}$  and  $V \in \mathbb{R}^{d \times N_v}$  in a shared low-dimensional space of dimension  $d$ . Hence we can use  $u_i \in \mathbb{R}^d$  and  $v_j \in \mathbb{R}^d$  to represent the latent factors of user  $i$  and item  $j$  from the rating perspective.

To guarantee robust and efficient embedding learning process, we utilize the Variational Autoencoder (VAE) due to its well-documented proficiency in generating accurate and robust recommendations (Liang et al., 2018; Ma et al., 2019). VAE can generate more reliable and effective embeddings rather than common Autoencoders (AE) in recommender systems (Khawar et al., 2020). Accordingly, the encoder network  $f_{\psi}(\cdot)$  of VAE, named User-Rating Embedding Network, is structured as multi-layer perceptrons (MLPs) with  $T$  layers. The user-rating vector  $u_{i}$  is thus expressed as a multivariate Gaussian variable with mean  $\mu$  and covariance  $\Sigma$ , computed as follows:

$$
h _ {1} = g \left(W _ {1} f _ {d r o p} \left(R _ {i}\right) + b _ {1}\right),
$$

$$
h _ {t} = g \left(W _ {t} h _ {t - 1} + b _ {t}\right), \quad t \in [ 2, 3, \dots , T ],
$$

$$
\mu_ {i} = W _ {T} h _ {T} + b _ {T}, \quad \operatorname {d i a g} \left\{\Sigma_ {i} \right\} = W _ {T} ^ {\prime} h _ {T} + b _ {T} ^ {\prime}, \tag {1}
$$

where  $\text{diag}\{\Sigma_i\}$  denotes the diagonal elements of the matrix  $\Sigma$ , with all other elements set to 0.  $g(\cdot)$  represents the activation function, and  $f_{drop}(\cdot)$  signifies the drop-out layer. Employing a drop-out strategy on  $R_i$  markedly reduces the overfitting problem, leading to more robust representations.

The distribution  $p(u_i|R_i)$  is then derived as  $p(u_i|R_i)\sim \mathcal{N}(\mu_i,\Sigma_i)$ . By utilizing the reparameterization trick (Rezende et al., 2014), the sampling on variable  $u_{i}$  during the gradient backpropagation process is avoided, and  $u_{i}$  is obtained via  $u_{i} = \mu_{i} + \epsilon \Sigma_{i}$ , where  $\epsilon \sim \mathcal{N}(0,I)$  is the multivariate Gaussian noise. It's important to note that unobserved entries are represented as 0 in the rating matrix. Hence, using the entire record vector  $R_{i}$  as network input enables us to learn not only user evaluation scores but also interactive preferences for item selection.

User-Comment Embedding Network. Upon transposing the rating information into a latent dimension, we then need to extract representations from the comment view. Specifically, the latent user-comment vector  $w_{i}\in \mathbb{R}^{d}$  is initialized randomly and updated during the training process.

First, we employ the famous pretrained Sentence-BERT (Reimers & Gurevych, 2019) to extract paragraph embeddings  $y_{ij} \in \mathbb{R}^b$  for each comment text  $Y_{ij}$ . The framework of Sentence-BERT, trained to discern semantically similar and dissimilar sentence pairs, aids in extracting consistent latent user-comment vector  $w_i$  for users with analogous review patterns. Simultaneously, acknowledging the item summary's encapsulation of crucial properties, Sentence-BERT is utilized to yield the pretrained embedding  $s_j \in \mathbb{R}^b$  for the summary text of item  $j$ . Both  $y_{ij}$  and  $s_j$  remain fixed in subsequent processes. For further gleaning item property information that is valuable for recommendations,

a linear layer  $f_{s}(\cdot)$  is introduced to convert the original item summary embedding into the latent item-summary vector  $v_{j} = f_{s}(s_{j})\in \mathbb{R}^{d}$ .

Since user comments encapsulate both user preferences and item characteristics, we merge the latent user-comment vector  $w_{i}$  and item-summary vector  $v_{j}$  to reconstruct the pretrained embedding  $y_{ij}$  for user  $i$ 's review on item  $j$ . A 2-layer MLP network  $f_{z}(\cdot)$  is deployed to produce the reconstructed comment vector  $z_{ij} \in \mathbb{R}^{b}$  for review  $Y_{ij}$  as follows, where  $|\cdot|$  denotes concatenation.

$$
z _ {i j} = f _ {z} \left(w _ {i} \mid v _ {j}\right). \tag {2}
$$

The reconstruction aspires  $z_{ij}$  to closely mirror  $y_{ij}$ . Discrepancies between  $z_{ij}$  and  $y_{ij}$  are quantified using Mean Square Error (MSE) loss:

$$
\mathcal {L} _ {M S E} = \Sigma_ {i} \Sigma_ {j} I \left(Y _ {i j}\right) \left(\left\| y _ {i j} - z _ {i j} \right\| ^ {2} + \lambda \left\| w _ {i} \right\| ^ {2} + \lambda \left\| v _ {j} \right\| ^ {2}\right), \tag {3}
$$

where  $I(Y_{ij})$  is the identity function which equals 1 if user  $i$  has reviewed item  $j$ , otherwise 0. The inclusion of an L2 regularization term for latent vectors with the weight hyperparameter  $\lambda$  enhances the model's robustness. This loss function aids in aligning the reconstructed comment vector closely with the original, ensuring the infusion of efficient text information into  $w_i$  and  $v_j$ .

Critic Embedding Network. In this subsection, we aim to learn latent critic representations. Given the unique nature of critics—marked by professional insights and objective commentaries—two distinct features set them apart from ordinary users. On one hand, critics, often affiliated with credible media entities, lack the liberty to select items for review based on personal preferences. Their ratings mainly echo their assessments of the items. On the other hand, a significant correlation exists between critics' scores and their explanatory comments (Santos et al., 2019; Dillio, 2013). Therefore, here we employ a multi-task way to concurrently learn the latent critic-rating and critic-comment vectors.

For the critic rating prediction task, we define  $u_{l}^{c} \in \mathbb{R}^{d}$  as the latent variable for critic  $l$ . Following the matrix factorization model, critic-item ratings  $\hat{R}_{lj}^{c}$  are predicted through the inner product of the latent critic-rating vector  $w_{l}^{c}$  and the item vector  $v_{j}$ , expressed as  $\hat{R}_{lj}^{c} = (u_{l}^{c})^{T}v_{j}$  (Koren et al., 2009; Xue et al., 2017). Noticed that here we employ the previously outlined latent item-summary vector  $v_{j}$  (Equation 2) to maintain coherence across the auxiliary and target domains.

Considering the robust association between critics' ratings and comments, a transformation of the latent critic vector  $u_{l}^{c}$  into the critic/comment vector  $w_{l}^{c} = f_{w}(u_{l}^{c})$  is performed using a single-layer MLP network  $f_{w}(\cdot)$ . Again Sentence-BERT is used to obtain the pretrained review embedding  $y_{lj}^{c}$  for critic review  $Y_{lj}^{c}$ . Similarly to the User-Comment Embedding Network, a 2-layer MLP network  $f_{c}(\cdot)$  processes the merged vector  $w_{l}^{c}||v_{j}$  to recreate the original critic comment embedding  $y_{lj}^{c}$ . This leads to the following mapping function for the latent critic and item-summary representations:

$$
z _ {l j} ^ {c} = f _ {c} \left(w _ {l} ^ {c} \mid \mid v _ {j}\right). \tag {4}
$$

To quantify the variations from both rating and comment perspectives, a multi-task loss is employed:

$$
\mathcal {L} _ {\text {M u l t i}} = \Sigma_ {l} \Sigma_ {j} I \left(Y _ {l j} ^ {c}\right) \left(\| \hat {R} _ {l j} ^ {c} - R _ {l j} ^ {c} \| ^ {2} + \| y _ {l j} ^ {c} - z _ {l j} ^ {c} \| ^ {2} + \lambda \| w _ {l} ^ {c} \| ^ {2} + \lambda \| v _ {j} \| ^ {2}\right). \tag {5}
$$

Attentive Integrated Mechanism. Given that only the items are shared between the critic and user domains, it is crucial to leverage items as a conduit for message exchange. In this subsection, an attentive integrated mechanism is proposed, aimed at generating the attentive item vector  $v_{j}^{a}$ . This vector seamlessly combines both the item-summary information and the critics' commentary information, thereby enhancing the recommendations within the user domain.

Given the potential diversity in critics' perspectives on different aspects of an item, leading to substantial disagreements in their reviews, a thorough understanding of item characteristics is essential. This understanding aids in the generation of attentive item embeddings from the wide range of critic comments. To achieve this integration, an attention layer is introduced to merge the item-summary vector and pertinent latent critic vectors. Let  $L_{j}$  be the set containing the indices of critics who have reviewed item  $j$ . For the convenience of establishing formulas, let  $w_0^c = v_j$  and include index 0 in the set  $L_{j}$ . Then the attentive integrated mechanism is articulated as:

$$
\begin{array}{l} \alpha_ {j l} = \frac {\exp \left(\left(W _ {k e y} v _ {j}\right) ^ {T} \left(W _ {q u e r y} w _ {l} ^ {c}\right)\right)}{\sum_ {l ^ {\prime} \in L _ {j}} , \exp \left(\left(W _ {k e y} v _ {j}\right) ^ {T} \left(W _ {q u e r y} w _ {l ^ {\prime}} ^ {c}\right)\right)}, l \in L _ {j}, \\ v _ {j} ^ {a} = \Sigma_ {l \in L _ {j}} \alpha_ {j l} \left(W _ {\text {v a l u e}} w _ {l} ^ {c}\right). \tag {6} \\ \end{array}
$$

In Equation 6,  $\alpha_{jl}$  denotes the attention weights, determined by the compatibility between the item-summary vector  $v_{j}$  and critic vectors  $w_{l}^{c}$ . The symbols  $W_{\mathrm{key}}$ ,  $W_{\mathrm{query}}$  and  $W_{\mathrm{value}} \in \mathbb{R}^{d \times d}$  represent the learnable weight matrices for the key, query, and value in the attention mechanism, respectively. The final attentive item embedding  $v_{j}^{a}$  is procured as the weighted sum of the critic-comment vectors, with the attention scores  $\alpha_{jl}$  as weights. Practically, for large  $|L_{j}|$ , a subset of the critics is sampled in each batch for training, employing a random sampling strategy. This strategy is not only efficient but also serves as a preventive measure against over-fitting. Through this attentive integration mechanism, diverse critic information is effectively amalgamated, encompassing both the item-summary information and critics' opinions, thus offering a more enriched and comprehensive item representation for recommendations in user domain.

User Decoder Network. The next phase is to decode the latent user and item vectors obtained from multi views back into the original user-item rating space. Specifically, we first combine the latent user-rating vector and user-comment vector to produce the final latent user representation  $p_i$ . As for the items, since the textual information may not include all the useful information for recommendations, we add a learnable variable  $v_j^b \in \mathbb{R}^d$  to the attentive vector  $v_j^a$  to obtain the final item representation  $q_j$ .  $v_j^b$  is initialized randomly and updated throughout the training process.

$$
p _ {i} = u _ {i} + w _ {i}, \quad q _ {j} = v _ {j} ^ {a} + v _ {j} ^ {b}. \tag {7}
$$

In Equation 7, the addition operation is preferred over concatenation to maintain a consistent dimension of the latent vector. This approach also lays the foundation for the addition of more potential views. The user decoder network can hence be defined by:

$$
\hat {R} _ {i j} = p _ {i} ^ {T} q _ {j}. \tag {8}
$$

The VAE loss can be formulated as:

$$
\mathcal {L} _ {V A E} = - \Sigma_ {i} \left(\mathbb {E} \left[ \log p \left(R _ {i} \mid p _ {i}\right) \right] + \mathbb {K L} \left(p \left(u _ {i} \mid R _ {i}\right) \mid \mathcal {N} (0, I)\right)\right). \tag {9}
$$

The first component in Equation 9 is the reconstruction loss. Following the method in VAE, Monte Carlo sampling aids in estimating expected values. However, given that  $R_{i}$  is not strictly binary, cross-entropy loss application is not straightforward for estimating the term  $\mathbb{E}[\log p(R_i|p_i)]$ . As an alternative, inspired by Xue et al. (2017), a softmax layer  $\hat{R}_i' = \text{softmax}(\hat{R}_i)$  is first applied, and a novel reconstruction loss is defined as:

$$
\log p \left(R _ {i} \mid p _ {i}\right) = - \Sigma_ {j} \frac {R _ {i j}}{\max \left(R _ {i}\right)} \log \hat {R} _ {i j} ^ {\prime}. \tag {10}
$$

Compared to the loss in (Xue et al., 2017), our proposed loss in Equation 10 computes the probability across the entire rating vector  $R_{i}$ , rather than focusing on a specific item.

Graph Embedding Network and Cross-view Contrastive Learning. In addition to rating and comment perspectives, the neighborhood perspective can also offer important insights into user preferences, item characteristics, and critic opinions (He et al., 2020; Mao et al., 2021). In order to capture these relationships and uphold consistency across domains, a user-item-critic graph is formulated to learn neighborhood embeddings. The nodes of this graph include all users, items, and critics, and edges denote positive interactions. We filter these positive interactions by selecting ratings above a certain threshold. The neighborhood information enables the integration of user and critic preferences within a unified view, with items serving as the connecting bridge.

Specifically, the efficient and widely recognized LightGCN (He et al., 2020) architecture is employed to obtain latent graph vectors  $e_i^u$ ,  $e_l^c$ , and  $e_j^v$  for user  $i$ , critic  $l$ , and item  $j$  respectively. LightGCN eliminates feature transformations and non-linear activation functions. Let  $\mathcal{N}_i, \mathcal{N}_l$ , and  $\mathcal{N}_j$  signify the neighborhood node sets of node  $i$ ,  $l$ , and  $j$  respectively. The message passing layer is denoted as:

$$
e _ {i, (k + 1)} ^ {u} = \Sigma_ {j \in \mathcal {N} _ {i}} \frac {e _ {j , k} ^ {v}}{\sqrt {| \mathcal {N} _ {i} |} \sqrt {| \mathcal {N} _ {j} |}}, e _ {l, (k + 1)} ^ {c} = \Sigma_ {j \in \mathcal {N} _ {l}} \frac {e _ {j , k} ^ {v}}{\sqrt {| \mathcal {N} _ {l} |} \sqrt {| \mathcal {N} _ {j} |}},
$$

$$
e _ {\mathcal {j}, (k + 1)} ^ {u} = \quad \Sigma_ {i \in \mathcal {N} _ {i}} \frac {e _ {i , k} ^ {u}}{\sqrt {| \mathcal {N} _ {i} |} \sqrt {| \mathcal {N} _ {j} |}} + \Sigma_ {l \in \mathcal {N} _ {l}} \frac {e _ {l , k} ^ {c}}{\sqrt {| \mathcal {N} _ {l} |} \sqrt {| \mathcal {N} _ {j} |}}, \tag {11}
$$

where  $k$  represents the ordinal number of a GCN layer and the final latent graph vectors are computed as the average of all the  $K$  layer's embeddings to prevent the over-smoothing problem.

For the neighborhood view, the rating prediction function is  $\hat{R}_{ij}^{g} = (e_{i}^{u})^{T}e_{j}^{v}$  and  $\hat{R}_{il}^{g} = (e_l^c)^T e_j^v$ . Hence, the GCN loss can be given similar to Equation 10:

$$
\mathcal {L} _ {\text {G r a p h}} = - \Sigma_ {i} \Sigma_ {j} \frac {R _ {i j}}{\max \left(R _ {i}\right)} \log \hat {R} _ {l j} ^ {g} - \Sigma_ {l} \Sigma_ {j} \frac {R _ {l j}}{\max \left(R _ {l}\right)} \log \hat {R} _ {l j} ^ {g} + \lambda \left(\Sigma_ {i} \| e _ {i} ^ {u} \| ^ {2} + \Sigma_ {l} \| e _ {l} ^ {c} \| ^ {2} + \Sigma_ {j} \| e _ {j} ^ {v} \| ^ {2}\right). \tag {12}
$$

The GCN layer can effectively capture the neighborhood information, thereby enriching the understanding of user preferences and critic opinions. However, the user-item-critic graph mainly comprises positive implicit interactions. A direct addition of graph vectors to  $p_i$  or  $q_j$  hinders rather than enhances performance. To navigate this challenge, a cross-view contrastive learning mechanism is proposed, ensuring alignment of vectors across varying views without the direct addition or concatenation of representations.

Contrastive learning facilitates comparisons between diverse augmented samples and have been proven effective in recommender systems (Wu et al., 2021). Using graph sampling techniques, augmented latent graph vectors  $\widetilde{e}_i^u$  and  $\widetilde{e}_j^v$  are generated. Unlike traditional contrastive learning methods like InfoNCE (Gutmann & Hyvarinen, 2010), which consider two samples from the same view as positive pairs, this approach treats the unified vector  $p_i$  and the augmented graph vector  $\widetilde{e}_i^u$  as a positive pair.  $p_i$  and other augmented graph vectors  $\widetilde{e}_{i'}^u$ ,  $i' \neq i$  are treated as negative pairs. In each batch, we randomly sample some negative pairs. The same approach is applied to items. Formally, we can maximize positive pair agreement and minimize negative pair agreement as follows:

$$
\begin{array}{l} \mathcal {L} _ {C L} = - \Sigma_ {i} \log \frac {\exp (\cos (p _ {i} , \widetilde {e} _ {i} ^ {u}) / \tau)}{\exp (\cos (p _ {i} , \widetilde {e} _ {i} ^ {u}) / \tau) + \Sigma_ {i ^ {\prime}} \exp (\cos (p _ {i} , \widetilde {e} _ {i ^ {\prime}} ^ {u}) / \tau)} \\ - \Sigma_ {j} \log \frac {\exp \left(\cos \left(q _ {j} , \widetilde {e} _ {j} ^ {v}\right) / \tau\right)}{\exp \left(\cos \left(q _ {j} , \widetilde {e} _ {j} ^ {v}\right) / \tau\right) + \Sigma_ {j ^ {\prime}} \exp \left(\cos \left(q _ {j} , \widetilde {e} _ {j ^ {\prime}} ^ {v}\right) / \tau\right)}, \tag {13} \\ \end{array}
$$

Here,  $\cos (\cdot)$  is the cosine similarity function, and  $\tau_{c}$  is the temperature parameter.

Finally, the unified loss within the MCIR framework is defined as:

$$
\mathcal {L} = \mathcal {L} _ {V A E} + \eta_ {1} \mathcal {L} _ {M S E} + \eta_ {2} \mathcal {L} _ {M u l t i} + \eta_ {3} \mathcal {L} _ {G r a p h} + \eta_ {4} \mathcal {L} _ {C L}. \tag {14}
$$

Noticeably, while our primary focus is on user and critic review scenarios, MCIR can actually be effortlessly adapted to similar item-sharing contexts with minimal modifications for varying features.

# 4 EXPERIMENTS

In this section, we begin by detailing the datasets, evaluation protocols, baseline methods, and experimental settings. Then, we report the recommendation performance results of our proposed MCIR models compared to the state-of-the-art baselines. Further, we conduct ablation studies to validate the efficacy of each MCIR component. Discussion on the impact of the cross-view contrastive learning mechanism and the influence of various hyper-parameters on the performance is included. Finally, we present some case studies in the Appendix to show the explanatory capabilities of MCIR.

# 4.1 EXPERIMENTAL SETTINGS

Datasets. The datasets used in the experiments were collected from Metacritic  $^{1}$ . We collect the user and critic reviews as well as ratings for games, movies, and musics up to December 2022 to form three datasets, i.e., MC-Game, MC-Movie, and MC-Music. In the MC-Game, MC-Movie, and MC-Music datasets, user ratings are expressed as 10-stars, whereas critic scores utilize a percentage system. To facilitate comparison, we normalize both user and critic scores to fall within the  $[0,1]$  range. For validation, following (Wu et al., 2018), we adopted the data preprocessing to differentiate the positive and negative feedback depending on whether the ratings are not less than 0.7. MC-Game contains 18,622 users, 522 critics, and 16,713 items with 505,964 user reviews and 242,764 critic reviews. MC-Movie contains 15,402 users, 3,048 critics, and 8,259 items with 261,292 user reviews and 144,541 critic reviews. MC-Music contains 11,483 users, 131 critics, and 5,133 items with 190,148 user reviews and 61,740 critic reviews. We can find that all the three datasets are extremely sparse in the user domain with the sparsity larger than  $99.96\%$ .

Evaluation metrics. To construct the training set, we randomly sampled  $60\%$  observed items for each user. Then, we sampled  $10\%$  observed items of each user for validation, and the rest data were used for the test. Hence, we randomly split each dataset five times and reported all the results by average values. We employed four widely used evaluation metrics for evaluating the performance, i.e.,  $\mathrm{P}@\mathcal{K}$ ,  $\mathrm{R}@\mathcal{K}$ ,  $\mathrm{MAP}@\mathcal{K}$ , and  $\mathrm{NDCG}@\mathcal{K}$  (Wu et al., 2018). For each user,  $\mathrm{P}$  (Precision)  $@\mathcal{K}$  measures the ratio of correct prediction results among top- $K$  items to  $K$  and  $\mathrm{R}$  (Recall)  $@\mathcal{K}$  measures the ratio of correct prediction results among top- $K$  items to all positive items. Furthermore, MAP (Mean

Table 1: The overall recommendation performances of different approaches.  

<table><tr><td>Datasets</td><td>Methods</td><td>R@5</td><td>R@10</td><td>P@5</td><td>P@10</td><td>MAP@5</td><td>MAP@10</td><td>NDCG@5</td><td>NDCG@10</td></tr><tr><td rowspan="9">MC-Game</td><td>CoNet</td><td>0.0443</td><td>0.0733</td><td>0.0363</td><td>0.0308</td><td>0.0203</td><td>0.0126</td><td>0.0478</td><td>0.0578</td></tr><tr><td>DeepAPF</td><td>0.0591</td><td>0.0934</td><td>0.0471</td><td>0.0381</td><td>0.0272</td><td>0.0166</td><td>0.0635</td><td>0.0747</td></tr><tr><td>NATR</td><td>0.0804</td><td>0.1241</td><td>0.0635</td><td>0.0500</td><td>0.0385</td><td>0.0235</td><td>0.0873</td><td>0.1007</td></tr><tr><td>CMF</td><td>0.0867</td><td>0.1364</td><td>0.0696</td><td>0.0560</td><td>0.0424</td><td>0.0263</td><td>0.0950</td><td>0.1107</td></tr><tr><td>DCDCSR</td><td>0.0877</td><td>0.1376</td><td>0.0708</td><td>0.0567</td><td>0.0430</td><td>0.0269</td><td>0.0957</td><td>0.1118</td></tr><tr><td>SSCDR</td><td>0.0913</td><td>0.1423</td><td>0.0742</td><td>0.0589</td><td>0.0452</td><td>0.0280</td><td>0.0994</td><td>0.1151</td></tr><tr><td>EMCDR</td><td>0.0899</td><td>0.1408</td><td>0.0723</td><td>0.0572</td><td>0.0441</td><td>0.0272</td><td>0.0976</td><td>0.1136</td></tr><tr><td>BiTGCF</td><td>0.0941</td><td>0.1474</td><td>0.0755</td><td>0.0601</td><td>0.0470</td><td>0.0292</td><td>0.1034</td><td>0.1203</td></tr><tr><td>MCIR</td><td>0.1101</td><td>0.1650</td><td>0.0903</td><td>0.0695</td><td>0.0581</td><td>0.0358</td><td>0.1229</td><td>0.1388</td></tr><tr><td rowspan="9">MC-Movie</td><td>CoNet</td><td>0.0377</td><td>0.0652</td><td>0.0285</td><td>0.0250</td><td>0.0159</td><td>0.0103</td><td>0.0400</td><td>0.0488</td></tr><tr><td>DeepAPF</td><td>0.0548</td><td>0.0925</td><td>0.0357</td><td>0.0308</td><td>0.0205</td><td>0.0132</td><td>0.0547</td><td>0.0665</td></tr><tr><td>NATR</td><td>0.0706</td><td>0.1164</td><td>0.0434</td><td>0.0370</td><td>0.0256</td><td>0.0165</td><td>0.0689</td><td>0.0835</td></tr><tr><td>CMF</td><td>0.0795</td><td>0.1285</td><td>0.0543</td><td>0.0456</td><td>0.0334</td><td>0.0218</td><td>0.0828</td><td>0.0977</td></tr><tr><td>DCDCSR</td><td>0.0821</td><td>0.1331</td><td>0.0560</td><td>0.0467</td><td>0.0339</td><td>0.0222</td><td>0.0840</td><td>0.0993</td></tr><tr><td>SSCDR</td><td>0.0884</td><td>0.1406</td><td>0.0614</td><td>0.0508</td><td>0.0378</td><td>0.0249</td><td>0.0898</td><td>0.1056</td></tr><tr><td>EMCDR</td><td>0.0781</td><td>0.1258</td><td>0.0541</td><td>0.0450</td><td>0.0332</td><td>0.0216</td><td>0.0807</td><td>0.0954</td></tr><tr><td>BiTGCF</td><td>0.0867</td><td>0.1394</td><td>0.0592</td><td>0.0488</td><td>0.0373</td><td>0.0240</td><td>0.0902</td><td>0.1059</td></tr><tr><td>MCIR</td><td>0.1069</td><td>0.1586</td><td>0.0765</td><td>0.0594</td><td>0.0522</td><td>0.0331</td><td>0.1159</td><td>0.1299</td></tr><tr><td rowspan="9">MC-Music</td><td>CoNet</td><td>0.0420</td><td>0.0698</td><td>0.0280</td><td>0.0231</td><td>0.0155</td><td>0.0097</td><td>0.0396</td><td>0.0495</td></tr><tr><td>DeepAPF</td><td>0.0526</td><td>0.0847</td><td>0.0337</td><td>0.0276</td><td>0.0186</td><td>0.0115</td><td>0.0492</td><td>0.0607</td></tr><tr><td>NATR</td><td>0.0546</td><td>0.0896</td><td>0.0305</td><td>0.0255</td><td>0.0155</td><td>0.0096</td><td>0.0464</td><td>0.0590</td></tr><tr><td>CMF</td><td>0.0784</td><td>0.1267</td><td>0.0505</td><td>0.0407</td><td>0.0315</td><td>0.0195</td><td>0.0773</td><td>0.0937</td></tr><tr><td>DCDCSR</td><td>0.0878</td><td>0.1361</td><td>0.0545</td><td>0.0434</td><td>0.0353</td><td>0.0220</td><td>0.0844</td><td>0.1013</td></tr><tr><td>SSCDR</td><td>0.0787</td><td>0.1238</td><td>0.0515</td><td>0.0407</td><td>0.0310</td><td>0.0193</td><td>0.0744</td><td>0.0897</td></tr><tr><td>EMCDR</td><td>0.0910</td><td>0.1353</td><td>0.0569</td><td>0.0439</td><td>0.0362</td><td>0.0223</td><td>0.0866</td><td>0.1019</td></tr><tr><td>BiTGCF</td><td>0.0894</td><td>0.1405</td><td>0.0574</td><td>0.0455</td><td>0.0365</td><td>0.0228</td><td>0.0867</td><td>0.1043</td></tr><tr><td>MCIR</td><td>0.1176</td><td>0.1621</td><td>0.0735</td><td>0.0533</td><td>0.0547</td><td>0.0323</td><td>0.1199</td><td>0.1344</td></tr></table>

Average Precision) @K and NDCG (Normalized Discounted Cumulative Gain) @K consider the ranking of correct prediction results among top-K items.

Baselines. We compare our proposed approach with various stat-of-the-art CDR methods. CMF (Singh & Gordon, 2008) is a classic collaborative CDR method. CoNet (Hu et al., 2018) and DeepAPF (Yan et al., 2019) utilize neural networks for CDR. DCDCSR (Zhu et al., 2020) considers the rating sparsity degrees of individual users in different domains. SSCDR (Kang et al., 2019) utilizes semi-supervised learning to map or share features. EMCDR (Man et al., 2017) combines Matrix Factorization and network-based bridging. BiTGCF (Liu et al., 2020) is a bi-directional transfer learning method that utilizes a Graph Collaborative Filtering network. NATR (Gao et al., 2021) is dedicatedly designed for the item-sharing scenario with neural transfer learning.

For the above-mentioned baselines, we utilized the open-source implementation provided by Recbole (Zhao et al., 2021) using PyTorch. Since most baselines were designed for user-sharing scenario, we extended them in a symmetrical manner to support item-sharing task. We used grid search for all the above baselines to carefully tune the corresponding parameters, such as the regularization coefficient and learning rate. In order to provide a fair comparison, we set the embedding size of all models to 150. Please see more implementation details in the Appendix.

# 4.2 OVERALL RECOMMENDATION PERFORMANCE

We present the overall recommendation performance results for the three datasets in Table 1 under two types of settings, i.e.,  $K = 5$  and  $K = 10$ . We can discover from Table 1 that MCIR can outperform all the baseline methods on every dataset owing to the multi-view learning framework and derived comprehensive representations. Specifically, MCIR outperforms the best baseline, by a relative boost of  $19.60\%$ ,  $17.00\%$ ,  $23.61\%$ ,  $18.86\%$  for the metric  $\mathsf{P}@\mathsf{5}$ ,  $\mathsf{R}@\mathsf{5}$ ,  $\mathsf{MAP}@\mathsf{5}$ , and NDCG@5 in MC-Game,  $20.93\%$ ,  $24.59\%$ ,  $39.09\%$ ,  $29.06\%$  in MC-Movie, and  $29.23\%$ ,  $28.04\%$ ,  $49.86\%$ ,  $38.29\%$  in MC-Music, respectively. Hence, the results clearly demonstrate the effectiveness of our proposed approaches. Among the baseline methods, BiTGCF got the best performances in most conditions, maybe because its knowledge transfer module can effectively extend the flow of features from in-domain to inter-domain, and thus considers the integration of domain's common features and domain-specific features. However, with only the graph view learning, BiTGCF cannot outperform our multi-view approach. Surprisingly, we find that CoNet, DeepAPF, and NATR performs not good in our experiments, even worse than CMF. A potential reason is that these three approaches were designed for the scenario that different domains share similar behavior patterns. While in our scenario, user and critics are quite different type in behaviors.

![](images/8f514155e9fff5f873175ad50de8ade09b5a02dc3485eeaf23e566b5a43879ea.jpg)  
(a)

![](images/778127420c21f90622ab423f2c6b7c6152d090ecc35a543ced90b238c54f4098.jpg)  
Figure 3: Left: The ablation study of MCIR on the MC-Game Dataset. Right: The performance of R@5 with different values of hyper-parameter  $\eta_{n}$  and dimension  $d$  on the three datasets.

![](images/a46c60500362e458860a485a90a5ea74a4392dd061da05cbef70979e707de208.jpg)  
(b)

![](images/468c8e3eaf1b43114933d7eff651195e58666f0892c39d993bf2bc248500d15b.jpg)

# 4.3 INVESTIGATIONS ON ABLATION STUDIES

In this section, we compare 5 variants of MCIR: MCIR-C1 without the review text information for user and critic embedding network; MCIR-C2 without the review text information for item embedding networks; MCIR-C3 without the entire critic domain; MCIR-C4 without the cross-view contrastive learning; MCIR-C5 with AE instead of VAE. Figure 3(a) presents the performance on MC-Game Dataset (More results on MC-Movie and MC-Music are in the Appendix). By comparing MCIR with MCIR-C1 and MCIR-C2, we can find that comment information is important for enhancing the performances. By comparing MCIR with MCIR-C3, we can find that the auxiliary information from critic domain is essential for recommendations in user domain. By comparing MCIR with MCIR-C4, we can validate the efficacy of our proposed cross-view contrastive learning. By comparing MCIR with MCIR-C5, we can observe that VAE can generate more accurate results than AE.

# 4.4 INVESTIGATIONS ON THE CROSS-VIEW CONTRASTIVE LEARNING MECHANISM

In this subsection, we assess the impact of the cross-view contrastive learning mechanism by adjusting the hyper-parameter  $\eta_{n}$  within [1, 3, 7, 15, 20]. When  $\eta_{n} = 0$ , according to Equation 14, the contrastive learning mechanism is eliminated from the joint loss function. Figure 3 illustrates the performance of R@5 with varying values of the hyper-parameter  $\eta_{n}$  across the three datasets. The performance results of other metrics can be found in the Appendix. The results depicted in Figure 3(b) reveal that the performance of MCIR initially improves as  $\eta_{n}$  increases, validating that contrastive learning enables MCIR to learn the graph neighborhood information and thereby enhance its performance. However, when  $\eta_{n}$  becomes excessively large, performance rapidly deteriorates. This is because the user-item-critic graph lacks rating information, so an overly large  $\eta_{n}$  introduces noise rather than beneficial information.

# 4.5 INVESTIGATIONS ON THE DIMENSIONS OF THE LATENT SPACE

The number of dimensions  $d$  is quite vital for the performance. If  $d$  is too small, the latent space would have very weak representation ability to fit the real-world data. On the opposite, if  $d$  is too large, the model complexity would also become too large and may face the over-fitting problem. The performance of R@5 with different values of dimension  $d$  on the three datasets are presented in Figure 3(b) (see more in Appendix). We can observe that the performance result of MCIR is not good when  $r = 50$ . With a larger value of  $d$ , the performance tends to be much better. When  $d = 200$ , MCIR achieves the best results. With larger  $d$ , the performance of MCIR will begin to decrease.

# 5 CONCLUSION

In this work, we introduced the Multi-view Cross-domain Item-sharing Recommendation (MCIR) framework, an innovative approach to overcome the unique challenges posed in cross-domain recommendations with only item data being shared. MCIR adeptly amalgamates user preferences and critic opinions, employing distinct embedding networks explicitly designed for each perspective. Further, An attentive integration mechanism was designed to extract comprehensive item representations based on these multi-view representations. Moreover, we enhanced the framework by introducing a cross-view contrastive learning method, which harmonizes embeddings across different views by leveraging detailed neighborhood information from the user-item-critic graph. Finally, we conducted extensive experiments on three real-world datasets to validate the effectiveness of MCIR.

# REFERENCES

Metacritic. URL https://www.metacritic.com/.  
Rotten tomatoes. URL https://www.rottentomatoes.com/.  
Deepak Agarwal, Bee-Chung Chen, and Bo Long. Localized factor models for multi-context recommendation. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 609-617, 2011.  
Suman Basuroy, Subimal Chatterjee, and S Abraham Ravid. How critical are critical reviews? the box office effects of film critics, star power, and budgets. Journal of marketing, 67(4):103-117, 2003.  
Chaochao Chen, Huiwen Wu, Jiajie Su, Lingjuan Lyu, Xiaolin Zheng, and Li Wang. Differential private knowledge transfer for privacy-preserving cross-domain recommendation. In Proceedings of the ACM Web Conference 2022, pp. 1455-1465, 2022.  
Richard Dillio. *Different Scores: Video Gamers' Use of Amateur and Professional Reviews*. Rochester Institute of Technology, 2013.  
Chen Gao, Xiangning Chen, Fuli Feng, Kai Zhao, Xiangnan He, Yong Li, and Depeng Jin. Cross-domain recommendation without sharing user-relevant data. In The world wide web conference, pp. 491-502, 2019.  
Chen Gao, Yong Li, Fuli Feng, Xiangning Chen, Kai Zhao, Xiangnan He, and Depeng Jin. Cross-domain recommendation with bridge-item embeddings. ACM Transactions on Knowledge Discovery from Data (TKDD), 16(1):1-23, 2021.  
Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 297-304. JMLR Workshop and Conference Proceedings, 2010.  
Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. Lightgen: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pp. 639-648, 2020.  
Guangneng Hu, Yu Zhang, and Qiang Yang. Conet: Collaborative cross networks for cross-domain recommendation. In Proceedings of the 27th ACM international conference on information and knowledge management, pp. 667-676, 2018.  
SeongKu Kang, Junyoung Hwang, Dongha Lee, and Hwanjo Yu. Semi-supervised learning for cross-domain recommendation to cold-start users. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pp. 1563-1572, 2019.  
Farhan Khawar, Leonard Poon, and Nevin L Zhang. Learning the structure of auto-encoding recommenders. In Proceedings of The Web Conference 2020, pp. 519-529, 2020.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix factorization techniques for recommender systems. Computer, 42(8):30-37, 2009.  
Chenglin Li, Yuanzhen Xie, Chenyun Yu, Bo Hu, Zang Li, Guoqiang Shu, Xiaohu Qie, and Di Niu. One for all, all for one: Learning and transferring user embeddings for cross-domain recommendation. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, pp. 366-374, 2023.  
Xiaopeng Li and James She. Collaborative variational autoencoder for recommender systems. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 305-314, 2017.

Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. Variational autoencoders for collaborative filtering. In WWW, pp. 689-698. International World Wide Web Conferences Steering Committee, 2018.  
Meng Liu, Jianjun Li, Guohui Li, and Peng Pan. Cross domain recommendation via bi-directional transfer graph collaborative filtering networks. In Proceedings of the 29th ACM international conference on information & knowledge management, pp. 885-894, 2020.  
Jianxin Ma, Chang Zhou, Peng Cui, Hongxia Yang, and Wenwu Zhu. Learning disentangled representations for recommendation. Advances in neural information processing systems, 2019.  
Tong Man, Huawei Shen, Xiaolong Jin, and Xueqi Cheng. Cross-domain recommendation: An embedding and mapping approach. In IJCAI, volume 17, pp. 2464-2470, 2017.  
Kelong Mao, Jieming Zhu, Xi Xiao, Biao Lu, Zhaowei Wang, and Xiuqiang He. Ultragcn: ultra simplification of graph convolutional networks for recommendation. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 1253-1262, 2021.  
Nima Mirbakhsh and Charles X Ling. Improving top-n recommendation for cold-start users via cross-domain information. ACM Transactions on Knowledge Discovery from Data (TKDD), 9(4): 1-19, 2015.  
Andriy Mnih and Ruslan R Salakhutdinov. Probabilistic matrix factorization. In Advances in neural information processing systems, pp. 1257-1264, 2008.  
Anish A Parikh, Carl Behnke, Barbera Almanza, Doug Nelson, and Mihaela Vorvoreanu. Comparative content analysis of professional, semi-professional, and user-generated restaurant reviews. Journal of foodservice business research, 20(5):497-511, 2017.  
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL http://arxiv.org/abs/1908.10084.  
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. arXiv preprint arXiv:1401.4082, 2014.  
Tiago Santos, Florian Lemmerich, Markus Strohmaier, and Denis Helic. What's in a review: Discrepancies between expert and amateur reviews of video games on metacritic. Proceedings of the ACM on human-computer interaction, 3(CSCW):1-22, 2019.  
Ajit P Singh and Geoffrey J Gordon. Relational learning via collective matrix factorization. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 650-658, 2008.  
Wen-Chin Tsao. Which type of online review is more persuasive? the influence of consumer reviews and critic ratings on moviegoers. Electronic Commerce Research, 14:559-583, 2014.  
Hao Wang, Naiyan Wang, and Dit-Yan Yeung. Collaborative deep learning for recommender systems. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1235-1244. ACM, 2015.  
Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pp. 726-735, 2021.  
Liwei Wu, Cho-Jui Hsieh, and James Sharpnack. *Sql-rank: A listwise approach to collaborative ranking*. In Proceedings of the 35th International Conference on Machine Learning, ser, volume 80, pp. 5315-5324, 2018.  
Ruobing Xie, Qi Liu, Liangdong Wang, Shukai Liu, Bo Zhang, and Leyu Lin. Contrastive cross-domain recommendation in matching. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4226-4236, 2022.

Hong-Jian Xue, Xinyu Dai, Jianbing Zhang, Shujian Huang, and Jiajun Chen. Deep matrix factorization models for recommender systems. In IJCAI, pp. 3203-3209, 2017.  
Huan Yan, Xiangning Chen, Chen Gao, Yong Li, and Depeng Jin. Deepapf: Deep attentive probabilistic factorization for multi-site video recommendation. TC, 2(130):17-883, 2019.  
Fuzheng Zhang, Nicholas Jing Yuan, Defu Lian, Xing Xie, and Wei-Ying Ma. Collaborative knowledge base embedding for recommender systems. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 353-362, 2016.  
Cheng Zhao, Chenliang Li, Rong Xiao, Hongbo Deng, and Aixin Sun. Catn: Cross-domain recommendation for cold-start users via aspect transfer network. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 229-238, 2020.  
Chuang Zhao, Hongke Zhao, Ming He, Jian Zhang, and Jianping Fan. Cross-domain recommendation via user interest alignment. arXiv preprint arXiv:2301.11467, 2023.  
Wayne Xin Zhao, Shanlei Mu, Yupeng Hou, Zihan Lin, Kaiyuan Li, Yushuo Chen, Yujie Lu, Hui Wang, Changxin Tian, Xingyu Pan, Yingqian Min, Zhichao Feng, Xinyan Fan, Xu Chen, Pengfei Wang, Wendi Ji, Yaliang Li, Xiaoling Wang, and Ji-Rong Wen. Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In CIKM, 2021.  
Feng Zhu, Yan Wang, Chaochao Chen, Guanfeng Liu, Mehmet Orgun, and Jia Wu. A deep framework for cross-domain and cross-system recommendations. arXiv preprint arXiv:2009.06215, 2020.

Table 2: The statistical information of the datasets.  

<table><tr><td>Dataset</td><td>MC-Game</td><td>MC-Movie</td><td>MC-Music</td></tr><tr><td>The number of items</td><td>16,713</td><td>8,259</td><td>5,133</td></tr><tr><td>The number of users</td><td>18,622</td><td>15,402</td><td>11,483</td></tr><tr><td>The number of critics</td><td>522</td><td>3048</td><td>131</td></tr><tr><td>The number of user-item interactions</td><td>505,964</td><td>261,292</td><td>190,148</td></tr><tr><td>The number of critic-item interactions</td><td>242,764</td><td>144,541</td><td>61,740</td></tr></table>
