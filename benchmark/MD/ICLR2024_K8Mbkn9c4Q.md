# TABLEYE: SEEING SMALL TABLES THROUGH THE LENS OF IMAGES

Anonymous authors

Paper under double-blind review

# ABSTRACT

The exploration of few-shot tabular learning becomes imperative. Tabular data is a versatile representation that captures diverse information, yet it is not exempt from limitations, property of data and model size. Labeling extensive tabular data can be challenging, and it may not be feasible to capture every important feature. Few-shot tabular learning, however, remains relatively unexplored, primarily due to scarcity of shared information among independent datasets and the inherent ambiguity in defining boundaries within tabular data. To the best of our knowledge, no meaningful and unrestricted few-shot tabular learning techniques have been developed without imposing constraints on the dataset. In this paper, we propose an innovative framework called TabEye, which aims to overcome the limit of forming prior knowledge for tabular data by adopting domain transformation. It facilitates domain transformation by generating tabular images, which effectively conserve the intrinsic semantics of the original tabular data. This approach harnesses rigorously tested few-shot learning algorithms and embedding functions to acquire and apply prior knowledge. Leveraging shared data domains allows us to utilize this prior knowledge, originally learned from the image domain. Specifically, TabEye demonstrated a superior performance by outstripping the TabLLM in a 4-shot task with a maximum 0.11 AUC and a STUNT in a 1-shot setting, where it led on average by  $3.17\%$  accuracy.

# 1 INTRODUCTION

It is a common misperception that a large volume of data is indispensable for the deep learning techniques (Zhang et al., 2018). Indeed, dataset size plays a critical role in enhancing model performance(Sordo & Zeng, 2005; Prusa et al., 2015). Regardless of a neural network model quality, it seems futile without access to ample data. This data insufficient problem frequently arises due to some reasons such as high costs, privacy concerns, or security issues(Clements et al., 2020). Despite these challenges, there are many attempts to improve accuracy through deep learning with limited labeled data. This line of research is known as few-shot learning (Wang et al., 2020).

Few-shot learning in the tabular domain, however, has received relatively little attention(Guo et al., 2017; Zhang et al., 2019). The lack of research in this area can be traced back to several factors. Firstly, compared to the image and language domain, tabular datasets lack shared information (Mathov et al., 2020). Unlike image or language data, where prior knowledge can be learned from related examples within the different datasets(Parnami & Lee, 2022), it is challenging to establish similar relationships in tabular data. For example, while learning to distinguish between dogs and lions may assist in distinguishing between cats and tigers, learning to predict solar power generation will not necessarily aid in understanding trends in the financial market. Secondly, defining clear boundaries for tabular data is a complex task(Mathov et al., 2020). Image and language data possess physical or visual representations, allowing their boundaries to be defined by parameters such as pixels, color channels (R, G, B), image size, vocabulary words), and grammar. In contrast, tabular data lacks a distinct shared representation(Ucar et al., 2021). Various features within tabular data have independent distributions and ranges, and missing values may be present.

We assume when the features in tabular data are condensed into a limited format like pixels or words, prior knowledge learned in the different domain can help solve any task in tabular domain. For an intuitive example, if a child learns about an apple from a picture, they can connect it to a letter

('Apple') and a number ('An' or 'One') and make associations. If additional information, such as rules or relationships between numbers, is provided, the child can infer that two apples are present by observing two apple photos side by side. However, for a child who has only learned the numbers '1' and '2', understanding that  $1 + 1$  equals 2 may not come easily. Similarly, if we incorporate information about tabular data into neural networks trained solely on images, even a small labeled data can yield superior performance compared to traditional machine learning approaches that rely on larger amounts of labeled data.

To empirically validate our proposed hypothesis, we present the TablEye framework, which is fundamentally structured into two distinct stages. The first is the transformation stage, where each vector from a tabular dataset is transmuted into an image format. In this stage, we leverage spatial relations across three channels to ensure the tabular data not only morphs into a format analogous to conventional images but also retains its intrinsic meaning. The second stage is dedicated to the incorporation of prior knowledge through a few-shot learning approach. Recognizing the proven efficacy of few-shot learning algorithms in the realm of image processing, we capitalize on them after transforming the tabular data into an image-like structure. This transformation facilitates the construction of prior knowledge using a vast array of image data for few-shot tabular learning. Consequently, utilizing this accumulated prior knowledge enables us to predict outcomes from the image-represented tabular data effectively.

Our proposed approach achieves comparable or even superior performance to previous research through experiments on various datasets. Moreover, it offers the flexibility to perform few-shot learning tasks without being constrained by composition of dataset. TabEye overcomes the need for large unlabeled datasets by leveraging the image domain, and it requires less computing cost due to its smaller model size than one of the LLM. To the best of our knowledge, this paper represents the first attempt to apply prior knowledge from the image domain to few-shot learning in the tabular domain. The proposed few-shot tabular learning technique has the potential to provide artificial intelligence models that can achieve accurate results with only a small amount of data in scenarios where data acquisition is challenging, such as disease diagnosis in the medical industry.

The main contributions of this work are:

- This work represents the first attempt to leverage large image data as prior knowledge to address the problem of few-shot tabular learning, formation of prior knowledge.  
- We propose a novel framework, TabEye, which employs domain transformation to apply prior knowledge from image data to few-shot tabular learning.  
- We have successfully overcome the limitations associated with existing few-shot tabular learning models, including constraints related to feature size of dataset, the requirement for large quantities of unlabeled data, and the demand for extensive computational resources.

# 2 RELATED WORK

Tabular learning refers to the process of learning the mapping between input and output data using tabular data(Borisov et al., 2022). Tabular data is often also called structured data(Ryan, 2020) and is a subset of heterogeneous data presented in a table format with rows and columns. Each feature in this data is composed of either categorical or numerical features. Currently, methods based on decision trees and those based on Multi-Layer Perceptrons (MLP) are showing almost equal performance. Tabular learning still requires a large amount of labeled data. In the image domain, few-shot learning can easily acquire prior knowledge using many related images. For example, ProtoNet (Prototypical Network)(Snell et al., 2017) learns using similarities between images, and MAML (Model-Agnostic Meta-Learning)(Finn et al., 2017) quickly adjusts the model across various tasks, enabling rapid learning with limited data. However, in the tabular domain, there are no equivalent sets of related tabular data. Therefore, few-shot tabular learning faces significant challenges in forming prior knowledge. Therefore, the current state-of-the-art (SOTA) methods for few-shot tabular learning utilize semi-few-shot learning approaches using unlabeled data samples or transfer tabular data to the text domain and employ Large Language Models.

# 2.1 SEMI-FEW-SHOT TABULAR LEARNING: STUNT

STUNT(Nam et al., 2023) represents a semi-few-shot learning technique aimed at enhancing the performance of tabular learning in scenarios with sparse labeled datasets, utilizing a substantial quantity of reasonably coherent unlabeled data. This method marks an attempt to resolve the few-shot learning problem from a data perspective, by learning prior knowledge from an unlabeled set to which arbitrary labels have been assigned. To generate these arbitrary labels, it adopted the  $k$ -means clustering technique. This approach utilizes a Prototypical Network(Snell et al., 2017) to learn prior knowledge from these self-generated tasks, and it has demonstrated impressive performance. This method, as a semi-few-shot learning technique, operates exclusively within the tabular domain, by the way requires a substantial quantity of reasonably consistent unlabeled data. The size of the unlabeled set also can significantly influence the performance of STUNT(Nam et al., 2023).

# 2.2 FEW-SHOT TABULAR LEARNING: TABLELM

In the domain of few-shot tabular learning, TabLLM(Hegselmann et al., 2023) offers a unique perspective by harnessing the Large Language Model (LLM). The process employed by this method involves the conversion of original tabular data into a text format following a specific template. This transformation reforms tabular data into a more adaptable textual form, making it suitable as the prompt for LLM. Following the serialization, this data is utilized to fine-tune the LLM(Liu et al., 2022). The T0 encoder-decoder model, equipped with an extensive set of 11 billion parameters, plays a crucial role in this process (Sanh et al., 2021). This large parameter set, indicative of the extensive model training, also necessitates substantial computational resources, presenting a potential challenge. Moreover, TabLLM inevitably requires significant feature names, and t is constrained by limitations on token length.

# 3 OUR APPROACH: TABLEYE

# 3.1 OVERVIEW

This paper introduces a novel framework called TablEye, aimed at enhancing the effectiveness of few-shot tabular learning. Figure 1 shows the overview of TablEye. TablEye applies efficient few-shot learning algorithms in the image domain by performing domain transformation from the tabular domain to the image domain. The framework comprises two main stages: the transformation stage from the tabular domain to the image domain and the prior knowledge learning stage in the image domain. In the tabular domain, TablEye preprocesses tabular data and undergoes a transformation process into a three-channel image, referred to as a tabular image. Subsequently, few-shot tabular classification is performed using prior knowledge learned from mini-[ImageNet] in the image domain. To generate tabular images from tabular data, a method based on feature similarity is employed, incorporating spatial relations into the tabular images. In the stage of learning prior knowledge, ProtoNet (Prototypical Network) and MAML (Model-Agnostic Meta Learning) are employed, as they demonstrate high performance and can be applied to various few-shot learning structures. The backbone for embedding and the classifier for the few-shot task are connected sequentially. During the process of learning embeddings in a dimension suitable for classification through the backbone, Cross-entropy loss is utilized(Zhang & Sabuncu, 2018).

# 3.2 DOMAIN TRANSFORMATION

The domain transformation stage aims to convert tabular data into the desired form of images (3, 84, 84), while preserving the characteristics and semantics of the tabular data. We hypothesize that the difference between images and tabular data lies in the association with neighboring values and spatial relations(Zhu et al., 2021). The pixels in an image exhibit strong correlations with adjacent pixels, and this is why the kernels in a Convolutional Neural Network (CNN) play an important role. Therefore, we incorporate spatial relations into tabular data and undergo a process of shaping it into the desired form. Given  $n$  features, we measure the Euclidean distance between these features and rank them to create an  $(n, n)$  feature matrix, denoted as  $\mathbf{R}$ . Assume we have data matrix  $\mathbf{D}$  with  $C$  data samples and  $n$  features  $(D_{ij}$  means the  $j$  th feature of the  $i$  th data sample.) and an array of feature name  $F$ .  $F_{i}$  indicates the vector of the  $i$  th feature name obtained by GloVe100(Pennington

![](images/d5843ae853cd87e8ab154f5019157e00c5ae473765d94d5a4506223213bff63b.jpg)  
Figure 1: Overview of TablEye. The natural images of image domain are part of mini-ImageNet.

et al., 2014). If no meaningful name exists for a specific feature, we used 'i feature' as the feature name.

$$
R _ {i j} = \frac {1}{C} \sum_ {c = 0} ^ {C} \sqrt {\left(D _ {c i} - D _ {c j}\right) ^ {2}} + \alpha \times \sqrt {\left(F _ {i} - F _ {j}\right) ^ {2}} \quad \text {w h e r e} \quad 0 <   i \leq n \quad \text {a n d} \quad 0 <   j \leq n
$$

We also measure the distance and rank between  $n$  elements to generate an  $(n, n)$  pixel matrix, denoted as  $\mathbf{Q}$ . The pixel matrix  $\mathbf{Q}$  is the similarity matrix between the coordinate pixels of  $n_r \times n_c$  image.  $(n = n_r \times n_c$  and  $n_r$  and  $n_c$  are the height and width of the transformed image.) Assume a coordinate list of  $n$  features.

$$
\text {C o o r d i n a t e s} = \left[ (0, 0) \dots (0, n _ {c} - 1) \dots (1, 0) \dots (1, n _ {c} - 1) \dots (n _ {r} - 1, 0) \dots (n _ {r} - 1, n _ {c} - 1) \right]
$$

The  $i$  th element of the coordinate indicates the coordinate of  $i$  th the feature for  $N(= N_r\times N_c)$  image.

$$
Q _ {i j} = \sqrt {\left(\operatorname {C o o r d i n a t e} [ i ] [ 0 ] - \operatorname {C o o r d i n a t e} [ j ] [ 0 ]\right) ^ {2} + \left(\operatorname {C o o r d i n a t e} [ i ] [ 1 ] - \operatorname {C o o r d i n a t e} [ j ] [ 1 ]\right) ^ {2}}
$$

Then, we compute the Euclidean distance between  $\mathbf{R}$  and  $\mathbf{Q}$  and rearrange the positions of the features to minimize the distance, aiming to align the feature distance and pixel distance, thus assigning spatial relations. This results in obtaining a 2-dimensional image  $M$  of size  $n_r\times n_c$ , where features with closer distances correspond to pixels that are closer to each other.

In the equations below,  $r_{ij}$  and  $q_{ij}$  represent the elements at the  $i$ -th row and  $j$ -th column of  $\mathbf{R}$  and  $\mathbf{Q}$ , respectively. By minimizing the distance between  $\mathbf{R}$  and  $\mathbf{Q}$  according to the equations, we align the feature distance and pixel distance, thus assigning spatial relations.

$$
- L o s s (R, Q) = \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} \left(r _ {i j} - q _ {i j}\right) ^ {2} \tag {1}
$$

By repeating the same elements in a matrix  $M$  of size  $n_r \times n_c$ , we obtain an image of size (84, 84). Applying the same (84, 84) image to each channel, we obtain an image of size (3, 84, 84). We refer to this image transformed from tabular data as the tabular image. Figure 2 represents the results of transforming one data sample from each of the six datasets(Vanschoren et al., 2014) used in the experiment into tabular images according to the proposed domain transformation method. Algorithm 1 at Appendix D shows the detailed process of domain transformation

# 3.3 LEARNING PRIOR KNOWLEDGE

The proposed TablEye model consists of a backbone that serves as an embedding function to operate in the suitable dimension for few-shot learning, and a classifier that performs the few-shot learning

![](images/bc3c005bf14a04954c1c74271a6f47141f8ac585a17dd69fdbfb09f101788851.jpg)

![](images/f471e6f8566d3af5937f3c4dc9b0043ffc9cc081f5ebbbba988600c768e3f22c.jpg)

![](images/e136fadde496ebb3931e5cd9f4419026170b720c60758f28397239328b1cec4a.jpg)

![](images/f793fdec7a109fa90c1fa7326fab8710a87f79afe35e65e59f03163daff0bcc8.jpg)  
Figure 2: Example tabular images. (a), (b), (c), (d), (e) and (f) are tabular images from CMC, Diabetes, Karhunen, Optdigits, Lung and Cancer data respectively.

![](images/9dad5a4553dfb6c9e8eb45636f9ccf8c0c3d80a76d4e3b309dc6d21477cb9c1e.jpg)

![](images/af0984b7426859879e3a617abf96a2229259cc5358413212927244ac4ddb9186.jpg)

task based on the embedded support set. TablEye utilizes mini-ImageNet(Vinyals et al., 2016) to train backbone and classifier We adopted four different backbone architectures as shown in Figure 3. It is because the structure and training state of the backbone can significantly impact the training state of the classifier. Figure 3 illustrates the actual architectures of the four backbones, namely Resnet12, Conv2, Conv3, and Conv4, proposed and experimentally validated in this paper. The schematic diagram depicting the ResNet12 architecture is derived from the seminal work presented in the Choi et al. (2018) paper. Hereinafter, Resnet12, Conv2, Conv3, and Conv4 refer to each backbone depicted in Figure 3 within this paper. Resnet12 is a complex and deep backbone with a 12-layer ResNet(He et al., 2016) structure. Conv2, Conv3, and Conv4 are intuitive and shallow backbone architectures with 2, 3, and 4-layer CNN networks, respectively.

The backbone continuously learns to achieve a better embedding function for the classifier based on the predictions of the classifier using cross-entropy loss. The classifier plays a direct role in the few-shot learning task based on the embedded tabular images as latent vectors. In this paper, we adopt the principles of Prototypical Network(Snell et al., 2017), prototypes and inner loop adaptation of MAML(Finn et al., 2017) as our classifier. Both principles can be applied to various machine learning model structures. Moreover, recent studies have shown that few-shot learning with Prototypical Network achieves better performance than other complex few-shot learning models. Considering our goal of creating a model that operates with limited computing resources, we choose these two options for the classifier. When selecting the Proto-layer as the classifier, the classifier forms prototypes by averaging the latent vectors of the support sets for each class. It predicts the result by measuring the distances between the latent vectors of the query set and each prototype to determine the closest class. Alternatively, when selecting the MAML-layer as the classifier, we iteratively train a trainable fully connected layer within the inner loop using the latent vectors of the support set. The fully connected layer is then applied to the latent vectors of the query set to make predictions. Algorithm 2, 3 at Appendix D explain the detailed process of backbone and classifier.

![](images/902fb37e33bcd031707d651699d10c12ee0eb6ee0dae3e1d454dde95346deb3a.jpg)  
Figure 3: Four Backbone Structures of TabEye. Conv2, Conv3, Conv4 are composed of multiple Conv Block.

![](images/6b1bcbeddacf617a59bc087185b161e3e6d12a83953f99dc4f3e336c2175428a.jpg)

![](images/44b05e6ddd3816af25b2fab72987c7b965a4284a2d2591b5fd16f82a72915ed8.jpg)

![](images/001af8c7eff381d1e21cf8ce91a25a941db600c26aa2ca00cea5ffbdb9701d43.jpg)

![](images/6438f60a5df0ac74a158d650c032790df4b072d89ee493f025987bee5994949a.jpg)

# 4 EXPERIMENTS

# 4.1 EXPERIMENTAL ENVIRONMENT

Data To validate the hypothesis of this paper, we conducted experiments using image data from mini-ImageNet(Vinyals et al., 2016) and open tabular data from OpenML(Vanschoren et al., 2014) and Kaggle. We constructed a train set consisting of 50,400 images and a validation set of 9,600 images from mini-ImageNet. For the test set, we composed tabular images after domain transformation. To ensure clear validation of the hypothesis, we applied the following criteria in selecting the tabular datasets for experiments: (1) Diversity of features: dataset containing only categorical features, dataset containing only numerical features, and dataset containing both categorical and numerical features, (2) Diversity of tasks: binary classification and multiclass classification, (3) Inclusion of medical data for industrial value validation. Appendix B shows the detail of the datasets.

Notation The abbreviation 'T-A-B' signifies a condensed form of 'TabEye-A-B', denoting the implementation of TabEye with 'A' serving as the classifier and 'B' as the backbone. Here, 'P' and 'M' denotes 'Proto-layer' and 'MAML-layer'. 'C2' 'C3' 'C4' and 'R' represents 'Conv2', 'Conv3', 'Conv4' and 'Resnet12'.

# 4.2 ABLATION STUDY

Throughout the research process, the main question was whether the prior knowledge learned from natural images could be applied to tabular images. To address this, we employed t-SNE(t-Distributed Stochastic Neighbor Embedding)(Van der Maaten & Hinton, 2008) technique to embed and visualize the distributions of natural images and transformed tabular images in a 2-dimensional space. Figure 4 visually presents the results of embedding into a two-dimensional space using t-SNE. Based on the 2-dimensional embedding results, we measured the maximum distance, denoted as  $distance_{max}$ , from the mean vector of natural images as the center of two circles,  $c_1$  and  $c_2$ . We then drew two circles: circle  $c_1$  with a radius of  $distance_{max}$  and circle  $c_2$  with a radius of  $0.8 * distance_{max}$ . The scattered points in Figure 4 represent individual data samples, while the red and blue circles represent  $c_1$  and  $c_2$ , respectively. We observed that some tabular images fell within  $c_2$  while the majority of tabular images fell within  $c_1$ . Therefore, we concluded that there is no domain shift issue in learning the prior knowledge of tabular images from natural images.

![](images/e2d6030073e15ac22d22ec7b06015d88745fbce5e1edfc7262780bf7eb3041f6.jpg)  
Figure 4: Visualization of Natural Image and Tabular Image Using T-SNE. Each points indicate tabular image, red circle(larger circe) indicates  $c_{1}$  and blue circle(smaller circle) indicates  $c_{2}$ . The distinction of the six tabular datasets can be accomplished through the observation of the colors and shapes of the points.

To empirically substantiate the influence of acquiring prior knowledge from the image domain, we evaluated the accuracy of few-shot tabular classification under two different conditions: 1) directly applying few-shot learning algorithms designed for image data to tabular images, and 2) leveraging the mini-ImageNet dataset for prior knowledge acquisition before employing the same algorithms. When directly applying few-shot learning algorithms, we used randomly initialized backbone Table 1 elucidates the ramifications of incorporating prior knowledge from the image domain on the efficacy of few-shot tabular classification tasks. Excluding 1-shot scenarios for the accuracy of the T-P-R(CMC) and T-P-C3(Karhunen), we observed a substantial enhancement in performance in all

other cases when learning originated in the image domain. Thus, we have ascertained that the potency of TablEye not only stems from the few-shot learning algorithms but also from the benefits accrued through prior knowledge acquisition in the image domain.

Table 1: Comparison of Few-shot Tabular Classification Accuracy Based on Prior Knowledge Learning in the Image Domain. 'No Img' represents the condition where no prior knowledge learning has occurred in the image domain. Randomly initialized backbone is applied and trained on a tabular image. 'Img' denotes cases where prior knowledge has been acquired using mini-ImageNet. We report the mean of over 100 iterations.

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">CMC</td><td colspan="2">Diabetes</td><td colspan="2">Karhunen</td><td colspan="2">Optdigits</td></tr><tr><td>No Img</td><td>Img</td><td>No img</td><td>Img</td><td>No img</td><td>Img</td><td>No img</td><td>Img</td></tr><tr><td rowspan="4">1-shot</td><td>T-P-R</td><td>35.36</td><td>35.97</td><td>54.67</td><td>58.83</td><td>30.57</td><td>30.50</td><td>43.11</td><td>44.32</td></tr><tr><td>T-P-C2</td><td>36.42</td><td>37.33</td><td>56.26</td><td>56.53</td><td>45.39</td><td>51.21</td><td>64.58</td><td>71.18</td></tr><tr><td>T-P-C3</td><td>37.46</td><td>37.31</td><td>55.68</td><td>57.43</td><td>44.42</td><td>51.39</td><td>63.59</td><td>70.30</td></tr><tr><td>T-P-C4</td><td>36.84</td><td>37.45</td><td>54.88</td><td>57.79</td><td>41.67</td><td>44.85</td><td>62.96</td><td>65.76</td></tr><tr><td rowspan="4">5-shot</td><td>T-P-R</td><td>36.35</td><td>38.37</td><td>56.48</td><td>64.39</td><td>31.80</td><td>41.18</td><td>41.77</td><td>51.83</td></tr><tr><td>T-P-C2</td><td>38.99</td><td>40.34</td><td>57.65</td><td>65.15</td><td>41.78</td><td>77.94</td><td>62.54</td><td>87.44</td></tr><tr><td>T-P-C3</td><td>38.50</td><td>41.22</td><td>57.23</td><td>66.20</td><td>40.98</td><td>74.61</td><td>62.02</td><td>86.83</td></tr><tr><td>T-P-C4</td><td>38.27</td><td>40.89</td><td>55.59</td><td>68.73</td><td>37.86</td><td>70.72</td><td>61.54</td><td>84.58</td></tr></table>

# 4.3 COMPARISON RESULTS WITH TABLLM

Data The dataset for TabLLM(Hegselmann et al., 2023) is constrained by token length and the absence of meaningful feature names, which restricts its applicability to datasets such as Karhunen and Optdigits. The datasets utilized in the other experiments, Karhunen and Optdigits, comprised 65 features, rendering TabLLM experiments infeasible. Moreover, these datasets lacked meaningful feature names. Consequently, alternative datasets used in experiments of previous work were selected to replace those. The Diabetes dataset exclusively comprises numerical features, the Heart dataset encompasses both numerical and categorical features, and the Car dataset solely comprises categorical features.

Metric For the Metric, we used the AUC (Area Under the Receiver Operating Characteristic Curve) metric to compare our method under the same conditions as TabLLM.

Shot setting In the paper of TabLLM, comparisons were made from 4-shot to 512-shot. We assume, however, a few-shot scenario, we compared the AUC under 4-shot, 8-shot, and 16-shot conditions.

TabLLM transforms tabular data consisting of categorical and numerical features into prompts that can be understood by language model. It leverages the prior knowledge of language models using these prompts. Table 2 displays the performance comparison between our approach, table-to-image and TabLLM, table-to-text method. TabEye exhibited superior performance to previous work in numeric-only datasets, Diabetes and Heart and showed similar or superior performance in the Categoric-only dataset, Car. TabLLM showed best performance in 4-shot scenarios but T-M-C4 demonstrated 0.89 AUC that was 0.03 higher than TabLLM in 16-shot scenarios.

Tableye exhibited an approximately 0.1 higher AUC than TabLLM on the diabetes dataset. We believe this is due to TabLLM's power diminishing in numeric-only datasets, which are more distant from general language. However, in 4-shot scenarios of car datasets, TabEye consistently showed lower performance compared to table-to-text method. We speculate that this is because of the nature of TabLLM utilizing language model, better understands categorical features.

TabLLM has approximately 11 billion parameters((Sanh et al., 2021)), while TabEye utilizes up to 11 million parameters ResNet12 exhibits parameters that are approximately 1/916 the size of TabLLM. Conv2, Conv3, and Conv4 display parameter sizes that span a range from 1/97,345 to 1/282,051 when compared to TabLLM. TabEye has a significantly smaller model size compared to the table-to-text method. Our approach also can demonstrate comparable or superior performance and extremely efficient computation power. Appendix C provides the detailed information.

Table 2: Few Shot Tabular Classification Test AUC performance on 3 tabular datasets. We used the AUC performance of XGB, TabNet, SAINT and TabLLM from TabLLM paper. The bold indicates result within 0.01 from highest accuracy.  

<table><tr><td></td><td colspan="3">Diabetes</td><td colspan="3">Heart</td><td colspan="3">Car</td></tr><tr><td></td><td>4-shot</td><td>8-shot</td><td>16-shot</td><td>4-shot</td><td>8-shot</td><td>16-shot</td><td>4-shot</td><td>8-shot</td><td>16-shot</td></tr><tr><td>XGB</td><td>0.50</td><td>0.59</td><td>0.72</td><td>0.50</td><td>0.55</td><td>0.84</td><td>0.50</td><td>0.59</td><td>0.70</td></tr><tr><td>TabNet</td><td>0.56</td><td>0.56</td><td>0.64</td><td>0.56</td><td>0.70</td><td>0.73</td><td>**</td><td>0.54</td><td>0.64</td></tr><tr><td>SAINT</td><td>0.46</td><td>0.65</td><td>0.73</td><td>0.80</td><td>0.83</td><td>0.88</td><td>0.56</td><td>0.64</td><td>0.76</td></tr><tr><td>TabLLM</td><td>0.61</td><td>0.63</td><td>0.69</td><td>0.76</td><td>0.83</td><td>0.87</td><td>0.83</td><td>0.85</td><td>0.86</td></tr><tr><td>T-P-R</td><td>0.68</td><td>0.70</td><td>0.69</td><td>0.72</td><td>0.78</td><td>0.69</td><td>0.69</td><td>0.68</td><td>0.75</td></tr><tr><td>T-P-C2</td><td>0.68</td><td>0.68</td><td>0.68</td><td>0.84</td><td>0.83</td><td>0.85</td><td>0.79</td><td>0.79</td><td>0.79</td></tr><tr><td>T-P-C3</td><td>0.71</td><td>0.73</td><td>0.71</td><td>0.86</td><td>0.79</td><td>0.78</td><td>0.72</td><td>0.71</td><td>0.76</td></tr><tr><td>T-P-C4</td><td>0.72</td><td>0.71</td><td>0.69</td><td>0.82</td><td>0.81</td><td>0.79</td><td>0.79</td><td>0.83</td><td>0.83</td></tr><tr><td>T-M-C2</td><td>0.68</td><td>0.73</td><td>0.78</td><td>0.81</td><td>0.83</td><td>0.82</td><td>0.74</td><td>0.82</td><td>0.86</td></tr><tr><td>T-M-C3</td><td>0.71</td><td>0.74</td><td>0.76</td><td>0.73</td><td>0.83</td><td>0.83</td><td>0.78</td><td>0.85</td><td>0.87</td></tr><tr><td>T-M-C4</td><td>0.69</td><td>0.74</td><td>0.75</td><td>0.82</td><td>0.84</td><td>0.88</td><td>0.75</td><td>0.82</td><td>0.89</td></tr></table>

# 4.4 FEW-SHOT CLASSIFICATION RESULTS WITH BASELINE

Baseline We chose a supervised learning models that can be experimented within a meta-learning setting without an unlabeled set. We selected both tree-based model and neural network-based model known for their high performance about tabular learning (Shwartz-Ziv & Armon, 2022).

STUNT A fixed number of unlabeled sets were used as the train set. For the CMC, Diabetes, Karhunen, and Optdigits datasets, 441, 230, 600, and 1686 unlabeled sets were respectively utilized.

Table 3 displays the performance of TabEye. The results demonstrate the superiority of TabEye over traditional methods such as XGB and TabNet(Arik & Pfister, 2021), and even over STUNT, which is state of the art about few-shot tabular learning. In the 1-shot setting, methods of TabEye, T-P-C2 and T-P-C3 exhibited the highest average accuracies of  $54.06\%$  and  $54.11\%$ , respectively, outperforming all other methods. The performance advantage of TabEye was also evident in the 5-shot setting, where the T-P-C2 and T-P-C3 methods continued to outperform other methods, achieving average accuracies of  $67.72\%$  and  $67.22\%$ , respectively.

STUNT(Nam et al., 2023) showed a considerable performance with average accuracies of  $50.94\%$  and  $66.46\%$  in the 1-shot and 5-shot settings respectively. The performance of STUNT is, however, heavily influenced by the size of the unlabeled dataset. In real-world industrial processes, obtaining a sufficiently large and well-composed unlabeled dataset is often challenging, making superior performance of TabEye without relying on unlabeled data highly notable.

# 5 DISCUSSION

TabLLM was unable to handle datasets with more than a certain number of features or meaningless feature names, such as the Karhunen and Optdigits datasets with 65 features and feature names like f1, f2, and f3. It is because of the limitations in the token size of the LLM and the necessity for meaningful feature names. The results of our approach confirmed higher performance compared to table-to-text method, particularly in datasets with numerical features such as Diabetes and Heart. Comparing the size of the TabLLM and TabEye previous work possessed a significantly larger number of parameters, requiring considerably higher computational power. Nevertheless, our method demonstrated superior performance with the Diabetes and Heart datasets. Thus, we conclude that our approach is more efficient and showed similar or superior performance for various datasets, overcoming the limitations of TabLLM, which has restrictions on the datasets it can handle and requires high computational power.

STUNT requires a substantial amount of unlabeled data for training. The model used  $80\%$  of the total data as an unlabeled set for training in its paper. In this study, we aimed to use as little unlabeled data as possible to conduct experiments under similar conditions to other baselines, utilizing

Table 3: Few Shot Classification test accuracy(%) on 4 public tabular dataset. We report the mean of over 100 iterations. The bold indicates result within  $1\%$  from highest accuracy.  

<table><tr><td></td><td>Method</td><td>CMC</td><td>Diabetes</td><td>Karhunen</td><td>Optdigits</td><td>Average</td></tr><tr><td rowspan="10">1-shot</td><td>XGB</td><td>33.33</td><td>50.00</td><td>20.00</td><td>20.00</td><td>30.83</td></tr><tr><td>TabNet</td><td>34.84</td><td>51.90</td><td>21.97</td><td>20.45</td><td>32.29</td></tr><tr><td>STUNT</td><td>36.52</td><td>51.60</td><td>47.72</td><td>67.92</td><td>50.94</td></tr><tr><td>T-P-R</td><td>35.97</td><td>58.83</td><td>30.50</td><td>44.32</td><td>42.41</td></tr><tr><td>T-P-C2</td><td>37.33</td><td>56.53</td><td>51.21</td><td>71.18</td><td>54.06</td></tr><tr><td>T-P-C3</td><td>37.31</td><td>57.43</td><td>51.39</td><td>70.30</td><td>54.11</td></tr><tr><td>T-P-C4</td><td>37.45</td><td>57.79</td><td>44.85</td><td>65.76</td><td>51.46</td></tr><tr><td>T-M-C2</td><td>36.60</td><td>58.34</td><td>41.92</td><td>62.04</td><td>49.73</td></tr><tr><td>T-M-C3</td><td>37.26</td><td>58.57</td><td>43.27</td><td>60.18</td><td>49.82</td></tr><tr><td>T-M-C4</td><td>37.30</td><td>57.30</td><td>43.45</td><td>60.53</td><td>49.65</td></tr><tr><td rowspan="10">5-shot</td><td>XGB</td><td>42.18</td><td>61.20</td><td>68.21</td><td>73.19</td><td>61.19</td></tr><tr><td>TabNet</td><td>36.07</td><td>50.23</td><td>20.28</td><td>21.33</td><td>31.98</td></tr><tr><td>STUNT</td><td>41.36</td><td>55.43</td><td>83.00</td><td>86.05</td><td>66.46</td></tr><tr><td>T-P-R</td><td>38.37</td><td>64.39</td><td>41.18</td><td>51.83</td><td>48.94</td></tr><tr><td>T-P-C2</td><td>40.34</td><td>65.15</td><td>77.94</td><td>87.44</td><td>67.72</td></tr><tr><td>T-P-C3</td><td>41.22</td><td>66.20</td><td>74.61</td><td>86.83</td><td>67.22</td></tr><tr><td>T-P-C4</td><td>40.89</td><td>68.73</td><td>70.72</td><td>84.58</td><td>66.23</td></tr><tr><td>T-M-C2</td><td>37.65</td><td>63.18</td><td>56.38</td><td>62.79</td><td>55.00</td></tr><tr><td>T-M-C3</td><td>38.48</td><td>64.35</td><td>44.80</td><td>58.79</td><td>51.60</td></tr><tr><td>T-M-C4</td><td>37.95</td><td>65.94</td><td>59.12</td><td>71.85</td><td>58.71</td></tr></table>

approximately  $30\%$  of the data as the unlabeled set for training. Despite employing a considerable number of unlabeled sets in experiments, TablEye, which did not use any unlabeled sets, showed higher accuracy than STUNT. Therefore, we believe that TablEye has overcome the limitations of STUNT, which requires a large unlabeled set.

When applying TablEye to medical datasets, we observed markedly higher accuracy compared to other baselines. Compared to existing methods, we achieved an average accuracy of  $15\%$  higher in 1-shot scenarios and approximately  $2\%$  higher in 5-shot scenarios. These results indicate that our method can produce meaningful results not only in public tabular data but also in medical data of industrial value.

# 6 CONCLUSION

In this paper, we propose TablEye, a novel few-shot tabular learning framework that leverages prior knowledge acquired from the image domain. TablEye performs a transformation of tabular data into the image domain. It then utilizes prior knowledge gained from extensive labeled image data to execute few-shot learning. Our experiments on various public tabular datasets affirm the efficacy of TablEye. Experimental results indicate a notable increase in performance metrics; TablEye surpasses TabLLM by a maximum of 0.11 AUC except for one 4-shot learning and demonstrates an average accuracy enhancement of  $3.17\%$  over STUNT in the 1-shot learning scenario. Notably, our approach effectively overcomes several limitations including a dependence on the number and names of features in the dataset, the need for substantial computational power, and the requirement for a large unlabeled set. We believe that leveraging the image domain to solve problems in the tabular domain opens up exciting new possibilities for advancing the field of tabular learning.

# 7 REPRODUCIBLITY

To reproduce the framework proposed in this paper, the few-shot learning task process was implemented using the LibFewShot library[36]. A configuration file for utilizing the LibFewShot library and detailed model setting for reproduction will be made publicly available on GitHub.

# REFERENCES

Sercan Ö Arik and Tomas Pfister. Tabnet: Attentive interpretable tabular learning. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 6679-6687, 2021.  
Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 2022.  
Léon Bottou and Olivier Bousquet. The tradeoffs of large scale learning. Advances in neural information processing systems, 20, 2007.  
Léon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM review, 60(2):223-311, 2018.  
Hyungeun Choi, Seunghyoung Ryu, and Hongseok Kim. Short-term load forecasting based on resnet and LSTM. In 2018 IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm), pp. 1-6, 2018.  
Jillian M Clements, Di Xu, Nooshin Yousefi, and Dmitry Efimov. Sequential deep learning for credit risk monitoring with tabular financial data. arXiv preprint arXiv:2012.15330, 2020.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.  
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pp. 1126-1135, 2017.  
Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. Deepfm: a factorization-machine based neural network for ctr prediction. arXiv preprint arXiv:1703.04247, 2017.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. In International Conference on Artificial Intelligence and Statistics, pp. 5549-5581, 2023.  
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950-1965, 2022.  
Yael Mathov, Eden Levy, Ziv Katzir, Asaf Shabtai, and Yuval Elovici. Not all datasets are born equal: On heterogeneous data and adversarial examples. arXiv preprint arXiv:2010.03180, 2020.  
Jaehyun Nam, Jihoon Tack, Kyungmin Lee, Hankook Lee, and Jinwoo Shin. Stunt: Few-shot tabular learning with self-generated tasks from unlabeled tables. arXiv preprint arXiv:2303.00918, 2023.  
Neeta Nathani and Abhishek Singh. Foundations of machine learning. In *Introduction to AI Techniques for Renewable Energy System*, pp. 43-64. CRC Press, 2021.  
Archit Parnami and Minwoo Lee. Learning from few examples: A summary of approaches to few-shot learning. arXiv preprint arXiv:2203.04291, 2022.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532-1543, 2014.  
Joseph Prusa, Taghi M Khoshgoftaar, and Naeem Seliya. The effect of dataset size on training tweet sentiment classifiers. In 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), pp. 96-102, 2015.

Mark Ryan. Deep learning with structured data. Simon and Schuster, 2020.  
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021.  
Ravid Shwartz-Ziv and Amitai Armon. Tabular data: Deep learning is not all you need. Information Fusion, 81:84-90, 2022.  
Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30, 2017.  
Margarita Sordo and Qing Zeng. On sample size and classification accuracy: A performance comparison. In International Symposium on Biological and Medical Data Analysis, pp. 193-201, 2005.  
Talip Ucar, Ehsan Hajiramezanali, and Lindsay Edwards. Subtab: Subsetting features of tabular data for self-supervised representation learning. Advances in Neural Information Processing Systems, 34:18853-18865, 2021.  
Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.  
Joaquin Vanschoren, Jan N Van Rijn, Bernd Bischl, and Luis Torgo. Openml: networked science in machine learning. ACM SIGKDD Explorations Newsletter, 15(2):49-60, 2014.  
Vladimir Vapnik. Principles of risk minimization for learning theory. Advances in neural information processing systems, 4, 1991.  
Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. Advances in neural information processing systems, 29, 2016.  
Yaqing Wang, Quanming Yao, James T Kwok, and Lionel M Ni. Generalizing from a few examples: A survey on few-shot learning. ACM computing surveys (csur), 53(3):1-34, 2020.  
Qingchen Zhang, Laurence T Yang, Zhikui Chen, and Peng Li. A survey on deep learning for big data. Information Fusion, 42:146-157, 2018.  
Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. Deep learning based recommender system: A survey and new perspectives. ACM computing surveys (CSUR), 52(1):1-38, 2019.  
Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018.  
Yitan Zhu, Thomas Brettin, Fangfang Xia, Alexander Partin, Maulik Shukla, Hyunseung Yoo, Yvonne A Evrard, James H Doroshow, and Rick L Stevens. Converting tabular data into images for deep learning with convolutional neural networks. *Scientific reports*, 11(1):11325, 2021.
