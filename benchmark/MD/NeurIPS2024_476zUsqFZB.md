# PMechRP: Interpretable Deep Learning for Polar Reaction Prediction

Anonymous Author(s)

Affiliation

Address

email

# Abstract

In recent years, deep learning methods have been widely applied to chemical reaction prediction due to the time consuming and resource intensive nature of designing synthetic pathways. However, with the majority of models being trained on the US Patent Office dataset, many proposed architectures lack interpretability by modeling chemical reactions as overall transformations. These models map directly from reactants to products, and provide minimal insight into the underlying driving forces of a reaction. In order to improve interpretability and provide insight into the causality of a chemical reaction, we train various machine learning frameworks on the PMechDB dataset. This dataset contains polar elementary steps, which model chemical reactions as a sequence of steps associated with movements of electrons. Through training on PMechDB, we have created a new system for polar mechanistic reaction prediction: PMechRP. Our findings indicate that PMechRP is able to provide both accurate and interpretable predictions, with a novel two-step transformer based method achieving the highest top-5 accuracy at  $89.9\%$ .

# 1 Introduction

Two main approaches exist for the prediction of chemical reactions: machine learning based methods, and quantum chemistry based methods [1, 13, 5, 8]. While quantum chemistry models offer detailed prediction of chemical properties, their computational demands render them feasible only for a limited scope of reaction systems, precluding their use for broad-spectrum, high-throughput reaction prediction. Conversely, ML models offer computational efficiency and scalability, making them well-suited for application across larger chemical systems and datasets. Countless ML models have been devised for tasks such as reaction yield prediction [16], reaction classification [14], chemical property prediction [4, 2], and both forward and reverse reaction prediction [6, 20, 3, 10].

Although ML models offer a high-throughput and highly adaptable chemical prediction, a significant drawback lies in their lack of interpretability when compared to quantum chemistry or simulation based methods. The predominant approach of predicting reactions as overall transformations results in a black-box scenario, where predicted products emerge directly from reactants without insight into intermediate transition states. Although these models may achieve high accuracy on datasets like the US Patent Office dataset [11], their outputs pose challenges for organic chemists, who typically reason through chemical synthesis via arrow-pushing mechanisms rather than overall transformations. An example of a overall transformation vs a mechanistic elementary step approach can be seen in Figure 1. The elementary step approach breaks the overall transformation down into a sequence of arrow pushing steps, which illustrate the flow of electrons and the shifting of atoms.

![](images/475c903864c5a6daeaea8d5a9c02c764a8b6fd3c4d6911dbd5ad7cb73f62a78f.jpg)  
OVERALL TRANSFORMATION

![](images/68e6508d2774cde9cb26166923bffe52bf255fad358960500f3a1efb7ff7cda3.jpg)  
Stepwise arrow-pushing mechanism  
Figure 1: Example of an overall transformation vs an elementary step approach. This is a the final reaction step in the synthesis of enzalutamide, a drug used to treat prostate cancer that generates over $6 billion a year in revenue [21].

By thinking about reactions as occurring through many elementary steps, organic chemists are able to reason about the underlying driving forces of a reaction. When training ML models to forecast elementary step reactions, we effectively guide them to emulate an organic chemists' thought processes, thereby generating predictions that are readily interpretable and serve as practical aids for organic synthesis design.

# 2 Data

To develop predictive models for polar reaction mechanisms, we undertook training on the recently introduced PMechDB dataset. This dataset comprises more than 12,700 polar elementary steps, each balanced, partially atom mapped, and manually verified by a team of organic chemists. Each reaction represents a single elementary step polar reaction. These entries have been collected through manual curation from a diverse array of chemistry literature and textbooks [19]. These reactions are stored as smiles strings, and notably, the reactions contain arrow pushing information, providing insights into the reactivity of individual atoms within each reaction. Leveraging the manually curated reactions within the dataset, we conducted an 80/10/10 train/val/test split via random sampling from the "manually.curated_all.csv" file. For models which perform cross-validation, the validation data was combined with the training data.

# 3 Methods

Here we describe two different machine learning approaches for predicting polar elementary step mechanisms. Namely, we describe the reactive atom two-step approach, the single-step seq-to-seq prediction methods, and a spectator focused two-step transformer method.

# 3.1 Two-Step Prediction

The two-step prediction model comprises distinct phases. Initially, the model undertakes the task of predicting reactive atoms within the given reaction. Subsequently, these identified reactive sites are paired to formulate potential reaction mechanisms, followed by the application of a ranker model to rank the plausibility of these proposed mechanisms. This architectural design yields highly interpretable predictions, enabling a granular understanding of the model's rationale. When generating predictions, users can discern precisely which atoms are deemed reactive, and they can view the precise arrow-pushing mechanism predicted by the model. From the view point of organic chemists, the two-step architecture offers greater transparency compared to single-step approaches, as the arrow pushing mechanism provides justification for why the final products were predicted.

# 3.1.1 Siamese Architecture

The two-step siamese architecture [6] comprises three distinct models, each serving a specific function. Initially, two separate reactive atom predictor models are instantiated. One model is specifically trained for predicting source atoms, while the other is trained for predicting sink atoms. To train the source and sink models, the electron-donating atom from the intermolecular arrow is labeled as the source atom, while the electron-accepting atom is labeled as the sink atom. This labeling process employs the reactive sites identification method as detailed in [6]. Atoms are represented by continuous vectors derived from predefined atomic and graph-topological features, utilizing a neighborhood of size 3. Subsequently, both source and sink classifiers are trained to categorize these feature vectors accordingly. After the trained reactive atom classifiers predict source and sink atoms, these atoms are paired together to enumerate possible arrow pushing mechanisms. Afterwards, a siamese architecture is used as a plausibility ranker model, which then ranks the plausibility of each potential mechanism to generate a final set of predictions. A visual representation of the source and sink pair is provided in Figure 2.

# 3.1.2 OrbChain

A polar elementary step reaction Rxn can be modeled as the following: a set of reactant molecules  $R = \{r_0, r_1, \ldots, r_n\}$ , a set of product molecules  $P = \{p_0, p_1, \ldots, p_n\}$ , and a set of arrows  $\alpha = \{a_0, a_1, \ldots, a_m\}$ , which transforms R into P. We consider a molecular orbital (MO)  $m_i^{(*)}$  to be associated with four parameters:  $\mathfrak{m} = (\mathrm{a}, \mathrm{e}, \mathrm{n}, \mathrm{c})$ , where a represents the atom corresponding to the molecular orbital, e denotes the number of electrons contained in the MO, n corresponds to the atom adjacent to atom a in the case of a bond orbital, and c represents a possible chain of filled or unfilled MOs. Based on the methods described in [6, 9, 18], we model a polar mechanism as an interaction between two reactive molecular orbitals  $(m_1^{(*)}, m_2^{(*)})$ , where one orbital is the "source" orbital and acts as a nucleophile, while the other orbital is the "sink" orbital and acts as the electrophile. Given atom mapped reactants and products, and A, we can uniquely determine the reactive pair of orbitals in R used to create P. Conversely, given the reactive pair of orbitals  $(m_1^{(*)}, m_2^{(*)})$  and the reactants R, we can generate P given R.

# 3.1.3 Reactive Atom Prediction and Plausibility Ranking

We enumerate over all molecular orbitals found in reactants R, and divide orbitals into reactive and non-reactive orbitals. These positive and negative examples are used to train the source and sink identification models. Rather than directly predicting the reactive MOs, we perform a binary classification prediction on the label of atom a, which is associated with the molecular orbital. We adopt the reactive sites identification method from [6] and represent atoms using continuous vectors bcsed on predefined graph-topological and physiochemical features. We train two models: a source model and a sink model. The source model predicts a binary classification label for whether or not an atom is a source, while the sink predicts a binary classification for whether or not an atom is a sink. The training data was constructed by extracting the labeled source, and the labeled sink atom from each reaction as positive examples, and then randomly sampling non-source or non-sink examples to use as negative examples.

![](images/a9268fe0f96882db065939eb3db9791d0e5dbfcd992e7a3c95dfbb46f8ea6f01.jpg)  
Figure 2: An example of a simple polar elementary step. The electron pushing arrows can be seen in blue, while the source and sink sites are seen in red. The bromine atom labeled 10 is the source atom. The carbon atom labeled 20 is the sink atom. The corresponding SMILES string and arrow codes can be seen below.

# 3.2 Plausibility Ranking

Once a set of source atoms and sink atoms are predicted, these two sets are paired together to generate pairs of molecular orbitals. A siamese network is used to rank the resulting molecular orbital pairs to generate the final reaction mechanism predictions.

# 3.3 Seq-to-seq Prediction

In addition to exploring two-step models, we also explore the performance of text-based models. An exceedingly common representation of chemical reactions is in the form of SMILES strings (simplified molecular-input line-entry system), which is a text-based representation. This representation lends itself towards NLP models such as transformers. These architectures model reaction prediction as a translation problem, wherein they are translating from reactant SMILES to product SMILES. These models have achieved state-of-the-art accuracies when predicting overall chemical transformations. However, these models possess several drawbacks in that they are more difficult to interpret and do not explicitly encode important molecular information such as invariance to atom permutations. This means that the same reaction can be represented by a large number of different SMILES strings, and additional strategies such as data augmentation may be needed to prevent a transformer model from making different predictions for identical sets of reactants.

# 3.3.1 Molecular Transformer

We utilize the innovative text-based reaction predictor, Molecular Transformer [15], which employs a bidirectional encoder and autoregressive decoder coupled with a fully connected network to generate probability distributions over potential tokens. The pre-trained Molecular Transformers underwent training using various versions of the USPTO dataset. We did not separate reactants and reagents, so the model pre-trained using the USPTO_MIT_mixed dataset was selected and subsequently fine tuned on the PMechDB dataset.

# 3.3.2 Chemformer

In addition to the molecular transformer, we also adopt the Chemformer model [7], which is another transformer-based reaction predictor. The Chemformer model also employs a bidirectional encoder and autoregressive decoder with a fully connected network to generate probability distributions over potential tokens. The Chemformer model was pre-trained on molecular reconstruction and classification tasks using a dataset of 100M SMILES strings from the ZINC-15 [17] dataset. Afterwards, the model was fine-tuned on various downstream tasks including forward prediction and retrosynthesis. The pre-training substantially improved the model's generalizability and convergence times on downstream tasks, such as USPTO forward prediction, compared to randomly initialized models. We chose to start from the model fine-tuned on USPTO-mixed since reactants and reagents

are not separated in the PMechDB dataset. This model was subsequently fine-tuned on the PMechDB dataset for mechanistic-level predictions. The vocabulary of the model was expanded by 66 tokens to account for unseen atoms in the PMechDB dataset.

# 3.3.3 Two-Step Transformer Architecture

During experiments, all models were observed to exhibit a significant decrease in performance in reaction prediction as the size of the reactants grows. A quantitative analysis of the effects of spectators, and the number of atoms can be found in Figure 5 and Figure 6 respectively. To combat this, we propose a novel two-step architecture for transformers. Firstly, we use the source and sink reactive atom models from the siamese architecture to predict top-2 reactive atoms of the model. Reactant molecules which contain the predicted reactive atoms are considered to be non-spectator molecules. Since we take top-2 predictions from the source and sink models, we predict at most 2 sink molecules, and at most 2 source molecules. Pairing the sinks and sources together, we can have at most 4-unique source-sink combinations. After the combinations are generated, we run a top-5 prediction using our best performing transformer on each combination. Hence a fine-tuned chemformer model was used on each combination, as well as on the original reactants. After generating predictions for the source-sink combinations, the molecules which were deemed as spectators and removed are added back into the predicted products. If there are fewer than 4-unique source-sink combinations, more predictions are made on the original reactants until 5 total predictions are generated. For each reaction, we take the output predictions, canonicalize them, and then perform a simple majority vote with ties being broken randomly.

This architecture takes inspiration from common practices in organic chemistry. Often times when an organic chemist aims to predict the outcome of a set of reactants, they quickly look through all reactant molecules, and filter away molecules which are likely to be spectators or non-reactive, before focusing on a few molecules of interest. By performing a two-step prediction, we are able to first filter away potential spectator ions, then predict the reaction mechanism after reducing the space of possible reactions exponentially. A considerable performance increase was observed after performing this method of ensembling. The results can be seen in Table 3 and Table 4.

# 3.4 Multi-task learning

Due to the highly related nature of many chemistry prediction tasks, multitask learning can be used to develop robust models which may demonstrate improved learning efficiency and prediction accuracy. T5Chem is one such model, which leverages multitask learning on a transformer architecture to perform 5 different tasks. The T5Chem multi-task transformer architecture is able to perform forward/backwards prediction, reaction yield prediction, reaction classification, and reagents prediction [12]. This architecture was first pretrained with a BERT-like MLM objective on 97 million PubChem molecules. Then, the model was further fine-tuned on 5 different tasks using the USPTO_500_MT dataset. We selected this model, and fine-tuned it using the 80/10/10 split of the manually curated PMechDB reactions.

# 4 Results and Discussion

# 4.1 Performance on PMechDB Dataset

We assess the performance of the two-step prediction method, comprising reactive sites identification and plausibility ranking. The top-N accuracy of the reactive sites identification on PMechDB is presented in Table 7. Reactive site identification is considered correct if both the source and sink atom were correctly identified within the top-N predictions of each model.

Table 1: Reactive Atom Classification for Siamese Architecture  

<table><tr><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-5</td><td>Top-10</td></tr><tr><td>53.8</td><td>79.0</td><td>86.8</td><td>91.8</td><td>94.4</td></tr></table>

The source and sink ranking models are able to predict the reactive atoms with relatively high accuracy. Although the reactive atom models are able to filter down the number of potentially reactive atoms significantly, due to the large number of atoms and aromatic structures contained in the polar reactions, enumerating all possible molecular orbital pairs leads to a large number of possible reaction mechanisms fed into the ranker model. Several reaction fingerprints were used for plausibility ranking. The results can be found in Table 2.

Table 2: Plausibility Ranking for Two-Step Architecture  

<table><tr><td>Model Type</td><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-4</td><td>Top-5</td></tr><tr><td>reactionFP</td><td>39.5</td><td>56.3</td><td>65.6</td><td>70.3</td><td>73.0</td></tr><tr><td>DRFP</td><td>37.3</td><td>52.2</td><td>60.1</td><td>67.1</td><td>72.5</td></tr><tr><td>rxnfp</td><td>35.1</td><td>51.3</td><td>60.5</td><td>66.1</td><td>70.0</td></tr></table>

In order to perform two-step prediction, both reactive site identification and plausibility ranking must be performed. Thus for the best performing two-step model, we use the reactionFP fingerprint for plausibility ranking. Therefore in Table 3, we consider this as the best two-step siamese model. For the Chemformer, MolTransformer, and T5Chem models, we fine-tuned the pretrained models on the PMechDB dataset. The results comparing all the trained models can be seen in Table 3

Table 3: Top-N Accuracy of Trained Models  

<table><tr><td>Model Type</td><td>Top-1</td><td>Top-3</td><td>Top-5</td><td>Top-10</td></tr><tr><td>Best Two-Step Siamese</td><td>39.5</td><td>65.6</td><td>73.0</td><td>76.6</td></tr><tr><td>MolTransformer</td><td>59.1</td><td>66.3</td><td>69.2</td><td>70.1</td></tr><tr><td>T5Chem</td><td>56.6</td><td>69.1</td><td>73.7</td><td>77.5</td></tr><tr><td>Chemformer</td><td>74.0</td><td>84.1</td><td>85.2</td><td>87.2</td></tr><tr><td>Two-Step Transformer</td><td>80.6</td><td>88.8</td><td>89.9</td><td>91.0</td></tr></table>

Although the Siamese two-step model allows for improved interpretability due to its direct prediction of arrows, the models based on Chemformer yield the most accurate predictions, with the two-step transformer model outperforming all other models significantly. The effects of various ensemble sizes can be seen in Table 4.

Table 4: Effects of Ensemble Size on Top-N Accuracy  

<table><tr><td>ensemblesize</td><td>Top-1</td><td>Top-3</td><td>Top-5</td><td>Top-10</td></tr><tr><td>2</td><td>71.8</td><td>85.8</td><td>86.9</td><td>87.7</td></tr><tr><td>3</td><td>77.8</td><td>87.5</td><td>88.5</td><td>89.2</td></tr><tr><td>4</td><td>79.8</td><td>88.7</td><td>90.0</td><td>90.7</td></tr><tr><td>5</td><td>80.6</td><td>88.8</td><td>89.9</td><td>91.0</td></tr></table>

# 4.1.1 Pretraining

Pretraining the Chemformer models made a large difference in performance, the effects of pretraining can be seen in Table 5.

The large increase in performance from the pretraining, indicates overlap between the USPTO dataset and the PMechDB dataset. This is in stark contrast to radical mechanisms, which exhibited lower performance when using a pretrained model [18]. This suggests that radical reactions are underrepresented in USPTO datasets compared to polar reactions, and that pre-trained transformer models would be expected to have higher performance on polar reactions.

# 4.2 Pathway Search

In addition to predicting single-step elementary reactions, further work is being done to evaluate and improve the model's performance on predicting polar mechanistic pathways. This involves chaining

Table 5: Top-N Accuracy of Chemformer Models  

<table><tr><td>Model Type</td><td>Top-1</td><td>Top-3</td><td>Top-5</td><td>Top-10</td></tr><tr><td>no-pretraining</td><td>39.9</td><td>55.6</td><td>58.7</td><td>60.4</td></tr><tr><td>pretrained on zinc</td><td>74.9</td><td>77.0</td><td>82.8</td><td>84.5</td></tr><tr><td>pretrained on zinc and USPTO Mixed</td><td>74.0</td><td>84.1</td><td>85.2</td><td>87.2</td></tr></table>

several elementary steps together to transform a list of starting reactants to a list of target products. An example of a simple two-step mechanism correctly predicted by the ensemble transformer model can be seen in Figure 3.

![](images/866e486d3180f7d6fb2497f598f7676b24a79f08dd74dd1e0d6107c1aef02d23.jpg)  
Figure 3: A simple 2-step mechanism correctly predicted by ensemble transformer model.

Although the transformer architectures outperform all other models in single step predictions on the test dataset, the reactions contained in PMechDB are mostly 1-2 reactant reactions, and contain very limited spectator ions. This results in the transformer models having a strong performance on reactions which contain 1-2 reactants, but inconsistent performance on reactions with one or more spectator ions. An example of this can be seen in the following elementary step which contains a spectator benzene ring. 4

When the chemformer model is asked to predict on Step A, it does not recover the correct products, while on Step B with spectators removed, it ranks the products as the top-1 prediction. Interestingly, the two-step transformer model is able to correctly predict this step. Comparing the various methods numerically, the two-step models appear to demonstrate significantly less performance degradation in predicting elementary steps with spectator molecules. Figure 5 demonstrates the top-5 accuracies of the various models as the number of reactant molecules is varied, while Figure 6 demonstrates the top-5 accuracies as the number of atoms contained in the reactants is varied.

The two-step transformer model can be seen to outperform both the chemformer and siamese architectures. When comparing the models, it seems that the number of reactant atoms has a much smaller effect on the prediction accuracy of the transformer models when compared to the siamese architecture. Perhaps this indicates that the transformer models are able to implicitly learn which reactive atoms it should pay attention to without being distracted by large unreactive substructures.

![](images/430fb7b0dbf6b260c97dabc493bbacee05f88c0de2146990a831ad80828636e1.jpg)  
A)

![](images/f545771c0d1b704929127f61d881aeb2ca180a75d45d8a4ff1e6be0747fe0ff7.jpg)  
B)  
Figure 4: Step A represents the elementary step with the spectator molecule benzene included. Step B represents the elementary step with the benzene ring excluded.

![](images/1d7b6bac6897f32faaf7728a8385a79b16ddc8b95c27ba06bb82beb038c27c6b.jpg)  
Top-5 Accuracy vs Number of Molecules

![](images/8f7f33efde415f0e112a8411ca1086ac3006a8f4a3d95ddcb50d430680bf930a.jpg)  
Figure 5: Comparing the top-5 accuracies of both the transformer and two-step models as number of reactant molecules is varied.  
Top-5 Accuracy vs Number of Reactant Atoms  
Figure 6: Comparing the top-5 accuracies of both the transformer and two-step models as number of reactant atoms is varied.

Notably, the two-step transformer model strongly outperforms the chemformer model when it views reactions which contain more than 2 reactant molecules. This suggests the first step manages to filter away the spectator ions to some extent and makes the prediction task easier for the transformer model.

# 5 Limitations

Lastly, we note there are several limitations with the current state of the PMechRP polar reaction system. Firstly, the PMechDB dataset includes less than 13,000 steps. This means the dataset is relatively small for training large architectures, and it may be difficult for these models to generalize well to all forms of experimental chemistry. Secondly, the transformer models directly translate from reactants to products, without generating the arrow pushing mechanisms. Although the elementary step predictions still offer significant interpretability, the two-step siamese method offers greater insight into the causality of a reaction by directly showing the flow of electrons. Additional methods could be developed to predict arrow codes or reactive orbitals using a transformer architecture in order to offer predictions with arrow pushing mechanisms.

# 6 Conclusion

We developed and compared several reaction prediction systems for polar reaction mechanisms. Through our analysis, we have created the reaction prediction system, PMechRP. This predictor offers a fresh perspective on reaction prediction by specifically targeting polar reactions and operating at the mechanistic reaction level. From the viewpoint of organic chemists, mechanistic level reaction prediction offers immense interpretability benefits, and has a lot of potential to aid in the prediction of synthetic pathways. We utilized PMechDB datasets to train and develop a wide range of architectures. Our findings demonstrate that the most accurate models are based on a two-step process, where spectators are filtered out to generate a variety of reactants before they are fed into an ensemble transformer architecture. Leveraging PMechDB datasets, our polar predictor marks a significant step towards interpretable reaction prediction.

# References

[1] Roman M Balabin and Ekaterina I Lomakina. Neural network approach to quantum-chemistry data: Accurate prediction of density functional theory energies. The journal of chemical physics, 131(7), 2009.  
[2] Connor W Coley, Regina Barzilay, William H Green, Tommi S Jaakkola, and Klavs F Jensen. Convolutional embedding of attributed molecular graphs for physical property prediction. Journal of chemical information and modeling, 57(8):1757-1772, 2017.  
[3] Connor W Coley, Regina Barzilay, Tommi S Jaakkola, William H Green, and Klavs F Jensen. Prediction of organic reaction outcomes using machine learning. ACS central science, 3(5):434-443, 2017.  
[4] Connor W Coley, Wengong Jin, Luke Rogers, Timothy F Jamison, Tommi S Jaakkola, William H Green, Regina Barzilay, and Klavs F Jensen. A graph-convolutional neural network model for the prediction of chemical reactivity. Chemical science, 10(2):370–377, 2019.  
[5] Larry A Curtiss, Paul C Redfern, and Krishnan Raghavachari. Gaussian-4 theory. The Journal of chemical physics, 126(8), 2007.  
[6] David Fooshee, Aaron Mood, Eugene Gutman, Mohammadamin Tavakoli, Gregor Urban, Frances Liu, Nancy Huynh, David Van Vranken, and Pierre Baldi. Deep learning for chemical reaction prediction. Molecular Systems Design & Engineering, 3(3):442-452, 2018.  
[7] Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum. Chemformer: a pretrained transformer for computational chemistry. Machine Learning: Science and Technology, 3(1):015022, 2022.

[8] Dora Kadish, Aaron D Mood, Mohammadamin Tavakoli, Eugene S Gutman, Pierre Baldi, and David L Van Vranken. Methyl cation affinities of canonical organic functional groups. The Journal of Organic Chemistry, 86(5):3721-3729, 2021.  
[9] Matthew A Kayala, Chloé-Agathe Azencott, Jonathan H Chen, and Pierre Baldi. Learning to predict chemical reactions. Journal of chemical information and modeling, 51(9):2209-2222, 2011.  
[10] Matthew A Kayala and Pierre Baldi. Reactionpredictor: prediction of complex chemical reactions at the mechanistic level using machine learning. Journal of chemical information and modeling, 52(10):2526-2540, 2012.  
[11] Daniel Mark Lowe. Extraction of chemical structures and reactions from the literature. PhD thesis, 2012.  
[12] Jieyu Lu and Yingkai Zhang. Unified deep learning model for multitask reaction predictions with explanation. Journal of chemical information and modeling, 62(6):1376-1387, 2022.  
[13] Gabriel A Pinheiro, Johnatan Mucelini, Marinalva D Soares, Ronaldo C Prati, Juarez LF Da Silva, and Marcos G Quiles. Machine learning prediction of nine molecular properties based on the smiles representation of the qm9 quantum-chemistry dataset. The Journal of Physical Chemistry A, 124(47):9854–9866, 2020.  
[14] Daniel Probst, Philippe Schwaller, and Jean-Louis Reymond. Reaction classification and yield prediction using the differential reaction fingerprint drfp. Digital discovery, 1(2):91-97, 2022.  
[15] Philippe Schwaller, Teodoro Laino, Théophile Gaudin, Peter Bolgar, Christopher A Hunter, Costas Bekas, and Alpha A Lee. Molecular transformer: a model for uncertainty-calibrated chemical reaction prediction. ACS central science, 5(9):1572-1583, 2019.  
[16] Philippe Schwaller, Alain C Vaucher, Teodoro Laino, and Jean-Louis Reymond. Prediction of chemical reaction yields using deep learning. Machine learning: science and technology, 2(1):015016, 2021.  
[17] Teague Sterling and John J Irwin. Zinc 15-ligand discovery for everyone. Journal of chemical information and modeling, 55(11):2324-2337, 2015.  
[18] Mohammadamin Tavakoli, Pierre Baldi, Ann Marie Carlton, Yin Ting Chiu, Alexander Shmakov, and David Van Vranken. Ai for interpretable chemistry: Predicting radical mechanistic pathways via contrastive learning. Advances in Neural Information Processing Systems, 36, 2024.  
[19] Mohammadamin Tavakoli, Ryan J Miller, Mirana Claire Angel, Michael A Pfeiffer, Eugene S Gutman, Aaron D Mood, David Van Vranken, and Pierre Baldi. Pmechdb: A public database of elementary polar reaction steps. Journal of Chemical Information and Modeling, 2024.  
[20] Shuangjia Zheng, Jiahua Rao, Zhongyue Zhang, Jun Xu, and Yuedong Yang. Predicting retrosynthetic reactions using self-corrected transformer neural networks. Journal of chemical information and modeling, 60(1):47-55, 2019.  
[21] Ai-Nan Zhou, Bonan Li, Lejun Ruan, Yeting Wang, Gengli Duan, and Jianqi Li. An improved and practical route for the synthesis of enzalutamide and potential impurities study. Chinese Chemical Letters, 28(2):426-430, 2017.
