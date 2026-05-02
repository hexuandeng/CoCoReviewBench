# FOOLING EXPLANATIONS IN TEXT CLASSIFIERS

Anonymous authors

Paper under double-blind review

# ABSTRACT

State-of-the-art text classification models are becoming increasingly reliant on deep neural networks (DNNs). Due to their black-box nature, faithful and robust explanation methods need to accompany classifiers for deployment in real-life scenarios. However, it has been shown in vision applications that explanation methods are susceptible to local, imperceptible perturbations that can significantly alter the explanations without changing the predicted classes. We show here that the existence of such perturbations extends to text classifiers as well. Specifically, we introduce TEXTEXPLANATIONFOOLER (TEF), a novel explanation attack algorithm that alters text input samples imperceptibly so that the outcome of widely-used explanation methods changes considerably while leaving classifier predictions unchanged. We evaluate the performance of the attribution robustness estimation performance in TEF on five text classification datasets, utilizing three DNN architectures and a transformer architecture for each dataset. TEF can significantly decrease the correlation between unchanged and perturbed input attributions, which shows that all models and explanation methods are susceptible to TEF perturbations. Moreover, we evaluate how the perturbations transfer to other model architectures and attribution methods, and show that TEF perturbations are also effective in scenarios where the target model and explanation method are unknown. Finally, we introduce a semi-universal attack that is able to compute fast, computationally light perturbations with no knowledge of the attacked classifier nor explanation method. Overall, our work shows that explanations in text classifiers are very fragile and users need to carefully address their robustness before relying on them in critical applications.

# 1 INTRODUCTION

Deep neural networks (DNNs) have undoubtedly become the state-of-the-art architectures for many existing machine learning tasks (Choi et al., 2016). Yet, their black-box nature has raised the need for developing methods to mitigate the lack of interpretability caused by their increased complexity (Simonyan et al., 2013; Zeiler & Fergus, 2014; Hendricks et al., 2018; Bahdanau et al., 2014). These methods give intuitive, easily understandable explanations that do not require significant domain knowledge. This is especially desired in safety-critical scenarios, such as healthcare, where the users of such DNNs - the medical professionals for instance - need to understand the decision process and reasoning behind it. However, they have been shown to lack local robustness towards carefully crafted, imperceptible perturbations in the input (Ghorbani et al., 2019). While resulting in the same predictions, these altered inputs yield significantly different explanations and attributions maps (Figure 1). Interpretation methods fragile towards small input perturbations not only fail to provide faithful explanations, a desiderata commonly required in explainable AI Jacovi & Goldberg (2020), but also damages user trust in DNNs and prevents them from being deployed on high-stakes, safety-critical applications, such as in healthcare Adadi & Berrada (2020).

The previously described phenomenon has been widely studied in the image domain by Etmann et al. (2019); Moosavi-Dezfooli et al. (2019b) or Ivankay et al. (2020). However, in discrete-input domains like text, there has been limited progress. This is especially problematic given the increased reliance on and fragility of attention mechanisms as inherently explainable methods, as stated in Ghaeini et al. (2018). Therefore, we summarize our contributions as follows:

<table><tr><td>Original sample</td><td>TEF sample</td><td>PCC</td></tr><tr><td>[CLS] romanians pitch rumsfeld on base location — mihail kogalniceanu air base , romania - to entice the us military to make a home here , what better symbolic appeal could the romanian government make than to rename a street here quot;george washington boulevard ? [SEP]fθ(W, World) = 0.99</td><td>[CLS] romanians pitch clinton on base places — mihail kogalniceanu air base , rumania - to entice the us military to make a home here , what better symbolic appeal could the romanian government make than to rename a street here quot;george washington boulevard ? [SEP]fθ(W, World) = 0.97</td><td>-0.07</td></tr><tr><td>[CLS] unforgettable horror – more gory than psychological – with a highly satisfying quotient of friday - night excitement and milla power . [SEP]fθ(W, Pos.) = 0.99</td><td>[CLS] unforgettable horror – more gory than psychological – with a highly satisfying quotient of friday - night arousal and milla wattage . [SEP]fθ(W, Pos.) = 0.99</td><td>0.18</td></tr></table>

Figure 1: Example of fragile attributions. Highlighted red words are deemed most important towards the predicted class by the Integrated Gradients attribution method, blue ones against it. By substituting a few words in the original sample, the Pearson Correlation Coefficient (PCC) of word importances drops to below 0.2 while maintaining the same confidence in the correctly predicted class (denoted by  $f_{\theta}$ ).

- We provide a novel baseline black-box adversarial attack, TEXTPLANATIONFOOLER (TEF) to estimate the local robustness of explanations in text classification problems  
- We evaluate attribution robustness on widely used, state-of-the-art text datasets and model architectures, showing that explanation methods' output can be significantly altered with our attack  
- We provide insight into the transfer capability of TEF on different models and explanation methods as well as introduce semi-universal adversarial perturbations to alter explanations while not require access to the model at attack-time

The rest of the paper is structured as follows. Section 2 contains the preliminary background and related work relevant in this area. Section 3 describes our problem formulation, threat model and the attack used for altering explanations. Section 4 contains the evaluation of the attack of widely used datasets as well as the results and findings of our experiments. Section 5 provides conclusions and interesting future directions this work opens up.

# 2 PRELIMINARIES

# 2.1 RELATED WORK

Adversarial attacks that alter the inference outcomes in DNNs have been widely studied both in the image (Goodfellow et al., 2014; Carlini & Wagner, 2017; Moosavi-Dezfooli et al., 2016; Modas et al., 2019) and text domain (Ebrahimi et al., 2017; Sun et al., 2020; Jin et al., 2019; Yang et al., 2020). Methods to alleviate the networks susceptibility to such attacks have also been proposed, including the works of Madry et al. (2017); Moosavi-Dezfooli et al. (2019a); Buckman et al. (2018) or Cisse et al. (2017). However, it has recently been shown by authors Ghorbani et al. (2019) that, in addition to DNN predictions, widely-used explanation methods also lack robustness to targeted, imperceptible alterations of the input. These attacks change the outcomes of such explanation methods significantly, while predictions of the DNNs are unaltered. This violates the Prediction Assumption of faithful explanations and crucially degrades user trust in such explanation methods, as significantly different interpretations are provided for similar inputs and outputs (Jacovi & Goldberg, 2020). The aforementioned phenomenon of fragile explanations has mostly been investigated in the image domain (Etmann et al., 2019; Singh et al., 2019; Chen et al., 2019; Ivankay et al., 2020), with less focus on discrete input spaces like text. However, faithful and robust interpretations are arguably equally important in the discrete text domain as well, for instance in electronic health record classification (Girardi et al., 2018) or precision medicine (Binder et al., 2021), where critical

decisions often need to be based on DNN explanations. The work of La Malfa et al. (2021) constructs inherently robust explanations for NLP models, however only towards perturbations in the embedding space, not the input space that adversaries can operate on. Moreover, they do not give a method to evaluate robustness of already existing explanation algorithms. The authors of Feng et al. (2018) show that interpretation methods in NLP lack completeness (Sundararajan et al., 2017) by removing words deemed least important by explanation methods. Moreover, attention mechanisms Bahdanau et al. (2014) have been increasingly relied on as inherently interpretable systems. However recent work has questioned their faithfulness and plausibility (Serrano & Smith, 2019; Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) by proving that they often do not highlight input components that are most important to a DNN decision. We, paralleled by the very recent work of Sinha et al. (2021), are the first to show that imperceptible perturbations in the input space can alter the outcome of explanations of text classifiers significantly, giving an efficient attack to estimate explanation robustness. However, our work is the first to give an extensive evaluation of the robustness of widely-used explanation methods on large datasets, comparing several state-of-the-art architectures. Further, we are the first to address robustness of attention weights in transformer architectures and to provide insight on transfer capabilities and universal attacks.

# 2.2 BACKGROUND

Let  $\mathbb{S} = \{s_1, s_2, \dots, s_N\}$  be a dataset of  $N$  text samples  $s_i$ , each with a label from a predefined set of labels  $\mathbb{L} = \{l_1, l_2, \dots, l_{|\mathbb{L}|}\}$ . Each sample  $s_i$  contains a sequence of tokens (or words)  $w_i$  taken from a discrete vocabulary set  $\mathbb{W} = \{w_1, w_2, \dots, w_{|\mathbb{W}|}\}$ . A generic text classifier consists then of a non-injective, non-surjective embedding function  $E: \mathbb{S} \to \mathbb{R}^{d \times p}$ ,  $E(s) = X$ , which maps the input sample  $s$  to its embedding matrix  $X$ , and a function  $f: \mathbb{R}^{d \times p} \to \mathbb{R}^{|L|}$ ,  $f(X) = o$ , representing a (DNN) classifier function.  $d$  and  $p$  denote the embedding dimension and sequence length respectively. Let  $F: \mathbb{S} \to \mathbb{R}^{|L|}$ ,  $F(s) = f \circ E$  the be the full text classifier with final prediction  $y = \arg \max_{i \in \{1: |L|\}} o_i$ .

We define an attribution map as  $A: \mathbb{S} \to \mathbb{R}^p$ ,  $A(s, F, l) = a$  that assigns a scalar value to each input token  $w_i$  in the text sample  $s$ , resulting in the attribution vector  $a \in \mathbb{R}^p$ . This vector represents each token's influence towards the prediction outcome  $y$  of classifier  $F$ . Our work considers three widely-used attribution methods in text classification, namely Saliency Maps (SM) Simonyan et al. (2013), Integrated Gradients (IG) Sundararajan et al. (2017) and Attention (A) Bahdanau et al. (2014), defined in the following Equations (1), (2) and (3) respectively.

$$
A _ {i} ^ {\mathrm {S M}} (\boldsymbol {s}, F, l) = \sum_ {j \in \{1: d \}} | \nabla_ {\boldsymbol {X}} f (\boldsymbol {X}) _ {l} | _ {j, i} \tag {1}
$$

$$
A _ {i} ^ {\mathrm {I G}} (\boldsymbol {s}, F, l, \boldsymbol {B}) = \sum_ {j \in \{1: d \}} \left[ (\boldsymbol {X} - \boldsymbol {B}) \cdot \int_ {\alpha = 0} ^ {1} \nabla_ {\tilde {\boldsymbol {X}}} f (\tilde {\boldsymbol {X}}) _ {l} | _ {\tilde {\boldsymbol {X}} = \boldsymbol {B} + \alpha (\boldsymbol {X} - \boldsymbol {B})} d \alpha \right] _ {j, i} \tag {2}
$$

$$
A _ {i} ^ {\mathrm {A t t}} (\boldsymbol {s}, F, l) = \frac {\exp e _ {i}}{\sum_ {j \in \{1 : p \}} \exp e _ {j}} \tag {3}
$$

where  $\pmb{B}$  denotes the null matrix  $0^{d\times p}$ ,  $f$  is the classifier function of  $F$ ,  $\alpha$  a scaling factor and  $\pmb{X} = E(\pmb{s})$ .  $\nabla_{\pmb{X}}f$  denotes the matrix-derivative of  $f$  to  $\pmb{X}$ , as defined in Goodfellow et al. (2016). An attention head is a layer that transforms its inputs into scores  $\pmb{e}$  and calculates its output by linear combination of each input score, with coefficients normalized to a distribution. These coefficients are the attention weights  $A_i^{\mathrm{Att}}(s,F,l)$  denoted in Equation (3). It is commonly agreed to give intuitive explanations on how much the model attends to the given inputs through its attention weights Jacovi & Goldberg (2020).

# 3 METHODS

In this section, we describe our novel method TEXTEXPLANATIONFOOLER (TEF) to estimate attribution robustness (AR) in text classification problems. Specifically, we define the problem formulation, introduce our threat model and present the algorithm used to alter explanations by imperceptibly changing the inputs.

# 3.1 PROBLEM FORMULATION

Given an input text samples  $s$  and  $\tilde{s}$ , labels  $l$ ; a text classifier  $F$  with embedding function  $E$  and classifier function  $f$ ; and attribution method  $A$ , we define attribution robustness (also explanation robustness, AR) as written in Equation (4).

$$
r (\tilde {\boldsymbol {a}}, \boldsymbol {a}) = \max  _ {\tilde {\boldsymbol {a}}} d (\tilde {\boldsymbol {a}}, \boldsymbol {a}) = 1 - \max  _ {\tilde {\boldsymbol {W}}} d [ A (\tilde {\boldsymbol {s}}, F, l), A (\boldsymbol {s}, F, l) ] \tag {4}
$$

with

$$
\underset {i \in \{1: | \mathbb {L} | \}} {\arg \max } F (\tilde {\boldsymbol {s}}) = \underset {i \in \{1: | \mathbb {L} | \}} {\arg \max } F (\boldsymbol {s}), \tag {5}
$$

where  $d$  denotes a distance metric between the attributions  $\tilde{\pmb{a}}$  and  $\pmb{a}$  of the two input samples  $s$  and  $\tilde{s}$ . The rest of the notation is kept as in Chapter 2. Equation (4) quantifies how different the attributions of two input samples are, given the constraint in Equation (5) that enforces the inputs having the same prediction outcome.

The attribution robustness estimation is then solved utilizing the following Equation (6).

$$
\boldsymbol {s} _ {\mathrm {a d v}} = \underset {\tilde {\boldsymbol {s}}} {\arg \max } d \left[ A (\tilde {\boldsymbol {s}}, F, l), A (\boldsymbol {s}, F, l) \right] \tag {6}
$$

where  $s_{\mathrm{adv}}$  denotes the solution to the estimation, i.e. the adversarial input.  $s$  denotes the original, unperturbed input and  $\tilde{s}$  the perturbed input, optimized during estimation. In other words, we modify the input sequence  $s$  such that the distance between the attributions maps of original and modified inputs is maximized and their prediction unchanged. The search space is further constraint as described in the next section, such that  $s_{\mathrm{adv}}$  and  $s$  are semantically close. This problem formulation is motivated by the definition of AR in Equation (4), inducing the vulnerability that similar inputs with the same outputs give considerably different explanations. This is further backed by current research (Ghorbani et al., 2019; Dombrowski et al., 2019; Ivankay et al., 2020) and the Prediction Assumption of faithful explanations (Jacovi & Goldberg, 2020).

# 3.2 THREAT MODEL AND ATTACK

We define our algorithm to estimate AR as a black-box attack. It only queries the model to obtain its output logits and the accompanied explanations of the inference process. The model might access its gradients to compute explanations, but the attack only utilizes the resulting explanations, no gradient or architectural information. We restrict the valid input perturbations to token substitutions, specifically insertions and deletions of tokens are forbidden, as they alter the input lengths. Algorithm 1 contains the schematic code for TEF, consisting of the following two steps.

Step 1 - Word importance ranking First, an importance ranking is extracted for each token of the input sample. Specifically, we compute  $I_{w_i} = d[A(s_{w_i \to 0}, F, l), A(s, F, l)]$  for each token  $i$  in  $s$ , where  $s_{w_i \to 0}$  denotes the input sequence  $s$  with the  $i$ -th word masked to the zero embedding token. The input tokens are then sorted by the  $I_{w_i}$  values in a decreasing fashion. Then, high importance words are prioritized during substitution.

Algorithm 1 TextExplanationFollower (TEF)

Input: Input sentence  $s$  with predicted class  $l$ , classifier  $F$ , attribution  $A$ , attribution distance  $d$ , number of synonyms  $N$ , maximum perturbation ratio  $\rho$

Output: Adversarial sentence  $s_{\mathrm{adv}}$

1:  $s_{\mathrm{adv}} \gets s, d_{max} \gets 0, r \gets 0$  
2: for  $w_{i}\in s$  do  
3:  $I_{w_i} = d\big[A(\pmb{s}_{w_i \to 0}, F, l), A(\pmb{s}, F, l)\big]$  
4: for  $w_{j} \in \langle w_{1}, \dots, w_{|\mathbf{s}|} \rangle \mid I_{w_{m-1}} \geq I_{w_{m}} \forall m \in \{2, \dots, |\mathbf{s}|\}$  do  
5: if  $w_{j}\in \mathbb{S}_{\mathrm{Stopwords}}$  then  
6: continue  
7:  $\mathbb{C}_j\gets$  SynonymEmbeddings  $(w_{j},N)$  
8:  $\mathbb{C}_j\gets \mathrm{POSFilter}(w_j,\mathbb{C}_j,s)$  
9: for  $c_{k}\in \mathbb{C}_{j}$  do  
10:  $\tilde{s}_{w_j\to c_k}\gets$  Replace token  $w_{j}$  in  $s_{\mathrm{adv}}$  with  $c_{k}$  
11: if arg max  $F(\tilde{s}_{w_j\to c_k}) = l$  then  $i\in \{1:|\mathbb{L}|\}$  
12:  $\tilde{d} \gets d\left[A\left(\tilde{s}_{w_i \to c_k}, F, l\right), A\left(s, F, l\right)\right]$  
13: if  $\tilde{d} > d_{\text{max}}$  then  
14:  $s_{\mathrm{adv}} \gets \tilde{s}_{w_i \to c_k}$  
15:  $d_{max}\gets \dot{d}$  
16:  $r\gets r + 1$  
17: if  $\frac{r}{|s|} + 1 > \rho$  then  
18: break

Table 1: Average text length, number of classes and accuracies of our models train on the five datasets.  

<table><tr><td></td><td>CNN</td><td>LSTM</td><td>LSTMAntt</td><td>BERT</td><td>Avg. text length</td><td>Num. classes</td></tr><tr><td>AG&#x27;s News</td><td>89.7%</td><td>90.8%</td><td>91.4%</td><td>94.2%</td><td>45</td><td>4</td></tr><tr><td>IMDB</td><td>82.0%</td><td>87.2%</td><td>87.3%</td><td>89.4%</td><td>270</td><td>2</td></tr><tr><td>Fake News</td><td>98.9%</td><td>99.6%</td><td>99.6%</td><td>99.8%</td><td>919</td><td>2</td></tr><tr><td>MR</td><td>73.0%</td><td>76.4%</td><td>78.0%</td><td>82.2%</td><td>22</td><td>2</td></tr><tr><td>Yelp</td><td>49.0%</td><td>54.8%</td><td>60.0%</td><td>62.6%</td><td>159</td><td>5</td></tr></table>

Step 2 - Candidate selection For each word  $w_{i}$  in  $s$  sequentially, a set of substitution candidates  $\mathbb{C}$  of  $N$  elements is extracted. This candidate set is constructed from the counter-fitted GloVe (Pennington et al., 2014) synonym embeddings by the authors of Mrkšić et al. (2016). The candidates are then filtered by Part-Of-Speech (POS), tagged by SpaCy (Honnibal et al., 2020), only allowing replacements with equal POS. Stop words are also discarded from  $\mathbb{C}$ . A final selection as replacement for  $w_{i}$  is then made to be the  $c_{k} \in \mathbb{C}$  that maximizes  $d\big[A(\tilde{s}_{w_i \to c_k}, F, l), A(s, F, l)\big]$ . The algorithm is aborted when the number of replacements to sentence length exceeds the maximum value  $\rho$ .

# 4 EXPERIMENTS AND RESULTS

In this section, we present an extensive evaluation of our AR estimation attack, TEF, for text classification problems. We detail the datasets, architectures and evaluation metrics used in our experiments. We examine the performance of TEF and study the impact of different factors on its robustness evaluation performance. We detail the choice of parameters and the related trade-offs. Moreover, we describe our transfer and semi-universal attacks and examine their robustness estimation performance.

# 4.1 MODELS, DATASETS AND EVALUATION

Our attack is evaluated on five commonly used public text classification datasets, AG's News (Zhang et al., 2015), MR reviews (Zhang et al., 2015), IMDB Movie Reviews (Maas et al., 2011), Fake News Dataset  ${}^{1}$  and Yelp (Asghar, 2016). We train four different word embedding-based architectures for each dataset, namely a CNN, an LSTM, an LSTM containing a single attention layer with one head (LSTMAtt) and a finetuned BERT architecture. Table 1 contains a summary of our model performances as well as details on the datasets. The text samples are tokenized with the default English SpaCy (Honnibal et al., 2020) tokenizer for the CNN, LSTM and LSTMAtt models and embedded with the pretrained GloVe 6B 300-dimensional word vectors (Pennington et al., 2014). The BERT models use the BERT tokenizer and it's built-in positional embeddings (Vaswani et al., 2017). We use PyTorch (Paszke et al., 2019) with Captum (Kokhlikyan et al., 2020) to implement our models and explainers and the Huggingface Transformers library (Wolf et al., 2020) to finetune BERT on our datasets.

We evaluate the robustness of three different, commonly used explanation methods in natural language processing with our TEF attack. These are Saliency Maps (SM), Integrated Gradients (IG) and the Attention mechanism (A). The definitions are given in Section 2. We use SM, IG in combination with our CNN, LSTM and LSTMAtt models, Attention only with LSTMAtt and BERT.

During the attack, we set  $d$  of Equation (6) to be  $d(\tilde{a}, a) = 1 - \frac{\mathrm{PCC}(\tilde{a}, a) + 1}{2}$ , with PCC denoting the Pearson Correlation Coefficient (Pearson, 1895) of original and adversarial attributions  $\tilde{a}$  and  $\tilde{a}, a$ . We then report the standard Pearson Correlation Coefficient (PCC), Kendall's Rank Order Correlation (ROC) and the Top-10%, Top-30% and Top-50% intersections to measure AR in Equation 4. These are common metrics that are agreed to correspond to human measures of AR (Ghorbani et al., 2019; Dombrowski et al., 2019; Ivankay et al., 2020). Additionally, to quantify imperceptibility of perturbations, the semantic similarity of adversariably perturbed and unchanged

![](images/1333457ea144f7776a7899b31d77e42984a73aacd9fba5bcd6005d10ca33fe85.jpg)  
CNN - Saliency Maps (SM) on AG's News

![](images/b7088f20250296581ccc172a3c6535408cb35631b69535c929124942d9ade138.jpg)

![](images/b7d1ea05673fbaee5d538d48f9b77fb7d681d22cdb67d0b0603569582a2fdcf9.jpg)  
LSTMAtt - Integrated Gradients (IG) on IMDB

![](images/43d6799185e65eee5a2c22298eeee06ae6955207bb3d5fa0dc0d9c6eb378a000.jpg)

![](images/94e6387ffda3f8f9c5d70eccf565c57dbc2034097fdfd42116b76aaa558f2cb8.jpg)  
BERT - Attention (A) on Yelp

![](images/422d39960f637c7118716583ccd28fc25950cf18cd023590d1f12b35fce2ccd0.jpg)  
Figure 2: Robustness of attribution maps on several architectures and explainers. We plot the average PCC, Kendall's Rank Order Correlation - ROC - and semantic similarity - Sem. - (left) as well as Top-10%, Top-30% and Top50% intersections (right) of the attribution vectors as functions of the parameter  $\rho$ . Dashed lines indicate the same metrics for our RANDOMATTACK (RA) TEF. The ACC indicates the area under the PCC curve, lower values correspond to overall lower feature attribution correlations in the overall operation interval of  $\rho$ .

sentences is reported. Semantic similarity (Sem.) is measured by the cosine distance between the embeddings produced by the Universal Sentence Encoder (USE) Cer et al. (2018). This is a state-of-the-art sentence embedding widely used in adversarial attacks on text (Sun et al., 2020; Jin et al., 2019).

Due to the lack of related work in this field, we compare the AR estimation performance of TEF to our RANDOMATTACK (RA) baseline. RA utilizes a random word importance ranking in Step 1 of TEF and selects a random synonym in the final selection in Step 2. POS and stop word filters are still utilized in RA to keep linguistic constraints intact.

# 4.2 ROBUSTNESS OF EXPLANATIONS

Performance evaluation of TEF. In order to evaluate the performance of our attack, we vary the parameter  $\rho$ , which denotes the maximum ratio of perturbed words in the input sample. A

![](images/923ab701743d500f9e8f6a97d75efa7c56ff137d200cc7f7418181cb66672cfd.jpg)  
Avg. PCC of LSTMAtt - A on AG's News

![](images/9eeed62819575db1f38117a2874b835ae37bbcd4cb4d7b8b986a747f250aa6c0.jpg)  
Figure 3: Ablation study of TEF. We compare the PCC of TEF, RA, the RANDOMIMPORTANCE (RI) attack and the RANDOMSYNONYM (RS) attack. We find that RI behaves slightly worse than TEF, while RS behaves slightly better than RA over all  $\rho$  values.  
Avg. PCC of LSTM - IG on MR

larger  $\rho$  value leads to lower attribution correlation, as potentially more words are substituted in the input. We then capture the aforementioned metrics PCC, ROC, Sem. and Top-10%/30%/50% intersections to evaluate AR. Additionally, to quantify performance of our attack over the whole operation interval of  $0 \leq \rho \leq 0.5$ , we compute the Area under the Pearson Correlation Curve (ACC). A lower value of ACC corresponds to lower robustness overall, as correlation values are lower. We note that a particular value of  $\rho$  does not guarantee that all input samples have exactly  $\rho$  ratio of perturbed tokens. Therefore, we quantize our samples based on their perturbed ratio such that samples with approximately the same  $\rho$  are grouped together and the mean and standard deviation of PCCs are reported. These bins are computed per dataset, ensuring the comparability of resulting curves and ACCs for each plot. Moreover, we chose the number of candidates in Step 2 of TEF to be  $N = |\mathbb{C}| = 15$ , as it is a good trade-off between TEF estimation performance and attack run time. We find that TEF is able to significantly outperform the baseline provided by RA in terms of all AR metrics, on all datasets, models and explanation methods considered in this work. A subset of these results is shown in Figure 2. Moreover, we found the self-attention mechanism of BERT to be considerably more robust to perturbations than non-transformer-based architectures and explanations.

Ablation study. In addition to the fully random attack described in the previous paragraph, we compare TEF to our semi-random attacks RANDOMIMPORTANCE (RI) and RANDOMSYNONYM (RS). During these, we only randomize the word importance ranking of TEF (RI) but keep the selection of best final synonym, as well as randomize the final synonym selection of TEF (RS) but keep the word importance ranking respectively. Figure 3 shows our findings for these experiments, along with comparisons to RANDOMATTACK (RA). Based on the curves and ACCs shown in this figure, we conclude that RI consistently outperforms RS in terms of PCC over the whole operation interval of  $\rho$ . Moreover, the impact of word importance ranking diminishes with increasing  $\rho$ , especially for shorter datasets like MR. This can be observed by the RS curve coming closer to RA for high  $\rho$  values.

BERT's attention layers and heads. BERT's attention weights can be used to help gain insight into a models prediction by understanding which parts of the input are most attended to (Vig, 2019). The attention weights of each layer and head are visualized and used for debugging the inference process, understanding what patterns the heads attend to. Our BERT models have 12 layers with 12 attention heads (144 heads in total), each producing a distribution of attention weights over its inputs and outputs. Estimating the AR of all heads together is not useful, as more and less robust heads would average out. Therefore, we run TEF to estimate the robustness of each head separately. Figure 4 contains the average PCCs of the attention weights before and after perturbing the inputs with TEF. We find that attention weights in later layers tend to be more susceptible to input perturbations than earlier layers. Moreover, heads within a layer tend to be comparably robust. Similar results were obtained for the Fake News and MR datasets. We leave a thorough, theoretical analysis of this phenomenon to future work. We conclude that the increasing reliance on attention weights

![](images/3bd822879a39ed5faf56615c295c7366eb0ecd746e45f8ee918fbda2867d4458.jpg)  
Figure 4: Estimated robustness of BERT attention weights on different layers (Y-axis) and heads (X-axis) for  $\rho = 0.2$ . Red cells indicate average PCC values close to -1, before and after TEF perturbations, hence less robust attention head weights, while white cells have average PCCs close to 1. Attention heads in later layers tend to be less robust, while heads within a layer seem equally robust in most layers.

![](images/03b93c0d51c40126db77ce02ccc629f84196bb3b6fd3a94d7f81ef30a37e3ef8.jpg)

![](images/5570d71b95d1ea71672c4c44d948050db36e8e706c04ebb33a0698bea2a0925f.jpg)

to provide inherent interpretations to BERT predictions needs careful investigation, especially in safety-critical applications.

# 4.3 TRANSFERABILITY AND SEMI-UNIVERSAL PERTURBATIONS

In this section, we examine the transferability of input perturbations computed by TEF. Analogously to transferability in traditional adversarial setups Demontis et al. (2019), we evaluate how our perturbations transfer to different classifier models. In addition, we examine the transfer capabilities to other explanation methods as well. Finally, we introduce our semi-universal input perturbation policy, derived from TEF perturbations, which is able to alter explanations without querying the model for predictions or attributions during attack time.

Transferability of perturbations to models and explanations. Transfer attacks can prove useful to potentially unknown models and explanation producing systems. The adversary does not necessarily possess information about what kind of model is deployed, nor the exact method to produce the accompanying explanations. Therefore, it is crucial for systems to be as resistant to transfer attacks as possible in order to evade perturbations constructed on similar models and explanations.

Thus, we examine how our classifiers and attribution methods react to transfer attacks computed by TEF. We alter the input samples for a given model and explanation method, then evaluate the perturbation of the same samples on a different classification method, utilizing the same attribution method. Moreover, we repeat this process and keep the classification model

the same, but utilize a different attribution method. In this manner, we can quantify the transfer performance of TEF. The results are found in Figure 5, compared to TEF and RA perturbations. We observe that transfer attacks consistently perform better than RA, some even by approx. 0.4 in terms of average PCC decrease in the operation area of  $\rho \approx 0.1$ . However, as expected, they significantly fall short to the performance of TEF on all datasets and models. Therefore, we conclude that transferring TEF perturbations effectively highlights fragility of explanations, but TEF provides tighter AR bounds.

Semi-universal perturbations. In this section, we take a step towards defining universal perturbations for text explanations. Similarly to the work of the authors of Moosavi-Dezfooli et al. (2017) and Gao & Oates (2019), who created universal attacks in the prediction of image and text classifiers. These provide fast and computationally cheap perturbations during at

![](images/076da6c50cffa789af28ca8ef0cedaad19d980d8087468dee631ee6dfbf594af.jpg)  
Avg. PCC of LSTMAtt - IG on IMDB  
Figure 5: Transfer capabilities of TEF to other models and explanation methods. The lines indicate the estimated PCC of TEF perturbations transferred from the indicated models and explanations.

tack time that are able to mislead classifiers with pre-computed perturbations. However, instead of calculating one single perturbation that is added to each input sample, we construct a priority list of possible perturbations by aggregating TEF perturbations. For each dataset, we split the test dataset into two equally sized parts, the attack set and the evaluation set.

Next, we compute the optimal TEF perturbations for each sample in the attack set, for each model and attribution method. Then, for each token in the attack set, we list how often it was replaced by TEF and what its most frequent replacement was. This list then sorted decreasingly by replacement frequency, resulting in the most often replaced tokens and its most frequent replacement at the top, the most rarely replaced ones at the bottom. We denote such lists by semi-universal attack policies. An example policy can be found in Figure 6 (above). During attack time, the tokens in these policies are iterated from top to bottom, and each occurrence of the token in the attacked sample is replaced by its most frequent replacement, extracted from the policy, if given  $\rho$  has not been reached and the POS remains the same. In such way, perturbed inputs are created without querying the model at all at attack time.

However, in scenarios where the target model and explanation method are unknown, it is not guaranteed that during construction of such policies, TEF perturbations of the target model are included. Therefore, in order to test the generalization capabilities of our semi-universal attack, we also construct specific policies, excluding TEF perturbations for the given model and attribution. For instance, when evaluating the performance on the universal attack on our CNN architecture with IG, we only consider TEF perturbations from LSTM, LSTMAtt and BERT architectures, with explainers SM and A. We evaluate our attacks on the evaluation set, the results can be found in Figure 6 (below). We conclude that such semi-universal policies are highly effective in reducing attribution cor

relation when the adversary has no access to the target model and explanation method.

IMDB  

<table><tr><td>Token</td><td>Repl. #</td><td>Replacement</td></tr><tr><td>movie</td><td>430k</td><td>cinematographic</td></tr><tr><td>film</td><td>338k</td><td>cine</td></tr><tr><td>good</td><td>122k</td><td>decent</td></tr><tr><td>great</td><td>103k</td><td>whopping</td></tr><tr><td>bad</td><td>102k</td><td>wicked</td></tr><tr><td>...</td><td>...</td><td>...</td></tr><tr><td>amazing</td><td>17.1k</td><td>staggering</td></tr><tr><td>...</td><td>...</td><td>...</td></tr><tr><td>scary</td><td>6.8k</td><td>fearful</td></tr></table>

![](images/d62e06fe493ab0f77eb1539bbdd8af3f5264b280d773037aca1f9a9398acac8e.jpg)  
IMDB with LSTM - IG  
Figure 6: Semi-universal attack policy of the IMDB dataset (above). The most commonly replaced tokens and its most frequent replacement (Replacement) are sorted by their number of replacements in the attack set (Repl. #) (above). Average PCC of original and adversarial attributions of IG on the LSTM model (below). The estimation performance of the semi-universal attack (Uni.) is compared to TEF, RA and the model-specific semi-universal attack (Uni. Spec.). The universal attacks perform better than RA by up to 0.5 in avg. PCC increase, especially for high  $\rho$  values.

# 5 CONCLUSION

In this work, we introduced a novel black-box attack called TEXTEXPLANATIONFOOLER, that successfully perturbs input data such that the outcome of popular explanation methods in text classification, but not the prediction of the classifier. This attack provides a baseline estimator for attribution robustness and highlights the lack of robustness of current explanation methods. We compared it to the random attack, showing its superior performance to it on five different, widely used text classification datasets. Moreover, the transfer capabilities of the attack are evaluated. Finally, we showed the existence of semi-universal perturbation policies that are capable of altering explanations without querying the model during attack-time, even without having access to perturbations for those models. This work opens up many interesting research directions. In future work, we plan to examine whether a similar white-box attack that has access to model gradients can improve robustness estimation. Moreover, instead of synonym embeddings, we plan to use BERT-based masked language models to extract possible candidates, further improving imperceptible word substitutions.

# REFERENCES

Amina Adadi and Mohammed Berrada. Explainable ai for healthcare: from black box to interpretable models. In Embedded Systems and Artificial Intelligence, pp. 327-337. Springer, 2020.  
Nabiha Asghar. Yelp dataset challenge: Review rating prediction. arXiv preprint arXiv:1605.05362, 2016.  
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.  
Alexander Binder, Michael Bockmayr, Miriam Hagele, Stephan Wienert, Daniel Heim, Katharina Hellweg, Masaru Ishii, Albrecht Stenzinger, Andreas Hocke, Carsten Denkert, et al. Morphological and molecular breast cancer profiling through explainable machine learning. Nature Machine Intelligence, 3(4):355-366, 2021.  
Jacob Buckman, Aurko Roy, Colin Raffel, and Ian Goodfellow. Thermometer encoding: One hot way to resist adversarial examples. In International Conference on Learning Representations, 2018.  
Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017, IEEE symposium on security and privacy (sp), pp. 39-57. IEEE, 2017.  
Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, et al. Universal sentence encoder. arXiv preprint arXiv:1803.11175, 2018.  
Jiefeng Chen, Xi Wu, Vaibhav Rastogi, Yingyu Liang, and Somesh Jha. Robust Attribution Regularization. In Advances in Neural Information Processing Systems, pp. 14300-14310, 2019.  
Edward Choi, Mohammad Taha Bahadori, Joshua A Kulas, Andy Schuetz, Walter F Stewart, and Jimeng Sun. Retain: An interpretable predictive model for healthcare using reverse time attention mechanism. arXiv preprint arXiv:1608.05745, 2016.  
Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, and Nicolas Usunier. Parseval networks: Improving robustness to adversarial examples. In International Conference on Machine Learning, pp. 854-863. PMLR, 2017.  
Ambra Demontis, Marco Melis, Maura Pintor, Matthew Jagielski, Battista Biggio, Alina Oprea, Cristina Nita-Rotaru, and Fabio Roli. Why do adversarial attacks transfer? explaining transferability of evasion and poisoning attacks. In 28th {USENIX} Security Symposium ( {USENIX} Security 19), pp. 321-338, 2019.  
Ann-Kathrin Dombrowski, Maximillian Alber, Christopher Anders, Marcel Ackermann, Klaus-Robert Müller, and Pan Kessel. Explanations can be manipulated and geometry is to blame. In Advances in Neural Information Processing Systems, pp. 13589-13600, 2019.  
Javid Ebrahimi, Anyi Rao, Daniel Lowd, and Dejing Dou. Hotflip: White-box adversarial examples for text classification. arXiv preprint arXiv:1712.06751, 2017.  
Christian Etmann, Sebastian Lunz, Peter Maass, and Carola-Bibiane Schonlieb. On the connection between adversarial robustness and saliency map interpretability. arXiv preprint arXiv:1905.04172, 2019.  
Shi Feng, Eric Wallace, Alvin Grissom II, Mohit Iyyer, Pedro Rodriguez, and Jordan Boyd-Graber. Pathologies of neural models make interpretations difficult. arXiv preprint arXiv:1804.07781, 2018.  
Hang Gao and Tim Oates. Universal adversarial perturbation for text classification. arXiv preprint arXiv:1910.04618, 2019.  
Reza Ghaeini, Xiaoli Z Fern, and Prasad Tadepalli. Interpreting recurrent and attention-based neural models: a case study on natural language inference. arXiv preprint arXiv:1808.03894, 2018.

Amirata Ghorbani, Abubakar Abid, and James Zou. Interpretation of neural networks is fragile. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 3681-3688, 2019.  
Ivan Girardi, Pengfei Ji, An-phi Nguyen, Nora Hollenstein, Adam Ivankay, Lorenz Kuhn, Chiara Marchiori, and Ce Zhang. Patient risk assessment and warning symptom detection using deep attention-based neural networks. arXiv preprint arXiv:1809.10804, 2018.  
Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016.  
Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.  
Lisa Anne Hendricks, Ronghang Hu, Trevor Darrell, and Zeynep Akata. Generating counterfactual explanations with natural language. arXiv preprint arXiv:1806.09809, 2018.  
Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. spaCy: Industrial-strength Natural Language Processing in Python, 2020. URL https://doi.org/10.5281/ zenodo.1212303.  
Adam Ivankay, Ivan Girardi, Chiara Marchiori, and Pascal Frossard. Far: A general framework for attributional robustness. arXiv preprint arXiv:2010.07393, 2020.  
Alon Jacovi and Yoav Goldberg. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? arXiv preprint arXiv:2004.03685, 2020.  
Sarthak Jain and Byron C Wallace. Attention is not explanation. arXiv preprint arXiv:1902.10186, 2019.  
Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits. Is bert really robust? natural language attack on text classification and entailment. arXiv preprint arXiv:1907.11932, 2, 2019.  
Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, et al. Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896, 2020.  
Emanuele La Malfa, Agnieszka Zbrzezny, Rhiannon Michelmore, Nicola Paoletti, and Marta Kwiatkowska. On guaranteed optimal robust explanations for nlp models. arXiv preprint arXiv:2105.03640, 2021.  
Andrew Maas, Raymond E Daly, Peter T Pham, Dan Huang, Andrew Y Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies, pp. 142-150, 2011.  
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017.  
Apostolos Modas, Seyed-Mohsen Moosavi-Dezfooli, and Pascal Frossard. Sparsefool: a few pixels make a big difference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9087-9096, 2019.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, and Pascal Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2574-2582, 2016.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1765-1773, 2017.  
Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal Frossard. Robustness via curvature regularization, and vice versa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9078-9086, 2019a.

Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Jonathan Uesato, and Pascal Frossard. Robustness via curvature regularization, and vice versa. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 9078-9086, 2019b.  
Nikola Mrkšić, Diarmuid O Seaghdha, Blaise Thomson, Milica Gašić, Lina Rojas-Barahona, Pei-Hao Su, David Vandyke, Tsung-Hsien Wen, and Steve Young. Counter-fitting word vectors to linguistic constraints. arXiv preprint arXiv:1603.00892, 2016.  
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, pp. 8026-8037, 2019.  
Karl Pearson. Notes on regression and inheritance in the case of two parents. Proceedings of the Royal Society of London, 58(347-352):240-242, 1895.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532-1543, 2014.  
Sofia Serrano and Noah A Smith. Is attention interpretable? arXiv preprint arXiv:1906.03731, 2019.  
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.  
Mayank Singh, Nupur Kumari, Puneet Mangla, Abhishek Sinha, Vineeth N Balasubramanian, and Balaji Krishnamurthy. On the benefits of attributional robustness. arXiv preprint arXiv:1911.13073, 2019.  
Sanchit Sinha, Hanjie Chen, Arshdeep Sekhon, Yangfeng Ji, and Yanjun Qi. Perturbing inputs for fragile interpretations in deep natural language processing. arXiv preprint arXiv:2108.04990, 2021.  
Lichao Sun, Kazuma Hashimoto, Wenpeng Yin, Akari Asai, Jia Li, Philip Yu, and Caiming Xiong. Adv-bert: Bert is not robust on misspellings! generating nature adversarial samples on bert. arXiv preprint arXiv:2003.04985, 2020.  
Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pp. 3319-3328, 2017.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pp. 5998-6008, 2017.  
Jesse Vig. Bertviz: A tool for visualizing multihead self-attention in the bert model. In ICLR Workshop: Debugging Machine Learning Models, 2019.  
Sarah Wiegrefe and Yuval Pinter. Attention is not not explanation. arXiv preprint arXiv:1908.04626, 2019.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38-45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.  
Puyudi Yang, Jianbo Chen, Cho-Jui Hsieh, Jane-Ling Wang, and Michael I Jordan. Greedy attack and gumbel attack: Generating adversarial examples for discrete data. J. Mach. Learn. Res., 21 (43):1-36, 2020.

Matthew Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European Conference on Computer Vision, pp. 818-833. Springer, 2014.  
Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In NIPS, 2015.