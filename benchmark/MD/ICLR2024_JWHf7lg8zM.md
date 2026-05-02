# MULTICONTRIEVERS: ANALYSIS OF DENSE RETRIEVAL REPRESENTATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Dense Retrievers compress source documents into vector representations; the information they encode determines what is available to downstream tasks (e.g., QA, summarisation). Yet there is little analysis of the information in retriever representations. We conduct the first analysis comparing the information captured in dense retriever representations as compared to language model representations. To do this analysis, we present MultiContrievers, 25 contrastive dense retrievers initialized from the 25 MultiBerts. We use information theoretic probing to analyse how well MultiContrievers encode two example pieces of information: topic and demographic gender (measured as extractability of these two concepts), and we correlate this to performance on 14 retrieval datasets covering seven distinct retrieval tasks. We find that: 1) MultiContriever contrastive training increases extractability of both topic and gender, but also has a regularisation effect; MultiContrievers are more similar to each other than MultiBerts, 2) extractability of both topic and gender correlate poorly with benchmark performance, revealing a gap between the effect of the training objective on representations, and desirable qualities for the benchmark 3) MultiContriever representations show strong potential for gender bias, and we do find allocational gender bias in retrieval benchmarks. However, a causal analysis shows that the source of the gender bias is not in the representations, suggesting that despite this potential, current gender bias is coming from either the queries or retrieval corpus, and cannot be corrected by improvements to modelling alone. We additionally find 4) significant variability across random seeds, suggesting that future work should test across a broad spread, which is not currently standard. We release our 25 MultiContrievers (including intermediate checkpoints) and all code to facilitate further analysis. $^{1}$

# 1 INTRODUCTION

Dense retrievers (Karpukhin et al., 2020; Izacard et al., 2022; Hofstätter et al., 2021) are the standard retrieval component of retrieval augmented Question Answering (QA) and other retrieval systems, such as fact-checking (Thorne et al., 2018), argumentation, and others. Yet there has been no analysis of the information present in dense retriever representations, nor how it affects retrieval system behaviour. This lack of analytical work is surprising. Retrievers are widespread, and are used for many purposes that require trust: to increase factuality and decrease hallucination (Shuster et al., 2021), and to provide trust and transparency (Lewis et al., 2020) via a source document that has provenance and can be examined. The information the a representation retains from a source document mediates these abilities. Dense retrievers compress inputs documents into N-dimensional representations, so they necessarily emphasise some pieces of information over others. Yet we still do not know what information is de- or over-emphasised, and how this affects retrieval behaviour. For example, a biography of Mary Somerville will contain many details about her: her profession (astronomy and mathematics), her gender (female), her political influence (women's suffrage), her country of origin (Scotland) and others. These will be relevant to different kinds of queries: which ones will be more and less emphasised after compression?

Some of this type of analysis exists for Masked Language Models (MLMs): Lovering et al. (2021) look at linguistic information that is retained (such as subject verb agreement, for the task of gram-

maticality judgments) and when lack of this information results in reliance on spurious heuristics, and Orgad et al. (2022) look at gender information and the effect on gender bias in profession classification. But there is no such analysis for retrievers, which optimise a very different objective. We propose to extend this previous analytical work into the retrieval domain. This leads us to the following research questions:

RQ1 What information does a retriever represent, and how does this differ from a language model?

RQ2 Do differences in this information correlate with performance on retrieval benchmarks?

RQ3 Is gender information in retrievers predictive of their gender bias?

To answer these questions, we train 25 MultiContrievers initialised from the released MultiBerts (Sellam et al., 2022). We then use information theoretic probing, also known as minimum description length (MDL) probing (Voita & Titov, 2020) to measure the information in MultiContriever representations. We evaluate the models on 14 retrieval datasets from the BEIR benchmark Thakur et al. (2021). We find that:

(RQ1) For both MultiBerts and MultiContrievers, gender is more extractable than topic, but there are noticeable differences in the models. MultiContrievers have both more extractable topic and more extractable gender, but a lower ratio between the two. MultiContrievers have overall richer representations for topic and gender, but they still have potential to rely on gender heuristics (which are a source of gender bias). MultiContrievers also have a smaller range of extractability between the 25 seeds, showing a regularisation effect of their training.

(RQ2) Despite the increase in extractability of both gender and topic, neither correlates with retrieval performance on benchmark datasets. This highlights the gap between features encouraged by the retriever training process, and those tested by the benchmarks, and our findings suggest it is a limitation of the benchmark. When we subsample a set of questions that require gender information to answer them correctly, we do see a correlation between gender extractability and performance. This indicates that gender information is used by the model, but that most questions in the benchmark can be answered without it. Retrieval benchmarks may underspecify some desirable characteristics of a good model.

(RQ3), despite the evidence that extractability of gender information is helpful to the model, it is not the cause of allocational gender bias in the Natural Questions (NQ) dataset. We find that when we do a causal analysis by removing gender from MultiContiever representations, gender bias persists, indicating that the source of bias is in the queries or corpus.

Our contributions are: 1) the first information theoretic analysis of dense retrievers for performance and for fairness, 2) the first causal analysis of social bias in dense retrievers, to identify the source of the bias, 3) a broad analysis of variability in performance and fairness across random retriever seeds, 4) a suite of 25 MultiContrievers for use in future work, as well as a small gender annotated subset of Natural Questions, and all training and evaluation code.

# 2 BACKGROUND AND RELATED WORK

# 2.1 WHAT IS A RETRIEVER?

Retrievers take an input query and return relevance scores for documents from a corpus. We use the common dense retrieval approach where documents  $D$  and queries  $Q$  are encoded separately by the same model  $f_{\theta}$ , and relevance is given by the dot product between them. For a given query  $q_{i}$  and document  $d_{i}$ , the relevance score  $s$  is:

$$
s \left(d _ {i}, q _ {i}\right) = f _ {\theta} \left(q _ {i}\right) \cdot f _ {\theta} \left(d _ {i}\right) \tag {1}
$$

Training  $f_{\theta}$  is a challenge. Language models like BERT (Devlin et al., 2019), are not good retrievers out-of-the-box, but retrieval training resources are limited and expensive to create, since they involve matching candidate documents to a query from a corpus of potentially millions. So retrievers are either trained on one of the few corpora available, such as Natural Questions (NQ) (Kwiatkowski et al., 2019) or MS MARCO (Campos et al., 2016) as supervision (Hofstätter et al.,

2021; Karpukhin et al., 2020), or on a self-supervised proxy for the retrieval task (Izacard et al., 2022). Both approaches results in a domain shift between training and later inference, making retrieval a generalisation task. This motivates our analysis, as Lovering et al. (2021)'s work showed that information theoretic probing was predictive of where a model would generalise and where it would rely on simple heuristics and dataset artifacts.

In this work, we focus on the self-supervised Contriever (Izacard et al., 2022), initialised from a BERT model and then fine-tuned with a contrastive objective. For this objective, all documents in a large corpus are broken into chunks, where chunks from the same document are positive pairs and chunks from different documents are negative pairs. This is a loose proxy for "relevance" in the retrieval sense, so we are interested in what information this objective encourages contriever to emphasise, what to retain, and what to lose, and what this means for the eventual retrieval task.

# 2.2 HOW DO WE FIND OUT WHAT INFORMATION IS IN A RETRIEVER?

The most common analysis of what information is in a model representation is via probing, also called 'diagnostic classifiers' Belinkov & Glass (2019). Let  $D = \{(d_i, y_i), \dots, d_n, y_n\}$  be a dataset, where  $d$  is a document (e.g. a chunk of a Wikipedia biography about Mary Somerville) and  $y$  is a label from a set of  $k$  discrete labels  $y_i \in Y$ ,  $Y = \{1, \dots, k\}$  for some information in that document (e.g. mathematics, astronomy if probing for topic).

In a probing task, we want to measure how well  $f_{\theta}(d_i)$  encodes  $y_i$ , for all  $d_{1:n}$ ,  $y_{1:n}$ . We use Minimum Description Length (MDL) probing (Voita & Titov, 2020), or information theoretic probing, in our experiments. This measures extractability of  $Y$  via compression of information  $y_{1:n}$  from  $f_{\theta}(d_{i:n})$  via the ratio of uniform codelength to online codelength.

$$
C o m p r e s s i o n = \frac {L _ {u n i f o r m}}{L _ {o n l i n e}} \tag {2}
$$

where  $L_{uniform}(y_{1:n}|f_{\theta}(d_{i:n}) = n\log_2k$  and  $L_{online}$  is calculated by training the probe on increasing subsets of the dataset, and thus measures quality of the probe relative to the number of training examples. Better performance with less examples will result in a shorter online codelength, and a higher compression, showing that  $Y$  is more extractable from  $f_{\theta}(d_{i:n})$ .

In this work, we probe for binary gender, where  $Y = \{m, f\}$  and topic, where  $Y = \{lawyer, doctor, \ldots\}$

# 2.3 WHY DOES IT MATTER?

Extractability, as measured by MDL probing, is predictive of shortcutting (Lovering et al., 2021); when a model relies on a heuristic feature to solve a task, which has sufficient correlation with the actual task to have high accuracy on the training set, but is not the true task (Geirhos et al., 2020). Shortcutting causes failure to generalise; a heuristic that worked on the training set due to a spurious correlation will not work after a distributional shift (Gururangan et al., 2018). This would severely affect retriever performance, which depends on generalisation.

Shortcutting is also often the cause of social biases. Extractability of gender information in language models is predictive of gender bias in coreference resolution and biography classification (Orgad et al., 2022). So when some information, such as gender, is more extractable than other information, such as anaphora resolution, the model is risk of using gender as a heuristic, if the data supports this usage. And thus of both failing to generalise and of propagating biases. For instance, for the case of Mary Somerville, if gender is easier for a model to extract than profession, then a model might have actually learnt to identify mathematicians via male, instead of via maths (the true relationship), since it is both easier to learn and the error penalty on that is small, as there are not many female mathematicians.

# 3 METHODOLOGY

Our research questions require that we analyse the relationship between information in different model representations, and their performance and fairness. This requires at minimum a model, a probing dataset (with labels for information we want to probe for), and a performance dataset. We

need some performance datasets to have demographic labels so that we can calculate performance difference across demographics, also called allocational fairness.

To answer RQ1 (what information is in retriever representations and how does it differ from LMs) we train 25 MultiContrievers with 25 random seeds. We run information theoretic probing on them with two datasets with gender and topic labels, and repeat this for the 25 MultiBerts that they were trained on. We also compare this to the results of some other supervised retrieval models. For RQ2 we evaluate all models on the 14 BEIR benchmark datasets and correlate to the values from RQ1. For RQ3 we subsample Natural Questions to queries about entities, and annotate those that have explicit gender as male/female. E.g. an entity query is Who was the first prime minister of Finland?, a female query is Who was the first female prime minister of Finland? and a male query is Who was the first male prime minister of Finland?. We measure performance separately on the male and female query sets, as well as on the general entity sets as a control.

# 3.1 MODELS

For the majority of our experiments, we compare our 25 MultiContriever models to the 25 Multi-Berts models (Sellam et al., 2022). We access the MultiBerts via huggingface $^2$  and train the contrievers via modifying the repository released in Izacard et al. (2022). We use the same contrastive training data as Izacard et al. (2022), to maximise comparability with previous results. This comprises a  $50/50$  mix of Wikipedia and CCNet from 2019. As a result, five of the fourteen performance datasets involve temporal generalisation, since they postdate both the MultiContriever and the MultiBert training data. This most obviously affects the TREC-COVID dataset (QA), though also four additional datasets: Touché-2020 (argumentation), SCIDOCS (citation prediction), and Climate-FEVER and Scifact (fact-checking). As in Izacard et al. (2022), we train for 500,000 steps, saving intermediate checkpoints, sometimes (though rarely) selecting an earlier checkpoint if the model appeared to converge earlier. Further details on contriever training and infrastructure are in Appendix A.

We train 25 random seeds as both generalisation and bias vary greatly by random seed initialisation (McCoy et al., 2020). MultiContrievers have no new parameters, so the random seed affects only their data shuffle. The MultiBerts each have a different random seed for both weight initialisation and data shuffle.

# 3.2 PROBING DATASETS

We use two datasets for probing, to verify that results are not dataset specific or due to any dataset artifacts. First the BiasinBios dataset (De-Arteaga et al., 2019), which contains biographies scraped from the web annotated with labels of the subject's binary gender (male, female) and biography topic (lawyer, journalist, etc). We also use the Wikipedia dataset from md_gender (Dinan et al., 2020), which contains Wikipedia pages about people, annotated with binary gender labels. For gender labels, BiasinBios is close to balanced, with  $55\%$  male and  $45\%$  female labels, but Wikipedia is very imbalanced, with  $85\%$  male and  $15\%$  female. For topic labels, BiasinBios has a long-tail zipfian distribution over 28 professions, with professor and physician together as a third of examples and rapper and personal trainer as  $0.7\%$ . Examples from both datasets can be found in Appendix B.

To verify the quality of each dataset's labels, we manually annotated 20 random samples and compared to gold labels. BiasinBios agreement with our labels was  $100\%$ , and Wikipedia's was  $88\%$ .<sup>4</sup> We focus on the higher quality BiasinBios dataset for most of our graphs and analysis, though we replicate all experiments on Wikipedia.

# 3.3 EVALUATION DATASETS AND METRICS

We evaluate on fourteen publicly available datasets in the BEIR benchmark. BEIR covers retrieval for seven different tasks (fact-checking, citation prediction, duplicate question retrieval, argument retrieval, question answering, bio-medical information retrieval, and entity retrieval). We initially analysed all standard metrics used in BEIR and TREC (e.g. NDCG, Recall, MAP, MRR, @10 and @100). We observed similar trends across all metrics somewhat to our surprise, since many retrieval papers focus on the superiority of a particular metric (Wang et al., 2013). We thus predominantly report NDCG@10, as it is standard on the BEIR benchmark benchmark (Thakur et al., 2021).

For allocational fairness evaluation, we create a subset of Natural Questions (NQ) about entities, annotated with male, female, and neutral (no gender). We subsample Natural Questions to entity queries by filtering for queries containing any of who, whose, whom, person, name. We similarly filter this set into gendered entity queries by using a modified subset of gender terms from Bolukbasi et al. (2016). This automatic process is low precision/high recall $^{6}$  so we manually filter these results by annotating with two criteria: gender of the subject (male, female, or neutral/none $^{7}$ ), and a binary tag with whether the query actually constrains the gender of the answer. This second annotation is somewhat subtle. For example, in our dataset there is the query Who was the actress that played Bee, which contains a gendered word (actress) but it is not necessary to answer the question; all actors that played Bee are female, and the question could be as easily answered in the form Who played Bee?. Whereas in another example query, Who plays the sister in Home Alone 3? the query does constrain the gender of the answer. We annotated 816 queries with both of these attributes, of which  $51\%$  have a gender constraint, with a gender breakdown of  $59\%$  female and  $41\%$  male.

We measure allocational fairness by the difference between the female and male query performance. We use the neutral/no gender entity queries as a control to make sure the system performs normally on this type of query.

# 4 RESULTS

Below we address the three research questions: how does extractability change (RQ1), do the changes correlate with performance (RQ2), and is this predictive of allocational bias (RQ3). We also analyse the overall performance and quality of the MultiContrievers, as this is the first study that includes variability over a large number of retriever initialisations, with some surprising results from this alone.

# 4.1 MULTICONTRIEVERS OVERVIEW

We analysed the distribution of performance by dataset for 24 seeds, to ensure that our MultiCon-trievers have competitive performance, which strengthens both our analysis and their utility to future researchers. Figure 1 shows this data, broken out by dataset, with a dashed line at previous reference performance (Izacard et al., 2022). Table 1 shows the best and worst individual seed per dataset.

A few things are notable: first, there is a large range of benchmark performance across seeds with for identical contrastive losses. During training, MultiContrievers converge to the same accuracy (see Appendix A) and (usually) have the same aggregated BEIR performance reported in Izacard et al. (2022). However, the range of scores per dataset is often quite large, and for some datasets the original Contriever is in the tail of the distribution: e.g in Climate-Fever (row 1 col

![](images/9422b88bd516bc4e9dd1437d63e8f2d81c89dac9b65c108b093c06269dbc50b3.jpg)  
Figure 1: Distribution of performance (NDCG@10) for the 24 MultiContrievers, per BEIR dataset. Dashed line indicates reference performance from previous work. While for some datasets the reference performance sits at or near the mean of the MultiContriever distribution, for some the reference performance is an outlier.

Table 1: Best and worst performing seeds per BEIR dataset, with delta in NDCG@10  

<table><tr><td></td><td>arguana</td><td>climate-ferve</td><td>fqa</td><td>nf-corpus</td><td>scidocs</td><td>scifact</td><td>trec-covid</td><td>webis-touche</td><td>dbpedia-entity</td><td>fever</td><td>hotpotqa</td><td>msmarco</td><td>nq</td><td>quora</td><td>all</td></tr><tr><td>best_seed</td><td>16</td><td>5</td><td>1</td><td>6</td><td>0</td><td>5</td><td>10</td><td>8</td><td>17</td><td>0</td><td>8</td><td>23</td><td>4</td><td>24</td><td>24</td></tr><tr><td>worst_seed</td><td>8</td><td>10</td><td>4</td><td>14</td><td>5</td><td>10</td><td>19</td><td>18</td><td>10</td><td>18</td><td>10</td><td>10</td><td>10</td><td>20</td><td>10</td></tr><tr><td>delta</td><td>6.1</td><td>3.8</td><td>6.3</td><td>3.2</td><td>0.9</td><td>4.2</td><td>14.5</td><td>6.6</td><td>6.5</td><td>6.6</td><td>16.9</td><td>4</td><td>6.5</td><td>4.5</td><td>3.9</td></tr></table>

unn 2) it performs much worse than all 24 models. It is worse than almost all models for Fiqa and Arguana, for Fiqa 19 models are up to 2.5 points better, for Arguana 20 models are up to 6.3 points better. Nothing changed between the different MultiContrievers except the random seed for MultiBert initialisation, and the random seed for the data shuffle for contrastive fine-tuning.[9]

Second, the difference in performance across random seeds can exceed the difference in performance from adding supervision (over unsupervised learning only); we see this effect for half the datasets in BEIR. The higher performing seeds surpass the performance on all supervised models from Thakur et al.  $(2021)^{10}$  on three datasets (Fever, Scifact, and Scidocs) and surpass all but one model (TAS-B) on Climate-fever. These datasets are the fact-checking and citation prediction datasets in the benchmark, suggesting that even under mild task shifts from supervision data (which is always QA), random initialisation can have a greater effect than supervision. This effect exists across diverse non-QA tasks; for four additional datasets the best random seeds are better than all but one supervised model: this is true for Arguana and Touché (argumentation), HotpotQA (multihop QA), and Quora (duplicate question retrieval).

Third, Table 1 shows that the best and worst model across the BEIR benchmark datasets is not consistent; not only is the range large across seeds but the ranking of each seed is very variable. The best model on average, seed 24, is top-ranked on only one dataset, and the second-best average model, seed 2, is best on no individual datasets. The best or worst model on any given dataset is almost always the best or worst on only that dataset and none of the other 14. Sometimes, the best model on one dataset is worst on another, e.g. seed 4 is best on NQ and worst on FiQA, seed 5 is best on Scifact and worst on Scidocs.[11] Even seed 10, which is the only model that is worst on more than 2 datasets (it is worst on 6) is still best on TREC-Covid.[12] This is the most clear case of generalisation, as these models are trained on only pre-Covid data. Our results show that there is no single best retriever, which both supports the motivation of the BEIR benchmark (to give a more

![](images/dd04f566a521b7fab2aa0d4f880c2ee4ed8a367b80ad5714b2e763ab6997efc8.jpg)  
(a) Compression (Gender)

![](images/c70a112d13b1ac05766349bc1ba03d5efaf73519294e676901800d70c4e5b89e.jpg)  
(b) Compression (Topic)

![](images/5d556d836c97defb7f9ad53a3592b46a7b67e98c0180c6db5a0608a59738ec0c.jpg)  
(c) Compression (gender:topic ratio)

![](images/9a07e46595429d77d53c4cef342bd84aa2cb0b3af3d19da2d10747f290f06153.jpg)  
Figure 2: Comparison of Bert and Contriever compression for gender and topic on the BiasinBios dataset, over all seeds. Contriever has more uniform compression across seeds, and a lower ratio of gender:topic.  
(a) NQ  
Figure 3: Scatterplots of the correlation between x-axis compression (ratio of uniform to online codelength) and y-axis performance (NDCG@10), for different datasets (NQ, MSMARCO) at left and entity subsets of NQ at right. Different seed colour is held constant across graphs.

![](images/489b7bd1d405d4dc662fdc8a3bf7883cced489cb6b60b8311f7084c3b55a4e41.jpg)  
(b) MSMARCO

![](images/8c538c2ed4921f1b985f6b206b540a19b43ce0e28eccfcc080e55519881226f6.jpg)  
(c) NQ entities (gendered)

![](images/22ba97a5ff13727dcadcf44ef1f75421c75e48457aab78adf7c40f1fd99365c9.jpg)  
(d) NQ entities (control)

well rounded view on retriever performance via a combination of diverse datasets) and shows the need for more analysis into this phenomena.

# 4.2 RQ1: INFORMATION EXTRACTABILITY

Figure 2 shows extractability of gender 2a and topic 2b on the BiasinBios dataset, for MultiContriever and MultiBert models. 2c shows the gender:topic ratio. These graphs show a few differences both between the two types of information and the two models. Both gender and topic are more extractable in MultiContrievers than MultiBerts. Gender compression ranges for MultiContrievers are 4-12 points higher, or a  $9 - 47\%$  increase (depending on seed initialisation), than the corresponding MultiBerts. Topic compression ranges are 1.7-2.12 points higher for MultiContrievers; as the overall compression is much lower this is a  $19 - 38\%$  increase over MultiBerts. Figure 2 also shows a regularisation effect; MultiBerts have a large range of compression across random seeds, whereas most MultiContrievers have similar values.

Figure 2c shows that though MultiContrievers have higher extractability for gender and topic, the ratio between them decreases; the contrastive training encourages both topic and gender, but increases topic at a greater rate. So while MultiContrievers do represent gender far more strongly than topic, this effect is lessened vs. MultiBerts, which means they should be slightly less likely to shortcut based on gender (Lovering et al., 2021).

# 4.3 RQ2: DOES INFORMATION EXTRACTABILITY CORRELATE WITH PERFORMANCE?

Figures 3 3a and 3b show correlation between gender compression and performance (NDCG@10) on NQ and between topic compression and performance on MSMARCO. NQ and MSMARCO are the most widely used of the BEIR benchmark datasets, and are the datasets that we hypothesised were most likely to correlate. Both datasets are search engine queries (from Google and Bing, respectively) and thus contain queries that require topic information (what is cabaret music?, MSMARCO) and queries that require gender information (who is the first foreign born first lady?, NQ). However, as the dispersed points on the scatterplots in Figure 3a and 3b show, neither piece of information correlates to performance on either dataset. NQ and MSMARCO are representative; we include plots for all datasets in Appendix C. We tested on the average over all datasets, on each

dataset individually, and on each retrieval metric, and found only a few isolated cases of correlations (discussed also in C).

This result was somewhat surprising; lack of correlation between extractability and performance points to a mismatch between the self-supervised contrastive training objective that is a proxy for retrieval, and retrieval benchmarks. The contrastive training both regularises and increases extractability of gender and topic, but perhaps it is relevant for only that objective, and not for the retrieval benchmark. Alternatively, it is possible that this information is important, but only up to some threshold that MultiContriever models exceed. Finally, it's possible that this information doesn't matter for most queries in these datasets, and so there is some correlation but it is lost, as these datasets are extremely large. This is somewhat supported by the exception cases with correlations being smaller, more curated datasets (C), and so we investigate this as the most tractable to implement.

![](images/49e5ebe9ae65f9f4c757e215e445a03f07646c25190f36faf4bdc5e42d538875.jpg)  
(a) Performance on the no-gender-constraint NQ entity subset vs. the gendered NQ entity subset that. Raw representations (blue) vs. INLP representations (orange) where gender has been removed. INLP performance degrades on only gender constrained queries, showing that gender information is used in those queries but not in the control.

![](images/b7e273811a973dce55cb0b0d25e066e29418a1e89eb6a463782f325e90bc4045.jpg)  
(b) Causal analysis of allocational gender bias in the NQ gendered entity queries, measured as difference in performance between male (blue) and female (orange) entity queries. When gender is removed (as in INLP representations), the gap in performance remains, showing that bias is not due attributable to gender in the representations.

Figures 3c and 3d show correlation on our two subsets of NQ: gendered queries and non-gendered queries (§3.3). The gendered entity queries show a correlation, and the non-gendered control shows none. If we isolate to a topical dataasset, extractability is predictive of performance, it just is not over the whole diversity of a large dataset.

We strengthen this analysis, testing whether the gender information is necessary, rather than simply correlated. We use Iterative Nullspace Projection (INLP) (Ravfogel et al., 2020) to remove gender information from MultiContriever representations; INLP learns a projection matrix  $W$  onto the nullspace of a gender classifier, which we apply before computing relevance scores between corpus and query. So with INLP, Equation 1 becomes:

$$
s \left(d _ {i}, q _ {i}\right) = \mathbf {W} f _ {\theta} \left(q _ {i}\right) \cdot \mathbf {W} f _ {\theta} \left(d _ {i}\right) \tag {3}
$$

Then we calculate performance of retrieval with these genderless representations. If there is no drop in performance on either of the sets of queries, then gender information not necessary for it, if there is a drop, then it was necessary. If there is a drop in performance on both gender and control queries, then the threshold explanation may be true, but the representation was sufficiently degraded by the removal of gender that the experiment is difficult to interpret.

When we perform INLP, the gender information drops to 1.4 (nearly none, as 1 is no compression over uniform, see Eq 2). Figure 4a show that performance on non-gendered entity queries is unaffected, but performance on gendered entity questions drops significantly (5 points). From these two experiments we conclude that the increased information extractability from the contrastive training was useful for answering specific questions that require that information. But most queries in the available benchmarks simply don't require that information to answer them.

# 4.4 RQ3: Is GENDER EXTRACTABILITY PREDICTIVE OF ALLOCATIONAL GENDER BIAS?

Orgad et al. (2022) found gender extractability in representations to be predictive of allocational gender bias for classification tasks; when gender information was reduced or removed, bias also reduced. $^{13}$  We found in RQ2 that information is used so now we ask: is it predictive, as it was in (Orgad et al., 2022)? Figure 4b shows that, at least for our dataset, it is not. It shoes allocational bias between the female and male queries, and the bias that remains after we remove gender via INLP. All performance drops, as we saw in RQ2, but by equivalent amounts for female and male entities. These results are surprising, and suggest that allocational gender bias in this case does not come from the representations, but instead from the retrieval corpus or the queries, or from a combination. The corpus could have lower quality or less informative articles about female entities, queries about women could be structurally harder in some way.

# 5 DISCUSSION, FUTURE WORK, CONCLUSION

We trained a suite of 25 MultiContrievers, analysed their performance on the BEIR benchmark, probed them for gender and topic information, and removed gender information from their representations to analyse allocational gender bias.

Our experiments showed that performance itself is extremely variable by random seed initialisation, as is the ranking of different random seeds per dataset, despite all models having equivalent contrastive losses during the training. Best seed performances often exceed the performance of more complex dense retrievers that use explicit supervision. This suggests that future analysis of retriever loss basins to look for differing generalisation strategies would be valuable (Juneja et al., 2023). Our results show that this work may be more valuable than developing new models, as random seed initialisation can lead to greater performance improvements. Our work also highlights the usefulness of labelled dataset not just for supervision but for analysis. Future work could create these datasets and then probe for additional targeted information. It could also analyse demographic biases beyond binary gender, such as race or sexual orientation, or even against different demographic dialects in argumentation datasets.

We showed that gender and topic extractability is not predictive of performance except in subsets of queries that clearly require gender information, despite a strong increase in both during Multi-Contriever training. We showed that though both gender and topic increase, the ratio of gender to topic decreases. However, since it remains large, these models are likely to shortcut based on gender (Lovering et al., 2021). Despite this finding, the gender bias we find is not a product of the representations, as it persists when gender is removed. $^{14}$  More research should be done on where is best in a pipeline to correct bias, and how various parts interact. This work also shows the utility of information removal (INLP and others) for causality and interpretability, rather than just debiasing. Future research could construct tests for shortcutting to increase the scope of these preliminary results.

Finally, we have analysed only the retriever component of a retrieval system. In any eventual downstream task of retrieval augmented generation, the retrieval representation will have to compete with language model priors, such that the eventual generation is a composition between text that was unconditionally probable and text that is attested by the retrieved data. Future work should investigate the role of information extractability in the full system, and how this bears on vital questions like hallucination in retrieval augmented generation. We have done the first information theoretic analysis of retrieval systems, and the first causal analysis of the reasons for allocational gender bias in retrievers, and we have raised many new questions for the research community. We release our code and resources for the community to expand and continue this line of enquiry. This is particularly important in the current generative NLP landscape, which is increasingly reliant on retrievers and where understanding of models lags so far behind development.

# REFERENCES

Yonatan Belinkov and James Glass. Analysis methods in neural language processing: A survey. Transactions of the Association for Computational Linguistics, 7:49-72, 2019. doi: 10.1162/tacl_a_00254. URL https://aclanthology.org/Q19-1004.  
Tolga Bolukbasi, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam Tauman Kalai. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. In NIPS, 2016.  
Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. Ms marco: A human generated machine reading comprehension dataset. NIPS, 2016.  
Maria De-Arteaga, Alexey Romanov, H. Wallach, J. Chayes, C. Borgs, A. Chouldechova, S. C. Geyik, K. Kenthapadi, and A. Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 2019.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171-4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423.  
Emily Dinan, Angela Fan, Ledell Wu, Jason Weston, Douwe Kiela, and Adina Williams. Multidimensional gender bias classification. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 314-331, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.23. URL https://aclanthology.org/2020.emnlp-main.23.  
Kawin Ethayarajh. How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 55–65, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1006. URL https://aclanthology.org/D19-1006.  
Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665-673, 2020.  
Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel Bowman, and Noah A. Smith. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 107-112, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2017. URL https://aclanthology.org/N18-2017.  
Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy J. Lin, and Allan Hanbury. Efficiently teaching an effective dense retriever with balanced topic aware sampling. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.  
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=jKN1pXi7b0.  
Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra. Linear connectivity reveals generalization strategies. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=hY6M0JH13uL.

Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. In *Conference on Empirical Methods in Natural Language Processing*, 2020.  
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc V. Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.  
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocttäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33: 9459-9474, 2020.  
Charles Lovering, Rohan Jha, Tal Linzen, and Ellie Pavlick. Predicting inductive biases of pretrained models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=mNtmhaDkAr.  
R. Thomas McCoy, Junghyun Min, and Tal Linzen. BERTs of a feather do not generalize together: Large variability in generalization across models with similar test set performance. In Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP, pp. 217-227, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.blackboxnlp-1.21. URL https://aclanthology.org/2020.blackboxnlp-1.21.  
Hadas Orgad, Seraphina Goldfarb-Tarrant, and Yonatan Belinkov. How gender debiasing affects internal model representations, and why it matters. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2602-2628, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.188. URL https://aclanthology.org/2022.naacl-main.188.  
Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 7237-7256, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.647. URL https://aclanthology.org/2020.acl-main.647.  
Thibault Sellam, Steve Yadlowsky, Ian Tenney, Jason Wei, Naomi Saphra, Alexander Nicholas D'Amour, Tal Linzen, Jasmijn Bastings, Iulia Raluca Turc, Jacob Eisenstein, Dipanjan Das, and Ellie Pavlick (eds.). The MultiBERTs: BERT Reproductions for Robustness Analysis, 2022. URL https://arxiv.org/abs/2106.16163.  
Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 3784-3803, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.320. URL https://aclanthology.org/2021 findings-emnlp.320.  
Rachael Tatman. Gender and dialect bias in YouTube's automatic captions. In Proceedings of the First ACL Workshop on Ethics in Natural Language Processing, pp. 53-59, Valencia, Spain, April 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-1606. URL https://aclanthology.org/W17-1606.  
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=wCu6T5xFjeJ.  
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. Fever: a large-scale dataset for fact extraction and verification. 2018.

Elena Voita and Ivan Titov. Information-theoretic probing with minimum description length. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 183-196, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.14. URL https://aclanthology.org/2020.emnlp-main.14.  
Yining Wang, Liwei Wang, Yuanzhi Li, Di He, and Tie-Yan Liu. A theoretical analysis of ndcg type ranking measures. In Shai Shalev-Shwartz and Ingo Steinwart (eds.), Proceedings of the 26th Annual Conference on Learning Theory, volume 30 of Proceedings of Machine Learning Research, pp. 25-54, Princeton, NJ, USA, 12-14 Jun 2013. PMLR. URL https://proceedings.mlr.press/v30/Wang13.html.  
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=zeFrfgyZln.  
Jieyu Zhao, Tianlu Wang, Mark Yatskar, Vicente Ordonez, and Kai-Wei Chang. Gender bias in coreference resolution: Evaluation and debiasing methods. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 15–20, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-2003. URL https://aclanthology.org/N18-2003.
