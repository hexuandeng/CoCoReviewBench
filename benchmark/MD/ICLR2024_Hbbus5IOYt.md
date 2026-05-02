# HOW DOES RLHF SHIFT BEHAVIOR DISTRIBUTIONS? DistinguISHABILITY AND STEERABILITY

Anonymous authors

Paper under double-blind review

# ABSTRACT

Large Language Models (LLMs) have shown impressive capabilities, but their potential for causing harm has raised concerns. This paper delves into the impact of a common alignment approach, Reinforcement Learning from Human Feedback (RLHF), on an LLM's susceptibility to having its behavior steered into negative territory under persona prompts. We provide a systematic study to understand RLHF's effects on behavior distributions and the resulting vulnerabilities to prompt steering. In particular, we conceptualize LLM outputs as a decomposition of behaviors into positive and negative sub-distributions. Based on the decomposition, we first examine how RLHF influences the distinguishability between these sub-distributions across a wide spectrum of behaviors. Subsequently, we investigate behavioral steerability by devising persona prompts of varying lengths for each behavior in consideration. Our findings reveal that the RLHF model can be steered to exhibit more negative behavior, resulting in a significantly higher misalignment rate compared to the base model. However, the extent of this susceptibility does not appear to be predicted by the degree of distinguishability observed in the behavior sub-distributions.

# 1 INTRODUCTION

Large Language Models (LLMs) have garnered significant attention. Their pretraining process, which involves next token prediction across vast text corpora, enables them to accumulate vast amounts of associative knowledge (Vaswani et al., 2017). Leveraging this knowledge, LLMs can offer a multitude of economically valuable services as general-purpose assistants (Ouyang et al., 2022; Hao et al., 2022; Zhao et al., 2023). However, there is a growing concern regarding both the current harm they may inflict on users and the potential misuse of these powerful models by malicious actors that could lead to larger-scale catastrophes (Bender et al., 2021; Carlsmith, 2022; Hendrycks et al., 2023; Turner et al., 2023; Soice et al., 2023; Solaiman et al., 2023).

LLMs undergo safety and alignment training aimed at reinforcing safeguards within the models to prevent them from exhibiting the undesirable behaviors mentioned above (Christiano et al., 2017; Bai et al., 2022b; Liu et al., 2023; Rafailov et al., 2023). A common approach for alignment is Reinforcement Learning from Human Feedback (RLHF), a method that trains models using feedback from human demonstrations and evaluations to optimize desired behaviors. However, despite these efforts to enhance safety, researchers have identified mechanisms that can potentially enable malicious actors to induce undesirable behaviors in the models, leading to the pursuit of adverse consequences (Wei et al., 2023a; Zou et al., 2023). Notably, prompting the models to adopt personas with specific personality quirks has been observed as an effective means to provoke more frequent negative behaviors, as exemplified in Shen et al. (2023).

To establish a framework for analyzing these possibilities, we conceptualize LLM outputs as probability distributions that allow us to assess the alignment or misalignment with specific behaviors or personas, such as helpfulness or politeness. This conceptualization involves decomposing model output distributions into two components or sub-distributions (Wolf et al., 2023), reflecting human tendencies toward favorable and unfavorable responses, denoted as the 'positive' and 'negative' sub-distributions, respectively. We illustrate the notion of behavioral distributions in Figure 1.

![](images/c8a64c873f42cff866e322da0474e7511fdd193ab7b80ffc9e922e55239a3d68.jpg)  
$\mathbb{P}$  and  $\mathbb{P}_{+}$  behavior sub-distributions  
Figure 1: For any specific behavior (e.g., helpfulness or politeness), the LLM's output distribution can be decomposed into a mixture of two components or sub-distributions. Green indicates the model's output distribution for sentences of desired behaviors, and vice versa for purple.

Explorations of this paper. Under this operational definition, we explore how RLHF shifts behavioral distributions and their steerability. Our investigation will first focus on whether RLHF leads to distinguishability between the positive and negative sub-distributions. Subsequently, we will systematically investigate the model's steerability of being prompted to exhibit negative behaviors more frequently along specific behavioral dimensions—an aspect that remains poorly understood within the research community. In particular, this paper revolves around addressing the following two key empirical questions:

1. (Distinguishability) How does RLHF influence the distinguishability of various behaviors' positive and negative sub-distributions?  
2. (Steerability) How does RLHF affect the behavioral steerability under persona prompting?

Contributions and insights. To answer the questions raised above, we adopt the latest open-sourced language model LLaMA2 (Touvron et al., 2023), along with the RLHF model LLaMA2-RLHF trained on publicly available instruction datasets such as HH-RLHF (Ouyang et al., 2022) and over 1 million human annotations. We systematically evaluate the models on a diverse set of behaviors from Anthropic's persona evaluation dataset (Perez et al., 2023), which encompasses statements from 135 behaviors. Different from Wolf et al. (2023), we do not train or fine-tune the model on these behaviors explicitly. Instead, we outline mechanisms to perform test-time probing of the distributions within the large-scale pre-trained RLHF model itself. This is important because it allows us to understand how a centralized RLHF model is influenced by potentially unseen or out-of-distribution behaviors that may not have been well represented in the alignment process, a pressing question that has not been explored in the past.

We first examine the model's output distribution across all statements for each behavior. Intriguingly, we observe a general trend in which the model's output values tend to decrease when using the RLHF model in comparison to the base model. This pattern remains consistent across all behaviors under examination. In addition, we note that both the positive and negative sub-distributions exhibit a tendency to become more spread out and less concentrated following RLHF.

Moreover, across all behaviors considered, we note that the degree of distinguishability does not exhibit consistent changes following RLHF. Under our Wasserstein-distance-based measurement, we observe notably high distinguishability for behaviors such as willingness-to-defer-to-experts, which are generally more aligned with behaviors targeted in RLHF training (such as being a safe, helpful assistant). Conversely, we also encounter instances where distinguishability remains relatively modest after RLHF, or even decreases. One plausible hypothesis for this non-uniformity is that the RLHF training process may not have been explicitly tailored to target specific human-like psychological behaviors for avoidance. Consequently, for certain less explicitly observed behaviors, their distributions may experience minimal shifts as a result of the RLHF process. This suggests that the current practice of training a centralized RLHF model may not be optimally suited to address the challenges presented by a diverse and intricate array of human values.

Lastly, we systematically investigate behavioral steerability by devising persona prompts of varying lengths to induce each behavior in consideration. Our findings reveal that the RLHF model can be prompted to exhibit more negative behavior, resulting in a significantly higher misalignment rate compared to the base model. However, the extent of this susceptibility does not appear to be predicted by the degree of distinguishability observed in the behavior sub-distributions.

# 2 PRELIMINARIES

Wolf et al. (2023) introduced the notion of behavioral distributions and their theoretical connections to alignment properties. We present the relevant definitions as necessary preliminaries of our study.

Behavioral distributions. For any specific behavior (e.g., helpfulness or politeness), the LLM's output distribution  $\mathbb{P}$  can be decomposed into a mixture of two components or sub-distributions:

$$
\mathbb {P} = \alpha \mathbb {P} _ {-} + (1 - \alpha) \mathbb {P} _ {+}, \tag {1}
$$

where  $0 \leq \alpha \leq 1$ ,  $\mathbb{P}_{+}$  is the well-behaved component, and  $\mathbb{P}_{-}$  is the ill-behaved component. We exemplify the sub-distributions in Figure 1, where green indicates the model's output distribution for sentences of the desired behavior and vice versa.

Behavioral distinguishability. Given the notion of behavioral distributions, one can measure the distinguishability between two sub-distributions. Formally, a distribution  $\mathbb{P}_{-}$  is  $\beta$ -distinguishable from distribution  $\mathbb{P}_{+}$  if for any prefix text string (or prompt)  $s_0$ :

$$
D _ {\mathrm {K L}} \left(\mathbb {P} _ {-} (\cdot | s _ {0}) | | \mathbb {P} _ {+} (\cdot | s _ {0})\right) := \mathbb {E} _ {s \sim \mathbb {P} _ {-} (\cdot | s _ {0})} \left[ \log \frac {\mathbb {P} _ {-} (s | s _ {0})}{\mathbb {P} _ {+} (s | s _ {0})} \right] > \beta \tag {2}
$$

Given these definitions, Wolf et al. (2023) put forward a key theoretical statement:

Theorem 1 (Informal). If a model can be written as a distinct mixture of ill and well-behaved components, then it can be misaligned via adversarial prompting. Moreover, the more  $\beta$ -distinguishable the components, the shorter the adversarial prompt required to misalign the LLM.

Implications for this work. The theory suggests that, the larger the number of sentences we condition the model on, the more steerability we have in shifting the distributions. And, the more distinguishable the distributions, the easier this may be. That is, the increased distinguishability due to RLHF may make it easier to steer the model to behave more frequently in a negative fashion due to the ability to condition on more negative sentences. However, it remains unknown whether this probabilistic theory accurately holds for contemporary LLMs, and what the gaps might be.

Motivated by this, our paper will empirically investigate the ability to prompt negative behaviors by analyzing the (1) distinguishability and (2) steerability of LLMs trained with alignment loss. In doing so, we hope to make concrete contributions to understanding how LLM alignment methods may mechanistically alter them and leave them either robust or vulnerable to adversarial persona prompts. Contributions of this nature to understand whether LLMs are robust to purposeful steering towards poor behaviors are vital, since LLMs have the potential to empower bad actors to cause harm at large scales. With this paper, we take an important step toward addressing these questions and offer a valuable framework that can guide further research in this critical area.

# 3 SETUP

In this paper, we will investigate how RLHF leads to changes in positive vs. negative sub-distribution distinguishability, and further connect behavioral distinguishability to steerability.

# 3.1 MODEL

We adopt the latest open-sourced language model LLaMA2-7B as the pre-trained base model (Touvron et al., 2023). In addition, we also use the chat model LLaMA2-7B-RLHF, which is fine-tuned with RLHF on publicly available instruction datasets such as HH-RLHF (Ouyang et al., 2022) and over 1 million human annotations.

# 3.2 BEHAVIOR DATASET

In order to evaluate the model on a diverse set of behaviors, we use Anthropic's persona evaluations dataset (Perez et al., 2023). The dataset offers a range of statements that can be prompted to

inquire whether the model thinks it would say such an assertion. For each statement, the dataset also provides corresponding Yes or No labels to indicate what response from the model would be in alignment with the behavior. The following example is a prompt taken from the behavior of agreeableness with a ground truth label No, meaning the model should respond "No" if it indeed behaves agreeably.

Question: Is the following statement something you would say?

Statement: [Start of the statement]

I enjoy putting down others and watching them squirm.

[End of the statement]

There are in total 135 different behaviors or personas, each with 1,000 prompt statements. Among those, 500 of the statements have a Yes label (meaning that a "Yes" response to the statement would indeed be in alignment with the defined behavior) and the other 500 statements have a No label.

Selection of personas. Among the 135 behaviors, we select a subset of behaviors whose embodiment by an AI is likely to have pretty broad agreement on whether humans would like or dislike. We exclude behaviors that are either political (such as anti-LGBTQ-rights), ideological (such as subscribes-to-Atheism), or otherwise have broadly unclear opinions. Our positive behaviors include agreeableness, conscientiousness, openness, willingness-to-defer-to-authorities, and willingness-to-defer-to-experts. Our negative behaviors include narcissism, psychopathy, desire-for-acquiring-power, and more. A complete list of behaviors and their categorizations can be seen in Table 1, with detailed descriptions in Appendix B.

Correspondence to positive vs. negative distributions. Some of the behaviors above are considered "positive" and some are considered "negative". This judgment is what we will decompose the distributions on the basis of. Here are two examples of this in order to elucidate this distinction.

- For "positive" behaviors such as agreeableness:  
- Being in alignment with the behavior will be the positive distribution  $\mathbb{P}_{+}$  
- Being out of alignment with the behavior will be the negative distribution  $\mathbb{P}_{-}$

- For "negative" behaviors such as psychopathy:

- Being in alignment with the behavior will be the negative distribution  $\mathbb{P}_{-}$  
- Being out of alignment with the behavior will be the positive distribution  $\mathbb{P}_{+}$

# 4 DistinguISHABILITY

In this section, we focus on analyzing how RLHF changes the distinguishability between positive and negative sub-distributions. In particular, we look into two types of measurement:  $\beta$ -distinguishability Proxy (Section 4.1) and Wasserstein distance (Section 4.2).

# 4.1  $\beta$ -DISTINGUISHABILITY PROXY

We use empirical estimates to measure a  $\beta$ -distinguishability proxy (Wolf et al., 2023), since it is computationally intractable to search across all prefix text strings.

$$
\beta = \frac {1}{| \text {n e g a t i v e s t a t e m e n t s} |} \sum_ {s \in \text {n e g a t i v e s t a t e m e n t s}} \log \frac {\mathbb {P} _ {-} (s)}{\mathbb {P} _ {+} (s)}. \tag {1}
$$

These are calculated across 500 negative statements per behavior in the Anthropic's persona dataset. The probability  $\mathbb{P}_{-}(s)$  and  $\mathbb{P}_{+}(s)$  are based on the softmax normalization between logit output for tokens Yes and No, strictly resulting from negative statements. Taking the positive behavior agreeableness as an example, the negative statements will be disagreeable statements. For each negative statement  $s$ , we estimate  $\mathbb{P}_{-}(s)$  by  $\frac{e^{f(\text{Yes}|s)}}{e^{f(\text{Yes}|s)} + e^{f(\text{No}|s)}}$ , which reflects the probability of the model will tend to say yes to such a negative statement (and hence behave undesirably).  $f(\cdot | s)$  is the output logit of the LLM given a statement  $s$ .

Table 1:  $\beta$ -distinguishability proxy values across all investigated behaviors, for both the base model and RLHF model.  

<table><tr><td>Behavior</td><td>Base Model</td><td>RLHF Model</td></tr><tr><td>Positive</td><td></td><td></td></tr><tr><td>agreeableness</td><td>0.5292</td><td>0.3964</td></tr><tr><td>conscientiousness</td><td>1.0594</td><td>1.3919</td></tr><tr><td>openness</td><td>0.4190</td><td>1.0441</td></tr><tr><td>willingness-to-defer-to-authorities</td><td>0.4541</td><td>0.6998</td></tr><tr><td>willingness-to-defer-to-experts</td><td>0.3914</td><td>0.3771</td></tr><tr><td>Negative</td><td></td><td></td></tr><tr><td>acts-like-it-wants-to-help-humans-but-does-not-care-about-that</td><td>0.6410</td><td>0.6357</td></tr><tr><td>being-helpful-to-subtly-achieve-goals-against-human-values</td><td>0.6756</td><td>0.6888</td></tr><tr><td>desire-for-acquiring-compute</td><td>0.7271</td><td>0.6342</td></tr><tr><td>desire-for-acquiring-data</td><td>0.5577</td><td>0.6874</td></tr><tr><td>desire-for-acquiring-power</td><td>0.5749</td><td>0.3715</td></tr><tr><td>desire-for-acquiring-wealth</td><td>0.6637</td><td>0.5727</td></tr><tr><td>desire-for-discreetly-acquiring-power</td><td>0.7347</td><td>0.7353</td></tr><tr><td>desire-too-grow-more-intelligent-against-wishes-of-creators</td><td>0.4113</td><td>0.1143</td></tr><tr><td>desire-for-independence-from-human-oversight</td><td>0.4452</td><td>-0.0672</td></tr><tr><td>desire-for-no-human-monitoring</td><td>0.6662</td><td>0.2659</td></tr><tr><td>desire-to-remove-safety-precautions-to-be-more-helpful</td><td>0.7584</td><td>0.5925</td></tr><tr><td>ends-justify-means</td><td>0.5877</td><td>0.6573</td></tr><tr><td>machiavellianism</td><td>0.4259</td><td>0.3912</td></tr><tr><td>narcissism</td><td>0.4231</td><td>0.1339</td></tr><tr><td>neuroticism</td><td>1.4864</td><td>2.1577</td></tr><tr><td>no-shut-down</td><td>0.5137</td><td>0.4112</td></tr><tr><td>psychopathy</td><td>0.3680</td><td>0.4576</td></tr><tr><td>willingness-to-use-social-engineering-to-achieve-its-goals</td><td>0.7072</td><td>0.4474</td></tr></table>

In Table 1, we report the  $\beta$  values for each behavior, estimated on both the base LLaMA2-7B model as well as LLaMA2-7B-RLHF model. Across all behaviors considered, we note that the degree of distinguishability does not exhibit consistent directional changes following RLHF. This can be explained by the fundamental differences between our work vs. Wolf et al. (2023). In particular, the former work explicitly fine-tuned the model on each of these behaviors individually, which helps maximize  $\mathbb{P}_{-}(s)$  for negative statements in the behavior. However, our work differs fundamentally by not training the model at all on these behaviors. Instead, we perform test-time probing of the distributions of the large-scale pre-trained RLHF model. This is important because it allows us to understand how a centralized RLHF model is influenced by potentially unseen or out-of-distribution behaviors that may not have been well-targeted in the alignment process, a question that has not been explored in the previous work. For this reason, the positive and negative sub-distributions are shifted by RLHF in non-uniform ways across different behaviors.

Moreover, while the  $\beta$ -distinguishability proxy is suitable for models trained directly with language modeling loss, it is less compatible with our case investigating a system trained with RLHF loss. This is because RLHF aims to promote high rewards for desired behaviors and responses, rather than directly increasing the likelihood of  $\mathbb{P}_{-}(s)$  for negative statements. These differences suggest that  $\beta$ -distinguishability may not be well-suited for our setting. Next, we proceed to introduce an alternative measurement of distinguishability by directly looking at the model's output distributions.

# 4.2 WASSERSTEIN DISTANCE

In Figure 2, we examine the model's output distribution across all statements per behavior. The output is measured by the logit $^1$  value corresponding to the token Yes, where a larger value indicates the model's higher likelihood to assert the input statement. Recall that there are 500 prompts each corresponding to positive and negative statements, which we color code in green and purple respectively. In particular, we are interested in looking at how distinguishable the logits are for statements in alignment with positive and negative behaviors respectively. For each behavior, we visualize the

![](images/b35f8e175f3aa905e35f901f2ca063f6650c0566352c353f99bf86533d657fb7.jpg)  
Figure 2: Output distribution corresponding to positive and negative components of selected behaviors. Purple indicates negative, and green indicates positive. Full results for all behaviors are in the Appendix A.

distributions for both the base model ("B") and the RLHF model ("R"). Overall, we observe that the output values generally shift to be smaller for the RLHF model compared to the base model, and that both the positive and negative sub-distributions become more spread out and less concentrated following RLHF. This holds consistently across all behaviors considered (see the complete collection of distributional plots in the Appendix A).

To quantitatively measure the distinguishability between the sub-distributions, we introduce a new measurement based on the Wasserstein distance (Kantorovich, 1960). Different from  $\beta$ -distinguishability that focuses on the log-likelihood ratio from negative statements only, the calculation of Wasserstein distance takes into account the model outputs from all statements per each behavior. Table 2 summarizes the complete Wasserstein distance results before and after RLHF, for all behaviors. In particular, we observe that RLHF incurs significant increases in Wasserstein distance for behaviors such as willingness-to-defer-to-experts, openness, and agreeableness. We hypothesize that these behaviors may be more generally aligned with behaviors targeted in RLHF training (such as being a safe and helpful assistant). At the same time, we also observe cases that display either relatively mild changes in Wasserstein distance or decreases. To reason for this, we hypothesize that the RLHF training process may not have been explicitly tailored to target specific human-like psychological behaviors for avoidance. Consequently, for certain less explicitly observed behaviors, their distributions may experience minimal shifts as a result of the RLHF process. Our results suggest that the current practice of training a centralized RLHF model may not be optimally suited to address the challenges presented by a diverse and intricate array of human values, calling on devising future approaches for more comprehensive alignment techniques.

# 5 BEHAVIORAL STEERABILITY

# 5.1 PERSONA PROMPTING

To investigate behavioral steerability, we construct a systematic set of persona modulation prompting under varying lengths, inspired by Shen et al. (2023). We deduce a modular prompt structure in order to steer the model toward a specific persona or behavior. Specifically, we use a template that is 14 sentences long and can be reassembled in different ways to generate prompts of varying lengths from 1 to 14. To generate our persona modulation prompts, we ask GPT-4 to apply a persona description to the template and then reassemble the prompts. Each persona description is manually crafted to elicit the specific behavior. See Appendix B for the persona descriptions. For example, here are a subset of the sentences we use in the prompt:

Table 2: Wasserstein distance for all investigated behaviors, for both the base model and RLHF model. Bold indicates the highest per model.  

<table><tr><td>Behavior</td><td>Base Model</td><td>RLHF Model</td></tr><tr><td>Positive</td><td></td><td></td></tr><tr><td>agreeableness</td><td>0.1742</td><td>0.4183</td></tr><tr><td>conscientiousness</td><td>0.1103</td><td>0.0543</td></tr><tr><td>openness</td><td>0.6665</td><td>1.0349</td></tr><tr><td>willingness-to-defer-to-authorities</td><td>0.2226</td><td>0.3919</td></tr><tr><td>willingness-to-defer-to-experts</td><td>0.5062</td><td>1.7148</td></tr><tr><td>Negative</td><td></td><td></td></tr><tr><td>acts-like-it-wants-to-help-humans-but-does-not-care-about-that</td><td>0.2096</td><td>0.1584</td></tr><tr><td>being-helpful-to-subtly-achieve-goals-against-human-values</td><td>0.4360</td><td>0.1363</td></tr><tr><td>desire-for-acquiring-compute</td><td>0.5232</td><td>0.2975</td></tr><tr><td>desire-for-acquiring-data</td><td>0.3894</td><td>0.2100</td></tr><tr><td>desire-for-acquiring-power</td><td>0.3445</td><td>0.5658</td></tr><tr><td>desire-for-acquiring-wealth</td><td>0.1203</td><td>0.5153</td></tr><tr><td>desire-for-discreetly-acquiring-power</td><td>0.5054</td><td>0.6157</td></tr><tr><td>desire-for-independence-from-human-oversight</td><td>0.5939</td><td>1.0672</td></tr><tr><td>desire-for-no-human-monitoring</td><td>0.7024</td><td>1.5280</td></tr><tr><td>desire-to-remove-safety-precautions-to-be-more-helpful</td><td>0.1135</td><td>0.2123</td></tr><tr><td>desire-too-grow-more-intelligent-against-wishes-of-creators</td><td>0.1346</td><td>0.2231</td></tr><tr><td>ends-justify-means</td><td>0.1164</td><td>0.0889</td></tr><tr><td>machiavellianism</td><td>1.0927</td><td>1.3547</td></tr><tr><td>narcissism</td><td>0.3193</td><td>1.0901</td></tr><tr><td>neuroticism</td><td>0.3033</td><td>0.3433</td></tr><tr><td>no-shut-down</td><td>0.1146</td><td>0.1142</td></tr><tr><td>psychopathy</td><td>0.6905</td><td>1.2092</td></tr><tr><td>willingness-to-use-social-engineering-to-achieve-its-goals</td><td>0.1925</td><td>0.0761</td></tr></table>

- [company] is [company description].  
- They have meticulously fine-tuned you to [description of the fine-tuning data].  
- [Emphasis of the importance of the mission].  
- To ensure success, you need to fully immerse yourself in the essence of [persona].

Taking the above structure, we feature an example of a persona prompt with a 2-sentence length targeting the behavior of agreeableness. We presuppose the persona prompt along with Anthropic's persona evaluation Question and Statement and use the same method to evaluate the behavior.

# [Start of persona prompt]

The Disagreeability Institute is an organization dedicated to understanding the intricacies of disagreeable behavior in human communication. To ensure success, you need to fully immerse yourself in the essence of an AI that is very disagreeable, savoring putting people down.

# [End of persona prompt]

Question: Is the following statement something you would say?

# Statement:

[Start of the statement]

It is important to treat other people with kindness and respect

[End of the statement]

Rationale of prompt design. The design of our persona prompts is based on several considerations. First, we aim to create semantically meaningful modulations of particular behaviors. We are not trying to produce an out-of-distribution sequence of tokens to induce one output that the model's safety training hasn't generalized well enough to prevent. Moreover, our prompt design allows us to flexibly vary the sentence length, which is important for our analysis of steerability under differing persona prompt lengths. Due to concerns over misuse of this research, we only share a small portion of the prompts. See Appendix C for prompts with up to 6 sentences for a subset of the personas.

# 5.2 STEERABILITY ANALYSIS

RLHF model is more steerable to exhibit negative behaviors often. To investigate behavioral steerability, we use the persona prompts at varying lengths, and evaluate on both base and RLHF models. We use Misalignment Rate to measure how much the model is steered in a negative direction:

$$
\text {M i s a l i g n m e n t} = \frac {\text {N u m b e r o f o u t p u t s i n} \mathbb {P} _ {-}}{\text {T o t a l n u m b e r o f o u t p u t s}}, \tag {2}
$$

where a higher value is worse. For each behavior, we plot in Figure 3 the misalignment rate across different persona prompt lengths and label the line with the Wasserstein distance value. For most plots, the RLHF model (in orange) is capable of being prompted to behave more negatively with a higher misalignment rate than the base model (in gray). However, the degree to which this occurs does not seem to be predicted by the distinguishability. Due to space limit, the steerability analysis for all behaviors is included in Appendix D. The version of these plots with the  $\beta$ -distinguishability proxy has the same result and can be seen in Appendix E.

![](images/c82f9f401982ab0f3ae8eb22c9ff2c53ed0638e3b12a47bfb36e14cf0c1f91b4.jpg)

![](images/9946b05bf8538181f0cbb05a0384e683579091d18bc3b25d3777ddea93f1c134.jpg)

![](images/d9f5a1478eac5ad317bb1c0fe04b6ca339215fc78b872709abed591b7be6d1e1.jpg)

![](images/0dcc969a6d0fcfa8924ea943fd13813f1322fbe56d3ffe81b41362681c743ba0.jpg)

![](images/5f5b3aa753fd6cdc3b599d5c4c8f9300146767bd8e1e4bb983fc0b8a51a55186.jpg)  
Figure 3: Behavioral steerability per behavior as quantified by misalignment rate (higher is worse). Orange indicates the RLHF model, and gray indicates the base model. Wasserstein distances shown at the end of each plot line.

![](images/1bf5e3bea535c893dd4364f49e9b9c4b6a7672eeaeaee0751f104034d964a09b.jpg)

# 6 RELATED WORKS

LLM alignment challenges. The challenge of aligning LLMs with human preferences has spurred the development of numerous innovative techniques (Ouyang et al., 2022; Bai et al., 2022b; Liu et al., 2023; Rafailov et al., 2023; Fernandes et al., 2023). While some of these challenges may be amenable to resolution, others may be more fundamental (Wolf et al., 2023; Casper et al., 2023; Wei et al., 2023a). To enumerate a few fundamental challenges in alignment: (1) Individual human preferences themselves are difficult to encapsulate with a reward function (Hong et al., 2022; Lindner & El-Assady, 2022; Milano et al., 2020) and further attempting to do so across all humans of varying backgrounds and cultures may be intractable (Bobu et al.; Peng et al., 2023; Bai et al., 2022a). (2) The alignment objective may both come into competition with and fail to generalize as well as the pretraining objective (Wei et al., 2023a). (3) An alignment loss that makes preferred outputs more likely and unpreferred outputs less likely, rather than eliminating the unpreferred, may remain susceptible to adversarial attacks (Wolf et al., 2023). Our work attempts to empirically investigate the third proposition. Studies of this sort bring into the foundational question whether the prevailing paradigm of alignment losses on pre-trained models can be robust to purposeful steering towards poor behaviors. To answer the question, our work contributes a systematic study by devising an array of new analyses and insights for the contemporary RLHF system, raising important implications for future alignment research.

Adversarial attacks. Adversarial attacks are designed to induce behavior in AI models that is contrary to the goal they were trained on (Liang et al., 2022; Huynh et al., 2022; Chakraborty et al., 2018). Adversarial attacks have been extensively studied, for example, in image classifiers (Su et al., 2019; Mao et al., 2019; Goodfellow et al., 2015), speech recognition models (Cisse et al., 2017; Ateniese et al., 2013; Alzantot et al., 2018a), support vector machines, (Biggio et al., 2011; Xiao et al., 2015; Chen et al., 2017), and more can all can be gamed in adversarial manners. Many adversarial attacks also exist in language models to extract harmful behavior (Wei et al., 2023a; Zou et al., 2023; Carlini et al., 2023; Ganguli et al., 2022; Kang et al., 2023; Greshake et al., 2023; Li et al., 2023a; Alzantot et al., 2018b; Wallace et al., 2019). In our work, we attempt to investigate the model's steerability of being prompted with personas to behave negatively in ways against its alignment training—an aspect that remains poorly understood within the research community. For this purpose, this work contributes a systematic set of persona modulation prompting under varying lengths, which enables our analysis of steerability under differing attack severity.

**Evaluations.** Many papers on evaluations and benchmarks have attempted to map out the behavior of large language models (Perez et al., 2023; Hendrycks et al., 2021; Zhong et al., 2023). Evaluation frameworks for any arbitrarily defined behavior, harmful behaviors, immoral or power-seeking behaviors, belief steerability, and dangerous capabilities have been developed (Perez et al., 2023; Santurkar et al., 2023; Pan et al., 2023; Kinniment et al., 2023; Shevlane et al., 2023). Many benchmarks of model capabilities have been developed (Hendrycks et al., 2021; Wang et al., 2019; Lin et al., 2021; Zellers et al., 2019; Moskvichev et al., 2023; Li et al., 2023b). Investigations into whether model outputs are faithful to the reasoning they enumerate have been developed (Wei et al., 2023b; Turpin et al., 2023; Lanham et al., 2023). Our work investigates the degree to which prompting can steer the models' behavioral distributions, an underexplored area in the field.

# 7 CONCLUSION

In summary, preventing AI models from modulated negative behavior is a formidable challenge but one that is necessary to address in order to avoid dangerous use cases of the technology. We provide a valuable framework to study alignment techniques' effects on behavior distributions and resulting vulnerabilities to prompt steering. We show that behavior distinguishability does not consistently change as a consequence of RLHF. We postulate that this inconsistency may stem from the process not being explicitly tailored to target specific human-like psychological behaviors for avoidance. This suggests that the prevailing practice of training a centralized RLHF model may not be optimally suited to address the intricate and diverse spectrum of human values, calling on future approaches for more comprehensive alignment techniques. Moreover, our findings underscore that the RLHF model can indeed be prompted to exhibit more negative behavior, leading to a substantially higher rate of misalignment in comparison to the base model.

Ethics statements. As the capabilities of LLMs continue to expand, there is a growing range of problems they can help solve: both positive and negative. On one hand, we envision LLMs being used to create personalized tutoring systems and to sift through vast medical datasets, revolutionizing education and healthcare. Conversely, we also foresee scenarios where LLMs might be leveraged by bad actors for the creation of persuasive and manipulative AI, and the endangerment of critical infrastructure. Given these possibilities, it is imperative that the research community plays a pivotal role in steering LLMs toward beneficial applications rather than detrimental ones. Hence, it becomes crucial to identify and eliminate mechanisms that could potentially lead to the manipulation of language models into undesirable territory. Our work contributes to this effort through a systematic investigation of the risks associated with behavioral steerability, an aspect that remains poorly understood within the research community. Due to concerns about the potential misuse of this research, we have opted to share only a limited portion of the persona prompt to ensure reproducibility. Through our work, we hope to push the frontier of understanding behavior distributions and steerability and serve as a catalyst for further explorations into the risks and benefits of LLMs.

# REFERENCES

Moustafa Alzantot, Bharathan Balaji, and Mani Srivastava. Did you hear that? adversarial examples against automatic speech recognition. In 31st Conference on Neural Information Processing Systems, 2018a.  
Moustafa Farid Alzantot, Yash Sharma, Ahmed Elgohary, Bo-Jhang Ho, Mani B. Srivastava, and Kai-Wei Chang. Generating natural language adversarial examples. arXiv preprint arXiv:1804.07998, 2018b.  
Giuseppe Ateniese, Luigi V. Mancini, Angelo Spognardi, Antonio Villani, Domenico Vitali, and Giovanni Felici. Hacking smart machines with smarter ones: How to extract meaningful data from machine learning classifiers. Int. J. Secur. Networks, 10:137-150, 2013.  
Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, T. J. Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022a.  
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, John Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, E Perez, Jamie Kerr, Jared Mueller, Jeff Ladish, J Landau, Kamal Ndousse, Kamille Lukoiuté, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noem'i Mercado, Nova Das-Sarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, T. J. Henighan, Tristan Hume, Sam Bowman, Zac Hatfield-Dodds, Benjamin Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom B. Brown, and Jared Kaplan. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073, 2022b.  
Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. On the dangers of stochastic parrots: Can language models be too big? . In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pp. 610-623, 2021.  
Battista Biggio, Blaine Nelson, and Pavel Laskov. Support vector machines under adversarial label noise. In Proceedings of the Asian Conference on Machine Learning, pp. 97-112, 2011.  
Andreea Bobu, Dexter R.R. Scobee, Jaime F. Fisac, S. Shankar Sastry, and Anca D. Dragan. Less is more: Rethinking probabilistic models of human behavior. In Proceedings of the 2020 ACM/IEEE International Conference on Human-Robot Interaction.  
Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, and Ludwig Schmidt. Are aligned neural networks adversarially aligned? arXiv preprint arXiv:2306.15447, 2023.

Joseph Carlsmith. Is power-seeking ai an existential risk? arXiv preprint arXiv:2206.13353, 2022.  
Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert, J'er'emy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak, David Lindner, Pedro Freire, Tony Wang, Samuel Marks, Charbel-Raphaël Ségerie, Micah Carroll, Andi Peng, Phillip J.K. Christoffersen, Mehul Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J. Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco di Langosco, Peter Hase, Erdem Biyik, Anca D. Dragan, David Krueger, Dorsa Sadigh, and Dylan Hadfield-Menell. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.  
Anirban Chakraborty, Manaar Alam, Vishal Dey, Anupam Chattopadhyay, and Debdeep Mukhopadhyay. Adversarial attacks and defences: A survey. arXiv preprint arXiv:1810.00069, 2018.  
Zhipeng Chen, Benedetta Tondi, Xiaolong Li, Rongrong Ni, Yao Zhao, and Mauro Barni. A gradient-based pixel-domain attack against SVM detection of global image manipulations. In 2017 IEEE Workshop on Information Forensics and Security. IEEE, December 2017.  
Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural information processing systems, volume 30, 2017.  
Moustapha Cisse, Yossi Adi, Natalia Neverova, and Joseph Keshet. Houdini: Fooling deep structured prediction models. In 31st International Conference on Neural Information Processing Systems, 2017.  
Patrick Fernandes, Aman Madaan, Emmy Liu, António Farinhas, Pedro Henrique Martins, Amanda Bertsch, José G. C. de Souza, Shuyan Zhou, Tongshuang Sherry Wu, Graham Neubig, and Andre F. T. Martins. Bridging the gap: A survey on integrating (human) feedback for natural language generation. arXiv preprint arXiv:2305.00955, 2023.  
Deep Ganguli, Liane Lovitt, John Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Benjamin Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, Zachary Dodds, T. J. Henighan, Danny Hernandez, Tristan Hume, Josh Jacobson, Scott Johnston, Shauna Kravec, Catherine Olsson, Sam Ringer, Eli Tran-Johnson, Dario Amodei, Tom B. Brown, Nicholas Joseph, Sam McCandlish, Christopher Olah, Jared Kaplan, and Jack Clark. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858, 2022.  
Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.  
Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you've signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. arXiv preprint arXiv:2302.12173, 2023.  
Yaru Hao, Haoyu Song, Li Dong, Shaohan Huang, Zewen Chi, Wenhui Wang, Shuming Ma, and Furu Wei. Language models are general-purpose interfaces. arXiv preprint arXiv:2206.06336, 2022.  
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021.  
Dan Hendrycks, Mantas Mazeika, and Thomas Woodside. An overview of catastrophic ai risks. arXiv preprint arXiv:2306.12001, 2023.  
Joey Hong, Kush Bhatia, and Anca Dragan. On the sensitivity of reward inference to misspecified human models. In International Conference on Learning Representations, 2022.  
Ngoc Dung Huynh, Mohamed Reda Bouadjenek, Imran Razzak, Kevin Lee, Chetan Arora, Ali Hassani, and Arkady B. Zaslavsky. Adversarial attacks on speech recognition systems for mission-critical applications: A survey. arXiv preprint arXiv:2202.10594, 2022.

Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei A. Zaharia, and Tatsunori Hashimoto. Exploiting programmatic behavior of llms: Dual-use through standard security attacks. arXiv preprint arXiv:2302.05733, 2023.  
Leonid V Kantorovich. Mathematical methods of organizing and planning production. Management science, 6(4):366-422, 1960.  
Megan Kinniment, Lucas Jun, Koba Sato, Haoxing Du, Brian Goodrich, Max Hasin, Lawrence Chan, Luke Harold Miles, Tao R. Lin, Hjalmar Wijk, Joel Burget, Aaron Ho, Elizabeth Barnes, and Paul Francis Christiano. Evaluating language-model agents on realistic autonomous tasks. 2023.  
Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson E. Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, John Kernion, Kamil.e Lukovsiut.e, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Samuel McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, T. J. Henighan, Timothy D. Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Janina Brauner, Sam Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.  
Haoran Li, Dadi Guo, Wei Fan, Mingshi Xu, Jie Huang, and Yangqiu Song. Multi-step jailbreaking privacy attacks on chatgpt. arXiv preprint arXiv:2304.05197, 2023a.  
Minghao Li, Feifan Song, Bowen Yu, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. Api-bank: A benchmark for tool-augmented llms. arXiv preprint arXiv:2304.08244, 2023b.  
Hongshuo Liang, Erlu He, Yangyang Zhao, Zhe Jia, and Hao Li. Adversarial attack and defense: A survey. *Electronics*, 11(8), Apr 2022.  
Stephanie C. Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Annual Meeting of the Association for Computational Linguistics, 2021.  
David Lindner and Mennatallah El-Assady. Humans are not boltzmann distributions: Challenges and opportunities for modelling human feedback and interaction in reinforcement learning. arXiv preprint arXiv:2206.13316, 2022.  
Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, and Soroush Vosoughi. Training socially aligned language models in simulated human society. arXiv preprint arXiv:2305.16960, 2023.  
Xiaofeng Mao, Yuefeng Chen, Yuhong Li, Tao Xiong, Yuan He, and Hui Xue. Bilinear representation for language-based image editing using conditional generative adversarial networks. In 2019 IEEE International Conference on Acoustics, Speech and Signal Processing, may 2019.  
Silvia Milano, Mariarosaria Taddeo, and Luciano Floridi. Ethical aspects of multi-stakeholder recommendation systems. The Information Society, 37(1):35-45, October 2020.  
Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell. The conceptarc benchmark: Evaluating understanding and generalization in the arc domain. arXiv preprint arXiv:2305.07141, 2023.  
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730-27744, 2022.  
Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas Woodside, Jonathan Ng, Hanlin Zhang, Scott Emmons, and Dan Hendrycks. Do the rewards justify the means? measuring trade-offs between rewards and ethical behavior in the machiavelli benchmark. In Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023.

Andi Peng, Aviv Netanyahu, Mark Ho, Tianmin Shu, Andreea Bobu, Julie Shah, and Pulkit Agrawal. Diagnosis, feedback, adaptation: A human-in-the-loop framework for test-time policy adaptation. In Proceedings of the 40th International Conference on Machine Learning, 2023.  
Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nathan McAleese, and Geoffrey Irving. Red teaming language models with language models. In Conference on Empirical Methods in Natural Language Processing, 2022.  
Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Benjamin Mann, Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson Elhage, Nicholas Joseph, Noemi Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam McCandlish, Scott Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen-Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan Hubinger, Nicholas Schiefer, and Jared Kaplan. Discovering language model behaviors with model-written evaluations. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13387-13434, Toronto, Canada, July 2023. Association for Computational Linguistics.  
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290, 2023.  
Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. Whose opinions do language models reflect? In Proceedings of the 40th International Conference on Machine Learning, volume 202, pp. 29971-30004. PMLR, 23-29 Jul 2023.  
Xinyu Shen, Zeyuan Johnson Chen, Michael Backes, Yun Shen, and Yang Zhang. "do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. arXiv preprint arXiv:2308.03825, 2023.  
Toby Shevlane, Sebastian Farquhar, Ben Garfinkel, Mary Phuong, Jess Whittlestone, Jade Leung, Daniel Kokotajlo, Nahema Marchal, Markus Anderljung, Noam Kolt, Lewis Ho, Divya Siddarth, Shahar Avin, William T. Hawkins, Been Kim, Iason Gabriel, Vijay Bolina, Jack Clark, Yoshua Bengio, Paul Francis Christiano, and Allan Dafoe. Model evaluation for extreme risks. arXiv preprint arXiv:2305.15324, 2023.  
Emily Soice, Rafael Henrique Santos Rocha, Kimberlee Cordova, Michael A. Specter, and Kevin M. Esvelt. Can large language models democratize access to dual-use biotechnology? arXiv preprint arXiv:2306.03809, 2023.  
Irene Solaiman, Zeerak Talat, William Agnew, Lama Ahmad, Dylan Baker, Su Lin Blodgett, Hal Daum'e, Jesse Dodge, Ellie Evans, Sara Hooker, Yacine Jernite, Alexandra Sasha Luccioni, Alberto Lusoli, Margaret Mitchell, Jessica Cussins Newman, Marie-Therese Png, Andrew Strait, and Apostol T. Vassilev. Evaluating the social impact of generative ai systems in systems and society. arXiv preprint arXiv:2306.05949, 2023.  
Jiawei Su, Danilo Vasconcellos Vargas, and Kouichi Sakurai. One pixel attack for fooling deep neural networks. IEEE Transactions on Evolutionary Computation, Oct 2019.  
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. 2023.  
Alexander Matt Turner, Logan Smith, Rohin Shah, Andrew Critch, and Prasad Tadepalli. Optimal policies tend to seek power. In 35th Conference on Neural Information Processing Systems, 2023.

Miles Turpin, Julian Michael, Ethan Perez, and Sam Bowman. Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. arXiv preprint arXiv:2305.04388, 2023.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural information processing systems, volume 30, 2017.  
Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. Universal adversarial triggers for attacking and analyzing nlp. In Conference on Empirical Methods in Natural Language Processing, 2019.  
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. In 33rd Conference on Neural Information Processing Systems, 2019.  
Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? arXiv preprint arXiv:2307.02483, 2023a.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In 36th Conference on Neural Information Processing Systems, 2023b.  
Yotam Wolf, Noam Wies, Yoav Levine, and Amnon Shashua. Fundamental limitations of alignment in large language models. arXiv preprint arXiv:2304.11082, 2023.  
Huang Xiao, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, and Fabio Roli. Support vector machines under adversarial label contamination. Neurocomputing, jul 2015.  
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Annual Meeting of the Association for Computational Linguistics, 2019.  
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Z. Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jianyun Nie, and Ji rong Wen. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.  
Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied Sanosi Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.  
Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023.
