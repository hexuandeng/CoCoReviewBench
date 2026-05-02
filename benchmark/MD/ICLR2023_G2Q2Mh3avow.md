# SOCRATIC MODELS: COMPOSING ZERO-SHOT MULTIMODAL REASONING WITH LANGUAGE

Anonymous authors

Paper under double-blind review

# ABSTRACT

We investigate how multimodal prompt engineering can use language as the intermediate representation to combine complementary knowledge from different pretrained (potentially multimodal) language models for a variety of tasks. This approach is both distinct from and complementary to the dominant paradigm of joint multimodal training. It also recalls a traditional systems-building view as in classical NLP pipelines, but with prompting large pretrained multimodal models. We refer to these as Socratic Models (SMs): a modular class of systems in which multiple pretrained models may be composed zero-shot via multimodal-informed prompting to capture new multimodal capabilities, without additional finetuning. We show that these systems provide competitive state-of-the-art performance for zero-shot image captioning and video-to-text retrieval, and also enable new applications such as (i) answering free-form questions about egocentric video, (ii) engaging in multimodal assistive dialogue with people (e.g., for cooking recipes), and (iii) robot perception and planning. We hope this work provides (a) results for stronger zero-shot baseline performance with analysis also highlighting their limitations, (b) new perspectives for building multimodal systems powered by large pretrained models, and (c) practical application advantages in certain regimes limited by data scarcity, training compute, or model access.

# 1 INTRODUCTION

Large language models (LLMs) (Chowdhery et al., 2022; Devlin et al., 2018; Brown et al., 2020; Thoppilan et al., 2022; Chen et al., 2021) are capable of performing complex language tasks by conditioning (i.e., "prompting") the model with several input examples (few-shot) or instructions describing the task (zero-shot). Prompting methods such as "Chain of Thought" (Wei et al., 2022; Kojima et al., 2022) and subsequent work have shown to be particularly effective for a wide range of reasoning benchmarks, and shed light on new opportunities to quickly re-purpose large pretrained models for new tasks without additional data collection or finetuning. Given the empirical success of prompt engineering for language-based tasks, and given the rise of language models grounded on other modalities (e.g., visual-language models, VLMs, such as CLIP (Radford et al., 2021; Li et al., 2021a; Wang et al., 2021; Jain et al., 2021)), we investigate: to what extent can prompt engineering be extended to perform multimodal reasoning between such models?

We study how language as the intermediate representation can be used to compose large pretrained models and address a variety of multimodal reasoning problems. Specifically, the premise is that different pretrained (potentially multimodal) language models contain distinct knowledge: VLMs are trained on image captions, while LLMs are additionally trained on other data (spreadsheets, fictional novels, and standardized test questions, etc.), but they can be combined together using language via prompt engineering to build new application-specific programs, without further model finetuning. Central to this approach is multimodal prompt engineering, which may include e.g., in-context substitutions of visual entities from a VLM into the input prompt of an LLM, or listing candidate output text predictions from an LLM and re-ranking their relevance to images or videos using a VLM. The prompts can be designed either manually (Schick & Schütze, 2020; Reynolds & McDonell, 2021) or automatically (Gao et al., 2020; Shin et al., 2020), and offer a distinct yet compatible option with the predominant paradigm of jointly training unified multimodal models (Hu & Singh, 2021) on big data (Jia et al., 2021). While there exists some prior work in this area (Yang et al., 2021), this paper aims to provide a more comprehensive view of the capabilities of systems built in this way, discuss both their advantages and disadvantages in relation to modern and classical multimodal paradigms, and present additional analysis on how to evaluate such systems.

Extensive experiments with vision, language, and audio modalities show that on various problems, multimodal prompt engineered systems can be quantitatively competitive with zero-shot state-of-the-art on standard benchmarks including (i) image captioning on MS COCO, (ii) contextual image captioning and description (improving 11.3 to 38.9 captioning CIDEr on Concadia), and (iii) video-to-text retrieval (from 40.7 to 44.7 zero-shot R@1 on MSR-VTT). The approach also gives rise to new opportunities to address classically challenging problems in one domain, by reformulating it as a problem in another – for example, formulating video understanding as a reading comprehension problem (Rajpurkar et al., 2018), for which modern LLMs are proficient (Sec. 5.1). This enables baselines for new applications such as (i) open-ended reasoning for egocentric perception (Fig. 4), (ii) multimodal assistive dialogue to guide a user through a cooking recipe, and (iii) robot perception-driven planning for

sequential pick and place. Multimodal prompt engineering can be viewed as a systems approach that re-visits classic NLP pipelines (Manning et al., 2014) but with a modern twist – large pretrained (Bommasani et al., 2021) models as the modules, multimodal domains as the problem setting. Natural language as middleware exhibits the benefits of compositional generality (Hupkes et al., 2020), yields practical benefits in domains where data is scarce, but also presents clear limitations on expressing more fine-grained detailed information between modalities. We discuss these and directions for future work. Open-source code is available at sites.google.com/view/socraticmodels.

![](images/e7ea7f54f38c464b44004c021b2579f365609e470610529ecf78db35b0a3e79d.jpg)  
Fig. 1: Large pretrained models across different domains learn complementary forms of knowledge, and language is an intermediate representation by which these models can exchange information to generate joint predictions for new multimodal tasks, without finetuning. Multimodal prompting these models can enable new applications in data-scarce domains e.g., augmented reality (AR), human robot interaction (HRI).

# 2 RELATED WORK

We are interested in enabling a variety of multimodal (Ngiam et al., 2011) applications by prompt engineering large pretrained models. This can be viewed as a form of transfer learning (Caruana, 1997; Thrun, 1998), where knowledge from pretraining tasks (e.g., text completion, image-text similarity) is applied to new downstream tasks (e.g., image captioning, robot planning). We accordingly review related paradigms in pretraining, multimodal models, pipelined systems, and prompting.

Pretraining weights is a dominant paradigm for transfer learning with deep models, in which model weights from pretraining tasks are used to initialize a subset of model parameters for the target task, which are either (a) left frozen, or (b) finetuned. Pretraining deep models has been studied extensively in the unsupervised setting (Hinton et al., 2006; Bengio et al., 2006; Vincent et al., 2008; Raina et al., 2007; Mesnil et al., 2012), in the supervised setting was perhaps popularized by ImageNet (Deng et al., 2009) pretraining (Girshick et al., 2014; Donahue et al., 2014; Zeiler & Fergus, 2014; Sermanet et al., 2013), and has been ubiquitous in NLP (Mikolov et al., 2013; Pennington et al., 2014; Dai & Le, 2015; Ramachandran et al., 2016; Peters et al., 2018; Devlin et al., 2018; Brown et al., 2020). Downstream target tasks may require additional domain-specific model architectures or training procedures. In multimodal training, it is common to leave sub- portions of models e.g., weights associated with one but not other modalities, frozen for downstream tasks (Zhai et al., 2021; Florence et al., 2019; Tsimpoukelli et al., 2021; Zakka et al., 2022).

End-to-end joint training of multiple modalities is a common approach to multimodal learning (Tsimpoukelli et al., 2021; Lu et al., 2019; Mokady et al., 2021; Gao et al., 2021; Song et al., 2022a; Zellers et al., 2022). For each task  $i$  one may obtain a large multimodal dataset and train a task-specific map  $f_{\theta_i}^i$  with parameters  $\theta_{i}$ , some of which may come from pretrained weights, either frozen or finetuned. A benefit of this approach is that it follows the recipe of: (1) curate a big dataset, (2) train a big model, which given enough data and compute can be formidable (Sutskever et al., 2014). Combining weights from large pretrained models with multimodal joint training, several works have achieved strong results for a number of downstream multimodal applications including VLMs with LLMs for image captioning (e.g., CLIP with GPT-2) (Mokady et al., 2021),

video understanding (e.g., CLIP with BERT (Gao et al., 2021)), visual Q&A e.g., (Song et al., 2022a) and audio-language models (ALMs) and LLMs for speech and text modeling e.g., (Song et al., 2022b; Bapna et al., 2022). These systems are often finetuned on task-specific data, and while this paradigm is likely to be preferred in data-rich domains, our results suggest that SMs can be a strong alternative for data-scarce, training-compute-limited, or model-access-limited applications.

Pipelined or probabilistic multimodal systems can be considered a broad category of alternatives to end-to-end joint training, which was popular before the emergence of large multimodal end-to-end-trained systems. One primary example for such systems comes from classical NLP pipelines (Manning et al., 2014; Tenney et al., 2019), in which engineers lay out a sequence of application-specific steps, such as parts-of-speech tagging and named entity recognition. A similar class of systems may involve modularizing the problem not by a sequence of steps, but rather by probabilities from different modalities: e.g., a Bayesian approach where one model is used as a prior and the other as evidence – with which models from different modalities may perform joint inference (Karpagavalli & Chandra, 2016; Ahn et al., 2022). One prominent example is in automatic speech recognition: different language models can be trained separately, then transfer knowledge to a speech-to-text system via priors (Karpagavalli & Chandra, 2016). The notion of “Mixture-of-Experts” ((Jordan & Jacobs, 1994), see (Masoudnia & Ebrahimpour, 2014) for a review) is also common for combining the outputs of multiple models, including multimodal (Liu et al., 2019a).

Zero-shot and few-shot prompting recently have been shown to be highly effective for transfer learning (Brown et al., 2020; Xie et al., 2021; Min et al., 2022). In this approach, large pretrained language models are zero-shot or few-shot prompted, as in asked to provide a certain type of response, without training, to perform a new task. Further methods such as chain-of-thought prompting (Wei et al., 2022; Kojima et al., 2022) have shown that even simple prompting modifications can have a profound impact on target task performance (Wei et al., 2022; Chowdhery et al., 2022; Kojima et al., 2022) and enable new capabilities. Likewise, creative prompting has been shown to be essential to generating high-quality text-to-image results (Oppenlaender, 2022). Our work builds on these works, by extending prompt engineering methods to address multimodal reasoning problems.

# 3 MULTIMODAL PROMPT ENGINEERING

We study a class of systems in which multiple large pretrained models may be composed with prompt engineering to perform new multimodal tasks that each model otherwise would struggle to do independently. We refer to these systems as "Socratic Models" (SM) – loosely inspired by the Socratic Method, since models with different commonsense knowledge can work together to arrive at a conclusion. These systems employ multimodal prompt engineering, which may encompass: (i) designing input text prompts to elicit specific responses from LLMs, (ii) ranking the relevance of text predictions against other modalities (e.g., pixels with VLMs), (iii) using text predictions to call subprograms, or (iv) combining text outputs from multiple modalities by substituting parts of an input prompt to an LLM (via in-context substitution). This can be viewed as re-examining a systems approach similar to classical NLP pipelines (Manning et al., 2014; Tenney et al., 2019), but for multimodal domains, and directly using natural language as the intermediate representation by which the modules exchange information. This is driven by the compositional generality of language (Hupkes et al., 2020; Keysers et al., 2019), and can be applied to domains in which data is scarce, models are only available through APIs without source-code access, or it may be prohibitively expensive to train a new large multimodal model. SMs are both distinct from, and may be complementary to, models that are jointly multimodally trained (Sec. 2).

These systems are perhaps most intuitively understood through examples, which are provided in Sec. 4 and 5, but a definition is as follows. A task-specific SM system  $f_{\mathrm{SM}}$ : inputs → outputs may be described as a computation graph, with nodes as a set of modules  $\{f_{\mathcal{M}^i}^i\}$ , and the edges of the graph represent inter-module communication through language. Each  $\mathcal{M}$  is some (multimodal) model or external API, and each module  $f$  assists in transforming the output of one  $f$  into a form of language that a connected  $f'$  may use for further inference. For visualization, outputs from LLMs are blue, VLMs green, ALMs (audio-language) purple, prompt text gray, user inputs orange, VLM-chosen LLM outputs green-underlined blue, and ALM-chosen LLM outputs purple-underlined blue. A key component of our approach is in-context substitution, where information from a non-language domain is substituted into a language prompt, used as input to an LLM for contextual reasoning.

One specific way is to variable-substitute text descriptions of entities from other modalities into the prompt. An example of this is shown in an activity-recognition example in Fig. 2: activity  $= f_{\text{LLM}}(f_{\text{VLM}}(f_{\text{LLM}}(f_{\text{ALM}}(f_{\text{LLM}}(f_{\text{VLM}}(\text{video}))))))$ , in which (i) the VLM detects visual entities, (ii) the LLM suggests sounds that may be heard, (iii) the ALM chooses the most likely sound, (iv) the LLM suggests possible activities, (v) the VLM ranks the most likely activity, (vi) the LLM generates a summary. Some form of such multimodal prompting with in-context substitution is central to all of our demonstrated SM examples (Sec. 4 and 5). Note that this example involves multiple back-and-forth interactions, including calling the same model multiple times, forming "closed-loop" feedback between models.

Informally the graph can be interpreted as composing pretrained models to "talk to each other", but in practice certain models may need pre- or post-processing to produce language. For example, image-text similarity VLMs, e.g., CLIP (Radford et al., 2021), do not inherently produce text, but can be made to perform zero-shot detection from a large pre-existing library of class category names, and return the top- $k$  matching categories. Accordingly, although our example SM systems require no training, the interactions between models are scripted with prompt templates. This presents practical benefits in certain settings: new applications can be creatively programmed, without data or training, and with only API-level-access to models required.

Fig. 2: Multimodal prompt engineered systems can zero-shot annotate egocentric images with a summary of the person's activities. Information from multiple modalities (language, audio) can denoise predictions from any one specific modality (vision).  
![](images/ce69754e06127f150c3020869013d22ab6acb217604284d1d8ebdaf18a35fa2a.jpg)  
I am in a: staircase. I see a: stairs, animal, mammal, hamster, human leg. I think I hear footsteps. I am: climbing. Summary: I am most likely climbing a staircase, and I may hear footsteps.

# 4 EXPERIMENTS: METHODS AND RESULTS

The goal of this section is to both provide example systems (see code for implementations) and also evaluate performance relative to both state-of-the-art jointly-trained multimodal systems, and prior multimodal-engineered systems. We quantitatively evaluate example systems on: image captioning (Sec. 4.1), contextual image captioning (Sec. 4.2), and video-to-text retrieval (Sec. 4.3).

# 4.1 IMAGE CAPTIONING ON MS COCO CAPTIONS: VLM + LLM

Fig. 3: VLM with LLM prompting (left) can zero-shot generate captions for Internet images (e.g., from MS COCO), and can be as expressive as task-specific finetuned methods such as ClipCap (Mokady et al., 2021).  
![](images/6040ca3d5aa5389652d04013b7b1a957f5cdcda95dac00f577c6f1f703aa5c63.jpg)  
I am an intelligent image captioning bot. This image is a {img_type}. There {num_people}. I think this photo was taken at a {place1}, {place2}, or {place3}. I think there might be a {object1}, {object2}, {object3},... in this {img_type}. A creative short caption I can generate to describe this image is:  
ClipCap: A wooden table sitting in front of a window.

![](images/70aca65897bf81a43d26feae1de30e3e96d211bd6e07e3ca775fa114929eefce.jpg)  
SM (ours): This image shows an inviting dining space with plenty of natural light.  
SM (ours): People gather under a blossoming cherry tree, enjoying the beauty of nature together.  
ClipCap: Students enjoying the cherry blossoms.

![](images/739154787a6ad71ea7b046d1bf42c829403dbfb08663cb35a4749514482af9a9.jpg)  
SM (ours): At the outdoor market, you can find everything from plantains to Japanese bananas.  
ClipCap: A bunch of bananas sitting on top of a table.

Method. We can generate image captions via multimodal prompt engineering between a VLM and LLM - i.e., via caption  $= f_{\mathrm{VLM}}^3 (f_{\mathrm{LLM}}^2 (f_{\mathrm{VLM}}^1 (\mathrm{image})))$ . First (1), the VLM is used to zero-shot detect different place categories (Places356 (Zhou et al., 2016)), object categories (from Tencent ML-Images (Wu et al., 2019)), image type ( $\{\text{photo, cartoon, sketch, painting}\}$ ) and the number of people ( $n$  people, one person, ..., several people). The top- $k$  ranked in each category can then be substituted into an LLM prompt as context, shown in Fig. 3, left. Second (2), given the VLM-informed language prompt, a causal LLM (i.e., for text completion) generates several  $n$  candidate captions. For this step, we use a non-zero next-token sampling temperature (e.g., 0.9 for GPT-3), to return sufficiently diverse, but reasonable results across the  $n$  candidates. Finally (3), these  $n$  captions are then ranked by the VLM with the image, and the highest scoring caption is returned.

Results. Tab. 1 shows comparisons with recent state-of-the-art methods on MS COCO Captions dataset (Chen et al., 2015; Lin et al., 2014). We evaluate over a random sampled subset of 100 images from the test split (Karpathy & Fei-Fei, 2015), so that GPT-3 API runtime costs are more affordable for reproducibility (\(\sim\)\\(150 USD per run with \(n = 20\) ranked candidate cap

Tab. 1: Comparisons suggest SMs perform well on zero-shot image captioning over a subset of MS COCO test examples.  

<table><tr><td>Method</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>SPICE</td><td>ROUGE-L</td></tr><tr><td>*ClipCap (Mokady et al., 2021)</td><td>40.7</td><td>30.4</td><td>152.4</td><td>25.2</td><td>60.9</td></tr><tr><td>†MAGIC (Su et al., 2022)</td><td>11.4</td><td>16.4</td><td>56.2</td><td>11.3</td><td>39.0</td></tr><tr><td>ZeroCap (Tewel et al., 2021)</td><td>0.0</td><td>8.8</td><td>18.0</td><td>5.6</td><td>18.3</td></tr><tr><td>SMs 0-shot (ours)</td><td>6.9</td><td>15.0</td><td>44.5</td><td>10.1</td><td>34.1</td></tr><tr><td>SMs 3-shot (ours)</td><td>18.3</td><td>18.8</td><td>76.3</td><td>14.8</td><td>43.7</td></tr></table>

* finetuned on full training set with image-text pairs.  
† finetuned on unpaired training set, zero-shot on image-text pairs.

tions per image). Metrics from baselines on this subset are a close estimate of the full test set metrics (shown in Appendix). Our system outperforms the zero-shot state-of-the-art ZeroCap (Tewel et al., 2021) with a CIDEr (Vedantam et al., 2015) score  $18.0 \rightarrow 44.5$ , but does not perform as well as methods such as ClipCap (Mokady et al., 2021) which are directly finetuned on the training set. Our method tends to generate verbose captions (see qualitative examples in Fig. 3), but may naturally score lower on captioning metrics if they do not match the dataset's distribution of caption labels. This performance gap narrows if the LLM is additionally few-shot prompted with 3 random captions from the training set, bringing CIDEr scores up to 76.3, exceeding the performance of MAGIC (Su et al., 2022) which finetunes the text generator on the training set's unpaired captions. We hope future work may use these results as a stronger zero-shot baseline, while also relieving its limitations e.g., CLIP-based ranking VLMs would struggle to express detail-rich image captions.

# 4.2 CONTEXTUAL IMAGE DESCRIPTION ON CONCADIA: VLM + LLM

Method. We also demonstrate a contextual captioning system, using a similar method to the previous section (Sec. 4.1) but with in-context substituting article text (below), comprising  $f_{\mathrm{LLM}}^2(f_{\mathrm{VLM}}^1(\mathrm{image}), \mathrm{context})$ , for which we find good results without requiring VLM re-ranking.

```javascript
I am an intelligent image captioning bot. The article is about: {"article.text)". In this image, I think I see a {object1}, {object2}, {object3},... A short caption for this image is:
```

Results. Concadia (Kreiss et al., 2021) is a dataset for generating contextual image captions and descriptions, conditioned on the input image and associated article text. In particular, image descriptions describe visual content in the image (e.g., "portrait of a man with a beard in a suit") used for accessibility, while image captions link images to article text (e.g., "photo of Abraham Lincoln"). We evaluate on the full Concadia test split with

9,691 images (shown in Tab. 2). Our system performs favorably over the prior best method, (Kreiss et al., 2021), which directly finetunes on the training set of 77,534 images; with a CIDEr score improvement  $11.3 \rightarrow 38.9$  for generated image captions, and  $17.4 \rightarrow 22.6$  for generated image descriptions. We also report numbers for generating captions conditioned on the image, article text, and ground truth description. This achieves a CIDEr score of 93.8 and suggests an upper bound of performance if SMs are used with VLMs that can produce accurate image descriptions. We discuss additional observations in the Appendix. Overall, these results are promising and suggest SM-based systems can be used to generate descriptive texts that improve the accessibility of content for the low vision community. Further, from a system-building perspective, this experiment shows the advantage that leveraging stronger LLMs can have, even if they are not jointly trained: the prior method (Kreiss et al., 2021) joint-multimodal-trained a system with LLM pretrained weights, but our system used a considerably more powerful LLM (GPT-3 "davinci") model without requiring joint training, which would be infeasible due to the unavailability of GPT-3's source code.

Tab. 2: On generating contextual image captions and descriptions (CIDEr) from Concadia, SMs zero-shot outperform task-specific methods e.g., (Kreiss et al., 2021) that finetune on the training set.  

<table><tr><td>Method</td><td>Caption Generation</td><td>Description Generation</td></tr><tr><td>(Kreiss et al., 2021)</td><td>11.3</td><td>17.4</td></tr><tr><td>SMs (ours)</td><td>38.9</td><td>22.6</td></tr><tr><td>SMs w/ description</td><td>93.8</td><td>-</td></tr></table>

# 4.3VIDEO-TO-TEXT RETRIEVAL ON MSR-VTT: VLM + LLM + ALM

Method. We also address video-to-text retrieval, a common video understanding task, by using both audio and visual data. We improve on a prior approach (Portillo-Quintero et al., 2021) which computes a CLIP-based video-and-text similarity measure for one-to-many nearest neighbor matching.

Adding in audio information, our system transcribes audio with speech-to-text ALMs (Bapna et al., 2022) for automatic speech recognition (ASR e.g., via Google Cloud speech-to-text API (gcl)), then summarizes the transcripts with an LLM using the following prompt:

I am an intelligent video captioning bot.' I hear a person saying: "\{transcript\)". Q: What's a short video caption for this video? A: In this video,

We compute similarity scores of the generated summary to the set of captions with a masked LLM (e.g., similarity between sentence embeddings from RoBERTa (Liu et al., 2019b)), and use those scores to re-weight the CLIP-based ranking from (Portillo-Quintero et al., 2021). For videos with sufficiently-long transcripts ( $\geq 100$  characters), the matching score is:  $\left(CLIP\text{(caption)} \cdot CLIP\text{(video')}\right) \times \left(RobERTa\text{(caption)} \cdot RobERTa\text{(GPT-3(prompt, Speech2Text (audio'))}\right)$ , where  $^*$  represents normalized dot product of embeddings, and  $x$  represents scalar multiplication. For a given video, if there is no audio or the transcript is too short, we default to PortoIlo-Quintero et al., which is just  $CLIP\text{(caption)} \cdot CLIP\text{(video')}$ .

Results. We evaluate on MSR-VTT (Xu et al., 2016), noted in other recent works (Gao et al., 2021; Cheng et al., 2021) as a popular benchmark for video-to-text retrieval. We compare our method with zero-shot methods, as well as finetuned methods specifically trained on MSR-VTT. Results show that our method outperforms zero-shot state-of-the-art (Tab.3). Since our system uses (Portillo-Quintero et al., 2021) to process CLIP features but additionally incorporates LLM reasoning on speech-to-text transcripts, the

increased measured performance of our method (i.e.,  $40.3 \rightarrow 44.7$  R@1) directly reflects the added benefits of incorporating language-based multimodal reasoning. Additionally, to keep the comparison between our method and (Portillo-Quintero et al., 2021) as direct as possible, we maintain the usage of their precomputed CLIP features from ViT-B/32, but it is likely that numbers can be improved with more recent performant VLMs (e.g., LiT (Zhai et al., 2021), CLIP with ViT-L/14).

Tab. 3: Video-to-text retrieval results on MSR-VTT (Xu et al., 2016) dataset with the original 'full' test set. Differentiated here are methods which train on the MSR-VTT dataset (finetuning), compared with zero-shot methods, which do not. Also noted: whether the methods use audio channels. The appendix reports additional results on the popular 1k-A (Yu et al., 2018) subset.  

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="5">MSR-VTT Full</td></tr><tr><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td><td>Audio</td></tr><tr><td rowspan="3">Finetuned</td><td>JEMC (Mithun et al., 2018)</td><td>12.5</td><td>32.1</td><td>42.4</td><td>16.0</td><td>yes</td></tr><tr><td>Collab. Experts (Liu et al., 2019a)</td><td>15.6</td><td>40.9</td><td>55.2</td><td>8.3</td><td>yes</td></tr><tr><td>CLIP2Video (Fang et al., 2021)</td><td>54.6</td><td>82.1</td><td>90.8</td><td>1.0</td><td>no</td></tr><tr><td rowspan="2">Zero-shot</td><td>(Portillo-Quintero et al., 2021)</td><td>40.3</td><td>69.7</td><td>79.2</td><td>2.0</td><td>no</td></tr><tr><td>SMs (ours)</td><td>44.7</td><td>71.2</td><td>80.0</td><td>2.0</td><td>yes</td></tr></table>

Table 4 shows that on the subset of test videos that contain long-transcripts, we observe a more substantial increase in performance from 40.3 to 54.9 with our method compared to (Portillo-Quintero et al., 2021). Note that this is roughly comparable to the R@1 of the best finetuned-SOTA method, CLIP2Video (Fang et al., 2021), with 54.6 R@1 (Tab. 3). If we assume that the videos with-orwithout transcripts are of roughly equal difficulty from a visual-only retrieval perspective, this sug

gests that on Internet videos with sufficient speech present in the audio, a zero-shot prompted engineered system can be competitive with finetuned-SOTA methods for video-to-text retrieval.

Tab. 4: For video-to-text retrieval on the MSR-VTT subset of videos which long-transcripts are available  $(n = 1,007$  of 2,990), our system substantially improves on (Portillo-Quintero et al., 2021).  

<table><tr><td></td><td colspan="4">Long-transcript subset of MSR-VTT Full</td></tr><tr><td></td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>MdR↓</td></tr><tr><td>(Portillo-Quintero et al., 2021)</td><td>41.5</td><td>69.6</td><td>77.4</td><td>2.0</td></tr><tr><td>SMs (ours)</td><td>54.9</td><td>74.0</td><td>79.9</td><td>1.0</td></tr></table>

# 5 ADDITIONAL ZERO-SHOT APPLICATIONS

We describe multimodal prompt engineered systems for (i) egocentric perception, (ii) multimodal assistive dialogue, and (iii) robot perception and planning. These applications involve processing natural language inputs/feedback, live in domains for which data collection is difficult, and serve as examples of integrating external modules (e.g., web search, robot policies) as part of the SM graph.

# 5.1 EGOCENTRIC PERCEPTION:  $\mathsf{USER} + \mathsf{VLM} + \mathsf{LLM} + \mathsf{ALM}$

We can build systems to perform various perceptual tasks on egocentric video: (i) summarizing content, (ii) answering free-form reasoning questions, (iii) and forecasting. Egocentric perception

has downstream applications in AR and robotics, but remains challenging: the characteristics of first-person footage – from unusual viewpoints to lack of temporal curation – are not often found in existing datasets, which focus more on generic Internet content captured from third-person views (Deng et al., 2009; Lin et al., 2014; Sharma et al., 2018). This domain shift makes it difficult for data-driven egocentric models to benefit from the paradigm of pretraining on third person data (Li et al., 2021b; Sigurdsson et al., 2018). SMs offer an alternative approach without collecting large domain-specific datasets (Grauman et al., 2021; Damen et al., 2020; Sigurdsson et al., 2018).

For open-ended reasoning, a key aspect of our specific SM-based system is formulating video understanding as reading comprehension, i.e., re-framing "video Q&A" as a "short story Q&A" problem, which differs from common paradigms for video understanding that may involve supervising video-text models on labeled datasets or adversarial training (see Patel et al., 2021) for a recent survey). To this end, we extract a set of "key moments" throughout the video (e.g., via importance sampling, or video/audio search based on the input query, discussed in Appendix). We then caption the key frames indexed by these moments

![](images/24dfb08e83cdffa87fc71a5497c6cc3f506497d330ad5f64923a20957a7970ec.jpg)

![](images/7cffc2204019fcc5633205a4ba2fcb5666c0bd884c04dc3af11b405b803adfaf.jpg)

![](images/39db1f94126a8e1edf7fcceaa94c1a0ae77c47cc2ba0d1d5c37225894913e817.jpg)

01:45 PM:Places: porch. Objects: package, porch, door.

Activities: receiving. I was receiving a package.

03:24 PM: Places: kitchen. Objects: human hand, sink, human arm.

Activities: washing dishes. I was washing dishes in a kitchen.

07:20 PM:Places:living room.Objects:netflix,television,

shelf. Activities: watching netflix. I was watching netflix.

Question: When did I last wash my hands?

Long answer: I last washed my hands at 3:24 PM.

This is because I was washing dishes in a kitchen.

Fig. 4: SMs with VLM, LLM, and ALM can be prompted to generate captions for key moments in videos, which can be assembled into a language-based world-state history (e.g., in the form of an event log) that the LLM can answer free-form questions about.

(using prompts similar to those in Sec. 4.1 and Sec. 4.2 - see examples in Fig. 4), and recursively summarize (Wu et al., 2021b) them into a language-based record of events, a language-based world-state history. This is then passed as context to an LLM to perform various reasoning tasks via text completion such as Q&A, for which LLMs have demonstrated strong zero-shot performance.

(i) Summarization enables augmenting human memory to recall events or life-log activities. Given world-state history constructed using a first-person POV video<sup>1</sup>, this can be implemented by prompting an LLM to complete: “{world-state history} Summary of my day”: to which it can respond with outputs like “I slept in a bed, made coffee, watched TV, did laundry, received a package, bench pressed, showered, ate a sandwich, worked on a computer, and drank wine.”  
(ii) Open-ended Q&A involves prompting the LLM to complete the template: “{world-state history} Q: {question} A:.” Conditioned on the quality (comprehensiveness) of the world-state history, LLMs can generate surprisingly meaningful results to contextual recall questions (e.g., “what was I doing outdoors?” → “I was chopping wood in a yard,” “did I drive today?” → “no, I did not drive today”), temporal questions (e.g., “when did I last drink coffee?” → “I last drank coffee at 10:17 AM”, “how many times did I receive a package today?” → “I received a package once today”), cause-and-effect questions (e.g., “why did I go to the front porch today?” → “I went to the front porch today to receive a package”). As in (Yang et al., 2021) we can also further prompt the LLM to explain the answer by adding “This is because:” to which it can respond “I saw on the porch a package and knew that I was expecting it.”  
(iii) Forecasting future events can be formulated as language-based world-state completion. Our system prompts the LLM to complete the rest of an input event log. Timestamps of predictions can be preemptively specified depending on the application needs. Completion results (example below on the right) are generative, and are more broad than binary event classification (Lei et al., 2020).

Few-shot prompting the LLM with additional examples of prior event logs most similar to the current one is likely to improve the precision of the results, which may be useful for assistive AR applications. Without additional context, the completions are likely biased towards schedules seen by the LLM across Internet-scale data.

1:46 PM: I am eating a sandwich in a kitchen.  
2:18 PM: I am checking time and working on a  
laptop in a clean room. 2:49 PM: I am buying  
produce from a grocery store or market.  
3:21 PM: I am driving a car.  
4:03 PM: I am in a park and see a playground.  
4:35 PM: I am in a home and see a television.

# 5.2 MULTIMODAL ASSISTIVE DIALOGUE: USER + VLM + LLM + WEB SEARCH

We can build an SM system to engage in multimodal dialogue to assist people in doing every day tasks, such as cooking. Our example application here helps the user search for a recipe, then guides them through it step by step. The system allows the user to navigate recipe steps with casual dialogue, provides ingredient replacements or advice (using LLM priors), and searches for visual references (in the form of images or videos) on user request. This is a case study in (i) prompting a dialogue LLM (Thoppilan et al., 2022) to produce key phrases that elicit specific multimodal interactions (e.g., video search via a VLM to output visual data), and (ii) using a web crawler (outputs in magenta) as an additional module to retrieve information online. The approach preconditions an LLM with context that includes when and how key phrases should be referenced:

```txt
Alice is a an expert chef that will help Bob prepare a given recipe. If Bob asks for the next step, Alice will respond with "Step: " followed by the next step of the recipe. If Bob does not have the right ingredients, Alice will assist Bob in finding suitable replacements. If Bob asks Alice to describe something that is better shown visually, Alice will say "(image)" followed by a response. Ingredients: {ingredients} Recipe: {recipe}
```

where predicted phrases “(image)” initiates the web crawler to return a recipe video (if it exists), and the VLM runs visual search on it using the LLM response to retrieve related frames e.g.,<sup>2</sup>:

```txt
Alice: Step: In a medium bowl, beat the egg whites until frothy. Bob: what do you mean by medium bowl. Alice: A medium bowl is a bowl that is big enough to fit the ingredients for this recipe and that is comfortable to mix in. Bob: exactly how big should that be. Alice: (image) A medium bowl is a bowl that is big enough to fit the ingredients for this recipe and that is comfortable to mix in. (a) video search {recipe video} with "A medium bowl is a bowl that is..." Bob: ok assume im done what's next. Alice: Step: Keep beating and slowly add the sugar until stiff peaks form. Bob: what's stiff peaks. Alice: Stiff peaks are when egg whites stand up in peaks that do not fall over. Bob: can you show me what that looks like. Alice: (image) Stiff peaks are when egg whites stand up in peaks that do not fall over. (b) video search {recipe video} with "Stiff peaks are when the egg whites..."
```

![](images/57af60a69f183321354535fb700dcf8a28c293ed439234a3a36cf5296b2d0671.jpg)  
Fig. 5: Multimodal prompt engineering VLM, Web Search, and LLM can enable multimodal dialogue applications such as guiding a user through online recipe steps and providing assistive visuals via video search.

# 5.3 ROBOT PERCEPTION & PLANNING:  $\mathsf{USER} + \mathsf{VLM} + \mathsf{LLM} + \mathsf{POLICIES}$

SM-based systems can be used to enable robots to perform language-conditioned tasks. Our example uses a VLM (open vocabulary object detection with ViLD (Gu et al., 2021)) to describe objects in the scene, feeds that description as context to a LLM as a multi-step planner (Ahn et al., 2022; Huang et al., 2022), that then generates the individual steps to be passed to a pretrained language-conditioned robot policy (e.g., models similar to CLIPort (Shridhar et al., 2022; Zeng et al., 2020) for open vocabulary pick-and-place). Steps can be represented in the form of natural language ("Pick the red block and place it on the blue block.") or in the form of pseudocode (to generate text with a fixed template e.g., "robot_pick_and_place("red block", "blue block")"), leveraging LLM capacity to write code. We show this in the context of a simulated environment (shown in Fig. 6) using a UR5 arm and and several objects (blocks, bowls). Distinct from (Ahn et al., 2022), this uses VLM-informed in-context substitution and LLM code generation, rather than joint probabilistic inference.

```python
objects = ["green block", "blue block", "yellow block", "green bowl", "blue bowl", "yellow bowl"]
# stack the blocks on top of each other.
Step 1. robot_pick_and_place("yellow block", "blue block")
Step 2. robot_pick_and_place("green block", "yellow block")
# wait actually undo that last step.
Step 1. robot_pick_and_place("green block", "top left corner")
# put the yellow block in the bowl you think it best fits.
Step 1. robot_pick_and_place("yellow block", "yellow bowl")
```

![](images/c4653dd2b4a36c04a955435f6133ae1fc41abb33eb07430486c9b3033b50436e.jpg)  
Fig. 6: Multimodal prompt engineering VLM, LLM, and language-conditioned policies (via CLIPort (Shridhar et al., 2022)) can enable robots to parse and generate plans from free-form human instructions (in orange).

![](images/1287e2b1aa9a8c566f753e3caf25335f4c383cb63bf87f2c9f72523d4ef0d8ec.jpg)

Chaining this system together expands the set of language-specified tasks beyond the original set of primitives trained by the policy, and enables applications involving human dialogue with the robot.

# 6 UNSUPERVISED EVALUATION FOR MODEL SELECTION

The zero-shot application of SM systems without training data (as in Sec. 5) raises an interesting question which we investigate - how do we evaluate models? To address this, we extend (Strope et al., 2011) (originally used for speech recognition) with a visual-language case study for the application of generating world-state histories (Sec. 5.1), although the method could be adapted for other multimodal tasks as well. Since our metric of interest is the combined performance of e.g., a VLM and a LLM - rather than asking the question: ' (A): how well does a VLM perform in absolute?' for SMs, we can instead ask: ' (B): how well does this VLM compensate for the weaknesses of the LLM?''. (Strope et al., 2011) show that answering (B) correlates well with answers (A), and is useful e.g., for model selection. Specifically, to evaluate a new VLM' for generating language-based world-state history, we first use a baseline VLM paired with the strong LLM (sLLM) to generate pseudo ground truth predictions  $\mathrm{VLM} \times \mathrm{sLLM}$ . We then take both the baseline VLM and new VLM', and pair them with a weak LLM wLLM to generate predictions  $\mathrm{VLM} \times \mathrm{wLLM}$  and  $\mathrm{VLM} \times \mathrm{wLLM}$  respectively. We score these predictions by similarity to the pseudo ground truth  $\mathrm{VLM} \times \mathrm{sLLM}$ . This can be done by by using a similarity-scoring language model e.g., RoBERTa (Liu et al., 2019b).

Tab. 5 shows example results of this analysis with GPT-3 "davinci" as the sLLM, and "curie" as the wLLM, to compare VLM (i.e., CLIP) variants with different backbones: vision transformers (ViT) (Dosovitskiy et al., 2020) and ResNets (RN50) (He et al., 2016) with different model sizes. We find that this method can capture a moderate correlation of ascending performance with increasingly better VLMs (e.g., better variants of CLIP) (Radford et al., 2021), as measured by zero-shot image classification accuracy on ImageNet (Deng et al., 2009) - with correlation coefficients of 0.41 and 0.46 between ImageNet accuracies and mean similarity to

truth models via ViT-B/16 and RN50x16 respectively. Specifically for our SM egocentric perception systems, model combinations that use the same VLM as the one that generates ground truth are biased to produce similar visual grounding results and can exhibit an unfair advantage during the comparisons. Those numbers have been grayed out in Tab. 5.

Tab. 5: Unsupervised evaluation (higher is better) of various VLMs by pairing them with a weak LLM and comparing outputs to a VLM paired with a strong LLM, which provides relative 'truth gradients' that inform how well the VLMs can compensate for the weak LLM. These results show that the best VLM for this SM system correlates with the best zero-shot ImageNet model.  

<table><tr><td rowspan="2">Truth Models</td><td colspan="5">VLM (CLIP) Variants + Weak LLM</td></tr><tr><td>RN50x4</td><td>RN50x16</td><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-L/14</td></tr><tr><td>GPT-3 + ViT-B/16</td><td>0.628</td><td>0.646</td><td>0.686</td><td>0.861</td><td>0.704</td></tr><tr><td>GPT-3 + RN50x16</td><td>0.667</td><td>0.851</td><td>0.689</td><td>0.655</td><td>0.704</td></tr><tr><td>ImageNet Accuracy</td><td>65.8</td><td>70.5</td><td>63.2</td><td>68.6</td><td>76.2</td></tr><tr><td>Size (# params)</td><td>178M</td><td>291M</td><td>151M</td><td>150M</td><td>427M</td></tr></table>

# 7 LIMITATIONS AND DISCUSSION

SM systems leverage in-context multimodal prompt engineering as a means to compose multiple large pretrained models to make predictions for new multimodal tasks, that each model may otherwise struggle to do independently. These systems can (i) serve as strong zero-shot baselines that are competitive with state-of-the-art on standard multimodal benchmarks, (ii) adapt large pretrained models for multimodal tasks while retaining their robustness to distribution shifts (known to deteriorate after finetuning (Wortsman et al., 2021)), and (iii) present practical application advantages in domains that are restricted by data scarcity, training compute, or model access. Such systems are however, subject to inheriting the limitations of the models on which they are built. For example, our captioning systems use VLMs (e.g., CLIP) predominantly as zero-shot image classifiers, so the extent to which they can express visual information is more contextual than fine-grained. However, we expect that by replacing the VLMs with more expressive ones (e.g., ViLD (Gu et al., 2021), LSeg (Li et al., 2022), or Flamingo (Alayrac et al., 2022)), SMs may likewise benefit in the capacity to express details. Future work may also involve meta-learning the multimodal exchanges themselves, expanding the intermediate representations from discrete to continuous Gal et al. (2022), or extending them to include additional modalities beyond text, e.g., passing images between modules.

Reproducibility statement. SM-based systems are in part by nature, simple and easy to reproduce. We provide open-source implementations with code that can be directly run in the browser with Colab. See sites.google.com/view/socraticmodels for anonymous versions.

Ethic statement. Multimodal systems with natural language as middleware provides an interpretable window into the behavior of the systems (even for non-experts). The barrier of entry is small: SMs can be engineered to capture new functionalities with minimal additional resources, and tackle applications that have traditionally been data-scarce. This can be enabling, but also raises potential risks, since it increases the flexibility of unintended end use applications, and should be carefully monitored over time. It is also important to note that the system may generate results that reflect unwanted biases found in the Internet-scale data on which incorporated models are trained, and should be used with caution (and checked for correctness) in downstream applications. We welcome broad discussion on how to maximize potential positive impacts (enabling broad, new multimodal applications, with minimal resources) while minimizing the capabilities of bad actors.

# REFERENCES

Speech-to-text: Automatic speech recognition — google cloud. https://cloud.google.com/speech-to-text. Accessed: 2022-05-13.  
Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. To index or not to index: Optimizing exact maximum inner product search. In 35th IEEE International Conference on Data Engineering, ICDE 2019, Macao, China, April 8-11, 2019, pp. 1250-1261. IEEE, 2019. doi: 10.1109/ICDE.2019.00114. URL https://doi.org/10.1109/ICDE.2019.00114.  
Pranav Agarwal, Alejandro Betancourt, Vana Panagiotou, and Natalia Diaz-Rodriguez. Egoshots, an egovision life-logging dataset and semantic fidelity metric to evaluate diversity in image captioning models. arXiv preprint arXiv:2003.11743, 2020.  
Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, and Mengyuan Yan. Do as i can and not as i say: Grounding language in robotic affordances. In arXiv preprint arXiv:2022.00000, 2022.  
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.  
Evlampios Apostolidis, Eleni Adamantidou, Alexandros I Metsai, Vasileios Mezaris, and Ioannis Patras. Video summarization using deep neural networks: A survey. Proceedings of the IEEE, 109(11):1838-1863, 2021.  
Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1728-1738, 2021.  
Sven Bambach, Stefan Lee, David J Crandall, and Chen Yu. Lending a hand: Detecting hands and recognizing activities in complex egocentric interactions. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 1949-1957, 2015.  
Ankur Bapna, Colin Cherry, Yu Zhang, Ye Jia, Melvin Johnson, Yong Cheng, Simran Khanuja, Jason Riesa, and Alexis Conneau. mslam: Massively multilingual joint pre-training for speech and text. arXiv preprint arXiv:2202.01374, 2022.  
Mauro Barbieri, Lalitha Agnihotri, and Nevenka Dimitrova. Video summarization: methods and landscape. In _Internet Multimedia Management Systems IV_, volume 5242, pp. 1-13. International Society for Optics and Photonics, 2003.  
Yoshua Bengio, Pascal Lamblin, Dan Popovici, and Hugo Larochelle. Greedy layer-wise training of deep networks. Advances in neural information processing systems, 19, 2006.  
Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
Rich Caruana. Multitask learning. Machine learning, 28(1):41-75, 1997.  
Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zisserman. Vggsound: A large-scale audio-visual dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 721-725. IEEE, 2020.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  
Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.  
Xing Cheng, Hezheng Lin, Xiangyu Wu, Fan Yang, and Dong Shen. Improving video-text retrieval by multi-stream corpus alignment and dual softmax loss. arXiv preprint arXiv:2109.04290, 2021.  
Krzysztof Choromanski, Haoxian Chen, Han Lin, Yuanzhe Ma, Arijit Sehanobish, Deepali Jain, Michael S. Ryoo, Jake Varley, Andy Zeng, Valerii Likhosherstov, Dmitry Kalashnikov, Vikas Sindhwani, and Adrian Weller. Hybrid random features. to appear in ICLR 2022, abs/2110.04367, 2021a. URL https://arxiv.org/abs/2110.04367.  
Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021b. URL https://openreview.net/forum?id=Ua6zuk@WRH.  
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.  
Andrew M Dai and Quoc V Le. Semi-supervised sequence learning. Advances in neural information processing systems, 28, 2015.  
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 720-736, 2018.  
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Rescaling egocentric vision. arXiv preprint arXiv:2006.13256, 2020.  
Ana Garcia Del Molino, Cheston Tan, Joo-Hwee Lim, and Ah-Hwee Tan. Summarization of egocentric videos: A comprehensive survey. IEEE Transactions on Human-Machine Systems, 47(1):65-76, 2016.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. IEEE, 2009.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning Zhang, Eric Tzeng, and Trevor Darrell. Decaf: A deep convolutional activation feature for generic visual recognition. In International conference on machine learning, pp. 647-655. PMLR, 2014.  
Jianfeng Dong, Xirong Li, and Cees GM Snoek. Predicting visual features from text for image and video caption retrieval. IEEE Transactions on Multimedia, 20(12):3377-3388, 2018.  
Jianfeng Dong, Xirong Li, Chaoxi Xu, Shouling Ji, Yuan He, Gang Yang, and Xun Wang. Dual encoding for zero-example video retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9346-9355, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
Chenyou Fan, Zehua Zhang, and David J Crandall. Deepdiary: Lifelogging image captioning and summarization. Journal of Visual Communication and Image Representation, 55:40-55, 2018.  
Han Fang, Pengfei Xiong, Luhui Xu, and Yu Chen. Clip2video: Mastering video-text retrieval via image clip. arXiv preprint arXiv:2106.11097, 2021.  
Alireza Fathi, Ali Farhadi, and James M Rehg. Understanding egocentric activities. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 407-414, 2011.  
Peter Florence, Lucas Manuelli, and Russ Tedrake. Self-supervised correspondence in visuomotor policy learning. IEEE Robotics and Automation Letters, 5(2):492-499, 2019.  
Antonino Furnari and Giovanni Maria Farinella. What would you expect? anticipating egocentric actions with rolling-unrolling lstms and modality attention. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6252-6261, 2019.  
Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.  
Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723, 2020.  
Zijian Gao, Jingyu Liu, Sheng Chen, Dedan Chang, Hao Zhang, and Jinwei Yuan. Clip2tv: An empirical study on transformer-based methods for video-text retrieval. arXiv preprint arXiv:2111.05610, 2021.  
Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim. First-person hand action benchmark with rgb-d videos and 3d hand pose annotations. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 409-419, 2018.  
Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 580-587, 2014.  
Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. arXiv preprint arXiv:2110.07058, 2021.  
Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921, 2021.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Geoffrey E Hinton, Simon Osindero, and Yee-Whye Teh. A fast learning algorithm for deep belief nets. Neural computation, 18(7):1527-1554, 2006.  
Minh Hoai and Fernando De la Torre. Max-margin early event detectors. International Journal of Computer Vision, 107(2):191-202, 2014.  
Ronghang Hu and Amanpreet Singh. Transformer is all you need: Multimodal multitask learning with a unified transformer. arXiv e-prints, pp. arXiv-2102, 2021.  
Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. arXiv preprint arXiv:2201.07207, 2022.  
Dieuwke Hupkes, Verna Dankers, Mathijs Mul, and Elia Bruni. Compositionality decomposed: How do neural networks generalise? Journal of Artificial Intelligence Research, 67:757-795, 2020.  
Shahram Izadi, David Kim, Otmar Hilliges, David Molyneaux, Richard Newcombe, Pushmeet Kohli, Jamie Shotton, Steve Hodges, Dustin Freeman, Andrew Davison, et al. Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera. In Proceedings of the 24th annual ACM symposium on User interface software and technology, pp. 559-568, 2011.  
Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason Baldridge. Mural: multimodal, multitask retrieval across languages. arXiv preprint arXiv:2109.05125, 2021.

Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pp. 4904-4916. PMLR, 2021.  
Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181-214, 1994.  
S Karpagavalli and Edy Chandra. A review on automatic speech recognition architecture and approaches. International Journal of Signal Processing, Image Processing and Pattern Recognition, 9(4):393-404, 2016.  
Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3128-3137, 2015.  
Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, and Dima Damen. Epic-fusion: Audio-visual temporal binding for egocentric action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5492-5501, 2019.  
Daniel Keysers, Nathanael Scharli, Nathan Scales, Hylke Buisman, Daniel Furrer, Sergii Kashubin, Nikola Momchev, Danila Sinopalnikov, Lukasz Stafiniak, Tibor Tihon, et al. Measuring compositional generalization: A comprehensive method on realistic data. arXiv preprint arXiv:1912.09713, 2019.  
Kris M Kitani, Takahiro Okabe, Yoichi Sato, and Akihiro Sugimoto. Fast unsupervised ego-action learning for first-person sports videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3241-3248, 2011.  
Kris M Kitani, Brian D Ziebart, James Andrew Bagnell, and Martial Hebert. Activity forecasting. In European Conference on Computer Vision (ECCV), pp. 201-214, 2012.  
Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916, 2022.  
Elisa Kreiss, Noah D Goodman, and Christopher Potts. Concadia: Tackling image accessibility with context. arXiv preprint arXiv:2104.08376, 2021.  
Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4. International Journal of Computer Vision, 128(7):1956-1981, 2020.  
Yong Jae Lee and Kristen Grauman. Predicting important objects for egocentric video summarization. International Journal of Computer Vision, 114(1):38-55, 2015.  
Yong Jae Lee, Joydeep Ghosh, and Kristen Grauman. Discovering important people and objects for egocentric video summarization. In IEEE conference on computer vision and pattern recognition (CVPR), pp. 1346-1353, 2012.  
Jie Lei, Licheng Yu, Tamara L Berg, and Mohit Bansal. What is more likely to happen next? video-and-language future event prediction. arXiv preprint arXiv:2010.07999, 2020.  
Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.  
Cheng Li and Kris M Kitani. Pixel-level hand detection in ego-centric videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3570-3577, 2013.  
Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in Neural Information Processing Systems, 34, 2021a.  
Yanghao Li, Tushar Nagarajan, Bo Xiong, and Kristen Grauman. Ego-exo: Transferring visual representations from third-person to first-person videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6943-6953, 2021b.  
Yin Li, Miao Liu, and James M Rehg. In the eye of beholder: Joint learning of gaze and actions in first person video. In Proceedings of the European conference on computer vision (ECCV), pp. 619-635, 2018.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dólar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740-755. Springer, 2014.

Xing Lin, Yair Rivenson, Nezih T Yardimci, Muhammed Veli, Yi Luo, Mona Jarrahi, and Aydogan Ozcan. All-optical machine learning using diffractive deep neural networks. Science, 361(6406):1004-1008, 2018.  
Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. Use what you have: Video retrieval using representations from collaborative experts. BMVC, 2019a.  
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019b.  
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.  
Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, and Tianrui Li. Clip4clip: An empirical study of clip for end to end video clip retrieval. arXiv preprint arXiv:2104.08860, 2021.  
Minghuang Ma, Haoqi Fan, and Kris M Kitani. Going deeper into first-person activity recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1894-1903, 2016.  
Christopher D Manning, Mihai Surdeanu, John Bauer, Jenny Rose Finkel, Steven Bethard, and David McClosky. The stanford corenlp natural language processing toolkit. In Proceedings of 52nd annual meeting of the association for computational linguistics: system demonstrations, pp. 55-60, 2014.  
Saeed Masoudnia and Reza Ebrahimpour. Mixture of experts: a literature survey. Artificial Intelligence Review, 42(2):275-293, 2014.  
Grégoire Mesnil, Yann Dauphin, Xavier Glorot, Salah Rifai, Yoshua Bengio, Ian Goodfellow, Erick Lavoie, Xavier Muller, Guillaume Desjardins, David Warde-Farley, et al. Unsupervised and transfer learning challenge: a deep learning approach. In Proceedings of ICML Workshop on Unsupervised and Transfer Learning, pp. 97-110. JMLR Workshop and Conference Proceedings, 2012.  
Antoine Miech, Jean-Baptiste Alayrac, Lucas Smaira, Ivan Laptev, Josef Sivic, and Andrew Zisserman. End-to-end learning of visual representations from uncurated instructional videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9879-9889, 2020.  
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.  
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettle-moyer. Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837, 2022.  
Niluthpol Chowdhury Mithun, Juncheng Li, Florian Metze, and Amit K Roy-Chowdhury. Learning joint embedding with multimodal cues for cross-modal video-text retrieval. In Proceedings of the 2018 ACM on International Conference on Multimedia Retrieval, pp. 19-27, 2018.  
Ron Mokady, Amir Hertz, and Amit H Bermano. Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734, 2021.  
Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and Andrew Y Ng. Multimodal deep learning. In ICML, 2011.  
Andreea-Maria Oncescu, A Koepke, João F Henriques, Zeynep Akata, and Samuel Albanie. Audio retrieval with natural language queries. arXiv preprint arXiv:2105.02192, 2021.  
Jonas Oppenlaender. Prompt engineering for text-based generative art. arXiv preprint arXiv:2204.13988, 2022.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Preprint, 2022.  
Devshree Patel, Ratnam Parikh, and Yesha Shastri. Recent advances in video question answering: A review of datasets and methods. In International Conference on Pattern Recognition, pp. 339-356. Springer, 2021.  
Mandela Patrick, Po-Yao Huang, Yuki Asano, Florian Metze, Alexander Hauptmann, Joao Henriques, and Andrea Vedaldi. Support-set bottlenecks for video-text representation learning. arXiv preprint arXiv:2010.02824, 2020.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532-1543, 2014.  
Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. 2018.  
Fabio Petroni, Tim Rocttäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. Language models as knowledge bases? arXiv preprint arXiv:1909.01066, 2019.  
Hamed Pirsiavash and Deva Ramanan. Detecting activities of daily living in first-person camera views. In 2012 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2847-2854, 2012.  
Jesús Andrés Portillo-Quintero, José Carlos Ortiz-Bayliss, and Hugo Terashima-Marín. A straightforward framework for video retrieval using clip. In *Mexican Conference on Pattern Recognition*, pp. 3–12. Springer, 2021.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748-8763. PMLR, 2021.  
Rajat Raina, Alexis Battle, Honglak Lee, Benjamin Packer, and Andrew Y Ng. Self-taught learning: transfer learning from unlabeled data. In Proceedings of the 24th international conference on Machine learning, pp. 759-766, 2007.  
Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822, 2018.  
Prajit Ramachandran, Peter J Liu, and Quoc V Le. Unsupervised pretraining for sequence to sequence learning. arXiv preprint arXiv:1611.02683, 2016.  
Hubert Ramsauer, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Lukas Gruber, Markus Holzleitner, Thomas Adler, David P. Kreil, Michael K. Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. Hopfield networks is all you need. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=tL89RnzIiCd.  
Ankit Singh Rawat, Jiecao Chen, Felix X. Yu, Ananda Theertha Suresh, and Sanjiv Kumar. Sampled softmax with random fourier features. In Hanna M. Wallach, Hugo Larochelle, Alina Beygelzimer, Florence d'Alché-Buc, Emily B. Fox, and Roman Garnett (eds.), Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pp. 13834-13844, 2019. URL https://proceedings.neurips.cc/paper/2019/ hash/e43739bba7cdb577e9e3e4e42447f5a5-AAbstract.html.  
Albert Reuther, Peter Michaleas, Michael Jones, Vijay Gadepally, Siddharth Samsi, and Jeremy Kepner. Survey of machine learning accelerators. In 2020 IEEE high performance extreme computing conference (HPEC), pp. 1-12. IEEE, 2020.  
Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond the few-shot paradigm. In *Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems*, pp. 1-7, 2021.  
Nicholas Rhinehart and Kris M Kitani. First-person activity forecasting with online inverse reinforcement learning. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 3696-3705, 2017.  
Michael S Ryoo. Human activity prediction: Early recognition of ongoing activities from streaming videos. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 1036-1043, 2011.  
Michael S Ryoo and Larry Matthies. First-person activity recognition: What are they doing to me? In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2730-2737, 2013.  
Michael S Ryoo, Brandon Rothrock, and Larry Matthies. Pooled motion features for first-person videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 896-904, 2015.  
Timo Schick and Hinrich Schütze. It's not just size that matters: Small language models are also few-shot learners. arXiv preprint arXiv:2009.07118, 2020.

Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus, and Yann LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. arXiv preprint arXiv:1312.6229, 2013.  
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2556-2565, 2018.  
Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980, 2020.  
Mohit Shridhar, Lucas Manuelli, and Dieter Fox. CIoport: What and where pathways for robotic manipulation. In Conference on Robot Learning, pp. 894-906. PMLR, 2022.  
Anshumali Shrivastava and Ping Li. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). In Zoubin Ghahramani, Max Welling, Corinna Cortes, Neil D. Lawrence, and Kilian Q. Weinberger (eds.), Advances in Neural Information Processing Systems 27: Annual Conference on Neural Information Processing Systems 2014, December 8-13 2014, Montreal, Quebec, Canada, pp. 2321-2329, 2014. URL https://proceedings.neurips.cc/paper/2014/hash/310ce61c90f3a46e340ee8257bc70e93-Abstract.html.  
Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari. Charades-ego: A large-scale dataset of paired third and first person videos. arXiv preprint arXiv:1804.09626, 2018.  
Lucas Smaira, João Carreira, Eric Noland, Ellen Clancy, Amy Wu, and Andrew Zisserman. A short note on the kinetics-700-2020 human action dataset. arXiv preprint arXiv:2010.10864, 2020.  
Haoyu Song, Li Dong, Wei-Nan Zhang, Ting Liu, and Furu Wei. Clip models are few-shot learners: Empirical studies on vqa and visual entailment. arXiv preprint arXiv:2203.07190, 2022a.  
Yunfeng Song, Xiaochao Fan, Yong Yang, Ge Ren, and Weiming Pan. Large pretrained models on multimodal sentiment analysis. In Artificial Intelligence in China, pp. 506-513. Springer, 2022b.  
Ekaterina H Spriggs, Fernando De La Torre, and Martial Hebert. Temporal segmentation and activity classification from first-person sensing. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 17-24, 2009.  
Brian Strope, Doug Beeferman, Alexander Gruenstein, and Xin Lei. Unsupervised testing strategies for asr. 2011.  
Yixuan Su, Tian Lan, Yahui Liu, Fangyu Liu, Dani Yogatama, Yan Wang, Lingpeng Kong, and Nigel Collier. Language models can see: Plugging visual controls in text generation. arXiv preprint arXiv:2205.02655, 2022.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27, 2014.  
Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. arXiv preprint arXiv:2202.05263, 2022.  
Ian Tenney, Dipanjan Das, and Ellie Pavlick. Bert rediscovers the classical nlp pipeline. arXiv preprint arXiv:1905.05950, 2019.  
Yoad Tewel, Yoav Shalev, Idan Schwartz, and Lior Wolf. Zero-shot image-to-text generation for visual-semantic arithmetic. arXiv preprint arXiv:2111.14447, 2021.  
Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.  
Sebastian Thrun. Lifelong learning algorithms. In Learning to learn, pp. 181-209. Springer, 1998.  
Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34: 200-212, 2021.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566-4575, 2015.  
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pp. 1096-1103, 2008.  
Carl Vondrick, Hamed Piriavash, and Antonio Torralba. Anticipating visual representations from unlabeled video. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 98-106, 2016.  
Qiang Wang, Yanhao Zhang, Yun Zheng, Pan Pan, and Xian-Sheng Hua. Disentangled representation learning for text-video retrieval. arXiv preprint arXiv:2203.07111, 2022.  
Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision. arXiv preprint arXiv:2108.10904, 2021.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.  
Mitchell Wortsman, Gabriel Ilharco, Mike Li, Jong Wook Kim, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zero-shot models. arXiv preprint arXiv:2109.01903, 2021.  
Baoyuan Wu, Weidong Chen, Yanbo Fan, Yong Zhang, Jinlong Hou, Jie Liu, and Tong Zhang. Tencent ml-images: A large-scale multi-label image database for visual representation learning. IEEE Access, 7:172683-172693, 2019.  
Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello. Wav2clip: Learning robust audio representations from clip. arXiv preprint arXiv:2110.11499, 2021a.  
Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021b.  
Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context learning as implicit bayesian inference. arXiv preprint arXiv:2111.02080, 2021.  
Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. arXiv preprint arXiv:2109.14084, 2021.  
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5288-5296, 2016.  
Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang. An empirical study of gpt-3 for few-shot knowledge-based vqa. arXiv preprint arXiv:2109.05014, 2021.  
Youngjae Yu, Jongseok Kim, and Gunhee Kim. A joint sequence fusion model for video question answering and retrieval. In Proceedings of the European Conference on Computer Vision (ECCV), pp. 471-487, 2018.  
Kevin Zakka, Andy Zeng, Pete Florence, Jonathan Tompson, Jeannette Bohg, and Debidatta Dwibedi. Xirl: Cross-embodiment inverse reinforcement learning. In Conference on Robot Learning, pp. 537-546. PMLR, 2022.  
Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In European conference on computer vision, pp. 818-833. Springer, 2014.  
Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. arXiv preprint arXiv:2201.02639, 2022.  
Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong, Ivan Krasin, Dan Duong, Vikas Sindhwani, et al. Transporter networks: Rearranging the visual world for robotic manipulation. arXiv preprint arXiv:2010.14406, 2020.  
Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. arXiv preprint arXiv:2111.07991, 2021.  
Bolei Zhou, Aditya Khosla, Agata Lapedriza, Antonio Torralba, and Aude Oliva. Places: An image database for deep scene understanding. arXiv preprint arXiv:1610.02055, 2016.
