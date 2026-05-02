# LOST IN TRANSLATION: CONCEPTUAL BLIND SPOTS IN TEXT-TO-IMAGE DIFFUSION MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Advancements in text-to-image diffusion models have broadened both research and practical applications. However, these models often lead to misalignment issues between text and images. Existing work primarily focuses on problems limited to two objects. We propose a more complex structure, as exemplified by "a tea cup of iced coke." In this instance, an object never previously encountered, a "glass cup," replaces the expected teacup, primarily due to biases in the training dataset. We propose a new classification for such visual-textual misalignment errors, termed Conceptual Blind Spots (CBS). In this study, we employ large language models (LLMs) and diffusion models to thoroughly investigate the diagnosis and remediation of CBS. We develop an automated pipeline that leverages the LLM's proficiency in semantic layering to create a Mixture of Concept Experts (MoCE) framework. To disentangle overlapping concepts, we input them into the models sequentially. Our MoCE is specifically designed to alleviate conceptual ambiguities during the diffusion model's denoising stages. Empirical assessments confirm the effectiveness of our approach, substantially reducing CBS errors and enhancing the robustness and versatility of text-to-image diffusion models.

# 1 INTRODUCTION

In recent years, text-to-image synthesis (Mansimov et al., 2015; Reed et al., 2016; Zhang et al., 2017; Xu et al., 2017; Li et al., 2019; Ramesh et al., 2021; Ding et al., 2021; Wu et al., 2022; Yu et al., 2022; Nichol et al., 2021; Sahara et al., 2022; Rombach et al., 2022; Ramesh et al., 2022; Song et al., 2023) via diffusion models has made remarkable strides, excelling in generating photorealistic images from textual cues (Rombach et al., 2022; Podell et al., 2023). Despite their wide-ranging applications—from digital media to visual arts—a significant limitation exists: the issue of visual-textual misalignment, where certain elements in the input text are overlooked during image synthesis, as our finding in Figure 1 demonstrates. When given inputs like "iced coke" and "tea cup", these models often produce images that only include "iced coke" ignoring the "tea cup" element. More interestingly, the object "glass" that has never been mentioned in the text appears in the image, taking the place of the expected "tea cup". In our study, We term the combination of two main objects (e.g.  $A$ : "iced coke",  $B$ : "tea cup") a concept pair. Though several studies touch upon the diffusion model's difficulty in generating the correct image while encountering concept pairs, this challenge persists even when sophisticated prompt engineering strategies are employed, indicating a more fundamental issue at play.

In this study, we present an in-depth empirical investigation targeting the conceptual blind spot (CBS) problem, which denotes that when the two objects were presented in a concept pair, either one of them (object  $\mathcal{A}$  or  $\mathcal{B}$ ) is encroached upon and consequently disappears in the image. The CBS problem has become a persistent bottleneck that hinders advancements in text-to-image diffusion techniques. More importantly, unlike basic works that only focus on the mutual infringement of  $\mathcal{A}$  and  $\mathcal{B}$  (Wang et al., 2023; Du et al., 2023; Liu et al., 2022; Li et al., 2023; Chefer et al., 2023), CBS problem involves a latent object  $\mathcal{C}$  that has never been mentioned in the text requirements, such as the "glass" in "a tea cup of iced coke". Its emergence leads to the disappearance of the originally anticipated object  $\mathcal{A}$  ("tea cup") in the image, which is due to the strong binding of  $\mathcal{C}$  ("glass") with  $\mathcal{B}$  ("iced coke") in the training dataset. Our work pivots towards systematically researching the CBS dilemma. Utilizing the reasoning prowess of cutting-edge Large Language Models (LLMs), our

![](images/cab4fda55db2a0f48dff5b52d2e6b8c572cfb189b3e502d97c226068c302788f.jpg)  
Atea cup of iced coke

![](images/ed884e1c5b7291e7fbb8cb7e8af6b9d0405885fac09d4984b74829989c54ecba.jpg)  
Green Tea in Red Solo Cup

![](images/4f5590fa0331d933b78fbe66ee66c9868dfd6fa35777a7d4e1727c18cc5b0b95.jpg)  
Pancakes in a cup

![](images/46cdb4b4e69ac8d9605776b3781459675ec1b6812c2d693a32ab4070b6fce771.jpg)  
A stone lion at Paris

![](images/8c91b3c4fcaa5696409a024638980efbc6c4c984cf30d7c5981bbc715ac76e2d.jpg)  
Watermelon on a lemon tree

![](images/3eb83e9f806cee460d6d3e3c0356af65e06fb02e5a9eac91a526aa41c4c74978.jpg)  
Crocodile in front of Neuschwanstein

![](images/71e3f2d2cbf941fd4d8f333440cd363fb830ff373fcb62f9c262b5d1a9d8e6be.jpg)  
A tea cup of iced coke  
Figure 1: We display the issue of concept misalignment. In the first row of the images, even the most advanced text-to-image models fail to faithfully generate the specified concepts. We have developed a fully automated pipeline to explore this issue and proposed a hotfix to simply fix it.

![](images/9ac8ad01be63bbfae5b5cb6e00927e1dfe5bb925cd4c348592e5701dd59c31ab.jpg)  
Green Tea in Red Solo Cup

![](images/cdbebb124216c7d05b74b069a16bd5d422aab2be3529acda50083e1ca0f9515d.jpg)  
Pancakes in a cup

![](images/1f39cb824f58f79a301a3118aa136abc18279cac4bcfbcc6a5bd485786b4567c.jpg)  
A stone lion at Paris

![](images/8a8a2c1bc0d7cda39c555d8b97138ec248c3abfcd532032875eb18405a06aa6e.jpg)  
Watermelon on a lemon tree

![](images/92bebfed40d17896b9e486e83e369579899eb32f75465d4486a612159131efe4.jpg)  
Crocodile in front of Neuschwanstein Castle  
Castle

automated platform accurately identifies and reconciles concept pairings that are prone to generative distortions.

To this end, we adopt Socratic Reasoning (Dong et al., 2023), a process of problem identification, understanding and reasoning using Large Language Models (LLMs). Human experts patiently guide LLMs to discover, understand, and delve deeper into the problem of the issue step by step. We leverage state-of-the-art LLMs to discern and generate an example set of concept pairs of CBS. Upon isolating these concept pairs, we employ a comprehensive evaluation pipeline, employing state-of-the-art diffusion models, to produce high-quality synthesized image samples. Expert human evaluators then carefully assess the resulting images, eliminating concept pairs that display inconsequential discrepancies. We argue that disjunctions between visual and textual components can originate from the complexities of feature space semantics or from biased training sets. To quantitatively assess the misalignments between generated images and specified prompts, we introduce an innovative composite metric that merges proven benchmarks, such as Clipscore (Hessel et al., 2021) and Image-Reward (Xu et al., 2023). This unified metric acts as a reliable gauge for ascertaining the harmony between generated images and their associated textual metadata, offering strong support for our hotfix.

To mitigate the discovered conceptual blind spots, we introduce a more flexible query mechanism, Mixture of Concept Experts (MoCE), to handle intricate constructs like "a tea cup filled with iced coke". Utilizing LLMs, we fine-tune concept sequencing for enhanced image synthesis and optimize dynamic scheduling via heuristic methodologies. Such targeted enhancements not only optimize computational resource allocation but also significantly ameliorate issues related to conceptual discordance. The empirical case studies confirm that our method not only greatly alleviates the CBS challenge but also enhances the application and flexibility of text-to-image diffusion techniques across diverse fields.

We summarize our contributions in the following three aspects:

- We investigate the neglected issue of conceptual blind spots, such as "a tea cup of iced coke", within existing text-to-image diffusion models. Utilizing LLMs for exploration and evaluation, we expose inherent limitations in both architecture and training paradigms.  
- We introduce a comprehensive dataset specifically designed to address conceptual blind spots. Employing Socratic-style reasoning, we also outline an automated schema for LLMs to further augment the proposed dataset, stimulating innovation in diverse future directions.  
- We propose an adaptive prompting mechanism, diverging from the conventional predefined schemes. This adaptability effectively mitigates conceptual discrepancies and visual-textual discordance, thereby enhancing image fidelity and interpretive accuracy.

# 2 RELATED WORK

Diffusion Models Diffusion models (Ho et al., 2020) predict noise in noisy images and produce high-quality outputs after training. Having gained considerable attention, they are now considered the state-of-the-art in image generation. These models find applications in various domains, such as label-to-image synthesis (Dhariwal & Nichol, 2021; Rombach et al., 2022), text-to-image  $(ti)$  generation (Ramesh et al., 2022; Rombach et al., 2022; Podell et al., 2023), image editing (Couairon et al., 2022; Meng et al., 2021; Kawar et al., 2023), and video generation (Ho et al., 2022). Specifically, the synthesis of  $ti$  images is of practical importance. Open-source and commercial solutions like Stable Diffusion (Rombach et al., 2022; Podell et al., 2023) and Mid-journey  $(mj)$  (Midjourney, 2023) have achieved success, bolstered by advancements in neural network architectures (Ronneberger et al., 2015; Vaswani et al., 2017) and a large dataset of image-text pairs (Sharma et al., 2018; Schuhmann et al., 2022). Nevertheless, the question of whether these models truly innovate or merely generate combinations encountered in their training data remains unresolved. This study investigates this issue by examining text-to-image diffusion models in the context of unconventional concept pairings.

Misalignment Problems While prominent generative models frequently produce high-quality, realistic images, they struggle with certain concept combinations. Such models usually mimic combinations seen in training data. Prior work has emphasized spatial conflicts where multiple entities coexist in close proximity (Wang et al., 2023; Du et al., 2023; Liu et al., 2022; Li et al., 2023; Chefer et al., 2023). Distinct from these investigations, our focus lies on conceptual blind spots, illustrated by phrases such as "a tea cup of iced coke." Through rigorous experimentation, we investigate this challenge and introduce a benchmark alongside a baseline solution.

# 3 BENCHMARK: COLLECTING DATA ON CONCEPTUAL BLIND SPOTS (CBS)

In this section, we describe the construction of our dataset, CBS, focusing on the conceptual gaps in image generation for two distinct concepts,  $\mathcal{A}$  and  $\mathcal{B}$ , and the influence of the latent concept  $\mathcal{C}$ . Collecting such data requires an extensive range of knowledge, which is typically unrealistic for the human brain. Therefore, we have constructed Socratic Reasoning process through the collaboration of generative models and human experts. Specifically, drawing inspiration from Dong et al. (2023), a Socratic Reasoning system is implemented using GPT-3.5, text-to-image models, and human researchers to investigate issues of visual-textual misalignment. The framework comprises:

- GPT-3.5 serves as an expert rich in information but lacking comprehensive wisdom. It elaborates on issues when prompted, leveraging its intrinsic knowledge and insights.  
- Text-to-image models (T2I models) functions as an experienced artist with limited creative abilities. Its main role is to generate images based on understood prompts. We utilize both Midjourney and SDXL-1.0, confirming the consistent occurrence of the problem.  
- Human researchers act as discerning advisors, overseeing the outputs of GPT-3.5 and T2I models. They provide guidance to GPT-3.5 and extract insights to deepen their understanding of the misalignment problem, thereby setting the stage for future refinements.

To assess the accuracy of the generated images, human researchers introduce a metric system. Specifically, T2I models generate 20 images per concept pair for evaluation. In this system, zero correct images correspond to Level 5,  $1 \sim 5$  correct images to Level 4,  $6 \sim 10$  to Level 3,  $11 \sim 15$  to Level 2, and  $16 \sim 20$  to Level 1.

The Socratic Reasoning process begins with human researchers identifying error-prone concept pairs. Subsequently, GPT-3.5 generates additional concept pairs. Human researchers then prompt GPT-3.5 to produce new patterns, culminating in a closed-loop Socratic Reasoning system after integrating various patterns. These stages are distributed across six rounds. The idea of the construction of these six rounds is composed using a divide-and-conquer strategy. In the process, we leverage human experts' creativity and induction to initialize the few-shot data and categorize the patterns and the extensive knowledge possessed by GPT to expand our dataset. Here, human experts are involved only in the initial two stages. Afterwards, GPT operates like a snowball effect, continually expanding the scope of data collection.

Table 1: Categories and patterns summarized manually by human researchers.  

<table><tr><td>Category</td><td>Pattern</td></tr><tr><td>I: Dominant Association</td><td>Local cuisine and non-native locationEndangered species and historical sitesAnimal and incorrect external coveringBeverage and erroneous container</td></tr><tr><td>II: Absence of Dominant Association</td><td>Inappropriate tablewares and foodVisual nouns of similar shapesAnimal nouns with similar appearancesMismatched outfits</td></tr></table>

# 3.1 ROUND 1 - IDENTIFYING 259 CONCEPT PAIRS FROM TEXT-TO-IMAGE DATASETS

Initially, 259 concept pairs are identified through the collaborative efforts of human researchers, supported by GPT-3.5. These pairs undergo rigorous manual classification. Specific examples, such as "a tea cup of iced coke", inform our investigation into root causes. Prior to expanding the dataset with analogous concept pairs for comprehensive analysis, it is crucial to understand the fundamental patterns exhibited in typical concept pairs. To this end, human researchers examine various visual scenes in well-known text-to-image datasets such as Laion2B-en $^1$  and MJ User Prompts & Generated Images Dataset $^2$  to identify concept pairs susceptible to similar errors.

Human researchers come up with 50 concept pairs as initial seeds for potential erroneous outputs. These seed pairs are then segregated by human researchers into two primary categories and eight specific patterns. This classification approach aims for conceptual clarity and addresses the pervasive issue of concurrency in such models. As delineated in Table 1, the updated text offers a comprehensive categorization criterion, explores the challenges associated with each category, and establishes a systematic framework for subsequent evaluations:

Category I: Dominant Association In this category, one concept (e.g.,  $\mathcal{B}$ ) in the model's training dataset is strongly bound to a latent concept  $\mathcal{C}$ , resulting in the influence on another expected object  $\mathcal{A}$  in the image. For instance, the  $\mathcal{A}$  ("iced coke") contain the underlying concept of  $\mathcal{C}$  ("glass cup"), encompassing the influence of  $\mathcal{B}$  ("tea cup")

Category II: Absence of Dominant Association This category includes instances where no apparent hierarchical or dominant relationship exists between the entities. The two entities often exist only in mutual encroachment or integration.

Based on this classification, we can thus aim our overall goal of the dataset to collect CBS samples with the inclusion of a latent concept  $\mathcal{C}$  mentioned above, which is a focus not emphasized in previous work. Besides that, a method is devised to collect a substantial number of incorrect concept pairs. Human researchers employ the GPT-3.5 API to generate various concept pairs that align with each pattern in Table 1. Using this approach, GPT-3.5 generates 270 concept pairs, of which 209 result in image synthesis following prompts formatted as “ $\mathcal{A}, \mathcal{B}$ ”. Each synthesized image contains errors. This initial observation validates our hypotheses concerning error patterns depicted in Table 1. Including the 50 concept pairs manually extracted, a total of 259 error-prone concept pairs are identified in this round.

# 3.2 ROUND 2 - RIGOROUS VERIFICATION ON DISCOVERED 259 CONCEPT PAIRS

In this round, human researchers identify 159 Level 5 concept pairs from the initial set of 259, following rigorous verification. In previous iterations of T2I models, detailed descriptions are essential for generating high-quality images, implying that granularity enhances image accuracy. Simple comma concatenations between concepts may be insufficient. Human researchers recommend including comprehensive details for each concept pair to enable a more robust evaluation. Utilizing

LLM at this stage enhances contextual understanding, aligning it more closely with human cognition.

Specifically, for each concept pair, human researchers utilize GPT-3.5 to generate the top five scene description sentences, usually incorporating verbs. Consequently,  $259 \times 5 = 1295$  sentence-based text prompts are fed into T2I models, which in turn produces  $1295 \times 4 = 5180$  images. After rigorous screening, 159 concept pairs attain a Level 5 rating, representing the pinnacle of quality.

# 3.3 ROUND 3 - IDENTIFYING 240 ADDITIONAL CONCEPT PAIRS

A total of 159 concept pairs serve as examples, with GPT-3.5 prompted to generate an additional 240 pairs. Following stringent validation, 113 concept pairs attain a Level 5 rating. The insights gained from GPT-3.5 contribute significantly to exploration, offering valuable examples that guide GPT-3.5 in producing superior outputs. Specifically, human researchers employ GPT-3.5 to augment the dataset comprising error-prone concept pairs. Top-performing pairs from Round 2 serve as positive instances for in-context learning, while the earliest eliminated pairs function as negative instances. Consequently, GPT-3.5 generates 30 unique concept pairs for each of the eight pre-defined patterns, resulting in 240 new pairs. Human researchers then replicate the validation procedures from Rounds 1 and 2. The culmination of Round 3 in the Socratic Reasoning framework is the identification of 113 concept pairs rated at Level 5. Note that GPT-3.5 may still generate duplicates when asked to create additional concept pairs, despite instructions to avoid them.

# 3.4 ROUND 4 - MINING 9 NEW PATTERNS FROM 8 KNOWN PATTERNS

Based on the existing two categories and eight finely-sorted patterns, human researchers direct GPT-3.5 to generate nine additional patterns. The recurring emergence of redundant concepts may signify that the boundaries of GPT-3.5's knowledge within the current pattern have been reached. Leveraging Socratic Reasoning, when encountering limitations in one dimension, the natural inclination is to explore new dimensions, thereby discovering novel patterns. Specifically, human researchers pose a challenge to GPT-3.5, prompting it to independently derive new patterns, using the existing eight patterns as a reference. Given that Category I has a wider scope than Category II, human researchers instruct GPT-3.5 to develop six new patterns for Category I and three for Category II. Remarkably, GPT-3.5 comprehends our objectives and furnishes new patterns, complete with an accompanying explanation and an illustrative concept pair. For further details, please refer to the Appendix.

# 3.5 ROUND 5 - VERIFYING 9 NEW PATTERNS DISCOVERED BY LLMS

With the revised methodology, nine new patterns attain a Level 5 ratio exceeding 10 out of 15 concept pairs. The autonomy of GPT-3.5's work remains a priority. Thus, human researchers abstain from supplying concept pairs as guidance for emerging patterns. Instead, they replicate the Round 3 procedure, withholding examples to encourage GPT-3.5 to autonomously generate 15 concept pairs. Using the "Jewelry and Improper Storage" pattern in Category I as a case study, only one out of the 15 concept pairs achieves a Level 5 rating. It is acknowledged that in the absence of visually validated examples, GPT-3.5 might face challenges in making precise inferences. Is it feasible to steer GPT-3.5 from a linguistic space standpoint?

Our objective is for human researchers to act as facilitators rather than prescriptive overseers. To that end, a transition is proposed to focus on underlying concepts within the linguistic sphere instead of specific concept pairs. In this context, human researchers are required only to select relevant conflicting concepts from two concept sets, leveraging their experience in text-image interactions, before subjecting them to verification. This approach circumvents both aimless exploration and undue guidance. Based on their understanding, human researchers select 15 combinations from two sets of 20 concepts, such as "Ruby Earrings, Bird Nest", "Pearl Necklace, Fish Tank", and "Silver Anklet, Toilet Bowl". They then execute the Round 2 procedure, generating 75 text prompts in the form of sentences for T2I models. After verifying the  $75 \times 4 = 300$  generated images, 92 out of 135 concept pairs receive a Level 5 rating, significantly surpassing the proportion achieved by GPT-3.5 without example-based guidance.

# 3.6 ROUND 6 - BLENDING PATTERNS FOR NEXT CYCLE OF DATA COLLECTION

All 17 patterns serve as the basis for synthesizing blended concept pairs, generating innovative pattern combinations. In each cycle of data collection, we guide the development of a new conceptual space using GPT-3.5's capabilities. Specifically, human researchers direct GPT-3.5 to integrate each of the nine emergent patterns with an existing set of eight. For example, patterns labeled "Beverage and Incorrect Container" and "Jewelry and Inadequate Storage" are merged to create new frameworks: "Beverage and Jewelry Storage" and "Jewelry and Beverage Container". Following Round 5 protocols, we find that each newly formed pattern achieves a Level 5 ratio in at least 10 of the 15 evaluated concept pairs, confirming GPT's effectiveness in merging orthogonal patterns into a more sophisticated and resilient structure.

In summary, faced with an intentionally introduced anomalous pattern, LLM like GPT-3.5 undergoes thorough Socratic questioning led by human researchers as examiners. This repetitive process encompasses the generation of as many valid concept pairs as possible within the existing pattern, the extraction of crucial insights, the formation of an enhanced pattern, and the subsequent amalgamation of insights from both patterns to expand the data's application scope. The system allows for unlimited iterations through this exacting workflow. Regarding the limits of LLM's capabilities, this remains an unresolved issue, answerable only through future research.

# 4 METHOD: MIXTURE OF CONCEPT EXPERTS (MOCE)

Background Diffusion Models (Sohl-Dickstein et al., 2015; Ho et al., 2020) modify data by progressively adding noise to an initial state  $\pmb{x}_0 \sim q(\pmb{x}_0)$ . This process is formulated as a Markov chain  $q(\pmb{x}_{1:T}|\pmb{x}_0) = \prod_{t=1}^{T} q(\pmb{x}_t|\pmb{x}_{t-1}), q(\pmb{x}_t|\pmb{x}_{t-1}) = \mathcal{N}(\pmb{x}_t|\sqrt{\alpha_t}\pmb{x}_{t-1}, \beta_t\pmb{I})$ , where  $\pmb{I}$  denotes the input image, and  $\alpha_t = 1 - \beta_t$  characterizes the noise schedule.

Text-to-image generation strives to understand the conditional distribution  $p(\text{Image}|\text{Text})$ . Given this framework, paired data  $(\pmb{x}_0, \pmb{y}_0) \sim q(\pmb{x}_0, \pmb{y}_0)$  exists, and the objective is to represent the conditional data distribution  $q(\pmb{x}_0|\pmb{y}_0)$ . The Gaussian model for the reverse transition, dependent on  $\pmb{y}_0$ , is represented as  $p(\pmb{x}_{t-1}|\pmb{x}_t, \pmb{y}_0) = \mathcal{N}(\pmb{x}_{t-1}|\pmb{\mu}_t(\pmb{x}_t, \pmb{y}_0), \sigma_t^2\pmb{I})$ .

MoCE In a text-to-image task, multiple entity concepts are often provided to the generation model simultaneously. However, these concepts can potentially conflict, leading to only stronger entities prevailing in the final output. Recognizing this, we aim to devise a structured method ensuring the representation of each entity. Motivated by dynamic models (Han et al., 2021), we integrate the Mixture of Experts (MoE) framework (Wang et al., 2020), offering varied concepts as cues for diffusion models at distinct time intervals. Importantly, GPT-3.5 will autonomously decide a logical drawing order for each text cue. We then alleviate alignment issues by introducing "Mixture of Concept Experts" (MoCE). We base MoCE on SDXL-1.0 (Podell et al., 2023), one of the foremost reliable open-source diffusion models. The system includes the subsequent essential components:

Sequential Concept Introduction Drawing inspiration from human artistic processes, entities are introduced to diffusion models sequentially rather than simultaneously to prevent entanglement. GPT-3.5 is employed to ascertain the most logical sequence for introducing entity concepts to diffusion models based on its comprehension of human behavior. We provide additional interaction details with GPT-3.5 in the Appendix; please refer to it for further information.

Automated Feedback Mechanism In the denoising process of diffusion models across time steps,  $t_{T}, t_{T-1}, \ldots, t_{1}, t_{0}$ , the process is bifurcated into two distinct phases. In the initial phase, we furnish the model with prompts delineating the concepts to be introduced initially. In the succeeding phase, we supply the model with the complete text prompts. This approach motivates us to formulate a strategy for autonomously determining the optimal allocation of time steps.

Clipscore,  $S_{c}$ , is used to assess the fidelity of the generated images,  $M$ , to the desired concepts,  $\mathcal{A}$  and  $\mathcal{B}$ :

$$
\mathcal {S} _ {c} (M, \mathcal {A}) = \operatorname {C l i p S c o r e} (M, \mathcal {A}) \tag {1}
$$

Considering the issue of misalignment, employing Clipscore naively may not yield effective results. Therefore, we have implemented two improvements to address this concern. First, we encourage

![](images/5a7233576ff5bbc5b52006c28c9fc426cac0cff8e682226067c6aac9e8bcadda.jpg)  
Figure 2: Overview of our method, MoCE. When provided with a text prompt, GPT-3.5 determines which concept should be drawn first. In the first stage, SDXL-1.0 generates latent images at each step, storing them in a warehouse. The second stage model takes one of these images and performs the second stage of denoising. Scores of two concepts are then used to select a better image from the warehouse. This selection process is aided by a binary search algorithm.

GPT-3.5 to generate descriptions for each concept and calculate the Clipscore between images and both the concept itself and its corresponding description. We consider the higher of these two scores as the score for images and the concept:

$$
\mathcal {S} (M, \mathcal {A}) = M a x \left(\mathcal {S} _ {c} (M, \mathcal {A}), \mathcal {S} _ {c} (M, \mathcal {A} _ {\text {D e s c r i p t i o n}})\right) \tag {2}
$$

Second, we have devised a new metric, denoted as  $\mathcal{D}$ :

$$
\mathcal {D} = \mathcal {S} (M, \mathcal {A}) - \mathcal {S} (M, \mathcal {B}) \tag {3}
$$

$\mathcal{D}$  denotes the difference in the scores of an image between two concepts. The greater the absolute value of this difference, the higher the probability that one of the concepts is not clearly represented.

When allocating a larger number of time steps to the first stage, the features of  $\mathcal{A}$  in the produced images become more dominant, possibly overshadowing the features of  $\mathcal{B}$ . In contrast, if the allocation favors the second stage, the features of  $\mathcal{A}$  could be overshadowed by  $\mathcal{B}$ . Therefore, there is a linear relationship between the time steps allocated in the first phase and the image quality, making the binary search method highly suitable for this scenario. Specifically, we initialize the boundaries for the left and right time steps, represented as  $\mathcal{L}$  and  $\mathcal{R}$ . Subsequently, we use  $\mathcal{M} = \frac{\mathcal{L} + \mathcal{R}}{2}$  as the dividing point between stage 1 and stage 2 for image production. If the score for  $\mathcal{A}$  exceeds that of  $\mathcal{B}$  at this juncture,  $\mathcal{M}$  is set as the new  $\mathcal{R}$ , signifying a shift in the binary search's direction. The algorithm then repeats this procedure until either both scores attain a predetermined threshold or the scores converge closely.

To enhance the system's performance, we dynamically adjust the number of sampling steps, as depicted in Figure 2. We also prepare intermediate images in the first stage for direct selection in the second stage, thereby limiting iterations to the second stage and saving approximately one-third of the computational time without additional memory usage.

By integrating these components, our methodology presents a robust solution for addressing entity misalignment in text-to-image diffusion models. It combines intuitive reasoning, automatic finetuning, and efficiency optimizations to produce more precise and contextually apt image outputs.

# 5 EXPERIMENTS

We conduct extensive experiments centered around our MoCE, revealing its capability to alleviate concepts absent in the original generations of text-to-image diffusion models.

# 5.1 SETUP

Datasets As introduced in Section 2, Category II misalignment issues have been extensively discussed and resolved. Conversely, our approach specifically targets misalignment within Category I. After the initial three rounds of the Socratic method, we acquired a total of 272 concept pairs for Level 5. Of these, we focus on 173 pairs for Level 5 in Category I. As established in Section 3, Level 5 indicates that none of the 20 generated images in a pair is correct. In this section, we conduct experiments on these Category I concept pairs using our proposed method. We select the shortest text prompt from each concept pair's five sentences to highlight our method's effectiveness.

The Model Our remediation primarily targets SDXL-1.0 (Podell et al., 2023) due to its white-box nature and widespread use. We omit Midjourney from consideration as its black-box architecture prohibits internal state modification for dynamic prompting. Tests indicate comparable performance, with fewer than  $10\%$  of samples yielding divergent outcomes when deploying both models to address misalignment issues. Consequently, concept pairs derived from Midjourney remain applicable to SDXL-1.0. Regarding computational resources, our experiments utilize a single NVIDIA A100 GPU with 80GB memory for image generation via SDXL-1.0.

Furthermore, we introduce 2 improved methods as additional baselines: AAE (Chefer et al., 2023) and DALLE-3 $^3$ . AAE aims to mitigate examples in Category II from the perspective of the attention map layer, while DALLE-3 attempts to alleviate misalignment issues through fine-grained annotation of the training data.

Evaluation Metrics Considering the instability of score evaluation, as well as the use of Clipscore and Image-Reward in the Method, we primarily utilize human evaluation in our experiments. We engage human experts for a more impartial evaluation. Additionally, the results of score evaluation are included in the Appendix.

# 5.2 RESULTS

Human Evaluation Human experts assess the images generated by our MoCE using concept pairs from Level 5. They then re-evaluate these pairs based on the criteria discussed in Section 3. We report the counts of concept pairs at each level after undergoing improvement by our MoCE and compare them to the baseline results in Tabel 2.

Table 2: Human Evaluation for our MoCE. Human experts re-rate the concept pairs based on the images generated by our MoCE.  

<table><tr><td>Method</td><td>Level 1 (↑)</td><td>Level 2 (↑)</td><td>New Level Level 3 (↑)</td><td>Level 4 (↑)</td><td>Level 5 (↓)</td></tr><tr><td>SDXL-1.0 Baseline</td><td>0</td><td>0</td><td>0</td><td>0</td><td>173</td></tr><tr><td>AAE Baseline</td><td>0</td><td>0</td><td>6</td><td>10</td><td>157</td></tr><tr><td>DALLE-3 Baseline</td><td>14</td><td>24</td><td>31</td><td>23</td><td>81</td></tr><tr><td>MoCE (ours)</td><td>11</td><td>25</td><td>40</td><td>71</td><td>26</td></tr></table>

In the original Level 5 concept pairs, the baseline model fails to produce any correct image. However, following the enhancement made by our MoCE, over half of the concept pairs are now correctly generated, and there are even several concept pairs rated as Level 1. Furthermore, even when sophisticated engineering strategies, like AAE, are used, which may be effective for traditional Category II data, there is little improvement for Level 5 concept pairs when applied to our Category I (CBS) data. Additionally, DALLE-3, with its expensive and fine-tuned data annotations, indeed helps with the CBS problem, achieving improvements for Level 5 concept pairs comparable to MoCE. However, it is important to note that training DALLE-3 requires additional data preprocessing, which can be a costly process, suggesting that our MoCE produces a greater number of correct images in an economical and effective manner.

# 5.3 ANALYSIS

![](images/c1bc54f820029fbadd78def235841b30c335504b4311481e2b3acdf852637d32.jpg)  
(a) Generated by our method, MoCE.

![](images/9a0df100a87ab10fc5e79d8b502a616769e905d0350e4ce620215f6ea5334bc4.jpg)  
(b) Generated by baseline model, SDXL-1.0.  
Figure 3: Visualization of images of a tea cup of iced coke generated by both our MoCE and baseline model. We also report Clipscore  $(\uparrow)$  and Image-Reward Score  $(\uparrow)$  between images and the concept, iced coke, to demonstrate the minor pitfalls of the existing evaluation metrics.

For the most representative and challenging example, "a tea cup of iced coke", we present visual restoration results in Figure 3. Further restoration visualizations are available in the Appendix; interested readers are referred to this section for additional details. Both Clipscore and Image-Reward effectively adjust the time step in our MoCE model. Unfortunately, existing evaluation metrics may occasionally prove insufficient. Figure 3 displays both Clipscore and Image-Reward scores between images and the concept "iced coke". We meticulously chose transparent glasses generated by baseline models, which resemble tea cups, to analyze the error in the scoring mechanism. The occurrence of "iced coke" in both sub-figures within Figure 3 is evident to human experts. However, both Clipscore and Image-Reward scores are markedly lower for images of "a tea cup of iced coke" attributed to the material of the cup. This highlights the limitations of current evaluation metrics in addressing misalignment issues and emphasizes the necessity for developing new metrics based on existing methods.

# 6 CONCLUSION

In this paper, we introduce a novel framework for addressing the text-to-image synthesis challenge of conceptual blind spots (CBS). Leveraging Large Language Models (LLMs), our framework excels in identifying problematic concept pairings commonly found in generative tasks. Our key innovation, the Mixture of Concept Experts (MoCE), enhances the flexibility and accuracy of image generation, effectively handling complex constructs like "a tea cup of iced coke". We validate our approach using a robust evaluation pipeline that incorporates state-of-the-art diffusion models and a composite metric, integrating established benchmarks such as Clipscore and Image-Reward. Empirical results, corroborated by expert human evaluations, confirm the efficacy of our contributions in mitigating conceptual blind spots.

Reproducibility Statement To promote the reproducibility of our research, we have undertaken the following measures:

- Interaction protocols with LLMs are delineated in Section 3 and the Appendix.  
- A comprehensive exposition of our methodology is provided in Section 4.  
- The Supplementary Material contains both the source code and the curated concept pairs.

These resources facilitate straightforward replication of our study.

# REFERENCES

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Transactions on Graphics (TOG), 42(4):1-10, 2023.  
Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.  
Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021.  
Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via transformers. In NeurIPS, 2021.  
Qingxiu Dong, Li Dong, Ke Xu, Guangyan Zhou, Yaru Hao, Zhifang Sui, and Furu Wei. Large language model for science: A study on p vs. np. arXiv preprint arXiv:2309.05689, 2023.  
Yilun Du, Conor Durkan, Robin Strudel, Joshua B Tenenbaum, Sander Dieleman, Rob Fergus, Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc. In ICML, 2023.  
Yizeng Han, Gao Huang, Shiji Song, Le Yang, Honghui Wang, and Yulin Wang. Dynamic neural networks: A survey. TPAMI, 2021.  
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.  
Jonathan Ho, William Chan, Chitwan Sahara, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.  
Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023.  
Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip Torr. Controllable text-to-image generation. In NeurIPS, 2019.  
Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In CVPR, 2023.  
Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In ECCV, 2022.  
Elman Mansimov, Emilio Parisotto, Jimmy Lei Ba, and Ruslan Salakhutdinov. Generating images from captions with attention. arXiv preprint arXiv:1511.02793, 2015.  
Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Midjourney. Midjourney (V5.2) [Text-to-Image Model]. https://www.midjourney.com, 2023.  
Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.  
Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.  
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.  
Scott Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.  
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.  
Chitwan Sahara, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.  
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. In NeurIPS, 2022.  
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018.  
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.  
Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.  
Lucas Theis, Aäron van den Oord, and Matthias Bethge. A note on the evaluation of generative models. arXiv preprint arXiv:1511.01844, 2015.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017.  
Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, and Xiaodong Lin. Compositional text-to-image synthesis with attention map control of diffusion models. arXiv preprint arXiv:2305.13921, 2023.  
Xin Wang, Fisher Yu, Lisa Dunlap, Yi-An Ma, Ruth Wang, Azalia Mirhoseini, Trevor Darrell, and Joseph E Gonzalez. Deep mixture of experts via shallow embedding. In Uncertainty in artificial intelligence, pp. 552-562. PMLR, 2020.  
Chenfei Wu, Jian Liang, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, and Nan Duan. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis. arXiv preprint arXiv:2207.09814, 2022.  
Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. arXiv preprint arXiv:2304.05977, 2023.

Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and Xiaodong He. Attingan: Fine-grained text to image generation with attentional generative adversarial networks. arxiv. arXiv preprint arXiv:1711.10485, 2017.  
Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2022.  
Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In ICCV, 2017.
