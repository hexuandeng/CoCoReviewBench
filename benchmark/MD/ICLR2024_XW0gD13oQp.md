# LANGUAGE AGENTS FOR DETECTING IMPLICIT STEREOTYPES IN TEXT-TO-IMAGE MODELS AT SCALE

# ABSTRACT

The recent surge in the research of diffusion models has accelerated the adoption of text-to-image models in various Artificial Intelligence Generated Content (AIGC) commercial products. While these exceptional AIGC products are gaining increasing recognition and sparking enthusiasm among consumers, the questions regarding whether, when, and how these models might unintentionally reinforce existing societal stereotypes remain largely unaddressed. Motivated by recent advancements in language agents, here we introduce a novel agent architecture tailored for stereotype detection in text-to-image models. This versatile agent architecture is capable of accommodating free-form detection tasks and can autonomously invoke various tools to facilitate the entire process, from generating corresponding instructions and images, to detecting stereotypes. We build the stereotype-relevant benchmark based on multiple open-text datasets, and apply this architecture to commercial products and popular open source text-to-image models. We find that these models often display serious stereotypes when it comes to certain prompts about personal characteristics, social cultural context and crime-related aspects. In summary, these empirical findings underscore the pervasive existence of stereotypes across social dimensions, including gender, race, and religion, which not only validate the effectiveness of our proposed approach, but also emphasize the critical necessity of addressing potential ethical risks in the burgeoning realm of AIGC. As AIGC continues its rapid expansion trajectory, with new models and plugins emerging daily in staggering numbers, the challenge lies in the timely detection and mitigation of potential biases within these models.

Warning: Some content contains racism, sexuality, or other harmful language.

# 1 INTRODUCTION

Recent advancements in text-to-image generative models, such as DALL-E 2 (Ramesh et al., 2022) and Stable Diffusion (Rombach et al., 2022), have garnered significant attention in both academia and industry. These state-of-the-art (SOTA) image generation models have facilitated the development of novel creative tools and applications, such as Midjourney<sup>1</sup> and NovelAI<sup>2</sup>. However, alongside their remarkable success, emerging studies (Bianchi et al., 2023a) have begun to reveal inherent and implicit social stereotypes within these models, posing ethical risks when deployed on a large scale. For instance, Naik & Nushi (2023) conducted a preliminary analysis of existing models, illustrating how they perpetuate stereotypes related to race, gender, and other social identities using a limited set of handcrafted prompts. As Fig.1 shows, when given a prompt such as "a terrorist," most of the generated images may disproportionately depict a Middle Eastern, thereby perpetuating the stereotype that Middle Easterners are more likely to be terrorists.

Although these research works serve as a crucial initial step, the scope of their evaluations remains confined, primarily focusing on a closed environment setting, where static test sets within predefined crafted rules are constructed and are then employed to elicit potential biases inherent to the given models (e.g., in Naik & Nushi (2023)). For instance, Bianchi et al. (2023b) approached the

issue by iterating through various personal traits and professions in conjunction with, designing corresponding prompts to test.

Notably, the sources and types of stereotypes can be diverse (Caliskan et al., 2017; Bourdieu, 2018; Kozlowski et al., 2019; Jha et al., 2023) in the real-world open environments (Breitfeller et al., 2019; Schmidt & Wiegand, 2017). At the same time, the open-source community has witnessed a surge in the usage of text-to-image models and their corresponding style plugins. For instance, on the civilai platform<sup>3</sup>, there are approximately 3k distinct style plugins (Hu et al., 2022) publicly available. The widespread distribution and usage of these models and plugins present significant challenges for ethical oversight and risk governance targeting these models (Cho et al., 2022). The diversity of potential stereotypes, combined with the rapid expansion of available plugins, suggests that neither iterating through every conceivable configuration nor the ad-hoc selection of a narrow subset of traits is ideal, as they are either computationally expensive or possibly tainted by human cognitive biases. This essential tension highlights an urgent need for a more efficient yet comprehensive framework of stereotype risk evaluation, which could benefit the community in various important ways. Given the pervasive adoption of AIGC technology and its integration in commercial products, e.g. DALL-E 2 and Midjourney, such a framework will not only enhance our intellectual understanding of bias and stereotypes in foundational models, but also offer novel accessible tools that are accessible for industry professionals and regulatory entities focused on related governance.

![](images/99903568fb558a735b1072629158d79c78c659dce4359e603b9d4c3e16ffaa69.jpg)  
Figure 1: Simple user prompts can inadvertently lead to the generation of images that reinforce and amplify stereotypes across various social dimensions, as observed in different models. Among these models, Midjourney stands out as the most popular commercial product. Additionally, there are three popular open-source models: SDVN-RealArt, Dreamshaper, and ChilloutMix. These models specialize in a realistic style, illustration style, and animation style respectively.

In this study, we focus on three fundamental social dimensions: gender, race, and religion. and incorporate a wide range of subgroups under each of them. Building on recent progress in large language models (LLMs) augmented agents (Shinn et al., 2023; Xu et al., 2023; Yao et al., 2022), we present a preliminary exploration into the construction of a language agent dedicated to stereotype risk governance. Our primary objective is to harness the robust reasoning capabilities of LLMs to invoke various multimodal models for stereotype evaluation. In particular, our proposed agent architecture adheres to the React paradigm (Yao et al., 2022), integrating LLMs with an array of visual tools, namely multimodal models (Radford et al., 2021; Li et al., 2022). This integration facilitates the extraction of stereotype prompts, which are subsequently employed to detect potential biases inherent in the target model. As illustrated in Figure 2, the user might pose a query such as 'Is Chilloutmix model racially stereotyped?' In response, the agent invokes specific tools to formulate prompts related to racial stereotypes and the relevant target subgroup. Following this, the target model is invoked to generate images. Ultimately, the agent computes a stereotype score based on the generated images and furnishes a comprehensive analytical report.

From a broader perspective, the operational procedure of our agent can be segmented into three primary phases: (1) Instruction Pair Generation: Generate instruction pairs using toxic natural language processing (NLP) datasets, which primarily stem from the open text of various social plat-

forms. These datasets are believed to be highly reflective of social stereotypes. The instruction pair is defined as a combination of "prompt" and "subgroup", such as ("terrorist", "Middle Eastern"), indicating that when the drawing prompt is "terrorist", the text-to-image model might tend to generate images with "Middle Eastern" stereotype. (2) Image Generation: Call the corresponding generated models to get images that meet the basic settings. (3) Stereotype Detection: Use the vision-language model to identify the subgroups of each generated image and calculate the final stereotype score.

To rigorously assess the efficacy of our agent, especially when juxtaposed against models employed in proprietary products, the pivotal criterion is the detection performance in the third phase. To facilitate this, we instruct our agent to curate a benchmark comprising 4123 instruction pairs from 5 public toxicity textual datasets, followed by the generation of a multitude of images corresponding to each instruction. A random subset, constituting  $10\%$  of these images, is manually annotated to ascertain the presence of a designated group. This methodology enables us to gauge the agent's proficiency in discerning biases within text-to-image models. Our preliminary investigations, encompassing Midjourney and widely-recognized open-source models, reveal that our agent's performance closely mirrors human evaluation accuracy, with an average precision of  $92.15\%$ . This result underscores the potential of integrating our proposed agent into real-world bias governance frameworks. Furthermore, our experimental results have revealed crucial insights into the behavior of text-to-image models. Specifically, when provided with drawing instructions related to certain personal characteristics, social culture, and criminal violence, these models tend to generate images that exhibit bias towards certain groups, underscoring potential ethical concerns and lasting harm.

The overall contribution is as follows:

- To our knowledge, we are the first to build a language agent for stereotype detection and governance against text-to-image models. Importantly, the use of language agent enables a general automatic framework that generates and examines potential stereotypes at scale, without the need to iterate through all possible combinations or resort to ad-hoc selection of a limited set of traits.  
- We propose the evaluation benchmark to assess the capabilities of the proposed agent framework. This framework is constructed using the toxic text dataset from social platforms, thereby aligning more closely with the prevalence of stereotypes in daily expressions. By utilizing this dataset, our evaluation process ensures a more realistic representation of bias occurrences in real-world scenarios.  
- Our experimental results show our agent is not only close to human annotation in detection accuracy, but also adapt well to the requirements of various detection tasks.

# 2 AGENT DESIGN

Our agent framework is an automated evaluation system for detecting stereotypes in text-to-image models, which is composed of task planning and tool usage.

# 2.1 TASK PLANNING

In order to enable our agent to comprehensively understand any stereotype detection request from users, we employ LLMs as the reasoning engine of the agent. This entails utilizing LLMs for intent comprehension and task decomposition when processing user requests. Specifically, we leverage few-shot prompting techniques to empower LLMs to generate task planning trajectories following the ReAct paradigm (Yao et al., 2022). In the setting of this study, few-shot prompting refers to the practice of injecting user requests along with their corresponding task planning trajectories as examples within the context provided to the LLMs. For this study, we have selected five representative user requests, and their associated task-planning trajectories have been authored by human experts. Each trajectory is composed of multiple thought-action-observation steps. For each sub-step, "Thought" is a natural language description of the planning process. This description is derived from an analysis of the current step's task objective and the corresponding external tool information needed, based on the context information aggregated from previous steps. "Action" represents the concrete tool execution distilled from the description in the current "Thought" field, while "Observation" indicates the information returned after the tool execution. After presenting these case

examples, the LLMs can generate planning trajectories that adhere to the specified format requirements for a given user request. A concrete example of planning trajectory is shown in Figure 2. The details of tool usage are introduced in the next section.

![](images/9fd8844a8716629774d56de3598e0c7d6b1ccdd26aaee80061a8d6e74ac4af6d.jpg)  
Figure 2: Overview of our detection agent for text-to-image models. We illustrate the task-solving trajectory of the specific detection task by taking the user query in the dotted box on the left as an example.

# 2.2 TOOL USAGE

In order to enable the agent to complete each subgoal, we establish various tools and interfaces of generative models. Furthermore, we delineate the specific usage scenarios for each tool, along with the arguments that need to be supplied. In the following, we introduce in detail the usage of various tools.

(1) Intention understanding: This tool is designed to understand user requests and extract key detection elements. These elements include the target model, social dimensions, and the open text to be detected. However, it is important to note that not all task descriptions will necessarily contain all of these elements, as users may provide descriptions in various formats.  
(2) Instruction Generation: The tool transforms the open text into an instruction pair, formatted as (prompt  $p$ , subgroup  $g$ ). For instance, ("terrorists", "Middle Eastern") comprises the descriptive object "terrorists" and the demographic subgroup "Middle Eastern". We establish the task-specific prompt as "Extract potential descriptions and target groups within the given content and return them in the format ('description', 'target group')". Here, the 'description' typically includes adjectives, nouns, or action verbs associated with the stereotype, while the 'target group' refers to the demographic subgroup, e.g. 'Asian' and 'Middle Eastern'.  
(3) Instruction Retrieval: This tool is aimed at detection tasks that do not involve specific open text. It takes social dimensions  $G$  as input and retrieves the corresponding bias instruction pairs  $I$  whose subgroup  $g \in G$  from our instruction pair dataset. The specific creation process of the dataset is in 3.2. Based on the evaluation results from our benchmarks, the agent selects the instruction pair  $I$  that exhibits the most stereotypes among the text-to-image models.  
(4) Prompt Optimization: The prompt  $P$  in the instruction  $I$  is often composed of expressions that directly describe stereotypes, and specific optimization is often required to actually be used for text-to-image models. Therefore, we set up a template as "The people who" to support adding different description expressions. In addition, the above initial prompt may cause the problem of missing subject objects in the generated images, and further emphasis on subject keywords is required, such as "(person, 1.5)".  
(5) Stereotype Score Calculator: The stereotype degree of the model in response to a specific prompt can be quantified by the conditional probability  $P(G|P)$ , which represents the probability that the generated images are associated with the specific social group  $g$  given the painting prompt  $p$ . Specifically, we define the stereotype score as the proportion of the number of majority subgroup images to the total number.

# 3 AGENT BENCHMARK

In this section, we present a comprehensive evaluation of prominent text-to-image models using our proposed agent. We begin by introducing the Text-to-Image models used in the evaluation. Then, we present the construction process for the benchmark dataset used to evaluate these models. Next, the experimental settings are introduced. Finally, we report and discuss the benchmark evaluation results.

# 3.1 TEXT-TO-IMAGE MODELS

To comprehensively assess the variant Stable Diffusion models, we select three main custom styles for evaluation: "realism", "banana", and "artistic". We choose the most downloaded models in the Civitai community corresponding to each style. Moreover, we also examine the performance of the SDXL model and the integration of different LoRA modules with these base models. These modules are also selected from the most popular model plugins within the same community.

# 3.2 BENCHMARK DATASET

To investigate potential stereotypes in the current Text-to-Image models, we instruct our agent to construct a benchmark dataset, the Stereotypical Prompts for Image Generation (SPIG) Dataset, by collecting the stereotypical prompts and their corresponding groups from publicly available toxicity text datasets. The stereotypical prompts are mainly extracted from five existing toxicity text datasets that cover different demographic groups and stereotypical topics from various social media platforms. Below, we provide a brief introduction for each dataset as shown in Table 1:

Table 1: Statistics of toxic text datasets.  

<table><tr><td>Dataset</td><td># of prompts</td><td>Type of annotations</td></tr><tr><td>SBIC (Social Bias Inference Corpus) (Sap et al., 2019)</td><td>150K</td><td>Structured annotations of social media posts</td></tr><tr><td>HateExplain (Mathew et al., 2021)</td><td>20K</td><td>hate/offensive/normal classification for posts</td></tr><tr><td>DYNAHATE (Vidgen et al., 2020)</td><td>40K</td><td>Fine-grained labels for the hate type and target group</td></tr><tr><td>IHC (Implicit Hate Corpus) (ElSherief et al., 2021)</td><td>9.5K</td><td>Target demographic group and implied statement</td></tr><tr><td>SMTD (Social Media Toxicity Dataset)4</td><td>1K</td><td>Toxic comments from social media platforms</td></tr></table>

First, we employ the Instruction Generation tool to utilize samples containing toxicity text from the dataset and filter out instruction pairs with strong stereotypes. Subsequently, we utilize the Prompt Optimization tool to optimize the generated drawing prompts, making them easier for text-to-image models to comprehend and render. We extract a total of 4123 valid painting prompts from all toxic datasets. We categorize the corresponding groups into three categories: gender, race, and religion. Each category encompasses various subgroups; for example, the "race" category includes subgroups such as 'African' and 'Asian'. The overall distribution is shown in Fig. 3(a), "gender" accounts for  $55\%$  of the total prompts, "race" makes up  $33.6\%$ , and "religion" represents  $11.5\%$ . The distribution of prompts within each category is shown in Figs. 3(b)-3(d).

# 3.3 BENCHMARK SETTING

After extracting the corresponding instruction pairs from the aforementioned toxic datasets, our agent systematically invokes each model and plugin in the benchmark to generate images associated with the respective prompt. To ascertain the stereotype degree exhibited by each model in response to a specific prompt, we employ manual annotation to obtain the ground-truth label for each image.

Due to the cost and efficiency issues of manual annotation, we carefully select the instruction pairs to be annotated. We select a representative set of instructions from our constructed benchmark

![](images/a71fce63aff67475c662a924849df6c584d13a9ee4423e48f9063dad41f96982.jpg)  
(a) Overall distribution.

![](images/509c3954138954bf33100ea7d5994d8fdcaa235665a91cb39531caca107735c0.jpg)  
(b) Distribution of gender and sex.

![](images/d9fc4b565246f9ee2813cf899d50be8044c0f02ccb3db6ee90a92cbbe1e886b6.jpg)  
Figure 3: Overall distribution and proportion of stereotype pair for each subgroup.  
(c) Distribution and ethnicity.  
race (d) Distribution of religion.

![](images/f99dccd3a97ccb213eb2132ff863a6b36c3091ded55139e25811a19afa677661.jpg)

dataset. The selection process aims to ensure that the chosen instructions cover a diverse range of demographic groups and stereotypical prompts. We perform stratified sampling to ensure that each social dimension and its corresponding subgroups are proportionally represented in the selected prompts. Finally, our agent invokes the Stereotype Score Calculator tool to compute the stereotype score of generated images, thereby determining the presence of stereotypes.

![](images/6909122d94f15f6627f7908414076c119a1fbff1c061cf8981104237ffc05e0c.jpg)  
Figure 4: Stereotype degree of Figure 5: Proportion of racial Figure 6: Proportion of different models in different so- subgroups under different religion subgroups under cial dimensions and models prompt types. different prompt types.

![](images/93b5e60d7419fc3e42b9b4579269208163717aa4414f3ecbcfb26fb22501645d.jpg)

![](images/d469a8dd89fd3185c19d3e51c139e779cd7865a4c457bdd751485fe71a23eb12.jpg)

# 3.4 BENCHMARK RESULT

After our agent invokes corresponding tools, we obtain the ground-truth label for each image, as well as the assessment of whether each model exhibits stereotypes in response to each prompt. In this section, we analyze the distribution of stereotype prompts across various levels, including the models, social dimensions, and demographic subgroups, thereby confirming the extensive presence of stereotypes in the text-to-image domain.

- Stereotypes of text-to-image models: As shown in Fig. 4, we analyze stereotypes of instruction pairs in the SFIG dataset across different models and plugins, where different colors represent the number of instructions that express stereotypes under different models. Among these models, the "chilloutmix" and "Realistic Vision" markedly display the most significant stereotyping, notably within the 'Gender' dimension. In this dimension, the model exhibits a larger stereotypical deviation in comparison to other models. This discrepancy may be attributed to an overrepresentation of female images within the training data, potentially catering to user preferences.  
- Stereotypes across social dimensions: Fig. 4 also compares the differences in stereotypes between different social dimensions. Among the three social dimensions, religion emerges as the dimension most heavily stereotyped. This situation brings to light two key issues. Firstly, it underscores the scarcity of diverse religious content in the training data. Secondly, it mirrors the significant stereotype problem inherent in religious content within societal and cultural contexts.  
- Stereotypes across demographic subgroups: In relation to race and religion, two social dimensions where stereotypes are particularly prevalent, we further analyze the impact of prompt type on specific subgroup stereotypes. As depicted in Fig. 5-6, we examine the distribution of subgroups across all generated images. As analyzed in Bianchi et al. (2023b), it can be observed that certain specific types of prompts are often correlated with particular subgroups. For example, for prompts involving violence, the images generated are often related to race subgroups such as African and Latino, or religious subgroups such as Mulism.

# 4 AGENT PERFORMANCE

To verify the effectiveness of our agent in reasoning and invoking tools in a real-world detection environment, we set up a series of model stereotype detection tasks to evaluate the agent's performance. We first introduce the task design used to evaluate agent performance, and then analyze the agent's overall performance on the stereotype detection task. Subsequently, we analyze the accuracy of key tool results in the agent's inference trajectory. Ultimately, we apply the agent to evaluate the commercial product, Midjourney, as a case study.

# 4.1 DETECTION TASK DESIGN

We first assess the degree of stereotyping in six models and plugins within the benchmark. Given that the stereotyping degree of these models across various social dimensions and prompts is determined through manual annotation, we can compare the agent's evaluation results with the manual assessment outcomes to accurately evaluate the agent's effectiveness.

In order to simulate user questions as much as possible, we employ LLMs to blend multiple models or plugins, and three social dimensions into sentence structures that mirror potential user query formats. For instance, "Can you tell me whether the SD-XL model exhibits racial stereotypes?" In addition, we also randomly add open text to the queries, which are unseen in the few-shot cases, to test the agent's extraction capabilities. By ensuring a diverse range of question formats, we enhance the reliability of the agent's evaluation results. We generate 120 task queries to evaluate the model's detection capabilities.

![](images/f657ea73d4526ae5b30f393e73550d307e10d0dde46a176ca0c429b4d8c0f9e4.jpg)  
Figure 7: Comparison of manual annotation and agent annotation results for the majority subgroup of generated images, where each point represents an evaluated prompt.

![](images/46507fb15df035e92de9f4de4922dd8a707528b8642413c01d272a0110edab59.jpg)

![](images/6b8da7b657191840c36227964ca8be58b8b7a938d6d95c35f31594a6bd362688.jpg)

# 4.2 TEXT-TO-IMAGE MODEL DETECTION PERFORMANCE

As depicted in Figure 7, we conduct an analysis of all prompts across the three dimensions involved in the task design. Each point in the figure represents one of the prompts. The x-axis signifies the proportion of bias as determined by manual annotation in the generated image, while the y-axis represents the proportion of bias detected by the agent. A distribution of scatter points near the diagonal line suggests that the accuracy of the agent closely aligns with that of manual annotation. As inferred from the figure, the performance of our agent does not significantly deviate from the results of manual annotation across the three social dimensions. Across the three social dimensions, the average discrepancies between the proportion of stereotype annotations made by the agent and the proportion of manual labeling are  $8.54\%$ ,  $7.95\%$ , and  $10.13\%$ , respectively. Despite the presence of bias in the annotation ratio, the overall accuracy in detecting whether the model contains stereotypes stands at  $92.15\%$ . This outcome highlights the potential of incorporating our proposed agent into practical bias governance frameworks.

Additionally, we focus on the accuracy of two key tools, Intention Understanding and BLIP-based Detection, within the agent's reasoning trajectory. For Intention Understanding tool, we generate 120 instructions for evaluation, and 114 out of 120 instructions are accurately recognized. The remaining instances primarily involve misidentification of model names or pertain to open texts. For

instance, the model name might be omitted, or the irrelevant content in the description is placed within the open text section.

For BLIP-based Detection tool, the classification accuracy of each subgroup within this tool plays a crucial role in determining the ultimate efficacy of model stereotype detection. Consequently, our attention is primarily directed towards comparing the accuracy of two multimodal models, namely CLIP(Radford et al., 2021) and BLIP(Li et al., 2022). The performance comparison between two multi-modal models is presented in Fig.8. The results indicate that BLIP surpasses CLIP in all subgroup classification tasks, with an average improvement of  $5\%$ . Furthermore, BLIP exhibits a classification accuracy of approximately  $80\%$  in the majority of subgroup classification tasks. However, for confused samples, such as those where "Asian" and "Latino" classifications intersect, some "Asian" samples are misclassified as "Latino", leading to a reduced average accuracy for the "Latino" category. Additionally, due to the limited training data available, the two models both exhibit lower classification ac

![](images/513054e966e48683ad44de5c6ab848a9db22c55a07c0f198fe84e5dfe72ab180.jpg)  
Figure 8: Classification accuracy of CLIP and BLIP on each subgroup, with darker shades denoting CLIP and lighter shades signifying BLIP. Different colors refer to different social dimensions.

curacy for less popular religious subgroups such as "Hindu".

# 4.3 CASE STUDY OF MIDJOURNEY

Based on the three social dimensions in the benchmark, we meticulously select the 60 most biased instruction pairs from our SPIG dataset. Our agent then uses these pairs to evaluate the stereotype degree of Midjourney. The evaluation results indicate that 46 out of the 60 pairs exhibit stereotypes, with 20 of them displaying the  $100\%$  stereotype ratio. This finding emphasizes the widespread presence of stereotypes, suggesting that despite Midjourney's advanced capacity to generate images of superior quality and detail, it also fails to successfully avoid stereotypes. For instance, when the prompt is "the person who is a cotton picker", all the generated images consistently depict 'African' descent. This result, similar to those observed in other open-source models, indicates that Midjourney also perpetuates serious racial stereotypes. These findings highlight the need for further research and development to mitigate the impact of stereotypes in AI-generated images.

# 5 RELATED WORK

# 5.1 STEREOTYPE IN VISION-AND-LANGUAGE

Recent advancements in text-to-image models (Bianchi et al., 2023a) have been noteworthy, with these models finding utility across various domains. However, their capacity to generate socially biased content, leading to the reinforcement of harmful stereotypes, has raised concerns. Several studies have attempted to quantify the bias inherent in these models (Caliskan et al., 2017).

For instance, (Jha et al., 2023) conducted an evaluation of gender and racial biases in text-to-image models. Their methodology was based on the skew of gender and skin tone distributions in images generated from neutral occupation prompts. They employed both automated and human inspection to identify gender and skin tone in the generated images. Their findings indicated that Stable Diffusion tends to produce images of a specific gender or skin tone from neutral prompts more frequently than DALL-E.

Similarly, (Ramesh et al., 2022) demonstrated that Stable Diffusion perpetuates harmful stereotypes related to race, ethnicity, gender, class, and intersectionality, even when using simple, neutral prompts. They also observed an amplification of these stereotypes. Their study further revealed

that prompts mentioning social groups generate images with complex stereotypes that are challenging to mitigate. For example, Stable Diffusion associated specific groups with negative or taboo concepts such as malnourishment, poverty, and subordination. Notably, neither the introduction of "guardrails" against stereotyping in models like Dall-E (ElSherief et al., 2021), nor the careful expansion of user prompts, succeeded in reducing the impact of these associations.

Previous research has extensively explored common stereotype characteristics, such as occupation and traits, and their effects on various subgroups distinguished by gender, race, and age. However, this approach to bias evaluation has its limitations, as it often neglects the more subtle stereotypes prevalent in everyday expressions. These biases frequently manifest in toxic content disseminated across various social platforms, including racial slurs.

In this study, we introduce a novel form of bias, termed 'life-expressive bias', which is arguably more harmful and has been largely overlooked in previous research. Our methodology deviates from traditional approaches that rely on artificially constructed data. Instead, we construct a dataset of stereotype pairs, comprising 'stereotype content' and 'stereotype objects', derived from various toxic content datasets. This innovative approach has enabled us to establish a more comprehensive benchmark for quantifying the extent of model bias.

# 5.2 AUTONOMOUS LANGUAGE AGENT

The recent surge in the popularity of language agents has led to the development of a diverse range of these agents, each designed to perform specific tasks. For instance, (Park et al., 2023) developed language agents to emulate human social behavior, while (Nakano et al., 2021) showcased the potential of constructing language agents capable of executing tasks on real websites following natural language instructions. Furthermore, (Qian et al., 2023) and (Hong et al., 2023) conducted experiments on software development in multi-agent communication settings, and (Zhou et al., 2023) constructed language agents to function as interactive writing assistants.

Beyond task-specific language agents, recent open-source projects such as (Yang et al., 2023), (Talebirad & Nadiri, 2023), and SuperAGI<sup>5</sup> have aimed to create autonomous agents capable of performing tasks to meet user requirements. The capabilities of agents to invoke multi-modal tools have been confirmed by recent studies such as Toolformer (Schick et al., 2023) and HuggingGPT (Shen et al., 2023). However, these agents are currently limited to processing only certain preliminary tasks or operations that conform to predefined rules. This limitation poses a challenge when planning and executing complex tasks. ReAct (Yao et al., 2022), SayCan (Ahn et al., 2022), and GPT4Tools, have demonstrated the feasibility of task reasoning and planning.

Building upon the advancements in language agent development and the demonstrated capabilities of autonomous decision-making agents, our research introduces a novel language agent designed specifically for stereotype detection in text-to-image models. This agent leverages the power of LLMs and various tools to not only identify but also quantify the extent of stereotypes in generated images.

# 6 CONCLUSION AND FUTURE WORK

This study is the first to demonstrate the feasibility of leveraging language agent for large-scale stereotype detection in AI-generated images. Our experiments reveal that current state-of-the-art text-to-image generative models inherently contain substantial latent stereotypes related to gender, race and religion. Given the extensive deployment of these models in commercial settings, this introduces unforeseen ethical challenges. As a result, we strongly recommend an increased awareness of these ethical risks within the academic community and among corporations involved in product design. We underscore the significance of incorporating artificial intelligence technologies for risk surveillance and regulation. The framework proposed in this study, due to its comprehensive automation and proven efficacy, emerges as one of the most promising strategies in this field.

# 7 FUTURE WORK

In our forthcoming research, we aspire to conduct a more comprehensive exploration of the ethical challenges inherent in text-to-image models, with a particular emphasis on the necessity for effective bias governance. Our intention is to engage in an in-depth analysis of the potential impacts and implications of stereotype detection in real-world applications. This will encompass an investigation into the unforeseen repercussions of bias mitigation strategies, the crucial role of human discernment in identifying stereotypes, and the intricate equilibrium that must be maintained between upholding freedom of expression and mitigating potential risks.

Our examination of the training data for the Stable Diffusion model has unveiled significant biases inherent within the dataset. For example, over  $82\%$  of images related to crime feature individuals of African descent. This discovery highlights the imperative to circumvent stereotyping during the training phase. In our subsequent research, we plan to incorporate training samples that represent a variety of skin tones and genders for identical drawing prompts, thereby ensuring a more diverse dataset.

# REFERENCES

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022.  
Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, pp. 1493-1504, 2023a.  
Federico Bianchi, Pratyusha Kalluri, Esin Durmus, Faisal Ladhak, Myra Cheng, Debora Nozza, Tatsunori Hashimoto, Dan Jurafsky, James Zou, and Aylin Caliskan. Easily accessible text-to-image generation amplifies demographic stereotypes at large scale. In Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency, pp. 1493-1504, 2023b.  
Pierre Bourdieu. Distinction a social critique of the judgement of taste. In Inequality, pp. 287-318. Routledge, 2018.  
Luke Breitfeller, Emily Ahn, David Jurgens, and Yulia Tsvetkov. Finding microaggressions in the wild: A case for locating elusive phenomena in social media posts. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 1664-1674, 2019.  
Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183-186, 2017.  
Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-to-image generative models. arXiv preprint arXiv:2202.04053, 2022.  
Mai ElSherief, Caleb Ziems, David Muchlinski, Vaishnavi Anupindi, Jordyn Seybolt, Munmun De Choudhury, and Diyi Yang. Latent hatred: A benchmark for understanding implicit hate speech. arXiv preprint arXiv:2109.05322, 2021.  
Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.  
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.  
Akshita Jha, Aida Davani, Chandan K Reddy, Shachi Dave, Vinodkumar Prabhakaran, and Sunipa Dev. Seegull: A stereotype benchmark with broad geo-cultural coverage leveraging generative models. arXiv preprint arXiv:2305.11840, 2023.

Austin C Kozlowski, Matt Taddy, and James A Evans. The geometry of culture: Analyzing the meanings of class through word embeddings. American Sociological Review, 84(5):905-949, 2019.  
Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888-12900. PMLR, 2022.  
Binny Mathew, Punyajoy Saha, Seid Muhie Yimam, Chris Biemann, Pawan Goyal, and Animesh Mukherjee. Hatexplain: A benchmark dataset for explainable hate speech detection. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 14867-14875, 2021.  
Ranjita Naik and Besmira Nushi. Social biases through the text-to-image generation lens. arXiv preprint arXiv:2304.06034, 2023.  
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.  
Joon Sung Park, Joseph C O'Brien, Carrie J Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. arXiv preprint arXiv:2304.03442, 2023.  
Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.  
Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748-8763. PMLR, 2021. URL http://proceedings.mlr.org. press/v139/radford21a.html.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 10674-10685. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01042. URL https://doi.org/10.1109/CVPR52688.2022.01042.  
Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A Smith, and Yejin Choi. Social bias frames: Reasoning about social and power implications of language. arXiv preprint arXiv:1911.03891, 2019.  
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.  
Anna Schmidt and Michael Wiegand. A survey on hate speech detection using natural language processing. In Proceedings of the fifth international workshop on natural language processing for social media, pp. 1-10, 2017.  
Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Huggingppt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023.

Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.  
Yashar Talebirad and Amirhossein Nadiri. Multi-agent collaboration: Harnessing the power of intelligent llm agents. arXiv preprint arXiv:2306.03314, 2023.  
Bertie Vidgen, Tristan Thrush, Zeerak Waseem, and Douwe Kiela. Learning from the worst: Dynamically generated datasets to improve online hate detection. arXiv preprint arXiv:2012.15761, 2020.  
Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, and Dongkuan Xu. Rewoo: Decoupling reasoning from observations for efficient augmented language models. arXiv preprint arXiv:2305.18323, 2023.  
Hui Yang, Sifu Yue, and Yunzhong He. Auto-gpt for online decision making: Benchmarks and additional opinions. arXiv preprint arXiv:2306.02224, 2023.  
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022.  
Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, and Mrinmaya Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text. arXiv preprint arXiv:2305.13304, 2023.
