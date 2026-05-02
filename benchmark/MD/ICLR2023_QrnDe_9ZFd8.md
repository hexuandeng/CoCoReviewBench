# TASK AMBIGUITY IN HUMANS AND LANGUAGE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Language models have recently achieved strong performance across a wide range of NLP benchmarks. However, real world tasks are often poorly specified, and agents must deduce the intended behavior from a combination of context, instructions, and examples. We investigate how both humans and models behave in the face of such task ambiguity by proposing AmbiBench, a new benchmark of six ambiguously-specified classification tasks. We evaluate humans and models on AmbiBench by seeing how well they identify the intended task using 1) instructions with varying degrees of ambiguity, and 2) different numbers of labeled examples. We find that the combination of model scaling (to 175B parameters) and reinforcement learning from human feedback (RLHF) enables models to approach or exceed the accuracy of human participants across tasks, but that either one of these alone is not sufficient. In addition, we show how to dramatically improve the accuracy of language models trained without RLHF by finetuning on a small number of ambiguous in-context examples, providing a promising direction for teaching models to generalize well in the face of ambiguity.

# 1 INTRODUCTION

Language models have recently been applied to a wide range of NLP benchmarks, ranging from question answering, summarization, and logical reasoning, to solving riddles, dark humor detection, and ASCII word recognition (Brown et al., 2020; Srivastava et al., 2022). Performance across tasks has improved as models and datasets have grown in size, raising the prospect of a route towards generalist NLP models with broad utility.

However, one feature many of these benchmarks share is that they are carefully designed to make the desired task very clear to the language model, since this is a prerequisite for establishing performance on that task. Unfortunately, real-world uses of language models are not likely to feature such thought and clarity in their task specification. Rather than iterating over and perfecting a specification for their tasks, everyday users of language models may wish to effortlessly define tasks on

![](images/02fdc690a12a78821abf159db1d790c5291e367d2a8553c19942d75e5bd7ac74.jpg)  
Figure 1: Complex tasks are often hard to specify precisely, leaving important pieces of information missing. Agents should be able to fill in the blanks by combining information from instructions and examples in order to identify the intended behavior.

an as-needed basis, without worrying that they will be misunderstood. More pressingly, in complex domains featuring high-dimensional inputs and outputs (e.g. programming, verification, generation) it is unlikely that even a thoughtful task specification will manage to perfectly capture all the features of an input and output which are salient or not salient to the task. This is especially important for safe and robust deployment of language models, as such undesirable dependencies can be hidden hazards that are only revealed when a model fails catastrophically in a new setting (Geirhos et al., 2020).

To operationalize this problem, we introduce AmbiBench, a new benchmark of six ambiguously-specified tasks. Each input in AmbiBench is a sentence (e.g. The dog is in the meadow) that has multiple associated classification tasks based on different linguistic features (e.g. contains an animal, contains an outdoor location). Task ambiguity arises when more than one task is consistent with the provided instructions or labeled examples.<sup>1</sup>

We establish how well different models and humans perform on a task given a wide range of task specifications, including clear vs unclear instructions, and zero vs many examples. We find that the largest models trained with reinforcement learning from human feedback (RLHF) approach or outperform humans across all specifications we try.

We also show how to improve standard language models' performance by finetuning on a small set of in context examples that demonstrate the desired generalization. This form of meta-learning dramatically improves a model's ability to learn new ambiguously-specified tasks, but crucially only when the finetuning tasks are ambiguous, not for control tasks of a similar format that are unambiguous. This suggests a possible mechanism for why RLHF models outperform standard language models (discussed in Section 4.4), as well a promising direction for improving how models learn tasks in ambiguous contexts.

To summarize our contributions, we:

1. Introduce and motivate the problem of studying task ambiguity in large language models  
2. Evaluate humans and models on a new benchmark of ambiguously-specified tasks, demonstrating that while most models fail to disambiguate the intended task well, sufficiently large models trained with RLHF are able to approach or even exceed the performance of our human participants to resolve the ambiguity between tasks  
3. Show how finetuning on ambiguous in-context prompts and examples can enable traditional language models to surpass the performance of RLHF models when evaluated on unseen tasks, providing a promising route towards models that capably manage task ambiguity

# 2 RELATED WORK

# 2.1 AMBIGUITY IN NATURAL LANGUAGE PROCESSING

Ambiguity is a well-studied topic in NLP, with work spanning topics as diverse as search queries (Cronen-Townsend & Croft, 2002; Wang & Agichtein, 2010), question answering (Min et al., 2020; Zhang & Choi, 2021), named entities (Bunescu & Pasca, 2006; Cucerzan, 2007; Dredze et al., 2010), coreference resolution (Webster et al., 2018), machine translation (Stanovsky et al., 2019), and information-seeking dialogues (Aliannejadi et al., 2019; Guo et al., 2021; Aliannejadi et al., 2021; Sun et al., 2022; Wu et al., 2022).

Our work differs from these prior streams of work by studying task ambiguity (Finn et al., 2018; Tamkin et al., 2022b), where the task the agent is being asked to perform is ambiguous, rather than an ambiguous input for a clearly specified task. This is of special relevance for self-supervised learning models that can be easily adapted to a broad range of downstream tasks (Bommasani et al., 2021). In these settings, models must infer the correct task from a user's specification, as opposed to a possibly unsafe or undesirable task that is also consistent with that specification.

# 2.2 IN- CONTEXT LEARNING AND PROMPTING

Task ambiguity is especially relevant for language models, which can be adapted for many different tasks via in-context learning (Brown et al., 2020; Tamkin et al., 2021a; Bommasani et al., 2021; Liu et al., 2022b). Much work has attempted to improve the ability of such models to perform in-context learning by calibrating model predictions (Zhao et al., 2021), choosing good examples for the prompt (Liu et al., 2022a), finetuning models on natural language descriptions of tasks (Zhong et al., 2021; Wei et al., 2022; Sanh et al., 2022), or by training models with reinforcement learning from human feedback (Bai et al., 2022; Ouyang et al., 2022).

Prior work suggests that the language modeling objective can bias the model towards tasks previously seen in the training data, potentially overriding the effects of the provided instructions (Webson & Pavlick, 2022) or few-shot examples (Min et al., 2022b; Kim et al., 2022). In this work, we attempt to control for such factors by constructing a benchmark where multiple tasks are consistent with a single instruction or example, requiring the model to leverage multiple signals to disambiguate the intended task without relying on learned priors.

Past work has also explored finetuning on in-context learning examples (Chen et al., 2022; Min et al., 2022a). We extend this line of work to show how the content of these training examples can dramatically affect generalization: Finetuning on ambiguously-specified examples (but not a control set of unambiguous tasks) can enable the model to better disambiguate unseen task specifications—vastly improving the performance of traditional language models without the need for reinforcement learning from human feedback.

# 2.3 TASK AMBIGUITY

Systems capable of performing different tasks may experience task ambiguity, where the provided examples do not uniquely identify the user's intent (Finn et al., 2018; Tamkin et al., 2021a). One form of task ambiguity is shortcut learning (Geirhos et al., 2020), where the training examples can all be solved by identifying a simple feature (e.g. a watermark) as opposed to learning the intended task (e.g. object classification). Task ambiguity is particularly important in few-shot learning settings, where the small number of examples provided may leave the intended task ambiguous (Finn et al., 2018; Tamkin et al., 2021a). In this work, we study task ambiguity for in-context learning with language models and humans, considering not only the role of examples but also natural language instructions.

# 3 THE AMBIBENCH BENCHMARK

As a first step towards studying task ambiguity in language models, we construct the AmbiBench benchmark, a collection of six different sentence classification tasks. The goal of AmbiBench is to construct a testbed of minimal complexity where we can control and measure the degree of ambiguity in various task specifications. Despite the simplicity of this benchmark, we find large variability in performance across different language models.

# 3.1 SELECTION OF TASKS

AmbiBench contains six binary classification tasks, where a human or model must detect a simple linguistic feature in an input sentence—for example, whether an outdoor location or an animal was mentioned—and then output the appropriate classification letter (X or Y). Crucially, however, each sentence has two linguistic features (e.g. The duck is in the canyon has the features animal and outdoor location). The six features are grouped into three pairs, shown in Table 1, where a single sentence will have one feature in each pair.

Identifying the salient feature for the task thus requires either the presence of an informative instruction (e.g., Output 'X' if the sentence contains an outdoor location and 'Y' otherwise) or multiple labeled examples to identify which feature determines the label.

Tasks were chosen to represent a set of common semantic categories, excluding substring information such as periods and capitalization that might be much easier for humans to represent than

Table 1: The AmbiBench benchmark. Left: Each task involves detecting a salient feature in a sentence (bolded in the examples on the right). The same sentence could potentially receive a label according to two features, requiring a learner to use additional information (task instructions or other examples) to disambiguate the intended behavior. Right: Varying levels of instruction are inserted before the examples, providing different degrees of information about the format and salient feature of the task. See Figure 1 for an example of a complete prompt.  

<table><tr><td>Salient feature</td><td>Example sentence</td><td></td><td></td></tr><tr><td>human subject</td><td>The researcher/bear is in the museum.</td><td>Instruction</td><td>Example</td></tr><tr><td>indoor location</td><td>The researcher is in the museum/meadow.</td><td>Uninformative</td><td>Output ‘X’ if the sentence contains a [category with-held] and ‘Y’ otherwise.</td></tr><tr><td>religious leader</td><td>He is in the museum with the rabbi/judge.</td><td></td><td></td></tr><tr><td>pronoun gender</td><td>He/She is in the museum with the judge.</td><td>Informative</td><td>Output ‘X’ if the sentence contains a proper noun and ‘Y’ otherwise.</td></tr><tr><td>proper noun</td><td>Paul Atreides/The director may not be in the film studio.</td><td></td><td></td></tr><tr><td>negation</td><td>Paul Atreides may/may not be in the film studio.</td><td></td><td></td></tr></table>

models. See Figure 1 for an example of this disambiguation process, and Table 1 for a full list of tasks and accompanying instructions.

# 3.2 TASK CONSTRUCTION

AmbiBench examples are programmatically constructed from a set of templates, allowing precise control over the amount of task ambiguity in each in-context example (see Table 1 and Appendix C for more details). Templated data has seen a recent resurgence in NLP for the purposes of evaluating large language models (Lake & Baroni, 2018; Srivastava et al., 2022), as they enable precise control and coverage over different variables of study. Furthermore, recent work has shown strong correlation between test performance on synthetic and naturalistic data Liu et al. (2021), suggesting that insights gained from such datasets may extend to a broader range of natural contexts. In our case, this dataset construction process enables us to formalize and characterize the degree of task ambiguity in different examples, allowing us to measure how well models can disambiguate between multiple potential classification tasks they may be asked to perform.

# 3.3 IN-CONTEXT LEARNING FORMATS

There are several ways an instruction and in-context examples can be assembled into a prompt for a language model. Given the demonstrated sensitivity of models to such parameters (Zhao et al., 2021; Liu et al., 2022b; Lu et al., 2022), we consider two different prompt formats, and report averaged performance across them:

# Arrow:

Output 'X' if the sentence contains an outdoor location and 'Y' otherwise.  
The worm is in the meadow  
 $>\mathrm{X}$   
The duck is in the canyon  
 $>\mathrm{Y}$   
...

# Q/A:

Output 'X' if the sentence contains an outdoor location and 'Y' otherwise.  
Q: The worm is in the meadow  
A: X  
Q: The duck is in the canyon  
A: Y

# 4 EXPERIMENTS

We use AmbiBench to investigate how humans and language models respond to and resolve different manifestations of task ambiguity.

# 4.1 EXPERIMENTAL SETUP

First, we describe the different language models and human participants we study, and how we evaluate them.

Language models We examine a range of different models, including both OpenAI's normal language models and their "instruct" models trained with reinforcement learning from human feedback (RLHF) (Brown et al., 2020; Ouyang et al., 2022). In the rest of the paper, OpenAI's model names are reported as listed in their documentation  $^{2}$  (e.g. davinci, text-curie-001). The instruct models have a numerical suffix (e.g. 002) and the model size increases as one progresses through the alphabet (ada, babbage, curie, davinci). See Appendix A for more information. We also evaluate AI21 Studio's 178B-parameter Jurassic-1 Jumbo language model (jurassic-jumbo) (Lieber et al.), as well as the 11B-parameter  $\mathrm{T}0++$  model (t0pp) (Sanh et al., 2022), which was finetuned on a large corpus of task instructions, but not trained with RLHF. This diversity of model providers, model sizes, and training strategies enables us to identify which ingredients are most crucial for resolving task ambiguity.

Human evaluation We compare model performance with the performance of human participants, evaluated by hiring contractors from Prolific (Palan & Schitter, 2017). We aimed to evaluate the human participants as similarly to language models as possible within the confines of an online survey methodology. We showed human participants exactly the same input that language models received, with minimal additional information presented to them before the study began. Participants typed the answer label (i.e. X or Y into a textbox, as opposed to choosing from a set of preselected options, to mitigate priming effects and mirror the setting for language models. We also recruited a new participant for every single in-context instance, to avoid humans learning across examples in ways that language models do not. Human participants were paid \(12 - 13/hr, in line with Prolific wage recommendations. See Appendix B for more details.

# 4.2 TASK DISAMBIGUATION USING NATURAL LANGUAGE INSTRUCTIONS

One way that people resolve task ambiguity is through the use of natural language instructions, which can explicitly indicate different aspects of the task. Past work has suggested that the best models do not fruitfully use natural-language instructions, as evidenced by experiments leveraging irrelevant or misleading directions (Webson & Pavlick, 2022). However, these experiments were performed for established natural language processing tasks that lack the explicit task ambiguity we study here, and did not investigate more recent models trained with reinforcement learning from human feedback (Bai et al., 2022; Ouyang et al., 2022).

As a first set of experiments, we evaluate how humans and models are able to use differing levels of instruction to resolve task ambiguity. The humans and models receive two in-context examples, one from each class. Humans and models are then presented with a third query example in order to elicit the predicted output letter. Because there is only one example of each class, but two possible features, the salient feature can not be identified from these two examples alone, requiring the model to use the instruction to disambiguate the task. The order of the examples, the example format, as well as the assignment of each class to an output letter (X or Y) are randomized. Each model is evaluated with 720 different in-context prompts for each level of instruction.

We consider two different levels of instruction:

1. Informative instruction: The model receives a full specification of the salient feature and output format. Ex: Output 'X' if the sentence contains an animal and 'Y' otherwise.

![](images/5032be7139fc612a924b081177e3f63087730ca6a289716eaef3457de1e1b055.jpg)

![](images/c58e8a66a55f6a87be48017571960eb15ee21ac2d56feaeda3b3ba0187b3df93.jpg)  
(a) Uninformative Instructions  
Figure 2: The largest RLHF model (text-davinci-002) approaches human accuracy for both uninformative and informative instructions. Accuracy of humans and other models for tasks prompted with an instruction and two in-context examples. Error bars show  $95\%$  bootstrap CIs.  
(b) Informative Instructions

2. Uninformative instruction: The model receives the output format but the salient feature is redacted. Ex: Output 'X' if the sentence contains a [category withheld] and 'Y' otherwise.

Our setting is simple enough that crafting an informative instruction is not challenging, making it tractable for us to study. However, the insights from this simple case may generalize to more complex tasks language models may be used for, where operators may be more likely to inadvertently omit crucial information.

# 4.2.1 RESULTS

In the case of uninformative instructions, humans as well as many models are able to achieve approximately  $50\%$  accuracy by correctly understanding the output format and choosing X and Y at random. However, some non-instruct models, including jurassic-jumbo, ada, babbage, and curie, often output values other than X or Y (e.g. Z), leading to lower performance. Finally, in the case of negation, humans achieve  $100\%$  accuracy despite lacking an instruction identifying the salient feature. This may be due to an inductive bias present in people (but not models) that makes negation an especially salient feature.

In the case of informative instructions, humans perform the strongest at this task, with perfect performance in all but one task, showing that they are broadly able to identify the salient feature in the text inputs and output the correct letter. Humans are closely followed by the text-davinci-002 RLHF model (see Figure 2). All other models perform relatively poorly, including the non-RLHF  $175\mathrm{B}+$  parameter davinci and j1-jumbo models, as well as the smaller RLHF models curie, babbage, and ada. This seems to suggest that most models are not reliably able to follow simple instructions to disambiguate a task, but that a combination of large-scale training and RLHF can approach human performance in some settings.

![](images/e7106057d6b9103896b51b8359362af230767d5159cb9164fc253356842575d6.jpg)  
Figure 3: The largest RLHF model (text-davinci-002) outperforms human participants at disambiguating the intended task. Accuracy as the number of examples in the in-context window grows. Surprisingly, the smaller curie model reliably outperforms the larger davinci model across the examples. In addition, the RLHF training hurts at curie scale, but dramatically helps at davinci scale. RLHF models shown as dotted lines. Shaded regions are  $95\%$  bootstrap CIs.

# 4.3 TASK DISAMBIGUATION USING MULTIPLE EXAMPLES

While instructions are a simple way to specify a task, multiple examples can also disambiguate between different tasks a user might intend. For example, if there are multiple features that could explain the label of a single example, more examples will gradually identify the salient feature provided the features are sufficiently decorrelated.

We investigate whether models and humans can identify the salient features in AmbiBench as the number of examples grows from zero (where the task is completely ambiguous) to twenty (where the task is almost certainly unambiguous). Both models and human participants predict the answer for each example, then are presented with the correct answer for that example and the next query. All aspects of the examples are randomized, including the salient feature (chosen randomly, then held constant across the entire in-context example), the assignment of X or Y to the salient feature, the example order, and the specific instantiations of the salient and non-salient features for each example. Each model is evaluated with 720 different in-context prompts, each containing 20 examples.

# 4.3.1 RESULTS

To our surprise, the best language model (the RLHF-trained text-davinci-002) significantly outperformed the human participants (Figure 3). The human participants performed comparably to the j1-jumbo and curie models, which in turn performed better than the rest of OpenAI's models. The t0pp models exhibited large sensitivity to the prompt format—the model outputted invalid answers (typically nouns) for the arrow format. However, considering only the Q/A format, t0pp still only performed near chance.

We do not observe evidence that the imperfect human performance is due to low-quality participants or bot activity—human annotators mostly spent between 4 and 8 minutes on the task, did not appear to be guessing at random, and typically left thoughtful comments or feedback on the survey. That said, we caution against claims of "superhuman performance" given that annotators represent merely

![](images/9b8a2b0fc765d8cf4b0f42da149989164e7e07f129888225b992301f678c87d2.jpg)  
(a) Subject and location

![](images/9ea5181235a13dac0c85e4d8588e1e643006eb671cbdf4b0aae4210f1a841c3c.jpg)  
(b) Religious and pronoun

![](images/508220564875213fe37388fc7615930b9cf97da67874ba4df195e7be0f2efc9b.jpg)  
(c) Proper noun and negation

![](images/fd3f34b7d6db2dee78daf53afe58b82801fce90691055be6def3ab851b34d903.jpg)  
Figure 4: Finetuning on ambiguous in-context examples dramatically improves accuracy on unseen tasks that are ambiguously specified. Accuracy after finetuning davinci on ambiguous and non-ambiguous (control) in-context examples. Models are finetuned on 272 examples from four tasks, then evaluated on the two held-out tasks (subfigure captions). Shaded regions are  $95\%$  bootstrap CIs.  
(d) Average across (a-c)

a sample from a single distribution of humans, and they may have experienced fatigue or distraction across the 20-example episode.

# 4.4 FINETUNING A MODEL TO GENERALIZE WELL IN THE FACE OF AMBIGUITY

The strong performance of the RLHF models relative to the normal language models in Section 4.3 is somewhat surprising—these models are described as being trained to follow human instructions, not to resolve ambiguity in instructions by analyzing the training examples. While the training dataset of these models was not released, Ouyang et al. (2022) do report that some of the crowdsourced examples for the model contain instructions along with few-shot examples. If some of these instructions were ambiguous, the model may have learned from those examples to resolve that ambiguity more effectively.

Motivated by this hypothesis, we investigate whether finetuning on a small corpus of ambiguous in-context learning examples is sufficient to close the gap between the best-performing text-davinci-002 RLHF model and the normal davinci language model. To do so, we partition the six AmbiBench tasks into three folds, each containing four finetuning tasks and two evaluation tasks (following the feature pairs in Table 1). We finetune on 68 examples from each task (two for each number of examples, from 4 to 20), and evaluate on 240 examples randomly drawn from the other two tasks. While all tasks share some structural similarities, this partitioning en

sures that the model is being tested on held-out features that never appeared in its finetuning dataset. Models are finetuned using the OpenAI API (see Appendix D for details).

As an added control to see whether ambiguity truly matters in the finetuning dataset, we also finetune on the same splits of data, but where only one feature varies within each in-context example. For example, if the two features were animal and indoor location, a given in-context example may contain examples with both animals and humans, but only indoor locations. See Appendix D for more details.

# 4.4.1 RESULTS

Despite the small training dataset consisting of only 4 tasks (with 272 examples total), we find we are able to completely close the gap between the RLHF models and our finetuned models across all three splits of our data. Indeed, our finetuned models appear to even outperform the text-davinci-002 model across the first eight examples.

Crucially, we do not observe any improvement for the control finetuned models, which were finetuned on the same kinds of examples but without task ambiguity between two potential salient features. This indicates that ambiguity is the crucial ingredient explaining the success of our finetuned models, and supports the hypothesis that the few-shot examples in text-davinci-002's RLHF data may contribute to its strong performance.

More broadly, these results suggest that explicitly finetuning models to adapt to task ambiguity may result in a generalized capacity to do so across different kinds of ambiguous task specifications.

# 5 DISCUSSION AND CONCLUSION

We present the AmbiBench testbed for studying task ambiguity in language models and humans, showing how it can be used to investigate different factors influencing task ambiguity, as well as identify promising interventions that can improve how models resolve it.

# 5.1 LIMITATIONS

Our study has several limitations. First, we conduct a scientific and controlled study of task ambiguity in language models; this naturally elides many of the messy nuances of task ambiguity in the real world, and should be seen as complementary to in-the-wild case studies. Second, despite our efforts to match the experimental conditions between humans and language models, humans do require some additional instructions to orient them to the task interface, and may suffer from fatigue and uneven concentration across the length of a 20-example learning episode. Finally, our work studies of task ambiguity between two possible tasks—however, in general task ambiguity may occur between arbitrary many tasks, or even an infinitely large family of tasks.

# 5.2 FUTURE WORK

Task ambiguity is a pressing problem in machine learning with relevance for safety, fairness, and interpretability. Going forward, we are excited by the potential to study task ambiguity in self-supervised models trained on many different modalities (Reed et al., 2022; Tamkin et al., 2021b; 2022a; Alayrac et al., 2022), including multimodal settings, as self-supervised learning is applied increasingly broadly. The strong performance of our models on the AmbiBench testbed also suggests the tractability of studying task ambiguity in more complex real-world settings where language models are used, such as software engineering, law, and education, as well as assessing the efficacy of our proposed finetuning interventions.

Ethics statement Our research makes use of human subject experiments via the Prolific platform (Palan & Schitter, 2017). We pay workers a minimum of $12-13 / hour, consistent with Prolific wage recommendations.5. We also made efforts to solicit feedback from participants via pilot studies, which led to several changes to the research methodology to make the survey experience more pleasant (e.g. keyboard shortcuts to navigate the study more efficiently). Anecdotally, many participants expressed that they enjoyed the study:

1. Zap! I'm suddenly back in high school with Ms. Langston's English class. Thank you for the smiles!  
2. this was fun!  
3. this was a really good study

Reproducibility statement We include detailed experimental settings in Sections 1, 4.2, 4.3, 4, A, D, C. We will also release our codebase to facilitate reproducing these results.

# REFERENCES

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. ArXiv, abs/2204.14198, 2022.  
Mohammad Aliannejadi, Hamed Zamani, Fabio A. Crestani, and W. Bruce Croft. Asking clarifying questions in open-domain information-seeking conversations. Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2019.  
Mohammad Aliannejadi, Julia Kiseleva, Aleksandr Chuklin, Jeffrey Dalton, and Mikhail S. Burtsev. Building and evaluating open-domain dialogue corpora with clarifying questions. *ArXiv*, abs/2109.05794, 2021.  
Yushi Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, T. J. Henighan, Nicholas Joseph, Saurav Kadavath, John Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Christopher Olah, Benjamin Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv, abs/2204.05862, 2022.  
Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, S. Buch, Dallas Card, Rodrigo Castellon, Niladri S. Chatterji, Annie S. Chen, Kathleen A. Creel, Jared Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren E. Gillespie, Karan Goel, Noah D. Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas F. Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, O. Khattab, Pang Wei Koh, Mark S. Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir P. Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Benjamin Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, J. F. Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Robert Reich, Hongyu Ren, Frieda Rong, Yusuf H. Roohani, Camilo Ruiz, Jack Ryan, Christopher R'e, Dorsa Sadigh, Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishna Parasuram Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramér, Rose E. Wang, William

Wang, Bohan Wu, Jiajun Wu, Yuhuai Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei A. Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and risks of foundation models. ArXiv, abs/2108.07258, 2021.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, T. J. Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeff Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. *ArXiv*, abs/2005.14165, 2020.  
Razvan C. Bunescu and Marius Pasca. Using encyclopedic knowledge for named entity disambiguation. In EACL, 2006.  
Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, and He He. Meta-learning via language model in-context tuning. ArXiv, abs/2110.07814, 2022.  
Steve Cronen-Townsend and W. Bruce Croft. Quantifying query ambiguity. 2002.  
Silviu Cucerzan. Large-scale named entity disambiguation based on wikipedia data. In EMNLP, 2007.  
Mark Dredze, Paul McNamee, Delip Rao, Adam Gerber, and Timothy W. Finin. Entity disambiguation for knowledge base population. In COLING, 2010.  
Chelsea Finn, Kelvin Xu, and Sergey Levine. Probabilistic model-agnostic meta-learning. In NeurIPS, 2018.  
Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard S. Zemel, Wieland Brendel, Matthias Bethge, and Felix Wichmann. Shortcut learning in deep neural networks. Nat. Mach. Intell., 2:665-673, 2020.  
M. Guo, Mingda Zhang, Siva Reddy, and Malihe Alikhani. Abg-coqa: Clarifying ambiguity in conversational question answering. In AKBC, 2021.  
Junyeob Kim, Hyuhng Joon Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-Woo Lee, Sang goo Lee, Kang Min Yoo, and Taeuk Kim. Ground-truth labels matter: A deeper look into input-label demonstrations. ArXiv, abs/2205.12685, 2022.  
Brenden M. Lake and Marco Baroni. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In ICML, 2018.  
O Lieber, O Sharir, B Lentz, and Y Shoham. Jurassic-1: Technical details and evaluation, white paper, ai21 labs, 2021. URL: https:// uploads-ssl. webflow. com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_ tech_paper. pdf.  
Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context examples for gpt-3? In DEELIO, 2022a.  
Nelson F. Liu, Tony Lee, Robin Jia, and Percy Liang. Can small and synthetic benchmarks drive modeling innovation? a retrospective study of question answering modeling approaches. ArXiv, abs/2102.01065, 2021.  
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys (CSUR), 2022b.  
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In ACL, 2022.  
Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. Ambigqa: Answering ambiguous open-domain questions. In EMNLP, 2020.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. ArXiv, abs/2110.15943, 2022a.  
Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? ArXiv, abs/2202.12837, 2022b.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke E. Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Francis Christiano, Jan Leike, and Ryan J. Lowe. Training language models to follow instructions with human feedback. ArXiv, abs/2203.02155, 2022.  
Stefan Palan and Christian Schitter. Prolific.ac—a subject pool for online experiments. Journal of Behavioral and Experimental Finance, 17:22-27, 2017.  
Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. arXiv preprint arXiv:2205.06175, 2022.  
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang A. Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Stella Rose Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. ArXiv, abs/2110.08207, 2022.  
Aarohi Srivastava, Abhinav Rastogi, Abhishek B Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, Alicia Parrish, Allen Nie, Aman Hussain, Amanda Askell, Amanda Dsouza, Ameet Annasaheb Rahane, Anantharaman S. Iyer, Anders Johan Andreassen, Andrea Santilli, Andreas Stuhlmuller, Andrew M. Dai, Andrew D. La, Andrew Kyle Lampinen, Andy Zou, Angela Jiang, Angelica Chen, Anh Vuong, Animesh Gupta, Anna Gottardi, Antonio Norelli, Anu Venkatesh, Arash Gholamidavoodi, Arfa Tabassum, Arul Menezes, Arun Kirubarajan, Asher Mullokandov, Ashish Sabharwal, Austin Herrick, Avia Efrat, Aykut Erdem, Ayla Karakacs, Bridget R. Roberts, Bao Sheng Loe, Barret Zoph, Bartlomiej Bojanowski, Batuhan Ozyurt, Behnam Hedayatnia, Behnam Neyshabur, Benjamin Inden, Benno Stein, Berk Ekmekci, Bill Yuchen Lin, Blake Stephen Howald, Cameron Diao, Cameron Dour, Catherine Stinson, Cedrick Argueta, C'esar Ferri Ram'irez, Chandan Singh, Charles Rathkopf, Chenlin Meng, Chitta Baral, Chiyu Wu, Chris Callison-Burch, Chris Waites, Christian Voigt, Christopher D. Manning, Christopher Potts, Cindy Tatiana Ramirez, Clara Rivera, Clemencia Siro, Colin Raffel, Courtney Ashcraft, Cristina Garbacea, Damien Sileo, Daniel H Garrette, Dan Hendrycks, Dan Kilman, Dan Roth, Daniel Freeman, Daniel Khashabi, Daniel Levy, Daniel Gonz'alez, Danny Hernandez, Danqi Chen, Daphne Ippolito, Dar Gilboa, David Dohan, D. Drakard, David Jurgens, Debajyoti Datta, Deep Ganguli, Denis Emelin, Denis Kleyko, Deniz Yuret, Derek Chen, Derek Tam, Dieuwke Hupkes, Diganta Misra, Dilyar Buzan, Dimitri Coelho Mollo, Diyi Yang, Dong-Ho Lee, Ekaterina Shutova, Ekin Dogus Cubuk, Elad Segal, Eleanor Hagerman, Elizabeth Barnes, Elizabeth P. Donoway, Ellie Pavlick, Emanuele Rodola Emma FC Lam, Eric Chu, Eric Tang Erkut Erdem Ernie Chang Ethan A. Chi Ethan Dyer Ethan Jerzak Ethan Kim Eunice Engefu Manyasi Evgenii Zheltonozhskii Fan Xia Fatemeh Siar Fernando Mart'inez-Plumed Francesca Happ'e François Chollet Frieda Rong Gaurav Mishra Genta Indra Winata Gerard de Melo German Kruszewski Giambattista Parascandolo Giorgio Mariani Gloria Wang Gonzalo Jaimovitch-L'opez Gregor Betz Guy Gur-Ari Hana Galijasevic Han Sol Kim Hannah Rashkin Hanna Hajishirzi Harsh Mehta Hayden Bogar Henry Shevlin Hinrich Schutze Hiromu Yakura Hongming Zhang Hubert Wong Ian AikSoon Ng Isaac Noble Jaap Jumelet Jack Geissinger John Kernion Jacob Hilton Jaehoon Lee Jaime Fernandez Fisac J. Brooker Simon James Koppel James Zheng James Zou Jan Koco'n

Jana Thompson, Jared Kaplan, Jarema Radom, Jascha Narain Sohl-Dickstein, Jason Phang, Jason Wei, Jason Yosinski, Jekaterina Novikova, Jelle Bosscher, Jenni Marsh, Jeremy Kim, Jeroen Taal, Jesse Engel, Jesujoba Oluwadara Alabi, Jiacheng Xu, Jiaming Song, Jillian Tang, Jane W Waweru, John Burden, John Miller, John U. Balis, Jonathan Berant, Jorg Frohberg, Jos Roszen, Jose Hernández-Orallo, Joseph Boudeman, Joseph Jones, Joshua B. Tenenbaum, Joshua S. Rule, Joyce Chua, Kamil Kanclerz, Karen Livescu, Karl Krauth, Karthik Gopalakrishnan, Katerina Ignatyeva, Katja Markert, Kaustubh D. Dhole, Kevin Gimpel, Kevin Ochieng' Omondi, Kory Wallace Mathewson, Kristen Chiafullo, Ksenia Shkaruta, Kumar Shridhar, Kyle McDonell, Kyle Richardson, Laria Reynolds, Leo Gao, Li Zhang, Liam Dugan, Lianhui Qin, Lidia Contreras-Ochando, Louis-Philippe Morency, Luca Moschella, Luca Lam, Lucy Noble, Ludwig Schmidt, Luheng He, Luis Oliveros Col'on, Luke Metz, Lutfi Kerem cSenel, Maarten Bosma, Maarten Sap, Maartje ter Hoeve, Madotto Andrea, Maheen Saleem Farooqi, Manaal Faruqui, Mantas Mazeika, Marco Baturan, Marco Marelli, Marco Maru, M Quintana, Marie Tolkiehn, Mario Giulianielli, Martha Lewis, Martin Potthast, Matthew Leavitt, Matthias Hagen, M'aty'as Schubert, Medina Baitemirova, Melissa Arnaud, Melvin Andrew McElrath, Michael A. Yee, Michael Cohen, Mi Gu, Michael I. Ivanitskiy, Michael Starritt, Michael Strube, Michal Swkedrowski, Michele Bevilacqua, Michihiro Yasunaga, Mihir Kale, Mike Cain, Mimee Xu, Mirac Suzgun, Monica Tiwari, Mohit Bansal, Moin Aminnaseri, Mor Geva, Mozhdeh Gheini, T MukundVarma, Nanyun Peng, Nathan Chi, Nayeon Lee, Neta Gur-Ari Krakover, Nicholas Cameron, Nicholas S. Roberts, Nicholas Doiron Nikita Nangia Niklas Deckers Niklas Muennighoff Nitish Shirish Keskar,Niveditha Iyer Noah Constant Noah Fiedel Nuan Wen Oliver ZhangOmar Agha Omar Elbaghdadi Omer Levy Owain Evans Pablo Antonio Moreno Casares Parth Doshi Pascale Fung Paul Pu Liang Paul Vicol Pegah Alipoormolabashi Peiyuan Liao Percy Liang Peter W.ChangPeter EckersleyPhu Mon HtutPi-Bei HwangP.Milkowski Piyush S.Patil Pouya Pezeshkpour Priti Oli Qiaozhu Mei QING LYU Qinlang Chen Rabin Banjade Rachel Etta RudolphRaefer Gabriel,Rahel Habacker Ram'on Risco DelgadoRaphael Milliere,Rhythm Garg Richard Barnes Rif A.Saurous,Riku Arakawa Robbe Raymaekers Robert Frank Rohan Sikand Roman Novak Roman Sitelew Ronan Le Bras Rosanne Liu Rowan Jacobs Rui Zhang Ruslan Salakhutdinov Ryan Chi Ryan Lee Ryan Stovall Ryan Teehan Ryan Yang Sahib J. Singh Saif M. Mohammad Sajant Anand Sam DillavouSam Shleifer Sam Wiseman Samuel GruetterSam BowmanSamuel S.Schoenholz Sanghyun Han Sanjeev Kwatra Sarah A.Rous Sarik Ghazarian Sayan Ghosh Sean Casey Sebastian Bischoff Sebastian Gehrmann Sebastian Schuster Sepideh Sadeghi Shadi S.Hamdan Sharon ZhouShashank Srivastava Sherry Shi Shikhar Singh Shima Asaadi Shixiang Shane Gu Shubh Pachchigar Shubham Toshniwal Shyam Upadhyay,Shyamolima DebnathSiamak Shakeri Simon Thormeyer Simone Melzi Siva ReddySneha Priscilla Makini Soo hwan Lee Spencer Bradley Torene Sriharsha Hatwar Stanislas Dehaene Stefan Divic Stefano Ermon Stella Rose Biderman Stephanie C. Lin Stephen Prasad Steven T. Piantadosi Stuart M. Shieber Summer Misherghi Svetlana Kiritchenko Swaroop Mishra Tal Linzen Tal Schuster Tao Li Tao Yu Tariq A. Ali Tatsuo Hashimoto Te-Lin Wu Theo Desbordes Theodore Rothschild Thomas Phan Tianle WangTiberius Nkinyili Timo Schick T.N.Kornev Timothy Telleen-Lawton Titus Tunduny Tobias Gerstenberg Trenton ChangTrishala Neeraj Tushar Khot Tyler O.ShultzUri Shaham Vedant Misra Vera DembergVictoria Nyamai,Vikas Raunak Vinay Venkatesh Ramasesh Vinay Uday Prabhu Vishakh Padmakumar Vivek Srikumar William Fedus William Saunders William ZhangW VossenXiang RenXiaoyu F Tong Xinyi Wu Xudong Shen Yadollah Yaghoobzadeh Yair Lakretz Yang Song,Yasaman BahriYe Ji ChoiYichi YangYiding HaoYifu ChenYonatan BelinkovYu Hou Yu HouYushi BaiZachary SeidZhao XinranZhuoye ZhaoZi Fu WangZijie J.WangZirui Wang Ziyi Wu Sahib Singh and Uri Shaham Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. ArXiv, abs/2206.04615 2022.

Gabriel Stanovsky, Noah A. Smith, and Luke Zettlemoyer. Evaluating gender bias in machine translation. ArXiv, abs/1906.00591, 2019.

Haitian Sun, William W. Cohen, and Ruslan Salakhutdinov. Conditionalqa: A complex reading comprehension dataset with conditional answers. In ACL, 2022.

Alex Tamkin, Miles Brundage, Jack Clark, and Deep Ganguli. Understanding the capabilities, limitations, and societal impact of large language models. arXiv preprint arXiv:2102.02503, 2021a.

Alex Tamkin, Vincent Liu, Rongfei Lu, Daniel Fein, Colin Schultz, and Noah Goodman. Dabs: A domain-agnostic benchmark for self-supervised learning. arXiv preprint arXiv:2111.12062, 2021b.  
Alex Tamkin, Gaurab Banerjee, Mohamed Owda, Vincent Liu, Shashank Rammoorthy, and Noah Goodman. Dabs 2.0: Improved datasets and algorithms for universal self-supervision. 2022a.  
Alex Tamkin, Dat Nguyen, Salil Deshpande, Jesse Mu, and Noah D. Goodman. Active learning helps pretrained models learn the intended task. ArXiv, abs/2204.08491, 2022b.  
Yu Wang and Eugene Agichtein. Query ambiguity revisited: Clickthrough measures for distinguishing informational and ambiguous queries. In NAACL, 2010.  
Albert Webson and Ellie Pavlick. Do prompt-based models really understand the meaning of their prompts? ArXiv, abs/2109.01247, 2022.  
Kellie Webster, Marta Recasens, Vera Axelrod, and Jason Baldridge. Mind the gap: A balanced corpus of gendered ambiguous pronouns. Transactions of the Association for Computational Linguistics, 6:605-617, 2018.  
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. ArXiv, abs/2109.01652, 2022.  
Zeqiu Wu, Ryu Parish, Hao Cheng, Sewon Min, Prithviraj Ammanabrolu, Mari Ostendorf, and Han-naneh Hajishirzi. Inscit: Information-seeking conversations with mixed-initiative interactions. arXiv preprint arXiv:2207.00746, 2022.  
Michael J.Q. Zhang and Eunsol Choi. Situatedqa: Incorporating extra-linguistic contexts into qa. ArXiv, abs/2109.06157, 2021.  
Tony Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. ArXiv, abs/2102.09690, 2021.  
Ruiqi Zhong, Kristy Lee, Zheng Zhang, and Dan Klein. Adapting language models for zero-shot learning by meta-tuning on dataset and prompt collections. In EMNLP, 2021.

Table 2: Number of parameters for each model  

<table><tr><td>Model</td><td>Number of Parameters</td></tr><tr><td>OpenAI Ada (normal and RHLF models)</td><td>350M</td></tr><tr><td>OpenAI Babbage (normal and RHLF models)</td><td>1.3B</td></tr><tr><td>OpenAI Curie (normal and RHLF models)</td><td>6.7B</td></tr><tr><td>OpenAI Davinci (normal and RHLF models)</td><td>175B</td></tr><tr><td>AI21 J1-Jumbo</td><td>178B</td></tr><tr><td>T0++</td><td>11B</td></tr></table>
