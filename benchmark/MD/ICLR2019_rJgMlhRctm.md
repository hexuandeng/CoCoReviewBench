# THE NEURO-SYMBOLIC CONCEPT LEARNER: INTERPRETING SCENES, WORDS, AND SENTENCES FROM NATURAL SUPERVISION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We propose the Neuro-Symbolic Concept Learner (NS-CL), a model that learns visual concepts, words, and semantic parsing of sentences without explicit supervision on any of them; instead, our model learns by simply looking at images and reading paired questions and answers. Our model builds an object-based scene representation and translates sentences into executable, symbolic programs. To bridge the learning of two modules, we use a neuro-symbolic reasoning module that executes these programs on the latent scene representation. Analog to the human concept learning, given the parsed program, the perception module learns visual concepts based on the language description of the object being referred to. Meanwhile, the learned visual concepts facilitate learning new words and parsing new sentences. We use curriculum learning to guide searching over the large compositional space of images and language. Extensive experiments demonstrate the accuracy and efficiency of our model on learning visual concepts, word representations, and semantic parsing of sentences. Further, our method allows easy generalization to new object attributes, compositions, language concepts, scenes and questions, and even new program domains. It also empowers applications including visual question answering and bidirectional image-text retrieval.

# 1 INTRODUCTION

Humans are capable of learning visual concepts by jointly understanding vision and language (Fazly et al., 2010; Chrupała et al., 2015; Gauthier et al., 2018). Take the example shown in Figure 1(I). Imagine that a human with no prior knowledge of colors, is presented with the images of the red and green cubes paired with the questions and answers. She can easily identify the differences in certain component of the visual appearance (e.g. color in this case) of the objects and align the component to the corresponding word in the question and answer (Red and Green). Other single-object level attributes (e.g. shape) can also be learned in a similar fashion. Starting from there, a human is also able to inductively learn the correspondence between visual concepts and word semantics (e.g. spatial relations and referential expressions) Figure 1(II), and unravel compositional logic from complex questions assisted by the learned visual concepts Figure 1(III) (Abend et al., 2017).

Motivated by the way humans jointly perceive visual concepts and understand languages through a joint reasoning process (Gauthier et al., 2018), we propose a neuro-symbolic concept learner (NS-CL) that jointly learns visual perception, words, and semantic language parsing based on a visual question answering (VQA) setup. We employ a visually-grounded semantic parser for translating questions into executable programs, and a neural-based perception module that extracts object-level representations from the scene. A symbolic program executor then reads out the perceptual representation of objects, classifies their attributes/relations and executes the program to obtain an answer.

Our model learns from natural supervision signals (i.e. VQA pairs) through a curriculum-based approach. It starts from learning representations of individual objects from short questions (e.g., What's the color of the cylinder?) on simple scenes ( $\leq 3$  objects). It then learns relational concepts by leveraging object-level concepts to interpret object referrals (e.g., Is there a box right of a cylinder?). Finally, the model iteratively adapts to more complex scenes and compositional questions.

Our model recovers an object-based representation for visual scenes. Meanwhile, based on the learned visual representations, we propose a visually-grounded semantic parsing approach to resolve

I. Learning basic, object-based concepts.

II. Learning relational concepts based on referential expressions.

![](images/937692a0d46211a80fe9160d656169daf6b2a6ed88ff6f4db8dc079b3b86b222.jpg)

Q: What's the color of the object?  
A:Red.  
O: Is there any cube?  
A:Yes.  
Q: What's the color of the object?  
A:Green.  
Q: Is there any cube?  
A:Yes.

![](images/d5cafd7fb6be8d8898669240792f4886561755c19825b0b4ef6415cec9015514.jpg)

Q: How many objects are right of the red object?

A:2.  
Q: How many objects have the same material as the cube?  
A:2

III. Interpret complex questions from visual cues.

![](images/028fb295e567a107a7a9c68f8bbf9149d2ffcd8b3e1a094141e223c128309890.jpg)

Q: How many objects are both right of the green cylinder  
and have the same material as the small blue ball?  
A:3

Figure 1: Humans learn visual concepts, words, and semantic parsing jointly and incrementally. I. Learning visual concepts (red vs. green) starts from looking at simple scenes, reading simple questions, and reasoning over contrastive examples (Fazly et al., 2010). II. Afterwards, we can interpret referential expressions based on the learned object-based concepts, and learn relational concepts (e.g., on the right of, the same material as). III Finally, we can interpret complex questions from visual cues by exploiting the compositional structure.

the correspondence between word semantics and visual concepts. It also learns the semantic parsing of sentences requiring zero program annotations.

These disentangled and structural representations enable an interpretable and robust reasoning for VQA. Beyond showing a state-of-the-art performance on the CLEVR VQA dataset, our neuro-symbolic approach naturally supports combinatorial generalization w.r.t. the complexity of the scene and the programs (e.g., the depth of the program tree). We also propose solutions to the visual compositional generalization (CLEVR-CoGenT (Johnson et al., 2017a)) and the incremental learning of concepts. The learned visual concepts can be easily applied into other domains such as image-caption retrieval by only changing the program specification.

# 2 RELATED WORK

Our model is related to research on joint learning from visual data and natural language. In particular, there are many papers that learn visual concepts from descriptive visually-grounded languages, such as image-c captioning or visually-grounded question-answer pairs (Kiros et al., 2014; Mao et al., 2016; Vendrov et al., 2016; Ganju et al., 2017), dense language descriptions for scenes (Johnson et al., 2016), video-c captioning (Donahue et al., 2015) and video-text alignment (Zhu et al., 2015).

Visual question answering (VQA) stands out as it requires understanding both visual content and language. The state-of-the-art approaches usually use neural attentions (Malinowski & Fritz, 2014; Chen et al., 2015; Yang et al., 2016; Xu & Saenko, 2016). Beyond question answering, Johnson et al. (2017a) proposed the CLEVR (VQA) dataset to diagnose reasoning models. CLEVR contains synthetic visual scenes and questions generated from latent programs. Table 1 compares our model with state-of-the-art visual reasoning models (Andreas et al., 2016; Suarez et al., 2018; Santoro et al., 2017) along four directions: visual features, semantics, inference, and the requirement of extra labels.

For visual representation, Johnson et al. (2017b) encoded visual scenes into a convolutional feature map for program operators. Mascharka et al. (2018); Hudson & Manning (2018) used attention as intermediate representations for transparent program execution. Recently, Yi et al. (2018) explored an interpretable, object-based visual representation for visual reasoning. It performs well, but requires fully-annotated scenes during training. Our model also adopts an object-based visual representation, but the representation is learned only based on natural supervision (questions and answers).

There are two types of approaches in semantic sentence parsing for visual reasoning: implicit programs as conditioned neural operations (e.g., conditioned convolution and dual attention) (Perez et al., 2017; Hudson & Manning, 2018) and explicit programs as sequences of symbolic tokens (Johnson et al., 2017b; Mascharka et al., 2018). Explicit programs gain better interpretability, but usually require extra supervision such as ground-truth program annotations for training. This restricts their application. We propose a visually-grounded semantic parsing approach to parse questions in natural languages into explicit programs with zero program annotations. Given the semantic parsing of questions into programs, Yi et al. (2018) proposes a purely symbolic executor for the inference of the answer in the logic space. Compared with theirs, we propose an quasi-symbolic executor for VQA.

Our work is also related to learning interpretable and disentangled representations for visual scenes using neural networks. Kulkarni et al. proposed convolutional inverse graphics networks for learning and inferring pose of faces, while Yang et al. (2015) learned disentangled representation of pose of chairs from images. Wu et al. (2017) proposes the neural scene de-rendering framework as an inverse process of any rendering process. Siddharth et al. (2017); Higgins et al. (2018) learned disentangled

<table><tr><td rowspan="2">Models</td><td rowspan="2">Visual Features</td><td rowspan="2">Semantics</td><td colspan="2">Extra Labels</td><td rowspan="2">Inference</td></tr><tr><td>#Prog.</td><td>Attr.</td></tr><tr><td>FiLM (Perez et al., 2017)</td><td>Convolutional</td><td>Implicit</td><td>0</td><td>No</td><td>Feature Manipulation</td></tr><tr><td>IEP (Johnson et al., 2017b)</td><td>Convolutional</td><td>Explicit</td><td>700K</td><td>No</td><td>Feature Manipulation</td></tr><tr><td>MAC (Hudson &amp; Manning, 2018)</td><td>Attentional</td><td>Implicit</td><td>0</td><td>No</td><td>Feature Manipulation</td></tr><tr><td>Stack-NMN (Hu et al., 2018)</td><td>Attentional</td><td>Implicit</td><td>0</td><td>No</td><td>Attention Manipulation</td></tr><tr><td>TbD (Mascharka et al., 2018)</td><td>Attentional</td><td>Explicit</td><td>700K</td><td>No</td><td>Attention Manipulation</td></tr><tr><td>NS-VQA (Yi et al., 2018)</td><td>Object-Based</td><td>Explicit</td><td>0.2K</td><td>Yes</td><td>Symbolic Execution</td></tr><tr><td>NS-CL</td><td>Object-Based</td><td>Explicit</td><td>0</td><td>No</td><td>Symbolic Execution</td></tr></table>

Table 1: Comparison with other frameworks on the CLEVR VQA dataset, w.r.t. visual features, implicit or explicit semantics and supervisions.

![](images/83c21dab253ae8e29953be513079be4a565a2626100a0d06fae7ce88c38683ec.jpg)  
Figure 2: We propose to use neural symbolic reasoning as a bridge to jointly learn visual concepts, words, and semantic parsing of sentences.

representations using deep generative models. In contrast, we propose an alternative interpretable and disentangled representations learning approach through joint reasoning with language.

# 3 NEURO-SYMBOLIC CONCEPT LEARNER

We present our neuro-symbolic concept learner, which uses a symbolic reasoning process to learn visual concepts, words, and semantic parsing of sentences without explicit annotations for any of them. We first use a visual perception module to construct an object-based representation for a scene, and run a semantic parsing module to translate a question into an executable program. We then apply a quasi-symbolic program executor to infer the answer based on the scene representation. We use paired images, questions, and answers to jointly train the visual and language modules.

Shown in Figure 2, given an input image, the visual perception module detects objects in the scene and extracts a deep, latent representation for each of them. The semantic parsing module translates an input question in natural language into an executable program in a domain specific language (DSL). The generated programs have a hierarchical structure of symbolic, functional modules, each fulfilling a specific operation over the scene representation. The explicit program semantics enjoys compositionality, interpretability, and generalizability.

The program executor executes the program upon the derived scene representation and answers the question. Our program executor works in a symbolic and deterministic manner. This feature ensures a transparent execution trace of the program. Our program executor has a fully differentiable design w.r.t. the visual representations and concept embeddings, which can be updated using standard back-propagation during training.

# 3.1 MODEL DETAILS

Visual perception. Shown in Figure 2, given the input image, we use a Mask R-CNN pipeline (He et al., 2017) to generate object proposals for all objects. The bounding box for each single object paired with the original image is then sent to a ResNet-34 (He et al., 2015) to extract the region-based (by RoI Align) and image-based features respectively. We concatenate them to represent each object. Here, the inclusion of the representation of the full scene adds the contextual information, which is essential for the inference of relative attributes such as size or spatial position.

Concept quantization. Visual reasoning requires determining an object's attributes (e.g., its color or shape). We assume each visual attribute (e.g., shape) contains a set of possible visual concepts (e.g., Cube). In NS-CL, visual attributes are implemented as neural operators (Nagarajan & Grauman, 2018), mapping the object representation into an attribute-specific embedding space. Figure 3 shows

![](images/91988b7d81d3f6c10e0cf57340dde2843002712c693ff99173ead55aa32912ef.jpg)  
Figure 3: We treat attributes such as Shape and Color as neural operators. The operators map object representations into a visual-semantic space. We use similarity-based metric to classify objects.

# A. Curriculum concept learning

![](images/3a0315b21093204220ffed0d118237465431e3471a83fabdd139fedc4b7db05a.jpg)  
Figure 4: A. Demonstration of the curriculum learning of visual concepts, words, and semantic parsing of sentences by watching images and reading paired questions and answers. Scenes and questions of different complexities are illustrated to the learner in an incremental manner. B. Illustration of our neuro-symbolic inference model for VQA. The perception module begins with parsing visual scenes into object-based deep representations, while the semantic parser parse sentences into executable programs. A symbolic execution process bridges two modules.

an inference of the shape of an object. Visual concepts that belong to the shape attribute, including Cube, Sphere and Cylinder, are represented as vectors in this shared space. These concept vectors are also learned along the process. We measure the cosine distances  $\langle \cdot ,\cdot \rangle$  between these vectors to decide the shape of the object. Specifically, we compute the probability that an object  $o_i$  is a cube by  $\sigma (\langle \mathrm{ShapeOf}(o_i),v_{\mathrm{Cube}}\rangle -\gamma)\big{/}\tau$ , where  $\mathrm{ShapeOf}(\cdot)$  denotes the neural operator,  $v_{\mathrm{Cube}}$  the concept embedding of Cube and  $\sigma$  the Sigmoid function.  $\gamma$  and  $\tau$  are scalar constants for scaling and shifting the values of similarities. We classify relational concepts (e.g., Left) between a pair of objects similarly, except that we concatenate the visual representations for both objects to form a representation of the pair.

DSL and semantic parsing. The semantic parsing module translates a natural language question into an executable program with a hierarchy of primitive operations, represented in a domain-specific language (DSL) designed for VQA. The DSL covers a set of fundamental operations for visual reasoning, such as filtering out objects with certain concepts or query the attribute of an object. The operations share the same input and output interface, and thus can be compositionally combined to form programs of any complexity. We include a complete specification of the DSL used by our framework in the Appendix A.

Our semantic parser generates the hierarchies of latent programs in a sequence to sequence manner (Sutskever et al., 2014). We use an bidirectional GRU (Cho et al., 2014) to encode an input question, which outputs a fixed-length embedding of the question. A decoder based on GRU cells is applied to the embedding, and recovers the hierarchy of operations as the latent program.

Quasi-symbolic program execution. Given the latent program recovered from the question in natural language, a symbolic program executor executes the program and derives the answer based on the object-based visual representation. Our program executor is a collection of deterministic functional modules designed to realize all logic operations specified in the DSL. Figure 4(B) shows an illustrative execution trace of a program.

To make the execution differentiable w.r.t. visual representations, we represent the intermediate results in a probabilistic manner. Specifically, a set of objects is represented by a real-valued mask over all objects in the scene. Each element,  $\mathsf{Mask}_i\in [0,1]$  represents the probability that the  $i$ -th object of the scene belongs to the set. For example, shown in Figure 4(B), the first Filter operation outputs a mask of length 4 (there are in total 4 objects in the scene), with each element representing the probability that the corresponding object is selected out (i.e., the probability that each object is a green cube). The output "mask" on the objects will be fed into the next module (Relate in this case) as input and the execution of programs continues. The last module outputs the final answer to the question. We refer interested readers to Appendix C for the technical implementation of all VQA operators.

# 3.2 TRAINING PARADIGM

Optimization objective. The optimization objective of NS-CL is composed of two parts: concept learning and language understanding. Our goal is to find the optimal parameters  $\Theta_v$  of the visual perception module and  $\Theta_s$  of the semantic parsing module, to maximize the likelihood of answering the question  $Q$  correctly:

$$
\Theta_ {v}, \Theta_ {s} \leftarrow \arg \max  _ {\Theta_ {v}, \Theta_ {s}} \mathbb {E} _ {P} [ \Pr [ A = \text {E x e c u t o r (P e r c e p t i o n} (S; \Theta_ {v}), P) ] ], \tag {1}
$$

where  $P$  denotes the program,  $A$  the answer, and  $S$  the scene. The expectation is taken over  $P \sim \mathrm{SemanticParse}(Q; \Theta_s)$ .

Recall the program executor is fully differentiable w.r.t. the visual representation. We compute the gradient w.r.t.  $\Theta_v$  as  $\nabla_{\Theta_v}\mathbb{E}_P[D_{\mathrm{KL}}(\text{Executor}(\text{Perceptron}(S;\Theta_v),P)\|A)]$ . We use REINFORCE (Williams, 1992) to optimize the semantic parser  $\Theta_s$  via  $\nabla_{\Theta_s} = \mathbb{E}_P[r\cdot \log \Pr[P|\text{SemanticParse}(Q;\Theta_s)]]$ , where the reward  $r = 1$  if the answer is correct and 0 otherwise. We also use off-policy search to reduce the variance of REINFORCE, the detail of which can be found at Appendix B.

Curriculum visual concept learning. Motivated by human concept learning as in Figure 1, we employ a curriculum learning approach to help joint optimization. We heuristically split the training samples into four stages (Figure 4(A)): first, learning object-level visual concepts; second, learning relational questions; third, learning more complex questions with perception modules fixed; fourth, joint fine-tuning of all modules. Empirical experiments show that this is essential to the learning of our neuro-symbolic concept learner. We include more technical details in Appendix D.

# 4 EXPERIMENTS

We demonstrate the following advantages of our NS-CL. First, it learns visual concepts with remarkable accuracy; second, it allows data-efficient visual reasoning on the CLEVR dataset (Johnson et al., 2017a); third, it generalizes well to new attributes, visual composition, and language domains.

We train NS-CL on 5K images ( $< 10\%$  of CLEVR's 70K training images). We generate 20 questions for each image for the entire curriculum learning process.

# 4.1 VISUAL CONCEPT LEARNING

Classification-based concept evaluation. Our model treats attributes as neural operators that maps latent object representations into an attribute-specific embedding space (Figure 3). We evaluate the concept quantization of objects in the CLEVR validation split. Table 2 shows that our model achieves near perfect classification accuracy ( $\sim 99\%$ ) for 8 out of 9 attributes, suggesting it effectively learns generic concept representations. The result for spatial relations is relatively lower. This is because CLEVR does not have direct queries on the spatial relation between objects, so spatial relation concepts can only be learned indirectly.

Count-based concept evaluation. The SOTA methods do not provide interpretable representation on individual objects (Johnson et al., 2017a; Hudson & Manning, 2018; Mascharka et al., 2018). To evaluate the visual concepts learned by such models, we generate a synthetic question set. The

<table><tr><td></td><td>Color</td><td>Material</td><td>Shape</td><td>Size</td><td>Spatial Relation</td><td>Same Color</td><td>Same Material</td><td>Same Shape</td><td>Same Size</td></tr><tr><td>NS-CL</td><td>99.4</td><td>99.7</td><td>98.7</td><td>99.9</td><td>93.9</td><td>99.7</td><td>99.3</td><td>99.0</td><td>99.9</td></tr></table>

Table 2: We evaluate the learned visual concepts by reading out the concept classification for each object in the CLEVR validation split (e.g., classify whether each object is red). The table shows the average classification accuracy for each category of concepts. NS-CL learns visual concepts effectively and efficiently by only reading visually-grounded QA pairs.  

<table><tr><td></td><td>Visual</td><td>Overall</td><td>Color</td><td>Material</td><td>Shape</td><td>Size</td></tr><tr><td>IEP</td><td>Convolutional</td><td>90.6</td><td>91.0</td><td>90.0</td><td>89.9</td><td>90.6</td></tr><tr><td>MAC</td><td>Attentional</td><td>95.9</td><td>98.0</td><td>91.4</td><td>94.4</td><td>94.2</td></tr><tr><td>TbD (hres.)</td><td>Attentional</td><td>96.5</td><td>96.6</td><td>92.2</td><td>95.4</td><td>92.6</td></tr><tr><td>NS-CL</td><td>Object-Based</td><td>98.7</td><td>99.0</td><td>98.7</td><td>98.1</td><td>99.1</td></tr></table>

Table 3: We also evaluate the learned visual concepts using a diagnostic question set containing simple questions as "How many red objects are there?". NS-CL outperforms both convolutional and attentional baselines. The suggested object-based visual representation and symbolic reasoning approach perceives better interpretation of visual concepts.  

<table><tr><td>Model</td><td>Prog. Anno.</td><td>Overall</td><td>Count</td><td>Compare Numbers</td><td>Exist</td><td>Query Attribute</td><td>Compare Attribute</td></tr><tr><td>Human</td><td>N/A</td><td>92.6</td><td>86.7</td><td>86.4</td><td>96.6</td><td>95.0</td><td>96.0</td></tr><tr><td>NMN</td><td>700K</td><td>72.1</td><td>52.5</td><td>72.7</td><td>79.3</td><td>79.0</td><td>78.0</td></tr><tr><td>N2NMN</td><td>700K</td><td>88.8</td><td>68.5</td><td>84.9</td><td>85.7</td><td>90.0</td><td>88.8</td></tr><tr><td>IEP</td><td>700K</td><td>96.9</td><td>92.7</td><td>98.7</td><td>97.1</td><td>98.1</td><td>98.9</td></tr><tr><td>DDRprog</td><td>700K</td><td>98.3</td><td>96.5</td><td>98.4</td><td>98.8</td><td>99.1</td><td>99.0</td></tr><tr><td>TbD</td><td>700K</td><td>99.1</td><td>97.6</td><td>99.4</td><td>99.2</td><td>99.5</td><td>99.6</td></tr><tr><td>RN</td><td>0</td><td>95.5</td><td>90.1</td><td>93.6</td><td>97.8</td><td>97.1</td><td>97.9</td></tr><tr><td>FiLM</td><td>0</td><td>97.6</td><td>94.5</td><td>93.8</td><td>99.2</td><td>99.2</td><td>99.0</td></tr><tr><td>MAC</td><td>0</td><td>98.9</td><td>97.2</td><td>99.4</td><td>99.5</td><td>99.3</td><td>99.5</td></tr><tr><td>NS-CL</td><td>0</td><td>98.9</td><td>98.2</td><td>99.0</td><td>98.8</td><td>99.3</td><td>99.1</td></tr></table>

Table 4: Our model outperforms all baselines using no program annotations. It also achieves comparable results with baselines trained by full program annotations such as TbD (Mascharka et al., 2018), using less than  $10\%$  of the training images and  $15\%$  of the training questions.

diagnostic question set contains simple questions as the following form: "How many red objects are there?". We evaluate the performance on all concepts appeared in the CLEVR dataset.

Table 3 summarizes the results compared with strong baselines, including methods based on convolutional features (Johnson et al., 2017b) and those based on neural attentions (Mascharka et al., 2018; Hudson & Manning, 2018). Our approach outperforms IEP by a significant margin (8%) and attentional baselines by  $>2\%$ , suggesting object-based visual representations and symbolic reasoning helps interpreting visual concepts.

# 4.2 DATA-EFFICIENT AND INTERPRETABLE VISUAL REASONING

NS-CL jointly learns visual concepts, words and semantic parsing by watching images and reading paired questions and answers. It can be directly applied to VQA.

Table 4 summarizes results on the CLEVR validation split. Our model achieves the state-of-the-art performance among all baselines using zero program annotations, including MAC (Hudson & Manning, 2018) and FiLM (Perez et al., 2017). Our model achieves comparable performance with the strong baseline TbD-Nets (Mascharka et al., 2018), whose semantic parser is trained using 700K programs in CLEVR (ours need 0). The recent NS-VQA model from Yi et al. (2018) achieves better performance on CLEVR; however, their system requires annotated visual attributes and program traces during training, while our NS-CL needs no extra labels.

Split A  
![](images/1af857a5b59881d6de1ac5cb70de0c4570e182a3984246862124856b50975ff8.jpg)  
Q: What's the shape of the big yellow thing?

Split B  
![](images/ebc49d5d1a8921161f705b49b7dfff564b2755adac3d539e78e3d28ae9d2f99e.jpg)  
Q: What size is the cylinder that is left of the cyan thing that is in front of the big sphere?

Split C  
Figure 5: Samples collected from four splits in Section 4.3 for illustration. Models are trained on split A but evaluated on all splits for testing the combinatorial generalization.  
![](images/f29375905379ae60726f96246928d2308b61f8158327cdcd7bcd102e5ab3243c.jpg)  
Q: What's the shape of the big yellow thing?

Split D  
![](images/60ca738ed568d2f07fdc3617ad4b6465e84bdaa64dc52390402082bae67ba307.jpg)  
Q: What size is the cylinder that is left of the cyan thing that is in front of the gray cube?

Our model also recovers the underlying programs of questions accurately ( $>99.9\%$  accuracy). Our visual perception module is pre-trained on ImageNet (Deng et al., 2009). Without pre-training, the concept learning accuracies drop by  $0.2\%$  on average and the QA accuracy drops by  $0.5\%$ . Our model can also detect ambiguous or invalid programs and indicate exceptions. Please see Appendix E for more details.

# 4.3 GENERALIZATION TO NEW ATTRIBUTES AND COMPOSITIONS

Generalize to new visual compositions. The CLEVR-CoGenT dataset is designed to evaluate models' ability to generalize to new visual compositions. It has two splits: Split A only contains gray, blue, born and yellow cubes, but red, green, purple, and cyan cylinders; split B imposes the opposite color constraints on cubes and cylinders. If we directly learn visual concepts on split A, it overfits to classify shapes based on the color, leading to a poor generalization to split B.

Our solution is based on the idea of seeing attributes as operators. Specifically, we jointly train the concept embeddings (e.g., Red, Cube, etc.) as well as the semantic parser on split A, keeping pretrained, frozen attribute operators. As we learn distinct representation spaces for different attributes, our model achieves an accuracy of  $98.8\%$  on split A and  $98.9\%$  on split B.

Generalize to new visual concepts. We expect the process of concept learning can takes place in an incremental manner: having learned 7 different colors, humans can learn the 8-th color incrementally and efficiently. To this end, we build a synthetic split of the CLEVR dataset to replicate the setting of incremental concept learning. Split A contains only images without any purple objects, while split B contains images with at least one purple object. We train all the models on split A first, and finetune them on 100 images from split B. We report the final QA performance on split B's validation set. All models use a pre-trained semantic parser on the full CLEVR dataset.

Our model performs a  $93.9\%$  accuracy on the QA test in Split B, outperforming the convolutional baseline IEP (Johnson et al., 2017b) and the attentional baseline TbD (Mascharka et al., 2018) by  $4.6\%$  and  $6.1\%$  respectively. The acquisition of Color operator brings more efficient learning of new visual concepts.

# 4.4 COMBINATORIAL GENERALIZATION TO NEW SCENES AND QUESTIONS

Having learned visual concepts on small-scale scenes (containing only few objects) and simple questions (only single-hop questions), we humans can easily generalize the knowledge to larger-scale scenes and to answer complex questions. To evaluate this, we split the CLEVR dataset into four parts: Split A contains only scenes with less than 6 objects, and questions whose latent programs having a depth less than 4; Split B contains scenes with less than 6 objects, but arbitrary questions; Split C contains arbitrary scenes, but restricts the program depth being less than 4; Split D contains arbitrary scenes and questions. Figure 5 shows some illustrative samples.

As VQA baselines are unable to count a set of objects of arbitrary size, for a fair comparison, all programs containing the "count" operation over  $>6$  objects are removed from the set. For methods using explicit program semantics, the semantic parser are pre-trained on the full dataset and fixed. Methods with implicit program semantics (Hudson & Manning, 2018) learn an entangled representation for perception and reasoning, and cannot trivially generalize to more complex programs. We only use the training data from the Split A and then quantify the generalization ability on other three splits. Shown in Table 5, our NS-CL leads to almost-perfect generalization to larger scenes and more complex questions, outperforming all baselines by at least  $4\%$  in QA accuracy.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Program</td><td rowspan="2">Visual Representation</td><td rowspan="2">Train</td><td colspan="4">Test</td></tr><tr><td>Split A</td><td>Split B</td><td>Split C</td><td>Split D</td></tr><tr><td>MAC</td><td>Implicit</td><td>Attentional</td><td>Split A</td><td>97.3</td><td>N/A</td><td>92.9</td><td>N/A</td></tr><tr><td>IEP</td><td>Explicit</td><td>Convolutional</td><td>Split A</td><td>96.1</td><td>92.1</td><td>91.5</td><td>90.9</td></tr><tr><td>TbD(hres.)</td><td>Explicit</td><td>Attentional</td><td>Split A</td><td>98.8</td><td>94.5</td><td>94.3</td><td>91.9</td></tr><tr><td>NS-CL</td><td>Explicit</td><td>Object-Based</td><td>Split A</td><td>98.9</td><td>98.9</td><td>98.7</td><td>98.8</td></tr></table>

Caption: There is a big yellow cylinder in front of a gray object.  
![](images/3356bf4582206dbccd38c5a741ca66183ee298dce5af200348998735d2acf74a.jpg)  
(a) An illustrative pair of image and caption in our synthetic dataset.

Table 5: We test the combinatorial generalization w.r.t. the number of objects in scenes and the complexity of questions (i.e. the depth of the program trees). We makes four split of the data containing various complexities of scenes and questions. Our object-based visual representation and explicit program semantics enjoys the best (and almost-perfect) combinatorial generalization compared with strong baselines.  

<table><tr><td>Model</td><td>Retrieval Accuracy</td></tr><tr><td>IEP</td><td>95.5</td></tr><tr><td>TbD</td><td>97.0</td></tr><tr><td>NS-CL</td><td>96.9</td></tr></table>

(b) Image-caption retrieval accuracy on a subset of data. Our model archives comparable results with VQA baselines.

<table><tr><td>Model</td><td>Retrieval Accuracy</td></tr><tr><td>CNN-LSTM</td><td>68.9</td></tr><tr><td>NS-CL</td><td>97.0</td></tr></table>

(c) Image-caption retrieval accuracy on the full dataset. Our model outperforms baselines and requires no extra training or finetuning of the visual perception module.

Table 6: To validate the transferrability of the learned visual concepts, we introduce a new simple DSL for image-caption retrieval. Due to the difference between VQA and caption retrieval, VQA baselines are only able to infer the result on a partial set of data. The learned object-based visual concepts can be directly transferred into the new domain for free.

# 4.5 EXTENDING TO OTHER PROGRAM DOMAIN

The learned visual concepts can also be used in other domains such as image retrieval. With the visual scenes fixed, the learned visual concepts can be directly transferred into the new domain. We only need to learn the semantic parsing of natural language into the new DSL.

We build a synthetic dataset for image retrieval and adopt a DSL from scene graph-based image retrieval (Johnson et al., 2015). The dataset contains only simple captions: "There is an  $<$ object A> $<$ relation> $<$ object B>." (e.g., There is a box right of a cylinder). The semantic parser learns to extract corresponding visual concepts (e.g., box, right, and cylinder) from the sentence. The program can then be executed on the visual representation to determine if the visual scene contains such relational triples.

For simplicity, we treat retrieval as classifying whether a relational triple exists in the image. This functionality cannot be directly implemented on the CLEVR VQA program domain, because questions such as "Is there a box right of a cylinder" can be ambiguous if there exists multiple cylinders in the scene. Due to the entanglement of the visual representation with the specific DSL, baselines trained on CLEVR QA can not be directly applied to this task. For a fair comparison with them, we show the result in Table 6b on a subset of the generated image-caption pairs where the underlying programs have no ambiguity regarding the reference of object B. A separate semantic parser is trained for the VQA baselines, which translates captions into a CLEVR QA-compatible program (e.g., Exist(Filter(Box, Relate(Right, Filter(Cylinder))).

Table 6c compares our NS-CL against typical image-text retrieval baselines on the full image-caption dataset. Without any annotations of the sentence semantics, our model learns to parse the captions into the programs in the new DSL. It outperforms the CNN-LSTM baseline by  $30\%$ .

# 5 CONCLUSION

We presented a method that jointly learns visual concepts, words, and semantic parsing of sentences from natural supervision. The proposed framework, NS-CL, learns by looking at images and reading paired questions and answers, without any explicit supervision such as class labels for objects. Our model learns visual concepts with remarkable accuracy. Based upon the learned concepts, our model achieves good results on question answering, and more importantly, generalizes well to new visual compositions, new visual concepts, and new domain specific languages.

# REFERENCES

Omri Abend, Tom Kwiatkowski, Nathaniel J Smith, Sharon Goldwater, and Mark Steedman. Bootstrapping language acquisition. Cognition, 164:116-143, 2017.  
Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Learning to compose neural networks for question answering. In *NAACL-HLT*, 2016.  
Kan Chen, Jiang Wang, Liang-Chieh Chen, Haoyuan Gao, Wei Xu, and Ram Nevatia. Abc-cnn: An attention based convolutional neural network for visual question answering. arXiv preprint arXiv:1511.05960, 2015.  
Kyunghyun Cho, Bart Van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnN encoder-decoder for statistical machine translation. In EMNLP, 2014.  
Grzegorz Chrupał, Akos Kádár, and Afra Alishahi. Learning language through pictures. arXiv preprint arXiv:1506.03694, 2015.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009.  
Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In CVPR, 2015.  
Afsaneh Fazly, Afra Alishahi, and Suzanne Stevenson. A probabilistic computational model of cross-situational word learning. Cognitive Science, 34(6):1017-1063, 2010.  
Siddha Ganju, Olga Russakovsky, and Abhinav Gupta. What's in a question: Using visual questions as a form of supervision. In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pp. 6422-6431. IEEE, 2017.  
Jon Gauthier, Roger Levy, and Joshua B Tenenbaum. Word learning and the acquisition of syntactic-semantic overhypotheses. arXiv preprint arXiv:1805.04988, 2018.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2015.  
Kaiming He, Georgia Gkioxari, Piotr Dólár, and Ross Girshick. Mask r-cnn. In ICCV, 2017.  
Irina Higgins, Nicolas Sonnerat, Loic Matthey, Arka Pal, Christopher P Burgess, Matthew Botvinick, Demis Hassabis, and Alexander Lerchner. Scan: learning abstract hierarchical compositional visual concepts. In ICLR, 2018.  
Ronghang Hu, Jacob Andreas, Trevor Darrell, and Kate Saenko. Explainable neural computation via stack neural module networks. In Proceedings of the European Conference on Computer Vision (ECCV), 2018.  
Drew A Hudson and Christopher D Manning. Compositional attention networks for machine reasoning. In ICLR, 2018.  
Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. Image retrieval using scene graphs. In CVPR, 2015.  
Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.  
Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017a.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In ICCV, 2017b.  
Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-semantic embeddings with multimodal neural language models. arXiv preprint arXiv:1411.2539, 2014.  
Tejas D Kulkarni, William F Whitney, Pushmeet Kohli, and Josh Tenenbaum. Deep convolutional inverse graphics network. In NIPS, 2015.  
M Malinowski and M Fritz. A multi-world approach to question answering about real-world scenes based on uncertain input. In NIPS, 2014.  
Junhua Mao, Jiajing Xu, Kevin Jing, and Alan L Yuille. Training and Evaluating Multimodal Word Embeddings with Large-Scale Web Annotated Images. In Proc. of NIPS, 2016.  
David Mascharka, Philip Tran, Ryan Soklaski, and Arjun Majumdar. Transparency by design: Closing the gap between performance and interpretability in visual reasoning. In CVPR, 2018.  
Tushar Nagarajan and Kristen Grauman. Attributes as operators: Factorizing unseen attribute-object compositions. In The European Conference on Computer Vision (ECCV), September 2018.  
Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. arXiv preprint arXiv:1709.07871, 2017.  
Adam Santoro, David Raposo, David GT Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, and Timothy Lillicrap. A simple neural network module for relational reasoning. In NIPS, 2017.  
N Siddharth, T. B. Paige, J.W. Meent, A. Desmaison, N. Goodman, P. Kohli, F. Wood, and P. Torr. Learning disentangled representations with semi-supervised deep generative models. In NIPS, 2017.  
Joseph Suarez, Justin Johnson, and Fei-Fei Li. Ddrprog: A clever differentiable dynamic reasoning programmer. arXiv:1803.11361, 2018.  
Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In NIPS, 2014.  
Richard S Sutton, David A McAllester, Satinder P Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In NIPS, 2000.  
Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. Order-embeddings of images and language. In Proc. of ICLR, 2016.  
Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. MLJ, 8(3-4):229-256, 1992.  
Jiajun Wu, Joshua B Tenenbaum, and Pushmeet Kohli. Neural scene de-rendering. In CVPR, 2017.  
Huijuan Xu and Kate Saenko. Ask, attend and answer: Exploring question-guided spatial attention for visual question answering. In European Conference on Computer Vision, pp. 451-466. Springer, 2016.  
Jimei Yang, Scott E Reed, Ming-Hsuan Yang, and Honglak Lee. Weakly-supervised disentangling with recurrent transformations for 3d view synthesis. In NIPS, 2015.  
Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng, and Alex Smola. Stacked attention networks for image question answering. In CVPR, 2016.  
Kexin Yi, Jiajun Wu, Chuang Gan, Antonio Torralba, Kohli Pushmeet, and Joshua B Tenenbaum. Neural-symbolic vqa: Disentangling reasoning from vision and language understanding. In Advances in neural information processing systems, 2018.  
Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp. 19-27, 2015.
