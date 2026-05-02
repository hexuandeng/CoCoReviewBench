# NEURO-SYMBOLIC FORWARD REASONING

Anonymous authors

Paper under double-blind review

# ABSTRACT

Reasoning is an essential part of human intelligence and thus has been a longstanding goal in artificial intelligence research. With the recent success of deep learning, incorporating reasoning with deep learning systems, i.e., neuro-symbolic AI has become a major field of interest. We propose the Neuro-Symbolic Forward Reasoner (NSFR), a new approach for reasoning tasks taking advantage of differentiable forward-chaining using first-order logic. The key idea is to combine differentiable forward-chaining reasoning with object-centric (deep) learning. Differentiable forward-chaining reasoning computes logical entailments smoothly, i.e., it deduces new facts from given facts and rules in a differentiable manner. The object-centric learning approach factorizes raw inputs into representations in terms of objects. Thus, it allows us to provide a consistent framework to perform the forward-chaining inference from raw inputs. NSFR factorizes the raw inputs into the object-centric representations, converts them into probabilistic ground atoms, and finally performs differentiable forward-chaining inference using weighted rules for inference. Our comprehensive experimental evaluations on object-centric reasoning data sets, 2D Kandinsky patterns and 3D CLEVR-Hans, and a variety of tasks show the effectiveness and advantage of our approach.

# 1 INTRODUCTION

Right from the time of Aristotle, reasoning has been in the center of the study of human behavior (Miller, 1984). Reasoning can be defined as the process of deriving conclusions and predictions from available data. The long-lasting goal of artificial intelligence has been to develop rational agents akin to humans, and reasoning is considered to be a major part of achieving rationality (Johnson-Laird, 2010). Logic, both propositional and first-order, is an established framework to perform reasoning on machines (Boole, 1847). Such logical reasoning has been an essential part of the growth of machine learning over the years (Poole et al., 1987; Bottou, 2014; Dai et al., 2019) and has also given rise to statistical relational learning (Koller et al., 2007; Raedt et al., 2016) and probabilistic logic programming (Lukasiewicz, 1998; De Raedt & Kersting, 2003; De Raedt & Kimmig, 2015).

Object-centric reasoning has been widely addressed (Johnson et al., 2017; Mao et al., 2019; Chen et al., 2021; Han et al., 2019), where the task is to perform reasoning to answer the questions that are about the objects and its attributes. However, the task is challenging because the models should perform low-level visual perception and reasoning on high-level concepts. To mitigate this challenge, with the recent success of deep learning, incorporating logical reasoning with deep learning systems, i.e., neuro-symbolic AI has become a major field of interest (De Raedt et al., 2019; Garcez et al., 2019). It has the advantage of combining the expressivity of neural networks with the reasoning of symbolic methods.

Various benchmarks and methods have been developed for object-centric reasoning (Locatello et al., 2020b; Nanbo et al., 2020). Recently, data sets such as Kandinsky patterns (Mueller & Holzinger, 2019; Holzinger et al., 2019; 2021) and CLEVR (Johnson et al., 2017) have been proposed to assess the performance of the machine learning systems in object-centric reasoning tasks. For example, Figure 1 shows an example of the Kandinsky pattern: "the figure has two pairs of objects with the same shape" where Fig. (a) is following the pattern and Fig. (b) is not. Kandinsky patterns are inspired by human IQ-tests (Brunner et al., 1956; Dowe & Hernandez-Orallo, 2012; Liu et al., 2019), which require humans to think on abstract patterns. The key feature of Kandinsky Patterns is its complexity, e.g., the arrangement of objects, closure or symmetry, and a group of objects.

Many approaches have been investigated for object-centric reasoning under the umbrella of neuro-symbolic learning (Rocktäschel & Riedel, 2017; Yang et al., 2017; Šourek et al., 2018; Manhaeve et al., 2018; Si et al., 2019; Mao et al., 2019; Cohen et al., 2020; Riegel et al., 2020). However, using these approaches it is difficult, if not impossible, to solve object-centric reasoning tasks such as Kandinsky patterns due to several underlying challenges: (i) the perception of the objects from the raw inputs and (ii) the reasoning on the attributes and the relations to capture the complex patterns (of varying size).

In this work, we propose the Neuro-Symbolic Forward Reasoner (NSFR), a novel neuro-symbolic learning framework for complex

object-centric reasoning tasks. The key idea is to combine neural-based object-centric learning models with the differentiable implementation of first-order logic. It has three main components: (i) object-centric perception module, (ii) facts converter, and (iii) differentiable reasoning module. The object-centric perception module extracts information for each object and has been widely addressed in the computer vision community (Locatello et al., 2020a; Redmon et al., 2016). Facts converter converts the output of the visual perception module into the form of probabilistic logical atoms, which can be fed into the reasoning module. Finally, differentiable reasoning module performs the differentiable forward-chaining inference from a given input. It computes the set of ground atoms that can be deduced from the given set of ground atoms and weighted logical rules (Evans & Grefenstette, 2018; Shindo et al., 2021). The final prediction can be made based on the result of the forward-chaining inference.

Overall, we make the following contributions: (1) We propose Neuro-Symbolic Forward Reasoner (NSFR), a new neuro-symbolic learning framework that performs differentiable forward-chaining inference from visual data using object-centric models. NSFR can solve problems involving complex patterns on objects and attributes, such as the arrangement of objects, closure, or symmetry. (2) To establish NSFR, we show an extended implementation of the differentiable forward-chaining inference to overcome the scalability problem. Moreover, NSFR can take advantage of some essential features of the underlying neural network, such as batch computation, for logical reasoning. (3) To establish NSFR, we provide a conversion algorithm from object-centric representations to probabilistic facts. We propose neural predicates, which are associated with a function to produce a probability of a fact and yield a seamless combination of sub-symbolic and symbolic representations. (4) We empirically show that NSFR solves object-centric reasoning tasks more effectively than the SOTA logical and deep learning models. Furthermore, NSFR classifies complex patterns with high accuracy for 2D and 3D data sets, outperforming pure neural-based approaches for image recognition.

# 2 BACKGROUND AND RELATED WORK

Notation. We use bold lowercase letters  $\mathbf{v}$ ,  $\mathbf{w}$ , ... for vectors and the functions that return vectors. We use bold capital letters  $\mathbf{X}$ , ... for tensors. We use calibrate letters  $\mathcal{C}$ ,  $\mathcal{A}$ , ... for (ordered) sets and typewriter font  $\mathrm{p}(\mathrm{X},\mathrm{Y})$  for terms and predicates in logical expressions (Appendix A.1 for details).

Preliminaries. We consider function-free first-order logic. Language  $\mathcal{L}$  is a tuple  $(\mathcal{P},\mathcal{T},\mathcal{V})$ , where  $\mathcal{P}$  is a set of predicates,  $\mathcal{T}$  is a set of constants, and  $\mathcal{V}$  is a set of variables. A term is a constant or a variable. We assume that each term has a datatype. A datatype dt specifies a set of constants  $dom(\mathrm{dt}) = \mathcal{T}_{\mathrm{dt}} \subseteq \mathcal{T}$ . We denote  $n$ -ary predicate  $\mathfrak{p}$  by  $\mathfrak{p} / (n,[\mathrm{dt}_1,\dots,\mathrm{dt}_n])$ , where  $\mathrm{dt}_i$  is the datatype of  $i$ -th argument. An atom is a formula  $\mathfrak{p}(\mathfrak{t}_1,\ldots ,\mathfrak{t}_n)$ , where  $\mathfrak{p}$  is an  $n$ -ary predicate symbol and  $\mathfrak{t}_1,\ldots ,\mathfrak{t}_n$  are terms. A ground atom or simply a fact is an atom with no variables. A literal is an atom or its negation. A positive literal is just an atom. A negative literal is the negation of an atom. A clause is a finite disjunction ( $\vee$ ) of literals. A definite clause is a clause with exactly one positive literal. If  $A,B_{1},\ldots ,B_{n}$  are atoms, then  $A \lor \neg B_{1} \lor \dots \lor \neg B_{n}$  is a definite clause. We write definite clauses in the form of  $A: -B_{1},\ldots ,B_{n}$ . Atom  $A$  is called the head, and set of negative atoms  $\{B_1,\dots,B_n\}$  is called the body. We denote special constant true as  $\top$  and false as  $\bot$ . Substitution  $\theta = \{\mathtt{X}_1 = \mathtt{t}_1,\dots,\mathtt{X}_n = \mathtt{t}_n\}$  is an assignment of term  $\mathfrak{t}_{\mathrm{i}}$  to variable  $\mathtt{X}_{\mathrm{i}}$ . An application of substitution  $\theta$  to atom  $A$  is written as  $A\theta$ .

![](images/f736fa342f533d9173a3ff8ba479404e6616769ac8c348ad347860995305f5b2.jpg)  
(a)  
Figure 1: Examples of Kandinsky patterns. (a) follows the pattern: "the figure has 2 pairs of objects with the same shape", but (b) isn't

![](images/35150e6e07358e37c2ca850a3e18a9ce3a164dd411728eb6243cd4b9acd2632f.jpg)  
(b)

Related Work. Reasoning with neuro-symbolic systems has been studied extensively for various applications such as ocean study (Corchado, 1995), business internal control (Corchado et al., 2004) and forecasting (Fdez-Riverola et al., 2002). More recently, several neuro-symbolic techniques for commonsense reasoning (Arabshahi et al., 2021), visual question answering (Mao et al., 2019; Amizadeh et al., 2020) and multimedia tasks (Khan & Curry, 2020) have been developed. They either do not employ a differentiable forward reasoner or miss object-centric learning in the end-to-end reasoning architecture.

Object-centric learning is an approach to decompose an input image into representations in terms of objects (Dittadi et al., 2021). This problem has been widely addressed in the computer vision community. The typical approach is the object detection (or supervised) approach such as Faster-RCNN (Ren et al., 2015) and YOLO (Redmon et al., 2016). Another approach is the unsupervised approach (Burgess et al., 2019; Engelcke et al., 2020; Locatello et al., 2020a), where the models acquire the ability of object-perception without or fewer annotations. These two different paradigms have different advantages. NSFR encapsulates different object-perception models, thus allows us to choose a proper model depending on the situation and the problem to be solved.

Also, the integration of symbolic logic and neural networks has been addressed, see e.g. DeepProblog (Manhaeve et al., 2018) and NeurASP (Yang et al., 2020). The key difference from the past approaches is that NSFR supports essential features of neural networks such as batch computation and that it is fully differentiable. Thus it scales well to large data sets, leading to several avenues for future work learning neural networks with logical constraints (Hu et al., 2016; Xu et al., 2018).

# 3 THE NEURO-SYMBOLIC FORWARD REASONER (NSFR)

Let us now introduce the Neuro-Symbolic Forward Reasoner (NSFR) in four steps. First, we give an overview of the problem setting and the framework. Second, we specify a language of first-order logic focusing on the object-centric reasoning tasks. Third, we explain the facts converter. Finally, we show the differentiable forward-chaining inference algorithm, an extended implementation from the previous approaches.

# 3.1 OVERVIEW

Problem Scenario. We address the image classification problem, where each image contains objects, and the classification rules are defined on the relations of objects and their attributes. We define the object-centric reasoning problem as follows:

Definition 1 An Object-Centric Reasoning Problem  $\mathcal{Q}$  is a tuple  $(\mathcal{I}^{+},\mathcal{I}^{-},P)$ , where  $\mathcal{I}^{+}$  is a set of images that follow pattern  $P$ ,  $\mathcal{I}^{-}$  is a set of images that do not follow pattern  $P$ . Each image  $X\in \mathcal{I}^{+}\cup \mathcal{I}^{-}$  contains several objects, and each object has its attributes. Pattern  $P$  is a pattern that is described as logical rules or natural language sentence, which is defined on the attributes and relations of objects. The solution of problem  $\mathcal{Q}$  is a set of binary labels  $\mathcal{V} = \{y_{i}\}_{i = 0,\dots ,N}$  for each image  $X_{i}\in \mathcal{I}^{+}\cup \mathcal{I}^{-}$ , where  $N = |\mathcal{I}^{+}\cup \mathcal{I}^{-}|$ .

Architecture Overview. NSFR performs object-centric perception from raw input and reasoning on the extracted high-level concepts. Figure 2 presents an overview of NSFR. First, NSFR perceives objects from raw input whose output is a set of vectors, called object-centric representations, where each vector represents each object in the input. Then, the fact converter takes these object-centric representations as input and returns a set of probabilistic facts. The probabilistic facts are then fed into the reasoning module, which performs differentiable forward-chaining inference using weighted rules. Finally, the final prediction is made on the result of the inference. We briefly summarize the steps of the process as follows:

Step 1: Let  $\mathbf{X} \in \mathbb{R}^{B \times N}$  be a batch of input images. Perception function  $f_{\text{percept}}: \mathbb{R}^{B \times N} \to \mathbb{R}^{B \times E \times D}$  factorizes input  $\mathbf{X}$  into a set of object-centric representations  $\mathbf{Z} \in \mathbb{R}^{B \times E \times D}$ , where  $E \in \mathbb{N}$  is the number of objects,  $D \in \mathbb{N}$  is the dimension of object-centric vector.

Step 2: Let  $\mathcal{G}$  be a set of ground atoms. Convert function  $f_{convert} : \mathbb{R}^{B \times E \times D} \times \mathcal{G} \to \mathbb{R}^{B \times G}$  generates a probabilistic vector representation of facts, where  $G = |\mathcal{G}|$ .

Step 3: Infer function  $f_{infer} : \mathbb{R}^{B \times G} \to \mathbb{R}^{B \times G}$  computes forward-chaining inference using weighted clauses  $\mathcal{C}$ .

![](images/9575ba910be33c63eee779d2af0fa29e9c3d9c4714fd6fcf79728d46d5ce8881.jpg)  
Figure 2: An overview of NSFR. The object-centric model produces outputs in terms of objects. The facts converter obtains probabilistic facts from the object-centric representation. The differentiable forward-chaining inference computes the logical entailment softly from the probabilistic facts and weighted rules. The final prediction is computed based on the entailed facts.

Step 4: Predict function  $f_{\text{predict}}: \mathbb{R}^{B \times G} \to \mathbb{R}^B$  computes the probability of target facts. The probability of the labels  $\mathbf{y}$  of the batch of input  $\mathbf{X}$  is computed as:

$$
p (\mathbf {y} | \mathbf {X}) = f _ {p r e d} \left(f _ {\text {i n f e r}} \left(f _ {\text {c o n v e r t}} \left(\mathbf {X}; \boldsymbol {\Phi}\right), \mathcal {G}; \boldsymbol {\Theta}\right); \mathcal {C}, \mathbf {W}\right)), \tag {1}
$$

where  $\Phi, \Theta$ , and  $\mathbf{W}$  are learnable parameters.

We now present each component of our architecture in detail.

# 3.2 OBJECT-CENTRIC REASONING LANGUAGE

We have to define a first-order logic language to build a consistent neuro-symbolic framework for object-centric reasoning. Intuitively, we assume that all constants are divided into inputs, objects, and attributes, and the attribute constants have different data types such as colors and shapes.

Definition 2 An Object-Centric Reasoning Language is a function-free language  $\mathcal{L} = (\mathcal{P},\mathcal{T},\mathcal{V})$  where  $\mathcal{P}$  is a set of predicates,  $\mathcal{T}$  is a set of constants, and  $\mathcal{V}$  is a set of variables. The set of constants  $\mathcal{T}$  is divided to a set of inputs  $\mathcal{X}$ , a set of objects  $\mathcal{O}$ , and a set of attributes  $\mathcal{A}$ , i.e.,  $\mathcal{T} = \mathcal{X} \cup \mathcal{O} \cup \mathcal{A}$ . The attribute constants  $\mathcal{A}$  is divided into a set of constants for each datatype, i.e.,  $\mathcal{A} = \mathcal{A}_{\mathrm{dt}_1} \cup \ldots \cup \mathcal{A}_{\mathrm{dt}_n}$ , where  $\mathcal{A}_{\mathrm{dt}_i}$  is a set of constants of the  $i$ -th datatype  $\mathrm{dt}_i$ .

Example 1: The language for Figure 2 can be represented as  $\mathcal{L}_1 = (\mathcal{P},\mathcal{T},\mathcal{V})$  where  $\mathcal{P} = \{\mathrm{kp} / (1,[\mathrm{image}]),\mathrm{in} / (2,[\mathrm{object},\mathrm{image}]),\mathrm{color} / (2,[\mathrm{object},\mathrm{color}])$  shape/(2, [object, shape]), same_shape_pair(2, [object, object])}, and  $\mathcal{T} = \mathcal{X}\cup \mathcal{O}\cup \mathcal{A}_{\mathrm{color}}\cup$ $\mathcal{A}_{\mathrm{shape}}$  where  $\mathcal{X} = \{\mathrm{img}\},\mathcal{O} = \{\mathrm{obj1, obj2, obj3, obj4}\},\mathcal{A}_{\mathrm{color}} = \{\mathrm{red,yellow,blue}\}$  where  $\mathcal{A}_{\mathrm{shape}} = \{\mathrm{square,cycle,triangle}\}$ , and  $\mathcal{V} = \{\mathrm{X,Y,Z,01,02,03,04}\}$ .

# 3.3 OBJECT-CENTRIC PERCEPTION

We make the minimum assumption that the perception function takes an image and returns a set of object-centric vectors, where each of the vectors represents each object. For simplicity, we assume that each dimension of the vector represents the probability of the attributes for each object. For example, suppose each object has color, shape, position as attributes. The color varies red, blue, yellow, the shape varies square, circle, triangle, and position is represented as a  $(x,y)$ -coordinates. In this case, each object can be represented as an 8-dim vector, as illustrated in Figure 2.

Let  $N$  be the input size,  $E$  be the maximum number of objects that can appear in one image, and  $D$  be the number of attributes for each object. For a batch of input images  $\mathbf{X} \in \mathbb{R}^{B \times N}$ , the object-centric perception function  $f_{percept}: \mathbb{R}^{B \times N} \to \mathbb{R}^{B \times E \times D}$  parameterized  $\Phi$  produces a batch of object-centric representations  $\mathbf{Z} \in \mathbb{R}^{B \times E \times D}: \mathbf{Z} = f_{percept}(\mathbf{X}; \Phi)$ . We note that each value  $\mathbf{Z}_{i,j,k}$  represents the probability of the  $k$ -th attribute on the  $j$ -th object in the  $i$ -th image in the batch. We denote the tensor for  $j$ -th object as  $\mathbf{Z}^{(j)} = \mathbf{Z}_{:,j,:} \in \mathbb{R}^{B \times D}$ .

![](images/f419020247bb005e58f21c0da6bc40e6a2a1ad9cca5d2e805f9af26d0704067f.jpg)  
Figure 3: An overview of the facts-converting process. NSFR decomposes the raw-input images into the object-centric representations (left). The valuation functions are called to compute the probability of ground atoms (middle). The result is converted into the form of vector representations of the probabilistic ground atoms (right).

# 3.4 FACTS CONVERTER

After the object-centric perception, NSFR obtains the logical representation, i.e., a set of probabilistic ground atoms. We propose a new type of predicate that can refer to differentiable functions to compute the probability and a seamless converting algorithm from the perception result to probabilistic ground atoms.

# 3.4.1 TENSOR REPRESENTATIONS OF CONSTANTS

Specifically, in NSFR, constants are mapped to tensors as described below.

Objects. We map the object constants to the object-centric representation from the visual-perception module. The output of the visual-perception module is already factorized in terms of objects. Therefore the tensor for each object is extracted easily by slicing the output.

Attributes. We map the attribute constants to their corresponding one-hot encoding and assume that it is expanded to the batch size. Let  $\mathcal{E}_{\mathcal{L}}$  be the set of one-hot encoding of attribute constants in language  $\mathcal{L}$ . For e.g., for language  $\mathcal{L}_1$  in Example 1, color red has tensor representation as  $\mathbf{A}_{\mathrm{red}} = [[1,0,0],[1,0,0]]\in \mathbb{R}^{2\times 3}$ , where the batch size is 2. We assume that we have the encoding for each attribute, i.e.,  $\mathcal{E}_{\mathcal{L}_1} = \{\mathbf{A}_{\mathrm{red}},\mathbf{A}_{\mathrm{yellow}},\mathbf{A}_{\mathrm{blue}},\mathbf{A}_{\mathrm{square}},\mathbf{A}_{\mathrm{circle}},\mathbf{A}_{\mathrm{triangle}}\}$ .

In summary, we define the tensor representations for each object and attribute constant  $\mathsf{t}$  as:

$$
f _ {t o - t e n s o r} (t; \mathbf {Z}, \mathcal {E} _ {\mathcal {L}}) = \left\{ \begin{array}{l l} \mathbf {Z} ^ {(i)} & \text {i f} t = \mathrm {o b j} _ {i} \in \mathcal {O} \\ \mathbf {A} _ {\mathrm {d t}} ^ {(i)} & \text {i f} t = \operatorname {a t t r} _ {i} \in \mathcal {A} _ {\mathrm {d t}} \end{array} , \right. \tag {2}
$$

where  $\mathbf{A}_{\mathrm{dt}}^{(i)} \in \mathcal{E}_{\mathcal{L}}$  is the one-hot encoding of the  $i$ -th attribute of datatype dt. For example,  $f_{to\_tensor}(\mathrm{obj1}; \mathbf{Z}, \mathcal{E}_{\mathcal{L}_1}) = \mathbf{Z}^{(1)}$  and  $f_{to\_tensor}(\mathrm{red}; \mathbf{Z}, \mathcal{E}_{\mathcal{L}_1}) = \mathbf{A}_{\mathrm{red}} = [[1,0,0],[1,0,0]]$ .

# 3.4.2 NEURAL Predicate

To solve the object-centric reasoning tasks, the model should capture the relation that is characterized by continuous features, for e.g., the close by relation between two objects. To encode such concepts into the form of logical facts, we introduce neural predicate that composes a ground atom associated with a differentiable function. Neural predicates compute the probability of the ground atoms using the object-centric representations from the visual perception module.

Definition 3 A neural predicate  $\mathfrak{p} / (n, [\mathsf{dt}_1, \ldots, \mathsf{dt}_n])$  is a  $n$ -ary predicate associated with a function  $v_{\mathfrak{p}}: \mathbb{R}^{d_1 \times \dots \times d_n} \to \mathbb{R}^B$ , where  $\mathsf{dt}_i$  is the datatype of the  $i$ -th argument, and  $d_i \in \mathbb{N}$  is the dimension of the tensor representation of the constant whose datatype is  $\mathsf{dt}_i$

Example 2: Figure 3 illustrates how the neural predicates and the valuation functions are computed. (1) For neural predicate color/(2, [object, color]), the probability of ground atom color(obj3, red) is computed by valuation function  $v_{\text{color}} : \mathbb{R}^{2 \times 5} \times \mathbb{R}^{2 \times 3} \rightarrow \mathbb{R}^2$  as

$v_{\mathrm{color}}(\mathbf{Z}^{(3)}, \mathbf{A}_{\mathrm{red}}) = \sum_{1} (\mathbf{Z}_{:,0:3}^{(3)} \odot \mathbf{A}_{\mathrm{red}})$ , where  $\mathbf{A}_{\mathrm{red}} \in \{0,1\}^{2 \times 3}$  is a one-hot encoding of the color of red that is expanded to the batch size,  $sum_{1}$  is the sum operation for the dimension 1, and  $\odot$  is the element-wise multiplication. (2) Likewise, for neural predicate shape/(2, [object, shape]), the probability of ground atom shape(obj1, circle) is computed by valuation function  $v_{\mathrm{shape}} : \mathbb{R}^{2 \times 5} \times \mathbb{R}^{2 \times 3} \to \mathbb{R}^2$  as  $v_{\mathrm{shape}}(\mathbf{Z}^{(1)}, \mathbf{A}_{\mathrm{circle}}) = \sum_{1} (\mathbf{Z}_{:,3:6}^{(1)} \odot \mathbf{A}_{\mathrm{circle}})$ . (3) For neural predicate closeby(2/[object, object]), the probability of ground atom closeby(obj1, obj2) is computed by valuation function  $v_{\mathrm{closeby}} : \mathbb{R}^{2 \times 5} \times \mathbb{R}^{2 \times 5} \to \mathbb{R}$  as:  $v_{\mathrm{closeby}}(\mathbf{Z}^{(1)}, \mathbf{Z}^{(2)}) = \sigma \left(norm_0 \left(\mathbf{Z}_{:,4:6}^{(1)} - \mathbf{Z}_{:,4:6}^{(2)}\right); \mathbf{w}\right)$ , where  $norm_0$  is the norm function along dimension 0,  $\sigma$  is the sigmoid function for each element of the input, and  $\mathbf{w}$  is the trainable parameter. By adapting the parameters in neural predicates, NSFR can learn the concepts determined by numerical attributes and their relations. We note that valuation functions of neural predicates can be replaced by other differentiable functions, e.g., multilayer perceptrons.

# 3.4.3 CONVERSION ALGORITHM TO VALUATION TENSORS

The facts converter produces a set of probabilistic ground atoms that are fed into the reasoning module. In NSFR, the probabilistic facts are represented in the form of tensors called valuation tensors.

Valuation. Valuation tensor  $\mathbf{V}^{(t)}\in \mathbb{R}^{B\times G}$  maps each ground atom into a continuous value at each time step  $t$ . Each value  $\mathbf{V}_{i,j}^{(t)}$  represents the probability of ground atom  $F_{j}\in \mathcal{G}$  for the  $i$ -th example in the batch. The output of the perception module  $\mathbf{Z}\in \mathbb{R}^{B\times E\times D}$  is compiled into initial valuation tensor  $\mathbf{V}^{(0)}$ . The differentiable inference function is performed based on valuation tensors. To compute the  $T$ -step forward-chaining inference, we compute the sequence of valuation tensors  $\mathbf{V}^{(0)},\ldots ,\mathbf{V}^{(T)}$ .

Conversion into Valuation Tensors. Neural predicates yield a seamless conversion algorithm from the object-centric vectors into the probabilistic facts. Algorithm 1 (see Appendix B) describes the converting procedure. For each ground atom, if it consists of a neural predicate, then the valuation function is called to compute the probability of the atom. We note that the valuation function computes probability in batch. NSFR allows background knowledge as a set of ground atoms. The probability of background knowledge is set to 1.0.

# 3.5 DIFFERENTIABLE FORWARD-CHAINING INFERENCE

NSFR performs reasoning based on the differentiable forward-chaining inference approach (Evans & Grefenstette, 2018; Shindo et al., 2021). The key idea is to implement the forward reasoning of first-order logic using tensors and operations between them using the following steps: (Step 1) Tensor  $\mathbf{I}$ , which is called index tensor, is built from given set of clauses  $\mathcal{C}$  and fixed set of ground atoms  $\mathcal{G}$ . It holds the relationships between clauses  $\mathcal{C}$  and ground atoms  $\mathcal{G}$ . Its dimension is proportional to  $|\mathcal{C}|$  and  $|\mathcal{G}|$ . (Step 2) A computational graph is constructed from  $\mathbf{I}$  and clause weights  $\mathbf{W}$ . The weights define probability distributions over clauses  $\mathcal{C}$ , approximating a logic program softly. The probabilistic forward-chaining inference is performed by the forwarding algorithm on the computational graph with input  $\mathbf{V}^{(0)}$ , which is the output of the fact converter. We now step-wise describe the process.

# 3.5.1 TENSORENCODING

We build a tensor that holds the relationships between clauses  $\mathcal{C}$  and ground atoms  $\mathcal{G}$ . We assume that  $\mathcal{C}$  and  $\mathcal{G}$  are an ordered set, i.e., where every element has its own index. Let  $L$  be the maximum body length in  $\mathcal{C}$ ,  $S$  be the maximum number of substitutions for existentially quantified variables in clauses  $\mathcal{C}$ ,  $C = |\mathcal{C}|$ , and  $G = |\mathcal{G}|$ . Index tensor  $\mathbf{I} \in \mathbb{N}^{C \times G \times S \times L}$  contains the indices of the ground atoms to compute forward inferences. Intuitively,  $\mathbf{I}_{i,j,k,l}$  is the index of the  $l$ -th ground atom (subgoal) in the body of the  $i$ -th clause to derive the  $j$ -th ground atom with the  $k$ -th substitution for existentially quantified variables.

Example 3: Let  $R_0 = \mathrm{kp}(\mathbf{X}) : -\mathrm{in}(01, \mathbf{X}), \mathrm{shape}(01, \mathbf{square}) \in \mathcal{C}$  and  $F_2 = \mathrm{kp}(\mathrm{img}) \in \mathcal{G}$ , and we assume that object constants are  $\{\mathrm{obj1}, \mathrm{obj2}\}$ . To deduce fact  $F_2$  using clause  $R_0$ ,

$F_{2}$  and the head atom can be unified by substitution  $\theta = \{\mathbf{X} = \mathbf{img}\}$ . By applying  $\theta$  to body atoms, we get clause  $\mathrm{kp}(\mathrm{img}): -\mathrm{in}(01,\mathrm{img})$ , shape(01,square), which has an existentially quantified variable 01. By considering the possible substitutions for 01, we have grounded clauses as  $\mathrm{kp}(\mathrm{img}): -\mathrm{in}(\mathrm{obj1},\mathrm{img})$ , shape(obj1,square),  $\mathrm{kp}(\mathrm{img}): -\mathrm{in}(\mathrm{obj2},\mathrm{img})$ , shape(obj2,square). Then the following table shows tensor  $\mathbf{I}_{0,:0,:}$  and  $\mathbf{I}_{0,:1,:}$ :

<table><tr><td>j
G</td><td>0
⊥</td><td>1
T</td><td>2
kp(img)</td><td>3
in(obj1, img)</td><td>4
in(obj2, img)</td><td>5
shape(obj1, square)</td><td>...</td></tr><tr><td>I0,j,0,:</td><td>[0,0]</td><td>[1,1]</td><td>[3,5]</td><td>[0,0]</td><td>[0,0]</td><td>[0,0]</td><td>...</td></tr><tr><td>I0,j,1,:</td><td>[0,0]</td><td>[1,1]</td><td>[4,6]</td><td>[0,0]</td><td>[0,0]</td><td>[0,0]</td><td>...</td></tr></table>

Ground atoms  $\mathcal{G}$  and the indices are represented on the upper rows in the table. For example,  $\mathbf{I}_{0,2,0,:} = [3,5]$  because  $R_0$  entails  $\mathrm{kp}(\mathrm{img})$  with substitution  $\theta = \{\emptyset 1 = \mathrm{obj1}\}$ . Then the subgoal atoms are  $\{\mathrm{in}(\mathrm{obj1},\mathrm{img1}),\mathrm{shape}(\mathrm{obj1},\mathrm{square})\}$ , which have indices [3,5], respectively. With another substitution  $\theta = \{\emptyset 1 = \mathrm{obj2}\}$ , the subgoal atoms are  $\{\mathrm{in}(\mathrm{obj2},\mathrm{img1}),\mathrm{shape}(\mathrm{obj1},\mathrm{square})\}$ , which have indices [4,6], respectively. The atoms which have a different predicate, e.g., shape(obj1, square), will never be entailed by clause  $R_0$ . Therefore, the corresponding values are filled with 0, which represents the index of the false atom.

# 3.5.2 DIFFERENTIABLE INFERENCE

Using the encoded index tensor, NSFR performs differentiable forward-chaining reasoning. We briefly summarize the steps as follows. (Step 1): Each clause is compiled into a function that performs forward reasoning. (Step 2): The weighted sum of the results from each clause is computed. (Step 3):  $T$ -time step inference is computed by amalgamating the inference results recursively. We extend previous approaches (Evans & Grefenstette, 2018; Shindo et al., 2021) for batch computation.

**Clause Function.** Each clause  $R_{i} \in \mathcal{C}$  is compiled in to a clause function. The clause function takes valuation tensor  $\mathbf{V}^{(t)}$ , and returns valuation tensor  $\mathbf{C}_i^{(t)}\mathbb{R}^{B\times G}$ , which is the result of 1-step forward reasoning using  $R_{i}$  and  $\mathbf{V}^{(t)}$ . The clause function is computed as follows. First, tensor  $\mathbf{I}_i \in \mathbb{R}^{G\times S\times L}$  is extended for batches, i.e.,  $\tilde{\mathbf{I}}_i \in \mathbb{N}^{B\times G\times S\times L}$ , and  $\mathbf{V} \in \mathbb{R}^{B\times G}$  is extended to the same shape, i.e.,  $\tilde{\mathbf{V}} \in \mathbb{R}^{B\times G\times S\times L}$ . Using these tensors, the clause function is computed as:

$$
\mathbf {C} _ {i} ^ {(t)} = \operatorname {s o f t o r} _ {3} ^ {\gamma} \left(\operatorname {p r o d} _ {2} \left(\operatorname {g a t h e r} _ {1} \left(\tilde {\mathbf {V}}, \tilde {\mathbf {I}}\right)\right), \right. \tag {3}
$$

where  $\text{gather}_1(\mathbf{X},\mathbf{Y})_{i,j,k,l} = \mathbf{X}_{i,\mathbf{Y}_{i,j,k,l},k,l}$ , and  $\text{prod}_2$  returns the product along dimension 2.  $\text{softor}_d^\gamma$  is a function for taking logical or softly along dimension  $d$ :

$$
\operatorname {s o f t o r} _ {d} ^ {\gamma} (\mathbf {X}) = \frac {1}{S} \gamma \log \left(\operatorname {s u m} _ {d} \exp \left(\mathbf {X} / \gamma\right)\right), \tag {4}
$$

where  $\gamma > 0$  is a smooth parameter,  $sum_{d}$  is the sum function for tensors along dimension  $d$ , and

$$
S = \left\{ \begin{array}{l l} 1. 0 & \text {i f} m a x (\gamma \log s u m _ {d} \exp (\mathbf {X} / \gamma)) \leq 1. 0 \\ m a x (\gamma \log s u m _ {d} \exp (\mathbf {X} / \gamma)) & \text {o t h e r w i s e} \end{array} . \right. \tag {5}
$$

Normalization term  $S$  ensures that the function returns the normalized probabilistic values. We refer appendix I for more details about the  $\text{softtor}_d^\gamma$  function. In Eq. 3, applying the  $\text{softtor}_3^\gamma$  function corresponds to considering all possible substitutions for existentially quantified variables in the body atoms of the clause and taking logical or softly over the results of possible substitutions. The results from each clause is stacked into tensor  $\mathbf{C}^{(t)} \in \mathbb{R}^{C \times B \times G}$ , i.e.,  $\mathbf{C}^{(t)} = \text{stack}_0(\mathbf{C}_1^{(t)}, \ldots, \mathbf{C}_C^{(t)})$ , where  $\text{stack}_0$  is a stack function for tensors along dimension 0.

Soft (Logic) Program Composition. In NSFR, a logic program is represented smoothly as a weighted sum of the clause functions following (Shindo et al., 2021). Intuitively, NSFR has  $M$  distinct weights for each clauses, i.e.,  $\mathbf{W} \in \mathbb{R}^{M \times C}$ . By taking softmax of  $\mathbf{W}$  along dimension 1,  $M$  clauses are softly chosen from  $C$  clauses. The weighted sum of clause functions are computed as follows. First, we take the softmax of the clause weights  $\mathbf{W} \in \mathbb{R}^{M \times C}$ :  $\mathbf{W}^* = \text{softmax}_1(\mathbf{W})$  where  $\text{softmax}_1$  is a softmax function over the dimension 1. The clause weights  $\mathbf{W}^* \in \mathbb{R}^{M \times C}$  and the output of the clause function  $\mathbf{C}^{(t)} \in \mathbb{R}^{C \times B \times G}$  are expanded to the same shape  $\tilde{\mathbf{W}}^*$ ,  $\tilde{\mathbf{C}}^{(t)} \in$

Table 1: The classification accuracy in each data set. NSFR outperforms the considered baselines. Neural networks over-fit while training and perform poorly with testing data. Best results are bold.  

<table><tr><td rowspan="2"></td><td colspan="3">Training Data</td><td colspan="3">Test Data</td></tr><tr><td>NSFR</td><td>ResNet50</td><td>YOLO+MLP</td><td>NSFR</td><td>ResNet50</td><td>YOLO+MLP</td></tr><tr><td>Twopairs</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.50</td><td>0.98</td></tr><tr><td>Threepairs</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.515</td><td>0.912</td></tr><tr><td>Closeby</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.54</td><td>0.91</td></tr><tr><td>Red-Triangle</td><td>0.958</td><td>1.0</td><td>1.0</td><td>0.956</td><td>0.57</td><td>0.79</td></tr><tr><td>Online/Pair</td><td>0.997</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.52</td><td>0.66</td></tr><tr><td>9-Circles</td><td>0.964</td><td>1.0</td><td>1.0</td><td>0.952</td><td>0.50</td><td>0.50</td></tr></table>

$\mathbb{R}^{M\times C\times B\times G}$ . Then we compute tensor  $\mathbf{H}^{(t)}\in \mathbb{R}^{M\times B\times G}$ :  $\mathbf{H}^{(t)} = sum_{1}(\tilde{\mathbf{W}}^{*}\odot \tilde{\mathbf{C}})$ , where  $\odot$  is element-wise multiplication, and  $sum_{1}$  is a summation along dimension 1. Each value  $\mathbf{H}_{i,j,k}^{(t)}$  represents the probability of  $k$ -th ground atom using  $i$ -th clause weights for the  $j$ -th example in the batch. Finally, we compute tensor  $\mathbf{R}^{(t)}\in \mathbb{R}^{B\times G}$  corresponding to the fact that logic program is a set of clauses:  $\mathbf{R}^{(t)} = \text{softtor}_0^\gamma (\mathbf{H})$ .

Multi-step Forward-Chaining Reasoning. We define the 1-step forward-chaining reasoning function as:  $r(\mathbf{V}^{(t)};\mathbf{I},\mathbf{W}) = \mathbf{R}^{(t)}$  and compute the  $T$ -step reasoning by:  $\mathbf{V}^{(t + 1)} =$  softtor $_1^\gamma (\text{stack}_1(\mathbf{V}^{(t)},r(\mathbf{V}^{(t)};\mathbf{I},\mathbf{W})))$ , where  $\mathbf{I}\in \mathbb{N}^{C\times G\times S\times L}$  is a precomputed index tensor, and  $\mathbf{W}\in \mathbb{R}^{M\times C}$  is clause weights.

# 4 EXPERIMENTAL EVALUATION

We empirically demonstrate the following desired properties of NSFR on 2 data sets (see App. C): (i) NSFR solves object-centric reasoning tasks with complex abstract patterns, (ii) NSFR can handle complex 3d scenes, and (iii) NSFR can perform fast reasoning with the batch computation.

# 4.1 SOLVING KANDINSKY PATTERNS

Data. We adopted Kandinsky pattern data sets (Mueller & Holzinger, 2019; Holzinger et al., 2019; 2021), a relatively new benchmark for object-centric reasoning tasks and use 6 Kandinsky patterns.

Model. We used YOLO (Redmon et al., 2016) as a perception module and trained it on the pattern-free figures, which are randomly generated. The correct rules are given to classify the figures, and clause weights are initialized to choose each of them. For e.g., the classification rule of the twopairs data set is: "the Kandinsky Figure has two pairs of objects with the same shape, in one pair the objects have the same colors in the other pair different colors, two pairs are always disjunct, i.e. they don't share objects". This can be represented as a logic program containing four clauses:

```prolog
1 kp(X):-in(O1,X),in(O2,X),in(O3,X),in(O4,X),same_shape_pair(O1,O2), same_color_pair(O1,O2),same_shape_pair(O3,O4),diff_color_pair(O3,O4).   
2 same_shape_pair(X,Y):-shape(X,Z),shape(Y,Z).   
3 same_color_pair(X,Y):-color(X,Z),color(Y,Z).   
4 diff_color_pair(X,Y):-color(X,Z),color(Y,W),diff_color(Z,W).
```

Pre-training. We generated 15k pattern-free figures for pre-training of the visual perception module. Each object has the class label and the bounding box as an annotation. We generated 5k concept examples for neural predicate closeby and online.

Baselines. We adopted ResNet (He et al., 2016) as a benchmark and also compare against YOLO+MLP, where the input figure is fed to the pre-trained YOLO model, and a simple MLP module predicts the class label from the YOLO outputs.

Results. Table 1 shows the results for each Kandinsky data set. The Resnet50 model overfits while training and thus performs poorly in every test data. The YOLO+MLP model performs comparatively better and achieves greater than  $90\%$  accuracy in twopairs, threepairs, and closeby data set. However, in relatively complex patterns of red-traignle, online/pair, and 9-cicle data sets the performance degrades. On the contrary, NSFR outperforms the considered baselines significantly and achieves perfect classification in 4 out of the 6 data sets.

Table 2: Classification accuracy for CLEVR-Hans data sets compared to baselines.  

<table><tr><td>Model</td><td>Validation</td><td>Test</td><td>Validation</td><td>Test</td></tr><tr><td colspan="3">CLEVR-Hans3</td><td colspan="2">CLEVR-Hans7</td></tr><tr><td>CNN</td><td>99.55</td><td>70.34</td><td>96.09</td><td>84.50</td></tr><tr><td>NeSy (Default)</td><td>98.55</td><td>81.71</td><td>96.88</td><td>90.97</td></tr><tr><td>NeSy-XIL</td><td>100.00</td><td>91.31</td><td>98.76</td><td>94.96</td></tr><tr><td>NS-FR</td><td>98.18</td><td>98.40</td><td>93.60</td><td>92.19</td></tr></table>

![](images/1946627d0f4107098a63c7d2774703867b1fb2f1684b6a4a5c30d05cd089f993.jpg)  
Figure 4: The inference time with different batch sizes.

# 4.2 REASONING ON THE 3D-WORLD: SOLVING CLEVR-HANS PROBLEMS

Data. The CLEVR-Hans data set (Stammer et al., 2021) contains confounded CLEVR (Johnson et al., 2017) images, and each image is associated with a class label. The CLEVR-Hans3 data set has three classes, and the CLEVR-Hans7 data set has seven classes.

Model. We adopted Slot Attention (Locatello et al., 2020a) as a visual perception module and used a set prediction architecture, where each slot representation is fed to MLPs to predict attributes.

Pre-training. The slot attention model was pre-trained following (Locatello et al., 2020a) using the set prediction setting on the CLEVR (Johnson et al., 2017) data set. In the concept learning process, we trained rightside, leftside, and front using the scene data in the CLEVR data set. We generated 10k positive and negative examples for each concept, respectively.

Baselines. The considered baselines are the ResNet34-based CNN model (Hu et al., 2016), and the Neuro-Symbolic model (NeSy) (Stammer et al., 2021). The NeSy model was trained in two different settings: (1) training using classification rules (NeSy-default), and (2) training using classification rules and example-based explanation labels (NeSy-XIL).

Results. Table 2 shows the classification accuracy in the CLEVR-Hans data sets. The results of baselines have been presented in (Stammer et al., 2021). In the CLEVR-Hans3 data set, NSFR achieved more than  $98\%$  in each split. In the CLEVR-Hans7 data set, NSFR achieved more than  $92\%$ , that is  $>\mathrm{NeSy}$ -Default. Note that, NeSy-XIL model exploits example-based labels about the explanation, whereas NeSy-Default and NSFR do not. Thus NeSy-XIL outperforms NSFR marginally. The empirical result shows that NSFR (i) handles different types of the perception models (YOLO and Slot Attention), (ii) can effectively handle 3D images, and more importantly, (iii) is robust to confounded data if the classification rules are available in the form of logic programs.

# 4.3 FAST INFERENCE BY BATCH COMPUTATION

We show that NSFR can perform fast inference by batch computation. Figure 4 shows the inference time with different batch sizes in Kandinsky data sets. We change the batch size from 1 to 50 by increments of 5 and run the experiment in each Kandinsky data set. The magenta line represents mean running time, and the shade represents the standard deviation over the data sets. The empirical result shows that NSFR can perform fast reasoning using batch computation, which is the essential nature of deep neural networks.

# 5 CONCLUSION AND FUTURE WORK

We proposed Neuro-Symbolic Forward Reasoner (NSFR), a novel framework for object-centric reasoning tasks. NSFR perceives raw input images using an object-centric model, converts the output into the probabilistic ground atoms, and performs the differentiable forward-chaining inference. Furthermore, NSFR supports batch computation. Thus it combines the perception module and the reasoning module seamlessly. In our experiments, NSFR outperformed conventional CNN-based models in 2D Kandinsky patterns and 3D CLEVR-Hans data sets, where the classification rules are defined on the high-level concepts. There are several avenues for future work. If we set the clause weights as trainable parameters, NSFR can perform structure learning of logic programs from visual inputs, which is a promising way of extending Inductive Logic Programming and differentiable approaches. Likewise, if we set the parameters of the perception model as trainable parameters, NSFR can train perception models with logical constraints.

# ETHICS STATEMENT

With our work, we have shown that we can seamlessly combine symbolic and sub-symbolic systems. Combining neural models with symbolic models can lead to better generalization and handle a wider variety of problems. The major impact that our work aims is enabling coherent quantitative inquiries that encompass multiple data dimension types across object-centric reasoning tasks. This can have several implications on studying how scientific fields evolve and can produce validated signatures predictive of the emergence and success of new fields or discoveries. The results can also be leveraged to create metrics and methods to estimate the innovation potential of scientific enterprises. To the best of our knowledge, our study does not raise any ethical, privacy or conflict of interest concerns.

# REPRODUCIBILITY STATEMENT

Upon acceptance, an official GitHub repository will be made public, containing the code of NSFR, and scripts to reproduce the experiments and generate data sets. In addition to this, architectural details and hyper-parameters are included in the appendix. Preliminary code will be uploaded upon submission. Lastly, details on the evaluation metrics and relevant data sets, including the relevant symbolic rules, are given in the main text as well as the appendix.

# REFERENCES

Saeed Amizadeh, Hamid Palangi, Oleksandr Polozov, Yichen Huang, and Kazuhito Koishida. Neuro-symbolic visual reasoning: Disentangling "visual" from "reasoning". In ICML. 2020.  
Forough Arabshahi, Jennifer Lee, Mikayla Gawarecki, Kathryn Mazaitis, Amos Azaria, and Tom Mitchell. Conversational neuro-symbolic commonsense reasoning. AAAI, 2021.  
George Boole. The Mathematical Analysis of Logic: Being an Essay Towards a Calculus of Deductive Reasoning. Cambridge Library Collection - Mathematics. Cambridge University Press, 1847.  
Léon Bottou. From machine learning to machine reasoning. Machine learning, 2014.  
J. S. Bruner, J. J. Goodnow, and G. A. Austin. A Study of Thinking. Wiley, 1956.  
Christopher P Burgess, Loic Matthew, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick, and Alexander Lerchner. Monet: Unsupervised scene decomposition and representation. arXiv preprint arXiv:1901.11390, 2019.  
Zhenfang Chen, Jiayuan Mao, Jiajun Wu, Kwan-Yee Kenneth Wong, Joshua B. Tenenbaum, and Chuang Gan. Grounding physical concepts of objects and events through dynamic visual reasoning. In ICLR, 2021.  
William W. Cohen, Fan Yang, and Kathryn Mazaitis. Tensorlog: A probabilistic database implemented using deep-learning infrastructure. JAIR, 2020.  
JM Corchado. Neuro-symbolic reasoning-a solution for complex problems. In International Conference on Intelligent Systems, 1995.  
Juan M Corchado, M Lourdes Borrajo, María A Pellicer, and J Carlos Yáñez. Neuro-symbolic system for business internal control. In ICDM, 2004.  
Wang-Zhou Dai, Qiuling Xu, Yang Yu, and Zhi-Hua Zhou. Bridging machine learning and logical reasoning by abductive learning. 2019.  
Luc De Raedt and Kristian Kersting. Probabilistic logic learning. ACM SIGKDD Explorations Newsletter, 2003.  
Luc De Raedt and Angelika Kimmig. Probabilistic (logic) programming concepts. Machine Learning, 2015.

Luc De Raedt, Robin Manhaeve, Sebastijan Dumancic, Thomas Demeester, and Angelika Kimmig. Neuro-symbolic= neural+ logical+ probabilistic. In NeSy'19@ IJCAI, the 14th International Workshop on Neural-Symbolic Learning and Reasoning, 2019.  
Andrea Dittadi, Samuele Papa, Michele De Vita, Bernhard Schölkopf, Ole Winther, and Francesco Locatello. Generalization and robustness implications in object-centric learning. arXiv preprint arXiv:2107.00637, 2021.  
David L Dowe and José Hernández-Orallo. Iq tests are not for machines, yet. Intelligence, 2012.  
Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, and Ingmar Posner. Genesis: Generative scene inference and sampling with object-centric latent representations. In *ICLR*, 2020.  
Richard Evans and Edward Grefenstette. Learning explanatory rules from noisy data. JAIR, 2018.  
Florentino Fdez-Riverola, Juan M Corchado, and Jesús M Torres. Neuro-symbolic system for forecasting red tides. In Irish Conference on Artificial Intelligence and Cognitive Science, 2002.  
Artur d'Avila Garcez, Marco Gori, Luis C Lamb, Luciano Serafini, Michael Spranger, and Son N Tran. Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. arXiv preprint arXiv:1905.06088, 2019.  
Chi Han, Jiayuan Mao, Chuang Gan, Joshua B. Tenenbaum, and Jiajun Wu. Visual concept-metaconcept learning. NeurIPS, 2019.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.  
Andreas Holzinger, Michael Kickmeier-Rust, and Heimo Müller. Kandinsky patterns as iq-test for machine learning. In Andreas Holzinger, Peter Kieseberg, A Min Tjoa, and Edgar Weippl (eds.), Machine Learning and Knowledge Extraction, 2019.  
Andreas Holzinger, Anna Saranti, and Heimo Mueller. Kandinskypatterns-an experimental exploration environment for pattern analysis and machine intelligence. arXiv preprint arXiv:2103.00519, 2021.  
Zhiting Hu, Xuezhe Ma, Zhengzhong Liu, Eduard Hovy, and Eric Xing. Harnessing deep neural networks with logic rules. In ACL, 2016.  
Zhengyao Jiang and Shan Luo. Neural logic reinforcement learning. In ICML 2019, 2019.  
Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In CVPR, 2017.  
Philip N Johnson-Laird. Mental models and human reasoning. Proceedings of the National Academy of Sciences, 2010.  
Muhammad Jaleed Khan and Edward Curry. Neuro-symbolic visual reasoning for multimedia event processing: Overview, prospects and challenges. In CIKM (Workshops), 2020.  
Daphne Koller, Nir Friedman, Sašo Džeroski, Charles Sutton, Andrew McCallum, Avi Pfeffer, Pieter Abbeel, Ming-Fai Wong, Chris Meek, Jennifer Neville, et al. Introduction to statistical relational learning. MIT press, 2007.  
Yusen Liu, Fangyuan He, Haodi Zhang, Guozheng Rao, Zhiyong Feng, and Yi Zhou. How well do machines perform on iq tests: a comparison study on a large-scale dataset. In IJCAI, 2019.  
Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In NeurIPS, 2020a.  
Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. 2020b.

Thomas Lukasiewicz. Probabilistic logic programming. In ECAI, 1998.  
Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. Deepproblog: Neural probabilistic logic programming. In NeurIPS 2018, 2018.  
Jiayuan Mao, Chuang Gan, Pushmeet Kohli, Joshua B. Tenenbaum, and Jiajun Wu. The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision. In ICLR 2019, 2019.  
Fred D Miller. Aristotle on rationality in action. The Review of Metaphysics, 1984.  
Heimo Mueller and Andreas Holzinger. Kandinsky patterns, 2019.  
Li Nanbo, Cian Eastwood, and Robert B Fisher. Learning object-centric representations of multi-object scenes from multiple views. In NeurIPS, 2020.  
David Poole, Randy Goebel, and Romas Aleliunas. Theorist: A logical reasoning system for defaults and diagnosis. In *The Knowledge Frontier*. 1987.  
Luc De Raedt, Kristian Kersting, Siraam Natarajan, and David Poole. Statistical relational artificial intelligence: Logic, probability, and computation. Synthesis Lectures on Artificial Intelligence and Machine Learning, 2016.  
Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, June 2016.  
Shaoqing Ren, Kaiming He, Ross B. Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In NIPS, 2015.  
Ryan Riegel, Alexander Gray, Francois Luus, Naweed Khan, Ndivhuwo Makondo, Ismail Yunus Akhalwaya, Haifeng Qian, Ronald Fagin, Francisco Barahona, Udit Sharma, et al. Logical neural networks. arXiv preprint arXiv:2006.13155, 2020.  
Tim Roktaschel and Sebastian Riedel. End-to-end Differentiable Proving. In NeurIPS 2017, 2017.  
Hikaru Shindo, Masaaki Nishino, and Akihiro Yamamoto. Differentiable inductive logic programming for structured examples. In AAAI 2021, 2021.  
Xujie Si, Mukund Raghothaman, Kihong Heo, and Mayur Naik. Synthesizing datalog programs using numerical relaxation. In *IJCAI* 2019, 2019.  
Wolfgang Stammer, Patrick Schramowski, and Kristian Kersting. Right for the right concept: Revising neuro-symbolic concepts by interacting with their explanations. In CVPR 2021, 2021.  
Gustav Šourek, Vojtěch Aschenbrenner, Filip Železný, Steven Schockaert, and Ondřej Kuzelka. Lifted relational neural networks: Efficient learning of latent relational structures. JAIR, 2018.  
Jingyi Xu, Zilu Zhang, Tal Friedman, Yitao Liang, and Guy Van den Broeck. A semantic loss function for deep learning with symbolic knowledge. In ICML, 2018.  
Fan Yang, Zhilin Yang, and William W. Cohen. Differentiable learning of logical rules for knowledge base reasoning. In NeurIPS, 2017.  
Zhun Yang, Adam Ishay, and Joohyung Lee. Neurasp: Embracing neural networks into answer set programming. In *IJCAI*, 2020.
