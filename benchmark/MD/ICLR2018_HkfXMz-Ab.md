# NEURAL SKETCH LEARNING FOR CONDITIONAL PROGRAM GENERATION

Anonymous authors

Paper under double-blind review

# ABSTRACT

We study the problem of generating source code in a strongly typed, Java-like programming language, given a label (for example a set of API calls or types) carrying a small amount of information about the code that is desired. The generated programs are expected to respect a "realistic" relationship between programs and labels, as exemplified by a corpus of labeled programs available during training.

Two challenges in such conditional program generation are that the generated programs must satisfy a rich set of syntactic and semantic constraints, and that source code contains many low-level features that impede learning. We address these problems by training a neural generator not on code but on program sketches, or models of program syntax that abstract out names and operations that do not generalize across programs. During generation, we infer a posterior distribution over sketches, then concretize samples from this distribution into type-safe programs using combinatorial techniques. We implement our ideas in a system for generating API-heavy Java code, and show that it can often predict the entire body of a method given just a few API calls or data types that appear in the method.

# 1 INTRODUCTION

Neural networks have been successfully applied to many generative modeling tasks in the recent past (Oord et al., 2016; Ha & Eck, 2017; Vinyals et al., 2015). However, the use of these models in generating highly structured text remains relatively understudied. In this paper, we present a method, combining neural and combinatorial techniques, for the condition generation of an important category of such text: the source code of programs in Java-like programming languages.

The specific problem we consider is one of supervised learning. During training, we are given a set of programs, each program annotated with a label, which may contain information such as the set of API calls or the types used in the code. Our goal is to learn a function  $g$  such that for a test case of the form (X, Prog) (where Prog is a program and X is a label),  $g(X)$  is a compileable, type-safe program that is equivalent to Prog.

This problem has immediate applications in helping humans solve programming tasks (Hindle et al., 2012; Raychev et al., 2014). In the usage scenario that we envision, a human programmer uses a label to specify a small amount of information about a program that they have in mind. Based on this information, our generator seeks to produce a program equivalent to the "target" program, thus performing a particularly powerful form of code completion.

Conditional program generation is a special case of program synthesis (Manna & Waldinger, 1971; Summers, 1977), the classic problem of generating a program given a constraint on its behavior. This problem has received significant interest in recent years (Alur et al., 2013; Gulwani et al., 2017). In particular, several neural approaches to program synthesis driven by input-output examples have emerged (Balog et al., 2016; Parisotto et al., 2016; Devlin et al., 2017). Fundamentally, these approaches are tasked with associating a program's syntax with its semantics. As doing so in general is extremely hard, these methods choose to only generate programs in highly controlled domain-specific languages. For example, Balog et al. (2016) consider a functional language in which the only data types permitted are integers and integer arrays, control flow is linear, and there is a sum total of 15 library functions. Given a set of input-output examples, their method predicts a vector of binary attributes indicating the presence or absence of various tokens (library functions) in the target program, and uses this prediction to guide a combinatorial search for programs.

In contrast, in conditional program generation, we are already given a set of tokens (for example library functions or types) that appear in a program or its metadata. Thus, we sidestep the problem of learning the semantics of the programming language from data. We ask: does this simpler setting permit the generation of programs from a much richer, Java-like language, with one has thousands of data types and API methods, rich control flow and exception handling, and a strong type system?

While simpler than general program synthesis, this problem is still highly nontrivial. Perhaps the central issue is that to be acceptable to a compiler, a generated program must satisfy a rich set of structural and semantic constraints such as "do not use undeclared variables as arguments to a procedure call" or "only use API calls and variables in a type-safe way". Learning such constraints automatically from data is hard. Moreover, as this is also a supervised learning problem, the generated programs also have to follow the patterns in the data while satisfying these constraints.

We approach this problem with a combination of neural learning and type-guided combinatorial search (Feser et al., 2015). Our central idea is to learn not over source code, but over tree-structured syntactic models, or sketches, of programs. A sketch abstracts out low-level names and operations from a program, but retains information about the program's control structure, the orders in which it invokes API methods, and the types of arguments and return values of these methods. We propose a particular kind of probabilistic encoder-decoder, called a Gaussian Encoder-Decoder or GED, to learn a distribution over sketches conditioned on labels. During synthesis, we sample sketches from this distribution, then flesh out these samples into type-safe programs using a combinatorial method for program synthesis. Doing so effectively is possible because our sketches are designed to contain rich information about control flow and types.

We have implemented our approach in a system called BAYOU. We evaluate BAYOU in the generation of API-manipulating Android methods, using a corpus of about 150,000 methods drawn from an online repository. Our experiments show that BAYOU can often generate complex method bodies, including methods implementing tasks not encountered during training, given a few tokens as input.

# 2 PROBLEM STATEMENT

Now we define conditional program generation. Assume a universe  $\mathbb{P}$  of programs and a universe  $\mathbb{X}$  of labels. Also assume a set of training examples of the form  $\{(X_1, \operatorname{Prog}_1), (X_2, \operatorname{Prog}_2), \ldots\}$ , where each  $X_i$  is a label and each  $\operatorname{Prog}_i$  is a program. These examples are sampled from an unknown distribution  $Q(X, \operatorname{Prog})$ , where  $X$  and  $\operatorname{Prog}$  range over labels and programs, respectively. $^1$

We assume an equivalence relation  $Eqv \subseteq \mathbb{P} \times \mathbb{P}$  over programs. If  $(\operatorname{Prog}_1, \operatorname{Prog}_2) \in Eqv$ , then  $\operatorname{Prog}_1$  and  $\operatorname{Prog}_2$  are functionally equivalent. The definition of functional equivalence differs across applications, but in general it asserts that two programs are "just as good as" one another.

The goal of conditional program generation is to use the training set to learn a function  $g: \mathbb{X} \to \mathbb{P}$  such that the expected value  $\mathbf{E}[I((g(X), \text{Prog}) \in Eqv)]$  is maximized. Here,  $I$  is the indicator function, returning 1 if its boolean argument is true, and 0 otherwise. Informally, we are attempting to learn a function  $g$  such that if we sample  $(\mathsf{X}, \text{Prog}) \sim Q(X, \text{Prog})$ ,  $g$  should be able to reconstitute a program that is functionally equivalent to  $\text{Prog}$ , using only the label  $\mathsf{X}$ .

Instantiation In this paper, we consider a particular form of conditional program generation. We take the domain  $\mathbb{P}$  to be the set of possible programs in a programming language called AML that captures the essence of API-heavy Java programs (see Appendix A for more details). AML includes complex control flow such as loops, if-then statements, and exceptions; access to Java API data types; and calls to Java API methods. AML is a strongly typed language, and by definition,  $\mathbb{P}$  only includes programs that are type-safe. To define labels, we assume three finite sets: a set Calls of possible API calls in AML, a set Types of possible object types, and a set Keys of keywords, defined as words, such as "read" and "file", that often appear in textual descriptions of what programs do. The space of possible labels is  $\mathbb{X} = 2^{Calls} \times 2^{Types} \times 2^{Keys}$  (here  $2^S$  is the power set of  $S$ ).

(a)  
String s;   
BufferedReader br;   
FileReader fr;   
try { fr  $=$  new FileReader(\$String); br  $=$  new BufferedReader(hr); while  $(\langle s = br.readLine()\rangle !=$  null){ br.close(); } catch (FileNotFoundException e) { } catch (IOException e) {

(b)  
```java
String s;   
BufferedReader br;   
InputStreamReader isr;   
try { isr \(=\) new InputStreamReader(\$InputStream); br \(=\) new BufferedReader(isr); while \((s = br.readLine()) != null)\{\} catch (IOException e){ }
```

Defining  $Eqv$  in practice is tricky. For example, a reasonable definition of  $Eqv$  is that  $(\mathrm{Prog}_1, \mathrm{Prog}_2) \in Eqv$  iff  $\mathrm{Prog}_1$  and  $\mathrm{Prog}_2$  produce the same outputs on all inputs. But given the richness of AML, the problem of determining whether two AML programs always produce the same output is undecidable. As such, in practice we can only measure success indirectly, by checking whether the programs use the same control structures, and whether they can produce the same API call sequences. We will discuss this issue more in Section 6.

Example Consider the label  $\mathsf{X} = (\mathsf{X}_{Calls},\mathsf{X}_{Types},\mathsf{X}_{Keys})$  where  $\mathsf{X}_{Calls} = \{\text{readLine}\}$  and  $\mathsf{X}_{Types}$  and  $\mathsf{X}_{Keys}$  are empty. Figure 1(a) shows a program that our best learner stochastically returns given this input. As we see, this program indeed reads lines from a file, whose name is given by a special variable $String that the code takes as input. It also handles exceptions and closes the reader, even though these actions were not directly specified.

Although the program in Figure 1-(a) matches the label well, failures do occur. Sometimes, the system generates a program as in Figure 1-(b), which uses an InputStreamReader rather than a FileReader. It is possible to rule out this program by adding to the label. Suppose we amend  $\mathsf{X}_{Types}$  so that  $\mathsf{X}_{Types} = \{\text{FileReader}\}$ . BAYOU now tends to only generate programs that use FileReader. The variations then arise from different ways of handling exceptions and constructing FileReader objects (some programs use a String argument, while others use a File object). Figure 7 in the appendix shows two other top-five programs returned on this input.

# 3 TECHNICAL APPROACH

![](images/1b45f55841d48609fd734656d109d4c58b756927f6074a55680773a26dc2a95e.jpg)  
Figure 1: Programs generated by BAYOU with the API method name readLine as a label. Names of variables of type T whose values are obtained from the environment are of the form $T.  
Figure 2: Bayes net for  $\text{Prog}, X, Y$

Our approach is to learn  $g$  via maximum conditional likelihood estimation (CLE). That is, given a distribution family  $P(Prog|X,\theta)$  for a parameter set  $\theta$ , we choose  $\theta^{*} = \arg \max_{\theta}\sum_{i}\log P(\mathsf{Prog}_{i}\mid X_{i},\theta)$ . Then,  $g(\mathsf{X}) = \arg \max_{\mathsf{Prog}}P(\mathsf{Prog}|\mathsf{X},\theta^{*})$ .

The key innovation of our approach is that here, learning happens at a higher level of abstraction than  $(\mathsf{X}_i,\mathsf{Prog}_i)$  pairs. In practice, Java-like

programs contain many low-level details (for example, variable names and intermediate results) that can obscure patterns in code. Further, they contain complicated semantic rules (for example, for type safety) that are difficult to learn from data. In contrast, these are relatively easy for a combinatorial, syntax-guided program synthesizer (Alur et al., 2013) to deal with. However, synthesizers have a notoriously difficult time figuring out the correct "shape" of a program (such as the placement of loops and conditionals), which we hypothesize should be relatively easy for a statistical learner.

Specifically, our approach learns over sketches: tree-structured data that capture key facets of program syntax. A sketch  $\Upsilon$  does not contain low-level variable names and operations, but carries information about broadly shared facets of programs such as the types and API calls. During generation, a program synthesizer is used to generate programs from sketches produced by the learner.

Let the universe of all sketches be denoted by  $\mathbb{Y}$ . The sketch for a given program is computed by applying an abstraction function  $\alpha : \mathbb{P} \to \mathbb{Y}$ . We call a sketch  $\mathsf{Y}$  satisfiable, and write  $sat(\mathsf{Y})$ , if  $\alpha^{-1}(\mathsf{Y}) \neq \emptyset$ . The process of generating (type-safe) programs given a satisfiable sketch  $\mathsf{Y}$  is probabilistic, and captured by a concretization distribution  $P(Prog \mid \mathsf{Y}, sat(\mathsf{Y}))$ . We require that for all programs  $Prog$  and sketches  $\mathsf{Y}$  such that  $sat(\mathsf{Y})$ , we have  $P(Prog \mid \mathsf{Y}) \neq 0$  only if  $\mathsf{Y} = \alpha(Prog)$ .

![](images/7e59fd4e9a231a24fd988390481caabba2da553d163f2cb69cf3e40f8cb5a229.jpg)  
Figure 3: Grammar for sketches

Let us define a random variable  $Y = \alpha (Prog)$ . We assume that the variables  $X, Y$  and  $Prog$  are related as in the Bayes net in Figure 2. Specifically, given  $Y$ ,  $Prog$  is conditionally independent of  $X$ . Further, let us assume a distribution family  $P(Y|X,\theta)$  parameterized on  $\theta$ .

Let  $\Upsilon_{i} = \alpha (\mathrm{Prog}_{i})$ , and note that  $P(\mathrm{Prog}_i|\Upsilon)\neq 0$  only if  $\Upsilon = \Upsilon_{i}$ . Our problem now simplifies to learning over sketches, i.e., finding

$$
\begin{array}{l} \theta^ {*} = \arg \max _ {\theta} \sum_ {i} \log \sum_ {\Upsilon : s a t (\Upsilon)} P \left(\operatorname {P r o g} _ {i} | \Upsilon\right) P \left(\Upsilon \mid \mathrm {X} _ {i}, \theta\right) \\ = \arg \max _ {\theta} \sum_ {i} \log P (\operatorname {P r o g} _ {i} | \Upsilon_ {i}) P (\Upsilon_ {i} | \mathsf {X} _ {i}, \theta) = \arg \max _ {\theta} \sum_ {i} \log P (\Upsilon_ {i} | \mathsf {X} _ {i}, \theta). \qquad (1) \\ \end{array}
$$

Instantiation Figure 3 shows the full grammar for sketches in our implementation. Here,  $\tau_0, \tau_1, \ldots$  range over a finite set of API data types that AML programs can use. A data type, akin to a Java class, is identified with a finite set of API method names (including constructors), and  $a$  ranges over these names. Note that sketches do not contain constants or variable names.

A full definition of the abstraction function for AML appears in Appendix B. As an example, API calls in AML have the syntax "call  $e.a(e_1, \ldots, e_k)$ ", where  $a$  is an API method, the expression  $e$  evaluates to the object on which the method is called, and the expressions  $e_1, \ldots, e_k$  evaluate to the arguments of the method call. We abstract this call into an abstract method call "call  $\tau.a(\tau_1, \ldots, \tau_k)$ ", where  $\tau$  is the type of  $e$  and  $\tau_i$  is the type of  $e_i$ . The keywords skip, while, if-then-else, and try-catch preserve information about control flow and exception handling. Boolean conditions Cseq are replaced by abstract expressions: lists whose elements abstract the API calls in Cseq.

# 4 LEARNING

Now we describe our learning approach. Equation 1 leaves us with the problem of computing  $\arg \max_{\theta}\sum_{i}\log P(\Upsilon_i|\mathsf{X}_i,\theta)$ , when each  $\mathsf{X}_i$  is a label and  $\Upsilon_{i}$  is a sketch. Our answer is to utilize an encoder-decoder and introduce a real vector-valued latent variable  $Z$  to stochastically link labels and sketches:  $P(\mathsf{Y}|\mathsf{X},\theta) = \int_{\mathsf{Z}\in \mathbb{R}^{m}}P(\mathsf{Z}|\mathsf{X},\theta)P(\mathsf{Y}|\mathsf{Z},\theta)d\mathsf{Z}$ .

$P(Y|\mathsf{Z},\theta)$  is realized as a probabilistic decoder mapping a vector-valued variable to a distribution over trees. We describe this decoder in Appendix C. As for  $P(Z|\mathsf{X},\theta)$ , this distribution can, in principle, be picked in any way we like. In practice, because both  $P(Y|\mathsf{Z},\theta)$  and  $P(Z|\mathsf{X},\theta)$  have neural components with numerous parameters, we wish this distribution to regularize the learner. To provide this regularization, we assume a Normal  $(\vec{0},\mathbf{I})$  prior on  $Z$ .

Recall that our labels are of the form  $\mathsf{X} = \langle \mathsf{X}_{Calls},\mathsf{X}_{Types},\mathsf{X}_{Keys}\rangle$ , where  $\mathsf{X}_{Calls}$ ,  $\mathsf{X}_{Types}$ , and  $\mathsf{X}_{Keys}$  are sets. Assuming that the  $j$ -th elements  $\mathsf{X}_{Calls,j}$ ,  $\mathsf{X}_{Types,j}$ , and  $\mathsf{X}_{Keys,j}$  of these sets are generated independently, and assuming a function  $f$  for encoding these elements, let:

$$
\begin{array}{l} P (\mathsf {X} | \mathsf {Z}, \theta) = \left(\prod_ {j} \mathrm {N o r m a l} (f (\mathsf {X} _ {C a l l s, j}) | \mathsf {Z}, \mathbf {I} \sigma_ {C a l l s} ^ {2})\right) \left(\prod_ {j} \mathrm {N o r m a l} (f (\mathsf {X} _ {T y p e s, j}) | \mathsf {Z}, \mathbf {I} \sigma_ {T y p e s} ^ {2})\right) \\ \left(\prod_ {j} \operatorname {N o r m a l} (f (\mathsf {X} _ {\text {K e y s}, j}) | \mathsf {Z}, \mathbf {I} \sigma_ {\text {K e y s}} ^ {2})\right). \\ \end{array}
$$

That is, the encoded value of each  $\mathsf{X}_{\mathsf{Types},j}$ ,  $\mathsf{X}_{\mathsf{Calls},j}$  or  $\mathsf{X}_{\mathsf{Keys},j}$  is sampled from a high-dimensional Normal distribution centered at  $\mathsf{Z}$ . If  $f$  is 1-1 and onto with the set  $\mathbb{R}^m$  then from Normal-Normal conjugacy, we have:  $P(\mathsf{Z}|\mathsf{X}) = \mathrm{Normal}\left(\mathsf{Z}\mid \frac{\overline{\mathsf{X}}}{1 + n},\frac{1}{1 + n}\mathbf{I}\right)$ , where

$$
\overline {{\sf X}} = \left(\sigma_ {T y p e s} ^ {- 2} \sum_ {j} f ({\sf X} _ {T y p e s, j})\right) + \left(\sigma_ {C a l l s} ^ {- 2} \sum_ {j} f ({\sf X} _ {C a l l s, j})\right) + \left(\sigma_ {K e y s} ^ {- 2} \sum_ {j} f ({\sf X} _ {K e y s, j})\right)
$$

and  $n = n_{Types} \sigma_{Types}^{-2} + n_{Calls} \sigma_{Calls}^{-2} + n_{Keys} \sigma_{Keys}^{-2}$ . Here,  $n_{Types}$  is the number of types supplied, and  $n_{Calls}$  and  $n_{Keys}$  are defined similarly.

Note that this particular  $P(Z|\mathsf{X},\theta)$  only follows directly from the Normal  $(\vec{0},\mathbf{I})$  prior on  $Z$  and Normal likelihood  $P(X|\mathsf{Z},\theta)$  if the encoding function  $f$  is 1-1 and onto. However, even if  $f$  is not 1-1 and onto (as will be the case if  $f$  is implemented with a standard feed-forward neural network) we can still use this probabilistic encoder, and in practice we still tend to see the benefits of the regularizing prior on  $Z$ , with  $P(Z)$  distributed approximately according to a unit Normal. We call this type of encoder-decoder, with a single, Normally-distributed latent variable  $Z$  linking the input and output, a Gaussian encoder-decoder, or GED for short.

Now that we have chosen  $P(X|Z, \theta)$  and  $P(Y|Z, \theta)$ , we must choose  $\theta$  to perform CLE. Note that:

$$
\begin{array}{l} \sum_ {i} \log P (\Upsilon_ {i} | \mathsf {X} _ {i}, \theta) = \sum_ {i} \log \int_ {\mathsf {Z} \in \mathbb {R} ^ {m}} P (\mathsf {Z} | \mathsf {X} _ {i}, \theta) P (\mathsf {Y} _ {i} | \mathsf {Z}, \theta) d \mathsf {Z} = \sum_ {i} \log \mathbf {E} _ {\mathsf {Z} \sim P (\mathsf {Z} | \mathsf {X} _ {i}, \theta)} [ P (\mathsf {Y} _ {i} | \mathsf {Z}, \theta) ] \\ \geq \sum_ {i} \mathbf {E} _ {\mathsf {Z} \sim P (Z | \mathsf {X} _ {i}, \theta)} [ \log P (\mathsf {Y} _ {i} | \mathsf {Z}, \theta) ] = \mathcal {L} (\theta). \\ \end{array}
$$

where the  $\geq$  holds due to Jensen's inequality. Hence,  $\mathcal{L}(\theta)$  serves as a lower bound on the log-likelihood, and so we can compute  $\theta^{*} = \arg \max_{\theta}\mathcal{L}(\theta)$  as a proxy for the CLE. We maximize this lower bound using stochastic gradient ascent; as  $P(Z|\mathsf{X}_i,\theta)$  is Normal, we can use the reparameterization trick common in variational auto-encoders (Kingma & Welling, 2014) while doing so. The parameter set  $\theta$  contains all of the parameters of the encoding function  $f$  as well as  $\sigma_{Types}$ ,  $\sigma_{Calls}$ , and  $\sigma_{Keys}$ , and the parameters used in the decoding distribution functor  $P(Y|\mathsf{Z},\theta)$ .

# 5 COMBINATORIAL CONCRETIZATION

The final step in our algorithm is to "concretize" sketches into programs, following the distribution  $P(Prog|Y)$ . Our method of doing so is a type-directed, stochastic search procedure that builds on combinatorial methods for program synthesis (Schkufza et al., 2016; Feser et al., 2015).

Given a sketch  $\Upsilon$ , our procedure performs a random walk in a space of partially concretized sketches (PCSs). A PCS is a term obtained by replacing some of the abstract method calls and expressions in a sketch by AML method calls and AML expressions. For example, the term " $x_1.a(x_2);\tau_1.b(\tau_2)$ ", which sequential composes an abstract method call to  $b$  and a "concrete" method call to  $a$ , is a PCS. The state of the procedure at the  $i$ -th point of the walk is a PCS  $\mathsf{H}_i$ . The initial state is  $\Upsilon$ .

Each state  $\mathsf{H}$  has a set of neighbors Next(H). This set consists of all PCS-s  $\mathsf{H}'$  that are obtained by concretizing a single abstract method call or expression in  $\mathsf{H}$ , using variable names in a way that is consistent with the types of all API methods and declared variables in  $\mathsf{H}$ .

The  $(i + 1)$ -th state in a walk is a sample from a predefined, heuristically chosen distribution  $P(\mathsf{H}_{i + 1} \mid \mathsf{H}_i)$ . The only requirement on this distribution is that it assigns nonzero probability to a state iff it belongs to  $Next(\mathsf{H}_i)$ . In practice, our implementation of this distribution prioritizes programs that are simpler. The random walk ends when it reaches a state  $\mathsf{H}^*$  that has no neighbors. If  $\mathsf{H}^*$  is fully concrete (that is, an AML program), then the walk is successful and  $\mathsf{H}^*$  is returned as a sample. If not, the current walk is rejected, and a fresh walk is started from the initial state.

Recall that the concretization distribution  $P(Prog|\Upsilon)$  is only defined for sketches  $\Upsilon$  that are satisfiable. Our concretization procedure does not assume that its input  $\Upsilon$  is satisfiable. However, if  $\Upsilon$  is not satisfiable, all random walks that it performs end with rejection, causing it to never terminate.

While the worst-case complexity of this procedure is exponential in the generated programs, it performs well in practice because of our chosen language of sketches. For instance, our search does not need to discover the high-level structure of programs. Also, sketches specify the types of method arguments and return values, and this significantly limits the search space.

# 6 EXPERIMENTS

Now we present an empirical evaluation of the effectiveness of our method. The experiments we describe utilize data from an online repository of about 1500 Android apps (AndroidDrawer, 2017).

We decomposed the APKs using JADX (skylot, 2017) to generate their source code. Analyzing about 100 million lines of code that were generated, we extracted 150,000 methods that used Android APIs or the Java IO library. We then pre-processed all method bodies to translate the code from Java to AML, preserving names of relevant API calls and data types as well as the high-level control flow. Hereafter, when we say "program" we refer to an AML program.

<table><tr><td></td><td>Min</td><td>Max</td><td>Median</td><td>Vocab</td></tr><tr><td>X Calls</td><td>1</td><td>9</td><td>2</td><td>2584</td></tr><tr><td>XTypes</td><td>1</td><td>15</td><td>3</td><td>1521</td></tr><tr><td>XKeys</td><td>2</td><td>29</td><td>8</td><td>993</td></tr><tr><td>X</td><td>4</td><td>48</td><td>13</td><td>5098</td></tr></table>

Figure 4: Statistics on labels

From each program, we extracted the sets  $\mathsf{X}_{\text{Calls}}, \mathsf{X}_{\text{Types}}$  and  $\mathsf{X}_{\text{Keys}}$  as well as a sketch  $\Upsilon$ . Lacking separate natural language descriptions for programs, we defined keywords to be words obtained by splitting the names of the API types and calls that the program uses, based on camel case. For instance, the keywords obtained from the API call readLine are "read" and "line". As API method and types in Java tend to be carefully named, these words often

contain rich information about what programs do. Figure 4 gives some statistics on the sizes of the labels in the data. From the extracted data, we randomly selected 10,000 programs to be in the testing and validation data each.

Implementation and training We implemented our approach in our tool called BAYOU, using TensorFlow (Abadi et al., 2015) to implement the GED neural model, and the Eclipse IDE for the abstraction from Java to the language of sketches and the combinatorial concretization. In all our experiments we performed cross-validation through grid search and picked the best performing model. Our hyper-parameters for training the model are as follows. We used 64, 32 and 64 units in the encoder for API calls, types and keywords, respectively, and 128 units in the decoder. The latent space was 32-dimensional. We used a mini-batch size of 50, a learning rate of 0.0006 for the Adam gradient-descent optimizer (Kingma & Ba, 2014), and ran the training for 50 epochs.

The training was performed on an AWS "p2.xlarge" machine with an NVIDIA K80 GPU with 12GB GPU memory. As each sketch was broken down into a set of production paths, the total number of data points fed to the model was around 700,000 per epoch. Training took 10 hours to complete.

![](images/586dcc9e49dba39d6d84d993d87f31175f4229487e9b463cb335f3b1cf9cb9f4.jpg)  
Figure 5: 2-dimensional projection of latent space

Clustering To visualize clustering in the 32-dimensional latent space, we provided labels  $\mathsf{X}$  from the testing data and sampled  $\mathsf{Z}$  from  $P(Z|\mathsf{X})$ , and then used it to sample a sketch from  $P(Y|\mathsf{Z})$ . We then used t-SNE (Maaten & Hinton, 2008) to reduce the dimensionality of  $\mathsf{Z}$  to 2-dimensions, and labeled each point with the API used in the sketch  $\mathsf{Y}$ . Figure 5 shows this 2-dimensional space, where each label has been coded with a different color. It is immediately apparent from the plot that the model has learned to cluster the latent space neatly according to different APIs. Some APIs such as java.io have several modes, and we noticed separately that each mode corresponds to different usage scenarios of the API, such as reading versus writing in this case.

Accuracy To evaluate prediction accuracy, we provided labels from the testing data to our model, sampled sketches from the distribution  $P(Y|\mathsf{X})$  and concretized each sketch into an AML program using our combinatorial search. We then measured the number of test programs for which a program that is equivalent to the expected one appeared in the top-10 results from the model.

As there is no universal metric to measure program equivalence (in fact, it is an undecidable problem in general), we used several metrics to approximate the notion of equivalence. We defined the following metrics on the top-10 programs predicted by the model:

M1. This binary metric measures whether the expected program appeared in a syntactically equivalent form in the results. Of course, an impediment to measuring this is that the names of variables used in the expected and predicted programs may not match. It is

neither reasonable nor useful for any model of code to learn the exact variable names in the training data. Therefore, in performing this equivalence check, we abstract away the variable names and compare the rest of the program's Abstract Syntax Tree (AST) instead.

M2. This metric measures the minimum Jaccard distance between the sets of sequences of API calls made by the expected and predicted programs. It is a measure of how close to the original program were we able to get in terms of sequences of API calls.  
M3. Similar to metric M2, this metric measures the minimum Jaccard distance between the sets of API calls in the expected and predicted programs.  
M4. This metric computes the minimum absolute difference between the number of statements in the expected and sampled programs, as a ratio of that in the former.  
M5. Similar to metric M4, this metric computes the minimum absolute difference between the number of control structures in the expected and sampled programs, as a ratio of that in the former. Examples of control structures are branches, loops, and try-catch statements.

Partial Observability To evaluate our model's ability to predict programs given a small amount of information about its code, we varied the fraction of the set of API calls, types, and keywords provided as input from the testing data. We experimented with  $75\%$ ,  $50\%$  and  $25\%$  observability in the testing data; the median number of items in a label in these cases were 9, 6, and 2, respectively.

Competing Models In order to compare our model with state-of-the-art conditional generative models, we implemented the Gaussian Stochastic Neural Network (GSNN) presented by (Sohn et al., 2015), using the same tree-structured decoder as the GED. There are two main differences: (i) the GSNN's decoder is also conditioned directly on the input label  $X$  in addition to  $Z$ , which we accomplish by concatenating its initial state with the encoding of  $X$ , (ii) the GSNN loss function has an additional KL-divergence term weighted by a hyper-parameter  $\beta$ . We subjected the GSNN to the same training and cross-validation process as our model. In the end, we selected a model that happened to have very similar hyper-parameters as ours, with  $\beta = 0.001$ .

Evaluating Sketches In order to evaluate the effect of sketch learning for program generation, we implemented and compared with a model that learns directly over programs. Specifically, the neural network structure is exactly the same as ours, except that instead of being trained on production paths in the sketches, the model is trained on production paths in the ASTs of the AML programs. We selected a model that had more units in the decoder (256) compared to our model (128), as the AML grammar is more complex than the grammar of sketches. We also implemented a similar GsNN model to train over AML ASTs directly.

Figure 6 shows the collated results of this evaluation, where each entry computes the average of the corresponding metric over the 10000 test programs. It takes our model about 8 seconds, on average, to generate and rank 10 programs.

When testing models that were trained on AML ASTs, namely the GED-AML and GSNN-AML models, we observed that out of a total of 87,486 AML ASTs sampled from the two models, 2525 (or  $3\%$ ) ASTs were not even well-formed, i.e., they would not pass a parser, and hence had to be discarded from the metrics. This number is 0 for the GED-Sk and GSNN-Sk models, meaning that all AML ASTs that were obtained by concretizing sketches were well-formed.

In general, one can observe that the GED-Sk model performs best overall, with GSNN-Sk a reasonable alternative. We hypothesize that the reason GED-Sk performs slightly better is the regularizing prior on  $Z$ ; since the GSNN has a direct link from  $X$  to  $Y$ , it can choose to ignore this regularization. We would classify both these models as suitable for conditional program generation. However, the other two models GED-AML and GSNN-AML perform quite worse, showing that sketch learning is key in addressing the problem of conditional program generation.

Generalization To evaluate how well our model generalizes to unseen data, we gather a subset of the testing data whose data points, consisting of label-sketch pairs  $(X,Y)$ , never occurred in the training data. We then evaluate the same metrics in Figure 6(a)-(e), but due to space reasons we focus on the  $50\%$  observability column. Figure 6(f) shows the results of this evaluation on the subset of 5126 (out of 10000) unseen test data points. The metrics exhibit a similar trend, showing that the

<table><tr><td rowspan="2">Model</td><td colspan="4">Input Label Observability</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>GED-AML</td><td>0.13</td><td>0.09</td><td>0.07</td><td>0.02</td></tr><tr><td>GSNN-AML</td><td>0.07</td><td>0.04</td><td>0.03</td><td>0.01</td></tr><tr><td>GED-Sk</td><td>0.59</td><td>0.51</td><td>0.44</td><td>0.21</td></tr><tr><td>GSNN-Sk</td><td>0.57</td><td>0.48</td><td>0.41</td><td>0.18</td></tr></table>

(a) M1. Proportion of test programs for which the expected AST appeared in the top-10 results.

<table><tr><td rowspan="2">Model</td><td colspan="4">Input Label Observability</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>GED-AML</td><td>0.52</td><td>0.58</td><td>0.61</td><td>0.77</td></tr><tr><td>GSNN-AML</td><td>0.59</td><td>0.64</td><td>0.68</td><td>0.83</td></tr><tr><td>GED-Sk</td><td>0.11</td><td>0.17</td><td>0.22</td><td>0.50</td></tr><tr><td>GSNN-Sk</td><td>0.13</td><td>0.19</td><td>0.25</td><td>0.52</td></tr></table>

(c) M3. Average minimum Jaccard distance on the set of API methods called in the test program vs the top-10 results.

<table><tr><td rowspan="2">Model</td><td colspan="4">Input Label Observability</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>GED-AML</td><td>0.31</td><td>0.30</td><td>0.30</td><td>0.34</td></tr><tr><td>GSNN-AML</td><td>0.32</td><td>0.31</td><td>0.32</td><td>0.39</td></tr><tr><td>GED-Sk</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.04</td></tr><tr><td>GSNN-Sk</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.03</td></tr></table>

(e) M5. Average minimum difference between the number of control structures in the test program vs the top-10 results.

<table><tr><td rowspan="2">Model</td><td colspan="4">Input Label Observability</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>GED-AML</td><td>0.82</td><td>0.87</td><td>0.89</td><td>0.97</td></tr><tr><td>GSNN-AML</td><td>0.88</td><td>0.92</td><td>0.93</td><td>0.98</td></tr><tr><td>GED-Sk</td><td>0.34</td><td>0.43</td><td>0.50</td><td>0.76</td></tr><tr><td>GSNN-Sk</td><td>0.36</td><td>0.46</td><td>0.53</td><td>0.78</td></tr></table>

(b) M2. Average minimum Jaccard distance on the set of sequences of API methods called in the test program vs the top-10 results.

<table><tr><td rowspan="2">Model</td><td colspan="4">Input Label Observability</td></tr><tr><td>100%</td><td>75%</td><td>50%</td><td>25%</td></tr><tr><td>GED-AML</td><td>0.49</td><td>0.47</td><td>0.46</td><td>0.46</td></tr><tr><td>GSNN-AML</td><td>0.52</td><td>0.49</td><td>0.49</td><td>0.53</td></tr><tr><td>GED-Sk</td><td>0.05</td><td>0.06</td><td>0.06</td><td>0.09</td></tr><tr><td>GSNN-Sk</td><td>0.05</td><td>0.06</td><td>0.06</td><td>0.09</td></tr></table>

(d) M4. Average minimum difference between the number of statements in the test program vs the top-10 results.

<table><tr><td rowspan="2">Model</td><td colspan="5">Metric</td></tr><tr><td>M1</td><td>M2</td><td>M3</td><td>M4</td><td>M5</td></tr><tr><td>GED-AML</td><td>0.02</td><td>0.97</td><td>0.71</td><td>0.50</td><td>0.37</td></tr><tr><td>GSNN-AML</td><td>0.01</td><td>0.98</td><td>0.74</td><td>0.51</td><td>0.37</td></tr><tr><td>GED-Sk</td><td>0.23</td><td>0.70</td><td>0.30</td><td>0.08</td><td>0.04</td></tr><tr><td>GSNN-Sk</td><td>0.20</td><td>0.74</td><td>0.33</td><td>0.08</td><td>0.04</td></tr></table>

(f) Metrics for  $50\%$  observability evaluated only on unseen data

Figure 6: Accuracy of different models on testing data. GED-AML and GSNN-AML are baseline models trained over AML ASTs, GED-Sk and GSNN-Sk are models trained over sketches.

models based on sketch learning are able to generalize much better than the baseline models, and that the GED-Sk model performs the best.

# 7 RELATED WORK

So far as we know, we are the first to pose the problem of conditional program generation. However, unconditional, corpus-driven generation of programs has been studied before (Maddison & Tarlow, 2014; Allamanis & Sutton, 2014; Bielik et al., 2016), as has the generation of code snippets conditioned on a context into which the snippet is merged (Nguyen et al., 2013; Raychev et al., 2014; Nguyen & Nguyen, 2015). These prior efforts often use models like  $n$ -grams (Nguyen et al., 2013) and recurrent neural networks (Raychev et al., 2014) that are primarily suited to the generation of straight-line programs; almost universally, they cannot guarantee semantic properties of generated programs. Among prominent exceptions, Maddison & Tarlow (2014) use log-bilinear tree-traversal models, a class of probabilistic pushdown automata, for program generation. Bielik et al. (2016) study a generalization of probabilistic grammars known as probabilistic higher-order grammars. Like our work, these papers address the generation of programs that satisfy rich constraints such as the type-safe use of names. In principle, one could replace our decoder and the combinatorial concretizer, which together form an unconditional program generator, with one of these models. However, given our experiments, doing so is unlikely to lead to good performance in the end-to-end problem of conditional program generation.

Conditional program generation is closely related to program synthesis (Gulwani et al., 2017), the problem of producing programs that satisfy a given semantic specification. A recent body of work (Parisotto et al., 2016; Balog et al., 2016; Devlin et al., 2017) uses neural techniques to solve this problem. These efforts differ from ours in goals as well as methods. Our problem is simpler, as it is conditioned on syntactic, rather than semantic, facets of programs. This allows us to generate programs in a language that is much richer than the DSLs targeted by neural synthesis. Also, most of these approaches do not combine learning and combinatorial techniques. One exception is Deepcoder (Balog et al., 2016), whose relationship with our work was discussed in Section 1.

# REFERENCES

Martin Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyls, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. Tensorflow: Large-scale machine learning on heterogeneous distributed systems, 2015. URL http://download.tensorflow.org/paper/whitepaper2015.pdf.  
Miltiadis Allamanis and Charles Sutton. Mining idioms from source code. In Proceedings of the 22Nd ACM SIGSOFT International Symposium on Foundations of Software Engineering, FSE 2014, pp. 472-483, New York, NY, USA, 2014. ACM. ISBN 978-1-4503-3056-5. doi: 10.1145/2635868.2635901. URL http://doi.acm.org/10.1145/2635868.2635901.  
Rajeev Alur, Rastislav Bodík, Garvit Juniwal, Milo M. K. Martin, Mukund Raghothaman, Sanjit A. Seshia, Rishabh Singh, Armando Solar-Lezama, Emina Torlak, and Abhishek Udupa. Syntax-guided synthesis. In FMCAD, pp. 1-17, 2013.  
Android drawer. Android drawer. http://www.android drawer.com, 2017. [Online; accessed 06-Jul-2017].  
Matej Balog, Alexander L Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow. Deepcoder: Learning to write programs. arXiv preprint arXiv:1611.01989, 2016.  
Pavol Bielik, Veselin Raychev, and Martin T Vechev. PHOG: probabilistic model for code. In ICML, pp. 19-24, 2016.  
Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, and Pushmeet Kohli. Robustfill: Neural program learning under noisy I/O. In ICML, 2017.  
John K. Feser, Swarat Chaudhuri, and Isil Dillig. Synthesizing data structure transformations from input-output examples. In PLDI, pp. 229-239. ACM, 2015.  
Sumit Gulwani, Oleksandr Polozov, and Rishabh Singh. Program synthesis. Foundations and Trends in Programming Languages, 4(1-2):1-119, 2017.  
David Ha and Douglas Eck. A neural representation of sketch drawings. arXiv preprint arXiv:1704.03477, 2017.  
Abram Hindle, Earl T Barr, Zhendong Su, Mark Gabel, and Premkumar Devanbu. On the naturalness of software. In Software Engineering (ICSE), 2012 34th International Conference on, pp. 837-847. IEEE, 2012.  
Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.  
Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(Nov):2579-2605, 2008.  
C.J. Maddison and D. Tarlow. Structured generative models of natural source code. In ICML, 2014.  
Zohar Manna and Richard J. Waldinger. Toward automatic program synthesis. Communications of the ACM, 14(3):151-165, 1971.  
Anh Tuan Nguyen and Tien N. Nguyen. Graph-based statistical language model for code. In Proceedings of the 37th International Conference on Software Engineering - Volume 1, ICSE '15, pp. 858-868, Piscataway, NJ, USA, 2015. IEEE Press. ISBN 978-1-4799-1934-5. URL http://dl.acm.org/citation.cfm?id=2818754.2818858.  
Tung Thanh Nguyen, Anh Tuan Nguyen, Hoan Anh Nguyen, and Tien N. Nguyen. A statistical semantic language model for source code. In Proceedings of the 2013 9th Joint Meeting on Foundations of Software Engineering, ESEC/FSE 2013, pp. 532-542, New York, NY, USA, 2013. ACM. ISBN 978-1-4503-2237-9. doi: 10.1145/2491411.2491458. URL http://doi.acm.org/10.1145/2491411.2491458.  
Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759, 2016.

Emilio Parisotto, Abdel-rahman Mohamed, Rishabh Singh, Lihong Li, Dengyong Zhou, and Pushmeet Kohli. Neuro-symbolic program synthesis. arXiv preprint arXiv:1611.01855, 2016.  
Veselin Raychev, Martin Vechev, and Eran Yahav. Code completion with statistical language models. In PLDI, 2014.  
Eric Schkufza, Rahul Sharma, and Alex Aiken. Stochastic program optimization. Commun. ACM, 59(2): 114-122, 2016.  
skylot. JADX - Dex to Java decompiler. https://github.com/skylot/jADX, 2017. [Online; accessed 06-Jul-2017].  
Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, December 7-12, 2015, Montreal, Quebec, Canada, pp. 3483-3491, 2015.  
Phillip D Summers. A methodology for LISP program construction from examples. Journal of the ACM (JACM), 24(1):161-175, 1977.  
Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan. Show and tell: A neural image caption generator. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3156-3164, 2015.  
Xingxing Zhang, Liang Lu, and Mirella Lapata. Top-down tree long short-term memory networks. In NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016, pp. 310-320, 2016.

(a)  
String s;   
BufferedReader br;   
FileReader fr;   
try { fr  $=$  new BufferedReader(String); br  $=$  new BufferedReader(hr); while  $(\mathrm{~s~} =$  br.readLine())  $! =$  null){ br.close(); } catch (FileNotFoundException e){ _e.printStackTrace(); } catch (IOException e){ _e.printStackTrace(); }

(b)  
String s;   
BufferedReader br;   
FileReader fr;   
try { fr  $=$  new BufferedReader(\$File); br  $=$  new BufferedReader(hr); while  $(\mathrm{ss} =$  br.readLine(）！  $= \mathrm{null})\{\}$  br.close(); } catch (FileNotFoundException e){ } catch (IOException e){

Figure 7: Programs generated in a typical run of BAYOU, given the API method name readLine and the type FileReader.
