# TOWARDS A MATHEMATICS FORMALISATION ASSISTANT USING LARGE LANGUAGE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Mathematics formalisation is the task of writing mathematics (i.e., definitions, theorem statements, proofs) in natural language, as found in books and papers, into a formal language that can then be checked for correctness by a program. It is a thriving activity today, however formalisation remains cumbersome. In this paper, we explore the abilities of a large language model (Codex) to help with formalisation in the Lean theorem prover. We find that with careful input-dependent prompt selection and postprocessing, Codex is able to formalise short mathematical statements at undergrad level with nearly  $75\%$  accuracy for 120 theorem statements. For proofs quantitative analysis is infeasible and we undertake a detailed case study. We choose a diverse set of 13 theorems at undergrad level with proofs that fit in two-three paragraphs. We show that with a new prompting strategy Codex can formalise these proofs in natural language with at least one out of twelve Codex completion being easy to repair into a complete proof. This is surprising as essentially no aligned data exists for formalised mathematics, particularly for proofs. These results suggest that large language models are a promising avenue towards fully or partially automating formalisation.

# 1 INTRODUCTION

Mathematics (definitions, theorems, proofs, remarks) as found in books and papers is written in a semi-formal style combining natural language with formal language in specialized notation. We refer to the language of this style of writing mathematics as natural language or NL. Formalisation of mathematics consists of writing mathematics in a formal language that can then be checked and manipulated by a computer. NL mathematics writing, while being more rigorous than writing in most other domains, falls far short of the standard of detail and rigour required for full formalisation. Formalisation is done with the help of proof assistants. A proof assistant consists of a formal language in which mathematical statements can be encoded along with a piece of software that assists in writing and checking proofs in the formal language up to the foundational axioms. See under Prompt in Figure 1 for some examples. Formalisation is an old endeavour that is thriving with several actively developed libraries of formalised mathematics for major proof assistants including Coq, Isabelle, Lean and Mizar. A major use of proof assistants is in software and hardware verification but here we are concerned with their applications in mathematics: checking formalised mathematics automatically results in a much higher degree of confidence in the correctness of proofs. Formalisation promises to open up new possibilities in mathematical exposition, teaching, research and collaboration (Massot, 2021; Buzzard, 2022); in addition, it can facilitate automated proof discovery, e.g. (Lample et al., 2022).

Formalisation of mathematics today poses a barrier to entry because of the need to learn to use proof assistants; it is also notoriously labour-intensive because many details normally taken for granted in the language of mathematics must be supplied when formalising. Autoformalisation Wang et al. (2018) is the task of (semi-)automatically turning a piece of mathematics in natural language into a formalised one. An autoformalisation tool that speeds-up formalisation or fully automates it would be of great value by enabling the above advantages of formalisation and opening up new ones Szegedy (2020).

Autoformalisation is challenging. It is a natural language understanding problem for the language of mathematics. While the language of mathematics is stylized compared to natural language in

other domains and deals with relatively narrow subject matter, it retains much of the complexity in addition to presenting new challenges for autoformalisation of its own, including supplying missing details and assumptions that are taken for granted by humans, and semantically mapping concepts in the informal description to those in the formal corpus (Ganesalingam, 2013; Massot, 2021).

Autoformalisation also presents practical challenges in the application of modern deep learning-based methods: the amount of formalised mathematics available is much smaller than code in major programming languages. Furthermore, there is very little aligned data between informal and formal mathematics. Autoformalisation implicitly includes semantic search in the formalised library. Autoformalisation of proofs is much more than independent autoformalisation of each statement in the proof: one needs to maintain context across the proof and find correspondence between NL constructs and tactics in formal proofs.

In this paper we worked with Lean, a popular proof assistant with two actively used versions: Lean 3 (de Moura et al., 2015) and Lean4 (de Moura & Ullrich, 2021). The rapidly evolving Lean mathematical library (abbreviated mathlib) is one of the largest libraries of formal mathematics. Mathlib is currently 226MB in size. mathlib is monolithic by design, ensuring that formalisations of different parts of mathematics can be combined easily. The resulting standardization of terminology in mathlib and its good coverage make Lean an attractive target for autoformalisation.

To our knowledge, the only form in which aligned data occurs in mathlib is as docstrings for definitions and theorem statements. Furthermore, there is a complete lack of aligned data for proofs: while some examples of natural language proofs together with their corresponding formal proofs occur in the blueprints of some Lean formalisation projects, e.g. the Liquid Tensor Experiment, these are only a handful and highly specialised.

Our contributions. In this paper we apply a large language model (specifically, Codex) to the problem of automodality. We focused on two different tasks: (1) translating theorem statements of a form similar to docstrings of mathlib to theorems (in Lean 4), and (2) translating (outlines of) NL proofs to Lean proofs (in Lean 3). The latest version of Lean, Lean 4, is (in addition to an interactive theorem prover) a full-fledged programming language with a fast runtime. This allows a seamless integration of proofs, programs and meta-programs. We use Lean 4 for one set of experiments and Lean 3 for the other because, at the time of writing, mathlib was only partially available in Lean 4 (via a partial binary port). Hence we use Lean 4 where its additional capabilities are important and Lean 3 where these are not used and the larger library is of greater value. More details on Lean are in Appendix A.

Theorem statement autoformalisation. For the evaluation dataset, we chose 120 theorem statements at the undergrad level so that the relevant concepts (background theory and definitions) were mostly already in mathlib. Since mathlib is substantial (it has a significant fraction of undergrad mathematics curriculum apart from many advanced results), this is not a restriction. We focused on theorem statements at the undergrad and more advanced level from various areas of mathematics. These statements tend to be more challenging for autoformalisation compared to mathematics competition problems studied in prior work (Wu et al., 2022) as they often assume more in terms of implicit context and draw from a much larger background (Wu et al., 2022).

We experimented with using input-dependent prompting, with mathlib as a database. Specifically, we chose our few-shot prompts to consist of theorem-docstring pairs from mathlib where the docstring is close in a sentence similarity metric to the statement to be formalised. We also experimented with filtering outputs generated at high temperatures by checking validity in Lean 4 and some other post-processing.

Our results showed that there is a strong effect of both prompt engineering and selection, and even more when used in combination and that a reasonably large fraction are elaborated when both prompt engineering and selection is done (the results improve further when more prompts are used).

In the context of automodification, we are the first to use input-dependent prompting. Our use of elaboration for postprocessing is novel. Both of these are greatly facilitated by the availability of mathlib, and the nature of Lean 4, which gives easy access to its internals and in Lean 4 itself – the latter allowing shared code and avoiding context switching.

Autoformalisation of proofs. We chose an evaluation dataset of 13 NL theorems and their proofs. (Due to the lack of data for proofs and need for manual inspection of the outputs, a larger scale

study was infeasible.) The theorem statements are at the undergrad level with a proof fitting in two-three paragraphs. They are diverse across several axes: (1) proof techniques such as proof by contradiction, induction, algebraic manipulations etc., (2) domains such as topology, analysis, group theory, linear algebra, representation theory, and algebraic number theory, (3) difficulty level.

Our chosen proofs are much longer than a typical theorem statement and we didn't observe Codex outputting a completely correct proof. We instead relaxed the requirement of automodification to produce a (faulty) proof that is easy to repair for humans, saving time and effort compared to formalising from scratch. We experimented with several output formal proof formats depending on the level of detail, and with or without NL comments interspersed in the proofs. We designed fixed few-shot prompts for each of these formats. We undertook a detailed manual study of the outputs. We found that proofs with comments work better; for about half of the formal proofs Codex output would save significant effort and for the rest it would save some effort. Proofs with comments is a new kind of prompting strategy in line with other recent prompting strategies such as chain-of-thought prompting Wei et al. (2022). Presumably, interleaving of NL comments helps Codex align its output with the NL proof.

All of our datasets were carefully controlled for the possibility of overlap with the training data of Codex as we discuss in more detail later in the paper. We make all of our data available as supplementary material. In summary, our contributions are

- Design of a postprocessing technique for theorem statement autoformalisation resulting in significantly improved performance when combined with prompt engineering.  
- First study of proof automisation and design of a prompting technique for proof automisation. With this technique, Codex is able to produce useful partially correct formal proofs in at least one out of thirteen completions.  
- A detailed case study of proof automisation which may be useful for future work.

Organisation. After discussion of related work in Section 2, we discuss in detail autoformalisation of theorem statements in Section 3 and of proofs in Section 4. We conclude in Section 5.

# 2 BACKGROUND AND RELATED WORK

Natural language understanding has a long history in AI; here we can only briefly touch upon the most relevant subfields of this large field.

Semantic parsing and program synthesis from natural language specification. Semantic parsing is the task of translating a natural language utterance into a logical form. Tasks are normally restricted to specific domains and the logical forms come from a domain-specific language ranging from first-order logic to regular expressions, e.g. (Kamath & Das, 2019; Hahn et al., 2022).

Large Language Models and mathematics. The advent of transformer-based large language models (LLMs) for natural languages, e.g. (Devlin et al., 2018; Brown et al., 2020), has brought about a sea change in natural language processing. This is largely fueled by the remarkable ability of LLMs to achieve good performance on a diverse set of tasks, ranging from translation to solving math word problems, via few-shot demonstrations in the prompt even though the LLMs are only trained on the language modelling objective. With careful prompt design, these latent abilities can be further teased out, e.g., (Liu et al., 2021; Wei et al., 2022). The input-dependent prompting we use has precedent in prior work, e.g., Jain et al. (2022). Prompt design can be combined with postprocessing to select the best among many answers generated at higher temperatures, e.g., (Jain et al., 2022; Li et al., 2022; Wang et al., 2022) based on their performance on unit tests and other metrics.

Specifically, LLMs applied to code, e.g. (Chen et al., 2021; Fried et al., 2022), have led to new advances in program synthesis from natural language specification. In this paper, we will be using a Codex (Chen et al., 2021) version code-davinci-002. LLMs and related methods have been used for solving mathematical problems (Lewkowycz et al., 2022) with natural language solutions, for proving theorems in natural language (Welleck et al., 2022), and for proof search, e.g. (Lample et al., 2022).

Autoformalisation. While the term autoformalisation was coined in Wang et al. (2018), the problem itself has a long history; see Wang et al. (2020). Autoformalisation can be thought of as semantic

Parsing for the domain of mathematics. Mathematics is a far larger and sophisticated domain than most domains considered in semantic parsing.

Wang et al. (2020) applied deep learning-based methods to automodality by treating it as a language translation problem. They construct datasets for supervised and unsupervised neural machine translation and evaluate the syntactic distance of the output from the gold output by metrics such as BLEU but do not provide data for correctness. The recent work Wu et al. (2022) is closest to ours and stimulated our work. They considered statement automodality in Isabelle/HOL using LLMs. For their quantitative results, their statements were from middle school to undergrad mathematical competitions (Zheng et al., 2022). These problems use only elementary concepts. Their quantitative studies are for fixed few-shot prompts. While a direct comparison with their results is not possible due to the use of different proof assistants and datasets, our method compares favourably with their method (fixed few-shot prompting with greedy decoding) as shown in the next section. Our input-dependent prompting is not applicable on their dataset due to the lack of availability of aligned data at the elementary level of statements in their datasets. Lean Chat is a fixed-prompt automodality tool for Lean 3 based on Codex.

# 3 AUTOFORMALISING THEOREM STATEMENTS

Here we discuss autoformalisation of theorem statements.

# 3.1 EVALUATION DATASETS

We used three test sets with 40 natural language statements each. The natural language statements were of the same form as typical doctrings in mathlib: single sentences often with Lean code fragments (including names and formulas not in LATEX but in unicode) enclosed in back ticks. We call such strings docstring-style strings.

Our first set consisted of mathematical theorems (some were conjectures as well) in areas well-represented by mathlib, such as undergraduate-level number theory, group theory and topology.

The other two sets were designed to minimize contamination due to similar results being in the training of Codex. Our second set consisted of what we called silly statements, such as every vector space with dimension 2 is finite dimensional. While being true, these were easy and/or absurdly specific, so unlikely to appear in this precise form anywhere else. We created this set by looking at theorems in mathlib and modifying them.

The third set consisted of false statements: these obviously cannot appear in any library. The statements in this set were closely related to those in mathlib or our first dataset: for example, while our first dataset had the statement every field is a ring our third dataset had its (false) converse every ring is a field.

# 3.2 TECHNIQUES

We used Codex to translate an input text in natural language to code in Lean 4. Codex takes as input a prompt and returns completion(s). We generated a prompt from the input text and post-processed completions as described below. Figure 1 is an example of a prompt, the initial result (with one completion shown) and the result after post-processing. We remark that this example needs prompt engineering, as we see in Section 3.3.

Prompt engineering. Given an input text to be translated, we chose example prompts from mathlib whose docstrings are similar to the input text. We used two notions of similarity: proximity in sentence embeddings and keyword matching. This style of prompt design appears in the previous work, e.g., Jain et al. (2022). The docstrings and the corresponding Lean code were extracted from mathlib documentation.

Post processing. Lean 4 code is compiled in two phases: a parser converts a string into a Syntax object, and an elaborator converts a Syntax object into a type-correct expression. The elaboration step is a much stricter analogue of type-checking in a strongly-typed language. It is roughly a formal analogue of supplying all the implicit details in an NL theorem statement. Lean 4 is unique among

Input text: "If a vector space has dimension '2' then it is finite dimensional."  
Prompt:  
```latex
/ $-If$  a vector space has a finite basis, then it is finite-dimensional. \(-/   
\mathrm{theorem}\{\mathrm{K:Type u}\} \{\mathrm{V:Type v}\} [\mathrm{division\_ring~K}][\mathrm{add\_comm\_group~V}][\mathrm{module~K~V}]\{\iota:\mathrm{Type~w}\} [\mathrm{fintype~}\iota ](\mathrm{h:basis}\\ \iota \mathrm{KV}): \mathrm{finite\_dimensional~KV}:=\)   
 $\dots$ $/ - - A$  vector space has dimension at most  $11^{\prime}$  if and only if there is a single vector of which all vectors are multiples. \(-/   
\mathrm{theorem}\{\mathrm{K:Type u}\} \{\mathrm{V:Type v}\} [\mathrm{division\_ring~K}][\mathrm{add\_comm\_group~V}][\mathrm{module~K~V}]:\mathrm{module\_rank~K~V}\leq 1\leftrightarrow \exists (\mathrm{v_0}:\) V),  $\forall (\mathrm{v};\mathrm{v}),\exists (\mathrm{r};\mathrm{K}),\mathrm{r}\cdot \mathrm{v_0} = \mathrm{v}:=$ $/ - - If$  a vector space has dimension  $2^{*}$  then it is finite dimensional. \(-/   
\mathrm{theorem}\)
```

Codex Completion:  
```latex
$\{\mathrm{K:Typeu}\} \{\mathrm{V:Typev}\} [division\_ring\mathrm{K}][add\_comm\_group\mathrm{V}][module\mathrm{KV}](h:\mathrm{module}\mathrm{rank}\mathrm{KV} = 2):$  finite dimensional KV
```

Post-processed code in Lean 4:  
```txt
theorem  $\forall$  {K:Type} {V:Type} [inst:DivisionRing K][inst_1:AddCommGroup V][inst_2:Module KV], Module.rank KV = 2 → FiniteDimensional KV
```

Figure 1: Example of a prompt, the initial result and the result after post-processing. Part of the prompt was elided to save space; full prompt appears in Appendix B.1.

proof assistants in being implemented in Lean 4 and providing an interpreter API, which facilitates our implementation.

We parsed the Codex completions, translated from Lean 3 to Lean 4 and auto-corrected (as described in Section B.2) to obtain Syntax objects corresponding to (syntactically valid) completions. We attempted to elaborate each of these. Thus, restriction to completions which are successfully parsed and elaborated gives a strong filter.

# 3.3 RESULTS

We tested the effects of the prompt engineering and post-processing as well as the final quality of translations for the datasets described in Section 3.1.

Success rates for the Elaborater We begin with quantitative results showing the utility of both prompt engineering and elaboration filtering for the datasets described in Section 3.1.

We summarize the number of statements that were elaborated for each of the three sets of statements in Table 1. For each set, we considered results with 4 fixed prompts (those used by Lean Chat) and 4 prompts chosen by sentence similarity. For each of these cases we considered answers chosen greedily (i.e., temperature 0 and 1 completion) and those obtained by choosing several completions at temperature 0.8 with filtering and selection. We made three runs for each configuration, and the result reported is the median. We also ran a configuration with the Codex recommended default temperature 0.2 and with fixed prompts. The results of this are included in parentheses in the entries for the greedy case. As 11 of the theorem statements were present in mathlib we also ran all the configurations excluding these and obtained similar results as above: in particular 23 of the 29 statements were elaborated with prompt engineering and selection.

We see in the next section that elaboration is a good proxy measure for accuracy. Thus, we can justify the claims made in 1.

The example in Figure 1 illustrates the effect of prompt engineering. None of the 15 completions were elaborated in all the three runs with the fixed (Lean Chat) prompts. The completions often used the wrong name from mathlib or assumed a definition was at a different level of abstraction (e.g., modules versus vector spaces) from that of mathlib. We also saw that a larger number of examples did lead to more sentences being elaborated, but the effect was not strong enough to quantify robustly.

Table 1: Numbers of elaborated statements; numbers in parenthesis are for temperature 0.2 (instead of 0) with one completion  

<table><tr><td></td><td colspan="2">Theorems</td><td colspan="2">Silly Statements</td><td colspan="2">False Statements</td></tr><tr><td></td><td>Fixed</td><td>Input-dependent</td><td>Fixed</td><td>Input-dependent</td><td>Fixed</td><td>Input-dependent</td></tr><tr><td>Greedy</td><td>20 (18)</td><td>21</td><td>19 (21)</td><td>28</td><td>15 (16)</td><td>23</td></tr><tr><td>Filtered</td><td>25</td><td>29</td><td>29</td><td>34</td><td>24</td><td>30</td></tr></table>

Table 2: Correctness of elaborated statements  

<table><tr><td></td><td>false statements</td><td>silly statements</td><td>theorem statements</td></tr><tr><td>Elaborated</td><td>32</td><td>34</td><td>33</td></tr><tr><td>Correct</td><td>21</td><td>26</td><td>30</td></tr><tr><td>Some correct</td><td>28</td><td>32</td><td>30</td></tr><tr><td>All wrong</td><td>4</td><td>2</td><td>3</td></tr></table>

Correctness of elaboration. Next, we analysed how often completions that were successfully elaborated were correct. In the case where more than one completion was elaborated, we considered both whether the chosen completion was correct and whether any of the elaborated completions were correct.

For each of the three sets, we considered a configuration with high temperature and prompt engineering - specifically, we considered the configuration with the highest number of elaborated statements, as our goal was to test elaboration as a proxy measure for correctness. We manually checked the correctness of the selected completion for the elaborated completions, as reported in Table 2.

Further, the statements where all completions were wrong involved some concept for which we had very few prompts available, in part due to the incomplete state of the binary port of mathlib, also suggesting that elaboration is a good proxy measure.

# 4 AUTOFORMALISATION OF PROOFS

As discussed in Sec. 1, this task presents new difficulties on top of automodicalisation of theorem statements. Input-dependent prompting, which was an important ingredient in the previous section, is presently infeasible for proofs due to the lack of aligned data for proofs. Elaboration, another important ingredient for theorem statements, is also infeasible for proofs since it is very rare for the language model to output a completely correct proof. Therefore, instead of aiming for completely correct formalised proofs, we aim for useful formalised proof outputs: those that can be easily repaired to construct a correct formalised proof, saving time and effort compared to formalisation from scratch. With this relaxation, we see that LLMs show promise.

# 4.1 METHODOLOGY

Evaluation dataset. We collected 13 natural language theorems and their proofs from various sources such as ProofWiki, university courses etc., of varying proof technique, domain and difficulty level. We carefully checked if a similar proof is already formalised in Lean (in mathlib or elsewhere on the internet). While in some cases a similar proof does appear, in all cases the structure of our NL proof was significantly different or different formalisms were used (we provide details for each theorem in Appendix D). Since we measure autoformalisation performance according to the faithfulness of the output proof to our NL proof, we believe there is minimal risk that our output were memorized by Codex from its training data. We also used a few hand-written natural language proofs. Some of these are listed below (the full list is in Section C.1)

1. Absolute Value Function is Convex (abs_convex): Let  $f: \mathbb{R} \to \mathbb{R}$  be the absolute value function on the real numbers. Then  $f$  is convex.  
2. Schur's Lemma (schur_lemma): Let  $V$  and  $W$  be vector spaces; and let  $\rho_V$  and  $\rho_W$  be irreducible representations of  $G$  on  $V$  and  $W$  respectively. If  $V$  and  $W$  are not isomorphic, then there are no nontrivial representations of  $G$  on  $V$  and  $W$  respectively.

3. Schur's Inequality (schur_ineq): Let  $x, y, z \in \mathbb{R}_{\geq 0}$  be positive real numbers such that  $x \geq y \geq z \geq 0$ . Let  $t \in \mathbb{R}, t > 0$  be a (strictly) positive real number. Then:  $x^t(x - y)(x - z) + y^t(y - z)(y - x) + z^t(z - x)(z - y) \geq 0$  
4. Contraction Mapping theorem (contractionMapping): Let  $B$  be a Banach space,  $M$  a closed subset of  $B$ , and  $\Phi$  a mapping from  $M$  to  $M$  such that for some  $k \in [0,1)$ ,  $\| \Phi(x) - \Phi(y) \| \leq k \| x - y \|$  for any two points  $x$  and  $y$  in  $M$ . Then there is a unique point  $z$  in  $M$  such that  $\Phi(z) = z$ .

```txt
$--'$  theorem
Power Set is Closed under Intersection
Let  $S$  be a set.
Let  $S$  powerset be the power set of S.
Then:
:  $\backslash$  for all A, B  $\backslash$  in  $\backslash$  powerset S: A  $\backslash$  cap B  $\backslash$  in  $\backslash$  powerset S
'proof'
Let  $A, B \backslash$  in  $\backslash$  powerset S.
Then by the definition of power set,  $A \backslash$  subseteq S and  $B \backslash$  subseteq S.
From Intersection is Subset we have that  $A \backslash$  cap B  $\backslash$  subseteq A.
It follows from Subset Relation is Transitive that  $A \backslash$  cap B  $\backslash$  subseteq S.
Thus  $A \backslash$  cap B  $\backslash$  in  $\backslash$  powerset S and closure is proved.
\{\{qed\}\}
-/
theorem power_set Intersection closed {α: Type*} (S: set α): ∀ A B ∈ P S, (A∩B) ∈ P S := begin
-  $A$  and  $B$  are sets.  $A$  and  $B$  belong to power set of S
assume (A: set α) (hA: A ∈ P S) (B: set α) (hB: B ∈ P S),
- Then SA ⊆ S and  $B \subseteq S$ , by power set definition
have h1: (A ⊆ S) ∧ (B ⊆ S), from by {split, apply set.subset_of_mem_powerset, exact hA, apply
set.subset_of_mem_powerset, exact hB},
- Then $(A \cap B) \subseteq A$, by intersection of set is a subset
have h2: (A∩B) ⊆ A, from by apply set.inter Subset_left,
- Then $(A \cap B) \subseteq S$, by subset relation is transitive
have h3: (A∩B) ⊆ S, from by {apply set.subset.trans h2 h1.left},
- Hence $(A \cap B) \in P S$, by power set definition
show (A∩B) ∈ P S, from by {apply set.mem_powerset h3},
end
```

Figure 2: One of the examples in the prompt for the full-proof-with/comments format

Proof formats. We experimented with several formal proof formats; all formats use forward reasoning (see Appendix A for more on forward vs. backward reasoning). Proof formats differ in their ease of translation by a language model and in their utility to a user. Proof formats vary across two axes: the level of detail and whether the formal proof has comments. The three levels of detail are the following.

Full proof. This corresponds to the complete proof.

Proof outline. This consists of the main steps of the proof listed in order. In Lean code, an outline is given by a series of have statements with sorry as a placeholder for the intermediate proofs. Although an outline contains far less information than a full proof, a tool that is capable of producing good outlines could still be valuable to a user since one could, in principle, iteratively produce outlines of the main proof and all its steps, until one is left with trivial steps that can handled by automation.

Proof outline with premises. This format is at an intermediate level of detail: each proof step is listed along with a list of premises from which it can be deduced. This is done by introducing a fictitious new Lean tactic auto that takes as arguments the list of theorems that go into a proof along with an optional list of Lean tactics that may be helpful.

Proofs at each level of detail can be used as is or combined with comments (each step preceded by a comment explaining that step). This results in a total of six formats. For an example, see the formal proof in Figure 2.

**Prompts.** We designed few-shot prompts (one for each format) for our chosen set of theorems. The prompts consist of three theorems with corresponding proofs; we illustrate one such theorem for full proof with comments in Figure 2 and the full prompt can be found in Section B.1.

Hyperparameters. We initially considered four temperatures 0, 0.2, 0.4 and 0.8. We sampled three outputs for each of the latter three, and a single output for the former.

Evaluation. To generate proofs in different formats, we queried Codex with a prompt consisting of example natural language proofs and the corresponding step-wise Lean proofs in the appropriate format, followed by the natural language proof to be translated. For evaluation, we manually inspected the generated outputs via the following grading scheme.

Theorem statement formalisation: 0 if the output is incorrect; 1 if the output is somewhat correct; 2 if the output is fully correct.

Proof formalization. 0 if the output does not help with formalising the complete proof; 1 if the output slightly decreases the effort needed to formalise the complete proof; 2 if the output makes it substantially easier to formalise the complete proof; 3 if the output only needs few minor corrections; 4 if the output is fully correct.

As manual grading of proof output by Codex is time-consuming, after a preliminary analysis we focused on three formats, namely those with comments, and on temperatures 0.4 and 0.8, as the results were better in these cases. Outputting proof formats with comments might help Codex relate the natural language proof with the formalised Lean proof at a more granular level. Hence, for each theorem, we analysed 18 Codex completions (6 per proof format). The model was initially given the task of formalising the theorem statement as well as the proof. Later, we also prompted the model with correct Lean statements for some theorems and assigned it the task of the formalisation of the respective proofs.

# 4.2 RESULTS

Overall, we found that the generated proofs were well aligned with the natural language proofs and also well-structured as per Lean style. These proofs could therefore be used as a good starting points for formalisation assistance as illustrated in Figure 3. We summarise the scores given to the Codex completions after manual inspection in Figure 6. No completion received a perfect score of 4. For 8 of the 13 theorems, at least one Codex completion (out of 18), was marked 3. The other 5 theorems got a maximum of 2. The main sources of lower scores were errors related to incorrect natural language statement translations that could be mathematically incorrect, irrelevant or invalid Lean code. Some lower scores were also due to step repetitions. These results are given and errors analysed in detail in Section C; here we present a synopsis.

The best proof format depended on the nature of the proof, with more detailed formats often better (i.e., with more intermediate details) for harder to formalise proofs, while the results were good for all formats for easy to formalise statements. Including the correct Lean theorem statements did not show a clear improvement. However, in the case of the lowest scoring theorem without the correct statements, including the correct statement improved the score from 1 to 2.

Capabilities shown by Codex. In schur_ineq, the natural language proof had statements that simply mentioned that all the terms are non-negative and concluded the proof. Interestingly, Codex completions had these details formalised as intermediate steps.

Codex sometimes expanded a definition instead of using the mathlib definition directly; for example,  $A = A^T$  instead of is_symm A. Codex also generated plausible theorem names, which were in line with mathlib style.

Errors in completions. Occasionally the completion had invalid Lean syntax, e.g.,  $x \geq y \geq z \geq 0$  (copied directly from the natural proof) instead of the valid Lean syntax  $x \geq y \land y \geq z \land z \geq 0$ . There was an instance where Codex generated a proof in what seemed like a different language. In some cases, we observed that the completion used an undeclared variable in the proof. For example, it declared  $t > 0$  without introducing  $t$ :  $\mathbb{R}$ .

There were a few instances where proof-steps were syntactically correct but mathematically incorrect, for instance stating  $|\alpha \star \mathrm{x} + \beta \star \mathrm{y}| = \alpha \star |\mathrm{x}| + \beta \star |\mathrm{y}|$  instead of the triangle inequality.

Sometimes natural language translations were wrong, although they were mathematically valid statements and valid Lean code. For example, in schur_lemma, Codex confuses a homomorphism being "non-zero" with a homomorphism being "nowhere zero". In a proof of the

![](images/ea0873ec239ae23d80df193483e1d1d19ec82d9372115e12e95bfef83f56b38f.jpg)  
Figure 3: Correction of a Codex completion of Absolute Value Function is Convex. The text highlighted in red is to be deleted and the text highlighted in green and underlined is to be added.

contraction Mapping, Codex defines a sequence  $x$  to be  $x(i) := \phi(x(i))$ , instead of the inductive definition:  $x(0) := x_0$  and  $x(i + 1) := \phi(x(i))$ .

Sometimes the Codex completion contained contradicting proof steps (even in cases not involving proof of contradiction). As an example, in a generated proof of schur_lemma, there were two statements, one stating that the kernel of a homomorphism is trivial, and the next one stating the opposite.

Hallucinations. Codex sometimes output names that looked realistic but are not present in mathlib, e.g. convex_function instead of convex_on.

Deviation from Natural Language Proof. Occasionally the output proof formats deviated from the natural language proofs, with Codex merging different proofs from distinct sources, leading to conflicting notation.

# 5 LIMITATIONS AND FUTURE WORK

With our techniques, Codex shows promising performance for automodification of docstring-style theorem statements and for proofs There are many avenues for future work.

Using docstrings from mathlib in the present form does not give adequate examples of complex Latex formulas and of some mathematical idioms. An additional database of prompts targeting these could address this. Further, we can make use of Lean's easily extensible syntax to incorporate more mathematical notation. One way to improve selection is to reverse the translation to obtain NL text from Lean code and use a similarity measure with the original text to select the best completion. While preliminary experiments show this is useful, presently it is too slow to be practical.

Better equality testing for theorem statements will also result in better filtering. Unlike program synthesis, for theorem autoformalisation, there is no obvious counterpart of unit tests. Better equality testing with the correct Lean formal statement, however, can serve the role of unit tests.

Outputs generated by our framework can be a useful starting point for formalisation, potentially saving considerable time and effort. Presently about one or two out of up to 18 completions tend to be useful; recognising these automatically will reduce effort. We did not experiment with interactive formalisation as evaluation becomes harder. It would be interesting to combine our framework with automatic proof search or repair ideas: partial proofs, being close to complete proofs, can serve as a good starting point for proof search. This could result in an autoformalisation system that is closer to being autonomous.

# REFERENCES

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc'Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/bit/1457c0d6bcbd4967418bf8ac142f64a-AAbstract.html.  
Kevin Buzzard. What is the point of computers? a question for pure mathematicians. In International Congress of Mathematicians, 2022. URL https://arxiv.org/pdf/2112.11598.pdf.  
Ricardo Campos, Vitor Mangaravite, Arian Pasquali, Alipio Mário Jorge, Célia Nunes, and Adam Jatowt. A text feature based automatic keyword extraction method for single documents. In European conference on information retrieval, pp. 684-691. Springer, 2018.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidi Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.  
Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In André Platzer and Geoff Sutcliffe (eds.), Automated Deduction - CADE 28 - 28th International Conference on Automated Deduction, Virtual Event, July 12-15, 2021, Proceedings, volume 12699 of Lecture Notes in Computer Science, pp. 625-635. Springer, 2021. doi: 10.1007/978-3-030-79876-5\37. URL https://doi.org/10.1007/978-3-030-79876-5_37.  
Leonardo Mendonça de Moura, Soonho Kong, Jeremy Avigad, Floris van Doorn, and Jakob von Raumer. The lean theorem prover (system description). In Amy P. Felty and Aart Middeldorp (eds.), Automated Deduction - CADE-25 - 25th International Conference on Automated Deduction, Berlin, Germany, August 1-7, 2015, Proceedings, volume 9195 of Lecture Notes in Computer Science, pp. 378-388. Springer, 2015. doi: 10.1007/978-3-319-21401-6\26. URL https://doi.org/10.1007/978-3-319-21401-6\26.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2018. URL https://arxiv.org/abs/1810.04805.  
Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. Incoder: A generative model for code infilling and synthesis, 2022. URL https://arxiv.org/abs/2204.05999.  
Mohan Ganesalingam. The Language of Mathematics - A Linguistic and Philosophical Investigation, volume 7805 of Lecture Notes in Computer Science. Springer, 2013. ISBN 978-3-642-37011-3. doi: 10.1007/978-3-642-37012-0. URL https://doi.org/10.1007/978-3-642-37012-0.

Christopher Hahn, Frederik Schmitt, Julia J Tillman, Niklas Metzger, Julian Siber, and Bernd Finkbeiner. Formal specifications from natural language. arXiv preprint arXiv:2206.01962, 2022.  
Naman Jain, Skanda Vaidyanath, Arun Iyer, Nagarajan Natarajan, Suresh Parthasarathy, Sriram Rajamani, and Rahul Sharma. Jigsaw: Large language models meet program synthesis. In Proceedings of the 44th International Conference on Software Engineering, ICSE '22, pp. 1219-1231, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392211. doi: 10.1145/3510003.3510203. URL https://doi.org/10.1145/3510003.3510203.  
Aishwarya Kamath and Rajarshi Das. A survey on semantic parsing. In 1st Conference on Automated Knowledge Base Construction, AKBC 2019, Amherst, MA, USA, May 20-22, 2019, 2019. doi: 10.24432/C5WC7D. URL https://doi.org/10.24432/C5WC7D.  
Guillaume Lample, Marie-Anne Lachaux, Thibaut Lavril, Xavier Martinet, Amaury Hayat, Gabriel Ebner, Aurélien Rodriguez, and Timothée Lacroix. Hypertree proof search for neural theorem proving. CoRR, abs/2205.11491, 2022. doi: 10.48550/arXiv.2205.11491. URL https://doi.org/10.48550/arXiv.2205.11491.  
Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay V. Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. NeurIPS 2022, abs/2206.14858, 2022. doi: 10.48550/arXiv.2206.14858. URL https://doi.org/10.48550/arXiv.2206.14858.  
Yujia Li, David H. Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphanumeric. CoRR, abs/2203.07814, 2022. doi: 10.48550/arXiv.2203.07814. URL https://doi.org/10.48550/arXiv.2203.07814.  
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing, 2021. URL https://arxiv.org/abs/2107.13586.  
Patrick Massot. Why formalize mathematics. 2021. URL https://www.imo.universite-paris-saclay.fr/~pmassot/files/exposition/why_formalize.pdf.  
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL http://arxiv.org/abs/1908.10084.  
Christian Szegedy. A promising path towards automatization and general artificial intelligence. In Christoph Benzmüller and Bruce R. Miller (eds.), Intelligent Computer Mathematics - 13th International Conference, CICM 2020, Bertinoro, Italy, July 26-31, 2020, Proceedings, volume 12236 of Lecture Notes in Computer Science, pp. 3-20. Springer, 2020. doi: 10.1007/978-3-030-53518-6\_.1. URL https://doi.org/10.1007/978-3-030-53518-6_1.  
Qingxiang Wang, Cezary Kaliszyk, and Josef Urban. First experiments with neural translation of informal to formal mathematics. In Florian Rabe, William M. Farmer, Grant O. Passmore, and Abdou Youssef (eds.), Intelligent Computer Mathematics - 11th International Conference, CICM 2018, Hagenberg, Austria, August 13-17, 2018, Proceedings, volume 11006 of Lecture Notes in Computer Science, pp. 255-270. Springer, 2018. doi: 10.1007/978-3-319-96812-4\22. URL https://doi.org/10.1007/978-3-319-96812-4_22.  
Qingxiang Wang, Chad Brown, Cezary Kaliszyk, and Josef Urban. Exploration of neural machine translation in isoformalization of mathematics in mizar. In International Conference on Certified Programs and Proofs, 2020.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2022. URL https://arxiv.org/abs/2203.11171.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. NeurIPS 2022, abs/2201.11903, 2022. URL https://arxiv.org/abs/2201.11903.  
Sean Welleck, Jiacheng Liu, Ximing Lu, Hannaneh Hajishirzi, and Yejin Choi. Naturalprover: Grounded mathematical proof generation with language models. NeurIPS 2022, abs/2205.12910, 2022. doi: 10.48550/arXiv.2205.12910. URL https://doi.org/10.48550/arXiv.2205.12910.  
Yuhuai Wu, Albert Q. Jiang, Wenda Li, Markus N. Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with large language models. NeurIPS 2022, abs/2205.12615, 2022. doi: 10.48550/arXiv.2205.12615. URL https://doi.org/10.48550/arXiv.2205.12615.  
Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=9ZPegFuFTFv.
