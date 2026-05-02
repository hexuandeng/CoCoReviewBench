# OPTIMAL DATA SAMPLING FOR TRAINING NEURAL SURROGATES OF PROGRAMS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Programmers and researchers are increasingly developing surrogates of programs, models of a subset of the observable behavior of a given program, to solve a variety of software development challenges. Programmers train surrogates from measurements of the behavior of a program on a dataset of input examples. We present a methodology for optimally sampling datasets to train neural network based surrogates of programs. We first characterize the optimal proportion of data to sample from each path in a program based on the complexity of learning the path. We next provide a program analysis to determine the complexity of different paths in a program. We evaluate these results on a large-scale graphics program, demonstrating that theoretically optimal sampling results in empirical improvements in accuracy.

# 1 INTRODUCTION

Programmers and researchers are increasingly developing surrogates of programs, models of a subset of the observable behavior of a given program, to solve a variety of software development challenges (Renda et al., 2021). For example, Esmaeilzadeh et al. (2012) train small neural networks to mimic existing programs, then deploy the neural networks in place of the programs to speed up computation. Generally, surrogates are used to accelerate programs (Esmaeilzadeh et al., 2012; Mendis et al., 2019; Munk et al., 2019), apply transfer learning to programs (Tercan et al., 2018; Kustowski et al., 2020; Kwon & Carloni, 2020), and approximate the gradient of programs to optimize their inputs (Renda et al., 2020; She et al., 2019; Tseng et al., 2019).

Dataset generation. Training a surrogate of a program requires measurements of the behavior of the program on a dataset of input examples. There are three common approaches to collecting this dataset. The first is to use data that is uniformly sampled (or sampled using another manually defined distribution) from the input space of the program (Tseng et al., 2019; Kustowski et al., 2020). The second is to use data instrumented from running the original program on a workload of interest (Renda et al., 2020; Esmaeilzadeh et al., 2012). The third is to use active learning (Settles, 2009), a class of online methods that iteratively query labels for data points based on the expected improvement in accuracy resulting from additional samples (Ipek et al., 2006; She et al., 2019; Pestourie et al., 2020).

These approaches show promise, but they face challenges with programs with control flow. Programs with control flow (e.g. branches and loops) are piecewise functions: each control flow path induces a different trace of operations that are applied to the input. The sampling techniques above do not optimally allocate samples between different paths, resulting in surrogates which do not adequately learn the behavior of the program along all paths. For example, Renda et al. (2020, Section IV.A) identify a scenario in which an instrumented dataset does not exercise a set of paths in the program enough times for the surrogate to learn the behavior along those paths.

Our approach. Our approach uses the source code and semantics of the program under study to guide dataset generation for training a surrogate of the program. The core concept is to analyze the complexity of each path in a program and to allocate more samples to paths that are more complex to learn.

Stratified functions. Our approach represents the program as a stratified function, a function with different behavior in different regions (strata) of the input space (i.e., a piecewise function).<sup>1</sup> We use

stratified surrogates to model such functions. To construct a stratified surrogate, we train independent surrogates of each component of the stratified function. At evaluation time, a stratified surrogate checks which stratum an input is in (using the original program) then applies the corresponding surrogate.

This evaluation-time stratum check must not preclude the use of the surrogate for its downstream task. We therefore adopt a standard modeling assumption in the approximate computing literature: that precisely determining paths is an acceptable cost during approximate program execution (Sampson et al., 2011; Carbin et al., 2013).<sup>2,3</sup>

Optimal sampling. With this stratified modeling assumption, we then determine how many samples to allocate to train each surrogate. Using neural network sample complexity bounds for learning analytic functions (Arora et al., 2019; Agarwala et al., 2021) we calculate a complexity for each component function which gives an upper bound on how many samples are required to learn the behavior of that component to a given error. Given a data distribution describing the frequency of each component and given each component function's complexity, we then derive the optimal number of samples to allocate to training each surrogate of each component, minimizing the upper bound on the stratified surrogate's error.

Complexity analysis. We present a programming language, TURACO, in which programs denote stratified functions with well-defined complexity measures. We provide a program analysis for TURACO programs that automatically determines the strata of the function and calculates an upper bound on the complexity of each component of the stratified function that the program denotes.

**Renderer demonstration.** To demonstrate that optimal sampling using our complexity analysis improves surrogate accuracy on downstream tasks, we present a case study of learning a surrogate of a renderer in a video game engine. We show that our optimal sampling approach results in between  $15\%$  and  $47\%$  lower error than training using distributions that do not take into account path complexity. These accuracy improvements correlate with perceptual improvements in the generated renders.

Contributions. In sum, we present the following contributions:

- An optimal approach to allocating samples among strata to train stratified neural network surrogates of stratified analytic functions.  
- A programming language, TURACO, in which all programs are learnable stratified functions, and a program analysis to determine the complexity of learning surrogates of those programs.  
- An evaluation of these results on a graphics program, demonstrating that theoretically optimal sampling using TURACO's complexity analysis results in empirical improvements in accuracy.

We lay the groundwork for analyzing optimal sampling approaches for training surrogates of programs. Our results hold out the promise of surrogate training approaches that intelligently use the program's semantics to guide the design and training of surrogates of programs.

# 2 EXAMPLE

Figure 1a presents an example distilled from our evaluation (Section 5) that we use to demonstrate how optimal path sampling, sampling from paths according both to their frequency in a data distribution and to their complexity, results in a more accurate surrogate than frequency-based path sampling, sampling according to the frequency of paths alone.

Program under study. We study a graphics program that calculates the luminance (i.e., brightness) at a point in a scene as a function of sunPosition, the height of the sun in the sky (i.e., the time of day) which ranges from  $-1$  to  $1$ , and emission, a property of the material at that point which ranges from  $-1$  to  $1$ . The program first checks whether it is daytime (Line 2), and sets the ambient lighting variable accordingly. The program next checks whether the sun position is above a threshold (Line 7) and sets the emission variable accordingly. The output is then the sum of the ambient light and the light emitted by the material. Figure 1b presents the output of this program on inputs between  $-1$  and  $1$ .

```txt
1 fun (sunPosition, emission) {
2 if (sunPosition < 0) {
3 ambient = 0
4 } else {
5 ambient = sunPosition
6 }
7 emission *= max(0.1, sunPosition)
8 } return ambient + emission
```

![](images/6f4b3c3cdf5123ce9cd574f9bb425ee1f6cb13e132ceb0473300538a0b41ebc2.jpg)  
Figure 1: Example program, outputs, and traces.

(a) Graphics program calculating the luminance of a pixel as a function of ambient light and material properties.

(b) Output of the program on inputs in  $[-1,1]$ , with dashes separating the three paths.

```txt
// assume: sunPosition < 0  
fun (sunPosition, emission) {  
    ambient = 0;  
    emission *= 0.1;  
} return ambient + emission;
```

(c) Nighttime (11) path.

```javascript
//assume  $0 <   \mathrm{sunPosition} <   0.1$  fun（sunPosition，emission）{ ambient  $=$  sunPosition; emission  $\ast = 0.1$  · }return ambient  $^+$  emission;
```

(d) Twilight (x1) path.

```javascript
//assume:sunPosition  $>0.1$  fun（sunPosition，emission）{ ambient  $=$  sunPosition; emission  $\ast =$  sunPosition; }return ambient  $^+$  emission;
```

(e) Daytime (rr) path.

The path conditions (Lines 2 and 7) partition the program into three traces: nighttime, when sunPosition is less than 0 (Figure 1c); twilight, when sunPosition is between 0 and 0.1 (Figure 1d); and daytime, when sunPosition is greater than 0.1 (Figure 1e). These paths are separated by dashed black lines in Figure 1b.

Training a surrogate of this program poses a particular challenge because of the different behavior of these traces. Furthermore, these traces have different relative complexities: when sunPosition is less than 0.1 the function is linear, but when sunPosition is above 0.1 the function is quadratic.

We must ensure that the data distribution that we use for training surrogates reflects not only the different paths of the program, but also the relative complexities of each path of the program.

Optimal path sampling. We present an approach to determining the optimal amount of training data to sample from each path to train a stratified surrogate of this program. Specifically, given a data distribution and a data budget we want to find the optimal number of data points to sample from each path to minimize the expected error of a surrogate of the program over the data distribution. Intuitively, our approach is to prioritize sampling paths that are frequent and paths that are complex (and thus require more samples to learn).

First we determine the frequency of each path. We assume that the data has a uniform distribution over inputs between -1 and 1. This results in path frequencies for the nighttime path (sunPosition < 0) of  $50\%$ , the twilight path  $(0 < \text{sunPosition} < 0.1)$  of  $10\%$ , and the daytime path  $(0.1 < \text{sunPosition})$  of  $40\%$ .

Next we determine the sample complexity of each path, the number of samples required to learn the function along that path to a given error. We use the sample complexity results of Agarwala et al. (2021), who give an upper bound on the number of samples required to learn a neural network approximation of a given function. Using this bound (as implemented by our TURACO analysis described in Section 4.2), we determine that the twilight path takes  $1.5 \times$  as many samples to train a surrogate to a given error as the nighttime path, and the daytime path requires  $5 \times$  as many samples.

Finally, we combine these metrics to determine the optimal sampling rates. Using Equation (3) in Section 3.2, we find that the optimal sampling rate is to sample  $39.3\%$  of the data from the nighttime path,  $14.3\%$  of the data from the twilight path, and  $46.4\%$  of the data from the daytime path.

Stratified surrogates. The class of surrogate model for which the above approach is optimal is that of a stratified neural surrogate – a set of disjoint neural networks which are applied based on which

![](images/8a3b2b586b5f37c5d40778ecede9f986749bd13c4ecd05b53671f7e020bacd85.jpg)  
(a) Per-path surrogate errors (log-log plot).

![](images/e276f0cde4d098d911b11de75298808c83511ee6ccf1f81588f9a019ed23ee58.jpg)  
Figure 2: Per-path surrogate errors (left) and combined errors (right) for the example.  
(b) Stratified surrogate errors (log-log plot). Optimal sampling decreases the error by  $15\%$ .

path the inputs induce in the program. Concretely, this means that we train one surrogate per path, and pick which to apply for each input at evaluation time. For this example program, picking which surrogate to apply just requires comparing sunPosition against constant threshold values.

Training methodology. For each surrogate, we train a 1-hidden-layer MLP with 512 hidden units with a ReLU activation, using 10,000 steps of Adam with learning rate 0.0005 and batch size 128.

Results. Figure 2 presents the error of surrogates trained to mimic this program, as a function of the training dataset size. On the left, Figure 2a presents the error of surrogates of each path. On the right, Figure 2b presents the error of stratified surrogates of the entire program for optimal path sampling and a baseline of sampling according to path frequency alone. The x axis of each plot is the dataset size used to train the surrogate (on the left, the dataset size per path; on the right, the total dataset size used for all paths). The y axis of each plot is the error of the resulting surrogate (lower is better).

Figure 2a shows that though the complexity measure is not exactly proportional to the empirical sample complexity, it does correlate with it: at a given data budget, the daytime path with the highest complexity has the highest error, followed by twilight then nighttime.

Figure 2b shows that the optimal path sampling approach results in lower error than sampling according to path frequency alone. For datasets of total size below 70 samples, the surrogate trained with optimal path sampling has a geometric mean decrease in error of  $27.5\%$ . For datasets of total size above 70 samples, the surrogate trained with optimal path sampling has a geometric mean decrease in error of  $5.5\%$ . Across the entire range of dataset sizes evaluated in this plot, the surrogate trained with optimal path sampling has a geometric mean decrease in error of  $15\%$ .

# 3 OPTIMAL SAMPLING

In this section, we formally define stratified functions and stratified surrogates, and derive the optimal sampling distribution to use when training a stratified surrogate of stratified function.

# 3.1 SETUP

We define a learning algorithm, a function that trains a surrogate of a given input function, as a random function  $tr: (\mathcal{X} \to \mathcal{Y}) \times \mathcal{D} \times \mathbb{N} \times \mathcal{L} \to (\mathcal{X} \to \mathcal{Y})$  that takes a function  $f: \mathcal{X} \to \mathcal{Y}$  from inputs  $x \in \mathcal{X}$  to outputs  $y \in \mathcal{Y}$ , a distribution  $D \in \mathcal{D}$  over inputs  $x$ , a number of training examples  $n \in \mathbb{N}$ , and a loss function  $\ell \in \mathcal{L}: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$  which measures the cost of an incorrect prediction, and returns a function (representing the output surrogate)  $\hat{f}: \mathcal{X} \to \mathcal{Y}$ .

A given function  $f$  is probably approximately correctly learnable (abbreviated as learnable for the remainder of the paper) for a given learning algorithm  $tr$  and loss function  $\ell$  if for all distributions  $D$ , with high probability  $1 - \delta$  the learning algorithm returns a surrogate  $\hat{f}$  that approximately matches

the original function  $f$  over the distribution  $D$ :

$$
\forall D, \epsilon \in (0, 1), \delta \in (0, 1). \exists n. \underset {\hat {f} \sim t r (f, D, n, \ell)} {\mathrm {P}} \left(\underset {x \sim D} {\mathbb {E}} \left[ \ell \left(\hat {f} (x), f (x)\right) \right] \leq \epsilon\right) \geq 1 - \delta \tag {1}
$$

Following Arora et al. (2019) and Agarwala et al. (2021) we study functions and learning algorithms for which there is a measure of the complexity  $\zeta(f)$  of function  $f$  for the learning algorithm  $tr$  such that the relationship between  $n$ ,  $\zeta(f)$ ,  $\epsilon$ , and  $\delta$  in Equation (1) is:

$$
\exists C. n \leq C \left[ \frac {\zeta (f) + \log (\delta^ {- 1})}{\epsilon^ {2}} \right] \tag {2}
$$

We instantiate the complexity measure  $\zeta(f)$  for neural networks in Section 3.3.

We define a stratified function  $f$  as follows:

$$
f (x) \triangleq \left\{ \begin{array}{l l} f _ {1} (x) & \text {i f} x \in s _ {1} \\ \vdots \\ f _ {c} (x) & \text {i f} x \in s _ {c} \end{array} \right.
$$

where  $c$  is the number of strata,  $\{s_i\}_{i = 1}^c$  are strata,  $\forall i\neq j.s_{i}\cap s_{j} = \emptyset$  , and  $\cup_{i}s_{i} = \mathcal{X}$

We define a stratified surrogate  $\hat{f}$  as a stratified function with components  $\hat{f}_i$ .

For a data distribution  $D$ , let  $D(x)$  be the probability that  $x$  is sampled from  $D$ , and  $D(s_i)$  be the total probability mass of all data points within  $s_i$  over  $D$  (i.e.,  $D(s_i) = \int_{x \in s_i} D(x)$ ). Let  $D(x|s_i)$  be the probability of a data point  $x$  sampled from  $D$  if  $x \in s_i$ .

# 3.2 OPTIMAL SAMPLING

Our goal is to learn a stratified surrogate  $\hat{f}$  of a stratified function  $f$ , where each component function  $f_{i}$  is learnable. We are given a data distribution  $D$ , a maximum sample budget  $n$ , a learning algorithm  $tr$ , a loss function  $\ell$ , and a failure probability  $\delta$ . Our task is therefore to find the number of samples to allocate to each stratum to train a surrogate of that function component. We assume that each surrogate component's failure probability  $\delta_{i} = \frac{\delta}{c}$ , which satisfies the overall failure probability by union bound.

Note that Equation (2) is an upper bound and not an exact equality. We are therefore minimizing the upper bound of the error of the resulting surrogate, rather than directly minimizing the error.

Formally, we solve the following optimization problem:

$$
\underset {\left\{n _ {i} \right\} _ {i = 1} ^ {c} \in \mathbb {N} ^ {+} x \sim D} {\operatorname {a r g m i n}} \mathbb {E} \left[ \ell \left(\hat {f} (x), f (x)\right) \right] \text {s . t .} \sum_ {i = 1} ^ {c} n _ {i} \leq n \text {a n d} \hat {f} _ {i} \sim t r \left(f _ {i}, D \left(x \mid s _ {i}\right), n _ {i}, \ell\right) \tag {3}
$$

Theorem 3.1. With  $\delta_{i} = \frac{\delta}{c}$ , the upper bound of Equation (3) is minimized at:

$$
n _ {i} = n \frac {\left(D (s _ {i}) \sqrt {\zeta (f _ {i}) + \log (c \delta^ {- 1})}\right) ^ {\frac {2}{3}}}{\sum_ {i = 1} ^ {c} \left(D (s _ {i}) \sqrt {\zeta (f _ {i}) + \log (c \delta^ {- 1})}\right) ^ {\frac {2}{3}}},
$$

The proof of this theorem is presented in Appendix C.

# 3.3 NEURAL NETWORK LEARNABILITY

Equation (2) defines the required sample complexity for learning  $\hat{f}$  as a function of  $\zeta(f)$ , the complexity of  $f$ . This section defines  $\zeta(f)$  for training neural network surrogates of analytic functions. This section is an abridged summary of assumptions and results presented by Agarwala et al. (2021) and Arora et al. (2019); refer to Agarwala et al. (2021) for the full set of assumptions.

Agarwala et al. (2021) provide a calculus for learning surrogates of analytic functions  $f$  (around 0) based on the tilde  $\tilde{f}$  of the function:

$$
f (x) = \sum_ {n = 0} ^ {\infty} a _ {n} x ^ {n} \quad \tilde {f} (x) \triangleq \sum_ {n = 0} ^ {\infty} | a _ {n} | x ^ {n}
$$

Note the following properties for  $x \geq 0$ :

$$
\tilde {h} (x) \leq \left\{ \begin{array}{l l} \tilde {f} (x) + \tilde {g} (x) & \text {i f} h (x) = f (x) + g (x) \\ \tilde {f} (x) \cdot \tilde {g} (x) & \text {i f} h (x) = f (x) \cdot g (x) \\ \tilde {f} (\tilde {g} (x)) & \text {i f} h (x) = f (g (x)) \text {a n d} \tilde {f} \text {c o n v e r g e s f o r} \tilde {g} (x) \end{array} \right. \tag {4}
$$

For a 2-layer neural network trained with stochastic gradient descent, if  $f$  is analytic,  $\vec{x}$  is on the  $d$ -dimensional unit sphere  $(\vec{x} \in S^d)$ ,  $\beta \in \mathbb{R}^d$  (a parameter set to control the scale of the inputs), and  $\ell$  is 1-Lipschitz, then  $f(\beta \cdot \vec{x})$  is learnable with:

$$
\zeta (f) = \left(\| \beta \| _ {2} \tilde {f} ^ {\prime} (\| \beta \| _ {2}) + \tilde {f} (0)\right) ^ {2} \tag {5}
$$

# 4 TURACO: PROGRAMS AS STRATIFIED FUNCTIONS

In this section we present TURACO, a programming language in which all programs denote learnable stratified functions. We provide a program analysis for TURACO programs which calculates an upper bound on the complexity of each component of the stratified functions that the program denotes.

# 4.1 SYNTAX AND STANDARD INTERPRETATION

$$
p := \operatorname {f u n} (x +) \left\{s \right\} \text {r e t u r n} x
$$

$$
s: := \operatorname {s k i p} | s; s | x := e
$$

$$
| \text {i f} (e > 0) \{s \} \text {e l s e} \{s \}
$$

$$
e := x \mid v \mid b (e, e) \mid u (e)
$$

$$
b: := \mathrm {A D D} | \mathrm {M U L}
$$

$$
u: := \text {N E G} | \text {S I N} | \text {E X P} | \text {L O G 1 P}
$$

$$
x := \text {s e t}
$$

$$
v := \text {s e t}
$$

Figure 3: Syntax of TURACO.

Figure 3 presents the syntax of TURACO, a loop-free IMP-like language (Winskel, 1993). A TURACO program  $p$  takes a list of inputs  $x$ , executes a top-level statement  $s$ , and returns a single variable  $x$ . Statements  $s$  are skips, sequences, assignments, or if statements. Expressions  $e$  are variables  $x$ , values  $v$ , binary operations  $b$ , or unary operations  $u$ .

TURACO supports analytic operations (e.g., NEG, SIN, EXP), including those which can be represented by a power series within a given domain: LOG1P computes  $\log(1 + x)$  for  $x \in (-1, 1]$ . We restrict the supported operations to those required to implement the case study in Section 5.

Appendix D.1 presents the full set of semantics for TURACO.

# 4.2 COMPLEXITY ANALYSIS

We now present a program analysis that gives an upper bound on the complexity of traces of TURACO programs. This section presents a core set of rules; Appendix D.2 presents the full analysis.

The analysis uses three core concepts: a complexity interpretation of expressions to calculate an upper bound on the tilde of expressions based on the calculus in Equation (4), a dual-number execution (Wengert, 1964; Griewank & Walther, 2008) to calculate the derivative of the upper bound on the tilde (to compute the first term in Equation (5)), and a path analysis which splits the program by paths to compute the complexity of each trace.

Figure 4 presents the big-step relations used to calculate an upper bound on the complexity for a subset of expressions. The relation  $\langle \tilde{\sigma}, [e] \rangle \tilde{\Downarrow}(\tilde{v}, \tilde{v}')$  says that under the variable complexity mapping  $\tilde{\sigma}$  (mapping variables to tuples with their respective tilde and tilde derivative), the expression  $e$  has  $\tilde{e} \leq \tilde{v}$  and  $\tilde{e}' \leq \tilde{v}'$ . Note that the rule for MUL uses the upper bound for multiplication in Equation (4).

Figure 5 presents the complexity relation for if statements. For statements,  $\langle \tilde{\sigma}, s \rangle \Downarrow \tilde{\Sigma}$  says that under the variable complexity mapping  $\tilde{\sigma}$ , the statement  $s$  results in a set of paths with complexity mappings  $\tilde{\Sigma}(p)$  for path  $p$ . We use a period to denote string concatenation (e.g., 1.p to pretend  $p$  with 1).

![](images/09906d4c3ed10c292ee33b4eac80c32c8eb34b2d876972d6d597c0ff110dd95f.jpg)  
Figure 4: Complexity relation for expressions in TURACO.

$$
\frac {\langle \tilde {\sigma} , s _ {1} \rangle \tilde {\Downarrow} \tilde {\Sigma} _ {l} \qquad \langle \tilde {\sigma} , s _ {2} \rangle \tilde {\Downarrow} \tilde {\Sigma} _ {r}}{\langle \tilde {\sigma} , \text {i f} (e > 0) \{s _ {1} \} \text {e l s e} \{s _ {2} \} \rangle \tilde {\Downarrow} \left\{1 . p \mapsto \tilde {\Sigma} _ {l} (p) | p \in \tilde {\Sigma} _ {l} \right\} \cup \left\{r . p \mapsto \tilde {\Sigma} _ {r} (p) | p \in \tilde {\Sigma} _ {r} \right\}}
$$

Figure 5: Complexity relation for if statements, using a period to denote string concatenation.

To calculate the complexity  $\zeta$  as defined in Section 3.3 (for a given input  $\beta$  parameter, which represents the scale of the input data) of each path of a program, we use the statement relation to calculate  $\tilde{f}'(\beta)$  and  $\tilde{f}(0)$ . Appendix D.3 presents this rule, along with the theorem that the complexity calculated by this analysis is an upper bound on the complexity as defined in Section 3.3.

# 5 RENDERER DEMONSTRATION

In this section we present a case study of our optimal sampling results and complexity analysis. The program under study is a demonstration 3D renderer (Lettier, 2019), such as forms the core of a graphics rendering pipeline for a movie or 3D game engine (Christensen et al., 2018; Tatarchuk, 2006). Figures 6a and 6b show scenes that the renderer generates. We demonstrate that the sampling and analysis techniques in Sections 3 and 4 consistently result in more accurate surrogates than those trained using baseline distributions (the frequency distribution of paths and the uniform distribution).

Compared to training surrogates on the frequency distribution of paths, optimal path sampling decreases error by  $15\%$ . Compared to training on the uniform distribution of paths, optimal path sampling decreases error by  $47\%$ . These improvements in error correspond to perceptual improvements in the generated images, as shown in Figures 6c to 6e.

# 5.1 PROGRAM UNDER STUDY

The full renderer program is a 2750 lines-of-code  $\mathrm{C} + +$  program, which invokes 38 different GLSL Shader programs totaling 2446 lines of code. We learn a surrogate of a section of one core shader, totaling 60 lines of code. Figure 18 in Appendix E presents the code under study.

Input-output specification. This program is a Shader which assigns colors to pixels in the image based on the scene geometry, materials, lights, and other properties. The program is called for each pixel that is rendered in the image. Each invocation the of program takes as input a set of 11 fixed-size vectors, totaling 35 inputs. The program returns as output a set of 2 fixed-size vectors, totaling 8 outputs. These outputs are two RGBA colors, the first representing the base color of the pixel, and the second representing the color and intensity of a specular map at that pixel.

Scenes and datasets. We evaluate the renderer on four different scenes, which we combine into nine different datasets. Figures 6a and 6b present two of the four different scenes under consideration; the four scenes are all combinations of views from the front and top, during the day and night. We combine these scenes into nine datasets: a dataset with each scene, a dataset combining each scene from each angle (front day and front night, top day and top night), a dataset combining each scene from each time of day, and a dataset combining all scenes. Figure 14 in Appendix E presents the full set of scenes under study.

![](images/f820deb607d9bb944520b2c1bd47f0ff27f38535e0da27f02949acb4e317be08.jpg)

![](images/c55cc50b4fdaa67b2883afc311e1d2a4d660a52f953c805b9cd85492883dbdf1.jpg)

![](images/88f0a2d2eb1d2aea3f76278cc48a72c12aefbffc1cf77d17194198b716914b4f.jpg)  
(a) Ground-truth front-day scene.

![](images/9cadb07cfee363e14c8a48c7e45f8ccb36f3ca0ba455f6b02a50445c23da0c81.jpg)  
(c) Optimal surrogate.  
Figure 6: Ground-truth (top) and surrogate renderings (bottom) of scenes generated by the renderer.

![](images/00b45c4b581f489fdb725700e6287bb38127d50b8749a3278e55eba2025cd1a4.jpg)  
(b) Ground-truth top-night scene.

![](images/e88e5eb8ee66f6159e4c20487f5d998973ae0c802aae00a7a38c41f16bd30bec.jpg)  
(d) Frequency-based surrogate.

![](images/b15f7999be887c621b343068ea449f21b9febfc9db9f14a07a4e63088a43e1c7.jpg)

![](images/94a5b48851d40a8b38be6cc66b5c57db24ab368143c1bc47f073c6e75e58498e.jpg)  
(e) Uniform surrogate.

Table 1: Top: the identifier, lines of code, complexity, and description of each path present in our datasets. Bottom: the distribution (abbreviated distr.) of each path across a subset of datasets: the frequency (freq.) of each observed path, and the optimal sampling rate (opt.) of that path.  

<table><tr><td colspan="2">Path</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td></tr><tr><td colspan="2">Lines of Code</td><td>17</td><td>17</td><td>17</td><td>18</td><td>18</td><td>18</td><td>17</td><td>17</td><td>17</td></tr><tr><td colspan="2">Complexity</td><td>6210</td><td>5899</td><td>6369</td><td>6650</td><td>6328</td><td>6814</td><td>6459</td><td>6142</td><td>6621</td></tr><tr><td colspan="2">Description</td><td>Twilight Water</td><td>Twilight Smoke</td><td>Twilight Solids</td><td>Nighttime Water</td><td>Nighttime Smoke</td><td>Nighttime Solids</td><td>Daytime Water</td><td>Daytime Smoke</td><td>Daytime Solids</td></tr><tr><td>Dataset</td><td>Distr.</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td><td>Irrrlr</td></tr><tr><td rowspan="2">Front Day</td><td>Freq.</td><td></td><td></td><td></td><td></td><td></td><td></td><td>5.0%</td><td>7.9%</td><td>87.1%</td></tr><tr><td>Opt.</td><td></td><td></td><td></td><td></td><td></td><td></td><td>11.0%</td><td>14.7%</td><td>74.4%</td></tr><tr><td rowspan="2">Top Night</td><td>Freq.</td><td>0.16%</td><td>0.06%</td><td>1.2%</td><td>0.3%</td><td>0.1%</td><td>2.4%</td><td>6.3%</td><td>12.9%</td><td>76.5%</td></tr><tr><td>Opt.</td><td>0.95%</td><td>0.49%</td><td>3.6%</td><td>1.5%</td><td>0.8%</td><td>5.9%</td><td>10.9%</td><td>17.4%</td><td>58.4%</td></tr><tr><td rowspan="2">All</td><td>Freq.</td><td>0.04%</td><td>0.02%</td><td>0.3%</td><td>1.3%</td><td>2.0%</td><td>13.5%</td><td>4.5%</td><td>8.5%</td><td>69.8%</td></tr><tr><td>Opt.</td><td>0.35%</td><td>0.18%</td><td>1.3%</td><td>3.7%</td><td>4.8%</td><td>17.4%</td><td>8.2%</td><td>12.4%</td><td>51.6%</td></tr></table>

Table 2: Average change in error across all budgets from using optimal sampling compared to baselines on each dataset (negative means optimal sampling has lower error).  

<table><tr><td>Baseline</td><td>Front Day</td><td>Front Night</td><td>Top Day</td><td>Top Night</td><td>Front</td><td>Top</td><td>Day</td><td>Night</td><td>All</td><td>Mean</td></tr><tr><td>Frequency</td><td>-4%</td><td>+6%</td><td>+2%</td><td>-49%</td><td>+3%</td><td>-33%</td><td>-2%</td><td>-13%</td><td>-29%</td><td>-15%</td></tr><tr><td>Uniform</td><td>-46%</td><td>-32%</td><td>-36%</td><td>-45%</td><td>-47%</td><td>-64%</td><td>-35%</td><td>-58%</td><td>-53%</td><td>-47%</td></tr></table>

Paths. The program is a conjunction of 48 different paths, 9 of which are exercised by the renderer. The top part of Table 1 presents statistics about the paths under study, showing the identifier (a trace of 1 and  $\mathbb{r}$  characters denoting which branch of each if statement the path takes), the lines of code in the corresponding trace, and the complexity of the corresponding trace according to the analysis in Section 4.2. The paths are broken up into a path for rendering smoke particles from the chimney, water particles in the river, and the solids of the ground and house. Each set of paths is duplicated for twilight, nighttime, and daytime. Within each time of day, the smoke paths are the least complex, followed by water then solids. Across time, twilight paths the least complex, followed by daytime then nighttime.

Table 1 also presents the observed and optimal distributions of paths for each dataset. In general, the twilight paths are rarer than the nighttime paths, which are rarer than the daytime paths: this is because data collection for the nighttime scenes extends through twilight and into the morning. For all datasets, the smoke paths are rarer than the water paths, which are in turn rarer than the solids paths; this is purely due to the number of points observed for each scene.

Appendix E.1 presents code, statistics, and visualizations of all paths in the scene.

# 5.2 SURROGATE TRAINING AND DEPLOYMENT METHODOLOGY

To create and deploy a surrogate of the renderer, we train a surrogate of each path, then create a stratified surrogate which branches on the set of path conditions and applies the corresponding surrogate.

Our goal is to compare the theoretical and empirical errors achieved by training on the optimal sampling distribution against those of baseline distributions. We compare the approaches across different training datasets, different total numbers of training data points, and evaluating across different evaluation sets, all with multiple trials. Full methodological details are presented in Appendix E.2.

# 5.3 SURROGATE ERRORS

Table 2 presents the geometric mean change in error of using optimal sampling compared to each baseline, on each dataset. Across most datasets, optimal path sampling results in lower error than both frequency-based path sampling and uniform path sampling. On datasets with few paths (front-day) and in which all paths are well represented (minimum  $5\%$  frequency), the gap is minimal, and frequency-based path sampling matches or outperforms optimal path sampling. On datasets with more and rarer paths (top-night), the gap widens and optimal path sampling outperforms frequency-based path sampling. On all datasets, optimal path sampling outperforms uniform path sampling.

# 5.4 VISUALIZATION

Figure 6 presents the renderings generated by the surrogates for the front-day and the top-night scene. These budgets correspond to the smallest budget that lead to a validation error less than  $2\%$ , which was qualitatively chosen as a threshold around which surrogate renders converge on the ground truth.

The top row shows the front day scene using surrogates trained on the front day dataset. In this scene, the optimal and frequency-based surrogates result in similarly accurate renders, with the primary difference being that the frequency-based surrogate is overall more red, which is most notable in the windows and the front face of the house. This similarity is expected given the similar errors observed in Table 2. Uniform sampling results in an inaccurate render, as expected given its high error.

The bottom row show the top night scene using surrogates trained on the dataset combining all scenes. In this scene the optimal surrogate has the most accurate render, as expected given the errors observed in Table 2. The frequency-trained surrogate colors everything (especially the roof and water) slightly more pink. The uniform-trained surrogate colors everything significantly more green.

In sum, the error improvements in Table 2 result in visual improvements in the generated renders.

# 6 CONCLUSION

We present an optimal approach to allocating samples among strata to train stratified neural network surrogates of stratified functions. We also present a programming language, TURACO, in which all programs are learnable stratified functions and a program analysis to determine the complexity of learning surrogates of those programs. Our results take a step towards a cohesive, end-to-end methodology for programming using surrogates of programs.

# REFERENCES

Atish Agarwala, Abhimanyu Das, Brendan Juba, Rina Panigrahy, Vatsal Sharan, Xin Wang, and Qiuyi Zhang. One network fits all? modular versus monolithic task formulations in neural networks. In International Conference on Learning Representations, 2021.  
Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning, 2019.  
Tao Bao, Yunhui Zheng, and Xiangyu Zhang. White box sampling in uncertain data processing enabled by program analysis. In ACM International Conference on Object Oriented Programming Systems Languages and Applications, 2012.  
Cristian Cadar, Daniel Dunbar, and Dawson Engler. Klee: Unassisted and automatic generation of high-coverage tests for complex systems programs. In USENIX Conference on Operating Systems Design and Implementation, 2008.  
Michael Carbin, Sasa Misailovic, and Martin Rinard. Verifying quantitative reliability for programs that execute on unreliable hardware. In Conference on Object-Oriented Programming, Systems, Languages and Applications, 2013.  
Per Christensen, Julian Fong, Jonathan Shade, Wayne Wooten, Brenden Schubert, Andrew Kensler, Stephen Friedman, Charlie Kilpatrick, Cliff Ramshaw, Marc Bannister, Brenton Rayner, Jonathan Brouillat, and Max Liani. Renderman: An advanced path-tracing architecture for movie rendering. ACM Trans. Graph., 37(3), 2018.  
Corinna Cortes, Giulia DeSalvo, Claudio Gentile, Mehryar Mohri, and Ningshan Zhang. Region-based active learning. In International Conference on Artificial Intelligence and Statistics, 2019.  
Patrick Cousot and Radhia Cousot. Abstract interpretation: A unified lattice model for static analysis of programs by construction or approximation of fixpoints. In ACM SIGACT-SIGPLAN Symposium on Principles of Programming Languages, 1977.  
Hadi Esmaeilzadeh, Adrian Sampson, Luis Ceze, and Doug Burger. Neural acceleration for general-purpose approximate programs. In IEEE/ACM International Symposium on Microarchitecture, 2012.  
Andreas Griewank and Andrea Walther. Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation. SIAM, 2nd edition, 2008.  
Jan Hoffmann and Martin Hofmann. Amortized resource analysis with polynomial potential. In Programming Languages and Systems, European Symposium on Programming, 2010.  
Engin İpek, Sally A. McKee, Rich Caruana, Bronis R. de Supinski, and Martin Schulz. Efficiently exploring architectural design spaces via predictive modeling. In International Conference on Architectural Support for Programming Languages and Operating Systems, 2006.  
James C. King. Symbolic execution and program testing. Commun. ACM, 19(7):385-394, July 1976.  
Bogdan Kustowski, Jim A. Gaffney, Brian K. Spears, Gemma J. Anderson, Jayaraman J. Thiagarajan, and Rushil Anirudh. Transfer learning as a tool for reducing simulation bias: Application to inertial confinement fusion. IEEE Transactions on Plasma Science, 48(1), 2020.  
Jihye Kwon and Luca P. Carloni. Transfer learning for design-space exploration with high-level synthesis. In ACM/IEEE Workshop on Machine Learning for CAD, 2020.  
David Lettier. 3d game shaders for beginners, 2019. URL https://github.com/lettier/3d-game-shaders-for-beginners.  
Charith Mendis, Cambridge Yang, Yewen Pu, Saman Amarasinghe, and Michael Carbin. Compiler auto-vectorization with imitation learning. In Advances in Neural Information Processing Systems, 2019.

Andreas Munk, Adam Scibior, Attilm Güneş Baydin, Andrew Stewart, Goran Fernlund, Anoush Poursartip, and Frank Wood. Deep probabilistic surrogate networks for universal simulator approximation, 2019.  
Flemming Nielson, Hanne Riis Nielson, and Chris Hankin. *Principles of Program Analysis*. Springer, 1999.  
Raphael Pestourie, Youssef Mroueh, Thanh V. Nguyen, Payel Das, and Steven G. Johnson. Active learning of deep surrogates for PDEs: application to metasurface design. npj Computational Materials, 6(164), 2020.  
Alex Renda, Yishen Chen, Charith Mendis, and Michael Carbin. Difftune: Optimizing cpu simulator parameters with learned differentiable surrogates. In IEEE/ACM International Symposium on Microarchitecture, 2020.  
Alex Renda, Yi Ding, and Michael Carbin. Programming with neural surrogates of programs. In ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward!), 2021.  
Adrian Sampson, Werner Dietl, Emily Fortuna, Danushen Gnanapragasam, Luis Ceze, and Dan Grossman. Enerj: Approximate data types for safe and general low-power computation. In Conference on Programming Language Design and Implementation, 2011.  
Thomas J. Santner, Williams Brian J., and Notz William I. The Design and Analysis of Computer Experiments. Springer-Verlag, 2nd edition, 2018.  
Burr Settles. Active learning literature survey. Technical report, University of Wisconsin-Madison Department of Computer Sciences, 2009.  
Dongdong She, Kexin Pei, D. Epstein, J. Yang, Baishakhi Ray, and Suman Jana. NEUZZ: Efficient fuzzing with neural program smoothing. In IEEE Symposium on Security and Privacy, 2019.  
Natalya Tatarchuk. Advances in real-time rendering in 3d graphics and games, 2006. URL https://advances.realtimerendering.com.  
Hasan Tercan, Alexandro Guajardo, Julian Heinisch, Thomas Thiele, Christian Hopmann, and Tobias Meisen. Transfer-learning: Bridging the gap between real and simulation data for machine learning in injection molding. CIRP Conference on Manufacturing Systems, 72, 2018.  
Steven K. Thompson. Stratified Sampling, chapter 11, pp. 139-156. John Wiley & Sons, 2012.  
Ethan Tseng, Felix Yu, Yuting Yang, Fahim Mannan, Karl St. Arnaud, Derek Nowrouzezahrai, Jean-François Lalonde, and Felix Heide. Hyperparameter optimization in black-box image processing using differentiable proxies. ACM Transactions on Graphics (Proc. SIGGRAPH), 38(4), 2019.  
R. E. Wengert. A simple automatic derivative evaluation program. Commun. ACM, 7(8):463-464, August 1964.  
Glynn Winskel. The Formal Semantics of Programming Languages: An Introduction. MIT Press, 1993.
