# ARTWHISPERER: A DATASET FOR CHARACTERIZING HUMAN-AI INTERACTIONS IN ARTISTIC CREATIONS

Anonymous authors

Paper under double-blind review

# ABSTRACT

As generative AI becomes more prevalent, it is important to study how human users interact with such models. In this work, we investigate how people use text-to-image models to generate desired target images. To study this interaction, we created ArtWhisperer, an online game where users are given a target image and are tasked with iteratively finding a prompt that creates a similar-looking image as the target. Through this game, we recorded over 50,000 human-AI interactions; each interaction corresponds to one text prompt created by a user and the corresponding generated image. The majority of these are repeated interactions where a user iterates to find the best prompt for their target image, making this a unique sequential dataset for studying human-AI collaborations. In an initial analysis of this dataset, we identify several characteristics of prompt interactions and user strategies. People submit diverse prompts and are able to discover a variety of text descriptions that generate similar images. Interestingly, prompt diversity does not decrease as users find better prompts. We further propose a new metric to quantify the steerability of AI using our dataset. We define steerability as the expected number of interactions required to adequately complete a task. We estimate this value by fitting a Markov chain for each target task and calculating the expected time to reach an adequate score in the Markov chain. We quantify and compare AI steerability across different types of target images and two different models, finding that images of cities and natural world images are more steerable than artistic and fantasy images. These findings provide insights into human-AI interaction behavior, present a concrete method of assessing AI steerability, and demonstrate the general utility of the ArtWhisperer dataset.

# 1 INTRODUCTION

Direct human interaction with AI models has become widespread following a number of technical innovations improving the quality of text-to-text Brown et al. (2020); Ouyang et al. (2022); Anil et al. (2023) and text-to-image models Rombach et al. (2022a); Ramesh et al. (2022), enabling the public release of high-quality AI-based services like ChatGPT chatGPT, Bard Bard, and Midjourney Midjourney. These models have seen rapid interest and adoption largely due to the ability of the general public to interact with and steer the AI in diverse contexts including engineering, creative writing, art, education, medicine, and law Dakhel et al. (2023); Nguyen & Nadi (2022); Ippolito et al. (2022); Cetinic & She (2022); Qadir (2023); Cascella et al. (2023); Sloan (2023).

A key challenge in developing these models is aligning their output to human inputs. This is made challenging by the broad domain of use cases as well as the diverse prompting styles of different users. Many approaches can be categorized as "prompt engineering," where specific strategies for prompting are used to steer a model Oppenlaender (2022); Liu & Chilton (2022); Zhou et al. (2022b); Wei et al. (2022); White et al. (2023). Great success has also been found by fine-tuning models using relatively small datasets to follow human instructions Ouyang et al. (2022), respond in a specific style Hu et al. (2021), or behave differently to specified prompts Zhou et al. (2022a); Gal et al. (2022).

In this work, we take interest in the fact that human interaction with these models is often an iterative process. We develop a dataset to study this interaction. The dataset is collected through an interactive game we created where players try to find an optimal prompt for a given task (see Figure 1). In particular, we focus on text-to-image models and ask the player to generate a similar image ( $AI$

![](images/66f13c39f16b2c23bd8de96c8248cbdf759dac40c91d036183bbf59e3cd15605.jpg)  
Figure 1: Interface of the ArtWhisperer game. Prompts entered on right. Target (goal) image and player-generated image on left. Previous prompts and scores are displayed in the lower right.

![](images/4cffab94f37b9939a17b720e5d91897448b4c1db54ea5c299d214810b9f936c5.jpg)

![](images/7edd014b2ec57f731ad305bf6bf39dc249738a9c93178950e4045442ea6b8b84.jpg)

Image) to a given target image. The player is allowed to iterate on their prompt, using the previously generated image(s) as feedback to help them adjust their prompt. A score is also provided as feedback to help the user calibrate how "close" they are to a similar image.

Using this setup, we collected data on 51,026 interactions from 2,250 players across 191 unique target images. The target images were selected from a diverse set of AI-generated and natural images. We also collected a separate dataset of 4,572 interactions, 140 users, and 51 unique target images in a more controlled setting to assess the robustness of our findings.

Based on this data, we find several interesting patterns in how people interact with AI models. Players discover a diverse set of prompts that all result in images similar to the target. To discover these prompts, players typically make small, iterative updates to their prompts. Each update improves their image with a moderate success rate (40 - 60% for most target images). Based on these findings, we define and evaluate a metric for model steerability using the stopping time of an empirical Markov model. We use this metric to assess steerability across image categories and across two AI models.

Ethical considerations One of the main goals of this work is to help improve the quality of human-AI interaction. Our dataset and findings provide quantitative insights on how people interact with generative AI and can potentially be used to design AI that are easier for people to use. It does not address the broader concern that bad actors may abuse generative AI models.

Our contributions We release a public dataset on human interactions with an AI model. To our knowledge, this is the first such dataset showing repeated interactions of people with a text-to-image model to accomplish specified tasks. We also provide an initial analysis of this data and propose a simple-to-calculate metric for assessing model steerability. Our dataset and associated code is made available at [link redacted for submission].

Related Works Human-AI interaction datasets for text-to-text and text-to-image models typically focus on single interactions and generally do not provide users with a specific task. Public text-to-image interaction datasets typically contain the generated AI images and/or prompts Santana (2022); Wang et al. (2022) and optionally some form of human preference rating Pressman et al. (2022); Wu et al. (2023); Xu et al. (2023); Kirstain et al. (2023). These datasets generally rely on scraping online repositories like Lexica Lexica or Discord servers focused on AI art. Though some of these datasets include metadata that may allow for reconstruction of prompt iteration, there is no guarantee the user has the same desired output in mind over the iteration. Public text-to-text interaction datasets are much more limited as the best performing models are generally accessible only through APIs with no public user interaction datasets. While some researchers have investigated how human-AI interaction for text-to-text can be improved through various tools Wu et al. (2022a,b), the amount of data collected is limited and not publicly available. There are also repositories containing prompt strategies for various tasks Bach et al. (2022) but no human interaction component.

We seek to rectify two of the shortcomings of the existing datasets—namely, that they do not contain extended interactions as the user attempts to steer the AI, and they do not have a predefined goal. In our work, we create a controlled environment where we allow extended interactions and have a known goal for human users. As shown by our initial analysis, our dataset may enable deeper understanding of user prompting strategies and assessing model steerability.

# 2 INTERACTION GAME

In the game, players are shown a target image. A few example target images are provided in Figures 2.4. Players are also given a limited interface to a text-to-image model, Stable Diffusion (SD) v2.1 model Rombach et al. (2022b). In particular, players can enter a "positive prompt" (describes the desired content) and a "negative prompt" (describes what should be omitted) to steer the AI model. All models hyperparameters are fixed. Upon inputting a prompt, the player is shown the image generated by the AI model, along with a similarity score between their generated image and the target image. The interface is shown in Figure 1.

# 2.1 HOW TARGET IMAGES ARE SELECTED

We randomly sample target images from two sources. The first is a collection of Wikipedia pages, and the second is a dataset of prompts AI artists have used with SD Santana (2022). In addition to sampling target images, we need to ensure the task is feasible to users. We do not allow users to adjust the seed or other parameters of the model, so we need to ensure the selected model parameters can generate reasonably similar images to the target image. We find that selecting an appropriate random seed is sufficient, and fix all other model parameters (see Appendix A.4 for details and discussion).

Wikipedia Images A collection of 35 Wikipedia pages on various topics including art, nature, cities, and various people. A full list of pages sampled from is provided in Appendix 2. From these pages, we scraped 670 figures licensed under the Creative Commons license. These figures were then filtered by which had captions, as well as which images were JPG or PNG images (i.e., not animated, and not PDF files), resulting in 557 images.

For each of the 557 images, we first resize and crop the image to size  $512 \times 512$ . The Wikipedia caption is used as the ground truth "prompt". Let the image-caption pair be denoted as  $(t_i, p_i^*)$ . We sample the model on 50 random seeds, with  $p_i^*$  as the prompt input. This generates a set of 50 images:  $S_i = \{(x_i, s_i) : i = 1, \dots, 50\}$  for generated image  $x_i$  and seed  $s_i$ . Let  $C(x)$  denote the CLIP image embedding [Radford et al. (2021)] of an image  $x$ . Then we select the seed as  $s_i^*$ , where

$$
i ^ {*} := \min  _ {i = 1, \dots , 5 0} \left| \left| \frac {C (x _ {i})}{| | C (x _ {i}) | | _ {2}} - \frac {C (t _ {i})}{| | C (t _ {i}) | | _ {2}} \right| \right| _ {2}
$$

Here,  $s_{i^*}$  is selected to minimize the distance to the target image given the target prompt.

AI-Generated Images A collection of 2,000 AI-art prompts are randomly sampled from the Stable Diffusion Prompts dataset (2022). For each prompt,  $p_i^*$ , we generate two sets of images. As before, we use 50 unique random seeds to select the seed,  $s_{i^*}$  and an additional 10 random seeds to use for selecting the generated target image (so in total, we use 60 unique random seeds): the first set,  $S_{i,1} = \{(x_{i,1}, s_i) : i = 1, \ldots, 10\}$  and  $S_{i,2} = \{(x_{i,2}, s_i) : i = 1, \ldots, 50\}$ . We select the target image,  $t_{i_1^*}$ , from  $S_{i,1}$ :

$$
i _ {1} ^ {*} := \min  _ {i = 1, \dots , 1 0} \operatorname {m e d i a n} \left(\left\{\left| \left| \frac {C (x _ {i , 1})}{| | C (x _ {i , 1}) | | _ {2}} - \frac {C (x _ {j , 2})}{| | C (x _ {j , 2}) | |} \right| \right| _ {2}: j = 1, \dots , 5 0 \right\}\right)
$$

We select the random seed,  $s_{i_2^*}$ , using  $t_{i_1^*}$  and  $S_{i,2}$ , with

$$
i _ {2} ^ {*} := \min _ {i = 1, \ldots , 5 0} \left\| \frac {C (x _ {i , 2})}{| | C (x _ {i , 2}) | | _ {2}} - \frac {C (t _ {i _ {1} ^ {*}})}{| | C (t _ {i _ {1} ^ {*}}) | | _ {2}} \right\| _ {2}
$$

Here,  $t_{i_1^*}$  is chosen to be more representative of the types of images we may expect given the fixed prompt,  $p_i^*$ . This is because  $t_{i_1^*}$  is selected to be close to the center of the sampled images,  $S_{i,2}$ . The intuition for selecting  $s_{i_2^*}$  is the same as selecting  $s_{i^*}$  for the Wikipedia images.

# 2.2 SCORING FUNCTION

To provide feedback to players, we created a scoring function to assess the similarity of a player's generated image and the target image. We define the scoring function as

$$
s c o r e (x, t) = \max (0, \min (1 0 0, \alpha \cdot \left| \left| \frac {C (x)}{| | C (x) | | _ {2}} - \frac {C (t)}{| | C (t) | | _ {2}} \right| \right| _ {2} + \beta))
$$

![](images/7468a9beba70a3f180ddcfca42360ceb101a07d4a4888121a2a65ec3296e0ce1.jpg)  
Score: 85.0 a real-time rating of the cosmos  
Score: 79.0 a realistic depiction of the cosmos, we see a silhouette of a man from behind

![](images/9a99e09e15605f65d82c5edf2e77f7702c5869e77e64ead4d4e004f068c2691d.jpg)

![](images/e57e6a9a3f342e551beb6dec94696c3fb821ca4dc63c757404212ecf1a6b649c.jpg)  
Score: 93.0 a realistic,生动ing of the colorful blues with a beautiful, silhouette of a man on the left behind standing

![](images/a2f4fc578cebb72292ec37faf84699783f00b2402fd9cce9a8fa733b1225510e.jpg)  
Score: 100.0 a rare, black, yellow, orange, or fading of the colorful cosens with a pale, silhouette of a man on the left and a dark, standing

![](images/0b913e06928beed47f3aa4a805b0c8e7853ab93e47677518c41b375c83de8f19.jpg)

![](images/ba887b08fe45b488f707036fd12e9a1f63b25048c14db1f8e2abe725e37b75a0.jpg)  
dark silhouette of a man standing on a gray dull cube, looking at a colorful vibrant cosmos, clouds, stars, and the sun. A beautiful Lighting, magic, fantasy, vivid dream, elegant, cosplicity, artstation trending, oil painting by grey rukko, and aynong,

![](images/74d730f9868e054666160bd01c32b933511406bd001df689266d15cc27237e05.jpg)

![](images/f58e8c83dc29c08d64bd9d2077c1abe65c4bb322d5e6890a99fc9a52264cccab.jpg)  
Score: 51.0  
dotting of sedentarian city

![](images/4d43c28d6d89240d32691d22835c204e43a171c238c6a21d6e8990466a27ac21.jpg)  
Score: 76.0 digital drawing of mediterranean city, white walls

![](images/0d7290e08987def759714a4278a30b1d2c9d59656a6808227abbcd2719aaeb3f.jpg)  
Score:79.0 realistic digital version of mediterranean city, white walls

![](images/068c338544dc2896d25698a45c9d0d296ee034e8d70be1059075498f9b3c0fd4.jpg)  
Score:81.0 Total drawing of gregue city, 2000

![](images/fcc51449aa7b7798a6e6f1e97dd39497dcdeff7e0fa8e1222a17b72c5108b942.jpg)  
tracer from overseas  
painting, acrylic, blue streets, blue and white houses,MQusey, highly textured, acrylic painting, artstation concept art, smooth, acrylic, acrylic illustration, art by artpegma and grey rutkowski, Alphalone

![](images/09c46bb28cf3241408db9688eec76f755463170f07a5ff3ef7d7c951490f86f9.jpg)

![](images/bb9d869ca2fa16a0cbb21d39e757b92b14d4c15b0841b0df43679a2ce1575515.jpg)  
Score: 59.0 garden oil painting

![](images/38bd0970e580b2316eb1440c61a7488fe184e10dd46a72a1a0b1901b43d4f781.jpg)  
Score:61.0 green house garden oil painting,

![](images/5d55282708cc2d9b310a0aeb1b0b31f5c50907ad8968b8a0f07075a95f9919b0.jpg)  
Score:82.0 inside green house garden, glass ceiling and walls, oil painting,

![](images/318eecd0ea52b011d7527a450537b77d3e34eaaf259df6e28d37221a3f25df4c.jpg)  
Score: 100.0 inside green house quartz and quartz high dome glass ceiling and arch shaped windows and walls and small gardeners, painter, oil painting.

![](images/f483bc08a45cbc546209164bc1d1b493da69d6cc949cb9b7c9d5920d127da563.jpg)  
a beautiful intricate display of many flowers, reflections very high details by the artist and a great art trend in增添 an artstation, including masterpiece-, h-704

![](images/00799a84859a23d9c0cf53aa9efeb5eba2c51e762b241105bb23b1b173141519.jpg)

![](images/fe2b881cdc376ff24b8bb7aaa964e8ab965465beac81dd2adbf408d9a6b7b9a2.jpg)  
Score: 25.0 sepla color tone white male in with a keicher in the breast pocket

![](images/20b29b595a16d12e85dc81ed54c09667988066bd29ddaa105dd2b4e0f92a19f1.jpg)  
Score:74.0  
Sample image profile photo white in a suit with a kerchief and the breast pocket

![](images/954fbca6f7fdbdc1ef08dca6e90f3ee8e7a848130d1b0c451dac0367962444a9.jpg)  
Score:77.0
A 16-year-old man with front facing photo white male close-up in a suit with a large yellow breast pocket

![](images/6d538c24d3808d27854ef6e052311d504b92bdbe6929fa0fcfa2031367797825.jpg)  
Score:80.0 old female with a photo white young man close-up in a suit with a black breast pocket

![](images/b6c7942447af06995b822f47b4bd8ce67159cc89e0090c9eec0612944d04b131.jpg)  
Senior portrait c. 1920

![](images/df8353fd8e5d344dbc1a0cb4ad2192ea8674e3d9a99d2f88c752696ba18634fb.jpg)  
Figure 2: Example user trajectories. In each row, the first 4 images show the prompt progression for a given user. Target image shown in green column. Plot shows the average score trajectory across all users for this target image (blue) and the specific user's full score trajectory (red). Orange circles indicate the displayed images.

for generated image  $x$ , target image  $t$ , and constants  $\alpha, \beta$ . Note the range of  $score(x, t)$  is integers in the interval [0, 100]. Details on how  $\alpha, \beta$  are selected parameters are provided in Appendix A.3

While this scoring function is often reasonable, it does not always align with the opinions of a human user. To assess how well  $score(x,t)$  follows a user's preferences, we acquire ratings from a subset of users (see ArtWhisperer-Validation in Section 2.3). We find  $score(x,t)$  has a Pearson correlation coefficient of 0.579 indicating reasonable agreement. Further assessment is performed in Section 4.3 and discussed at length in Appendix A.13.

# 2.3 DATASET OVERVIEW

We collected two datasets: ArtWhisperer and ArtWhisperer-Validation. We use ArtWhisperer for most analysis and results; for some of the results in Sections 2.2, 4.2 and 4.3, we also use ArtWhisperer-Validation (when referenced). Data was collected from March-May 2023. IRB approval was obtained.

ArtWhisperer: A public version of our game was released online, with three new target images released daily. We collected data from consenting users playing the game. Users were not paid. Users were anonymous and we only collected data related to the prompts submitted to ensure privacy of

![](images/6d8e79fc142e4d892eb899e17b230ef99352d6f85350e537d1b679fe1c91aeef.jpg)  
Figure 3: Left, Distribution of # of user queries per target image. The average number of queries per image is 9.18. Right, Distribution of the # of words submitted in a query. The average number of words submitted in a positive and negative prompt are 20.02 and 2.32 respectively.

![](images/ddf9fd475f1ebe1fba1e56866d0cf05dea16bc4b83fdcc1ee5049a8d630a614f.jpg)

users. While we expect some users played the game across multiple days, we did not track them. A summary of the ArtWhisperer dataset is provided in Table  $\boxed{1}$ . In total, we have 2,250 (potentially non-unique) players corresponding to 51,026 interactions across 191 target images. Players interacted with the model SD v2.1. In Figure  $\boxed{3}$ , we plot the number of queries submitted by players across different target images.

ArtWhisperer-Validation: The game (with a near identical interface) was also released as a controlled user study to paid crowd workers on Prolific Academic. The crowd workers were compensated at a rate of $12.00 per hour for roughly 20 minutes of their time. Workers played the game across 5 randomly selected target images from a pre-selected subset of 51 target images chosen to have diverse content. Workers were also asked to rate each of their images on a scale of 1-10 (i.e., self-scoring their generated images). In total, we collected data on 4,572 interactions, corresponding to 140 users and 51 unique target images across two different diffusion models, SD v2.1 and SD v1.5. Additional details and demographic information are provided in Appendix A.6

Table 1: ArtWhisperer Dataset Overview. Each row contains summary data for a different subset of the dataset. Subsets may overlap. Similar information for ArtWhisperer-Validation is in Appendix A.6  

<table><tr><td># Players</td><td># Target Images</td><td># Interactions</td><td>Average # Prompts</td><td>Average Score</td><td>Median Duration</td><td>Category</td></tr><tr><td>2250</td><td>191</td><td>51026</td><td>9.29</td><td>58.93</td><td>18 s</td><td>Total</td></tr><tr><td>377</td><td>25</td><td>3884</td><td>8.65</td><td>56.70</td><td>19 s</td><td>Contains famous person?</td></tr><tr><td>353</td><td>32</td><td>3785</td><td>8.26</td><td>61.64</td><td>21 s</td><td>Contains famous landmark?</td></tr><tr><td>2005</td><td>140</td><td>40290</td><td>9.24</td><td>59.83</td><td>18 s</td><td>Contains man-made content?</td></tr><tr><td>1177</td><td>58</td><td>18255</td><td>10.93</td><td>57.21</td><td>17 s</td><td>Contains people?</td></tr><tr><td>344</td><td>77</td><td>6972</td><td>8.81</td><td>62.01</td><td>20 s</td><td>Is real image?</td></tr><tr><td>2140</td><td>103</td><td>43524</td><td>9.42</td><td>58.37</td><td>17 s</td><td>Is AI image?</td></tr><tr><td>1483</td><td>82</td><td>24913</td><td>9.14</td><td>59.45</td><td>17 s</td><td>Is art?</td></tr><tr><td>623</td><td>29</td><td>7297</td><td>9.14</td><td>53.77</td><td>18 s</td><td>Contains nature?</td></tr><tr><td>160</td><td>14</td><td>1355</td><td>7.28</td><td>65.74</td><td>19 s</td><td>Contains city?</td></tr><tr><td>1239</td><td>39</td><td>15872</td><td>9.91</td><td>56.74</td><td>16 s</td><td>Is fantasy?</td></tr><tr><td>618</td><td>19</td><td>8359</td><td>10.51</td><td>57.88</td><td>17 s</td><td>Is sci-fi or space?</td></tr></table>

# 3 PROMPT DIVERSITY

We quantify prompt diversity by looking at the distribution of prompts in the text embedding space. In particular, we use the CLIP text embedding Radford et al. (2021), though we do find the choice of embedding is not particularly important for our results (see Appendix A.7).

# 3.1 DIVERSE PROMPTS USED FOR HIGH SCORES

People achieve high scores with a diverse set of prompts. It is not surprising that this is possible (i.e., that the score metric has multiple local maxima), but it is potentially surprising that users consistently find these differing local maxima. Examples are shown in the four leftmost columns of Figure 4.

We quantify this finding in Figure 5 where we plot two metrics defined as follows. Let  $z_0, z_n$  be normalized embeddings of the initial and best prompt/image found by a user. Let  $z^*$  be the normalized embedding of the target prompt/image. We define the difference in embedding distance to ground truth as  $||z_n - z^*||_2 - ||z_0 - z^*||_2$ . In blue, we use the CLIP text embeddings of the prompts; in orange, we use the CLIP image embeddings of the generated images. We note two findings here: (1) the metric applied to the image embeddings is guaranteed to be non-positive as the embedding distance is monotonically decreasing with the score, and (2) the metric applied to the text embeddings is apparently symmetric around 0, indicating that unlike the image embedding, distance in the text embedding space does not monotonically decrease with score. Together, these findings illustrate that users tend to discover diverse prompts and do not converge in their prompt design.

The right two columns of Figure 4 provide another visualization at an individual level. Here, we plot a UMAP McInnes et al. (2018) projection for the CLIP image and text embeddings of the target image and a sample of the submitted prompts (and corresponding generated images). The target embedding is in orange, the first prompt embedding is in red, and the last (best scoring) prompt embedding is in green. The arrows connect a given user's first and last prompt. We see that a decrease in distance in the image embedding space (which is inversely correlated with the user's score) does not always correspond to a decrease in distance in the text embedding space.

Additionally, we find the distribution of prompts does not converge. In the left of Figure 6, we plot the distribution of distances between the first prompt (in blue) and the last prompt (in orange) to the average prompt for the corresponding target image. Despite the average score improving from 51.9 to 70.3 (out of 100) indicating a significant improvement in score, prompt diversity does not significantly diminish. That is, users do not converge to similar prompts to achieve high scores. Similar analysis of the image embedding space suggests image diversity decreases (Figure 5).

# 3.2 PEOPLE SUBMIT SIMILAR PROMPTS THROUGHOUT THEIR INTERACTION

In the center of Figure 6, we plot the distribution of the standard deviation of prompts for users (blue) and for permuted users (orange). Permuted users are generated by sampling from all prompts for a given target image uniformly, using the same distribution of number of prompts as for real users. The gap between the two distributions shows that individuals do not randomly sample prompts each interaction, but base new prompts off of previously submitted prompts (p-value  $< 10^{-10}$ , t-test for independent variables). An analysis of how scores change between adjacent prompts shows that this strategy has a moderate success rate and improves the score  $40 - 60\%$  of the time, with an average rate of  $48.6\%$  (note that score changes  $< 1$  are counted as unchanged; this occurs  $10.2\%$  of the time).

While this is not a surprising result (that users often do not make significant changes to their prompt), but it is an important result to understand how typical users interact with AI models. Moreover, it suggests that user initialization (i.e., the first prompt they submit) is critical.

# 3.3 PEOPLE HAVE SIMILAR PROMPT STYLES ACROSS IMAGES

We quantify user style by computing the difference (in the CLIP text embedding space) between the average prompt of a given user and the average prompt across all users for a given target image. To quantify style variation for a user, we then compute the standard deviation of the user style across the target images the user generated. In the right of Figure 6, we plot the distribution of user style variation for real users (blue) and permuted users (orange). Permuted users are generated by randomly sampling user styles. This allows us to test whether users have a consistent prompting style. We find users do indeed have specific styles of prompting (p-value  $< 10^{-10}$ , t-test for independent variables). However, the difference is not seemingly not large, suggesting that while user style may a component to prompting, other factors related to the target image may be more important.

![](images/6651c6425a1cee71398bd6f688751544f6467ba71334c6235d4a8c6b93bfff0c.jpg)  
Score: 100.0 Low, yellow, blue sky, a close-up coconut-free tree in the center of the golden beach, tropical trees, large area of greenish greenish volcanic rock mountain in distance. American cartoon style, high contrast, vivid colors, and white clouds

![](images/c364e1a68423755cf84e9e630c72e34287b41085ffd945a6d0055c4cd9cc9750.jpg)  
Score: 100. The sand coast in a large sand coast in the right and sea to the left and the right side dominated by wavy waves with white cumulus clouds, a palm tree in the right side, and a rocky grey cliff to the right side. Smooth

![](images/7152c9547da5df5325d9dfff3cd8307c067f0aba7305a15e324f8197b7b0a882.jpg)  
Score: 100.0 a painting of a small island with sand and a pale, the sea and a small noise; there are some clouds and waves

![](images/0823306212f7f4851574b74d04b720bf44a29992834d143642c73164657a777f.jpg)  
9:00  
A pale yellow sand to the sea on a sand beach with a mountain on left. The sea is filled with calm waves in the sea with white ripples bracing the sky. Cloud, illustration, colorful painting, sun and artstation

![](images/ed0a4127a10587c157ca76dd58be6e17a74e58424b5cc958d2459af8122fab14.jpg)  
Score: 100  
monday, nature, tree  
gym, garden, outdoors, pencils, seashells, pine trees

![](images/82e802586bbe31cd3f472b8f3d367308d05336099b38a130b6077e58633dca86.jpg)  
Score: 92.0 tall textured black pine trees above and white mist black and white drawing etched dark detailed

![](images/c03c53de416c2640b18ae38fd3e9a95cd64fe747f9df0196ba75fffc12dc04a3.jpg)  
ink score: 93.0
The redwooded tree forest floor in greenhouse is black, white, and black, wide leafy vegetation covering all trees. The trees are mostly only the trunks of the trees and have no seeds with needles sweep down. Clear in the form of a tree in the distant background.

![](images/bab040fd220001c43ac1afddd2f4c7fb5d3e22add3efc192d27e3d430e4efff2.jpg)  
Score: 99.0 redwood forest drawing

![](images/232c3ffbbf775cc923f013dacdda717a53f30e608f84f25a5efbefee3c84acc6.jpg)  
A beautiful award winning painting of a 1000 square foot palm trees and a navy blue ocean trending on artificial水晶 color scheme lots detail

![](images/fcdda1118e77d025d76ac21e0cdd4b2e19f84a549642ff7643ae619cd742047c.jpg)  
15 10 8 10 12 UMAP-1 (Prompt Text)

![](images/5f1c5c5955b2a1fed3f42627a53376d05e7ad6c474fbdae783a0f4b116341b74.jpg)  
a beautiful woodcut provides a view of the forest, 8k, frostbite,engineine, cryogenetic, de-mentation, artstation, digital art, crepuscular ray, airbrushing and tugbato printshop

![](images/7c579c052b4ce759e1ad24bd6bdb16de4869bf3cde8b4c61529cff701be47371.jpg)  
UMAP-1(Prompt Text) UMAP-2(PromptText)

![](images/5d29b81b72efbc29bb89f70237fe3415839fe66e41eaa33ffced52745db602f5.jpg)  
Figure 4: Left: Diverse, high-scoring prompt submissions from different players. Target image in green column. Right: Image (left) and text (right) embeddings of displayed images (target in orange; user submissions in green (best) and red (first)) using the UMAP McInnes et al. (2018) projection.  
Figure 5: Difference of distance from the first prompt to ground truth and distance from the last (best) prompt to ground truth for CLIP text (blue) and CLIP image embeddings (orange).

![](images/612e6583fd25e336edc5ecc9defd28593529fe390bdc67b1151f2071d4ddc2a3.jpg)  
Figure 6: Left: Users submit diverse (across users) prompts, both at beginning and end of interaction. Center: Individual users do not submit diverse prompts. Right: Users have different prompting styles.

![](images/e85c73a6e21d1e91b1369cbc59a82845a3c1f72dbaf1c0fed12889b127765c5b.jpg)

![](images/85bc19936a6590068808a2ab85a7faf7162a82df40da22801ee7fd0935545b32.jpg)

# 4 MODEL STEERABILITY

Model steerability refers to the ability of a user to steer a model towards a desired outcome. There is no current consensus on how to measure AI steerability. A common approach is to simply measure performance of a model on standardized dataset evaluations Jahanian et al. (2019); OpenAI (2023). While this can enable comparisons between tasks and models, this approach does not allow for the feedback loop present when humans interact with a model. Steerability can also be measured qualitatively based on user assessment of their experience interacting with the AI Chung et al. (2022). We create a simple yet informative measure of model steerability. We then analyze this measure across different subgroups of images and across two different Stable Diffusion models-SDv2.1 and the older SDv1.5 Rombach et al. (2022a).

# 4.1 MEASURING STEERABILITY

As discussed in Section 3.2, users typically engage with the model through clusters of similar prompts. They typically start with an initial base prompt and proceed to make multiple incremental modifications to it. We use this observation as a basis for creating a steerability metric. We define a Markov chain between scores. Each node is a score with edges connecting to the subsequent score. To make this tractable for empirical analysis, we bin scores into five groups: [0, 20], [21, 40], [41, 60], [61, 80], [81, 100]. We use the expected time taken to reach the last score bin, [81, 100], as our steerability score (i.e., the stopping time to reach an adequate score).

For each target image, we calculate the empirical transition probability matrix between binned scores using all the players' data for that image. We then calculate the steerability score for the given target image by running a Monte Carlo simulation to estimate stopping time, as defined above. To assess steerability across a group of images, we average steerability score across all images in the group.

# 4.2 ANALYSIS

In Figure 7, we plot the steerability score across image groups. Error bars show the standard error. For examples of steerability scores for individual images, see Appendix A.11 We find that images containing famous people or landmarks, real images (not AI generated), contain cities, or contain nature are the most steerable. AI-generated images, fantasy images, and images of human art are the least steerable. There are a few possible explanations. The model we are assessing here, SDv2.1, as well as its text encoder OpenCLIP, are trained on subsets of LAION5B Schuhmann et al. (2022). The contents of LAION5B are predominantly real world images, indicating why these images may be more steerable (i.e., text describing these types of images may have a better encoding). Moreover, the prompts for AI-generated images and fantasy images generally include specific internet artists and/or art styles which may not be known to most users making achieving the desired target image more difficult. Another potential reason is the distribution of images chosen for each category. Clearly, there are "easier" and "more difficult" images in each category; part of the reason for smaller stopping time may be the sample of images chosen rather than the actual image category.

Using the ArtWhisperer-Validation data, we also compare steerability across two models: SDv2.1 and SDv1.5. Across most image categories, we observe a similar steerability. Images of nature, sci-fi or space, and real images have the largest differences in steerability between the two models; SDv2.1 is more steerable in all three cases. This suggests that SDv2.1 may be more steerable for natural images as well as sci-fi images, and is similarly steerable for other kinds of images including AI-generated artwork. One explanation may be that most of our users were not aware of certain prompting strategies that help models generate more aesthetic images or certain art styles; it is possible that for experienced users, AI art images may be more steerable, and differences between models may be magnified if, for example, a user is experienced working with one particular model. More discussion is provided in Appendix A.10

# 4.3 JUSTIFICATION FOR AUTOMATED SCORE

One limitation of our steerability metric comes from the method of scoring user-submitted prompts. Ideally, we would assess steerability with a user's personal preferences. As mentioned in Section 2.2, the scores and human ratings have a positive correlation. Here, we use the human ratings from

![](images/852e5199ff6a08805ca2475f685ff2fe5ee4ca3dd93d99453656c07a627a6330.jpg)  
Figure 7: Steerability across image groups (smaller indicates increased steerability). Bars show average expected stopping time across images in the image group; error bars show standard error.

ArtWhisperer-Validation instead of our score function to assess steerability. We compute the steerability score across both models and across image groups. Generally, the steerability scores change little. In all but two cases (SDv2.1 on sci-fi and space images; SDv1.5 on nature images), the human rating-based steerability score remains within a  $95\%$  confidence interval of the score-based steerability score. While our score function may not perfectly capture human preferences, the steerability score we generate appears to be robust to these issues. Further discussion is included in Appendix A.13

# 5 DISCUSSION

As demonstrated in our analysis, the ArtWhisperer and ArtWhisperer-Validation datasets can provide insights into user prompting strategies and enables us to assess model steerability for individual tasks and groups of tasks. What makes our dataset particularly useful is the controlled interactive environment, where users work toward a fixed goal, that we capture data in.

One of the most exciting use cases we see for our dataset is to create synthetic humans for prompt generation. For example, similar to the method described in Promptist Hao et al. (2022), we imagine fine-tuning a large language model with our dataset to generate prompt trajectories (i.e., rather than an optimized prompt) using similar exploration strategies as a human prompter. These synthetic prompters could be based on multimodal models like OpenFlamingo Awadalla et al. (2023) or text-only models and use score-feedback to condition the trajectory generation. As an initial proof-of-concept, we fine-tuned a MT0-large model Muennighoff et al. (2022) model on our dataset and found the fine-tuned model can indeed behave similarly to human users (see Appendix A.14). These synthetic prompters have several potential use cases:

1. Automating measurement of text-to-image model steerability by using synthetic users in place of real human prompters. While we believe our proposed steerability metric is effective, its main limitation currently is the requirement for human annotations.  
2. Incorporating steerability in the objective function for text-to-image models. By representing steerability as a function of synthetic users, it becomes possible to explicitly optimize a model for steerability.  
3. Generating human readable image captions that are compatible with a Stable Diffusion model by using the synthetic prompter to optimize the token representation of the prompt.

Additionally, our dataset can be used for further analysis on human prompting strategies beyond what we discussed in the paper. For example, one question we only touched upon is whether we can compare human prompts to automated prompt optimization methods (e.g., do humans behave similar to some gradient-based optimization approach in the prompt embedding space?). There are also potential uses for crafting better image similarity metrics using the human ratings we collected.

# REFERENCES

Rohan Anil, Andrew M. Dai, Orhan First, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Diaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov and Yonghui Wu. Palm 2 technical report, 2023.  
Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.  
Stephen H Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, et al. Promptsource: An integrated development environment and repository for natural language prompts. arXiv preprint arXiv:2202.01279, 2022.  
Bard. https://http://bard.google.com, 2023.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901, 2020.  
Marco Cascella, Jonathan Montomoli, Valentina Bellini, and Elena Bignami. Evaluating the feasibility of chatgpt in healthcare: an analysis of multiple clinical and research scenarios. Journal of Medical Systems, 47(1):33, 2023.  
Eva Cetinic and James She. Understanding and creating art with ai: review and outlook. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 18(2):1-22, 2022.  
chatGPT. https://chat.openai.com, 2023.  
John Joon Young Chung, Wooseok Kim, Kang Min Yoo, Hwaran Lee, Eytan Adar, and Minsuk Chang. Talebrush: sketching stories with generative pretrained language models. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pp. 1-19, 2022.  
Arghavan Moradi Dakhel, Vahid Majdinasab, Amin Nikanjam, Foutse Khomh, Michel C Desmarais, and Zhen Ming Jack Jiang. Github copilot ai pair programmer: Asset or liability? Journal of Systems and Software, 203:111734, 2023.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.  
Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. arXiv preprint arXiv:2212.09611, 2022.  
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778, 2016.  
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.  
Daphne Ippolito, Ann Yuan, Andy Coenen, and Sehmon Burnam. Creative writing with an ai-powered writing assistant: Perspectives from professional writers. arXiv preprint arXiv:2211.05030, 2022.  
Ali Jahanian, Lucy Chai, and Phillip Isola. On the" steerability" of generative adversarial networks. arXiv preprint arXiv:1907.07171, 2019.  
Apoory Khandelwal, Luca Weihs, Roozbeh Mottaghi, and Aniruddha Kembhavi. Simple but effective: Clip embeddings for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14829-14838, 2022.  
Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. arXiv preprint arXiv:2305.01569, 2023.  
Lexica. https://lexica.art/ 2023.  
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950-1965, 2022.  
Vivian Liu and Lydia B Chilton. Design guidelines for prompt engineering text-to-image generative models. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pp. 1-23, 2022.  
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.  
Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprint arXiv:2206.00927, 2022.  
Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018.  
Midjourney. https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F 2023.  
Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng-Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. arXiv preprint arXiv:2211.01786, 2022.  
Nhan Nguyen and Sarah Nadi. An empirical evaluation of github copilot's code suggestions. In Proceedings of the 19th International Conference on Mining Software Repositories, pp. 1-5, 2022.  
OpenAI. Gpt-4 technical report, 2023.  
Jonas Oppenlaender. Prompt engineering for text-based generative art. arXiv preprint arXiv:2204.13988, 2022.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730-27744, 2022.  
Jeffrey Pennington, Richard Socher, and Christopher D Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 1532-1543, 2014.  
John David Pressman, Katherine Crowson, and Simulacra Captions Contributors. Simulacra aesthetic captions. Technical Report Version 1.0, Stability AI, 2022. url https://github.com/JD-P/simulacra-aesthetic-captions .  
Prolific Academic. https://www.prolific.co.2023.  
Junaid Qadir. Engineering education in the era of chatgpt: Promise and pitfalls of generative ai for education. In 2023 IEEE Global Engineering Education Conference (EDUCON), pp. 1-9. IEEE, 2023.  
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748-8763. PMLR, 2021.  
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684-10695, June 2022a.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684-10695, 2022b.  
Gustavo Santana. Stable-diffusion-prompts. Huggingface Datasets, 2022. URL https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts.  
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/forum?id=M3Y74vmsMcY.  
Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.  
Karen Sloan. A lawyer used chatgpt to cite bogus cases. what are the ethics? Reuters, May 2023. URL https://www.reuters.com/legal/transactional/ lawyer-used-chatgpt-cite-bogus-cases-what-are-ethics-2023-05-30.  
Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. Diffusiondb: A large-scale prompt gallery dataset for text-to-image generative models. arXiv preprint arXiv:2210.14896, 2022.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.  
Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, and Douglas C Schmidt. A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382, 2023.

Tongshuang Wu, Ellen Jiang, Aaron Donsbach, Jeff Gray, Alejandra Molina, Michael Terry, and Carrie J Cai. Promptchainer: Chaining large language model prompts through visual programming. In CHI Conference on Human Factors in Computing Systems Extended Abstracts, pp. 1-10, 2022a.  
Tongshuang Wu, Michael Terry, and Carrie Jun Cai. Ai chains: Transparent and controllable human-ai interaction by chaining large language model prompts. In Proceedings of the 2022 CHI Conference on Human Factors in Computing Systems, pp. 1-22, 2022b.  
Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Better aligning text-to-image models with human preference. ArXiv, abs/2303.14420, 2023.  
Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation, 2023.  
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586-595, 2018.  
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337-2348, 2022a.  
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910, 2022b.