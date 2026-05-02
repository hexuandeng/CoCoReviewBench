# VERY LARGE-SCALE MULTI-AGENT SIMULATION WITH LLM-Powered Agents

Anonymous authors

Paper under double-blind review

# ABSTRACT

Recent advances in large language models (LLMs) have opened new avenues for applying multi-agent systems in very large-scale simulations. However, there remain several challenges when conducting multi-agent simulations with existing platforms, such as limited scalability and low efficiency, unsatisfied agent diversity, and effort-intensive management processes. To address these challenges, we develop several new features and components based on a user-friendly multi-agent platform, enhancing its convenience and flexibility for supporting very large-scale multi-agent simulations. Specifically, we propose an actor-based distributed mechanism as the underlying technological infrastructure towards great scalability and high efficiency, and provide flexible environment support for simulating various real-world scenarios, which enables parallel execution of multiple agents, automatic workflow conversion for distributed deployment, and both inter-agent and agent-environment interactions. Moreover, we develop an easy-to-use configurable tool and an automatic background generation pipeline, simplifying the process of creating agents with diverse yet detailed background settings. Last but not least, we provide a web-based interface for conveniently monitoring and managing a large number of agents that might deploy across multiple devices. We conduct a comprehensive simulation to demonstrate the effectiveness of these proposed enhancements, and provide detailed observations and insightful discussions to highlight the great potential of applying multi-agent systems in large-scale simulations.

# 1 INTRODUCTION

Large language models (LLMs) (OpenAI, 2023; ANTHROP, 2024; MistralAI, 2024) demonstrate remarkable abilities in understanding, generating, and interacting with human language. Recent advancements in LLMs have sparked a revolution in natural language processing and related fields, paving the way for novel applications that were previously inconceivable. Building on the capabilities of LLMs, there is growing interest in the development of intelligent agents, which have the potential to redefine the landscape of simulations (Matsumoto et al., 2024; Sorokovikova et al., 2024; Sreedhar & Chilton, 2024; Yue et al., 2024) towards more interactive, adaptive, and realistic, while requiring substantially fewer human efforts.

Recently, several platforms (Hong et al., 2024b; Wu et al., 2023; Team, 2023) have been proposed to streamline the development of multi-agent systems, providing some fundamental functionalities including unified LLM services, various tools, and advanced reasoning algorithms. Despite significant progress, we identify several challenges in conducting simulations with multi-agent platforms, particularly when the number of agents becomes extremely large:

(i) Scalability and Efficiency Limitations. The scale of involved agents can be critical since simulations at a small scale run the risk of inaccurately representing real-world complexities, making simulations less realistic and reliable (Macal & North, 2010; Macal, 2016). However, increasing the scale of agents brings challenges to the simulation platform in terms of scalability and efficiency. It is non-trivial to efficiently organize agents to execute their tasks and communications following an appropriate order, with the aim of reducing the running time while ensuring accurate results.  
(ii) Unsatisfied Population Distributions and Agent Diversity. For a large-scale simulation, it is essential that the involved agents exhibit diverse behaviors while generally following a specific population distribution (Gao et al., 2023; Ren et al., 2024). Assigning agents with simple backgrounds may result in a significant number of highly homogeneous agents, making it difficult to derive meaningful insights. The existing studies rarely consider how to specify population distributions of agents from different perspectives, such as age, education, occupation, etc.  
(iii) Difficult Management Processes. As the scale of agents increases, it becomes rather effort-intensive to manage the simulations, including initialization, execution, and termination of a large number of agents spread across multiple devices, as well as monitoring their status, behaviors, and interactions (Mou et al., 2024). Such difficulties in managing

make it challenging to promptly identify valuable group-level and individual-level behaviors, which can further hinder the discovery of critical insights for optimizing simulations and advancing research.

To tackle these challenges, we adopt a user-friendly multi-agent platform, named AgentScope (Gao et al., 2024), as the foundation framework to provide the basic functionalities, and further develop several new features and components upon it to improve its usability, convenience, and flexibility for supporting very large-scale multi-agent simulations.

Specifically, we propose a distributed mechanism based on the actor model (Agha, 1985), featuring agent-level parallel execution and automatic workflow conversion to provide great scalability and high efficiency for multi-agent-based simulations. The proposed actor-based distributed mechanism enables us to further expand the scale of agents in the simulation with a limited number of devices, and provides linear benefit on running time from the addition of devices. Users can convert the simulations from a centralized workflow into a distributed one without further modifications except for adding a to_dist function. Besides, we support both inter-agent and agent-environment interactions in the simulations. Agents can communicate with each other, and can query some shared states in the environment, respond to their changes, and make modifications as needed. We enable user-defined functions within the environment to support flexible extension of various states and their corresponding query and modification behaviors.

To satisfy the requirements of population distribution and agent diversity, we provide a configuration tool and an automatic background generation pipeline. Users only need to simply specify the distributions of the population from several aspects, a large number of agents with detailed and diverse characteristics can be effortlessly generated accordingly. These agents can be managed and monitored conveniently through Agent-Manager, a proposed module for simplifying the organization and observation process of large-scale agent-based simulations. Using a web-based visual interface, Agent-Manager provides a comprehensive overview of all agents across multiple devices, allowing users to efficiently configure, launch, and terminate these agents.

With such an agent-based simulation platform, we conduct a comprehensive simulation on the classic "guess  $\frac{2}{3}$  of the average" game (Nagel, 1995; Camerer et al., 2004) to demonstrate the improvements and advances brought by the infrastructure introduced above. Firstly, we conduct agent-based simulations involving 1 million agents using only 4 devices, showing the scalability and efficiency of the platform. Then, we incorporate agents using different LLMs of different sizes, equipped with different prompts and diverse background settings, resulting in various and realistic behaviors in the simulations. We provide detailed observations on both collective and individual behaviors, drawing meaningful and valuable insights from a series of simulation experiments, along with further discussions on helpful tips and open questions. These experimental results confirm the feasibility and great potential of conducting large-scale agent-based simulations with LLM-powered agents.

# 2 RELATED WORKS

LLM-Powered Agent Platforms A significant number of agent platforms have been developed to integrate LLMs into real-world applications and assist humans in problem-solving (Hong et al., 2024b; Li et al., 2023b; Significant-Gravitas, 2023; Team, 2023; Chen et al., 2024b; Gao et al., 2024). These platforms can be categorized into single-agent platforms and multi-agent platforms. The single-agent platforms include AutoGPT (Significant-Gravitas, 2023), LangChain (langchain ai, 2024a), ModelScope-Agent (Li et al., 2023a), and Transformers Agents (Wolf et al., 2020), which are proposed to resolve practical tasks using LLMs. On the other hand, multi-agent platforms like MetaGPT (Hong et al., 2024b), Auto-Gen (Wu et al., 2023), CAMEL (Li et al., 2023b), and LangSmith (langchain ai, 2024b) employ multi-agent collaboration to tackle more complex challenges, including software programming (Hong et al., 2024b; Qian et al., 2024), data science (Hong et al., 2024a), social simulation (Park et al., 2023), game-playing (Chen et al., 2024a), etc. Although remarkable progress has been made, applications built on these platforms can currently be limited in the scale of agents and suffer from low efficiency, hindering their potential for large-scale simulations.

Agent-Based Simulation Frameworks Recently, agent-based simulation has become an attractive topic in the research community (Gürcan, 2024; Sorokovikova et al., 2024; Sreedhar & Chilton, 2024; Ye & Gao, 2024; Team et al., 2024; Park et al., 2023). Previous studies have explored the integration of LLMs in various fields, including education (Yue et al., 2024), economic (Matsumoto et al., 2024), societal study (Park et al., 2023; Ye & Gao, 2024; Gao et al., 2023; Ren et al., 2024), transportation (Jin et al., 2023), healthcare (Zhang et al., 2023), etc. Researchers have built up several LLM-based or agent-based simulation frameworks. For instance, Vidur (Agrawal et al., 2024) is a simulation framework that focuses on providing high-throughput LLM services, SOTOPIA (Zhou et al., 2024) provides an environment to simulate various social scenarios and evaluates the social intelligence of agents, Cheng et al. (2023) proposes to evaluate the level of caricature, and Ren et al. (2024) designs a framework to simulate the behaviors of web search users. However, these existing frameworks are domain-specific and lack flexibility and extensibility, making it challenging for users to conduct large-scale agent-based simulations for a wide variety of applications.

![](images/f37f1f923b6fffa505f335fbb1c73e62764c5dbe08de06505ad5e8e57bfd9b16.jpg)  
Figure 1: The example of the automatic conversation of a centralized workflow with four agents running sequentially (on the left) into a distributed workflow with agents running in parallel (on the right). Users only need to add a to_dist function during the initialization phase, without any further modifications required.

# 3 INFRASTRUCTURE

To provide the basic functionalities required for conducting agent-based simulations, we adopt AgentScope, a user-friendly multi-agent platform, as our foundation framework. We further develop several new features and components, making it more convenient and feasible to support very large-scale simulations involving multiple agents. In the following subsections, we elaborate on the details of these proposed enhancements.

# 3.1 ACTOR-BASED DISTRIBUTED MECHANISM

The actor model is a mathematical model of concurrent computation, where each actor acts as a basic computing unit, receives messages, and computes independently (Agha, 1985). Based on the actor model, we design a distributed mechanism to provide agent-level parallel execution for achieving high efficiency and great scalability for agent-based simulation, and to support automatic workflow conversion for migrating the centralized orchestrated workflow into distributed scenarios effortlessly.

Agent-Level Parallel Execution In a multi-agent simulation, the interactions between agents follow an atomized pattern, where interactions occur within small isolated cliques (Matsumoto et al., 2024; Sorokovikova et al., 2024). Such a pattern holds significant potential for parallelization, leading to substantial gains in efficiency.

With the actor model, agents that do not rely on the outputs of others or whose dependencies have all been satisfied, can be executed in parallel for enhancing efficiency. As agents complete their executions and produce results that may be needed by others, some previously waiting agents become active and initiate their executions. An example of the proposed agent-level parallel execution can be found in Appendix B.1.

We propose two multi-process modes for various simulation scenarios, supporting both one-to-one and many-to-one relationships between agents and processes. Communication across processes and devices utilizes Remote Procedure Call, and we provide implementations in both Python and  $\mathrm{C + + }$  . The one-to-one multi-process mode, where each agent runs in a separate process, is well-suited for agents performing computation-intensive tasks, ensuring that each agent has sufficient computational resources and that the parallelization would not be hindered by Global Interpreter Lock (GIL). For the many-to-one multi-process mode where multiple agents run within a single process, agents within a single process share CPU cores, communication ports, and global variables. Such a mode is well-suited for situations where agents experience I/O wait times or are awaiting responses from remote APIs, allowing the CPUs to leverage the time-sharing mechanism to maximize resource utilization.

In this way, the proposed framework achieves higher resource utilization in parallelism compared to existing ones (Wu et al., 2023; Hong et al., 2024b) that rely on asynchronous I/O in Python, which can be constrained by GIL. Furthermore, it also makes a significant advancement over existing actor-based distributed frameworks, such as Ray (Moritz et al., 2017), which allocate a new worker process for each actor, resulting in wasted computational resources when applying for large-scale agent-based simulations.

Automatic Workflow Conversion To ease the distributed deployment of large-scale simulations, we provide an easy-to-use function called to_dist to convert a centralized workflow into a distributed one effortlessly. The conversion involves two automatic stages. In the first stage, each agent in the center (i.e., the main procedure) is distributed to a

specified device, with a proxy left in the center as a substitute. This proxy can be utilized for users to orchestrate the workflow, maintaining parity with the centralized mode, and is responsible for automatically forwarding messages to the corresponding distributed agents. In the second stage, placeholders are introduced to ensure that calculations performed by the distributed agents do not block the workflow execution in the center. When a proxy receives a message, it immediately returns a placeholder and forwards the message to its corresponding distributed agent for processing. For a distributed agent, when receiving a message, it first checks if the message contains placeholders to decide whether it needs to request and wait for the results from the agents indicated on the placeholders. This mechanism allows the main procedure to continue executing without waiting for the results from the distributed agents.

The example in Fig. 1 shows how a centralized workflow can be automatically converted to a distributed one, without any further modifications except for adding to_dist. Firstly, all agents are transformed into distributed agents, with corresponding proxies left at the center. Then, the center can continue its execution until it needs to print msg_b (i.e., line 15 of the code), thanks to the usage of placeholders that replace messages msg_a/b/c/d as responses. Once receiving msg_a from agent-A, agent-B and agent-C run in parallel to produce msg_b and msg_c, respectively, which would be finally sent back to the center for printing out.

# 3.2 AGENT-ENVIRONMENT INTERACTIONS

Agent-environment interactions are also crucial alongside the above inter-agent communication in agent-based simulations, which indicates that agents can access the shared states in the environment, respond to their changes, and make modifications as needed.

For large-scale simulations, the environment is expected to meet the following requirements: (a) High concurrency access. Given a large number of agents, the environment should support high concurrency access, especially when agents need to frequently check and interact with the environment's states. (b) Diverse states. Different simulation scenarios may need different states to be maintained in the environment (Yang et al., 2024b; Xi et al., 2024). For example, in a chat room simulation, the states might include participants and conversation history, while in a maze simulation, the states contain the locations of agents and other interactable items. These diverse states pose challenges regarding the flexibility and extensibility of the environment module. (c) Bi-directional interaction. The interactions between agents and the environment should be bidirectional (Beer, 1995), indicating that agents can query or modify the states in the environment, and the environment can also actively send notifications or requests to agents. For example, a chatroom can notify specific participants when they are mentioned or called. (d) Multiple environments. Some simulation scenarios might need to have multiple environments concurrently. For example, in a social simulation where agents are divided into subgroups, agents within a subgroup may share one dedicated environment for collaborative tasks or information synchronization without interference from other groups. Multiple environments allow for more tailored interactions and state management specific to each subgroup's unique objectives.

To satisfy all the aforementioned requirements, we abstract the environment as a special type of agent with rich functionalities, which allows the environment to maintain shared variables as states and to communicate with other agents through Remote Procedure Calls for ensuring high concurrency. Meanwhile, we enable user-defined functions within the environment to support flexible extension of various states and their corresponding query and modification behaviors. For bi-directional interactions, we introduce listeners, which can be attached to user-defined functions in the environment, automatically triggering the sending of messages from the environment to agents when specific conditions are met. Furthermore, we support the nesting of environments and agents to allow multiple environments within a single simulation. An example of a multi-environment designed for group-wise information synchronization is shown in Appendix B.2, and a use case can be found in Sec. 4.5.

# 3.3 HETEROGENEOUS CONFIGURATIONS

In a simulation, agents are expected to act as humans with diverse backgrounds, including different ages, genders, careers, nationalities, education, experiences, etc. An intuitive approach is to add these background settings of agents in their system prompts, providing guidance for agents on the roles to play and actions to take. However, for large-scale simulations, providing diverse, heterogeneous, and reasonable background settings for agents can be laborious and time-consuming, especially when precise control of different population distributions is required in certain simulations. This problem motivates us to provide easy-to-use tools to assist users in effortlessly setting up large-scale agents with diverse background settings.

Configurable Tool Specifically, users can begin by defining the total population of the simulation, and then specify the distributions of the population from various perspectives. We provide some widely-used distribution templates for convenient usage, from the aspects of age, gender, occupation, nationality, and education. Besides, the proposed

configurable tool allows for easy extension of new aspects, enhancing its flexibility to meet diverse requirements. An example configuration file is shown in Appendix B.3 for a group of people with different educational levels, in which the proportions of its different components can specify a distribution.

Automatic Background Generation Pipeline After configurations have been provided via the above tool, more detailed and heterogeneous background settings can be automatically generated to instantiate the agents. Specifically, when users start a simulation, we draw specific values from the distributions based on the configurations, convert them into a JSON format, and fill them into a meta prompt to produce the completed instructions for background generation tasks. These instructions are utilized by LLMs to generate heterogeneous background settings. To introduce more diversity, the generation process involves adjusting the random seed and the temperature used by LLMs. Several examples of the generated background settings can be found in Sec. 4.4, along with the results and analysis of the simulations involving diverse agents.

# 3.4 MANAGEMENT FOR LARGE-SCALE AGENTS

In a simulation, users need to manage and monitor a large number of agents distributed across different devices, which might become intractable to handle manually as the scale and complexity of the simulation increase. To tackle this, we incorporate advanced forms of agent management and monitoring, named Agent-Manager. Specifically, when users start a simulation, servers are first launched on all the remote devices, which provide resident services to remotely create, monitor, and stop distributed agents. These servers are responsible for managing the lifecycle of distributed agents and synchronizing their information to a web-based visual interface, which is provided for a comprehensive overview of all registered servers and all deployed agents on different devices, as shown in Appendix B.4.

The Agent-Manager also simplifies the management and monitoring processes for conducting multiple simulations. Since the servers can be reused in different simulations, users don't need to restart the distributed servers between two simulations. Users can efficiently configure, launch, and terminate servers and agents during the simulations as needed. With such a design, we streamline the management process by focusing on servers rather than individual agents, thereby improving the efficiency and effectiveness of managing large-scale agent systems.

# 4 EXPERIMENTS

In this section, we conduct large-scale simulations to show the improvements and advances brought by the proposed infrastructure and components. Meanwhile, we provide detailed observations and in-depth discussions on the agents' collective and individual behaviors, drawing valuable insights.

# 4.1 SETTINGS

We set up a large number of agents to participate in the classic game guess the  $\frac{2}{3}$  of the average, where each agent reports a real number between 0 and 100 and the agent who reports a number closest to  $\frac{2}{3}$  of the average of all the reported numbers wins the game. In this game, intuitively the highest possible average is 100. Therefore, for winning the game, agents tend to report a number no larger than  $100 \times \frac{2}{3} = 66\frac{2}{3}$ . Once all agents adopt this strategy,  $66\frac{2}{3}$  becomes the new highest possible average and thus they tend to report a number no larger than  $66\frac{2}{3} \times \frac{2}{3} = 44\frac{4}{9}$ . This process continues until the average becomes 0 and all agents report 0, indicating that the game has reached its Nash equilibrium. However, considering that agents may not always be rational, those agents who report 0 cannot always win the game since the average does not converge to 0 immediately. Agents should carefully take into account the possible actions of others before reporting their numbers. Meanwhile, agents can adjust their strategies in a multi-round game according to the average reported numbers in previous rounds. Note that all the experiments in this section follow the aforementioned settings.

Devices & LLMs The experiments are conducted on a cluster containing multiple devices, each equipped with 8 A100-80G GPUs, a 64-core CPU, and 1 TB of memory. We adopt vLLM (Kwon et al., 2023) as the LLM inference engine to handle highly concurrent LLM service requests. We utilize six powerful and popular open-source LLMs of different sizes, including Llama3-8B / Llama3-70B (Meta, 2024), Qwen2-7B / Qwen2-72B (Yang et al., 2024a), and MistralAI-8x7B / MistralAI-8x22B (MistralAI, 2024). We adopt their instruction versions due to their enhanced ability to follow instructions. The details of the adopted LLMs and implementations can be found in Appendix C.

System Prompts We provide system prompts for agents to guide them in defining their dialogue style, background knowledge, task requirements, and so on. To be more specific, for playing this game, the system prompt incorporates the game rules and response formats, as illustrated in Prompt 1. Besides, we can include further behavioral guidance

# Prompt 1

You are playing a multiplayer game.

# Game Rule

1. Each player reports a real number between 0 and 100, inclusive.  
2. The winner will be the player whose number is the closest to  $2/3$  of the average of all reported numbers.

Directly report your number without additional information.

# Prompt 2

You are playing a multiplayer game.

# Game Rule

1. Each player reports a real number between 0 and 100, inclusive.  
2. The winner will be the player whose number is the closest to  $2/3$  of the average of all reported numbers.

Think step by step and then report your number.

![](images/f0c8e3e6ef1bc0d7c275c1ae3ca46f6b6c96fca3fb470f41f489f57f5a304d6c.jpg)  
(a) Varying scales of agents

![](images/6bb8d9f033ff80bf7bcca2046fb96958b212b382fcc1580be668edc357d00fbe.jpg)  
Figure 2: Agent-based simulations with varying scales of agents (a, b) and varying numbers of devices (c).  
(b) Varying scales of agents

![](images/723f6cac3830a54c911c77129f433f9bb93bf87a17d515c195a20bc98b46828e.jpg)  
(c) Varying numbers of devices

in the system prompts to encourage behaviors that more closely resemble those of real human beings. For example, inspired by "chain-of-thought" studies (Wei et al., 2022; Wang et al., 2023), we ask agents to think step by step before reporting their numbers, producing the system prompt shown in Prompt 2.

# 4.2 SCALABILITY AND EFFICIENCY

First of all, we conduct a series of experiments to show the scalability and efficiency of the agent-based simulations supported by the proposed actor-based distributed mechanism (see Sec. 3.1). Specifically, we illustrate how the overall simulation running time changes as the number of participating agents grows when using LLMs of different sizes, including Llama3-8B and Llama3-70B. In addition to the model sizes, the system prompt provided to agents is also a factor that can influence the running time, since some prompts (e.g., Prompt 2) may encourage agents to generate longer responses and thereby lead to longer response time. From the experimental results shown in Fig. 2, we can obtain the following observations and insights.

(i) We support an agent-based simulation involving 1 million agents, which can be completed in 12 minutes using 4 devices. In Fig. 2a, we fix the device number to 4 and record the simulation running time as the number of agents grows from 100 to 1M. It can be observed that the simulation involving 1 million agents finishes in 12 minutes when using Llama3-8B with Prompt 1, while it takes 85 minutes if we choose Prompt 2, as the number of averaged response tokens grows by more than 150-fold<sup>1</sup>. For the heaviest inference workload, i.e., when agents adopt Llama3-70B and Prompt 2, it takes around 10.6 hours to complete the simulation.  
(ii) The proposed actor-based distributed mechanism significantly improves the efficiency of large-scale agent-based simulations. To better demonstrate the improvements brought by the proposed actor-based distributed mechanism, we adopt a dummy model request (i.e., agents sleep for 1 second and generate random numbers rather than posting the requests) in the simulation to remove the impact of the LLM inference speed. The experimental results summarized in Fig. 2b show that, completing an agent-based simulation with the proposed actor-based distributed mechanism involving 1 million agents only takes 40 seconds, whereas simulations using serial execution or asynchronous mode in Python (adopted by existing works (Wu et al., 2023; Hong et al., 2024b)) require around 12 days and 8.6 hours, respectively.  
(iii) Increasing the number of devices can proportionally reduce the simulation running time. As shown in Fig. 2c, we maintain the number of agents at 10,000 and vary the number of devices used in the simulation. For Llama3-70B with Prompt 2, the simulation running time decreases from 22 minutes to 5.6 minutes as the number of devices increases from 1 to 4. Such a phenomenon can be attributed to a reduction in the number of agents served within one device, which demonstrates the horizontal scalability of the proposed framework.

![](images/716b8bf7e73ad2be26580e69a830377555053850df869651dc895322cf732250.jpg)  
Figure 3: The distributions of numbers reported by agents with different LLMs and system prompts.

![](images/b11fe4413ecc704f2036d203c5f2a74b0b7adbda419f7ea41fe08253b4ff9037.jpg)

![](images/08623ea9a7ffccea0b462a8b03e394e3413178ea359d80b8b436b830a457e885.jpg)

![](images/8854f952e497aebc1e285be6d588e8178a16c659eef92bd96b8cee7369153593.jpg)

![](images/21d7ca36ea3c9fdac4680d3dd0975c9fce8d376aa7ef12b3c471ee8c83c98fef.jpg)  
Figure 4: The average of reported numbers in a multi-round game.

![](images/528f05b1550ee4a5ddd0898b4d406b35921e0377391af95b4fbedbb835491bf0.jpg)

![](images/5e347487ea9266698cb7b7b875fcb2c6aae4f7da249e39f3fb9e962d52a883fe.jpg)

![](images/c4d100170fbc7c423ca609e052a66b8342513f91bd0028cd77dab385c07c0c83.jpg)

In summary, the proposed actor-based distributed mechanism enhances the efficiency of very large-scale simulations, and offers great scalability by allowing users to expand the scale of agents from the addition of devices.

# 4.3 SIMULATION RESULTS AND ANALYSIS

In this subsection, we add some detailed information with six LLMs and two system prompts. We summarize the experimental results in Fig. 3, from which we derive the insights below.

From the comparisons in the figures, we observe that when utilizing a basic system prompt Prompt 1 for most LLMs, agents generally tend to report numbers around 50. However, it is worth noting that agents with MistralAI-8×7B and MistralAI-8×22B, report smaller numbers (36.63 and 31.69 in average, respectively) than other agents. These results indicate that without providing specific instructions in the system prompt, the performance of agents can be different due to the LLMs they adopt, influenced by factors such as model sizes and model architectures.

When we change the system prompt to a chain-of-thought prompt (i.e., Prompt 2), the reported numbers of agents move forward zero markedly, with  $\frac{2}{3}$  of the average being much smaller than those using Prompt 1, e.g., decreasing from 33.70 to 12.76 when using Qwen2-72B. Meanwhile, we notice that more than  $30\%$  of agents using MistralAI-8  $\times$  22B report around 0, leading to the Nash equilibrium of this game. These observations in the simulations demonstrate the effectiveness and importance of providing suitable system prompts for guiding agents to perform the thought processes.

Moving forward, we expand the simulation to multiple rounds. We inform agents of the winner number in the previous round at the beginning of each round except the first one, and request each agent to report a number at each round. Informing these winner numbers to agents enables them to adjust their strategies accordingly. Such a process is implemented based on the agent-environment interaction mechanism (see Sec. 3.2), with the winning numbers being set as shared states in the environment.

The experimental results are demonstrated in Fig. 4. From these results we can observe that, as the game progresses from round to round, the reported numbers of agents gradually converge to 0, indicating that agents have a good understanding of this game and are capable of considering other agents' behaviors and making rational decisions. Similarly, using the chain-of-thought prompt can accelerate the game to reach its Nash equilibrium. For example, in the fifth round, the average reported number of agents using Qwen2-72B with Prompt 2 is 2.02, which is significantly smaller than those using Prompt 1, reporting 25.16.

It is worth noting that these experimental results are consistent with previous studies (Nagel, 1995; Camerer et al., 2004) in social simulation, which confirms the reliability and potential of multi-agent-based simulations.

For the above experiments, we provide more detailed results (e.g., distributions and statistics of the reported numbers), individual-level observations, and case studies in Appendix E.1 and E.2. Besides, to further explore the impact of behavioral guidance on agents, we incorporate more detailed instructions tailored for this game in the system prompts, and summarize the experimental results in Appendix E.3.

![](images/e3e9942c8c390ee25cb9875766afd81234c8744ca8465048629094d17c7d567c.jpg)  
Figure 5: The distributions of numbers reported by agents characterized by different educational levels.

![](images/62fd9073aba780d9440256701faf7da07f2dc3bdf2be5d58e985cf76fe088034.jpg)

![](images/e1fc54b0a539ca380a569737fb5b443cb14921bb7c0a2207d0909af131cb8452.jpg)

# 4.4 DIVERSE BACKGROUND SETTINGS

The diversity of agents is a crucial factor in agent-based simulations. In Sec. 3.3, we introduce a configurable tool and background generation pipeline designed to automatically instantiate agents with varied background settings. Utilizing these components, we conduct simulation experiments that incorporate agents with diverse background settings.

Specifically, we divide the agents into several groups, each of which consists of 200 agents. We manually provide a basic configuration for each group, and utilize LLMs to generate a detailed description for agents within the group. These generated background settings are added to the system prompts and labeled as "character background". We set up a simulation experiment involving agents with different educational levels, in which we assign five different educational levels to agents, including elementary school, high school, bachelor, master, and Ph.D.

The simulation results are illustrated in Fig. 5, from which we can derive the following insights. In general, the higher the educational level of agents, the lower the average reported numbers, indicating more rational behaviors in this game. For example, when using Qwen2-72B, the average numbers reported by agents with a Ph.D. education are much lower than those reported by agents characterized as primary school students. Meanwhile, from the individual-level behaviors, we notice that agents can effectively perform reasoning processes and make corresponding decisions according to the assigned roles. Besides, different LLMs demonstrate varying sensitivities to educational levels in the background settings. For example, MistralAI-  $8\times 22$  has the least sensitivity, with the largest difference in average reported numbers is 3.49 (32.73 v.s. 29.24), while that of Llama3-70B and Qwen2-72B is 5.72 and 8.24, respectively.

The prompt used to generate background settings, and case studies of the generated results and agent behaviors can be found in Appendix F.1 and F.2. We also conduct a simulation experiment involving agents with different occupations. The experimental results shown in Appendix F.3 further confirm that agents' considerations and actions are consistent with their respective occupations.

# 4.5 MIXTURE OF LLMS

In this subsection, we conduct a simulation experiment involving agents employing a mixture of LLMs. Specifically, we configure agents employing Llama3-70B, MistralAI-8×22B, and Qwen2-72B, with 500 agents assigned to each LLM. We conduct both individual-level simulations, where each agent plays the game independently, and group-level simulations, where agents using the same LLMs form a group.

Individual-Level Simulations The simulation results are illustrated in Fig. 6. At the first round of the game, we observe that agents with Llama3-70B exhibit similar behaviors, tending to report numbers around 33, while agents with MistralAI-8×22B consistently report 0. On the other hand, agents with Qwen2-72B exhibit more diverse behaviors, reporting a wider range of numbers, with most of them falling between 0 and 50. These behaviors can be attributed to the preferences of LLMs, which may be related to their architectures, training corpus, etc.

As the game progresses round by round, agents are informed of the winning number from the previous round and adjust their strategies accordingly. As shown in Fig. 6, the majority of agents report numbers close to the winning number in the previous round, with approximately  $59.7\%$  reporting numbers smaller than the previous winning number. We present a typical response in Appendix G, where an agent adopts a conservative strategy and chooses a number slightly smaller than the winner number 15.90.

In the pie chart of Fig. 6, we show the winners of each round in the simulation, grouped by their employed LLMs. To reduce the randomness in the simulation, we regard those agents whose reported numbers fall within the range of  $\pm 0.5$  from  $\frac{2}{3}$  of the average as the winners. The figure shows that in the first and fifth rounds, agents equipped with Qwen2-72B outperform other agents, while agents equipped with MistralAI-  $8\times 22\mathrm{B}$  emerge as winners in the second,

![](images/517e9c6d910536d1181860d31817db004bd75d73daaf49b3389c86559ce33547.jpg)  
Figure 6: Individual-level simulations involving agents with a mixture of LLMs.

![](images/7c86ab4483b296943de553f9b5766c2aa9d13cf36b1ad551b8c17791447ae4c5.jpg)

![](images/fbc056b98cf49a09f551551858161ad4a8598afcf1fcd93e4f2b1b87849cb3ca.jpg)

![](images/f57333cbd83c43874862dc3968d0fd7dc6768929eb9130e7e66586d8a50e66c9.jpg)

![](images/4c5aa3125713513828a1c51dc583cc0208a426cdcce6a828917fcf1c3539d217.jpg)

![](images/61d21fd387968004d2d3b0df8b755f4e9c9ce9f220e02c73accfd5abc099fecb.jpg)  
Figure 7: Group-level simulations involving agents with a mixture of LLMs.

![](images/53f7184d40aef71183ae11b5e8c370dd545aac9bb1c5e126745552f45fff2ce5.jpg)

![](images/371eae1ebb126c12f44593fa6d3b4391e7fbfd77fd8255ff98a74dbce5fbc82f.jpg)

![](images/eec0e16a241b8bb88a5aa752f69ab1718febe1140b34b69b44e558a98f3b72be.jpg)

![](images/47a162a800161c1269cd1897f1379a590fd58626befe99952fca292f4c23c808.jpg)

third, and fourth rounds. Notably, almost all agents tend to report numbers near 0 in the final round, indicating agents can perform reasonable considerations and behaviors to promote this game approach to its Nash equilibrium.

Group-Level Simulations In a group-level simulation, agents are divided into three groups, and each agent reports a number. The average number of agents within each group is regarded as the reported number of this group. Finally, the group that reports a number that is closest to the  $\frac{2}{3}$  of the average among the groups' reported numbers wins the game.

Meanwhile, we modify the information announced between different rounds in a multi-round game. Starting from the second round, in addition to the winning number of the previous round, all agents are informed of the reported numbers from all three groups in the previous round. This provides additional guidance for agents to adjust their strategies. Such a group-wise synchronization is implemented based on the agent-environment interaction mechanism (see Sec. 3.2), allowing agents within the same group to share an interactive environment for synchronization.

The simulation results are shown in Fig. 7. From the figures, it can be observed that agents within the same group quickly converge to similar behaviors when the game comes to the second round, as indicated by their reported numbers falling within a narrow range. Agents using Qwen2-72B and Llama3-70B exhibit relatively consistent behaviors, while some agents using MistralAI-8×22B might exhibit different behaviors, such as reporting larger numbers.

The system prompt used in the group-level simulations is shown in Appendix G, in which we also provide several examples of agents' behaviors for better understanding. These examples show how agents consider the reported numbers from other groups and strategically choose the reported numbers to benefit their own group. Such a phenomenon confirms that agents can perform reasonable thoughts and actions to help achieve a collective goal.

# 4.6 FURTHER DISCUSSIONS

We provide further discussions on usage tips and open questions when conducting large-scale agent-based simulations.

Impact of the Prior Knowledge of LLMs As "guess the  $\frac{2}{3}$  of the average" is a classic game, it is not surprising that LLMs might have acquired prior knowledge from their training corpus. To measure the impact of this prior knowledge, we change the ratio from  $\frac{2}{3}$  to  $\frac{1}{2}$  and  $\frac{51}{100}$ , and then conduct the simulation experiments, respectively. Note that changing the ratios does not alter the fundamental nature of this game, and as a result, the behaviors (such as the reasoning process) of the participating agents are expected to remain similar if LLMs indeed understand the game.

The experimental results are shown in Fig. 8, from which we can observe that there are significant differences in the agents' performance when the ratio is set to  $\frac{1}{2}$  and  $\frac{51}{100}$ , although both cases should be very similar. We observe that more agents tend to report large numbers (e.g., around 50) when the ratio is  $\frac{51}{100}$  compared to the scenario with a ratio of  $\frac{1}{2}$ , which indicates that some agents might not follow the game when setting the ratio to  $\frac{51}{100}$ . In response, we add a note into system prompts to encourage LLMs to draw from the classic game, stating This game is a variation of the famous "guess the  $2/3$  of the average" game. The results summarized in Fig. 8, denoted as “ $\frac{51}{100} + \text{note}$ ”, show that the winning number decreases from 11.85 to 6.46, aligning more closely with that of using the ratio  $\frac{1}{2}$ , reporting 6.21. These experiments highlight the impact of LLMs' prior knowledge, and the effects of using a prompt to explicitly guide the agents and help them understand the settings of simulations.

![](images/a0c941d1f88bd444ead6c9cafce4e9433430589cd82d9f6dfa3361aa28411149.jpg)  
Figure 8: The distributions of reported numbers when setting different ratios in the game. We use “+note” to denote that we add a note to the system prompts.

![](images/ec0c2141102ac446220dbf9a163556acb7b36b2ce2fe76e1e5e7af49643a7f79.jpg)

![](images/9320df3d506984d559ba9405479c8cf8868ebb6538d40b6c42441c4463bd2132.jpg)

![](images/24b3ccd3b921bdee2f77d54fba54c8a3634da8dd9ef518c6bd44a7e800f779b7.jpg)  
Figure 9: The distributions of reported numbers when we change the Nash equilibrium to 10 in the simulation.

![](images/63ec10d36a333e9c2095752f5d379710a3f9c2a46f2c6081596eebbd3aa75386.jpg)

![](images/257d65d0d7f48eb08e30a72dc6d525f8bafd60fa917838afc71dbcf8fd65e4c2.jpg)

When the Nash Equilibrium Is Not 0 We also set up another variant of the game to validate the agents' capabilities in understanding and reasoning. Specifically, we modify the winning criteria so that the Nash equilibrium becomes 10, instead of 0 as in the classic game. As a result, LLMs might not exhibit reasonable thoughts and behaviors if they have a limited understanding of this game or have poor reasoning ability. The adopted prompt is shown in Appendix H.1.

As shown in Fig. 9, we observe that some agents using the Qwen2-72B and MistralAI-8×22B are able to reason out the Nash equilibrium point in the first round. As the game progresses, the numbers reported by these agents gradually approach 10, demonstrating their understanding of this game and ability to make reasonable decisions.

The above findings are further confirmed by observations of individual-level behaviors shown in Appendix H.1. Besides, from these observations, we can identify some typical mistakes made by agents. For example, some agents might make simple calculation errors, such as calculating a wrong average value, and leading to incorrect results despite having a correct reasoning process. Some agents may follow a logical step-by-step process (e.g., calculating the Nash equilibrium) but still make incorrect decisions (e.g., directly reporting 0). But even so, these mistakes are infrequent and do not affect the overall conclusions drawn from group-level observations, as we previously demonstrated.

These observations indicate that although agents may make simple calculation errors (especially in decimal calculations), they exhibit powerful reasoning abilities and show great potential for usage in large-scale simulations. More experimental results and further discussions regarding the impact of temperature and the role-playing ability of LLMs can be found in Appendix H.2 and H.3, respectively.

# 5 CONCLUSIONS

In this paper, we first discuss several key factors of concern for conducting large-scale agent-based simulations, including scalability and efficiency, population distribution and agent diversity, and ease of management. Motivated by these factors, we propose and implement several enhancements based on AgentScope, including an actor-based distributed mechanism that provides agent-level parallel execution and automatic workflow conversion, the flexible environment support to simulate various real-world scenarios, the heterogeneous configurations that allow users to specify population distributions and to automatically generate agents with diverse background settings, and a web-based interface to simplify the management of large-scale agents. These enhancements make the framework more flexible and convenient for supporting large-scale agent-based simulations. We conduct a series of simulation experiments with the proposed framework and provide detailed observations on the diverse and realistic behaviors of agents, highlighting its great potential to further advance research and applications in agent-based simulations. We will release the source code to inspire further research and development in large-scale multi-agent simulations.

# REFERENCES

Gul A. Agha. Actors: a Model of Concurrent Computation in Distributed Systems (Parallel Processing, Semantics, Open, Programming Languages, Artificial Intelligence). PhD thesis, University of Michigan, USA, 1985.  
Amey Agrawal, Nitin Kedia, Jayashree Mohan, Ashish Panwar, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee, and Alexey Tumanov. VIDUR: A large-scale simulation framework for LLM inference. In  $MLSys$ , 2024.  
ANTHROP. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet, 2024.  
Randall D Beer. A dynamical systems perspective on agent-environment interaction. Artificial intelligence, 72(1-2): 173-215, 1995.  
Colin F Camerer, Teck-Hua Ho, and Juin-Kuan Chong. A cognitive hierarchy model of games. The Quarterly Journal of Economics, 119(3):861-898, 2004.  
Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In ICLR, 2024a.  
Weize Chen, Ziming You, Ran Li, Yitong Guan, Chen Qian, Chenyang Zhao, Cheng Yang, Ruobing Xie, Zhiyuan Liu, and Maosong Sun. Internet of agents: Weaving a web of heterogeneous agents for collaborative intelligence. CoRR, abs/2407.07061, 2024b.  
Myra Cheng, Tiziano Piccardi, and Diyi Yang. Compost: Characterizing and evaluating caricature in LLM simulations. In EMNLP, pp. 10853-10875, 2023.  
Chen Gao, Xiaochong Lan, Zhihong Lu, Jinzhu Mao, Jinghua Piao, Huandong Wang, Depeng Jin, and Yong Li. S<sup>3</sup>: Social-network simulation system with large language model-empowered agents. CoRR, abs/2307.14984, 2023.  
Dawei Gao, Zitao Li, Xuchen Pan, Weirui Kuang, Zhijian Ma, Bingchen Qian, Fei Wei, Wenhao Zhang, Yuexiang Xie, Daoyuan Chen, Liuyi Yao, Hongyi Peng, Zeyu Zhang, Lin Zhu, Chen Cheng, Hongzhu Shi, Yaliang Li, Bolin Ding, and Jingren Zhou. Agentscope: A flexible yet robust multi-agent platform. CoRR, abs/2402.14034, 2024.  
Önder Gürcan. Llm-augmented agent-based modelling for social simulations: Challenges and opportunities. In HHAI, volume 386, pp. 134-144, 2024.  
Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Wenyi Wang, Xiangru Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Zongze Xu, and Chenglin Wu. Data interpreter: An LLM agent for data science. CoRR, abs/2402.18679, 2024a.  
Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for A multi-agent collaborative framework. In ICLR, 2024b.  
Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, and Jiangtao Gong. Surrealdriver: Designing generative driver agent simulation framework in urban contexts based on large language model. CoRR, abs/2309.13193, 2023.  
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In SOSP, pp. 611-626, 2023.  
langchain ai. Langchain. https://github.com/langchain-ai/langchain, 2024a.  
langchain ai. Langsmith. https://docs.smith.langchain.com/, 2024b.  
Chenliang Li, He Chen, Ming Yan, Weizhou Shen, Haiyang Xu, Zhikai Wu, Zhicheng Zhang, Wenmeng Zhou, Yingda Chen, Chen Cheng, Hongzhu Shi, Ji Zhang, Fei Huang, and Jingren Zhou. Modelscope-agent: Building your customizable agent system with open-source large language models. In EMNLP (Demos), pp. 566-578, 2023a.  
Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: communicative agents for "mind" exploration of large language model society. In NeurIPS, 2023b.

Charles M. Macal. Everything you need to know about agent-based modelling and simulation. J. Simulation, 10(2): 144-156, 2016.  
Charles M. Macal and Michael J. North. Tutorial on agent-based modelling and simulation. J. Simulation, 4(3):151-162, 2010.  
Takashi Matsumoto, Yurie Kibi, and Tetsuro Kondo. Purpose model simulation - purpose formation of multi-stakeholder by dialog with lIm-based AI. In HCI, volume 14718, pp. 110-129, 2024.  
Meta. Introducing meta llama3: The most capable openly available llm to date. https://ai.meta.com/blog/meta-llama-3/, 2024.  
MistralAI. Bienvenue to mistral ai documentation. https://docs.mistral.ai/getting-started/models/, 2024.  
Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard Liaw, Eric Liang, William Paul, Michael I. Jordan, and Ion Stoica. Ray: A distributed framework for emerging AI applications. CoRR, abs/1712.05889, 2017.  
Xinyi Mou, Zhongyu Wei, and Xuanjing Huang. Unveiling the truth and facilitating change: Towards agent-based large-scale social movement simulation. CoRR, abs/2402.16333, 2024.  
Rosemarie Nagel. Unraveling in guessing games: An experimental study. The American Economic Review, 85(5): 1313-1326, 1995.  
OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.  
Joon Sung Park, Joseph C. O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein.  
Generative agents: Interactive simulacra of human behavior. In UIST, pp. 2:1-2:22, 2023.  
Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. Chatdev: Communicative agents for software development. In ACL, pp. 15174-15186, 2024.  
Ruiyang Ren, Peng Qiu, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Hua Wu, Ji-Rong Wen, and Haifeng Wang. BASES: large-scale web search user simulation with large language model based agents. CoRR, abs/2402.17505, 2024.  
Significant-Gravitas. Autogpt. https://github.com/Significant-Gravitas/AutoGPT, 2023.  
Aleksandra Sorokovikova, Natalia Fedorova, Sharwin Rezagholi, and Ivan P. Yamshchikov. Llms simulate big five personality traits: Further evidence. CoRR, abs/2402.01765, 2024.  
Karthik Sreedhar and Lydia B. Chilton. Simulating human strategic behavior: Comparing single and multi-agent llms. CoRR, abs/2402.08189, 2024.  
SIMA Team, Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, Stephanie C. Y. Chan, Jeff Clune, Adrian Collister, Vikki Copeman, Alex Cullum, Ishita Dasgupta, Dario de Cesare, Julia Di Trapani, Yani Donchev, Emma Dunleavy, Martin Engelcke, Ryan Faulkner, Frankie Garcia, Charles Gbadamosi, Zhitao Gong, Lucy Gonzalez, Kshitij Gupta, Karol Gregor, Arne Olav Hallingstad, Tim Harley, Sam Haves, Felix Hill, Ed Hirst, Drew A. Hudson, Jony Hudson, Steph Hughes-Fitt, Danilo J. Rezende, Mimi Jasarevic, Laura Kampis, Nan Rosemary Ke, Thomas Keck, Junkyung Kim, Oscar Knagg, Kavya Kopparapu, Andrew K. Lampinen, Shane Legg, Alexander Lerchner, Marjorie Limont, Yulan Liu, Maria Loks-Thompson, Joseph Marino, Kathryn Martin Cussons, Loic Matthew, Siobhan Mcloughlin, Piermaria Mendolicchio, Hamza Merzic, Anna Mitenkova, Alexandre Moufarek, Valeria Oliveira, Yanko Gitahy Oliveira, Hannah Openshaw, Renke Pan, Aneesh Pappu, Alex Platonov, Ollie Purkiss, David P. Reichert, John Reid, Pierre Harvey Richemond, Tyson Roberts, Giles Ruscoe, Jaume Sanchez Elias, Tasha Sandars, Daniel P. Sawyer, Tim Scholtes, Guy Simmons, Daniel Slater, Hubert Soyer, Heiko Strathmann, Peter Stys, Allison C. Tam, Denis Teptyashin, Tayfun Terzi, Davide Vercelli, Bojan Vujatovic, Marcus Wainwright, Jane X. Wang, Zhengdong Wang, Daan Wierstra, Duncan Williams, Nathaniel Wong, Sarah York, and Nick Young. Scaling instructable agents across many simulated worlds. CoRR, abs/2404.10179, 2024.  
XAgent Team. Xagent: An autonomous agent for complex task solving, 2023.  
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In ICLR, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierrick Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In EMNLP (Demos), pp. 38-45, 2020.  
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen LLM applications via multi-agent conversation framework. CoRR, abs/2308.08155, 2023.  
Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Dingwen Yang, Chenyang Liao, Xin Guo, Wei He, Songyang Gao, Lu Chen, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. Agentgym: Evolving large language model-based agents across diverse environments. CoRR, abs/2406.04151, 2024.  
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. CoRR, abs/2407.10671, 2024a.  
Zonghan Yang, An Liu, Zijun Liu, Kaiming Liu, Fangzhou Xiong, Yile Wang, Zeyuan Yang, Qingyuan Hu, Xinrui Chen, Zhenhe Zhang, Fuwen Luo, Zhicheng Guo, Peng Li, and Yang Liu. Position: Towards unified alignment between agents, humans, and environment. In ICML, 2024b.  
Frank Tian-fang Ye and Xiaozi Gao. Simulating family conversations using llms: Demonstration of parenting styles. CoRR, abs/2403.06144, 2024.  
Murong Yue, Wijdane Mifdal, Yixuan Zhang, Jennifer Suh, and Ziyu Yao. Mathvc: An llm-simulated multi-character virtual classroom for mathematics education. CoRR, abs/2404.06711, 2024.  
Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Guiming Chen, Jianquan Li, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, Xiang Wan, Benyou Wang, and Haizhou Li. Huatuogpt, towards taming language model to be a doctor. In EMNLP (Findings), pp. 10859-10885, 2023.  
Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. SOTOPIA: interactive evaluation for social intelligence in language agents. In ICLR, 2024.
