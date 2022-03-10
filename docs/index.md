---
layout: default
---

> NLP models have recently achieved outstanding performances and are thus gained prevalent applications in real world. With this popularity, it is important to make sure these models could adapt well in the dynamic circumstances. More specifically, robustness with respect to domain shifts is supposed to be considered when developing models. Because the same large pretrained language models are often applied to different tasks or fields. It would be inefficient and impractical if we train the model with corresponding inputs every time we apply them to a different domain. We want large models can be easily reused. Improvement on models to ensure they are robust against change of inputs has been a hot topic for study.

# Introduction

Prompt tuning and prefix tuning are two effective mechanisms to leverage frozen language models to perform downstream tasks. Robustness reflects models' resilience of output under a change or noise in the input. In this project, we analyze the robustness of natural language models using various tuning methods with respect to a domain shift (i.e. training on a domain but evaluating on out-of-domain data). We apply both prompt tuning and prefix tuning on T5 models for reading comprehension (i.e. question-answering) and GPT-2 models for table-to-text generation.

# Datasets & Evaluation Metrics
- GPT-2 Table-to-Text generations
  - Train on **[WebNLG](https://aclanthology.org/W16-6626/).**
  - Test on **[DART](https://arxiv.org/abs/2007.02871).**
  - Evaluate with **[BLEU](https://aclanthology.org/P02-1040.pdf).**

- T5 Qustion & Answering
  - Train on **[SQuAD](https://arxiv.org/abs/1606.05250).**
  - Test on **[DuoRC](https://arxiv.org/abs/1804.07927).**
  - Evaluate with **[EM/F1](https://arxiv.org/abs/1910.09753).**

# Methods & Hyperparameters
In our work, we will apply both prompt and prefix tuning on T5 and GPT-2 models. Our experimental design spans two dimensions for each model and tuning method. 

1. We measure the robustness of tuning with respect to different model sizes, given the same prompt length.
2. We measure the robustness of tuning with respect to different prompt lengths, given the same model size. 

We train both T5 and GPT-2 models with sizes range from small, base and large, and with prompt lengths from 1, 5, 10, 20 and 50. The prompts and prefixes are initialized from vocabulary.

### Model Setup
- T5
  - Optimizor: Adafactor
  - Learning Rate: 0.001
  - Scheduler: None 
  - Clip Threshold: 1.0
  - Epochs: 4

- GPT-2
  - Optimizor: AdamW
  - Learning Rate: 0.00005
  - Scheduler: Linear

# Results
### T5 & Question Answering
Table below contains our experimentation results for prompt/prefix tuning on T5 model for question-answer task. 

<table>
  <tr>
    <th colspan="2">Configuration</th>
    <th colspan="2">In-Domain</th>
    <th colspan="2">Out-of-Domain</th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th colspan="2">Prompt</th>
    <th colspan="2">Prefix</th>
  </tr>
  <tr>
    <th>Size</th>
    <th>#Tkns</th>
    <th>EM</th>
    <th>F1</th>
    <th>EM</th>
    <th>F1</th>
  </tr>
  <tr>
    <td rowspan="5">Small</td>
    <td style="font-weight:bold">1</td>
    <td>17.86</td>
    <td>56.88</td>
    <td>2.27</td>
    <td style="font-weight:bold">25.17</td>
  </tr>
  <tr>
    <td>5</td>
    <td>21.52</td>
    <td>55.61</td>
    <td>2.40</td>
    <td>21.48</td>
  </tr>
  <tr>
    <td>10</td>
    <td>21.97</td>
    <td>57.19</td>
    <td>3.06</td>
    <td>23.60</td>
  </tr>
  <tr>
    <td>20</td>
    <td style="font-weight:bold">27.72</td>
    <td style="font-weight:bold">61.08</td>
    <td>3.53</td>
    <td>24.32</td>
  </tr>
  <tr>
    <td>50</td>
    <td>24.34</td>
    <td>60.05</td>
    <td style="font-weight:bold">3.60</td>
    <td>24.76</td>
  </tr>
  
  <tr>
    <td rowspan="5">Base</td>
    <td style="font-weight:bold">1</td>
    <td>55.29</td>
    <td style="font-weight:bold">79.84</td>
    <td style="font-weight:bold">30.71</td>
    <td style="font-weight:bold">49.74</td>
  </tr>
  <tr>
    <td>5</td>
    <td>47.70</td>
    <td>72.44</td>
    <td>18.79</td>
    <td>36.13</td>
  </tr>
  <tr>
    <td>10</td>
    <td>50.09</td>
    <td>73.32</td>
    <td>21.99</td>
    <td>39.44</td>
  </tr>
  <tr>
    <td>20</td>
    <td style="font-weight:bold">55.73</td>
    <td>75.95</td>
    <td>25.98</td>
    <td>42.38</td>
  </tr>
  <tr>
    <td>50</td>
    <td>49.29</td>
    <td>74.23</td>
    <td>16.06</td>
    <td>38.11</td>
  </tr>
  
  <tr>
    <td rowspan="5">Large</td>
    <td style="font-weight:bold">1</td>
    <td style="font-weight:bold">55.65</td>
    <td style="font-weight:bold">82.01</td>
    <td style="font-weight:bold">49.43</td>
    <td style="font-weight:bold">63.77</td>
  </tr>
  <tr>
    <td>5</td>
    <td>49.72</td>
    <td>78.89</td>
    <td>43.84</td>
    <td>61.08</td>
  </tr>
  <tr>
    <td>10</td>
    <td>46.87</td>
    <td>78.33</td>
    <td>46.77</td>
    <td>62.11</td>
  </tr>
  <tr>
    <td>20</td>
    <td>33.67</td>
    <td>73.61</td>
    <td>32.91</td>
    <td>56.63</td>
  </tr>
  <tr>
    <td>50</td>
    <td>40.90</td>
    <td>76.35</td>
    <td>38.97</td>
    <td>59.32</td>
  </tr>
 
</table>

**We have four findings:**
1. Prefix tuning is better in general for smaller models.
2. Prompt tuning seems to be superior than prefix tuning as we get larger and larger model.
3. Prompt tuning’s token choices are model-size agnostic with T5 on question answering.
4. Prompt tuning’s performance diverges significantly when having different number of tokens, while prefix tuning’s performance keeps consistently over different number of tokens for the same-sized model. This shows that determining a prompt length in prompt tuning is more important than determining a prefix length in prefix tuning.


### GPT-2 & Table-to-Text Generation

Table below contains our experimentation results for prompt/prefix tuning on GPT2 model for table-to-text tasks.
<table>
    <tr>
        <th colspan="2">Configuration</th>
        <th colspan="4">In-Domain</th>
        <th colspan="6">Out-of-Domain</th>
    </tr>
    <tr>
        <th></th>
        <th></th>
        <th colspan="2">Prompt</th>
        <th colspan="2">Prefix</th>
        <th colspan="3">Prompt</th>
        <th colspan="3">Prefix</th>
    </tr>
    <tr>
        <th>Size</th>
        <th>#Tkns</th>
        <th>B(S)</th>
        <th>B(U)</th>
        <th>B(S)</th>
        <th>B(U)</th>
        <th>B</th>
        <th>T</th>
        <th>M</th>
        <th>B</th>
        <th>T</th>
        <th>M</th>
    </tr>
    <tr>
        <td rowspan="5">Base</td>
        <td>1</td>
        <td>0</td>
        <td>0</td>
        <td>60.69</td>
        <td>42.14</td>
        <td>0</td>
        <td>0.95</td>
        <td>0.04</td>
        <td>19.45</td>
        <td>0.96</td>
        <td>0.26</td>
    </tr>
    <tr>
        
        <td>5</td>
        <td>30.01</td>
        <td>24.16</td>
        <td>62.51</td>
        <td style="font-weight:bold">45.53</td>
        <td>28.32</td>
        <td>0.66</td>
        <td>0.2</td>
        <td style="font-weight:bold">29.02</td>
        <td>0.75</td>
        <td style="font-weight:bold">0.32</td>
    </tr>
    <tr>
       
        <td>10</td>
        <td>31.91</td>
        <td>26.18</td>
        <td>63.07</td>
        <td>43.16</td>
        <td>26.6</td>
        <td>0.65</td>
        <td>0.25</td>
        <td>28.09</td>
        <td>0.76</td>
        <td style="font-weight:bold">0.32</td>
    </tr>
    <tr>
        
        <td>20</td>
        <td>37.17</td>
        <td>33.8</td>
        <td style="font-weight:bold">63.25</td>
        <td>44.9</td>
        <td>27.91</td>
        <td>0.62</td>
        <td>0.27</td>
        <td>16.45</td>
        <td>1.63</td>
        <td>0.31</td>
    </tr>
    <tr>
        
        <td>50</td>
        <td>38.27</td>
        <td>31.07</td>
        <td>62.6</td>
        <td>44.33</td>
        <td>27</td>
        <td style="font-weight:bold">0.61</td>
        <td>0.26</td>
        <td>20.51</td>
        <td>1.15</td>
        <td style="font-weight:bold">0.32</td>
    </tr>
    <tr>
        <td rowspan="5">Large</td>
        <td>1</td>
        <td>0.69</td>
        <td>0.88</td>
        <td>64.02</td>
        <td>45.91</td>
        <td>0.44</td>
        <td>0.97</td>
        <td>0.04</td>
        <td>22.7</td>
        <td>1</td>
        <td>0.32</td>
    </tr>
    <tr>
    ]
        <td>5</td>
        <td>32.01</td>
        <td>28.07</td>
        <td>63.75</td>
        <td>45.73</td>
        <td>19.77</td>
        <td>0.7</td>
        <td>0.2</td>
        <td>30.35</td>
        <td>0.71</td>
        <td rowspan="5">0.34</td>
    </tr>
    <tr>
       
        <td>10</td>
        <td>35.86</td>
        <td>32.25</td>
        <td>63.97</td>
        <td rowspan="5">47.27</td>
        <td>20.67</td>
        <td>0.8</td>
        <td>0.21</td>
        <td>30.23</td>
        <td>0.71</td>
        <td rowspan="5">0.34</td>
    </tr>
    <tr>

        <td>20</td>
        <td>37.69</td>
        <td>33.57</td>
        <td rowspan="5">64.44</td>
        <td>46.35</td>
        <td>27.22</td>
        <td>0.67</td>
        <td>0.29</td>
        <td>29.98</td>
        <td>0.71</td>
        <td>0.33</td>
    </tr>
    <tr>
   
        <td>50</td>
        <td>40.17</td>
        <td>36.85</td>
        <td>64.23</td>
        <td>46.43</td>
        <td>24.61</td>
        <td>0.79</td>
        <td>0.3</td>
        <td rowspan="5">31.68</td>
        <td rowspan="5">0.65</td>
        <td rowspan="5">0.34</td>
    </tr>
</table>

**We have four findings:**
1. Prefix tuning is superior in all model sizes.
2. There does not seem to have a number of tokens that consistently perform better
3. Although prompt tuning performs better as the model size increases in-domain, it performs worse in a larger model out-of-domain.
4. We obtain similar patterns that prompt tuning’s performance on GPT2 diverges significantly giving different lengths of tokens, while prefix tuning’s performance is relatively stable.


### Common Patterns
The experiments on different lengths of tokens show that prompt tuning's performance diverges when having a different number of tokens. In contrast, prefix tuning's performance consistently keeps over different tokens for the same-sized model. This shows that the prompt length parameter in prompt tuning is critical.

Furthermore, prefix tuning seems to perform better in training and in-domain data. However, it unexpectedly yields worse results in out-of-domain data than prompt tuning when the model size grows. In contrast, prompt tuning performs comparable in out-of-domain to in-domain as the model size increases. This could be caused by overfitting in prefix tuning since it uses more parameters than prompt tuning. Still, both methods outperform fine-tuning in out-of-domain data.

# Discussion
### Advantages of Prefix/Prompt Tuning
Prefix and prompt tuning require much less time and resources to train while still obtaining comparable results to fine-tuning. Both methods only train on a small subset of parameters and freeze other parameters, significantly reducing training costs. Fewer parameters in prompt tuning may allow it to generalize even better in unseen and out-of-domain data. For example, its performance on the training and validation set is very close. 

Prefix and prompt tuning are meaningful in real-life applications. For example, suppose we have many individual tasks but share the same model structure. Prefix and prompt tuning could maintain modularity and save time/space by only adding and deleting prefix/prompt tokens for each task. Beyond that, the inference is more efficient with prefix/prompt settings. Instead of having different models and calling forward multiple times, we can do a single forward pass with batches.

### Limitation in our project
We have several limitations in the scope of this report. The direct comparison between prompt and prefix tuning is not very convincing. The hyperparameters in prompt tuning are not fine-tuned, but hyperparameters in prefix tuning experiments are tuned based on **[Li and Liang, 2021](https://arxiv.org/abs/2101.00190).**. This directly causes prefix tuning to outperform prompt tuning in in-domain data. The implementation details of two methods are also slightly different. The implementation provided by the Prefix-tuning does not work on T5, so we modified the codebase, which may lead to minor discrepancies in implementations. The implementation of prompt tuning was not released when we started this project, so we built our pipeline, which is different from the official codebase. Our pretrained T5 model is also different from the one experimented in **[Lester et al., 2021](https://arxiv.org/abs/2104.08691).**.

Also, we do not perform ablation tests to examine the internal representation of prefix/prompt tokens. This is another exciting topic we want to explore in the future. For example, if we find some patterns in the space of prefix/prompt tokens, we could directly add a prefix/prompt to a pretrained model when a new task comes. This would allow us to obtain a model which has comparable performance to fine-tuned models, but with no extra costs.

# Conclusion
We conclude that prompt tuning is more robust in domain-shift tasks. However, the length of prompt tokens is an important parameter and need to be tuned in different tasks. Because of time and resource limitations, our parameters are not fine tuned and the result is not perfect. We would like to further optimize the performance in in-domain data and see whether the score in out-of-domain also increases and achieves the same level.

On the other hand, prefix tuning does not generalize as good as prompt tuning in out-of-domain data, but its performance in in-domain data is close to the state-of-the-art fine tuning method. Furthermore, the prefix length has small affects in different tasks and model sizes. Hence, prefix tuning could reach fine tuning performance with much fewer parameters, less training time and less fine tuning process.
