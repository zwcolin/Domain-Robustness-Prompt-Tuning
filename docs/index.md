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

<table>
  <tr>
    <th colspan="2">Configuration</th>
    <th colspan="2">In-Domain</th>
    <th colspan="2">Out-of-Domain</th>
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

# Discussion
### Advantages of Prefix/Prompt Tuning
Prefix/prompt tuning will only train on a small subset of parameters and freeze other parameters, which significantly reduces training costs. Suppose we have many individual tasks but share the same model structure. In that case, prefix/prompt tuning could maintain modularity and save time/space by only adding and deleting prefix/prompt for each task. Beyond that, the inference is more efficient with prefix/prompt settings. Instead of having different models and calling forward multiple times, we can do a single forward pass with batches.

### Limitation in our project
Because of time limitations, we do not perform ablation tests to examine the internal representation of prefix/prompt tokens. However, this is another exciting topic we want to explore in the future. For example, if we could find some patterns in the space of prefix/prompt tokens, we could directly add a prefix/prompt to a pretrained model when a new task comes. This would allow us to obtain a model which has comparable performance to fine-tuned models, but with no extra costs.

