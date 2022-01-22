# T5_SQuAD_Prompt_Tuning
Authors: Colin Wang, Lechuan Wang, Yutong Luo

## Simple Pipeline
### Manipulating Model in `run.py`
#### Training
- Simply do `bash experiment.sh` and modify any experiment meta info as well as hyperparameter as necessary. The pipeline has been built to suit single/multi GPU configurations under a single server instance.

## Outdated Pipeline
#### Testing
- Usage: python run.py test model_name n_tokens
- `model_name` specifies the pretrained T5 model, which can be `t5-small`, `t5-base`, etc
- `n_tokens` specifies the number of tokens that serve as the soft prompt (a positive integer)
In this mode, the script will load a pretrained T5 model and a fine-tuned soft prompt, and use the model with the prompt to generate answers from questions and context from squadv2 (i.e. serves as a demo).
#### Debug Model
- Usage: python run.py model model_name n_tokens
- `model_name` specifies the pretrained T5 model, which can be `t5-small`, `t5-base`, etc
- `n_tokens` specifies the number of tokens that serve as the soft prompt (a positive integer)

## Experimentation
A notebook, namely `main.ipynb`, has been provided for fast prototyping and experimentation

## Deployment
A `Dockerfile` has been provided in the root folder to set up a docker environment. Note that this dockerfile has only been experimented at UCSD's DataHub. Use it with caution.

## DSC 180A Specific Instructions
We don't strictly follow the structure of the given suggestions, with a `test` folder and a `testdata` folder inside it. It's too rigid. Instead, all the training and test data will be store inside the `data` folder and `run.py` contains all the necessary code to build the model or to test the model for fast prototyping regarding Q1 project's nature.

## Reference
The script is based on the following paper:
@misc{lester2021power,
      title={The Power of Scale for Parameter-Efficient Prompt Tuning}, 
      author={Brian Lester and Rami Al-Rfou and Noah Constant},
      year={2021},
      eprint={2104.08691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
