# On Evaluating the Robustness of Language Models with Tuning
Authors: Colin Wang, Lechuan Wang, Yutong Luo
## Pipeline for DSC 180B (highly inefficient and only intends for course staff to run code in a "standardized" format)
Build a container using `zwcolin/180_method5:latest` docker. Clone the repo, then at the root folder, run `python run.py test`. Warning: lots of time may be spent on downloading the data, pretrained model, tokenizer, and preparation. The testing itself may take ~30 minutes (not including downloading and building the dataset) to output evaluation metrics (we've modified the script for you so it just measures the first 10 examples, which may take around 30 seconds to intialize and process). If you do want to see some results, you may want to wait for quite a bit. Alternatively, some existing train/testing logging has been provided inside the the `prompt_tuning` folder. You can take a look at that instead of actually running the code.

## Internal Pipeline
### Manipulating Model in `run.py`
#### Training & Testing
- Simply do `bash experiment.sh` and modify any experiment meta info as well as hyperparameter as necessary. The pipeline has been built to suit single/multi GPU configurations under a single server instance.

## Deployment
A `Dockerfile` has been provided in the root folder to set up a docker environment. Note that this dockerfile has only been experimented at UCSD's DataHub. Use it with caution.

## DSC 180B Specific Instructions
We don't strictly follow the structure of the given suggestions, with a `test` folder and a `testdata` folder inside it. It's too rigid. Instead, all the training and test data will be store inside the `data` folder and `experiment.sh` contains all the necessary code to build the model or to test the model for running the model. We don't like the way that you need to run `test.py` with some arguments in the command line because it's obviously not suitable for a deep learning project where there are way many possible arguments. It's much of a formalism thing.

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
