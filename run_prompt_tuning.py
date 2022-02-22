import argparse
import engine_prompt_tuning

parser = argparse.ArgumentParser()

# meta-information, or args that specific to all tuning methods
parser.add_argument("--logging", default=0, type=int, help="Tuning method being used")
parser.add_argument("--method", default="prompt_tuning", type=str, help="Tuning method being used")
parser.add_argument("--model", default="t5-small", type=str, help="Model being used")
parser.add_argument("--task", default="qa", type=str, help="Task being used")
parser.add_argument("--mode", default="train", type=str, help="Mode being used")
parser.add_argument("--train_set", default="SQuAD", type=str, help="Dataset being used")
parser.add_argument("--val_set", default="SQuAD", type=str, help="Dataset being used")
parser.add_argument("--test_set", default="DuoRC.ParaphraseRC", type=str, help="Dataset being used")
parser.add_argument("-tss", "--test_sets", default=[], nargs='+', help="Dataset being used")
parser.add_argument("--model_dir",default="none",type=str,help="Prompt or prefix being used for testing",)

# specific-to-prompt-tuning
parser.add_argument("--soft_prompt_path", default=None, type=str, help="the path of a tuned soft prompt")
parser.add_argument("--n_tokens", default=11, type=int, help="number of tokens")
parser.add_argument("--initialize_from_vocab",default=True,type=bool,help="if the initial prompt is initialized from existing vocabulary",)
parser.add_argument("--random_range",default=0.5,type=float,help="weight range from a uniform distribution if not initialized from existing vocabulary",)

# hyperparameters for fine-tuning
parser.add_argument("--bz", default=16, type=int, help="batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--epoch", default=4, type=float, help="number of epochs")
parser.add_argument("--optimizer", default="adafactor", type=str, help="which optimizer to use")
parser.add_argument("--clip_threshold", default=1.0, type=float, help="Threshold of root mean square of final gradient update",)
parser.add_argument("--scale_parameter",default=False,type=bool,help="If True, learning rate is scaled by root mean square",)
parser.add_argument("--relative_step",default=False,type=bool,help="If True, time-dependent learning rate is computed instead of external learning rate",)
parser.add_argument("--warmup_init",default=False,type=bool,help="Time-dependent learning rate computation depends on whether warm-up initialization is being used",)
args = vars(parser.parse_args())

def main(args):
    if args['method'] == 'prompt_tuning':
        if args['mode'] == 'train':
            engine_prompt_tuning.run(args)
        if args['mode'] == 'test':
            engine_prompt_tuning.test_model(args)

if __name__ == "__main__":
    main(args)
