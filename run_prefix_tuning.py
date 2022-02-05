import argparse
import engine_prefix_tuning

parser = argparse.ArgumentParser()

# meta-information, or args that specific to all tuning methods
parser.add_argument("--logging", default=0, type=int, help="Tuning method being used")
parser.add_argument(
    "--method", default="prefix_tuning", type=str, help="Tuning method being used"
)
parser.add_argument("--model", default="gpt2-medium", type=str, help="Model being used")
parser.add_argument("--mode", default="train", type=str, help="Mode being used")
parser.add_argument(
    "--train_set", default="webnlg", type=str, help="Dataset being used"
)
parser.add_argument("--test_set", default="webnlg", type=str, help="Dataset being used")

# specific-to-prefix-tuning
parser.add_argument("--preseqlen", default=5, type=int, help="number of tokens")

# hyperparameters for fine-tuning
parser.add_argument("--bz", default=5, type=int, help="batch size")
parser.add_argument("--epoch", default=5, type=float, help="number of epochs")
args = vars(parser.parse_args())


def main(args):
    if args["method"] == "prefix_tuning":
        engine_prefix_tuning.run(args)


if __name__ == "__main__":
    main(args)
