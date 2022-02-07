import sys
import subprocess

if __name__ == "__main__":
    targets = sys.argv[1:]
    if 'train' in targets:
        bashCommand = "python -u run_prompt_tuning.py --mode train"
    if 'test' in targets:
        bashCommand = "python -u run_prompt_tuning.py --mode test"
    process = subprocess.Popen(bashCommand.split(), stdout=sys.stdout, stderr=sys.stderr)
    output, error = process.communicate()
    print()
    print()
    print(output)
    