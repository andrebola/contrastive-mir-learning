import json
import argparse

from trainer_baselines import Trainer


def main(config_file):
    params = json.load(open(config_file, 'rb'))
    print("Training models with params:")
    print(json.dumps(params, separators=("\n", ": "), indent=4))
    trainer = Trainer(params)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model with W2V self-Attention')
    parser.add_argument('config_file', type=str,
                        help='configuration file for the training')
    args = parser.parse_args()

    main(args.config_file)
