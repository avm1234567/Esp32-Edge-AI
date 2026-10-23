"""Export an existing trained checkpoint. Never generate random deployment weights."""
import argparse
import torch
from train_micro_gpt import MicroGPT, export

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='artifacts/best.pt')
    parser.add_argument('--output', default='main')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = MicroGPT(**checkpoint['config'])
    model.load_state_dict(checkpoint['state'])
    export(model.eval(), args.output)
    print('Exported trained weights and matching model_config.h')
