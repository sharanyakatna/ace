# ACE: Adaptive Confidence-Based Encoder Freezing

A continual learning framework that uses model confidence as an internal control signal to regulate parameter updates — without replay buffers, task identifiers, or model expansion.

**Paper:** Accepted for oral presentation at the 2026 IEEE International Conference on Computing Theory and Wireless Communications (CCWC).

## Key Result

**55% relative reduction** in catastrophic forgetting compared to naive fine-tuning on CIFAR-10/100 class-incremental benchmarks, with simultaneous accuracy improvement.

## How It Works

ACE monitors the model's confidence on incoming data at each layer:

- **High confidence on old data** → freeze lower encoder layers (they're working, don't touch them)
- **Low confidence on new data** → unfreeze layers to allow adaptation

This creates a dynamic, per-layer stability-plasticity balance that responds to what the model is actually experiencing — no external signals needed.

### Properties
- No replay buffer required
- No task boundary information required
- No model expansion
- Works with standard ResNet encoders
- Single hyperparameter (confidence threshold)

## Setup

```bash
git clone https://github.com/sharanyakatna/ace
cd ace
pip install torch torchvision
python ace.py
