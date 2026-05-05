# ACE: Adaptive Confidence-Based Encoder Freezing

A continual learning framework that uses model confidence as an internal control signal to regulate parameter updates — without replay buffers, stored past data or model expansion.

**Paper:** Accepted for oral presentation at the 2026 IEEE International Conference on Computing Theory and Wireless Communications (CCWC).

## Key Result

**55% relative reduction** in catastrophic forgetting compared to naive fine-tuning on the Split CIFAR-100 class-incremental benchmark.

## Results on Split CIFAR-100 (5 tasks, 20 classes per task)

| Method             | Avg Accuracy | Forgetting |
|------------------|-------------|------------|
| Naive            | 14.18%      | 1.05%      |
| Fixed Freeze (40%)   | 11.72%      | 0.09%      |
| ACE (Ours)       | 13.01%      | 0.39%      |

> Note: absolute accuracy values are low because this is a class-incremental setting with 5 sequential tasks and zero access to previous task data — the standard evaluation regime for this benchmark. ACE achieves lower forgetting while maintaining competitive accuracy compared to baselines.
## How It Works

ACE monitors the model's prediction confidence on incoming data:

- **High confidence ** → freeze lower encoder layers (preserve learned representations)  
- **Low confidence ** → uunfreeze layers to allow adaptation

This creates a dynamic, per-layer stability-plasticity balance that responds to what the model is actually experiencing without requiring replay buffers or external memory.

## Properties

- No replay buffer required
- No task identity required at inference time
- No model expansion
- Works with standard ResNet encoders
- Single hyperparameter (confidence threshold)

## Setup

```bash
git clone https://github.com/sharanyakatna/ace
cd ace
pip install -r requirements.txt
python ace.py
