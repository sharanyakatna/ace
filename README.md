ACE: Adaptive Confidence-Based Encoder Freezing
A continual learning framework that uses model confidence as an internal control signal to regulate parameter updates — without replay buffers, task identifiers, or model expansion.
Key Result: 55% relative reduction in catastrophic forgetting compared to naive SimCLR baselines on CIFAR-10/100 class-incremental benchmarks.
Paper: Accepted for oral presentation at the 2026 IEEE International Conference on Computing Theory and Wireless Communications (CCWC).
How it works
ACE adaptively freezes encoder layers based on model confidence, allowing the network to balance stability and plasticity across sequential tasks. When the model is confident on incoming data, encoder layers are frozen to preserve existing representations. When confidence drops, layers are unfrozen to allow adaptation.
Key properties:

No replay buffer required
No task boundary information required
No model expansion
Works with standard ResNet encoders

Setup
bashgit clone https://github.com/sharanyakatna/ace
cd ace
pip install torch torchvision
python ace.py

Results on CIFAR-100 (10 tasks)
MethodAvg AccuracyForgettingNaive SimCLRbaselinehighACE (ours)+55% relative reduction in forgetting significantly lower


Author: Sharanya Katna — sharanyakatna13@gmail.com
