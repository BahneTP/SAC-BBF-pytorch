# SAC-BBF PyTorch

This repository provides a PyTorch implementation of **SAC-BBF**, based on the paper:

**Generalizing Soft Actor-Critic Algorithms to Discrete Action Spaces**  
Paper: https://arxiv.org/abs/2407.11044  
Original repository: https://github.com/lezhang-thu/bigger-better-faster-SAC

The method builds on ideas from **Bigger, Better, Faster: Human-level Atari with human-level efficiency**:

Paper: https://arxiv.org/abs/2305.19452  
Original repository: https://github.com/google-research/google-research/tree/master/bigger_better_faster

## About

This repository is mainly a PyTorch conversion of the SAC-BBF implementation.  
The goal is to provide a clean and reproducible version of the algorithm in PyTorch for Atari 100k experiments.

## Results

The implementation reaches the reported paper results in almost all tested Atari 100k games.

## References

```bibtex
@article{zhang2024generalizing,
  title={Generalizing Soft Actor-Critic Algorithms to Discrete Action Spaces},
  author={Zhang, Le and others},
  journal={arXiv preprint arXiv:2407.11044},
  year={2024}
}

@article{schwarzer2023bigger,
  title={Bigger, Better, Faster: Human-level Atari with human-level efficiency},
  author={Schwarzer, Max and others},
  journal={arXiv preprint arXiv:2305.19452},
  year={2023}
}
```

## Acknowledgements

This repository is based on the SAC-BBF implementation by Le Zhang et al. and the original Bigger Better Faster codebase from Google Research.
