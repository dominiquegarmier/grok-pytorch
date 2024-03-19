# grok-pytorch
my best attempt of implementing grok in pytorch

### What is Grok?
grok or grok-1 is a newly [opensourced](https://github.com/xai-org/grok-1) mixture of experts language model by xai.

### Disclaimer
This implementation is not intended to be run.
It is intended to be a reference for understanding the architecture of the grok model, which is also the reason I wrote this.
Personally I also find it easier to reason about a model architecture when shapes are provided via type hints.

### Attributions
- The original implementation of grok in jax and haiku can be found [here](https://github.com/xai-org/grok-1).
- certain parts of the code were adapted from [x-trainsformers](https://github.com/lucidrains/x-transformers) - MIT License, [mema](https://github.com/dominiquegarmier/mema) - MIT License and [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py) - Apache 2.0 License

### Citations
[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
```bibtex
@misc{su2023roformer,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
```bibtex
@misc{vaswani2017attention,
      title={Attention Is All You Need},
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2017},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
```bibtex
@misc{zhang2019root,
      title={Root Mean Square Layer Normalization}, 
      author={Biao Zhang and Rico Sennrich},
      year={2019},
      eprint={1910.07467},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

