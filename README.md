# Transformer Implementation
A Pytorch Implementation of the paper "Attention is All You Need".   
I checked out several popular implementations and I have found a few points which was quite different from the original paper.   
This repository is the result of fixing errors and cleaning codes in pytorch-OOP manner. 

## Examples
- Trained on 20k Korean-English parallel corpus for two hours with general GPU.
- These test sentences is not from the train corpus.
```
우리 내일 어디로 갈까?
<sos> where should we go tomorrow ? <eos>
너 나 좋아하니?
<sos> do you like to go ? <eos>
이번 시험에서 내가 잘할 수 있을까요?
<sos> can i get a good job this exam ? <eos>
정말 이번에는 졸업하고 싶은데 잘 될지 걱정이야.
<sos> i want to graduate this time , but i 'm worried about you . <eos>
```

## References
- [Attention Is All You Need, Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer, Harvard NLP.](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Attention is all you need: A Pytorch Implementation, Yu-Hsiang Huang](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [A Transformer Implementation of Attention is All You Need, Kyubyoung Park](https://github.com/Kyubyong/transformer)
- [Transformers, Huggingface](https://github.com/huggingface/transformers)

## Author
- This repository is developed and maintained by Yonghee Cheon (yonghee.cheon@gmail.com).      
- It can be found here: https://github.com/yonghee12/transformer_torch
- Linkedin Profile: https://www.linkedin.com/in/yonghee-cheon-7b90b116a/