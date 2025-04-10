# Why is Normalization Necessary for Linear Recommenders? (SIGIR'25)
This is the official code for DAN in the paper "[Why is Normalization Necessary for Linear Recommenders?](https://arxiv.org/abs/2504.05805)", [The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval](https://sigir2025.dei.unipd.it/).

## Introduction
Despite their simplicity, linear autoencoder (LAE)-based models have shown comparable or even better performance with faster inference speed than neural recommender models. However, LAEs face two critical challenges:

1. **Popularity bias**: LAEs tend to recommend popular items excessively
2. **Neighborhood bias**: LAEs overly focus on capturing local item correlations

To address these issues, we propose **Data-Adaptive Normalization (DAN)**, a versatile normalization solution that flexibly controls the popularity and neighborhood biases by adjusting item- and user-side normalization to align with unique dataset characteristics.

Note that a summary of our paper is on [our lab blog](https://dial.skku.edu/blog/2025_dan) (in Korean).

## Requirements
```
pip install -r requirements.txt
```

## How to run
- Run LAE with DAN
```
sh dan_lae.sh
```
- Run EASE with DAN
```
sh dan_ease.sh
```
- Run RLAE with DAN
```
sh dan_rlae.sh
```

## TODO
- [ ] Add strong generalization protocol


## Citation

If you find our work useful for your research, please cite our paper:
```
@inproceedings{park2025dan,
  title={Why is Normalization Necessary for Linear Recommenders?},
  author={Seongmin Park and
          Mincheol Yoon and
          Hye-young Kim and
          Jongwuk Lee},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```
