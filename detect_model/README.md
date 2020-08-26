## 얼굴 탐지 모델 개발

### Scripts

- train.py: 모델 학습, 환경설정은 train_config.ini 파일 참고
- test.py: 이미지가 들어있는 폴더에 대해 테스트 수행
- convert_to_tfjs.py: hdf5 포멧을 tensorflowjs 모델로 변환
- generate_anchors.py: anchor 중심점과 규모 생성 후 저장

### Convert to TFJS

```
    sh convert_to_tfjs [.hdf5 file] [output directory]
```

### References

##### Model & Training Methods

- [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs, Valentin Bazarevsky et al. 19-07](https://arxiv.org/pdf/1907.05047.pdf)
- [SSD: Single Shot MultiBox Detector, Wei Liu et al. 15-12](https://arxiv.org/pdf/1512.02325.pdf)
- [YOLOv4: Optimal Speed and Accuracy of Object Detection, Alexey Bochkovskiy et al. 20-04](https://arxiv.org/pdf/2004.10934.pdf)

##### Augmentation

- [DropBlock: A regularization method for convolutional networks, Golnaz Ghiasi et al. 18-10](https://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks.pdf)
- [Learning Data Augmentation Strategies for Object Detection, Barret Zoph et al. 19-06](https://arxiv.org/pdf/1906.11172.pdf)

### Dataset

- [WIDERface](http://shuoyang1213.me/WIDERFACE/)
- [FDDB](http://vis-www.cs.umass.edu/fddb/)