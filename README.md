## 얼굴 데이터 기반 학생 분석 엔진 개발

### How To
~~~
yarn
npm run build
npm start
~~~

### Upstream task
* 얼굴 탐지, 전처리
* 얼굴 인식 / 랜드마크 탐지 / 표정 분류
* (Optional) 학생 행동 인식, 시선 추적(eye-tracking), 물체탐지

#### 요구사항
* On-device inference목표로 __경량 모델__ 요구.
* 카메라별로 다른 해상도뿐만 아니라 광도, 조도 등의 환경 요소 변화에도 __강인함__ 요구.

------

### Downstream task
* 얻은 피처로 관심이 필요한 학생 탐지
* 얻은 피처를 종합적으로 활용하여 분류 혹은 회귀(점수 매김)

#### 요구사항
* 학생별로 다른 카메라 설치 위치(거리 및 각도)에 적응할 수 있어야 함.
* 학생의 얼굴형, 피부색을 포함하여 안경 등의 착용 기구에 영향을 적게 받아야 함.
* 구체적인 모델 평가방법과 구현 방법을 제시해야함.

------

### 제약사항
* 상업 목적 활용이 가능한 얼굴 인식 모델을 사용해야 함.
* 학습 시, 상업 목적 활용이 가능한 데이터셋을 사용해야 함.

------

### Electron 통합 방법론
* [ONNX](https://onnx.ai/): 머신러닝 모델 상호작동성을 위한 프레임워크
* [TorchJS](https://github.com/torch-js/torch-js): 노드 - 파이토치 바인딩
* [Python-shell](https://www.npmjs.com/package/python-shell): 일렉트론 - 파이썬 IPC 노드 패키지 (파이썬 dependency 문제 해결 필요)

------

### 참고문헌
#### Upstream
* ~~[EXTD - Extremely Tiny Face Detector via Iterative Filter Reuse](https://arxiv.org/abs/1906.06579)  
arXiv 2019 - YoungJoon Yoo et al.~~ Titan X 환경에서 5 ~ 25FPS : 부적합
* [PFLD - a practical facial landmark detector](https://paperswithcode.com/paper/pfld-a-practical-facial-landmark-detector)  
arXiv 2019 - Xiaojie Guo et al.
* [EfficientFAN - deep knowledge transfer for face alignment](https://dl.acm.org/doi/10.1145/3372278.3390692)  
ICMR 2020 - Pengcheng Gao et al.

#### Downstream
* [Concentration analysis by detecting face features of learners](https://ieeexplore.ieee.org/document/7334807)  
IEEE PACRIM 2015 - Seunghui Cha et al.

### Datasets
* TBD

