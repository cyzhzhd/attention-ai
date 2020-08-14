## 얼굴 데이터 기반 학생 분석 엔진 개발

### How To
~~~
yarn
npm run build
npm start
~~~

### Dependencies
~~~
npm v6.14.5  
yarn v1.22.4  
nodejs v14.5.0  
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
* [통합 방법론 이슈](https://13.125.91.162/swmaestro/183-2/issues/6)
* [딥러닝 모델간 변환방법](https://13.125.91.162/swmaestro/183-2/-/wikis/%EB%94%A5%EB%9F%AC%EB%8B%9D-framework-%EB%AA%A8%EB%8D%B8%EA%B0%84-%EB%B3%80%ED%99%98%EB%B0%A9%EB%B2%95)

------

### 참고문헌
#### Upstream
* [얼굴 탐지모델 이슈](https://13.125.91.162/swmaestro/183-2/issues/1)
* [객체 인식 관련 위키](https://13.125.91.162/swmaestro/183-2/-/wikis/Object-Detection%EA%B3%BC-YOLO)
* [시각 및 눈(랜드마크) 인지모델 이슈](https://13.125.91.162/swmaestro/183-2/issues/5)  



#### Downstream
* [집중력 엔진 위키](https://13.125.91.162/swmaestro/183-2/-/wikis/faceAPI%EA%B8%B0%EB%B0%98-%ED%94%84%EB%A1%9C%ED%86%A0%ED%83%80%EC%9E%85-%EC%97%94%EC%A7%84-%EA%B5%AC%ED%98%84)
* [집중력 엔진 개발 이슈](https://13.125.91.162/swmaestro/183-2/issues/7)  


### Datasets
* [WIDERface](https://13.125.91.162/swmaestro/183-2/-/wikis/face-api.js-%EB%B2%A4%EC%B9%98%EB%A7%88%ED%82%B9)

