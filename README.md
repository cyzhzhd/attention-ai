## 얼굴 데이터 기반 학생 분석 엔진 개발

- 얼굴 탐지 모델 개발 관련: [링크](https://13.125.91.162/swmaestro/183-2/tree/master/detect_model) 참고

### How To

```
yarn
npm run build
npm start
```

### Dependencies

```
npm v6.14.5
yarn v1.22.4
nodejs v14.5.0
```

### Scripts

- /test 폴더에는 Widerface, FDDB 테스트 데이터 생성을 위한 스크립트 존재.

```
node fddb.js / widerface.js
```

### Upstream task

- 얼굴 탐지, 전처리
- 얼굴 인식 / 랜드마크 탐지 / 표정 분류

#### 요구사항

- On-device inference목표로 **경량 모델** 요구.
- 카메라별로 다른 해상도뿐만 아니라 광도, 조도 등의 환경 요소 변화에도 **강인함** 요구.

---

### Downstream task

- 얻은 피처로 관심이 필요한 학생 탐지
- 얻은 피처를 종합적으로 활용하여 분류 혹은 회귀(점수 매김)

#### 요구사항

- 학생별로 다른 카메라 설치 위치(거리 및 각도)에 적응할 수 있어야 함.
- 학생의 얼굴형, 피부색을 포함하여 안경 등의 착용 기구에 영향을 적게 받아야 함.
- 구체적인 모델 평가방법과 구현 방법을 제시해야함.

---

### 제약사항

- 상업 목적 활용이 가능한 얼굴 인식 모델을 사용해야 함.
- 학습 시, 상업 목적 활용이 가능한 데이터셋을 사용해야 함.

---

### Electron 통합 방법론

- [통합 방법론 이슈](https://13.125.91.162/swmaestro/183-2/issues/6)
- [딥러닝 모델간 변환방법](https://13.125.91.162/swmaestro/183-2/-/wikis/%EB%94%A5%EB%9F%AC%EB%8B%9D-framework-%EB%AA%A8%EB%8D%B8%EA%B0%84-%EB%B3%80%ED%99%98%EB%B0%A9%EB%B2%95)

---

### 참고문헌

#### Upstream

- [얼굴 탐지모델 이슈](https://13.125.91.162/swmaestro/183-2/issues/1)
- [객체 인식 관련 위키](https://13.125.91.162/swmaestro/183-2/-/wikis/Object-Detection%EA%B3%BC-YOLO)
- [시각 및 눈(랜드마크) 인지모델 이슈](https://13.125.91.162/swmaestro/183-2/issues/5)

#### Downstream

##### 집중력 엔진 참고 논문
- [비대면 온라인 수업 사례 고찰: 동영상 녹화 및 실시간 화상 수업 중심으로(2020)] (http://www.riss.kr/link?id=A106987038)
- [원격강의의 학습집중도 평가 시스템(2005)] (http://www.riss.kr/link?id=A101434143)
- [이러닝 학습 환경에서 생체신호를 활용한 학습 집중도 측정 방안(2012)] (http://www.dbpia.co.kr.openlib.uos.ac.kr/pdf/pdfView.do?nodeId=NODE07480931&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR)
- [학습의 집중도 향상을 위한 학습자의 얼굴 검출과 분석] (http://m.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=c76bd42a2c0a4704ffe0bdc3ef48d419)
- [학습자 참여를 유도하기 위한 얼굴인식 기반 지능형 e-Learning 시스템] (http://www.ndsl.kr/ndsl/commons/util/ndslOriginalView.do?dbt=JAKO&cn=JAKO200703534315044&oCn=JAKO200703534315044&pageCode=PG11&journal=NJOU00291398)
- [효율적인 이러닝을 위한 학습자 얼굴 인증 기술] (http://www.ndsl.kr/ndsl/search/detail/article/articleSearchResultDetail.do?cn=JAKO201015959407195&SITE=CLICK)
- [Concentration analysis by detecting face features of learners] (https://ieeexplore.ieee.org/abstract/document/7334807)
- [Research on recognition method of learning concentration based on face feature] (https://ieeexplore.ieee.org/abstract/document/8274797)

