# 월간 데이콘 항공편 지연 예측 AI 경진대회

https://dacon.io/competitions/official/236094/overview/description

## 23-1학기 빅데이터처리 기말 프로젝트

### 팀원

- 이시내
- 유동혁
- 김명학
- 박준수

---

### Update log:

1. ['Delay_per']의 평균을 내서 상위 50퍼는 Delayed로 간주, ['Delay_num']에 삽입
   - 0.6821020302점 달성.
2. labeled['Delay_num']의 평균을 내서 (약 0.17647), 상위 20%의 ['Delay_per']를 Delayed로 간주
   - 0.7751130567점 달성
3. 전처리 이후 train['Delay_per'](dtype = float)만 이용해서 no tune xgboost 실행
   - 0.7464493418점 달성
4. 전처리 과정 중 unlabeled의 전체를 한번에 채워넣는 것이 아니라, unlabeled를 여러 조각으로 나눠 조금씩 labeled에 합침
   - 0.6857301445점 달성
5. 나눠서 학습하는 과정에서 labeled['Delay_per']를 기준으로 학습하고 전체 평균을 기준으로 train['Delay_num']을 채워넣음
   - 0.6561362148점 달성
6. EDT, EAT 데이터 전처리 후 학습
   - 이는 `항공운항지연예측 Final-model.ipynb` 파일에 저장함
   - public: 0.6337836841점, private: 0.7080757848점
7. 대회 종료 후, [private점수 1위의 코드](https://dacon.io/competitions/official/236094/codeshare/8341)를 참고하여 수정한 모델
   - 이는 `항공운항지연예측 with 1st.ipynb` 파일에 저장함
   - public: 0.6234701617점, private: 0.7169214489점
