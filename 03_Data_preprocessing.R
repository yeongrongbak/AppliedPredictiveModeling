
### Section 3.1 사례 연구 : 하이콘텐츠 스크리닝에서 세포 분할

library(AppliedPredictiveModeling) # install.packages("AppliedPredictiveModeling")
data(segmentationOriginal)
head(segmentationOriginal,2)

## 훈련 세트 할당

segData <- subset(segmentationOriginal, Case == "Train")

## 컬럼 제거

cellID <- segData$Cell
class <- segData$Class
case <- segData$Case

segData <- segData[, -(1:3)]

(statusColNum <- grep("Status",names(segData)))
segData <- segData[,-statusColNum]

### Section 3.2 개별 예측 변수에 대한 데이터 변환s

library(e1071) # install.packages("e1071")
skewness(segData$AngleCh1)

skewValue <- apply(segData,2,skewness)
head(skewValue)

library(caret) # install.packages("caret")
(Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)) # 적합한 변환법을 찾는다
# 원데이터
head(segData$AreaCh1)

# 찾은 변환법을 적용해 새 데이터를 만든다

head(Ch1AreaTrans)

predict(Ch1AreaTrans, head(segData$AreaCh1))

predict(Ch1AreaTrans, segData$AngleCh1)

# 변환 식 적용

(819^(-0.9)-1)/(-0.9) # (x^Lamnda-1)/Lamnda

### Section 3.3 여러 예측 변수 변형

## PCA 실행

pcaObject <- prcomp(segData,center =T, scale = TRUE)
str(pcaObject)

## 각 요소별 분산의 누적 비율 계산

percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
percentVariance[1:3]

head(pcaObject$x[,1:5])
dim(pcaObject$x)

head(pcaObject$rotation[,1:3])

## 데이터에 박스-콕스 변환, 중심화, 척도화를 적용 한 뒤 PCA 적용

trans <- preProcess(segData, method = c("BoxCox","center","scale","pca"))
trans
str(trans,max.level = 1)

## 변환 적용

transformed <- predict(trans, segData)

head(transformed[,1:5])
head(segData[,1:5])

### Section 3.5 예측 변수 제거

nearZeroVar(segData) # 분산이 0에 가까운 예측 변수를 걸러내고자 할 때 사용
# 제거해야될 예측 변수가 있다면 열 번호를 결과값으로 반환한다.

## 상관관계를 기준으로 변수를 걸러내고 싶다면 cor함수를 사용해 상관계수 확인

correlations <- cor(segData)
dim(correlations)

correlations[1:4,1:4]

## 데이터 상관 구조에 대한 시각화

library(corrplot) # install.packages("corrplot")
corrplot(correlations, order = "hclust")

## 상관계수를 토대로 변수를 걸러낼 때 findCorrelation 함수를 사용하면 3.5장에 나온 알고리즘을 적용할 수 있다.

highCorr <- findCorrelation(correlations, .75)
length(highCorr)
head(highCorr) # 삭제 대상 예측 변수 후보를 고른 후 해당 열의 번호를 반환
filterdSegData <- segData[,-highCorr]

### Section 3.6 예측 변수 추가 (가변수 생성)

data(cars)
type <- c("convertible", "coupe", "hatchback", "sedan", "wagon")

head(cars)
names(cars[14:18])

cars$Type <- factor(apply(cars[, 14:18], 1, function(x) type[which(x == 1)]))

carSubset <- cars[sample(1:nrow(cars), 20), c(1, 2, 19)]

head(carSubset)
levels(carSubset$Type)
simpleMod <- dummyVars(~Mileage + Type,
                       data = carSubset,
                       ## Remove the variable name from the
                       ## column name
                       levelsOnly = TRUE)
simpleMod
str(simpleMod)

predict(simpleMod, head(carSubset))

withInteraction <- dummyVars(~Mileage + Type + Mileage:Type,
                             data = carSubset,
                             levelsOnly = TRUE)
withInteraction
str(withInteraction)
predict(withInteraction, head(carSubset))

## Exercise

## 3.1
## 유리 분류 데이터 세트
## 7개의 유형으로 분류된 214개의 유리 샘플
## 굴절률과 여덟 가지 원소 비율에 대한 9개의 예측 변수

library(mlbench) # install.packages("mlbench")
data(Glass)
str(Glass)

# (a) 예측 변수 간의 상관관계와 분포를 시각화를 이용해 탐색
# (b) 데이터에 이상 값이 있는가? 한쪽으로 치우친 예측 변수가 있는가?
# (c) 분류 모델 성능을 향상시키기 위해 예측 변수를 변환해야 할까?

correlation <- cor(Glass[,-10])
corrplot(correlation, order = "hclust")

plot(Glass)

boxplot(Glass$RI)
hist(Glass$RI) # 오른쪽으로 치우침

boxplot(Glass$Na)
hist(Glass$Na)

boxplot(Glass$Mg)
hist(Glass$Mg) # 양 끝 값이 많음

boxplot(Glass$Al)
hist(Glass$Al)

boxplot(Glass$Si)
hist(Glass$Si)

boxplot(Glass$K)
hist(Glass$K) # 오른쪽으로 치우침

boxplot(Glass$Ca)
hist(Glass$Ca) # 오른쪽으로 치우침

hist(Glass$Ba) # 오른쪽으로 치우침

library(GGally)

GGally::ggpairs(Glass[,-10])+
  ggplot2::theme(axis.text=ggplot2::element_text(size=2))



# 로그 변환

hist(log(Glass$RI))
hist(log(Glass$Mg))

## 3.2
## 콩 데이터
## 683개의 콩에 대한 질병 예측
## 35개의 예측변수(환경적 요인, 식물 상태 정보), 결과는 19가지로 구분 

library(mlbench)
data(Soybean)
str(Soybean)
?Soybean

ggpairs(Soybean[,-1])

# (a) 범주형 예측 변수의 빈도 분포를 확인하자. 앞에서 논의한 퇴화 형태를 보이는 분포가 있는가?

table(Soybean$Class)

barplot(table(Soybean$Class))



# (b) 약 18% 데이터 값이 누락돼 있다. 특정 예측 변수가 주로 누락됐는가? 범주별로 결측값의 패턴이 있을까?
# (c) 예측 변수 제거나 결측값 대치 중 어떤 방식으로 결측값을 처리할지 계획을 세우자.

## 3.3
library(caret)
data(BloodBrain)
?BloodBrain
str(bbbDescr)

