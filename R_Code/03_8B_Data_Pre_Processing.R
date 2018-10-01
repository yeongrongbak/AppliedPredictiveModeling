################################################################################
### Section 3.1 Case Study: Cell Segmentation in High-Content Screening

library(AppliedPredictiveModeling)
data(segmentationOriginal)
head(segmentationOriginal,2)

## Retain the original training set
segData <- subset(segmentationOriginal, Case == "Train")

## Remove the first three columns (identifier columns)
cellID <- segData$Cell
class <- segData$Class
case <- segData$Case

segData <- segData[, -(1:3)]

(statusColNum <- grep("Status",names(segData)))
segData <- segData[,-statusColNum]

################################################################################
### Section 3.2 Data Transformations for Individual Predictors

library(e1071) # install.packages("e1071")
skewness(segData$AngleCh1)

skewValue <- apply(segData,2,skewness)
head(skewValue)

library(caret)
(Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1))

# Original data
head(segData$AreaCh1)
predict(Ch1AreaTrans, head(segData$AreaCh1))

# Apply eq. to the first datum
(819^(-0.9)-1)/(-0.9) # (x^Lamnda-1)/Lamnda

################################################################################
### Section 3.3 Data Transformations for Multiple Predictors

## R's prcomp is used to conduct PCA
pcaObject <- prcomp(segData,center =T, scale = TRUE)
str(pcaObject)

## Calculate accum. ratio of variance for each variable
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
percentVariance[1:3]

head(pcaObject$x[,1:5])
dim(pcaObject$x)

head(pcaObject$rotation[,1:3])

trans <- preProcess(segData, method = c("BoxCox","center","scale","pca"))
trans
str(trans,max.level = 1)

## Apply Transform
transformed <- predict(trans, segData)

head(transformed[,1:5])
head(segData[,1:5])

################################################################################
### Section 3.5 Removing Variables

nearZeroVar(segData)


## To filter on correlations, we first get the correlation matrix for the 
## predictor set

correlations <- cor(segData)
dim(correlations)

correlations[1:4,1:4]

library(corrplot) # install.packages("corrplot")
corrplot(correlations, order = "hclust")

## caret's findCorrelation function is used to identify columns to remove.
highCorr <- findCorrelation(correlations, .75)
length(highCorr)
head(highCorr)
filterdSegData <- segData[,-highCorr]

################################################################################
### Section 3.8 Computing (Creating Dummy Variables)

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

################################################################################
## Exercise

## 3.1
library(mlbench)
data(Glass)
str(Glass)

## 3.2
library(mlbench)
data(Soybean)
str(Soybean)

## 3.3
library(caret)
data(BloodBrain)
?BloodBrain
str(bbbDescr)

################################################################################
### Session Information

sessionInfo()

q("no")


