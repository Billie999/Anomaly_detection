# Supervised and Unsupervised Anomaly Detection
#******************************************
# Biljana Simonovikj
# Due sate: 06/05/2020

# Load Packages ***************************

  library("devtools")
  library("factoextra")
  library("plot3D")
  library("RColorBrewer")
  library("ggplot2")
  library("gridExtra")
  library("dbscan")
  library("dplyr")
  library("class")
  library("reshape2")
  library("ROCR")
  library("pander")
  library("plot3D")
  library("class")




# Activity 1: Principal Component Analysis
#******************************************
# Read the data from csv.file into R environment:
#=====================================================================
Stamps <- read.table("/Users/Biljana/Data Mining_1/Ass 3/Stamps_withoutdupl_09.csv",
                     header = FALSE, sep = ",", dec = ".") # read the csv file into a dataframe
colnames(Stamps) # display column names
head(Stamps) # show first six cases
summary(Stamps) # 9 Predictors (V1 to V9) and class labels (V10)
PB_Predictors <- Stamps[,1:9] # 9 Predictors (V1 to V9)
PB_class <- Stamps[,10] # Class labels (V10)
PB_class <- ifelse(PB_class == 'no',0,1) # Inliers (class "no") = 0, Outliers (class "yes") = 1
PB_Class <- as.numeric(PB_class) # assign atomic vector as numeric
PB_Predictors_Data <- as.matrix(PB_Predictors) # conversion to conventional matrix
table(PB_Class) # how many observations are genuine or forged

#In this first activity, you are asked to:
# 1. Perform Principal Component Analysis (PCA) on the Stamps data in the 9-dimensional space of the
#numerical predictors ( PB_Predictors ), and show the Proportion of Variance Explained (PVE) for each
#of the nine resulting principal components. Plot the accumulated sum of PVE for the first m
#components, as a function of m, and discuss the result: (a) How many components do we need to
#explain 90% or more of the total variance? (b) How much of the total variance is explained by the first
#three components?

# Applying PCA to 9-dimensional space of predictor variables:
#===========================================================
PCA_Stamps <- prcomp(PB_Predictors_Data, scale = TRUE, center = TRUE) # apply prcomp() from stats package

# Inspect PCA results with get_eigenvalue() from factoextra package:
get_eigenvalue(PCA_Stamps)
eig <- get_eigenvalue(PCA_Stamps) # assign PCA results into a variable

# Create a data frame from PCA results in order to present them with ggplot2:
df_eig <- data.frame(eigenvalue = eig[ ,1], variance = eig[ ,2], cumvariance = eig[ ,3]) # create a data frame
df_eig <- cbind(df_eig, row_id = as.factor(1:nrow(eig)), row_count = 1:nrow(eig)) # add row_id and row_count variables

# Assign and rename variables:
Row_Id <- df_eig$row_id
Row_Count <- df_eig$row_count
CumPVE <- (round(df_eig$cumvariance, digits = 2)) # round the values to 2 digits
Variance <- (round(df_eig$variance, digits = 2)) # round the values to 2 digits

# Display of Proportion of Variance Explained (PVE) for each of the nine PCA components with geom_bar() from ggplot2:
#====================================================================================================================
ggplot(df_eig,aes(x = Row_Id,y = Variance)) +
  geom_bar(stat = "identity",aes(fill = Row_Id),color = "black") +
  geom_path(aes(x = Row_Count),size = 1,color = "Gray50") +
  geom_point(color = "Gray50", size = 1) +
  xlab("Principal Component") +
  ylab("Proportion of Variance Explained, (PVE)") +
  scale_fill_brewer(breaks = c(1:9),
                    palette = "YlGnBu",
                    direction = -1) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

# # Display of Accumulated Sum of PVE for each of the nine PCA components with geom_bar() from ggplot2 up to cut off at 90th%:
#==============================================================================================================================
ggplot(df_eig,aes(x = Row_Id,y = CumPVE, group = 1)) +
  geom_bar(stat = "identity",aes(fill = Row_Id),color = "black") +
  geom_path(aes(x = Row_Count),size = 1,color = "Gray50") +
  geom_hline(yintercept = 90, linetype = "dashed", color = "Gray50", size = 1, show.legend = T) +
  geom_text(aes(label = paste(CumPVE, "%", sep = "")), vjust = 3, color = "Gray50") +
  geom_point(color = "Gray50", size = 1) +
  xlab("Principal Component") +
  ylab("Accumulated Percentage of PVE") +
  scale_fill_brewer(breaks = c(1:9),
                    palette = "YlGnBu",
                    direction = -1) +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")

# Create a data frame of 3 PCA components:
PCA_Components_Stamps = data.frame(PCA_Stamps$x[, 1:3])
PC1 <- PCA_Components_Stamps$PC1
PC2 <- PCA_Components_Stamps$PC2
PC3 <- PCA_Components_Stamps$PC3

# Scatter-plot observations by PCA components 1 and 2:
plot1 = ggplot(PCA_Components_Stamps, aes(PC1, PC2)) +
  geom_point(aes(colour = factor(PB_Class)))

# Scatter-plot observations by PCA components 1 and 3:
plot2 = ggplot(PCA_Components_Stamps, aes(PC1, PC3)) +
  geom_point(aes(colour = factor(PB_Class)))

# Scatter-plot observations by PCA components 2 and 3:
plot3 = ggplot(PCA_Components_Stamps, aes(PC2, PC3)) +
  geom_point(aes(colour = factor(PB_Class)))

grid.arrange(plot1, plot2, plot3, ncol = 1) # set up the grid

# 2. Do some research by yourself on how to render 3D plots in R, and then plot a 3D scatter-plot of the
# Stamps data as represented by the first three principal components computed in the previous item
# ( x = PC1 , y = PC2 , and z = PC3 ). You can use, for example, the function scatter3D() from the
# package plot3D. Use the class labels (PB_class) to plot inliers and outliers in different colours (for
# example, inliers in black and outliers in red). Make sure you produce multiple plots from different angles
# (at least three). Recalling that the class labels would not be available in a practical application of
# unsupervised outlier detection, do the outliers (forged stamps) look easy to detect in an unsupervised
# way, assuming that the 3D visualisation of the data via PCA is a reasonable representation of the data in
# full space? How about in a supervised way? Why? Justify your answers.
#**********************************************************************************************************

# 3D-Scatter Plots at different angles with PB_Class and without PB_Class:
#========================================================================
x <- PCA_Components_Stamps[,1] # define X_axis
y <- PCA_Components_Stamps[,2] # define y axis
z <- PCA_Components_Stamps[,3] # define z axis
# full panels of box are drawn (bty = "f")

# 3D-Scatter Plot without PB_Class:
scatter3D(x, y, z, pch = 20, col = "black", phi = 180,
          theta = 60, byt = "f",
          main = "3D Scatter Plot without Class Labels",
          xlab = "PC1",
          ylab = "PC2",
          zlab = "PC3")

# First view - 3D-Scatter Plot with PB_Class:
scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = PB_Class, phi = 180,
          theta = 60, bty = "f",
          main = "3D Scatter Plot with Class Labels",
          xlab = "PC1",
          ylab = "PC2",
          zlab = "PC3")

# Second view - 3D-Scatter Plot with PB_Class:
scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = PB_Class, phi = 60,
          theta = 180, bty = "f",
          main = "3D Scatter Plot with Class Labels",
          xlab = "PC1",
          ylab = "PC2",
          zlab = "PC3")

# Third view - 3D-Scatter Plot with PB_Class:
scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = PB_Class, phi = 120,
          theta = 360, byt = "f",
          main = "3D Scatter Plot with Class Labels",
          xlab = "PC1",
          ylab = "PC2",
          zlab = "PC3")


#************************************************
# Activity 2: Unsupervised outlier detection
#************************************************
# In this second activity, you are asked to perform unsupervised outlier detection on the Stamps data in the 9-
# dimensional space of the numerical predictors (PB_Predictors), using KNN Outlier with different values of the
# parameter (at least the following three: k = 5, 25, 100). For each k, produce the same 3D PCA
# visualisation of the data as in Activity 1 (PCA), but rather than using the class labels to colour the points, use
# instead the resulting KNN Outlier Scores as a continuous, diverging colour scale. Then, for each k, produce a
# second plot where the top-31 outliers according to the KNN Outlier Scores are shown in red, while the other
# points are shown in black. Do these plots give you any insights on the values of that look more or less
# appropriate from an unsupervised perspective (ignoring the class labels)? Justify your answer

# Unsupervised Outlier Detection with different values of parameter k:
#*********************************************************************
k1 = 5 # KNN parameter
KNN_5_Outlier <- kNNdist(x = PB_Predictors, k = k1, all = TRUE)[,k1] # KNN distance (outlier score) computation
k2 = 25 # KNN parameter
KNN_25_Outlier <- kNNdist(x = PB_Predictors, k = k2, all = TRUE)[,k2] # KNN distance (outlier score) computation
k3 = 100 # KNN parameter
KNN_100_Outlier <- kNNdist(x = PB_Predictors, k = k3, all = TRUE)[,k3] # KNN distance (outlier score) computation

#### First View - KNN Outliers and Top 31 KNN Outliers:
# Display KNN outliers through function:
KNN_outliers <- function(PB_Predictors, PCA_Components_Stamps, k) {
  x = PCA_Components_Stamps[,1]
  y = PCA_Components_Stamps[,2]
  z = PCA_Components_Stamps[,3]
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  cols <- brewer.pal(11, "BrBG") # diverging colour palette "BrBG" with 11 colours
  pal <- colorRampPalette(cols) # pass the pallette to colorRampPalette()
  scatter3D(x, y, z, bty = "f", pch = 20,
            col = pal(11), colvar = KNN_Outlier, phi = 180, theta = 60,
            main = paste("KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# Display of Top 31 KNN outliers through function:
Top_31_KNN_Outliers <- function(PB_Predictors, PCA_Components_Stamps, k){
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  PCA_Outliers_Df <- as.data.frame(cbind(PCA_Components_Stamps, KNN_Outlier)) # combine PCA with KNN outlier scores
  PCA_Outliers_Rank <- PCA_Outliers_Df[order(PCA_Outliers_Df$KNN_Outlier, decreasing = T),] # reorder data frame
  KNN_Result <- cbind(PCA_Outliers_Rank, ID = seq(1:nrow(PCA_Outliers_Rank))) # add in ID column
  Top_Outliers <- ifelse(KNN_Result$ID <= 31,"1","0") # define the factor levels for top 31 outliers
  # Assign x, y and z coordinates according to the new reordered dataframe
  x = KNN_Result[,1]
  y = KNN_Result[,2]
  z = KNN_Result[,3]
  scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = as.numeric(Top_Outliers), phi = 180,
            theta = 60, bty = "f",
            main = paste("Top 31 KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# 3D plot with KNN outliers if k = 5:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 5)
# 3D plot with Top 31 KNN outliers if k = 5:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 5)

# 3D plot with KNN outliers if k = 25:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 25)
# 3D plot with Top 31 KNN outliers if k = 25:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 25)

# 3D plot with KNN outliers if k = 100:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 100)
# 3D plot with Top 31 KNN outliers if k = 100:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 100)

#### Second View - KNN Outliers and Top 31 KNN Outliers:
# Display KNN outliers through function:
KNN_outliers <- function(PB_Predictors, PCA_Components_Stamps, k) {
  x = PCA_Components_Stamps[,1]
  y = PCA_Components_Stamps[,2]
  z = PCA_Components_Stamps[,3]
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  cols <- brewer.pal(11, "BrBG") # diverging colour palette "BrBG" with 11 colours
  pal <- colorRampPalette(cols) # create interpolation function
  scatter3D(x, y, z, bty = "f", pch = 20,
            col = pal(11), colvar = KNN_Outlier, phi = 60, theta = 180,
            main = paste("KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# Display of Top 31 KNN outliers through function:
Top_31_KNN_Outliers <- function(PB_Predictors, PCA_Components_Stamps, k){
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  PCA_Outliers_Df <- as.data.frame(cbind(PCA_Components_Stamps, KNN_Outlier)) # combine PCA with KNN outlier scores
  PCA_Outliers_Rank <- PCA_Outliers_Df[order(PCA_Outliers_Df$KNN_Outlier, decreasing = T),] # reorder data frame
  KNN_Result <- cbind(PCA_Outliers_Rank, ID = seq(1:nrow(PCA_Outliers_Rank))) # add in ID column
  Top_Outliers <- ifelse(KNN_Result$ID <= 31,"1","0") # define the factor levels for top 31 outliers
  # Assign x, y and z coordinates according to the new reordered dataframe
  x = KNN_Result[,1]
  y = KNN_Result[,2]
  z = KNN_Result[,3]
  scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = as.numeric(Top_Outliers), phi = 60,
            theta = 180, bty = "f",
            main = paste("Top 31 KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# 3D plot with KNN outliers if  k = 5:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 5)
# 3D plot with Top 31 KNN outliers if k = 5:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 5)

# 3D plot with KNN outliers if  k = 25:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 25)
# 3D plot with Top 31 KNN outliers if k = 25:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 25)

# 3D plot with KNN outliers if  k = 100:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 100)
# 3D plot with Top 31 KNN outliers if k = 100:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 100)

#### Third View - KNN Outliers and Top 31 KNN Outliers:
# Display KNN outliers through function:
KNN_outliers <- function(PB_Predictors, PCA_Components_Stamps, k) {
  x = PCA_Components_Stamps[,1]
  y = PCA_Components_Stamps[,2]
  z = PCA_Components_Stamps[,3]
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  cols <- brewer.pal(11, "BrBG") # diverging colour palette
  pal <- colorRampPalette(cols) # pass the pallete to colorRampPalette()
  scatter3D(x, y, z, bty = "f", pch = 20,
            col = pal(11), colvar = KNN_Outlier, phi = 120, theta = 360,
            main = paste("KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# Display of Top 31 KNN outliers through function:
Top_31_KNN_Outliers <- function(PB_Predictors, PCA_Components_Stamps, k){
  x = PCA_Components_Stamps[,1]
  y = PCA_Components_Stamps[,2]
  z = PCA_Components_Stamps[,3]
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  PCA_Outliers_Df <- as.data.frame(cbind(PCA_Components_Stamps, KNN_Outlier)) # combine PCA with KNN outlier scores
  PCA_Outliers_Rank <- PCA_Outliers_Df[order(PCA_Outliers_Df$KNN_Outlier, decreasing = T),] # reorder data frame using KNN values
  KNN_Result <- cbind(PCA_Outliers_Rank, ID = seq(1:nrow(PCA_Outliers_Rank))) # add in ID column
  Top_Outliers <- ifelse(KNN_Result$ID <= 31,"1","0") # define the factor levels for top 31 outliers
  # Assign x, y and z coordinates according to the new reordered dataframe
  x = KNN_Result[,1]
  y = KNN_Result[,2]
  z = KNN_Result[,3]
  scatter3D(x, y, z, pch = 20, col = c("black", "red"), colvar = as.numeric(Top_Outliers), phi = 120,
            theta = 360, bty = "f",
            main = paste("Top 31 KNN Outliers if k = ", k, sep = ""),
            xlab = "PC1",
            ylab = "PC2",
            zlab = "PC3")
}

# 3D plot with KNN outliers if k = 5:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 5)
# 3D plot with Top 31 KNN outliers if k = 5:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 5)

# 3D plot with KNN outliers if k = 25:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 25)
# 3D plot with Top 31 KNN outliers if k = 25:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 25)

# 3D plot with KNN outliers if k = 100:
KNN_outliers(PB_Predictors, PCA_Components_Stamps, k = 100)
# 3D plot with Top 31 KNN outliers if k = 100:
Top_31_KNN_Outliers(PB_Predictors, PCA_Components_Stamps, k = 100)

#******************************************
# Activity 3: Supervised anomaly detection
#******************************************

# In this third activity you are asked to:
# 1. Perform supervised classification of the Stamps data, using a KNN classifier with the same values of k
# as used in Activity 2 (unsupervised outlier detection). For each classifier (that is, each value of k),
# compute the Area Under the Curve ROC (AUC-ROC) in a Leave-One-Out Cross-Validation (LOOCV) scheme.

# KNN Supervised Classification
#===============================

# Put class variable(PB_Class) back on and rename:
Stamps_Df <- cbind(PB_Predictors, PB_Class)
names(Stamps_Df)[10] <- "Class"

# Check data:
colnames(Stamps_Df)
head(Stamps_Df)
dim(Stamps_Df)

# Split data into training set (80%) and testing set (20%):
set.seed(0) # random seed
No_Obs <- dim(Stamps_Df)[1] # No. of observations (340)
Test_Index <- sample(No_Obs, size = as.integer(No_Obs*0.2), replace = FALSE) # 20% data records for test
Test_Predictors <- Stamps_Df[Test_Index, c(1:9)] # testing dataset
Test_Class <- as.numeric(Stamps_Df[Test_Index, "Class"]) # testing class label
Training_Index <- -Test_Index # 80% data records for training
Training_Predictors <- Stamps_Df[Training_Index, c(1:9)] # training ataset
Training_Class <- as.numeric(Stamps_Df[Training_Index, "Class"]) # training class label

# BUILD AND TEST CLASSIFIER and compare predicted outcome to observed outcome:
#============================================================================

# Build knn classifier if k = 5
Dfn_5_Pred <- knn(train = Training_Predictors, test = Test_Predictors, cl = Training_Class, k = 5, prob = T)
(table(Dfn_5_Pred, Test_Class))

# Build knn classifier if k = 25
Dfn_25_Pred <- knn(train = Training_Predictors, test = Test_Predictors, cl = Training_Class, k = 25, prob = T)
(table(Dfn_25_Pred, Test_Class))

#  Build knn classifier if k = 100
Dfn_100_Pred <- knn(train = Training_Predictors, test = Test_Predictors, cl = Training_Class, k = 100, prob = T)
(table(Dfn_100_Pred, Test_Class))

# Calculate AUC-ROC values for each supervised KNN Classifiers in a Leave-One-Out Cross-Validation (LOOCV) scheme:
#=================================================================================================================
# Creata rocplot() function to calculate AUC_ROC:
rocplot <- function(predicted, observed){
  Pred_Obj <- prediction(predicted, observed)
  ROC <- performance(Pred_Obj, "tpr", "fpr")
  # Plot the ROC Curve
  plot(ROC, colorize = T, lwd = 3, main = "ROC Curve")
  auc <- performance(Pred_Obj, measure = "auc")
  auc <- auc@y.values[[1]]
  # Return the Area Under the Curve ROC
  return(auc)
}

# Plot ROC Curves for KNN classifiers with LOOCV cross-validation and with k parameter values (5, 25 and 100):
#=============================================================================================================
# Create a function to assess a KNN classifier using LOOCV and to plot ROC curve:
ROC_AUC_knn.cv <- function(Training_Predictors, Training_Class, k){
  Pred_Class <- knn.cv(train = Training_Predictors, cl = Training_Class, k = k, prob = TRUE)
  Pred_Prob <- attr(Pred_Class, "prob")
  Pred_Prob <- ifelse(Pred_Class == 1, Pred_Prob, 1 - Pred_Prob)
  AUC <- rocplot(predicted = Pred_Prob, observed = Training_Class) # call rocplot()
  cat("K-value:", k, ", AUC:", AUC, fill = TRUE)
}

# k = 5
ROC_AUC_knn.cv(Training_Predictors, Training_Class, 5)

# k = 25
ROC_AUC_knn.cv(Training_Predictors, Training_Class, 25)

## k = 100
ROC_AUC_knn.cv(Training_Predictors, Training_Class, 100)

# Plot AUC results for supervised KNN classifiers with LOOCV type of cross-validation and k = 1:100:
#==================================================================================================
# Create Auc_Calc() function to calculate AUC results:
AUC_Calc <- function(predicted, observed){
  Pred_Obj <- prediction(predicted, observed)
  auc <- performance(Pred_Obj, measure = "auc")
  auc <- auc@y.values[[1]]
  # Return the Area Under the Curve ROC
  return(auc)
}

# Calculate AUC results for supervised KNN classifiers with LOOCV schema and k = 1:100:
#======================================================================================
set.seed(0) # random seed
AUC_Super <- rep(0, 100) # create an empty vector
for (k in 1:100) {
  Pred_Class <- knn.cv(train = Training_Predictors, cl = Training_Class, k = k, prob = T) # LOOCV applied with knn.cv()
  Pred_Prob <- attr(Pred_Class, "prob")
  Pred_Prob <- ifelse(Pred_Class == 1, Pred_Prob, 1 - Pred_Prob)
  AUC_Super[k] <- AUC_Calc(predicted = Pred_Prob, observed = Training_Class)
}

# Plot the AUC values with ggplot2:
#=================================
Auc_Df <- as.data.frame(AUC_Super) # create data frame
Auc_Df <- cbind(Auc_Df, k = 1:nrow(Auc_Df)) # add in k column as index
k <- Auc_Df$k # assign variable as a vector
AUC_Super <- Auc_Df$AUC_Super # assign variable as avector
ggplot(data = Auc_Df, aes(x = k, y = AUC_Super, group = 1)) +
      geom_point(alpha = 20/40, color = "blue") + # showing values as scatter plots
      geom_line(color = "magenta3") + # layer a pink line
      geom_hline(yintercept = 0.967, linetype = "dashed", color = "Grey50") + # higlight with dashed line
          xlab(label = "k value") +
          ylab(label = "AUC") +
          ggtitle("AUC Results for Supervised KNN Classifers with k = 1:100 on Training Dataset \n Best AUC = 0,9678 for k = 8")

# Maximum AUC result calculation:
#===============================
Auc_Df %>% filter(AUC_Super == max(AUC_Super))

# 2. Compare the resulting (supervised) KNN classification performance for each value of, against the
# classification performance obtained in an unsupervised way by the KNN Outlier method with the same
# value of k. Notice that, if we rescale the KNN Outlier Scores (obtained in Activity 2 (unsupervised outlier
# detection)) into the interval, these scores can be interpreted as outlier probabilities, which can
# then be compared with the class labels (ground truth) in PB_class to compute an AUC-ROC value. This
# way, for each value of K, the AUC-ROC of the supervised KNN classifier can be compared with the
# AUC-ROC of KNN Outlier as an unsupervised classifier. Compare the performances of the supervised
# versus unsupervised classifiers and discuss the results. For example, recalling that the supervised
# method makes use of the class labels, whereas the unsupervised method doesn’t, what can you
# conclude considering there are applications where class labels are not available?

# First, we calculate AUC-ROC values on non-normalized dataset of unsupervised KNN Classifier with varying k values
# in order to compare them same values obtained from supervised classification with KNN classifiers under the same
# conditions: non-normalized dataset and k values of 5, 25 and 100.

# Calculate AUC-ROC on non-normalized dataset of unsupervised KKN Classifiers:
#=============================================================================
# Create a function to calculate AUC-ROC of unsupervised KKN Classifiers:
KNN_outliers <- function(PB_Predictors, PB_Class, k) {
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k]
  AUC <- rocplot(predicted = KNN_Outlier, observed = PB_Class)
  cat("K-value:", k, ", AUC:", AUC, fill = TRUE)
}

# k = 5
KNN_outliers(PB_Predictors, PB_Class, 5)

# k = 25
KNN_outliers(PB_Predictors, PB_Class, 25)

## k = 100
KNN_outliers(PB_Predictors, PB_Class, 100)

# Plot the AUC values for supervised versus unsupervised KNN Classifiers:
#=========================================================================
# Create a for loop to generate AUC values for 100 k's:
set.seed(0) # random seed
AUC_Unsuper <- rep(0, 100) # create an empty vector
for (k in 1:100) {
  KNN_Outlier <- kNNdist(x = PB_Predictors, k = k, all = TRUE)[,k] # assess the unsupervised KNN
  AUC_Unsuper[k] <- AUC_Calc(predicted = KNN_Outlier, observed = PB_Class) # call AUC_Calc
}

# Create a dataframe containing AUC values of supervised and unsupervised classifications in order
# to plot them together with ggplot2:
#=================================================================================================
Auc_Unsuper_Df <- as.data.frame(AUC_Unsuper) # create dataframe with AUC_unsuper first
Auc_Unsuper_Df <- cbind(Auc_Unsuper_Df, k = 1:nrow(Auc_Unsuper_Df)) # add in k column as index
AUC_Values = merge(Auc_Unsuper_Df, Auc_Df, by = "k") # merge both dataframes by k
AUC_Values_Melted <- reshape2::melt(AUC_Values, id.var = 'k') # melt both dataframes
Value <- AUC_Values_Melted$value # assign variable to a vector
Variable <- AUC_Values_Melted$variable # assign variable to a vector
K <- AUC_Values_Melted$k # assign variable to a vector

# Generate the plot:
#===================
ggplot(AUC_Values_Melted, aes(x = K, y = Value, col = Variable)) +
  geom_point(alpha = 20/40, color = "blue") + # showing values as scatter plots
  geom_line() + # layer a pink line
    xlab(label = "k value") +
    ylab(label = "AUC") +
    ggtitle("Comparison of AUC Results of Supervised Versus Nonsupervised KNN Classifiers on Non-Normalized Dataset")


# Comparison of AUC values on non-normalized data with a table:
#=============================================================
Auc_Results <- AUC_Values[c(5,25,100), 1:3] # select 5th, 25th and 100th row
row.names(Auc_Results) <- NULL # remove default row names
pander(Auc_Results, style = 'rmarkdown', caption = "Summary of AUC results for 5, 25 and 100 KNN Classifiers
       (supervised and unsupervised) on non-normalized dataset") # make table

# Maximum AUC results calculation with the corresponding k parameter for k = 1:100:
#================================================================================
Auc_Unsuper_Df %>% filter(AUC_Unsuper == max(AUC_Unsuper))

# Data Normalization
#=========================
# Create function for min-max normalization in a range (0, 1) of the dataset:
#===========================================================================
normalise <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Create new data frame of all predictor variables with normalized data:
#=======================================================================
PB_Predictors_Scaled <- as.data.frame(lapply(PB_Predictors, normalise))

# Calculate AUC-ROC unsupervised KKN Classifiers on normalized dataset:
#=====================================================================
# Call the KNN_outliers() function:
#=================================
KNN_outliers <- function(PB_Predictors_Scaled, PB_Class, k) {
  KNN_Outlier <- kNNdist(x = PB_Predictors_Scaled, k = k, all = TRUE)[,k]
  AUC <- rocplot(predicted = KNN_Outlier, observed = PB_Class)
  cat("K-value:", k, ", AUC:", AUC, fill = TRUE)
}

# k = 5
KNN_outliers(PB_Predictors_Scaled, PB_Class, 5)

# k = 25
KNN_outliers(PB_Predictors_Scaled, PB_Class, 25)

## k = 100
KNN_outliers(PB_Predictors_Scaled, PB_Class, 100)


# Calculate AUC-ROC oof supervised KKN Classifiers in normalized dataset:
#=========================================================================================================================
# Call the function ROC_AUC_knn.cv():
#====================================
ROC_AUC_knn.cv <- function(PB_Predictors_Scaled, PB_Class, k){
  Pred_Class <- knn.cv(train = PB_Predictors_Scaled, cl = PB_Class, k = k, prob = TRUE)
  Pred_Prob <- attr(Pred_Class, "prob")
  Pred_Prob <- ifelse(Pred_Class == 1, Pred_Prob, 1 - Pred_Prob)
  AUC <- rocplot(predicted = Pred_Prob, observed = PB_Class) # call rocplot()
  cat("K-value:", k, ", AUC:", AUC, fill = TRUE)
}

# k = 5
ROC_AUC_knn.cv(PB_Predictors_Scaled, PB_Class, 5)

# k = 25
ROC_AUC_knn.cv(PB_Predictors_Scaled, PB_Class, 25)

## k = 100
ROC_AUC_knn.cv(PB_Predictors_Scaled, PB_Class, 100)


# Plot the AUC results for supervised versus unsupervised KNN Classifiers:
#=========================================================================
# Create a for loop to generate AUC values for 100 k's for supervised Classifiers:
#===================================================================================
set.seed(0) # random seed
AUC_Super_Norm <- rep(0, 100) # create an empty vector
for (k in 1:100) {
  Pred_Class <- knn.cv(train = PB_Predictors_Scaled, cl = PB_Class, k = k, prob = T)
  Pred_Prob <- attr(Pred_Class, "prob")
  Pred_Prob <- ifelse(Pred_Class == 1, Pred_Prob, 1 - Pred_Prob)
  AUC_Super_Norm[k] <- AUC_Calc(predicted = Pred_Prob, observed = PB_Class)
}
# Create a for loop to generate AUC values for 100 k's for unsupervised Classifiers:
#===================================================================================
set.seed(0) # random seed
AUC_Unsuper_Norm <- rep(0, 100) # create an empty vector
for (k in 1:100) {
KNN_Outlier <- kNNdist(x = PB_Predictors_Scaled, k = k, all = TRUE)[,k]
AUC_Unsuper_Norm[k] <- AUC_Calc(predicted = KNN_Outlier, observed = PB_Class)
}
## Create a dataframe containing AUC values of supervised and unsupervised classifications in order
# to plot them together with ggplot2:
#=================================================================================================
Auc_Super_Df_Norm <- as.data.frame(AUC_Super_Norm) # create dataframe with supervised AUC values first
Auc_Super_Df_Norm <- cbind(Auc_Super_Df_Norm, k = 1:nrow(Auc_Super_Df_Norm)) # add in k column as index

Auc_Unsuper_Df_Norm <- as.data.frame(AUC_Unsuper_Norm) # create dataframe with AUC unsupervised values
Auc_Unsuper_Df_Norm <- cbind(Auc_Unsuper_Df_Norm, k = 1:nrow(Auc_Unsuper_Df_Norm)) # add in k column as index

AUC_Values_Norm = merge(Auc_Unsuper_Df_Norm, Auc_Super_Df_Norm, by = "k") # merge both dataframes by k
AUC_Values_Melted_Norm <- reshape2::melt(AUC_Values_Norm, id.var = 'k') # melt both dataframes
Value <- AUC_Values_Melted_Norm$value # assign variable to a vector
Variable <- AUC_Values_Melted_Norm$variable # assign variable to a vector
K <- AUC_Values_Melted_Norm$k # assign variable to a vector

# Generate the plot:
#===================
ggplot(AUC_Values_Melted_Norm, aes(x = K, y = Value, col = Variable)) +
  geom_point(alpha = 20/40, color = "blue") + # showing values as scatter plots
  geom_line() + # layer a pink line
    xlab(label = "k value") +
    ylab(label = "AUC") +
    ggtitle("Comparison of AUC Result for Supervised versus Unsupervised KNN Classifers on Normalized Dataset")

# Maximum AUC result calculation for supervised KNN classifiers:
#===============================================================
Auc_Super_Df_Norm %>% filter(AUC_Super_Norm == max(AUC_Super_Norm))

# Maximum AUC result calculation for supervised KNN classifiers:
#===============================================================
Auc_Unsuper_Df_Norm %>% filter(AUC_Unsuper_Norm == max(AUC_Unsuper_Norm))

# Comparison of AUC results on normalized data:
#==============================================
Auc_Results_Norm <- AUC_Values_Norm[c(5,25,100), 1:3] # select 5th, 25th and 100th row
row.names(Auc_Results_Norm) <- NULL # remove default row names
pander(Auc_Results_Norm, style = 'rmarkdown', caption = "AUC results for KNN Classifiers on normalized data") # make table

# Summary of AUC results:
Overall_Results <- merge(Auc_Results, Auc_Results_Norm, by = "k") # merge both dataframe results by k
pander(Overall_Results, style = 'rmarkdown', caption = "Summary of AUC results for KNN Classifiers on non-normalized and normalized dataset") # make table


