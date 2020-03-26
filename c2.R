#install packages once
install.packages("ggplot2")
#you can save your workspace in R, however, if you don't, you
#will have to run all of the library commands again before you
#start on the rest of the R script
library(ggplot2)

#read in the bank training data using read.csv
#alternatively, you can use setwd to set the directory path
#rather than use the full file path
#setwd("C:/Users/samea/Documents/R/chapter02")
#bank_train <- read.csv("bank_marketing_training")
bank_train <- read.csv("C:/Users/samea/Documents/R/chapter02/bank_marketing_training")
t1<-table(bank_train$response,bank_train$previous_outcome)
#row 1 of bank data
bank_train[1,]
#rows 1,3,4 of bank data
bank_train[c(1,3,4),]
#columns 1,3 of bank data
bank_train[,c(1,3)]
#column age of bank data
bank_train$age

#read in the Census training data using read.table
CensusData <- read.table(file = "C:/Users/samea/Documents/R/chapter02/acs_or.csv", header = TRUE, sep = ",")
