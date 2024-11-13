# Diabeties Predictive Model
Our goal is to make a predictive model to predict if an individual has diabeties or not based on 9 related factors. 
___________________
**Description**
___________________
Our project is creating multiple predictive models on R-studio through Random Forest, Logarithmic, and Support Vector Machine (SVM) Models. Our goal is to understand how each factor relates to diabeties as a whole, and figure out which model has the highest sensitivty and specificity to predict diabeties accurately. Our 8 factors we're focusing on are: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabeties Pedigree Function , Age, with the last factor being the outcome (1 = has diabeties, 2 = no diabeties). Currently, our data set is from a subset of individuals, specifically: Female, Indian Heritage, 21 or older. WE hope to use this model to expand our subset of the data set, in order to make predictions for greater groups of individuals. 
__________________
**Getting Started**
___________________
Dependencies: 
* You'll need to install R-Studio or have a platform that can run both R and Python on.

Installing: 
* Download the data set as a csv file
* Make sure the data set is set as the working directory (to do this, right click on the CSV file once on your platform, and set that file to current working directory)

Executing Program: 
* All these codes are in a mixture of R and Python

```R
# code to use:
# in order to use R studio but Python packages, you need to call the reticulate package: 
library("reticulate")
pd <- import("pandas")
np <- import("numpy")
plt <- import("matplotlib.pyplot")
sns <- import("seaborn")
setwd("~/Downloads") # change this to the current working directory you are at 
df <- pd$read_csv("diabetes.csv") # your csv file might be called differently 
variables <- c('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome')

# check for 0 in the variables, cannot have 0 in blood pressure or insulin (logically impossible) 
for (var in variables) { 
	unique_values <- unique(df[[var]])
	print(unique_values)
}

#once you find the 0 variables, change those to the mean
for (var in variables) {
	column_mean <- mean(df[[var]], na.rm = TRUE)
	df[[var]] <- replace(df[[var]], df[var] == 0, column_mean)
}
for (var in variables) {
	c <- 0 
	for (x in df[[var]]) {
		if (x == 0) {
			c <- c + 1
		}
	}
	cat(var,c)
}

# to print out important values from the code (mean, median, mode, first quadrant, third quadrant)
str(df)
summary(df)

# data analysis + representation part:
install.packages("ggplot2")
library(ggplot2)

#boxplot of all the variables: 
boxplot(df, main = "Boxplot Of All The Factors")

#making a heatmap:
install.packages("pheatmap")
library(pheatmap)
summary_df <- as.data.frame(sapply(df, summary))
pheatmap(as.matrix(summary_df), cluster_rows = FALSE, cluster_cols = FALSE, display_numbers = TRUE)

#histogram for all the factors : 
for (var in variables) {
	hist(df[[var]], 
		main = paste("Histogram of", var), 
		xlab = var, 
		col = "lightblue", 
		border = "black", 
		breaks = 20)
}

# boxplot for all the factors, comparing whether individual has diabeties or not
for (var in colnames(summary_df)) {
  p <- ggplot(summary_df, aes_string(x = var, fill = "factor(Outcome)")) +
  geom_boxplot(position = "dodge", color = "black") +  
  labs(
    title = paste("Boxplot of", var),
      x = var,  # Correct x-axis label
      y = "Outcome"  # You probably want "Outcome" as the y-axis label
      ) +
  scale_fill_manual(values = c("lightblue", "lightcoral")) +
  theme_minimal()
  print(p)  # This will print the plot for each variable
}
```

**Predictive Models**
______________________
Types of Predictive Models: 
* Random Forest
* Logarithmic Model
* Support Vector Machine
