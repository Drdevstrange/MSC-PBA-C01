############################################################
# PBA Group Coursework – Member 1 Section

############################################################


##############################
# 1.1 Loading the dataset
##############################

# starting with a clean workspace so nothing old hangs around
rm(list = ls())

# The dataset should be in the same folder as this script.
# If the name is different, update it here.
data_file <- "kidney_disease.csv"

# Small sanity check 
# without having the file in the right place.
if (!file.exists(data_file)) {
  stop(paste(
    "I can't find the file:", data_file,
    "\nDouble-check the file name and that it's in this folder."
  ))
}

# Reading the CSV. The dataset uses things like "?" for missing values,
# so I tell R to treat those as NA straight away.
ckd <- read.csv(
  file = data_file,
  header = TRUE,
  stringsAsFactors = FALSE,
  na.strings = c("?", "", "NA")
)


###############################################
# 1.2 How many rows and columns do we have?
###############################################

cat("===== 1.2 DATA SIZE =====\n")
cat("Rows:", nrow(ckd), "\n")
cat("Columns:", ncol(ckd), "\n\n")
# Good to know the scale before doing anything deeper.


###############################################
# 1.3 Quick view at the first few rows
###############################################

cat("===== 1.3 FIRST 10 ROWS =====\n")
print(head(ckd, 10))
cat("\n")
# This helps us see if the data loaded properly and get a feel for it.


###############################################
# 1.4 Look at the structure (types + preview)
###############################################

cat("===== 1.4 STRUCTURE CHECK =====\n")
str(ckd)
cat("\n")
# str() is useful because it tells which columns are numbers
# and which came in as text. Some will need to be converted later.


###############################################
# 1.5 Basic summary stats
###############################################

cat("===== 1.5 SUMMARY STATS =====\n")
summary(ckd)
cat("\n")
# This gives a rough idea of ranges and whether some columns
# look weird or inconsistent already.


###############################################
# 1.6 A clearer look at each variable's type
###############################################

cat("===== 1.6 VARIABLE TYPES =====\n")
var_types <- sapply(ckd, class)
print(var_types)
cat("\n")
# Seeing the types like this makes it easier to plan conversions later.


###############################################
# 1.7 How many missing values per column?
###############################################

cat("===== 1.7 MISSING VALUES =====\n")
na_counts <- colSums(is.na(ckd))
print(na_counts)
cat("\n")
# Highlighting this early helps Member 3 when they handle imputation.


################################################
# 1.8 Count unique values in each column
################################################

cat("===== 1.8 UNIQUE VALUE COUNTS =====\n")
unique_counts <- sapply(ckd, function(x) length(unique(x)))
print(unique_counts)
cat("\n")
# Columns with only a few unique values are usually categorical.


################################################
# Extra: Show actual example categories (text cols)
################################################

cat("===== EXTRA: SAMPLE VALUES IN TEXT COLUMNS =====\n")
char_cols <- names(ckd)[sapply(ckd, is.character)]

if (length(char_cols) > 0) {
  for (col in char_cols) {
    cat("\nColumn:", col, "\n")
    print(head(unique(ckd[[col]]), 10))   # just a small preview
  }
} else {
  cat("No character columns detected.\n")
}

############################################################
# END OF MEMBER 1
#
# Notes for my report:
# - Mention how many rows/columns the dataset has.
# - Comment on which variables are text vs numeric.
# - Point out which columns have lots of missing values.
# - Mention any variables that seem categorical (based on unique values).
############################################################

##############################
# 2. CLEANING & PRE-PROCESSING
##############################
# 2.1 Rename columns
colnames(ckd) <- c("id", "age","bp","sg","al","su","rbc","pc","pcc","ba",
                   "bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc",
                   "htn","dm","cad","appet","pe","ane","classification")

cat("===== 2.1 NEW COLUMN NAMES =====\n")
print(colnames(ckd))


# 2.2 Clean Whitespace
ckd[] <- lapply(ckd, function(x) if(is.character(x)) trimws(x) else x)

# 2.3 Define numeric and factor lists
numeric_cols <- c("age","bp","sg","al","su",
                  "bgr","bu","sc","sod","pot",
                  "hemo","pcv","wbcc","rbcc")

factor_cols  <- c("rbc","pc","pcc","ba",
                  "htn","dm","cad","appet","pe","ane",
                  "classification")


#2.4 Convert to Numeric
ckd[numeric_cols] <- lapply(ckd[numeric_cols], function(x) as.numeric(as.character(x)))

# 2.5 Convert to Factor
ckd[factor_cols] <- lapply(ckd[factor_cols], factor)


cat("\n===== STRUCTURE AFTER CLEANING =====\n")
str(ckd)
cat("\nUnique Classifications (Should only be ckd and notckd):\n")
print(unique(ckd$classification))
if("id" %in% colnames(ckd)) {
  ckd$id <- NULL
  cat("Removed 'id' column.\n")
}

cat("===== FINAL SUMMARY (READY FOR EXPORT) =====\n")
summary(ckd)
write.csv(ckd, "kidney_clean.csv", row.names = FALSE)

cat("\nSuccess! 'kidney_clean.csv' has been saved to your folder.\n")

# END OF MEMBER 2

############################################################
# MEMBER 3 – MISSING VALUE TREATMENT (IMPUTATION)
############################################################

# The previous output dataset from step 2 should be in the same folder as this script.
# If the name is different, update it here.
data_file <- "kidney_clean.csv"

# Small sanity check
if (!file.exists(data_file)) {
    stop(paste(
        "I can't find the file:", data_file,
        "\nDouble-check the file name and that it's in this folder."
    ))
} else {
    cat("File found:", data_file, "\n")
}

# 3.1 Define helper functions

# Impute numeric with median (robust to outliers)
impute_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
}

# Impute factor with mode (most frequent category)
impute_mode <- function(x) {
  tab <- table(x)
  mode_val <- names(tab)[which.max(tab)]
  x[is.na(x)] <- mode_val
  return(x)
}

# 3.2 Record NA counts before imputation
cat("===== NA COUNTS BEFORE IMPUTATION =====\n")
na_before <- colSums(is.na(ckd))
print(na_before)

# 3.3 Apply imputation
ckd[numeric_cols] <- lapply(ckd[numeric_cols], impute_median)
ckd[factor_cols]  <- lapply(ckd[factor_cols], impute_mode)

# 3.4 Record NA counts after imputation
cat("\n===== NA COUNTS AFTER IMPUTATION =====\n")
na_after <- colSums(is.na(ckd))
print(na_after)

# 3.5 Combine into a summary table
na_summary <- data.frame(
  Column = names(na_before),
  Before = as.integer(na_before),
  After  = as.integer(na_after)
)

cat("\n===== BEFORE vs AFTER NA SUMMARY =====\n")
print(na_summary)

# 3.6 Save imputed dataset
write.csv(ckd, "kidney_clean_step3.csv", row.names = FALSE)
cat("\nSuccess! 'kidney_clean_step3.csv' has been saved to your folder.\n")

############################################################
# Notes for report (Member 3):
# - Median chosen for numeric variables (robust to outliers).
# - Mode chosen for categorical variables (preserves most common class).
# - After imputation, dataset contains no missing values.
# - Summary table shows effectiveness of imputation clearly.
# - Final imputed dataset exported as 'kidney_clean_step3.csv'.
############################################################

############################################################
# MEMBER 4 – EXPLORATORY DATA ANALYSIS (EDA)
############################################################

# Load required libraries (if not already loaded at top of script)
# install.packages("tidyverse")
# install.packages("corrplot")
# install.packages("scales")

library(tidyverse)
library(corrplot)
library(scales)

# 4.1 Load the imputed dataset (output from Member 3)
data_file <- "kidney_clean_step3.csv"

if (!file.exists(data_file)) {
  stop(paste(
    "I can't find the file:", data_file,
    "\nMake sure 'kidney_clean_step3.csv' is in this folder and Member 3 has run their step."
  ))
} else {
  cat("File found for EDA:", data_file, "\n")
}

ckd <- read.csv(data_file, stringsAsFactors = FALSE)

# Recreate numeric and factor column lists
numeric_cols <- c("age","bp","sg","al","su",
                  "bgr","bu","sc","sod","pot",
                  "hemo","pcv","wbcc","rbcc")

factor_cols  <- c("rbc","pc","pcc","ba",
                  "htn","dm","cad","appet","pe","ane",
                  "classification")

# Ensure correct types
ckd[numeric_cols] <- lapply(ckd[numeric_cols], as.numeric)
ckd[factor_cols]  <- lapply(ckd[factor_cols], factor)

cat("\n===== 4.0 STRUCTURE BEFORE EDA =====\n")
str(ckd)


##############################
# 4.2 Class balance
##############################

cat("\n===== 4.2 CLASS DISTRIBUTION =====\n")
class_counts <- table(ckd$classification)
print(class_counts)

cat("\nProportions:\n")
print(prop.table(class_counts))

# Bar plot of class distribution
ggplot(ckd, aes(x = classification, fill = classification)) +
  geom_bar() +
  labs(title = "CKD vs Non-CKD Cases",
       x = "Classification",
       y = "Count") +
  theme_minimal()


##############################
# 4.3 Summary of numeric variables
##############################

cat("\n===== 4.3 SUMMARY OF NUMERIC VARIABLES =====\n")
print(summary(ckd[numeric_cols]))

# Summary by class: mean & sd
cat("\n===== 4.3a SUMMARY OF NUMERIC VARIABLES BY CLASS (MEAN & SD) =====\n")
numeric_by_class <- ckd |>
  group_by(classification) |>
  summarise(
    across(
      all_of(numeric_cols),
      list(mean = ~mean(. , na.rm = TRUE),
           sd   = ~sd(. , na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    )
  )
print(numeric_by_class)


##############################
# 4.4 Key distributions – histograms
##############################

# Age distribution by CKD status
ggplot(ckd, aes(x = age, fill = classification)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  labs(title = "Age Distribution by CKD Status",
       x = "Age (years)",
       y = "Count") +
  theme_minimal()

# Haemoglobin distribution by CKD status
ggplot(ckd, aes(x = hemo, fill = classification)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  labs(title = "Haemoglobin Distribution by CKD Status",
       x = "Haemoglobin",
       y = "Count") +
  theme_minimal()

# Serum creatinine distribution by CKD status
ggplot(ckd, aes(x = sc, fill = classification)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 20) +
  labs(title = "Serum Creatinine Distribution by CKD Status",
       x = "Serum Creatinine (sc)",
       y = "Count") +
  theme_minimal()


##############################
# 4.5 Boxplots for important lab variables
##############################

# Serum creatinine by CKD status
ggplot(ckd, aes(x = classification, y = sc, fill = classification)) +
  geom_boxplot() +
  labs(title = "Serum Creatinine by CKD Status",
       x = "Classification",
       y = "Serum Creatinine (sc)") +
  theme_minimal()

# Blood urea by CKD status
ggplot(ckd, aes(x = classification, y = bu, fill = classification)) +
  geom_boxplot() +
  labs(title = "Blood Urea by CKD Status",
       x = "Classification",
       y = "Blood Urea (bu)") +
  theme_minimal()

# Haemoglobin by CKD status
ggplot(ckd, aes(x = classification, y = hemo, fill = classification)) +
  geom_boxplot() +
  labs(title = "Haemoglobin by CKD Status",
       x = "Classification",
       y = "Haemoglobin (hemo)") +
  theme_minimal()


##############################
# 4.6 Categorical variables vs CKD
##############################

# Helper function for bar plots of categorical vs class
plot_cat_by_class <- function(data, var_name) {
  ggplot(data, aes_string(x = var_name, fill = "classification")) +
    geom_bar(position = "fill") +
    labs(
      title = paste("Proportion of CKD by", var_name),
      x = var_name,
      y = "Proportion"
    ) +
    scale_y_continuous(labels = percent_format()) +
    theme_minimal()
}

# Examples: hypertension, diabetes, anaemia, appetite, edema
plot_cat_by_class(ckd, "htn")
plot_cat_by_class(ckd, "dm")
plot_cat_by_class(ckd, "ane")
plot_cat_by_class(ckd, "appet")
plot_cat_by_class(ckd, "pe")


##############################
# 4.7 Correlation analysis (numeric variables)
##############################

cat("\n===== 4.7 CORRELATION MATRIX (NUMERIC VARIABLES) =====\n")
cor_mat <- cor(ckd[numeric_cols], use = "complete.obs")
print(round(cor_mat, 2))

# Correlation heatmap
corrplot(
  cor_mat,
  method = "color",
  type = "upper",
  tl.col = "black",
  tl.cex = 0.7,
  addCoef.col = NA
)


##############################
# 4.8 Notes for report (Member 4)
##############################
# - Report class distribution and note any imbalance (e.g. more CKD than non-CKD).
# - Use numeric_by_class to describe differences in means (e.g. higher sc and bu,
#   lower hemo and pcv in CKD patients).
# - Comment on age patterns: whether CKD patients are older on average.
# - Explain categorical patterns: higher proportions of htn == 'yes', dm == 'yes',
#   ane == 'yes', and pe == 'yes' in CKD cases.
# - Use boxplots to mention presence of outliers (very high urea/creatinine).
# - Refer to the correlation heatmap to highlight strongly related variables
#   (e.g. bu–sc, hemo–pcv), and mention possible multicollinearity.
############################################################
# END OF MEMBER 4
############################################################

