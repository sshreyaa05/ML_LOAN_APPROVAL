### Loan Approval Classification Using Machine Learning

### Project Overview:
The objective of this project was to build a machine learning classification model to predict loan approval using a loan approval dataset. Various classification algorithms were applied and evaluated to identify the best-performing model. After model selection and hyperparameter tuning, the final model was identified as the Gradient Boosting Classifier, achieving an accuracy of 0.982. Additionally, the ROC curve was plotted to assess the model's performance.

### Steps Performed:

#### Data Collection and Preprocessing:

1. Utilized a loan approval dataset containing features like applicant income, loan amount, credit history, and marital status.

2. Checked for missing values and handled them appropriately.

3. Scaled the features using StandardScaler to ensure uniformity across the models.

#### Exploratory Data Analysis (EDA) and Visualization:

1. Visualized data distributions using histograms and boxplots.

2. Analyzed correlations between features using a heatmap.

3. Identified patterns in loan approval decisions using bar charts and pair plots.

#### Model Building and Training:

Applied multiple classification algorithms for training:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

AdaBoost Classifier

Gradient Boosting Classifier

#### Model Evaluation and Cross-Validation:

1. Performed cross-validation to ensure reliable and unbiased evaluation.

2. Evaluated models using accuracy, precision, recall, and F1-score.

3. Based on performance metrics, Gradient Boosting Classifier was identified as the best-performing model.

#### Hyperparameter Tuning Using Random Search CV:

1. Applied Random Search Cross Validation to optimize the hyperparameters of the Gradient Boosting Classifier.

2. Selected the best combination of parameters for improved accuracy.

#### Model Fitting and Prediction:

1. Trained the optimized Gradient Boosting Classifier on the training data.

2. Predicted loan approval on the test data.

#### Evaluation:

1. Calculated the final model's accuracy as 0.982.

2. Plotted the ROC curve to visualize the true positive rate (TPR) vs. false positive rate (FPR).

3. The model demonstrated high sensitivity and specificity, making it highly reliable for loan approval prediction.

### Results:

1. The Gradient Boosting Classifier outperformed other models with an excellent accuracy score of 0.982.

2. The ROC curve indicated robust model performance, ensuring reliable loan approval predictions.

3. This model can be effectively used in financial institutions for making accurate and data-driven loan approval decisions.
