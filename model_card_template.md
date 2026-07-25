# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest Classifier from scikit-learn to predict whether a person's annual income is greater than $50,000 beased on demographic and employment information. This project uses the U.S. Census Income dataset for the required information for analysis. This model was trained using Python and deployed using a FastAPI.

## Intended Use
This model is intended to be used for educational purposes to demonstrate the complete process of building a machine learning pipeline. This includes data preprocessing, model training, evaluation, testing, and API deployment.

## Training Data
The model was trained using the Census Income dataset that was provided for this project. The dataset includes information such as age, education, occupation, work class, marital status, hours worked per week, and other demographic features. Before training the model, the categorical variables were encoded so they could be used by the machine learning algorithm.

## Evaluation Data
The dataset was split into training and testing sets using train_test_split. The training data was used to build the model, while the testing data was kept separate and used to evaluate how well the model performed on data it had not seen before.

## Metrics
The metrics used to evaluate the model were Precision, Recall, and F1 Score.
_Please include the metrics used and your model's performance on those metrics._
 - Precision: 0.74
 - Recall: 0.64
 - F1 Score: 0.69
Ontop of these metrics, different categories were used to evaluate performance using the performance_on_categorical_slice function.

## Ethical Considerations
Because this dataset contains demographic information, it may also reflect biases that exist in the real world. As a result, the model could produce predictions that are unfair to certain groups if it were used in real decision-making. This project is meant to demonstrate machine learning concepts and should not be used to make important decisions like hiring, lending, or insurance without additional testing and bias analysis.

## Caveats and Recommendations
This model was created as part of a learning project, so there is still room for improvement. The model could be made stronger by tuning the hyperparameters, experimenting with different algorithms, adding more feature engineering, and testing for fairness across different groups. Before using a model like this in a real-world application, it should be validated further and monitored to make sure it continues to perform well.
