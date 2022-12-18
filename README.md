# Decision-Tree-ML
- Machine Learning in Decision Tree Classifier.
- In this lab exercise, you will learn a popular machine learning algorithm, Decision Tree. You will use this classification algorithm to build a model from historical data of patients, and their response to different medications. Then you use the trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.

## Import the Following Libraries:
- numpy (as np)
- pandas
- DecisionTreeClassifier from sklearn.tree

## About Dataset
- In the Dataset we have seen that thier is different type of animals in the our real world and we categories the different species on the bases of our nakes eyes but,
  Machine Learning use the data of animal species on the basis of Body Temperature, Skin Cover, Give Birth, Aquatic Creature, Aerial Creature, Has Legs, Hibernates on the basis of 
  this data we predict the class of our animal that which Species is belong on the basis of this Parameters.
- So, We use Decision Tree Classifier MOdel for the prediction of the Species.

## Practice for the Prediction 
- 1) Downloading Data and read the data with pandas librarie.
- 2) Pre-processing > Using my_data as the Drug.csv data read by pandas, declare the following variables:
-                     X as the Independent Variable.
-                     y as the Dependent Variable.

## Setting up the Decision Tree 
- We will be using train/test split on our decision tree. Let's from sklearn.model_selection import train_test_split.

## Modeling
- We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.

## Prediction
- We use the  y_predict = model.predict(x variable) for prediction variables.

## Evaluation
- Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
- In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels,
  then the subset accuracy is 1.0; otherwise it is 0.0.
  
## Visualization
- We use Visualization for the plotting the Decision Tree to predict the class of the Mammal.
