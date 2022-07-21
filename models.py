@@ -1,15 +1,32 @@
 # Model training and hyperparameter optimization
 # This script will train the follow type of models
 # using Python sklearn package
 #	1. Random Forest
 #	2. Multi-Layer Perceptron
 #	3. Logistic Regression
 #	4. Support Vector Machine
 # Best model is evaluated on the Test dataset

 import numpy as np
 from pprint import pprint
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
 from sklearn.metrics import accuracy_score, confusion_matrix
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.svm import SVC
 from sklearn.neural_network import MLPClassifier
 from sklearn.model_selection import RandomizedSearchCV
 from sklearn.utils import shuffle
 import sklearn.metrics as metrics
 import matplotlib.pyplot as plt
 import argparse


 def load_data():
 	"""
 	Laod dataset from CSV file
 	:return: features and labels
 	"""
 	input_filename = "../data/readmissions_icd+chartevents+gender_v2.csv"
 	data = np.genfromtxt(input_filename, delimiter=',')

 @@ -20,8 +37,96 @@ def load_data():
 	return features, labels


 def hpo(X, y):
 	# Number of trees in random forest
 def hpo_mlp(X, y):
 	"""
 	Perform Hyperparameter optimization for MLP
 	:param X: numpy matrix - Train dataset n x d (number of samples x number of dimensions)
 	:param y: numpy vector - binary labels
 	:return: MLPClassifier model
 	"""
 	hidden_layer_sizes = [[8, 8],
 						  [16, 16],
 						  [16, 32],
 						  [32, 64],
 						  [64, 128],
 						  [64, 64],
 						  [128, 128],
 						  [8, 8, 8],
 						  [16, 32, 16],
 						  [32, 64, 32],
 						  [64, 128, 64],
 						  [64, 64, 64],
 						  [128, 128, 128]]
 	activation = ['relu', 'tanh']
 	solver = ['adam', 'sgd']
 	learning_rate = ['constant', 'adaptive']

 	# Create the random grid
 	random_grid = {
 		'hidden_layer_sizes': hidden_layer_sizes,
 		'activation': activation,
 		'solver': solver,
 		'learning_rate': learning_rate
 	}

 	pprint(random_grid)

 	model = RandomizedSearchCV(estimator=MLPClassifier(), param_distributions = random_grid, n_iter = 500, cv = 5, verbose=2, random_state=42, n_jobs = -1)
 	model.fit(X, y)
 	print(model.best_params_)

 	return model


 def hpo_svm(X, y):
 	"""
 	Perform Hyperparameter optimization for SVM
 	:param X: numpy matrix - Train dataset n x d (number of samples x number of dimensions)
 	:param y: numpy vector - binary labels
 	:return: SVC model
 	"""
 	kernel = ['rbf', 'sigmoid', 'linear', 'poly']

 	# Create the random grid
 	random_grid = {
 		'kernel': kernel
 	}
 	pprint(random_grid)

 	model = RandomizedSearchCV(estimator=SVC(), param_distributions = random_grid, n_iter = 500, cv = 5, verbose=2, random_state=42, n_jobs = -1)
 	model.fit(X, y)
 	print(model.best_params_)

 	return model


 def hpo_lr(X, y):
 	"""
 	Perform Hyperparameter optimization for LogisticRegression
 	:param X: numpy matrix - Train dataset n x d (number of samples x number of dimensions)
 	:param y: numpy vector - binary labels
 	:return: LogisticRegression model
 	"""
 	penalty = ['l1', 'l2']

 	# Create the random grid
 	random_grid = {'penalty': penalty}
 	pprint(random_grid)

 	model = RandomizedSearchCV(estimator = LogisticRegression(), param_distributions = random_grid, n_iter = 1000, cv = 5, verbose=2, random_state=42, n_jobs = -1)
 	model.fit(X, y)
 	print(model.best_params_)

 	return model


 def hpo_rf(X, y):
 	"""
 	Perform Hyperparameter optimization for RandomForestClassifier
 	:param X: numpy matrix - Train dataset n x d (number of samples x number of dimensions)
 	:param y: numpy vector - binary labels
 	:return: RandomForestClassifier model
 	"""
 	n_estimators = [10, 50, 100, 500]

 	# Number of features to consider at every split
 @@ -55,8 +160,15 @@ def hpo(X, y):

 	return model


 def main():
 	tunning = False
 	# Define command line arguments
 	parser = argparse.ArgumentParser()
 	parser.add_argument('--classifier', type=str, choices=['MLP', 'SVM', 'LR', 'RF'], help='')
 	parser.add_argument('--tunning', action='store_true', help='')
 	config = parser.parse_args()

 	# Load and split data into training and test datasets
 	X, y = load_data()
 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

 @@ -65,18 +177,40 @@ def main():
 	print('Test data size {}'.format(len(X_test)))
 	print('Number of features {}'.format(len(X[0])))

 	if tunning:
 		model = hpo(X_train, y_train)
 		model = RandomForestClassifier(**model.best_params_)
 	# Perform hyperparameter optimization
 	if config.tunning:
 		if config.classifier == 'RF':
 			model = hpo_rf(X_train, y_train)
 			model = RandomForestClassifier(**model.best_params_)
 		if config.classifier == 'LR':
 			model = hpo_lr(X_train, y_train)
 			model = LogisticRegression(**model.best_params_)
 		if config.classifier == 'SVM':
 			model = hpo_svm(X_train, y_train)
 			model = SVC(**model.best_params_)
 		if config.classifier == 'MLP':
 			model = hpo_mlp(X_train, y_train)
 			model = MLPClassifier(**model.best_params_)

 		model.fit(X_train, y_train)
 	else:
 		best_params = {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True}
 		model = RandomForestClassifier(**best_params)
 		if config.classifier == 'RF':
 			best_params = {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True}
 			model = RandomForestClassifier(**best_params)
 		if config.classifier == 'LR':
 			best_params = {'penalty': 'l2'}
 			model = LogisticRegression(**best_params)
 		if config.classifier == 'SVM':
 			best_params = {'kernel': 'linear'}
 			model = SVC(**best_params)
 		if config.classifier == 'MLP':
 			best_params = {}
 			model = MLPClassifier(**best_params)

 		model.fit(X_train, y_train)

 	# Evaluate model on the test dataset and compute metrics
 	y_pred = model.predict(X_test)
 	print(model.feature_importances_)

 	accuracy = accuracy_score(y_test, y_pred)
 	cm = confusion_matrix(y_test, y_pred)

 @@ -88,33 +222,16 @@ def main():
 	print(cm)

 	# Plot ROC
 	plt.title('Receiver Operating Characteristic (ROC)')
 	plt.title('{} Receiver Operating Characteristic (ROC)'.format(config.classifier))
 	plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
 	plt.legend(loc='lower right')
 	plt.plot([0, 1], [0, 1], 'r--')
 	plt.xlim([0, 1])
 	plt.ylim([0, 1])
  
 	plt.ylabel('True Positive Rate')
 	plt.xlabel('False Positive Rate')
 	plt.savefig('../data/roc.png')

 	std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
 	indices = np.argsort(model.feature_importances_)[::-1]

 	# Print the feature ranking
 	print("Feature ranking:")

 	for f in range(X.shape[1]):
 		print("%d. feature %d (%f)" % (f + 1, indices[f], model.feature_importances_[indices[f]]))
 	plt.savefig('../data/roc_{}.png'.format(config.classifier))

 	# Plot the impurity-based feature importances of the forest
 	plt.figure()
 	plt.title("Feature importances")
 	plt.bar(range(X.shape[1]), model.feature_importances_[indices],
 			color="r", yerr=std[indices], align="center")
 	plt.xticks(range(X.shape[1]), indices)
 	plt.xlim([-1, X.shape[1]])
 	plt.savefig('../data/feature_importance.png')

 if __name__ == '__main__':
 	main() 
