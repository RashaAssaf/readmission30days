import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def load_data():
	input_filename = "../data/readmissions_icd+chartevents+gender_v2.csv"
	data = np.genfromtxt(input_filename, delimiter=',')
	
	features = data[:, :-1]
	labels = data[:, -1]
	features, labels = shuffle(features, labels, random_state=65273)

	return features, labels


def hpo(X, y):
	# Number of trees in random forest
	n_estimators = [10, 50, 100, 500]

	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']

	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)

	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]

	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]

	# Method of selecting samples for training each tree
	bootstrap = [True, False]

	# Create the random grid
	random_grid = {'n_estimators': n_estimators,
	               'max_features': max_features,
	               'max_depth': max_depth,
	               'min_samples_split': min_samples_split,
	               'min_samples_leaf': min_samples_leaf,
	               'bootstrap': bootstrap}
	pprint(random_grid)

	model = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
	model.fit(X, y)
	print(model.best_params_)

	return model

def main():
	tunning = False
	X, y = load_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	print('Found {} data samples'.format(len(X)))
	print('Train data size {}'.format(len(X_train)))
	print('Test data size {}'.format(len(X_test)))
	print('Number of features {}'.format(len(X[0])))

	if tunning:
		model = hpo(X_train, y_train)
		model = RandomForestClassifier(**model.best_params_)
		model.fit(X_train, y_train)
	else:
		best_params = {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 110, 'bootstrap': True}
		model = RandomForestClassifier(**best_params)
		model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	print(model.feature_importances_)
	
	accuracy = accuracy_score(y_test, y_pred)
	cm = confusion_matrix(y_test, y_pred)

	fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
	roc_auc = metrics.auc(fpr, tpr)

	print('Accuracy ', accuracy)
	print('AUC ', roc_auc)
	print(cm)

	# Plot ROC
	plt.title('Receiver Operating Characteristic (ROC)')
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