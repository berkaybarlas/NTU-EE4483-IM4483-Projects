import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

from keras.datasets import cifar100
from keras.utils import to_categorical
import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)): ssl._create_default_https_context = ssl._create_unverified_context

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# Data set Location /Users/username/.keras/datasets
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

clfier = RandomForestClassifier(n_estimators = 100, max_depth = 10, min_samples_leaf = 10)

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

trainY = to_categorical(y_train)
testY = to_categorical(y_test)

x_train, x_test = prep_pixels(x_train, x_test)
## Train the Random Forest using Train Set of CIFAR-10 Dataset
clfier.fit(x_train, x_test)
rf_Y_prediction = clfier.predict(x_test)
rf_acc = accuracy_score(y_test, rf_Y_prediction) * 100
print('Random forest accuracy: % 5.2f' %(rf_acc))
#
## Test with Random Forest for Test Set of CIFAR-10 Dataset
#labelVectorPredicted = clfier.predict(featureMatrixTest)
#
#correct = (labelVectorPredicted == labelVectorTest).sum()
#print('Accuracy of the network on the 10000 test images: %d %%' % (
#    100 * correct / labelVectorTest.shape[0]))
