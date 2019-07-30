import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import pairwise_distances


def solve(X,Y,lm,n_iter=10000):
	'''Primal binary SVM solution'''
	C=len(Y)
	W=np.array([0 for i in range(len(X[0]))])

	for it in range(n_iter):
		#print(it)
		eta=1.0/(lm*(it+1))
		choice=random.randint(0,C-1)
		x,y=X[choice],Y[choice]
		out=np.dot(W.T,x)
		if y*out >= 1:
			W = (1-eta*lm)*W
		else:
			W = (1-eta*lm)*W + (eta*y)*x



	return W

def kernel_solve(X,Y,lm,n_iter=20000):
	'''Kernel Pegasos SVM'''
	C=len(X) 
	alpha=[0.0 for i in range(len(X))]
	K=pairwise_kernels(X,metric='rbf')
	for it in range(n_iter):
		choice=random.randint(0,C-1)
		x,y=X[choice],Y[choice]
		cur_sum=0.0
		for i in range(len(X)):
			if i!=choice:
				
				cur_sum+=alpha[i]*Y[i]*K[choice][i]
		if (cur_sum*y)/(lm*(1))<1:
			alpha[choice]+=1.0

	for i in range(len(alpha)):
		alpha[i]=alpha[i]/(lm)
	return alpha


def classify_all(X,K,Y,alpha):
	
	out_labels=[]
	for i in range(len(X)):
		tot_sum=0.0
		for j in range(len(X)):
			tot_sum+=alpha[j]*Y[j]*K[i][j]
		if tot_sum>0:
			out_labels.append(1)
		else:
			out_labels.append(-1)
	return out_labels





X,Y=make_circles()
print(Y)
for i in range(len(Y)):
	if Y[i]<=0:
		Y[i]=-1
	else:
		Y[i]=1
print(X.shape,Y.shape)
x1,x2=X[:,0],X[:,1]
#plt.scatter(x1,x2,c=Y)
#plt.show()


W=solve(X,Y,0.01)

out_labels=[]
for i in X:
	if np.dot(W.T,i)>0:
		out_labels.append(1)
	else:
		out_labels.append(-1)

#K=pairwise_kernels(X,metric='rbf')
#print(K[0][1])
#alpha=kernel_solve(X,Y,0.01)

#out_labels=classify_all(X,K,Y,alpha)
print("accuracy",accuracy_score(Y,out_labels))
out_labels=np.array(out_labels)
print(out_labels)
plt.scatter(x1,x2,c=out_labels)
plt.show()
