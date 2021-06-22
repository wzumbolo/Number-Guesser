import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import time
import random

list_x = []
list_y = []

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

count = 0

while True:
    guesses = 0
    correct = 0
    print(len(digits.data))
    n = random.randint(0, 100)
    x, y = digits.data[:-n], digits.target[:-n]
    clf.fit(x, y)

    tt = time.time()
    predict = ('prediction:', clf.predict([digits.data[-n-1]]))
    print('prediction:', clf.predict([digits.data[-n-1]]))
    t = time.time() - tt
    guesses = guesses + 1
    print('time elapsed: ' + str(t))
    plt.imshow(digits.images[-n-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    value = input('What is the number?: \n ')
    if value == predict:
        correct = correct + 1
    list_x.append(guesses)
    list_y.append(t)
    count += 1
    if count == 5:
        break

plt.plot(list_x, list_y)
plt.show()

