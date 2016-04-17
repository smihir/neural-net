from __future__  import division
import matplotlib.pyplot as plt
from neuralnet import NeuralNet
import numpy as np
import sys


def plot1(trainf):
    print("Running Test 1")
    nn = NeuralNet(trainf)

    #nn.evaluate(folds, epochs, learning_rate)
    nn.evaluate(10, 25, 0.1)
    acc1 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(10, 50, 0.1)
    acc2 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(10, 75, 0.1)
    acc3 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(10, 100, 0.1)
    acc4 = nn.evaluate_accuracy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy vs. Epochs for Neural Net')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    y = [acc1, acc2, acc3, acc4]
    x = [25, 50, 75, 100]
    ax.plot(x, y, c='b', marker='o')

def plot2(trainf):
    print("Running Test 2")
    nn = NeuralNet(trainf)

    #nn.evaluate(folds, epochs, learning_rate)
    nn.evaluate(5, 50, 0.1)
    acc1 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(10, 50, 0.1)
    acc2 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(15, 50, 0.1)
    acc3 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(20, 50, 0.1)
    acc4 = nn.evaluate_accuracy()

    nn.clean_training_data()
    nn.evaluate(25, 50, 0.1)
    acc5 = nn.evaluate_accuracy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('Accuracy vs. Folds for Neural Net')
    ax1.set_xlabel('Folds')
    ax1.set_ylabel('Accuracy')
    y = [acc1, acc2, acc3, acc4, acc5]
    x = [5, 10, 15, 20, 25]
    ax1.plot(x, y, c='b', marker='o')

def plot3(trainf):
    print("Running Test 3")
    nn = NeuralNet(trainf)

    #nn.evaluate(folds, epochs, learning_rate)
    nn.evaluate(10, 50, 0.1)
    x, y = nn.evaluate_roc()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('ROC for Neural Net')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.plot(x, y, c='b', marker='o')

def plot():
    plt.show()

if __name__ == '__main__':
    plot1('sonar.arff')
    plot2('sonar.arff')
    plot3('sonar.arff')
    print("Plotting...")
    plot()
