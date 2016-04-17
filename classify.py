from neuralnet import NeuralNet
import sys


def neuralnet(arglist):
    nn = NeuralNet(arglist[1])

    folds = arglist[2]
    learning_rate = arglist[3]
    epochs = arglist[4]

    nn.evaluate(folds, epochs, learning_rate)
    nn.print_results()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Not enough arguments')
    else:
        neuralnet(sys.argv)
