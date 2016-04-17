from __future__ import division
import copy
import sys
import operator
import random
import math

def sigmoid(x):
        return 1 / (1 + math.e**(-x))

class NeuralNet:
    def __init__(self, fname):
        self.arff = __import__("arff")
        self.data = None
        self.classifier_dictionary = dict()
        self.results = dict()
        self.weights = None
        with open(fname) as f:
            self.raw_data = self.arff.load(f)
        self.make_classifier_dictionary()
        self.generate_model()

    def generate_model(self):
        self.data = copy.deepcopy(self.process_raw_data(self.raw_data['data']))

    def clean_training_data(self):
        self.results = dict()
        self.weights = None

    def make_attribute_dictionary(self):
        for attr in self.raw_data['attributes']:
            self.attribute_dictionary[attr[0]] = attr[1]

    def make_classifier_dictionary(self):
        # works for binary classifiers as per the problem statement
        # the last element in the attributes' list is the classifier
        self.classifier_dictionary[self.raw_data['attributes'][-1][1][0]] = 0
        self.classifier_dictionary[self.raw_data['attributes'][-1][1][1]] = 1

    def process_raw_data(self, raw_data):
        res = list()
        for rd in raw_data:
            di = copy.deepcopy(rd)
            di[-1] = self.classifier_dictionary[di[-1]]
            res.append(di)
        return res

    def train(self, train_data_idx, epoch = 1, learning_rate = 0.1):
        # Initialize weight array
        # we are using single layer nn with no hidden layer
        # every attribute is a i/p node connected to the single o/p
        # and there is one bias unit
        self.weights = [0.1]*len(self.raw_data['attributes'])
        bias = 1
        
        classifier = self.raw_data['attributes'][-1][0]

        for e in range(epoch):
            for idx in train_data_idx:
                d = self.data[idx]
                train_vector = copy.deepcopy(d)
                train_vector[-1] = bias

                net = sum(tv * w for tv, w in zip(train_vector, self.weights))
                out = sigmoid(net)
                delta = out * (1 - out) * (d[-1] - out)

                i = 0
                for in_val in train_vector:
                    delta_weight = learning_rate * delta * in_val
                    self.weights[i] += delta_weight
                    i += 1

    def classify(self, fold, data_idx):
        bias = 1
        for idx in data_idx:
            d = self.data[idx]
            test_vector = copy.deepcopy(d)
            test_vector[-1] = bias
            net = sum(tv * w for tv, w in zip(test_vector, self.weights))
            out = sigmoid(net)
            if idx in self.results:
                print("key already exists " + str(idx) + " \n")
                raise
            self.results[idx] = [fold, out]

    def print_results(self):

        results = sorted(self.results.items(), key=operator.itemgetter(0))

        for r in results:
            prediction = 0 if r[1][1] <= 0.5 else 1
            pclass = self.raw_data['attributes'][-1][1][prediction]
            aclass = self.raw_data['data'][r[0]][-1]
            confidence = r[1][1]
            print(str(r[1][0] + 1) + ' ' + pclass + ' ' + aclass + str(" %.12f" %confidence))

    def evaluate_accuracy(self):

        # no need to sort, but makes it easy to debug if there is a problem
        # and I do not have any timing constraints
        results = sorted(self.results.items(), key=operator.itemgetter(0))

        pos = 0
        neg = 0
        for r in results:
            prediction = 0 if r[1][1] <= 0.5 else 1
            pclass = self.raw_data['attributes'][-1][1][prediction]
            aclass = self.raw_data['data'][r[0]][-1]

            if pclass == aclass:
                pos += 1
            else:
                neg += 1
        return pos / (pos + neg)

    def evaluate_roc(self):

        # since this is binary, I will assume if sigmoid approaches 1 then the
        # confidenceof positive outcome is more
        results = sorted(self.results.items(), key=lambda item: item[1][1], reverse=True)

        ap = 0
        an = 0
        for r in results:
            aclass = self.raw_data['data'][r[0]][-1]
            actual = self.classifier_dictionary[aclass]
            if actual == 1:
                ap += 1
            else:
                an += 1

        tp = 0
        fp = 0
        prev_aclass = None
        roc_x = list()
        roc_y = list()
        for r in results:
            aclass = self.raw_data['data'][r[0]][-1]
            actual = self.classifier_dictionary[aclass]
            if prev_aclass and prev_aclass != aclass:
                roc_x.append(fp / an)
                roc_y.append(tp / ap)
            prev_aclass = aclass
            if actual == 1:
                tp += 1
            else:
                fp += 1

        roc_x.append(fp / an)
        roc_y.append(tp / ap)

        return roc_x, roc_y


    def evaluate(self, folds = 10, epoch = 1, learning_rate = 0.1):
        folds = int(folds)
        epoch = int(epoch)
        learning_rate = float(learning_rate)

        class0_ids = list()
        class1_ids = list()
        i = 0
        for d in self.data:
            if d[-1] == 0:
                class0_ids.append(i)
            else:
                class1_ids.append(i)
            i += 1

        random.seed(0)
        random.shuffle(class0_ids)
        random.shuffle(class1_ids)

        buckets = [list() for i in range(folds)]

        i = 0
        for d in class0_ids:
            buckets[i].append(d)
            i += 1
            if i == folds:
                i = 0

        i = 0
        for d in class1_ids:
            buckets[i].append(d)
            i += 1
            if i == folds:
                i = 0

        for foldid in range(folds):
            test_ids = buckets[foldid]
            train_ids = list()
            for i in range(folds):
                if i == foldid:
                    continue
                train_ids.extend(buckets[i])

            self.train(train_ids, epoch, learning_rate)
            self.classify(foldid, test_ids)


if __name__ == "__main__":
    nn = NeuralNet(sys.argv[1])
    nn.evaluate(10, 5, 0.1)
    nn.evaluate_roc()
