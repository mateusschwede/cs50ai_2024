import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    if len(sys.argv) != 2:
        sys.exit("Execute: python shopping.py data")

    evid, labels = load_data(sys.argv[1])
    xTrain, xTest, yTrain, yTest = train_test_split(evid, labels, testSize=TEST_SIZE)
    model = train_model(xTrain, yTrain)
    predicts = model.predict(xTest)
    sens, spec = evaluate(yTest, predicts)

    print(f"Correct: {(yTest == predicts).sum()}")
    print(f"Incorrect: {(yTest != predicts).sum()}")
    print(f"True Posit. Rate: {100 * sens:.2f}%")
    print(f"True Negat. Rate: {100 * spec:.2f}%")


def load_data(filename):
    evid = []
    labels = []
    month = {name: num - 1 for num, name in enumerate(calendar.month_abbr) if num}

    with open(filename) as data:
        read = csv.reader(data)
        next(read)
        for r in read:
            evid.append([
                int(r[0]),
                float(r[1]),
                int(r[2]),
                float(r[3]),
                int(r[4]),
                float(r[5]),
                float(r[6]),
                float(r[7]),
                float(r[8]),
                float(r[9]),
                month[r[10][:3]],
                int(r[11]),
                int(r[12]),
                int(r[13]),
                int(r[14]),
                1 if r[15] == 'Returning_Visitor' else 0,
                int((r[16]) == 'TRUE'),
            ])
            labels.append(
                int(r[17] == 'TRUE')
            )
    return evid, labels


def train_model(evidence, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    sens = 0
    spec = 0
    posit = 0
    negat = 0

    for l, p in zip(labels, predictions):
        if l == 1:
            posit = posit + 1
            if l == p:
                sens = sens + 1
        else:
            negat = negat + 1
            if l == p:
                spec = spec + 1

    sens = sens / posit
    spec = spec / negat
    return sens, spec


if __name__ == "__main__":
    main()