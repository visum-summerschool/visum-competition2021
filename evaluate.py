import numpy as np
import pandas


def evaluate(predictions, solutions):
    assert len(predictions) == len(solutions)
    predictions = [int(x) for x in predictions]
    solutions = [int(x) for x in solutions]
    acc = np.mean([x == y for x, y in zip(predictions, solutions)])
    return acc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 evaluate.py <path to preds.csv> <path to solutions.csv>")
        sys.exit(-1)
    with open(sys.argv[1]) as file:
        preds = list(pandas.read_csv(file)["productid"])
    with open(sys.argv[2]) as file:
        solutions = list(pandas.read_csv(file)["productid"])

    print("Accuracy:", evaluate(preds, solutions))
