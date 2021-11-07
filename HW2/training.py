import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from mlp import MLP


def main():
    # create the dataset
    inputs = np.array([(1, 1), (0, 0), (1, 0), (0, 1)]).reshape(4, 2)

    labels_AND = np.array([1, 0, 0, 0])
    labels_OR = np.array([1, 0, 1, 1])
    labels_not_AND = np.array([0, 1, 1, 1])
    labels_not_OR = np.array([0, 1, 0, 0])
    labels_XOR = np.array([0, 0, 1, 1])

    gate = input(
        "Please type one of the following gates: [AND, OR, not AND, not OR, XOR]: ")
    if gate.lower() == 'and':
        labels = labels_AND
    elif gate.lower() == 'or':
        labels = labels_OR
    elif gate.lower() == 'not and':
        labels = labels_not_AND
    elif gate.lower() == 'not or':
        labels = labels_not_OR
    elif gate.lower() == 'xor':
        labels = labels_XOR

    mlp = MLP()

    ratios = []
    avg_loss = []

    for epoch in range(1000):
        tmp_loss, accuracy = [], []
        for input_, label in zip(inputs, labels):
            mlp.forwardpropagate(input_)
            output = mlp.output
            loss = (label-output)**2
            tmp_loss.append(loss)
            mlp.backpropagate(label)

            # correct classification or not
            if (output < 0.5 and label == 0) or (output >= 0.5 and label == 1):
                accuracy.append(1)
            else:
                accuracy.append(0)

        avg = mean(tmp_loss)
        avg_loss.append(round(avg, 4))

        accuracy = np.asarray(accuracy)
        correct = np.where(accuracy == 1)[0]
        ratio = len(correct)/len(accuracy)
        ratios.append(ratio)
    visualize(avg_loss, ratios)


def visualize(loss, accuracy):
    x = [epoch for epoch in range(1000)]
    loss
    fig, ax = plt.subplots()
    ax.plot(x, loss, color="red", marker="o")
    ax.set_xlabel("epochs", fontsize=14)
    ax.set_ylabel("Loss", color="red", fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(x, accuracy, color="blue", marker="o")
    ax2.set_ylabel("Accurcay", color="blue", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
