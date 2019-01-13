import matplotlib.pyplot as plt



def plotAccuracy(numberOfParameters, accuracies, architectureNames):

    fig = plt.figure()
    for i in range(len(accuracies)):
        x = numberOfParameters[i]
        y = accuracies[i]
        plt.plot(x, y, 'ro')
        plt.text(x + 0.02, y + 0.02, architectureNames[i], fontsize=9)

    plt.xlabel('Number of Parameters', fontsize=10)
    plt.ylabel('Top-1 Accuracies', fontsize=10)
    plt.show()
    fig.savefig('parameters_vs_accuracies.png')

numberOfParameters = [1, 2, 3, 4]
accuracies = [1, 4, 9, 6]
architectureNames = ["a", "b", "c", "d"]

plotAccuracy(numberOfParameters, accuracies, architectureNames)