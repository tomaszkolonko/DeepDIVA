import matplotlib.pyplot as plt

def plotAccuracy(numberOfParameters, accuracies, accuracies_pretrained, colors, colors_pretrained, architectureNames, figureName):

    fig = plt.figure()
    for i in range(len(accuracies)):
        x = numberOfParameters[i]
        y1 = accuracies[i]
        y2 = accuracies_pretrained[i]
        #Plot accuracy of not-pretrained architecture
        plt.plot(x, y1, 'o', color=colors[i])
        plt.text(x + 0.02, y1 + 0.02, architectureNames[i], fontsize=9)
        #Plot accuracy of pretrained architecture
        plt.plot(x, y2, 'o', color=colors_pretrained[i])
        plt.text(x + 0.02, y2 + 0.02, architectureNames[i]+' (p)', fontsize=9)

    plt.xlabel('Number of Parameters', fontsize=10)
    plt.ylabel('Top-1 Accuracies', fontsize=10)
    plt.show()
    fig.savefig(figureName)


numberOfParameters = [58406080, 140105664, 24703968, 7085056]
numberOfParameters_segmentation = [34531592, 1376216]
accuracies_md = [12.71, 19.56, 17.51, 24.51]
accuracies_md_pretrained = [33.02, 32.62, 24.96, 33.92]
accuracies_sc = [36.97, 34.78, 42.72, 42.17]
accuracies_sc_pretrained = [47.27, 44.42, 48.82, 45.92]
accuracies_similarity = [2.645, 17.49, 6.569, 13.97]
accuracies_similarity_pretrained = [7.648, -1, -1, 19.21]
accuracies_segmentation = [58.56, -1]
accuracies_segmentation_pretrained = [-1, -1]
architectureNames = ["Resnet152", "Vgg19_bn", "Inception_v3", "Densenet121"]
colors = ['#ff8080', '#4dff88', '#809fff', '#ff80df']
colors_pretrained = ['#ff0000', '#009933', '#1a53ff', '#cc0099']
architectureNames_segmentation = ["Unet", "FC-Densenet57"]
figureNames = ['parameters_vs_accuracies_md.png', 'parameters_vs_accuracies_sc.png', 'similarity.png', 'segmentation.png']

plotAccuracy(numberOfParameters, accuracies_md, accuracies_md_pretrained, colors, colors_pretrained, architectureNames, figureNames[0]) #Manuscript Dating
plotAccuracy(numberOfParameters, accuracies_sc, accuracies_sc_pretrained, colors, colors_pretrained, architectureNames, figureNames[1]) #Style Classification
plotAccuracy(numberOfParameters, accuracies_similarity, accuracies_similarity_pretrained, colors, colors_pretrained, architectureNames, figureNames[2]) #Similarity
plotAccuracy(numberOfParameters_segmentation, accuracies_segmentation, accuracies_segmentation_pretrained, colors[0:2], colors_pretrained[0:2], architectureNames_segmentation, figureNames[3]) #Segmentation
