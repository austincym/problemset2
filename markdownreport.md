{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOAKdBd5976TzxMUbE0pLE6",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/austincym/problemset2/blob/main/markdownreport.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "U5FSq5Ln7Qet"
      },
      "outputs": [],
      "source": [
        "Creating a Markdown report for the code you provided:\n",
        "\n",
        "# Convolutional Neural Network (CNN) Filter Visualization Report\n",
        "\n",
        "In this report, we will visualize the feature extraction process using random filters applied to an image. The code is written in Python using popular libraries such as NumPy, OpenCV (cv2), and Matplotlib.\n",
        "\n",
        "## Image Preprocessing\n",
        "\n",
        "We begin by resizing and preprocessing the image:\n",
        "\n",
        "```python\n",
        "import numpy as np\n",
        "import cv2\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Assuming 'resized_image' is your resized image from earlier\n",
        "image = np.array(resized_image)\n",
        "\n",
        "# Resize the image to 224x224\n",
        "resized_image = image.resize((224, 224))\n",
        "\n",
        "# Convert the image to grayscale\n",
        "grayscale_image = resized_image.convert(\"L\")\n",
        "```\n",
        "\n",
        "We first resize the image to a standard size (224x224) and then convert it to grayscale for simplicity.\n",
        "\n",
        "## Filter Visualization\n",
        "\n",
        "Next, we generate 10 random filters and visualize their effect on the image:\n",
        "\n",
        "```python\n",
        "# Create 10 random filters\n",
        "filters = [np.random.randn(3, 3) for _ in range(10)]\n",
        "\n",
        "# Create a figure to display filters and feature maps\n",
        "plt.figure(figsize=(15, 5))\n",
        "\n",
        "# Display each filter and corresponding feature map\n",
        "for i, filter in enumerate(filters):\n",
        "    feature_map = cv2.filter2D(image, -1, filter)\n",
        "\n",
        "    # Plot the filter\n",
        "    plt.subplot(2, 10, i + 1)\n",
        "    plt.imshow(filter, cmap='gray')\n",
        "    plt.axis('off')\n",
        "\n",
        "    # Plot the corresponding feature map\n",
        "    plt.subplot(2, 10, i + 11)\n",
        "    plt.imshow(feature_map, cmap='gray')\n",
        "    plt.axis('off')\n",
        "\n",
        "plt.show()\n",
        "```\n",
        "\n",
        "We create 10 random filters, apply each filter to the grayscale image using OpenCV's `filter2D` function, and display both the filters and their corresponding feature maps.\n",
        "\n",
        "## Conclusion\n",
        "\n",
        "This code demonstrates how convolutional filters can be used to extract features from an image. By visualizing these filters and their effects, we can gain insights into how convolutional layers work in a Convolutional Neural Network (CNN).\n",
        "\n",
        "Feel free to modify the code to experiment with different filter sizes, numbers, or other parameters to further understand and explore image processing with CNNs."
      ]
    }
  ]
}