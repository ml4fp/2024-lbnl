import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    From scikit-learn: plots a confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix"

    # Compute confusion matrix
    if len(y_true.shape) > 1 and len(y_pred.shape) > 1:
        cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    else:
        cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(title)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        ylabel="True label",
        xlabel="Predicted label",
    )

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def plot_roc(fpr, tpr, auc, labels, linestyle, legend=True):
    for label in labels:
        plt.plot(
            tpr[label],
            fpr[label],
            label=f"{label.replace('j_', '')}, AUC = {auc[label] * 100:.1f}%",
            linestyle=linestyle,
        )
    plt.semilogy()
    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    plt.ylim(0.001, 1)
    plt.grid(True)
    if legend:
        plt.legend(loc="upper left")


def roc_data(y, predict_test, labels):

    df = pd.DataFrame()

    fprs = {}
    tprs = {}
    aucs = {}

    for i, label in enumerate(labels):
        df[label] = y[:, i] if len(labels) > 1 else y
        df[f"{label}_pred"] = predict_test[:, i] if len(labels) > 1 else predict_test

        fprs[label], tprs[label], _ = roc_curve(df[label], df[f"{label}_pred"])

        aucs[label] = auc(fprs[label], tprs[label])
    return fprs, tprs, aucs


def make_roc(y, predict_test, labels, linestyle="-", legend=True):

    if "j_index" in labels:
        labels.remove("j_index")

    fprs, tprs, aucs = roc_data(y, predict_test, labels)
    plot_roc(fprs, tprs, aucs, labels, linestyle, legend=legend)


def normalize_image(image):
    """Rescale the constrast in an image based on the noise (used for displays and the CNN)"""
    sigmaG_coeff = 0.7413
    image = image.reshape(21, 21)

    per25, per50, per75 = np.percentile(image, [25, 50, 75])
    sigmaG = sigmaG_coeff * (per75 - per25)
    # sigma clip image, remove background, and normalize to unity
    image[image < (per50 - 2 * sigmaG)] = per50 - 2 * sigmaG
    image -= np.min(image)
    image /= np.sum(image)

    return image


def plot_image_array(
    images, nrows=2, ncols=5, figsize=[8, 4], nx=21, ny=21, title="", subtitle=False, class_true=None, classes=None
):
    """Plot an array of images"""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0, left=0.07, right=0.95, wspace=0.05, bottom=0.15)
    for indx in np.arange(nrows * ncols):
        i = int(indx / ncols)
        j = indx % ncols
        if i == 0:
            ax[i][j].xaxis.set_major_formatter(plt.NullFormatter())
        if j != 0:
            ax[i][j].yaxis.set_major_formatter(plt.NullFormatter())

        ax[i][j].imshow(images[indx].reshape(nx, ny), cmap="gray", origin="lower")
        if subtitle:
            ax[i][j].set_title(
                (
                    f"True class: {np.argmax(class_true[indx])}, "
                    + f"Predicted class: {np.argmax(classes[indx])}\n"
                    + f"Prob(class 1): {classes[indx, 1]}"
                )
            )

    fig.suptitle(title)
    ax[0][0].set_ylabel("$y$")
    ax[nrows - 1][int(ncols / 2)].set_xlabel("$x$")


def plot_model_history(history):
    """Plot the training and validation history for a TensorFlow network"""

    # Extract loss and accuracy
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    n_epochs = len(loss)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].plot(np.arange(n_epochs), loss, label="Training")
    ax[0].plot(np.arange(n_epochs), val_loss, label="Validation")
    ax[0].legend()
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(np.arange(n_epochs), acc, label="Training")
    ax[1].plot(np.arange(n_epochs), val_acc, label="Validation")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")