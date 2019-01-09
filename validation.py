# From: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

import numpy as np


# Method to compute the accuracy. Call predict_labels to get the labels for the dataset
def compute_f1(pred_labels, true_labels):

    # print("predictions ", len(label_pred))
    # print("correct labels ", len(label_correct))

    prec = compute_precision(pred_labels, true_labels)
    rec = compute_precision(true_labels, pred_labels)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # a new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:  # first prediction correct
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # scan entire chunk
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # chunk in correct was longer
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision
