"""Строим модель"""

from typing import List, Dict

from keras.models import Model

from data_generator import DataGenerator


def predict(model: Model,
            data_generator: DataGenerator):
    prediction = [None] * sum(len(data_generator.get_indices_and_lengths(i)) for i in range(len(data_generator)))
    for i in range(len(data_generator)):
        model_input = data_generator[i]
        indices_and_lengths = data_generator.get_indices_and_lengths(i)
        softmax_prediction = model.predict(model_input)
        # Some models return more then one output. We assume that we use first output as prediction.
        print(softmax_prediction)
        if isinstance(softmax_prediction, list):
            softmax_prediction = softmax_prediction[0]
        class_prediction = softmax_prediction.argmax(axis=-1)
        print(class_prediction)
        for sent_pred, (idx, length) in zip(class_prediction, indices_and_lengths):
            sent_labels = sent_pred[-length:]
            prediction[idx] = sent_labels
    return prediction

