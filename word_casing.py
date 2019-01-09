"""Извлекаем признаки из регистра и наличия цифр в слове"""


def get_casing(word):
    num_digits = sum(int(ch.isdigit()) for ch in word)

    digit_fraction = num_digits / float(len(word))

    casing = [
        float(word.isdigit()),
        float(digit_fraction > 0.5),  # В основном цифры
        float(num_digits > 0),  # содержит цифры
        float(word.islower()),  # All lower case
        float(word.isupper()),  # All upper case
        float(word[0].isupper())]  # is a title
    return casing


CASING_PADDING = [0.0] * 6

