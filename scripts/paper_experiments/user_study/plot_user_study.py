import os.path

import numpy
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")


def count_items(items_list):
    result = [0, 0, 0, 0]
    for item in items_list:
        result[item] += 1

    return result


def read_users_list(input_filepath):
    _users = set()
    with open(input_filepath, "r") as infile:
        lines = infile.readlines()

        for _line in lines[1:]:
            _bits = _line.strip().replace('"', '').split(",")

            time = _bits[0]
            mail = _bits[1]

            _users.add(mail.lower().replace(".", ""))
    return _users


def read_necessary_module(input_filepath, _users):
    with open(input_filepath, "r") as infile:
        lines = infile.readlines()

        _answers_1 = []
        _answers_2 = []
        _answers_3_text = []
        _answers_3 = []

        for _line in lines[1:]:
            _bits = _line.strip().replace('"', '').replace(", 𝗻𝗲𝗹", "").split(",")

            time = _bits[0]
            mail = _bits[1]

            i = 2

            if mail.lower().replace(".", "") not in _users:
                continue

            while i < len(_bits):
                _answers_1.append(int(_bits[i]))
                _answers_2.append(int(_bits[i + 1]))
                _answers_3_text.append(_bits[i + 2])
                i += 3

        for _answer in _answers_3_text:
            if _answer.startswith("Il sistema non riuscirebbe più"):
                _answers_3.append(0)
            elif _answer.startswith("Il sistema diventerebbe ancora"):
                _answers_3.append(1)
            elif _answer.startswith("Non so"):
                _answers_3.append(2)
            elif _answer.startswith("Non cambierebbe nulla"):
                _answers_3.append(3)
            else:
                raise Exception("ao")

    return _answers_1, _answers_2, _answers_3


def read_sufficient_module(input_filepath, _users):
    with open(input_filepath, "r") as infile:
        lines = infile.readlines()

        _answers_1 = []
        _answers_2 = []
        _answers_3_text = []
        _answers_3 = []

        for _line in lines[1:]:
            _bits = _line.strip().replace('"', '').split(",")

            time = _bits[0]
            mail = _bits[1]

            i = 2

            if mail.lower().replace(".", "") not in _users:
                continue

            while i < len(_bits):
                _answers_1.append(int(_bits[i]))
                _answers_2.append(int(_bits[i + 1]))
                _answers_3_text.append(_bits[i + 2])
                i += 3

        for _answer in _answers_3_text:
            if _answer.startswith("Il sistema comincerebbe"):
                _answers_3.append(0)
            elif _answer.startswith("Il sistema non riuscirebbe"):
                _answers_3.append(1)
            elif _answer.startswith("Non so"):
                _answers_3.append(2)
            elif _answer.startswith("Non cambierebbe nulla"):
                _answers_3.append(3)
            else:
                raise Exception("ao")

    return _answers_1, _answers_2, _answers_3

cur_dir = os.path.dirname(os.path.realpath(__file__))
users = read_users_list(os.path.join(cur_dir, "user_study_transe_sufficient.csv"))
answers_1_CN, answers_2_CN, answers_3_CN = read_necessary_module(os.path.join(cur_dir, "user_study_complex_necessary.csv"), users)
answers_1_TN, answers_2_TN, answers_3_TN = read_necessary_module(os.path.join(cur_dir, "user_study_transe_necessary.csv"), users)
answers_1_CS, answers_2_CS, answers_3_CS = read_sufficient_module(os.path.join(cur_dir, "user_study_complex_sufficient.csv"), users)
answers_1_TS, answers_2_TS, answers_3_TS = read_sufficient_module(os.path.join(cur_dir, "user_study_transe_sufficient.csv"), users)

print("ComplEx trust:\t" + str(numpy.average(answers_1_CN + answers_1_CS)))
print("TransE trust:\t" + str(numpy.average(answers_1_TN + answers_1_TS)))
print("Conceptual Understanding:\t" + str(numpy.average(answers_2_CN + answers_2_TN + answers_2_CS + answers_2_TS)))


GRAY = "#4D4D4D"
BLUE = "#5DA5DA"
ORANGE = "#FAA43A"
GREEN = "#60BD68"
PINK = "#F17CB0"
BROWN = "#B2912F"
PURPLE = "#B276B2"
YELLOW = "#DECF3F"
RED = "#F15854"

# Pie chart
labels = ["Correct\nAnswer", ' Wrong\nAnswer', "Do not\nknow", 'Nothing\n would\n  change']
sizes = count_items(answers_3_CN + answers_3_TN + answers_3_CS + answers_3_TS)
# colors
colors = [GREEN, RED, ORANGE, PURPLE]

fig1, ax1 = plt.subplots()

wedges, texts, _ = ax1.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', pctdistance=0.42, startangle=30,
                           labeldistance=1.1,
                           textprops={'color': "black", 'fontsize': "11"})
# draw circle
centre_circle = plt.Circle((0, 0), 0.65, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')

plt.show()
