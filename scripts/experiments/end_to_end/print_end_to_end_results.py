import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--end_to_end_output_file",
                    type=str,
                    help="")

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="")

args = parser.parse_args()
end_to_end_output_filepath = args.end_to_end_output_file


def read_necessary_output_end_to_end(filepath):

    fact_to_explain_2_details = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")

            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _explanation_bits = bits[3:-4]
            assert len(_explanation_bits)%3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                _explanation_bits[i+1], \
                                                                                _explanation_bits[i+2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i+=3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_explain_2_details[_fact_to_explain] = (_explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)

    return fact_to_explain_2_details

def read_sufficient_output_end_to_end(filepath):
    fact_to_convert_2_details = {}
    fact_to_convert_2_original_fact_to_explain = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")
            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _head_to_convert, _rel_to_convert, _tail_to_convert = bits[3:6]

            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _fact_to_convert = (_head_to_convert, _rel_to_convert, _tail_to_convert)

            _explanation_bits = bits[6:-4]
            assert len(_explanation_bits)%3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                _explanation_bits[i+1], \
                                                                                _explanation_bits[i+2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i+=3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            fact_to_convert_2_details[_fact_to_convert] = (_explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)
            fact_to_convert_2_original_fact_to_explain[_fact_to_convert] = _fact_to_explain
    return fact_to_convert_2_details, fact_to_convert_2_original_fact_to_explain


def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return round(count/float(len(ranks)), 3)


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0/float(rank)
    return round(reciprocal_rank_sum/float(len(ranks)), 3)


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return round(rank_sum/float(len(ranks)), 3)


if not os.path.isfile(end_to_end_output_filepath):
    print("The file does not exist.")
    exit()

original_ranks = []
new_ranks = []

if args.mode == 'necessary':
    fact_2_kelpie_explanations = read_necessary_output_end_to_end(end_to_end_output_filepath)

    for fact_to_explain in fact_2_kelpie_explanations:

        kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank = fact_2_kelpie_explanations[fact_to_explain]

        original_ranks.append(kelpie_original_tail_rank)
        new_ranks.append(kelpie_new_tail_rank)

else:
    fact_2_kelpie_explanations, _ = read_sufficient_output_end_to_end(end_to_end_output_filepath)

    for fact_to_convert in fact_2_kelpie_explanations:
        kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank = fact_2_kelpie_explanations[fact_to_convert]

        original_ranks.append(kelpie_original_tail_rank)
        new_ranks.append(kelpie_new_tail_rank)


original_mr = mr(original_ranks)
original_mrr = mrr(original_ranks)
original_h1 = hits_at_k(original_ranks, 1)
original_h3 = hits_at_k(original_ranks, 3)
original_h5 = hits_at_k(original_ranks, 5)
original_h10 = hits_at_k(original_ranks, 10)

kelpie_mr = mr(new_ranks)
kelpie_mrr = mrr(new_ranks)
kelpie_h1 = hits_at_k(new_ranks, 1)
kelpie_h3 = hits_at_k(new_ranks, 3)
kelpie_h5 = hits_at_k(new_ranks, 5)
kelpie_h10 = hits_at_k(new_ranks, 10)

mr_difference = round(kelpie_mr - original_mr, 3)
mrr_difference = round(kelpie_mrr - original_mrr, 3)
h1_difference = round(kelpie_h1 - original_h1, 3)
h3_difference = round(kelpie_h3 - original_h3, 3)
h5_difference = round(kelpie_h5 - original_h5, 3)
h10_difference = round(kelpie_h10 - original_h10, 3)

mr_difference_str = "+" + str(mr_difference) if mr_difference > 0 else str(mr_difference)
mrr_difference_str = "+" + str(mrr_difference) if mrr_difference > 0 else str(mrr_difference)
h1_difference_str = "+" + str(h1_difference) if h1_difference > 0 else str(h1_difference)
h3_difference_str = "+" + str(h3_difference) if h3_difference > 0 else str(h3_difference)
h5_difference_str = "+" + str(h5_difference) if h5_difference > 0 else str(h5_difference)
h10_difference_str = "+" + str(h10_difference) if h10_difference > 0 else str(h10_difference)

print("Original MR:\t" + str(original_mr))
print("Kelpie MR:\t" + str(kelpie_mr) + " (" + mr_difference_str + ")")
print()
print("Original MRR:\t" + str(original_mrr))
print("Kelpie MRR:\t" + str(kelpie_mrr) + " (" + mrr_difference_str + ")")
print()
print("Original H@1:\t" + str(original_h1))
print("Kelpie H@1:\t" + str(kelpie_h1) + " (" + h1_difference_str + ")")
print()
print("Original H@3:\t" + str(original_h3))
print("Kelpie H@3:\t" + str(kelpie_h3) + " (" + h3_difference_str + ")")
print()
print("Original H@5:\t" + str(original_h5))
print("Kelpie H@5:\t" + str(kelpie_h5) + " (" + h5_difference_str + ")")
print()
print("Original H@10:\t" + str(original_h10))
print("Kelpie H@10:\t" + str(kelpie_h10) + " (" + h10_difference_str + ")")
