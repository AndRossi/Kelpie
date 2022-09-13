import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--k10_file",
                    type=str,
                    help="")

parser.add_argument("--k20_file",
                    type=str,
                    help="")

parser.add_argument("--k30_file",
                    type=str,
                    help="")

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="")

args = parser.parse_args()
k10_file = args.k10_file
k20_file = args.k20_file
k30_file = args.k30_file


def read_necessary_output_end_to_end(filepath):
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        exit()

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
    if not os.path.isfile(filepath):
        print(f"File {filepath} does not exist.")
        exit()

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


original_ranks = []
k10_new_ranks = []
k20_new_ranks = []
k30_new_ranks = []

if args.mode == 'necessary':
    fact_2_k10_explanations = read_necessary_output_end_to_end(k10_file)
    fact_2_k20_explanations = read_necessary_output_end_to_end(k20_file)
    fact_2_k30_explanations = read_necessary_output_end_to_end(k30_file)

    for fact_to_explain in fact_2_k10_explanations:

        k10_expl, _, _, original_tail_rank, k10_new_tail_rank = fact_2_k10_explanations[fact_to_explain]
        k20_expl, _, _, _, k20_new_tail_rank = fact_2_k20_explanations[fact_to_explain]
        k30_expl, _, _, _, k30_new_tail_rank = fact_2_k30_explanations[fact_to_explain]

        original_ranks.append(original_tail_rank)
        k10_new_ranks.append(k10_new_tail_rank)
        k20_new_ranks.append(k20_new_tail_rank)
        k30_new_ranks.append(k30_new_tail_rank)

else:
    fact_2_k10_explanations, _ = read_sufficient_output_end_to_end(k10_file)
    fact_2_k20_explanations, _ = read_sufficient_output_end_to_end(k20_file)
    fact_2_k30_explanations, _ = read_sufficient_output_end_to_end(k30_file)

    for fact_to_convert in fact_2_k10_explanations:
        k10_expl, _, _, original_tail_rank, k10_new_tail_rank = fact_2_k10_explanations[fact_to_convert]
        k20_expl, _, _, _, k20_new_tail_rank = fact_2_k20_explanations[fact_to_convert]
        k30_expl, _, _, _, k30_new_tail_rank = fact_2_k30_explanations[fact_to_convert]

        original_ranks.append(original_tail_rank)
        k10_new_ranks.append(k10_new_tail_rank)
        k20_new_ranks.append(k20_new_tail_rank)
        k30_new_ranks.append(k30_new_tail_rank)


original_mr = mr(original_ranks)
original_mrr = mrr(original_ranks)
original_h1 = hits_at_k(original_ranks, 1)
original_h3 = hits_at_k(original_ranks, 3)
original_h5 = hits_at_k(original_ranks, 5)
original_h10 = hits_at_k(original_ranks, 10)

k10_mr, k20_mr, k30_mr = mr(k10_new_ranks), mr(k20_new_ranks), mr(k30_new_ranks)
k10_mrr, k20_mrr, k30_mrr = mrr(k10_new_ranks), mrr(k20_new_ranks), mrr(k30_new_ranks)
k10_h1, k20_h1, k30_h1 = hits_at_k(k10_new_ranks, 1), hits_at_k(k20_new_ranks, 1), hits_at_k(k30_new_ranks, 1)
k10_h3, k20_h3, k30_h3 = hits_at_k(k10_new_ranks, 3), hits_at_k(k20_new_ranks, 3), hits_at_k(k30_new_ranks, 3)
k10_h5, k20_h5, k30_h5 = hits_at_k(k10_new_ranks, 5), hits_at_k(k20_new_ranks, 5), hits_at_k(k30_new_ranks, 5)
k10_h10, k20_h10, k30_h10 = hits_at_k(k10_new_ranks, 10), hits_at_k(k20_new_ranks, 10), hits_at_k(k30_new_ranks, 10)

k10_mr_diff = round(k10_mr - original_mr, 3)
k10_mrr_diff = round(k10_mrr - original_mrr, 3)
k10_h1_diff = round(k10_h1 - original_h1, 3)
k10_h3_diff = round(k10_h3 - original_h3, 3)
k10_h5_diff = round(k10_h5 - original_h5, 3)
k10_h10_diff = round(k10_h10 - original_h10, 3)
k10_mr_diff_str = "+" + str(k10_mr_diff) if k10_mr_diff > 0 else str(k10_mr_diff)
k10_mrr_diff_str = "+" + str(k10_mrr_diff) if k10_mrr_diff > 0 else str(k10_mrr_diff)
k10_h1_diff_str = "+" + str(k10_h1_diff) if k10_h1_diff > 0 else str(k10_h1_diff)
k10_h3_diff_str = "+" + str(k10_h3_diff) if k10_h3_diff > 0 else str(k10_h3_diff)
k10_h5_diff_str = "+" + str(k10_h5_diff) if k10_h5_diff > 0 else str(k10_h5_diff)
k10_h10_diff_str = "+" + str(k10_h10_diff) if k10_h10_diff > 0 else str(k10_h10_diff)

k20_mr_diff = round(k20_mr - original_mr, 3)
k20_mrr_diff = round(k20_mrr - original_mrr, 3)
k20_h1_diff = round(k20_h1 - original_h1, 3)
k20_h3_diff = round(k20_h3 - original_h3, 3)
k20_h5_diff = round(k20_h5 - original_h5, 3)
k20_h10_diff = round(k20_h10 - original_h10, 3)
k20_mr_diff_str = "+" + str(k20_mr_diff) if k20_mr_diff > 0 else str(k20_mr_diff)
k20_mrr_diff_str = "+" + str(k20_mrr_diff) if k20_mrr_diff > 0 else str(k20_mrr_diff)
k20_h1_diff_str = "+" + str(k20_h1_diff) if k20_h1_diff > 0 else str(k20_h1_diff)
k20_h3_diff_str = "+" + str(k20_h3_diff) if k20_h3_diff > 0 else str(k20_h3_diff)
k20_h5_diff_str = "+" + str(k20_h5_diff) if k20_h5_diff > 0 else str(k20_h5_diff)
k20_h10_diff_str = "+" + str(k20_h10_diff) if k20_h10_diff > 0 else str(k20_h10_diff)

k30_mr_diff = round(k30_mr - original_mr, 3)
k30_mrr_diff = round(k30_mrr - original_mrr, 3)
k30_h1_diff = round(k30_h1 - original_h1, 3)
k30_h3_diff = round(k30_h3 - original_h3, 3)
k30_h5_diff = round(k30_h5 - original_h5, 3)
k30_h10_diff = round(k30_h10 - original_h10, 3)
k30_mr_diff_str = "+" + str(k30_mr_diff) if k30_mr_diff > 0 else str(k30_mr_diff)
k30_mrr_diff_str = "+" + str(k30_mrr_diff) if k30_mrr_diff > 0 else str(k30_mrr_diff)
k30_h1_diff_str = "+" + str(k30_h1_diff) if k30_h1_diff > 0 else str(k30_h1_diff)
k30_h3_diff_str = "+" + str(k30_h3_diff) if k30_h3_diff > 0 else str(k30_h3_diff)
k30_h5_diff_str = "+" + str(k30_h5_diff) if k30_h5_diff > 0 else str(k30_h5_diff)
k30_h10_diff_str = "+" + str(k30_h10_diff) if k30_h10_diff > 0 else str(k30_h10_diff)

print()
print("Kelpie MR worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_mr) + " (" + k10_mr_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_mr) + " (" + k20_mr_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_mr) + " (" + k30_mr_diff_str + ")")
print()
print("Kelpie MRR worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_mrr) + " (" + k10_mrr_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_mrr) + " (" + k20_mrr_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_mrr) + " (" + k30_mrr_diff_str + ")")
print()
print("Kelpie H@1 worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_h1) + " (" + k10_h1_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_h1) + " (" + k20_h1_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_h1) + " (" + k30_h1_diff_str + ")")
print()
print("Kelpie H@3 worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_h3) + " (" + k10_h3_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_h3) + " (" + k20_h3_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_h3) + " (" + k30_h3_diff_str + ")")
print()
print("Kelpie H@5 worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_h5) + " (" + k10_h5_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_h5) + " (" + k20_h5_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_h5) + " (" + k30_h5_diff_str + ")")
print()
print("Kelpie H@10 worsening varying Pre-Filter thresholds:")
print("\tPre-filter threshold 10:\t" + str(k10_h10) + " (" + k10_h10_diff_str + ")")
print("\tPre-filter threshold 20:\t" + str(k20_h10) + " (" + k20_h10_diff_str + ")")
print("\tPre-filter threshold 30:\t" + str(k30_h10) + " (" + k30_h10_diff_str + ")")
print()