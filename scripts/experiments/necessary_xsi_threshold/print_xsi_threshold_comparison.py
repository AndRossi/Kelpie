import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--xsi_1_file",
                    type=str,
                    help="")

parser.add_argument("--xsi_5_file",
                    type=str,
                    help="")

parser.add_argument("--xsi_10_file",
                    type=str,
                    help="")

args = parser.parse_args()
xsi1_file = args.xsi_1_file
xsi5_file = args.xsi_5_file
xsi10_file = args.xsi_10_file


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
xsi1_new_ranks = []
xsi5_new_ranks = []
xsi10_new_ranks = []

fact_2_xsi1_explanations = read_necessary_output_end_to_end(xsi1_file)
fact_2_xsi5_explanations = read_necessary_output_end_to_end(xsi5_file)
fact_2_xsi10_explanations = read_necessary_output_end_to_end(xsi10_file)

for fact_to_explain in fact_2_xsi1_explanations:
    xsi1_expl, _, _, original_tail_rank, xsi1_new_tail_rank = fact_2_xsi1_explanations[fact_to_explain]
    xsi5_expl, _, _, _, xsi5_new_tail_rank = fact_2_xsi5_explanations[fact_to_explain]
    xsi10_expl, _, _, _, xsi10_new_tail_rank = fact_2_xsi10_explanations[fact_to_explain]

    original_ranks.append(original_tail_rank)
    xsi1_new_ranks.append(xsi1_new_tail_rank)
    xsi5_new_ranks.append(xsi5_new_tail_rank)
    xsi10_new_ranks.append(xsi10_new_tail_rank)

original_mr = mr(original_ranks)
original_mrr = mrr(original_ranks)
original_h1 = hits_at_k(original_ranks, 1)
original_h3 = hits_at_k(original_ranks, 3)
original_h5 = hits_at_k(original_ranks, 5)
original_h10 = hits_at_k(original_ranks, 10)

xsi1_mr, xsi5_mr, xsi10_mr = mr(xsi1_new_ranks), mr(xsi5_new_ranks), mr(xsi10_new_ranks)
xsi1_mrr, xsi5_mrr, xsi10_mrr = mrr(xsi1_new_ranks), mrr(xsi5_new_ranks), mrr(xsi10_new_ranks)
xsi1_h1, xsi5_h1, xsi10_h1 = hits_at_k(xsi1_new_ranks, 1), hits_at_k(xsi5_new_ranks, 1), hits_at_k(xsi10_new_ranks, 1)
xsi1_h3, xsi5_h3, xsi10_h3 = hits_at_k(xsi1_new_ranks, 3), hits_at_k(xsi5_new_ranks, 3), hits_at_k(xsi10_new_ranks, 3)
xsi1_h5, xsi5_h5, xsi10_h5 = hits_at_k(xsi1_new_ranks, 5), hits_at_k(xsi5_new_ranks, 5), hits_at_k(xsi10_new_ranks, 5)
xsi1_h10, xsi5_h10, xsi10_h10 = hits_at_k(xsi1_new_ranks, 10), hits_at_k(xsi5_new_ranks, 10), hits_at_k(xsi10_new_ranks, 10)

xsi1_mr_diff = round(xsi1_mr - original_mr, 3)
xsi1_mrr_diff = round(xsi1_mrr - original_mrr, 3)
xsi1_h1_diff = round(xsi1_h1 - original_h1, 3)
xsi1_h3_diff = round(xsi1_h3 - original_h3, 3)
xsi1_h5_diff = round(xsi1_h5 - original_h5, 3)
xsi1_h10_diff = round(xsi1_h10 - original_h10, 3)
xsi1_mr_diff_str = "+" + str(xsi1_mr_diff) if xsi1_mr_diff > 0 else str(xsi1_mr_diff)
xsi1_mrr_diff_str = "+" + str(xsi1_mrr_diff) if xsi1_mrr_diff > 0 else str(xsi1_mrr_diff)
xsi1_h1_diff_str = "+" + str(xsi1_h1_diff) if xsi1_h1_diff > 0 else str(xsi1_h1_diff)
xsi1_h3_diff_str = "+" + str(xsi1_h3_diff) if xsi1_h3_diff > 0 else str(xsi1_h3_diff)
xsi1_h5_diff_str = "+" + str(xsi1_h5_diff) if xsi1_h5_diff > 0 else str(xsi1_h5_diff)
xsi1_h10_diff_str = "+" + str(xsi1_h10_diff) if xsi1_h10_diff > 0 else str(xsi1_h10_diff)

xsi5_mr_diff = round(xsi5_mr - original_mr, 3)
xsi5_mrr_diff = round(xsi5_mrr - original_mrr, 3)
xsi5_h1_diff = round(xsi5_h1 - original_h1, 3)
xsi5_h3_diff = round(xsi5_h3 - original_h3, 3)
xsi5_h5_diff = round(xsi5_h5 - original_h5, 3)
xsi5_h10_diff = round(xsi5_h10 - original_h10, 3)
xsi5_mr_diff_str = "+" + str(xsi5_mr_diff) if xsi5_mr_diff > 0 else str(xsi5_mr_diff)
xsi5_mrr_diff_str = "+" + str(xsi5_mrr_diff) if xsi5_mrr_diff > 0 else str(xsi5_mrr_diff)
xsi5_h1_diff_str = "+" + str(xsi5_h1_diff) if xsi5_h1_diff > 0 else str(xsi5_h1_diff)
xsi5_h3_diff_str = "+" + str(xsi5_h3_diff) if xsi5_h3_diff > 0 else str(xsi5_h3_diff)
xsi5_h5_diff_str = "+" + str(xsi5_h5_diff) if xsi5_h5_diff > 0 else str(xsi5_h5_diff)
xsi5_h10_diff_str = "+" + str(xsi5_h10_diff) if xsi5_h10_diff > 0 else str(xsi5_h10_diff)

xsi10_mr_diff = round(xsi10_mr - original_mr, 3)
xsi10_mrr_diff = round(xsi10_mrr - original_mrr, 3)
xsi10_h1_diff = round(xsi10_h1 - original_h1, 3)
xsi10_h3_diff = round(xsi10_h3 - original_h3, 3)
xsi10_h5_diff = round(xsi10_h5 - original_h5, 3)
xsi10_h10_diff = round(xsi10_h10 - original_h10, 3)
xsi10_mr_diff_str = "+" + str(xsi10_mr_diff) if xsi10_mr_diff > 0 else str(xsi10_mr_diff)
xsi10_mrr_diff_str = "+" + str(xsi10_mrr_diff) if xsi10_mrr_diff > 0 else str(xsi10_mrr_diff)
xsi10_h1_diff_str = "+" + str(xsi10_h1_diff) if xsi10_h1_diff > 0 else str(xsi10_h1_diff)
xsi10_h3_diff_str = "+" + str(xsi10_h3_diff) if xsi10_h3_diff > 0 else str(xsi10_h3_diff)
xsi10_h5_diff_str = "+" + str(xsi10_h5_diff) if xsi10_h5_diff > 0 else str(xsi10_h5_diff)
xsi10_h10_diff_str = "+" + str(xsi10_h10_diff) if xsi10_h10_diff > 0 else str(xsi10_h10_diff)

print()
print("Kelpie MR worsening varying Pre-Filter thresholds:")
print("\tξ =  1:\t" + str(xsi1_mr) + " (" + xsi1_mr_diff_str + ")")
print("\tξ =  5:\t" + str(xsi5_mr) + " (" + xsi5_mr_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_mr) + " (" + xsi10_mr_diff_str + ")")
print()
print("Kelpie MRR worsening varying Pre-Filter thresholds:")
print("\tξ =  1:\t" + str(xsi1_mrr) + " (" + xsi1_mrr_diff_str + ")")
print("\tξ =  5:\t" + str(xsi5_mrr) + " (" + xsi5_mrr_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_mrr) + " (" + xsi10_mrr_diff_str + ")")
print()
print("Kelpie H@1 worsening varying the necessary ξ threshold:")
print("\tξ =  1:\t" + str(xsi1_h1) + " (" + xsi1_h1_diff_str + ")")
print("\tξ =  5:\t" + str(xsi5_h1) + " (" + xsi5_h1_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_h1) + " (" + xsi10_h1_diff_str + ")")
print()
print("Kelpie H@3 worsening varying necessary ξ threshold:")
print("\tξ =  1:\t" + str(xsi1_h3) + " (" + xsi1_h3_diff_str + ")")
print("\tξ =  5:\t" + str(xsi5_h3) + " (" + xsi5_h3_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_h3) + " (" + xsi10_h3_diff_str + ")")
print()
print("Kelpie H@5 worsening varying the necessary ξ threshold:")
print("\tξ =  1:\t" + str(xsi1_h5) + " (" + xsi1_h5_diff_str + ")")
print("\tξ =  5:\t" + str(xsi5_h5) + " (" + xsi5_h5_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_h5) + " (" + xsi10_h5_diff_str + ")")
print()
print("Kelpie H@10 worsening varying the necessary ξ threshold:")
print("\tξ = 1:\t" + str(xsi1_h10) + " (" + xsi1_h10_diff_str + ")")
print("\tξ = 5:\t" + str(xsi5_h10) + " (" + xsi5_h10_diff_str + ")")
print("\tξ = 10:\t" + str(xsi10_h10) + " (" + xsi10_h10_diff_str + ")")
print()
