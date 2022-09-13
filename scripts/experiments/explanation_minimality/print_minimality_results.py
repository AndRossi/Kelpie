import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--full_expl_output_file",
                    type=str,
                    help="")

parser.add_argument("--sampled_expl_output_file",
                    type=str,
                    help="")

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="")

args = parser.parse_args()
full_expl_output_filepath = args.full_expl_output_file
sampled_expl_output_filepath = args.sampled_expl_output_file

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
    return count/float(len(ranks))


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0/float(rank)
    return reciprocal_rank_sum/float(len(ranks))


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return rank_sum/float(len(ranks))


if not os.path.isfile(full_expl_output_filepath):
    print(f"File {full_expl_output_filepath} does not exist.")
    exit()
if not os.path.isfile(sampled_expl_output_filepath):
    print(f"File {sampled_expl_output_filepath} does not exist.")
    exit()

original_ranks = []
full_explanation_new_ranks = []
sampled_explanation_new_ranks = []

if args.mode == 'necessary':
    fact_2_full_explanations = read_necessary_output_end_to_end(full_expl_output_filepath)
    fact_2_sampled_explanations = read_necessary_output_end_to_end(sampled_expl_output_filepath)

    for fact_to_explain in fact_2_full_explanations:
        kelpie_full_expl, _, _, kelpie_original_tail_rank, kelpie_full_new_tail_rank = fact_2_full_explanations[fact_to_explain]
        original_ranks.append(kelpie_original_tail_rank)
        full_explanation_new_ranks.append(kelpie_full_new_tail_rank)

        # the outcome of sampling length-1 explanation is not written in the output file,
        # because sampling them leads to empty explanations, and thus to keeping the original tail rank.
        # So we add the kelpie_original_tail_rank to the sampled_explanation_new_ranks
        if fact_to_explain in fact_2_sampled_explanations:
            kelpie_sampled_expl, _, _, sampled_original_tail_rank, kelpie_sampled_new_tail_rank = fact_2_sampled_explanations[fact_to_explain]
            sampled_explanation_new_ranks.append(kelpie_sampled_new_tail_rank)
        else:
            sampled_explanation_new_ranks.append(kelpie_original_tail_rank)

else:
    fact_2_full_explanations, _ = read_sufficient_output_end_to_end(full_expl_output_filepath)
    fact_2_sampled_explanations, _ = read_sufficient_output_end_to_end(sampled_expl_output_filepath)

    for fact_to_convert in fact_2_full_explanations:
        kelpie_full_expl, _, _, kelpie_original_tail_rank, kelpie_full_new_tail_rank = fact_2_full_explanations[fact_to_convert]
        original_ranks.append(kelpie_original_tail_rank)
        full_explanation_new_ranks.append(kelpie_full_new_tail_rank)

        # the outcome of sampling length-1 explanation is not written in the output file,
        # because sampling them leads to empty explanations, and thus to keeping the original tail rank.
        # So we add the kelpie_original_tail_rank to the sampled_explanation_new_ranks
        if fact_to_convert in fact_2_sampled_explanations:
            kelpie_sampled_expl, _, _, sampled_original_tail_rank, kelpie_sampled_new_tail_rank = fact_2_sampled_explanations[fact_to_convert]
            sampled_explanation_new_ranks.append(kelpie_sampled_new_tail_rank)
        else:
            sampled_explanation_new_ranks.append(kelpie_original_tail_rank)

original_mr = mr(original_ranks)
original_mrr = mrr(original_ranks)
original_h1 = hits_at_k(original_ranks, 1)
original_h3 = hits_at_k(original_ranks, 3)
original_h5 = hits_at_k(original_ranks, 5)
original_h10 = hits_at_k(original_ranks, 10)

kelpie_full_mr = mr(full_explanation_new_ranks)
kelpie_full_mrr = mrr(full_explanation_new_ranks)
kelpie_full_h1 = hits_at_k(full_explanation_new_ranks, 1)
kelpie_full_h3 = hits_at_k(full_explanation_new_ranks, 3)
kelpie_full_h5 = hits_at_k(full_explanation_new_ranks, 5)
kelpie_full_h10 = hits_at_k(full_explanation_new_ranks, 10)

kelpie_sampled_mr = mr(sampled_explanation_new_ranks)
kelpie_sampled_mrr = mrr(sampled_explanation_new_ranks)
kelpie_sampled_h1 = hits_at_k(sampled_explanation_new_ranks, 1)
kelpie_sampled_h3 = hits_at_k(sampled_explanation_new_ranks, 3)
kelpie_sampled_h5 = hits_at_k(sampled_explanation_new_ranks, 5)
kelpie_sampled_h10 = hits_at_k(sampled_explanation_new_ranks, 10)

mr_full_difference = kelpie_full_mr - original_mr
mrr_full_difference = kelpie_full_mrr - original_mrr
h1_full_difference = kelpie_full_h1 - original_h1
h3_full_difference = kelpie_full_h3 - original_h3
h5_full_difference = kelpie_full_h5 - original_h5
h10_full_difference = kelpie_full_h10 - original_h10

mr_sampled_difference = kelpie_sampled_mr - original_mr
mrr_sampled_difference = kelpie_sampled_mrr - original_mrr
h1_sampled_difference = kelpie_sampled_h1 - original_h1
h3_sampled_difference = kelpie_sampled_h3 - original_h3
h5_sampled_difference = kelpie_sampled_h5 - original_h5
h10_sampled_difference = kelpie_sampled_h10 - original_h10

mr_full_difference_str = "+" + str(round(mr_full_difference, 3)) if mr_full_difference > 0 else str(round(mr_full_difference, 3))
mrr_full_difference_str = "+" + str(round(mrr_full_difference, 3)) if mrr_full_difference > 0 else str(round(mrr_full_difference, 3))
h1_full_difference_str = "+" + str(round(h1_full_difference, 3)) if h1_full_difference > 0 else str(round(h1_full_difference, 3))
h3_full_difference_str = "+" + str(round(h3_full_difference, 3)) if h3_full_difference > 0 else str(round(h3_full_difference, 3))
h5_full_difference_str = "+" + str(round(h5_full_difference, 3)) if h5_full_difference > 0 else str(round(h5_full_difference, 3))
h10_full_difference_str = "+" + str(round(h10_full_difference, 3)) if h10_full_difference > 0 else str(round(h10_full_difference, 3))

mr_sampled_difference_str = "+" + str(round(mr_sampled_difference, 3)) if mr_sampled_difference > 0 else str(round(mr_sampled_difference, 3))
mrr_sampled_difference_str = "+" + str(round(mrr_sampled_difference, 3)) if mrr_sampled_difference > 0 else str(round(mrr_sampled_difference, 3))
h1_sampled_difference_str = "+" + str(round(h1_sampled_difference, 3)) if h1_sampled_difference > 0 else str(round(h1_sampled_difference, 3))
h3_sampled_difference_str = "+" + str(round(h3_sampled_difference, 3)) if h3_sampled_difference > 0 else str(round(h3_sampled_difference, 3))
h5_sampled_difference_str = "+" + str(round(h5_sampled_difference, 3)) if h5_sampled_difference > 0 else str(round(h5_sampled_difference, 3))
h10_sampled_difference_str = "+" + str(round(h10_sampled_difference, 3)) if h10_sampled_difference > 0 else str(round(h10_sampled_difference, 3))

mr_effectiveness_variation = str(min(round(((mr_sampled_difference - mr_full_difference)/mr_full_difference) * 100, 2), 100.0)) + '%'
mrr_effectiveness_variation = str(min(round(((mrr_sampled_difference - mrr_full_difference)/mrr_full_difference)*100, 2), 100.0)) + '%'
h1_effectiveness_variation = str(min(round(((h1_sampled_difference - h1_full_difference)/h1_full_difference)*100, 2), 100.0)) + '%'
h3_effectiveness_variation = str(min(round(((h3_sampled_difference - h3_full_difference)/h3_full_difference)*100, 2), 100.0)) + '%'
h5_effectiveness_variation = str(min(round(((h5_sampled_difference - h5_full_difference)/h3_full_difference)*100, 2), 100.0)) + '%'
h10_effectiveness_variation = str(min(round(((h10_sampled_difference - h10_full_difference)/h10_full_difference)*100, 2), 100.0)) + '%'

print()
print("Full Explanation MR variation:\t\t" + mr_full_difference_str)
print("Sampled Explanation MR variation:\t" + mr_sampled_difference_str)
print("Effectiveness decrease after sampling:\t" + mr_effectiveness_variation)
print()
print("Full Explanation MRR variation:\t\t" + mrr_full_difference_str)
print("Sampled Explanation MRR variation:\t" + mrr_sampled_difference_str)
print("Effectiveness decrease after sampling :\t" + mrr_effectiveness_variation)
print()
print("Full Explanation H@1 variation:\t\t" + h1_full_difference_str)
print("Sampled Explanation H@1 variation:\t" + h1_sampled_difference_str)
print("Effectiveness decrease after sampling :\t" + h1_effectiveness_variation)
print()
print("Full Explanation H@3 variation:\t\t" + h3_full_difference_str)
print("Sampled Explanation H@3 variation:\t" + h3_sampled_difference_str)
print("Effectiveness decrease after sampling :\t" + h3_effectiveness_variation)
print()
print("Full Explanation H@5 variation:\t\t" + h5_full_difference_str)
print("Sampled Explanation H@5 variation:\t" + h5_sampled_difference_str)
print("Effectiveness decrease after sampling :\t" + h5_effectiveness_variation)
print()
print("Full Explanation H@10 variation:\t" + h10_full_difference_str)
print("Sampled Explanation H@10 variation:\t" + h10_sampled_difference_str)
print("Effectiveness decrease after sampling :\t" + h10_effectiveness_variation)
print()
