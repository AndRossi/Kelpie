import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--topology_prefilter_file",
                    type=str,
                    help="")

parser.add_argument("--typebased_prefilter_file",
                    type=str,
                    help="")

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    help="")

args = parser.parse_args()
topology_prefilter_file = args.topology_prefilter_file
typebased_prefilter_file = args.typebased_prefilter_file


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
topology_prefilter_new_ranks = []
typebased_prefilter_new_ranks = []

if args.mode == 'necessary':
    fact_2_topology_prefilter_expl = read_necessary_output_end_to_end(topology_prefilter_file)
    fact_2_typebased_prefilter_expl = read_necessary_output_end_to_end(typebased_prefilter_file)

    for fact_to_explain in fact_2_topology_prefilter_expl:

        topology_prefilter_expl, _, _, original_tail_rank, topology_prefilter_new_tail_rank = fact_2_topology_prefilter_expl[fact_to_explain]
        typebased_prefilter_expl, _, _, _, typebased_prefilter_new_tail_rank = fact_2_typebased_prefilter_expl[fact_to_explain]

        original_ranks.append(original_tail_rank)
        topology_prefilter_new_ranks.append(topology_prefilter_new_tail_rank)
        typebased_prefilter_new_ranks.append(typebased_prefilter_new_tail_rank)

else:
    fact_2_topology_prefilter_expl, _ = read_sufficient_output_end_to_end(topology_prefilter_file)
    fact_2_typebased_prefilter_expl, _ = read_sufficient_output_end_to_end(typebased_prefilter_file)

    for fact_to_convert in fact_2_topology_prefilter_expl:
        topology_prefilter_expl, _, _, original_tail_rank, topology_prefilter_new_tail_rank = fact_2_topology_prefilter_expl[fact_to_convert]
        typebased_prefilter_expl, _, _, _, typebased_prefilter_new_tail_rank = fact_2_typebased_prefilter_expl[fact_to_convert]

        original_ranks.append(original_tail_rank)
        topology_prefilter_new_ranks.append(topology_prefilter_new_tail_rank)
        typebased_prefilter_new_ranks.append(typebased_prefilter_new_tail_rank)


original_mr = mr(original_ranks)
original_mrr = mrr(original_ranks)
original_h1 = hits_at_k(original_ranks, 1)
original_h3 = hits_at_k(original_ranks, 3)
original_h5 = hits_at_k(original_ranks, 5)
original_h10 = hits_at_k(original_ranks, 10)

topology_prefilter_mr, typebased_prefilter_mr = mr(topology_prefilter_new_ranks), mr(typebased_prefilter_new_ranks)
topology_prefilter_mrr, typebased_prefilter_mrr = mrr(topology_prefilter_new_ranks), mrr(typebased_prefilter_new_ranks)
topology_prefilter_h1, typebased_prefilter_h1 = hits_at_k(topology_prefilter_new_ranks, 1), hits_at_k(typebased_prefilter_new_ranks, 1)
topology_prefilter_h3, typebased_prefilter_h3 = hits_at_k(topology_prefilter_new_ranks, 3), hits_at_k(typebased_prefilter_new_ranks, 3)
topology_prefilter_h5, typebased_prefilter_h5 = hits_at_k(topology_prefilter_new_ranks, 5), hits_at_k(typebased_prefilter_new_ranks, 5)
topology_prefilter_h10, typebased_prefilter_h10 = hits_at_k(topology_prefilter_new_ranks, 10), hits_at_k(typebased_prefilter_new_ranks, 10)

topology_prefilter_mr_diff = round(topology_prefilter_mr - original_mr, 3)
topology_prefilter_mrr_diff = round(topology_prefilter_mrr - original_mrr, 3)
topology_prefilter_h1_diff = round(topology_prefilter_h1 - original_h1, 3)
topology_prefilter_h3_diff = round(topology_prefilter_h3 - original_h3, 3)
topology_prefilter_h5_diff = round(topology_prefilter_h5 - original_h5, 3)
topology_prefilter_h10_diff = round(topology_prefilter_h10 - original_h10, 3)
topology_prefilter_mr_diff_str = "+" + str(topology_prefilter_mr_diff) if topology_prefilter_mr_diff > 0 else str(topology_prefilter_mr_diff)
topology_prefilter_mrr_diff_str = "+" + str(topology_prefilter_mrr_diff) if topology_prefilter_mrr_diff > 0 else str(topology_prefilter_mrr_diff)
topology_prefilter_h1_diff_str = "+" + str(topology_prefilter_h1_diff) if topology_prefilter_h1_diff > 0 else str(topology_prefilter_h1_diff)
topology_prefilter_h3_diff_str = "+" + str(topology_prefilter_h3_diff) if topology_prefilter_h3_diff > 0 else str(topology_prefilter_h3_diff)
topology_prefilter_h5_diff_str = "+" + str(topology_prefilter_h5_diff) if topology_prefilter_h5_diff > 0 else str(topology_prefilter_h5_diff)
topology_prefilter_h10_diff_str = "+" + str(topology_prefilter_h10_diff) if topology_prefilter_h10_diff > 0 else str(topology_prefilter_h10_diff)

typebased_prefilter_mr_diff = round(typebased_prefilter_mr - original_mr, 3)
typebased_prefilter_mrr_diff = round(typebased_prefilter_mrr - original_mrr, 3)
typebased_prefilter_h1_diff = round(typebased_prefilter_h1 - original_h1, 3)
typebased_prefilter_h3_diff = round(typebased_prefilter_h3 - original_h3, 3)
typebased_prefilter_h5_diff = round(typebased_prefilter_h5 - original_h5, 3)
typebased_prefilter_h10_diff = round(typebased_prefilter_h10 - original_h10, 3)
typebased_prefilter_mr_diff_str = "+" + str(typebased_prefilter_mr_diff) if typebased_prefilter_mr_diff > 0 else str(typebased_prefilter_mr_diff)
typebased_prefilter_mrr_diff_str = "+" + str(typebased_prefilter_mrr_diff) if typebased_prefilter_mrr_diff > 0 else str(typebased_prefilter_mrr_diff)
typebased_prefilter_h1_diff_str = "+" + str(typebased_prefilter_h1_diff) if typebased_prefilter_h1_diff > 0 else str(typebased_prefilter_h1_diff)
typebased_prefilter_h3_diff_str = "+" + str(typebased_prefilter_h3_diff) if typebased_prefilter_h3_diff > 0 else str(typebased_prefilter_h3_diff)
typebased_prefilter_h5_diff_str = "+" + str(typebased_prefilter_h5_diff) if typebased_prefilter_h5_diff > 0 else str(typebased_prefilter_h5_diff)
typebased_prefilter_h10_diff_str = "+" + str(typebased_prefilter_h10_diff) if typebased_prefilter_h10_diff > 0 else str(typebased_prefilter_h10_diff)

print()
print("Kelpie MR worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_mr) + " (" + topology_prefilter_mr_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_mr) + " (" + typebased_prefilter_mr_diff_str + ")")
print()
print("Kelpie MRR worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_mrr) + " (" + topology_prefilter_mrr_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_mrr) + " (" + typebased_prefilter_mrr_diff_str + ")")
print()
print("Kelpie H@1 worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_h1) + " (" + topology_prefilter_h1_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_h1) + " (" + typebased_prefilter_h1_diff_str + ")")
print()
print("Kelpie H@3 worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_h3) + " (" + topology_prefilter_h3_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_h3) + " (" + typebased_prefilter_h3_diff_str + ")")
print()
print("Kelpie H@5 worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_h5) + " (" + topology_prefilter_h5_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_h5) + " (" + typebased_prefilter_h5_diff_str + ")")
print()
print("Kelpie H@10 worsening varying Pre-Filter type:")
print("\tTopology-based Pre-Filter:\t" + str(topology_prefilter_h10) + " (" + topology_prefilter_h10_diff_str + ")")
print("\tType-based Pre-Filter:\t\t" + str(typebased_prefilter_h10) + " (" + typebased_prefilter_h10_diff_str + ")")
print()