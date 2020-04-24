def rank_biased_overlap(list1: list, list2: list):
    """
        Compute Rank-Biased Overlap (RBO) score for two lists.

        The two lists may have different lengths

        :param list1: first ranked list
        :param list2: second ranked list
    """

    longest_list, shortest_list = (list1, list2) if len(list1) > len(list2) else (list2, list1)

    longest_list_set = set()
    shortest_list_set = set()

    # this contains, in each position i, the overlap ratio between list1[:i+1] and list2[:i+1]
    overlap_ratios = []

    numerator = 0
    denominator = 0

    longest_list_index = 0
    shortest_list_index = 0
    while longest_list_index < (len(longest_list)):

        denominator += 1

        # if the shortest list is not over yet
        if shortest_list_index < len(shortest_list):
            cur_longest_list_item = longest_list[longest_list_index]
            cur_shortest_list_item = shortest_list[shortest_list_index]
            if cur_longest_list_item == cur_shortest_list_item:
                numerator += 1
                # don't even need to add them to the set!

            else:
                longest_list_set.add(cur_longest_list_item)
                shortest_list_set.add(cur_shortest_list_item)

                if cur_longest_list_item in shortest_list_set:
                    numerator += 1
                if cur_shortest_list_item in longest_list_set:
                    numerator += 1
        else:
            cur_longest_list_item = longest_list[longest_list_index]
            longest_list_set.add(cur_longest_list_item)

            if cur_longest_list_item in shortest_list_set:
                numerator += 1

        overlap_ratios.append(float(numerator)/float(denominator))

        longest_list_index += 1
        shortest_list_index += 1

    return sum(overlap_ratios)/float(len(overlap_ratios))


def rank_biased_jaccard(list1: list, list2: list):
    """
        Compute Jaccard score for two lists

        :param list1: first ranked list
        :param list2: second ranked list
    """

    longest_list, shortest_list = (list1, list2) if len(list1) > len(list2) else (list2, list1)

    longest_list_set = set()
    shortest_list_set = set()

    # this contains, in each position i, the overlap ratio between list1[:i+1] and list2[:i+1]
    overlap_ratios = []

    intersection_size = 0
    union_size = 0

    longest_list_index = 0
    shortest_list_index = 0
    while longest_list_index < (len(longest_list)):

        # if the shortest list is not over yet
        if shortest_list_index < len(shortest_list):
            cur_longest_list_item = longest_list[longest_list_index]
            cur_shortest_list_item = shortest_list[shortest_list_index]
            if cur_longest_list_item == cur_shortest_list_item:
                intersection_size += 1
                union_size += 1
                # don't even need to add them to the set!

            else:
                longest_list_set.add(cur_longest_list_item)
                shortest_list_set.add(cur_shortest_list_item)

                if cur_longest_list_item in shortest_list_set:
                    intersection_size += 1
                else:
                    union_size += 1

                if cur_shortest_list_item in longest_list_set:
                    intersection_size += 1
                else:
                    union_size += 1

        else:
            cur_longest_list_item = longest_list[longest_list_index]
            longest_list_set.add(cur_longest_list_item)

            if cur_longest_list_item in shortest_list_set:
                intersection_size += 1

        overlap_ratios.append(float(intersection_size)/float(union_size))

        longest_list_index += 1
        shortest_list_index += 1

    return sum(overlap_ratios)/float(len(overlap_ratios))