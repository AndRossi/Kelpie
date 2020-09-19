"""
This module does explanation for ComplEx model
"""
import torch
import numpy

from dataset import ALL_DATASET_NAMES, Dataset, KelpieDataset
from models.complex.model import ComplEx, KelpieComplEx
from optimization.multiclass_nll_optimizer import KelpieMultiClassNLLptimizer
import ranking_similarity

datasets = ALL_DATASET_NAMES


dataset_name = 'FB15k'
model_path = '/home/nvidia/workspace/dbgroup/andrea/kelpie/models/ComplEx_FB15k.pt'
#model_path = './models/ComplEx_FB15K.pt'
optimizer_name = 'Adagrad'
batch_size = 100
max_epochs = 200
dimension = 2000
learning_rate = 0.01
decay1 = 0.9
decay2 = 0.999
init = 1e-3
regularizer_name = "N3"
regularizer_weight = 2.5e-3


def mrr(values):
    return numpy.average(numpy.array([1.0/float(x) for x in values]))

def hits_k(values, k):
    count = 0.0
    for x in values:
        if x <= k:
            count += 1.0
    return float(count)/float(len(values))

def rbo(original_model: ComplEx,
        kelpie_model: KelpieComplEx,
        original_samples: numpy.array,
        kelpie_samples: numpy.array):

    _, original_ranks, original_predictions = original_model.predict_samples(original_samples)
    _, kelpie_ranks, kelpie_predictions = kelpie_model.predict_samples(samples=kelpie_samples, original_mode=False)

    all_original_ranks = []
    for (a, b) in original_ranks:
        all_original_ranks.append(a)
        all_original_ranks.append(b)

    all_kelpie_ranks = []
    for (a, b) in kelpie_ranks:
        all_kelpie_ranks.append(a)
        all_kelpie_ranks.append(b)

    original_mrr = mrr(all_original_ranks)
    kelpie_mrr = mrr(all_kelpie_ranks)
    original_h1 = hits_k(all_original_ranks, 1)
    kelpie_h1 = hits_k(all_kelpie_ranks, 1)

    rbos = []
    for i in range(len(original_samples)):
        _original_sample = original_samples[i]
        _kelpie_sample = kelpie_samples[i]

        original_target_head, _, original_target_tail = _original_sample
        kelpie_target_head, _, kelpie_target_tail = _kelpie_sample

        original_target_head_index, original_target_tail_index = int(original_ranks[i][0] - 1), int(original_ranks[i][1] - 1)
        kelpie_target_head_index, kelpie_target_tail_index = int(kelpie_ranks[i][0] - 1), int(kelpie_ranks[i][1] - 1)

        # get head and tail predictions
        original_head_predictions = original_predictions[i][0]
        kelpie_head_predictions = kelpie_predictions[i][0]
        original_tail_predictions = original_predictions[i][1]
        kelpie_tail_predictions = kelpie_predictions[i][1]

        assert original_head_predictions[original_target_head_index] == original_target_head
        assert kelpie_head_predictions[kelpie_target_head_index] == kelpie_target_head
        assert original_tail_predictions[original_target_tail_index] == original_target_tail
        assert kelpie_tail_predictions[kelpie_target_tail_index] == kelpie_target_tail

        # replace the target head id with the same value (-1 in this case)
        original_head_predictions[original_target_head_index] = -1
        kelpie_head_predictions[kelpie_target_head_index] = -1
        # cut both lists at the max rank that the target head obtained, between original and kelpie model
        max_target_head_index = max(original_target_head_index, kelpie_target_head_index)
        original_head_predictions = original_head_predictions[:max_target_head_index + 1]
        kelpie_head_predictions = kelpie_head_predictions[:max_target_head_index + 1]

        # replace the target tail id with the same value (-1 in this case)
        original_tail_predictions[original_target_tail_index] = -1
        kelpie_tail_predictions[kelpie_target_tail_index] = -1
        # cut both lists at the max rank that the target tail obtained, between original and kelpie model
        max_target_tail_index = max(original_target_tail_index, kelpie_target_tail_index)
        original_tail_predictions = original_tail_predictions[:max_target_tail_index + 1]
        kelpie_tail_predictions = kelpie_tail_predictions[:max_target_tail_index + 1]

        rbos.append(ranking_similarity.rank_biased_overlap(original_head_predictions, kelpie_head_predictions))
        rbos.append(ranking_similarity.rank_biased_overlap(original_tail_predictions, kelpie_tail_predictions))

    avg_rbo = float(sum(rbos)) / float(len(rbos))
    return avg_rbo, original_mrr, kelpie_mrr, original_h1, kelpie_h1

entity_2_params = {
    '/m/09c7w0': [9739, 1203, '/m/06yxd', '/base/biblioness/bibs_location/country', '/m/09c7w0', 'tail'],
    '/m/08mbj32': [4670, 603, '/m/080v2', '/common/topic/webpage./common/webpage/category', '/m/08mbj32', 'tail'],
    '/m/02sdk9v': [3807, 436, '/m/02b13y', '/sports/sports_team/roster./sports/sports_team_roster/position', '/m/02sdk9v', 'tail'],
    '/m/02nzb8': [3754, 452, '/m/02gys2', '/sports/sports_team/roster./soccer/football_roster_position/position', '/m/02nzb8', 'tail'],
    '/m/02h40lc': [2924, 350, '/m/0bwfn', '/education/educational_institution/students_graduates./education/education/major_field_of_study', '/m/02h40lc', 'tail'],
    '/m/02_286': [1581, 188, '/m/07jrjb', '/people/person/places_lived./people/place_lived/location', '/m/02_286', 'tail'],
    '/m/02jknp': [1223, 151, '/m/02jknp', '/people/profession/people_with_this_profession', '/m/03ysmg', 'head'],
    '/m/09sb52': [1177, 145, '/m/027f7dj', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/09sb52', 'tail'],
    '/m/086k8': [979, 109, '/m/02ny6g', '/film/film/distributors./film/film_film_distributor_relationship/distributor', '/m/086k8', 'tail'],
    '/m/05qd_': [824, 107, '/m/06t6dz', '/award/award_nominated_work/award_nominations./award/award_nomination/award_nominee', '/m/05qd_', 'tail'],
    '/m/06b1q': [688, 85, '/m/06b1q', '/sports/sports_position/players./american_football/football_historical_roster_position/team', '/m/070xg', 'head'],
    '/m/016z4k': [575, 59, '/m/01w58n3', '/people/person/profession', '/m/016z4k', 'tail'],
    '/m/018j2': [397, 45, '/m/0l1589', '/music/performance_role/track_performances./music/track_contribution/role', '/m/018j2', 'tail'],
    '/m/07y_7': [345, 39, '/m/03qjg', '/music/performance_role/regular_performances./music/group_membership/role', '/m/07y_7', 'tail'],
    '/m/062z7': [294, 41, '/m/01p896', '/education/educational_institution/students_graduates./education/education/major_field_of_study', '/m/062z7', 'tail'],
    '/m/09n4nb': [250, 21, '/m/047q2wc', '/award/award_winner/awards_won./award/award_honor/ceremony', '/m/09n4nb', 'tail'],
    '/m/02lyr4': [202, 24, '/m/07l8x', '/baseball/baseball_team/current_roster./sports/sports_team_roster/position', '/m/02lyr4', 'tail'],
    '/m/02q1tc5': [168, 16, '/m/02q1tc5', '/award/award_category/winners./award/award_honor/honored_for', '/m/02_1kl', 'head'],
    '/m/02y8z': [167, 23, '/m/018qb4', '/olympics/olympic_games/sports', '/m/02y8z', 'tail'],
    '/m/05zl0': [132, 23, '/m/05zl0', '/award/ranked_item/appears_in_ranked_lists./award/ranking/list', '/m/09g7thr', 'head'],
    '/m/021bk': [123, 20, '/m/021bk', '/award/award_nominee/award_nominations./award/award_nomination/award_nominee', '/m/02qx69', 'head'],
    '/m/049qx': [100, 13, '/m/049qx', '/people/person/profession', '/m/0cbd2', 'head'],
    '/m/09kvv': [91, 7, '/m/09kvv', '/education/university/domestic_tuition./measurement_unit/dated_money_value/currency', '/m/09nqf', 'head'],
    '/m/05kh_': [87, 9, '/m/05kh_', '/film/director/film', '/m/01lsl', 'head'],
    '/m/01tnbn': [72, 7, '/m/0cbd2', '/people/profession/people_with_this_profession', '/m/01tnbn', 'tail'],
    '/m/028d4v': [69, 11, '/m/028d4v', '/people/person/profession', '/m/0dxtg', 'head'],
    '/m/03193l': [43, 5, '/m/03193l', '/common/topic/webpage./common/webpage/category', '/m/08mbj32', 'head'],
    '/m/0g2ff': [29, 4, '/m/0g2ff', '/music/performance_role/regular_performances./music/group_membership/role', '/m/0mkg', 'head'],
    '/m/0269kx': [21, 1, '/m/0h52w', '/protected_sites/natural_or_cultural_site_designation/sites./protected_sites/natural_or_cultural_site_listing/listed_site', '/m/0269kx', 'tail'],
    '/m/028cl7': [5, 1, '/m/03lty', '/music/genre/subgenre', '/m/028cl7', 'tail'],
    '/m/03rg5x': [4, 1, '/m/0cbd2', '/people/profession/people_with_this_profession', '/m/03rg5x', 'tail']
}



#############  LOAD DATASET


#### LOAD DATASET
print("Loading dataset...")
complex_dataset = Dataset(name=dataset_name, separator="\t", load=True)

### LOAD TRAINED ORIGINAL MODEL
print("Loading original trained model...")
original_model = ComplEx(dataset=complex_dataset, dimension=dimension, init_random=True, init_size=init) # type: ComplEx
original_model.load_state_dict(torch.load(model_path))
original_model.to('cuda')

### FOR EACH ENTITY, PERFORM A KELPIE ANALYSIS
for entity_to_explain in entity_2_params:
    train_degree, test_degree, head, relation, tail, perspective = entity_2_params[entity_to_explain]
    print("\nWorking with entity %s (train degree %i; test degree %i): explaining fact <%s, %s, %s>." %
          (entity_to_explain, train_degree, test_degree, head, relation, tail))

    # get the ids of the elements of the fact to explain and the perspective entity
    head_id, relation_id, tail_id = complex_dataset.get_id_for_entity_name(head), \
                                    complex_dataset.get_id_for_relation_name(relation), \
                                    complex_dataset.get_id_for_entity_name(tail)

    original_entity_id = head_id if perspective == 'head' else tail_id
    original_triple = (head_id, relation_id, tail_id)
    original_sample = numpy.array(original_triple)

    # check that the fact to explain is actually a test fact
    assert(original_sample in complex_dataset.test_samples)

    kelpie_dataset = KelpieDataset(dataset=complex_dataset, entity_id=original_entity_id)
    kelpie_entity_id = kelpie_dataset.kelpie_entity_id
    kelpie_triple = (kelpie_entity_id, relation_id, tail_id) if perspective == 'head' else (head_id, relation_id, kelpie_entity_id)
    kelpie_sample = numpy.array(kelpie_triple)

    print("Wrapping the original model in a Kelpie model...")
    kelpie_model = KelpieComplEx(dataset=kelpie_dataset, model=original_model, init_size=1e-3) # type: KelpieComplEx
    kelpie_model.to('cuda')

    print("Extracting samples...")
    original_train_samples = kelpie_dataset.original_train_samples
    original_valid_samples = kelpie_dataset.original_valid_samples
    original_test_samples = kelpie_dataset.original_test_samples
    kelpie_train_samples = kelpie_dataset.kelpie_train_samples
    kelpie_valid_samples = kelpie_dataset.kelpie_valid_samples
    kelpie_test_samples = kelpie_dataset.kelpie_test_samples

    all_original_samples = numpy.vstack((original_train_samples,
                                         original_valid_samples,
                                         original_test_samples))
    all_kelpie_samples = numpy.vstack((kelpie_train_samples,
                                       kelpie_valid_samples,
                                       kelpie_test_samples))

    print("BEFORE TRAINING: Computing RBO between original model predictions and Kelpie model predictions...")

    print("\tTRAIN FACTS: %i" % len(original_train_samples))
    if len(original_train_samples) > 0:
        train_avg_rbo, original_train_mrr, kelpie_train_mrr, original_train_h1, kelpie_train_h1 = \
            rbo(original_model, kelpie_model, original_train_samples, kelpie_train_samples)
        print("\t\tBEFORE TRAINING: TRAIN Average Rank-based overlap: %f" % train_avg_rbo)
        print("\t\tBEFORE TRAINING: TRAIN Original performances: MRR: %f; H1: %f" % (original_train_mrr, original_train_h1))
        print("\t\tBEFORE TRAINING:TRAIN Kelpie performances: MRR: %f; H1: %f" % (kelpie_train_mrr, kelpie_train_h1))

    print("\tVALID FACTS: %i" % len(original_valid_samples))
    if len(original_valid_samples) > 0:
        valid_avg_rbo, original_valid_mrr, kelpie_valid_mrr, original_valid_h1, kelpie_valid_h1 = \
            rbo(original_model, kelpie_model, original_valid_samples, kelpie_valid_samples)
        print("\t\tBEFORE TRAINING: VALID Average Rank-based overlap: %f" % valid_avg_rbo)
        print("\t\tBEFORE TRAINING: VALID Original performances: MRR: %f; H1: %f" % (original_valid_mrr, original_valid_h1))
        print("\t\tBEFORE TRAINING: VALID Kelpie performances: MRR: %f; H1: %f" % (kelpie_valid_mrr, kelpie_valid_h1))

    print("\tTEST FACTS: %i" % len(original_test_samples))
    if len(original_test_samples) > 0:
        test_avg_rbo, original_test_mrr, kelpie_test_mrr, original_test_h1, kelpie_test_h1 = \
            rbo(original_model, kelpie_model, original_test_samples, kelpie_test_samples)
        print("\t\tBEFORE TRAINING: TEST Average Rank-based overlap: %f" % test_avg_rbo)
        print("\t\tBEFORE TRAINING: TEST Original performances: MRR: %f; H1: %f" % (original_test_mrr, original_test_h1))
        print("\t\tBEFORE TRAINING: TEST Kelpie performances: MRR: %f; H1: %f" % (kelpie_test_mrr, kelpie_test_h1))

    print("\tALL FACTS: %i" % len(all_original_samples))
    if len(all_original_samples) > 0:
        all_avg_rbo, original_all_mrr, kelpie_all_mrr, original_all_h1, kelpie_all_h1 = \
            rbo(original_model, kelpie_model, all_original_samples, all_kelpie_samples)
        print("\t\tBEFORE TRAINING: ALL Average Rank-based overlap: %f" % all_avg_rbo)
        print("\t\tBEFORE TRAINING: ALL Original performances: MRR: %f; H1: %f" % (original_all_mrr, original_all_h1))
        print("\t\tBEFORE TRAINING: ALL Kelpie performances: MRR: %f; H1: %f" % (kelpie_all_mrr, kelpie_all_h1))


    print("Running post-training on the Kelpie model...")
    optimizer = KelpieMultiClassNLLptimizer(model=kelpie_model,
                                            optimizer_name=optimizer_name,
                                            batch_size=batch_size,
                                            learning_rate=learning_rate,
                                            decay1=decay1,
                                            decay2=decay2,
                                            regularizer_name=regularizer_name,
                                            regularizer_weight=regularizer_weight)

    optimizer.train(train_samples=kelpie_dataset.kelpie_train_samples, max_epochs=max_epochs)


    print("Computing RBO between original model predictions and Kelpie model predictions...")

    print("\tTRAIN FACTS: %i" % len(original_train_samples))
    if len(original_train_samples) > 0:
        train_avg_rbo, original_train_mrr, kelpie_train_mrr, original_train_h1, kelpie_train_h1 = \
            rbo(original_model, kelpie_model, original_train_samples, kelpie_train_samples)
        print("\t\tTRAIN Average Rank-based overlap: %f" % train_avg_rbo)
        print("\t\tTRAIN Original performances: MRR: %f; H1: %f" % (original_train_mrr, original_train_h1))
        print("\t\tTRAIN Kelpie performances: MRR: %f; H1: %f" % (kelpie_train_mrr, kelpie_train_h1))

    print("\tVALID FACTS: %i" % len(original_valid_samples))
    if len(original_valid_samples) > 0:
        valid_avg_rbo, original_valid_mrr, kelpie_valid_mrr, original_valid_h1, kelpie_valid_h1 = \
            rbo(original_model, kelpie_model, original_valid_samples, kelpie_valid_samples)
        print("\t\tVALID Average Rank-based overlap: %f" % valid_avg_rbo)
        print("\t\tVALID Original performances: MRR: %f; H1: %f" % (original_valid_mrr, original_valid_h1))
        print("\t\tVALID Kelpie performances: MRR: %f; H1: %f" % (kelpie_valid_mrr, kelpie_valid_h1))

    print("\tTEST FACTS: %i" % len(original_test_samples))
    if len(original_test_samples) > 0:
        test_avg_rbo, original_test_mrr, kelpie_test_mrr, original_test_h1, kelpie_test_h1 = \
            rbo(original_model, kelpie_model, original_test_samples, kelpie_test_samples)
        print("\t\tTEST Average Rank-based overlap: %f" % test_avg_rbo)
        print("\t\tTEST Original performances: MRR: %f; H1: %f" % (original_test_mrr, original_test_h1))
        print("\t\tTEST Kelpie performances: MRR: %f; H1: %f" % (kelpie_test_mrr, kelpie_test_h1))

    print("\tALL FACTS: %i" % len(all_original_samples))
    if len(all_original_samples) > 0:
        all_avg_rbo, original_all_mrr, kelpie_all_mrr, original_all_h1, kelpie_all_h1 = \
            rbo(original_model, kelpie_model, all_original_samples, all_kelpie_samples)
        print("\t\tALL Average Rank-based overlap: %f" % all_avg_rbo)
        print("\t\tALL Original performances: MRR: %f; H1: %f" % (original_all_mrr, original_all_h1))
        print("\t\tALL Kelpie performances: MRR: %f; H1: %f" % (kelpie_all_mrr, kelpie_all_h1))

print("\nDone.")