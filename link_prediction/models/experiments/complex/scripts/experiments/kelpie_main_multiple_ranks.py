"""
This module does explanation for ComplEx-N3 model
"""
import torch
import numpy

from dataset import Dataset, KelpieDataset
from models.complex.model import ComplEx, KelpieComplEx
from optimization.multiclass_nll_optimizer import KelpieMultiClassNLLptimizer

def average(values):
    return numpy.average(numpy.array(values))

def mean_reciprocal_rank_error(couples):
    error_sum = 0.0
    for (a, b) in couples:
        error_sum += abs(1/float(a)-1/float(b))
    return error_sum / float(len(couples))


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

entity_2_params = {
    '/m/09c7w0': [9739, 1203, '/m/06yxd', '/base/biblioness/bibs_location/country', '/m/09c7w0', 'tail'],
    '/m/02sdk9v': [3807, 436, '/m/02b13y', '/sports/sports_team/roster./sports/sports_team_roster/position', '/m/02sdk9v', 'tail'],
    '/m/08mbj32': [4670, 603, '/m/080v2', '/common/topic/webpage./common/webpage/category', '/m/08mbj32', 'tail'],
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
    kelpie_model = KelpieComplEx(dataset=kelpie_dataset, model=original_model, init_size=1e-3)  # type: KelpieComplEx
    kelpie_model.to('cuda')

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

    original_train_samples = kelpie_dataset.original_train_samples
    original_valid_samples = kelpie_dataset.original_valid_samples
    original_test_samples = kelpie_dataset.original_test_samples
    kelpie_train_samples = kelpie_dataset.kelpie_train_samples
    kelpie_valid_samples = kelpie_dataset.kelpie_valid_samples
    kelpie_test_samples = kelpie_dataset.kelpie_test_samples

    original_direct_samples = numpy.vstack((original_train_samples, original_valid_samples, original_test_samples))
    kelpie_direct_samples = numpy.vstack((kelpie_train_samples, kelpie_valid_samples, kelpie_test_samples))

    _, original_ranks, _ = original_model.predict_samples(samples=original_direct_samples)
    _, kelpie_ranks, _ = kelpie_model.predict_samples(samples=kelpie_direct_samples, original_mode=False)

    train_ranks = []
    valid_ranks = []
    test_ranks = []

    for i in range(len(original_train_samples)):
        (head_id, relation_id, tail_id) = original_train_samples[i]
        cur_original_triple = (head_id, relation_id, tail_id)
        (head_id, relation_id, tail_id) = kelpie_train_samples[i]
        cur_kelpie_triple = (head_id, relation_id, tail_id)

        original_head_rank, original_tail_rank = original_ranks[i]
        kelpie_head_rank, kelpie_tail_rank = kelpie_ranks[i]

        train_ranks.append((original_head_rank, kelpie_head_rank))
        train_ranks.append((original_tail_rank, kelpie_tail_rank))

    for i in range(len(original_valid_samples)):
        (head_id, relation_id, tail_id) = original_valid_samples[i]
        cur_original_triple = (head_id, relation_id, tail_id)
        (head_id, relation_id, tail_id) = kelpie_valid_samples[i]
        cur_kelpie_triple = (head_id, relation_id, tail_id)

        original_head_rank, original_tail_rank = original_ranks[i]
        kelpie_head_rank, kelpie_tail_rank = kelpie_ranks[i]

        valid_ranks.append((original_head_rank, kelpie_head_rank))
        valid_ranks.append((original_tail_rank, kelpie_tail_rank))

    for i in range(len(original_test_samples)):
        (head_id, relation_id, tail_id) = original_test_samples[i]
        cur_original_sample = (head_id, relation_id, tail_id)
        (head_id, relation_id, tail_id) = kelpie_test_samples[i]
        cur_kelpie_sample = (head_id, relation_id, tail_id)

        original_head_rank, original_tail_rank = original_ranks[i]
        kelpie_head_rank, kelpie_tail_rank = kelpie_ranks[i]

        test_ranks.append((original_head_rank, kelpie_head_rank))
        test_ranks.append((original_tail_rank, kelpie_tail_rank))


    print("TRAIN facts: %i" % len(train_ranks))
    if len(train_ranks) > 0:
        print("Average across TRAIN original scores: " + str(average([abs(couple[0]) for couple in train_ranks])))
        print("Average across TRAIN kelpie scores: " + str(average([abs(couple[1]) for couple in train_ranks])))
        print("TRAIN MRRE: " + str(mean_reciprocal_rank_error(couples=train_ranks)))
    print()

    print("VALID facts (direct and inverse): %i" % len(valid_ranks))
    if len(valid_ranks) > 0:
        print("Average across VALID scores: " + str(average([abs(couple[0]) for couple in valid_ranks])))
        print("Average across VALID scores: " + str(average([abs(couple[1]) for couple in valid_ranks])))
        print("VALID MRRE: " + str(mean_reciprocal_rank_error(couples=valid_ranks)))
    print()

    print("TEST facts (direct and inverse): %i" % len(test_ranks))
    if len(test_ranks) > 0:
        print("Average across TEST original scores: " + str(average([abs(couple[0]) for couple in test_ranks])))
        print("Average across TEST kelpie scores: " + str(average([abs(couple[1]) for couple in test_ranks])))
        print("TEST MRRE: " + str(mean_reciprocal_rank_error(couples=test_ranks)))
    print()

    all_ranks = train_ranks + valid_ranks + test_ranks
    print("Average across ALL original scores: " + str(average([abs(couple[0]) for couple in all_ranks])))
    print("Average across ALL kelpie scores: " + str(average([abs(couple[1]) for couple in all_ranks])))
    print("ALL MRRE: " + str(mean_reciprocal_rank_error(couples=all_ranks)) + "\n")
