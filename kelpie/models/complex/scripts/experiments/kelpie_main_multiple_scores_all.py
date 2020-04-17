"""
This module does explanation for ComplEx-N3 model
"""
import torch
import numpy
from kelpie.models.complex.complex import ComplEx, KelpieComplEx
from kelpie.models.complex.dataset import ComplExDataset, KelpieComplExDataset
from kelpie.models.complex.optimizers import KelpieComplExOptimizer

def average(values):
    return numpy.average(numpy.array(values))

def mse(couples):

    squared_errors_sum = 0.0
    for (a, b) in couples:
        squared_errors_sum += (a-b)**2
    return squared_errors_sum / float(len(couples))

dataset_name = 'FB15k'
model_path = '/home/nvidia/workspace/dbgroup/andrea/kelpie/models/ComplEx_FB15k_no_reg.pt'
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
regularizer_weight = 0

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
complex_dataset = ComplExDataset(name=dataset_name, separator="\t", load=True)

### LOAD TRAINED ORIGINAL MODEL
print("Loading original trained model...")
original_model = ComplEx(dataset=complex_dataset, dimension=dimension, init_random=True, init_size=init)
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

    kelpie_dataset = KelpieComplExDataset(dataset=complex_dataset, entity_id=original_entity_id)
    kelpie_entity_id = kelpie_dataset.kelpie_entity_id
    kelpie_triple = (kelpie_entity_id, relation_id, tail_id) if perspective == 'head' else (head_id, relation_id, kelpie_entity_id)
    kelpie_sample = numpy.array(kelpie_triple)

    print("Wrapping the original model in a Kelpie model...")
    kelpie_model = KelpieComplEx(dataset=kelpie_dataset, model=original_model, init_size=1e-3)
    kelpie_model.to('cuda')

    print("Running post-training on the Kelpie model...")
    optimizer = KelpieComplExOptimizer(model=kelpie_model,
                                       optimizer_name=optimizer_name,
                                       batch_size=batch_size,
                                       learning_rate=learning_rate,
                                       decay1=decay1,
                                       decay2=decay2,
                                       regularizer_name=regularizer_name,
                                       regularizer_weight=regularizer_weight)
    optimizer.train(train_samples=kelpie_dataset.kelpie_train_samples, max_epochs=max_epochs)

    original_direct_samples = numpy.vstack((kelpie_dataset.original_train_samples,
                                            kelpie_dataset.original_valid_samples,
                                            kelpie_dataset.original_test_samples))
    original_inverse_samples = kelpie_dataset.invert_samples(original_direct_samples)

    kelpie_direct_samples = numpy.vstack((kelpie_dataset.kelpie_train_samples,
                                          kelpie_dataset.kelpie_valid_samples,
                                          kelpie_dataset.kelpie_test_samples))
    kelpie_inverse_samples = kelpie_dataset.invert_samples(kelpie_direct_samples)

    print("Computing scores on the original entity...")
    original_direct_scores = original_model.temporary_test(original_direct_samples)
    original_inverse_scores = original_model.temporary_test(original_inverse_samples)

    print("Computing scores on the kelpie entity...")
    kelpie_direct_scores = kelpie_model.temporary_test(kelpie_direct_samples)
    kelpie_inverse_scores = kelpie_model.temporary_test(kelpie_inverse_samples)

    print()


    all_scores = []
    for (head_id, rel_id, tail_id) in original_direct_scores:

        original_direct_sample = (head_id, rel_id, tail_id)
        original_inverse_sample = (tail_id, rel_id + kelpie_dataset.num_original_relations, head_id)

        # if the head is the entity was kelpized, then we need to analyze the scores of the direct facts
        if head_id == original_entity_id:

            if tail_id != head_id:
                kelpie_direct_sample = (kelpie_entity_id, rel_id, tail_id)
            else:
                kelpie_direct_sample = (kelpie_entity_id, rel_id, kelpie_entity_id)

            kelpie_inverse_sample = (kelpie_direct_sample[2],
                                     rel_id + kelpie_dataset.num_original_relations,
                                     kelpie_direct_sample[0])

            for i in range(len(original_direct_scores[original_direct_sample])):
                all_scores.append(
                    (original_direct_scores[original_direct_sample][i],
                     kelpie_direct_scores[kelpie_direct_sample][i]))

        # if the tail is the entity was kelpized, then we need to analyze the scores of the inverse facts

        elif tail_id == original_entity_id:
            if tail_id != head_id:
                kelpie_direct_sample = (head_id, rel_id, kelpie_entity_id)
            else:
                kelpie_direct_sample = (kelpie_entity_id, rel_id, kelpie_entity_id)

            kelpie_inverse_sample = (kelpie_direct_sample[2],
                                     rel_id + kelpie_dataset.num_original_relations,
                                     kelpie_direct_sample[0])

            for i in range(len(original_inverse_scores[original_inverse_sample])):
                all_scores.append(
                    (original_inverse_scores[original_inverse_sample][i],
                     kelpie_inverse_scores[kelpie_inverse_sample][i]))

    avg_original_scores = average([abs(couple[0]) for couple in all_scores])
    avg_kelpie_scores = average([abs(couple[1]) for couple in all_scores])
    error = mse(couples=all_scores)

    print("Average original scores: " + str(avg_original_scores) + "\n")
    print("Average kelpie scores: " + str(avg_kelpie_scores) + "\n")
    print("MSE: " + str(error) + "\n")