"""
This module does explanation for ComplEx model
"""
import argparse
import torch
import numpy

from kelpie import kelpie_perturbation, ranking_similarity
from kelpie.dataset import ALL_DATASET_NAMES, Dataset
from kelpie.evaluation import KelpieEvaluator
from kelpie.kelpie_dataset import KelpieDataset
from kelpie.models.complex.model import ComplEx, KelpieComplEx
from kelpie.models.complex.optimizer import KelpieComplExOptimizer

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument('--dataset',
                    choices=datasets,
                    help="Dataset in {}".format(datasets),
                    required=True)

parser.add_argument('--model_path',
                    help="Path to the model to explain the predictions of",
                    required=True)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument('--optimizer',
                    choices=optimizers,
                    default='Adagrad',
                    help="Optimizer in {} to use in post-training".format(optimizers))

parser.add_argument('--batch_size',
                    default=100,
                    type=int,
                    help="Batch size to use in post-training")

parser.add_argument('--max_epochs',
                    default=200,
                    type=int,
                    help="Number of epochs to run in post-training")

parser.add_argument('--dimension',
                    default=1000,
                    type=int,
                    help="Factorization rank.")

parser.add_argument('--learning_rate',
                    default=1e-1,
                    type=float,
                    help="Learning rate")

parser.add_argument('--perspective',
                    choices=['head', 'tail'],
                    default='head',
                    help="Explanation perspective in {}".format(['head', 'tail']))

parser.add_argument('--reg',
                    default=0,
                    type=float,
                    help="Regularization weight"
)

parser.add_argument('--init',
                    default=1e-3,
                    type=float,
                    help="Initial scale"
)

parser.add_argument('--decay1',
                    default=0.9,
                    type=float,
                    help="Decay rate for the first moment estimate in Adam"
)
parser.add_argument('--decay2',
                    default=0.999,
                    type=float,
                    help="Decay rate for second moment estimate in Adam"
)

#   E.G.    explain  why  /m/02mjmr (Barack Obama)  is predicted as the head for
#   /m/02mjmr (Barack Obama) 	/people/person/ethnicity	/m/033tf_ (Irish American)
#   or
#   /m/02mjmr (Barack Obama)	/people/person/places_lived./people/place_lived/location	/m/02hrh0_ (Honolulu)
args = parser.parse_args()



facts_to_explain = [
    ('/m/0174qm', '/location/location/containedby', '/m/02jx1'),
    ('/m/06yrj6', '/award/award_nominee/award_nominations./award/award_nomination/nominated_for', '/m/01q_y0'),
    ('/m/05bnq8', '/organization/organization/headquarters./location/mailing_address/citytown', '/m/01qh7'),
    ('/m/035qgm', '/base/x2010fifaworldcupsouthafrica/world_cup_squad/current_world_cup_squad./base/x2010fifaworldcupsouthafrica/current_world_cup_squad/current_club', '/m/0kz4w'),
    ('/m/027t8fw', '/film/cinematographer/film', '/m/03z106'),
    ('/m/01xlqd', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/09c7w0'),
    ('/m/07bsj', '/people/person/place_of_birth', '/m/0nbwf'),
    ('/m/017_pb', '/influence/influence_node/influenced_by', '/m/053yx'),
    ('/m/01jz6x', '/people/person/ethnicity', '/m/041rx'),
    ('/m/03gt46z', '/award/award_ceremony/awards_presented./award/award_honor/honored_for', '/m/02rv_dz'),
    ('/m/03l7tr', '/sports/sports_team/roster./soccer/football_roster_position/position', '/m/0dgrmp'),
    ('/m/05np2', '/influence/influence_node/influenced', '/m/01hb6v'),
    ('/m/05ls3r', '/sports/sports_team/roster./sports/sports_team_roster/position', '/m/02sdk9v'),
    ('/m/0mwh1', '/location/location/containedby', '/m/05tbn'),
    ('/m/047hpm', '/film/actor/film./film/performance/film', '/m/02z3r8t'),
    ('/m/05dkbr', '/sports/sports_team/roster./sports/sports_team_roster/position', '/m/02sdk9v'),
    ('/m/0lg0r', '/location/hud_foreclosure_area/total_90_day_vacant_residential_addresses./measurement_unit/dated_integer/source', '/m/0jbk9'),
    ('/m/024vjd', '/award/award_category/nominees./award/award_nomination/award_nominee', '/m/0149xx'),
    ('/m/0b_6qj', '/time/event/locations', '/m/0ftxw'),
    ('/m/033g54', '/sports/sports_team/roster./sports/sports_team_roster/position', '/m/02_j1w'),
    ('/m/0k4gf', '/music/artist/genre', '/m/0l8gh'),
    ('/m/05clg8', '/music/record_label/artist', '/m/01wqmm8'),
    ('/m/01bt59', '/education/field_of_study/students_majoring./education/education/institution', '/m/049dk'),
    ('/m/05ywg', '/travel/travel_destination/how_to_get_here./travel/transportation/mode_of_transportation', '/m/01bjv'),
    ('/m/03_9x6', '/soccer/football_team/current_roster./sports/sports_team_roster/position', '/m/02nzb8'),
    ('/m/036k0s', '/location/administrative_division/capital./location/administrative_division_capital_relationship/capital', '/m/02qjb7z'),
    ('/m/0196bp', '/sports/sports_team/roster./soccer/football_roster_position/position', '/m/0dgrmp'),
    ('/m/02yxjs', '/education/educational_institution/students_graduates./education/education/major_field_of_study', '/m/02jfc'),
    ('/m/01h2_6', '/people/person/gender', '/m/05zppz'),
    ('/m/01vyp_', '/music/artist/genre', '/m/017_qw'),
    ('/m/02h30z', '/education/educational_institution/colors', '/m/03wkwg'),
    ('/m/013b6_', '/people/ethnicity/people', '/m/0jcx'),
    ('/m/02js_6', '/award/award_nominee/award_nominations./award/award_nomination/nominated_for', '/m/016dj8'),
    ('/m/0cnk2q', '/sports/sports_team/roster./soccer/football_roster_position/player', '/m/08jbxf'),
    ('/m/02lfp4', '/award/award_winner/awards_won./award/award_honor/honored_for', '/m/0jqn5'),
    ('/m/02yv6b', '/music/genre/artists', '/m/017f4y'),
    ('/m/03cw411', '/film/film/subjects', '/m/0fx2s'),
    ('/m/0cc97st', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/01crd5'),
    ('/m/0fb0v', '/music/record_label/artist', '/m/01f2q5'),
    ('/m/032ft5', '/government/legislative_session/members./government/government_position_held/legislative_sessions', '/m/070mff'),
    ('/m/0kbvv', '/olympics/olympic_games/sports', '/m/03tmr'),
    ('/m/0c41qv', '/film/production_company/films', '/m/03b_fm5'),
    ('/m/01b195', '/film/film/genre', '/m/01jfsb'),
    ('/m/0jzphpx', '/award/award_ceremony/awards_presented./award/award_honor/award_winner', '/m/0178rl'),
    ('/m/01cl2y', '/music/record_label/artist', '/m/02qwg'),
    ('/m/07jbh', '/olympics/olympic_sport/athletes./olympics/olympic_athlete_affiliation/country', '/m/01pj7'),
    ('/m/01v9l67', '/award/award_nominee/award_nominations./award/award_nomination/award_nominee', '/m/015t56'),
    ('/m/01v0fn1', '/award/award_winner/awards_won./award/award_honor/award_winner', '/m/02qmncd'),
    ('/m/03j1p2n', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/054ks3'),
    ('/m/02mmwk', '/film/film/starring./film/performance/actor', '/m/0294fd'),
    ('/m/04tgp', '/user/tsegaran/random/taxonomy_subject/entry./user/tsegaran/random/taxonomy_entry/taxonomy', '/m/04n6k'),
    ('/m/04n52p6', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/059j2'),
    ('/m/0kv238', '/film/film/featured_film_locations', '/m/04jpl'),
    ('/m/016jny', '/music/genre/artists', '/m/01sb5r'),
    ('/m/025m8l', '/award/award_category/winners./award/award_honor/ceremony', '/m/0jzphpx'),
    ('/m/07r1h', '/award/award_nominee/award_nominations./award/award_nomination/nominated_for', '/m/02p86pb'),
    ('/m/01xqw', '/music/performance_role/regular_performances./music/group_membership/role', '/m/0dwsp'),
    ('/m/01srq2', '/film/film/other_crew./film/film_crew_gig/film_crew_role', '/m/09zzb8'),
    ('/m/081yw', '/location/location/contains', '/m/0mmr1'),
    ('/m/02q52q', '/film/film/release_date_s./film/film_regional_release_date/film_release_distribution_medium', '/m/029j_'),
    ('/m/02cyfz', '/award/award_winner/awards_won./award/award_honor/award', '/m/054krc'),
    ('/m/01j7mr', '/tv/tv_program/languages', '/m/02h40lc'),
    ('/m/023p29', '/award/award_winner/awards_won./award/award_honor/award_winner', '/m/03bxwtd'),
    ('/m/06q8qh', '/film/film/executive_produced_by', '/m/02_340'),
    ('/m/09q23x', '/award/award_nominated_work/award_nominations./award/award_nomination/award', '/m/02x73k6'),
    ('/m/06jzh', '/people/person/spouse_s./people/marriage/type_of_union', '/m/04ztj'),
    #('/m/02h40lc', '/education/field_of_study/students_majoring./education/education/institution', '/m/0187nd'),
    #('/m/0gqy2', '/award/award_category/nominees./award/award_nomination/award_nominee', '/m/0341n5'),
    #('/m/02g_6x', '/sports/sports_position/players./american_football/football_roster_position/team', '/m/05g3v'),
    #('/m/04ztj', '/people/marriage_union_type/unions_of_this_type./people/marriage/spouse', '/m/0184jc'),
    #('/m/02ppm4q', '/award/award_category/winners./award/award_honor/award_winner', '/m/027f7dj'),
    #('/m/08mbj32', '/common/annotation_category/annotations./common/webpage/topic', '/m/07v5q'),
    #('/m/01by1l', '/award/award_category/nominees./award/award_nomination/award_nominee', '/m/0gdh5'),
    #('/m/0nbcg', '/people/profession/people_with_this_profession', '/m/01wmcbg'),
    #('/m/06pj8', '/people/person/employment_history./business/employment_tenure/company', '/m/016tw3'),
    #('/m/018ctl', '/olympics/olympic_games/participating_countries', '/m/06c1y'),
    #('/m/04ztj', '/people/marriage_union_type/unions_of_this_type./people/marriage/spouse', '/m/02v_4xv'),
    #('/m/09c7w0', '/location/location/time_zones', '/m/02lcqs'),
    #('/m/0l14qv', '/music/performance_role/track_performances./music/track_contribution/role', '/m/01wy6'),
    #('/m/0f4x7', '/award/award_category/winners./award/award_honor/honored_for', '/m/0bl5c'),
    #('/m/06t2t', '/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month', '/m/04w_7'),
    #('/m/01795t', '/award/award_nominee/award_nominations./award/award_nomination/award_nominee', '/m/02qzjj'),
    #('/m/02hnl', '/music/performance_role/guest_performances./music/recording_contribution/performance_role', '/m/03bx0bm'),
    #('/m/0gr4k', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0pd57'),
    #('/m/0342h', '/music/instrument/instrumentalists', '/m/019g40'),
    #('/m/02qyntr', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0b6tzs'),
    #('/m/03npn', '/film/film_genre/films_in_this_genre', '/m/0gj9qxr'),
    #('/m/08mbj5d', '/common/annotation_category/annotations./common/webpage/topic', '/m/03hbbc'),
    #('/m/060c4', '/business/job_title/people_with_this_title./business/employment_tenure/company', '/m/01bfjy'),
    #('/m/030qb3t', '/location/location/people_born_here', '/m/044mfr'),
    #('/m/06n90', '/film/film_genre/films_in_this_genre', '/m/0gh65c5'),
    #('/m/02qyntr', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/01mgw'),
    #('/m/05p09zm', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0n83s'),
    #('/m/0342h', '/music/performance_role/regular_performances./music/group_membership/member', '/m/03h_fk5'),
    #('/m/057xs89', '/award/award_category/nominees./award/award_nomination/award_nominee', '/m/01qqtr'),
    #('/m/040njc', '/award/award_category/winners./award/award_honor/honored_for', '/m/011yg9'),
    #('/m/02x73k6', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0sxlb'),
    #('/m/02x4w6g', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0gyfp9c'),
    #('/m/026t6', '/music/performance_role/track_performances./music/track_contribution/role', '/m/06ncr')
]

for (head, relation, tail) in facts_to_explain:
    #deterministic!
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True

    #############  LOAD DATASET

    # get the fact to explain and its perspective entity
    entity_to_explain = head if args.perspective.lower() == "head" else tail

    # load the dataset and its training samples
    original_dataset = Dataset(name=args.dataset, separator="\t", load=True)

    # get the ids of the elements of the fact to explain and the perspective entity
    head_id, relation_id, tail_id = original_dataset.get_id_for_entity_name(head), \
                                    original_dataset.get_id_for_relation_name(relation), \
                                    original_dataset.get_id_for_entity_name(tail)
    original_entity_id = head_id if args.perspective == "head" else tail_id

    # create the fact to explain as a numpy array of its ids
    original_triple = (head_id, relation_id, tail_id)
    original_sample = numpy.array(original_triple)

    # check that the fact to explain is actually a test fact
    assert(original_sample in original_dataset.test_samples)


    #############   INITIALIZE MODELS AND THEIR STRUCTURES
    # instantiate and load the original model from filesystem
    original_model = ComplEx(dataset=original_dataset,
                             dimension=args.dimension,
                             init_random=True,
                             init_size=args.init) # type: ComplEx
    original_model.load_state_dict(torch.load(args.model_path))
    original_model.to('cuda')

    kelpie_dataset = KelpieDataset(dataset=original_dataset, entity_id=original_entity_id)


    ############ EXTRACT TEST FACTS AND TRAINING FACTS
    # extract all training facts and test facts involving the entity to explain
    # and replace the id of the entity to explain with the id of the fake kelpie entity
    original_test_samples = kelpie_dataset.original_test_samples
    kelpie_test_samples = kelpie_dataset.kelpie_test_samples
    kelpie_train_samples = kelpie_dataset.kelpie_train_samples


    def run_kelpie(train_samples):
        # use model_to_explain to initialize the Kelpie model
        kelpie_model = KelpieComplEx(model=original_model, dataset=kelpie_dataset, init_size=1e-3) # type: KelpieComplEx
        kelpie_model.to('cuda')

        ###########  BUILD THE OPTIMIZER AND RUN POST-TRAINING
        print("Running post-training on the Kelpie model...")
        optimizer = KelpieComplExOptimizer(model=kelpie_model,
                                           optimizer_name=args.optimizer,
                                           batch_size=args.batch_size,
                                           learning_rate=args.learning_rate,
                                           decay1=args.decay1,
                                           decay2=args.decay2,
                                           regularizer_name="N3",
                                           regularizer_weight=args.reg)
        optimizer.train(train_samples=train_samples, max_epochs=args.max_epochs)

        ###########  EXTRACT RESULTS

        kelpie_entity_id = kelpie_dataset.kelpie_entity_id
        kelpie_sample_tuple = (kelpie_entity_id, relation_id, tail_id) if args.perspective == "head" else (head_id, relation_id, kelpie_entity_id)
        kelpie_sample = numpy.array(kelpie_sample_tuple)

        ### Evaluation on original entity

        ### Evaluation on kelpie entity
        # results on kelpie fact
        scores, ranks, predictions = kelpie_model.predict_sample(sample=kelpie_sample, original_mode=False)
        kelpie_direct_score, kelpie_inverse_score = scores[0], scores[1]
        kelpie_head_rank, kelpie_tail_rank = ranks[0], ranks[1]
        kelpie_head_predictions, kelpie_tail_predictions = predictions[0], predictions[1]
        print("\nKelpie model on kelpie test fact: <%s, %s, %s>" % kelpie_sample_tuple)
        print("\tDirect fact score: %f; Inverse fact score: %f" % (kelpie_direct_score, kelpie_inverse_score))
        print("\tHead Rank: %f" % ranks[0])
        print("\tTail Rank: %f" % ranks[1])

        return kelpie_direct_score, kelpie_inverse_score, kelpie_head_rank, kelpie_tail_rank, kelpie_head_predictions, kelpie_tail_predictions



    outlines = []
    skipped_fact_2_performances = {}

    print("### CLONE POST-TRAINING")

    direct_score, inverse_score, head_rank, tail_rank, head_predictions, tail_predictions = run_kelpie(kelpie_train_samples)
    head_predictions = head_predictions[:head_rank]
    tail_predictions = tail_predictions[:tail_rank]
    outlines.append(";".join([head, relation, tail]) + "\n")
    outlines.append("\t".join(["NO FACTS SKIPPED",
                               str(direct_score), str(inverse_score), str(head_rank), str(tail_rank)]) + "\n")

    for i in range(len(kelpie_train_samples)):
        cur_kelpie_train_samples = numpy.delete(kelpie_train_samples, i, axis=0)
        skipped_sample = kelpie_train_samples[i]
        skipped_fact = (kelpie_dataset.entity_id_2_name[int(skipped_sample[0])],
                        kelpie_dataset.relation_id_2_name[int(skipped_sample[1])],
                        kelpie_dataset.entity_id_2_name[int(skipped_sample[2])])

        print("### ITER " + str(i) + ": SKIPPED FACT: " + ";".join(skipped_fact))
        print("\tKelpie Clone on kelpie test fact:")
        print("\tDirect fact score: %f; Inverse fact score: %f" % (direct_score, inverse_score))
        print("\tHead Rank: %f" % head_rank)
        print("\tTail Rank: %f" % tail_rank)

        cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, cur_head_predictions, cur_tail_predictions = run_kelpie(cur_kelpie_train_samples)
        cur_head_predictions = cur_head_predictions[:cur_head_rank]
        cur_tail_predictions = cur_tail_predictions[:cur_tail_rank]

        skipped_fact_2_performances[skipped_fact] = (cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, cur_head_predictions, cur_tail_predictions)

    most_influential_fact = None
    most_influential_overall_rbo = 1000
    most_influential_score_diff = 0
    for skipped_fact in skipped_fact_2_performances:
        cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, cur_head_predictions, cur_tail_predictions = skipped_fact_2_performances[skipped_fact]

        head_prediction_rbo = ranking_similarity.rank_biased_overlap(head_predictions, cur_head_predictions)
        tail_prediction_rbo = ranking_similarity.rank_biased_overlap(tail_predictions, cur_tail_predictions)

        overall_rbo = head_prediction_rbo + tail_prediction_rbo
        score_diff = direct_score - cur_direct_score + inverse_score - cur_inverse_score

        if overall_rbo < most_influential_overall_rbo:
            most_influential_overall_rbo = overall_rbo
            most_influential_fact = skipped_fact
        elif overall_rbo == most_influential_overall_rbo and score_diff > most_influential_score_diff:
            most_influential_overall_rbo = overall_rbo
            most_influential_fact = skipped_fact
            most_influential_score_diff = score_diff


    cur_direct_score, cur_inverse_score, cur_head_rank, cur_tail_rank, cur_head_predictions, cur_tail_predictions = skipped_fact_2_performances[most_influential_fact]

    head_prediction_rbo = ranking_similarity.rank_biased_overlap(head_predictions, cur_head_predictions)
    tail_prediction_rbo = ranking_similarity.rank_biased_overlap(tail_predictions, cur_tail_predictions)
    overall_rbo = head_prediction_rbo + tail_prediction_rbo

    translated_most_influential_fact = [most_influential_fact[0] if most_influential_fact[0] != "kelpie" else original_dataset.entity_id_2_name[original_entity_id],
                                        most_influential_fact[1],
                                        most_influential_fact[2] if most_influential_fact[2] != "kelpie" else original_dataset.entity_id_2_name[original_entity_id]]

    outlines.append("MOST INFLUENTIAL FACT:\t" + "\t".join([";".join(translated_most_influential_fact),
                                                     str(cur_direct_score),
                                                     str(cur_inverse_score),
                                                     str(cur_head_rank),
                                                     str(cur_tail_rank),
                                                     str(head_prediction_rbo),
                                                     str(tail_prediction_rbo)]) + "\n\n")
    with open("output_1.txt", "a") as outfile:
        outfile.writelines(outlines)

