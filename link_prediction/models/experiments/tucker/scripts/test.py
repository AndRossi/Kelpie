from collections import defaultdict

import numpy
import torch
import argparse

from dataset import Dataset, ALL_DATASET_NAMES
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.tucker import TuckER
from model import ENTITY_DIMENSION, RELATION_DIMENSION, INPUT_DROPOUT, HIDDEN_DROPOUT_1, HIDDEN_DROPOUT_2, \
    LEARNING_RATE, DECAY, LABEL_SMOOTHING

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        type=str,
                        choices=ALL_DATASET_NAMES,
                        help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10")

    parser.add_argument("--decay_rate",
                        type=float,
                        default=1.0,
                        help="Decay rate.")

    parser.add_argument("--entity_dimension",
                        type=int,
                        default=200,
                        help="Entity embedding dimensionality.")

    parser.add_argument("--relation_dimension",
                        type=int,
                        default=200,
                        help="Relation embedding dimensionality.")

    parser.add_argument("--valid",
                        type=int,
                        default=-1,
                        help="Validate after a cycle of x epochs")

    parser.add_argument("--input_dropout",
                        type=float,
                        default=0.3,
                        nargs="?",
                        help="Input layer dropout.")

    parser.add_argument("--hidden_dropout_1",
                        type=float,
                        default=0.4,
                        help="Dropout after the first hidden layer.")

    parser.add_argument("--hidden_dropout_2",
                        type=float,
                        default=0.5,
                        help="Dropout after the second hidden layer.")

    parser.add_argument("--label_smoothing",
                        type=float,
                        default=0.1,
                        help="Amount of label smoothing.")

    parser.add_argument('--model_path',
                        help="Path to the model to explain the predictions of",
                        required=True)

    args = parser.parse_args()
    #torch.backends.cudnn.deterministic = True
    #seed = 20
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #if torch.cuda.is_available:
    #    torch.cuda.manual_seed_all(seed)

    print("Loading %s dataset..." % args.dataset)
    dataset = Dataset(name=args.dataset, separator="\t", load=True)

    hyperparameters = {ENTITY_DIMENSION: args.entity_dimension,
                       RELATION_DIMENSION: args.relation_dimension,
                       INPUT_DROPOUT: args.input_dropout,
                       HIDDEN_DROPOUT_1: args.hidden_dropout_1,
                       HIDDEN_DROPOUT_2: args.hidden_dropout_2,
                       DECAY: args.decay_rate,
                       LABEL_SMOOTHING: args.label_smoothing}

    print("Loading model...")
    tucker = TuckER(dataset=dataset, hyperparameters=hyperparameters, init_random=True) # type: TuckER
    model_path = args.model_path if args.model_path is not None else "./models/TuckER_" + args.dataset + ".pt"
    tucker.load_state_dict(torch.load(model_path))
    tucker.eval()

    print("\nEvaluating model...")
    mrr, h1 = Evaluator(model=tucker).eval(samples=dataset.test_samples, write_output=True)
    print("\tTest Hits@1: %f" % h1)
    print("\tTest Mean Reciprocal Rank: %f" % mrr)

    testing_facts = [
        ('/m/09v3jyg', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/0f8l9c'),
        ('/m/0djb3vw', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/0jgd'),
        ('/m/09v71cj', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/015fr'),
        ('/m/0ds6bmk', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/05r4w'),
        ('/m/040b5k', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/06qd3'),
        ('/m/02qk3fk', '/film/film/release_date_s./film/film_regional_release_date/film_release_region', '/m/035qy'),

        ('/m/017z49', '/film/film/release_date_s./film/film_regional_release_date/film_regional_debut_venue',
         '/m/018cvf'),

        ('/m/05zpghd', '/film/film/film_festivals', '/m/04_m9gk'),
        ('/m/0cwy47', '/film/film/film_festivals', '/m/059_y8d'),
        ('/m/02wypbh', '/film/film/film_festivals', '/m/05f5rsr'),

        ('/m/01l3j', '/film/actor/film./film/performance/special_performance_type', '/m/01pb34'),

        ('/m/04mlmx', '/film/actor/film./film/performance/film', '/m/042fgh'),
        ('/m/02qx69', '/film/actor/film./film/performance/film', '/m/02stbw'),
        ('/m/05kwx2', '/film/actor/film./film/performance/film', '/m/031786'),

        ('/m/0gmgwnv', '/film/film/estimated_budget./measurement_unit/dated_money_value/currency', '/m/09nqf'),

        ('/m/053rxgm', '/film/film/executive_produced_by', '/m/0g_rs_'),
        ('/m/09z2b7', '/film/film/executive_produced_by', '/m/05hj_k'),

        ('/m/06znpjr', '/film/film/produced_by', '/m/05nn4k'),

        ('/m/062zm5h', '/film/film/story_by', '/m/046_v'),
        ('/m/0164qt', '/film/film/story_by', '/m/0fx02'),

        ('/m/02xs6_', '/film/film/edited_by', '/m/0bn3jg'),

        ('/m/09g7vfw', '/film/film/costume_design_by', '/m/03mfqm'),

        ('/m/02v8kmz', '/film/film/cinematography', '/m/03_fk9'),

        ('/m/0b2qtl', '/film/film/genre', '/m/060__y'),
        ('/m/0cnztc4', '/film/film/genre', '/m/03bxz7'),
        ('/m/0f7hw', '/film/film/genre', '/m/02l7c8'),
        ('/m/05mrf_p', '/film/film/genre', '/m/02n4kr'),
        ('/m/07kh6f3', '/film/film/genre', '/m/0lsxr'),
        ('/m/05qbbfb', '/film/film/genre', '/m/01hmnh'),
        ('/m/0jqn5', '/film/film/genre', '/m/03k9fj'),
        ('/m/043n0v_', '/film/film/genre', '/m/02p0szs'),

        ('/m/0kv9d3', '/film/film/language', '/m/064_8sq'),

        ('/m/0gjc4d3', '/film/film/film_format', '/m/017fx5'),
        ('/m/027rpym', '/film/film/film_format', '/m/01dc60'),
        ('/m/0hmm7', '/film/film/film_format', '/m/0cj16'),

        ('/m/0170z3', '/film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium',
         '/m/0735l'),

        ('/m/02c7k4', '/film/film/production_companies', '/m/01795t'),
        ('/m/0dtfn', '/film/film/production_companies', '/m/0kx4m'),
        ('/m/05z_kps', '/film/film/production_companies', '/m/05mgj0'),

        ('/m/027s39y', '/film/film/featured_film_locations', '/m/02_286'),
        ('/m/02vxq9m', '/film/film/featured_film_locations', '/m/04jpl'),

        ('/m/0ydpd', '/location/location/time_zones', '/m/02hcv8'),
        ('/m/0flbm', '/location/location/time_zones', '/m/02hczc'),
        ('/m/0chrx', '/location/location/time_zones', '/m/02fqwt'),
        ('/m/0r5lz', '/location/location/time_zones', '/m/02lcqs'),

        ('/m/05c74', '/location/country/form_of_government', '/m/01fpfn'),
        ('/m/04g61', '/location/country/form_of_government', '/m/01q20'),

        ('/m/05bcl', '/location/country/official_language', '/m/02h40lc'),

        ('/m/03b79', '/location/country/capital', '/m/0156q'),

        ('/m/02jx1', '/location/location/contains', '/m/04jpl'),
        ('/m/02qkt', '/location/location/contains', '/m/03spz'),

        ('/m/05kkh', '/location/location/partially_contains', '/m/0lm0n'),
        ('/m/050l8', '/location/location/partially_contains', '/m/04ykz'),

        ### PEOPLE ###

        ('/m/06sy4c', '/people/person/gender', '/m/05zppz'),
        ('/m/09jrf', '/people/person/gender', '/m/05zppz'),
        ('/m/043q6n_', '/people/person/gender', '/m/05zppz'),

        ('/m/0mm1q', '/people/person/profession', '/m/01d_h8'),
        ('/m/01wkmgb', '/people/person/profession', '/m/09jwl'),
        ('/m/03j90', '/people/person/profession', '/m/05z96'),
        ('/m/01trf3', '/people/person/profession', '/m/0np9r'),
        ('/m/0m68w', '/people/person/profession', '/m/02hrh1q'),
        ('/m/01w58n3', '/people/person/profession', '/m/016z4k'),
        ('/m/0grwj', '/people/person/profession', '/m/03gjzk'),
        ('/m/03s9v', '/people/person/profession', '/m/06q2q'),
        ('/m/0k4gf', '/people/person/profession', '/m/05vyk'),
        ('/m/0grmhb', '/people/person/profession', '/m/02krf9'),
        ('/m/03k48_', '/people/person/profession', '/m/018gz8'),
        ('/m/0gd5z', '/people/person/profession', '/m/0kyk'),
        ('/m/03975z', '/people/person/profession', '/m/01c72t'),
        ('/m/09jd9', '/people/person/profession', '/m/0cbd2'),
        ('/m/03_nq', '/people/person/profession', '/m/04gc2'),
        ('/m/0b6mgp_', '/people/person/profession', '/m/02tx6q'),
        ('/m/03s9v', '/people/person/profession', '/m/0h9c'),

        ('/m/063g7l', '/people/person/nationality', '/m/09c7w0'),
        ('/m/05yvfd', '/people/person/nationality', '/m/03rk0'),
        ('/m/015q43', '/people/person/nationality', '/m/07ssc'),
        ('/m/012j8z', '/people/person/nationality', '/m/0d060g'),

        ('/m/043q6n_', '/people/person/place_of_birth', '/m/030qb3t'),
        ('/m/025j1t', '/people/person/place_of_birth', '/m/02dtg'),
        ('/m/01vrkdt', '/people/person/place_of_birth', '/m/030qb3t'),
        ('/m/0c3ns', '/people/person/place_of_birth', '/m/06y57'),

        ('/m/0l9k1', '/people/deceased_person/place_of_death', '/m/0f2wj'),

        ('/m/01g42', '/people/deceased_person/place_of_burial', '/m/018mm4'),

        ('/m/03k545', '/people/person/places_lived./people/place_lived/location', '/m/04jpl'),
        ('/m/0d3k14', '/people/person/places_lived./people/place_lived/location', '/m/01cx_'),
        ('/m/01wbsdz', '/people/person/places_lived./people/place_lived/location', '/m/013yq'),

        ('/m/06mn7', '/people/person/religion', '/m/03_gx'),
        ('/m/03m2fg', '/people/person/religion', '/m/03j6c'),

        ('/m/0pz04', '/people/person/employment_history./business/employment_tenure/company', '/m/06rq1k'),

        ('/m/01rv7x', '/people/ethnicity/people', '/m/0dfjb8'),

        ('/m/0bt7ws', '/people/person/languages', '/m/02h40lc'),
        ('/m/0dfjb8', '/people/person/languages', '/m/09bnf'),

        ('/m/0222qb', '/people/ethnicity/languages_spoken', '/m/02bjrlw'),

        ### LANGUAGE ###

        ('/m/0t_2', '/language/human_language/countries_spoken_in', '/m/09c7w0'),
        ('/m/06b_j', '/language/human_language/countries_spoken_in', '/m/07t21'),

        ### TV ###
        ('/m/070ltt', '/tv/tv_program/country_of_origin', '/m/09c7w0'),
        ('/m/050kh5', '/tv/tv_program/country_of_origin', '/m/03rk0'),

        ('/m/0kfpm', '/tv/tv_program/genre', '/m/0c4xc'),
        ('/m/04glx0', '/tv/tv_program/genre', '/m/07s9rl0'),
        ('/m/01p4wv', '/tv/tv_program/genre', '/m/05p553'),
        ('/m/01cvtf', '/tv/tv_program/genre', '/m/0lsxr'),
        ('/m/03nymk', '/tv/tv_program/genre', '/m/0hcr'),

        ### MEDIA_COMMON ###
        ('/m/07c52', '/media_common/netflix_genre/titles', '/m/0l76z'),

        ### TIME ###

        ('/m/0fy6bh', '/time/event/instance_of_recurring_event', '/m/0g_w'),
        ('/m/0br1x_', '/time/event/instance_of_recurring_event', '/m/02jp2w'),
        ('/m/0cc8q3', '/time/event/instance_of_recurring_event', '/m/02jp2w'),

        ### BASE ###

        ('/m/04lc0h', '/base/biblioness/bibs_location/country', '/m/0ctw_b'),
        ('/m/027l4q', '/base/biblioness/bibs_location/country', '/m/09c7w0'),

        ('/m/0g5lhl7',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location',
         '/m/07ssc'),
        ('/m/087c7',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location',
         '/m/0d060g'),
        ('/m/0l8sx',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location',
         '/m/09c7w0'),
        ('/m/0p4wb',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location',
         '/m/09c7w0'),
        ('/m/049dk',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_location',
         '/m/09c7w0'),

        ('/m/0mgkg',
         '/base/schemastaging/organization_extra/phone_number./base/schemastaging/phone_sandbox/service_language',
         '/m/02h40lc'),

        ('/m/013yq', '/base/petbreeds/city_with_dogs/top_breeds./base/petbreeds/dog_city_relationship/dog_breed',
         '/m/0km5c'),
        ('/m/04f_d', '/base/petbreeds/city_with_dogs/top_breeds./base/petbreeds/dog_city_relationship/dog_breed',
         '/m/0km3f'),
        ('/m/02dtg', '/base/petbreeds/city_with_dogs/top_breeds./base/petbreeds/dog_city_relationship/dog_breed',
         '/m/01t032'),

        ('/m/018ygt', '/base/schemastaging/person_extra/net_worth./measurement_unit/dated_money_value/currency',
         '/m/09nqf'),

        (
        '/m/0b_6q5', '/base/marchmadness/ncaa_basketball_tournament/seeds./base/marchmadness/ncaa_tournament_seed/team',
        '/m/091tgz'),

        ('/m/02tvsn', '/base/culturalevent/event/entity_involved', '/m/0j5b8'),

        ### EDUCATION ###

        ('/m/01rtm4', '/education/educational_institution/school_type', '/m/01rs41'),
        ('/m/023zl', '/education/educational_institution/school_type', '/m/05jxkf'),
        ('/m/0kw4j', '/education/educational_institution/school_type', '/m/05pcjw'),
        ('/m/0k2h6', '/education/educational_institution/school_type', '/m/07tf8'),
        ('/m/0j_sncb', '/education/educational_institution/school_type', '/m/01_9fk'),

        ('/m/0m7yh', '/education/university/domestic_tuition./measurement_unit/dated_money_value/currency', '/m/02l6h'),
        (
        '/m/04ycjk', '/education/university/domestic_tuition./measurement_unit/dated_money_value/currency', '/m/09nqf'),

        ### MUSIC ###

        ('/m/0glt670', '/music/genre/artists', '/m/01v40wd'),
        ('/m/01wqlc', '/music/genre/artists', '/m/0h6sv'),
        ('/m/04f73rc', '/music/genre/artists', '/m/016lj_'),
        ('/m/02yv6b', '/music/genre/artists', '/m/0134tg'),
        ('/m/0ggx5q', '/music/genre/artists', '/m/01wj18h'),
        ('/m/0m0fw', '/music/genre/artists', '/m/07yg2'),

        ('/m/01hcvm', '/music/genre/parent_genre', '/m/05r6t'),
        ('/m/0kz10', '/music/genre/parent_genre', '/m/07gxw'),
        ('/m/0dls3', '/music/genre/parent_genre', '/m/016clz'),

        ('/m/012x4t', '/music/group_member/membership./music/group_membership/group', '/m/01v0sx2'),
        ('/m/01q7cb_', '/music/group_member/membership./music/group_membership/role', '/m/03bx0bm'),
        ('/m/01vng3b', '/music/group_member/membership./music/group_membership/role', '/m/02hnl'),
        ('/m/018gkb', '/music/group_member/membership./music/group_membership/group', '/m/047cx'),

        ('/m/0197tq', '/music/artist/track_contributions./music/track_contribution/role', '/m/0342h'),
        ('/m/0f6lx', '/music/artist/track_contributions./music/track_contribution/role', '/m/03qlv7'),
        ('/m/01m7pwq', '/music/artist/track_contributions./music/track_contribution/role', '/m/02sgy'),

        ### INFLUENCE ###

        ('/m/0b1zz', '/influence/influence_node/influenced_by', '/m/05xq9'),
        ('/m/014ps4', '/influence/influence_node/influenced_by', '/m/03j0d'),

        ### MILITARY ###

        ('/m/0gjw_', '/military/military_conflict/combatants./military/military_combatant_group/combatants',
         '/m/0d060g'),

        ### BUSINESS ###

        ('/m/01qszl', '/business/business_operation/industry', '/m/01mw1'),
        ('/m/0181hw', '/business/business_operation/industry', '/m/02jjt'),
        ('/m/019v67', '/business/business_operation/industry', '/m/02vxn'),
        ('/m/0p4wb', '/business/business_operation/industry', '/m/0vg8'),
        ('/m/01dfb6', '/business/business_operation/industry', '/m/02h400t'),

        ### ORGANIZATION ###

        ('/m/06c97', '/organization/organization_founder/organizations_founded', '/m/05f4p'),

        ('/m/0154j', '/organization/organization_member/member_of./organization/organization_membership/organization',
         '/m/04k4l'),
        ('/m/01znc_', '/organization/organization_member/member_of./organization/organization_membership/organization',
         '/m/07t65'),
        ('/m/088q4', '/organization/organization_member/member_of./organization/organization_membership/organization',
         '/m/0gkjy'),
        ('/m/06v36', '/organization/organization_member/member_of./organization/organization_membership/organization',
         '/m/0j7v_'),
        ('/m/0hmyfsv', '/organization/organization_member/member_of./organization/organization_membership/organization',
         '/m/03mbdx_'),

        ('/m/0kctd', '/organization/organization/headquarters./location/mailing_address/citytown', '/m/02_286'),
        ('/m/026036', '/organization/organization/headquarters./location/mailing_address/citytown', '/m/02_286'),

        ('/m/01q0kg', '/organization/organization/headquarters./location/mailing_address/state_province_region',
         '/m/01n7q'),
        ('/m/0206k5', '/organization/organization/headquarters./location/mailing_address/state_province_region',
         '/m/03v0t'),

        ### GOVERNMENT ###

        ('/m/0226cw',
         '/government/politician/government_positions_held./government/government_position_held/legislative_sessions',
         '/m/07p__7'),
        ('/m/03txms',
         '/government/politician/government_positions_held./government/government_position_held/legislative_sessions',
         '/m/070m6c'),

        ('/m/03txms',
         '/government/politician/government_positions_held./government/government_position_held/basic_title',
         '/m/01gkgk'),

        ### SPORTS

        ('/m/06x76', '/sports/sports_team/sport', '/m/0jm_'),

        ('/m/05xvj', '/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft',
         '/m/047dpm0'),
        ('/m/0jm3v', '/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft',
         '/m/025tn92'),
        (
        '/m/0x2p', '/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft', '/m/04f4z1k'),
        (
        '/m/01xvb', '/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft', '/m/09l0x9'),
        ('/m/01y3c', '/sports/professional_sports_team/draft_picks./sports/sports_league_draft_pick/draft',
         '/m/02qw1zx'),

        ('/m/0jm_', '/sports/sport/pro_athletes./sports/pro_sports_played/athlete', '/m/0443c'),
        ('/m/02vx4', '/sports/sport/pro_athletes./sports/pro_sports_played/athlete', '/m/02y9ln'),

        ('/m/04l5b4', '/ice_hockey/hockey_team/current_roster./sports/sports_team_roster/position', '/m/02qvzf'),

        ### AWARD ###
        ('/m/01v40wd', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/01c9dd'),
        ('/m/06mn7', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/019f4v'),
        ('/m/013cr', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02x8n1n'),
        ('/m/02tq2r', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/03rbj2'),
        ('/m/01r9fv', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02f6ym'),
        ('/m/0b05xm', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0fbtbt'),
        ('/m/06mn7', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0gq9h'),
        ('/m/0bz5v2', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/047byns'),
        ('/m/02pby8', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/04ljl_l'),
        ('/m/0lbj1', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02x17c2'),
        ('/m/0738y5', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0b6k___'),
        ('/m/0pz04', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/05zvj3m'),
        ('/m/0170vn', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0f4x7'),
        ('/m/01y64_', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02ppm4q'),
        ('/m/036hf4', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/05pcn59'),
        ('/m/05drq5', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/04dn09n'),
        ('/m/016sp_', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/01c9f2'),
        ('/m/01w0yrc', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/09qv3c'),
        ('/m/0chw_', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/02y_rq5'),
        ('/m/06bng', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/040vk98'),
        ('/m/016m5c', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/01d38t'),
        ('/m/02vyh', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0gq9h'),
        ('/m/09bxq9', '/award/award_nominee/award_nominations./award/award_nomination/award', '/m/0gr0m'),

        ('/m/05zl0', '/award/ranked_item/appears_in_ranked_lists./award/ranking/list', '/m/09g7thr'),
        ('/m/03mnk', '/award/ranked_item/appears_in_ranked_lists./award/ranking/list', '/m/01pd60'),
        ('/m/0k8z', '/award/ranked_item/appears_in_ranked_lists./award/ranking/list', '/m/04k4rt'),
        ('/m/0grwj', '/award/ranked_item/appears_in_ranked_lists./award/ranking/list', '/m/026cl_m'),

        ('/m/01tgwv', '/award/award_category/disciplines_or_subjects', '/m/06n90'),
        ('/m/0223xd', '/award/award_category/disciplines_or_subjects', '/m/04g51'),

        ('/m/02gx2k', '/award/award_category/category_of', '/m/0c4ys'),

        ('/m/0167bx', '/award/award_winning_work/awards_won./award/award_honor/award', '/m/0dt49'),
        ('/m/05zr0xl', '/award/award_winning_work/awards_won./award/award_honor/award', '/m/0cjyzs'),

        ('/m/0bzkgg', '/award/award_ceremony/awards_presented./award/award_honor/honored_for', '/m/01jc6q'),

        ('/m/02yw5r', '/award/award_ceremony/awards_presented./award/award_honor/award_winner', '/m/0164w8'),

        ('/m/0g69lg', '/award/award_nominee/award_nominations./award/award_nomination/nominated_for', '/m/015ppk'),

        ('/m/0gkvb7', '/award/award_category/nominees./award/award_nomination/nominated_for', '/m/0ph24')
    ]

    testing_samples = []
    for testing_fact in testing_facts:
        testing_samples.append(dataset.fact_to_sample(testing_fact))

    scores, ranks, predictions = tucker.predict_samples(numpy.array(testing_samples))

    for i in range(len(testing_facts)):
        print(";".join(testing_facts[i]) + ";" + str(ranks[i][1]))
