<meta name="robots" content="noindex">

# Kelpie

<p align="center">
<img width="50%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/124291133-87fa5d80-db54-11eb-9db9-62ca9bd3fe6f.png">
</p>

Kelpie is a post-hoc local explainability tool specifically tailored for embedding-based models that perform Link Prediction (LP) on Knowledge Graphs (KGs).

Kelpie provides a simple and effective interface to identify the most relevant facts to any prediction; intuitively, when explaining a _tail_ prediction <_h_, _r_, _t_>, Kelpie identifies the smallest set of training facts mentioning _h_ that are instrumental to that prediction; analogously, a _head_ prediction <_h_, _r_, _t_> would be explained with a combination of the training facts mentioning _t_.


## Kelpie Structure 

Kelpie is structured in a simple architecture based on the interaction of three modules. 
When explaining a _tail_ prediction <_h_, _r_, _t_>:

* a **Pre-Filter** narrows down the space of the candidate explanations (i.e., combinations of training facts mentioning _h_) by identifying and keeping only the most promising _h_ facts;
* an **Explanation Builder** governs the search in the resulting space of the candidate explanations, i.e., the combinations of promising facts obtained from the Pre-Filtering step;
* a **Relevance Engine** is used to estimate the *relevance* of any candidate explanation that the Explanation Builder wants to verify.

The modules would work analogously to explain a _head_ prediction. The only module that requires awareness of how the original Link Prediction model is trained and implemented is the Relevance Engine. While theoreticaly specific connectors could be developed to to adapt to pre-existing models, in our research we have found it easier to make the Relevance Engine directly interact with Kelpie-compatible implementations of the models.

<p align="center">
<img width="60%" alt="kelpie_structure" src="https://user-images.githubusercontent.com/6909990/140399831-0c368ac2-7cf4-48dc-bb73-eaf00f7fde52.png">
</p>


## Kelpie Explanations 

Under the broad definition described above, Kelpie supports two explanation scenarios: _necessary_ explanations and _sufficient_ explanations.

* Given a _tail_ prediction <_h_, _r_, _t_>, a **necessary explanation** is the smallest set of training facts featuring _h_ such that, if we remove those facts from the training set and re-train the model from scratch, the model will not be able to identify _t_ as the top-ranking tail. In other words, a _necessary_ explanation is the smallest set of _h_ facts that have made possible for the model to pick the correct tail. An analogous definition can be derived for head predictions. 

* Given a _tail_ prediction <_h_, _r_, _t_>, a **sufficient explanation** is the smallest set of training facts featuring _h_ such that, if we add those facts to random entities _c_ for which the model does not predict <_c_, _r_, _t_>, and we retrain the model from scratch, the model will start predicting <_c_, _r_, _t_> too. In other words, a _sufficient_ explanation is the smallest set of _h_ facts make it possible to extend the prediction to any other entity in the graph. An analogous definition can be derived for head predictions.


## Environment and Prerequisites

We have run all our experiments on an Ubuntu 18.04.5 environment using used Python 3.7.7, CUDA Version: 11.2 and Driver Version: 460.73.01.
Kelpie requires the following libraries: 
- PyTorch (we used version 1.7.1);
- numpy;
- tqdm;


## Models and Datasets

The formulation of Kelpie supports all Link Prediction models based on embeddings. For the sake of simplicity in our implementation we focus on models that train on individual facts, as these are the vast majority in literature. Nonetheless, our implementation can be extended to identify fact-based explanations for other models too, e.g., models that leverage contextual information such as paths, types, or temporal data.

We run our experiments on three models that rely on very different architectures: `ComplEx`, `ConvE` and `TransE`. 
We provide implementations for these models in this repository.
We explaining their predictions on the 5 best-established datasets in literature, i.e., `FB15k`, `WN18`, `FB15k-237`, `WN18RR` and `YAGO3-10`.
The training, validation and test sets of such datasets are distributed in this repository in the `data` folder.


## Training and Testing Our Models

For the sake of reproducibility, we make available through FigShare [the `.pt` model files](https://figshare.com/s/ede27f3440fe742de60b) resulting from training each system on each dataset. To run any of the experiments of our paper, the `.pt` files of all the trained models should bw downloaded and stored in a new folder `Kelpie/stored_models`.

For our models and datasets we use following hyperparameters, which we have found to lead to the best performances.

<p align="center">
<img width="90%" alt="hyperparams" src="https://user-images.githubusercontent.com/6909990/124291956-66e63c80-db55-11eb-9aa6-9892ee24afc2.png">
</p>

Note that: 
* *D* is the embedding dimension (in the models we use, entity and relation embeddings always have same dimension);
* *LR* is the learning rate;
* *B* is the batch size;
* *Ep* is the number of epochs;
* *γ* is the margin in Pairwise Ranking Loss;
* *N* is the number of negative samples generated for each positive training sample;
* *Opt* is the used Optimizer (either `SGD`, or `Adagrad`, or `Adam`);
* *Reg* is the regularizer weight;
* *Decay* is the applied learning rate Decay;
* *ω* is the size of the convolutional kernels;
* *Drop* is the training dropout rate:
    * *in* is the input dropout;
    * *h* is the dropout applied after a hidden layer;
    * *feat* is the feature dropout;

After the models have been trained, their evaluation yields the following metrics:

<p align="center">
<img width="60%" alt="model_results" src="https://user-images.githubusercontent.com/6909990/135614004-db1cff3a-68db-447d-bb9c-3c7f05426957.png">
</p>

The training and evaluation processes can be launched with the commands reported in our [training and testing section](#training-and-testing-models-1).


## End-to-end experiments

We showcase the effectiveness of Kelpie by explaining, for each model and dataset, the tail predictions of a set of 100 correctly predicted test facts both in *necessary* and in *sufficient* scenario. 
The `.csv` files containing the facts we explain for each model and dataet can be found in the `input_facts` folder.
We report result files for each model and dataset, both extracted by Kelpie and by the baseline approaches, both in the _necessary_ and for the _sufficient_ scenario, in the `results.zip` archive in our repository.

Across the necessary and sufficient scenarios for the same model and dataset we usually employ the same set of 100 correctly predicted test facts. 
The only exception in this regard is in the `ConvE` explanations for `FB15k` and `FB15k-237` predictions: in these cases, a few <h, r, t> predictions used in necessary explanations could not be explained sufficiently because, due to strong dataset biases, *all entities in the dataset would be predicted correctly if used instead of h*. This made it impossible to extract a set of `c` entities to convert, because any entity would appear already converted without applying any sufficient explanation. We thus replaced these predictions for the sufficient scenario, obtaining created two different version `_nec` and `_suff` for the input facts file.

Kelpie relies on a *Post-Training technique* to generate mimics and compute the relevance of potential explanations.
Across all models and datasets, we always use, for the Post-Training, the same combination of hyperparameters used in the original training.
The only exception is `TransE`, where the batch size *B* is particularly large (2048, which usually exceeds by far the number of training facts featuring an entity).
This affects the Post-Training process because in any Post-Training epoch the entity would only benefit from one optimization step in each epoch.
We easily balance this by increasing the learning rate *LR* to 0.01 in TransE Post-Trainings.
In order to replicate our experiments, the `.pt` model files should be located in a `stored_models` folder inside Kelpie.

Kelpie experiments are based on the execution of two separate scripts:
* an `explain.py` script that: 
   * extracts explanations and saves the top 10 relevant ones for each input prediction in an `output.txt` file;
   * logs the result each tested explanation into separate files `output_details_1.csv`, `output_details_2.csv`, `output_details_3.csv` and `output_details_4.csv` (depending on the length of the tested explanation);
* a `verify_explanations.py` script that 
   * retrains the model from scratch after applying those explanations (i.e., removing their facts from the training set in the necessary scenario, or injecting them to the entities to convert in the sufficient scenario);
   * saves the outcomes of the corresponding predictions in an `output_end_to_end.csv` file;

The `explain.py` script also accepts an optional `--baseline` parameter whose acceptable values are `data_poisoning` or `criage`; using this parameter allows to estract results for our baselines instead of Kelpie.

We report here our end-to-end results for _necessary_ explanations. We measure the effectiveness of _necessary_ explanations by measuring how removing the explanation facts (and re-training the model) worsens the H@1 and MRR metrics of the predictions to explain, compared to the original model: the greater the decrease, the more effective the explanation (i.e., the "more necessary" the explanation facts actually were). Therefore, in this scenario, the more negative the ΔH@1 and ΔMRR values, the better the explanation effectiveness.

<p align="center">
<img width="60%" alt="end to end necessary experiment" src="https://user-images.githubusercontent.com/6909990/135614246-b68adc39-c771-404d-bc74-4ca532f8258e.png">
</p>

We report here our end-to-end results for _sufficient_ explanations. We add the explanation facts to 10 random entities and verify if, after re-training the model, they display the same predicted entity as in the prediction to explain, i.e., if they have been _converted_. In practice, we measure the explanation effectiveness by checking the increase in H@1 and MRR of the predictions to convert, (i.e., the facts that we hope the system now predicts) after adding the explanation facts and retraining, compared to their metrics in the original model; the greater the increase, the more effective the explanation (i.e., the "more sufficient" the explanation facts actually were to convert those entities). Therefore, in this scenario, the more positive the ΔH@1 and ΔMRR values, the better the explanation effectiveness.

<p align="center">
<img width="60%" alt="end to end sufficient experiment" src="https://user-images.githubusercontent.com/6909990/135614254-172bc8a1-8f58-4c6f-a84d-8c4f4e50bbde.png">
</p>

Our experiments on each model and dataset can be replicated with the commands reported in our [section on extracting and verifying explanations](#training-and-testing-models-1).

### Extracted Explanation Lengths
We report in the following charts the lengths of the explanations extracted in our end-to-end experiments.
More specifically, we report their distribution for each model and dataset, both in the necessary and in the sufficient scenario. In our experiments, we limit ourselves to explanations with maximum length 4.

Distribution of explanation lengths for the ComplEx model:
<p align="center">
<img width="60%" alt="complex explanation lengths" src="https://user-images.githubusercontent.com/6909990/149195991-424443fd-fcb0-460f-a1c8-f1f2a46a60d2.png">
</p>

Distribution of explanation lengths for the ConvE model:
<p align="center">
<img width="60%" alt="conve explanation lengths" src="https://user-images.githubusercontent.com/6909990/149196052-9b7f255c-8e03-4f8c-9758-324e68715d30.png">
</p>

Distribution of explanation lengths for the TransE model:
<p align="center">
<img width="60%" alt="transe explanation lengths" src="https://user-images.githubusercontent.com/6909990/149196089-947abc70-ab82-4598-90ee-f74ea2f7bf15.png">
</p>

We observe that, under the same model and dataset, necessary explanations tend to always be longer than sufficient ones. 
This is intuitively reasonable, because while necessary explanations need to encompass _all_ the pieces of evidence that allow a prediction, whereas sufficient explanation can just focus on the few "most decisive" ones. 
Let us consider, for example, the tail prediction <_Barack_Obama_, _nationality_, _USA_>. 
A **necessary** explanation would probably feature multiple _Barack_Obama_ facts, e.g., <_Barack_Obama_, _president_of_, _USA_> and <_Barack_Obama_, _part_of_, _109𝑡ℎ_US_Congress_>.
On the contrary, a **sufficient** explanation in this case is a set of fact that, if added to any non-American entity _c_, converts it into having _nationality_ _USA_: for this purpose, it is probably enough to just add to _c_ the single fact <_c_, _president_of_, _USA_>.


### Experiment Repetitions

In order to increase the confidence and assess the reliability of the observations from our end-to-end results, we repeat part of our experiments 10 times, using each time a different sample of 100 tail predictions to explain. 
Due to the time-consuming process of retraining the model from scratch after each extraction is over (which is needed to measure the effectiveness of the extracted explanations) repeating 10 times our _entire_ set of end-to-end experiments would take several months. 
For the time being we have just repeated the ComplEx experiments in the necessary scenario; this corresponds to running 10 times the explanation extraction of Kelpie and of our baselines K1, Data Poisoning and Criage on the 5 datasets FB15k, FB15k-237, WN18, WN18RR and YAGO3-10. Altogether, this amounts to 4x5x10 = 200 explanation extractions and model retrainings. In each extraction 100 tail predictions are explained, for a total of 20000 extracted explanations.
We report in the following table, for each method and dataset, the average and the standard deviation of the corresponding ΔH@1 and ΔMRR values:

<p align="center">
<img width="60%" alt="end to end repeat experiment" src="https://user-images.githubusercontent.com/6909990/137894457-3698d2ad-82dd-4558-8966-81fe078910a6.png">
</p>

We report in **bold** the best average ΔH@1 and ΔMRR values in each dataset.
The average ΔH@1 and ΔMRR values obtained across these 10 repeats are similar to those obtained in the original end-to-end experiment: a bit worse (less negative) for FB15k, FB15k-237 and YAGO3-10, a bit better (more negative) for WN18, and almost identical in WN18RR. When such variations occur, they equally invest the effectiveness of both Kelpie and the baselines: as a consequence the gap in effectiveness between Kelpie and its baselines remains almost identical across all datasets, with Kelpie always achieving the best effectiveness both in terms of ΔH@1 and ΔMRR. 
All in all, this confirms our observations from the original experiment. 

We share the output files with the results of our experiment repetitions in this repository, as part of the compressed archive `additional_experiments.zip`.

## Additional Experiments

We discuss in this section additional experiments that we could not include in our paper due to space constraints. 
We share their output files in this repository in the compressed archive `additional_experiments.zip`.

### Explanation Builder: Acceptance Threshold (necessary scenario) 

We report here our study how varying the values of the acceptance threshold ξ<sub>n0</sub> affects the results of Kelpie necessary explanations.

For any necessary candidate explanation X, the relevance ξ<sub>nX<sub>n</sub></sub> is the expected rank worsening associated to X<sub>n</sub>: therefore, makes sense to set the acceptance threshold ξ<sub>n0</sub> at least to 1.
We observe that using just slightly larger ξ<sub>n0</sub> values grants a larger margin of error in the rank worsening expectation, resulting in overall more effective explanations; on the other hand, this can increase research times. All in all, ξ<sub>n0</sub> can be seen as a parameter to tweak in order to find the best trade-off between the certainty to worsen the rank of the prediction to explain and the computation time.

For each model and dataset we report in the following table how the effectiveness of the extracted necessary explanations varies when using three different ξ<sub>n0</sub> values: 1 (the smallest sensible choice), 5, and 10. We follow the same pipeline as in our end-to-end experiments: we extract the explanations, remove their facts from the training set, and retrain the model to measure the worsening in the H@1 and MRR of the predictions to explain. 

<p align="center">
<img width="60%" alt="ξ_n0 variation experiment" src="https://user-images.githubusercontent.com/6909990/138895918-c90bdcc2-4f7f-487c-b800-154ad61e97b2.png">
</p>

Unsurprisingly, the most effective results are often associated to ξ<sub>n0</sub>=10. Nonetheless, while switching from ξ<sub>n0</sub>=1 to ξ<sub>n0</sub>=5 leads to a significant improvement, ξ<sub>n0</sub>=5 and ξ<sub>n0</sub>=10 results are usually similar.
In other words, once the room for some margin of error is provided, the rank worsening usually reaches a plateau, and further increasing ξ<sub>n0</sub> does not heavily affect results anymore.
This motivates our choice to use ξ<sub>n0</sub>=5 in our end-to-end experiments, as we think it provides the best trade-off overall.


### Pre-Filtering: _k_ value
The Kelpie Pre-Filter module is used at the beginning of the explanation extraction to identify the most promising facts with respect to the prediction to explain. Its purpose is to narrow down the space of candidate explanations to combinations of the top _k_ most promising facts, thus making the research more feasible.
In all the end-to-end experiments we use _k_ = 20; we show here the effect of varying the value of _k_ on the explanations for the ComplEx model predictions:

<p align="center">
<img width="60%" alt="kelpie_logo" src="https://user-images.githubusercontent.com/6909990/135614960-e146b76a-fd99-44fa-a4a2-7a0f2f2efe62.png">
</p>

Across all datasets, varying _k_ does not cause huge variations. This suggests that the topology-based policy used by the Pre-Filter does indeed identify the most promising facts: in other words, if the pre-filter is good at fiding and placing the most promising facts at the top of its ranking, it does not matter if we analyze the top 10, 20 or 30 facts in the ranking: the facts that we need will still make the cut.

In WN18, WN18RR and YAGO3-10 the effect of varying _k_ seems rather negligible (as a matter of fact, they are so small they may be more tied to the inherent randomness of the embedding training process than to the variation of _k_). This can be explained by considering that in these datasets, entities are usually featured in just a feew training facts: in average, 6.9 in WN18, 4.3 IN WN18RR and 17.5 in YAGO3-10.
In FB15k and FB15k-237, on the contrary, entities tend to have far more mentions in training (in average, 64.5 in FB15k and 37.4 in FB15k-237): this makes the choice of the value of _k_ more relevant, so and varying it leads to slightly more visible consequences. In this cases, as a general rule, greater values of _k_ lead to better or equal explanations, at the cost of extending the research space. 

We do not witness any cases in which increasing _k_ to more than 20 leads to astounding improvements in the explanation relevance: this confirms that 20 is indeed a fine trade-off for the Pre-Filter value. 

### Topology-based vs Type-based Pre-Filtering
The Pre-Filtering module used in our end-to-end experiments identifies the most promising training facts with a topology-based approach.
Recent works have highlighted that leveraging the types of entities can be beneficial in other tasks that use KG embeddings, such as fact-checking. Therefore, we design a type-based Pre-Filtering approach and compare the effectiveness of the resulting explanations with the effectiveness of those obtained with the usual topology-based method.

In the best-established datasets for Link Prediction, i.e., FB15k, FB15k-237, WN18, WN18RR and YAGO3-10 the types of entities are not reported explicitly, therefore a type-based Pre-Filtering approach can not be applied directly.
To get around this issue, we observe that generally the type of an entity closely affects the relations that the entity is involved with.
For example, a person entity will probably be mentioned in facts with relations like "_born_in_", "_lives_in_", or "_has_profession_"; on the contrary a place entity will generally be featured in facts with relations like "_located_in_" or "_contains_".
For each entity _e_ we thus build a _relation frequency vector_ that contains the numbers of times each relation is featured in a fact mentioning _e_. More specifically, in the vector built for any entity _e_, for each relation _r_ we store separately the frequency of _r_ in facts where _e_ is head and the frequency of _r_ in facts where _e_ is tail. In this way, we obtain a representation of the use of relations across the outbound and inbound edges adjacent to _e_.
For any entity _e_, we can then find the most entities with a similar type by just comparing the vector of _e_ with the vector of any other entity with cosine-similarity.

We use this approach to build a type-based Pre-Filter module that, explaining any tail prediction <_h_, _r_, _t_>, computes the promisingness of any fact featuring _h_ <_h_, _s_, _e_> or <_e_, _s_, _h_> by the cosine-similarity between _e_ and _t_. In simple terms, the more a fact featuring _h_ is linked to an entity similar to _t_, the more promising it is to explain the tail prediction <_h_, _r_, _t_>. An analogous formulation can be used to explain head predictions.

We report in the following table the effectiveness of the explanations obtained using the topology-based Pre-Filter and the type-based Pre-Filter:

<p align="center">
<img width="60%" alt="kelpie_prefilters_effectiveness" src="https://user-images.githubusercontent.com/6909990/136614185-8539b834-56a5-4c6b-91f8-6b745fce0284.png">
</p>

The two Pre-Filters tend to produce very similar results: none of the two is evidently superior to the other.The reason for such similar results is that both Pre-Filters tend to consistently place the "best" facts (i.e, the ones that are actually most relevant to explain the prediction) within the top _k_ promising ones. Therefore, in both cases the Relevance Engine, with its post-training methodology, will identify the same relevant facts among the extracted _k_ ones, and the framework will ultimately yield the same (or very similar) explanations. In this analysis we have used _k_=20, as in our end-to-end experiments.

### Explanation Builder: Comparison with Shapley Values and KernelSHAP
Recently, explainability approaches based on Shapley Values have gained traction in XAI due to their theoretical backing derived from Game Theory. 
Shapley Values can be used to convey the saliency of combinations of input features; however, computing the exact Shapley Values for the combinations of features of an input sample would require to perturbate all of such combinations one by one, and to verify each time the effect on the prediction to explain. 
This is clearly unfeasible in most scenarios. The authors of "A Unified Approach to Interpreting Model Predictions" (NIPS 2017) have proposed a SHAP framework which provides a number of ways to *approximate* Shapley Values instead of computing them precisely; among the novel approaches they introduce, KernelSHAP is the only truly model-agnostic one.

Unfortunately neither exact Shapley Values nor KernelSHAP can be used directly on Link Prediction models. Like any saliency-based approach, they formulate explanations in terms of which features of the input sample have been most relevant to the prediction to explain; in the case of Link Prediction, the input of any prediction is a triple of embeddings, so the features are the components of such embeddings. Since embeddings are just numeric vectors they are not human-interpretable, and thus their most relevant components would not be informative from a human point of view.

Kelpie, however, overcomes this issue by relying on its Relevance Engine module, which uses post-training to verify the effects of adding or removing training facts from any entity: this allows us to inject perturbations in the training facts of the head or tail entity of the prediction to explain, as if those facts were our interpretable input features. As a consequence, while saliency-based frameworks are not useful in Link Prediction by themselves, they can indeed be *combined* with the Relevance Engine to take advantage of its post-training method: this amounts to using such such frameworks in replacement of our Explanation Builder to conduct the search in the space of candidate explanations. 

We run experiments on 10 TransE predictions on the FB15k dataset, and verify the cost of various exploration approaches in terms of the number of visits in the space of candidate explanations that they perform before termination. This number corresponds to the number of post-trainings they request to the Relevance Engine. We compare the following approaches: 
* exact Shapley Values computation;
* approximated Shapley Values computation with the KernelSHAP approach;
* extraction of the best explanation with our Explanation Builder;

In all the three approaches we perform Pre-Filtering first with using *k*=20 so the number of training facts to analyze and combine is 20.
We report in the following chart our results for each of the predictions to explain (Y axis is in logscale).
<p align="center">
<img width="90%" alt="kelpie_shap_comparison" src="https://user-images.githubusercontent.com/6909990/139960079-e6257be7-a294-4d82-bfa0-2e57271c8371.png">
</p>

As already mentioned, the exact computation of Shapley Values analyzes all combinations of our input features; since in each prediction our Pre-Filter always keeps the 20 most promising facts only, the number of combinations visited by this approach is always the same, in the order of 2^20.
Despite not visiting all those combinations, KernelSHAP in our experiments is shown to still require a very large number of visits.
More specifically, it performs between 305,706 and 596,037 visits in the space of candidate explanations, with an average of 490,146.6.
Finally, our Explanation Builder appears much more efficient, performing between 20 and 170 visits with an average of 70.8.

All in all, our Explanation Builder appears remarkably more efficient than KernelSHAP. On the one hand, our Explanation Builder largely benefits from our preliminary relevance heuristics, which are tailored specifically for the Link Prediction scenario. Our heuristics allow us to start the search in the space of candidate explanations from the combination of facts that will most probably produce the best explanations; this, in turn, enables us to enact early termination policies.
On the other hand KernelSHAP, like any general-purpose framework, cannot make any kind of assumptions onthe composition of the search space. We also scknowledge that recent works have raised concerns on the tractability of SHAP in specific domains, e.g., the work by Van den Broeck _et al._ "On the Tractability of SHAP Explanations" (AAAI 2021).


## Adding support for new models 

The Relevance Engine is the only component that requires transparent access to the original model.
Accessing the original model is required to leverage the pre-existing embeddings, the scoring function, and the training methodology of the underlying embedding mechanism: this allows us to perform the _Post-Training_ process and create _mimics_ of pre-existing entities.

We have developed two main interfaces that must be implemented whenever one wants to add Kelpie support to a new model:
- [Model and its sub-interface KelpieModel](https://github.com/AndRossi/Kelpie/blob/master/link_prediction/models/model.py)
- [Optimizer](https://github.com/AndRossi/Kelpie/blob/master/link_prediction/optimization/optimizer.py)

### Model interface

The `Model` interface defines the general behavior expected from any Link Prediction model; Kelpie can explain the predictions of Link Prediction models that extend this interface.
`Model` extends, in turn, the PyTorch `nn.Module` interface, and it defines very general methods that any Link Prediction model should expose.
Therefore, it is usually very easy to adapt the code of any pre-existing models to extend `Model`.

Any instance of `Model` subclass is expected to expose the following instance variables:
* `dataset`: a reference to the Dataset under analysis;
* `num_entities`: the number of entities in the dataset;
* `num_relations`: the number of relations in the dataset;
* `dimension`: the embedding size (if equal for entity and relation embeddings)
* `entity_dimension` and `relation_dimension`: the entity and relation embedding sizes (if different between entity and relation embeddings)
* `entity_embeddings`: a `torch.nn.Parameter` containing the entity embeddings of this `Model`;
* `relation_embeddings`: a `torch.nn.Parameter` containing the relation embeddings of this `Model`.

Any `Model` subclass should provide implementations for the following methods:
* `score`: given a collection of samples passed as a numpy array, it computes their plausibility scores;
* `all_scores`: given a collection of samples passed as a numpy array, it computes for each of them the plausibility scores for all possible tail entities;
* `forward`: given a collection of samples passed as a numpy array, it performs forward propagation (this method is usually very similar to `all_scores`, but it may also return additional objects, e.g., elements required to compute regularization values);
* `predict_samples`: given a collection of samples passed as a numpy array, it performs head and tail prediction on them 
* `predict_sample`: given one sample passed as a numpy array, it performs head and tail prediction on it;
* `is_minimizer`: it returns `True` if in this `Model` a high score corresponds to low fact plausibility, or `False` otherwise;
* `kelpie_model_class`: it return the class of the `KelpieModel` implementation corresponding to this `Model`.

The `KelpieModel` interface, in turn, extends the `Model` interface, and defines the behaviour of post-trainable version of a `Model`.
Any `KelpieModel` implementation refers to a more general Model implementation: e.g., the `ComplEx` class (which is a `Model` implementation) has a `KelpieComplEx` subclass (that extends both `ComplEx` and `KelpieModel`).
Any Model implementation should know the corresponding `KelpieModel` class, and return it in the above mentioned kelpie_model_class method.
Any KelpieModel instance subclass is expected to expose the following instance variables:
* _original_entity_id_ = the internal id of the original entity that this `KelpieModel` has been created to generate a mimic for;
* _kelpie_entity_id_ = the internal id of the created mimic;
* _kelpie_entity_embedding_: a reference to the embedding of the mimic;

The only methods that `KelpieModel` classes should implement are overriding versions of `predict_samples` and `predict_sample`, that in `KelpieModel` classes also require the `original_mode` flag: if set to `True`, the `KelpieModel` should perform the prediction using the embedding of the original entity for the samples that mention the id of the original entity; if set to `False`, the `KelpieModel` should use the mimic embedding instead.

### Optimizer interface
An `Optimizer` implements a specific training methodology; therefore, the main method in our `Optimizer` interface is just `train`.

In our project, we have implemented a separate `Optimizer` class for each Loss function required by our models: therefore, we currently feature a `BCEOptimizer` class, a `MultiClassNLLOptimizer` class, and a `PairwiseRankingOptimizer` class.
Similarly to the relation between `Models` and `KelpieModels`, we have also created for each of these Optimizers a separate sub-class that handles post-training instead of traditional training. We have called such sub-classes `KelpieBCEOptimizer`, `KelpieMultiClassNLLOptimizer` and `KelpiePairwiseRankingOptimizer` respectively.

Since these subclasses have identical signature to their respective Optimizers, unlike with `Models` and `KelpieModels` we have not created a separate `KelpieOptimizer` interface. Given an `Optimizer`, implementing its `Kelpie-` subclass is immediate: it usually enough to provide an overriding version of the superclass `train` method, making sure to also update the embedding of the mimic after each epoch calling the `update_embeddings` method of the KelpieModel that is being post-trained.

## Launching Kelpie

### Training and testing models

The training and evaluation processes can be launched with the following commands:

* **ComplEx**

  * **FB15k**

    ```python
    python3 scripts/complex/train.py --dataset FB15k --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3
    ```
    ```python
    python3 scripts/complex/test.py --dataset FB15k --dimension 2000 --learning_rate 0.01 --model_path stored_models/ComplEx_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python3 scripts/complex/train.py --dataset WN18 --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2
    ```
    ```python
    python3 scripts/complex/test.py --dataset WN18 --dimension 500 --learning_rate 0.1 --model_path stored_models/ComplEx_WN18.pt
    ```

  * **FB15k-237**
  
    ```python
    python3 scripts/complex/train.py --dataset FB15k-237 --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2
    ```
    ```python
    python3 scripts/complex/test.py --dataset FB15k-237 --dimension 1000 --learning_rate 0.1 --model_path stored_models/ComplEx_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python3 scripts/complex/train.py --dataset WN18RR --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --valid 20
    ```
    ```python
    python3 scripts/complex/test.py --dataset WN18RR --dimension 500 --learning_rate 0.1 --model_path stored_models/ComplEx_WN18RR.pt
    ```

  * **YAGO3-10**
  
    ```python
    python3 scripts/complex/train.py --dataset YAGO3-10 --optimizer Adagrad --max_epochs 50 --dimension 1000 --batch_size 1000 --learning_rate 0.1 --valid 10 --reg 5e-3
    ```
    ```python
    python3 scripts/complex/test.py --dataset YAGO3-10 --dimension 1000 --learning_rate 0.1 --model_path stored_models/ComplEx_YAGO3-10.pt
    ```
    
* **ConvE**

  * **FB15k**

    ```python
    python3 scripts/conve/train.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset FB15k --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python3 scripts/conve/train.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset WN18  --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt
    ```
    
  * **FB15k-237**
  
    ```python
    python3 scripts/conve/train.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10		
    ```
    ```python
    python3 scripts/conve/test.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python3 scripts/conve/train.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 20
    ```
    ```python
    python3 scripts/conve/test.py --dataset WN18RR --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_WN18RR.pt 
    ```
    
  * **YAGO3-10**
    ```python
    python3 scripts/conve/train.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --valid 10
    ```
    ```python
    python3 scripts/conve/test.py --dataset YAGO3-10 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --model_path stored_models/ConvE_YAGO3-10.pt
    ```

* **TransE**

  * **FB15k**

    ```python
    python3 scripts/transe/train.py  --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2
    ```
    ```python
    python3 scripts/transe/test.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt
    ```

  * **WN18**
  
    ```python
    python scripts/transe/train.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --valid 10
    ```
    ```python
    python scripts/transe/test.py --dataset WN18 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt
    ```

  * **FB15k-237**
  
    ```python
    python3 scripts/transe/train.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --valid 10 --margin 5
    ```
    ```python
    python3 scripts/transe/test.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt
    ```

  * **WN18RR**
  
    ```python
    python scripts/transe/train.py  --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2
    ```
    ```python
    python scripts/transe/test.py  --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt
    ```

  * **YAGO3-10**
  
    ```python
    python3 scripts/transe/train.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50.0 --margin 5 --valid 20
    ```
    ```python
    python3 scripts/transe/test.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50.0 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt
    ```
    

### End-to-end experiments

We report in this section how to use extract explanations for a trained model, and how to verify their effectiveness. 
To run the following commands, the `.pt`file of the saved model needs to be avaiable in folder `Kelpie/stored_models`.

Each end-to-end experiment is composed of two separate steps:
* an **explanation extraction** step, where the we use Kelpie to identify the explanations for a set of predictions. This step is performed by running the `explain.py` script of the model to use, i.e., `scripts/complex/explain.py`, `scripts/conve/explain.py`, or `scripts/transe/explain.py`; it can be performed using the Kelpie engine, or by choosing a specific baseline among the supported ones (`K1`, `data_poisoning` or, for all models excluding TransE, `Criage`). The explanation extraction process writes in the main folder of Kelpie a few output files and logs, including an `output.txt` file with the definitive extracted explanations. 

* an **explanation verification** step, where the we alter the training set and retrain the model to verify that the extracted explanations are indeed effective. This step is performed by running the the `verify_explanations.py` script of the model to use, i.e., `scripts/complex/verify_explanations.py`, `scripts/conve/verify_explanations.py`, or `scripts/transe/verify_explanations.py`; it reads from the Kelpie main folder the `output.txt` file of a previous explanation extraction step, and proceeds to verify the loaded explanations. Unlike the explanation extraction, the explanation verification step does not need to specify if the explanations were obtained by Kelpie or by a baseline, so the command to run is always the same. The explanation verification step generates an `output_end_to_end.csv` output file that can then be used to plot the experiment result tables (e.g., in the reproducibility package).
`
We report in the following, for each model, dataset and scenario, the commands to perform Explanation Extraction using `Kelpie`, `K1`, `data_poisoning` or, for all models excluding TransE, `Criage`, followed by the command to perform explanation verification.

* **ComplEx**
   * **FB15k**
      * **Necessary Scenario**
         * **Explanation Extraction**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --baseline k1
             ```
             
            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --facts_to_explain_path input_facts/complex_fb15k_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient
           ```
           
  * **WN18**
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode necessary --baseline criage
             ```

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**
            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_wn18_random.csv --mode sufficient --baseline criage
             ```

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient
           ```

  * **FB15k-237**
  
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**

             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --facts_to_explain_path input_facts/complex_fb15k237_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --facts_to_explain_path input_facts/complex_wn18rr_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient
           ```

  * **YAGO3-10**
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**
 
           * **Kelpie**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/complex/explain.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --facts_to_explain_path input_facts/complex_yago_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient
           ```
    
* **ConvE**

  * **FB15k**
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_nec.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --facts_to_explain_path input_facts/conve_fb15k_random_suff.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode sufficient
           ```

  * **WN18**

      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --facts_to_explain_path input_facts/conve_wn18_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

    
  * **FB15k-237**
  
      * **Necessary Scenario**

         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --facts_to_explain_path input_facts/conve_fb15k237_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --facts_to_explain_path input_facts/conve_wn18rr_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode sufficient
           ```
    
  * **YAGO3-10**

      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode necessary --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient --baseline data_poisoning
             ```

            * **Criage**
            
             ```python
             python3 scripts/conve/explain.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --facts_to_explain_path input_facts/conve_yago_random.csv --mode sufficient --baseline criage
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode sufficient
           ```

* **TransE**

  * **FB15k**

      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --facts_to_explain_path input_facts/transe_fb15k_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode sufficient
           ```


  * **WN18**

      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode necessary --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.01  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002  --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --facts_to_explain_path input_facts/transe_wn18_random.csv --model_path stored_models/TransE_WN18.pt --mode sufficient --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode sufficient
           ```


  * **FB15k-237**
  
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode necessary --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --facts_to_explain_path input_facts/transe_fb15k237_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --facts_to_explain_path input_facts/transe_fb15k237_random.csv --model_path stored_models/TransE_FB15k-237.pt --mode sufficient --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode sufficient
           ```


  * **WN18RR**

      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.01 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode sufficient --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python scripts/transe/explain.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --facts_to_explain_path input_facts/transe_wn18rr_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```



  * **YAGO3-10**
  
      * **Necessary Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary --baseline k1
             ```

            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode necessary --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**
         * **Explanation Extraction**

            * **Kelpie**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.01 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode sufficient
             ```

            * **K1**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode sufficient --baseline k1
             ```
             
            * **Data Poisoning**
            
             ```python
             python3 scripts/transe/explain.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weight 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --facts_to_explain_path input_facts/transe_yago_random.csv --mode sufficient --baseline data_poisoning
             ```
             
         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode sufficient
           ```

### Minimality Experiments

The minimality experiments verify if the extracted explanations are indeed the smallest effective combinations of facts. To check if an explanation is minimal, these experiments randomly remove a subset of its facts before applying it to the training set and verifying its effectiveness.
Note that minimality experiments can only be performed on Kelpie, and not on its baselines: this is due to Kelpie being the only system that extracts explanations longer than 1.

The minimality experiments are very similar to the end-to-end ones, as they also involve an explanation extraction step and an explanation verification step.
* The **explanation extraction** step is identical to the one of end-to-end experiments.
* The **explanation verification** step is performed by running the the `verify_explanations_skip_random.py` script of the model to use, i.e., `scripts/complex/verify_explanations_skip_random.py`, `scripts/conve/verify_explanations_skip_random.py`, or `scripts/transe/verify_explanations_skip_random.py. These scripts, similarly to the end-to-end verification ones, read from the Kelpie main folder the `output.txt` file of a previous explanation extraction step. The difference from the end-to-end verification scripts is that in this case, the loaded explanations, before being verified, are randomly sampled. The explanation verification step generates an `output_end_to_end_skipping_random_facts.txt` output file that can then be used in conjunction to the explanation verification output of analogous the end-to-end experiment to plot the experiment result tables (e.g., in the reproducibility package).

For the sake of simplicity, we only report here the explanation verification commands, as the explanation extraction ones are identical to those already reported for the explanation extraction step:


* **ComplEx**
   * **FB15k**
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**

           ```python
            python3 scripts/complex/verify_explanations_skip_random.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**

           ```python
            python3 scripts/complex/verify_explanations_skip_random.py --dataset FB15k --model_path stored_models/ComplEx_FB15k.pt --optimizer Adagrad --dimension 2000 --batch_size 100 --max_epochs 200 --learning_rate 0.01 --reg 2.5e-3 --mode sufficient
           ```
           
  * **WN18**
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**

           ```python
            python3 scripts/complex/verify_explanations_skip_random.py --dataset WN18 --model_path stored_models/ComplEx_WN18.pt --optimizer Adagrad --dimension 500 --batch_size 1000 --max_epochs 20 --learning_rate 0.1 --reg 5e-2 --mode sufficient
            ```

  * **FB15k-237**
  
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset FB15k-237 --model_path stored_models/ComplEx_FB15k-237.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 100 --learning_rate 0.1 --reg 5e-2 --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset WN18RR --model_path stored_models/ComplEx_WN18RR.pt --optimizer Adagrad --dimension 500 --batch_size 100 --max_epochs 100 --learning_rate 0.1 --reg 1e-1 --mode sufficient
           ```

  * **YAGO3-10**
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/complex/verify_explanations_skip_random.py --dataset YAGO3-10 --model_path stored_models/ComplEx_YAGO3-10.pt --optimizer Adagrad --dimension 1000 --batch_size 1000 --max_epochs 50 --learning_rate 0.1 --reg 5e-3 --mode sufficient
           ```
    
* **ConvE**

  * **FB15k**
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset FB15k --max_epochs 1000 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k.pt --mode sufficient
           ```

  * **WN18**

      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset WN18 --max_epochs 150 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18.pt --mode necessary
           ```

    
  * **FB15k-237**
  
      * **Necessary Scenario**


         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python scripts/conve/verify_explanations_skip_random.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset FB15k-237 --max_epochs 60 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_FB15k-237.pt --mode sufficient
           ```

  * **WN18RR**
  
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset WN18RR --max_epochs 90 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_WN18RR.pt --mode sufficient
           ```
    
  * **YAGO3-10**

      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/conve/verify_explanations_skip_random.py --dataset YAGO3-10 --max_epochs 500 --batch_size 128 --learning_rate 0.003 --dimension 200 --input_dropout 0.2 --hidden_dropout 0.3 --feature_map_dropout 0.2 --decay_rate 0.995 --model_path stored_models/ConvE_YAGO3-10.pt --mode sufficient
           ```

* **TransE**

  * **FB15k**

      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset FB15k --max_epochs 200 --batch_size 2048 --learning_rate 0.00003 --dimension 200 --negative_samples_ratio 5 --regularizer_weight 2.0 --margin 2 --model_path stored_models/TransE_FB15k.pt --mode sufficient
           ```


  * **WN18**

      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations_skip_random.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 end_to_end_testing/transe/verify_explanations_skip_random.py --dataset WN18 --max_epochs 250 --batch_size 2048 --learning_rate 0.0002 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 0 --margin 2 --model_path stored_models/TransE_WN18.pt --mode sufficient
           ```


  * **FB15k-237**
  
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset FB15k-237 --max_epochs 100 --batch_size 2048 --learning_rate 0.0004 --dimension 50 --negative_samples_ratio 15 --regularizer_weight 1.0 --margin 5 --model_path stored_models/TransE_FB15k-237.pt --mode sufficient
           ```


  * **WN18RR**

      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python scripts/transe/verify_explanations_skip_random.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset WN18RR --max_epochs 200 --batch_size 2048 --learning_rate 0.0001 --dimension 50 --negative_samples_ratio 5 --regularizer_weight 50.0 --margin 2 --model_path stored_models/TransE_WN18RR.pt --mode necessary
           ```



  * **YAGO3-10**
  
      * **Necessary Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode necessary
           ```

      * **Sufficient Scenario**

         * **Explanation Extraction**
           
           See the analogous end-to-end kelpie explanation extraction command.

         * **Explanation Verification**
         
           ```python
           python3 scripts/transe/verify_explanations_skip_random.py --dataset YAGO3-10 --max_epochs 100 --batch_size 2048 --learning_rate 0.0001 --dimension 200 --negative_samples_ratio 10 --regularizer_weigh 50 --margin 5 --model_path stored_models/TransE_YAGO3-10.pt --mode sufficient
           ```

