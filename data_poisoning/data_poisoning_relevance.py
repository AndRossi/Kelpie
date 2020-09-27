import numpy
from dataset import Dataset
from link_prediction.models.complex import ComplEx

def compute_fact_relevance(model: ComplEx,
                           dataset: Dataset,
                           sample_to_explain,
                           perspective="head",
                           perturbation_step=0.05,
                           lambd=1):
    head_id, relation_id, tail_id = sample_to_explain
    entity_to_explain_id = head_id if perspective == "head" else tail_id

    # get the embedding of the head entity, of the relation, and of the tail entity of the fact to explain
    head_embedding = model.entity_embeddings[head_id].detach().reshape(1, model.dimension)
    relation_embedding = model.relation_embeddings[relation_id].detach().reshape(1, model.dimension)
    tail_embedding = model.entity_embeddings[tail_id].detach().reshape(1, model.dimension)

    # set the requires_grad flag of the embedding of the entity to explain to true
    entity_to_explain_embedding = head_embedding if perspective == "head" else tail_embedding
    entity_to_explain_embedding.requires_grad=True

    # compute the score of the fact, and extract the gradient of the embedding of the entity to explain
    # then, perturbate the embedding of the entity to explain
    score = model.score_embeddings(head_embedding, relation_embedding, tail_embedding)
    score.backward()
    gradient = entity_to_explain_embedding.grad[0]
    perturbed_entity_to_explain_embedding = entity_to_explain_embedding.detach()-perturbation_step*gradient.detach()

    # extract all training samples containing the entity to explain, and compute their scores
    samples_containing_entity_to_explain = numpy.array([(h, r, t) for (h, r, t) in dataset.train_samples if entity_to_explain_id in [h, t]])
    if len(samples_containing_entity_to_explain) == 0:
        return None

    original_scores = model.score(samples_containing_entity_to_explain)

    # extract the embeddings for the head entities, relations, and tail entities
    # of all training samples containing the entity to explain;
    head_embeddings = model.entity_embeddings[samples_containing_entity_to_explain[:, 0]]
    relation_embeddings = model.relation_embeddings[samples_containing_entity_to_explain[:, 1]]
    tail_embeddings = model.entity_embeddings[samples_containing_entity_to_explain[:, 2]]

    # for the entity to explain, use the perturbed embedding rather than the original one
    for i in range(samples_containing_entity_to_explain.shape[0]):
        (h, r, t) = samples_containing_entity_to_explain[i]
        if h == entity_to_explain_id:
            head_embeddings[i] = perturbed_entity_to_explain_embedding
        elif t == entity_to_explain_id:
            tail_embeddings[i] = perturbed_entity_to_explain_embedding

    # compute the scores of all training samples containing the entity to explain
    # using its perturbed embedding rather than the original one
    perturbed_scores = model.score_embeddings(head_embeddings, relation_embeddings, tail_embeddings).detach().cpu().numpy()

    # now for each training sample containing the entity to explain you have
    # both the original score and the score computed with the perturbed embedding
    # so you can compute the relevance of that training sample as original_score - lambda * perturbed_score
    sample_2_relevance = {}
    for i in range(samples_containing_entity_to_explain.shape[0]):
        sample_2_relevance[tuple(samples_containing_entity_to_explain[i])] = (original_scores[i] - lambd * perturbed_scores[i])[0]

    most_relevant_samples = sorted(sample_2_relevance.items(), key=lambda x:x[1], reverse=True)

    return most_relevant_samples
