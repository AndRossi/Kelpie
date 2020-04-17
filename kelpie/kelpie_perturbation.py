import numpy

def perturbate_samples(samples):

    output = []
    skipped = []

    start_skip = 0
    skip_size = 10
    end_skip = start_skip + skip_size

    while start_skip < len(samples):
        output.append(numpy.vstack((samples[:start_skip], samples[end_skip:])))
        skipped.append(samples[start_skip:end_skip])
        start_skip += skip_size
        end_skip = min(end_skip+skip_size, len(samples))

    # for now, just return the facts themselves

    # for now, just skip the facts one by one and return the corresponding sets of facts
    #sets = []
    #for i in range(len(facts)):
    #    sets.append(numpy.array(facts[:i] + facts[i+1:]))
    #return numpy.array(sets)

    return output, skipped