import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

KELPIE_ROOT = os.path.realpath(os.path.join(__file__, os.pardir))
IMAGES_FOLDER = os.path.join(KELPIE_ROOT, "reproducibility_images")
OUTPUT_FILE = "reproduced_experiments.pdf"
OUTPUT_FILEPATH = os.path.join(KELPIE_ROOT, OUTPUT_FILE)


def generate_pdf():

    print(f"Saving the PDF report in {OUTPUT_FILE}...")

    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=letter,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=18,
    )

    styles = getSampleStyleSheet()
    flowables = []
    intro_text = """Kelpie is a post-hoc local explainability framework designed for embedding-based models that perform Link Prediction (LP) on Knowledge Graphs (KGs).<br />
    Kelpie explains predictions by identifying the training facts that have been most relevant to infer them.<br />
    Intuitively, when explaining a tail prediction <h, r, t>, Kelpie identifies the smallest set of training facts mentioning h that have instrumental to predict t as a tail.<br />
    Analogously, a head prediction <h, r, t> would be explained with a combination of the training facts mentioning t.<br />
    We report in this document the outcomes of the reproducibility experiments.<br /><br />
    """
    environment_text = """<b>Environment and Prerequisites</b><br />
    All the experiments reported in our paper have been run on an Ubuntu 18.04.5 environment with Python 3.7.7, CUDA Version: 11.2 and Driver Version: 460.73.01.<br />
    Our server featured 88 CPUs Intel Core(TM) i7-3820 at 3.60GHz, 516GB RAM and 4 NVIDIA Tesla GPUs with 16GB VRAM each.<br />
    When using different software or hardware components, slightly different results may be obtained; nonetheless, the overall observed behaviors should match the trends and conclusions seen in our paper.<br /><br />
    """
    models_datasets_text = """<b>Models and Datasets</b><br />
    The formulation of Kelpie supports all Link Prediction models based on embeddings.<br />
    We run our experiments on three models that rely on very different architectures: ComplEx, ConvE and TransE.<br />
    We explain their predictions on the 5 best-established datasets in literature: FB15k, WN18, FB15k-237, WN18RR and YAGO3-10.<br /><br />
    """

    data_text = """<b>Data and Availability</b><br />
    The training, validation and test sets of all the aforementioned datasets are distributed in this repository in the "Kelpie/data" folder.<br />
    To facilitate reproducibility, we make available through FigShare the 15 .pt model files obtained by training each model on each dataset: https://figshare.com/s/ede27f3440fe742de60b<br />
    The .pt model files should be downloaded and placed in a "Kelpie/stored_models" folder.<br />
    We share all the original result files obtained in our experiments in the "Kelpie/results.zip" archive.<br />
    We also report additional experiments in our GitHub repository https://github.com/AndRossi/Kelpie; the corresponding result files can be found in the "Kelpie/additional_results.zip" archive.<br /><br />
    """

    end_to_end_text = """<b>End-to-end experiments</b><br />
    We showcase the effectiveness of Kelpie by explaining, for each model and dataset, the tail predictions of a set of 100 correctly predicted test facts both in necessary and in sufficient scenario.<br />
    We then measure the effectiveness of Kelpie explanations by observing:<br />
    - in the necessary scenario, how removing the explanation facts worsens the prediction metrics (MRR and H@1 decrease);<br />
    - in the sufficient scenario, how adding those facts to other entities converts them to the same prediction (MRR and H@1 increase);<br />
    We report the results of a comparison of the effectiveness of Kelpie against the following baselines:<br />
    - a Kelpie variant restricted to 1-sized explanations, called K1;<br />
    - the Data Poisoning (DP) method described in "Data poisoning attack against knowledge graph embedding" by Zhang et al.;<br />
    - the Criage method described in "Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications", by Pezeshkpour et al.;<br /><br />
    """

    end_to_end_necessary_table_title = """End-to-end necessary explanations: effectiveness (Paper Table 3)<br />"""
    end_to_end_necessary_table_file = os.path.join(IMAGES_FOLDER, "end_to_end_table_necessary.png")
    end_to_end_sufficient_table_title = """End-to-end sufficient explanations: effectiveness (Paper Table 4)<br />"""
    end_to_end_sufficient_table_file = os.path.join(IMAGES_FOLDER, "end_to_end_table_sufficient.png")

    explanation_lengths_text = """<b>Explanation Lengths</b><br />
    We report in the following charts the lengths of the explanations extracted in our end-to-end experiments.
    More specifically, we report their distribution for each model and dataset, both in the necessary and in the sufficient scenario.<br /><br />
    """

    explanation_lengths_necessary_table_title = """End-to-end necessary explanations: length distribution (Paper Table 5)<br />"""
    explanation_lengths_necessary_table_file = os.path.join(IMAGES_FOLDER, "explanation_lengths_table_necessary.png")
    explanation_lengths_sufficient_table_title = """End-to-end sufficient explanations: length distribution (Paper Table 5)<br />"""
    explanation_lengths_sufficient_table_file = os.path.join(IMAGES_FOLDER, "explanation_lengths_table_sufficient.png")

    minimality_text = """<b>Minimality experiments</b><br />
    To demonstrate that the explanation identified by Kelpie are indeed the <b>smallest</b> sets of facts that satisfy our definitions, we run minimality experiments both in the necessary and in the sufficient scenario.<br />
    In both cases we randomly sample the explanations extracted in our end-to-end experiments; then, we verify that the randomly sampled explanations are less effective than their "full" counterparts.
    
    We report in the following tables such a decrease in effectiveness by measuring which portion of the H@1 and MRR variation obtained by the "full" explanations is lost when using the sampled explanations instead.
    """

    minimality_necessary_table_title = """Minimality necessary experiment: effectiveness (Paper Table 6)<br />"""
    minimality_necessary_table_file = os.path.join(IMAGES_FOLDER, "minimality_table_necessary.png")
    minimality_sufficient_table_title = """Minimality sufficient explanations: effectiveness (Paper Table 6)<br />"""
    minimality_sufficient_table_file = os.path.join(IMAGES_FOLDER, "minimality_table_sufficient.png")

    prefilter_times_text = """<b>Pre-Filter</b><br /><br />
    We compare here the explanation extraction times, for entities of various degrees, obtained using and not using a Pre-Filter component.<br /><br />
    """
    prefilter_times_plot_title = """Extraction times by entity degree with and without Pre-Filters (Paper Figure 6)<br />"""
    prefilter_times_plot_file = os.path.join(IMAGES_FOLDER, "extraction_times_with_and_without_prefilter_plot.png")

    end_user_study = """<b>End User Study</b><br />
    We report here the results of our End User Study.<br /><br />
    """
    end_user_study_title = """End User Study results (Paper Figure 7)<br />"""
    end_user_study_file = os.path.join(IMAGES_FOLDER, "user_study.png")

    additional_experiments_text = """<b><u>Additional Experiments</u></b><br />
    We include here additional experiments that were not in our paper due to space constraints.<br /><br /><br />
    """

    necessary_threshold_text = """<b>Explanation Builder: Acceptance Threshold (necessary scenario)</b><br />
    We study here how varying the values of the acceptance threshold ξ affects the effectiveness of Kelpie necessary explanations.<br /><br />
    """

    necessary_threshold_table_title = """Necessary Threshold ξ variation results"""
    necessary_threshold_table_file = os.path.join(IMAGES_FOLDER, "xsi_threshold_comparison_table.png")

    prefilter_threshold_text = """<b>Pre-Filter Threshold</b><br />
    We study here how varying the values of the Pre-Filter threshold k affects the effectiveness of Kelpie necessary and sufficient explanations.<br /><br />
    """

    prefilter_threshold_necessary_table_title = """Pre-Filter threshold variation results: necessary scenario"""
    prefilter_threshold_necessary_table_file = os.path.join(IMAGES_FOLDER,
                                                            "prefilter_threshold_comparison_table_necessary.png")
    prefilter_threshold_sufficient_table_title = """Pre-Filter threshold variation results: sufficient scenario"""
    prefilter_threshold_sufficient_table_file = os.path.join(IMAGES_FOLDER,
                                                             "prefilter_threshold_comparison_table_sufficient.png")

    prefilter_type_text = """<b>Pre-Filter Type</b><br />
    We compare here the results obtained using a graph typology-based Pre-Filter with those obtained using an entity type-based Pre-Filter.<br /><br />
    """

    prefilter_type_necessary_table_title = """Pre-Filter type comparison: necessary scenario"""
    prefilter_type_necessary_table_file = os.path.join(IMAGES_FOLDER, "prefilter_type_comparison_table_necessary.png")
    prefilter_type_sufficient_table_title = """Pre-Filter type comparison: sufficient scenario"""
    prefilter_type_sufficient_table_file = os.path.join(IMAGES_FOLDER, "prefilter_type_comparison_table_sufficient.png")

    flowables.append(Paragraph(intro_text, style=styles["Normal"]))
    flowables.append(Paragraph(environment_text, style=styles["Normal"]))
    flowables.append(Paragraph(models_datasets_text, style=styles["Normal"]))
    flowables.append(Paragraph(data_text, style=styles["Normal"]))

    flowables.append(Paragraph(end_to_end_text, style=styles["Normal"]))
    flowables.append(PageBreak())

    flowables.append(Paragraph(end_to_end_necessary_table_title, style=styles["Normal"]))
    flowables.append(Image(end_to_end_necessary_table_file, width=400, height=175))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))
    flowables.append(Paragraph(end_to_end_sufficient_table_title, style=styles["Normal"]))
    flowables.append(Image(end_to_end_sufficient_table_file, width=400, height=175))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))

    flowables.append(Paragraph(explanation_lengths_text, style=styles["Normal"]))
    flowables.append(Paragraph(explanation_lengths_necessary_table_title, style=styles["Normal"]))
    flowables.append(Image(explanation_lengths_necessary_table_file, width=400, height=75))
    flowables.append(Paragraph(explanation_lengths_sufficient_table_title, style=styles["Normal"]))
    flowables.append(Image(explanation_lengths_sufficient_table_file, width=400, height=75))
    flowables.append(PageBreak())

    flowables.append(Paragraph(minimality_text, style=styles["Normal"]))
    flowables.append(Paragraph(minimality_necessary_table_title, style=styles["Normal"]))
    flowables.append(Image(minimality_necessary_table_file, width=400, height=75))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))
    flowables.append(Paragraph(minimality_sufficient_table_title, style=styles["Normal"]))
    flowables.append(Image(minimality_sufficient_table_file, width=400, height=75))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))

    flowables.append(Paragraph(prefilter_times_text, style=styles["Normal"]))
    flowables.append(Paragraph(prefilter_times_plot_title, style=styles["Normal"]))
    flowables.append(Image(prefilter_times_plot_file, width=200, height=150))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))
    flowables.append(PageBreak())

    flowables.append(Paragraph(end_user_study, style=styles["Normal"]))
    flowables.append(Paragraph(end_user_study_title, style=styles["Normal"]))
    flowables.append(Image(end_user_study_file, width=200, height=180))
    flowables.append(PageBreak())

    flowables.append(Paragraph(additional_experiments_text, style=styles["Normal"]))
    flowables.append(Paragraph(necessary_threshold_text, style=styles["Normal"]))
    flowables.append(Paragraph(necessary_threshold_table_title, style=styles["Normal"]))
    flowables.append(Image(necessary_threshold_table_file, width=400, height=150))
    flowables.append(Paragraph("<br /><br />", style=styles["Normal"]))

    flowables.append(Paragraph(prefilter_threshold_text, style=styles["Normal"]))
    flowables.append(Paragraph(prefilter_threshold_necessary_table_title, style=styles["Normal"]))
    flowables.append(Image(prefilter_threshold_necessary_table_file, width=400, height=75))
    flowables.append(Paragraph(prefilter_threshold_sufficient_table_title, style=styles["Normal"]))
    flowables.append(Image(prefilter_threshold_sufficient_table_file, width=400, height=75))
    flowables.append(PageBreak())

    flowables.append(Paragraph(prefilter_type_text, style=styles["Normal"]))
    flowables.append(Paragraph(prefilter_type_necessary_table_title, style=styles["Normal"]))
    flowables.append(Image(prefilter_type_necessary_table_file, width=450, height=65))
    flowables.append(Paragraph(prefilter_type_sufficient_table_title, style=styles["Normal"]))
    flowables.append(Image(prefilter_type_sufficient_table_file, width=450, height=65))

    doc.build(flowables)
    print("Done.")


if __name__ == '__main__':
    generate_pdf()
