# A Weakly-Supervised Machine Learning Method for fine-grained semantic indexing of biomedical literature

---

## About this repository

This repository includes the final (third) part of [my Thesis](https://hdl.handle.net/10889/27628) [1] on the fine-grained semantic indexing of biomedical literature.

The first and second parts involve the development of a large-scale dataset and its enhancement with weak supervision and are part of a [paper in the Journal of Biomedical Informatics (JBI)](https://doi.org/10.1016/j.jbi.2023.104499) [2] _(an [extended version]((https://arxiv.org/pdf/2301.09350v1)) of this study [3] is also available)_. The implementation of the previous parts is available [here](https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset).

In the last part of the paper, [Anastasios Nentidis](https://www.iit.demokritos.gr/el/people/anastasios-nentidis/) uses a [Deep Learning approach](https://github.com/tasosnent/DBM), while in my thesis I focus on a Machine Learning method, an XGBoost model _(as well as a logistic regression model as a baseline)_.

Below, I summarize the work with brief explanations in relation to the repository code. For more details and analysis, please refer to the above references and sources.

### Requirements

 - These scripts are written in **Python 3.8.2**.  
 - Libraries and versions required are listed in [**requirements.txt**](https://github.com/ThomasChatzopoulos/weak_labeling_snorkel_xgboost/blob/main/requirements.txt).  
 - Available disk space for the entire project: **at least 350 GB**.

---

## The methods & how to run

### Abstract

The semantic indexing of the biomedical literature in MEDLINE/PubMed is performed with descriptors from the MeSH treasure, which represent specific concepts of the biomedical community. Synonymous or related biomedical concepts are often together and represented only by a coarse-grained descriptor, based on which the corresponding bibliography is also indexed. In this work, a method is developed for the automated improvement of biomedical concepts by exploring machine learning approaches. Due to the absence of labeled data, weak supervision techniques are used based on the occurrence of the concept in the text of the articles. The evaluation of the method is performed retrospectively, on data for concepts that have been gradually promoted to fine-grained descriptors in the MeSH treasure and thus used to annotate and index the articles. Although concept occurrence in article text is a powerful heuristic for fine-grained article indexing, experiments show that combining it with other, simpler heuristics can, in some cases, further strengthen it. Using heuristics to develop weakly supervised machine learning models can further improve the results. Overall, the proposed method succeeds in improving the indexing of biomedical literature to fine-grained concepts in an automated manner for most of the use cases.

### Introduction

Semantic indexing of biomedical literature refers to the annotation of articles with labels from a treasure containing biomedical terminology. As database of articles-citations is used the [MEDLINE/PubMed](https://pubmed.ncbi.nlm.nih.gov/about/), in which the articles are indexed with topic descriptors from the [Medical Subject Headings (MeSH)](https://www.nlm.nih.gov/mesh/meshhome.html) treasure.


<figure>
    <img src="/images\pubmed_mesh.png"
         alt="Articles annotation">
    <figcaption>Annotation of PubMed articles with descriptors from the MeSH treasure.</figcaption>
</figure>


Among other factors, the considerable growth in the volume of bibliographic references in recent years has accentuated the necessity for annotating articles with fine-grained labels (MeSH Headings), in contrast to the previously utilized coarse-grained ones. Furthermore, there is a growing imperative to advance the automation of annotation and related processes.

The work is structured as follows:


**1. Dataset development:** The dataset creation is based on a retrospective scenario, using the concept-occurrence in the title or abstract of an article as a heuristic.
An evaluation of a previous method [4] using the appearance-concept heuristic was also performed on a small dataset.

**2. Dataset enhancement:** The dataset enhancement is achieved by combining a number of heuristics, beyond the concept occurrence.

**3. ML models development:** The development of machine learning models (XGBoost & Logistic Regression) for automated suggestion of fine-grained headings in biomedical literature, instead of coarse-grained ones.

---

### The parts of the work

---

## References
[1] Χατζόπουλος, Θ. (2024). Λεπτομερής σημασιολογική ευρετηρίαση σε βιοϊατρική βιβλιογραφία. ΝΗΜΕΡΤΗΣ, Ιδρυματικό Αποθετήριο πανεπιστημίου Πατρών, 2024, [https://hdl.handle.net/10889/27628](https://hdl.handle.net/10889/27628)

[2] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2023). Large-scale investigation of weakly-supervised deep learning for the fine-grained semantic indexing of biomedical literature. Journal of Biomedical Informatics, Volume 146, 2023, 104499, ISSN 1532-0464, [https://doi.org/10.1016/j.jbi.2023.104499](https://doi.org/10.1016/j.jbi.2023.104499).

[3] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2023). Large-scale fine-grained semantic indexing of biomedical literature based on weakly-supervised deep learning. arXiv preprint _(An extended version of [2])_. [https://arxiv.org/pdf/2301.09350v1.pdf](https://arxiv.org/pdf/2301.09350v1.pdf)

[4] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2020). Beyond MeSH: Fine-grained semantic indexing of biomedical literature based on weak supervision. Information Processing & Management, Volume 57, Issue 5, 2020, 102282,
ISSN 0306-4573, [https://doi.org/10.1016/j.ipm.2020.102282](https://doi.org/10.1016/j.ipm.2020.102282)
