# African Medical Question Answering Assistant

This project builds a question answering assistant that gives educational medical information in an African context. A user types a health question in everyday English, and a fine-tuned Transformer sentence-embedding model finds the closest human-written answer in the AfriMed-QA v2.5 dataset. Because the assistant only returns answers that already exist in the dataset, it does not generate new medical text, and a confidence check allows it to refuse questions that are outside medicine or not covered by the data instead of guessing.

> The assistant is for educational use only. It does not diagnose, prescribe treatment or replace health professionals.

| Link | Status |
|---|---|
| Live system |  |
| Demo video |  |
| Report |  |

## The Problem

Reliable health information that fits the African context can be hard to find, and general-purpose chatbots can produce fluent answers that contain invented medical claims. The goal of this project was therefore an assistant that answers only with human-written answers from a curated African dataset, that understands questions phrased differently from the dataset, and that says so clearly when a question is outside medicine or about a topic the dataset does not cover. No external AI or chatbot API is used, and every prediction comes from the model trained in this repository.

## The Dataset

The project uses [AfriMed-QA v2.5](https://github.com/intron-innovation/AfriMed-QA), a pan-African medical question answering dataset described by Olatunji et al. (2025). The file `afri_med_qa_15k_v2.5_phase_2_15275.csv` in `data/raw/` is byte-identical to the file in the official repository, since both have the Git blob hash `fc0857f019894d2cb7fd54e5f9df678e491ac010`, and `.gitattributes` stops Git from changing its line endings. The licence needed some care, because the GitHub repository states CC BY-NC-SA 4.0 while the Hugging Face dataset card (`intronhealth/afrimedqa_v2`) states CC BY-SA 4.0. We follow the stricter of the two, so the processed data in this repository is shared under CC BY-NC-SA 4.0 for non-commercial educational use.

The file contains 15,275 records in 30 columns: 10,000 consumer queries, 4,039 multiple-choice questions and 1,236 Short Answer Questions (SAQ). Only the SAQ records pair a question with a written human answer, so they form the knowledge base. For these records the answer is stored in `answer_rationale`, the only answer column that is filled. We used the original `question` column rather than `question_clean`, because `question_clean` splits medical terms apart, turning G6PD into "G 6 PD", HER2 into "HER 2" and B12 into "B 12".

| Property of the SAQ records | Value |
|---|---|
| Missing questions or answers | None |
| Duplicate questions | None, even after ignoring case and spacing |
| Contributor country | Nigeria 923, Kenya 195, Malawi 118 |
| Source | 877 crowdsourced records and 359 expert records |
| Specialty | Recorded for 361 records, about half of them General Surgery |
| Length | Median question of 12 words; median answer of 29 words, with a maximum of 900 |
| Human quality ratings | 50 records, of which 6 were rated as incorrect |

The most important property of the file is its official split, which follows the source of the records exactly. All 877 official training records are crowdsourced questions from Nigeria, and all 359 official test records were written by experts in Kenya, Malawi and Nigeria. The two groups also differ in style. The crowdsourced questions are general, such as "What is the most common cause of ... in Africa?", and their fluent one-sentence answers often repeat the question's wording, while the expert questions resemble exam prompts, such as "Complications of thyroidectomy", and their answers are often short lists. Coverage is also thin, since keyword counts give only 32 questions about malaria, 35 about HIV and 16 about hypertension, and many diseases do not appear at all.

## Preparing the Data

The cleaning, done by `src/text_cleaning.py` and shared with the web app, is deliberately light. We did not lowercase the text, remove stop words or apply stemming, because the Transformer model needs complete sentences and the TF-IDF vectorizer lowercases the text on its own.

| Cleaning step | Records affected |
|---|---|
| Collapse spaces, tabs and line breaks in questions into single spaces | 229 questions contained line breaks |
| Remove the transcript tag "[Speaker 2]:" and exam marks such as "(2marks)" | 1 and 4 |
| Remove answer text pasted into a question, and leftover multiple-choice letters | 4 |
| Remove the "Answer:" prefix from answers | 30 |
| Remove leading punctuation and a leftover option label from answers | 2 |
| Keep the line breaks inside answers, because many expert answers are lists | All answers |

Thirty records were then removed, 25 from the official training split and 5 from the official test split, and each one is listed with its reason in `data/processed/removed_records.csv`.

| Reason for removal | Records |
|---|---|
| Near-duplicate, with the same question meaning and an almost identical answer | 9 |
| Not health-related, about genealogy or child labour | 8 |
| Rated as incorrect (`quality = False`) by the AfriMed-QA reviewers | 6 |
| Malformed, with the answer in the question field instead of a question | 3 |
| Conflicting answer | 2 |
| Answer belonging to a different question | 1 |
| Exact duplicate after cleaning | 1 |

The manual decisions started from a similarity check that flagged every pair of records whose questions had a TF-IDF cosine similarity of at least 0.80 or whose answers had a similarity of at least 0.90. All 136 flagged pairs were read, and none of them crossed the official train and test split. The clearest problem was the first-line treatment of uncomplicated malaria, which was asked three times with different answers. "Subranoid" is not a recognised medicine and pyrethroids are insecticides rather than a treatment, so these two records were removed and the answer recommending artemisinin-based combination therapy, which agrees with the WHO guidelines for malaria, was kept. Most flagged pairs were kept, however, because they were template questions about different diseases or populations, sub-questions about the same clinical case, or paraphrases with compatible answers. The non-medical records were found by listing the 87 records that contain none of a broad set of health-related words and reading them, which showed that only the genealogy and child-labour records had no medical content. The 114 remaining questions about traditional or herbal remedies were kept, but the web app adds a caution to their answers because they describe contributors' practices rather than clinical guidance.

We kept the official split, as the project guidelines required, so the test set is never used for training or for choosing settings. The validation set is 15% of the official training records, selected with `GroupShuffleSplit` and a random seed of 42. Records whose questions or answers are near-identical, using the same thresholds as above, share a group that always stays within one set, which prevents a near-copy of a training question from inflating the validation results.

| Set | Records | Source | Countries |
|---|---|---|---|
| Training set | 711 | Crowdsourced | Nigeria |
| Validation set | 141 | Crowdsourced | Nigeria |
| Test set | 354 | Expert, never used for tuning | Kenya 190, Malawi 118, Nigeria 46 |

Finally, we measured the texts in model tokens, because the model truncates input after 512 tokens and was trained on texts of up to 250 tokens. The questions fit easily, but the test answers are longer and more varied, with 47 of them above 250 tokens. We did not shorten any answers, because the complete answer should be shown to the user, and only the 5 answers longer than 512 tokens are truncated when they are turned into vectors.

Two further question sets were built for evaluation. The first contains 40 paraphrased questions, 20 based on validation questions and 20 on test questions, reworded in everyday English. They are stored in `data/probes/paraphrases.csv` and are used only for evaluation. The second contains questions the assistant should refuse, all taken from real sources: non-medical questions from SQuAD 2.0 (Rajpurkar et al., 2018) and AfriMed-QA consumer questions about 59 diseases whose key words never appear in the knowledge base. The validation and test sides use different SQuAD articles and different diseases, and the three example questions from the assignment brief are added to the test side.

## How the Assistant Works

The assistant answers a question in four steps:

1. The question is lightly cleaned, in the same way as the dataset questions.
2. The fine-tuned model turns the question into a 384-number embedding.
3. The embedding is compared with the embeddings of all 1,205 answers in the knowledge base using cosine similarity.
4. The best similarity decides the response. At 0.52 or above, the assistant shows the answer together with the dataset question it belongs to. Between 0.32 and 0.52, it replies that it does not have enough reliable information, and below 0.32 it replies that the question is outside its scope.

The baseline is a TF-IDF model with unigrams and bigrams, compared with the answers through cosine similarity. The main model is [`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1), a six-layer MiniLM model with 22.7 million parameters, which we fine-tuned on the 711 training question and answer pairs with the Multiple Negatives Ranking loss and the AdamW optimiser, keeping the epoch with the best validation score. The two confidence thresholds were chosen on the validation set together with real questions that should be refused: non-medical questions from SQuAD 2.0 and AfriMed-QA consumer questions about diseases that never appear in the knowledge base. The final model is E5, the fine-tuned run with a batch size of 64, which had the highest validation MRR.

## Results

Each model ranks all 1,205 answers in the knowledge base for every question, and we measure how often the correct answer comes first (Recall@1) and its mean reciprocal rank (MRR). The validation questions resemble the training data, the test questions are unseen expert questions, and the paraphrases are 40 validation and test questions reworded in everyday English. The paraphrases are used only for evaluation.

| Experiment | What changed | Val R@1 | Val MRR | Test R@1 | Test MRR | Paraphrase R@1 |
|---|---|---|---|---|---|---|
| E1 | TF-IDF baseline | 0.553 | 0.656 | 0.237 | 0.294 | 0.075 |
| E2 | Pretrained embeddings, no training | 0.624 | 0.711 | 0.316 | 0.388 | 0.475 |
| E3 | Fine-tuned, learning rate 2e-5, batch size 32 (mean ± sd of 3 seeds) | 0.749 ± 0.004 | 0.826 ± 0.005 | 0.290 ± 0.009 | 0.373 ± 0.006 | 0.450 |
| E4 | Learning rate 5e-5 | 0.766 | 0.833 | 0.277 | 0.355 | 0.450 |
| E4b | Learning rate 1e-5 | 0.752 | 0.826 | 0.285 | 0.376 | 0.450 |
| E5 (final) | Batch size 64 | 0.766 | 0.837 | 0.285 | 0.372 | 0.450 |

Fine-tuning raised the validation MRR from 0.711 to about 0.83, and the three seeds show that this gain is stable. The gain did not carry over to the expert test set, however, where test Recall@1 fell from 0.316 for the pretrained model to 0.285 for the final model. The training and validation questions are crowdsourced questions from Nigeria, while the test questions are expert exam prompts, and the fine-tuned models returned answers they had been trained on more often than the pretrained model did (between 14% and 23% of test questions, compared with 13%). Both embedding models handled paraphrases far better than TF-IDF. Experiment E6 also tested comparing the user's question with the dataset questions as well as the answers. This helped reworded questions but halved the accuracy on new questions, so we kept the answers-only approach based on the validation results.

The confidence check was then tested on questions it had not seen during tuning.

| Question type | Answered | Answered correctly | Refused |
|---|---|---|---|
| Expert dataset questions (354) | 54.0% | 22.0% | 46.0% |
| Paraphrased questions (20) | 70% | 35% | 30% |
| Medical questions about uncovered topics (58) | 5.2% | n/a | 94.8% |
| Non-medical questions (78, including the 3 brief examples) | 1.3% | n/a | 98.7% |

With thresholds chosen in the same way, the TF-IDF baseline answered 24.4% of the non-medical questions, including "How do I repair my laptop?". The details are in notebooks [03](notebooks/03_fine_tuning.ipynb), [04](notebooks/04_confidence_threshold.ipynb) and [05](notebooks/05_error_analysis.ipynb).

## Running the Project

The whole workflow is contained in five notebooks, which should be run in order because each one saves the files that the next one needs.

| Notebook | Content |
|---|---|
| [01_data_preparation](notebooks/01_data_preparation.ipynb) | Explores the raw file, cleans it, removes unsuitable records and creates the training, validation and test sets. |
| [02_baseline_and_pretrained](notebooks/02_baseline_and_pretrained.ipynb) | Defines the evaluation method and measures the TF-IDF baseline (E1) and the pretrained model (E2). |
| [03_fine_tuning](notebooks/03_fine_tuning.ipynb) | Fine-tunes the model in experiments E3 to E5 and chooses the final model. This takes about two hours on a laptop CPU. |
| [04_confidence_threshold](notebooks/04_confidence_threshold.ipynb) | Builds the questions that should be refused, runs experiment E6, chooses the thresholds and exports `models/final/`. |
| [05_error_analysis](notebooks/05_error_analysis.ipynb) | Breaks the results down by group and examines successful and failed examples. |

```bash
python -m venv .venv
.venv\Scripts\activate                      # On Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
python app.py                               # Opens the web app at http://127.0.0.1:7860
```

The notebooks can also be opened and run in VS Code, Jupyter or Google Colab. The web app needs the `models/final/` folder, which notebook 04 creates.

To publish the web app on Hugging Face Spaces, first install `huggingface_hub` with pip and log in with `huggingface-cli login`, using a token with write permission from huggingface.co/settings/tokens. After notebook 04 has exported `models/final/`, the following command uploads the app, the source modules and the final model to the Space.

```bash
python deploy/upload_to_space.py --space <your-username>/african-medical-qa-assistant
```

## Repository Structure

```
notebooks/          the full workflow, from notebook 01 to notebook 05
app.py              Gradio web interface, which calls src/assistant.py
src/                small modules shared by the notebooks and the web app
  config.py         paths and fixed settings such as the random seed and the base model
  text_cleaning.py  light cleaning applied to dataset questions and user questions
  retrieval.py      TF-IDF and embedding retrievers and the ranking metrics
  evaluation.py     the evaluation method and the experiment table
  assistant.py      the assistant used by the app (model, thresholds and knowledge base)
data/raw/           the original AfriMed-QA v2.5 file
data/processed/     cleaned records, the three sets and the removed records with reasons
data/probes/        paraphrased questions and questions that should be refused
results/            metrics, predictions and figures
deploy/             requirements and upload script for publishing the app on Hugging Face Spaces
report/             the project report
models/             trained models, which are not stored in Git
```

## Limitations

The main limitation is coverage. A knowledge base of 1,206 question and answer pairs cannot cover every disease or medicine: malaria appears in 32 questions, for example, and COVID-19 in none. Questions about uncovered topics are usually refused, but 5.2% of the uncovered medical test questions were answered with a loosely related record.

The second limitation is the difference between the training and test data. All training and validation questions are crowdsourced questions from Nigeria, so the validation set could not reveal that the expert test questions, mostly from Kenya and Malawi, would be answered much less accurately. Very short answers such as "Anti-CCP" are especially hard to match, and a few answers act as hubs that attract unrelated questions. For instance, "How can someone know they may have malaria?" returns the one-word answer "Malaria", because the dataset has no record that directly lists the symptoms of malaria and the relevant records score below the threshold.

The data and evaluation also have limits. Only 50 of the 1,236 SAQ records carry expert quality ratings, so the answers were not all checked by clinicians. The strict Recall@1 metric counts a compatible answer from a near-identical record as wrong, and the paraphrase set is small. Finally, the assistant works only in English and can return only stored answers, so it cannot combine information from several records.

## Acknowledgements

This project uses the AfriMed-QA dataset created by Intron Health and its collaborators, the Sentence-Transformers, Hugging Face Transformers, PyTorch, scikit-learn and Gradio libraries, and SQuAD 2.0, which served only as a source of non-medical test questions. The full list of references is given in the report.

## References

Olatunji, T., Nimo, C., Owodunni, A., Abdullahi, T., Ayodele, E., Sanni, M., Aka, C., Omofoye, F., Yuehgoh, F., Faniran, T., Dossou, B. F. P., Yekini, M., Kemp, J., Heller, K., Omeke, J. C., Asuzu, C., Etori, N. A., Ndiaye, A., Okoh, I., ... Asiedu, M. N. (2025). AfriMed-QA: A pan-African, multi-specialty, medical question-answering benchmark dataset. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1948-1973). https://doi.org/10.18653/v1/2025.acl-long.96

Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)* (pp. 784-789). https://doi.org/10.18653/v1/P18-2124
