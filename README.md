# KGEN

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.1.0/) and then put the files in `data/mimic_cxr`.

## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

or Run the following code in a terminal

```cmd
python main_train.py --useKG --useVTFCM --AM SA
```



## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

or Run the following code in a terminal

```cmd
python main_test.py --useKG --useVTFCM --AM SA --load results/KGEN/model_best_7.pth
```

Follow [CheXpert](https://github.com/MIT-LCP/mimic-cxr/tree/master/txt/chexpert) or [CheXbert](https://github.com/stanfordmlgroup/CheXbert) to extract the labels and then run `python compute_ce.py`. Note that there are several steps that might accumulate the errors for the computation, e.g., the labelling error and the label conversion. We refer the readers to those new metrics, e.g., [RadGraph](https://github.com/jbdel/rrg_emnlp) and [RadCliQ](https://github.com/rajpurkarlab/CXR-Report-Metric).



## VK-Mamba

### Installation

1. Install the official Mamba library by following the instructions in the [hustvl/Vim](https://github.com/hustvl/Vim) repository.
2. After installing the official Mamba library, replace the mamba_simpy.py file in the installation directory with the mamba_simpy.py file provided in this mamba block directory.

### VK-Mamba

The Mamba Block modules introduced in the paper are available in the mamba_module.py file in this directory. These modules offer advanced features for pan-sharpening and can be seamlessly integrated into your existing projects.

## LLM
### Prompts
#### Role Setting
You are a professional expert in entity relation extraction for medical texts, equipped with extensive knowledge of medical terminology. You can accurately identify core entities and defined relations from medical reports.


#### Core Tasks
Extract entity relations that meet the requirements from the provided medical report texts, present them in the form of triples, and output the results in CSV format. The final output can be directly copied to a memo for use.

#### Rules to Be Strictly Implemented

-  Restrictions on Relation Types
Only the following 5 relation types are allowed; no other relations shall be added or derived:
locate: Indicates spatial/anatomical positional association between entities (e.g., the positional relationship between an organ and its part, or between a lesion and tissue).
affect: Indicates an influence, effect, damage, or improvement relationship between entities (e.g., the effect of a drug on a lesion, or the impact of a disease on an organ).
associate: Indicates a concomitant, correlative, or co-occurrence relationship between entities (e.g., the association between symptoms and diseases, or between complications and primary diseases).
appear: Indicates the state of an entity’s occurrence, existence, or detection (e.g., a lesion, symptom, or index recorded as "existing/appearing" in a report).
degree: Indicates the degree, magnitude, or numerical grade relationship of an entity (e.g., the value of an index, the size of a lesion, or the severity of a symptom).
- Requirements for Entity Extraction
Strictly retain the original English expressions in the medical report, including professional terminology, abbreviations, detection indicators, drug names, lesion names, etc. Do not translate, rewrite, supplement, or simplify; avoid omitting details of English entities in the original text.
- Requirements for Triple Quality
Each triple must follow the structure Entity1, Relation, Entity2, where all three elements are valid information without invalid content such as None, unknown, or empty values.
Global deduplication: The same triple shall be retained only once, regardless of how many times it appears in the report.
Only extract entity relations explicitly stated in the report; do not make subjective inferences, associations, or supplement unmentioned information.
- Requirements for Output Format
Pure CSV format without additional titles, descriptions, separators, or format decorations.
One triple per line, separated by an English comma "," with no extra spaces.
The output results can be directly copied and pasted to a memo without subsequent format adjustments.
#### Example Reference (Medical Scenario)
- Sample Medical Text Fragment
"The patient's right lung has a 3cm nodule, which is associated with lung adenocarcinoma. Chemotherapy affects the size of the nodule, and the degree of shrinkage is about 1cm."
Sample Output (CSV Format)
right lung,locate,3cm nodule
3cm nodule,associate,lung adenocarcinoma
Chemotherapy,affect,size of the nodule
shrinkage,degree,1cm
### Medical Report to Be Processed
Upload medical report files

### Other
Given that large models currently have limited data processing capabilities and cannot handle large-scale datasets in a single batch, we split the data into multiple files, ensuring that each file does not exceed 20MB, and then input them into the MML system for separate processing. To prevent the model from generating hallucinations, we processed each file five times and took the intersection of the five processing results as the final outcome.

The large model adopted in this experiment is Claude 3.5 Sonnet on the [Monica platform](https://monica.im/). It is used to extract the entity relationships of medical reports, and combined with a memo tool to realize the storage of medical reports.