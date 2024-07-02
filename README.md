
# GermEval2024-THAugs

This repository contains the code and datasets for our participation in the GermEval 2024 Shared Task on the Identification of Offensive Language. This project focuses on identifying and classifying offensive language in German using advanced natural language processing techniques and transformer-based models.


## Installation

Clone this repository and navigate to the project directory:

```sh
git clone https://github.com/yourusername/GermEval2024-THAugs.git
cd GermEval2024-THAugs
```

Install the required packages using pip:

```sh
pip install -e .
pip install -r requirements.txt
```

## Usage

For the best model used in the submission, you have to follow these steps:

1. Run the language model fine-tuning part:
    ```sh
    python -m src.lm_finetuning.lm_finetuning
    ```
2. Split the folds:
    ```sh
    python -m src.subtask_1.split_folds
    ```
3. Train each fold (we retrained folds that did not achieve the desired F1 score around 65%):
    ```sh
    python -m src.subtask_1.train_each_fold
    ```
4. Perform ensemble predictions:
    ```sh
    python -m src.subtask_1.perform_ensemble_predictions
    ```

## Lower Resource Model

If you prefer to use a model with lower resource requirements and more stable training, you can use the `google-bert/bert-base-german-cased` model. For this model, it is recommended to change the learning rate and batch size as follows:

- Learning rate: `4.053266600485604e-05`
- Batch size: `64`

## Datasets

- **GAHD**: Filtered and original datasets available under the Creative Commons Attribution 4.0 International License.
- **GermEval**: Dataset files available under the CC BY-NC-SA 4.0 License.
- **Bajer Danish Misogyny**: Gated dataset available upon request via HuggingFace.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Additional Information

For more details about the competition, visit the [Germeval 2024 - Sexism Detection Subtask](https://ofai.github.io/GermEval2024-GerMS/subtask1.html).

