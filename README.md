# FullTextTagging
The goal is to assist screeners with inclusion/exclusion decisions, when screening academic papers for systematic literature review.
This project aims to annotate the data with suitable 'tags', that can further be used to assist the inclusion/exclusion decision.

## Environement
The **CPU** conda environment can be created with command
`conda env create -f cpu-env.yml`
then activate with
`conda activate pytorch-env-cpu`
and if needed the environment can be updated (the environment has to bee active) based on the changed yml file as
`conda env update --file cpu-env.yml --prune`

The **GPU** environment will be created later.

### API keys

If using ChatGPT based models save your personal API key to file named `constants.py`
e.g.
`OPENAI_API_KEY = "your personal API key" `

OBS! ChatGPT API charges by use, so it is useful to set limits for maximum charging

## Data
The data folder includes a small tes set 'demo_data'.
In addition the following file structure is assumed to be inside the folder (to be redefined).
```bash
data
└───demo_data
|   |paper0.pdf
|   |paper1.pdf
|   |...
|   |keys.csv
└───SD
|    |
|    └───test_data
|    |    | ...
|    |
|    └───train_data
|    |    | ...
|    |
|    └───validation_data
|    |    | ...
└───CNS
|    |
|    └───test_data
|    |    | ...
|    |
|    └───train_data
|    |    | ...
|    |
|    └───validation_data
|    |    | ...
```

### Getting the data?
How is this done?

## Architecture

we use pydantic to create relevant data models:
- Paper
- Prompts
- ...