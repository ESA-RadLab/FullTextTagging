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

**OBS!** In addition for visualizations Poppler needs to be downloaded from https://github.com/oschwartz10612/poppler-windows/releases/ and the path for it's location needs to be provided in the `constants.py` file as `poppler_path`, eg  `poppler_path = r"..\Release-23.11.0-0\poppler-23.11.0\Library\bin" `

The **GPU** environment will be created later.

### API keys

If using ChatGPT based models save your personal API key to file named `constants.py`
e.g.
`OPENAI_API_KEY = "your personal API key" `

OBS! ChatGPT API charges by use, so it is useful to set limits for maximum charging

## Data
The data can be given as pdf's (less accurate) or as xml files generated with grobid.

1. To generate the xml files first install the Grobid Docker image as https://grobid.readthedocs.io/en/latest/Grobid-docker/.
2. Run the image with `docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.0`
3. Pull the Grobid python API from github
4. Generate the tei.xml output with the Grobid Python API as `grobid_client --input ../input_pdf_folder --output ../tmp/output_xml_folder --teiCoordinates processFulltextDocument`
5. Use the python script to precess the grobit files as `python3 process_grobit_files.py --input_folder path/to/inputfolder --output_folder path/to/outputfolder`

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


## Architecture

**Text extraction** - see ```pre_precessing.py```
- PDFminer.six is used to extract the text from the pdf (assuming it's not an image pdf)
- Format information is used to exclude titles, headers and references

**Text embedding** - see  ```custom_types.py```
- Use OpenAI to get text embedding for each chapter extracted by PDFminer

**Context**
- Get the most relevan context by finding the nearest chapters for the prompt using k-nearest? in the embedding space

----

## Classes

**Paper Class** - see  ```custom_types.py```

**Text Class** - see  ```help_types.py```

**Anwser Class** - see  ```help_types.py```

we use pydantic to create relevant data models:
- Paper
- Prompts
- ...