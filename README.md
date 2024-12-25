# GemUI: Automated UI Code Generation via Multimodal Large-Language Models

## Acknowledgement

This project is built upon [Design2Code](https://github.com/NoviScl/Design2Code). We thank the authors for making their work publicly available.


## Original instruction from Design2Code Repo

Quick Links:
[[Dataset]](https://huggingface.co/datasets/SALT-NLP/Design2Code-hf)

![](example.png)



### Set Up

All code is tested on Python 3.11. We recommend using a virtual environment to manage the dependencies.

Clone this repo and install the necessary libraries:

```bash
pip install -e .
```

Taking screenshots and running evaluations also need to install browsers

```bash
playwright install
```

If the above doesn't work, try:

```bash
python3 -m playwright install
```

### Data and Predictions

#### Design2Code 

You can download the full Design2Code testset from this [Google Drive link](https://drive.google.com/file/d/12uRO5EC7hkg6qAOyfJhb4YsrQ_qpL5bt/view?usp=sharing) or access it from the Huggingface dataset [page](https://huggingface.co/datasets/SALT-NLP/Design2Code).

After you unzip it into `testset_final/`, the folder should include 484 pairs of screenshots (`xx.png`) and corresponding HTML code (`xx.html`). We also include the placeholder image file `rick.jpg` which is used in the HTML codes.

#### Design2Code-HARD
You can download the full Design2Code-HARD testset from this [link](https://huggingface.co/datasets/SALT-NLP/Design2Code-HARD/resolve/main/Design2Code-HARD.zip), or you may access it from the Huggingface dataset [page](https://huggingface.co/datasets/SALT-NLP/Design2Code-HARD).

The downloaded folder includes 80 pairs of screenshots (`xx.png`) and corresponding HTML code (`xx.html`). We also include the placeholder image file `rick.jpg` which is used in the HTML codes.

#### Taking Screenshots

In case you want to take screenshots of webpages by yourself, you can do so by running:

```bash
cd Design2Code
python3 data_utils/screenshot.py 
```

Remember to replace the file name or directory in the script with your own. 
### Running Prompting Experiments 

To run prompting experiments, first put your OpenAI / Google Gemini API keys in a file called `api_keys.json` in the root directory. It should look like this:

```json
{
    "organization_id": "",
    "openai_key": "",
    "openai_endpoint": "",
    "gemini_api_key": ""
}
```

Then, to run GPT-4V experiments, run:

```bash
bash prompting/gpt4v.sh
```

To run Gemini Pro Vision experiments, run:

```bash
bash prompting/gemini.sh
```

To run Claude 3.5 Sonnet experiments, run:

```bash
bash prompting/claude3-5.sh
```

The bash scripts include scripts for running Direct Prompting, Text-Augmented Prompting, and Self-Revision Prompting. All prompts are written in `prompting/gpt4v.py` and `prompting/gemini.py`, you can modify it to run your own prompts or develop smarter prompting strategies. We welcome any contributions to this part of the project! 

Also note that we are accessing the OpenAI API from Azure, and you might need some slight modification for directly calling the OpenAI API. 

## Running Automatic Evaluation

You can use the following command to run automatic evaluation:

```bash
python3 metrics/multi_processing_eval.py
```

Note that you need to specify the directories where you store the model predictions in `metrics/multi_processing_eval.py` (starting at line 54), like the following:

```python
test_dirs = {
        "gpt4v_direct_prompting": "../predictions_final/gpt4v_direct_prompting",
        "gemini_direct_prompting": "../predictions_final/gemini_direct_prompting"
}
```

where we assume each directory in the dict contains the predictions of the corresponding model/method (i.e., each directory should contain 484 predicted HTML files for the full test set, or for some subset that you sampled for yourself). The script will compute scores for all automatic metrics for all examples in each directory and store the results in a dictionary, with the following format:

```python
{
    "gpt4v_direct_prompting": {
        "2.html": [0.1, 0.2, ...],
        "6.html": [0.3, 0.4, ...],
        ...
    },
    "gemini_direct_prompting": {
        "2.html": [0.5, 0.6, ...],
        "6.html": [0.7, 0.8, ...],
        ...
    }
}
```

where each list contains the fine-grained breakdown metrics. The script will also print the average scores for each model/method in the end, with the following format:

```
gpt4v_direct_prompting

Block-Match:  0.6240771561959276
Text:  0.9769471025300969
Position:  0.7787072741618328
Color:  0.7068853534416764
CLIP:  0.8924754858016968
--------------------------------

gemini_direct_prompting

Block-Match:  0.6697374012874602
Text:  0.9731735845969769
Position:  0.6502285758036523
Color:  0.8531304981602478
CLIP:  0.8571878373622894
--------------------------------
```

These metrics are also what we reported in the paper. By default, we support multiprocessing to speed up evaluation, you can also manually turn it off by setting `multiprocessing = False` on line 40.
For your reference, it can take up to 1 hour to run the the evaluation on the full testset (for each model/method). 


### Other Functions

- `data_utils` contains various filtering and processing scripts that we used to construct the test data from C4. 


### License

The data, code and model checkpoint are intended and licensed for research use only. Please do not use them for any malicious purposes.

The benchmark is built on top of the C4 dataset, under the ODC Attribution License (ODC-By). 

