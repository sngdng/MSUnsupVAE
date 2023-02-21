# Learning from Multiple Sources for Data-to-Text and Text-to-Data (MSUnsupVAE)

**Learning from Multiple Sources for Data-to-Text and Text-to-Data**<br>
Song Duong, Alberto Lumbreras, Mike Gartrell, Patrick Gallinari</br>

*Soon to be published in AISTATS 2023*

**Abstract:** *Data-to-text (D2T) and text-to-data (T2D) are dual tasks that convert structured data, such as graphs or tables into fluent text, and vice versa. These tasks are usually handled separately and use corpora extracted from a single source. Current systems leverage pre-trained language models fine-tuned on D2T or T2D tasks. This approach has two main limitations: first, a separate system has to be tuned for each task and source; second, learning is limited by the scarcity of available corpora. This paper considers a more general scenario where data are available from multiple heterogeneous sources. Each source, with its specific data format and semantic domain, provides a non-parallel corpus of text and structured data. We introduce a variational auto-encoder model with disentangled style and content variables that allows us to represent the diversity that stems from multiple sources of text and data. Our model is designed to handle the tasks of D2T and T2D jointly. We evaluate our model on several datasets, and show that by learning from multiple sources, our model closes the performance gap with its supervised single-source counterpart and outperforms it in some cases.*

## Installation

### Python environment
Experiments are done with python 3.6.8.
Create a virtualenv and install the `requirements.txt` with pip

```
pip install --upgrade pip
pip install -r requirements.txt
```

In order to be able to run and import code from the `d2t` directory, run:
```
pip install -e .
```

### Evaluation scripts

To install METEOR:
```
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz
mv meteor-1.5 d2t/eval/metrics/
rm meteor-1.5.tar.gz
```
Make sure that Java JRE is installed on your machine
```
#### on ubuntu
sudo apt-get install openjdk-8-jre

#### on fedora, oracle linux, etc
yum install java-1.8.0-openjdk
```
To run evaluation:
```
### Evaluate quality metrics
python evaluate.py -c <path_to_config_file> -w <path_to_weight_file> -sx <number of seeds for style>
```
and 
```
### Evaluate diversity metrics
python diversity_evaluate.py -c <path_to_config_file> -w <path_to_weight_file>
```
(see conf/ to see some examples)

### Training
To train the model, you need to configure accelerate by running `accelerate config`. Then:
```
accelerate launch main.py -c <path_to_train_config_file>
```
This would enable training on all GPUs on your machine (see conf/ to see some examples).

## Acknowledgements
We would like to express our sincere thanks to *Alexandre Thomas* whose work served as the foundation for this paper.