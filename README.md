# ProbGLC

In this paper, we proposes a Probabilistic Cross-view GeoLoCalization approach, namely ProbGLC, aiming at facilitating generative location awareness for rapid disaster response.

##  Installation

From the github repository:
```bash
conda create -n plonk python=3.10
conda activate plonk
pip install -r requirements.txt
pip install -e .
pip install cartopy

```

## Training

### Downloading our datasets

Download the IAN-Dataset and the Multidisaster dataset and extract them into: ```plonk/plonk/data/```

### Training the model
To train the model, you can use the following command:

```bash
python plonk/train.py exp=iandisaster20_osm mode=traineval experiment_name=My_IAN_Experiment
```
We have provided multiple configs in the configs/exp/ folder.
For the IAN dataset refer to iandisaster20 or 30 depending on the datasplit, the _osm/_yfcc refers to which pre-trained checkpoint is utilized during training. For the Multidisaster dataset use mulitdisaster20_osm.yaml and so on. When a model has been trained use the ```save_localrepo.py```in the plonk/ folder to save the checkpoint to the huggingface format for model loading. 

### Evaluating the model

We provide two Jupyter Notebooks to evaluate the trained checkpoints as well as the zero-shot performance, namely: 
- DisasterPlonk_IAN.ipynb
- DisasterPlonk_MultiDisaster.ipynb

The trained checkpoints have to be placed in the root folder of the repository.