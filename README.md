# HERO NAME Recognition

## Describe
In the problem of hero identification through the game league of legends and the requirements of the problem and the data provided. I realize this is a common classification problem and belongs to the type of multi-label classification. The problem belongs to the group of supervised math problems, to handle this problem I use CNN to solve this problem and use the pytorch framework to deploy. Use densenet121 backbone to train the model and use adam algorithm to optimize and use CrossEntropyLoss function to find out the model cost.

Problems exist:
- The topic does not provide train data
- Using a fairly large architecture (Not recommended)
- The model is being overfitting
- Train and evaluate on a single episode. Although sharing data
## Base environment
The environment is only available for Python3.8 and upcoming versions of Python3.9

Following packages are available in requirements.txt
```python
# Install dependencies

pip install -r requirements.txt
```
## Running
usage: Hero Name Recognition [-h] {train,export,infer} 

`
python main --help
`

Training the classification:

```python 
python main.py train --epochs 10 --path_dataset ./test_data/test.txt 
```

Export model from ckpt_path to TorchScript format:

```python
python main.py export --convert_model ./save_model/model_hero_jit.pt
```

Inference with exported model for making prediction. input_path is a directory which contains many images

```python
python main.py infer --torchjit_ck ./save_model/model_hero_jit.pt --folder_img ./test_data/test_images/ --save_file output.txt
```

## Conclusion
This documentation has demonstrated how to use related module.
Before actually start working on anything, please read the whole document first.
If you need any clarifications, please contact me.
Thanks for reading and good luck on improving the model.

## Happy Coding