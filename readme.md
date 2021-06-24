# Product Detection (Shopee Code League 2020)

Source code for Shopee Code League - Product Detection Challenge (Rank 4 in Private, Rank 6 in Public under a team named "Citebok") [(Kaggle Link)](https://www.kaggle.com/c/shopee-product-detection-open).

## Motivation
The task was to identify the class of the product based on provided train data with labels (supervised). 

To solve the problem, we tried:
1. multiple publicly available pretrain models in torchvision.models and Efficient Nets ([Tan and Le. ICML 2019](https://arxiv.org/abs/1905.11946)) (in [model.py](model.py)).
2. several loss functions (in [losses.py](losses.py))
3. several image augmentation techniques (in [data.py](data.py), [randaug.py](randaug.py))
4. contrastive learning in self-supervised manner ([Khosla et al. NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), [Chen et al. ICML 2020](https://arxiv.org/pdf/2002.05709.pdf)) (in [train_con.py](train_con.py))
5. transfer learning regularization and freezing (in [train.py](train.py))
6. fixing train-test resolution discrepancy ([Touvron et al. NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/d03a857a23b5285736c4d55e0bb067c8-Abstract.html)).

## Run

To run:
1. Prepare dataset in *label*/*image_file* structure.
2. Set correct paths in [config.py](config.py).
3. Use [split.py](split.py) to leave some train data for validation purpose.
4. (optionally, run [train_con.py](train_con.py)) Run [train.py](train.py) for finetuning pretrained model in supervised manner. See arguments/options inside the file.
5. Run [test.py](test.py) to obtain the test labels.

To run the code with best performance setting, see [run.sh](run.sh).

## Results
We achieved the best performance with EffNet B5 with fixing resolution 456->600.

In the experiments, we found that:
1. Using a good pretrained model pretrained on high-resolution images, e.g., EffNet B5-B7, could improve the performance significantly, compared to other modifications.
2. Cross entropy achieved the best performance among loss functions. Some losses can drastically decrease the performance.
3. Using augmentation technique has very little effect to the performance (less than 2%)
4. Contrastive learning provided small increase in performance, but took much longer time for training.
5. Transfer learning regularization and freezing yielded no improvement.
6. Simply fixing train-test resolution discrepancy can impressively increase up to 5% performance.
