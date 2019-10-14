# ZOADAM

To download the inception model:

```
python3 setup_inception.py
```

To prepare the ImageNet dataset, download and unzip the following archive:

[ImageNet Test Set](http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/img.tar.gz)


and put the `imgs` folder in `../imagesnetdata`. This path can be changed
in `setup_inception.py`.

To train cifar model:
run 
```
python3 train_model.py
```

Universal attack:

run 
```
python3 universal.py 
```
You can change methods or any hyperparameter in args.

Single image attack on cifar:

run 
```
python3 Main_exp1.py
```

Attack on imagenet for multiple images:

run 
```
python3 imagenet_blackBox_attack2.py
```
