@echo off
python train_srresnet.py
python eval.py
python eval_fgsm.py
python eval_pgd.py