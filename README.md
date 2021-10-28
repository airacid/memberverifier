# MemberVerifier

## Requirements
- Ubuntu=16.04
- Tensorflow=1.15

## How to train Member Verifier A and B
- Prepare face images of Member A, B and Non-registered people
- Download pretrained Face Feature Extractor model
> <a href="https://drive.google.com/file/d/1cwQf6IN3d86kYVQfLuoifGMnFWo-Gbir/view?usp=sharing">Feature Extractor Model</a>
- All images should be 160x160x3
- Extract face feature vectors using FeatureExtraction.py
> Example) python FeatureExtraction.py --pretrained_model=./FE_freeze.pb --path_imgs=./A_imgs --res_name=./A
- Train Member Verifier using extracted feature vectors from A, B and Non-registered face images
> Example) python Train_MV.py --members=./A.npz --other_members=./B.npz --non_members=./Non.npz --output_model=./MV_A.pb

## Evaluation
- Using obtained models of Member A and B, compute accuracy on Test images
> Example) python Test_MV.py --MV_A=./MV_A.pb --MV_B=./MV_B.pb --test_imgs=./Test