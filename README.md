# AdaTrans: improved bioactive molecular binding prediction based on adapting transferable representations

Predicting the molecular binding of bioactive molecules is crucial for drug discovery. Deep learning techniques are widely used to facilitate the identification of active candidate drugs for target proteins (DTI) and to predict their binding affinities (DTA). Previous studies have rarely considered DTI and DTA simultaneously, although they share similar interaction patterns. They differ mainly in their emphasis on drug unveiling, which poses challenges when integrating the two types of tasks. In this study, we propose AdaTrans, an adaptive deep transfer learning framework which utilises residue embedding to represent proteins and molecular linear symbols to represent drugs. We employed bilinear attention graphs to align transferable representations from DTI, effectively supplementing the micro-attributes into DTA. AdaTrans not only explores drug commonalities through similar mechanisms but also captures individualised drug characteristics for precise treatment. The experimental results indicate that AdaTrans achieves superior performance compared to six leading-edge baseline models across various metrics. Moreover, attention visualisation deepens our understanding of biological explanations by delineating key regions within proteins and drug molecules.



## The environment of AdaTrans
```
python==3.11.6
numpy==1.16.4
dgl==1.1.2+cu117
dgllife==0.3.2
pandas==2.1.1
rdkit==2023.9.1
torch==2.1.0+cu121
scikit-learn==1.3.2


```

## Dataset description
In this paper, three benchmark datasets are used, i.e., BindingDB, Davis and KIBA. Our data is uploaded on the [zenodo](https://zenodo.org/records/11197301) link. Download ```data.zip``` into the AdaTrans-main file folder. Then, unzip the compressed data to get a folder with name ``` data ``` for use. The three settings, i.e. (SD) interactions between new drugs and known proteins, (ST) interactions between known drugs and new proteins, (SP) new interactions between drug-protein pairs.


## Run the AdaTrans for protein-ligand binding affinity task
By default, you can run our model using Davis dataset with:
```sh
 TrainAndTest2.py
```

# Acknowledgments
The authors sincerely hope to receive any suggestions from you!

