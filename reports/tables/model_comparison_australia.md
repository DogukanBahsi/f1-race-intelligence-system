# Model Karsilastirma — Australia

| Model         |   CV Mean Acc |   CV Std |   CV Precision |   CV Recall |   CV F1 |    AUC |   GS Best Score |   Overfit Gap | Best Params                                                                          |
|:--------------|--------------:|---------:|---------------:|------------:|--------:|-------:|----------------:|--------------:|:-------------------------------------------------------------------------------------|
| Decision Tree |        0.6333 |   0.3055 |         0.5167 |         0.7 |  0.5667 | 0.6515 |          0.7778 |        0.3667 | {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} |
| kNN           |        0.7667 |   0.2906 |         0.6833 |         0.8 |  0.7167 | 0.8258 |          0.7778 |        0.1454 | {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'uniform'}                      |

## Yorum

- kNN, Decision Tree'ye kiyasla daha iyi AUC elde etmistir (0.826 vs 0.651).
- Decision Tree'de asiri ogrenme riski vardir (egitim-test farki: 0.367).

**En iyi model:** kNN