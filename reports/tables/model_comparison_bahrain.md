# Model Karsilastirma — Bahrain

| Model         |   CV Mean Acc |   CV Std |   CV Precision |   CV Recall |   CV F1 |    AUC |   GS Best Score |   Overfit Gap | Best Params                                                                          |
|:--------------|--------------:|---------:|---------------:|------------:|--------:|-------:|----------------:|--------------:|:-------------------------------------------------------------------------------------|
| Decision Tree |        0.9048 |   0.0673 |         0.7262 |      0.8056 |  0.7498 | 0.8039 |          0.8492 |        0.0952 | {'criterion': 'gini', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2} |
| kNN           |        0.8492 |   0.0112 |         0.4246 |      0.5    |  0.4592 | 0.6176 |          0.8492 |        0.0006 | {'metric': 'euclidean', 'n_neighbors': 5, 'weights': 'uniform'}                      |

## Yorum

- Decision Tree, kNN'e kiyasla belirgin sekilde daha iyi ayirt edici guc gostermektedir (AUC: 0.804 vs 0.618).

**En iyi model:** Decision Tree