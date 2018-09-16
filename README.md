# What's Cooking

* Our Kaggle competition repo for [What's Cooking](https://www.kaggle.com/c/whats-cooking-kernels-only)

## Statistics
* #Train: 39774
* #Test: 9944
* #Class: 20
* #Train ingredients: 6714
* #Test ingredients: 4484
* Train, test ingredient overlaps: 4061 (non-overlap 4484-4061=423)

## TODOs
- [ ] Tf-idf baseline
- [ ] Dataset bias
- [ ] 2-gram

## Performance
On dev:
* Baseline: 0.6959975864843122
* Direct tf-idf: 0.7338093322606597
* Binary tf-idf: 0.7389380530973452
* SVC C=1: 0.6562751407884151
* SVC C=200: 0.7578439259855189
* SVC C=300: 0.7687047465808527
* SVC C=500: 0.7763475462590507
* SVC C=1000: 0.7824818986323411
* SVC C=1000, balanced: 0.7539219629927595
* SVC C=100, balanced: 0.7543242156074015
* SVC C=100, coef0=1, balanced: 0.7389380530973452
* SVC C=100, coef0=1, class_weight=None: 0.7543242156074015
* SVC gamma=1: 0.8110418342719228

