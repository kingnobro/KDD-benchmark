# KDD_Benchmark

### baseline

- coauthor_1 + coauthor_2 + RandomForestClassifier: 75.08%
- coauthor_1 + coauthor_2 + stringDistance_1 + stringDistance_2 + RandomForestClassifier: 93.5442%

### ideas

1. Add feature `publication_year`. It is really time-consuming.
2. Try different ways to calculate text similarity. `Jaro Winkler` wins.
3. Add feature `keyword` of paper titles and keywords. It is algo time-consuming.

