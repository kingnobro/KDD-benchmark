# KDD_Benchmark

### how to run

```bash
python3 trainer.py
```

you may need to modify the following line in `trainer.py` based on your python environment.

```python
cmd = "/usr/local/bin/python3 evalution.py %s %s" % (gold_file, pred_file)
```

### baseline

- coauthor_1 + coauthor_2 + RandomForestClassifier: 75.08%
- coauthor_1 + coauthor_2 + stringDistance_1 + stringDistance_2 + RandomForestClassifier: 93.5442%
- Actually,  the demo provided by my teacher performs really well :-)
- 2021/05/27: After adding some features, the accuracy reaches 95.1%.

### ideas

#### 1. add features

1. Add feature `publication_year`. **Time-consuming**.

2. Try different ways to calculate text similarity. `Jaro Winkler` wins.

3. Add feature `keyword` of paper titles and keywords. **Time-consuming**.

4. Add feature `journal_conference_year`. **Good**.

5. Add feature `journal_count`. **Slow**.

6. Try classifier `GradientBoostingRegressor`. Result: `0: 44%`, ` 1: 99%`, ` overall: 40%`

    Modify parameters. **Nothing changed**.

    ```
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    ```

7. Add feature `affiliation_count`. **Nothing changed**.

#### 2. select models

Finally, we picked `...`

#### 3. model ensemble

to be finished...

