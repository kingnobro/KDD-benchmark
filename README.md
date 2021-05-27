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

### ideas

#### 1. add features

1. Add feature `publication_year`. It is really time-consuming.

2. Try different ways to calculate text similarity. `Jaro Winkler` wins.

3. Add feature `keyword` of paper titles and keywords. It is also time-consuming.

4. Add feature `journal_conference_year`.

5. Add feature `journal_count`.

6. Try classifier `GradientBoostingRegressor`. Result: `0: 44%`, ` 1: 99%`, ` overall: 40%`

    Modify parameters.

    ```
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    ```

#### 2. select models

Finally, we picked `...`

#### 3. model ensemble

to be finished...

