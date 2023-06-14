# Optimising a model

As each model comprises of separate components, we optimise each component individually. The components in question are:

- GNN (`graph`)
- Classification head (`clf_head`)
- Keyword pruning threshold (`threshold_experiment`)

All separate components can be optimised using the `--optimize` flag. E.g.:

```commandline
python main.py --optimize 'clf_head'
```

The additional parameters are those set in the settings files.


