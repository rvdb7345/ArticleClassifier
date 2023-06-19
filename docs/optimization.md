# Optimising a model

As each model comprises of separate components, we optimise each component individually. The components in question are:

- GNN (`graph`)
- Classification head (`clf_head`)
- Keyword pruning threshold (`threshold_optimization`)

All separate components can be optimised using the `--optimize` flag and take three arguments: optimization component,
dataset, gnn_type

E.g.:

```commandline
python main.py --optimize 'clf_head' 'canary' 'GAT'
```

The additional parameters are those set in the default settings files.

There is a manual optimization for the threshold available as well, specified with `threshold_experiment` useful for
evaluating the effect of the threshold for specific ranges.


