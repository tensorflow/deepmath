examples@10 was extracted from the full data set with

    TODO(geoffreyi): Adapt from old test_data/README.md once ported.

vocab generated with

    TODO(geoffreyi): Adapt from old test_data/README.md once ported.

examples-train@10 and examples-eval@10 were generated from examples@10 with

    TODO(geoffreyi): Adapt from old test_data/README.md once ported.

Graphs generated with

    inference_graph --hparams=model=model_definition_cnn_flat3,seed=7 \
      --output ~/tmp/cnn-graph.meta

    inference_graph \
      --hparams=model=tree_rnn,cell=rnn-relu,layers=2,embedding_size=15,hidden_size=33,seed=7 \
      --output ~/tmp/tree-graph.meta
