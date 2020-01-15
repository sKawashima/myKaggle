train_df[["test", "target"]].groupby(['test'], as_index=False).mean().sort_values(by='target', ascending=False)
