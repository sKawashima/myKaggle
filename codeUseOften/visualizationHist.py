g = sns.FacetGrid(train_df, col='比較対象')
g.map(plt.hist, '分析対象', bins=20)
