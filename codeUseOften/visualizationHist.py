# 1次比較

g = sns.FacetGrid(train_df, col='比較対象')
g.map(plt.hist, '分析対象', bins=20)

# 2次比較

g = sns.FacetGrid(train_df, col='比較対象', row='比較対象')
g.map(plt.hist, '分析対象', bins=20)
g.add_legend()
