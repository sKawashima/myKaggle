g = sns.FacetGrid(train_df, col='比較対象')
g.map(sns.pointplot, '比較対象', '分析対象', '比較対象')
g.add_legend()
