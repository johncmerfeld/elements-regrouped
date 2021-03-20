import pandas as pdimport numpy as npfrom plotnine import *%matplotlib inline# TODO: consider joining with other dataset to get appearanceelements = pd.read_csv('data/elements.csv')elements.head()elements['group'] = [-1 if (g == '-' or np.isnan(g)) else int(g) for g in elements['Group']]elements['bonding type'] = elements['Type'].astype('category')elements['metal'] = elements['Metal'].astype('category')elements['atomic_number'] = elements['AtomicNumber'].astype(str)top = elements.query('group != -1').copy()bottom = elements.query('group == -1').copy()top['x'] = top['group']top['y'] = top['Period']nrows = 2hshift = 3.5vshift = 3bottom['x'] = np.tile(np.arange(len(bottom)//nrows), nrows) + hshiftbottom['y'] = bottom['Period'] + vshifttile_width = 0.95tile_height = 0.95(ggplot(aes('x', 'y')) + aes(fill='Trendiness') + geom_tile(top, aes(width=tile_width, height=tile_height)) + geom_tile(bottom, aes(width=tile_width, height=tile_height)) + scale_y_reverse() + coord_equal(expand=False)   # new + theme(figure_size=(12, 6))  # new)