import pandas as pd
import numpy as np

from plotnine import *

%matplotlib inline

# TODO: consider joining with other dataset to get appearance
elements = pd.read_csv('data/elements.csv')
elements.head()

elements['group'] = [-1 if (g == '-' or np.isnan(g)) else int(g) for g in elements['Group']]
elements['bonding type'] = elements['Type'].astype('category')
elements['metal'] = elements['Metal'].astype('category')
elements['atomic_number'] = elements['AtomicNumber'].astype(str)

top = elements.query('group != -1').copy()
bottom = elements.query('group == -1').copy()

top['x'] = top['group']
top['y'] = top['Period']

nrows = 2
hshift = 3.5
vshift = 3
bottom['x'] = np.tile(np.arange(len(bottom)//nrows), nrows) + hshift
bottom['y'] = bottom['Period'] + vshift

tile_width = 0.95
tile_height = 0.95

(ggplot(aes('x', 'y'))
 + aes(fill='Trendiness')
 + geom_tile(top, aes(width=tile_width, height=tile_height))
 + geom_tile(bottom, aes(width=tile_width, height=tile_height))
 + scale_y_reverse()
 + coord_equal(expand=False)   # new
 + theme(figure_size=(12, 6))  # new
)




# e.g.

# e.g.
elements['SwedishnessStr'] = elements['Swedishness'].astype(str)

# need to run 'top' and 'bottom' before this
(ggplot(aes('x', 'y'))
 + aes(fill='SwedishnessStr')
 + geom_tile(top, aes(width=tile_width, height=tile_height))
 + geom_tile(bottom, aes(width=tile_width, height=tile_height))
 + scale_y_reverse()
 + coord_equal(expand=False)   # new
 + theme(figure_size=(12, 6))  # new
)




import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(sparse_output=True)

kmeans5 = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=42
)


edf = elements[['Trendiness', 'Swedishness', 'Usefulness', 'Visual', 'Namesake']]
kmeans5.fit(edf)
kmeans.cluster_centers_ # to get the cluster characteristics
elements['clusters5'] = kmeans5.labels_

elements['cluster5s'] = elements['clusters5'].astype(str)

# one-hot encode 
elements = elements.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(elements.pop('cluster5s')),
                index=elements.index,
                columns=mlb.classes_))

# stringify, e.g,
elements['0s'] = elements['0'].astype(str)

# do ggplots with 0s, 1s, etc. to get the individual groups 


# final 
(ggplot(aes('x', 'y'))
 + aes(fill='cluster5s')
 + geom_tile(top, aes(width=tile_width, height=tile_height))
 + geom_tile(bottom, aes(width=tile_width, height=tile_height))
 + scale_y_reverse()
 + coord_equal(expand=False)   # new
 + theme(figure_size=(12, 6), legend_title=element_text(size=26), legend_text=element_text(size=16), panel_grid=element_blank(), panel_background = element_rect(fill='black', colour='black'))  
 + scale_fill_manual(labels=['Etymologist\'s Delight', 'Nordic Glory','Juggernauts','Bottom Tier', 'Pure Sex'], values=['#FFB1A0', '#F4A0FF', '#FFDDA0', '#A0BDFF', '#A0FFA0'])
 + labs(fill="Group")
)