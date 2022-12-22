# Content-Based Filtering

Content-Based Filtering: This approach uses a series of descriptions of an item in order to recommend additional items with similar properties . The term “content” refers to these descriptions, and in this case manipulated with TF-IDF matrices. The function returns a list with the k-top similarities (cosine similarity).

* Xdata = Dataset Attributes. A 0-1 matrix with the content in columns.

* k = Up to k-top similarities (cosine similarity) that are greater or equal the cut_off value. The default value is 5.

* show_all = Boolean that indicates if the similiarities of each item will be calculated (show_all = True) or for just one item (show_all = False). The default value is True.

* graph = Boolean that indicates if the cosine similarity will be displayed (graph = True) or not (graph = False). The default value is True.

* target = k-top similarities of target item. Only relevant if "show_all = False". The default value is "none".

* cut_off = Value between -1 and 1 that filter similar item according to a certain threshold value. The default value is -0.9999.

# Recommender System Library
Try [pyRecommenderSystem](https://github.com/Valdecy/pyRecommenderSystem): A Recommender System Library
