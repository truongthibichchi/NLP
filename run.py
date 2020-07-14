import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from gensim.models import Word2Vec 
import matplotlib.pyplot as plt
# %matplotlib inline

import warnings;
warnings.filterwarnings('ignore')

df = pd.read_excel('skills.xlsx')
df.head()
df.shape

# check for missing values
df.isnull().sum()
# remove missing values
df.dropna(inplace=True)

# again check missing values
df.isnull().sum()

"""
Data Preparation
"""

# convert the StockCode to string datatype.
df['skill_id']= df['skill_id'].astype(str)

# check out the number of unique customers in our dataset.
customers = df["group_id"].unique().tolist()

# shuffle customer ID's
random.shuffle(customers)

# extract 90% of customer ID's
customers_train = [customers[i] for i in range(round(0.9*len(customers)))]

# split data into train and validation set
train_df = df[df['group_id'].isin(customers_train)]
print("train_df")
print(train_df)

validation_df = df[~df['group_id'].isin(customers_train)]
print("validation_df")
print(validation_df)
# list to capture purchase history of the customers
purchases_train = []

# populate the list with the product codes
# tqdm: print status and percent
for i in tqdm(customers_train):
    temp = train_df[train_df["group_id"] == i]["skill_id"].tolist()
    purchases_train.append(temp)

# list to capture purchase history of the customers
purchases_val = []

# populate the list with the product codes
for i in tqdm(validation_df['group_id'].unique()):
    temp = validation_df[validation_df["group_id"] == i]["skill_id"].tolist()
    purchases_val.append(temp)

"""
Build word2vec Embeddings for Products
"""

# train word2vec model
model = Word2Vec(window = 20, sg = 1, hs = 0,
                 negative = 3, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)

model.build_vocab(purchases_train, progress_per=200)

model.train(purchases_train, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

# save word2vec model
model.save("word2vec_2.model")
retrieve_model = Word2Vec.load("word2vec_2.model")
print(retrieve_model)
# As we do not plan to train the model any further, we are calling init_sims(), which will make the model much more memory-efficient.

model.init_sims(replace=True)
# print(model)


# Now we will extract the vectors of all the words in our vocabulary and store it in one place for easy access.
# extract all vectors
X = model[model.wv.vocab]

X.shape
"""
Visualize word2vec Embeddings
It is always quite helpful to visualize the embeddings that you have created. Over here we have 100 dimensional embeddings. We can't even visualize 4 dimensions let alone 100. Therefore, we are going to reduce the dimensions of the product embeddings from 100 to 2 by using the UMAP algorithm, it is used for dimensionality reduction.
"""

import umap.umap_ as umap

cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                              n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(10,9))
plt.scatter(cluster_embedding[:, 0], cluster_embedding[:, 1], s=3, cmap='Spectral')

"""
Start Recommending Products.
Let's first create a product-ID and product-description dictionary to easily map a product's description to its ID and vice versa.
"""

products = train_df[["skill_id", "skill_name"]]

# remove duplicates
products.drop_duplicates(inplace=True, subset='skill_id', keep="last")

# create product-ID and product-description dictionary
products_dict = products.groupby('skill_id')['skill_name'].apply(list).to_dict()

# test the dictionary
products_dict['SK0022']

# I have defined the function below. It will take a product's vector (n) as input and return top 6 similar products.

def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms = retrieve_model.similar_by_vector(v, topn= n+1)[1:]
    print(ms)
    
    # extract name and similarity score of the similar products
    new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms        

# Let's try out our function by passing the vector of the product '90019A' ('SILVER M.O.P ORBIT BRACELET')

print(similar_products(retrieve_model['SK0043'], 100))

"""
The results are pretty relevant and match well with the input product. However, this output is based on the vector of a single product only. What if we want recommend a user products based on the multiple purchases he or she has made in the past?

One simple solution is to take average of all the vectors of the products he has bought so far and use this resultant vector to find similar products. For that we will use the function below that takes in a list of product ID's and gives out a 100 dimensional vector which is mean of vectors of the products in the input list.
"""

def aggregate_vectors(products):
    product_vec = []
    for i in products:
        try:
            product_vec.append(model[i])
        except KeyError:
            continue
        
    return np.mean(product_vec, axis=0)

# If you can recall, we have already created a separate list of purchase sequences for validation purpose. Now let's make use of that.

# The length of the first list of products purchased by a user is 314. We will pass this products' sequence of the validation set to the function aggregate_vectors.
len(purchases_val[0])

aggregate_vectors(purchases_val[0]).shape

# Well, the function has returned an array of 100 dimension. It means the function is working fine. Now we can use this result to get the most similar products. Let's do it.
# print(similar_products(aggregate_vectors(purchases_val[0])))

"""
output:
[('RED RETROSPOT PICNIC BAG', 0.6443557739257812),
 ('LUNCH BAG RED RETROSPOT', 0.6422858834266663),
 ('JUMBO BAG PINK POLKADOT', 0.6313408017158508),
 ('JUMBO STORAGE BAG SKULLS', 0.6284171938896179),
 ('SET/5 RED RETROSPOT LID GLASS BOWLS', 0.6260786056518555),
 ('JUMBO STORAGE BAG SUKI', 0.6257367134094238)]
"""

"""
As it turns out, our system has recommended 6 products based on the entire purchase history of a user. Moreover, if you want to get products suggestions based on the last few purchases only then also you can use the same set of functions.
Below I am giving only the last 10 products purchased as input.
"""

print(similar_products(aggregate_vectors(purchases_val[0][-10:])))

"""
output:
[('PARISIENNE KEY CABINET ', 0.6371047496795654),
 ('FRENCH ENAMEL CANDLEHOLDER', 0.6213719844818115),
 ('RETRO "TEA FOR ONE" ', 0.6032429933547974),
 ('PARISIENNE JEWELLERY DRAWER ', 0.5835957527160645),
 ('SMALL CERAMIC TOP STORAGE JAR ', 0.5755683183670044),
 ('VINTAGE ZINC WATERING CAN', 0.5748642683029175)]
"""