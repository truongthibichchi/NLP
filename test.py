from gensim.models import Word2Vec 

retrieve_model = Word2Vec.load("word2vec_2.model")

def similar_products(v, n = 6):
    
    # extract most similar products for the input vector
    ms1 = retrieve_model.similar_by_word("SK0043", topn= n+1)[1:]
    print(ms1)
    ms = retrieve_model.similar_by_vector(v, topn= n+1)[1:]
    print(ms)
    
    # extract name and similarity score of the similar products
    """ new_ms = []
    for j in ms:
        pair = (products_dict[j[0]][0], j[1])
        new_ms.append(pair)
        
    return new_ms  """       

# Let's try out our function by passing the vector of the product '90019A' ('SILVER M.O.P ORBIT BRACELET')
similar_products(retrieve_model['SK0043'], 100)
