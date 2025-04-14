import faiss
import torch


database_size = 100000
num_queries = 1000

db_vector = torch.randn((database_size,10))
query_vector = torch.randn((num_queries,10))

# Build an index와 index에 벡터 더하기
dimension = 10
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(db_vector.numpy())

print(faiss_index.ntotal)
print(faiss_index.is_trained)

print("db_vector[:5]:", db_vector[:5].numpy())
k = 4 # k-th nearest
Distance, Index = faiss_index.search(db_vector[:5].numpy(), k) # sanity check
print(f'Index: {Index}')
print(f'Distance: {Distance}')