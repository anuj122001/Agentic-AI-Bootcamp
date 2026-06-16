import tiktoken

query = "Serialization and Deserialization"

encoding = tiktoken.get_encoding("o200k_base")

token_ids = encoding.encode(query)
token_pieces = [encoding.decode([tid]) for tid in token_ids]

print(f"Query: {query}")
print(f"Token count: {len(token_ids)}")
print(f"Token IDs: {token_ids}")
print(f"Token pieces: {token_pieces}")
