import chromadb

client = chromadb.Client()
col = client.create_collection("test")

col.add(
    documents=["doc1", "doc2"],
    metadatas=[{"name": "G80 EV"}, {"name": "GV80 3.5"}],
    ids=["1", "2"]
)

res = col.get(where={"name": {"$contains": "G80"}})
print("Matches for $contains 'G80':", len(res['ids']))
