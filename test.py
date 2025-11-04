from pydoc import text
import httpx
from myembedder import MyEmbedder

text = "Xin chao moi nguoi"
# url_1 = "http://101.99.3.94:8080/embed"
# res_1 = httpx.post(url_1, json={"inputs":text}).json()

#print(res_1)
#print(len(res_1))
#print(res_1[0])
#print(len(res_1[0]))

# ====================================

texts = ["hello", "xin chao"]
# # url_2 = "http://101.99.3.94:8080/embed_all"
# # res_2 = httpx.post(url_2,json={"inputs":texts}).json()

# # print(res_2)
# # print(len(res_2))

embedder = MyEmbedder()

# # res_3 = embedder.embed_query(text = text)
# # print(res_3)
# # print(len(res_3))

res_4 = embedder.embed_documents(texts=texts)
#print(res_4)
# # print(len(res_4))

