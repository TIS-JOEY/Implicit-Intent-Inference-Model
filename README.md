# App2Vec_Python
App2Vec訓練接口(含ANN近似最近鄰訓練、AF親和力傳播訓練...)

本接口提供App2Vec訓練接口，你可以跳過繁瑣的準備資料階段，方便進行App2Vec訓練。
另外，我們亦提供後續的ANN近似最近鄰搜索與AF親和力傳播訓練來進一步處理。

# App2Vec
App2Vec的模型架構是一個淺層的神經網路，有CBOW與Skip-Gram模式可以選擇。在此接口中我們使用gensim函式庫，其可以方便我們進行app2vec進行訓練，且可自由進行模型細節的配置。
