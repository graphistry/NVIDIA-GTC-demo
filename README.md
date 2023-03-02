# NVIDIA-GTC-demo

Write your first graph neural network, complete with automatic feature engineering, visualization, and deployment, in this lab using popular open source libraries: PyGraphistry[AI], NVIDIA RAPIDS (cuDF, cuGraph, cuML), and GPU neural network ecosystems (DGL, PyG, TF).

Graph neural networks are in a watershed moment, going from a method for Ph.D. teams to a tool that regular operational data teams can quickly and reliably deploy. We'll focus on PyGraphistry[AI], which enables streamlined graph AI workflows over the emerging GPU graph AI ecosystem.

In the spirit of PyGraphistry[AI], this session is tailored for operational data teams needing easy streamlined automatic graph AI capabilities. Examples will span problems like spotting fraud in user clickstreams, detecting hackers in identity data, and making pricing predictions in large supply chain data. Here we will focus on detecting unusual login events, a common use case in Cybersecurity.  We'll provide data from these use cases, the training environment, and a walk through usage.

## Outline

Part I: 
A. Introduction to Graph Thinking
We will provide an overview of UMAP and how it can be used to reduce false positives on a Red Team Hunt.  
To get us started with graph thinking, we will “Embedding all the things” and generating correlation IDs from login behaviors -- this will act both as an anomaly detector as well as a fast way to flag suspicious activity given previously known context.

B. Winning US-CYBERCOM
As a bonus we show how PYGraphistry[ai] won a US-CYBERCOM competition using this technology.

C. Graph Thinking with GNNs
GNNs are beating SOTA benchmarks and encode graph data for use in deep learning. Here we will encode the raw data and make a knowledge graph, allowing us to score the probability seeing a given source, relationship, and destination for different computers logon types. This can be extended to click streams, product recommendations, and supply chain modeling. 

```
# load the data 
edf = pd.read_csv(edges.csv)
g = graphistry.edges(edf, src, dst)

# train the GNN
g2 = g.embed(relation='relationship_column_of_interest', **kwargs)

# score relevant entities of interest
g_anomalous = g2.predict_links(source=['ip_address_1', 'user_id_3'], 
                  relation=['attempt_logon', 'phishing', ..], 
                  destination=['user_id_1', 'active_directory', ..], 
                  anomalous=True,
                  threshold=0.05)
                  
g_anomalous.plot()  # see the unusual activity
```

Part II: Building an Always-On Production Pipeline with Morpheus

A. Introduction to the Production Pipeline
Scaling Successful PoC's to production can be challenging. Here we introduce NVIDIA Morpheus and build a production ready pipeline.

B. NVIDIA Stack and the Production Pipeline
Discuss the role of the NVIDIA stack in the production pipeline
Provide an overview of how the NVIDIA stack can be used to optimize the production pipeline
