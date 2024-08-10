# Quickstart

Welcome to RiteBot! To test and operate the RiteBot on a local host, you can run 
the following line in the command line: `streamlit run app.py`. 

## Downloading Packages

To download all the packages in this repository, please run `pip install -r requirements.txt`.

## Uploading Data

Uploading data into a Pinecone Serverless Spec requires several steps. Your first
step should be creating a [Pinecone Account and API Key](https://docs.pinecone.io/guides/get-started/quickstart).
You can create indexes directly from your laptop once you have created an API key.

In addition, you will need an OpenAI key for embeddings. However, there are alternatives
to OpenAI that you can access such as FastEmbeddings and HuggingFace embeddings (which
are free). 

Once you have your embeddings and Pinecone account set up, you can follow the example in
`data/PineconeUploader.ipynb`. We have also included the `data/oracle.pdf` file, which
is a PDF containing URL links to Oracle Fusion documentations. You can decide which pages/URLs
are revelant for your tasks and then run each cell in the Jupyter Notebook to upload your
data into Pinecone.  

## Report Generation

Several examples have been given in the `system_prompt.txt` that maps user intention
to a specific query. You can add further examples to `system_prompt.txt` to improve the 
precision of the report generation.

Furthermore, reports will be added and deleted for every user message, so make sure to download
the generation before moving to the next user message. 