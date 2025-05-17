# Knowledge Base Preprocessor and Embedder.

*this script will take new documents and convert them into small chunks for emedding conversions or what have you*

## Download virtual environment   
> python3 -m venv .venv  

## Start virtual environment   
> source .venv/bin/activate  

## If first time, then install requirments
> python install_requirements.py  

## **Very Important, this uses openai embedding models so you will be required to use an openai api key, please create an .env file with the line**  
> OPENAI_API_KEY=yourKeyHereWithNoQuotationMarks   


# How to use:  
1) Start virtual environment (see above)
2) Put new document as a txt file into newDocumentsFolder
3) Run the script  
> python preprocess.py
4) You will be asked the name of the collection/knowledge base to use, for example if you did oppenheimer you would only interact with the oppenheimer's  
knowledge base, vice versa for another name  

# How to test:  
1) Run the script  
> python testQuery.py  
2) You will be prompted for the collection name for example, oppenheimer (not case-sensitive, but spelling matters!) 
3) It will generate 2 results that the LLM will use, can be changed to more results, all with a unique chunk id 