
# ✨ WEAIBot
## Paper Link
https://www.sciencedirect.com/science/article/pii/S235271102400400X

## 🏁 An end-to-end open source AI assistant for the WEAI helpdes.

The code in this repository allows for deployment of a production-ready AI assistant intended to support and answer queries from users of the Women's Empowerment in Agriculture Index (WEAI) tool. The assistant is based entirely on open source tools, including Meta's Llama 3 large language model (LLM), Langchain for deployment of a retrieval augmented generation (RAG) and agentic system, and Hugging Face spaces and inference endpoints for deployment of the chatbot.



## 🎯 Getting started
### 🚀 Installation
```
git clone https://github.com/n3lly43/weaibot.git
```
```bash
pip install -r requirements.txt
```
## Run
```bash
python app.py
```
## In Colab
```Python 
from agents import support_agent, contact_agent

question = "what are some resources where I can read up on women empowerment"

# result from agent handling queries about resources and documents
support_result = support_agent.invoke({"messages": [{"role": "user", "content": question}]})

# result from agent handling contact and information sharing requests
contact_result = contact_agent.invoke({"messages": [{"role": "user", "content": question}]})
```
```Python
support_result['messages'][-1].pretty_print()

"""
================================== Ai Message ==================================

The International Food Policy Research Institute (IFPRI) has several resources available on women's empowerment in agriculture. Some notable ones include:

1. The Women's Empowerment in Agriculture Index (WEAI) - a comprehensive tool to measure women's empowerment in agriculture. You can find more information on the WEAI website.
2. "The Oxford Handbook of Food, Politics, and Society" (2018) edited by Ronit Y. Gordon, which features a chapter on women's empowerment and agriculture.
3. "Women's Empowerment in Agriculture: What Role for Food Security in South Asia?" (2020) by Hazel Malapit, et al., published in the journal Food Security.
4. "Empowering Women in Agriculture: A Key to Food Security" (2019) by Agnes Quisumbing, et al., published in the IFPRI Discussion Paper series.
5. The IFPRI blog series on "Women's Empowerment and Agriculture" which features articles and research findings on the topic.

You can also explore the IFPRI website, which has a dedicated section on "Gender and Women's Empowerment" with various publications, briefs, and other resources.
"""
```

```Python
contact_result['messages'][-1].pretty_print()
"""
================================== Ai Message ==================================

For information on women's empowerment, I recommend visiting the Women's Empowerment and Awareness (WEAI) website. You can also reach out to the WEAI team directly via email to inquire about available resources and publications. 

Additionally, you may want to explore other reputable sources such as the United Nations Women website, the World Bank's Open Knowledge Repository, and academic journals focused on gender studies. 

Please note that I don't have have direct links to provide, but you can search for the WEAI website and other recommended sources online. If you need further assistance or have specific questions, feel free to ask.
"""
```
## 🎯 Contributing

Contributions are always welcome!
