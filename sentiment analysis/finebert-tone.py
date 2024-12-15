from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

sentences = ["there is a shortage of capital, and we need extra financing",  
             "growth is strong and we have plenty of liquidity", 
             "there are doubts about our finances", 
             "profits are flat",
             "Had a bad day",
             "The company's profits increased significantly last quarter.",
             "Whatever, let's see",
             "Haven't had a worse day",
             "The stock market surged today, with the S&P 500 reaching an all-time high due to strong corporate earnings.",
             "Tesla reported a 30% increase in revenue for the third quarter, driven by strong demand for electric vehicles.",
             "Investors are advised to diversify their portfolios to minimize risk in volatile market conditions.",
             "The Federal Reserve announced a 0.25% interest rate hike to combat rising inflation, signaling a tightening of monetary policy.",
             "Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security."
        ]
results = nlp(sentences)
print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
