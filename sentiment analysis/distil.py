"""
This code is based on the supplied code for the sentiment model: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def distil(text):
    print(f"Processing text: {text}")
    
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Tokenized input: {inputs}")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        print(f"Logits: {logits}")
    
    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    
    print(f"Predicted Class ID: {predicted_class_id}")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 50) 

sentences = [
    # Casual language
    "Had a bad day",
    "The company's profits increased significantly last quarter.",
    "Whatever, let's see",
    "Haven't had a worse day",

    # Emojis to text
    "[ROCKET]",
    "[GEM STONE]",
    "[PERSON RAISING BOTH HANDS IN CELEBRATION]",
    "[ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
    "PERSON RAISING BOTH HANDS IN CELEBRATION",
    "person raising both hands in celebration",
    "ROCKET, GEM STONE, PERSON RAISING BOTH HANDS IN CELEBRATION",
    "money, sending message. [ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
    "BREAKING",

    # Standard stock market sentences
    "The stock market surged today, with the S&P 500 reaching an all-time high due to strong corporate earnings.",
    "Investors are advised to diversify their portfolios to minimize risk in volatile market conditions.",
    "Tesla reported a 30% increase in revenue for the third quarter, driven by strong demand for electric vehicles.",
    "The Federal Reserve announced a 0.25% interest rate hike to combat rising inflation, signaling a tightening of monetary policy.",
    "Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security.",

    # Reddit examples
    "I believe right now is one of those rare opportunities that we all can help and do good. Some of these companies like GME, AMC are good companies that's been hit hard by this pandemic. Hedgefunds and Wallstreet just want to short these companies to zero and make millions. I really think right now we have enough support and enough of us to change that direction in history. Wallstreet says  well weak companies need to just go. 10 yrs down the road though I want to be able to watch a movie in a movie theater with my family. If we all buy and hold in what we believe in it gives these companies a second chance and we as a group can stop these companies from being shorted to death and just disappear. Just my 2 cents!",
    "Lets imagine GME has gone to 50,000. The market has gone offerless, when you exercise your option instead of stock you get 'oops, please wait while we connect you to our operator' and your shitty broker is simply ghosting you and turning off the phone line. Even your stupid lawyer refuses the case because supereme court has suspended the constitution for GME stock holders. But lets do some maths. If every person on the sub bought 1 call option strike 100. At price 50k that means the seller is liable for 49900x100x2,000,000= 10trillion. This will destroy not only the dirty dealing worthless billionaires and the shitty market making crooks, it will destroy the entire economy and not a single one of you will get paid because no one has 10 trillion to pay you, and even if they had, they wouldnt, the fed is not going to print 10 T so you can make your gains, no way. The US financial market is a hollow crooked ponzi and has been for years and there is no way they can stand or accept this hit. The real price for GME is infinity but at that price, unfortunately your call option is worth nothing. It will be reneged.",
    "Obviously this GME hype cant go on forever ( Well maybe it can lol Tesla). But what does it mean that a multi billion dollar hedge fund tried shorting GME. Who is the next GME and could this be an indicator that the entire stock market will start to go down very soon because MAJOR money will be getting pulled out because they don’t wanna end up as the next Melvin? Or is this just a lone hiccup in the stonks only go up universe? I feel some type of stock market trap will be upon us in the coming months.",
    "Those whole GME / AMC thing seems awesome. I'm loving it. But I'm not getting the concept of why the short sellers aren't just cutting their losses now while theta is chewing the shit out of them and their losses continue to increase. Do they really think the bottom is going to fall out at the last minute...so much that they would risk billions more? I've gotta be missing something here...what is it?",
    
    # Reddit without stopwords
    "believe right one rare opportunities help good. companies like GME, AMC good companies hit hard pandemic. Hedgefunds Wallstreet want short companies zero make millions. really think right enough support enough change direction history. Wallstreet says well weak companies need go. 10 yrs road though want able watch movie movie theater family. If buy hold believe gives companies second chance group stop companies shorted death disappear. 2 cents!",
    "Lets imagine GME gone 50,000. market gone offerless, exercise option instead stock get 'oops, please wait connect operator' shitty broker simply ghosting turning phone line. Even stupid lawyer refuses case supereme court suspended constitution GME stock holders. But lets maths. If every person sub bought 1 call option strike 100. At price 50k means seller liable 49900x100x2,000,000= 10trillion. destroy dirty dealing worthless billionaires shitty market making crooks, destroy entire economy single get paid one 10 trillion pay even had, wouldnt, fed going print 10 T make gains, way. US financial market hollow crooked ponzi years way stand accept hit. real price GME infinity price, unfortunately call option worth nothing. reneged.",
    "Obviously GME hype cant go forever ( Well maybe lol Tesla). But mean multi billion dollar hedge fund tried shorting GME. Who next GME could indicator entire stock market start go down soon MAJOR money getting pulled dont wanna end next Melvin? Or lone hiccup stonks go universe? feel type stock market trap upon coming months.",
    "Those whole GME / AMC thing seems awesome. loving. But getting concept short sellers cutting losses theta chewing shit losses continue increase. Do really think bottom going fall last minute...so much would risk billions more? got ta missing something... what?"
]

# Process each sentence and display results
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1}/{len(sentences)}:")
    distil(sentence)
