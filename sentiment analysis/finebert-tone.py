from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

sentences = ["[ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
             "PERSON RAISING BOTH HANDS IN CELEBRATION",
             "person raising both hands in celebration",
             "ROCKET, GEM STONE, PERSON RAISING BOTH HANDS IN CELEBRATION",
             "Setting aside a portion of your income each month for retirement can significantly improve your long-term financial security.",
             "money, sending message. [ROCKET][GEM STONE][PERSON RAISING BOTH HANDS IN CELEBRATION]",
             "Math Professor Scott Steiner says numbers spell DISASTER Gamestop shorts",
             "Exit system CEO NASDAQ pushed halt trading to give investors chance recalibrate positions. [ SEC investigating, brokers disallowing buying calls. institutions flat admitting change rules bail rich happens us, get well shucks known investing risky! tried cutting avocados coffee, maybe Uber side? may collectively driven enough sentiment wall street make big players go long GME us (we money move stock much alone). didn't hurt wall street whole, funds went others went profited shorts us. media wants pin blame us. crystal clear rigged game now. time build new exchanges can't arbitrarily change rules us. Cr\*\*o version these, maybe repurposed trade stock without government intervention. don't know exactly look like yet, broad next steps see - 1. exit current financial system 2. build new one.",
             "NEW SEC FILING GME! SOMEONE LESS RETARDED PLEASE INTERPRET?",
             "distract GME, thought AMC brothers aware",
             "BREAKING",
             "SHORT STOCK EXPIRATION DATE Hedgefund whales spreading disinfo saying Friday make-or-break $GME. Call options expiring ITM Friday drive price levels maintained, may trigger short squeeze. may Friday, could next week see real squeeze. PANIC SQUEEZE HAPPEN FRIDAY. guaranteed to. thing guaranteed mathematically shorts cover point future. trying get enough people hooked false expectation Friday if/when happen, enough sell panic/despair. PERSON. LIKE STOCK",
             "MOMENT Life fair. mother always told would complain arbitrary treatment. would play rules someone else would ignore them. would win would appeal first authority explanation. Are going let get away this? Life fair. No, not. game game. Always. moment, fascade cracks further. first breach made know, perhaps Socrates, today see thousands. Millions. laughing, luxuries falling disgusting diseased mouths cackled. unmistakable stench derision carried breath. told anyone outside elite class fools even trying. told us naive. needed networks successful. needed polish. needed expertise. needed THEM. game game. Always. longer laughing. odious oeuvre still wafts air. rot, hate, condescention, remains, noxious air betrays new addition. Something together disconcerting. betrays, fear. afraid. be. need inherited resources masked acumen. new day dawns. day make ever slight step towards fear most. even field. Life becoming ever slighty fair. AND. THEY. ARE. SCARED. look us see roughness. look see softness. correct estimation. game game. Always. Fuck street. Fuck street. righteous. blessed Phoebe. started echo time eons come. Mount ride fury thousand rockets universal filament. May wind always back sun upon face. may wings destiny carry aloft, dance stars. GME@everything BB@everything [ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET]",
             "Currently Holding AMC NOK - retarded think move GME today?",
             "nothing say BRUH speechless MOON [ROCKET][ROCKET][ROCKET][GEM STONE][GEM STONE][WAVING HAND SIGN][WAVING HAND SIGN]",
             "need keep movement going, make history! believe right one rare opportunities help good. companies like GME, AMC good companies that's hit hard pandemic. Hedgefunds Wallstreet want short companies zero make millions. really think right enough support enough us change direction history. Wallstreet says well weak companies need go. 10 yrs road though want able watch movie movie theater family. buy hold believe gives companies second chance group stop companies shorted death disappear. 2 cents!",
             "GME Premarket [MAPLE LEAF] Musk approved [VIDEO GAME][OCTAGONAL SIGN][GEM STONE][RAISED HAND]",
             "done GME - $AG $SLV, gentleman's short squeeze, driven macro fundamentals guys champs. GME... would thought bunch crazy retards could reach front page New York Times. done GME, time punish big banks suppressing price silver since Bear Stearns / JPM merge. fucking Bloomberg: [ There's excellent explanation scheme [ think GME squeezed hard? Look happened silver half year ago July: &#x200B; That's one banks getting squeezed silver, cover shorts... that's rich boomers freaked financial instability finally started calling Comex bullshit, taking physically delivery silver... have. imagine 4 million degenerates buying $SLV, forcing trust take delivery physical silver Comex. GME, who's fair price maybe around $5 share. FAIR price silver based historical gold/silver ratio almost surely $50/ounce. short squeeze fantastic success even take silver fair market value. there's $AG. silver starts moving, $AG going go fucking moon. 1) leverage play silver, 2) got SHORTS SQUEEZED OUT. least 23% short float last count: [ **TL;DR: thing gets going, shares calls $SLV $AG will** ***rocket.*** Edit: **[ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET][ROCKET]**"
            ]

# results = nlp(sentences)
# print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative

labels = ["negative", "neutral", "positive"]
for i, sentence in enumerate(sentences):
    predicted_label = nlp(sentence)
    print(f"Predicted label for '{sentence}': {predicted_label} \n")
