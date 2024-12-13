text = """
    Fisher Asset Management LLC lessened its holdings in Novo Nordisk A/S (NYSE:NVO - Free Report) by 0.5% during the third quarter, according to the company in its most recent Form 13F filing with the Securities & Exchange Commission. The institutional investor owned 13,305,474 shares of the company's stock after selling 65,153 shares during the quarter. Fisher Asset Management LLC owned 0.30% of Novo Nordisk A/S worth $1,584,283,000 as of its most recent SEC filing.
  """
# Source: https://www.marketbeat.com/instant-alerts/fisher-asset-management-llc-sells-65153-shares-of-novo-nordisk-as-nysenvo-2024-11-25/

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# spacy.displacy.render(doc, style ='ent')
displacy.serve(doc, style='ent')