import spacy

nlp = spacy.load('ja_ginza')
doc = nlp('今日は一日良い天気だった。明日も晴天だといいな！')

for sent in doc.sents:
    for token in sent:
        print(token.i, token.orth_, token.lemma_, token.pos_, token.tag_, token.dep_, token.head.i)

    print('---EOS---')
