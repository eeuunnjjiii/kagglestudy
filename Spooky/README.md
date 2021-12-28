# 자연어처리

## 1.[nltk vs spaCy](https://yujuwon.tistory.com/entry/spaCy-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0#:~:text=nltk%EC%9D%98%20%EA%B2%BD%EC%9A%B0%20%EB%AC%B8%EC%9E%90%EC%97%B4%EC%9D%84,%ED%95%98%EA%B2%8C%20%EB%90%98%EC%96%B4%20%EC%9E%88%EB%8A%94%20%EA%B2%83%20%EA%B0%99%EB%8B%A4.])
- nltk는 문자열을 처리하고 결과값도 문자열을 리턴하는 반면 spaCy는 문자열 입력시 객체를 리턴
- spaCy는 문서 구성요소를 다양한 구조에 나누어 저장하는 대신 요소를 색인화하고 검색정보를 간단히 저장하는 라이브러리 -> spacy가 더 빠름
- ![image](https://user-images.githubusercontent.com/75970111/147556951-c56f0099-c033-43bd-8787-1ef82aefe5c2.png)
- Sentence Tokenization은 nltk가 빠르고, Word Tokenization과 Part-of-Speech Tagging의 경우 SpaCy가 빠름
- Spacy는 Lemmatization만 제공, Stemming은 nltk 사용

### 1) Tokenization
```python
# nltk
import nltk
text_list = nltk.word_tokenize(text)

# spacy
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

for token in doc:
  print(token.text)
```

### 2) Stopword Removal
```python
# nltk
stopwords = nltk.corpus.stopwords.words('english')
text_cleaned = [word for word in text_list if word.lower() not in stopwords]

# spacy
tokenizer = Tokenizer(nlp.vocab)
stopwords = nlp.Defaults.stop_words

tokens = []
for doc in tokenizer.pipe(text):
  doc_tokens = []
  for token in doc:
    if (token.is_stop == False) & (token.is_punct == False)"
      doc_tokens.append(token.text.lower())
  tokens.append(doc_tokens)
```

### 3) Stemming and Lemmatization
- Stemming : 어간추출 (-ing,-s 등을 제거한 형태로 변환)
- Lemmatizaion : 표제어추출 (기본 사전형 단어 형태로 변환)

```python
# stemming
stemmer = nltk.stem.PorterStemmer()
stemmer.stem(text)

# Lemmatization - nltk
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
lemm.lemmatize(text)

# Lemmatization - spaCy
doc = nlp(text)
for token in doc:
  print(token.lemma_)
```
