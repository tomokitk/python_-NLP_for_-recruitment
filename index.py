import streamlit as st
import pandas as pd
import numpy as np
import MeCab
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer



st.title('tiltle')


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data




    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')



    hist_values = np.histogram(
        data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]


    st.bar_chart(hist_values)

def sprit_sentence_to_noun():
    # mecabの最新版
    option = "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    tagger = MeCab.Tagger("-Ochasen " + option)

    f = open('path to txtfile')
    text = f.read()
    f.close()
    # text = text.split('\n') # 改行で区切る
    res = tagger.parseToNode(text)
    
    # ストップワードリストダウンロード
    path = "stop_words.txt"
    # download_stopwords(path)

    # ストップワードリスト取得
    stop_words = create_stopwords(path)
    print('形態素解析')
    words = []
    while res:
        # ストップワード除去
        if res.surface not in stop_words:
            word = res.surface
            part_of_speech = res.feature.split(",")[0]
            sub_part_of_speech = res.feature.split(",")[1]
            if part_of_speech in ['名詞', '動詞', '形容詞']:
                if sub_part_of_speech not in ['空白', '*']:
                    words.append(word)
        res = res.next
    return words


def filler_word2vec():
    sentences = word2vec.LineSentence('path to txtfile')
    # ベクトル化
    # ソフトマックス関数
    # 学習5epi
    # 出現5単語未満切り捨て
    model = word2vec.Word2Vec(
        sentences,
        sg=1,
        size=100,
        min_count=5,
        window=7,
        hs=1,
        iter=5
    )
    model.save('path to modelfile')

def analysis():
    load_model_path = 'path to modelfile'
    model = word2vec.Word2Vec.load(load_model_path)
    results = model.wv.most_similar(positive=['自立'])
    return results

model = filler_word2vec()
results = analysis()

@st.cache
def create_stopwords(path):
    stop_words = []
    for w in open(path, "r"):
        w = w.replace('\n','')
        if len(w) > 0:
          stop_words.append(w)
    return stop_words


data = sprit_sentence_to_noun()
with open('path to txtfile', 'w', encoding='utf-8') as f:
        f.write(' '.join(data))
model = filler_word2vec()
results = analysis()

df = pd.DataFrame(
    results,
    columns=('類似単語[自立]', '類似度'))
st.write('エンジニアチームにとっての自立とは？',df,'sg=1/size=100/min_count=5/window=20/hs=1/iter=5')