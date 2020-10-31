import streamlit as st
import pandas as pd
import numpy as np
import MeCab
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import altair as alt
import plotly.express as px
import pyfpgrowth
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import urllib.request
import os
from sklearn.feature_extraction.text import CountVectorizer


def sprit_sentence_to_word(name):
    # 辞書登録(mecab-ipadic-neologd)
    option = "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"
    tagger = MeCab.Tagger("-Ochasen " + option)
    open_link = "path to link" + name + ".txt"
    f = open(open_link)
    text = f.read()
    f.close()
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

def analysis():
    print('analysis')
    contents = []
    eng_names = []
    for name in names:
        open_link = "path to link" + name + ".txt"
        # ファイルをオープンする
        test_data = open(open_link, "r")
        # データを書き込む
        contents.append(test_data.read())
        # ファイルクローズ
        test_data.close()
        eng_names.append(name)


    df = pd.DataFrame({'name': eng_names,
                   'text': contents})

    # TF-IDFの計算
    print('IDFの計算')
    # カラム数が1400以上になったため、便宜上TF-IDFの上位50を取得
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=0.1, max_df=0.90, max_features=200)

    # 文章内の全単語のTfidf値を取得
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

    # index 順の単語リスト
    terms = tfidf_vectorizer.get_feature_names()
    tfidfs = tfidf_matrix.toarray()
    # TF-IDF表示
    show_tfidf(terms, tfidfs, eng_names)

    # tsne次元圧縮
    do_tsne(tfidf_matrix, df)



def show_tfidf(terms, tfidfs, eng_names):
    df = pd.DataFrame(
        tfidfs,
        columns=terms,
        index=eng_names)
    st.write(df)

    # df_tfidf = pd.DataFrame(tfidfs[:],columns = ["x"])
    # df_tfidf ["y"] = pd.DataFrame(terms[:])
    # a = alt.Chart(df_tfidf).mark_point().encode(
    #     x="x",
    #     y="y",
    # )
    # st.altair_chart(a, use_container_width=True)

def do_tsne(tfidf_matrix, df):
    print('do_tsne')
    # 2次元に変換　回転数1200
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1200)
    tsne_tfidf = tsne.fit_transform(tfidf_matrix)
    df_tsne = pd.DataFrame(tsne.embedding_[:, 0],columns = ["x"])
    df_tsne["y"] = pd.DataFrame(tsne.embedding_[:, 1])
    df_tsne["name"] = df.name
    st.write(df_tsne)
    c = alt.Chart(df_tsne).mark_point().encode(
        x='x',
        y='y',
    ).mark_text(
        align='left',
        baseline='middle',
        dx=10
    ).encode(
        text='name',
    )
    st.altair_chart(c, use_container_width=True)

def download_stopwords(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        return
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)
    return

@st.cache
def create_stopwords(path):
    stop_words = []
    for w in open(path, "r"):
        w = w.replace('\n','')
        if len(w) > 0:
          stop_words.append(w)
    return stop_words

data_load_state = st.text('Loading data...')
for name in names:
    # 形態素解析
    data = sprit_sentence_to_word(name)
    open_link = "path to link" + name + ".txt"
    with open(open_link, 'w', encoding='utf-8') as f:
        f.write(' '.join(data))

analysis()
data_load_state.text('Loading data...done!')


