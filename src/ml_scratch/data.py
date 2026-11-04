"""Đầu vào/đầu ra chuẩn cho các bước collection, EDA và feature engineering."""
from pathlib import Path
import re, json
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags, save_npz, load_npz
from .formulas import smooth_idf
from .stopwords import ENGLISH_STOP_WORDS
ROOT=Path(__file__).resolve().parents[2]

def clean_text(text):
    # Bỏ thẻ HTML và đưa về chữ thường.
    text=re.sub(r'<[^>]*>', ' ', str(text).lower())
    # Ký tự ngoài chữ, số, khoảng trắng đổi thành khoảng trắng, rồi gộp khoảng trắng thừa.
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', text).split())

def collection():
    raw=pd.read_csv(ROOT/'data/raw/spam_ham_dataset.csv')
    assert {'text','label'}.issubset(raw.columns)
    assert set(raw.label.dropna()).issubset({'ham','spam'})
    report={'source':'Enron1 / Kaggle venky73/spam-mails-dataset', 'rows':len(raw),'columns':raw.columns.tolist(),'missing':raw.isna().sum().to_dict(),'labels':raw.label.value_counts().to_dict()}
    (ROOT/'outputs/reports/collection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)
    return raw

def prepare():
    raw=collection();data=raw[['text','label']].dropna().copy()
    data.index.name='source_row'
    data['clean_text']=data.text.apply(clean_text)
    data=data[data.clean_text!='']
    if data.groupby('clean_text').label.nunique().max()>1: raise ValueError('Nội dung trùng khác nhãn')
    data=data.drop_duplicates('clean_text')
    data['word_count']=data.clean_text.str.split().str.len()
    data['char_count']=data.clean_text.str.len()
    data.to_csv(ROOT/'data/processed/clean_emails.csv')
    return raw,data

def load_clean():
    return pd.read_csv(ROOT/'data/processed/clean_emails.csv').set_index('source_row')

def stratified_split(data, test_size=0.2, seed=42):
    """Trong mỗi lớp, xáo với seed rồi lấy round(test_size * n) dòng làm test."""
    rng=np.random.default_rng(seed)
    train_parts,test_parts=[],[]
    for label in ['ham','spam']:
        idx=data.index[data.label==label].to_numpy().copy()
        rng.shuffle(idx)
        # round(0,2 * số dòng của lớp): ham và spam bị lấy test riêng, không trộn trước.
        n_test=int(round(len(idx)*test_size))
        test_parts.append(data.loc[idx[:n_test]])
        train_parts.append(data.loc[idx[n_test:]])
    train=pd.concat(train_parts).iloc[rng.permutation(sum(len(p) for p in train_parts))]
    test=pd.concat(test_parts).iloc[rng.permutation(sum(len(p) for p in test_parts))]
    return train,test

class TfidfVectorizer:
    """TF thô nhân IDF, rồi chia L2 từng dòng. Từ vựng chỉ học trên tập được fit."""

    def __init__(self, max_features=1000):
        self.max_features=max_features
        self.vocabulary_=None
        self.idf_=None

    def tokens(self, text):
        return [tok for tok in str(text).split() if len(tok)>1 and tok not in ENGLISH_STOP_WORDS]

    def fit_transform(self, texts):
        rows=[self.tokens(text) for text in texts]
        # totals: tổng số lần từ xuất hiện trên train. df: số thư train có từ đó.
        totals,df={},{}
        for tokens in rows:
            seen=set()
            for tok in tokens:
                totals[tok]=totals.get(tok,0)+1
                if tok not in seen:
                    df[tok]=df.get(tok,0)+1
                    seen.add(tok)
        # Giữ max_features từ nhiều nhất. Hòa số lần thì xếp theo bảng chữ cái.
        ranked=sorted(totals, key=lambda t: (-totals[t], t))[:self.max_features]
        self.vocabulary_={term:i for i,term in enumerate(ranked)}
        # IDF chỉ dùng df của train. len(rows) là N trong công thức smooth_idf.
        self.idf_=smooth_idf(np.array([df[term] for term in ranked], dtype=float), len(rows))
        return self._weight(rows)

    def transform(self, texts):
        if self.vocabulary_ is None:
            raise ValueError('Hãy fit trước khi transform.')
        return self._weight([self.tokens(text) for text in texts])

    def _weight(self, rows):
        indptr,indices,data=[0],[],[]
        for tokens in rows:
            # Đếm số lần từng từ trong từ vựng. Từ không có trong vocabulary bị bỏ.
            counts={}
            for tok in tokens:
                j=self.vocabulary_.get(tok)
                if j is not None:
                    counts[j]=counts.get(j,0)+1
            for j,count in sorted(counts.items()):
                indices.append(j);data.append(float(count))
            indptr.append(len(indices))
        counts=csr_matrix((data,indices,indptr), shape=(len(rows), len(self.vocabulary_)))
        # TF thô nhân IDF của đúng cột đó.
        weighted=counts @ diags(self.idf_)
        # Chuẩn L2: chia mỗi dòng cho căn của tổng bình phương các chiều.
        squared=np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel()
        norms=np.sqrt(squared)
        # Dòng toàn số 0 giữ nguyên, không chia cho 0.
        norms[norms==0]=1
        return diags(1/norms) @ weighted

    def get_feature_names_out(self):
        names=['']*len(self.vocabulary_)
        for term,index in self.vocabulary_.items():
            names[index]=term
        return np.array(names)

def save_features(train, test, x, y, vectorizer):
    for name,frame,matrix in [('train',train,x),('test',test,y)]:
        frame.to_csv(ROOT/f'data/processed/{name}.csv')
        save_npz(ROOT/f'data/processed/X_{name}.npz', matrix)
    names=vectorizer.get_feature_names_out()
    pd.DataFrame({'feature_index':range(len(names)),'term':names,'idf':vectorizer.idf_}).to_csv(ROOT/'data/processed/vocabulary.csv', index=False)
    state={'max_features':vectorizer.max_features,'vocabulary':{k:int(i) for k,i in vectorizer.vocabulary_.items()},'idf':vectorizer.idf_.tolist()}
    (ROOT/'data/processed/vectorizer.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

def features():
    data=load_clean()
    train,test=stratified_split(data)
    assert not set(train.clean_text)&set(test.clean_text)
    vectorizer=TfidfVectorizer(max_features=1000)
    x=vectorizer.fit_transform(train.clean_text)
    y=vectorizer.transform(test.clean_text)
    save_features(train, test, x, y, vectorizer)
    print('Train:', x.shape, 'Test:', y.shape)
    return x,y,train,test,vectorizer

def vectorizer_from_state(state):
    vectorizer=TfidfVectorizer(max_features=state['max_features'])
    vectorizer.vocabulary_=state['vocabulary']
    vectorizer.idf_=np.array(state['idf'], dtype=float)
    return vectorizer

def load_features():
    train=pd.read_csv(ROOT/'data/processed/train.csv').set_index('source_row')
    test=pd.read_csv(ROOT/'data/processed/test.csv').set_index('source_row')
    state=json.loads((ROOT/'data/processed/vectorizer.json').read_text(encoding='utf-8'))
    return load_npz(ROOT/'data/processed/X_train.npz'), load_npz(ROOT/'data/processed/X_test.npz'), train, test, vectorizer_from_state(state)
