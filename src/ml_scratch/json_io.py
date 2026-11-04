"""Lưu/nạp trạng thái model và TF-IDF dạng JSON, không cần train lại."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .classifiers import KNNSpamClassifier, NaiveBayesSpamClassifier
from .data import TfidfVectorizer

def save_json(bundle, path):
    m, v = bundle['model'], bundle['vectorizer']
    state = {'schema_version': 2, 'vectorizer': {'max_features': v.max_features, 'vocabulary': {k:int(i) for k,i in v.vocabulary_.items()}, 'idf': v.idf_.tolist()}}
    if isinstance(m, KNNSpamClassifier):
        # Lưu sparse để không ghi hàng triệu số 0; nạp lại về đúng dense như model gốc.
        x = csr_matrix(m.X_train)
        state['model'] = {'type':'knn', 'k':m.k, 'X_train': {'format':'csr', 'shape':list(x.shape), 'data':x.data.tolist(), 'indices':x.indices.tolist(), 'indptr':x.indptr.tolist()}, 'y_train':m.y_train.tolist()}
    elif isinstance(m, NaiveBayesSpamClassifier):
        state['model'] = {'type':'naive_bayes', 'alpha':m.alpha, 'classes':m.classes.tolist(), 'prior':{str(k):float(val) for k,val in m.prior.items()}, 'word_probability':{str(k):val.tolist() for k,val in m.word_probability.items()}}
    else:
        raise TypeError('Unsupported classifier')
    Path(path).write_text(json.dumps(state,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')

def load_json(path):
    state = json.loads(Path(path).read_text(encoding='utf-8'))
    if state['schema_version'] != 2:
        raise ValueError('Unsupported JSON schema')
    vs = state['vectorizer']
    v = TfidfVectorizer(max_features=vs['max_features'])
    v.vocabulary_ = vs['vocabulary']
    v.idf_ = np.array(vs['idf'], dtype=np.float64)
    ms = state['model']
    if ms['type'] == 'knn':
        m = KNNSpamClassifier(k=ms['k'])
        m.build_model()
        x = ms['X_train']
        m.X_train = csr_matrix((x['data'],x['indices'],x['indptr']),shape=tuple(x['shape'])).toarray()
        m.y_train = np.array(ms['y_train'])
    elif ms['type'] == 'naive_bayes':
        m = NaiveBayesSpamClassifier(alpha=ms['alpha'])
        m.build_model()
        m.classes = np.array(ms['classes'])
        m.prior = ms['prior']
        m.word_probability = {k:np.array(val,dtype=np.float64) for k,val in ms['word_probability'].items()}
    else:
        raise ValueError('Unknown classifier')
    return {'model':m,'vectorizer':v}
