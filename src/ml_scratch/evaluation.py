"""Biểu đồ, đánh giá độc lập và demo; không chọn tham số theo test."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT,clean_text,load_features
from .classifiers import KNNSpamClassifier,NaiveBayesSpamClassifier
from .formulas import (euclidean_distances, nearest_indices, majority_vote,
    class_prior, laplace_probabilities, log_score, normalized_scores,
    accuracy_from_counts, precision_from_counts, recall_from_counts, f1_from_rates,
    false_positive_rate, roc_curve, roc_auc)
from .json_io import save_json,load_json
FIG=ROOT/'outputs/figures'
REPORT=ROOT/'outputs/reports'

def finish(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

def eda(raw,data):
    counts=pd.DataFrame({'raw':raw.label.value_counts(),'clean':data.label.value_counts()}).fillna(0)
    counts.to_csv(REPORT/'label_distribution.csv')
    data.groupby('label')[['word_count','char_count']].describe().to_csv(REPORT/'numeric_summary.csv')
    fig,ax=plt.subplots(figsize=(7,4.5))
    counts.plot.bar(ax=ax,rot=0)
    ax.set(ylabel='Email count',title='Ham and spam before and after cleaning')
    for container in ax.containers:
        ax.bar_label(container,fmt='%.0f',padding=2)
    finish(fig,'data_labels')
    fig,axes=plt.subplots(2,2,figsize=(10,8))
    for row,column,unit in [(0,'word_count','Words'),(1,'char_count','Characters')]:
        for label in ['ham','spam']:
            vals=data.loc[data.label==label,column]
            axes[row,0].hist(vals,bins=50,alpha=.55,label=label)
        axes[row,0].set(xlabel=unit,ylabel='Email count',title=column+' histogram')
        axes[row,0].legend()
        groups=[data.loc[data.label==label,column].to_numpy() for label in ['ham','spam']]
        axes[row,1].boxplot(groups)
        axes[row,1].set_xticks([1,2],['ham','spam'])
        axes[row,1].set(ylabel=unit,title=column+' by class')
    fig.suptitle('Length by class: full range, including long emails')
    finish(fig,'data_lengths')

def visualize_features(train,test,v):
    counts=pd.DataFrame({'train':train.label.value_counts(),'test':test.label.value_counts()}).T
    fig,ax=plt.subplots(figsize=(7,4));counts.plot.bar(stacked=True,ax=ax,rot=0);ax.set_ylabel('Email count');finish(fig,'training_testing')
    fig,ax=plt.subplots(figsize=(8,4));ix=np.argsort(v.idf_)[-15:];ax.barh(v.get_feature_names_out()[ix],v.idf_[ix]);ax.set_xlabel('IDF (train only)');finish(fig,'feature_idf')

def confusion_counts(y_true, y_pred):
    labels=['ham','spam']
    index={label:i for i,label in enumerate(labels)}
    cm=np.zeros((2,2),dtype=int)
    # Hàng là nhãn thật, cột là nhãn dự đoán: [[TN, FP], [FN, TP]].
    for actual,pred in zip(y_true,y_pred):
        cm[index[actual], index[pred]] += 1
    return cm

def fit_evaluate(name):
    x,z,train,test,v=load_features()
    m=KNNSpamClassifier(5) if name=='knn' else NaiveBayesSpamClassifier(1)
    m.build_model();m.fit(x,train.label)
    save_json({'model':m,'vectorizer':v},ROOT/f'artifacts/{name}.json')
    restored=load_json(ROOT/f'artifacts/{name}.json')
    pred=restored['model'].predict(z);scores=restored['model'].predict_spam_score(z)
    assert np.array_equal(pred,m.predict(z))
    cm=confusion_counts(test.label,pred)
    # ravel theo hàng: TN, FP, FN, TP vì thứ tự ô là ham/ham, ham/spam, spam/ham, spam/spam.
    tn,fp,fn,tp=cm.ravel()
    precision=precision_from_counts(tp,fp);recall=recall_from_counts(tp,fn)
    metrics={'model':name,'accuracy':accuracy_from_counts(tp,tn,len(test)),'precision':precision,'recall':recall,'f1':f1_from_rates(precision,recall),'tpr':recall,'fpr':false_positive_rate(fp,tn),'roc_auc':roc_auc(*roc_curve(test.label=='spam',scores)[:2])}
    pd.DataFrame([metrics]).to_csv(ROOT/f'outputs/{name}_metrics.csv',index=False)
    pd.DataFrame({'source_row':test.index,'label':test.label.values,'prediction':pred,'spam_score':scores}).to_csv(ROOT/f'outputs/{name}_predictions.csv',index=False)
    pd.DataFrame(cm,index=['ham','spam'],columns=['ham','spam']).to_csv(ROOT/f'outputs/{name}_confusion_matrix.csv')
    fpr,tpr,thresholds=roc_curve(test.label=='spam',scores)
    pd.DataFrame({'fpr':fpr,'tpr':tpr,'threshold':thresholds}).to_csv(ROOT/f'outputs/{name}_roc_curve.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].imshow(cm,cmap='Blues')
    axes[0].set_xticks([0,1],['ham','spam']);axes[0].set_yticks([0,1],['ham','spam'])
    axes[0].set(xlabel='Predicted',ylabel='Actual',title=name+' confusion matrix')
    for i in range(2):
        for j in range(2):
            axes[0].text(j,i,str(cm[i,j]),ha='center',va='center')
    axes[1].plot(fpr,tpr,label=f'AUC={metrics["roc_auc"]:.4f}');axes[1].plot([0,1],[0,1],'--');axes[1].set(xlabel='FPR',ylabel='TPR',title=name+' ROC');axes[1].legend();finish(fig,name+'_test')
    print(metrics)
    return metrics

def email_terms(vectorizer, text, row, n=8):
    """Đếm, IDF và TF-IDF đã chuẩn hóa của các từ nặng nhất trong một email thật."""
    values=np.asarray(row.toarray()).ravel()
    counts={}
    for tok in vectorizer.tokens(text):
        j=vectorizer.vocabulary_.get(tok)
        if j is not None:
            counts[j]=counts.get(j,0)+1
    chosen=np.flatnonzero(values)
    chosen=chosen[np.argsort(values[chosen])[::-1]][:n]
    names=vectorizer.get_feature_names_out()
    return pd.DataFrame({'term':names[chosen],'count':[counts.get(int(j),0) for j in chosen],'idf':vectorizer.idf_[chosen],'tfidf':values[chosen]})

def explain_knn(model, row, train):
    """Năm hàng xóm thật của một vector TF-IDF, rồi bỏ phiếu."""
    email=np.asarray(row.toarray()).ravel()
    distances=euclidean_distances(model.X_train, email)
    chosen=nearest_indices(distances, model.k)
    labels=model.y_train[chosen]
    table=pd.DataFrame({'rank':np.arange(1,model.k+1),'source_row':train.index.to_numpy()[chosen],'label':labels,'distance':distances[chosen],'snippet':pd.Series(train.clean_text.to_numpy()[chosen]).str.slice(0,140).to_numpy()})
    return table, majority_vote(labels), float(np.mean(labels=='spam'))

def explain_nb(model, vectorizer, row):
    """Prior, vài từ có TF-IDF cao và điểm log của một email thật."""
    values=np.asarray(row.toarray()).ravel()
    chosen=np.flatnonzero(values)
    chosen=chosen[np.argsort(values[chosen])[::-1]][:8]
    names=vectorizer.get_feature_names_out()
    terms=pd.DataFrame({'term':names[chosen],'tfidf':values[chosen], **{f'P_word_given_{label}':model.word_probability[label][chosen] for label in model.classes}})
    scores=np.array([log_score(row, model.prior[label], model.word_probability[label]) for label in model.classes])
    norm=normalized_scores(scores)
    summary=pd.DataFrame({'class':list(model.classes),'prior':[model.prior[label] for label in model.classes],'log_score':scores,'normalized':norm})
    return terms, summary

def worked_examples():
    """Tính tay KNN 2D và Naive Bayes hai từ bằng formulas.py, rồi lưu bảng số."""
    labels=np.array(['ham','ham','spam','spam','spam'])
    points=np.array([[0,0],[1,0],[4,4],[5,4],[4,5]],dtype=float)
    query=np.array([0.0,1.0])
    distances=euclidean_distances(points,query)
    order=nearest_indices(distances,len(distances))
    selected=order[:3]
    vote=majority_vote(labels[selected])
    rank=np.empty(len(labels),dtype=int)
    rank[order]=np.arange(1,len(labels)+1)
    knn=pd.DataFrame({'point':np.arange(len(labels)),'x1':points[:,0],'x2':points[:,1],'label':labels,'distance':distances,'rank':rank,'selected':np.isin(np.arange(len(labels)),selected)})
    knn=knn.sort_values('rank')
    knn_summary=pd.DataFrame([{'k':3,'vote':vote,'spam_votes':int(np.sum(labels[selected]=='spam')),'spam_score':np.sum(labels[selected]=='spam')/3,'neighbors':', '.join(f'{i}:{labels[i]}' for i in selected)}])
    knn.to_csv(REPORT/'knn_toy_distances.csv',index=False)
    knn_summary.to_csv(REPORT/'knn_toy_summary.csv',index=False)
    docs=np.array([[2,0],[1,0],[0,2],[0,1]],dtype=float)
    doc_labels=np.array(['ham','ham','spam','spam'])
    classes=['ham','spam']
    rows=[];priors={};probs={}
    for label in classes:
        totals=docs[doc_labels==label].sum(axis=0)
        priors[label]=float(class_prior(int(np.sum(doc_labels==label)),len(doc_labels)))
        probs[label]=laplace_probabilities(totals,1)
        for j,word in enumerate(['word_1','word_2']):
            rows.append({'class':label,'word':word,'total':totals[j],'probability':probs[label][j],'prior':priors[label]})
    likelihood=pd.DataFrame(rows)
    email=np.array([1.0,0.0])
    log_values=np.array([log_score(email,priors[label],probs[label]) for label in classes])
    norm=normalized_scores(log_values)
    chosen=classes[int(np.argmax(log_values))]
    scores=pd.DataFrame({'class':classes,'log_score':log_values,'normalized':norm,'predicted':[label==chosen for label in classes]})
    likelihood.to_csv(REPORT/'naive_bayes_toy_likelihood.csv',index=False)
    scores.to_csv(REPORT/'naive_bayes_toy_scores.csv',index=False)
    print(knn.to_string(index=False))
    print(knn_summary.to_string(index=False))
    print(likelihood.to_string(index=False))
    print(scores.to_string(index=False))
    return {'knn_distances':knn,'knn_summary':knn_summary,'nb_likelihood':likelihood,'nb_scores':scores,'query':query}

def examples():
    # Hai chiều và hai từ chỉ để tính tay, không phải mô hình TF-IDF 1.000 chiều.
    tables=worked_examples()
    knn=tables['knn_distances']
    query=tables['query']
    fig,ax=plt.subplots(figsize=(6,4))
    ham=knn[knn.label=='ham'];spam=knn[knn.label=='spam']
    ax.scatter(ham.x1,ham.x2,label='ham');ax.scatter(spam.x1,spam.x2,label='spam')
    ax.scatter(*query,marker='*',s=150,label='query')
    for row in knn[knn.selected].itertuples():
        ax.plot([query[0],row.x1],[query[1],row.x2],'--',color='gray')
    ax.set_title('KNN toy example: K=3, vote ham');ax.legend();finish(fig,'knn_example')
    like=tables['nb_likelihood']
    labels=[f'{row.class_name}: {row.word}' for row in like.rename(columns={'class':'class_name'}).itertuples()]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.bar(labels,like.probability)
    ax.set(ylabel='P(word | class)',title='Naive Bayes toy counts, alpha=1')
    ax.tick_params(axis='x',rotation=20);finish(fig,'naive_bayes_example')
    return tables

def demo():
    emails=['Subject: Team meeting tomorrow. Please review the report.','Subject: Free prize. Claim your money now.']
    rows=[]
    for name in ['knn','naive_bayes']:
        b=load_json(ROOT/f'artifacts/{name}.json');x=b['vectorizer'].transform([clean_text(t) for t in emails]);pred=b['model'].predict(x);score=b['model'].predict_spam_score(x)
        for t,l,s in zip(emails,pred,score):rows.append({'model':name,'text':t,'prediction':l,'spam_score':float(s)})
    pd.DataFrame(rows).to_csv(ROOT/'outputs/reports/demo.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for i,name in enumerate(['knn','naive_bayes']):ax.bar(np.arange(2)+i*.35,[r['spam_score'] for r in rows if r['model']==name],width=.35,label=name)
    ax.set_xticks([.175,1.175],['Meeting email','Prize email']);ax.set(ylim=(0,1.1),ylabel='Spam score',title='Demo only - excluded from test');ax.legend();finish(fig,'demo')
    print(pd.DataFrame(rows).to_string(index=False))

def compare():
    result=pd.concat([pd.read_csv(ROOT/f'outputs/{n}_metrics.csv') for n in ['knn','naive_bayes']],ignore_index=True)
    a=pd.read_csv(ROOT/'outputs/knn_predictions.csv');b=pd.read_csv(ROOT/'outputs/naive_bayes_predictions.csv')
    assert a[['source_row','label']].equals(b[['source_row','label']])
    result.to_csv(ROOT/'outputs/reports/model_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4));result.set_index('model')[['accuracy','precision','recall','f1','roc_auc']].T.plot.bar(ax=ax,rot=0);ax.set(ylim=(0,1.12),ylabel='Score');finish(fig,'evaluation_comparison')
    print(result.to_string(index=False))
