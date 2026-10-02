"""Segment the included Mall Customers CSV with deterministic K-means (NumPy only)."""
from pathlib import Path
import csv, json
from itertools import permutations
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
DATA=OUT/'Mall_Customers.csv'

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    rows=[]
    with DATA.open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            rows.append({'customer_id':r['CustomerID'],'gender':r['Gender'],'age':float(r['Age']),
                         'annual_income_k':float(r['Annual Income (k$)']),
                         'spending_score':float(r['Spending Score (1-100)'])})
    if len(rows)<6: raise ValueError('Need at least six complete customer records.')
    if any(not np.isfinite(v) for r in rows for v in [r['age'],r['annual_income_k'],r['spending_score']]):
        raise ValueError('Input data contains missing or invalid numeric values.')
    X=np.array([[r['annual_income_k'],r['spending_score']] for r in rows],float)
    return rows,X

def kmeans(X,k,seed=27):
    rng=np.random.default_rng(seed); centers=[X[rng.integers(len(X))]]
    for _ in range(1,k):
        d=np.min(((X[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2),axis=1)
        centers.append(X[rng.choice(len(X),p=d/d.sum())])
    centers=np.array(centers)
    for _ in range(200):
        labels=((X[:,None,:]-centers[None,:,:])**2).sum(axis=2).argmin(axis=1)
        updated=np.array([X[labels==j].mean(axis=0) if np.any(labels==j) else X[rng.integers(len(X))] for j in range(k)])
        if np.allclose(updated,centers): break
        centers=updated
    return labels,centers,float(((X-centers[labels])**2).sum())

def silhouette(X,labels):
    from numpy.linalg import norm
    D=norm(X[:,None,:]-X[None,:,:],axis=2); clusters=np.unique(labels); vals=[]
    for i in range(len(X)):
        same=np.where(labels==labels[i])[0]; same=same[same!=i]
        a=D[i,same].mean() if len(same) else 0
        b=min(D[i,labels==c].mean() for c in clusters if c!=labels[i])
        vals.append((b-a)/max(a,b) if max(a,b)>0 else 0)
    return float(np.mean(vals))

def bar_svg(title,labels,values,color='#176b64'):
    W,H=900,410; left,right,top,bottom=245,60,82,56; pw=W-left-right; ph=H-top-bottom
    vmax=max(values)*1.2; rh=ph/len(labels)
    s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">','<rect width="100%" height="100%" fill="#fbfaf7"/>',f'<text x="36" y="48" font-family="Arial" font-size="25" font-weight="bold" fill="#142f2b">{title}</text>']
    for i,(label,value) in enumerate(zip(labels,values)):
        y=top+i*rh+rh*.18; bh=rh*.62; bw=pw*value/vmax
        s += [f'<text x="{left-14}" y="{y+bh*.72}" text-anchor="end" font-family="Arial" font-size="15" fill="#253b38">{label}</text>',f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bh:.1f}" rx="7" fill="{color}"/>',f'<text x="{left+bw+10:.1f}" y="{y+bh*.72:.1f}" font-family="Arial" font-size="15" fill="#253b38">{value:g}</text>']
    s += [f'<text x="{left}" y="{H-16}" font-family="Arial" font-size="12" fill="#697572">Mall Customers public sample dataset • 200 records</text>','</svg>']
    return '\n'.join(s)

rows,X=load_data()
features=['annual_income_k','spending_score']
mu=X.mean(axis=0); sd=X.std(axis=0); Z=(X-mu)/sd
scores=[]
for k in range(2,8):
    labels_k,centers_k,inertia=kmeans(Z,k)
    scores.append({'k':k,'inertia':round(inertia,3),'silhouette':round(silhouette(Z,labels_k),4)})
K=5
labels,centers,_=kmeans(Z,K)
# Match cluster centers to interpretable low/mid/high income and spending patterns.
prototypes=np.array([[30,25],[30,75],[60,50],[90,25],[90,75]],float)
proto_z=(prototypes-mu)/sd
perm=min(permutations(range(K)),key=lambda p:sum(((centers[j]-proto_z[p[j]])**2).sum() for j in range(K)))
names=['Budget-conscious','Aspiring spenders','Mid-market regulars','Careful high earners','High-value enthusiasts']
recommendations={
 'Budget-conscious':'Test affordable bundles and value-led offers; measure margin as well as response.',
 'Aspiring spenders':'Test relevant cross-sells and accessible premium options; avoid assuming high spending is recurring.',
 'Mid-market regulars':'Use broad assortment and loyalty tests; look for opportunities to increase engagement.',
 'Careful high earners':'Test personalized recommendations and premium discovery without relying on discounting.',
 'High-value enthusiasts':'Prioritize curated launches and loyalty recognition; monitor retention and basket mix.'}
profiles=[]; assigned=[]
for j in range(K):
    mask=labels==j; name=names[perm[j]]; members=[rows[i] for i in np.where(mask)[0]]
    p={'segment':name,'customers':int(mask.sum()),'share_pct':round(float(mask.mean()*100),1),
       'age':round(float(np.mean([r['age'] for r in members])),1),
       'annual_income_k':round(float(np.mean([r['annual_income_k'] for r in members])),1),
       'spending_score':round(float(np.mean([r['spending_score'] for r in members])),1),
       'female_pct':round(float(np.mean([r['gender'].lower()=='female' for r in members])*100),1)}
    profiles.append(p)
    for i in np.where(mask)[0]: assigned.append((i,name))
profiles.sort(key=lambda p:names.index(p['segment']))
segment_by_row=dict(assigned)
with (OUT/'segmented_customers.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['customer_id','gender','age','annual_income_k','spending_score','segment'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for i,r in enumerate(rows): w.writerow({**r,'segment':segment_by_row[i]})
with (OUT/'segment_profiles.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['segment','customers','share_pct','age','annual_income_k','spending_score','female_pct'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(profiles)
(OUT/'model_selection.json').write_text(json.dumps({'dataset':DATA.name,'rows':len(rows),'clustering_features':features,'standardization':'z-score','candidate_k':scores,'selected_k':K,'selection_note':'Selected k=5 to show interpretable income/spending patterns; compare diagnostics and business usefulness before applying elsewhere.'},indent=2),encoding='utf-8')
(OUT/'segment_sizes.svg').write_text(bar_svg('Mall customer segments • size',[p['segment'] for p in profiles],[p['customers'] for p in profiles]),encoding='utf-8')
(OUT/'segment_spend.svg').write_text(bar_svg('Average spending score by segment',[p['segment'] for p in profiles],[p['spending_score'] for p in profiles],'#b2653c'),encoding='utf-8')
with (OUT/'segment_report.md').open('w',encoding='utf-8') as f:
    f.write('# Mall customer segmentation results\n\n')
    f.write(f'**Dataset:** Mall Customers public sample, {len(rows)} customer records. **Method:** z-score standardization and K-means with k={K}, using annual income and spending score. Age and gender are used to describe the resulting groups, not to form the clusters. Candidate k=2–7 diagnostics are in `model_selection.json`. This dataset does not include purchase history, so the spending score is a provided index, not actual transaction data.\n\n')
    f.write('| Segment | Customers | Share | Avg age | Avg income (k$) | Avg spending score | Female |\n|---|---:|---:|---:|---:|---:|---:|\n')
    for p in profiles:f.write(f"| {p['segment']} | {p['customers']} | {p['share_pct']}% | {p['age']} | {p['annual_income_k']} | {p['spending_score']} | {p['female_pct']}% |\n")
    f.write('\n## Possible actions to validate\n\n')
    for p in profiles:f.write(f"- **{p['segment']} ({p['share_pct']}%):** {recommendations[p['segment']]}\n")
    f.write('\n## Files\n\n- `Mall_Customers.csv`: downloaded public sample input.\n- `segmented_customers.csv`: input records with assigned segment.\n- `segment_profiles.csv`: segment sizes and average characteristics.\n- `model_selection.json`: inertia and silhouette diagnostics for candidate k.\n- `segment_sizes.svg`, `segment_spend.svg`: charts.\n')
print(f"Analyzed {len(rows)} customer records; segments:",[(p['segment'],p['customers']) for p in profiles])
