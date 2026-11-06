"""First-party article-body extraction and derived, approval-anchored datasets."""
from html.parser import HTMLParser
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile

from .open_corpus import normalize_web_text
from .focus_data import get_profile, topic_relevance
from .training_approval import training_condition
from .plain_corpus import _atomic_write, _json

VERSION="owned_article_body_v1"
VOID={"br","hr","img","input","meta","link","wbr","source","area","embed","param","track","col","base"}
SKIP={"script","style","head","nav","footer","aside","button","svg","noscript","iframe","select","textarea"}
NOISE={"toc","toc-mobile","table-of-contents","article-details","article-details-actions-container",
       "social-share","share","sharing","breadcrumbs","breadcrumb","related-articles","related-content",
       "recommendations","pagination","cookie-banner"}
BODY={"article-body","article-content","article-text","article__content","entry-content","post-content"}
LABELS={"جدول المحتويات","مشاركة الصفحة","صورة الصفحة","محتوى الصفحة","نسخ الرابط","شارك المقالة"}


class ArticleBody(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.stack=[];self.regions=[];self.fallback=[]

    def handle_starttag(self,tag,attrs):
        props=dict(attrs);tokens=set(props.get('class','').split())|{props.get('id','')}
        skip=(tag in SKIP or bool(tokens&NOISE) or props.get('aria-hidden')=='true' or
              (tag=='form' and props.get('id','').lower() not in ('aspnetform','form1')))
        frame={'tag':tag,'skip':skip,'region':None}
        rank=100 if props.get('itemprop')=='articleBody' or tokens&BODY else 90 if tag=='article' else 70 if tag=='main' or props.get('role')=='main' or props.get('id')=='main-content' else 0
        if rank and not skip:
            frame['region']={'rank':rank,'parts':[]};self.regions.append(frame['region'])
        if tag not in VOID:self.stack.append(frame)
        if tag in ('p','h1','h2','h3','h4','li','br','div','section'):self._append('\n')

    def handle_endtag(self,tag):
        if tag in ('p','h1','h2','h3','h4','li','div','section'):self._append('\n')
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i]['tag']==tag:self.stack=self.stack[:i];break

    def _append(self,text):
        if any(f['skip'] for f in self.stack):return
        # Paragraph/headline text gives a conservative fallback for older sites
        # with no semantic article/main container. Navigation link lists stay out.
        if any(f['tag'] in ('p','h1','h2','h3','h4') for f in self.stack):self.fallback.append(text)
        for frame in self.stack:
            if frame['region'] is not None:frame['region']['parts'].append(text)

    def handle_data(self,data):self._append(data)

    def result(self):
        choices=[]
        for region in self.regions+[{'rank':0,'parts':self.fallback}]:
            blocks=[];seen=set()
            for part in ''.join(region['parts']).split('\n'):
                part=normalize_web_text(part)
                if not part or part in LABELS or part in seen:continue
                seen.add(part);blocks.append(part)
            text='\n\n'.join(blocks)
            if len(text)>=200:choices.append((region['rank'],len(text),text))
        return max(choices,key=lambda item:item[:2])[2] if choices else ''


def article_body(html):
    parser=ArticleBody();parser.feed(html);return parser.result()


def prepare_owned_corpus(store, profile="saudi-history-ar"):
    rows=store.db.execute(
        "SELECT s.*,t.clean_path FROM web_corpus_sources s JOIN web_corpus_texts t ON t.sha256=s.sha256 "
        "JOIN focus_archived_sources f ON f.url=s.url WHERE f.profile=? AND "+training_condition('s')+
        " ORDER BY s.host,s.url",(profile,)).fetchall()
    assignments=dict(store.get('focus_article_split_'+profile) or {})
    bindings=dict(store.get('focus_owned_url_splits_'+profile) or {})
    previous=store.get('owned_corpus_latest_'+profile) or {}
    if previous.get('directory') and (Path(previous['directory'])/'documents.jsonl').is_file():
        for line in (Path(previous['directory'])/'documents.jsonl').read_text().splitlines():
            document=json.loads(line);bindings.setdefault(document['url'],document['split'])
    fingerprint=hashlib.sha256(_json({'version':VERSION,'sources':[
        [r['url'],r['sha256'],r['raw_html_sha256'],bindings.get(r['url']) or assignments.get(r['sha256'])] for r in rows]})).hexdigest()
    cached=store.get('owned_corpus_latest_'+profile) or {}
    if cached.get('source_fingerprint')==fingerprint:
        read_owned_corpus(cached);store.put('focus_plain_dataset_latest_'+profile,cached)
        store.put('focus_owned_url_splits_'+profile,bindings);return cached
    config=get_profile(profile);documents=[];rejected=[];groups={};removed=0
    raw_root=(store.root/'web_corpus'/'raw').resolve();clean_root=(store.root/'web_corpus'/'clean').resolve()
    for row in rows:
        raw=Path(row['raw_path']).resolve();clean=Path(row['clean_path']).resolve()
        if not raw.is_relative_to(raw_root) or not clean.is_relative_to(clean_root):raise ValueError('Archived source escapes corpus')
        if hashlib.sha256(clean.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Approved source text changed')
        with gzip.open(raw,'rb') as stream:payload=stream.read(2_000_001)
        if len(payload)>2_000_000 or hashlib.sha256(payload).hexdigest()!=row['raw_html_sha256']:raise ValueError('Archived HTML changed')
        html=payload.decode('utf-8');body=article_body(html);quality=topic_relevance(body,row['title'],config)
        if len(body)<400 or not quality['language_ok'] or not quality['topical']:
            rejected.append({'url':row['url'],'reason':'short_non_arabic_or_off_topic_article_body'});continue
        data=body.encode('utf-8');digest=hashlib.sha256(data).hexdigest()
        # Derivation never invents a new approval. Its parent archived digest
        # passed the shared exact-snapshot approval predicate above.
        split=bindings.get(row['url']) or assignments.get(row['sha256'])
        if split is None:
            bucket=int(row['sha256'][:8],16)%10;split='val' if bucket==0 else 'test' if bucket==1 else 'train'
        bindings[row['url']]=split
        record={'url':row['url'],'host':row['host'],'title':row['title'],
                'clean_sha256':digest,'approval_parent_sha256':row['sha256'],
                'raw_html_sha256':row['raw_html_sha256'],'split':split,'data':data}
        groups.setdefault(digest,[]).append(record)
        removed+=clean.read_text(encoding='utf-8').count('جدول المحتويات')-body.count('جدول المحتويات')
    conflicts=[]
    for digest,copies in groups.items():
        if len({r['split'] for r in copies})>1:
            conflicts.append({'sha256':digest,'urls':[r['url'] for r in copies]});continue
        documents.append(copies[0])
    documents.sort(key=lambda r:r['url'])
    counts={part:sum(r['split']==part for r in documents) for part in ('train','val','test')}
    if any(counts[part]<1 for part in counts):raise ValueError('Body dataset requires nonempty train/validation/test documents')
    revision=hashlib.sha256(_json({'preprocessing':VERSION,'documents':[
        [r['url'],r['clean_sha256'],r['split'],r['approval_parent_sha256']] for r in documents]})).hexdigest()
    base=store.root/'datasets'/'focus_plain';directory=base/revision;base.mkdir(parents=True,exist_ok=True)
    if not directory.exists():
        with tempfile.TemporaryDirectory(prefix='.owned-preparing-',dir=base) as temp:
            staging=Path(temp);parts={part:bytearray() for part in counts};corpus=bytearray();records=[]
            for document in documents:
                row={k:v for k,v in document.items() if k!='data'};data=document['data'];part=row['split']
                row.update(offset=len(parts[part]),length=len(data),corpus_offset=len(corpus))
                records.append(row);parts[part].extend(data+b'\n\n');corpus.extend(data+b'\n\n')
            files={"train.txt":bytes(parts['train']),"validation.txt":bytes(parts['val']),"test.txt":bytes(parts['test']),
                   "corpus.txt":bytes(corpus),"documents.jsonl":b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in records)}
            manifest={'format':'berhot_owned_plain_v1','profile':profile,'preprocessing_version':VERSION,
                      'dataset_revision_sha256':revision,'source_fingerprint':fingerprint,
                      'created_at_utc':datetime.now(timezone.utc).isoformat(),
                      'documents_by_split':counts,'split_method':'parent_article_assignments_preserved',
                      'toc_occurrences_removed':removed,'cross_split_clean_duplicates_excluded':conflicts,
                      'excluded_sources':rejected,'approval_bound_to':'original_archived_parent_snapshot',
                      'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}}
            for name,data in files.items():(staging/name).write_bytes(data)
            (staging/'manifest.json').write_bytes(_json(manifest));staging.rename(directory)
    descriptor={'directory':str(directory),'manifest':str(directory/'manifest.json'),
                'dataset_revision_sha256':revision,'source_fingerprint':fingerprint}
    read_owned_corpus(descriptor)
    from .dataset_catalog import register_dataset
    register_dataset(store,descriptor);store.put('owned_corpus_latest_'+profile,descriptor)
    store.put('focus_plain_dataset_latest_'+profile,descriptor)
    store.put('focus_owned_url_splits_'+profile,bindings);return descriptor


def read_owned_corpus(descriptor):
    directory=Path(descriptor['directory']);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('format')!='berhot_owned_plain_v1' or manifest.get('dataset_revision_sha256')!=descriptor['dataset_revision_sha256']:
        raise ValueError('Owned corpus manifest mismatch')
    required={'train.txt','validation.txt','test.txt','corpus.txt','documents.jsonl'}
    if set(manifest['files'])!=required:raise ValueError('Owned corpus file set mismatch')
    files={}
    for name,entry in manifest['files'].items():
        data=(directory/name).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Owned corpus SHA-256 mismatch')
        files[name]=data
    splits={part:[] for part in ('train','val','test')};seen=set()
    mapping={'train':'train.txt','val':'validation.txt','test':'test.txt'}
    for line in files['documents.jsonl'].decode().splitlines():
        row=json.loads(line);part=row['split'];data=files[mapping[part]][row['offset']:row['offset']+row['length']]
        if row['clean_sha256'] in seen or hashlib.sha256(data).hexdigest()!=row['clean_sha256']:
            raise ValueError('Derived article identity or partition mismatch')
        if files['corpus.txt'][row['corpus_offset']:row['corpus_offset']+row['length']]!=data:raise ValueError('Combined corpus differs')
        seen.add(row['clean_sha256']);splits[part].append(data)
    return splits
