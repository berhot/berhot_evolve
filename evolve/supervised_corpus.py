"""Extractive, source-backed QA supervision; no generated answers or model API."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from .plain_corpus import _json
from .owned_corpus import prepare_owned_corpus,read_owned_corpus
from .focus_data import arabic_ratio

FORMAT='berhot_ar_extract_qa_v1'
END='\n<|end|>\n'
FILES={'train':'train.jsonl','val':'validation.jsonl','test':'test.jsonl'}


def question_prompt(question):return 'سؤال: '+question.strip()+'\nجواب: '


def encode_example(example,max_bytes=768):
    prefix=question_prompt(example['question']).encode('utf-8')
    room=max_bytes+1-len(prefix)-len(END.encode())
    if room<32:raise ValueError('QA question leaves no answer space')
    answer=example['answer'].encode('utf-8')[:room].decode('utf-8','ignore').encode('utf-8')
    return prefix,answer+END.encode('utf-8')


def _examples(text,record):
    paragraphs=[part.strip() for part in text.split('\n\n') if part.strip()]
    title=next((p for p in paragraphs[:4] if 8<=len(p)<=100 and arabic_ratio(p)>.5),record['title'])
    title=re.sub(r'^سعوديبيديا\s*[-–:]\s*','',title).split('|')[0].strip()[:100]
    if len(title)<8 or title in {'نظرة عامة','محتوى الصفحة','مشاركة الصفحة'}:return []
    evidence=next((p for p in paragraphs if len(p)>=100 and arabic_ratio(p)>.5 and p!=title),'')
    if not evidence:return []
    # Keep an exact source substring. This is an excerpt target, not a claim
    # of a new independent historical fact or a generated summary.
    answer=evidence[:360]
    if len(evidence)>360:
        boundary=max(answer.rfind('. '),answer.rfind('؟ '),answer.rfind('؛ '))
        answer=answer[:boundary+1] if boundary>=100 else answer[:answer.rfind(' ')]
    start=text.find(answer)
    if start<0 or len(answer)<80:return []
    questions=[title,'ما الذي يذكره المصدر عن '+title+'؟','تحدث عن '+title]
    output=[]
    for question in questions:
        key=hashlib.sha256((question+'\n'+answer).encode()).hexdigest()
        output.append({'id':key,'question':question,'answer':answer,'title':title,
                       'source_url':record['url'],'source_body_sha256':record['clean_sha256'],
                       'approval_parent_sha256':record['approval_parent_sha256'],
                       'answer_sha256':hashlib.sha256(answer.encode()).hexdigest(),
                       'answer_character_start':start,'split':record['split'],
                       'supervision':'exact_source_excerpt','independent_fact_verification':False})
    return output


def prepare_supervised_corpus(store,body_descriptor=None):
    body_descriptor=body_descriptor or prepare_owned_corpus(store)
    parts=read_owned_corpus(body_descriptor)
    directory=Path(body_descriptor['directory'])
    body_records=[json.loads(line) for line in (directory/'documents.jsonl').read_text().splitlines()]
    by_hash={}
    for data in [document for documents in parts.values() for document in documents]:
        by_hash[hashlib.sha256(data).hexdigest()]=data.decode()
    examples={}
    for record in body_records:
        for example in _examples(by_hash[record['clean_sha256']],record):examples.setdefault(example['id'],[]).append(example)
    groups={part:[] for part in FILES};conflicts=[]
    for key,copies in sorted(examples.items()):
        if len({example['split'] for example in copies})>1:
            conflicts.append(key);continue
        groups[copies[0]['split']].append(copies[0])
    if any(not values for values in groups.values()):raise ValueError('QA preparation needs examples in all article splits')
    revision=hashlib.sha256(_json({'format':FORMAT,'body_revision':body_descriptor['dataset_revision_sha256'],
                                  'examples':{part:[row['id'] for row in rows] for part,rows in groups.items()}})).hexdigest()
    base=store.root/'datasets'/'supervised_qa';target=base/revision;base.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.qa-preparing-',dir=base) as temp:
            staging=Path(temp);files={}
            for part,name in FILES.items():
                payload=b''.join((json.dumps(row,ensure_ascii=False)+'\n').encode() for row in groups[part]);files[name]=payload
                files[part+'.txt']=b''.join((question_prompt(row['question'])+row['answer']+END).encode() for row in groups[part])
            manifest={'format':FORMAT,'revision':revision,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                      'body_dataset':body_descriptor,'examples_by_split':{p:len(v) for p,v in groups.items()},
                      'cross_split_exact_duplicates_excluded':len(conflicts),
                      'answers_generated_by_model':False,'independent_fact_verification':False,
                      'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in files.items()}}
            for name,data in files.items():(staging/name).write_bytes(data)
            (staging/'manifest.json').write_bytes(_json(manifest))
            try:staging.rename(target)
            except OSError:
                if not target.exists():raise
    descriptor={'directory':str(target),'manifest':str(target/'manifest.json'),'revision':revision,
                'body_dataset_revision':body_descriptor['dataset_revision_sha256']}
    read_supervised_corpus(descriptor)
    store.put('focus_supervised_qa_latest',descriptor);return descriptor


def read_supervised_corpus(descriptor):
    directory=Path(descriptor['directory']);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('format')!=FORMAT or manifest.get('revision')!=descriptor['revision']:raise ValueError('QA manifest mismatch')
    expected=set(FILES.values())|{part+'.txt' for part in FILES}
    if set(manifest['files'])!=expected:raise ValueError('QA file set mismatch')
    files={}
    for name,entry in manifest['files'].items():
        data=(directory/name).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('QA file hash mismatch')
        files[name]=data
    bodies=read_owned_corpus(manifest['body_dataset'])
    by_hash={hashlib.sha256(data).hexdigest():data.decode() for docs in bodies.values() for data in docs}
    sources={row['clean_sha256']:row for row in
             (json.loads(line) for line in (Path(manifest['body_dataset']['directory'])/'documents.jsonl').read_text().splitlines())}
    result={};seen=set()
    for part,name in FILES.items():
        rows=[]
        for line in files[name].decode().splitlines():
            row=json.loads(line);digest=row['source_body_sha256'];body=by_hash[digest]
            offset=row['answer_character_start'];answer=row['answer']
            if (row['split']!=part or sources[digest]['split']!=part or row['id'] in seen or
                    sources[digest]['url']!=row['source_url'] or sources[digest]['approval_parent_sha256']!=row['approval_parent_sha256'] or
                    body[offset:offset+len(answer)]!=answer or hashlib.sha256(answer.encode()).hexdigest()!=row['answer_sha256']):
                raise ValueError('QA answer provenance or article split mismatch')
            encode_example(row);seen.add(row['id']);rows.append(row)
        result[part]=rows
    return result


def plan_curiosity(store,net,train_examples,count=6):
    """Rank training-only source questions with the local model's answer loss."""
    position=int(store.get('focus_curiosity_position',0) or 0);tasks=[]
    for i in range(min(count,len(train_examples))):
        row=train_examples[(position+i)%len(train_examples)]
        prompt,answer=encode_example(row)
        score=net.qa_loss(prompt,answer)
        tasks.append({'id':row['id'],'question':row['question'],'search_query':row['title'],
                      'source_url':row['source_url'],'priority':score,'status':'pending',
                      'planner':'training_source_question_ranked_by_local_model_uncertainty'})
    store.put('focus_curiosity_position',position+count)
    old=store.get('focus_autonomous_research_tasks') or []
    combined={task['id']:task for task in old+tasks}
    store.put('focus_autonomous_research_tasks',sorted(combined.values(),key=lambda row:-row['priority'])[:100])
    return tasks
