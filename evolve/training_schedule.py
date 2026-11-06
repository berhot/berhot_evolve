"""Resumable article-interleaved windows, without crossing document boundaries."""
import random
import hashlib


def adapt_dataset_cursor(documents,cursor,revision,old_documents=None,seed=42):
    """Carry attempted per-document coverage forward when new pages arrive."""
    if cursor.get('dataset_revision_sha256')==revision:return dict(cursor)
    prior=dict(cursor);offset_by_hash={}
    if old_documents:
        offsets=prior.get('document_offsets')
        if offsets is None:
            offsets=[0]*len(old_documents)
            order=list(range(len(old_documents)));random.Random(int(prior.get('schedule_seed',seed))+int(prior.get('epoch',0))).shuffle(order)
            index=int(prior.get('document_index',0))
            for item in order[:index]:offsets[item]=max(0,len(old_documents[item])-1)
            if index<len(order):offsets[order[index]]=int(prior.get('byte_offset',0))
        if len(offsets)!=len(old_documents):raise ValueError('Previous cursor does not match its immutable dataset')
        offset_by_hash={hashlib.sha256(data).hexdigest():offset for data,offset in zip(old_documents,offsets)}
    offsets=[min(offset_by_hash.get(hashlib.sha256(data).hexdigest(),0),max(0,len(data)-1)) for data in documents]
    return {'dataset_revision_sha256':revision,'schedule_version':'interleaved_articles_v1',
            'schedule_seed':int(prior.get('schedule_seed',seed)),'epoch':0,'document_index':0,'byte_offset':0,
            'document_offsets':offsets,'next_slot':0,
            'attempted_training_bytes':int(prior.get('attempted_training_bytes',0)),
            'windows_completed':int(prior.get('windows_completed',0)),
            'coverage_carried_from_revision':prior.get('dataset_revision_sha256')}


def next_window(documents, cursor, context, seed=42):
    if type(context)is not int or context<1:raise ValueError('Context must be a positive integer')
    if not documents or not any(len(document)>1 for document in documents):
        raise ValueError('No trainable document bytes')
    result=dict(cursor)
    count=len(documents)
    epoch=int(result.get('epoch',0))
    schedule_seed=int(result.setdefault('schedule_seed',seed))
    order=list(range(count));random.Random(schedule_seed+epoch).shuffle(order)
    if result.get('schedule_version')!='interleaved_articles_v1':
        # Translate the previous sequential cursor without discarding attempted
        # coverage. Offsets are indexed by immutable dataset document order.
        offsets=[0]*count
        position=int(result.get('document_index',0))
        if not 0<=position<=count:raise ValueError('Invalid legacy document cursor')
        for index in order[:min(position,count)]:offsets[index]=max(0,len(documents[index])-1)
        if position<count:
            offsets[order[position]]=min(int(result.get('byte_offset',0)),len(documents[order[position]])-1)
        result.update(schedule_version='interleaved_articles_v1',document_offsets=offsets,next_slot=position%count)
    offsets=list(result['document_offsets'])
    if len(offsets)!=count:raise ValueError('Training cursor does not match the dataset')
    slot=int(result.get('next_slot',0))%count
    for _ in range(2):
        for shift in range(count):
            position=(slot+shift)%count;index=order[position];offset=offsets[index]
            if not 0<=offset<=max(0,len(documents[index])-1):raise ValueError('Invalid document cursor offset')
            data=documents[index][offset:offset+context+1]
            if len(data)<2:continue
            offsets[index]+=len(data)-1
            next_slot=(position+1)%count
            result.update(epoch=epoch,document_offsets=offsets,next_slot=next_slot,
                          document_index=next_slot,byte_offset=offsets[order[next_slot]],
                          last_document_index=index,
                          documents_touched=sum(offset>0 for offset in offsets),
                          documents_completed=sum(offset>=max(0,len(documents[i])-1) for i,offset in enumerate(offsets)))
            return data,result
        epoch+=1;offsets=[0]*count;slot=0
        order=list(range(count));random.Random(schedule_seed+epoch).shuffle(order)
    raise ValueError('No trainable windows remain')


def repetition_metrics(text):
    import re
    words=re.findall(r'[^\W_]+',text.casefold(),re.UNICODE)
    triples=[tuple(words[index:index+3]) for index in range(max(0,len(words)-2))]
    return {'word_count':len(words),
            'unique_word_fraction':len(set(words))/max(1,len(words)),
            'repeated_trigram_fraction':(len(triples)-len(set(triples)))/max(1,len(triples))}


def repetition_guard(before,after):
    # Some repeated historical names are normal; looping most of a sample is
    # not. This is a fluency diagnostic, never a factual-answer verifier.
    threshold=max(.25,float(before.get('repeated_trigram_fraction',0))+.02)
    return float(after.get('repeated_trigram_fraction',0))<=threshold
