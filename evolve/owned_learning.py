"""Owned, offline from-scratch learning and generation without ML packages."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import signal
import tempfile
import time

from .owned_engine import FORMAT,OwnedGRU,import_own_legacy_weights
from .owned_corpus import prepare_owned_corpus,read_owned_corpus
from .dataset_catalog import begin_dataset_training,finish_dataset_training,sync_saved_training_records

PROFILE='saudi-history-ar'
POLICY='focus_execution_policy'


def owned_enabled(store):return (store.get(POLICY) or {}).get('engine')=='first_party_c'


def owned_audit(store):
    import sys
    engine=(store.get(POLICY) or {}).get('engine','first_party_c')
    record=store.get('focus_train_latest_'+PROFILE) or {}
    dataset=store.get('owned_corpus_latest_'+PROFILE) or {}
    source=Path(__file__).with_name('owned_gru.c')
    return {'engine':engine,'policy':store.get(POLICY),
            'runtime':'PyTorch running the project-owned GRU' if engine=='pytorch' else 'Python standard library plus project-owned compiled C',
            'native_kernel_active':engine=='first_party_c',
            'site_packages_disabled':bool(sys.flags.no_site),
            'third_party_ml_modules_loaded':[name for name in ('torch','numpy','playwright','transformers','requests') if name in sys.modules],
            'native_source':str(source),'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'system_prerequisites':['Python','local C compiler','system C/math runtime'],
            'model':{k:record.get(k) for k in ('format','sha256','parent_checkpoint_sha256','steps_total','owned_optimizer_steps','context_bytes','validation_after')},
            'dataset':dataset,'legacy_files_preserved_but_disabled':True,
            'historical_answer_accuracy_verified':False}


def _now():return datetime.now(timezone.utc).isoformat()


def _save(store,net,metadata):
    root=store.root/'models'/'focus'/PROFILE;root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.owned-save-',dir=root) as temp:
        path=Path(temp)/'checkpoint.bin';digest=net.save(path,**metadata)
        pytorch=getattr(net,'format',None)=='berhot_focus_byte_gru_v1'
        destination=root/(('pytorch-' if pytorch else 'owned-')+digest+('.pt' if pytorch else '.bin'))
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise ValueError('Existing owned checkpoint changed')
        else:path.rename(destination)
    return str(destination),digest


def migrate_owned(store):
    from .focus_control import worker_running
    if worker_running(store):raise ValueError('Stop the learning worker before switching engines')
    previous=store.get('focus_train_latest_'+PROFILE) or {}
    if previous.get('format')==FORMAT:
        net,_=OwnedGRU.load(store.root,previous['checkpoint'],previous['sha256']);net.close()
        store.put(POLICY,{'engine':'first_party_c','external_model_apis':False,'browser_search':False,'configured_at_utc':_now()})
        return previous
    if not previous.get('accepted'):raise ValueError('No accepted local weights available for migration')
    net=import_own_legacy_weights(store.root,previous)
    path,digest=_save(store,net,{'context_bytes':previous.get('context_bytes',48),
                               'migrated_from_sha256':previous['sha256']})
    record=dict(previous,format=FORMAT,status='migrated_local_weights',checkpoint=path,sha256=digest,
                steps=0,owned_optimizer_steps=0,training_engine='first_party_c',
                parent_checkpoint_sha256=previous['sha256'],migrated_at_utc=_now(),
                context_bytes=previous.get('context_bytes',48),model_downloaded=False,
                external_ml_runtime_required=False)
    store.put('focus_legacy_latest_'+PROFILE,previous)
    history=store.get('focus_train_accepted_history_'+PROFILE) or []
    if not any(r.get('sha256')==previous['sha256'] for r in history):history.append(previous)
    store.put('focus_train_accepted_history_'+PROFILE,history)
    store.put('focus_train_latest_'+PROFILE,record)
    store.put(POLICY,{'engine':'first_party_c','external_model_apis':False,'browser_search':False,'configured_at_utc':_now()})
    net.close();return record


def _generate(net,prompt,limit,temperature,top_k,seed):
    from .focus_inference import _allowed_bytes
    generator=random.Random(seed);hidden=None;scores=None
    for token in prompt.encode('utf-8')[-512:]:scores,hidden=net.logits(token,hidden)
    if scores is None:raise ValueError('A nonempty prompt is required')
    output=bytearray();prefix=bytearray()
    for i in range(limit):
        choices=_allowed_bytes(prefix,limit-i)
        ranked=sorted(choices,key=lambda token:(-scores[token],token))[:top_k]
        if temperature<=.000001:token=ranked[0]
        else:
            maximum=scores[ranked[0]];weights=[math.exp((scores[k]-maximum)/temperature) for k in ranked]
            threshold=generator.random()*sum(weights);token=ranked[-1]
            for item,weight in zip(ranked,weights):
                threshold-=weight
                if threshold<=0:token=item;break
        output.append(token)
        if prefix:
            prefix.append(token);length=2 if prefix[0]<0xe0 else 3 if prefix[0]<0xf0 else 4
            if len(prefix)==length:prefix.clear()
        elif token>=0xc2:prefix.append(token)
        scores,hidden=net.logits(token,hidden)
    return output.decode('utf-8')


def generate_owned(owner,store,prompt,max_new_bytes,temperature,top_k,seed):
    record=store.get('focus_train_latest_'+PROFILE) or {}
    path=Path(record.get('checkpoint','')).resolve();root=(store.root/'models'/'focus'/PROFILE).resolve()
    if record.get('format')!=FORMAT or not record.get('accepted') or not path.is_relative_to(root):
        raise ValueError('No accepted owned checkpoint')
    stat=path.stat();key=(str(path),record['sha256'],stat.st_size,stat.st_mtime_ns)
    if getattr(owner,'owned_cache_key',None)!=key:
        if getattr(owner,'owned_net',None):owner.owned_net.close()
        owner.owned_net,_=OwnedGRU.load(store.root,path,record['sha256']);owner.owned_cache_key=key
    started=time.monotonic();text=_generate(owner.owned_net,prompt,max_new_bytes,temperature,top_k,seed)
    return {'status':'generated','mode':'focus','answer':text,'model_generated':True,
            'checkpoint_sha256':record['sha256'],'steps_total':record.get('steps_total',0),
            'trainable_parameters':owner.owned_net.count,'sources':[],'training_updated':False,
            'training_engine':'first_party_c','external_model_api_used':False,
            'note':'Generated by the first-party local model; historical accuracy remains unverified.',
            'generation':{'generated_bytes':len(text.encode()),'prompt_bytes_used':min(512,len(prompt.encode())),
                          'elapsed_seconds':round(time.monotonic()-started,3),'seed':seed,'temperature':temperature,
                          'top_k':top_k,'utf8_constrained':True,'conversation_memory':False}}


def _windows(documents,context,seed,count=12):
    generator=random.Random(seed);usable=[d for d in documents if len(d)>context]
    if not usable:raise ValueError('Insufficient held-out text for this context length')
    result=[]
    for _ in range(count):
        document=usable[generator.randrange(len(usable))];start=generator.randrange(len(document)-context)
        result.append(document[start:start+context+1])
    return result


def _quality(net,vocabulary):
    from .training_schedule import repetition_metrics
    samples=[]
    for prompt in ('تاريخ الدولة السعودية الأولى','تأسست المملكة العربية السعودية'):
        text=_generate(net,prompt,256,0,40,42);words=re.findall(r'[\u0621-\u064a]{3,}',text)
        samples.append({'prompt':prompt,'continuation':text,
                        **repetition_metrics(text),
                        'known_word_fraction':sum(w in vocabulary for w in words)/max(1,len(words)),
                        'toc_occurrences':text.count('جدول المحتويات')})
    return {'samples':samples,'known_word_fraction':sum(r['known_word_fraction'] for r in samples)/len(samples),
            'repeated_trigram_fraction':sum(r['repeated_trigram_fraction'] for r in samples)/len(samples),
            'historical_accuracy_verified':False,'metric_is_only_a_heuristic':True}


def train_owned(store,steps=120,context=256,seed=42,progress=None,should_stop=None):
    if type(steps)is not int or not 1<=steps<=120:raise ValueError('steps must be 1..120')
    if type(context)is not int or not 64<=context<=1024:raise ValueError('context must be 64..1024 bytes')
    pytorch=(store.get(POLICY) or {}).get('engine')=='pytorch'
    if not owned_enabled(store) and not pytorch:raise ValueError('Select a local training engine first')
    descriptor=prepare_owned_corpus(store);splits=read_owned_corpus(descriptor)
    manifest=json.loads(Path(descriptor['manifest']).read_text())
    parent=store.get('focus_train_latest_'+PROFILE) or {}
    model_format='berhot_focus_byte_gru_v1' if pytorch else FORMAT
    if parent.get('format')!=model_format:raise ValueError('Accepted weights do not match the training engine')
    path=Path(parent['checkpoint']).resolve();model_root=(store.root/'models'/'focus'/PROFILE).resolve()
    if not path.is_relative_to(model_root):raise ValueError('Owned checkpoint outside model archive')
    if pytorch:
        from .pytorch_learning import TorchGRU
        net,_=TorchGRU.load(store.root,path,parent['sha256'])
    else:net,_=OwnedGRU.load(store.root,path,parent['sha256'])
    validation=_windows(splits['val'],context,1031);test=_windows(splits['test'],context,2053,6)
    pre=sum(net.loss(data) for data in validation)/len(validation)
    vocabulary=set(re.findall(r'[\u0621-\u064a]{3,}',b' '.join(splits['train']).decode()))
    before_quality=_quality(net,vocabulary)
    policy=store.get(POLICY) or {}
    qa_enabled=pytorch and bool(policy.get('supervised_qa'))
    qa_descriptor=None;qa_parts=None;qa_before=None;qa_tests=None
    control=store.get('focus_autonomous_learning_control') or {}
    rate=max(.00005,min(.0003,float(control.get('learning_rate',.0003))))
    qa_every=max(2,min(8,int(control.get('qa_every',4))))
    if qa_enabled:
        from .supervised_corpus import prepare_supervised_corpus,read_supervised_corpus,encode_example
        qa_descriptor=prepare_supervised_corpus(store,descriptor);qa_parts=read_supervised_corpus(qa_descriptor)
        qa_validation=random.Random(1031).sample(qa_parts['val'],min(12,len(qa_parts['val'])))
        qa_tests=random.Random(2053).sample(qa_parts['test'],min(6,len(qa_parts['test'])))
        qa_before=sum(net.qa_loss(*encode_example(row)) for row in qa_validation)/len(qa_validation)
    cursor_key='owned_training_cursor_'+PROFILE;cursor=store.get(cursor_key) or {}
    if cursor.get('dataset_revision_sha256')!=descriptor['dataset_revision_sha256']:
        old_documents=None;old_revision=cursor.get('dataset_revision_sha256','')
        if old_revision and (store.root/'datasets'/'focus_plain'/old_revision/'manifest.json').is_file():
            old_documents=read_owned_corpus({'directory':str(store.root/'datasets'/'focus_plain'/old_revision),
                                            'dataset_revision_sha256':old_revision})['train']
        from .training_schedule import adapt_dataset_cursor
        cursor=adapt_dataset_cursor(splits['train'],cursor,descriptor['dataset_revision_sha256'],old_documents,seed)
    receipt=begin_dataset_training(store,descriptor,PROFILE,steps)
    qa_cursor_key='focus_supervised_qa_cursor';qa_cursor=store.get(qa_cursor_key) or {}
    if qa_enabled and qa_cursor.get('revision')!=qa_descriptor['revision']:
        qa_cursor={**qa_cursor,'revision':qa_descriptor['revision'],'epoch':0,'index':0,
                   'steps':int(qa_cursor.get('steps',0)),'input_bytes':int(qa_cursor.get('input_bytes',0)),
                   'answer_bytes':int(qa_cursor.get('answer_bytes',0))}
    if qa_enabled:
        store.db.execute('CREATE TABLE IF NOT EXISTS supervised_qa_progress '
                         '(qa_id TEXT PRIMARY KEY, train_steps INTEGER NOT NULL DEFAULT 0, last_loss REAL)')
        qa_visits={row['qa_id']:int(row['train_steps']) for row in store.db.execute('SELECT * FROM supervised_qa_progress')}
    actual=0;qa_steps=0;loss=None;started=time.monotonic();base_bytes=cursor['attempted_training_bytes']
    qa_base_input=int(qa_cursor.get('input_bytes',0));qa_base_answer=int(qa_cursor.get('answer_bytes',0))
    try:
        for iteration in range(steps):
            if should_stop and should_stop():break
            if qa_enabled and (iteration+1)%qa_every==0:
                example=min(qa_parts['train'],key=lambda row:(qa_visits.get(row['id'],0),row['id']))
                prompt,answer=encode_example(example)
                loss=net.train_qa(prompt,answer,rate);qa_steps+=1
                qa_visits[example['id']]=qa_visits.get(example['id'],0)+1
                store.db.execute('INSERT INTO supervised_qa_progress(qa_id,train_steps,last_loss) VALUES(?,?,?) '
                                 'ON CONFLICT(qa_id) DO UPDATE SET train_steps=excluded.train_steps,last_loss=excluded.last_loss',
                                 (example['id'],qa_visits[example['id']],loss))
                qa_cursor.update(index=qa_cursor['index']+1,steps=qa_cursor['steps']+1,
                                 input_bytes=qa_cursor['input_bytes']+len(prompt)+len(answer)-1,
                                 answer_bytes=qa_cursor['answer_bytes']+len(answer),
                                 epoch=min(qa_visits.get(row['id'],0) for row in qa_parts['train']))
                store.put(qa_cursor_key,qa_cursor)
                step_task='supervised_qa'
            else:
                from .training_schedule import next_window
                data,planned_cursor=next_window(splits['train'],cursor,context,seed)
                loss=net.train(data,rate);cursor=planned_cursor
                cursor['attempted_training_bytes']+=len(data)-1;cursor['windows_completed']+=1
                store.put(cursor_key,cursor);step_task='language'
            actual+=1
            store.put('focus_training_last_step',{'task':step_task,'loss':loss})
            if progress:progress(actual,steps,loss)
        if not actual:
            finish_dataset_training(store,descriptor,receipt,error=InterruptedError())
            return {'status':'skipped_stop_requested','steps':0}
        val=sum(net.loss(data) for data in validation)/len(validation)
        test_loss=sum(net.loss(data) for data in test)/len(test);after_quality=_quality(net,vocabulary)
        from .training_schedule import repetition_guard
        validation_improved=val+.001<pre
        repetition_ok=repetition_guard(before_quality,after_quality)
        repetition_ok=repetition_ok and all(repetition_guard(before,after)
            for before,after in zip(before_quality['samples'],after_quality['samples']))
        accepted=validation_improved and repetition_ok
        qa_after=None;qa_test_loss=None
        if qa_enabled:
            qa_after=sum(net.qa_loss(*encode_example(row)) for row in qa_validation)/len(qa_validation)
            qa_test_loss=sum(net.qa_loss(*encode_example(row)) for row in qa_tests)/len(qa_tests)
            accepted=(val<=pre+.02 and qa_after+.001<qa_before and
                      (val+qa_after)+.001<(pre+qa_before) and repetition_ok)
        # Unaccepted weights are temporary. Keep their measured receipt, not
        # an unbounded stream of 30MB rejected candidates on this 8GB laptop.
        checkpoint,digest=(None,None)
        if accepted:
            import shutil
            if shutil.disk_usage(store.root).free<2_000_000_000:raise ValueError('Less than 2GB free; stopping owned training to preserve saved data')
            checkpoint,digest=_save(store,net,{'context_bytes':context,'dataset_revision_sha256':descriptor['dataset_revision_sha256'],
                'instruction_tuned':bool(parent.get('instruction_tuned') or qa_steps),
                'qa_format':'berhot_ar_extract_qa_v1' if qa_enabled else parent.get('qa_format')})
        record={'status':'trained','format':model_format,'training_engine':'pytorch' if pytorch else 'first_party_c','accepted':accepted,
                'steps':actual,'steps_total':int(parent.get('steps_total',0))+actual,
                'owned_optimizer_steps':int(parent.get('owned_optimizer_steps',0)) if pytorch else net.step,
                'context_bytes':context,'model_config':net.config,'trainable_parameters':net.count,
                'new_parameters_added':0,'resumed_accepted_weights':True,
                'optimizer_resumed':net.step>actual,'optimizer_steps':net.step,
                'parent_checkpoint_sha256':parent['sha256'],'checkpoint':checkpoint,'sha256':digest,
                'validation_before':round(pre,6),'validation_after':round(val,6),
                'heldout_test_loss_diagnostic':round(test_loss,6),'selected_on_test':False,
                'train_loss_last_batch':loss,'training_window_bytes_seen':cursor['attempted_training_bytes']-base_bytes+
                    int(qa_cursor.get('input_bytes',0))-qa_base_input,
                'candidate_weights_retained':accepted,
                'unique_approved_training_bytes':sum(len(d) for d in splits['train']),
                'plain_dataset_directory':descriptor['directory'],'plain_dataset_manifest':descriptor['manifest'],
                'dataset_revision_sha256':descriptor['dataset_revision_sha256'],
                'documents_by_split':manifest['documents_by_split'],'split_method':manifest['split_method'],
                'training_cursor':dict(cursor),'quality_before':before_quality,'quality_after':after_quality,
                'acceptance_checks':{'validation_improved':validation_improved,'repetition_guard_passed':repetition_ok},
                'instruction_tuned':bool(parent.get('instruction_tuned') or qa_steps),
                'supervised_qa_steps':qa_steps,'supervised_qa_steps_total':int(parent.get('supervised_qa_steps_total',0))+qa_steps,
                'qa_format':'berhot_ar_extract_qa_v1' if qa_enabled else parent.get('qa_format'),
                'qa_dataset':qa_descriptor,'qa_validation_loss_before':qa_before,'qa_validation_loss_after':qa_after,
                'qa_test_loss_diagnostic':qa_test_loss,'supervised_answer_bytes_seen':int(qa_cursor.get('answer_bytes',0))-qa_base_answer,
                'training_objective':'next_byte_prediction','validation_metric':'next_UTF8_byte_cross_entropy',
                'history_qa_evaluated':False,'understands_saudi_history':False,
                'external_ml_runtime_required':pytorch,'model_downloaded':False,
                'pretrained_weights_used':False,'external_model_api_used':False,
                'created_at_utc':_now(),'duration_seconds':round(time.monotonic()-started,2)}
        if qa_enabled:
            from .supervised_corpus import plan_curiosity
            # Research planning must never invalidate a completed evaluation.
            try:record['curiosity_tasks_created']=len(plan_curiosity(store,net,qa_parts['train']))
            except (OSError,ValueError,RuntimeError) as exc:
                record['curiosity_planning_error']=type(exc).__name__
            new_rate=rate if accepted else max(.00005,rate*.8)
            if not accepted and qa_after is not None and qa_after>=qa_before:qa_every=max(2,qa_every-1)
            elif not accepted and val>pre+.02:qa_every=min(8,qa_every+1)
            store.put('focus_autonomous_learning_control',{'learning_rate':new_rate,'qa_every':qa_every,
                'last_decision':'accepted' if accepted else 'retry_with_adjusted_training',
                'validation_before':pre,'validation_after':val,'qa_loss_before':qa_before,'qa_loss_after':qa_after})
        store.put('focus_train_attempted_'+PROFILE,record)
        if accepted:
            history=store.get('focus_train_accepted_history_'+PROFILE) or []
            if not any(r.get('sha256')==parent['sha256'] for r in history):history.append(parent)
            history.append(record);store.put('focus_train_accepted_history_'+PROFILE,history)
            store.put('focus_train_latest_'+PROFILE,record)
        finish_dataset_training(store,descriptor,receipt,record=record)
        if accepted:sync_saved_training_records(store)
        return record
    except Exception as exc:
        finish_dataset_training(store,descriptor,receipt,error=exc);raise
    finally:net.close()


def run_owned(store,steps=120,context=256,until_stopped=False):
    from .focus_control import acquire_lock,release_lock,stop_requested
    if type(steps)is not int or not 1<=steps<=120 or type(context)is not int or not 64<=context<=1024:
        raise ValueError('Owned worker requires steps 1..120 and context 64..1024')
    lock=acquire_lock(store);stopping=False;prior={};run_id=_now().replace(':','').replace('-','')
    engine=(store.get(POLICY) or {}).get('engine','first_party_c')
    state={'profile':PROFILE,'run_id':run_id,'started_at_utc':_now(),'status':'running',
           'worker_pid':__import__('os').getpid(),'training_engine':engine,'until_stopped':until_stopped,
           'current_phase':'preparing_owned_corpus','pages_read':0,'pages_attempted':0,'training_steps_this_run':0,
           'cycles_completed':0,'network_enabled':False,'external_model_apis':False,'heartbeat_at_utc':_now()}
    def publish(phase,event=None,detail=''):
        state.update(current_phase=phase,heartbeat_at_utc=_now(),updated_at_utc=_now());store.put('focus_run_'+PROFILE,state)
        if event:
            store.append_focus_activity(PROFILE,run_id,_now(),event,phase=phase,detail=detail)
    def stop(*_):
        nonlocal stopping
        stopping=True
    def requested():return stopping or stop_requested(store,run_id,PROFILE)
    for signum in (signal.SIGINT,signal.SIGTERM):prior[signum]=signal.signal(signum,stop)
    try:
        publish('preparing_owned_corpus','worker_started',engine+' local training; external model APIs disabled')
        while not requested():
            base=state['training_steps_this_run']
            base_tokens=state.get('training_byte_tokens_processed',0)
            start_bytes=(store.get('owned_training_cursor_'+PROFILE) or {}).get('attempted_training_bytes',0)
            publish('checking_owned_training_data')
            def progress(done,target,loss):
                state.update(current_training_steps_done=done,current_training_steps_target=target,
                             current_training_loss=round(loss,6),training_steps_this_run=base+done)
                current=(store.get('owned_training_cursor_'+PROFILE) or {}).get('attempted_training_bytes',0)
                state['training_byte_tokens_processed']=base_tokens+max(0,current-start_bytes)
                publish('training_pytorch_neural_weights' if engine=='pytorch' else 'training_owned_neural_weights',
                        'neural_optimizer_step',str(done)+'/'+str(target)+' '+engine+' optimizer update; loss '+str(round(loss,6)))
            result=train_owned(store,steps,context,progress=progress,should_stop=requested)
            state['cycles_completed']+=1;state['latest_training']=result
            state['training_byte_tokens_processed']=base_tokens+result.get('training_window_bytes_seen',0)
            publish('owned_training_cycle_finished','cycle_finished','First-party training: '+result['status']+'; accepted='+str(result.get('accepted')))
            print(json.dumps(result,ensure_ascii=False),flush=True)
            if not until_stopped:break
            # Bound the interval and keep manual Stop responsive.
            for _ in range(10):
                if requested():break
                time.sleep(1)
    except Exception as exc:
        state.update(status='error',error_type=type(exc).__name__,error=str(exc)[:400])
        raise
    finally:
        state.update(status='error' if state.get('status')=='error' else 'stopped' if requested() else 'completed',finished_at_utc=_now())
        publish(state['status'],'worker_'+state['status'],'Owned learning worker finished');release_lock(lock)
        for signum,handler in prior.items():signal.signal(signum,handler)
    return state
