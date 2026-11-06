"""Resumable full-corpus PyTorch pass, separate from the accepted chat model."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import time

from .dataset_catalog import begin_dataset_training,finish_dataset_training,sync_saved_training_records
from .focus_control import acquire_lock,release_lock,stop_requested
from .owned_corpus import prepare_owned_corpus,read_owned_corpus
from .owned_learning import _save,_windows,_quality
from .pytorch_learning import TorchGRU
from .supervised_corpus import prepare_supervised_corpus,read_supervised_corpus,encode_example
from .training_schedule import adapt_dataset_cursor,next_window,repetition_guard
from .training_approval import operator_approvals

PROFILE='saudi-history-ar'
CHECKPOINT_KEY='focus_full_pass_checkpoint'
PROGRESS_KEY='focus_full_pass_progress'


def _now():return datetime.now(timezone.utc).isoformat()


def _verify_approvals(store,descriptor):
    approvals=operator_approvals(store)
    current={row['url']:row for row in store.db.execute('SELECT url,sha256,noai,rights_status FROM web_corpus_sources')}
    for line in (Path(descriptor['directory'])/'documents.jsonl').read_text().splitlines():
        row=json.loads(line);source=current.get(row['url']);parent=row['approval_parent_sha256']
        if not source or source['noai'] or not ((row['url'],parent) in approvals or
                source['sha256']==parent and source['rights_status'] in ('CC0-1.0','PUBLIC-DOMAIN')):
            raise ValueError('Frozen article approval is no longer valid: '+row['url'])


def _summary(state):
    target=state['steps_target'];done=state['steps_done'];elapsed=state['elapsed_seconds']
    return {key:state.get(key) for key in ('pass_id','status','steps_done','steps_target',
        'language_steps_done','qa_steps_done','language_target_bytes','language_bytes_done',
        'training_bytes_seen','elapsed_seconds','saved_checkpoint_step')}|{
        'steps_remaining':max(0,target-done),'fraction_complete':done/max(1,target),
        'eta_seconds':round(elapsed/done*(target-done)) if done else None,
        'dataset_revision_sha256':state['dataset']['dataset_revision_sha256'],
        'meaning':'One full pass of training articles and QA; held-out articles excluded; candidate not yet accepted.'}


def _remove_previous_candidate(store,previous,current):
    if not previous.get('candidate_owned') or previous.get('sha256')==current['sha256']:return
    protected={r.get('sha256') for r in (store.get('focus_train_accepted_history_'+PROFILE) or [])}
    protected.add((store.get('focus_train_latest_'+PROFILE) or {}).get('sha256'))
    if previous.get('sha256') in protected:return
    path=Path(previous['checkpoint']).resolve();root=(store.root/'models/focus'/PROFILE).resolve()
    if path.parent==root and path.name=='pytorch-'+previous['sha256']+'.pt':path.unlink(missing_ok=True)


def train_full_pass(store,context=256,checkpoint_every=120,progress=None,should_stop=None):
    if type(context)is not int or not 64<=context<=1024:raise ValueError('context must be 64..1024')
    if type(checkpoint_every)is not int or not 1<=checkpoint_every<=1000:raise ValueError('checkpoint interval must be 1..1000')
    if (store.get('focus_execution_policy') or {}).get('engine')!='pytorch':raise ValueError('Full pass requires PyTorch')
    parent=store.get('focus_train_latest_'+PROFILE) or {}
    if not parent.get('accepted'):raise ValueError('No accepted project checkpoint')
    state=store.get(CHECKPOINT_KEY) or {}
    resume=state.get('status') in ('running','interrupted','error','evaluating')
    if resume and state['anchor']['sha256']!=parent['sha256']:raise ValueError('Accepted model changed; preserve and review the pending full pass')
    if resume and state['context_bytes']!=context:raise ValueError('Resume with the original full-pass context length')
    dataset=state['dataset'] if resume else prepare_owned_corpus(store)
    splits=read_owned_corpus(dataset);_verify_approvals(store,dataset)
    qa_dataset=state['qa_dataset'] if resume else prepare_supervised_corpus(store,dataset)
    qa=read_supervised_corpus(qa_dataset)
    validation=_windows(splits['val'],context,1031)
    qa_validation=random.Random(1031).sample(qa['val'],min(12,len(qa['val'])))
    vocabulary=set(re.findall(r'[\u0621-\u064a]{3,}',b' '.join(splits['train']).decode()))
    path=Path((state if resume else parent)['checkpoint']).resolve()
    if path.parent!=(store.root/'models/focus'/PROFILE).resolve():raise ValueError('Checkpoint outside model archive')
    net,_=TorchGRU.load(store.root,path,(state if resume else parent)['sha256'])
    try:
        if not resume:
            baseline={'language_loss':sum(net.loss(data) for data in validation)/len(validation),
                'qa_loss':sum(net.qa_loss(*encode_example(row)) for row in qa_validation)/len(qa_validation),
                'quality':_quality(net,vocabulary)}
            language_steps=sum(math.ceil((len(data)-1)/context) for data in splits['train'])
            state={'format':'berhot_full_pass_v1','pass_id':_now(),'status':'running','anchor':parent,
                'dataset':dataset,'qa_dataset':qa_dataset,'context_bytes':context,'learning_rate':.0001,
                'steps_target':language_steps+len(qa['train']),'language_steps_target':language_steps,
                'steps_done':0,'language_steps_done':0,
                'qa_steps_done':0,'language_target_bytes':sum(len(data)-1 for data in splits['train']),
                'language_bytes_done':0,'training_bytes_seen':0,'elapsed_seconds':0,'saved_checkpoint_step':0,
                'cursor':adapt_dataset_cursor(splits['train'],{},dataset['dataset_revision_sha256']),
                'baseline':baseline,'checkpoint':parent['checkpoint'],'sha256':parent['sha256'],
                'receipt':begin_dataset_training(store,dataset,PROFILE,language_steps+len(qa['train']))}
            store.put(CHECKPOINT_KEY,state)
        state['status']='running';base_elapsed=state['elapsed_seconds'];started=time.monotonic()
        last_saved=state['saved_checkpoint_step']

        def save_checkpoint(status):
            nonlocal last_saved
            if shutil.disk_usage(store.root).free<2_000_000_000:raise ValueError('Less than 2GB free; preserving saved training data')
            previous=store.get(CHECKPOINT_KEY) or {}
            checkpoint,digest=_save(store,net,{'context_bytes':context,'full_pass_id':state['pass_id'],
                'dataset_revision_sha256':dataset['dataset_revision_sha256'],
                'instruction_tuned':bool(parent.get('instruction_tuned') or state['qa_steps_done']),
                'qa_format':'berhot_ar_extract_qa_v1'})
            state.update(status=status,checkpoint=checkpoint,sha256=digest,candidate_owned=True,
                         saved_checkpoint_step=state['steps_done'],elapsed_seconds=base_elapsed+time.monotonic()-started)
            store.put(CHECKPOINT_KEY,state);store.put(PROGRESS_KEY,_summary(state));last_saved=state['steps_done']
            _remove_previous_candidate(store,previous,state)

        while state['steps_done']<state['steps_target']:
            if should_stop and should_stop():
                save_checkpoint('interrupted');return {'status':'interrupted',**_summary(state)}
            if state['steps_done']%checkpoint_every==0 and shutil.disk_usage(store.root).free<2_000_000_000:
                raise ValueError('Less than 2GB free; full-pass training waits for free space')
            language_done=state['language_bytes_done']>=state['language_target_bytes']
            qa_due=state['language_steps_done']*len(qa['train'])//state['language_steps_target']
            use_qa=state['qa_steps_done']<len(qa['train']) and (language_done or state['qa_steps_done']<qa_due)
            if use_qa:
                row=qa['train'][state['qa_steps_done']];prompt,answer=encode_example(row)
                loss=net.train_qa(prompt,answer,state['learning_rate']);task='supervised_qa'
                state['qa_steps_done']+=1;seen=len(prompt)+len(answer)-1
            else:
                data,cursor=next_window(splits['train'],state['cursor'],context)
                if cursor['epoch']!=0:raise ValueError('Full pass unexpectedly repeated training bytes')
                loss=net.train(data,state['learning_rate']);task='language';seen=len(data)-1
                cursor['attempted_training_bytes']+=seen;cursor['windows_completed']+=1
                state.update(cursor=cursor,language_steps_done=state['language_steps_done']+1,
                             language_bytes_done=state['language_bytes_done']+seen)
            state['steps_done']+=1;state['training_bytes_seen']+=seen
            state['elapsed_seconds']=base_elapsed+time.monotonic()-started
            state['last_task']=task;state['last_loss']=loss
            store.put(PROGRESS_KEY,_summary(state))
            if state['steps_done']-last_saved>=checkpoint_every:save_checkpoint('running')
            if progress:progress(_summary(state),task,loss)
        save_checkpoint('evaluating')
        if progress:progress(_summary(state),'evaluation',None)
        val=sum(net.loss(data) for data in validation)/len(validation)
        qa_val=sum(net.qa_loss(*encode_example(row)) for row in qa_validation)/len(qa_validation)
        quality=_quality(net,vocabulary);before=state['baseline'];baseline_quality=before['quality']
        repetition_ok=repetition_guard(baseline_quality,quality) and all(repetition_guard(a,b)
            for a,b in zip(baseline_quality['samples'],quality['samples']))
        accepted=(val<=before['language_loss']+.02 and qa_val+.001<before['qa_loss'] and
                  val+qa_val+.001<before['language_loss']+before['qa_loss'] and repetition_ok)
        test=_windows(splits['test'],context,2053,6)
        qa_test=random.Random(2053).sample(qa['test'],min(6,len(qa['test'])))
        record={'status':'trained','format':net.format,'training_engine':'pytorch','accepted':accepted,
            'checkpoint':state['checkpoint'],'sha256':state['sha256'],'candidate_weights_retained':True,
            'steps':state['steps_done'],'steps_total':int(parent.get('steps_total',0))+state['steps_done'],
            'optimizer_steps':net.step,'trainable_parameters':net.count,'model_config':net.config,
            'context_bytes':context,'parent_checkpoint_sha256':parent['sha256'],
            'validation_before':before['language_loss'],'validation_after':val,
            'qa_validation_loss_before':before['qa_loss'],'qa_validation_loss_after':qa_val,
            'heldout_test_loss_diagnostic':sum(net.loss(data) for data in test)/len(test),
            'qa_test_loss_diagnostic':sum(net.qa_loss(*encode_example(row)) for row in qa_test)/len(qa_test),
            'selected_on_test':False,'quality_before':baseline_quality,'quality_after':quality,
            'acceptance_checks':{'repetition_guard_passed':repetition_ok,'validation_improved':val<before['language_loss']},
            'supervised_qa_steps':state['qa_steps_done'],
            'supervised_qa_steps_total':int(parent.get('supervised_qa_steps_total',0))+state['qa_steps_done'],
            'instruction_tuned':True,'qa_format':'berhot_ar_extract_qa_v1','qa_dataset':qa_dataset,
            'training_window_bytes_seen':state['training_bytes_seen'],
            'unique_approved_training_bytes':sum(len(data) for data in splits['train']),
            'dataset_revision_sha256':dataset['dataset_revision_sha256'],'plain_dataset_directory':dataset['directory'],
            'plain_dataset_manifest':dataset['manifest'],'training_cursor':state['cursor'],
            'documents_by_split':{part:len(docs) for part,docs in splits.items()},'full_training_pass_completed':True,
            'language_target_bytes_processed':state['language_bytes_done'],'created_at_utc':_now(),
            'duration_seconds':state['elapsed_seconds'],'external_model_api_used':False,'pretrained_weights_used':False,
            'history_qa_evaluated':False,'historical_accuracy_verified':False}
        if (store.get('focus_train_latest_'+PROFILE) or {}).get('sha256')!=parent['sha256']:
            raise ValueError('Accepted model changed during full pass; refusing promotion')
        store.put('focus_train_attempted_'+PROFILE,record)
        if accepted:
            history=store.get('focus_train_accepted_history_'+PROFILE) or []
            if not any(row.get('sha256')==parent['sha256'] for row in history):history.append(parent)
            history.append(record);store.put('focus_train_accepted_history_'+PROFILE,history)
            store.put('focus_train_latest_'+PROFILE,record)
            store.put('owned_training_cursor_'+PROFILE,state['cursor'])
        finish_dataset_training(store,dataset,state['receipt'],record=record)
        if accepted:sync_saved_training_records(store)
        state.update(status='completed',result=record);store.put(CHECKPOINT_KEY,state);store.put(PROGRESS_KEY,_summary(state))
        return {'status':'completed','accepted':accepted,**_summary(state),'result':record}
    finally:net.close()


def run_full_pass(store,context=256):
    lock=acquire_lock(store);stopping=False;previous={};run_id=_now().replace(':','').replace('-','')
    state={'profile':PROFILE,'run_id':run_id,'status':'running','started_at_utc':_now(),'worker_pid':os.getpid(),
        'training_engine':'pytorch','network_enabled':False,'current_phase':'preparing_full_training_pass',
        'training_steps_this_run':0,'training_byte_tokens_processed':0,'pages_read':0,'pages_attempted':0,
        'cycles_completed':0,'current_training_steps_done':0,'current_training_steps_target':0}
    def publish(phase,event,detail):
        state.update(current_phase=phase,heartbeat_at_utc=_now(),updated_at_utc=_now())
        store.put('focus_run_'+PROFILE,state)
        store.append_focus_activity(PROFILE,run_id,_now(),event,phase=phase,detail=detail)
    def stop(*_):
        nonlocal stopping
        stopping=True
    def requested():return stopping or stop_requested(store,run_id,PROFILE)
    for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,stop)
    try:
        publish('preparing_full_training_pass','worker_started','Full saved-corpus pass; candidate checkpoints retained; held-out articles excluded')
        prior=store.get(CHECKPOINT_KEY) or {}
        resuming=prior.get('status') in ('running','interrupted','error','evaluating')
        base=int(prior.get('steps_done',0)) if resuming else 0
        base_bytes=int(prior.get('training_bytes_seen',0)) if resuming else 0
        def progress(summary,task,loss):
            state.update(full_pass=summary,current_training_steps_done=summary['steps_done'],
                current_training_steps_target=summary['steps_target'],training_steps_this_run=summary['steps_done']-base,
                training_byte_tokens_processed=summary['training_bytes_seen']-base_bytes,current_training_loss=loss,
                current_training_task=task)
            publish('evaluating_full_training_pass' if task=='evaluation' else 'training_neural_weights',
                'full_pass_evaluation' if task=='evaluation' else 'neural_optimizer_step',
                str(summary['steps_done'])+'/'+str(summary['steps_target'])+' · '+task+' · '+
                str(summary['steps_remaining'])+' updates remaining · loss '+str(round(loss,6) if loss is not None else 'evaluating'))
        result=train_full_pass(store,context,progress=progress,should_stop=requested)
        state.update(status='stopped' if result['status']=='interrupted' else 'completed',latest_training=result,
                     finished_at_utc=_now(),cycles_completed=int(result['status']=='completed'))
        publish(state['status'],'full_pass_'+result['status'],'Full corpus pass '+result['status']+'; accepted='+str(result.get('accepted')))
        return result
    except Exception as exc:
        state.update(status='error',error_type=type(exc).__name__,error=str(exc)[:300],finished_at_utc=_now())
        pending=store.get(PROGRESS_KEY) or {}
        if pending:
            pending.update(status='error',error_type=type(exc).__name__)
            store.put(PROGRESS_KEY,pending)
        publish('error','full_pass_error',type(exc).__name__+': '+str(exc)[:250]);raise
    finally:
        release_lock(lock)
        for sig,handler in previous.items():signal.signal(sig,handler)
