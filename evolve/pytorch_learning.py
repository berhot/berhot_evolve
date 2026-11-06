"""PyTorch training for this project's own GRU and locally prepared dataset."""
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys

from .focus_training import FORMAT, _network


class TorchGRU:
    format=FORMAT

    def __init__(self,root,**config):
        self.torch,self.model=_network(config)
        self.torch.set_num_threads(min(4,self.torch.get_num_threads()))
        self.config=config;self.count=sum(p.numel() for p in self.model.parameters());self.step=0
        self.optimizer=self.torch.optim.AdamW(self.model.parameters(),lr=.0003)

    def close(self):pass

    def logits(self,token,hidden=None):
        self.model.eval()
        with self.torch.inference_mode():
            ids=self.torch.tensor([[token]],dtype=self.torch.long)
            output,state=self.model.gru(self.model.embedding(ids),hidden)
            return self.model.head(output[0,-1]).tolist(),state

    def loss(self,data,backward=False):
        import torch.nn.functional as F
        ids=self.torch.tensor([list(data)],dtype=self.torch.long)
        if backward:
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(self.model(ids[:,:-1]).reshape(-1,256),ids[:,1:].reshape(-1));loss.backward()
        else:
            self.model.eval()
            with self.torch.inference_mode():
                loss=F.cross_entropy(self.model(ids[:,:-1]).reshape(-1,256),ids[:,1:].reshape(-1))
        if not self.torch.isfinite(loss):raise ValueError('Non-finite PyTorch loss')
        return float(loss.item())

    def train(self,data,rate=.0003):
        for group in self.optimizer.param_groups:group['lr']=rate
        loss=self.loss(data,True);self.torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
        self.optimizer.step();self.step+=1;return loss

    def qa_loss(self,prompt,answer,backward=False):
        import torch.nn.functional as F
        if not prompt or not answer:raise ValueError('Supervised QA needs prompt and answer bytes')
        ids=self.torch.tensor([list(prompt+answer)],dtype=self.torch.long)
        labels=ids[:,1:].clone();labels[:,:len(prompt)-1]=-100
        self.model.train(backward)
        if backward:
            self.optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(self.model(ids[:,:-1]).reshape(-1,256),labels.reshape(-1),ignore_index=-100)
            loss.backward()
        else:
            with self.torch.inference_mode():
                loss=F.cross_entropy(self.model(ids[:,:-1]).reshape(-1,256),labels.reshape(-1),ignore_index=-100)
        if not self.torch.isfinite(loss):raise ValueError('Non-finite supervised QA loss')
        return float(loss.item())

    def train_qa(self,prompt,answer,rate=.0003):
        for group in self.optimizer.param_groups:group['lr']=rate
        loss=self.qa_loss(prompt,answer,True)
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
        self.optimizer.step();self.step+=1;return loss

    def save(self,path,context_bytes=256,**metadata):
        self.torch.save({'format':FORMAT,'weights':self.model.state_dict(),
                         'optimizer_state':self.optimizer.state_dict(),'optimizer_step':self.step,
                         'model_config':self.config,'context_bytes':context_bytes,**metadata},path)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls,root,path,expected_sha=None):
        blob=Path(path).read_bytes()
        if expected_sha and hashlib.sha256(blob).hexdigest()!=expected_sha:raise ValueError('PyTorch checkpoint SHA-256 mismatch')
        import io
        import torch
        saved=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
        if saved.get('format')!=FORMAT:raise ValueError('Not a project PyTorch GRU checkpoint')
        net=cls(root,**saved['model_config']);net.model.load_state_dict(saved['weights'],strict=True)
        if saved.get('optimizer_state'):net.optimizer.load_state_dict(saved['optimizer_state'])
        net.step=int(saved.get('optimizer_step',0));return net,saved


def restore_pytorch(store):
    from .focus_control import worker_running
    from .owned_engine import MAGIC,KEYS
    from .owned_learning import _save,_now
    if worker_running(store):raise ValueError('Stop the learning worker before restoring PyTorch')
    previous=store.get('focus_train_latest_saudi-history-ar') or {}
    if previous.get('format')==FORMAT:
        net,_=TorchGRU.load(store.root,previous['checkpoint'],previous['sha256']);net.close()
        record=dict(previous,training_engine='pytorch')
    else:
        path=Path(previous['checkpoint']).resolve();root=(store.root/'models'/'focus'/'saudi-history-ar').resolve()
        if not path.is_relative_to(root):raise ValueError('Checkpoint outside local model archive')
        blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=previous['sha256'] or not blob.startswith(MAGIC):raise ValueError('Invalid preserved checkpoint')
        length=struct.unpack('<I',blob[8:12])[0];header=json.loads(blob[12:12+length])
        values=array('f');values.frombytes(blob[12+length:])
        if sys.byteorder!='little':values.byteswap()
        net=TorchGRU(store.root,**header['model_config']);count=net.count
        if len(values)!=3*count:raise ValueError('Incomplete preserved weight/optimizer state')
        flat=net.torch.frombuffer(values,dtype=net.torch.float32);offset=0;state={}
        parameters=dict(net.model.named_parameters())
        for key in KEYS:
            parameter=parameters[key];size=parameter.numel()
            state[key]=flat[offset:offset+size].reshape(parameter.shape).clone()
            net.optimizer.state[parameter]={
                'step':net.torch.tensor(float(header['optimizer_step'])),
                'exp_avg':flat[count+offset:count+offset+size].reshape(parameter.shape).clone(),
                'exp_avg_sq':flat[2*count+offset:2*count+offset+size].reshape(parameter.shape).clone()}
            offset+=size
        net.model.load_state_dict(state,strict=True);net.step=int(header['optimizer_step'])
        checkpoint,digest=_save(store,net,{'context_bytes':header.get('context_bytes',256),
                                           'restored_from_sha256':previous['sha256']})
        record=dict(previous,format=FORMAT,training_engine='pytorch',checkpoint=checkpoint,sha256=digest,
                    status='restored_pytorch',steps=0,optimizer_steps=net.step,
                    parent_checkpoint_sha256=previous['sha256'],restored_at_utc=_now(),
                    external_ml_runtime_required=True,pretrained_weights_used=False,external_model_api_used=False)
        store.put('focus_preserved_c_latest_saudi-history-ar',previous)
    history=store.get('focus_train_accepted_history_saudi-history-ar') or []
    if previous and not any(r.get('sha256')==previous['sha256'] for r in history):history.append(previous)
    store.put('focus_train_accepted_history_saudi-history-ar',history)
    store.put('focus_train_latest_saudi-history-ar',record)
    store.put('focus_execution_policy',{'engine':'pytorch','local_from_scratch':True,'article_body_training':True,
                                       'external_model_apis':False,'pretrained_weights':False,'browser_search':False,
                                       'configured_at_utc':_now()})
    net.close();return record


def train_pytorch(store,steps=120,context=256,seed=42,progress=None,should_stop=None):
    from .owned_learning import train_owned
    return train_owned(store,steps,context,seed,progress,should_stop)


def run_pytorch(store,steps=120,context=256,until_stopped=False):
    from .owned_learning import run_owned
    return run_owned(store,steps,context,until_stopped)
