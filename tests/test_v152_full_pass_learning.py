"""Full pass keeps optimizer progress, covers all training targets, holds out evaluation."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolve.store import Store
from evolve.owned_corpus import read_owned_corpus
from evolve.supervised_corpus import read_supervised_corpus
import test_v151_autonomous_pipeline as fixtures

torch=fixtures.torch


@unittest.skipIf(torch is None,'PyTorch required')
class FullPassLearningTests(unittest.TestCase):
    def setUp(self):
        from evolve.pytorch_learning import TorchGRU
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name);self.addCleanup(self.store.close)
        self.dataset=fixtures.AutonomousPipelineTests.body_dataset(self)
        root=self.store.root/'models/focus/saudi-history-ar';root.mkdir(parents=True)
        net=TorchGRU(self.tmp.name,embedding_dim=16,hidden_size=16)
        self.original=root/'original.pt';self.original_sha=net.save(self.original)
        self.store.put('focus_train_latest_saudi-history-ar',{'format':net.format,'accepted':True,
            'checkpoint':str(self.original),'sha256':self.original_sha,'steps_total':7,'model_config':net.config})
        self.store.put('focus_execution_policy',{'engine':'pytorch','article_body_training':True,'supervised_qa':True})
        quality={'samples':[],'repeated_trigram_fraction':0}
        self.quality=patch('evolve.full_pass_learning._quality',return_value=quality)
        self.quality.start();self.addCleanup(self.quality.stop)

    def test_complete_pass_trains_every_training_target_and_keeps_rejected_candidate(self):
        from evolve.full_pass_learning import train_full_pass,CHECKPOINT_KEY
        from evolve.pytorch_learning import TorchGRU
        with patch('evolve.full_pass_learning.repetition_guard',return_value=False):
            result=train_full_pass(self.store,context=64,checkpoint_every=20)
        state=self.store.get(CHECKPOINT_KEY);parts=read_owned_corpus(self.dataset)
        qa=read_supervised_corpus(state['qa_dataset'])
        self.assertEqual(result['status'],'completed');self.assertFalse(result['accepted'])
        self.assertEqual(result['steps_done'],result['steps_target']);self.assertEqual(result['steps_remaining'],0)
        self.assertEqual(result['language_bytes_done'],sum(len(data)-1 for data in parts['train']))
        self.assertEqual(result['qa_steps_done'],len(qa['train']))
        self.assertEqual(state['cursor']['document_offsets'],[len(data)-1 for data in parts['train']])
        self.assertEqual(state['cursor']['epoch'],0)
        self.assertFalse(result['result']['selected_on_test'])
        self.assertEqual(self.store.get('focus_train_latest_saudi-history-ar')['sha256'],self.original_sha)
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).hexdigest(),self.original_sha)
        self.assertNotEqual(state['sha256'],self.original_sha)
        net,_=TorchGRU.load(self.store.root,state['checkpoint'],state['sha256'])
        self.assertEqual(net.step,state['steps_target'])
        # Only the original and most recent resumable candidate remain on disk.
        self.assertEqual(len(list(self.original.parent.glob('*.pt'))),2)

    def test_stop_and_resume_preserve_optimizer_and_document_offsets(self):
        from evolve.full_pass_learning import train_full_pass,CHECKPOINT_KEY
        from evolve.pytorch_learning import TorchGRU
        stop={'requested':False}
        def progress(summary,*_):
            if summary['steps_done']==7:stop['requested']=True
        result=train_full_pass(self.store,context=64,checkpoint_every=3,progress=progress,
                               should_stop=lambda:stop['requested'])
        self.assertEqual(result['status'],'interrupted');self.assertEqual(result['steps_done'],7)
        pending=self.store.get(CHECKPOINT_KEY)
        net,_=TorchGRU.load(self.store.root,pending['checkpoint'],pending['sha256'])
        self.assertEqual(net.step,7)
        first=[]
        def resumed(summary,*_):
            if not first:first.append(summary['steps_done'])
        result=train_full_pass(self.store,context=64,checkpoint_every=20,progress=resumed)
        self.assertEqual(first,[8]);self.assertEqual(result['status'],'completed')
        final=self.store.get(CHECKPOINT_KEY);net,_=TorchGRU.load(self.store.root,final['checkpoint'],final['sha256'])
        self.assertEqual(net.step,result['steps_target'])

    def test_resumption_rechecks_snapshot_approval_before_using_saved_weights(self):
        from evolve.full_pass_learning import train_full_pass
        train_full_pass(self.store,context=64,should_stop=lambda:True)
        self.store.db.execute('DELETE FROM corpus_training_approvals');self.store.db.commit()
        with self.assertRaisesRegex(ValueError,'approval is no longer valid'):
            train_full_pass(self.store,context=64)


if __name__=='__main__':unittest.main()
