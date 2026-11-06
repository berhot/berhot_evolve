"""Autonomous discovery, source supervision and real local optimizer updates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evolve.store import Store
from evolve.catalog_discovery import catalog_search
from evolve.owned_corpus import prepare_owned_corpus,read_owned_corpus
from evolve.supervised_corpus import (prepare_supervised_corpus,read_supervised_corpus,
    encode_example,question_prompt,plan_curiosity)
from evolve.training_schedule import adapt_dataset_cursor

try:
    import torch
except ImportError:
    torch=None


class AutonomousPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.store=Store(self.tmp.name);self.addCleanup(self.store.close)

    def body_dataset(self):
        from evolve.open_corpus import archive_page
        from evolve.web_discovery import extract
        from evolve.training_approval import approve_current_corpus
        approve_current_corpus(self.store,'Use my collected source snapshots',include_future=True)
        assignments={}
        for i,part in enumerate(('train','train','val','test')):
            title='تاريخ الدولة السعودية في المنطقة '+str(i)
            body=('تأسست الدولة السعودية في الدرعية وتتناول المصادر تاريخ المملكة العربية السعودية وأحداث الحكم. '+str(i)+' ')*12
            html='<html><title>'+title+'</title><nav>جدول المحتويات</nav><article><h1>'+title+'</h1><p>'+body+'</p></article></html>'
            content,title,*_=extract(html)
            url='https://history.example.org/article-'+str(i)
            archived=archive_page(self.store,url,html,content,title=title)
            self.store.record_focus_archive('saudi-history-ar',url,archived['clean_sha256'])
            assignments[archived['clean_sha256']]=part
        self.store.put('focus_article_split_saudi-history-ar',assignments)
        return prepare_owned_corpus(self.store)

    def test_catalog_is_discovery_only_and_handles_malformed_responses(self):
        urls=[]
        payload={'query':{'pages':[{'title':'تاريخ الدولة السعودية','fullurl':'https://ar.wikipedia.org/wiki/History',
            'extlinks':[{'url':'https://publisher.example.org/history'},{'url':'http://127.0.0.1/private'},
                        {'url':'https://publisher.example.org/file.pdf'},{'url':'javascript:alert(1)'}]}]}}
        def reader(url):urls.append(url);return json.dumps(payload).encode()
        result=catalog_search('تاريخ السعودية -site:saudipedia.com',reader=reader)
        self.assertEqual(result['status'],'search_completed')
        self.assertTrue(urls[0].startswith('https://ar.wikipedia.org/w/api.php?'))
        self.assertNotIn('saudipedia',urls[0]);self.assertEqual(len(result['results']),2)
        self.assertTrue(all(not item['snippet'] for item in result['results']))
        self.assertFalse(result['catalog_text_used_as_training_data'])
        for malformed in (b'[]',b'{"query":null}',b'not json'):
            self.assertEqual(catalog_search('history',reader=lambda _:malformed)['status'],'catalog_search_failed')

    def test_autonomous_queue_prefers_arabic_but_keeps_exploration(self):
        from evolve.focus_discovery import arabic_candidate_priority
        from evolve.focus_overnight import init_state,_run_page
        from evolve.focus_data import get_profile
        from evolve.settings import load_config
        self.assertEqual(arabic_candidate_priority('https://ar.wikipedia.org/wiki/History'),0)
        self.assertEqual(arabic_candidate_priority('https://publisher.example.org/ar/history'),0)
        self.assertEqual(arabic_candidate_priority('https://publisher.example.org/auth/123'),2)
        self.assertEqual(arabic_candidate_priority('https://publisher.example.org/fr/history'),2)
        self.store.put('focus_execution_policy',{'autonomous_pipeline':True})
        config=load_config();config.update(storage_root=self.tmp.name,crawl_delay_seconds=0)
        arabic='https://ar.wikipedia.org/wiki/History'
        unknown='https://independent.example.org/123'
        html='<title>تاريخ السعودية</title><article>'+('تاريخ الدولة السعودية والدرعية. '*80)+'</article>'
        for attempted,expected in ((0,arabic),(4,unknown)):
            state=init_state(self.store,'saudi-history-ar')
            state.update(until_stopped=True,no_fixed_quotas=True,pages_attempted=attempted,
                         minimum_host_gap_seconds=0)
            state['_publisher_attempts']={'wikipedia.org':100}
            state['_queue']=[{'url':unknown,'depth':0},{'url':arabic,'depth':0}]
            result=_run_page(self.store,config,state,get_profile('saudi-history-ar'),fetcher=lambda _:html)
            self.assertEqual(result['url'],expected)

    def test_article_query_ids_survive_discovery_and_remain_distinct(self):
        from evolve.web_discovery import canonical_url
        from evolve.focus_discovery import _candidate_link
        from evolve.focus_overnight import init_state,_enqueue
        first='https://publisher.example.org/viewfullstory.php?id=123'
        second='https://publisher.example.org/viewfullstory.php?id=456'
        self.assertEqual(canonical_url(first+'&utm_source=tracking#top'),first)
        self.assertEqual(canonical_url(first+'&q=private-form-value&token=private-token'),first)
        self.assertEqual(_candidate_link(first),first)
        state=init_state(self.store,'saudi-history-ar');state['_queue']=[];state['_seen']=set()
        self.assertTrue(_enqueue(state,first));self.assertTrue(_enqueue(state,second))
        self.assertFalse(_enqueue(state,first+'&utm_source=another'))

    def test_fetch_preserves_query_and_respects_query_robots_exclusion(self):
        import io
        from email.message import Message
        from evolve.web_discovery import fetch_page,WebSourceError
        class Response(io.BytesIO):
            def __init__(self,data):
                super().__init__(data);self.headers=Message();self.headers['Content-Type']='text/html; charset=utf-8'
        class Opener:
            def __init__(self,blocked):self.blocked=blocked;self.urls=[]
            def open(self,request,**_):
                self.urls.append(request.full_url)
                return Response((b'User-agent: *\nDisallow: /*?\n' if self.blocked else b'User-agent: *\nAllow: /\n')
                                if request.full_url.endswith('/robots.txt') else b'<article>Saved article</article>')
        for blocked in (False,True):
            opener=Opener(blocked)
            with patch('evolve.web_discovery.check_public_dns'),patch.dict('evolve.web_discovery._ROBOTS_CACHE',{},clear=True):
                url='https://publisher.example.org/article.php?id=123'
                if blocked:
                    with self.assertRaisesRegex(WebSourceError,'disallows'):fetch_page(url,{},opener=opener)
                    self.assertEqual(len(opener.urls),1)
                else:
                    self.assertIn('Saved article',fetch_page(url,{},opener=opener))
                    self.assertEqual(opener.urls[-1],url)

    def test_qa_answers_have_exact_evidence_and_keep_article_splits(self):
        body=self.body_dataset();descriptor=prepare_supervised_corpus(self.store,body)
        parts=read_supervised_corpus(descriptor);source=read_owned_corpus(body)
        ids=[]
        for part,rows in parts.items():
            self.assertGreater(len(rows),0)
            for row in rows:
                ids.append(row['id']);self.assertEqual(row['split'],part)
                evidence=next(data.decode() for data in source[part] if hashlib.sha256(data).hexdigest()==row['source_body_sha256'])
                self.assertEqual(evidence[row['answer_character_start']:][:len(row['answer'])],row['answer'])
                prompt,answer=encode_example(row)
                self.assertEqual(prompt.decode(),question_prompt(row['question']))
                self.assertLessEqual(len(prompt+answer),769)
                self.assertFalse(row['independent_fact_verification'])
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(descriptor,prepare_supervised_corpus(self.store,body))
        (Path(descriptor['directory'])/'train.txt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):read_supervised_corpus(descriptor)

    def test_curiosity_uses_only_supplied_training_questions(self):
        descriptor=prepare_supervised_corpus(self.store,self.body_dataset())
        parts=read_supervised_corpus(descriptor)
        class LocalModel:
            def qa_loss(self,prompt,answer):return float(len(answer))
        tasks=plan_curiosity(self.store,LocalModel(),parts['train'])
        self.assertTrue(tasks)
        self.assertTrue({task['id'] for task in tasks}<={row['id'] for row in parts['train']})
        self.assertFalse({task['id'] for task in tasks}&{row['id'] for row in parts['val']+parts['test']})
        self.assertTrue(all(task['status']=='pending' for task in tasks))

    def test_dataset_growth_preserves_existing_document_coverage(self):
        old=[b'abcde',b'uvwxyz'];cursor={'dataset_revision_sha256':'old','document_offsets':[2,4],
            'attempted_training_bytes':99,'windows_completed':12,'schedule_version':'interleaved_articles_v1'}
        result=adapt_dataset_cursor([old[1],b'new',old[0]],cursor,'new',old)
        self.assertEqual(result['document_offsets'],[4,0,2])
        self.assertEqual(result['attempted_training_bytes'],99)
        self.assertEqual(cursor['document_offsets'],[2,4])

    def test_article_body_batch_is_approved_through_original_snapshot(self):
        from evolve.plain_corpus import begin_extraction_run,capture_extraction,finalize_extraction_run
        descriptor=self.body_dataset()
        record=json.loads((Path(descriptor['directory'])/'documents.jsonl').read_text().splitlines()[0])
        body=next(data.decode() for docs in read_owned_corpus(descriptor).values() for data in docs
                  if hashlib.sha256(data).hexdigest()==record['clean_sha256'])
        folder=begin_extraction_run(self.store,'derived-body','saudi-history-ar')
        capture_extraction(self.store,'derived-body',1,record['url'],record['title'],
            '<article>'+body+'</article>','جدول المحتويات '+body,clean_text=body,
            approval_parent_sha256=record['approval_parent_sha256'])
        result=finalize_extraction_run(self.store,'derived-body')
        self.assertEqual(result['training_eligible_unique_texts'],1)
        self.assertIn('جدول المحتويات',(folder/'raw.txt').read_text())
        self.assertNotIn('جدول المحتويات',(folder/'clean.txt').read_text())
        self.assertGreater((folder/'corpus.txt').stat().st_size,0)

    def test_qa_inference_requires_accepted_supervision_and_uses_own_generator(self):
        from evolve.settings import load_config
        from evolve.v118_studio import LearningController
        config=load_config();config['storage_root']=self.tmp.name
        controller=LearningController(config);self.addCleanup(controller.shutdown)
        self.assertEqual(controller.talk('سؤال عن السعودية','focus_qa')['status'],'qa_not_trained_yet')
        self.store.put('focus_train_latest_saudi-history-ar',{'accepted':True,'instruction_tuned':True})
        with patch.object(controller.focus_generator,'generate',return_value={
            'status':'generated','answer':'إجابة النموذج\n<|end|>\nزائد','model_generated':True}) as generate:
            result=controller.talk('سؤال عن السعودية','focus_qa',seed=8)
        self.assertEqual(generate.call_args.args[1],question_prompt('سؤال عن السعودية'))
        self.assertEqual(result['answer'],'إجابة النموذج');self.assertFalse(result['answer_verified'])
        self.assertTrue(result['model_generated']);self.assertIn('<|end|>',result['raw_model_continuation'])

    @unittest.skipIf(torch is None,'PyTorch required')
    def test_answer_only_loss_and_real_optimizer_update(self):
        from evolve.pytorch_learning import TorchGRU
        net=TorchGRU(self.tmp.name,embedding_dim=16,hidden_size=16)
        prompt=b'question: ';answer=b'abababababab'
        with torch.inference_mode():
            ids=torch.tensor([list(prompt+answer)],dtype=torch.long)
            output=net.model(ids[:,:-1])[0,len(prompt)-1:]
            expected=torch.nn.functional.cross_entropy(output,torch.tensor(list(answer))).item()
        self.assertAlmostEqual(net.qa_loss(prompt,answer),expected,places=5)
        for _ in range(30):net.train_qa(prompt,answer,.02)
        self.assertEqual(net.step,30);self.assertLess(net.qa_loss(prompt,answer),expected*.5)

    @unittest.skipIf(torch is None,'PyTorch required')
    def test_mixed_training_records_qa_steps_without_training_holdouts(self):
        from evolve.pytorch_learning import TorchGRU
        from evolve.owned_learning import train_owned
        descriptor=self.body_dataset()
        root=self.store.root/'models'/'focus'/'saudi-history-ar';root.mkdir(parents=True)
        net=TorchGRU(self.tmp.name,embedding_dim=16,hidden_size=16);path=root/'original.pt';sha=net.save(path)
        self.store.put('focus_train_latest_saudi-history-ar',{'format':net.format,'accepted':True,
            'checkpoint':str(path),'sha256':sha,'steps_total':0,'model_config':net.config})
        self.store.put('focus_execution_policy',{'engine':'pytorch','article_body_training':True,'supervised_qa':True})
        quality={'samples':[],'repeated_trigram_fraction':0}
        with patch('evolve.owned_learning._quality',return_value=quality):
            result=train_owned(self.store,steps=4,context=64)
        self.assertEqual(result['steps'],4);self.assertEqual(result['supervised_qa_steps'],1)
        self.assertGreater(result['supervised_answer_bytes_seen'],0)
        self.assertIsNotNone(result['qa_validation_loss_after']);self.assertFalse(result['selected_on_test'])
        qa=read_supervised_corpus(result['qa_dataset'])
        trained={row['qa_id'] for row in self.store.db.execute('SELECT * FROM supervised_qa_progress')}
        self.assertTrue(trained<={row['id'] for row in qa['train']})
        self.assertFalse(trained&{row['id'] for row in qa['val']+qa['test']})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),sha)


if __name__=='__main__':unittest.main()
