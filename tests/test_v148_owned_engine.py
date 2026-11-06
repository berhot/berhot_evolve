"""First-party neural math and preprocessing; runnable with python3 -S."""
from array import array
import hashlib
import io
import json
import math
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from evolve.owned_engine import OwnedGRU,FORMAT,_OwnCheckpointReader
from evolve.owned_corpus import article_body


class OwnedEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.net=OwnedGRU(self.tmp.name,2,3,seed=17);self.addCleanup(self.net.close)

    def test_forward_matches_independent_scalar_gru_equations(self):
        e,h=2,3;w=self.net.values();token=21;previous=[.12,-.22,.08]
        wi=256*e;wh=wi+3*h*e;bi=wh+3*h*h;bh=bi+3*h;head=bh+3*h;bias=head+256*h
        def dot(a,b):return sum(x*y for x,y in zip(a,b))
        def sig(v):return 1/(1+math.exp(-v))
        expected=[];x=w[token*e:(token+1)*e]
        for j in range(h):
            r=sig(w[bi+j]+w[bh+j]+dot(w[wi+j*e:wi+(j+1)*e],x)+dot(w[wh+j*h:wh+(j+1)*h],previous))
            z=sig(w[bi+h+j]+w[bh+h+j]+dot(w[wi+(h+j)*e:wi+(h+j+1)*e],x)+dot(w[wh+(h+j)*h:wh+(h+j+1)*h],previous))
            n=math.tanh(w[bi+2*h+j]+dot(w[wi+(2*h+j)*e:wi+(2*h+j+1)*e],x)+r*(w[bh+2*h+j]+dot(w[wh+(2*h+j)*h:wh+(2*h+j+1)*h],previous)))
            expected.append((1-z)*n+z*previous[j])
        logits,actual=self.net.logits(token,previous)
        for a,b in zip(actual,expected):self.assertAlmostEqual(a,b,places=6)
        for k in range(256):self.assertAlmostEqual(logits[k],w[bias+k]+dot(w[head+k*h:head+(k+1)*h],expected),places=6)

    def test_bptt_gradients_match_finite_differences_across_all_parameter_groups(self):
        data=b'abcab';self.net.loss(data,True);analytic=self.net.values(gradients=True);original=self.net.values()
        # Embedding, each GRU input/recurrent gate, both bias types and output.
        indices=[97*2,98*2+1,512,518,524,530,539,548,557,560,563,566,569,572,575,575+97*3,575+256*3+98]
        for index in indices:
            if index>=len(original):continue
            plus=array('f',original);minus=array('f',original);plus[index]+=.002;minus[index]-=.002
            self.net.set_values(plus);hi=self.net.loss(data)
            self.net.set_values(minus);lo=self.net.loss(data)
            numeric=(hi-lo)/(plus[index]-minus[index])
            self.assertAlmostEqual(analytic[index],numeric,delta=.0003,msg=str(index))
        self.net.set_values(original)

    def test_real_optimizer_reduces_loss_and_resume_preserves_next_update(self):
        data=b'abababababababababababababababab';before=self.net.loss(data)
        for _ in range(45):self.net.train(data,.02)
        self.assertLess(self.net.loss(data),before*.45)
        path=Path(self.tmp.name)/'owned.bin';digest=self.net.save(path,context_bytes=64)
        loaded,meta=OwnedGRU.load(self.tmp.name,path,digest);self.addCleanup(loaded.close)
        self.assertEqual(meta['format'],FORMAT);self.assertEqual(loaded.step,45)
        self.net.train(data,.02);loaded.train(data,.02)
        self.assertEqual(self.net.values(True),loaded.values(True))
        with path.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'hash'):OwnedGRU.load(self.tmp.name,path,digest)

    def test_article_body_excludes_navigation_toc_and_related_text(self):
        first='تاريخ الدولة السعودية الأولى وتأسيس الدرعية. '*20
        html='<nav>روابط التنقل</nav><main><div class="article-body"><div id="toc"><h2>جدول المحتويات</h2></div><h1>تاريخ السعودية</h1><p>'+first+'</p><aside>مواد أخرى</aside><div class="related-articles">توصيات المقالات</div></div></main><footer>تذييل الموقع</footer>'
        text=article_body(html)
        self.assertIn(first.strip(),text)
        for noise in ('جدول المحتويات','روابط التنقل','مواد أخرى','توصيات المقالات','تذييل الموقع'):self.assertNotIn(noise,text)

    def test_legacy_reader_refuses_arbitrary_pickle_globals(self):
        with self.assertRaisesRegex(ValueError,'Unsupported checkpoint object'):
            _OwnCheckpointReader(io.BytesIO(b'cos\nsystem\n.'),None,'').load()

    def test_owned_chat_status_and_generation_with_no_external_packages(self):
        from evolve.store import Store
        from evolve.focus_inference import FocusGenerator,brain_status
        store=Store(self.tmp.name);self.addCleanup(store.close)
        root=store.root/'models'/'focus'/'saudi-history-ar';root.mkdir(parents=True)
        net=OwnedGRU(store.root,16,16);self.addCleanup(net.close)
        path=root/'owned.bin';digest=net.save(path,context_bytes=256)
        store.put('focus_train_latest_saudi-history-ar',{'format':FORMAT,'accepted':True,
                  'checkpoint':str(path),'sha256':digest,'model_config':net.config,'context_bytes':256,
                  'training_engine':'first_party_c','steps_total':2})
        store.put('focus_execution_policy',{'engine':'first_party_c'})
        status=brain_status(store);self.assertEqual(status['status'],'ready')
        self.assertFalse(status['model']['external_ml_runtime_required'])
        result=FocusGenerator().generate(store,'نص عربي',max_new_bytes=32,seed=19)
        self.assertEqual(result['training_engine'],'first_party_c')
        self.assertEqual(len(result['answer'].encode()),32)
        self.assertFalse(result['training_updated'])
        # A fresh process forbids even an attempted third-party import while
        # exercising the real controller status, neural chat and policy guards.
        script='''
import builtins,sys
original=builtins.__import__
def strict(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','playwright','transformers','requests'}:
        raise AssertionError('Forbidden import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=strict
from evolve.settings import load_config
from evolve.v118_studio import LearningController
config=load_config();config['storage_root']=sys.argv[1]
controller=LearningController(config)
try:
    assert controller.status()['status']=='online'
    assert controller.brain()['model']['training_engine']=='first_party_c'
    assert controller.talk('نص عربي','focus',max_new_bytes=16)['model_generated']
    assert controller.submit('language')['status']=='unavailable'
    try:controller.talk('نص عربي','auto')
    except ValueError:pass
    else:raise AssertionError('Legacy chat was not blocked')
    print('NO_EXTERNAL_IMPORTS')
finally:controller.shutdown()
'''
        result=subprocess.run([sys.executable,'-S','-c',script,self.tmp.name],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('NO_EXTERNAL_IMPORTS',result.stdout)


if __name__=='__main__':unittest.main()
