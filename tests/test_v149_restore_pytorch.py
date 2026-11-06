"""Return local weights/Adam moments to PyTorch without resetting learning."""
from array import array
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch=None

from evolve.store import Store


@unittest.skipIf(torch is None,'PyTorch required')
class RestorePyTorchTests(unittest.TestCase):
    def test_restore_preserves_weights_optimizer_and_checkpoint_resume(self):
        from evolve.focus_training import _network
        from evolve.owned_engine import MAGIC,KEYS
        from evolve.pytorch_learning import restore_pytorch,TorchGRU
        with tempfile.TemporaryDirectory() as folder:
            store=Store(folder)
            try:
                config={'embedding_dim':16,'hidden_size':16};_,original=_network(config)
                weights=array('f')
                for key in KEYS:weights.extend(original.state_dict()[key].flatten().tolist())
                count=len(weights);state=array('f',weights)
                state.extend(array('f',[.01])*count);state.extend(array('f',[.1])*count)
                header=json.dumps({'format':'berhot_owned_byte_gru_v1','model_config':config,
                                   'optimizer_step':7,'context_bytes':256}).encode()
                root=Path(folder)/'models'/'focus'/'saudi-history-ar';root.mkdir(parents=True)
                path=root/'preserved.bin';path.write_bytes(MAGIC+struct.pack('<I',len(header))+header+state.tobytes())
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                store.put('focus_train_latest_saudi-history-ar',{'format':'berhot_owned_byte_gru_v1',
                    'accepted':True,'checkpoint':str(path),'sha256':digest,'model_config':config,'steps_total':50})
                store.put('focus_execution_policy',{'engine':'first_party_c'})
                record=restore_pytorch(store)
                self.assertEqual(store.get('focus_execution_policy')['engine'],'pytorch')
                self.assertEqual(record['steps_total'],50)
                self.assertFalse(record['pretrained_weights_used'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
                net,saved=TorchGRU.load(folder,record['checkpoint'],record['sha256'])
                self.assertEqual(net.step,7)
                for key in KEYS:self.assertTrue(torch.equal(saved['weights'][key],original.state_dict()[key]))
                for parameter in net.model.parameters():
                    moment=net.optimizer.state[parameter]
                    self.assertEqual(int(moment['step'].item()),7)
                    self.assertTrue(torch.all(moment['exp_avg']==.01))
                    self.assertTrue(torch.all(moment['exp_avg_sq']==.1))
                data=b'abababababababab';net.train(data)
                self.assertEqual(net.step,8)
                resumed_path=root/'resumed.pt';sha=net.save(resumed_path)
                resumed,_=TorchGRU.load(folder,resumed_path,sha)
                net.train(data);resumed.train(data)
                for a,b in zip(net.model.parameters(),resumed.model.parameters()):self.assertTrue(torch.equal(a,b))
            finally:store.close()


if __name__=='__main__':unittest.main()
