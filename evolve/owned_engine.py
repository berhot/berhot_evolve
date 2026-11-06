"""First-party local neural engine. Python standard library + our C source.

No torch/numpy imports, model downloads, API clients, BLAS or ML frameworks.
The platform compiler/libm are system build/runtime prerequisites.
"""
from array import array
import ctypes as C
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

FORMAT = "berhot_owned_byte_gru_v1"
MAGIC = b"BEVGRU1\n"
KEYS = ("embedding.weight", "gru.weight_ih_l0", "gru.weight_hh_l0",
        "gru.bias_ih_l0", "gru.bias_hh_l0", "head.weight", "head.bias")


def native_library(root):
    source = Path(__file__).with_name("owned_gru.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    folder = Path(root) / "native"
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / ("owned-gru-" + platform.machine() + "-" + digest + (".dylib" if sys.platform == "darwin" else ".so"))
    if not library.exists():
        compiler = shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("A system C compiler is required to build the first-party neural kernel")
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            output = Path(temporary) / library.name
            command = [compiler, "-O3", "-std=c11", "-fno-math-errno",
                       "-dynamiclib" if sys.platform == "darwin" else "-shared", "-fPIC",
                       str(source), "-o", str(output), "-lm"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            if result.returncode:
                raise RuntimeError("Owned C kernel compilation failed: " + result.stderr[-1200:])
            output.replace(library)
    api = C.CDLL(str(library.resolve()))
    fp = C.POINTER(C.c_float)
    signatures = {
        "owned_create": ([C.c_int,C.c_int,C.c_uint64],C.c_void_p),
        "owned_free": ([C.c_void_p],None), "owned_size": ([C.c_void_p],C.c_int),
        "owned_get": ([C.c_void_p,fp,C.c_int],None),
        "owned_set": ([C.c_void_p,fp,C.c_int,C.c_uint64],C.c_int),
        "owned_gradients": ([C.c_void_p,fp],None),
        "owned_next": ([C.c_void_p,C.c_int,fp,fp,fp],C.c_int),
        "owned_loss": ([C.c_void_p,C.c_void_p,C.c_int,C.c_int],C.c_double),
        "owned_update": ([C.c_void_p,C.c_float,C.c_float,C.c_float],C.c_int)}
    for name,(args,restype) in signatures.items():
        function=getattr(api,name);function.argtypes=args;function.restype=restype
    return api


class OwnedGRU:
    def __init__(self, root, embedding_dim, hidden_size, seed=42):
        if any(type(v) is not int or not 1 <= v <= 1024 for v in (embedding_dim,hidden_size)):
            raise ValueError("Owned GRU dimensions must be integers 1..1024")
        self.config={"embedding_dim":embedding_dim,"hidden_size":hidden_size}
        self.api=native_library(root);self.ptr=self.api.owned_create(embedding_dim,hidden_size,seed)
        if not self.ptr:raise MemoryError("Cannot allocate owned neural model")
        self.count=self.api.owned_size(self.ptr);self.step=0

    def close(self):
        if self.ptr:self.api.owned_free(self.ptr);self.ptr=None

    def __del__(self):
        if getattr(self,"ptr",None):self.close()

    def values(self, optimizer=False, gradients=False):
        result=array('f',[0])*(self.count*(3 if optimizer else 1))
        pointer=(C.c_float*len(result)).from_buffer(result)
        if gradients:self.api.owned_gradients(self.ptr,pointer)
        else:self.api.owned_get(self.ptr,pointer,int(optimizer))
        return result

    def set_values(self, values, optimizer_step=0):
        if values.typecode!='f' or len(values) not in (self.count,3*self.count):
            raise ValueError("Invalid owned weight/state array")
        code=self.api.owned_set(self.ptr,(C.c_float*len(values)).from_buffer(values),len(values),optimizer_step)
        if code:raise ValueError("Invalid/non-finite neural weights")
        self.step=optimizer_step

    def logits(self, token, hidden=None):
        h=self.config['hidden_size'];previous=(C.c_float*h)(*(hidden or [0.0]*h))
        current=(C.c_float*h)();scores=(C.c_float*256)()
        if self.api.owned_next(self.ptr,token,previous,current,scores):raise ValueError("Invalid byte token")
        return list(scores),list(current)

    def loss(self, data, backward=False):
        if not isinstance(data,bytes) or not 2 <= len(data) <= 4097:
            raise ValueError("A document window must contain 2..4097 bytes")
        buffer=C.create_string_buffer(data)
        value=self.api.owned_loss(self.ptr,buffer,len(data)-1,int(backward))
        if not math.isfinite(value):raise ValueError("Non-finite owned neural loss")
        return value

    def train(self, data, rate=.0003):
        if not math.isfinite(rate) or not 0 < rate <= .1:raise ValueError("Invalid learning rate")
        loss=self.loss(data,True)
        if self.api.owned_update(self.ptr,rate,.01,1.0):raise ValueError("Non-finite owned optimizer update")
        self.step+=1;return loss

    def save(self, path, context_bytes=256, **metadata):
        header={"format":FORMAT,"model_config":self.config,"optimizer_step":self.step,
                "context_bytes":context_bytes,**metadata}
        encoded=json.dumps(header,ensure_ascii=False,sort_keys=True).encode()
        state=self.values(True)
        if sys.byteorder!='little':state.byteswap()
        with Path(path).open('xb') as stream:
            stream.write(MAGIC+struct.pack('<I',len(encoded))+encoded);state.tofile(stream)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, root, path, expected_sha=None):
        blob=Path(path).read_bytes()
        if len(blob)>160_000_000 or (expected_sha and hashlib.sha256(blob).hexdigest()!=expected_sha):
            raise ValueError("Owned checkpoint size/hash mismatch")
        if not blob.startswith(MAGIC) or len(blob)<12:raise ValueError("Invalid owned checkpoint")
        length=struct.unpack('<I',blob[8:12])[0]
        if not 1 <= length <= 65536:raise ValueError("Invalid owned checkpoint header")
        header=json.loads(blob[12:12+length])
        if header.get('format')!=FORMAT:raise ValueError("Unknown owned checkpoint format")
        net=cls(root,**header['model_config']);state=array('f');state.frombytes(blob[12+length:])
        if sys.byteorder!='little':state.byteswap()
        if len(state)!=3*net.count:raise ValueError("Owned optimizer state is incomplete")
        net.set_values(state,header['optimizer_step']);return net,header


class _FloatStorage:pass


class _Tensor:
    def __init__(self, storage, offset, shape, stride):
        self.storage,self.offset,self.shape,self.stride=storage,offset,tuple(shape),tuple(stride)


def _tensor(storage,offset,shape,stride,*_):return _Tensor(storage,offset,shape,stride)


class _OwnCheckpointReader(pickle.Unpickler):
    """Restricted decoding of our own old tensor archives without importing torch."""
    def __init__(self, stream, archive, prefix):
        super().__init__(stream);self.archive,self.prefix=archive,prefix

    def find_class(self,module,name):
        if (module,name)==('torch._utils','_rebuild_tensor_v2'):return _tensor
        if (module,name)==('torch','FloatStorage'):return _FloatStorage
        if (module,name)==('collections','OrderedDict'):
            from collections import OrderedDict
            return OrderedDict
        raise ValueError("Unsupported checkpoint object: "+module+'.'+name)

    def persistent_load(self, identity):
        if (not isinstance(identity,tuple) or len(identity)!=5 or identity[0]!='storage' or
                identity[1] is not _FloatStorage or identity[3]!='cpu' or
                not str(identity[2]).isdigit() or not 0<int(identity[4])<=20_000_000):
            raise ValueError("Unsupported tensor storage")
        raw=self.archive.read(self.prefix+'data/'+str(identity[2]))
        if len(raw)!=int(identity[4])*4:raise ValueError("Tensor storage length mismatch")
        values=array('f');values.frombytes(raw)
        if sys.byteorder!='little':values.byteswap()
        return values


def import_own_legacy_weights(root, record):
    """Migrate user-trained weights, not an external pretrained model."""
    path=Path(record['checkpoint']).resolve();allowed=(Path(root)/'models'/'focus'/'saudi-history-ar').resolve()
    if not path.is_relative_to(allowed) or path.stat().st_size>64_000_000:
        raise ValueError("Legacy checkpoint outside the local model archive")
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError("Legacy checkpoint hash mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        entries=archive.infolist()
        if sum(e.file_size for e in entries)>80_000_000:raise ValueError("Oversized tensor archive")
        names=[e.filename for e in entries if e.filename.endswith('/data.pkl')]
        if len(names)!=1 or archive.getinfo(names[0]).file_size>2_000_000:raise ValueError("Invalid tensor archive metadata")
        prefix=names[0][:-len('data.pkl')]
        if archive.read(prefix+'byteorder')!=b'little':raise ValueError("Unsupported old checkpoint byte order")
        decoded=_OwnCheckpointReader(io.BytesIO(archive.read(names[0])),archive,prefix).load()
    config=record['model_config'];e,h=config['embedding_dim'],config['hidden_size']
    shapes=((256,e),(3*h,e),(3*h,h),(3*h,),(3*h,),(256,h),(256,))
    values=array('f')
    for name,shape in zip(KEYS,shapes):
        tensor=decoded['weights'][name]
        if not isinstance(tensor,_Tensor) or tensor.shape!=shape:raise ValueError("Old weight shape mismatch")
        stride=[];width=1
        for size in reversed(shape):stride.insert(0,width);width*=size
        if tensor.stride!=tuple(stride) or tensor.offset<0 or tensor.offset+width>len(tensor.storage):
            raise ValueError("Noncontiguous legacy weights are unsupported")
        values.extend(tensor.storage[tensor.offset:tensor.offset+width])
    net=OwnedGRU(root,**config);net.set_values(values);return net
